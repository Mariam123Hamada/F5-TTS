import argparse
import glob
import json
import os
import random
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files


def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", t)).strip()


def read_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line[:-1] for line in f]          # same reading logic as f5_tts.model.utils.get_tokenizer


def split_sources(hours_by_src, val_hours, test_hours, seed, max_heldout_frac=0.25, max_unit_factor=1.5):
    """Assign whole recordings to test / val / train. Deterministic for a given seed.
    - recordings are shuffled, then filled into test first, then val, until the target hours are reached
    - a recording longer than max_unit_factor x the target of a split is not used for that split (so one huge
      recording cannot become the whole test set)
    - at most max_heldout_frac of all hours is ever held out
    Returns (train_srcs, val_srcs, test_srcs)."""
    srcs = sorted(hours_by_src)
    random.Random(seed).shuffle(srcs)
    total = sum(hours_by_src.values())
    test, val, h_test, h_val = [], [], 0.0, 0.0
    for s in srcs:
        h = hours_by_src[s]
        if (h_test + h_val + h) > max_heldout_frac * total:
            continue
        if test_hours > 0 and h_test < test_hours and h <= max_unit_factor * test_hours:
            test.append(s); h_test += h
        elif val_hours > 0 and h_val < val_hours and h <= max_unit_factor * val_hours:
            val.append(s); h_val += h
    held = set(test) | set(val)
    train = [s for s in srcs if s not in held]
    return train, val, test


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="saudi_lora", help="dataset name -> data/<name>_custom (+ _val_custom, _test_custom)")
    ap.add_argument("--repo", required=True, help="HF dataset repo with 'datasets N' folders")
    ap.add_argument("--data_dir", default="saudi_data", help="local folder for the downloaded wavs + meta")
    ap.add_argument("--download", action="store_true", help="download wavs + meta from the HF repo first (resumable)")
    ap.add_argument("--folders", default="", help="comma separated folder names to use, e.g. 'datasets 36,datasets 37' (default: all)")
    ap.add_argument("--min_dur", type=float, default=1.5)
    ap.add_argument("--max_dur", type=float, default=25.0)
    ap.add_argument("--min_sim", type=float, default=0.40, help="voice match with the target speaker (stored in meta as 'sim')")
    ap.add_argument("--min_logprob", type=float, default=-0.60, help="Whisper confidence (stored in meta as 'logprob')")
    ap.add_argument("--min_token_count", type=int, default=5, help="a character not in the base vocab is added only if it appears at least this often in the TRAIN part")
    ap.add_argument("--max_unk_frac", type=float, default=0.03, help="drop a sample if more than this fraction of its tokens is dropped")
    ap.add_argument("--val_hours", type=float, default=1.0, help="hours of whole recordings for the validation set (0 = none)")
    ap.add_argument("--test_hours", type=float, default=2.0, help="hours of whole recordings for the test set (0 = none)")
    ap.add_argument("--split_seed", type=int, default=13)
    args = ap.parse_args()

    from datasets import Dataset as HFDataset
    from datasets.arrow_writer import ArrowWriter
    import soundfile as sf

    from f5_tts.model.utils import convert_char_to_pinyin

    token = os.environ.get("HF_TOKEN")
    if args.download:
        from huggingface_hub import snapshot_download

        print(f"downloading wavs + meta from {args.repo} (resumable) ...")
        snapshot_download(repo_id=args.repo, repo_type="dataset", token=token, local_dir=args.data_dir,
                          allow_patterns=["datasets */wavs/*", "datasets */meta/*.jsonl"], max_workers=8)

    base_vocab_path = str(files("f5_tts").joinpath("../../data/Emilia_ZH_EN_pinyin/vocab.txt"))
    base_vocab = read_vocab(base_vocab_path)
    base_set = set(base_vocab)
    assert base_vocab[0] == " ", "base vocab must start with a space"
    print(f"base vocab: {len(base_vocab)} tokens")

    metas = sorted(glob.glob(os.path.join(args.data_dir, "datasets */meta/*.jsonl")))
    wanted = {f.strip() for f in args.folders.split(",") if f.strip()}
    if wanted:
        metas = [m for m in metas if os.path.basename(os.path.dirname(os.path.dirname(m))) in wanted]
    if not metas:
        raise SystemExit(f"no meta files found under {args.data_dir} (did you pass --download?)")
    print(f"{len(metas)} meta files (= source recordings)")

    # ---------------- 1) filter rows; remember the source recording of every row ----------------
    stats, total_sec, cand = Counter(), 0.0, []
    for mp in metas:
        folder = os.path.dirname(os.path.dirname(mp))
        src = os.path.relpath(mp, args.data_dir)                       # e.g. "datasets 36/meta/file_00.jsonl"
        with open(mp, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                stats["rows"] += 1
                dur = float(r.get("dur", 0))
                total_sec += dur
                wav = os.path.join(folder, "wavs", os.path.basename(r["audio"]))
                text = clean_text(r.get("text", ""))
                if not os.path.exists(wav):                                stats["drop: wav missing"] += 1; continue
                if not (args.min_dur <= dur <= args.max_dur):              stats["drop: duration"] += 1; continue
                if float(r.get("sim", 1.0)) < args.min_sim:                stats["drop: low voice match"] += 1; continue
                if float(r.get("logprob", 0.0)) < args.min_logprob:        stats["drop: low whisper confidence"] += 1; continue
                if not text:                                               stats["drop: empty text"] += 1; continue
                cand.append({"wav": wav, "text": text, "src": src, "dur": dur})
    print("filter report:", dict(stats))
    if not cand:
        raise SystemExit("nothing left after filtering - relax --min_sim / --min_logprob")

    # ---------------- 2) split BY RECORDING ----------------
    hours_by_src = defaultdict(float)
    for c in cand:
        hours_by_src[c["src"]] += c["dur"] / 3600
    do_split = (args.val_hours > 0 or args.test_hours > 0)
    if do_split and len(hours_by_src) < 5:
        print(f"WARNING: only {len(hours_by_src)} recordings - too few to split safely, everything goes to training")
        do_split = False
    if do_split:
        train_srcs, val_srcs, test_srcs = split_sources(hours_by_src, args.val_hours, args.test_hours, args.split_seed)
    else:
        train_srcs, val_srcs, test_srcs = sorted(hours_by_src), [], []
    split_of = {s: "train" for s in train_srcs}
    split_of.update({s: "val" for s in val_srcs})
    split_of.update({s: "test" for s in test_srcs})
    for c in cand:
        c["split"] = split_of[c["src"]]
    print("\nsplit by recording:")
    for name, srcs in (("train", train_srcs), ("val", val_srcs), ("test", test_srcs)):
        n = sum(1 for c in cand if c["split"] == name)
        print(f"  {name:5s}: {len(srcs):3d} recordings | {n:6d} segments | {sum(hours_by_src[s] for s in srcs):6.2f} h")

    # ---------------- 3) tokenize exactly like inference ----------------
    tokens_list = []
    for i in range(0, len(cand), 200):
        tokens_list += convert_char_to_pinyin([c["text"] for c in cand[i:i + 200]], polyphone=True)

    # ---------------- 4) extend the vocab using the TRAIN part only ----------------
    counts_train = Counter(tok for c, toks in zip(cand, tokens_list) if c["split"] == "train"
                           for tok in toks if tok not in base_set)
    new_tokens = [t for t, n in counts_train.most_common() if n >= args.min_token_count and t != "\n"]
    keep_set = base_set | set(new_tokens)
    dropped_tokens = Counter(tok for toks in tokens_list for tok in toks if tok not in keep_set)
    print(f"\nnew tokens added to the vocab (from TRAIN only): {len(new_tokens)} -> {''.join(new_tokens)}")
    if dropped_tokens:
        print("tokens removed from the text (not in the vocab):", dict(dropped_tokens.most_common(15)))

    # ---------------- 5) real durations from the files ----------------
    def dur_of(p):
        try:
            return sf.info(p).duration
        except Exception:
            return 0.0

    with ThreadPoolExecutor(16) as ex:
        durs = list(ex.map(dur_of, [c["wav"] for c in cand]))

    rows = {"train": [], "val": [], "test": []}
    n_unk_drop = Counter()
    for c, toks, d in zip(cand, tokens_list, durs):
        if d <= 0.3:
            continue
        kept = [t for t in toks if t in keep_set]
        if not kept or (len(toks) - len(kept)) / max(len(toks), 1) > args.max_unk_frac:
            n_unk_drop[c["split"]] += 1
            continue
        rows[c["split"]].append({"audio_path": c["wav"], "text": kept, "duration": d, "_text": c["text"], "_src": c["src"]})
    if n_unk_drop:
        print("dropped because of too many unknown tokens:", dict(n_unk_drop))

    # ---------------- 6) write the datasets ----------------
    def write_split(out_dir, split_rows, with_manifest):
        os.makedirs(out_dir, exist_ok=True)
        # explicit finalize(): `with ArrowWriter(...)` does not flush the last partial batch in recent `datasets`
        w = ArrowWriter(path=f"{out_dir}/raw.arrow", writer_batch_size=100)
        for r in split_rows:
            w.write({"audio_path": r["audio_path"], "text": r["text"], "duration": r["duration"]})
        w.finalize()
        n_written = len(HFDataset.from_file(f"{out_dir}/raw.arrow"))
        if n_written != len(split_rows):
            raise RuntimeError(f"{out_dir}/raw.arrow has {n_written} rows, expected {len(split_rows)}")
        with open(f"{out_dir}/duration.json", "w", encoding="utf-8") as f:
            json.dump({"duration": [r["duration"] for r in split_rows]}, f)
        with open(f"{out_dir}/vocab.txt", "w", encoding="utf-8") as f:
            for tok in base_vocab + new_tokens:
                f.write(tok + "\n")
        if with_manifest:                                   # readable transcripts for WER / CER / speaker similarity
            with open(f"{out_dir}/manifest.jsonl", "w", encoding="utf-8") as f:
                for r in split_rows:
                    f.write(json.dumps({"audio_path": r["audio_path"], "text": r["_text"],
                                        "duration": round(r["duration"], 3), "source": r["_src"]}, ensure_ascii=False) + "\n")

    data_root = str(files("f5_tts").joinpath("../../data"))
    outs = {"train": f"{data_root}/{args.name}_custom", "val": f"{data_root}/{args.name}_val_custom",
            "test": f"{data_root}/{args.name}_test_custom"}
    for split in ("val", "test"):                           # remove stale held-out sets from an older run
        if not rows[split] and os.path.isdir(outs[split]):
            shutil.rmtree(outs[split])
    for split in ("train", "val", "test"):
        if rows[split]:
            write_split(outs[split], rows[split], with_manifest=(split != "train"))

    report = {
        "seed": args.split_seed, "split_unit": "recording (one meta jsonl file)",
        "vocab": {"base": len(base_vocab), "new": len(new_tokens)},
        "splits": {s: {"segments": len(rows[s]), "hours": round(sum(r["duration"] for r in rows[s]) / 3600, 3),
                       "recordings": sorted({r["_src"] for r in rows[s]})} for s in rows},
    }
    with open(f"{outs['train']}/split_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    with open(f"{outs['train']}/vocab_extension.json", "w", encoding="utf-8") as f:
        json.dump({"base_vocab_size": len(base_vocab), "new_tokens": new_tokens}, f, ensure_ascii=False, indent=1)

    print("\nDONE")
    for s in ("train", "val", "test"):
        if rows[s]:
            print(f"  {s:5s}: {len(rows[s]):6d} segments | {sum(r['duration'] for r in rows[s]) / 3600:6.2f} h | {outs[s]}")
    print(f"  vocab: {len(base_vocab)} + {len(new_tokens)} new = {len(base_vocab) + len(new_tokens)}")
    print(f"  split report: {outs['train']}/split_report.json  (lists the recordings of every split)")


if __name__ == "__main__":
    main()