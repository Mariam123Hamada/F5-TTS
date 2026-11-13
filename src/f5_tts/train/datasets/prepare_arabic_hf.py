"""
Build an F5-TTS training set from the wavs + meta/*.jsonl folders ("datasets N") stored in a HF dataset repo,
for fine-tuning the ORIGINAL F5-TTS base model on a new language (Arabic) with an EXTENDED vocabulary.

Output (tokenizer "custom"):  data/<name>_custom/{raw.arrow, duration.json, vocab.txt, vocab_extension.json}
    vocab.txt = base vocab (same token order, so the pretrained embedding rows stay valid) + NEW tokens appended.

Text is converted with the same `convert_char_to_pinyin` that the inference code uses, so training and
inference tokenization match.

Example:
    python -m f5_tts.train.datasets.prepare_arabic_hf --name saudi_lora \
        --repo Mariam-Abdelsalam/Saudi_Dataset --data_dir /kaggle/working/saudi_data --download
(set HF_TOKEN in the environment; the repo is private)
"""

import argparse
import glob
import json
import os
import re
import shutil
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files


def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", t)).strip()


def read_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line[:-1] for line in f]          # same reading logic as f5_tts.model.utils.get_tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="saudi_lora", help="dataset name -> data/<name>_custom")
    ap.add_argument("--repo", default="Mariam-Abdelsalam/Saudi_Dataset", help="HF dataset repo with 'datasets N' folders")
    ap.add_argument("--data_dir", default="saudi_data", help="local folder for the downloaded wavs + meta")
    ap.add_argument("--download", action="store_true", help="download wavs + meta from the HF repo first (resumable)")
    ap.add_argument("--folders", default="", help="comma separated folder names to use, e.g. 'datasets 36,datasets 37' (default: all)")
    ap.add_argument("--min_dur", type=float, default=1.5)
    ap.add_argument("--max_dur", type=float, default=25.0)
    ap.add_argument("--min_sim", type=float, default=0.40, help="voice match with the target speaker (stored in meta as 'sim')")
    ap.add_argument("--min_logprob", type=float, default=-0.60, help="Whisper confidence (stored in meta as 'logprob')")
    ap.add_argument("--min_token_count", type=int, default=5, help="a character not in the base vocab is added only if it appears at least this often")
    ap.add_argument("--max_unk_frac", type=float, default=0.03, help="drop a sample if more than this fraction of its tokens is dropped")
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
    print(f"{len(metas)} meta files")

    # ---------------- 1) filter rows ----------------
    stats, total_sec, cand = Counter(), 0.0, []
    for mp in metas:
        folder = os.path.dirname(os.path.dirname(mp))
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
                cand.append((wav, text))
    print("filter report:", dict(stats))
    if not cand:
        raise SystemExit("nothing left after filtering - relax --min_sim / --min_logprob")

    # ---------------- 2) tokenize exactly like inference ----------------
    tokens_list = []
    for i in range(0, len(cand), 200):
        tokens_list += convert_char_to_pinyin([t for _, t in cand[i:i + 200]], polyphone=True)

    # ---------------- 3) extend the vocab with the new characters ----------------
    counts = Counter(tok for toks in tokens_list for tok in toks if tok not in base_set)
    new_tokens = [t for t, n in counts.most_common() if n >= args.min_token_count and t != "\n"]
    keep_set = base_set | set(new_tokens)
    dropped_tokens = Counter({t: n for t, n in counts.items() if t not in keep_set})
    print(f"new tokens added to the vocab: {len(new_tokens)} -> {''.join(new_tokens)}")
    if dropped_tokens:
        print("rare tokens removed from the text:", dict(dropped_tokens.most_common(15)))

    # ---------------- 4) durations (real, from the files) ----------------
    def dur_of(p):
        try:
            return sf.info(p).duration
        except Exception:
            return 0.0

    with ThreadPoolExecutor(16) as ex:
        durs = list(ex.map(dur_of, [w for w, _ in cand]))

    rows, durations, n_unk_drop = [], [], 0
    for (wav, _), toks, d in zip(cand, tokens_list, durs):
        if d <= 0.3:
            continue
        kept = [t for t in toks if t in keep_set]
        if not kept or (len(toks) - len(kept)) / max(len(toks), 1) > args.max_unk_frac:
            n_unk_drop += 1
            continue
        rows.append({"audio_path": wav, "text": kept, "duration": d})
        durations.append(d)
    print(f"dropped {n_unk_drop} samples because of too many unknown tokens")

    # ---------------- 5) write the dataset ----------------
    out_dir = str(files("f5_tts").joinpath(f"../../data/{args.name}_custom"))
    os.makedirs(out_dir, exist_ok=True)
    # explicit finalize(): the `with ArrowWriter(...)` pattern does not flush the last partial batch in recent `datasets`
    w = ArrowWriter(path=f"{out_dir}/raw.arrow", writer_batch_size=100)
    for r in rows:
        w.write(r)
    w.finalize()
    n_written = len(HFDataset.from_file(f"{out_dir}/raw.arrow"))
    if n_written != len(rows):
        raise RuntimeError(f"raw.arrow has {n_written} rows, expected {len(rows)}")
    with open(f"{out_dir}/duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": durations}, f)
    with open(f"{out_dir}/vocab.txt", "w", encoding="utf-8") as f:
        for tok in base_vocab + new_tokens:
            f.write(tok + "\n")
    with open(f"{out_dir}/vocab_extension.json", "w", encoding="utf-8") as f:
        json.dump({"base_vocab_size": len(base_vocab), "new_tokens": new_tokens}, f, ensure_ascii=False, indent=1)

    print(f"\nDONE -> {out_dir}")
    print(f"samples: {len(rows)} | hours kept: {sum(durations) / 3600:.2f} of {total_sec / 3600:.2f} | "
          f"vocab: {len(base_vocab)} + {len(new_tokens)} new = {len(base_vocab) + len(new_tokens)}")


if __name__ == "__main__":
    main()