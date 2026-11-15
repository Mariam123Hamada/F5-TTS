import argparse
import json
import math
import os
import shutil
import time
from importlib.resources import files

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from f5_tts.model import CFM, DiT
from f5_tts.model.dataset import DynamicBatchSampler, collate_fn, load_dataset
from f5_tts.model.lora import (
    EMBED_KEY,
    count_params,
    expand_embedding_rows,
    inject_lora,
    load_trainable_state_dict,
    merge_and_unload,
    param_groups,
    set_trainable,
    trainable_state_dict,
    val_loss_sum,
)
from f5_tts.model.utils import get_tokenizer


# --- F5TTS_v1_Base architecture (identical to src/f5_tts/configs/F5TTS_v1_Base.yaml) ---
def model_cfg(checkpoint_activations=False):
    return dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, text_mask_padding=True, conv_layers=4,
                pe_attn_head=None, attn_backend="torch", attn_mask_enabled=False,
                checkpoint_activations=checkpoint_activations)


MEL_CFG = dict(n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=100, target_sample_rate=24000,
               mel_spec_type="vocos")


def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tuning of F5-TTS v1 Base")
    p.add_argument("--dataset_name", required=True, help="data/<name>_custom must exist (see prepare_arabic_hf.py)")
    p.add_argument("--pretrain", default=None, help="base checkpoint (.safetensors/.pt). Default: SWivid/F5-TTS F5TTS_v1_Base")
    p.add_argument("--ckpt_dir", default=None, help="where lora_*.pt are written (default ckpts/<dataset_name>_lora)")
    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_targets", default="attn,ff", help="comma list from: attn, ff, adaln")
    p.add_argument("--no_train_text", action="store_true", help="do NOT train the text embedding/blocks (only if vocab not extended)")
    # optimisation
    p.add_argument("--lr_lora", type=float, default=1e-4)
    p.add_argument("--lr_text", type=float, default=2e-4, help="lr of the text embedding/blocks")
    p.add_argument("--batch_size_per_gpu", type=int, default=6000, help="frames per GPU per step (1 frame = 256/24000 s)")
    p.add_argument("--max_samples", type=int, default=32)
    p.add_argument("--grad_accumulation_steps", type=int, default=2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--num_warmup_updates", type=int, default=300)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--checkpoint_activations", action="store_true", help="saves a lot of GPU memory, ~30% slower")
    # saving / stopping
    p.add_argument("--last_per_updates", type=int, default=500)
    p.add_argument("--save_per_updates", type=int, default=2000)
    p.add_argument("--keep_last_n", type=int, default=2)
    p.add_argument("--max_hours", type=float, default=0.0, help="stop cleanly after this many hours (0 = no limit)")
    p.add_argument("--log_every", type=int, default=50)
    # validation (held-out recordings, see prepare_arabic_hf.py)
    p.add_argument("--val_dataset_name", default=None, help="e.g. saudi_lora_val  -> data/saudi_lora_val_custom")
    p.add_argument("--val_per_updates", type=int, default=1000)
    p.add_argument("--val_max_batches", type=int, default=60, help="batches per validation pass (0 = whole set)")
    p.add_argument("--seed", type=int, default=666)
    # export
    p.add_argument("--export_only", action="store_true", help="merge LoRA into the base weights and write model_merged.safetensors")
    p.add_argument("--export_from", default=None, help="lora checkpoint to export (default: <ckpt_dir>/lora_last.pt)")
    return p.parse_args()


# ------------------------------------------------------------------------------------------------
def data_paths(name):
    d = str(files("f5_tts").joinpath(f"../../data/{name}_custom"))
    return d, os.path.join(d, "vocab.txt")


def default_base_ckpt():
    from huggingface_hub import hf_hub_download

    return hf_hub_download("SWivid/F5-TTS", "F5TTS_v1_Base/model_1250000.safetensors")


def load_base_state_dict(path):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        sd = load_file(path, device="cpu")
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)["ema_model_state_dict"]
    sd = {k.replace("ema_model.", ""): v for k, v in sd.items() if k not in ("initted", "update", "step")}
    for k in ("mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"):  # same patch as the stock Trainer
        sd.pop(k, None)
    return sd


def build_model(args, vocab_path, log=print):
    """CFM(DiT v1 Base) with the base weights loaded; the text embedding table is grown for the extended vocab."""
    vocab_char_map, vocab_size = get_tokenizer(vocab_path, "custom")
    model = CFM(
        transformer=DiT(**model_cfg(args.checkpoint_activations), text_num_embeds=vocab_size,
                        mel_dim=MEL_CFG["n_mel_channels"]),
        mel_spec_kwargs=MEL_CFG,
        vocab_char_map=vocab_char_map,
    )
    sd = load_base_state_dict(args.pretrain or default_base_ckpt())
    old_rows = sd[EMBED_KEY].shape[0]
    if old_rows != vocab_size + 1:
        sd[EMBED_KEY] = expand_embedding_rows(sd[EMBED_KEY], vocab_size + 1, seed=args.seed)
        log(f"text embedding grown: {old_rows} -> {vocab_size + 1} rows ({vocab_size + 1 - old_rows} new tokens)")
    model.load_state_dict(sd, strict=True)
    return model, vocab_size


def lora_meta(args, vocab_size):
    return dict(lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_targets=args.lora_targets,
                train_text=not args.no_train_text, vocab_size=vocab_size)


# ------------------------------------------------------------------------------------------------
def export(args):
    ckpt_dir = args.ckpt_dir or str(files("f5_tts").joinpath(f"../../ckpts/{args.dataset_name}_lora"))
    src = args.export_from or os.path.join(ckpt_dir, "lora_last.pt")
    ck = torch.load(src, map_location="cpu", weights_only=True)
    m = ck["meta"]
    _, vocab_path = data_paths(args.dataset_name)
    args.lora_targets = m["lora_targets"]
    model, _ = build_model(args, vocab_path)
    inject_lora(model, m["lora_targets"], r=m["lora_r"], alpha=m["lora_alpha"], dropout=0.0)
    set_trainable(model, train_text=m["train_text"])
    load_trainable_state_dict(model, ck["trainable"])
    merge_and_unload(model)

    sd = {"ema_model." + k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()
          if k not in ("mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window")}
    from safetensors.torch import save_file

    out = os.path.join(ckpt_dir, "model_merged.safetensors")
    save_file(sd, out)
    shutil.copy2(vocab_path, os.path.join(ckpt_dir, "vocab.txt"))
    print(f"update {ck.get('update')} merged -> {out}\nvocab -> {os.path.join(ckpt_dir, 'vocab.txt')}")
    print("inference:  f5-tts_infer-cli --model F5TTS_v1_Base --ckpt_file model_merged.safetensors "
          "--vocab_file vocab.txt --ref_audio REF.wav --ref_text '...' --gen_text '...'")


# ------------------------------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.export_only:
        return export(args)

    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accumulation_steps, kwargs_handlers=[ddp])
    is_main = accelerator.is_main_process
    log = print if is_main else (lambda *a, **k: None)

    ckpt_dir = args.ckpt_dir or str(files("f5_tts").joinpath(f"../../ckpts/{args.dataset_name}_lora"))
    os.makedirs(ckpt_dir, exist_ok=True)
    data_dir, vocab_path = data_paths(args.dataset_name)
    last_path = os.path.join(ckpt_dir, "lora_last.pt")

    # ---- model: base weights -> LoRA -> trainable selection ----
    model, vocab_size = build_model(args, vocab_path, log)
    names = inject_lora(model, args.lora_targets, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    set_trainable(model, train_text=not args.no_train_text)
    total, trainable = count_params(model)
    log(f"LoRA injected in {len(names)} Linear layers | params: {total/1e6:.1f}M total, {trainable/1e6:.2f}M trainable "
        f"({100*trainable/total:.2f}%) | vocab {vocab_size}")
    meta = lora_meta(args, vocab_size)

    optimizer = AdamW(param_groups(model, args.lr_lora, args.lr_text))

    # ---- data (frame-based dynamic batches, exactly like the stock Trainer) ----
    train_dataset = load_dataset(args.dataset_name, "custom", mel_spec_kwargs=MEL_CFG)
    accelerator.even_batches = False
    sampler = SequentialSampler(train_dataset)
    batch_sampler = DynamicBatchSampler(sampler, args.batch_size_per_gpu, max_samples=args.max_samples,
                                        random_seed=args.seed, drop_residual=False)
    dataloader = DataLoader(train_dataset, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0, batch_sampler=batch_sampler)

    val_loader = None
    if args.val_dataset_name:
        val_dataset = load_dataset(args.val_dataset_name, "custom", mel_spec_kwargs=MEL_CFG)
        val_batches = DynamicBatchSampler(SequentialSampler(val_dataset), args.batch_size_per_gpu,
                                          max_samples=args.max_samples, random_seed=args.seed, drop_residual=False)
        val_loader = DataLoader(val_dataset, collate_fn=collate_fn, num_workers=2, pin_memory=True,
                                batch_sampler=val_batches)

    # ---- scheduler: linear warmup then linear decay (same as the stock Trainer) ----
    warmup_updates = args.num_warmup_updates * accelerator.num_processes
    total_updates = math.ceil(len(dataloader) / args.grad_accumulation_steps) * args.epochs
    decay_updates = max(total_updates - warmup_updates, 1)
    warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
    decay = LinearLR(optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_updates])

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    raw_model = accelerator.unwrap_model(model)
    trainable_params = [p for p in raw_model.parameters() if p.requires_grad]

    # ---- resume ----
    global_update = 0
    best_val = float("inf")
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=True)
        load_trainable_state_dict(raw_model, ck["trainable"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        global_update = int(ck["update"])
        best_val = float(ck.get("best_val", best_val))
        del ck
        log(f"RESUMED from {last_path} at update {global_update}")

    orig_epoch_step = len(dataloader)
    start_step = global_update * args.grad_accumulation_steps
    skipped_epoch = int(start_step // orig_epoch_step)
    skipped_batch = start_step % orig_epoch_step
    skipped_loader = accelerator.skip_first_batches(dataloader, num_batches=skipped_batch)
    log(f"batches/epoch per GPU: {orig_epoch_step} | updates/epoch ~ {math.ceil(orig_epoch_step/args.grad_accumulation_steps)} "
        f"| planned total updates ~ {total_updates}")

    def save(update, last):
        accelerator.wait_for_everyone()
        if not is_main:
            return
        payload = dict(trainable=trainable_state_dict(raw_model), update=update, meta=meta)
        path = last_path if last else os.path.join(ckpt_dir, f"lora_{update}.pt")
        if last:
            payload.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), best_val=best_val)
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)                       # atomic: an uploader never sees a half-written file
        if not last and args.keep_last_n > 0:
            snaps = sorted((f for f in os.listdir(ckpt_dir) if f.startswith("lora_") and f.endswith(".pt")
                            and f != "lora_last.pt" and f.split("_")[1].split(".")[0].isdigit()),
                           key=lambda f: int(f.split("_")[1].split(".")[0]))
            while len(snaps) > args.keep_last_n:
                os.remove(os.path.join(ckpt_dir, snaps.pop(0)))
        print(f"saved {'last' if last else 'snapshot'} checkpoint at update {update}", flush=True)

    def run_validation():
        """Mean validation loss over all processes (one collective call at the end, so ranks cannot deadlock)."""
        s_, n_ = val_loss_sum(model, val_loader, accelerator.device, seed=1234, max_batches=args.val_max_batches)
        t = accelerator.gather(torch.tensor([[s_, float(n_)]], device=accelerator.device)).sum(0)
        return (t[0] / t[1].clamp(min=1)).item()

    def save_best(update, val):
        if not is_main:
            return
        path = os.path.join(ckpt_dir, "lora_best.pt")
        tmp = path + ".tmp"
        torch.save(dict(trainable=trainable_state_dict(raw_model), update=update, meta=meta, val_loss=val), tmp)
        os.replace(tmp, path)

    if val_loader is not None and global_update == 0:
        v0 = run_validation()
        log(f"VAL baseline (before any training) | val_loss {v0:.4f}")

    # ---- training loop ----
    t0, stop = time.time(), False
    for epoch in range(skipped_epoch, args.epochs):
        model.train()
        if epoch == skipped_epoch:
            current, initial = skipped_loader, math.ceil(skipped_batch / args.grad_accumulation_steps)
        else:
            current, initial = dataloader, 0
        if hasattr(dataloader, "batch_sampler") and hasattr(dataloader.batch_sampler, "set_epoch"):
            dataloader.batch_sampler.set_epoch(epoch)

        bar = tqdm(range(math.ceil(len(dataloader) / args.grad_accumulation_steps)), desc=f"Epoch {epoch+1}/{args.epochs}",
                   unit="update", disable=not accelerator.is_local_main_process, initial=initial, mininterval=30)
        for batch in current:
            with accelerator.accumulate(model):
                text_inputs = batch["text"]
                mel_spec = batch["mel"].permute(0, 2, 1)
                mel_lengths = batch["mel_lengths"]
                loss, _, _ = model(mel_spec, text=text_inputs, lens=mel_lengths)
                accelerator.backward(loss)
                if args.max_grad_norm > 0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_update += 1
                bar.update(1)
                if is_main and global_update % args.log_every == 0:
                    print(f"update {global_update} | epoch {epoch+1} | loss {loss.item():.4f} | "
                          f"lr {scheduler.get_last_lr()[0]:.2e} | {(time.time()-t0)/60:.1f} min", flush=True)
                if global_update % args.last_per_updates == 0:
                    save(global_update, last=True)
                if global_update % args.save_per_updates == 0:
                    save(global_update, last=False)
                if val_loader is not None and global_update % args.val_per_updates == 0:
                    v = run_validation()
                    improved = v < best_val
                    if improved:
                        best_val = v
                        save_best(global_update, v)
                    log(f"VAL update {global_update} | val_loss {v:.4f} | best {best_val:.4f}" + ("  <- new best, saved lora_best.pt" if improved else ""))
                if args.max_hours > 0:                                     # all processes must stop together
                    flag = torch.tensor([int(time.time() - t0 > args.max_hours * 3600)], device=accelerator.device)
                    if accelerator.gather(flag).max().item() > 0:
                        stop = True
                        break
        bar.close()
        if stop:
            break

    save(global_update, last=True)
    log("time limit reached - state saved, run again to resume" if stop else "training finished")
    accelerator.end_training()


if __name__ == "__main__":
    main()