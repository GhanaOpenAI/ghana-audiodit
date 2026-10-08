"""
train_ghana.py — LoRA fine-tuning of LongCat-AudioDiT on the Ghana 3h/lang set.

Adapted from zjubinchen/LongCat-AudioDiT train/train.py:
  * reads pre-encoded Wav-VAE latents (cache_latents.py) instead of streaming wavs
  * single device (cuda:0), bf16 autocast + TF32, no Accelerate/DDP
  * CFG dropout drops text AND prompt jointly (paper §3) and builds the
    unconditional input exactly like inference does (modeling_audiodit.forward):
      text_emb -> 0 (text mask kept), latent_cond -> 0, noisy prompt region of x_t -> 0
  * held-out validation loss + listening samples at every checkpoint

Usage:
    python train/train_ghana.py --config train/config_ghana.yaml
"""

from __future__ import annotations

import argparse
import functools
import logging
import math
import random
import sys
import time
from pathlib import Path

FINETUNE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FINETUNE_DIR.parent))
sys.path.insert(0, str(FINETUNE_DIR))

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

import ghana_audiodit.audiodit  # noqa: F401  (registers AudioDiT with transformers)
from ghana_audiodit.audiodit import AudioDiTModel
from ghana_audiodit.audiodit.modeling_audiodit import lens_to_mask
from dataset_ghana import GhanaLatentDataset, ghana_collate_fn
from lora_utils import inject_lora, load_lora, save_lora
from noprompt_test import run_noprompt_samples
from sample import run_samples

log = logging.getLogger(__name__)

# full-tune components (parameter-name substrings inside model.transformer)
_FULL_SUBSTRINGS: dict[str, list[str]] = {
    "dit_adaln":            ["adaln_mlp.", "adaln_global_mlp."],
    "text_embed":           ["text_embed."],
    "text_conv":            ["text_conv_layer."],
    "latent_embed":         ["latent_embed."],
    "latent_cond_embedder": ["latent_cond_embedder."],
    "input_embed":          ["input_embed."],
    "output_proj":          ["norm_out.", "proj_out"],
    "time_embed":           ["time_embed."],
}


def setup_model(cfg: dict, device: torch.device) -> AudioDiTModel:
    comps, lora_cfg = cfg.get("components", {}), cfg.get("lora", {})
    model = AudioDiTModel.from_pretrained(cfg["model"]["model_dir"]).to(device)
    model.vae.to_half()

    init_from = cfg["training"].get("init_from")
    if init_from:
        # weights only (LoRA + full-tune extras); optimizer/scheduler start fresh
        load_lora(model, init_from, trainable=True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Initialised from {init_from}; trainable: {trainable:,}")
        return model

    inject_lora(
        model,
        r=lora_cfg.get("r", 32),
        lora_alpha=lora_cfg.get("alpha", lora_cfg.get("r", 32)),
        lora_dropout=lora_cfg.get("dropout", 0.0),
        include_ffn=comps.get("dit_ffn", {}).get("mode") == "lora",
    )
    for comp, subs in _FULL_SUBSTRINGS.items():
        if comps.get(comp, {}).get("mode") == "full":
            for name, p in model.transformer.named_parameters():
                if any(s in name for s in subs) and "lora_" not in name:
                    p.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return model


def cfm_loss(model: AudioDiTModel, batch: dict, device: torch.device, dtype: torch.dtype,
             p_uncond: float = 0.1, generator: torch.Generator | None = None) -> torch.Tensor:
    """Conditional flow matching loss on the generated (non-prompt) frames."""
    z1 = batch["latent"].to(device)                                   # [B, T, 64]
    lens = batch["latent_lens"].to(device)
    B, T, _ = z1.shape
    T_p = torch.minimum(batch["prompt_frames"].to(device), lens - 1)   # 0 = no prompt

    z0 = torch.randn(z1.shape, device=device, generator=generator)
    t = torch.rand(B, device=device, generator=generator)
    x_t = (1.0 - t[:, None, None]) * z0 + t[:, None, None] * z1
    v_target = z1 - z0

    prompt_mask = lens_to_mask(T_p, T)                                 # [B, T]
    audio_mask = lens_to_mask(lens, T)
    latent_cond = z1 * prompt_mask.unsqueeze(-1)

    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    text_emb = model.encode_text(input_ids, attn)                       # no_grad inside
    text_len = attn.sum(dim=1)
    text_mask = lens_to_mask(text_len, text_emb.shape[1])

    if p_uncond > 0:
        drop = torch.rand(B, device=device, generator=generator) < p_uncond
        keep = (~drop).view(B, 1, 1)
        text_emb = text_emb * keep.to(text_emb.dtype)
        latent_cond = latent_cond * keep.to(latent_cond.dtype)
        x_t = x_t.masked_fill((drop.view(B, 1) & prompt_mask).unsqueeze(-1), 0.0)

    with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32):
        out = model.transformer(
            x=x_t, text=text_emb, text_len=text_len, time=t,
            mask=audio_mask, cond_mask=text_mask, latent_cond=latent_cond,
        )
    v_pred = out["last_hidden_state"].float()

    gen_mask = audio_mask & ~prompt_mask
    return F.mse_loss(v_pred[gen_mask], v_target[gen_mask])


@torch.no_grad()
def validate(model, val_dl, device, dtype) -> float:
    model.eval()
    g = torch.Generator(device=device).manual_seed(0)   # same noise/t every time
    random.seed(0)                                       # same prompt splits every time
    tot, n = 0.0, 0
    for batch in val_dl:
        tot += cfm_loss(model, batch, device, dtype, p_uncond=0.0, generator=g).item()
        n += 1
    model.train()
    model.vae.eval()
    model.text_encoder.eval()
    return tot / max(n, 1)


def save_checkpoint(model, out_dir: Path, step: int, optimizer, scheduler):
    ckpt = out_dir / f"step_{step:07d}"
    save_lora(model, ckpt)
    torch.save({"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
               ckpt / "training_state.pt")
    # keep the newest N checkpoints
    keep = sorted(out_dir.glob("step_*"))
    for old in keep[: -model._keep_last_n] if model._keep_last_n else []:
        for f in sorted(old.rglob("*"), reverse=True):
            f.unlink() if f.is_file() else f.rmdir()
        old.rmdir()
    log.info(f"Saved → {ckpt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_ghana.yaml")
    args = ap.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute() and not cfg_path.exists():
        cfg_path = FINETUNE_DIR / cfg_path
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
    tr, dc = cfg["training"], cfg["data"]
    random.seed(tr.get("seed", 42))
    torch.manual_seed(tr.get("seed", 42))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda:0")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[cfg["model"].get("dtype", "bfloat16")]

    model = setup_model(cfg, device)
    model._keep_last_n = tr.get("keep_last_n", 0)
    model.train()
    model.vae.eval()
    model.text_encoder.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)

    ds_kwargs = {k: dc[k] for k in ("min_audio_sec", "max_audio_sec", "prompt_min_sec", "prompt_max_sec",
                                    "min_gen_sec", "prompt_frac_lo", "prompt_frac_hi", "min_text_chars")
                 if k in dc}
    collate = functools.partial(ghana_collate_fn, tokenizer=tokenizer, max_text_len=dc.get("max_text_len", 512))
    p_no_prompt = dc.get("p_no_prompt", 0.0)
    train_ds = GhanaLatentDataset(dc["train_manifest"], langs=dc.get("langs"), p_no_prompt=p_no_prompt, **ds_kwargs)
    val_ds = GhanaLatentDataset(dc["val_manifest"], langs=dc.get("langs"), **ds_kwargs)
    # second validation set scored entirely without a prompt
    val_np_ds = GhanaLatentDataset(dc["val_manifest"], langs=dc.get("langs"), p_no_prompt=1.0, **ds_kwargs) \
        if p_no_prompt else None
    nw = tr.get("num_workers", 8)
    train_dl = DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=True, drop_last=True,
                          num_workers=nw, pin_memory=True, persistent_workers=nw > 0, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=tr["batch_size"], shuffle=False, num_workers=0, collate_fn=collate)
    val_np_dl = DataLoader(val_np_ds, batch_size=tr["batch_size"], shuffle=False, num_workers=0,
                           collate_fn=collate) if val_np_ds else None
    log.info(f"train: {len(train_ds)} clips → {len(train_dl)} steps/epoch; val: {len(val_ds)} clips")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=tr["learning_rate"], betas=tuple(tr.get("betas", (0.9, 0.95))),
                                  weight_decay=tr.get("weight_decay", 0.01))
    scheduler = get_cosine_schedule_with_warmup(optimizer, tr.get("warmup_steps", 500), tr["steps"])

    out_dir = Path(tr["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    if tr.get("resume_from"):
        load_lora(model, tr["resume_from"], trainable=True)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=tr["learning_rate"], betas=tuple(tr.get("betas", (0.9, 0.95))),
                                      weight_decay=tr.get("weight_decay", 0.01))
        scheduler = get_cosine_schedule_with_warmup(optimizer, tr.get("warmup_steps", 500), tr["steps"])
        ts = torch.load(Path(tr["resume_from"]) / "training_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(ts["optimizer"])
        scheduler.load_state_dict(ts["scheduler"])
        step = ts["step"]
        log.info(f"Resumed from {tr['resume_from']} at step {step}")

    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    grad_acc = tr.get("gradient_accumulation", 1)
    log_every, save_every, val_every = tr.get("log_every", 50), tr.get("save_every", 1000), tr.get("val_every", 500)
    p_uncond = tr.get("p_uncond", 0.1)

    def do_val(step):
        vl = validate(model, val_dl, device, dtype)
        msg = f"step={step:6d}  val_loss={vl:.4f}"
        writer.add_scalar("val/loss", vl, step)
        if val_np_dl is not None:
            vn = validate(model, val_np_dl, device, dtype)
            msg += f"  val_loss_noprompt={vn:.4f}"
            writer.add_scalar("val/loss_noprompt", vn, step)
        log.info(msg)

    def do_samples(step):
        run_samples(model, tokenizer, cfg, out_dir, step, writer)
        if cfg.get("samples", {}).get("noprompt"):
            run_noprompt_samples(model, tokenizer, cfg, out_dir, step, writer)

    if step == 0:
        do_val(0)
        if cfg.get("samples", {}).get("at_start", True):
            do_samples(0)

    run_loss, t0 = 0.0, time.time()
    optimizer.zero_grad(set_to_none=True)
    micro = 0
    while step < tr["steps"]:
        for batch in train_dl:
            loss = cfm_loss(model, batch, device, dtype, p_uncond=p_uncond) / grad_acc
            loss.backward()
            run_loss += loss.item()
            micro += 1
            if micro % grad_acc:
                continue

            gn = torch.nn.utils.clip_grad_norm_(params, tr.get("max_grad_norm", 1.0))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % log_every == 0:
                avg = run_loss / log_every
                if not math.isfinite(avg):
                    raise RuntimeError(f"non-finite loss at step {step}")
                lr = scheduler.get_last_lr()[0]
                log.info(f"step={step:6d}  loss={avg:.4f}  gnorm={gn:.3f}  lr={lr:.2e}  "
                         f"{(time.time() - t0) / log_every:.2f}s/step")
                writer.add_scalar("train/loss", avg, step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/grad_norm", gn, step)
                run_loss, t0 = 0.0, time.time()

            if step % val_every == 0:
                do_val(step)

            if step % save_every == 0 or step == tr["steps"]:
                save_checkpoint(model, out_dir, step, optimizer, scheduler)
                do_samples(step)
                t0 = time.time()

            if step >= tr["steps"]:
                break

    save_lora(model, out_dir / "merged", merged=True)
    writer.close()
    log.info("Training complete.")


if __name__ == "__main__":
    main()
