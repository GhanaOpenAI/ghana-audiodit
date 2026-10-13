"""
eval_checkpoints_cer.py — Evaluate Character Error Rate (CER) across checkpoints
using omniASR CTC-300M (sherpa-onnx) on held-out evaluation pairs.

Computes:
  1. The "Real Speech" floor CER (ASR on the ground-truth audio).
  2. Prompt-conditioned synthesis CER for each checkpoint.
  3. CER Universal (normalised under africa-g2p) and CER Native.

Usage:
    python training/eval_checkpoints_cer.py \
        --config training/configs/config_twi_multispeaker_prompted.yaml \
        --n_utts 6
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import sherpa_onnx
import soundfile as sf
import torch
import yaml
from africa_g2p import UNIVERSAL, convert_lang
from transformers import AutoTokenizer

import ghana_audiodit.audiodit  # noqa: F401
from ghana_audiodit.audiodit import AudioDiTModel
from lora_utils import load_lora
from sample import synth

DEFAULT_ASR_DIR = "/mnt/volume_d2wey28/models/sherpa/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-2025-11-12"


def norm(text: str) -> str:
    out = []
    for ch in text.upper():
        if ch.isspace():
            out.append(" ")
        elif unicodedata.category(ch)[0] in ("L", "N", "M"):
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 100.0
    prev = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        cur = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return min(prev[-1] / len(ref) * 100, 100.0)


def to_univ(text: str, code: str = "twi") -> str:
    try:
        return norm(convert_lang(text, code, UNIVERSAL))
    except Exception:
        return norm(text)


def load_pairs(val_manifest: str, n_utts: int, seed: int = 42):
    clips = []
    for line in open(val_manifest, encoding="utf-8"):
        c = json.loads(line)
        if c.get("wav") and os.path.exists(c["wav"]) and len(c.get("text_native", c["text"])) >= 25:
            clips.append(c)

    rng = random.Random(seed)
    rng.shuffle(clips)
    if len(clips) < n_utts + 1:
        raise ValueError(f"Need at least {n_utts + 1} validation clips with wavs, found {len(clips)}")

    prompt_clip = clips[0]
    target_clips = clips[1: n_utts + 1]
    return prompt_clip, target_clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training/configs/config_twi_multispeaker_prompted.yaml")
    ap.add_argument("--asr_dir", default=DEFAULT_ASR_DIR)
    ap.add_argument("--n_utts", type=int, default=6, help="Number of held-out test sentences")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out_dir = Path(cfg["training"]["output_dir"])

    print(f"[cer_eval] Initializing sherpa-onnx omniASR from {args.asr_dir}...")
    rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=f"{args.asr_dir}/model.onnx",
        tokens=f"{args.asr_dir}/tokens.txt",
        num_threads=4,
        decoding_method="greedy_search",
    )

    def transcribe(wav_path: str) -> str:
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        stream = rec.create_stream()
        stream.accept_waveform(sr, audio.mean(axis=1))
        rec.decode_stream(stream)
        return (stream.result.text or "").strip()

    prompt_clip, target_clips = load_pairs(cfg["data"]["val_manifest"], args.n_utts)
    print(f"[cer_eval] Loaded prompt clip {prompt_clip['id']} and {len(target_clips)} evaluation targets.")

    # 1. Ground truth real speech floor
    real_cers_univ = []
    for tgt in target_clips:
        hyp = transcribe(tgt["wav"])
        ref_u = to_univ(tgt.get("text_native", tgt["text"]))
        hyp_u = to_univ(hyp)
        real_cers_univ.append(cer(ref_u, hyp_u))
    floor_cer = sum(real_cers_univ) / len(real_cers_univ)
    print(f"\n=======================================================")
    print(f"  REAL RECORDINGS FLOOR CER (omniASR baseline): {floor_cer:.2f}%")
    print(f"=======================================================\n")

    # Discover checkpoints
    ckpts = [None]  # None = base model at step 0
    saved_dirs = sorted(glob.glob(str(out_dir / "step_*")))
    for sd in saved_dirs:
        if os.path.exists(os.path.join(sd, "adapter_model.safetensors")):
            ckpts.append(Path(sd))

    print(f"{'Checkpoint':16s} | {'Universal CER':15s} | {'Native CER':15s} | {'Time':8s}")
    print("-" * 62)

    eval_tmp_dir = out_dir / "eval_cer_tmp"
    eval_tmp_dir.mkdir(parents=True, exist_ok=True)

    results = [{"checkpoint": "real_audio", "cer_universal": floor_cer, "step": -1}]

    for ckpt in ckpts:
        step_name = "base_model" if ckpt is None else ckpt.name
        t0 = time.time()

        # Load AudioDiT model
        model = AudioDiTModel.from_pretrained(cfg["model"]["model_dir"]).to(args.device)
        model.vae.to_half()
        if ckpt is not None:
            load_lora(model, ckpt)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)

        step_cers_u, step_cers_n = [], []
        for i, tgt in enumerate(target_clips):
            gen_text = tgt.get("text_native", tgt["text"])
            prompt_text = prompt_clip.get("text_native", prompt_clip["text"])
            try:
                wav = synth(model, tokenizer, prompt_clip["wav"], prompt_text, gen_text,
                            nfe=cfg.get("samples", {}).get("nfe", 16),
                            cfg_strength=cfg.get("samples", {}).get("cfg_strength", 4.0),
                            guidance_method=cfg.get("samples", {}).get("guidance_method", "apg"))
                synth_path = eval_tmp_dir / f"{step_name}_{i}.wav"
                sf.write(str(synth_path), wav, model.config.sampling_rate)
                hyp = transcribe(str(synth_path))

                ref_u = to_univ(gen_text)
                hyp_u = to_univ(hyp)
                ref_n = norm(gen_text)
                hyp_n = norm(hyp)

                step_cers_u.append(cer(ref_u, hyp_u))
                step_cers_n.append(cer(ref_n, hyp_n))
            except Exception as e:
                print(f"Error on {step_name} utt {i}: {e}")

        # Free GPU memory immediately
        del model
        torch.cuda.empty_cache()

        dt = time.time() - t0
        avg_u = sum(step_cers_u) / max(len(step_cers_u), 1)
        avg_n = sum(step_cers_n) / max(len(step_cers_n), 1)

        print(f"{step_name:16s} | {avg_u:13.2f}% | {avg_n:13.2f}% | {dt:6.1f}s")
        results.append({
            "checkpoint": step_name,
            "cer_universal": round(avg_u, 2),
            "cer_native": round(avg_n, 2),
            "step": int(step_name.split("_")[1]) if step_name.startswith("step_") else 0,
            "seconds": round(dt, 1)
        })

    if args.out_json:
        json.dump(results, open(args.out_json, "w"), indent=2)

    print("\n[cer_eval] Evaluation complete.")


if __name__ == "__main__":
    main()
