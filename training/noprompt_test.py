"""
noprompt_test.py — Generate without a voice prompt (random voice), base vs fine-tuned.

Duration per sentence = language's median latent-frames-per-character (from the
training manifest) × target character count.

    python train/noprompt_test.py --config train/config_ghana_universal.yaml \
        --ckpt .../checkpoints/lora_r64_universal/step_0028000 --seeds 2
Output: <ckpt parent>/noprompt_test/{base,step_XXXXXXX}/<lang>_seed<k>.wav + texts.tsv
"""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

FINETUNE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FINETUNE_DIR.parent))
sys.path.insert(0, str(FINETUNE_DIR))

import soundfile as sf
import torch
import yaml
from transformers import AutoTokenizer

import ghana_audiodit.audiodit  # noqa: F401  (registers AudioDiT with transformers)
from ghana_audiodit.audiodit import AudioDiTModel
from lora_utils import load_lora
from ghana_audiodit.utils import normalize_text


def frames_per_char(manifest: str) -> dict[str, float]:
    per = defaultdict(list)
    for line in open(manifest, encoding="utf-8"):
        m = json.loads(line)
        n = len(normalize_text(m["text"]).replace(" ", ""))
        if n >= 10:
            per[m["lang"]].append(m["frames"] / n)
    return {k: statistics.median(v) for k, v in per.items()}


@torch.no_grad()
def generate(model, tokenizer, text: str, frames: int, seed: int, s_cfg: dict):
    torch.manual_seed(seed)
    inputs = tokenizer([normalize_text(text).strip()], padding="longest", return_tensors="pt")
    out = model(
        input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
        duration=frames, steps=s_cfg.get("nfe", 16),
        cfg_strength=s_cfg.get("cfg_strength", 4.0),
        guidance_method=s_cfg.get("guidance_method", "apg"),
    )
    return out.waveform.squeeze().float().cpu().numpy()


_FPC_CACHE: dict[str, dict[str, float]] = {}


def run_noprompt_samples(model, tokenizer, cfg: dict, out_dir: Path, step: int, writer=None, seeds: int = 2):
    """Per-checkpoint no-prompt samples, same target sentences as run_samples."""
    from sample import pick_pairs

    s_cfg = cfg.get("samples", {})
    manifest = cfg["data"]["train_manifest"]
    if manifest not in _FPC_CACHE:
        _FPC_CACHE[manifest] = frames_per_char(manifest)
    fpc = _FPC_CACHE[manifest]
    sr = model.config.sampling_rate
    max_frames = int(model.config.max_wav_duration * sr // model.config.latent_hop)
    d = out_dir / "samples_noprompt" / f"step_{step:07d}"
    d.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    n = 0
    for lang, _, t in pick_pairs(cfg["data"]["val_manifest"], s_cfg.get("langs"), s_cfg.get("seed", 0)):
        text = t["text"]
        frames = min(max(int(fpc[lang] * len(normalize_text(text).replace(" ", ""))), 12), max_frames)
        for k in range(seeds):
            try:
                wav = generate(model, tokenizer, text, frames, 1000 + k, s_cfg)
            except Exception as e:  # never kill training over a sample
                print(f"[noprompt] {lang} failed: {e}", flush=True)
                continue
            sf.write(d / f"{lang}_seed{k}.wav", wav, sr)
            n += 1
            if writer is not None and k == 0:
                writer.add_audio(f"samples_noprompt/{lang}", torch.from_numpy(wav), step, sample_rate=sr)
    print(f"[noprompt] wrote {n} samples → {d}", flush=True)
    if was_training:
        model.train()
        model.vae.eval()
        model.text_encoder.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    s_cfg = cfg.get("samples", {})
    ckpt = Path(args.ckpt)
    out_root = ckpt.parent / "noprompt_test"

    # same target sentences as the prompted samples at this checkpoint
    rows = list(csv.DictReader(open(ckpt.parent / "samples" / ckpt.name / "texts.tsv", encoding="utf-8"),
                               delimiter="\t"))
    fpc = frames_per_char(cfg["data"]["train_manifest"])

    model = AudioDiTModel.from_pretrained(cfg["model"]["model_dir"]).to("cuda:0")
    model.vae.to_half()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
    sr = model.config.sampling_rate
    max_frames = int(model.config.max_wav_duration * sr // model.config.latent_hop)

    for tag in ("base", ckpt.name):
        if tag != "base":
            load_lora(model, ckpt)
            model.eval()
        d = out_root / tag
        d.mkdir(parents=True, exist_ok=True)
        log = ["lang\tseed\tframes\ttext"]
        for r in rows:
            lang, text = r["lang"], r["gen_text"]
            frames = min(max(int(fpc[lang] * len(normalize_text(text).replace(" ", ""))), 12), max_frames)
            for k in range(args.seeds):
                wav = generate(model, tokenizer, text, frames, 1000 + k, s_cfg)
                sf.write(d / f"{lang}_seed{k}.wav", wav, sr)
                log.append(f"{lang}\t{k}\t{frames}\t{text}")
        (d / "texts.tsv").write_text("\n".join(log) + "\n", encoding="utf-8")
        print(f"[noprompt] {tag}: {len(rows) * args.seeds} wavs → {d}", flush=True)


if __name__ == "__main__":
    main()
