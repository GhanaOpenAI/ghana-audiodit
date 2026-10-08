"""
sample.py — Generate listening samples from held-out (val) clips.

For each chosen language: prompt = one val clip (audio + its transcript),
target text = another val clip's transcript in the same language. Output wavs
go to <out_dir>/samples/step_XXXXXXX/<lang>.wav with a texts.tsv alongside.

Used from train.py at every checkpoint, or standalone:
    python train/sample.py --config train/config_ghana.yaml --ckpt checkpoints/step_0002000
    python train/sample.py --config train/config_ghana.yaml            # base model
"""

import argparse
import json
import random
import sys
from pathlib import Path

FINETUNE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FINETUNE_DIR.parent))
sys.path.insert(0, str(FINETUNE_DIR))

import soundfile as sf
import torch
import yaml

from cache_latents import normalize_loudness
from ghana_audiodit.utils import load_audio, normalize_text


def pick_pairs(val_manifest: str, langs: list[str] | None, seed: int = 0):
    by_lang: dict[str, list[dict]] = {}
    with open(val_manifest, encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            by_lang.setdefault(m["lang"], []).append(m)
    rng = random.Random(seed)
    pairs = []
    for lang in (langs or sorted(by_lang)):
        clips = by_lang.get(lang, [])
        # prompt: 3–10 s clip; target: a different clip with a non-trivial text
        prompts = [c for c in clips if 35 <= c["frames"] <= 117]
        if not prompts or len(clips) < 2:
            continue
        p = rng.choice(prompts)
        # filter on native length so both text modes pick the same pairs
        targets = [c for c in clips if c["id"] != p["id"] and len(c.get("text_native", c["text"])) >= 20]
        if not targets:
            continue
        pairs.append((lang, p, rng.choice(targets)))
    return pairs


@torch.no_grad()
def synth(model, tokenizer, prompt_wav_path: str, prompt_text: str, gen_text: str,
          nfe: int = 16, cfg_strength: float = 4.0, guidance_method: str = "apg"):
    sr = model.config.sampling_rate
    hop = model.config.latent_hop
    max_frames = int(model.config.max_wav_duration * sr // hop)

    prompt_text, gen_text = normalize_text(prompt_text).strip(), normalize_text(gen_text).strip()
    inputs = tokenizer([f"{prompt_text} {gen_text}"], padding="longest", return_tensors="pt")
    # prompts must get the same loudness normalisation as the training data
    prompt_wav = torch.from_numpy(normalize_loudness(load_audio(prompt_wav_path, sr)[0].numpy())).view(1, 1, -1)

    _, prompt_frames = model.encode_prompt_audio(prompt_wav)
    # Duration from the prompt speaker's own character rate (language-agnostic).
    rate = prompt_frames / max(len(prompt_text.replace(" ", "")), 1)
    gen_frames = int(rate * len(gen_text.replace(" ", "")) * 1.05)
    duration = min(prompt_frames + max(gen_frames, 12), max_frames)

    out = model(
        input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
        prompt_audio=prompt_wav, duration=duration, steps=nfe,
        cfg_strength=cfg_strength, guidance_method=guidance_method,
    )
    return out.waveform.squeeze().float().cpu().numpy()


def run_samples(model, tokenizer, cfg: dict, out_dir: Path, step: int, writer=None):
    s_cfg = cfg.get("samples", {})
    pairs = pick_pairs(cfg["data"]["val_manifest"], s_cfg.get("langs"), s_cfg.get("seed", 0))
    step_dir = out_dir / "samples" / f"step_{step:07d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    torch.manual_seed(1234)
    rows = []
    for lang, p, t in pairs:
        wav_path = Path(p["wav"])
        try:
            wav = synth(model, tokenizer, str(wav_path), p["text"], t["text"],
                        nfe=s_cfg.get("nfe", 16), cfg_strength=s_cfg.get("cfg_strength", 4.0),
                        guidance_method=s_cfg.get("guidance_method", "apg"))
        except Exception as e:  # never kill training over a sample
            print(f"[samples] {lang} failed: {e}", flush=True)
            continue
        sr = model.config.sampling_rate
        sf.write(step_dir / f"{lang}.wav", wav, sr)
        (step_dir.parent / "prompts").mkdir(exist_ok=True)
        sf.write(step_dir.parent / "prompts" / f"{lang}_prompt.wav", load_audio(str(wav_path), sr)[0].numpy(), sr)
        rows.append(f"{lang}\t{p['id']}\t{p['text']}\t{t['text']}")
        if writer is not None:
            writer.add_audio(f"samples/{lang}", torch.from_numpy(wav), step, sample_rate=sr)
    (step_dir / "texts.tsv").write_text("lang\tprompt_id\tprompt_text\tgen_text\n" + "\n".join(rows) + "\n",
                                        encoding="utf-8")
    print(f"[samples] wrote {len(rows)} samples → {step_dir}", flush=True)
    if was_training:
        model.train()
        model.vae.eval()
        model.text_encoder.eval()


def main():
    import audiodit  # noqa: F401
    from audiodit import AudioDiTModel
    from transformers import AutoTokenizer
    from lora_utils import load_lora

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="LoRA checkpoint dir (omit for base model)")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    model = AudioDiTModel.from_pretrained(cfg["model"]["model_dir"]).to("cuda:0")
    model.vae.to_half()
    if args.ckpt:
        load_lora(model, args.ckpt)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
    step = int(Path(args.ckpt).name.split("_")[-1]) if args.ckpt else 0
    out_dir = Path(args.out_dir or cfg["training"]["output_dir"])
    run_samples(model, tokenizer, cfg, out_dir, step)


if __name__ == "__main__":
    main()
