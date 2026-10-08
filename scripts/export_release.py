"""
export_release.py — Build the Hugging Face release folder from a training checkpoint.

    python scripts/export_release.py \
        --ckpt   <run>/step_0020000 \
        --train_manifest <workdir>/latents/train_universal.jsonl \
        --out    <release>

Produces:
    model.safetensors, config.json      merged model (LoRA folded into the base), fp32 like the base
    configuration_audiodit.py, modeling_audiodit.py
                                        so AutoModel.from_pretrained(..., trust_remote_code=True) works
    lora/                               the LoRA adapter + full-tuned layers, for further fine-tuning
    rates.json                          median latent frames per character, per language (durations)
    LICENSE-longcat-audiodit            upstream MIT license (weights card: CC-BY-NC-4.0, following the data)
"""

import argparse
import json
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import torch

import ghana_audiodit.audiodit  # noqa: F401
from ghana_audiodit.audiodit import AudioDiTModel
from ghana_audiodit.utils import normalize_text
from lora_utils import load_lora


def rates(manifest: str) -> dict[str, float]:
    per = defaultdict(list)
    for line in open(manifest, encoding="utf-8"):
        m = json.loads(line)
        n = len(normalize_text(m["text"]).replace(" ", ""))
        if n >= 10:
            per[m["lang"]].append(m["frames"] / n)
    return {k: round(statistics.median(v), 4) for k, v in sorted(per.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="meituan-longcat/LongCat-AudioDiT-1B")
    args = ap.parse_args()
    out, ckpt = Path(args.out), Path(args.ckpt)
    out.mkdir(parents=True, exist_ok=True)

    model = AudioDiTModel.from_pretrained(args.base)
    load_lora(model, ckpt)
    model.transformer = model.transformer.merge_and_unload()
    model.config.auto_map = {"AutoConfig": "configuration_audiodit.AudioDiTConfig",
                             "AutoModel": "modeling_audiodit.AudioDiTModel"}
    model.save_pretrained(str(out), safe_serialization=True)
    pkg = ROOT / "ghana_audiodit" / "audiodit"
    for f in ("configuration_audiodit.py", "modeling_audiodit.py"):
        shutil.copy(pkg / f, out / f)
    print(f"[export] merged model → {out}", flush=True)

    (out / "lora").mkdir(exist_ok=True)
    for f in ("adapter_config.json", "adapter_model.safetensors", "extras.pt"):
        shutil.copy(ckpt / f, out / "lora" / f)
    (out / "rates.json").write_text(json.dumps(rates(args.train_manifest), indent=1), encoding="utf-8")
    shutil.copy(ROOT / "LICENSE-longcat-audiodit", out / "LICENSE-longcat-audiodit")
    print(f"[export] lora/, rates.json, LICENSE-longcat-audiodit → {out}", flush=True)

    # sanity: merged transformer must equal base+LoRA on a random input
    ref = AudioDiTModel.from_pretrained(args.base)
    load_lora(ref, ckpt)
    merged = AudioDiTModel.from_pretrained(str(out))
    ref.eval(), merged.eval()
    torch.manual_seed(0)
    B, T, S = 1, 40, 12
    kw = dict(x=torch.randn(B, T, 64), text=torch.randn(B, S, 768), text_len=torch.tensor([S]),
              time=torch.tensor([0.5]), mask=torch.ones(B, T, dtype=torch.bool),
              cond_mask=torch.ones(B, S, dtype=torch.bool), latent_cond=torch.zeros(B, T, 64))
    with torch.no_grad():
        d = (ref.transformer(**kw)["last_hidden_state"] - merged.transformer(**kw)["last_hidden_state"]).abs().max()
    print(f"[export] merged vs base+LoRA max |diff| = {d.item():.2e}", flush=True)
    assert d < 1e-3, "merged weights do not reproduce base+LoRA"


if __name__ == "__main__":
    main()
