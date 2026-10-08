"""
eval_noprompt_cer.py — No-prompt CER of a released model (and the base model), averaged over seeds.

Same 12 held-out sentences as the side evaluator (Asante Twi, Ewe, Dagbani × 4), generated
without a voice prompt with several seeds (several random voices) and transcribed with
omniASR CTC-300M (sherpa-onnx, CPU). Averaging over seeds removes most of the
"which voice did this seed land on" noise of a single no-prompt sample.

    OMNIASR_SHERPA_DIR=... python training/eval_noprompt_cer.py --release <release dir> \
        --val_manifest <workdir>/latents/val_universal.jsonl [--base meituan-longcat/LongCat-AudioDiT-1B]
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import AutoTokenizer

from cer_watch import build_eval_set, cer, norm, run_asr, to_univ
from ghana_audiodit import GhanaTTS
from ghana_audiodit.audiodit import AudioDiTModel

LANGS = ["Asante_Twi_twi", "Ewe_ewe", "Dagbani_dag"]


def score(tts: GhanaTTS, items, seeds, tag: str, tmp: Path) -> dict:
    jobs, refs = [], []
    for it in items:
        for seed in seeds:
            r = tts.synthesize(it["target"]["text"], language=it["lang"], seed=seed, text_is_universal=True)
            f = tmp / f"{tag}_{it['n']}_{seed}.wav"
            r.save(f)
            jobs.append({"wav": str(f)})
            refs.append(it)
    hyps = run_asr(jobs, tmp)
    per_lang: dict[str, list[tuple[float, float]]] = {}
    for it, h in zip(refs, hyps):
        ref = it["target"]["text_native"]
        per_lang.setdefault(it["lang"], []).append(
            (cer(norm(ref), norm(h)), cer(norm(to_univ(ref, it["lang"])), norm(to_univ(h, it["lang"])))))
    lang_avg = {k: (sum(a for a, _ in v) / len(v), sum(b for _, b in v) / len(v)) for k, v in per_lang.items()}
    nat = sum(a for a, _ in lang_avg.values()) / len(lang_avg)
    uni = sum(b for _, b in lang_avg.values()) / len(lang_avg)
    detail = "  ".join(f"{k.split('_')[0]} {b:.1f}" for k, (_, b) in sorted(lang_avg.items()))
    print(f"{tag:10s} no-prompt CER  universal {uni:5.1f}  normal {nat:5.1f}   ({len(jobs)} clips; universal by language: {detail})",
          flush=True)
    return {"universal": uni, "normal": nat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--val_manifest", required=True)
    ap.add_argument("--base", default="meituan-longcat/LongCat-AudioDiT-1B")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    items = build_eval_set(args.val_manifest, 4, draw=4, langs=LANGS)
    seeds = list(range(args.seeds))
    tmp = Path(tempfile.mkdtemp())

    if args.base:
        base = AudioDiTModel.from_pretrained(args.base).to("cuda")
        base.vae.to_half()
        base.transformer.to(torch.bfloat16)
        base.eval()
        tts = GhanaTTS(base, AutoTokenizer.from_pretrained(base.config.text_encoder_model), Path(args.release))
        score(tts, items, seeds, "base", tmp)
        del tts, base
        torch.cuda.empty_cache()
    score(GhanaTTS.from_pretrained(args.release), items, seeds, "release", tmp)


if __name__ == "__main__":
    main()
