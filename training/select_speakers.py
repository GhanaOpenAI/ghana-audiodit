"""
select_speakers.py — Pick one built-in ("fixed") speaker per language for inference.

For each language: N random training clips of 5–9 s with a reasonable transcript
are transcribed with omniASR CTC 300M (sherpa-onnx, CPU); the clip omniASR finds
clearest (lowest CER vs its transcript) becomes that language's fixed speaker.
Saved as 24 kHz mono, RMS-normalised to -23 dBFS (the training-data level).

    python train/select_speakers.py --manifest .../latents/train_universal.jsonl --out .../release/speakers
Output: <out>/<lang>.wav + <out>/speakers.json
"""

import argparse
import json
import random
import tempfile
from pathlib import Path

import librosa
import soundfile as sf

from cache_latents import normalize_loudness
from cer_watch import cer, norm, run_asr

FRAME_RATE = 24000 / 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_lang", type=int, default=12)
    ap.add_argument("--min_sec", type=float, default=5.0)
    ap.add_argument("--max_sec", type=float, default=9.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    by_lang: dict[str, list[dict]] = {}
    for line in open(args.manifest, encoding="utf-8"):
        m = json.loads(line)
        sec = m["frames"] / FRAME_RATE
        native = m.get("text_native", m["text"])
        if args.min_sec <= sec <= args.max_sec and 40 <= len(native) <= 200:
            by_lang.setdefault(m["lang"], []).append(m)
    cands = []
    for lang in sorted(by_lang):
        clips = sorted(by_lang[lang], key=lambda m: m["id"])
        cands += random.Random(0).sample(clips, min(args.per_lang, len(clips)))
    print(f"[speakers] {len(cands)} candidates over {len(by_lang)} languages", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        hyps = run_asr([{"wav": m["wav"]} for m in cands], Path(tmp))
    best: dict[str, tuple[float, dict, str]] = {}
    for m, h in zip(cands, hyps):
        ref = m.get("text_native", m["text"])
        c = cer(norm(ref), norm(h))
        sec = m["frames"] / FRAME_RATE
        key = (c, abs(sec - 7.0))          # clearest first, then closest to 7 s
        if m["lang"] not in best or key < best[m["lang"]][0]:
            best[m["lang"]] = (key, m, h)

    meta = {}
    for lang, ((c, _), m, h) in sorted(best.items()):
        audio, _ = librosa.load(m["wav"], sr=24000, mono=True)
        sf.write(out / f"{lang}.wav", normalize_loudness(audio), 24000)
        meta[lang] = {"file": f"{lang}.wav", "text": m["text"], "text_native": m.get("text_native", m["text"]),
                      "source_id": m["id"], "seconds": round(len(audio) / 24000, 2), "asr_cer": round(c, 1)}
        print(f"[speakers] {lang:24s} CER {c:5.1f}  {meta[lang]['seconds']:4.1f}s  {meta[lang]['text_native'][:60]}",
              flush=True)
    json.dump(meta, open(out / "speakers.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[speakers] wrote {len(meta)} speakers → {out}", flush=True)


if __name__ == "__main__":
    main()
