"""
make_card_samples.py — Audio samples for the model card, generated with the released model.

Held-out sentences (validation split, never trained on) for a showcase set of languages,
generated without a voice prompt with two seeds (two different voices).

    python scripts/make_card_samples.py --model <release dir> --val_manifest <workdir>/latents/val_universal.jsonl
Output: <release>/samples/<lang>_seed{0,1}.wav + samples.json
"""

import argparse
import json
import random
from pathlib import Path

from ghana_audiodit import GhanaTTS
from ghana_audiodit.text import split_sentences

SHOWCASE = ["Asante_Twi_twi", "Akuapem_Twi_twi", "Fante_fat", "Ewe_ewe", "Dagbani_dag", "Gonja_gjn",
            "Dangme_ada", "Hausa_hau", "English_eng"]
# The held-out English transcripts are noisy ASR-style turns; use a clean sentence instead.
OVERRIDES = {"English_eng": "Good evening, and welcome to the news from Accra. Here are the top stories of the day."}
CODESWITCH = ("Asante_Twi_twi", "Ɛnnɛ, Microsoft de Abilities for Jobs Program yi baeɛ, "
                                 "na ɛsɛ sɛ yɛn mmeranteɛ ne mmabaa bɔ mmɔden fa so.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--val_manifest", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.model) / "samples"
    out.mkdir(exist_ok=True)

    by_lang: dict[str, list[str]] = {}
    for line in open(args.val_manifest, encoding="utf-8"):
        m = json.loads(line)
        # whole transcripts, or single sentences of long ones (English turns run 170–280 chars)
        for t in [m.get("text_native", m["text"])] + split_sentences(m.get("text_native", m["text"]), 130):
            if 40 <= len(t) <= 130 and t[0].isupper() and not any(ch.isdigit() for ch in t):
                by_lang.setdefault(m["lang"], []).append(t)
    items = [(lang, OVERRIDES.get(lang) or random.Random(0).choice(sorted(set(by_lang[lang]))), lang)
             for lang in SHOWCASE]
    items.append((CODESWITCH[0], CODESWITCH[1], "Twi_English_codeswitch"))

    tts = GhanaTTS.from_pretrained(args.model)
    meta = []
    for lang, text, name in items:
        for k, seed in enumerate((args.seed, args.seed + 1)):
            r = tts.synthesize(text, language=lang, seed=seed)
            r.save(out / f"{name}_seed{k}.wav")
            meta.append({"name": name, "language": lang, "voice": k, "seed": seed, "text": text,
                         "file": f"samples/{name}_seed{k}.wav", "seconds": round(r.seconds, 1)})
            print(f"[card] {name:24s} seed {seed:4d} {r.seconds:4.1f}s  {text[:60]}", flush=True)
    (out / "samples.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
