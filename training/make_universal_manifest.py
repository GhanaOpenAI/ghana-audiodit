"""
make_universal_manifest.py — Rewrite cached-latent manifests with africa-g2p UNIVERSAL text.

The audio (and so the cached latents) is unchanged; only the transcript is
converted from each language's native orthography into the shared universal
grapheme set via universal_text.to_universal: English rows are left in normal
spelling, and English words inside Ghanaian-language text are kept as written
(africa-g2p's English skip; universal_text refuses to run if it is off).

    latents/train.jsonl -> latents/train_universal.jsonl
    latents/val.jsonl   -> latents/val_universal.jsonl

Each row keeps the native transcript as "text_native".

Usage:
    python train/make_universal_manifest.py --latent_dir .../latents
"""

import argparse
import collections
import importlib.metadata
import json
import random
import re
from pathlib import Path

from universal_text import to_universal as _to_universal


def to_universal(text: str, code: str) -> str:
    return re.sub(r"\s+", " ", _to_universal(text, code)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_dir", required=True)
    ap.add_argument("--min_chars", type=int, default=4)
    ap.add_argument("--suffix", default="universal_enskip",
                    help="writes <split>_<suffix>.jsonl (never overwrite the files a running job uses)")
    args = ap.parse_args()
    d = Path(args.latent_dir)
    print(f"africa-g2p {importlib.metadata.version('africa-g2p')}", flush=True)

    for split in ("train", "val"):
        src, dst = d / f"{split}.jsonl", d / f"{split}_{args.suffix}.jsonl"
        stats = collections.Counter()
        examples: dict[str, tuple[str, str]] = {}
        with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
            for line in fi:
                m = json.loads(line)
                code = m["lang"].rsplit("_", 1)[-1]
                try:
                    u = to_universal(m["text"], code)
                except Exception:
                    stats["error"] += 1
                    continue
                if len(u.replace(" ", "")) < args.min_chars:
                    stats["too_short"] += 1
                    continue
                m["text_native"], m["text"] = m["text"], u
                fo.write(json.dumps(m, ensure_ascii=False) + "\n")
                stats["ok"] += 1
                examples.setdefault(m["lang"], (m["text_native"], u))
        print(f"[{split}] {dict(stats)} → {dst}", flush=True)

    # character inventory before/after, so the shrink in symbol set is visible
    nat, uni = collections.Counter(), collections.Counter()
    for line in open(d / f"train_{args.suffix}.jsonl", encoding="utf-8"):
        m = json.loads(line)
        nat.update(m["text_native"].lower())
        uni.update(m["text"].lower())
    print(f"distinct chars: native={len(nat)}  universal={len(uni)}")
    print("non-ASCII left in universal:", "".join(sorted(c for c in uni if ord(c) > 127)))
    for lang, (n, u) in sorted(random.Random(0).sample(sorted(examples.items()), 6)):
        print(f"  {lang:24s} {n[:60]!r}\n  {'':24s} {u[:60]!r}")


if __name__ == "__main__":
    main()
