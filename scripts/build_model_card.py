"""
build_model_card.py — Fill release/README.md (template) with the sample-audio and language tables.

    python scripts/build_model_card.py --release <release dir> [--repo ghanaopenai/ghana-audiodit]
Writes <release>/README.md.
"""

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
from ghana_audiodit.languages import LANGUAGES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--repo", default="ghanaopenai/ghana-audiodit")
    args = ap.parse_args()
    rel = Path(args.release)
    base = f"https://huggingface.co/{args.repo}/resolve/main"

    samples = json.loads((rel / "samples" / "samples.json").read_text(encoding="utf-8"))
    by_name: dict[str, dict] = {}
    for s in samples:
        by_name.setdefault(s["name"], {"text": s["text"], "language": s["language"]})[s["mode"]] = s["file"]
    audio = lambda f: f'<audio controls preload="none" src="{base}/{f}"></audio>'
    rows = []
    for name, s in by_name.items():
        label = "Twi + English (code-switching)" if "codeswitch" in name else LANGUAGES[s["language"]]["name"]
        note = " *(written for this page)*" if s["language"] == "English_eng" or "codeswitch" in name else ""
        rows.append(f"| {label} | {html.escape(s['text'])}{note} | {audio(s['noprompt'])} | {audio(s['speaker'])} |")

    speakers = json.loads((rel / "speakers" / "speakers.json").read_text(encoding="utf-8"))
    lang_rows = ["| Language | `language=` | ISO 639-3 | Built-in speaker |", "| --- | --- | --- | --- |"]
    for k, v in sorted(LANGUAGES.items(), key=lambda kv: kv[1]["name"]):
        spk = speakers.get(k)
        lang_rows.append(f"| {v['name']} | `{k}` | `{v['code']}` | "
                         + (f"[{spk['seconds']:.1f} s]({base}/speakers/{spk['file']})" if spk else "—") + " |")

    card = (ROOT / "release" / "README.md").read_text(encoding="utf-8")
    card = card.replace("<!-- SAMPLES -->", "\n".join(rows)).replace("<!-- LANGUAGES -->", "\n".join(lang_rows))
    (rel / "README.md").write_text(card, encoding="utf-8")
    print(f"[card] {len(rows)} sample rows, {len(lang_rows) - 2} languages → {rel / 'README.md'}")


if __name__ == "__main__":
    main()
