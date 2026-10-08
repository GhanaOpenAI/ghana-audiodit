"""Text preparation: native orthography -> africa-g2p universal graphemes.

The model was trained on transcripts in africa-g2p's *universal* spelling (one shared
grapheme set across Ghanaian languages), so input text must be converted the same way.
English rows were kept in normal spelling, and English words inside Ghanaian-language
text are left as written by africa-g2p's English skip — which only works when the
optional ``pyspellchecker`` package is installed. Without it the skip switches off
silently and English is mangled (code -> chode), so this module refuses to run.
"""

from __future__ import annotations

import re

import africa_g2p.convert as _conv
from africa_g2p import UNIVERSAL, convert_lang

from .languages import LANGUAGES, resolve
from .utils import normalize_text

if _conv._spell is None:
    raise ImportError("africa-g2p English-word skip is off: `pip install pyspellchecker` "
                      "(otherwise English words in Ghanaian text are mangled, e.g. code -> chode)")


def to_universal(text: str, language: str) -> str:
    """Convert native-orthography ``text`` in ``language`` (key, ISO code or name) to universal."""
    code = LANGUAGES[resolve(language)]["code"]
    text = " ".join(text.split())
    if code == "eng":
        return text
    return re.sub(r"\s+", " ", convert_lang(text, code, UNIVERSAL)).strip()


def prepare(text: str, language: str, already_universal: bool = False) -> str:
    """Universal conversion + the model's own normalisation (lowercase, strip quotes)."""
    text = text if already_universal else to_universal(text, language)
    return normalize_text(text).strip()


def split_sentences(text: str, max_chars: int = 220) -> list[str]:
    """Split long input into chunks of whole sentences, each at most ~max_chars."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?;:])\s+", " ".join(text.split())) if s.strip()]
    chunks, cur = [], ""
    for s in sents:
        while len(s) > max_chars:                      # one very long sentence: break on commas/spaces
            cut = s.rfind(",", 0, max_chars)
            cut = cut if cut > max_chars // 3 else s.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(s[:cut + 1].strip())
            s = s[cut + 1:].strip()
        if cur and len(cur) + 1 + len(s) > max_chars:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks
