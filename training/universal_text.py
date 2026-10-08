"""
universal_text.py — Native orthography -> africa-g2p universal, with English words kept.

africa-g2p leaves English words (code, music, Python…) unconverted, but only when the
optional `pyspellchecker` package is importable; without it the skip switches off
silently and English is mangled (code -> chode, music -> musich). This module refuses
to run in that state, so every text-prep path gets the same behaviour.
"""

import africa_g2p.convert as _conv
from africa_g2p import UNIVERSAL, convert_lang

if _conv._spell is None:
    raise ImportError("africa-g2p English-word skip is OFF: `pip install pyspellchecker` "
                      "(otherwise English words in Ghanaian text get mangled, e.g. code -> chode)")

KEEP_NATIVE = {"eng"}   # English rows stay in normal spelling


def to_universal(text: str, code: str) -> str:
    if code in KEEP_NATIVE:
        return text
    return convert_lang(text, code, UNIVERSAL)
