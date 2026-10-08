"""ghana-audiodit: text-to-speech for 43 Ghanaian languages, fine-tuned from LongCat-AudioDiT-1B.

    from ghana_audiodit import GhanaTTS
    tts = GhanaTTS.from_pretrained()                    # ghanaopenai/ghana-audiodit
    tts.synthesize("Akwaaba, wo ho te sɛn?", language="Asante_Twi_twi").save("out.wav")
"""

from .languages import LANGUAGES, resolve
from .text import prepare, to_universal
from .tts import DEFAULT_REPO, GhanaTTS, Synthesis

__all__ = ["GhanaTTS", "Synthesis", "LANGUAGES", "DEFAULT_REPO", "resolve", "prepare", "to_universal"]
__version__ = "0.1.0"
