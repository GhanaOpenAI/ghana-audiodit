"""GhanaTTS: text-to-speech for 43 Ghanaian languages (+ Ghanaian English).

The model speaks without reference audio: the voice comes from the random starting noise,
and ``seed`` makes it reproducible. Voice cloning (from your own recordings or from built-in
speakers) is deliberately not offered — the model was trained mostly without voice prompts
(85 % of examples) and prompted generation was not reliable enough to ship.

Long input is split into sentence chunks of at most ~18 s of speech (training clips were
≤ 20 s). Every chunk uses the same seed, which keeps the voice as similar as possible,
but a long passage may still change voice between chunks.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from . import audiodit  # noqa: F401  (registers AudioDiT with transformers)
from .audiodit import AudioDiTModel
from .languages import LANGUAGES, resolve
from .text import prepare, split_sentences

DEFAULT_REPO = "ghanaopenai/ghana-audiodit"
CHUNK_SECONDS = 18.0


@dataclass
class Synthesis:
    audio: np.ndarray                 # float32 mono
    sample_rate: int
    seed: int
    model_text: list[str] = field(default_factory=list)   # universal-spelling text the model read, per chunk

    @property
    def seconds(self) -> float:
        return len(self.audio) / self.sample_rate

    def save(self, path: str | Path) -> None:
        import soundfile as sf
        sf.write(str(path), self.audio, self.sample_rate)


class GhanaTTS:
    def __init__(self, model: AudioDiTModel, tokenizer, root: Path):
        self.model, self.tokenizer, self.root = model, tokenizer, root
        self.device = next(model.parameters()).device
        self.sample_rate = model.config.sampling_rate
        self.hop = model.config.latent_hop
        self.frame_rate = self.sample_rate / self.hop
        self.max_frames = int(model.config.max_wav_duration * self.frame_rate)
        self.rates = json.loads((root / "rates.json").read_text(encoding="utf-8"))   # latent frames per character

    @classmethod
    def from_pretrained(cls, repo_or_path: str = DEFAULT_REPO, device: str | None = None,
                        revision: str | None = None, dtype: str = "bfloat16") -> "GhanaTTS":
        """Load the model. On GPU the diffusion transformer runs in ``dtype`` ("bfloat16", as in
        training: ~4 GB VRAM; or "float32": ~6 GB). The text encoder stays fp32, the VAE fp16."""
        root = Path(repo_or_path)
        if not root.is_dir():
            root = Path(snapshot_download(repo_or_path, revision=revision,
                                          allow_patterns=["*.json", "*.safetensors", "*.py"]))
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = AudioDiTModel.from_pretrained(str(root)).to(device)
        if device.startswith("cuda"):
            model.vae.to_half()
            model.transformer.to({"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype])
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
        return cls(model, tokenizer, root)

    @property
    def languages(self) -> dict[str, dict[str, str]]:
        return LANGUAGES

    @torch.no_grad()
    def _generate(self, text: str, frames: int, steps: int, cfg_strength: float, guidance_method: str) -> np.ndarray:
        inputs = self.tokenizer([text], return_tensors="pt")
        out = self.model(input_ids=inputs.input_ids.to(self.device),
                         attention_mask=inputs.attention_mask.to(self.device),
                         duration=min(frames, self.max_frames), steps=steps, cfg_strength=cfg_strength,
                         guidance_method=guidance_method)
        return out.waveform.squeeze().float().cpu().numpy()

    def synthesize(
        self,
        text: str,
        language: str = "Asante_Twi_twi",
        seed: int | None = None,
        steps: int = 16,
        cfg_strength: float = 4.0,
        guidance_method: str = "apg",
        speed: float = 1.0,
        text_is_universal: bool = False,
        pause_seconds: float = 0.25,
    ) -> Synthesis:
        """Synthesise ``text`` (normal spelling of ``language``; English words may be mixed in).

        ``seed`` picks the voice (random if None; the one used is returned on the result).
        ``speed`` > 1 speaks faster. ``steps`` trades speed for slightly cleaner audio.
        """
        language = resolve(language)
        seed = random.randrange(2**31) if seed is None else int(seed)
        rate = self.rates[language]
        max_chars = max(60, int(CHUNK_SECONDS * self.frame_rate / rate))

        chunks = split_sentences(text, max_chars)
        if not chunks:
            raise ValueError("empty text")
        pieces, model_text = [], []
        gap = np.zeros(int(pause_seconds * self.sample_rate), dtype=np.float32)
        for chunk in chunks:
            target = prepare(chunk, language, already_universal=text_is_universal)
            model_text.append(target)
            torch.manual_seed(seed)                       # same seed per chunk: keep the voice close
            frames = max(int(rate * len(target.replace(" ", "")) / speed * 1.05), 12)
            pieces += [self._generate(target, frames, steps, cfg_strength, guidance_method), gap]
        return Synthesis(np.concatenate(pieces[:-1]).astype(np.float32), self.sample_rate, seed, model_text)
