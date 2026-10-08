"""GhanaTTS: text-to-speech for 43 Ghanaian languages (+ Ghanaian English).

Three voice modes:
  "noprompt"  no reference audio; the model picks a voice from the random starting
              noise (``seed`` makes it reproducible). Default.
  "speaker"   the language's built-in speaker (a training-set voice shipped with the model).
  "prompt"    your own reference audio (3–12 s) plus its transcript: the output clones that voice.

Long input is split into sentence chunks. In "noprompt" mode the first chunk's output is
reused as the voice prompt for the rest, so the whole passage keeps one voice.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from . import audiodit  # noqa: F401  (registers AudioDiT with transformers)
from .audio import load_prompt
from .audiodit import AudioDiTModel
from .languages import LANGUAGES, resolve
from .text import prepare, split_sentences

DEFAULT_REPO = "ghanaopenai/ghana-audiodit"
MODES = ("noprompt", "speaker", "prompt")


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
        self.max_frames = int(model.config.max_wav_duration * self.sample_rate // self.hop)
        self.speakers = json.loads((root / "speakers" / "speakers.json").read_text(encoding="utf-8"))
        self.rates = json.loads((root / "rates.json").read_text(encoding="utf-8"))   # latent frames per character

    @classmethod
    def from_pretrained(cls, repo_or_path: str = DEFAULT_REPO, device: str | None = None,
                        revision: str | None = None, dtype: str = "bfloat16") -> "GhanaTTS":
        """Load the model. On GPU the diffusion transformer runs in ``dtype`` ("bfloat16", as in
        training: ~4.5 GB VRAM; or "float32": ~6 GB). The text encoder stays fp32, the VAE fp16."""
        root = Path(repo_or_path)
        if not root.is_dir():
            root = Path(snapshot_download(repo_or_path, revision=revision))
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = AudioDiTModel.from_pretrained(str(root)).to(device)
        if device.startswith("cuda"):
            model.vae.to_half()
            model.transformer.to({"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype])
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
        return cls(model, tokenizer, root)

    # ── helpers ───────────────────────────────────────────────────────────
    @property
    def languages(self) -> dict[str, dict[str, str]]:
        return LANGUAGES

    def speaker_path(self, language: str) -> Path:
        return self.root / "speakers" / self.speakers[resolve(language)]["file"]

    def _frames(self, n_chars: int, rate: float, speed: float) -> int:
        return max(int(rate * n_chars / speed * 1.05), 12)

    @torch.no_grad()
    def _generate(self, text: str, frames: int, prompt: tuple[torch.Tensor, str] | None, steps: int,
                  cfg_strength: float, guidance_method: str) -> np.ndarray:
        if prompt is None:
            inputs = self.tokenizer([text], return_tensors="pt")
            kw = {}
            duration = min(frames, self.max_frames)
        else:
            wav, ptext = prompt
            inputs = self.tokenizer([f"{ptext} {text}"], return_tensors="pt")
            p_frames = -(-wav.shape[-1] // self.hop)
            kw = {"prompt_audio": wav.view(1, 1, -1)}
            duration = min(p_frames + frames, self.max_frames)
        out = self.model(input_ids=inputs.input_ids.to(self.device),
                         attention_mask=inputs.attention_mask.to(self.device),
                         duration=duration, steps=steps, cfg_strength=cfg_strength,
                         guidance_method=guidance_method, **kw)
        return out.waveform.squeeze().float().cpu().numpy()

    # ── public API ────────────────────────────────────────────────────────
    def synthesize(
        self,
        text: str,
        language: str = "Asante_Twi_twi",
        mode: str = "noprompt",
        prompt_audio: str | Path | tuple[np.ndarray, int] | None = None,
        prompt_text: str | None = None,
        prompt_language: str | None = None,
        seed: int | None = None,
        steps: int = 16,
        cfg_strength: float = 4.0,
        guidance_method: str = "apg",
        speed: float = 1.0,
        text_is_universal: bool = False,
        max_chunk_chars: int = 220,
        pause_seconds: float = 0.25,
    ) -> Synthesis:
        """Synthesise ``text`` (native spelling of ``language``; English words may be mixed in)."""
        language = resolve(language)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        seed = random.randrange(2**31) if seed is None else int(seed)
        rate = self.rates[language]

        prompt = None
        if mode == "speaker":
            spk = self.speakers[language]
            prompt = (load_prompt(self.speaker_path(language), self.sample_rate), spk["text"])
        elif mode == "prompt":
            if prompt_audio is None or not prompt_text:
                raise ValueError("mode='prompt' needs prompt_audio and prompt_text (its transcript)")
            ptext = prepare(prompt_text, resolve(prompt_language or language))
            prompt = (load_prompt(prompt_audio, self.sample_rate), ptext)
        if prompt is not None:
            # speak at the reference speaker's own pace
            n = max(len(prompt[1].replace(" ", "")), 1)
            rate = min(max(-(-prompt[0].shape[-1] // self.hop) / n, rate * 0.6), rate * 1.6)

        chunks = split_sentences(text, max_chunk_chars)
        if not chunks:
            raise ValueError("empty text")
        pieces, model_text = [], []
        gap = np.zeros(int(pause_seconds * self.sample_rate), dtype=np.float32)
        for i, chunk in enumerate(chunks):
            target = prepare(chunk, language, already_universal=text_is_universal)
            model_text.append(target)
            torch.manual_seed(seed + i)
            audio = self._generate(target, self._frames(len(target.replace(" ", "")), rate, speed),
                                   prompt, steps, cfg_strength, guidance_method)
            if mode == "noprompt" and i == 0 and len(chunks) > 1:
                # keep one voice across the passage: the first chunk becomes the prompt
                wav = torch.from_numpy(audio)
                wav = wav[: (wav.shape[-1] // self.hop) * self.hop]
                prompt = (wav, target)
            pieces += [audio, gap]
        return Synthesis(np.concatenate(pieces[:-1]).astype(np.float32), self.sample_rate, seed, model_text)
