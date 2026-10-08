"""Reference-audio handling for voice prompts."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch

TARGET_DBFS = -23.0          # training data was RMS-normalised to this level
MIN_PROMPT_S, MAX_PROMPT_S = 1.0, 15.0


def normalize_loudness(audio: np.ndarray) -> np.ndarray:
    """RMS-normalise to -23 dBFS, peak-limited to 0.95.

    Loud recordings give Wav-VAE latents several times the normal scale, which the model
    was never trained on; every training clip and every prompt goes through this.
    """
    audio = audio * (10 ** (TARGET_DBFS / 20) / (np.sqrt(np.mean(audio ** 2)) + 1e-9))
    return (audio / max(1.0, np.abs(audio).max() / 0.95)).astype(np.float32)


def load_prompt(src: str | Path | tuple[np.ndarray, int], sample_rate: int) -> torch.Tensor:
    """Load a reference clip as a 1-D float tensor at ``sample_rate``, loudness-normalised."""
    if isinstance(src, (str, Path)):
        audio, _ = librosa.load(str(src), sr=sample_rate, mono=True)
    else:
        audio, sr = src
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=0 if audio.shape[0] < audio.shape[-1] else 1)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    audio, _ = librosa.effects.trim(audio, top_db=40)
    sec = len(audio) / sample_rate
    if not MIN_PROMPT_S <= sec <= MAX_PROMPT_S:
        raise ValueError(f"reference audio must be {MIN_PROMPT_S:.0f}–{MAX_PROMPT_S:.0f} s of speech (got {sec:.1f} s)")
    if np.abs(audio).max() < 1e-4:
        raise ValueError("reference audio is silent")
    return torch.from_numpy(normalize_loudness(audio))
