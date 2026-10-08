"""
dataset_ghana.py — Map-style dataset over cached Wav-VAE latents (see cache_latents.py).

Each item is one full clip. The prompt/generation boundary is sampled per item
(as a fraction of the clip, clamped by min/max seconds), mirroring the random
context mask used in the paper: the model sees the prompt latent as clean
context and learns to generate the rest. With probability p_no_prompt the
prompt is empty and the whole clip is generated from text alone (no-prompt TTS).
The text is the clip's full transcript, matching inference where
text = prompt_text + " " + gen_text.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ghana_audiodit.utils import normalize_text

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

FRAME_RATE = 24000 / 2048  # 11.72 latent frames per second


class GhanaLatentDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        min_audio_sec: float = 2.5,
        max_audio_sec: float = 20.0,
        prompt_min_sec: float = 0.8,
        prompt_max_sec: float = 10.0,
        min_gen_sec: float = 1.0,
        prompt_frac_lo: float = 0.2,
        prompt_frac_hi: float = 0.7,
        min_text_chars: int = 4,
        langs: list[str] | None = None,
        p_no_prompt: float = 0.0,
    ):
        self.p_no_prompt = p_no_prompt
        self.prompt_min = max(1, round(prompt_min_sec * FRAME_RATE))
        self.prompt_max = round(prompt_max_sec * FRAME_RATE)
        self.min_gen = max(1, round(min_gen_sec * FRAME_RATE))
        self.max_frames = round(max_audio_sec * FRAME_RATE)
        self.frac_lo, self.frac_hi = prompt_frac_lo, prompt_frac_hi

        min_frames = max(round(min_audio_sec * FRAME_RATE), self.prompt_min + self.min_gen)
        self.items = []
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                m = json.loads(line)
                if langs and m["lang"] not in langs:
                    continue
                if m["frames"] < min_frames:
                    continue
                if len(m["text"].replace(" ", "")) < min_text_chars:
                    continue
                self.items.append(m)
        print(f"[GhanaLatentDataset] {manifest}: {len(self.items)} clips, "
              f"{len({m['lang'] for m in self.items})} languages", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        m = self.items[idx]
        z = torch.load(m["latent"], weights_only=True).float()  # [T, 64]
        # Clips are capped at max_audio_sec in caching; truncation here would
        # desync text and audio, so it is only a safety net.
        z = z[: self.max_frames]
        T = z.shape[0]

        if self.p_no_prompt and random.random() < self.p_no_prompt:
            return {"latent": z, "text": m["text"], "prompt_frames": 0, "lang": m["lang"]}
        p = int(T * random.uniform(self.frac_lo, self.frac_hi))
        p = min(p, T - self.min_gen, self.prompt_max)
        p = max(p, self.prompt_min)
        return {"latent": z, "text": m["text"], "prompt_frames": p, "lang": m["lang"]}


def ghana_collate_fn(batch, tokenizer, max_text_len: int = 512):
    lens = torch.tensor([b["latent"].shape[0] for b in batch], dtype=torch.long)
    T = int(lens.max())
    latents = torch.stack([F.pad(b["latent"], (0, 0, 0, T - b["latent"].shape[0])) for b in batch])
    enc = tokenizer(
        [normalize_text(b["text"]) for b in batch],
        padding="longest", truncation=True, max_length=max_text_len, return_tensors="pt",
    )
    return {
        "latent": latents,                                   # [B, T, 64]
        "latent_lens": lens,                                 # [B]
        "prompt_frames": torch.tensor([b["prompt_frames"] for b in batch], dtype=torch.long),
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "text": [b["text"] for b in batch],
        "lang": [b["lang"] for b in batch],
    }
