"""
cache_latents.py — Pre-encode the Ghana 3h/lang speech set into Wav-VAE latents.

Input layout (fish-s2-ghana/data10r, plus data/ for languages missing there):
    <data_root>/<split>/<Lang_code>/<id>.wav   16 kHz mono
    <data_root>/<split>/<Lang_code>/<id>.lab   native-orthography transcript

Output:
    <out_dir>/<split>/<Lang_code>/<id>.pt      fp16 latent [T_lat, 64]
    <out_dir>/<split>.jsonl                    {"lang","id","text","latent","frames"}

Each clip is resampled to 24 kHz, RMS-normalised to -23 dBFS (peak-limited to
0.95) and encoded on its own via
model.encode_prompt_audio (same padding/off-frame trim as inference), so no
batch zero-padding leaks into the non-causal VAE boundary frames.

Usage:
    python train/cache_latents.py --data_root .../fish-s2-ghana/data10r .../fish-s2-ghana/data \
        --out_dir .../latents
    A language is taken from the first root that has it.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import ghana_audiodit.audiodit  # noqa: F401  (registers AudioDiT with transformers)
from ghana_audiodit.audiodit import AudioDiTModel

SPLITS = {"raw": "train", "val_raw": "val"}
TARGET_RMS = 10 ** (-23 / 20)


def normalize_loudness(audio: np.ndarray) -> np.ndarray:
    """Loud recordings (-13..-15 dB RMS) give Wav-VAE latents 3-4x the normal
    scale and base-model CFM loss ~10 instead of ~1; -23 dBFS fixes both."""
    audio = audio * (TARGET_RMS / (np.sqrt(np.mean(audio ** 2)) + 1e-9))
    return (audio / max(1.0, np.abs(audio).max() / 0.95)).astype(np.float32)


def load_clip(wav_path: Path, sr: int, clip_peak: float):
    lab = wav_path.with_suffix(".lab")
    if not lab.exists():
        return None, "no_lab"
    text = " ".join(lab.read_text(encoding="utf-8").split())
    if not text:
        return None, "empty_text"
    audio, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    peak = float(abs(audio).max()) if audio.size else 0.0
    if peak >= clip_peak:
        return None, "clipped"
    if peak < 1e-4:
        return None, "silent"
    return (torch.from_numpy(normalize_loudness(audio)), text), None


class _ClipLoader(Dataset):
    """Decoding/resampling in worker processes (threads starve on the GIL)."""

    def __init__(self, wavs, sr, clip_peak):
        self.wavs, self.sr, self.clip_peak = wavs, sr, clip_peak

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, i):
        torch.set_num_threads(1)
        return (self.wavs[i], *load_clip(self.wavs[i], self.sr, self.clip_peak))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_dir", default="meituan-longcat/LongCat-AudioDiT-1B")
    ap.add_argument("--clip_peak", type=float, default=0.99)
    ap.add_argument("--min_sec", type=float, default=1.0)
    ap.add_argument("--max_sec", type=float, default=20.0)
    ap.add_argument("--num_workers", type=int, default=10)
    ap.add_argument("--limit_per_lang", type=int, default=0, help="debug: only N clips per language")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = AudioDiTModel.from_pretrained(args.model_dir).to(device)
    model.vae.to_half()
    model.eval()
    sr = model.config.sampling_rate

    roots, out_dir = [Path(r) for r in args.data_root], Path(args.out_dir)

    for src_split, dst_split in SPLITS.items():
        wavs, seen = [], set()
        lang_dirs = [d for r in roots if (r / src_split).is_dir() for d in sorted((r / src_split).iterdir())]
        for lang_dir in lang_dirs:
            if lang_dir.name in seen:
                continue
            seen.add(lang_dir.name)
            fs = sorted(lang_dir.glob("*.wav"))
            if args.limit_per_lang:
                fs = fs[: args.limit_per_lang]
            wavs += fs

        manifest = out_dir / f"{dst_split}.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        skipped: dict[str, int] = {}
        n_ok, total_sec = 0, 0.0

        with open(manifest, "w", encoding="utf-8") as mf:
            loader = DataLoader(_ClipLoader(wavs, sr, args.clip_peak), batch_size=None,
                                num_workers=args.num_workers, prefetch_factor=8)
            for wav_path, item, why in tqdm(loader, total=len(wavs), desc=dst_split, mininterval=30):
                if item is None:
                    skipped[why] = skipped.get(why, 0) + 1
                    continue
                audio, text = item
                sec = audio.shape[-1] / sr
                if sec < args.min_sec or sec > args.max_sec:
                    skipped["duration"] = skipped.get("duration", 0) + 1
                    continue

                lang = wav_path.parent.name
                out_pt = out_dir.absolute() / dst_split / lang / f"{wav_path.stem}.pt"
                if not out_pt.exists():
                    with torch.no_grad():
                        z, _ = model.encode_prompt_audio(audio.view(1, 1, -1))
                    out_pt.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(z[0].to(torch.float16).cpu().clone(), out_pt)
                    frames = z.shape[1]
                else:
                    frames = torch.load(out_pt, weights_only=True).shape[0]

                mf.write(json.dumps({
                    "lang": lang, "id": wav_path.stem, "text": text,
                    "latent": str(out_pt), "frames": frames, "wav": str(wav_path.absolute()),
                }, ensure_ascii=False) + "\n")
                n_ok += 1
                total_sec += sec

        print(f"[{dst_split}] {len(seen)} languages", flush=True)
        print(f"[{dst_split}] kept {n_ok} clips, {total_sec/3600:.1f} h; skipped {skipped}", flush=True)


if __name__ == "__main__":
    main()
