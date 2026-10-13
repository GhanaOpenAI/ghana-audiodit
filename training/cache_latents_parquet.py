"""
cache_latents_parquet.py — Pre-encode parquet-based TTS datasets (e.g. ghanaopenai/twi-multispeaker-tts)
into Wav-VAE latents for AudioDiT training.

Reads Parquet shards containing audio bytes, UTMOS scores, and transcripts:
  1. Filters by minimum UTMOS quality (e.g. >= 2.0) and duration.
  2. Decodes FLAC audio bytes, resamples to 24 kHz (model SR), and RMS-normalises to -23 dBFS.
  3. Pre-encodes audio into Wav-VAE fp16 latents [T_lat, 64] using low-VRAM encoder.
  4. Converts transcripts to africa-g2p universal spelling (with English skip).
  5. Splits into train and held-out validation sets, saving val .wavs for sample listening.
  6. Outputs manifests: train.jsonl, val.jsonl, train_universal.jsonl, val_universal.jsonl.

Usage:
    python training/cache_latents_parquet.py \
        --parquet_dir /path/to/parquets \
        --out_dir /path/to/workdir/latents \
        --text_column transcription \
        --min_utmos 2.0
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from tqdm import tqdm

import ghana_audiodit.audiodit  # noqa: F401
from ghana_audiodit.audiodit import AudioDiTModel
from universal_text import to_universal

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("cache_latents_parquet")

TARGET_RMS = 10 ** (-23 / 20)  # -23 dBFS standard


def normalize_loudness(audio: np.ndarray) -> np.ndarray:
    """Loud recordings give Wav-VAE latents 3-4x normal scale; -23 dBFS fixes it."""
    rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
    audio = audio * (TARGET_RMS / rms)
    peak = np.abs(audio).max()
    return (audio / max(1.0, peak / 0.95)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Cache Wav-VAE latents from Parquet dataset")
    ap.add_argument("--parquet_dir", help="Directory containing .parquet shard files")
    ap.add_argument("--parquet_pattern", default="*.parquet", help="Glob pattern for parquet files")
    ap.add_argument("--dataset_name", default="ghanaopenai/twi-multispeaker-tts",
                    help="HuggingFace dataset name if downloading directly")
    ap.add_argument("--out_dir", required=True, help="Directory to save latents and manifests")
    ap.add_argument("--model_dir", default="meituan-longcat/LongCat-AudioDiT-1B",
                    help="AudioDiT model repo or path containing Wav-VAE")
    ap.add_argument("--text_column", default="transcription",
                    help="Column name for transcript (e.g. 'transcription' or 'griot_transcription')")
    ap.add_argument("--lang", default="Asante_Twi_twi", help="Dataset language folder name")
    ap.add_argument("--lang_code", default="twi", help="africa-g2p language code")
    ap.add_argument("--min_utmos", type=float, default=2.0, help="Minimum UTMOS quality score")
    ap.add_argument("--min_sec", type=float, default=2.5, help="Minimum audio duration in seconds")
    ap.add_argument("--max_sec", type=float, default=30.5, help="Maximum audio duration in seconds")
    ap.add_argument("--clip_peak", type=float, default=1.02, help="Drop clips exceeding peak threshold")
    ap.add_argument("--val_count", type=int, default=150, help="Number of held-out clips for validation")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of processed clips (debug)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_wav_dir = out_dir / "val_wavs"
    val_wav_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover parquet files
    parquet_files: list[Path] = []
    if args.parquet_dir:
        p_dir = Path(args.parquet_dir)
        parquet_files = sorted(p_dir.glob(args.parquet_pattern))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files matching {args.parquet_pattern} in {p_dir}")
        log.info(f"Found {len(parquet_files)} parquet shards in {p_dir}")
    else:
        log.info(f"Downloading/loading dataset from Hugging Face: {args.dataset_name}")
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        remote_files = fs.glob(f"datasets/{args.dataset_name}/data/*.parquet")
        log.info(f"Found {len(remote_files)} shards on Hugging Face")
        from huggingface_hub import hf_hub_download
        for rf in remote_files:
            fname = Path(rf).name
            local_path = hf_hub_download(repo_id=args.dataset_name, repo_type="dataset", filename=f"data/{fname}")
            parquet_files.append(Path(local_path))

    # 2. Load low-VRAM VAE on device
    log.info(f"Loading AudioDiT VAE from {args.model_dir} on {args.device}...")
    base_model = AudioDiTModel.from_pretrained(args.model_dir)
    config = base_model.config
    vae = base_model.vae.to_half().to(args.device)
    vae.eval()
    sr = config.sampling_rate  # 24000
    hop = config.latent_hop    # 2048
    del base_model
    gc.collect()
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    log.info(f"VAE loaded. Target SR={sr}, hop={hop}. GPU VRAM={torch.cuda.memory_allocated() / 1e6:.1f} MB")

    # 3. Read metadata from parquets and filter candidates
    log.info(f"Scanning parquet shards for candidates (min_utmos={args.min_utmos})...")
    candidates = []
    columns_to_read = ["id", "duration", "utmos", args.text_column]

    for shard_idx, shard_path in enumerate(parquet_files):
        table = pq.read_table(shard_path, columns=columns_to_read)
        ids = table.column("id").to_pylist()
        durs = table.column("duration").to_pylist()
        utmos_list = table.column("utmos").to_pylist()
        texts = table.column(args.text_column).to_pylist()

        for row_idx, (cid, dur, ut, txt) in enumerate(zip(ids, durs, utmos_list, texts)):
            if ut < args.min_utmos:
                continue
            if dur < args.min_sec or dur > args.max_sec:
                continue
            txt_clean = " ".join(str(txt or "").split())
            if len(txt_clean.replace(" ", "")) < 4:
                continue
            candidates.append({
                "shard_path": str(shard_path),
                "row_idx": row_idx,
                "id": cid,
                "duration": dur,
                "utmos": float(ut),
                "text_native": txt_clean,
            })
            if args.limit and len(candidates) >= args.limit:
                break
        if args.limit and len(candidates) >= args.limit:
            break

    total_candidates = len(candidates)
    log.info(f"Found {total_candidates} candidate clips meeting filters (total {total_candidates * 30 / 3600:.1f} hours)")
    if total_candidates == 0:
        raise RuntimeError("No candidate clips found with current filter settings!")

    # Deterministic train / val split
    random.shuffle(candidates)
    val_count = min(args.val_count, max(1, int(total_candidates * 0.05)))
    val_items = candidates[:val_count]
    train_items = candidates[val_count:]
    log.info(f"Split: {len(train_items)} train clips, {len(val_items)} validation clips")

    # 4. Process clips shard by shard to minimize I/O
    resampler_cache = {}

    def get_resampler(orig_sr: int) -> T.Resample:
        if orig_sr not in resampler_cache:
            resampler_cache[orig_sr] = T.Resample(orig_sr, sr)
        return resampler_cache[orig_sr]

    def encode_latents(audio_24k: np.ndarray) -> tuple[torch.Tensor, int]:
        """Encode 24 kHz normalized audio to fp16 latents [T, 64]."""
        wav = torch.from_numpy(audio_24k).to(args.device, torch.float16).view(1, 1, -1)
        full_hop = hop
        off = 3
        if wav.shape[-1] % full_hop != 0:
            wav = F.pad(wav, (0, full_hop - wav.shape[-1] % full_hop))
        wav = F.pad(wav, (0, full_hop * off))
        with torch.no_grad():
            latent = vae.encode(wav)
        if off != 0:
            latent = latent[..., :-off]
        z = latent.permute(0, 2, 1)  # [1, T, 64]
        return z[0].cpu().clone(), z.shape[1]

    # Map items by shard
    for split_name, split_items in [("val", val_items), ("train", train_items)]:
        items_by_shard: dict[str, list[dict]] = {}
        for it in split_items:
            items_by_shard.setdefault(it["shard_path"], []).append(it)

        manifest_native_path = out_dir / f"{split_name}.jsonl"
        manifest_univ_path = out_dir / f"{split_name}_universal.jsonl"
        log.info(f"[{split_name}] Encoding latents for {len(split_items)} clips...")

        skipped = {}
        n_ok = 0
        total_audio_sec = 0.0

        with open(manifest_native_path, "w", encoding="utf-8") as f_nat, \
             open(manifest_univ_path, "w", encoding="utf-8") as f_uni:

            for shard_path, shard_entries in tqdm(items_by_shard.items(), desc=f"Shards ({split_name})"):
                table = pq.read_table(shard_path, columns=["id", "audio", args.text_column])
                audio_col = table.column("audio")

                for entry in shard_entries:
                    r_idx = entry["row_idx"]
                    cid = entry["id"]
                    text_nat = entry["text_native"]

                    # Universal conversion
                    try:
                        text_uni = to_universal(text_nat, args.lang_code)
                    except Exception as e:
                        skipped["g2p_error"] = skipped.get("g2p_error", 0) + 1
                        continue

                    if len(text_uni.replace(" ", "")) < 4:
                        skipped["too_short_uni"] = skipped.get("too_short_uni", 0) + 1
                        continue

                    # Decode audio
                    audio_struct = audio_col[r_idx].as_py()
                    audio_bytes = audio_struct.get("bytes")
                    if not audio_bytes:
                        skipped["no_bytes"] = skipped.get("no_bytes", 0) + 1
                        continue

                    try:
                        raw_wav, orig_sr = sf.read(io.BytesIO(audio_bytes))
                    except Exception:
                        skipped["audio_decode_err"] = skipped.get("audio_decode_err", 0) + 1
                        continue

                    if raw_wav.ndim > 1:
                        raw_wav = raw_wav.mean(axis=1)

                    peak = float(np.abs(raw_wav).max()) if raw_wav.size else 0.0
                    if peak >= args.clip_peak:
                        skipped["clipped"] = skipped.get("clipped", 0) + 1
                        continue
                    if peak < 1e-4:
                        skipped["silent"] = skipped.get("silent", 0) + 1
                        continue

                    # Resample to 24 kHz
                    if orig_sr != sr:
                        resampler = get_resampler(orig_sr)
                        wav_tensor = resampler(torch.from_numpy(raw_wav).float().unsqueeze(0)).squeeze(0)
                        audio_24k = wav_tensor.numpy()
                    else:
                        audio_24k = raw_wav.astype(np.float32)

                    # Loudness normalisation
                    audio_norm = normalize_loudness(audio_24k)
                    dur_sec = len(audio_norm) / sr

                    # Latent output path
                    out_pt = out_dir / split_name / args.lang / f"{cid}.pt"
                    out_pt.parent.mkdir(parents=True, exist_ok=True)

                    if not out_pt.exists():
                        z, frames = encode_latents(audio_norm)
                        torch.save(z.to(torch.float16), out_pt)
                    else:
                        frames = torch.load(out_pt, weights_only=True).shape[0]

                    val_wav_file = ""
                    if split_name == "val":
                        val_wav_path = val_wav_dir / f"{cid}.wav"
                        if not val_wav_path.exists():
                            sf.write(str(val_wav_path), audio_norm, sr)
                        val_wav_file = str(val_wav_path.resolve())

                    rec_nat = {
                        "lang": args.lang,
                        "id": cid,
                        "text": text_nat,
                        "latent": str(out_pt.resolve()),
                        "frames": frames,
                        "wav": val_wav_file,
                        "utmos": entry["utmos"],
                    }
                    rec_uni = {
                        "lang": args.lang,
                        "id": cid,
                        "text": text_uni,
                        "text_native": text_nat,
                        "latent": str(out_pt.resolve()),
                        "frames": frames,
                        "wav": val_wav_file,
                        "utmos": entry["utmos"],
                    }

                    f_nat.write(json.dumps(rec_nat, ensure_ascii=False) + "\n")
                    f_uni.write(json.dumps(rec_uni, ensure_ascii=False) + "\n")
                    n_ok += 1
                    total_audio_sec += dur_sec

            log.info(f"[{split_name}] Completed {n_ok} clips, {total_audio_sec / 3600:.1f} h. Skipped: {skipped}")

    log.info(f"Latent caching and manifest creation complete in {out_dir}")


if __name__ == "__main__":
    main()
