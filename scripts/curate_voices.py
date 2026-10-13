"""
curate_voices.py — Curate diverse, high-quality prompt voices from ghanaopenai/twi-multispeaker-tts
and package them with descriptive acoustic profiles (e.g. bright_low, deep_excited, etc.).
"""

import io
import json
import os
import glob
import librosa
import numpy as np
import pyarrow.parquet as pq
import sherpa_onnx
import soundfile as sf
import torchaudio.transforms as T
import torch
from africa_g2p import UNIVERSAL, convert_lang

ASR_DIR = "/mnt/volume_d2wey28/models/sherpa/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-2025-11-12"
DATA_DIR = "/mnt/volume_d2wey28/projects/perfect-twi-tts/hf_upload/data"
OUT_DIR = "/mnt/volume_d2wey28/projects/longcat-audiodit-ghana/voices"
TARGET_RMS = 10 ** (-23 / 20)

def normalize_loudness(audio: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
    audio = audio * (TARGET_RMS / rms)
    peak = np.abs(audio).max()
    return (audio / max(1.0, peak / 0.95)).astype(np.float32)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=f"{ASR_DIR}/model.onnx",
        tokens=f"{ASR_DIR}/tokens.txt",
        num_threads=4,
        decoding_method="greedy_search"
    )

    shards = sorted(glob.glob(f"{DATA_DIR}/*.parquet"))
    print(f"Scanning {len(shards)} shards for voice profiles...")

    resampler = T.Resample(16000, 24000)

    # We want profiles:
    # 1. deep_calm (male, F0 ~100-120 Hz, calm/moderate energy)
    # 2. deep_excited (male, F0 ~125-145 Hz, dynamic energy)
    # 3. warm_storyteller (male/neutral, F0 ~135-160 Hz, measured pace)
    # 4. bright_low (female alto, F0 ~180-210 Hz, rich timbre)
    # 5. bright_clear (female soprano, F0 ~240-270 Hz, crisp articulation)
    # 6. lively_host (female, F0 ~220-250 Hz, upbeat broadcast tempo)

    profiles = {
        "deep_calm": {"min_f0": 95, "max_f0": 120, "found": None, "desc": "Deep, authoritative baritone voice with calm cadence"},
        "deep_excited": {"min_f0": 125, "max_f0": 145, "found": None, "desc": "Rich, dynamic baritone with energetic delivery"},
        "warm_storyteller": {"min_f0": 145, "max_f0": 175, "found": None, "desc": "Warm narrative timbre with clear, expressive pacing"},
        "bright_low": {"min_f0": 180, "max_f0": 215, "found": None, "desc": "Rich, grounded alto voice with natural resonance"},
        "lively_host": {"min_f0": 220, "max_f0": 245, "found": None, "desc": "Upbeat, energetic broadcast host with vibrant cadence"},
        "bright_clear": {"min_f0": 245, "max_f0": 285, "found": None, "desc": "Crisp, bright feminine tone with articulate clarity"}
    }

    for shard in shards:
        if all(p["found"] for p in profiles.values()):
            break
        table = pq.read_table(shard, columns=["id", "audio", "utmos", "duration"])
        ids = table.column("id").to_pylist()
        utmos_list = table.column("utmos").to_pylist()
        audios = table.column("audio")

        for idx, (cid, ut) in enumerate(zip(ids, utmos_list)):
            if ut < 2.5:
                continue
            audio_bytes = audios[idx].as_py()["bytes"]
            raw_wav, orig_sr = sf.read(io.BytesIO(audio_bytes))
            if raw_wav.ndim > 1:
                raw_wav = raw_wav.mean(axis=1)

            # Estimate pitch (F0) using pyin/yin
            try:
                f0, _, _ = librosa.pyin(raw_wav[:16000*8], fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=orig_sr)
                f0_clean = f0[~np.isnan(f0)]
                if len(f0_clean) < 15:
                    continue
                med_f0 = float(np.median(f0_clean))
            except Exception:
                continue

            # Check matching profile
            for name, prof in profiles.items():
                if prof["found"] is None and prof["min_f0"] <= med_f0 <= prof["max_f0"]:
                    # Take a clean 4.5s segment (e.g. from 0.5s to 5.0s)
                    seg_16k = raw_wav[int(0.5 * orig_sr) : int(5.5 * orig_sr)]
                    if len(seg_16k) < int(4.5 * orig_sr) or np.abs(seg_16k).max() < 0.05:
                        continue
                    
                    # Resample to 24 kHz
                    seg_24k = resampler(torch.from_numpy(seg_16k).float().unsqueeze(0)).squeeze(0).numpy()
                    seg_norm = normalize_loudness(seg_24k)

                    # Transcribe segment with omniASR
                    stream = rec.create_stream()
                    stream.accept_waveform(24000, seg_norm)
                    rec.decode_stream(stream)
                    transcript = (stream.result.text or "").strip()
                    if len(transcript.split()) < 3:
                        continue

                    # Transcribe / universal
                    transcript_uni = convert_lang(transcript, "twi", UNIVERSAL)

                    prof["found"] = {
                        "name": name,
                        "display_name": name.replace("_", " ").title(),
                        "id": cid,
                        "median_f0_hz": round(med_f0, 1),
                        "utmos": round(float(ut), 2),
                        "text": transcript,
                        "text_universal": transcript_uni,
                        "audio": seg_norm,
                        "desc": prof["desc"]
                    }
                    print(f"Matched [{name}]: {cid} (F0={med_f0:.1f} Hz, UTMOS={ut:.2f}) -> {transcript}")
                    break

    voice_meta = []
    for name, prof in profiles.items():
        data = prof["found"]
        if not data:
            print(f"Warning: profile {name} not found!")
            continue
        wav_path = os.path.join(OUT_DIR, f"{name}.wav")
        txt_path = os.path.join(OUT_DIR, f"{name}.txt")
        sf.write(wav_path, data["audio"], 24000)
        with open(txt_path, "w", encoding="utf-8") as tf:
            tf.write(data["text_universal"])

        voice_meta.append({
            "id": name,
            "display_name": data["display_name"],
            "description": data["desc"],
            "pitch_hz": data["median_f0_hz"],
            "utmos": data["utmos"],
            "source_clip_id": data["id"],
            "prompt_text": data["text"],
            "prompt_text_universal": data["text_universal"],
            "file": f"{name}.wav"
        })

    json_path = os.path.join(OUT_DIR, "voices.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(voice_meta, jf, indent=2, ensure_ascii=False)
    print(f"\nSuccessfully curated and saved {len(voice_meta)} voices to {OUT_DIR}")

if __name__ == "__main__":
    main()
