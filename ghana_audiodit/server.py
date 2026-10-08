"""
Inference API for ghana-audiodit (powers the Hugging Face Space).

    pip install "ghana-audiodit[server] @ git+https://github.com/GhanaOpenAI/ghana-audiodit"
    uvicorn ghana_audiodit.server:app --host 0.0.0.0 --port 8210

Environment
    GHANA_AUDIODIT_MODEL   model repo or local folder (default ghanaopenai/ghana-audiodit)
    GHANA_AUDIODIT_DTYPE   bfloat16 (default, ~4.5 GB VRAM) or float32 (~6 GB)
    OMNIASR_SHERPA_DIR     local omniASR sherpa-onnx model folder, or
    OMNIASR_SHERPA_REPO    a Hugging Face repo to download it from; with either set, reference
                           audio uploaded without a transcript is transcribed automatically
    MAX_CHARS, MAX_QUEUE   request limits (600 characters, 8 waiting requests)

Endpoints
    GET  /health                    model + queue status
    GET  /languages                 the 43 languages and their built-in speakers
    GET  /speakers/{language}.wav   preview a built-in speaker
    POST /synthesize                multipart form -> audio/wav
         text            required, native spelling (English words may be mixed in), ≤ MAX_CHARS
         language        key / ISO code / name, default Asante_Twi_twi
         mode            noprompt (default) | speaker | prompt
         prompt_audio    file, required for mode=prompt (1–15 s of speech)
         prompt_text     its transcript; if empty the server transcribes it with omniASR
         prompt_language language of the reference audio (default: language)
         seed, steps, cfg_strength, speed   optional
    Response headers: X-Seed, X-Duration, X-Model-Text, X-Prompt-Text (URL-encoded)
"""

from __future__ import annotations

import asyncio
import io
import os
import time
import urllib.parse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from ghana_audiodit import LANGUAGES, MODES, GhanaTTS, resolve

MODEL = os.environ.get("GHANA_AUDIODIT_MODEL", "ghanaopenai/ghana-audiodit")
DTYPE = os.environ.get("GHANA_AUDIODIT_DTYPE", "bfloat16")
ASR_DIR = os.environ.get("OMNIASR_SHERPA_DIR")       # optional: transcribe prompts without text
ASR_REPO = os.environ.get("OMNIASR_SHERPA_REPO")
MAX_CHARS = int(os.environ.get("MAX_CHARS", "600"))
MAX_UPLOAD = 10 * 1024 * 1024
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "8"))

app = FastAPI(title="ghana-audiodit", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"],
                   expose_headers=["X-Seed", "X-Duration", "X-Model-Text", "X-Prompt-Text", "X-Elapsed"])

state: dict = {"tts": None, "asr": None, "waiting": 0, "served": 0}
gpu_lock = asyncio.Lock()


@app.on_event("startup")
def load() -> None:
    tts = GhanaTTS.from_pretrained(MODEL, dtype=DTYPE)
    # warm-up: the first synthesis compiles GPU kernels (~30 s); do it before taking traffic
    tts.synthesize("Akwaaba.", language="Asante_Twi_twi", seed=0)
    state["tts"] = tts
    asr_dir = ASR_DIR
    if not asr_dir and ASR_REPO:
        from huggingface_hub import snapshot_download
        asr_dir = snapshot_download(ASR_REPO, allow_patterns=["model.onnx", "tokens.txt"])
    if asr_dir and Path(asr_dir, "model.onnx").exists():
        import sherpa_onnx
        state["asr"] = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
            model=f"{asr_dir}/model.onnx", tokens=f"{asr_dir}/tokens.txt", num_threads=4,
            decoding_method="greedy_search")


def transcribe(audio: np.ndarray, sr: int) -> str:
    stream = state["asr"].create_stream()
    stream.accept_waveform(sr, audio)
    state["asr"].decode_stream(stream)
    return (stream.result.text or "").strip()


def hdr(s: str) -> str:
    return urllib.parse.quote(s, safe=" ,.;:!?'-")


@app.get("/health")
def health() -> dict:
    tts = state["tts"]
    return {"status": "ok" if tts else "loading", "model": MODEL, "dtype": DTYPE,
            "device": str(tts.device) if tts else None,
            "languages": len(LANGUAGES), "modes": list(MODES), "prompt_asr": state["asr"] is not None,
            "queue": state["waiting"], "served": state["served"], "max_chars": MAX_CHARS}


@app.get("/languages")
def languages() -> list[dict]:
    spk = state["tts"].speakers if state["tts"] else {}
    return [{"key": k, "name": v["name"], "code": v["code"],
             "speaker": {"text": spk[k]["text_native"], "seconds": spk[k]["seconds"]} if k in spk else None}
            for k, v in sorted(LANGUAGES.items(), key=lambda kv: kv[1]["name"])]


@app.get("/speakers/{language}.wav")
def speaker(language: str):
    try:
        return FileResponse(state["tts"].speaker_path(resolve(language)), media_type="audio/wav")
    except (KeyError, ValueError):
        raise HTTPException(404, "unknown language")


@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    language: str = Form("Asante_Twi_twi"),
    mode: str = Form("noprompt"),
    prompt_audio: UploadFile | None = File(None),
    prompt_text: str = Form(""),
    prompt_language: str = Form(""),
    seed: int | None = Form(None),
    steps: int = Form(16),
    cfg_strength: float = Form(4.0),
    speed: float = Form(1.0),
):
    tts = state["tts"]
    if tts is None:
        raise HTTPException(503, "model is loading, try again shortly")
    text = " ".join(text.split())
    if not text:
        raise HTTPException(400, "text is empty")
    if len(text) > MAX_CHARS:
        raise HTTPException(400, f"text is longer than {MAX_CHARS} characters")
    if mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    try:
        language = resolve(language)
        prompt_language = resolve(prompt_language) if prompt_language else language
    except ValueError as e:
        raise HTTPException(400, str(e))
    steps, cfg_strength, speed = int(np.clip(steps, 4, 32)), float(np.clip(cfg_strength, 1.0, 8.0)), float(np.clip(speed, 0.7, 1.4))

    kw: dict = {}
    used_prompt_text = ""
    if mode == "prompt":
        if prompt_audio is None:
            raise HTTPException(400, "mode=prompt needs prompt_audio")
        raw = await prompt_audio.read()
        if len(raw) > MAX_UPLOAD:
            raise HTTPException(400, "reference audio is larger than 10 MB")
        try:
            audio, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
        except Exception:
            raise HTTPException(400, "could not decode reference audio (use wav, mp3, ogg, flac or webm)")
        used_prompt_text = " ".join(prompt_text.split())
        if not used_prompt_text:
            if state["asr"] is None:
                raise HTTPException(400, "prompt_text (the transcript of the reference audio) is required")
            used_prompt_text = await asyncio.to_thread(transcribe, audio.astype(np.float32), sr)
            if not used_prompt_text:
                raise HTTPException(400, "could not transcribe the reference audio; please type its transcript")
        kw = {"prompt_audio": (audio, sr), "prompt_text": used_prompt_text, "prompt_language": prompt_language}

    if state["waiting"] >= MAX_QUEUE:
        raise HTTPException(503, "busy, please retry in a moment")
    state["waiting"] += 1
    try:
        async with gpu_lock:
            t0 = time.time()
            try:
                out = await asyncio.to_thread(tts.synthesize, text, language=language, mode=mode, seed=seed,
                                              steps=steps, cfg_strength=cfg_strength, speed=speed, **kw)
            except ValueError as e:
                raise HTTPException(400, str(e))
            elapsed = time.time() - t0
    finally:
        state["waiting"] -= 1
    state["served"] += 1

    buf = io.BytesIO()
    sf.write(buf, out.audio, out.sample_rate, format="WAV", subtype="PCM_16")
    headers = {"X-Seed": str(out.seed), "X-Duration": f"{out.seconds:.2f}", "X-Elapsed": f"{elapsed:.2f}",
               "X-Model-Text": hdr(" ".join(out.model_text)[:1500]), "X-Prompt-Text": hdr(used_prompt_text[:500])}
    return Response(buf.getvalue(), media_type="audio/wav", headers=headers)
