"""
Inference API for ghana-audiodit (powers the Hugging Face Space).

    pip install "ghana-audiodit[server] @ git+https://github.com/GhanaOpenAI/ghana-audiodit"
    uvicorn ghana_audiodit.server:app --host 0.0.0.0 --port 8210

Environment
    GHANA_AUDIODIT_MODEL   model repo or local folder (default ghanaopenai/ghana-audiodit)
    GHANA_AUDIODIT_DTYPE   bfloat16 (default, ~4 GB VRAM) or float32 (~6 GB)
    MAX_CHARS, MAX_QUEUE   request limits (600 characters, 8 waiting requests)

Endpoints
    GET  /health                    model + queue status
    GET  /languages                 the 43 languages
    POST /synthesize                form fields -> audio/wav
         text        required, normal spelling (English words may be mixed in), ≤ MAX_CHARS
         language    key / ISO code / name, default Asante_Twi_twi
         seed, steps, cfg_strength, speed   optional
    Response headers: X-Seed, X-Duration, X-Elapsed, X-Model-Text (URL-encoded)
"""

from __future__ import annotations

import asyncio
import io
import os
import time
import urllib.parse

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from ghana_audiodit import LANGUAGES, GhanaTTS, resolve

MODEL = os.environ.get("GHANA_AUDIODIT_MODEL", "ghanaopenai/ghana-audiodit")
DTYPE = os.environ.get("GHANA_AUDIODIT_DTYPE", "bfloat16")
MAX_CHARS = int(os.environ.get("MAX_CHARS", "600"))
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "8"))

app = FastAPI(title="ghana-audiodit", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"],
                   expose_headers=["X-Seed", "X-Duration", "X-Model-Text", "X-Elapsed"])

state: dict = {"tts": None, "waiting": 0, "served": 0}
gpu_lock = asyncio.Lock()


@app.on_event("startup")
def load() -> None:
    tts = GhanaTTS.from_pretrained(MODEL, dtype=DTYPE)
    # warm-up: the first synthesis compiles GPU kernels (~30 s); do it before taking traffic
    tts.synthesize("Akwaaba.", language="Asante_Twi_twi", seed=0)
    state["tts"] = tts


def hdr(s: str) -> str:
    return urllib.parse.quote(s, safe=" ,.;:!?'-")


@app.get("/health")
def health() -> dict:
    tts = state["tts"]
    return {"status": "ok" if tts else "loading", "model": MODEL, "dtype": DTYPE,
            "device": str(tts.device) if tts else None, "languages": len(LANGUAGES),
            "queue": state["waiting"], "served": state["served"], "max_chars": MAX_CHARS}


@app.get("/languages")
def languages() -> list[dict]:
    return [{"key": k, "name": v["name"], "code": v["code"]}
            for k, v in sorted(LANGUAGES.items(), key=lambda kv: kv[1]["name"])]



@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    language: str = Form("Asante_Twi_twi"),
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
    try:
        language = resolve(language)
    except ValueError as e:
        raise HTTPException(400, str(e))
    steps, cfg_strength, speed = int(np.clip(steps, 4, 32)), float(np.clip(cfg_strength, 1.0, 8.0)), float(np.clip(speed, 0.7, 1.4))

    if state["waiting"] >= MAX_QUEUE:
        raise HTTPException(503, "busy, please retry in a moment")
    state["waiting"] += 1
    try:
        async with gpu_lock:
            t0 = time.time()
            try:
                out = await asyncio.to_thread(tts.synthesize, text, language=language, seed=seed,
                                              steps=steps, cfg_strength=cfg_strength, speed=speed)
            except ValueError as e:
                raise HTTPException(400, str(e))
            elapsed = time.time() - t0
    finally:
        state["waiting"] -= 1
    state["served"] += 1

    buf = io.BytesIO()
    sf.write(buf, out.audio, out.sample_rate, format="WAV", subtype="PCM_16")
    headers = {"X-Seed": str(out.seed), "X-Duration": f"{out.seconds:.2f}", "X-Elapsed": f"{elapsed:.2f}",
               "X-Model-Text": hdr(" ".join(out.model_text)[:1500])}
    return Response(buf.getvalue(), media_type="audio/wav", headers=headers)
