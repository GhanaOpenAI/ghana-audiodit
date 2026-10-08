"""
Deploy the ghana-audiodit inference API on Modal (https://modal.com).

    pip install modal && modal setup          # once
    modal deploy deploy/modal_app.py          # prints https://<workspace>--ghana-audiodit-api.modal.run
    modal run deploy/modal_app.py             # one test synthesis -> modal_test.wav

The endpoint is the same API as server/ (GET /health, GET /languages, GET /speakers/{lang}.wav,
POST /synthesize); point space/config.json at the URL to use the web demo with it.

GPU: an L4 (24 GB) is plenty — the model needs ~4.5 GB in bfloat16. On a T4 (no native bfloat16)
set GHANA_AUDIODIT_DTYPE=float32 (~6 GB). The model (~6 GB) and the omniASR model used to
transcribe uploaded reference audio are cached in a Modal Volume, so only the first cold start
downloads them. Containers scale to zero after SCALEDOWN seconds idle.
"""

import modal

GPU = "L4"
SCALEDOWN = 300          # seconds idle before the container stops (you pay while it runs)
REPO = "git+https://github.com/GhanaOpenAI/ghana-audiodit"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(f"ghana-audiodit[server] @ {REPO}")
    .env({
        "HF_HOME": "/cache/hf",
        "GHANA_AUDIODIT_MODEL": "ghanaopenai/ghana-audiodit",
        "GHANA_AUDIODIT_DTYPE": "bfloat16",
        "OMNIASR_SHERPA_REPO": "csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-2025-11-12",
    })
)
cache = modal.Volume.from_name("ghana-audiodit-cache", create_if_missing=True)
app = modal.App("ghana-audiodit", image=image)


@app.function(gpu=GPU, volumes={"/cache": cache}, scaledown_window=SCALEDOWN, timeout=600)
@modal.concurrent(max_inputs=8)          # requests queue on one GPU; the server runs one synthesis at a time
@modal.asgi_app()
def api():
    from ghana_audiodit.server import app as fastapi_app
    return fastapi_app


@app.function(gpu=GPU, volumes={"/cache": cache}, timeout=900)
def synthesize(text: str, language: str = "Asante_Twi_twi", mode: str = "noprompt", seed: int | None = None) -> bytes:
    """Direct (non-HTTP) synthesis, e.g. from a batch job: returns WAV bytes."""
    import io

    import soundfile as sf
    from ghana_audiodit import GhanaTTS

    tts = GhanaTTS.from_pretrained()
    out = tts.synthesize(text, language=language, mode=mode, seed=seed)
    cache.commit()
    buf = io.BytesIO()
    sf.write(buf, out.audio, out.sample_rate, format="WAV")
    return buf.getvalue()


@app.local_entrypoint()
def main(text: str = "Akwaaba! Wo ho te sɛn ɛnnɛ?", language: str = "Asante_Twi_twi"):
    wav = synthesize.remote(text, language)
    with open("modal_test.wav", "wb") as f:
        f.write(wav)
    print(f"wrote modal_test.wav ({len(wav) / 1024:.0f} KB)")
