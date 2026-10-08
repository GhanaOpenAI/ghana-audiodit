# ghana-audiodit inference API — runs on any machine with an NVIDIA GPU (≥ 6 GB), an NVIDIA driver
# supporting CUDA 12.8, and the NVIDIA container toolkit.
#
#   docker run --gpus all -p 8000:8000 ghcr.io/ghanaopenai/ghana-audiodit:latest
#   curl -X POST localhost:8000/synthesize -F "text=Akwaaba!" -F language=Asante_Twi_twi -o out.wav
#
# Build args
#   BAKE_MODEL=1 (default)  model inside the image; runs fully offline
#   BAKE_MODEL=0            "slim": downloads the model on first start (mount a volume at /models to keep it)
#   MODEL_REVISION          model revision to bake (default: main)
#
# Same stack the release was tested on: Python 3.12, torch 2.9.1 + CUDA 12.8 (the torch wheel
# bundles the CUDA libraries), pinned packages in docker/requirements.lock.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PATH=/opt/venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ffmpeg libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* && python3 -m venv /opt/venv

WORKDIR /app
RUN pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
COPY docker/requirements.lock /app/docker/requirements.lock
RUN pip install -r docker/requirements.lock
COPY pyproject.toml README.md LICENSE LICENSE-longcat-audiodit /app/
COPY ghana_audiodit /app/ghana_audiodit
RUN pip install --no-deps . && python -c "import ghana_audiodit, ghana_audiodit.text, ghana_audiodit.server"

ARG BAKE_MODEL=1
ARG MODEL_REVISION=main
ENV HF_HOME=/models GHANA_AUDIODIT_MODEL=ghanaopenai/ghana-audiodit GHANA_AUDIODIT_DTYPE=bfloat16
RUN if [ "$BAKE_MODEL" = "1" ]; then python -c "\
from huggingface_hub import snapshot_download as d; \
d('ghanaopenai/ghana-audiodit', revision='$MODEL_REVISION', allow_patterns=['*.json','*.safetensors','*.py','speakers/*']); \
d('google/umt5-base', allow_patterns=['*.json','*.model'])"; fi
# offline when baked, so the container never needs the network
ENV HF_HUB_OFFLINE=${BAKE_MODEL}

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,json,sys; sys.exit(json.load(urllib.request.urlopen('http://localhost:8000/health'))['status']!='ok')"
CMD ["uvicorn", "ghana_audiodit.server:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "75"]
