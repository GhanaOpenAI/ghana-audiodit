#!/usr/bin/env bash
# Keep the ghana-audiodit API up: restart uvicorn if it exits (crash / OOM).
#   VENV=/path/to/.venv GHANA_AUDIODIT_MODEL=ghanaopenai/ghana-audiodit PORT=8210 server/run_supervised.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck disable=SC1091
source "${VENV:?set VENV to the virtualenv}/bin/activate"
PORT="${PORT:-8210}"
LOG="${LOG:-server.log}"
while true; do
    echo "[supervisor] starting uvicorn on :$PORT at $(date -Is)" >> "$LOG"
    python -m uvicorn ghana_audiodit.server:app --host 127.0.0.1 --port "$PORT" --timeout-keep-alive 75 >> "$LOG" 2>&1
    echo "[supervisor] uvicorn exited with $? at $(date -Is); restarting in 5s" >> "$LOG"
    sleep 5
done
