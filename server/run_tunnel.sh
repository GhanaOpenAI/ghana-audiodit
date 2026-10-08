#!/usr/bin/env bash
# Expose the API through Cloudflare, restarting cloudflared if it exits.
#   CONFIG=cloudflared.yml server/run_tunnel.sh        named tunnel (permanent hostname)
#   QUICK=1 PORT=8210 server/run_tunnel.sh             quick tunnel (random *.trycloudflare.com URL,
#                                                      written to $URL_FILE when it comes up)
set -uo pipefail
LOG="${LOG:-tunnel.log}"
while true; do
    if [ "${QUICK:-0}" = "1" ]; then
        # empty HOME: otherwise cloudflared applies ~/.cloudflared/config.yml ingress rules and 404s
        mkdir -p "${QUICK_HOME:-/tmp/cloudflared-quick}"
        HOME="${QUICK_HOME:-/tmp/cloudflared-quick}" \
            cloudflared tunnel --no-autoupdate --url "http://localhost:${PORT:-8210}" >> "$LOG" 2>&1 &
        pid=$!
        for _ in $(seq 60); do
            url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" | tail -1)
            [ -n "$url" ] && { echo "$url" > "${URL_FILE:-tunnel_url.txt}"; break; }
            sleep 1
        done
        wait $pid
    else
        cloudflared tunnel --no-autoupdate --config "${CONFIG:?set CONFIG}" run >> "$LOG" 2>&1
    fi
    echo "[tunnel] cloudflared exited at $(date -Is); restarting in 5s" >> "$LOG"
    sleep 5
done
