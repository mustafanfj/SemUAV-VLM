#!/usr/bin/env bash
# Optional: start Jupyter for interactive viewing of the v3r2 notebook on 127.0.0.1:8888 (inside tmux).
# Do NOT use this while run_semuav_v3r2.sh is running a phase (both would need GPU 1).
set -Eeuo pipefail
umask 077
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ss -ltn 2>/dev/null | grep -q '127.0.0.1:8888\|\*:8888\|0.0.0.0:8888'; then
    echo "STOP: port 8888 is in use. Stop the old Jupyter server first (Ctrl+C in its tmux pane)." >&2
    exit 1
fi
source "$HERE/semuav_v3r2_env.sh"
cd "$RELEASE_ROOT"
/usr/bin/python3 -I -B "$RELEASE_ROOT/v3_deployment.py" verify
CUDA_VISIBLE_DEVICES='' "$CANDIDATE/bin/python" -I -B "$RELEASE_ROOT/v3r2_preflight.py"
echo "Kernel to select in Jupyter: Python (SemUAV-VLM 2026-10-13 v3r2)"
exec "$CANDIDATE/bin/jupyter-lab" --no-browser --ip=127.0.0.1 --port=8888 \
    --ServerApp.port_retries=0 --ServerApp.root_dir="$RELEASE_ROOT"
