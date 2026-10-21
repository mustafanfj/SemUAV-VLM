#!/usr/bin/env bash
# Run v3r2 notebook phases headlessly, one after another (run inside tmux; survives a closed laptop).
#   bash run_semuav_v3r2.sh prepare validation     <- validation results (this step)
#   bash run_semuav_v3r2.sh freeze                  <- later, after supervisor sign-off
# Executed notebooks and logs go to ~/semuav_v3r2_outputs/. The run stops at the first failure.
set -Eeuo pipefail
umask 077
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ "$#" -ge 1 ] || { echo "usage: bash run_semuav_v3r2.sh prepare validation" >&2; exit 2; }
source "$HERE/semuav_v3r2_env.sh"
mkdir -p "$HOME/semuav_v3r2_outputs"
exec 9>"$HOME/semuav_v3r2_outputs/.runner.lock"
flock -n 9 || { echo "STOP: another v3r2 run is already in progress." >&2; exit 1; }
if pgrep -f "jupyter-lab.*$RELEASE_ROOT" >/dev/null; then
    echo "STOP: the interactive v3r2 Jupyter is running; stop it first (it would compete for GPU 1)." >&2
    exit 1
fi
cd "$RELEASE_ROOT"
/usr/bin/python3 -I -B "$RELEASE_ROOT/v3_deployment.py" verify
CUDA_VISIBLE_DEVICES='' "$CANDIDATE/bin/python" -I -B "$RELEASE_ROOT/v3r2_preflight.py"
for phase in "$@"; do
    echo "===== $(date -u +%FT%TZ) phase: $phase ====="
    "$CANDIDATE/bin/python" -I -B "$HERE/run_notebook_v3r2.py" "$phase"
done
echo "===== $(date -u +%FT%TZ) ALL REQUESTED PHASES FINISHED: $* ====="
