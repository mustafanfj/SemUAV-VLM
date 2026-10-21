#!/usr/bin/env bash
# Pack the v3r2 validation results for review: executed notebooks + logs, paper figures and tables,
# and the small result files. Read-only on the run. Output: ~/semuav_v3r2_validation_bundle.tar.gz
set -Eeuo pipefail
STUDY='/data/Lane Detection Datasets/musta/semuav_vlm'
RUN="$(ls -d "$STUDY"/.cache/visdrone_paper_v3r2/runs/semuav_visdrone_v3r2-full-* | head -1)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/semuav_v3r2_env.sh"
FIGS="$SEMUAV_OUTPUT_ROOT/semuav_visdrone_v3r2/$(basename "$RUN")/paper_figures_validation"
[ -d "$FIGS" ] || FIGS=""
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
B="$TMP/semuav_v3r2_validation"; mkdir -p "$B/results" "$B/notebook_runs"
cp "$HOME"/semuav_v3r2_outputs/*.ipynb "$HOME"/semuav_v3r2_outputs/*.log "$B/notebook_runs/" 2>/dev/null || true
find "$RUN/results" -maxdepth 1 -type f -size -20M -exec cp {} "$B/results/" \;
cp -r "$RUN/results/validation_policy_reduction" "$B/results/" 2>/dev/null || true
if [ -n "$FIGS" ]; then mkdir -p "$B/paper_figures"; find "$FIGS" -maxdepth 1 -type f -exec cp {} "$B/paper_figures/" \; ; fi
bash "$HERE/status_semuav_v3r2.sh" > "$B/STATUS.txt" 2>&1 || true
tar -czf "$HOME/semuav_v3r2_validation_bundle.tar.gz" -C "$TMP" semuav_v3r2_validation
ls -lh "$HOME/semuav_v3r2_validation_bundle.tar.gz"
