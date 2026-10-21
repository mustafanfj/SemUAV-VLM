#!/usr/bin/env bash
# Deploy the SemUAV-VLM v3r2 source sibling (CPU only; no GPU, no dataset read, no test access).
# 1. checks this delivery's checksums, 2. refuses while the v3r1 kernel is still running,
# 3. deploys the sealed source + kernel + v3r1 reuse inventory and runs the CPU test suite,
# 4. checks that the v3r2 configuration equals v3r1 except release identity.
set -Eeuo pipefail
umask 077
export LC_ALL=C TZ=UTC PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

STUDY_ROOT='/data/Lane Detection Datasets/musta/semuav_vlm'
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_NAME='SemUAV-VLM-VisDrone-Paper-Study-20261013-v3r2.zip'
PACKAGE='SemUAV-VLM-VisDrone-Paper-Study'
RELEASE_ROOT="$STUDY_ROOT/releases/release_20261013_visdrone_paper_v3r2/$PACKAGE"
V3R1_RELEASE_ROOT="$STUDY_ROOT/releases/release_20260928_visdrone_paper_v3r1/$PACKAGE"
V3R1_STATUS="$STUDY_ROOT/.cache/visdrone_paper_v3r1/runs/semuav_visdrone_v3r1-full-448d523d65-7bb899d11e/status.sqlite"
CANDIDATE_PY="$STUDY_ROOT/venvs/release_20260912_visdrone_inference_candidate_v2r3/bin/python"

cd "$HERE"
sha256sum --check --strict SHA256SUMS

# The v3r1 run must be quiescent while its validation results are inventoried.
for pid in $(pgrep -f ipykernel_launcher || true); do
    if [ "$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)" = "$V3R1_RELEASE_ROOT" ]; then
        echo "STOP: the v3r1 notebook kernel (pid $pid) is still running. Shut it down in Jupyter first." >&2
        exit 1
    fi
done
if [ -n "$(find "$V3R1_STATUS" -mmin -10 2>/dev/null)" ]; then
    echo "STOP: the v3r1 run changed in the last 10 minutes; make sure nothing is running, then retry." >&2
    exit 1
fi

ARCHIVE_SHA256="$(sha256sum "$HERE/$ARCHIVE_NAME" | cut -d' ' -f1)"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
/usr/bin/python3 -I -B -c 'import sys, zipfile; sys.stdout.buffer.write(zipfile.ZipFile(sys.argv[1]).read(sys.argv[2]))' \
    "$HERE/$ARCHIVE_NAME" "$PACKAGE/v3_deployment.py" > "$WORK/v3_deployment.py"
env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C TZ=UTC PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    /usr/bin/python3 -I -B "$WORK/v3_deployment.py" deploy \
    --archive "$HERE/$ARCHIVE_NAME" --archive-sha256 "$ARCHIVE_SHA256"

echo "--- v3r2 launch environment (derived from the v3r1 run) ---"
/usr/bin/python3 -I -B "$RELEASE_ROOT/v3r2_launch_env.py" --check
LAUNCH_ENV="$(/usr/bin/python3 -I -B "$RELEASE_ROOT/v3r2_launch_env.py")"
(
    cd "$RELEASE_ROOT"
    eval "$LAUNCH_ENV"
    export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SEMUAV_MODELS_LOCAL_ONLY=1
    "$CANDIDATE_PY" -I -B "$RELEASE_ROOT/v3r2_preflight.py"
)
echo "V3R2_DEPLOYMENT_AND_PREFLIGHT_COMPLETE"
