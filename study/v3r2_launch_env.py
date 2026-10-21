"""Print the v3r2 launch environment derived from the quiescent v3r1 run (stdlib only).

v3r2 must see exactly the data, output, detector, feature-head and
target-hardware locations that v3r1 used, so its configuration differs from
v3r1 only by release identity (which the exact-reuse check requires).  Those
locations are read from the v3r1 run manifest and printed as shell
``export`` lines.  Only the cache root changes, to the v3r2 namespace.

    eval "$(/usr/bin/python3 -I -B v3r2_launch_env.py)"
    /usr/bin/python3 -I -B v3r2_launch_env.py --check   # print, export nothing
"""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import sys

STUDY_ROOT = Path("/data/Lane Detection Datasets/musta/semuav_vlm")
V3R1_RUN_ROOT = (
    STUDY_ROOT / ".cache" / "visdrone_paper_v3r1" / "runs"
    / "semuav_visdrone_v3r1-full-448d523d65-7bb899d11e"
)
V3R2_CACHE_ROOT = STUDY_ROOT / ".cache" / "visdrone_paper_v3r2"
PATH_VARIABLES = {
    ("paths", "data_root"): "SEMUAV_DATA_ROOT",
    ("paths", "output_root"): "SEMUAV_OUTPUT_ROOT",
    ("paths", "annotations"): "SEMUAV_ANNOTATIONS",
    ("paths", "test_annotations"): "SEMUAV_TEST_ANNOTATIONS",
    ("paths", "split_manifest"): "SEMUAV_SPLIT_MANIFEST",
    ("paths", "channel_traces"): "SEMUAV_CHANNEL_TRACES",
    ("paths", "telemetry_manifest"): "SEMUAV_TELEMETRY_MANIFEST",
    ("models", "reused_v1_artifacts", "source_frozen_input_dir"): "SEMUAV_REUSED_V1_FROZEN_INPUT_DIR",
    ("models", "detector", "checkpoint"): "SEMUAV_DETECTOR_CHECKPOINT",
    ("models", "detector", "checkpoint_sha256"): "SEMUAV_DETECTOR_SHA256",
    ("models", "detector", "training_receipt"): "SEMUAV_DETECTOR_TRAINING_RECEIPT",
    ("models", "feature_baseline", "receiver_head"): "SEMUAV_FEATURE_RECEIVER_HEAD",
    ("models", "feature_baseline", "receiver_head_revision"): "SEMUAV_FEATURE_HEAD_SHA256",
    ("hardware", "hardware_id"): "SEMUAV_TARGET_HARDWARE_ID",
    ("hardware", "target_log"): "SEMUAV_TARGET_LOG",
}


def launch_environment() -> dict[str, str]:
    manifest = json.loads((V3R1_RUN_ROOT / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("run_name") != "semuav_visdrone_v3r1" or manifest.get("profile") != "full":
        raise SystemExit("STOP: unexpected v3r1 run manifest identity")
    config = manifest["config"]
    v3r1_project = str(config["paths"]["project_root"])
    environment = {"SEMUAV_CACHE_ROOT": str(V3R2_CACHE_ROOT)}
    for keys, variable in PATH_VARIABLES.items():
        value: object = config
        for key in keys:
            value = value[key]  # type: ignore[index]
        text = str(value)
        if text.startswith(v3r1_project):
            # Resolved inside the sealed v3r1 source tree: v3r2 resolves the
            # same relative default inside its own sealed tree.
            continue
        environment[variable] = text
    return environment


def main() -> None:
    environment = launch_environment()
    check = "--check" in sys.argv[1:]
    for name, value in sorted(environment.items()):
        line = f"{name}={shlex.quote(value)}"
        print(f"# {line}" if check else f"export {line}")


if __name__ == "__main__":
    main()
