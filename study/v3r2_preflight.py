"""CPU-only preflight: v3r2 configuration must equal v3r1 except release identity.

Run with the candidate Python, from the v3r2 release root, under the launch
environment printed by ``v3r2_launch_env.py``.  VLM revisions are taken from
the v3r1 manifest for this comparison only (the notebook derives the same
values from the shared model lock).  Exit status 0 means validation reuse of
v3r1 computations can be exact; anything else stops before Jupyter starts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from semuav.config import load_config  # noqa: E402
from semuav.registry import _plain, hash_object  # noqa: E402
from semuav.v3r1_import import _config_differences, _normalized_config  # noqa: E402
from v3r2_launch_env import V3R1_RUN_ROOT  # noqa: E402

REVISION_VARIABLES = {
    ("vlms", 0): "SEMUAV_QWEN_REVISION",
    ("vlms", 1): "SEMUAV_SMOLVLM_REVISION",
    ("upper_bound",): "SEMUAV_UPPER_BOUND_REVISION",
}


def main() -> int:
    manifest = json.loads((V3R1_RUN_ROOT / "run_manifest.json").read_text(encoding="utf-8"))
    predecessor = manifest["config"]
    if hash_object(predecessor) != manifest["config_hash"]:
        print("STOP: v3r1 manifest config does not match its recorded hash")
        return 1
    saved = {name: os.environ.get(name) for name in REVISION_VARIABLES.values()}
    try:
        for path, variable in REVISION_VARIABLES.items():
            node = predecessor["models"][path[0]]
            node = node[path[1]] if len(path) > 1 else node
            os.environ[variable] = str(node["revision"])
        current = _plain(load_config(ROOT / "configs" / "visdrone_paper.yaml").as_dict())
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    differences = _config_differences(_normalized_config(predecessor), _normalized_config(current))
    if differences:
        print("STOP: v3r2 configuration differs from v3r1 beyond release identity:")
        for item in differences[:25]:
            print("   ", item)
        return 1
    print("V3R2_PREFLIGHT_CONFIG_EQUIVALENT_TO_V3R1", manifest["config_hash"][:10])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
