"""Build the compact figure-data folder from a validation/test results bundle.

The figure scripts read ``exp4_cells.csv.gz`` and ``exp4_mode_counts.csv.gz``.
Both are exact re-aggregations of the run's own policy reduction:

  * exp4_cells       <- results/<phase>_policy_reduction/policy_pair_cells.csv.gz
  * exp4_mode_counts <- results/<phase>_policy_reduction/controller_summary.csv.gz

Usage:
    python figdata_from_bundle.py <bundle_results_dir> <out_dir> [phase]
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))

import sys
from pathlib import Path
import pandas as pd

MODE_COLUMNS = {
    "l0_heartbeat": "L0_HEARTBEAT", "l1_alert": "L1_ALERT", "l2_semantics": "L2_SEMANTICS",
    "l3_semantics_roi": "L3_SEMANTICS_ROI", "l4_features": "L4_FEATURES", "l5_image_or_clip": "L5_IMAGE_OR_CLIP",
}


def main() -> None:
    results, out = Path(sys.argv[1]), Path(sys.argv[2])
    phase = sys.argv[3] if len(sys.argv) > 3 else "validation"
    reduction = results / f"{phase}_policy_reduction"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results").exists() or (out / "results").symlink_to(results.resolve(), target_is_directory=True)

    cells = pd.read_csv(reduction / "policy_pair_cells.csv.gz")
    if not (cells.critical_trials + cells.noncritical_trials).eq(cells.trials).all():
        raise SystemExit("critical + non-critical trials do not add up to trials")
    pd.DataFrame({
        "policy": cells.policy, "trace_id": cells.trace_id, "visual_group_id": cells.visual_group_id,
        "correct_timely_action": cells.correct_timely_actions, "critical": cells.critical_trials,
        "critical_miss": cells.critical_misses, "false_alert": cells.false_alerts,
        "total_wire_bytes": cells.total_wire_bytes, "energy_j": cells.total_energy_j, "decisions": cells.trials,
    }).to_csv(out / "exp4_cells.csv.gz", index=False)

    summary = pd.read_csv(reduction / "controller_summary.csv.gz")
    rows = [
        {"policy": r.policy, "selected_mode": mode, "n": getattr(r, f"mode_frequency_{key}") * r.decisions}
        for r in summary.itertuples() for key, mode in MODE_COLUMNS.items()
    ]
    pd.DataFrame(rows).to_csv(out / "exp4_mode_counts.csv.gz", index=False)
    print("wrote", out / "exp4_cells.csv.gz", "and", out / "exp4_mode_counts.csv.gz")


if __name__ == "__main__":
    main()
