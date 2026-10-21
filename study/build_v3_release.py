"""Build the source-only v3r2 ZIP and its deterministic checksum inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent
ARCHIVE_NAME = "SemUAV-VLM-VisDrone-Paper-Study-20261013-v3r2.zip"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(destination: Path) -> Path:
    excluded_names = {"RELEASE_MANIFEST.json", "RELEASE_MANIFEST_SHA256.txt"}
    files = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and not any(part in {"__pycache__", ".git", ".pytest_cache", ".ipynb_checkpoints"}
                    for part in path.relative_to(ROOT).parts)
        and path.suffix not in {".pyc", ".pyo", ".zip", ".log"}
        and not (path.parent == ROOT and path.name in excluded_names)
    )
    if any(path.is_symlink() for path in files):
        raise ValueError("Release files must not be symlinks")
    inventory = [
        {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
         "sha256": sha256(path),
         "role": "historical_reference" if "legacy" in path.relative_to(ROOT).parts
                 else "source_payload"}
        for path in files
    ]
    manifest = {
        "schema_version": "semuav.visdrone-v3-source-release.v1",
        "release_id": "release_20261013_visdrone_paper_v3r2",
        "release_revision": "v3r2",
        "run_name": "semuav_visdrone_v3r2",
        "archive_name": ARCHIVE_NAME,
        "archive_member_count": len(files) + 2,
        "study_entrypoint": "SemUAV_VLM_VisDrone_Paper_Study.ipynb",
        "empirical_results_included": False,
        "dataset_material_included": False,
        "payload_file_count": len(files),
        "payload_total_uncompressed_bytes": sum(row["bytes"] for row in inventory),
        "payload_inventory": inventory,
        "predecessor": {
            "release_id": "release_20260921_visdrone_paper_v2r13",
            "archive_sha256": "5c5191b0bf439623bd868acc0a6222de44573c0c46416131e4842fff5b2f5636",
            "release_manifest_sha256": "53dcc739804542fda76a65ce32d88d654e49e1d09f518c35de93ef18b407d8b3",
            "deployment_receipt_sha256": "110fb6d8c061639746b807b76a5069ab76ceb453a88d4d90941ca368c9ae7c4e",
            "operator_observed_stop": "Experiment 2 native-cache video encoder: height not divisible by 2 (1360x765)",
            "runtime_receipt_bytes_available_locally": False,
        },
        "v3r2_pre_test_amendment": {
            "record": "PRE_TEST_CALIBRATION_GATE_AMENDMENT_20261013_V3R2.md",
            "predecessor_release_id": "release_20260928_visdrone_paper_v3r1",
            "predecessor_run_name": "semuav_visdrone_v3r1",
            "predecessor_code_sha256": "7bb899d11e95b5780278fe38ca653dad34e71137e2f296c7a72056586711db98",
            "untouched_test_accessed": False,
            "protocol_frozen_before_amendment": False,
            "receipted_v3r1_corrections_in_source": [
                "experiment3_episode_key_merge", "policy_reducer_usecols_order",
                "l0_l2_cbor_wire_format_and_rate_screened_best_fixed",
                "calibration_aware_controller_with_class_balanced_profiles",
            ],
            "calibration_gate": "measured FAIL retained; routes to H4 NOT_ESTIMABLE when the frozen controller cannot consume confidence",
            "validation_reuse": "exact hash-verified v3r1 stages for unchanged producers; Experiment 4 recomputed",
            "raw_inference_producer_and_measurement_sources_changed": False,
        },
        "v3r1_source_correction": {
            "supersedes_source_release": "release_20260928_visdrone_paper_v3",
            "superseded_archive_sha256": "25b9a80da04e95d15ddc9a46ca8de0682e377def9913c768f67b38db0b3e83ae",
            "operator_observed_cpu_test_run": {"tests_run": 568, "failures": 0, "errors": 3},
            "correction_scope": "test-helper imports isolated from third-party tests package namespace",
            "scientific_producer_or_experimental_design_changed_from_v3": False,  # v3r1 relative to v3
            "failed_v3_source_and_logs_preserved": True,
            "failed_v3_deployment_receipt_required": False,
            "cpu_acceptance_gate_bypassed": False,
        },
        "compatibility": {
            "registered_experimental_design_changed": False,
            "requirements_changed": False,
            "raw_inference_producer_and_measurement_sources_changed": False,
            "video_wire_format_amended_for_odd_geometry": True,
            "native_receiver_caches": "validation reused exactly from v3r1 (hash-verified, re-validated); test built fresh",
            "controller_revised_before_test": True,
            "calibration_gate_rule_amended_before_test": True,
            "fresh_three_model_runtime_smoke_required": True,
            "fresh_scientific_environment_seal_required": True,
            "historical_receipts_modified": False,
        },
        "raw_inference_recovery": {
            "receipt_sha256": "6ca88cb5e82c8b33e27fd229443e233a48323d91fd73d0e79ec51ff8fc61a608",
            "artifacts": 132, "rows": 4152,
            "scope": "authenticated_raw_v2r9_experiment1_only",
            "source_bytes_included": False,
        },
        "software_validation_record": "VALIDATION_REPORT.md",
        "audit_record": "V3_AUDIT_AND_CHANGES.md",
    }
    manifest_path = ROOT / "RELEASE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    checksum_files = sorted(files + [manifest_path])
    checksum_path = ROOT / "RELEASE_MANIFEST_SHA256.txt"
    checksum_path.write_text("".join(sorted(
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n"
        for path in checksum_files
    )))
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / ARCHIVE_NAME
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for path in sorted(checksum_files + [checksum_path]):
            info = zipfile.ZipInfo("SemUAV-VLM-VisDrone-Paper-Study/" + path.relative_to(ROOT).as_posix(),
                                   date_time=(2026, 10, 13, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            output.writestr(info, path.read_bytes())
    print(json.dumps({"archive": str(archive.resolve()), "sha256": sha256(archive),
                      "payload_files": len(files), "archive_members": len(files) + 2}))
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    build(parser.parse_args().destination)
