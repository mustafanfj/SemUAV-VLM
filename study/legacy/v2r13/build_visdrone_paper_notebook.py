from __future__ import annotations

"""Build the phase-safe VisDrone paper-study notebook.

The notebook is intentionally a thin orchestrator.  Scientific implementation,
validation, caching, and claim gates live in the adjacent ``semuav`` package so
long GPU stages can resume independently of Jupyter cell state.
"""

import copy
import hashlib
import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "SemUAV_VLM_VisDrone_Paper_Study.ipynb"


def _source(value: str) -> str:
    return textwrap.dedent(value).strip() + "\n"


def md(value: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(value)}


def code(value: str, *tags: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": list(tags)} if tags else {},
        "outputs": [],
        "source": _source(value),
    }


V2R13_DEPLOYMENT_RECEIPT_FIXTURE_RECORD_KEYS = frozenset({
    "created_at_utc", "release_root", "deployment_mode", "release",
    "installer", "deployment_script", "builder", "source_release",
    "authenticated_base_v2r3", "candidate_environment", "unit_tests",
    "jupyter_kernel", "jupyter_launch_target", "trusted_code_cache_namespace",
    "attempt_directory", "deployment_log", "experiment1_recovery",
    "predecessor_v2r12_state",
})


def _v2r12_predecessor_observed_terminal_boundary_fixture(
    correction_record_sha256: str,
) -> dict[str, object]:
    """Return the canonical v2r11 pre-smoke stop carried by v2r12."""

    if (
        not isinstance(correction_record_sha256, str)
        or len(correction_record_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in correction_record_sha256
        )
    ):
        raise ValueError("correction_record_sha256 must be one lowercase SHA-256")
    return {
        "schema_version": (
            "semuav.visdrone-v2r11-operator-observed-terminal-boundary.v1"
        ),
        "predecessor_release_id": "release_20260921_visdrone_paper_v2r11",
        "predecessor_run_name": "semuav_visdrone_v2r11",
        "correction_record": (
            "PRE_SMOKE_PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_20260921_V2R12.md"
        ),
        "correction_record_sha256": correction_record_sha256,
        "evidence_model": (
            "authenticated_source_deployment_and_operator_observed_pre_smoke_stop"
        ),
        "operator_observed_execution": {
            "source_deployment_ready": True,
            "deployment_receipt_sha256": (
                "befdce3067b571933dd94d35758c2c29385a06a8bb02ea93ae681117dcc17f3e"
            ),
            "prepare_phase_entered": True,
            "candidate_ready_verification_entered": True,
            "candidate_ready_verification_passed": False,
            "failure_stage": "prepare_candidate_ready_receipt_verification",
            "failure_exception_type": "StudyGateError",
            "failure_kind": (
                "immediate_predecessor_boundary_validated_against_older_v2r9_schema"
            ),
            "runtime_smoke_entered": False,
            "trusted_code_cache_frozen": False,
            "scientific_environment_sealed": False,
            "validation_software_acceptance_entered": False,
            "validation_experiments_1_to_7_started": False,
            "derived_results_generated": False,
            "report_or_figures_generated": False,
            "protocol_frozen": False,
            "untouched_test_accessed": False,
            "v2r9_experiment1_recovery_receipt_preserved": True,
            "v2r9_experiment1_recovery_artifact_count": 132,
            "v2r9_experiment1_recovery_row_count": 4152,
        },
        "source_authenticated_defect": {
            "schema_version": (
                "semuav.visdrone-v2r12-predecessor-boundary-routing-correction.v1"
            ),
            "defect_kind": (
                "deployment_receipt_immediate_predecessor_boundary_checked_as_v2r9_boundary"
            ),
            "affected_entrypoint": "notebook _verify_candidate_ready_receipt",
            "actual_boundary_schema": (
                "semuav.visdrone-v2r10-operator-observed-terminal-boundary.v1"
            ),
            "incorrect_expected_boundary_schema": (
                "semuav.visdrone-v2r9-operator-observed-terminal-boundary.v1"
            ),
            "correction": (
                "verify_each_terminal_boundary_at_its_authenticated_provenance_layer"
            ),
            "v2r11_receipt_mutated": False,
            "scientific_contract_changed": False,
            "requirements_changed": False,
        },
        "inherited_v2r10_terminal_boundary_preserved": True,
        "inherited_v2r9_terminal_boundary_preserved": True,
        "v2r12_fresh_runtime_smoke_required": True,
        "v2r12_raw_experiment1_recovery_authorized": True,
        "v2r12_raw_experiment1_recovery_schema": (
            "semuav.visdrone-v2r9-exp1-recovery.v1"
        ),
        "v2r12_derived_results_recomputation_required": True,
        "v2r12_downstream_results_recomputation_required": True,
        "v2r11_derived_scientific_outputs_reused": False,
    }


def _v2r13_predecessor_observed_terminal_boundary_fixture(correction_record_sha256):
    """Canonical sealed v2r12 native visual receiver stop, with no derived reuse."""
    if (not isinstance(correction_record_sha256, str) or len(correction_record_sha256) != 64
        or any(c not in "0123456789abcdef" for c in correction_record_sha256)):
        raise ValueError("correction_record_sha256 must be one lowercase SHA-256")
    value = {'correction_record': 'PRE_VALIDATION_LOW_EVIDENCE_PACKET_CORRECTION_20260921_V2R13.md',
     'correction_record_sha256': 'CORRECTION_SHA256',
     'evidence_model': 'authenticated_source_runtime_seal_and_operator_observed_validation_stop',
     'inherited_v2r10_terminal_boundary_preserved': True,
     'inherited_v2r11_terminal_boundary_preserved': True,
     'inherited_v2r9_terminal_boundary_preserved': True,
     'operator_observed_execution': {'deployment_receipt_sha256': '889850427153f9559fc038e2f10df174874dd59e8bff0473f61dcce1b62d2463',
                                     'experiment2_frontier_completed_in_observed_invocation': False,
                                     'failure_call_path': ['run_validation_perception_and_frontier',
                                                           '_resolve_native_cache',
                                                           'build_native_receiver_cache',
                                                           'nvml_measurement_runner',
                                                           'RegisteredVisDroneNativeReceiver.__call__',
                                                           'RegisteredVisDroneNativeReceiver._frame_action',
                                                           'DetectorOnlyBackend.infer',
                                                           'validate_semantic_content_packet',
                                                           'validate_semantic_packet'],
                                     'failure_exception_type': 'ValueError',
                                     'failure_kind': 'positive_event_with_insufficient_evidence_rejected_by_frozen_semantic_schema',
                                     'failure_message': 'Insufficient evidence requires '
                                                        'event=insufficient_evidence and '
                                                        'recommended_action=REVIEW',
                                     'failure_stage': 'validation_native_receiver_cache_visual_payload_warmup',
                                     'later_scientific_artifacts_must_be_absent_at_deployment': True,
                                     'later_validation_completion_observed': False,
                                     'native_receiver_cache_build_returned': False,
                                     'protocol_freeze_observed': False,
                                     'report_or_figures_completion_observed': False,
                                     'runtime_smoke_three_of_three_passed': True,
                                     'scientific_environment_sealed': True,
                                     'source_deployment_ready': True,
                                     'trusted_code_cache_frozen': True,
                                     'untouched_test_access_observed': False,
                                     'v2r9_experiment1_recovery_artifact_count': 132,
                                     'v2r9_experiment1_recovery_receipt_preserved': True,
                                     'v2r9_experiment1_recovery_row_count': 4152,
                                     'validation_experiments_1_2_entered': True,
                                     'validation_experiments_1_2_returned': False},
     'predecessor_release_id': 'release_20260921_visdrone_paper_v2r12',
     'predecessor_run_name': 'semuav_visdrone_v2r12',
     'schema_version': 'semuav.visdrone-v2r12-operator-observed-terminal-boundary.v1',
     'source_authenticated_defect': {'affected_entrypoint': 'semuav.perception.DetectorOnlyBackend.infer',
                                     'correction': 'normalize_event_to_insufficient_evidence_before_existing_action_mapping_when_evidence_is_false',
                                     'defect_kind': 'positive_event_survives_sufficient_evidence_false',
                                     'mission_ontology_changed': False,
                                     'predecessor_source_or_receipts_mutated': False,
                                     'previously_schema_valid_predictions_changed': False,
                                     'registered_dataset_or_split_changed': False,
                                     'registered_models_changed': False,
                                     'requirements_changed': False,
                                     'schema_version': 'semuav.visdrone-v2r13-low-evidence-packet-correction.v1',
                                     'scientific_contract_changed': False,
                                     'semantic_schema_changed': False,
                                     'thresholds_changed': False},
     'v2r12_derived_scientific_outputs_reused': False,
     'v2r13_derived_results_recomputation_required': True,
     'v2r13_downstream_results_recomputation_required': True,
     'v2r13_fresh_runtime_smoke_required': True,
     'v2r13_raw_experiment1_recovery_authorized': True,
     'v2r13_raw_experiment1_recovery_schema': 'semuav.visdrone-v2r9-exp1-recovery.v1'}
    value["correction_record_sha256"] = correction_record_sha256
    return value


def _v2r13_authenticated_predecessor_fixture():
    return {'active_model_lock_path': '/data/Lane Detection '
                               'Datasets/musta/semuav_vlm/outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/MODEL_LOCK.json',
     'active_model_lock_sha256': '16bfbc1d2365ac0a0adb2ff20d2f5785fa3f1b05675159304aedf4c6f7c1cd2a',
     'active_model_set_sha256': '6f83118bb62cd68c98e9ee2afd947022c04088b9329a0eeef832eba17c2fb08c',
     'archive_member_count': 113,
     'archive_path': '/data/Lane Detection '
                     'Datasets/musta/semuav_vlm/incoming/SemUAV-VLM-VisDrone-Paper-Study-20260921-v2r12.zip',
     'archive_sha256': 'a53600fff9f50174af6c4ed17352900981cd9684333c4a54749a427be7a1bdc3',
     'builder_sha256': '799c5fc179fd6e5c4e62ec6dae081528ba5cddbc123a5647c526f80f1ebdbc00',
     'checksum_inventory_sha256': '8fd90b8ba5f32bc0da435119db9bff09fd7c908b6388887e67cfa430b8266285',
     'deployer_path': '/data/Lane Detection '
                      'Datasets/musta/semuav_vlm/incoming/deploy_semuav_visdrone_v2r12_20260921.sh',
     'deployer_sha256': '622699a1634429d30e8bd4a1a9817ead0b28f99fad46b2fc36726d6246552d4b',
     'deployment_receipt': '/data/Lane Detection '
                           'Datasets/musta/semuav_vlm/private/visdrone_v2r12_deployment/DEPLOYMENT_READY.json',
     'deployment_receipt_sha256': '889850427153f9559fc038e2f10df174874dd59e8bff0473f61dcce1b62d2463',
     'experiment1_recovery_artifact_count': 132,
     'experiment1_recovery_receipt': '/data/Lane Detection '
                                     'Datasets/musta/semuav_vlm/private/visdrone_v2r10_deployment/V2R9_EXP1_RECOVERY_SOURCE.json',
     'experiment1_recovery_receipt_bytes': 37416,
     'experiment1_recovery_receipt_sha256': '6ca88cb5e82c8b33e27fd229443e233a48323d91fd73d0e79ec51ff8fc61a608',
     'experiment1_recovery_row_count': 4152,
     'inference_environment_fingerprint_path': '/data/Lane Detection '
                                               'Datasets/musta/semuav_vlm/outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/INFERENCE_ENVIRONMENT_FINGERPRINT.json',
     'inference_environment_fingerprint_sha256': '39d293199a08ada1465ca335fb9481d7e7548ad20d4ab26493c68910ed771f81',
     'inference_pip_freeze_path': '/data/Lane Detection '
                                  'Datasets/musta/semuav_vlm/outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/INFERENCE_PIP_FREEZE.txt',
     'inference_pip_freeze_sha256': '5b6aacbcd2b891c1bf968511d17b3385aff2233e99d0bae440f81bc7f977d5d2',
     'model_lock_derivation_path': '/data/Lane Detection '
                                   'Datasets/musta/semuav_vlm/outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/MODEL_LOCK_DERIVATION.json',
     'model_lock_derivation_sha256': '14b0623dd68166608a7f27ee55a2e87ee945ae3fd28e8b72daff472c4ff99de7',
     'model_status': {'qwen3_vl_2b': 'PASS',
                      'qwen3_vl_8b_upper_bound': 'PASS',
                      'smolvlm2_2b': 'PASS'},
     'notebook_sha256': '06b47108768a7ce566ddf480f285148f15797a87055d46e6d179e15aa69a3e80',
     'payload_file_count': 111,
     'payload_total_uncompressed_bytes': 4188275,
     'release_id': 'release_20260921_visdrone_paper_v2r12',
     'release_manifest_sha256': '69bfb381877a419c698ba381f2af637208462ae764dd2f1b8abbb9c8c92729d5',
     'release_path': '/data/Lane Detection '
                     'Datasets/musta/semuav_vlm/releases/release_20260921_visdrone_paper_v2r12/SemUAV-VLM-VisDrone-Paper-Study',
     'requirements_sha256': '028a2757d2e1565db90ff26a26fd6e564bd592c868ed093d632cb930d438dfa5',
     'reused_artifacts_receipt_path': '/data/Lane Detection '
                                      'Datasets/musta/semuav_vlm/outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/REUSED_V1_FROZEN_ARTIFACTS.json',
     'reused_artifacts_receipt_sha256': '68839ef0f3d3cf5b52a4b986c4fb24e0238f40b3bce44e13e45d6106ca22b811',
     'run_name': 'semuav_visdrone_v2r12',
     'runtime_smoke_all_passed': True,
     'runtime_smoke_failure_count': 0,
     'runtime_smoke_input_path': '/data/Lane Detection '
                                 'Datasets/musta/semuav_vlm/outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/MODEL_RUNTIME_SMOKE_INPUT.json',
     'runtime_smoke_input_sha256': 'f08a98d1b5f094c1254bbd1ce7a80d13de24d74bc544ddbe2151ff04f7160b1f',
     'runtime_smoke_isolation_schema': 'semuav.runtime-smoke-subprocess.v2',
     'runtime_smoke_pass_count': 3,
     'runtime_smoke_report_path': '/data/Lane Detection '
                                  'Datasets/musta/semuav_vlm/outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/MODEL_RUNTIME_SMOKE.json',
     'runtime_smoke_report_sha256': '0d43d5045adb0d3bea3aca2650ca8c832c076e60932c6662dbcd0abb13799fb2',
     'runtime_smoke_run': True,
     'runtime_smoke_schema_version': 4,
     'scientific_environment_seal_path': '/data/Lane Detection '
                                         'Datasets/musta/semuav_vlm/outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/INFERENCE_ENVIRONMENT_SEAL.json',
     'scientific_environment_seal_sha256': 'fa7fbaaa0b16090e48724389b455ad5d8e21490d0af4e14f55d62557bfab81d7',
     'scientific_environment_sealed': True,
     'scientific_experiments_started': True,
     'scientific_validation_authorized': True,
     'source_deployment_ready': True,
     'trusted_code_cache_receipt_path': '/data/Lane Detection '
                                        'Datasets/musta/semuav_vlm/outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/TRUSTED_CODE_CACHE.json',
     'trusted_code_cache_receipt_sha256': 'c2b821ad0c04b6eb1bbc5d11f505361a55cea29ca046309f47f4b521a900e2f5',
     'unchanged_after_deployment_actions': True,
     'untouched_test_accessed': False,
     'validation_experiments_started': True}


def _v2r13_deployment_receipt_fixture(
    *,
    release_manifest: dict[str, object],
    deployment_receipt_sha256: str = "0" * 64,
    artifact_records: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the exact 52-key v2r13 receipt envelope for regression tests.

    Production deployment does not call this helper.  The deployer independently
    creates and replays the same envelope.  Tests supply materialized path/hash
    records through ``artifact_records`` and use ``deployment_receipt_sha256`` as
    the externally pinned digest after serializing the returned object; the digest
    is deliberately not serialized as an unbound 53rd field.
    """

    if (
        not isinstance(deployment_receipt_sha256, str)
        or len(deployment_receipt_sha256) != 64
        or any(character not in "0123456789abcdef" for character in deployment_receipt_sha256)
    ):
        raise ValueError("deployment_receipt_sha256 must be one lowercase SHA-256")
    manifest_fields = {
        "release_id", "run_name", "authenticated_predecessor_v2r9",
        "authenticated_predecessor_v2r10", "authenticated_predecessor_v2r11",
        "authenticated_predecessor_v2r12",
        "predecessor_observed_terminal_boundary", "model_set_amendment",
        "temporal_feasibility_amendment", "source_compatibility",
    }
    if not isinstance(release_manifest, dict) or not manifest_fields.issubset(
        release_manifest
    ):
        raise ValueError("release_manifest lacks v2r13 receipt-bound fields")
    if (
        release_manifest["release_id"]
        != "release_20260921_visdrone_paper_v2r13"
        or release_manifest["run_name"] != "semuav_visdrone_v2r13"
    ):
        raise ValueError("release_manifest is not the exact v2r13 identity")
    manifest_boundary = release_manifest["predecessor_observed_terminal_boundary"]
    if not isinstance(manifest_boundary, dict):
        raise ValueError("release_manifest predecessor boundary is not an object")
    correction_sha256 = manifest_boundary.get("correction_record_sha256")
    if manifest_boundary != _v2r13_predecessor_observed_terminal_boundary_fixture(
        correction_sha256  # type: ignore[arg-type]
    ):
        raise ValueError("release_manifest predecessor boundary is not canonical")

    if artifact_records is None:
        artifact_records = {
            "created_at_utc": "2026-09-21T00:00:00Z",
            "release_root": "/fixture/release_20260921_visdrone_paper_v2r13",
            "deployment_mode": "NEW_ATOMIC_DEPLOYMENT",
            "release": {},
            "installer": {},
            "deployment_script": {},
            "builder": {},
            "source_release": {},
            "authenticated_base_v2r3": {},
            "candidate_environment": {},
            "unit_tests": {},
            "jupyter_kernel": {},
            "jupyter_launch_target": {},
            "trusted_code_cache_namespace": {},
            "attempt_directory": "/fixture/attempt",
            "deployment_log": {},
            "experiment1_recovery": {},
            "predecessor_v2r12_state": {},
        }
    if (
        not isinstance(artifact_records, dict)
        or set(artifact_records) != V2R13_DEPLOYMENT_RECEIPT_FIXTURE_RECORD_KEYS
    ):
        raise ValueError("artifact_records must supply the exact fixture record set")

    receipt = {
        "schema_version": "semuav.visdrone-v2r13-source-deployment.v1",
        "status": "READY_FOR_DEVELOPMENT_SMOKE_ONLY",
        "release_id": "release_20260921_visdrone_paper_v2r13",
        "run_name": "semuav_visdrone_v2r13",
        "scope": (
            "immutable_v2r13_source_sibling_tests_and_kernel_registration_only"
        ),
        "scientific_contract_changed": False,
        "candidate_requirements_changed": False,
        "package_installation_performed": False,
        "candidate_builder_invoked": False,
        "candidate_environment_modified": False,
        "base_v2r3_release_modified": False,
        "predecessor_v2r9_release_modified": False,
        "predecessor_v2r10_release_modified": False,
        "predecessor_v2r11_release_modified": False,
        "predecessor_v2r12_release_modified": False,
        "scientific_validation_authorized": False,
        "scientific_validation_run": False,
        "scientific_experiments_run": False,
        "runtime_smoke_run": False,
        "untouched_test_run": False,
        "validation_or_test_images_or_labels_read": False,
        "gpu_accessed": False,
        "model_weights_loaded": False,
        "jupyter_launched": False,
        "release_revision": {
            "revision": "v2r13",
            "supersedes_release_id": "release_20260921_visdrone_paper_v2r12",
        },
        "authenticated_predecessor_v2r9": copy.deepcopy(
            release_manifest["authenticated_predecessor_v2r9"]
        ),
        "authenticated_predecessor_v2r10": copy.deepcopy(
            release_manifest["authenticated_predecessor_v2r10"]
        ),
        "authenticated_predecessor_v2r11": copy.deepcopy(
            release_manifest["authenticated_predecessor_v2r11"]
        ),
        "authenticated_predecessor_v2r12": copy.deepcopy(
            release_manifest["authenticated_predecessor_v2r12"]
        ),
        "predecessor_observed_terminal_boundary": copy.deepcopy(
            release_manifest["predecessor_observed_terminal_boundary"]
        ),
        "model_set_amendment": copy.deepcopy(
            release_manifest["model_set_amendment"]
        ),
        "temporal_feasibility_amendment": copy.deepcopy(
            release_manifest["temporal_feasibility_amendment"]
        ),
        "source_compatibility": copy.deepcopy(
            release_manifest["source_compatibility"]
        ),
        "next_gate": "FRESH_V2R13_THREE_MODEL_DEVELOPMENT_RUNTIME_SMOKE",
        **copy.deepcopy(artifact_records),
    }
    expected_keys = {
        "schema_version", "status", "created_at_utc", "release_id",
        "run_name", "release_root", "deployment_mode", "scope",
        "scientific_contract_changed", "candidate_requirements_changed",
        "package_installation_performed", "candidate_builder_invoked",
        "candidate_environment_modified", "base_v2r3_release_modified",
        "predecessor_v2r9_release_modified",
        "predecessor_v2r10_release_modified",
        "predecessor_v2r11_release_modified",
        "predecessor_v2r12_release_modified", "predecessor_v2r12_state",
        "scientific_validation_authorized", "scientific_validation_run",
        "scientific_experiments_run", "runtime_smoke_run",
        "untouched_test_run", "validation_or_test_images_or_labels_read",
        "gpu_accessed", "model_weights_loaded", "jupyter_launched",
        "release_revision", "release", "installer", "deployment_script",
        "builder", "source_release", "authenticated_base_v2r3",
        "authenticated_predecessor_v2r9",
        "authenticated_predecessor_v2r10",
        "authenticated_predecessor_v2r11",
        "authenticated_predecessor_v2r12",
        "predecessor_observed_terminal_boundary", "candidate_environment",
        "model_set_amendment", "temporal_feasibility_amendment",
        "source_compatibility", "unit_tests", "jupyter_kernel",
        "jupyter_launch_target", "trusted_code_cache_namespace",
        "attempt_directory", "deployment_log", "experiment1_recovery",
        "next_gate",
    }
    if len(expected_keys) != 52 or set(receipt) != expected_keys:
        raise AssertionError("v2r13 fixture receipt is not the exact 52-key envelope")
    return receipt


cells = [
    md("""
    # SemUAV-VLM: VisDrone paper study — inference source release v2r13

    **Mission-conditioned progressive representation selection for bandwidth-constrained
    urban UAV visual uplinks**

    This notebook is the v2r13 real-data orchestrator for the registered
    `visdrone_primary` scope. It uses the six official VisDrone 2019 DET/MOT
    release components, publisher boxes/tracks, immutable model snapshots,
    measured receiver paths, paired network trials, and fail-closed scientific
    claim gates. The implementation lives in the adjacent `semuav/` package so
    every long stage is testable, resumable, and hash-bound outside notebook state.

    **A clean Run All never crosses a scientific phase.** Only the phase selected
    in the parameter cell can execute, and execution is disabled by default.

    The inherited outcome-blind v2r7 amendment fixes validation at seven eligible
    MOT sequences for the unchanged 64-frame contiguous episode. v2r9 freshly
    recomputed that exact capacity, created or verified its temporal sample, and
    completed all 132 raw Experiment 1 detector/VLM shards (4,152 inference rows),
    calibration, model selection, and semantic-packet construction. Its first native
    Experiment 2 receiver plan then stopped because the registered L0 heartbeat
    intentionally carries no action while the v2r9 receiver treated every semantic
    packet as action-bearing. v2r10 corrected that receiver path: it validates the
    complete L0 heartbeat and maps it to the already registered `NO_ACTION` outcome,
    and it binds every pixel-receiver detector request to authoritative pre-shared
    capture time and flight telemetry. v2r13 inherits that correction unchanged.
    v2r10 later stopped in software acceptance before any validation experiment
    executed because its generic unit-test child inherited the parent-only recovery
    authorization. v2r11 removed exactly those three recovery variables from the
    child; v2r13 inherits that isolation unchanged. v2r11 then stopped before its
    runtime smoke because the immediate-predecessor boundary was checked against
    the older v2r9 schema, a routing defect corrected by v2r12. After its
    fresh smoke passed, v2r12 reached the native visual receiver but weak detector
    evidence retained a positive event and violated the unchanged packet schema.
    v2r13 normalizes that event to `insufficient_evidence` before the existing
    `REVIEW` mapping. Previously valid raw predictions remain unchanged. It may import only
    the exact, hash-inventoried raw v2r9 inference shards; all derived metrics, selection,
    semantic packets, receiver caches, and downstream results are recomputed. v2r13
    also requires another fresh exact-capacity recomputation before accepting or
    writing its own sample. With twelve validation traces this yields
    exactly 84
    fully paired validation cells. The untouched-test temporal cohort remains 15
    sequences by 30 traces, or 450 group-by-trace cells; it remains below the 30
    groups and 500 cells required for H2 and cannot meet H3's separate 140
    positive-group and 73 negative-group minima.
    H2 and H3 are therefore structurally `NOT_ESTIMABLE` in this campaign;
    their measured endpoints remain descriptive. H1 is confirmatory, while H4
    is confirmatory only when both frozen semantic-denominator gates pass at the
    registered calibrated-confidence threshold of 0.80.
    """),
    md("""
    ## Source-truth endpoint and prohibited-claim boundary

    The primary endpoint is the first **correct ground-station notification before
    the registered deadline measured from capture**. `RESPOND` means immediate
    notification only; it does not mean dispatch, collision detection, or a claim
    about physical distance.

    Publisher truth supports sensor-space statements about 2-D boxes, tracks,
    category, truncation, occlusion, registered image-plane ROI occupancy, and
    image-plane overlap. It does **not** support claims of accident detection,
    emergency dispatch, road obstruction, physical proximity, wildfire, or
    cross-dataset out-of-distribution generalization. Those claims remain prohibited
    even when a plot looks favorable. Optional human construct-validity results are
    reported separately and require two blinded annotators plus adjudication.
    """),
    md("""
    ## Frozen study sequence

    | Phase | Data visible | Permitted mutation | Scientific output |
    |---|---|---|---|
    | `prepare` | one registered train canary + immutable v1 receipts | reuse receipt, inference lock, three-model smoke | none |
    | `validation` | train/calibration/validation | calibration, selection, receiver caches, protocol freeze | development only |
    | `test` | untouched test-dev, once | one complete paired confirmatory run and immutable ledger | H1–H4 |
    | `report` | finalized hashed result artifacts only | figures, tables, manifests | read-only recomputation |
    | `embedded` | finalized paper artifacts only | frozen H5 status rendering | `NOT_APPLICABLE_NO_TARGET_EVIDENCE` for this paper |

    The reused v1 publisher train material was split by independent sequence/group
    into model-training and calibration material. Publisher validation remains
    outcome-blind validation and is not read by the v2r13 prepare/smoke gate.
    Publisher test-dev remains path-only until the frozen protocol is explicitly
    authorized exactly once.
    """),
    md("""
    ## NYUAD execution contract

    Run on the compute workstation, not a login node. Keep the environment,
    checkpoints, model caches, temporary files, data, and outputs under `/data`.
    Expose physical GPU 1 with `CUDA_VISIBLE_DEVICES=1`; inside Python it appears as
    `cuda:0`. Use the `Python (SemUAV-VLM 2026-09-21 v2r13)` kernel and verify the
    printed `sys.executable` path. The v1 detector-training environment is
    historical evidence only and must never launch this notebook. The exact terminal and browser
    commands are in `VISDRONE_PAPER_RUN_GUIDE.md`.
    """),
    code("""
    # EDIT ONLY THIS CELL. Save, then use Kernel -> Restart Kernel and Run All Cells.
    import sys
    V2R13_EXPECTED_CANDIDATE_PYTHON_TEXT = (
        "/data/Lane Detection Datasets/musta/semuav_vlm/venvs/"
        "release_20260912_visdrone_inference_candidate_v2r3/bin/python"
    )
    V2R13_EXPECTED_CANDIDATE_PREFIX_TEXT = (
        "/data/Lane Detection Datasets/musta/semuav_vlm/venvs/"
        "release_20260912_visdrone_inference_candidate_v2r3"
    )
    if (
        sys.executable != V2R13_EXPECTED_CANDIDATE_PYTHON_TEXT
        or sys.prefix != V2R13_EXPECTED_CANDIDATE_PREFIX_TEXT
        or sys.flags.isolated != 1
        or sys.flags.dont_write_bytecode != 1
        or sys.dont_write_bytecode is not True
    ):
        raise RuntimeError(
            "Select the exact isolated v2r13 kernelspec before running any "
            "notebook cell"
        )
    V2R13_EXPECTED_RELEASE_ROOT_TEXT = (
        "/data/Lane Detection Datasets/musta/semuav_vlm/releases/"
        "release_20260921_visdrone_paper_v2r13/"
        "SemUAV-VLM-VisDrone-Paper-Study"
    )
    sys.path[:] = [
        entry for entry in sys.path
        if str(entry) not in {"", "."}
        and str(entry).rstrip("/") != V2R13_EXPECTED_RELEASE_ROOT_TEXT
    ]

    import ast
    import copy
    import hashlib
    import io
    import json
    import os
    import stat
    import zipfile
    from pathlib import Path, PurePosixPath

    PROJECT_ROOT = Path(os.environ.get("SEMUAV_PROJECT_ROOT", Path.cwd())).resolve()
    CONFIG_PATH = PROJECT_ROOT / "configs" / "visdrone_paper.yaml"
    EXPECTED_STUDY_STORAGE_ROOT = Path(
        "/data/Lane Detection Datasets/musta/semuav_vlm"
    )
    STUDY_STORAGE_ROOT = Path(os.environ.get(
        "SEMUAV_STUDY_ROOT", str(EXPECTED_STUDY_STORAGE_ROOT)
    )).resolve()

    # Fail before importing any release Python module if the launch shell can
    # influence imports or if executable bytecode exists in the source tree.
    required_absent_environment = (
        "BASH_ENV", "ENV", "LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONHOME",
        "PYTHONPATH",
        "SEMUAV_QWEN_REVISION", "SEMUAV_SMOLVLM_REVISION",
        "SEMUAV_MOONDREAM_REVISION", "SEMUAV_UPPER_BOUND_REVISION",
        "SEMUAV_MOONDREAM_TOKENIZER_LOCK",
        "SEMUAV_ENABLE_V2R9_EXP1_RECOVERY",
        "SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT",
        "SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT_SHA256",
    )
    contaminated_environment = [
        name for name in required_absent_environment
        if os.environ.get(name) is not None
    ]
    if contaminated_environment:
        raise ValueError(
            f"Unsafe inherited launch environment: {contaminated_environment}"
        )
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise ValueError("PYTHONNOUSERSITE must be exactly 1")
    deployment_receipt_sha256 = os.environ.get(
        "SEMUAV_V2R13_DEPLOYMENT_RECEIPT_SHA256"
    )
    V2R13_PREIMPORT_SOURCE_VERIFIED = False
    if (
        deployment_receipt_sha256 is None
        or len(deployment_receipt_sha256) != 64
        or any(character not in "0123456789abcdef" for character in deployment_receipt_sha256)
    ):
        raise ValueError(
            "SEMUAV_V2R13_DEPLOYMENT_RECEIPT_SHA256 must be one lowercase "
            "SHA-256 before notebook source authentication"
        )

    # Authenticate the complete deployed v2r13 source tree before putting it on
    # sys.path or importing any release module.  The deployment receipt digest
    # is supplied out-of-band by the operator; the receipt binds the fixed
    # release manifest, checksum inventory, and sealed source archive.  Every
    # live payload must be byte-identical to that inventory except the notebook,
    # whose authored content may differ only in the ten approved parameter RHS
    # literals (outputs and execution counts are non-authored state).
    preimported_release_modules = sorted(
        name for name in sys.modules
        if name == "semuav" or name.startswith("semuav.")
    )
    if preimported_release_modules:
        raise ValueError(
            "V2R13 source bootstrap rejected: release modules were imported "
            f"before source authentication: {preimported_release_modules}"
        )
    expected_release_root = (
        EXPECTED_STUDY_STORAGE_ROOT / "releases"
        / "release_20260921_visdrone_paper_v2r13"
        / "SemUAV-VLM-VisDrone-Paper-Study"
    )
    expected_receipt_path = (
        EXPECTED_STUDY_STORAGE_ROOT / "private"
        / "visdrone_v2r13_deployment" / "DEPLOYMENT_READY.json"
    )
    expected_archive_path = (
        EXPECTED_STUDY_STORAGE_ROOT / "incoming"
        / "SemUAV-VLM-VisDrone-Paper-Study-20260921-v2r13.zip"
    )
    manifest_path = expected_release_root / "RELEASE_MANIFEST.json"
    checksum_path = expected_release_root / "RELEASE_MANIFEST_SHA256.txt"
    notebook_relative = "SemUAV_VLM_VisDrone_Paper_Study.ipynb"
    checkpoint_relative = (
        ".ipynb_checkpoints/"
        "SemUAV_VLM_VisDrone_Paper_Study-checkpoint.ipynb"
    )
    approved_parameter_assignment_order = (
        "PHASE", "EXECUTE_PHASE", "FREEZE_PROTOCOL",
        "AUTHORIZE_TEST_ONCE", "RUN_NAME", "RESUME", "DEVICE",
        "RUN_MODEL_SMOKE", "BUILD_NATIVE_RECEIVER_CACHE",
        "BUILD_FEATURE_CACHE",
    )
    approved_parameter_assignments = set(
        approved_parameter_assignment_order
    )

    def bootstrap_fail(message):
        raise ValueError(f"V2R13 source bootstrap rejected: {message}")

    def duplicate_free_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                bootstrap_fail(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def exact_sha256(value, label):
        digest = str(value)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            bootstrap_fail(f"{label} is not one lowercase SHA-256")
        return digest

    def read_exact_regular(path, label):
        path = Path(path)
        try:
            before = path.lstat()
        except OSError as exc:
            bootstrap_fail(f"{label} is missing: {path}: {exc}")
        if (
            path.is_symlink()
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or path.resolve() != path
        ):
            bootstrap_fail(
                f"{label} is not an exact single-link regular file: {path}"
            )
        try:
            with path.open("rb") as handle:
                opened = os.fstat(handle.fileno())
                if (
                    opened.st_dev != before.st_dev
                    or opened.st_ino != before.st_ino
                    or not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                ):
                    bootstrap_fail(f"{label} changed while opening: {path}")
                payload = handle.read()
            after = path.lstat()
        except OSError as exc:
            bootstrap_fail(f"{label} could not be read: {path}: {exc}")
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_nlink != 1
        ):
            bootstrap_fail(f"{label} changed while being read: {path}")
        return payload

    def decode_object(payload, label):
        try:
            value = json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=duplicate_free_object,
            )
        except (UnicodeError, json.JSONDecodeError) as exc:
            bootstrap_fail(f"{label} is not duplicate-free UTF-8 JSON: {exc}")
        if not isinstance(value, dict):
            bootstrap_fail(f"{label} is not one JSON object")
        return value

    def normalized_parameter_source(source, label, expected_source=None):
        try:
            module = ast.parse(source)
        except SyntaxError as exc:
            bootstrap_fail(f"{label} parameter cell is invalid Python: {exc}")

        def assignments(parsed, parsed_label):
            result = {}
            for statement in parsed.body:
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id
                    in approved_parameter_assignments
                ):
                    name = statement.targets[0].id
                    if name in result:
                        bootstrap_fail(
                            f"{parsed_label} repeats approved parameter {name}"
                        )
                    result[name] = statement
            stored_names = [
                node.id for node in ast.walk(parsed)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Store)
                and node.id in approved_parameter_assignments
            ]
            if (
                tuple(result) != approved_parameter_assignment_order
                or sorted(stored_names)
                != sorted(approved_parameter_assignments)
            ):
                bootstrap_fail(
                    f"{parsed_label} approved parameter assignments changed"
                )
            return result

        observed = assignments(module, label)
        expected = None
        if expected_source is not None:
            try:
                expected_module = ast.parse(expected_source)
            except SyntaxError as exc:
                bootstrap_fail(
                    f"{label} sealed parameter cell is invalid Python: {exc}"
                )
            expected = assignments(expected_module, f"{label} sealed")
        bool_parameters = {
            "EXECUTE_PHASE", "FREEZE_PROTOCOL", "AUTHORIZE_TEST_ONCE",
            "RESUME", "RUN_MODEL_SMOKE", "BUILD_NATIVE_RECEIVER_CACHE",
            "BUILD_FEATURE_CACHE",
        }
        for name, statement in observed.items():
            value = statement.value
            matches_sealed = expected is None or ast.dump(
                value, include_attributes=False,
            ) == ast.dump(expected[name].value, include_attributes=False)
            safe_literal = (
                isinstance(value, ast.Constant)
                and (
                    (name in bool_parameters and type(value.value) is bool)
                    or (
                        name == "PHASE"
                        and value.value in {
                            "prepare", "validation", "test", "report",
                            "embedded",
                        }
                    )
                    or (
                        name == "RUN_NAME"
                        and value.value == "semuav_visdrone_v2r13"
                    )
                    or (name == "DEVICE" and value.value == "cuda:0")
                )
            )
            if not matches_sealed and not safe_literal:
                bootstrap_fail(
                    f"{label} parameter {name} has an unsafe right-hand side"
                )
        encoded = source.encode("utf-8")
        lines = encoded.splitlines(keepends=True)
        replacements = []
        for statement in observed.values():
            value = statement.value
            start = (
                sum(len(line) for line in lines[: value.lineno - 1])
                + value.col_offset
            )
            end = (
                sum(len(line) for line in lines[: value.end_lineno - 1])
                + value.end_col_offset
            )
            replacements.append((start, end))
        for start, end in sorted(replacements, reverse=True):
            encoded = (
                encoded[:start] + b"<APPROVED_PARAMETER_RHS>"
                + encoded[end:]
            )
        return encoded.decode("utf-8")

    def authored_contract(notebook, label, expected_parameter_source=None):
        if not isinstance(notebook, dict) or not isinstance(
            notebook.get("cells"), list
        ):
            bootstrap_fail(f"{label} is not one notebook object")
        contract = copy.deepcopy(notebook)
        metadata = contract.get("metadata", {})
        if not isinstance(metadata, dict):
            bootstrap_fail(f"{label} notebook metadata is malformed")
        for transient in ("language_info", "widgets", "signature"):
            metadata.pop(transient, None)
        parameter_cells = 0
        for cell in contract["cells"]:
            if not isinstance(cell, dict):
                bootstrap_fail(f"{label} contains a non-object cell")
            metadata = cell.get("metadata", {})
            if not isinstance(metadata, dict):
                bootstrap_fail(f"{label} cell metadata is malformed")
            for transient in (
                "trusted", "execution", "ExecuteTime", "collapsed",
                "scrolled",
            ):
                metadata.pop(transient, None)
            tags = metadata.get("tags", [])
            if not isinstance(tags, list) or any(
                not isinstance(tag, str) for tag in tags
            ):
                bootstrap_fail(f"{label} cell tags are malformed")
            raw_source = cell.get("source", "")
            if isinstance(raw_source, list):
                if any(not isinstance(part, str) for part in raw_source):
                    bootstrap_fail(
                        f"{label} cell source contains a non-string"
                    )
                raw_source = "".join(raw_source)
            if not isinstance(raw_source, str):
                bootstrap_fail(f"{label} cell source is malformed")
            cell["source"] = raw_source
            if "parameters" in tags:
                parameter_cells += 1
                cell["source"] = normalized_parameter_source(
                    raw_source, label, expected_parameter_source,
                )
            if cell.get("cell_type") == "code":
                cell.pop("execution_count", None)
                cell.pop("outputs", None)
        if parameter_cells != 1:
            bootstrap_fail(
                f"{label} notebook metadata/parameter cell changed"
            )
        return contract

    if STUDY_STORAGE_ROOT != EXPECTED_STUDY_STORAGE_ROOT:
        bootstrap_fail("SEMUAV_STUDY_ROOT is not the fixed study root")
    if (
        PROJECT_ROOT != expected_release_root
        or expected_release_root.is_symlink()
        or not expected_release_root.is_dir()
        or expected_release_root.resolve() != expected_release_root
    ):
        bootstrap_fail("PROJECT_ROOT is not the exact v2r13 release root")

    receipt_bytes = read_exact_regular(
        expected_receipt_path, "deployment receipt",
    )
    if hashlib.sha256(receipt_bytes).hexdigest() != deployment_receipt_sha256:
        bootstrap_fail("deployment receipt digest differs from operator pin")
    receipt = decode_object(receipt_bytes, "deployment receipt")
    expected_receipt_keys = {
        "schema_version", "status", "created_at_utc", "release_id",
        "run_name", "release_root", "deployment_mode", "scope",
        "scientific_contract_changed", "candidate_requirements_changed",
        "package_installation_performed", "candidate_builder_invoked",
        "candidate_environment_modified", "base_v2r3_release_modified",
        "predecessor_v2r9_release_modified",
        "predecessor_v2r10_release_modified",
        "predecessor_v2r11_release_modified",
        "predecessor_v2r12_release_modified", "predecessor_v2r12_state",
        "scientific_validation_authorized",
        "scientific_validation_run", "scientific_experiments_run",
        "runtime_smoke_run", "untouched_test_run",
        "validation_or_test_images_or_labels_read", "gpu_accessed",
        "model_weights_loaded", "jupyter_launched", "release_revision",
        "release", "installer", "deployment_script", "builder",
        "source_release", "authenticated_base_v2r3",
        "authenticated_predecessor_v2r9", "authenticated_predecessor_v2r10",
        "authenticated_predecessor_v2r11",
        "authenticated_predecessor_v2r12",
        "predecessor_observed_terminal_boundary", "candidate_environment",
        "model_set_amendment", "temporal_feasibility_amendment",
        "source_compatibility", "unit_tests", "jupyter_kernel",
        "jupyter_launch_target", "trusted_code_cache_namespace",
        "attempt_directory", "deployment_log", "experiment1_recovery",
        "next_gate",
    }
    if len(expected_receipt_keys) != 52 or set(receipt) != expected_receipt_keys:
        bootstrap_fail("deployment receipt 52-key envelope changed")
    release = receipt.get("release")
    source_release = receipt.get("source_release")
    if not isinstance(release, dict) or not isinstance(source_release, dict):
        bootstrap_fail("deployment receipt release records are malformed")
    if (
        receipt.get("schema_version")
        != "semuav.visdrone-v2r13-source-deployment.v1"
        or receipt.get("status") != "READY_FOR_DEVELOPMENT_SMOKE_ONLY"
        or receipt.get("release_id")
        != "release_20260921_visdrone_paper_v2r13"
        or receipt.get("run_name") != "semuav_visdrone_v2r13"
        or Path(str(receipt.get("release_root", "")))
        != expected_release_root
        or receipt.get("deployment_mode") not in {
            "NEW_ATOMIC_DEPLOYMENT",
            "REVERIFY_EXISTING_EXACT_DEPLOYMENT",
        }
        or receipt.get("scope")
        != "immutable_v2r13_source_sibling_tests_and_kernel_registration_only"
        or receipt.get("runtime_smoke_run") is not False
        or receipt.get("scientific_contract_changed") is not False
        or receipt.get("candidate_requirements_changed") is not False
        or receipt.get("package_installation_performed") is not False
        or receipt.get("candidate_builder_invoked") is not False
        or receipt.get("candidate_environment_modified") is not False
        or receipt.get("base_v2r3_release_modified") is not False
        or receipt.get("predecessor_v2r9_release_modified") is not False
        or receipt.get("predecessor_v2r10_release_modified") is not False
        or receipt.get("predecessor_v2r11_release_modified") is not False
        or receipt.get("predecessor_v2r12_release_modified") is not False
        or receipt.get("scientific_validation_authorized") is not False
        or receipt.get("validation_or_test_images_or_labels_read") is not False
        or receipt.get("scientific_validation_run") is not False
        or receipt.get("scientific_experiments_run") is not False
        or receipt.get("untouched_test_run") is not False
        or receipt.get("gpu_accessed") is not False
        or receipt.get("model_weights_loaded") is not False
        or receipt.get("jupyter_launched") is not False
        or receipt.get("next_gate")
        != "FRESH_V2R13_THREE_MODEL_DEVELOPMENT_RUNTIME_SMOKE"
        or release.get("path") != str(expected_release_root)
        or release.get("release_id")
        != "release_20260921_visdrone_paper_v2r13"
        or release.get("run_name") != "semuav_visdrone_v2r13"
        or release.get("archive_name")
        != "SemUAV-VLM-VisDrone-Paper-Study-20260921-v2r13.zip"
        or Path(str(release.get("archive_source", "")))
        != expected_archive_path
        or Path(str(source_release.get("archive_path", "")))
        != expected_archive_path
    ):
        bootstrap_fail("deployment receipt identity/safety state changed")
    recovery = receipt.get("experiment1_recovery")
    expected_recovery_path = (
        EXPECTED_STUDY_STORAGE_ROOT / "private"
        / "visdrone_v2r10_deployment"
        / "V2R9_EXP1_RECOVERY_SOURCE.json"
    )
    expected_sealed_predecessor = {
        'active_model_lock_path': str(EXPECTED_STUDY_STORAGE_ROOT / 'outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/MODEL_LOCK.json'),
        'active_model_lock_sha256': '16bfbc1d2365ac0a0adb2ff20d2f5785fa3f1b05675159304aedf4c6f7c1cd2a',
        'active_model_set_sha256': '6f83118bb62cd68c98e9ee2afd947022c04088b9329a0eeef832eba17c2fb08c',
        'archive_member_count': 113,
        'archive_path': str(EXPECTED_STUDY_STORAGE_ROOT / 'incoming/SemUAV-VLM-VisDrone-Paper-Study-20260921-v2r12.zip'),
        'archive_sha256': 'a53600fff9f50174af6c4ed17352900981cd9684333c4a54749a427be7a1bdc3',
        'builder_sha256': '799c5fc179fd6e5c4e62ec6dae081528ba5cddbc123a5647c526f80f1ebdbc00',
        'checksum_inventory_sha256': '8fd90b8ba5f32bc0da435119db9bff09fd7c908b6388887e67cfa430b8266285',
        'deployer_path': str(EXPECTED_STUDY_STORAGE_ROOT / 'incoming/deploy_semuav_visdrone_v2r12_20260921.sh'),
        'deployer_sha256': '622699a1634429d30e8bd4a1a9817ead0b28f99fad46b2fc36726d6246552d4b',
        'deployment_receipt': str(EXPECTED_STUDY_STORAGE_ROOT / 'private/visdrone_v2r12_deployment/DEPLOYMENT_READY.json'),
        'deployment_receipt_sha256': '889850427153f9559fc038e2f10df174874dd59e8bff0473f61dcce1b62d2463',
        'experiment1_recovery_artifact_count': 132,
        'experiment1_recovery_receipt': str(EXPECTED_STUDY_STORAGE_ROOT / 'private/visdrone_v2r10_deployment/V2R9_EXP1_RECOVERY_SOURCE.json'),
        'experiment1_recovery_receipt_bytes': 37416,
        'experiment1_recovery_receipt_sha256': '6ca88cb5e82c8b33e27fd229443e233a48323d91fd73d0e79ec51ff8fc61a608',
        'experiment1_recovery_row_count': 4152,
        'inference_environment_fingerprint_path': str(EXPECTED_STUDY_STORAGE_ROOT / 'outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/INFERENCE_ENVIRONMENT_FINGERPRINT.json'),
        'inference_environment_fingerprint_sha256': '39d293199a08ada1465ca335fb9481d7e7548ad20d4ab26493c68910ed771f81',
        'inference_pip_freeze_path': str(EXPECTED_STUDY_STORAGE_ROOT / 'outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/INFERENCE_PIP_FREEZE.txt'),
        'inference_pip_freeze_sha256': '5b6aacbcd2b891c1bf968511d17b3385aff2233e99d0bae440f81bc7f977d5d2',
        'model_lock_derivation_path': str(EXPECTED_STUDY_STORAGE_ROOT / 'outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/MODEL_LOCK_DERIVATION.json'),
        'model_lock_derivation_sha256': '14b0623dd68166608a7f27ee55a2e87ee945ae3fd28e8b72daff472c4ff99de7',
        'model_status': {'qwen3_vl_2b': 'PASS', 'qwen3_vl_8b_upper_bound': 'PASS', 'smolvlm2_2b': 'PASS'},
        'notebook_sha256': '06b47108768a7ce566ddf480f285148f15797a87055d46e6d179e15aa69a3e80',
        'payload_file_count': 111,
        'payload_total_uncompressed_bytes': 4188275,
        'release_id': 'release_20260921_visdrone_paper_v2r12',
        'release_manifest_sha256': '69bfb381877a419c698ba381f2af637208462ae764dd2f1b8abbb9c8c92729d5',
        'release_path': str(EXPECTED_STUDY_STORAGE_ROOT / 'releases/release_20260921_visdrone_paper_v2r12/SemUAV-VLM-VisDrone-Paper-Study'),
        'requirements_sha256': '028a2757d2e1565db90ff26a26fd6e564bd592c868ed093d632cb930d438dfa5',
        'reused_artifacts_receipt_path': str(EXPECTED_STUDY_STORAGE_ROOT / 'outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/REUSED_V1_FROZEN_ARTIFACTS.json'),
        'reused_artifacts_receipt_sha256': '68839ef0f3d3cf5b52a4b986c4fb24e0238f40b3bce44e13e45d6106ca22b811',
        'run_name': 'semuav_visdrone_v2r12',
        'runtime_smoke_all_passed': True,
        'runtime_smoke_failure_count': 0,
        'runtime_smoke_input_path': str(EXPECTED_STUDY_STORAGE_ROOT / 'outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/MODEL_RUNTIME_SMOKE_INPUT.json'),
        'runtime_smoke_input_sha256': 'f08a98d1b5f094c1254bbd1ce7a80d13de24d74bc544ddbe2151ff04f7160b1f',
        'runtime_smoke_isolation_schema': 'semuav.runtime-smoke-subprocess.v2',
        'runtime_smoke_pass_count': 3,
        'runtime_smoke_report_path': str(EXPECTED_STUDY_STORAGE_ROOT / 'outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/MODEL_RUNTIME_SMOKE.json'),
        'runtime_smoke_report_sha256': '0d43d5045adb0d3bea3aca2650ca8c832c076e60932c6662dbcd0abb13799fb2',
        'runtime_smoke_run': True,
        'runtime_smoke_schema_version': 4,
        'scientific_environment_seal_path': str(EXPECTED_STUDY_STORAGE_ROOT / 'outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/INFERENCE_ENVIRONMENT_SEAL.json'),
        'scientific_environment_seal_sha256': 'fa7fbaaa0b16090e48724389b455ad5d8e21490d0af4e14f55d62557bfab81d7',
        'scientific_environment_sealed': True,
        'scientific_experiments_started': True,
        'scientific_validation_authorized': True,
        'source_deployment_ready': True,
        'trusted_code_cache_receipt_path': str(EXPECTED_STUDY_STORAGE_ROOT / 'outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/TRUSTED_CODE_CACHE.json'),
        'trusted_code_cache_receipt_sha256': 'c2b821ad0c04b6eb1bbc5d11f505361a55cea29ca046309f47f4b521a900e2f5',
        'unchanged_after_deployment_actions': True,
        'untouched_test_accessed': False,
        'validation_experiments_started': True,
    }
    if receipt.get("authenticated_predecessor_v2r12") != expected_sealed_predecessor:
        bootstrap_fail("authenticated v2r12 immediate predecessor record changed")
    predecessor_state = receipt.get("predecessor_v2r12_state")
    expected_state_paths = {
        "release_root": str(EXPECTED_STUDY_STORAGE_ROOT / "releases/release_20260921_visdrone_paper_v2r12/SemUAV-VLM-VisDrone-Paper-Study"),
        "output_root": str(EXPECTED_STUDY_STORAGE_ROOT / "outputs/visdrone_paper/semuav_visdrone_v2r12"),
        "cache_root": str(EXPECTED_STUDY_STORAGE_ROOT / ".cache/visdrone_paper_v2r12"),
        "trusted_code_cache_root": str(EXPECTED_STUDY_STORAGE_ROOT / ".cache/huggingface/modules-v2r12"),
    }
    expected_state_keys = set(expected_state_paths) | {
        "schema_version", "release_tree", "output_tree", "cache_tree",
        "trusted_code_cache_tree", "unchanged_after_deployment_actions",
    }
    if (not isinstance(predecessor_state, dict) or set(predecessor_state) != expected_state_keys
        or predecessor_state.get("schema_version") != "semuav.visdrone-v2r12-preserved-state.v1"
        or predecessor_state.get("unchanged_after_deployment_actions") is not True
        or any(predecessor_state.get(key) != value for key, value in expected_state_paths.items())):
        bootstrap_fail("The preserved v2r12 state record changed")
    for state_key in ("release_tree", "output_tree", "cache_tree", "trusted_code_cache_tree"):
        state_parts = str(predecessor_state[state_key]).split()
        if (len(state_parts) != 3 or len(state_parts[0]) != 64
            or any(c not in "0123456789abcdef" for c in state_parts[0])
            or not all(part.isascii() and part.isdigit() for part in state_parts[1:])
            or (state_key != "trusted_code_cache_tree" and any(int(part) <= 0 for part in state_parts[1:]))):
            bootstrap_fail("The preserved v2r12 tree fingerprint is malformed")

    expected_immediate_predecessor_release = (
        EXPECTED_STUDY_STORAGE_ROOT / "releases"
        / "release_20260921_visdrone_paper_v2r11"
        / "SemUAV-VLM-VisDrone-Paper-Study"
    )
    expected_immediate_predecessor_archive = (
        EXPECTED_STUDY_STORAGE_ROOT / "incoming"
        / "SemUAV-VLM-VisDrone-Paper-Study-20260921-v2r11.zip"
    )
    expected_immediate_predecessor_deployer = (
        EXPECTED_STUDY_STORAGE_ROOT / "incoming"
        / "deploy_semuav_visdrone_v2r11_20260921.sh"
    )
    expected_immediate_predecessor_receipt = (
        EXPECTED_STUDY_STORAGE_ROOT / "private"
        / "visdrone_v2r11_deployment" / "DEPLOYMENT_READY.json"
    )
    expected_immediate_predecessor = {
        "release_id": "release_20260921_visdrone_paper_v2r11",
        "run_name": "semuav_visdrone_v2r11",
        "release_path": str(expected_immediate_predecessor_release),
        "archive_path": str(expected_immediate_predecessor_archive),
        "archive_sha256": (
            "737c168c8710d9f111f13ce138fefc9b3dc2f2bc4da9fc858a3cc11e76c465b8"
        ),
        "release_manifest_sha256": (
            "b98c11aefe1617e6d2bffcaf9107b0708d31f45040db76d959a8c6664d7ee2ec"
        ),
        "checksum_inventory_sha256": (
            "a761c0f0ae1f8d1ded79d388885f25c7dc4766cf94fe41b7120aa24d620e6ab0"
        ),
        "payload_file_count": 109,
        "archive_member_count": 111,
        "payload_total_uncompressed_bytes": 4063633,
        "builder_sha256": (
            "c2edf8ebec76727fcaf2d785c01a3555704d4b5b639b8c373021588b1acd1447"
        ),
        "notebook_sha256": (
            "9b6b4d4585036d0d2c45df1fce21af0f4a08fbcc90f9406cdeefb4c690b2154c"
        ),
        "deployer_path": str(expected_immediate_predecessor_deployer),
        "deployer_sha256": (
            "ee4ce4ad394e46eb85d4c83040f9f95020bf685436e06a2dfc1b8411d5905bb5"
        ),
        "deployment_receipt": str(expected_immediate_predecessor_receipt),
        "deployment_receipt_sha256": (
            "befdce3067b571933dd94d35758c2c29385a06a8bb02ea93ae681117dcc17f3e"
        ),
        "experiment1_recovery_receipt": str(expected_recovery_path),
        "experiment1_recovery_receipt_sha256": (
            "6ca88cb5e82c8b33e27fd229443e233a48323d91fd73d0e79ec51ff8fc61a608"
        ),
        "experiment1_recovery_receipt_bytes": 37416,
        "experiment1_recovery_artifact_count": 132,
        "experiment1_recovery_row_count": 4152,
        "active_model_set_sha256": (
            "6f83118bb62cd68c98e9ee2afd947022c04088b9329a0eeef832eba17c2fb08c"
        ),
        "requirements_sha256": (
            "028a2757d2e1565db90ff26a26fd6e564bd592c868ed093d632cb930d438dfa5"
        ),
        "source_deployment_ready": True,
        "runtime_smoke_run": False,
        "scientific_environment_sealed": False,
        "scientific_validation_authorized": False,
        "scientific_experiments_started": False,
        "validation_experiments_started": False,
        "untouched_test_accessed": False,
        "unchanged_after_deployment_actions": True,
    }
    if receipt.get("authenticated_predecessor_v2r11") != (
        expected_immediate_predecessor
    ):
        bootstrap_fail("authenticated v2r11 immediate predecessor record changed")
    expected_source_predecessor_release = (
        EXPECTED_STUDY_STORAGE_ROOT / "releases"
        / "release_20260921_visdrone_paper_v2r10"
        / "SemUAV-VLM-VisDrone-Paper-Study"
    )
    expected_source_predecessor_archive = (
        EXPECTED_STUDY_STORAGE_ROOT / "incoming"
        / "SemUAV-VLM-VisDrone-Paper-Study-20260921-v2r10.zip"
    )
    expected_source_predecessor_deployer = (
        EXPECTED_STUDY_STORAGE_ROOT / "incoming"
        / "deploy_semuav_visdrone_v2r10_20260921.sh"
    )
    expected_source_predecessor_receipt = (
        EXPECTED_STUDY_STORAGE_ROOT / "private"
        / "visdrone_v2r10_deployment" / "DEPLOYMENT_READY.json"
    )
    expected_source_predecessor = {
        "release_id": "release_20260921_visdrone_paper_v2r10",
        "run_name": "semuav_visdrone_v2r10",
        "release_path": str(expected_source_predecessor_release),
        "archive_path": str(expected_source_predecessor_archive),
        "archive_sha256": (
            "f35723efcbda801bc1c3a8796e2fa6603b8d8bcacdcf66b7186df8d7a314af6a"
        ),
        "release_manifest_sha256": (
            "99323dd6109eca6d4942a120a3bb383f077d4a8f95869169f57f75e57574ac3c"
        ),
        "checksum_inventory_sha256": (
            "53bd76f3ffd5510b0d8c8cef6bfb82a6bdf63562c6dadd08c849d224e3513649"
        ),
        "requirements_sha256": (
            "028a2757d2e1565db90ff26a26fd6e564bd592c868ed093d632cb930d438dfa5"
        ),
        "deployer_path": str(expected_source_predecessor_deployer),
        "deployer_sha256": (
            "e8b49ea192b955bf07f8b53a0f6130f3eb0bd182c3a1ea28ede6f10e89bf79cf"
        ),
        "deployment_receipt": str(expected_source_predecessor_receipt),
        "deployment_receipt_sha256": (
            "95a64dad33ae9fa21974b8ba629bc9ec972ad989245bef5b38df42b00b2fb2e3"
        ),
        "payload_file_count": 108,
        "archive_member_count": 110,
        "payload_total_uncompressed_bytes": 4009068,
        "builder_sha256": (
            "09478e1f93768d80b589110e45edc05fa0c1cb3cd63bbf6e6e29ed15a27a73ab"
        ),
        "notebook_sha256": (
            "f35bde4c03536b97bf82ab719ad5a516225bcc321d5a878971410b4e86a16108"
        ),
        "experiment1_recovery_receipt": str(expected_recovery_path),
        "experiment1_recovery_receipt_sha256": (
            "6ca88cb5e82c8b33e27fd229443e233a48323d91fd73d0e79ec51ff8fc61a608"
        ),
        "experiment1_recovery_receipt_bytes": 37416,
        "experiment1_recovery_artifact_count": 132,
        "experiment1_recovery_row_count": 4152,
        "active_model_set_sha256": (
            "6f83118bb62cd68c98e9ee2afd947022c04088b9329a0eeef832eba17c2fb08c"
        ),
        "runtime_smoke_all_passed": True,
        "runtime_smoke_pass_count": 3,
        "scientific_environment_sealed": True,
        "validation_experiments_started": False,
        "untouched_test_accessed": False,
        "unchanged_after_deployment_actions": True,
    }
    if receipt.get("authenticated_predecessor_v2r10") != (
        expected_source_predecessor
    ):
        bootstrap_fail("authenticated v2r10 source predecessor record changed")
    if (
        not isinstance(recovery, dict)
        or set(recovery) != {
            "path", "bytes", "sha256", "schema_version", "status",
            "artifact_count", "inference_row_count",
        }
        or Path(str(recovery.get("path", ""))) != expected_recovery_path
        or recovery.get("schema_version")
        != "semuav.visdrone-v2r9-exp1-recovery.v1"
        or recovery.get("status")
        != "QUIESCENT_V2R9_EXP1_SOURCE_FROZEN"
        or recovery.get("artifact_count") != 132
        or recovery.get("inference_row_count") != 4152
        or type(recovery.get("bytes")) is not int
        or recovery["bytes"] <= 0
    ):
        bootstrap_fail("v2r9 Experiment-1 recovery record changed")
    recovery_sha256 = exact_sha256(
        recovery.get("sha256"), "v2r9 Experiment-1 recovery digest",
    )
    recovery_bytes = read_exact_regular(
        expected_recovery_path, "v2r9 Experiment-1 recovery receipt",
    )
    if (
        len(recovery_bytes) != recovery["bytes"]
        or hashlib.sha256(recovery_bytes).hexdigest() != recovery_sha256
    ):
        bootstrap_fail("v2r9 Experiment-1 recovery receipt changed")
    os.environ["SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT"] = str(
        expected_recovery_path
    )
    os.environ["SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT_SHA256"] = (
        recovery_sha256
    )
    manifest_sha256 = exact_sha256(
        release.get("manifest_sha256"), "release manifest digest",
    )
    checksum_sha256 = exact_sha256(
        release.get("checksum_inventory_sha256"),
        "release checksum digest",
    )
    archive_sha256 = exact_sha256(
        release.get("archive_sha256"), "release archive digest",
    )
    if (
        source_release.get("manifest_sha256") != manifest_sha256
        or source_release.get("checksum_inventory_sha256")
        != checksum_sha256
        or source_release.get("archive_sha256") != archive_sha256
    ):
        bootstrap_fail("duplicate release source digests disagree")

    manifest_bytes = read_exact_regular(manifest_path, "release manifest")
    checksum_bytes = read_exact_regular(checksum_path, "checksum inventory")
    archive_bytes = read_exact_regular(expected_archive_path, "source archive")
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
        bootstrap_fail("release manifest bytes changed")
    if hashlib.sha256(checksum_bytes).hexdigest() != checksum_sha256:
        bootstrap_fail("checksum inventory bytes changed")
    if hashlib.sha256(archive_bytes).hexdigest() != archive_sha256:
        bootstrap_fail("source archive bytes changed")

    manifest = decode_object(manifest_bytes, "release manifest")
    methodology = manifest.get("methodology")
    if (
        manifest.get("schema_version")
        != "semuav.visdrone-paper-release.v1"
        or manifest.get("release_id")
        != "release_20260921_visdrone_paper_v2r13"
        or manifest.get("run_name") != "semuav_visdrone_v2r13"
        or manifest.get("package_name")
        != "SemUAV-VLM-VisDrone-Paper-Study"
        or manifest.get("archive_name")
        != "SemUAV-VLM-VisDrone-Paper-Study-20260921-v2r13.zip"
        or manifest.get("study_entrypoint") != notebook_relative
        or manifest.get("notebook_builder")
        != "build_visdrone_paper_notebook.py"
        or manifest.get("pre_validation_native_receiver_correction")
        != "PRE_VALIDATION_NATIVE_RECEIVER_RECOVERY_20260921_V2R10.md"
        or manifest.get("pre_validation_acceptance_recovery_isolation")
        != "PRE_VALIDATION_ACCEPTANCE_RECOVERY_ISOLATION_20260921_V2R11.md"
        or manifest.get(
            "pre_smoke_predecessor_boundary_routing_correction"
        )
        != "PRE_SMOKE_PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_20260921_V2R12.md"
        or manifest.get("authenticated_predecessor_v2r12") != expected_sealed_predecessor
        or manifest.get("pre_validation_low_evidence_packet_correction")
        != "PRE_VALIDATION_LOW_EVIDENCE_PACKET_CORRECTION_20260921_V2R13.md"
        or manifest.get("authenticated_predecessor_v2r11")
        != expected_immediate_predecessor
        or manifest.get("authenticated_predecessor_v2r10")
        != expected_source_predecessor
        or not isinstance(methodology, dict)
        or methodology.get("native_receiver_correction")
        != "PRE_VALIDATION_NATIVE_RECEIVER_RECOVERY_20260921_V2R10.md"
    ):
        bootstrap_fail("release manifest identity changed")
    inventory = manifest.get("payload_inventory")
    if (
        not isinstance(inventory, list)
        or type(manifest.get("payload_file_count")) is not int
        or len(inventory) != manifest["payload_file_count"]
    ):
        bootstrap_fail("release payload inventory count changed")
    records = {}
    total_bytes = 0
    for record in inventory:
        if not isinstance(record, dict) or set(record) != {
            "path", "bytes", "sha256", "role",
        }:
            bootstrap_fail("release payload record envelope changed")
        relative_text = record["path"]
        if not isinstance(relative_text, str):
            bootstrap_fail("release payload path is not text")
        relative = PurePosixPath(relative_text)
        if (
            relative.is_absolute()
            or relative.as_posix() != relative_text
            or any(part in {"", ".", ".."} for part in relative.parts)
            or "\\\\" in relative_text
            or relative_text in {
                "RELEASE_MANIFEST.json", "RELEASE_MANIFEST_SHA256.txt",
            }
            or relative_text in records
            or type(record["bytes"]) is not int
            or record["bytes"] < 0
            or not isinstance(record["role"], str)
            or not record["role"]
        ):
            bootstrap_fail(f"unsafe payload envelope: {relative_text!r}")
        records[relative_text] = {
            "bytes": record["bytes"],
            "sha256": exact_sha256(
                record["sha256"], f"payload {relative_text} digest",
            ),
            "role": record["role"],
        }
        total_bytes += record["bytes"]
    if (
        manifest.get("payload_total_uncompressed_bytes") != total_bytes
        or manifest.get("archive_member_count") != len(records) + 2
        or notebook_relative not in records
    ):
        bootstrap_fail("release payload totals/entrypoint changed")
    expected_checksum_lines = [
        f"{record['sha256']}  {name}"
        for name, record in records.items()
    ]
    expected_checksum_lines.append(
        f"{manifest_sha256}  RELEASE_MANIFEST.json"
    )
    expected_checksum_bytes = (
        "\\n".join(sorted(expected_checksum_lines)) + "\\n"
    ).encode("utf-8")
    if checksum_bytes != expected_checksum_bytes:
        bootstrap_fail("checksum inventory is not the exact manifest projection")

    expected_directories = {
        PurePosixPath(*PurePosixPath(name).parts[:index]).as_posix()
        for name in records
        for index in range(1, len(PurePosixPath(name).parts))
    }
    observed_names = set()
    checkpoint_path = None
    for source_entry in expected_release_root.rglob("*"):
        relative_text = source_entry.relative_to(
            expected_release_root
        ).as_posix()
        relative_parts = PurePosixPath(relative_text).parts
        try:
            entry_stat = source_entry.lstat()
        except OSError as exc:
            bootstrap_fail(f"source entry changed during scan: {exc}")
        if source_entry.is_symlink():
            bootstrap_fail(f"source tree contains a symlink: {relative_text}")
        if (
            "__pycache__" in relative_parts
            or PurePosixPath(relative_text).suffix in {".pyc", ".pyo"}
        ):
            bootstrap_fail(
                f"source tree contains executable bytecode: {relative_text}"
            )
        if ".ipynb_checkpoints" in relative_parts:
            if source_entry.is_dir() and relative_text == ".ipynb_checkpoints":
                continue
            if (
                stat.S_ISREG(entry_stat.st_mode)
                and entry_stat.st_nlink == 1
                and relative_text == checkpoint_relative
            ):
                checkpoint_path = source_entry
                continue
            bootstrap_fail(
                f"unexpected Jupyter checkpoint entry: {relative_text}"
            )
        if source_entry.is_dir():
            if relative_text not in expected_directories:
                bootstrap_fail(f"unlisted source directory: {relative_text}")
            continue
        if not stat.S_ISREG(entry_stat.st_mode) or entry_stat.st_nlink != 1:
            bootstrap_fail(f"non-regular/hard-linked source entry: {relative_text}")
        if relative_text in {
            "RELEASE_MANIFEST.json", "RELEASE_MANIFEST_SHA256.txt",
        }:
            continue
        if relative_text not in records:
            bootstrap_fail(f"unlisted source payload: {relative_text}")
        observed_names.add(relative_text)
    if observed_names != set(records):
        bootstrap_fail("source payload set differs from release inventory")

    for relative_text, record in records.items():
        payload_path = expected_release_root.joinpath(
            *PurePosixPath(relative_text).parts
        )
        if relative_text == notebook_relative:
            continue
        payload = read_exact_regular(
            payload_path, f"source payload {relative_text}",
        )
        if (
            len(payload) != record["bytes"]
            or hashlib.sha256(payload).hexdigest() != record["sha256"]
        ):
            bootstrap_fail(f"source payload bytes changed: {relative_text}")

    archive_prefix = "SemUAV-VLM-VisDrone-Paper-Study/"
    expected_archive_names = {
        archive_prefix + name for name in records
    } | {
        archive_prefix + "RELEASE_MANIFEST.json",
        archive_prefix + "RELEASE_MANIFEST_SHA256.txt",
    }
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
            members = archive.infolist()
            member_names = [member.filename for member in members]
            if (
                len(member_names) != len(set(member_names))
                or set(member_names) != expected_archive_names
            ):
                bootstrap_fail("sealed archive member set changed")
            member_by_name = {
                member.filename: member for member in members
            }
            for name, member in member_by_name.items():
                member_mode = (member.external_attr >> 16) & 0o170000
                if (
                    member.is_dir()
                    or member.flag_bits & 0x1
                    or member_mode not in {0, stat.S_IFREG}
                ):
                    bootstrap_fail(f"unsafe sealed archive member: {name}")
            sealed_manifest_bytes = archive.read(
                archive_prefix + "RELEASE_MANIFEST.json"
            )
            sealed_checksum_bytes = archive.read(
                archive_prefix + "RELEASE_MANIFEST_SHA256.txt"
            )
            sealed_notebook_bytes = archive.read(
                archive_prefix + notebook_relative
            )
    except (OSError, zipfile.BadZipFile, KeyError, RuntimeError) as exc:
        bootstrap_fail(f"sealed archive cannot be verified: {exc}")
    if (
        sealed_manifest_bytes != manifest_bytes
        or sealed_checksum_bytes != checksum_bytes
        or len(sealed_notebook_bytes) != records[notebook_relative]["bytes"]
        or hashlib.sha256(sealed_notebook_bytes).hexdigest()
        != records[notebook_relative]["sha256"]
    ):
        bootstrap_fail("sealed archive metadata/notebook bytes changed")
    sealed_notebook = decode_object(
        sealed_notebook_bytes, "sealed authored notebook",
    )
    live_notebook = decode_object(
        read_exact_regular(
            expected_release_root / notebook_relative, "live notebook",
        ),
        "live notebook",
    )
    sealed_parameter_source = next((
        "".join(cell.get("source", []))
        if isinstance(cell.get("source", ""), list)
        else cell.get("source", "")
        for cell in sealed_notebook.get("cells", [])
        if "parameters" in cell.get("metadata", {}).get("tags", [])
    ), None)
    if not isinstance(sealed_parameter_source, str):
        bootstrap_fail("sealed notebook has no unique parameter source")
    sealed_contract = authored_contract(
        sealed_notebook, "sealed authored notebook",
    )
    live_contract = authored_contract(
        live_notebook, "live notebook", sealed_parameter_source,
    )
    if live_contract != sealed_contract:
        bootstrap_fail(
            "live notebook differs outside approved parameter RHS values"
        )
    if checkpoint_path is not None:
        checkpoint_notebook = decode_object(
            read_exact_regular(checkpoint_path, "notebook checkpoint"),
            "notebook checkpoint",
        )
        checkpoint_contract = authored_contract(
            checkpoint_notebook,
            "notebook checkpoint",
            sealed_parameter_source,
        )
        if checkpoint_contract != sealed_contract:
            bootstrap_fail(
                "notebook checkpoint differs outside approved parameter RHS values"
            )
    print(
        "V2R13_PREIMPORT_LIVE_SOURCE_INVENTORY_VERIFIED",
        manifest_sha256,
        f"payload_count={len(records)}",
    )
    V2R13_PREIMPORT_SOURCE_VERIFIED = True

    # These are the only operator-editable assignments.  They execute only
    # after the saved notebook and complete live release have authenticated.
    PHASE = os.environ.get("SEMUAV_VISDRONE_PHASE", "prepare").strip().lower()
    EXECUTE_PHASE = False
    FREEZE_PROTOCOL = False
    AUTHORIZE_TEST_ONCE = False
    RUN_NAME = os.environ.get("SEMUAV_RUN_NAME", "semuav_visdrone_v2r13")
    RESUME = True
    DEVICE = os.environ.get("SEMUAV_DEVICE", "cuda:0")
    RUN_MODEL_SMOKE = False
    BUILD_NATIVE_RECEIVER_CACHE = False
    BUILD_FEATURE_CACHE = False

    if RUN_NAME != "semuav_visdrone_v2r13":
        raise ValueError("The official v2r13 notebook requires RUN_NAME=semuav_visdrone_v2r13")
    if DEVICE != "cuda:0":
        raise ValueError("The official v2r13 notebook requires DEVICE=cuda:0")
    expected_checkpoint = (
        PROJECT_ROOT / ".ipynb_checkpoints"
        / "SemUAV_VLM_VisDrone_Paper_Study-checkpoint.ipynb"
    )
    for source_entry in PROJECT_ROOT.rglob("*"):
        relative_parts = source_entry.relative_to(PROJECT_ROOT).parts
        if source_entry.is_symlink():
            raise ValueError(f"Source tree contains a symbolic link: {source_entry}")
        if "__pycache__" in relative_parts or source_entry.suffix in {".pyc", ".pyo"}:
            raise ValueError(f"Source tree contains executable bytecode: {source_entry}")
        if ".ipynb_checkpoints" in relative_parts:
            if source_entry == expected_checkpoint.parent and source_entry.is_dir():
                continue
            if source_entry == expected_checkpoint and source_entry.is_file():
                continue
            raise ValueError(f"Unexpected Jupyter checkpoint entry: {source_entry}")

    # Root filesystem is intentionally never used for runtime caches. These
    # defaults take effect before importing model/CUDA libraries in the next
    # cell and may be overridden only by another absolute path below /data.
    cache_defaults = {
        "TMPDIR": STUDY_STORAGE_ROOT / "tmp",
        "XDG_CACHE_HOME": STUDY_STORAGE_ROOT / ".cache" / "xdg",
        "XDG_CONFIG_HOME": STUDY_STORAGE_ROOT / ".config",
        "HF_HOME": STUDY_STORAGE_ROOT / ".cache" / "huggingface",
        "HUGGINGFACE_HUB_CACHE": STUDY_STORAGE_ROOT / ".cache" / "huggingface",
        "HF_HUB_CACHE": STUDY_STORAGE_ROOT / ".cache" / "huggingface",
        "HF_MODULES_CACHE": STUDY_STORAGE_ROOT / ".cache" / "huggingface" / "modules-v2r13",
        "HF_DATASETS_CACHE": STUDY_STORAGE_ROOT / ".cache" / "huggingface" / "datasets",
        "TRANSFORMERS_CACHE": STUDY_STORAGE_ROOT / ".cache" / "huggingface",
        "TORCH_HOME": STUDY_STORAGE_ROOT / ".cache" / "torch",
        "TORCH_EXTENSIONS_DIR": STUDY_STORAGE_ROOT / ".cache" / "torch_extensions",
        "CUDA_CACHE_PATH": STUDY_STORAGE_ROOT / ".cache" / "cuda",
        "TRITON_CACHE_DIR": STUDY_STORAGE_ROOT / ".cache" / "triton",
        "YOLO_CONFIG_DIR": STUDY_STORAGE_ROOT / ".config" / "ultralytics",
        "PIP_CACHE_DIR": STUDY_STORAGE_ROOT / ".cache" / "pip",
        "JUPYTER_DATA_DIR": STUDY_STORAGE_ROOT / ".cache" / "jupyter" / "data",
        "JUPYTER_CONFIG_DIR": STUDY_STORAGE_ROOT / ".cache" / "jupyter" / "config-v2r13",
        "JUPYTER_RUNTIME_DIR": STUDY_STORAGE_ROOT / ".cache" / "jupyter" / "runtime-v2r13",
        "IPYTHONDIR": STUDY_STORAGE_ROOT / ".cache" / "ipython-v2r13",
        "MPLCONFIGDIR": STUDY_STORAGE_ROOT / ".cache" / "matplotlib-v2r13",
    }
    for variable, default_path in cache_defaults.items():
        os.environ.setdefault(variable, str(default_path))
    if EXECUTE_PHASE:
        for variable in cache_defaults:
            path = Path(os.environ[variable]).expanduser().resolve()
            if path != Path("/data") and Path("/data") not in path.parents:
                raise ValueError(f"{variable} must resolve below /data: {path}")
            path.mkdir(parents=True, exist_ok=True)

    ALLOWED_PHASES = {"prepare", "validation", "test", "report", "embedded"}
    if PHASE not in ALLOWED_PHASES:
        raise ValueError(f"PHASE must be one of {sorted(ALLOWED_PHASES)}")
    if PHASE == "validation":
        os.environ["SEMUAV_ENABLE_V2R9_EXP1_RECOVERY"] = "1"
    else:
        os.environ.pop("SEMUAV_ENABLE_V2R9_EXP1_RECOVERY", None)
    os.environ["SEMUAV_MODELS_LOCAL_ONLY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["PIP_NO_INDEX"] = "1"
    os.environ["PIP_NO_INPUT"] = "1"
    if PHASE != "validation" and FREEZE_PROTOCOL:
        raise ValueError("FREEZE_PROTOCOL may be true only in validation")
    if PHASE != "test" and AUTHORIZE_TEST_ONCE:
        raise ValueError("AUTHORIZE_TEST_ONCE may be true only in test")
    V2R13_BOOTSTRAP_PHASE = PHASE
    V2R13_BOOTSTRAP_EXECUTE_PHASE = EXECUTE_PHASE
    V2R13_APPROVED_CONTROL_SNAPSHOT = tuple(
        (name, type(value).__name__, value)
        for name, value in (
            ("PHASE", PHASE),
            ("EXECUTE_PHASE", EXECUTE_PHASE),
            ("FREEZE_PROTOCOL", FREEZE_PROTOCOL),
            ("AUTHORIZE_TEST_ONCE", AUTHORIZE_TEST_ONCE),
            ("RUN_NAME", RUN_NAME),
            ("RESUME", RESUME),
            ("DEVICE", DEVICE),
            ("RUN_MODEL_SMOKE", RUN_MODEL_SMOKE),
            ("BUILD_NATIVE_RECEIVER_CACHE", BUILD_NATIVE_RECEIVER_CACHE),
            ("BUILD_FEATURE_CACHE", BUILD_FEATURE_CACHE),
        )
    )
    print({
        "phase": PHASE,
        "execute": EXECUTE_PHASE,
        "run_name": RUN_NAME,
        "resume": RESUME,
        "device": DEVICE,
        "project_root": str(PROJECT_ROOT),
        "offline_models": os.environ.get("SEMUAV_MODELS_LOCAL_ONLY") == "1",
    })
    """, "parameters"),
    code("""
    if V2R13_PREIMPORT_SOURCE_VERIFIED is not True:
        raise RuntimeError(
            "V2R13 pre-import source authentication did not complete"
        )

    import ast
    import grp
    import hashlib
    import json
    import subprocess
    import sys
    import tempfile
    from pathlib import Path, PurePosixPath

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    try:
        from IPython.display import display
    except ImportError:
        display = print

    def _require_v2r13_preimport_source_gate():
        if not EXECUTE_PHASE:
            return
        current_controls = tuple(
            (name, type(value).__name__, value)
            for name, value in (
                ("PHASE", PHASE),
                ("EXECUTE_PHASE", EXECUTE_PHASE),
                ("FREEZE_PROTOCOL", FREEZE_PROTOCOL),
                ("AUTHORIZE_TEST_ONCE", AUTHORIZE_TEST_ONCE),
                ("RUN_NAME", RUN_NAME),
                ("RESUME", RESUME),
                ("DEVICE", DEVICE),
                ("RUN_MODEL_SMOKE", RUN_MODEL_SMOKE),
                ("BUILD_NATIVE_RECEIVER_CACHE", BUILD_NATIVE_RECEIVER_CACHE),
                ("BUILD_FEATURE_CACHE", BUILD_FEATURE_CACHE),
            )
        )
        if (
            V2R13_PREIMPORT_SOURCE_VERIFIED is not True
            or V2R13_BOOTSTRAP_EXECUTE_PHASE is not True
            or PHASE != V2R13_BOOTSTRAP_PHASE
            or current_controls != V2R13_APPROVED_CONTROL_SNAPSHOT
        ):
            raise RuntimeError(
                "Restart the kernel and rerun the authenticated v2r13 parameter "
                "cell before any scientific action"
            )

    _require_v2r13_preimport_source_gate()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from semuav.config import load_config
    from semuav.registry import (
        RunContext, StudyGateError, atomic_write_frame, atomic_write_json,
        atomic_write_text,
        authorize_test_once, finalize_test_once, freeze_protocol,
        frozen_file_manifest, hash_file, hash_object,
        load_report_safe_finalized_test_handoff, notebook_source_hash,
        resolve_frozen_artifact,
        selected_vlm_claim_gate as registered_selected_vlm_claim_gate,
    )
    from semuav.visdrone_paper import (
        DEVELOPMENT_COMPONENTS, UNTOUCHED_TEST_COMPONENTS,
        load_untouched_test_permit, publisher_split_mapping,
    )
    from semuav.visdrone_models import (
        OFFICIAL_RUNTIME_MODEL_IDS,
        RUNTIME_SMOKE_ISOLATION_SCHEMA,
        RUNTIME_SMOKE_REPORT_SCHEMA_VERSION,
        derive_v2r6_three_model_lock,
        model_loader_environment,
        require_runtime_smoke_pass,
        run_one_image_runtime_smoke, validate_registered_model_configuration,
        verify_detector_checkpoint, verify_detector_training_receipt,
        verify_huggingface_model_lock,
    )
    from semuav.perception import (
        InferenceRequest, VISDRONE_GENERATION_POLICY,
        VISDRONE_GENERATION_POLICY_SHA256,
        VISDRONE_GENERATION_POLICY_VERSION,
        VISDRONE_GENERATION_STRATEGY_BY_MODEL,
        VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256,
        VISDRONE_PROPOSAL_DECISION_PROMPT_PATH,
        prompt_for_mission, visdrone_assistant_prefill_for_model,
        visdrone_generation_policy_sha256,
    )
    from semuav.visdrone_workflow import (
        SOURCE_PROVENANCE_SCOPE, WORKFLOW_VERSION,
        audit_publisher_splits, build_readiness_claim_ledger,
        export_paper_tables, load_prepared_visdrone_bundle,
        prepare_visdrone_bundle,
        read_analysis_manifest,
        registered_mission_policies, registered_temporal_sampling_spec,
        verify_visdrone_source_provenance,
        select_registered_analysis_sample_from_config, verify_artifact_hash_manifest,
        select_registered_temporal_sample_from_config,
        verify_paper_completion_receipt,
        verify_visdrone_media_partition_audit,
        write_artifact_hash_manifest, write_registered_analysis_sample,
        write_registered_temporal_sample,
    )
    from semuav.visdrone_runner import (
        evaluate_optional_embedded_h5,
        recompute_hypotheses_from_frozen_results,
        run_confirmatory_test_once,
        run_validation_network_and_controller,
        run_validation_perception_and_frontier,
    )
    from semuav.figures import build_hypothesis_effects_frame, generate_publication_figures
    from semuav.experiments import (
        EXPERIMENT6_CAMPAIGN_VERSION, EXPERIMENT6_REGISTERED_CASES,
        EXPERIMENT6_REGISTERED_STATE_STRATEGIES,
        EXPERIMENT6_STATE_STRATEGY_VERSION,
        experiment6_receiver_robustness, experiment6_registered_campaign,
        experiment7_readiness,
    )
    from semuav.exp6_strategies import experiment6_state_strategy_comparison
    from semuav.hardware import (
        environment_fingerprint, environment_report,
        require_exclusive_physical_gpu,
    )
    REGISTERED_RUNTIME_SMOKE_MODEL_IDS = (
        "qwen3_vl_2b",
        "smolvlm2_2b",
        "qwen3_vl_8b_upper_bound",
    )
    if tuple(OFFICIAL_RUNTIME_MODEL_IDS) != REGISTERED_RUNTIME_SMOKE_MODEL_IDS:
        raise RuntimeError("Core and notebook v2r13 model registrations differ")
    print("Python", sys.version.split()[0], "| executable", sys.executable, "| phase", PHASE)
    """),
    code("""
    # Resolve model commits from an existing byte-level lock before creating the
    # scientific RunContext. This makes exact commits part of the config hash.
    preliminary_cfg = load_config(CONFIG_PATH)
    RUN_OUTPUT_DIR = preliminary_cfg.paths.output_root / RUN_NAME
    FROZEN_INPUT_DIR = RUN_OUTPUT_DIR / "frozen_inputs"

    def _assert_safe_output_directory(path, *, label, create):
        candidate = Path(path).expanduser()
        # Check the lexical path and every existing parent before creation. This
        # prevents RESUME from redirecting v2r13 receipts through a precreated
        # run/frozen-input symlink into validation or test material.
        for component in (candidate, *candidate.parents):
            if component.is_symlink():
                raise StudyGateError(f"{label} path contains a symbolic link: {component}")
        if candidate.exists() and not candidate.is_dir():
            raise StudyGateError(f"{label} exists but is not a directory: {candidate}")
        if create:
            candidate.mkdir(parents=True, exist_ok=True)
            if candidate.is_symlink() or not candidate.is_dir():
                raise StudyGateError(f"{label} could not be created safely: {candidate}")
        return candidate

    _assert_safe_output_directory(
        RUN_OUTPUT_DIR, label="v2r13 run output directory", create=False,
    )
    _assert_safe_output_directory(
        FROZEN_INPUT_DIR, label="v2r13 frozen-input directory", create=False,
    )
    reused_cfg = preliminary_cfg.models["reused_v1_artifacts"]
    REUSED_V1_FROZEN_INPUT_DIR = Path(
        reused_cfg["source_frozen_input_dir"]
    ).expanduser().resolve()
    SOURCE_MODEL_LOCK = REUSED_V1_FROZEN_INPUT_DIR / "MODEL_LOCK.json"
    MODEL_LOCK = FROZEN_INPUT_DIR / "MODEL_LOCK.json"
    MODEL_LOCK_DERIVATION = FROZEN_INPUT_DIR / "MODEL_LOCK_DERIVATION.json"
    MODEL_SET_AMENDMENT_RECORD = (
        PROJECT_ROOT / "PRE_VALIDATION_MODEL_SET_AMENDMENT_20260914_V2R6.md"
    )
    TEMPORAL_FEASIBILITY_AMENDMENT_RECORD = (
        PROJECT_ROOT
        / "PRE_EXPERIMENT_TEMPORAL_FEASIBILITY_AMENDMENT_20260914_V2R7.md"
    )
    RUNTIME_CORRECTION_RECORD = (
        PROJECT_ROOT / "PRE_VALIDATION_NATIVE_RECEIVER_RECOVERY_20260921_V2R10.md"
    )
    ACCEPTANCE_RECOVERY_ISOLATION_RECORD = (
        PROJECT_ROOT
        / "PRE_VALIDATION_ACCEPTANCE_RECOVERY_ISOLATION_20260921_V2R11.md"
    )
    PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_RECORD = (
        PROJECT_ROOT
        / "PRE_SMOKE_PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_20260921_V2R12.md"
    )
    LOW_EVIDENCE_PACKET_CORRECTION_RECORD = (
        PROJECT_ROOT / "PRE_VALIDATION_LOW_EVIDENCE_PACKET_CORRECTION_20260921_V2R13.md"
    )
    MODEL_SET_AMENDMENT_ID = (
        "semuav.visdrone-v2r6-prospective-moondream-exclusion.v1"
    )
    MODEL_SET_EXCLUSION_REASON = (
        "v2r5_moondream2_runtime_smoke_fail_schema_before_"
        "validation_or_test_access"
    )
    if PHASE in {"prepare", "validation", "test"}:
        expected_model_lock_sha256 = str(
            reused_cfg.get("expected_sha256", {}).get("model_lock", "")
        ).lower()
        if (
            SOURCE_MODEL_LOCK.is_symlink()
            or not SOURCE_MODEL_LOCK.is_file()
            or len(expected_model_lock_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_model_lock_sha256
            )
        ):
            raise StudyGateError("MODEL_LOCK.json is missing, unsafe, or lacks its exact anchor")
        try:
            model_lock_bytes = SOURCE_MODEL_LOCK.read_bytes()
        except OSError as exc:
            raise StudyGateError("MODEL_LOCK.json cannot be read safely") from exc
        if hashlib.sha256(model_lock_bytes).hexdigest() != expected_model_lock_sha256:
            raise StudyGateError("MODEL_LOCK.json differs from its exact configured SHA-256")
        try:
            locked_payload = json.loads(model_lock_bytes.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise StudyGateError("MODEL_LOCK.json is not valid anchored JSON") from exc
        revision_env = {
            "qwen3_vl_2b": "SEMUAV_QWEN_REVISION",
            "smolvlm2_2b": "SEMUAV_SMOLVLM_REVISION",
            "qwen3_vl_8b_upper_bound": "SEMUAV_UPPER_BOUND_REVISION",
        }
        locked_revision_environment = {
            revision_env[str(item.get("model_id", ""))]: str(
                item.get("resolved_revision", "")
            )
            for item in locked_payload.get("models", [])
            if str(item.get("model_id", "")) in revision_env
        }
        expected_revision_names = set(revision_env.values())
        if (
            set(locked_revision_environment) != expected_revision_names
            or any(
                not isinstance(value, str) or len(value) != 40
                or any(character not in "0123456789abcdef" for character in value)
                for value in locked_revision_environment.values()
            )
        ):
            raise StudyGateError(
                "The anchored source MODEL_LOCK.json has invalid active revision bindings"
            )
        if EXECUTE_PHASE and PHASE == "prepare":
            active_model_lock = derive_v2r6_three_model_lock(
                SOURCE_MODEL_LOCK,
                destination=MODEL_LOCK,
                derivation_destination=MODEL_LOCK_DERIVATION,
                expected_source_sha256=expected_model_lock_sha256,
                amendment_id=MODEL_SET_AMENDMENT_ID,
                exclusion_reason=MODEL_SET_EXCLUSION_REASON,
                allow_create=True,
            )
        elif PHASE in {"validation", "test"}:
            if (
                MODEL_LOCK.is_symlink()
                or not MODEL_LOCK.is_file()
                or MODEL_LOCK_DERIVATION.is_symlink()
                or not MODEL_LOCK_DERIVATION.is_file()
            ):
                raise StudyGateError(
                    "The inherited v2r6 three-model lock and derivation receipt must "
                    "already exist before validation/test"
                )
            active_model_lock = derive_v2r6_three_model_lock(
                SOURCE_MODEL_LOCK,
                destination=MODEL_LOCK,
                derivation_destination=MODEL_LOCK_DERIVATION,
                expected_source_sha256=expected_model_lock_sha256,
                amendment_id=MODEL_SET_AMENDMENT_ID,
                exclusion_reason=MODEL_SET_EXCLUSION_REASON,
                allow_create=False,
            )
            if active_model_lock.get("derivation_mode") != {
                "model_lock": "REVERIFIED_EXISTING",
                "receipt": "REVERIFIED_EXISTING",
            }:
                raise StudyGateError(
                    "Validation/test cannot create the inherited v2r6 model-set amendment"
                )
        else:
            active_model_lock = None
        if active_model_lock is not None:
            authenticated_model_loader_environment = model_loader_environment(
                MODEL_LOCK
            )
            authenticated_revision_environment = {
                name: authenticated_model_loader_environment.get(name)
                for name in expected_revision_names
            }
            if locked_revision_environment != authenticated_revision_environment:
                raise StudyGateError(
                    "Derived MODEL_LOCK.json and source revision bindings differ"
                )
        for variable, revision in locked_revision_environment.items():
            configured = os.environ.get(variable)
            if configured is not None and configured != revision:
                raise StudyGateError(f"{variable} conflicts with MODEL_LOCK.json")
            os.environ[variable] = revision
    cfg = load_config(CONFIG_PATH)
    temporal_feasibility_amendment = dict(
        cfg.benchmark.get("temporal_feasibility_amendment", {})
    )
    expected_temporal_feasibility_amendment = {
        "schema": "semuav.visdrone-temporal-feasibility-amendment.v1",
        "record": TEMPORAL_FEASIBILITY_AMENDMENT_RECORD.name,
        "predecessor_release": "release_20260914_visdrone_paper_v2r6",
        "predecessor_run_name": "semuav_visdrone_v2r6",
        "predecessor_registered_validation_sequences": 15,
        "predecessor_observed_eligible_validation_sequences": 7,
        "amended_validation_sequences": 7,
        "retained_test_sequences": 15,
        "observations_per_sequence": 64,
        "validation_trace_realizations": 12,
        "amended_validation_fully_paired_scene_trace_cells": 84,
        "validation_development_bytes_read": True,
        "validation_outcomes_evaluated": False,
        "temporal_artifact_written": False,
        "cache_build_started": False,
        "experiments_1_to_7_started": False,
        "protocol_frozen": False,
        "untouched_test_accessed": False,
        "scientific_contract_changed": True,
    }
    if temporal_feasibility_amendment != expected_temporal_feasibility_amendment:
        raise StudyGateError(
            "The exact inherited v2r7 temporal-feasibility amendment changed"
        )
    if (
        TEMPORAL_FEASIBILITY_AMENDMENT_RECORD.is_symlink()
        or not TEMPORAL_FEASIBILITY_AMENDMENT_RECORD.is_file()
    ):
        raise StudyGateError(
            "The inherited v2r7 temporal-feasibility amendment record is unsafe"
        )
    temporal_feasibility_amendment = {
        **temporal_feasibility_amendment,
        "record_sha256": hash_file(TEMPORAL_FEASIBILITY_AMENDMENT_RECORD),
    }
    if (
        cfg.runtime.get("pre_validation_runtime_correction_record")
        != RUNTIME_CORRECTION_RECORD.name
        or RUNTIME_CORRECTION_RECORD.is_symlink()
        or not RUNTIME_CORRECTION_RECORD.is_file()
    ):
        raise StudyGateError("The inherited v2r10 native-receiver correction is unsafe")
    runtime_correction_record_sha256 = hash_file(RUNTIME_CORRECTION_RECORD)
    if (
        cfg.runtime.get("pre_validation_acceptance_recovery_isolation_record")
        != ACCEPTANCE_RECOVERY_ISOLATION_RECORD.name
        or ACCEPTANCE_RECOVERY_ISOLATION_RECORD.is_symlink()
        or not ACCEPTANCE_RECOVERY_ISOLATION_RECORD.is_file()
    ):
        raise StudyGateError(
            "The v2r13 acceptance/recovery-isolation correction is unsafe"
        )
    acceptance_recovery_isolation_record_sha256 = hash_file(
        ACCEPTANCE_RECOVERY_ISOLATION_RECORD
    )
    if (
        cfg.runtime.get(
            "pre_smoke_predecessor_boundary_routing_correction_record"
        ) != PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_RECORD.name
        or PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_RECORD.is_symlink()
        or not PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_RECORD.is_file()
    ):
        raise StudyGateError(
            "The v2r13 predecessor-boundary-routing correction is unsafe"
        )
    predecessor_boundary_routing_correction_sha256 = hash_file(
        PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_RECORD
    )
    if (
        cfg.runtime.get("pre_validation_low_evidence_packet_correction_record")
        != LOW_EVIDENCE_PACKET_CORRECTION_RECORD.name
        or LOW_EVIDENCE_PACKET_CORRECTION_RECORD.is_symlink()
        or not LOW_EVIDENCE_PACKET_CORRECTION_RECORD.is_file()
    ):
        raise StudyGateError("The v2r13 low-evidence packet correction is unsafe")
    low_evidence_packet_correction_sha256 = hash_file(LOW_EVIDENCE_PACKET_CORRECTION_RECORD)
    expected_experiment1_recovery = {
        "schema": "semuav.visdrone-v2r9-exp1-recovery.v1",
        "predecessor_release": "release_20260920_visdrone_paper_v2r9",
        "predecessor_run_name": "semuav_visdrone_v2r9",
        "imported_scope": "raw_experiment1_detector_and_vlm_shards_only",
        "derived_results_recomputed": True,
        "downstream_results_recomputed": True,
        "untouched_test_accessed": False,
    }
    if dict(cfg.runtime.get("experiment1_recovery", {})) != expected_experiment1_recovery:
        raise StudyGateError("The v2r13 Experiment-1 recovery contract changed")
    cfg_dict = cfg.as_dict()
    ctx = RunContext.create(
        project_root=PROJECT_ROOT,
        cache_root=cfg.paths.cache_root,
        profile=cfg.profile.value,
        run_name=RUN_NAME,
        config=cfg_dict,
        resume=RESUME,
    )
    _assert_safe_output_directory(
        RUN_OUTPUT_DIR, label="v2r13 run output directory", create=False,
    )
    NOTEBOOK_PATH = PROJECT_ROOT / "SemUAV_VLM_VisDrone_Paper_Study.ipynb"
    RESULTS_DIR = ctx.run_root / "results"
    FIGURE_DIR = cfg.paths.output_root / RUN_NAME / ctx.run_root.name / "paper_figures"
    TABLE_DIR = cfg.paths.output_root / RUN_NAME / ctx.run_root.name / "paper_tables"
    MANIFEST_DIR = cfg.paths.output_root / RUN_NAME / ctx.run_root.name / "manifests"
    for directory in (RESULTS_DIR, FIGURE_DIR, TABLE_DIR, MANIFEST_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    VISDRONE_ROOT = next(spec.root for spec in cfg.enabled_datasets if spec.adapter == "visdrone")
    _assert_safe_output_directory(
        FROZEN_INPUT_DIR, label="v2r13 frozen-input directory", create=True,
    )
    MODEL_SMOKE = FROZEN_INPUT_DIR / "MODEL_RUNTIME_SMOKE.json"
    MODEL_SMOKE_INPUT_RECEIPT = FROZEN_INPUT_DIR / "MODEL_RUNTIME_SMOKE_INPUT.json"
    INFERENCE_ENVIRONMENT_FINGERPRINT = (
        FROZEN_INPUT_DIR / "INFERENCE_ENVIRONMENT_FINGERPRINT.json"
    )
    INFERENCE_PIP_FREEZE = FROZEN_INPUT_DIR / "INFERENCE_PIP_FREEZE.txt"
    INFERENCE_ENVIRONMENT_SEAL = FROZEN_INPUT_DIR / "INFERENCE_ENVIRONMENT_SEAL.json"
    TRUSTED_CODE_CACHE_RECEIPT = FROZEN_INPUT_DIR / "TRUSTED_CODE_CACHE.json"
    CANDIDATE_READY_RECEIPT = Path(
        preliminary_cfg.runtime["candidate_ready_receipt"]
    ).expanduser()
    REUSED_ARTIFACTS_RECEIPT = FROZEN_INPUT_DIR / "REUSED_V1_FROZEN_ARTIFACTS.json"
    SMOKE_CANARY = PROJECT_ROOT / str(cfg.models["runtime_smoke_canary"])
    DETECTOR_DATA = cfg.paths.cache_root / "ultralytics_visdrone_det"
    DETECTOR_PLAN = REUSED_V1_FROZEN_INPUT_DIR / "detector_training_plan.json"
    DETECTOR_TRAINING_RECEIPT = (
        REUSED_V1_FROZEN_INPUT_DIR / "DETECTOR_TRAINING_RECEIPT.json"
    )
    DETECTOR_TRAINING_ENVIRONMENT_FINGERPRINT = (
        REUSED_V1_FROZEN_INPUT_DIR / "ENVIRONMENT_FINGERPRINT.json"
    )
    DETECTOR_TRAINING_PIP_FREEZE = REUSED_V1_FROZEN_INPUT_DIR / "PIP_FREEZE.txt"
    FEATURE_HEAD_DIR = REUSED_V1_FROZEN_INPUT_DIR / "feature_head"
    FEATURE_HEAD_LOCK = FEATURE_HEAD_DIR / "FEATURE_HEAD_LOCK.json"
    FEATURE_HEAD_WEIGHTS = FEATURE_HEAD_DIR / "feature_head_weights.npz"
    FEATURE_ENCODER_RECEIPT = FEATURE_HEAD_DIR / "FEATURE_ENCODER_RECEIPT.json"
    DEVELOPMENT_BUNDLE_DIR = REUSED_V1_FROZEN_INPUT_DIR / "development_bundle"
    DEVELOPMENT_SAMPLE_ROOT = (
        REUSED_V1_FROZEN_INPUT_DIR / "development_registered_samples"
    )
    VALIDATION_TEMPORAL_SAMPLE_DIR = (
        FROZEN_INPUT_DIR / "validation_registered_temporal_sample"
    )
    TEST_BUNDLE_DIR = RESULTS_DIR / "test_bundle"
    TEST_SAMPLE_DIR = RESULTS_DIR / "test_registered_sample"
    TEST_TEMPORAL_SAMPLE_DIR = RESULTS_DIR / "test_registered_temporal_sample"
    DEVELOPMENT_SOURCE = DEVELOPMENT_BUNDLE_DIR / "source_manifest.csv.gz"
    FROZEN_SPLITS = DEVELOPMENT_BUNDLE_DIR / "publisher_split_manifest.csv.gz"
    DEVELOPMENT_ANALYSIS = DEVELOPMENT_BUNDLE_DIR / "analysis_manifest.csv.gz"
    REGISTERED_TRACES = DEVELOPMENT_BUNDLE_DIR / "registered_trace_manifest.csv"
    MEDIA_PARTITION_AUDIT_RECEIPT = (
        DEVELOPMENT_BUNDLE_DIR / "VISDRONE_MEDIA_PARTITION_AUDIT.json"
    )
    MEDIA_PARTITION_HASH_AUDIT = (
        DEVELOPMENT_BUNDLE_DIR / "visdrone_media_partition_hash_audit.csv.gz"
    )
    TEST_GROUP_CAPACITY_AUDIT = (
        DEVELOPMENT_BUNDLE_DIR / "visdrone_test_group_capacity_audit.csv"
    )
    PROTOCOL = ctx.run_root / "frozen_protocol.json"
    TEST_LEDGER = ctx.run_root / "test_access_ledger.json"
    PROVISIONAL_CLAIM_LEDGER = RESULTS_DIR / "report_claim_status_PROVISIONAL.csv.gz"
    FINAL_CLAIM_LEDGER = TABLE_DIR / "claim_ledger.csv.gz"
    PAPER_ARTIFACT_MANIFEST = MANIFEST_DIR / "PAPER_ARTIFACTS_HASH_MANIFEST.json"
    PAPER_RESULTS_MANIFEST = MANIFEST_DIR / "PAPER_RESULTS_MANIFEST.json"
    FINAL_COMPLETION_RECEIPT = MANIFEST_DIR / "FINAL_COMPLETION_RECEIPT.json"
    REPORT_ENVIRONMENT_RECEIPT = MANIFEST_DIR / "REPORT_ENVIRONMENT_RECEIPT.json"

    # This inventory is deliberately limited to bytes that are safe to inspect
    # before the scientific-use seal exists.  In particular, no file below the
    # prepared development bundle and no calibration/validation registered
    # sample may enter the pre-seal verifier.  Full development verification is
    # a separate, post-seal operation below.
    PRESEAL_REUSED_V1_ARTIFACT_PATHS = {
        "model_lock": SOURCE_MODEL_LOCK,
        "detector_training_environment_fingerprint": (
            DETECTOR_TRAINING_ENVIRONMENT_FINGERPRINT
        ),
        "detector_training_pip_freeze": DETECTOR_TRAINING_PIP_FREEZE,
        "detector_training_plan": DETECTOR_PLAN,
        "detector_training_receipt": DETECTOR_TRAINING_RECEIPT,
        "registered_train_sample": (
            DEVELOPMENT_SAMPLE_ROOT / "train" / "registered_analysis_sample.csv.gz"
        ),
        "registered_train_sampling_audit": (
            DEVELOPMENT_SAMPLE_ROOT / "train" / "registered_analysis_sampling_audit.csv"
        ),
        "registered_train_sample_artifacts": (
            DEVELOPMENT_SAMPLE_ROOT / "train" / "REGISTERED_ANALYSIS_SAMPLE_ARTIFACTS.json"
        ),
        "feature_head_lock": FEATURE_HEAD_LOCK,
        "feature_head_weights": FEATURE_HEAD_WEIGHTS,
        "feature_encoder_receipt": FEATURE_ENCODER_RECEIPT,
    }

    PRESEAL_FORBIDDEN_REUSE_ROOTS = (
        DEVELOPMENT_BUNDLE_DIR,
        DEVELOPMENT_SAMPLE_ROOT / "calibration",
        DEVELOPMENT_SAMPLE_ROOT / "validation",
    )
    PRESEAL_FORBIDDEN_PUBLISHER_COMPONENTS = {
        "VisDrone2019-DET-val",
        "VisDrone2019-MOT-val",
    }

    def _assert_preseal_reuse_path_allowed(path):
        candidate = Path(path).expanduser().resolve()
        for forbidden_root in PRESEAL_FORBIDDEN_REUSE_ROOTS:
            root = Path(forbidden_root).expanduser().resolve()
            if candidate == root or root in candidate.parents:
                raise StudyGateError(
                    f"Pre-seal reuse verification cannot inspect deferred development bytes: {candidate}"
                )
        if PRESEAL_FORBIDDEN_PUBLISHER_COMPONENTS.intersection(candidate.parts):
            raise StudyGateError(
                f"Pre-seal reuse verification cannot inspect publisher validation bytes: {candidate}"
            )
        return candidate

    def _preseal_artifact_record(path, *, source_scope):
        artifact = Path(path).expanduser()
        if artifact.is_symlink():
            raise StudyGateError(f"Reused artifact must not be a symlink: {artifact}")
        try:
            resolved = _assert_preseal_reuse_path_allowed(
                artifact.resolve(strict=True)
            )
        except FileNotFoundError as exc:
            raise StudyGateError(f"Required reused artifact is missing: {artifact}") from exc
        if not resolved.is_file():
            raise StudyGateError(f"Required reused artifact is not a file: {resolved}")
        return {
            "path": str(resolved),
            "bytes": int(resolved.stat().st_size),
            "sha256": hash_file(resolved),
            "source_scope": str(source_scope),
        }

    def _verify_preseal_registered_train_artifact_manifest(inventory):
        # Parse only the fixed manifest itself before considering any entry.
        # Entry paths are never resolved or opened from untrusted JSON: the
        # complete two-entry object must equal records for already-approved,
        # already-hashed train files.
        manifest_path = PRESEAL_REUSED_V1_ARTIFACT_PATHS[
            "registered_train_sample_artifacts"
        ]
        manifest_record = inventory.get("registered_train_sample_artifacts", {})
        if (
            manifest_path.is_symlink()
            or not manifest_path.is_file()
            or str(manifest_record.get("path", "")) != str(manifest_path.resolve())
        ):
            raise StudyGateError("Registered train artifact manifest is missing or unsafe")

        def reject_duplicate_keys(pairs):
            parsed = {}
            for key, value in pairs:
                if key in parsed:
                    raise StudyGateError(
                        f"Registered train artifact manifest repeats key {key!r}"
                    )
                parsed[key] = value
            return parsed

        try:
            manifest_bytes = manifest_path.read_bytes()
        except OSError as exc:
            raise StudyGateError("Registered train artifact manifest is unreadable") from exc
        if (
            len(manifest_bytes) != int(manifest_record.get("bytes", -1))
            or hashlib.sha256(manifest_bytes).hexdigest()
            != str(manifest_record.get("sha256", ""))
        ):
            raise StudyGateError(
                "Registered train artifact manifest changed after inventory capture"
            )
        try:
            payload = json.loads(
                manifest_bytes.decode("utf-8"),
                object_pairs_hook=reject_duplicate_keys,
            )
        except StudyGateError:
            raise
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise StudyGateError("Registered train artifact manifest is unreadable") from exc

        expected_artifacts = {
            "registered_analysis_sample": {
                "path": "registered_analysis_sample.csv.gz",
                "sha256": inventory["registered_train_sample"]["sha256"],
                "bytes": inventory["registered_train_sample"]["bytes"],
            },
            "sampling_audit": {
                "path": "registered_analysis_sampling_audit.csv",
                "sha256": inventory["registered_train_sampling_audit"]["sha256"],
                "bytes": inventory["registered_train_sampling_audit"]["bytes"],
            },
        }
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        selection_sha256 = (
            str(metadata.get("selection_sha256", ""))
            if isinstance(metadata, dict) else ""
        )
        if (
            not isinstance(payload, dict)
            or set(payload) != {
                "schema_version", "workflow_version", "root", "metadata", "artifacts"
            }
            or payload.get("schema_version") != 1
            or payload.get("workflow_version") != WORKFLOW_VERSION
            or payload.get("root") != "."
            or not isinstance(metadata, dict)
            or set(metadata) != {"selection_sha256"}
            or len(selection_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in selection_sha256
            )
            or payload.get("artifacts") != expected_artifacts
        ):
            raise StudyGateError(
                "Registered train artifact manifest is not the exact two-file envelope"
            )
        return {
            "registered_analysis_sample": Path(
                inventory["registered_train_sample"]["path"]
            ),
            "sampling_audit": Path(
                inventory["registered_train_sampling_audit"]["path"]
            ),
        }

    def _read_verified_reused_artifacts_receipt(*, expected_sha256=None):
        if REUSED_ARTIFACTS_RECEIPT.is_symlink() or not REUSED_ARTIFACTS_RECEIPT.is_file():
            raise StudyGateError("The reused-v1 receipt is missing, unsafe, or not a regular file")
        try:
            receipt_bytes = REUSED_ARTIFACTS_RECEIPT.read_bytes()
            receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
            receipt_payload = json.loads(receipt_bytes.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise StudyGateError("The reused-v1 receipt is unreadable") from exc
        if not isinstance(receipt_payload, dict):
            raise StudyGateError("The reused-v1 receipt must contain one JSON object")
        if expected_sha256 is not None:
            expected = str(expected_sha256).lower()
            if (
                len(expected) != 64
                or any(character not in "0123456789abcdef" for character in expected)
                or receipt_sha256 != expected
            ):
                raise StudyGateError("The reused-v1 receipt SHA-256 changed")
        return {
            "path": REUSED_ARTIFACTS_RECEIPT.resolve(),
            "sha256": receipt_sha256,
            "payload": receipt_payload,
        }

    def _verify_or_create_reused_v1_artifacts_receipt(*, allow_create):
        reuse = cfg.models["reused_v1_artifacts"]
        if (
            reuse.get("receipt_schema") != "semuav.reused-v1-frozen-artifacts.v1"
            or reuse.get("source_run_name") != "semuav_visdrone_v1"
            or RUN_NAME != "semuav_visdrone_v2r13"
        ):
            raise StudyGateError("The v1-to-v2r13 frozen-artifact reuse identity changed")
        if not REUSED_V1_FROZEN_INPUT_DIR.is_absolute():
            raise StudyGateError("The reused v1 frozen-input root must be absolute")
        if (
            REUSED_V1_FROZEN_INPUT_DIR == FROZEN_INPUT_DIR.resolve()
            or FROZEN_INPUT_DIR.resolve() in REUSED_V1_FROZEN_INPUT_DIR.parents
        ):
            raise StudyGateError("The reused v1 root overlaps the writable v2r13 root")

        # Establish the fixed, outcome-safe inventory before parsing any
        # self-describing receipt.  This prevents a tampered receipt from
        # redirecting pre-seal verification into deferred validation material.
        inventory_paths = dict(PRESEAL_REUSED_V1_ARTIFACT_PATHS)
        inventory_paths["detector_checkpoint"] = Path(
            cfg.models["detector"]["checkpoint"]
        )
        inventory = {
            name: _preseal_artifact_record(
                path,
                source_scope=(
                    "v1_frozen_input"
                    if name in PRESEAL_REUSED_V1_ARTIFACT_PATHS
                    else "v1_receipt_bound_external_asset"
                ),
            )
            for name, path in sorted(inventory_paths.items())
        }
        expected_hashes = dict(reuse.get("expected_sha256", {}))
        required_preseal_anchors = {
            "model_lock",
            "detector_training_environment_fingerprint",
            "detector_training_pip_freeze",
            "detector_training_plan",
            "detector_training_receipt",
            "feature_head_lock",
            "registered_train_sample",
            "registered_train_sampling_audit",
        }
        if set(expected_hashes) != required_preseal_anchors:
            raise StudyGateError(
                "Reused v1 pre-seal artifact anchors are incomplete or expanded"
            )
        for name, expected_sha256 in sorted(expected_hashes.items()):
            if name not in inventory:
                raise StudyGateError(f"Unknown pre-seal v1 artifact hash anchor: {name}")
            if (
                len(str(expected_sha256)) != 64
                or inventory[name]["sha256"] != str(expected_sha256).lower()
            ):
                raise StudyGateError(f"Reused v1 artifact hash changed: {name}")

        # The smoke canary is bound to this exact registered train sample.  Its
        # portable artifact manifest contains train bytes only.
        train_artifacts = _verify_preseal_registered_train_artifact_manifest(
            inventory
        )
        if (
            train_artifacts.get("registered_analysis_sample")
            != PRESEAL_REUSED_V1_ARTIFACT_PATHS["registered_train_sample"].resolve()
            or train_artifacts.get("sampling_audit")
            != PRESEAL_REUSED_V1_ARTIFACT_PATHS[
                "registered_train_sampling_audit"
            ].resolve()
        ):
            raise StudyGateError("Registered train sample manifest names unexpected artifacts")

        verified_model_lock = verify_huggingface_model_lock(
            MODEL_LOCK, models=cfg.models,
        )
        configured_detector_receipt = Path(
            cfg.models["detector"]["training_receipt"]
        ).expanduser().resolve()
        if configured_detector_receipt != DETECTOR_TRAINING_RECEIPT.resolve():
            raise StudyGateError(
                "Configured detector receipt does not identify the reused v1 receipt"
            )
        verified_detector_checkpoint = verify_detector_checkpoint(
            cfg.models["detector"]["checkpoint"],
            cfg.models["detector"]["checkpoint_sha256"],
        )

        detector_plan_payload = json.loads(DETECTOR_PLAN.read_text(encoding="utf-8"))
        detector_receipt_payload = json.loads(
            DETECTOR_TRAINING_RECEIPT.read_text(encoding="utf-8")
        )
        try:
            receipt_plan = detector_receipt_payload["training_plan"]
            receipt_environment = detector_receipt_payload["training_environment"]
            receipt_checkpoint = detector_receipt_payload["run_artifacts"]["checkpoint"]
        except (KeyError, TypeError) as exc:
            raise StudyGateError("Detector training receipt structure is incomplete") from exc
        if (
            detector_plan_payload.get("schema_version")
            != "semuav.visdrone-detector-training-plan.v1"
            or detector_plan_payload.get("publisher_validation_used") is not False
            or detector_plan_payload.get("untouched_test_used") is not False
            or detector_plan_payload.get("publisher_partitions")
            != ["VisDrone2019-DET-train"]
            or Path(str(detector_plan_payload.get("expected_checkpoint", ""))).expanduser().resolve()
            != Path(cfg.models["detector"]["checkpoint"]).expanduser().resolve()
            or detector_receipt_payload.get("schema_version")
            != "semuav.visdrone-detector-training-receipt.v1"
            or Path(str(receipt_plan.get("path", ""))).expanduser().resolve()
            != DETECTOR_PLAN.resolve()
            or str(receipt_plan.get("sha256", ""))
            != inventory["detector_training_plan"]["sha256"]
            or Path(str(receipt_environment.get("path", ""))).expanduser().resolve()
            != DETECTOR_TRAINING_ENVIRONMENT_FINGERPRINT.resolve()
            or str(receipt_environment.get("sha256", ""))
            != inventory["detector_training_environment_fingerprint"]["sha256"]
            or Path(str(receipt_checkpoint.get("path", ""))).expanduser().resolve()
            != Path(verified_detector_checkpoint["checkpoint"]).resolve()
            or str(receipt_checkpoint.get("sha256", ""))
            != verified_detector_checkpoint["sha256"]
            or int(receipt_checkpoint.get("bytes", -1))
            != int(verified_detector_checkpoint["bytes"])
        ):
            raise StudyGateError(
                "Pre-seal detector plan/receipt/checkpoint identity changed"
            )
        verified_detector_training = {
            **detector_receipt_payload,
            "receipt_path": str(DETECTOR_TRAINING_RECEIPT.resolve()),
            "receipt_sha256": inventory["detector_training_receipt"]["sha256"],
            "status": "PRESEAL_EXACT_ANCHOR_VERIFIED",
        }

        feature_cfg = cfg.models["feature_baseline"]
        if Path(feature_cfg["receiver_head"]).expanduser().resolve() != FEATURE_HEAD_LOCK.resolve():
            raise StudyGateError("Configured feature head does not identify the reused v1 lock")
        if str(feature_cfg["receiver_head_revision"]).lower() != hash_file(FEATURE_HEAD_LOCK):
            raise StudyGateError("Configured feature-head SHA-256 differs from reused v1 bytes")
        head_lock_payload = json.loads(FEATURE_HEAD_LOCK.read_text(encoding="utf-8"))
        unhashed_head_lock = dict(head_lock_payload)
        expected_head_revision = str(unhashed_head_lock.pop("revision", ""))
        if (
            head_lock_payload.get("schema") != "semuav.visdrone-feature-head.v1"
            or head_lock_payload.get("training_split") != "train"
            or not expected_head_revision
            or hash_object(unhashed_head_lock) != expected_head_revision
            or Path(head_lock_payload["weights_path"]).expanduser().resolve()
            != FEATURE_HEAD_WEIGHTS.resolve()
            or str(head_lock_payload.get("weights_sha256", ""))
            != inventory["feature_head_weights"]["sha256"]
        ):
            raise StudyGateError("Reused train-only feature-head lock changed")
        encoder_receipt = json.loads(FEATURE_ENCODER_RECEIPT.read_text(encoding="utf-8"))
        feature_encoder_checkpoint = Path(
            head_lock_payload["encoder_checkpoint_path"]
        ).expanduser().resolve()
        inventory["feature_encoder_checkpoint"] = _preseal_artifact_record(
            feature_encoder_checkpoint,
            source_scope="v1_receipt_bound_external_asset",
        )
        if (
            encoder_receipt.get("feature_head_lock_sha256") != hash_file(FEATURE_HEAD_LOCK)
            or encoder_receipt.get("feature_head_weights_sha256") != hash_file(FEATURE_HEAD_WEIGHTS)
            or Path(str(encoder_receipt.get("encoder_checkpoint_path", ""))).expanduser().resolve()
            != feature_encoder_checkpoint
            or encoder_receipt.get("encoder_checkpoint_sha256")
            != inventory["feature_encoder_checkpoint"]["sha256"]
            or head_lock_payload.get("encoder_checkpoint_sha256")
            != inventory["feature_encoder_checkpoint"]["sha256"]
        ):
            raise StudyGateError("Reused feature-head/encoder receipt changed")

        payload = {
            "schema_version": "semuav.reused-v1-frozen-artifacts.v1",
            "source_run_name": "semuav_visdrone_v1",
            "destination_run_name": RUN_NAME,
            "reuse_mode": "immutable_read_only_reference_without_retraining",
            "source_frozen_input_dir": str(REUSED_V1_FROZEN_INPUT_DIR),
            "historical_detector_training_environment_only": True,
            "historical_environment_fingerprint_sha256": inventory[
                "detector_training_environment_fingerprint"
            ]["sha256"],
            "historical_pip_freeze_sha256": inventory[
                "detector_training_pip_freeze"
            ]["sha256"],
            "inference_runtime_identity_stored_separately": True,
            "verification_scope": (
                "preseal_registered_train_plus_model_detector_feature_provenance"
            ),
            "validation_derived_artifacts_read_before_seal": False,
            "full_development_bundle_verification_deferred_until_after_seal": True,
            "artifacts": inventory,
        }
        if REUSED_ARTIFACTS_RECEIPT.is_symlink():
            raise StudyGateError("The reused-v1 receipt must not be a symbolic link")
        if REUSED_ARTIFACTS_RECEIPT.exists():
            receipt_state = _read_verified_reused_artifacts_receipt()
            if receipt_state["payload"] != payload:
                raise StudyGateError(
                    "Reused-v1 artifact bytes differ from the frozen v2r13 receipt"
                )
        elif allow_create:
            atomic_write_json(REUSED_ARTIFACTS_RECEIPT, payload)
        else:
            raise StudyGateError(
                "REUSED_V1_FROZEN_ARTIFACTS.json must exist before validation/test"
            )
        receipt_state = _read_verified_reused_artifacts_receipt()
        if receipt_state["payload"] != payload:
            raise StudyGateError("The written reused-v1 receipt differs from its verified payload")
        return {
            "receipt": payload,
            "receipt_sha256": receipt_state["sha256"],
            "model_lock": verified_model_lock,
            "detector_training": verified_detector_training,
            "detector_checkpoint": verified_detector_checkpoint,
            "head_lock": head_lock_payload,
            "detector_plan": detector_plan_payload,
        }

    def _verify_full_reused_v1_development_bundle():
        # This operation intentionally opens validation-derived manifests.  It
        # therefore authenticates the actual on-disk scientific-use seal and
        # every bound receipt itself.  A caller-supplied status dictionary is
        # never sufficient authority to cross this boundary.
        scientific_environment = _verify_inference_environment_seal()
        development_bundle = load_prepared_visdrone_bundle(DEVELOPMENT_BUNDLE_DIR)
        registered_sample_artifacts = {}
        for split_name in ("train", "calibration", "validation"):
            registered_sample_artifacts[split_name] = verify_artifact_hash_manifest(
                DEVELOPMENT_SAMPLE_ROOT / split_name
                / "REGISTERED_ANALYSIS_SAMPLE_ARTIFACTS.json"
            )
        verified_detector_training = verify_detector_training_receipt(
            DETECTOR_TRAINING_RECEIPT,
            checkpoint=cfg.models["detector"]["checkpoint"],
            expected_checkpoint_sha256=cfg.models["detector"]["checkpoint_sha256"],
            training_plan_path=DETECTOR_PLAN,
            environment_fingerprint_path=(
                DETECTOR_TRAINING_ENVIRONMENT_FINGERPRINT
            ),
        )
        verified_detector_checkpoint = verify_detector_checkpoint(
            cfg.models["detector"]["checkpoint"],
            cfg.models["detector"]["checkpoint_sha256"],
            training_receipt=DETECTOR_TRAINING_RECEIPT,
        )
        return {
            "scientific_environment": scientific_environment,
            "development_bundle": development_bundle,
            "registered_sample_artifacts": registered_sample_artifacts,
            "detector_training": verified_detector_training,
            "detector_checkpoint": verified_detector_checkpoint,
        }

    def _read_regular_file_bytes(path, *, label):
        candidate = Path(path).expanduser()
        if candidate.is_symlink() or not candidate.is_file():
            raise StudyGateError(f"{label} is missing, unsafe, or not a regular file")
        try:
            return candidate.read_bytes()
        except OSError as exc:
            raise StudyGateError(f"{label} cannot be read safely") from exc

    def _decode_json_object_bytes(payload_bytes, *, label):
        def reject_duplicate_keys(pairs):
            payload = {}
            for key, value in pairs:
                if key in payload:
                    raise StudyGateError(f"{label} repeats key {key!r}")
                payload[key] = value
            return payload

        try:
            payload = json.loads(
                payload_bytes.decode("utf-8"),
                object_pairs_hook=reject_duplicate_keys,
            )
        except StudyGateError:
            raise
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise StudyGateError(f"{label} is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise StudyGateError(f"{label} must contain one JSON object")
        return payload

    def _trusted_code_cache_root():
        configured = str(
            cfg.runtime["required_environment_guards"]["HF_MODULES_CACHE"]
        )
        observed = os.environ.get("HF_MODULES_CACHE")
        if observed != configured:
            raise StudyGateError(
                "HF_MODULES_CACHE differs from the registered v2r13 cache root"
            )
        root = Path(configured).expanduser()
        _assert_safe_output_directory(
            root, label="v2r13 trusted-code cache", create=True,
        )
        return root.resolve()

    def _require_pristine_trusted_code_cache():
        root = _trusted_code_cache_root()
        if TRUSTED_CODE_CACHE_RECEIPT.exists() or TRUSTED_CODE_CACHE_RECEIPT.is_symlink():
            raise StudyGateError(
                "The v2r13 trusted-code receipt already exists; runtime smoke cannot rerun"
            )
        if any(root.iterdir()):
            raise StudyGateError(
                "The dedicated v2r13 trusted-code cache is not pristine"
            )
        return root

    def _trusted_code_cache_payload():
        root = _trusted_code_cache_root()
        entries = sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if entries:
            raise StudyGateError(
                "The v2r13 three-model runtime unexpectedly populated the "
                "trusted remote-code cache"
            )
        return {
            "schema_version": "semuav.visdrone-trusted-code-cache.v2",
            "scope": "no_active_trusted_remote_code",
            "scientific_results_eligible": False,
            "cache_root": str(root),
            "model_lock_path": str(MODEL_LOCK.resolve()),
            "model_lock_sha256": hash_file(MODEL_LOCK),
            "model_lock_derivation_path": str(MODEL_LOCK_DERIVATION.resolve()),
            "model_lock_derivation_sha256": hash_file(MODEL_LOCK_DERIVATION),
            "registered_model_ids": list(REGISTERED_RUNTIME_SMOKE_MODEL_IDS),
            "excluded_model_ids": ["moondream2"],
            "module_roots": [],
            "file_count": 0,
            "files": [],
            "tree_sha256": hash_object([]),
        }

    def _capture_or_verify_trusted_code_cache(*, allow_create):
        current = _trusted_code_cache_payload()
        if TRUSTED_CODE_CACHE_RECEIPT.is_symlink() or (
            TRUSTED_CODE_CACHE_RECEIPT.exists()
            and not TRUSTED_CODE_CACHE_RECEIPT.is_file()
        ):
            raise StudyGateError("The v2r13 trusted-code receipt is unsafe")
        if TRUSTED_CODE_CACHE_RECEIPT.exists():
            observed = _decode_json_object_bytes(
                _read_regular_file_bytes(
                    TRUSTED_CODE_CACHE_RECEIPT,
                    label="v2r13 trusted-code cache receipt",
                ),
                label="v2r13 trusted-code cache receipt",
            )
            if observed != current:
                raise StudyGateError("The sealed v2r13 trusted-code cache changed")
        elif not allow_create:
            raise StudyGateError(
                "The v2r13 trusted-code cache was not frozen during prepare"
            )
        else:
            atomic_write_json(TRUSTED_CODE_CACHE_RECEIPT, current)
        return {
            "payload": current,
            "path": TRUSTED_CODE_CACHE_RECEIPT.resolve(),
            "sha256": hash_file(TRUSTED_CODE_CACHE_RECEIPT),
        }

    def _capture_or_verify_inference_runtime(*, allow_create):
        _require_v2r13_preimport_source_gate()
        _require_exclusive_scientific_gpu("inference_runtime_fingerprint")
        required_python = Path(
            cfg.runtime["required_inference_python"]
        ).expanduser().resolve()
        if Path(sys.executable).resolve() != required_python:
            raise StudyGateError(
                f"v2r13 requires the clean inference Python {required_python}; "
                f"observed {Path(sys.executable).resolve()}"
            )
        lock_paths = (
            INFERENCE_ENVIRONMENT_FINGERPRINT,
            INFERENCE_PIP_FREEZE,
        )
        for lock_path in lock_paths:
            if lock_path.is_symlink() or (
                lock_path.exists() and not lock_path.is_file()
            ):
                raise StudyGateError(
                    f"The v2r13 inference-runtime lock is unsafe: {lock_path}"
                )
        current_preflight = environment_report(
            PROJECT_ROOT, cfg.paths.cache_root, DEVICE, cfg.profile.value,
        )
        if current_preflight["status"].eq("FAIL").any():
            failed = current_preflight.loc[
                current_preflight["status"].eq("FAIL"), "item"
            ].astype(str).tolist()
            raise StudyGateError(f"Inference runtime preflight failed: {failed}")
        current_environment = environment_fingerprint(current_preflight)
        current_pip_freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            check=True, text=True, capture_output=True,
        ).stdout
        existing_count = sum(path.is_file() for path in lock_paths)
        if existing_count == 1:
            raise StudyGateError("The v2r13 inference-runtime lock is incomplete")
        if existing_count == 0:
            if not allow_create:
                raise StudyGateError(
                    "The v2r13 inference-runtime lock must be frozen during prepare"
                )
            atomic_write_json(INFERENCE_ENVIRONMENT_FINGERPRINT, current_environment)
            atomic_write_text(INFERENCE_PIP_FREEZE, current_pip_freeze)
        environment_bytes = _read_regular_file_bytes(
            INFERENCE_ENVIRONMENT_FINGERPRINT,
            label="v2r13 inference environment fingerprint",
        )
        pip_freeze_bytes = _read_regular_file_bytes(
            INFERENCE_PIP_FREEZE,
            label="v2r13 inference distribution inventory",
        )
        stored_environment = _decode_json_object_bytes(
            environment_bytes,
            label="v2r13 inference environment fingerprint",
        )
        try:
            stored_pip_freeze = pip_freeze_bytes.decode("utf-8")
        except UnicodeError as exc:
            raise StudyGateError(
                "The v2r13 inference distribution inventory is not UTF-8"
            ) from exc
        if stored_environment != current_environment:
            raise StudyGateError("The v2r13 inference environment fingerprint changed")
        if stored_pip_freeze != current_pip_freeze:
            raise StudyGateError("The v2r13 inference distribution inventory changed")
        return {
            "preflight": current_preflight,
            "environment": current_environment,
            "pip_freeze": current_pip_freeze,
            "environment_sha256": hashlib.sha256(environment_bytes).hexdigest(),
            "pip_freeze_sha256": hashlib.sha256(pip_freeze_bytes).hexdigest(),
        }

    def _verified_runtime_guards():
        expected_guards = {
            "CUDA_VISIBLE_DEVICES": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "YOLO_AUTOINSTALL": "false",
            "PIP_NO_INDEX": "1",
            "PIP_NO_INPUT": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "SEMUAV_MODELS_LOCAL_ONLY": "1",
            "HF_HUB_CACHE": (
                "/data/Lane Detection Datasets/musta/semuav_vlm/"
                ".cache/huggingface"
            ),
            "HUGGINGFACE_HUB_CACHE": (
                "/data/Lane Detection Datasets/musta/semuav_vlm/"
                ".cache/huggingface"
            ),
            "TRANSFORMERS_CACHE": (
                "/data/Lane Detection Datasets/musta/semuav_vlm/"
                ".cache/huggingface"
            ),
            "HF_MODULES_CACHE": (
                "/data/Lane Detection Datasets/musta/semuav_vlm/"
                ".cache/huggingface/modules-v2r13"
            ),
        }
        configured_guards = {
            str(name): str(value)
            for name, value in cfg.runtime["required_environment_guards"].items()
        }
        if configured_guards != expected_guards:
            raise StudyGateError("The v2r13 scientific runtime-guard contract changed")
        observed_guards = {
            name: os.environ.get(name) for name in sorted(expected_guards)
        }
        if observed_guards != {
            name: expected_guards[name] for name in sorted(expected_guards)
        }:
            raise StudyGateError(
                f"Required v2r13 runtime guards are not exact: {observed_guards}"
            )
        required_absent = (
            "BASH_ENV", "ENV", "LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH",
            "SEMUAV_MOONDREAM_REVISION", "SEMUAV_MOONDREAM_TOKENIZER_LOCK",
        )
        required_revisions = (
            "SEMUAV_QWEN_REVISION", "SEMUAV_SMOLVLM_REVISION",
            "SEMUAV_UPPER_BOUND_REVISION",
        )
        expected_revision_guards = {
            name: authenticated_model_loader_environment.get(name)
            for name in required_revisions
        }
        locked_revision_guards = {
            name: locked_revision_environment.get(name)
            for name in required_revisions
        }
        observed_revision_guards = {
            name: os.environ.get(name) for name in required_revisions
        }
        contaminated = {
            name: os.environ.get(name) for name in required_absent
            if os.environ.get(name) is not None
        }
        if (
            contaminated
            or expected_revision_guards != locked_revision_guards
            or observed_revision_guards != expected_revision_guards
            or any(
                not isinstance(value, str) or len(value) != 40
                or any(character not in "0123456789abcdef" for character in value)
                for value in expected_revision_guards.values()
            )
            or os.environ.get("PYTHONNOUSERSITE") != "1"
        ):
            raise StudyGateError(
                "The v2r13 launch environment is not isolated: "
                f"contaminated={contaminated}, "
                f"revisions={observed_revision_guards}, "
                f"PYTHONNOUSERSITE={os.environ.get('PYTHONNOUSERSITE')!r}"
            )
        return {
            "required_present": observed_guards,
            "required_absent": {name: None for name in required_absent},
            "authenticated_model_revisions": observed_revision_guards,
            "PYTHONNOUSERSITE": "1",
        }

    def _verified_generation_policy():
        configured = dict(cfg.models.get("generation_policy", {}))
        configured_sha256 = str(configured.pop("sha256", "")).lower()
        expected_sha256 = visdrone_generation_policy_sha256(
            int(cfg.models["max_new_tokens"])
        )
        if (
            configured != VISDRONE_GENERATION_POLICY
            or configured_sha256 != VISDRONE_GENERATION_POLICY_SHA256
            or expected_sha256 != VISDRONE_GENERATION_POLICY_SHA256
        ):
            raise StudyGateError(
                "The v2r13 generation-policy contract differs from the registered policy"
            )
        return {
            "version": VISDRONE_GENERATION_POLICY_VERSION,
            "sha256": expected_sha256,
        }

    def _require_exclusive_scientific_gpu(stage):
        # This just-in-time occupancy gate is deliberately separate from the
        # immutable environment fingerprint because OS PIDs are volatile.
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise StudyGateError(
                "Scientific GPU execution requires exact CUDA_VISIBLE_DEVICES=1"
            )
        record = require_exclusive_physical_gpu(
            1, allowed_pids=(os.getpid(),),
        )
        print(
            "PHYSICAL_GPU_1_EXCLUSIVE_GATE_PASS",
            str(stage),
            json.dumps(record, sort_keys=True),
        )
        return record

    def _verify_candidate_ready_receipt(*, require_inference_lock=True):
        # This is the sole active candidate/deployment verifier. V2r12 is a
        # source-only sibling: it owns the active source, archive, deployment
        # receipt, and kernel registration, while reusing the
        # exact authenticated v2r3 candidate without rebuilding or installing a
        # package. The candidate receipt therefore remains bound to its original
        # v2r3 source release. The deployment receipt is an exact 52-key
        # envelope; the candidate receipt may retain additional historical audit
        # fields. Every field used for scientific sealing is checked here before
        # any model or source-data access.

        _verified_runtime_guards()
        if type(require_inference_lock) is not bool:
            raise StudyGateError("Candidate receipt lock mode must be boolean")

        study_root = Path("/data/Lane Detection Datasets/musta/semuav_vlm")
        release_id = "release_20260921_visdrone_paper_v2r13"
        run_name = "semuav_visdrone_v2r13"
        release_root = (
            study_root / "releases" / release_id
            / "SemUAV-VLM-VisDrone-Paper-Study"
        )
        candidate_root = (
            study_root / "venvs"
            / "release_20260912_visdrone_inference_candidate_v2r3"
        )
        candidate_python = candidate_root / "bin" / "python"
        candidate_receipt_path = candidate_root / "SEMUAV_CANDIDATE_READY.json"
        candidate_source_release_id = "release_20260912_visdrone_paper_v2r3"
        candidate_source_run_name = "semuav_visdrone_v2r3"
        candidate_source_root = (
            study_root / "releases" / candidate_source_release_id
            / "SemUAV-VLM-VisDrone-Paper-Study"
        )
        builder_path = (
            study_root / "incoming"
            / "build_semuav_visdrone_inference_candidate_v2r3_20260912.sh"
        )
        deployer_path = (
            study_root / "incoming"
            / "deploy_semuav_visdrone_v2r13_20260921.sh"
        )
        archive_path = (
            study_root / "incoming"
            / "SemUAV-VLM-VisDrone-Paper-Study-20260921-v2r13.zip"
        )
        deployment_receipt_path = (
            study_root / "private" / "visdrone_v2r13_deployment"
            / "DEPLOYMENT_READY.json"
        )
        source_predecessor_release_id = (
            "release_20260921_visdrone_paper_v2r10"
        )
        source_predecessor_run_name = "semuav_visdrone_v2r10"
        source_predecessor_release_root = (
            study_root / "releases" / source_predecessor_release_id
            / "SemUAV-VLM-VisDrone-Paper-Study"
        )
        source_predecessor_archive_path = (
            study_root / "incoming"
            / "SemUAV-VLM-VisDrone-Paper-Study-20260921-v2r10.zip"
        )
        source_predecessor_deployer_path = (
            study_root / "incoming"
            / "deploy_semuav_visdrone_v2r10_20260921.sh"
        )
        source_predecessor_deployment_receipt_path = (
            study_root / "private" / "visdrone_v2r10_deployment"
            / "DEPLOYMENT_READY.json"
        )
        immediate_predecessor_release_id = (
            "release_20260921_visdrone_paper_v2r11"
        )
        immediate_predecessor_run_name = "semuav_visdrone_v2r11"
        immediate_predecessor_release_root = (
            study_root / "releases" / immediate_predecessor_release_id
            / "SemUAV-VLM-VisDrone-Paper-Study"
        )
        immediate_predecessor_archive_path = (
            study_root / "incoming"
            / "SemUAV-VLM-VisDrone-Paper-Study-20260921-v2r11.zip"
        )
        immediate_predecessor_deployer_path = (
            study_root / "incoming"
            / "deploy_semuav_visdrone_v2r11_20260921.sh"
        )
        immediate_predecessor_deployment_receipt_path = (
            study_root / "private" / "visdrone_v2r11_deployment"
            / "DEPLOYMENT_READY.json"
        )
        expected_recovery_path = (
            study_root / "private" / "visdrone_v2r10_deployment"
            / "V2R9_EXP1_RECOVERY_SOURCE.json"
        )
        predecessor_release_id = "release_20260920_visdrone_paper_v2r9"
        predecessor_run_name = "semuav_visdrone_v2r9"
        predecessor_release_root = (
            study_root / "releases" / predecessor_release_id
            / "SemUAV-VLM-VisDrone-Paper-Study"
        )
        predecessor_archive_path = (
            study_root / "incoming"
            / "SemUAV-VLM-VisDrone-Paper-Study-20260920-v2r9.zip"
        )
        predecessor_deployer_path = (
            study_root / "incoming"
            / "deploy_semuav_visdrone_v2r9_20260920.sh"
        )
        predecessor_deployment_receipt_path = (
            study_root / "private" / "visdrone_v2r9_deployment"
            / "DEPLOYMENT_READY.json"
        )
        predecessor_frozen_root = (
            study_root / "outputs" / "visdrone_paper"
            / predecessor_run_name / "frozen_inputs"
        )
        predecessor_candidate = (
            study_root / "venvs"
            / "release_20260909_visdrone_inference_candidate_v2r2"
        )
        manifest_path = release_root / "RELEASE_MANIFEST.json"
        checksums_path = release_root / "RELEASE_MANIFEST_SHA256.txt"
        requirements_path = release_root / "requirements.txt"
        candidate_manifest_path = candidate_source_root / "RELEASE_MANIFEST.json"
        candidate_checksums_path = (
            candidate_source_root / "RELEASE_MANIFEST_SHA256.txt"
        )
        candidate_requirements_path = candidate_source_root / "requirements.txt"

        def require_sha256(value, label):
            digest = str(value)
            if (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise StudyGateError(f"{label} is not one lowercase SHA-256")
            return digest

        def require_mapping(value, required_keys, label):
            if not isinstance(value, dict):
                raise StudyGateError(f"{label} is not an object")
            missing = set(required_keys) - set(value)
            if missing:
                raise StudyGateError(f"{label} is missing fields: {sorted(missing)}")
            return value

        def read_and_hash(path, label):
            content = _read_regular_file_bytes(path, label=label)
            return content, hashlib.sha256(content).hexdigest()

        def verify_file_record(value, label, expected_path=None):
            record = require_mapping(value, {"path", "sha256"}, label)
            path = Path(str(record["path"])).expanduser()
            if expected_path is not None and path.resolve() != expected_path.resolve():
                raise StudyGateError(f"{label} path changed")
            content, digest = read_and_hash(path, label)
            if digest != require_sha256(record["sha256"], f"{label} digest"):
                raise StudyGateError(f"{label} bytes changed")
            if "bytes" in record and (
                type(record["bytes"]) is not int or record["bytes"] != len(content)
            ):
                raise StudyGateError(f"{label} byte count changed")
            return {"path": path, "bytes": content, "sha256": digest}

        configured_receipt = Path(
            str(cfg.runtime["candidate_ready_receipt"])
        ).expanduser()
        configured_python = Path(
            str(cfg.runtime["required_inference_python"])
        ).expanduser()
        if (
            PROJECT_ROOT.resolve() != release_root
            or release_root.is_symlink()
            or not release_root.is_dir()
            or configured_receipt != candidate_receipt_path
            or CANDIDATE_READY_RECEIPT != candidate_receipt_path
            or configured_python != candidate_python
            or not candidate_python.is_file()
            or candidate_root.is_symlink()
            or not candidate_root.is_dir()
            or candidate_source_root.is_symlink()
            or not candidate_source_root.is_dir()
            or Path(sys.executable).absolute() != candidate_python
            or Path(sys.prefix).resolve() != candidate_root.resolve()
        ):
            raise StudyGateError(
                "The active project/Python is not the exact v2r13 source deployment "
                "with the reused v2r3 candidate"
            )

        external_deployment_sha256 = require_sha256(
            os.environ.get("SEMUAV_V2R13_DEPLOYMENT_RECEIPT_SHA256", ""),
            "Externally supplied v2r13 deployment-receipt digest",
        )
        deployment_bytes, deployment_sha256 = read_and_hash(
            deployment_receipt_path, "v2r13 deployment receipt",
        )
        if deployment_sha256 != external_deployment_sha256:
            raise StudyGateError(
                "The v2r13 deployment receipt differs from its external digest"
            )
        deployment = _decode_json_object_bytes(
            deployment_bytes, label="v2r13 deployment receipt",
        )
        expected_deployment_keys = {
            "schema_version", "status", "created_at_utc", "release_id",
            "run_name", "release_root", "deployment_mode", "scope",
            "scientific_contract_changed", "candidate_requirements_changed",
            "package_installation_performed", "candidate_builder_invoked",
            "candidate_environment_modified", "base_v2r3_release_modified",
            "predecessor_v2r9_release_modified",
            "predecessor_v2r10_release_modified",
            "predecessor_v2r11_release_modified",
        "predecessor_v2r12_release_modified", "predecessor_v2r12_state",
            "scientific_validation_authorized", "scientific_validation_run",
            "scientific_experiments_run", "runtime_smoke_run",
            "untouched_test_run", "validation_or_test_images_or_labels_read",
            "gpu_accessed", "model_weights_loaded", "jupyter_launched",
            "release_revision", "release", "installer", "deployment_script",
            "builder", "source_release", "authenticated_base_v2r3",
            "authenticated_predecessor_v2r9",
            "authenticated_predecessor_v2r10",
            "authenticated_predecessor_v2r11",
        "authenticated_predecessor_v2r12",
            "predecessor_observed_terminal_boundary", "candidate_environment",
            "model_set_amendment", "temporal_feasibility_amendment",
            "source_compatibility", "unit_tests", "jupyter_kernel",
            "jupyter_launch_target", "trusted_code_cache_namespace",
            "attempt_directory", "deployment_log", "experiment1_recovery",
            "next_gate",
        }
        require_mapping(
            deployment, expected_deployment_keys, "v2r13 deployment receipt",
        )
        if (
            len(expected_deployment_keys) != 52
            or set(deployment) != expected_deployment_keys
        ):
            raise StudyGateError(
                "The v2r13 deployment receipt contains unbound extra fields"
            )
        if (
            deployment["schema_version"]
            != "semuav.visdrone-v2r13-source-deployment.v1"
            or deployment["status"] != "READY_FOR_DEVELOPMENT_SMOKE_ONLY"
            or deployment["release_id"] != release_id
            or deployment["run_name"] != run_name
            or Path(str(deployment["release_root"])).resolve() != release_root
            or deployment["deployment_mode"] not in {
                "NEW_ATOMIC_DEPLOYMENT",
                "REVERIFY_EXISTING_EXACT_DEPLOYMENT",
            }
            or deployment["scope"]
            != "immutable_v2r13_source_sibling_tests_and_kernel_registration_only"
            or deployment["next_gate"]
            != "FRESH_V2R13_THREE_MODEL_DEVELOPMENT_RUNTIME_SMOKE"
            or any(
                deployment[name] is not False
                for name in (
                    "scientific_validation_authorized",
                    "scientific_validation_run", "scientific_experiments_run",
                    "runtime_smoke_run", "untouched_test_run",
                    "validation_or_test_images_or_labels_read", "gpu_accessed",
                    "model_weights_loaded", "jupyter_launched",
                )
            )
            or deployment["package_installation_performed"] is not False
            or deployment["candidate_builder_invoked"] is not False
            or deployment["candidate_environment_modified"] is not False
            or deployment["base_v2r3_release_modified"] is not False
            or deployment["predecessor_v2r9_release_modified"] is not False
            or deployment["predecessor_v2r10_release_modified"] is not False
            or deployment["predecessor_v2r11_release_modified"] is not False
            or deployment["predecessor_v2r12_release_modified"] is not False
            or deployment["scientific_contract_changed"] is not False
            or deployment["candidate_requirements_changed"] is not False
        ):
            raise StudyGateError("The v2r13 deployment identity or safety state changed")

        recovery_record = require_mapping(
            deployment["experiment1_recovery"], {
                "path", "bytes", "sha256", "schema_version", "status",
                "artifact_count", "inference_row_count",
            }, "v2r9 Experiment-1 recovery record",
        )
        recovery_file = verify_file_record(
            recovery_record,
            "v2r9 Experiment-1 recovery receipt",
            expected_recovery_path,
        )
        if (
            set(recovery_record) != {
                "path", "bytes", "sha256", "schema_version", "status",
                "artifact_count", "inference_row_count",
            }
            or recovery_record["schema_version"]
            != "semuav.visdrone-v2r9-exp1-recovery.v1"
            or recovery_record["status"]
            != "QUIESCENT_V2R9_EXP1_SOURCE_FROZEN"
            or recovery_record["artifact_count"] != 132
            or recovery_record["inference_row_count"] != 4152
        ):
            raise StudyGateError(
                "The v2r9 Experiment-1 recovery receipt contract changed"
            )
        recovery_payload = _decode_json_object_bytes(
            recovery_file["bytes"], label="v2r9 Experiment-1 recovery receipt",
        )
        expected_recovery_keys = {
            "schema_version", "status", "created_at_utc",
            "predecessor_release_id", "predecessor_run_name",
            "predecessor_run_root", "predecessor_config_sha256",
            "predecessor_code_sha256", "predecessor_model_lock_sha256",
            "run_manifest", "artifact_count", "inference_row_count",
            "artifact_inventory_sha256", "artifacts", "untouched_test_accessed",
            "protocol_frozen", "native_cache_completed",
        }
        if set(recovery_payload) != expected_recovery_keys:
            raise StudyGateError(
                "The v2r9 Experiment-1 recovery receipt envelope changed"
            )
        recovery_config_sha256 = require_sha256(
            recovery_payload["predecessor_config_sha256"],
            "v2r9 recovery config digest",
        )
        recovery_run_root = (
            study_root / ".cache" / "visdrone_paper_v2r9" / "runs"
            / (
                "semuav_visdrone_v2r9-full-d799763aef-"
                "fbf80e5ec3"
            )
        )
        if (
            recovery_payload["schema_version"]
            != "semuav.visdrone-v2r9-exp1-recovery.v1"
            or recovery_payload["status"]
            != "QUIESCENT_V2R9_EXP1_SOURCE_FROZEN"
            or recovery_payload["predecessor_release_id"] != predecessor_release_id
            or recovery_payload["predecessor_run_name"] != predecessor_run_name
            or Path(str(recovery_payload["predecessor_run_root"])).resolve()
            != recovery_run_root
            or not recovery_config_sha256.startswith("d799763aef")
            or recovery_payload["predecessor_code_sha256"]
            != "fbf80e5ec3067ff8ca9c150c71a0ec2a3bcf5cffe9ec3663bb79c973789bdf76"
            or recovery_payload["predecessor_model_lock_sha256"]
            != "340dd59ca55a9be491ac806b50f9778a12d2b0b4a447485627aaf422dda05353"
            or recovery_payload["artifact_count"] != 132
            or recovery_payload["inference_row_count"] != 4152
            or recovery_payload["untouched_test_accessed"] is not False
            or recovery_payload["protocol_frozen"] is not False
            or recovery_payload["native_cache_completed"] is not False
        ):
            raise StudyGateError(
                "The authenticated v2r9 Experiment-1 recovery boundary changed"
            )
        recovery_run_manifest = require_mapping(
            recovery_payload["run_manifest"], {"path", "bytes", "sha256"},
            "v2r9 recovery run-manifest record",
        )
        if (
            set(recovery_run_manifest) != {"path", "bytes", "sha256"}
            or Path(str(recovery_run_manifest["path"])).resolve()
            != recovery_run_root / "run_manifest.json"
            or type(recovery_run_manifest["bytes"]) is not int
            or recovery_run_manifest["bytes"] <= 0
        ):
            raise StudyGateError("The v2r9 recovery run-manifest record changed")
        require_sha256(
            recovery_run_manifest["sha256"], "v2r9 recovery run-manifest digest",
        )
        recovery_artifacts = recovery_payload["artifacts"]
        if not isinstance(recovery_artifacts, list) or len(recovery_artifacts) != 132:
            raise StudyGateError("The v2r9 recovery artifact inventory changed")
        recovery_rows = 0
        recovery_paths = set()
        for recovery_artifact in recovery_artifacts:
            if not isinstance(recovery_artifact, dict) or set(recovery_artifact) != {
                "relative_path", "bytes", "sha256", "rows",
            }:
                raise StudyGateError("A v2r9 recovery artifact record is malformed")
            relative_text = str(recovery_artifact["relative_path"])
            relative_path = PurePosixPath(relative_text)
            if (
                relative_path.is_absolute()
                or relative_path.as_posix() != relative_text
                or any(part in {"", ".", ".."} for part in relative_path.parts)
                or len(relative_path.parts) != 3
                or relative_path.parts[0] != "stages"
                or not relative_path.parts[1].startswith("01_vlm_")
                or not relative_path.parts[2].endswith(".csv.gz")
                or relative_text in recovery_paths
                or type(recovery_artifact["bytes"]) is not int
                or recovery_artifact["bytes"] <= 0
                or type(recovery_artifact["rows"]) is not int
                or recovery_artifact["rows"] <= 0
            ):
                raise StudyGateError("A v2r9 recovery artifact path/count is unsafe")
            require_sha256(
                recovery_artifact["sha256"],
                f"v2r9 recovery artifact digest for {relative_text}",
            )
            recovery_paths.add(relative_text)
            recovery_rows += recovery_artifact["rows"]
        if (
            recovery_rows != 4152
            or hash_object(recovery_artifacts)
            != require_sha256(
                recovery_payload["artifact_inventory_sha256"],
                "v2r9 recovery artifact-inventory digest",
            )
        ):
            raise StudyGateError("The v2r9 recovery artifact inventory totals changed")
        if (
            os.environ.get("SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT")
            != str(expected_recovery_path)
            or os.environ.get("SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT_SHA256")
            != recovery_file["sha256"]
        ):
            raise StudyGateError(
                "The authenticated recovery environment differs from deployment"
            )

        expected_sealed_predecessor_evidence = {
            'active_model_lock_path': str(study_root / 'outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/MODEL_LOCK.json'),
            'active_model_lock_sha256': '16bfbc1d2365ac0a0adb2ff20d2f5785fa3f1b05675159304aedf4c6f7c1cd2a',
            'active_model_set_sha256': '6f83118bb62cd68c98e9ee2afd947022c04088b9329a0eeef832eba17c2fb08c',
            'archive_member_count': 113,
            'archive_path': str(study_root / 'incoming/SemUAV-VLM-VisDrone-Paper-Study-20260921-v2r12.zip'),
            'archive_sha256': 'a53600fff9f50174af6c4ed17352900981cd9684333c4a54749a427be7a1bdc3',
            'builder_sha256': '799c5fc179fd6e5c4e62ec6dae081528ba5cddbc123a5647c526f80f1ebdbc00',
            'checksum_inventory_sha256': '8fd90b8ba5f32bc0da435119db9bff09fd7c908b6388887e67cfa430b8266285',
            'deployer_path': str(study_root / 'incoming/deploy_semuav_visdrone_v2r12_20260921.sh'),
            'deployer_sha256': '622699a1634429d30e8bd4a1a9817ead0b28f99fad46b2fc36726d6246552d4b',
            'deployment_receipt': str(study_root / 'private/visdrone_v2r12_deployment/DEPLOYMENT_READY.json'),
            'deployment_receipt_sha256': '889850427153f9559fc038e2f10df174874dd59e8bff0473f61dcce1b62d2463',
            'experiment1_recovery_artifact_count': 132,
            'experiment1_recovery_receipt': str(study_root / 'private/visdrone_v2r10_deployment/V2R9_EXP1_RECOVERY_SOURCE.json'),
            'experiment1_recovery_receipt_bytes': 37416,
            'experiment1_recovery_receipt_sha256': '6ca88cb5e82c8b33e27fd229443e233a48323d91fd73d0e79ec51ff8fc61a608',
            'experiment1_recovery_row_count': 4152,
            'inference_environment_fingerprint_path': str(study_root / 'outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/INFERENCE_ENVIRONMENT_FINGERPRINT.json'),
            'inference_environment_fingerprint_sha256': '39d293199a08ada1465ca335fb9481d7e7548ad20d4ab26493c68910ed771f81',
            'inference_pip_freeze_path': str(study_root / 'outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/INFERENCE_PIP_FREEZE.txt'),
            'inference_pip_freeze_sha256': '5b6aacbcd2b891c1bf968511d17b3385aff2233e99d0bae440f81bc7f977d5d2',
            'model_lock_derivation_path': str(study_root / 'outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/MODEL_LOCK_DERIVATION.json'),
            'model_lock_derivation_sha256': '14b0623dd68166608a7f27ee55a2e87ee945ae3fd28e8b72daff472c4ff99de7',
            'model_status': {'qwen3_vl_2b': 'PASS', 'qwen3_vl_8b_upper_bound': 'PASS', 'smolvlm2_2b': 'PASS'},
            'notebook_sha256': '06b47108768a7ce566ddf480f285148f15797a87055d46e6d179e15aa69a3e80',
            'payload_file_count': 111,
            'payload_total_uncompressed_bytes': 4188275,
            'release_id': 'release_20260921_visdrone_paper_v2r12',
            'release_manifest_sha256': '69bfb381877a419c698ba381f2af637208462ae764dd2f1b8abbb9c8c92729d5',
            'release_path': str(study_root / 'releases/release_20260921_visdrone_paper_v2r12/SemUAV-VLM-VisDrone-Paper-Study'),
            'requirements_sha256': '028a2757d2e1565db90ff26a26fd6e564bd592c868ed093d632cb930d438dfa5',
            'reused_artifacts_receipt_path': str(study_root / 'outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/REUSED_V1_FROZEN_ARTIFACTS.json'),
            'reused_artifacts_receipt_sha256': '68839ef0f3d3cf5b52a4b986c4fb24e0238f40b3bce44e13e45d6106ca22b811',
            'run_name': 'semuav_visdrone_v2r12',
            'runtime_smoke_all_passed': True,
            'runtime_smoke_failure_count': 0,
            'runtime_smoke_input_path': str(study_root / 'outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/MODEL_RUNTIME_SMOKE_INPUT.json'),
            'runtime_smoke_input_sha256': 'f08a98d1b5f094c1254bbd1ce7a80d13de24d74bc544ddbe2151ff04f7160b1f',
            'runtime_smoke_isolation_schema': 'semuav.runtime-smoke-subprocess.v2',
            'runtime_smoke_pass_count': 3,
            'runtime_smoke_report_path': str(study_root / 'outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/MODEL_RUNTIME_SMOKE.json'),
            'runtime_smoke_report_sha256': '0d43d5045adb0d3bea3aca2650ca8c832c076e60932c6662dbcd0abb13799fb2',
            'runtime_smoke_run': True,
            'runtime_smoke_schema_version': 4,
            'scientific_environment_seal_path': str(study_root / 'outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/INFERENCE_ENVIRONMENT_SEAL.json'),
            'scientific_environment_seal_sha256': 'fa7fbaaa0b16090e48724389b455ad5d8e21490d0af4e14f55d62557bfab81d7',
            'scientific_environment_sealed': True,
            'scientific_experiments_started': True,
            'scientific_validation_authorized': True,
            'source_deployment_ready': True,
            'trusted_code_cache_receipt_path': str(study_root / 'outputs/visdrone_paper/semuav_visdrone_v2r12/frozen_inputs/TRUSTED_CODE_CACHE.json'),
            'trusted_code_cache_receipt_sha256': 'c2b821ad0c04b6eb1bbc5d11f505361a55cea29ca046309f47f4b521a900e2f5',
            'unchanged_after_deployment_actions': True,
            'untouched_test_accessed': False,
            'validation_experiments_started': True,
        }
        sealed_predecessor_evidence = require_mapping(
            deployment["authenticated_predecessor_v2r12"],
            set(expected_sealed_predecessor_evidence), "authenticated v2r12 predecessor evidence",
        )
        if sealed_predecessor_evidence != expected_sealed_predecessor_evidence:
            raise StudyGateError("The authenticated v2r12 predecessor evidence changed")
        predecessor_state = deployment.get("predecessor_v2r12_state")
        expected_state_paths = {
            "release_root": str(study_root / "releases/release_20260921_visdrone_paper_v2r12/SemUAV-VLM-VisDrone-Paper-Study"),
            "output_root": str(study_root / "outputs/visdrone_paper/semuav_visdrone_v2r12"),
            "cache_root": str(study_root / ".cache/visdrone_paper_v2r12"),
            "trusted_code_cache_root": str(study_root / ".cache/huggingface/modules-v2r12"),
        }
        expected_state_keys = set(expected_state_paths) | {
            "schema_version", "release_tree", "output_tree", "cache_tree",
            "trusted_code_cache_tree", "unchanged_after_deployment_actions",
        }
        if (not isinstance(predecessor_state, dict) or set(predecessor_state) != expected_state_keys
            or predecessor_state.get("schema_version") != "semuav.visdrone-v2r12-preserved-state.v1"
            or predecessor_state.get("unchanged_after_deployment_actions") is not True
            or any(predecessor_state.get(key) != value for key, value in expected_state_paths.items())):
            raise StudyGateError("The preserved v2r12 state record changed")
        for state_key in ("release_tree", "output_tree", "cache_tree", "trusted_code_cache_tree"):
            state_parts = str(predecessor_state[state_key]).split()
            if (len(state_parts) != 3 or len(state_parts[0]) != 64
                or any(c not in "0123456789abcdef" for c in state_parts[0])
                or not all(part.isascii() and part.isdigit() for part in state_parts[1:])
                or (state_key != "trusted_code_cache_tree" and any(int(part) <= 0 for part in state_parts[1:]))):
                raise StudyGateError("The preserved v2r12 tree fingerprint is malformed")
        sealed_predecessor_records = {}
        sealed_predecessor_root = Path(sealed_predecessor_evidence["release_path"])
        for label, path_key, digest_key in (
            ("archive", "archive_path", "archive_sha256"),
            ("deployer", "deployer_path", "deployer_sha256"),
            ("deployment", "deployment_receipt", "deployment_receipt_sha256"),
            ("recovery", "experiment1_recovery_receipt", "experiment1_recovery_receipt_sha256"),
            ("reused", "reused_artifacts_receipt_path", "reused_artifacts_receipt_sha256"),
            ("environment", "inference_environment_fingerprint_path", "inference_environment_fingerprint_sha256"),
            ("pip_freeze", "inference_pip_freeze_path", "inference_pip_freeze_sha256"),
            ("smoke_input", "runtime_smoke_input_path", "runtime_smoke_input_sha256"),
            ("smoke_report", "runtime_smoke_report_path", "runtime_smoke_report_sha256"),
            ("model_lock", "active_model_lock_path", "active_model_lock_sha256"),
            ("derivation", "model_lock_derivation_path", "model_lock_derivation_sha256"),
            ("trusted_cache", "trusted_code_cache_receipt_path", "trusted_code_cache_receipt_sha256"),
            ("seal", "scientific_environment_seal_path", "scientific_environment_seal_sha256"),
        ):
            sealed_predecessor_records[label] = verify_file_record(
                {"path": sealed_predecessor_evidence[path_key], "sha256": sealed_predecessor_evidence[digest_key]},
                "v2r12 predecessor " + label, Path(sealed_predecessor_evidence[path_key]),
            )
        for label, name, digest_key in (
            ("manifest", "RELEASE_MANIFEST.json", "release_manifest_sha256"),
            ("checksums", "RELEASE_MANIFEST_SHA256.txt", "checksum_inventory_sha256"),
            ("builder", "build_visdrone_paper_notebook.py", "builder_sha256"),
            ("requirements", "requirements.txt", "requirements_sha256"),
        ):
            sealed_predecessor_records[label] = verify_file_record(
                {"path": sealed_predecessor_root / name, "sha256": sealed_predecessor_evidence[digest_key]},
                "v2r12 predecessor " + label, sealed_predecessor_root / name,
            )
        sealed_predecessor_manifest_payload = _decode_json_object_bytes(
            sealed_predecessor_records["manifest"]["bytes"], label="v2r12 predecessor manifest",
        )
        sealed_predecessor_deployment = _decode_json_object_bytes(
            sealed_predecessor_records["deployment"]["bytes"], label="v2r12 predecessor deployment",
        )
        if (
            sealed_predecessor_manifest_payload.get("release_id") != sealed_predecessor_evidence["release_id"]
            or sealed_predecessor_manifest_payload.get("run_name") != sealed_predecessor_evidence["run_name"]
            or sealed_predecessor_deployment.get("release_id") != sealed_predecessor_evidence["release_id"]
            or sealed_predecessor_deployment.get("schema_version") != "semuav.visdrone-v2r12-source-deployment.v1"
            or sealed_predecessor_deployment.get("runtime_smoke_run") is not False
            or sealed_predecessor_deployment.get("scientific_experiments_run") is not False
            or sealed_predecessor_deployment.get("untouched_test_run") is not False
            or any(sealed_predecessor_manifest_payload.get(key) != deployment[key]
                   or sealed_predecessor_deployment.get(key) != deployment[key]
                   for key in ("authenticated_predecessor_v2r9", "authenticated_predecessor_v2r10", "authenticated_predecessor_v2r11"))
        ):
            raise StudyGateError("The v2r12 historical deployment identity changed")
        expected_v2r12_terminal_boundary = {'correction_record': 'PRE_VALIDATION_LOW_EVIDENCE_PACKET_CORRECTION_20260921_V2R13.md',
         'correction_record_sha256': low_evidence_packet_correction_sha256,
         'evidence_model': 'authenticated_source_runtime_seal_and_operator_observed_validation_stop',
         'inherited_v2r10_terminal_boundary_preserved': True,
         'inherited_v2r11_terminal_boundary_preserved': True,
         'inherited_v2r9_terminal_boundary_preserved': True,
         'operator_observed_execution': {'deployment_receipt_sha256': '889850427153f9559fc038e2f10df174874dd59e8bff0473f61dcce1b62d2463',
                                         'experiment2_frontier_completed_in_observed_invocation': False,
                                         'failure_call_path': ['run_validation_perception_and_frontier',
                                                               '_resolve_native_cache',
                                                               'build_native_receiver_cache',
                                                               'nvml_measurement_runner',
                                                               'RegisteredVisDroneNativeReceiver.__call__',
                                                               'RegisteredVisDroneNativeReceiver._frame_action',
                                                               'DetectorOnlyBackend.infer',
                                                               'validate_semantic_content_packet',
                                                               'validate_semantic_packet'],
                                         'failure_exception_type': 'ValueError',
                                         'failure_kind': 'positive_event_with_insufficient_evidence_rejected_by_frozen_semantic_schema',
                                         'failure_message': 'Insufficient evidence requires '
                                                            'event=insufficient_evidence and '
                                                            'recommended_action=REVIEW',
                                         'failure_stage': 'validation_native_receiver_cache_visual_payload_warmup',
                                         'later_scientific_artifacts_must_be_absent_at_deployment': True,
                                         'later_validation_completion_observed': False,
                                         'native_receiver_cache_build_returned': False,
                                         'protocol_freeze_observed': False,
                                         'report_or_figures_completion_observed': False,
                                         'runtime_smoke_three_of_three_passed': True,
                                         'scientific_environment_sealed': True,
                                         'source_deployment_ready': True,
                                         'trusted_code_cache_frozen': True,
                                         'untouched_test_access_observed': False,
                                         'v2r9_experiment1_recovery_artifact_count': 132,
                                         'v2r9_experiment1_recovery_receipt_preserved': True,
                                         'v2r9_experiment1_recovery_row_count': 4152,
                                         'validation_experiments_1_2_entered': True,
                                         'validation_experiments_1_2_returned': False},
         'predecessor_release_id': 'release_20260921_visdrone_paper_v2r12',
         'predecessor_run_name': 'semuav_visdrone_v2r12',
         'schema_version': 'semuav.visdrone-v2r12-operator-observed-terminal-boundary.v1',
         'source_authenticated_defect': {'affected_entrypoint': 'semuav.perception.DetectorOnlyBackend.infer',
                                         'correction': 'normalize_event_to_insufficient_evidence_before_existing_action_mapping_when_evidence_is_false',
                                         'defect_kind': 'positive_event_survives_sufficient_evidence_false',
                                         'mission_ontology_changed': False,
                                         'predecessor_source_or_receipts_mutated': False,
                                         'previously_schema_valid_predictions_changed': False,
                                         'registered_dataset_or_split_changed': False,
                                         'registered_models_changed': False,
                                         'requirements_changed': False,
                                         'schema_version': 'semuav.visdrone-v2r13-low-evidence-packet-correction.v1',
                                         'scientific_contract_changed': False,
                                         'semantic_schema_changed': False,
                                         'thresholds_changed': False},
         'v2r12_derived_scientific_outputs_reused': False,
         'v2r13_derived_results_recomputation_required': True,
         'v2r13_downstream_results_recomputation_required': True,
         'v2r13_fresh_runtime_smoke_required': True,
         'v2r13_raw_experiment1_recovery_authorized': True,
         'v2r13_raw_experiment1_recovery_schema': 'semuav.visdrone-v2r9-exp1-recovery.v1'}
        observed_terminal_boundary = require_mapping(
            deployment["predecessor_observed_terminal_boundary"],
            set(expected_v2r12_terminal_boundary), "operator-observed v2r12 terminal boundary",
        )
        if observed_terminal_boundary != expected_v2r12_terminal_boundary:
            raise StudyGateError("The operator-observed v2r12 terminal boundary changed")

        immediate_predecessor_evidence_required = {
            "release_id", "run_name", "release_path", "archive_path",
            "archive_sha256", "release_manifest_sha256",
            "checksum_inventory_sha256", "payload_file_count",
            "archive_member_count", "payload_total_uncompressed_bytes",
            "builder_sha256", "notebook_sha256", "deployer_path",
            "deployer_sha256", "deployment_receipt",
            "deployment_receipt_sha256", "experiment1_recovery_receipt",
            "experiment1_recovery_receipt_sha256",
            "experiment1_recovery_receipt_bytes",
            "experiment1_recovery_artifact_count",
            "experiment1_recovery_row_count", "active_model_set_sha256",
            "requirements_sha256", "source_deployment_ready",
            "runtime_smoke_run", "scientific_environment_sealed",
            "scientific_validation_authorized",
            "scientific_experiments_started",
            "validation_experiments_started", "untouched_test_accessed",
            "unchanged_after_deployment_actions",
        }
        immediate_predecessor_evidence = require_mapping(
            deployment["authenticated_predecessor_v2r11"],
            immediate_predecessor_evidence_required,
            "authenticated hash-backed v2r11 immediate predecessor evidence",
        )
        expected_immediate_predecessor_evidence = {
            "release_id": immediate_predecessor_release_id,
            "run_name": immediate_predecessor_run_name,
            "release_path": str(immediate_predecessor_release_root),
            "archive_path": str(immediate_predecessor_archive_path),
            "archive_sha256": (
                "737c168c8710d9f111f13ce138fefc9b3dc2f2bc4da9fc858a3cc11e76c465b8"
            ),
            "release_manifest_sha256": (
                "b98c11aefe1617e6d2bffcaf9107b0708d31f45040db76d959a8c6664d7ee2ec"
            ),
            "checksum_inventory_sha256": (
                "a761c0f0ae1f8d1ded79d388885f25c7dc4766cf94fe41b7120aa24d620e6ab0"
            ),
            "payload_file_count": 109,
            "archive_member_count": 111,
            "payload_total_uncompressed_bytes": 4063633,
            "builder_sha256": (
                "c2edf8ebec76727fcaf2d785c01a3555704d4b5b639b8c373021588b1acd1447"
            ),
            "notebook_sha256": (
                "9b6b4d4585036d0d2c45df1fce21af0f4a08fbcc90f9406cdeefb4c690b2154c"
            ),
            "deployer_path": str(immediate_predecessor_deployer_path),
            "deployer_sha256": (
                "ee4ce4ad394e46eb85d4c83040f9f95020bf685436e06a2dfc1b8411d5905bb5"
            ),
            "deployment_receipt": str(
                immediate_predecessor_deployment_receipt_path
            ),
            "deployment_receipt_sha256": (
                "befdce3067b571933dd94d35758c2c29385a06a8bb02ea93ae681117dcc17f3e"
            ),
            "experiment1_recovery_receipt": str(expected_recovery_path),
            "experiment1_recovery_receipt_sha256": (
                "6ca88cb5e82c8b33e27fd229443e233a48323d91fd73d0e79ec51ff8fc61a608"
            ),
            "experiment1_recovery_receipt_bytes": 37416,
            "experiment1_recovery_artifact_count": 132,
            "experiment1_recovery_row_count": 4152,
            "active_model_set_sha256": (
                "6f83118bb62cd68c98e9ee2afd947022c04088b9329a0eeef832eba17c2fb08c"
            ),
            "requirements_sha256": (
                "028a2757d2e1565db90ff26a26fd6e564bd592c868ed093d632cb930d438dfa5"
            ),
            "source_deployment_ready": True,
            "runtime_smoke_run": False,
            "scientific_environment_sealed": False,
            "scientific_validation_authorized": False,
            "scientific_experiments_started": False,
            "validation_experiments_started": False,
            "untouched_test_accessed": False,
            "unchanged_after_deployment_actions": True,
        }
        if (
            set(immediate_predecessor_evidence)
            != immediate_predecessor_evidence_required
            or immediate_predecessor_evidence
            != expected_immediate_predecessor_evidence
        ):
            raise StudyGateError(
                "The authenticated hash-backed v2r11 immediate predecessor "
                "evidence changed"
            )

        immediate_predecessor_archive = verify_file_record(
            {
                "path": immediate_predecessor_evidence["archive_path"],
                "sha256": immediate_predecessor_evidence["archive_sha256"],
            },
            "v2r11 immediate predecessor archive",
            immediate_predecessor_archive_path,
        )
        immediate_predecessor_manifest = verify_file_record(
            {
                "path": immediate_predecessor_release_root
                / "RELEASE_MANIFEST.json",
                "sha256": immediate_predecessor_evidence[
                    "release_manifest_sha256"
                ],
            },
            "v2r11 immediate predecessor release manifest",
            immediate_predecessor_release_root / "RELEASE_MANIFEST.json",
        )
        verify_file_record(
            {
                "path": immediate_predecessor_release_root
                / "RELEASE_MANIFEST_SHA256.txt",
                "sha256": immediate_predecessor_evidence[
                    "checksum_inventory_sha256"
                ],
            },
            "v2r11 immediate predecessor checksum inventory",
            immediate_predecessor_release_root / "RELEASE_MANIFEST_SHA256.txt",
        )
        verify_file_record(
            {
                "path": immediate_predecessor_evidence["deployer_path"],
                "sha256": immediate_predecessor_evidence["deployer_sha256"],
            },
            "v2r11 immediate predecessor deployer",
            immediate_predecessor_deployer_path,
        )
        immediate_predecessor_deployment_receipt = verify_file_record(
            {
                "path": immediate_predecessor_evidence["deployment_receipt"],
                "sha256": immediate_predecessor_evidence[
                    "deployment_receipt_sha256"
                ],
            },
            "v2r11 immediate predecessor deployment receipt",
            immediate_predecessor_deployment_receipt_path,
        )
        # Parse now; its exact v2r10 boundary is checked after the current
        # v2r11 top-level boundary so each provenance layer remains distinct.
        immediate_predecessor_manifest_payload = _decode_json_object_bytes(
            immediate_predecessor_manifest["bytes"],
            label="v2r11 immediate predecessor release manifest",
        )
        immediate_predecessor_deployment = _decode_json_object_bytes(
            immediate_predecessor_deployment_receipt["bytes"],
            label="v2r11 immediate predecessor deployment receipt",
        )
        if (
            immediate_predecessor_manifest_payload.get("release_id")
            != immediate_predecessor_release_id
            or immediate_predecessor_manifest_payload.get("run_name")
            != immediate_predecessor_run_name
            or immediate_predecessor_manifest_payload.get("payload_file_count")
            != immediate_predecessor_evidence["payload_file_count"]
            or immediate_predecessor_deployment.get("schema_version")
            != "semuav.visdrone-v2r11-source-deployment.v1"
            or immediate_predecessor_deployment.get("release_id")
            != immediate_predecessor_release_id
            or immediate_predecessor_deployment.get("run_name")
            != immediate_predecessor_run_name
            or immediate_predecessor_deployment.get("runtime_smoke_run")
            is not False
            or immediate_predecessor_deployment.get(
                "scientific_validation_authorized"
            ) is not False
            or immediate_predecessor_deployment.get("scientific_experiments_run")
            is not False
            or immediate_predecessor_deployment.get("untouched_test_run")
            is not False
        ):
            raise StudyGateError(
                "The pinned v2r11 immediate predecessor state changed"
            )

        source_predecessor_evidence_required = {
            "release_id", "run_name", "release_path", "archive_path",
            "archive_sha256", "release_manifest_sha256",
            "checksum_inventory_sha256", "requirements_sha256",
            "deployer_path", "deployer_sha256", "deployment_receipt",
            "deployment_receipt_sha256", "payload_file_count",
            "archive_member_count", "payload_total_uncompressed_bytes",
            "builder_sha256", "notebook_sha256",
            "experiment1_recovery_receipt",
            "experiment1_recovery_receipt_sha256",
            "experiment1_recovery_receipt_bytes",
            "experiment1_recovery_artifact_count",
            "experiment1_recovery_row_count", "active_model_set_sha256",
            "runtime_smoke_all_passed", "runtime_smoke_pass_count",
            "scientific_environment_sealed", "validation_experiments_started",
            "untouched_test_accessed", "unchanged_after_deployment_actions",
        }
        source_predecessor_evidence = require_mapping(
            deployment["authenticated_predecessor_v2r10"],
            source_predecessor_evidence_required,
            "authenticated hash-backed v2r10 source predecessor evidence",
        )
        expected_source_predecessor_evidence = {
            "release_id": source_predecessor_release_id,
            "run_name": source_predecessor_run_name,
            "release_path": str(source_predecessor_release_root),
            "archive_path": str(source_predecessor_archive_path),
            "archive_sha256": (
                "f35723efcbda801bc1c3a8796e2fa6603b8d8bcacdcf66b7186df8d7a314af6a"
            ),
            "release_manifest_sha256": (
                "99323dd6109eca6d4942a120a3bb383f077d4a8f95869169f57f75e57574ac3c"
            ),
            "checksum_inventory_sha256": (
                "53bd76f3ffd5510b0d8c8cef6bfb82a6bdf63562c6dadd08c849d224e3513649"
            ),
            "requirements_sha256": (
                "028a2757d2e1565db90ff26a26fd6e564bd592c868ed093d632cb930d438dfa5"
            ),
            "deployer_path": str(source_predecessor_deployer_path),
            "deployer_sha256": (
                "e8b49ea192b955bf07f8b53a0f6130f3eb0bd182c3a1ea28ede6f10e89bf79cf"
            ),
            "deployment_receipt": str(
                source_predecessor_deployment_receipt_path
            ),
            "deployment_receipt_sha256": (
                "95a64dad33ae9fa21974b8ba629bc9ec972ad989245bef5b38df42b00b2fb2e3"
            ),
            "payload_file_count": 108,
            "archive_member_count": 110,
            "payload_total_uncompressed_bytes": 4009068,
            "builder_sha256": (
                "09478e1f93768d80b589110e45edc05fa0c1cb3cd63bbf6e6e29ed15a27a73ab"
            ),
            "notebook_sha256": (
                "f35bde4c03536b97bf82ab719ad5a516225bcc321d5a878971410b4e86a16108"
            ),
            "experiment1_recovery_receipt": str(expected_recovery_path),
            "experiment1_recovery_receipt_sha256": (
                "6ca88cb5e82c8b33e27fd229443e233a48323d91fd73d0e79ec51ff8fc61a608"
            ),
            "experiment1_recovery_receipt_bytes": 37416,
            "experiment1_recovery_artifact_count": 132,
            "experiment1_recovery_row_count": 4152,
            "active_model_set_sha256": (
                "6f83118bb62cd68c98e9ee2afd947022c04088b9329a0eeef832eba17c2fb08c"
            ),
            "runtime_smoke_all_passed": True,
            "runtime_smoke_pass_count": 3,
            "scientific_environment_sealed": True,
            "validation_experiments_started": False,
            "untouched_test_accessed": False,
            "unchanged_after_deployment_actions": True,
        }
        if (
            set(source_predecessor_evidence)
            != source_predecessor_evidence_required
            or source_predecessor_evidence
            != expected_source_predecessor_evidence
        ):
            raise StudyGateError(
                "The authenticated hash-backed v2r10 source predecessor "
                "evidence changed"
            )

        source_predecessor_archive = verify_file_record(
            {
                "path": source_predecessor_evidence["archive_path"],
                "sha256": source_predecessor_evidence["archive_sha256"],
            },
            "v2r10 source predecessor archive",
            source_predecessor_archive_path,
        )
        source_predecessor_manifest = verify_file_record(
            {
                "path": source_predecessor_release_root / "RELEASE_MANIFEST.json",
                "sha256": source_predecessor_evidence[
                    "release_manifest_sha256"
                ],
            },
            "v2r10 source predecessor release manifest",
            source_predecessor_release_root / "RELEASE_MANIFEST.json",
        )
        verify_file_record(
            {
                "path": (
                    source_predecessor_release_root
                    / "RELEASE_MANIFEST_SHA256.txt"
                ),
                "sha256": source_predecessor_evidence[
                    "checksum_inventory_sha256"
                ],
            },
            "v2r10 source predecessor checksum inventory",
            source_predecessor_release_root / "RELEASE_MANIFEST_SHA256.txt",
        )
        verify_file_record(
            {
                "path": source_predecessor_release_root / "requirements.txt",
                "sha256": source_predecessor_evidence["requirements_sha256"],
            },
            "v2r10 source predecessor requirements",
            source_predecessor_release_root / "requirements.txt",
        )
        verify_file_record(
            {
                "path": source_predecessor_evidence["deployer_path"],
                "sha256": source_predecessor_evidence["deployer_sha256"],
            },
            "v2r10 source predecessor deployer",
            source_predecessor_deployer_path,
        )
        source_predecessor_deployment_receipt = verify_file_record(
            {
                "path": source_predecessor_evidence["deployment_receipt"],
                "sha256": source_predecessor_evidence[
                    "deployment_receipt_sha256"
                ],
            },
            "v2r10 source predecessor deployment receipt",
            source_predecessor_deployment_receipt_path,
        )
        if (
            source_predecessor_evidence["experiment1_recovery_receipt"]
            != str(expected_recovery_path)
            or source_predecessor_evidence[
                "experiment1_recovery_receipt_sha256"
            ] != recovery_file["sha256"]
            or source_predecessor_evidence[
                "experiment1_recovery_receipt_bytes"
            ] != len(recovery_file["bytes"])
            or source_predecessor_evidence[
                "experiment1_recovery_artifact_count"
            ] != recovery_payload["artifact_count"]
            or source_predecessor_evidence[
                "experiment1_recovery_row_count"
            ] != recovery_payload["inference_row_count"]
        ):
            raise StudyGateError(
                "The v2r10 source predecessor recovery binding changed"
            )
        source_predecessor_manifest_payload = _decode_json_object_bytes(
            source_predecessor_manifest["bytes"],
            label="v2r10 source predecessor release manifest",
        )
        source_predecessor_inventory = source_predecessor_manifest_payload.get(
            "payload_inventory"
        )
        source_predecessor_inventory_by_path = {
            str(record.get("path", "")): record
            for record in source_predecessor_inventory
            if isinstance(record, dict)
        } if isinstance(source_predecessor_inventory, list) else {}
        source_predecessor_builder_record = (
            source_predecessor_inventory_by_path.get(
                "build_visdrone_paper_notebook.py"
            )
        )
        source_predecessor_notebook_record = (
            source_predecessor_inventory_by_path.get(
                "SemUAV_VLM_VisDrone_Paper_Study.ipynb"
            )
        )
        if (
            source_predecessor_manifest_payload.get("schema_version")
            != "semuav.visdrone-paper-release.v1"
            or source_predecessor_manifest_payload.get("release_id")
            != source_predecessor_release_id
            or source_predecessor_manifest_payload.get("run_name")
            != source_predecessor_run_name
            or source_predecessor_manifest_payload.get("archive_name")
            != source_predecessor_archive_path.name
            or source_predecessor_manifest_payload.get("payload_file_count")
            != source_predecessor_evidence["payload_file_count"]
            or source_predecessor_manifest_payload.get("archive_member_count")
            != source_predecessor_evidence["archive_member_count"]
            or source_predecessor_manifest_payload.get(
                "payload_total_uncompressed_bytes"
            ) != source_predecessor_evidence[
                "payload_total_uncompressed_bytes"
            ]
            or source_predecessor_manifest_payload.get(
                "empirical_results_included"
            ) is not False
            or source_predecessor_manifest_payload.get(
                "dataset_material_included"
            ) is not False
            or not isinstance(source_predecessor_inventory, list)
            or len(source_predecessor_inventory)
            != source_predecessor_evidence["payload_file_count"]
            or len(source_predecessor_inventory_by_path)
            != len(source_predecessor_inventory)
            or not isinstance(source_predecessor_builder_record, dict)
            or source_predecessor_builder_record.get("sha256")
            != source_predecessor_evidence["builder_sha256"]
            or not isinstance(source_predecessor_notebook_record, dict)
            or source_predecessor_notebook_record.get("sha256")
            != source_predecessor_evidence["notebook_sha256"]
        ):
            raise StudyGateError(
                "The pinned v2r10 source predecessor manifest changed"
            )
        source_predecessor_deployment = _decode_json_object_bytes(
            source_predecessor_deployment_receipt["bytes"],
            label="v2r10 source predecessor deployment receipt",
        )
        require_mapping(source_predecessor_deployment, {
            "schema_version", "status", "release_id", "run_name",
            "release_root", "scientific_validation_authorized",
            "scientific_experiments_run", "runtime_smoke_run",
            "untouched_test_run", "validation_or_test_images_or_labels_read",
            "experiment1_recovery",
        }, "v2r10 source predecessor deployment receipt")
        if (
            source_predecessor_deployment["schema_version"]
            != "semuav.visdrone-v2r10-source-deployment.v1"
            or source_predecessor_deployment["status"]
            != "READY_FOR_DEVELOPMENT_SMOKE_ONLY"
            or source_predecessor_deployment["release_id"]
            != source_predecessor_release_id
            or source_predecessor_deployment["run_name"]
            != source_predecessor_run_name
            or Path(str(source_predecessor_deployment["release_root"])).resolve()
            != source_predecessor_release_root.resolve()
            or source_predecessor_deployment[
                "scientific_validation_authorized"
            ] is not False
            or source_predecessor_deployment["scientific_experiments_run"]
            is not False
            or source_predecessor_deployment["runtime_smoke_run"] is not False
            or source_predecessor_deployment["untouched_test_run"] is not False
            or source_predecessor_deployment[
                "validation_or_test_images_or_labels_read"
            ] is not False
            or source_predecessor_deployment["experiment1_recovery"]
            != recovery_record
        ):
            raise StudyGateError(
                "The pinned v2r10 source predecessor deployment receipt changed"
            )
        # The v2r10 layer must preserve the complete, canonical v2r9 stop.
        # Equality against an exact object prevents two mutually consistent
        # but jointly tampered predecessor files from passing this gate.
        expected_inherited_v2r9_boundary = {
            "correction_record": (
                "PRE_VALIDATION_NATIVE_RECEIVER_RECOVERY_20260921_V2R10.md"
            ),
            "correction_record_sha256": (
                "84510ede90584161c6254e530b68977b070097837197d31e64b50a6925380bec"
            ),
            "evidence_model": (
                "source_authenticated_defect_separate_from_operator_"
                "observed_execution"
            ),
            "operator_observed_execution": {
                "detector_prerequisite_completed": True,
                "detector_residency_receipt_written": True,
                "detector_stage_cache_written": True,
                "eligible_validation_sequences": 7,
                "evidence_level": (
                    "operator_observed_terminal_output_not_persisted_in_"
                    "v2r9_deployment_receipt"
                ),
                "exact_eligible_capacity_recomputed": True,
                "experiment2_frontier_completed": False,
                "experiments_3_to_7_started": False,
                "failure_exception_type": "StudyGateError",
                "failure_message": (
                    "Decoded semantic packet has unregistered action for "
                    "visdrone:det_train:0000030_00754_d_0000036:profile:"
                    "general_mobility"
                ),
                "failure_sample_id": (
                    "visdrone:det_train:0000030_00754_d_0000036:profile:"
                    "general_mobility"
                ),
                "failure_stage": (
                    "first_selected_semantic_native_receiver_l0_heartbeat"
                ),
                "failure_variant": "l0_cbor",
                "feature_cache_built": False,
                "final_experiment1_artifacts_persisted": False,
                "full_development_bundle_verified": True,
                "general_experiment1_routes_complete": 4,
                "model_selection_computed_in_memory": True,
                "model_selection_persisted": False,
                "native_receiver_cache_built": False,
                "native_receiver_cache_rows_persisted": 0,
                "native_receiver_cache_started": True,
                "protocol_frozen": False,
                "raw_experiment1_artifact_count": 132,
                "raw_experiment1_inference_row_count": 4152,
                "raw_experiment1_shard_artifacts_complete": True,
                "report_or_figures_generated": False,
                "runtime_smoke_passed": True,
                "schema_version": (
                    "semuav.visdrone-v2r9-operator-observed-"
                    "validation-stop.v1"
                ),
                "scientific_environment_sealed": True,
                "semantic_packet_cache_computed_in_memory": True,
                "temporal_experiment1_routes_complete": 2,
                "trusted_code_cache_frozen": True,
                "untouched_test_accessed": False,
                "validation_experiments_1_2_entered": True,
                "validation_input_gate_completed": True,
                "validation_temporal_sample_created_or_verified": True,
                "validation_vlm_inference_executed": True,
            },
            "predecessor_release_id": predecessor_release_id,
            "predecessor_run_name": predecessor_run_name,
            "schema_version": (
                "semuav.visdrone-v2r9-operator-observed-terminal-boundary.v1"
            ),
            "source_authenticated_defect": {
                "affected_entrypoint": (
                    "RegisteredVisDroneNativeReceiver.__call__"
                ),
                "affected_module": "semuav.visdrone_receiver_runtime",
                "affected_path": (
                    "semuav.visdrone_receivers.build_native_receiver_cache -> "
                    "wire_decode_and_receiver -> RegisteredVisDroneNativeReceiver"
                ),
                "affected_variant": "l0_cbor",
                "authoritative_pixel_control_fields": [
                    "capture_time_us", "gps_lat", "gps_lon", "altitude_m",
                    "yaw_deg",
                ],
                "corrected_fusion_policy_version": (
                    "semuav.visdrone-native-fusion.v3"
                ),
                "correction": (
                    "strictly_validate_canonical_actionless_l0_heartbeat_then_"
                    "return_registered_NO_ACTION_and_bind_authoritative_"
                    "control_time_and_flight_telemetry_into_pixel_receiver_"
                    "requests"
                ),
                "defect_kind": (
                    "actionless_l0_heartbeat_rejected_by_generic_semantic_"
                    "action_path"
                ),
                "exception_message_template": (
                    "Decoded semantic packet has unregistered action for "
                    "{sample_id}"
                ),
                "exception_type": "StudyGateError",
                "latent_affected_entrypoint": (
                    "RegisteredVisDroneNativeReceiver._frame_action"
                ),
                "latent_defect_kind": (
                    "pixel_receiver_request_used_synthetic_time_and_partial_"
                    "telemetry"
                ),
                "predecessor_fusion_policy_version": (
                    "semuav.visdrone-native-fusion.v2"
                ),
                "registered_l0_receiver_action": "NO_ACTION",
                "regression_coverage": [
                    "canonical_l0_heartbeat_returns_NO_ACTION",
                    "l0_with_action_field_rejected",
                    "actionless_l1_and_l2_rejected",
                    "unknown_semantic_variant_rejected",
                    "variant_specific_fusion_receipt_bound",
                    "pixel_receiver_request_binds_authoritative_time_and_telemetry",
                    "l3_roi_receiver_preserves_authoritative_telemetry",
                    "registered_native_cache_l0_through_visual_and_video_end_to_end",
                ],
                "requirements_changed": False,
                "schema_version": (
                    "semuav.visdrone-v2r10-source-authenticated-native-"
                    "receiver-correction.v1"
                ),
                "scientific_contract_changed": False,
                "source_evidence_level": (
                    "authenticated_v2r9_and_v2r10_native_receiver_source"
                ),
            },
            "v2r10_derived_results_recomputation_required": True,
            "v2r10_downstream_results_recomputation_required": True,
            "v2r10_fresh_runtime_smoke_required": True,
            "v2r10_raw_experiment1_recovery_authorized": True,
            "v2r10_raw_experiment1_recovery_schema": (
                "semuav.visdrone-v2r9-exp1-recovery.v1"
            ),
            "v2r9_derived_scientific_outputs_reused": False,
        }
        inherited_v2r9_boundary = require_mapping(
            source_predecessor_manifest_payload.get(
                "predecessor_observed_terminal_boundary"
            ),
            set(expected_inherited_v2r9_boundary),
            "authenticated v2r10 manifest inherited v2r9 terminal boundary",
        )
        receipt_inherited_v2r9_boundary = require_mapping(
            source_predecessor_deployment.get(
                "predecessor_observed_terminal_boundary"
            ),
            set(expected_inherited_v2r9_boundary),
            "authenticated v2r10 receipt inherited v2r9 terminal boundary",
        )
        if (
            inherited_v2r9_boundary != expected_inherited_v2r9_boundary
            or receipt_inherited_v2r9_boundary
            != expected_inherited_v2r9_boundary
            or set(inherited_v2r9_boundary)
            != set(expected_inherited_v2r9_boundary)
            or set(receipt_inherited_v2r9_boundary)
            != set(expected_inherited_v2r9_boundary)
            or set(inherited_v2r9_boundary["operator_observed_execution"])
            != set(expected_inherited_v2r9_boundary[
                "operator_observed_execution"
            ])
            or set(inherited_v2r9_boundary["source_authenticated_defect"])
            != set(expected_inherited_v2r9_boundary[
                "source_authenticated_defect"
            ])
        ):
            raise StudyGateError(
                "The authenticated v2r9 native-receiver boundary changed"
            )

        deployment_source_compatibility = require_mapping(
            deployment["source_compatibility"], {
                "active_source_release_id", "candidate_source_release_id",
                "requirements_byte_identical", "candidate_requirements_changed",
                "scientific_contract_changed", "generation_policy_version",
                "generation_policy_sha256", "model_decision_contract",
                "measurement_schema_version", "runner_version",
                "model_max_new_tokens", "runtime_smoke_input_schema",
                "runtime_smoke_report_schema_version",
                "runtime_smoke_isolation_schema", "runtime_smoke_worker_input_schema",
                "runtime_smoke_worker_result_schema", "active_model_ids",
                "trusted_code_cache_receipt_schema", "excluded_model_ids",
                "fresh_runtime_smoke_required",
                "predecessor_smoke_rows_reused",
                "temporal_validation_sequences", "temporal_test_sequences",
                "temporal_observations_per_sequence",
                "temporal_source_frame_stride", "temporal_target_rate_hz",
                "temporal_feasibility_amendment_record",
                "native_receiver_correction_record",
            }, "v2r13 deployment source compatibility",
        )
        if (
            deployment_source_compatibility["active_source_release_id"] != release_id
            or deployment_source_compatibility["candidate_source_release_id"]
            != candidate_source_release_id
            or deployment_source_compatibility["requirements_byte_identical"] is not True
            or deployment_source_compatibility["candidate_requirements_changed"] is not False
            or deployment_source_compatibility["scientific_contract_changed"] is not False
            or deployment_source_compatibility["generation_policy_version"]
            != VISDRONE_GENERATION_POLICY_VERSION
            or deployment_source_compatibility["generation_policy_sha256"]
            != VISDRONE_GENERATION_POLICY_SHA256
            or deployment_source_compatibility["model_decision_contract"]
            != cfg.models["semantic_contract"]
            or deployment_source_compatibility["measurement_schema_version"]
            != "semuav.repeated-inference-measurement.v2"
            or deployment_source_compatibility["runner_version"]
            != "semuav.visdrone-paper-runner.v7"
            or deployment_source_compatibility["model_max_new_tokens"]
            != int(cfg.models["max_new_tokens"])
            or deployment_source_compatibility["runtime_smoke_input_schema"]
            != "semuav.model-runtime-smoke-input.v3"
            or deployment_source_compatibility["runtime_smoke_report_schema_version"]
            != RUNTIME_SMOKE_REPORT_SCHEMA_VERSION
            or deployment_source_compatibility["runtime_smoke_isolation_schema"]
            != RUNTIME_SMOKE_ISOLATION_SCHEMA
            or deployment_source_compatibility["runtime_smoke_worker_input_schema"]
            != "semuav.runtime-smoke-worker-input.v2"
            or deployment_source_compatibility["runtime_smoke_worker_result_schema"]
            != "semuav.runtime-smoke-worker-result.v2"
            or deployment_source_compatibility["trusted_code_cache_receipt_schema"]
            != "semuav.visdrone-trusted-code-cache.v2"
            or deployment_source_compatibility["active_model_ids"]
            != list(REGISTERED_RUNTIME_SMOKE_MODEL_IDS)
            or deployment_source_compatibility["excluded_model_ids"] != ["moondream2"]
            or deployment_source_compatibility["fresh_runtime_smoke_required"] is not True
            or deployment_source_compatibility["predecessor_smoke_rows_reused"] is not False
            or deployment_source_compatibility["temporal_validation_sequences"] != 7
            or deployment_source_compatibility["temporal_test_sequences"] != 15
            or deployment_source_compatibility["temporal_observations_per_sequence"] != 64
            or deployment_source_compatibility["temporal_source_frame_stride"] != 1
            or deployment_source_compatibility["temporal_target_rate_hz"] != 5
            or deployment_source_compatibility[
                "temporal_feasibility_amendment_record"
            ] != TEMPORAL_FEASIBILITY_AMENDMENT_RECORD.name
            or deployment_source_compatibility[
                "native_receiver_correction_record"
            ] != RUNTIME_CORRECTION_RECORD.name
        ):
            raise StudyGateError("The v2r13 deployment source contract changed")

        expected_deployment_temporal_amendment = {
            **temporal_feasibility_amendment,
            "fresh_runtime_smoke_required": True,
            "predecessor_smoke_rows_reused": False,
        }
        deployment_temporal_amendment = require_mapping(
            deployment["temporal_feasibility_amendment"],
            set(expected_deployment_temporal_amendment),
            "inherited v2r7 deployment temporal-feasibility amendment",
        )
        if deployment_temporal_amendment != expected_deployment_temporal_amendment:
            raise StudyGateError(
                "The inherited v2r7 deployment temporal-feasibility amendment changed"
            )

        def authenticated_receiver_source_facts(path, label):
            receiver_bytes, receiver_sha256 = read_and_hash(path, label)
            try:
                receiver_tree = ast.parse(receiver_bytes.decode("utf-8"))
            except (UnicodeError, SyntaxError) as exc:
                raise StudyGateError(f"{label} is not valid UTF-8 Python") from exc

            fusion_policy_versions = [
                node.value.value
                for node in receiver_tree.body
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "NATIVE_RECEIVER_FUSION_POLICY_VERSION"
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ]
            functions = {
                node.name: node
                for node in receiver_tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            classes = {
                node.name: node
                for node in receiver_tree.body
                if isinstance(node, ast.ClassDef)
            }
            fusion_receipt = functions.get("native_receiver_fusion_receipt")
            registered_context = functions.get("registered_receiver_context")
            receiver_class = classes.get("RegisteredVisDroneNativeReceiver")
            receiver_call = None
            frame_action = None
            l0_validator = None
            if receiver_class is not None:
                receiver_call = next(
                    (
                        node for node in receiver_class.body
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name == "__call__"
                    ),
                    None,
                )
                frame_action = next(
                    (
                        node for node in receiver_class.body
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name == "_frame_action"
                    ),
                    None,
                )
                l0_validator = next(
                    (
                        node for node in receiver_class.body
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name == "_validate_l0_heartbeat"
                    ),
                    None,
                )
            if (
                fusion_receipt is None
                or registered_context is None
                or receiver_call is None
                or frame_action is None
            ):
                raise StudyGateError(f"{label} lacks the native receiver contract")

            call_literals = {
                node.value
                for node in ast.walk(receiver_call)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
            receipt_literals = {
                node.value
                for node in ast.walk(fusion_receipt)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
            semantic_media_branch = any(
                isinstance(node, ast.Compare)
                and isinstance(node.left, ast.Attribute)
                and node.left.attr == "media_kind"
                and any(
                    isinstance(value, ast.Constant) and value.value == "semantic"
                    for value in node.comparators
                )
                for node in ast.walk(receiver_call)
            )
            l0_returns_no_action = any(
                isinstance(node, ast.If)
                and any(
                    isinstance(value, ast.Constant) and value.value == "l0_cbor"
                    for value in ast.walk(node.test)
                )
                and any(
                    isinstance(value, ast.Return)
                    and isinstance(value.value, ast.Constant)
                    and value.value.value == "NO_ACTION"
                    for value in ast.walk(node)
                )
                for node in ast.walk(receiver_call)
            )
            context_constructor_calls = [
                node
                for node in ast.walk(registered_context)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "ReceiverControlContext"
            ]
            registered_control_fields = sorted({
                keyword.arg
                for call in context_constructor_calls
                for keyword in call.keywords
                if keyword.arg is not None
            })
            inference_requests = [
                node
                for node in ast.walk(frame_action)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "InferenceRequest"
            ]
            pixel_request_binds_capture_time = any(
                len(request.args) >= 5
                and isinstance(request.args[3], ast.Attribute)
                and isinstance(request.args[3].value, ast.Name)
                and request.args[3].value.id == "control_context"
                and request.args[3].attr == "capture_time_us"
                for request in inference_requests
            )
            control_telemetry_assignments = {
                target.id
                for node in ast.walk(frame_action)
                if isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "control_context"
                and node.value.attr == "telemetry"
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            pixel_request_binds_telemetry = any(
                len(request.args) >= 5
                and isinstance(request.args[4], ast.Name)
                and request.args[4].id in control_telemetry_assignments
                for request in inference_requests
            )
            l0_validator_binds_control_context = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_validate_l0_heartbeat"
                and any(
                    keyword.arg == "control_context"
                    and isinstance(keyword.value, ast.Attribute)
                    and keyword.value.attr == "control_context"
                    for keyword in node.keywords
                )
                for node in ast.walk(receiver_call)
            )

            return {
                "sha256": receiver_sha256,
                "fusion_policy_versions": fusion_policy_versions,
                "registered_receiver_present": receiver_class is not None,
                "receiver_call_present": receiver_call is not None,
                "semantic_media_branch_present": semantic_media_branch,
                "semantic_variant_literals": sorted(
                    call_literals & {"l0_cbor", "l1_cbor", "l2_cbor"}
                ),
                "l0_returns_registered_no_action": l0_returns_no_action,
                "l0_validator_present": l0_validator is not None,
                "l0_validator_binds_control_context": (
                    l0_validator_binds_control_context
                ),
                "registered_control_context_fields": registered_control_fields,
                "pixel_request_binds_control_capture_time": (
                    pixel_request_binds_capture_time
                ),
                "pixel_request_binds_control_telemetry": (
                    pixel_request_binds_telemetry
                ),
                "fusion_receipt_has_variant_policy": (
                    "action_source_by_semantic_variant" in receipt_literals
                ),
                "fusion_receipt_l0_policy": (
                    "validated_heartbeat_then_registered_NO_ACTION"
                    if "validated_heartbeat_then_registered_NO_ACTION"
                    in receipt_literals else None
                ),
            }

        predecessor_receiver_facts = authenticated_receiver_source_facts(
            predecessor_release_root / "semuav" / "visdrone_receiver_runtime.py",
            "authenticated v2r9 native-receiver source",
        )
        corrected_receiver_facts = authenticated_receiver_source_facts(
            PROJECT_ROOT / "semuav" / "visdrone_receiver_runtime.py",
            "authenticated v2r13 native-receiver source",
        )
        if predecessor_receiver_facts != {
            "sha256": (
                "28ab74b2856bf52103c6a3d9191bfc51c837075d5025ccf0d2222590a759f32e"
            ),
            "fusion_policy_versions": ["semuav.visdrone-native-fusion.v2"],
            "registered_receiver_present": True,
            "receiver_call_present": True,
            "semantic_media_branch_present": False,
            "semantic_variant_literals": [],
            "l0_returns_registered_no_action": False,
            "l0_validator_present": False,
            "l0_validator_binds_control_context": False,
            "registered_control_context_fields": [
                "mission_profile", "registered_roi_xyxy",
            ],
            "pixel_request_binds_control_capture_time": False,
            "pixel_request_binds_control_telemetry": False,
            "fusion_receipt_has_variant_policy": False,
            "fusion_receipt_l0_policy": None,
        }:
            raise StudyGateError(
                "The authenticated v2r9 native-receiver defect facts changed"
            )
        if corrected_receiver_facts != {
            "sha256": corrected_receiver_facts["sha256"],
            "fusion_policy_versions": ["semuav.visdrone-native-fusion.v3"],
            "registered_receiver_present": True,
            "receiver_call_present": True,
            "semantic_media_branch_present": True,
            "semantic_variant_literals": ["l0_cbor", "l1_cbor", "l2_cbor"],
            "l0_returns_registered_no_action": True,
            "l0_validator_present": True,
            "l0_validator_binds_control_context": True,
            "registered_control_context_fields": [
                "altitude_m", "capture_time_us", "gps_lat", "gps_lon",
                "mission_profile", "registered_roi_xyxy", "yaw_deg",
            ],
            "pixel_request_binds_control_capture_time": True,
            "pixel_request_binds_control_telemetry": True,
            "fusion_receipt_has_variant_policy": True,
            "fusion_receipt_l0_policy": (
                "validated_heartbeat_then_registered_NO_ACTION"
            ),
        }:
            raise StudyGateError(
                "The authenticated v2r13 native-receiver correction facts changed"
            )

        # The authenticated v2r12 predecessor carries the v2r11 pre-smoke
        # stop.  The distinct v2r10 acceptance-stop and v2r9 native-receiver
        # boundaries remain authenticated at their own predecessor layers.
        expected_source_authenticated_defect = {
            "schema_version": (
                "semuav.visdrone-v2r12-predecessor-boundary-routing-"
                "correction.v1"
            ),
            "defect_kind": (
                "deployment_receipt_immediate_predecessor_boundary_checked_"
                "as_v2r9_boundary"
            ),
            "affected_entrypoint": "notebook _verify_candidate_ready_receipt",
            "actual_boundary_schema": (
                "semuav.visdrone-v2r10-operator-observed-terminal-boundary.v1"
            ),
            "incorrect_expected_boundary_schema": (
                "semuav.visdrone-v2r9-operator-observed-terminal-boundary.v1"
            ),
            "correction": (
                "verify_each_terminal_boundary_at_its_authenticated_"
                "provenance_layer"
            ),
            "v2r11_receipt_mutated": False,
            "scientific_contract_changed": False,
            "requirements_changed": False,
        }
        expected_operator_observed_execution = {
            "source_deployment_ready": True,
            "deployment_receipt_sha256": (
                "befdce3067b571933dd94d35758c2c29385a06a8bb02ea93ae681117dcc17f3e"
            ),
            "prepare_phase_entered": True,
            "candidate_ready_verification_entered": True,
            "candidate_ready_verification_passed": False,
            "failure_stage": "prepare_candidate_ready_receipt_verification",
            "failure_exception_type": "StudyGateError",
            "failure_kind": (
                "immediate_predecessor_boundary_validated_against_older_"
                "v2r9_schema"
            ),
            "runtime_smoke_entered": False,
            "trusted_code_cache_frozen": False,
            "scientific_environment_sealed": False,
            "validation_software_acceptance_entered": False,
            "validation_experiments_1_to_7_started": False,
            "derived_results_generated": False,
            "report_or_figures_generated": False,
            "protocol_frozen": False,
            "untouched_test_accessed": False,
            "v2r9_experiment1_recovery_receipt_preserved": True,
            "v2r9_experiment1_recovery_artifact_count": 132,
            "v2r9_experiment1_recovery_row_count": 4152,
        }
        expected_observed_terminal_boundary = {
            "schema_version": (
                "semuav.visdrone-v2r11-operator-observed-terminal-boundary.v1"
            ),
            "evidence_model": (
                "authenticated_source_deployment_and_operator_observed_"
                "pre_smoke_stop"
            ),
            "correction_record": (
                PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_RECORD.name
            ),
            "correction_record_sha256": (
                predecessor_boundary_routing_correction_sha256
            ),
            "predecessor_release_id": (
                "release_20260921_visdrone_paper_v2r11"
            ),
            "predecessor_run_name": "semuav_visdrone_v2r11",
            "source_authenticated_defect": expected_source_authenticated_defect,
            "operator_observed_execution": expected_operator_observed_execution,
            "inherited_v2r10_terminal_boundary_preserved": True,
            "inherited_v2r9_terminal_boundary_preserved": True,
            "v2r12_fresh_runtime_smoke_required": True,
            "v2r12_raw_experiment1_recovery_authorized": True,
            "v2r12_raw_experiment1_recovery_schema": (
                "semuav.visdrone-v2r9-exp1-recovery.v1"
            ),
            "v2r12_derived_results_recomputation_required": True,
            "v2r12_downstream_results_recomputation_required": True,
            "v2r11_derived_scientific_outputs_reused": False,
        }
        inherited_v2r11_terminal_boundary = require_mapping(
            sealed_predecessor_deployment["predecessor_observed_terminal_boundary"],
            set(expected_observed_terminal_boundary),
            "operator-observed v2r11 terminal boundary",
        )
        if inherited_v2r11_terminal_boundary != expected_observed_terminal_boundary:
            raise StudyGateError(
                "The operator-observed v2r11 terminal boundary changed"
            )
        if set(inherited_v2r11_terminal_boundary) != set(expected_observed_terminal_boundary):
            raise StudyGateError(
                "The operator-observed v2r11 boundary contains unbound extra fields"
            )
        if (
            set(inherited_v2r11_terminal_boundary["source_authenticated_defect"])
            != set(expected_source_authenticated_defect)
            or set(inherited_v2r11_terminal_boundary["operator_observed_execution"])
            != set(expected_operator_observed_execution)
        ):
            raise StudyGateError(
                "The v2r11 boundary nested evidence contains unbound extra fields"
            )

        if sealed_predecessor_manifest_payload.get("predecessor_observed_terminal_boundary") != expected_observed_terminal_boundary:
            raise StudyGateError("The inherited v2r11 manifest terminal boundary changed")

        expected_inherited_v2r10_source_defect = {
            "schema_version": (
                "semuav.visdrone-v2r11-acceptance-recovery-isolation.v1"
            ),
            "defect_kind": (
                "recovery_capability_inherited_by_software_acceptance_child"
            ),
            "affected_entrypoint": (
                "notebook software-acceptance subprocess environment"
            ),
            "removed_from_child_only": [
                "SEMUAV_ENABLE_V2R9_EXP1_RECOVERY",
                "SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT",
                "SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT_SHA256",
            ],
            "parent_environment_preserved": True,
            "recovery_importer_full_run_guard_preserved": True,
            "scientific_contract_changed": False,
            "requirements_changed": False,
            "model_or_metric_outcomes_used": False,
        }
        expected_inherited_v2r10_execution = {
            "source_deployment_ready": True,
            "runtime_smoke_passed": True,
            "runtime_smoke_pass_count": 3,
            "scientific_environment_sealed": True,
            "v2r9_experiment1_recovery_receipt_frozen": True,
            "v2r9_experiment1_recovery_artifact_count": 132,
            "v2r9_experiment1_recovery_row_count": 4152,
            "validation_parent_recovery_enabled": True,
            "software_acceptance_entered": True,
            "software_acceptance_passed": False,
            "failure_stage": "validation_software_acceptance_subprocess",
            "failure_test": (
                "test_detector_revision_and_vlm_cache_bind_checkpoint_residency"
            ),
            "failure_exception_type": "StudyGateError",
            "failure_kind": (
                "validation_only_recovery_environment_leaked_into_generic_"
                "test_fixture"
            ),
            "validation_experiments_1_to_7_started": False,
            "derived_results_generated": False,
            "report_or_figures_generated": False,
            "protocol_frozen": False,
            "untouched_test_accessed": False,
        }
        expected_inherited_v2r10_boundary = {
            "schema_version": (
                "semuav.visdrone-v2r10-operator-observed-terminal-boundary.v1"
            ),
            "predecessor_release_id": (
                "release_20260921_visdrone_paper_v2r10"
            ),
            "predecessor_run_name": "semuav_visdrone_v2r10",
            "correction_record": (
                "PRE_VALIDATION_ACCEPTANCE_RECOVERY_ISOLATION_20260921_V2R11.md"
            ),
            "correction_record_sha256": (
                "ff0f0329cbb06e2f754c243578bc80c538a208a2f7df72caf0264e6d6fc505ef"
            ),
            "evidence_model": (
                "authenticated_source_and_operator_observed_acceptance_stop"
            ),
            "operator_observed_execution": expected_inherited_v2r10_execution,
            "source_authenticated_defect": expected_inherited_v2r10_source_defect,
            "inherited_v2r9_terminal_boundary_preserved": True,
            "v2r11_fresh_runtime_smoke_required": True,
            "v2r11_raw_experiment1_recovery_authorized": True,
            "v2r11_derived_results_recomputation_required": True,
            "v2r11_downstream_results_recomputation_required": True,
        }
        manifest_inherited_v2r10_boundary = require_mapping(
            immediate_predecessor_manifest_payload.get(
                "predecessor_observed_terminal_boundary"
            ),
            set(expected_inherited_v2r10_boundary),
            "v2r11 manifest inherited v2r10 terminal boundary",
        )
        receipt_inherited_v2r10_boundary = require_mapping(
            immediate_predecessor_deployment.get(
                "predecessor_observed_terminal_boundary"
            ),
            set(expected_inherited_v2r10_boundary),
            "v2r11 receipt inherited v2r10 terminal boundary",
        )
        if (
            manifest_inherited_v2r10_boundary
            != expected_inherited_v2r10_boundary
            or receipt_inherited_v2r10_boundary
            != expected_inherited_v2r10_boundary
            or set(manifest_inherited_v2r10_boundary)
            != set(expected_inherited_v2r10_boundary)
            or set(receipt_inherited_v2r10_boundary)
            != set(expected_inherited_v2r10_boundary)
            or set(expected_inherited_v2r10_source_defect)
            != set(manifest_inherited_v2r10_boundary["source_authenticated_defect"])
            or set(expected_inherited_v2r10_execution)
            != set(manifest_inherited_v2r10_boundary["operator_observed_execution"])
        ):
            raise StudyGateError(
                "The authenticated v2r10 acceptance-stop boundary changed"
            )

        predecessor_evidence_required = {
            "release_path", "release_id", "run_name", "archive_path",
            "archive_sha256", "release_manifest_sha256",
            "checksum_inventory_sha256", "deployer_path", "deployer_sha256",
            "deployment_receipt", "deployment_receipt_sha256",
            "reused_artifacts_receipt_path",
            "reused_artifacts_receipt_sha256",
            "inference_environment_fingerprint_path",
            "inference_environment_fingerprint_sha256",
            "inference_pip_freeze_path", "inference_pip_freeze_sha256",
            "runtime_smoke_input_path",
            "runtime_smoke_input_sha256", "runtime_smoke_report_path",
            "runtime_smoke_report_sha256", "runtime_smoke_schema_version",
            "runtime_smoke_isolation_schema", "runtime_smoke_all_passed",
            "runtime_smoke_pass_count", "runtime_smoke_failure_count",
            "model_status", "active_model_lock_path",
            "active_model_lock_sha256", "active_model_set_sha256",
            "model_lock_derivation_path",
            "model_lock_derivation_sha256",
            "trusted_code_cache_receipt_path",
            "trusted_code_cache_receipt_sha256",
            "scientific_environment_seal_path",
            "scientific_environment_seal_sha256",
            "unchanged_after_deployment_actions",
        }
        predecessor_evidence = require_mapping(
            deployment["authenticated_predecessor_v2r9"],
            predecessor_evidence_required,
            "authenticated hash-backed v2r9 predecessor evidence",
        )
        if set(predecessor_evidence) != predecessor_evidence_required:
            raise StudyGateError(
                "Authenticated v2r9 evidence contains unbound extra fields"
            )
        predecessor_expected_status = {
            "qwen3_vl_2b": "PASS",
            "smolvlm2_2b": "PASS",
            "qwen3_vl_8b_upper_bound": "PASS",
        }
        predecessor_exact_hashes = {
            "archive_sha256": (
                "73b94ea64ac21fc61c7a3e0cec1038397c6dfc327c58d962ac96dd3052e48cf7"
            ),
            "release_manifest_sha256": (
                "c63bddc8676d05d5ee9a3f196daa407681177fb980385c4750fb60536baf7efd"
            ),
            "checksum_inventory_sha256": (
                "2df86df69fa36ae975f4f380b851dcd6fb28a59ff7868ff45f73f6dc806bd7dc"
            ),
            "deployer_sha256": (
                "3f6b863284313418bf21097bfa0e548f19b22fef11db1175560e09f9b3b48f28"
            ),
            "deployment_receipt_sha256": (
                "b5962f818a3ffa616e24bbf3ce188c150eeed0c0401ca101e9c483e2b13aef27"
            ),
            "reused_artifacts_receipt_sha256": (
                "ab90dc31f7bf9ae7957316f3e36058991272d78a8db1b729d68346bca46dab78"
            ),
            "inference_environment_fingerprint_sha256": (
                "424c57628cfbab7b15ce5546af997e6773db75ab8ea72274d0b76ccd827d5c10"
            ),
            "inference_pip_freeze_sha256": (
                "5b6aacbcd2b891c1bf968511d17b3385aff2233e99d0bae440f81bc7f977d5d2"
            ),
            "runtime_smoke_input_sha256": (
                "a3c9d33c87d2b620acd41be55d5b5494cdb75bcbe6a0abb82a5f625cce4674e1"
            ),
            "runtime_smoke_report_sha256": (
                "7642b89cd36a87faafc62c44e88baec5bb45d3ea8d7c3438e2ad0293aab585b5"
            ),
            "active_model_lock_sha256": (
                "340dd59ca55a9be491ac806b50f9778a12d2b0b4a447485627aaf422dda05353"
            ),
            "active_model_set_sha256": (
                "6f83118bb62cd68c98e9ee2afd947022c04088b9329a0eeef832eba17c2fb08c"
            ),
            "model_lock_derivation_sha256": (
                "947c5e1ad828eed9beaf6538f227fd871b406f3fab11bf6ce7202dabf23749b0"
            ),
            "trusted_code_cache_receipt_sha256": (
                "ad45cd62384697243de0391bc929b05356ba4252d80fd93854f5d1a90d4a55ed"
            ),
            "scientific_environment_seal_sha256": (
                "01f18450cac8bc5d1edf4dd45cf7302e0b2903d53609e0850cddf723bbf69ec3"
            ),
        }
        if (
            Path(str(predecessor_evidence["release_path"])).resolve()
            != predecessor_release_root.resolve()
            or predecessor_evidence["release_id"] != predecessor_release_id
            or predecessor_evidence["run_name"] != predecessor_run_name
            or any(
                require_sha256(
                    predecessor_evidence[name], f"v2r9 predecessor {name}"
                ) != digest
                for name, digest in predecessor_exact_hashes.items()
            )
            or predecessor_evidence["runtime_smoke_schema_version"] != 4
            or predecessor_evidence["runtime_smoke_isolation_schema"]
            != "semuav.runtime-smoke-subprocess.v2"
            or predecessor_evidence["runtime_smoke_all_passed"] is not True
            or predecessor_evidence["runtime_smoke_pass_count"] != 3
            or predecessor_evidence["runtime_smoke_failure_count"] != 0
            or predecessor_evidence["model_status"] != predecessor_expected_status
            or predecessor_evidence["unchanged_after_deployment_actions"] is not True
        ):
            raise StudyGateError(
                "The authenticated hash-backed v2r9 predecessor evidence changed"
            )
        predecessor_archive = verify_file_record(
            {
                "path": predecessor_evidence["archive_path"],
                "sha256": predecessor_evidence["archive_sha256"],
            },
            "v2r9 predecessor archive", predecessor_archive_path,
        )
        predecessor_deployer = verify_file_record(
            {
                "path": predecessor_evidence["deployer_path"],
                "sha256": predecessor_evidence["deployer_sha256"],
            },
            "v2r9 predecessor deployer", predecessor_deployer_path,
        )
        predecessor_deployment_receipt = verify_file_record(
            {
                "path": predecessor_evidence["deployment_receipt"],
                "sha256": predecessor_evidence["deployment_receipt_sha256"],
            },
            "v2r9 predecessor deployment receipt",
            predecessor_deployment_receipt_path,
        )
        predecessor_deployment = _decode_json_object_bytes(
            predecessor_deployment_receipt["bytes"],
            label="v2r9 predecessor deployment receipt",
        )
        require_mapping(predecessor_deployment, {
            "schema_version", "status", "release_id", "run_name",
            "release_root", "scientific_validation_authorized",
            "scientific_experiments_run", "runtime_smoke_run",
            "validation_or_test_images_or_labels_read", "gpu_accessed",
            "model_weights_loaded", "jupyter_launched", "source_compatibility",
        }, "v2r9 predecessor deployment receipt")
        if (
            predecessor_deployment["schema_version"]
            != "semuav.visdrone-v2r9-source-deployment.v1"
            or predecessor_deployment["status"]
            != "READY_FOR_DEVELOPMENT_SMOKE_ONLY"
            or predecessor_deployment["release_id"] != predecessor_release_id
            or predecessor_deployment["run_name"] != predecessor_run_name
            or Path(str(predecessor_deployment["release_root"])).resolve()
            != predecessor_release_root.resolve()
            or any(
                predecessor_deployment[name] is not False
                for name in (
                    "scientific_validation_authorized",
                    "scientific_experiments_run", "runtime_smoke_run",
                    "validation_or_test_images_or_labels_read", "gpu_accessed",
                    "model_weights_loaded", "jupyter_launched",
                )
            )
        ):
            raise StudyGateError("The pinned v2r9 deployment receipt changed")
        predecessor_manifest = verify_file_record(
            {
                "path": predecessor_release_root / "RELEASE_MANIFEST.json",
                "sha256": predecessor_evidence["release_manifest_sha256"],
            },
            "v2r9 predecessor release manifest",
            predecessor_release_root / "RELEASE_MANIFEST.json",
        )
        predecessor_checksums = verify_file_record(
            {
                "path": predecessor_release_root / "RELEASE_MANIFEST_SHA256.txt",
                "sha256": predecessor_evidence["checksum_inventory_sha256"],
            },
            "v2r9 predecessor checksum inventory",
            predecessor_release_root / "RELEASE_MANIFEST_SHA256.txt",
        )
        predecessor_reused = verify_file_record(
            {
                "path": predecessor_evidence["reused_artifacts_receipt_path"],
                "sha256": predecessor_evidence["reused_artifacts_receipt_sha256"],
            },
            "v2r9 predecessor reused-artifact receipt",
            predecessor_frozen_root / "REUSED_V1_FROZEN_ARTIFACTS.json",
        )
        predecessor_fingerprint = verify_file_record(
            {
                "path": predecessor_evidence[
                    "inference_environment_fingerprint_path"
                ],
                "sha256": predecessor_evidence[
                    "inference_environment_fingerprint_sha256"
                ],
            },
            "v2r9 predecessor inference-environment fingerprint",
            predecessor_frozen_root / "INFERENCE_ENVIRONMENT_FINGERPRINT.json",
        )
        predecessor_freeze = verify_file_record(
            {
                "path": predecessor_evidence["inference_pip_freeze_path"],
                "sha256": predecessor_evidence["inference_pip_freeze_sha256"],
            },
            "v2r9 predecessor inference distribution inventory",
            predecessor_frozen_root / "INFERENCE_PIP_FREEZE.txt",
        )
        predecessor_smoke_input = verify_file_record(
            {
                "path": predecessor_evidence["runtime_smoke_input_path"],
                "sha256": predecessor_evidence["runtime_smoke_input_sha256"],
            },
            "v2r9 predecessor runtime-smoke input",
            predecessor_frozen_root / "MODEL_RUNTIME_SMOKE_INPUT.json",
        )
        predecessor_smoke_report = verify_file_record(
            {
                "path": predecessor_evidence["runtime_smoke_report_path"],
                "sha256": predecessor_evidence["runtime_smoke_report_sha256"],
            },
            "v2r9 predecessor passing runtime-smoke report",
            predecessor_frozen_root / "MODEL_RUNTIME_SMOKE.json",
        )
        predecessor_smoke = _decode_json_object_bytes(
            predecessor_smoke_report["bytes"],
            label="v2r9 predecessor passing runtime-smoke report",
        )
        predecessor_rows = predecessor_smoke.get("models")
        predecessor_isolation = predecessor_smoke.get("execution_isolation")
        predecessor_worker_artifacts = (
            predecessor_isolation.get("artifacts")
            if isinstance(predecessor_isolation, dict) else None
        )
        if (
            predecessor_smoke.get("schema_version") != 4
            or predecessor_smoke.get("all_passed") is not True
            or not isinstance(predecessor_rows, list)
            or any(not isinstance(row, dict) for row in predecessor_rows)
            or tuple(str(row.get("model_id", "")) for row in predecessor_rows)
            != REGISTERED_RUNTIME_SMOKE_MODEL_IDS
            or {
                str(row.get("model_id", "")): str(row.get("status", ""))
                for row in predecessor_rows
            } != predecessor_expected_status
            or not isinstance(predecessor_isolation, dict)
            or predecessor_isolation.get("schema_version")
            != "semuav.runtime-smoke-subprocess.v2"
            or not isinstance(predecessor_worker_artifacts, list)
            or len(predecessor_worker_artifacts) != 3
            or hash_object(predecessor_worker_artifacts)
            != predecessor_isolation.get("artifacts_sha256")
            or predecessor_smoke.get("model_lock_sha256")
            != predecessor_evidence["active_model_lock_sha256"]
            or predecessor_smoke.get("model_set_sha256")
            != predecessor_evidence["active_model_set_sha256"]
        ):
            raise StudyGateError(
                "The authenticated v2r9 passing runtime-smoke report changed"
            )
        require_runtime_smoke_pass(predecessor_smoke_report["path"])
        for worker_record in predecessor_worker_artifacts:
            if not isinstance(worker_record, dict):
                raise StudyGateError("A v2r9 worker-artifact record is malformed")
            for stream_name in ("input", "result", "stdout", "stderr"):
                stream_record = worker_record.get(stream_name)
                if stream_record is not None:
                    verify_file_record(
                        stream_record,
                        f"v2r9 {worker_record.get('model_id')} worker {stream_name}",
                    )

        predecessor_smoke_input_payload = _decode_json_object_bytes(
            predecessor_smoke_input["bytes"],
            label="v2r9 predecessor runtime-smoke input",
        )
        predecessor_registered_set = predecessor_smoke_input_payload.get(
            "registered_model_set"
        )
        if (
            predecessor_smoke_input_payload.get("schema_version")
            != "semuav.model-runtime-smoke-input.v3"
            or predecessor_smoke_input_payload.get("scientific_results_eligible")
            is not False
            or predecessor_smoke_input_payload.get("excluded_model_ids")
            != ["moondream2"]
            or predecessor_smoke_input_payload.get("model_lock_sha256")
            != predecessor_evidence["active_model_lock_sha256"]
            or not isinstance(predecessor_registered_set, dict)
            or predecessor_registered_set.get("model_set_sha256")
            != predecessor_evidence["active_model_set_sha256"]
            or tuple(
                str(item.get("model_id", ""))
                for item in predecessor_registered_set.get("models", [])
                if isinstance(item, dict)
            ) != REGISTERED_RUNTIME_SMOKE_MODEL_IDS
        ):
            raise StudyGateError("The pinned v2r9 runtime-smoke input changed")

        predecessor_active_lock = verify_file_record(
            {
                "path": predecessor_evidence["active_model_lock_path"],
                "sha256": predecessor_evidence["active_model_lock_sha256"],
            },
            "v2r9 predecessor active model lock",
            predecessor_frozen_root / "MODEL_LOCK.json",
        )
        predecessor_lock_payload = _decode_json_object_bytes(
            predecessor_active_lock["bytes"],
            label="v2r9 predecessor active model lock",
        )
        if (
            predecessor_lock_payload.get("model_set_sha256")
            != predecessor_evidence["active_model_set_sha256"]
            or tuple(
                str(item.get("model_id", ""))
                for item in predecessor_lock_payload.get("models", [])
                if isinstance(item, dict)
            ) != REGISTERED_RUNTIME_SMOKE_MODEL_IDS
        ):
            raise StudyGateError("The pinned v2r9 active model set changed")
        predecessor_lock_derivation = verify_file_record(
            {
                "path": predecessor_evidence["model_lock_derivation_path"],
                "sha256": predecessor_evidence["model_lock_derivation_sha256"],
            },
            "v2r9 predecessor model-lock derivation",
            predecessor_frozen_root / "MODEL_LOCK_DERIVATION.json",
        )
        predecessor_derivation_payload = _decode_json_object_bytes(
            predecessor_lock_derivation["bytes"],
            label="v2r9 predecessor model-lock derivation",
        )
        if (
            predecessor_derivation_payload.get("derived_model_lock_sha256")
            != predecessor_evidence["active_model_lock_sha256"]
            or predecessor_derivation_payload.get("derived_model_set_sha256")
            != predecessor_evidence["active_model_set_sha256"]
            or predecessor_derivation_payload.get("prior_runtime_smoke_rows_reused")
            is not False
            or predecessor_derivation_payload.get("validation_or_test_evidence_used")
            is not False
        ):
            raise StudyGateError("The pinned v2r9 model-lock derivation changed")

        predecessor_trusted = verify_file_record(
            {
                "path": predecessor_evidence["trusted_code_cache_receipt_path"],
                "sha256": predecessor_evidence[
                    "trusted_code_cache_receipt_sha256"
                ],
            },
            "v2r9 predecessor trusted-code cache receipt",
            predecessor_frozen_root / "TRUSTED_CODE_CACHE.json",
        )
        predecessor_trusted_payload = _decode_json_object_bytes(
            predecessor_trusted["bytes"],
            label="v2r9 predecessor trusted-code cache receipt",
        )
        if (
            predecessor_trusted_payload.get("schema_version")
            != "semuav.visdrone-trusted-code-cache.v2"
            or predecessor_trusted_payload.get("scope")
            != "no_active_trusted_remote_code"
            or predecessor_trusted_payload.get("model_lock_sha256")
            != predecessor_evidence["active_model_lock_sha256"]
            or predecessor_trusted_payload.get("model_lock_derivation_sha256")
            != predecessor_evidence["model_lock_derivation_sha256"]
            or predecessor_trusted_payload.get("registered_model_ids")
            != list(REGISTERED_RUNTIME_SMOKE_MODEL_IDS)
            or predecessor_trusted_payload.get("excluded_model_ids")
            != ["moondream2"]
            or predecessor_trusted_payload.get("module_roots") != []
            or predecessor_trusted_payload.get("files") != []
            or predecessor_trusted_payload.get("file_count") != 0
            or predecessor_trusted_payload.get("tree_sha256") != hash_object([])
        ):
            raise StudyGateError("The pinned v2r9 trusted-code cache changed")

        predecessor_seal = verify_file_record(
            {
                "path": predecessor_evidence["scientific_environment_seal_path"],
                "sha256": predecessor_evidence[
                    "scientific_environment_seal_sha256"
                ],
            },
            "v2r9 predecessor scientific-use seal",
            predecessor_frozen_root / "INFERENCE_ENVIRONMENT_SEAL.json",
        )
        predecessor_seal_payload = _decode_json_object_bytes(
            predecessor_seal["bytes"],
            label="v2r9 predecessor scientific-use seal",
        )
        predecessor_seal_smoke = predecessor_seal_payload.get("runtime_smoke")
        predecessor_seal_trusted = predecessor_seal_payload.get(
            "trusted_code_cache"
        )
        predecessor_seal_reused = predecessor_seal_payload.get(
            "reused_v1_frozen_artifacts_receipt"
        )
        if (
            predecessor_seal_payload.get("schema_version")
            != cfg.runtime["inference_environment_seal_schema"]
            or predecessor_seal_payload.get("status")
            != cfg.runtime["inference_environment_seal_status"]
            or predecessor_seal_payload.get("scientific_results_authorized")
            is not True
            or predecessor_seal_payload.get(
                "validation_or_test_data_read_before_seal"
            ) is not False
            or predecessor_seal_payload.get("registered_model_ids")
            != list(REGISTERED_RUNTIME_SMOKE_MODEL_IDS)
            or predecessor_seal_payload.get("excluded_model_ids")
            != ["moondream2"]
            or not isinstance(predecessor_seal_smoke, dict)
            or predecessor_seal_smoke.get("sha256")
            != predecessor_evidence["runtime_smoke_report_sha256"]
            or predecessor_seal_smoke.get("all_passed") is not True
            or not isinstance(predecessor_seal_trusted, dict)
            or predecessor_seal_trusted.get("sha256")
            != predecessor_evidence["trusted_code_cache_receipt_sha256"]
            or predecessor_seal_trusted.get("file_count") != 0
            or not isinstance(predecessor_seal_reused, dict)
            or predecessor_seal_reused.get("sha256")
            != predecessor_evidence["reused_artifacts_receipt_sha256"]
        ):
            raise StudyGateError("The pinned v2r9 scientific-use seal changed")

        live_deployer = verify_file_record(
            deployment["deployment_script"], "v2r13 deployment script",
            deployer_path,
        )
        live_builder = verify_file_record(
            deployment["builder"], "v2r13 candidate builder", builder_path,
        )

        deployed_candidate = require_mapping(
            deployment["candidate_environment"], {
                "path", "ready_receipt", "ready_receipt_sha256",
                "pip_freeze_sha256",
            }, "reused v2r3 candidate",
        )
        if (
            Path(str(deployed_candidate["path"])).resolve()
            != candidate_root.resolve()
            or Path(str(deployed_candidate["ready_receipt"])).resolve()
            != candidate_receipt_path.resolve()
        ):
            raise StudyGateError("The deployed v2r13 candidate paths changed")
        candidate_receipt_bytes, candidate_receipt_sha256 = read_and_hash(
            candidate_receipt_path, "v2r3 candidate ready receipt",
        )
        if candidate_receipt_sha256 != require_sha256(
            deployed_candidate["ready_receipt_sha256"],
            "Deployed candidate-receipt digest",
        ):
            raise StudyGateError("The deployed candidate receipt bytes changed")
        ready = _decode_json_object_bytes(
            candidate_receipt_bytes, label="v2r3 candidate ready receipt",
        )
        require_mapping(ready, {
            "schema_version", "status", "candidate_only",
            "scientific_results_authorized", "validation_or_test_data_read",
            "legacy_environment_invoked", "canonical_release_modified",
            "candidate_environment", "builder", "source_release",
            "predecessor_environment_equivalence", "requested_pins",
            "validated_runtime", "runtime_guards", "artifacts",
            "build_record_directory", "build_log", "reproducibility_boundary",
        }, "v2r3 candidate ready receipt")
        if (
            ready["schema_version"] != cfg.runtime["candidate_ready_receipt_schema"]
            or ready["status"] != cfg.runtime["candidate_ready_status"]
            or ready["candidate_only"] is not True
            or any(
                ready[name] is not False
                for name in (
                    "scientific_results_authorized", "validation_or_test_data_read",
                    "legacy_environment_invoked", "canonical_release_modified",
                )
            )
            or Path(str(ready["candidate_environment"])).resolve()
            != candidate_root.resolve()
            or not isinstance(ready["requested_pins"], dict)
            or not ready["requested_pins"]
            or not isinstance(ready["validated_runtime"], dict)
            or not ready["validated_runtime"]
        ):
            raise StudyGateError("The reused v2r3 candidate identity or safety state changed")

        candidate_guards = require_mapping(
            ready["runtime_guards"], {
                "YOLO_AUTOINSTALL", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                "PIP_NO_INDEX_during_validation",
                "CUDA_VISIBLE_DEVICES_during_build",
                "NVIDIA_VISIBLE_DEVICES_during_build",
                "inherited_loader_environment_cleared",
            }, "v2r3 candidate runtime guards",
        )
        if (
            candidate_guards["YOLO_AUTOINSTALL"] != "false"
            or candidate_guards["HF_HUB_OFFLINE"] != "1"
            or candidate_guards["TRANSFORMERS_OFFLINE"] != "1"
            or candidate_guards["PIP_NO_INDEX_during_validation"] != "1"
            or candidate_guards["CUDA_VISIBLE_DEVICES_during_build"] != ""
            or candidate_guards["NVIDIA_VISIBLE_DEVICES_during_build"] != "void"
            or candidate_guards["inherited_loader_environment_cleared"] is not True
        ):
            raise StudyGateError("The v2r3 candidate build guards changed")

        candidate_builder = require_mapping(
            ready["builder"], {"path", "sha256"}, "candidate builder binding",
        )
        if (
            Path(str(candidate_builder["path"])).resolve() != builder_path.resolve()
            or require_sha256(
                candidate_builder["sha256"], "Candidate builder digest",
            ) != live_builder["sha256"]
        ):
            raise StudyGateError("Candidate and deployment builder bindings differ")

        predecessor = require_mapping(
            ready["predecessor_environment_equivalence"], {
                "predecessor_candidate_environment", "pip_freeze_sha256",
                "equivalence_scope",
            }, "candidate predecessor equivalence",
        )
        if (
            Path(str(predecessor["predecessor_candidate_environment"])).resolve()
            != predecessor_candidate.resolve()
            or predecessor["equivalence_scope"]
            != "installed_distribution_inventory_only"
        ):
            raise StudyGateError("Candidate predecessor equivalence changed")

        manifest_bytes, manifest_sha256 = read_and_hash(
            manifest_path, "v2r13 release manifest",
        )
        checksums_bytes, checksums_sha256 = read_and_hash(
            checksums_path, "v2r13 checksum inventory",
        )
        requirements_bytes, requirements_sha256 = read_and_hash(
            requirements_path, "v2r13 requirements",
        )
        source = require_mapping(ready["source_release"], {
            "path", "release_id", "run_name", "manifest_sha256",
            "checksum_inventory_sha256", "requirements_sha256",
        }, "candidate source release")
        candidate_manifest_bytes, candidate_manifest_sha256 = read_and_hash(
            candidate_manifest_path, "v2r3 candidate-source release manifest",
        )
        candidate_checksums_bytes, candidate_checksums_sha256 = read_and_hash(
            candidate_checksums_path, "v2r3 candidate-source checksum inventory",
        )
        candidate_requirements_bytes, candidate_requirements_sha256 = read_and_hash(
            candidate_requirements_path, "v2r3 candidate-source requirements",
        )
        candidate_source_hashes = {
            "manifest_sha256": candidate_manifest_sha256,
            "checksum_inventory_sha256": candidate_checksums_sha256,
            "requirements_sha256": candidate_requirements_sha256,
        }
        candidate_manifest = _decode_json_object_bytes(
            candidate_manifest_bytes,
            label="v2r3 candidate-source release manifest",
        )
        if (
            Path(str(source["path"])).resolve() != candidate_source_root.resolve()
            or source["release_id"] != candidate_source_release_id
            or source["run_name"] != candidate_source_run_name
            or candidate_manifest.get("release_id") != candidate_source_release_id
            or candidate_manifest.get("run_name") != candidate_source_run_name
            or candidate_manifest.get("empirical_results_included") is not False
            or candidate_manifest.get("dataset_material_included") is not False
            or any(
                require_sha256(source[name], f"Candidate source {name}") != digest
                for name, digest in candidate_source_hashes.items()
            )
        ):
            raise StudyGateError("The candidate source-release binding changed")

        active_source_hashes = {
            "manifest_sha256": manifest_sha256,
            "checksum_inventory_sha256": checksums_sha256,
            "requirements_sha256": requirements_sha256,
        }
        if (
            requirements_bytes != candidate_requirements_bytes
            or requirements_sha256 != candidate_requirements_sha256
        ):
            raise StudyGateError(
                "The v2r13 source requirements differ from the v2r3 candidate source"
            )

        deployed_source = require_mapping(deployment["source_release"], {
            "archive_path", "archive_sha256", "manifest_sha256",
            "checksum_inventory_sha256", "requirements_sha256",
        }, "deployed v2r13 source")
        archive = verify_file_record(
            {
                "path": deployed_source["archive_path"],
                "sha256": deployed_source["archive_sha256"],
            },
            "v2r13 source archive", archive_path,
        )
        if any(
            require_sha256(deployed_source[name], f"Deployed source {name}")
            != digest
            for name, digest in active_source_hashes.items()
        ):
            raise StudyGateError("Active v2r13 and deployment source hashes differ")

        manifest = _decode_json_object_bytes(
            manifest_bytes, label="v2r13 release manifest",
        )
        require_mapping(manifest, {
            "schema_version", "release_id", "archive_name", "run_name",
            "empirical_results_included", "dataset_material_included",
            "payload_file_count", "payload_inventory", "validation",
            "methodology", "pre_validation_native_receiver_correction",
            "pre_validation_acceptance_recovery_isolation",
            "pre_smoke_predecessor_boundary_routing_correction",
            "authenticated_predecessor_v2r11",
        "authenticated_predecessor_v2r12",
            "authenticated_predecessor_v2r10",
        }, "v2r13 release manifest")
        inventory = manifest["payload_inventory"]
        manifest_methodology = manifest["methodology"]
        if (
            manifest["schema_version"] != "semuav.visdrone-paper-release.v1"
            or manifest["release_id"] != release_id
            or manifest["archive_name"] != archive_path.name
            or manifest["run_name"] != run_name
            or manifest["empirical_results_included"] is not False
            or manifest["dataset_material_included"] is not False
            or manifest["pre_validation_native_receiver_correction"]
            != RUNTIME_CORRECTION_RECORD.name
            or manifest["pre_validation_acceptance_recovery_isolation"]
            != ACCEPTANCE_RECOVERY_ISOLATION_RECORD.name
            or manifest[
                "pre_smoke_predecessor_boundary_routing_correction"
            ] != PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_RECORD.name
            or manifest["authenticated_predecessor_v2r12"] != sealed_predecessor_evidence
            or manifest.get("pre_validation_low_evidence_packet_correction") != LOW_EVIDENCE_PACKET_CORRECTION_RECORD.name
            or manifest.get("predecessor_observed_terminal_boundary") != observed_terminal_boundary
            or manifest["authenticated_predecessor_v2r11"]
            != immediate_predecessor_evidence
            or manifest["authenticated_predecessor_v2r10"]
            != source_predecessor_evidence
            or not isinstance(manifest_methodology, dict)
            or manifest_methodology.get("native_receiver_correction")
            != RUNTIME_CORRECTION_RECORD.name
            or not isinstance(inventory, list)
            or manifest["payload_file_count"] != len(inventory)
        ):
            raise StudyGateError("The v2r13 source manifest identity changed")
        checksum_lines = []
        inventory_by_path = {}
        for record in inventory:
            require_mapping(
                record, {"path", "bytes", "sha256", "role"},
                "v2r13 manifest payload record",
            )
            relative_path = str(record["path"])
            if (
                not relative_path
                or relative_path in inventory_by_path
                or PurePosixPath(relative_path).is_absolute()
                or ".." in PurePosixPath(relative_path).parts
                or type(record["bytes"]) is not int
                or record["bytes"] < 0
            ):
                raise StudyGateError("The v2r13 payload inventory is unsafe")
            digest = require_sha256(
                record["sha256"], f"v2r13 payload digest for {relative_path}",
            )
            inventory_by_path[relative_path] = record
            checksum_lines.append(f"{digest}  {relative_path}")
        checksum_lines.append(f"{manifest_sha256}  RELEASE_MANIFEST.json")
        expected_checksums_bytes = (
            "\\n".join(sorted(checksum_lines)) + "\\n"
        ).encode("utf-8")
        requirements_record = inventory_by_path.get("requirements.txt")
        receiver_source_record = inventory_by_path.get(
            "semuav/visdrone_receiver_runtime.py"
        )
        correction_record = inventory_by_path.get(RUNTIME_CORRECTION_RECORD.name)
        isolation_record = inventory_by_path.get(
            ACCEPTANCE_RECOVERY_ISOLATION_RECORD.name
        )
        low_evidence_record = inventory_by_path.get(LOW_EVIDENCE_PACKET_CORRECTION_RECORD.name)
        boundary_routing_record = inventory_by_path.get(
            PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_RECORD.name
        )
        if (
            checksums_bytes != expected_checksums_bytes
            or not isinstance(requirements_record, dict)
            or requirements_record["bytes"] != len(requirements_bytes)
            or requirements_record["sha256"] != requirements_sha256
            or not isinstance(receiver_source_record, dict)
            or receiver_source_record["sha256"]
            != corrected_receiver_facts["sha256"]
            or not isinstance(correction_record, dict)
            or correction_record["sha256"] != runtime_correction_record_sha256
            or not isinstance(isolation_record, dict)
            or isolation_record["sha256"]
            != acceptance_recovery_isolation_record_sha256
            or not isinstance(low_evidence_record, dict)
            or low_evidence_record["sha256"] != low_evidence_packet_correction_sha256
            or not isinstance(boundary_routing_record, dict)
            or boundary_routing_record["sha256"]
            != predecessor_boundary_routing_correction_sha256
        ):
            raise StudyGateError(
                "The v2r13 manifest/checksum/source-correction binding changed"
            )

        artifacts = require_mapping(
            ready["artifacts"], {"pip_freeze"}, "candidate artifacts",
        )
        verified_artifacts = {
            label: verify_file_record(record, f"Candidate artifact {label}")
            for label, record in artifacts.items()
        }
        pip_freeze = verified_artifacts["pip_freeze"]
        pip_freeze_sha256 = pip_freeze["sha256"]
        if (
            require_sha256(
                deployed_candidate["pip_freeze_sha256"],
                "Deployed candidate pip-freeze digest",
            ) != pip_freeze_sha256
            or require_sha256(
                predecessor["pip_freeze_sha256"],
                "Predecessor-equivalent pip-freeze digest",
            ) != pip_freeze_sha256
        ):
            raise StudyGateError("The candidate pip-freeze bindings differ")
        current_pip_freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            check=True, capture_output=True, text=False,
        ).stdout
        if current_pip_freeze != pip_freeze["bytes"]:
            raise StudyGateError("Current distributions differ from the candidate receipt")
        if require_inference_lock or INFERENCE_PIP_FREEZE.exists() or (
            INFERENCE_PIP_FREEZE.is_symlink()
        ):
            inference_pip_freeze = _read_regular_file_bytes(
                INFERENCE_PIP_FREEZE,
                label="v2r13 inference distribution inventory",
            )
            if inference_pip_freeze != pip_freeze["bytes"]:
                raise StudyGateError(
                    "Candidate receipt and inference pip-freeze inventory differ"
                )

        validation = require_mapping(
            manifest["validation"], {"unit_tests", "unit_subtests"},
            "v2r13 manifest validation",
        )
        deployed_tests = require_mapping(
            deployment["unit_tests"], {"primary", "subtests", "junit"},
            "v2r13 deployed unit tests",
        )
        for deployed_name, manifest_name in (
            ("primary", "unit_tests"), ("subtests", "unit_subtests"),
        ):
            observed = require_mapping(
                deployed_tests[deployed_name], {"run", "passed", "failed"},
                f"v2r13 deployed {deployed_name} tests",
            )
            if observed != validation[manifest_name] or (
                observed["run"] != observed["passed"] or observed["failed"] != 0
            ):
                raise StudyGateError("The deployed v2r13 unit-test totals changed")
        verify_file_record(
            deployed_tests["junit"], "v2r13 deployment JUnit report",
        )

        kernel = require_mapping(
            deployment["jupyter_kernel"], {
                "name", "display_name", "kernel_json", "kernel_json_sha256",
            }, "v2r13 Jupyter kernel",
        )
        expected_kernel_name = "semuav-vlm-20260921-v2r13"
        expected_kernel_json = (
            study_root / ".cache" / "jupyter" / "data" / "kernels"
            / expected_kernel_name / "kernel.json"
        )
        kernel_record = verify_file_record(
            {"path": kernel["kernel_json"], "sha256": kernel["kernel_json_sha256"]},
            "v2r13 kernel.json", expected_kernel_json,
        )
        kernel_payload = _decode_json_object_bytes(
            kernel_record["bytes"], label="v2r13 kernel.json",
        )
        expected_kernel_argv = [
            str(candidate_python.absolute()), "-I", "-B", "-m",
            "ipykernel_launcher", "-f", "{connection_file}",
        ]
        if (
            kernel["name"] != expected_kernel_name
            or kernel["display_name"] != "Python (SemUAV-VLM 2026-09-21 v2r13)"
            or kernel_payload.get("display_name") != kernel["display_name"]
            or kernel_payload.get("language") != "python"
            or kernel_payload.get("argv") != expected_kernel_argv
            or set(kernel_payload) != {
                "argv", "display_name", "language", "metadata",
                "kernel_protocol_version",
            }
            or kernel_payload.get("kernel_protocol_version") != "5.5"
            or kernel_payload.get("metadata") != {
                "debugger": True,
                "supported_encryption": "curve",
            }
        ):
            raise StudyGateError("The v2r13 Jupyter-kernel binding changed")

        boundary = require_mapping(
            ready["reproducibility_boundary"], {
                "post_install_freeze_recorded",
                "scientific_use_requires_later_exact_freeze_runtime_fingerprint_and_smoke_seal",
            }, "v2r3 reproducibility boundary",
        )
        if (
            boundary["post_install_freeze_recorded"] is not True
            or boundary[
                "scientific_use_requires_later_exact_freeze_runtime_fingerprint_and_smoke_seal"
            ] is not True
        ):
            raise StudyGateError("The v2r3 reproducibility boundary changed")

        source_compatibility = {
            "schema_version": "semuav.visdrone-source-candidate-binding.v1",
            "release_id": release_id,
            "run_name": run_name,
            "release_path": str(release_root),
            "release_manifest_sha256": manifest_sha256,
            "checksum_inventory_sha256": checksums_sha256,
            "requirements_sha256": requirements_sha256,
            "candidate_environment": str(candidate_root),
            "candidate_pip_freeze_sha256": pip_freeze_sha256,
            "scientific_contract_changed": False,
            "candidate_requirements_changed": False,
            "candidate_environment_reused_without_rebuild": True,
            "candidate_environment_modified": False,
            "package_installation_performed": False,
            "active_and_candidate_source_requirements_identical": True,
            "candidate_source_release_id": candidate_source_release_id,
            "candidate_source_run_name": candidate_source_run_name,
            "candidate_source_path": str(candidate_source_root),
            "candidate_source_manifest_sha256": candidate_manifest_sha256,
            "candidate_source_checksum_inventory_sha256": (
                candidate_checksums_sha256
            ),
            "candidate_source_requirements_sha256": candidate_requirements_sha256,
            "active_model_ids": list(REGISTERED_RUNTIME_SMOKE_MODEL_IDS),
            "excluded_model_ids": ["moondream2"],
            "measurement_schema_version": (
                "semuav.repeated-inference-measurement.v2"
            ),
            "runner_version": "semuav.visdrone-paper-runner.v7",
            "runtime_smoke_input_schema": (
                "semuav.model-runtime-smoke-input.v3"
            ),
            "runtime_smoke_report_schema_version": (
                RUNTIME_SMOKE_REPORT_SCHEMA_VERSION
            ),
            "runtime_smoke_isolation_schema": RUNTIME_SMOKE_ISOLATION_SCHEMA,
            "runtime_smoke_worker_input_schema": (
                "semuav.runtime-smoke-worker-input.v2"
            ),
            "runtime_smoke_worker_result_schema": (
                "semuav.runtime-smoke-worker-result.v2"
            ),
            "trusted_code_cache_receipt_schema": (
                "semuav.visdrone-trusted-code-cache.v2"
            ),
            "fresh_runtime_smoke_required": True,
            "predecessor_smoke_rows_reused": False,
            "temporal_validation_sequences": 7,
            "temporal_test_sequences": 15,
            "temporal_observations_per_sequence": 64,
            "temporal_source_frame_stride": 1,
            "temporal_target_rate_hz": 5,
            "temporal_feasibility_amendment_record": (
                TEMPORAL_FEASIBILITY_AMENDMENT_RECORD.name
            ),
            "native_receiver_correction_record": RUNTIME_CORRECTION_RECORD.name,
            "authenticated_predecessor_v2r9": dict(predecessor_evidence),
            "authenticated_predecessor_v2r10": dict(
                source_predecessor_evidence
            ),
            "authenticated_predecessor_v2r12": dict(sealed_predecessor_evidence),
            "predecessor_v2r12_state": dict(predecessor_state),
            "authenticated_predecessor_v2r11": dict(
                immediate_predecessor_evidence
            ),
            "predecessor_observed_terminal_boundary": dict(
                observed_terminal_boundary
            ),
            "temporal_feasibility_amendment": dict(
                deployment_temporal_amendment
            ),
            "deployment_receipt_sha256": deployment_sha256,
            "archive_sha256": archive["sha256"],
            "builder_sha256": live_builder["sha256"],
            "deployer_sha256": live_deployer["sha256"],
        }
        return {
            "payload": ready,
            "path": candidate_receipt_path,
            "sha256": candidate_receipt_sha256,
            "artifacts": ready["artifacts"],
            "source_compatibility": source_compatibility,
        }

    def _verified_model_lock_derivation():
        receipt_bytes = _read_regular_file_bytes(
            MODEL_LOCK_DERIVATION,
            label="v2r13 model-lock derivation receipt",
        )
        receipt = _decode_json_object_bytes(
            receipt_bytes,
            label="v2r13 model-lock derivation receipt",
        )
        source_payload = _decode_json_object_bytes(
            _read_regular_file_bytes(
                SOURCE_MODEL_LOCK,
                label="historical source model lock",
            ),
            label="historical source model lock",
        )
        active_payload = _decode_json_object_bytes(
            _read_regular_file_bytes(MODEL_LOCK, label="active v2r13 model lock"),
            label="active v2r13 model lock",
        )
        receipt_keys = {
            "schema_version", "source_model_lock_path",
            "source_model_lock_sha256", "source_model_set_sha256",
            "source_model_ids", "active_model_ids", "excluded_model_ids",
            "amendment_id", "exclusion_reason", "derivation_operation",
            "prior_runtime_smoke_rows_reused",
            "validation_or_test_evidence_used", "created_utc", "receipt_type",
            "derived_model_lock_path", "derived_model_lock_sha256",
            "derived_model_set_sha256",
        }
        expected_derivation = {
            "schema_version": (
                "semuav.visdrone-v2r6-three-model-lock-derivation.v1"
            ),
            "source_model_lock_path": str(SOURCE_MODEL_LOCK.resolve()),
            "source_model_lock_sha256": hash_file(SOURCE_MODEL_LOCK),
            "source_model_set_sha256": str(source_payload.get("model_set_sha256", "")),
            "source_model_ids": [
                "qwen3_vl_2b", "smolvlm2_2b", "moondream2",
                "qwen3_vl_8b_upper_bound",
            ],
            "active_model_ids": list(REGISTERED_RUNTIME_SMOKE_MODEL_IDS),
            "excluded_model_ids": ["moondream2"],
            "amendment_id": MODEL_SET_AMENDMENT_ID,
            "exclusion_reason": MODEL_SET_EXCLUSION_REASON,
            "derivation_operation": "ordered_filter_excluding_only_moondream2",
            "prior_runtime_smoke_rows_reused": False,
            "validation_or_test_evidence_used": False,
        }
        embedded_derivation = active_payload.get("derivation")
        amendment_config = cfg.models.get("model_set_amendment")
        if (
            not isinstance(amendment_config, dict)
            or amendment_config.get("schema")
            != "semuav.visdrone-prevalidation-model-set-amendment.v1"
            or amendment_config.get("record") != MODEL_SET_AMENDMENT_RECORD.name
            or amendment_config.get("predecessor_release")
            != "release_20260913_visdrone_paper_v2r5"
            or amendment_config.get("predecessor_smoke_status")
            != "FAILED_3_OF_4_PASS"
            or amendment_config.get("timing")
            != "before_calibration_validation_protocol_freeze_and_test"
            or amendment_config.get("scientific_contract_changed") is not True
            or amendment_config.get("retained_vlm_order")
            != list(REGISTERED_RUNTIME_SMOKE_MODEL_IDS)
            or amendment_config.get("registered_vlm_count") != 3
            or amendment_config.get("compact_selection_candidates")
            != list(REGISTERED_RUNTIME_SMOKE_MODEL_IDS[:2])
            or amendment_config.get("compact_selection_candidate_count") != 2
            or amendment_config.get("upper_bound_reference")
            != REGISTERED_RUNTIME_SMOKE_MODEL_IDS[2]
            or amendment_config.get("required_runtime_smoke")
            != "exact_3_of_3_pass"
            or amendment_config.get("experiment1_general_pipeline_count") != 4
            or amendment_config.get("experiment1_temporal_pipeline_count") != 2
            or amendment_config.get("excluded_models") != {
                "moondream2": {
                    "reason": "one_call_exact_schema_feasibility_failure",
                    "predecessor_status": "FAIL_SCHEMA",
                    "accuracy_evaluated": False,
                    "validation_or_test_outcome_used": False,
                    "scientific_claims_permitted": False,
                }
            }
            or MODEL_SET_AMENDMENT_RECORD.is_symlink()
            or not MODEL_SET_AMENDMENT_RECORD.is_file()
            or
            set(receipt) != receipt_keys
            or any(receipt.get(key) != value for key, value in expected_derivation.items())
            or embedded_derivation != expected_derivation
            or receipt.get("receipt_type")
            != "prospective_pre_validation_model_set_amendment"
            or receipt.get("derived_model_lock_path") != str(MODEL_LOCK.resolve())
            or receipt.get("derived_model_lock_sha256") != hash_file(MODEL_LOCK)
            or receipt.get("derived_model_set_sha256")
            != active_payload.get("model_set_sha256")
            or receipt.get("created_utc") != active_payload.get("created_utc")
        ):
            raise StudyGateError(
                "The inherited v2r6 model-lock derivation receipt changed"
            )
        return {
            "amendment_id": MODEL_SET_AMENDMENT_ID,
            "exclusion_reason": MODEL_SET_EXCLUSION_REASON,
            "source_model_lock_path": str(SOURCE_MODEL_LOCK.resolve()),
            "source_model_lock_sha256": hash_file(SOURCE_MODEL_LOCK),
            "active_model_lock_path": str(MODEL_LOCK.resolve()),
            "active_model_lock_sha256": hash_file(MODEL_LOCK),
            "active_model_set_sha256": str(active_payload["model_set_sha256"]),
            "derivation_receipt_path": str(MODEL_LOCK_DERIVATION.resolve()),
            "derivation_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "registered_model_ids": list(REGISTERED_RUNTIME_SMOKE_MODEL_IDS),
            "excluded_model_ids": ["moondream2"],
            "prior_runtime_smoke_rows_reused": False,
            "validation_or_test_evidence_used": False,
            "amendment_record_path": str(MODEL_SET_AMENDMENT_RECORD.resolve()),
            "amendment_record_sha256": hash_file(MODEL_SET_AMENDMENT_RECORD),
            "predecessor_release_id": "release_20260913_visdrone_paper_v2r5",
            "predecessor_runtime_smoke_status": "FAILED_3_OF_4_PASS",
        }

    def _runtime_smoke_model_identity():
        verified = verify_huggingface_model_lock(MODEL_LOCK, models=cfg.models)
        models = [
            {
                "model_id": str(item["model_id"]),
                "repo": str(item["repo"]),
                "revision": str(item["resolved_revision"]),
                "role": str(item["role"]),
            }
            for item in verified["models"]
        ]
        observed_ids = tuple(item["model_id"] for item in models)
        if observed_ids != REGISTERED_RUNTIME_SMOKE_MODEL_IDS:
            raise StudyGateError(
                "The active model lock is not in the exact registered v2r13 order: "
                f"{observed_ids}"
            )
        return {
            "model_set_sha256": str(verified["model_set_sha256"]),
            "models": models,
            "models_sha256": hash_object(models),
        }

    def _load_frozen_runtime_smoke_input_receipt():
        model_identity = _runtime_smoke_model_identity()
        model_lock_derivation = _verified_model_lock_derivation()
        prompt_text = prompt_for_mission("visdrone_urban").strip()
        generation_policy = _verified_generation_policy()
        expected_generation = {
            **generation_policy,
            "max_new_tokens": int(cfg.models["max_new_tokens"]),
        }
        expected_processed_input = {
            "shared_letterboxed_rgb_size": list(map(int, cfg.models["input_size"])),
            "letterbox": "aspect_preserving",
            "proposal_context": dict(VISDRONE_GENERATION_POLICY["proposal_context"]),
        }
        receipt_bytes = _read_regular_file_bytes(
            MODEL_SMOKE_INPUT_RECEIPT,
            label="frozen runtime-smoke input receipt",
        )
        payload = _decode_json_object_bytes(
            receipt_bytes,
            label="frozen runtime-smoke input receipt",
        )
        request_payload = payload.get("request")
        if not isinstance(request_payload, dict):
            raise StudyGateError("The frozen runtime-smoke request is malformed")
        proposals = request_payload.get("detector_proposals")
        if not isinstance(proposals, list):
            raise StudyGateError("The frozen runtime-smoke proposal list is malformed")
        if (
            payload.get("schema_version") != "semuav.model-runtime-smoke-input.v3"
            or payload.get("scientific_results_eligible") is not False
            or payload.get("excluded_model_ids") != ["moondream2"]
            or payload.get("input_contract") != cfg.models["semantic_contract"]
            or Path(str(payload.get("canary_asset_path", ""))).expanduser().resolve()
            != SMOKE_CANARY.resolve()
            or str(payload.get("canary_asset_sha256", "")).lower()
            != hash_file(SMOKE_CANARY)
            or Path(str(payload.get("model_lock_path", ""))).expanduser().resolve()
            != MODEL_LOCK.resolve()
            or str(payload.get("model_lock_sha256", "")).lower()
            != hash_file(MODEL_LOCK)
            or payload.get("model_lock_derivation") != model_lock_derivation
            or {"moondream_tokenizer", "moondream_tokenizer_sha256"}
            .intersection(payload)
            or payload.get("registered_model_set") != model_identity
            or payload.get("prompt") != prompt_text
            or payload.get("prompt_sha256") != hash_object(prompt_text)
            or payload.get("generation_policy") != expected_generation
            or payload.get("processed_input_contract")
            != expected_processed_input
            or payload.get("request_sha256") != hash_object(request_payload)
            or payload.get("detector_proposals_sha256") != hash_object(proposals)
            or int(payload.get("detector_proposal_count", -1)) != len(proposals)
            or payload.get("source_image_sha256") != request_payload.get("image_sha256")
        ):
            raise StudyGateError("The frozen runtime-smoke input receipt changed")
        return payload

    def _inference_environment_seal_payload(
        *, candidate_ready, inference_runtime, smoke_input, smoke,
        reused_receipt_sha256, trusted_code_cache,
    ):
        model_lock_derivation = _verified_model_lock_derivation()
        prohibited_moondream_fields = {
            "moondream_tokenizer", "moondream_tokenizer_sha256",
        }
        if (
            prohibited_moondream_fields.intersection(smoke_input)
            or prohibited_moondream_fields.intersection(smoke)
            or prohibited_moondream_fields.intersection(
                trusted_code_cache["payload"]
            )
            or smoke_input.get("model_lock_derivation")
            != model_lock_derivation
            or trusted_code_cache["payload"].get("schema_version")
            != "semuav.visdrone-trusted-code-cache.v2"
            or trusted_code_cache["payload"].get("scope")
            != "no_active_trusted_remote_code"
            or trusted_code_cache["payload"].get("model_lock_path")
            != str(MODEL_LOCK.resolve())
            or trusted_code_cache["payload"].get("model_lock_sha256")
            != hash_file(MODEL_LOCK)
            or trusted_code_cache["payload"].get("model_lock_derivation_path")
            != str(MODEL_LOCK_DERIVATION.resolve())
            or trusted_code_cache["payload"].get("model_lock_derivation_sha256")
            != hash_file(MODEL_LOCK_DERIVATION)
            or trusted_code_cache["payload"].get("registered_model_ids")
            != list(REGISTERED_RUNTIME_SMOKE_MODEL_IDS)
            or trusted_code_cache["payload"].get("excluded_model_ids")
            != ["moondream2"]
            or trusted_code_cache["payload"].get("module_roots") != []
            or trusted_code_cache["payload"].get("files") != []
            or trusted_code_cache["payload"].get("file_count") != 0
            or trusted_code_cache["payload"].get("tree_sha256")
            != hash_object([])
        ):
            raise StudyGateError(
                "Inference-seal inputs retain excluded-Moondream state or trusted code"
            )
        return {
            "schema_version": cfg.runtime["inference_environment_seal_schema"],
            "status": cfg.runtime["inference_environment_seal_status"],
            "scientific_results_authorized": True,
            "authorized_scope": "validation_and_one_time_test_only",
            "candidate_environment_promoted_in_place": True,
            "validation_or_test_data_read_before_seal": False,
            "inference_environment_id": cfg.runtime["inference_environment_id"],
            "candidate_ready_receipt": {
                "path": str(candidate_ready["path"]),
                "sha256": candidate_ready["sha256"],
                "schema_version": candidate_ready["payload"]["schema_version"],
                "status": candidate_ready["payload"]["status"],
                "source_scientific_results_authorized": False,
            },
            "source_compatibility": candidate_ready["source_compatibility"],
            "python": {
                "executable": str(Path(sys.executable).resolve()),
                "prefix": str(Path(sys.prefix).resolve()),
            },
            "inference_environment_fingerprint": {
                "path": str(INFERENCE_ENVIRONMENT_FINGERPRINT.resolve()),
                "sha256": inference_runtime["environment_sha256"],
            },
            "inference_pip_freeze": {
                "path": str(INFERENCE_PIP_FREEZE.resolve()),
                "sha256": inference_runtime["pip_freeze_sha256"],
            },
            "registered_model_ids": list(REGISTERED_RUNTIME_SMOKE_MODEL_IDS),
            "excluded_model_ids": ["moondream2"],
            "model_set_amendment": model_lock_derivation,
            "runtime_smoke_input": {
                "path": str(MODEL_SMOKE_INPUT_RECEIPT.resolve()),
                "sha256": hash_file(MODEL_SMOKE_INPUT_RECEIPT),
                "request_sha256": smoke_input["request_sha256"],
                "detector_proposals_sha256": smoke_input[
                    "detector_proposals_sha256"
                ],
            },
            "runtime_smoke": {
                "path": str(MODEL_SMOKE.resolve()),
                "sha256": hash_file(MODEL_SMOKE),
                "all_passed": True,
                "passed_model_ids": [
                    str(row["model_id"]) for row in smoke["models"]
                ],
                "schema_version": RUNTIME_SMOKE_REPORT_SCHEMA_VERSION,
                "execution_isolation_schema": RUNTIME_SMOKE_ISOLATION_SCHEMA,
            },
            "generation_policy": _verified_generation_policy(),
            "trusted_code_cache": {
                "path": str(trusted_code_cache["path"]),
                "sha256": trusted_code_cache["sha256"],
                "tree_sha256": trusted_code_cache["payload"]["tree_sha256"],
                "schema_version": trusted_code_cache["payload"]["schema_version"],
                "scope": trusted_code_cache["payload"]["scope"],
                "file_count": 0,
            },
            "reused_v1_frozen_artifacts_receipt": {
                "path": str(REUSED_ARTIFACTS_RECEIPT.resolve()),
                "sha256": str(reused_receipt_sha256),
            },
            "config_hash": ctx.config_hash,
            "code_hash": ctx.code_hash,
            "runtime_guards": _verified_runtime_guards(),
        }

    def _create_inference_environment_seal(
        *, inference_runtime, smoke_input, smoke, reused_inputs,
        trusted_code_cache,
    ):
        candidate_ready = _verify_candidate_ready_receipt()
        reused_receipt = _read_verified_reused_artifacts_receipt(
            expected_sha256=reused_inputs["receipt_sha256"],
        )
        if reused_receipt["payload"] != reused_inputs["receipt"]:
            raise StudyGateError("Post-smoke reused-v1 receipt payload changed")
        expected = _inference_environment_seal_payload(
            candidate_ready=candidate_ready,
            inference_runtime=inference_runtime,
            smoke_input=smoke_input,
            smoke=smoke,
            reused_receipt_sha256=reused_receipt["sha256"],
            trusted_code_cache=trusted_code_cache,
        )
        if INFERENCE_ENVIRONMENT_SEAL.is_symlink():
            raise StudyGateError("Inference environment seal must not be a symlink")
        if INFERENCE_ENVIRONMENT_SEAL.exists():
            existing = json.loads(
                INFERENCE_ENVIRONMENT_SEAL.read_text(encoding="utf-8")
            )
            if existing != expected:
                raise StudyGateError("Existing inference environment seal differs")
        else:
            atomic_write_json(INFERENCE_ENVIRONMENT_SEAL, expected)
        return expected

    def _verify_inference_environment_seal():
        # This verifier reads only environment/provenance receipts. It must pass
        # before validation or untouched-test source data are opened.
        if (
            not INFERENCE_ENVIRONMENT_SEAL.is_file()
            or INFERENCE_ENVIRONMENT_SEAL.is_symlink()
        ):
            raise StudyGateError(
                "INFERENCE_ENVIRONMENT_SEAL.json is required for scientific execution"
            )
        candidate_ready = _verify_candidate_ready_receipt()
        inference_runtime = _capture_or_verify_inference_runtime(allow_create=False)
        smoke_input = _load_frozen_runtime_smoke_input_receipt()
        smoke = _verify_runtime_smoke_report(smoke_input)
        trusted_code_cache = _capture_or_verify_trusted_code_cache(
            allow_create=False,
        )
        reused_inputs = _verify_or_create_reused_v1_artifacts_receipt(
            allow_create=False,
        )
        expected = _inference_environment_seal_payload(
            candidate_ready=candidate_ready,
            inference_runtime=inference_runtime,
            smoke_input=smoke_input,
            smoke=smoke,
            reused_receipt_sha256=reused_inputs["receipt_sha256"],
            trusted_code_cache=trusted_code_cache,
        )
        observed = json.loads(
            INFERENCE_ENVIRONMENT_SEAL.read_text(encoding="utf-8")
        )
        if observed != expected:
            raise StudyGateError("The inference environment scientific-use seal changed")
        return {
            "payload": observed,
            "sha256": hash_file(INFERENCE_ENVIRONMENT_SEAL),
            "runtime": inference_runtime,
            "smoke_input": smoke_input,
            "smoke": smoke,
            "trusted_code_cache": trusted_code_cache,
            "reused_inputs": reused_inputs,
        }

    def _load_verified_runtime_smoke_canary():
        expected_canary_sha256 = str(
            cfg.models["runtime_smoke_canary_sha256"]
        ).lower()
        if not SMOKE_CANARY.is_file() or SMOKE_CANARY.is_symlink():
            raise StudyGateError(f"Runtime-smoke canary is missing or unsafe: {SMOKE_CANARY}")
        if hash_file(SMOKE_CANARY) != expected_canary_sha256:
            raise StudyGateError("Runtime-smoke canary asset hash changed")
        canary = json.loads(SMOKE_CANARY.read_text(encoding="utf-8"))
        if (
            canary.get("schema_version")
            != "semuav.visdrone-runtime-smoke-canary.v1"
            or canary.get("scope") != "development_registered_train_canary"
            or canary.get("scientific_results_eligible") is not False
            or canary.get("source_run_name") != "semuav_visdrone_v1"
            or canary.get("split") != "train"
            or canary.get("component") != "det_train"
            or canary.get("publisher_partition") != "train"
            or canary.get("study_partition") != "development_train"
        ):
            raise StudyGateError("Runtime-smoke canary provenance is not development-train only")
        relative_text = str(canary.get("image_relative_path", ""))
        relative = PurePosixPath(relative_text)
        if (
            not relative_text
            or relative.is_absolute()
            or relative.as_posix() != relative_text
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise StudyGateError("Runtime-smoke canary image path is not source-relative")
        image_source = VISDRONE_ROOT.joinpath(*relative.parts)
        if image_source.is_symlink():
            raise StudyGateError("Runtime-smoke canary image must not be a symlink")
        image_path = image_source.resolve(strict=True)
        if image_path.parent.name != "images" or "VisDrone2019-DET-train" not in image_path.parts:
            raise StudyGateError("Runtime-smoke canary escaped publisher DET train")
        if hash_file(image_path) != str(canary.get("image_sha256", "")).lower():
            raise StudyGateError("Runtime-smoke canary image bytes changed")

        train_manifest_path = PRESEAL_REUSED_V1_ARTIFACT_PATHS[
            "registered_train_sample"
        ]
        train_manifest = read_analysis_manifest(train_manifest_path)
        matches = train_manifest.loc[
            train_manifest["sample_id"].astype(str).eq(str(canary["sample_id"]))
        ]
        if len(matches) != 1:
            raise StudyGateError("Runtime-smoke canary is not unique in the registered train sample")
        selected = matches.iloc[0]
        recorded_telemetry = json.loads(str(selected["telemetry_json"]))
        if (
            str(selected["source_sample_id"]) != str(canary["source_sample_id"])
            or str(selected["source_uid"]) != str(canary["source_uid"])
            or str(selected["component"]) != "det_train"
            or str(selected["split"]) != "train"
            or str(selected["mission"]) != str(canary["mission"])
            or int(selected["capture_time_us"]) != int(canary["capture_time_us"])
            or Path(str(selected["image_path"])).expanduser().resolve() != image_path
            or recorded_telemetry != canary["telemetry"]
        ):
            raise StudyGateError("Runtime-smoke canary differs from its registered train row")

        proposals = canary.get("detector_proposals")
        policy = cfg.models["proposal_policy"]
        if (
            not isinstance(proposals, list)
            or len(proposals) != int(canary.get("detector_proposal_count", -1))
            or len(proposals) != int(policy["top_k"])
        ):
            raise StudyGateError("Runtime-smoke canary proposal count changed")
        allowed_labels = set(
            cfg.benchmark["mission_profiles"]["general_mobility"]["priority_categories"]
        )
        previous_confidence = float("inf")
        for index, proposal in enumerate(proposals, start=1):
            if not isinstance(proposal, dict) or set(proposal) != {
                "bbox_normalized_xyxy", "confidence", "label", "track_id",
            }:
                raise StudyGateError("Runtime-smoke canary proposal schema changed")
            box = proposal["bbox_normalized_xyxy"]
            confidence = float(proposal["confidence"])
            if (
                not isinstance(box, list) or len(box) != 4
                or not np.isfinite(np.asarray(box, dtype=float)).all()
                or not all(0.0 <= float(value) <= 1.0 for value in box)
                or not (float(box[0]) < float(box[2]) and float(box[1]) < float(box[3]))
                or str(proposal["label"]) not in allowed_labels
                or int(proposal["track_id"]) != index
                or confidence < float(policy["minimum_confidence"])
                or confidence > previous_confidence
            ):
                raise StudyGateError("Runtime-smoke canary proposal content/order changed")
            previous_confidence = confidence
        source_proposal_bytes = (
            json.dumps(proposals, indent=2, sort_keys=True, ensure_ascii=False) + "\\n"
        ).encode("utf-8")
        if hashlib.sha256(source_proposal_bytes).hexdigest() != str(
            canary.get("source_detector_proposals_sha256", "")
        ).lower():
            raise StudyGateError("Runtime-smoke canary proposal source hash changed")
        return canary, image_path, proposals

    def _runtime_smoke_input_payload(canary, image_path, proposals):
        model_identity = _runtime_smoke_model_identity()
        model_lock_derivation = _verified_model_lock_derivation()
        prompt_text = prompt_for_mission("visdrone_urban").strip()
        generation_policy = {
            **_verified_generation_policy(),
            "max_new_tokens": int(cfg.models["max_new_tokens"]),
        }
        processed_input_contract = {
            "shared_letterboxed_rgb_size": list(map(int, cfg.models["input_size"])),
            "letterbox": "aspect_preserving",
            "proposal_context": dict(VISDRONE_GENERATION_POLICY["proposal_context"]),
        }
        request_payload = {
            "observation_id": str(canary["sample_id"]),
            "mission": str(canary["mission"]),
            "capture_time_us": int(canary["capture_time_us"]),
            "telemetry": dict(canary["telemetry"]),
            "image_sha256": hash_file(image_path),
            "detector_proposals": list(proposals),
        }
        return {
            "schema_version": "semuav.model-runtime-smoke-input.v3",
            "input_contract": str(cfg.models["semantic_contract"]),
            "canary_asset_path": str(SMOKE_CANARY.resolve()),
            "canary_asset_sha256": hash_file(SMOKE_CANARY),
            "model_lock_path": str(MODEL_LOCK.resolve()),
            "model_lock_sha256": hash_file(MODEL_LOCK),
            "model_lock_derivation": model_lock_derivation,
            "excluded_model_ids": ["moondream2"],
            "registered_model_set": model_identity,
            "prompt": prompt_text,
            "prompt_sha256": hash_object(prompt_text),
            "generation_policy": generation_policy,
            "processed_input_contract": processed_input_contract,
            "source_image_relative_path": str(canary["image_relative_path"]),
            "source_image_sha256": hash_file(image_path),
            "source_detector_proposals_sha256": str(
                canary["source_detector_proposals_sha256"]
            ),
            "detector_proposal_count": len(proposals),
            "detector_proposals_sha256": hash_object(list(proposals)),
            "request_sha256": hash_object(request_payload),
            "request": request_payload,
            "scientific_results_eligible": False,
        }

    def _verify_runtime_smoke_report(input_payload):
        require_runtime_smoke_pass(MODEL_SMOKE)
        smoke = json.loads(MODEL_SMOKE.read_text(encoding="utf-8"))
        generation_policy = _verified_generation_policy()
        verified_lock = verify_huggingface_model_lock(
            MODEL_LOCK, models=cfg.models,
        )
        expected_models = {
            str(item["model_id"]): {
                "repo": str(item["repo"]),
                "revision": str(item["resolved_revision"]),
            }
            for item in verified_lock["models"]
        }
        rows = smoke.get("models", [])
        expected_prompt_sha256 = hash_object(
            prompt_for_mission("visdrone_urban").strip()
        )
        expected_context_sha256 = str(
            smoke.get("expected_decision_context_sha256", "")
        )
        execution_isolation = smoke.get("execution_isolation")
        isolation_artifacts = (
            execution_isolation.get("artifacts")
            if isinstance(execution_isolation, dict) else None
        )
        isolation_environment = (
            execution_isolation.get("worker_environment")
            if isinstance(execution_isolation, dict) else None
        )
        isolation_valid = bool(
            isinstance(execution_isolation, dict)
            and execution_isolation.get("schema_version")
            == RUNTIME_SMOKE_ISOLATION_SCHEMA
            and execution_isolation.get("strategy")
            == "fresh_exec_per_locked_model"
            and isinstance(isolation_environment, dict)
            and isolation_environment.get("SEMUAV_MOONDREAM_TOKENIZER_LOCK")
            is None
            and isolation_environment.get("SEMUAV_MOONDREAM_REVISION") is None
            and not {
                "moondream_tokenizer", "moondream_tokenizer_sha256",
            }.intersection(execution_isolation)
            and isinstance(isolation_artifacts, list)
            and len(isolation_artifacts) == 3
            and tuple(
                str(item.get("model_id", ""))
                for item in isolation_artifacts
                if isinstance(item, dict)
            ) == REGISTERED_RUNTIME_SMOKE_MODEL_IDS
            and hash_object(isolation_artifacts)
            == execution_isolation.get("artifacts_sha256")
        )
        row_provenance_valid = True
        decision_context_hashes = set()
        for row in rows if isinstance(rows, list) else ():
            if not isinstance(row, dict):
                row_provenance_valid = False
                continue
            model_id = str(row.get("model_id", ""))
            expected_model = expected_models.get(model_id, {})
            raw_text = row.get("raw_text")
            decision_core_json = row.get("decision_core_json")
            try:
                decision_core = json.loads(str(decision_core_json))
            except (TypeError, ValueError, json.JSONDecodeError):
                decision_core = None
            decision_context_sha256 = str(
                row.get("decision_context_sha256", "")
            )
            decision_context_hashes.add(decision_context_sha256)
            expected_strategy = VISDRONE_GENERATION_STRATEGY_BY_MODEL.get(model_id)
            expected_prefill = visdrone_assistant_prefill_for_model(model_id)
            generated_continuation = (
                raw_text[len(expected_prefill):]
                if isinstance(raw_text, str)
                and raw_text.startswith(expected_prefill)
                else None
            )
            worker_provenance_valid = bool(
                row.get("worker_isolation_schema")
                == RUNTIME_SMOKE_ISOLATION_SCHEMA
                and row.get("worker_result_valid") is True
                and row.get("worker_returncode") == 0
                and row.get("worker_signal") is None
                and row.get("worker_timed_out") is False
                and not any(
                    str(key).startswith("worker_moondream_")
                    for key in row
                )
            )
            moondream_fields = (
                "moondream_image_positions",
                "moondream_native_template_tokens",
                "moondream_question_tokens",
                "moondream_prefill_tokens",
                "moondream_generation_reserve_tokens",
                "moondream_required_tokens",
                "moondream_effective_context_tokens",
                "moondream_context_margin_tokens",
                "moondream_tokenizer_lock_sha256",
                "moondream_tokenizer_artifact_sha256",
            )
            generation_provenance_valid = bool(
                isinstance(row.get("input_tokens"), int)
                and not isinstance(row.get("input_tokens"), bool)
                and row["input_tokens"] > 0
                and isinstance(row.get("output_tokens"), int)
                and not isinstance(row.get("output_tokens"), bool)
                and row["output_tokens"] > 0
                and row.get("terminated_by_eos") is True
                and row.get("reached_token_cap") is False
                and not any(field in row for field in moondream_fields)
            )
            row_provenance_valid = row_provenance_valid and bool(
                model_id in expected_models
                and row.get("repo") == expected_model.get("repo")
                and row.get("revision") == expected_model.get("revision")
                and row.get("record_model_id") == model_id
                and row.get("record_revision") == expected_model.get("revision")
                and row.get("record_observation_id")
                == input_payload["request"]["observation_id"]
                and row.get("record_prompt_hash") == expected_prompt_sha256
                and row.get("input_contract") == input_payload["input_contract"]
                and row.get("request_sha256") == input_payload["request_sha256"]
                and row.get("detector_proposals_sha256")
                == input_payload["detector_proposals_sha256"]
                and row.get("detector_proposal_count")
                == input_payload["detector_proposal_count"]
                and row.get("image_sha256") == input_payload["source_image_sha256"]
                and row.get("decision_contract_version")
                == input_payload["input_contract"]
                and row.get("decision_core_valid") is True
                and isinstance(decision_core, dict)
                and row.get("decision_proposal_count")
                == input_payload["detector_proposal_count"]
                and row.get("decision_composer_policy_sha256")
                == VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256
                and decision_context_sha256 == expected_context_sha256
                and row.get("generation_policy_version")
                == generation_policy["version"]
                and row.get("generation_policy_sha256")
                == generation_policy["sha256"]
                and row.get("generation_strategy") == expected_strategy
                and row.get("generation_max_new_tokens")
                == int(cfg.models["max_new_tokens"])
                and row.get("assistant_prefill_characters")
                == len(expected_prefill)
                and row.get("assistant_prefill_sha256")
                == hashlib.sha256(expected_prefill.encode("utf-8")).hexdigest()
                and isinstance(generated_continuation, str)
                and row.get("generated_continuation_sha256")
                == hashlib.sha256(
                    generated_continuation.encode("utf-8")
                ).hexdigest()
                and isinstance(raw_text, str) and bool(raw_text)
                and row.get("raw_text_sha256")
                == hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
                and isinstance(row.get("parsed_packet_sha256"), str)
                and len(row["parsed_packet_sha256"]) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in row["parsed_packet_sha256"]
                )
                and generation_provenance_valid
                and worker_provenance_valid
            )
        if (
            smoke.get("schema_version") != RUNTIME_SMOKE_REPORT_SCHEMA_VERSION
            or smoke.get("all_passed") is not True
            or not isinstance(rows, list)
            or len(rows) != 3
            or any(not isinstance(row, dict) for row in rows)
            or tuple(str(row.get("model_id")) for row in rows)
            != REGISTERED_RUNTIME_SMOKE_MODEL_IDS
            or {str(row.get("model_id")) for row in rows} != set(expected_models)
            or any(str(row.get("status")) != "PASS" for row in rows)
            or not row_provenance_valid
            or not isolation_valid
            or len(decision_context_hashes) != 1
            or {"moondream_tokenizer", "moondream_tokenizer_sha256"}
            .intersection(smoke)
            or {"moondream_tokenizer", "moondream_tokenizer_sha256"}
            .intersection(input_payload)
            or Path(str(smoke.get("model_lock", ""))).expanduser().resolve()
            != MODEL_LOCK.resolve()
            or str(smoke.get("model_lock_sha256", "")) != hash_file(MODEL_LOCK)
            or str(smoke.get("model_set_sha256", ""))
            != str(verified_lock["model_set_sha256"])
            or str(smoke.get("code_sha256", "")) != ctx.code_hash
            or str(smoke.get("composer_policy_sha256", ""))
            != VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256
            or str(smoke.get("generation_policy_version", ""))
            != generation_policy["version"]
            or str(smoke.get("generation_policy_sha256", ""))
            != generation_policy["sha256"]
            or len(expected_context_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_context_sha256
            )
            or str(smoke.get("prompt_sha256", "")) != expected_prompt_sha256
            or str(smoke.get("image_sha256", ""))
            != input_payload["source_image_sha256"]
            or str(smoke.get("request_sha256", ""))
            != input_payload["request_sha256"]
            or str(smoke.get("detector_proposals_sha256", ""))
            != input_payload["detector_proposals_sha256"]
            or smoke.get("detector_proposal_count")
            != input_payload["detector_proposal_count"]
            or str(smoke.get("input_contract", ""))
            != input_payload["input_contract"]
        ):
            raise StudyGateError(
                "Runtime smoke is not an exact 3/3 PASS for the bound production-shaped input"
            )
        return smoke

    print("Run root:", ctx.run_root)
    print("Config hash:", ctx.config_hash)
    print("Code hash:", ctx.code_hash)
    print("VisDrone root:", VISDRONE_ROOT)
    print("Reused immutable v1 inputs:", REUSED_V1_FROZEN_INPUT_DIR)
    print("Scientific scope:", cfg.study_scope.value, "| truth:", cfg.truth_mode.value)
    """),
    md("""
    ## Provenance, runtime, and software acceptance

    These checks are evidence, not scientific outcomes. `flame3 blocked` is not
    relevant to this single-dataset scope. Any failed test or missing source receipt
    blocks scientific execution.
    """),
    code("""
    source_receipt = VISDRONE_ROOT / "SOURCE_RECEIPT.json"
    source_inventory = VISDRONE_ROOT / "SOURCE_FILES_SHA256.jsonl"
    if EXECUTE_PHASE and PHASE in {"prepare", "validation", "test"}:
        _require_v2r13_preimport_source_gate()
        _require_exclusive_scientific_gpu(f"{PHASE}_initial_runtime_preflight")
        preflight = environment_report(
            PROJECT_ROOT, cfg.paths.cache_root, DEVICE, cfg.profile.value,
        )
        source_checks = pd.DataFrame([
            {"artifact": "SOURCE_RECEIPT.json", "path": str(source_receipt), "present": source_receipt.is_file()},
            {"artifact": "SOURCE_FILES_SHA256.jsonl", "path": str(source_inventory), "present": source_inventory.is_file()},
        ])
    elif PHASE in {"prepare", "validation", "test"}:
        preflight = pd.DataFrame([
            {
                "item": "scientific runtime preflight",
                "status": "DEFERRED",
                "value": "EXECUTE_PHASE=False",
                "required_for": PHASE,
            },
        ])
        source_checks = pd.DataFrame([
            {
                "artifact": "source provenance",
                "path": "deferred until authenticated phase execution",
                "present": "DEFERRED",
            },
        ])
    else:
        # Report/embedded never probe data, checkpoint, or GPU paths. Their
        # dedicated cells consume only a finalized hash-verified handoff.
        preflight = pd.DataFrame([
            {"item": "python", "status": "PASS", "value": sys.version.split()[0], "required_for": PHASE},
            {"item": "project_root", "status": "PASS", "value": str(PROJECT_ROOT), "required_for": PHASE},
            {"item": "package:numpy", "status": "PASS", "value": np.__version__, "required_for": "report"},
            {"item": "package:pandas", "status": "PASS", "value": pd.__version__, "required_for": "report"},
        ])
        source_checks = pd.DataFrame([
            {"artifact": "source provenance", "path": "deferred to finalized protocol handoff", "present": "DEFERRED"},
        ])
    display(preflight)
    display(source_checks)
    if EXECUTE_PHASE and PHASE in {"prepare", "validation", "test"}:
        _require_v2r13_preimport_source_gate()
        if preflight["status"].eq("FAIL").any():
            failed = preflight.loc[preflight["status"].eq("FAIL"), "item"].astype(str).tolist()
            raise StudyGateError(f"Environment preflight failed: {failed}")
        if not source_checks["present"].all():
            raise StudyGateError("Official VisDrone source provenance is incomplete")
    """),
    code("""
    # The live notebook must retain its exact lock-bound loader environment for
    # the runtime smoke and every later scientific phase.  A small subset of
    # unit tests intentionally creates temporary model locks, however, so pass
    # them a private copy with lock-specific cache/revision values and the
    # parent-only scientific recovery capability removed.
    # Offline execution remains mandatory in both the parent and child.
    unittest_offline_guards = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "SEMUAV_MODELS_LOCAL_ONLY": "1",
    }
    for variable, expected in unittest_offline_guards.items():
        if os.environ.get(variable) != expected:
            raise RuntimeError(
                f"Notebook runtime offline guard is not exact: {variable}"
            )

    unittest_environment = os.environ.copy()
    unittest_lock_specific_variables = (
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "SEMUAV_QWEN_REVISION",
        "SEMUAV_SMOLVLM_REVISION",
        "SEMUAV_MOONDREAM_REVISION",
        "SEMUAV_UPPER_BOUND_REVISION",
        "SEMUAV_MOONDREAM_TOKENIZER_LOCK",
    )
    # Recovery is a validation-only capability bound to the authenticated
    # scientific parent process.  Generic software-acceptance fixtures use
    # non-scientific run contexts and must never inherit that capability.
    # Remove all three bindings only from the child environment; leave the
    # parent unchanged so the subsequent official validation cells can import
    # the deployment-frozen v2r9 Experiment-1 shards.
    unittest_recovery_variables = (
        "SEMUAV_ENABLE_V2R9_EXP1_RECOVERY",
        "SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT",
        "SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT_SHA256",
    )
    for variable in (
        *unittest_lock_specific_variables,
        *unittest_recovery_variables,
    ):
        unittest_environment.pop(variable, None)
    if any(
        variable in unittest_environment
        for variable in unittest_recovery_variables
    ):
        raise RuntimeError(
            "Unit-test subprocess recovery controls were not isolated"
        )
    unittest_environment["SEMUAV_LIVE_NOTEBOOK_SELFTEST"] = "1"
    for variable, expected in unittest_offline_guards.items():
        if unittest_environment.get(variable) != expected:
            raise RuntimeError(
                f"Unit-test subprocess offline guard is not exact: {variable}"
            )

    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(PROJECT_ROOT / "tests"), "-v"],
        cwd=PROJECT_ROOT, text=True, capture_output=True,
        env=unittest_environment,
    )
    print(tests.stdout)
    print(tests.stderr)
    if tests.returncode:
        raise RuntimeError("Software acceptance tests failed; do not execute a study phase")
    print("Software acceptance tests: PASS")
    """),
    md("""
    ## Publisher partition audit

    This cell is read-only. It establishes that DET/MOT train and validation are
    development data and that both test-dev components remain untouched.
    """),
    code("""
    component_registry = publisher_split_mapping()
    display(component_registry)
    assert set(component_registry.loc[
        component_registry["study_partition"].eq("untouched_test"), "component"
    ]) == {"det_test_dev", "mot_test_dev"}
    """),
    md("""
    ## Registered source-truth rule and mission policies

    Every gold endpoint is derived deterministically from publisher category,
    box/track visibility, the frozen image-plane ROI, and the registered mission
    profile. Neither a detector nor a VLM supplies truth. The table below is an
    input contract, not a result.
    """),
    code("""
    source_truth_policies = registered_mission_policies(cfg.benchmark)
    display(source_truth_policies)
    """),
    md("""
    # Phase 1 — v2r13 native-receiver correction and fresh three-model smoke

    This sibling release does not rebuild data, retrain the detector, or refit the
    feature head. It authenticates v2r9's hash-backed completed 3/3 development
    smoke and scientific-use seal. It separately records the source-authenticated
    L0-heartbeat/native-receiver defect and the operator-observed validation
    boundary: all 132 raw Experiment 1 shards (4,152 rows), model selection, and
    semantic-packet construction completed, then execution stopped on the first
    native `l0_cbor` receiver plan. It permits only hash-inventoried, byte-identical
    raw Experiment 1 shard recovery; all derived metrics, selection, semantic
    packets, receiver caches, and downstream results are recomputed. The corrected
    receiver both validates L0 before returning registered `NO_ACTION` and binds
    authoritative pre-shared capture time plus latitude, longitude, altitude, and
    yaw into every pixel detector request; L3 overrides only the detector ROI with
    received-crop coordinates. The full registered L0-through-video native cache is
    covered end-to-end before scientific use. It inherits
    the controlling v2r6 Moondream exclusion and derives a fresh run-local
    three-row active model lock.
    No v2r9 smoke row or derived validation artifact is reused. Before smoke it verifies only the
    registered train sample plus exact model, detector, and train-only feature-head
    provenance, then records that restricted inventory in
    `REUSED_V1_FROZEN_ARTIFACTS.json`. It does not open or hash the prepared
    development bundle or registered calibration/validation samples until the
    scientific-use seal has passed in the validation phase. The historical
    `ENVIRONMENT_FINGERPRINT.json` and `PIP_FREEZE.txt` remain evidence of
    detector training only.

    The clean v2r13 inference environment is frozen separately as
    `INFERENCE_ENVIRONMENT_FINGERPRINT.json` and `INFERENCE_PIP_FREEZE.txt`.
    The runtime smoke uses one release-shipped, hash-bound, registered DET-train
    canary with its exact 32 detector proposals. It is development diagnostics,
    never a scientific result, and validation remains locked until all three
    registered models pass that exact production-shaped request. Only then is
    `INFERENCE_ENVIRONMENT_SEAL.json` written to promote the same exact candidate
    environment, in place, for scientific validation and the one-time test.
    """),
    code("""
    reused_inputs = inference_runtime = scientific_environment_seal = None
    if PHASE == "prepare" and EXECUTE_PHASE:
        _require_v2r13_preimport_source_gate()
        # Authenticate the externally hash-bound v2r13 deployment and the clean
        # current-shell guard set before any pre-smoke registration work.
        _verify_candidate_ready_receipt(require_inference_lock=False)
        validate_registered_model_configuration(cfg.models)
        reused_inputs = _verify_or_create_reused_v1_artifacts_receipt(
            allow_create=True,
        )
        pre_smoke_reuse_receipt_sha256 = reused_inputs["receipt_sha256"]
        inference_runtime = _capture_or_verify_inference_runtime(
            allow_create=True,
        )
        print(
            "REUSED_V1_FROZEN_ARTIFACTS_VERIFIED",
            reused_inputs["receipt_sha256"],
            "artifact_count=",
            len(reused_inputs["receipt"]["artifacts"]),
        )
        print(
            "HISTORICAL_DETECTOR_TRAINING_ENVIRONMENT_PRESERVED",
            hash_file(DETECTOR_TRAINING_ENVIRONMENT_FINGERPRINT),
            hash_file(DETECTOR_TRAINING_PIP_FREEZE),
        )
        print(
            "V2R13_INFERENCE_ENVIRONMENT_FROZEN",
            inference_runtime["environment_sha256"],
            inference_runtime["pip_freeze_sha256"],
        )
        print(
            "V2R13_THREE_MODEL_LOCK_VERIFIED",
            hash_file(MODEL_LOCK),
            "derivation_receipt_sha256=",
            hash_file(MODEL_LOCK_DERIVATION),
        )

        if RUN_MODEL_SMOKE:
            _require_pristine_trusted_code_cache()
            _require_exclusive_scientific_gpu("prepare_runtime_smoke")
            canary, smoke_image_path, detector_proposals = (
                _load_verified_runtime_smoke_canary()
            )
            smoke_input = _runtime_smoke_input_payload(
                canary, smoke_image_path, detector_proposals,
            )
            if (
                MODEL_SMOKE_INPUT_RECEIPT.exists()
                or MODEL_SMOKE_INPUT_RECEIPT.is_symlink()
            ):
                existing_smoke_input = _load_frozen_runtime_smoke_input_receipt()
                if existing_smoke_input != smoke_input:
                    raise StudyGateError(
                        "Existing runtime-smoke input receipt differs from the canary"
                    )
            else:
                atomic_write_json(MODEL_SMOKE_INPUT_RECEIPT, smoke_input)
            request = InferenceRequest(
                observation_id=str(canary["sample_id"]),
                image_path=str(smoke_image_path),
                mission=str(canary["mission"]),
                capture_time_us=int(canary["capture_time_us"]),
                telemetry=dict(canary["telemetry"]),
                detector_proposals=tuple(detector_proposals),
            )
            post_smoke_runtime = None
            try:
                smoke_frame = run_one_image_runtime_smoke(
                    MODEL_LOCK, request, output_path=MODEL_SMOKE,
                    device=DEVICE, dtype=str(cfg.runtime["dtype"]),
                    max_new_tokens=int(cfg.models["max_new_tokens"]),
                    prompt=prompt_for_mission("visdrone_urban"),
                )
                display(smoke_frame)
                verified_smoke = _verify_runtime_smoke_report(smoke_input)
            finally:
                # Detect any import-time package mutation before leaving smoke.
                post_smoke_runtime = _capture_or_verify_inference_runtime(
                    allow_create=False,
                )
            trusted_code_cache = _capture_or_verify_trusted_code_cache(
                allow_create=True,
            )
            post_smoke_reused_inputs = (
                _verify_or_create_reused_v1_artifacts_receipt(
                    allow_create=False,
                )
            )
            if (
                post_smoke_reused_inputs["receipt_sha256"]
                != pre_smoke_reuse_receipt_sha256
            ):
                raise StudyGateError(
                    "Reused-v1 receipt changed while the runtime smoke was executing"
                )
            scientific_environment_seal = _create_inference_environment_seal(
                inference_runtime=post_smoke_runtime,
                smoke_input=smoke_input,
                smoke=verified_smoke,
                reused_inputs=post_smoke_reused_inputs,
                trusted_code_cache=trusted_code_cache,
            )
            print(
                "MODEL_RUNTIME_SMOKE_3_OF_3_PASS",
                hash_file(MODEL_SMOKE),
                "input_receipt_sha256=",
                hash_file(MODEL_SMOKE_INPUT_RECEIPT),
            )
            print(
                "TRUSTED_CODE_CACHE_FROZEN",
                trusted_code_cache["sha256"],
                "tree_sha256=",
                trusted_code_cache["payload"]["tree_sha256"],
            )
            print(
                "INFERENCE_ENVIRONMENT_SEALED_FOR_SCIENTIFIC_USE",
                hash_file(INFERENCE_ENVIRONMENT_SEAL),
            )
        else:
            print("Model runtime smoke paused; set RUN_MODEL_SMOKE=True to run it.")
    else:
        print("V2R13 reuse-registration/model-smoke stage paused.")
    """, "prepare-only"),
    md("""
    ### Prepare-phase stop

    Stop here. Continue to validation only after the output includes
    `REUSED_V1_FROZEN_ARTIFACTS_VERIFIED`,
    `V2R13_INFERENCE_ENVIRONMENT_FROZEN`, `MODEL_RUNTIME_SMOKE_3_OF_3_PASS`, and
    `TRUSTED_CODE_CACHE_FROZEN`, and
    `INFERENCE_ENVIRONMENT_SEALED_FOR_SCIENTIFIC_USE`. Do not run detector
    training, feature-head training, package installation, model downloads, or
    data preparation from this notebook.
    """),
    code("""
    if PHASE == "prepare":
        print("PHASE_BARRIER: v2r13 prepare complete; validation/test/report cells cannot execute in this run")
    """, "phase-barrier"),
    md("""
    # Phase 2 — validation and protocol freeze

    Validation verifies the previously completed, production-shaped 3/3 runtime
    smoke and all pre-seal reused-v1/inference-runtime provenance. Only after that
    seal passes does it verify and load the full frozen development bundle and
    registered validation sample, before it produces exact detector/VLM caches,
    calibrates confidence,
    applies the registered VLM go/no-go rule, builds measured native L0–L5 receiver
    caches, and executes development versions of Experiments 1–7. Experiment 7 is
    readiness only; no Jetson claim can arise from RTX 4090 measurements.
    """),
    code("""
    validation_inputs = {}
    if PHASE == "validation" and EXECUTE_PHASE:
        _require_v2r13_preimport_source_gate()
        _require_exclusive_scientific_gpu("validation_input_gate")
        scientific_environment = _verify_inference_environment_seal()
        reused_inputs = _verify_or_create_reused_v1_artifacts_receipt(
            allow_create=False,
        )
        full_reused_inputs = _verify_full_reused_v1_development_bundle()
        if (
            full_reused_inputs["scientific_environment"]["sha256"]
            != scientific_environment["sha256"]
        ):
            raise StudyGateError(
                "Scientific-use seal changed before full development verification"
            )
        inference_runtime = scientific_environment["runtime"]
        smoke_input = scientific_environment["smoke_input"]
        verified_model_lock = verify_huggingface_model_lock(
            MODEL_LOCK, models=cfg.models,
        )
        verified_detector_training = full_reused_inputs["detector_training"]
        verified_detector_checkpoint = full_reused_inputs["detector_checkpoint"]
        development_bundle = full_reused_inputs["development_bundle"]
        configured_detector_receipt = DETECTOR_TRAINING_RECEIPT.resolve()
        print(
            "VALIDATION_3_OF_3_RUNTIME_SMOKE_GATE_PASS",
            hash_file(MODEL_SMOKE),
            "reuse_receipt_sha256=",
            reused_inputs["receipt_sha256"],
            "inference_environment_sha256=",
            inference_runtime["environment_sha256"],
            "scientific_seal_sha256=",
            scientific_environment["sha256"],
        )
        print(
            "POST_SEAL_FULL_DEVELOPMENT_BUNDLE_VERIFIED",
            DEVELOPMENT_BUNDLE_DIR,
            "registered_splits=",
            sorted(full_reused_inputs["registered_sample_artifacts"]),
        )
        validation_temporal_spec = registered_temporal_sampling_spec(
            cfg.benchmark, "validation"
        )
        test_temporal_spec = registered_temporal_sampling_spec(
            cfg.benchmark, "test"
        )
        registered_validation_traces = int(
            cfg.network["validation_trace_realizations"]
        )
        registered_validation_paired_cells = int(
            cfg.network["validation_fully_paired_scene_trace_cells"]
        )
        derived_validation_paired_cells = (
            int(validation_temporal_spec["sequences_per_split"])
            * registered_validation_traces
        )
        if (
            int(validation_temporal_spec["sequences_per_split"]) != 7
            or int(test_temporal_spec["sequences_per_split"]) != 15
            or int(validation_temporal_spec["observations_per_sequence"]) != 64
            or int(test_temporal_spec["observations_per_sequence"]) != 64
            or int(validation_temporal_spec["source_frame_stride"]) != 1
            or int(test_temporal_spec["source_frame_stride"]) != 1
            or int(validation_temporal_spec["target_rate_hz"]) != 5
            or int(test_temporal_spec["target_rate_hz"]) != 5
            or registered_validation_traces != 12
            or registered_validation_paired_cells != 84
            or derived_validation_paired_cells != registered_validation_paired_cells
            or int(cfg.network["validation_static_min_independent_groups"]) != 15
            or int(cfg.benchmark["temporal_sampling"]["minimum_temporal_groups"])
            != 30
            or int(cfg.benchmark["temporal_sampling"]["minimum_group_trace_cells"])
            != 500
            or int(cfg.evaluation["mission_priority_positive_units"]) != 140
            or int(cfg.evaluation["mission_priority_negative_units"]) != 73
        ):
            raise StudyGateError(
                "The inherited v2r7 temporal-feasibility amendment contract changed"
            )
        print(
            "V2R13_TEMPORAL_FEASIBILITY_CONTRACT_VERIFIED",
            "validation_sequences=7",
            "validation_traces=12",
            "validation_paired_cells=84",
            "test_sequences=15",
            "amendment_sha256=",
            temporal_feasibility_amendment["record_sha256"],
        )
        validation_provenance = verify_visdrone_source_provenance(
            VISDRONE_ROOT, full_integrity=True,
            components=DEVELOPMENT_COMPONENTS,
        )
        media_partition_audit = verify_visdrone_media_partition_audit(
            MEDIA_PARTITION_AUDIT_RECEIPT,
            expected_inventory_sha256=validation_provenance["inventory_sha256"],
        )
        for budget_name in ("primary_wire_budget_bytes_per_s", "primary_energy_budget_j_per_s"):
            try:
                budget_value = float(cfg.evaluation[budget_name])
            except (KeyError, TypeError, ValueError) as exc:
                raise StudyGateError(f"Frozen numeric budget is missing: {budget_name}") from exc
            if not np.isfinite(budget_value) or budget_value <= 0:
                raise StudyGateError(f"Frozen numeric budget must be positive: {budget_name}")
        h2_budget_tolerance = float(
            cfg.evaluation["h2_matched_budget_relative_tolerance"]
        )
        if not np.isclose(h2_budget_tolerance, 0.05, rtol=0.0, atol=1e-12):
            raise StudyGateError("H2 matched-budget relative tolerance must remain frozen at 5%")
        if cfg.evaluation["energy_claim_scope"] != (
            "combined_sender_and_ground_gpu_board_nvml_compute_only_excludes_radio_system"
        ):
            raise StudyGateError("Registered combined compute-energy scope changed")
        h4_high_confidence_threshold = float(
            cfg.evaluation["h4_high_confidence_threshold"]
        )
        if not np.isclose(
            h4_high_confidence_threshold, 0.80, rtol=0.0, atol=1e-12,
        ):
            raise StudyGateError(
                "H4 high-confidence semantic threshold must remain frozen at 0.80"
            )
        registered_input_size = tuple(int(value) for value in cfg.models.get(
            "input_size", (768, 768)
        ))
        if registered_input_size != (768, 768):
            raise StudyGateError("All registered VLMs must receive the same 768x768 input")
        if cfg.models.get("input_policy") != (
            "common_aspect_preserving_letterbox_then_pinned_native_processor_v1"
        ):
            raise StudyGateError("Registered letterbox/native-processor input policy changed")
        registered_selection = cfg.evaluation["model_selection"]
        if registered_selection != {
            "estimand": "group_equal_critical_action_plus_object_exact_v2",
            "weights": {
                "critical_action_accuracy": 0.50,
                "objects_exact_accuracy": 0.50,
            },
            "tie_break": [
                "selection_score_desc",
                "critical_action_accuracy_desc",
                "objects_exact_accuracy_desc",
                "full_path_latency_p95_ms_asc",
                "model_id_asc",
            ],
        }:
            raise StudyGateError("Pre-validation v2 model-selection amendment changed")
        split_manifest = pd.read_csv(development_bundle.split_manifest, compression="gzip")
        split_audit = audit_publisher_splits(split_manifest)
        if not split_audit["passed"].astype(bool).all():
            raise StudyGateError("Frozen publisher split audit failed")
        analysis_manifest = development_bundle.load_analysis_manifest()
        registered_frames = {}
        for split_name in ("train", "calibration", "validation"):
            sample_dir = DEVELOPMENT_SAMPLE_ROOT / split_name
            verify_artifact_hash_manifest(sample_dir / "REGISTERED_ANALYSIS_SAMPLE_ARTIFACTS.json")
            selected, selected_audit = select_registered_analysis_sample_from_config(
                analysis_manifest, benchmark=cfg.benchmark,
                split=split_name, seed=int(cfg.seed),
            )
            frozen_audit = pd.read_csv(sample_dir / "registered_analysis_sampling_audit.csv")
            if set(selected_audit["selection_sha256"].astype(str)) != set(frozen_audit["selection_sha256"].astype(str)):
                raise StudyGateError(f"Frozen registered {split_name} sample differs from recomputation")
            registered_frames[split_name] = selected
        perception_manifest = pd.concat(
            [registered_frames["calibration"], registered_frames["validation"]],
            ignore_index=True,
        )
        validation_manifest = registered_frames["validation"].copy()
        validation_temporal_selected, validation_temporal_audit = (
            select_registered_temporal_sample_from_config(
                analysis_manifest, benchmark=cfg.benchmark,
                split="validation", seed=int(cfg.seed),
            )
        )
        validation_capacity_rows = validation_temporal_audit.loc[
            validation_temporal_audit["gate"].astype(str).eq(
                "eligible_sequence_capacity"
            )
        ]
        if (
            len(validation_capacity_rows) != 1
            or json.loads(str(validation_capacity_rows.iloc[0]["observed"])) != 7
        ):
            raise StudyGateError(
                "The v2r13 validation temporal-capacity audit is not exactly seven"
            )
        print(
            "V2R13_TEMPORAL_ELIGIBLE_CAPACITY_VERIFIED",
            "eligible_validation_sequences=7",
        )
        if VALIDATION_TEMPORAL_SAMPLE_DIR.exists():
            validation_temporal_paths = {
                "sample": VALIDATION_TEMPORAL_SAMPLE_DIR
                / "registered_temporal_sample.csv.gz",
                "audit": VALIDATION_TEMPORAL_SAMPLE_DIR
                / "registered_temporal_sampling_audit.csv",
                "artifact_manifest": VALIDATION_TEMPORAL_SAMPLE_DIR
                / "REGISTERED_TEMPORAL_SAMPLE_ARTIFACTS.json",
            }
            verify_artifact_hash_manifest(
                validation_temporal_paths["artifact_manifest"]
            )
            frozen_temporal_audit = pd.read_csv(
                validation_temporal_paths["audit"], keep_default_na=False,
            )
            frozen_temporal_sample = read_analysis_manifest(
                validation_temporal_paths["sample"]
            )
            try:
                pd.testing.assert_frame_equal(
                    frozen_temporal_sample.reset_index(drop=True),
                    validation_temporal_selected.reset_index(drop=True),
                    check_dtype=False,
                    check_exact=True,
                )
                pd.testing.assert_frame_equal(
                    frozen_temporal_audit.reset_index(drop=True),
                    validation_temporal_audit.reset_index(drop=True),
                    check_dtype=False,
                    check_exact=True,
                )
            except AssertionError as exc:
                raise StudyGateError(
                    "Frozen registered validation temporal sample/audit differs "
                    "from fresh deterministic recomputation"
                ) from exc
        else:
            validation_temporal_paths = write_registered_temporal_sample(
                validation_temporal_selected, validation_temporal_audit,
                VALIDATION_TEMPORAL_SAMPLE_DIR,
            )
        validation_temporal_manifest = read_analysis_manifest(
            validation_temporal_paths["sample"]
        )
        if set(
            validation_temporal_manifest["temporal_selection_sha256"].astype(str)
        ) != set(validation_temporal_audit["selection_sha256"].astype(str)):
            raise StudyGateError(
                "Persisted validation temporal sample is not bound to its audit"
            )
        display(validation_temporal_audit)
        all_traces = development_bundle.load_trace_manifest()
        validation_traces = all_traces[all_traces["split"].eq("validation")].copy()
        test_trace_design = all_traces[all_traces["split"].eq("test")].copy()
        if validation_traces.empty:
            raise StudyGateError("Registered validation traces are missing")
        validation_trace_realizations = int(
            validation_traces["trace_id"].astype(str).nunique()
        )
        test_trace_design_realizations = int(
            test_trace_design["trace_id"].astype(str).nunique()
        )
        if validation_trace_realizations != int(cfg.network["validation_trace_realizations"]):
            raise StudyGateError("Validation trace coverage differs from its frozen design")
        if test_trace_design_realizations != int(cfg.network["held_out_trace_realizations"]):
            raise StudyGateError(
                "Frozen outcome-blind test trace design must contain exactly "
                "the registered trace count"
            )
        if set(test_trace_design["family"].astype(str)) != set(cfg.network["held_out_trace_families"]):
            raise StudyGateError("Frozen untouched-test trace families differ from configuration")
        untouched_component_count = int(
            component_registry["study_partition"].astype(str).eq("untouched_test").sum()
        )
        configured_test_visual_group_cap = int(
            cfg.benchmark["analysis_sampling"]["max_groups_per_component"]["test"]
        ) * untouched_component_count
        if (
            configured_test_visual_group_cap != 30
            or configured_test_visual_group_cap
            != int(cfg.evaluation["natural_test_units"])
        ):
            raise StudyGateError("Bounded test sample must freeze exactly 30 natural groups")
        validation_temporal_groups = int(
            validation_temporal_manifest["visual_group_id"].astype(str).nunique()
        )
        validation_paired_cells = (
            validation_temporal_groups * validation_trace_realizations
        )
        if validation_paired_cells != int(
            cfg.network["validation_fully_paired_scene_trace_cells"]
        ):
            raise StudyGateError(
                "Validation temporal paired scene-trace design is not exactly covered"
            )
        planned_test_temporal_groups = int(test_temporal_spec["sequences_per_split"])
        planned_test_temporal_cells = (
            planned_test_temporal_groups * test_trace_design_realizations
        )
        print(
            "V2R13_TEMPORAL_CARTESIAN_DESIGN_VERIFIED",
            "validation_sequences=",
            validation_temporal_groups,
            "validation_traces=",
            validation_trace_realizations,
            "validation_cells=",
            validation_paired_cells,
            "test_sequences=",
            planned_test_temporal_groups,
            "test_traces=",
            test_trace_design_realizations,
            "test_cells=",
            planned_test_temporal_cells,
        )
        if not (
            planned_test_temporal_groups
            < int(test_temporal_spec["minimum_temporal_groups"])
            and planned_test_temporal_cells
            < int(test_temporal_spec["minimum_group_trace_cells"])
        ):
            raise StudyGateError(
                "The frozen bounded MOT design no longer supports registered H2 status"
            )
        if not (
            planned_test_temporal_groups
            < int(cfg.evaluation["mission_priority_positive_units"])
            and planned_test_temporal_groups
            < int(cfg.evaluation["mission_priority_negative_units"])
        ):
            raise StudyGateError(
                "The frozen bounded temporal cohort no longer supports registered H3 status"
            )
        validation_inputs = {
            "train_manifest": registered_frames["train"],
            "perception_manifest": perception_manifest,
            "validation_manifest": validation_manifest,
            "temporal_manifest": validation_temporal_manifest,
            "temporal_sample_paths": validation_temporal_paths,
            # The validation runner consumes validation realizations and also
            # verifies the frozen, outcome-blind test trace design.  Preserve
            # the complete preregistered trace table after both split-specific
            # design checks above; passing only validation rows makes the
            # required test-design gate unexecutable.
            "traces": all_traces,
            "model_lock": MODEL_LOCK,
            "model_lock_derivation": MODEL_LOCK_DERIVATION,
            "detector_checkpoint": Path(cfg.models["detector"]["checkpoint"]),
        }
        print(
            "VALIDATION_INPUTS_VERIFIED",
            {name: len(frame) for name, frame in registered_frames.items()},
            "validation_trace_realizations=", validation_trace_realizations,
            "validation_paired_scene_trace_cells=", validation_paired_cells,
            "frozen_test_trace_design_realizations=", test_trace_design_realizations,
            "bounded_test_visual_groups=", configured_test_visual_group_cap,
            "validation_temporal_groups=", validation_temporal_groups,
            "planned_test_temporal_groups=", planned_test_temporal_groups,
            "planned_test_temporal_group_trace_cells=", planned_test_temporal_cells,
            "H2=STRUCTURALLY_NOT_ESTIMABLE",
            "H3=STRUCTURALLY_NOT_ESTIMABLE",
        )
    else:
        print("Validation input gate paused.")
    """, "validation-only"),
    md("""
    ## VLM go/no-go and Experiments 1–2 — L0–L5 receiver frontier

    Every VLM verifier receives the same 768×768 aspect-preserving letterboxed RGB
    image plus the same ordered detector proposals. Its pinned native processor
    then applies model-specific normalization and tokenization; the study does not
    assume identical internal visual-token grids. The bounded VLM output selects
    proposal IDs and visibility status; a deterministic host composer constructs
    the complete transmitted packet. The model lock records the exact repository
    commit, snapshot inventory, and processor/config bytes used by each backend.

    Experiment 1 measures complete preprocessing-to-decoding latency, GPU-board
    energy, memory, schema validity, semantic validity, composed action accuracy,
    exact composed objects, and calibrated confidence. The pre-validation v2
    selection estimand is group-equal critical-action plus object-exact accuracy;
    it applies to the composed detector+verifier+host-composer pipeline, not
    free-form VLM packet authorship. Model selection uses calibration/validation only.
    Experiment 2 measures actual wire bytes and native receiver outcomes for L0
    alert, L1/L2 semantics, L3 grounded semantic ROI, L4 INT8/INT4 split features,
    L5 images, and the registered H.264 clip baseline. Gold action is never an
    input to candidate generation or native receivers.
    """),
    code("""
    exp1_detail = exp1_summary = exp2_frontier = selection = None
    if PHASE == "validation" and EXECUTE_PHASE:
        _require_v2r13_preimport_source_gate()
        _require_exclusive_scientific_gpu("validation_experiments_1_2")
        phase12 = run_validation_perception_and_frontier(
            ctx=ctx, cfg=cfg, manifest=validation_inputs["perception_manifest"],
            model_lock=MODEL_LOCK, device=DEVICE,
            feature_head_lock=FEATURE_HEAD_LOCK,
            build_native_receiver_cache=BUILD_NATIVE_RECEIVER_CACHE,
            build_feature_cache=BUILD_FEATURE_CACHE,
            temporal_manifest=validation_inputs["temporal_manifest"],
        )
        exp1_detail = phase12["exp1_detail"]
        exp1_summary = phase12["exp1_summary"]
        exp2_frontier = phase12["exp2_frontier"]
        selection = phase12["selection"]
        selected_compact_model = (
            str(selection.get("selected_compact_model", ""))
            if isinstance(selection, dict) else ""
        )
        temporal_experiment1_model_ids = [
            "detector_only", selected_compact_model,
        ]
        if (
            not isinstance(selection, dict)
            or selected_compact_model
            not in set(REGISTERED_RUNTIME_SMOKE_MODEL_IDS[:2])
            or str(selection.get("best_compact_vlm", ""))
            != selected_compact_model
            or str(selection.get("selected_model", ""))
            not in {"detector_only", selected_compact_model}
            or selection.get("temporal_experiment1_model_ids")
            != temporal_experiment1_model_ids
            or type(selection.get("temporal_experiment1_model_count")) is not int
            or selection.get("temporal_experiment1_model_count") != 2
        ):
            raise StudyGateError(
                "Validation selection does not preserve the exact detector plus "
                "selected-compact temporal Experiment 1 routes"
            )
        if {"moondream_tokenizer", "moondream_tokenizer_sha256"}.intersection(
            selection
        ):
            raise StudyGateError(
                "Validation selection retained excluded-Moondream runtime state"
            )
        processor_provenance = pd.DataFrame([
            {
                "model_id": item["model_id"],
                "repository": item["repo"],
                "resolved_revision": item["resolved_revision"],
                "shared_letterboxed_rgb_input": "768x768",
                "native_processor_after_letterbox": True,
                "snapshot_inventory_sha256": item["inventory_sha256"],
                "processor_config_files": json.dumps(sorted(
                    str(file_item.get("path", file_item.get("relative_path", "")))
                    for file_item in item.get("files", [])
                    if any(
                        token in str(file_item.get("path", file_item.get("relative_path", ""))).lower()
                        for token in ("processor", "preprocessor", "tokenizer", "config")
                    )
                )),
            }
            for item in verified_model_lock["models"]
        ])
        display(exp1_summary)
        display(processor_provenance)
        display(exp2_frontier.groupby(["mode", "variant"], as_index=False)["payload_bytes"].mean())
        if not bool(selection.get("vlm_claim_enabled", False)):
            print("VLM GO/NO-GO: NO-GO. Detector fallback continues; VLM claims are disabled.")
        else:
            print("VLM GO/NO-GO: GO", selection["selected_model"])
    else:
        print("Validation Experiments 1–2 paused.")
    """, "validation-only"),
    md("""
    ## Experiments 3–5 — transport, adaptive control, and safety endpoints

    Experiment 3 estimates performance only over the fixed registered link grid
    using five stochastic repeats and actual MTU-safe protocol bytes. Its paired
    outcomes are averaged over grid cells and repeats within each visual group
    before the visual-group bootstrap; thousands of grid cells are not labeled as
    independent traces, and H1 does not generalize beyond this registered grid.
    Experiment 4 evaluates fixed and
    adaptive policies on held-out trace families using only causal controller
    state. Experiment 5 reports mission-priority positive and negative cohorts,
    false alerts, misses, selective risk, and uncertainty escalation. No OOD claim
    is made in this VisDrone-only scope. Validation power is a nonblocking design
    sensitivity diagnostic that reports the assumed effect, null margin, observed
    discordance, and simulated power for each baseline; it is never relabeled as
    achieved 80% power when the observed diagnostic is lower. H2 pairwise
    wire/combined-compute matching is likewise a claim-estimability audit, not a
    completion gate for the other experiments: every registered comparator must
    be audited, while any unmatched family is frozen as `NOT_ESTIMABLE` rather
    than repaired by post-validation traffic or energy padding.
    """),
    code("""
    exp3_detail = exp3_summary = validation_resource_summary = None
    exp4_summary = exp5_summary = None
    if PHASE == "validation" and EXECUTE_PHASE:
        _require_v2r13_preimport_source_gate()
        _require_exclusive_scientific_gpu("validation_experiments_3_5")
        phase345 = run_validation_network_and_controller(
            ctx=ctx, cfg=cfg,
            manifest=validation_inputs["validation_manifest"],
            traces=validation_inputs["traces"],
            exp2_frontier=exp2_frontier,
            selection=selection,
            temporal_manifest=validation_inputs["temporal_manifest"],
        )
        exp3_detail, exp3_summary = phase345["exp3_detail"], phase345["exp3_summary"]
        validation_resource_summary = phase345["resource_summary"]
        exp4_summary, exp5_summary = phase345["exp4_summary"], phase345["exp5_summary"]
        validation_h2_matched_budget = pd.read_csv(
            selection["gate_artifacts"]["validation_h2_matched_budget"],
            compression="gzip",
        )
        validation_power_diagnostic = pd.read_csv(
            selection["gate_artifacts"]["validation_power_by_baseline"],
            compression="gzip",
        )
        required_h2_audit_columns = {
            "reference", "wire_relative_delta", "energy_relative_delta",
            "wire_match_pass", "energy_match_pass", "common_cap_pass", "passed",
        }
        missing_h2_audit_columns = required_h2_audit_columns.difference(
            validation_h2_matched_budget.columns
        )
        if missing_h2_audit_columns:
            raise StudyGateError(
                "H2 pairwise audit lacks registered columns: "
                f"{sorted(missing_h2_audit_columns)}"
            )
        required_h2_baselines = tuple(map(
            str, selection["policy_spec"]["required_h2_baselines"]
        ))
        observed_h2_baselines = tuple(
            validation_h2_matched_budget["reference"].astype(str)
        )
        if (
            validation_h2_matched_budget["reference"].astype(str).duplicated().any()
            or set(observed_h2_baselines) != set(required_h2_baselines)
            or len(observed_h2_baselines) != len(required_h2_baselines)
        ):
            raise StudyGateError(
                "H2 pairwise audit does not cover the exact registered comparator family"
            )
        if not np.isfinite(validation_h2_matched_budget[[
            "wire_relative_delta", "energy_relative_delta",
        ]].to_numpy(float)).all():
            raise StudyGateError("H2 pairwise audit contains non-finite resource deltas")
        audited_h2_unmatched = sorted(
            validation_h2_matched_budget.loc[
                ~validation_h2_matched_budget["passed"].astype(bool), "reference"
            ].astype(str)
        )
        frozen_h2_unmatched = sorted(map(
            str, selection.get("h2_validation_unmatched_baselines", ())
        ))
        if audited_h2_unmatched != frozen_h2_unmatched:
            raise StudyGateError(
                "Frozen H2 unmatched comparator family differs from the audited artifact"
            )
        expected_h2_family_status = (
            "NOT_ESTIMABLE" if audited_h2_unmatched else "MATCHED"
        )
        if str(selection.get("h2_validation_family_status", "")) != expected_h2_family_status:
            raise StudyGateError("Frozen H2 family status differs from its pairwise audit")
        validation_gate_table = pd.read_csv(
            selection["network_artifacts"]["validation_gates"],
            compression="gzip",
        )
        h2_registration_gate = validation_gate_table.loc[
            validation_gate_table["gate"].astype(str).eq(
                "h2_comparator_and_resource_audits_registered"
            )
        ]
        if len(h2_registration_gate) != 1 or not bool(
            h2_registration_gate["passed"].astype(bool).iloc[0]
        ):
            raise StudyGateError(
                "H2 comparator/resource audit registration is incomplete"
            )
        if validation_power_diagnostic["blocking_protocol_gate"].astype(bool).any():
            raise StudyGateError("Validation power must remain a nonblocking diagnostic")
        display(exp3_summary)
        display(validation_resource_summary)
        display(validation_h2_matched_budget)
        if audited_h2_unmatched:
            print(
                "H2 validation family: NOT_ESTIMABLE; unmatched registered baselines:",
                audited_h2_unmatched,
            )
        else:
            print("H2 validation family: MATCHED")
        display(validation_power_diagnostic)
        display(exp4_summary)
        display(exp5_summary)
    else:
        print("Validation Experiments 3–5 paused.")
    """, "validation-only"),
    md("""
    ## Experiments 6–7 — protocol robustness and hardware readiness

    Experiment 6 executes every registered receiver/transport fault and the three
    frozen state strategies under every seed. Experiment 7 records runtime and
    external-evidence readiness only. H5 is frozen as
    `NOT_APPLICABLE_NO_TARGET_EVIDENCE` for this paper. Any target-hardware claim
    requires a separately preregistered study established before its evidence is
    collected.
    """),
    code("""
    exp6_detail = exp6_summary = exp6_strategy_detail = exp6_strategy_summary = None
    exp7_readiness = exp7_hardware = None
    if PHASE == "validation" and EXECUTE_PHASE:
        _require_v2r13_preimport_source_gate()
        _require_exclusive_scientific_gpu("validation_experiments_6_7")
        exp6_contract = experiment6_receiver_robustness()
        if not exp6_contract["passed"].astype(bool).all():
            raise StudyGateError("Receiver-state unit contract failed")
        exp6_detail, exp6_summary = experiment6_registered_campaign(
            seeds=tuple(cfg.seeds), scientific=False,
            registered_cases=tuple(EXPERIMENT6_REGISTERED_CASES),
        )
        exp6_strategy_detail, exp6_strategy_summary = experiment6_state_strategy_comparison(
            seeds=tuple(cfg.seeds), scientific=False,
            registered_strategies=tuple(EXPERIMENT6_REGISTERED_STATE_STRATEGIES),
            periodic_keyframe_interval=3,
        )
        atomic_write_frame(RESULTS_DIR / "validation_exp6_fault_detail.csv.gz", exp6_detail)
        atomic_write_frame(RESULTS_DIR / "validation_exp6_fault_summary.csv.gz", exp6_summary)
        atomic_write_frame(RESULTS_DIR / "validation_exp6_state_detail.csv.gz", exp6_strategy_detail)
        atomic_write_frame(RESULTS_DIR / "validation_exp6_state_summary.csv.gz", exp6_strategy_summary)
        exp7_readiness, exp7_hardware = experiment7_readiness(
            PROJECT_ROOT, cfg.paths.cache_root, DEVICE, cfg.profile.value,
        )
        if exp7_hardware is None:
            exp7_hardware = pd.DataFrame([{
                "hypothesis": "H5",
                "status": "NOT_APPLICABLE_NO_TARGET_EVIDENCE",
                "eligible": False,
                "reason": (
                    "No separately registered identity-bound Jetson-class raw trials"
                ),
            }])
        atomic_write_frame(RESULTS_DIR / "validation_exp7_readiness.csv.gz", exp7_readiness)
        atomic_write_frame(RESULTS_DIR / "validation_exp7_hardware.csv.gz", exp7_hardware)
        display(exp6_summary)
        display(exp6_strategy_summary)
        display(exp7_readiness)
    else:
        print("Validation Experiments 6–7 paused.")
    """, "validation-only"),
    md("""
    ## H1–H5 validation claim screen and immutable protocol

    Freeze only after every validation gate passes and the selected model,
    calibration, controller, budgets, candidate variants, seeds, prompt, source
    receipts, split, trace design, receiver caches, and executable notebook have
    been reviewed. Freezing does not open the test set.
    """),
    code("""
    validation_claim_ledger = None
    if PHASE == "validation" and EXECUTE_PHASE:
        _require_v2r13_preimport_source_gate()
        _require_exclusive_scientific_gpu("validation_claim_and_freeze")
        validation_gates_passed = bool(
            selection is not None
            and selection.get("validation_gates_passed") is True
            and exp1_detail is not None and not exp1_detail.empty
            and exp2_frontier is not None and not exp2_frontier.empty
            and exp3_summary is not None and not exp3_summary.empty
            and exp4_summary is not None and not exp4_summary.empty
            and exp5_summary is not None and not exp5_summary.empty
            and exp6_summary is not None and not exp6_summary.empty
            and exp6_strategy_summary is not None and not exp6_strategy_summary.empty
            and exp7_readiness is not None and not exp7_readiness.empty
            and not exp7_readiness["status"].astype(str).eq("FAIL").any()
        )
        validation_selected_vlm_used = bool(
            selection is not None
            and registered_selected_vlm_claim_gate(selection)
        )
        validation_detector_inference = bool(
            exp1_detail is not None
            and exp1_detail.get("model_id", pd.Series(dtype=str)).astype(str).eq(
                "detector_only"
            ).any()
        )
        validation_selected_vlm_inference = bool(
            validation_selected_vlm_used
            and exp1_detail is not None
            and exp1_detail.get("model_id", pd.Series(dtype=str)).astype(str).eq(
                str(selection.get("selected_model"))
            ).any()
        )
        validation_claim_ledger = build_readiness_claim_ledger(
            phase="validation",
            evidence={
                "official_visdrone_receipt": True,
                "publisher_group_splits_passed": True,
                "real_detector_inference": validation_detector_inference,
                "real_vlm_inference": validation_selected_vlm_inference,
                "selected_vlm_inference": validation_selected_vlm_inference,
                "validation_gates_passed": validation_gates_passed,
                "protocol_frozen": PROTOCOL.is_file(),
                "one_time_test_complete": False,
                "paper_artifacts_hashed": False,
            },
            selected_vlm_used=validation_selected_vlm_used,
        )
        display(validation_claim_ledger)
        atomic_write_frame(RESULTS_DIR / "validation_claim_status.csv.gz", validation_claim_ledger)
        if FREEZE_PROTOCOL:
            if not validation_gates_passed:
                raise StudyGateError("Cannot freeze; one or more registered validation gates failed")
            selection_required = {
                "selected_model", "selected_compact_model",
                "temporal_experiment1_model_ids",
                "temporal_experiment1_model_count", "best_compact_vlm",
                "upper_bound_model", "calibration",
                "model_lock_sha256", "detector_revision",
                "detector_checkpoint_sha256", "feature_head_lock",
                "native_receiver_cache", "detector_native_receiver_cache",
                "registered_feature_cache", "policy_spec", "policy_spec_path",
                "candidate_profiles_path", "exp3_variant_spec",
                "h4_high_confidence_threshold",
                "validation_gates_passed",
            }
            selection_missing = sorted(selection_required.difference(selection))
            if selection_missing:
                raise StudyGateError(
                    f"Cannot freeze; validation selection is incomplete: {selection_missing}"
                )
            selected_compact_model = str(selection["selected_compact_model"])
            temporal_experiment1_model_ids = [
                "detector_only", selected_compact_model,
            ]
            if (
                selected_compact_model
                not in set(REGISTERED_RUNTIME_SMOKE_MODEL_IDS[:2])
                or str(selection["best_compact_vlm"]) != selected_compact_model
                or str(selection["selected_model"])
                not in {"detector_only", selected_compact_model}
                or selection["temporal_experiment1_model_ids"]
                != temporal_experiment1_model_ids
                or type(selection["temporal_experiment1_model_count"]) is not int
                or selection["temporal_experiment1_model_count"] != 2
            ):
                raise StudyGateError(
                    "Cannot freeze; selected compact or temporal Experiment 1 "
                    "identity is invalid"
                )
            if not np.isclose(
                float(selection["h4_high_confidence_threshold"]),
                h4_high_confidence_threshold, rtol=0.0, atol=1e-12,
            ):
                raise StudyGateError(
                    "Validation H4 threshold differs from the frozen configuration"
                )
            selection_path = atomic_write_json(
                RESULTS_DIR / "validation_selection.json", selection,
            )
            validation_selection_summary = pd.DataFrame(
                selection.get("selection_summary", [])
            )
            validation_calibration_gate = pd.DataFrame(
                selection.get("calibration_gate", [])
            )
            if (
                validation_selection_summary.empty
                or "model_id" not in validation_selection_summary
                or validation_calibration_gate.empty
                or "model_id" not in validation_calibration_gate
                or not set(validation_calibration_gate["model_id"].astype(str)).issubset(
                    set(validation_selection_summary["model_id"].astype(str))
                )
            ):
                raise StudyGateError(
                    "Validation model comparison/calibration evidence is malformed"
                )
            calibration_columns = {
                column: f"calibration_{column}"
                for column in validation_calibration_gate.columns
                if column != "model_id"
            }
            validation_calibration_gate = validation_calibration_gate.rename(
                columns=calibration_columns
            )
            validation_model_selection_gate = validation_selection_summary.merge(
                validation_calibration_gate, on="model_id", how="left",
                validate="one_to_one",
            )
            locked_model_metadata = {
                str(item["model_id"]): item
                for item in verified_model_lock.get("models", [])
                if isinstance(item, dict) and item.get("model_id")
            }
            calibration_eligible = {
                model_id: bool(item.get("calibration_eligible", True))
                for model_id, item in locked_model_metadata.items()
            }
            calibration_eligible["detector_only"] = True
            model_roles = {
                model_id: str(item.get("role", "compact"))
                for model_id, item in locked_model_metadata.items()
            }
            model_roles["detector_only"] = "detector_comparator"
            validation_model_selection_gate["calibration_eligible"] = (
                validation_model_selection_gate["model_id"].astype(str).map(
                    calibration_eligible
                ).fillna(False).astype(bool)
            )
            validation_model_selection_gate["model_role"] = (
                validation_model_selection_gate["model_id"].astype(str).map(
                    model_roles
                ).fillna("unregistered")
            )
            raw_calibration_pass = validation_model_selection_gate.get(
                "calibration_passed", pd.Series(
                    pd.NA, index=validation_model_selection_gate.index,
                    dtype="boolean",
                ),
            ).astype("boolean")
            validation_model_selection_gate["calibration_gate_pass"] = (
                raw_calibration_pass
            )
            validation_model_selection_gate["calibration_status"] = np.where(
                raw_calibration_pass.notna(),
                np.where(raw_calibration_pass.fillna(False), "PASS", "FAIL"),
                np.where(
                    validation_model_selection_gate["calibration_eligible"],
                    "NOT_AVAILABLE", "INELIGIBLE_NOT_AVAILABLE",
                ),
            )
            validation_model_selection_gate["selected_for_confirmatory"] = (
                validation_model_selection_gate["model_id"].astype(str).eq(
                    str(selection["selected_model"])
                )
            )
            validation_model_selection_gate["best_compact_vlm"] = str(
                selection.get("best_compact_vlm", "")
            )
            validation_model_selection_gate["selected_compact_model"] = (
                selected_compact_model
            )
            validation_model_selection_gate[
                "selected_for_temporal_experiment1"
            ] = validation_model_selection_gate["model_id"].astype(str).isin(
                temporal_experiment1_model_ids
            )
            validation_model_selection_gate[
                "temporal_experiment1_model_ids"
            ] = json.dumps(temporal_experiment1_model_ids, separators=(",", ":"))
            validation_model_selection_gate[
                "temporal_experiment1_model_count"
            ] = 2
            validation_model_selection_gate["upper_bound_model"] = str(
                selection["upper_bound_model"]
            )
            validation_model_selection_gate["vlm_claim_enabled"] = bool(
                selection.get("vlm_claim_enabled", False)
            )
            validation_model_selection_gate["eligible_compact_models"] = json.dumps(
                list(map(str, selection.get("eligible_compact_models", ()))),
                separators=(",", ":"),
            )
            validation_model_selection_gate["model_selection_estimand"] = str(
                selection["model_selection"]["estimand"]
            )
            validation_model_selection_gate[
                "calibration_blocking_for_confirmatory"
            ] = validation_model_selection_gate["model_id"].astype(str).isin(
                temporal_experiment1_model_ids
            )
            validation_model_selection_gate["evidence_partition"] = (
                "DEVELOPMENT_VALIDATION_MODEL_SELECTION_ONLY"
            )
            blocking_calibration = validation_model_selection_gate.loc[
                validation_model_selection_gate[
                    "calibration_blocking_for_confirmatory"
                ].astype(bool)
            ]
            if (
                blocking_calibration.empty
                or blocking_calibration["calibration_gate_pass"].isna().any()
                or not blocking_calibration["calibration_gate_pass"].astype(bool).all()
            ):
                raise StudyGateError(
                    "Detector/selected-compact calibration gate is incomplete or failed"
                )
            validation_model_selection_gate_path = atomic_write_frame(
                RESULTS_DIR / "validation_model_selection_gate.csv.gz",
                validation_model_selection_gate,
            )
            head_lock_payload = json.loads(FEATURE_HEAD_LOCK.read_text(encoding="utf-8"))
            detector_plan_payload = json.loads(DETECTOR_PLAN.read_text(encoding="utf-8"))
            required = {
                "source_receipt": source_receipt,
                "source_hash_inventory": source_inventory,
                "split_manifest": FROZEN_SPLITS,
                "development_analysis_manifest": DEVELOPMENT_ANALYSIS,
                "registered_trace_manifest": REGISTERED_TRACES,
                "preparation_artifact_manifest": DEVELOPMENT_BUNDLE_DIR / "PREPARATION_ARTIFACTS.json",
                "preparation_receipt": DEVELOPMENT_BUNDLE_DIR / "PREPARED_VISDRONE_BUNDLE.json",
                "media_partition_audit_receipt": MEDIA_PARTITION_AUDIT_RECEIPT,
                "media_partition_hash_audit": MEDIA_PARTITION_HASH_AUDIT,
                "test_group_capacity_audit": TEST_GROUP_CAPACITY_AUDIT,
                "registered_train_sample": DEVELOPMENT_SAMPLE_ROOT / "train" / "registered_analysis_sample.csv.gz",
                "registered_train_sample_audit": DEVELOPMENT_SAMPLE_ROOT / "train" / "registered_analysis_sampling_audit.csv",
                "registered_train_sample_artifacts": DEVELOPMENT_SAMPLE_ROOT / "train" / "REGISTERED_ANALYSIS_SAMPLE_ARTIFACTS.json",
                "registered_calibration_sample": DEVELOPMENT_SAMPLE_ROOT / "calibration" / "registered_analysis_sample.csv.gz",
                "registered_calibration_sample_audit": DEVELOPMENT_SAMPLE_ROOT / "calibration" / "registered_analysis_sampling_audit.csv",
                "registered_calibration_sample_artifacts": DEVELOPMENT_SAMPLE_ROOT / "calibration" / "REGISTERED_ANALYSIS_SAMPLE_ARTIFACTS.json",
                "registered_validation_sample": DEVELOPMENT_SAMPLE_ROOT / "validation" / "registered_analysis_sample.csv.gz",
                "registered_validation_sample_audit": DEVELOPMENT_SAMPLE_ROOT / "validation" / "registered_analysis_sampling_audit.csv",
                "registered_validation_sample_artifacts": DEVELOPMENT_SAMPLE_ROOT / "validation" / "REGISTERED_ANALYSIS_SAMPLE_ARTIFACTS.json",
                "registered_validation_temporal_sample": validation_inputs[
                    "temporal_sample_paths"
                ]["sample"],
                "registered_validation_temporal_sample_audit": validation_inputs[
                    "temporal_sample_paths"
                ]["audit"],
                "registered_validation_temporal_sample_artifacts": validation_inputs[
                    "temporal_sample_paths"
                ]["artifact_manifest"],
                "source_model_lock": SOURCE_MODEL_LOCK,
                "model_lock": MODEL_LOCK,
                "model_lock_derivation": MODEL_LOCK_DERIVATION,
                "model_set_amendment_record": MODEL_SET_AMENDMENT_RECORD,
                "temporal_feasibility_amendment_record": (
                    TEMPORAL_FEASIBILITY_AMENDMENT_RECORD
                ),
                "runtime_smoke_canary": SMOKE_CANARY,
                "model_runtime_smoke_input": MODEL_SMOKE_INPUT_RECEIPT,
                "model_runtime_smoke": MODEL_SMOKE,
                "reused_v1_frozen_artifacts_receipt": REUSED_ARTIFACTS_RECEIPT,
                "historical_detector_training_environment_fingerprint": (
                    DETECTOR_TRAINING_ENVIRONMENT_FINGERPRINT
                ),
                "historical_detector_training_pip_freeze": (
                    DETECTOR_TRAINING_PIP_FREEZE
                ),
                "inference_environment_fingerprint": (
                    INFERENCE_ENVIRONMENT_FINGERPRINT
                ),
                "inference_pip_freeze": INFERENCE_PIP_FREEZE,
                "inference_environment_seal": INFERENCE_ENVIRONMENT_SEAL,
                "trusted_code_cache_receipt": TRUSTED_CODE_CACHE_RECEIPT,
                "detector_checkpoint": Path(cfg.models["detector"]["checkpoint"]),
                "detector_training_receipt": configured_detector_receipt,
                "detector_training_plan": DETECTOR_PLAN,
                "detector_base_checkpoint": Path(detector_plan_payload["base_model_path"]),
                "detector_dataset_yaml": Path(detector_plan_payload["dataset_yaml"]),
                "detector_conversion_manifest": Path(detector_plan_payload["conversion_manifest"]),
                "detector_conversion_summary": Path(detector_plan_payload["conversion_summary"]),
                "detector_study_source_manifest": Path(detector_plan_payload["study_source_manifest"]),
                "detector_leakage_audit": Path(detector_plan_payload["detector_leakage_audit"]),
                "validation_selection": selection_path,
                "validation_exp1_perception_summary": Path(
                    selection["artifacts"]["exp1_summary"]
                ),
                "validation_model_selection_gate": (
                    validation_model_selection_gate_path
                ),
                "selected_native_receiver_cache": Path(selection["native_receiver_cache"]),
                "detector_native_receiver_cache": Path(selection["detector_native_receiver_cache"]),
                "registered_feature_cache": Path(selection["registered_feature_cache"]),
                "feature_head_lock": FEATURE_HEAD_LOCK,
                "feature_head_weights": Path(head_lock_payload["weights_path"]),
                "feature_encoder_receipt": FEATURE_ENCODER_RECEIPT,
                "feature_encoder_checkpoint": Path(head_lock_payload["encoder_checkpoint_path"]),
                "policy_spec": Path(selection["policy_spec_path"]),
                "candidate_profiles": Path(selection["candidate_profiles_path"]),
                "exp3_variant_spec": Path(selection["exp3_variant_spec"]),
                "validation_claim_status": RESULTS_DIR / "validation_claim_status.csv.gz",
                "validation_exp6_fault_detail": RESULTS_DIR / "validation_exp6_fault_detail.csv.gz",
                "validation_exp6_fault_summary": RESULTS_DIR / "validation_exp6_fault_summary.csv.gz",
                "validation_exp6_state_detail": RESULTS_DIR / "validation_exp6_state_detail.csv.gz",
                "validation_exp6_state_summary": RESULTS_DIR / "validation_exp6_state_summary.csv.gz",
                "validation_exp7_readiness": RESULTS_DIR / "validation_exp7_readiness.csv.gz",
                "validation_exp7_hardware": RESULTS_DIR / "validation_exp7_hardware.csv.gz",
            "proposal_decision_prompt": VISDRONE_PROPOSAL_DECISION_PROMPT_PATH,
            }
            for group_name in ("artifacts", "network_artifacts", "gate_artifacts"):
                group = selection.get(group_name, {})
                if not isinstance(group, dict):
                    raise StudyGateError(f"Validation selection {group_name} must be a mapping")
                for artifact_name, artifact_path in sorted(group.items()):
                    required[f"selection_{group_name}_{artifact_name}"] = Path(artifact_path)
            missing = [name for name, path in required.items() if not Path(path).is_file()]
            if missing:
                raise StudyGateError(f"Cannot freeze; required artifacts are missing: {missing}")
            decisions = {
                "study_scope": cfg.study_scope.value,
                "truth_mode": cfg.truth_mode.value,
                "selected_model": selection["selected_model"],
                "selected_compact_model": selected_compact_model,
                "temporal_experiment1_model_ids": temporal_experiment1_model_ids,
                "temporal_experiment1_model_count": 2,
                "vlm_claim_enabled": bool(selection.get("vlm_claim_enabled", False)),
                "registered_model_ids": list(REGISTERED_RUNTIME_SMOKE_MODEL_IDS),
                "excluded_model_ids": ["moondream2"],
                "model_set_amendment_id": MODEL_SET_AMENDMENT_ID,
                "model_set_exclusion_reason": MODEL_SET_EXCLUSION_REASON,
                "model_set_amendment": _verified_model_lock_derivation(),
                "registered_budgets": {
                    "wire_bytes_per_s": cfg.evaluation["primary_wire_budget_bytes_per_s"],
                    "gpu_board_energy_j_per_s": cfg.evaluation["primary_energy_budget_j_per_s"],
                    "h2_pairwise_relative_tolerance": cfg.evaluation[
                        "h2_matched_budget_relative_tolerance"
                    ],
                    "energy_claim_scope": cfg.evaluation["energy_claim_scope"],
                    "reported_components": [
                        "sender_compute_energy_j_per_s",
                        "ground_receiver_compute_energy_j_per_s",
                    ],
                },
                "h4_high_confidence_threshold": h4_high_confidence_threshold,
                "seeds": list(cfg.seeds),
                "policies": list(cfg.controller["policies"]),
                "analysis_sampling": dict(cfg.benchmark["analysis_sampling"]),
                "temporal_sampling": dict(cfg.benchmark["temporal_sampling"]),
                "temporal_feasibility_amendment": dict(
                    temporal_feasibility_amendment
                ),
                "sample_minima": {
                    "validation_fully_paired_scene_trace_cells": int(
                        cfg.network["validation_fully_paired_scene_trace_cells"]
                    ),
                    "validation_trace_realizations": int(
                        cfg.network["validation_trace_realizations"]
                    ),
                    "fully_paired_scene_trace_cells": int(
                        cfg.evaluation["fully_paired_scene_trace_cells"]
                    ),
                    "independent_visual_groups": int(
                        cfg.evaluation["natural_test_units"]
                    ),
                    "independent_traces": int(cfg.network["held_out_trace_realizations"]),
                    "mission_priority_positive_units": int(
                        cfg.evaluation["mission_priority_positive_units"]
                    ),
                    "mission_priority_negative_units": int(
                        cfg.evaluation["mission_priority_negative_units"]
                    ),
                    "bounded_test_visual_groups": configured_test_visual_group_cap,
                    "planned_test_temporal_visual_groups": planned_test_temporal_groups,
                    "planned_test_temporal_group_trace_cells": planned_test_temporal_cells,
                    "h2_structural_status": "NOT_ESTIMABLE",
                    "h3_structural_status": "NOT_ESTIMABLE",
                    "held_out_trace_realizations": int(cfg.network["held_out_trace_realizations"]),
                    "held_out_trace_families": len(cfg.network["held_out_trace_families"]),
                },
                "model_input_contract": {
                    "shared_letterboxed_rgb_size": [768, 768],
                    "letterbox": "aspect_preserving",
                    "semantic_contract": cfg.models["semantic_contract"],
                    "host_composer_contract": cfg.models["host_composer_contract"],
                    "architecture": cfg.models["semantic_architecture"],
                    "ordered_detector_proposals_supplied": True,
                    "vlm_role": "bounded_proposal_verifier",
                    "packet_author": "deterministic_host_composer",
                    "post_letterbox_processing": (
                        "pinned_native_processor_model_specific_normalization_and_tokenization"
                    ),
                    "identical_internal_visual_token_grid_assumed": False,
                },
                "inference_environment_fingerprint_sha256": hash_file(
                    INFERENCE_ENVIRONMENT_FINGERPRINT
                ),
                "inference_pip_freeze_sha256": hash_file(INFERENCE_PIP_FREEZE),
                "inference_environment_seal_sha256": hash_file(
                    INFERENCE_ENVIRONMENT_SEAL
                ),
                "trusted_code_cache_receipt_sha256": hash_file(
                    TRUSTED_CODE_CACHE_RECEIPT
                ),
                "historical_detector_training_environment_fingerprint_sha256": (
                    hash_file(DETECTOR_TRAINING_ENVIRONMENT_FINGERPRINT)
                ),
                "historical_detector_training_pip_freeze_sha256": hash_file(
                    DETECTOR_TRAINING_PIP_FREEZE
                ),
                "reused_v1_frozen_artifacts_receipt_sha256": hash_file(
                    REUSED_ARTIFACTS_RECEIPT
                ),
                "runtime_smoke_input_sha256": hash_file(
                    MODEL_SMOKE_INPUT_RECEIPT
                ),
                "model_lock_derivation_sha256": hash_file(
                    MODEL_LOCK_DERIVATION
                ),
                "detector_training_receipt_sha256": verified_detector_training[
                    "receipt_sha256"
                ],
                "experiment6_campaign": {
                    "version": EXPERIMENT6_CAMPAIGN_VERSION,
                    "cases": list(EXPERIMENT6_REGISTERED_CASES),
                    "seeds": list(cfg.seeds),
                },
                "experiment6_state_strategy_comparison": {
                    "version": EXPERIMENT6_STATE_STRATEGY_VERSION,
                    "strategies": list(EXPERIMENT6_REGISTERED_STATE_STRATEGIES),
                    "seeds": list(cfg.seeds),
                    "periodic_keyframe_interval": 3,
                    "fault": "drop_update_2_application_ack_once",
                },
                "artifact_manifest": frozen_file_manifest(required),
                "selection": selection,
                "required_dataset_provenance_artifacts": ["source_receipt", "source_hash_inventory"],
                "notebook_path": str(NOTEBOOK_PATH),
                "notebook_source_hash": notebook_source_hash(NOTEBOOK_PATH),
                "prohibited_claims": list(cfg.benchmark["prohibited_claims"]),
                "h5_applicable": False,
            }
            freeze_protocol(
                ctx, decisions=decisions,
                validation_summary={
                    "all_registered_gates_passed": validation_gates_passed,
                    "claim_ledger": str(RESULTS_DIR / "validation_claim_status.csv.gz"),
                    "selected_model": str(selection["selected_model"]),
                    "selected_compact_model": selected_compact_model,
                    "temporal_experiment1_model_ids": temporal_experiment1_model_ids,
                    "temporal_experiment1_model_count": 2,
                },
            )
            print("PROTOCOL_FROZEN", PROTOCOL, hash_file(PROTOCOL))
    else:
        print("Validation claim/freeze stage paused.")
    """, "validation-only"),
    code("""
    if PHASE == "validation":
        print("PHASE_BARRIER: validation complete; test remains locked in this Run All")
    """, "phase-barrier"),
    md("""
    # Phase 3 — one-time untouched test

    Before running this phase, archive the entire validation run, verify the frozen
    protocol hash, disable all network model access, and review the command with the
    research lead. Setting `AUTHORIZE_TEST_ONCE=True` creates a durable OPEN ledger.
    Any interruption consumes the run; resume is forbidden. A documented amendment
    must use a new run name and rerun every comparator.
    """),
    code("""
    test_outputs = None
    if PHASE == "test" and EXECUTE_PHASE:
        _require_v2r13_preimport_source_gate()
        if RESUME:
            raise StudyGateError("Set RESUME=False for the one-time untouched test")
        if not AUTHORIZE_TEST_ONCE:
            raise StudyGateError("Untouched test is locked; explicitly set AUTHORIZE_TEST_ONCE=True")
        protocol_preview = _decode_json_object_bytes(
            _read_regular_file_bytes(PROTOCOL, label="frozen protocol"),
            label="frozen protocol",
        )
        preview_decisions = protocol_preview.get("decisions")
        preview_selection = (
            preview_decisions.get("selection")
            if isinstance(preview_decisions, dict) else None
        )
        preview_validation_summary = protocol_preview.get("validation_summary")
        preview_sample_minima = (
            preview_decisions.get("sample_minima")
            if isinstance(preview_decisions, dict) else None
        )
        preview_selected_compact_model = (
            str(preview_selection.get("selected_compact_model", ""))
            if isinstance(preview_selection, dict) else ""
        )
        preview_temporal_experiment1_model_ids = [
            "detector_only", preview_selected_compact_model,
        ]
        if (
            not isinstance(preview_decisions, dict)
            or
            not isinstance(preview_selection, dict)
            or not isinstance(preview_validation_summary, dict)
            or preview_decisions.get("registered_model_ids")
            != list(REGISTERED_RUNTIME_SMOKE_MODEL_IDS)
            or preview_decisions.get("excluded_model_ids") != ["moondream2"]
            or preview_decisions.get("model_set_amendment_id")
            != MODEL_SET_AMENDMENT_ID
            or preview_decisions.get("model_set_exclusion_reason")
            != MODEL_SET_EXCLUSION_REASON
            or preview_decisions.get("model_set_amendment")
            != _verified_model_lock_derivation()
            or preview_decisions.get("temporal_feasibility_amendment")
            != temporal_feasibility_amendment
            or preview_decisions.get("temporal_sampling")
            != dict(cfg.benchmark["temporal_sampling"])
            or not isinstance(preview_sample_minima, dict)
            or preview_sample_minima.get(
                "validation_fully_paired_scene_trace_cells"
            ) != 84
            or preview_sample_minima.get("validation_trace_realizations") != 12
            or preview_sample_minima.get("planned_test_temporal_visual_groups")
            != 15
            or preview_sample_minima.get(
                "planned_test_temporal_group_trace_cells"
            ) != 450
            or preview_decisions.get("model_lock_derivation_sha256")
            != hash_file(MODEL_LOCK_DERIVATION)
            or preview_selected_compact_model
            not in set(REGISTERED_RUNTIME_SMOKE_MODEL_IDS[:2])
            or str(preview_selection.get("best_compact_vlm", ""))
            != preview_selected_compact_model
            or str(preview_selection.get("selected_model", ""))
            not in {"detector_only", preview_selected_compact_model}
            or preview_selection.get("temporal_experiment1_model_ids")
            != preview_temporal_experiment1_model_ids
            or type(preview_selection.get("temporal_experiment1_model_count"))
            is not int
            or preview_selection.get("temporal_experiment1_model_count") != 2
            or preview_decisions.get("selected_compact_model")
            != preview_selected_compact_model
            or preview_decisions.get("temporal_experiment1_model_ids")
            != preview_temporal_experiment1_model_ids
            or type(preview_decisions.get("temporal_experiment1_model_count"))
            is not int
            or preview_decisions.get("temporal_experiment1_model_count") != 2
            or preview_validation_summary.get("selected_model")
            != str(preview_selection.get("selected_model", ""))
            or preview_validation_summary.get("selected_compact_model")
            != preview_selected_compact_model
            or preview_validation_summary.get("temporal_experiment1_model_ids")
            != preview_temporal_experiment1_model_ids
            or type(preview_validation_summary.get(
                "temporal_experiment1_model_count"
            )) is not int
            or preview_validation_summary.get("temporal_experiment1_model_count")
            != 2
            or {"moondream_tokenizer", "moondream_tokenizer_sha256"}
            .intersection(preview_decisions)
            or {"moondream_tokenizer", "moondream_tokenizer_sha256"}
            .intersection(preview_selection)
        ):
            raise StudyGateError(
                "Frozen protocol differs from the inherited v2r6 model-set or "
                "inherited v2r7 temporal-feasibility amendment"
            )
        _require_exclusive_scientific_gpu("test_pre_open")
        preopen_scientific_environment = _verify_inference_environment_seal()
        preopen_reuse = _verify_or_create_reused_v1_artifacts_receipt(
            allow_create=False,
        )
        preopen_runtime = preopen_scientific_environment["runtime"]
        smoke_input = preopen_scientific_environment["smoke_input"]
        # Before OPEN, verify only the signed inventory envelope and development
        # bytes. Publisher test-dev pixels/labels are not opened or hashed.
        pretest_envelope = verify_visdrone_source_provenance(VISDRONE_ROOT)
        verify_visdrone_source_provenance(
            VISDRONE_ROOT, full_integrity=True,
            components=DEVELOPMENT_COMPONENTS,
        )
        verify_visdrone_media_partition_audit(
            MEDIA_PARTITION_AUDIT_RECEIPT,
            expected_inventory_sha256=pretest_envelope["inventory_sha256"],
        )
        print(
            "PRE_OPEN_INFERENCE_ENVIRONMENT_MATCH",
            preopen_runtime["environment_sha256"],
            preopen_runtime["pip_freeze_sha256"],
            "reuse_receipt_sha256=",
            preopen_reuse["receipt_sha256"],
            "scientific_seal_sha256=",
            preopen_scientific_environment["sha256"],
        )
        frozen = authorize_test_once(ctx, PROTOCOL)
        test_report_artifacts = []
        test_restricted_artifacts = []
        try:
            permit = load_untouched_test_permit(
                TEST_LEDGER,
                expected_protocol_hash=frozen["protocol_hash"],
                expected_config_hash=ctx.config_hash,
                expected_code_hash=ctx.code_hash,
            )
            test_provenance = verify_visdrone_source_provenance(
                VISDRONE_ROOT, full_integrity=True,
                components=UNTOUCHED_TEST_COMPONENTS,
            )
            verified_partition_audit = verify_visdrone_media_partition_audit(
                MEDIA_PARTITION_AUDIT_RECEIPT,
                expected_inventory_sha256=test_provenance["inventory_sha256"],
                test_permit=permit, require_test_permit=True,
            )
            test_source_integrity_path = atomic_write_json(
                RESULTS_DIR / "test_source_integrity.json",
                {
                    "test_source_provenance": test_provenance,
                    "media_partition_audit": verified_partition_audit,
                    "test_ledger_sha256": permit.ledger_sha256,
                },
            )
            test_report_artifacts.append(test_source_integrity_path)
            test_bundle = prepare_visdrone_bundle(
                VISDRONE_ROOT, TEST_BUNDLE_DIR,
                benchmark=cfg.benchmark, partition="test",
                seed=int(cfg.seed),
                calibration_fraction=float(cfg.split.ratios["calibration"]),
                minimum_test_groups=int(cfg.evaluation["natural_test_units"]),
                test_permit=permit, require_official_receipt=True,
            )
            test_restricted_artifacts.extend([
                test_bundle.source_manifest,
                test_bundle.split_manifest,
                test_bundle.source_truth,
                test_bundle.analysis_manifest,
                test_bundle.split_audit,
                test_bundle.trace_audit,
                test_bundle.annotation_exclusions,
                test_bundle.artifact_manifest,
                test_bundle.receipt,
            ])
            full_test_manifest = test_bundle.load_analysis_manifest()
            test_manifest, test_sampling_audit = select_registered_analysis_sample_from_config(
                full_test_manifest, benchmark=cfg.benchmark,
                split="test", seed=int(cfg.seed), test_permit=permit,
            )
            test_sample_paths = write_registered_analysis_sample(
                test_manifest, test_sampling_audit, TEST_SAMPLE_DIR,
            )
            test_restricted_artifacts.extend(test_sample_paths.values())
            display(test_sampling_audit)
            test_temporal_selected, test_temporal_audit = (
                select_registered_temporal_sample_from_config(
                    full_test_manifest, benchmark=cfg.benchmark,
                    split="test", seed=int(cfg.seed), test_permit=permit,
                )
            )
            test_temporal_paths = write_registered_temporal_sample(
                test_temporal_selected, test_temporal_audit,
                TEST_TEMPORAL_SAMPLE_DIR,
            )
            verify_artifact_hash_manifest(test_temporal_paths["artifact_manifest"])
            test_temporal_manifest = read_analysis_manifest(
                test_temporal_paths["sample"]
            )
            if set(
                test_temporal_manifest["test_ledger_sha256"].astype(str)
            ) != {permit.ledger_sha256}:
                raise StudyGateError(
                    "Persisted temporal test sample is not bound to the OPEN ledger"
                )
            test_restricted_artifacts.extend(test_temporal_paths.values())
            display(test_temporal_audit)
            development_bundle = load_prepared_visdrone_bundle(DEVELOPMENT_BUNDLE_DIR)
            all_traces = development_bundle.load_trace_manifest()
            test_traces = all_traces[all_traces["split"].eq("test")].copy()
            if test_traces.empty:
                raise StudyGateError("Registered untouched-test traces are missing")
            natural_units = int(test_manifest["visual_group_id"].astype(str).nunique())
            temporal_priority_positive_units = int(test_temporal_manifest.loc[
                test_temporal_manifest["critical"].astype(bool)
                & test_temporal_manifest["gold_action"].astype(str).eq("RESPOND"),
                "visual_group_id",
            ].astype(str).nunique())
            temporal_priority_negative_units = int(test_temporal_manifest.loc[
                ~test_temporal_manifest["critical"].astype(bool), "visual_group_id",
            ].astype(str).nunique())
            test_trace_realizations = int(test_traces["trace_id"].astype(str).nunique())
            test_trace_families = int(test_traces["family"].astype(str).nunique())
            planned_paired_cells = natural_units * test_trace_realizations
            temporal_visual_groups = int(
                test_temporal_manifest["visual_group_id"].astype(str).nunique()
            )
            temporal_group_trace_cells = (
                temporal_visual_groups * test_trace_realizations
            )
            temporal_observations_per_group = int(
                test_temporal_manifest.groupby("visual_group_id")[
                    "source_sample_id"
                ].nunique().min()
            )
            temporal_spec = registered_temporal_sampling_spec(cfg.benchmark, "test")
            h3_negative_minimum = int(cfg.evaluation["mission_priority_negative_units"])
            test_sample_gate = pd.DataFrame([
                {"gate": "planned_paired_visual_trace_cells", "observed": planned_paired_cells,
                 "required": int(cfg.evaluation["fully_paired_scene_trace_cells"]),
                 "required_for": "study_completion"},
                {"gate": "independent_visual_groups", "observed": natural_units,
                 "required": int(cfg.evaluation["natural_test_units"]),
                 "required_for": "study_completion"},
                {"gate": "held_out_trace_realizations", "observed": test_trace_realizations,
                 "required": int(cfg.network["held_out_trace_realizations"]),
                 "required_for": "study_completion"},
                {"gate": "held_out_trace_families", "observed": test_trace_families,
                 "required": len(cfg.network["held_out_trace_families"]),
                 "required_for": "study_completion"},
                {"gate": "temporal_mission_priority_positive_visual_groups",
                 "observed": temporal_priority_positive_units,
                 "required": int(cfg.evaluation["mission_priority_positive_units"]),
                 "required_for": "H3_estimability"},
                {"gate": "temporal_mission_priority_negative_visual_groups",
                 "observed": temporal_priority_negative_units,
                 "required": h3_negative_minimum,
                 "required_for": "H3_estimability"},
                {"gate": "temporal_visual_groups", "observed": temporal_visual_groups,
                 "required": int(temporal_spec["minimum_temporal_groups"]),
                 "required_for": "H2_estimability"},
                {"gate": "temporal_group_trace_cells", "observed": temporal_group_trace_cells,
                 "required": int(temporal_spec["minimum_group_trace_cells"]),
                 "required_for": "H2_estimability"},
                {"gate": "temporal_observations_per_group",
                 "observed": temporal_observations_per_group,
                 "required": int(temporal_spec["observations_per_sequence"]),
                 "required_for": "H2_estimability"},
                {"gate": "H3_group_trace_equal_false_alert_upper_95", "observed": np.nan,
                 "required": float(cfg.evaluation["false_alert_limit"]),
                 "required_for": "H3_confirmatory_safety"},
            ])
            minimum_rows = ~test_sample_gate["gate"].eq(
                "H3_group_trace_equal_false_alert_upper_95"
            )
            test_sample_gate["passed"] = False
            test_sample_gate.loc[minimum_rows, "passed"] = (
                test_sample_gate.loc[minimum_rows, "observed"]
                >= test_sample_gate.loc[minimum_rows, "required"]
            )
            test_sample_gate["status"] = np.where(
                test_sample_gate["gate"].eq("H3_group_trace_equal_false_alert_upper_95"),
                "DEFERRED_TO_CONFIRMATORY_TRIALS",
                np.where(test_sample_gate["passed"], "PASS", "NOT_ESTIMABLE"),
            )
            test_sample_gate_path = atomic_write_frame(
                RESULTS_DIR / "test_sample_gate_table.csv.gz", test_sample_gate,
            )
            test_cluster_manifest_path = atomic_write_frame(
                RESULTS_DIR / "test_sample_cluster_manifest_PRE_RUN.csv.gz",
                test_manifest[[
                    column for column in (
                        "sample_id", "split", "component", "visual_group_id",
                        "mission_profile", "source_uid", "sequence_id", "frame_index",
                    ) if column in test_manifest.columns
                ]].copy(),
            )
            test_report_artifacts.extend([
                test_sample_gate_path, test_cluster_manifest_path,
            ])
            display(test_sample_gate)
            completion_rows = test_sample_gate["required_for"].eq("study_completion")
            if not test_sample_gate.loc[completion_rows, "passed"].astype(bool).all():
                raise StudyGateError("Untouched-test completion design minima are not satisfied")
            _require_exclusive_scientific_gpu("test_confirmatory_execution")
            test_outputs = run_confirmatory_test_once(
                ctx=ctx, cfg=cfg, manifest=test_manifest, traces=test_traces,
                protocol=frozen, model_lock=MODEL_LOCK, device=DEVICE,
                temporal_manifest=test_temporal_manifest,
            )
            test_runtime_identity = test_outputs.get("runtime_identity")
            test_runtime_identity_path = Path(
                test_outputs.get("runtime_identity_path", "")
            ).expanduser()
            if (
                not isinstance(test_runtime_identity, dict)
                or {"moondream_tokenizer", "moondream_tokenizer_sha256"}
                .intersection(test_runtime_identity)
                or test_runtime_identity.get("model_lock") != {
                    "path": str(MODEL_LOCK.resolve()),
                    "sha256": hash_file(MODEL_LOCK),
                }
                or test_runtime_identity_path.is_symlink()
                or not test_runtime_identity_path.is_file()
                or _decode_json_object_bytes(
                    _read_regular_file_bytes(
                        test_runtime_identity_path,
                        label="test inference-runtime identity",
                    ),
                    label="test inference-runtime identity",
                ) != test_runtime_identity
            ):
                raise StudyGateError(
                    "Confirmatory runner changed the frozen inference identity"
                )
            restricted_raw_partitions = tuple(
                Path(path).expanduser().resolve()
                for path in test_outputs["restricted_raw_partition_artifacts"]
            )
            if (
                not restricted_raw_partitions
                or len(set(restricted_raw_partitions)) != len(restricted_raw_partitions)
                or any(not path.is_file() for path in restricted_raw_partitions)
            ):
                raise StudyGateError(
                    "Confirmatory runner returned an invalid raw-partition receipt set"
                )
            test_restricted_artifacts.extend(restricted_raw_partitions)
            result_paths = [
                *[Path(path) for path in test_outputs["result_artifacts"]],
                *test_report_artifacts,
            ]
            finalize_test_once(
                ctx, status="COMPLETE", result_artifacts=result_paths,
                restricted_artifacts=test_restricted_artifacts,
            )
            print("ONE_TIME_TEST_COMPLETE", len(test_manifest), TEST_LEDGER)
        except StudyGateError as exc:
            blocker = RESULTS_DIR / "test_blocked.json"
            atomic_write_json(blocker, {
                "status": "BLOCKED", "phase": "test",
                "error_type": type(exc).__name__, "message": str(exc),
            })
            finalize_test_once(
                ctx, status="BLOCKED",
                result_artifacts=[blocker, *test_report_artifacts],
                restricted_artifacts=test_restricted_artifacts,
            )
            raise
        except Exception as exc:
            failure = RESULTS_DIR / "test_failure.json"
            atomic_write_json(failure, {
                "status": "FAILED", "phase": "test",
                "error_type": type(exc).__name__, "message": str(exc),
            })
            finalize_test_once(
                ctx, status="FAILED",
                result_artifacts=[failure, *test_report_artifacts],
                restricted_artifacts=test_restricted_artifacts,
            )
            raise
    else:
        print("One-time test stage paused and locked.")
    """, "test-only"),
    code("""
    if PHASE == "test":
        print("PHASE_BARRIER: one-time test ends here; report cells do not execute in this Run All")
    """, "phase-barrier"),
    md("""
    # Phase 4 — read-only paper report

    Report mode consumes only the COMPLETE test ledger and its verified result
    hashes. It never opens VisDrone images, annotations, model checkpoints, or the
    one-time test permit. Tables and figures are regenerated from detailed frozen
    outputs and are marked scientific only when every applicable gate passes.
    The complete detector-comparator, Qwen3-VL-2B, SmolVLM2-2.2B, and
    Qwen3-VL-8B upper-bound comparison/calibration table is loaded separately
    from checksum-bound **DEVELOPMENT/VALIDATION** selection evidence;
    it is never mislabeled as an untouched-test calibration result. Untouched-test
    Experiment 1 remains a distinct selected-model evaluation.
    """),
    code("""
    report_frames = {}
    policy_reduction_frames = {}
    hypothesis_effects = claim_ledger = None
    scientific_evidence_verified = False
    hypothesis_statuses = {}
    if PHASE == "report" and EXECUTE_PHASE:
        _require_v2r13_preimport_source_gate()
        if FINAL_COMPLETION_RECEIPT.exists():
            verify_paper_completion_receipt(
                FINAL_COMPLETION_RECEIPT,
                expected_results_manifest=PAPER_RESULTS_MANIFEST,
                expected_claim_ledger=FINAL_CLAIM_LEDGER,
                expected_protocol=PROTOCOL,
                expected_test_ledger=TEST_LEDGER,
                expected_report_environment_receipt=REPORT_ENVIRONMENT_RECEIPT,
            )
            raise StudyGateError(
                "Paper report is already finalized and immutable; use a new RUN_NAME"
            )
        report_csv_contract = {
            # Compact perception/static outputs.
            "test_exp1_perception_detail.csv.gz": "test_exp1_perception_detail",
            "test_exp1_perception_summary.csv.gz": "test_exp1_perception_summary",
            "test_sample_cluster_manifest.csv.gz": "test_sample_cluster_manifest",
            "test_temporal_cluster_manifest.csv.gz": "test_temporal_cluster_manifest",
            "test_dataset_summary.csv.gz": "test_dataset_summary",
            "test_exp2_representation_frontier.csv.gz": "test_exp2_representation_frontier",
            "test_exp3_static_partition_index.csv.gz": "test_exp3_static_partition_index",
            "test_exp3_static_summary.csv.gz": "test_exp3_static_summary",
            # Disk-reduced controller/statistical inputs. Raw policy trials and
            # raw policy partitions are deliberately absent from this allowlist.
            "policy_pair_cells.csv.gz": "policy_pair_cells",
            "h3_negative_episode_cells.csv.gz": "h3_negative_episode_cells",
            "resource_summary.csv.gz": "resource_summary",
            "resource_rate_summary.csv.gz": "resource_rate_summary",
            "resource_episode_cells.csv.gz": "resource_episode_cells",
            "resource_cluster_cells.csv.gz": "resource_cluster_cells",
            "controller_summary.csv.gz": "controller_summary",
            "exp5_summary.csv.gz": "exp5_summary",
            "h4_matched_coverage_cells.csv.gz": "h4_matched_coverage_cells",
            "h4_matched_coverage_audit.csv.gz": "h4_matched_coverage_audit",
            "h4_per_trace_audit.csv.gz": "h4_per_trace_audit",
            "h4_trace_spill_index.csv.gz": "h4_trace_spill_index",
            "pairing_audit.csv.gz": "pairing_audit",
            "io_audit.csv.gz": "io_audit",
            # Frozen hypothesis/gate outputs and small protocol summaries.
            "test_budget_compliance.csv.gz": "test_budget_compliance",
            "confirmatory_gate_table.csv.gz": "confirmatory_gate_table",
            "test_h1_fixed_grid_group_aggregates.csv.gz": "test_h1_fixed_grid_group_aggregates",
            "test_h1_fixed_grid_coverage_audit.csv.gz": "test_h1_fixed_grid_coverage_audit",
            "h1_semantic_noninferiority.csv.gz": "h1_semantic_noninferiority",
            "test_h2_matched_budget.csv.gz": "test_h2_matched_budget",
            "h2_confirmatory_report.csv.gz": "h2_confirmatory_report",
            "h3_mission_awareness.csv.gz": "h3_mission_awareness",
            "h4_uncertainty_escalation.csv.gz": "h4_uncertainty_escalation",
            "test_receiver_robustness_detail.csv.gz": "test_receiver_robustness_detail",
            "test_receiver_robustness_summary.csv.gz": "test_receiver_robustness_summary",
            "test_receiver_state_strategy_detail.csv.gz": "test_receiver_state_strategy_detail",
            "test_receiver_state_strategy_summary.csv.gz": "test_receiver_state_strategy_summary",
            "hypothesis_status.csv.gz": "hypothesis_status",
            "final_completion_status.csv.gz": "final_completion_status",
            "test_episode_trace_coverage.csv.gz": "test_episode_trace_coverage",
            "test_dynamic_cartesian_coverage_audit.csv.gz": "test_dynamic_cartesian_coverage_audit",
        }
        report_nonframe_contract = {
            "policy_reduction_receipt.json",
            "test_policy_partition_index.csv",
        }
        forbidden_report_artifacts = {
            "test_policy_trials.csv.gz",
            "test_controller_behavior_detail.csv.gz",
            "test_exp5_risk_uncertainty_detail.csv.gz",
            "test_h4_matched_coverage_trials.csv.gz",
        }
        handoff = load_report_safe_finalized_test_handoff(
            ctx, PROTOCOL,
            required_artifacts=tuple(report_csv_contract) + tuple(sorted(report_nonframe_contract)),
        )
        expected_notebook_source_hash = str(
            handoff.protocol.get("decisions", {}).get("notebook_source_hash", "")
        )
        current_notebook_source_hash = notebook_source_hash(NOTEBOOK_PATH)
        if (
            len(expected_notebook_source_hash) != 64
            or current_notebook_source_hash != expected_notebook_source_hash
        ):
            raise StudyGateError(
                "Notebook cell sources changed after protocol freeze; report is blocked"
            )
        print("REPORT_NOTEBOOK_SOURCE_MATCH", current_notebook_source_hash)
        frozen_artifacts = handoff.protocol["decisions"]["artifact_manifest"]
        frozen_environment_path = resolve_frozen_artifact(
            handoff.protocol, "inference_environment_fingerprint",
            INFERENCE_ENVIRONMENT_FINGERPRINT,
        )
        frozen_pip_freeze_path = resolve_frozen_artifact(
            handoff.protocol, "inference_pip_freeze", INFERENCE_PIP_FREEZE,
        )
        _require_exclusive_scientific_gpu("report_runtime_preflight")
        report_preflight = environment_report(
            PROJECT_ROOT, cfg.paths.cache_root, DEVICE, cfg.profile.value,
        )
        if report_preflight["status"].eq("FAIL").any():
            failed = report_preflight.loc[
                report_preflight["status"].eq("FAIL"), "item"
            ].astype(str).tolist()
            raise StudyGateError(f"Report runtime preflight failed: {failed}")
        report_environment = environment_fingerprint(report_preflight)
        if report_environment != json.loads(
            frozen_environment_path.read_text(encoding="utf-8")
        ):
            raise StudyGateError(
                "Report runtime differs from the frozen validation/test environment"
            )
        report_pip_freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            check=True, text=True, capture_output=True,
        ).stdout
        if report_pip_freeze != frozen_pip_freeze_path.read_text(encoding="utf-8"):
            raise StudyGateError(
                "Report distributions differ from the frozen validation/test lock"
            )
        report_environment_receipt = atomic_write_json(
            REPORT_ENVIRONMENT_RECEIPT,
            {
                "schema_version": "semuav.report-environment.v1",
                "environment_fingerprint_sha256": hash_file(
                    frozen_environment_path
                ),
                "pip_freeze_sha256": hash_file(frozen_pip_freeze_path),
                "observed_environment_fingerprint": report_environment,
                "python_executable": sys.executable,
                "python_version": sys.version,
                "notebook_source_hash": current_notebook_source_hash,
                "status": "VERIFIED_EXACT_MATCH",
            },
        )
        print("REPORT_ENVIRONMENT_MATCH", hash_file(report_environment_receipt))
        forbidden_present = sorted(forbidden_report_artifacts.intersection(handoff.artifacts))
        if forbidden_present:
            raise StudyGateError(
                "Report handoff exposes prohibited raw policy artifacts: "
                f"{forbidden_present}"
            )
        max_report_rows = 250_000
        max_report_compressed_bytes = 64 * 1024 * 1024
        max_report_metadata_bytes = 8 * 1024 * 1024
        frozen_validation_contract = {
            "validation_exp1_perception_summary": (
                "validation_exp1_perception_summary.csv.gz",
                "validation_exp1_perception_summary",
            ),
            "validation_model_selection_gate": (
                "validation_model_selection_gate.csv.gz",
                "validation_model_selection_gate",
            ),
        }
        frozen_validation_evidence_paths = []
        for artifact_key, (expected_name, frame_name) in (
            frozen_validation_contract.items()
        ):
            path = resolve_frozen_artifact(handoff.protocol, artifact_key)
            if path.name != expected_name or path.stat().st_size > max_report_compressed_bytes:
                raise StudyGateError(
                    f"Frozen validation evidence path/size changed: {artifact_key}"
                )
            frame = pd.read_csv(
                path, compression="gzip", nrows=max_report_rows + 1,
                low_memory=False,
            )
            if frame.empty or len(frame) > max_report_rows:
                raise StudyGateError(
                    f"Frozen validation evidence is empty or oversized: {artifact_key}"
                )
            frame["evidence_partition"] = (
                "DEVELOPMENT_VALIDATION_MODEL_SELECTION_ONLY"
            )
            report_frames[frame_name] = frame
            frozen_validation_evidence_paths.append(path)
        validation_model_summary = report_frames[
            "validation_exp1_perception_summary"
        ]
        validation_selection_gate = report_frames[
            "validation_model_selection_gate"
        ]
        protocol_decisions = handoff.protocol["decisions"]
        protocol_validation_summary = handoff.protocol["validation_summary"]
        selection = protocol_decisions["selection"]
        temporal_gate_required_columns = {
            "model_id", "selected_for_confirmatory",
            "selected_compact_model", "selected_for_temporal_experiment1",
            "temporal_experiment1_model_ids",
            "temporal_experiment1_model_count", "evidence_partition",
        }
        temporal_gate_missing_columns = sorted(
            temporal_gate_required_columns.difference(validation_selection_gate.columns)
        )
        if temporal_gate_missing_columns:
            raise StudyGateError(
                "Frozen model-selection gate omits temporal Experiment 1 fields: "
                f"{temporal_gate_missing_columns}"
            )
        selected_compact_model = str(selection.get("selected_compact_model", ""))
        temporal_experiment1_model_ids = [
            "detector_only", selected_compact_model,
        ]
        comparison_models = set(validation_model_summary["model_id"].astype(str))
        gate_models = set(validation_selection_gate["model_id"].astype(str))
        expected_models = {
            str(row["model_id"]) for row in selection["selection_summary"]
        }
        required_roles = {
            "detector_only", str(selection["best_compact_vlm"]),
            str(selection["upper_bound_model"]),
        }
        selected_rows = validation_selection_gate.loc[
            validation_selection_gate["selected_for_confirmatory"].astype(bool)
        ]
        temporal_rows = validation_selection_gate.loc[
            validation_selection_gate[
                "selected_for_temporal_experiment1"
            ].astype(bool)
        ]
        temporal_route_json = json.dumps(
            temporal_experiment1_model_ids, separators=(",", ":"),
        )
        if (
            comparison_models != gate_models
            or comparison_models != expected_models
            or not required_roles.issubset(comparison_models)
            or selected_compact_model
            not in set(REGISTERED_RUNTIME_SMOKE_MODEL_IDS[:2])
            or str(selection.get("best_compact_vlm", ""))
            != selected_compact_model
            or str(selection.get("selected_model", ""))
            not in {"detector_only", selected_compact_model}
            or selection.get("temporal_experiment1_model_ids")
            != temporal_experiment1_model_ids
            or type(selection.get("temporal_experiment1_model_count")) is not int
            or selection.get("temporal_experiment1_model_count") != 2
            or protocol_decisions.get("selected_compact_model")
            != selected_compact_model
            or protocol_decisions.get("temporal_experiment1_model_ids")
            != temporal_experiment1_model_ids
            or type(protocol_decisions.get("temporal_experiment1_model_count"))
            is not int
            or protocol_decisions.get("temporal_experiment1_model_count") != 2
            or protocol_validation_summary.get("selected_model")
            != str(selection.get("selected_model", ""))
            or protocol_validation_summary.get("selected_compact_model")
            != selected_compact_model
            or protocol_validation_summary.get("temporal_experiment1_model_ids")
            != temporal_experiment1_model_ids
            or type(protocol_validation_summary.get(
                "temporal_experiment1_model_count"
            )) is not int
            or protocol_validation_summary.get("temporal_experiment1_model_count")
            != 2
            or len(selected_rows) != 1
            or str(selected_rows.iloc[0]["model_id"])
            != str(selection["selected_model"])
            or list(temporal_rows["model_id"].astype(str))
            != temporal_experiment1_model_ids
            or not validation_selection_gate[
                "selected_compact_model"
            ].astype(str).eq(selected_compact_model).all()
            or not validation_selection_gate[
                "temporal_experiment1_model_ids"
            ].astype(str).eq(temporal_route_json).all()
            or not pd.to_numeric(
                validation_selection_gate["temporal_experiment1_model_count"],
                errors="coerce",
            ).eq(2).all()
            or set(validation_selection_gate["evidence_partition"].astype(str))
            != {"DEVELOPMENT_VALIDATION_MODEL_SELECTION_ONLY"}
        ):
            raise StudyGateError(
                "Frozen development model-comparison/selection evidence is inconsistent"
            )
        report_media_audit = verify_visdrone_media_partition_audit(
            frozen_artifacts["media_partition_audit_receipt"]["path"],
            expected_inventory_sha256=(
                frozen_artifacts["source_hash_inventory"]["sha256"]
            ),
        )
        for filename in sorted(report_nonframe_contract):
            path = handoff.artifact(filename)
            if path.stat().st_size > max_report_metadata_bytes:
                raise StudyGateError(
                    f"Report metadata exceeds the 8 MiB cap: {filename}"
                )
        for filename, frame_name in report_csv_contract.items():
            path = handoff.artifact(filename)
            if path.stat().st_size > max_report_compressed_bytes:
                raise StudyGateError(
                    f"Report artifact exceeds the 64 MiB compressed cap: {filename}"
                )
            frame = pd.read_csv(
                path, compression="gzip", nrows=max_report_rows + 1, low_memory=False,
            )
            if len(frame) > max_report_rows:
                raise StudyGateError(
                    f"Report artifact exceeds the 250000-row cap: {filename}"
                )
            report_frames[frame_name] = frame
        reduction_names = (
            "policy_pair_cells", "h3_negative_episode_cells",
            "resource_summary", "resource_rate_summary", "resource_episode_cells",
            "resource_cluster_cells", "controller_summary",
            "exp5_summary", "h4_matched_coverage_cells",
            "h4_matched_coverage_audit", "h4_per_trace_audit",
            "h4_trace_spill_index", "pairing_audit", "io_audit",
        )
        policy_reduction_frames = {
            name: report_frames[name] for name in reduction_names
        }
        reduction_receipt_path = handoff.artifact("policy_reduction_receipt.json")
        reduction_receipt = json.loads(
            reduction_receipt_path.read_text(encoding="utf-8")
        )
        reduction_parameters = reduction_receipt.get("parameters")
        try:
            reduction_h4_threshold = float(
                reduction_parameters["semantic_acceptance_threshold"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StudyGateError(
                "Policy-reduction receipt lacks its H4 threshold"
            ) from exc
        if not np.isclose(
            reduction_h4_threshold,
            float(cfg.evaluation["h4_high_confidence_threshold"]),
            rtol=0.0, atol=1e-12,
        ):
            raise StudyGateError(
                "Policy-reduction receipt differs from the frozen H4 threshold"
            )
        reduction_artifacts = reduction_receipt.get("artifacts", {})
        if not isinstance(reduction_artifacts, dict) or set(reduction_artifacts) != set(
            reduction_names
        ):
            raise StudyGateError(
                "Policy-reduction receipt has an incomplete compact artifact domain"
            )
        for name in reduction_names:
            item = reduction_artifacts[name]
            if not isinstance(item, dict):
                raise StudyGateError(
                    f"Policy-reduction receipt entry is malformed: {name}"
                )
            artifact_path = handoff.artifact(f"{name}.csv.gz")
            if (
                Path(str(item.get("path", ""))).expanduser().resolve()
                != artifact_path.resolve()
                or str(item.get("sha256", "")) != hash_file(artifact_path)
                or int(item.get("row_count", -1)) != len(policy_reduction_frames[name])
            ):
                raise StudyGateError(
                    f"Policy-reduction receipt binding changed: {name}"
                )
        policy_index_path = handoff.artifact("test_policy_partition_index.csv")
        max_policy_partitions = 10_000
        policy_partition_index = pd.read_csv(
            policy_index_path, nrows=max_policy_partitions + 1,
        )
        if len(policy_partition_index) > max_policy_partitions:
            raise StudyGateError("Policy partition index exceeds the 10000-row cap")
        required_policy_index = {
            "artifact_path", "artifact_sha256", "row_count",
            "trace_id", "replicate_seed", "stage",
        }
        if missing := required_policy_index.difference(policy_partition_index.columns):
            raise StudyGateError(
                f"Policy partition index lacks receipt fields: {sorted(missing)}"
            )
        reduction_sources = reduction_receipt.get("source_partitions")
        if not isinstance(reduction_sources, list):
            raise StudyGateError("Policy-reduction receipt lacks source partitions")
        receipt_partition_bindings = sorted(
            (
                str(Path(str(item.get("path", ""))).expanduser()),
                str(item.get("sha256", "")), int(item.get("row_count", -1)),
                str(item.get("trace_id", "")), int(item.get("replicate_seed", -1)),
                str(item.get("stage", "")),
            )
            for item in reduction_sources if isinstance(item, dict)
        )
        index_partition_bindings = sorted(
            (
                str(Path(str(row.artifact_path)).expanduser()),
                str(row.artifact_sha256), int(row.row_count), str(row.trace_id),
                int(row.replicate_seed), str(row.stage),
            )
            for row in policy_partition_index.itertuples(index=False)
        )
        if (
            len(receipt_partition_bindings) != len(reduction_sources)
            or receipt_partition_bindings != index_partition_bindings
        ):
            raise StudyGateError(
                "Policy-reduction receipt is not bound to the frozen partition index"
            )
        reduction_io = policy_reduction_frames["io_audit"]
        if (
            len(reduction_io) != 1
            or bool(reduction_io.iloc[0][
                "h4_campaign_selected_trial_rows_materialized"
            ])
            or int(reduction_io.iloc[0]["maximum_observed_chunk_rows"])
            > int(reduction_io.iloc[0]["configured_chunk_size"])
        ):
            raise StudyGateError("Policy reduction violated its bounded-memory contract")
        recompute_inputs = {
            "exp3_detail": report_frames["test_exp3_static_partition_index"],
            "h1_fixed_grid_aggregates": report_frames[
                "test_h1_fixed_grid_group_aggregates"
            ],
            "h1_fixed_grid_audit": report_frames[
                "test_h1_fixed_grid_coverage_audit"
            ],
            "manifest": report_frames["test_sample_cluster_manifest"],
            "temporal_manifest": report_frames["test_temporal_cluster_manifest"],
            "policy_spec": selection["policy_spec"],
            "policy_reduction": policy_reduction_frames,
        }
        hypothesis_reports = recompute_hypotheses_from_frozen_results(
            recompute_inputs, cfg=cfg,
            cluster_columns=("visual_group_id", "trace_id"),
            bootstrap_repetitions=int(cfg.evaluation["bootstrap_repetitions"]),
        )
        registered_hypothesis_artifacts = {
            "h1_report": "h1_semantic_noninferiority",
            "h2_report": "h2_confirmatory_report",
            "h2_matched_budget": "test_h2_matched_budget",
            "h3_report": "h3_mission_awareness",
            "h4_report": "h4_uncertainty_escalation",
            "h4_coverage_audit": "h4_matched_coverage_audit",
            "hypothesis_status": "hypothesis_status",
        }
        for result_name, artifact_name in registered_hypothesis_artifacts.items():
            expected = report_frames[artifact_name]
            observed = hypothesis_reports[result_name]
            pd.testing.assert_frame_equal(
                observed.reset_index(drop=True), expected.reset_index(drop=True),
                check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12,
            )
        hypothesis_effects = build_hypothesis_effects_frame(
            h1_report=hypothesis_reports["h1_report"],
            h2_report=hypothesis_reports["h2_report"],
            h3_report=hypothesis_reports["h3_report"],
            h4_report=hypothesis_reports["h4_report"],
            hypothesis_status=hypothesis_reports["hypothesis_status"],
            embedded_retest_effects=None,
            embedded_retest_status=None,
        )
        status_value = hypothesis_reports["hypothesis_status"]
        hypothesis_statuses = {}
        not_estimable_reports = {
            "H2": hypothesis_reports["h2_report"],
            "H3": hypothesis_reports["h3_report"],
            "H4": hypothesis_reports["h4_report"],
        }
        not_estimable_count_fields = {
            "H2": (
                ("temporal_visual_groups", "required_temporal_visual_groups"),
                ("temporal_group_trace_cells", "required_temporal_group_trace_cells"),
                ("temporal_observations_per_group", "required_temporal_observations_per_group"),
            ),
            "H3": (
                ("positive_visual_groups", "required_positive_visual_groups"),
                ("negative_visual_groups", "required_negative_visual_groups"),
            ),
            "H4": (
                ("matched_coverage_selected_rows", "required_matched_coverage_selected_rows"),
                (
                    "matched_coverage_group_trace_cells",
                    "required_matched_coverage_group_trace_cells",
                ),
                ("high_confidence_rows", "required_high_confidence_rows"),
                (
                    "high_confidence_group_trace_cells",
                    "required_high_confidence_group_trace_cells",
                ),
            ),
        }
        not_estimable_binary_fields = {
            # A matched-coverage audit can fail even when the aggregate H4
            # denominator counts happen to clear their numeric minima.  Keep
            # that structural failure explicit instead of either promoting an
            # inestimable result or blocking an otherwise valid null report.
            "H4": (
                "coverage_gate_pass", "high_confidence_gate_pass",
                "h4_count_gate_pass",
            ),
        }
        for row in status_value.itertuples(index=False):
            hypothesis = str(row.hypothesis)
            execution_status = str(getattr(row, "execution_status", "")).upper()
            if hypothesis == "H5":
                hypothesis_statuses[hypothesis] = "N_A"
            elif execution_status == "NOT_ESTIMABLE":
                source = not_estimable_reports.get(hypothesis)
                if not isinstance(source, pd.DataFrame) or source.empty:
                    raise StudyGateError(
                        f"{hypothesis} NOT_ESTIMABLE lacks its frozen report"
                    )
                source_row = source.iloc[0]
                reasons = [
                    str(value).strip()
                    for value in source.get(
                        "not_estimable_reason", pd.Series(dtype=str)
                    ).dropna().unique()
                    if str(value).strip()
                ]
                count_gates = []
                for observed_field, required_field in not_estimable_count_fields.get(
                    hypothesis, ()
                ):
                    if observed_field not in source_row or required_field not in source_row:
                        continue
                    observed = int(pd.to_numeric(source_row[observed_field], errors="raise"))
                    required = int(pd.to_numeric(source_row[required_field], errors="raise"))
                    if observed < 0 or required <= 0:
                        raise StudyGateError(
                            f"{hypothesis} has an invalid estimability count gate"
                        )
                    count_gates.append({
                        "gate": observed_field,
                        "observed": observed,
                        "required": required,
                        "passed": observed >= required,
                    })
                for gate_field in not_estimable_binary_fields.get(hypothesis, ()):
                    if gate_field not in source.columns:
                        raise StudyGateError(
                            f"{hypothesis} lacks its registered estimability gate: "
                            f"{gate_field}"
                        )
                    gate_values = source[gate_field].dropna()
                    if gate_values.empty or not gate_values.map(
                        lambda value: isinstance(value, (bool, np.bool_))
                    ).all():
                        raise StudyGateError(
                            f"{hypothesis} estimability gate is not boolean: {gate_field}"
                        )
                    passed = bool(gate_values.astype(bool).all())
                    count_gates.append({
                        "gate": gate_field,
                        "observed": int(passed),
                        "required": 1,
                        "passed": passed,
                    })
                if not count_gates or all(gate["passed"] for gate in count_gates):
                    raise StudyGateError(
                        f"{hypothesis} NOT_ESTIMABLE lacks an explicit failed count gate"
                    )
                hypothesis_statuses[hypothesis] = {
                    "status": "NOT_ESTIMABLE",
                    "reason": "; ".join(sorted(set(reasons))) or (
                        f"{hypothesis} failed its registered estimability count gate"
                    ),
                    "count_gates": count_gates,
                }
            elif execution_status == "COMPLETE":
                hypothesis_statuses[hypothesis] = (
                    "PASS" if bool(row.claim_supported) else "FAIL"
                )
            else:
                raise StudyGateError(
                    f"{hypothesis} has unsupported execution_status={execution_status!r}"
                )
        exp1_test = report_frames["test_exp1_perception_detail"]
        forbidden_fixture = False
        for frame in report_frames.values():
            forbidden_fixture = forbidden_fixture or bool(
                ("scientific" in frame and not frame["scientific"].astype(bool).all())
                or ("is_fixture" in frame and frame["is_fixture"].astype(bool).any())
                or (
                    "evidence_class" in frame
                    and frame["evidence_class"].astype(str).str.contains(
                        "fixture|synthetic|mock", case=False, regex=True,
                    ).any()
                )
            )
        selected_model = str(selection["selected_model"])
        selected_vlm_used = registered_selected_vlm_claim_gate(selection)
        real_vlm_inference = bool(
            selected_vlm_used
            and exp1_test["model_id"].astype(str).eq(selected_model).any()
        )
        completion = report_frames["final_completion_status"]
        registered_status = status_value[
            status_value["hypothesis"].astype(str).isin({"H1", "H2", "H3", "H4"})
        ].copy()
        if (
            len(registered_status) != 4
            or set(registered_status["hypothesis"].astype(str))
            != {"H1", "H2", "H3", "H4"}
        ):
            raise StudyGateError(
                "Frozen hypothesis status must contain H1--H4 exactly once"
            )
        expected_evaluated = int(
            registered_status["execution_status"].astype(str).str.upper().eq("COMPLETE").sum()
        )
        expected_not_estimable = int(
            registered_status["execution_status"].astype(str).str.upper().eq(
                "NOT_ESTIMABLE"
            ).sum()
        )
        completion_count_columns = {
            "hypotheses_registered", "hypotheses_evaluated",
            "hypotheses_not_estimable",
        }
        if len(completion) != 1 or not completion_count_columns.issubset(completion.columns):
            raise StudyGateError(
                "Completion status lacks exact registered/evaluated/not-estimable counts"
            )
        completion_row = completion.iloc[0]
        if (
            int(completion_row["hypotheses_registered"]) != 4
            or int(completion_row["hypotheses_evaluated"]) != expected_evaluated
            or int(completion_row["hypotheses_not_estimable"])
            != expected_not_estimable
        ):
            raise StudyGateError(
                "Completion hypothesis counts disagree with frozen hypothesis status"
            )
        completion_pass = bool(
            completion_row.get("execution_status") == "COMPLETE"
            and bool(completion_row.get("empirical_complete", False))
        )
        scientific_evidence_verified = bool(
            handoff.ledger.get("status") == "COMPLETE"
            and cfg.profile.value == "full"
            and cfg.study_scope.value == "visdrone_primary"
            and cfg.truth_mode.value == "publisher_source_rule"
            and handoff.protocol["validation_summary"].get("all_registered_gates_passed") is True
            and {"source_receipt", "source_hash_inventory"}.issubset(frozen_artifacts)
            and bool(report_media_audit.get("passed", False))
            and completion_pass
            and not forbidden_fixture
        )
        report_frames["paper_exp7_readiness"] = pd.DataFrame([
            {
                "item": "verified one-time workstation campaign",
                "status": "PASS" if completion_pass else "FAIL",
            },
            {
                "item": "Jetson-class target evidence (H5)",
                "status": "SKIP",
            },
        ])
        claim_ledger = build_readiness_claim_ledger(
            phase="report",
            evidence={
                "official_visdrone_receipt": True,
                "publisher_group_splits_passed": True,
                "real_detector_inference": exp1_test["model_id"].astype(str).eq("detector_only").any(),
                "real_vlm_inference": real_vlm_inference,
                "selected_vlm_inference": real_vlm_inference,
                "validation_gates_passed": bool(handoff.protocol["validation_summary"].get("all_registered_gates_passed", False)),
                "protocol_frozen": True,
                "one_time_test_complete": completion_pass,
                "paper_artifacts_hashed": False,
            },
            selected_vlm_used=selected_vlm_used,
            hypotheses=hypothesis_statuses,
        )
        atomic_write_frame(PROVISIONAL_CLAIM_LEDGER, claim_ledger)
        display(claim_ledger)
    else:
        print("Read-only report stage paused.")
    """, "report-only"),
    md("""
    ## Publication figures

    Every figure is exported at 300 dpi in PNG plus PDF/SVG, receives a source
    hash manifest, and is visibly watermarked unless the confirmatory claim ledger
    verifies applicable scientific completion. Missing inputs create explicit
    skipped-manifest entries rather than invented bars or placeholder curves.
    """),
    code("""
    publication_figures = {}
    if PHASE == "report" and EXECUTE_PHASE:
        _require_v2r13_preimport_source_gate()
        # Null, negative, and failed hypotheses are still real empirical results.
        # Scientific figure status follows verified evidence provenance, never
        # whether an effect supports a claim.
        scientific = bool(scientific_evidence_verified)
        frames = {
            "dataset_manifest": report_frames.get("test_sample_cluster_manifest"),
            "exp1_summary": report_frames.get("test_exp1_perception_summary"),
            "exp2_frontier": report_frames.get("test_exp2_representation_frontier"),
            "exp3_summary": report_frames.get("test_exp3_static_summary"),
            "exp4_summary": report_frames.get("resource_summary"),
            "exp4_controller_summary": report_frames.get("controller_summary"),
            "exp5_summary": report_frames.get("exp5_summary"),
            "exp6_campaign_summary": report_frames.get("test_receiver_robustness_summary"),
            "exp7_readiness": report_frames.get("paper_exp7_readiness"),
            "hypothesis_effects": hypothesis_effects,
        }
        publication_figures = generate_publication_figures(
            frames=frames, output_dir=FIGURE_DIR, scientific=scientific,
        )
        for figure_id, figure in publication_figures.items():
            print("Publication figure:", figure_id)
            display(figure)
            plt.close(figure)
        print("FIGURES_COMPLETE", FIGURE_DIR, "scientific=", scientific)
    else:
        print("Publication figure stage paused.")
    """, "report-only"),
    md("""
    ## Publication tables, reproducibility manifest, and claim ledger

    Tables include exact Ns, independent group counts, paired effect sizes,
    confidence intervals, adjusted p-values, latency quantiles, payload bytes,
    GPU-board energy, memory, false alerts, critical misses, and protocol fault
    outcomes. Energy is labeled by measurement scope; no total-system or radio
    claim is made without measured coefficients.
    """),
    code("""
    if PHASE == "report" and EXECUTE_PHASE:
        _require_v2r13_preimport_source_gate()
        if not scientific_evidence_verified:
            raise StudyGateError("Verified COMPLETE real-data handoff is required for the paper report")
        paper_table_frame_names = {
            "validation_exp1_perception_summary",
            "validation_model_selection_gate",
            "test_dataset_summary",
            "test_exp1_perception_summary",
            "test_exp2_representation_frontier",
            "test_exp3_static_summary",
            "resource_summary", "resource_rate_summary", "controller_summary",
            "exp5_summary", "h4_matched_coverage_audit", "h4_per_trace_audit",
            "pairing_audit", "io_audit", "test_budget_compliance",
            "confirmatory_gate_table",
            "test_h1_fixed_grid_group_aggregates",
            "test_h1_fixed_grid_coverage_audit", "h1_semantic_noninferiority",
            "test_h2_matched_budget", "h2_confirmatory_report",
            "h3_mission_awareness", "h4_uncertainty_escalation",
            "test_receiver_robustness_summary",
            "test_receiver_state_strategy_summary", "hypothesis_status",
            "final_completion_status", "test_episode_trace_coverage",
            "test_dynamic_cartesian_coverage_audit", "paper_exp7_readiness",
        }
        table_inputs = {
            **{
                name: frame for name, frame in report_frames.items()
                if isinstance(frame, pd.DataFrame)
                and name in paper_table_frame_names
            },
            "hypothesis_effects": hypothesis_effects,
        }
        table_inputs = {
            name: frame for name, frame in table_inputs.items()
            if isinstance(frame, pd.DataFrame) and not frame.empty
        }
        table_manifest = export_paper_tables(table_inputs, TABLE_DIR, scientific=True)
        if handoff.restricted_evidence_receipt_path is None:
            raise StudyGateError("Finalized handoff lacks its restricted-evidence receipt")
        frozen_partition_audit_paths = [
            Path(frozen_artifacts[name]["path"])
            for name in (
                "media_partition_audit_receipt",
                "media_partition_hash_audit",
                "test_group_capacity_audit",
            )
        ]
        manifest_inputs = [
            PROTOCOL, TEST_LEDGER,
            handoff.restricted_evidence_receipt_path,
            report_environment_receipt,
            *frozen_partition_audit_paths,
            *frozen_validation_evidence_paths,
            *sorted(handoff.artifacts.values()),
            *[
                path for path in sorted(TABLE_DIR.glob("*"))
                if path.resolve() != FINAL_CLAIM_LEDGER.resolve()
            ],
            *sorted(FIGURE_DIR.glob("*")),
        ]
        named_manifest_inputs = {
            f"artifact_{index:04d}_{path.name}": path
            for index, path in enumerate(manifest_inputs)
            if path.is_file()
        }
        common_root = Path(os.path.commonpath([str(path.resolve()) for path in named_manifest_inputs.values()]))
        if common_root.is_file():
            common_root = common_root.parent
        paper_artifact_manifest = write_artifact_hash_manifest(
            named_manifest_inputs,
            PAPER_ARTIFACT_MANIFEST,
            metadata={
                "study_scope": cfg.study_scope.value,
                "truth_mode": cfg.truth_mode.value,
                "config_hash": ctx.config_hash,
                "code_hash": ctx.code_hash,
                "notebook_source_hash": notebook_source_hash(NOTEBOOK_PATH),
                "environment_fingerprint_sha256": handoff.protocol["decisions"].get(
                    "inference_environment_fingerprint_sha256"
                ),
                "report_environment_receipt_sha256": hash_file(
                    report_environment_receipt
                ),
                "prohibited_claims": list(cfg.benchmark["prohibited_claims"]),
                "h5_status": "NOT_APPLICABLE_NO_TARGET_EVIDENCE",
            },
            root=common_root,
        )
        verify_artifact_hash_manifest(paper_artifact_manifest)

        # Only after every table/figure/result hash has been written and checked
        # may a claim become eligible. The final ledger is bound to the verified
        # inner manifest and is intentionally excluded from it. The outer
        # envelope binds both levels, and a tiny completion receipt is written
        # last; consumers must require that receipt.
        final_claim_ledger = build_readiness_claim_ledger(
            phase="report",
            evidence={
                "official_visdrone_receipt": True,
                "publisher_group_splits_passed": True,
                "real_detector_inference": exp1_test["model_id"].astype(str).eq("detector_only").any(),
                "real_vlm_inference": real_vlm_inference,
                "selected_vlm_inference": real_vlm_inference,
                "validation_gates_passed": bool(
                    handoff.protocol["validation_summary"].get("all_registered_gates_passed", False)
                ),
                "protocol_frozen": True,
                "one_time_test_complete": completion_pass,
                "paper_artifacts_hashed": True,
            },
            selected_vlm_used=selected_vlm_used,
            hypotheses=hypothesis_statuses,
        )
        final_claim_ledger["paper_artifact_manifest_sha256"] = hash_file(
            paper_artifact_manifest
        )
        final_claim_ledger["report_environment_receipt_sha256"] = hash_file(
            report_environment_receipt
        )
        final_claim_ledger["paper_bundle_status"] = (
            "ELIGIBLE_PENDING_COMPLETION_RECEIPT"
        )
        atomic_write_frame(FINAL_CLAIM_LEDGER, final_claim_ledger)
        envelope_inputs = {
            "paper_artifact_manifest": paper_artifact_manifest,
            "final_claim_ledger": FINAL_CLAIM_LEDGER,
            "frozen_protocol": PROTOCOL,
            "test_access_ledger": TEST_LEDGER,
        }
        envelope_root = Path(os.path.commonpath([
            str(Path(path).resolve()) for path in envelope_inputs.values()
        ]))
        if envelope_root.is_file():
            envelope_root = envelope_root.parent
        paper_manifest = write_artifact_hash_manifest(
            envelope_inputs,
            PAPER_RESULTS_MANIFEST,
            metadata={
                "bundle_status": "ELIGIBLE_PENDING_COMPLETION_RECEIPT",
                "inner_manifest_sha256": hash_file(paper_artifact_manifest),
                "final_claim_ledger_sha256": hash_file(FINAL_CLAIM_LEDGER),
                "report_environment_receipt_sha256": hash_file(
                    report_environment_receipt
                ),
                "energy_claim_scope": cfg.evaluation["energy_claim_scope"],
            },
            root=envelope_root,
        )
        verify_artifact_hash_manifest(paper_manifest)
        completion_receipt = atomic_write_json(
            FINAL_COMPLETION_RECEIPT,
            {
                "schema_version": "semuav.paper-completion-receipt.v1",
                "status": "COMPLETE",
                "paper_results_manifest_path": str(paper_manifest.resolve()),
                "paper_results_manifest_sha256": hash_file(paper_manifest),
                "final_claim_ledger_path": str(FINAL_CLAIM_LEDGER.resolve()),
                "final_claim_ledger_sha256": hash_file(FINAL_CLAIM_LEDGER),
                "report_environment_receipt_path": str(
                    report_environment_receipt.resolve()
                ),
                "report_environment_receipt_sha256": hash_file(
                    report_environment_receipt
                ),
                "protocol_sha256": hash_file(PROTOCOL),
                "test_ledger_sha256": hash_file(TEST_LEDGER),
            },
        )
        verified_completion = verify_paper_completion_receipt(
            completion_receipt,
            expected_results_manifest=paper_manifest,
            expected_claim_ledger=FINAL_CLAIM_LEDGER,
            expected_protocol=PROTOCOL,
            expected_test_ledger=TEST_LEDGER,
            expected_report_environment_receipt=report_environment_receipt,
        )
        print("PAPER_REPORT_COMPLETE", completion_receipt)
    else:
        print("Publication table/manifest stage paused.")
    """, "report-only"),
    md("""
    # Phase 5 — frozen H5 not-applicable status

    H5 was preregistered as `NOT_APPLICABLE_NO_TARGET_EVIDENCE` for this
    VisDrone-only paper and
    cannot be enabled by adding Jetson evidence after protocol freeze. This phase
    only verifies the completed paper envelope and renders that frozen status; it
    never opens the one-time test or target logs. Any Jetson H5 study requires a
    separately preregistered run and notebook before target evidence is collected.
    """),
    code("""
    if PHASE == "embedded" and EXECUTE_PHASE:
        _require_v2r13_preimport_source_gate()
        verify_paper_completion_receipt(
            FINAL_COMPLETION_RECEIPT,
            expected_results_manifest=PAPER_RESULTS_MANIFEST,
            expected_claim_ledger=FINAL_CLAIM_LEDGER,
            expected_protocol=PROTOCOL,
            expected_test_ledger=TEST_LEDGER,
            expected_report_environment_receipt=REPORT_ENVIRONMENT_RECEIPT,
        )
        h5 = evaluate_optional_embedded_h5(
            handoff={}, cfg=cfg, raw_trial_path=None,
        )
        display(h5)
    else:
        print(
            "H5 status phase paused. This paper freezes H5 as "
            "NOT_APPLICABLE_NO_TARGET_EVIDENCE."
        )
    """, "embedded-only"),
    md("""
    ## Final interpretation rule

    A completed run may claim only rows marked `eligible=True` in the hashed
    claim ledger. A VLM no-go result is a valid study outcome and converts the
    deployed method to detector fallback; it does not authorize a contextual-VLM
    claim. H5 is frozen as `NOT_APPLICABLE_NO_TARGET_EVIDENCE`, separate, and not
    required for H1–H4. The prohibited-claim list is
    immutable and applies to the manuscript, abstract, captions, and conclusions.
    """),
]


for index, cell in enumerate(cells):
    source = str(cell["source"])
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:8]
    cell["id"] = f"visdrone-{index:02d}-{digest}"


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (SemUAV-VLM 2026-09-21 v2r13)",
            "language": "python",
            "name": "semuav-vlm-20260921-v2r13",
        },
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT} with {len(cells)} cells")
