"""Executable regression for the v2r11 -> v2r13 boundary-routing defect.

The tests call the verifier generated into the notebook.  Only external file
acquisition, the already-tested runtime guard, and ``pip freeze`` are replaced
with deterministic fixtures; the receipt envelope and every terminal-boundary
comparison execute unchanged.
"""

from __future__ import annotations

import ast
import copy
import hashlib as _real_hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

# Load the archived builder in a temporary directory: importing the active v3
# authoring script here would test the wrong receipt lineage and write its notebook.
ROOT = Path(__file__).resolve().parents[1] / "legacy" / "v2r13"
with tempfile.TemporaryDirectory() as _builder_directory:
    _builder_source = ROOT / "build_visdrone_paper_notebook.py"
    _builder_namespace = {
        "__file__": str(Path(_builder_directory) / _builder_source.name),
        "__name__": "historical_v2r13_builder_fixture",
        "print": lambda *args, **kwargs: None,
    }
    exec(compile(_builder_source.read_text(encoding="utf-8"),
                 str(_builder_source), "exec"), _builder_namespace)
    notebook_builder = SimpleNamespace(**_builder_namespace)
NOTEBOOK = ROOT / "SemUAV_VLM_VisDrone_Paper_Study.ipynb"
PRODUCTION_STUDY_ROOT = "/data/Lane Detection Datasets/musta/semuav_vlm"
MODEL_IDS = ("qwen3_vl_2b", "smolvlm2_2b", "qwen3_vl_8b_upper_bound")


class StudyGateError(RuntimeError):
    """Fixture exception used by the generated verifier."""


class _AuthenticatedBytes(bytes):
    """Bytes returned by the mocked authenticated-file acquisition surface."""

    def __new__(cls, payload: bytes, claimed_sha256: str):
        value = super().__new__(cls, payload)
        value.claimed_sha256 = claimed_sha256
        return value


class _ClaimedDigest:
    def __init__(self, digest: str):
        self._digest = digest

    def hexdigest(self) -> str:
        return self._digest


class _AcquisitionHashlib:
    """Preserve real hashing except for explicitly authenticated fixture bytes."""

    @staticmethod
    def sha256(payload: bytes = b""):
        claimed = getattr(payload, "claimed_sha256", None)
        if claimed is not None:
            return _ClaimedDigest(str(claimed))
        return _real_hashlib.sha256(payload)


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _hash_object(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return _real_hashlib.sha256(payload).hexdigest()


def _generated_verifier_source() -> tuple[str, ast.FunctionDef]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    helper_source = next(
        str(cell.get("source", ""))
        for cell in notebook["cells"]
        if "def _verify_candidate_ready_receipt" in str(cell.get("source", ""))
    )
    module = ast.parse(helper_source)
    function = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_verify_candidate_ready_receipt"
    )
    return helper_source, function


def _assignment_value(
    function: ast.FunctionDef, name: str, namespace: dict[str, object],
) -> object:
    matches = [
        node for node in function.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    if len(matches) != 1:
        raise AssertionError(f"generated verifier assignment {name!r} is not singular")
    expression = ast.Expression(body=copy.deepcopy(matches[0].value))
    ast.fix_missing_locations(expression)
    return eval(compile(expression, f"<{name}-fixture>", "eval"), namespace)


class _StudyRootRewriter(ast.NodeTransformer):
    def __init__(self, replacement: str):
        self.replacement = replacement
        self.replacements = 0

    def visit_Constant(self, node: ast.Constant):
        if node.value == PRODUCTION_STUDY_ROOT:
            self.replacements += 1
            return ast.copy_location(ast.Constant(self.replacement), node)
        return node


def _replace_root(value: object, study_root: Path) -> object:
    if isinstance(value, str):
        return value.replace(PRODUCTION_STUDY_ROOT, str(study_root))
    if isinstance(value, list):
        return [_replace_root(item, study_root) for item in value]
    if isinstance(value, dict):
        return {key: _replace_root(item, study_root) for key, item in value.items()}
    return copy.deepcopy(value)


def _source_compatibility() -> dict[str, object]:
    """Use the real release contract so fixture verification detects drift."""
    return json.loads((ROOT / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))["source_compatibility"]


class VerifierHarness:
    def __init__(self, directory: str):
        self.study_root = Path(directory).resolve() / "study"
        self.package = "SemUAV-VLM-VisDrone-Paper-Study"
        self.release_root = (
            self.study_root / "releases"
            / "release_20260921_visdrone_paper_v2r13" / self.package
        )
        self.candidate_root = (
            self.study_root / "venvs"
            / "release_20260912_visdrone_inference_candidate_v2r3"
        )
        self.candidate_python = self.candidate_root / "bin" / "python"
        self.registry: dict[str, bytes] = {}
        self.labels: dict[str, str] = {}
        self.helper_source, self.function = _generated_verifier_source()
        self.release_root.mkdir(parents=True)
        self.candidate_python.parent.mkdir(parents=True)
        self.candidate_python.write_bytes(b"fixture-python\n")
        self._build()

    def _path(self, *parts: str) -> Path:
        return self.study_root.joinpath(*parts)

    def _put(
        self, path: Path, payload: object, *, claimed_sha256: str | None = None,
        label: str | None = None,
    ) -> bytes:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = payload if isinstance(payload, bytes) else _canonical_bytes(payload)
        path.write_bytes(bytes(raw) or b"fixture\n")
        stored: bytes = raw
        if claimed_sha256 is not None:
            stored = _AuthenticatedBytes(raw, claimed_sha256)
        self.registry[str(path.resolve())] = stored
        if label is not None:
            self.labels[label] = str(path.resolve())
        return stored

    def _expected_records(self) -> dict[str, object]:
        predecessor_release_id = "release_20260920_visdrone_paper_v2r9"
        predecessor_run_name = "semuav_visdrone_v2r9"
        immediate_release_id = "release_20260921_visdrone_paper_v2r11"
        source_release_id = "release_20260921_visdrone_paper_v2r10"
        paths = {
            "study_root": self.study_root,
            "release_root": self.release_root,
            "candidate_root": self.candidate_root,
            "candidate_python": self.candidate_python,
            "candidate_receipt_path": self.candidate_root / "SEMUAV_CANDIDATE_READY.json",
            "candidate_source_root": self._path(
                "releases", "release_20260912_visdrone_paper_v2r3", self.package,
            ),
            "builder_path": self._path(
                "incoming", "build_semuav_visdrone_inference_candidate_v2r3_20260912.sh",
            ),
            "deployer_path": self._path(
                "incoming", "deploy_semuav_visdrone_v2r13_20260921.sh",
            ),
            "archive_path": self._path(
                "incoming", f"{self.package}-20260921-v2r13.zip",
            ),
            "deployment_receipt_path": self._path(
                "private", "visdrone_v2r13_deployment", "DEPLOYMENT_READY.json",
            ),
            "source_predecessor_release_id": source_release_id,
            "source_predecessor_run_name": "semuav_visdrone_v2r10",
            "source_predecessor_release_root": self._path(
                "releases", source_release_id, self.package,
            ),
            "source_predecessor_archive_path": self._path(
                "incoming", f"{self.package}-20260921-v2r10.zip",
            ),
            "source_predecessor_deployer_path": self._path(
                "incoming", "deploy_semuav_visdrone_v2r10_20260921.sh",
            ),
            "source_predecessor_deployment_receipt_path": self._path(
                "private", "visdrone_v2r10_deployment", "DEPLOYMENT_READY.json",
            ),
            "immediate_predecessor_release_id": immediate_release_id,
            "immediate_predecessor_run_name": "semuav_visdrone_v2r11",
            "immediate_predecessor_release_root": self._path(
                "releases", immediate_release_id, self.package,
            ),
            "immediate_predecessor_archive_path": self._path(
                "incoming", f"{self.package}-20260921-v2r11.zip",
            ),
            "immediate_predecessor_deployer_path": self._path(
                "incoming", "deploy_semuav_visdrone_v2r11_20260921.sh",
            ),
            "immediate_predecessor_deployment_receipt_path": self._path(
                "private", "visdrone_v2r11_deployment", "DEPLOYMENT_READY.json",
            ),
            "expected_recovery_path": self._path(
                "private", "visdrone_v2r10_deployment", "V2R9_EXP1_RECOVERY_SOURCE.json",
            ),
            "predecessor_release_id": predecessor_release_id,
            "predecessor_run_name": predecessor_run_name,
            "predecessor_release_root": self._path(
                "releases", predecessor_release_id, self.package,
            ),
            "predecessor_archive_path": self._path(
                "incoming", f"{self.package}-20260920-v2r9.zip",
            ),
            "predecessor_deployer_path": self._path(
                "incoming", "deploy_semuav_visdrone_v2r9_20260920.sh",
            ),
            "predecessor_deployment_receipt_path": self._path(
                "private", "visdrone_v2r9_deployment", "DEPLOYMENT_READY.json",
            ),
        }
        paths["predecessor_frozen_root"] = self._path(
            "outputs", "visdrone_paper", predecessor_run_name, "frozen_inputs",
        )
        paths["predecessor_candidate"] = self._path(
            "venvs", "release_20260909_visdrone_inference_candidate_v2r2",
        )
        expected_immediate = _assignment_value(
            self.function, "expected_immediate_predecessor_evidence", paths,
        )
        expected_source = _assignment_value(
            self.function, "expected_source_predecessor_evidence", paths,
        )
        inherited_v2r9 = _assignment_value(
            self.function, "expected_inherited_v2r9_boundary", paths,
        )
        source_defect = _assignment_value(
            self.function, "expected_source_authenticated_defect", paths,
        )
        operator_execution = _assignment_value(
            self.function, "expected_operator_observed_execution", paths,
        )
        correction_path = ROOT / (
            "PRE_SMOKE_PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_20260921_V2R12.md"
        )
        correction_sha = _real_hashlib.sha256(correction_path.read_bytes()).hexdigest()
        paths.update({
            "expected_source_authenticated_defect": source_defect,
            "expected_operator_observed_execution": operator_execution,
            "PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_RECORD": correction_path,
            "predecessor_boundary_routing_correction_sha256": correction_sha,
        })
        historical_v2r11 = _assignment_value(
            self.function, "expected_observed_terminal_boundary", paths,
        )
        low_record = ROOT / "PRE_VALIDATION_LOW_EVIDENCE_PACKET_CORRECTION_20260921_V2R13.md"
        low_sha = _real_hashlib.sha256(low_record.read_bytes()).hexdigest()
        paths["low_evidence_packet_correction_sha256"] = low_sha
        top_boundary = _assignment_value(self.function, "expected_v2r12_terminal_boundary", paths)
        sealed = _assignment_value(self.function, "expected_sealed_predecessor_evidence", paths)
        inherited_v2r10_defect = _assignment_value(
            self.function, "expected_inherited_v2r10_source_defect", paths,
        )
        inherited_v2r10_execution = _assignment_value(
            self.function, "expected_inherited_v2r10_execution", paths,
        )
        paths.update({
            "expected_inherited_v2r10_source_defect": inherited_v2r10_defect,
            "expected_inherited_v2r10_execution": inherited_v2r10_execution,
        })
        inherited_v2r10 = _assignment_value(
            self.function, "expected_inherited_v2r10_boundary", paths,
        )
        return {
            **paths,
            "immediate": expected_immediate,
            "source": expected_source,
            "inherited_v2r9": inherited_v2r9,
            "inherited_v2r10": inherited_v2r10,
            "top_boundary": top_boundary,
            "historical_v2r11": historical_v2r11,
            "sealed": sealed,
            "low_sha": low_sha,
            "correction_sha": correction_sha,
        }

    def _build(self) -> None:
        r = self._expected_records()
        self.records = r
        predecessor_v2r9 = _replace_root(
            json.loads((ROOT / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))[
                "authenticated_predecessor_v2r9"
            ],
            self.study_root,
        )
        # The currently copied control is v2r11 before final sealing and v2r13
        # afterwards; the inherited v2r9 object is byte-stable in both.
        predecessor_v2r9["release_path"] = str(r["predecessor_release_root"])
        self.predecessor_v2r9 = predecessor_v2r9

        recovery_root = self._path(
            ".cache", "visdrone_paper_v2r9", "runs",
            "semuav_visdrone_v2r9-full-d799763aef-fbf80e5ec3",
        )
        artifacts = []
        for index in range(132):
            rows = 31 if index < 131 else 91
            artifacts.append({
                "relative_path": f"stages/01_vlm_fixture_{index:03d}/part.csv.gz",
                "bytes": 1,
                "sha256": "a" * 64,
                "rows": rows,
            })
        recovery = {
            "schema_version": "semuav.visdrone-v2r9-exp1-recovery.v1",
            "status": "QUIESCENT_V2R9_EXP1_SOURCE_FROZEN",
            "created_at_utc": "2026-09-21T00:00:00Z",
            "predecessor_release_id": "release_20260920_visdrone_paper_v2r9",
            "predecessor_run_name": "semuav_visdrone_v2r9",
            "predecessor_run_root": str(recovery_root),
            "predecessor_config_sha256": "d799763aef" + "0" * 54,
            "predecessor_code_sha256": (
                "fbf80e5ec3067ff8ca9c150c71a0ec2a3bcf5cffe9ec3663bb79c973789bdf76"
            ),
            "predecessor_model_lock_sha256": (
                "340dd59ca55a9be491ac806b50f9778a12d2b0b4a447485627aaf422dda05353"
            ),
            "run_manifest": {
                "path": str(recovery_root / "run_manifest.json"),
                "bytes": 1,
                "sha256": "b" * 64,
            },
            "artifact_count": 132,
            "inference_row_count": 4152,
            "artifact_inventory_sha256": _hash_object(artifacts),
            "artifacts": artifacts,
            "untouched_test_accessed": False,
            "protocol_frozen": False,
            "native_cache_completed": False,
        }
        recovery_raw = self._put(
            r["expected_recovery_path"], recovery,
            claimed_sha256="6ca88cb5e82c8b33e27fd229443e233a48323d91fd73d0e79ec51ff8fc61a608",
        )
        recovery_record = {
            "path": str(r["expected_recovery_path"]),
            "bytes": len(recovery_raw),
            "sha256": "6ca88cb5e82c8b33e27fd229443e233a48323d91fd73d0e79ec51ff8fc61a608",
            "schema_version": "semuav.visdrone-v2r9-exp1-recovery.v1",
            "status": "QUIESCENT_V2R9_EXP1_SOURCE_FROZEN",
            "artifact_count": 132,
            "inference_row_count": 4152,
        }
        # The verifier requires the historical byte count to equal the frozen
        # 37,416-byte receipt.  Padding is JSON whitespace and leaves semantics.
        unpadded = bytes(recovery_raw)
        if len(unpadded) > 37416:
            raise AssertionError("synthetic recovery fixture unexpectedly grew")
        padded = unpadded[:-1] + (b" " * (37416 - len(unpadded))) + b"\n"
        self.registry[str(r["expected_recovery_path"].resolve())] = (
            _AuthenticatedBytes(padded, recovery_record["sha256"])
        )
        r["expected_recovery_path"].write_bytes(padded)
        recovery_record["bytes"] = 37416

        v2r11_manifest = {
            "release_id": "release_20260921_visdrone_paper_v2r11",
            "run_name": "semuav_visdrone_v2r11",
            "payload_file_count": 109,
            "predecessor_observed_terminal_boundary": r["inherited_v2r10"],
        }
        v2r11_receipt = {
            "schema_version": "semuav.visdrone-v2r11-source-deployment.v1",
            "release_id": "release_20260921_visdrone_paper_v2r11",
            "run_name": "semuav_visdrone_v2r11",
            "runtime_smoke_run": False,
            "scientific_validation_authorized": False,
            "scientific_experiments_run": False,
            "untouched_test_run": False,
            "predecessor_observed_terminal_boundary": r["inherited_v2r10"],
        }
        self._put(
            r["immediate_predecessor_archive_path"], b"v2r11 archive",
            claimed_sha256=r["immediate"]["archive_sha256"],
        )
        self._put(
            r["immediate_predecessor_release_root"] / "RELEASE_MANIFEST.json",
            v2r11_manifest,
            claimed_sha256=r["immediate"]["release_manifest_sha256"],
        )
        self._put(
            r["immediate_predecessor_release_root"] / "RELEASE_MANIFEST_SHA256.txt",
            b"v2r11 checksums\n",
            claimed_sha256=r["immediate"]["checksum_inventory_sha256"],
        )
        self._put(
            r["immediate_predecessor_deployer_path"], b"v2r11 deployer\n",
            claimed_sha256=r["immediate"]["deployer_sha256"],
        )
        self._put(
            r["immediate_predecessor_deployment_receipt_path"], v2r11_receipt,
            claimed_sha256=r["immediate"]["deployment_receipt_sha256"],
        )

        inventory = [
            {"path": "build_visdrone_paper_notebook.py", "bytes": 1,
             "sha256": r["source"]["builder_sha256"], "role": "source"},
            {"path": "SemUAV_VLM_VisDrone_Paper_Study.ipynb", "bytes": 1,
             "sha256": r["source"]["notebook_sha256"], "role": "notebook"},
        ]
        inventory.extend({
            "path": f"fixture/{index:03d}.txt", "bytes": 1,
            "sha256": f"{index:064x}"[-64:], "role": "fixture",
        } for index in range(106))
        v2r10_manifest = {
            "schema_version": "semuav.visdrone-paper-release.v1",
            "release_id": "release_20260921_visdrone_paper_v2r10",
            "run_name": "semuav_visdrone_v2r10",
            "archive_name": r["source_predecessor_archive_path"].name,
            "payload_file_count": 108,
            "archive_member_count": 110,
            "payload_total_uncompressed_bytes": 4009068,
            "empirical_results_included": False,
            "dataset_material_included": False,
            "payload_inventory": inventory,
            "predecessor_observed_terminal_boundary": r["inherited_v2r9"],
        }
        v2r10_receipt = {
            "schema_version": "semuav.visdrone-v2r10-source-deployment.v1",
            "status": "READY_FOR_DEVELOPMENT_SMOKE_ONLY",
            "release_id": "release_20260921_visdrone_paper_v2r10",
            "run_name": "semuav_visdrone_v2r10",
            "release_root": str(r["source_predecessor_release_root"]),
            "scientific_validation_authorized": False,
            "scientific_experiments_run": False,
            "runtime_smoke_run": False,
            "untouched_test_run": False,
            "validation_or_test_images_or_labels_read": False,
            "experiment1_recovery": recovery_record,
            "predecessor_observed_terminal_boundary": r["inherited_v2r9"],
        }
        for path, payload, digest in (
            (r["source_predecessor_archive_path"], b"v2r10 archive", r["source"]["archive_sha256"]),
            (r["source_predecessor_release_root"] / "RELEASE_MANIFEST.json", v2r10_manifest, r["source"]["release_manifest_sha256"]),
            (r["source_predecessor_release_root"] / "RELEASE_MANIFEST_SHA256.txt", b"v2r10 checksums\n", r["source"]["checksum_inventory_sha256"]),
            (r["source_predecessor_release_root"] / "requirements.txt", b"requirements\n", r["source"]["requirements_sha256"]),
            (r["source_predecessor_deployer_path"], b"v2r10 deployer\n", r["source"]["deployer_sha256"]),
            (r["source_predecessor_deployment_receipt_path"], v2r10_receipt, r["source"]["deployment_receipt_sha256"]),
        ):
            self._put(path, payload, claimed_sha256=digest)

        sealed = r["sealed"]
        sealed_root = Path(sealed["release_path"])
        sealed_manifest = {
            "release_id": sealed["release_id"], "run_name": sealed["run_name"],
            "authenticated_predecessor_v2r9": self.predecessor_v2r9,
            "authenticated_predecessor_v2r10": r["source"],
            "authenticated_predecessor_v2r11": r["immediate"],
            "predecessor_observed_terminal_boundary": r["historical_v2r11"],
        }
        sealed_receipt = {**sealed_manifest,
            "schema_version": "semuav.visdrone-v2r12-source-deployment.v1",
            "runtime_smoke_run": False, "scientific_experiments_run": False,
            "untouched_test_run": False,
        }
        for path_key, digest_key in (
            ("archive_path", "archive_sha256"), ("deployer_path", "deployer_sha256"),
            ("reused_artifacts_receipt_path", "reused_artifacts_receipt_sha256"),
            ("inference_environment_fingerprint_path", "inference_environment_fingerprint_sha256"),
            ("inference_pip_freeze_path", "inference_pip_freeze_sha256"),
            ("runtime_smoke_input_path", "runtime_smoke_input_sha256"),
            ("runtime_smoke_report_path", "runtime_smoke_report_sha256"),
            ("active_model_lock_path", "active_model_lock_sha256"),
            ("model_lock_derivation_path", "model_lock_derivation_sha256"),
            ("trusted_code_cache_receipt_path", "trusted_code_cache_receipt_sha256"),
            ("scientific_environment_seal_path", "scientific_environment_seal_sha256"),
        ):
            self._put(Path(sealed[path_key]), b"sealed fixture", claimed_sha256=sealed[digest_key])
        self._put(Path(sealed["deployment_receipt"]), sealed_receipt,
                  claimed_sha256=sealed["deployment_receipt_sha256"])
        for name, digest_key, payload in (
            ("RELEASE_MANIFEST.json", "release_manifest_sha256", sealed_manifest),
            ("RELEASE_MANIFEST_SHA256.txt", "checksum_inventory_sha256", b"sealed checksums"),
            ("build_visdrone_paper_notebook.py", "builder_sha256", b"sealed builder"),
            ("requirements.txt", "requirements_sha256", b"sealed requirements"),
        ):
            self._put(sealed_root / name, payload, claimed_sha256=sealed[digest_key])

        self._build_v2r9_acquisitions(r)
        self._build_current_acquisitions(r, recovery_record)

    def _build_v2r9_acquisitions(self, r: dict[str, object]) -> None:
        evidence = self.predecessor_v2r9
        v2r9_receipt = {
            "schema_version": "semuav.visdrone-v2r9-source-deployment.v1",
            "status": "READY_FOR_DEVELOPMENT_SMOKE_ONLY",
            "release_id": "release_20260920_visdrone_paper_v2r9",
            "run_name": "semuav_visdrone_v2r9",
            "release_root": str(r["predecessor_release_root"]),
            "scientific_validation_authorized": False,
            "scientific_experiments_run": False,
            "runtime_smoke_run": False,
            "validation_or_test_images_or_labels_read": False,
            "gpu_accessed": False,
            "model_weights_loaded": False,
            "jupyter_launched": False,
            "source_compatibility": {},
        }
        workers = [{"model_id": model_id} for model_id in MODEL_IDS]
        smoke = {
            "schema_version": 4,
            "all_passed": True,
            "models": [{"model_id": model_id, "status": "PASS"} for model_id in MODEL_IDS],
            "execution_isolation": {
                "schema_version": "semuav.runtime-smoke-subprocess.v2",
                "artifacts": workers,
                "artifacts_sha256": _hash_object(workers),
            },
            "model_lock_sha256": evidence["active_model_lock_sha256"],
            "model_set_sha256": evidence["active_model_set_sha256"],
        }
        model_records = [{"model_id": model_id} for model_id in MODEL_IDS]
        smoke_input = {
            "schema_version": "semuav.model-runtime-smoke-input.v3",
            "scientific_results_eligible": False,
            "excluded_model_ids": ["moondream2"],
            "model_lock_sha256": evidence["active_model_lock_sha256"],
            "registered_model_set": {
                "model_set_sha256": evidence["active_model_set_sha256"],
                "models": model_records,
            },
        }
        lock = {"model_set_sha256": evidence["active_model_set_sha256"], "models": model_records}
        derivation = {
            "derived_model_lock_sha256": evidence["active_model_lock_sha256"],
            "derived_model_set_sha256": evidence["active_model_set_sha256"],
            "prior_runtime_smoke_rows_reused": False,
            "validation_or_test_evidence_used": False,
        }
        trusted = {
            "schema_version": "semuav.visdrone-trusted-code-cache.v2",
            "scope": "no_active_trusted_remote_code",
            "model_lock_sha256": evidence["active_model_lock_sha256"],
            "model_lock_derivation_sha256": evidence["model_lock_derivation_sha256"],
            "registered_model_ids": list(MODEL_IDS),
            "excluded_model_ids": ["moondream2"],
            "module_roots": [], "files": [], "file_count": 0,
            "tree_sha256": _hash_object([]),
        }
        seal = {
            "schema_version": "semuav.visdrone-inference-environment-seal.v1",
            "status": "SEALED_FOR_SCIENTIFIC_USE",
            "scientific_results_authorized": True,
            "validation_or_test_data_read_before_seal": False,
            "registered_model_ids": list(MODEL_IDS),
            "excluded_model_ids": ["moondream2"],
            "runtime_smoke": {"sha256": evidence["runtime_smoke_report_sha256"], "all_passed": True},
            "trusted_code_cache": {"sha256": evidence["trusted_code_cache_receipt_sha256"], "file_count": 0},
            "reused_v1_frozen_artifacts_receipt": {"sha256": evidence["reused_artifacts_receipt_sha256"]},
        }
        payloads = {
            evidence["archive_path"]: b"v2r9 archive",
            evidence["deployer_path"]: b"v2r9 deployer",
            evidence["deployment_receipt"]: v2r9_receipt,
            str(r["predecessor_release_root"] / "RELEASE_MANIFEST.json"): {},
            str(r["predecessor_release_root"] / "RELEASE_MANIFEST_SHA256.txt"): b"v2r9 checksums",
            evidence["reused_artifacts_receipt_path"]: {},
            evidence["inference_environment_fingerprint_path"]: {},
            evidence["inference_pip_freeze_path"]: b"historical-freeze\n",
            evidence["runtime_smoke_input_path"]: smoke_input,
            evidence["runtime_smoke_report_path"]: smoke,
            evidence["active_model_lock_path"]: lock,
            evidence["model_lock_derivation_path"]: derivation,
            evidence["trusted_code_cache_receipt_path"]: trusted,
            evidence["scientific_environment_seal_path"]: seal,
        }
        digest_keys = {
            evidence["archive_path"]: "archive_sha256",
            evidence["deployer_path"]: "deployer_sha256",
            evidence["deployment_receipt"]: "deployment_receipt_sha256",
            str(r["predecessor_release_root"] / "RELEASE_MANIFEST.json"): "release_manifest_sha256",
            str(r["predecessor_release_root"] / "RELEASE_MANIFEST_SHA256.txt"): "checksum_inventory_sha256",
            evidence["reused_artifacts_receipt_path"]: "reused_artifacts_receipt_sha256",
            evidence["inference_environment_fingerprint_path"]: "inference_environment_fingerprint_sha256",
            evidence["inference_pip_freeze_path"]: "inference_pip_freeze_sha256",
            evidence["runtime_smoke_input_path"]: "runtime_smoke_input_sha256",
            evidence["runtime_smoke_report_path"]: "runtime_smoke_report_sha256",
            evidence["active_model_lock_path"]: "active_model_lock_sha256",
            evidence["model_lock_derivation_path"]: "model_lock_derivation_sha256",
            evidence["trusted_code_cache_receipt_path"]: "trusted_code_cache_receipt_sha256",
            evidence["scientific_environment_seal_path"]: "scientific_environment_seal_sha256",
        }
        for raw_path, payload in payloads.items():
            self._put(Path(raw_path), payload, claimed_sha256=evidence[digest_keys[raw_path]])

        predecessor_source = b'''\
NATIVE_RECEIVER_FUSION_POLICY_VERSION = "semuav.visdrone-native-fusion.v2"
def native_receiver_fusion_receipt():
    return {"policy": "legacy"}
def registered_receiver_context():
    return ReceiverControlContext(mission_profile=None, registered_roi_xyxy=None)
class RegisteredVisDroneNativeReceiver:
    def __call__(self, decoded):
        return decoded
    def _frame_action(self, decoded):
        return InferenceRequest("id", "image", "mission", 0, {})
'''
        self._put(
            r["predecessor_release_root"] / "semuav" / "visdrone_receiver_runtime.py",
            predecessor_source,
            claimed_sha256="28ab74b2856bf52103c6a3d9191bfc51c837075d5025ccf0d2222590a759f32e",
        )

    def _build_current_acquisitions(
        self, r: dict[str, object], recovery_record: dict[str, object],
    ) -> None:
        requirements = (ROOT / "requirements.txt").read_bytes()
        receiver = (ROOT / "semuav" / "visdrone_receiver_runtime.py").read_bytes()
        runtime_record = (ROOT / "PRE_VALIDATION_NATIVE_RECEIVER_RECOVERY_20260921_V2R10.md").read_bytes()
        acceptance_record = (ROOT / "PRE_VALIDATION_ACCEPTANCE_RECOVERY_ISOLATION_20260921_V2R11.md").read_bytes()
        boundary_record = (ROOT / "PRE_SMOKE_PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_20260921_V2R12.md").read_bytes()
        low_record = (ROOT / "PRE_VALIDATION_LOW_EVIDENCE_PACKET_CORRECTION_20260921_V2R13.md").read_bytes()
        paths_and_bytes = {
            "PRE_VALIDATION_LOW_EVIDENCE_PACKET_CORRECTION_20260921_V2R13.md": low_record,
            "requirements.txt": requirements,
            "semuav/visdrone_receiver_runtime.py": receiver,
            "PRE_VALIDATION_NATIVE_RECEIVER_RECOVERY_20260921_V2R10.md": runtime_record,
            "PRE_VALIDATION_ACCEPTANCE_RECOVERY_ISOLATION_20260921_V2R11.md": acceptance_record,
            "PRE_SMOKE_PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_20260921_V2R12.md": boundary_record,
        }
        inventory = []
        for relative, payload in paths_and_bytes.items():
            path = self.release_root / relative
            self._put(path, payload)
            inventory.append({
                "path": relative, "bytes": len(payload),
                "sha256": _real_hashlib.sha256(payload).hexdigest(),
                "role": "fixture",
            })
        manifest = {
            "schema_version": "semuav.visdrone-paper-release.v1",
            "release_id": "release_20260921_visdrone_paper_v2r13",
            "run_name": "semuav_visdrone_v2r13",
            "archive_name": r["archive_path"].name,
            "empirical_results_included": False,
            "dataset_material_included": False,
            "payload_file_count": len(inventory),
            "payload_inventory": inventory,
            "validation": {
                "unit_tests": {"run": 1, "passed": 1, "failed": 0},
                "unit_subtests": {"run": 1, "passed": 1, "failed": 0},
            },
            "methodology": {
                "native_receiver_correction": "PRE_VALIDATION_NATIVE_RECEIVER_RECOVERY_20260921_V2R10.md",
            },
            "pre_validation_native_receiver_correction": "PRE_VALIDATION_NATIVE_RECEIVER_RECOVERY_20260921_V2R10.md",
            "pre_validation_acceptance_recovery_isolation": "PRE_VALIDATION_ACCEPTANCE_RECOVERY_ISOLATION_20260921_V2R11.md",
            "pre_smoke_predecessor_boundary_routing_correction": "PRE_SMOKE_PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_20260921_V2R12.md",
            "authenticated_predecessor_v2r9": self.predecessor_v2r9,
            "authenticated_predecessor_v2r10": r["source"],
            "authenticated_predecessor_v2r11": r["immediate"],
            "authenticated_predecessor_v2r12": r["sealed"],
            "pre_validation_low_evidence_packet_correction": "PRE_VALIDATION_LOW_EVIDENCE_PACKET_CORRECTION_20260921_V2R13.md",
            "predecessor_observed_terminal_boundary": r["top_boundary"],
            "model_set_amendment": {},
            "temporal_feasibility_amendment": {
                "fresh_runtime_smoke_required": True,
                "predecessor_smoke_rows_reused": False,
            },
            "source_compatibility": _source_compatibility(),
        }
        manifest_bytes = _canonical_bytes(manifest)
        manifest_sha = _real_hashlib.sha256(manifest_bytes).hexdigest()
        checksum_lines = [f"{item['sha256']}  {item['path']}" for item in inventory]
        checksum_lines.append(f"{manifest_sha}  RELEASE_MANIFEST.json")
        checksums = ("\n".join(sorted(checksum_lines)) + "\n").encode("utf-8")
        checksums_sha = _real_hashlib.sha256(checksums).hexdigest()
        requirements_sha = _real_hashlib.sha256(requirements).hexdigest()
        self._put(self.release_root / "RELEASE_MANIFEST.json", manifest_bytes)
        self._put(self.release_root / "RELEASE_MANIFEST_SHA256.txt", checksums)
        candidate_source = r["candidate_source_root"]
        candidate_manifest = {
            "release_id": "release_20260912_visdrone_paper_v2r3",
            "run_name": "semuav_visdrone_v2r3",
            "empirical_results_included": False,
            "dataset_material_included": False,
        }
        candidate_manifest_bytes = self._put(
            candidate_source / "RELEASE_MANIFEST.json", candidate_manifest,
        )
        candidate_checksums = self._put(
            candidate_source / "RELEASE_MANIFEST_SHA256.txt", b"candidate checksums\n",
        )
        self._put(candidate_source / "requirements.txt", requirements)
        builder_bytes = self._put(r["builder_path"], b"candidate builder\n",
            claimed_sha256="01831628385717ff8db927ea80e617060fbf8b51ea4d8d851ef4ad942d5da494")
        deployer_bytes = self._put(r["deployer_path"], b"v2r13 deployer\n")
        archive_bytes = self._put(r["archive_path"], b"v2r13 archive\n")
        pip_path = self.candidate_root / "pip-freeze.txt"
        pip_bytes = self._put(pip_path, b"fixture==1\n")
        self.pip_bytes = bytes(pip_bytes)
        candidate_ready = {
            "schema_version": "semuav.inference-candidate-ready.v1",
            "status": "READY",
            "candidate_only": True,
            "scientific_results_authorized": False,
            "validation_or_test_data_read": False,
            "legacy_environment_invoked": False,
            "canonical_release_modified": False,
            "candidate_environment": str(self.candidate_root),
            "builder": {"path": str(r["builder_path"]), "sha256": _AcquisitionHashlib.sha256(builder_bytes).hexdigest()},
            "source_release": {
                "path": str(candidate_source),
                "release_id": "release_20260912_visdrone_paper_v2r3",
                "run_name": "semuav_visdrone_v2r3",
                "manifest_sha256": _real_hashlib.sha256(candidate_manifest_bytes).hexdigest(),
                "checksum_inventory_sha256": _real_hashlib.sha256(candidate_checksums).hexdigest(),
                "requirements_sha256": requirements_sha,
            },
            "predecessor_environment_equivalence": {
                "predecessor_candidate_environment": str(r["predecessor_candidate"]),
                "pip_freeze_sha256": _real_hashlib.sha256(pip_bytes).hexdigest(),
                "equivalence_scope": "installed_distribution_inventory_only",
            },
            "requested_pins": {"fixture": True},
            "validated_runtime": {"fixture": True},
            "runtime_guards": {
                "YOLO_AUTOINSTALL": "false", "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "PIP_NO_INDEX_during_validation": "1",
                "CUDA_VISIBLE_DEVICES_during_build": "",
                "NVIDIA_VISIBLE_DEVICES_during_build": "void",
                "inherited_loader_environment_cleared": True,
            },
            "artifacts": {"pip_freeze": {
                "path": str(pip_path),
                "sha256": _real_hashlib.sha256(pip_bytes).hexdigest(),
            }},
            "build_record_directory": "/fixture", "build_log": {},
            "reproducibility_boundary": {
                "post_install_freeze_recorded": True,
                "scientific_use_requires_later_exact_freeze_runtime_fingerprint_and_smoke_seal": True,
            },
        }
        candidate_ready_bytes = self._put(
            self.candidate_root / "SEMUAV_CANDIDATE_READY.json", candidate_ready,
        )
        kernel_path = self._path(
            ".cache", "jupyter", "data", "kernels",
            "semuav-vlm-20260921-v2r13", "kernel.json",
        )
        kernel = {
            "argv": [str(self.candidate_python.absolute()), "-I", "-B", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Python (SemUAV-VLM 2026-09-21 v2r13)",
            "language": "python", "metadata": {"debugger": True, "supported_encryption": "curve"},
            "kernel_protocol_version": "5.5",
        }
        kernel_bytes = self._put(kernel_path, kernel)
        junit_path = self.study_root / "private" / "fixture.junit.xml"
        junit_bytes = self._put(junit_path, b"<testsuites/>\n")
        artifact_records = {
            "created_at_utc": "2026-09-21T00:00:00Z",
            "release_root": str(self.release_root),
            "deployment_mode": "NEW_ATOMIC_DEPLOYMENT",
            "release": {},
            "installer": {"path": str(r["deployer_path"]), "sha256": _real_hashlib.sha256(deployer_bytes).hexdigest()},
            "deployment_script": {"path": str(r["deployer_path"]), "sha256": _real_hashlib.sha256(deployer_bytes).hexdigest()},
            "builder": {"path": str(r["builder_path"]), "sha256": _AcquisitionHashlib.sha256(builder_bytes).hexdigest()},
            "source_release": {
                "archive_path": str(r["archive_path"]),
                "archive_sha256": _real_hashlib.sha256(archive_bytes).hexdigest(),
                "manifest_sha256": manifest_sha,
                "checksum_inventory_sha256": checksums_sha,
                "requirements_sha256": requirements_sha,
            },
            "authenticated_base_v2r3": {},
            "candidate_environment": {
                "path": str(self.candidate_root),
                "ready_receipt": str(self.candidate_root / "SEMUAV_CANDIDATE_READY.json"),
                "ready_receipt_sha256": _real_hashlib.sha256(candidate_ready_bytes).hexdigest(),
                "pip_freeze_sha256": _real_hashlib.sha256(pip_bytes).hexdigest(),
            },
            "unit_tests": {
                "primary": {"run": 1, "passed": 1, "failed": 0},
                "subtests": {"run": 1, "passed": 1, "failed": 0},
                "junit": {"path": str(junit_path), "sha256": _real_hashlib.sha256(junit_bytes).hexdigest()},
            },
            "jupyter_kernel": {
                "name": "semuav-vlm-20260921-v2r13",
                "display_name": "Python (SemUAV-VLM 2026-09-21 v2r13)",
                "kernel_json": str(kernel_path),
                "kernel_json_sha256": _real_hashlib.sha256(kernel_bytes).hexdigest(),
            },
            "jupyter_launch_target": {}, "trusted_code_cache_namespace": {},
            "attempt_directory": "/fixture/attempt", "deployment_log": {},
            "experiment1_recovery": recovery_record,
            "predecessor_v2r12_state": {
                "schema_version": "semuav.visdrone-v2r12-preserved-state.v1",
                "release_root": r["sealed"]["release_path"],
                "output_root": str(self.study_root / "outputs/visdrone_paper/semuav_visdrone_v2r12"),
                "cache_root": str(self.study_root / ".cache/visdrone_paper_v2r12"),
                "trusted_code_cache_root": str(self.study_root / ".cache/huggingface/modules-v2r12"),
                "release_tree": "a" * 64 + " 1 1", "output_tree": "b" * 64 + " 1 1",
                "cache_tree": "c" * 64 + " 1 1", "trusted_code_cache_tree": "d" * 64 + " 0 0",
                "unchanged_after_deployment_actions": True,
            },
        }
        self.manifest = manifest
        self.receipt = notebook_builder._v2r13_deployment_receipt_fixture(
            release_manifest=manifest, artifact_records=artifact_records,
        )
        self.receipt_path = r["deployment_receipt_path"]

    def verify(self, receipt: dict[str, object] | None = None) -> dict[str, object]:
        receipt = copy.deepcopy(self.receipt if receipt is None else receipt)
        receipt_bytes = _canonical_bytes(receipt)
        self._put(self.receipt_path, receipt_bytes)
        os.environ["SEMUAV_V2R13_DEPLOYMENT_RECEIPT_SHA256"] = (
            _real_hashlib.sha256(receipt_bytes).hexdigest()
        )
        recovery = receipt["experiment1_recovery"]
        recovery_env = {
            "SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT": str(recovery["path"]),
            "SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT_SHA256": str(recovery["sha256"]),
        }
        previous_recovery_env = {
            key: os.environ.get(key) for key in recovery_env
        }
        os.environ.update(recovery_env)
        function = copy.deepcopy(self.function)
        rewriter = _StudyRootRewriter(str(self.study_root))
        function = rewriter.visit(function)
        if rewriter.replacements != 1:
            raise AssertionError("generated verifier study-root literal changed")
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))

        def read_regular(path, *, label):
            key = str(Path(path).resolve())
            if key not in self.registry:
                raise AssertionError(f"unregistered acquisition {label}: {key}")
            return self.registry[key]

        def decode(payload, *, label):
            try:
                value = json.loads(bytes(payload).decode("utf-8"))
            except Exception as exc:  # pragma: no cover - diagnostic guard
                raise StudyGateError(f"{label} is not JSON") from exc
            if not isinstance(value, dict):
                raise StudyGateError(f"{label} is not one object")
            return value

        namespace = {
            "Path": Path, "PurePosixPath": PurePosixPath, "ast": ast,
            "hashlib": _AcquisitionHashlib, "os": os,
            "sys": SimpleNamespace(executable=str(self.candidate_python), prefix=str(self.candidate_root)),
            "subprocess": SimpleNamespace(run=lambda *args, **kwargs: SimpleNamespace(stdout=self.pip_bytes)),
            "StudyGateError": StudyGateError,
            "_verified_runtime_guards": lambda: None,
            "_read_regular_file_bytes": read_regular,
            "_decode_json_object_bytes": decode,
            "hash_object": _hash_object,
            "require_runtime_smoke_pass": lambda path: None,
            "PROJECT_ROOT": self.release_root,
            "CANDIDATE_READY_RECEIPT": self.candidate_root / "SEMUAV_CANDIDATE_READY.json",
            "INFERENCE_PIP_FREEZE": self.release_root / "absent-inference-freeze.txt",
            "PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_RECORD": ROOT / "PRE_SMOKE_PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_20260921_V2R12.md",
            "RUNTIME_CORRECTION_RECORD": ROOT / "PRE_VALIDATION_NATIVE_RECEIVER_RECOVERY_20260921_V2R10.md",
            "ACCEPTANCE_RECOVERY_ISOLATION_RECORD": ROOT / "PRE_VALIDATION_ACCEPTANCE_RECOVERY_ISOLATION_20260921_V2R11.md",
            "TEMPORAL_FEASIBILITY_AMENDMENT_RECORD": ROOT / "PRE_EXPERIMENT_TEMPORAL_FEASIBILITY_AMENDMENT_20260914_V2R7.md",
            "predecessor_boundary_routing_correction_sha256": self.records["correction_sha"],
            "LOW_EVIDENCE_PACKET_CORRECTION_RECORD": ROOT / "PRE_VALIDATION_LOW_EVIDENCE_PACKET_CORRECTION_20260921_V2R13.md",
            "low_evidence_packet_correction_sha256": self.records["low_sha"],
            "runtime_correction_record_sha256": _real_hashlib.sha256((ROOT / "PRE_VALIDATION_NATIVE_RECEIVER_RECOVERY_20260921_V2R10.md").read_bytes()).hexdigest(),
            "acceptance_recovery_isolation_record_sha256": _real_hashlib.sha256((ROOT / "PRE_VALIDATION_ACCEPTANCE_RECOVERY_ISOLATION_20260921_V2R11.md").read_bytes()).hexdigest(),
            "temporal_feasibility_amendment": self.manifest.get("temporal_feasibility_amendment", {}),
            "REGISTERED_RUNTIME_SMOKE_MODEL_IDS": MODEL_IDS,
            "RUNTIME_SMOKE_REPORT_SCHEMA_VERSION": 4,
            "RUNTIME_SMOKE_ISOLATION_SCHEMA": "semuav.runtime-smoke-subprocess.v2",
            "VISDRONE_GENERATION_POLICY_VERSION": self.manifest["source_compatibility"]["generation_policy_version"],
            "VISDRONE_GENERATION_POLICY_SHA256": self.manifest["source_compatibility"]["generation_policy_sha256"],
            "cfg": SimpleNamespace(
                runtime={
                    "candidate_ready_receipt": str(self.candidate_root / "SEMUAV_CANDIDATE_READY.json"),
                    "required_inference_python": str(self.candidate_python),
                    "candidate_ready_receipt_schema": "semuav.inference-candidate-ready.v1",
                    "candidate_ready_status": "READY",
                    "inference_environment_seal_schema": "semuav.visdrone-inference-environment-seal.v1",
                    "inference_environment_seal_status": "SEALED_FOR_SCIENTIFIC_USE",
                },
                models={"semantic_contract": "semuav.visdrone-proposal-decision.v2", "max_new_tokens": self.manifest["source_compatibility"]["model_max_new_tokens"]},
            ),
        }
        try:
            exec(compile(module, "<generated-candidate-verifier>", "exec"), namespace)
            return namespace["_verify_candidate_ready_receipt"](
                require_inference_lock=False,
            )
        finally:
            for key, value in previous_recovery_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class V2R13BoundaryRoutingRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.previous_pin = os.environ.get("SEMUAV_V2R13_DEPLOYMENT_RECEIPT_SHA256")
        self.addCleanup(self._restore_pin)
        self.harness = VerifierHarness(self.temp.name)

    def _restore_pin(self) -> None:
        if self.previous_pin is None:
            os.environ.pop("SEMUAV_V2R13_DEPLOYMENT_RECEIPT_SHA256", None)
        else:
            os.environ["SEMUAV_V2R13_DEPLOYMENT_RECEIPT_SHA256"] = self.previous_pin

    def test_exact_fixture_passes_real_generated_verifier(self) -> None:
        self.assertEqual(len(self.harness.receipt), 52)
        self.assertEqual(
            self.harness.receipt["predecessor_observed_terminal_boundary"],
            notebook_builder._v2r13_predecessor_observed_terminal_boundary_fixture(
                self.harness.records["low_sha"],
            ),
        )
        result = self.harness.verify()
        self.assertEqual(
            result["source_compatibility"][
                "predecessor_observed_terminal_boundary"
            ],
            self.harness.receipt["predecessor_observed_terminal_boundary"],
        )
        self.assertEqual(result["source_compatibility"]["run_name"], "semuav_visdrone_v2r13")

    def test_semantic_boundary_mutations_fail_after_receipt_repin(self) -> None:
        mutations = []
        wrong_layer = copy.deepcopy(self.harness.receipt)
        wrong_layer["predecessor_observed_terminal_boundary"] = copy.deepcopy(
            self.harness.records["inherited_v2r9"]
        )
        mutations.append(("wrong_historical_layer", wrong_layer))

        hybrid = copy.deepcopy(self.harness.receipt)
        hybrid_boundary = hybrid["predecessor_observed_terminal_boundary"]
        hybrid_boundary["v2r10_fresh_runtime_smoke_required"] = True
        mutations.append(("hybrid_outer_key", hybrid))

        branch_paths = {
            "outer": (),
            "source": ("source_authenticated_defect",),
            "execution": ("operator_observed_execution",),
        }
        canonical_boundary = self.harness.receipt[
            "predecessor_observed_terminal_boundary"
        ]
        for branch, path in branch_paths.items():
            canonical_target = canonical_boundary
            for part in path:
                canonical_target = canonical_target[part]
            for key in canonical_target:
                missing = copy.deepcopy(self.harness.receipt)
                target = missing["predecessor_observed_terminal_boundary"]
                for part in path:
                    target = target[part]
                target.pop(key)
                mutations.append((f"missing_{branch}_{key}", missing))

            extra = copy.deepcopy(self.harness.receipt)
            target = extra["predecessor_observed_terminal_boundary"]
            for part in path:
                target = target[part]
            target["unbound_fixture_field"] = False
            mutations.append((f"extra_{branch}", extra))

        field_mutations = (
            ("schema", ("schema_version",), "semuav.wrong.v1"),
            ("predecessor", ("predecessor_release_id",), "release_wrong"),
            ("correction_digest", ("correction_record_sha256",), "0" * 64),
            ("smoke", ("operator_observed_execution", "runtime_smoke_three_of_three_passed"), False),
            ("recovery_rows", ("operator_observed_execution", "v2r9_experiment1_recovery_row_count"), 4151),
            ("auth_flag", ("v2r13_raw_experiment1_recovery_authorized",), False),
        )
        for label, path, value in field_mutations:
            mutated = copy.deepcopy(self.harness.receipt)
            target = mutated["predecessor_observed_terminal_boundary"]
            for part in path[:-1]:
                target = target[part]
            target[path[-1]] = value
            mutations.append((label, mutated))

        for label, mutated in mutations:
            with self.subTest(label=label):
                with self.assertRaises(StudyGateError):
                    self.harness.verify(mutated)

    def test_jointly_repinned_historical_boundary_mutations_fail(self) -> None:
        layer_cases = {
            "v2r11": (
                ("execution_value", "set", "runtime_smoke_pass_count", 2),
                ("execution_missing", "remove", "report_or_figures_generated", None),
                ("execution_extra", "set", "unbound_fixture_field", False),
            ),
            "v2r10": (
                ("execution_value", "set", "runtime_smoke_passed", False),
                ("execution_missing", "remove", "report_or_figures_generated", None),
                ("execution_extra", "set", "unbound_fixture_field", False),
            ),
        }
        for layer, cases in layer_cases.items():
            for label, action, key, value in cases:
                with self.subTest(layer=layer, mutation=label):
                    harness = VerifierHarness(
                        str(Path(self.temp.name) / f"{layer}-{label}")
                    )
                    if layer == "v2r11":
                        manifest_path = Path(
                            harness.records["immediate_predecessor_release_root"]
                        ) / "RELEASE_MANIFEST.json"
                        receipt_path = Path(
                            harness.records[
                                "immediate_predecessor_deployment_receipt_path"
                            ]
                        )
                        manifest_sha256 = harness.records["immediate"][
                            "release_manifest_sha256"
                        ]
                        receipt_sha256 = harness.records["immediate"][
                            "deployment_receipt_sha256"
                        ]
                    else:
                        manifest_path = Path(
                            harness.records["source_predecessor_release_root"]
                        ) / "RELEASE_MANIFEST.json"
                        receipt_path = Path(
                            harness.records[
                                "source_predecessor_deployment_receipt_path"
                            ]
                        )
                        manifest_sha256 = harness.records["source"][
                            "release_manifest_sha256"
                        ]
                        receipt_sha256 = harness.records["source"][
                            "deployment_receipt_sha256"
                        ]

                    for path, claimed_sha256 in (
                        (manifest_path, manifest_sha256),
                        (receipt_path, receipt_sha256),
                    ):
                        payload = json.loads(path.read_text(encoding="utf-8"))
                        execution = payload[
                            "predecessor_observed_terminal_boundary"
                        ]["operator_observed_execution"]
                        if action == "remove":
                            execution.pop(key)
                        else:
                            execution[key] = value
                        harness._put(
                            path, payload, claimed_sha256=str(claimed_sha256),
                        )

                    with self.assertRaises(StudyGateError):
                        harness.verify()


if __name__ == "__main__":
    unittest.main()
