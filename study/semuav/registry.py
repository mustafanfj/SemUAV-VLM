from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pandas as pd


class StudyGateError(RuntimeError):
    """A required scientific or operational prerequisite is not satisfied."""


EMBEDDED_HANDOFF_REQUIRED_ARTIFACTS = (
    "test_detector_residency.json",
    "test_exp1_perception_detail.csv.gz",
    "test_exp1_perception_summary.csv.gz",
    "test_temporal_exp1_perception_detail.csv.gz",
    "test_temporal_exp1_perception_summary.csv.gz",
    "test_sample_cluster_manifest.csv.gz",
    "test_temporal_cluster_manifest.csv.gz",
    "test_temporal_sample_gate.csv.gz",
    "test_dataset_summary.csv.gz",
    "test_exp2_representation_frontier.csv.gz",
    "test_temporal_exp2_representation_frontier.csv.gz",
    "test_exp3_static_partition_index.csv.gz",
    "test_exp3_static_summary.csv.gz",
    "test_policy_partition_index.csv",
    "policy_pair_cells.csv.gz",
    "h3_negative_episode_cells.csv.gz",
    "resource_summary.csv.gz",
    "resource_rate_summary.csv.gz",
    "resource_episode_cells.csv.gz",
    "resource_cluster_cells.csv.gz",
    "controller_summary.csv.gz",
    "exp5_summary.csv.gz",
    "h4_matched_coverage_cells.csv.gz",
    "h4_matched_coverage_audit.csv.gz",
    "h4_per_trace_audit.csv.gz",
    "h4_trace_spill_index.csv.gz",
    "pairing_audit.csv.gz",
    "io_audit.csv.gz",
    "policy_reduction_receipt.json",
    "test_dynamic_cartesian_coverage_audit.csv.gz",
    "test_controller_behavior_summary.csv.gz",
    "test_exp5_risk_uncertainty_summary.csv.gz",
    "test_budget_compliance.csv.gz",
    "confirmatory_gate_table.csv.gz",
    "h1_semantic_noninferiority.csv.gz",
    "test_h1_fixed_grid_group_aggregates.csv.gz",
    "test_h1_fixed_grid_coverage_audit.csv.gz",
    "test_h2_matched_budget.csv.gz",
    "h2_confirmatory_report.csv.gz",
    "h3_mission_awareness.csv.gz",
    "h4_uncertainty_escalation.csv.gz",
    "test_h4_matched_coverage_gate.csv.gz",
    "test_receiver_robustness_detail.csv.gz",
    "test_receiver_robustness_summary.csv.gz",
    "test_receiver_state_strategy_detail.csv.gz",
    "test_receiver_state_strategy_summary.csv.gz",
    "hypothesis_status.csv.gz",
    "final_completion_status.csv.gz",
)
RESTRICTED_TEST_EVIDENCE_SCHEMA = "semuav.restricted-test-evidence.v1"
RESTRICTED_TEST_EVIDENCE_RECEIPT = "restricted_test_evidence_receipt.json"


@dataclasses.dataclass(frozen=True)
class FinalizedTestHandoff:
    """Verified read-only receipt for post-test embedded evaluation."""

    protocol: Mapping[str, Any]
    ledger: Mapping[str, Any]
    ledger_path: Path
    artifacts: Mapping[str, Path]
    restricted_evidence_receipt_path: Path | None = None
    restricted_artifacts: Mapping[str, Path] = dataclasses.field(default_factory=dict)

    def artifact(self, filename: str) -> Path:
        try:
            return self.artifacts[filename]
        except KeyError as exc:
            raise StudyGateError(
                f"Finalized test handoff does not contain required artifact {filename!r}"
            ) from exc


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_object(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def hash_file(path: str | Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def notebook_source_hash(path: str | Path) -> str:
    """Hash notebook cell sources while ignoring execution counts and outputs."""
    notebook_path = Path(path)
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = [
        {"cell_type": cell.get("cell_type"), "source": cell.get("source", "")}
        for cell in payload.get("cells", [])
        if "parameters" not in cell.get("metadata", {}).get("tags", [])
    ]
    return hash_object({"nbformat": payload.get("nbformat"), "cells": sources})


def frozen_file_manifest(paths: Mapping[str, str | Path | None]) -> dict[str, dict[str, Any]]:
    """Create a portable checksum manifest for every outcome-affecting file.

    The paths are recorded as resolved paths for auditability.  Scientific
    callers should include data/split/annotation/trace manifests, checkpoints,
    receiver caches, prompts, schemas, and validation-selected artifacts.
    """
    result: dict[str, dict[str, Any]] = {}
    for name, value in sorted(paths.items()):
        if value is None:
            raise StudyGateError(f"Frozen artifact {name!r} has no configured path")
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise StudyGateError(f"Frozen artifact {name!r} is missing: {path}")
        result[str(name)] = {
            "path": str(path),
            "sha256": hash_file(path),
            "bytes": path.stat().st_size,
        }
    return result


def verify_frozen_file_manifest(manifest: Mapping[str, Mapping[str, Any]]) -> None:
    """Fail closed if any file protected by a frozen protocol has changed."""
    if not manifest:
        raise StudyGateError("Frozen protocol has no outcome-affecting artifact manifest")
    for name, item in sorted(manifest.items()):
        if not isinstance(item, Mapping):
            raise StudyGateError(f"Frozen artifact {name!r} has invalid manifest metadata")
        path = Path(str(item.get("path", ""))).expanduser().resolve()
        expected = str(item.get("sha256", ""))
        try:
            expected_bytes = int(item.get("bytes", -1))
        except (TypeError, ValueError):
            expected_bytes = -1
        if not path.is_file():
            raise StudyGateError(f"Frozen artifact {name!r} is missing at test access: {path}")
        observed = hash_file(path)
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or observed != expected:
            raise StudyGateError(
                f"Frozen artifact {name!r} changed before test access "
                f"(expected {expected or 'MISSING_HASH'}, observed {observed})"
            )
        if expected_bytes < 0 or path.stat().st_size != expected_bytes:
            raise StudyGateError(
                f"Frozen artifact {name!r} size changed or lacks a valid frozen byte count"
            )


def resolve_frozen_artifact(
    frozen_protocol: Mapping[str, Any],
    artifact_name: str,
    configured_path: str | Path | None = None,
) -> Path:
    """Resolve an outcome-affecting test input from the frozen protocol.

    Test code must not verify one frozen file and subsequently read a different
    path supplied through an environment variable.  This helper returns the
    path recorded in ``decisions.artifact_manifest`` only after checking its
    current digest.  When a current configured path is supplied, both its
    resolved path *and* digest must match the frozen entry.
    """
    manifest = frozen_protocol.get("decisions", {}).get("artifact_manifest", {})
    item = manifest.get(str(artifact_name)) if isinstance(manifest, Mapping) else None
    if not isinstance(item, Mapping):
        raise StudyGateError(
            f"Frozen protocol has no artifact-manifest entry {artifact_name!r}"
        )
    frozen_path = Path(str(item.get("path", ""))).expanduser().resolve()
    expected_hash = str(item.get("sha256", ""))
    if not frozen_path.is_file():
        raise StudyGateError(
            f"Frozen artifact {artifact_name!r} is missing: {frozen_path}"
        )
    observed_hash = hash_file(frozen_path)
    if len(expected_hash) != 64 or observed_hash != expected_hash:
        raise StudyGateError(
            f"Frozen artifact {artifact_name!r} changed "
            f"(expected {expected_hash or 'MISSING_HASH'}, observed {observed_hash})"
        )
    expected_bytes = item.get("bytes")
    if expected_bytes is not None and int(expected_bytes) != frozen_path.stat().st_size:
        raise StudyGateError(f"Frozen artifact {artifact_name!r} size changed")
    if configured_path is not None:
        current_path = Path(configured_path).expanduser().resolve()
        if current_path != frozen_path:
            raise StudyGateError(
                f"Configured {artifact_name!r} path differs from the frozen protocol "
                f"({current_path} != {frozen_path})"
            )
        if not current_path.is_file() or hash_file(current_path) != expected_hash:
            raise StudyGateError(
                f"Configured {artifact_name!r} does not match the frozen checksum"
            )
    return frozen_path


def _required_hypotheses(value: Any) -> set[str]:
    """Expand compact gate labels such as ``H2-H4`` into hypothesis IDs."""
    text = str(value).strip().upper()
    hypotheses: set[str] = set()
    for start, end in re.findall(r"H([1-5])(?:\s*-\s*H?([1-5]))?", text):
        first = int(start)
        last = int(end) if end else first
        if last < first:
            first, last = last, first
        hypotheses.update(f"H{number}" for number in range(first, last + 1))
    return hypotheses


def confirmatory_gate_pass(
    gate_table: pd.DataFrame,
    hypothesis: str,
    *,
    required_policies: Iterable[str] = (),
) -> bool:
    """Return whether every registered gate applicable to a claim passed.

    Policy-specific presence and matched-budget rows are filtered to the
    policies used by that hypothesis.  Both gate kinds must exist for every
    requested policy, preventing a missing row from being mistaken for a pass.
    """
    required_columns = {"gate", "passed", "required_for"}
    if not isinstance(gate_table, pd.DataFrame) or gate_table.empty:
        return False
    if missing := required_columns.difference(gate_table.columns):
        raise ValueError(f"Confirmatory gate table missing columns: {sorted(missing)}")
    hypothesis = str(hypothesis).upper()
    if not re.fullmatch(r"H[1-5]", hypothesis):
        raise ValueError("hypothesis must be H1 through H5")
    applicable = gate_table[gate_table["required_for"].map(
        lambda value: hypothesis in _required_hypotheses(value)
    )].copy()
    policies = tuple(dict.fromkeys(str(policy) for policy in required_policies))
    if "policy" in applicable.columns:
        policy_values = applicable["policy"].fillna("").astype(str)
        applicable = applicable[policy_values.eq("") | policy_values.isin(policies)].copy()
    if applicable.empty or not applicable["passed"].astype(bool).all():
        return False
    if policies:
        if "policy" not in applicable.columns:
            return False
        for policy in policies:
            policy_rows = applicable[applicable["policy"].fillna("").astype(str).eq(policy)]
            gates = set(policy_rows["gate"].astype(str))
            if f"policy_present:{policy}" not in gates or f"matched_budget:{policy}" not in gates:
                return False
    return True


def selected_vlm_claim_gate(selection: Mapping[str, Any]) -> bool:
    """Require a frozen contextual VLM rather than detector-only fallback."""
    selected = str(selection.get("selected_model", "")).strip()
    return (
        bool(selection.get("vlm_claim_enabled", False))
        and bool(selected)
        and selected != "detector_only"
    )


def matched_coverage_semantic_errors(
    frame: pd.DataFrame,
    policies: Iterable[str],
    target_coverage: float,
    *,
    score_column: str = "calibrated_correctness_probability",
    mode_column: str = "selected_mode",
    correct_column: str = "correct_action",
    endpoint_column: str = "matched_coverage_semantic_error",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct a label-blind, equal-coverage H4 endpoint.

    Within every independent visual group and policy, the top-scored semantic
    predictions are selected at a predeclared coverage.  Counts are identical
    between policies in every group; score ties are resolved only by frozen
    paired-design keys.  The returned audit fails when any policy/group lacks
    enough finite calibrated semantic scores.
    """
    policies = tuple(dict.fromkeys(str(policy) for policy in policies))
    if len(policies) < 2:
        raise ValueError("Matched-coverage analysis requires at least two policies")
    if not 0 < float(target_coverage) <= 1:
        raise ValueError("target_coverage must lie in (0, 1]")
    pair_columns = [
        "sample_id", "visual_group_id", "mission_profile", "trace_id", "replicate_seed",
    ]
    required = set(pair_columns + ["policy", score_column, mode_column, correct_column])
    if missing := required.difference(frame.columns):
        raise ValueError(f"Matched-coverage input missing columns: {sorted(missing)}")
    subset = frame[frame["policy"].astype(str).isin(policies)].copy().reset_index(drop=True)
    if set(subset["policy"].astype(str)) != set(policies):
        raise StudyGateError("Matched-coverage input is missing a registered policy")
    if subset.duplicated(pair_columns + ["policy"]).any():
        raise StudyGateError("Matched-coverage input has duplicate paired trial rows")
    key_sets = {
        policy: set(map(tuple, group[pair_columns].astype(str).to_numpy()))
        for policy, group in subset.groupby("policy", sort=False)
    }
    if any(keys != next(iter(key_sets.values())) for keys in key_sets.values()):
        raise StudyGateError("Matched-coverage policies do not share identical paired trials")

    subset[endpoint_column] = False
    subset["matched_coverage_selected"] = False
    semantic_modes = {"L1_ALERT", "L2_SEMANTICS"}
    audit_rows: list[dict[str, Any]] = []
    for policy in policies:
        policy_rows = subset[subset["policy"].astype(str).eq(policy)]
        selected_indices: list[Any] = []
        insufficient_groups = 0
        group_coverages: list[float] = []
        available_total = 0
        for _, group in policy_rows.groupby("visual_group_id", sort=True, dropna=False):
            target_count = max(1, int(math.ceil(float(target_coverage) * len(group))))
            scores = pd.to_numeric(group[score_column], errors="coerce")
            eligible = group[
                group[mode_column].astype(str).isin(semantic_modes)
                & scores.notna() & scores.between(0.0, 1.0)
            ].copy()
            available_total += len(eligible)
            if len(eligible) < target_count:
                insufficient_groups += 1
                continue
            eligible["__match_score"] = pd.to_numeric(eligible[score_column], errors="raise")
            tie_columns = []
            for number, column in enumerate(pair_columns):
                tie = f"__match_tie_{number}"
                eligible[tie] = eligible[column].astype(str)
                tie_columns.append(tie)
            ranked = eligible.sort_values(
                ["__match_score", *tie_columns],
                ascending=[False, *([True] * len(tie_columns))],
                kind="mergesort",
            )
            chosen = list(ranked.index[:target_count])
            selected_indices.extend(chosen)
            group_coverages.append(target_count / len(group))
        if selected_indices:
            subset.loc[selected_indices, "matched_coverage_selected"] = True
            wrong = ~subset.loc[selected_indices, correct_column].astype(bool)
            subset.loc[selected_indices, endpoint_column] = wrong.to_numpy()
        selected_policy = subset[
            subset["policy"].astype(str).eq(policy)
            & subset["matched_coverage_selected"].astype(bool)
        ]
        group_risk = (
            selected_policy.groupby("visual_group_id", dropna=False)[endpoint_column].mean()
            if len(selected_policy) else pd.Series(dtype=float)
        )
        audit_rows.append({
            "policy": policy,
            "selection_rule": "top_calibrated_semantic_score_within_visual_group",
            "target_coverage": float(target_coverage),
            "visual_groups": int(policy_rows["visual_group_id"].nunique(dropna=False)),
            "available_semantic_rows": int(available_total),
            "selected_rows": int(len(selected_policy)),
            "realized_group_equal_coverage": (
                float(sum(group_coverages) / len(group_coverages)) if group_coverages else 0.0
            ),
            "group_equal_selective_risk": (
                float(group_risk.mean()) if len(group_risk) else float("nan")
            ),
            "insufficient_visual_groups": int(insufficient_groups),
            "passed": bool(insufficient_groups == 0 and len(group_coverages) > 0),
        })
    audit = pd.DataFrame(audit_rows)
    if audit["passed"].astype(bool).all():
        selected_counts = subset.groupby(
            ["policy", "visual_group_id"], dropna=False,
        )["matched_coverage_selected"].sum().unstack("policy")
        if selected_counts[list(policies)].nunique(axis=1).ne(1).any():
            raise StudyGateError("Matched-coverage selection counts differ between policies")
    return subset, audit


def verify_content_snapshot(path: str | Path) -> None:
    """Verify every source file enumerated by a frozen dataset snapshot."""
    snapshot_path = Path(path).expanduser().resolve()
    if not snapshot_path.is_file():
        raise StudyGateError(f"Dataset content snapshot is missing: {snapshot_path}")
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    files = payload.get("files", [])
    if not isinstance(files, list) or not files:
        raise StudyGateError("Dataset content snapshot has no files")
    for item in files:
        source = Path(str(item.get("path", ""))).expanduser().resolve()
        expected = str(item.get("sha256", ""))
        if not source.is_file() or hash_file(source) != expected:
            raise StudyGateError(f"Dataset source changed after protocol freeze: {source}")


def _verify_required_dataset_provenance_artifacts(frozen: Mapping[str, Any]) -> None:
    decisions = frozen.get("decisions", {})
    manifest = decisions.get("artifact_manifest", {})
    required_provenance = decisions.get("required_dataset_provenance_artifacts", ())
    if required_provenance:
        if (
            not isinstance(required_provenance, (list, tuple))
            or not all(isinstance(name, str) and name for name in required_provenance)
        ):
            raise StudyGateError("Frozen protocol has invalid required dataset-provenance names")
        missing = sorted(set(required_provenance).difference(manifest))
        if missing:
            raise StudyGateError(
                f"Frozen protocol omits required dataset provenance artifacts: {missing}"
            )
        verify_frozen_file_manifest({name: manifest[name] for name in required_provenance})


def _verify_protocol_artifacts(frozen: Mapping[str, Any]) -> None:
    decisions = frozen.get("decisions", {})
    manifest = decisions.get("artifact_manifest", {})
    verify_frozen_file_manifest(manifest)
    _verify_required_dataset_provenance_artifacts(frozen)
    snapshot = manifest.get("dataset_content_snapshot", {})
    if snapshot:
        verify_content_snapshot(snapshot.get("path", ""))
    expected_notebook = frozen.get("decisions", {}).get("notebook_source_hash")
    notebook_path = frozen.get("decisions", {}).get("notebook_path")
    if expected_notebook:
        if not notebook_path or notebook_source_hash(notebook_path) != expected_notebook:
            raise StudyGateError("Notebook cell sources changed after protocol freeze")


def source_tree_hash(root: str | Path) -> str:
    root = Path(root)
    digest = hashlib.sha256()
    paths: set[Path] = set()
    for pattern in (
        "semuav/*.py", "tests/*.py", "configs/*.yaml", "prompts/*",
        "annotations/mission_ontology.yaml", "annotations/ANNOTATION_GUIDE.md",
        "build_*_notebook.py", "requirements.txt",
    ):
        paths.update(path for path in root.glob(pattern) if path.is_file())
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path


def atomic_write_json(path: str | Path, value: Any) -> Path:
    return atomic_write_text(path, json.dumps(_plain(value), indent=2, sort_keys=True) + "\n")


def atomic_write_frame(path: str | Path, frame: pd.DataFrame) -> Path:
    """Write a gzipped CSV atomically; never relies on optional Parquet engines."""
    path = Path(path)
    if not str(path).endswith(".csv.gz"):
        raise ValueError("Data-frame artifacts must use the .csv.gz suffix")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".csv.gz", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(tmp_name, index=False, compression="gzip")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "NO_GIT_COMMIT"


@dataclasses.dataclass(frozen=True)
class RunContext:
    project_root: Path
    run_root: Path
    profile: str
    run_name: str
    config: Mapping[str, Any]
    config_hash: str
    code_hash: str
    created_utc: str
    resume: bool = True

    @classmethod
    def create(
        cls,
        project_root: str | Path,
        cache_root: str | Path,
        profile: str,
        run_name: str,
        config: Mapping[str, Any],
        resume: bool = True,
    ) -> "RunContext":
        project_root = Path(project_root).resolve()
        cache_root = Path(cache_root).resolve()
        config_hash = hash_object(config)
        code_hash = source_tree_hash(project_root)
        run_id = f"{run_name}-{profile}-{config_hash[:10]}-{code_hash[:10]}"
        run_root = cache_root / "runs" / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        for name in ("stages", "blobs", "checkpoints", "logs", "results", "plots"):
            (run_root / name).mkdir(exist_ok=True)
        created_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ctx = cls(
            project_root=project_root,
            run_root=run_root,
            profile=profile,
            run_name=run_name,
            config=_plain(config),
            config_hash=config_hash,
            code_hash=code_hash,
            created_utc=created_utc,
            resume=resume,
        )
        ctx._initialize()
        return ctx

    @property
    def is_scientific(self) -> bool:
        return self.profile == "full"

    @property
    def db_path(self) -> Path:
        return self.run_root / "status.sqlite"

    def _initialize(self) -> None:
        manifest_path = self.run_root / "run_manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous["config_hash"] != self.config_hash or previous["code_hash"] != self.code_hash:
                raise StudyGateError("Run directory hash collision or modified immutable run")
        else:
            atomic_write_json(
                manifest_path,
                {
                    "scientific_status": "SCIENTIFIC_PROFILE_GATES_PENDING" if self.is_scientific else "NON_SCIENTIFIC",
                    "profile": self.profile,
                    "run_name": self.run_name,
                    "config_hash": self.config_hash,
                    "code_hash": self.code_hash,
                    "git_commit": _git_commit(self.project_root),
                    "created_utc": self.created_utc,
                    "python": sys.version,
                    "platform": platform.platform(),
                    "config": self.config,
                },
            )
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS stages (
                    name TEXT PRIMARY KEY, cache_key TEXT NOT NULL, status TEXT NOT NULL,
                    artifact TEXT, checksum TEXT, rows INTEGER, started_utc TEXT,
                    ended_utc TEXT, message TEXT
                )"""
            )

    def stage_dir(self, name: str) -> Path:
        path = self.run_root / "stages" / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def stage_record(self, name: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            row = con.execute("SELECT * FROM stages WHERE name=?", (name,)).fetchone()
        return dict(row) if row else None

    def stage_complete(self, name: str, cache_key: str) -> bool:
        row = self.stage_record(name)
        if not (self.resume and row and row["status"] == "COMPLETE" and row["cache_key"] == cache_key):
            return False
        artifact = Path(row["artifact"])
        return artifact.exists() and hash_file(artifact) == row["checksum"]

    def set_stage(
        self,
        name: str,
        cache_key: str,
        status: str,
        artifact: str | Path | None = None,
        rows: int | None = None,
        message: str = "",
        started_utc: str | None = None,
    ) -> None:
        artifact_s = str(artifact) if artifact else None
        checksum = hash_file(artifact_s) if artifact_s and Path(artifact_s).is_file() else None
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """INSERT OR REPLACE INTO stages
                   (name,cache_key,status,artifact,checksum,rows,started_utc,ended_utc,message)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (name, cache_key, status, artifact_s, checksum, rows, started_utc or now, now, message),
            )

    def status_frame(self) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as con:
            return pd.read_sql_query("SELECT * FROM stages ORDER BY name", con)

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            raise StudyGateError(message)

    def assert_scientific_output_path(self, path: str | Path) -> None:
        path = Path(path).resolve()
        if not self.is_scientific and "dry_run" not in str(path).lower():
            raise StudyGateError("Non-scientific profiles may write only clearly marked dry-run paths")


def frame_stage(
    ctx: RunContext,
    name: str,
    key_material: Any,
    compute: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    cache_key = hash_object({"name": name, "key": key_material, "config": ctx.config_hash, "code": ctx.code_hash})
    artifact = ctx.stage_dir(name) / f"{cache_key}.csv.gz"
    if (
        os.environ.get("SEMUAV_ENABLE_V3R1_VALIDATION_IMPORT") == "1"
        and not ctx.stage_complete(name, cache_key)
    ):
        # v3r2 validation only: install an exact, hash-verified v3r1 result
        # for unchanged producers; otherwise compute normally below.
        from .v3r1_import import import_stage

        import_stage(ctx, name, key_material, current_key=cache_key)
    if ctx.stage_complete(name, cache_key):
        return pd.read_csv(artifact, compression="gzip")
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ctx.set_stage(name, cache_key, "RUNNING", message="started", started_utc=started)
    try:
        frame = compute()
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"Stage {name} returned {type(frame).__name__}, expected DataFrame")
        atomic_write_frame(artifact, frame)
        ctx.set_stage(name, cache_key, "COMPLETE", artifact, len(frame), started_utc=started)
        return frame
    except Exception as exc:
        ctx.set_stage(name, cache_key, "BLOCKED", message=f"{type(exc).__name__}: {exc}", started_utc=started)
        raise


def _load_verified_protocol(
    ctx: RunContext,
    protocol_artifact: str | Path,
    *,
    verify_outcome_artifacts: bool = True,
) -> dict[str, Any]:
    protocol_artifact = Path(protocol_artifact)
    ctx.require(protocol_artifact.is_file(), "Freeze the protocol before accessing the untouched test set")
    frozen = json.loads(protocol_artifact.read_text())
    ctx.require(frozen.get("config_hash") == ctx.config_hash, "Frozen protocol does not match this configuration")
    ctx.require(frozen.get("code_hash") == ctx.code_hash, "Code changed after protocol freeze")
    expected_protocol_hash = str(frozen.get("protocol_hash", ""))
    unhashed = dict(frozen)
    unhashed.pop("protocol_hash", None)
    ctx.require(
        bool(expected_protocol_hash) and hash_object(unhashed) == expected_protocol_hash,
        "Frozen protocol content does not match its protocol hash",
    )
    if verify_outcome_artifacts:
        _verify_protocol_artifacts(frozen)
    return frozen


@contextmanager
def test_access_guard(ctx: RunContext, protocol_artifact: str | Path):
    """Permit test-set reads only after an immutable protocol artifact exists."""
    frozen = _load_verified_protocol(ctx, protocol_artifact)
    yield frozen


def authorize_test_once(ctx: RunContext, protocol_artifact: str | Path) -> dict[str, Any]:
    """Open the untouched test exactly once and leave a durable audit ledger.

    An interrupted/failed confirmatory run is not silently restartable.  A
    documented amendment must use a new run name and rerun every comparator.
    """
    frozen = _load_verified_protocol(ctx, protocol_artifact)
    ledger = ctx.run_root / "test_access_ledger.json"
    if ledger.exists():
        previous = json.loads(ledger.read_text())
        raise StudyGateError(
            "Untouched test access has already been consumed for this run "
            f"({previous.get('status', 'UNKNOWN')}); use a documented amended run name"
        )
    atomic_write_json(ledger, {
        "status": "OPEN",
        "opened_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_hash": frozen.get("protocol_hash"),
        "config_hash": ctx.config_hash,
        "code_hash": ctx.code_hash,
    })
    return frozen


def finalize_test_once(
    ctx: RunContext,
    *,
    status: str,
    result_artifacts: Iterable[str | Path] = (),
    restricted_artifacts: Iterable[str | Path] = (),
) -> Path:
    """Finalize one-time test access with an explicit disclosure boundary.

    ``result_artifacts`` are report-consumable outputs and are rehashed by
    both handoff loaders. ``restricted_artifacts`` are test inputs and
    preparation evidence (for example, source truth or analysis manifests).
    Their hashes live only inside a nested receipt.  The report-safe loader
    authenticates that receipt as one opaque frozen object; the full embedded
    loader additionally opens and rehashes every underlying restricted file.
    """

    ledger = ctx.run_root / "test_access_ledger.json"
    ctx.require(ledger.is_file(), "No open untouched-test ledger exists")
    payload = json.loads(ledger.read_text())
    ctx.require(payload.get("status") == "OPEN", "Untouched-test ledger is already finalized")
    if status not in {"COMPLETE", "FAILED", "BLOCKED"}:
        raise ValueError("Test final status must be COMPLETE, FAILED, or BLOCKED")
    result_artifacts = tuple(result_artifacts)
    restricted_artifacts = tuple(restricted_artifacts)
    ctx.require(bool(result_artifacts), "Finalized test ledger requires result artifacts")
    run_root = ctx.run_root.resolve()

    def artifact_hashes(
        values: Iterable[str | Path], *, kind: str,
    ) -> dict[str, dict[str, Any]]:
        items: dict[str, dict[str, Any]] = {}
        for artifact in values:
            path = Path(artifact).expanduser().resolve()
            try:
                path.relative_to(run_root)
            except ValueError as exc:
                raise StudyGateError(
                    f"{kind} artifact escapes the immutable run directory: {path}"
                ) from exc
            ctx.require(path.is_file() and not path.is_symlink(), f"{kind} artifact is missing or unsafe: {path}")
            key = str(path)
            ctx.require(key not in items, f"Duplicate {kind.lower()} artifact: {path}")
            items[key] = {"sha256": hash_file(path), "bytes": int(path.stat().st_size)}
        return items

    report_entries = artifact_hashes(result_artifacts, kind="Test result")
    restricted_entries = artifact_hashes(
        restricted_artifacts, kind="Restricted test evidence",
    )
    overlap = sorted(set(report_entries).intersection(restricted_entries))
    ctx.require(not overlap, f"Artifacts cannot be both report and restricted evidence: {overlap}")

    receipt_path = (run_root / RESTRICTED_TEST_EVIDENCE_RECEIPT).resolve()
    ctx.require(
        not receipt_path.exists(),
        "Restricted test-evidence receipt already exists; use a new run",
    )
    receipt_payload = {
        "schema_version": RESTRICTED_TEST_EVIDENCE_SCHEMA,
        "protocol_hash": payload.get("protocol_hash"),
        "config_hash": ctx.config_hash,
        "code_hash": ctx.code_hash,
        "artifact_count": len(restricted_entries),
        "artifacts": restricted_entries,
    }
    atomic_write_json(receipt_path, receipt_payload)
    report_hashes = {
        path: str(item["sha256"]) for path, item in report_entries.items()
    }
    payload.update({
        "status": status,
        "closed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # ``result_hashes`` is retained as a compatibility alias, but contains
        # report outputs only. Restricted paths never appear in the ledger.
        "result_hashes": report_hashes,
        "report_result_hashes": report_hashes,
        "restricted_evidence_receipt": {
            "schema_version": RESTRICTED_TEST_EVIDENCE_SCHEMA,
            "path": str(receipt_path),
            "sha256": hash_file(receipt_path),
            "artifact_count": len(restricted_entries),
        },
    })
    return atomic_write_json(ledger, payload)


def _verify_restricted_evidence_receipt(
    ctx: RunContext,
    ledger: Mapping[str, Any],
    *,
    verify_underlying_artifacts: bool,
) -> tuple[Path, dict[str, Path]]:
    """Authenticate the nested restricted receipt, optionally reopening inputs."""

    metadata = ledger.get("restricted_evidence_receipt")
    ctx.require(
        isinstance(metadata, Mapping),
        "COMPLETE test ledger has no restricted-evidence receipt",
    )
    ctx.require(
        metadata.get("schema_version") == RESTRICTED_TEST_EVIDENCE_SCHEMA,
        "Restricted test-evidence receipt schema is invalid",
    )
    receipt_path = Path(str(metadata.get("path", ""))).expanduser().resolve()
    expected_path = (ctx.run_root.resolve() / RESTRICTED_TEST_EVIDENCE_RECEIPT).resolve()
    ctx.require(
        receipt_path == expected_path,
        "Restricted test-evidence receipt has an unexpected path",
    )
    ctx.require(
        receipt_path.is_file() and not receipt_path.is_symlink(),
        "Restricted test-evidence receipt is missing or unsafe",
    )
    expected_hash = str(metadata.get("sha256", ""))
    ctx.require(
        re.fullmatch(r"[0-9a-f]{64}", expected_hash) is not None
        and hash_file(receipt_path) == expected_hash,
        "Restricted test-evidence receipt changed",
    )
    count = metadata.get("artifact_count")
    ctx.require(
        isinstance(count, int) and not isinstance(count, bool) and count >= 0,
        "Restricted test-evidence receipt count is invalid",
    )
    if not verify_underlying_artifacts:
        # Deliberately do not parse the receipt: report generation is allowed
        # to authenticate its digest, but never learn/open its restricted paths.
        return receipt_path, {}

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    ctx.require(
        receipt.get("schema_version") == RESTRICTED_TEST_EVIDENCE_SCHEMA,
        "Restricted test-evidence receipt content has an invalid schema",
    )
    for field, expected in (
        ("protocol_hash", ledger.get("protocol_hash")),
        ("config_hash", ctx.config_hash),
        ("code_hash", ctx.code_hash),
        ("artifact_count", count),
    ):
        ctx.require(
            receipt.get(field) == expected,
            f"Restricted test-evidence receipt {field} binding differs",
        )
    entries = receipt.get("artifacts")
    ctx.require(isinstance(entries, Mapping), "Restricted test-evidence artifact map is invalid")
    ctx.require(len(entries) == count, "Restricted test-evidence artifact count differs")
    run_root = ctx.run_root.resolve()
    artifacts: dict[str, Path] = {}
    for artifact_name, item in sorted(entries.items()):
        ctx.require(isinstance(item, Mapping), "Restricted test-evidence entry is malformed")
        path = Path(str(artifact_name)).expanduser().resolve()
        try:
            path.relative_to(run_root)
        except ValueError as exc:
            raise StudyGateError(
                f"Restricted test-evidence artifact escapes the immutable run directory: {path}"
            ) from exc
        ctx.require(path.is_file() and not path.is_symlink(), f"Restricted test evidence is missing or unsafe: {path}")
        expected = str(item.get("sha256", ""))
        expected_bytes = item.get("bytes")
        ctx.require(
            re.fullmatch(r"[0-9a-f]{64}", expected) is not None
            and hash_file(path) == expected,
            f"Restricted test evidence changed: {path}",
        )
        ctx.require(
            isinstance(expected_bytes, int)
            and not isinstance(expected_bytes, bool)
            and expected_bytes >= 0
            and path.stat().st_size == expected_bytes,
            f"Restricted test-evidence size changed: {path}",
        )
        if path.name in artifacts:
            raise StudyGateError(
                f"Ambiguous duplicate restricted artifact basename in receipt: {path.name}"
            )
        artifacts[path.name] = path
    return receipt_path, artifacts


def _load_finalized_test_handoff_core(
    ctx: RunContext,
    protocol_artifact: str | Path,
    *,
    required_artifacts: Iterable[str],
    verify_dataset_provenance: bool,
    verify_restricted_evidence: bool,
    consumer: str,
) -> FinalizedTestHandoff:
    """Shared finalized-ledger verifier with an explicit dependency boundary."""

    frozen = _load_verified_protocol(
        ctx, protocol_artifact, verify_outcome_artifacts=False,
    )
    if verify_dataset_provenance:
        _verify_required_dataset_provenance_artifacts(frozen)
    ledger_path = ctx.run_root / "test_access_ledger.json"
    ctx.require(
        ledger_path.is_file(),
        f"{consumer} handoff requires a finalized untouched-test ledger",
    )
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ctx.require(
        ledger.get("status") == "COMPLETE",
        f"{consumer} handoff requires a COMPLETE test ledger; "
        f"got {ledger.get('status', 'UNKNOWN')}",
    )
    ctx.require(bool(ledger.get("closed_utc")), "COMPLETE test ledger has no close timestamp")
    ctx.require(
        ledger.get("protocol_hash") == frozen.get("protocol_hash"),
        "Test ledger is bound to a different frozen protocol",
    )
    ctx.require(ledger.get("config_hash") == ctx.config_hash, "Test ledger configuration hash differs")
    ctx.require(ledger.get("code_hash") == ctx.code_hash, "Test ledger code hash differs")
    result_hashes = ledger.get("report_result_hashes")
    ctx.require(isinstance(result_hashes, Mapping) and bool(result_hashes), "COMPLETE test ledger has no result hashes")
    ctx.require(
        ledger.get("result_hashes") == result_hashes,
        "Test ledger report-result compatibility hash map differs",
    )

    receipt_path, restricted_artifacts = _verify_restricted_evidence_receipt(
        ctx,
        ledger,
        verify_underlying_artifacts=verify_restricted_evidence,
    )

    run_root = ctx.run_root.resolve()
    artifacts: dict[str, Path] = {}
    for artifact_name, expected_hash in sorted(result_hashes.items()):
        path = Path(str(artifact_name)).expanduser().resolve()
        try:
            path.relative_to(run_root)
        except ValueError as exc:
            raise StudyGateError(f"Test ledger artifact escapes the immutable run directory: {path}") from exc
        ctx.require(path.is_file(), f"Finalized untouched-test result is missing: {path}")
        ctx.require(
            isinstance(expected_hash, str)
            and len(expected_hash) == 64
            and hash_file(path) == expected_hash,
            f"Finalized untouched-test result changed: {path}",
        )
        if path.name in artifacts:
            raise StudyGateError(f"Ambiguous duplicate test artifact basename in ledger: {path.name}")
        artifacts[path.name] = path

    required = tuple(str(name) for name in required_artifacts)
    missing = sorted(set(required).difference(artifacts))
    ctx.require(not missing, f"Finalized test handoff is missing required artifacts: {missing}")
    return FinalizedTestHandoff(
        protocol=frozen,
        ledger=ledger,
        ledger_path=ledger_path.resolve(),
        artifacts=artifacts,
        restricted_evidence_receipt_path=receipt_path,
        restricted_artifacts=restricted_artifacts,
    )


def load_finalized_test_handoff(
    ctx: RunContext,
    protocol_artifact: str | Path,
    *,
    required_artifacts: Iterable[str] = EMBEDDED_HANDOFF_REQUIRED_ARTIFACTS,
) -> FinalizedTestHandoff:
    """Load a strict post-test handoff for non-report consumers.

    This path preserves the original full verification behavior: required
    frozen dataset-provenance artifacts are rehashed before the immutable test
    results are exposed. It is appropriate only while those registered inputs
    remain mounted and available.
    """

    return _load_finalized_test_handoff_core(
        ctx,
        protocol_artifact,
        required_artifacts=required_artifacts,
        verify_dataset_provenance=True,
        verify_restricted_evidence=True,
        consumer="Embedded",
    )


def load_report_safe_finalized_test_handoff(
    ctx: RunContext,
    protocol_artifact: str | Path,
    *,
    required_artifacts: Iterable[str] = EMBEDDED_HANDOFF_REQUIRED_ARTIFACTS,
) -> FinalizedTestHandoff:
    """Load finalized paper results without reopening scientific inputs.

    The report boundary verifies only the self-hashed frozen-protocol receipt,
    its COMPLETE ledger binding, and every ledger-listed result artifact below
    the immutable run directory. It deliberately does *not* resolve, stat,
    hash, or open dataset receipts/inventories, source snapshots, model or
    detector checkpoints, notebooks, CUDA executables, or environment paths
    named by the frozen input manifest. Any report-required frozen media audit
    is verified separately by its dedicated artifact verifier.
    """

    return _load_finalized_test_handoff_core(
        ctx,
        protocol_artifact,
        required_artifacts=required_artifacts,
        verify_dataset_provenance=False,
        verify_restricted_evidence=False,
        consumer="Report",
    )


def freeze_protocol(ctx: RunContext, decisions: Mapping[str, Any], validation_summary: Mapping[str, Any]) -> Path:
    path = ctx.run_root / "frozen_protocol.json"
    if path.exists():
        raise StudyGateError("Protocol is already frozen; start a new run to change it")
    payload = {
        "config_hash": ctx.config_hash,
        "code_hash": ctx.code_hash,
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decisions": _plain(decisions),
        "validation_summary": _plain(validation_summary),
    }
    payload["protocol_hash"] = hash_object(payload)
    return atomic_write_json(path, payload)
