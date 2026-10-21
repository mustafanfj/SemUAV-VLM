"""Typed configuration and scientific-mode gates for the SemUAV-VLM study.

Configuration files are intentionally boring YAML. Paths are made absolute at
load time: ``project_root`` is relative to the YAML, dataset/trace paths are
relative to ``data_root``, and artifact paths are relative to the project. Merely parsing a full configuration is
allowed so that a notebook can display a useful pre-flight report; calling
``StudyConfig.assert_ready`` is the mandatory boundary before any real run.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml


class ConfigError(ValueError):
    """The study configuration is malformed or scientifically unsafe."""


class PrerequisiteError(RuntimeError):
    """A configured run cannot start because an external prerequisite is absent."""


class Profile(str, Enum):
    """Execution profiles, ordered by increasing scientific scope."""

    DRY_RUN = "dry_run"
    SMOKE_REAL = "smoke_real"
    FULL = "full"

    @classmethod
    def parse(cls, value: str | "Profile") -> "Profile":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ConfigError(f"Unknown profile {value!r}; expected one of: {allowed}") from exc


class StudyScope(str, Enum):
    """Immutable scientific claim boundary for a full-profile run."""

    FULL_PROPOSAL = "full_proposal"
    VISDRONE_PRIMARY = "visdrone_primary"

    @classmethod
    def parse(cls, value: str | "StudyScope") -> "StudyScope":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ConfigError(
                f"Unknown study_scope {value!r}; expected one of: {allowed}"
            ) from exc


class TruthMode(str, Enum):
    """Registered origin of the study endpoint labels."""

    ADJUDICATED_MISSION_LABELS = "adjudicated_mission_labels"
    PUBLISHER_SOURCE_RULE = "publisher_source_rule"

    @classmethod
    def parse(cls, value: str | "TruthMode") -> "TruthMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ConfigError(
                f"Unknown truth_mode {value!r}; expected one of: {allowed}"
            ) from exc


@dataclasses.dataclass(frozen=True)
class PathsConfig:
    project_root: Path
    data_root: Path
    output_root: Path
    cache_root: Path
    annotations: Path | None = None
    test_annotations: Path | None = None
    split_manifest: Path | None = None
    channel_traces: Path | None = None
    telemetry_manifest: Path | None = None


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    name: str
    adapter: str
    root: Path
    mission: str
    enabled: bool = True
    required: bool = True
    synthetic: bool = False
    options: Mapping[str, Any] = dataclasses.field(default_factory=dict)


_VISDRONE_CONFIRMATORY_COMPONENTS = {
    "det_train": "1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn",
    "det_val": "1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59",
    "det_test_dev": "1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V",
    "mot_train": "1-qX2d-P1Xr64ke6nTdlm33om1VxCUTSh",
    "mot_val": "1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu",
    "mot_test_dev": "14z8Acxopj1d86-qhsF1NwS4Bv3KYa4Wu",
}
_UAVID_CONFIRMATORY_SOURCE = {
    "persistent_id": "doi:10.17026/dans-x9f-w9sa",
    "version": "1.0",
}
_UAVREASON_CONFIRMATORY_SOURCE = {
    "annotations_repo": "jarvissun/UAVReason_vqa",
    "annotations_revision": "e459740b936b0fb489ce5fede0c2077ee28999c8",
    "images_repo": "sijieaaa/UAVScenes",
    "images_revision": "04a224fd14b0b8fdfd2633e68f20471b85c611a5",
}
_FLAME3_CONFIRMATORY_DOI = "10.21227/w0mz-aq48"


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_provenance_errors(
    spec: DatasetSpec,
) -> tuple[list[str], Mapping[str, Any] | None]:
    """Validate a prepared source receipt and its file-hash inventory.

    The inventory file and every declared source are rehashed, tree membership
    is checked for unlisted additions, and the frozen protocol subsequently
    protects both provenance files verbatim.
    """

    errors: list[str] = []
    receipt_path = spec.root / "SOURCE_RECEIPT.json"
    inventory_path = spec.root / "SOURCE_FILES_SHA256.jsonl"
    if not receipt_path.is_file():
        errors.append(f"dataset {spec.name!r} source receipt is missing: {receipt_path}")
    if not inventory_path.is_file():
        errors.append(
            f"dataset {spec.name!r} source file hash inventory is missing: {inventory_path}"
        )
    if not receipt_path.is_file():
        return errors, None

    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"dataset {spec.name!r} source receipt is invalid: {receipt_path} ({exc})"], None
    if not isinstance(payload, Mapping):
        return [f"dataset {spec.name!r} source receipt must be a JSON object: {receipt_path}"], None
    if payload.get("schema_version") != "semuav.dataset-source.v1":
        errors.append(f"dataset {spec.name!r} source receipt has an unsupported schema")
    expected_dataset = spec.adapter.removeprefix("generic_")
    if str(payload.get("dataset", "")).strip().lower() != expected_dataset:
        errors.append(
            f"dataset {spec.name!r} source receipt identifies {payload.get('dataset')!r}, "
            f"expected {expected_dataset!r}"
        )
    source = payload.get("source")
    if not isinstance(source, Mapping) or not str(source.get("landing_page", "")).strip():
        errors.append(f"dataset {spec.name!r} source receipt lacks source.landing_page provenance")
    manifest_info = payload.get("content_hash_manifest")
    if not isinstance(manifest_info, Mapping):
        errors.append(f"dataset {spec.name!r} source receipt lacks content_hash_manifest metadata")
        return errors, payload
    if not inventory_path.is_file():
        return errors, payload
    declared_path = Path(str(manifest_info.get("path", ""))).expanduser()
    if not declared_path.is_absolute():
        declared_path = spec.root / declared_path
    if declared_path.resolve() != inventory_path.resolve():
        errors.append(
            f"dataset {spec.name!r} receipt points to a different source hash inventory: "
            f"{declared_path}"
        )
    expected_hash = str(manifest_info.get("sha256", "")).strip().lower()
    try:
        observed_hash = _sha256_file(inventory_path)
    except OSError as exc:
        errors.append(f"dataset {spec.name!r} source hash inventory cannot be read: {exc}")
        return errors, payload
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash) or expected_hash != observed_hash:
        errors.append(
            f"dataset {spec.name!r} source hash inventory digest does not match its receipt"
        )

    try:
        declared_files = int(manifest_info.get("files", -1))
        declared_bytes = int(manifest_info.get("bytes", -1))
    except (TypeError, ValueError):
        declared_files = declared_bytes = -1
    observed_files = 0
    observed_bytes = 0
    seen_paths: set[str] = set()
    dataset_root = spec.root.resolve()
    try:
        with inventory_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(
                        f"dataset {spec.name!r} source hash inventory line {line_number} "
                        f"is invalid JSON: {exc}"
                    )
                    continue
                if not isinstance(record, Mapping):
                    errors.append(
                        f"dataset {spec.name!r} source hash inventory line {line_number} "
                        "must be an object"
                    )
                    continue
                relative_text = str(record.get("path", "")).strip()
                relative = PurePosixPath(relative_text)
                try:
                    size = int(record.get("bytes", -1))
                except (TypeError, ValueError):
                    size = -1
                digest = str(record.get("sha256", "")).strip().lower()
                if (
                    not relative_text
                    or relative.is_absolute()
                    or ".." in relative.parts
                    or "\\" in relative_text
                    or relative_text in seen_paths
                    or size < 0
                    or not re.fullmatch(r"[0-9a-f]{64}", digest)
                ):
                    errors.append(
                        f"dataset {spec.name!r} source hash inventory line {line_number} "
                        "has invalid/duplicate path, size, or SHA-256 metadata"
                    )
                    continue
                seen_paths.add(relative_text)
                declared_source_path = dataset_root / Path(*relative.parts)
                source_path = declared_source_path.resolve()
                try:
                    source_path.relative_to(dataset_root)
                except ValueError:
                    errors.append(
                        f"dataset {spec.name!r} source hash inventory escapes its dataset root: "
                        f"{relative_text}"
                    )
                    continue
                if declared_source_path.is_symlink():
                    errors.append(
                        f"dataset {spec.name!r} inventoried source path is a symlink: "
                        f"{declared_source_path}"
                    )
                    continue
                if not source_path.is_file() or source_path.stat().st_size != size:
                    errors.append(
                        f"dataset {spec.name!r} inventoried source file is missing or changed size: "
                        f"{source_path}"
                    )
                    continue
                try:
                    observed_source_hash = _sha256_file(source_path)
                except OSError as exc:
                    errors.append(
                        f"dataset {spec.name!r} inventoried source file cannot be hashed: "
                        f"{source_path} ({exc})"
                    )
                    continue
                if observed_source_hash != digest:
                    errors.append(
                        f"dataset {spec.name!r} inventoried source file content changed: "
                        f"{source_path}"
                    )
                    continue
                observed_files += 1
                observed_bytes += size
    except (OSError, UnicodeError) as exc:
        errors.append(f"dataset {spec.name!r} source hash inventory is unreadable: {exc}")
        return errors, payload
    if observed_files <= 0:
        errors.append(f"dataset {spec.name!r} source hash inventory contains no valid source files")
    if declared_files != observed_files or declared_bytes != observed_bytes:
        errors.append(
            f"dataset {spec.name!r} source hash inventory totals differ from its receipt "
            f"({observed_files} files/{observed_bytes} bytes observed, "
            f"{declared_files} files/{declared_bytes} bytes declared)"
        )
    actual_paths: set[str] = set()
    try:
        for source_path in spec.root.rglob("*"):
            if source_path.name in {"SOURCE_RECEIPT.json", "SOURCE_FILES_SHA256.jsonl"}:
                continue
            if source_path.is_symlink():
                errors.append(
                    f"dataset {spec.name!r} source tree contains an untracked symlink: {source_path}"
                )
                continue
            if source_path.is_file():
                actual_paths.add(source_path.relative_to(spec.root).as_posix())
    except OSError as exc:
        errors.append(f"dataset {spec.name!r} source tree cannot be inventoried: {exc}")
    unlisted = sorted(actual_paths - seen_paths)
    if unlisted:
        errors.append(
            f"dataset {spec.name!r} source tree contains {len(unlisted)} unlisted file(s); "
            f"first: {unlisted[0]}"
        )
    receipt_inventory = payload.get("inventory")
    try:
        envelope_files = int(receipt_inventory.get("files", -1))
        envelope_bytes = int(receipt_inventory.get("bytes", -1))
    except (AttributeError, TypeError, ValueError):
        envelope_files = envelope_bytes = -1
    expected_envelope_files = declared_files + 1
    expected_envelope_bytes = declared_bytes + inventory_path.stat().st_size
    if (
        envelope_files != expected_envelope_files
        or envelope_bytes != expected_envelope_bytes
    ):
        errors.append(
            f"dataset {spec.name!r} source receipt inventory envelope is inconsistent "
            f"({envelope_files} files/{envelope_bytes} bytes declared, "
            f"{expected_envelope_files} files/{expected_envelope_bytes} bytes expected)"
        )
    return errors, payload


def _confirmatory_source_semantics_errors(
    spec: DatasetSpec,
    receipt: Mapping[str, Any],
) -> list[str]:
    """Bind full-profile receipts to the exact registered publisher sources."""

    adapter = spec.adapter.removeprefix("generic_")
    source = receipt.get("source", {})
    source = source if isinstance(source, Mapping) else {}
    errors: list[str] = []
    if adapter == "visdrone":
        artifacts = receipt.get("artifacts", ())
        if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
            artifacts = ()
        component_records = [
            item for item in artifacts
            if isinstance(item, Mapping) and "component" in item
        ]
        observed = [
            (str(item.get("component", "")), str(item.get("google_drive_id", "")))
            for item in component_records
        ]
        expected = set(_VISDRONE_CONFIRMATORY_COMPONENTS.items())
        if len(observed) != len(expected) or set(observed) != expected:
            errors.append(
                "VisDrone full-confirmatory receipt must contain exactly all six registered "
                "release components with their official Google Drive IDs"
            )
    elif adapter == "uavid":
        for field, expected in _UAVID_CONFIRMATORY_SOURCE.items():
            if str(source.get(field, "")) != expected:
                errors.append(
                    f"UAVid full-confirmatory receipt source.{field} must equal {expected!r}"
                )
    elif adapter == "uavreason":
        for field, expected in _UAVREASON_CONFIRMATORY_SOURCE.items():
            observed = str(source.get(field, ""))
            if observed != expected or (
                field.endswith("_revision")
                and not re.fullmatch(r"[0-9a-f]{40}", observed)
            ):
                errors.append(
                    f"UAVReason full-confirmatory receipt source.{field} does not match "
                    "the registered repository/pinned commit"
                )
    elif adapter == "flame3":
        if str(source.get("doi", "")).strip().lower() != _FLAME3_CONFIRMATORY_DOI:
            errors.append(
                "FLAME3 full-confirmatory receipt source.doi must equal "
                f"{_FLAME3_CONFIRMATORY_DOI!r}"
            )
    return errors


@dataclasses.dataclass(frozen=True)
class SplitConfig:
    seed: int = 20260823
    ratios: Mapping[str, float] = dataclasses.field(
        default_factory=lambda: {
            "train": 0.60,
            "calibration": 0.10,
            "validation": 0.10,
            "test": 0.20,
        }
    )
    leakage_keys: tuple[str, ...] = (
        "split_group",
        "flight_id",
        "sequence_id",
        "location_id",
        "event_id",
    )
    strata: tuple[str, ...] = ("dataset", "mission", "critical_event")
    require_frozen_manifest: bool = True
    min_test_units: int = 500
    min_critical_test_events: int = 140

    def validate(self) -> None:
        expected = {"train", "calibration", "validation", "test"}
        if set(self.ratios) != expected:
            raise ConfigError(f"split.ratios must contain exactly {sorted(expected)}")
        values = [float(self.ratios[name]) for name in expected]
        if any(value < 0.0 for value in values):
            raise ConfigError("split ratios cannot be negative")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ConfigError(f"split ratios must sum to 1.0, got {sum(values):.12g}")
        if not self.leakage_keys or "split_group" not in self.leakage_keys:
            raise ConfigError("split.leakage_keys must include split_group")
        if self.min_test_units < 0 or self.min_critical_test_events < 0:
            raise ConfigError("minimum test counts cannot be negative")


@dataclasses.dataclass(frozen=True)
class AnnotationConfig:
    enabled: bool = True
    sample_column: str = "sample_id"
    task_column: str = "task"
    label_column: str = "label"
    annotator_column: str = "annotator_id"
    blinded_column: str = "blinded"
    adjudicated_label_column: str = "adjudicated_label"
    adjudicator_column: str = "adjudicator_id"
    critical_column: str = "is_critical"
    required_annotators: int = 2
    require_blinding: bool = True
    require_manifest_coverage: bool = True
    require_adjudication_on_disagreement: bool = True
    min_observed_agreement: float = 0.80
    min_chance_corrected_agreement: float = 0.80
    min_rare_chance_corrected_agreement: float = 0.70
    min_evidence_geometry_iou: float = 0.70

    def validate(self) -> None:
        if self.required_annotators < 2:
            raise ConfigError("annotations.required_annotators must be at least 2")
        for name, value in (
            ("min_observed_agreement", self.min_observed_agreement),
            ("min_chance_corrected_agreement", self.min_chance_corrected_agreement),
            ("min_rare_chance_corrected_agreement", self.min_rare_chance_corrected_agreement),
        ):
            if not -1.0 <= float(value) <= 1.0:
                raise ConfigError(f"annotations.{name} must be within [-1, 1]")
        if not 0.0 <= float(self.min_evidence_geometry_iou) <= 1.0:
            raise ConfigError("annotations.min_evidence_geometry_iou must be within [0, 1]")


@dataclasses.dataclass(frozen=True)
class StudyConfig:
    schema_version: int
    study_id: str
    profile: Profile
    seed: int
    seeds: tuple[int, ...]
    allow_synthetic: bool
    paths: PathsConfig
    datasets: tuple[DatasetSpec, ...]
    split: SplitConfig
    annotations: AnnotationConfig
    study_scope: StudyScope = StudyScope.FULL_PROPOSAL
    truth_mode: TruthMode = TruthMode.ADJUDICATED_MISSION_LABELS
    runtime: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    models: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    payloads: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    network: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    controller: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    evaluation: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    hardware: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    benchmark: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    raw: Mapping[str, Any] = dataclasses.field(default_factory=dict, repr=False, compare=False)

    @property
    def is_scientific(self) -> bool:
        return self.profile is Profile.FULL

    @property
    def uses_real_data(self) -> bool:
        return self.profile is not Profile.DRY_RUN

    @property
    def enabled_datasets(self) -> tuple[DatasetSpec, ...]:
        return tuple(spec for spec in self.datasets if spec.enabled)

    def dataset(self, name: str) -> DatasetSpec:
        matches = [spec for spec in self.datasets if spec.name == name]
        if not matches:
            raise KeyError(name)
        return matches[0]

    def dataset_provenance_paths(self) -> dict[str, Path]:
        """Return receipt/inventory files that must enter the frozen protocol."""

        paths: dict[str, Path] = {}
        for spec in self.enabled_datasets:
            if spec.synthetic or spec.adapter == "synthetic":
                continue
            paths[f"dataset_{spec.name}_source_receipt"] = spec.root / "SOURCE_RECEIPT.json"
            paths[f"dataset_{spec.name}_source_file_hash_inventory"] = (
                spec.root / "SOURCE_FILES_SHA256.jsonl"
            )
        return paths

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ConfigError(f"Unsupported schema_version={self.schema_version}; expected 1")
        if not self.study_id.strip():
            raise ConfigError("study_id cannot be empty")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ConfigError("seeds must be a non-empty list of unique integers")
        if not self.enabled_datasets:
            raise ConfigError("At least one dataset must be enabled")
        names = [spec.name for spec in self.datasets]
        if len(names) != len(set(names)):
            raise ConfigError("Dataset names must be unique")
        adapters = {"synthetic", "visdrone", "uavid", "uavreason", "flame3", "generic_flame3"}
        bad = sorted({spec.adapter for spec in self.datasets} - adapters)
        if bad:
            raise ConfigError(f"Unsupported dataset adapter(s): {bad}")
        if self.profile is Profile.FULL:
            if self.allow_synthetic:
                raise ConfigError("full profile must set allow_synthetic: false")
            synthetic = [spec.name for spec in self.enabled_datasets if spec.synthetic]
            if synthetic:
                raise ConfigError(f"full profile cannot enable synthetic datasets: {synthetic}")
            required = (
                {"visdrone", "uavid", "uavreason", "flame3"}
                if self.study_scope is StudyScope.FULL_PROPOSAL
                else {"visdrone"}
            )
            present = {spec.adapter.removeprefix("generic_") for spec in self.enabled_datasets}
            missing = sorted(required - present)
            if missing:
                scope_detail = (
                    "all proposal datasets"
                    if self.study_scope is StudyScope.FULL_PROPOSAL
                    else "the primary VisDrone dataset"
                )
                raise ConfigError(
                    f"full profile/{self.study_scope.value} must enable {scope_detail}; "
                    f"missing {missing}"
                )
            if self.study_scope is StudyScope.VISDRONE_PRIMARY and present != {"visdrone"}:
                raise ConfigError(
                    "visdrone_primary scope must enable exactly VisDrone; evaluate other "
                    "datasets only in a separately frozen scope"
                )
            if (
                self.truth_mode is TruthMode.PUBLISHER_SOURCE_RULE
                and self.study_scope is not StudyScope.VISDRONE_PRIMARY
            ):
                raise ConfigError(
                    "publisher_source_rule truth is registered only for visdrone_primary"
                )
            if not self.annotations.enabled and self.truth_mode is not TruthMode.PUBLISHER_SOURCE_RULE:
                raise ConfigError(
                    "full profile requires annotations unless truth_mode is the frozen "
                    "VisDrone publisher-source rule"
                )
            if self.truth_mode is TruthMode.PUBLISHER_SOURCE_RULE:
                required_benchmark = {
                    "publisher_partition_policy",
                    "mission_profiles",
                    "visibility_rule",
                    "registered_roi_xyxy",
                    "action_rule_version",
                }
                if missing_benchmark := sorted(required_benchmark - set(self.benchmark)):
                    raise ConfigError(
                        "publisher_source_rule benchmark section is missing: "
                        f"{missing_benchmark}"
                    )
            required_models = {"detector", "vlms", "upper_bound"}
            missing_models = sorted(required_models - set(self.models))
            if missing_models:
                raise ConfigError(f"full profile models section is missing: {missing_models}")
            if self.study_scope is StudyScope.VISDRONE_PRIMARY:
                from .perception import (
                    VISDRONE_GENERATION_POLICY,
                    VISDRONE_GENERATION_POLICY_SHA256,
                    VISDRONE_REGISTERED_MAX_NEW_TOKENS,
                )

                configured_policy = self.models.get("generation_policy")
                if not isinstance(configured_policy, Mapping):
                    raise ConfigError(
                        "visdrone_primary requires models.generation_policy"
                    )
                policy_payload = dict(configured_policy)
                configured_policy_sha256 = str(
                    policy_payload.pop("sha256", "")
                ).lower()
                if (
                    policy_payload != VISDRONE_GENERATION_POLICY
                    or configured_policy_sha256
                    != VISDRONE_GENERATION_POLICY_SHA256
                    or self.models.get("max_new_tokens")
                    != VISDRONE_REGISTERED_MAX_NEW_TOKENS
                ):
                    raise ConfigError(
                        "VisDrone VLM generation policy differs from the registered policy"
                    )
            if not self.evaluation.get("primary_wire_budget_bytes_per_s") or not self.evaluation.get("primary_energy_budget_j_per_s"):
                raise ConfigError("full profile requires positive frozen wire and energy budgets")
            energy_claim_scope = str(
                self.evaluation.get("energy_claim_scope", "total_system_measured")
            ).strip().lower()
            visdrone_compute_scope = (
                "combined_sender_and_ground_gpu_board_nvml_compute_only_excludes_radio_system"
            )
            allowed_energy_scopes = {
                "total_system_measured", "gpu_board_nvml_only", visdrone_compute_scope,
            }
            if energy_claim_scope not in allowed_energy_scopes:
                raise ConfigError(
                    "evaluation.energy_claim_scope is not a registered measurement scope"
                )
            if (
                self.study_scope is StudyScope.VISDRONE_PRIMARY
                and energy_claim_scope != visdrone_compute_scope
            ):
                raise ConfigError(
                    "visdrone_primary must use the combined sender-and-ground GPU-board "
                    "NVML compute-only scope until radio/system energy is measured"
                )
        elif self.profile is Profile.SMOKE_REAL:
            if self.allow_synthetic:
                raise ConfigError("smoke_real must set allow_synthetic: false")
            if any(spec.synthetic for spec in self.enabled_datasets):
                raise ConfigError("smoke_real cannot enable synthetic datasets")
        self.split.validate()
        self.annotations.validate()

    def prerequisite_errors(self, stage: str = "ingest") -> list[str]:
        """Return external readiness failures without mutating or probing accelerators.

        ``stage='ingest'`` checks only the source roots needed to construct the
        canonical manifest.  Annotation, telemetry, trace, and frozen-split
        artifacts are downstream products/inputs and are checked only when a
        validation, test, or embedded run consumes them.
        """

        if stage not in {"ingest", "validation", "test", "embedded"}:
            raise ConfigError("stage must be 'ingest', 'validation', 'test', or 'embedded'")
        if self.profile is Profile.DRY_RUN:
            return []
        errors: list[str] = []
        source_receipts: dict[str, Mapping[str, Any]] = {}
        for spec in self.enabled_datasets:
            if not spec.root.is_dir():
                errors.append(f"dataset {spec.name!r} root is missing: {spec.root}")
        consumes_prepared_inputs = stage in {"validation", "test", "embedded"}
        if consumes_prepared_inputs:
            if self.annotations.enabled and self.truth_mode is not TruthMode.PUBLISHER_SOURCE_RULE:
                annotation_path = (
                    self.paths.test_annotations
                    if stage in {"test", "embedded"}
                    else self.paths.annotations
                )
                if annotation_path is None or not annotation_path.is_file():
                    errors.append(f"annotation table is missing for {stage}: {annotation_path}")
            if self.paths.channel_traces is None or not self.paths.channel_traces.exists():
                errors.append(f"channel traces are missing: {self.paths.channel_traces}")
            if self.paths.telemetry_manifest is None or not self.paths.telemetry_manifest.is_file():
                errors.append(f"telemetry manifest is missing: {self.paths.telemetry_manifest}")
            for spec in self.enabled_datasets:
                if spec.synthetic or spec.adapter == "synthetic" or not spec.root.is_dir():
                    continue
                provenance_errors, receipt = _dataset_provenance_errors(spec)
                errors.extend(provenance_errors)
                if receipt is not None:
                    source_receipts[spec.name] = receipt
        if consumes_prepared_inputs and self.split.require_frozen_manifest:
            if self.paths.split_manifest is None or not self.paths.split_manifest.is_file():
                errors.append(f"frozen split manifest is missing: {self.paths.split_manifest}")
        if self.profile is Profile.FULL and consumes_prepared_inputs:
            allowed_confirmatory_scopes = {
                "visdrone": "publisher_acquisition_official_release_components",
                "uavid": "publisher_archive_dans_v1_confirmatory",
                "uavreason": "publisher_acquisition_pinned_huggingface_snapshots",
                "flame3": "complete_ieee_dataport_full_confirmatory_user_staged",
            }
            for spec in self.enabled_datasets:
                if spec.synthetic or spec.adapter == "synthetic":
                    continue
                receipt = source_receipts.get(spec.name)
                if receipt is None:
                    continue
                source = receipt.get("source", {})
                scope = str(source.get("scope", "")).strip().lower() if isinstance(source, Mapping) else ""
                adapter = spec.adapter.removeprefix("generic_")
                if scope != allowed_confirmatory_scopes.get(adapter):
                    errors.append(
                        f"dataset {spec.name!r} source scope {scope or 'MISSING_SCOPE'!r} "
                        f"is not confirmatory publisher/acquisition provenance for {stage}; "
                        "user-staged preparation receipts cannot certify a full-profile phase"
                    )
                errors.extend(_confirmatory_source_semantics_errors(spec, receipt))
            for spec in self.enabled_datasets:
                if spec.adapter not in {"flame3", "generic_flame3"}:
                    continue
                receipt_path = spec.root / "SOURCE_RECEIPT.json"
                receipt = source_receipts.get(spec.name)
                if receipt is None:
                    continue
                source = receipt.get("source", {})
                scope = str(source.get("scope", "")).strip().lower() if isinstance(source, Mapping) else ""
                complete_scope = scope.startswith("complete_ieee_dataport_full_confirmatory")
                if not complete_scope:
                    detail = scope or "missing source.scope"
                    errors.append(
                        "FLAME3 source receipt is not full-confirmatory IEEE DataPort scope "
                        f"({detail}); Kaggle pilot subsets cannot enter {stage}: {receipt_path}"
                    )
            configured_models = list(self.models.get("vlms", [])) + [self.models.get("upper_bound", {})]
            for model in configured_models:
                repo = model.get("repo", "unknown") if isinstance(model, Mapping) else "unknown"
                revision = model.get("revision", "") if isinstance(model, Mapping) else ""
                if not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
                    errors.append(f"model revision is not a resolved 40-character commit: {repo}")
            detector = self.models.get("detector", {})
            checkpoint = Path(str(detector.get("checkpoint", ""))) if isinstance(detector, Mapping) else Path("")
            if not checkpoint.is_file():
                errors.append(f"detector checkpoint is missing: {checkpoint}")
            for name in ("primary_wire_budget_bytes_per_s", "primary_energy_budget_j_per_s"):
                try:
                    valid_budget = float(self.evaluation.get(name, 0)) > 0
                except (TypeError, ValueError):
                    valid_budget = False
                if not valid_budget:
                    errors.append(f"frozen positive evaluation budget is missing: {name}")
            energy_claim_scope = str(
                self.evaluation.get("energy_claim_scope", "total_system_measured")
            ).strip().lower()
            if energy_claim_scope == "total_system_measured":
                for name in ("radio_tx_j_per_byte", "radio_rx_j_per_byte"):
                    try:
                        valid_radio = float(self.hardware.get(name, 0)) > 0
                    except (TypeError, ValueError):
                        valid_radio = False
                    if not valid_radio:
                        errors.append(
                            f"measured positive radio-energy coefficient is missing: {name}"
                        )
        if stage == "embedded":
            target_log = self.hardware.get("target_log")
            if not target_log or not Path(str(target_log)).is_file():
                errors.append(f"target-hardware log is missing: {target_log}")
            hardware_id = str(self.hardware.get("hardware_id", "")).strip()
            if not hardware_id or hardware_id.startswith("REQUIRED_"):
                errors.append("a concrete target hardware asset ID is missing: hardware.hardware_id")
        return errors

    def assert_ready(self, stage: str = "ingest") -> None:
        """Hard gate at the dry-run/real-study boundary."""

        self.validate()
        failures = self.prerequisite_errors(stage)
        if failures:
            bullets = "\n".join(f"  - {item}" for item in failures)
            raise PrerequisiteError(
                f"{self.profile.value} profile cannot start ({stage} gate failed):\n{bullets}"
            )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-friendly snapshot with resolved paths."""

        def plain(value: Any) -> Any:
            if dataclasses.is_dataclass(value):
                return {field.name: plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, Mapping):
                return {str(key): plain(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [plain(item) for item in value]
            return value

        result = plain(self)
        result.pop("raw", None)
        return result


_ENV_WITH_DEFAULT = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")
_UNRESOLVED_ENV = re.compile(r"\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)")


def _expand_string(value: str) -> str:
    value = _ENV_WITH_DEFAULT.sub(lambda match: os.environ.get(match.group(1), match.group(2)), value)
    return os.path.expanduser(os.path.expandvars(value))


def _expand_nested(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_string(value)
    if isinstance(value, Mapping):
        return {str(key): _expand_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_nested(item) for item in value)
    return value


def _check_unresolved(value: str, field: str) -> None:
    match = _UNRESOLVED_ENV.search(value)
    if match:
        raise ConfigError(f"Unresolved environment variable {match.group(0)!r} in {field}")


def _path(value: Any, base: Path, field: str, *, optional: bool = False) -> Path | None:
    if value in (None, ""):
        if optional:
            return None
        raise ConfigError(f"{field} is required")
    expanded = _expand_string(str(value))
    _check_unresolved(expanded, field)
    result = Path(expanded)
    if not result.is_absolute():
        result = base / result
    return result.resolve()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field} must be a mapping")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError(f"{field} must be a sequence")
    return value


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"Unknown key(s) in {field}: {unknown}")


def load_study_config(
    path: str | Path,
    *,
    profile: str | Profile | None = None,
    validate_prerequisites: bool = False,
    prerequisite_stage: str = "ingest",
) -> StudyConfig:
    """Load a YAML configuration into immutable typed dataclasses.

    Set ``validate_prerequisites=True`` at an execution boundary.  The default
    remains parse-only so an interactive notebook can render all missing-input
    diagnostics in one pre-flight table.
    """

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw = _mapping(loaded or {}, "root")
    _reject_unknown(
        raw,
        {
            "schema_version",
            "study_id",
            "profile",
            "study_scope",
            "truth_mode",
            "seed",
            "seeds",
            "allow_synthetic",
            "paths",
            "datasets",
            "split",
            "annotations",
            "runtime",
            "models",
            "payloads",
            "network",
            "controller",
            "evaluation",
            "hardware",
            "benchmark",
        },
        "root",
    )
    # The selected YAML already encodes its profile.  A process-wide
    # SEMUAV_PROFILE is used by the notebook to choose the YAML file, but must
    # not silently reinterpret other configs (notably acceptance tests).
    selected_profile = Profile.parse(profile or raw.get("profile", "dry_run"))
    selected_scope = StudyScope.parse(raw.get("study_scope", "full_proposal"))
    selected_truth_mode = TruthMode.parse(
        raw.get("truth_mode", "adjudicated_mission_labels")
    )

    paths_raw = _mapping(raw.get("paths", {}), "paths")
    _reject_unknown(
        paths_raw,
        {
            "project_root",
            "data_root",
            "output_root",
            "cache_root",
            "annotations",
            "test_annotations",
            "split_manifest",
            "channel_traces",
            "telemetry_manifest",
        },
        "paths",
    )
    project_root = _path(paths_raw.get("project_root", ".."), config_path.parent, "paths.project_root")
    assert project_root is not None
    data_root = _path(paths_raw.get("data_root", "data"), project_root, "paths.data_root")
    output_root = _path(paths_raw.get("output_root", "outputs"), project_root, "paths.output_root")
    cache_root = _path(paths_raw.get("cache_root", ".cache"), project_root, "paths.cache_root")
    assert data_root is not None and output_root is not None and cache_root is not None
    paths = PathsConfig(
        project_root=project_root,
        data_root=data_root,
        output_root=output_root,
        cache_root=cache_root,
        annotations=_path(paths_raw.get("annotations"), project_root, "paths.annotations", optional=True),
        test_annotations=_path(paths_raw.get("test_annotations"), project_root, "paths.test_annotations", optional=True),
        split_manifest=_path(
            paths_raw.get("split_manifest"), project_root, "paths.split_manifest", optional=True
        ),
        channel_traces=_path(
            paths_raw.get("channel_traces"), data_root, "paths.channel_traces", optional=True
        ),
        telemetry_manifest=_path(
            paths_raw.get("telemetry_manifest"), data_root, "paths.telemetry_manifest", optional=True
        ),
    )

    datasets_raw = _mapping(raw.get("datasets", {}), "datasets")
    datasets: list[DatasetSpec] = []
    for name, value in datasets_raw.items():
        item = _mapping(value, f"datasets.{name}")
        _reject_unknown(
            item,
            {"adapter", "root", "mission", "enabled", "required", "synthetic", "options"},
            f"datasets.{name}",
        )
        adapter = str(item.get("adapter", name)).strip().lower()
        dataset_root = _path(item.get("root", name), data_root, f"datasets.{name}.root")
        assert dataset_root is not None
        options = _mapping(item.get("options", {}), f"datasets.{name}.options")
        datasets.append(
            DatasetSpec(
                name=str(name),
                adapter=adapter,
                root=dataset_root,
                mission=str(item.get("mission", "urban_situational_awareness")),
                enabled=bool(item.get("enabled", True)),
                required=bool(item.get("required", True)),
                synthetic=bool(item.get("synthetic", adapter == "synthetic")),
                options=dict(options),
            )
        )

    split_raw = _mapping(raw.get("split", {}), "split")
    _reject_unknown(
        split_raw,
        {
            "seed",
            "ratios",
            "leakage_keys",
            "strata",
            "require_frozen_manifest",
            "min_test_units",
            "min_critical_test_events",
        },
        "split",
    )
    ratios = _mapping(split_raw.get("ratios", SplitConfig().ratios), "split.ratios")
    split = SplitConfig(
        seed=int(split_raw.get("seed", raw.get("seed", 20260823))),
        ratios={str(key): float(value) for key, value in ratios.items()},
        leakage_keys=tuple(
            str(value) for value in _sequence(split_raw.get("leakage_keys", SplitConfig().leakage_keys), "split.leakage_keys")
        ),
        strata=tuple(
            str(value) for value in _sequence(split_raw.get("strata", SplitConfig().strata), "split.strata")
        ),
        require_frozen_manifest=bool(split_raw.get("require_frozen_manifest", True)),
        min_test_units=int(split_raw.get("min_test_units", 500)),
        min_critical_test_events=int(split_raw.get("min_critical_test_events", 140)),
    )

    annotations_raw = _mapping(raw.get("annotations", {}), "annotations")
    annotation_fields = {field.name for field in dataclasses.fields(AnnotationConfig)}
    _reject_unknown(annotations_raw, annotation_fields, "annotations")
    annotations = AnnotationConfig(**{key: value for key, value in annotations_raw.items()})

    seeds_value = raw.get("seeds", [raw.get("seed", 20260823)])
    seeds = tuple(int(value) for value in _sequence(seeds_value, "seeds"))
    config = StudyConfig(
        schema_version=int(raw.get("schema_version", 1)),
        study_id=str(raw.get("study_id", "semuav_vlm")),
        profile=selected_profile,
        study_scope=selected_scope,
        truth_mode=selected_truth_mode,
        seed=int(raw.get("seed", seeds[0])),
        seeds=seeds,
        allow_synthetic=bool(raw.get("allow_synthetic", selected_profile is Profile.DRY_RUN)),
        paths=paths,
        datasets=tuple(datasets),
        split=split,
        annotations=annotations,
        runtime=dict(_expand_nested(_mapping(raw.get("runtime", {}), "runtime"))),
        models=dict(_expand_nested(_mapping(raw.get("models", {}), "models"))),
        payloads=dict(_expand_nested(_mapping(raw.get("payloads", {}), "payloads"))),
        network=dict(_expand_nested(_mapping(raw.get("network", {}), "network"))),
        controller=dict(_expand_nested(_mapping(raw.get("controller", {}), "controller"))),
        evaluation=dict(_expand_nested(_mapping(raw.get("evaluation", {}), "evaluation"))),
        hardware=dict(_expand_nested(_mapping(raw.get("hardware", {}), "hardware"))),
        benchmark=dict(_expand_nested(_mapping(raw.get("benchmark", {}), "benchmark"))),
        raw=dict(raw),
    )
    if validate_prerequisites:
        # ``assert_ready`` performs schema validation before checking external
        # prerequisites, so do not validate the same configuration twice.
        config.assert_ready(prerequisite_stage)
    else:
        config.validate()
    return config


# Concise alias for notebook code.
load_config = load_study_config


__all__ = [
    "AnnotationConfig",
    "ConfigError",
    "DatasetSpec",
    "PathsConfig",
    "PrerequisiteError",
    "Profile",
    "SplitConfig",
    "StudyConfig",
    "StudyScope",
    "TruthMode",
    "load_config",
    "load_study_config",
]
