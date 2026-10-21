"""Fail-closed orchestration helpers for the real-data VisDrone paper study.

The functions in this module bridge the publisher-grounded records in
``semuav.visdrone_paper`` to the canonical manifests consumed by Experiments
1--6.  They intentionally make only *sensor-space* claims supported by the
official VisDrone boxes/tracks and a frozen image-plane rule.  In particular,
they never create accident, emergency-dispatch, road-obstruction, wildfire,
physical-proximity, or cross-domain OOD labels.

Publisher ``test-dev`` remains behind the existing hash-bound, one-time test
ledger.  Every public path that can read test annotations or image metadata
revalidates the open ledger before doing so.
"""

from __future__ import annotations

import csv
import dataclasses
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from .network import (
    Direction,
    EndBehavior,
    LinkSegment,
    LinkTrace,
    TraceSemantics,
    scientific_episode_coverage_audit,
    scientific_trace_design_audit,
)
from .registry import StudyGateError, canonical_json, hash_file, hash_object
from .visdrone_paper import (
    ACTION_NO_ACTION,
    ACTION_RESPOND,
    ACTION_REVIEW,
    DEVELOPMENT_COMPONENTS,
    EVENT_IMAGE_PLANE_OVERLAP,
    EVENT_NO_PRIORITY_VISIBLE,
    EVENT_PRIORITY_EVIDENCE_LIMITED,
    EVENT_PRIORITY_OUTSIDE_ROI,
    EVENT_PRIORITY_ROI_OCCUPANCY,
    MISSION_VISDRONE_URBAN,
    REGISTERED_MISSION_PROFILES,
    RELATION_IMAGE_PLANE_OVERLAP,
    UNTOUCHED_TEST_COMPONENTS,
    VISDRONE_COMPONENTS,
    NormalizedROI,
    RegisteredBenchmarkPolicy,
    SourceDerivedTruth,
    UntouchedTestAccessError,
    UntouchedTestPermit,
    build_development_manifest,
    build_untouched_test_manifest,
    derive_source_truth,
    load_untouched_test_permit,
    parse_det_annotations,
    parse_det_row,
    parse_mot_annotations,
)


WORKFLOW_VERSION = "semuav.visdrone-paper-workflow.v1"
PAPER_COMPLETION_RECEIPT_SCHEMA = "semuav.paper-completion-receipt.v1"
TRUTH_SCOPE = "publisher_2d_boxes_registered_sensor_rule"
SOURCE_PROVENANCE_SCOPE = "publisher_acquisition_official_release_components"
REQUIRED_SOURCE_COMPONENTS = frozenset(VISDRONE_COMPONENTS)

# Publisher-controlled acquisition coordinates recorded in
# ``configs/datasets.yaml``. Repeating them here keeps the scientific gate
# independent of a mutable run-time dataset registry.
_OFFICIAL_VISDRONE_ARTIFACTS: Mapping[str, tuple[str, str]] = {
    "det_train": (
        "VisDrone2019-DET-train.zip",
        "1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn",
    ),
    "det_val": (
        "VisDrone2019-DET-val.zip",
        "1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59",
    ),
    "det_test_dev": (
        "VisDrone2019-DET-test-dev.zip",
        "1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V",
    ),
    "mot_train": (
        "VisDrone2019-MOT-train.zip",
        "1-qX2d-P1Xr64ke6nTdlm33om1VxCUTSh",
    ),
    "mot_val": (
        "VisDrone2019-MOT-val.zip",
        "1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu",
    ),
    "mot_test_dev": (
        "VisDrone2019-MOT-test-dev.zip",
        "14z8Acxopj1d86-qhsF1NwS4Bv3KYa4Wu",
    ),
}
MEDIA_PARTITION_AUDIT_SCHEMA = "semuav.visdrone-media-partition-audit.v1"
MINIMUM_TEST_NATURAL_GROUPS = 30
_VISDRONE_MEDIA_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
_PREPARE_PROGRESS_INTERVAL = 1_000


_PrepareProgress = Callable[[str, int, int], None]


def _notify_prepare_progress(
    progress: _PrepareProgress | None,
    stage: str,
    completed: int,
    total: int,
) -> None:
    """Report bounded within-stage progress without changing scientific state."""

    if progress is None:
        return
    done = int(completed)
    count = int(total)
    if done == 1 or done == count or done % _PREPARE_PROGRESS_INTERVAL == 0:
        progress(str(stage), done, count)

VALIDATION_TRACE_FAMILIES: tuple[str, ...] = (
    "steady_low",
    "steady_mid",
    "steady_high",
    "periodic_congestion",
    "bursty_loss",
    "delay_spike",
)
TEST_TRACE_FAMILIES: tuple[str, ...] = (
    "sudden_drop",
    "recovery",
    "ramp",
    "markov",
    "fading",
    "outage",
)

TEMPORAL_SAMPLING_SCHEMA = "semuav.visdrone-temporal-controller.v1"
TEMPORAL_SAMPLING_METHOD = "outcome_blind_mot_sequence_hash_contiguous_v1"
TEMPORAL_COMPONENT_BY_SPLIT: Mapping[str, str] = {
    "validation": "mot_val",
    "test": "mot_test_dev",
}
REGISTERED_TEMPORAL_TRACE_DURATION_US = 18_000_000
REGISTERED_TEMPORAL_TRACE_DYNAMIC_US = 12_000_000
REGISTERED_TEMPORAL_TRACE_SEGMENT_US = 1_000_000

PROHIBITED_CLAIMS: tuple[str, ...] = (
    "accident_detection",
    "emergency_dispatch",
    "physical_proximity",
    "road_obstruction",
    "wildfire",
    "cross_dataset_ood",
)

_HEX64 = re.compile(r"[0-9a-f]{64}")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

ANNOTATION_QUALITY_POLICY_VERSION = "visdrone.det-exact-zero-quality.v1"
ANNOTATION_EXCLUSIONS_FILENAME = "PUBLISHER_ANNOTATION_EXCLUSIONS.csv"
ANNOTATION_EXCLUSION_COLUMNS: tuple[str, ...] = (
    "policy_version",
    "component",
    "sample_id",
    "source_uid",
    "annotation_relative_path",
    "frame_index",
    "source_line",
    "raw_row_sha256",
    "bbox_width",
    "bbox_height",
    "score",
    "category_id",
    "publisher_ignored",
    "action",
    "reason",
)
_EXCLUDE_ROW = "exclude_annotation_row"
_EXCLUDE_SAMPLE = "exclude_source_sample"
_IGNORED_ZERO_REASON = "publisher_ignored_exact_zero_extent_bbox"
_EVALUATED_ZERO_REASON = "publisher_evaluated_exact_zero_extent_bbox"


@dataclasses.dataclass(frozen=True)
class PreparedVisDroneBundle:
    """Paths for one immutable preparation-phase output bundle."""

    output_dir: Path
    source_manifest: Path
    split_manifest: Path
    source_truth: Path
    analysis_manifest: Path
    trace_manifest: Path
    split_audit: Path
    trace_audit: Path
    annotation_exclusions: Path
    artifact_manifest: Path
    receipt: Path
    partition: str
    test_permit: UntouchedTestPermit | None = dataclasses.field(
        default=None, repr=False, compare=False,
    )

    def _authorize_content_read(self) -> None:
        if self.partition == "test":
            _verify_test_permit(self.test_permit)

    def load_source_manifest(self) -> pd.DataFrame:
        self._authorize_content_read()
        return pd.read_csv(self.source_manifest, compression="gzip")

    def load_source_truth(self) -> pd.DataFrame:
        self._authorize_content_read()
        return pd.read_csv(self.source_truth, compression="gzip")

    def load_analysis_manifest(self) -> pd.DataFrame:
        self._authorize_content_read()
        return read_analysis_manifest(self.analysis_manifest)

    def load_trace_manifest(self) -> pd.DataFrame:
        return pd.read_csv(self.trace_manifest)

    def load_annotation_exclusions(self) -> pd.DataFrame:
        self._authorize_content_read()
        return pd.read_csv(self.annotation_exclusions, keep_default_na=False)


def _atomic_bytes(path: Path, payload: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def _atomic_text(path: Path, text: str) -> Path:
    return _atomic_bytes(path, text.encode("utf-8"))


def _atomic_json(path: Path, value: Any) -> Path:
    return _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _deterministic_csv(frame: pd.DataFrame, path: Path, *, compress: bool = False) -> Path:
    text = frame.to_csv(index=False, lineterminator="\n", float_format="%.12g")
    payload = gzip.compress(text.encode("utf-8"), compresslevel=6, mtime=0) if compress else text.encode("utf-8")
    return _atomic_bytes(path, payload)


def _stable_unit_interval(*parts: Any) -> float:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _verify_test_permit(
    permit: UntouchedTestPermit | None,
    frame: pd.DataFrame | None = None,
) -> UntouchedTestPermit:
    if permit is None:
        raise UntouchedTestAccessError(
            "Publisher test-dev requires the active hash-bound one-time test permit"
        )
    current = load_untouched_test_permit(
        permit.ledger_path,
        expected_protocol_hash=permit.protocol_hash,
        expected_config_hash=permit.config_hash,
        expected_code_hash=permit.code_hash,
    )
    if current.ledger_sha256 != permit.ledger_sha256:
        raise UntouchedTestAccessError("Test-access ledger changed after authorization")
    if frame is not None and not frame.empty:
        if "test_ledger_sha256" not in frame:
            raise UntouchedTestAccessError("Test rows are not bound to a test-access ledger")
        observed = set(frame["test_ledger_sha256"].fillna("").astype(str))
        if observed != {permit.ledger_sha256}:
            raise UntouchedTestAccessError("Test rows are bound to a different test-access ledger")
    return current


def _receipt_integer(value: Any, *, field: str, minimum: int = 0) -> int:
    """Parse an integer receipt field without accepting booleans or floats."""

    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StudyGateError(f"VisDrone source receipt has invalid {field}")
    return value


def _safe_inventory_path(value: Any, *, root: Path, line_number: int) -> tuple[str, Path]:
    """Resolve one canonical POSIX inventory path below ``root``."""

    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise StudyGateError(
            f"VisDrone source inventory line {line_number} has an unsafe path"
        )
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise StudyGateError(
            f"VisDrone source inventory line {line_number} has a non-canonical path"
        )
    path = root.joinpath(*relative.parts)
    try:
        path.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise StudyGateError(
            f"VisDrone source inventory line {line_number} escapes the dataset root"
        ) from exc
    return value, path


def _component_for_inventory_path(relative: str) -> tuple[str | None, tuple[str, ...]]:
    """Return the official component and path relative to its directory."""

    parts = PurePosixPath(relative).parts
    matches = [
        (component, index)
        for index, part in enumerate(parts)
        for component, spec in VISDRONE_COMPONENTS.items()
        if part == spec.directory_name
    ]
    if not matches:
        return None, ()
    if len(matches) != 1:
        raise StudyGateError(f"Ambiguous VisDrone component path in inventory: {relative}")
    component, index = matches[0]
    return component, tuple(parts[index + 1:])


def verify_visdrone_source_provenance(
    data_root: str | Path,
    *,
    full_integrity: bool = False,
    components: Sequence[str] | None = None,
    progress: _PrepareProgress | None = None,
) -> Mapping[str, Any]:
    """Verify the official receipt and its complete source-tree inventory.

    The lightweight form still checks the receipt/inventory hashes, envelope,
    canonical unique paths, exact tree membership, symlink absence, and every
    listed file size. ``full_integrity=True`` additionally recomputes every
    source-file SHA-256 and requires nonempty content beneath all six official
    component directories. Scientific prepare/validation/test entry points
    use the full form; the lighter form exists for fast status checks and
    explicitly non-scientific fixtures.
    """

    requested_components = (
        frozenset(REQUIRED_SOURCE_COMPONENTS)
        if components is None else
        frozenset(str(component) for component in components)
    )
    unknown_components = sorted(requested_components - REQUIRED_SOURCE_COMPONENTS)
    if unknown_components:
        raise ValueError(f"Unknown VisDrone integrity components: {unknown_components}")
    if full_integrity and not requested_components:
        raise ValueError("Full VisDrone integrity verification requires at least one component")

    unresolved_root = Path(data_root).expanduser()
    if unresolved_root.is_symlink():
        raise StudyGateError("VisDrone dataset root must not be a symbolic link")
    root = unresolved_root.resolve()
    if not root.is_dir():
        raise StudyGateError(f"VisDrone dataset root is missing: {root}")
    receipt_path = root / "SOURCE_RECEIPT.json"
    inventory_path = root / "SOURCE_FILES_SHA256.jsonl"
    if receipt_path.is_symlink() or inventory_path.is_symlink():
        raise StudyGateError("VisDrone provenance files must not be symbolic links")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StudyGateError(f"VisDrone source receipt is missing or invalid: {receipt_path}") from exc
    if not isinstance(receipt, Mapping):
        raise StudyGateError("VisDrone source receipt must be a JSON object")
    if receipt.get("schema_version") != "semuav.dataset-source.v1":
        raise StudyGateError("VisDrone source receipt has an incompatible schema_version")
    if receipt.get("dataset") != "visdrone":
        raise StudyGateError("VisDrone source receipt has the wrong dataset identity")
    source = receipt.get("source")
    if not isinstance(source, Mapping) or source.get("scope") != SOURCE_PROVENANCE_SCOPE:
        raise StudyGateError(
            "VisDrone paper evidence requires the official six-component acquisition receipt"
        )

    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        raise StudyGateError("VisDrone source receipt has no component artifacts")
    by_component: dict[str, Mapping[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise StudyGateError("VisDrone source receipt contains a malformed artifact")
        if "component" not in item:
            # Acquisition ``prepare`` receipts may carry a non-component audit
            # record; it cannot substitute for an official component.
            continue
        component = str(item["component"])
        if component not in REQUIRED_SOURCE_COMPONENTS:
            raise StudyGateError(f"Unknown VisDrone receipt component: {component}")
        if component in by_component:
            raise StudyGateError(f"Duplicate VisDrone receipt component: {component}")
        by_component[component] = item
    observed_components = frozenset(by_component)
    if observed_components != REQUIRED_SOURCE_COMPONENTS:
        missing = sorted(REQUIRED_SOURCE_COMPONENTS.difference(observed_components))
        raise StudyGateError(f"VisDrone source receipt lacks official components: {missing}")
    for component in sorted(REQUIRED_SOURCE_COMPONENTS):
        item = by_component[component]
        digest = str(item.get("sha256", "")).lower()
        _receipt_integer(
            item.get("bytes"), field=f"artifact bytes for {component}", minimum=1,
        )
        if _HEX64.fullmatch(digest) is None:
            raise StudyGateError(f"VisDrone source receipt has invalid hash for {component}")
        expected_archive, expected_drive_id = _OFFICIAL_VISDRONE_ARTIFACTS[component]
        if Path(str(item.get("archive", ""))).name != expected_archive:
            raise StudyGateError(f"VisDrone source receipt has wrong archive for {component}")
        if str(item.get("google_drive_id", "")) != expected_drive_id:
            raise StudyGateError(f"VisDrone source receipt has wrong publisher ID for {component}")

    content_manifest = receipt.get("content_hash_manifest")
    inventory_envelope = receipt.get("inventory")
    if not isinstance(content_manifest, Mapping) or not isinstance(inventory_envelope, Mapping):
        raise StudyGateError("VisDrone source receipt lacks its hash-inventory envelope")
    recorded_inventory_path = content_manifest.get("path")
    if not isinstance(recorded_inventory_path, str) or not recorded_inventory_path:
        raise StudyGateError("VisDrone source receipt has an invalid inventory path")
    candidate_inventory = Path(recorded_inventory_path).expanduser()
    if not candidate_inventory.is_absolute():
        candidate_inventory = root / candidate_inventory
    if candidate_inventory.resolve(strict=False) != inventory_path:
        raise StudyGateError("VisDrone source receipt points to another hash inventory")
    try:
        inventory_stat = inventory_path.stat()
    except OSError as exc:
        raise StudyGateError(f"VisDrone source hash inventory is missing: {inventory_path}") from exc
    recorded_inventory_sha = str(content_manifest.get("sha256", "")).lower()
    if _HEX64.fullmatch(recorded_inventory_sha) is None:
        raise StudyGateError("VisDrone source receipt has an invalid inventory SHA-256")
    actual_inventory_sha = hash_file(inventory_path)
    if actual_inventory_sha != recorded_inventory_sha:
        raise StudyGateError("VisDrone source hash inventory does not match its receipt")

    records: dict[str, tuple[Path, int, str]] = {}
    total_bytes = 0
    previous_path: str | None = None
    try:
        with inventory_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.endswith("\n") or not raw_line.strip():
                    raise StudyGateError(
                        f"VisDrone source inventory line {line_number} is blank or unterminated"
                    )
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise StudyGateError(
                        f"VisDrone source inventory line {line_number} is invalid JSON"
                    ) from exc
                if not isinstance(record, Mapping):
                    raise StudyGateError(
                        f"VisDrone source inventory line {line_number} is not an object"
                    )
                relative, path = _safe_inventory_path(
                    record.get("path"), root=root, line_number=line_number,
                )
                if relative in records:
                    raise StudyGateError(f"Duplicate VisDrone source inventory path: {relative}")
                if previous_path is not None and relative <= previous_path:
                    raise StudyGateError("VisDrone source inventory paths are not strictly sorted")
                previous_path = relative
                size = _receipt_integer(
                    record.get("bytes"), field=f"inventory bytes at line {line_number}", minimum=0,
                )
                digest = str(record.get("sha256", "")).lower()
                if _HEX64.fullmatch(digest) is None:
                    raise StudyGateError(
                        f"VisDrone source inventory line {line_number} has an invalid SHA-256"
                    )
                records[relative] = (path, size, digest)
                total_bytes += size
    except (OSError, UnicodeError) as exc:
        raise StudyGateError(f"VisDrone source hash inventory is unreadable: {inventory_path}") from exc

    recorded_files = _receipt_integer(
        content_manifest.get("files"), field="content_hash_manifest.files", minimum=0,
    )
    recorded_bytes = _receipt_integer(
        content_manifest.get("bytes"), field="content_hash_manifest.bytes", minimum=0,
    )
    if recorded_files != len(records) or recorded_bytes != total_bytes:
        raise StudyGateError("VisDrone content hash manifest counts do not match its records")
    envelope_files = _receipt_integer(
        inventory_envelope.get("files"), field="inventory.files", minimum=1,
    )
    envelope_bytes = _receipt_integer(
        inventory_envelope.get("bytes"), field="inventory.bytes", minimum=0,
    )
    if envelope_files != len(records) + 1:
        raise StudyGateError("VisDrone receipt inventory file count has the wrong envelope")
    if envelope_bytes != total_bytes + inventory_stat.st_size:
        raise StudyGateError("VisDrone receipt inventory byte count has the wrong envelope")

    actual_files: dict[str, Path] = {}
    for entry in root.rglob("*"):
        if entry.is_symlink():
            raise StudyGateError(f"VisDrone source tree contains a symbolic link: {entry}")
        if entry.is_dir():
            continue
        if not entry.is_file():
            raise StudyGateError(f"VisDrone source tree contains an unsupported entry: {entry}")
        if entry in {receipt_path, inventory_path}:
            continue
        relative = entry.relative_to(root).as_posix()
        actual_files[relative] = entry
    listed = set(records)
    actual = set(actual_files)
    if listed != actual:
        missing = sorted(listed - actual)[:5]
        unlisted = sorted(actual - listed)[:5]
        raise StudyGateError(
            "VisDrone source tree and inventory differ "
            f"(missing={missing}, unlisted={unlisted})"
        )
    integrity_total = sum(
        _component_for_inventory_path(relative)[0] in requested_components
        for relative in records
    ) if full_integrity else 0
    integrity_completed = 0
    for relative, (path, size, digest) in records.items():
        if path.is_symlink() or not path.is_file():
            raise StudyGateError(f"VisDrone source inventory target is not a regular file: {relative}")
        before = path.stat()
        if before.st_size != size:
            raise StudyGateError(f"VisDrone source file size differs from inventory: {relative}")
        component, _ = _component_for_inventory_path(relative)
        if full_integrity and component in requested_components:
            actual_digest = hash_file(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise StudyGateError(f"VisDrone source file changed during verification: {relative}")
            if actual_digest != digest:
                raise StudyGateError(f"VisDrone source file SHA-256 differs from inventory: {relative}")
            integrity_completed += 1
            _notify_prepare_progress(
                progress,
                "source_integrity_files",
                integrity_completed,
                integrity_total,
            )

    if full_integrity:
        component_counts = {component: 0 for component in requested_components}
        for relative in records:
            component, _ = _component_for_inventory_path(relative)
            if component in component_counts:
                component_counts[component] += 1
        empty = sorted(component for component, count in component_counts.items() if count == 0)
        if empty:
            raise StudyGateError(
                f"VisDrone source inventory has no files for official components: {empty}"
            )

    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": hash_file(receipt_path),
        "inventory_path": str(inventory_path),
        "inventory_sha256": actual_inventory_sha,
        "source_files": len(records),
        "source_bytes": total_bytes,
        "full_integrity": bool(full_integrity),
        "full_integrity_components": sorted(requested_components) if full_integrity else [],
        "scope": SOURCE_PROVENANCE_SCOPE,
        "components": sorted(REQUIRED_SOURCE_COMPONENTS),
    }


def _inventory_media_audit_frames(
    root: Path,
    *,
    inventory_path: Path,
    seed: int,
    calibration_fraction: float,
    minimum_test_groups: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build outcome-blind split/hash rows from the custodian inventory only."""

    if not 0.0 < float(calibration_fraction) < 1.0:
        raise ValueError("calibration_fraction must lie strictly within (0, 1)")
    if int(minimum_test_groups) <= 0:
        raise ValueError("minimum_test_groups must be positive")
    rows: list[dict[str, Any]] = []
    with inventory_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            record = json.loads(raw_line)
            relative, _ = _safe_inventory_path(
                record.get("path"), root=root, line_number=line_number,
            )
            component, within_component = _component_for_inventory_path(relative)
            if component is None or not within_component:
                continue
            spec = VISDRONE_COMPONENTS[component]
            suffix = PurePosixPath(within_component[-1]).suffix.lower()
            if suffix not in _VISDRONE_MEDIA_SUFFIXES:
                continue
            if spec.task == "DET":
                if len(within_component) != 2 or within_component[0] != "images":
                    continue
                stem = PurePosixPath(within_component[-1]).stem
                natural_identity = stem.split("_", 1)[0]
            elif spec.task == "MOT":
                if len(within_component) != 3 or within_component[0] != "sequences":
                    continue
                natural_identity = within_component[1]
            else:  # pragma: no cover - frozen registry excludes this branch
                continue
            split_group = f"visdrone:{component}:{natural_identity}"
            publisher_partition = spec.publisher_partition.value
            if publisher_partition == "train":
                score = _stable_unit_interval(
                    WORKFLOW_VERSION, int(seed), component, split_group,
                )
                study_partition = (
                    "calibration" if score < float(calibration_fraction) else "train"
                )
            elif publisher_partition == "val":
                study_partition = "validation"
            elif publisher_partition == "test-dev":
                study_partition = "test"
            else:  # pragma: no cover - frozen registry excludes this branch
                raise StudyGateError(f"Unsupported publisher partition: {publisher_partition}")
            digest = str(record.get("sha256", "")).lower()
            if _HEX64.fullmatch(digest) is None:
                raise StudyGateError(f"Invalid media SHA-256 in source inventory: {relative}")
            rows.append({
                "component": component,
                "publisher_partition": publisher_partition,
                "study_partition": study_partition,
                "split_group": split_group,
                "relative_media_path": relative,
                "media_sha256": digest,
                "bytes": _receipt_integer(
                    record.get("bytes"), field=f"media bytes at line {line_number}", minimum=0,
                ),
            })
    media = pd.DataFrame.from_records(rows)
    if media.empty:
        raise StudyGateError("VisDrone source inventory contains no recognized media")
    if media["relative_media_path"].astype(str).duplicated().any():
        raise StudyGateError("VisDrone media inventory paths are duplicated")
    hash_partitions = media.groupby("media_sha256", sort=True)["study_partition"].agg(
        lambda values: tuple(sorted(set(map(str, values))))
    )
    cross_hashes = set(hash_partitions[hash_partitions.map(len).gt(1)].index.astype(str))
    media["duplicate_study_partitions"] = media["media_sha256"].map(
        lambda digest: canonical_json(list(hash_partitions.loc[str(digest)]))
    )
    media["cross_partition_duplicate"] = media["media_sha256"].astype(str).isin(cross_hashes)
    media = media.sort_values(
        ["study_partition", "component", "split_group", "relative_media_path"],
        kind="stable",
        ignore_index=True,
    )
    test = media.loc[media["study_partition"].astype(str).eq("test")]
    capacity_rows: list[dict[str, Any]] = []
    for component in UNTOUCHED_TEST_COMPONENTS:
        groups = int(
            test.loc[test["component"].astype(str).eq(component), "split_group"].nunique()
        )
        capacity_rows.append({
            "scope": component,
            "available_natural_groups": groups,
            "required_natural_groups": 1,
            "passed": groups >= 1,
            "evidence_source": "SOURCE_FILES_SHA256_path_identity_only",
        })
    total_groups = int(test[["component", "split_group"]].drop_duplicates().shape[0])
    capacity_rows.append({
        "scope": "all_test_components",
        "available_natural_groups": total_groups,
        "required_natural_groups": int(minimum_test_groups),
        "passed": total_groups >= int(minimum_test_groups),
        "evidence_source": "SOURCE_FILES_SHA256_path_identity_only",
    })
    capacity = pd.DataFrame.from_records(capacity_rows)
    return media, capacity


def write_visdrone_media_partition_audit(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 20260823,
    calibration_fraction: float = 0.20,
    minimum_test_groups: int = MINIMUM_TEST_NATURAL_GROUPS,
    test_permit: UntouchedTestPermit | None = None,
    require_test_permit: bool = False,
) -> Mapping[str, Any]:
    """Freeze the inventory-only media leakage and test-capacity audit.

    No image pixels or annotation files are opened. When
    ``require_test_permit`` is true, authorization is checked before even the
    receipt/inventory metadata is read. The artifacts are written before a
    failing duplicate/capacity gate is raised so diagnostics remain available
    to direct callers.
    """

    if require_test_permit:
        _verify_test_permit(test_permit)
    root = Path(data_root).expanduser().resolve()
    provenance = verify_visdrone_source_provenance(root)
    media, capacity = _inventory_media_audit_frames(
        root,
        inventory_path=Path(str(provenance["inventory_path"])),
        seed=int(seed),
        calibration_fraction=float(calibration_fraction),
        minimum_test_groups=int(minimum_test_groups),
    )
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    media_path = _deterministic_csv(
        media, destination / "visdrone_media_partition_hash_audit.csv.gz", compress=True,
    )
    capacity_path = _deterministic_csv(
        capacity, destination / "visdrone_test_group_capacity_audit.csv",
    )
    duplicate_hashes = int(
        media.loc[media["cross_partition_duplicate"].astype(bool), "media_sha256"].nunique()
    )
    capacity_passed = bool(capacity["passed"].astype(bool).all())
    passed = duplicate_hashes == 0 and capacity_passed
    receipt_path = destination / "VISDRONE_MEDIA_PARTITION_AUDIT.json"
    receipt = {
        "schema_version": MEDIA_PARTITION_AUDIT_SCHEMA,
        "workflow_version": WORKFLOW_VERSION,
        "source_inventory_sha256": provenance["inventory_sha256"],
        "split_seed": int(seed),
        "calibration_fraction": float(calibration_fraction),
        "minimum_test_groups": int(minimum_test_groups),
        "media_rows": int(len(media)),
        "cross_partition_duplicate_hashes": duplicate_hashes,
        "capacity_passed": capacity_passed,
        "passed": passed,
        "test_metadata_only": True,
        "test_labels_or_pixels_read": False,
        "artifacts": {
            "media_partition_hash_audit": {
                "path": media_path.name,
                "bytes": media_path.stat().st_size,
                "sha256": hash_file(media_path),
            },
            "test_group_capacity_audit": {
                "path": capacity_path.name,
                "bytes": capacity_path.stat().st_size,
                "sha256": hash_file(capacity_path),
            },
        },
    }
    _atomic_json(receipt_path, receipt)
    if duplicate_hashes:
        raise StudyGateError(
            f"{duplicate_hashes} exact media SHA-256 value(s) cross VisDrone study partitions; "
            f"inspect {media_path}"
        )
    if not capacity_passed:
        observed = int(
            capacity.loc[
                capacity["scope"].astype(str).eq("all_test_components"),
                "available_natural_groups",
            ].iloc[0]
        )
        raise StudyGateError(
            f"Outcome-blind test inventory has {observed} natural groups, fewer than the "
            f"registered minimum {int(minimum_test_groups)}; inspect {capacity_path}"
        )
    return {
        "receipt_path": receipt_path,
        "media_audit_path": media_path,
        "capacity_audit_path": capacity_path,
        "receipt_sha256": hash_file(receipt_path),
        "inventory_sha256": provenance["inventory_sha256"],
        "media_rows": int(len(media)),
        "test_natural_groups": int(
            capacity.loc[
                capacity["scope"].astype(str).eq("all_test_components"),
                "available_natural_groups",
            ].iloc[0]
        ),
        "passed": True,
    }


def verify_visdrone_media_partition_audit(
    receipt_path: str | Path,
    *,
    expected_inventory_sha256: str | None = None,
    test_permit: UntouchedTestPermit | None = None,
    require_test_permit: bool = False,
) -> Mapping[str, Any]:
    """Verify the frozen audit without reopening dataset media or labels."""

    if require_test_permit:
        _verify_test_permit(test_permit)
    path = Path(receipt_path).expanduser().resolve()
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StudyGateError(f"VisDrone media-partition audit receipt is invalid: {path}") from exc
    if not isinstance(receipt, Mapping) or receipt.get("schema_version") != MEDIA_PARTITION_AUDIT_SCHEMA:
        raise StudyGateError("VisDrone media-partition audit has an incompatible schema")
    inventory_sha = str(receipt.get("source_inventory_sha256", "")).lower()
    if _HEX64.fullmatch(inventory_sha) is None:
        raise StudyGateError("VisDrone media-partition audit has an invalid inventory binding")
    if expected_inventory_sha256 is not None and inventory_sha != str(expected_inventory_sha256).lower():
        raise StudyGateError("VisDrone media-partition audit is bound to another source inventory")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise StudyGateError("VisDrone media-partition audit has no artifact envelope")
    expected_names = {
        "media_partition_hash_audit": "visdrone_media_partition_hash_audit.csv.gz",
        "test_group_capacity_audit": "visdrone_test_group_capacity_audit.csv",
    }
    resolved: dict[str, Path] = {}
    for key, filename in expected_names.items():
        item = artifacts.get(key)
        if not isinstance(item, Mapping) or item.get("path") != filename:
            raise StudyGateError(f"VisDrone media-partition audit has invalid artifact {key}")
        candidate = path.parent / filename
        size = _receipt_integer(item.get("bytes"), field=f"{key}.bytes", minimum=1)
        digest = str(item.get("sha256", "")).lower()
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or candidate.stat().st_size != size
            or _HEX64.fullmatch(digest) is None
            or hash_file(candidate) != digest
        ):
            raise StudyGateError(f"VisDrone media-partition audit artifact changed: {candidate}")
        resolved[key] = candidate
    media = pd.read_csv(resolved["media_partition_hash_audit"], compression="gzip")
    capacity = pd.read_csv(resolved["test_group_capacity_audit"])
    media_required = {
        "component", "publisher_partition", "study_partition", "split_group",
        "relative_media_path", "media_sha256", "bytes",
        "duplicate_study_partitions", "cross_partition_duplicate",
    }
    capacity_required = {
        "scope", "available_natural_groups", "required_natural_groups", "passed",
        "evidence_source",
    }
    if not media_required.issubset(media.columns) or not capacity_required.issubset(capacity.columns):
        raise StudyGateError("VisDrone media-partition audit tables have incomplete schemas")
    if len(media) != int(receipt.get("media_rows", -1)):
        raise StudyGateError("VisDrone media-partition audit row count changed")
    if media["relative_media_path"].astype(str).duplicated().any():
        raise StudyGateError("VisDrone media-partition audit contains duplicate paths")
    if not media["media_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise StudyGateError("VisDrone media-partition audit contains invalid hashes")
    crossing = media.groupby("media_sha256", sort=True)["study_partition"].nunique().gt(1)
    duplicate_hashes = int(crossing.sum())
    recorded_cross = media["cross_partition_duplicate"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    expected_cross = media["media_sha256"].astype(str).map(crossing.to_dict()).astype(bool)
    if recorded_cross.isna().any() or not recorded_cross.astype(bool).equals(expected_cross):
        raise StudyGateError("Frozen VisDrone media duplicate flags are inconsistent")
    if duplicate_hashes or int(receipt.get("cross_partition_duplicate_hashes", -1)) != 0:
        raise StudyGateError("Frozen VisDrone media audit contains cross-partition duplicates")
    minimum = int(receipt.get("minimum_test_groups", 0))
    if minimum <= 0:
        raise StudyGateError("Frozen VisDrone media audit has no valid test-group minimum")
    test = media.loc[media["study_partition"].astype(str).eq("test")]
    observed_groups = int(test[["component", "split_group"]].drop_duplicates().shape[0])
    expected_capacity_scopes = set(UNTOUCHED_TEST_COMPONENTS) | {"all_test_components"}
    if (
        set(capacity["scope"].astype(str)) != expected_capacity_scopes
        or capacity["scope"].astype(str).duplicated().any()
    ):
        raise StudyGateError("Frozen VisDrone capacity audit has invalid scopes")
    capacity_flags = capacity["passed"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    expected_capacity_flags = (
        pd.to_numeric(capacity["available_natural_groups"], errors="coerce")
        >= pd.to_numeric(capacity["required_natural_groups"], errors="coerce")
    )
    if capacity_flags.isna().any() or not capacity_flags.astype(bool).equals(expected_capacity_flags):
        raise StudyGateError("Frozen VisDrone capacity pass flags are inconsistent")
    recorded_total = capacity.loc[capacity["scope"].astype(str).eq("all_test_components")]
    if (
        len(recorded_total) != 1
        or observed_groups != int(recorded_total.iloc[0]["available_natural_groups"])
        or observed_groups < minimum
        or not bool(receipt.get("capacity_passed"))
        or not bool(receipt.get("passed"))
    ):
        raise StudyGateError("Frozen VisDrone test-group capacity audit failed verification")
    for component in UNTOUCHED_TEST_COMPONENTS:
        observed = int(
            test.loc[test["component"].astype(str).eq(component), "split_group"].nunique()
        )
        recorded = capacity.loc[capacity["scope"].astype(str).eq(component)]
        if len(recorded) != 1 or observed != int(recorded.iloc[0]["available_natural_groups"]) or observed < 1:
            raise StudyGateError(f"Frozen VisDrone capacity audit is invalid for {component}")
    return {
        "receipt_path": path,
        "receipt_sha256": hash_file(path),
        "inventory_sha256": inventory_sha,
        "media_audit_path": resolved["media_partition_hash_audit"],
        "capacity_audit_path": resolved["test_group_capacity_audit"],
        "media_rows": int(len(media)),
        "test_natural_groups": observed_groups,
        "passed": True,
    }


def assign_publisher_splits(
    source_manifest: pd.DataFrame,
    *,
    calibration_fraction: float = 0.20,
    seed: int = 20260823,
    test_permit: UntouchedTestPermit | None = None,
) -> pd.DataFrame:
    """Map publisher partitions to group-safe study splits deterministically.

    Publisher train components alone are divided into ``train`` and
    ``calibration`` using a component-scoped natural-group hash. Publisher val
    maps to ``validation`` and publisher test-dev maps to ``test``.
    """

    required = {
        "sample_id", "component", "publisher_partition", "study_partition",
        "split_group", "sequence_id",
    }
    missing = sorted(required.difference(source_manifest.columns))
    if missing:
        raise ValueError(f"VisDrone source manifest is missing columns: {missing}")
    if not 0.0 < float(calibration_fraction) < 1.0:
        raise ValueError("calibration_fraction must lie strictly within (0, 1)")
    frame = source_manifest.copy()
    if frame.empty or frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("VisDrone source manifest must be nonempty with unique sample IDs")
    unknown = sorted(set(frame["component"].astype(str)) - set(VISDRONE_COMPONENTS))
    if unknown:
        raise ValueError(f"Unknown VisDrone components: {unknown}")
    test_mask = frame["component"].astype(str).isin(UNTOUCHED_TEST_COMPONENTS)
    if test_mask.any():
        _verify_test_permit(test_permit, frame.loc[test_mask])

    component_partition = {
        key: spec.publisher_partition.value for key, spec in VISDRONE_COMPONENTS.items()
    }
    inconsistent = frame.apply(
        lambda row: str(row.publisher_partition) != component_partition[str(row.component)],
        axis=1,
    )
    if inconsistent.any():
        raise ValueError("Publisher partition metadata disagrees with the component registry")

    split_by_unit: dict[tuple[str, str], str] = {}
    for component, group in frame.groupby("component", sort=True):
        publisher_partition = component_partition[str(component)]
        for split_group in sorted(set(group["split_group"].astype(str))):
            if not split_group.strip():
                raise ValueError("Every VisDrone row requires a nonempty natural split_group")
            if publisher_partition == "train":
                score = _stable_unit_interval(WORKFLOW_VERSION, int(seed), component, split_group)
                split = "calibration" if score < float(calibration_fraction) else "train"
            elif publisher_partition == "val":
                split = "validation"
            elif publisher_partition == "test-dev":
                split = "test"
            else:  # pragma: no cover - registry enum prevents this branch
                raise ValueError(f"Unsupported publisher partition: {publisher_partition}")
            split_by_unit[(str(component), split_group)] = split

    frame["split"] = [
        split_by_unit[(str(row.component), str(row.split_group))]
        for row in frame.itertuples(index=False)
    ]
    frame["split_unit"] = [
        f"visdrone:{row.component}:{row.split_group}"
        for row in frame.itertuples(index=False)
    ]
    crossing = frame.groupby("split_unit", sort=True)["split"].nunique()
    if (crossing != 1).any():
        raise AssertionError("A VisDrone natural group crossed study splits")
    frame["split_assignment_version"] = WORKFLOW_VERSION
    frame["split_assignment_seed"] = int(seed)
    frame["calibration_fraction"] = float(calibration_fraction)
    return frame.sort_values(
        ["split", "component", "sequence_id", "frame_index", "sample_id"],
        kind="stable",
        ignore_index=True,
    )


def audit_publisher_splits(frame: pd.DataFrame) -> pd.DataFrame:
    """Return executable publisher/split invariants as a compact gate table."""

    rows: list[dict[str, Any]] = []

    def gate(name: str, passed: bool, observed: Any, required: Any) -> None:
        rows.append({
            "gate": name,
            "passed": bool(passed),
            "observed": canonical_json(observed),
            "required": canonical_json(required),
        })

    required = {"component", "publisher_partition", "split", "split_unit", "sample_id"}
    missing = sorted(required.difference(frame.columns))
    gate("split_schema", not missing, missing, [])
    if missing or frame.empty:
        gate("split_nonempty", False, int(len(frame)), "> 0")
        return pd.DataFrame(rows)
    gate("split_nonempty", True, int(len(frame)), "> 0")
    expected = {"train": {"train", "calibration"}, "val": {"validation"}, "test-dev": {"test"}}
    bad_mapping = []
    for partition, group in frame.groupby("publisher_partition", sort=True):
        observed = set(group["split"].astype(str))
        if observed - expected.get(str(partition), set()):
            bad_mapping.append({"publisher_partition": str(partition), "splits": sorted(observed)})
    gate("publisher_partition_mapping", not bad_mapping, bad_mapping, expected)
    crossing = frame.groupby("split_unit", sort=True)["split"].nunique()
    bad_groups = crossing[crossing != 1].index.astype(str).tolist()
    gate("natural_group_split_isolation", not bad_groups, bad_groups, [])
    duplicate_ids = frame.loc[
        frame["sample_id"].astype(str).duplicated(keep=False), "sample_id"
    ].astype(str).unique().tolist()
    gate("source_sample_ids_unique", not duplicate_ids, duplicate_ids[:20], [])
    return pd.DataFrame(rows)


def registered_mission_policies(
    benchmark: Mapping[str, Any],
) -> Mapping[str, RegisteredBenchmarkPolicy]:
    """Build the three exact source-rule policies from frozen configuration."""

    roi_values = benchmark.get("registered_roi_xyxy")
    if not isinstance(roi_values, Sequence) or isinstance(roi_values, (str, bytes)) or len(roi_values) != 4:
        raise ValueError("benchmark.registered_roi_xyxy must have four coordinates")
    roi = NormalizedROI("registered_image_plane_roi", *(float(value) for value in roi_values))
    profiles = benchmark.get("mission_profiles")
    if not isinstance(profiles, Mapping) or set(profiles) != set(REGISTERED_MISSION_PROFILES):
        raise ValueError("benchmark must define exactly the three registered mission profiles")
    visibility = benchmark.get("visibility_rule", {})
    if not isinstance(visibility, Mapping):
        raise ValueError("benchmark.visibility_rule must be a mapping")
    rule_version = str(benchmark.get("action_rule_version", "")).strip()
    if not rule_version:
        raise ValueError("benchmark.action_rule_version must be frozen")
    policies: dict[str, RegisteredBenchmarkPolicy] = {}
    for name in sorted(profiles):
        item = profiles[name]
        if not isinstance(item, Mapping):
            raise ValueError(f"Mission profile {name!r} must be a mapping")
        policies[name] = RegisteredBenchmarkPolicy.from_category_names(
            rule_id=f"{rule_version}:{name}",
            roi=roi,
            mission_profile=name,
            priority_categories=item.get("priority_categories", ()),
            occupancy_rule=str(benchmark.get("occupancy_rule", "center")),
            min_roi_visible_fraction=float(benchmark.get("min_roi_visible_fraction", 0.25)),
            min_image_plane_iou=float(benchmark.get("min_image_plane_iou", 0.05)),
            min_score=float(benchmark.get("min_source_score", 0.5)),
            max_truncation=int(visibility.get("visible_max_truncation", 1)),
            max_occlusion=int(visibility.get("visible_max_occlusion", 1)),
        )
        deadline_ms = item.get("deadline_ms")
        if isinstance(deadline_ms, bool) or not isinstance(deadline_ms, (int, float)) or float(deadline_ms) <= 0:
            raise ValueError(f"Mission profile {name!r} requires a positive deadline_ms")
    return policies


def _image_dimensions(
    source_manifest: pd.DataFrame,
    *,
    progress: _PrepareProgress | None = None,
) -> dict[str, tuple[int, int]]:
    sizes: dict[str, tuple[int, int]] = {}
    rows = source_manifest[["sample_id", "media_path"]]
    total = int(len(rows))
    for completed, row in enumerate(rows.itertuples(index=False), start=1):
        path = Path(str(row.media_path)).expanduser().resolve()
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
        except Exception as exc:
            raise StudyGateError(f"Cannot verify VisDrone media dimensions: {path}") from exc
        if int(width) <= 0 or int(height) <= 0:
            raise StudyGateError(f"VisDrone media has invalid dimensions: {path}")
        sizes[str(row.sample_id)] = (int(width), int(height))
        _notify_prepare_progress(progress, "media_dimensions", completed, total)
    return sizes


def _det_objects_with_annotation_quality(
    text: str,
    *,
    component: str,
    sample_id: str,
    source_uid: str,
    annotation_relative_path: str,
    frame_index: int,
) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
    """Parse DET rows while auditing only exact-zero publisher geometry.

    ``parse_det_row`` remains the authority for every schema and domain check.
    For an exact-zero row only, a temporary positive-extent probe is passed to
    that strict parser so all *other* fields are validated. The probe is never
    returned or written as truth. Negative or non-finite extents follow the
    unmodified strict path and therefore remain fatal.
    """

    objects: list[Any] = []
    exclusions: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            fields = next(csv.reader([raw_line], skipinitialspace=True))
        except csv.Error:
            # Preserve the strict parser's public failure behavior and message.
            parse_det_row(
                raw_line, component=component, sample_id=sample_id,
                frame_index=frame_index, source_line=line_number,
            )
            raise AssertionError("unreachable")
        fields = [value.strip() for value in fields]
        if fields:
            fields[0] = fields[0].lstrip("\ufeff")
        while fields and fields[-1] == "":
            fields.pop()
        if len(fields) != 8 or any(value == "" for value in fields):
            parse_det_row(
                raw_line, component=component, sample_id=sample_id,
                frame_index=frame_index, source_line=line_number,
            )
            raise AssertionError("unreachable")
        try:
            width = float(fields[2])
            height = float(fields[3])
        except ValueError:
            parse_det_row(
                raw_line, component=component, sample_id=sample_id,
                frame_index=frame_index, source_line=line_number,
            )
            raise AssertionError("unreachable")
        if (
            not math.isfinite(width)
            or not math.isfinite(height)
            or width < 0
            or height < 0
        ):
            # NaN/Inf/negative geometry is not a registered exclusion.
            parse_det_row(
                raw_line, component=component, sample_id=sample_id,
                frame_index=frame_index, source_line=line_number,
            )
            raise AssertionError("unreachable")
        if width != 0.0 and height != 0.0:
            objects.append(
                parse_det_row(
                    raw_line, component=component, sample_id=sample_id,
                    frame_index=frame_index, source_line=line_number,
                )
            )
            continue

        # Validate score/category/visibility and every remaining field without
        # accepting fabricated geometry into a scientific object.
        probe = list(fields)
        if width == 0.0:
            probe[2] = "1"
        if height == 0.0:
            probe[3] = "1"
        validated = parse_det_row(
            probe, component=component, sample_id=sample_id,
            frame_index=frame_index, source_line=line_number,
        )
        publisher_ignored = bool(
            validated.score == 0.0 or validated.category_id in {0, 11}
        )
        exclusions.append({
            "policy_version": ANNOTATION_QUALITY_POLICY_VERSION,
            "component": str(component),
            "sample_id": str(sample_id),
            "source_uid": str(source_uid),
            "annotation_relative_path": str(annotation_relative_path),
            "frame_index": int(frame_index),
            "source_line": int(line_number),
            "raw_row_sha256": hashlib.sha256(raw_line.encode("utf-8")).hexdigest(),
            "bbox_width": float(width),
            "bbox_height": float(height),
            "score": float(validated.score),
            "category_id": int(validated.category_id),
            "publisher_ignored": publisher_ignored,
            "action": _EXCLUDE_ROW if publisher_ignored else _EXCLUDE_SAMPLE,
            "reason": _IGNORED_ZERO_REASON if publisher_ignored else _EVALUATED_ZERO_REASON,
        })
    return tuple(objects), exclusions


def _objects_and_annotation_quality(
    source_manifest: pd.DataFrame,
    *,
    source_root: Path,
    test_permit: UntouchedTestPermit | None,
    progress: _PrepareProgress | None = None,
) -> tuple[pd.DataFrame, Mapping[str, Sequence[Any]], pd.DataFrame]:
    """Apply the registered DET quality rule before scientific splitting.

    Ignored exact-zero rows are removed individually. If an evaluated DET row
    has exact-zero extent, its whole source sample is removed before split,
    truth, sampling, or detector preparation so it cannot be relabeled as a
    negative example. MOT annotations retain the strict publisher parser.
    """

    test_rows = source_manifest[
        source_manifest["component"].astype(str).isin(UNTOUCHED_TEST_COMPONENTS)
    ]
    if not test_rows.empty:
        _verify_test_permit(test_permit, test_rows)
    objects: dict[str, Sequence[Any]] = {}
    exclusions: list[dict[str, Any]] = []
    mot_cache: dict[tuple[str, str], Mapping[int, Sequence[Any]]] = {}
    ordered = source_manifest.sort_values("sample_id", kind="stable").to_dict("records")
    total = len(ordered)
    for completed, row in enumerate(ordered, start=1):
        component = str(row["component"])
        sample_id = str(row["sample_id"])
        annotation = Path(str(row["annotation_path"])).expanduser().resolve()
        try:
            annotation_relative = annotation.relative_to(source_root).as_posix()
        except ValueError as exc:
            raise StudyGateError(
                f"VisDrone annotation escapes the verified dataset root: {annotation}"
            ) from exc
        if VISDRONE_COMPONENTS[component].task == "MOT":
            key = (component, str(annotation))
            if key in mot_cache:
                objects[sample_id] = mot_cache[key].get(int(row["frame_index"]), ())
                _notify_prepare_progress(
                    progress, "annotation_quality_samples", completed, total,
                )
                continue
        try:
            text = annotation.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as exc:
            raise StudyGateError(f"Cannot read VisDrone source annotation: {annotation}") from exc
        if VISDRONE_COMPONENTS[component].task == "DET":
            parsed, excluded = _det_objects_with_annotation_quality(
                text,
                component=component,
                sample_id=sample_id,
                source_uid=str(row["source_uid"]),
                annotation_relative_path=annotation_relative,
                frame_index=int(row["frame_index"]),
            )
            objects[sample_id] = parsed
            exclusions.extend(excluded)
            _notify_prepare_progress(
                progress, "annotation_quality_samples", completed, total,
            )
            continue
        key = (component, str(annotation))
        sequence_name = str(row["sequence_id"]).split(":", 1)[-1]
        by_frame: dict[int, list[Any]] = defaultdict(list)
        for obj in parse_mot_annotations(
            text, component=component, sequence_id=sequence_name,
        ):
            by_frame[int(obj.frame_index)].append(obj)
        mot_cache[key] = {
            frame: tuple(sorted(values, key=lambda item: item.object_uid))
            for frame, values in by_frame.items()
        }
        objects[sample_id] = mot_cache[key].get(int(row["frame_index"]), ())
        _notify_prepare_progress(
            progress, "annotation_quality_samples", completed, total,
        )

    audit = pd.DataFrame.from_records(exclusions, columns=ANNOTATION_EXCLUSION_COLUMNS)
    if not audit.empty:
        audit = audit.sort_values(
            ["component", "sample_id", "source_line"],
            kind="stable", ignore_index=True,
        )
    excluded_samples = set(
        audit.loc[audit["action"].eq(_EXCLUDE_SAMPLE), "sample_id"].astype(str)
    )
    filtered = source_manifest.loc[
        ~source_manifest["sample_id"].astype(str).isin(excluded_samples)
    ].copy()
    if filtered.empty:
        raise StudyGateError(
            "Annotation quality policy excluded every VisDrone source sample"
        )
    objects = {
        sample_id: values
        for sample_id, values in objects.items()
        if sample_id not in excluded_samples
    }
    expected_ids = set(filtered["sample_id"].astype(str))
    if set(objects) != expected_ids:
        raise StudyGateError("Annotation quality scan did not cover every retained sample")
    return filtered.reset_index(drop=True), objects, audit


def _objects_by_sample(
    source_manifest: pd.DataFrame,
    *,
    test_permit: UntouchedTestPermit | None,
) -> Mapping[str, Sequence[Any]]:
    """Parse each DET file once and each MOT sequence file once."""

    test_rows = source_manifest[
        source_manifest["component"].astype(str).isin(UNTOUCHED_TEST_COMPONENTS)
    ]
    if not test_rows.empty:
        _verify_test_permit(test_permit, test_rows)
    objects: dict[str, Sequence[Any]] = {}
    mot_cache: dict[tuple[str, str], Mapping[int, Sequence[Any]]] = {}
    for row in source_manifest.sort_values("sample_id", kind="stable").to_dict("records"):
        component = str(row["component"])
        annotation = Path(str(row["annotation_path"])).expanduser().resolve()
        if VISDRONE_COMPONENTS[component].task == "MOT":
            key = (component, str(annotation))
            if key in mot_cache:
                objects[str(row["sample_id"])] = mot_cache[key].get(int(row["frame_index"]), ())
                continue
        try:
            text = annotation.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as exc:
            raise StudyGateError(f"Cannot read VisDrone source annotation: {annotation}") from exc
        if VISDRONE_COMPONENTS[component].task == "DET":
            objects[str(row["sample_id"])] = parse_det_annotations(
                text,
                component=component,
                sample_id=str(row["sample_id"]),
                frame_index=int(row["frame_index"]),
            )
            continue
        key = (component, str(annotation))
        sequence_name = str(row["sequence_id"]).split(":", 1)[-1]
        by_frame: dict[int, list[Any]] = defaultdict(list)
        for obj in parse_mot_annotations(
            text, component=component, sequence_id=sequence_name,
        ):
            by_frame[int(obj.frame_index)].append(obj)
        mot_cache[key] = {
            frame_index: tuple(sorted(values, key=lambda item: item.object_uid))
            for frame_index, values in by_frame.items()
        }
        objects[str(row["sample_id"])] = mot_cache[key].get(int(row["frame_index"]), ())
    return objects


def _primary_event(truth: SourceDerivedTruth) -> str:
    events = set(truth.events)
    if truth.gold_action == ACTION_RESPOND:
        return EVENT_PRIORITY_ROI_OCCUPANCY
    if EVENT_PRIORITY_EVIDENCE_LIMITED in events:
        return EVENT_PRIORITY_EVIDENCE_LIMITED
    if EVENT_PRIORITY_OUTSIDE_ROI in events:
        return EVENT_PRIORITY_OUTSIDE_ROI
    if EVENT_IMAGE_PLANE_OVERLAP in events:
        return EVENT_IMAGE_PLANE_OVERLAP
    return EVENT_NO_PRIORITY_VISIBLE


def _canonical_semantics(truth: SourceDerivedTruth) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    facts = {item.object_uid: item for item in truth.objects}
    objects = [
        {
            "label": item.category,
            "id": item.object_uid,
            "bbox_normalized_xyxy": [float(value) for value in item.bbox_normalized_xyxy],
            "confidence": 1.0,
            "publisher_truncation": int(item.truncation),
            "publisher_occlusion": int(item.occlusion),
            "registered_priority": bool(item.registered_priority),
        }
        for item in truth.objects
    ]
    relation_keys: set[tuple[str, str, str]] = set()
    relations: list[dict[str, Any]] = []
    for relation in truth.relations:
        if relation.kind != RELATION_IMAGE_PLANE_OVERLAP or relation.object_uid is None:
            continue
        left = facts.get(relation.subject_uid)
        right = facts.get(relation.object_uid)
        if left is None or right is None:
            continue
        key = (left.category, RELATION_IMAGE_PLANE_OVERLAP, right.category)
        if key in relation_keys:
            continue
        relation_keys.add(key)
        relations.append({
            "subject": left.category,
            "predicate": RELATION_IMAGE_PLANE_OVERLAP,
            "object": right.category,
            "image_plane_iou": float(relation.value or 0.0),
        })
    return objects, relations


def _gold_evidence_roi(
    truth: SourceDerivedTruth,
    policy: RegisteredBenchmarkPolicy,
) -> list[float]:
    """Select one publisher box for the registered evidence-localization endpoint."""

    priority = [item for item in truth.objects if item.registered_priority]
    if truth.gold_action == ACTION_RESPOND:
        roi = policy.roi
        inside = []
        for item in priority:
            x1, y1, x2, y2 = item.bbox_normalized_xyxy
            center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if roi.x_min <= center_x <= roi.x_max and roi.y_min <= center_y <= roi.y_max:
                inside.append(item)
        if inside:
            return [float(value) for value in inside[0].bbox_normalized_xyxy]
    if priority:
        return [float(value) for value in priority[0].bbox_normalized_xyxy]
    if truth.objects:
        return [float(value) for value in truth.objects[0].bbox_normalized_xyxy]
    return []


def build_analysis_manifest(
    source_manifest: pd.DataFrame,
    *,
    image_sizes: Mapping[str, tuple[int, int]],
    benchmark: Mapping[str, Any],
    test_permit: UntouchedTestPermit | None = None,
    capture_period_us: int = 200_000,
    prepared_objects_by_sample: Mapping[str, Sequence[Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand source records across three profiles into experiment-ready rows.

    Returns ``(analysis_manifest, source_truth_table)``.  No telemetry is
    inferred from imagery: constant coordinates, altitude, and yaw are marked
    as registered scenario inputs rather than measured flight telemetry.
    """

    required = {
        "sample_id", "dataset", "mission", "component", "task",
        "publisher_partition", "study_partition", "source_uid", "media_path",
        "annotation_path", "sequence_id", "frame_index", "split_group",
        "split", "split_unit", "has_ground_truth",
    }
    missing = sorted(required.difference(source_manifest.columns))
    if missing:
        raise ValueError(f"Split VisDrone source manifest is missing columns: {missing}")
    if int(capture_period_us) <= 0:
        raise ValueError("capture_period_us must be positive")
    source = source_manifest.sort_values(
        ["component", "sequence_id", "frame_index", "source_uid"], kind="stable",
    ).reset_index(drop=True)
    if set(source["mission"].astype(str)) != {MISSION_VISDRONE_URBAN}:
        raise ValueError("VisDrone analysis accepts only mission=visdrone_urban")
    test_rows = source[source["component"].astype(str).isin(UNTOUCHED_TEST_COMPONENTS)]
    if not test_rows.empty:
        _verify_test_permit(test_permit, test_rows)
    unknown_sizes = sorted(set(source["sample_id"].astype(str)) - set(map(str, image_sizes)))
    if unknown_sizes:
        raise ValueError(f"Missing registered image dimensions; first: {unknown_sizes[:5]}")
    policies = registered_mission_policies(benchmark)
    deadlines_ms = {
        name: float(benchmark["mission_profiles"][name]["deadline_ms"])
        for name in policies
    }
    if prepared_objects_by_sample is None:
        objects_by_sample = _objects_by_sample(source, test_permit=test_permit)
    else:
        objects_by_sample = {
            str(sample_id): tuple(values)
            for sample_id, values in prepared_objects_by_sample.items()
        }
        expected_object_ids = set(source["sample_id"].astype(str))
        if set(objects_by_sample) != expected_object_ids:
            raise ValueError(
                "Prepared annotation objects must exactly cover the source manifest"
            )

    analysis_records: list[dict[str, Any]] = []
    truth_records: list[dict[str, Any]] = []
    ordered_profiles = tuple(sorted(policies))
    for profile_index, profile in enumerate(ordered_profiles):
        policy = policies[profile]
        for source_index, row in enumerate(source.to_dict("records")):
            base_id = str(row["sample_id"])
            width, height = image_sizes[base_id]
            truth = derive_source_truth(
                sample_id=base_id,
                objects=objects_by_sample[base_id],
                image_width=int(width),
                image_height=int(height),
                policy=policy,
            )
            sample_id = f"{base_id}:profile:{profile}"
            event = _primary_event(truth)
            semantic_objects, semantic_relations = _canonical_semantics(truth)
            full_truth = truth.as_dict()
            truth_hash = hash_object(full_truth)
            capture_index = profile_index * len(source) + source_index
            capture_time_us = int(capture_index * int(capture_period_us))
            deadline_us = capture_time_us + int(round(deadlines_ms[profile] * 1000.0))
            evidence_limited = EVENT_PRIORITY_EVIDENCE_LIMITED in set(truth.events)
            critical = truth.gold_action == ACTION_RESPOND
            risk = 1.0 if critical else 0.5 if truth.gold_action == ACTION_REVIEW else 0.0
            uncertainty = 1.0 if evidence_limited else 0.0
            telemetry = {
                "gps_lat": 0.0,
                "gps_lon": 0.0,
                "altitude_m": 0.0,
                "yaw_deg": 0.0,
                "mission_profile": profile,
                "registered_roi_xyxy": policy.roi.as_list(),
            }
            truth_records.append({
                "sample_id": sample_id,
                "source_sample_id": base_id,
                "mission_profile": profile,
                "gold_action": truth.gold_action,
                "primary_event": event,
                "events_json": canonical_json(list(truth.events)),
                "objects_json": canonical_json([item.as_dict() for item in truth.objects]),
                "relations_json": canonical_json([item.as_dict() for item in truth.relations]),
                "priority_endpoint_positive": bool(truth.priority_endpoint_positive),
                "rule_id": truth.rule_id,
                "truth_scope": truth.truth_scope,
                "truth_sha256": truth_hash,
                "test_ledger_sha256": str(row.get("test_ledger_sha256", "")),
            })
            analysis_records.append({
                "sample_id": sample_id,
                "source_sample_id": base_id,
                "visual_group_id": str(row["split_unit"]),
                "split_unit": str(row["split_unit"]),
                "split_group": str(row["split_group"]),
                "sequence_id": str(row["sequence_id"]),
                "event_group": str(row["split_unit"]),
                "dataset": "visdrone",
                "component": str(row["component"]),
                "task": str(row["task"]),
                "publisher_partition": str(row["publisher_partition"]),
                "study_partition": str(row["study_partition"]),
                "source_uid": str(row["source_uid"]),
                "mission": MISSION_VISDRONE_URBAN,
                "mission_profile": profile,
                "split": str(row["split"]),
                "cohort": "natural_test" if str(row["split"]) == "test" else "development",
                "media_path": str(row["media_path"]),
                "image_path": str(row["media_path"]),
                "annotation_path": str(row["annotation_path"]),
                "media_type": "image",
                "frame_index": int(row["frame_index"]),
                "width": int(width),
                "height": int(height),
                "has_ground_truth": True,
                "is_synthetic": False,
                "capture_time_us": capture_time_us,
                "trace_time_us": capture_time_us,
                "deadline_us": deadline_us,
                "critical": bool(critical),
                "critical_event": bool(critical),
                "gold_action": truth.gold_action,
                "event": event,
                "objects": semantic_objects,
                "relations": semantic_relations,
                "sufficient_evidence": True,
                "evidence_roi": _gold_evidence_roi(truth, policy),
                "registered_roi_xyxy": policy.roi.as_list(),
                "gold_evidence_definition": "publisher_object_box_normalized_xyxy",
                "priority_endpoint_positive": bool(truth.priority_endpoint_positive),
                "risk": risk,
                "uncertainty": uncertainty,
                "battery_fraction": 1.0,
                "thermal_c": 0.0,
                "ood": False,
                "telemetry_json": canonical_json(telemetry),
                "telemetry_provenance": "registered_scenario_not_measured_flight_telemetry",
                "truth_scope": truth.truth_scope,
                "truth_sha256": truth_hash,
                "source_rule_id": truth.rule_id,
                "registered_roi_relations_json": canonical_json([
                    relation.as_dict() for relation in truth.relations
                    if relation.kind != RELATION_IMAGE_PLANE_OVERLAP
                ]),
                "test_ledger_sha256": str(row.get("test_ledger_sha256", "")),
            })

    analysis = pd.DataFrame.from_records(analysis_records)
    truth_table = pd.DataFrame.from_records(truth_records)
    if analysis["sample_id"].duplicated().any() or truth_table["sample_id"].duplicated().any():
        raise AssertionError("Profile-expanded VisDrone sample identifiers are not unique")
    if set(analysis["truth_scope"]) != {TRUTH_SCOPE}:
        raise AssertionError("VisDrone analysis escaped the registered sensor-space truth scope")

    analysis["clip_paths_json"] = "[]"
    for (_, profile), group in analysis.groupby(["sequence_id", "mission_profile"], sort=True):
        ordered = group.sort_values(["frame_index", "source_uid"], kind="stable")
        paths = ordered["media_path"].astype(str).tolist()
        indices = ordered.index.tolist()
        if len(paths) < 2:
            continue
        for position, row_index in enumerate(indices):
            # The clip is causal: it ends at the current frame and never
            # exposes a future frame to an earlier decision.
            start = max(0, position - 2)
            clip = paths[start : position + 1]
            if len(clip) >= 2:
                analysis.at[row_index, "clip_paths_json"] = canonical_json(clip)
    analysis = analysis.sort_values(
        ["split", "mission_profile", "component", "sequence_id", "frame_index", "sample_id"],
        kind="stable",
        ignore_index=True,
    )
    truth_table = truth_table.sort_values("sample_id", kind="stable", ignore_index=True)
    return analysis, truth_table


def _serialized_analysis_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    serialized = frame.copy()
    for column in ("objects", "relations", "evidence_roi"):
        if column in serialized:
            serialized[column] = serialized[column].map(
                lambda value: value if isinstance(value, str) else canonical_json(value)
            )
    return serialized


def read_analysis_manifest(path: str | Path) -> pd.DataFrame:
    """Read a prepared analysis manifest and restore its structured columns."""

    manifest_path = Path(path).expanduser().resolve()
    frame = pd.read_csv(
        manifest_path,
        compression="gzip" if str(manifest_path).endswith(".gz") else None,
        keep_default_na=False,
    )
    for column in ("objects", "relations", "evidence_roi"):
        if column not in frame:
            continue
        try:
            frame[column] = frame[column].map(json.loads)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise StudyGateError(
                f"Prepared analysis manifest has invalid JSON column {column!r}"
            ) from exc
    for column in (
        "has_ground_truth", "is_synthetic", "critical", "critical_event",
        "sufficient_evidence", "priority_endpoint_positive", "ood",
    ):
        if column in frame:
            normalized = frame[column].astype(str).str.strip().str.lower()
            if not normalized.isin({"true", "false"}).all():
                raise StudyGateError(f"Prepared analysis manifest has invalid Boolean {column!r}")
            frame[column] = normalized.eq("true")
    return frame


def select_registered_analysis_sample(
    manifest: pd.DataFrame,
    *,
    seed: int,
    max_groups_per_component: int | None,
    max_frames_per_group: int | None,
    splits: Sequence[str] | None = None,
    test_permit: UntouchedTestPermit | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select an outcome-blind, profile-complete registered analysis sample.

    Natural groups are ranked only by a seeded hash of split, component, and
    split-unit identity. If a group frame cap is supplied, temporally ordered
    source frames are selected at deterministic, evenly spaced positions.
    Gold actions, events, objects, risk, uncertainty, and all model/network
    outcomes are never read by the selection algorithm.
    """

    sampling_columns = (
        "sample_id", "source_sample_id", "mission_profile", "component", "split",
        "split_unit", "sequence_id", "frame_index", "source_uid",
    )
    missing = sorted(set(sampling_columns).difference(manifest.columns))
    if missing:
        raise ValueError(f"Analysis sampling manifest is missing columns: {missing}")
    if manifest.empty or manifest["sample_id"].astype(str).duplicated().any():
        raise ValueError("Analysis sampling requires nonempty, unique sample rows")
    if max_groups_per_component is not None and int(max_groups_per_component) <= 0:
        raise ValueError("max_groups_per_component must be positive or None")
    if max_frames_per_group is not None and int(max_frames_per_group) <= 0:
        raise ValueError("max_frames_per_group must be positive or None")
    requested_splits = (
        tuple(str(value) for value in splits)
        if splits is not None else
        tuple(sorted(set(manifest["split"].astype(str))))
    )
    if not requested_splits or len(requested_splits) != len(set(requested_splits)):
        raise ValueError("splits must contain one or more unique split names")
    unknown_splits = sorted(set(requested_splits) - {"train", "calibration", "validation", "test"})
    if unknown_splits:
        raise ValueError(f"Unsupported analysis split(s): {unknown_splits}")
    work = manifest.loc[manifest["split"].astype(str).isin(requested_splits)].copy()
    if work.empty:
        raise StudyGateError("Registered analysis sampling selected no requested split rows")
    test_rows = work[work["split"].astype(str).eq("test")]
    if not test_rows.empty:
        _verify_test_permit(test_permit, test_rows)

    expected_profiles = set(REGISTERED_MISSION_PROFILES)
    source_profile_counts = work.groupby("source_sample_id", sort=True)["mission_profile"].agg(
        lambda values: set(map(str, values))
    )
    incomplete_sources = source_profile_counts[source_profile_counts.map(lambda values: values != expected_profiles)]
    if not incomplete_sources.empty:
        raise StudyGateError(
            "Registered analysis sampling requires all three mission-profile pairs per source"
        )
    duplicates = work.duplicated(["source_sample_id", "mission_profile"], keep=False)
    if duplicates.any():
        raise StudyGateError("A source/profile pair occurs more than once in the analysis manifest")

    base_columns = (
        "source_sample_id", "component", "split", "split_unit", "sequence_id",
        "frame_index", "source_uid",
    )
    for column in base_columns[1:]:
        inconsistent = work.groupby("source_sample_id", sort=True)[column].nunique(dropna=False)
        if (inconsistent != 1).any():
            raise StudyGateError(
                f"Source/profile pairs disagree on outcome-blind field {column!r}"
            )
    base = work.loc[:, base_columns].drop_duplicates("source_sample_id")
    selected_source_ids: list[str] = []
    selected_group_rows: list[dict[str, Any]] = []
    for (split, component), component_rows in base.groupby(["split", "component"], sort=True):
        groups = component_rows[["split_unit"]].drop_duplicates().copy()
        groups["selection_hash"] = groups["split_unit"].map(
            lambda unit: hashlib.sha256(
                canonical_json([
                    WORKFLOW_VERSION, "analysis-sample", int(seed),
                    str(split), str(component), str(unit),
                ]).encode("utf-8")
            ).hexdigest()
        )
        groups = groups.sort_values(["selection_hash", "split_unit"], kind="stable")
        if max_groups_per_component is not None:
            groups = groups.head(int(max_groups_per_component))
        for group_rank, group_row in enumerate(groups.itertuples(index=False), start=1):
            unit_rows = component_rows[
                component_rows["split_unit"].astype(str).eq(str(group_row.split_unit))
            ].sort_values(["frame_index", "source_uid", "source_sample_id"], kind="stable")
            original_frames = len(unit_rows)
            if max_frames_per_group is not None and original_frames > int(max_frames_per_group):
                limit = int(max_frames_per_group)
                positions = (
                    np.array([(original_frames - 1) // 2], dtype=int)
                    if limit == 1 else
                    np.unique(np.rint(np.linspace(0, original_frames - 1, limit)).astype(int))
                )
                unit_rows = unit_rows.iloc[positions]
            selected_source_ids.extend(unit_rows["source_sample_id"].astype(str))
            selected_group_rows.append({
                "split": str(split),
                "component": str(component),
                "split_unit": str(group_row.split_unit),
                "group_rank": int(group_rank),
                "group_selection_hash": str(group_row.selection_hash),
                "original_source_frames": int(original_frames),
                "selected_source_frames": int(len(unit_rows)),
            })
    selected_source_ids = sorted(set(selected_source_ids))
    selected = work.loc[work["source_sample_id"].astype(str).isin(selected_source_ids)].copy()
    selected = selected.sort_values(
        ["split", "component", "split_unit", "frame_index", "source_uid", "mission_profile"],
        kind="stable",
        ignore_index=True,
    )
    selected_profiles = selected.groupby("source_sample_id", sort=True)["mission_profile"].agg(
        lambda values: set(map(str, values))
    )
    profile_complete = bool(
        len(selected_profiles)
        and selected_profiles.map(lambda values: values == expected_profiles).all()
    )
    profile_counts = selected.groupby("mission_profile", sort=True)["source_sample_id"].nunique().to_dict()
    profile_balanced = len(set(profile_counts.values())) == 1 and set(profile_counts) == expected_profiles
    selection_payload = {
        "workflow_version": WORKFLOW_VERSION,
        "seed": int(seed),
        "max_groups_per_component": max_groups_per_component,
        "max_frames_per_group": max_frames_per_group,
        "splits": list(requested_splits),
        "selected_pairs": selected[["source_sample_id", "mission_profile"]].astype(str).values.tolist(),
    }
    selection_sha256 = hash_object(selection_payload)

    audit_rows: list[dict[str, Any]] = []

    def gate(name: str, passed: bool, observed: Any, required: Any) -> None:
        audit_rows.append({
            "gate": name,
            "passed": bool(passed),
            "observed": canonical_json(observed),
            "required": canonical_json(required),
            "selection_sha256": selection_sha256,
            "sampling_seed": int(seed),
            "sampling_columns_json": canonical_json(list(sampling_columns)),
            "outcome_columns_read_json": "[]",
        })

    gate("selection_nonempty", not selected.empty, int(len(selected)), "> 0")
    gate("all_source_profiles_complete", incomplete_sources.empty, int(len(incomplete_sources)), 0)
    gate("selected_source_profiles_complete", profile_complete, profile_counts, sorted(expected_profiles))
    gate("profile_counts_balanced", profile_balanced, profile_counts, "equal counts")
    group_audit = pd.DataFrame.from_records(selected_group_rows)
    group_cap_ok = bool(
        max_groups_per_component is None
        or group_audit.groupby(["split", "component"])["split_unit"].nunique().le(
            int(max_groups_per_component)
        ).all()
    )
    frame_cap_ok = bool(
        max_frames_per_group is None
        or group_audit["selected_source_frames"].le(int(max_frames_per_group)).all()
    )
    gate("registered_group_cap", group_cap_ok, group_audit.to_dict("records"), max_groups_per_component)
    gate("deterministic_temporal_frame_cap", frame_cap_ok, group_audit.to_dict("records"), max_frames_per_group)
    gate("outcome_blind_selection", True, list(sampling_columns), "no label/model/network outcome columns")
    gate(
        "test_rows_hash_bound",
        test_rows.empty or test_rows["test_ledger_sha256"].astype(str).eq(test_permit.ledger_sha256).all(),
        int(len(test_rows)),
        "no test rows or every row bound to the active OPEN ledger",
    )
    audit = pd.DataFrame.from_records(audit_rows)
    if not audit["passed"].astype(bool).all():
        failed = audit.loc[~audit["passed"].astype(bool), "gate"].astype(str).tolist()
        raise StudyGateError(f"Registered analysis sampling audit failed: {failed}")
    return selected, audit


def registered_analysis_sampling_limits(
    benchmark: Mapping[str, Any],
    split: str,
) -> tuple[int, int]:
    """Resolve the frozen group/frame caps for one study split."""

    split_name = str(split).strip()
    if split_name not in {"train", "calibration", "validation", "test"}:
        raise ValueError(f"Unsupported analysis split: {split!r}")
    sampling = benchmark.get("analysis_sampling")
    if not isinstance(sampling, Mapping):
        raise StudyGateError("benchmark.analysis_sampling is not registered")
    if sampling.get("method") != "outcome_blind_group_hash_temporal_v1":
        raise StudyGateError("Unregistered VisDrone analysis sampling method")
    groups = sampling.get("max_groups_per_component")
    frames = sampling.get("max_frames_per_group")
    if not isinstance(groups, Mapping) or not isinstance(frames, Mapping):
        raise StudyGateError("Registered analysis sampling caps are incomplete")
    try:
        group_cap = int(groups[split_name])
        frame_cap = int(frames[split_name])
    except (KeyError, TypeError, ValueError) as exc:
        raise StudyGateError(f"Registered analysis sampling lacks valid {split_name} caps") from exc
    if group_cap <= 0 or frame_cap <= 0:
        raise StudyGateError("Registered analysis sampling caps must be positive")
    return group_cap, frame_cap


def select_registered_analysis_sample_from_config(
    manifest: pd.DataFrame,
    *,
    benchmark: Mapping[str, Any],
    split: str,
    seed: int,
    test_permit: UntouchedTestPermit | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the exact frozen sampling caps for one phase split."""

    group_cap, frame_cap = registered_analysis_sampling_limits(benchmark, split)
    return select_registered_analysis_sample(
        manifest,
        seed=seed,
        max_groups_per_component=group_cap,
        max_frames_per_group=frame_cap,
        splits=(split,),
        test_permit=test_permit,
    )


def write_registered_analysis_sample(
    sample: pd.DataFrame,
    audit: pd.DataFrame,
    output_dir: str | Path,
) -> Mapping[str, Path]:
    """Persist a selection and its immutable sampling audit with hashes."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise StudyGateError(f"Registered analysis sample output already exists: {destination}")
    selection_hashes = set(audit.get("selection_sha256", pd.Series(dtype=str)).astype(str))
    if len(selection_hashes) != 1 or not audit["passed"].astype(bool).all():
        raise StudyGateError("Registered analysis sample requires one passing immutable audit")
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        sample_path = staging / "registered_analysis_sample.csv.gz"
        audit_path = staging / "registered_analysis_sampling_audit.csv"
        _deterministic_csv(_serialized_analysis_manifest(sample), sample_path, compress=True)
        _deterministic_csv(audit, audit_path)
        manifest_path = write_artifact_hash_manifest(
            {"registered_analysis_sample": sample_path, "sampling_audit": audit_path},
            staging / "REGISTERED_ANALYSIS_SAMPLE_ARTIFACTS.json",
            metadata={"selection_sha256": next(iter(selection_hashes))},
            root=staging,
        )
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "sample": destination / sample_path.name,
        "audit": destination / audit_path.name,
        "artifact_manifest": destination / manifest_path.name,
    }


def registered_temporal_sampling_spec(
    benchmark: Mapping[str, Any],
    split: str,
) -> Mapping[str, Any]:
    """Resolve and validate the frozen MOT replay design for one study split.

    Publisher frame-rate metadata is not assumed. The registered integer
    source-frame stride and controlled replay rate are frozen independently,
    so the sample definition remains explicit and auditable.
    """

    split_name = str(split).strip()
    if split_name not in TEMPORAL_COMPONENT_BY_SPLIT:
        raise ValueError("Registered temporal sampling supports validation or test only")
    sampling = benchmark.get("temporal_sampling")
    if not isinstance(sampling, Mapping):
        raise StudyGateError("benchmark.temporal_sampling is not registered")
    if sampling.get("schema") != TEMPORAL_SAMPLING_SCHEMA:
        raise StudyGateError("Unregistered VisDrone temporal sampling schema")
    if sampling.get("method") != TEMPORAL_SAMPLING_METHOD:
        raise StudyGateError("Unregistered VisDrone temporal sampling method")
    sequence_limits = sampling.get("sequences_per_split")
    if not isinstance(sequence_limits, Mapping):
        raise StudyGateError("Registered temporal sequence limits are incomplete")
    component_by_split = sampling.get("component_by_split")
    if not isinstance(component_by_split, Mapping) or {
        str(key): str(value) for key, value in component_by_split.items()
    } != dict(TEMPORAL_COMPONENT_BY_SPLIT):
        raise StudyGateError("Registered temporal split/component mapping is invalid")
    try:
        sequences = int(sequence_limits[split_name])
        observations = int(sampling["observations_per_sequence"])
        source_stride = int(sampling["source_frame_stride"])
        target_rate_hz = int(sampling["target_rate_hz"])
        minimum_temporal_groups = int(sampling["minimum_temporal_groups"])
        minimum_group_trace_cells = int(sampling["minimum_group_trace_cells"])
    except (KeyError, TypeError, ValueError) as exc:
        raise StudyGateError("Registered temporal sampling values are incomplete") from exc
    if min(
        sequences, observations, source_stride, target_rate_hz,
        minimum_temporal_groups, minimum_group_trace_cells,
    ) <= 0:
        raise StudyGateError("Registered temporal sampling values must be positive")
    legacy_sequence_key = f"{split_name}_sequences"
    try:
        flat_sequence_limit = int(sampling[legacy_sequence_key])
    except (KeyError, TypeError, ValueError) as exc:
        raise StudyGateError("Registered temporal split sequence limit is incomplete") from exc
    if flat_sequence_limit != sequences:
        raise StudyGateError("Registered temporal sequence limits disagree")
    if 1_000_000 % target_rate_hz:
        raise StudyGateError("Registered temporal target rate needs an integral microsecond period")
    if target_rate_hz != 5:
        raise StudyGateError("Registered VisDrone temporal replay rate must be exactly 5 Hz")
    return {
        "method": TEMPORAL_SAMPLING_METHOD,
        "split": split_name,
        "component": TEMPORAL_COMPONENT_BY_SPLIT[split_name],
        "sequences_per_split": sequences,
        "observations_per_sequence": observations,
        "source_frame_stride": source_stride,
        "target_rate_hz": target_rate_hz,
        "capture_period_us": 1_000_000 // target_rate_hz,
        "minimum_temporal_groups": minimum_temporal_groups,
        "minimum_group_trace_cells": minimum_group_trace_cells,
    }


def _candidate_temporal_window_starts(
    frame_indices: Sequence[int],
    *,
    observations: int,
    stride: int,
) -> list[int]:
    """Return starts whose complete arithmetic frame window is available."""

    available = {int(value) for value in frame_indices}
    if len(available) < int(observations):
        return []
    span = (int(observations) - 1) * int(stride)
    maximum = max(available)
    return [
        start
        for start in sorted(available)
        if start + span <= maximum
        and all(start + offset * int(stride) in available for offset in range(int(observations)))
    ]


def select_registered_temporal_sample(
    manifest: pd.DataFrame,
    *,
    split: str,
    seed: int,
    sequences_per_split: int,
    observations_per_sequence: int = 64,
    source_frame_stride: int = 1,
    target_rate_hz: int = 5,
    test_permit: UntouchedTestPermit | None = None,
    exact_eligible_sequence_capacity: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select deterministic, outcome-blind MOT replay windows.

    Selection reads only publisher/study identity, frame availability, and the
    three registered mission-profile identities.  A sequence is eligible only
    when one exact ``observations_per_sequence`` arithmetic frame window exists
    at ``source_frame_stride``.  The selected replay is rebased to the frozen
    target rate after selection; this timing transform cannot influence which
    sequence or window is chosen.
    """

    split_name = str(split).strip()
    if split_name not in TEMPORAL_COMPONENT_BY_SPLIT:
        raise ValueError("Registered temporal sampling supports validation or test only")
    if exact_eligible_sequence_capacity is not None and split_name != "validation":
        raise ValueError(
            "Exact eligible-sequence capacity is permitted only for validation"
        )
    component = TEMPORAL_COMPONENT_BY_SPLIT[split_name]
    numeric_values = {
        "sequences_per_split": sequences_per_split,
        "observations_per_sequence": observations_per_sequence,
        "source_frame_stride": source_frame_stride,
        "target_rate_hz": target_rate_hz,
    }
    try:
        numeric_values = {name: int(value) for name, value in numeric_values.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("Temporal sampling parameters must be integers") from exc
    if any(value <= 0 for value in numeric_values.values()):
        raise ValueError("Temporal sampling parameters must be positive")
    sequences_per_split = numeric_values["sequences_per_split"]
    observations_per_sequence = numeric_values["observations_per_sequence"]
    source_frame_stride = numeric_values["source_frame_stride"]
    target_rate_hz = numeric_values["target_rate_hz"]
    if exact_eligible_sequence_capacity is not None:
        if (
            type(exact_eligible_sequence_capacity) is not int
            or exact_eligible_sequence_capacity <= 0
        ):
            raise ValueError(
                "exact_eligible_sequence_capacity must be a positive integer"
            )
        if exact_eligible_sequence_capacity != sequences_per_split:
            raise ValueError(
                "Exact eligible-sequence capacity must equal sequences_per_split"
            )
    if 1_000_000 % target_rate_hz:
        raise ValueError("target_rate_hz must divide one million microseconds exactly")
    if target_rate_hz != 5:
        raise ValueError("Registered VisDrone temporal replay rate must be exactly 5 Hz")
    capture_period_us = 1_000_000 // target_rate_hz

    sampling_columns = (
        "sample_id", "source_sample_id", "mission_profile", "component", "split",
        "split_unit", "sequence_id", "frame_index", "source_uid",
    )
    timing_columns = ("capture_time_us", "deadline_us", "media_path")
    missing = sorted(set((*sampling_columns, *timing_columns)).difference(manifest.columns))
    if missing:
        raise ValueError(f"Temporal sampling manifest is missing columns: {missing}")
    if manifest.empty or manifest["sample_id"].astype(str).duplicated().any():
        raise ValueError("Temporal sampling requires nonempty, unique sample rows")

    split_rows = manifest.loc[manifest["split"].astype(str).eq(split_name)]
    if split_rows.empty:
        raise StudyGateError(f"Temporal sampling found no rows for split {split_name!r}")
    mot_rows = split_rows.loc[split_rows["component"].astype(str).str.startswith("mot_")]
    unexpected_mot = sorted(set(mot_rows["component"].astype(str)) - {component})
    if unexpected_mot:
        raise StudyGateError(
            f"Temporal split {split_name!r} contains unregistered MOT components: {unexpected_mot}"
        )
    scoped_mask = split_rows["component"].astype(str).eq(component)
    scoped_identity = split_rows.loc[scoped_mask, sampling_columns].copy()
    if scoped_identity.empty:
        raise StudyGateError(
            f"Temporal sampling requires publisher component {component!r} in split {split_name!r}"
        )
    if split_name == "test":
        scoped_original = split_rows.loc[scoped_mask].copy()
        _verify_test_permit(test_permit, scoped_original)

    # From this point through the selection decision, retain only the declared
    # outcome-blind columns. Labels, objects, actions, risk, model outputs, and
    # network outcomes cannot be referenced by the ranking/window code.
    work = scoped_identity
    expected_profiles = set(REGISTERED_MISSION_PROFILES)
    profile_sets = work.groupby("source_sample_id", sort=True)["mission_profile"].agg(
        lambda values: set(map(str, values))
    )
    if profile_sets.empty or not profile_sets.map(lambda values: values == expected_profiles).all():
        raise StudyGateError(
            "Registered temporal sampling requires all three mission-profile copies per source frame"
        )
    if work.duplicated(["source_sample_id", "mission_profile"], keep=False).any():
        raise StudyGateError("A temporal source/profile pair occurs more than once")
    for column in (
        "component", "split", "split_unit", "sequence_id", "frame_index", "source_uid",
    ):
        counts = work.groupby("source_sample_id", sort=True)[column].nunique(dropna=False)
        if (counts != 1).any():
            raise StudyGateError(
                f"Temporal source/profile pairs disagree on outcome-blind field {column!r}"
            )
    numeric_frames = pd.to_numeric(work["frame_index"], errors="coerce")
    if numeric_frames.isna().any() or not np.equal(numeric_frames, np.floor(numeric_frames)).all():
        raise StudyGateError("Temporal sampling requires integral publisher frame indices")
    work["frame_index"] = numeric_frames.astype(np.int64)
    base = work.drop_duplicates("source_sample_id").copy()
    if base.duplicated(["sequence_id", "frame_index"], keep=False).any():
        raise StudyGateError("A MOT sequence/frame identity maps to multiple source observations")
    sequence_units = base.groupby("sequence_id", sort=True)["split_unit"].nunique(dropna=False)
    if (sequence_units != 1).any():
        raise StudyGateError("A temporal sequence maps to more than one natural split unit")

    eligible: list[dict[str, Any]] = []
    for sequence_id, sequence in base.groupby("sequence_id", sort=True):
        sequence = sequence.sort_values(["frame_index", "source_uid"], kind="stable")
        starts = _candidate_temporal_window_starts(
            sequence["frame_index"].astype(int).tolist(),
            observations=observations_per_sequence,
            stride=source_frame_stride,
        )
        if not starts:
            continue
        split_unit = str(sequence["split_unit"].iloc[0])
        sequence_hash = hashlib.sha256(canonical_json([
            WORKFLOW_VERSION, TEMPORAL_SAMPLING_METHOD, "sequence", int(seed),
            split_name, component, str(sequence_id), split_unit,
        ]).encode("utf-8")).hexdigest()
        window_rows = []
        for start in starts:
            end = start + (observations_per_sequence - 1) * source_frame_stride
            window_hash = hashlib.sha256(canonical_json([
                WORKFLOW_VERSION, TEMPORAL_SAMPLING_METHOD, "window", int(seed),
                split_name, component, str(sequence_id), split_unit,
                int(start), int(end), observations_per_sequence, source_frame_stride,
            ]).encode("utf-8")).hexdigest()
            window_rows.append((window_hash, int(start), int(end)))
        window_hash, start, end = min(window_rows)
        eligible.append({
            "sequence_id": str(sequence_id),
            "split_unit": split_unit,
            "available_source_frames": int(sequence["frame_index"].nunique()),
            "eligible_windows": int(len(starts)),
            "sequence_selection_hash": sequence_hash,
            "window_selection_hash": window_hash,
            "window_start_frame": start,
            "window_end_frame": end,
        })
    eligible.sort(key=lambda row: (row["sequence_selection_hash"], row["sequence_id"]))
    if len(eligible) < sequences_per_split:
        raise StudyGateError(
            "Registered temporal sampling has insufficient eligible MOT sequences: "
            f"{len(eligible)}/{sequences_per_split} in {split_name}; each needs "
            f"{observations_per_sequence} frames at source stride {source_frame_stride}"
        )
    if (
        exact_eligible_sequence_capacity is not None
        and len(eligible) != exact_eligible_sequence_capacity
    ):
        raise StudyGateError(
            "Predecessor-observed validation temporal capacity changed: "
            f"observed={len(eligible)!r}, "
            f"expected={exact_eligible_sequence_capacity}"
        )
    chosen = [
        {**item, "sequence_rank": int(rank)}
        for rank, item in enumerate(eligible[:sequences_per_split], start=1)
    ]
    selected_source_ids: list[str] = []
    window_by_sequence: dict[str, dict[str, Any]] = {}
    for item in chosen:
        sequence = base.loc[base["sequence_id"].astype(str).eq(item["sequence_id"])]
        desired = {
            item["window_start_frame"] + offset * source_frame_stride
            for offset in range(observations_per_sequence)
        }
        selected_frames = sequence.loc[sequence["frame_index"].astype(int).isin(desired)]
        if len(selected_frames) != observations_per_sequence:
            raise AssertionError("A validated temporal window changed during materialization")
        selected_source_ids.extend(selected_frames["source_sample_id"].astype(str))
        window_by_sequence[item["sequence_id"]] = item

    if split_name != "test":
        # Full development rows are materialized only after the outcome-blind
        # exact-capacity gate has passed.
        scoped_original = split_rows.loc[scoped_mask].copy()
    selected = scoped_original.loc[
        scoped_original["source_sample_id"].astype(str).isin(selected_source_ids)
    ].copy()
    selected["frame_index"] = pd.to_numeric(selected["frame_index"], errors="raise").astype(np.int64)
    selected = selected.sort_values(
        ["sequence_id", "mission_profile", "frame_index", "source_uid", "sample_id"],
        kind="stable", ignore_index=True,
    )
    expected_rows = sequences_per_split * observations_per_sequence * len(expected_profiles)
    if len(selected) != expected_rows:
        raise StudyGateError(
            f"Temporal sample has {len(selected)} profile rows; expected exactly {expected_rows}"
        )

    selection_payload = {
        "workflow_version": WORKFLOW_VERSION,
        "method": TEMPORAL_SAMPLING_METHOD,
        "seed": int(seed),
        "split": split_name,
        "component": component,
        "sequences_per_split": sequences_per_split,
        "observations_per_sequence": observations_per_sequence,
        "source_frame_stride": source_frame_stride,
        "target_rate_hz": target_rate_hz,
        "windows": [
            {
                key: item[key]
                for key in (
                    "sequence_id", "split_unit", "sequence_rank",
                    "sequence_selection_hash", "window_selection_hash",
                    "window_start_frame", "window_end_frame",
                )
            }
            for item in chosen
        ],
        "selected_source_ids": sorted(map(str, selected_source_ids)),
    }
    selection_sha256 = hash_object(selection_payload)
    selected["temporal_selection_sha256"] = selection_sha256
    selected["temporal_sampling_method"] = TEMPORAL_SAMPLING_METHOD
    selected["registered_target_rate_hz"] = target_rate_hz
    selected["registered_source_frame_stride"] = source_frame_stride
    selected["publisher_manifest_capture_time_us"] = pd.to_numeric(
        selected["capture_time_us"], errors="raise",
    ).astype(np.int64)
    selected["publisher_manifest_deadline_us"] = pd.to_numeric(
        selected["deadline_us"], errors="raise",
    ).astype(np.int64)
    deadline_budget = (
        selected["publisher_manifest_deadline_us"]
        - selected["publisher_manifest_capture_time_us"]
    )
    if (deadline_budget <= 0).any():
        raise StudyGateError("Temporal sample contains a nonpositive registered deadline budget")
    selected["temporal_observation_index"] = selected.groupby(
        ["sequence_id", "mission_profile"], sort=True,
    ).cumcount().astype(np.int64)
    selected["capture_time_us"] = (
        selected["temporal_observation_index"] * capture_period_us
    ).astype(np.int64)
    selected["trace_time_us"] = selected["capture_time_us"]
    selected["deadline_us"] = selected["capture_time_us"] + deadline_budget.to_numpy(np.int64)
    selected["temporal_sequence_rank"] = selected["sequence_id"].astype(str).map(
        lambda value: int(window_by_sequence[value]["sequence_rank"])
    )
    selected["temporal_window_start_frame"] = selected["sequence_id"].astype(str).map(
        lambda value: int(window_by_sequence[value]["window_start_frame"])
    )
    selected["temporal_window_end_frame"] = selected["sequence_id"].astype(str).map(
        lambda value: int(window_by_sequence[value]["window_end_frame"])
    )
    selected["temporal_episode_id"] = [
        hash_object({
            "selection_sha256": selection_sha256,
            "sequence_id": str(sequence_id),
            "mission_profile": str(profile),
        })[:20]
        for sequence_id, profile in zip(selected["sequence_id"], selected["mission_profile"])
    ]

    # Rebuild clips from the selected 5 Hz replay only. Each clip is causal and
    # cannot expose an unselected future/source frame.
    selected["clip_paths_json"] = "[]"
    for (_, _), group in selected.groupby(["sequence_id", "mission_profile"], sort=True):
        ordered = group.sort_values("temporal_observation_index", kind="stable")
        paths = ordered["media_path"].astype(str).tolist()
        for position, row_index in enumerate(ordered.index):
            if position:
                selected.at[row_index, "clip_paths_json"] = canonical_json(
                    paths[max(0, position - 2): position + 1]
                )

    profile_sets = selected.groupby("source_sample_id", sort=True)["mission_profile"].agg(
        lambda values: set(map(str, values))
    )
    profile_complete = bool(
        len(profile_sets) and profile_sets.map(lambda values: values == expected_profiles).all()
    )
    frame_steps_ok = True
    timing_steps_ok = True
    for (_, profile), group in selected.groupby(["sequence_id", "mission_profile"], sort=True):
        ordered = group.sort_values("temporal_observation_index", kind="stable")
        frame_steps_ok = frame_steps_ok and bool(
            np.diff(ordered["frame_index"].to_numpy(np.int64)).tolist()
            == [source_frame_stride] * (observations_per_sequence - 1)
        )
        timing_steps_ok = timing_steps_ok and bool(
            np.diff(ordered["capture_time_us"].to_numpy(np.int64)).tolist()
            == [capture_period_us] * (observations_per_sequence - 1)
        )

    audit_rows: list[dict[str, Any]] = []

    def gate(name: str, passed: bool, observed: Any, required: Any) -> None:
        audit_rows.append({
            "record_type": "gate",
            "gate": name,
            "passed": bool(passed),
            "observed": canonical_json(observed),
            "required": canonical_json(required),
            "split": split_name,
            "component": component,
            "sequence_id": "",
            "split_unit": "",
            "sequence_rank": 0,
            "sequence_selection_hash": "",
            "window_selection_hash": "",
            "available_source_frames": 0,
            "eligible_windows": 0,
            "window_start_frame": 0,
            "window_end_frame": 0,
            "selected_source_observations": 0,
            "selection_sha256": selection_sha256,
            "sampling_seed": int(seed),
            "sampling_method": TEMPORAL_SAMPLING_METHOD,
            "source_frame_stride": source_frame_stride,
            "target_rate_hz": target_rate_hz,
            "capture_period_us": capture_period_us,
            "sampling_columns_json": canonical_json(list(sampling_columns)),
            "outcome_columns_read_json": "[]",
        })

    gate("mot_component_only", set(selected["component"].astype(str)) == {component},
         sorted(set(selected["component"].astype(str))), [component])
    gate(
        "eligible_sequence_capacity",
        len(eligible) >= sequences_per_split,
        int(len(eligible)),
        {"at_least": sequences_per_split},
    )
    if exact_eligible_sequence_capacity is not None:
        gate(
            "exact_verified_validation_eligible_sequence_capacity",
            len(eligible) == exact_eligible_sequence_capacity,
            int(len(eligible)),
            exact_eligible_sequence_capacity,
        )
    gate("exact_registered_sequence_count", selected["sequence_id"].nunique() == sequences_per_split,
         int(selected["sequence_id"].nunique()), sequences_per_split)
    gate("exact_observations_per_sequence", bool(
        selected.groupby(["sequence_id", "mission_profile"])["source_sample_id"]
        .nunique().eq(observations_per_sequence).all()
    ), observations_per_sequence, observations_per_sequence)
    gate("all_source_profiles_complete", profile_complete,
         int(profile_sets.map(lambda values: values == expected_profiles).sum()), len(profile_sets))
    gate("source_frame_stride_exact", frame_steps_ok, source_frame_stride, source_frame_stride)
    gate("registered_five_hz_timing", timing_steps_ok and target_rate_hz == 5,
         {"target_rate_hz": target_rate_hz, "capture_period_us": capture_period_us},
         {"target_rate_hz": 5, "capture_period_us": 200_000})
    gate("outcome_blind_selection", True, list(sampling_columns),
         "identity/frame availability/profile pairing only")
    gate("test_rows_hash_bound", split_name != "test" or set(
        selected["test_ledger_sha256"].astype(str)
    ) == {str(test_permit.ledger_sha256)}, int(len(selected)),
         "validation rows or every test row bound to the active OPEN ledger")

    for item in chosen:
        audit_rows.append({
            "record_type": "selected_sequence_window",
            "gate": "selected_sequence_window",
            "passed": True,
            "observed": canonical_json({
                "available_source_frames": item["available_source_frames"],
                "eligible_windows": item["eligible_windows"],
            }),
            "required": canonical_json({
                "observations": observations_per_sequence,
                "stride": source_frame_stride,
            }),
            "split": split_name,
            "component": component,
            "sequence_id": item["sequence_id"],
            "split_unit": item["split_unit"],
            "sequence_rank": int(item["sequence_rank"]),
            "sequence_selection_hash": item["sequence_selection_hash"],
            "window_selection_hash": item["window_selection_hash"],
            "available_source_frames": int(item["available_source_frames"]),
            "eligible_windows": int(item["eligible_windows"]),
            "window_start_frame": int(item["window_start_frame"]),
            "window_end_frame": int(item["window_end_frame"]),
            "selected_source_observations": observations_per_sequence,
            "selection_sha256": selection_sha256,
            "sampling_seed": int(seed),
            "sampling_method": TEMPORAL_SAMPLING_METHOD,
            "source_frame_stride": source_frame_stride,
            "target_rate_hz": target_rate_hz,
            "capture_period_us": capture_period_us,
            "sampling_columns_json": canonical_json(list(sampling_columns)),
            "outcome_columns_read_json": "[]",
        })
    audit = pd.DataFrame.from_records(audit_rows)
    if audit.empty or not audit["passed"].astype(bool).all():
        failed = audit.loc[~audit["passed"].astype(bool), "gate"].astype(str).tolist()
        raise StudyGateError(f"Registered temporal sampling audit failed: {failed}")
    return selected.reset_index(drop=True), audit.reset_index(drop=True)


def select_registered_temporal_sample_from_config(
    manifest: pd.DataFrame,
    *,
    benchmark: Mapping[str, Any],
    split: str,
    seed: int,
    test_permit: UntouchedTestPermit | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply exactly the hash-bound temporal replay design in configuration."""

    spec = registered_temporal_sampling_spec(benchmark, split)
    amendment = benchmark.get("temporal_feasibility_amendment")
    exact_capacity = None
    if str(spec["split"]) == "validation" and amendment is not None:
        if not isinstance(amendment, Mapping):
            raise StudyGateError(
                "Temporal feasibility amendment must be a mapping"
            )
        exact_capacity = amendment.get(
            "predecessor_observed_eligible_validation_sequences"
        )
        if (
            type(exact_capacity) is not int
            or exact_capacity <= 0
            or exact_capacity != int(spec["sequences_per_split"])
        ):
            raise StudyGateError(
                "Temporal feasibility amendment capacity differs from configuration"
            )
    selected, audit = select_registered_temporal_sample(
        manifest,
        split=str(spec["split"]),
        seed=int(seed),
        sequences_per_split=int(spec["sequences_per_split"]),
        observations_per_sequence=int(spec["observations_per_sequence"]),
        source_frame_stride=int(spec["source_frame_stride"]),
        target_rate_hz=int(spec["target_rate_hz"]),
        test_permit=test_permit,
        exact_eligible_sequence_capacity=exact_capacity,
    )
    if exact_capacity is not None:
        capacity_rows = audit.loc[
            audit["gate"].astype(str).eq(
                "exact_verified_validation_eligible_sequence_capacity"
            )
        ]
        try:
            observed_capacity = json.loads(
                str(capacity_rows.iloc[0]["observed"])
            )
        except (IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise StudyGateError(
                "Temporal eligibility-capacity audit is malformed"
            ) from exc
        if (
            len(capacity_rows) != 1
            or type(observed_capacity) is not int
            or observed_capacity != exact_capacity
            or not bool(capacity_rows.iloc[0]["passed"])
        ):
            raise StudyGateError(
                "Predecessor-observed validation temporal capacity changed: "
                f"observed={observed_capacity!r}, expected={exact_capacity}"
            )
    return selected, audit


def write_registered_temporal_sample(
    sample: pd.DataFrame,
    audit: pd.DataFrame,
    output_dir: str | Path,
) -> Mapping[str, Path]:
    """Persist one immutable temporal replay sample and its detailed audit."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise StudyGateError(f"Registered temporal sample output already exists: {destination}")
    if sample.empty or audit.empty or "selection_sha256" not in audit:
        raise StudyGateError("Registered temporal sample and audit must be nonempty")
    selection_hashes = set(audit["selection_sha256"].astype(str))
    sample_hashes = set(sample.get("temporal_selection_sha256", pd.Series(dtype=str)).astype(str))
    if (
        len(selection_hashes) != 1
        or selection_hashes != sample_hashes
        or not audit["passed"].astype(bool).all()
    ):
        raise StudyGateError("Registered temporal sample requires one passing bound audit")
    selection_sha256 = next(iter(selection_hashes))
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        sample_path = staging / "registered_temporal_sample.csv.gz"
        audit_path = staging / "registered_temporal_sampling_audit.csv"
        _deterministic_csv(_serialized_analysis_manifest(sample), sample_path, compress=True)
        _deterministic_csv(audit, audit_path)
        manifest_path = write_artifact_hash_manifest(
            {"registered_temporal_sample": sample_path, "temporal_sampling_audit": audit_path},
            staging / "REGISTERED_TEMPORAL_SAMPLE_ARTIFACTS.json",
            metadata={
                "selection_sha256": selection_sha256,
                "sampling_method": TEMPORAL_SAMPLING_METHOD,
            },
            root=staging,
        )
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "sample": destination / sample_path.name,
        "audit": destination / audit_path.name,
        "artifact_manifest": destination / manifest_path.name,
    }


def _trace_shape(family: str, position: float, rng: np.random.Generator) -> tuple[float, float, float, float]:
    """Return rate Mbps, one-way delay ms, jitter ms, and packet loss."""

    noise = float(rng.uniform(-0.08, 0.08))
    if family == "steady_low":
        return 0.35 * (1 + noise), 95.0, 8.0, 0.04
    if family == "steady_mid":
        return 2.0 * (1 + noise), 45.0, 4.0, 0.015
    if family == "steady_high":
        return 12.0 * (1 + noise), 18.0, 2.0, 0.002
    if family == "periodic_congestion":
        low = int(position * 12) % 4 in {1, 2}
        return (0.25 if low else 5.0) * (1 + noise), 80.0 if low else 25.0, 9.0, 0.06 if low else 0.01
    if family == "bursty_loss":
        burst = int(position * 12) % 5 == 2
        return 2.5 * (1 + noise), 40.0, 6.0, 0.18 if burst else 0.005
    if family == "delay_spike":
        spike = 0.38 <= position <= 0.62
        return 3.0 * (1 + noise), 190.0 if spike else 28.0, 20.0 if spike else 3.0, 0.02
    if family == "sudden_drop":
        return (8.0 if position < 0.4 else 0.2) * (1 + noise), 75.0, 10.0, 0.06
    if family == "recovery":
        return (0.2 if position < 0.45 else 8.0) * (1 + noise), 70.0, 8.0, 0.04
    if family == "ramp":
        return (0.15 + 10.0 * position) * (1 + noise), 55.0 - 25.0 * position, 5.0, 0.02
    if family == "markov":
        poor = bool(rng.random() < (0.68 if position < 0.5 else 0.38))
        return (0.3 if poor else 6.0) * (1 + noise), 100.0 if poor else 30.0, 12.0, 0.10 if poor else 0.01
    if family == "fading":
        wave = (1.0 + math.sin(2.0 * math.pi * 3.0 * position)) / 2.0
        return (0.25 + 7.0 * wave) * (1 + noise), 35.0 + 45.0 * (1.0 - wave), 9.0, 0.01 + 0.08 * (1.0 - wave)
    if family == "outage":
        outage = 0.35 <= position < 0.6
        return (0.0 if outage else 5.0) * (1 + noise), 40.0, 5.0, 1.0 if outage else 0.01
    raise ValueError(f"Unregistered trace family: {family}")


def audit_registered_temporal_trace_alignment(frame: pd.DataFrame) -> pd.DataFrame:
    """Audit the frozen 64-observation/5 Hz episode-aligned trace envelope."""

    rows: list[dict[str, Any]] = []

    def gate(name: str, passed: bool, observed: Any, required: Any) -> None:
        rows.append({
            "gate": name,
            "passed": bool(passed),
            "observed": canonical_json(observed),
            "required": canonical_json(required),
        })

    required_columns = {
        "trace_id", "split", "family", "direction", "start_us", "end_us",
        "trace_phase", "service_rate_bps", "base_one_way_delay_us",
        "jitter_parameter_us", "packet_loss_parameter",
    }
    missing = sorted(required_columns.difference(frame.columns))
    gate("temporal_trace_schema", not missing, missing, [])
    if missing or frame.empty:
        gate("temporal_trace_nonempty", False, int(len(frame)), "> 0")
        return pd.DataFrame.from_records(rows)
    gate("temporal_trace_nonempty", True, int(len(frame)), "> 0")

    timing_failures: list[str] = []
    phase_failures: list[str] = []
    hold_failures: list[str] = []
    transition_failures: list[str] = []
    metric_columns = [
        "service_rate_bps", "base_one_way_delay_us",
        "jitter_parameter_us", "packet_loss_parameter",
    ]
    dynamic_families = (
        set(VALIDATION_TRACE_FAMILIES) | set(TEST_TRACE_FAMILIES)
    ) - {"steady_low", "steady_mid", "steady_high"}
    expected_starts = np.arange(
        0,
        REGISTERED_TEMPORAL_TRACE_DURATION_US,
        REGISTERED_TEMPORAL_TRACE_SEGMENT_US,
        dtype=np.int64,
    )
    for (trace_id, direction), group in frame.groupby(["trace_id", "direction"], sort=True):
        key = f"{trace_id}:{direction}"
        ordered = group.sort_values("start_us", kind="stable")
        starts = pd.to_numeric(ordered["start_us"], errors="coerce").to_numpy()
        ends = pd.to_numeric(ordered["end_us"], errors="coerce").to_numpy()
        if (
            len(ordered) != len(expected_starts)
            or not np.array_equal(starts, expected_starts)
            or not np.array_equal(ends, expected_starts + REGISTERED_TEMPORAL_TRACE_SEGMENT_US)
        ):
            timing_failures.append(key)
            continue
        dynamic = ordered.loc[ordered["trace_phase"].astype(str).eq("dynamic")]
        hold = ordered.loc[ordered["trace_phase"].astype(str).eq("terminal_hold")]
        if (
            len(dynamic) != REGISTERED_TEMPORAL_TRACE_DYNAMIC_US // REGISTERED_TEMPORAL_TRACE_SEGMENT_US
            or len(hold) != (
                REGISTERED_TEMPORAL_TRACE_DURATION_US - REGISTERED_TEMPORAL_TRACE_DYNAMIC_US
            ) // REGISTERED_TEMPORAL_TRACE_SEGMENT_US
            or int(dynamic["end_us"].max()) != REGISTERED_TEMPORAL_TRACE_DYNAMIC_US
            or int(hold["start_us"].min()) != REGISTERED_TEMPORAL_TRACE_DYNAMIC_US
        ):
            phase_failures.append(key)
            continue
        terminal_values = hold[metric_columns].reset_index(drop=True)
        final_dynamic = dynamic.iloc[-1][metric_columns]
        if (
            any(terminal_values[column].nunique(dropna=False) != 1 for column in metric_columns)
            or any(terminal_values.iloc[0][column] != final_dynamic[column] for column in metric_columns)
        ):
            hold_failures.append(key)
        family = str(ordered["family"].iloc[0])
        if family in dynamic_families and len(dynamic[metric_columns].drop_duplicates()) < 2:
            transition_failures.append(key)

    gate("one_second_contiguous_segments", not timing_failures, timing_failures, [])
    gate("twelve_second_dynamic_six_second_hold", not phase_failures, phase_failures, [])
    gate("terminal_hold_repeats_last_dynamic_state", not hold_failures, hold_failures, [])
    gate("dynamic_families_change_before_terminal_hold", not transition_failures,
         transition_failures, [])
    test_traces = int(frame.loc[frame["split"].astype(str).eq("test"), "trace_id"].nunique())
    gate("thirty_held_out_test_traces", test_traces == 30, test_traces, 30)
    last_capture_us = (64 - 1) * (1_000_000 // 5)
    state_expiry_grace_us = 2_000_000
    required_expiry_us = last_capture_us + 2_000_000 + state_expiry_grace_us
    gate(
        "five_hz_episode_deadline_and_state_expiry_fit_trace",
        required_expiry_us < REGISTERED_TEMPORAL_TRACE_DURATION_US,
        {
            "last_capture_us": last_capture_us,
            "max_registered_deadline_us": last_capture_us + 2_000_000,
            "state_expiry_grace_us": state_expiry_grace_us,
            "required_expiry_us": required_expiry_us,
            "trace_end_us": REGISTERED_TEMPORAL_TRACE_DURATION_US,
        },
        "last capture, maximum 2 s deadline, and default 2 s state-expiry grace are measured",
    )
    coverage_traces: list[LinkTrace] = []
    for (trace_id, direction), group in frame.groupby(["trace_id", "direction"], sort=True):
        ordered = group.sort_values("start_us", kind="stable")
        segments = tuple(
            LinkSegment(
                int(row.start_us), int(row.end_us), int(row.service_rate_bps),
                int(row.base_one_way_delay_us), int(row.jitter_parameter_us),
                float(row.packet_loss_parameter),
                queue_capacity_bytes=(
                    int(row.queue_capacity_bytes)
                    if hasattr(row, "queue_capacity_bytes") and pd.notna(row.queue_capacity_bytes)
                    else None
                ),
            )
            for row in ordered.itertuples(index=False)
        )
        coverage_traces.append(LinkTrace(
            str(trace_id), Direction(str(direction)), TraceSemantics.RAW_SERVICE,
            segments, EndBehavior.TERMINATE,
        ))
    coverage_manifest = pd.DataFrame({
        "capture_time_us": np.arange(64, dtype=np.int64) * 200_000,
        "trace_time_us": np.arange(64, dtype=np.int64) * 200_000,
        "deadline_us": np.arange(64, dtype=np.int64) * 200_000 + 2_000_000,
    })
    coverage_audit = scientific_episode_coverage_audit(
        coverage_traces,
        coverage_manifest,
        state_expiry_grace_us=state_expiry_grace_us,
    )
    failed_coverage = (
        coverage_audit.loc[~coverage_audit["passed"].astype(bool), "gate"].astype(str).tolist()
        if not coverage_audit.empty and "passed" in coverage_audit
        else ["coverage_audit_missing"]
    )
    gate(
        "scientific_episode_coverage_audit",
        bool(not coverage_audit.empty and coverage_audit["passed"].astype(bool).all()),
        {"failed_gates": failed_coverage, "audit": coverage_audit.to_dict("records")},
        "all scientific_episode_coverage_audit gates pass",
    )
    return pd.DataFrame.from_records(rows)


def generate_registered_trace_manifest(
    *,
    seed: int = 20260823,
    validation_realizations_per_family: int = 2,
    test_realizations_per_family: int = 5,
    duration_us: int = REGISTERED_TEMPORAL_TRACE_DURATION_US,
    segment_us: int = REGISTERED_TEMPORAL_TRACE_SEGMENT_US,
) -> pd.DataFrame:
    """Generate registered *emulated* raw-service uplink/downlink traces.

    The six test families are disjoint from validation and five realizations
    per test family yield the registered 30 held-out traces.  These traces are
    controlled emulation inputs, never represented as measured field traces.
    """

    for name, value in (
        ("validation_realizations_per_family", validation_realizations_per_family),
        ("test_realizations_per_family", test_realizations_per_family),
        ("duration_us", duration_us),
        ("segment_us", segment_us),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if (
        int(duration_us) != REGISTERED_TEMPORAL_TRACE_DURATION_US
        or int(segment_us) != REGISTERED_TEMPORAL_TRACE_SEGMENT_US
    ):
        raise ValueError(
            "Registered VisDrone traces are frozen at 18 s with 1 s segments"
        )
    records: list[dict[str, Any]] = []
    design = (
        ("validation", VALIDATION_TRACE_FAMILIES, int(validation_realizations_per_family)),
        ("test", TEST_TRACE_FAMILIES, int(test_realizations_per_family)),
    )
    segments = int(duration_us) // int(segment_us)
    dynamic_segments = REGISTERED_TEMPORAL_TRACE_DYNAMIC_US // int(segment_us)
    for split, families, realizations in design:
        for family in families:
            for realization in range(realizations):
                trace_seed = _stable_seed(WORKFLOW_VERSION, int(seed), split, family, realization)
                trace_id = f"visdrone-{split}-{family}-{realization:02d}"
                for direction in ("uplink", "downlink"):
                    rng = np.random.default_rng(_stable_seed(trace_seed, direction))
                    direction_scale = 1.0 if direction == "uplink" else 0.92
                    last_dynamic_shape: tuple[float, float, float, float] | None = None
                    for segment_index in range(segments):
                        start_us = segment_index * int(segment_us)
                        end_us = start_us + int(segment_us)
                        if segment_index < dynamic_segments:
                            position = (segment_index + 0.5) / dynamic_segments
                            last_dynamic_shape = _trace_shape(family, position, rng)
                            trace_phase = "dynamic"
                        else:
                            if last_dynamic_shape is None:  # pragma: no cover - constants prevent it
                                raise AssertionError("Trace has no dynamic state to hold")
                            trace_phase = "terminal_hold"
                        rate_mbps, delay_ms, jitter_ms, loss = last_dynamic_shape
                        rate_bps = max(0, int(round(rate_mbps * direction_scale * 1_000_000)))
                        records.append({
                            "trace_id": trace_id,
                            "family": family,
                            "split": split,
                            "seed": int(trace_seed),
                            "realization": int(realization),
                            "direction": direction,
                            "semantics": "raw_service",
                            "start_us": int(start_us),
                            "end_us": int(end_us),
                            "trace_phase": trace_phase,
                            "registered_dynamic_end_us": REGISTERED_TEMPORAL_TRACE_DYNAMIC_US,
                            "registered_trace_end_us": REGISTERED_TEMPORAL_TRACE_DURATION_US,
                            "registered_temporal_observations": 64,
                            "registered_temporal_rate_hz": 5,
                            "service_rate_bps": rate_bps,
                            "base_one_way_delay_us": int(round(delay_ms * 1000.0)),
                            "jitter_parameter_us": int(round(jitter_ms * 1000.0)),
                            "packet_loss_parameter": min(1.0, max(0.0, float(loss))),
                            "queue_capacity_bytes": 131_072,
                            "end_behavior": "terminate",
                            "provenance_class": "registered_emulation",
                            "trace_source": "deterministic_registered_synthetic_channel",
                            "scientific_scope": "controlled_network_emulation_not_field_measurement",
                        })
    frame = pd.DataFrame.from_records(records).sort_values(
        ["split", "family", "trace_id", "direction", "start_us"],
        kind="stable",
        ignore_index=True,
    )
    audit = scientific_trace_design_audit(frame, required_test_families=TEST_TRACE_FAMILIES)
    if audit.empty or not audit["passed"].astype(bool).all():
        failed = audit.loc[~audit["passed"].astype(bool), "gate"].astype(str).tolist()
        raise AssertionError(f"Registered trace design failed its own audit: {failed}")
    temporal_audit = audit_registered_temporal_trace_alignment(frame)
    if temporal_audit.empty or not temporal_audit["passed"].astype(bool).all():
        failed = temporal_audit.loc[
            ~temporal_audit["passed"].astype(bool), "gate"
        ].astype(str).tolist()
        raise AssertionError(f"Registered temporal trace alignment failed: {failed}")
    test_traces = frame.loc[frame["split"].eq("test"), "trace_id"].nunique()
    if test_traces != len(TEST_TRACE_FAMILIES) * int(test_realizations_per_family):
        raise AssertionError("Registered test trace realization count is inconsistent")
    return frame


def _markdown_table(frame: pd.DataFrame) -> str:
    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(cell(column) for column in frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows]) + "\n"


def _latex_table(frame: pd.DataFrame) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            rendered = f"{float(value):.8g}"
        else:
            rendered = str(value).replace("\n", " ")
        # Replace the backslash first and then scan original characters so
        # replacement braces are not escaped a second time.
        return "".join(replacements.get(character, character) for character in rendered)

    columns = "l" * len(frame.columns)
    lines = [
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(cell(column) for column in frame.columns) + r" \\",
        r"\midrule",
    ]
    lines.extend(
        " & ".join(cell(value) for value in row) + r" \\"
        for row in frame.itertuples(index=False, name=None)
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def export_paper_tables(
    tables: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    scientific: bool,
) -> pd.DataFrame:
    """Export each nonempty table as deterministic CSV, Markdown, and LaTeX."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if not tables:
        raise ValueError("At least one paper table is required")
    records: list[dict[str, Any]] = []
    for name, table in sorted(tables.items()):
        if _SAFE_NAME.fullmatch(str(name)) is None:
            raise ValueError(f"Unsafe paper-table name: {name!r}")
        if not isinstance(table, pd.DataFrame) or table.empty:
            raise StudyGateError(f"Paper table {name!r} is missing or empty")
        csv_path = destination / f"{name}.csv"
        markdown_path = destination / f"{name}.md"
        latex_path = destination / f"{name}.tex"
        _deterministic_csv(table, csv_path)
        _atomic_text(markdown_path, _markdown_table(table))
        _atomic_text(latex_path, _latex_table(table))
        for format_name, path in (
            ("csv", csv_path), ("markdown", markdown_path), ("latex", latex_path),
        ):
            records.append({
                "table": str(name),
                "format": format_name,
                "path": str(path),
                "sha256": hash_file(path),
                "bytes": path.stat().st_size,
                "rows": int(len(table)),
                "columns": int(len(table.columns)),
                "scientific": bool(scientific),
                "evidence_class": "empirical" if scientific else "non_scientific",
            })
    manifest = pd.DataFrame.from_records(records)
    _deterministic_csv(manifest, destination / "TABLE_MANIFEST.csv")
    return manifest


def build_readiness_claim_ledger(
    *,
    phase: str,
    evidence: Mapping[str, Any],
    hypotheses: Mapping[str, str | Mapping[str, Any]] | None = None,
    construct_validity_status: str = "NOT_RUN",
    selected_vlm_used: bool = False,
    deployed_perception_scope: str | None = None,
) -> pd.DataFrame:
    """Build a fail-closed ledger for paper statements and H1--H5.

    Evidence is never inferred from filenames.  Callers must explicitly pass
    the Boolean results of real-data, real-inference, gate, freeze, and test
    checks.  Missing evidence is false.
    """

    phase_name = str(phase).strip().lower()
    if phase_name not in {"prepare", "validation", "test", "report", "embedded"}:
        raise ValueError("phase must be prepare, validation, test, report, or embedded")
    if not isinstance(selected_vlm_used, (bool, np.bool_)):
        raise ValueError("selected_vlm_used must be Boolean")
    expected_scope = (
        "selected_vlm_scope" if bool(selected_vlm_used)
        else "detector_semantic_scope"
    )
    perception_scope = (
        expected_scope
        if deployed_perception_scope is None
        else str(deployed_perception_scope).strip()
    )
    if perception_scope not in {"detector_semantic_scope", "selected_vlm_scope"}:
        raise ValueError("deployed_perception_scope is invalid")
    if perception_scope != expected_scope:
        raise ValueError(
            "deployed_perception_scope is inconsistent with selected_vlm_used"
        )
    generic_required_evidence = (
        "official_visdrone_receipt",
        "publisher_group_splits_passed",
        "real_detector_inference",
        "validation_gates_passed",
        "protocol_frozen",
        "one_time_test_complete",
        "paper_artifacts_hashed",
    )
    evidence_state = {
        name: bool(evidence.get(name, False)) for name in generic_required_evidence
    }
    empirical_ready = all(evidence_state.values())
    selected_vlm_inference = bool(evidence.get("selected_vlm_inference", False))
    vlm_specific_ready = bool(
        empirical_ready and selected_vlm_used and selected_vlm_inference
    )
    rows: list[dict[str, Any]] = []

    def claim(
        item_type: str,
        item_id: str,
        description: str,
        status: str,
        reason: str,
        *,
        count_gates: Sequence[Mapping[str, Any]] = (),
        scope: str = "visdrone_sensor_space",
    ) -> None:
        rows.append({
            "phase": phase_name,
            "item_type": item_type,
            "item_id": item_id,
            "description": description,
            "status": status,
            "eligible": status in {"PASS", "ELIGIBLE"},
            "reason": reason,
            "count_gates_json": canonical_json(list(count_gates)) if count_gates else "[]",
            "scope": scope,
            "deployed_perception_scope": perception_scope,
        })

    missing = [name for name, passed in evidence_state.items() if not passed]
    claim(
        "claim", "visdrone_sensor_space_results",
        "Publisher-box-grounded VisDrone sensor-space endpoint results",
        "ELIGIBLE" if empirical_ready else "BLOCKED",
        "all empirical gates passed" if empirical_ready else f"missing evidence: {missing}",
    )
    vlm_missing = []
    if not empirical_ready:
        vlm_missing.extend(missing)
    if not selected_vlm_used:
        vlm_missing.append("frozen selected VLM was not the deployed perception path")
    if not selected_vlm_inference:
        vlm_missing.append("measured inference rows from the frozen selected VLM")
    claim(
        "claim", "selected_vlm_contribution",
        "Contribution attributable to the validation-selected contextual VLM",
        "ELIGIBLE" if vlm_specific_ready else "BLOCKED",
        (
            "generic empirical gates, frozen selected-model gate, and selected-VLM "
            "inference evidence passed"
            if vlm_specific_ready else f"missing selected-VLM evidence: {vlm_missing}"
        ),
        scope="visdrone_selected_vlm",
    )
    construct = str(construct_validity_status).strip().upper()
    if construct not in {"PASS", "FAIL", "NOT_RUN", "BLOCKED"}:
        raise ValueError("construct_validity_status is invalid")
    claim(
        "claim", "human_construct_validity_audit",
        "Optional blinded two-annotator construct-validity audit",
        "ELIGIBLE" if construct == "PASS" and empirical_ready else "BLOCKED",
        "audit passed" if construct == "PASS" and empirical_ready else f"audit status={construct}",
    )

    hypothesis_values = {str(key).upper(): value for key, value in (hypotheses or {}).items()}
    allowed = {"PASS", "FAIL", "BLOCKED", "NOT_RUN", "NOT_ESTIMABLE", "N_A"}
    for hypothesis in ("H1", "H2", "H3", "H4"):
        raw_result = hypothesis_values.get(hypothesis, "NOT_RUN")
        explicit_reason = ""
        count_gates: list[dict[str, Any]] = []
        if isinstance(raw_result, Mapping):
            result = str(raw_result.get("status", "")).strip().upper()
            explicit_reason = str(raw_result.get("reason", "")).strip()
            raw_gates = raw_result.get("count_gates", ())
            if not isinstance(raw_gates, Sequence) or isinstance(raw_gates, (str, bytes)):
                raise ValueError(f"{hypothesis} count_gates must be a sequence")
            for raw_gate in raw_gates:
                if not isinstance(raw_gate, Mapping):
                    raise ValueError(f"{hypothesis} has a malformed count gate")
                gate = str(raw_gate.get("gate", "")).strip()
                observed = raw_gate.get("observed")
                required = raw_gate.get("required")
                if (
                    not gate
                    or isinstance(observed, bool)
                    or not isinstance(observed, int)
                    or observed < 0
                    or isinstance(required, bool)
                    or not isinstance(required, int)
                    or required <= 0
                ):
                    raise ValueError(f"{hypothesis} has an invalid count gate")
                passed = observed >= required
                if "passed" in raw_gate:
                    recorded_pass = raw_gate["passed"]
                    if (
                        not isinstance(recorded_pass, (bool, np.bool_))
                        or bool(recorded_pass) != passed
                    ):
                        raise ValueError(f"{hypothesis} count-gate pass flag is inconsistent")
                count_gates.append({
                    "gate": gate,
                    "observed": observed,
                    "required": required,
                    "passed": passed,
                })
        else:
            result = str(raw_result).strip().upper()
        if result not in allowed:
            raise ValueError(f"Invalid hypothesis status for {hypothesis}: {result}")
        if result == "NOT_ESTIMABLE":
            if not explicit_reason:
                raise ValueError(f"{hypothesis} NOT_ESTIMABLE requires an explicit reason")
            if not count_gates or all(gate["passed"] for gate in count_gates):
                raise ValueError(
                    f"{hypothesis} NOT_ESTIMABLE requires at least one explicit failed count gate"
                )
        if not empirical_ready and result == "PASS":
            result = "BLOCKED"
            explicit_reason = "empirical readiness gates are incomplete"
        claim(
            "hypothesis", hypothesis,
            f"Registered VisDrone hypothesis {hypothesis}",
            result,
            (
                explicit_reason if explicit_reason else
                "confirmatory result" if result in {"PASS", "FAIL"} else
                "confirmatory evidence unavailable"
            ),
            count_gates=count_gates,
        )
    claim(
        "hypothesis", "H5",
        "Embedded Jetson hypothesis is outside the workstation-only VisDrone paper run",
        "N_A",
        "requires separately registered target-device raw evidence",
    )
    for item in PROHIBITED_CLAIMS:
        claim(
            "prohibited_claim", item, item.replace("_", " "), "BLOCKED",
            "not identifiable from VisDrone publisher boxes/tracks and this sensor-space rule",
        )
    return pd.DataFrame.from_records(rows)


def write_artifact_hash_manifest(
    artifacts: Mapping[str, str | Path],
    output_path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
) -> Path:
    """Write a portable SHA-256/size manifest for immutable result artifacts."""

    path = Path(output_path).expanduser().resolve()
    base = Path(root).expanduser().resolve() if root is not None else path.parent
    entries: dict[str, dict[str, Any]] = {}
    if not artifacts:
        raise ValueError("At least one artifact is required")
    for name, value in sorted(artifacts.items()):
        if _SAFE_NAME.fullmatch(str(name)) is None:
            raise ValueError(f"Unsafe artifact name: {name!r}")
        artifact = Path(value).expanduser().resolve()
        if artifact == path:
            raise ValueError("An artifact hash manifest cannot include itself")
        if not artifact.is_file():
            raise StudyGateError(f"Artifact {name!r} is missing: {artifact}")
        try:
            relative = artifact.relative_to(base)
        except ValueError as exc:
            raise StudyGateError(f"Artifact {name!r} escapes manifest root {base}") from exc
        entries[str(name)] = {
            "path": relative.as_posix(),
            "sha256": hash_file(artifact),
            "bytes": artifact.stat().st_size,
        }
    try:
        root_reference = Path(os.path.relpath(base, path.parent)).as_posix()
    except ValueError as exc:
        raise StudyGateError("Artifact-manifest root is not portable from its output path") from exc
    payload = {
        "schema_version": 1,
        "workflow_version": WORKFLOW_VERSION,
        "root": root_reference,
        "metadata": dict(metadata or {}),
        "artifacts": entries,
    }
    return _atomic_json(path, payload)


def verify_artifact_hash_manifest(path: str | Path) -> Mapping[str, Path]:
    """Verify every portable artifact entry and return resolved paths."""

    manifest_source = Path(path).expanduser()
    if manifest_source.is_symlink():
        raise StudyGateError(f"Artifact manifest is a symbolic link: {manifest_source}")
    manifest_path = manifest_source.resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StudyGateError(f"Artifact manifest is unreadable: {manifest_path}") from exc
    entries = payload.get("artifacts") if isinstance(payload, Mapping) else None
    if not isinstance(entries, Mapping) or not entries:
        raise StudyGateError("Artifact hash manifest has no entries")
    root_value = Path(str(payload.get("root", "."))).expanduser()
    base = (
        root_value.resolve()
        if root_value.is_absolute()
        else (manifest_path.parent / root_value).resolve()
    )
    resolved: dict[str, Path] = {}
    for name, item in sorted(entries.items()):
        if not isinstance(item, Mapping):
            raise StudyGateError(f"Artifact manifest entry {name!r} is invalid")
        candidate_source = base / str(item.get("path", ""))
        if candidate_source.is_symlink():
            raise StudyGateError(
                f"Artifact manifest entry {name!r} is a symbolic link"
            )
        candidate = candidate_source.resolve()
        try:
            candidate.relative_to(base)
        except ValueError as exc:
            raise StudyGateError(f"Artifact manifest entry {name!r} escapes its root") from exc
        expected = str(item.get("sha256", ""))
        try:
            expected_bytes = int(item.get("bytes", -1))
        except (TypeError, ValueError):
            expected_bytes = -1
        if (
            not candidate.is_file()
            or _HEX64.fullmatch(expected) is None
            or hash_file(candidate) != expected
            or candidate.stat().st_size != expected_bytes
        ):
            raise StudyGateError(f"Artifact changed or is missing: {candidate}")
        resolved[str(name)] = candidate
    return resolved


def verify_paper_completion_receipt(
    receipt_path: str | Path,
    *,
    expected_results_manifest: str | Path | None = None,
    expected_claim_ledger: str | Path | None = None,
    expected_protocol: str | Path | None = None,
    expected_test_ledger: str | Path | None = None,
    expected_report_environment_receipt: str | Path | None = None,
) -> Mapping[str, Any]:
    """Verify the final, non-self-referential paper completion envelope.

    The completion receipt is intentionally written after the outer result
    manifest.  This verifier closes that final link: it authenticates every
    path/hash named by the receipt, verifies both levels of artifact manifests,
    checks the outer-manifest metadata bindings, and confirms that the claim
    ledger remains in the receipt-pending state that the receipt completes.
    Callers may supply the expected run paths to prevent a valid receipt from
    being replayed for another run.
    """

    receipt_source = Path(receipt_path).expanduser()
    if receipt_source.is_symlink():
        raise StudyGateError(
            f"Paper completion receipt is a symbolic link: {receipt_source}"
        )
    path = receipt_source.resolve()
    if not path.is_file():
        raise StudyGateError(f"Paper completion receipt is missing or unsafe: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StudyGateError(f"Paper completion receipt is unreadable: {path}") from exc
    if not isinstance(payload, Mapping):
        raise StudyGateError("Paper completion receipt must be a mapping")
    if payload.get("schema_version") != PAPER_COMPLETION_RECEIPT_SCHEMA:
        raise StudyGateError("Paper completion receipt schema is invalid")
    if payload.get("status") != "COMPLETE":
        raise StudyGateError("Paper completion receipt is not COMPLETE")

    def verified_file(
        path_field: str,
        hash_field: str,
        expected_path: str | Path | None,
    ) -> Path:
        candidate_source = Path(str(payload.get(path_field, ""))).expanduser()
        if candidate_source.is_symlink():
            raise StudyGateError(
                f"Paper completion receipt {path_field} is a symbolic link"
            )
        candidate = candidate_source.resolve()
        if expected_path is not None and candidate != Path(expected_path).expanduser().resolve():
            raise StudyGateError(f"Paper completion receipt {path_field} differs")
        digest = str(payload.get(hash_field, "")).lower()
        if (
            not candidate.is_file()
            or _HEX64.fullmatch(digest) is None
            or hash_file(candidate) != digest
        ):
            raise StudyGateError(
                f"Paper completion receipt binding changed: {path_field}"
            )
        return candidate

    results_manifest = verified_file(
        "paper_results_manifest_path",
        "paper_results_manifest_sha256",
        expected_results_manifest,
    )
    claim_ledger = verified_file(
        "final_claim_ledger_path",
        "final_claim_ledger_sha256",
        expected_claim_ledger,
    )
    report_environment_receipt = verified_file(
        "report_environment_receipt_path",
        "report_environment_receipt_sha256",
        expected_report_environment_receipt,
    )
    outer_paths = verify_artifact_hash_manifest(results_manifest)
    required_outer = {
        "paper_artifact_manifest",
        "final_claim_ledger",
        "frozen_protocol",
        "test_access_ledger",
    }
    if missing := sorted(required_outer.difference(outer_paths)):
        raise StudyGateError(
            f"Paper results manifest lacks completion bindings: {missing}"
        )
    if outer_paths["final_claim_ledger"].resolve() != claim_ledger:
        raise StudyGateError("Outer paper manifest names another claim ledger")

    expected_protocol_path = (
        Path(expected_protocol).expanduser().resolve()
        if expected_protocol is not None else outer_paths["frozen_protocol"].resolve()
    )
    expected_test_ledger_path = (
        Path(expected_test_ledger).expanduser().resolve()
        if expected_test_ledger is not None else outer_paths["test_access_ledger"].resolve()
    )
    if outer_paths["frozen_protocol"].resolve() != expected_protocol_path:
        raise StudyGateError("Outer paper manifest names another frozen protocol")
    if outer_paths["test_access_ledger"].resolve() != expected_test_ledger_path:
        raise StudyGateError("Outer paper manifest names another test ledger")
    for artifact, hash_field, label in (
        (expected_protocol_path, "protocol_sha256", "protocol"),
        (expected_test_ledger_path, "test_ledger_sha256", "test ledger"),
    ):
        digest = str(payload.get(hash_field, "")).lower()
        if (
            not artifact.is_file()
            or artifact.is_symlink()
            or _HEX64.fullmatch(digest) is None
            or hash_file(artifact) != digest
        ):
            raise StudyGateError(f"Paper completion receipt {label} binding changed")

    inner_manifest = outer_paths["paper_artifact_manifest"].resolve()
    inner_paths = verify_artifact_hash_manifest(inner_manifest)
    if report_environment_receipt not in {
        artifact.resolve() for artifact in inner_paths.values()
    }:
        raise StudyGateError(
            "Paper artifact manifest lacks the verified report environment receipt"
        )
    try:
        inner_payload = json.loads(inner_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StudyGateError("Paper artifact manifest metadata is unreadable") from exc
    inner_metadata = (
        inner_payload.get("metadata") if isinstance(inner_payload, Mapping) else None
    )
    if (
        not isinstance(inner_metadata, Mapping)
        or inner_metadata.get("report_environment_receipt_sha256")
        != hash_file(report_environment_receipt)
    ):
        raise StudyGateError(
            "Paper artifact manifest report-environment binding changed"
        )
    try:
        outer_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StudyGateError("Paper results manifest metadata is unreadable") from exc
    metadata = outer_payload.get("metadata") if isinstance(outer_payload, Mapping) else None
    if not isinstance(metadata, Mapping):
        raise StudyGateError("Paper results manifest metadata is missing")
    if (
        metadata.get("bundle_status") != "ELIGIBLE_PENDING_COMPLETION_RECEIPT"
        or metadata.get("inner_manifest_sha256") != hash_file(inner_manifest)
        or metadata.get("final_claim_ledger_sha256") != hash_file(claim_ledger)
        or metadata.get("report_environment_receipt_sha256")
        != hash_file(report_environment_receipt)
    ):
        raise StudyGateError("Paper results manifest completion metadata changed")

    ledger = pd.read_csv(claim_ledger, compression="infer")
    required_ledger = {
        "paper_artifact_manifest_sha256", "paper_bundle_status",
        "report_environment_receipt_sha256",
    }
    missing_ledger = required_ledger.difference(ledger.columns)
    if ledger.empty or missing_ledger:
        detail = sorted(missing_ledger) if not ledger.empty else sorted(required_ledger)
        raise StudyGateError(f"Final claim ledger lacks completion columns: {detail}")
    if (
        set(ledger["paper_artifact_manifest_sha256"].astype(str))
        != {hash_file(inner_manifest)}
        or set(ledger["paper_bundle_status"].astype(str))
        != {"ELIGIBLE_PENDING_COMPLETION_RECEIPT"}
        or set(ledger["report_environment_receipt_sha256"].astype(str))
        != {hash_file(report_environment_receipt)}
    ):
        raise StudyGateError("Final claim ledger completion binding changed")
    return {
        "receipt": dict(payload),
        "receipt_path": path,
        "paper_results_manifest": results_manifest,
        "paper_artifact_manifest": inner_manifest,
        "final_claim_ledger": claim_ledger,
        "report_environment_receipt": report_environment_receipt,
        "frozen_protocol": expected_protocol_path,
        "test_access_ledger": expected_test_ledger_path,
    }


def _bundle_from_directory(
    output_dir: Path,
    partition: str,
    test_permit: UntouchedTestPermit | None = None,
) -> PreparedVisDroneBundle:
    return PreparedVisDroneBundle(
        output_dir=output_dir,
        source_manifest=output_dir / "source_manifest.csv.gz",
        split_manifest=output_dir / "publisher_split_manifest.csv.gz",
        source_truth=output_dir / "source_truth.csv.gz",
        analysis_manifest=output_dir / "analysis_manifest.csv.gz",
        trace_manifest=output_dir / "registered_trace_manifest.csv",
        split_audit=output_dir / "publisher_split_audit.csv",
        trace_audit=output_dir / "registered_trace_audit.csv",
        annotation_exclusions=output_dir / ANNOTATION_EXCLUSIONS_FILENAME,
        artifact_manifest=output_dir / "PREPARATION_ARTIFACTS.json",
        receipt=output_dir / "PREPARED_VISDRONE_BUNDLE.json",
        partition=partition,
        test_permit=test_permit,
    )


def _verify_annotation_exclusion_audit(
    audit_path: Path,
    *,
    source_manifest_path: Path,
    receipt: Mapping[str, Any],
) -> pd.DataFrame:
    """Verify the semantic contract of the hash-bound quality audit."""

    try:
        audit = pd.read_csv(audit_path, keep_default_na=False)
        source = pd.read_csv(source_manifest_path, compression="gzip")
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise StudyGateError("Prepared annotation quality audit is unreadable") from exc
    if tuple(audit.columns) != ANNOTATION_EXCLUSION_COLUMNS:
        raise StudyGateError("Prepared annotation quality audit has the wrong schema")
    policy = receipt.get("annotation_quality_policy")
    if not isinstance(policy, Mapping):
        raise StudyGateError("Prepared bundle lacks its annotation quality policy")
    if (
        policy.get("version") != ANNOTATION_QUALITY_POLICY_VERSION
        or policy.get("evaluated_zero_action") != _EXCLUDE_SAMPLE
        or policy.get("ignored_zero_action") != _EXCLUDE_ROW
        or policy.get("negative_or_malformed_action") != "fail_closed"
    ):
        raise StudyGateError("Prepared annotation quality policy changed")
    rows = receipt.get("rows")
    if not isinstance(rows, Mapping):
        raise StudyGateError("Prepared bundle receipt lacks row counts")

    def receipt_count(values: Mapping[str, Any], key: str) -> int:
        value = values.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise StudyGateError(
                f"Prepared bundle receipt has invalid {key!r} count"
            )
        return int(value)

    source_before_count = receipt_count(rows, "source_before_annotation_quality")
    source_count = receipt_count(rows, "source")
    exclusion_count = receipt_count(rows, "annotation_exclusions")
    policy_exclusion_count = receipt_count(policy, "excluded_annotation_rows")
    policy_sample_count = receipt_count(policy, "excluded_source_samples")
    if (
        exclusion_count != len(audit)
        or policy_exclusion_count != len(audit)
        or source_count != len(source)
    ):
        raise StudyGateError("Prepared annotation/source count changed")

    retained_ids = set(source["sample_id"].astype(str))
    if len(retained_ids) != len(source):
        raise StudyGateError("Prepared source manifest repeats a sample id")
    recorded_counts = policy.get("action_counts")
    if not isinstance(recorded_counts, Mapping):
        raise StudyGateError("Prepared annotation action counts are missing")
    if audit.empty:
        if (
            policy_sample_count != 0
            or source_before_count != source_count
            or dict(recorded_counts)
        ):
            raise StudyGateError("Empty annotation audit records excluded source samples")
        return audit
    if audit[["component", "sample_id", "source_line"]].duplicated().any():
        raise StudyGateError("Prepared annotation quality audit repeats a source row")
    if not audit["policy_version"].astype(str).eq(
        ANNOTATION_QUALITY_POLICY_VERSION
    ).all():
        raise StudyGateError("Prepared annotation quality rows use another policy")
    if not audit["component"].astype(str).isin(
        [key for key, spec in VISDRONE_COMPONENTS.items() if spec.task == "DET"]
    ).all():
        raise StudyGateError("Annotation exclusion audit contains a non-DET component")
    if (
        audit["sample_id"].astype(str).eq("").any()
        or audit["source_uid"].astype(str).eq("").any()
        or not audit["raw_row_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all()
    ):
        raise StudyGateError("Annotation exclusion audit has invalid source identity")
    for value in audit["annotation_relative_path"].astype(str):
        relative = PurePosixPath(value)
        if (
            not value
            or relative.is_absolute()
            or relative.as_posix() != value
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise StudyGateError("Annotation exclusion audit has an unsafe source path")
    numeric = audit[[
        "frame_index", "source_line", "bbox_width", "bbox_height",
        "score", "category_id",
    ]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise StudyGateError("Annotation exclusion audit has invalid numeric values")
    widths = numeric["bbox_width"].astype(float)
    heights = numeric["bbox_height"].astype(float)
    if (
        (widths < 0).any()
        or (heights < 0).any()
        or (~((widths == 0.0) | (heights == 0.0))).any()
        or (numeric["frame_index"] < 1).any()
        or (numeric["source_line"] < 1).any()
        or (~numeric["score"].between(0.0, 1.0)).any()
        or (~numeric["category_id"].isin(range(12))).any()
    ):
        raise StudyGateError("Annotation exclusion audit violates the exact-zero policy")
    ignored_flags = audit["publisher_ignored"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    expected_ignored = (
        numeric["score"].astype(float).eq(0.0)
        | numeric["category_id"].astype(int).isin({0, 11})
    )
    if ignored_flags.isna().any() or not ignored_flags.astype(bool).equals(expected_ignored):
        raise StudyGateError("Annotation exclusion audit has inconsistent ignore flags")
    expected_actions = expected_ignored.map(
        {True: _EXCLUDE_ROW, False: _EXCLUDE_SAMPLE}
    )
    expected_reasons = expected_ignored.map(
        {True: _IGNORED_ZERO_REASON, False: _EVALUATED_ZERO_REASON}
    )
    if (
        not audit["action"].astype(str).equals(expected_actions)
        or not audit["reason"].astype(str).equals(expected_reasons)
    ):
        raise StudyGateError("Annotation exclusion audit has inconsistent actions")
    excluded_ids = set(
        audit.loc[audit["action"].astype(str).eq(_EXCLUDE_SAMPLE), "sample_id"].astype(str)
    )
    row_only_ids = set(audit["sample_id"].astype(str)) - excluded_ids
    if excluded_ids & retained_ids or not row_only_ids.issubset(retained_ids):
        raise StudyGateError("Annotation quality actions disagree with retained source rows")
    if (
        policy_sample_count != len(excluded_ids)
        or source_before_count != source_count + len(excluded_ids)
    ):
        raise StudyGateError("Prepared annotation excluded-sample count changed")

    # Row-only exclusions must still bind to the retained publisher source
    # identity. Evaluated-degenerate samples are deliberately absent from the
    # scientific source manifest and remain bound by this audit's artifact hash.
    retained_by_id = source.set_index(source["sample_id"].astype(str), drop=False)
    for record in audit[
        audit["action"].astype(str).eq(_EXCLUDE_ROW)
    ].to_dict("records"):
        source_row = retained_by_id.loc[str(record["sample_id"])]
        relative_annotation = str(record["annotation_relative_path"])
        annotation_path = str(source_row["annotation_path"]).replace("\\", "/")
        if (
            str(source_row["component"]) != str(record["component"])
            or str(source_row["source_uid"]) != str(record["source_uid"])
            or int(source_row["frame_index"]) != int(record["frame_index"])
            or not annotation_path.endswith("/" + relative_annotation)
        ):
            raise StudyGateError(
                "Annotation row exclusion disagrees with retained source identity"
            )
    action_counts = {
        str(key): int(value)
        for key, value in audit["action"].astype(str).value_counts(sort=False).items()
    }
    if dict(recorded_counts) != action_counts:
        raise StudyGateError("Prepared annotation action counts changed")
    return audit


def load_prepared_visdrone_bundle(
    output_dir: str | Path,
    *,
    test_permit: UntouchedTestPermit | None = None,
) -> PreparedVisDroneBundle:
    """Load and verify a previously prepared immutable bundle."""

    destination = Path(output_dir).expanduser().resolve()
    receipt_path = destination / "PREPARED_VISDRONE_BUNDLE.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StudyGateError(f"Prepared VisDrone bundle receipt is invalid: {receipt_path}") from exc
    if receipt.get("workflow_version") != WORKFLOW_VERSION:
        raise StudyGateError("Prepared VisDrone bundle uses another workflow version")
    partition = str(receipt.get("partition", ""))
    if partition == "test":
        _verify_test_permit(test_permit)
    verified_artifacts = verify_artifact_hash_manifest(
        destination / "PREPARATION_ARTIFACTS.json"
    )
    if verified_artifacts.get("publisher_annotation_exclusions") != (
        destination / ANNOTATION_EXCLUSIONS_FILENAME
    ):
        raise StudyGateError("Prepared bundle does not bind its annotation quality audit")
    bundle = _bundle_from_directory(destination, partition, test_permit)
    required_paths = (
        bundle.source_manifest,
        bundle.split_manifest,
        bundle.source_truth,
        bundle.analysis_manifest,
        bundle.trace_manifest,
        bundle.split_audit,
        bundle.trace_audit,
        bundle.annotation_exclusions,
        bundle.artifact_manifest,
        bundle.receipt,
    )
    for artifact in required_paths:
        if not artifact.is_file():
            raise StudyGateError(f"Prepared VisDrone bundle is incomplete: {artifact}")
    _verify_annotation_exclusion_audit(
        bundle.annotation_exclusions,
        source_manifest_path=bundle.source_manifest,
        receipt=receipt,
    )
    return bundle


def prepare_visdrone_bundle(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    benchmark: Mapping[str, Any],
    partition: str = "development",
    seed: int = 20260823,
    calibration_fraction: float = 0.20,
    minimum_test_groups: int = MINIMUM_TEST_NATURAL_GROUPS,
    test_permit: UntouchedTestPermit | None = None,
    require_official_receipt: bool = True,
) -> PreparedVisDroneBundle:
    """Prepare an immutable development or ledger-gated test analysis bundle."""

    def report_progress(
        stage: str,
        completed: int | None = None,
        total: int | None = None,
        **details: Any,
    ) -> None:
        payload: dict[str, Any] = {"stage": str(stage)}
        if completed is not None and total is not None:
            payload.update({
                "completed": int(completed),
                "total": int(total),
                "percent": round(
                    100.0 * int(completed) / max(int(total), 1), 1,
                ),
            })
        payload.update(details)
        print(
            "VISDRONE_PREPARE_PROGRESS " + canonical_json(payload),
            flush=True,
        )

    def report_count(stage: str, completed: int, total: int) -> None:
        report_progress(stage, completed=completed, total=total)

    partition_name = str(partition).strip().lower()
    if partition_name not in {"development", "test"}:
        raise ValueError("partition must be development or test")
    # Authorization precedes even opaque provenance hashing. Although the
    # verifier never interprets annotations, opening or hashing publisher
    # test-dev bytes is still test-source access and must fail closed first.
    if partition_name == "test":
        _verify_test_permit(test_permit)
    source_root = Path(data_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    integrity_components = (
        UNTOUCHED_TEST_COMPONENTS
        if partition_name == "test" else
        DEVELOPMENT_COMPONENTS
    )
    request = {
        "workflow_version": WORKFLOW_VERSION,
        "annotation_quality_policy_version": ANNOTATION_QUALITY_POLICY_VERSION,
        "partition": partition_name,
        "seed": int(seed),
        "calibration_fraction": float(calibration_fraction),
        "minimum_test_groups": int(minimum_test_groups),
        "require_official_receipt": bool(require_official_receipt),
        "benchmark": json.loads(canonical_json(benchmark)),
        "test_ledger_sha256": test_permit.ledger_sha256 if test_permit else "",
    }
    request_hash = hash_object(request)
    report_progress(
        "start",
        partition=partition_name,
        output_dir=str(destination),
    )
    if destination.exists():
        report_progress("existing_bundle_verification_start")
        bundle = load_prepared_visdrone_bundle(destination, test_permit=test_permit)
        receipt = json.loads(bundle.receipt.read_text(encoding="utf-8"))
        if receipt.get("request_hash") != request_hash:
            raise StudyGateError(
                "Existing VisDrone preparation bundle was built from different registered inputs"
            )
        if require_official_receipt:
            report_progress("source_integrity_start")
            current_provenance = verify_visdrone_source_provenance(
                source_root,
                full_integrity=True,
                components=integrity_components,
                progress=report_count,
            )
            frozen_provenance = receipt.get("source_provenance")
            binding_keys = (
                "receipt_sha256",
                "inventory_sha256",
                "source_files",
                "source_bytes",
                "full_integrity",
                "full_integrity_components",
                "scope",
                "components",
            )
            if (
                not isinstance(frozen_provenance, Mapping)
                or canonical_json({
                    key: frozen_provenance.get(key) for key in binding_keys
                }) != canonical_json({
                    key: current_provenance.get(key) for key in binding_keys
                })
            ):
                raise StudyGateError(
                    "Existing VisDrone preparation bundle is bound to different source provenance"
                )
            report_progress("source_integrity_complete")
        report_progress("existing_bundle_verified")
        return bundle

    destination.parent.mkdir(parents=True, exist_ok=True)
    media_audit: Mapping[str, Any] | None = None
    preflight_dir = destination.parent / f"{destination.name}.preflight"
    if require_official_receipt:
        report_progress("media_preflight_start", diagnostic_dir=str(preflight_dir))
        try:
            media_audit = write_visdrone_media_partition_audit(
                source_root,
                preflight_dir,
                seed=int(seed),
                calibration_fraction=float(calibration_fraction),
                minimum_test_groups=int(minimum_test_groups),
                test_permit=test_permit,
                require_test_permit=partition_name == "test",
            )
        except Exception as exc:
            report_progress(
                "media_preflight_failed",
                diagnostic_dir=str(preflight_dir),
                error_type=type(exc).__name__,
            )
            raise
        report_progress(
            "media_preflight_complete",
            diagnostic_dir=str(preflight_dir),
            media_rows=int(media_audit["media_rows"]),
            test_natural_groups=int(media_audit["test_natural_groups"]),
        )

    report_progress("source_integrity_start")
    provenance = (
        verify_visdrone_source_provenance(
            source_root,
            full_integrity=True,
            components=integrity_components,
            progress=report_count,
        )
        if require_official_receipt else
        {"scope": "test_fixture_explicitly_non_scientific", "receipt_sha256": ""}
    )
    report_progress("source_integrity_complete")
    if partition_name == "test":
        _verify_test_permit(test_permit)
        report_progress("source_manifest_start")
        unfiltered_source = build_untouched_test_manifest(
            source_root, test_permit=test_permit,
        )
    else:
        report_progress("source_manifest_start")
        unfiltered_source = build_development_manifest(source_root)
    report_progress("source_manifest_complete", rows=int(len(unfiltered_source)))
    report_progress("annotation_quality_start", rows=int(len(unfiltered_source)))
    source, prepared_objects, annotation_exclusions = _objects_and_annotation_quality(
        unfiltered_source,
        source_root=source_root,
        test_permit=test_permit,
        progress=report_count,
    )
    report_progress(
        "annotation_quality_complete",
        retained_rows=int(len(source)),
        excluded_rows=int(len(annotation_exclusions)),
    )
    report_progress("publisher_split_start")
    split = assign_publisher_splits(
        source,
        calibration_fraction=calibration_fraction,
        seed=seed,
        test_permit=test_permit,
    )
    split_audit = audit_publisher_splits(split)
    if not split_audit["passed"].astype(bool).all():
        raise StudyGateError("Publisher partition/split audit failed")
    report_progress("publisher_split_complete", rows=int(len(split)))
    report_progress("media_dimensions_start", rows=int(len(split)))
    sizes = _image_dimensions(split, progress=report_count)
    report_progress("media_dimensions_complete", rows=int(len(sizes)))
    report_progress("analysis_manifest_start")
    analysis, truth = build_analysis_manifest(
        split,
        image_sizes=sizes,
        benchmark=benchmark,
        test_permit=test_permit,
        prepared_objects_by_sample=prepared_objects,
    )
    report_progress(
        "analysis_manifest_complete",
        analysis_rows=int(len(analysis)),
        truth_rows=int(len(truth)),
    )
    report_progress("trace_manifest_start")
    traces = generate_registered_trace_manifest(seed=seed)
    trace_audit = scientific_trace_design_audit(
        traces, required_test_families=TEST_TRACE_FAMILIES,
    )
    if not trace_audit["passed"].astype(bool).all():
        raise StudyGateError("Registered trace audit failed")
    report_progress("trace_manifest_complete", rows=int(len(traces)))

    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    if staging.exists():  # practically impossible, but never reuse a stale staging directory
        raise StudyGateError(f"Preparation staging path unexpectedly exists: {staging}")
    staging.mkdir()
    try:
        report_progress("bundle_write_start", staging_dir=str(staging))
        if media_audit is not None:
            copied_audit = dict(media_audit)
            for key in ("receipt_path", "media_audit_path", "capacity_audit_path"):
                source_path = Path(str(media_audit[key]))
                target_path = staging / source_path.name
                shutil.copyfile(source_path, target_path)
                copied_audit[key] = target_path
            media_audit = copied_audit
        _deterministic_csv(source, staging / "source_manifest.csv.gz", compress=True)
        _deterministic_csv(split, staging / "publisher_split_manifest.csv.gz", compress=True)
        _deterministic_csv(truth, staging / "source_truth.csv.gz", compress=True)
        _deterministic_csv(
            _serialized_analysis_manifest(analysis),
            staging / "analysis_manifest.csv.gz",
            compress=True,
        )
        _deterministic_csv(traces, staging / "registered_trace_manifest.csv")
        _deterministic_csv(split_audit, staging / "publisher_split_audit.csv")
        _deterministic_csv(trace_audit, staging / "registered_trace_audit.csv")
        annotation_exclusions_path = _deterministic_csv(
            annotation_exclusions,
            staging / ANNOTATION_EXCLUSIONS_FILENAME,
        )
        excluded_source_samples = int(
            annotation_exclusions.loc[
                annotation_exclusions["action"].astype(str).eq(_EXCLUDE_SAMPLE),
                "sample_id",
            ].astype(str).nunique()
            if not annotation_exclusions.empty else 0
        )
        action_counts = {
            str(key): int(value)
            for key, value in annotation_exclusions["action"].astype(str)
            .value_counts(sort=False).items()
        }
        annotation_quality_policy = {
            "version": ANNOTATION_QUALITY_POLICY_VERSION,
            "ignored_zero_action": _EXCLUDE_ROW,
            "evaluated_zero_action": _EXCLUDE_SAMPLE,
            "negative_or_malformed_action": "fail_closed",
            "excluded_annotation_rows": int(len(annotation_exclusions)),
            "excluded_source_samples": excluded_source_samples,
            "action_counts": action_counts,
        }
        artifacts = {
            "source_manifest": staging / "source_manifest.csv.gz",
            "publisher_split_manifest": staging / "publisher_split_manifest.csv.gz",
            "source_truth": staging / "source_truth.csv.gz",
            "analysis_manifest": staging / "analysis_manifest.csv.gz",
            "registered_trace_manifest": staging / "registered_trace_manifest.csv",
            "publisher_split_audit": staging / "publisher_split_audit.csv",
            "registered_trace_audit": staging / "registered_trace_audit.csv",
            "publisher_annotation_exclusions": annotation_exclusions_path,
        }
        if media_audit is not None:
            artifacts.update({
                "media_partition_hash_audit": Path(media_audit["media_audit_path"]),
                "test_group_capacity_audit": Path(media_audit["capacity_audit_path"]),
                "media_partition_audit_receipt": Path(media_audit["receipt_path"]),
            })
        write_artifact_hash_manifest(
            artifacts,
            staging / "PREPARATION_ARTIFACTS.json",
            metadata={
                "partition": partition_name,
                "request_hash": request_hash,
                "source_provenance": provenance,
                "annotation_quality_policy": annotation_quality_policy,
            },
            root=staging,
        )
        _atomic_json(staging / "PREPARED_VISDRONE_BUNDLE.json", {
            "schema_version": 1,
            "workflow_version": WORKFLOW_VERSION,
            "partition": partition_name,
            "request_hash": request_hash,
            "source_provenance": provenance,
            "annotation_quality_policy": annotation_quality_policy,
            "rows": {
                "source_before_annotation_quality": int(len(unfiltered_source)),
                "source": int(len(source)),
                "annotation_exclusions": int(len(annotation_exclusions)),
                "analysis": int(len(analysis)),
                "source_truth": int(len(truth)),
                "trace_segments": int(len(traces)),
            },
            "claim_scope": "VisDrone publisher-box/track sensor-space endpoint only",
            "prohibited_claims": list(PROHIBITED_CLAIMS),
        })
        os.replace(staging, destination)
        report_progress("bundle_complete", output_dir=str(destination))
        if require_official_receipt:
            shutil.rmtree(preflight_dir, ignore_errors=True)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_prepared_visdrone_bundle(destination, test_permit=test_permit)


def load_visdrone_results(
    run_root: str | Path,
    *,
    artifact_manifest: str = "PAPER_RESULTS_MANIFEST.json",
) -> Mapping[str, Any]:
    """Load only hash-verified CSV/CSV.GZ/JSON paper result artifacts."""

    root = Path(run_root).expanduser().resolve()
    paths = verify_artifact_hash_manifest(root / artifact_manifest)
    loaded: dict[str, Any] = {}
    for name, path in paths.items():
        suffixes = path.suffixes
        if suffixes[-2:] == [".csv", ".gz"]:
            loaded[name] = pd.read_csv(path, compression="gzip")
        elif path.suffix == ".csv":
            loaded[name] = pd.read_csv(path)
        elif path.suffix == ".json":
            loaded[name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            loaded[name] = path
    return loaded


__all__ = [
    "ANNOTATION_EXCLUSIONS_FILENAME",
    "ANNOTATION_QUALITY_POLICY_VERSION",
    "MEDIA_PARTITION_AUDIT_SCHEMA",
    "MINIMUM_TEST_NATURAL_GROUPS",
    "PAPER_COMPLETION_RECEIPT_SCHEMA",
    "PROHIBITED_CLAIMS",
    "PreparedVisDroneBundle",
    "REGISTERED_TEMPORAL_TRACE_DURATION_US",
    "REGISTERED_TEMPORAL_TRACE_DYNAMIC_US",
    "REGISTERED_TEMPORAL_TRACE_SEGMENT_US",
    "SOURCE_PROVENANCE_SCOPE",
    "TEST_TRACE_FAMILIES",
    "TEMPORAL_COMPONENT_BY_SPLIT",
    "TEMPORAL_SAMPLING_METHOD",
    "TEMPORAL_SAMPLING_SCHEMA",
    "TRUTH_SCOPE",
    "VALIDATION_TRACE_FAMILIES",
    "WORKFLOW_VERSION",
    "assign_publisher_splits",
    "audit_registered_temporal_trace_alignment",
    "audit_publisher_splits",
    "build_analysis_manifest",
    "build_readiness_claim_ledger",
    "export_paper_tables",
    "generate_registered_trace_manifest",
    "load_prepared_visdrone_bundle",
    "load_visdrone_results",
    "prepare_visdrone_bundle",
    "read_analysis_manifest",
    "registered_temporal_sampling_spec",
    "registered_mission_policies",
    "registered_analysis_sampling_limits",
    "select_registered_analysis_sample",
    "select_registered_analysis_sample_from_config",
    "select_registered_temporal_sample",
    "select_registered_temporal_sample_from_config",
    "verify_artifact_hash_manifest",
    "verify_paper_completion_receipt",
    "verify_visdrone_media_partition_audit",
    "verify_visdrone_source_provenance",
    "write_artifact_hash_manifest",
    "write_registered_analysis_sample",
    "write_registered_temporal_sample",
    "write_visdrone_media_partition_audit",
]
