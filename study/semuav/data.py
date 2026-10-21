"""Dataset adapters, canonical manifests, independent splits, and label gates.

The module does not download data and never guesses scientific labels.  It
normalizes existing source files into one manifest, keeps every related
flight/video/location/event in one split, and refuses full-study execution when
real-data, blinding, agreement, adjudication, or frozen-test gates are unmet.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import AnnotationConfig, DatasetSpec, Profile, SplitConfig, StudyConfig
from .protocol import MISSION_ONTOLOGY


class DataGateError(RuntimeError):
    """Raised when a data-integrity or scientific-validity gate fails."""


CANONICAL_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "dataset",
    "mission",
    "media_path",
    "media_type",
    "annotation_path",
    "source_uid",
    "split_group",
    "flight_id",
    "sequence_id",
    "location_id",
    "event_id",
    "frame_index",
    "timestamp_ms",
    "width",
    "height",
    "has_ground_truth",
    "critical_event",
    "is_synthetic",
    "metadata_json",
)

STRING_COLUMNS = (
    "sample_id",
    "dataset",
    "mission",
    "media_path",
    "media_type",
    "annotation_path",
    "source_uid",
    "split_group",
    "flight_id",
    "sequence_id",
    "location_id",
    "event_id",
    "metadata_json",
)
BOOLEAN_COLUMNS = ("has_ground_truth", "critical_event", "is_synthetic")
INTEGER_COLUMNS = ("frame_index", "width", "height")
NATURAL_GROUP_COLUMNS: tuple[str, ...] = (
    "flight_id",
    "sequence_id",
    "location_id",
    "event_id",
)
ANNOTATION_TASKS: tuple[str, ...] = (
    "gold_action",
    "critical",
    "cohort",
    "sufficient_evidence",
    "event",
    "objects",
    "relations",
    "evidence_roi",
)
TELEMETRY_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "capture_time_us",
    "gps_lat",
    "gps_lon",
    "altitude_m",
    "yaw_deg",
    "mission_profile",
    "deadline_us",
    "trace_time_us",
    "battery_fraction",
    "thermal_c",
)
PREPARATION_BUNDLE_FILENAMES: Mapping[str, str] = {
    "canonical_manifest": "canonical_manifest.csv.gz",
    "proposed_split_manifest": "proposed_split_manifest.csv.gz",
    "annotation_assignment_skeleton": "annotation_assignment_skeleton.csv",
    "telemetry_skeleton": "telemetry_skeleton.csv",
    "bundle_manifest": "PREPARATION_BUNDLE.json",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXCLUDED_IMAGE_PARENTS = {
    "annotation",
    "annotations",
    "gt",
    "label",
    "labels",
    "mask",
    "masks",
    "segmentation",
    "segmentations",
}

_GROUP_PLACEHOLDERS = {
    "na",
    "n/a",
    "none",
    "null",
    "unknown",
    "unset",
    "missing",
    "nan",
    "<na>",
    "placeholder",
    "required",
    "tbd",
    "todo",
}


@dataclasses.dataclass(frozen=True)
class AuditReport:
    name: str
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    statistics: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            details = "\n".join(f"  - {message}" for message in self.errors)
            raise DataGateError(f"{self.name} failed:\n{details}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "statistics": dict(self.statistics),
        }


@dataclasses.dataclass(frozen=True)
class AnnotationGateReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    n_rows: int
    n_items: int
    n_disagreements: int
    n_adjudicated_disagreements: int
    observed_agreement: float
    chance_corrected_agreement: float
    manifest_coverage: float
    by_task: Mapping[str, Mapping[str, float | int]]

    @property
    def passed(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            details = "\n".join(f"  - {message}" for message in self.errors)
            raise DataGateError(f"annotation gate failed:\n{details}")

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self) | {"passed": self.passed}


def _stable_id(*parts: Any, length: int = 24) -> str:
    text = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", value))


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)):
            return bool(missing)
    except (TypeError, ValueError):
        pass
    return False


def _meaningful_group_values(series: pd.Series) -> pd.Series:
    """Identify explicit group IDs while rejecting common placeholder values."""

    values = series.fillna("").astype(str).str.strip()
    lowered = values.str.lower()
    return values.ne("") & ~lowered.isin(_GROUP_PLACEHOLDERS) & ~lowered.str.startswith("required_")


def _natural_group_coverage(data: pd.DataFrame) -> pd.Series:
    coverage = pd.Series(False, index=data.index, dtype=bool)
    for column in NATURAL_GROUP_COLUMNS:
        if column in data:
            coverage |= _meaningful_group_values(data[column])
    return coverage


def _sha256_file(path: Path, *, chunk_size_bytes: int = 8 * 1024 * 1024) -> str:
    if chunk_size_bytes <= 0:
        raise ValueError("chunk_size_bytes must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _coerce_bool(series: pd.Series, *, default: bool = False) -> pd.Series:
    truthy = {"1", "true", "t", "yes", "y"}
    falsey = {"0", "false", "f", "no", "n", ""}

    def convert(value: Any) -> bool:
        if value is None or pd.isna(value):
            return default
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return bool(value)
        text = str(value).strip().lower()
        if text in truthy:
            return True
        if text in falsey:
            return False
        raise DataGateError(f"Cannot interpret {value!r} as Boolean")

    return series.map(convert).astype("boolean")


def _canonical_json(value: Any) -> str:
    if _is_blank(value):
        return "{}"
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = {"source_metadata": value}
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
    )


def canonicalize_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Return an ordered, typed, deterministic canonical manifest."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("manifest must be a pandas DataFrame")
    data = frame.copy()
    defaults: Mapping[str, Any] = {
        "sample_id": "",
        "dataset": "",
        "mission": "",
        "media_path": "",
        "media_type": "image",
        "annotation_path": "",
        "source_uid": "",
        "split_group": "",
        "flight_id": "",
        "sequence_id": "",
        "location_id": "",
        "event_id": "",
        "frame_index": pd.NA,
        "timestamp_ms": np.nan,
        "width": pd.NA,
        "height": pd.NA,
        "has_ground_truth": False,
        "critical_event": False,
        "is_synthetic": False,
        "metadata_json": "{}",
    }
    for column, default in defaults.items():
        if column not in data:
            data[column] = default

    for column in STRING_COLUMNS:
        data[column] = data[column].fillna("").astype(str).str.strip()
    for column in BOOLEAN_COLUMNS:
        data[column] = _coerce_bool(data[column])
    for column in INTEGER_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype("Int64")
    data["timestamp_ms"] = pd.to_numeric(data["timestamp_ms"], errors="coerce").astype(float)
    data["metadata_json"] = data["metadata_json"].map(_canonical_json)

    missing_source = data["source_uid"].eq("")
    data.loc[missing_source, "source_uid"] = data.loc[missing_source, "media_path"]
    missing_id = data["sample_id"].eq("")
    data.loc[missing_id, "sample_id"] = [
        _stable_id(dataset, source)
        for dataset, source in data.loc[missing_id, ["dataset", "source_uid"]].itertuples(index=False)
    ]
    identifier_order = ("event_id", "location_id", "flight_id", "sequence_id", "source_uid")
    missing_group = data["split_group"].eq("")
    for index in data.index[missing_group]:
        row = data.loc[index]
        chosen = next((str(row[key]) for key in identifier_order if str(row[key]).strip()), str(row["sample_id"]))
        data.at[index, "split_group"] = f"{row['dataset']}:{chosen}"

    extras = sorted(set(data.columns) - set(CANONICAL_COLUMNS))
    return data.loc[:, list(CANONICAL_COLUMNS) + extras].sort_values(
        ["dataset", "source_uid", "sample_id"], kind="stable"
    ).reset_index(drop=True)


def _discover_images(root: Path, patterns: Sequence[str] | None = None) -> list[Path]:
    if patterns:
        paths = {path for pattern in patterns for path in root.glob(pattern) if path.is_file()}
    else:
        paths = {path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
    return sorted(
        (
            path
            for path in paths
            if path.suffix.lower() in IMAGE_EXTENSIONS
            and not any(part.lower() in EXCLUDED_IMAGE_PARENTS for part in path.relative_to(root).parts[:-1])
        ),
        key=lambda path: _natural_key(path.relative_to(root).as_posix()),
    )


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def _candidate_annotation(image: Path, directory_names: Sequence[str], suffixes: Sequence[str]) -> Path | None:
    candidates: list[Path] = []
    for ancestor in (image.parent, *image.parents):
        if ancestor.name.lower() in {"images", "image", "rgb", "frames", "sequences"}:
            for directory in directory_names:
                candidates.extend((ancestor.parent / directory / image.name).with_suffix(suffix) for suffix in suffixes)
            break
    for directory in directory_names:
        candidates.extend((image.parent.parent / directory / image.name).with_suffix(suffix) for suffix in suffixes)
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


def _base_record(
    spec: DatasetSpec,
    image: Path,
    *,
    annotation: Path | None = None,
    sequence_id: str = "",
    flight_id: str = "",
    location_id: str = "",
    event_id: str = "",
    frame_index: int | None = None,
    critical_event: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_uid = _relative_or_absolute(image, spec.root)
    group = event_id or location_id or flight_id or sequence_id or source_uid
    return {
        "dataset": spec.name,
        "mission": spec.mission,
        "media_path": str(image.resolve()),
        "media_type": "image",
        "annotation_path": str(annotation) if annotation else "",
        "source_uid": source_uid,
        "split_group": f"{spec.name}:{group}",
        "flight_id": flight_id,
        "sequence_id": sequence_id,
        "location_id": location_id,
        "event_id": event_id,
        "frame_index": frame_index if frame_index is not None else pd.NA,
        "has_ground_truth": bool(annotation),
        "critical_event": critical_event,
        "is_synthetic": spec.synthetic,
        "metadata_json": metadata or {},
    }


class DatasetAdapter:
    """Base class for read-only filesystem-to-manifest adapters."""

    def __init__(self, spec: DatasetSpec):
        self.spec = spec

    def build(self) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def patterns(self) -> tuple[str, ...]:
        value = self.spec.options.get("image_globs", ())
        if isinstance(value, str):
            return (value,)
        return tuple(str(item) for item in value)


class SyntheticAdapter(DatasetAdapter):
    """Generated rows for integration tests only; never creates scientific data."""

    def build(self) -> pd.DataFrame:
        count = int(self.spec.options.get("n_samples", 120))
        group_size = max(1, int(self.spec.options.get("group_size", 6)))
        critical_fraction = float(self.spec.options.get("critical_fraction", 0.20))
        records = []
        for index in range(count):
            group = index // group_size
            critical = (group % max(1, round(1.0 / max(critical_fraction, 1e-9)))) == 0
            virtual = self.spec.root / f"sequence_{group:04d}" / f"frame_{index:06d}.jpg"
            records.append(
                _base_record(
                    self.spec,
                    virtual,
                    sequence_id=f"sequence_{group:04d}",
                    event_id=f"event_{group:04d}" if critical else "",
                    frame_index=index % group_size,
                    critical_event=critical,
                    metadata={"fixture": True, "never_publish": True},
                )
            )
        return canonicalize_manifest(pd.DataFrame.from_records(records))


class VisDroneAdapter(DatasetAdapter):
    """Adapter for VisDrone DET and VID directory layouts."""

    def build(self) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for image in _discover_images(self.spec.root, self.patterns or None):
            relative = image.relative_to(self.spec.root)
            lower_parts = [part.lower() for part in relative.parts]
            sequence = ""
            if "sequences" in lower_parts:
                position = lower_parts.index("sequences")
                if position + 1 < len(relative.parts) - 1:
                    sequence = relative.parts[position + 1]
            if not sequence:
                sequence = image.stem.split("_")[0]
            match = re.search(r"(\d+)$", image.stem)
            frame_index = int(match.group(1)) if match else None
            annotation = _candidate_annotation(image, ("annotations", "Annotations"), (".txt", ".json"))
            if annotation is None and "sequences" in lower_parts:
                position = lower_parts.index("sequences")
                sequence_root = Path(*image.parts[: len(image.parts) - len(relative.parts) + position])
                candidates = (
                    sequence_root / "annotations" / f"{sequence}.txt",
                    sequence_root / "Annotations" / f"{sequence}.txt",
                )
                annotation = next((path.resolve() for path in candidates if path.is_file()), None)
            records.append(
                _base_record(
                    self.spec,
                    image,
                    annotation=annotation,
                    flight_id=sequence,
                    sequence_id=sequence,
                    frame_index=frame_index,
                    metadata={"layout": "VID" if "sequences" in lower_parts else "DET"},
                )
            )
        return canonicalize_manifest(pd.DataFrame.from_records(records))


class UAVidAdapter(DatasetAdapter):
    """Adapter for UAVid ``seq*/Images`` and ``seq*/Labels`` layouts."""

    def build(self) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for image in _discover_images(self.spec.root, self.patterns or None):
            relative = image.relative_to(self.spec.root)
            sequence = next(
                (part for part in reversed(relative.parts[:-1]) if re.fullmatch(r"(?i)(?:seq|sequence)[-_]?\d+", part)),
                image.parent.parent.name if image.parent.name.lower() in {"images", "rgb", "frames"} else image.parent.name,
            )
            match = re.search(r"(\d+)$", image.stem)
            frame_index = int(match.group(1)) if match else None
            annotation = _candidate_annotation(
                image, ("Labels", "labels", "Masks", "masks", "annotations"), (".png", ".tif", ".json")
            )
            records.append(
                _base_record(
                    self.spec,
                    image,
                    annotation=annotation,
                    flight_id=sequence,
                    sequence_id=sequence,
                    frame_index=frame_index,
                )
            )
        return canonicalize_manifest(pd.DataFrame.from_records(records))


def _read_source_manifest(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path)
    if suffixes.endswith(".jsonl") or suffixes.endswith(".ndjson"):
        return pd.read_json(path, lines=True)
    if suffixes.endswith(".json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            for key in ("samples", "images", "records", "data"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise DataGateError(f"Unsupported JSON manifest structure: {path}")
        return pd.DataFrame.from_records(payload)
    raise DataGateError(f"Unsupported source manifest format: {path}")


def _first_present(frame: pd.DataFrame, aliases: Sequence[str], default: Any = "") -> pd.Series:
    for name in aliases:
        if name in frame:
            return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def _normalize_source_manifest(spec: DatasetSpec, source: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    media = _first_present(source, ("media_path", "image_path", "image", "path", "file_name"))
    annotation = _first_present(source, ("annotation_path", "annotation", "label_path", "mask_path"))

    def resolve(value: Any) -> str:
        if _is_blank(value):
            return ""
        path = Path(str(value))
        if not path.is_absolute():
            path = (source_path.parent / path).resolve()
        return str(path)

    normalized = pd.DataFrame(index=source.index)
    normalized["dataset"] = spec.name
    normalized["mission"] = spec.mission
    normalized["media_path"] = media.map(resolve)
    normalized["annotation_path"] = annotation.map(resolve)
    embedded_label_columns = [
        column
        for column in ("label", "labels", "answer", "answers", "reasoning", "class", "target")
        if column in source
    ]
    embedded_ground_truth = pd.Series(False, index=source.index)
    for column in embedded_label_columns:
        embedded_ground_truth |= source[column].map(lambda value: not _is_blank(value))
    normalized.loc[
        normalized["annotation_path"].eq("") & embedded_ground_truth, "annotation_path"
    ] = str(source_path.resolve())
    normalized["source_uid"] = _first_present(source, ("source_uid", "id", "sample_id", "uid"))
    blank_source = normalized["source_uid"].fillna("").astype(str).str.strip().eq("")
    normalized.loc[blank_source, "source_uid"] = normalized.loc[blank_source, "media_path"].map(
        lambda value: _relative_or_absolute(Path(value), spec.root) if value else ""
    )
    normalized["sample_id"] = _first_present(source, ("sample_id",))
    normalized["split_group"] = _first_present(source, ("split_group", "group_id"))
    normalized["flight_id"] = _first_present(source, ("flight_id", "flight"))
    normalized["sequence_id"] = _first_present(source, ("sequence_id", "video_id", "sequence"))
    normalized["location_id"] = _first_present(source, ("location_id", "site_id", "location"))
    normalized["event_id"] = _first_present(source, ("event_id", "incident_id", "event"))
    normalized["frame_index"] = _first_present(source, ("frame_index", "frame_id", "frame"), pd.NA)
    normalized["timestamp_ms"] = _first_present(source, ("timestamp_ms", "time_ms"), np.nan)
    normalized["width"] = _first_present(source, ("width",), pd.NA)
    normalized["height"] = _first_present(source, ("height",), pd.NA)
    path_ground_truth = normalized["annotation_path"].map(lambda value: bool(value and Path(value).is_file()))
    if "has_ground_truth" in source:
        explicit_ground_truth = _coerce_bool(source["has_ground_truth"])
    else:
        explicit_ground_truth = pd.Series(False, index=source.index, dtype="boolean")
    normalized["has_ground_truth"] = path_ground_truth | embedded_ground_truth | explicit_ground_truth
    normalized["critical_event"] = _first_present(source, ("critical_event", "is_critical", "hazard"), False)
    normalized["is_synthetic"] = spec.synthetic
    normalized["metadata_json"] = [
        {key: value for key, value in row.items() if key not in normalized.columns}
        for row in source.to_dict(orient="records")
    ]
    return canonicalize_manifest(normalized)


class UAVReasonAdapter(DatasetAdapter):
    """UAVReason adapter; prefers an explicit CSV/JSON/JSONL sample manifest."""

    def build(self) -> pd.DataFrame:
        configured = self.spec.options.get("manifest")
        candidates = [self.spec.root / str(configured)] if configured else []
        candidates.extend(
            self.spec.root / name
            for name in ("manifest.csv", "manifest.jsonl", "metadata.jsonl", "annotations.json")
        )
        source_path = next((path for path in candidates if path.is_file()), None)
        if source_path:
            return _normalize_source_manifest(self.spec, _read_source_manifest(source_path), source_path)
        return _generic_image_manifest(self.spec, annotation_suffixes=(".json", ".jsonl", ".txt"))


class Flame3Adapter(DatasetAdapter):
    """Generic FLAME 3 adapter supporting class folders or an explicit manifest."""

    def build(self) -> pd.DataFrame:
        configured = self.spec.options.get("manifest")
        if configured:
            source_path = self.spec.root / str(configured)
            if source_path.is_file():
                return _normalize_source_manifest(self.spec, _read_source_manifest(source_path), source_path)
        frame = _generic_image_manifest(self.spec, annotation_suffixes=(".json", ".txt", ".png"))
        if frame.empty:
            return frame
        keywords = tuple(str(value).lower() for value in self.spec.options.get("critical_keywords", ["fire", "flame", "smoke"]))
        inferred = frame["source_uid"].str.lower().map(lambda value: any(keyword in value for keyword in keywords))
        frame.loc[:, "critical_event"] = inferred.astype("boolean")
        event_depth = max(1, int(self.spec.options.get("event_parent_depth", 1)))
        for index, path_text in frame["media_path"].items():
            path = Path(path_text)
            ancestor = path.parent
            for _ in range(event_depth - 1):
                ancestor = ancestor.parent
            event = ancestor.name
            frame.at[index, "event_id"] = event
            frame.at[index, "split_group"] = f"{self.spec.name}:{event}"
        return canonicalize_manifest(frame)


def _generic_image_manifest(spec: DatasetSpec, annotation_suffixes: Sequence[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    configured_patterns = spec.options.get("image_globs", ())
    patterns = (configured_patterns,) if isinstance(configured_patterns, str) else tuple(configured_patterns)
    for image in _discover_images(spec.root, patterns or None):
        sequence = image.parent.name
        annotation = _candidate_annotation(
            image, ("annotations", "Annotations", "labels", "Labels"), annotation_suffixes
        )
        records.append(
            _base_record(
                spec,
                image,
                annotation=annotation,
                sequence_id=sequence,
                event_id=sequence if spec.mission == "wildfire_emergency" else "",
            )
        )
    return canonicalize_manifest(pd.DataFrame.from_records(records))


ADAPTERS: Mapping[str, type[DatasetAdapter]] = {
    "synthetic": SyntheticAdapter,
    "visdrone": VisDroneAdapter,
    "uavid": UAVidAdapter,
    "uavreason": UAVReasonAdapter,
    "flame3": Flame3Adapter,
    "generic_flame3": Flame3Adapter,
}


def build_dataset_manifest(spec: DatasetSpec) -> pd.DataFrame:
    if spec.adapter not in ADAPTERS:
        raise DataGateError(f"No adapter registered for {spec.adapter!r}")
    if spec.adapter != "synthetic" and not spec.root.is_dir():
        raise DataGateError(f"Dataset root does not exist for {spec.name}: {spec.root}")
    return ADAPTERS[spec.adapter](spec).build()


def build_canonical_manifest(config: StudyConfig, *, enforce_prerequisites: bool = True) -> pd.DataFrame:
    """Build and audit the combined manifest for all enabled datasets."""

    if enforce_prerequisites:
        config.assert_ready("ingest")
    pieces = [build_dataset_manifest(spec) for spec in config.enabled_datasets]
    manifest = canonicalize_manifest(pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame())
    report = audit_manifest(manifest, config)
    report.raise_for_errors()
    return manifest


def audit_manifest(manifest: pd.DataFrame, config: StudyConfig | None = None) -> AuditReport:
    data = canonicalize_manifest(manifest)
    errors: list[str] = []
    warnings: list[str] = []
    if data.empty:
        errors.append("manifest contains no samples")
    for column in ("sample_id", "dataset", "mission", "media_path", "source_uid", "split_group"):
        count = int(data[column].eq("").sum())
        if count:
            errors.append(f"{count} row(s) have an empty {column}")
    duplicate_ids = data.loc[data["sample_id"].duplicated(keep=False), "sample_id"].unique()
    if len(duplicate_ids):
        errors.append(f"duplicate sample_id values ({len(duplicate_ids)} unique duplicates)")
    duplicate_source = data.duplicated(["dataset", "source_uid"], keep=False)
    if duplicate_source.any():
        errors.append(f"{int(duplicate_source.sum())} rows duplicate dataset/source_uid")
    real = ~data["is_synthetic"].astype(bool)
    natural_group_coverage = _natural_group_coverage(data)
    missing_natural_groups = real & ~natural_group_coverage
    if missing_natural_groups.any():
        message = (
            f"{int(missing_natural_groups.sum())} real row(s) lack a meaningful natural grouping ID; "
            f"populate at least one of {list(NATURAL_GROUP_COLUMNS)} with a flight, sequence, "
            "location, or event identity (source/sample fallbacks are not leakage-safe)"
        )
        if config is not None and config.profile in {Profile.FULL, Profile.SMOKE_REAL}:
            errors.append(message)
        else:
            warnings.append(message)
    missing_media = [value for value in data.loc[real, "media_path"].unique() if not Path(value).is_file()]
    if missing_media:
        errors.append(f"{len(missing_media)} real media file(s) are missing; first: {missing_media[0]}")
    missing_labels = data.loc[real & data["has_ground_truth"].astype(bool), "annotation_path"].map(
        lambda value: not value or not Path(value).is_file()
    )
    if missing_labels.any():
        errors.append(f"{int(missing_labels.sum())} claimed ground-truth file(s) are missing")
    if config:
        present = set(data["dataset"])
        required = {spec.name for spec in config.enabled_datasets if spec.required}
        missing = sorted(required - present)
        if missing:
            errors.append(f"required datasets have no rows: {missing}")
        if config.profile is Profile.FULL:
            synthetic_count = int(data["is_synthetic"].sum())
            if synthetic_count:
                errors.append(f"full profile contains {synthetic_count} synthetic row(s)")
            unlabelled = int((~data["has_ground_truth"].astype(bool)).sum())
            if unlabelled:
                errors.append(f"full profile contains {unlabelled} sample(s) without source ground truth")
        elif config.profile is Profile.DRY_RUN and not data["is_synthetic"].all():
            warnings.append("dry_run manifest includes real files; outputs remain non-scientific")
    statistics = {
        "rows": len(data),
        "datasets": data["dataset"].value_counts().sort_index().to_dict(),
        "split_groups": int(data["split_group"].nunique()),
        "critical_rows": int(data["critical_event"].sum()),
        "synthetic_rows": int(data["is_synthetic"].sum()),
        "real_rows_with_natural_group_id": int((real & natural_group_coverage).sum()),
        "real_rows_without_natural_group_id": int(missing_natural_groups.sum()),
    }
    return AuditReport("canonical manifest audit", tuple(errors), tuple(warnings), statistics)


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _independence_units(manifest: pd.DataFrame, leakage_keys: Sequence[str]) -> pd.Series:
    data = manifest.reset_index(drop=True)
    union = _UnionFind(len(data))
    for key in leakage_keys:
        if key not in data:
            raise DataGateError(f"Leakage key {key!r} is absent from the manifest")
        values = data[key].fillna("").astype(str).str.strip()
        eligible = data.loc[values.ne(""), ["dataset"]].copy()
        eligible["value"] = values[values.ne("")]
        # ``GroupBy.groups`` preserves the original row labels; ``indices`` is
        # positional within the filtered frame and would join the wrong rows.
        for indices in eligible.groupby(["dataset", "value"], sort=False).groups.values():
            if len(indices) > 1:
                first = int(indices[0])
                for other in indices[1:]:
                    union.union(first, int(other))
    members: dict[int, list[str]] = {}
    for index, sample_id in enumerate(data["sample_id"].astype(str)):
        members.setdefault(union.find(index), []).append(sample_id)
    unit_by_root = {root: _stable_id(*sorted(sample_ids), length=20) for root, sample_ids in members.items()}
    return pd.Series([unit_by_root[union.find(index)] for index in range(len(data))], index=data.index, dtype="string")


def _allocation_counts(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    exact = {name: float(ratio) * total for name, ratio in ratios.items()}
    counts = {name: math.floor(value) for name, value in exact.items()}
    remainder = total - sum(counts.values())
    order = sorted(ratios, key=lambda name: (-(exact[name] - counts[name]), name))
    for name in order[:remainder]:
        counts[name] += 1
    positive = [name for name, ratio in ratios.items() if ratio > 0]
    if total >= len(positive):
        for empty in [name for name in positive if counts[name] == 0]:
            donor = max(positive, key=lambda name: counts[name])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[empty] += 1
    return counts


def assign_group_safe_splits(manifest: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    """Assign deterministic splits to connected independent units.

    Rows sharing *any* configured flight/video/location/event identity are joined
    into one connected component.  This is stricter than grouping on a single
    column and protects against transitive leakage (flight A at location X,
    flight B at the same location, and event Y spanning both flights).
    """

    config.validate()
    data = canonicalize_manifest(manifest)
    if data.empty:
        raise DataGateError("Cannot split an empty manifest")
    data["split_unit"] = _independence_units(data, config.leakage_keys)
    available_strata = [column for column in config.strata if column in data]
    unit_rows: list[dict[str, Any]] = []
    for unit, group in data.groupby("split_unit", sort=False):
        row: dict[str, Any] = {"split_unit": unit}
        for column in available_strata:
            values = sorted({str(value) for value in group[column].fillna("")})
            row[column] = "|".join(values)
        unit_rows.append(row)
    units = pd.DataFrame.from_records(unit_rows)
    units["stratum"] = (
        units[available_strata].astype(str).agg("\x1e".join, axis=1) if available_strata else "all"
    )
    assignment: dict[str, str] = {}
    for stratum, group in units.groupby("stratum", sort=True):
        ordered = sorted(
            group["split_unit"].astype(str),
            key=lambda unit: _stable_id(config.seed, stratum, unit, length=64),
        )
        counts = _allocation_counts(len(ordered), config.ratios)
        offset = 0
        for split_name in config.ratios:
            for unit in ordered[offset : offset + counts[split_name]]:
                assignment[unit] = split_name
            offset += counts[split_name]
    data["split"] = data["split_unit"].map(assignment).astype("string")
    report = audit_split_leakage(data, config)
    report.raise_for_errors()
    return data


def find_cross_split_content_duplicates(
    manifest: pd.DataFrame,
    *,
    chunk_size_bytes: int = 8 * 1024 * 1024,
) -> pd.DataFrame:
    """Return exact real-media duplicates that occur in more than one split.

    File size is used as a lossless candidate filter: only sizes represented in
    multiple splits are hashed, and each resolved path is hashed at most once.
    This keeps the audit exact without eagerly reading every byte in a large
    corpus.
    """

    if chunk_size_bytes <= 0:
        raise ValueError("chunk_size_bytes must be positive")
    required = {"sample_id", "media_path", "split"}
    if missing := required.difference(manifest.columns):
        raise DataGateError(f"Content-duplicate audit is missing columns: {sorted(missing)}")
    data = canonicalize_manifest(manifest)
    real = data.loc[~data["is_synthetic"].astype(bool), ["sample_id", "media_path", "split"]]
    path_records: list[dict[str, Any]] = []
    for row in real.itertuples(index=False):
        value = str(row.media_path).strip()
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            # Missing-file diagnostics belong to the canonical manifest audit;
            # do not obscure them with a secondary hashing failure.
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise DataGateError(f"Cannot stat media for content-duplicate audit: {path} ({exc})") from exc
        path_records.append({
            "sample_id": str(row.sample_id),
            "split": str(row.split),
            "media_path": str(path),
            "bytes": int(size),
        })
    columns = ("sha256", "bytes", "splits", "media_paths", "sample_ids")
    if not path_records:
        result = pd.DataFrame(columns=columns)
        result.attrs["candidate_files_hashed"] = 0
        return result
    paths = pd.DataFrame.from_records(path_records)
    candidate_sizes = paths.groupby("bytes", sort=False)["split"].nunique().loc[lambda values: values > 1].index
    candidates = paths.loc[paths["bytes"].isin(candidate_sizes)].copy()
    unique_paths = sorted(candidates["media_path"].unique())
    hashes: dict[str, str] = {}
    for value in unique_paths:
        path = Path(value)
        try:
            hashes[value] = _sha256_file(path, chunk_size_bytes=chunk_size_bytes)
        except OSError as exc:
            raise DataGateError(f"Cannot hash media for content-duplicate audit: {path} ({exc})") from exc
    candidates["sha256"] = candidates["media_path"].map(hashes)
    duplicate_rows: list[dict[str, Any]] = []
    for digest, group in candidates.groupby("sha256", sort=True):
        splits = tuple(sorted(set(group["split"].astype(str))))
        if len(splits) <= 1:
            continue
        duplicate_rows.append({
            "sha256": str(digest),
            "bytes": int(group["bytes"].iloc[0]),
            "splits": splits,
            "media_paths": tuple(sorted(set(group["media_path"].astype(str)))),
            "sample_ids": tuple(sorted(set(group["sample_id"].astype(str)))),
        })
    result = pd.DataFrame.from_records(duplicate_rows, columns=columns)
    result.attrs["candidate_files_hashed"] = len(unique_paths)
    return result


def audit_split_leakage(manifest: pd.DataFrame, config: SplitConfig) -> AuditReport:
    data = canonicalize_manifest(manifest)
    errors: list[str] = []
    warnings: list[str] = []
    if "split" not in manifest:
        return AuditReport("split leakage audit", ("manifest has no split column",))
    data["split"] = data["split"].fillna("").astype(str)
    allowed = set(config.ratios)
    invalid = sorted(set(data["split"]) - allowed)
    if invalid:
        errors.append(f"invalid split labels: {invalid}")
    real = ~data["is_synthetic"].astype(bool)
    missing_natural_groups = real & ~_natural_group_coverage(data)
    if missing_natural_groups.any():
        errors.append(
            f"{int(missing_natural_groups.sum())} real row(s) lack a meaningful natural grouping ID; "
            f"split auditing requires at least one of {list(NATURAL_GROUP_COLUMNS)}"
        )
    for key in config.leakage_keys:
        if key not in data:
            errors.append(f"missing leakage key: {key}")
            continue
        subset = data.loc[data[key].fillna("").astype(str).str.strip().ne(""), ["dataset", key, "split"]]
        leak = subset.groupby(["dataset", key], dropna=False)["split"].nunique().loc[lambda values: values > 1]
        if len(leak):
            errors.append(f"{len(leak)} {key} value(s) cross splits; first: {leak.index[0]}")
    duplicated_media = data.groupby("media_path")["split"].nunique().loc[lambda values: values > 1]
    if len(duplicated_media):
        errors.append(f"{len(duplicated_media)} media paths cross splits")
    content_duplicates = pd.DataFrame()
    try:
        content_duplicates = find_cross_split_content_duplicates(data)
    except DataGateError as exc:
        errors.append(str(exc))
    if len(content_duplicates):
        first = content_duplicates.iloc[0]
        errors.append(
            f"{len(content_duplicates)} exact media-content SHA-256 value(s) cross splits; "
            f"first: {first.sha256} in {list(first.splits)}"
        )
    if "split_unit" in manifest:
        unit_splits = manifest.groupby("split_unit")["split"].nunique()
        if (unit_splits > 1).any():
            errors.append(f"{int((unit_splits > 1).sum())} independence unit(s) cross splits")
    counts = data["split"].value_counts().reindex(config.ratios, fill_value=0).to_dict()
    empty = [name for name, ratio in config.ratios.items() if ratio > 0 and counts[name] == 0]
    if empty:
        warnings.append(f"empty requested split(s), usually due to too few independent units: {empty}")
    return AuditReport(
        "split leakage audit",
        tuple(errors),
        tuple(warnings),
        {
            "rows_by_split": counts,
            "independent_units": int(manifest.get("split_unit", pd.Series(dtype=str)).nunique()),
            "content_hash_candidate_files": int(content_duplicates.attrs.get("candidate_files_hashed", 0)),
            "cross_split_exact_content_hashes": int(len(content_duplicates)),
        },
    )


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if str(path).endswith(".gz") else None
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False, compression=compression)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _annotation_assignment_skeleton(
    split_manifest: pd.DataFrame,
    config: AnnotationConfig,
) -> pd.DataFrame:
    context_columns = [
        column
        for column in ("sample_id", "dataset", "mission", "split", "split_unit")
        if column in split_manifest
    ]
    samples = split_manifest.loc[:, context_columns].sort_values("sample_id", kind="stable")
    records: list[dict[str, Any]] = []
    for sample in samples.to_dict(orient="records"):
        sample_id = str(sample.pop("sample_id"))
        for task in ANNOTATION_TASKS:
            for slot in range(1, config.required_annotators + 1):
                records.append({
                    config.sample_column: sample_id,
                    **sample,
                    config.task_column: task,
                    "annotator_slot": slot,
                    config.label_column: "",
                    config.annotator_column: "",
                    config.blinded_column: "",
                    config.adjudicated_label_column: "",
                    config.adjudicator_column: "",
                    config.critical_column: "",
                    "evidence_geometry_json": "",
                    "notes": "",
                })
    columns = [
        config.sample_column,
        *[column for column in context_columns if column != "sample_id"],
        config.task_column,
        "annotator_slot",
        config.label_column,
        config.annotator_column,
        config.blinded_column,
        config.adjudicated_label_column,
        config.adjudicator_column,
        config.critical_column,
        "evidence_geometry_json",
        "notes",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def _telemetry_skeleton(split_manifest: pd.DataFrame) -> pd.DataFrame:
    sample_ids = split_manifest["sample_id"].astype(str).sort_values(kind="stable")
    telemetry = pd.DataFrame({"sample_id": sample_ids.to_numpy()})
    for column in TELEMETRY_COLUMNS[1:]:
        telemetry[column] = ""
    return telemetry.loc[:, list(TELEMETRY_COLUMNS)]


def export_preparation_bundle(
    manifest: pd.DataFrame | StudyConfig,
    destination: str | Path,
    annotation_config: AnnotationConfig | None = None,
    *,
    split_config: SplitConfig | None = None,
    overwrite: bool = False,
) -> Mapping[str, Path]:
    """Atomically publish inspectable inputs for the post-ingest workflow.

    ``manifest`` may be a :class:`StudyConfig`, in which case the canonical
    manifest and deterministic proposed split are built directly.  A DataFrame
    may already contain ``split``/``split_unit`` or may be paired with
    ``split_config``.  The exporter deliberately does *not* freeze the proposal
    and leaves all annotation labels, identities, adjudication fields, and
    telemetry values blank.
    """

    study_config: StudyConfig | None = manifest if isinstance(manifest, StudyConfig) else None
    if study_config is not None:
        if annotation_config is not None or split_config is not None:
            raise TypeError("annotation_config/split_config must be omitted when exporting from StudyConfig")
        canonical = build_canonical_manifest(study_config)
        annotation_config = study_config.annotations
        split_config = study_config.split
    elif isinstance(manifest, pd.DataFrame):
        canonical = canonicalize_manifest(manifest)
        annotation_config = annotation_config or AnnotationConfig()
        split_config = split_config or SplitConfig()
    else:
        raise TypeError("manifest must be a pandas DataFrame or StudyConfig")

    manifest_report = audit_manifest(canonical, study_config)
    manifest_report.raise_for_errors()
    if {"split", "split_unit"}.issubset(canonical.columns):
        proposed = canonical.copy()
    else:
        if split_config is None:
            raise DataGateError(
                "Preparation export requires split/split_unit columns or an explicit split_config"
            )
        proposed = assign_group_safe_splits(canonical, split_config)
    if split_config is None:
        raise DataGateError("Preparation export requires split_config to audit the proposed split")
    split_report = audit_split_leakage(proposed, split_config)
    split_report.raise_for_errors()

    bundle_root = Path(destination).expanduser().resolve()
    if not bundle_root.name:
        raise DataGateError("Preparation bundle destination must be a named directory")
    if bundle_root.exists() and not overwrite:
        raise DataGateError(f"Preparation bundle destination already exists: {bundle_root}")
    if bundle_root.exists() and not bundle_root.is_dir():
        raise DataGateError(f"Preparation bundle destination is not a directory: {bundle_root}")
    bundle_root.parent.mkdir(parents=True, exist_ok=True)

    annotation_config.validate()
    canonical_output = canonicalize_manifest(canonical).loc[:, list(CANONICAL_COLUMNS)]
    split_output = proposed.loc[:, ["sample_id", "split", "split_unit"]].sort_values(
        "sample_id", kind="stable"
    )
    if split_output["sample_id"].duplicated().any():
        raise DataGateError("Preparation split proposal contains duplicate sample_id values")
    annotation_output = _annotation_assignment_skeleton(proposed, annotation_config)
    telemetry_output = _telemetry_skeleton(proposed)
    frames = {
        "canonical_manifest": canonical_output,
        "proposed_split_manifest": split_output,
        "annotation_assignment_skeleton": annotation_output,
        "telemetry_skeleton": telemetry_output,
    }

    staging = Path(tempfile.mkdtemp(prefix=f".{bundle_root.name}.staging-", dir=bundle_root.parent))
    backup: Path | None = None
    try:
        for key, frame in frames.items():
            _atomic_csv(frame, staging / PREPARATION_BUNDLE_FILENAMES[key])
        artifacts = {
            key: {
                "path": PREPARATION_BUNDLE_FILENAMES[key],
                "rows": int(len(frame)),
                "sha256": _sha256_file(staging / PREPARATION_BUNDLE_FILENAMES[key]),
            }
            for key, frame in frames.items()
        }
        _atomic_json({
            "schema_version": "semuav.preparation-bundle.v1",
            "status": "PREPARATION_ONLY_REQUIRES_HUMAN_REVIEW_AND_FREEZE",
            "split_is_frozen": False,
            "labels_or_telemetry_invented": False,
            "manifest_audit": manifest_report.as_dict(),
            "split_audit": split_report.as_dict(),
            "artifacts": artifacts,
        }, staging / PREPARATION_BUNDLE_FILENAMES["bundle_manifest"])

        if bundle_root.exists():
            backup = Path(tempfile.mkdtemp(
                prefix=f".{bundle_root.name}.backup-", dir=bundle_root.parent,
            ))
            backup.rmdir()
            os.replace(bundle_root, backup)
        try:
            os.replace(staging, bundle_root)
        except Exception:
            if backup is not None and backup.exists() and not bundle_root.exists():
                os.replace(backup, bundle_root)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None and backup.exists() and not bundle_root.exists():
            os.replace(backup, bundle_root)
    return {
        key: (bundle_root / filename).resolve()
        for key, filename in PREPARATION_BUNDLE_FILENAMES.items()
    }


def freeze_split_manifest(manifest: pd.DataFrame, path: str | Path, *, overwrite: bool = False) -> Path:
    """Persist only immutable split identities; refuses accidental replacement."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise DataGateError(f"Frozen split manifest already exists: {destination}")
    required = {"sample_id", "split", "split_unit"}
    missing = required - set(manifest)
    if missing:
        raise DataGateError(f"Cannot freeze splits; missing columns: {sorted(missing)}")
    frozen = manifest.loc[:, ["sample_id", "split", "split_unit"]].sort_values("sample_id", kind="stable")
    if frozen["sample_id"].duplicated().any():
        raise DataGateError("Cannot freeze splits with duplicate sample_id values")
    _atomic_csv(frozen, destination)
    return destination.resolve()


def apply_frozen_splits(manifest: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise DataGateError(f"Frozen split manifest does not exist: {source}")
    frozen = pd.read_csv(source)
    required = {"sample_id", "split", "split_unit"}
    if missing := required - set(frozen):
        raise DataGateError(f"Frozen split manifest is missing columns: {sorted(missing)}")
    if frozen["sample_id"].duplicated().any():
        raise DataGateError("Frozen split manifest contains duplicate sample_id values")
    data = canonicalize_manifest(manifest)
    current, expected = set(data["sample_id"]), set(frozen["sample_id"].astype(str))
    if current != expected:
        raise DataGateError(
            "Frozen/current sample IDs differ: "
            f"{len(current - expected)} new, {len(expected - current)} missing"
        )
    data = data.drop(columns=[column for column in ("split", "split_unit") if column in data])
    return data.merge(frozen.astype({"sample_id": str}), on="sample_id", how="left", validate="one_to_one")


def load_annotations(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise DataGateError(f"Annotation table does not exist: {source}")
    suffixes = "".join(source.suffixes).lower()
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(source)
    if suffixes.endswith(".jsonl") or suffixes.endswith(".ndjson"):
        return pd.read_json(source, lines=True)
    raise DataGateError("Annotation table must be CSV, CSV.GZ, JSONL, or NDJSON")


def _normalized_label(value: Any) -> str:
    if _is_blank(value):
        return ""
    if isinstance(value, str):
        text = value.strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return text
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
    )


def _agreement(rows: pd.DataFrame, item_columns: Sequence[str], label_column: str) -> tuple[float, float, int]:
    matches = 0
    pairs = 0
    disagreements = 0
    for _, group in rows.groupby(list(item_columns), dropna=False, sort=False):
        labels = group[label_column].tolist()
        unique = set(labels)
        disagreements += int(len(unique) > 1)
        for left, right in itertools.combinations(labels, 2):
            pairs += 1
            matches += int(left == right)
    observed = matches / pairs if pairs else float("nan")
    frequencies = rows[label_column].value_counts(normalize=True)
    chance = float(np.square(frequencies.to_numpy(dtype=float)).sum()) if len(frequencies) else float("nan")
    if not np.isfinite(observed) or not np.isfinite(chance):
        corrected = float("nan")
    elif chance >= 1.0 - 1e-12:
        corrected = 1.0 if observed >= 1.0 - 1e-12 else float("nan")
    else:
        corrected = (observed - chance) / (1.0 - chance)
    return observed, corrected, disagreements


def _evidence_box(value: Any) -> tuple[float, float, float, float] | tuple[()] | None:
    """Parse a normalized evidence box; ``()`` is an explicit no-evidence label."""
    if _is_blank(value):
        return None
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if parsed == []:
        return ()
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 4:
        return None
    try:
        box = tuple(float(item) for item in parsed)
    except (TypeError, ValueError):
        return None
    if (
        not all(np.isfinite(box))
        or not all(0.0 <= item <= 1.0 for item in box)
        or box[2] <= box[0]
        or box[3] <= box[1]
    ):
        return None
    return box


def _evidence_box_iou(
    left: tuple[float, float, float, float] | tuple[()],
    right: tuple[float, float, float, float] | tuple[()],
) -> float:
    if not left and not right:
        return float("nan")
    if not left or not right:
        return 0.0
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def _evidence_geometry_agreement(
    rows: pd.DataFrame,
    config: AnnotationConfig,
) -> tuple[dict[str, float | int | bool], list[str]]:
    geometry_rows = rows.loc[rows[config.task_column].eq("evidence_roi")].copy()
    threshold = float(config.min_evidence_geometry_iou)
    report: dict[str, float | int | bool] = {
        "rows": int(len(geometry_rows)),
        "items": int(geometry_rows[config.sample_column].nunique()) if len(geometry_rows) else 0,
        "pairwise_comparisons": 0,
        "pairwise_median_iou": float("nan"),
        "pairwise_gate_passed": False,
        "adjudication_comparisons": 0,
        "adjudication_items": 0,
        "adjudication_required_items": 0,
        "adjudication_unresolved_items": 0,
        "adjudication_median_iou": float("nan"),
        "adjudication_gate_applicable": False,
        "adjudication_gate_passed": True,
        "minimum_iou": threshold,
    }
    errors: list[str] = []
    if geometry_rows.empty:
        if config.enabled:
            errors.append("evidence_roi localization task is missing")
        return report, errors

    geometry_rows["__geometry"] = geometry_rows[config.label_column].map(_evidence_box)
    invalid_labels = geometry_rows["__geometry"].isna()
    if invalid_labels.any():
        errors.append(f"{int(invalid_labels.sum())} evidence_roi label(s) are not normalized boxes or []")

    pairwise_scores: list[float] = []
    adjudication_scores: list[float] = []
    adjudication_items = 0
    adjudication_required_items = 0
    adjudication_unresolved_items = 0
    for _, group in geometry_rows.groupby(config.sample_column, dropna=False, sort=False):
        valid_labels = [value for value in group["__geometry"] if value is not None]
        for left, right in itertools.combinations(valid_labels, 2):
            score = _evidence_box_iou(left, right)
            if np.isfinite(score):
                pairwise_scores.append(score)

        # Only coordinate disagreements need a documented geometry
        # adjudication. They never enter categorical agreement/kappa.
        if group["__label"].nunique() <= 1:
            continue
        adjudication_required_items += 1
        if config.adjudicated_label_column not in group:
            adjudication_unresolved_items += 1
            continue
        adjudicated = [
            _evidence_box(value)
            for value in group[config.adjudicated_label_column]
            if not _is_blank(value)
        ]
        adjudicated = [value for value in adjudicated if value is not None]
        unique_targets = set(adjudicated)
        adjudicators = (
            group[config.adjudicator_column].fillna("").astype(str).str.strip()
            if config.adjudicator_column in group else pd.Series([], dtype=str)
        )
        complete_adjudicator = (
            len(adjudicators) == len(group)
            and adjudicators.ne("").all()
            and adjudicators.nunique() == 1
        )
        if len(adjudicated) != len(group) or len(unique_targets) != 1 or not complete_adjudicator:
            adjudication_unresolved_items += 1
            continue
        target = next(iter(unique_targets))
        item_scores = [
            _evidence_box_iou(value, target)
            for value in valid_labels
        ]
        item_scores = [value for value in item_scores if np.isfinite(value)]
        if item_scores:
            adjudication_items += 1
            adjudication_scores.extend(item_scores)

    pairwise_median = float(np.median(pairwise_scores)) if pairwise_scores else float("nan")
    adjudication_median = float(np.median(adjudication_scores)) if adjudication_scores else float("nan")
    report.update({
        "pairwise_comparisons": len(pairwise_scores),
        "pairwise_median_iou": pairwise_median,
        "pairwise_gate_passed": bool(np.isfinite(pairwise_median) and pairwise_median >= threshold),
        "adjudication_comparisons": len(adjudication_scores),
        "adjudication_items": adjudication_items,
        "adjudication_required_items": adjudication_required_items,
        "adjudication_unresolved_items": adjudication_unresolved_items,
        "adjudication_median_iou": adjudication_median,
        "adjudication_gate_applicable": bool(adjudication_required_items),
        "adjudication_gate_passed": bool(
            not adjudication_unresolved_items
            and (not adjudication_scores or adjudication_median >= threshold)
        ),
    })
    if not pairwise_scores:
        errors.append("evidence_roi pairwise localization IoU is undefined (no supported evidence-box pairs)")
    elif pairwise_median < threshold:
        errors.append(
            f"evidence_roi pairwise median IoU {pairwise_median:.3f} is below {threshold:.3f}"
        )
    if adjudication_scores and adjudication_median < threshold:
        errors.append(
            f"evidence_roi adjudication median IoU {adjudication_median:.3f} is below {threshold:.3f}"
        )
    if config.require_adjudication_on_disagreement and adjudication_unresolved_items:
        errors.append(
            f"{adjudication_unresolved_items} evidence_roi disagreement(s) lack one documented geometry adjudication"
        )
    return report, errors


def audit_annotations(
    annotations: pd.DataFrame,
    manifest: pd.DataFrame,
    config: AnnotationConfig,
) -> AnnotationGateReport:
    """Check independent annotation, agreement, and adjudication requirements."""

    config.validate()
    rows = annotations.copy()
    required = {
        config.sample_column,
        config.task_column,
        config.label_column,
        config.annotator_column,
    }
    missing = sorted(required - set(rows))
    if missing:
        return AnnotationGateReport(
            errors=(f"annotation table is missing columns: {missing}",),
            warnings=(),
            n_rows=len(rows),
            n_items=0,
            n_disagreements=0,
            n_adjudicated_disagreements=0,
            observed_agreement=float("nan"),
            chance_corrected_agreement=float("nan"),
            manifest_coverage=0.0,
            by_task={},
        )
    errors: list[str] = []
    warnings: list[str] = []
    item_columns = [config.sample_column, config.task_column]
    rows[config.sample_column] = rows[config.sample_column].fillna("").astype(str).str.strip()
    rows[config.task_column] = rows[config.task_column].fillna("").astype(str).str.strip()
    rows[config.annotator_column] = rows[config.annotator_column].fillna("").astype(str).str.strip()
    rows["__label"] = rows[config.label_column].map(_normalized_label)
    # Evidence boxes are continuous geometry, not categorical labels.  Their
    # agreement is evaluated by the dedicated IoU gate below; including their
    # coordinate strings in observed agreement/kappa would reject nearby boxes
    # that meet the preregistered localization threshold.
    categorical_rows = rows.loc[~rows[config.task_column].eq("evidence_roi")].copy()
    empty_required = (
        rows[config.sample_column].eq("")
        | rows[config.task_column].eq("")
        | rows[config.annotator_column].eq("")
        | rows["__label"].eq("")
    )
    if empty_required.any():
        errors.append(f"{int(empty_required.sum())} annotation row(s) have empty required values")
    duplicate = rows.duplicated(item_columns + [config.annotator_column], keep=False)
    if duplicate.any():
        errors.append(f"{int(duplicate.sum())} row(s) duplicate an annotator within an item")
    rater_counts = rows.groupby(item_columns, dropna=False)[config.annotator_column].nunique()
    too_few = rater_counts < config.required_annotators
    if too_few.any():
        errors.append(
            f"{int(too_few.sum())} item(s) have fewer than {config.required_annotators} independent annotators"
        )
    if config.require_blinding:
        if config.blinded_column not in rows:
            errors.append(f"blinding column is missing: {config.blinded_column}")
        else:
            blinded = _coerce_bool(rows[config.blinded_column])
            if not blinded.all():
                errors.append(f"{int((~blinded).sum())} annotation row(s) are not marked blinded")

    observed, corrected, disagreement_count = _agreement(
        categorical_rows, item_columns, "__label"
    )
    if not np.isfinite(observed):
        errors.append("observed agreement is undefined (no annotator pairs)")
    elif observed < config.min_observed_agreement:
        errors.append(
            f"observed agreement {observed:.3f} is below {config.min_observed_agreement:.3f}"
        )
    if not np.isfinite(corrected):
        errors.append("chance-corrected agreement is undefined")
    elif corrected < config.min_chance_corrected_agreement:
        errors.append(
            f"chance-corrected agreement {corrected:.3f} is below "
            f"{config.min_chance_corrected_agreement:.3f}"
        )

    adjudicated_count = 0
    if disagreement_count and config.require_adjudication_on_disagreement:
        if config.adjudicated_label_column not in rows:
            errors.append(f"adjudicated label column is missing: {config.adjudicated_label_column}")
        else:
            normalized_adjudication = rows[config.adjudicated_label_column].map(_normalized_label)
            rows["__adjudicated"] = normalized_adjudication
            missing_adjudicator_column = config.adjudicator_column not in rows
            unresolved_adjudications: list[tuple[str, ...]] = []
            for _, group in categorical_rows.groupby(item_columns, dropna=False, sort=False):
                if group["__label"].nunique() <= 1:
                    continue
                adjudicated_values = group["__adjudicated"].astype(str).str.strip()
                adjudicator_values = (
                    pd.Series([], dtype=str) if missing_adjudicator_column
                    else group[config.adjudicator_column].fillna("").astype(str).str.strip()
                )
                complete_label = adjudicated_values.ne("").all() and adjudicated_values.nunique() == 1
                complete_adjudicator = (
                    not missing_adjudicator_column
                    and adjudicator_values.ne("").all()
                    and adjudicator_values.nunique() == 1
                )
                if complete_label and complete_adjudicator:
                    adjudicated_count += 1
                else:
                    key = tuple(str(group.iloc[0][column]) for column in item_columns)
                    unresolved_adjudications.append(key)
            if missing_adjudicator_column:
                errors.append(f"adjudicator column is missing: {config.adjudicator_column}")
            if unresolved_adjudications:
                examples = unresolved_adjudications[:5]
                errors.append(
                    f"{len(unresolved_adjudications)} disagreement(s) lack one documented adjudication; "
                    f"examples: {examples}"
                )

    manifest_ids = set(canonicalize_manifest(manifest)["sample_id"])
    annotated_ids = set(rows[config.sample_column]) - {""}
    covered = len(manifest_ids & annotated_ids)
    coverage = covered / len(manifest_ids) if manifest_ids else 0.0
    unknown = annotated_ids - manifest_ids
    if unknown:
        errors.append(f"annotations reference {len(unknown)} unknown sample_id value(s)")
    if config.require_manifest_coverage and covered < len(manifest_ids):
        errors.append(f"annotation coverage is {covered}/{len(manifest_ids)} samples")

    by_task: dict[str, Mapping[str, float | int]] = {}
    for task, group in categorical_rows.groupby(config.task_column, dropna=False, sort=True):
        task_observed, task_corrected, task_disagreements = _agreement(group, item_columns, "__label")
        by_task[str(task)] = {
            "rows": len(group),
            "items": int(group.groupby(item_columns).ngroups),
            "disagreements": task_disagreements,
            "observed_agreement": task_observed,
            "chance_corrected_agreement": task_corrected,
        }
        if not np.isfinite(task_observed) or task_observed < config.min_observed_agreement:
            errors.append(
                f"task {task!r} observed agreement "
                f"{task_observed:.3f} is below {config.min_observed_agreement:.3f}"
            )
        if not np.isfinite(task_corrected) or task_corrected < config.min_chance_corrected_agreement:
            errors.append(
                f"task {task!r} chance-corrected agreement "
                f"{task_corrected:.3f} is below {config.min_chance_corrected_agreement:.3f}"
            )
    geometry_report, geometry_errors = _evidence_geometry_agreement(rows, config)
    by_task["__geometry__:evidence_roi"] = geometry_report
    errors.extend(geometry_errors)
    if config.critical_column not in rows:
        errors.append(f"rare-class gate column is missing: {config.critical_column}")
    else:
        critical_mask = _coerce_bool(rows[config.critical_column]).astype(bool)
        critical_rows = categorical_rows.loc[critical_mask.reindex(categorical_rows.index, fill_value=False)]
        if not critical_rows.empty:
            for task, rare_group in critical_rows.groupby(config.task_column, sort=True):
                rare_observed, rare_corrected, rare_disagreements = _agreement(
                    rare_group, item_columns, "__label"
                )
                key = f"__rare__:{task}"
                by_task[key] = {
                    "rows": len(rare_group),
                    "items": int(rare_group.groupby(item_columns).ngroups),
                    "disagreements": rare_disagreements,
                    "observed_agreement": rare_observed,
                    "chance_corrected_agreement": rare_corrected,
                }
                if not np.isfinite(rare_corrected) or rare_corrected < config.min_rare_chance_corrected_agreement:
                    errors.append(
                        f"rare confirmatory task {task!r} chance-corrected agreement "
                        f"{rare_corrected:.3f} is below {config.min_rare_chance_corrected_agreement:.3f}"
                    )
    return AnnotationGateReport(
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(warnings),
        n_rows=len(rows),
        n_items=int(categorical_rows.groupby(item_columns, dropna=False).ngroups),
        n_disagreements=disagreement_count,
        n_adjudicated_disagreements=adjudicated_count,
        observed_agreement=observed,
        chance_corrected_agreement=corrected,
        manifest_coverage=coverage,
        by_task=by_task,
    )


def resolved_annotations(annotations: pd.DataFrame, config: AnnotationConfig) -> pd.DataFrame:
    """Return one final label per sample/task after consensus or adjudication."""

    rows = annotations.copy()
    item_columns = [config.sample_column, config.task_column]
    rows["__label"] = rows[config.label_column].map(_normalized_label)
    if config.adjudicated_label_column in rows:
        rows["__adjudicated"] = rows[config.adjudicated_label_column].map(_normalized_label)
    else:
        rows["__adjudicated"] = ""
    records: list[dict[str, Any]] = []
    for key, group in rows.groupby(item_columns, dropna=False, sort=True):
        labels = sorted(set(group["__label"]) - {""})
        adjudicated = sorted(set(group["__adjudicated"]) - {""})
        if len(labels) == 1:
            final = labels[0]
            source = "consensus"
        elif len(adjudicated) == 1:
            final = adjudicated[0]
            source = "adjudication"
        else:
            raise DataGateError(f"Cannot resolve annotation item {key}: unresolved disagreement")
        key_values = key if isinstance(key, tuple) else (key,)
        record = dict(zip(item_columns, key_values))
        record.update({"final_label": final, "resolution": source, "n_annotators": group[config.annotator_column].nunique()})
        records.append(record)
    return pd.DataFrame.from_records(records)


def prepare_analysis_manifest(
    manifest: pd.DataFrame,
    annotations: pd.DataFrame,
    annotation_config: AnnotationConfig,
    telemetry_manifest: str | Path | pd.DataFrame,
) -> pd.DataFrame:
    """Join exogenous mission/telemetry inputs with adjudicated visible-fact labels."""

    gate = audit_annotations(annotations, manifest, annotation_config)
    gate.raise_for_errors()
    labels = resolved_annotations(annotations, annotation_config)
    label_wide = labels.pivot(
        index=annotation_config.sample_column,
        columns=annotation_config.task_column,
        values="final_label",
    ).reset_index()
    telemetry = telemetry_manifest.copy() if isinstance(telemetry_manifest, pd.DataFrame) else pd.read_csv(telemetry_manifest)
    telemetry_required = set(TELEMETRY_COLUMNS)
    if missing := telemetry_required.difference(telemetry.columns):
        raise DataGateError(f"telemetry/mission manifest is missing: {sorted(missing)}")
    if telemetry["sample_id"].duplicated().any():
        raise DataGateError("telemetry/mission manifest has duplicate sample_id rows")
    base_required = {"sample_id", "media_path", "mission", "split", "split_unit"}
    if missing := base_required.difference(manifest.columns):
        raise DataGateError(f"split manifest is missing analysis fields: {sorted(missing)}")
    result = manifest.merge(telemetry, on="sample_id", validate="one_to_one", how="left")
    result = result.merge(label_wide, on="sample_id", validate="one_to_one", how="left")
    required_tasks = set(ANNOTATION_TASKS)
    if missing := required_tasks.difference(result.columns):
        raise DataGateError(f"annotation overlay lacks required task(s): {sorted(missing)}")
    required_values = list(telemetry_required - {"sample_id"}) + list(required_tasks)
    missing_rows = result[required_values].isna().any(axis=1)
    if missing_rows.any():
        raise DataGateError(f"{int(missing_rows.sum())} samples lack telemetry, mission assignment, or final labels")
    result["gold_action"] = result["gold_action"].astype(str).str.strip('"')
    allowed_actions = {"NO_ACTION", "REVIEW", "RESPOND"}
    if invalid := sorted(set(result["gold_action"]) - allowed_actions):
        raise DataGateError(f"invalid gold action labels: {invalid}")
    boolean_map = {"true": True, "false": False}
    result["critical"] = result["critical"].map(_normalized_label).str.strip('"').str.lower().map(boolean_map)
    result["sufficient_evidence"] = result["sufficient_evidence"].map(_normalized_label).str.strip('"').str.lower().map(boolean_map)
    if result[["critical", "sufficient_evidence"]].isna().any().any():
        raise DataGateError("critical and sufficient_evidence labels must be Boolean")
    result["cohort"] = result["cohort"].astype(str).str.strip('"')
    allowed_cohorts = {"natural_test", "safety", "ood", "development"}
    if invalid_cohorts := sorted(set(result["cohort"]) - allowed_cohorts):
        raise DataGateError(f"invalid cohort labels: {invalid_cohorts}")
    result["event"] = result["event"].map(_normalized_label).str.strip('"')
    allowed_events = {
        "urban": {"none_visible", "visible_road_obstruction", "visible_close_proximity", "insufficient_evidence"},
        "wildfire": {"none_visible", "visible_smoke", "visible_flame", "insufficient_evidence"},
        "visdrone_urban": {
            "no_registered_priority_visible",
            "visible_priority_outside_roi",
            "visible_priority_roi_occupancy",
            "priority_target_low_visibility",
            "visible_image_plane_overlap",
            "insufficient_evidence",
        },
    }
    invalid_events = [
        (row.sample_id, row.mission, row.event)
        for row in result[["sample_id", "mission", "event"]].itertuples(index=False)
        if row.mission not in allowed_events or row.event not in allowed_events[row.mission]
    ]
    if invalid_events:
        raise DataGateError(f"invalid mission event labels; first: {invalid_events[0]}")
    for field in ("objects", "relations"):
        def parse_list(value: Any, *, field: str = field) -> list[Any]:
            try:
                parsed = json.loads(str(value))
            except Exception as exc:
                raise DataGateError(f"{field} must be a canonical JSON list") from exc
            if not isinstance(parsed, list):
                raise DataGateError(f"{field} must be a canonical JSON list")
            return parsed
        result[field] = result[field].map(parse_list)

    def claim_label(item: Any, keys: Sequence[str], field: str) -> str:
        if isinstance(item, str) and item.strip():
            return item.strip()
        if isinstance(item, Mapping):
            for key in keys:
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        raise DataGateError(f"{field} entries require an explicit frozen-ontology label")

    def reference_value(item: Any) -> str | None:
        if isinstance(item, (str, int)) and not isinstance(item, bool):
            value = str(item).strip()
            return value or None
        if isinstance(item, Mapping):
            for key in ("track_id", "id", "label", "class", "name"):
                value = item.get(key)
                if isinstance(value, (str, int)) and not isinstance(value, bool):
                    normalized = str(value).strip()
                    if normalized:
                        return normalized
        return None

    for row in result[["sample_id", "mission", "objects", "relations"]].itertuples(index=False):
        ontology = MISSION_ONTOLOGY.get(str(row.mission))
        if ontology is None:
            raise DataGateError(f"sample {row.sample_id} has unsupported mission {row.mission!r}")
        object_references: set[str] = set()
        for item in row.objects:
            if not isinstance(item, Mapping):
                raise DataGateError(
                    f"sample {row.sample_id} object entries must be JSON objects with a label"
                )
            label = claim_label(item, ("label", "class", "name"), "objects")
            if label not in ontology["objects"]:
                raise DataGateError(
                    f"sample {row.sample_id} has unsupported {row.mission} object label {label!r}"
                )
            object_references.add(label)
            if isinstance(item, Mapping):
                for key in ("track_id", "id"):
                    reference = reference_value(item.get(key))
                    if reference is not None:
                        object_references.add(reference)
                        object_references.add(f"{label}:{reference}")
        for relation in row.relations:
            if not isinstance(relation, Mapping):
                raise DataGateError(
                    f"sample {row.sample_id} relation entries must be JSON objects"
                )
            predicate = claim_label(
                relation, ("predicate", "relation", "label", "type"), "relations",
            )
            if predicate not in ontology["relations"]:
                raise DataGateError(
                    f"sample {row.sample_id} has unsupported {row.mission} relation label {predicate!r}"
                )
            left = reference_value(relation.get("subject"))
            right = reference_value(relation.get("object"))
            if left is None or right is None:
                raise DataGateError(
                    f"sample {row.sample_id} relation {predicate!r} has incomplete object references"
                )
            if left not in ontology["objects"] or right not in ontology["objects"]:
                raise DataGateError(
                    f"sample {row.sample_id} relation {predicate!r} must use frozen object labels"
                )
            missing_references = [
                reference for reference in (left, right) if reference not in object_references
            ]
            if missing_references:
                raise DataGateError(
                    f"sample {row.sample_id} relation {predicate!r} references absent objects: "
                    f"{missing_references}"
                )

    inconsistent_insufficient = (~result["sufficient_evidence"].astype(bool)) & (
        result["event"].ne("insufficient_evidence") | result["gold_action"].ne("REVIEW")
    )
    inconsistent_supported = result["sufficient_evidence"].astype(bool) & result["event"].eq(
        "insufficient_evidence"
    )
    if inconsistent_insufficient.any() or inconsistent_supported.any():
        raise DataGateError(
            "gold sufficient_evidence, event, and ground action violate the frozen abstention contract"
        )
    def parse_roi(value: Any) -> list[float]:
        try:
            parsed = json.loads(str(value))
        except Exception as exc:
            raise DataGateError("evidence_roi must be a normalized JSON box [x1,y1,x2,y2] or []") from exc
        if parsed == []:
            return []
        if not isinstance(parsed, list) or len(parsed) != 4:
            raise DataGateError("evidence_roi must be a normalized JSON box [x1,y1,x2,y2] or []")
        box = [float(item) for item in parsed]
        if not all(0 <= item <= 1 for item in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise DataGateError("evidence_roi coordinates are invalid")
        return box
    result["evidence_roi"] = result["evidence_roi"].map(parse_roi)
    missing_supported_roi = result["sufficient_evidence"].astype(bool) & result["evidence_roi"].map(lambda value: len(value) != 4)
    if missing_supported_roi.any():
        raise DataGateError(f"{int(missing_supported_roi.sum())} evidence-supported samples lack an evidence ROI")
    result["ood"] = result["cohort"].eq("ood")
    result["image_path"] = result["media_path"]
    result["visual_group_id"] = result["split_unit"]
    result["clip_paths_json"] = "[]"
    sequence_rows = result[result["sequence_id"].fillna("").astype(str).str.strip().ne("")]
    for _, group in sequence_rows.groupby(["dataset", "sequence_id"], sort=False):
        ordered = group.sort_values(["timestamp_ms", "frame_index"], na_position="last")
        paths = ordered["media_path"].astype(str).tolist()
        indices = ordered.index.tolist()
        if len(paths) < 2:
            continue
        for position, row_index in enumerate(indices):
            start = max(0, min(position - 1, len(paths) - 3))
            clip = paths[start : min(len(paths), start + 3)]
            if len(clip) >= 2:
                result.at[row_index, "clip_paths_json"] = json.dumps(clip)
    result["telemetry_json"] = result.apply(
        lambda row: json.dumps({
            "gps_lat": float(row.gps_lat), "gps_lon": float(row.gps_lon),
            "altitude_m": float(row.altitude_m), "yaw_deg": float(row.yaw_deg),
        }, sort_keys=True), axis=1,
    )
    return result


def assert_study_data_ready(
    config: StudyConfig,
    manifest: pd.DataFrame,
    annotations: pd.DataFrame | None = None,
    *,
    require_frozen_test: bool = True,
) -> Mapping[str, Any]:
    """Single hard gate before computing any result presented as full-study evidence."""

    stage = "test" if require_frozen_test else "ingest"
    config.assert_ready(stage)
    manifest_report = audit_manifest(manifest, config)
    manifest_report.raise_for_errors()
    outputs: dict[str, Any] = {"manifest": manifest_report.as_dict()}

    if config.profile is Profile.DRY_RUN:
        outputs["scientific_status"] = "NON_SCIENTIFIC_SYNTHETIC_DRY_RUN"
        return outputs
    if "split" not in manifest or "split_unit" not in manifest:
        raise DataGateError("real-study manifest must contain split and split_unit columns")
    split_report = audit_split_leakage(manifest, config.split)
    split_report.raise_for_errors()
    outputs["split"] = split_report.as_dict()

    if require_frozen_test and config.paths.split_manifest:
        frozen = apply_frozen_splits(canonicalize_manifest(manifest), config.paths.split_manifest)
        current = manifest.loc[:, ["sample_id", "split", "split_unit"]].copy()
        check = current.merge(
            frozen.loc[:, ["sample_id", "split", "split_unit"]],
            on="sample_id",
            suffixes=("_current", "_frozen"),
            validate="one_to_one",
        )
        mismatch = (check["split_current"] != check["split_frozen"]) | (
            check["split_unit_current"] != check["split_unit_frozen"]
        )
        if mismatch.any():
            raise DataGateError(f"{int(mismatch.sum())} current split assignment(s) differ from frozen manifest")

    if config.profile is Profile.FULL:
        test = manifest.loc[manifest["split"].astype(str).eq("test")]
        independent_test_units = int(test["split_unit"].nunique())
        if independent_test_units < config.split.min_test_units:
            raise DataGateError(
                f"full test has {independent_test_units} independent units; "
                f"requires at least {config.split.min_test_units}"
            )
        critical = test.loc[_coerce_bool(test["critical_event"]).astype(bool)]
        critical_ids = critical["event_id"].replace("", pd.NA).dropna().astype(str)
        critical_events = int(critical_ids.nunique())
        if critical_events < config.split.min_critical_test_events:
            raise DataGateError(
                f"full test has {critical_events} independent critical events; "
                f"requires at least {config.split.min_critical_test_events}"
            )
        if annotations is None:
            raise DataGateError("full profile requires the blinded/adjudicated annotation table")
        annotation_report = audit_annotations(annotations, manifest, config.annotations)
        annotation_report.raise_for_errors()
        outputs["annotations"] = annotation_report.as_dict()
        outputs["scientific_status"] = "FULL_DATA_GATES_PASSED"
    else:
        if annotations is not None and config.annotations.enabled:
            annotation_report = audit_annotations(annotations, manifest, config.annotations)
            annotation_report.raise_for_errors()
            outputs["annotations"] = annotation_report.as_dict()
        outputs["scientific_status"] = "SMOKE_REAL_NON_PUBLISHABLE"
    return outputs


__all__ = [
    "ADAPTERS",
    "ANNOTATION_TASKS",
    "AuditReport",
    "AnnotationGateReport",
    "CANONICAL_COLUMNS",
    "DataGateError",
    "DatasetAdapter",
    "Flame3Adapter",
    "SyntheticAdapter",
    "TELEMETRY_COLUMNS",
    "UAVReasonAdapter",
    "UAVidAdapter",
    "VisDroneAdapter",
    "apply_frozen_splits",
    "assert_study_data_ready",
    "assign_group_safe_splits",
    "audit_annotations",
    "audit_manifest",
    "audit_split_leakage",
    "build_canonical_manifest",
    "build_dataset_manifest",
    "canonicalize_manifest",
    "export_preparation_bundle",
    "find_cross_split_content_duplicates",
    "freeze_split_manifest",
    "load_annotations",
    "prepare_analysis_manifest",
    "resolved_annotations",
]
