"""VisDrone publisher-label parsing and registered sensor-space endpoints.

This module deliberately limits its truth claims to facts present in the
official VisDrone DET/MOT annotations: 2-D boxes, track identifiers, publisher
categories, truncation, and occlusion.  A registered rule may turn those facts
into three operational labels (``RESPOND``, ``REVIEW``, and ``NO_ACTION``),
where ``RESPOND`` means an immediate ground notification for a registered
priority-category box occupying a declared image-plane ROI.

Publisher ``test-dev`` media and labels are excluded by default.  Functions
that can enumerate or read that partition require a verified, open one-time
test ledger.  Manifest construction records paths only; it never decodes image
pixels.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import re
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


class VisDroneFormatError(ValueError):
    """Raised when an official annotation row violates its registered schema."""


class UntouchedTestAccessError(RuntimeError):
    """Raised when publisher test-dev content is requested without authorization."""


class PublisherPartition(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST_DEV = "test-dev"


class StudyPartition(str, Enum):
    DEVELOPMENT_TRAIN = "development_train"
    DEVELOPMENT_VALIDATION = "development_validation"
    UNTOUCHED_TEST = "untouched_test"


@dataclasses.dataclass(frozen=True)
class VisDroneComponent:
    key: str
    directory_name: str
    task: str
    publisher_partition: PublisherPartition
    study_partition: StudyPartition

    @property
    def is_untouched_test(self) -> bool:
        return self.study_partition is StudyPartition.UNTOUCHED_TEST


VISDRONE_COMPONENTS: Mapping[str, VisDroneComponent] = {
    "det_train": VisDroneComponent(
        "det_train",
        "VisDrone2019-DET-train",
        "DET",
        PublisherPartition.TRAIN,
        StudyPartition.DEVELOPMENT_TRAIN,
    ),
    "det_val": VisDroneComponent(
        "det_val",
        "VisDrone2019-DET-val",
        "DET",
        PublisherPartition.VAL,
        StudyPartition.DEVELOPMENT_VALIDATION,
    ),
    "det_test_dev": VisDroneComponent(
        "det_test_dev",
        "VisDrone2019-DET-test-dev",
        "DET",
        PublisherPartition.TEST_DEV,
        StudyPartition.UNTOUCHED_TEST,
    ),
    "mot_train": VisDroneComponent(
        "mot_train",
        "VisDrone2019-MOT-train",
        "MOT",
        PublisherPartition.TRAIN,
        StudyPartition.DEVELOPMENT_TRAIN,
    ),
    "mot_val": VisDroneComponent(
        "mot_val",
        "VisDrone2019-MOT-val",
        "MOT",
        PublisherPartition.VAL,
        StudyPartition.DEVELOPMENT_VALIDATION,
    ),
    "mot_test_dev": VisDroneComponent(
        "mot_test_dev",
        "VisDrone2019-MOT-test-dev",
        "MOT",
        PublisherPartition.TEST_DEV,
        StudyPartition.UNTOUCHED_TEST,
    ),
}

DEVELOPMENT_COMPONENTS: tuple[str, ...] = (
    "det_train",
    "det_val",
    "mot_train",
    "mot_val",
)
UNTOUCHED_TEST_COMPONENTS: tuple[str, ...] = ("det_test_dev", "mot_test_dev")

VISDRONE_CATEGORY_NAMES: Mapping[int, str] = {
    0: "ignored_regions",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning_tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}
VISDRONE_CATEGORY_IDS: Mapping[str, int] = {
    name: category_id for category_id, name in VISDRONE_CATEGORY_NAMES.items()
}
VISDRONE_EVALUATED_CATEGORY_IDS = frozenset(range(1, 11))

MISSION_VISDRONE_URBAN = "visdrone_urban"
REGISTERED_MISSION_PROFILES = frozenset(
    {"vulnerable_road_user", "heavy_vehicle", "general_mobility"}
)

ACTION_RESPOND = "RESPOND"
ACTION_REVIEW = "REVIEW"
ACTION_NO_ACTION = "NO_ACTION"
REGISTERED_ACTIONS = frozenset({ACTION_RESPOND, ACTION_REVIEW, ACTION_NO_ACTION})

EVENT_PRIORITY_ROI_OCCUPANCY = "visible_priority_roi_occupancy"
EVENT_PRIORITY_OUTSIDE_ROI = "visible_priority_outside_roi"
EVENT_IMAGE_PLANE_OVERLAP = "visible_image_plane_overlap"
EVENT_PRIORITY_EVIDENCE_LIMITED = "priority_target_low_visibility"
EVENT_NO_PRIORITY_VISIBLE = "no_registered_priority_visible"
REGISTERED_SENSOR_EVENTS = frozenset(
    {
        EVENT_PRIORITY_ROI_OCCUPANCY,
        EVENT_PRIORITY_OUTSIDE_ROI,
        EVENT_IMAGE_PLANE_OVERLAP,
        EVENT_PRIORITY_EVIDENCE_LIMITED,
        EVENT_NO_PRIORITY_VISIBLE,
    }
)

RELATION_INSIDE_ROI = "inside_registered_roi"
RELATION_OUTSIDE_ROI = "outside_registered_roi"
RELATION_IMAGE_PLANE_OVERLAP = "image_plane_overlap"
REGISTERED_SENSOR_RELATIONS = frozenset(
    {RELATION_INSIDE_ROI, RELATION_OUTSIDE_ROI, RELATION_IMAGE_PLANE_OVERLAP}
)

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
_HEX64 = re.compile(r"[0-9a-f]{64}")


def component_spec(component: str | VisDroneComponent) -> VisDroneComponent:
    """Resolve a registered component key or official directory name."""

    if isinstance(component, VisDroneComponent):
        return component
    text = str(component).strip()
    if text in VISDRONE_COMPONENTS:
        return VISDRONE_COMPONENTS[text]
    matches = [
        spec
        for spec in VISDRONE_COMPONENTS.values()
        if spec.directory_name.casefold() == text.casefold()
    ]
    if len(matches) == 1:
        return matches[0]
    raise VisDroneFormatError(f"Unknown VisDrone component: {component!r}")


def publisher_split_mapping() -> pd.DataFrame:
    """Return the immutable publisher-to-study partition mapping."""

    return pd.DataFrame.from_records(
        [
            {
                "component": spec.key,
                "official_directory": spec.directory_name,
                "task": spec.task,
                "publisher_partition": spec.publisher_partition.value,
                "study_partition": spec.study_partition.value,
            }
            for spec in VISDRONE_COMPONENTS.values()
        ]
    ).sort_values("component", kind="stable", ignore_index=True)


@dataclasses.dataclass(frozen=True)
class UntouchedTestPermit:
    """Cryptographically identified authorization for one open test ledger."""

    ledger_path: Path
    ledger_sha256: str
    protocol_hash: str
    config_hash: str
    code_hash: str
    opened_utc: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_untouched_test_permit(
    ledger_path: str | Path,
    *,
    expected_protocol_hash: str | None = None,
    expected_config_hash: str | None = None,
    expected_code_hash: str | None = None,
) -> UntouchedTestPermit:
    """Verify an open one-time ledger and bind a permit to its exact bytes."""

    path = Path(ledger_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UntouchedTestAccessError(f"Cannot read test-access ledger: {path}") from exc
    if not isinstance(payload, Mapping) or payload.get("status") != "OPEN":
        raise UntouchedTestAccessError("Untouched-test ledger must exist and have status OPEN")

    values = {
        "protocol_hash": str(payload.get("protocol_hash", "")).lower(),
        "config_hash": str(payload.get("config_hash", "")).lower(),
        "code_hash": str(payload.get("code_hash", "")).lower(),
    }
    for name, value in values.items():
        if _HEX64.fullmatch(value) is None:
            raise UntouchedTestAccessError(f"Untouched-test ledger has invalid {name}")

    expected = {
        "protocol_hash": expected_protocol_hash,
        "config_hash": expected_config_hash,
        "code_hash": expected_code_hash,
    }
    for name, registered in expected.items():
        if registered is not None and values[name] != str(registered).lower():
            raise UntouchedTestAccessError(f"Untouched-test ledger {name} does not match")

    opened_utc = str(payload.get("opened_utc", "")).strip()
    if not opened_utc:
        raise UntouchedTestAccessError("Untouched-test ledger has no opening timestamp")
    return UntouchedTestPermit(
        ledger_path=path,
        ledger_sha256=_sha256(path),
        protocol_hash=values["protocol_hash"],
        config_hash=values["config_hash"],
        code_hash=values["code_hash"],
        opened_utc=opened_utc,
    )


def _require_test_permit(
    component: str | VisDroneComponent,
    permit: UntouchedTestPermit | None,
) -> VisDroneComponent:
    spec = component_spec(component)
    if spec.is_untouched_test and permit is None:
        raise UntouchedTestAccessError(
            f"Component {spec.key!r} is the untouched publisher test-dev partition"
        )
    if spec.is_untouched_test and permit is not None:
        try:
            current_hash = _sha256(permit.ledger_path)
            payload = json.loads(permit.ledger_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise UntouchedTestAccessError("Test-access ledger is no longer readable") from exc
        if (
            current_hash != permit.ledger_sha256
            or not isinstance(payload, Mapping)
            or payload.get("status") != "OPEN"
        ):
            raise UntouchedTestAccessError("Test-access ledger changed after authorization")
    return spec


@dataclasses.dataclass(frozen=True)
class VisDroneObject:
    """One validated publisher annotation in image coordinates."""

    component: str
    sequence_id: str
    frame_index: int
    object_id: int
    left: float
    top: float
    width: float
    height: float
    score: float
    category_id: int
    truncation: int
    occlusion: int
    source_line: int

    @property
    def category(self) -> str:
        return VISDRONE_CATEGORY_NAMES[self.category_id]

    @property
    def object_uid(self) -> str:
        return (
            f"{self.component}:{self.sequence_id}:"
            f"{self.frame_index:07d}:{self.object_id:07d}"
        )

    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        return (self.left, self.top, self.left + self.width, self.top + self.height)

    @property
    def is_publisher_evaluated_category(self) -> bool:
        return self.category_id in VISDRONE_EVALUATED_CATEGORY_IDS


def _fields(row: str | Sequence[Any], expected: int) -> list[str]:
    if isinstance(row, str):
        parsed = next(csv.reader([row], skipinitialspace=True))
    else:
        parsed = [str(value) for value in row]
    parsed = [value.strip() for value in parsed]
    if parsed:
        parsed[0] = parsed[0].lstrip("\ufeff")
    while parsed and parsed[-1] == "":
        parsed.pop()
    if len(parsed) != expected or any(value == "" for value in parsed):
        raise VisDroneFormatError(
            f"Expected exactly {expected} populated fields, observed {len(parsed)}"
        )
    return parsed


def _integer(value: str, name: str) -> int:
    if re.fullmatch(r"[-+]?\d+", value) is None:
        raise VisDroneFormatError(f"{name} must be an integer, observed {value!r}")
    return int(value)


def _number(value: str, name: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise VisDroneFormatError(f"{name} must be numeric, observed {value!r}") from exc
    if not math.isfinite(result):
        raise VisDroneFormatError(f"{name} must be finite")
    return result


def _validate_object(obj: VisDroneObject) -> VisDroneObject:
    spec = component_spec(obj.component)
    if obj.width <= 0 or obj.height <= 0:
        raise VisDroneFormatError("Bounding-box width and height must be positive")
    if not 0.0 <= obj.score <= 1.0:
        raise VisDroneFormatError("Score must be within [0, 1]")
    if obj.category_id not in VISDRONE_CATEGORY_NAMES:
        raise VisDroneFormatError(f"Unsupported publisher category id: {obj.category_id}")
    if obj.truncation not in {0, 1, 2}:
        raise VisDroneFormatError("Truncation must be 0, 1, or 2")
    if obj.occlusion not in {0, 1, 2}:
        raise VisDroneFormatError("Occlusion must be 0, 1, or 2")
    if obj.frame_index < 1:
        raise VisDroneFormatError("Frame index must be at least 1")
    if obj.object_id < 0:
        raise VisDroneFormatError("Object id must be nonnegative")
    if spec.task not in {"DET", "MOT"}:
        raise VisDroneFormatError(f"Unsupported component task: {spec.task}")
    return obj


def parse_det_row(
    row: str | Sequence[Any],
    *,
    component: str = "det_train",
    sample_id: str,
    frame_index: int = 1,
    source_line: int = 1,
) -> VisDroneObject:
    """Parse one official 8-field VisDrone DET row."""

    spec = component_spec(component)
    if spec.task != "DET":
        raise VisDroneFormatError(f"DET row cannot be assigned to {spec.key}")
    values = _fields(row, 8)
    obj = VisDroneObject(
        component=spec.key,
        sequence_id=str(sample_id),
        frame_index=int(frame_index),
        object_id=int(source_line),
        left=_number(values[0], "bbox_left"),
        top=_number(values[1], "bbox_top"),
        width=_number(values[2], "bbox_width"),
        height=_number(values[3], "bbox_height"),
        score=_number(values[4], "score"),
        category_id=_integer(values[5], "category"),
        truncation=_integer(values[6], "truncation"),
        occlusion=_integer(values[7], "occlusion"),
        source_line=int(source_line),
    )
    return _validate_object(obj)


def parse_mot_row(
    row: str | Sequence[Any],
    *,
    component: str = "mot_train",
    sequence_id: str,
    source_line: int = 1,
) -> VisDroneObject:
    """Parse one official 10-field VisDrone MOT row."""

    spec = component_spec(component)
    if spec.task != "MOT":
        raise VisDroneFormatError(f"MOT row cannot be assigned to {spec.key}")
    values = _fields(row, 10)
    obj = VisDroneObject(
        component=spec.key,
        sequence_id=str(sequence_id),
        frame_index=_integer(values[0], "frame_index"),
        object_id=_integer(values[1], "target_id"),
        left=_number(values[2], "bbox_left"),
        top=_number(values[3], "bbox_top"),
        width=_number(values[4], "bbox_width"),
        height=_number(values[5], "bbox_height"),
        score=_number(values[6], "score"),
        category_id=_integer(values[7], "category"),
        truncation=_integer(values[8], "truncation"),
        occlusion=_integer(values[9], "occlusion"),
        source_line=int(source_line),
    )
    return _validate_object(obj)


def parse_det_annotations(
    text: str,
    *,
    component: str,
    sample_id: str,
    frame_index: int = 1,
) -> tuple[VisDroneObject, ...]:
    """Parse a complete DET annotation payload, retaining source line numbers."""

    return tuple(
        parse_det_row(
            line,
            component=component,
            sample_id=sample_id,
            frame_index=frame_index,
            source_line=line_number,
        )
        for line_number, line in enumerate(text.splitlines(), start=1)
        if line.strip()
    )


def parse_mot_annotations(
    text: str,
    *,
    component: str,
    sequence_id: str,
    frame_index: int | None = None,
) -> tuple[VisDroneObject, ...]:
    """Parse a MOT sequence payload and optionally select one frame."""

    objects = tuple(
        parse_mot_row(
            line,
            component=component,
            sequence_id=sequence_id,
            source_line=line_number,
        )
        for line_number, line in enumerate(text.splitlines(), start=1)
        if line.strip()
    )
    if frame_index is None:
        return objects
    return tuple(obj for obj in objects if obj.frame_index == int(frame_index))


def _component_directory(root: Path, spec: VisDroneComponent) -> Path:
    direct = root / spec.directory_name
    if direct.is_dir():
        return direct.resolve()
    # A single publisher wrapper directory is tolerated.  This only inspects
    # directory entries; it does not enumerate or open media.
    if root.is_dir():
        matches = [
            child / spec.directory_name
            for child in root.iterdir()
            if child.is_dir() and (child / spec.directory_name).is_dir()
        ]
        if len(matches) == 1:
            return matches[0].resolve()
    raise FileNotFoundError(f"Missing official VisDrone component: {direct}")


def _frame_index(stem: str, fallback: int) -> int:
    match = re.search(r"(\d+)$", stem)
    value = int(match.group(1)) if match else fallback
    return max(1, value)


def _media_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing media directory: {directory}")
    return sorted(
        (
            path.resolve()
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        ),
        key=lambda path: path.name,
    )


def _manifest_records(
    root: Path,
    spec: VisDroneComponent,
    permit: UntouchedTestPermit | None,
) -> list[dict[str, Any]]:
    component_root = _component_directory(root, spec)
    records: list[dict[str, Any]] = []
    permit_hash = permit.ledger_sha256 if spec.is_untouched_test and permit else ""
    if spec.task == "DET":
        images = _media_files(component_root / "images")
        annotations = component_root / "annotations"
        for ordinal, image in enumerate(images, start=1):
            annotation = annotations / f"{image.stem}.txt"
            if not annotation.is_file():
                raise FileNotFoundError(f"Missing publisher annotation: {annotation}")
            # Official DET filenames encode a capture/sequence prefix before
            # the first underscore.  Keep those neighboring frames in one
            # independence group rather than treating each image as a flight.
            source_sequence = image.stem.split("_", 1)[0]
            sequence_id = f"{spec.key}:{source_sequence}"
            records.append(
                {
                    "sample_id": f"visdrone:{spec.key}:{image.stem}",
                    "dataset": "visdrone",
                    "mission": MISSION_VISDRONE_URBAN,
                    "component": spec.key,
                    "task": spec.task,
                    "publisher_partition": spec.publisher_partition.value,
                    "study_partition": spec.study_partition.value,
                    "source_uid": f"{spec.key}/images/{image.name}",
                    "media_path": str(image),
                    "annotation_path": str(annotation.resolve()),
                    "sequence_id": sequence_id,
                    "frame_index": _frame_index(image.stem, ordinal),
                    "split_group": f"visdrone:{sequence_id}",
                    "has_ground_truth": True,
                    "test_ledger_sha256": permit_hash,
                }
            )
    else:
        sequences = component_root / "sequences"
        annotations = component_root / "annotations"
        if not sequences.is_dir():
            raise FileNotFoundError(f"Missing sequence directory: {sequences}")
        for sequence in sorted(
            (path for path in sequences.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        ):
            annotation = annotations / f"{sequence.name}.txt"
            if not annotation.is_file():
                raise FileNotFoundError(f"Missing publisher annotation: {annotation}")
            sequence_id = f"{spec.key}:{sequence.name}"
            for ordinal, image in enumerate(_media_files(sequence), start=1):
                records.append(
                    {
                        "sample_id": f"visdrone:{spec.key}:{sequence.name}:{image.stem}",
                        "dataset": "visdrone",
                        "mission": MISSION_VISDRONE_URBAN,
                        "component": spec.key,
                        "task": spec.task,
                        "publisher_partition": spec.publisher_partition.value,
                        "study_partition": spec.study_partition.value,
                        "source_uid": f"{spec.key}/sequences/{sequence.name}/{image.name}",
                        "media_path": str(image),
                        "annotation_path": str(annotation.resolve()),
                        "sequence_id": sequence_id,
                        "frame_index": _frame_index(image.stem, ordinal),
                        "split_group": f"visdrone:{sequence_id}",
                        "has_ground_truth": True,
                        "test_ledger_sha256": permit_hash,
                    }
                )
    return records


def build_source_manifest(
    root: str | Path,
    *,
    components: Sequence[str] = DEVELOPMENT_COMPONENTS,
    test_permit: UntouchedTestPermit | None = None,
) -> pd.DataFrame:
    """Build a path-only manifest for registered components.

    Authorization is checked for every requested component before the dataset
    root is inspected.  Image dimensions and pixels are intentionally absent;
    consumers must obtain them only within their authorized study phase.
    """

    specs = tuple(component_spec(component) for component in components)
    if not specs:
        raise ValueError("At least one VisDrone component is required")
    duplicate_keys = [spec.key for spec in specs]
    if len(duplicate_keys) != len(set(duplicate_keys)):
        raise ValueError("VisDrone components must be unique")
    for spec in specs:
        _require_test_permit(spec, test_permit)

    data_root = Path(root).expanduser().resolve()
    records: list[dict[str, Any]] = []
    for spec in specs:
        records.extend(_manifest_records(data_root, spec, test_permit))
    if not records:
        raise VisDroneFormatError("Registered VisDrone components contain no media")
    manifest = pd.DataFrame.from_records(records)
    if manifest["sample_id"].duplicated().any():
        raise VisDroneFormatError("VisDrone sample identifiers are not unique")
    if manifest[["component", "source_uid"]].duplicated().any():
        raise VisDroneFormatError("VisDrone component/source identifiers are not unique")
    return manifest.sort_values(
        ["study_partition", "component", "sequence_id", "frame_index", "source_uid"],
        kind="stable",
        ignore_index=True,
    )


def build_development_manifest(root: str | Path) -> pd.DataFrame:
    """Build the publisher train/validation manifest without test-dev access."""

    return build_source_manifest(root, components=DEVELOPMENT_COMPONENTS)


def build_untouched_test_manifest(
    root: str | Path,
    *,
    test_permit: UntouchedTestPermit,
) -> pd.DataFrame:
    """Build the test-dev manifest after the one-time ledger is open."""

    return build_source_manifest(
        root,
        components=UNTOUCHED_TEST_COMPONENTS,
        test_permit=test_permit,
    )


def load_objects_for_manifest_row(
    row: Mapping[str, Any] | pd.Series,
    *,
    test_permit: UntouchedTestPermit | None = None,
) -> tuple[VisDroneObject, ...]:
    """Load publisher boxes for one manifest row, with test-dev gating."""

    spec = _require_test_permit(str(row["component"]), test_permit)
    if spec.is_untouched_test:
        bound_hash = str(row.get("test_ledger_sha256", ""))
        if test_permit is None or bound_hash != test_permit.ledger_sha256:
            raise UntouchedTestAccessError(
                "Test manifest row is not bound to the active one-time ledger"
            )
    annotation = Path(str(row["annotation_path"])).expanduser().resolve()
    try:
        text = annotation.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise VisDroneFormatError(f"Cannot read publisher annotation: {annotation}") from exc
    if spec.task == "DET":
        return parse_det_annotations(
            text,
            component=spec.key,
            sample_id=str(row["sample_id"]),
            frame_index=int(row["frame_index"]),
        )
    sequence_id = str(row["sequence_id"])
    sequence_name = sequence_id.split(":", 1)[-1]
    return parse_mot_annotations(
        text,
        component=spec.key,
        sequence_id=sequence_name,
        frame_index=int(row["frame_index"]),
    )


@dataclasses.dataclass(frozen=True)
class NormalizedROI:
    """A declared closed rectangle in normalized image coordinates."""

    name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        if not self.name.strip():
            raise ValueError("ROI name must be populated")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("ROI coordinates must be finite")
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("ROI coordinates must lie within [0, 1]")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("ROI must have positive normalized area")

    def as_list(self) -> list[float]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]


@dataclasses.dataclass(frozen=True)
class RegisteredBenchmarkPolicy:
    """Frozen mapping from publisher boxes to a sensor-space action endpoint."""

    rule_id: str
    roi: NormalizedROI
    priority_category_ids: frozenset[int]
    mission_profile: str
    occupancy_rule: str = "center"
    min_roi_visible_fraction: float = 0.25
    min_image_plane_iou: float = 0.05
    min_score: float = 0.5
    max_truncation: int = 2
    max_occlusion: int = 2

    def __post_init__(self) -> None:
        if not self.rule_id.strip():
            raise ValueError("Registered rule id must be populated")
        if self.mission_profile not in REGISTERED_MISSION_PROFILES:
            raise ValueError(
                "mission_profile must be one of the three registered VisDrone profiles"
            )
        if not self.priority_category_ids:
            raise ValueError("At least one registered priority category is required")
        if not self.priority_category_ids.issubset(VISDRONE_EVALUATED_CATEGORY_IDS):
            raise ValueError("Priority categories must be publisher-evaluated ids 1 through 10")
        if self.occupancy_rule not in {"center", "visible_bbox_overlap"}:
            raise ValueError("occupancy_rule must be center or visible_bbox_overlap")
        for name, value in (
            ("min_roi_visible_fraction", self.min_roi_visible_fraction),
            ("min_image_plane_iou", self.min_image_plane_iou),
            ("min_score", self.min_score),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie within [0, 1]")
        if self.max_truncation not in {0, 1, 2} or self.max_occlusion not in {0, 1, 2}:
            raise ValueError("Maximum truncation and occlusion must be 0, 1, or 2")

    @classmethod
    def from_category_names(
        cls,
        *,
        rule_id: str,
        roi: NormalizedROI,
        mission_profile: str,
        priority_categories: Iterable[str],
        **kwargs: Any,
    ) -> "RegisteredBenchmarkPolicy":
        names = tuple(str(name).strip() for name in priority_categories)
        unknown = sorted(set(names) - set(VISDRONE_CATEGORY_IDS))
        if unknown:
            raise ValueError(f"Unknown VisDrone priority categories: {unknown}")
        return cls(
            rule_id=rule_id,
            roi=roi,
            priority_category_ids=frozenset(VISDRONE_CATEGORY_IDS[name] for name in names),
            mission_profile=mission_profile,
            **kwargs,
        )


@dataclasses.dataclass(frozen=True)
class SensorObjectFact:
    object_uid: str
    category_id: int
    category: str
    bbox_normalized_xyxy: tuple[float, float, float, float]
    truncation: int
    occlusion: int
    registered_priority: bool

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SensorRelation:
    kind: str
    subject_uid: str
    object_uid: str | None = None
    value: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in REGISTERED_SENSOR_RELATIONS:
            raise ValueError(f"Unregistered sensor relation: {self.kind}")

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SourceDerivedTruth:
    sample_id: str
    mission: str
    mission_profile: str
    gold_action: str
    events: tuple[str, ...]
    objects: tuple[SensorObjectFact, ...]
    relations: tuple[SensorRelation, ...]
    priority_endpoint_positive: bool
    evidence_roi: NormalizedROI
    rule_id: str
    truth_scope: str = "publisher_2d_boxes_registered_sensor_rule"

    def __post_init__(self) -> None:
        if self.gold_action not in REGISTERED_ACTIONS:
            raise ValueError(f"Unregistered action: {self.gold_action}")
        if not set(self.events).issubset(REGISTERED_SENSOR_EVENTS):
            raise ValueError("Truth contains an unregistered sensor event")

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "mission": self.mission,
            "mission_profile": self.mission_profile,
            "gold_action": self.gold_action,
            "events": list(self.events),
            "objects": [obj.as_dict() for obj in self.objects],
            "relations": [relation.as_dict() for relation in self.relations],
            "priority_endpoint_positive": self.priority_endpoint_positive,
            "evidence_roi": {
                "name": self.evidence_roi.name,
                "xyxy": self.evidence_roi.as_list(),
            },
            "rule_id": self.rule_id,
            "truth_scope": self.truth_scope,
        }


def _visible_box(
    obj: VisDroneObject,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = obj.bbox_xyxy
    clipped = (
        max(0.0, min(float(image_width), x1)),
        max(0.0, min(float(image_height), y1)),
        max(0.0, min(float(image_width), x2)),
        max(0.0, min(float(image_height), y2)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def _normalize_box(
    box: tuple[float, float, float, float], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    return (
        box[0] / image_width,
        box[1] / image_height,
        box[2] / image_width,
        box[3] / image_height,
    )


def _intersection(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    overlap = _intersection(first, second)
    union = _area(first) + _area(second) - overlap
    return overlap / union if union > 0 else 0.0


def _occupies_roi(
    box: tuple[float, float, float, float],
    policy: RegisteredBenchmarkPolicy,
) -> bool:
    roi = (policy.roi.x_min, policy.roi.y_min, policy.roi.x_max, policy.roi.y_max)
    if policy.occupancy_rule == "center":
        center_x = (box[0] + box[2]) / 2.0
        center_y = (box[1] + box[3]) / 2.0
        return roi[0] <= center_x <= roi[2] and roi[1] <= center_y <= roi[3]
    visible_area = _area(box)
    return visible_area > 0 and _intersection(box, roi) / visible_area >= policy.min_roi_visible_fraction


def derive_source_truth(
    *,
    sample_id: str,
    objects: Sequence[VisDroneObject],
    image_width: int,
    image_height: int,
    policy: RegisteredBenchmarkPolicy,
) -> SourceDerivedTruth:
    """Apply a frozen, sensor-space policy to official boxes for one image."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive")
    visible: list[tuple[VisDroneObject, tuple[float, float, float, float]]] = []
    for obj in objects:
        if not obj.is_publisher_evaluated_category:
            continue
        if obj.score < policy.min_score:
            continue
        clipped = _visible_box(obj, image_width, image_height)
        if clipped is not None:
            visible.append((obj, _normalize_box(clipped, image_width, image_height)))

    visible.sort(key=lambda pair: pair[0].object_uid)
    facts = tuple(
        SensorObjectFact(
            object_uid=obj.object_uid,
            category_id=obj.category_id,
            category=obj.category,
            bbox_normalized_xyxy=box,
            truncation=obj.truncation,
            occlusion=obj.occlusion,
            registered_priority=obj.category_id in policy.priority_category_ids,
        )
        for obj, box in visible
    )

    priority = [pair for pair in visible if pair[0].category_id in policy.priority_category_ids]
    evidence_limited = [
        pair
        for pair in priority
        if pair[0].truncation > policy.max_truncation
        or pair[0].occlusion > policy.max_occlusion
    ]
    respond_eligible = [pair for pair in priority if pair not in evidence_limited]
    inside = [pair for pair in respond_eligible if _occupies_roi(pair[1], policy)]
    geometric_inside = [pair for pair in priority if _occupies_roi(pair[1], policy)]
    outside = [pair for pair in priority if pair not in geometric_inside]
    relations: list[SensorRelation] = [
        SensorRelation(RELATION_INSIDE_ROI, obj.object_uid)
        for obj, _ in geometric_inside
    ]
    relations.extend(
        SensorRelation(RELATION_OUTSIDE_ROI, obj.object_uid)
        for obj, _ in outside
    )

    overlapping = False
    for index, (first_obj, first_box) in enumerate(visible):
        for second_obj, second_box in visible[index + 1 :]:
            overlap_iou = _iou(first_box, second_box)
            if overlap_iou >= policy.min_image_plane_iou:
                overlapping = True
                relations.append(
                    SensorRelation(
                        RELATION_IMAGE_PLANE_OVERLAP,
                        first_obj.object_uid,
                        second_obj.object_uid,
                        overlap_iou,
                    )
                )

    events: list[str] = []
    if inside:
        events.append(EVENT_PRIORITY_ROI_OCCUPANCY)
    if outside:
        events.append(EVENT_PRIORITY_OUTSIDE_ROI)
    if evidence_limited:
        events.append(EVENT_PRIORITY_EVIDENCE_LIMITED)
    if overlapping:
        events.append(EVENT_IMAGE_PLANE_OVERLAP)
    if not priority:
        events.append(EVENT_NO_PRIORITY_VISIBLE)

    action = ACTION_RESPOND if inside else ACTION_REVIEW if priority else ACTION_NO_ACTION
    return SourceDerivedTruth(
        sample_id=str(sample_id),
        mission=MISSION_VISDRONE_URBAN,
        mission_profile=policy.mission_profile,
        gold_action=action,
        events=tuple(events),
        objects=facts,
        relations=tuple(relations),
        priority_endpoint_positive=bool(inside),
        evidence_roi=policy.roi,
        rule_id=policy.rule_id,
    )


def build_source_truth_table(
    manifest: pd.DataFrame,
    *,
    image_sizes: Mapping[str, tuple[int, int]],
    policy: RegisteredBenchmarkPolicy,
    test_permit: UntouchedTestPermit | None = None,
) -> pd.DataFrame:
    """Build JSON-safe registered truth rows without decoding media."""

    required = {
        "sample_id",
        "component",
        "annotation_path",
        "sequence_id",
        "frame_index",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    records: list[dict[str, Any]] = []
    for row in manifest.sort_values("sample_id", kind="stable").to_dict("records"):
        sample_id = str(row["sample_id"])
        if sample_id not in image_sizes:
            raise ValueError(f"Missing registered image dimensions for {sample_id}")
        width, height = image_sizes[sample_id]
        truth = derive_source_truth(
            sample_id=sample_id,
            objects=load_objects_for_manifest_row(row, test_permit=test_permit),
            image_width=int(width),
            image_height=int(height),
            policy=policy,
        )
        payload = truth.as_dict()
        records.append(
            {
                "sample_id": sample_id,
                "mission": payload["mission"],
                "mission_profile": payload["mission_profile"],
                "gold_action": payload["gold_action"],
                "priority_endpoint_positive": payload["priority_endpoint_positive"],
                "events_json": json.dumps(payload["events"], sort_keys=True, separators=(",", ":")),
                "objects_json": json.dumps(payload["objects"], sort_keys=True, separators=(",", ":")),
                "relations_json": json.dumps(payload["relations"], sort_keys=True, separators=(",", ":")),
                "evidence_roi_json": json.dumps(payload["evidence_roi"], sort_keys=True, separators=(",", ":")),
                "rule_id": payload["rule_id"],
                "truth_scope": payload["truth_scope"],
            }
        )
    return pd.DataFrame.from_records(records)


__all__ = [
    "ACTION_NO_ACTION",
    "ACTION_RESPOND",
    "ACTION_REVIEW",
    "DEVELOPMENT_COMPONENTS",
    "EVENT_IMAGE_PLANE_OVERLAP",
    "EVENT_NO_PRIORITY_VISIBLE",
    "EVENT_PRIORITY_EVIDENCE_LIMITED",
    "EVENT_PRIORITY_OUTSIDE_ROI",
    "EVENT_PRIORITY_ROI_OCCUPANCY",
    "NormalizedROI",
    "MISSION_VISDRONE_URBAN",
    "PublisherPartition",
    "REGISTERED_ACTIONS",
    "REGISTERED_MISSION_PROFILES",
    "REGISTERED_SENSOR_EVENTS",
    "REGISTERED_SENSOR_RELATIONS",
    "RELATION_IMAGE_PLANE_OVERLAP",
    "RELATION_INSIDE_ROI",
    "RELATION_OUTSIDE_ROI",
    "RegisteredBenchmarkPolicy",
    "SensorObjectFact",
    "SensorRelation",
    "SourceDerivedTruth",
    "StudyPartition",
    "UNTOUCHED_TEST_COMPONENTS",
    "UntouchedTestAccessError",
    "UntouchedTestPermit",
    "VISDRONE_CATEGORY_IDS",
    "VISDRONE_CATEGORY_NAMES",
    "VISDRONE_COMPONENTS",
    "VisDroneComponent",
    "VisDroneFormatError",
    "VisDroneObject",
    "build_development_manifest",
    "build_source_manifest",
    "build_source_truth_table",
    "build_untouched_test_manifest",
    "component_spec",
    "derive_source_truth",
    "load_objects_for_manifest_row",
    "load_untouched_test_permit",
    "parse_det_annotations",
    "parse_det_row",
    "parse_mot_annotations",
    "parse_mot_row",
    "publisher_split_mapping",
]
