from __future__ import annotations

import dataclasses
import enum
import hashlib
import io
import json
import math
import re
import struct
import subprocess
import zlib
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from .registry import StudyGateError, canonical_json
from .runtime import resolve_ffmpeg_executable


PROTOCOL_VERSION = 1
IPV4_HEADER_BYTES = 20
UDP_HEADER_BYTES = 8
FRAGMENT_HEADER_BYTES = 36
MESSAGE_HEADER_BYTES = struct.calcsize("!4sBBBBQQQqqI8s")


class PayloadMode(enum.IntEnum):
    L0_HEARTBEAT = 0
    L1_ALERT = 1
    L2_SEMANTICS = 2
    L3_SEMANTICS_ROI = 3
    L4_FEATURES = 4
    L5_IMAGE_OR_CLIP = 5


class StateKind(str, enum.Enum):
    NONE = "none"
    KEYFRAME = "keyframe"
    DELTA = "delta"


class ApplyStatus(str, enum.Enum):
    APPLIED = "applied"
    DUPLICATE = "duplicate"
    BASE_MISMATCH = "base_mismatch"
    STALE = "stale"
    EXPIRED = "expired"
    DECODE_ERROR = "decode_error"
    VERSION_MISMATCH = "version_mismatch"


@dataclasses.dataclass(frozen=True)
class ReliabilityProfile:
    # Covers the preregistered worst-case BDP (20 Mbps × 200 ms ≈ 500 kB)
    # at the 1,136-byte fragment payload.  A window of eight would impose an
    # artificial RTT throughput ceiling unrelated to the configured link.
    window_fragments: int = 512
    retry_limit: int = 2
    min_rto_us: int = 200_000
    max_rto_us: int = 2_000_000
    delayed_ack_us: int = 10_000
    ack_every_n_fragments: int = 2

    def __post_init__(self) -> None:
        if self.window_fragments < 1 or self.retry_limit < 0:
            raise ValueError("Invalid reliability window or retry limit")
        if not 0 < self.min_rto_us <= self.max_rto_us:
            raise ValueError("Invalid RTO bounds")


@dataclasses.dataclass(frozen=True)
class AppMessage:
    run_id: str
    logical_observation_id: str
    message_id: int
    mode: PayloadMode
    capture_us: int
    ready_us: int
    action_deadline_us: int
    state_expiry_us: int
    priority: int
    payload: bytes
    state_kind: StateKind = StateKind.NONE
    base_state_seq: int | None = None
    resulting_state_seq: int | None = None
    reliability: ReliabilityProfile = dataclasses.field(default_factory=ReliabilityProfile)

    def __post_init__(self) -> None:
        if self.ready_us < self.capture_us:
            raise ValueError("A message cannot be ready before capture")
        if self.state_expiry_us < self.action_deadline_us:
            raise ValueError("State expiry must not precede action deadline")
        if self.state_kind is StateKind.DELTA and self.base_state_seq is None:
            raise ValueError("A delta requires a base_state_seq")
        if self.state_kind is not StateKind.NONE and self.resulting_state_seq is None:
            raise ValueError("A semantic state message requires resulting_state_seq")

    @property
    def payload_hash(self) -> bytes:
        return hashlib.sha256(self.payload).digest()

    @property
    def wire_payload(self) -> bytes:
        return encode_app_envelope(self)

    @property
    def wire_payload_hash(self) -> bytes:
        return hashlib.sha256(self.wire_payload).digest()


@dataclasses.dataclass(frozen=True)
class DecodedEnvelope:
    protocol_version: int
    mode: PayloadMode
    state_kind: StateKind
    priority: int
    capture_us: int
    action_deadline_us: int
    state_expiry_us: int
    base_state_seq: int | None
    resulting_state_seq: int | None
    payload: bytes
    payload_hash_prefix: bytes


_STATE_TO_INT = {StateKind.NONE: 0, StateKind.KEYFRAME: 1, StateKind.DELTA: 2}
_INT_TO_STATE = {value: key for key, value in _STATE_TO_INT.items()}


def encode_app_envelope(message: AppMessage) -> bytes:
    header = struct.pack(
        "!4sBBBBQQQqqI8s",
        b"SUV1", PROTOCOL_VERSION, int(message.mode), _STATE_TO_INT[message.state_kind],
        int(np.clip(message.priority, 0, 255)), message.capture_us,
        message.action_deadline_us, message.state_expiry_us,
        message.base_state_seq if message.base_state_seq is not None else -1,
        message.resulting_state_seq if message.resulting_state_seq is not None else -1,
        len(message.payload), message.payload_hash[:8],
    )
    if len(header) != MESSAGE_HEADER_BYTES:
        raise AssertionError("Application message header size drift")
    return header + message.payload


def decode_app_envelope(data: bytes) -> DecodedEnvelope:
    if len(data) < MESSAGE_HEADER_BYTES:
        raise ValueError("Truncated application envelope")
    fields = struct.unpack("!4sBBBBQQQqqI8s", data[:MESSAGE_HEADER_BYTES])
    magic, version, mode, state_kind, priority, capture, deadline, expiry, base, result, size, hash_prefix = fields
    payload = data[MESSAGE_HEADER_BYTES:]
    if magic != b"SUV1" or version != PROTOCOL_VERSION:
        raise ValueError("Application envelope version mismatch")
    if len(payload) != size or hashlib.sha256(payload).digest()[:8] != hash_prefix:
        raise ValueError("Application envelope length or payload hash mismatch")
    return DecodedEnvelope(
        version, PayloadMode(mode), _INT_TO_STATE[state_kind], priority, capture,
        deadline, expiry, None if base < 0 else base, None if result < 0 else result,
        payload, hash_prefix,
    )


@dataclasses.dataclass(frozen=True)
class Fragment:
    protocol_version: int
    message_id: int
    fragment_index: int
    fragment_count: int
    payload_offset: int
    fragment_payload: bytes
    message_hash_prefix: bytes
    crc32: int

    @property
    def udp_payload_bytes(self) -> int:
        return FRAGMENT_HEADER_BYTES + len(self.fragment_payload)

    @property
    def ip_packet_bytes(self) -> int:
        return IPV4_HEADER_BYTES + UDP_HEADER_BYTES + self.udp_payload_bytes

    def to_bytes(self) -> bytes:
        if len(self.message_hash_prefix) != 8:
            raise ValueError("message_hash_prefix must contain 8 bytes")
        header = struct.pack(
            "!BBQHHII8sI2x",
            self.protocol_version,
            0,
            self.message_id,
            self.fragment_index,
            self.fragment_count,
            self.payload_offset,
            len(self.fragment_payload),
            self.message_hash_prefix,
            self.crc32,
        )
        if len(header) != FRAGMENT_HEADER_BYTES:
            raise AssertionError("Fragment header size drift")
        return header + self.fragment_payload

    @classmethod
    def from_bytes(cls, data: bytes) -> "Fragment":
        if len(data) < FRAGMENT_HEADER_BYTES:
            raise ValueError("Truncated fragment")
        fields = struct.unpack("!BBQHHII8sI2x", data[:FRAGMENT_HEADER_BYTES])
        version, _, message_id, index, count, offset, size, hash_prefix, crc = fields
        payload = data[FRAGMENT_HEADER_BYTES:]
        if len(payload) != size or zlib.crc32(payload) & 0xFFFFFFFF != crc:
            raise ValueError("Fragment length or CRC mismatch")
        return cls(version, message_id, index, count, offset, payload, hash_prefix, crc)


@dataclasses.dataclass(frozen=True)
class FragmentAck:
    message_id: int
    received_indices: tuple[int, ...]
    generated_us: int


@dataclasses.dataclass(frozen=True)
class ApplicationAck:
    message_id: int
    status: ApplyStatus
    receiver_state_seq: int
    action_ready_us: int | None
    generated_us: int
    semantic_state_message: bool = False
    state_update_us: int | None = None


REQUIRED_SEMANTIC_FIELDS = {
    "schema_version",
    "observation_id",
    "mission",
    "capture_time_us",
    "telemetry",
    "sufficient_evidence",
    "event",
    "objects",
    "relations",
    "risk",
    "uncertainty",
    "recommended_action",
}

SEMANTIC_ONTOLOGY_VERSION = "semuav.ontology.v1"
VISDRONE_PROPOSAL_DECISION_VERSION = "semuav.visdrone-proposal-decision.v2"
VISDRONE_COMPOSED_PACKET_AUTHORSHIP = "deterministic_host_composer"
VISDRONE_COMPOSED_PROVENANCE_FIELDS = frozenset({
    "semantic_packet_authorship",
    "decision_contract_version",
    "decision_composer_policy_sha256",
    "decision_context_sha256",
})
MISSION_ONTOLOGY: Mapping[str, Mapping[str, frozenset[str]]] = {
    "visdrone_urban": {
        "objects": frozenset({
            "pedestrian", "people", "bicycle", "car", "van", "truck",
            "tricycle", "awning_tricycle", "bus", "motor",
        }),
        # These predicates describe only normalized image-plane geometry.
        "relations": frozenset({"image_plane_overlap"}),
        "events": frozenset({
            "no_registered_priority_visible",
            "visible_priority_outside_roi",
            "visible_priority_roi_occupancy",
            "priority_target_low_visibility",
            "visible_image_plane_overlap",
            "insufficient_evidence",
        }),
    },
    "urban": {
        "objects": frozenset({
            "pedestrian", "people", "bicycle", "car", "van", "truck",
            "tricycle", "awning_tricycle", "bus", "motor",
        }),
        "relations": frozenset({
            "near", "overlapping", "stopped_in_roadway", "entering_roi", "leaving_roi",
        }),
        "events": frozenset({
            "none_visible", "visible_road_obstruction", "visible_close_proximity",
            "insufficient_evidence",
        }),
    },
    "wildfire": {
        "objects": frozenset({
            "visible_smoke", "visible_flame", "vegetation", "structure",
            "vehicle", "person",
        }),
        "relations": frozenset({
            "smoke_near_structure", "flame_near_vegetation", "person_near_visible_fire",
        }),
        "events": frozenset({
            "none_visible", "visible_smoke", "visible_flame", "insufficient_evidence",
        }),
    },
}


def _ontology_item_label(item: Any, keys: Sequence[str], kind: str) -> str:
    if isinstance(item, str):
        label = item
    elif isinstance(item, Mapping):
        label = next(
            (item[key] for key in keys if isinstance(item.get(key), str) and item[key].strip()),
            None,
        )
        if label is None:
            raise ValueError(f"Scientific {kind} records require an explicit ontology label")
    else:
        raise ValueError(f"Scientific {kind} records must be strings or objects")
    return str(label).strip()


def _claim_reference(value: Any) -> str | None:
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        normalized = str(value).strip()
        return normalized or None
    if isinstance(value, Mapping):
        for key in ("track_id", "id", "label", "class", "name"):
            candidate = value.get(key)
            if isinstance(candidate, (str, int)) and not isinstance(candidate, bool):
                normalized = str(candidate).strip()
                if normalized:
                    return normalized
    return None


def validate_semantic_content_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Validate semantic content against the frozen ontology.

    This is deliberately stricter than the base wire-schema check and
    deliberately separate from transmission provenance/calibration metadata.
    Experiment 1 uses it before model selection; the sender adds the latter
    metadata only after a calibration model has been frozen.  Semantic content
    may be model-authored for legacy missions or deterministically host-composed
    from a bounded proposal decision for the amended VisDrone pipeline.
    """
    normalized = validate_semantic_packet(packet)
    ontology = MISSION_ONTOLOGY[str(normalized["mission"])]
    event = normalized["event"]
    if not isinstance(event, str) or event not in ontology["events"]:
        raise ValueError(
            f"Unsupported {normalized['mission']} event ontology label: {event!r}"
        )
    object_references: set[str] = set()
    for item in normalized["objects"]:
        if not isinstance(item, Mapping):
            raise ValueError("Semantic object records must be JSON objects")
        label = _ontology_item_label(item, ("label", "class", "name"), "object")
        if label not in ontology["objects"]:
            raise ValueError(
                f"Unsupported {normalized['mission']} object ontology label: {label!r}"
            )
        box = item.get("bbox_normalized_xyxy")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(
                "Semantic objects require bbox_normalized_xyxy with four coordinates"
            )
        try:
            coordinates = [float(value) for value in box]
        except (TypeError, ValueError) as exc:
            raise ValueError("Object bbox_normalized_xyxy coordinates must be numeric") from exc
        if (
            not all(np.isfinite(coordinates))
            or not 0 <= coordinates[0] < coordinates[2] <= 1
            or not 0 <= coordinates[1] < coordinates[3] <= 1
        ):
            raise ValueError("Object bbox_normalized_xyxy must be valid within [0, 1]")
        try:
            confidence = float(item["confidence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Semantic objects require numeric confidence") from exc
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Object confidence must lie within [0, 1]")
        object_references.add(label)
        for key in ("track_id", "id"):
            reference = _claim_reference(item.get(key))
            if reference is not None:
                object_references.add(reference)
                object_references.add(f"{label}:{reference}")
    for item in normalized["relations"]:
        if not isinstance(item, Mapping):
            raise ValueError("Semantic relation records must be JSON objects")
        label = _ontology_item_label(
            item, ("predicate", "relation", "label", "type"), "relation",
        )
        if label not in ontology["relations"]:
            raise ValueError(
                f"Unsupported {normalized['mission']} relation ontology label: {label!r}"
            )
        reference_pair = next(
            (
                (left, right) for left, right in (
                    ("subject", "object"), ("subject_id", "object_id"),
                    ("source", "target"),
                ) if left in item or right in item
            ),
            None,
        )
        if reference_pair is None:
            raise ValueError(
                f"Semantic relation {label!r} requires subject and object references"
            )
        left_key, right_key = reference_pair
        left = _claim_reference(item.get(left_key))
        right = _claim_reference(item.get(right_key))
        if left is None or right is None:
            raise ValueError(f"Scientific relation {label!r} has incomplete object references")
        if left not in ontology["objects"] or right not in ontology["objects"]:
            raise ValueError(
                f"Semantic relation {label!r} must reference frozen object labels"
            )
        missing_references = [
            reference for reference in (left, right) if reference not in object_references
        ]
        if missing_references:
            raise ValueError(
                f"Scientific relation {label!r} references absent objects: {missing_references}"
            )
    if not isinstance(normalized.get("evidence"), list):
        raise ValueError("Semantic content packets require an evidence list")
    for evidence in normalized["evidence"]:
        if not isinstance(evidence, Mapping):
            raise ValueError("Every evidence record must be an object")
        evidence_id = evidence.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise ValueError("Every semantic evidence record requires evidence_id")
        linked_claims = evidence.get("linked_claims")
        if (
            not isinstance(linked_claims, list)
            or not linked_claims
            or any(not isinstance(claim, str) or not claim.strip() for claim in linked_claims)
        ):
            raise ValueError("Evidence linked_claims must be a nonempty list of strings")
        box = evidence.get("bbox_normalized_xyxy")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(
                "Semantic evidence requires bbox_normalized_xyxy"
            )
        try:
            coordinates = [float(value) for value in box]
        except (TypeError, ValueError) as exc:
            raise ValueError("Evidence bbox_normalized_xyxy coordinates must be numeric") from exc
        if (
            not all(np.isfinite(coordinates))
            or not 0 <= coordinates[0] < coordinates[2] <= 1
            or not 0 <= coordinates[1] < coordinates[3] <= 1
        ):
            raise ValueError("Evidence bbox_normalized_xyxy must be valid within [0, 1]")
    grounded_claims = bool(normalized["objects"] or normalized["relations"]) or normalized["event"] not in {
        "none_visible",
        "no_registered_priority_visible",
        "insufficient_evidence",
    }
    if grounded_claims and not normalized["evidence"]:
        raise ValueError("Grounded semantic claims require at least one evidence record")
    if not grounded_claims and normalized["evidence"]:
        raise ValueError("Evidence must be empty when the packet contains no grounded claim")
    return normalized


def validate_semantic_packet(
    packet: Mapping[str, Any], *, require_transmission_metadata: bool = False,
) -> dict[str, Any]:
    missing = REQUIRED_SEMANTIC_FIELDS.difference(packet)
    if missing:
        raise ValueError(f"Semantic packet missing fields: {sorted(missing)}")
    if packet["schema_version"] != "semuav.v1":
        raise ValueError("Unsupported semantic schema version")
    if packet["mission"] not in MISSION_ONTOLOGY:
        raise ValueError(
            "mission must be one of the frozen mission ontology identifiers"
        )
    if not isinstance(packet["observation_id"], str) or not packet["observation_id"].strip():
        raise ValueError("observation_id must be a nonempty string")
    try:
        capture_time_us = float(packet["capture_time_us"])
    except (TypeError, ValueError) as exc:
        raise ValueError("capture_time_us must be a nonnegative integer") from exc
    if not np.isfinite(capture_time_us) or capture_time_us < 0 or not capture_time_us.is_integer():
        raise ValueError("capture_time_us must be a nonnegative integer")
    if not isinstance(packet["telemetry"], Mapping):
        raise ValueError("telemetry must be an object")
    telemetry: dict[str, float] = {}
    for name in ("gps_lat", "gps_lon", "altitude_m", "yaw_deg"):
        if name not in packet["telemetry"]:
            raise ValueError(f"telemetry.{name} is authoritative and required")
        try:
            telemetry[name] = float(packet["telemetry"][name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"telemetry.{name} must be numeric") from exc
        if not np.isfinite(telemetry[name]):
            raise ValueError(f"telemetry.{name} must be finite")
    if not -90.0 <= telemetry["gps_lat"] <= 90.0:
        raise ValueError("telemetry.gps_lat must lie within [-90, 90]")
    if not -180.0 <= telemetry["gps_lon"] <= 180.0:
        raise ValueError("telemetry.gps_lon must lie within [-180, 180]")
    if telemetry["altitude_m"] < 0.0:
        raise ValueError("telemetry.altitude_m must be nonnegative")
    if not 0.0 <= telemetry["yaw_deg"] < 360.0:
        raise ValueError("telemetry.yaw_deg must lie within [0, 360)")
    if not isinstance(packet["sufficient_evidence"], bool):
        raise ValueError("sufficient_evidence must be boolean")
    if packet["recommended_action"] not in {"NO_ACTION", "REVIEW", "RESPOND"}:
        raise ValueError("Unknown action label")
    if not packet["sufficient_evidence"]:
        if packet["event"] != "insufficient_evidence" or packet["recommended_action"] != "REVIEW":
            raise ValueError(
                "Insufficient evidence requires event=insufficient_evidence and recommended_action=REVIEW"
            )
    elif packet["event"] == "insufficient_evidence":
        raise ValueError("event=insufficient_evidence requires sufficient_evidence=false")
    for name in ("risk", "uncertainty"):
        value = float(packet[name])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    if not isinstance(packet["objects"], list) or not isinstance(packet["relations"], list):
        raise ValueError("objects and relations must be lists")
    if require_transmission_metadata:
        if not isinstance(packet.get("evidence"), list):
            raise ValueError("Scientific semantic packets require an evidence list")
        try:
            calibrated_probability = float(packet["calibrated_correctness_probability"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Scientific semantic packets require calibrated_correctness_probability"
            ) from exc
        if not np.isfinite(calibrated_probability) or not 0.0 <= calibrated_probability <= 1.0:
            raise ValueError("calibrated_correctness_probability must lie within [0, 1]")
        provenance = packet.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("Scientific semantic packets require provenance")
        required_provenance = {
            "source_model", "model_revision", "prompt_hash",
            "ontology_version", "calibration", "schema_version",
        }
        if missing_provenance := required_provenance.difference(provenance):
            raise ValueError(
                f"Semantic packet provenance missing fields: {sorted(missing_provenance)}"
            )
        if provenance["schema_version"] != packet["schema_version"]:
            raise ValueError("Semantic packet and provenance schema versions differ")
        for name in ("source_model", "ontology_version"):
            if not isinstance(provenance[name], str) or not provenance[name].strip():
                raise ValueError(f"Semantic packet provenance {name} must be nonempty")
        if provenance["ontology_version"] != SEMANTIC_ONTOLOGY_VERSION:
            raise ValueError(
                f"Semantic packet ontology_version must be {SEMANTIC_ONTOLOGY_VERSION}"
            )
        composed_fields_present = VISDRONE_COMPOSED_PROVENANCE_FIELDS.intersection(
            provenance
        )
        if composed_fields_present:
            missing_composed = VISDRONE_COMPOSED_PROVENANCE_FIELDS.difference(
                provenance
            )
            if missing_composed:
                raise ValueError(
                    "Composed VisDrone provenance is incomplete; missing="
                    f"{sorted(missing_composed)}"
                )
            if str(packet["mission"]) != "visdrone_urban":
                raise ValueError(
                    "Proposal-decision composer provenance is restricted to visdrone_urban"
                )
            if (
                provenance["semantic_packet_authorship"]
                != VISDRONE_COMPOSED_PACKET_AUTHORSHIP
            ):
                raise ValueError(
                    "Composed VisDrone semantic_packet_authorship must identify the "
                    "deterministic host composer"
                )
            if (
                provenance["decision_contract_version"]
                != VISDRONE_PROPOSAL_DECISION_VERSION
            ):
                raise ValueError("Unsupported VisDrone proposal-decision contract version")
            for name in (
                "decision_composer_policy_sha256", "decision_context_sha256",
            ):
                if not re.fullmatch(r"[0-9a-f]{64}", str(provenance[name])):
                    raise ValueError(
                        f"Semantic packet provenance {name} must be a SHA-256 hex digest"
                    )
        ontology = MISSION_ONTOLOGY[str(packet["mission"])]
        event = packet["event"]
        if not isinstance(event, str) or event not in ontology["events"]:
            raise ValueError(f"Unsupported {packet['mission']} event ontology label: {event!r}")
        object_references: set[str] = set()
        for item in packet["objects"]:
            label = _ontology_item_label(item, ("label", "class", "name"), "object")
            if label not in ontology["objects"]:
                raise ValueError(
                    f"Unsupported {packet['mission']} object ontology label: {label!r}"
                )
            object_references.add(label)
            if isinstance(item, Mapping):
                for key in ("track_id", "id"):
                    reference = _claim_reference(item.get(key))
                    if reference is not None:
                        object_references.add(reference)
                        object_references.add(f"{label}:{reference}")
        for item in packet["relations"]:
            label = _ontology_item_label(
                item, ("predicate", "relation", "label", "type"), "relation",
            )
            if label not in ontology["relations"]:
                raise ValueError(
                    f"Unsupported {packet['mission']} relation ontology label: {label!r}"
                )
            if not isinstance(item, Mapping):
                continue
            reference_pair = next(
                (
                    (left, right) for left, right in (
                        ("subject", "object"), ("subject_id", "object_id"), ("source", "target"),
                    )
                    if left in item or right in item
                ),
                None,
            )
            if reference_pair is None:
                continue
            left_key, right_key = reference_pair
            left = _claim_reference(item.get(left_key))
            right = _claim_reference(item.get(right_key))
            if left is None or right is None:
                raise ValueError(f"Scientific relation {label!r} has incomplete object references")
            missing_references = [
                reference for reference in (left, right) if reference not in object_references
            ]
            if missing_references:
                raise ValueError(
                    f"Scientific relation {label!r} references absent objects: {missing_references}"
                )
        if not re.fullmatch(r"[0-9a-f]{40}", str(provenance["model_revision"])):
            raise ValueError("Semantic packet model_revision must be a 40-character lowercase hex commit")
        if not re.fullmatch(r"[0-9a-f]{64}", str(provenance["prompt_hash"])):
            raise ValueError("Semantic packet prompt_hash must be a SHA-256 hex digest")
        calibration = provenance["calibration"]
        if not isinstance(calibration, Mapping):
            raise ValueError("Semantic packet calibration provenance must be an object")
        if not isinstance(calibration.get("method"), str) or not str(calibration["method"]).strip():
            raise ValueError("Semantic packet calibration method must be nonempty")
        try:
            temperature = float(calibration["temperature"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Semantic packet calibration temperature is required") from exc
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Semantic packet calibration temperature must be finite and positive")
        for evidence in packet["evidence"]:
            if not isinstance(evidence, Mapping):
                raise ValueError("Every evidence record must be an object")
            linked_claims = evidence.get("linked_claims")
            if (
                not isinstance(linked_claims, list)
                or not linked_claims
                or any(not isinstance(claim, str) or not claim.strip() for claim in linked_claims)
            ):
                raise ValueError("Evidence linked_claims must be a nonempty list of strings")
            integrity = evidence.get("integrity_sha256")
            if integrity is not None and not re.fullmatch(r"[0-9a-f]{64}", str(integrity)):
                raise ValueError("Evidence integrity_sha256 must be a SHA-256 hex digest")
            for box_name in ("bbox_normalized_xyxy", "bbox_xyxy", "bbox", "box"):
                if box_name not in evidence:
                    continue
                box = evidence[box_name]
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    raise ValueError(f"Evidence {box_name} must contain four coordinates")
                try:
                    coordinates = [float(value) for value in box]
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Evidence {box_name} coordinates must be numeric") from exc
                if not all(np.isfinite(coordinates)):
                    raise ValueError(f"Evidence {box_name} coordinates must be finite")
                if min(coordinates) < 0 or coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
                    raise ValueError(f"Evidence {box_name} coordinates are invalid")
                if "normalized" in box_name and max(coordinates) > 1:
                    raise ValueError(f"Evidence {box_name} must lie within [0, 1]")
    return json.loads(canonical_json(packet))


def semantic_level_view(packet: Mapping[str, Any], level: PayloadMode) -> dict[str, Any]:
    packet = validate_semantic_packet(packet)
    common = {
        "schema_version": packet["schema_version"],
        "observation_id": packet["observation_id"],
        "mission": packet["mission"],
        "capture_time_us": packet["capture_time_us"],
        "telemetry": packet["telemetry"],
    }
    if "provenance" in packet:
        common["provenance"] = packet["provenance"]
    if level == PayloadMode.L0_HEARTBEAT:
        return common
    if level == PayloadMode.L1_ALERT:
        if "calibrated_correctness_probability" not in packet:
            raise ValueError("L1 alerts require calibrated_correctness_probability")
        if "provenance" not in packet or "confidence_provenance" not in packet:
            raise ValueError("L1 alerts require model and confidence provenance")
        return common | {
            "sufficient_evidence": packet["sufficient_evidence"],
            "event": packet["event"],
            "risk": packet["risk"],
            "uncertainty": packet["uncertainty"],
            "calibrated_correctness_probability": packet["calibrated_correctness_probability"],
            "recommended_action": packet["recommended_action"],
            **(
                {"confidence_provenance": packet["confidence_provenance"]}
                if "confidence_provenance" in packet else {}
            ),
        }
    return packet


def encode_mapping(
    value: Mapping[str, Any],
    *,
    scientific: bool,
    wire_format: str | None = None,
) -> tuple[bytes, str]:
    """Encode a mapping, optionally preserving an already-declared format.

    With no requested format, scientific runs require CBOR while debug runs
    prefer CBOR and fall back to canonical JSON.  Callers that derive a delta
    from an existing payload must pass that payload's ``wire_format`` so a
    dependency difference cannot change the encoding without changing the
    transmitted metadata.
    """
    if wire_format not in {None, "cbor", "json-debug"}:
        raise ValueError(f"Unknown wire format {wire_format}")
    if wire_format == "json-debug":
        if scientific:
            raise StudyGateError("JSON debug encoding is forbidden in the full profile")
        return canonical_json(value).encode("utf-8"), "json-debug"
    try:
        import cbor2  # type: ignore
    except ImportError:
        if scientific or wire_format == "cbor":
            raise StudyGateError("Full-profile wire measurements require cbor2; JSON is debug-only")
        return canonical_json(value).encode("utf-8"), "json-debug"
    return cbor2.dumps(value, canonical=True), "cbor"


def decode_mapping(data: bytes, wire_format: str) -> dict[str, Any]:
    if wire_format == "cbor":
        try:
            import cbor2  # type: ignore
        except ImportError as exc:
            raise StudyGateError("Install cbor2 to decode the experimental wire format") from exc
        value = cbor2.loads(data)
    elif wire_format == "json-debug":
        value = json.loads(data)
    else:
        raise ValueError(f"Unknown wire format {wire_format}")
    if not isinstance(value, dict):
        raise ValueError("Wire payload is not a mapping")
    return value


def make_state_delta(base: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    """JSON Merge Patch-like delta with an explicit delete list."""
    changed = {k: v for k, v in current.items() if k not in base or base[k] != v}
    deleted = sorted(set(base).difference(current))
    return {"set": changed, "delete": deleted}


def apply_state_delta(base: Mapping[str, Any], delta: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key in delta.get("delete", []):
        result.pop(key, None)
    result.update(delta.get("set", {}))
    return result


def fragment_message(message: AppMessage, mtu_bytes: int = 1200) -> tuple[Fragment, ...]:
    capacity = mtu_bytes - IPV4_HEADER_BYTES - UDP_HEADER_BYTES - FRAGMENT_HEADER_BYTES
    if capacity <= 0:
        raise ValueError("MTU is too small for protocol headers")
    wire_payload = message.wire_payload
    count = max(1, math.ceil(len(wire_payload) / capacity))
    if count > 65535:
        raise ValueError("Message requires more than 65,535 fragments")
    prefix = message.wire_payload_hash[:8]
    fragments = []
    for index in range(count):
        offset = index * capacity
        payload = wire_payload[offset : offset + capacity]
        fragment = Fragment(
            protocol_version=PROTOCOL_VERSION,
            message_id=message.message_id,
            fragment_index=index,
            fragment_count=count,
            payload_offset=offset,
            fragment_payload=payload,
            message_hash_prefix=prefix,
            crc32=zlib.crc32(payload) & 0xFFFFFFFF,
        )
        if fragment.ip_packet_bytes > mtu_bytes:
            raise AssertionError("IP fragmentation would occur")
        fragments.append(fragment)
    return tuple(fragments)


@dataclasses.dataclass
class ReassemblyBuffer:
    fragment_count: int
    message_hash_prefix: bytes
    message_id: int | None = None
    fragments: dict[int, Fragment] = dataclasses.field(default_factory=dict)

    def add(self, fragment: Fragment) -> bool:
        if fragment.protocol_version != PROTOCOL_VERSION:
            raise ValueError("Fragment protocol version mismatch")
        if fragment.fragment_count != self.fragment_count or fragment.message_hash_prefix != self.message_hash_prefix:
            raise ValueError("Fragment metadata mismatch")
        if self.message_id is None:
            self.message_id = fragment.message_id
        elif fragment.message_id != self.message_id:
            raise ValueError("Fragments from different messages cannot be combined")
        if not 0 <= fragment.fragment_index < self.fragment_count:
            raise ValueError("Fragment index out of range")
        new_start = fragment.payload_offset
        new_end = new_start + len(fragment.fragment_payload)
        for index, existing in self.fragments.items():
            if index == fragment.fragment_index:
                if existing != fragment:
                    raise ValueError("Conflicting duplicate fragment")
                return len(self.fragments) == self.fragment_count
            start = existing.payload_offset
            end = start + len(existing.fragment_payload)
            if max(start, new_start) < min(end, new_end):
                raise ValueError("Fragment payload ranges overlap")
        self.fragments.setdefault(fragment.fragment_index, fragment)
        return len(self.fragments) == self.fragment_count

    def assemble(self) -> bytes:
        if len(self.fragments) != self.fragment_count:
            raise ValueError("Message is incomplete")
        ordered = sorted(self.fragments.values(), key=lambda item: item.payload_offset)
        expected_offset = 0
        for fragment in ordered:
            if fragment.payload_offset != expected_offset:
                raise ValueError("Fragment offsets are not contiguous from zero")
            expected_offset += len(fragment.fragment_payload)
        data = b"".join(item.fragment_payload for item in ordered)
        if hashlib.sha256(data).digest()[:8] != self.message_hash_prefix:
            raise ValueError("Whole-message hash mismatch")
        return data


@dataclasses.dataclass
class ReceiverSemanticState:
    applied_seq: int = 0
    state: dict[str, Any] = dataclasses.field(default_factory=dict)
    last_applied_message_id: int | None = None
    last_update_us: int | None = None
    applied_messages: set[int] = dataclasses.field(default_factory=set)

    def apply(self, message: AppMessage, now_us: int, decoded: Mapping[str, Any]) -> ApplyStatus:
        if now_us > message.state_expiry_us:
            return ApplyStatus.EXPIRED
        if message.message_id in self.applied_messages:
            return ApplyStatus.DUPLICATE
        if message.state_kind is StateKind.NONE:
            self.applied_messages.add(message.message_id)
            return ApplyStatus.APPLIED
        result_seq = int(message.resulting_state_seq or 0)
        if result_seq <= self.applied_seq:
            return ApplyStatus.STALE
        if message.state_kind is StateKind.DELTA:
            if message.base_state_seq != self.applied_seq:
                return ApplyStatus.BASE_MISMATCH
            next_state = apply_state_delta(self.state, decoded)
        else:
            next_state = dict(decoded)
        self.state = next_state
        self.applied_seq = result_seq
        self.last_applied_message_id = message.message_id
        self.last_update_us = now_us
        self.applied_messages.add(message.message_id)
        return ApplyStatus.APPLIED


@dataclasses.dataclass
class SenderReceiverBelief:
    confirmed_seq: int = 0
    status: str = "known"
    feedback_arrival_us: int | None = None
    receiver_update_us: int | None = None
    latest_feedback_generated_us: int | None = None

    def on_application_ack(self, ack: ApplicationAck, now_us: int) -> None:
        if (
            ack.receiver_state_seq < self.confirmed_seq
            or (
                self.latest_feedback_generated_us is not None
                and ack.generated_us < self.latest_feedback_generated_us
            )
        ):
            return
        self.latest_feedback_generated_us = ack.generated_us
        self.confirmed_seq = ack.receiver_state_seq
        self.feedback_arrival_us = now_us
        if ack.semantic_state_message:
            if ack.state_update_us is not None and (
                self.receiver_update_us is None or ack.state_update_us >= self.receiver_update_us
            ):
                self.receiver_update_us = ack.state_update_us
            self.status = "known" if ack.status in {
                ApplyStatus.APPLIED, ApplyStatus.DUPLICATE, ApplyStatus.STALE,
            } else "uncertain"


def encode_image(image: Image.Image, codec: str, quality: int, scale: float = 1.0) -> bytes:
    if not 0 < scale <= 1.0 or not 1 <= quality <= 100:
        raise ValueError("Invalid image scale or quality")
    image = image.convert("RGB")
    if scale != 1.0:
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
    codec = codec.upper()
    if codec == "JPG":
        codec = "JPEG"
    if codec == "AVIF":
        # Pillow discovers AVIF through the separately installed plugin.  The
        # import is intentionally explicit so a scientific run cannot silently
        # omit the registered AVIF baseline just because plugin registration
        # has not happened elsewhere in the process.
        try:
            import pillow_avif  # noqa: F401
        except ImportError as exc:
            raise StudyGateError(
                "Pillow AVIF support is unavailable; install pillow-avif-plugin"
            ) from exc
    output = io.BytesIO()
    try:
        image.save(output, format=codec, quality=quality, optimize=False)
    except (KeyError, OSError) as exc:
        raise StudyGateError(f"Pillow codec {codec} is unavailable; install the required plugin") from exc
    return output.getvalue()


def decode_image(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB").copy()


VIDEO_GEOMETRY_POLICY_VERSION = "semuav.video-geometry.v1"
# ISO BMFF's UUID extension keeps the transmitted payload a regular MP4.
# This metadata is emitted only when yuv420p requires padding. Its complete
# 40 bytes are included in len(payload), wire fragmentation, and payload hashes.
_VIDEO_GEOMETRY_UUID = bytes.fromhex("6754b733a69854e6964024b713eff8cb")
_VIDEO_GEOMETRY_FIELDS = struct.Struct("!IIII")


def video_geometry_from_payload(payload: bytes) -> tuple[int, int, int] | None:
    """Read original width/height/frame count from the transmitted MP4 only.

    Even-sized and historical MP4s need no crop and may omit the extension.
    Invalid boxes and ambiguous/malformed geometry fail closed. In particular,
    a byte pattern inside encoded picture data is never treated as metadata.
    """

    position = 0
    geometry: tuple[int, int, int] | None = None
    while position < len(payload):
        if len(payload) - position < 8:
            raise StudyGateError("Transmitted video has a truncated MP4 box")
        size, kind = struct.unpack_from("!I4s", payload, position)
        header_size = 8
        if size == 1:
            if len(payload) - position < 16:
                raise StudyGateError("Transmitted video has a truncated extended MP4 box")
            size = struct.unpack_from("!Q", payload, position + 8)[0]
            header_size = 16
        elif size == 0:
            size = len(payload) - position
        if size < header_size or size > len(payload) - position:
            raise StudyGateError("Transmitted video has an invalid MP4 box size")
        if kind == b"uuid":
            if size < header_size + 16:
                raise StudyGateError("Transmitted video has a truncated MP4 UUID box")
            start = position + header_size
            if payload[start : start + 16] == _VIDEO_GEOMETRY_UUID:
                if geometry is not None or size != header_size + 16 + _VIDEO_GEOMETRY_FIELDS.size:
                    raise StudyGateError("Transmitted video geometry is duplicated or malformed")
                version, width, height, count = _VIDEO_GEOMETRY_FIELDS.unpack_from(payload, start + 16)
                if version != 1 or width < 1 or height < 1 or count < 2:
                    raise StudyGateError("Transmitted video geometry values/version are invalid")
                geometry = (width, height, count)
        position += size
    return geometry


def encode_video_clip(
    images: Sequence[Image.Image],
    codec: str = "libx264",
    crf: int = 28,
    fps: int = 5,
) -> bytes:
    """Encode real same-size frames with lossless right/bottom even padding.

    yuv420p requires even coded width/height. Padding preserves every source
    pixel and its coordinates; resizing to an even size would distort the
    registered detector ROI. Original dimensions travel in a UUID MP4 box so
    receivers remove padding before inference without consulting a manifest.
    """
    if len(images) < 2:
        raise ValueError("A video candidate requires at least two real consecutive frames")
    ffmpeg = resolve_ffmpeg_executable()
    if ffmpeg is None:
        raise StudyGateError(
            "FFmpeg is required for the practical video baseline; install the "
            "pinned imageio-ffmpeg dependency or provide ffmpeg on PATH"
        )
    if not 0 <= crf <= 51 or fps <= 0:
        raise ValueError("Invalid CRF or frame rate")
    first_size = images[0].size
    if any(image.size != first_size for image in images):
        raise StudyGateError("Video clip frame dimensions differ; silent resizing is not permitted")
    if any(not 0 < dimension <= 0xFFFFFFFF for dimension in first_size):
        raise ValueError("Video frame dimensions must be positive uint32 values")
    padded_size = tuple(dimension + dimension % 2 for dimension in first_size)
    requires_padding = padded_size != first_size
    stream = io.BytesIO()
    for image in images:
        frame = image.convert("RGB")
        if requires_padding:
            # Pad in RGB before chroma subsampling. Never discard an odd edge,
            # scale pixels, or move the top-left origin.
            padded = Image.new("RGB", padded_size, (0, 0, 0))
            padded.paste(frame, (0, 0))
            frame = padded
        frame.save(stream, format="PNG")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "image2pipe",
        "-vcodec", "png", "-framerate", str(fps), "-i", "pipe:0", "-an",
        "-c:v", codec, "-preset", "medium", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "frag_keyframe+empty_moov",
        "-f", "mp4", "pipe:1",
    ]
    process = subprocess.run(command, input=stream.getvalue(), capture_output=True)
    if process.returncode:
        raise StudyGateError(f"FFmpeg video encoding failed: {process.stderr.decode(errors='replace')[:500]}")
    payload = process.stdout
    if not payload:
        raise StudyGateError("FFmpeg video encoder returned an empty payload")
    if requires_padding:
        fields = _VIDEO_GEOMETRY_FIELDS.pack(1, *first_size, len(images))
        payload += struct.pack("!I4s", 24 + len(fields), b"uuid") + _VIDEO_GEOMETRY_UUID + fields
    return payload


@dataclasses.dataclass(frozen=True)
class QuantizedTensor:
    bits: int
    shape: tuple[int, ...]
    scale: float
    zero_point: int
    data: bytes

    @property
    def wire_bytes(self) -> int:
        return len(serialize_quantized_tensor(self))


_QUANTIZED_TENSOR_MAGIC = b"QTF1"
_QUANTIZED_TENSOR_HEADER = struct.Struct("!4sBBBBd")


def serialize_quantized_tensor(tensor: QuantizedTensor) -> bytes:
    """Serialize a quantized feature tensor without hidden receiver metadata.

    The header carries a magic/version tag, bit width, rank, zero point,
    reserved byte, IEEE-754 scale, and every uint32 dimension.  The packed
    quantized values follow.  This makes the exact bytes counted by Experiment
    2 sufficient for receiver-side reconstruction.
    """

    if tensor.bits not in {4, 8}:
        raise ValueError("Only INT4 and INT8 quantized tensors can be serialized")
    shape = tuple(int(value) for value in tensor.shape)
    if len(shape) > 255 or any(value <= 0 or value > 0xFFFFFFFF for value in shape):
        raise ValueError("Quantized tensor dimensions must be positive uint32 values")
    scale = float(tensor.scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Quantized tensor scale must be finite and positive")
    zero_point = int(tensor.zero_point)
    if not 0 <= zero_point <= (1 << tensor.bits) - 1:
        raise ValueError("Quantized tensor zero point is outside its bit range")
    count = int(np.prod(shape, dtype=np.int64)) if shape else 1
    expected_data_bytes = math.ceil(count * tensor.bits / 8)
    if len(tensor.data) != expected_data_bytes:
        raise ValueError("Quantized tensor data length does not match shape and bit width")
    header = _QUANTIZED_TENSOR_HEADER.pack(
        _QUANTIZED_TENSOR_MAGIC,
        tensor.bits,
        len(shape),
        zero_point,
        0,
        scale,
    )
    dimensions = b"".join(struct.pack("!I", value) for value in shape)
    return header + dimensions + tensor.data


def deserialize_quantized_tensor(payload: bytes) -> QuantizedTensor:
    """Decode and validate the canonical quantized-feature wire payload."""

    payload = bytes(payload)
    if len(payload) < _QUANTIZED_TENSOR_HEADER.size:
        raise ValueError("Quantized tensor payload is truncated")
    magic, bits, rank, zero_point, reserved, scale = _QUANTIZED_TENSOR_HEADER.unpack_from(payload)
    if magic != _QUANTIZED_TENSOR_MAGIC:
        raise ValueError("Quantized tensor payload magic/version is invalid")
    if reserved != 0:
        raise ValueError("Quantized tensor reserved header byte is nonzero")
    if bits not in {4, 8}:
        raise ValueError("Quantized tensor payload has an unsupported bit width")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Quantized tensor payload scale is invalid")
    if not 0 <= zero_point <= (1 << bits) - 1:
        raise ValueError("Quantized tensor payload zero point is invalid")
    dimension_end = _QUANTIZED_TENSOR_HEADER.size + 4 * rank
    if len(payload) < dimension_end:
        raise ValueError("Quantized tensor dimension header is truncated")
    shape = tuple(
        struct.unpack_from("!I", payload, _QUANTIZED_TENSOR_HEADER.size + 4 * index)[0]
        for index in range(rank)
    )
    if any(value <= 0 for value in shape):
        raise ValueError("Quantized tensor payload contains an empty dimension")
    count = int(np.prod(shape, dtype=np.int64)) if shape else 1
    expected_data_bytes = math.ceil(count * bits / 8)
    data = payload[dimension_end:]
    if len(data) != expected_data_bytes:
        raise ValueError("Quantized tensor payload data length does not match its header")
    return QuantizedTensor(bits, shape, float(scale), int(zero_point), data)


def quantize_tensor(array: np.ndarray, bits: int) -> QuantizedTensor:
    if bits not in {4, 8}:
        raise ValueError("Only registered INT4 and INT8 feature tensors are supported")
    values = np.asarray(array, dtype=np.float32)
    qmin, qmax = 0, (1 << bits) - 1
    minimum = float(values.min(initial=0.0))
    maximum = float(values.max(initial=0.0))
    scale = (maximum - minimum) / qmax if maximum > minimum else 1.0
    zero_point = int(np.clip(round(-minimum / scale), qmin, qmax))
    quantized = np.clip(np.rint(values / scale + zero_point), qmin, qmax).astype(np.uint8).ravel()
    if bits == 4:
        if len(quantized) % 2:
            quantized = np.concatenate([quantized, np.zeros(1, dtype=np.uint8)])
        packed = quantized[0::2] | (quantized[1::2] << 4)
        data = packed.tobytes()
    else:
        data = quantized.tobytes()
    return QuantizedTensor(bits, tuple(values.shape), scale, zero_point, data)


def dequantize_tensor(tensor: QuantizedTensor) -> np.ndarray:
    count = int(np.prod(tensor.shape))
    packed = np.frombuffer(tensor.data, dtype=np.uint8)
    if tensor.bits == 4:
        values = np.empty(len(packed) * 2, dtype=np.uint8)
        values[0::2] = packed & 0x0F
        values[1::2] = packed >> 4
        values = values[:count]
    elif tensor.bits == 8:
        values = packed[:count]
    else:
        raise ValueError("Unsupported quantization")
    return ((values.astype(np.float32) - tensor.zero_point) * tensor.scale).reshape(tensor.shape)


def representation_wire_bytes(payload: bytes, mtu_bytes: int = 1200) -> int:
    dummy = AppMessage("measure", "measure", 1, PayloadMode.L0_HEARTBEAT, 0, 0, 1, 1, 0, payload)
    return sum(fragment.ip_packet_bytes for fragment in fragment_message(dummy, mtu_bytes))
