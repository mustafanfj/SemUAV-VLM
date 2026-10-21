"""Frozen native-receiver and split-feature cache producers.

The Experiment 2 scientific path consumes receiver decisions and resource
measurements from frozen caches.  This module creates those caches from the
*transmitted representation bytes*.  Receiver callables never receive the
manifest row or its gold action, which prevents an evaluation label from
silently becoming a candidate-generation input.

The measurement runner is deliberately supplied by the caller.  A scientific
run can therefore wrap the operations with the registered wall-clock and
energy meter, while unit tests can use a deterministic CPU-only runner.
Mission profile, ROI, capture time, and authoritative flight telemetry are
supplied through a separately frozen, gold-free control table and are
delivered directly to receiver callables.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from .protocol import (
    PayloadMode,
    QuantizedTensor,
    decode_image,
    decode_mapping,
    deserialize_quantized_tensor,
    dequantize_tensor,
    encode_image,
    encode_mapping,
    encode_video_clip,
    quantize_tensor,
    serialize_quantized_tensor,
    semantic_level_view,
    validate_semantic_packet,
    video_geometry_from_payload,
)
from .registry import StudyGateError, atomic_write_frame, atomic_write_json, hash_file, hash_object
from .runtime import resolve_ffmpeg_executable


REPRESENTATION_COST_SCOPE = "representation_encoding_only_excludes_perception"
REGISTERED_ACTIONS = frozenset({"NO_ACTION", "REVIEW", "RESPOND"})
REGISTERED_MISSION_PROFILES = frozenset(
    {"vulnerable_road_user", "heavy_vehicle", "general_mobility"}
)
CONTROL_CONTEXT_SCHEMA_VERSION = "semuav.visdrone-receiver-control.v2"
NATIVE_RECEIVER_FUSION_RECEIPT_SCHEMA = (
    "semuav.visdrone-native-fusion-receipt.v1"
)


@dataclasses.dataclass(frozen=True)
class ReceiverControlContext:
    """Gold-free control state pre-shared by sender and receiver.

    This mission configuration is neither inferred from ``sample_id`` nor
    read from the evaluation manifest during inference.  It is supplied as a
    separately frozen input and its canonical digest is bound into every
    receiver-cache row.
    """

    mission_profile: str
    registered_roi_xyxy: tuple[float, float, float, float]
    capture_time_us: int
    gps_lat: float
    gps_lon: float
    altitude_m: float
    yaw_deg: float
    schema_version: str = CONTROL_CONTEXT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONTROL_CONTEXT_SCHEMA_VERSION:
            raise ValueError("Receiver control-context schema version is not registered")
        if self.mission_profile not in REGISTERED_MISSION_PROFILES:
            raise ValueError(
                "Receiver mission_profile must be one of the three registered VisDrone profiles"
            )
        roi = tuple(self.registered_roi_xyxy)
        if len(roi) != 4 or any(
            isinstance(value, bool) or not math.isfinite(float(value))
            for value in roi
        ):
            raise ValueError("registered_roi_xyxy must contain four finite numbers")
        x1, y1, x2, y2 = map(float, roi)
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError("registered_roi_xyxy must be an ordered normalized rectangle")
        object.__setattr__(self, "registered_roi_xyxy", (x1, y1, x2, y2))
        if (
            isinstance(self.capture_time_us, bool)
            or not isinstance(self.capture_time_us, (int, np.integer))
            or int(self.capture_time_us) < 0
        ):
            raise ValueError("Receiver capture_time_us must be a nonnegative integer")
        object.__setattr__(self, "capture_time_us", int(self.capture_time_us))
        telemetry: dict[str, float] = {}
        for name in ("gps_lat", "gps_lon", "altitude_m", "yaw_deg"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(f"Receiver {name} must be a finite number")
            try:
                telemetry[name] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Receiver {name} must be a finite number") from exc
            if not math.isfinite(telemetry[name]):
                raise ValueError(f"Receiver {name} must be a finite number")
        if not -90.0 <= telemetry["gps_lat"] <= 90.0:
            raise ValueError("Receiver gps_lat must lie within [-90, 90]")
        if not -180.0 <= telemetry["gps_lon"] <= 180.0:
            raise ValueError("Receiver gps_lon must lie within [-180, 180]")
        if telemetry["altitude_m"] < 0.0:
            raise ValueError("Receiver altitude_m must be nonnegative")
        if not 0.0 <= telemetry["yaw_deg"] < 360.0:
            raise ValueError("Receiver yaw_deg must lie within [0, 360)")
        for name, value in telemetry.items():
            object.__setattr__(self, name, value)

    @property
    def telemetry(self) -> dict[str, Any]:
        """Return the authoritative, outcome-independent receiver telemetry."""

        return {
            "gps_lat": self.gps_lat,
            "gps_lon": self.gps_lon,
            "altitude_m": self.altitude_m,
            "yaw_deg": self.yaw_deg,
            "mission_profile": self.mission_profile,
            "registered_roi_xyxy": list(self.registered_roi_xyxy),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mission_profile": self.mission_profile,
            "registered_roi_xyxy": list(self.registered_roi_xyxy),
            "capture_time_us": self.capture_time_us,
            "authoritative_telemetry": self.telemetry,
        }

    @property
    def sha256(self) -> str:
        return hash_object(self.as_dict())


def uniform_receiver_control_contexts(
    sample_ids: Sequence[str],
    *,
    mission_profile: str,
    registered_roi_xyxy: Sequence[float],
    capture_time_us: int,
    gps_lat: float,
    gps_lon: float,
    altitude_m: float,
    yaw_deg: float,
) -> dict[str, ReceiverControlContext]:
    """Create an explicit, uniform pre-shared control table for one run."""

    identifiers = tuple(str(value).strip() for value in sample_ids)
    if not identifiers or any(not value for value in identifiers):
        raise ValueError("Control-context sample IDs must be nonempty")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Control-context sample IDs must be unique")
    context = ReceiverControlContext(
        mission_profile=str(mission_profile),
        registered_roi_xyxy=tuple(float(value) for value in registered_roi_xyxy),
        capture_time_us=capture_time_us,
        gps_lat=gps_lat,
        gps_lon=gps_lon,
        altitude_m=altitude_m,
        yaw_deg=yaw_deg,
    )
    return {sample_id: context for sample_id in identifiers}


def _control_context_lookup(
    manifest: pd.DataFrame,
    control_contexts: Mapping[str, ReceiverControlContext],
) -> dict[str, ReceiverControlContext]:
    """Validate exact sample coverage without consulting outcome columns."""

    if not isinstance(control_contexts, Mapping) or not control_contexts:
        raise StudyGateError("A nonempty pre-shared receiver control-context table is required")
    manifest_ids = set(manifest["sample_id"].astype(str))
    lookup: dict[str, ReceiverControlContext] = {}
    for sample_id, context in control_contexts.items():
        identifier = str(sample_id)
        if not identifier or identifier in lookup:
            raise StudyGateError(
                "Receiver control-context sample IDs must be nonempty and unique after normalization"
            )
        if not isinstance(context, ReceiverControlContext):
            raise TypeError(
                f"Control context for {identifier!r} must be ReceiverControlContext"
            )
        lookup[identifier] = context
    if missing := sorted(manifest_ids - set(lookup)):
        raise StudyGateError(
            f"Receiver control contexts miss {len(missing)} sample(s); examples: {missing[:5]}"
        )
    if extras := sorted(set(lookup) - manifest_ids):
        raise StudyGateError(
            f"Receiver control contexts have {len(extras)} out-of-scope sample(s); examples: {extras[:5]}"
        )
    return lookup


@dataclasses.dataclass(frozen=True)
class MeasurementContext:
    """Identity supplied to a caller-owned latency/energy measurement runner."""

    sample_id: str
    variant: str
    stage: str


@dataclasses.dataclass(frozen=True)
class MeasuredValue:
    """An operation result with measurements from the registered meter."""

    value: Any
    latency_ms: float
    energy_j: float


MeasurementRunner = Callable[[MeasurementContext, Callable[[], Any]], MeasuredValue]


@dataclasses.dataclass(frozen=True)
class DecodedNativePayload:
    """Receiver input reconstructed exclusively from transmitted bytes."""

    sample_id: str
    variant: str
    source_model: str
    media_kind: str
    payload_sha256: str
    control_context: ReceiverControlContext
    control_context_sha256: str
    semantic_packet: Mapping[str, Any] | None = None
    frames: tuple[Image.Image, ...] = ()
    transmitted_roi_xyxy: tuple[float, float, float, float] | None = None

    @property
    def mission_profile(self) -> str:
        return self.control_context.mission_profile

    @property
    def registered_roi_xyxy(self) -> tuple[float, float, float, float]:
        return self.control_context.registered_roi_xyxy


NativeReceiver = Callable[[DecodedNativePayload], str | Mapping[str, Any]]


@dataclasses.dataclass(frozen=True)
class NativeReceiverDesign:
    """Frozen Experiment 2 representation variants."""

    semantic_source_model: str
    visual_source_model: str = "ground_visual_receiver"
    image_codecs: tuple[str, ...] = ("JPEG", "WEBP", "AVIF")
    image_qualities: tuple[int, ...] = (25, 50, 80)
    image_scales: tuple[float, ...] = (0.5, 1.0)
    video_crfs: tuple[int, ...] = (23, 35)
    video_fps: int = 5

    def __post_init__(self) -> None:
        if not self.semantic_source_model.strip() or not self.visual_source_model.strip():
            raise ValueError("Native receiver source-model identifiers must be populated")
        codecs = tuple(str(value).upper() for value in self.image_codecs)
        if not codecs or len(set(codecs)) != len(codecs):
            raise ValueError("image_codecs must be a nonempty unique sequence")
        if any(not 1 <= int(value) <= 100 for value in self.image_qualities):
            raise ValueError("image_qualities must lie within [1, 100]")
        if any(not 0 < float(value) <= 1 for value in self.image_scales):
            raise ValueError("image_scales must lie within (0, 1]")
        if any(not 0 <= int(value) <= 51 for value in self.video_crfs):
            raise ValueError("video_crfs must lie within [0, 51]")
        if int(self.video_fps) <= 0:
            raise ValueError("video_fps must be positive")


@dataclasses.dataclass(frozen=True)
class FeatureEncoderInput:
    """Gold-free input to the registered sender-side split encoder."""

    sample_id: str
    image: Image.Image
    encoder_revision: str
    split_layer: str


@dataclasses.dataclass(frozen=True)
class FeatureHeadInput:
    """Gold-free receiver input reconstructed from the quantized payload."""

    sample_id: str
    variant: str
    features: np.ndarray
    quantized: QuantizedTensor
    encoder_revision: str
    split_layer: str
    receiver_head_revision: str
    control_context: ReceiverControlContext
    control_context_sha256: str

    @property
    def mission_profile(self) -> str:
        return self.control_context.mission_profile

    @property
    def registered_roi_xyxy(self) -> tuple[float, float, float, float]:
        return self.control_context.registered_roi_xyxy


FeatureEncoder = Callable[[FeatureEncoderInput], np.ndarray]
FeatureHead = Callable[[FeatureHeadInput], str | Mapping[str, Any]]


@dataclasses.dataclass(frozen=True)
class FeatureCacheDesign:
    encoder_revision: str
    split_layer: str
    receiver_head_revision: str
    sender_hardware_id: str
    receiver_hardware_id: str
    quantization_bits: tuple[int, ...] = (8, 4)
    variant_prefix: str = "registered_feature"

    def __post_init__(self) -> None:
        required = (
            self.encoder_revision,
            self.split_layer,
            self.receiver_head_revision,
            self.sender_hardware_id,
            self.receiver_hardware_id,
            self.variant_prefix,
        )
        if any(not str(value).strip() for value in required):
            raise ValueError("Feature cache revisions, hardware, and variant prefix must be populated")
        bits = tuple(int(value) for value in self.quantization_bits)
        if not bits or set(bits).difference({4, 8}) or len(set(bits)) != len(bits):
            raise ValueError("quantization_bits must be a nonempty unique subset of {4, 8}")


@dataclasses.dataclass(frozen=True)
class _NativePayloadPlan:
    sample_id: str
    variant: str
    source_model: str
    media_kind: str
    encode: Callable[[], bytes]


def _require_manifest(manifest: pd.DataFrame, *, require_clip_paths: bool = True) -> None:
    required = {"sample_id", "image_path"}
    if require_clip_paths:
        required.add("clip_paths_json")
    if missing := required.difference(manifest.columns):
        raise ValueError(f"Native receiver manifest missing columns: {sorted(missing)}")
    if manifest.empty:
        raise StudyGateError("Native receiver manifest is empty")
    identifiers = manifest["sample_id"].astype(str)
    if identifiers.eq("").any() or identifiers.duplicated().any():
        raise StudyGateError("Native receiver manifest sample_id values must be nonempty and unique")


def _semantic_lookup(
    manifest: pd.DataFrame,
    semantic_packet_cache: pd.DataFrame,
    semantic_source_model: str,
) -> dict[str, Mapping[str, Any]]:
    required = {"sample_id", "packet_json", "source_model"}
    if missing := required.difference(semantic_packet_cache.columns):
        raise ValueError(f"Semantic packet cache missing columns: {sorted(missing)}")
    if semantic_packet_cache.empty:
        raise StudyGateError("Semantic packet cache is empty")
    records: dict[str, Mapping[str, Any]] = {}
    for sample_id, group in semantic_packet_cache.groupby(
        semantic_packet_cache["sample_id"].astype(str), sort=False,
    ):
        if len(group) > 1 and group[["packet_json", "source_model"]].astype(str).nunique().gt(1).any():
            raise StudyGateError(f"Conflicting semantic packets exist for sample {sample_id}")
        row = group.iloc[0]
        if str(row["source_model"]) != semantic_source_model:
            raise StudyGateError(
                f"Semantic source mismatch for {sample_id}: {row['source_model']} != {semantic_source_model}"
            )
        try:
            packet = json.loads(str(row["packet_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise StudyGateError(f"Semantic packet JSON is invalid for {sample_id}") from exc
        packet = validate_semantic_packet(packet, require_transmission_metadata=True)
        if str(packet["observation_id"]) != str(sample_id):
            raise StudyGateError(f"Semantic packet observation_id mismatch for {sample_id}")
        if str(packet["provenance"]["source_model"]) != semantic_source_model:
            raise StudyGateError(f"Semantic packet provenance source mismatch for {sample_id}")
        records[str(sample_id)] = packet
    manifest_ids = set(manifest["sample_id"].astype(str))
    cache_ids = set(records)
    if missing := sorted(manifest_ids - cache_ids):
        raise StudyGateError(
            f"Semantic packet cache misses {len(missing)} manifest sample(s); examples: {missing[:5]}"
        )
    if extras := sorted(cache_ids - manifest_ids):
        raise StudyGateError(
            f"Semantic packet cache has {len(extras)} out-of-scope sample(s); examples: {extras[:5]}"
        )
    return records


def _parse_clip_paths(value: Any, sample_id: str) -> tuple[str, ...]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ()
    try:
        parsed = json.loads(str(value)) if isinstance(value, str) else value
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StudyGateError(f"clip_paths_json is invalid for {sample_id}") from exc
    if not isinstance(parsed, list) or any(not isinstance(path, str) or not path for path in parsed):
        raise StudyGateError(f"clip_paths_json must be a JSON list of paths for {sample_id}")
    return tuple(parsed)


def _native_payload_plans(
    manifest: pd.DataFrame,
    semantic_packet_cache: pd.DataFrame,
    design: NativeReceiverDesign,
) -> Iterator[_NativePayloadPlan]:
    """Yield exact payload operations used by Experiment 2."""

    _require_manifest(manifest)
    packets = _semantic_lookup(manifest, semantic_packet_cache, design.semantic_source_model)
    # Local import avoids making experiments.py depend on this producer module.
    from .experiments import select_grounded_semantic_roi

    for item in manifest.itertuples(index=False):
        sample_id = str(item.sample_id)
        image_path = Path(str(item.image_path))
        if not image_path.is_file():
            raise StudyGateError(f"Image is missing for {sample_id}: {image_path}")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB").copy()
        packet = packets[sample_id]

        for level, prefix in (
            (PayloadMode.L0_HEARTBEAT, "l0"),
            (PayloadMode.L1_ALERT, "l1"),
            (PayloadMode.L2_SEMANTICS, "l2"),
        ):
            content = packet if level is PayloadMode.L2_SEMANTICS else semantic_level_view(packet, level)

            def encode_semantic(content: Mapping[str, Any] = content) -> bytes:
                return encode_mapping(content, scientific=True)[0]

            # Full-profile encoding is CBOR; the explicit level prefix is
            # necessary because receiver-cache identity includes variant but
            # not mode.
            yield _NativePayloadPlan(
                sample_id, f"{prefix}_cbor", design.semantic_source_model,
                "semantic", encode_semantic,
            )

        normalized_roi = select_grounded_semantic_roi(packet, image.size)
        x1, y1, x2, y2 = normalized_roi
        roi_box = (
            int(x1 * image.width),
            int(y1 * image.height),
            max(int(x1 * image.width) + 1, int(x2 * image.width)),
            max(int(y1 * image.height) + 1, int(y2 * image.height)),
        )
        roi = image.crop(roi_box).copy()

        def encode_l3(
            packet: Mapping[str, Any] = packet,
            roi: Image.Image = roi,
            normalized_roi: Sequence[float] = tuple(normalized_roi),
        ) -> bytes:
            roi_bytes = encode_image(roi, "JPEG", 50, 1.0)
            l3_packet = json.loads(json.dumps(packet))
            l3_packet.setdefault("evidence", []).append({
                "evidence_id": "transmitted-roi-0",
                "linked_claims": ["event", "recommended_action"],
                "bbox_normalized_xyxy": [float(value) for value in normalized_roi],
                "codec": "jpeg",
                "integrity_sha256": hashlib.sha256(roi_bytes).hexdigest(),
            })
            validate_semantic_packet(l3_packet, require_transmission_metadata=True)
            semantics, _ = encode_mapping(l3_packet, scientific=True)
            return len(semantics).to_bytes(4, "big") + semantics + roi_bytes

        yield _NativePayloadPlan(
            sample_id, "cbor+jpeg_q50_roi", design.semantic_source_model,
            "semantic_roi", encode_l3,
        )

        for codec in design.image_codecs:
            normalized_codec = str(codec).upper()
            for quality in design.image_qualities:
                for scale in design.image_scales:
                    variant = f"{normalized_codec.lower()}_q{int(quality)}_s{float(scale)}"

                    def encode_still(
                        image: Image.Image = image,
                        codec: str = normalized_codec,
                        quality: int = int(quality),
                        scale: float = float(scale),
                    ) -> bytes:
                        return encode_image(image, codec, quality, scale)

                    yield _NativePayloadPlan(
                        sample_id, variant, design.visual_source_model,
                        "image", encode_still,
                    )

        def encode_lossless(image: Image.Image = image) -> bytes:
            output = io.BytesIO()
            image.save(output, format="PNG")
            return output.getvalue()

        yield _NativePayloadPlan(
            sample_id, "lossless_png_reference", design.visual_source_model,
            "image", encode_lossless,
        )

        clip_paths = _parse_clip_paths(getattr(item, "clip_paths_json"), sample_id)
        if len(clip_paths) >= 2:
            clip_images: list[Image.Image] = []
            for clip_path_text in clip_paths:
                clip_path = Path(clip_path_text)
                if not clip_path.is_file():
                    raise StudyGateError(f"Clip frame is missing for {sample_id}: {clip_path}")
                with Image.open(clip_path) as opened:
                    clip_images.append(opened.convert("RGB").copy())
            frozen_clip = tuple(clip_images)
            for crf in design.video_crfs:
                variant = f"h264_crf{int(crf)}_fps{int(design.video_fps)}"

                def encode_video(
                    images: tuple[Image.Image, ...] = frozen_clip,
                    crf: int = int(crf),
                    fps: int = int(design.video_fps),
                ) -> bytes:
                    return encode_video_clip(images, "libx264", crf, fps)

                yield _NativePayloadPlan(
                    sample_id, variant, design.visual_source_model,
                    "video", encode_video,
                )


def _decode_video(payload: bytes, temporary_root: str | Path | None) -> tuple[Image.Image, ...]:
    geometry = video_geometry_from_payload(payload)
    ffmpeg = resolve_ffmpeg_executable()
    if ffmpeg is None:
        raise StudyGateError("FFmpeg is required to decode the transmitted video baseline")
    temp_parent = None if temporary_root is None else str(Path(temporary_root))
    if temp_parent is not None:
        Path(temp_parent).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="semuav-video-decode-", dir=temp_parent) as directory:
        root = Path(directory)
        input_path = root / "payload.mp4"
        input_path.write_bytes(payload)
        output_pattern = root / "frame-%06d.png"
        process = subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(input_path),
                "-vsync", "0", str(output_pattern),
            ],
            check=False,
            capture_output=True,
        )
        if process.returncode:
            raise StudyGateError(
                "FFmpeg video decoding failed: "
                + process.stderr.decode(errors="replace")[:500]
            )
        frames = []
        for frame_path in sorted(root.glob("frame-*.png")):
            with Image.open(frame_path) as image:
                frames.append(image.convert("RGB").copy())
        if len(frames) < 2:
            raise StudyGateError("Decoded video baseline contains fewer than two frames")
        if geometry is not None:
            width, height, count = geometry
            padded_size = (width + width % 2, height + height % 2)
            if len(frames) != count or any(frame.size != padded_size for frame in frames):
                raise StudyGateError("Decoded video dimensions/frame count disagree with transmitted geometry")
            # Restore the full original coordinate plane before detector/ROI
            # inference. The geometry comes only from the counted wire bytes.
            frames = [frame.crop((0, 0, width, height)) for frame in frames]
        return tuple(frames)


def _decode_native_payload(
    plan: _NativePayloadPlan,
    payload: bytes,
    temporary_root: str | Path | None,
    control_context: ReceiverControlContext,
) -> DecodedNativePayload:
    semantic: Mapping[str, Any] | None = None
    frames: tuple[Image.Image, ...] = ()
    transmitted_roi_xyxy: tuple[float, float, float, float] | None = None
    if plan.media_kind == "semantic":
        semantic = decode_mapping(payload, "cbor")
    elif plan.media_kind == "semantic_roi":
        if len(payload) < 5:
            raise StudyGateError("Transmitted L3 payload is truncated")
        semantic_length = int.from_bytes(payload[:4], "big")
        if semantic_length <= 0 or 4 + semantic_length >= len(payload):
            raise StudyGateError("Transmitted L3 semantic prefix length is invalid")
        semantic = decode_mapping(payload[4 : 4 + semantic_length], "cbor")
        frames = (decode_image(payload[4 + semantic_length :]),)
        evidence = semantic.get("evidence", [])
        transmitted_evidence = [
            item for item in evidence
            if isinstance(item, Mapping)
            and str(item.get("evidence_id", "")) == "transmitted-roi-0"
        ] if isinstance(evidence, list) else []
        if len(transmitted_evidence) != 1:
            raise StudyGateError("Decoded L3 payload lacks transmitted ROI evidence")
        transmitted = transmitted_evidence[0]
        expected_hash = str(transmitted.get("integrity_sha256", ""))
        actual_hash = hashlib.sha256(payload[4 + semantic_length :]).hexdigest()
        if expected_hash != actual_hash:
            raise StudyGateError("Decoded L3 ROI integrity hash does not match its semantic prefix")
        box = transmitted.get("bbox_normalized_xyxy")
        if (
            not isinstance(box, (list, tuple)) or len(box) != 4
            or any(not math.isfinite(float(value)) for value in box)
        ):
            raise StudyGateError("Decoded L3 ROI evidence has invalid normalized geometry")
        values = tuple(float(value) for value in box)
        if not (
            0.0 <= values[0] < values[2] <= 1.0
            and 0.0 <= values[1] < values[3] <= 1.0
        ):
            raise StudyGateError("Decoded L3 ROI evidence has invalid normalized geometry")
        transmitted_roi_xyxy = values
    elif plan.media_kind == "image":
        frames = (decode_image(payload),)
    elif plan.media_kind == "video":
        frames = _decode_video(payload, temporary_root)
    else:  # pragma: no cover - internal construction prevents this branch
        raise AssertionError(f"Unknown native media kind {plan.media_kind}")
    return DecodedNativePayload(
        sample_id=plan.sample_id,
        variant=plan.variant,
        source_model=plan.source_model,
        media_kind=plan.media_kind,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        control_context=control_context,
        control_context_sha256=control_context.sha256,
        semantic_packet=semantic,
        frames=frames,
        transmitted_roi_xyxy=transmitted_roi_xyxy,
    )


def _validated_fusion_receipt(receipt: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return canonical receipt JSON, version, and digest after strict checks."""

    if not isinstance(receipt, Mapping):
        raise StudyGateError("Native receiver fusion receipt must be a mapping")
    payload = dict(receipt)
    if payload.get("schema_version") != NATIVE_RECEIVER_FUSION_RECEIPT_SCHEMA:
        raise StudyGateError("Native receiver fusion receipt schema is not registered")
    version = str(payload.get("policy_version", "")).strip()
    detector_revision = str(payload.get("detector_revision", "")).strip()
    if not version or not detector_revision:
        raise StudyGateError("Native receiver fusion receipt lacks policy/detector revision")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return encoded, version, hash_object(payload)


def _coerce_measured(value: Any, context: MeasurementContext) -> MeasuredValue:
    if not isinstance(value, MeasuredValue):
        raise TypeError(
            f"Measurement runner must return MeasuredValue for {context.sample_id}/{context.variant}/{context.stage}"
        )
    latency = float(value.latency_ms)
    energy = float(value.energy_j)
    if not math.isfinite(latency) or latency < 0:
        raise StudyGateError(f"Non-finite or negative latency for {context}")
    if not math.isfinite(energy) or energy < 0:
        raise StudyGateError(f"Non-finite or negative energy for {context}")
    # Preserve optional repeated-measurement receipts supplied by a stricter
    # MeasuredValue subclass. Reconstructing the base class would silently
    # discard repetition hashes and the frozen comparison policy.
    return value


def _measurement_metadata_json(value: MeasuredValue) -> str:
    metadata = getattr(value, "measurement_metadata", {})
    if not isinstance(metadata, Mapping):
        raise StudyGateError("Measurement metadata must be a mapping")
    return json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"))


def _coerce_action(value: str | Mapping[str, Any], *, sample_id: str, variant: str) -> str:
    if isinstance(value, Mapping):
        action = value.get("receiver_action", value.get("recommended_action"))
    else:
        action = value
    action_text = str(action).strip() if action is not None else ""
    if action_text not in REGISTERED_ACTIONS:
        raise StudyGateError(
            f"Receiver returned an unregistered action for {sample_id}/{variant}: {action_text!r}"
        )
    return action_text


def build_native_receiver_cache(
    manifest: pd.DataFrame,
    semantic_packet_cache: pd.DataFrame,
    *,
    design: NativeReceiverDesign,
    control_contexts: Mapping[str, ReceiverControlContext],
    receiver: NativeReceiver,
    measurement_runner: MeasurementRunner,
    sender_hardware_id: str,
    receiver_hardware_id: str,
    sender_revision: str,
    receiver_revision: str,
    receiver_fusion_receipt: Mapping[str, Any],
    temporary_root: str | Path | None = None,
    output_path: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
    checkpoint_context_sha256: str | None = None,
) -> pd.DataFrame:
    """Build the complete native receiver cache required by Experiment 2.

    Sender latency/energy cover representation encoding only.  Receiver
    latency/energy cover wire decoding plus the supplied native receiver.  The
    receiver input has no gold label or manifest row.
    It does receive the explicit pre-shared mission profile and registered ROI.

    Optional atomic row checkpoints preserve completed measurements after an
    interrupted build. The caller must bind the sealed run/config/environment
    and measurement settings in ``checkpoint_context_sha256``. Producer source,
    design, hardware/revision and fusion identities are additionally bound here;
    resumed rows are re-encoded once to verify their exact wire bytes. Historical
    cache rows are never imported into a different checkpoint context.
    """

    provenance = (
        sender_hardware_id,
        receiver_hardware_id,
        sender_revision,
        receiver_revision,
    )
    if any(not str(value).strip() for value in provenance):
        raise ValueError("Native receiver sender/receiver hardware and revisions must be populated")
    fusion_json, fusion_version, fusion_sha256 = _validated_fusion_receipt(
        receiver_fusion_receipt,
    )
    if str(receiver_fusion_receipt.get("detector_revision")) != str(receiver_revision):
        raise StudyGateError(
            "Native receiver fusion receipt and receiver_revision disagree"
        )
    context_lookup = _control_context_lookup(manifest, control_contexts)
    checkpoint_directory: Path | None = None
    checkpoint_identity: str | None = None
    if (checkpoint_root is None) != (checkpoint_context_sha256 is None):
        raise ValueError("Native checkpoints require both a directory and sealed context digest")
    if checkpoint_root is not None:
        digest = str(checkpoint_context_sha256)
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("Native checkpoint context must be a lowercase SHA-256 digest")
        module_root = Path(__file__).resolve().parent
        checkpoint_identity = hash_object({
            "schema_version": "semuav.native-receiver-checkpoint-context.v1",
            "sealed_run_context_sha256": digest,
            "design": dataclasses.asdict(design),
            "source_sha256": {
                name: hash_file(module_root / name)
                for name in ("protocol.py", "visdrone_receivers.py", "visdrone_receiver_runtime.py")
            },
            "sender_hardware_id": sender_hardware_id,
            "receiver_hardware_id": receiver_hardware_id,
            "sender_revision": sender_revision,
            "receiver_revision": receiver_revision,
            "receiver_fusion_policy_sha256": fusion_sha256,
        })
        checkpoint_directory = Path(checkpoint_root) / checkpoint_identity
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for plan in _native_payload_plans(manifest, semantic_packet_cache, design):
        control_context = context_lookup[plan.sample_id]
        fixed_row = {
            "sample_id": plan.sample_id,
            "variant": plan.variant,
            "source_model": plan.source_model,
            "media_kind": plan.media_kind,
            "hardware_id": receiver_hardware_id,
            "sender_hardware_id": sender_hardware_id,
            "receiver_hardware_id": receiver_hardware_id,
            "sender_revision": sender_revision,
            "receiver_revision": receiver_revision,
            "receiver_fusion_policy_json": fusion_json,
            "receiver_fusion_policy_version": fusion_version,
            "receiver_fusion_policy_sha256": fusion_sha256,
            "control_context_sha256": control_context.sha256,
            "cost_scope": REPRESENTATION_COST_SCOPE,
        }
        checkpoint_path: Path | None = None
        if checkpoint_directory is not None:
            key = hash_object([plan.sample_id, plan.variant, plan.source_model])
            checkpoint_path = checkpoint_directory / f"{key}.json"
            if checkpoint_path.is_file():
                try:
                    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                    cached_row = checkpoint["row"]
                    if (
                        checkpoint["schema_version"] != "semuav.native-receiver-checkpoint.v1"
                        or checkpoint["context_sha256"] != checkpoint_identity
                        or not isinstance(cached_row, dict)
                        or checkpoint["row_sha256"] != hash_object(cached_row)
                        or any(cached_row.get(name) != value for name, value in fixed_row.items())
                    ):
                        raise ValueError("checkpoint identity/hash mismatch")
                    payload = plan.encode()
                    if (
                        cached_row.get("payload_sha256") != hashlib.sha256(payload).hexdigest()
                        or cached_row.get("payload_bytes") != len(payload)
                    ):
                        raise ValueError("checkpoint wire payload changed")
                    for name in ("generation_ms", "energy_j", "receiver_latency_ms", "receiver_energy_j"):
                        if not math.isfinite(float(cached_row[name])) or float(cached_row[name]) < 0:
                            raise ValueError("checkpoint measurement is invalid")
                    _coerce_action(cached_row["receiver_action"], sample_id=plan.sample_id, variant=plan.variant)
                except (KeyError, TypeError, ValueError, OSError) as exc:
                    raise StudyGateError(
                        f"Native receiver checkpoint is invalid for {plan.sample_id}/{plan.variant}: {exc}"
                    ) from exc
                rows.append(cached_row)
                continue
        encode_context = MeasurementContext(plan.sample_id, plan.variant, "representation_encoding")
        encoded = _coerce_measured(measurement_runner(encode_context, plan.encode), encode_context)
        if not isinstance(encoded.value, (bytes, bytearray)):
            raise TypeError("Representation encoder measurement must return bytes")
        payload = bytes(encoded.value)
        if not payload:
            raise StudyGateError(f"Encoded payload is empty for {plan.sample_id}/{plan.variant}")

        def receive() -> Any:
            decoded = _decode_native_payload(
                plan, payload, temporary_root, control_context,
            )
            return receiver(decoded)

        receiver_context = MeasurementContext(plan.sample_id, plan.variant, "wire_decode_and_receiver")
        received = _coerce_measured(
            measurement_runner(receiver_context, receive), receiver_context,
        )
        action = _coerce_action(received.value, sample_id=plan.sample_id, variant=plan.variant)
        row = {
            **fixed_row,
            "receiver_action": action,
            "generation_ms": encoded.latency_ms,
            "energy_j": encoded.energy_j,
            "receiver_latency_ms": received.latency_ms,
            "receiver_energy_j": received.energy_j,
            "encoding_measurement_metadata_json": _measurement_metadata_json(encoded),
            "receiver_measurement_metadata_json": _measurement_metadata_json(received),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "payload_bytes": len(payload),
        }
        if checkpoint_path is not None:
            atomic_write_json(checkpoint_path, {
                "schema_version": "semuav.native-receiver-checkpoint.v1",
                "context_sha256": checkpoint_identity,
                "row_sha256": hash_object(row),
                "row": row,
            })
        rows.append(row)
    frame = pd.DataFrame(rows)
    validate_native_receiver_cache(
        frame,
        manifest,
        semantic_packet_cache,
        design=design,
        control_contexts=control_contexts,
        expected_fusion_receipt=receiver_fusion_receipt,
        verify_payload_hashes=False,
    )
    if output_path is not None:
        atomic_write_frame(output_path, frame)
    return frame


def validate_native_receiver_cache(
    cache: pd.DataFrame,
    manifest: pd.DataFrame,
    semantic_packet_cache: pd.DataFrame,
    *,
    design: NativeReceiverDesign,
    control_contexts: Mapping[str, ReceiverControlContext],
    expected_fusion_receipt: Mapping[str, Any] | None = None,
    verify_payload_hashes: bool = True,
) -> pd.DataFrame:
    """Fail closed on cache schema, coverage, costs, provenance, and payload hashes."""

    required = {
        "sample_id", "variant", "source_model", "media_kind", "receiver_action",
        "generation_ms", "energy_j", "receiver_latency_ms", "receiver_energy_j",
        "hardware_id", "sender_hardware_id", "receiver_hardware_id",
        "sender_revision", "receiver_revision", "payload_sha256", "payload_bytes",
        "control_context_sha256", "cost_scope", "receiver_fusion_policy_json",
        "receiver_fusion_policy_version", "receiver_fusion_policy_sha256",
    }
    if missing := required.difference(cache.columns):
        raise StudyGateError(f"Native receiver cache schema is incomplete: {sorted(missing)}")
    if cache.empty:
        raise StudyGateError("Native receiver cache is empty")
    key = ["sample_id", "variant", "source_model"]
    if cache.duplicated(key).any():
        raise StudyGateError("Native receiver cache contains duplicate candidate keys")
    if set(cache["receiver_action"].astype(str)).difference(REGISTERED_ACTIONS):
        raise StudyGateError("Native receiver cache contains an unregistered receiver action")
    numeric = [
        "generation_ms", "energy_j", "receiver_latency_ms", "receiver_energy_j",
        "payload_bytes",
    ]
    converted = cache[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(converted.to_numpy(dtype=float)).all():
        raise StudyGateError("Native receiver cache contains non-finite measurements")
    if (converted.to_numpy(dtype=float) < 0).any() or (converted["payload_bytes"] <= 0).any():
        raise StudyGateError("Native receiver cache contains invalid negative costs or empty payloads")
    if not cache["payload_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise StudyGateError("Native receiver cache payload_sha256 is invalid")
    if not cache["control_context_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise StudyGateError("Native receiver cache control_context_sha256 is invalid")
    for column in (
        "hardware_id", "sender_hardware_id", "receiver_hardware_id",
        "sender_revision", "receiver_revision",
    ):
        if cache[column].astype(str).str.strip().eq("").any():
            raise StudyGateError(f"Native receiver cache provenance is empty: {column}")
    fusion_json_values = set(cache["receiver_fusion_policy_json"].astype(str))
    fusion_versions = set(cache["receiver_fusion_policy_version"].astype(str))
    fusion_hashes = set(cache["receiver_fusion_policy_sha256"].astype(str))
    if len(fusion_json_values) != 1 or len(fusion_versions) != 1 or len(fusion_hashes) != 1:
        raise StudyGateError("Native receiver cache mixes receiver fusion policies")
    try:
        cached_fusion = json.loads(next(iter(fusion_json_values)))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StudyGateError("Native receiver cache fusion receipt is invalid JSON") from exc
    cached_json, cached_version, cached_hash = _validated_fusion_receipt(cached_fusion)
    if (
        next(iter(fusion_json_values)) != cached_json
        or next(iter(fusion_versions)) != cached_version
        or next(iter(fusion_hashes)) != cached_hash
    ):
        raise StudyGateError("Native receiver cache fusion receipt/hash is invalid")
    if not cache["receiver_revision"].astype(str).eq(
        str(cached_fusion["detector_revision"])
    ).all():
        raise StudyGateError("Native receiver cache fusion detector revision changed")
    if expected_fusion_receipt is not None:
        expected_json, expected_version, expected_hash = _validated_fusion_receipt(
            expected_fusion_receipt,
        )
        if (
            cached_json != expected_json
            or cached_version != expected_version
            or cached_hash != expected_hash
        ):
            raise StudyGateError(
                "Native receiver cache differs from the expected fusion policy"
            )
    if not cache["cost_scope"].astype(str).eq(REPRESENTATION_COST_SCOPE).all():
        raise StudyGateError("Native receiver cache cost_scope would double-charge perception")

    context_lookup = _control_context_lookup(manifest, control_contexts)
    expected_rows: list[dict[str, Any]] = []
    for plan in _native_payload_plans(manifest, semantic_packet_cache, design):
        record = {
            "sample_id": plan.sample_id,
            "variant": plan.variant,
            "source_model": plan.source_model,
            "media_kind": plan.media_kind,
            "control_context_sha256": context_lookup[plan.sample_id].sha256,
        }
        if verify_payload_hashes:
            payload = plan.encode()
            record["payload_sha256"] = hashlib.sha256(payload).hexdigest()
            record["payload_bytes"] = len(payload)
        expected_rows.append(record)
    expected = pd.DataFrame(expected_rows)
    if expected.duplicated(key).any():
        raise StudyGateError("Registered native design contains colliding candidate keys")
    actual_keys = set(map(tuple, cache[key].astype(str).to_numpy()))
    expected_keys = set(map(tuple, expected[key].astype(str).to_numpy()))
    if missing := sorted(expected_keys - actual_keys):
        raise StudyGateError(
            f"Native receiver cache misses {len(missing)} registered candidate(s); examples: {missing[:5]}"
        )
    if extras := sorted(actual_keys - expected_keys):
        raise StudyGateError(
            f"Native receiver cache has {len(extras)} unregistered candidate(s); examples: {extras[:5]}"
        )
    comparison_columns = [*key, "media_kind", "control_context_sha256"]
    if verify_payload_hashes:
        comparison_columns += ["payload_sha256", "payload_bytes"]
    actual = cache[comparison_columns].copy()
    if verify_payload_hashes:
        actual["payload_bytes"] = pd.to_numeric(
            actual["payload_bytes"], errors="raise",
        ).astype(int)
    merged = expected.merge(actual, on=key, suffixes=("_expected", "_actual"), validate="one_to_one")
    if not merged["media_kind_expected"].eq(merged["media_kind_actual"]).all():
        raise StudyGateError("Native receiver cache media_kind does not match the registered candidate")
    if not merged["control_context_sha256_expected"].eq(
        merged["control_context_sha256_actual"]
    ).all():
        raise StudyGateError(
            "Native receiver cache control context does not match the frozen pre-shared input"
        )
    if verify_payload_hashes:
        if not merged["payload_sha256_expected"].eq(merged["payload_sha256_actual"]).all():
            raise StudyGateError("Native receiver cache payload hash does not match transmitted bytes")
        if not merged["payload_bytes_expected"].eq(merged["payload_bytes_actual"]).all():
            raise StudyGateError("Native receiver cache payload length does not match transmitted bytes")
    return cache


def build_registered_feature_cache(
    manifest: pd.DataFrame,
    *,
    design: FeatureCacheDesign,
    control_contexts: Mapping[str, ReceiverControlContext],
    encoder: FeatureEncoder,
    receiver_head: FeatureHead,
    measurement_runner: MeasurementRunner,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build registered INT8/INT4 split-feature candidates from real images."""

    _require_manifest(manifest, require_clip_paths=False)
    context_lookup = _control_context_lookup(manifest, control_contexts)
    rows: list[dict[str, Any]] = []
    for item in manifest.itertuples(index=False):
        sample_id = str(item.sample_id)
        control_context = context_lookup[sample_id]
        image_path = Path(str(item.image_path))
        if not image_path.is_file():
            raise StudyGateError(f"Image is missing for {sample_id}: {image_path}")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB").copy()
        encoder_input = FeatureEncoderInput(
            sample_id=sample_id,
            image=image,
            encoder_revision=design.encoder_revision,
            split_layer=design.split_layer,
        )
        encoder_context = MeasurementContext(
            sample_id, f"{design.variant_prefix}_encoder", "feature_encoder",
        )
        encoded = _coerce_measured(
            measurement_runner(encoder_context, lambda: encoder(encoder_input)),
            encoder_context,
        )
        features = np.asarray(encoded.value, dtype=np.float32)
        if features.size == 0 or not np.isfinite(features).all():
            raise StudyGateError(f"Feature encoder returned an empty or non-finite tensor for {sample_id}")

        for bits in design.quantization_bits:
            bits = int(bits)
            variant = f"{design.variant_prefix}_int{bits}"
            quant_context = MeasurementContext(sample_id, variant, "feature_quantization")
            quantized_measurement = _coerce_measured(
                measurement_runner(quant_context, lambda bits=bits: quantize_tensor(features, bits)),
                quant_context,
            )
            quantized = quantized_measurement.value
            if not isinstance(quantized, QuantizedTensor):
                raise TypeError("Feature quantization measurement must return QuantizedTensor")

            payload = serialize_quantized_tensor(quantized)

            def run_receiver_head(
                payload: bytes = payload,
                sample_id: str = sample_id,
                variant: str = variant,
            ) -> Any:
                # Reconstruct the tensor exclusively from the exact bytes that
                # Experiment 2 counts on the wire.  Shape, scale, and zero
                # point are therefore not hidden out-of-band inputs.
                wire_tensor = deserialize_quantized_tensor(payload)
                head_input = FeatureHeadInput(
                    sample_id=sample_id,
                    variant=variant,
                    features=dequantize_tensor(wire_tensor),
                    quantized=wire_tensor,
                    encoder_revision=design.encoder_revision,
                    split_layer=design.split_layer,
                    receiver_head_revision=design.receiver_head_revision,
                    control_context=control_context,
                    control_context_sha256=control_context.sha256,
                )
                return receiver_head(head_input)

            head_context = MeasurementContext(sample_id, variant, "feature_decode_and_receiver_head")
            received = _coerce_measured(
                measurement_runner(head_context, run_receiver_head),
                head_context,
            )
            action = _coerce_action(received.value, sample_id=sample_id, variant=variant)
            rows.append({
                "sample_id": sample_id,
                "variant": variant,
                "source_model": "registered_feature_receiver",
                "payload_blob": payload.hex(),
                "payload_bytes": len(payload),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "receiver_action": action,
                "generation_ms": encoded.latency_ms + quantized_measurement.latency_ms,
                "energy_j": encoded.energy_j + quantized_measurement.energy_j,
                "encoder_latency_ms": encoded.latency_ms,
                "encoder_energy_j": encoded.energy_j,
                "quantization_latency_ms": quantized_measurement.latency_ms,
                "quantization_energy_j": quantized_measurement.energy_j,
                "receiver_latency_ms": received.latency_ms,
                "receiver_energy_j": received.energy_j,
                "encoder_measurement_metadata_json": _measurement_metadata_json(encoded),
                "quantization_measurement_metadata_json": _measurement_metadata_json(
                    quantized_measurement,
                ),
                "receiver_measurement_metadata_json": _measurement_metadata_json(received),
                "encoder_revision": design.encoder_revision,
                "split_layer": design.split_layer,
                "receiver_head_revision": design.receiver_head_revision,
                "sender_hardware_id": design.sender_hardware_id,
                "receiver_hardware_id": design.receiver_hardware_id,
                "tensor_shape": json.dumps(list(quantized.shape), separators=(",", ":")),
                "quantization": f"int{bits}",
                "quantization_scale": quantized.scale,
                "quantization_zero_point": quantized.zero_point,
                "control_context_sha256": control_context.sha256,
                "cost_scope": REPRESENTATION_COST_SCOPE,
            })
    frame = pd.DataFrame(rows)
    validate_registered_feature_cache(
        frame, manifest, design=design, control_contexts=control_contexts,
    )
    if output_path is not None:
        atomic_write_frame(output_path, frame)
    return frame


def validate_registered_feature_cache(
    cache: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    design: FeatureCacheDesign,
    control_contexts: Mapping[str, ReceiverControlContext],
) -> pd.DataFrame:
    """Validate feature-cache coverage and byte/provenance binding."""

    required = {
        "sample_id", "variant", "source_model", "payload_blob", "payload_bytes",
        "payload_sha256", "receiver_action", "generation_ms", "energy_j",
        "receiver_latency_ms", "receiver_energy_j", "encoder_revision", "split_layer",
        "receiver_head_revision", "sender_hardware_id", "receiver_hardware_id",
        "tensor_shape", "quantization", "quantization_scale",
        "quantization_zero_point", "control_context_sha256", "cost_scope",
    }
    if missing := required.difference(cache.columns):
        raise StudyGateError(f"Registered feature cache schema is incomplete: {sorted(missing)}")
    if cache.empty:
        raise StudyGateError("Registered feature cache is empty")
    key = ["sample_id", "variant"]
    if cache.duplicated(key).any():
        raise StudyGateError("Registered feature cache contains duplicate candidate keys")
    manifest_ids = set(manifest["sample_id"].astype(str))
    expected_variants = {
        f"{design.variant_prefix}_int{int(bits)}" for bits in design.quantization_bits
    }
    expected_keys = {(sample_id, variant) for sample_id in manifest_ids for variant in expected_variants}
    actual_keys = set(map(tuple, cache[key].astype(str).to_numpy()))
    if missing := sorted(expected_keys - actual_keys):
        raise StudyGateError(
            f"Registered feature cache misses {len(missing)} candidate(s); examples: {missing[:5]}"
        )
    if extras := sorted(actual_keys - expected_keys):
        raise StudyGateError(
            f"Registered feature cache has {len(extras)} unregistered candidate(s); examples: {extras[:5]}"
        )
    if set(cache["receiver_action"].astype(str)).difference(REGISTERED_ACTIONS):
        raise StudyGateError("Registered feature cache contains an unregistered receiver action")
    if not cache["control_context_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise StudyGateError("Registered feature cache control_context_sha256 is invalid")
    context_lookup = _control_context_lookup(manifest, control_contexts)
    expected_context_hashes = cache["sample_id"].astype(str).map(
        {sample_id: context.sha256 for sample_id, context in context_lookup.items()}
    )
    if not cache["control_context_sha256"].astype(str).eq(expected_context_hashes).all():
        raise StudyGateError(
            "Registered feature cache control context does not match the frozen pre-shared input"
        )
    expected_constants = {
        "source_model": "registered_feature_receiver",
        "encoder_revision": design.encoder_revision,
        "split_layer": design.split_layer,
        "receiver_head_revision": design.receiver_head_revision,
        "sender_hardware_id": design.sender_hardware_id,
        "receiver_hardware_id": design.receiver_hardware_id,
        "cost_scope": REPRESENTATION_COST_SCOPE,
    }
    for column, expected in expected_constants.items():
        if not cache[column].astype(str).eq(str(expected)).all():
            raise StudyGateError(f"Registered feature cache {column} differs from the frozen design")
    numeric = [
        "payload_bytes", "generation_ms", "energy_j", "receiver_latency_ms",
        "receiver_energy_j", "quantization_scale", "quantization_zero_point",
    ]
    converted = cache[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(converted.to_numpy(dtype=float)).all():
        raise StudyGateError("Registered feature cache contains non-finite measurements")
    if (converted[["generation_ms", "energy_j", "receiver_latency_ms", "receiver_energy_j"]] < 0).any().any():
        raise StudyGateError("Registered feature cache contains negative costs")
    if (converted["payload_bytes"] <= 0).any() or (converted["quantization_scale"] <= 0).any():
        raise StudyGateError("Registered feature cache contains invalid payload length or scale")

    for row in cache.itertuples(index=False):
        try:
            payload = bytes.fromhex(str(row.payload_blob))
        except ValueError as exc:
            raise StudyGateError("Registered feature payload_blob is not hexadecimal") from exc
        if len(payload) != int(row.payload_bytes):
            raise StudyGateError("Registered feature payload length does not match payload_bytes")
        if hashlib.sha256(payload).hexdigest() != str(row.payload_sha256):
            raise StudyGateError("Registered feature payload hash does not match payload_blob")
        if not str(row.payload_sha256).islower() or len(str(row.payload_sha256)) != 64:
            raise StudyGateError("Registered feature payload_sha256 is invalid")
        try:
            shape_value = json.loads(str(row.tensor_shape))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise StudyGateError("Registered feature tensor_shape is not valid JSON") from exc
        if (
            not isinstance(shape_value, list)
            or not shape_value
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape_value)
        ):
            raise StudyGateError("Registered feature tensor_shape must contain positive integers")
        quantization = str(row.quantization)
        if quantization not in {"int4", "int8"}:
            raise StudyGateError("Registered feature quantization is not INT4 or INT8")
        bits = int(quantization[3:])
        if str(row.variant) != f"{design.variant_prefix}_int{bits}":
            raise StudyGateError("Registered feature variant and quantization disagree")
        try:
            wire_tensor = deserialize_quantized_tensor(payload)
        except ValueError as exc:
            raise StudyGateError("Registered feature wire payload is invalid") from exc
        if wire_tensor.shape != tuple(shape_value) or wire_tensor.bits != bits:
            raise StudyGateError("Registered feature wire metadata differs from declared shape or quantization")
        if not math.isclose(
            wire_tensor.scale, float(row.quantization_scale),
            rel_tol=0.0, abs_tol=0.0,
        ):
            raise StudyGateError("Registered feature wire scale differs from its cache record")
        if wire_tensor.zero_point != int(row.quantization_zero_point):
            raise StudyGateError("Registered feature wire zero point differs from its cache record")
    return cache


__all__ = [
    "CONTROL_CONTEXT_SCHEMA_VERSION",
    "NATIVE_RECEIVER_FUSION_RECEIPT_SCHEMA",
    "DecodedNativePayload",
    "FeatureCacheDesign",
    "FeatureEncoderInput",
    "FeatureHeadInput",
    "MeasuredValue",
    "MeasurementContext",
    "NativeReceiverDesign",
    "ReceiverControlContext",
    "REPRESENTATION_COST_SCOPE",
    "build_native_receiver_cache",
    "build_registered_feature_cache",
    "validate_native_receiver_cache",
    "validate_registered_feature_cache",
    "uniform_receiver_control_contexts",
]
