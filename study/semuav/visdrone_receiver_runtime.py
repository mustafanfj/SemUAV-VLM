"""Concrete, gold-free receiver runtime for the VisDrone paper workflow.

The cache producers in :mod:`semuav.visdrone_receivers` deliberately accept
callables.  This module supplies the production callables used by the paper
notebook: NVML-backed measurement, a native decoded-pixel receiver, and a
development-only frozen split-feature head.  Evaluation labels are used only
while fitting the feature head on rows whose frozen split is ``train``.  Native
receiver decisions consume transmitted bytes plus a pre-shared, registered
mission-profile/ROI context; they never receive a gold action or annotation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .hardware import profile_callable
from .perception import DetectorOnlyBackend, InferenceRequest, UltralyticsDetectorTracker
from .protocol import (
    SEMANTIC_ONTOLOGY_VERSION,
    VISDRONE_COMPOSED_PACKET_AUTHORSHIP,
    VISDRONE_COMPOSED_PROVENANCE_FIELDS,
    VISDRONE_PROPOSAL_DECISION_VERSION,
    VIDEO_GEOMETRY_POLICY_VERSION,
    QuantizedTensor,
    serialize_quantized_tensor,
)
from .registry import (
    StudyGateError,
    atomic_write_json,
    canonical_json,
    hash_file,
    hash_object,
    sha256_bytes,
)
from .visdrone_receivers import (
    DecodedNativePayload,
    FeatureEncoderInput,
    FeatureHeadInput,
    MeasuredValue,
    MeasurementContext,
    NATIVE_RECEIVER_FUSION_RECEIPT_SCHEMA,
    ReceiverControlContext,
)


FEATURE_HEAD_SCHEMA = "semuav.visdrone-feature-head.v1"
REGISTERED_ACTIONS = ("NO_ACTION", "REVIEW", "RESPOND")
RECEIVER_MEASUREMENT_SCHEMA = "semuav.receiver-repeated-measurement.v1"
NATIVE_RECEIVER_FUSION_POLICY_VERSION = "semuav.visdrone-native-fusion.v3"
FEATURE_TENSOR_ABSOLUTE_TOLERANCE = 1e-6


def native_receiver_fusion_receipt(
    *,
    detector_revision: str,
    priority_profiles: Mapping[str, Sequence[str]],
    registered_roi_xyxy: Sequence[float],
    priority_confidence: float,
    image_plane_iou: float,
) -> dict[str, Any]:
    """Freeze the gold-free native decision rule applied to each wire mode."""

    revision = str(detector_revision).strip()
    roi = tuple(float(value) for value in registered_roi_xyxy)
    if not revision:
        raise ValueError("Native receiver detector revision must be populated")
    if (
        len(roi) != 4
        or not (0.0 <= roi[0] < roi[2] <= 1.0)
        or not (0.0 <= roi[1] < roi[3] <= 1.0)
    ):
        raise ValueError("Native receiver registered ROI is invalid")
    profiles = {
        str(profile): sorted({str(label) for label in labels})
        for profile, labels in sorted(priority_profiles.items())
    }
    if not profiles or any(not profile or not labels for profile, labels in profiles.items()):
        raise ValueError("Native receiver priority profiles must be populated")
    return {
        "schema_version": NATIVE_RECEIVER_FUSION_RECEIPT_SCHEMA,
        "policy_version": NATIVE_RECEIVER_FUSION_POLICY_VERSION,
        "detector_revision": revision,
        "action_source_by_media_kind": {
            "semantic": "registered_variant_specific_semantic_policy",
            "semantic_roi": (
                "severity_max_transmitted_semantic_action_and_"
                "detector_on_transmitted_roi_pixels"
            ),
            "image": "detector_on_decoded_pixels",
            "video": "detector_on_decoded_pixels_current_frame",
        },
        "action_source_by_semantic_variant": {
            "l0_cbor": "validated_heartbeat_then_registered_NO_ACTION",
            "l1_cbor": "transmitted_recommended_action",
            "l2_cbor": "transmitted_recommended_action",
        },
        "semantic_roi_crop_policy": {
            "detector_roi_xyxy": [0.0, 0.0, 1.0, 1.0],
            "transmitted_roi_geometry_source": "authenticated_semantic_evidence",
            "fusion_order": list(REGISTERED_ACTIONS),
        },
        "video_geometry_policy": {
            "version": VIDEO_GEOMETRY_POLICY_VERSION,
            "chroma_pixel_format": "yuv420p",
            "padding": "right_bottom_to_even_without_resampling",
            "source_geometry": "transmitted_mp4_uuid_when_padded",
            "receiver_geometry": "crop_padding_before_detector_and_registered_roi",
            "metadata_cost": "40_transmitted_bytes_for_padded_clips",
            "mixed_frame_sizes": "fail_closed",
        },
        "registered_full_frame_roi_xyxy": list(roi),
        "priority_profiles": profiles,
        "priority_confidence": float(priority_confidence),
        "image_plane_iou": float(image_plane_iou),
    }


@dataclasses.dataclass(frozen=True)
class RepeatedMeasuredValue(MeasuredValue):
    """Measured value plus an auditable repeated-output consistency receipt."""

    measurement_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)


def _exact_output_bytes(value: Any) -> bytes:
    """Canonicalize an exact receiver output without lossy string coercion."""

    if isinstance(value, bytes):
        return b"bytes\0" + value
    if isinstance(value, bytearray):
        return b"bytes\0" + bytes(value)
    if isinstance(value, QuantizedTensor):
        return b"quantized\0" + serialize_quantized_tensor(value)
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return (
            b"ndarray\0"
            + str(contiguous.dtype).encode("ascii")
            + b"\0"
            + canonical_json(list(contiguous.shape)).encode("utf-8")
            + b"\0"
            + contiguous.tobytes(order="C")
        )
    if dataclasses.is_dataclass(value):
        return _exact_output_bytes(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return canonical_json(value).encode("utf-8")
    if isinstance(value, (list, tuple, str, int, float, bool)) or value is None:
        return canonical_json(value).encode("utf-8")
    raise StudyGateError(
        f"Repeated receiver output type is not canonically serializable: {type(value).__name__}"
    )


def nvml_measurement_runner(
    context: MeasurementContext,
    operation: Any,
    *,
    device_index: int = 0,
    warmup_iterations: int = 5,
    steady_state_repetitions: int = 3,
) -> MeasuredValue:
    """Measure one callable with the registered GPU-board NVML meter."""

    warmups = int(warmup_iterations)
    repetitions = int(steady_state_repetitions)
    if warmups < 0 or repetitions <= 0:
        raise ValueError("Measurement warmups/repetitions are invalid")
    for _ in range(warmups):
        operation()
    values: list[Any] = []
    latency_values: list[float] = []
    energy_values: list[float] = []
    for _ in range(repetitions):
        value, profile = profile_callable(
            operation, device_index=int(device_index), measure_energy=True,
        )
        values.append(value)
        latency_values.append(float(profile["latency_ms"]))
        energy_values.append(float(profile["energy_j"]))
    if any(not math.isfinite(value) or value < 0 for value in latency_values):
        raise StudyGateError("NVML measurement returned invalid latency")
    if any(not math.isfinite(value) or value < 0 for value in energy_values):
        raise StudyGateError("NVML measurement returned invalid GPU-board energy")
    latency = float(np.median(latency_values))
    energy = float(np.median(energy_values))

    arrays = [np.asarray(value) for value in values if isinstance(value, np.ndarray)]
    if arrays:
        if len(arrays) != len(values) or not all(
            np.issubdtype(array.dtype, np.floating) for array in arrays
        ):
            raise StudyGateError("Repeated receiver output types changed across repetitions")
        baseline = arrays[0]
        if baseline.size == 0 or not np.isfinite(baseline).all():
            raise StudyGateError("Repeated feature tensor is empty or non-finite")
        for array in arrays[1:]:
            if (
                array.shape != baseline.shape
                or array.dtype != baseline.dtype
                or not np.isfinite(array).all()
                or not np.allclose(
                    array,
                    baseline,
                    rtol=0.0,
                    atol=FEATURE_TENSOR_ABSOLUTE_TOLERANCE,
                    equal_nan=False,
                )
            ):
                raise StudyGateError(
                    "Repeated floating feature tensors exceed the frozen absolute tolerance"
                )
        returned_value: Any = np.median(np.stack(arrays, axis=0), axis=0).astype(
            baseline.dtype, copy=False,
        )
        comparison = "floating_feature_tensor_absolute_tolerance_then_downstream_exact"
        tolerance: float | None = FEATURE_TENSOR_ABSOLUTE_TOLERANCE
    else:
        serialized = [_exact_output_bytes(value) for value in values]
        if any(value != serialized[0] for value in serialized[1:]):
            raise StudyGateError(
                "Repeated serialized payload, quantized tensor, or receiver action is nondeterministic"
            )
        returned_value = values[0]
        comparison = "byte_identical_canonical_output"
        tolerance = None

    repetition_hashes = [sha256_bytes(_exact_output_bytes(value)) for value in values]
    metadata = {
        "schema_version": RECEIVER_MEASUREMENT_SCHEMA,
        "sample_id": str(context.sample_id),
        "variant": str(context.variant),
        "stage": str(context.stage),
        "warmup_iterations": warmups,
        "steady_state_repetitions": repetitions,
        "warmups_excluded": True,
        "resource_aggregation": "median_latency_ms_and_gpu_board_energy_j",
        "output_aggregation": (
            "elementwise_median" if arrays else "first_byte_identical_repetition"
        ),
        "output_comparison": comparison,
        "feature_tensor_absolute_tolerance": tolerance,
        "repetition_output_sha256": repetition_hashes,
    }
    return RepeatedMeasuredValue(
        value=returned_value,
        latency_ms=latency,
        energy_j=energy,
        measurement_metadata=metadata,
    )


def registered_receiver_context(
    manifest: pd.DataFrame,
    *,
    registered_roi_xyxy: Sequence[float],
) -> dict[str, ReceiverControlContext]:
    """Build the explicit pre-shared control table from registered inputs.

    Only outcome-independent fields are read: sample identity, capture time,
    authoritative flight telemetry, mission profile, and the pre-registered
    image-plane ROI.  Gold actions, annotations, and evidence ROIs are neither
    accepted nor copied into receiver inputs.
    """

    required = {
        "sample_id", "mission_profile", "capture_time_us", "telemetry_json",
    }
    if missing := required.difference(manifest.columns):
        raise StudyGateError(
            f"Registered receiver context is missing columns: {sorted(missing)}"
        )
    if manifest.empty or manifest["sample_id"].astype(str).duplicated().any():
        raise StudyGateError("Registered receiver context requires unique nonempty samples")
    frozen_roi = tuple(float(value) for value in registered_roi_xyxy)
    contexts: dict[str, ReceiverControlContext] = {}
    for row in manifest[
        ["sample_id", "mission_profile", "capture_time_us", "telemetry_json"]
    ].itertuples(index=False):
        sample_id = str(row.sample_id).strip()
        if not sample_id or sample_id in contexts:
            raise StudyGateError(
                "Registered receiver context requires unique nonempty normalized sample IDs"
            )
        try:
            telemetry = (
                dict(row.telemetry_json)
                if isinstance(row.telemetry_json, Mapping)
                else json.loads(str(row.telemetry_json))
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise StudyGateError(
                f"Registered receiver telemetry is invalid for {sample_id}"
            ) from exc
        if not isinstance(telemetry, Mapping):
            raise StudyGateError(
                f"Registered receiver telemetry is invalid for {sample_id}"
            )
        if telemetry.get("mission_profile") != str(row.mission_profile):
            raise StudyGateError(
                f"Registered receiver mission profile differs for {sample_id}"
            )
        telemetry_roi = telemetry.get("registered_roi_xyxy")
        if (
            not isinstance(telemetry_roi, (list, tuple))
            or len(telemetry_roi) != 4
            or any(isinstance(value, bool) for value in telemetry_roi)
        ):
            raise StudyGateError(
                f"Registered receiver ROI is invalid for {sample_id}"
            )
        try:
            observed_roi = tuple(float(value) for value in telemetry_roi)
        except (TypeError, ValueError) as exc:
            raise StudyGateError(
                f"Registered receiver ROI is invalid for {sample_id}"
            ) from exc
        if not all(math.isfinite(value) for value in observed_roi) or observed_roi != frozen_roi:
            raise StudyGateError(
                f"Registered receiver ROI differs for {sample_id}"
            )
        try:
            contexts[sample_id] = ReceiverControlContext(
                mission_profile=str(row.mission_profile),
                registered_roi_xyxy=frozen_roi,
                capture_time_us=row.capture_time_us,
                gps_lat=telemetry.get("gps_lat"),
                gps_lon=telemetry.get("gps_lon"),
                altitude_m=telemetry.get("altitude_m"),
                yaw_deg=telemetry.get("yaw_deg"),
            )
        except (TypeError, ValueError) as exc:
            raise StudyGateError(
                f"Registered receiver context is invalid for {sample_id}"
            ) from exc
    return contexts


class RegisteredVisDroneNativeReceiver:
    """Decode semantic packets directly and pixels through the frozen detector.

    ``sample_id`` remains an observation identifier only.  Mission profile and
    ROI arrive inside :class:`DecodedNativePayload` as an explicitly pre-shared
    control context, never via an identifier lookup or evaluation manifest.
    For a clip, prior causal frames initialize tracker state, but the returned
    action is the action for the final/current frame only.  This matches the
    current-frame source endpoint and prevents an earlier critical object from
    being credited as a correct current-frame alert.
    """

    def __init__(
        self,
        detector: UltralyticsDetectorTracker,
        *,
        detector_revision: str,
        priority_profiles: Mapping[str, Sequence[str]],
        registered_roi_xyxy: Sequence[float],
        priority_confidence: float = 0.05,
        image_plane_iou: float = 0.05,
        temporary_root: str | Path | None = None,
    ) -> None:
        self.detector = detector
        self.registered_roi_xyxy = tuple(float(value) for value in registered_roi_xyxy)
        self.temporary_root = (
            Path(temporary_root).expanduser().resolve()
            if temporary_root is not None else None
        )
        if self.temporary_root is not None:
            self.temporary_root.mkdir(parents=True, exist_ok=True)
        all_labels = sorted({
            str(label)
            for labels in priority_profiles.values()
            for label in labels
        })
        self.fusion_receipt = native_receiver_fusion_receipt(
            detector_revision=detector_revision,
            priority_profiles=priority_profiles,
            registered_roi_xyxy=self.registered_roi_xyxy,
            priority_confidence=priority_confidence,
            image_plane_iou=image_plane_iou,
        )
        self.policy = DetectorOnlyBackend(
            detector_revision,
            urban_policy="registered_roi_occupancy",
            urban_priority_labels=all_labels,
            urban_priority_profiles=priority_profiles,
            urban_priority_roi_xyxy=self.registered_roi_xyxy,
            urban_priority_confidence=float(priority_confidence),
            urban_image_plane_iou=float(image_plane_iou),
        )
        # L3 transmits an already cropped image.  Its detector geometry is
        # therefore normalized in crop coordinates, so the registered policy
        # must cover the complete received crop rather than reapply the
        # original full-frame ROI a second time.
        self.roi_crop_policy = DetectorOnlyBackend(
            detector_revision,
            urban_policy="registered_roi_occupancy",
            urban_priority_labels=all_labels,
            urban_priority_profiles=priority_profiles,
            urban_priority_roi_xyxy=(0.0, 0.0, 1.0, 1.0),
            urban_priority_confidence=float(priority_confidence),
            urban_image_plane_iou=float(image_plane_iou),
        )

    def _frame_action(
        self,
        sample_id: str,
        frame: Any,
        *,
        control_context: ReceiverControlContext,
        persist: bool,
        policy: DetectorOnlyBackend | None = None,
        detector_roi_xyxy: Sequence[float] | None = None,
    ) -> str:
        from PIL import Image

        if not isinstance(frame, Image.Image):
            raise TypeError("Decoded native frame must be a PIL image")
        descriptor, name = tempfile.mkstemp(
            prefix="semuav-receiver-", suffix=".png",
            dir=str(self.temporary_root) if self.temporary_root else None,
        )
        os.close(descriptor)
        path = Path(name)
        try:
            frame.convert("RGB").save(path, format="PNG")
            proposals = self.detector.detect(str(path), persist=persist)
            telemetry = control_context.telemetry
            if detector_roi_xyxy is not None:
                telemetry["registered_roi_xyxy"] = [
                    float(value) for value in detector_roi_xyxy
                ]
            request = InferenceRequest(
                sample_id,
                str(path),
                "visdrone_urban",
                control_context.capture_time_us,
                telemetry,
                tuple(proposals),
            )
            record = (policy or self.policy).infer(request)
            if not record.schema_valid or not isinstance(record.parsed, Mapping):
                raise StudyGateError(
                    f"Frozen native receiver produced an invalid packet for {sample_id}"
                )
            action = str(record.parsed.get("recommended_action", ""))
            if action not in REGISTERED_ACTIONS:
                raise StudyGateError(
                    f"Frozen native receiver produced unregistered action {action!r}"
                )
            return action
        finally:
            path.unlink(missing_ok=True)

    def _validate_l0_heartbeat(
        self,
        packet: Mapping[str, Any],
        *,
        sample_id: str,
        source_model: str,
        control_context: ReceiverControlContext,
    ) -> None:
        """Validate the exact registered L0 view before applying NO_ACTION.

        L0 deliberately carries no sender-authored action.  Returning the
        registered ``NO_ACTION`` endpoint is therefore valid only after the
        decoded bytes have been shown to be the canonical heartbeat for this
        observation and the separately frozen receiver context.
        """

        required = {
            "schema_version", "observation_id", "mission",
            "capture_time_us", "telemetry", "provenance",
        }
        if set(packet) != required:
            raise StudyGateError(
                f"Decoded L0 heartbeat has noncanonical fields for {sample_id}"
            )
        if packet["schema_version"] != "semuav.v1":
            raise StudyGateError(
                f"Decoded L0 heartbeat has unregistered schema for {sample_id}"
            )
        if not isinstance(packet["observation_id"], str) or packet["observation_id"] != sample_id:
            raise StudyGateError(
                f"Decoded L0 heartbeat observation differs for {sample_id}"
            )
        if packet["mission"] != "visdrone_urban":
            raise StudyGateError(
                f"Decoded L0 heartbeat has unregistered mission for {sample_id}"
            )
        capture_time = packet["capture_time_us"]
        if (
            isinstance(capture_time, bool)
            or not isinstance(capture_time, int)
            or capture_time < 0
            or capture_time != control_context.capture_time_us
        ):
            raise StudyGateError(
                f"Decoded L0 heartbeat has invalid capture time for {sample_id}"
            )

        telemetry = packet["telemetry"]
        if not isinstance(telemetry, Mapping):
            raise StudyGateError(
                f"Decoded L0 heartbeat has invalid telemetry for {sample_id}"
            )
        telemetry_values: dict[str, float] = {}
        for name in ("gps_lat", "gps_lon", "altitude_m", "yaw_deg"):
            value = telemetry.get(name)
            if isinstance(value, bool):
                raise StudyGateError(
                    f"Decoded L0 heartbeat has invalid telemetry for {sample_id}"
                )
            try:
                telemetry_values[name] = float(value)
            except (TypeError, ValueError) as exc:
                raise StudyGateError(
                    f"Decoded L0 heartbeat has invalid telemetry for {sample_id}"
                ) from exc
            if not math.isfinite(telemetry_values[name]):
                raise StudyGateError(
                    f"Decoded L0 heartbeat has invalid telemetry for {sample_id}"
                )
        if (
            not -90.0 <= telemetry_values["gps_lat"] <= 90.0
            or not -180.0 <= telemetry_values["gps_lon"] <= 180.0
            or telemetry_values["altitude_m"] < 0.0
            or not 0.0 <= telemetry_values["yaw_deg"] < 360.0
        ):
            raise StudyGateError(
                f"Decoded L0 heartbeat has invalid telemetry for {sample_id}"
            )
        expected_telemetry = control_context.telemetry
        if any(
            telemetry_values[name] != float(expected_telemetry[name])
            for name in ("gps_lat", "gps_lon", "altitude_m", "yaw_deg")
        ):
            raise StudyGateError(
                f"Decoded L0 heartbeat telemetry differs for {sample_id}"
            )
        if telemetry.get("mission_profile") != control_context.mission_profile:
            raise StudyGateError(
                f"Decoded L0 heartbeat mission profile differs for {sample_id}"
            )
        transmitted_roi = telemetry.get("registered_roi_xyxy")
        if (
            not isinstance(transmitted_roi, (list, tuple))
            or len(transmitted_roi) != 4
            or any(isinstance(value, bool) for value in transmitted_roi)
        ):
            raise StudyGateError(
                f"Decoded L0 heartbeat registered ROI differs for {sample_id}"
            )
        try:
            transmitted_roi_values = tuple(float(value) for value in transmitted_roi)
        except (TypeError, ValueError) as exc:
            raise StudyGateError(
                f"Decoded L0 heartbeat registered ROI differs for {sample_id}"
            ) from exc
        if (
            not all(math.isfinite(value) for value in transmitted_roi_values)
            or transmitted_roi_values != tuple(control_context.registered_roi_xyxy)
        ):
            raise StudyGateError(
                f"Decoded L0 heartbeat registered ROI differs for {sample_id}"
            )

        provenance = packet["provenance"]
        required_provenance = {
            "source_model", "model_revision", "prompt_hash",
            "ontology_version", "calibration", "schema_version",
        }
        if (
            not isinstance(provenance, Mapping)
            or required_provenance.difference(provenance)
            or provenance.get("schema_version") != packet["schema_version"]
            or provenance.get("source_model") != source_model
            or provenance.get("ontology_version") != SEMANTIC_ONTOLOGY_VERSION
            or not re.fullmatch(r"[0-9a-f]{40}", str(provenance.get("model_revision", "")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(provenance.get("prompt_hash", "")))
        ):
            raise StudyGateError(
                f"Decoded L0 heartbeat has invalid provenance for {sample_id}"
            )
        calibration = provenance.get("calibration")
        if not isinstance(calibration, Mapping):
            raise StudyGateError(
                f"Decoded L0 heartbeat has invalid provenance for {sample_id}"
            )
        method = calibration.get("method")
        temperature = calibration.get("temperature")
        confidence_source = calibration.get("confidence_source")
        if isinstance(temperature, bool):
            raise StudyGateError(
                f"Decoded L0 heartbeat has invalid provenance for {sample_id}"
            )
        try:
            temperature_value = float(temperature)
        except (TypeError, ValueError) as exc:
            raise StudyGateError(
                f"Decoded L0 heartbeat has invalid provenance for {sample_id}"
            ) from exc
        if (
            not isinstance(method, str)
            or not method.strip()
            or not math.isfinite(temperature_value)
            or temperature_value <= 0.0
            or not isinstance(confidence_source, str)
            or not confidence_source.strip()
        ):
            raise StudyGateError(
                f"Decoded L0 heartbeat has invalid provenance for {sample_id}"
            )

        composed_fields = VISDRONE_COMPOSED_PROVENANCE_FIELDS.intersection(provenance)
        if composed_fields:
            if (
                composed_fields != VISDRONE_COMPOSED_PROVENANCE_FIELDS
                or provenance.get("semantic_packet_authorship")
                != VISDRONE_COMPOSED_PACKET_AUTHORSHIP
                or provenance.get("decision_contract_version")
                != VISDRONE_PROPOSAL_DECISION_VERSION
                or any(
                    not re.fullmatch(r"[0-9a-f]{64}", str(provenance.get(name, "")))
                    for name in (
                        "decision_composer_policy_sha256",
                        "decision_context_sha256",
                    )
                )
            ):
                raise StudyGateError(
                    f"Decoded L0 heartbeat has invalid provenance for {sample_id}"
                )

    def __call__(self, decoded: DecodedNativePayload) -> str:
        sample_id = str(decoded.sample_id)
        if decoded.control_context_sha256 != decoded.control_context.sha256:
            raise StudyGateError("Native receiver control context hash is invalid")
        if tuple(decoded.registered_roi_xyxy) != self.registered_roi_xyxy:
            raise StudyGateError("Native receiver control ROI differs from its frozen policy")
        if decoded.media_kind == "semantic":
            packet = decoded.semantic_packet
            if packet is None or decoded.frames or decoded.transmitted_roi_xyxy is not None:
                raise StudyGateError(
                    f"Native semantic payload has incompatible decoded evidence for {sample_id}"
                )
            variant = str(decoded.variant)
            if variant == "l0_cbor":
                self._validate_l0_heartbeat(
                    packet,
                    sample_id=sample_id,
                    source_model=str(decoded.source_model),
                    control_context=decoded.control_context,
                )
                return "NO_ACTION"
            if variant not in {"l1_cbor", "l2_cbor"}:
                raise StudyGateError(
                    f"Decoded semantic payload has unregistered variant for {sample_id}"
                )
            action = str(packet.get("recommended_action", ""))
            if action not in REGISTERED_ACTIONS:
                raise StudyGateError(
                    f"Decoded semantic packet has unregistered action for {sample_id}"
                )
            return action
        if decoded.media_kind == "semantic_roi":
            if (
                decoded.semantic_packet is None
                or len(decoded.frames) != 1
                or decoded.transmitted_roi_xyxy is None
            ):
                raise StudyGateError(
                    f"Native L3 payload is missing semantic or ROI evidence for {sample_id}"
                )
            semantic_action = str(
                decoded.semantic_packet.get("recommended_action", "")
            )
            if semantic_action not in REGISTERED_ACTIONS:
                raise StudyGateError(
                    f"Decoded semantic packet has unregistered action for {sample_id}"
                )
            roi_action = self._frame_action(
                sample_id,
                decoded.frames[0],
                control_context=decoded.control_context,
                persist=False,
                policy=self.roi_crop_policy,
                detector_roi_xyxy=(0.0, 0.0, 1.0, 1.0),
            )
            severity = {action: index for index, action in enumerate(REGISTERED_ACTIONS)}
            return max((semantic_action, roi_action), key=severity.__getitem__)
        if decoded.semantic_packet is not None:
            raise StudyGateError(
                f"Native visual payload unexpectedly contains semantics for {sample_id}"
            )
        if not decoded.frames:
            raise StudyGateError(
                f"Native visual payload reconstructed no frames for {sample_id}"
            )
        current_action: str | None = None
        for index, frame in enumerate(decoded.frames):
            current_action = self._frame_action(
                sample_id, frame, control_context=decoded.control_context,
                persist=index > 0,
            )
        if current_action is None:  # defensive; decoded.frames is nonempty above
            raise StudyGateError(f"Native visual payload has no current frame for {sample_id}")
        return current_action


class ResNet50Layer3Encoder:
    """Frozen ImageNet-V2 ResNet-50 encoder through layer3."""

    def __init__(self, device: str = "cuda:0", *, allow_download: bool = False) -> None:
        try:
            import torch
            from torchvision.models import ResNet50_Weights, resnet50
        except ImportError as exc:  # pragma: no cover - production dependency
            raise StudyGateError("torch and torchvision are required for L4") from exc
        self._torch = torch
        self.device = str(device)
        self.weights_id = "torchvision.ResNet50_Weights.IMAGENET1K_V2"
        weights = ResNet50_Weights.IMAGENET1K_V2
        checkpoint_name = Path(str(weights.url)).name
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name
        if not allow_download and not checkpoint.is_file():
            raise StudyGateError(
                "Frozen ResNet-50 IMAGENET1K_V2 checkpoint is not in the local "
                f"torch cache: {checkpoint}. Run feature-head preparation first."
            )
        model = resnet50(weights=weights)
        if not checkpoint.is_file():
            raise StudyGateError("torchvision did not materialize the registered encoder checkpoint")
        self.checkpoint_path = checkpoint.resolve()
        self.checkpoint_sha256 = hash_file(self.checkpoint_path)
        self.preprocess = weights.transforms()
        self.encoder = torch.nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3,
        ).to(self.device).eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        digest = hashlib.sha256()
        for name, tensor in sorted(self.encoder.state_dict().items()):
            contiguous = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            digest.update(contiguous.numpy().tobytes(order="C"))
        self.state_sha256 = digest.hexdigest()

    def __call__(self, item: FeatureEncoderInput) -> np.ndarray:
        if str(item.encoder_revision) != "IMAGENET1K_V2" or str(item.split_layer) != "layer3":
            raise StudyGateError("L4 encoder request differs from the registered ResNet-50 layer3 design")
        tensor = self.preprocess(item.image.convert("RGB")).unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            features = self.encoder(tensor).squeeze(0)
        return features.detach().float().cpu().numpy()


def _pooled_feature(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim < 1 or array.size == 0 or not np.isfinite(array).all():
        raise StudyGateError("Registered feature tensor is empty or non-finite")
    if array.ndim == 1:
        return array
    return array.reshape(array.shape[0], -1).mean(axis=1)


def _profile_one_hot(profile: str, profiles: Sequence[str]) -> np.ndarray:
    if profile not in profiles:
        raise StudyGateError(f"Unregistered mission profile {profile!r} at the feature receiver")
    vector = np.zeros(len(profiles), dtype=np.float32)
    vector[list(profiles).index(profile)] = 1.0
    return vector


@dataclasses.dataclass(frozen=True)
class FrozenFeatureHead:
    weights: np.ndarray
    bias: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    mission_profiles: tuple[str, ...]
    revision: str

    def __call__(self, item: FeatureHeadInput) -> str:
        if item.control_context_sha256 != item.control_context.sha256:
            raise StudyGateError("Feature receiver control context hash is invalid")
        profile = item.mission_profile
        pooled = _pooled_feature(item.features)
        vector = np.concatenate([
            (pooled - self.mean) / self.scale,
            _profile_one_hot(profile, self.mission_profiles),
        ]).astype(np.float32)
        if vector.shape[0] != self.weights.shape[1]:
            raise StudyGateError("Feature receiver input dimension differs from its frozen head")
        logits = self.weights @ vector + self.bias
        return REGISTERED_ACTIONS[int(np.argmax(logits))]


def train_and_freeze_feature_head(
    manifest: pd.DataFrame,
    encoder: ResNet50Layer3Encoder,
    *,
    output_dir: str | Path,
    seed: int,
    epochs: int = 200,
) -> tuple[FrozenFeatureHead, Path]:
    """Fit and freeze a linear head using only the development ``train`` split."""

    required = {"sample_id", "split", "gold_action", "mission_profile", "image_path"}
    if missing := required.difference(manifest.columns):
        raise StudyGateError(f"Feature-head training manifest is missing: {sorted(missing)}")
    train = manifest.loc[manifest["split"].astype(str).eq("train")].copy()
    if train.empty:
        raise StudyGateError("Feature-head fitting requires frozen development train rows")
    if set(manifest.loc[train.index, "split"].astype(str)) != {"train"}:
        raise AssertionError("Feature-head fitter escaped the development train split")
    if train["sample_id"].astype(str).duplicated().any():
        raise StudyGateError("Feature-head training sample IDs are duplicated")
    if set(train["gold_action"].astype(str)) != set(REGISTERED_ACTIONS):
        raise StudyGateError("Feature-head training requires all three registered actions")
    profiles = tuple(sorted(train["mission_profile"].astype(str).unique()))
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for item in train.sort_values("sample_id", kind="mergesort").itertuples(index=False):
        from PIL import Image

        path = Path(str(item.image_path))
        if not path.is_file():
            raise StudyGateError(f"Feature-head training image is missing: {path}")
        with Image.open(path) as opened:
            encoder_input = FeatureEncoderInput(
                sample_id=str(item.sample_id), image=opened.convert("RGB").copy(),
                encoder_revision="IMAGENET1K_V2", split_layer="layer3",
            )
        pooled = _pooled_feature(encoder(encoder_input))
        rows.append(np.concatenate([
            pooled, _profile_one_hot(str(item.mission_profile), profiles),
        ]))
        labels.append(REGISTERED_ACTIONS.index(str(item.gold_action)))
    matrix = np.stack(rows).astype(np.float32)
    visual_dim = matrix.shape[1] - len(profiles)
    mean = matrix[:, :visual_dim].mean(axis=0)
    scale = matrix[:, :visual_dim].std(axis=0)
    scale[scale < 1e-6] = 1.0
    matrix[:, :visual_dim] = (matrix[:, :visual_dim] - mean) / scale

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - production dependency
        raise StudyGateError("torch is required to train the registered L4 head") from exc
    torch.manual_seed(int(seed))
    x = torch.from_numpy(matrix)
    y = torch.tensor(labels, dtype=torch.long)
    model = torch.nn.Linear(matrix.shape[1], len(REGISTERED_ACTIONS))
    counts = torch.bincount(y, minlength=len(REGISTERED_ACTIONS)).float()
    class_weights = counts.sum() / (len(REGISTERED_ACTIONS) * counts)
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=max(1, int(epochs)),
        tolerance_grad=1e-9, tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Any:
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y, weight=class_weights)
        loss.backward()
        return loss

    optimizer.step(closure)
    weights = model.weight.detach().cpu().numpy().astype(np.float32)
    bias = model.bias.detach().cpu().numpy().astype(np.float32)

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    weights_path = destination / "feature_head_weights.npz"
    temporary = destination / ".feature_head_weights.tmp.npz"
    np.savez_compressed(
        temporary, weights=weights, bias=bias, mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
    )
    os.replace(temporary, weights_path)
    payload: dict[str, Any] = {
        "schema": FEATURE_HEAD_SCHEMA,
        "encoder": encoder.weights_id,
        "encoder_checkpoint_path": str(encoder.checkpoint_path),
        "encoder_checkpoint_sha256": encoder.checkpoint_sha256,
        "encoder_state_sha256": encoder.state_sha256,
        "encoder_revision": "IMAGENET1K_V2",
        "split_layer": "layer3",
        "receiver": "global_average_linear_with_registered_mission_profile",
        "actions": list(REGISTERED_ACTIONS),
        "mission_profiles": list(profiles),
        "training_split": "train",
        "training_rows": int(len(train)),
        "training_sample_ids_sha256": hash_object(sorted(train["sample_id"].astype(str))),
        "seed": int(seed),
        "weights_path": str(weights_path),
        "weights_sha256": hash_file(weights_path),
    }
    payload["revision"] = hash_object(payload)
    lock_path = destination / "FEATURE_HEAD_LOCK.json"
    atomic_write_json(lock_path, payload)
    return FrozenFeatureHead(
        weights, bias, mean.astype(np.float32), scale.astype(np.float32),
        profiles, str(payload["revision"]),
    ), lock_path


def load_frozen_feature_head(
    lock_path: str | Path,
    *,
    encoder: ResNet50Layer3Encoder,
) -> FrozenFeatureHead:
    """Verify and load a development-trained feature head for a later phase."""

    path = Path(lock_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StudyGateError(f"Cannot read feature-head lock: {path}") from exc
    if payload.get("schema") != FEATURE_HEAD_SCHEMA or payload.get("training_split") != "train":
        raise StudyGateError("Feature-head lock is not a development-train artifact")
    if payload.get("encoder") != encoder.weights_id:
        raise StudyGateError("Feature-head lock names a different frozen encoder")
    if payload.get("encoder_revision") != "IMAGENET1K_V2" or payload.get("split_layer") != "layer3":
        raise StudyGateError("Feature-head lock differs from the registered ResNet-50 layer3 design")
    if payload.get("encoder_state_sha256") != encoder.state_sha256:
        raise StudyGateError(
            "Frozen feature head was trained with different ResNet-50 layer3 bytes"
        )
    if payload.get("encoder_checkpoint_sha256") != encoder.checkpoint_sha256:
        raise StudyGateError("Frozen ResNet-50 checkpoint bytes changed")
    expected_revision = str(payload.get("revision", ""))
    unhashed = dict(payload)
    unhashed.pop("revision", None)
    if not expected_revision or hash_object(unhashed) != expected_revision:
        raise StudyGateError("Feature-head lock self-hash is invalid")
    weights_path = Path(str(payload.get("weights_path", ""))).expanduser().resolve()
    if not weights_path.is_file() or hash_file(weights_path) != payload.get("weights_sha256"):
        raise StudyGateError("Feature-head weights are missing or changed")
    with np.load(weights_path, allow_pickle=False) as arrays:
        weights = np.asarray(arrays["weights"], dtype=np.float32)
        bias = np.asarray(arrays["bias"], dtype=np.float32)
        mean = np.asarray(arrays["mean"], dtype=np.float32)
        scale = np.asarray(arrays["scale"], dtype=np.float32)
    if (
        weights.ndim != 2 or bias.shape != (len(REGISTERED_ACTIONS),)
        or weights.shape[0] != len(REGISTERED_ACTIONS)
        or mean.ndim != 1 or scale.shape != mean.shape
        or weights.shape[1] != mean.shape[0] + len(payload.get("mission_profiles", []))
        or not np.isfinite(weights).all() or not np.isfinite(bias).all()
        or not np.isfinite(mean).all() or not np.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise StudyGateError("Feature-head arrays have an invalid shape or value")
    return FrozenFeatureHead(
        weights, bias, mean, scale,
        tuple(map(str, payload["mission_profiles"])), expected_revision,
    )


def prepare_feature_head_artifact(
    train_manifest: pd.DataFrame,
    *,
    output_dir: str | Path,
    device: str,
    seed: int,
) -> Mapping[str, Any]:
    """Prepare the exact train-only L4 artifact before offline evaluation.

    This is deliberately a separate lifecycle step.  Calling it may populate
    the torchvision weights cache.  Validation and test later instantiate the
    same encoder with local-only model settings and verify its byte-level state
    digest against the returned lock.
    """

    encoder = ResNet50Layer3Encoder(device, allow_download=True)
    _, lock_path = train_and_freeze_feature_head(
        train_manifest, encoder, output_dir=output_dir, seed=seed,
    )
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    return {
        "lock_path": str(lock_path),
        "lock_sha256": hash_file(lock_path),
        "weights_path": str(payload["weights_path"]),
        "weights_sha256": str(payload["weights_sha256"]),
        "encoder_checkpoint_path": str(payload["encoder_checkpoint_path"]),
        "encoder_checkpoint_sha256": str(payload["encoder_checkpoint_sha256"]),
        "encoder_state_sha256": str(payload["encoder_state_sha256"]),
        "revision": str(payload["revision"]),
    }


__all__ = [
    "FEATURE_TENSOR_ABSOLUTE_TOLERANCE",
    "NATIVE_RECEIVER_FUSION_POLICY_VERSION",
    "FEATURE_HEAD_SCHEMA",
    "FrozenFeatureHead",
    "REGISTERED_ACTIONS",
    "RECEIVER_MEASUREMENT_SCHEMA",
    "RepeatedMeasuredValue",
    "RegisteredVisDroneNativeReceiver",
    "ResNet50Layer3Encoder",
    "load_frozen_feature_head",
    "nvml_measurement_runner",
    "native_receiver_fusion_receipt",
    "prepare_feature_head_artifact",
    "registered_receiver_context",
    "train_and_freeze_feature_head",
]
