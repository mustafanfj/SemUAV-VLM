from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from PIL import Image

from semuav.experiments import _semantic_item_key
from semuav.perception import DetectorOnlyBackend, InferenceRequest, run_detector_cache
from semuav.protocol import quantize_tensor
from semuav.registry import StudyGateError, hash_file, hash_object
from semuav.visdrone_receiver_runtime import (
    FEATURE_TENSOR_ABSOLUTE_TOLERANCE,
    FrozenFeatureHead,
    RECEIVER_MEASUREMENT_SCHEMA,
    RegisteredVisDroneNativeReceiver,
    ResNet50Layer3Encoder,
    load_frozen_feature_head,
    nvml_measurement_runner,
    registered_receiver_context,
)
from semuav.visdrone_receivers import (
    DecodedNativePayload,
    FeatureHeadInput,
    MeasurementContext,
    ReceiverControlContext,
)


class _Detector:
    def __init__(self) -> None:
        self.paths: list[str] = []

    def detect(self, image_path: str, *, persist: bool = False):
        del persist
        self.paths.append(image_path)
        return ()


class _Policy:
    def __init__(self) -> None:
        self.requests = []

    def infer(self, request):
        self.requests.append(request)
        return types.SimpleNamespace(
            schema_valid=True,
            parsed={"recommended_action": "REVIEW"},
        )


class ReceiverControlContextRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ReceiverControlContext(
            mission_profile="heavy_vehicle",
            registered_roi_xyxy=(0.2, 0.3, 0.8, 0.9),
            capture_time_us=123,
            gps_lat=24.0,
            gps_lon=54.0,
            altitude_m=70.0,
            yaw_deg=0.0,
        )

    def test_native_visual_receiver_uses_explicit_context_not_sample_lookup(self) -> None:
        detector = _Detector()
        receiver = RegisteredVisDroneNativeReceiver(
            detector,
            detector_revision="detector-sha",
            priority_profiles={
                "vulnerable_road_user": ("pedestrian",),
                "heavy_vehicle": ("truck",),
                "general_mobility": ("car",),
            },
            registered_roi_xyxy=self.context.registered_roi_xyxy,
        )
        policy = _Policy()
        receiver.policy = policy
        decoded = DecodedNativePayload(
            sample_id="opaque:id/that-contains-no-profile",
            variant="jpeg_q50_s1",
            source_model="ground_visual_receiver",
            media_kind="image",
            payload_sha256="a" * 64,
            control_context=self.context,
            control_context_sha256=self.context.sha256,
            frames=(Image.new("RGB", (8, 8), "white"),),
        )

        self.assertEqual(receiver(decoded), "REVIEW")
        self.assertEqual(len(policy.requests), 1)
        request = policy.requests[0]
        self.assertEqual(request.telemetry["mission_profile"], "heavy_vehicle")
        self.assertEqual(
            request.telemetry["registered_roi_xyxy"],
            list(self.context.registered_roi_xyxy),
        )
        self.assertEqual(request.observation_id, decoded.sample_id)
        self.assertEqual(request.capture_time_us, self.context.capture_time_us)
        self.assertEqual(request.telemetry, self.context.telemetry)

    def test_native_receiver_rejects_changed_control_context_hash(self) -> None:
        receiver = RegisteredVisDroneNativeReceiver(
            _Detector(),
            detector_revision="detector-sha",
            priority_profiles={"heavy_vehicle": ("truck",)},
            registered_roi_xyxy=self.context.registered_roi_xyxy,
        )
        decoded = DecodedNativePayload(
            sample_id="opaque",
            variant="l2_semantics",
            source_model="model",
            media_kind="semantic",
            payload_sha256="a" * 64,
            control_context=self.context,
            control_context_sha256="0" * 64,
            semantic_packet={"recommended_action": "RESPOND"},
        )
        with self.assertRaisesRegex(StudyGateError, "context hash"):
            receiver(decoded)

    def _semantic_receiver(self) -> RegisteredVisDroneNativeReceiver:
        return RegisteredVisDroneNativeReceiver(
            _Detector(), detector_revision="detector-sha",
            priority_profiles={"heavy_vehicle": ("truck",)},
            registered_roi_xyxy=self.context.registered_roi_xyxy,
        )

    def _heartbeat(self) -> dict:
        return {
            "schema_version": "semuav.v1",
            "observation_id": "heartbeat-sample",
            "mission": "visdrone_urban",
            "capture_time_us": 123,
            "telemetry": {
                "gps_lat": 24.0,
                "gps_lon": 54.0,
                "altitude_m": 70.0,
                "yaw_deg": 0.0,
                "mission_profile": self.context.mission_profile,
                "registered_roi_xyxy": list(self.context.registered_roi_xyxy),
            },
            "provenance": {
                "source_model": "model",
                "model_revision": "a" * 40,
                "prompt_hash": "b" * 64,
                "ontology_version": "semuav.ontology.v1",
                "calibration": {
                    "method": "temperature_scaling",
                    "temperature": 1.0,
                    "confidence_source": "generated_token_probability",
                },
                "schema_version": "semuav.v1",
                "semantic_packet_authorship": "deterministic_host_composer",
                "decision_contract_version": "semuav.visdrone-proposal-decision.v2",
                "decision_composer_policy_sha256": "c" * 64,
                "decision_context_sha256": "d" * 64,
            },
        }

    def _semantic_payload(self, variant: str, packet: dict) -> DecodedNativePayload:
        return DecodedNativePayload(
            sample_id="heartbeat-sample", variant=variant,
            source_model="model", media_kind="semantic",
            payload_sha256="a" * 64, control_context=self.context,
            control_context_sha256=self.context.sha256,
            semantic_packet=packet,
        )

    def test_l0_heartbeat_returns_registered_no_action(self) -> None:
        receiver = self._semantic_receiver()
        self.assertEqual(
            receiver(self._semantic_payload("l0_cbor", self._heartbeat())),
            "NO_ACTION",
        )
        self.assertEqual(receiver.detector.paths, [])

    def test_l0_heartbeat_rejects_transmitted_action(self) -> None:
        receiver = self._semantic_receiver()
        packet = {**self._heartbeat(), "recommended_action": "NO_ACTION"}
        with self.assertRaisesRegex(StudyGateError, "noncanonical fields"):
            receiver(self._semantic_payload("l0_cbor", packet))

    def test_l0_heartbeat_binds_transmitted_and_preshared_context(self) -> None:
        receiver = self._semantic_receiver()
        for field, value in (
            ("mission_profile", "general_mobility"),
            ("registered_roi_xyxy", [0.0, 0.0, 1.0, 1.0]),
        ):
            with self.subTest(field=field):
                packet = self._heartbeat()
                packet["telemetry"][field] = value
                with self.assertRaisesRegex(
                    StudyGateError, "mission profile|registered ROI",
                ):
                    receiver(self._semantic_payload("l0_cbor", packet))

    def test_l0_heartbeat_rejects_noncanonical_capture_time(self) -> None:
        receiver = self._semantic_receiver()
        for value in (True, 123.0, -1):
            with self.subTest(value=value):
                packet = self._heartbeat()
                packet["capture_time_us"] = value
                with self.assertRaisesRegex(StudyGateError, "capture time"):
                    receiver(self._semantic_payload("l0_cbor", packet))

    def test_l0_heartbeat_rejects_invalid_or_partial_provenance(self) -> None:
        receiver = self._semantic_receiver()
        mutations = (
            ("ontology_version", "unregistered.ontology"),
            ("model_revision", "not-a-revision"),
            ("prompt_hash", "not-a-hash"),
            ("decision_context_sha256", None),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                packet = self._heartbeat()
                if value is None:
                    packet["provenance"].pop(field)
                else:
                    packet["provenance"][field] = value
                with self.assertRaisesRegex(StudyGateError, "invalid provenance"):
                    receiver(self._semantic_payload("l0_cbor", packet))

    def test_actionless_l1_and_l2_remain_fail_closed(self) -> None:
        receiver = self._semantic_receiver()
        for variant in ("l1_cbor", "l2_cbor"):
            with self.subTest(variant=variant):
                with self.assertRaisesRegex(StudyGateError, "unregistered action"):
                    receiver(self._semantic_payload(variant, self._heartbeat()))

    def test_unknown_semantic_variant_is_rejected(self) -> None:
        receiver = self._semantic_receiver()
        packet = {**self._heartbeat(), "recommended_action": "NO_ACTION"}
        with self.assertRaisesRegex(StudyGateError, "unregistered variant"):
            receiver(self._semantic_payload("unknown_semantic", packet))

    def test_visual_clip_returns_current_frame_action_not_prior_maximum(self) -> None:
        receiver = RegisteredVisDroneNativeReceiver(
            _Detector(), detector_revision="detector-sha",
            priority_profiles={"heavy_vehicle": ("truck",)},
            registered_roi_xyxy=self.context.registered_roi_xyxy,
        )

        class SequencePolicy:
            def __init__(self) -> None:
                self.actions = iter(("RESPOND", "NO_ACTION"))

            def infer(self, request):
                del request
                return types.SimpleNamespace(
                    schema_valid=True,
                    parsed={"recommended_action": next(self.actions)},
                )

        receiver.policy = SequencePolicy()
        decoded = DecodedNativePayload(
            sample_id="current-frame-endpoint", variant="h264_crf23",
            source_model="ground_visual_receiver", media_kind="video",
            payload_sha256="a" * 64, control_context=self.context,
            control_context_sha256=self.context.sha256,
            frames=(
                Image.new("RGB", (8, 8), "red"),
                Image.new("RGB", (8, 8), "white"),
            ),
        )
        self.assertEqual(receiver(decoded), "NO_ACTION")

    def test_l3_fuses_authenticated_roi_pixels_with_transmitted_semantics(self) -> None:
        receiver = RegisteredVisDroneNativeReceiver(
            _Detector(), detector_revision="detector-sha",
            priority_profiles={"heavy_vehicle": ("truck",)},
            registered_roi_xyxy=self.context.registered_roi_xyxy,
        )
        roi_policy = _Policy()
        receiver.roi_crop_policy = roi_policy
        decoded = DecodedNativePayload(
            sample_id="l3-current-roi", variant="cbor+jpeg_q50_roi",
            source_model="model", media_kind="semantic_roi",
            payload_sha256="a" * 64, control_context=self.context,
            control_context_sha256=self.context.sha256,
            semantic_packet={"recommended_action": "NO_ACTION"},
            frames=(Image.new("RGB", (8, 8), "white"),),
            transmitted_roi_xyxy=(0.2, 0.3, 0.8, 0.9),
        )
        frozen_context_before = self.context.as_dict()
        self.assertEqual(receiver(decoded), "REVIEW")
        self.assertEqual(self.context.as_dict(), frozen_context_before)
        self.assertEqual(len(roi_policy.requests), 1)
        self.assertEqual(
            roi_policy.requests[0].telemetry["registered_roi_xyxy"],
            [0.0, 0.0, 1.0, 1.0],
        )
        self.assertEqual(
            roi_policy.requests[0].capture_time_us,
            self.context.capture_time_us,
        )
        for name in ("gps_lat", "gps_lon", "altitude_m", "yaw_deg"):
            self.assertEqual(
                roi_policy.requests[0].telemetry[name],
                self.context.telemetry[name],
            )
        self.assertEqual(
            receiver.fusion_receipt["policy_version"],
            "semuav.visdrone-native-fusion.v3",
        )
        self.assertEqual(
            receiver.fusion_receipt["action_source_by_semantic_variant"]["l0_cbor"],
            "validated_heartbeat_then_registered_NO_ACTION",
        )

    def test_registered_context_binds_authoritative_time_and_telemetry(self) -> None:
        manifest = pd.DataFrame([{
            "sample_id": "registered-sample",
            "mission_profile": self.context.mission_profile,
            "capture_time_us": self.context.capture_time_us,
            "telemetry_json": json.dumps(self.context.telemetry),
            # Outcome fields may coexist in the manifest but must not enter the
            # independently hashed receiver context.
            "gold_action": "RESPOND",
        }])
        observed = registered_receiver_context(
            manifest,
            registered_roi_xyxy=self.context.registered_roi_xyxy,
        )["registered-sample"]
        self.assertEqual(observed.capture_time_us, self.context.capture_time_us)
        self.assertEqual(observed.telemetry, self.context.telemetry)
        self.assertNotIn("gold_action", observed.as_dict())

    def test_registered_context_rejects_telemetry_context_disagreement(self) -> None:
        for field, value in (
            ("mission_profile", "general_mobility"),
            ("registered_roi_xyxy", [0.0, 0.0, 1.0, 1.0]),
        ):
            with self.subTest(field=field):
                telemetry = self.context.telemetry
                telemetry[field] = value
                manifest = pd.DataFrame([{
                    "sample_id": "registered-sample",
                    "mission_profile": self.context.mission_profile,
                    "capture_time_us": self.context.capture_time_us,
                    "telemetry_json": json.dumps(telemetry),
                }])
                with self.assertRaisesRegex(
                    StudyGateError, "mission profile differs|ROI differs",
                ):
                    registered_receiver_context(
                        manifest,
                        registered_roi_xyxy=self.context.registered_roi_xyxy,
                    )

    def test_feature_head_uses_context_directly_and_rejects_hash_change(self) -> None:
        profiles = (
            "general_mobility",
            "heavy_vehicle",
            "vulnerable_road_user",
        )
        head = FrozenFeatureHead(
            weights=np.zeros((3, 5), dtype=np.float32),
            bias=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            mean=np.zeros(2, dtype=np.float32),
            scale=np.ones(2, dtype=np.float32),
            mission_profiles=profiles,
            revision="head-sha",
        )
        quantized = quantize_tensor(np.array([0.25, -0.5], dtype=np.float32), 8)
        item = FeatureHeadInput(
            sample_id="opaque:no-profile",
            variant="registered_feature_int8",
            features=np.array([0.25, -0.5], dtype=np.float32),
            quantized=quantized,
            encoder_revision="IMAGENET1K_V2",
            split_layer="layer3",
            receiver_head_revision="head-sha",
            control_context=self.context,
            control_context_sha256=self.context.sha256,
        )
        self.assertEqual(head(item), "REVIEW")
        changed = type(item)(
            **{
                **item.__dict__,
                "control_context_sha256": "0" * 64,
            }
        )
        with self.assertRaisesRegex(StudyGateError, "context hash"):
            head(changed)


class FrozenFeatureArtifactTests(unittest.TestCase):
    def _artifact(self, root: Path):
        checkpoint = root / "resnet50-11ad3fa6.pth"
        checkpoint.write_bytes(b"registered-resnet-checkpoint")
        weights_path = root / "feature_head_weights.npz"
        np.savez_compressed(
            weights_path,
            weights=np.zeros((3, 5), dtype=np.float32),
            bias=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            mean=np.zeros(2, dtype=np.float32),
            scale=np.ones(2, dtype=np.float32),
        )
        encoder = types.SimpleNamespace(
            weights_id="torchvision.ResNet50_Weights.IMAGENET1K_V2",
            checkpoint_path=checkpoint,
            checkpoint_sha256=hash_file(checkpoint),
            state_sha256="1" * 64,
        )
        payload = {
            "schema": "semuav.visdrone-feature-head.v1",
            "encoder": encoder.weights_id,
            "encoder_checkpoint_path": str(checkpoint),
            "encoder_checkpoint_sha256": encoder.checkpoint_sha256,
            "encoder_state_sha256": encoder.state_sha256,
            "encoder_revision": "IMAGENET1K_V2",
            "split_layer": "layer3",
            "receiver": "global_average_linear_with_registered_mission_profile",
            "actions": ["NO_ACTION", "REVIEW", "RESPOND"],
            "mission_profiles": [
                "general_mobility",
                "heavy_vehicle",
                "vulnerable_road_user",
            ],
            "training_split": "train",
            "training_rows": 3,
            "training_sample_ids_sha256": "2" * 64,
            "seed": 17,
            "weights_path": str(weights_path),
            "weights_sha256": hash_file(weights_path),
        }
        payload["revision"] = hash_object(payload)
        lock_path = root / "FEATURE_HEAD_LOCK.json"
        lock_path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        return encoder, lock_path, weights_path

    def test_feature_lock_binds_encoder_state_checkpoint_and_head_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            encoder, lock_path, weights_path = self._artifact(Path(directory))
            self.assertEqual(load_frozen_feature_head(lock_path, encoder=encoder).revision,
                             json.loads(lock_path.read_text())["revision"])

            weights_path.write_bytes(b"changed")
            with self.assertRaisesRegex(StudyGateError, "weights are missing or changed"):
                load_frozen_feature_head(lock_path, encoder=encoder)

    def test_feature_lock_rejects_changed_encoder_identity_or_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            encoder, lock_path, _ = self._artifact(Path(directory))
            wrong_identity = types.SimpleNamespace(
                **{**encoder.__dict__, "weights_id": "unregistered/encoder"}
            )
            with self.assertRaisesRegex(StudyGateError, "different frozen encoder"):
                load_frozen_feature_head(lock_path, encoder=wrong_identity)
            wrong_state = types.SimpleNamespace(
                **{**encoder.__dict__, "state_sha256": "9" * 64}
            )
            with self.assertRaisesRegex(StudyGateError, "different ResNet-50 layer3 bytes"):
                load_frozen_feature_head(lock_path, encoder=wrong_state)
            wrong_checkpoint = types.SimpleNamespace(
                **{**encoder.__dict__, "checkpoint_sha256": "8" * 64}
            )
            with self.assertRaisesRegex(StudyGateError, "checkpoint bytes changed"):
                load_frozen_feature_head(lock_path, encoder=wrong_checkpoint)

    def test_encoder_local_only_mode_fails_before_downloader_is_called(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            calls: list[object] = []
            torch_module = types.ModuleType("torch")
            torch_module.hub = types.SimpleNamespace(get_dir=lambda: directory)
            models_module = types.ModuleType("torchvision.models")
            models_module.ResNet50_Weights = types.SimpleNamespace(
                IMAGENET1K_V2=types.SimpleNamespace(
                    url="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
                )
            )

            def resnet50(*, weights):
                calls.append(weights)
                raise AssertionError("downloader/model constructor must not run")

            models_module.resnet50 = resnet50
            torchvision_module = types.ModuleType("torchvision")
            torchvision_module.models = models_module
            with mock.patch.dict(
                sys.modules,
                {
                    "torch": torch_module,
                    "torchvision": torchvision_module,
                    "torchvision.models": models_module,
                },
            ):
                with self.assertRaisesRegex(StudyGateError, "not in the local torch cache"):
                    ResNet50Layer3Encoder("cpu", allow_download=False)
            self.assertEqual(calls, [])


class DetectorProfilePairingTests(unittest.TestCase):
    def test_detector_runs_once_per_source_and_replicates_profile_costs(self) -> None:
        class CountingDetector:
            def __init__(self) -> None:
                self.calls: list[tuple[str, bool]] = []

            def detect(self, image_path: str, *, persist: bool = False):
                self.calls.append((image_path, persist))
                return [{
                    "label": "car", "confidence": 0.9,
                    "bbox_normalized_xyxy": [0.1, 0.1, 0.2, 0.2],
                    "call": len(self.calls),
                }]

        profiles = ("general_mobility", "heavy_vehicle", "vulnerable_road_user")
        rows = []
        for source, sequence, frame in (
            ("source-1", "sequence-a", 0),
            ("source-2", "sequence-a", 1),
            ("source-3", "sequence-b", 0),
        ):
            for profile in profiles:
                rows.append({
                    "sample_id": f"{source}:{profile}",
                    "source_sample_id": source,
                    "image_path": f"/{source}.jpg",
                    "sequence_id": sequence,
                    "frame_index": frame,
                    "dataset": "visdrone",
                    "component": "det_val",
                    "mission_profile": profile,
                })
        detector = CountingDetector()
        cache = run_detector_cache(detector, pd.DataFrame(rows))
        self.assertEqual(len(detector.calls), 3)
        self.assertEqual([persist for _, persist in detector.calls], [False, True, False])
        self.assertEqual(len(cache), 9)
        for _, copies in cache.groupby("source_sample_id"):
            self.assertEqual(copies["proposals_json"].nunique(), 1)
            self.assertEqual(copies["latency_ms"].nunique(), 1)

    def test_registered_overlap_threshold_and_symmetric_semantic_identity(self) -> None:
        backend = DetectorOnlyBackend(
            "a" * 40, urban_policy="registered_roi_occupancy",
            urban_priority_labels=("car", "pedestrian"),
            urban_priority_profiles={"general_mobility": ("car", "pedestrian")},
            urban_image_plane_iou=0.05,
        )

        def relations(second_box):
            request = InferenceRequest(
                "overlap", "unused.png", "visdrone_urban", 0,
                {
                    "mission_profile": "general_mobility",
                    "gps_lat": 0.0, "gps_lon": 0.0,
                    "altitude_m": 70.0, "yaw_deg": 0.0,
                },
                (
                    {"label": "car", "confidence": 0.9,
                     "bbox_normalized_xyxy": [0.0, 0.0, 0.5, 0.5]},
                    {"label": "pedestrian", "confidence": 0.9,
                     "bbox_normalized_xyxy": second_box},
                ),
            )
            record = backend.infer(request)
            assert record.parsed is not None
            return record.parsed["relations"]

        self.assertEqual(relations([0.49, 0.0, 0.99, 0.5]), [])
        observed = relations([0.45, 0.0, 0.95, 0.5])
        self.assertEqual(observed[0]["predicate"], "image_plane_overlap")
        forward = {"subject": "car", "predicate": "image_plane_overlap", "object": "pedestrian"}
        reverse = {"subject": "pedestrian", "predicate": "image_plane_overlap", "object": "car"}
        self.assertEqual(
            _semantic_item_key(forward, kind="relation"),
            _semantic_item_key(reverse, kind="relation"),
        )


class RepeatedReceiverMeasurementTests(unittest.TestCase):
    @staticmethod
    def _profile(function, **kwargs):
        del kwargs
        value = function()
        ordinal = RepeatedReceiverMeasurementTests._profile.calls
        RepeatedReceiverMeasurementTests._profile.calls += 1
        return value, {
            "latency_ms": [1.0, 3.0, 2.0][ordinal % 3],
            "energy_j": [0.1, 0.3, 0.2][ordinal % 3],
        }

    def setUp(self) -> None:
        self._profile.calls = 0
        self.context = MeasurementContext("sample", "variant", "feature_encoder")

    def test_float_tensor_tolerance_records_hashes_and_median_resources(self) -> None:
        tolerance = FEATURE_TENSOR_ABSOLUTE_TOLERANCE
        values = iter((
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([1.0 + tolerance / 2, 2.0], dtype=np.float32),
            np.array([1.0 - tolerance / 2, 2.0], dtype=np.float32),
        ))
        with mock.patch(
            "semuav.visdrone_receiver_runtime.profile_callable",
            side_effect=self._profile,
        ):
            measured = nvml_measurement_runner(
                self.context, lambda: next(values),
                warmup_iterations=0, steady_state_repetitions=3,
            )
        np.testing.assert_allclose(measured.value, [1.0, 2.0], rtol=0, atol=tolerance)
        self.assertEqual(measured.latency_ms, 2.0)
        self.assertAlmostEqual(measured.energy_j, 0.2)
        metadata = measured.measurement_metadata
        self.assertEqual(metadata["schema_version"], RECEIVER_MEASUREMENT_SCHEMA)
        self.assertEqual(len(metadata["repetition_output_sha256"]), 3)
        self.assertEqual(metadata["output_aggregation"], "elementwise_median")
        self.assertTrue(metadata["warmups_excluded"])

    def test_float_tensor_outside_tolerance_fails_closed(self) -> None:
        values = iter((
            np.array([1.0], dtype=np.float32),
            np.array([1.0 + FEATURE_TENSOR_ABSOLUTE_TOLERANCE * 4], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
        ))
        with mock.patch(
            "semuav.visdrone_receiver_runtime.profile_callable",
            side_effect=self._profile,
        ), self.assertRaisesRegex(StudyGateError, "frozen absolute tolerance"):
            nvml_measurement_runner(
                self.context, lambda: next(values),
                warmup_iterations=0, steady_state_repetitions=3,
            )

    def test_nondeterministic_action_or_quantized_bytes_fail_closed(self) -> None:
        for values in (
            iter(("REVIEW", "RESPOND", "REVIEW")),
            iter((
                quantize_tensor(np.array([0.1, 0.2], dtype=np.float32), 8),
                quantize_tensor(np.array([0.1, 0.9], dtype=np.float32), 8),
                quantize_tensor(np.array([0.1, 0.2], dtype=np.float32), 8),
            )),
        ):
            self._profile.calls = 0
            with self.subTest(values=values), mock.patch(
                "semuav.visdrone_receiver_runtime.profile_callable",
                side_effect=self._profile,
            ), self.assertRaisesRegex(StudyGateError, "nondeterministic"):
                nvml_measurement_runner(
                    self.context, lambda values=values: next(values),
                    warmup_iterations=0, steady_state_repetitions=3,
                )


if __name__ == "__main__":
    unittest.main()
