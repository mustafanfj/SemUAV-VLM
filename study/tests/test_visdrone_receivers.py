from __future__ import annotations

import hashlib
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

from semuav.experiments import experiment2_representation_frontier
from semuav.perception import calibrated_semantic_packet_cache
from semuav.registry import StudyGateError, hash_object
from semuav.runtime import resolve_ffmpeg_executable
from semuav.visdrone_receiver_runtime import RegisteredVisDroneNativeReceiver
from semuav.protocol import (
    dequantize_tensor,
    deserialize_quantized_tensor,
    quantize_tensor,
    serialize_quantized_tensor,
)
from semuav.visdrone_receivers import (
    DecodedNativePayload,
    FeatureCacheDesign,
    MeasuredValue,
    NativeReceiverDesign,
    NATIVE_RECEIVER_FUSION_RECEIPT_SCHEMA,
    ReceiverControlContext,
    build_native_receiver_cache,
    build_registered_feature_cache,
    validate_native_receiver_cache,
    validate_registered_feature_cache,
)


def _fusion_receipt(detector_revision: str = "receiver-revision") -> dict:
    return {
        "schema_version": NATIVE_RECEIVER_FUSION_RECEIPT_SCHEMA,
        "policy_version": "test-native-fusion.v1",
        "detector_revision": detector_revision,
        "action_source_by_media_kind": {
            "semantic": "semantic",
            "semantic_roi": "semantic_plus_roi",
            "image": "decoded_pixels",
            "video": "decoded_pixels_current_frame",
        },
    }


def _packet() -> dict:
    return {
        "schema_version": "semuav.v1",
        "observation_id": "s1",
        "mission": "visdrone_urban",
        "capture_time_us": 0,
        "telemetry": {
            "gps_lat": 24.0,
            "gps_lon": 54.0,
            "altitude_m": 70.0,
            "yaw_deg": 0.0,
            "mission_profile": "general_mobility",
            "registered_roi_xyxy": [0.25, 0.25, 0.75, 0.85],
        },
        "sufficient_evidence": True,
        "event": "visible_priority_roi_occupancy",
        "objects": [{
            "label": "car",
            "bbox_normalized_xyxy": [0.1, 0.2, 0.6, 0.8],
            "confidence": 0.9,
        }],
        "relations": [],
        "risk": 0.8,
        "uncertainty": 0.2,
        "recommended_action": "RESPOND",
        "evidence": [{
            "evidence_id": "event-0",
            "linked_claims": ["event"],
            "bbox_normalized_xyxy": [0.1, 0.2, 0.6, 0.8],
        }],
    }


def _semantic_cache() -> pd.DataFrame:
    inference = pd.DataFrame([{
        "observation_id": "s1",
        "parsed_json": json.dumps(_packet()),
        "schema_valid": True,
        "confidence": 0.8,
        "model_revision": "a" * 40,
        "prompt_hash": "b" * 64,
        "confidence_source": "generated_token_probability",
    }])
    return calibrated_semantic_packet_cache(
        inference, 1.0, source_model="model_a",
    )


def _measurer(context, operation):
    # The important producer contract is that the operation is executed by the
    # registered runner and all returned values are finite.  Hardware-specific
    # meters replace these deterministic constants in the real notebook.
    return MeasuredValue(operation(), latency_ms=0.25, energy_j=0.01)


def _fake_cbor_module():
    # The CI base image does not necessarily include optional cbor2.  This
    # preserves the canonical mapping round trip while the deployment suite
    # exercises the pinned real dependency.
    return types.SimpleNamespace(
        dumps=lambda value, canonical=True: json.dumps(
            value, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8"),
        loads=lambda data: json.loads(data),
    )


class VisDroneNativeReceiverCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.image_path = self.root / "sample.png"
        Image.new("RGB", (32, 24), "red").save(self.image_path)
        self.manifest = pd.DataFrame([{
            "sample_id": "s1",
            "visual_group_id": "g1",
            "image_path": str(self.image_path),
            "gold_action": "RESPOND",
            "clip_paths_json": "[]",
        }])
        self.semantic_cache = _semantic_cache()
        self.control_contexts = {
            "s1": ReceiverControlContext(
                mission_profile="general_mobility",
                registered_roi_xyxy=(0.25, 0.25, 0.75, 0.85),
                capture_time_us=0,
                gps_lat=24.0,
                gps_lon=54.0,
                altitude_m=70.0,
                yaw_deg=0.0,
            )
        }
        self.design = NativeReceiverDesign(
            semantic_source_model="model_a",
            image_codecs=("JPEG",),
            image_qualities=(50,),
            image_scales=(1.0,),
            video_crfs=(23,),
        )

    def tearDown(self) -> None:
        self.directory.cleanup()

    @staticmethod
    def receiver(request: DecodedNativePayload) -> str:
        # Only decoded wire content is available.  In particular, there is no
        # gold_action or manifest row on the receiver request.
        assert not hasattr(request, "gold_action")
        assert request.mission_profile == "general_mobility"
        assert request.registered_roi_xyxy == (0.25, 0.25, 0.75, 0.85)
        assert request.control_context_sha256 == request.control_context.sha256
        if request.semantic_packet is not None:
            return str(request.semantic_packet.get("recommended_action", "NO_ACTION"))
        return "RESPOND" if request.frames else "REVIEW"

    def build(self) -> pd.DataFrame:
        return build_native_receiver_cache(
            self.manifest,
            self.semantic_cache,
            design=self.design,
            control_contexts=self.control_contexts,
            receiver=self.receiver,
            measurement_runner=_measurer,
            sender_hardware_id="sender-cpu",
            receiver_hardware_id="receiver-cpu",
            sender_revision="sender-revision",
            receiver_revision="receiver-revision",
            receiver_fusion_receipt=_fusion_receipt(),
            temporary_root=self.root,
        )

    def test_cache_has_unique_semantic_levels_and_runs_scientific_frontier(self) -> None:
        with mock.patch.dict(sys.modules, {"cbor2": _fake_cbor_module()}):
            cache = self.build()
            self.assertEqual(
                set(cache["variant"]),
                {
                    "l0_cbor", "l1_cbor", "l2_cbor", "cbor+jpeg_q50_roi",
                    "jpeg_q50_s1.0", "lossless_png_reference",
                },
            )
            self.assertFalse(cache.duplicated(["sample_id", "variant", "source_model"]).any())
            validate_native_receiver_cache(
                cache, self.manifest, self.semantic_cache, design=self.design,
                control_contexts=self.control_contexts,
            )
            frontier = experiment2_representation_frontier(
                self.manifest,
                scientific=True,
                semantic_packet_cache=self.semantic_cache,
                native_receiver_cache=cache,
                semantic_source_model="model_a",
                image_codecs=("JPEG",),
                image_qualities=(50,),
                image_scales=(1.0,),
            )
        semantic = frontier[frontier["mode"].isin({
            "L0_HEARTBEAT", "L1_ALERT", "L2_SEMANTICS",
        })]
        self.assertEqual(set(semantic["variant"]), {"l0_cbor", "l1_cbor", "l2_cbor"})
        self.assertTrue(np.isfinite(cache[[
            "generation_ms", "energy_j", "receiver_latency_ms", "receiver_energy_j",
        ]].to_numpy(float)).all())

    @unittest.skipUnless(resolve_ffmpeg_executable(), "FFmpeg is unavailable")
    def test_registered_receiver_builds_l0_through_video_cache_with_provenance(self) -> None:
        second_path = self.root / "registered-receiver-second.png"
        Image.new("RGB", (32, 24), "blue").save(second_path)
        self.manifest.loc[0, "clip_paths_json"] = json.dumps([
            str(self.image_path), str(second_path),
        ])
        detector = mock.Mock()
        detector.detect.return_value = ()
        receiver = RegisteredVisDroneNativeReceiver(
            detector,
            detector_revision="receiver-revision",
            priority_profiles={"general_mobility": ("car",)},
            registered_roi_xyxy=(0.25, 0.25, 0.75, 0.85),
            temporary_root=self.root / "registered_receiver_tmp",
        )
        with mock.patch.dict(sys.modules, {"cbor2": _fake_cbor_module()}):
            cache = build_native_receiver_cache(
                self.manifest,
                self.semantic_cache,
                design=self.design,
                control_contexts=self.control_contexts,
                receiver=receiver,
                measurement_runner=_measurer,
                sender_hardware_id="sender-cpu",
                receiver_hardware_id="receiver-cpu",
                sender_revision="sender-revision",
                receiver_revision="receiver-revision",
                receiver_fusion_receipt=receiver.fusion_receipt,
                temporary_root=self.root,
            )
            validate_native_receiver_cache(
                cache,
                self.manifest,
                self.semantic_cache,
                design=self.design,
                control_contexts=self.control_contexts,
            )

        actions = cache.set_index("variant")["receiver_action"].to_dict()
        self.assertEqual(
            set(actions),
            {
                "l0_cbor", "l1_cbor", "l2_cbor", "cbor+jpeg_q50_roi",
                "jpeg_q50_s1.0", "lossless_png_reference",
                "h264_crf23_fps5",
            },
        )
        self.assertEqual(actions["l0_cbor"], "NO_ACTION")
        self.assertEqual(actions["l1_cbor"], "RESPOND")
        self.assertEqual(actions["l2_cbor"], "RESPOND")
        self.assertEqual(actions["cbor+jpeg_q50_roi"], "RESPOND")
        self.assertEqual(actions["jpeg_q50_s1.0"], "REVIEW")
        self.assertEqual(actions["lossless_png_reference"], "REVIEW")
        self.assertEqual(actions["h264_crf23_fps5"], "REVIEW")
        self.assertEqual(
            receiver.fusion_receipt["action_source_by_semantic_variant"]["l0_cbor"],
            "validated_heartbeat_then_registered_NO_ACTION",
        )
        expected_fusion_json = json.dumps(
            receiver.fusion_receipt, sort_keys=True, separators=(",", ":"),
        )
        self.assertEqual(
            set(cache["receiver_fusion_policy_json"]), {expected_fusion_json},
        )
        self.assertEqual(
            set(cache["receiver_fusion_policy_version"]),
            {"semuav.visdrone-native-fusion.v3"},
        )
        self.assertEqual(
            set(cache["receiver_fusion_policy_sha256"]),
            {hash_object(receiver.fusion_receipt)},
        )
        self.assertEqual(set(cache["sender_revision"]), {"sender-revision"})
        self.assertEqual(set(cache["receiver_revision"]), {"receiver-revision"})
        self.assertEqual(
            set(cache["control_context_sha256"]),
            {self.control_contexts["s1"].sha256},
        )
        self.assertEqual(detector.detect.call_count, 5)

    def test_cache_validation_recomputes_exact_transmitted_payload_hash(self) -> None:
        with mock.patch.dict(sys.modules, {"cbor2": _fake_cbor_module()}):
            cache = self.build()
            corrupted = cache.copy()
            corrupted.loc[
                corrupted["variant"].eq("jpeg_q50_s1.0"), "payload_sha256",
            ] = hashlib.sha256(b"not-the-transmitted-image").hexdigest()
            with self.assertRaisesRegex(StudyGateError, "payload hash"):
                validate_native_receiver_cache(
                    corrupted,
                    self.manifest,
                    self.semantic_cache,
                    design=self.design,
                    control_contexts=self.control_contexts,
                )

    def test_native_cache_binds_gold_free_pre_shared_control_context(self) -> None:
        with mock.patch.dict(sys.modules, {"cbor2": _fake_cbor_module()}):
            cache = self.build()
            expected_hash = self.control_contexts["s1"].sha256
            self.assertEqual(set(cache["control_context_sha256"]), {expected_hash})

            corrupted = cache.copy()
            corrupted.loc[0, "control_context_sha256"] = "0" * 64
            with self.assertRaisesRegex(StudyGateError, "control context"):
                validate_native_receiver_cache(
                    corrupted,
                    self.manifest,
                    self.semantic_cache,
                    design=self.design,
                    control_contexts=self.control_contexts,
                    verify_payload_hashes=False,
                )

            other_context = {
                "s1": ReceiverControlContext(
                    mission_profile="heavy_vehicle",
                    registered_roi_xyxy=(0.25, 0.25, 0.75, 0.85),
                    capture_time_us=0,
                    gps_lat=24.0,
                    gps_lon=54.0,
                    altitude_m=70.0,
                    yaw_deg=0.0,
                )
            }
            with self.assertRaisesRegex(StudyGateError, "control context"):
                validate_native_receiver_cache(
                    cache,
                    self.manifest,
                    self.semantic_cache,
                    design=self.design,
                    control_contexts=other_context,
                    verify_payload_hashes=False,
                )

    @unittest.skipUnless(resolve_ffmpeg_executable(), "FFmpeg is unavailable")
    def test_video_receiver_observes_frames_decoded_from_transmitted_bytes(self) -> None:
        second_path = self.root / "second.png"
        Image.new("RGB", (32, 24), "blue").save(second_path)
        self.manifest.loc[0, "clip_paths_json"] = json.dumps([
            str(self.image_path), str(second_path),
        ])
        observed: dict[str, object] = {}

        def receiver(request: DecodedNativePayload) -> str:
            if request.media_kind == "video":
                observed["frames"] = len(request.frames)
                observed["sizes"] = tuple(frame.size for frame in request.frames)
            if request.semantic_packet is not None:
                return str(request.semantic_packet.get("recommended_action", "NO_ACTION"))
            return "RESPOND"

        with mock.patch.dict(sys.modules, {"cbor2": _fake_cbor_module()}):
            cache = build_native_receiver_cache(
                self.manifest,
                self.semantic_cache,
                design=self.design,
                control_contexts=self.control_contexts,
                receiver=receiver,
                measurement_runner=_measurer,
                sender_hardware_id="sender-cpu",
                receiver_hardware_id="receiver-cpu",
                sender_revision="sender-revision",
                receiver_revision="receiver-revision",
                receiver_fusion_receipt=_fusion_receipt(),
                temporary_root=self.root,
            )
            validate_native_receiver_cache(
                cache, self.manifest, self.semantic_cache, design=self.design,
                control_contexts=self.control_contexts,
            )
            frontier = experiment2_representation_frontier(
                self.manifest,
                scientific=True,
                semantic_packet_cache=self.semantic_cache,
                native_receiver_cache=cache,
                semantic_source_model="model_a",
                image_codecs=("JPEG",),
                image_qualities=(50,),
                image_scales=(1.0,),
                video_crfs=(23,),
                video_fps=5,
            )
            self.assertIn("h264_crf23_fps5", set(frontier["variant"]))

            # Extra cache variants are just as dangerous as missing variants:
            # they indicate the cache and candidate design were frozen under
            # different payload grids.
            extra = cache[cache["variant"].eq("h264_crf23_fps5")].copy()
            extra["variant"] = "h264_crf35_fps5"
            mismatched = pd.concat([cache, extra], ignore_index=True)
            with self.assertRaisesRegex(StudyGateError, "variant coverage differs"):
                experiment2_representation_frontier(
                    self.manifest,
                    scientific=True,
                    semantic_packet_cache=self.semantic_cache,
                    native_receiver_cache=mismatched,
                    semantic_source_model="model_a",
                    image_codecs=("JPEG",),
                    image_qualities=(50,),
                    image_scales=(1.0,),
                    video_crfs=(23,),
                    video_fps=5,
                )
        self.assertEqual(observed["frames"], 2)
        self.assertEqual(observed["sizes"], ((32, 24), (32, 24)))
        self.assertIn("h264_crf23_fps5", set(cache["variant"]))


class VisDroneFeatureCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        image_path = self.root / "sample.png"
        Image.new("RGB", (10, 8), (10, 20, 30)).save(image_path)
        self.manifest = pd.DataFrame([{
            "sample_id": "s1",
            "image_path": str(image_path),
            "gold_action": "RESPOND",
            "visual_group_id": "g1",
            "clip_paths_json": "[]",
        }])
        self.design = FeatureCacheDesign(
            encoder_revision="encoder-commit",
            split_layer="backbone.block4",
            receiver_head_revision="head-commit",
            sender_hardware_id="sender-cpu",
            receiver_hardware_id="receiver-cpu",
        )
        self.control_contexts = {
            "s1": ReceiverControlContext(
                mission_profile="general_mobility",
                registered_roi_xyxy=(0.25, 0.25, 0.75, 0.85),
                capture_time_us=0,
                gps_lat=24.0,
                gps_lon=54.0,
                altitude_m=70.0,
                yaw_deg=0.0,
            )
        }

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_feature_cache_uses_quantized_wire_tensor_and_has_full_coverage(self) -> None:
        seen = []

        def encoder(request):
            self.assertFalse(hasattr(request, "gold_action"))
            return np.linspace(-1, 2, 24, dtype=np.float32).reshape(2, 3, 4)

        def head(request):
            self.assertFalse(hasattr(request, "gold_action"))
            self.assertEqual(request.mission_profile, "general_mobility")
            self.assertEqual(
                request.registered_roi_xyxy, (0.25, 0.25, 0.75, 0.85),
            )
            self.assertEqual(request.control_context_sha256, request.control_context.sha256)
            seen.append((request.variant, request.features.copy()))
            return {"receiver_action": "RESPOND"}

        cache = build_registered_feature_cache(
            self.manifest,
            design=self.design,
            control_contexts=self.control_contexts,
            encoder=encoder,
            receiver_head=head,
            measurement_runner=_measurer,
        )
        validate_registered_feature_cache(
            cache,
            self.manifest,
            design=self.design,
            control_contexts=self.control_contexts,
        )
        self.assertEqual(
            set(cache["variant"]),
            {"registered_feature_int8", "registered_feature_int4"},
        )
        self.assertEqual(set(cache["tensor_shape"]), {"[2,3,4]"})
        self.assertEqual(
            dict(zip(cache["quantization"], cache["payload_bytes"])),
            {"int8": 52, "int4": 40},
        )
        self.assertEqual(len(seen), 2)
        self.assertFalse(np.array_equal(seen[0][1], np.linspace(-1, 2, 24).reshape(2, 3, 4)))

    def test_feature_cache_validation_rejects_payload_length_or_revision_tampering(self) -> None:
        cache = build_registered_feature_cache(
            self.manifest,
            design=self.design,
            control_contexts=self.control_contexts,
            encoder=lambda request: np.ones((2, 3), dtype=np.float32),
            receiver_head=lambda request: "NO_ACTION",
            measurement_runner=_measurer,
        )
        wrong_length = cache.copy()
        wrong_length.loc[0, "payload_bytes"] += 1
        with self.assertRaisesRegex(StudyGateError, "payload length"):
            validate_registered_feature_cache(
                wrong_length, self.manifest, design=self.design,
                control_contexts=self.control_contexts,
            )

        wrong_context = cache.copy()
        wrong_context.loc[0, "control_context_sha256"] = "0" * 64
        with self.assertRaisesRegex(StudyGateError, "control context"):
            validate_registered_feature_cache(
                wrong_context,
                self.manifest,
                design=self.design,
                control_contexts=self.control_contexts,
            )
        wrong_revision = cache.copy()
        wrong_revision.loc[0, "encoder_revision"] = "other"
        with self.assertRaisesRegex(StudyGateError, "encoder_revision"):
            validate_registered_feature_cache(
                wrong_revision, self.manifest, design=self.design,
                control_contexts=self.control_contexts,
            )

    def test_scientific_frontier_counts_complete_feature_wire_payload(self) -> None:
        semantic_cache = _semantic_cache()
        native_design = NativeReceiverDesign(
            semantic_source_model="model_a",
            image_codecs=("JPEG",),
            image_qualities=(50,),
            image_scales=(1.0,),
            video_crfs=(23,),
        )

        def native_receiver(request):
            if request.semantic_packet is not None:
                return request.semantic_packet.get("recommended_action", "NO_ACTION")
            return "RESPOND"

        feature_cache = build_registered_feature_cache(
            self.manifest,
            design=self.design,
            control_contexts=self.control_contexts,
            encoder=lambda request: np.ones((2, 3), dtype=np.float32),
            receiver_head=lambda request: "RESPOND",
            measurement_runner=_measurer,
        )
        with mock.patch.dict(sys.modules, {"cbor2": _fake_cbor_module()}):
            native_cache = build_native_receiver_cache(
                self.manifest,
                semantic_cache,
                design=native_design,
                control_contexts=self.control_contexts,
                receiver=native_receiver,
                measurement_runner=_measurer,
                sender_hardware_id="sender-cpu",
                receiver_hardware_id="receiver-cpu",
                sender_revision="sender-revision",
                receiver_revision="receiver-revision",
                receiver_fusion_receipt=_fusion_receipt(),
            )
            frontier = experiment2_representation_frontier(
                self.manifest,
                scientific=True,
                semantic_packet_cache=semantic_cache,
                native_receiver_cache=native_cache,
                registered_feature_cache=feature_cache,
                semantic_source_model="model_a",
                image_codecs=("JPEG",),
                image_qualities=(50,),
                image_scales=(1.0,),
            )
        l4 = frontier[frontier["mode"].eq("L4_FEATURES")].set_index("variant")
        expected = feature_cache.set_index("variant")
        self.assertEqual(
            l4["payload_bytes"].astype(int).to_dict(),
            expected["payload_bytes"].astype(int).to_dict(),
        )
        for payload_hex in l4["payload_blob"]:
            deserialize_quantized_tensor(bytes.fromhex(payload_hex))

    def test_quantized_tensor_wire_payload_carries_every_decode_parameter(self) -> None:
        values = np.linspace(-3, 7, 35, dtype=np.float32).reshape(5, 7)
        for bits in (8, 4):
            with self.subTest(bits=bits):
                quantized = quantize_tensor(values, bits)
                payload = serialize_quantized_tensor(quantized)
                decoded = deserialize_quantized_tensor(payload)
                self.assertEqual(decoded.bits, bits)
                self.assertEqual(decoded.shape, values.shape)
                self.assertEqual(decoded.scale, quantized.scale)
                self.assertEqual(decoded.zero_point, quantized.zero_point)
                np.testing.assert_array_equal(
                    dequantize_tensor(decoded), dequantize_tensor(quantized),
                )
                self.assertEqual(len(payload), quantized.wire_bytes)
                with self.assertRaisesRegex(ValueError, "data length"):
                    deserialize_quantized_tensor(payload + b"hidden-byte")


if __name__ == "__main__":
    unittest.main()
