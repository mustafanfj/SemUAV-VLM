"""Weak native detections must remain valid registered abstentions.

The receiver regressions use its actual DetectorOnlyBackend and packet
validator. Only detector predictions are supplied as deterministic fixtures;
the scientific policy, decoder-facing receiver and schema are not mocked.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from semuav.perception import DetectorOnlyBackend, InferenceRequest
from semuav.protocol import validate_semantic_content_packet
from semuav.registry import StudyGateError
from semuav.visdrone_receiver_runtime import RegisteredVisDroneNativeReceiver
from semuav.visdrone_receivers import DecodedNativePayload, ReceiverControlContext


ROI = (0.25, 0.25, 0.75, 0.85)
INSIDE = (0.4, 0.4, 0.6, 0.6)
OUTSIDE = (0.0, 0.0, 0.1, 0.1)


def proposal(label="truck", confidence=0.2, box=INSIDE):
    return {
        "label": label,
        "confidence": confidence,
        "bbox_normalized_xyxy": list(box),
    }


def request(proposals=(), *, mission="visdrone_urban", profile="heavy_vehicle"):
    return InferenceRequest(
        "frozen-valid", "unused.png", mission, 123,
        {
            "gps_lat": 24.0, "gps_lon": 54.0, "altitude_m": 70.0,
            "yaw_deg": 0.0, "mission_profile": profile,
            "registered_roi_xyxy": list(ROI),
        },
        tuple(proposals),
    )


def backend():
    return DetectorOnlyBackend(
        "a" * 40,
        urban_policy="registered_roi_occupancy",
        urban_priority_labels=("truck",),
        urban_priority_profiles={"heavy_vehicle": ("truck",)},
        urban_priority_roi_xyxy=ROI,
        urban_priority_confidence=0.05,
    )


# Frozen from the unmodified v2r12 policy using these exact inputs. Digests
# cover every packet field plus action confidence and confidence provenance;
# nondeterministic latency is deliberately excluded. This guards compatibility
# of previously valid raw Experiment-1 content without loading an old release
# or editing the implementation under test.
VALID_PREDECESSOR_CASES = (
    ("empty", (), "insufficient_evidence", "REVIEW",
     "38cf618b4e8ede65e43bf8d8afb2d2f03aabf1b3784920aa19b149f3e54ed540"),
    ("below_priority", (proposal(confidence=0.04),),
     "insufficient_evidence", "REVIEW",
     "d953d58ef4fb75c29ecdc591dc49832809f4ad62501ebfa306b925569d05ae03"),
    ("priority_inside", (proposal(confidence=0.875),),
     "visible_priority_roi_occupancy", "RESPOND",
     "5d58b0aecafb3de95982ddbcc952f22aa15fdeb744a2f1f162f39604302a069b"),
    ("priority_outside", (proposal(confidence=0.875, box=OUTSIDE),),
     "visible_priority_outside_roi", "REVIEW",
     "444368351c63a1ecc4da4f3aa55dc836a1240edc880c75c4b21814f573718cc2"),
    ("nonpriority", (proposal("car", 0.875),),
     "no_registered_priority_visible", "NO_ACTION",
     "18f27f752e4948affddb976dedfbda40e608f0ed5fe40d434fca81a5fcc235fb"),
    ("evidence_threshold", (proposal(confidence=0.35),),
     "visible_priority_roi_occupancy", "REVIEW",
     "d93d49452e037c26a4162248554fd7d565405fea5ed96c55308598d3a6b258aa"),
    ("mixed", (proposal(confidence=0.125), proposal("car", 0.875, OUTSIDE)),
     "visible_priority_roi_occupancy", "REVIEW",
     "e08fb63f775310837a6b690eacb9299fd6c697f08e03bb746a26688aba2327ca"),
)


class DetectorEvidenceContractTests(unittest.TestCase):
    def assert_abstention(self, record):
        self.assertTrue(record.schema_valid)
        packet = record.parsed
        self.assertIsNotNone(packet)
        self.assertFalse(packet["sufficient_evidence"])
        self.assertEqual(packet["event"], "insufficient_evidence")
        self.assertEqual(packet["recommended_action"], "REVIEW")
        self.assertEqual(validate_semantic_content_packet(packet), packet)
        self.assertEqual(json.loads(record.raw_text), packet)

    def test_weak_priority_inside_and_outside_are_valid_abstentions(self):
        for box in (INSIDE, OUTSIDE):
            for confidence in (0.0, 0.049999, 0.05, 0.2, 0.349999):
                with self.subTest(box=box, confidence=confidence):
                    record = backend().infer(request((proposal(confidence=confidence, box=box),)))
                    self.assert_abstention(record)
                    self.assertEqual(record.parsed["objects"][0]["confidence"], confidence)
                    self.assertEqual(record.parsed["objects"][0]["bbox_normalized_xyxy"], list(box))
                    self.assertEqual(record.parsed["uncertainty"], 1.0 - confidence)

    def test_evidence_threshold_and_respond_rule_are_unchanged(self):
        cases = (
            (0.35, INSIDE, "visible_priority_roi_occupancy", "REVIEW"),
            (0.849999, INSIDE, "visible_priority_roi_occupancy", "REVIEW"),
            (0.85, INSIDE, "visible_priority_roi_occupancy", "RESPOND"),
            (0.875, INSIDE, "visible_priority_roi_occupancy", "RESPOND"),
            (0.35, OUTSIDE, "visible_priority_outside_roi", "REVIEW"),
            (0.875, OUTSIDE, "visible_priority_outside_roi", "REVIEW"),
        )
        for confidence, box, event, action in cases:
            with self.subTest(confidence=confidence, box=box):
                record = backend().infer(request((proposal(confidence=confidence, box=box),)))
                self.assertTrue(record.schema_valid)
                self.assertTrue(record.parsed["sufficient_evidence"])
                self.assertEqual(record.parsed["event"], event)
                self.assertEqual(record.parsed["recommended_action"], action)

    def test_empty_and_weak_nonpriority_detections_abstain(self):
        for proposals in ((), (proposal("car", 0.2),),
                          (proposal("car", 0.2), proposal("truck", 0.125, OUTSIDE))):
            with self.subTest(proposals=proposals):
                self.assert_abstention(backend().infer(request(proposals)))

    def test_strong_nonpriority_evidence_does_not_suppress_weak_priority_geometry(self):
        for box, event in ((INSIDE, "visible_priority_roi_occupancy"),
                           (OUTSIDE, "visible_priority_outside_roi")):
            with self.subTest(box=box):
                record = backend().infer(request((
                    proposal(confidence=0.125, box=box), proposal("car", 0.875, OUTSIDE),
                )))
                self.assertTrue(record.parsed["sufficient_evidence"])
                self.assertEqual(record.parsed["event"], event)
                self.assertEqual(record.parsed["recommended_action"], "REVIEW")

    def test_unknown_profile_rejected_even_without_sufficient_evidence(self):
        for proposals in ((), (proposal(),), (proposal(confidence=0.875),)):
            with self.subTest(proposals=proposals):
                with self.assertRaisesRegex(StudyGateError, "Unregistered VisDrone mission_profile"):
                    backend().infer(request(proposals, profile="unknown_profile"))

    def test_low_confidence_wildfire_smoke_and_flame_abstain(self):
        for label in ("smoke", "flame"):
            for confidence in (0.05, 0.2, 0.349999):
                with self.subTest(label=label, confidence=confidence):
                    record = DetectorOnlyBackend("a" * 40).infer(request(
                        (proposal(label, confidence),), mission="wildfire",
                    ))
                    self.assert_abstention(record)
                    self.assertEqual(record.parsed["risk"], confidence)

    def test_sufficient_wildfire_event_unchanged(self):
        for label, event in (("smoke", "visible_smoke"), ("flame", "visible_flame")):
            with self.subTest(label=label):
                record = DetectorOnlyBackend("a" * 40).infer(request(
                    (proposal(label, 0.875),), mission="wildfire",
                ))
                self.assertTrue(record.parsed["sufficient_evidence"])
                self.assertEqual(record.parsed["event"], event)
                self.assertEqual(record.parsed["recommended_action"], "RESPOND")

    def test_all_frozen_predecessor_valid_packet_fields_are_identical(self):
        for name, proposals, event, action, expected_digest in VALID_PREDECESSOR_CASES:
            with self.subTest(case=name):
                record = backend().infer(request(proposals))
                self.assertEqual(record.parsed["event"], event)
                self.assertEqual(record.parsed["recommended_action"], action)
                content = {
                    "packet": record.parsed,
                    "confidence": record.confidence,
                    "confidence_source": record.confidence_source,
                }
                digest = hashlib.sha256(json.dumps(
                    content, sort_keys=True, separators=(",", ":"),
                ).encode("utf-8")).hexdigest()
                self.assertEqual(digest, expected_digest)


class _PredictionSequence:
    def __init__(self, predictions):
        self.predictions = iter(predictions)
        self.calls = []

    def detect(self, image_path, *, persist=False):
        # The actual receiver must materialize a readable reconstructed image.
        with Image.open(image_path) as image:
            image.load()
        self.calls.append((image_path, persist))
        return next(self.predictions)


class NativeWeakEvidenceRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.context = ReceiverControlContext(
            mission_profile="heavy_vehicle", registered_roi_xyxy=ROI,
            capture_time_us=123, gps_lat=24.0, gps_lon=54.0,
            altitude_m=70.0, yaw_deg=0.0,
        )

    def run_receiver(self, predictions, *, media_kind="image", semantic_action=None):
        detector = _PredictionSequence(predictions)
        context_before = self.context.as_dict()
        with tempfile.TemporaryDirectory() as directory:
            receiver = RegisteredVisDroneNativeReceiver(
                detector, detector_revision="a" * 40,
                priority_profiles={"heavy_vehicle": ("truck",)},
                registered_roi_xyxy=ROI, priority_confidence=0.05,
                temporary_root=directory,
            )
            self.assertIsInstance(receiver.policy, DetectorOnlyBackend)
            self.assertIsInstance(receiver.roi_crop_policy, DetectorOnlyBackend)
            decoded = DecodedNativePayload(
                sample_id="weak-evidence-receiver", source_model="ground_visual_receiver",
                variant={"image": "jpeg_q50_s1", "video": "h264_crf23",
                         "semantic_roi": "cbor+jpeg_q50_roi"}[media_kind],
                media_kind=media_kind, payload_sha256="b" * 64,
                control_context=self.context, control_context_sha256=self.context.sha256,
                frames=tuple(Image.new("RGB", (16, 16), "white") for _ in predictions),
                semantic_packet=(
                    {"recommended_action": semantic_action}
                    if semantic_action is not None else None
                ),
                transmitted_roi_xyxy=ROI if media_kind == "semantic_roi" else None,
            )
            action = receiver(decoded)
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(self.context.as_dict(), context_before)
        return action, detector.calls

    def test_full_frame_receiver_returns_review_for_weak_priority_predictions(self):
        for box in (INSIDE, OUTSIDE):
            with self.subTest(box=box):
                action, calls = self.run_receiver(((proposal(box=box),),))
                self.assertEqual(action, "REVIEW")
                self.assertEqual([persist for _, persist in calls], [False])

    def test_crop_receiver_preserves_fusion_for_weak_pixels(self):
        for semantic, expected in (("NO_ACTION", "REVIEW"), ("REVIEW", "REVIEW"),
                                   ("RESPOND", "RESPOND")):
            with self.subTest(semantic=semantic):
                action, calls = self.run_receiver(
                    ((proposal(box=OUTSIDE),),), media_kind="semantic_roi",
                    semantic_action=semantic,
                )
                self.assertEqual(action, expected)
                self.assertEqual([persist for _, persist in calls], [False])

    def test_crop_geometry_still_covers_the_whole_received_crop(self):
        action, _ = self.run_receiver(
            ((proposal(confidence=0.875, box=OUTSIDE),),),
            media_kind="semantic_roi", semantic_action="NO_ACTION",
        )
        self.assertEqual(action, "RESPOND")

    def test_clip_uses_final_frame_when_confidence_crosses_evidence_threshold(self):
        weak = (proposal(confidence=0.2),)
        strong = (proposal(confidence=0.875),)
        nonpriority = (proposal("car", 0.875),)
        for predictions, expected in (((strong, weak), "REVIEW"),
                                       ((weak, strong), "RESPOND"),
                                       ((weak, nonpriority), "NO_ACTION")):
            with self.subTest(expected=expected):
                action, calls = self.run_receiver(predictions, media_kind="video")
                self.assertEqual(action, expected)
                self.assertEqual([persist for _, persist in calls], [False, True])


if __name__ == "__main__":
    unittest.main()
