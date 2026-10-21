from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from semuav.perception import (
    DETECTOR_PROPOSAL_ORDERING,
    UltralyticsDetectorTracker,
    detector_execution_revision,
    detector_inference_spec,
)


class _Tensor:
    def __init__(self, values):
        self.values = values

    def int(self):
        def convert(value):
            if isinstance(value, list):
                return [convert(item) for item in value]
            return int(value)

        return _Tensor(convert(self.values))

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class _Boxes:
    def __init__(self):
        self.xyxy = _Tensor([
            [10, 10, 40, 50],   # unsupported label; filtered before top-k
            [20, 10, 80, 50],   # car, track 5
            [20, 10, 80, 50],   # car, track 2 (tie breaker)
            [30, 10, 90, 50],   # person alias -> pedestrian
            [40, 10, 100, 50],  # below frozen minimum confidence
        ])
        self.cls = _Tensor([99, 4, 4, 0, 6])
        self.conf = _Tensor([0.99, 0.90, 0.90, 0.90, 0.04])
        self.id = _Tensor([1, 5, 2, 3, 4])

    def __len__(self):
        return 5


class _Result:
    boxes = _Boxes()
    orig_shape = (100, 200)
    names = {0: "person", 4: "car", 6: "truck", 99: "airplane"}


class _SpyYOLO:
    def __init__(self, checkpoint: str):
        self.checkpoint = checkpoint
        self.calls = []

    def track(self, image_path: str, **kwargs):
        self.calls.append((image_path, kwargs))
        return [_Result()]


class FrozenDetectorPolicyTests(unittest.TestCase):
    def test_detect_forwards_frozen_parameters_filters_sorts_and_caps_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "detector.pt"
            checkpoint.write_bytes(b"weights")
            instances = []

            def yolo_factory(value):
                instance = _SpyYOLO(value)
                instances.append(instance)
                return instance

            fake_module = types.SimpleNamespace(YOLO=yolo_factory)
            with patch.dict(sys.modules, {"ultralytics": fake_module}):
                detector = UltralyticsDetectorTracker(
                    str(checkpoint),
                    "botsort.yaml",
                    "cuda:1",
                    image_size=1280,
                    confidence=0.001,
                    iou=0.70,
                    max_detections=300,
                    seed=20260823,
                    top_k=3,
                    minimum_confidence=0.05,
                    ordering=DETECTOR_PROPOSAL_ORDERING,
                )
            proposals = detector.detect("frame.jpg", persist=True)

            self.assertEqual(instances[0].checkpoint, str(checkpoint))
            image_path, kwargs = instances[0].calls[0]
            self.assertEqual(image_path, "frame.jpg")
            self.assertEqual(kwargs, {
                "tracker": "botsort.yaml",
                "persist": True,
                "device": "cuda:1",
                "imgsz": 1280,
                "conf": 0.001,
                "iou": 0.70,
                "max_det": 300,
                "deterministic": True,
                "seed": 20260823,
                "verbose": False,
            })
            self.assertEqual(len(proposals), 3)
            self.assertEqual(
                [(item["label"], item["track_id"]) for item in proposals],
                [("car", 2), ("car", 5), ("pedestrian", 3)],
            )
            self.assertTrue(all(item["confidence"] >= 0.05 for item in proposals))
            self.assertNotIn("airplane", {item["label"] for item in proposals})

    def test_revision_binds_every_frozen_inference_and_proposal_parameter(self) -> None:
        baseline = detector_execution_revision("a" * 64, "botsort.yaml")
        variants = (
            {"image_size": 960},
            {"confidence": 0.01},
            {"iou": 0.60},
            {"max_detections": 200},
            {"seed": 7},
            {"top_k": 16},
            {"minimum_confidence": 0.10},
        )
        for values in variants:
            with self.subTest(values=values):
                self.assertNotEqual(
                    baseline,
                    detector_execution_revision("a" * 64, "botsort.yaml", **values),
                )

    def test_invalid_or_unfrozen_detector_policy_fails_closed(self) -> None:
        invalid = (
            {"image_size": 0},
            {"confidence": -0.1},
            {"iou": 1.1},
            {"max_detections": 0},
            {"seed": -1},
            {"top_k": 301},
            {"minimum_confidence": 0.0001},
            {"ordering": "input_order"},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                detector_inference_spec(**values)


if __name__ == "__main__":
    unittest.main(verbosity=2)
