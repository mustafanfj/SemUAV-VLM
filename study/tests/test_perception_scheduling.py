from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import pandas as pd

from semuav.control import DeadlineGreedyPolicy, FixedVariantPolicy
from semuav.experiments import run_paired_policy_trials
from semuav.network import Direction, constant_trace
from semuav.perception import UltralyticsDetectorTracker
from semuav.protocol import PayloadMode, encode_mapping


class PerceptionSchedulingTests(unittest.TestCase):
    def test_cached_detector_is_materialized_before_vlm_measurement(self) -> None:
        class FakeParameter:
            device = "cpu"

            @staticmethod
            def numel() -> int:
                return 3

        class FakeUnderlying:
            def __init__(self) -> None:
                self.evaluated = False

            def eval(self) -> "FakeUnderlying":
                self.evaluated = True
                return self

            @staticmethod
            def parameters():
                return iter([FakeParameter()])

        class FakeYOLO:
            def __init__(self) -> None:
                self.model = FakeUnderlying()
                self.moves: list[str] = []

            def to(self, device: str) -> "FakeYOLO":
                self.moves.append(device)
                return self

        tracker = UltralyticsDetectorTracker.__new__(UltralyticsDetectorTracker)
        tracker.model = FakeYOLO()
        tracker.tracker = "botsort.yaml"
        tracker.device = "cpu"
        report = tracker.ensure_resident()
        self.assertEqual(tracker.model.moves, ["cpu"])
        self.assertTrue(tracker.model.model.evaluated)
        self.assertEqual(report["observed_device"], "cpu")
        self.assertEqual(report["parameter_count"], 3)
        self.assertEqual(
            report["strategy"],
            "always_on_detector_plus_one_triggered_context_model",
        )

    def test_detector_residency_rejects_empty_or_mixed_device_parameters(self) -> None:
        class Parameter:
            def __init__(self, device: str, count: int = 1) -> None:
                self.device, self.count = device, count

            def numel(self) -> int:
                return self.count

        class Underlying:
            def __init__(self, parameters: list[Parameter]) -> None:
                self._parameters = parameters

            def eval(self) -> "Underlying":
                return self

            def parameters(self):
                return iter(self._parameters)

        class YOLO:
            def __init__(self, parameters: list[Parameter]) -> None:
                self.model = Underlying(parameters)

            def to(self, device: str) -> "YOLO":
                return self

        for parameters in ([], [Parameter("cpu"), Parameter("meta")]):
            tracker = UltralyticsDetectorTracker.__new__(UltralyticsDetectorTracker)
            tracker.model, tracker.tracker, tracker.device = YOLO(parameters), "botsort.yaml", "cpu"
            with self.subTest(parameters=parameters), self.assertRaises(Exception):
                tracker.ensure_resident()

        cuda_tracker = UltralyticsDetectorTracker.__new__(UltralyticsDetectorTracker)
        cuda_tracker.model = YOLO([Parameter("cuda:1")])
        cuda_tracker.tracker, cuda_tracker.device = "botsort.yaml", "cuda:0"
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda device: None,
            memory_allocated=lambda device: 1024,
        ))
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaises(Exception):
                cuda_tracker.ensure_resident()

    def setUp(self) -> None:
        packet = {
            "schema_version": "semuav.v1",
            "observation_id": "scene-1",
            "mission": "urban",
            "capture_time_us": 0,
            "telemetry": {
                "gps_lat": 24.0, "gps_lon": 54.0,
                "altitude_m": 50.0, "yaw_deg": 0.0,
            },
            "event": "visible_road_obstruction",
            "risk": 0.9,
            "uncertainty": 0.1,
            "recommended_action": "RESPOND",
        }
        semantic_payload, wire_format = encode_mapping(packet, scientific=False)
        self.semantic_variant = f"{wire_format}:deployed"
        self.image_variant = "opaque_image"
        self.candidates = pd.DataFrame([
            {
                "sample_id": "scene-1", "mode": "L1_ALERT",
                "variant": self.semantic_variant, "source_model": "compact-vlm",
                "payload_blob": semantic_payload.hex(), "receiver_action": "RESPOND",
                "generation_ms": 1.0, "energy_j": 1.0,
                "receiver_latency_ms": 0.0, "receiver_energy_j": 0.5,
                "calibrated_correctness_probability": 0.99,
            },
            {
                "sample_id": "scene-1", "mode": "L5_IMAGE_OR_CLIP",
                "variant": self.image_variant, "source_model": "ground_visual_receiver",
                "payload_blob": b"image".hex(), "receiver_action": "RESPOND",
                "generation_ms": 5.0, "energy_j": 0.25,
                "receiver_latency_ms": 0.0, "receiver_energy_j": 0.1,
                "calibrated_correctness_probability": 0.80,
            },
        ])
        self.profiles = pd.DataFrame([
            {
                "mode": "L1_ALERT", "variant": self.semantic_variant,
                "payload_bytes": len(semantic_payload), "generation_ms": 1.0,
                "energy_j": 1.0, "predicted_task_success": 0.99,
                "predicted_high_confidence_error": 0.0,
                "receiver_latency_ms": 0.0, "receiver_energy_j": 0.5,
            },
            {
                "mode": "L5_IMAGE_OR_CLIP", "variant": self.image_variant,
                "payload_bytes": 5, "generation_ms": 5.0,
                "energy_j": 0.25, "predicted_task_success": 0.80,
                "predicted_high_confidence_error": 0.0,
                "receiver_latency_ms": 0.0, "receiver_energy_j": 0.1,
            },
        ])
        self.traces = [
            constant_trace("perception-cost", Direction.UPLINK, 100.0, 0.0, 0.0),
            constant_trace("perception-cost", Direction.DOWNLINK, 100.0, 0.0, 0.0),
        ]

    @staticmethod
    def manifest(deadline_us: int) -> pd.DataFrame:
        return pd.DataFrame([{
            "sample_id": "scene-1", "visual_group_id": "group-1",
            "capture_time_us": 0, "trace_time_us": 0,
            "deadline_us": deadline_us, "mission_profile": "routine",
            "gold_action": "RESPOND", "critical": True, "ood": False,
            "risk": 0.9, "uncertainty": 0.1,
            "detector_risk": 0.9, "detector_uncertainty": 0.1,
            "context_triggered": True,
            "detector_latency_ms": 20.0, "detector_energy_j": 2.0,
            "context_model_latency_ms": 40.0, "context_model_energy_j": 3.0,
            "battery_fraction": 0.8, "thermal_c": 55.0,
        }])

    def test_pending_perception_is_visible_to_planner_and_charged_once(self) -> None:
        # Encoding-only cost makes L1 look feasible, but its still-pending
        # detector + VLM work pushes completion beyond this deadline.  L5 does
        # not require those stages and must therefore be selected.
        adaptive = run_paired_policy_trials(
            self.manifest(50_000), self.candidates, self.traces,
            [DeadlineGreedyPolicy()], (7,), self.profiles,
        ).iloc[0]
        self.assertEqual(adaptive.selected_mode, "L5_IMAGE_OR_CLIP")
        self.assertEqual(float(adaptive.detector_energy_j), 0.0)
        self.assertEqual(float(adaptive.context_model_energy_j), 0.0)

        # When L1 is forced, the actual readiness and energy ledger include the
        # 20 ms detector and 40 ms VLM exactly once, plus 1 ms encoding.
        fixed = run_paired_policy_trials(
            self.manifest(200_000), self.candidates, self.traces,
            [FixedVariantPolicy(
                PayloadMode.L1_ALERT, self.semantic_variant,
                "fixed_deployed_semantic",
            )],
            (7,), self.profiles,
        ).iloc[0]
        self.assertEqual(int(fixed.ready_us - fixed.capture_us), 61_000)
        self.assertAlmostEqual(float(fixed.detector_energy_j), 2.0)
        self.assertAlmostEqual(float(fixed.context_model_energy_j), 3.0)
        self.assertAlmostEqual(float(fixed.energy_j), 6.5)

        # The context-aware path runs both stages before the decision.  Its
        # physical capture-to-ready time and ledger must be identical—not add
        # the same stages again after the decision.
        predecision = run_paired_policy_trials(
            self.manifest(200_000), self.candidates, self.traces,
            [FixedVariantPolicy(
                PayloadMode.L1_ALERT, self.semantic_variant, "proposed_full",
            )],
            (7,), self.profiles,
        ).iloc[0]
        self.assertEqual(int(predecision.ready_us - predecision.capture_us), 61_000)
        self.assertAlmostEqual(float(predecision.energy_j), 6.5)

    def test_perception_server_serializes_captures_faster_than_vlm(self) -> None:
        second_candidates = self.candidates.copy()
        second_candidates["sample_id"] = "scene-2"
        candidates = pd.concat([self.candidates, second_candidates], ignore_index=True)

        first = self.manifest(200_000).iloc[0].to_dict()
        second = dict(first)
        second.update({
            "sample_id": "scene-2", "visual_group_id": "group-2",
            "capture_time_us": 10_000, "trace_time_us": 10_000,
            "deadline_us": 210_000,
        })
        overlap_manifest = pd.DataFrame([first, second])
        policies = [
            FixedVariantPolicy(
                PayloadMode.L1_ALERT, self.semantic_variant,
                "fixed_serial_semantic",
            ),
            FixedVariantPolicy(
                PayloadMode.L1_ALERT, self.semantic_variant, "proposed_full",
            ),
        ]
        rows = run_paired_policy_trials(
            overlap_manifest, candidates, self.traces, policies,
            (7,), self.profiles,
        ).sort_values(["policy", "capture_us"])

        for policy_name in ("fixed_serial_semantic", "proposed_full"):
            policy_rows = rows[rows["policy"].eq(policy_name)].reset_index(drop=True)
            # Capture 2 arrives at 10 ms while capture 1 owns the server until
            # 60 ms, so its perception starts at 60 ms and completes at 120 ms.
            self.assertEqual(policy_rows["perception_start_us"].astype(int).tolist(), [0, 60_000])
            self.assertEqual(policy_rows["perception_done_us"].astype(int).tolist(), [60_000, 120_000])
            self.assertEqual(policy_rows["perception_queue_wait_ms"].tolist(), [0.0, 50.0])
            self.assertEqual(policy_rows["perception_service_ms"].tolist(), [60.0, 60.0])
            # Representation encoding is a separate 1 ms stage after perception.
            self.assertEqual(policy_rows["ready_us"].astype(int).tolist(), [61_000, 121_000])
            self.assertTrue(policy_rows["energy_j"].eq(6.5).all())

        # The second ordinary-policy decision must see the existing 50 ms queue
        # plus 60 ms service in its planning profile and choose the feasible L5.
        planning_manifest = overlap_manifest.copy()
        # The simulator starts from a causal 100 ms one-way-delay estimate.  A
        # 200 ms first deadline admits semantic perception; the second 130 ms
        # absolute deadline admits L5 but not queued semantic perception.
        planning_manifest.loc[planning_manifest["sample_id"].eq("scene-1"), "deadline_us"] = 200_000
        planning_manifest.loc[planning_manifest["sample_id"].eq("scene-2"), "deadline_us"] = 130_000
        planned = run_paired_policy_trials(
            planning_manifest, candidates, self.traces,
            [DeadlineGreedyPolicy()], (7,), self.profiles,
        ).sort_values("capture_us")
        self.assertEqual(
            planned["selected_mode"].tolist(),
            ["L1_ALERT", "L5_IMAGE_OR_CLIP"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
