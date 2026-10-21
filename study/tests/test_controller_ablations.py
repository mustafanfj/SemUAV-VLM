from __future__ import annotations

import dataclasses
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from semuav.control import (
    CandidateProfile,
    ControllerObservation,
    FullPolicy,
    LinkEstimate,
    NoEvidenceEscalationPolicy,
    NoMissionRiskPolicy,
    NoReceiverStatePolicy,
    QueueSnapshot,
    RiskAwareNoUncertaintyPolicy,
)
from semuav.experiments import run_paired_policy_trials
from semuav.network import Direction, constant_trace
from semuav.protocol import PayloadMode, SenderReceiverBelief, StateKind, encode_mapping


def _observation(
    *,
    risk: float = 0.0,
    uncertainty: float = 0.0,
    mission_priority: float = 0.0,
    receiver_status: str = "known",
) -> ControllerObservation:
    return ControllerObservation(
        now_us=1_000_000,
        capture_us=1_000_000,
        action_deadline_us=5_000_000,
        link=LinkEstimate(1_000_000, 10, 100_000_000.0, 1_000.0, 0.0, 10_000.0),
        queue=QueueSnapshot(1_000_000, (0,) * 11, 0, 0, (None,) * 11, (0,) * 11),
        receiver_belief=SenderReceiverBelief(status=receiver_status),
        mission_priority=mission_priority,
        risk=risk,
        uncertainty=uncertainty,
        age_of_information_us=None,
        battery_fraction=0.8,
        thermal_c=55.0,
    )


def _candidate(
    mode: PayloadMode,
    *,
    success: float,
    evidence: float = 0.55,
    high_confidence_error: float = 0.0,
    state_kind: StateKind = StateKind.NONE,
    variant: str | None = None,
) -> CandidateProfile:
    return CandidateProfile(
        mode=mode,
        payload_bytes=20,
        generation_us=0,
        energy_j=0.0,
        predicted_task_success=success,
        predicted_high_confidence_error=high_confidence_error,
        evidence_strength=evidence,
        state_kind=state_kind,
        available_us=0,
        variant=variant or mode.name.lower(),
    )


class ControllerComponentAblationTests(unittest.TestCase):
    def test_mission_risk_ablation_removes_urgent_semantic_override(self) -> None:
        candidates = (
            _candidate(PayloadMode.L1_ALERT, success=0.55),
            _candidate(PayloadMode.L4_FEATURES, success=0.99),
        )
        observation = _observation(risk=0.95, mission_priority=0.9)
        self.assertEqual(FullPolicy().select(observation, candidates).mode, PayloadMode.L1_ALERT)
        self.assertEqual(
            NoMissionRiskPolicy().select(observation, candidates).mode,
            PayloadMode.L4_FEATURES,
        )

    def test_resource_score_includes_receiver_compute_energy(self) -> None:
        sender_cheap_receiver_expensive = dataclasses.replace(
            _candidate(PayloadMode.L1_ALERT, success=0.9),
            energy_j=0.1, receiver_energy_j=5.0,
        )
        sender_costlier_total_cheaper = dataclasses.replace(
            _candidate(PayloadMode.L2_SEMANTICS, success=0.9),
            energy_j=0.2, receiver_energy_j=0.1,
        )
        plan = FullPolicy(
            byte_weight=0.0, energy_weight=1.0, evidence_weight=0.0,
        ).select(
            _observation(),
            (sender_cheap_receiver_expensive, sender_costlier_total_cheaper),
        )
        self.assertEqual(plan.mode, PayloadMode.L2_SEMANTICS)

    def test_uncertainty_ablation_removes_observation_and_profile_error_trigger(self) -> None:
        candidates = (
            _candidate(PayloadMode.L2_SEMANTICS, success=0.91),
            _candidate(
                PayloadMode.L3_SEMANTICS_ROI,
                success=0.78,
                evidence=1.0,
                high_confidence_error=0.20,
            ),
        )
        observation = _observation(uncertainty=0.9)
        self.assertEqual(
            FullPolicy(byte_weight=0.0).select(observation, candidates).mode,
            PayloadMode.L3_SEMANTICS_ROI,
        )
        self.assertEqual(
            RiskAwareNoUncertaintyPolicy(byte_weight=0.0).select(
                observation, candidates,
            ).mode,
            PayloadMode.L2_SEMANTICS,
        )

    def test_no_evidence_comparator_is_strictly_l0_to_l2(self) -> None:
        candidates = (
            _candidate(PayloadMode.L2_SEMANTICS, success=0.40),
            _candidate(PayloadMode.L3_SEMANTICS_ROI, success=0.95, evidence=1.0),
            _candidate(PayloadMode.L4_FEATURES, success=0.99, evidence=1.0),
            _candidate(PayloadMode.L5_IMAGE_OR_CLIP, success=1.0, evidence=1.0),
        )
        plan = NoEvidenceEscalationPolicy(byte_weight=0.0).select(
            _observation(uncertainty=0.9), candidates,
        )
        self.assertEqual(plan.mode, PayloadMode.L2_SEMANTICS)

    def test_receiver_state_ablation_removes_repair_override(self) -> None:
        candidates = (
            _candidate(
                PayloadMode.L2_SEMANTICS,
                success=0.35,
                state_kind=StateKind.KEYFRAME,
                variant="keyframe",
            ),
            _candidate(
                PayloadMode.L2_SEMANTICS,
                success=0.95,
                state_kind=StateKind.DELTA,
                variant="delta",
            ),
        )
        observation = _observation(receiver_status="uncertain")
        self.assertEqual(FullPolicy().select(observation, candidates).variant, "keyframe")
        self.assertEqual(
            NoReceiverStatePolicy().select(observation, candidates).variant,
            "delta",
        )


class EpisodeAblationIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.traces = [
            constant_trace("ablation", Direction.UPLINK, 100.0, 0.0, 0.0, duration_s=20),
            constant_trace("ablation", Direction.DOWNLINK, 100.0, 0.0, 0.0, duration_s=20),
        ]

    @staticmethod
    def _semantic_packet(
        sample_id: str,
        event: str = "visible_road_obstruction",
        wire_format: str | None = None,
    ) -> tuple[bytes, str]:
        packet = {
            "schema_version": "semuav.v1",
            "observation_id": sample_id,
            "mission": "urban",
            "capture_time_us": 0,
            "telemetry": {
                "gps_lat": 24.0,
                "gps_lon": 54.0,
                "altitude_m": 50.0,
                "yaw_deg": 0.0,
            },
            "sufficient_evidence": True,
            "event": event,
            "objects": [],
            "relations": [],
            "risk": 0.5,
            "uncertainty": 0.2,
            "recommended_action": "RESPOND",
        }
        # ``scientific=False`` permits the JSON debug fallback when cbor2 is
        # unavailable; it does not force JSON when cbor2 is installed.  Keep
        # the fixture's variant bound to the encoder's actual wire format so
        # the same tests exercise both supported dependency environments.
        return encode_mapping(
            packet,
            scientific=False,
            wire_format=wire_format,
        )

    def _trigger_inputs(self, detector_risk: float, detector_uncertainty: float):
        deployed, deployed_wire_format = self._semantic_packet("scene-1")
        detector, detector_wire_format = self._semantic_packet("scene-1")
        candidates = pd.DataFrame([
            {
                "sample_id": "scene-1", "mode": "L1_ALERT",
                "variant": f"{deployed_wire_format}:compact-vlm", "source_model": "compact-vlm",
                "payload_blob": deployed.hex(), "receiver_action": "RESPOND",
                "generation_ms": 1.0, "energy_j": 0.1,
                "receiver_latency_ms": 0.0, "receiver_energy_j": 0.0,
                "calibrated_correctness_probability": 0.9,
            },
            {
                "sample_id": "scene-1", "mode": "L1_ALERT",
                "variant": f"{detector_wire_format}:detector", "source_model": "detector_only",
                "payload_blob": detector.hex(), "receiver_action": "RESPOND",
                "generation_ms": 1.0, "energy_j": 0.1,
                "receiver_latency_ms": 0.0, "receiver_energy_j": 0.0,
                "calibrated_correctness_probability": 0.8,
            },
        ])
        profiles = pd.DataFrame([
            {
                "mode": row.mode, "variant": row.variant,
                "payload_bytes": len(bytes.fromhex(row.payload_blob)),
                "generation_ms": row.generation_ms, "energy_j": row.energy_j,
                "predicted_task_success": 0.9,
                "predicted_high_confidence_error": 0.0,
                "receiver_latency_ms": 0.0, "receiver_energy_j": 0.0,
            }
            for row in candidates.itertuples(index=False)
        ])
        manifest = pd.DataFrame([{
            "sample_id": "scene-1", "visual_group_id": "group-1",
            "capture_time_us": 0, "trace_time_us": 0,
            "deadline_us": 5_000_000, "mission_profile": "routine",
            "gold_action": "RESPOND", "critical": True, "ood": False,
            "risk": detector_risk, "uncertainty": detector_uncertainty,
            "detector_risk": detector_risk,
            "detector_uncertainty": detector_uncertainty,
            # Deliberately combined and true: ablations must recompute their
            # trigger from the signals they retain instead of reusing this bit.
            "context_triggered": True,
            "detector_latency_ms": 1.0, "detector_energy_j": 0.1,
            "context_model_latency_ms": 2.0, "context_model_energy_j": 0.2,
            "battery_fraction": 0.8, "thermal_c": 55.0,
        }])
        return manifest, candidates, profiles

    def test_vlm_trigger_respects_risk_and_uncertainty_ablations(self) -> None:
        manifest, candidates, profiles = self._trigger_inputs(0.1, 0.9)
        policies = [
            FullPolicy(),
            RiskAwareNoUncertaintyPolicy(),
            NoMissionRiskPolicy(),
        ]
        uncertainty_rows = run_paired_policy_trials(
            manifest, candidates, self.traces, policies, (7,), profiles,
        ).set_index("policy")
        self.assertTrue(bool(uncertainty_rows.loc["proposed_full", "decision_context_available"]))
        self.assertFalse(bool(uncertainty_rows.loc[
            "risk_aware_no_uncertainty", "decision_context_available",
        ]))
        self.assertTrue(bool(uncertainty_rows.loc[
            "ablate_mission_risk", "decision_context_available",
        ]))
        self.assertEqual(
            float(uncertainty_rows.loc["risk_aware_no_uncertainty", "decision_uncertainty"]),
            0.0,
        )

        manifest, candidates, profiles = self._trigger_inputs(0.9, 0.1)
        risk_rows = run_paired_policy_trials(
            manifest, candidates, self.traces, policies, (7,), profiles,
        ).set_index("policy")
        self.assertTrue(bool(risk_rows.loc[
            "risk_aware_no_uncertainty", "decision_context_available",
        ]))
        self.assertFalse(bool(risk_rows.loc[
            "ablate_mission_risk", "decision_context_available",
        ]))
        self.assertEqual(
            float(risk_rows.loc["ablate_mission_risk", "decision_risk"]), 0.0,
        )
        self.assertEqual(
            float(risk_rows.loc["ablate_mission_risk", "decision_mission_priority"]),
            0.0,
        )

    def test_receiver_state_ablation_forces_keyframes_not_ack_deltas(self) -> None:
        candidate_rows = []
        manifest_rows = []
        for index, sample_id in enumerate(("state-1", "state-2")):
            # Force the portable debug format even when cbor2 is installed.
            # The second full-policy state becomes a delta and must remain JSON
            # rather than following the process's preferred encoder.
            payload, wire_format = self._semantic_packet(
                sample_id,
                wire_format="json-debug",
            )
            candidate_rows.append({
                "sample_id": sample_id, "mode": "L2_SEMANTICS",
                "variant": f"{wire_format}:detector", "source_model": "detector_only",
                "payload_blob": payload.hex(), "receiver_action": "RESPOND",
                "generation_ms": 1.0, "energy_j": 0.1,
                "receiver_latency_ms": 0.0, "receiver_energy_j": 0.0,
                "calibrated_correctness_probability": 0.9,
            })
            capture_us = index * 2_000_000
            manifest_rows.append({
                "sample_id": sample_id, "visual_group_id": f"group-{index}",
                "capture_time_us": capture_us, "trace_time_us": capture_us,
                "deadline_us": capture_us + 1_000_000,
                "mission_profile": "routine", "gold_action": "RESPOND",
                "critical": True, "ood": False, "risk": 0.1,
                "uncertainty": 0.1, "detector_risk": 0.1,
                "detector_uncertainty": 0.1, "context_triggered": False,
                "detector_latency_ms": 0.0, "detector_energy_j": 0.0,
                "context_model_latency_ms": 0.0, "context_model_energy_j": 0.0,
                "battery_fraction": 0.8, "thermal_c": 55.0,
            })
        candidates = pd.DataFrame(candidate_rows)
        profiles = pd.DataFrame([{
            "mode": "L2_SEMANTICS", "variant": candidate_rows[0]["variant"],
            "payload_bytes": len(bytes.fromhex(candidate_rows[0]["payload_blob"])),
            "generation_ms": 1.0, "energy_j": 0.1,
            "predicted_task_success": 0.9,
            "predicted_high_confidence_error": 0.0,
            "receiver_latency_ms": 0.0, "receiver_energy_j": 0.0,
        }])
        # Make the process-wide preferred encoder observably non-JSON.  The
        # episode must still preserve the candidate's declared JSON format for
        # its derived delta.
        fake_cbor2 = SimpleNamespace(
            dumps=lambda *_args, **_kwargs: b"\xacnot-json",
        )
        with mock.patch.dict(sys.modules, {"cbor2": fake_cbor2}):
            rows = run_paired_policy_trials(
                pd.DataFrame(manifest_rows), candidates, self.traces,
                [FullPolicy(), NoReceiverStatePolicy()], (7,), profiles,
            ).sort_values(["policy", "capture_us"])
        full = rows[rows["policy"].eq("proposed_full")]
        ablated = rows[rows["policy"].eq("ablate_receiver_state")]
        self.assertEqual(full["state_kind"].tolist(), ["keyframe", "delta"])
        self.assertEqual(ablated["state_kind"].tolist(), ["keyframe", "keyframe"])
        self.assertTrue(full["state_valid"].all())
        self.assertTrue(ablated["state_valid"].all())
        self.assertTrue(ablated["decision_receiver_belief_status"].eq("known").all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
