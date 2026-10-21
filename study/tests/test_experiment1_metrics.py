from __future__ import annotations

import json
import unittest
from typing import Any

import numpy as np
import pandas as pd

from semuav.experiments import experiment1_perception
from semuav.perception import (
    InferenceRecord,
    ModelSpec,
    VISDRONE_PROPOSAL_DECISION_CONTRACT_VERSION,
)
from semuav.protocol import validate_semantic_packet
from semuav.registry import StudyGateError, hash_object


class ScriptedBackend:
    def __init__(self, packets: dict[str, dict[str, Any]], confidence: dict[str, float | None]):
        self.spec = ModelSpec("scripted", "local/test", "1" * 40)
        self.packets = packets
        self.confidence = confidence

    def infer(self, request: Any, prompt: str, max_new_tokens: int) -> InferenceRecord:
        packet = self.packets[request.observation_id]
        parsed = validate_semantic_packet(packet)
        return InferenceRecord(
            observation_id=request.observation_id,
            model_id=self.spec.model_id,
            model_revision=self.spec.revision,
            raw_text=json.dumps(packet),
            parsed=parsed,
            schema_valid=True,
            error=None,
            input_tokens=10,
            output_tokens=20,
            confidence=self.confidence[request.observation_id],
            confidence_source="scripted_probability",
            latency_ms=10.0,
            peak_memory_mb=100.0,
            energy_j=0.5,
            prompt_hash=hash_object(prompt),
            peak_temperature_c=67.0,
        )


class ScriptedVisDroneVerifierBackend(ScriptedBackend):
    def infer(self, request: Any, prompt: str, max_new_tokens: int) -> InferenceRecord:
        record = super().infer(request, prompt, max_new_tokens)
        record.decision_contract_version = VISDRONE_PROPOSAL_DECISION_CONTRACT_VERSION
        record.decision_core_valid = True
        record.decision_core_json = json.dumps({
            "sufficient_evidence": True,
            "accepted_proposal_ids": [],
            "limited_visibility_proposal_ids": [],
        }, sort_keys=True, separators=(",", ":"))
        record.decision_context_sha256 = "2" * 64
        record.decision_proposal_count = 0
        record.decision_composer_policy_sha256 = "3" * 64
        record.terminated_by_eos = True
        record.reached_token_cap = False
        return record


def packet(
    sample_id: str,
    *,
    capture_time_us: int = 0,
    action: str,
    event: str,
    objects: list[Any],
    relations: list[Any],
    sufficient_evidence: bool,
    evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    canonical_objects = []
    for index, item in enumerate(objects):
        canonical = dict(item) if isinstance(item, dict) else {"label": str(item)}
        canonical.setdefault("bbox_normalized_xyxy", [0.05 * index, 0.05 * index, 0.5, 0.5])
        canonical.setdefault("confidence", 0.9)
        canonical_objects.append(canonical)
    grounded = event not in {"none_visible", "insufficient_evidence"} or bool(canonical_objects or relations)
    canonical_evidence = evidence if evidence is not None else ([{
        "evidence_id": "claim-0", "linked_claims": ["event"],
        "bbox_normalized_xyxy": [0.0, 0.0, 1.0, 1.0],
    }] if grounded else [])
    canonical_evidence = [dict(item) for item in canonical_evidence]
    for index, item in enumerate(canonical_evidence):
        item.setdefault("evidence_id", f"claim-{index}")
    return {
        "schema_version": "semuav.v1",
        "observation_id": sample_id,
        "mission": "urban",
        "capture_time_us": capture_time_us,
        "telemetry": {
            "gps_lat": 24.0, "gps_lon": 54.0,
            "altitude_m": 70.0, "yaw_deg": 0.0,
        },
        "sufficient_evidence": sufficient_evidence,
        "event": event,
        "objects": canonical_objects,
        "relations": relations,
        "risk": 0.5,
        "uncertainty": 0.5,
        "recommended_action": action,
        "evidence": canonical_evidence,
    }


class Experiment1MetricTests(unittest.TestCase):
    def test_visdrone_metrics_are_labeled_as_composed_pipeline_not_vlm_authorship(self) -> None:
        manifest = pd.DataFrame([{
            "sample_id": "s1", "visual_group_id": "g1", "image_path": "unused.png",
            "mission": "visdrone_urban", "capture_time_us": 0,
            "telemetry_json": json.dumps({
                "gps_lat": 24.0, "gps_lon": 54.0, "altitude_m": 70.0,
                "yaw_deg": 0.0,
            }),
            "gold_action": "NO_ACTION", "critical": False,
            "split": "validation", "event": "no_registered_priority_visible",
            "objects": [], "relations": [], "sufficient_evidence": True,
        }])
        composed = packet(
            "s1", action="NO_ACTION", event="no_registered_priority_visible",
            objects=[], relations=[], sufficient_evidence=True, evidence=[],
        )
        composed["mission"] = "visdrone_urban"
        detail, summary = experiment1_perception(
            manifest,
            [ScriptedVisDroneVerifierBackend({"s1": composed}, {"s1": 0.9})],
            scientific=False,
        )
        for column in (
            "action_metric_scope", "event_metric_scope",
            "object_metric_scope", "relation_metric_scope",
        ):
            self.assertEqual(
                detail.iloc[0][column], "composed_detector_verifier_pipeline"
            )
            self.assertEqual(
                summary.iloc[0][column], "composed_detector_verifier_pipeline"
            )
        self.assertTrue(bool(detail.iloc[0].detector_proposal_conditioned))
        self.assertFalse(bool(
            detail.iloc[0].free_form_vlm_packet_authorship_claim_eligible
        ))
        self.assertIn(
            "not_free_form_vlm_generation", detail.iloc[0].semantic_claim_scope
        )

    def test_scientific_visdrone_rejects_unbound_legacy_full_packet_backend(self) -> None:
        manifest = pd.DataFrame([{
            "sample_id": "s1", "visual_group_id": "g1", "image_path": "unused.png",
            "mission": "visdrone_urban", "capture_time_us": 0,
            "telemetry_json": json.dumps({
                "gps_lat": 24.0, "gps_lon": 54.0, "altitude_m": 70.0,
                "yaw_deg": 0.0,
            }),
            "gold_action": "NO_ACTION", "critical": False,
            "split": "validation", "event": "no_registered_priority_visible",
            "objects": [], "relations": [], "sufficient_evidence": True,
        }])
        packet_value = packet(
            "s1", action="NO_ACTION", event="no_registered_priority_visible",
            objects=[], relations=[], sufficient_evidence=True, evidence=[],
        )
        packet_value["mission"] = "visdrone_urban"
        detector_cache = pd.DataFrame([{
            "sample_id": "s1", "proposals_json": "[]",
        }])
        with self.assertRaisesRegex(StudyGateError, "proposal-decision contract"):
            experiment1_perception(
                manifest,
                [ScriptedBackend({"s1": packet_value}, {"s1": 0.9})],
                scientific=True,
                detector_cache=detector_cache,
            )

    def test_registered_mission_telemetry_cannot_be_omitted_or_changed(self) -> None:
        telemetry = {
            "gps_lat": 24.0,
            "gps_lon": 54.0,
            "altitude_m": 70.0,
            "yaw_deg": 0.0,
            "mission_profile": "heavy_vehicle",
            "registered_roi_xyxy": [0.25, 0.25, 0.75, 0.85],
        }
        manifest = pd.DataFrame([{
            "sample_id": "s1", "visual_group_id": "g1", "image_path": "unused.png",
            "mission": "urban", "capture_time_us": 0,
            "telemetry_json": json.dumps(telemetry), "gold_action": "NO_ACTION",
            "critical": False, "split": "validation", "event": "none_visible",
            "objects": [], "relations": [], "sufficient_evidence": True,
        }])
        changed = packet(
            "s1", action="NO_ACTION", event="none_visible", objects=[], relations=[],
            sufficient_evidence=True,
        )
        changed["telemetry"].update({
            "mission_profile": "general_mobility",
            "registered_roi_xyxy": [0.25, 0.25, 0.75, 0.85],
        })
        detail, _ = experiment1_perception(
            manifest, [ScriptedBackend({"s1": changed}, {"s1": 0.9})], scientific=False,
        )
        self.assertFalse(bool(detail.iloc[0].authoritative_consistent))
        self.assertTrue(bool(detail.iloc[0].output_invalid))

    def test_authoritative_request_fields_cannot_be_changed_by_model(self) -> None:
        telemetry = {"gps_lat": 24.0, "gps_lon": 54.0, "altitude_m": 70.0, "yaw_deg": 0.0}
        manifest = pd.DataFrame([{
            "sample_id": "s1", "visual_group_id": "g1", "image_path": "unused.png",
            "mission": "urban", "capture_time_us": 7,
            "telemetry_json": json.dumps(telemetry), "gold_action": "NO_ACTION",
            "critical": False, "split": "validation", "event": "none_visible",
            "objects": [], "relations": [], "sufficient_evidence": True,
        }])
        changed = packet(
            "s1", action="NO_ACTION", event="none_visible", objects=[], relations=[],
            sufficient_evidence=True,
        )
        changed["mission"] = "wildfire"
        changed["capture_time_us"] = 8
        detail, summary = experiment1_perception(
            manifest, [ScriptedBackend({"s1": changed}, {"s1": 0.9})], scientific=False,
        )
        row = detail.iloc[0]
        self.assertTrue(bool(row.semantic_valid))
        self.assertFalse(bool(row.authoritative_consistent))
        self.assertTrue(bool(row.output_invalid))
        self.assertFalse(bool(row.action_correct))
        self.assertFalse(bool(row.event_correct))
        self.assertFalse(bool(row.objects_exact))
        self.assertFalse(bool(row.relations_exact))
        self.assertFalse(bool(row.evidence_state_correct))
        self.assertEqual(float(summary.iloc[0].authoritative_consistency_rate), 0.0)

    def test_group_equal_semantic_localization_selective_and_confidence_metrics(self) -> None:
        manifest = pd.DataFrame([
            {
                "sample_id": "s1", "visual_group_id": "g1", "image_path": "unused-1.png",
                "mission": "urban", "capture_time_us": 0, "telemetry_json": json.dumps({
                    "gps_lat": 24.0, "gps_lon": 54.0, "altitude_m": 70.0, "yaw_deg": 0.0,
                }),
                "gold_action": "RESPOND", "critical": True, "split": "test",
                "event": "visible_road_obstruction",
                "objects": [{"label": "car"}, {"label": "pedestrian"}],
                "relations": [{"subject": "car", "predicate": "near", "object": "pedestrian"}],
                "sufficient_evidence": True, "evidence_roi": [0.0, 0.0, 0.5, 0.5],
            },
            {
                "sample_id": "s2", "visual_group_id": "g1", "image_path": "unused-2.png",
                "mission": "urban", "capture_time_us": 1, "telemetry_json": json.dumps({
                    "gps_lat": 24.0, "gps_lon": 54.0, "altitude_m": 70.0, "yaw_deg": 0.0,
                }),
                "gold_action": "REVIEW", "critical": False, "split": "test",
                "event": "none_visible", "objects": [], "relations": [],
                "sufficient_evidence": False, "evidence_roi": [],
            },
            {
                "sample_id": "s3", "visual_group_id": "g2", "image_path": "unused-3.png",
                "mission": "urban", "capture_time_us": 2, "telemetry_json": json.dumps({
                    "gps_lat": 24.0, "gps_lon": 54.0, "altitude_m": 70.0, "yaw_deg": 0.0,
                }),
                "gold_action": "RESPOND", "critical": True, "split": "test",
                "event": "visible_road_obstruction",
                "objects": [{"label": "smoke"}],
                "relations": [{"subject": "smoke", "predicate": "over", "object": "road"}],
                "sufficient_evidence": True, "evidence_roi": [0.5, 0.5, 1.0, 1.0],
            },
        ])
        packets = {
            "s1": packet(
                "s1", action="RESPOND", event="visible_road_obstruction",
                objects=[
                    {"label": "car", "bbox_normalized_xyxy": [0.0, 0.0, 0.5, 0.5], "confidence": 0.9},
                    {"label": "pedestrian"},
                    {"label": "bicycle", "bbox_normalized_xyxy": [0.5, 0.5, 0.8, 0.8]},
                ],
                relations=[{"subject": "car", "predicate": "near", "object": "pedestrian"}],
                sufficient_evidence=True,
                evidence=[{"linked_claims": ["event"], "bbox_normalized_xyxy": [0.0, 0.0, 0.5, 0.5]}],
            ),
            "s2": packet(
                "s2", capture_time_us=1, action="REVIEW", event="insufficient_evidence", objects=[], relations=[],
                sufficient_evidence=False,
            ),
            "s3": packet(
                "s3", capture_time_us=2, action="REVIEW", event="insufficient_evidence", objects=[], relations=[],
                sufficient_evidence=False,
            ),
        }
        detail, summary = experiment1_perception(
            manifest,
            [ScriptedBackend(packets, {"s1": 0.9, "s2": 0.8, "s3": 0.7})],
            scientific=False,
        )
        result = summary.iloc[0]
        s1_detail = detail.loc[detail["observation_id"].eq("s1")].iloc[0]
        self.assertFalse(bool(s1_detail.objects_exact))
        self.assertTrue(bool(s1_detail.relations_exact))

        # g1 has two frames and g2 one; each visual group contributes equal total weight.
        self.assertAlmostEqual(float(result.action_accuracy), 0.5)
        self.assertEqual(int(result.independent_visual_groups), 2)
        self.assertAlmostEqual(float(result.object_precision), 2.0 / 3.0)
        self.assertAlmostEqual(float(result.object_recall), 0.5)
        self.assertAlmostEqual(float(result.object_f1), 4.0 / 7.0)
        self.assertAlmostEqual(float(result.object_count_mae), 0.75)
        self.assertAlmostEqual(float(result.relation_precision), 1.0)
        self.assertAlmostEqual(float(result.relation_recall), 1.0 / 3.0)
        self.assertAlmostEqual(float(result.grounded_hallucination_rate), 0.25)
        self.assertAlmostEqual(float(result.grounded_omission_rate), 0.5)
        self.assertAlmostEqual(float(result.abstention_rate), 0.75)
        self.assertAlmostEqual(float(result.selective_coverage), 0.25)
        self.assertAlmostEqual(float(result.selective_action_accuracy), 1.0)
        self.assertEqual(int(result.evidence_roi_units), 2)
        self.assertAlmostEqual(float(result.evidence_box_coverage), 1.0 / 3.0)
        self.assertAlmostEqual(float(result.evidence_box_median_iou), 0.0)
        self.assertAlmostEqual(float(result.evidence_box_recall_at_0_7), 1.0 / 3.0)
        self.assertAlmostEqual(float(result.raw_confidence_brier), 0.2575)
        self.assertAlmostEqual(float(result.raw_confidence_auprc), 1.0)
        self.assertTrue(np.isfinite(float(result.raw_confidence_ece_15_equal_mass)))
        self.assertEqual(int(result.temperature_units), 3)
        self.assertAlmostEqual(float(result.temperature_coverage), 1.0)
        self.assertAlmostEqual(float(result.peak_temperature_c), 67.0)

        s1 = detail.set_index("observation_id").loc["s1"]
        self.assertEqual(float(s1.object_tp), 2.0)
        self.assertEqual(float(s1.object_fp), 1.0)
        self.assertEqual(float(s1.object_fn), 0.0)
        self.assertEqual(float(s1.evidence_box_best_iou), 1.0)
        self.assertTrue(bool(s1.grounded_hallucination))
        self.assertFalse(bool(s1.grounded_omission))

    def test_semantic_exact_ignores_geometry_confidence_and_track_metadata(self) -> None:
        manifest = pd.DataFrame([{
            "sample_id": "s1", "visual_group_id": "g1", "image_path": "unused.png",
            "mission": "urban", "capture_time_us": 0,
            "telemetry_json": json.dumps({
                "gps_lat": 24.0, "gps_lon": 54.0,
                "altitude_m": 70.0, "yaw_deg": 0.0,
            }),
            "gold_action": "NO_ACTION", "critical": False, "split": "validation",
            "event": "none_visible", "objects": [{"label": "car"}],
            "relations": [], "sufficient_evidence": True,
        }])
        predicted = packet(
            "s1", action="NO_ACTION", event="none_visible",
            objects=[{
                "label": "car", "bbox_normalized_xyxy": [0.1, 0.2, 0.7, 0.8],
                "confidence": 0.37, "track_id": 99,
            }],
            relations=[], sufficient_evidence=True,
        )
        detail, _ = experiment1_perception(
            manifest, [ScriptedBackend({"s1": predicted}, {"s1": 0.8})],
            scientific=False,
        )
        self.assertTrue(bool(detail.iloc[0].objects_exact))

    def test_absent_optional_ground_truth_stays_unavailable(self) -> None:
        manifest = pd.DataFrame([{
            "sample_id": "s1", "visual_group_id": "g1", "image_path": "unused.png",
            "mission": "urban", "capture_time_us": 0,
            "telemetry_json": json.dumps({
                "gps_lat": 24.0, "gps_lon": 54.0, "altitude_m": 70.0, "yaw_deg": 0.0,
            }),
            "gold_action": "NO_ACTION", "split": "validation",
        }])
        scripted = packet(
            "s1", action="NO_ACTION", event="none_visible", objects=[], relations=[],
            sufficient_evidence=True,
        )
        detail, summary = experiment1_perception(
            manifest, [ScriptedBackend({"s1": scripted}, {"s1": None})], scientific=False,
        )
        result = summary.iloc[0]
        self.assertTrue(np.isnan(float(result.object_precision)))
        self.assertTrue(np.isnan(float(result.relation_recall)))
        self.assertTrue(np.isnan(float(result.grounded_hallucination_rate)))
        self.assertTrue(np.isnan(float(result.evidence_box_median_iou)))
        self.assertTrue(np.isnan(float(result.raw_confidence_brier)))
        self.assertEqual(int(result.evidence_roi_units), 0)
        self.assertFalse(bool(detail.iloc[0].object_metrics_available))
        self.assertFalse(bool(detail.iloc[0].evidence_box_evaluable))


if __name__ == "__main__":
    unittest.main()
