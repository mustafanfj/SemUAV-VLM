from __future__ import annotations

import json
import dataclasses
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from semuav.experiments import (
    experiment2_representation_frontier,
    select_grounded_semantic_roi,
)
from semuav.perception import (
    DetectorOnlyBackend, InferenceRequest, MISSION_PROMPT, MISSION_PROMPT_PATH,
    calibrated_semantic_packet_cache,
    detector_execution_revision, run_inference_cache_sharded,
)
from semuav.protocol import (
    VISDRONE_COMPOSED_PACKET_AUTHORSHIP,
    VISDRONE_PROPOSAL_DECISION_VERSION,
    validate_semantic_content_packet,
    validate_semantic_packet,
)
from semuav.registry import RunContext


def raw_packet() -> dict:
    return {
        "schema_version": "semuav.v1", "observation_id": "s1",
        "mission": "urban", "capture_time_us": 0,
        "telemetry": {"gps_lat": 24.0, "gps_lon": 54.0, "altitude_m": 70.0, "yaw_deg": 0.0},
        "sufficient_evidence": True, "event": "visible_road_obstruction",
        "objects": [{
            "label": "car", "bbox_normalized_xyxy": [0.1, 0.2, 0.6, 0.8],
            "confidence": 0.9,
        }],
        "relations": [], "risk": 0.8, "uncertainty": 0.2,
        "recommended_action": "RESPOND",
        "evidence": [{
            "evidence_id": "event-0", "linked_claims": ["event"],
            "bbox_normalized_xyxy": [0.1, 0.2, 0.6, 0.8],
        }],
    }


def inference_rows() -> pd.DataFrame:
    return pd.DataFrame([{
        "observation_id": "s1", "parsed_json": json.dumps(raw_packet()),
        "schema_valid": True, "confidence": 0.8,
        "model_revision": "a" * 40, "prompt_hash": "b" * 64,
        "confidence_source": "generated_token_probability",
    }])


class TransportProvenanceTests(unittest.TestCase):
    def test_scientific_composed_visdrone_packet_requires_complete_decision_lineage(self) -> None:
        packet = raw_packet()
        packet["mission"] = "visdrone_urban"
        packet["event"] = "visible_priority_roi_occupancy"
        packet["calibrated_correctness_probability"] = 0.8
        packet["provenance"] = {
            "source_model": "qwen3_vl_2b",
            "model_revision": "a" * 40,
            "prompt_hash": "b" * 64,
            "ontology_version": "semuav.ontology.v1",
            "calibration": {"method": "temperature_scaling", "temperature": 1.0},
            "schema_version": "semuav.v1",
            "semantic_packet_authorship": VISDRONE_COMPOSED_PACKET_AUTHORSHIP,
            "decision_contract_version": VISDRONE_PROPOSAL_DECISION_VERSION,
            "decision_composer_policy_sha256": "c" * 64,
            "decision_context_sha256": "d" * 64,
        }
        self.assertEqual(
            validate_semantic_packet(packet, require_transmission_metadata=True),
            packet,
        )

        incomplete = json.loads(json.dumps(packet))
        incomplete["provenance"].pop("decision_context_sha256")
        with self.assertRaisesRegex(ValueError, "provenance is incomplete"):
            validate_semantic_packet(incomplete, require_transmission_metadata=True)

        mislabeled = json.loads(json.dumps(packet))
        mislabeled["provenance"]["semantic_packet_authorship"] = "model_authored"
        with self.assertRaisesRegex(ValueError, "deterministic host composer"):
            validate_semantic_packet(mislabeled, require_transmission_metadata=True)

    def test_frozen_prompt_is_single_source_and_enumerates_output_shapes(self) -> None:
        self.assertEqual(MISSION_PROMPT_PATH.read_text(encoding="utf-8").strip(), MISSION_PROMPT)
        for token in (
            "urban events=[", "wildfire objects=[", "bbox_normalized_xyxy",
            '"linked_claims"', '"predicate"', "evidence",
        ):
            self.assertIn(token, MISSION_PROMPT)

    def test_calibrated_packet_preserves_declared_normalized_evidence_coordinates(self) -> None:
        cache = calibrated_semantic_packet_cache(
            inference_rows(), 1.0, source_model="model_a",
        )
        packet = json.loads(cache.iloc[0].packet_json)
        self.assertEqual(
            packet["evidence"][0]["bbox_normalized_xyxy"],
            [0.1, 0.2, 0.6, 0.8],
        )
        self.assertEqual(packet["provenance"]["source_model"], "model_a")
        validate_semantic_packet(packet, require_transmission_metadata=True)

    def test_scientific_metadata_rejects_empty_claims_and_bad_integrity_hash(self) -> None:
        cache = calibrated_semantic_packet_cache(
            inference_rows(), 1.0, source_model="model_a",
        )
        packet = json.loads(cache.iloc[0].packet_json)
        packet["evidence"][0]["linked_claims"] = []
        with self.assertRaisesRegex(ValueError, "nonempty list"):
            validate_semantic_packet(packet, require_transmission_metadata=True)
        packet["evidence"][0]["linked_claims"] = ["object:car"]
        packet["evidence"][0]["integrity_sha256"] = "not-a-hash"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate_semantic_packet(packet, require_transmission_metadata=True)
        packet = json.loads(cache.iloc[0].packet_json)
        packet["calibrated_correctness_probability"] = 1.1
        with self.assertRaisesRegex(ValueError, "calibrated_correctness_probability"):
            validate_semantic_packet(packet, require_transmission_metadata=True)

    def test_experiment2_rejects_packet_provenance_source_mismatch(self) -> None:
        semantic_cache = calibrated_semantic_packet_cache(
            inference_rows(), 1.0, source_model="model_a",
        )
        packet = json.loads(semantic_cache.iloc[0].packet_json)
        packet["provenance"]["source_model"] = "model_b"
        semantic_cache.loc[0, "packet_json"] = json.dumps(packet)
        receiver_columns = [
            "sample_id", "variant", "source_model", "receiver_action",
            "generation_ms", "energy_j", "receiver_latency_ms",
            "receiver_energy_j", "hardware_id", "receiver_revision",
            "payload_sha256", "cost_scope",
        ]
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.png"
            Image.new("RGB", (16, 16), "black").save(image_path)
            manifest = pd.DataFrame([{
                "sample_id": "s1", "visual_group_id": "g1",
                "image_path": str(image_path), "gold_action": "RESPOND",
                "clip_paths_json": "[]",
            }])
            with self.assertRaisesRegex(Exception, "provenance source differs"):
                experiment2_representation_frontier(
                    manifest, scientific=True,
                    semantic_packet_cache=semantic_cache,
                    native_receiver_cache=pd.DataFrame(columns=receiver_columns),
                    semantic_source_model="model_a",
                )

    def test_scientific_packets_enforce_frozen_mission_ontology_and_references(self) -> None:
        cache = calibrated_semantic_packet_cache(
            inference_rows(), 1.0, source_model="model_a",
        )
        valid = json.loads(cache.iloc[0].packet_json)
        mutations = [
            ("event", lambda packet: packet.update(event="visible_hazard"), "event ontology"),
            ("object", lambda packet: packet["objects"][0].update(label="drone"), "object ontology"),
            (
                "relation",
                lambda packet: packet.update(relations=[{
                    "subject": "car", "predicate": "near", "object": "pedestrian",
                }]),
                "references absent objects",
            ),
            (
                "ontology version",
                lambda packet: packet["provenance"].update(ontology_version="semuav-mission-v1"),
                "ontology_version",
            ),
        ]
        for name, mutate, message in mutations:
            with self.subTest(name=name):
                packet = json.loads(json.dumps(valid))
                mutate(packet)
                with self.assertRaisesRegex(ValueError, message):
                    validate_semantic_packet(packet, require_transmission_metadata=True)

    def test_packet_rejects_invalid_telemetry_and_fourth_action(self) -> None:
        for field, value, message in (
            ("gps_lat", 91.0, "gps_lat"),
            ("gps_lon", -181.0, "gps_lon"),
            ("altitude_m", -0.1, "altitude_m"),
            ("yaw_deg", 360.0, "yaw_deg"),
        ):
            with self.subTest(field=field):
                packet = raw_packet()
                packet["telemetry"][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_semantic_packet(packet)
        packet = raw_packet()
        packet["sufficient_evidence"] = False
        packet["event"] = "insufficient_evidence"
        packet["recommended_action"] = "INSUFFICIENT_EVIDENCE"
        with self.assertRaisesRegex(ValueError, "Unknown action"):
            validate_semantic_packet(packet)

    def test_detector_only_has_registered_count_based_critical_action_rule(self) -> None:
        request = InferenceRequest(
            observation_id="dense", image_path="unused.png", mission="urban",
            capture_time_us=0,
            telemetry={"gps_lat": 24.0, "gps_lon": 54.0, "altitude_m": 70.0, "yaw_deg": 0.0},
            detector_proposals=tuple(
                {
                    "label": "car", "confidence": 0.9, "track_id": index,
                    "bbox_normalized_xyxy": [0.05 + 0.25 * index, 0.1, 0.2 + 0.25 * index, 0.4],
                }
                for index in range(3)
            ),
        )
        record = DetectorOnlyBackend("a" * 40).infer(request)
        self.assertTrue(record.schema_valid, record.error)
        assert record.parsed is not None
        self.assertTrue(record.parsed["sufficient_evidence"])
        self.assertEqual(record.parsed["event"], "visible_road_obstruction")
        self.assertEqual(record.parsed["recommended_action"], "RESPOND")
        self.assertGreaterEqual(float(record.confidence), 0.85)

        routine = DetectorOnlyBackend("a" * 40).infer(dataclasses.replace(
            request,
            observation_id="routine",
            detector_proposals=({
                "label": "car", "confidence": 0.9, "track_id": 1,
                "bbox_normalized_xyxy": [0.1, 0.1, 0.3, 0.4],
            },),
        ))
        assert routine.parsed is not None
        self.assertEqual(routine.parsed["recommended_action"], "NO_ACTION")
        self.assertGreater(float(routine.confidence), 0.5)
        self.assertEqual(routine.confidence_source, "detector_rule_action_margin")

    def test_detector_only_registered_roi_policy_never_invents_road_obstruction(self) -> None:
        request = InferenceRequest(
            observation_id="registered-roi", image_path="unused.png", mission="urban",
            capture_time_us=0,
            telemetry={
                "gps_lat": 24.0, "gps_lon": 54.0,
                "altitude_m": 70.0, "yaw_deg": 0.0,
            },
            detector_proposals=(
                {
                    "label": "pedestrian", "confidence": 0.92,
                    "bbox_normalized_xyxy": [0.45, 0.40, 0.55, 0.70],
                },
                {
                    "label": "car", "confidence": 0.95,
                    "bbox_normalized_xyxy": [0.02, 0.05, 0.18, 0.25],
                },
                {
                    "label": "car", "confidence": 0.93,
                    "bbox_normalized_xyxy": [0.75, 0.05, 0.90, 0.25],
                },
            ),
        )
        backend = DetectorOnlyBackend(
            "a" * 40,
            urban_policy="registered_roi_occupancy",
            urban_priority_labels=("pedestrian",),
            urban_priority_roi_xyxy=(0.25, 0.25, 0.75, 0.75),
        )
        record = backend.infer(request)
        assert record.parsed is not None
        self.assertTrue(record.schema_valid, record.error)
        self.assertEqual(record.parsed["event"], "none_visible")
        self.assertEqual(record.parsed["recommended_action"], "RESPOND")
        self.assertNotIn(
            "visible_road_obstruction",
            record.raw_text,
        )
        self.assertNotIn("near", {item["predicate"] for item in record.parsed["relations"]})

        outside_roi = backend.infer(dataclasses.replace(
            request,
            observation_id="outside-roi",
            detector_proposals=({
                "label": "pedestrian", "confidence": 0.92,
                "bbox_normalized_xyxy": [0.01, 0.01, 0.10, 0.10],
            },),
        ))
        assert outside_roi.parsed is not None
        self.assertEqual(outside_roi.parsed["event"], "none_visible")
        self.assertEqual(outside_roi.parsed["recommended_action"], "REVIEW")
        self.assertNotEqual(backend.rule_config_hash, "")

    def test_detector_only_registered_roi_policy_requires_frozen_supported_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires frozen"):
            DetectorOnlyBackend(
                "a" * 40,
                urban_policy="registered_roi_occupancy",
            )
        with self.assertRaisesRegex(ValueError, "unsupported labels"):
            DetectorOnlyBackend(
                "a" * 40,
                urban_policy="registered_roi_occupancy",
                urban_priority_labels=("accident",),
            )

    def test_visdrone_registered_policy_uses_profile_and_sensor_space_labels(self) -> None:
        backend = DetectorOnlyBackend(
            "a" * 40,
            urban_policy="registered_roi_occupancy",
            urban_priority_labels=("pedestrian",),
            urban_priority_profiles={
                "vulnerable_road_user": ("pedestrian", "bicycle"),
                "heavy_vehicle": ("truck", "bus"),
            },
            urban_priority_roi_xyxy=(0.25, 0.25, 0.75, 0.85),
        )
        request = InferenceRequest(
            observation_id="visdrone-profile",
            image_path="unused.png",
            mission="visdrone_urban",
            capture_time_us=0,
            telemetry={
                "gps_lat": 0.0, "gps_lon": 0.0,
                "altitude_m": 70.0, "yaw_deg": 0.0,
                "mission_profile": "heavy_vehicle",
            },
            detector_proposals=({
                "label": "pedestrian", "confidence": 0.99,
                "bbox_normalized_xyxy": [0.4, 0.4, 0.5, 0.6],
            },),
        )
        record = backend.infer(request)
        assert record.parsed is not None
        self.assertEqual(record.parsed["event"], "no_registered_priority_visible")
        self.assertEqual(record.parsed["recommended_action"], "NO_ACTION")
        self.assertNotIn("near", record.raw_text)
        self.assertNotIn("obstruction", record.raw_text)

    def test_detector_only_normalizes_pixel_boxes_and_derives_geometry_relations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "frame.png"
            Image.new("RGB", (200, 100), color=(32, 32, 32)).save(image_path)
            request = InferenceRequest(
                observation_id="geometry", image_path=str(image_path), mission="urban",
                capture_time_us=0,
                telemetry={
                    "gps_lat": 24.0, "gps_lon": 54.0,
                    "altitude_m": 70.0, "yaw_deg": 0.0,
                },
                detector_proposals=(
                    {"label": "car", "confidence": 0.9, "class_id": 3, "bbox_xyxy": [20, 10, 80, 60]},
                    {"label": "pedestrian", "confidence": 0.8, "class_id": 0, "bbox_xyxy": [90, 10, 150, 60]},
                ),
            )
            record = DetectorOnlyBackend("a" * 40).infer(request)
            assert record.parsed is not None
            first = record.parsed["objects"][0]
            self.assertEqual(first["bbox_normalized_xyxy"], [0.1, 0.1, 0.4, 0.6])
            self.assertNotIn("bbox_xyxy", first)
            self.assertNotIn("class_id", first)
            self.assertEqual(record.parsed["relations"], [{
                "subject": "car", "predicate": "near", "object": "pedestrian",
            }])
            self.assertTrue(record.parsed["evidence"])

    def test_strict_model_content_rejects_prompt_contract_violations(self) -> None:
        corruptions = []
        value = raw_packet(); value["objects"] = ["car"]
        corruptions.append(value)
        value = raw_packet(); value["objects"][0].pop("bbox_normalized_xyxy")
        corruptions.append(value)
        value = raw_packet(); value["objects"][0].pop("confidence")
        corruptions.append(value)
        value = raw_packet(); value["relations"] = ["near"]
        corruptions.append(value)
        value = raw_packet(); value.pop("evidence")
        corruptions.append(value)
        value = raw_packet(); value["evidence"][0]["bbox_normalized_xyxy"] = [0, 0, 2, 1]
        corruptions.append(value)
        value = raw_packet(); value["evidence"][0].pop("evidence_id")
        corruptions.append(value)
        value = raw_packet(); value["event"] = "none_visible"; value["objects"] = []; value["evidence"][0]["linked_claims"] = ["none"]
        corruptions.append(value)
        for value in corruptions:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_semantic_content_packet(value)

    def test_l3_roi_prefers_prompt_normalized_object_geometry(self) -> None:
        packet = raw_packet()
        packet["evidence"] = []
        packet["objects"][0]["bbox_normalized_xyxy"] = [0.2, 0.1, 0.7, 0.9]
        self.assertEqual(
            select_grounded_semantic_roi(packet, (1920, 1080)),
            [0.2, 0.1, 0.7, 0.9],
        )

    def test_l3_roi_prefers_event_only_grounded_evidence(self) -> None:
        packet = raw_packet()
        packet["objects"] = []
        packet["evidence"][0]["bbox_normalized_xyxy"] = [0.3, 0.2, 0.8, 0.7]
        self.assertEqual(
            select_grounded_semantic_roi(packet, (1920, 1080)),
            [0.3, 0.2, 0.8, 0.7],
        )

    def test_detector_revision_and_vlm_cache_bind_checkpoint_residency(self) -> None:
        revision_a = detector_execution_revision("a" * 64, "botsort.yaml")
        revision_b = detector_execution_revision("b" * 64, "botsort.yaml")
        self.assertNotEqual(revision_a, revision_b)
        self.assertEqual(len(revision_a), 40)
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "frame.png"
            Image.new("RGB", (64, 64), color=(32, 32, 32)).save(image_path)
            request = InferenceRequest(
                "cache-context", str(image_path), "urban", 0,
                {"gps_lat": 24.0, "gps_lon": 54.0, "altitude_m": 70.0, "yaw_deg": 0.0},
                ({
                    "label": "car", "confidence": 0.9,
                    "bbox_normalized_xyxy": [0.1, 0.1, 0.5, 0.5],
                },),
            )
            ctx = RunContext.create(
                Path(__file__).resolve().parents[1], Path(directory) / "cache",
                "dry_run", "residency-key", {"test": True}, resume=True,
            )
            backend = DetectorOnlyBackend(revision_a)
            stage_name = "01_vlm_detector_only_shard_00000"
            run_inference_cache_sharded(
                backend, [request], ctx, scientific=False,
                execution_context={"detector_revision": revision_a, "residency": "footprint-a"},
            )
            first_artifact = Path(ctx.stage_record(stage_name)["artifact"])
            run_inference_cache_sharded(
                backend, [request], ctx, scientific=False,
                execution_context={"detector_revision": revision_b, "residency": "footprint-b"},
            )
            second_artifact = Path(ctx.stage_record(stage_name)["artifact"])
            self.assertNotEqual(first_artifact, second_artifact)
            self.assertTrue(first_artifact.is_file())
            self.assertTrue(second_artifact.is_file())

    def test_model_authored_content_rejects_out_of_ontology_labels(self) -> None:
        packet = raw_packet()
        packet["objects"] = [{"label": "smoke"}]
        with self.assertRaisesRegex(ValueError, "object ontology"):
            validate_semantic_content_packet(packet)


if __name__ == "__main__":
    unittest.main()
