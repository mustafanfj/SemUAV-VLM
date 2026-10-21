from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd
import yaml

from semuav.config import AnnotationConfig
from semuav.data import DataGateError, prepare_analysis_manifest
from semuav.protocol import MISSION_ONTOLOGY, SEMANTIC_ONTOLOGY_VERSION


ROOT = Path(__file__).resolve().parents[1]


class ResolvedAnnotationOntologyTests(unittest.TestCase):
    def test_shipped_yaml_matches_the_runtime_mission_ontology(self) -> None:
        payload = yaml.safe_load(
            (ROOT / "annotations" / "mission_ontology.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(payload["schema_version"], SEMANTIC_ONTOLOGY_VERSION)
        for mission, runtime in MISSION_ONTOLOGY.items():
            self.assertIn(mission, payload)
            for field in ("objects", "relations", "events"):
                self.assertEqual(set(payload[mission][field]), set(runtime[field]))

    @staticmethod
    def manifest() -> pd.DataFrame:
        return pd.DataFrame([{
            "sample_id": "s1", "dataset": "VisDrone", "media_path": "unused.jpg",
            "mission": "urban", "split": "validation", "split_unit": "unit-1",
            "sequence_id": "sequence-1", "timestamp_ms": 0.0, "frame_index": 0,
        }])

    @staticmethod
    def telemetry() -> pd.DataFrame:
        return pd.DataFrame([{
            "sample_id": "s1", "capture_time_us": 0, "gps_lat": 24.0,
            "gps_lon": 54.0, "altitude_m": 70.0, "yaw_deg": 0.0,
            "mission_profile": "standard", "deadline_us": 1_000_000,
            "trace_time_us": 0, "battery_fraction": 0.9, "thermal_c": 50.0,
        }])

    @staticmethod
    def annotations(
        *,
        objects: list[object] | None = None,
        relations: list[object] | None = None,
    ) -> pd.DataFrame:
        objects = objects if objects is not None else [
            {"label": "car", "track_id": "car-1"},
            {"label": "pedestrian", "track_id": "person-1"},
        ]
        relations = relations if relations is not None else [{
            "subject": "car", "predicate": "near", "object": "pedestrian",
        }]
        labels = {
            "gold_action": "REVIEW",
            "critical": "false",
            "cohort": "development",
            "sufficient_evidence": "true",
            "event": "visible_close_proximity",
            "objects": json.dumps(objects, sort_keys=True),
            "relations": json.dumps(relations, sort_keys=True),
            "evidence_roi": "[0.1,0.1,0.8,0.8]",
        }
        rows = []
        for task, label in labels.items():
            for annotator in ("a1", "a2"):
                rows.append({
                    "sample_id": "s1", "task": task, "label": label,
                    "annotator_id": annotator, "blinded": True,
                    "adjudicated_label": "", "adjudicator_id": "",
                    "is_critical": False,
                })
        return pd.DataFrame(rows)

    @staticmethod
    def config() -> AnnotationConfig:
        return AnnotationConfig()

    def prepare(self, annotations: pd.DataFrame) -> pd.DataFrame:
        return prepare_analysis_manifest(
            self.manifest(), annotations, self.config(), self.telemetry(),
        )

    def test_supported_dict_claims_and_relation_references_are_accepted(self) -> None:
        result = self.prepare(self.annotations())
        self.assertEqual(result.iloc[0].objects[0]["label"], "car")
        self.assertEqual(result.iloc[0].relations[0]["predicate"], "near")

    def test_unsupported_gold_object_is_rejected(self) -> None:
        with self.assertRaisesRegex(DataGateError, "unsupported urban object"):
            self.prepare(self.annotations(objects=[{"label": "drone"}]))

    def test_gold_relation_must_reference_resolved_objects(self) -> None:
        relations = [{"subject": "car", "predicate": "near", "object": "truck"}]
        with self.assertRaisesRegex(DataGateError, "references absent objects"):
            self.prepare(self.annotations(relations=relations))

    def test_gold_objects_and_relations_reject_bare_strings(self) -> None:
        with self.assertRaisesRegex(DataGateError, "object entries must be JSON objects"):
            self.prepare(self.annotations(objects=["car"], relations=[]))
        with self.assertRaisesRegex(DataGateError, "relation entries must be JSON objects"):
            self.prepare(self.annotations(relations=["near"]))


if __name__ == "__main__":
    unittest.main()
