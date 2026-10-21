from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from semuav.visdrone_paper import (
    ACTION_NO_ACTION,
    ACTION_RESPOND,
    ACTION_REVIEW,
    EVENT_IMAGE_PLANE_OVERLAP,
    EVENT_NO_PRIORITY_VISIBLE,
    EVENT_PRIORITY_EVIDENCE_LIMITED,
    EVENT_PRIORITY_OUTSIDE_ROI,
    EVENT_PRIORITY_ROI_OCCUPANCY,
    MISSION_VISDRONE_URBAN,
    RELATION_IMAGE_PLANE_OVERLAP,
    RELATION_INSIDE_ROI,
    NormalizedROI,
    RegisteredBenchmarkPolicy,
    UntouchedTestAccessError,
    VisDroneFormatError,
    build_development_manifest,
    build_source_manifest,
    build_source_truth_table,
    build_untouched_test_manifest,
    derive_source_truth,
    load_objects_for_manifest_row,
    load_untouched_test_permit,
    parse_det_annotations,
    parse_det_row,
    parse_mot_annotations,
    parse_mot_row,
    publisher_split_mapping,
)


def _write(path: Path, payload: str | bytes = b"fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_bytes(payload)
    return path


def _policy(
    priority_categories: tuple[str, ...] = ("car",),
    **kwargs: object,
) -> RegisteredBenchmarkPolicy:
    return RegisteredBenchmarkPolicy.from_category_names(
        rule_id="visdrone-source-rule-v1",
        roi=NormalizedROI("registered-center", 0.25, 0.25, 0.75, 0.75),
        mission_profile="general_mobility",
        priority_categories=priority_categories,
        **kwargs,
    )


class OfficialAnnotationParserTests(unittest.TestCase):
    def test_det_row_parses_trailing_comma_and_publisher_fields(self) -> None:
        obj = parse_det_row(
            "10,20,30,40,1,4,1,2,",
            component="det_train",
            sample_id="frame-001",
            frame_index=17,
            source_line=3,
        )
        self.assertEqual(obj.category, "car")
        self.assertEqual(obj.bbox_xyxy, (10.0, 20.0, 40.0, 60.0))
        self.assertEqual(obj.frame_index, 17)
        self.assertEqual(obj.object_id, 3)
        self.assertTrue(obj.object_uid.startswith("det_train:frame-001:"))

    def test_mot_row_parses_frame_track_and_component_identity(self) -> None:
        obj = parse_mot_row(
            "7,42,10,20,30,40,1,9,0,1,",
            component="mot_val",
            sequence_id="uav000001",
            source_line=9,
        )
        self.assertEqual(obj.frame_index, 7)
        self.assertEqual(obj.object_id, 42)
        self.assertEqual(obj.category, "bus")
        self.assertEqual(obj.object_uid, "mot_val:uav000001:0000007:0000042")

    def test_component_identity_prevents_det_mot_and_split_collisions(self) -> None:
        det_train = parse_det_row(
            "1,2,3,4,1,1,0,0",
            component="det_train",
            sample_id="same",
        )
        det_val = parse_det_row(
            "1,2,3,4,1,1,0,0",
            component="det_val",
            sample_id="same",
        )
        mot_train = parse_mot_row(
            "1,1,1,2,3,4,1,1,0,0",
            component="mot_train",
            sequence_id="same",
        )
        self.assertEqual(len({det_train.object_uid, det_val.object_uid, mot_train.object_uid}), 3)

    def test_complete_payloads_skip_blank_lines_and_filter_mot_frame(self) -> None:
        det = parse_det_annotations(
            "1,2,3,4,1,1,0,0\n\n5,6,7,8,1,4,0,0,\n",
            component="det_val",
            sample_id="d1",
        )
        mot = parse_mot_annotations(
            "1,5,1,2,3,4,1,4,0,0\n2,5,5,6,7,8,1,4,0,0\n",
            component="mot_train",
            sequence_id="m1",
            frame_index=2,
        )
        self.assertEqual(len(det), 2)
        self.assertEqual([obj.frame_index for obj in mot], [2])

    def test_parser_rejects_wrong_schema_and_invalid_domains(self) -> None:
        invalid = (
            "1,2,3,4,1,4,0",
            "1,2,0,4,1,4,0,0",
            "1,2,3,4,2,4,0,0",
            "1,2,3,4,1,12,0,0",
            "1,2,3,4,1,4,3,0",
        )
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(VisDroneFormatError):
                parse_det_row(row, sample_id="bad")
        with self.assertRaises(VisDroneFormatError):
            parse_mot_row(
                "0,1,1,2,3,4,1,4,0,0",
                sequence_id="bad",
            )


class RegisteredSensorTruthTests(unittest.TestCase):
    @staticmethod
    def det(row: str, source_line: int = 1):
        return parse_det_row(
            row,
            component="det_train",
            sample_id="sample",
            frame_index=1,
            source_line=source_line,
        )

    def test_priority_roi_occupancy_maps_to_respond_notification(self) -> None:
        truth = derive_source_truth(
            sample_id="s1",
            objects=[self.det("40,40,20,20,1,4,0,0")],
            image_width=100,
            image_height=100,
            policy=_policy(),
        )
        self.assertEqual(truth.mission, MISSION_VISDRONE_URBAN)
        self.assertEqual(truth.mission_profile, "general_mobility")
        self.assertEqual(truth.gold_action, ACTION_RESPOND)
        self.assertTrue(truth.priority_endpoint_positive)
        self.assertIn(EVENT_PRIORITY_ROI_OCCUPANCY, truth.events)
        self.assertIn(RELATION_INSIDE_ROI, {relation.kind for relation in truth.relations})

    def test_priority_visible_outside_roi_maps_to_review(self) -> None:
        truth = derive_source_truth(
            sample_id="s2",
            objects=[self.det("0,0,10,10,1,4,0,0")],
            image_width=100,
            image_height=100,
            policy=_policy(),
        )
        self.assertEqual(truth.gold_action, ACTION_REVIEW)
        self.assertFalse(truth.priority_endpoint_positive)
        self.assertEqual(truth.events, (EVENT_PRIORITY_OUTSIDE_ROI,))

    def test_no_registered_priority_maps_to_no_action(self) -> None:
        truth = derive_source_truth(
            sample_id="s3",
            objects=[self.det("40,40,20,20,1,1,0,0")],
            image_width=100,
            image_height=100,
            policy=_policy(("car",)),
        )
        self.assertEqual(truth.gold_action, ACTION_NO_ACTION)
        self.assertEqual(truth.events, (EVENT_NO_PRIORITY_VISIBLE,))

    def test_low_visibility_priority_target_maps_to_review(self) -> None:
        truth = derive_source_truth(
            sample_id="s4",
            objects=[self.det("40,40,20,20,1,4,0,2")],
            image_width=100,
            image_height=100,
            policy=_policy(max_occlusion=1),
        )
        self.assertEqual(truth.gold_action, ACTION_REVIEW)
        self.assertFalse(truth.priority_endpoint_positive)
        self.assertIn(EVENT_PRIORITY_EVIDENCE_LIMITED, truth.events)
        self.assertIn(RELATION_INSIDE_ROI, {relation.kind for relation in truth.relations})

    def test_image_plane_overlap_is_explicit_and_numeric(self) -> None:
        truth = derive_source_truth(
            sample_id="s5",
            objects=[
                self.det("30,30,40,40,1,4,0,0", 1),
                self.det("40,40,40,40,1,1,0,0", 2),
            ],
            image_width=100,
            image_height=100,
            policy=_policy(min_image_plane_iou=0.01),
        )
        overlaps = [r for r in truth.relations if r.kind == RELATION_IMAGE_PLANE_OVERLAP]
        self.assertEqual(len(overlaps), 1)
        self.assertGreater(float(overlaps[0].value), 0.0)
        self.assertIn(EVENT_IMAGE_PLANE_OVERLAP, truth.events)

    def test_outputs_are_limited_to_registered_sensor_space_vocabulary(self) -> None:
        truth = derive_source_truth(
            sample_id="s6",
            objects=[self.det("35,35,30,30,1,4,0,0")],
            image_width=100,
            image_height=100,
            policy=_policy(),
        )
        payload = json.dumps(truth.as_dict(), sort_keys=True).lower()
        disallowed = ("accident", "emergency", "road_obstruction", "physical_near")
        self.assertFalse(any(term in payload for term in disallowed))

    def test_policy_requires_one_exact_registered_profile(self) -> None:
        with self.assertRaises(ValueError):
            RegisteredBenchmarkPolicy.from_category_names(
                rule_id="bad-profile",
                roi=NormalizedROI("roi", 0.1, 0.1, 0.9, 0.9),
                mission_profile="unregistered",
                priority_categories=("car",),
            )


class PublisherManifestAndTestGuardTests(unittest.TestCase):
    @staticmethod
    def build_tree(root: Path) -> None:
        for component in ("train", "val"):
            det = root / f"VisDrone2019-DET-{component}"
            _write(det / "images" / "000001.jpg")
            _write(det / "annotations" / "000001.txt", "10,10,20,20,1,4,0,0\n")

            mot = root / f"VisDrone2019-MOT-{component}"
            _write(mot / "sequences" / "uav000001" / "0000001.jpg")
            _write(
                mot / "annotations" / "uav000001.txt",
                "1,7,10,10,20,20,1,4,0,0\n2,7,11,11,20,20,1,4,0,0\n",
            )

        for task in ("DET", "MOT"):
            test = root / f"VisDrone2019-{task}-test-dev"
            if task == "DET":
                _write(test / "images" / "000002.jpg")
                _write(test / "annotations" / "000002.txt", "10,10,20,20,1,4,0,0\n")
            else:
                _write(test / "sequences" / "uav000002" / "0000001.jpg")
                _write(test / "annotations" / "uav000002.txt", "1,8,10,10,20,20,1,4,0,0\n")

    @staticmethod
    def ledger(root: Path, *, status: str = "OPEN") -> Path:
        return _write(
            root / "test_access_ledger.json",
            json.dumps(
                {
                    "status": status,
                    "opened_utc": "2026-08-31T12:00:00Z",
                    "protocol_hash": "a" * 64,
                    "config_hash": "b" * 64,
                    "code_hash": "c" * 64,
                }
            ),
        )

    def test_publisher_partition_mapping_is_exact(self) -> None:
        mapping = publisher_split_mapping().set_index("component")
        self.assertEqual(mapping.loc["det_train", "study_partition"], "development_train")
        self.assertEqual(mapping.loc["mot_val", "study_partition"], "development_validation")
        self.assertEqual(mapping.loc["det_test_dev", "study_partition"], "untouched_test")
        self.assertEqual(mapping.loc["mot_test_dev", "publisher_partition"], "test-dev")

    def test_development_manifest_distinguishes_all_components_and_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_tree(root)
            manifest = build_development_manifest(root)
            self.assertEqual(len(manifest), 4)
            self.assertEqual(
                set(manifest["component"]),
                {"det_train", "det_val", "mot_train", "mot_val"},
            )
            self.assertTrue(manifest["sample_id"].is_unique)
            self.assertTrue(manifest["split_group"].str.contains(r"det_|mot_", regex=True).all())
            self.assertTrue(manifest["mission"].eq(MISSION_VISDRONE_URBAN).all())
            self.assertFalse(manifest["media_path"].str.contains("test-dev").any())

    def test_det_filename_sequence_prefix_defines_independence_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_tree(root)
            det = root / "VisDrone2019-DET-train"
            for stem in ("000123_00001_d_0000001", "000123_00002_d_0000002"):
                _write(det / "images" / f"{stem}.jpg")
                _write(det / "annotations" / f"{stem}.txt", "10,10,20,20,1,4,0,0\n")
            manifest = build_source_manifest(root, components=("det_train",))
            related = manifest[manifest["source_uid"].str.contains("000123_")]
            self.assertEqual(len(related), 2)
            self.assertEqual(related["split_group"].nunique(), 1)

    def test_test_guard_runs_before_root_or_media_inspection(self) -> None:
        missing_root = Path("/definitely/not/a/visdrone/root")
        with self.assertRaises(UntouchedTestAccessError):
            build_source_manifest(missing_root, components=("det_test_dev",))

    def test_test_manifest_requires_verified_open_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_tree(root)
            with self.assertRaises(UntouchedTestAccessError):
                load_untouched_test_permit(self.ledger(root, status="COMPLETE"))
            permit = load_untouched_test_permit(
                self.ledger(root),
                expected_protocol_hash="a" * 64,
                expected_config_hash="b" * 64,
                expected_code_hash="c" * 64,
            )
            manifest = build_untouched_test_manifest(root, test_permit=permit)
            self.assertEqual(len(manifest), 2)
            self.assertTrue(manifest["study_partition"].eq("untouched_test").all())
            self.assertTrue(manifest["test_ledger_sha256"].eq(permit.ledger_sha256).all())

    def test_annotation_reads_repeat_test_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_tree(root)
            permit = load_untouched_test_permit(self.ledger(root))
            row = build_untouched_test_manifest(root, test_permit=permit).iloc[0]
            with self.assertRaises(UntouchedTestAccessError):
                load_objects_for_manifest_row(row)
            objects = load_objects_for_manifest_row(row, test_permit=permit)
            self.assertEqual(len(objects), 1)

            altered = row.copy()
            altered["test_ledger_sha256"] = "0" * 64
            with self.assertRaises(UntouchedTestAccessError):
                load_objects_for_manifest_row(altered, test_permit=permit)

            payload = json.loads(permit.ledger_path.read_text(encoding="utf-8"))
            payload["status"] = "COMPLETE"
            permit.ledger_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(UntouchedTestAccessError):
                load_objects_for_manifest_row(row, test_permit=permit)

    def test_truth_table_uses_registered_dimensions_and_source_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_tree(root)
            manifest = build_source_manifest(root, components=("det_train", "mot_train"))
            sizes = {sample_id: (100, 100) for sample_id in manifest["sample_id"]}
            truth = build_source_truth_table(manifest, image_sizes=sizes, policy=_policy())
            self.assertEqual(len(truth), 2)
            self.assertTrue(truth["mission"].eq(MISSION_VISDRONE_URBAN).all())
            self.assertTrue(truth["gold_action"].eq(ACTION_REVIEW).all())
            self.assertTrue(truth["truth_scope"].str.startswith("publisher_2d_boxes").all())


if __name__ == "__main__":
    unittest.main()
