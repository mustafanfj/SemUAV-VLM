from __future__ import annotations

import contextlib
import io
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from PIL import Image

from semuav.registry import StudyGateError, hash_file
from semuav.visdrone_paper import (
    ACTION_NO_ACTION,
    ACTION_RESPOND,
    MISSION_VISDRONE_URBAN,
    UntouchedTestAccessError,
    VisDroneFormatError,
    build_source_manifest,
    load_untouched_test_permit,
)
from semuav.visdrone_workflow import (
    ANNOTATION_QUALITY_POLICY_VERSION,
    MINIMUM_TEST_NATURAL_GROUPS,
    PROHIBITED_CLAIMS,
    SOURCE_PROVENANCE_SCOPE,
    TEST_TRACE_FAMILIES,
    VALIDATION_TRACE_FAMILIES,
    assign_publisher_splits,
    audit_publisher_splits,
    build_analysis_manifest,
    build_readiness_claim_ledger,
    export_paper_tables,
    generate_registered_trace_manifest,
    load_prepared_visdrone_bundle,
    prepare_visdrone_bundle,
    read_analysis_manifest,
    registered_analysis_sampling_limits,
    registered_mission_policies,
    select_registered_analysis_sample,
    select_registered_analysis_sample_from_config,
    verify_artifact_hash_manifest,
    verify_paper_completion_receipt,
    verify_visdrone_media_partition_audit,
    verify_visdrone_source_provenance,
    write_artifact_hash_manifest,
    write_registered_analysis_sample,
    write_visdrone_media_partition_audit,
)


def benchmark() -> dict[str, object]:
    return {
        "action_rule_version": "visdrone_source_rule_v1",
        "registered_roi_xyxy": [0.25, 0.25, 0.75, 0.85],
        "visibility_rule": {
            "visible_max_occlusion": 1,
            "visible_max_truncation": 1,
        },
        "mission_profiles": {
            "vulnerable_road_user": {
                "priority_categories": [
                    "pedestrian", "people", "bicycle", "motor", "tricycle",
                    "awning_tricycle",
                ],
                "deadline_ms": 700,
            },
            "heavy_vehicle": {
                "priority_categories": ["truck", "bus"],
                "deadline_ms": 1200,
            },
            "general_mobility": {
                "priority_categories": [
                    "pedestrian", "people", "bicycle", "car", "van", "truck",
                    "tricycle", "awning_tricycle", "bus", "motor",
                ],
                "deadline_ms": 2000,
            },
        },
    }


def image(path: Path, size: tuple[int, int] = (100, 100)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (30, 60, 90)).save(path)
    return path


def text(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def build_development_tree(root: Path) -> None:
    for partition in ("train", "val"):
        det = root / f"VisDrone2019-DET-{partition}"
        image(det / "images" / f"000001_{partition}_000001.jpg")
        text(
            det / "annotations" / f"000001_{partition}_000001.txt",
            "40,40,20,20,1,4,0,0\n",
        )
        mot = root / f"VisDrone2019-MOT-{partition}"
        image(mot / "sequences" / "uav000001" / "0000001.jpg")
        text(
            mot / "annotations" / "uav000001.txt",
            "1,7,40,40,20,20,1,4,0,0\n",
        )


def build_test_tree(root: Path) -> None:
    det = root / "VisDrone2019-DET-test-dev"
    image(det / "images" / "000002.jpg")
    text(det / "annotations" / "000002.txt", "40,40,20,20,1,4,0,0\n")
    mot = root / "VisDrone2019-MOT-test-dev"
    image(mot / "sequences" / "uav000002" / "0000001.jpg")
    text(mot / "annotations" / "uav000002.txt", "1,8,40,40,20,20,1,4,0,0\n")


def make_fixture_media_unique(root: Path) -> None:
    paths = (
        root / "VisDrone2019-DET-train" / "images" / "000001_train_000001.jpg",
        root / "VisDrone2019-DET-val" / "images" / "000001_val_000001.jpg",
        root / "VisDrone2019-MOT-train" / "sequences" / "uav000001" / "0000001.jpg",
        root / "VisDrone2019-MOT-val" / "sequences" / "uav000001" / "0000001.jpg",
        root / "VisDrone2019-DET-test-dev" / "images" / "000002.jpg",
        root / "VisDrone2019-MOT-test-dev" / "sequences" / "uav000002" / "0000001.jpg",
    )
    for index, path in enumerate(paths, start=1):
        image(path, (100 + index, 100))


def write_receipt(root: Path) -> Path:
    official = {
        "det_train": ("VisDrone2019-DET-train.zip", "1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn"),
        "det_val": ("VisDrone2019-DET-val.zip", "1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59"),
        "det_test_dev": ("VisDrone2019-DET-test-dev.zip", "1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V"),
        "mot_train": ("VisDrone2019-MOT-train.zip", "1-qX2d-P1Xr64ke6nTdlm33om1VxCUTSh"),
        "mot_val": ("VisDrone2019-MOT-val.zip", "1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu"),
        "mot_test_dev": ("VisDrone2019-MOT-test-dev.zip", "14z8Acxopj1d86-qhsF1NwS4Bv3KYa4Wu"),
    }
    artifacts = [
        {
            "component": name,
            "archive": str(root.parent / archive),
            "google_drive_id": drive_id,
            "sha256": format(index + 1, "064x"),
            "bytes": index + 1,
        }
        for index, (name, (archive, drive_id)) in enumerate(official.items())
    ]
    inventory_path = root / "SOURCE_FILES_SHA256.jsonl"
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"SOURCE_RECEIPT.json", inventory_path.name}:
            continue
        payload = path.read_bytes()
        records.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    inventory_text = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    text(inventory_path, inventory_text)
    inventory_bytes = inventory_path.read_bytes()
    receipt = {
        "schema_version": "semuav.dataset-source.v1",
        "dataset": "visdrone",
        "source": {"scope": SOURCE_PROVENANCE_SCOPE},
        "artifacts": artifacts,
        "content_hash_manifest": {
            "path": str(inventory_path),
            "files": len(records),
            "bytes": sum(int(record["bytes"]) for record in records),
            "sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        },
        "inventory": {
            "files": len(records) + 1,
            "bytes": sum(int(record["bytes"]) for record in records) + len(inventory_bytes),
        },
    }
    return text(root / "SOURCE_RECEIPT.json", json.dumps(receipt))


def write_ledger(root: Path, status: str = "OPEN") -> Path:
    return text(
        root / "test_access_ledger.json",
        json.dumps({
            "status": status,
            "opened_utc": "2026-08-31T12:00:00Z",
            "protocol_hash": "a" * 64,
            "config_hash": "b" * 64,
            "code_hash": "c" * 64,
        }),
    )


class PublisherSplitWorkflowTests(unittest.TestCase):
    def test_train_hash_split_is_deterministic_group_safe_and_component_scoped(self) -> None:
        rows = []
        for component in ("det_train", "mot_train"):
            for group in range(40):
                for frame in range(2):
                    rows.append({
                        "sample_id": f"{component}-{group}-{frame}",
                        "component": component,
                        "publisher_partition": "train",
                        "study_partition": "development_train",
                        "split_group": f"group-{group}",
                        "sequence_id": f"{component}:group-{group}",
                        "frame_index": frame + 1,
                    })
        source = pd.DataFrame(rows)
        first = assign_publisher_splits(source, seed=17)
        second = assign_publisher_splits(source.sample(frac=1.0, random_state=8), seed=17)
        mapping_first = first.set_index("sample_id")["split"].to_dict()
        mapping_second = second.set_index("sample_id")["split"].to_dict()
        self.assertEqual(mapping_first, mapping_second)
        self.assertEqual(first.groupby("split_unit")["split"].nunique().max(), 1)
        self.assertEqual(set(first["split"]), {"train", "calibration"})
        self.assertTrue(audit_publisher_splits(first)["passed"].all())

    def test_val_maps_to_validation_and_test_requires_open_hash_bound_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_test_tree(root)
            permit = load_untouched_test_permit(write_ledger(root))
            test = build_source_manifest(
                root,
                components=("det_test_dev", "mot_test_dev"),
                test_permit=permit,
            )
            with self.assertRaises(UntouchedTestAccessError):
                assign_publisher_splits(test)
            assigned = assign_publisher_splits(test, test_permit=permit)
            self.assertTrue(assigned["split"].eq("test").all())
            payload = json.loads(permit.ledger_path.read_text(encoding="utf-8"))
            payload["status"] = "COMPLETE"
            permit.ledger_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(UntouchedTestAccessError):
                assign_publisher_splits(test, test_permit=permit)


class AnalysisManifestTests(unittest.TestCase):
    @staticmethod
    def sampling_manifest(*, split: str = "validation", ledger_hash: str = "") -> pd.DataFrame:
        rows = []
        profiles = ("vulnerable_road_user", "heavy_vehicle", "general_mobility")
        component = "det_test_dev" if split == "test" else "det_val"
        for group_index in range(4):
            for frame_index in range(1, 6):
                source_id = f"source-{group_index}-{frame_index}"
                for profile in profiles:
                    rows.append({
                        "sample_id": f"{source_id}:{profile}",
                        "source_sample_id": source_id,
                        "mission_profile": profile,
                        "component": component,
                        "split": split,
                        "split_unit": f"unit-{group_index}",
                        "sequence_id": f"sequence-{group_index}",
                        "frame_index": frame_index,
                        "source_uid": f"{component}/{group_index}/{frame_index}.jpg",
                        "test_ledger_sha256": ledger_hash,
                        "gold_action": "RESPOND" if frame_index % 2 else "NO_ACTION",
                        "event": "outcome-must-not-affect-selection",
                        "risk": frame_index / 10,
                    })
        return pd.DataFrame(rows)

    def test_registered_profiles_expand_source_truth_without_unsupported_claims(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            det = root / "VisDrone2019-DET-train"
            frame = image(det / "images" / "000001_000001.jpg")
            text(det / "annotations" / "000001_000001.txt", "40,40,20,20,1,4,0,0\n")
            source = assign_publisher_splits(
                build_source_manifest(root, components=("det_train",)), seed=5,
            )
            with Image.open(frame) as opened:
                sizes = {str(source.iloc[0].sample_id): opened.size}
            analysis, truth = build_analysis_manifest(
                source, image_sizes=sizes, benchmark=benchmark(),
            )
            self.assertEqual(len(analysis), 3)
            self.assertEqual(len(truth), 3)
            self.assertEqual(set(analysis["mission_profile"]), {
                "vulnerable_road_user", "heavy_vehicle", "general_mobility",
            })
            actions = analysis.set_index("mission_profile")["gold_action"]
            self.assertEqual(actions["general_mobility"], ACTION_RESPOND)
            self.assertEqual(actions["heavy_vehicle"], ACTION_NO_ACTION)
            self.assertEqual(actions["vulnerable_road_user"], ACTION_NO_ACTION)
            self.assertTrue(analysis["mission"].eq(MISSION_VISDRONE_URBAN).all())
            self.assertFalse(analysis["is_synthetic"].any())
            self.assertTrue(analysis["ood"].eq(False).all())
            telemetry = [json.loads(value) for value in analysis["telemetry_json"]]
            self.assertEqual(
                {item["mission_profile"] for item in telemetry},
                {"vulnerable_road_user", "heavy_vehicle", "general_mobility"},
            )
            self.assertTrue(all(len(item["registered_roi_xyxy"]) == 4 for item in telemetry))
            payload = analysis.to_json().lower()
            self.assertFalse(any(term in payload for term in PROHIBITED_CLAIMS))
            self.assertTrue(truth["truth_sha256"].str.fullmatch(r"[0-9a-f]{64}").all())

    def test_analysis_serialization_round_trip_restores_json_structures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_development_tree(root)
            write_receipt(root)
            bundle = prepare_visdrone_bundle(
                root, root / "prepared", benchmark=benchmark(), seed=11,
                require_official_receipt=False,
            )
            loaded = bundle.load_analysis_manifest()
            direct = read_analysis_manifest(bundle.analysis_manifest)
            self.assertEqual(len(loaded), 12)
            self.assertEqual(len(direct), 12)
            self.assertTrue(loaded["objects"].map(lambda value: isinstance(value, list)).all())
            self.assertTrue(loaded["relations"].map(lambda value: isinstance(value, list)).all())
            self.assertTrue(loaded["evidence_roi"].map(lambda value: len(value) == 4).all())
            again = prepare_visdrone_bundle(
                root, root / "prepared", benchmark=benchmark(), seed=11,
                require_official_receipt=False,
            )
            self.assertEqual(again.artifact_manifest, bundle.artifact_manifest)

    def test_clip_paths_are_component_scoped_and_strictly_causal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mot = root / "VisDrone2019-MOT-train"
            paths = [
                image(mot / "sequences" / "uav000003" / f"{index:07d}.jpg")
                for index in (1, 2, 3)
            ]
            text(
                mot / "annotations" / "uav000003.txt",
                "".join(
                    f"{index},7,40,40,20,20,1,4,0,0\n" for index in (1, 2, 3)
                ),
            )
            source = assign_publisher_splits(
                build_source_manifest(root, components=("mot_train",)), seed=5,
            )
            sizes = {str(sample_id): (100, 100) for sample_id in source["sample_id"]}
            analysis, _ = build_analysis_manifest(
                source, image_sizes=sizes, benchmark=benchmark(),
            )
            profile = analysis[
                analysis["mission_profile"].eq("general_mobility")
            ].sort_values("frame_index")
            clips = [json.loads(value) for value in profile["clip_paths_json"]]
            self.assertEqual(clips[0], [])
            self.assertEqual(clips[1], [str(paths[0].resolve()), str(paths[1].resolve())])
            self.assertEqual(
                clips[2],
                [str(path.resolve()) for path in paths],
            )

    def test_registered_policy_configuration_is_exact_and_deadlines_are_positive(self) -> None:
        policies = registered_mission_policies(benchmark())
        self.assertEqual(set(policies), {
            "vulnerable_road_user", "heavy_vehicle", "general_mobility",
        })
        malformed = benchmark()
        malformed["mission_profiles"] = {"general_mobility": malformed["mission_profiles"]["general_mobility"]}
        with self.assertRaises(ValueError):
            registered_mission_policies(malformed)

    def test_registered_sampling_is_outcome_blind_temporal_and_profile_complete(self) -> None:
        manifest = self.sampling_manifest()
        selected, audit = select_registered_analysis_sample(
            manifest,
            seed=123,
            max_groups_per_component=2,
            max_frames_per_group=3,
        )
        relabeled = manifest.copy()
        relabeled["gold_action"] = "CHANGED"
        relabeled["event"] = "CHANGED"
        relabeled["risk"] = 1.0 - relabeled["risk"]
        repeated, repeated_audit = select_registered_analysis_sample(
            relabeled,
            seed=123,
            max_groups_per_component=2,
            max_frames_per_group=3,
        )
        self.assertEqual(
            selected["sample_id"].tolist(), repeated["sample_id"].tolist(),
        )
        self.assertEqual(
            audit["selection_sha256"].iloc[0], repeated_audit["selection_sha256"].iloc[0],
        )
        self.assertTrue(audit["passed"].all())
        self.assertTrue(audit["outcome_columns_read_json"].eq("[]").all())
        profile_sets = selected.groupby("source_sample_id")["mission_profile"].agg(set)
        self.assertTrue(profile_sets.map(lambda value: len(value) == 3).all())
        self.assertEqual(selected["split_unit"].nunique(), 2)
        self.assertLessEqual(
            selected.drop_duplicates("source_sample_id").groupby("split_unit").size().max(), 3,
        )

    def test_registered_test_sampling_revalidates_the_open_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            permit = load_untouched_test_permit(write_ledger(root))
            manifest = self.sampling_manifest(
                split="test", ledger_hash=permit.ledger_sha256,
            )
            with self.assertRaises(UntouchedTestAccessError):
                select_registered_analysis_sample(
                    manifest,
                    seed=9,
                    max_groups_per_component=1,
                    max_frames_per_group=2,
                )
            selected, audit = select_registered_analysis_sample(
                manifest,
                seed=9,
                max_groups_per_component=1,
                max_frames_per_group=2,
                test_permit=permit,
            )
            self.assertTrue(selected["split"].eq("test").all())
            self.assertTrue(audit["passed"].all())

    def test_sampling_wrapper_consumes_only_frozen_config_caps(self) -> None:
        configured = benchmark()
        configured["analysis_sampling"] = {
            "method": "outcome_blind_group_hash_temporal_v1",
            "max_groups_per_component": {
                "train": 1, "calibration": 1, "validation": 2, "test": 3,
            },
            "max_frames_per_group": {
                "train": 2, "calibration": 2, "validation": 3, "test": 4,
            },
        }
        self.assertEqual(
            registered_analysis_sampling_limits(configured, "validation"), (2, 3),
        )
        selected, audit = select_registered_analysis_sample_from_config(
            self.sampling_manifest(),
            benchmark=configured,
            split="validation",
            seed=33,
        )
        self.assertEqual(selected["split_unit"].nunique(), 2)
        self.assertLessEqual(
            selected.drop_duplicates("source_sample_id").groupby("split_unit").size().max(), 3,
        )
        self.assertTrue(audit["passed"].all())
        configured["analysis_sampling"]["method"] = "outcome_aware"
        with self.assertRaises(StudyGateError):
            registered_analysis_sampling_limits(configured, "validation")

    def test_registered_sample_export_is_hash_bound_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected, audit = select_registered_analysis_sample(
                self.sampling_manifest(),
                seed=7,
                max_groups_per_component=1,
                max_frames_per_group=2,
            )
            outputs = write_registered_analysis_sample(selected, audit, root / "sample")
            self.assertEqual(set(outputs), {"sample", "audit", "artifact_manifest"})
            verify_artifact_hash_manifest(outputs["artifact_manifest"])
            with self.assertRaises(StudyGateError):
                write_registered_analysis_sample(selected, audit, root / "sample")


class TraceAndReportingTests(unittest.TestCase):
    def test_registered_trace_design_is_disjoint_paired_raw_service_and_explicitly_emulated(self) -> None:
        first = generate_registered_trace_manifest(seed=99)
        second = generate_registered_trace_manifest(seed=99)
        pd.testing.assert_frame_equal(first, second)
        validation = set(first.loc[first["split"].eq("validation"), "family"])
        test = set(first.loc[first["split"].eq("test"), "family"])
        self.assertEqual(validation, set(VALIDATION_TRACE_FAMILIES))
        self.assertEqual(test, set(TEST_TRACE_FAMILIES))
        self.assertFalse(validation & test)
        self.assertEqual(first.loc[first["split"].eq("test"), "trace_id"].nunique(), 30)
        self.assertTrue(first["semantics"].eq("raw_service").all())
        self.assertTrue(first["provenance_class"].eq("registered_emulation").all())
        self.assertEqual(
            first.groupby("trace_id")["direction"].agg(lambda value: set(value)).tolist()[0],
            {"uplink", "downlink"},
        )

    def test_readiness_ledger_fails_closed_and_permanently_blocks_unsupported_claims(self) -> None:
        blocked = build_readiness_claim_ledger(phase="report", evidence={})
        sensor = blocked[blocked["item_id"].eq("visdrone_sensor_space_results")].iloc[0]
        self.assertEqual(sensor.status, "BLOCKED")
        prohibited = blocked[blocked["item_type"].eq("prohibited_claim")]
        self.assertEqual(set(prohibited["item_id"]), set(PROHIBITED_CLAIMS))
        self.assertTrue(prohibited["status"].eq("BLOCKED").all())
        complete = build_readiness_claim_ledger(
            phase="report",
            evidence={
                "official_visdrone_receipt": True,
                "publisher_group_splits_passed": True,
                "real_detector_inference": True,
                "real_vlm_inference": True,
                "validation_gates_passed": True,
                "protocol_frozen": True,
                "one_time_test_complete": True,
                "paper_artifacts_hashed": True,
            },
            hypotheses={"H1": "PASS", "H2": "FAIL", "H3": "BLOCKED", "H4": "PASS"},
        )
        self.assertEqual(
            complete.loc[complete["item_id"].eq("visdrone_sensor_space_results"), "status"].item(),
            "ELIGIBLE",
        )
        self.assertEqual(complete.loc[complete["item_id"].eq("H5"), "status"].item(), "N_A")

    def test_not_estimable_hypothesis_preserves_reason_and_count_gates(self) -> None:
        evidence = {
            "official_visdrone_receipt": True,
            "publisher_group_splits_passed": True,
            "real_detector_inference": True,
            "real_vlm_inference": True,
            "validation_gates_passed": True,
            "protocol_frozen": True,
            "one_time_test_complete": True,
            "paper_artifacts_hashed": True,
        }
        ledger = build_readiness_claim_ledger(
            phase="report",
            evidence=evidence,
            hypotheses={
                "H3": {
                    "status": "NOT_ESTIMABLE",
                    "reason": "registered mission-priority group minima were not met",
                    "count_gates": [
                        {
                            "gate": "mission_priority_positive_visual_groups",
                            "observed": 117,
                            "required": 140,
                            "passed": False,
                        },
                        {
                            "gate": "mission_priority_negative_visual_groups",
                            "observed": 73,
                            "required": 73,
                            "passed": True,
                        },
                    ],
                },
            },
        )
        h3 = ledger.loc[ledger["item_id"].eq("H3")].iloc[0]
        self.assertEqual(h3.status, "NOT_ESTIMABLE")
        self.assertFalse(bool(h3.eligible))
        self.assertIn("group minima", h3.reason)
        gates = json.loads(h3.count_gates_json)
        self.assertEqual(gates[0]["observed"], 117)
        self.assertEqual(gates[0]["required"], 140)
        self.assertFalse(gates[0]["passed"])
        with self.assertRaisesRegex(ValueError, "explicit reason"):
            build_readiness_claim_ledger(
                phase="report",
                evidence=evidence,
                hypotheses={"H3": "NOT_ESTIMABLE"},
            )

    def test_detector_fallback_preserves_generic_claim_but_cannot_certify_vlm_claims(self) -> None:
        generic_evidence = {
            "official_visdrone_receipt": True,
            "publisher_group_splits_passed": True,
            "real_detector_inference": True,
            "real_vlm_inference": True,  # A nonselected/upper-bound row is insufficient.
            "validation_gates_passed": True,
            "protocol_frozen": True,
            "one_time_test_complete": True,
            "paper_artifacts_hashed": True,
        }
        fallback = build_readiness_claim_ledger(
            phase="report",
            evidence=generic_evidence,
            hypotheses={"H1": "PASS"},
            selected_vlm_used=False,
        )
        self.assertEqual(
            fallback.loc[
                fallback["item_id"].eq("visdrone_sensor_space_results"), "status"
            ].item(),
            "ELIGIBLE",
        )
        self.assertEqual(
            fallback.loc[
                fallback["item_id"].eq("selected_vlm_contribution"), "status"
            ].item(),
            "BLOCKED",
        )
        h1 = fallback.loc[fallback["item_id"].eq("H1")].iloc[0]
        self.assertEqual(h1.status, "PASS")
        self.assertEqual(h1.deployed_perception_scope, "detector_semantic_scope")

        selected = build_readiness_claim_ledger(
            phase="report",
            evidence={**generic_evidence, "selected_vlm_inference": True},
            hypotheses={"H1": "PASS"},
            selected_vlm_used=True,
        )
        self.assertEqual(
            selected.loc[
                selected["item_id"].eq("selected_vlm_contribution"), "status"
            ].item(),
            "ELIGIBLE",
        )
        self.assertEqual(
            selected.loc[selected["item_id"].eq("H1"), "status"].item(),
            "PASS",
        )
        self.assertTrue(
            selected["deployed_perception_scope"].eq("selected_vlm_scope").all()
        )

    def test_paper_tables_and_hash_manifest_detect_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table_manifest = export_paper_tables(
                {"main_results": pd.DataFrame({"policy": ["proposed"], "rate": [0.95]})},
                root / "tables",
                scientific=True,
            )
            self.assertEqual(set(table_manifest["format"]), {"csv", "markdown", "latex"})
            manifest = write_artifact_hash_manifest(
                {"results": root / "tables" / "main_results.csv"},
                root / "PAPER_RESULTS_MANIFEST.json",
                root=root,
                metadata={"evidence_class": "empirical"},
            )
            verified = verify_artifact_hash_manifest(manifest)
            self.assertIn("results", verified)
            with (root / "tables" / "main_results.csv").open("a", encoding="utf-8") as handle:
                handle.write("tampered\n")
            with self.assertRaises(StudyGateError):
                verify_artifact_hash_manifest(manifest)


class PaperCompletionReceiptTests(unittest.TestCase):
    def test_completion_receipt_verifies_both_manifests_and_all_run_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "frozen_protocol.json"
            test_ledger = root / "test_access_ledger.json"
            result = root / "result.csv"
            report_environment = root / "REPORT_ENVIRONMENT_RECEIPT.json"
            protocol.write_text('{"protocol_hash":"frozen"}\n', encoding="utf-8")
            test_ledger.write_text('{"status":"COMPLETE"}\n', encoding="utf-8")
            result.write_text("metric,value\nrate,0.9\n", encoding="utf-8")
            report_environment.write_text(
                '{"status":"VERIFIED_EXACT_MATCH"}\n', encoding="utf-8",
            )
            inner = write_artifact_hash_manifest(
                {
                    "result": result,
                    "report_environment_receipt": report_environment,
                },
                root / "PAPER_ARTIFACTS_HASH_MANIFEST.json",
                root=root,
                metadata={
                    "report_environment_receipt_sha256": hash_file(
                        report_environment
                    ),
                },
            )
            claim_ledger = root / "claim_ledger.csv.gz"
            pd.DataFrame([{
                "item_id": "H1",
                "paper_artifact_manifest_sha256": hash_file(inner),
                "paper_bundle_status": "ELIGIBLE_PENDING_COMPLETION_RECEIPT",
                "report_environment_receipt_sha256": hash_file(
                    report_environment
                ),
            }]).to_csv(claim_ledger, index=False, compression="gzip")
            outer = write_artifact_hash_manifest(
                {
                    "paper_artifact_manifest": inner,
                    "final_claim_ledger": claim_ledger,
                    "frozen_protocol": protocol,
                    "test_access_ledger": test_ledger,
                },
                root / "PAPER_RESULTS_MANIFEST.json",
                root=root,
                metadata={
                    "bundle_status": "ELIGIBLE_PENDING_COMPLETION_RECEIPT",
                    "inner_manifest_sha256": hash_file(inner),
                    "final_claim_ledger_sha256": hash_file(claim_ledger),
                    "report_environment_receipt_sha256": hash_file(
                        report_environment
                    ),
                },
            )
            receipt = root / "FINAL_COMPLETION_RECEIPT.json"
            receipt.write_text(json.dumps({
                "schema_version": "semuav.paper-completion-receipt.v1",
                "status": "COMPLETE",
                "paper_results_manifest_path": str(outer.resolve()),
                "paper_results_manifest_sha256": hash_file(outer),
                "final_claim_ledger_path": str(claim_ledger.resolve()),
                "final_claim_ledger_sha256": hash_file(claim_ledger),
                "report_environment_receipt_path": str(
                    report_environment.resolve()
                ),
                "report_environment_receipt_sha256": hash_file(
                    report_environment
                ),
                "protocol_sha256": hash_file(protocol),
                "test_ledger_sha256": hash_file(test_ledger),
            }), encoding="utf-8")

            verified = verify_paper_completion_receipt(
                receipt,
                expected_results_manifest=outer,
                expected_claim_ledger=claim_ledger,
                expected_protocol=protocol,
                expected_test_ledger=test_ledger,
                expected_report_environment_receipt=report_environment,
            )
            self.assertEqual(verified["paper_artifact_manifest"], inner.resolve())

            receipt_link = root / "completion-link.json"
            try:
                receipt_link.symlink_to(receipt)
            except (OSError, NotImplementedError):
                receipt_link = None
            if receipt_link is not None:
                with self.assertRaisesRegex(StudyGateError, "symbolic link"):
                    verify_paper_completion_receipt(receipt_link)

            protocol.write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(StudyGateError):
                verify_paper_completion_receipt(
                    receipt,
                    expected_results_manifest=outer,
                    expected_claim_ledger=claim_ledger,
                    expected_protocol=protocol,
                    expected_test_ledger=test_ledger,
                    expected_report_environment_receipt=report_environment,
                )


class SourceProvenanceAndBundleTests(unittest.TestCase):
    @staticmethod
    def _complete_source(root: Path) -> None:
        build_development_tree(root)
        build_test_tree(root)
        write_receipt(root)

    def test_official_receipt_requires_all_six_hashed_components(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_source(root)
            verified = verify_visdrone_source_provenance(root, full_integrity=True)
            self.assertEqual(verified["scope"], SOURCE_PROVENANCE_SCOPE)
            self.assertTrue(verified["full_integrity"])
            receipt = json.loads((root / "SOURCE_RECEIPT.json").read_text(encoding="utf-8"))
            receipt["artifacts"].pop()
            (root / "SOURCE_RECEIPT.json").write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaises(StudyGateError):
                verify_visdrone_source_provenance(root)

    def test_source_inventory_rejects_wrong_envelope_unsafe_paths_and_unlisted_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_source(root)
            receipt_path = root / "SOURCE_RECEIPT.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["content_hash_manifest"]["files"] += 1
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(StudyGateError, "counts"):
                verify_visdrone_source_provenance(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_source(root)
            inventory_path = root / "SOURCE_FILES_SHA256.jsonl"
            records = [
                json.loads(line)
                for line in inventory_path.read_text(encoding="utf-8").splitlines()
            ]
            records[0]["path"] = "../escape.bin"
            inventory_text = "".join(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                for record in records
            )
            inventory_path.write_text(inventory_text, encoding="utf-8")
            receipt_path = root / "SOURCE_RECEIPT.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["content_hash_manifest"]["sha256"] = hashlib.sha256(
                inventory_text.encode("utf-8")
            ).hexdigest()
            receipt["inventory"]["bytes"] = (
                int(receipt["content_hash_manifest"]["bytes"])
                + len(inventory_text.encode("utf-8"))
            )
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(StudyGateError, "non-canonical|unsafe|escapes"):
                verify_visdrone_source_provenance(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_source(root)
            (root / "UNLISTED.bin").write_bytes(b"not in the inventory")
            with self.assertRaisesRegex(StudyGateError, "inventory differ"):
                verify_visdrone_source_provenance(root)

    def test_full_integrity_detects_same_size_content_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_source(root)
            target = root / "VisDrone2019-DET-train" / "annotations" / "000001_train_000001.txt"
            payload = bytearray(target.read_bytes())
            payload[0] ^= 1
            target.write_bytes(payload)
            # The status-mode check deliberately avoids reading every source
            # byte, while the scientific gate recomputes all content hashes.
            verify_visdrone_source_provenance(root)
            with self.assertRaisesRegex(StudyGateError, "SHA-256"):
                verify_visdrone_source_provenance(root, full_integrity=True)

    def test_full_integrity_component_subset_does_not_hash_test_media(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_source(root)
            target = root / "VisDrone2019-DET-test-dev" / "images" / "000002.jpg"
            payload = bytearray(target.read_bytes())
            payload[-1] ^= 1
            target.write_bytes(payload)
            development = verify_visdrone_source_provenance(
                root,
                full_integrity=True,
                components=("det_train", "det_val", "mot_train", "mot_val"),
            )
            self.assertEqual(
                development["full_integrity_components"],
                ["det_train", "det_val", "mot_train", "mot_val"],
            )
            with self.assertRaisesRegex(StudyGateError, "SHA-256"):
                verify_visdrone_source_provenance(
                    root,
                    full_integrity=True,
                    components=("det_test_dev", "mot_test_dev"),
                )

    def test_inventory_only_duplicate_and_test_group_capacity_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "VisDrone"
            # Thirty independent test identities, with deliberately no test
            # annotations: this gate is path/hash based and outcome blind.
            for index in range(MINIMUM_TEST_NATURAL_GROUPS - 1):
                path = root / "VisDrone2019-DET-test-dev" / "images" / f"{index:07d}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"det-test-{index}".encode("ascii"))
            mot = root / "VisDrone2019-MOT-test-dev" / "sequences" / "sequence-001" / "0000001.jpg"
            mot.parent.mkdir(parents=True, exist_ok=True)
            mot.write_bytes(b"mot-test-unique")
            write_receipt(root)
            output = Path(directory) / "audit"
            artifacts = write_visdrone_media_partition_audit(root, output)
            self.assertEqual(artifacts["test_natural_groups"], MINIMUM_TEST_NATURAL_GROUPS)
            self.assertFalse((root / "VisDrone2019-DET-test-dev" / "annotations").exists())
            permit = load_untouched_test_permit(write_ledger(Path(directory)))
            verified = verify_visdrone_media_partition_audit(
                artifacts["receipt_path"],
                expected_inventory_sha256=artifacts["inventory_sha256"],
                test_permit=permit,
                require_test_permit=True,
            )
            self.assertTrue(verified["passed"])

    def test_inventory_only_audit_fails_on_cross_partition_media_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "VisDrone"
            train = root / "VisDrone2019-DET-train" / "images" / "train-sequence_1.jpg"
            train.parent.mkdir(parents=True, exist_ok=True)
            train.write_bytes(b"duplicated-media")
            for index in range(MINIMUM_TEST_NATURAL_GROUPS - 1):
                path = root / "VisDrone2019-DET-test-dev" / "images" / f"{index:07d}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(
                    b"duplicated-media" if index == 0 else f"test-{index}".encode("ascii")
                )
            mot = root / "VisDrone2019-MOT-test-dev" / "sequences" / "sequence-001" / "0000001.jpg"
            mot.parent.mkdir(parents=True, exist_ok=True)
            mot.write_bytes(b"unique-mot")
            write_receipt(root)
            output = Path(directory) / "audit"
            with self.assertRaisesRegex(StudyGateError, "cross VisDrone study partitions"):
                write_visdrone_media_partition_audit(root, output)
            receipt = json.loads(
                (output / "VISDRONE_MEDIA_PARTITION_AUDIT.json").read_text(encoding="utf-8")
            )
            self.assertFalse(receipt["passed"])
            rows = pd.read_csv(
                output / "visdrone_media_partition_hash_audit.csv.gz", compression="gzip",
            )
            duplicated = rows[rows["cross_partition_duplicate"].astype(bool)]
            crossed = set(duplicated["study_partition"])
            self.assertIn("test", crossed)
            self.assertEqual(len(crossed.intersection({"train", "calibration"})), 1)

    def test_prepare_capacity_preflight_precedes_expensive_scans_and_preserves_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "VisDrone"
            build_development_tree(root)
            build_test_tree(root)
            make_fixture_media_unique(root)
            write_receipt(root)
            destination = Path(directory) / "prepared"
            with (
                patch(
                    "semuav.visdrone_workflow.build_development_manifest",
                    side_effect=AssertionError("expensive source scan ran"),
                ) as source_scan,
                patch(
                    "semuav.visdrone_workflow._objects_and_annotation_quality",
                    side_effect=AssertionError("expensive annotation scan ran"),
                ) as annotations,
                patch(
                    "semuav.visdrone_workflow._image_dimensions",
                    side_effect=AssertionError("expensive image scan ran"),
                ) as dimensions,
                patch(
                    "semuav.visdrone_workflow.build_analysis_manifest",
                    side_effect=AssertionError("expensive analysis ran"),
                ) as analysis,
            ):
                with self.assertRaisesRegex(
                    StudyGateError,
                    "Outcome-blind test inventory has 2 natural groups, fewer than the registered minimum 30",
                ):
                    prepare_visdrone_bundle(
                        root,
                        destination,
                        benchmark=benchmark(),
                        minimum_test_groups=MINIMUM_TEST_NATURAL_GROUPS,
                    )
            source_scan.assert_not_called()
            annotations.assert_not_called()
            dimensions.assert_not_called()
            analysis.assert_not_called()
            diagnostics = destination.parent / f"{destination.name}.preflight"
            capacity_path = diagnostics / "visdrone_test_group_capacity_audit.csv"
            media_path = diagnostics / "visdrone_media_partition_hash_audit.csv.gz"
            receipt_path = diagnostics / "VISDRONE_MEDIA_PARTITION_AUDIT.json"
            self.assertTrue(capacity_path.is_file())
            self.assertTrue(media_path.is_file())
            self.assertTrue(receipt_path.is_file())
            self.assertFalse(destination.exists())
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertFalse(receipt["passed"])
            self.assertFalse(receipt["capacity_passed"])
            self.assertTrue(receipt["test_metadata_only"])
            self.assertFalse(receipt["test_labels_or_pixels_read"])
            capacity = pd.read_csv(capacity_path)
            total = capacity.loc[capacity["scope"].eq("all_test_components")].iloc[0]
            self.assertEqual(int(total["available_natural_groups"]), 2)
            self.assertEqual(int(total["required_natural_groups"]), 30)
            self.assertFalse(bool(total["passed"]))

    def test_failed_prepare_capacity_diagnostics_are_byte_stable_across_retries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "VisDrone"
            build_development_tree(root)
            build_test_tree(root)
            make_fixture_media_unique(root)
            write_receipt(root)
            destination = Path(directory) / "prepared"
            diagnostics = destination.parent / f"{destination.name}.preflight"
            artifact_names = (
                "visdrone_test_group_capacity_audit.csv",
                "visdrone_media_partition_hash_audit.csv.gz",
                "VISDRONE_MEDIA_PARTITION_AUDIT.json",
            )
            snapshots = []
            messages = []
            for _ in range(2):
                with self.assertRaises(StudyGateError) as captured:
                    prepare_visdrone_bundle(
                        root,
                        destination,
                        benchmark=benchmark(),
                        minimum_test_groups=MINIMUM_TEST_NATURAL_GROUPS,
                    )
                messages.append(str(captured.exception))
                snapshots.append({
                    name: hash_file(diagnostics / name)
                    for name in artifact_names
                })
            self.assertEqual(messages[0], messages[1])
            self.assertEqual(snapshots[0], snapshots[1])
            self.assertFalse(list(destination.parent.glob(f".{destination.name}.staging-*")))

    def test_prepare_progress_reports_real_within_stage_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "VisDrone"
            build_development_tree(root)
            build_test_tree(root)
            make_fixture_media_unique(root)
            write_receipt(root)
            destination = Path(directory) / "prepared"
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                bundle = prepare_visdrone_bundle(
                    root,
                    destination,
                    benchmark=benchmark(),
                    minimum_test_groups=2,
                )
            events = [
                json.loads(line.split(" ", 1)[1])
                for line in stream.getvalue().splitlines()
                if line.startswith("VISDRONE_PREPARE_PROGRESS ")
            ]
            stages = [str(event["stage"]) for event in events]
            self.assertLess(
                stages.index("media_preflight_complete"),
                stages.index("annotation_quality_start"),
            )
            for stage in (
                "source_integrity_files",
                "annotation_quality_samples",
                "media_dimensions",
            ):
                counts = [
                    (int(event["completed"]), int(event["total"]))
                    for event in events if event["stage"] == stage
                ]
                self.assertGreaterEqual(len(counts), 2)
                self.assertEqual(counts[-1][0], counts[-1][1])
                self.assertTrue(all(
                    earlier[0] < later[0]
                    for earlier, later in zip(counts, counts[1:])
                ))
                self.assertEqual(len({total for _, total in counts}), 1)
            self.assertEqual(stages[-1], "bundle_complete")
            self.assertTrue(bundle.receipt.is_file())
            self.assertFalse((destination.parent / f"{destination.name}.preflight").exists())

    def test_existing_bundle_rechecks_bound_source_bytes_before_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "VisDrone"
            build_development_tree(root)
            build_test_tree(root)
            make_fixture_media_unique(root)
            write_receipt(root)
            destination = Path(directory) / "prepared"
            prepare_visdrone_bundle(
                root,
                destination,
                benchmark=benchmark(),
                minimum_test_groups=2,
            )
            target = (
                root / "VisDrone2019-DET-train" / "annotations"
                / "000001_train_000001.txt"
            )
            payload = bytearray(target.read_bytes())
            payload[0] ^= 1
            target.write_bytes(payload)
            with self.assertRaisesRegex(StudyGateError, "SHA-256"):
                prepare_visdrone_bundle(
                    root,
                    destination,
                    benchmark=benchmark(),
                    minimum_test_groups=2,
                )

    def test_existing_bundle_rejects_changed_source_provenance_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "VisDrone"
            build_development_tree(root)
            build_test_tree(root)
            make_fixture_media_unique(root)
            write_receipt(root)
            destination = Path(directory) / "prepared"
            bundle = prepare_visdrone_bundle(
                root,
                destination,
                benchmark=benchmark(),
                minimum_test_groups=2,
            )
            receipt = json.loads(bundle.receipt.read_text(encoding="utf-8"))
            receipt["source_provenance"]["inventory_sha256"] = "f" * 64
            bundle.receipt.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(
                StudyGateError,
                "bound to different source provenance",
            ):
                prepare_visdrone_bundle(
                    root,
                    destination,
                    benchmark=benchmark(),
                    minimum_test_groups=2,
                )

    def test_source_inventory_rejects_symlinks_and_wrong_publisher_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_source(root)
            target = root / "VisDrone2019-DET-train" / "annotations" / "000001_train_000001.txt"
            (root / "linked-source.txt").symlink_to(target)
            with self.assertRaisesRegex(StudyGateError, "symbolic link"):
                verify_visdrone_source_provenance(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_source(root)
            receipt_path = root / "SOURCE_RECEIPT.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["schema_version"] = "semuav.dataset-source.v0"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(StudyGateError, "schema_version"):
                verify_visdrone_source_provenance(root)

    def test_test_permit_is_rejected_before_any_source_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing_source = root / "source-must-not-be-inspected"
            with self.assertRaises(UntouchedTestAccessError):
                prepare_visdrone_bundle(
                    missing_source,
                    root / "prepared-missing-permit",
                    benchmark=benchmark(),
                    partition="test",
                )

            permit = load_untouched_test_permit(write_ledger(root))
            ledger = json.loads(permit.ledger_path.read_text(encoding="utf-8"))
            ledger["status"] = "COMPLETE"
            permit.ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            with self.assertRaises(UntouchedTestAccessError):
                prepare_visdrone_bundle(
                    missing_source,
                    root / "prepared-invalid-permit",
                    benchmark=benchmark(),
                    partition="test",
                    test_permit=permit,
                )

    def test_test_bundle_cannot_open_or_resume_after_ledger_closes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_test_tree(root)
            write_receipt(root)
            permit = load_untouched_test_permit(write_ledger(root))
            bundle = prepare_visdrone_bundle(
                root,
                root / "prepared-test",
                benchmark=benchmark(),
                partition="test",
                test_permit=permit,
                require_official_receipt=False,
            )
            self.assertEqual(bundle.partition, "test")
            self.assertTrue(bundle.load_analysis_manifest()["split"].eq("test").all())
            with self.assertRaises(UntouchedTestAccessError):
                load_prepared_visdrone_bundle(bundle.output_dir)
            load_prepared_visdrone_bundle(bundle.output_dir, test_permit=permit)
            payload = json.loads(permit.ledger_path.read_text(encoding="utf-8"))
            payload["status"] = "COMPLETE"
            permit.ledger_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(UntouchedTestAccessError):
                bundle.load_analysis_manifest()
            with self.assertRaises(UntouchedTestAccessError):
                prepare_visdrone_bundle(
                    root,
                    root / "another-test",
                    benchmark=benchmark(),
                    partition="test",
                    test_permit=permit,
                    require_official_receipt=False,
                )

    def test_exact_zero_det_rows_are_hash_bound_and_excluded_at_registered_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_development_tree(root)
            retained_annotation = (
                root / "VisDrone2019-DET-train" / "annotations"
                / "000001_train_000001.txt"
            )
            ignored_row = "10,10,20,0,0,0,0,0"
            retained_annotation.write_text(
                "40,40,20,20,1,4,0,0\n" + ignored_row + "\n",
                encoding="utf-8",
            )
            excluded_stem = "000002_train_000001"
            image(
                root / "VisDrone2019-DET-train" / "images"
                / f"{excluded_stem}.jpg"
            )
            evaluated_row = "10,10,20,0,1,4,0,0"
            text(
                root / "VisDrone2019-DET-train" / "annotations"
                / f"{excluded_stem}.txt",
                "40,40,20,20,1,4,0,0\n" + evaluated_row + "\n",
            )

            first = prepare_visdrone_bundle(
                root, root / "prepared-a", benchmark=benchmark(), seed=11,
                require_official_receipt=False,
            )
            second = prepare_visdrone_bundle(
                root, root / "prepared-b", benchmark=benchmark(), seed=11,
                require_official_receipt=False,
            )
            audit = first.load_annotation_exclusions()
            self.assertEqual(len(audit), 2)
            self.assertEqual(
                set(audit["policy_version"]),
                {ANNOTATION_QUALITY_POLICY_VERSION},
            )
            self.assertEqual(set(audit["source_line"].astype(int)), {2})
            self.assertEqual(
                set(audit["action"]),
                {"exclude_annotation_row", "exclude_source_sample"},
            )
            self.assertEqual(
                set(audit["raw_row_sha256"]),
                {
                    hashlib.sha256(ignored_row.encode("utf-8")).hexdigest(),
                    hashlib.sha256(evaluated_row.encode("utf-8")).hexdigest(),
                },
            )
            # The audit is per publisher row, not multiplied by the three
            # registered mission profiles, and is byte-deterministic.
            self.assertEqual(
                first.annotation_exclusions.read_bytes(),
                second.annotation_exclusions.read_bytes(),
            )

            excluded_sample_id = f"visdrone:det_train:{excluded_stem}"
            retained_sample_id = "visdrone:det_train:000001_train_000001"
            source = first.load_source_manifest()
            split = pd.read_csv(first.split_manifest, compression="gzip")
            analysis = first.load_analysis_manifest()
            truth = first.load_source_truth()
            self.assertEqual(len(source), 4)
            self.assertNotIn(excluded_sample_id, set(source["sample_id"]))
            self.assertNotIn(excluded_sample_id, set(split["sample_id"]))
            self.assertNotIn(excluded_sample_id, set(analysis["source_sample_id"]))
            self.assertNotIn(excluded_sample_id, set(truth["source_sample_id"]))
            retained = analysis[analysis["source_sample_id"].eq(retained_sample_id)]
            self.assertEqual(len(retained), 3)
            self.assertTrue(retained["objects"].map(lambda rows: len(rows) == 1).all())

            artifact_paths = verify_artifact_hash_manifest(first.artifact_manifest)
            self.assertEqual(
                artifact_paths["publisher_annotation_exclusions"],
                first.annotation_exclusions.resolve(),
            )
            receipt = json.loads(first.receipt.read_text(encoding="utf-8"))
            self.assertEqual(receipt["rows"]["source_before_annotation_quality"], 5)
            self.assertEqual(receipt["rows"]["source"], 4)
            self.assertEqual(receipt["rows"]["annotation_exclusions"], 2)
            self.assertEqual(
                receipt["annotation_quality_policy"],
                {
                    "action_counts": {
                        "exclude_annotation_row": 1,
                        "exclude_source_sample": 1,
                    },
                    "evaluated_zero_action": "exclude_source_sample",
                    "excluded_annotation_rows": 2,
                    "excluded_source_samples": 1,
                    "ignored_zero_action": "exclude_annotation_row",
                    "negative_or_malformed_action": "fail_closed",
                    "version": ANNOTATION_QUALITY_POLICY_VERSION,
                },
            )

            changed_receipt = json.loads(second.receipt.read_text(encoding="utf-8"))
            changed_receipt["rows"]["source"] += 1
            second.receipt.write_text(json.dumps(changed_receipt), encoding="utf-8")
            with self.assertRaisesRegex(StudyGateError, "count changed"):
                load_prepared_visdrone_bundle(second.output_dir)

            first.annotation_exclusions.write_text(
                first.annotation_exclusions.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(StudyGateError, "Artifact changed"):
                load_prepared_visdrone_bundle(first.output_dir)

    def test_annotation_quality_policy_fails_closed_outside_exact_zero(self) -> None:
        invalid_rows = {
            "negative extent": "40,40,-1,20,1,4,0,0\n",
            "nonfinite extent": "40,40,nan,20,1,4,0,0\n",
            "malformed row": "40,40,20\n",
            "invalid score": "40,40,0,20,2,4,0,0\n",
            "invalid category": "40,40,0,20,1,12,0,0\n",
            "invalid visibility": "40,40,0,20,1,4,3,0\n",
        }
        for name, row in invalid_rows.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                build_development_tree(root)
                target = (
                    root / "VisDrone2019-DET-train" / "annotations"
                    / "000001_train_000001.txt"
                )
                target.write_text(row, encoding="utf-8")
                with self.assertRaises(VisDroneFormatError):
                    prepare_visdrone_bundle(
                        root, root / "prepared", benchmark=benchmark(),
                        require_official_receipt=False,
                    )


if __name__ == "__main__":
    unittest.main()
