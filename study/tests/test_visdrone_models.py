import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import yaml
from PIL import Image

from semuav.perception import (
    VISDRONE_GENERATION_POLICY_VERSION,
    VISDRONE_GENERATION_STRATEGY_BY_MODEL,
    VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256,
    aspect_preserving_letterbox,
    visdrone_assistant_prefill_for_model,
    visdrone_generation_policy_sha256,
    visdrone_proposal_decision_context,
)
from semuav.registry import StudyGateError, hash_file, hash_object, sha256_bytes
from semuav.visdrone_models import (
    DETECTOR_TRAINING_PLAN_SCHEMA,
    DETECTOR_TRAINING_RECEIPT_SCHEMA,
    OFFICIAL_COMPACT_MODEL_REPOS,
    OFFICIAL_MODEL_BACKENDS,
    OFFICIAL_RUNTIME_MODEL_IDS,
    OFFICIAL_UPPER_BOUND_ID,
    OFFICIAL_UPPER_BOUND_REPO,
    detector_training_command_plan,
    freeze_huggingface_models,
    prepare_ultralytics_visdrone_det,
    require_runtime_smoke_pass,
    run_one_image_runtime_smoke,
    validate_registered_model_configuration,
    verify_detector_checkpoint,
    verify_detector_training_receipt,
    verify_huggingface_model_lock,
    verify_ultralytics_visdrone_det,
    write_detector_training_receipt,
)


def registered_models():
    return {
        "vlms": [
            {
                "id": model_id,
                "repo": repo,
                "backend": OFFICIAL_MODEL_BACKENDS[model_id],
                "revision": "REQUIRED_40_HEX_COMMIT",
                "quantization": "none",
            }
            for model_id, repo in OFFICIAL_COMPACT_MODEL_REPOS.items()
        ],
        "upper_bound": {
            "id": OFFICIAL_UPPER_BOUND_ID,
            "repo": OFFICIAL_UPPER_BOUND_REPO,
            "backend": "qwen3_vl",
            "revision": "REQUIRED_40_HEX_COMMIT",
            "quantization": "none",
        },
    }


class FrozenModelAssetTests(unittest.TestCase):
    def _freeze(self, root: Path):
        cache = root / "hf-cache"
        snapshots = {}

        def resolver(repo, requested):
            self.assertIsNone(requested)
            return hashlib.sha1(repo.encode()).hexdigest()

        def downloader(repo, revision, cache_root):
            snapshot = cache_root / "snapshots" / revision
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text(
                json.dumps({"repo": repo, "revision": revision}), encoding="utf-8"
            )
            (snapshot / "weights.bin").write_bytes(repo.encode() * 2)
            snapshots[(repo, revision)] = snapshot.resolve()
            return snapshot

        lock_path = root / "MODEL_LOCK.json"
        payload = freeze_huggingface_models(
            registered_models(),
            cache_root=cache,
            lock_path=lock_path,
            resolver=resolver,
            downloader=downloader,
        )
        return lock_path, payload, snapshots

    def test_registered_configuration_requires_exact_official_model_set(self):
        entries = validate_registered_model_configuration(registered_models())
        self.assertEqual(
            [entry["model_id"] for entry in entries],
            [
                "qwen3_vl_2b",
                "smolvlm2_2b",
                "qwen3_vl_8b_upper_bound",
            ],
        )
        self.assertNotIn("moondream2", {
            entry["model_id"] for entry in entries
        })
        bad = registered_models()
        bad["vlms"][0]["repo"] = "third-party/repack"
        with self.assertRaisesRegex(StudyGateError, "official repository"):
            validate_registered_model_configuration(bad)

        missing = registered_models()
        missing["vlms"] = missing["vlms"][:-1]
        with self.assertRaisesRegex(StudyGateError, "exactly the two"):
            validate_registered_model_configuration(missing)

        added_moondream = registered_models()
        added_moondream["vlms"].append({
            "id": "moondream2",
            "repo": "vikhyatk/moondream2",
            "backend": "moondream2",
            "revision": "REQUIRED_40_HEX_COMMIT",
            "quantization": "none",
        })
        with self.assertRaisesRegex(StudyGateError, "exactly the two"):
            validate_registered_model_configuration(added_moondream)

        placeholder_upper = registered_models()
        placeholder_upper["upper_bound"] = {
            "id": "upper_bound", "repo": "REQUIRED_UPPER_BOUND_REPO"
        }
        with self.assertRaisesRegex(StudyGateError, "upper bound"):
            validate_registered_model_configuration(placeholder_upper)

    def test_runtime_smoke_gate_requires_exact_three_model_order(self):
        passing = pd.DataFrame.from_records([
            {"model_id": model_id, "status": "PASS", "error": None}
            for model_id in OFFICIAL_RUNTIME_MODEL_IDS
        ])
        require_runtime_smoke_pass(passing)

        reordered = passing.iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(StudyGateError, "exact.*3-model|order"):
            require_runtime_smoke_pass(reordered)

        with_moondream = pd.concat([
            passing,
            pd.DataFrame.from_records([{
                "model_id": "moondream2",
                "status": "PASS",
                "error": None,
            }]),
        ], ignore_index=True)
        with self.assertRaisesRegex(StudyGateError, "exact.*3-model"):
            require_runtime_smoke_pass(with_moondream)

        with self.assertRaisesRegex(StudyGateError, "exact.*3-model"):
            require_runtime_smoke_pass(passing.iloc[:-1].copy())

    def test_freeze_inventories_every_snapshot_and_offline_verifies_all_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path, payload, snapshots = self._freeze(root)
            self.assertEqual(len(payload["models"]), 3)
            self.assertTrue(all(item["file_count"] == 2 for item in payload["models"]))
            checked = []

            def offline(repo, revision, cache_root):
                checked.append((repo, revision, cache_root.resolve()))
                return snapshots[(repo, revision)]

            verified = verify_huggingface_model_lock(
                lock_path,
                models=registered_models(),
                offline_checker=offline,
            )
            self.assertEqual(len(verified["offline_verification"]), 3)
            self.assertEqual(len(checked), 3)
            self.assertTrue(all(item["status"] == "OFFLINE_VERIFIED" for item in verified["offline_verification"]))

    def test_lock_rejects_changed_snapshot_and_implicit_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path, payload, snapshots = self._freeze(root)
            with self.assertRaisesRegex(StudyGateError, "already exists"):
                freeze_huggingface_models(
                    registered_models(),
                    cache_root=root / "hf-cache",
                    lock_path=lock_path,
                    resolver=lambda repo, revision: hashlib.sha1(repo.encode()).hexdigest(),
                    downloader=lambda repo, revision, cache: snapshots[(repo, revision)],
                )
            Path(payload["models"][0]["snapshot_path"], "weights.bin").write_bytes(b"tampered")
            with self.assertRaisesRegex(StudyGateError, "bytes changed"):
                verify_huggingface_model_lock(
                    lock_path,
                    offline_checker=lambda repo, revision, cache: snapshots[(repo, revision)],
                )


class VisDroneDetectorPreparationTests(unittest.TestCase):
    @staticmethod
    def _write_component(root: Path, name: str, stem: str, annotation: str):
        component = root / name
        (component / "images").mkdir(parents=True, exist_ok=True)
        (component / "annotations").mkdir(exist_ok=True)
        Image.new("RGB", (100, 50), (20, 40, 60)).save(component / "images" / f"{stem}.jpg")
        (component / "annotations" / f"{stem}.txt").write_text(annotation, encoding="utf-8")

    def _write_detector_fixture(self, source: Path):
        train = "VisDrone2019-DET-train"
        rows = (
            (
                "000001_00001_d_0000001",
                "10,10,20,10,1,1,0,0\n"
                "0,0,4,4,1,0,0,0\n"
                "0,0,0,4,0,4,0,0\n",
            ),
            ("000001_00002_d_0000002", "80,40,30,20,1,10,1,2\n"),
            ("000002_00001_d_0000001", "10,10,20,10,1,4,0,0\n"),
            ("000003_00001_d_0000001", "10,10,20,10,1,6,0,0\n"),
            ("000004_00001_d_0000001", "10,10,20,10,1,8,0,0\n"),
        )
        for stem, annotation in rows:
            self._write_component(source, train, stem, annotation)
        # Malformed publisher validation and test rows prove preparation never
        # opens either component during detector optimization setup.
        self._write_component(
            source,
            "VisDrone2019-DET-val",
            "publisher_val_secret",
            "this,is,publisher,validation,and,must,stay,external\n",
        )
        self._write_component(
            source,
            "VisDrone2019-DET-test-dev",
            "publisher_test_secret",
            "this,is,the,untouched,test,partition\n",
        )

    @staticmethod
    def _study_manifest_frame(source: Path) -> pd.DataFrame:
        component_contract = {
            "VisDrone2019-DET-train": (
                "det_train", "train", "development_train"
            ),
            "VisDrone2019-DET-val": (
                "det_val", "val", "development_validation"
            ),
            "VisDrone2019-DET-test-dev": (
                "det_test_dev", "test-dev", "untouched_test"
            ),
        }
        rows = []
        for directory_name, (
            component,
            publisher_partition,
            study_partition,
        ) in component_contract.items():
            for image_path in sorted((source / directory_name / "images").glob("*.jpg")):
                natural_group = image_path.stem.split("_", 1)[0]
                if component == "det_train":
                    split = "calibration" if natural_group == "000004" else "train"
                elif component == "det_val":
                    split = "validation"
                else:
                    split = "test"
                rows.append(
                    {
                        "sample_id": f"visdrone:{component}:{image_path.stem}",
                        "source_uid": f"{component}/images/{image_path.name}",
                        "component": component,
                        "task": "DET",
                        "publisher_partition": publisher_partition,
                        "study_partition": study_partition,
                        "split": split,
                        "split_unit": f"visdrone:{component}:{natural_group}",
                        "media_path": str(image_path.resolve()),
                        "annotation_path": str(
                            (
                                source
                                / directory_name
                                / "annotations"
                                / f"{image_path.stem}.txt"
                            ).resolve()
                        ),
                    }
                )
        return pd.DataFrame.from_records(rows)

    def _write_study_manifest(self, source: Path, path: Path) -> tuple[Path, pd.DataFrame]:
        frame = self._study_manifest_frame(source)
        frame.to_csv(path, index=False)
        return path, frame

    def test_ultralytics_tree_uses_train_only_and_group_isolated_internal_val(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "VisDrone"
            self._write_detector_fixture(source)
            study_manifest, study_frame = self._write_study_manifest(
                source, root / "FROZEN_DEVELOPMENT_SPLIT.csv"
            )
            output = root / "ultralytics"
            result = prepare_ultralytics_visdrone_det(
                source,
                output,
                study_source_manifest=study_manifest,
                seed=17,
                internal_validation_fraction=0.25,
            )
            self.assertEqual(result["images"], 4)
            self.assertEqual(result["source_partitions"], ["VisDrone2019-DET-train"])
            self.assertFalse(result["publisher_validation_used"])
            self.assertFalse(result["publisher_test_used"])
            self.assertEqual(result["natural_groups"], 3)
            self.assertEqual(result["detector_train_groups"], 2)
            self.assertEqual(result["internal_validation_groups"], 1)
            manifest_rows = [
                json.loads(line)
                for line in (output / "CONVERSION_MANIFEST.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                {row["publisher_component"] for row in manifest_rows},
                {"VisDrone2019-DET-train"},
            )
            self.assertFalse(any("DET-val" in row["source_image"] for row in manifest_rows))
            self.assertFalse(any("test-dev" in row["source_image"] for row in manifest_rows))
            calibration_ids = set(
                study_frame.loc[
                    study_frame["split"].eq("calibration"), "sample_id"
                ].astype(str)
            )
            converted_ids = {row["study_source_id"] for row in manifest_rows}
            self.assertFalse(converted_ids.intersection(calibration_ids))
            by_group = {}
            for row in manifest_rows:
                by_group.setdefault(row["natural_group"], set()).add(row["partition"])
                self.assertTrue((output / row["generated_image"]).is_symlink())
            self.assertTrue(all(len(partitions) == 1 for partitions in by_group.values()))
            self.assertEqual(
                {row["partition"] for row in manifest_rows},
                {"detector_train", "internal_val"},
            )
            category_one = next(
                row for row in manifest_rows if row["source_image"].endswith("000001_00001_d_0000001.jpg")
            )
            converted = (output / category_one["generated_label"]).read_text().splitlines()
            self.assertEqual(len(converted), 1)
            self.assertTrue(converted[0].startswith("0 "))
            self.assertEqual(category_one["publisher_rows_ignored"], 2)
            config = yaml.safe_load((output / "visdrone_det.yaml").read_text())
            self.assertEqual(len(config["names"]), 10)
            self.assertEqual(config["train"], "images/detector_train")
            self.assertEqual(config["val"], "images/internal_val")
            source_audit = json.loads(
                (output / "DETECTOR_LEAKAGE_AUDIT.json").read_text()
            )
            self.assertTrue(source_audit["passed"])
            self.assertTrue(source_audit["zero_forbidden_source_overlap"])
            self.assertEqual(
                source_audit["allowed_source_ids"],
                {
                    "count": len(converted_ids),
                    "sha256": hash_object(sorted(converted_ids)),
                },
            )
            self.assertEqual(
                source_audit["study_split_source_ids"]["calibration"]["count"],
                1,
            )
            self.assertEqual(
                source_audit["source_id_overlap"]["calibration"]["count"],
                0,
            )
            verified = verify_ultralytics_visdrone_det(output)
            self.assertEqual(verified["status"], "VERIFIED")
            tampered_label = output / manifest_rows[0]["generated_label"]
            tampered_label.write_text("tampered\n")
            with self.assertRaisesRegex(StudyGateError, "label hash changed"):
                verify_ultralytics_visdrone_det(output)

    def test_detector_split_rejects_one_group_and_verifier_recomputes_assignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "one-group"
            for index in (1, 2):
                self._write_component(
                    source,
                    "VisDrone2019-DET-train",
                    f"000001_0000{index}_d_000000{index}",
                    "10,10,20,10,1,4,0,0\n",
                )
            self._write_component(
                source,
                "VisDrone2019-DET-train",
                "000004_00001_d_0000001",
                "10,10,20,10,1,4,0,0\n",
            )
            one_manifest, _ = self._write_study_manifest(
                source, root / "ONE_GROUP_STUDY_SPLIT.csv"
            )
            with self.assertRaisesRegex(StudyGateError, "at least two"):
                prepare_ultralytics_visdrone_det(
                    source,
                    root / "one-output",
                    study_source_manifest=one_manifest,
                )

            source = root / "many-groups"
            self._write_detector_fixture(source)
            many_manifest, _ = self._write_study_manifest(
                source, root / "MANY_GROUP_STUDY_SPLIT.csv"
            )
            output = root / "many-output"
            prepare_ultralytics_visdrone_det(
                source,
                output,
                study_source_manifest=many_manifest,
                seed=23,
                internal_validation_fraction=0.25,
            )
            manifest_path = output / "CONVERSION_MANIFEST.jsonl"
            records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
            target_group = records[0]["natural_group"]
            replacement = (
                "internal_val"
                if records[0]["partition"] == "detector_train"
                else "detector_train"
            )
            for row in records:
                if row["natural_group"] == target_group:
                    row["partition"] = replacement
            manifest_path.write_text(
                "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in records)
            )
            summary_path = output / "CONVERSION_SUMMARY.json"
            summary = json.loads(summary_path.read_text())
            summary["manifest_sha256"] = hash_file(manifest_path)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            with self.assertRaisesRegex(StudyGateError, "Generated detector paths disagree|assignments do not match"):
                verify_ultralytics_visdrone_det(output)

    def test_verifier_rejects_publisher_validation_annotation_substitution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "VisDrone"
            self._write_detector_fixture(source)
            study_manifest, _ = self._write_study_manifest(
                source, root / "FROZEN_DEVELOPMENT_SPLIT.csv"
            )
            output = root / "prepared"
            prepare_ultralytics_visdrone_det(
                source,
                output,
                study_source_manifest=study_manifest,
                seed=17,
                internal_validation_fraction=0.25,
            )
            manifest_path = output / "CONVERSION_MANIFEST.jsonl"
            records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
            publisher_validation = (
                source
                / "VisDrone2019-DET-val"
                / "annotations"
                / "publisher_val_secret.txt"
            ).resolve()
            records[0]["source_annotation"] = str(publisher_validation)
            records[0]["source_annotation_sha256"] = hash_file(publisher_validation)
            manifest_path.write_text(
                "".join(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    for row in records
                )
            )
            summary_path = output / "CONVERSION_SUMMARY.json"
            summary = json.loads(summary_path.read_text())
            summary["manifest_sha256"] = hash_file(manifest_path)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            with self.assertRaisesRegex(
                StudyGateError,
                "source paths disagree|publisher validation/test labels",
            ):
                verify_ultralytics_visdrone_det(output)

    def test_manifest_rejects_cross_split_aliases_groups_and_component_spoofing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "VisDrone"
            self._write_detector_fixture(source)
            original = self._study_manifest_frame(source)
            train_index = original.index[
                original["component"].eq("det_train")
                & original["split"].eq("train")
            ][0]
            calibration_index = original.index[
                original["component"].eq("det_train")
                & original["split"].eq("calibration")
            ][0]
            validation_index = original.index[
                original["component"].eq("det_val")
            ][0]

            cases = []
            media_alias = original.copy()
            media_alias.loc[calibration_index, "media_path"] = media_alias.loc[
                train_index, "media_path"
            ]
            cases.append(("media-alias", media_alias, "aliases the same media"))

            group_crossing = original.copy()
            group_crossing.loc[calibration_index, "split_unit"] = group_crossing.loc[
                train_index, "split_unit"
            ]
            cases.append(("group-crossing", group_crossing, "crossing study splits"))

            component_spoof = original.copy()
            component_spoof.loc[validation_index, "split"] = "train"
            cases.append(
                (
                    "component-spoof",
                    component_spoof,
                    "component/partition metadata is inconsistent",
                )
            )

            for name, frame, error in cases:
                with self.subTest(name=name):
                    manifest = root / f"{name}.csv"
                    frame.to_csv(manifest, index=False)
                    with self.assertRaisesRegex(StudyGateError, error):
                        prepare_ultralytics_visdrone_det(
                            source,
                            root / f"output-{name}",
                            study_source_manifest=manifest,
                        )

    def test_allowed_manifest_row_cannot_point_outside_publisher_det_train(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "VisDrone"
            self._write_detector_fixture(source)
            frame = self._study_manifest_frame(source)
            train_index = frame.index[
                frame["component"].eq("det_train")
                & frame["split"].eq("train")
            ][0]
            validation_index = frame.index[frame["component"].eq("det_val")][0]
            frame.loc[train_index, "media_path"] = frame.loc[
                validation_index, "media_path"
            ]
            frame.loc[train_index, "annotation_path"] = frame.loc[
                validation_index, "annotation_path"
            ]
            frame = frame.drop(index=validation_index)
            manifest = root / "PATH_SPOOF.csv"
            frame.to_csv(manifest, index=False)
            with self.assertRaisesRegex(StudyGateError, "not an exact publisher DET-train"):
                prepare_ultralytics_visdrone_det(
                    source,
                    root / "prepared",
                    study_source_manifest=manifest,
                )

    def test_explicit_dataframe_allowlist_is_frozen_and_verifiable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "VisDrone"
            self._write_detector_fixture(source)
            frame = self._study_manifest_frame(source)
            output = root / "prepared"
            prepare_ultralytics_visdrone_det(
                source,
                output,
                study_source_manifest=frame,
                seed=17,
                internal_validation_fraction=0.25,
            )
            audit = json.loads((output / "DETECTOR_LEAKAGE_AUDIT.json").read_text())
            self.assertEqual(audit["input_manifest"]["kind"], "explicit_dataframe")
            self.assertEqual(audit["input_manifest"]["path"], "")
            self.assertEqual(audit["input_manifest"]["sha256"], "")
            self.assertEqual(verify_ultralytics_visdrone_det(output)["status"], "VERIFIED")

    def test_verifier_rejects_changed_original_frozen_source_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "VisDrone"
            self._write_detector_fixture(source)
            study_manifest, _ = self._write_study_manifest(
                source, root / "FROZEN_DEVELOPMENT_SPLIT.csv"
            )
            output = root / "prepared"
            prepare_ultralytics_visdrone_det(
                source,
                output,
                study_source_manifest=study_manifest,
                seed=17,
                internal_validation_fraction=0.25,
            )
            study_manifest.write_text(
                study_manifest.read_text() + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                StudyGateError, "Original frozen detector study manifest changed"
            ):
                verify_ultralytics_visdrone_det(output)

    def test_verifier_recomputes_zero_overlap_after_calibration_id_injection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "VisDrone"
            self._write_detector_fixture(source)
            study_manifest, study_frame = self._write_study_manifest(
                source, root / "FROZEN_DEVELOPMENT_SPLIT.csv"
            )
            output = root / "prepared"
            prepare_ultralytics_visdrone_det(
                source,
                output,
                study_source_manifest=study_manifest,
                seed=17,
                internal_validation_fraction=0.25,
            )
            calibration = study_frame.loc[
                study_frame["split"].eq("calibration")
            ].iloc[0]
            manifest_path = output / "CONVERSION_MANIFEST.jsonl"
            records = [
                json.loads(line) for line in manifest_path.read_text().splitlines()
            ]
            records[0]["study_source_id"] = str(calibration["sample_id"])
            records[0]["study_source_uid"] = str(calibration["source_uid"])
            records[0]["study_split"] = "calibration"
            records[0]["study_split_unit"] = str(calibration["split_unit"])
            manifest_path.write_text(
                "".join(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    for row in records
                )
            )
            summary_path = output / "CONVERSION_SUMMARY.json"
            summary = json.loads(summary_path.read_text())
            summary["manifest_sha256"] = hash_file(manifest_path)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            with self.assertRaisesRegex(
                StudyGateError,
                "leakage audit disagrees|overlap calibration",
            ):
                verify_ultralytics_visdrone_det(output)

    def test_training_helper_only_plans_and_checkpoint_hash_must_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "VisDrone"
            self._write_detector_fixture(source)
            study_manifest, _ = self._write_study_manifest(
                source, root / "FROZEN_DEVELOPMENT_SPLIT.csv"
            )
            prepared = prepare_ultralytics_visdrone_det(
                source,
                root / "prepared",
                study_source_manifest=study_manifest,
                seed=17,
                internal_validation_fraction=0.25,
            )
            base_model = root / "yolo11n.pt"
            base_model.write_bytes(b"prefetched-base-model")
            plan = detector_training_command_plan(
                prepared["dataset_yaml"],
                root / "runs",
                base_model=base_model,
                epochs=2,
                batch=1,
                seed=17,
            )
            self.assertEqual(plan["argv"][:3], ["yolo", "detect", "train"])
            self.assertEqual(plan["schema_version"], DETECTOR_TRAINING_PLAN_SCHEMA)
            self.assertEqual(plan["base_model_sha256"], hash_file(base_model))
            self.assertEqual(plan["argv"][3], f"model={base_model.resolve()}")
            self.assertFalse(plan["untouched_test_used"])
            self.assertFalse(plan["publisher_validation_used"])
            self.assertTrue(plan["zero_forbidden_source_overlap"])
            self.assertEqual(plan["allowed_source_id_count"], 4)
            self.assertTrue(Path(plan["study_source_manifest"]).is_file())
            self.assertTrue(Path(plan["detector_leakage_audit"]).is_file())
            self.assertEqual(plan["publisher_partitions"], ["VisDrone2019-DET-train"])
            self.assertEqual(plan["optimization_partitions"], ["detector_train", "internal_val"])
            self.assertFalse(Path(plan["expected_checkpoint"]).exists())
            with self.assertRaisesRegex(StudyGateError, "seed differs"):
                detector_training_command_plan(
                    prepared["dataset_yaml"], root / "wrong-seed",
                    base_model=base_model, epochs=2, batch=1, seed=18,
                )

            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"frozen-detector")
            expected = hash_file(checkpoint)
            record = verify_detector_checkpoint(checkpoint, expected)
            self.assertEqual(record["status"], "VERIFIED")
            with self.assertRaisesRegex(StudyGateError, "mismatch"):
                verify_detector_checkpoint(checkpoint, "0" * 64)

    def test_completed_training_receipt_binds_plan_environment_and_yolo_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "VisDrone"
            self._write_detector_fixture(source)
            study_manifest, _ = self._write_study_manifest(
                source, root / "FROZEN_DEVELOPMENT_SPLIT.csv"
            )
            prepared = prepare_ultralytics_visdrone_det(
                source,
                root / "prepared",
                study_source_manifest=study_manifest,
                seed=17,
                internal_validation_fraction=0.25,
            )
            base_model = root / "yolo11n.pt"
            base_model.write_bytes(b"prefetched-base-model")
            plan = detector_training_command_plan(
                prepared["dataset_yaml"], root / "runs",
                base_model=base_model, epochs=2, image_size=768,
                batch=1, workers=1, device="0", seed=17,
            )
            plan_path = root / "DETECTOR_TRAINING_PLAN.json"
            plan_path.write_text(
                json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8",
            )
            checkpoint = Path(plan["expected_checkpoint"])
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"trained-best-checkpoint")
            arguments = {
                token.split("=", 1)[0]: token.split("=", 1)[1]
                for token in plan["argv"][3:]
            }
            arguments.update({"task": "detect", "mode": "train"})
            (checkpoint.parent.parent / "args.yaml").write_text(
                yaml.safe_dump(arguments, sort_keys=True), encoding="utf-8",
            )
            (checkpoint.parent.parent / "results.csv").write_text(
                "epoch,metrics/mAP50(B)\n1,0.1\n2,0.2\n", encoding="utf-8",
            )
            environment = root / "ENVIRONMENT_FINGERPRINT.json"
            environment.write_text(
                json.dumps({
                    "schema_version": "semuav.environment.v2",
                    "python": {"path": "/registered/python", "sha256": "a" * 64},
                    "ffmpeg": {"path": "/registered/ffmpeg", "sha256": "b" * 64},
                }, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            receipt_path = root / "DETECTOR_TRAINING_RECEIPT.json"
            receipt = write_detector_training_receipt(
                plan_path,
                checkpoint=checkpoint,
                environment_fingerprint_path=environment,
                output_path=receipt_path,
            )
            self.assertEqual(
                receipt["schema_version"], DETECTOR_TRAINING_RECEIPT_SCHEMA,
            )
            self.assertEqual(
                receipt["run_artifacts"]["results_csv"]["completed_epochs"], 2,
            )
            verified = verify_detector_training_receipt(
                receipt_path,
                checkpoint=checkpoint,
                expected_checkpoint_sha256=hash_file(checkpoint),
                training_plan_path=plan_path,
                environment_fingerprint_path=environment,
            )
            self.assertEqual(verified["status"], "VERIFIED")
            checkpoint_record = verify_detector_checkpoint(
                checkpoint, hash_file(checkpoint), training_receipt=receipt_path,
            )
            self.assertEqual(
                checkpoint_record["training_receipt_sha256"], hash_file(receipt_path),
            )
            with self.assertRaisesRegex(StudyGateError, "already exists"):
                write_detector_training_receipt(
                    plan_path,
                    checkpoint=checkpoint,
                    environment_fingerprint_path=environment,
                    output_path=receipt_path,
                )
            invalid_epoch_sequences = {
                "zero_based": [0, 1],
                "negative": [-1, 2],
                "missing_final": [1],
                "gap": [1, 3],
                "duplicate": [1, 1],
                "out_of_order": [2, 1],
                "fractional": [1, 1.5],
                "precision_rounding": [1, "1.99999999999999999"],
                "non_numeric": [1, "invalid"],
                "infinite": [1, "inf"],
                "extra": [1, 2, 3],
            }
            results_path = checkpoint.parent.parent / "results.csv"
            for label, sequence in invalid_epoch_sequences.items():
                with self.subTest(epoch_sequence=label):
                    rows = "\n".join(f"{epoch},0.9" for epoch in sequence)
                    results_path.write_text(
                        f"epoch,metrics/mAP50(B)\n{rows}\n", encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        StudyGateError, "complete every registered epoch",
                    ):
                        verify_detector_training_receipt(receipt_path)
            results_path.write_text(
                "epoch,epoch,metrics/mAP50(B)\n1,1,0.1\n2,2,0.2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(StudyGateError, "duplicate columns"):
                verify_detector_training_receipt(receipt_path)


class RuntimeSmokeReportTests(FrozenModelAssetTests):
    def test_cleanup_failure_preserves_primary_error_and_writes_complete_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path, payload, snapshots = self._freeze(root)
            image = root / "frame.jpg"
            Image.new("RGB", (16, 12), (1, 2, 3)).save(image)
            request = SimpleNamespace(
                observation_id="visdrone:det_train:cleanup-smoke",
                image_path=str(image),
                mission="visdrone_urban",
                capture_time_us=123,
                telemetry={"battery_fraction": 0.9},
                detector_proposals=(),
            )
            factory_calls = []

            class FailingBackend:
                def infer(self, request, prompt, max_new_tokens):
                    raise RuntimeError("primary inference failure")

            def factory(entry, device, dtype):
                factory_calls.append(str(entry["model_id"]))
                return FailingBackend()

            def fail_empty_cache():
                raise RuntimeError("deliberate CUDA cleanup failure")

            fake_torch = SimpleNamespace(
                cuda=SimpleNamespace(
                    is_available=lambda: True,
                    empty_cache=fail_empty_cache,
                )
            )
            report_path = root / "SMOKE.json"
            with patch.dict("os.environ", {}, clear=True), patch.dict(
                "sys.modules", {"torch": fake_torch}
            ):
                frame = run_one_image_runtime_smoke(
                    lock_path,
                    request,
                    output_path=report_path,
                    prompt="FROZEN VISDRONE PROMPT",
                    backend_factory=factory,
                    offline_checker=(
                        lambda repo, revision, cache: snapshots[(repo, revision)]
                    ),
                )

            self.assertEqual(len(factory_calls), 1)
            self.assertEqual(len(frame), len(payload["models"]))
            self.assertEqual(frame["model_id"].nunique(), len(payload["models"]))
            self.assertTrue(frame["status"].eq("ERROR").all())
            first_error = str(frame.iloc[0]["error"])
            self.assertIn("RuntimeError: primary inference failure", first_error)
            self.assertIn(
                "runtime cleanup failed: RuntimeError: deliberate CUDA cleanup failure",
                first_error,
            )
            remaining_errors = frame.iloc[1:]["error"].astype(str)
            self.assertTrue(remaining_errors.str.startswith(
                "Not run after unrecoverable runtime cleanup failure: "
            ).all())
            self.assertTrue(remaining_errors.str.contains(
                "RuntimeError: deliberate CUDA cleanup failure",
                regex=False,
            ).all())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(len(report["models"]), len(payload["models"]))
            self.assertFalse(report["all_passed"])
            with self.assertRaisesRegex(StudyGateError, "do not begin"):
                require_runtime_smoke_pass(report_path)

    def test_smoke_report_records_schema_and_runtime_failures_without_omission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path, payload, snapshots = self._freeze(root)
            image = root / "frame.jpg"
            Image.new("RGB", (16, 12), (1, 2, 3)).save(image)
            request = SimpleNamespace(
                observation_id="visdrone:det_train:smoke",
                image_path=str(image),
                mission="visdrone_urban",
                capture_time_us=123,
                telemetry={"battery_fraction": 0.9},
                detector_proposals=({
                    "label": "car",
                    "bbox_normalized_xyxy": [0.1, 0.2, 0.3, 0.4],
                    "confidence": 0.8,
                },),
            )
            with Image.open(image) as opened_image:
                source_image = opened_image.convert("RGB")
            _, preprocess = aspect_preserving_letterbox(source_image, (768, 768))
            decision_context_sha256 = hash_object(
                visdrone_proposal_decision_context(
                    request,
                    source_image.size,
                    (768, 768),
                    preprocess=preprocess,
                )
            )

            class FakeBackend:
                def __init__(self, entry):
                    self.entry = entry

                def infer(self, request, prompt, max_new_tokens):
                    self.assert_prompt(prompt)
                    model_id = self.entry["model_id"]
                    if model_id == "smolvlm2_2b":
                        raise RuntimeError("deliberate runtime failure")
                    raw_text = (
                        '{"accepted_proposal_ids":[0],'
                        '"limited_visibility_proposal_ids":[],'
                        '"sufficient_evidence":true}'
                    )
                    strategy = VISDRONE_GENERATION_STRATEGY_BY_MODEL[model_id]
                    prefill = visdrone_assistant_prefill_for_model(model_id)
                    continuation = raw_text[len(prefill):]
                    return SimpleNamespace(
                        observation_id=request.observation_id,
                        model_id=model_id,
                        model_revision=self.entry["resolved_revision"],
                        schema_valid=model_id != "qwen3_vl_2b",
                        error=(
                            "ValueError: Model response is not one exact JSON object"
                            if model_id == "qwen3_vl_2b" else None
                        ),
                        latency_ms=5.0,
                        raw_text=raw_text,
                        parsed={"schema_version": "semuav.v1"},
                        input_tokens=100,
                        output_tokens=20,
                        prompt_hash=hash_object(prompt),
                        decision_contract_version=(
                            "semuav.visdrone-proposal-decision.v2"
                        ),
                        decision_core_valid=True,
                        decision_core_json=raw_text,
                        decision_context_sha256=decision_context_sha256,
                        decision_proposal_count=1,
                        decision_composer_policy_sha256=(
                            VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256
                        ),
                        terminated_by_eos=True,
                        reached_token_cap=False,
                        generation_policy_version=(
                            VISDRONE_GENERATION_POLICY_VERSION
                        ),
                        generation_policy_sha256=(
                            visdrone_generation_policy_sha256(max_new_tokens)
                        ),
                        generation_strategy=strategy,
                        generation_max_new_tokens=max_new_tokens,
                        assistant_prefill_sha256=sha256_bytes(
                            prefill.encode("utf-8")
                        ),
                        assistant_prefill_characters=len(prefill),
                        generated_continuation_sha256=sha256_bytes(
                            continuation.encode("utf-8")
                        ),
                    )

                @staticmethod
                def assert_prompt(prompt):
                    if prompt != "FROZEN VISDRONE PROMPT":
                        raise AssertionError("runtime smoke did not forward the explicit prompt")

            frame = run_one_image_runtime_smoke(
                lock_path,
                request,
                output_path=root / "SMOKE.json",
                prompt="FROZEN VISDRONE PROMPT",
                backend_factory=lambda entry, device, dtype: FakeBackend(entry),
                offline_checker=lambda repo, revision, cache: snapshots[(repo, revision)],
            )
            self.assertEqual(len(frame), 3)
            self.assertEqual(frame["model_id"].nunique(), 3)
            self.assertEqual(
                frame["model_id"].tolist(),
                [
                    "qwen3_vl_2b",
                    "smolvlm2_2b",
                    "qwen3_vl_8b_upper_bound",
                ],
            )
            statuses = frame.set_index("model_id")["status"].to_dict()
            self.assertEqual(statuses["smolvlm2_2b"], "ERROR")
            self.assertEqual(statuses["qwen3_vl_2b"], "FAIL_SCHEMA")
            self.assertEqual(sum(value == "PASS" for value in statuses.values()), 1)
            qwen_error = frame.set_index("model_id").loc["qwen3_vl_2b", "error"]
            self.assertEqual(
                qwen_error,
                "ValueError: Model response is not one exact JSON object",
            )
            report = json.loads((root / "SMOKE.json").read_text())
            self.assertEqual(len(report["models"]), 3)
            self.assertFalse(report["all_passed"])
            self.assertEqual(report["model_lock_sha256"], hash_file(lock_path))
            self.assertRegex(report["code_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(report["composer_policy_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                report["input_contract"],
                "semuav.visdrone-proposal-decision.v2",
            )
            self.assertEqual(report["detector_proposal_count"], 1)
            self.assertEqual(
                report["detector_proposals_sha256"],
                hash_object(list(request.detector_proposals)),
            )
            expected_request_sha256 = hash_object({
                "observation_id": request.observation_id,
                "mission": request.mission,
                "capture_time_us": request.capture_time_us,
                "telemetry": dict(request.telemetry),
                "image_sha256": hash_file(image),
                "detector_proposals": list(request.detector_proposals),
            })
            self.assertEqual(report["request_sha256"], expected_request_sha256)
            self.assertTrue(
                frame["request_sha256"].eq(expected_request_sha256).all()
            )
            qwen_row = frame.set_index("model_id").loc["qwen3_vl_2b"]
            self.assertEqual(qwen_row["raw_text"], (
                '{"accepted_proposal_ids":[0],'
                '"limited_visibility_proposal_ids":[],'
                '"sufficient_evidence":true}'
            ))
            self.assertRegex(str(qwen_row["raw_text_sha256"]), r"^[0-9a-f]{64}$")
            self.assertRegex(
                str(qwen_row["parsed_packet_sha256"]), r"^[0-9a-f]{64}$"
            )
            with self.assertRaisesRegex(StudyGateError, "do not begin"):
                require_runtime_smoke_pass(frame)


if __name__ == "__main__":
    unittest.main()
