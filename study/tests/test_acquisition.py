import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import semuav.acquisition as acquisition_module

from semuav.acquisition import (
    AcquisitionError,
    acquire_datasets,
    dataset_source_table,
    dataset_status,
    load_dataset_registry,
    main,
    prepare_dataset_manifests,
    prepare_flame3_manifest,
    prepare_uavreason_manifests,
    safe_extract_zip,
)


def _visdrone_fixture_members(component, *, wrapped=True):
    directory = Path(component["archive"]).stem
    prefix = f"{directory}/" if wrapped else ""
    if str(component["name"]).startswith("det_"):
        media = "images/sample.jpg"
        annotation = "annotations/sample.txt"
    else:
        media = "sequences/sequence-01/0000001.jpg"
        annotation = "annotations/sequence-01.txt"
    return {
        f"{prefix}{media}": b"image",
        f"{prefix}{annotation}": b"label",
    }


def _write_visdrone_fixture_archive(path, component, *, wrapped=True):
    with zipfile.ZipFile(path, "w") as handle:
        for relative, payload in _visdrone_fixture_members(
            component, wrapped=wrapped,
        ).items():
            handle.writestr(relative, payload)


class DatasetAcquisitionTests(unittest.TestCase):
    def test_registry_has_exact_proposal_sources(self):
        registry = load_dataset_registry()
        self.assertEqual(set(registry["datasets"]), {"visdrone", "uavid", "uavreason", "flame3"})
        table = dataset_source_table()
        self.assertEqual(len(table), 4)
        self.assertTrue(table["official_url"].str.startswith("https://").all())

    def test_terms_gate_precedes_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(AcquisitionError, "terms"):
                acquire_datasets(temporary, ["visdrone"], accept_terms=False)

    def test_zip_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../escape.txt", "bad")
            with self.assertRaisesRegex(AcquisitionError, "Unsafe"):
                safe_extract_zip(archive, Path(temporary) / "out")

    def test_uavreason_normalization_separates_temporal_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotation_dir = root / "annotations"
            annotation_dir.mkdir()
            image_dir = root / "UAVScenes" / "scene1"
            image_dir.mkdir(parents=True)
            (image_dir / "a.jpg").write_bytes(b"a")
            (image_dir / "b.jpg").write_bytes(b"b")
            records = [
                {
                    "image": ["UAVScenes/scene1/a.jpg"],
                    "conversations": [{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}],
                    "meta": {"scene": "scene1", "task": "single"},
                },
                {
                    "image": ["UAVScenes/scene1/a.jpg", "UAVScenes/scene1/b.jpg"],
                    "conversations": [{"from": "human", "value": "Q2"}, {"from": "gpt", "value": "A2"}],
                    "meta": {"scene": "scene1", "task": "temporal"},
                },
            ]
            source = annotation_dir / "samples.jsonl"
            source.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
            report = prepare_uavreason_manifests(root)
            self.assertEqual(report["single_frame_rows"], 1)
            self.assertEqual(report["temporal_rows_excluded_from_single_image_pipeline"], 1)
            row = json.loads((root / "manifest.jsonl").read_text().strip())
            self.assertIsInstance(row["image_path"], str)
            self.assertNotIn("['", row["image_path"])

    def test_uavreason_missing_rgb_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotation_dir = root / "annotations"
            annotation_dir.mkdir()
            record = {
                "image": "UAVScenes/scene1/not_downloaded.jpg",
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ],
            }
            (annotation_dir / "missing.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(AcquisitionError, "missing UAVScenes RGB"):
                prepare_uavreason_manifests(root)
            self.assertFalse((root / "manifest.jsonl").exists())

    def test_uavreason_registry_pins_minimal_upstream_snapshot(self):
        hf = load_dataset_registry()["datasets"]["uavreason"]["huggingface"]
        self.assertRegex(hf["annotations_revision"], r"^[0-9a-f]{40}$")
        self.assertRegex(hf["images_revision"], r"^[0-9a-f]{40}$")
        self.assertIn("interval5_CAM_LIDAR.zip", hf["images_allow_patterns"])
        self.assertNotIn("UAVReason_depth", " ".join(hf["images_allow_patterns"]))

    def test_flame_manifest_never_invents_mission_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event = root / "sycan" / "capture"
            event.mkdir(parents=True)
            (event / "frame_001_rgb.jpg").write_bytes(b"rgb")
            (event / "frame_001_thermal.tiff").write_bytes(b"thermal")
            report = prepare_flame3_manifest(root)
            frame = pd.read_csv(root / "manifest.csv", keep_default_na=False)
            self.assertEqual(report["mission_labels_inferred"], False)
            self.assertEqual(frame.loc[0, "label"], "")
            self.assertEqual(frame.loc[0, "event_id"], "sycan")

    def test_flame_unpaired_rgb_is_routed_out_of_scientific_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event = root / "sycan" / "capture"
            event.mkdir(parents=True)
            (event / "frame_001_rgb.jpg").write_bytes(b"rgb")
            (event / "frame_001_thermal.tiff").write_bytes(b"thermal")
            (event / "frame_002_rgb.jpg").write_bytes(b"unpaired")
            report = prepare_flame3_manifest(root)
            scientific = pd.read_csv(root / "manifest.csv", keep_default_na=False)
            excluded = pd.read_csv(root / "unpaired_rgb_manifest.csv", keep_default_na=False)
            self.assertEqual(len(scientific), 1)
            self.assertTrue(scientific["has_ground_truth"].all())
            self.assertEqual(len(excluded), 1)
            self.assertFalse(excluded.loc[0, "has_ground_truth"])
            self.assertEqual(excluded.loc[0, "label"], "")
            self.assertIn("without fabricating", excluded.loc[0, "exclusion_reason"])
            self.assertFalse(report["unpaired_ground_truth_fabricated"])

    def test_prepare_cli_raises_when_any_dataset_is_blocked(self):
        blocked = pd.DataFrame([{"dataset": "flame3", "status": "blocked"}])
        with patch("semuav.acquisition.prepare_dataset_manifests", return_value=blocked):
            with self.assertRaisesRegex(AcquisitionError, "preparation blocked"):
                main(["--data-root", "/tmp/data", "--action", "prepare", "--datasets", "flame3"])

    def test_user_staged_full_flame_archive_records_confirmatory_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "flame3_full.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("sycan/frame_001_rgb.jpg", b"rgb")
                handle.writestr("sycan/frame_001_thermal.tiff", b"thermal")
            with (
                patch.dict("os.environ", {"SEMUAV_FLAME3_FULL_ARCHIVE": str(archive)}),
                patch(
                    "semuav.acquisition.disk_preflight",
                    return_value={
                        "estimated_download_gb": 0.001,
                        "recommended_free_gb": 0.002,
                        "available_free_gb": 100.0,
                    },
                ),
            ):
                result = acquire_datasets(root / "data", ["flame3"], accept_terms=True)
            self.assertEqual(result.iloc[0]["status"], "downloaded_full_confirmatory_scope")
            receipt = json.loads(
                (root / "data" / "FLAME3" / "SOURCE_RECEIPT.json").read_text(encoding="utf-8")
            )
            self.assertIn("full_confirmatory", receipt["source"]["scope"])
            self.assertTrue((root / "data" / "FLAME3" / "SOURCE_FILES_SHA256.jsonl").is_file())

    def test_partial_visdrone_receipt_is_upgraded_and_full_receipt_is_not_downgraded(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            registry = load_dataset_registry()["datasets"]["visdrone"]
            destination = data_root / registry["destination"]
            first = registry["components"][0]
            first_root = destination / Path(first["archive"]).stem
            (first_root / "images").mkdir(parents=True)
            (first_root / "annotations").mkdir()
            (first_root / "images" / "existing.jpg").write_bytes(b"image")
            (first_root / "annotations" / "existing.txt").write_text("label")
            acquisition_module._write_receipt(
                destination,
                "visdrone",
                {
                    "landing_page": registry["landing_page"],
                    "scope": "publisher_acquisition_official_release_components",
                },
                [{
                    "component": first["name"],
                    "google_drive_id": first["google_drive_id"],
                }],
            )
            download_dir = data_root / ".downloads" / "visdrone"
            download_dir.mkdir(parents=True)
            for component in registry["components"]:
                _write_visdrone_fixture_archive(
                    download_dir / component["archive"], component,
                )

            fake_gdown = SimpleNamespace(
                download=lambda **kwargs: self.fail("cached archives should not redownload")
            )
            with (
                patch.dict(sys.modules, {"gdown": fake_gdown}),
                patch.dict("os.environ", {"SEMUAV_VISDRONE_COMPONENTS": ""}),
                patch("semuav.acquisition.disk_preflight", return_value={
                    "estimated_download_gb": 1.0,
                    "recommended_free_gb": 2.0,
                    "available_free_gb": 100.0,
                }),
            ):
                result = acquire_datasets(data_root, ["visdrone"], accept_terms=True)
            self.assertEqual(result.loc[0, "status"], "downloaded")
            receipt = json.loads((destination / "SOURCE_RECEIPT.json").read_text())
            observed = {
                (item["component"], item["google_drive_id"])
                for item in receipt["artifacts"]
            }
            expected = {
                (item["name"], item["google_drive_id"])
                for item in registry["components"]
            }
            self.assertEqual(observed, expected)

            with (
                patch.dict("os.environ", {"SEMUAV_VISDRONE_COMPONENTS": "det_train"}),
                patch("semuav.acquisition.disk_preflight", return_value={
                    "estimated_download_gb": 1.0,
                    "recommended_free_gb": 2.0,
                    "available_free_gb": 100.0,
                }),
                patch("semuav.acquisition._acquire_visdrone") as acquire,
            ):
                subset = acquire_datasets(data_root, ["visdrone"], accept_terms=True)
            acquire.assert_not_called()
            self.assertEqual(subset.loc[0, "status"], "already_present")

    def test_rootless_visdrone_archive_is_normalized_under_component_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            registry = load_dataset_registry()["datasets"]["visdrone"]
            component = next(
                item for item in registry["components"]
                if item["name"] == "det_test_dev"
            )
            download_dir = data_root / ".downloads" / "visdrone"
            download_dir.mkdir(parents=True)
            _write_visdrone_fixture_archive(
                download_dir / component["archive"], component, wrapped=False,
            )
            fake_gdown = SimpleNamespace(
                download=lambda **kwargs: self.fail("cached archive should not redownload")
            )
            with (
                patch.dict(sys.modules, {"gdown": fake_gdown}),
                patch.dict(
                    "os.environ", {"SEMUAV_VISDRONE_COMPONENTS": "det_test_dev"},
                ),
                patch("semuav.acquisition.disk_preflight", return_value={
                    "estimated_download_gb": 1.0,
                    "recommended_free_gb": 2.0,
                    "available_free_gb": 100.0,
                }),
            ):
                result = acquire_datasets(data_root, ["visdrone"], accept_terms=True)

            destination = data_root / registry["destination"]
            component_root = destination / Path(component["archive"]).stem
            self.assertEqual(result.loc[0, "status"], "downloaded")
            self.assertTrue((component_root / "images" / "sample.jpg").is_file())
            self.assertTrue((component_root / "annotations" / "sample.txt").is_file())
            self.assertFalse((destination / "images").exists())
            self.assertFalse((destination / "annotations").exists())
            inventory = [
                json.loads(line)["path"]
                for line in (destination / "SOURCE_FILES_SHA256.jsonl").read_text().splitlines()
            ]
            self.assertTrue(inventory)
            self.assertTrue(
                all(path.startswith(f"{component_root.name}/") for path in inventory)
            )
            receipt = json.loads((destination / "SOURCE_RECEIPT.json").read_text())
            artifact = receipt["artifacts"][0]
            self.assertEqual(artifact["archive_layout"], "rootless")
            self.assertEqual(artifact["normalized_directory"], component_root.name)
            with (
                patch.dict(
                    "os.environ", {"SEMUAV_VISDRONE_COMPONENTS": "det_test_dev"},
                ),
                patch("semuav.acquisition.disk_preflight", return_value={
                    "estimated_download_gb": 1.0,
                    "recommended_free_gb": 2.0,
                    "available_free_gb": 100.0,
                }),
                patch("semuav.acquisition._acquire_visdrone") as acquire,
            ):
                repeated = acquire_datasets(
                    data_root, ["visdrone"], accept_terms=True,
                )
            acquire.assert_not_called()
            self.assertEqual(repeated.loc[0, "status"], "already_present")

    def test_wrapped_visdrone_archive_is_not_double_nested(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            registry = load_dataset_registry()["datasets"]["visdrone"]
            component = registry["components"][0]
            download_dir = data_root / ".downloads" / "visdrone"
            download_dir.mkdir(parents=True)
            _write_visdrone_fixture_archive(
                download_dir / component["archive"], component, wrapped=True,
            )
            fake_gdown = SimpleNamespace(
                download=lambda **kwargs: self.fail("cached archive should not redownload")
            )
            with (
                patch.dict(sys.modules, {"gdown": fake_gdown}),
                patch.dict("os.environ", {"SEMUAV_VISDRONE_COMPONENTS": component["name"]}),
                patch("semuav.acquisition.disk_preflight", return_value={
                    "estimated_download_gb": 1.0,
                    "recommended_free_gb": 2.0,
                    "available_free_gb": 100.0,
                }),
            ):
                acquire_datasets(data_root, ["visdrone"], accept_terms=True)

            destination = data_root / registry["destination"]
            component_root = destination / Path(component["archive"]).stem
            self.assertTrue((component_root / "images" / "sample.jpg").is_file())
            self.assertFalse((component_root / component_root.name).exists())

    def test_exact_legacy_rootless_visdrone_layout_is_migrated(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            registry = load_dataset_registry()["datasets"]["visdrone"]
            component = next(
                item for item in registry["components"]
                if item["name"] == "det_test_dev"
            )
            destination = data_root / registry["destination"]
            for relative, payload in _visdrone_fixture_members(
                component, wrapped=False,
            ).items():
                path = destination / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            acquisition_module._write_receipt(
                destination,
                "visdrone",
                {
                    "landing_page": registry["landing_page"],
                    "provider": registry["provider"],
                    "scope": "publisher_acquisition_official_release_components",
                    "license_or_terms": registry["license"],
                },
                [{
                    "component": component["name"],
                    "archive": str(
                        data_root / ".downloads" / "visdrone" / component["archive"]
                    ),
                    "google_drive_id": component["google_drive_id"],
                }],
            )
            download_dir = data_root / ".downloads" / "visdrone"
            download_dir.mkdir(parents=True)
            _write_visdrone_fixture_archive(
                download_dir / component["archive"], component, wrapped=False,
            )
            fake_gdown = SimpleNamespace(
                download=lambda **kwargs: self.fail("cached archive should not redownload")
            )
            with (
                patch.dict(sys.modules, {"gdown": fake_gdown}),
                patch.dict(
                    "os.environ", {"SEMUAV_VISDRONE_COMPONENTS": "det_test_dev"},
                ),
                patch("semuav.acquisition.disk_preflight", return_value={
                    "estimated_download_gb": 1.0,
                    "recommended_free_gb": 2.0,
                    "available_free_gb": 100.0,
                }),
            ):
                acquire_datasets(data_root, ["visdrone"], accept_terms=True)

            component_root = destination / Path(component["archive"]).stem
            self.assertTrue((component_root / "images" / "sample.jpg").is_file())
            self.assertFalse((destination / "images").exists())
            self.assertFalse((destination / "annotations").exists())

    def test_mismatched_legacy_rootless_layout_fails_without_deleting_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            registry = load_dataset_registry()["datasets"]["visdrone"]
            component = next(
                item for item in registry["components"]
                if item["name"] == "det_test_dev"
            )
            destination = data_root / registry["destination"]
            (destination / "images").mkdir(parents=True)
            (destination / "annotations").mkdir()
            (destination / "images" / "sample.jpg").write_bytes(b"different")
            (destination / "annotations" / "sample.txt").write_bytes(b"label")
            download_dir = data_root / ".downloads" / "visdrone"
            download_dir.mkdir(parents=True)
            _write_visdrone_fixture_archive(
                download_dir / component["archive"], component, wrapped=False,
            )
            fake_gdown = SimpleNamespace(
                download=lambda **kwargs: self.fail("cached archive should not redownload")
            )
            with (
                patch.dict(sys.modules, {"gdown": fake_gdown}),
                patch.dict(
                    "os.environ", {"SEMUAV_VISDRONE_COMPONENTS": "det_test_dev"},
                ),
                patch("semuav.acquisition.disk_preflight", return_value={
                    "estimated_download_gb": 1.0,
                    "recommended_free_gb": 2.0,
                    "available_free_gb": 100.0,
                }),
            ):
                with self.assertRaisesRegex(
                    AcquisitionError, "file size differs|content differs",
                ):
                    acquire_datasets(data_root, ["visdrone"], accept_terms=True)
            self.assertEqual(
                (destination / "images" / "sample.jpg").read_bytes(), b"different",
            )
            self.assertTrue((destination / "annotations" / "sample.txt").is_file())
            self.assertFalse(
                (destination / Path(component["archive"]).stem).exists()
            )

    def test_subset_repair_preserves_other_official_component_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            registry = load_dataset_registry()["datasets"]["visdrone"]
            test_component = next(
                item for item in registry["components"]
                if item["name"] == "det_test_dev"
            )
            download_dir = data_root / ".downloads" / "visdrone"
            download_dir.mkdir(parents=True)
            for component in registry["components"]:
                _write_visdrone_fixture_archive(
                    download_dir / component["archive"],
                    component,
                    wrapped=component["name"] != "det_test_dev",
                )
            fake_gdown = SimpleNamespace(
                download=lambda **kwargs: self.fail("cached archives should not redownload")
            )
            preflight = patch("semuav.acquisition.disk_preflight", return_value={
                "estimated_download_gb": 1.0,
                "recommended_free_gb": 2.0,
                "available_free_gb": 100.0,
            })
            with (
                patch.dict(sys.modules, {"gdown": fake_gdown}),
                patch.dict("os.environ", {"SEMUAV_VISDRONE_COMPONENTS": ""}),
                preflight,
            ):
                acquire_datasets(data_root, ["visdrone"], accept_terms=True)

            destination = data_root / registry["destination"]
            test_root = destination / Path(test_component["archive"]).stem
            os.replace(test_root / "images", destination / "images")
            os.replace(test_root / "annotations", destination / "annotations")
            test_root.rmdir()
            old_receipt = json.loads((destination / "SOURCE_RECEIPT.json").read_text())
            acquisition_module._write_receipt(
                destination,
                "visdrone",
                old_receipt["source"],
                old_receipt["artifacts"],
            )

            with (
                patch.dict(sys.modules, {"gdown": fake_gdown}),
                patch.dict(
                    "os.environ", {"SEMUAV_VISDRONE_COMPONENTS": "det_test_dev"},
                ),
                patch("semuav.acquisition.disk_preflight", return_value={
                    "estimated_download_gb": 1.0,
                    "recommended_free_gb": 2.0,
                    "available_free_gb": 100.0,
                }),
            ):
                acquire_datasets(data_root, ["visdrone"], accept_terms=True)

            repaired = json.loads((destination / "SOURCE_RECEIPT.json").read_text())
            self.assertEqual(
                {item["component"] for item in repaired["artifacts"]},
                {item["name"] for item in registry["components"]},
            )
            self.assertTrue(
                acquisition_module._visdrone_receipt_covers(
                    destination, registry["components"],
                )
            )

    def test_mixed_visdrone_archive_layout_fails_without_publishing_component(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            registry = load_dataset_registry()["datasets"]["visdrone"]
            component = registry["components"][0]
            directory = Path(component["archive"]).stem
            download_dir = data_root / ".downloads" / "visdrone"
            download_dir.mkdir(parents=True)
            with zipfile.ZipFile(download_dir / component["archive"], "w") as handle:
                handle.writestr(f"{directory}/images/sample.jpg", b"image")
                handle.writestr("annotations/sample.txt", b"label")
            fake_gdown = SimpleNamespace(
                download=lambda **kwargs: self.fail("cached archive should not redownload")
            )
            with (
                patch.dict(sys.modules, {"gdown": fake_gdown}),
                patch.dict("os.environ", {"SEMUAV_VISDRONE_COMPONENTS": component["name"]}),
                patch("semuav.acquisition.disk_preflight", return_value={
                    "estimated_download_gb": 1.0,
                    "recommended_free_gb": 2.0,
                    "available_free_gb": 100.0,
                }),
            ):
                with self.assertRaisesRegex(AcquisitionError, "mixes wrapped and rootless"):
                    acquire_datasets(data_root, ["visdrone"], accept_terms=True)
            self.assertFalse((data_root / registry["destination"] / directory).exists())

    def test_visdrone_receipt_does_not_cover_missing_physical_component(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = load_dataset_registry()["datasets"]["visdrone"]
            component = registry["components"][0]
            acquisition_module._write_receipt(
                root,
                "visdrone",
                {
                    "scope": "publisher_acquisition_official_release_components",
                },
                [{
                    "component": component["name"],
                    "archive": component["archive"],
                    "google_drive_id": component["google_drive_id"],
                }],
            )
            self.assertFalse(
                acquisition_module._visdrone_receipt_covers(root, [component])
            )

    def test_status_is_read_only_and_reports_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            status = dataset_status(temporary)
            self.assertEqual(len(status), 4)
            self.assertFalse(status["present"].any())
            self.assertIn("receipt_scope", status.columns)
            self.assertIn("resolved_source", status.columns)

    def test_status_rejects_legacy_rootless_visdrone_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            destination = data_root / "VisDrone"
            (destination / "images").mkdir(parents=True)
            (destination / "annotations").mkdir()
            (destination / "images" / "sample.jpg").write_bytes(b"image")
            (destination / "annotations" / "sample.txt").write_bytes(b"label")
            status = dataset_status(data_root)
            visdrone = status.loc[status["dataset"].eq("visdrone")].iloc[0]
            self.assertFalse(bool(visdrone["ingest_ready"]))
            self.assertIn("normalized component layout incomplete", visdrone["status_detail"])


if __name__ == "__main__":
    unittest.main()
