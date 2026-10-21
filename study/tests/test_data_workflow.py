from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from PIL import Image

import semuav.acquisition as acquisition_module
import semuav.data as data_module
from semuav.config import (
    AnnotationConfig,
    DatasetSpec,
    PathsConfig,
    Profile,
    SplitConfig,
    StudyConfig,
    load_study_config,
)
from semuav.data import (
    ANNOTATION_TASKS,
    TELEMETRY_COLUMNS,
    DataGateError,
    audit_manifest,
    audit_split_leakage,
    export_preparation_bundle,
    find_cross_split_content_duplicates,
)
from semuav.perception import (
    HFMultimodalBackend,
    InferenceRequest,
    ModelSpec,
    MoondreamBackend,
    run_detector_cache,
)


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _real_manifest(paths: list[Path]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "sample_id": f"sample-{index}",
            "dataset": "real",
            "mission": "urban",
            "media_path": str(path),
            "source_uid": path.name,
            "sequence_id": f"sequence-{index}",
            "split_group": f"real:sequence-{index}",
            "is_synthetic": False,
            "frame_index": 0,
        }
        for index, path in enumerate(paths)
    ])


class PrerequisiteStageTests(unittest.TestCase):
    def _full_config(self, root: Path) -> StudyConfig:
        base = load_study_config(ROOT / "configs" / "full.yaml")
        datasets = []
        for spec in base.datasets:
            dataset_root = root / spec.name
            dataset_root.mkdir(parents=True)
            datasets.append(dataclasses.replace(spec, root=dataset_root))
        return dataclasses.replace(
            base,
            datasets=tuple(datasets),
            paths=dataclasses.replace(
                base.paths,
                annotations=root / "missing-development.csv",
                test_annotations=root / "missing-test.csv",
                split_manifest=root / "missing-splits.csv.gz",
                channel_traces=root / "missing-traces.csv",
                telemetry_manifest=root / "missing-telemetry.csv",
            ),
        )

    @staticmethod
    def _write_provenance(config: StudyConfig, *, user_staged: str | None = None) -> None:
        registry = acquisition_module.load_dataset_registry()["datasets"]
        scopes = {
            "visdrone": "publisher_acquisition_official_release_components",
            "uavid": "publisher_archive_dans_v1_confirmatory",
            "uavreason": "publisher_acquisition_pinned_huggingface_snapshots",
            "flame3": "complete_ieee_dataport_full_confirmatory_user_staged",
        }
        for spec in config.datasets:
            (spec.root / "source.bin").write_bytes(spec.name.encode("utf-8"))
            adapter = spec.adapter.removeprefix("generic_")
            scope = scopes[adapter]
            if user_staged == spec.name:
                scope = "user_staged_publisher_layout_source_not_remotely_verified"
            source = {"landing_page": registry[adapter]["landing_page"], "scope": scope}
            artifacts: list[dict[str, str]] = []
            if adapter == "visdrone":
                artifacts = [
                    {
                        "component": str(item["name"]),
                        "google_drive_id": str(item["google_drive_id"]),
                    }
                    for item in registry[adapter]["components"]
                ]
            elif adapter == "uavid":
                source.update({
                    "persistent_id": registry[adapter]["dataverse"]["persistent_id"],
                    "version": registry[adapter]["dataverse"]["version"],
                })
            elif adapter == "uavreason":
                source.update({
                    field: registry[adapter]["huggingface"][field]
                    for field in (
                        "annotations_repo", "annotations_revision",
                        "images_repo", "images_revision",
                    )
                })
            elif adapter == "flame3":
                source["doi"] = registry[adapter]["doi"]
            acquisition_module._write_receipt(
                spec.root,
                adapter,
                source,
                artifacts,
            )

    def test_ingest_requires_only_dataset_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._full_config(Path(directory))
            self._write_provenance(config)
            flame = next(spec for spec in config.datasets if spec.adapter == "flame3")
            receipt_path = flame.root / "SOURCE_RECEIPT.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["source"]["scope"] = "official_kaggle_pilot_subsets_not_full_confirmatory"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

            self.assertEqual(config.prerequisite_errors("ingest"), [])
            validation = config.prerequisite_errors("validation")
            self.assertTrue(any("annotation table" in error for error in validation))
            self.assertTrue(any("channel traces" in error for error in validation))
            self.assertTrue(any("telemetry manifest" in error for error in validation))
            self.assertTrue(any("frozen split" in error for error in validation))
            self.assertTrue(any("Kaggle pilot" in error for error in validation))
            test = config.prerequisite_errors("test")
            self.assertTrue(any("missing-test.csv" in error for error in test))

    def test_full_confirmatory_flame_receipt_is_accepted_after_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._full_config(Path(directory))
            self._write_provenance(config)
            errors = config.prerequisite_errors("validation")
            provenance_terms = (
                "source receipt", "source file hash inventory", "inventoried source",
                "source scope", "unlisted file", "provenance",
            )
            self.assertFalse(any(
                any(term in error for term in provenance_terms) for error in errors
            ), errors)

    def test_confirmatory_phases_reject_user_staged_unverified_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._full_config(Path(directory))
            self._write_provenance(config, user_staged="visdrone")
            for stage in ("validation", "test", "embedded"):
                with self.subTest(stage=stage):
                    errors = config.prerequisite_errors(stage)
                    self.assertTrue(any(
                        "user-staged preparation receipts cannot certify" in error
                        for error in errors
                    ))

    def test_source_inventory_detects_same_size_mutation_and_unlisted_addition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._full_config(root)
            self._write_provenance(config)
            visdrone = next(spec for spec in config.datasets if spec.adapter == "visdrone")
            source = visdrone.root / "source.bin"
            source.write_bytes(b"X" * len(source.read_bytes()))
            mutated = config.prerequisite_errors("validation")
            self.assertTrue(any("content changed" in error for error in mutated))
            source.write_bytes(visdrone.name.encode("utf-8"))
            (visdrone.root / "unlisted.bin").write_bytes(b"extra")
            unlisted = config.prerequisite_errors("validation")
            self.assertTrue(any("unlisted file" in error for error in unlisted))

    def test_all_real_dataset_receipts_and_inventories_are_freeze_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._full_config(Path(directory))
            paths = config.dataset_provenance_paths()
            self.assertEqual(len(paths), 8)
            for spec in config.datasets:
                self.assertIn(f"dataset_{spec.name}_source_receipt", paths)
                self.assertIn(f"dataset_{spec.name}_source_file_hash_inventory", paths)

    def test_full_profile_rejects_a_confirmatory_scope_from_the_wrong_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._full_config(Path(directory))
            self._write_provenance(config)
            visdrone = next(spec for spec in config.datasets if spec.adapter == "visdrone")
            receipt_path = visdrone.root / "SOURCE_RECEIPT.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["source"]["scope"] = "complete_ieee_dataport_full_confirmatory_user_staged"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            errors = config.prerequisite_errors("validation")
            self.assertTrue(any("not confirmatory" in error for error in errors))

    def test_full_profile_rejects_partial_visdrone_release_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._full_config(Path(directory))
            self._write_provenance(config)
            visdrone = next(spec for spec in config.datasets if spec.adapter == "visdrone")
            receipt_path = visdrone.root / "SOURCE_RECEIPT.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["artifacts"] = receipt["artifacts"][:-1]
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            errors = config.prerequisite_errors("validation")
            self.assertTrue(any(
                "exactly all six registered release components" in error
                for error in errors
            ))

    def test_full_profile_rejects_mismatched_publisher_pins(self) -> None:
        mutations = (
            ("uavid", "persistent_id", "doi:10.0000/not-uavid"),
            ("uavid", "version", "2.0"),
            ("uavreason", "annotations_repo", "someone/other-repo"),
            ("uavreason", "images_revision", "0" * 40),
            ("flame3", "doi", "10.0000/not-flame"),
        )
        for adapter, field, replacement in mutations:
            with self.subTest(adapter=adapter, field=field), tempfile.TemporaryDirectory() as directory:
                config = self._full_config(Path(directory))
                self._write_provenance(config)
                spec = next(item for item in config.datasets if item.adapter == adapter)
                receipt_path = spec.root / "SOURCE_RECEIPT.json"
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                receipt["source"][field] = replacement
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                errors = config.prerequisite_errors("validation")
                self.assertTrue(any(field in error for error in errors), errors)


class PreparationBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        media = [
            _write(self.root / f"frame-{index}.jpg", bytes([index + 1]) * (index + 1))
            for index in range(4)
        ]
        self.manifest = _real_manifest(media)
        self.split = SplitConfig()
        self.annotations = AnnotationConfig(required_annotators=2)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_export_is_unfrozen_and_never_populates_labels_or_telemetry(self) -> None:
        destination = self.root / "preparation"
        outputs = export_preparation_bundle(
            self.manifest,
            destination,
            self.annotations,
            split_config=self.split,
        )
        self.assertEqual(set(outputs), {
            "canonical_manifest",
            "proposed_split_manifest",
            "annotation_assignment_skeleton",
            "telemetry_skeleton",
            "bundle_manifest",
        })
        self.assertTrue(all(path.is_file() for path in outputs.values()))

        split = pd.read_csv(outputs["proposed_split_manifest"])
        self.assertEqual(list(split.columns), ["sample_id", "split", "split_unit"])
        self.assertEqual(len(split), len(self.manifest))
        self.assertEqual(set(split["split"]), set(self.split.ratios))

        annotations = pd.read_csv(outputs["annotation_assignment_skeleton"], keep_default_na=False)
        self.assertEqual(
            len(annotations),
            len(self.manifest) * len(ANNOTATION_TASKS) * self.annotations.required_annotators,
        )
        self.assertEqual(set(annotations["task"]), set(ANNOTATION_TASKS))
        for column in (
            "label", "annotator_id", "blinded", "adjudicated_label",
            "adjudicator_id", "is_critical", "evidence_geometry_json", "notes",
        ):
            self.assertTrue(annotations[column].eq("").all(), column)

        telemetry = pd.read_csv(outputs["telemetry_skeleton"], keep_default_na=False)
        self.assertEqual(list(telemetry.columns), list(TELEMETRY_COLUMNS))
        self.assertTrue(telemetry[list(TELEMETRY_COLUMNS[1:])].eq("").all().all())
        receipt = json.loads(outputs["bundle_manifest"].read_text(encoding="utf-8"))
        self.assertFalse(receipt["split_is_frozen"])
        self.assertFalse(receipt["labels_or_telemetry_invented"])

        original_hash = hashlib.sha256(outputs["bundle_manifest"].read_bytes()).hexdigest()
        with self.assertRaises(DataGateError):
            export_preparation_bundle(
                self.manifest,
                destination,
                self.annotations,
                split_config=self.split,
            )
        self.assertEqual(
            hashlib.sha256(outputs["bundle_manifest"].read_bytes()).hexdigest(),
            original_hash,
        )

    def test_failed_export_never_publishes_a_partial_bundle(self) -> None:
        destination = self.root / "failed-preparation"
        original = data_module._atomic_csv
        calls = 0

        def fail_second(frame: pd.DataFrame, path: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected write failure")
            original(frame, path)

        with patch("semuav.data._atomic_csv", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "injected write failure"):
                export_preparation_bundle(
                    self.manifest,
                    destination,
                    self.annotations,
                    split_config=self.split,
                )
        self.assertFalse(destination.exists())
        self.assertFalse(list(self.root.glob(".failed-preparation.staging-*")))


class DataIntegrityAuditTests(unittest.TestCase):
    def _smoke_config(self, root: Path) -> StudyConfig:
        paths = PathsConfig(root, root, root / "output", root / "cache")
        dataset = DatasetSpec("real", "visdrone", root, "urban", synthetic=False)
        return StudyConfig(
            schema_version=1,
            study_id="audit",
            profile=Profile.SMOKE_REAL,
            seed=1,
            seeds=(1,),
            allow_synthetic=False,
            paths=paths,
            datasets=(dataset,),
            split=SplitConfig(),
            annotations=AnnotationConfig(enabled=False, require_manifest_coverage=False),
        )

    def test_real_profiles_reject_source_uid_fallback_as_a_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = _write(root / "frame.jpg", b"frame")
            manifest = _real_manifest([media]).drop(columns=["sequence_id"])
            report = audit_manifest(manifest, self._smoke_config(root))
            self.assertFalse(report.passed)
            self.assertTrue(any("meaningful natural grouping ID" in error for error in report.errors))

            manifest["split"] = "train"
            manifest["split_unit"] = "unit-1"
            split_report = audit_split_leakage(manifest, SplitConfig())
            self.assertTrue(any("meaningful natural grouping ID" in error for error in split_report.errors))

            synthetic = manifest.copy()
            synthetic["is_synthetic"] = True
            synthetic_report = audit_split_leakage(synthetic, SplitConfig())
            self.assertFalse(any("natural grouping ID" in error for error in synthetic_report.errors))

    def test_exact_content_copies_crossing_splits_are_rejected_lazily(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _write(root / "first.jpg", b"identical-content")
            second = _write(root / "second.jpg", b"identical-content")
            unique = _write(root / "different-size.jpg", b"x")
            manifest = _real_manifest([first, second, unique])
            manifest["split"] = ["train", "test", "validation"]
            manifest["split_unit"] = ["unit-1", "unit-2", "unit-3"]

            duplicates = find_cross_split_content_duplicates(manifest)
            self.assertEqual(len(duplicates), 1)
            self.assertEqual(duplicates.attrs["candidate_files_hashed"], 2)
            self.assertEqual(set(duplicates.iloc[0]["splits"]), {"train", "test"})
            report = audit_split_leakage(manifest, SplitConfig())
            self.assertTrue(any("exact media-content SHA-256" in error for error in report.errors))

            manifest.loc[manifest["sample_id"].eq("sample-1"), "split"] = "train"
            same_split = audit_split_leakage(manifest, SplitConfig())
            self.assertFalse(any("exact media-content SHA-256" in error for error in same_split.errors))


class DetectorTrackingBoundaryTests(unittest.TestCase):
    def test_persistence_is_scoped_to_nonblank_dataset_sequence_pairs(self) -> None:
        class RecordingDetector:
            def __init__(self) -> None:
                self.calls: list[tuple[str, bool]] = []

            def detect(self, image_path: str, persist: bool = False) -> list[dict[str, object]]:
                self.calls.append((image_path, persist))
                return []

        detector = RecordingDetector()
        manifest = pd.DataFrame([
            {"sample_id": "a1", "dataset": "a", "image_path": "a1.jpg", "sequence_id": "shared", "frame_index": 1},
            {"sample_id": "a2", "dataset": "a", "image_path": "a2.jpg", "sequence_id": "shared", "frame_index": 2},
            {"sample_id": "b1", "dataset": "b", "image_path": "b1.jpg", "sequence_id": "shared", "frame_index": 1},
            {"sample_id": "b2", "dataset": "b", "image_path": "b2.jpg", "sequence_id": "shared", "frame_index": 2},
            {"sample_id": "c1", "dataset": "c", "image_path": "c1.jpg", "sequence_id": "", "frame_index": 1},
            {"sample_id": "c2", "dataset": "c", "image_path": "c2.jpg", "sequence_id": "  ", "frame_index": 2},
        ])
        run_detector_cache(detector, manifest)
        self.assertEqual(detector.calls, [
            ("a1.jpg", False),
            ("a2.jpg", True),
            ("b1.jpg", False),
            ("b2.jpg", True),
            ("c1.jpg", False),
            ("c2.jpg", False),
        ])

    def test_persistence_resets_between_dataset_components_with_shared_sequence_ids(self) -> None:
        class RecordingDetector:
            def __init__(self) -> None:
                self.calls: list[tuple[str, bool]] = []

            def detect(self, image_path: str, persist: bool = False) -> list[dict[str, object]]:
                self.calls.append((image_path, persist))
                return []

        detector = RecordingDetector()
        manifest = pd.DataFrame([
            {
                "sample_id": "det-1", "dataset": "visdrone",
                "component": "VisDrone2019-DET-train", "image_path": "det-1.jpg",
                "sequence_id": "000001", "frame_index": 1,
            },
            {
                "sample_id": "det-2", "dataset": "visdrone",
                "component": "VisDrone2019-DET-train", "image_path": "det-2.jpg",
                "sequence_id": "000001", "frame_index": 2,
            },
            {
                "sample_id": "mot-1", "dataset": "visdrone",
                "component": "VisDrone2019-MOT-train", "image_path": "mot-1.jpg",
                "sequence_id": "000001", "frame_index": 1,
            },
            {
                "sample_id": "mot-2", "dataset": "visdrone",
                "component": "VisDrone2019-MOT-train", "image_path": "mot-2.jpg",
                "sequence_id": "000001", "frame_index": 2,
            },
        ])
        run_detector_cache(detector, manifest)
        self.assertEqual(detector.calls, [
            ("det-1.jpg", False),
            ("det-2.jpg", True),
            ("mot-1.jpg", False),
            ("mot-2.jpg", True),
        ])


class OfflineModelLoadingTests(unittest.TestCase):
    def test_moondream_query_freezes_zero_temperature(self) -> None:
        settings_seen: list[dict[str, object]] = []

        class Model:
            def __init__(self):
                self.model = types.SimpleNamespace(
                    tokenizer=types.SimpleNamespace(
                        encode=lambda question: types.SimpleNamespace(ids=[10] * 100),
                    ),
                    config=types.SimpleNamespace(
                        text=types.SimpleNamespace(max_context=2048, vocab_size=51200),
                        tokenizer=types.SimpleNamespace(templates={
                            "query": {"prefix": [1, 15381, 2], "suffix": [3]},
                        }),
                    ),
                )

            def encode_image(self, image):
                del image
                return types.SimpleNamespace(pos=730)

            def query(self, *, image, question, settings):
                del image, question
                settings_seen.append(dict(settings))
                return {"answer": '{"e":true,"a":[],"l":[]}'}

        class InferenceMode:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, traceback):
                del exc_type, exc, traceback
                return False

        class Cuda:
            @staticmethod
            def reset_peak_memory_stats(device):
                del device

            @staticmethod
            def synchronize(device):
                del device

            @staticmethod
            def max_memory_allocated(device):
                del device
                return 0

        fake_torch = types.SimpleNamespace(
            cuda=Cuda(), inference_mode=lambda: InferenceMode(),
        )
        backend = object.__new__(MoondreamBackend)
        backend.torch = fake_torch
        backend.device = "cpu"
        backend.model = Model()
        backend.spec = ModelSpec("moondream2", "vikhyatk/moondream2", "a" * 40)

        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "frame.png"
            Image.new("RGB", (12, 8), "white").save(image_path)
            request = InferenceRequest(
                observation_id="sample", image_path=str(image_path),
                mission="visdrone_urban", capture_time_us=0,
                telemetry={}, detector_proposals=(),
            )
            profile = {
                "latency_ms": 1.0, "energy_j": 0.1,
                "peak_temperature_c": 40.0,
            }
            with patch.dict(
                "semuav.perception.VISDRONE_GENERATION_STRATEGY_BY_MODEL",
                {"moondream2": "native_query"},
                clear=False,
            ), patch(
                "semuav.hardware.profile_callable",
                side_effect=lambda operation, **kwargs: (operation(), profile),
            ):
                backend.infer(request, max_new_tokens=77)
        self.assertEqual(
            settings_seen,
            [{"max_tokens": 77, "temperature": 0.0}],
        )

    def test_local_only_flag_reaches_hf_processor_and_model_loader(self) -> None:
        class Loaded:
            def to(self, device: str) -> "Loaded":
                return self

            def eval(self) -> "Loaded":
                return self

        class ProcessorLoader:
            calls: list[tuple[str, dict[str, object]]] = []

            @classmethod
            def from_pretrained(cls, repo: str, **kwargs: object) -> object:
                cls.calls.append((repo, kwargs))
                return object()

        class ModelLoader:
            calls: list[tuple[str, dict[str, object]]] = []

            @classmethod
            def from_pretrained(cls, repo: str, **kwargs: object) -> Loaded:
                cls.calls.append((repo, kwargs))
                return Loaded()

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoProcessor = ProcessorLoader
        fake_transformers.AutoModelForImageTextToText = ModelLoader
        fake_transformers.AutoModelForCausalLM = ModelLoader
        fake_torch = types.ModuleType("torch")
        fake_torch.bfloat16 = object()
        spec = ModelSpec("model", "owner/model", "a" * 40)

        with tempfile.TemporaryDirectory() as temporary:
            cache_root = str(Path(temporary).resolve())
            with (
                patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}),
                patch.dict(os.environ, {
                    "SEMUAV_MODELS_LOCAL_ONLY": "true",
                    "HF_HUB_CACHE": cache_root,
                    "HUGGINGFACE_HUB_CACHE": cache_root,
                    "TRANSFORMERS_CACHE": cache_root,
                }),
            ):
                HFMultimodalBackend(spec, device="cpu")

        self.assertTrue(ProcessorLoader.calls[0][1]["local_files_only"])
        self.assertEqual(len(ModelLoader.calls), 1)
        self.assertTrue(ModelLoader.calls[0][1]["local_files_only"])
        self.assertNotIn("cache_dir", ModelLoader.calls[0][1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
