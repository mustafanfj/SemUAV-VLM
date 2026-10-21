from __future__ import annotations

import dataclasses
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from semuav.perception import (
    LEGACY_MOONDREAM_INFERENCE_FIELDS,
    InferenceRecord,
    InferenceRequest,
    ModelSpec,
)
from semuav.registry import StudyGateError, hash_file, hash_object
from semuav.visdrone_measurement import (
    MEASUREMENT_SCHEMA_VERSION,
    RepeatedMeasurementPerceptionBackend,
    run_repeated_detector_measurement,
)


class _FakeDetector:
    def __init__(self, generation: int, *, vary_output: bool = False) -> None:
        self.generation = generation
        self.vary_output = vary_output
        self.tracker = "botsort.yaml"
        self.device = "cuda:0"
        self.inference_spec = {"frozen": True, "seed": 20260823}
        self.calls: list[tuple[str, bool]] = []
        self.stream_position = 0

    def detect(self, image_path: str, persist: bool = False):
        self.calls.append((image_path, persist))
        self.stream_position = self.stream_position + 1 if persist else 1
        generation = self.generation if self.vary_output else 0
        return [{
            "label": "car",
            "confidence": 0.9,
            "track_id": self.stream_position + generation * 100,
            "bbox_normalized_xyxy": [0.1, 0.1, 0.2, 0.2],
        }]


class _FakeBackend:
    def __init__(
        self,
        *,
        nondeterministic: bool = False,
        nondeterministic_lineage: bool = False,
    ) -> None:
        self.spec = ModelSpec("selected_vlm", "local", "a" * 40)
        self.nondeterministic = nondeterministic
        self.nondeterministic_lineage = nondeterministic_lineage
        self.calls: list[str] = []
        self.measured_calls = 0

    def infer(self, request: InferenceRequest, prompt: str, max_new_tokens: int) -> InferenceRecord:
        self.calls.append(request.observation_id)
        is_warmup = request.observation_id == "warmup-development"
        if is_warmup:
            ordinal = len(self.calls)
            raw_text = f"ignored-warmup-{ordinal}"
            parsed = {"ignored": ordinal}
            latency, energy, memory, temperature = (999.0, 99.0, 999.0, 99.0)
        else:
            ordinal = self.measured_calls
            self.measured_calls += 1
            index = ordinal % 3
            raw_text = f"stable-{ordinal}" if self.nondeterministic else "stable"
            parsed = {"recommended_action": "REVIEW"}
            latency = [1.0, 3.0, 2.0][index]
            energy = [0.1, 0.3, 0.2][index]
            memory = [10.0, 12.0, 11.0][index]
            temperature = [50.0, 52.0, 51.0][index]
        record = InferenceRecord(
            observation_id=request.observation_id,
            model_id=self.spec.model_id,
            model_revision=self.spec.revision,
            raw_text=raw_text,
            parsed=parsed,
            schema_valid=True,
            error=None,
            input_tokens=100,
            output_tokens=20,
            confidence=0.75,
            confidence_source="registered_score",
            latency_ms=latency,
            peak_memory_mb=memory,
            energy_j=energy,
            prompt_hash=hash_object(prompt),
            peak_temperature_c=temperature,
            decision_contract_version="semuav.visdrone-proposal-decision.v2",
            decision_core_valid=True,
            decision_core_json=(
                '{"accepted_proposal_ids":[],"limited_visibility_proposal_ids":[],'
                '"sufficient_evidence":true}'
            ),
            decision_context_sha256="b" * 64,
            decision_proposal_count=0,
            decision_composer_policy_sha256="c" * 64,
            terminated_by_eos=True,
            reached_token_cap=False,
        )
        if self.nondeterministic_lineage and not is_warmup:
            record.decision_context_sha256 = f"{self.measured_calls:064x}"
        return record


class RepeatedDetectorMeasurementTests(unittest.TestCase):
    def _inputs(self, directory: str):
        root = Path(directory)
        warmup = root / "warmup.jpg"
        image_a = root / "frame-0001.jpg"
        image_b = root / "frame-0002.jpg"
        for path, content in (
            (warmup, b"warmup"), (image_a, b"a"), (image_b, b"b"),
        ):
            path.write_bytes(content)
        manifest = pd.DataFrame([
            {"sample_id": "a-profile-1", "source_sample_id": "a", "image_path": str(image_a), "sequence_id": "seq", "frame_index": 1},
            {"sample_id": "a-profile-2", "source_sample_id": "a", "image_path": str(image_a), "sequence_id": "seq", "frame_index": 1},
            {"sample_id": "b-profile-1", "source_sample_id": "b", "image_path": str(image_b), "sequence_id": "seq", "frame_index": 2},
            {"sample_id": "b-profile-2", "source_sample_id": "b", "image_path": str(image_b), "sequence_id": "seq", "frame_index": 2},
        ])
        return warmup, manifest

    def test_fresh_stream_repetitions_aggregate_and_exclude_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            warmup, manifest = self._inputs(directory)
            instances: list[_FakeDetector] = []

            def factory():
                detector = _FakeDetector(len(instances))
                instances.append(detector)
                return detector

            measurement_index = {"value": 0}

            def profile_callable(function, *, device_index: int, measure_energy: bool):
                measurement_index["value"] += 1
                value = measurement_index["value"]
                return function(), {
                    "latency_ms": float(value),
                    "energy_j": float(value) / 10,
                    "peak_temperature_c": 50.0 + value,
                }

            class FakeCuda:
                @staticmethod
                def is_available():
                    return True

                @staticmethod
                def reset_peak_memory_stats(device_index):
                    return None

                @staticmethod
                def max_memory_allocated(device_index):
                    return measurement_index["value"] * 2**20

            fake_torch = types.SimpleNamespace(cuda=FakeCuda())
            with mock.patch("semuav.hardware.profile_callable", side_effect=profile_callable), mock.patch.dict(
                "sys.modules", {"torch": fake_torch},
            ):
                output = run_repeated_detector_measurement(
                    factory,
                    manifest,
                    warmup_image_path=warmup,
                    warmup_image_sha256=hash_file(warmup),
                    warmup_source_id="warmup-development",
                    warmup_split="calibration",
                )

            self.assertEqual(len(instances), 4)
            self.assertEqual(len(instances[0].calls), 5)
            self.assertTrue(all(not persist for _, persist in instances[0].calls))
            for measured_instance in instances[1:]:
                self.assertEqual(len(measured_instance.calls), 7)
                self.assertEqual(
                    measured_instance.calls[:5], [(str(warmup), False)] * 5,
                )
                self.assertEqual(
                    [persist for _, persist in measured_instance.calls[5:]],
                    [False, True],
                )
            self.assertEqual(len(output), 4)
            by_source = output.drop_duplicates("source_sample_id").set_index("source_sample_id")
            self.assertEqual(by_source.loc["a", "latency_ms"], 3.0)
            self.assertAlmostEqual(by_source.loc["a", "energy_j"], 0.3)
            self.assertEqual(by_source.loc["a", "peak_memory_mb"], 5.0)
            self.assertEqual(by_source.loc["a", "peak_temperature_c"], 55.0)
            self.assertEqual(by_source.loc["b", "latency_ms"], 4.0)
            self.assertEqual(by_source.loc["b", "peak_memory_mb"], 6.0)
            self.assertTrue(output["measurement_warmup_excluded"].all())
            self.assertTrue(output["measurement_schema_version"].eq(MEASUREMENT_SCHEMA_VERSION).all())
            self.assertTrue(output["measurement_repetitions"].eq(3).all())
            self.assertTrue(
                output["measurement_state_reset"].str.contains(
                    "each_warmed_unmeasured", regex=False,
                ).all()
            )

    def test_nondeterministic_or_reused_tracker_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            warmup, manifest = self._inputs(directory)
            generation = {"value": 0}

            def varying_factory():
                value = generation["value"]
                generation["value"] += 1
                return _FakeDetector(value, vary_output=True)

            profile = {
                "latency_ms": 1.0,
                "energy_j": 0.1,
                "peak_temperature_c": 40.0,
            }
            fake_torch = types.SimpleNamespace(
                cuda=types.SimpleNamespace(
                    is_available=lambda: False,
                    reset_peak_memory_stats=lambda _: None,
                ),
            )
            # CUDA-off would make memory unavailable, but nondeterminism is
            # checked before resource aggregation.
            with mock.patch(
                "semuav.hardware.profile_callable",
                side_effect=lambda function, **_: (function(), profile),
            ), mock.patch.dict("sys.modules", {"torch": fake_torch}):
                with self.assertRaisesRegex(StudyGateError, "nondeterministic"):
                    run_repeated_detector_measurement(
                        varying_factory,
                        manifest,
                        warmup_image_path=warmup,
                        warmup_image_sha256=hash_file(warmup),
                        warmup_source_id="warmup-development",
                        warmup_split="train",
                    )

            reused = _FakeDetector(0)
            with self.assertRaisesRegex(StudyGateError, "reused tracker state"):
                run_repeated_detector_measurement(
                    lambda: reused,
                    manifest,
                    warmup_image_path=warmup,
                    warmup_image_sha256=hash_file(warmup),
                    warmup_source_id="warmup-development",
                    warmup_split="validation",
                )

    def test_invalid_measurement_configuration_and_warmup_hash_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            warmup, manifest = self._inputs(directory)
            with self.assertRaisesRegex(ValueError, "positive integer"):
                run_repeated_detector_measurement(
                    lambda: _FakeDetector(0), manifest,
                    warmup_image_path=warmup, warmup_image_sha256=hash_file(warmup),
                    warmup_source_id="warm", warmup_split="train", repetitions=0,
                )
            with self.assertRaisesRegex(StudyGateError, "changed"):
                run_repeated_detector_measurement(
                    lambda: _FakeDetector(0), manifest,
                    warmup_image_path=warmup, warmup_image_sha256="0" * 64,
                    warmup_source_id="warm", warmup_split="train",
                )


class RepeatedVLMMeasurementTests(unittest.TestCase):
    def _requests(self, directory: str):
        root = Path(directory)
        warmup_image = root / "warmup.jpg"
        measured_image = root / "measured.jpg"
        warmup_image.write_bytes(b"warmup")
        measured_image.write_bytes(b"measured")
        warmup = InferenceRequest(
            "warmup-development", str(warmup_image), "urban", 1, {}, (),
        )
        measured = InferenceRequest("measured", str(measured_image), "urban", 2, {}, ())
        return warmup_image, warmup, measured

    def test_warms_once_and_aggregates_deterministic_measured_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            warmup_image, warmup, measured = self._requests(directory)
            backend = _FakeBackend()
            wrapper = RepeatedMeasurementPerceptionBackend(
                backend,
                warmup_request=warmup,
                warmup_image_sha256=hash_file(warmup_image),
                warmup_split="validation",
                warmup_prompt="frozen warmup prompt",
                warmup_max_new_tokens=32,
            )
            first = wrapper.infer(measured, "frozen measured prompt", 64)
            second = wrapper.infer(measured, "frozen measured prompt", 64)
            self.assertEqual(len(backend.calls), 11)
            self.assertEqual(backend.calls[:5], ["warmup-development"] * 5)
            self.assertEqual(first.latency_ms, 2.0)
            self.assertAlmostEqual(first.energy_j, 0.2)
            self.assertEqual(first.peak_memory_mb, 12.0)
            self.assertEqual(first.peak_temperature_c, 52.0)
            self.assertEqual(first.decision_context_sha256, "b" * 64)
            self.assertEqual(first.decision_composer_policy_sha256, "c" * 64)
            self.assertTrue(first.terminated_by_eos)
            self.assertFalse(first.reached_token_cap)
            self.assertEqual(second.latency_ms, 2.0)
            self.assertEqual(
                wrapper.measurement_metadata["measurement_schema_version"],
                MEASUREMENT_SCHEMA_VERSION,
            )
            self.assertTrue(wrapper.measurement_metadata["measurement_warmup_excluded"])

    def test_nondeterministic_vlm_output_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            warmup_image, warmup, measured = self._requests(directory)
            wrapper = RepeatedMeasurementPerceptionBackend(
                _FakeBackend(nondeterministic=True),
                warmup_request=warmup,
                warmup_image_sha256=hash_file(warmup_image),
                warmup_split="calibration",
                warmup_prompt="warmup prompt",
                warmup_max_new_tokens=32,
            )
            with self.assertRaisesRegex(StudyGateError, "nondeterministic"):
                wrapper.infer(measured, "measured prompt", 64)

    def test_nondeterministic_vlm_decision_lineage_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            warmup_image, warmup, measured = self._requests(directory)
            wrapper = RepeatedMeasurementPerceptionBackend(
                _FakeBackend(nondeterministic_lineage=True),
                warmup_request=warmup,
                warmup_image_sha256=hash_file(warmup_image),
                warmup_split="calibration",
                warmup_prompt="warmup prompt",
                warmup_max_new_tokens=32,
            )
            with self.assertRaisesRegex(StudyGateError, "nondeterministic"):
                wrapper.infer(measured, "measured prompt", 64)

    def test_active_measurement_commitment_excludes_legacy_moondream_provenance(self) -> None:
        self.assertEqual(
            MEASUREMENT_SCHEMA_VERSION,
            "semuav.repeated-inference-measurement.v2",
        )
        request = InferenceRequest("measured", "/not-read.jpg", "urban", 2, {}, ())
        baseline = _FakeBackend().infer(request, "measured prompt", 256)
        expected = {
            "generation_policy_version": "semuav.visdrone-generation-policy.v5",
            "generation_policy_sha256": "d" * 64,
            "generation_strategy": "qwen_open_brace_assistant_continuation",
            "generation_max_new_tokens": 256,
            "assistant_prefill_sha256": "e" * 64,
            "assistant_prefill_characters": 1,
            "generated_continuation_sha256": "f" * 64,
        }
        record = dataclasses.replace(baseline, **expected)
        committed = RepeatedMeasurementPerceptionBackend._committed_record(record)
        payload = json.loads(committed)
        self.assertEqual(
            {field: payload[field] for field in expected},
            expected,
        )
        active_row = record.as_row()
        self.assertFalse(
            set(LEGACY_MOONDREAM_INFERENCE_FIELDS).intersection(payload)
        )
        self.assertFalse(
            set(LEGACY_MOONDREAM_INFERENCE_FIELDS).intersection(active_row)
        )

        for field, value in expected.items():
            replacement = value + "-changed" if isinstance(value, str) else value + 1
            with self.subTest(field=field):
                changed = dataclasses.replace(record, **{field: replacement})
                self.assertNotEqual(
                    RepeatedMeasurementPerceptionBackend._committed_record(changed),
                    committed,
                )

        for field in LEGACY_MOONDREAM_INFERENCE_FIELDS:
            value = "1" * 64 if field.endswith("sha256") else 1
            with self.subTest(legacy_field=field):
                contaminated = dataclasses.replace(record, **{field: value})
                with self.assertRaisesRegex(
                    StudyGateError, "Legacy Moondream provenance is forbidden",
                ):
                    contaminated.as_row()
                with self.assertRaisesRegex(
                    StudyGateError, "Legacy Moondream provenance is forbidden",
                ):
                    RepeatedMeasurementPerceptionBackend._committed_record(
                        contaminated
                    )

    def test_invalid_vlm_measurement_configuration_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            warmup_image, warmup, _ = self._requests(directory)
            with self.assertRaisesRegex(ValueError, "positive integer"):
                RepeatedMeasurementPerceptionBackend(
                    _FakeBackend(),
                    warmup_request=warmup,
                    warmup_image_sha256=hash_file(warmup_image),
                    warmup_split="train",
                    warmup_prompt="warmup prompt",
                    warmup_max_new_tokens=32,
                    warmup_iterations=0,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
