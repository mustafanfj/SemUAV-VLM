from __future__ import annotations

import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from semuav.registry import RunContext, StudyGateError, frame_stage, hash_file, hash_object
import semuav.visdrone_runner as runner


ACTIVE_MODEL_IDS = (
    "qwen3_vl_2b",
    "smolvlm2_2b",
    "qwen3_vl_8b_upper_bound",
)


def _cfg() -> dict[str, object]:
    return {
        "models": {
            "vlms": [
                {"id": "qwen3_vl_2b"},
                {"id": "smolvlm2_2b"},
            ],
            "upper_bound": {"id": "qwen3_vl_8b_upper_bound"},
        },
    }


class V2R9RuntimeModelBindingTests(unittest.TestCase):
    def test_runner_version_and_scientific_entrypoints_have_no_tokenizer_arg(self) -> None:
        self.assertEqual(
            runner.RUNNER_VERSION,
            "semuav.visdrone-paper-runner.v7",
        )
        for function in (
            runner._bind_runtime_model_lock,
            runner.run_validation_perception_and_frontier,
            runner.run_confirmatory_test_once,
        ):
            parameters = inspect.signature(function).parameters
            self.assertNotIn("moondream_tokenizer_lock", parameters)
            self.assertNotIn("moondream_tokenizer_cache", parameters)

    def test_runtime_lock_verification_uses_active_config_and_exact_environment(self) -> None:
        verified = {
            "models": [{"model_id": model_id} for model_id in ACTIVE_MODEL_IDS]
        }
        environment = {
            "SEMUAV_QWEN_REVISION": "1" * 40,
            "SEMUAV_SMOLVLM_REVISION": "2" * 40,
            "SEMUAV_UPPER_BOUND_REVISION": "3" * 40,
            "HF_HUB_OFFLINE": "1",
        }
        config = _cfg()
        with (
            mock.patch.object(
                runner,
                "verify_huggingface_model_lock",
                return_value=verified,
            ) as verify,
            mock.patch.object(
                runner,
                "model_loader_environment",
                return_value=environment,
            ) as loader_environment,
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            observed = runner._bind_runtime_model_lock(
                "MODEL_LOCK.json", config
            )

            self.assertIs(observed, verified)
            self.assertEqual(
                {name: os.environ[name] for name in environment},
                environment,
            )
            self.assertNotIn("SEMUAV_MOONDREAM_REVISION", os.environ)
            self.assertNotIn("SEMUAV_MOONDREAM_TOKENIZER_LOCK", os.environ)
        verify.assert_called_once_with(
            "MODEL_LOCK.json", models=config["models"]
        )
        loader_environment.assert_called_once_with("MODEL_LOCK.json")

    def test_validation_binds_three_model_lock_before_manifest_access(self) -> None:
        ctx = SimpleNamespace(is_scientific=True)
        with mock.patch.object(
            runner,
            "_bind_runtime_model_lock",
            side_effect=StudyGateError("MODEL_LOCK_SENTINEL"),
        ) as bind:
            with self.assertRaisesRegex(StudyGateError, "MODEL_LOCK_SENTINEL"):
                runner.run_validation_perception_and_frontier(
                    ctx=ctx,
                    cfg=_cfg(),
                    manifest=pd.DataFrame(),
                    model_lock="MODEL_LOCK.json",
                    device="cuda:0",
                    build_native_receiver_cache=True,
                    build_feature_cache=True,
                    feature_head_lock="FEATURE_HEAD_LOCK.json",
                )
        bind.assert_called_once_with("MODEL_LOCK.json", _cfg())

    def test_confirmatory_rejects_model_lock_drift_before_ledger_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_lock = root / "MODEL_LOCK.json"
            model_lock.write_text('{"v2r9":true}\n', encoding="utf-8")
            ctx = SimpleNamespace(run_root=root)
            protocol = {
                "protocol_hash": "6" * 64,
                "decisions": {
                    "selection": {
                        "model_lock_sha256": "0" * 64,
                    },
                },
            }
            with mock.patch.object(
                runner,
                "_bind_runtime_model_lock",
                return_value={
                    "models": [
                        {"model_id": model_id} for model_id in ACTIVE_MODEL_IDS
                    ],
                },
            ):
                with self.assertRaisesRegex(
                    StudyGateError, "MODEL_LOCK differs",
                ):
                    runner.run_confirmatory_test_once(
                        ctx=ctx,
                        cfg=_cfg(),
                        manifest=pd.DataFrame(),
                        traces=None,
                        protocol=protocol,
                        model_lock=model_lock,
                        device="cuda:0",
                    )
            self.assertNotEqual(hash_file(model_lock), "0" * 64)
            self.assertFalse((root / "test_access_ledger.json").exists())

    def test_runner_source_has_no_active_moondream_dependency(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8").lower()
        self.assertNotIn("moondream", source)
        self.assertNotIn("starmie", source)

    def test_active_model_factory_rejects_historical_backend(self) -> None:
        with self.assertRaisesRegex(
            StudyGateError, "unsupported backend",
        ):
            runner._model_backend(
                {
                    "model_id": "moondream2",
                    "repo": "vikhyatk/moondream2",
                    "resolved_revision": "1" * 40,
                    "role": "compact",
                    "backend": "moondream2",
                },
                "cuda:0",
                (768, 768),
            )

    def test_warmup_request_constructs_inference_request_from_inline_proposals(self) -> None:
        proposal = {"proposal_id": 4, "category": "car", "score": 0.75}
        request = runner._warmup_request({
            "sample_id": "mot_val:seq01:000001",
            "image_path": "/immutable/validation/frame.jpg",
            "mission": "visdrone_urban",
            "capture_time_us": 1_234_567,
            "telemetry_json": '{"altitude_m":42.5}',
            "detector_proposals_json": json.dumps([proposal]),
        })

        self.assertIsInstance(request, runner.InferenceRequest)
        self.assertEqual(request.observation_id, "mot_val:seq01:000001")
        self.assertEqual(request.image_path, "/immutable/validation/frame.jpg")
        self.assertEqual(request.mission, "visdrone_urban")
        self.assertEqual(request.capture_time_us, 1_234_567)
        self.assertEqual(request.telemetry, {"altitude_m": 42.5})
        self.assertEqual(tuple(request.detector_proposals), (proposal,))

    def test_warmup_request_constructs_inference_request_from_detector_cache(self) -> None:
        proposal = {"proposal_id": 9, "category": "bus", "score": 0.625}
        detector_cache = pd.DataFrame([
            {
                "sample_id": "mot_val:seq02:000003",
                "proposals_json": json.dumps([proposal]),
            },
        ])
        request = runner._warmup_request(
            {
                "sample_id": "mot_val:seq02:000003",
                "image_path": "/immutable/validation/other-frame.jpg",
                "mission": "visdrone_urban",
                "capture_time_us": 7_654_321,
                "telemetry_json": '{"speed_mps":3.25}',
            },
            detector_cache=detector_cache,
        )

        self.assertIsInstance(request, runner.InferenceRequest)
        self.assertEqual(request.observation_id, "mot_val:seq02:000003")
        self.assertEqual(request.image_path, "/immutable/validation/other-frame.jpg")
        self.assertEqual(request.mission, "visdrone_urban")
        self.assertEqual(request.capture_time_us, 7_654_321)
        self.assertEqual(request.telemetry, {"speed_mps": 3.25})
        self.assertEqual(tuple(request.detector_proposals), (proposal,))

    def test_measurement_metadata_normalizes_numpy_scalars_to_exact_json(self) -> None:
        raw = {
            "measurement_schema_version": "semuav.repeated-inference-measurement.v2",
            "measurement_warmup_excluded": np.bool_(True),
            "measurement_warmup_iterations": np.int64(5),
            "measurement_repetitions": np.uint32(3),
            "nested": {"temperature_c": np.float32(47.25)},
            "repetition_ids": (np.int16(1), np.int16(2), np.int16(3)),
        }
        normalized = runner._normalize_measurement_metadata(raw)

        self.assertIs(type(normalized["measurement_warmup_excluded"]), bool)
        self.assertIs(type(normalized["measurement_warmup_iterations"]), int)
        self.assertIs(type(normalized["measurement_repetitions"]), int)
        self.assertIs(type(normalized["nested"]["temperature_c"]), float)
        self.assertEqual(normalized["repetition_ids"], [1, 2, 3])
        self.assertEqual(
            hash_object(normalized),
            hash_object({
                "measurement_schema_version": (
                    "semuav.repeated-inference-measurement.v2"
                ),
                "measurement_warmup_excluded": True,
                "measurement_warmup_iterations": 5,
                "measurement_repetitions": 3,
                "nested": {"temperature_c": 47.25},
                "repetition_ids": [1, 2, 3],
            }),
        )
        self.assertEqual(
            json.loads(json.dumps(normalized, allow_nan=False)), normalized,
        )

    def test_measurement_metadata_rejects_nonfinite_and_unsupported_values(self) -> None:
        for value, expected in (
            (np.float64(np.nan), "non-finite"),
            (np.float32(np.inf), "non-finite"),
            (np.longdouble("1.25"), "unsupported"),
            (np.clongdouble(1 + 2j), "unsupported"),
            (pd.NA, "unsupported"),
            ({"not": {"json"}}, "unsupported"),
        ):
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(StudyGateError, expected):
                    runner._normalize_measurement_metadata({"measurement_value": value})
        with self.assertRaisesRegex(StudyGateError, "invalid key"):
            runner._normalize_measurement_metadata({1: np.int64(5)})

    def test_validation_and_confirmatory_normalize_before_cache_key(self) -> None:
        for function in (
            runner.run_validation_perception_and_frontier,
            runner.run_confirmatory_test_once,
        ):
            with self.subTest(function=function.__name__):
                source = inspect.getsource(function)
                self.assertEqual(source.count("_normalize_measurement_metadata("), 1)
                normalize_at = source.index("_normalize_measurement_metadata(")
                cache_at = source.index(
                    '"measurement": measurement_metadata', normalize_at,
                )
                self.assertLess(normalize_at, cache_at)
                self.assertNotIn("default=str", source)

    def test_validation_int64_measurement_metadata_reaches_frame_stage(self) -> None:
        """Reproduce the v2r9 first-shard failure through its runner boundary."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            ctx = RunContext.create(
                project, root / "cache", "full", "v2r9-int64-regression",
                {"regression": "v2r9-detector-measurement-int64"}, resume=False,
            )
            model_lock = root / "MODEL_LOCK.json"
            model_lock.write_text('{"models":[]}\n', encoding="utf-8")
            sample = pd.DataFrame([{
                "sample_id": "mot_val:seq01:000001",
                "visual_group_id": "mot_val:seq01",
                "split": "calibration",
                "mission": "visdrone_urban",
                "image_path": str(root / "frame.jpg"),
                "capture_time_us": 1_000_000,
                "telemetry_json": '{"altitude_m":50.0}',
            }])
            detector_cache = pd.DataFrame([{
                "sample_id": "mot_val:seq01:000001",
                "proposals_json": "[]",
                "measurement_schema_version": (
                    "semuav.repeated-inference-measurement.v2"
                ),
                "measurement_warmup_iterations": np.int64(5),
                "measurement_repetitions": np.int64(3),
                "measurement_warmup_excluded": np.bool_(True),
            }])
            self.assertIsInstance(
                detector_cache.iloc[0]["measurement_warmup_iterations"],
                np.integer,
            )
            observed: dict[str, object] = {}

            def exercise_first_shard(
                manifest: pd.DataFrame,
                backends: object,
                **kwargs: object,
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
                del manifest, backends
                context = kwargs["inference_cache_context"]
                assert isinstance(context, dict)
                measurement = context["measurement"]
                assert isinstance(measurement, dict)
                observed.update(measurement)
                staged = frame_stage(
                    ctx,
                    "01_vlm_detector_only_shard_00000",
                    {
                        "execution_context": context,
                        "requests": [{"observation_id": "mot_val:seq01:000001"}],
                    },
                    lambda: pd.DataFrame([{"observation_id": "mot_val:seq01:000001"}]),
                )
                return staged, pd.DataFrame([{"model_id": "detector_only"}])

            cfg = {
                "benchmark": {
                    "analysis_sampling": {"method": "registered"},
                    "registered_roi_xyxy": [0.25, 0.25, 0.75, 0.85],
                    "min_image_plane_iou": 0.05,
                },
                "models": {
                    "max_new_tokens": 384,
                    "input_size": [768, 768],
                    "proposal_policy": {"minimum_confidence": 0.05},
                },
                "runtime": {"warmup_iterations": 5, "steady_state_repetitions": 3},
            }
            warmup = {
                "sample_id": "mot_val:seq01:000001",
                "source_id": "mot_val:seq01:000001",
                "image_path": str(root / "frame.jpg"),
                "image_sha256": "0" * 64,
                "split": "calibration",
                "mission": "visdrone_urban",
                "capture_time_us": 1_000_000,
                "telemetry_json": '{"altitude_m":50.0}',
            }

            with (
                mock.patch.object(runner, "_bind_runtime_model_lock", return_value={"models": []}),
                mock.patch.object(runner, "_require_registered_phase_sample", return_value=sample),
                mock.patch.object(runner, "_development_warmup_spec", return_value=warmup),
                mock.patch.object(
                    runner,
                    "_detector_and_cache",
                    return_value=(None, detector_cache, "a" * 40, {"sha256": "b" * 64}),
                ),
                mock.patch.object(runner, "_priority_profiles", return_value={"general": ("car",)}),
                mock.patch.object(runner, "experiment1_perception", side_effect=exercise_first_shard),
                mock.patch.object(
                    runner,
                    "_fit_validation_selection",
                    side_effect=RuntimeError("SELECTION_SENTINEL"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "SELECTION_SENTINEL"):
                    runner.run_validation_perception_and_frontier(
                        ctx=ctx,
                        cfg=cfg,
                        manifest=sample,
                        model_lock=model_lock,
                        device="cuda:0",
                        build_native_receiver_cache=False,
                        build_feature_cache=False,
                        feature_head_lock=root / "FEATURE_HEAD_LOCK.json",
                    )

            self.assertIs(type(observed["measurement_warmup_iterations"]), int)
            self.assertEqual(observed["measurement_warmup_iterations"], 5)
            self.assertEqual(
                ctx.stage_record("01_vlm_detector_only_shard_00000")["status"],
                "COMPLETE",
            )


if __name__ == "__main__":
    unittest.main()
