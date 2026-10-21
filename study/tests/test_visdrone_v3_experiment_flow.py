from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import semuav.figures as figures
import semuav.visdrone_runner as runner
from semuav.registry import StudyGateError, atomic_write_frame, hash_file


class V3ExperimentFlowTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_callback_is_synchronous_and_does_not_mutate_result(self):
        frame = pd.DataFrame({"model_id": ["real_model"], "action_accuracy": [0.75]})
        events = []
        def callback(experiment, phase, snapshots):
            events.append((experiment, phase))
            snapshots["exp1_summary"].loc[0, "action_accuracy"] = 0.0
            snapshots["extra"] = pd.DataFrame()
        runner._emit_experiment_complete(callback, 1, "validation", {"exp1_summary": frame})
        events.append("next_experiment")
        self.assertEqual(events, [(1, "validation"), "next_experiment"])
        self.assertEqual(frame.loc[0, "action_accuracy"], 0.75)

    def test_cache_binding_rejects_valid_but_different_existing_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "cache.csv.gz"
            original = pd.DataFrame({"sample_id": ["sample-a"], "latency_ms": [1.2345678901234567]})
            atomic_write_frame(artifact, original)
            original_hash = hash_file(artifact)
            changed = original.copy()
            changed.loc[0, "latency_ms"] = 9.0
            with self.assertRaisesRegex(StudyGateError, "differ from existing canonical artifact"):
                runner._bind_verified_cache_artifact(changed, artifact)
            self.assertEqual(hash_file(artifact), original_hash)
            consumed = runner._bind_verified_cache_artifact(original, artifact)
            self.assertEqual(consumed.loc[0, "latency_ms"], original.loc[0, "latency_ms"])
            self.assertEqual(hash_file(artifact), original_hash)

    def test_true_request_resumes_verified_native_cache_without_remeasuring(self):
        from types import SimpleNamespace
        from dataclasses import replace
        from test_visdrone_runner_contract import _ctx, _cfg
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ctx = replace(_ctx(root), resume=True)
            cfg = _cfg()
            cache = pd.DataFrame([{
                "sample_id": "s1", "variant": "jpeg", "sender_revision": runner.RUNNER_VERSION,
                "receiver_revision": "receiver-r1", "sender_hardware_id": "gpu:cuda:0",
                "receiver_hardware_id": "gpu:cuda:0",
            }])
            artifact = runner._results(ctx) / "validation_compact_native_receiver_cache.csv.gz"
            atomic_write_frame(artifact, cache)
            initial_hash = hash_file(artifact)
            receiver = SimpleNamespace(fusion_receipt={"revision": "r1"})
            with (
                mock.patch.object(runner, "RegisteredVisDroneNativeReceiver", return_value=receiver),
                mock.patch.object(runner, "_receiver_control_contexts", return_value={}),
                mock.patch.object(runner, "validate_native_receiver_cache") as validate,
                mock.patch.object(runner, "produce_native_receiver_cache") as produce,
            ):
                consumed, returned = runner._resolve_native_cache(
                    True, ctx=ctx, cfg=cfg, manifest=pd.DataFrame(),
                    semantic_cache=pd.DataFrame(), detector=object(),
                    detector_revision="receiver-r1", semantic_source_model="compact",
                    device="cuda:0", phase="validation",
                )
            produce.assert_not_called()
            validate.assert_called_once()
            self.assertEqual(returned, artifact)
            self.assertEqual(hash_file(artifact), initial_hash)
            self.assertEqual(consumed["sample_id"].tolist(), ["s1"])

    def test_resume_context_is_bound_to_exact_environment_seal(self):
        from dataclasses import replace
        from test_visdrone_runner_contract import _ctx
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"paths": {"output_root": str(root / "outputs")}, "runtime": {
                "inference_environment_seal_status": "SEALED_FOR_SCIENTIFIC_VALIDATION_AND_TEST",
            }}
            ctx = replace(_ctx(root), config=config, resume=True)
            arguments = {"phase": "validation", "source_model": "compact", "device": "cuda:0"}
            with self.assertRaisesRegex(StudyGateError, "environment seal"):
                runner._native_checkpoint_context(ctx, config, **arguments)
            seal = root / "outputs" / ctx.run_name / "frozen_inputs" / "INFERENCE_ENVIRONMENT_SEAL.json"
            seal.parent.mkdir(parents=True)
            seal.write_text('{"environment": "original"}')
            first = runner._native_checkpoint_context(ctx, config, **arguments)
            seal.write_text('{"environment": "changed"}')
            second = runner._native_checkpoint_context(ctx, config, **arguments)
            self.assertNotEqual(first, second)
            self.assertEqual(second, runner._native_checkpoint_context(ctx, config, **arguments))

    def test_new_cache_binding_returns_canonical_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "cache.csv.gz"
            frame = pd.DataFrame({"sample_id": ["s1"], "latency_ms": [0.012345678901234567]})
            consumed = runner._bind_verified_cache_artifact(frame, artifact)
            pd.testing.assert_frame_equal(
                consumed, pd.read_csv(artifact, float_precision="round_trip"), check_exact=True,
            )

    def test_pareto_compares_only_the_same_observations_and_removes_equal_cost_dominance(self):
        frame = pd.DataFrame([
            {"sample_id": "det-only", "mode": "L5_IMAGE_OR_CLIP", "variant": "image", "wire_bytes": 50, "native_receiver_correct": 1.0},
            {"sample_id": "mot-paired", "mode": "L5_IMAGE_OR_CLIP", "variant": "image", "wire_bytes": 100, "native_receiver_correct": 0.0},
            {"sample_id": "mot-paired", "mode": "L5_IMAGE_OR_CLIP", "variant": "clip_low", "wire_bytes": 100, "native_receiver_correct": 0.5},
            {"sample_id": "mot-paired", "mode": "L5_IMAGE_OR_CLIP", "variant": "clip_high", "wire_bytes": 100, "native_receiver_correct": 1.0},
        ])
        figure, _ = figures._figure_exp2_pareto({"exp2_frontier": frame})
        frontier = next(line for line in figure.axes[0].lines if line.get_label() == "Pareto frontier")
        self.assertEqual(list(frontier.get_xdata()), [100.0])
        self.assertEqual(list(frontier.get_ydata()), [1.0])
        points = figure.axes[0].collections[0].get_offsets()
        self.assertEqual(sorted(float(value) for value in points[:, 1]), [0.0, 0.5, 1.0])
        self.assertIn("Same 1 observations", figure.axes[0].get_title())

    def test_filtered_manifests_merge_and_remove_stale_outputs(self):
        first = pd.DataFrame([{
            "model_id": "real", "action_accuracy": .6,
            "full_path_latency_p50_ms": 2., "full_path_latency_p95_ms": 3.,
        }])
        seventh = pd.DataFrame([{"item": "workstation", "status": "PASS"}])
        with tempfile.TemporaryDirectory() as directory:
            figures.generate_publication_figures(
                {"exp1_summary": first}, directory, True, experiment=1,
                evidence_label="VALIDATION — DEVELOPMENT RESULTS", strict=True,
            )
            figures.generate_publication_figures(
                {"exp7_readiness": seventh}, directory, True, experiment=7, strict=True,
            )
            manifest = json.loads((Path(directory) / "figure_manifest.json").read_text())
            self.assertEqual({row["figure_id"] for row in manifest["figures"]}, {
                "exp1_perception_metrics", "exp1_latency", "exp7_readiness_hardware",
            })
            self.assertEqual(manifest["figures"][0]["evidence_label"], "VALIDATION — DEVELOPMENT RESULTS")
            figures.generate_publication_figures({}, directory, True, experiment=1)
            self.assertTrue((Path(directory) / "exp7_readiness_hardware.png").is_file())
            self.assertFalse((Path(directory) / "exp1_perception_metrics.png").exists())
            manifest = json.loads((Path(directory) / "figure_manifest.json").read_text())
            self.assertEqual(len(manifest["figures"]), 3)

    def test_forest_renders_not_estimable_status_without_inventing_effect(self):
        frame = pd.DataFrame([
            {"hypothesis": f"H{i}", "estimate": .1 if i in (1,4) else None,
             "ci_low": .02 if i in (1,4) else None, "ci_high": .2 if i in (1,4) else None,
             "effect_available": i in (1,4),
             "execution_status": "COMPLETE" if i in (1,4) else "NOT_ESTIMABLE",
             "unavailable_reason": "Insufficient independent groups" if i in (2,3) else "No embedded evaluation"}
            for i in range(1,6)
        ])
        figure, _ = figures._figure_hypothesis_forest({"hypothesis_effects": frame})
        self.assertEqual(len(figure.axes[0].get_yticklabels()), 5)
        self.assertEqual(len(figure.axes[0].lines[0].get_xdata()), 2)
        annotation = " ".join(text.get_text() for text in figure.axes[0].texts)
        self.assertIn("Not Estimable", annotation)
        self.assertIn("Insufficient independent groups", annotation)

    def test_strict_export_failure_is_recorded_and_raised(self):
        frame = pd.DataFrame([{"item": "workstation", "status": "PASS"}])
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            figures, "_atomic_save_figure", side_effect=OSError("disk full"),
        ):
            with self.assertRaisesRegex(RuntimeError, "disk full"):
                figures.generate_publication_figures(
                    {"exp7_readiness": frame}, directory, True, experiment=7, strict=True,
                )
            manifest = json.loads((Path(directory) / "figure_manifest.json").read_text())
            self.assertIn("disk full", manifest["figures"][0]["reason"])

    def test_missing_safety_denominator_keeps_other_endpoints(self):
        frame = pd.DataFrame([{
            "subset": "all", "policy": "proposed_full", "critical_miss_rate": .1,
            "false_alert_rate": .2, "high_confidence_semantic_error_rate": float("nan"),
        }])
        figure, _ = figures._figure_exp5_safety({"exp5_summary": frame})
        self.assertTrue(any("Not estimable" in text.get_text() for axis in figure.axes for text in axis.texts))


if __name__ == "__main__":
    unittest.main()
