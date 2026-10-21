from __future__ import annotations

import ast
import inspect
import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from semuav.network import (
    Direction,
    EndBehavior,
    LinkSegment,
    LinkTrace,
    TraceSemantics,
    constant_trace,
)
from semuav.registry import RunContext, StudyGateError, hash_file
from semuav.visdrone_paper import UntouchedTestAccessError
import semuav.visdrone_runner as runner


VISDRONE_PROFILES = {
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
}


def _function_tree(function: object) -> ast.FunctionDef:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    node = tree.body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def _leaf_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _calls(function: object, name: str) -> list[ast.Call]:
    return [
        node for node in ast.walk(_function_tree(function))
        if isinstance(node, ast.Call) and _leaf_name(node.func) == name
    ]


def _ctx(root: Path) -> RunContext:
    run_root = root / "run"
    (run_root / "results").mkdir(parents=True)
    return RunContext(
        project_root=root,
        run_root=run_root,
        profile="full",
        run_name="visdrone-contract",
        config={},
        config_hash="b" * 64,
        code_hash="c" * 64,
        created_utc="2026-08-31T00:00:00Z",
        resume=False,
    )


def _cfg(*, feature_lock: Path | None = None) -> dict[str, object]:
    feature = {
        "encoder_revision": "IMAGENET1K_V2",
        "split_layer": "layer3",
        "quantization": ["int8", "int4"],
    }
    if feature_lock is not None:
        feature.update({
            "receiver_head": str(feature_lock.resolve()),
            "receiver_head_revision": hash_file(feature_lock),
        })
    return {
        "seed": 20260823,
        "seeds": [20260823, 20260829, 20260907],
        "split": {"min_test_units": 30, "min_critical_test_events": 140},
        "benchmark": {
            "registered_roi_xyxy": [0.25, 0.25, 0.75, 0.85],
            "min_image_plane_iou": 0.05,
            "mission_profiles": VISDRONE_PROFILES,
            "analysis_sampling": {
                "method": "outcome_blind_group_hash_temporal_v1",
                "max_groups_per_component": {
                    "train": 80,
                    "calibration": 80,
                    "validation": 128,
                    "test": 600,
                },
                "max_frames_per_group": {
                    "train": 8,
                    "calibration": 8,
                    "validation": 8,
                    "test": 8,
                },
            },
            "temporal_sampling": {
                "schema": "semuav.visdrone-temporal-controller.v1",
                "method": "outcome_blind_mot_sequence_hash_contiguous_v1",
                "validation_sequences": 7,
                "test_sequences": 15,
                "sequences_per_split": {"validation": 7, "test": 15},
                "observations_per_sequence": 64,
                "source_frame_stride": 1,
                "target_rate_hz": 5,
                "minimum_temporal_groups": 30,
                "minimum_group_trace_cells": 500,
                "component_by_split": {
                    "validation": "mot_val", "test": "mot_test_dev",
                },
            },
        },
        "models": {"feature_baseline": feature},
        "payloads": {
            "image_codecs": ["jpeg", "webp", "avif"],
            "image_qualities": [25, 50, 80],
            "image_scales": [0.5, 1.0],
            "video_crf": [23, 35],
            "video_fps": 5,
        },
        "evaluation": {
            "energy_claim_scope": "combined_sender_and_ground_gpu_board_nvml_compute_only_excludes_radio_system",
            "radio_energy_included": False,
            "primary_wire_budget_bytes_per_s": 1000.0,
            "primary_energy_budget_j_per_s": 1000.0,
            "natural_test_units": 30,
            "mission_priority_positive_units": 140,
            "mission_priority_negative_units": 73,
            "false_alert_limit": 0.05,
            "bootstrap_repetitions": 200,
            "h2_matched_budget_relative_tolerance": 0.05,
            "primary_margin": 0.03,
            "h4_matched_coverage": 0.10,
            "h4_high_confidence_threshold": 0.80,
            "expected_not_estimable_hypotheses": ["H2", "H3"],
            "expected_not_estimable_reasons": {
                "H2": (
                    "bounded_15_sequence_by_30_trace_temporal_design_has_15_groups_"
                    "and_450_group_trace_cells_below_30_and_500_minima"
                ),
                "H3": (
                    "bounded_15_sequence_temporal_visdrone_design_below_140_"
                    "positive_and_73_negative_group_minima"
                ),
            },
            "model_selection": {
                "estimand": "group_equal_critical_action_plus_object_exact_v2",
                "weights": {
                    "critical_action_accuracy": 0.5,
                    "objects_exact_accuracy": 0.5,
                },
                "tie_break": [
                    "selection_score_desc",
                    "critical_action_accuracy_desc",
                    "objects_exact_accuracy_desc",
                    "full_path_latency_p95_ms_asc",
                    "model_id_asc",
                ],
            },
        },
        "hardware": {
            "radio_tx_j_per_byte": 0.0,
            "radio_rx_j_per_byte": 0.0,
        },
    }


class ConfirmatoryLedgerOwnershipContractTests(unittest.TestCase):
    def test_runner_requires_an_existing_open_ledger_and_never_creates_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ctx = _ctx(root)
            protocol = root / "frozen_protocol.json"
            protocol.write_text(json.dumps({
                "protocol_hash": "a" * 64,
                "decisions": {"selection": {}},
            }), encoding="utf-8")
            ledger = ctx.run_root / "test_access_ledger.json"

            with self.assertRaises((StudyGateError, UntouchedTestAccessError)):
                runner.run_confirmatory_test_once(
                    ctx,
                    _cfg(),
                    pd.DataFrame({"split": ["test"]}),
                    traces=None,
                    protocol=protocol,
                    model_lock=root / "MODEL_LOCK.json",
                    device="cuda:0",
                )
            self.assertFalse(ledger.exists())

    def test_validation_runner_rejects_wrong_temporal_group_trace_factorization(self) -> None:
        source = inspect.getsource(runner.run_validation_network_and_controller)
        self.assertIn(
            '"gate": "exact_validation_temporal_groups"', source,
        )
        self.assertIn(
            "validation_temporal_groups == registered_validation_temporal_groups",
            source,
        )
        self.assertIn(
            'registered_temporal_sampling_spec(\n'
            '            _section(cfg, "benchmark"), "validation",',
            source,
        )
        self.assertIn(
            'validation_trace_realizations == int(', source,
        )
        self.assertNotIn(
            'validation_trace_realizations >= int(', source,
        )
        self.assertIn(
            'len(frozen_test_uplink_traces) == int(', source,
        )
        self.assertNotIn(
            'len(frozen_test_uplink_traces) >= int(', source,
        )

    def test_runner_source_has_no_ledger_mutation_or_status_classification(self) -> None:
        tree = _function_tree(runner.run_confirmatory_test_once)
        called = {
            _leaf_name(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        }
        self.assertNotIn("authorize_test_once", called)
        self.assertNotIn("finalize_test_once", called)
        self.assertNotIn("write_test_access_ledger", called)
        # The notebook owns BLOCKED (StudyGateError) versus FAILED (unexpected)
        # classification and is also the sole ledger finalizer.  Cleanup-only
        # handlers (for example optional torch imports) remain allowed.
        for handler in (
            node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)
        ):
            handler_calls = {
                _leaf_name(node.func)
                for node in ast.walk(handler)
                if isinstance(node, ast.Call)
            }
            self.assertFalse({
                "authorize_test_once", "finalize_test_once",
                "write_test_access_ledger",
            }.intersection(handler_calls))
            handler_literals = {
                node.value for node in ast.walk(handler)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
            self.assertFalse({"BLOCKED", "FAILED"}.intersection(handler_literals))


class RunnerCallContractTests(unittest.TestCase):
    def test_h5_not_applicable_status_is_exact_and_consistent(self) -> None:
        expected = "NOT_APPLICABLE_NO_TARGET_EVIDENCE"
        self.assertEqual(runner.H5_NOT_APPLICABLE_STATUS, expected)

        cfg = _cfg()
        cfg["evaluation"]["not_applicable_hypotheses"] = ["H5"]
        status = runner.evaluate_optional_embedded_h5(
            handoff={}, cfg=cfg, raw_trial_path=None,
        )
        self.assertEqual(status.to_dict("records"), [{
            "hypothesis": "H5",
            "execution_status": expected,
            "claim_supported": False,
            "evidence_complete": False,
            "detail": (
                "Target-device generalization was not registered for the "
                "VisDrone-only paper scope"
            ),
        }])

        hypothesis_source = inspect.getsource(
            runner._recompute_hypotheses_from_frozen_results_legacy
        )
        completion_source = inspect.getsource(runner.run_confirmatory_test_once)
        self.assertIn(
            '"execution_status": H5_NOT_APPLICABLE_STATUS',
            hypothesis_source,
        )
        self.assertIn('"h5_status": H5_NOT_APPLICABLE_STATUS', completion_source)

    def test_payload_video_and_semantic_threshold_are_single_source(self) -> None:
        cfg = _cfg()
        self.assertEqual(
            runner._video_payload_arguments(cfg),
            {"video_crfs": [23, 35], "video_fps": 5},
        )
        self.assertEqual(
            runner._image_payload_arguments(cfg),
            {
                "image_codecs": ["jpeg", "webp", "avif"],
                "image_qualities": [25, 50, 80],
                "image_scales": [0.5, 1.0],
            },
        )
        self.assertEqual(runner._h4_high_confidence_threshold(cfg), 0.80)
        for function in (
            runner.run_validation_perception_and_frontier,
            runner.run_confirmatory_test_once,
        ):
            frontier_calls = _calls(
                function, "experiment2_representation_frontier_sharded",
            )
            self.assertGreaterEqual(len(frontier_calls), 1)
            for call in frontier_calls:
                self.assertTrue(any(
                    keyword.arg is None
                    and isinstance(keyword.value, ast.Call)
                    and _leaf_name(keyword.value.func) == "_image_payload_arguments"
                    for keyword in call.keywords
                ))
                self.assertTrue(any(
                    keyword.arg is None
                    and isinstance(keyword.value, ast.Call)
                    and _leaf_name(keyword.value.func) == "_video_payload_arguments"
                    for keyword in call.keywords
                ))
        for function in (
            runner.run_validation_network_and_controller,
            runner.run_confirmatory_test_once,
        ):
            for call_name in (
                "run_paired_policy_trials_sharded",
                "reduce_policy_trial_partitions",
            ):
                calls = _calls(function, call_name)
                self.assertEqual(len(calls), 1)
                threshold = [
                    keyword.value for keyword in calls[0].keywords
                    if keyword.arg == "semantic_acceptance_threshold"
                ]
                self.assertEqual(len(threshold), 1)
                self.assertIsInstance(threshold[0], ast.Call)
                self.assertEqual(
                    _leaf_name(threshold[0].func),
                    "_h4_high_confidence_threshold",
                )

    def test_all_feature_cache_calls_match_the_helper_signature(self) -> None:
        allowed = set(inspect.signature(runner._resolve_feature_cache).parameters)
        calls = _calls(runner.run_validation_perception_and_frontier, "_resolve_feature_cache")
        calls += _calls(runner.run_confirmatory_test_once, "_resolve_feature_cache")
        self.assertGreaterEqual(len(calls), 2)
        for call in calls:
            supplied = {keyword.arg for keyword in call.keywords if keyword.arg is not None}
            self.assertFalse(
                supplied.difference(allowed),
                f"Unknown _resolve_feature_cache keyword(s): {sorted(supplied.difference(allowed))}",
            )

    def test_custom_feature_cache_producer_receives_explicit_control_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ctx = _ctx(root)
            lock = root / "FEATURE_HEAD_LOCK.json"
            lock.write_text("{}", encoding="utf-8")
            cfg = _cfg(feature_lock=lock)
            manifest = pd.DataFrame([{
                "sample_id": "s1",
                "mission_profile": "general_mobility",
                "image_path": str(root / "image.jpg"),
            }])
            observed: dict[str, object] = {}

            def producer(**kwargs: object) -> pd.DataFrame:
                observed.update(kwargs)
                return pd.DataFrame([{"sample_id": "s1", "variant": "int8"}])

            fake_encoder = object()
            fake_head = SimpleNamespace(revision="head-revision")
            explicit_context = {"s1": object()}
            with (
                mock.patch.object(runner, "ResNet50Layer3Encoder", return_value=fake_encoder),
                mock.patch.object(runner, "load_frozen_feature_head", return_value=fake_head),
                mock.patch.object(runner, "_receiver_control_contexts", return_value=explicit_context),
                mock.patch.object(runner, "validate_registered_feature_cache"),
            ):
                cache, artifact, returned_lock = runner._resolve_feature_cache(
                    producer,
                    ctx=ctx,
                    cfg=cfg,
                    phase_manifest=manifest,
                    device="cuda:0",
                    phase="validation",
                    frozen_head_lock=lock,
                )

            self.assertEqual(len(cache), 1)
            self.assertEqual(returned_lock, lock.resolve())
            self.assertTrue(artifact.is_file())
            self.assertIs(observed["manifest"], manifest)
            self.assertIs(observed["control_contexts"], explicit_context)
            self.assertIs(observed["encoder"], fake_encoder)
            self.assertIs(observed["receiver_head"], fake_head)
            self.assertNotIn("full_manifest", observed)
            self.assertNotIn("gold_manifest", observed)

    def test_gpu_board_only_zero_radio_energy_reaches_policy_trials(self) -> None:
        cfg = _cfg()
        self.assertEqual(runner._radio_energy_arguments(cfg), {
            "radio_tx_j_per_byte": 0.0,
            "radio_rx_j_per_byte": 0.0,
            "radio_energy_included": False,
        })
        for function in (
            runner.run_validation_network_and_controller,
            runner.run_confirmatory_test_once,
        ):
            calls = _calls(function, "run_paired_policy_trials_sharded")
            self.assertEqual(len(calls), 1)
            self.assertTrue(any(
                keyword.arg is None
                and isinstance(keyword.value, ast.Call)
                and _leaf_name(keyword.value.func) == "_radio_energy_arguments"
                for keyword in calls[0].keywords
            ))


class FrozenSelectionContractTests(unittest.TestCase):
    def test_temporal_experiment1_keeps_one_compact_route_during_detector_fallback(self) -> None:
        for scientific_runner in (
            runner.run_validation_perception_and_frontier,
            runner.run_confirmatory_test_once,
        ):
            with self.subTest(runner=scientific_runner.__name__):
                self.assertEqual(
                    len(_calls(scientific_runner, "_require_selected_compact_model")),
                    1,
                )
                self.assertEqual(
                    len(_calls(
                        scientific_runner,
                        "_require_exact_temporal_experiment1_routes",
                    )),
                    1,
                )

        selected_compact = "smolvlm2_2b"
        locked_models = [
            {"model_id": "qwen3_vl_2b", "role": "compact"},
            {"model_id": selected_compact, "role": "compact"},
            {"model_id": "qwen3_vl_8b_upper_bound", "role": "upper_bound"},
        ]
        selection = {
            # Deployment may legitimately fall back while temporal Experiment 1
            # retains a separately frozen, validation-ranked compact comparison.
            "selected_model": "detector_only",
            "selected_compact_model": selected_compact,
            "best_compact_vlm": selected_compact,
            "calibration": {
                "detector_only": {"temperature": 1.0},
                selected_compact: {"temperature": 1.1},
            },
            "selection_summary": [
                {"model_id": "detector_only"},
                {"model_id": selected_compact},
            ],
            "temporal_experiment1_model_ids": [
                "detector_only", selected_compact,
            ],
            "temporal_experiment1_model_count": 2,
        }
        self.assertEqual(
            runner._require_selected_compact_model(selection, locked_models),
            selected_compact,
        )

        detail = pd.DataFrame({
            "model_id": [
                "detector_only", "detector_only",
                selected_compact, selected_compact,
            ],
        })
        summary = pd.DataFrame({
            "model_id": ["detector_only", selected_compact],
        })
        runner._require_exact_temporal_experiment1_routes(
            detail, summary, selected_compact_model=selected_compact,
        )

        invalid_routes = {
            "missing compact": ["detector_only"],
            "reversed order": [selected_compact, "detector_only"],
            "extra upper bound": [
                "detector_only", selected_compact,
                "qwen3_vl_8b_upper_bound",
            ],
            "wrong compact": ["detector_only", "qwen3_vl_2b"],
        }
        for label, model_ids in invalid_routes.items():
            with self.subTest(label=label):
                invalid = pd.DataFrame({"model_id": model_ids})
                with self.assertRaisesRegex(
                    StudyGateError,
                    "Temporal Experiment 1 must contain exactly detector_only",
                ):
                    runner._require_exact_temporal_experiment1_routes(
                        invalid,
                        summary,
                        selected_compact_model=selected_compact,
                    )

        invalid_selection = dict(selection)
        invalid_selection["selected_compact_model"] = (
            "qwen3_vl_8b_upper_bound"
        )
        with self.assertRaisesRegex(
            StudyGateError, "Frozen selected compact model",
        ):
            runner._require_selected_compact_model(
                invalid_selection, locked_models,
            )

    def test_temporal_trace_covers_deadline_and_state_expiry_grace(self) -> None:
        manifest = pd.DataFrame([
            {
                "sample_id": f"s-{index}",
                "sequence_id": "sequence-1",
                "mission_profile": "general_mobility",
                "capture_time_us": index * 200_000,
                "trace_time_us": index * 200_000,
                "deadline_us": index * 200_000 + 2_000_000,
                "frame_index": index,
            }
            for index in range(64)
        ])

        def traces(duration_s: float):
            return [
                LinkTrace(
                    "trace-1", direction, TraceSemantics.RAW_SERVICE,
                    (LinkSegment(
                        0, int(duration_s * 1_000_000), 5_000_000,
                        10_000, 0, 0.0,
                    ),),
                    EndBehavior.TERMINATE,
                )
                for direction in (Direction.UPLINK, Direction.DOWNLINK)
            ]

        with self.assertRaisesRegex(StudyGateError, "exceed trace coverage"):
            runner._rebase_registered_episodes(
                manifest, traces(16.0), state_expiry_grace_us=2_000_000,
            )
        rebased, audit = runner._rebase_registered_episodes(
            manifest, traces(18.0), state_expiry_grace_us=2_000_000,
        )
        self.assertEqual(int(rebased["trace_time_us"].max()), 12_600_000)
        self.assertEqual(int(audit["last_state_expiry_us"].max()), 16_600_000)
        self.assertTrue(audit["passed"].all())
        self.assertTrue(audit["scientific_episode_coverage_passed"].all())

    def test_h1_group_bootstrap_is_invariant_to_uniform_grid_or_seed_duplication(self) -> None:
        rows = []
        for group_index in range(6):
            for profile in ("general_mobility", "heavy_vehicle"):
                for policy, value in (("semantic", 0.8), ("image", 0.75)):
                    rows.append({
                        "visual_group_id": f"g{group_index}",
                        "mission_profile": profile, "policy": policy,
                        "utility": value + group_index * 0.001,
                        "fixed_grid_rows": 100,
                    })
        collapsed = pd.DataFrame(rows)
        base = runner._group_only_paired_bootstrap(
            collapsed, "semantic", "image", endpoint="utility",
            repetitions=300, seed=19, confidence=0.95,
        )
        # Repeating every registered condition or stochastic repeat changes
        # only the audited exposure count after within-grid averaging.
        duplicated_exposure = collapsed.copy()
        duplicated_exposure["fixed_grid_rows"] *= 2
        repeated = runner._group_only_paired_bootstrap(
            duplicated_exposure, "semantic", "image", endpoint="utility",
            repetitions=300, seed=19, confidence=0.95,
        )
        self.assertEqual(base, repeated)
        self.assertEqual(
            base["method"], "visual_group_bootstrap_over_registered_fixed_grid_v1",
        )

    def test_dynamic_cartesian_audit_rejects_missing_duplicate_and_extra_cells(self) -> None:
        manifest = pd.DataFrame({"sample_id": ["s1", "s2"]})
        traces = []
        for trace_id in ("t1", "t2"):
            traces.extend([
                constant_trace(trace_id, Direction.UPLINK, 5.0, 10.0, 0.0),
                constant_trace(trace_id, Direction.DOWNLINK, 5.0, 10.0, 0.0),
            ])
        policies = [SimpleNamespace(name="p1"), SimpleNamespace(name="p2")]
        rows = [
            {
                "sample_id": sample_id, "trace_id": trace_id,
                "replicate_seed": seed, "policy": policy.name,
            }
            for sample_id in ("s1", "s2")
            for trace_id in ("t1", "t2")
            for seed in (11, 13)
            for policy in policies
        ]
        complete = pd.DataFrame(rows)
        audit = runner._dynamic_cartesian_coverage_audit(
            complete, manifest, traces, policies, (11, 13),
        )
        self.assertTrue(audit["passed"].all())

        missing = runner._dynamic_cartesian_coverage_audit(
            complete.iloc[:-1], manifest, traces, policies, (11, 13),
        )
        self.assertFalse(missing.loc[
            missing["gate"].eq("exact_cartesian_rows"), "passed"
        ].item())
        duplicated = runner._dynamic_cartesian_coverage_audit(
            pd.concat([complete, complete.iloc[[0]]], ignore_index=True),
            manifest, traces, policies, (11, 13),
        )
        self.assertFalse(duplicated.loc[
            duplicated["gate"].eq("no_duplicate_cartesian_keys"), "passed"
        ].item())
        extra = complete.copy()
        extra.loc[len(extra)] = {
            "sample_id": "extra", "trace_id": "t1",
            "replicate_seed": 11, "policy": "p1",
        }
        extra_audit = runner._dynamic_cartesian_coverage_audit(
            extra, manifest, traces, policies, (11, 13),
        )
        self.assertFalse(extra_audit.loc[
            extra_audit["gate"].eq("sample_domain"), "passed"
        ].item())

    def test_h3_uses_episode_equal_two_way_clustered_false_alert_endpoint(self) -> None:
        cfg = _cfg()

        def trials(positive_groups: int, one_false_episode: bool = False) -> pd.DataFrame:
            rows = []
            for policy in ("proposed_full", "channel_only_hysteresis"):
                for index in range(positive_groups):
                    rows.append({
                        "policy": policy, "visual_group_id": f"positive-{index}",
                        "mission_profile": "general_mobility", "trace_id": "t1",
                        "replicate_seed": 11, "critical": True, "false_alert": False,
                    })
                for index in range(73):
                    for trace_id in ("t1", "t2"):
                        rows.append({
                            "policy": policy,
                            "visual_group_id": f"negative-{index}",
                            "mission_profile": "general_mobility",
                            "trace_id": trace_id, "replicate_seed": 11,
                            "critical": False,
                            "false_alert": bool(
                                one_false_episode and policy == "proposed_full"
                                and index == 0 and trace_id == "t1"
                            ),
                        })
            return pd.DataFrame(rows)

        passed = runner._h3_estimability_gate(trials(140), cfg)
        self.assertTrue(passed["estimable"])
        self.assertTrue(passed["false_alert_gate_pass"])
        self.assertEqual(
            passed["estimand"],
            "group_trace_equal_negative_decision_false_alert_rate_v1",
        )
        self.assertLessEqual(
            passed["proposed_false_alert_cluster_bootstrap_upper_95"], 0.05,
        )

        underpowered = runner._h3_estimability_gate(trials(139), cfg)
        self.assertFalse(underpowered["estimable"])
        self.assertFalse(underpowered["false_alert_gate_pass"])

        sparse = trials(140, one_false_episode=True)
        sparse_gate = runner._h3_estimability_gate(sparse, cfg)
        self.assertAlmostEqual(
            sparse_gate["proposed_false_alert_rate"], 1 / (73 * 2), places=12,
        )
        # A max-across-campaign estimator would instead be 1/73.
        self.assertNotAlmostEqual(
            sparse_gate["proposed_false_alert_rate"], 1 / 73, places=12,
        )

        negative = sparse.loc[~sparse["critical"]].copy()
        duplicate_frames = pd.concat([sparse, negative], ignore_index=True)
        duplicated_gate = runner._h3_estimability_gate(duplicate_frames, cfg)
        self.assertAlmostEqual(
            duplicated_gate["proposed_false_alert_rate"],
            sparse_gate["proposed_false_alert_rate"], places=12,
        )
        self.assertAlmostEqual(
            duplicated_gate["proposed_false_alert_cluster_bootstrap_upper_95"],
            sparse_gate["proposed_false_alert_cluster_bootstrap_upper_95"], places=12,
        )

        duplicated_seed = negative.copy()
        duplicated_seed["replicate_seed"] = 13
        seed_expanded = pd.concat([sparse, duplicated_seed], ignore_index=True)
        seed_gate = runner._h3_estimability_gate(seed_expanded, cfg)
        self.assertAlmostEqual(
            seed_gate["proposed_false_alert_rate"],
            sparse_gate["proposed_false_alert_rate"], places=12,
        )
        self.assertAlmostEqual(
            seed_gate["proposed_false_alert_cluster_bootstrap_upper_95"],
            sparse_gate["proposed_false_alert_cluster_bootstrap_upper_95"], places=12,
        )

    def test_h2_pairwise_budget_requires_five_percent_wire_and_combined_energy_match(self) -> None:
        cfg = _cfg()
        rows = []
        for policy, wire, energy in (
            ("proposed_full", 100.0, 10.0),
            ("matched", 100.0, 10.0),
            ("unmatched", 80.0, 8.0),
        ):
            rows.append({
                "policy": policy, "visual_group_id": "group-1",
                "trace_id": "trace-1", "replicate_seed": 1,
                "capture_us": 0, "total_wire_bytes": wire,
                "energy_j": energy,
                "representation_energy_j": energy * 0.4,
                "detector_energy_j": energy * 0.2,
                "context_model_energy_j": energy * 0.2,
                "receiver_energy_j_charged": energy * 0.2,
            })
        report = runner._h2_pairwise_budget_match(
            pd.DataFrame(rows), ("matched", "unmatched"), cfg,
        ).set_index("reference")
        self.assertTrue(bool(report.loc["matched", "passed"]))
        self.assertFalse(bool(report.loc["unmatched", "passed"]))
        self.assertEqual(
            report.loc["matched", "energy_scope"],
            "combined_sender_and_ground_gpu_board_nvml_compute_only_excludes_radio_system",
        )

    def test_unmatched_h2_resources_do_not_block_study_completion(self) -> None:
        pairwise = pd.DataFrame([
            {
                "reference": "matched",
                "wire_relative_delta": 0.0,
                "energy_relative_delta": 0.0,
                "wire_match_pass": True,
                "energy_match_pass": True,
                "common_cap_pass": True,
                "passed": True,
            },
            {
                "reference": "unmatched",
                "wire_relative_delta": 0.25,
                "energy_relative_delta": 0.20,
                "wire_match_pass": False,
                "energy_match_pass": False,
                "common_cap_pass": True,
                "passed": False,
            },
        ])
        resources = pd.DataFrame([
            {
                "policy": policy,
                "wire_bytes_per_s": 100.0,
                "energy_j_per_s": 1.0,
                "wire_cap_bytes_per_s": 1050.0,
                "energy_cap_j_per_s": 1050.0,
                "wire_pass": True,
                "energy_pass": True,
                "eligible": True,
            }
            for policy in ("proposed_full", "matched", "unmatched")
        ])
        gate = runner._validation_h2_resource_audit_registration(
            pairwise, resources, ("matched", "unmatched"),
            ("proposed_full", "matched", "unmatched"),
        )
        self.assertTrue(bool(gate["passed"]))
        self.assertFalse(bool(gate["pairwise_all_matched"]))
        self.assertIn("nonblocking H2 claim-ineligibility", gate["detail"])

    def test_model_selection_ranks_eligible_candidates_by_frozen_group_equal_estimand(self) -> None:
        cfg = _cfg()
        evaluation = cfg["evaluation"]
        assert isinstance(evaluation, dict)
        evaluation.update({
            "calibration_fitting_estimand": "visual_group_equal_temperature_nll_v1",
            "minimum_calibration_independent_groups": 10,
            "calibration_incorrect_group_diagnostic_target": 50,
            "calibration_bootstrap_repetitions": 50,
            "calibration_bootstrap_upper_quantile": 0.95,
            "calibration_ece_upper_limit": 1.0,
            "calibration_priority_ece_upper_limit": 1.0,
            "max_vlm_full_path_p95_latency_ms": 1000,
            "max_vlm_peak_memory_mb": 12000,
            "min_schema_valid_rate": 0.98,
            "min_semantic_valid_rate": 0.98,
            "min_priority_recall": 0.5,
            "minimum_vlm_context_gain": 0.03,
        })
        rows = []
        specifications = {
            "detector_only": (0.5, 0.5, 100.0),
            "compact_ineligible": (1.0, 1.0, 2000.0),
            "compact_eligible": (0.8, 0.8, 100.0),
        }
        for model_id, (critical_action, object_exact, latency) in specifications.items():
            for split in ("calibration", "validation"):
                for index in range(10):
                    action_correct = (
                        index < int(round(10 * critical_action))
                        if split == "validation" else bool(index % 2)
                    )
                    metric_scope = (
                        runner.DETECTOR_ONLY_METRIC_SCOPE
                        if model_id == "detector_only"
                        else runner.COMPOSED_PIPELINE_METRIC_SCOPE
                    )
                    rows.append({
                        "model_id": model_id, "split": split,
                        "confidence": 0.5,
                        "action_correct": action_correct,
                        # Deliberately disagree with the v2 endpoints so this
                        # test fails if the retired event/relation estimand is
                        # accidentally restored.
                        "event_correct": not action_correct,
                        "objects_exact": (
                            index < int(round(10 * object_exact))
                            if split == "validation" else bool(index % 2)
                        ),
                        "relations_exact": not bool(
                            index < int(round(10 * object_exact))
                        ),
                        "critical": True,
                        "visual_group_id": f"{split}-group-{index}",
                        "schema_valid": True, "semantic_valid": True,
                        "authoritative_consistent": True,
                        "full_path_latency_ms": latency,
                        "peak_memory_mb": 1000.0,
                        "action_metric_scope": metric_scope,
                        "event_metric_scope": metric_scope,
                        "object_metric_scope": metric_scope,
                        "relation_metric_scope": metric_scope,
                    })

        class Calibrator:
            temperature = 1.0

            def fit(self, confidence, labels, *, sample_weight=None):
                del confidence, labels
                self.sample_weight = pd.Series(sample_weight, dtype=float)
                return self

            def transform(self, confidence):
                return pd.to_numeric(confidence).to_numpy(float)

        locked = [
            {"model_id": "compact_ineligible", "role": "compact"},
            {"model_id": "compact_eligible", "role": "compact"},
            {"model_id": "upper", "role": "upper_bound"},
        ]
        with mock.patch.object(
            runner, "TemperatureCalibrator", return_value=Calibrator(),
        ):
            selection = runner._fit_validation_selection(
                pd.DataFrame(rows), cfg, locked,
            )
        self.assertEqual(selection["unconstrained_best_compact_vlm"], "compact_ineligible")
        self.assertEqual(selection["best_eligible_compact_vlm"], "compact_eligible")
        self.assertEqual(selection["selected_model"], "compact_eligible")
        self.assertEqual(selection["selected_compact_model"], "compact_eligible")
        self.assertEqual(
            selection["temporal_experiment1_model_ids"],
            ["detector_only", "compact_eligible"],
        )
        self.assertEqual(selection["temporal_experiment1_model_count"], 2)
        summary = pd.DataFrame(selection["selection_summary"]).set_index("model_id")
        self.assertAlmostEqual(summary.loc["compact_eligible", "selection_score"], 0.8)
        self.assertEqual(
            summary.loc["compact_eligible", "selection_metric_scope"],
            runner.COMPOSED_PIPELINE_METRIC_SCOPE,
        )
        self.assertFalse(bool(summary.loc[
            "compact_eligible", "free_form_vlm_packet_authorship_claim_eligible"
        ]))
        self.assertEqual(
            selection["model_selection"]["estimand"],
            "group_equal_critical_action_plus_object_exact_v2",
        )
        self.assertEqual(
            selection["model_selection"]["claim_scope"],
            runner.COMPOSED_PIPELINE_CLAIM_SCOPE,
        )
        self.assertEqual(
            selection["calibration"]["compact_eligible"]["fitting_estimand"],
            "visual_group_equal_temperature_nll_v1",
        )
        gates = pd.DataFrame(selection["calibration_gate"]).set_index("model_id")
        self.assertTrue(bool(gates.loc["compact_eligible", "passed"]))
        self.assertFalse(bool(gates.loc[
            "compact_eligible", "calibration_incorrect_group_diagnostic_met"
        ]))
        self.assertFalse(bool(gates.loc[
            "compact_eligible", "calibration_incorrect_group_diagnostic_is_blocking"
        ]))
        self.assertTrue(all(
            float(item["confidence_threshold"]) == 0.80
            for item in selection["calibration"].values()
        ))

    def test_confirmatory_runner_consumes_frozen_selection_without_refitting(self) -> None:
        source = textwrap.dedent(inspect.getsource(runner.run_confirmatory_test_once))
        tree = ast.parse(source)
        called = {
            _leaf_name(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        }
        self.assertNotIn("build_causal_candidate_profiles", called)
        self.assertIn("_read_frozen_selection", called)
        self.assertIn("select_experiment3_candidate_variants", called)
        variant_calls = _calls(
            runner.run_confirmatory_test_once,
            "select_experiment3_candidate_variants",
        )
        self.assertTrue(all(
            any(keyword.arg == "frozen_spec" for keyword in call.keywords)
            for call in variant_calls
        ))
        for required in (
            "policy_spec",
            "feature_head",
            "candidate_profiles",
            "exp3_variant_spec",
        ):
            self.assertIn(required, source)

    def test_experiment6_uses_registered_config_seeds(self) -> None:
        tree = _function_tree(runner.run_confirmatory_test_once)
        assignments: dict[str, ast.AST] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = node.value
                for target in targets:
                    if isinstance(target, ast.Name) and value is not None:
                        assignments[target.id] = value

        def is_cfg_seed_expression(value: ast.AST) -> bool:
            if isinstance(value, ast.Call):
                return (
                    _leaf_name(value.func) in {"_seeds", "_exp6_seeds"}
                    and any(isinstance(arg, ast.Name) and arg.id == "cfg" for arg in value.args)
                )
            if isinstance(value, ast.Name) and value.id in assignments:
                return is_cfg_seed_expression(assignments[value.id])
            return False

        for name in (
            "experiment6_registered_campaign",
            "experiment6_state_strategy_comparison",
        ):
            calls = _calls(runner.run_confirmatory_test_once, name)
            self.assertEqual(len(calls), 1)
            seed_keywords = [keyword for keyword in calls[0].keywords if keyword.arg == "seeds"]
            self.assertEqual(len(seed_keywords), 1)
            self.assertTrue(
                is_cfg_seed_expression(seed_keywords[0].value),
                f"{name} seeds must come from the frozen configuration",
            )

    def test_registered_visdrone_mission_profiles_are_accepted_and_paired(self) -> None:
        cfg = _cfg()
        profiles = runner._priority_profiles(cfg)
        self.assertEqual(set(profiles), set(VISDRONE_PROFILES))
        self.assertEqual(profiles["heavy_vehicle"], ("truck", "bus"))

        rows = []
        for profile in VISDRONE_PROFILES:
            rows.append({
                "sample_id": f"source-1:{profile}",
                "source_sample_id": "source-1",
                "split": "validation",
                "component": "det_val",
                "visual_group_id": "det_val:group-1",
                "mission_profile": profile,
                "gold_action": "RESPOND",
                "image_path": "/registered/source-1.jpg",
            })
        checked = runner._require_registered_phase_sample(
            pd.DataFrame(rows), cfg, expected_splits=("validation",),
        )
        self.assertEqual(len(checked), 3)
        self.assertEqual(set(checked["mission_profile"]), set(VISDRONE_PROFILES))


if __name__ == "__main__":
    unittest.main()
