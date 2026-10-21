from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from semuav.evaluation import resource_summary, two_way_clustered_paired_bootstrap
from semuav.experiments import experiment4_controller_behavior
from semuav.policy_reducer import (
    REQUIRED_COLUMNS,
    compact_two_way_clustered_paired_bootstrap,
    load_verified_partition_index,
    reduce_policy_trial_partitions,
)
from semuav.registry import StudyGateError, hash_file
from semuav.visdrone_runner import (
    _compact_dynamic_hypothesis_reports,
    _visdrone_experiment5,
)


POLICIES = (
    "proposed_full",
    "ablate_evidence_escalation",
    "heuristic_lookahead_reference",
)


def _compact_report_cfg() -> dict[str, object]:
    return {
        "seed": 20260823,
        "seeds": [11, 13],
        "split": {"min_critical_test_events": 140},
        "benchmark": {
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
        "evaluation": {
            "primary_margin": 0.03,
            "bootstrap_repetitions": 101,
            "primary_wire_budget_bytes_per_s": 1000.0,
            "primary_energy_budget_j_per_s": 1000.0,
            "h2_matched_budget_relative_tolerance": 0.05,
            "mission_priority_positive_units": 140,
            "mission_priority_negative_units": 73,
            "false_alert_limit": 0.05,
            "energy_claim_scope": (
                "combined_sender_and_ground_gpu_board_nvml_compute_only_"
                "excludes_radio_system"
            ),
            "radio_energy_included": False,
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
        },
        "hardware": {
            "radio_tx_j_per_byte": 0.0,
            "radio_rx_j_per_byte": 0.0,
        },
    }


def _small_temporal_manifest() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "visual_group_id": group,
            "mission_profile": "registered_temporal",
            "temporal_observation_index": observation,
        }
        for group in ("group-1", "group-2")
        for observation in range(3)
    ])


def _trial_rows() -> pd.DataFrame:
    rows = []
    for trace_number, trace_id in enumerate(("trace-a", "trace-b")):
        for seed in (11, 13):
            for group_number, visual_group in enumerate(("group-1", "group-2")):
                for sample_number in range(3):
                    sample_id = f"{visual_group}-sample-{sample_number}"
                    capture_us = sample_number * 200_000
                    for policy_number, policy in enumerate(POLICIES):
                        critical = sample_number == 0
                        proposed = policy == "proposed_full"
                        reference = policy == "heuristic_lookahead_reference"
                        correct = bool(reference or proposed or sample_number != 0)
                        timely = bool(correct and not (
                            proposed and trace_id == "trace-b" and sample_number == 2
                        ))
                        delivered = not (
                            policy == "ablate_evidence_escalation"
                            and trace_id == "trace-b" and sample_number == 2
                        )
                        mode = (
                            "L2_SEMANTICS"
                            if policy != "heuristic_lookahead_reference"
                            else "L5_IMAGE_OR_CLIP"
                        )
                        score = (
                            0.98 - 0.1 * sample_number - 0.01 * (seed == 13)
                            if mode == "L2_SEMANTICS" else np.nan
                        )
                        rows.append({
                            "sample_id": sample_id,
                            "visual_group_id": visual_group,
                            "mission_profile": "urgent" if group_number else "routine",
                            "trace_id": trace_id,
                            "replicate_seed": seed,
                            "policy": policy,
                            "episode_id": f"{visual_group}::profile",
                            "capture_us": capture_us,
                            "action_ready_us": (
                                capture_us + 30_000 + 1000 * policy_number
                                if delivered else np.nan
                            ),
                            "deadline_us": capture_us + 100_000,
                            "decision_us": capture_us,
                            "selected_mode": mode,
                            "variant": "semantic-v1" if mode == "L2_SEMANTICS" else "jpeg-q50",
                            "gold_action": "RESPOND" if critical else "NO_ACTION",
                            "receiver_action": (
                                "RESPOND" if critical and correct else "NO_ACTION"
                            ),
                            "correct_action": correct,
                            "correct_timely_action": timely,
                            "delivered_by_deadline": delivered,
                            "false_alert": False,
                            "critical_miss": bool(critical and not timely),
                            "critical": critical,
                            "total_wire_bytes": 500 + 25 * policy_number + sample_number,
                            "energy_j": 0.02 + 0.001 * policy_number,
                            "calibrated_correctness_probability": score,
                            "high_confidence_semantic_error": bool(
                                mode == "L2_SEMANTICS" and score >= 0.8 and not correct
                            ),
                            "evidence_escalated": proposed and sample_number == 0,
                            "application_payload_bytes": 100 + sample_number,
                            "full_message_received": delivered,
                            "state_valid": delivered,
                            "state_kind": "none",
                            "apply_status": "applied" if delivered else "stale",
                            "original_uplink_bytes": 600 + sample_number,
                            "retry_uplink_bytes": 10 * sample_number,
                            "decision_queue_pending_bytes": 20 + sample_number,
                            "decision_queue_outstanding_unacked_bytes": 5 + sample_number,
                            "decision_queue_messages": sample_number,
                            "decision_queue_oldest_age_ms": float(sample_number * 2),
                            "decision_backlog_bytes": 25 + 2 * sample_number,
                            "decision_sender_sequence_gap": int(not delivered),
                            "episode_outage_count": 0,
                            "episode_outages_recovered": 0,
                            "episode_outages_unrecovered": 0,
                            "outage_recovery_event": False,
                            "outage_recovery_latency_ms": np.nan,
                            "representation_energy_j": 0.010,
                            "detector_energy_j": 0.004,
                            "context_model_energy_j": 0.003,
                            "receiver_energy_j_charged": 0.003 + 0.001 * policy_number,
                        })
    frame = pd.DataFrame(rows)
    assert not set(REQUIRED_COLUMNS).difference(frame.columns)
    return frame


def _write_partitions(
    root: Path,
    frame: pd.DataFrame,
    *,
    by: tuple[str, ...] = ("trace_id", "replicate_seed"),
) -> pd.DataFrame:
    records = []
    for number, (key, group) in enumerate(frame.groupby(list(by), sort=True)):
        path = root / f"partition-{number}.csv.gz"
        group.to_csv(path, index=False, compression="gzip")
        key = key if isinstance(key, tuple) else (key,)
        values = dict(zip(by, key))
        records.append({
            "stage": f"stage-{number}",
            "artifact_path": str(path),
            "artifact_sha256": hash_file(path),
            "row_count": len(group),
            "trace_id": values.get("trace_id"),
            "replicate_seed": values.get("replicate_seed"),
            "episodes": group["episode_id"].nunique(),
        })
    return pd.DataFrame(records)


class PolicyReducerTests(unittest.TestCase):
    maxDiff = None

    def test_small_reduction_matches_legacy_resource_controller_and_exp5(self) -> None:
        frame = _trial_rows()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = _write_partitions(root, frame)
            result = reduce_policy_trial_partitions(
                index, root / "reduced", chunk_size=7, h4_target_coverage=0.5,
            )

            expected_resource = resource_summary(frame).sort_values("policy").reset_index(drop=True)
            observed_resource = result.frames["resource_summary"].sort_values("policy").reset_index(drop=True)
            assert_frame_equal(
                observed_resource[expected_resource.columns], expected_resource,
                check_dtype=False, rtol=1e-12, atol=1e-12,
            )

            _, expected_controller = experiment4_controller_behavior(frame)
            observed_controller = result.frames["controller_summary"]
            expected_controller = expected_controller.set_index("policy")
            observed_controller = observed_controller.set_index("policy")
            comparison = [
                "episodes", "decisions", "correct_timely_actions",
                "total_energy_j", "mode_switches", "mode_switch_rate",
                "oscillations_aba", "oscillation_rate", "dwell_runs",
                "completed_dwell_runs", "mean_dwell_decisions",
                "median_dwell_decisions", "mean_completed_dwell_ms",
                "median_completed_dwell_ms", "p95_completed_dwell_ms",
                "aborted_generated_payload_bytes", "wasted_uplink_wire_bytes",
                "semantic_predictions", "semantic_accepted",
                "semantic_selective_coverage", "semantic_selective_risk",
                "high_confidence_semantic_error_rate",
            ]
            assert_frame_equal(
                observed_controller[comparison], expected_controller[comparison],
                check_dtype=False, rtol=1e-12, atol=1e-12,
            )

            _, expected_exp5 = _visdrone_experiment5(frame)
            observed_exp5 = result.frames["exp5_summary"]
            keys = ["subset", "policy"]
            expected_exp5 = expected_exp5.sort_values(keys).reset_index(drop=True)
            observed_exp5 = observed_exp5.sort_values(keys).reset_index(drop=True)
            metrics = [
                *keys, "trials", "independent_visual_groups",
                "correct_timely_rate", "false_alert_rate", "critical_miss_rate",
                "mean_wire_bytes", "mean_gpu_board_energy_j",
            ]
            assert_frame_equal(
                observed_exp5[metrics], expected_exp5[metrics],
                check_dtype=False, rtol=1e-12, atol=1e-12,
            )

            controller = observed_controller
            self.assertEqual(
                int(controller.loc["proposed_full", "high_confidence_semantic_rows"]),
                16,
            )
            self.assertEqual(
                int(controller.loc["proposed_full", "high_confidence_semantic_errors"]),
                0,
            )
            self.assertEqual(
                float(controller.loc[
                    "proposed_full", "high_confidence_semantic_error_rate"
                ]),
                0.0,
            )
            self.assertEqual(
                int(controller.loc[
                    "heuristic_lookahead_reference", "high_confidence_semantic_rows"
                ]),
                0,
            )
            self.assertTrue(np.isnan(controller.loc[
                "heuristic_lookahead_reference",
                "high_confidence_semantic_error_rate",
            ]))

            raw_effect = two_way_clustered_paired_bootstrap(
                frame, "proposed_full", "ablate_evidence_escalation",
                repetitions=257, seed=37,
            )
            compact_effect = compact_two_way_clustered_paired_bootstrap(
                result.frames["policy_pair_cells"],
                "proposed_full", "ablate_evidence_escalation",
                endpoint="correct_timely_rate", repetitions=257, seed=37,
            )
            for field in ("estimate", "ci_low", "ci_high", "p_two_sided"):
                self.assertAlmostEqual(
                    float(getattr(raw_effect, field)),
                    float(getattr(compact_effect, field)), places=14,
                )
            self.assertEqual(raw_effect.repetitions, compact_effect.repetitions)
            self.assertEqual(raw_effect.visual_clusters, compact_effect.visual_clusters)
            self.assertEqual(raw_effect.trace_clusters, compact_effect.trace_clusters)

    def test_structural_h2_blanks_confirmatory_inference_and_h4_is_conjunctive(self) -> None:
        frame = _trial_rows()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = reduce_policy_trial_partitions(
                _write_partitions(root, frame), root / "reduced",
                chunk_size=7, h4_target_coverage=0.5,
            )
            reports = _compact_dynamic_hypothesis_reports(
                result.frames,
                {
                    "required_h2_baselines": ["ablate_evidence_escalation"],
                    "validation_h2_unmatched_baselines": [],
                },
                _small_temporal_manifest(), _compact_report_cfg(),
                repetitions=101,
            )
        h2 = reports["h2_report"].iloc[0]
        self.assertEqual(h2["execution_status"], "NOT_ESTIMABLE")
        for field in ("estimate", "ci_low", "ci_high", "p_two_sided", "holm_p"):
            self.assertTrue(pd.isna(h2[field]), field)
        self.assertTrue(pd.isna(h2["exceeds_margin"]))
        self.assertTrue(np.isfinite(h2["descriptive_underpowered_estimate"]))
        self.assertFalse(bool(h2["claim_supported"]))

        h3 = reports["h3_report"].iloc[0]
        self.assertEqual(h3["execution_status"], "NOT_ESTIMABLE")
        self.assertFalse(bool(h3["false_alert_gate_pass"]))
        self.assertIn("false_alert_descriptive_limit_pass", h3.index)

        h4 = reports["h4_report"].iloc[0]
        self.assertEqual(h4["execution_status"], "COMPLETE")
        self.assertIn(
            "not_free_form_vlm_semantic_generation", h4["claim_scope"]
        )
        self.assertEqual(
            h4["semantic_packet_authorship"], "deterministic_host_composer"
        )
        self.assertTrue(bool(h4["detector_proposal_conditioned"]))
        self.assertFalse(bool(
            h4["free_form_vlm_packet_authorship_claim_eligible"]
        ))
        self.assertNotEqual(
            float(h4["matched_coverage_error_reduction"]),
            float(h4["high_confidence_error_reduction"]),
        )
        self.assertEqual(
            bool(h4["claim_supported"]),
            bool(
                h4["matched_coverage_claim_pass"]
                and h4["high_confidence_claim_pass"]
            ),
        )

    def test_zero_high_confidence_denominator_is_not_zero_error(self) -> None:
        frame = _trial_rows()
        semantic = frame["selected_mode"].isin(["L1_ALERT", "L2_SEMANTICS"])
        frame.loc[semantic, "calibrated_correctness_probability"] = 0.10
        frame["high_confidence_semantic_error"] = False
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = reduce_policy_trial_partitions(
                _write_partitions(root, frame), root / "reduced",
                chunk_size=7, h4_target_coverage=0.5,
                semantic_acceptance_threshold=0.80,
            )
            cells = result.frames["h4_matched_coverage_cells"]
            self.assertTrue(cells["high_confidence_semantic_rows"].eq(0).all())
            self.assertTrue(cells["high_confidence_semantic_errors"].eq(0).all())
            self.assertTrue(cells["high_confidence_semantic_error_rate"].isna().all())
            self.assertFalse(cells["high_confidence_semantic_passed"].any())
            reports = _compact_dynamic_hypothesis_reports(
                result.frames,
                {
                    "required_h2_baselines": ["ablate_evidence_escalation"],
                    "validation_h2_unmatched_baselines": [],
                },
                _small_temporal_manifest(), _compact_report_cfg(),
                repetitions=51,
            )
        h4 = reports["h4_report"].iloc[0]
        self.assertEqual(h4["execution_status"], "NOT_ESTIMABLE")
        self.assertEqual(int(h4["high_confidence_group_trace_cells"]), 0)
        self.assertGreater(
            int(h4["required_high_confidence_group_trace_cells"]), 0,
        )
        self.assertFalse(bool(h4["high_confidence_gate_pass"]))
        self.assertFalse(bool(h4["claim_supported"]))
        for field in (
            "matched_coverage_holm_adjusted_p",
            "high_confidence_holm_adjusted_p",
            "holm_adjusted_p",
        ):
            self.assertTrue(pd.isna(h4[field]), field)

    def test_partition_order_and_boundaries_do_not_change_compact_estimands(self) -> None:
        frame = _trial_rows()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_root = root / "first"
            second_root = root / "second"
            first_root.mkdir()
            second_root.mkdir()
            first_index = _write_partitions(first_root, frame)
            # Different physical boundaries: each trace is a partition rather
            # than each trace/seed pair.
            second_index = _write_partitions(second_root, frame, by=("trace_id",))
            first = reduce_policy_trial_partitions(
                first_index.sample(frac=1.0, random_state=3),
                root / "out-first", chunk_size=5, h4_target_coverage=0.5,
            )
            second = reduce_policy_trial_partitions(
                second_index.iloc[::-1], root / "out-second",
                chunk_size=11, h4_target_coverage=0.5,
            )
            compare = (
                "policy_pair_cells", "h3_negative_episode_cells",
                "resource_summary", "resource_rate_summary",
                "controller_summary", "exp5_summary",
                "h4_matched_coverage_cells", "h4_matched_coverage_audit",
            )
            for name in compare:
                assert_frame_equal(
                    first.frames[name].reset_index(drop=True),
                    second.frames[name].reset_index(drop=True),
                    check_dtype=False, rtol=1e-12, atol=1e-12,
                    obj=name,
                )

    def test_partition_column_order_does_not_change_reduction(self) -> None:
        # v3r2 correction: pandas returns usecols in file order; the reducer
        # must bind values by name, not by the file's physical column order.
        frame = _trial_rows()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "natural").mkdir()
            (root / "reversed").mkdir()
            natural = reduce_policy_trial_partitions(
                _write_partitions(root / "natural", frame), root / "out-natural",
                chunk_size=7, h4_target_coverage=0.5,
            )
            reversed_frame = frame[list(reversed(frame.columns))]
            reordered = reduce_policy_trial_partitions(
                _write_partitions(root / "reversed", reversed_frame), root / "out-reversed",
                chunk_size=7, h4_target_coverage=0.5,
            )
            for name in ("policy_pair_cells", "resource_summary", "controller_summary", "exp5_summary"):
                assert_frame_equal(
                    natural.frames[name].reset_index(drop=True),
                    reordered.frames[name].reset_index(drop=True),
                    check_dtype=False, rtol=1e-12, atol=1e-12, obj=name,
                )

    def test_every_trial_read_is_chunked_with_explicit_usecols(self) -> None:
        frame = _trial_rows()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = _write_partitions(root, frame)
            original = pd.read_csv
            calls = []

            def audited_read_csv(*args, **kwargs):
                calls.append(dict(kwargs))
                return original(*args, **kwargs)

            with mock.patch("semuav.policy_reducer.pd.read_csv", side_effect=audited_read_csv):
                result = reduce_policy_trial_partitions(
                    index, root / "reduced", chunk_size=4, h4_target_coverage=0.5,
                )
            self.assertTrue(calls)
            self.assertTrue(all(call.get("chunksize") == 4 for call in calls))
            self.assertTrue(all(call.get("usecols") for call in calls))
            audit = result.frames["io_audit"].iloc[0]
            self.assertLessEqual(int(audit.maximum_observed_chunk_rows), 4)
            self.assertTrue(bool(audit.passed))

    def test_h4_is_spilled_per_trace_without_campaign_selected_rows(self) -> None:
        frame = _trial_rows()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = _write_partitions(root, frame)
            # Any campaign-wide concat would fail this test.  Compact rows are
            # accumulated as records and written per trace.
            with mock.patch.object(pd, "concat", side_effect=AssertionError("no concat")):
                result = reduce_policy_trial_partitions(
                    index, root / "reduced", chunk_size=5, h4_target_coverage=0.5,
                )
            spill = result.frames["h4_trace_spill_index"]
            self.assertEqual(set(spill["trace_id"]), {"trace-a", "trace-b"})
            for row in spill.itertuples(index=False):
                path = Path(row.artifact_path)
                self.assertTrue(path.is_file())
                self.assertEqual(hash_file(path), row.artifact_sha256)
                per_trace = pd.read_csv(path, compression="gzip")
                self.assertEqual(set(per_trace["trace_id"]), {row.trace_id})
                self.assertNotIn("matched_coverage_selected", per_trace.columns)
                self.assertEqual(len(per_trace), int(row.row_count))
            audit = result.frames["io_audit"].iloc[0]
            self.assertEqual(int(audit.h4_trace_spills), 2)
            self.assertFalse(bool(audit.h4_campaign_selected_trial_rows_materialized))
            cells = result.frames["h4_matched_coverage_cells"]
            self.assertEqual(
                len(cells),
                frame["visual_group_id"].nunique()
                * frame["trace_id"].nunique()
                * 2,
            )

    def test_index_accepts_runner_schema_and_rejects_checksum_change(self) -> None:
        frame = _trial_rows()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = _write_partitions(root, frame)
            normalized = load_verified_partition_index(index)
            self.assertEqual(len(normalized), len(index))
            json_path = root / "index.json"
            json_path.write_text(json.dumps({
                "partitions": [
                    {
                        "artifact": row.artifact_path,
                        "sha256": row.artifact_sha256,
                        "rows": int(row.row_count),
                    }
                    for row in index.itertuples(index=False)
                ]
            }), encoding="utf-8")
            self.assertEqual(len(load_verified_partition_index(json_path)), len(index))
            corrupt = Path(index.iloc[0]["artifact_path"])
            with corrupt.open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(StudyGateError, "checksum changed"):
                load_verified_partition_index(index)


if __name__ == "__main__":
    unittest.main()
