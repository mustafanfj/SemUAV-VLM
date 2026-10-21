from __future__ import annotations

import tempfile
import unittest

import numpy as np
import pandas as pd

from semuav.control import FixedModePolicy, LookaheadReferencePolicy, OraclePolicy
from semuav.experiments import (
    experiment2_representation_frontier,
    experiment4_controller_behavior,
    make_dry_run_fixtures,
    run_paired_policy_trials,
)
from semuav.network import Direction, constant_trace
from semuav.protocol import PayloadMode


def _controller_row(
    policy: str,
    index: int,
    *,
    mode: str,
    correct_timely: bool,
    correct_action: bool,
    apply_status: str = "applied",
    full_message_received: bool = True,
    state_valid: bool = True,
    recovery: bool = False,
) -> dict[str, object]:
    return {
        "policy": policy,
        "sample_id": f"sample-{index}",
        "visual_group_id": f"visual-{index}",
        "mission_profile": "standard",
        "trace_id": "held-out-outage-1",
        "replicate_seed": 17,
        "decision_us": index * 1_000_000,
        "selected_mode": mode,
        "variant": f"registered-{mode.lower()}",
        "application_payload_bytes": 100,
        "full_message_received": full_message_received,
        "state_valid": state_valid,
        "state_kind": "keyframe" if policy == "proposed_full" else "none",
        "apply_status": apply_status,
        "correct_action": correct_action,
        "correct_timely_action": correct_timely,
        "original_uplink_bytes": 120,
        "retry_uplink_bytes": 20 if policy == "proposed_full" and index == 1 else 0,
        "total_wire_bytes": 140 if policy == "proposed_full" and index == 1 else 120,
        "energy_j": 2.0,
        "decision_queue_pending_bytes": index * 10,
        "decision_queue_outstanding_unacked_bytes": index * 3,
        "decision_queue_messages": index,
        "decision_queue_oldest_age_ms": float(index * 5),
        "decision_backlog_bytes": index * 13,
        "decision_sender_sequence_gap": 1 if policy == "proposed_full" and index == 3 else 0,
        "episode_outage_count": 1,
        "episode_outages_recovered": 1,
        "episode_outages_unrecovered": 0,
        "outage_recovery_event": recovery,
        "outage_recovery_latency_ms": 250.0 if recovery else np.nan,
        "calibrated_correctness_probability": 0.9,
        "high_confidence_semantic_error": (
            policy == "proposed_full" and not correct_action
        ),
    }


class Experiment4ControllerReportTests(unittest.TestCase):
    def test_behavior_summary_definitions_and_paired_lookahead_gap(self) -> None:
        rows: list[dict[str, object]] = []
        proposed_modes = [
            "L1_ALERT", "L2_SEMANTICS", "L1_ALERT", "L1_ALERT", "L2_SEMANTICS",
        ]
        proposed_success = [True, False, True, False, True]
        proposed_correct = [True, False, True, True, True]
        for index, (mode, timely, correct) in enumerate(
            zip(proposed_modes, proposed_success, proposed_correct)
        ):
            rows.append(_controller_row(
                "proposed_full", index, mode=mode,
                correct_timely=timely, correct_action=correct,
                apply_status="stale" if index == 1 else "base_mismatch" if index == 3 else "applied",
                full_message_received=index != 1,
                state_valid=index != 3,
                recovery=index == 2,
            ))
            rows.append(_controller_row(
                "heuristic_lookahead_reference", index, mode="L5_IMAGE_OR_CLIP",
                correct_timely=True, correct_action=True,
                recovery=index == 1,
            ))

        detail, summary = experiment4_controller_behavior(pd.DataFrame(rows))
        proposed = summary.set_index("policy").loc["proposed_full"]
        self.assertEqual(int(proposed.mode_switches), 3)
        self.assertAlmostEqual(float(proposed.mode_switch_rate), 3 / 4)
        self.assertEqual(int(proposed.oscillations_aba), 1)
        self.assertAlmostEqual(float(proposed.oscillation_rate), 1 / 3)
        self.assertAlmostEqual(float(proposed.mean_completed_dwell_ms), 4_000 / 3)
        self.assertAlmostEqual(float(proposed.mode_frequency_l1_alert), 3 / 5)
        self.assertAlmostEqual(float(proposed.mode_frequency_l2_semantics), 2 / 5)
        self.assertEqual(float(proposed.aborted_generated_payload_bytes), 200.0)
        self.assertEqual(float(proposed.wasted_uplink_wire_bytes), 260.0)
        self.assertAlmostEqual(float(proposed.stale_state_rate), 1 / 5)
        self.assertAlmostEqual(float(proposed.sequence_gap_rejection_rate), 1 / 5)
        self.assertAlmostEqual(float(proposed.decision_sequence_gap_rate), 1 / 5)
        self.assertEqual(int(proposed.outages_recovered), 1)
        self.assertEqual(float(proposed.median_outage_recovery_ms), 250.0)
        self.assertAlmostEqual(float(proposed.mean_lookahead_reference_signed_gap), 2 / 5)
        self.assertAlmostEqual(
            float(proposed.lookahead_reference_signed_gap_percentage_points), 40.0,
        )
        self.assertNotIn("mean_oracle_regret", summary.columns)
        self.assertAlmostEqual(float(proposed.semantic_selective_coverage), 1.0)
        self.assertAlmostEqual(float(proposed.semantic_selective_risk), 1 / 5)
        self.assertAlmostEqual(
            float(proposed.semantic_selective_mission_failure_rate), 2 / 5,
        )
        self.assertAlmostEqual(float(proposed.joules_per_correct_timely_action), 10 / 3)
        paired = detail[detail["policy"].eq("proposed_full")]
        self.assertEqual(int(paired["lookahead_reference_missed_opportunity"].sum()), 2)
        self.assertEqual(int(paired["lookahead_reference_inversion"].sum()), 0)
        self.assertNotIn("oracle_regret", detail.columns)

        legacy = pd.DataFrame(rows).replace({
            "policy": {"heuristic_lookahead_reference": "future_oracle"},
        })
        with self.assertWarns(DeprecationWarning):
            legacy_detail, legacy_summary = experiment4_controller_behavior(legacy)
        self.assertIn("lookahead_reference_signed_gap", legacy_detail.columns)
        self.assertIn(
            "mean_lookahead_reference_signed_gap", legacy_summary.columns,
        )
        self.assertNotIn("oracle_regret", legacy_detail.columns)

    def test_runner_records_only_as_of_decision_controller_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = make_dry_run_fixtures(directory, 4)
            candidates = experiment2_representation_frontier(manifest, scientific=False)
            trace = constant_trace(
                "causal-controller-trace", Direction.UPLINK,
                rate_mbps=2.0, one_way_ms=20.0, loss=0.0,
                duration_s=30.0,
            )
            trials = run_paired_policy_trials(
                manifest, candidates, [trace],
                [FixedModePolicy(PayloadMode.L1_ALERT), LookaheadReferencePolicy()], (19,),
            )
        for as_of_column in ("decision_link_as_of_us", "decision_queue_as_of_us"):
            self.assertTrue((trials[as_of_column] <= trials["decision_us"]).all())
        feedback = trials["decision_receiver_feedback_arrival_us"].dropna()
        self.assertTrue(
            (feedback <= trials.loc[feedback.index, "decision_us"]).all()
        )
        detail, summary = experiment4_controller_behavior(trials)
        self.assertEqual(len(detail), len(trials))
        self.assertEqual(
            set(summary["policy"]),
            {"fixed_l1_alert", "heuristic_lookahead_reference"},
        )

    def test_deprecated_oracle_class_uses_non_oracle_label(self) -> None:
        self.assertEqual(OraclePolicy().name, "heuristic_lookahead_reference")


if __name__ == "__main__":
    unittest.main(verbosity=2)
