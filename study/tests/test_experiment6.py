from __future__ import annotations

import json
import unittest

import pandas as pd

from semuav.experiments import (
    EXPERIMENT6_CAMPAIGN_VERSION,
    EXPERIMENT6_REGISTERED_CASES,
    EXPERIMENT6_REGISTERED_STATE_STRATEGIES,
    EXPERIMENT6_STATE_STRATEGY_VERSION,
    experiment6_registered_campaign,
)
from semuav.exp6_strategies import experiment6_state_strategy_comparison
from semuav.registry import StudyGateError


class Experiment6RegisteredCampaignTests(unittest.TestCase):
    def test_registered_campaign_covers_and_passes_every_fault(self) -> None:
        detail, summary = experiment6_registered_campaign(
            seeds=(601,), scientific=False,
        )

        self.assertEqual(tuple(summary["case"]), EXPERIMENT6_REGISTERED_CASES)
        self.assertEqual(set(detail["case"]), set(EXPERIMENT6_REGISTERED_CASES))
        self.assertEqual(len(detail), len(EXPERIMENT6_REGISTERED_CASES))
        self.assertTrue(detail["fault_supported"].all())
        self.assertTrue(detail["passed"].all())
        self.assertTrue(summary["all_passed"].all())
        self.assertTrue(summary["fault_supported"].all())
        self.assertTrue(summary["campaign_version"].eq(EXPERIMENT6_CAMPAIGN_VERSION).all())
        self.assertTrue((summary["runs"] == 1).all())

        by_case = detail.set_index("case")
        self.assertEqual(
            by_case.loc["model_version_mismatch_rejected", "observed_status"],
            "version_mismatch",
        )
        self.assertGreater(
            int(by_case.loc["corruption_crc_retransmission", "retry_uplink_bytes"]),
            0,
        )
        priority_evidence = json.loads(
            by_case.loc["urgent_priority_bypass", "event_evidence"]
        )
        self.assertLess(
            priority_evidence["bulk_first_tx_us"],
            priority_evidence["urgent_first_tx_us"],
        )
        self.assertLess(
            priority_evidence["urgent_first_tx_us"],
            priority_evidence["next_bulk_tx_us"],
        )
        self.assertTrue(priority_evidence["urgent_payload_contract_valid"])

    def test_campaign_is_deterministic_for_registered_seeds(self) -> None:
        first_detail, first_summary = experiment6_registered_campaign(
            seeds=(607, 613), scientific=False,
        )
        second_detail, second_summary = experiment6_registered_campaign(
            seeds=(607, 613), scientific=False,
        )

        pd.testing.assert_frame_equal(first_detail, second_detail)
        pd.testing.assert_frame_equal(first_summary, second_summary)
        self.assertEqual(len(first_detail), 2 * len(EXPERIMENT6_REGISTERED_CASES))

    def test_case_set_cannot_be_changed_at_execution_time(self) -> None:
        with self.assertRaises(StudyGateError):
            experiment6_registered_campaign(
                seeds=(601,), scientific=False,
                registered_cases=EXPERIMENT6_REGISTERED_CASES[:-1],
            )

    def test_registered_state_strategies_are_paired_and_report_recovery(self) -> None:
        detail, summary = experiment6_state_strategy_comparison(
            seeds=(601, 607), scientific=False,
        )
        self.assertEqual(
            tuple(summary["strategy"]), EXPERIMENT6_REGISTERED_STATE_STRATEGIES,
        )
        self.assertTrue(
            summary["strategy_version"].eq(
                EXPERIMENT6_STATE_STRATEGY_VERSION
            ).all()
        )
        self.assertTrue(summary["coverage_complete"].all())
        self.assertTrue(summary["ack_loss_fault_exercised"].all())
        self.assertEqual(
            len(detail), 2 * 6 * len(EXPERIMENT6_REGISTERED_STATE_STRATEGIES),
        )
        self.assertFalse(detail["scientific_wire_format"].any())

        by_strategy = summary.set_index("strategy")
        delta = by_strategy.loc["application_acked_deltas"]
        periodic = by_strategy.loc["periodic_semantic_keyframes"]
        explicit = by_strategy.loc["explicit_resynchronization"]
        self.assertEqual(int(delta.base_mismatch_errors), 2)
        self.assertEqual(int(periodic.base_mismatch_errors), 2)
        self.assertEqual(int(explicit.base_mismatch_errors), 0)
        self.assertEqual(int(explicit.sequence_gap_updates), 0)
        self.assertEqual(int(delta.resync_attempts), 0)
        self.assertEqual(int(periodic.resync_successes), 2)
        self.assertEqual(int(explicit.resync_successes), 2)
        self.assertGreater(int(delta.wasted_uplink_bytes), 0)
        self.assertGreater(int(periodic.wasted_uplink_bytes), 0)
        self.assertEqual(int(explicit.wasted_uplink_bytes), 0)
        self.assertLess(
            float(explicit.median_time_to_restored_correct_state_ms),
            float(periodic.median_time_to_restored_correct_state_ms),
        )
        for required_metric in (
            "stale_state_error_rate", "sequence_gap_rate",
            "keyframe_frequency", "resync_success_rate", "resync_wire_bytes",
            "wire_delivery_rate", "valid_state_delivery_rate",
        ):
            self.assertIn(required_metric, summary.columns)

    def test_state_strategy_comparison_is_deterministic_and_frozen(self) -> None:
        first = experiment6_state_strategy_comparison(
            seeds=(613,), scientific=False,
        )
        second = experiment6_state_strategy_comparison(
            seeds=(613,), scientific=False,
        )
        pd.testing.assert_frame_equal(first[0], second[0])
        pd.testing.assert_frame_equal(first[1], second[1])
        with self.assertRaises(StudyGateError):
            experiment6_state_strategy_comparison(
                seeds=(613,), scientific=False,
                registered_strategies=EXPERIMENT6_REGISTERED_STATE_STRATEGIES[:-1],
            )
        with self.assertRaises(StudyGateError):
            experiment6_state_strategy_comparison(
                seeds=(613,), scientific=False,
                periodic_keyframe_interval=4,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
