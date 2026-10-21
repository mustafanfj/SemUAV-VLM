from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from semuav.experiments import experiment5_risk_uncertainty


class Experiment5Tests(unittest.TestCase):
    def test_registered_cohorts_and_safety_metrics_are_reported(self) -> None:
        rows = []
        specifications = [
            ("n-positive", "natural_test", True, False, "RESPOND", "RESPOND", True, 0.9),
            ("n-negative", "natural_test", False, False, "NO_ACTION", "NO_ACTION", True, 0.1),
            ("s-positive", "safety", True, False, "RESPOND", "REVIEW", True, 0.7),
            ("s-negative", "safety", False, False, "NO_ACTION", "RESPOND", True, 0.6),
            ("ood", "ood", False, True, "REVIEW", "REVIEW", True, 0.4),
        ]
        for group_id, cohort, critical, ood, gold, received, delivered, risk in specifications:
            rows.append({
                "visual_group_id": group_id, "cohort": cohort, "critical": critical,
                "ood": ood, "uncertainty": 0.4 if ood else 0.1, "policy": "p",
                "gold_action": gold, "receiver_action": received,
                "delivered_by_deadline": delivered, "risk": risk,
                "correct_timely_action": delivered and gold == received,
                "false_alert": delivered and received == "RESPOND" and gold != "RESPOND",
                "critical_miss": gold == "RESPOND" and not (delivered and gold == received),
                "high_confidence_semantic_error": False,
                "selected_mode": "L2_SEMANTICS",
                "calibrated_correctness_probability": 0.9,
                "correct_action": delivered and gold == received,
                "evidence_escalated": ood, "total_wire_bytes": 100,
                "energy_j": 0.2,
            })
        detail, summary = experiment5_risk_uncertainty(pd.DataFrame(rows))
        self.assertEqual(
            set(summary["subset"]),
            {
                "natural_prevalence", "safety_critical_positive",
                "safety_hard_negative", "ood", "high_uncertainty",
            },
        )
        self.assertTrue({
            "respond_precision", "respond_recall", "risk_auprc",
            "action_confusion_group_equal_json", "independent_visual_groups",
        }.issubset(summary.columns))
        natural = summary[summary["subset"].eq("natural_prevalence")].iloc[0]
        self.assertAlmostEqual(float(natural["respond_precision"]), 1.0)
        self.assertAlmostEqual(float(natural["respond_recall"]), 1.0)
        self.assertTrue(np.isfinite(float(natural["risk_auprc"])))
        self.assertEqual(int(natural["high_confidence_semantic_rows"]), 2)
        self.assertEqual(int(natural["high_confidence_semantic_errors"]), 0)
        self.assertEqual(float(natural["high_confidence_semantic_error_rate"]), 0.0)
        self.assertFalse(detail.empty)

    def test_low_confidence_semantic_error_is_excluded_from_denominator(self) -> None:
        rows = []
        for group, confidence, correct in (
            ("high", 0.90, True),
            ("low", 0.79, False),
        ):
            rows.append({
                "visual_group_id": group,
                "cohort": "natural_test",
                "critical": False,
                "ood": False,
                "uncertainty": 0.1,
                "policy": "p",
                "gold_action": "NO_ACTION",
                "receiver_action": "NO_ACTION" if correct else "RESPOND",
                "delivered_by_deadline": True,
                "risk": 0.1,
                "correct_action": correct,
                "correct_timely_action": correct,
                "false_alert": not correct,
                "critical_miss": False,
                "high_confidence_semantic_error": False,
                "selected_mode": "L2_SEMANTICS",
                "calibrated_correctness_probability": confidence,
                "evidence_escalated": False,
                "total_wire_bytes": 100,
                "energy_j": 0.2,
            })
        _, summary = experiment5_risk_uncertainty(
            pd.DataFrame(rows), semantic_acceptance_threshold=0.80,
        )
        natural = summary[summary["subset"].eq("natural_prevalence")].iloc[0]
        self.assertEqual(int(natural["high_confidence_semantic_rows"]), 1)
        self.assertEqual(int(natural["high_confidence_semantic_errors"]), 0)
        self.assertEqual(float(natural["high_confidence_semantic_error_rate"]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
