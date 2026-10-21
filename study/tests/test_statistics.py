from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from semuav.evaluation import (
    assert_paired_design,
    budget_compliance,
    holm_adjust,
    two_way_clustered_paired_bootstrap,
    validation_power_simulation,
    wilson_interval,
)


class StatisticsTests(unittest.TestCase):
    def fixture(self) -> pd.DataFrame:
        rows = []
        for visual in range(20):
            for trace in range(6):
                for policy in ("baseline", "proposed"):
                    rows.append({
                        "sample_id": f"v{visual}-sample",
                        "visual_group_id": f"v{visual}",
                        "mission_profile": "standard",
                        "trace_id": f"t{trace}",
                        "replicate_seed": 11,
                        "policy": policy,
                        "correct_timely_action": bool(policy == "proposed" or (visual + trace) % 4),
                    })
        return pd.DataFrame(rows)

    def test_two_way_bootstrap_is_paired(self) -> None:
        frame = self.fixture()
        assert_paired_design(frame)
        effect = two_way_clustered_paired_bootstrap(frame, "proposed", "baseline", repetitions=500, seed=9)
        self.assertGreater(effect.estimate, 0)
        self.assertEqual(effect.visual_clusters, 20)
        self.assertEqual(effect.trace_clusters, 6)

    def test_clustered_effect_point_estimate_is_not_frame_weighted(self) -> None:
        rows = []
        # The long visual group has ten positive frames; the short group has
        # one negative frame.  The registered estimand weights the two visual
        # groups equally, so the effect is zero rather than 9/11.
        for visual, samples, proposed, baseline in (
            ("long", 10, 1, 0), ("short", 1, 0, 1),
        ):
            for sample in range(samples):
                for policy, outcome in (("proposed", proposed), ("baseline", baseline)):
                    rows.append({
                        "sample_id": f"{visual}-{sample}",
                        "visual_group_id": visual,
                        "mission_profile": "standard",
                        "trace_id": "t0", "replicate_seed": 11,
                        "policy": policy, "correct_timely_action": outcome,
                    })
        effect = two_way_clustered_paired_bootstrap(
            pd.DataFrame(rows), "proposed", "baseline", repetitions=200, seed=4,
        )
        self.assertAlmostEqual(effect.estimate, 0.0)

    def test_holm_monotonic(self) -> None:
        adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20})
        self.assertLessEqual(adjusted["a"], adjusted["b"])
        self.assertLessEqual(adjusted["b"], adjusted["c"])

    def test_power_screen_tests_effect_above_noninferiority_margin(self) -> None:
        at_margin = validation_power_simulation(
            0.2, 500, 500, effect=0.03, null_margin=0.03,
            simulations=2000, seed=7,
        )
        above_margin = validation_power_simulation(
            0.2, 500, 500, effect=0.12, null_margin=0.03,
            simulations=2000, seed=7,
        )
        self.assertLess(at_margin, above_margin)

    def test_wilson(self) -> None:
        low, high = wilson_interval(90, 100)
        self.assertLess(low, 0.9)
        self.assertGreater(high, 0.9)

    def test_resource_budgets_are_rates_not_per_trial(self) -> None:
        frame = pd.DataFrame([
            {"policy": "p", "trace_id": "t", "replicate_seed": 1, "capture_us": 0, "total_wire_bytes": 100, "energy_j": 1.0},
            {"policy": "p", "trace_id": "t", "replicate_seed": 1, "capture_us": 1_000_000, "total_wire_bytes": 100, "energy_j": 1.0},
        ])
        report = budget_compliance(frame, 100.0, 1.0)
        self.assertAlmostEqual(float(report.iloc[0].wire_bytes_per_s), 100.0)
        self.assertAlmostEqual(float(report.iloc[0].energy_j_per_s), 1.0)
        self.assertTrue(bool(report.iloc[0].eligible))


if __name__ == "__main__":
    unittest.main(verbosity=2)
