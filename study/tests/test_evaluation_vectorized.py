from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from semuav.evaluation import (
    resource_rate_summary,
    two_way_clustered_paired_bootstrap,
)


class VectorizedTwoWayBootstrapTests(unittest.TestCase):
    @staticmethod
    def _paired_sparse_fixture() -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        deltas = {
            ("g0", "t0"): 1.0,
            ("g0", "t1"): 0.0,
            ("g0", "t2"): -1.0,
            ("g1", "t0"): 1.0,
            ("g1", "t1"): 1.0,
            # g1/t2 is intentionally absent.
            ("g2", "t0"): 0.0,
            ("g2", "t1"): -1.0,
            ("g2", "t2"): 1.0,
            ("g3", "t0"): -1.0,
            # g3/t1 is intentionally absent.
            ("g3", "t2"): 0.0,
        }
        for (visual_group, trace), delta in deltas.items():
            proposed = 1.0 if delta > 0 else 0.0
            reference = 1.0 if delta < 0 else 0.0
            sample_id = f"{visual_group}-{trace}"
            for policy, outcome in (
                ("proposed", proposed), ("reference", reference),
            ):
                rows.append({
                    "sample_id": sample_id,
                    "visual_group_id": visual_group,
                    "mission_profile": "standard",
                    "trace_id": trace,
                    "replicate_seed": 17,
                    "policy": policy,
                    "score": outcome,
                })
        return pd.DataFrame(rows)

    @staticmethod
    def _scalar_multinomial_reference(
        frame: pd.DataFrame,
        *,
        repetitions: int,
        seed: int,
        confidence: float,
    ) -> tuple[float, float, float, float]:
        pivot = frame.pivot(
            index=[
                "sample_id", "visual_group_id", "mission_profile",
                "trace_id", "replicate_seed",
            ],
            columns="policy",
            values="score",
        ).reset_index()
        pivot["delta"] = pivot["proposed"] - pivot["reference"]
        values = pivot.groupby(
            ["visual_group_id", "trace_id"], as_index=False,
        )["delta"].mean()
        visual = pivot["visual_group_id"].astype(str).unique()
        traces = pivot["trace_id"].astype(str).unique()
        estimate = float(values["delta"].mean())
        rng = np.random.default_rng(seed)
        visual_weights = np.empty((repetitions, len(visual)), dtype=np.int64)
        trace_weights = np.empty((repetitions, len(traces)), dtype=np.int64)
        for repetition in range(repetitions):
            visual_weights[repetition] = np.bincount(
                rng.choice(len(visual), len(visual), replace=True),
                minlength=len(visual),
            )
            trace_weights[repetition] = np.bincount(
                rng.choice(len(traces), len(traces), replace=True),
                minlength=len(traces),
            )
        visual_index = {name: index for index, name in enumerate(visual)}
        trace_index = {name: index for index, name in enumerate(traces)}
        draws: list[float] = []
        for repetition in range(repetitions):
            numerator = 0.0
            denominator = 0.0
            for row in values.itertuples(index=False):
                weight = (
                    visual_weights[repetition, visual_index[str(row.visual_group_id)]]
                    * trace_weights[repetition, trace_index[str(row.trace_id)]]
                )
                numerator += weight * float(row.delta)
                denominator += weight
            if denominator:
                draws.append(numerator / denominator)
        draws_array = np.asarray(draws, dtype=float)
        alpha = 1 - confidence
        low, high = np.quantile(draws_array, [alpha / 2, 1 - alpha / 2])
        p_value = (
            1 + np.sum(np.abs(draws_array - estimate) >= abs(estimate))
        ) / (len(draws_array) + 1)
        return estimate, float(low), float(high), float(p_value)

    def test_vectorized_sparse_bootstrap_matches_scalar_estimand(self) -> None:
        frame = self._paired_sparse_fixture()
        repetitions = 2_000
        seed = 91
        expected = self._scalar_multinomial_reference(
            frame, repetitions=repetitions, seed=seed, confidence=0.95,
        )
        actual = two_way_clustered_paired_bootstrap(
            frame,
            "proposed",
            "reference",
            endpoint="score",
            repetitions=repetitions,
            seed=seed,
        )
        self.assertAlmostEqual(actual.estimate, expected[0], places=15)
        self.assertAlmostEqual(actual.ci_low, expected[1], places=15)
        self.assertAlmostEqual(actual.ci_high, expected[2], places=15)
        self.assertAlmostEqual(actual.p_two_sided, expected[3], places=15)
        self.assertEqual(actual.repetitions, repetitions)

    def test_vectorized_bootstrap_is_seed_deterministic_at_ten_thousand_reps(self) -> None:
        rows: list[dict[str, object]] = []
        for visual in range(30):
            for trace in range(30):
                for policy in ("reference", "proposed"):
                    rows.append({
                        "sample_id": f"g{visual}-t{trace}",
                        "visual_group_id": f"g{visual}",
                        "mission_profile": "standard",
                        "trace_id": f"t{trace}",
                        "replicate_seed": 5,
                        "policy": policy,
                        "score": float(
                            policy == "proposed" or (visual + trace) % 5 != 0
                        ),
                    })
        frame = pd.DataFrame(rows)
        first = two_way_clustered_paired_bootstrap(
            frame, "proposed", "reference", endpoint="score",
            repetitions=10_000, seed=20260831,
        )
        second = two_way_clustered_paired_bootstrap(
            frame, "proposed", "reference", endpoint="score",
            repetitions=10_000, seed=20260831,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.repetitions, 10_000)


class EpisodeResourceRateTests(unittest.TestCase):
    def test_overlapping_episode_clocks_have_separate_exposure(self) -> None:
        rows: list[dict[str, object]] = []
        for episode in ("episode-a", "episode-b"):
            for capture_us in (0, 1_000_000):
                rows.append({
                    "policy": "p",
                    "trace_id": "trace-0",
                    "replicate_seed": 3,
                    "episode_id": episode,
                    "capture_us": capture_us,
                    "total_wire_bytes": 100.0,
                    "energy_j": 1.0,
                    "sender_compute_energy_j": 0.25,
                    "receiver_energy_j_charged": 0.75,
                })
        report = resource_rate_summary(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(int(report["episodes"]), 2)
        self.assertAlmostEqual(float(report["exposure_s"]), 4.0)
        self.assertAlmostEqual(float(report["wire_bytes_per_s"]), 100.0)
        self.assertAlmostEqual(float(report["energy_j_per_s"]), 1.0)
        self.assertAlmostEqual(
            float(report["sender_compute_energy_j_per_s"]), 0.25,
        )
        self.assertAlmostEqual(
            float(report["receiver_energy_j_charged_per_s"]), 0.75,
        )
        self.assertAlmostEqual(float(report["total_wire_bytes"]), 400.0)
        self.assertAlmostEqual(float(report["energy_j"]), 4.0)

    def test_supplied_episode_id_must_be_complete(self) -> None:
        frame = pd.DataFrame([
            {
                "policy": "p", "trace_id": "t", "replicate_seed": 1,
                "episode_id": None, "capture_us": 0,
                "total_wire_bytes": 1, "energy_j": 0.1,
            },
        ])
        with self.assertRaisesRegex(ValueError, "episode_id cannot be missing"):
            resource_rate_summary(frame)


if __name__ == "__main__":
    unittest.main(verbosity=2)
