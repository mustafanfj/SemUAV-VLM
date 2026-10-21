from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from semuav.experiments import (
    EXPERIMENT3_PUBLICATION_SUMMARY_MAX_ROWS,
    EXPERIMENT3_PUBLICATION_SUMMARY_SCHEMA,
    _experiment3_publication_partial,
    assert_experiment3_partition_index_coverage,
    experiment3_static_transport,
    experiment3_static_transport_partitioned,
    experiment3_publication_summary_row_bound,
    select_experiment3_manifest,
)
from semuav.registry import RunContext, StudyGateError, hash_file


ROOT = Path(__file__).resolve().parents[1]


class Experiment3PartitionedTests(unittest.TestCase):
    @staticmethod
    def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
        manifest = pd.DataFrame([
            {
                "sample_id": "partition-1", "capture_time_us": 0,
                "trace_time_us": 0, "deadline_us": 2_000_000,
            },
            {
                "sample_id": "partition-2", "capture_time_us": 100_000,
                "trace_time_us": 100_000, "deadline_us": 2_100_000,
            },
        ])
        candidates = pd.DataFrame([
            {
                "sample_id": sample_id, "mode": "L5_IMAGE_OR_CLIP",
                "variant": variant, "source_model": "ground_visual_receiver",
                "payload_blob": (variant.encode("utf-8") * 50).hex(),
                "generation_ms": 1.0, "energy_j": 0.01,
                "receiver_latency_ms": 1.0, "receiver_energy_j": 0.001,
                "native_receiver_correct": True,
            }
            for variant in ("partition-a", "partition-b")
            for sample_id in ("partition-1", "partition-2")
        ])
        return candidates, manifest

    @staticmethod
    def _kwargs() -> dict[str, object]:
        return {
            "shard_size": 2,
            "rates_mbps": (2.0,), "rtts_ms": (20.0,),
            "losses": (0.0,), "loss_models": ("iid",),
            "jitters_ms": (0.0,), "queue_kb": (128,),
        }

    @staticmethod
    def _context(directory: str, name: str) -> RunContext:
        return RunContext.create(
            ROOT, directory, "dry_run", name, {"test": name}, resume=True,
        )

    def test_summarizes_one_raw_cache_partition_at_a_time_and_reuses_cache(self) -> None:
        candidates, manifest = self._inputs()
        with tempfile.TemporaryDirectory() as directory:
            ctx = self._context(directory, "experiment3-partition-memory")
            real_concat = pd.concat
            raw_concat_sizes: list[int] = []

            def concat_spy(objects, *args, **kwargs):
                frames = list(objects)
                if frames and all(
                    isinstance(frame, pd.DataFrame) and "observation_id" in frame.columns
                    for frame in frames
                ):
                    raw_concat_sizes.append(sum(len(frame) for frame in frames))
                return real_concat(frames, *args, **kwargs)

            with (
                mock.patch("semuav.experiments.pd.concat", side_effect=concat_spy),
                mock.patch(
                    "semuav.experiments.summarize_experiment3_static_transport",
                    side_effect=AssertionError("per-stream campaign summary is forbidden"),
                ),
                mock.patch(
                    "semuav.experiments._experiment3_publication_partial",
                    wraps=_experiment3_publication_partial,
                ) as reduce_spy,
            ):
                summary, partition_index = experiment3_static_transport_partitioned(
                    candidates, manifest, ctx, registered_seeds=(7, 11),
                    **self._kwargs(),
                )

            # Two intact streams form two shards for each of two seeds.  Every
            # reducer input is exactly one shard, and no campaign-level raw or
            # per-stream summary concat is allowed.
            self.assertEqual(reduce_spy.call_count, 4)
            self.assertEqual(
                [len(call.args[0]) for call in reduce_spy.call_args_list],
                [2, 2, 2, 2],
            )
            self.assertTrue(raw_concat_sizes)
            self.assertLessEqual(max(raw_concat_sizes), 2)
            # Seeds and visual streams are reduced out of the publication row
            # domain: two representations x one registered link condition.
            self.assertEqual(len(summary), 2)
            self.assertEqual(
                set(summary["publication_summary_schema"]),
                {EXPERIMENT3_PUBLICATION_SUMMARY_SCHEMA},
            )
            self.assertTrue(summary["exact_registered_coverage"].astype(bool).all())
            self.assertTrue(summary["registered_seed_count"].eq(2).all())
            self.assertTrue(summary["observed_seed_count"].eq(2).all())
            self.assertTrue(summary["expected_observation_rows"].eq(4).all())
            self.assertTrue(summary["observation_rows"].eq(4).all())
            self.assertTrue(summary["registered_grid_condition_count"].eq(1).all())
            self.assertEqual(summary["raw_partition_index_sha256"].nunique(), 1)
            self.assertTrue(summary["exact_partition_coverage"].astype(bool).all())
            self.assertFalse(any(column.endswith("_sum") for column in summary.columns))
            self.assertEqual(
                set(partition_index.columns),
                {
                    "seed", "shard", "stage", "artifact_path",
                    "artifact_sha256", "row_count", "design_hash",
                },
            )
            self.assertEqual(set(partition_index["seed"]), {7, 11})
            self.assertTrue(partition_index["row_count"].eq(2).all())
            for row in partition_index.itertuples(index=False):
                path = Path(row.artifact_path)
                self.assertTrue(path.is_file())
                self.assertEqual(hash_file(path), row.artifact_sha256)
            per_seed_designs = {
                seed: tuple(
                    rows.sort_values("shard")["design_hash"].astype(str)
                )
                for seed, rows in partition_index.groupby("seed")
            }
            self.assertEqual(per_seed_designs[7], per_seed_designs[11])
            assert_experiment3_partition_index_coverage(
                partition_index, (7, 11),
            )

            # A second call must summarize cached partitions without invoking
            # the raw simulator again.
            with mock.patch(
                "semuav.experiments.experiment3_static_transport",
                side_effect=AssertionError("raw cache unexpectedly recomputed"),
            ):
                cached_summary, cached_index = experiment3_static_transport_partitioned(
                    candidates, manifest, ctx, registered_seeds=(7, 11),
                    **self._kwargs(),
                )
            pd.testing.assert_frame_equal(
                summary, cached_summary, check_dtype=False,
                check_exact=False, rtol=1e-12, atol=1e-12,
            )
            pd.testing.assert_frame_equal(
                partition_index, cached_index, check_dtype=False,
            )


    def test_publication_row_cap_is_checked_before_any_static_campaign(self) -> None:
        candidates, manifest = self._inputs()
        with tempfile.TemporaryDirectory() as directory:
            ctx = self._context(directory, "experiment3-publication-cap")
            with mock.patch(
                "semuav.experiments.experiment3_static_transport",
                side_effect=AssertionError("campaign must not start after cap failure"),
            ):
                with self.assertRaisesRegex(
                    StudyGateError, "paper-facing summary would exceed",
                ):
                    experiment3_static_transport_partitioned(
                        candidates, manifest, ctx, registered_seeds=(7, 11),
                        publication_summary_max_rows=1,
                        **self._kwargs(),
                    )

    def test_registered_paper_design_has_a_safe_group_independent_row_bound(self) -> None:
        # Conservative representation envelope: six payload modes, five
        # candidate sources, and three frozen wire-span variants per pair.
        # Thirty groups x three profiles x five seeds do not enter the compact
        # row domain.
        candidates = pd.DataFrame([
            {
                "mode": f"L{mode}", "source_model": f"source-{source}",
                "variant": f"variant-{variant}", "sample_id": f"sample-{sample}",
            }
            for mode in range(6)
            for source in range(5)
            for variant in range(3)
            for sample in range(30 * 3)
        ])
        rows = experiment3_publication_summary_row_bound(
            candidates,
            rates_mbps=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20),
            rtts_ms=(20, 80, 200), losses=(0.0, 0.01, 0.05, 0.15),
            loss_models=("iid", "gilbert_elliott"),
            jitters_ms=(0, 10), queue_kb=(32, 128, 512),
        )
        self.assertEqual(rows, 6 * 5 * 3 * 8 * 3 * 4 * 2 * 2 * 3)
        self.assertEqual(rows, 103_680)
        self.assertLess(rows, EXPERIMENT3_PUBLICATION_SUMMARY_MAX_ROWS)

    def test_rejects_a_cached_shard_missing_one_registered_condition(self) -> None:
        candidates, manifest = self._inputs()
        candidates = candidates[candidates["variant"].eq("partition-a")].copy()
        with tempfile.TemporaryDirectory() as directory:
            ctx = self._context(directory, "experiment3-partition-truncated")
            real_transport = experiment3_static_transport

            def truncated_transport(*args, **kwargs):
                return real_transport(*args, **kwargs).iloc[:-1].copy()

            with mock.patch(
                "semuav.experiments.experiment3_static_transport",
                side_effect=truncated_transport,
            ):
                with self.assertRaisesRegex(
                    StudyGateError, "exact registered condition coverage",
                ):
                    experiment3_static_transport_partitioned(
                        candidates, manifest, ctx, registered_seeds=(5,),
                        **self._kwargs(),
                    )

    def test_partition_index_gate_rejects_missing_seed_coverage_and_bad_hash(self) -> None:
        candidates, manifest = self._inputs()
        with tempfile.TemporaryDirectory() as directory:
            ctx = self._context(directory, "experiment3-partition-index-gates")
            _, partition_index = experiment3_static_transport_partitioned(
                candidates, manifest, ctx, registered_seeds=(3, 4),
                **self._kwargs(),
            )
            incomplete = partition_index.drop(
                partition_index[
                    partition_index["seed"].eq(4)
                    & partition_index["shard"].eq(1)
                ].index
            ).reset_index(drop=True)
            with self.assertRaisesRegex(StudyGateError, "incomplete condition coverage"):
                assert_experiment3_partition_index_coverage(
                    incomplete, (3, 4), verify_artifacts=False,
                )

            bad_checksum = partition_index.copy()
            bad_checksum.loc[0, "artifact_sha256"] = "0" * 64
            with self.assertRaisesRegex(StudyGateError, "checksum changed"):
                assert_experiment3_partition_index_coverage(
                    bad_checksum, (3, 4), verify_artifacts=True,
                )


class Experiment3ManifestSelectionTests(unittest.TestCase):
    @staticmethod
    def _selection_manifest() -> pd.DataFrame:
        rows = []
        group_specs = (
            ("dataset-a", "urban", "a-urban-1", 4),
            ("dataset-a", "urban", "a-urban-2", 3),
            ("dataset-a", "wildfire", "a-fire-1", 5),
            ("dataset-a", "wildfire", "a-fire-2", 3),
            ("dataset-b", "urban", "b-urban-1", 4),
            ("dataset-b", "wildfire", "b-fire-1", 4),
            ("dataset-b", "wildfire", "b-fire-2", 4),
        )
        for dataset, mission, group_id, observations in group_specs:
            for frame_index in range(observations):
                capture_us = frame_index * 100_000
                rows.append({
                    "sample_id": f"{group_id}-{frame_index}",
                    "dataset": dataset, "mission": mission,
                    "visual_group_id": group_id, "split_unit": group_id,
                    "frame_index": frame_index,
                    "timestamp_ms": frame_index * 100,
                    "capture_time_us": capture_us,
                    "trace_time_us": capture_us,
                    "deadline_us": capture_us + 2_000_000,
                    # These deliberate outcomes must not affect selection.
                    "gold_action": "RESPOND" if frame_index % 2 else "IGNORE",
                    "critical": bool(frame_index % 2),
                    "native_receiver_correct": bool((frame_index + 1) % 2),
                })
        return pd.DataFrame(rows)

    def test_group_selection_is_whole_stratified_stable_and_outcome_blind(self) -> None:
        manifest = self._selection_manifest()
        selected = select_experiment3_manifest(
            manifest, max_groups_per_dataset=2, seed=2026,
        )
        self.assertEqual(
            selected.groupby("dataset")["visual_group_id"].nunique().to_dict(),
            {"dataset-a": 2, "dataset-b": 2},
        )
        # With two available mission strata and a cap of two, balanced rounds
        # retain one whole group from each mission in each dataset.
        self.assertTrue(
            selected.groupby("dataset")["mission"].nunique().eq(2).all()
        )
        original_sizes = manifest.groupby("visual_group_id").size()
        selected_sizes = selected.groupby("visual_group_id").size()
        self.assertEqual(
            selected_sizes.to_dict(),
            original_sizes.loc[selected_sizes.index].to_dict(),
        )
        self.assertEqual(
            selected.attrs["experiment3_selection"]["group_column"],
            "visual_group_id",
        )

        relabeled = manifest.sample(frac=1.0, random_state=9).reset_index(drop=True)
        relabeled["gold_action"] = "CHANGED"
        relabeled["critical"] = ~relabeled["critical"]
        relabeled["native_receiver_correct"] = ~relabeled["native_receiver_correct"]
        repeated = select_experiment3_manifest(
            relabeled, max_groups_per_dataset=2, seed=2026,
        )
        self.assertEqual(
            selected["sample_id"].tolist(), repeated["sample_id"].tolist(),
        )
        self.assertEqual(
            selected["experiment3_selection_group_hash"].tolist(),
            repeated["experiment3_selection_group_hash"].tolist(),
        )

    def test_observation_cap_is_temporally_uniform_and_preserves_timing(self) -> None:
        observations = 9
        manifest = pd.DataFrame([
            {
                "sample_id": f"temporal-{index}", "dataset": "dataset-a",
                "mission": "urban", "split_unit": "temporal-group",
                "frame_index": index, "timestamp_ms": index * 125,
                "capture_time_us": 1_000_000 + index * 125_000,
                "trace_time_us": 2_000_000 + index * 125_000,
                "deadline_us": 3_500_000 + index * 125_000,
            }
            for index in range(observations)
        ])
        shuffled = manifest.sample(frac=1.0, random_state=17).reset_index(drop=True)
        selected = select_experiment3_manifest(
            shuffled, max_groups_per_dataset=1, seed=99,
            max_observations_per_group=4,
        )
        self.assertEqual(
            selected["sample_id"].tolist(),
            ["temporal-0", "temporal-3", "temporal-5", "temporal-8"],
        )
        self.assertEqual(
            selected["experiment3_selection_observation_rank"].tolist(),
            [0, 3, 5, 8],
        )
        self.assertTrue(
            selected["experiment3_selection_original_group_rows"].eq(9).all()
        )
        expected_timing = manifest.set_index("sample_id").loc[
            selected["sample_id"],
            ["capture_time_us", "trace_time_us", "deadline_us"],
        ].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            selected[["capture_time_us", "trace_time_us", "deadline_us"]],
            expected_timing,
        )

        repeated = select_experiment3_manifest(
            manifest.iloc[::-1].reset_index(drop=True),
            max_groups_per_dataset=1, seed=99,
            max_observations_per_group=4,
        )
        self.assertEqual(selected["sample_id"].tolist(), repeated["sample_id"].tolist())

    def test_selection_rejects_inconsistent_independence_groups(self) -> None:
        manifest = self._selection_manifest().head(2).copy()
        manifest.loc[manifest.index[-1], "split_unit"] = "different-unit"
        with self.assertRaisesRegex(StudyGateError, "same independence groups"):
            select_experiment3_manifest(
                manifest, max_groups_per_dataset=1, seed=7,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
