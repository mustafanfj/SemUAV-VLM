from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from semuav.experiments import run_paired_policy_trials_sharded
from semuav.network import Direction, constant_trace
from semuav.registry import RunContext


class DynamicCoarsePartitionTests(unittest.TestCase):
    def test_stage_count_does_not_multiply_by_episode_count_and_resumes(self) -> None:
        episode_count = 7
        trace_count = 2
        seed_count = 2
        policy_count = 2
        manifest = pd.DataFrame([
            {
                "sample_id": f"sample-{index}",
                "episode_id": f"episode-{index}",
                "visual_group_id": f"group-{index}",
                "mission_profile": "general_mobility",
            }
            for index in range(episode_count)
        ])
        candidates = pd.DataFrame([
            {"sample_id": f"sample-{index}", "mode": "L1_ALERT", "variant": "v1"}
            for index in range(episode_count)
        ])
        traces = []
        for index in range(trace_count):
            trace_id = f"trace-{index}"
            traces.extend([
                constant_trace(trace_id, Direction.UPLINK, 5.0, 10.0, 0.0),
                constant_trace(trace_id, Direction.DOWNLINK, 5.0, 10.0, 0.0),
            ])
        policies = [SimpleNamespace(name=f"policy-{index}") for index in range(policy_count)]
        seeds = (11, 13)
        profiles = pd.DataFrame([{"profile": "registered"}])

        def fake_trial(
            episode_manifest: pd.DataFrame,
            _candidate_frame: pd.DataFrame,
            trace_pair: object,
            selected_policies: object,
            selected_seeds: object,
            _profiles: pd.DataFrame,
            **_kwargs: object,
        ) -> pd.DataFrame:
            return pd.DataFrame([{
                "policy": selected_policies[0].name,
                "sample_id": str(episode_manifest.iloc[0]["sample_id"]),
                "trace_id": trace_pair[0].trace_id,
                "replicate_seed": int(selected_seeds[0]),
            }])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ctx = RunContext.create(root, root / "cache", "full", "coarse", {})
            with mock.patch(
                "semuav.experiments.run_paired_policy_trials", side_effect=fake_trial,
            ) as execute:
                result = run_paired_policy_trials_sharded(
                    manifest, candidates, traces, policies, seeds, profiles, ctx,
                    scientific=True,
                )

            expected_partitions = trace_count * seed_count
            self.assertEqual(
                execute.call_count,
                episode_count * expected_partitions * policy_count,
            )
            self.assertEqual(
                len(result), episode_count * expected_partitions * policy_count,
            )
            status = ctx.status_frame()
            partition_stages = status[
                status["name"].str.startswith("04_policy_partition_")
            ]
            self.assertEqual(len(partition_stages), expected_partitions)
            self.assertNotEqual(len(partition_stages), episode_count * expected_partitions)
            index_paths = list(
                (ctx.run_root / "stages" / "04_policy_partition_aggregate")
                .glob("*.index.json")
            )
            self.assertEqual(len(index_paths), 1)
            index = json.loads(index_paths[0].read_text(encoding="utf-8"))
            self.assertEqual(index["partition_count"], expected_partitions)
            self.assertEqual(index["episode_count"], episode_count)

            resumed = dataclasses.replace(ctx, resume=True)
            with mock.patch(
                "semuav.experiments.run_paired_policy_trials",
                side_effect=AssertionError("completed partitions must be reused"),
            ) as execute_again:
                replay = run_paired_policy_trials_sharded(
                    manifest, candidates, traces, policies, seeds, profiles, resumed,
                    scientific=True,
                )
            self.assertEqual(execute_again.call_count, 0)
            pd.testing.assert_frame_equal(result, replay)


if __name__ == "__main__":
    unittest.main()
