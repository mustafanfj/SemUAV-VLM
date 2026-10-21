from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from semuav.registry import StudyGateError
from semuav.visdrone_paper import (
    REGISTERED_MISSION_PROFILES,
    UntouchedTestAccessError,
    load_untouched_test_permit,
)
from semuav.visdrone_workflow import (
    TEMPORAL_SAMPLING_METHOD,
    TEMPORAL_SAMPLING_SCHEMA,
    audit_registered_temporal_trace_alignment,
    generate_registered_trace_manifest,
    read_analysis_manifest,
    registered_temporal_sampling_spec,
    select_registered_temporal_sample,
    select_registered_temporal_sample_from_config,
    verify_artifact_hash_manifest,
    write_registered_temporal_sample,
)


def _ledger(path: Path) -> Path:
    path.write_text(json.dumps({
        "status": "OPEN",
        "opened_utc": "2026-09-01T00:00:00Z",
        "protocol_hash": "a" * 64,
        "config_hash": "b" * 64,
        "code_hash": "c" * 64,
    }), encoding="utf-8")
    return path


def _temporal_manifest(
    *,
    split: str = "validation",
    sequences: int = 4,
    frames: int = 20,
    ledger_sha256: str = "",
) -> pd.DataFrame:
    component = "mot_test_dev" if split == "test" else "mot_val"
    deadlines = {
        "vulnerable_road_user": 700_000,
        "heavy_vehicle": 1_200_000,
        "general_mobility": 2_000_000,
    }
    rows = []
    for sequence_index in range(sequences):
        sequence_id = f"sequence-{sequence_index:02d}"
        for frame_index in range(1, frames + 1):
            source_id = f"{sequence_id}-frame-{frame_index:07d}"
            original_capture_us = (sequence_index * frames + frame_index) * 33_333
            for profile in REGISTERED_MISSION_PROFILES:
                rows.append({
                    "sample_id": f"{source_id}:profile:{profile}",
                    "source_sample_id": source_id,
                    "mission_profile": profile,
                    "component": component,
                    "split": split,
                    "split_unit": f"visdrone:{component}:{sequence_id}",
                    "sequence_id": sequence_id,
                    "frame_index": frame_index,
                    "source_uid": f"{component}/{sequence_id}/{frame_index:07d}.jpg",
                    "media_path": f"/data/{component}/{sequence_id}/{frame_index:07d}.jpg",
                    "capture_time_us": original_capture_us,
                    "trace_time_us": original_capture_us,
                    "deadline_us": original_capture_us + deadlines[profile],
                    "clip_paths_json": "[]",
                    "test_ledger_sha256": ledger_sha256,
                    "gold_action": "RESPOND" if frame_index % 2 else "NO_ACTION",
                    "event": "must-not-drive-selection",
                    "risk": frame_index / max(frames, 1),
                })
    return pd.DataFrame(rows)


class RegisteredTemporalSamplingTests(unittest.TestCase):
    def test_direct_selector_requires_an_explicit_sequence_count(self) -> None:
        with self.assertRaises(TypeError):
            select_registered_temporal_sample(
                _temporal_manifest(), split="validation", seed=1,
            )

    def test_paper_config_freezes_amended_validation_and_unchanged_test_capacity(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        payload = yaml.safe_load(
            (project_root / "configs" / "visdrone_paper.yaml").read_text(encoding="utf-8")
        )
        spec = registered_temporal_sampling_spec(payload["benchmark"], "validation")
        self.assertEqual(spec["sequences_per_split"], 7)
        self.assertEqual(spec["observations_per_sequence"], 64)
        self.assertEqual(spec["source_frame_stride"], 1)
        self.assertEqual(spec["target_rate_hz"], 5)
        test_spec = registered_temporal_sampling_spec(payload["benchmark"], "test")
        self.assertEqual(test_spec["sequences_per_split"], 15)
        self.assertEqual(test_spec["observations_per_sequence"], 64)
        self.assertEqual(test_spec["component"], "mot_test_dev")
        self.assertEqual(payload["network"]["validation_trace_realizations"], 12)
        self.assertEqual(
            payload["network"]["validation_fully_paired_scene_trace_cells"],
            spec["sequences_per_split"]
            * payload["network"]["validation_trace_realizations"],
        )
        self.assertEqual(
            payload["network"]["validation_fully_paired_scene_trace_cells"], 84,
        )

    def test_v2r9_validation_wrapper_rechecks_exact_eligible_capacity(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        payload = yaml.safe_load(
            (project_root / "configs" / "visdrone_paper.yaml").read_text(
                encoding="utf-8"
            )
        )
        selected, audit = select_registered_temporal_sample_from_config(
            _temporal_manifest(sequences=7, frames=64),
            benchmark=payload["benchmark"], split="validation", seed=17,
        )
        capacity = audit.loc[
            audit["gate"].eq("eligible_sequence_capacity")
        ]
        self.assertEqual(len(capacity), 1)
        self.assertEqual(json.loads(capacity.iloc[0]["observed"]), 7)
        exact_capacity = audit.loc[
            audit["gate"].eq(
                "exact_verified_validation_eligible_sequence_capacity"
            )
        ]
        self.assertEqual(len(exact_capacity), 1)
        self.assertEqual(json.loads(exact_capacity.iloc[0]["observed"]), 7)
        self.assertEqual(json.loads(exact_capacity.iloc[0]["required"]), 7)
        self.assertTrue(bool(exact_capacity.iloc[0]["passed"]))
        self.assertEqual(selected["sequence_id"].nunique(), 7)

        with self.assertRaisesRegex(
            StudyGateError, "insufficient eligible MOT sequences",
        ):
            select_registered_temporal_sample_from_config(
                _temporal_manifest(sequences=6, frames=64),
                benchmark=payload["benchmark"], split="validation", seed=17,
            )
        with self.assertRaisesRegex(
            StudyGateError,
            "Predecessor-observed validation temporal capacity changed",
        ):
            select_registered_temporal_sample_from_config(
                _temporal_manifest(sequences=8, frames=64),
                benchmark=payload["benchmark"], split="validation", seed=17,
            )

    def test_v2r9_test_wrapper_selects_15_from_larger_eligible_capacity(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        payload = yaml.safe_load(
            (project_root / "configs" / "visdrone_paper.yaml").read_text(
                encoding="utf-8"
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            permit = load_untouched_test_permit(
                _ledger(Path(directory) / "ledger.json")
            )
            manifest = _temporal_manifest(
                split="test", sequences=16, frames=64,
                ledger_sha256=permit.ledger_sha256,
            )
            selected, audit = select_registered_temporal_sample_from_config(
                manifest, benchmark=payload["benchmark"], split="test",
                seed=17, test_permit=permit,
            )
            self.assertEqual(selected["sequence_id"].nunique(), 15)
            capacity = audit.loc[audit["gate"].eq("eligible_sequence_capacity")]
            self.assertEqual(json.loads(capacity.iloc[0]["observed"]), 16)
            self.assertFalse(
                audit["gate"].eq(
                    "exact_verified_validation_eligible_sequence_capacity"
                ).any()
            )
            with self.assertRaisesRegex(ValueError, "only for validation"):
                select_registered_temporal_sample(
                    manifest, split="test", seed=17,
                    sequences_per_split=15, observations_per_sequence=64,
                    source_frame_stride=1, target_rate_hz=5,
                    test_permit=permit, exact_eligible_sequence_capacity=15,
                )

    def test_outcome_blind_contiguous_five_hz_profile_complete_selection(self) -> None:
        manifest = _temporal_manifest()
        first, first_audit = select_registered_temporal_sample(
            manifest,
            split="validation",
            seed=71,
            sequences_per_split=2,
            observations_per_sequence=4,
            source_frame_stride=2,
            target_rate_hz=5,
        )
        relabeled = manifest.copy()
        relabeled["gold_action"] = "CHANGED"
        relabeled["event"] = "CHANGED"
        relabeled["risk"] = 1.0 - relabeled["risk"]
        second, second_audit = select_registered_temporal_sample(
            relabeled,
            split="validation",
            seed=71,
            sequences_per_split=2,
            observations_per_sequence=4,
            source_frame_stride=2,
            target_rate_hz=5,
        )
        self.assertEqual(first["sample_id"].tolist(), second["sample_id"].tolist())
        self.assertEqual(
            first_audit["selection_sha256"].iloc[0],
            second_audit["selection_sha256"].iloc[0],
        )
        self.assertEqual(len(first), 2 * 4 * len(REGISTERED_MISSION_PROFILES))
        self.assertEqual(set(first["component"]), {"mot_val"})
        self.assertTrue(first_audit["passed"].all())
        self.assertTrue(first_audit["outcome_columns_read_json"].eq("[]").all())
        windows = first_audit[first_audit["record_type"].eq("selected_sequence_window")]
        self.assertEqual(len(windows), 2)
        for (_, profile), group in first.groupby(["sequence_id", "mission_profile"]):
            ordered = group.sort_values("temporal_observation_index")
            self.assertEqual(np.diff(ordered["frame_index"]).tolist(), [2, 2, 2])
            self.assertEqual(np.diff(ordered["capture_time_us"]).tolist(), [200_000] * 3)
            self.assertEqual(ordered["temporal_observation_index"].tolist(), [0, 1, 2, 3])
        profile_sets = first.groupby("source_sample_id")["mission_profile"].agg(set)
        self.assertTrue(profile_sets.map(lambda value: value == set(REGISTERED_MISSION_PROFILES)).all())

    def test_selection_ignores_det_rows_but_rejects_wrong_mot_partition(self) -> None:
        mot = _temporal_manifest(sequences=2)
        det = mot.head(3).copy()
        det["sample_id"] = "det-" + det["sample_id"].astype(str)
        det["source_sample_id"] = "det-" + det["source_sample_id"].astype(str)
        det["component"] = "det_val"
        selected, _ = select_registered_temporal_sample(
            pd.concat([mot, det], ignore_index=True), split="validation", seed=2,
            sequences_per_split=1, observations_per_sequence=4,
            source_frame_stride=2, target_rate_hz=5,
        )
        self.assertEqual(set(selected["component"]), {"mot_val"})
        wrong = mot.copy()
        wrong["component"] = "mot_train"
        with self.assertRaises(StudyGateError):
            select_registered_temporal_sample(
                wrong, split="validation", seed=2, sequences_per_split=1,
                observations_per_sequence=4, source_frame_stride=2,
                target_rate_hz=5,
            )

    def test_fails_closed_for_insufficient_windows_and_incomplete_profiles(self) -> None:
        sparse = _temporal_manifest(sequences=2, frames=5)
        with self.assertRaisesRegex(StudyGateError, "insufficient eligible MOT sequences"):
            select_registered_temporal_sample(
                sparse, split="validation", seed=1, sequences_per_split=2,
                observations_per_sequence=4, source_frame_stride=2,
                target_rate_hz=5,
            )
        incomplete = _temporal_manifest(sequences=2)
        incomplete = incomplete.loc[
            ~(
                incomplete["source_sample_id"].eq("sequence-00-frame-0000001")
                & incomplete["mission_profile"].eq(sorted(REGISTERED_MISSION_PROFILES)[0])
            )
        ]
        with self.assertRaisesRegex(StudyGateError, "all three mission-profile"):
            select_registered_temporal_sample(
                incomplete, split="validation", seed=1, sequences_per_split=1,
                observations_per_sequence=4, source_frame_stride=2,
                target_rate_hz=5,
            )

    def test_test_selection_requires_and_revalidates_open_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            permit = load_untouched_test_permit(_ledger(root / "ledger.json"))
            manifest = _temporal_manifest(
                split="test", sequences=2, ledger_sha256=permit.ledger_sha256,
            )
            with self.assertRaises(UntouchedTestAccessError):
                select_registered_temporal_sample(
                    manifest, split="test", seed=3, sequences_per_split=1,
                    observations_per_sequence=4, source_frame_stride=2,
                    target_rate_hz=5,
                )
            selected, audit = select_registered_temporal_sample(
                manifest, split="test", seed=3, sequences_per_split=1,
                observations_per_sequence=4, source_frame_stride=2,
                target_rate_hz=5, test_permit=permit,
            )
            self.assertTrue(selected["split"].eq("test").all())
            self.assertTrue(audit["passed"].all())
            payload = json.loads(permit.ledger_path.read_text(encoding="utf-8"))
            payload["status"] = "COMPLETE"
            permit.ledger_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(UntouchedTestAccessError):
                select_registered_temporal_sample(
                    manifest, split="test", seed=3, sequences_per_split=1,
                    observations_per_sequence=4, source_frame_stride=2,
                    target_rate_hz=5, test_permit=permit,
                )

    def test_config_wrapper_and_immutable_artifact_contract(self) -> None:
        benchmark = {
            "temporal_sampling": {
                "schema": TEMPORAL_SAMPLING_SCHEMA,
                "method": TEMPORAL_SAMPLING_METHOD,
                "validation_sequences": 2,
                "test_sequences": 2,
                "sequences_per_split": {"validation": 2, "test": 2},
                "observations_per_sequence": 4,
                "source_frame_stride": 2,
                "target_rate_hz": 5,
                "minimum_temporal_groups": 4,
                "minimum_group_trace_cells": 5,
                "component_by_split": {
                    "validation": "mot_val", "test": "mot_test_dev",
                },
            },
        }
        spec = registered_temporal_sampling_spec(benchmark, "validation")
        self.assertEqual(spec["component"], "mot_val")
        self.assertEqual(spec["capture_period_us"], 200_000)
        selected, audit = select_registered_temporal_sample_from_config(
            _temporal_manifest(), benchmark=benchmark, split="validation", seed=6,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = write_registered_temporal_sample(selected, audit, root / "temporal")
            self.assertEqual(set(outputs), {"sample", "audit", "artifact_manifest"})
            verify_artifact_hash_manifest(outputs["artifact_manifest"])
            persisted_sample = read_analysis_manifest(outputs["sample"])
            persisted_audit = pd.read_csv(
                outputs["audit"], keep_default_na=False,
            )
            pd.testing.assert_frame_equal(
                persisted_sample.reset_index(drop=True),
                selected.reset_index(drop=True),
                check_dtype=False,
                check_exact=True,
            )
            pd.testing.assert_frame_equal(
                persisted_audit.reset_index(drop=True),
                audit.reset_index(drop=True),
                check_dtype=False,
                check_exact=True,
            )
            with self.assertRaises(StudyGateError):
                write_registered_temporal_sample(selected, audit, root / "temporal")
            outputs["audit"].write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(StudyGateError):
                verify_artifact_hash_manifest(outputs["artifact_manifest"])


class TemporalTraceAlignmentTests(unittest.TestCase):
    def test_registered_traces_are_episode_aligned_with_terminal_hold(self) -> None:
        frame = generate_registered_trace_manifest(seed=17)
        audit = audit_registered_temporal_trace_alignment(frame)
        self.assertTrue(audit["passed"].all(), audit.to_dict("records"))
        self.assertTrue(bool(audit.set_index("gate").loc[
            "scientific_episode_coverage_audit", "passed",
        ]))
        self.assertEqual(frame.loc[frame["split"].eq("test"), "trace_id"].nunique(), 30)
        self.assertEqual(frame["end_us"].max(), 18_000_000)
        self.assertEqual(frame["start_us"].nunique(), 18)
        self.assertEqual(set(frame.loc[frame["start_us"].ge(12_000_000), "trace_phase"]), {
            "terminal_hold",
        })
        truncated = frame.loc[frame["end_us"].le(16_000_000)].copy()
        truncated_audit = audit_registered_temporal_trace_alignment(truncated).set_index("gate")
        self.assertFalse(bool(truncated_audit.loc[
            "scientific_episode_coverage_audit", "passed",
        ]))
        corrupted = frame.copy()
        index = corrupted.index[
            corrupted["trace_phase"].eq("terminal_hold")
        ][-1]
        corrupted.at[index, "service_rate_bps"] += 1
        failed = audit_registered_temporal_trace_alignment(corrupted)
        row = failed.set_index("gate").loc["terminal_hold_repeats_last_dynamic_state"]
        self.assertFalse(bool(row["passed"]))


if __name__ == "__main__":
    unittest.main()
