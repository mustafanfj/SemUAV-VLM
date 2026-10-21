from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from semuav.hardware import (
    EMBEDDED_STAGE_CHAIN,
    EMBEDDED_STAGE_ROLES,
    assert_environment_ready,
    embedded_gate,
    environment_fingerprint,
    evaluate_embedded_retest,
    evaluate_h5_status,
    import_embedded_jsonl,
    import_embedded_retest_summary,
    require_exclusive_physical_gpu,
)
from semuav.registry import StudyGateError, hash_file


class PhysicalGpuExclusivityTests(unittest.TestCase):
    @staticmethod
    def _result(stdout: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["nvidia-smi"], returncode=0, stdout=stdout, stderr="",
        )

    def test_empty_physical_gpu_passes_with_stable_pid_free_record(self) -> None:
        with patch(
            "semuav.hardware.subprocess.run", return_value=self._result("\n"),
        ) as query:
            record = require_exclusive_physical_gpu(1, allowed_pids=(os.getpid(),))

        query.assert_called_once_with(
            [
                "nvidia-smi", "-i", "1", "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=True, capture_output=True, text=True, timeout=10.0,
        )
        self.assertEqual(record["status"], "PASS")
        self.assertEqual(record["physical_gpu_index"], 1)
        self.assertNotIn(os.getpid(), record.values())
        self.assertNotIn("observed_pids", record)

    def test_current_kernel_is_the_only_permitted_compute_pid(self) -> None:
        current_pid = os.getpid()
        with patch(
            "semuav.hardware.subprocess.run",
            return_value=self._result(f" {current_pid} \n"),
        ):
            record = require_exclusive_physical_gpu(
                1, allowed_pids=(current_pid,),
            )
        self.assertEqual(
            record["allowed_process_scope"], "current_scientific_kernel_only",
        )

    def test_unexpected_compute_pid_stops_with_identity_inspection_instruction(self) -> None:
        with patch(
            "semuav.hardware.subprocess.run",
            return_value=self._result(f"{os.getpid()}\n424242\n"),
        ):
            with self.assertRaisesRegex(
                StudyGateError,
                r"unexpected compute PID\(s\): 424242.*Inspect process identity",
            ):
                require_exclusive_physical_gpu(
                    1, allowed_pids=(os.getpid(),),
                )

    def test_query_failure_and_malformed_rows_fail_closed(self) -> None:
        cases = (
            FileNotFoundError("nvidia-smi"),
            subprocess.CalledProcessError(1, ["nvidia-smi"]),
            self._result("No running processes found\n"),
            self._result("123, python\n"),
            self._result("0\n"),
        )
        for outcome in cases:
            with self.subTest(outcome=type(outcome).__name__, value=str(outcome)):
                if isinstance(outcome, BaseException):
                    context = patch(
                        "semuav.hardware.subprocess.run", side_effect=outcome,
                    )
                else:
                    context = patch(
                        "semuav.hardware.subprocess.run", return_value=outcome,
                    )
                with context, self.assertRaises(StudyGateError):
                    require_exclusive_physical_gpu(
                        1, allowed_pids=(os.getpid(),),
                    )


class EmbeddedHandoffTests(unittest.TestCase):
    baselines = (
        "baseline_a", "baseline_b", "channel_only_hysteresis",
        "ablate_evidence_escalation",
    )
    policies = ("proposed_full", *baselines)
    minima = {
        "natural_test_units": 2,
        "safety_critical_positive_units": 2,
        "safety_noncritical_units": 2,
        "ood_units": 2,
        "held_out_trace_realizations": 2,
        "held_out_trace_families": 2,
    }

    @classmethod
    def _raw_trials(cls, target_log_sha256: str) -> pd.DataFrame:
        sample_specs = [
            *(("natural_test", False, "NO_ACTION") for _ in range(2)),
            *(("safety", True, "RESPOND") for _ in range(2)),
            *(("safety", False, "NO_ACTION") for _ in range(2)),
            *(("ood", False, "NO_ACTION") for _ in range(2)),
        ]
        rows: list[dict[str, object]] = []
        for sample_index, (cohort, critical, gold_action) in enumerate(sample_specs):
            base_ns = 1_000_000_000 + sample_index * 1_000_000_000
            for policy in cls.policies:
                receiver_action = gold_action if policy == "proposed_full" else "REVIEW"
                if receiver_action == gold_action and policy != "proposed_full":
                    receiver_action = "NO_ACTION"
                for stage_index, (event, role) in enumerate(zip(EMBEDDED_STAGE_CHAIN, EMBEDDED_STAGE_ROLES)):
                    radio_delta = 0.002 if event == "tx_done" else 0.0
                    wire_delta = 100 if event == "tx_done" else 0
                    rows.append({
                        "trial_id": f"trial-{sample_index}",
                        "sample_id": f"sample-{sample_index}",
                        "visual_group_id": f"visual-{sample_index}",
                        "mission_profile": "wildfire" if critical else "urban",
                        "cohort": cohort,
                        "trace_id": f"trace-{sample_index % 2}",
                        "trace_family": f"family-{sample_index % 2}",
                        "replicate_seed": 11,
                        "policy": policy,
                        "event": event,
                        "stage_index": stage_index,
                        "monotonic_ns": base_ns + stage_index * 1_000_000,
                        "clock_domain_id": "ptp-domain-1",
                        "process_role": role,
                        "link_profile_id": f"link-{sample_index % 2}",
                        "hardware_id": "jetson-orin-01",
                        "platform_class": "jetson_class",
                        "absolute_deadline_ns": base_ns + 50_000_000,
                        "gold_action": gold_action,
                        "receiver_action": receiver_action,
                        "critical": critical,
                        "payload_valid": True,
                        "state_valid": True,
                        "selected_mode": "L2_SEMANTICS",
                        "semantic_prediction": True,
                        "calibrated_correctness_probability": 0.95,
                        "stage_compute_energy_j": 0.001,
                        "stage_radio_energy_j": radio_delta,
                        "wire_bytes_delta": wire_delta,
                        "trial_total_energy_j": 0.010,
                        "trial_total_wire_bytes": 100,
                        "memory_mb": 1000.0,
                        "temperature_c": 60.0,
                        "protocol_hash": "protocol-1",
                        "target_log_sha256": target_log_sha256,
                        "sender_revision": "sender-commit",
                        "receiver_revision": "receiver-commit",
                        "analysis_plan_version": "semuav.analysis.v1",
                    })
        primary = pd.DataFrame(rows)
        alternate = primary.copy()
        alternate["trial_id"] = alternate["trial_id"].astype(str) + "-alternate"
        alternate["trace_id"] = alternate["trace_id"].map({"trace-0": "trace-1", "trace-1": "trace-0"})
        alternate["trace_family"] = alternate["trace_family"].map({"family-0": "family-1", "family-1": "family-0"})
        alternate["link_profile_id"] = alternate["link_profile_id"].map({"link-0": "link-1", "link-1": "link-0"})
        alternate["monotonic_ns"] += 100_000_000
        alternate["absolute_deadline_ns"] += 100_000_000
        return pd.concat([primary, alternate], ignore_index=True)

    @classmethod
    def _retest_frame(cls, trial_artifact: Path, target_log: Path) -> pd.DataFrame:
        common = {
            "estimate": 1.0, "ci_low": 1.0, "ci_high": 1.0,
            "adjusted_p": 0.02, "margin": 0.03,
            "paired_design_pass": True, "matched_budget_pass": True,
            "sample_gate_pass": True, "calibration_gate_pass": True,
            "false_alert_rate": float("nan"), "reference_false_alert_rate": float("nan"),
            "method_selective_risk": float("nan"), "reference_selective_risk": float("nan"),
            "method_selective_coverage": float("nan"), "reference_selective_coverage": float("nan"),
            "h4_matched_coverage": 0.10,
            "h4_coverage_selection_rule": "top_calibrated_semantic_score_within_visual_group",
            "h4_coverage_gate_pass": True,
            "protocol_hash": "protocol-1",
            "target_log_sha256": hash_file(target_log), "hardware_id": "jetson-orin-01",
            "sender_revision": "sender-commit", "receiver_revision": "receiver-commit",
            "analysis_plan_version": "semuav.analysis.v1",
            "trial_artifact_path": str(trial_artifact.resolve()),
            "trial_artifact_sha256": hash_file(trial_artifact),
        }
        rows = [
            {**common, "hypothesis": "H2", "comparison": baseline, "endpoint": "correct_timely_action"}
            for baseline in cls.baselines
        ]
        rows.extend([
            {
                **common, "hypothesis": "H3",
                "comparison": "proposed_full_vs_channel_only_hysteresis",
                "endpoint": "critical_miss_reduction", "margin": 0.0,
                "adjusted_p": 0.01, "false_alert_rate": 0.0,
                "reference_false_alert_rate": 0.0,
            },
            {
                **common, "hypothesis": "H4",
                "comparison": "proposed_full_vs_ablate_evidence_escalation",
                "endpoint": "high_confidence_semantic_error_reduction", "margin": 0.0,
                "estimate": 0.5, "ci_low": 0.0, "ci_high": 1.0,
                "adjusted_p": 0.505, "method_selective_risk": 0.0,
                "reference_selective_risk": 1.0, "method_selective_coverage": 0.5,
                "reference_selective_coverage": 0.5,
            },
        ])
        return pd.DataFrame(rows)

    @classmethod
    def _evaluate(cls, summary: pd.DataFrame) -> pd.DataFrame:
        return evaluate_embedded_retest(
            summary, required_h2_baselines=cls.baselines,
            primary_margin=0.03, false_alert_limit=0.05,
            protocol_hash="protocol-1",
            target_log_sha256=str(summary.iloc[0]["target_log_sha256"]),
            expected_hardware_id="jetson-orin-01",
            wire_budget_bytes_per_s=1_000_000,
            energy_budget_j_per_s=1000,
            sample_minima=cls.minima,
            registered_calibration_gate_pass=True,
            bootstrap_repetitions=199,
            bootstrap_seed=7,
            h4_matched_coverage=0.10,
            h4_coverage_selection_rule="top_calibrated_semantic_score_within_visual_group",
        )

    @classmethod
    def _write_inputs(cls, root: Path) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
        target_log = root / "target.jsonl"
        target_log.write_text("registered-target-session", encoding="utf-8")
        raw = cls._raw_trials(hash_file(target_log))
        trial_artifact = root / "embedded_trials.csv.gz"
        raw.to_csv(trial_artifact, index=False, compression="gzip")
        summary = cls._retest_frame(trial_artifact, target_log)
        return target_log, trial_artifact, raw, summary

    def test_embedded_retest_recomputes_h2_h4_and_h5_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, raw, summary = self._write_inputs(root)
            summary_path = root / "summary.csv.gz"
            summary.to_csv(summary_path, index=False, compression="gzip")
            loaded = import_embedded_retest_summary(summary_path)
            retest = self._evaluate(loaded)
            self.assertEqual(set(retest["hypothesis"]), {"H2", "H3", "H4"})
            support = retest.set_index("hypothesis")["claim_supported"].to_dict()
            self.assertTrue(support["H2"])
            self.assertTrue(support["H3"])
            self.assertFalse(support["H4"])
            self.assertTrue(retest["raw_recomputed"].all())

            original = pd.DataFrame({
                "hypothesis": ["H2", "H3", "H4"],
                "claim_supported": [True, True, True],
            })
            hardware_gates = embedded_gate(
                raw, deadline_ms=20.0, memory_limit_mb=2000.0, thermal_limit_c=82.0,
            )
            h5 = evaluate_h5_status(original, retest, hardware_gates)
            self.assertFalse(bool(h5.iloc[0]["claim_supported"]))
            self.assertEqual(h5.iloc[0]["execution_status"], "EMPIRICAL_COMPLETE")

    def test_embedded_retest_fails_closed_on_missing_baseline_or_mutated_trials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, trial_artifact, _, summary = self._write_inputs(root)
            with self.assertRaises(StudyGateError):
                self._evaluate(summary[~summary["comparison"].eq("baseline_b")])
            trial_artifact.write_bytes(b"mutated-target-trials")
            with self.assertRaises(StudyGateError):
                self._evaluate(summary)

    def test_crafted_summary_cannot_override_raw_statistics_or_prerequisite_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, summary = self._write_inputs(root)
            forged = summary.copy()
            forged.loc[forged["hypothesis"].eq("H2"), "estimate"] = 0.99
            with self.assertRaises(StudyGateError):
                self._evaluate(forged)
            with self.assertRaises(StudyGateError):
                evaluate_embedded_retest(
                    summary, required_h2_baselines=self.baselines,
                    primary_margin=0.03, false_alert_limit=0.05,
                    protocol_hash="protocol-1",
                    target_log_sha256=str(summary.iloc[0]["target_log_sha256"]),
                    expected_hardware_id="jetson-orin-01",
                    wire_budget_bytes_per_s=1_000_000,
                    energy_budget_j_per_s=1000,
                    sample_minima=self.minima,
                    registered_calibration_gate_pass=False,
                    bootstrap_repetitions=199,
                    h4_matched_coverage=0.10,
                    h4_coverage_selection_rule="top_calibrated_semantic_score_within_visual_group",
                )

    def test_embedded_h4_requires_frozen_matched_coverage_from_raw_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, trial_artifact, raw, summary = self._write_inputs(root)
            unavailable = (
                raw["visual_group_id"].eq("visual-0")
                & raw["policy"].eq("ablate_evidence_escalation")
            )
            raw.loc[unavailable, "selected_mode"] = "L5_IMAGE_OR_CLIP"
            raw.loc[unavailable, "semantic_prediction"] = False
            raw.to_csv(trial_artifact, index=False, compression="gzip")
            summary["trial_artifact_sha256"] = hash_file(trial_artifact)
            with self.assertRaises(StudyGateError):
                self._evaluate(summary)

    def test_ordered_stage_schema_derives_latency_and_checks_energy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, raw, _ = self._write_inputs(root)
            one_chain = raw[(raw["trial_id"].eq("trial-0")) & raw["policy"].eq("proposed_full")].copy()
            one_chain = one_chain.drop(columns="target_log_sha256")
            one_chain["latency_ms"] = 0.1
            path = root / "target.jsonl"
            one_chain.to_json(path, orient="records", lines=True)
            loaded = import_embedded_jsonl(
                path, expected_hardware_id="jetson-orin-01",
                expected_platform_class="jetson_class",
            )
            self.assertTrue(loaded["latency_ms"].eq(7.0).all())
            gates = embedded_gate(
                loaded, deadline_ms=6.0, memory_limit_mb=2000.0, thermal_limit_c=82.0,
            )
            self.assertFalse(bool(gates.loc[gates["gate"].eq("deadline"), "passed"].iloc[0]))

            bad_order = one_chain.copy()
            bad_order.loc[bad_order["stage_index"].eq(2), "event"] = "enqueue"
            bad_order.to_json(path, orient="records", lines=True)
            with self.assertRaises(StudyGateError):
                import_embedded_jsonl(path)
            bad_energy = one_chain.copy()
            bad_energy["trial_total_energy_j"] = 999.0
            bad_energy.to_json(path, orient="records", lines=True)
            with self.assertRaises(StudyGateError):
                import_embedded_jsonl(path)

    def test_environment_fingerprint_is_order_stable_and_gate_is_scoped(self) -> None:
        report = pd.DataFrame([
            {"item": "gpu", "status": "FAIL", "value": "missing", "required_for": "real_profiles"},
            {"item": "python", "status": "PASS", "value": "3.12", "required_for": "all"},
        ])
        first = environment_fingerprint(report)
        second = environment_fingerprint(report.iloc[::-1].reset_index(drop=True))
        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], "semuav.environment.v2")
        self.assertEqual(len(first["python"]["sha256"]), 64)
        self.assertIn("torch_runtime", first)
        self.assertIn("ffmpeg", first)
        if first["ffmpeg"]["path"]:
            self.assertEqual(len(first["ffmpeg"]["sha256"]), 64)
            self.assertIn("configuration", first["ffmpeg"]["buildconf_output"])
        assert_environment_ready(report, ("all",))
        with self.assertRaises(StudyGateError):
            assert_environment_ready(report, ("all", "real_profiles"))

    def test_embedded_log_rejects_missing_or_wrong_hardware_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, raw, _ = self._write_inputs(root)
            one_chain = raw[(raw["trial_id"].eq("trial-0")) & raw["policy"].eq("proposed_full")].copy()
            one_chain = one_chain.drop(columns="target_log_sha256")
            path = root / "target.jsonl"
            one_chain.drop(columns="hardware_id").to_json(path, orient="records", lines=True)
            with self.assertRaises(StudyGateError):
                import_embedded_jsonl(path)
            one_chain["hardware_id"] = "workstation-01"
            one_chain.to_json(path, orient="records", lines=True)
            with self.assertRaises(StudyGateError):
                import_embedded_jsonl(
                    path, expected_hardware_id="jetson-orin-01",
                    expected_platform_class="jetson_class",
                )

    def test_h5_rejects_unrecomputed_retest_status(self) -> None:
        original = pd.DataFrame({"hypothesis": ["H2", "H3", "H4"], "claim_supported": [True] * 3})
        crafted = pd.DataFrame({"hypothesis": ["H2", "H3", "H4"], "claim_supported": [True] * 3})
        hardware = pd.DataFrame({
            "gate": ["deadline"], "passed": [True], "evidence_complete": [True],
        })
        with self.assertRaises(StudyGateError):
            evaluate_h5_status(original, crafted, hardware)


if __name__ == "__main__":
    unittest.main(verbosity=2)
