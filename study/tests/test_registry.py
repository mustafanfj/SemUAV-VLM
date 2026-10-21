from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from semuav.registry import (
    EMBEDDED_HANDOFF_REQUIRED_ARTIFACTS,
    RunContext,
    StudyGateError,
    authorize_test_once,
    confirmatory_gate_pass,
    finalize_test_once,
    freeze_protocol,
    frozen_file_manifest,
    load_finalized_test_handoff,
    load_report_safe_finalized_test_handoff,
    matched_coverage_semantic_errors,
    notebook_source_hash,
    resolve_frozen_artifact,
    selected_vlm_claim_gate,
    verify_frozen_file_manifest,
)


class FrozenArtifactTests(unittest.TestCase):
    @staticmethod
    def _test_context(directory: str) -> RunContext:
        root = Path(directory)
        run_root = root / "run"
        (run_root / "results").mkdir(parents=True)
        return RunContext(
            project_root=root, run_root=run_root, profile="full", run_name="handoff",
            config={}, config_hash="config-hash", code_hash="code-hash",
            created_utc="2026-08-23T00:00:00Z", resume=False,
        )

    def test_frozen_manifest_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.txt"
            path.write_text("frozen", encoding="utf-8")
            manifest = frozen_file_manifest({"artifact": path})
            verify_frozen_file_manifest(manifest)
            path.write_text("changed", encoding="utf-8")
            with self.assertRaises(StudyGateError):
                verify_frozen_file_manifest(manifest)

    def test_test_input_must_resolve_to_frozen_path_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frozen_path = Path(directory) / "receiver.csv.gz"
            frozen_path.write_bytes(b"registered receiver")
            alternate = Path(directory) / "alternate.csv.gz"
            alternate.write_bytes(frozen_path.read_bytes())
            protocol = {"decisions": {"artifact_manifest": frozen_file_manifest({
                "native_receiver_cache": frozen_path,
            })}}
            self.assertEqual(
                resolve_frozen_artifact(
                    protocol, "native_receiver_cache", frozen_path,
                ),
                frozen_path.resolve(),
            )
            # Identical bytes at another environment-selected path are not the
            # registered artifact and cannot bypass the post-freeze path bind.
            with self.assertRaises(StudyGateError):
                resolve_frozen_artifact(
                    protocol, "native_receiver_cache", alternate,
                )
            frozen_path.write_bytes(b"mutated")
            with self.assertRaises(StudyGateError):
                resolve_frozen_artifact(protocol, "native_receiver_cache")

    def test_claim_gate_requires_applicable_policy_and_common_gates(self) -> None:
        rows = [
            {"gate": "sample_minimum", "passed": True, "required_for": "H1-H5", "policy": ""},
            {"gate": "annotation_overall", "passed": True, "required_for": "H1-H5", "policy": ""},
        ]
        for policy in ("proposed_full", "channel_only_hysteresis"):
            rows.extend([
                {"gate": f"policy_present:{policy}", "passed": True, "required_for": "H2-H3", "policy": policy},
                {"gate": f"matched_budget:{policy}", "passed": True, "required_for": "H2-H3", "policy": policy},
            ])
        gates = pd.DataFrame(rows)
        policies = ("proposed_full", "channel_only_hysteresis")
        self.assertTrue(confirmatory_gate_pass(gates, "H3", required_policies=policies))
        gates.loc[gates["gate"].eq("sample_minimum"), "passed"] = False
        self.assertFalse(confirmatory_gate_pass(gates, "H3", required_policies=policies))
        gates.loc[gates["gate"].eq("sample_minimum"), "passed"] = True
        gates = gates[~gates["gate"].eq("matched_budget:channel_only_hysteresis")]
        self.assertFalse(confirmatory_gate_pass(gates, "H3", required_policies=policies))

    def test_detector_fallback_cannot_enable_vlm_hypotheses(self) -> None:
        self.assertFalse(selected_vlm_claim_gate({
            "selected_model": "detector_only", "vlm_claim_enabled": False,
        }))
        self.assertFalse(selected_vlm_claim_gate({
            "selected_model": "detector_only", "vlm_claim_enabled": True,
        }))
        self.assertTrue(selected_vlm_claim_gate({
            "selected_model": "qwen3_vl_2b", "vlm_claim_enabled": True,
        }))
        gates = pd.DataFrame([{
            "gate": "selected_context_is_vlm", "passed": False,
            "required_for": "H1-H5", "policy": "",
        }])
        self.assertFalse(confirmatory_gate_pass(gates, "H1"))

    def test_h4_selection_matches_coverage_within_every_visual_group(self) -> None:
        rows = []
        for policy in ("proposed_full", "ablate_evidence_escalation"):
            for visual_group in ("g1", "g2"):
                for index in range(5):
                    rows.append({
                        "sample_id": f"{visual_group}-{index}",
                        "visual_group_id": visual_group,
                        "mission_profile": "routine",
                        "trace_id": "trace-a",
                        "replicate_seed": 7,
                        "policy": policy,
                        "calibrated_correctness_probability": 0.99 - 0.1 * index,
                        "selected_mode": "L2_SEMANTICS",
                        "correct_action": not (index == 0 and policy == "ablate_evidence_escalation"),
                    })
        matched, audit = matched_coverage_semantic_errors(
            pd.DataFrame(rows),
            ("proposed_full", "ablate_evidence_escalation"),
            0.4,
        )
        self.assertTrue(audit["passed"].all())
        counts = matched.groupby(["policy", "visual_group_id"])[
            "matched_coverage_selected"
        ].sum().unstack("policy")
        self.assertTrue(counts.nunique(axis=1).eq(1).all())
        self.assertTrue(counts.eq(2).all().all())
        risks = audit.set_index("policy")["group_equal_selective_risk"]
        self.assertEqual(float(risks["proposed_full"]), 0.0)
        self.assertEqual(float(risks["ablate_evidence_escalation"]), 0.5)

    def test_h4_coverage_gate_fails_when_a_group_lacks_semantic_scores(self) -> None:
        rows = []
        for policy in ("proposed_full", "ablate_evidence_escalation"):
            for index in range(4):
                rows.append({
                    "sample_id": f"s{index}", "visual_group_id": "g1",
                    "mission_profile": "routine", "trace_id": "trace-a",
                    "replicate_seed": 7, "policy": policy,
                    "calibrated_correctness_probability": 0.9 - index * 0.1,
                    "selected_mode": (
                        "L5_IMAGE_OR_CLIP"
                        if policy == "ablate_evidence_escalation" else "L2_SEMANTICS"
                    ),
                    "correct_action": True,
                })
        _, audit = matched_coverage_semantic_errors(
            pd.DataFrame(rows),
            ("proposed_full", "ablate_evidence_escalation"),
            0.25,
        )
        self.assertFalse(bool(audit.set_index("policy").loc[
            "ablate_evidence_escalation", "passed",
        ]))

    def test_notebook_hash_ignores_outputs_and_parameter_cell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.ipynb"
            base = {
                "nbformat": 4,
                "cells": [
                    {"cell_type": "code", "metadata": {"tags": ["parameters"]}, "source": "PHASE='validation'", "outputs": []},
                    {"cell_type": "code", "metadata": {}, "source": "result = 1", "outputs": []},
                ],
            }
            import json
            path.write_text(json.dumps(base), encoding="utf-8")
            before = notebook_source_hash(path)
            base["cells"][0]["source"] = "PHASE='test'"
            base["cells"][1]["outputs"] = [{"output_type": "stream", "text": "1"}]
            path.write_text(json.dumps(base), encoding="utf-8")
            self.assertEqual(before, notebook_source_hash(path))
            base["cells"][1]["source"] = "result = 2"
            path.write_text(json.dumps(base), encoding="utf-8")
            self.assertNotEqual(before, notebook_source_hash(path))

    def test_finalized_test_handoff_verifies_and_loads_hashed_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ctx = self._test_context(directory)
            frozen_input = Path(directory) / "frozen-input.txt"
            frozen_input.write_text("immutable", encoding="utf-8")
            protocol = freeze_protocol(
                ctx,
                {"artifact_manifest": frozen_file_manifest({"input": frozen_input})},
                {},
            )
            authorize_test_once(ctx, protocol)
            result = ctx.run_root / "results" / "test_policy_trials.csv.gz"
            result.write_bytes(b"finalized-result")
            finalize_test_once(ctx, status="COMPLETE", result_artifacts=[result])
            # Embedded handoff consumes only the protocol receipt and finalized
            # results; it must not reopen the original test/model inputs.
            frozen_input.unlink()

            handoff = load_finalized_test_handoff(
                ctx, protocol, required_artifacts=("test_policy_trials.csv.gz",),
            )
            self.assertEqual(handoff.artifact("test_policy_trials.csv.gz"), result.resolve())
            self.assertEqual(handoff.ledger["status"], "COMPLETE")

            result.write_bytes(b"mutated-result")
            with self.assertRaises(StudyGateError):
                load_finalized_test_handoff(
                    ctx, protocol, required_artifacts=("test_policy_trials.csv.gz",),
                )

    def test_open_test_ledger_cannot_be_used_for_embedded_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ctx = self._test_context(directory)
            frozen_input = Path(directory) / "frozen-input.txt"
            frozen_input.write_text("immutable", encoding="utf-8")
            protocol = freeze_protocol(
                ctx,
                {"artifact_manifest": frozen_file_manifest({"input": frozen_input})},
                {},
            )
            authorize_test_once(ctx, protocol)
            with self.assertRaises(StudyGateError):
                load_finalized_test_handoff(
                    ctx, protocol, required_artifacts=("test_policy_trials.csv.gz",),
                )

    def test_report_safe_handoff_never_reopens_frozen_inputs_and_detects_result_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ctx = self._test_context(directory)
            inputs = {
                "source_receipt": root / "data" / "SOURCE_RECEIPT.json",
                "source_hash_inventory": root / "data" / "SOURCE_FILES_SHA256.jsonl",
                "model_checkpoint": root / "models" / "vlm.safetensors",
                "cuda_executable": root / "cuda" / "nvcc",
            }
            for name, path in inputs.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"frozen-{name}", encoding="utf-8")
            protocol = freeze_protocol(
                ctx,
                {
                    "artifact_manifest": frozen_file_manifest(inputs),
                    "required_dataset_provenance_artifacts": [
                        "source_receipt", "source_hash_inventory",
                    ],
                },
                {},
            )
            authorize_test_once(ctx, protocol)
            result = ctx.run_root / "results" / "test_policy_trials.csv.gz"
            result.write_bytes(b"immutable-paper-result")
            finalize_test_once(ctx, status="COMPLETE", result_artifacts=[result])

            for path in inputs.values():
                path.unlink()
            handoff = load_report_safe_finalized_test_handoff(
                ctx,
                protocol,
                required_artifacts=("test_policy_trials.csv.gz",),
            )
            self.assertEqual(handoff.artifact("test_policy_trials.csv.gz"), result.resolve())
            with self.assertRaises(StudyGateError):
                load_finalized_test_handoff(
                    ctx,
                    protocol,
                    required_artifacts=("test_policy_trials.csv.gz",),
                )

            result.write_bytes(b"tampered-paper-result")
            with self.assertRaisesRegex(StudyGateError, "result changed"):
                load_report_safe_finalized_test_handoff(
                    ctx,
                    protocol,
                    required_artifacts=("test_policy_trials.csv.gz",),
                )

    def test_restricted_test_evidence_is_opaque_to_report_but_rehashed_by_full_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ctx = self._test_context(directory)
            protocol_input = Path(directory) / "protocol-input.txt"
            protocol_input.write_text("frozen protocol input", encoding="utf-8")
            protocol = freeze_protocol(
                ctx,
                {"artifact_manifest": frozen_file_manifest({"input": protocol_input})},
                {},
            )
            authorize_test_once(ctx, protocol)
            result = ctx.run_root / "results" / "test_policy_trials.csv.gz"
            result.write_bytes(b"report-consumable-result")
            restricted = []
            for name in (
                "test_source_truth.csv.gz",
                "test_analysis_manifest.csv.gz",
                "PREPARED_VISDRONE_BUNDLE.json",
            ):
                path = ctx.run_root / "restricted" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"restricted:{name}".encode("utf-8"))
                restricted.append(path)
            ledger_path = finalize_test_once(
                ctx,
                status="COMPLETE",
                result_artifacts=[result],
                restricted_artifacts=restricted,
            )
            ledger_text = ledger_path.read_text(encoding="utf-8")
            self.assertNotIn(str(restricted[0].resolve()), ledger_text)
            self.assertNotIn(str(restricted[1].resolve()), ledger_text)
            self.assertIn("restricted_evidence_receipt", ledger_text)

            full = load_finalized_test_handoff(
                ctx,
                protocol,
                required_artifacts=("test_policy_trials.csv.gz",),
            )
            self.assertEqual(set(full.restricted_artifacts), {path.name for path in restricted})

            # Once restricted data are unmounted, report generation still
            # authenticates the opaque nested receipt and public result only.
            restricted[0].unlink()
            report = load_report_safe_finalized_test_handoff(
                ctx,
                protocol,
                required_artifacts=("test_policy_trials.csv.gz",),
            )
            self.assertEqual(report.restricted_artifacts, {})
            self.assertIsNotNone(report.restricted_evidence_receipt_path)
            with self.assertRaisesRegex(StudyGateError, "Restricted test evidence"):
                load_finalized_test_handoff(
                    ctx,
                    protocol,
                    required_artifacts=("test_policy_trials.csv.gz",),
                )

            result.write_bytes(b"tampered-report-consumable-result")
            with self.assertRaisesRegex(StudyGateError, "result changed"):
                load_report_safe_finalized_test_handoff(
                    ctx,
                    protocol,
                    required_artifacts=("test_policy_trials.csv.gz",),
                )

    def test_default_handoff_requires_h1_streaming_aggregates_and_coverage_audit(self) -> None:
        required_h1 = {
            "test_h1_fixed_grid_group_aggregates.csv.gz",
            "test_h1_fixed_grid_coverage_audit.csv.gz",
        }
        self.assertTrue(required_h1.issubset(EMBEDDED_HANDOFF_REQUIRED_ARTIFACTS))
        self.assertIn(
            "test_dynamic_cartesian_coverage_audit.csv.gz",
            EMBEDDED_HANDOFF_REQUIRED_ARTIFACTS,
        )
        for compact_name in (
            "test_policy_partition_index.csv",
            "policy_pair_cells.csv.gz",
            "resource_episode_cells.csv.gz",
            "h4_matched_coverage_cells.csv.gz",
            "policy_reduction_receipt.json",
        ):
            self.assertIn(compact_name, EMBEDDED_HANDOFF_REQUIRED_ARTIFACTS)
        for prohibited_raw_name in (
            "test_policy_trials.csv.gz",
            "test_controller_behavior_detail.csv.gz",
            "test_exp5_risk_uncertainty_detail.csv.gz",
            "test_h4_matched_coverage_trials.csv.gz",
            "test_calibration_report.csv.gz",
        ):
            self.assertNotIn(prohibited_raw_name, EMBEDDED_HANDOFF_REQUIRED_ARTIFACTS)
        with tempfile.TemporaryDirectory() as directory:
            ctx = self._test_context(directory)
            protocol_input = Path(directory) / "protocol-input.txt"
            protocol_input.write_text("frozen", encoding="utf-8")
            protocol = freeze_protocol(
                ctx,
                {"artifact_manifest": frozen_file_manifest({"input": protocol_input})},
                {},
            )
            authorize_test_once(ctx, protocol)
            artifacts = []
            for name in EMBEDDED_HANDOFF_REQUIRED_ARTIFACTS:
                if name in required_h1:
                    continue
                path = ctx.run_root / "results" / name
                path.write_bytes(f"result:{name}".encode("utf-8"))
                artifacts.append(path)
            finalize_test_once(ctx, status="COMPLETE", result_artifacts=artifacts)
            with self.assertRaisesRegex(
                StudyGateError,
                "test_h1_fixed_grid_(group_aggregates|coverage_audit)",
            ):
                load_report_safe_finalized_test_handoff(ctx, protocol)


if __name__ == "__main__":
    unittest.main(verbosity=2)
