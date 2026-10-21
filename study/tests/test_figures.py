from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from semuav.figures import (
    HYPOTHESIS_EFFECT_COLUMNS,
    OUTPUT_DPI,
    build_hypothesis_effects_frame,
    generate_publication_figures,
)


class PublicationFigureTests(unittest.TestCase):
    def hypothesis_reports(self) -> dict[str, pd.DataFrame]:
        h1 = pd.DataFrame([
            {
                "semantic_mode": mode,
                "image_variant": "jpeg_q80",
                "utility_difference": estimate,
                "utility_ci_low": estimate - 0.02,
                "utility_ci_high": estimate + 0.02,
            }
            for mode, estimate in (("L2_SEMANTICS", 0.04), ("L3_SEMANTICS_ROI", 0.06))
        ])
        h2 = pd.DataFrame([
            {
                "method": "proposed_full",
                "reference": baseline,
                "endpoint": "correct_timely_action",
                "estimate": estimate,
                "ci_low": estimate - 0.02,
                "ci_high": estimate + 0.02,
            }
            for baseline, estimate in (("fixed_image", 0.08), ("detector_only", 0.11))
        ])
        h3 = pd.DataFrame([{
            "critical_miss_reduction": 0.09,
            "ci_low": 0.04,
            "ci_high": 0.14,
        }])
        h4 = pd.DataFrame([{
            "endpoint": (
                "conjunctive_matched_coverage_and_high_confidence_"
                "semantic_error_reduction"
            ),
            # Deliberately conflicting legacy aliases prove that the figure
            # adapter reads only the two registered component-specific fields.
            "estimate": 0.99,
            "ci_low": 0.98,
            "ci_high": 1.00,
            "holm_adjusted_p": 0.99,
            "matched_coverage_error_reduction": 0.03,
            "matched_coverage_ci_low": 0.01,
            "matched_coverage_ci_high": 0.05,
            "matched_coverage_holm_adjusted_p": 0.04,
            "high_confidence_error_reduction": 0.05,
            "high_confidence_ci_low": 0.02,
            "high_confidence_ci_high": 0.08,
            "high_confidence_holm_adjusted_p": 0.03,
            "execution_status": "COMPLETE",
            "not_estimable_reason": "",
            "claim_supported": True,
        }])
        status = pd.DataFrame([
            {
                "hypothesis": f"H{number}",
                "execution_status": (
                    "EMPIRICAL_COMPLETE" if number < 5
                    else "TARGET_METRICS_COMPLETE_RETEST_REQUIRED"
                ),
                "claim_supported": number < 5,
            }
            for number in range(1, 6)
        ])
        return {"h1": h1, "h2": h2, "h3": h3, "h4": h4, "status": status}

    def complete_frames(self) -> dict[str, pd.DataFrame]:
        manifest = pd.DataFrame([
            {"sample_id": "s1", "visual_group_id": "g1", "split": "train", "dataset": "A"},
            {"sample_id": "s2", "visual_group_id": "g1", "split": "train", "dataset": "A"},
            {"sample_id": "s3", "visual_group_id": "g2", "split": "validation", "dataset": "A"},
            {"sample_id": "s4", "visual_group_id": "g3", "split": "validation", "dataset": "B"},
            {"sample_id": "s5", "visual_group_id": "g4", "split": "test", "dataset": "A"},
            {"sample_id": "s6", "visual_group_id": "g5", "split": "test", "dataset": "B"},
        ])
        exp1 = pd.DataFrame([
            {
                "model_id": "detector_only", "action_accuracy": 0.72,
                "critical_action_recall": 0.68, "semantic_valid_rate": 1.0,
                "schema_valid_rate": 1.0, "selective_action_accuracy": 0.74,
                "grounded_hallucination_rate": 0.05,
                "full_path_latency_p50_ms": 12.0, "full_path_latency_p95_ms": 18.0,
            },
            {
                "model_id": "qwen", "action_accuracy": 0.88,
                "critical_action_recall": 0.91, "semantic_valid_rate": 0.97,
                "schema_valid_rate": 0.99, "selective_action_accuracy": 0.93,
                "grounded_hallucination_rate": 0.03,
                "full_path_latency_p50_ms": 82.0, "full_path_latency_p95_ms": 120.0,
            },
        ])
        exp2_rows = []
        for sample, adjustment in (("s1", 0.0), ("s2", 0.02)):
            for mode, variant, wire, correct in (
                ("L1_ALERT", "alert", 180, 0.61),
                ("L2_SEMANTICS", "semantic", 900, 0.84),
                ("L3_SEMANTICS_ROI", "semantic_roi", 8_000, 0.91),
                ("L5_IMAGE_OR_CLIP", "jpeg_q80", 80_000, 0.95),
            ):
                exp2_rows.append({
                    "sample_id": sample, "mode": mode, "variant": variant,
                    "wire_bytes": wire, "native_receiver_correct": correct + adjustment,
                    "source_model": "measured_model",
                })
        exp2 = pd.DataFrame(exp2_rows)
        exp3 = pd.DataFrame([
            {"rate_mbps": rate, "rtt_ms": rtt, "deadline_success_probability": value}
            for rate, rtt, value in (
                (0.1, 20, 0.25), (1.0, 20, 0.81),
                (0.1, 200, 0.08), (1.0, 200, 0.62),
            )
        ])
        exp4_rows = []
        modes = ("L1_ALERT", "L2_SEMANTICS", "L5_IMAGE_OR_CLIP")
        for policy_index, policy in enumerate(("fixed_image", "proposed_full")):
            for trial in range(6):
                exp4_rows.append({
                    "policy": policy,
                    "correct_timely_action": bool(
                        trial < (3 if policy == "fixed_image" else 5)
                    ),
                    "total_wire_bytes": (80_000 if policy == "fixed_image" else 2_000 + trial * 100),
                    "energy_j": (1.2 if policy == "fixed_image" else 0.35 + trial * 0.01),
                    "selected_mode": (
                        "L5_IMAGE_OR_CLIP" if policy == "fixed_image" else modes[(trial + policy_index) % len(modes)]
                    ),
                })
        exp4 = pd.DataFrame(exp4_rows)
        exp5 = pd.DataFrame([
            {
                "subset": subset, "policy": policy,
                "critical_miss_rate": miss, "false_alert_rate": alert,
                "high_confidence_semantic_error_rate": semantic_error,
            }
            for subset, policy, miss, alert, semantic_error in (
                ("natural_prevalence", "fixed_image", 0.20, 0.08, 0.10),
                ("natural_prevalence", "proposed_full", 0.08, 0.05, 0.03),
                ("ood", "fixed_image", 0.28, 0.12, 0.18),
                ("ood", "proposed_full", 0.12, 0.07, 0.06),
            )
        ])
        exp6 = pd.DataFrame([
            {"case": "data_loss_retransmission", "pass_rate": 1.0},
            {"case": "application_ack_loss_safe_uncertainty", "pass_rate": 1.0},
            {"case": "corruption_crc_retransmission", "pass_rate": 2 / 3},
        ])
        exp7 = pd.DataFrame([
            {"item": "python", "status": "PASS", "value": "3.11", "required_for": "all"},
            {"item": "cuda", "status": "PASS", "value": "available", "required_for": "real_profiles"},
            {"item": "tc_netem", "status": "FAIL", "value": "missing", "required_for": "experiment_7b"},
        ])
        hypotheses = pd.DataFrame([
            {
                "hypothesis": f"H{number}", "comparison": f"comparison_{number}",
                "estimate": 0.01 * number, "ci_low": -0.01 + 0.01 * number,
                "ci_high": 0.03 + 0.01 * number,
            }
            for number in range(1, 6)
        ])
        return {
            "manifest": manifest,
            "exp1_summary": exp1,
            "exp2_frontier": exp2,
            "exp3_summary": exp3,
            "exp4_policy": exp4,
            "exp5_summary": exp5,
            "exp6_campaign_summary": exp6,
            "exp7_readiness": exp7,
            "hypothesis_effects": hypotheses,
        }

    def tearDown(self) -> None:
        plt.close("all")

    def test_complete_suite_writes_three_formats_and_hash_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            figures = generate_publication_figures(
                self.complete_frames(), directory, scientific=False,
            )
            self.assertEqual(len(figures), 11)
            for figure_id, figure in figures.items():
                for extension in ("png", "pdf", "svg"):
                    output = Path(directory) / f"{figure_id}.{extension}"
                    self.assertTrue(output.is_file(), output)
                    self.assertGreater(output.stat().st_size, 100)
                watermark_text = [text.get_text() for text in figure.texts]
                self.assertIn("NON-SCIENTIFIC FIXTURE", watermark_text)

            png = Path(directory) / "exp2_pareto_frontier.png"
            with Image.open(png) as image:
                # A publication-size 8.6-inch figure at 300 dpi remains well
                # above screen-only resolution even after tight cropping.
                self.assertGreater(image.width, 1_800)
                self.assertGreater(image.height, 1_000)
                self.assertEqual(OUTPUT_DPI, 300)

            manifest = pd.read_csv(Path(directory) / "figure_manifest.csv")
            self.assertEqual(set(manifest["status"]), {"generated"})
            self.assertEqual(len(manifest), 11)
            payload = json.loads(
                (Path(directory) / "figure_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(payload["figures"]), 11)
            self.assertEqual(set(payload["source_frame_hashes"]), set(self.complete_frames()))
            self.assertTrue(
                all(len(digest) == 64 for digest in payload["source_frame_hashes"].values())
            )

    def test_missing_columns_are_skipped_explicitly_without_placeholder_plot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            figures = generate_publication_figures(
                {"manifest": pd.DataFrame({"sample_id": ["s1"]})},
                directory,
                scientific=True,
            )
            self.assertEqual(figures, {})
            manifest = pd.read_csv(Path(directory) / "figure_manifest.csv")
            dataset_row = manifest.loc[
                manifest["figure_id"].eq("dataset_split_balance")
            ].iloc[0]
            self.assertEqual(dataset_row["status"], "skipped")
            self.assertIn("missing required columns: split", dataset_row["reason"])
            self.assertFalse(
                (Path(directory) / "dataset_split_balance.png").exists()
            )
            self.assertTrue(manifest["reason"].astype(str).str.len().gt(0).all())

    def test_scientific_plot_is_unwatermarked_but_fixture_marker_fails_safe(self) -> None:
        frames = self.complete_frames()
        with tempfile.TemporaryDirectory() as directory:
            figures = generate_publication_figures(
                {"exp1_summary": frames["exp1_summary"]},
                directory,
                scientific=True,
            )
            self.assertEqual(set(figures), {"exp1_perception_metrics", "exp1_latency"})
            for figure in figures.values():
                self.assertNotIn(
                    "NON-SCIENTIFIC FIXTURE",
                    [text.get_text() for text in figure.texts],
                )

        fixture_frontier = frames["exp2_frontier"].copy()
        fixture_frontier["source_model"] = "GENERATED_FIXTURE"
        with tempfile.TemporaryDirectory() as directory:
            figures = generate_publication_figures(
                {"exp2_frontier": fixture_frontier},
                directory,
                scientific=True,
            )
            figure = figures["exp2_pareto_frontier"]
            self.assertIn(
                "NON-SCIENTIFIC FIXTURE",
                [text.get_text() for text in figure.texts],
            )
            manifest = pd.read_csv(Path(directory) / "figure_manifest.csv")
            row = manifest.loc[
                manifest["figure_id"].eq("exp2_pareto_frontier")
            ].iloc[0]
            self.assertTrue(bool(row["watermarked_non_scientific"]))

    def test_input_contract_rejects_non_dataframe_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(TypeError, "DataFrame or None"):
                generate_publication_figures(
                    {"exp1_summary": "not-a-frame"},  # type: ignore[arg-type]
                    directory,
                    scientific=False,
                )

    def test_combined_hypothesis_effects_preserve_test_intervals_and_h5_status(self) -> None:
        reports = self.hypothesis_reports()
        effects = build_hypothesis_effects_frame(
            h1_report=reports["h1"],
            h2_report=reports["h2"],
            h3_report=reports["h3"],
            h4_report=reports["h4"],
            hypothesis_status=reports["status"],
        )
        self.assertEqual(tuple(effects.columns), HYPOTHESIS_EFFECT_COLUMNS)
        available = effects[effects["effect_available"].astype(bool)]
        self.assertEqual(set(available["hypothesis"]), {"H1", "H2", "H3", "H4"})
        self.assertEqual(len(available[available["hypothesis"].eq("H1")]), 2)
        h4 = available[available["hypothesis"].eq("H4")]
        self.assertEqual(len(h4), 2)
        self.assertEqual(
            set(h4["component_hypothesis"]),
            {
                "H4_MATCHED_COVERAGE_SELECTIVE_RISK",
                "H4_HIGH_CONFIDENCE_SEMANTIC_ERROR",
            },
        )
        h4_by_endpoint = h4.set_index("endpoint")
        matched = h4_by_endpoint.loc["matched_coverage_selective_risk_reduction"]
        high_confidence = h4_by_endpoint.loc[
            "high_confidence_semantic_error_reduction"
        ]
        self.assertAlmostEqual(float(matched["estimate"]), 0.03)
        self.assertAlmostEqual(float(matched["ci_low"]), 0.01)
        self.assertAlmostEqual(float(matched["ci_high"]), 0.05)
        self.assertAlmostEqual(float(matched["holm_adjusted_p"]), 0.04)
        self.assertIn("matched_coverage", matched["effect_provenance"])
        self.assertAlmostEqual(float(high_confidence["estimate"]), 0.05)
        self.assertAlmostEqual(float(high_confidence["ci_low"]), 0.02)
        self.assertAlmostEqual(float(high_confidence["ci_high"]), 0.08)
        self.assertAlmostEqual(float(high_confidence["holm_adjusted_p"]), 0.03)
        self.assertIn("high_confidence", high_confidence["effect_provenance"])
        self.assertEqual(
            set(h4["p_value_provenance"]),
            {"holm_family_adjusted_two_sided_p"},
        )
        h1 = available[available["hypothesis"].eq("H1")]
        self.assertTrue((h1["ci_low"] <= h1["estimate"]).all())
        self.assertTrue((h1["ci_high"] >= h1["estimate"]).all())
        h5 = effects[effects["hypothesis"].eq("H5")]
        self.assertEqual(len(h5), 1)
        self.assertFalse(bool(h5.iloc[0]["effect_available"]))
        self.assertTrue(pd.isna(h5.iloc[0]["estimate"]))
        self.assertIn("no standalone scalar estimand", h5.iloc[0]["unavailable_reason"])

    def test_h4_unavailable_components_remain_explicit_and_do_not_publish_sentinel_p(self) -> None:
        reports = self.hypothesis_reports()
        reports["h4"] = reports["h4"].assign(
            matched_coverage_error_reduction=float("nan"),
            matched_coverage_ci_low=float("nan"),
            matched_coverage_ci_high=float("nan"),
            matched_coverage_holm_adjusted_p=1.0,
            high_confidence_error_reduction=float("nan"),
            high_confidence_ci_low=float("nan"),
            high_confidence_ci_high=float("nan"),
            high_confidence_holm_adjusted_p=1.0,
            execution_status="NOT_ESTIMABLE",
            not_estimable_reason="registered H4 denominator gate failed",
            claim_supported=False,
        )
        effects = build_hypothesis_effects_frame(
            h1_report=reports["h1"],
            h2_report=reports["h2"],
            h3_report=reports["h3"],
            h4_report=reports["h4"],
            hypothesis_status=reports["status"],
        )
        h4 = effects[effects["hypothesis"].eq("H4")]
        self.assertEqual(len(h4), 2)
        self.assertFalse(h4["effect_available"].astype(bool).any())
        self.assertTrue(
            h4[["estimate", "ci_low", "ci_high", "holm_adjusted_p"]]
            .isna().all().all()
        )
        self.assertEqual(set(h4["execution_status"]), {"NOT_ESTIMABLE"})
        self.assertEqual(
            set(h4["unavailable_reason"]),
            {"registered H4 denominator gate failed"},
        )

    def test_not_estimable_h2_cannot_leak_accidentally_finite_inferential_values(self) -> None:
        reports = self.hypothesis_reports()
        reports["h2"] = reports["h2"].assign(
            execution_status="NOT_ESTIMABLE",
            not_estimable_reason="registered temporal group minimum was not met",
            claim_supported=False,
            holm_p=0.001,
        )
        reports["status"].loc[
            reports["status"]["hypothesis"].eq("H2"),
            ["execution_status", "claim_supported"],
        ] = ["NOT_ESTIMABLE", False]
        effects = build_hypothesis_effects_frame(
            h1_report=reports["h1"],
            h2_report=reports["h2"],
            h3_report=reports["h3"],
            h4_report=reports["h4"],
            hypothesis_status=reports["status"],
        )
        h2 = effects[effects["hypothesis"].eq("H2")]
        self.assertEqual(len(h2), 2)
        self.assertFalse(h2["effect_available"].astype(bool).any())
        self.assertTrue(
            h2[["estimate", "ci_low", "ci_high", "holm_adjusted_p"]]
            .isna().all().all()
        )
        self.assertEqual(set(h2["execution_status"]), {"NOT_ESTIMABLE"})
        self.assertEqual(
            set(h2["unavailable_reason"]),
            {"registered temporal group minimum was not met"},
        )

    def test_missing_h4_component_columns_are_rejected_without_legacy_alias_fallback(self) -> None:
        reports = self.hypothesis_reports()
        legacy_only = reports["h4"][[
            "high_confidence_error_reduction", "ci_low", "ci_high",
        ]]
        with self.assertRaisesRegex(
            ValueError, "component-specific effect columns",
        ):
            build_hypothesis_effects_frame(h4_report=legacy_only)

    def test_verified_embedded_effects_become_h5_replication_components(self) -> None:
        reports = self.hypothesis_reports()
        reports["status"].loc[
            reports["status"]["hypothesis"].eq("H5"), "execution_status"
        ] = "EMPIRICAL_COMPLETE"
        embedded_effects = pd.DataFrame([
            {
                "hypothesis": hypothesis,
                "comparison": comparison,
                "endpoint": endpoint,
                "estimate": estimate,
                "ci_low": estimate - 0.02,
                "ci_high": estimate + 0.02,
            }
            for hypothesis, comparison, endpoint, estimate in (
                ("H2", "fixed_image", "correct_timely_action", 0.07),
                (
                    "H3", "proposed_full_vs_channel_only_hysteresis",
                    "critical_miss_reduction", 0.08,
                ),
                (
                    "H4", "proposed_full_vs_ablate_evidence_escalation",
                    "high_confidence_semantic_error_reduction", 0.04,
                ),
            )
        ])
        embedded_status = pd.DataFrame([
            {
                "hypothesis": hypothesis,
                "execution_status": "EMBEDDED_RETEST_COMPLETE",
                "claim_supported": True,
                "raw_recomputed": True,
                "source_kind": "hashed_raw_stage_trial_recomputation",
            }
            for hypothesis in ("H2", "H3", "H4")
        ])
        effects = build_hypothesis_effects_frame(
            h1_report=reports["h1"],
            h2_report=reports["h2"],
            h3_report=reports["h3"],
            h4_report=reports["h4"],
            hypothesis_status=reports["status"],
            embedded_retest_effects=embedded_effects,
            embedded_retest_status=embedded_status,
        )
        h5 = effects[effects["hypothesis"].eq("H5")]
        self.assertEqual(len(h5), 3)
        self.assertTrue(h5["effect_available"].astype(bool).all())
        self.assertEqual(set(h5["component_hypothesis"]), {"H2", "H3", "H4"})
        self.assertEqual(
            set(h5["effect_provenance"]),
            {"verified_hashed_raw_stage_trial_recomputation"},
        )
        self.assertTrue(h5[["estimate", "ci_low", "ci_high"]].notna().all().all())

    def test_unverified_embedded_summary_is_rejected(self) -> None:
        embedded_effects = pd.DataFrame([{
            "hypothesis": "H2",
            "comparison": "baseline",
            "endpoint": "correct_timely_action",
            "estimate": 0.1,
            "ci_low": 0.05,
            "ci_high": 0.15,
        }])
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            build_hypothesis_effects_frame(
                embedded_retest_effects=embedded_effects,
            )
        unverified_status = pd.DataFrame([
            {
                "hypothesis": hypothesis,
                "claim_supported": False,
                "raw_recomputed": False,
                "source_kind": "self_reported_summary",
            }
            for hypothesis in ("H2", "H3", "H4")
        ])
        with self.assertRaisesRegex(ValueError, "raw_recomputed=True"):
            build_hypothesis_effects_frame(
                embedded_retest_effects=embedded_effects,
                embedded_retest_status=unverified_status,
            )

    def test_forest_renders_h1_h4_and_explicit_non_scalar_h5_status(self) -> None:
        reports = self.hypothesis_reports()
        effects = build_hypothesis_effects_frame(
            h1_report=reports["h1"],
            h2_report=reports["h2"],
            h3_report=reports["h3"],
            h4_report=reports["h4"],
            hypothesis_status=reports["status"],
        )
        with tempfile.TemporaryDirectory() as directory:
            figures = generate_publication_figures(
                {"hypothesis_effects": effects}, directory, scientific=True,
            )
            self.assertIn("hypotheses_h1_h5_forest", figures)
            figure = figures["hypotheses_h1_h5_forest"]
            axis_text = " ".join(
                text.get_text() for axis in figure.axes for text in axis.texts
            )
            self.assertIn("No registered scalar effect", axis_text)
            tick_text = " ".join(
                tick.get_text() for axis in figure.axes for tick in axis.get_yticklabels()
            )
            self.assertIn("Matched Coverage Selective Risk", tick_text)
            self.assertIn("High Confidence Semantic Error", tick_text)
            manifest = pd.read_csv(Path(directory) / "figure_manifest.csv")
            row = manifest.loc[
                manifest["figure_id"].eq("hypotheses_h1_h5_forest")
            ].iloc[0]
            self.assertEqual(row["status"], "generated")

    def test_notebook_publication_uses_complete_frozen_split_manifest(self) -> None:
        builder = (
            Path(__file__).resolve().parents[1] / "build_full_notebook.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"dataset_manifest": publication_split_manifest', builder)
        self.assertIn(
            'resolve_frozen_artifact(\n            frozen_test_protocol, "split_manifest"',
            builder,
        )
        publication_section = builder.split("publication_frames = {", 1)[1]
        self.assertNotIn("analysis_manifest if analysis_manifest", publication_section)


if __name__ == "__main__":
    unittest.main(verbosity=2)
