from __future__ import annotations

import unittest

import pandas as pd

from semuav.config import AnnotationConfig
from semuav.data import audit_annotations


class AnnotationGeometryAuditTests(unittest.TestCase):
    @staticmethod
    def config() -> AnnotationConfig:
        return AnnotationConfig(
            min_observed_agreement=0.80,
            min_chance_corrected_agreement=0.80,
            min_rare_chance_corrected_agreement=0.70,
            min_evidence_geometry_iou=0.70,
        )

    @staticmethod
    def annotations(second_box: str) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "sample_id": "s1", "task": "evidence_roi",
                "label": "[0.0,0.0,0.5,0.5]", "annotator_id": "a1",
                "blinded": True, "adjudicated_label": "[0.0,0.0,0.5,0.5]",
                "adjudicator_id": "judge", "is_critical": True,
            },
            {
                "sample_id": "s1", "task": "evidence_roi",
                "label": second_box, "annotator_id": "a2",
                "blinded": True, "adjudicated_label": "[0.0,0.0,0.5,0.5]",
                "adjudicator_id": "judge", "is_critical": True,
            },
            {
                "sample_id": "s1", "task": "event",
                "label": "visible_road_obstruction", "annotator_id": "a1",
                "blinded": True, "adjudicated_label": "",
                "adjudicator_id": "", "is_critical": True,
            },
            {
                "sample_id": "s1", "task": "event",
                "label": "visible_road_obstruction", "annotator_id": "a2",
                "blinded": True, "adjudicated_label": "",
                "adjudicator_id": "", "is_critical": True,
            },
        ])

    def test_pairwise_and_adjudication_iou_pass_above_threshold(self) -> None:
        report = audit_annotations(
            self.annotations("[0.02,0.02,0.52,0.52]"),
            pd.DataFrame({"sample_id": ["s1"]}),
            self.config(),
        )
        geometry = report.by_task["__geometry__:evidence_roi"]
        self.assertTrue(report.passed, report.errors)
        self.assertGreater(float(geometry["pairwise_median_iou"]), 0.70)
        self.assertGreater(float(geometry["adjudication_median_iou"]), 0.70)
        self.assertTrue(bool(geometry["pairwise_gate_passed"]))
        self.assertTrue(bool(geometry["adjudication_gate_passed"]))
        self.assertEqual(float(report.observed_agreement), 1.0)
        self.assertEqual(float(report.chance_corrected_agreement), 1.0)
        self.assertNotIn("evidence_roi", report.by_task)
        self.assertIn("__rare__:event", report.by_task)
        self.assertNotIn("__rare__:evidence_roi", report.by_task)

    def test_geometry_gate_fails_below_threshold_without_relaxing_kappa(self) -> None:
        report = audit_annotations(
            self.annotations("[0.5,0.5,1.0,1.0]"),
            pd.DataFrame({"sample_id": ["s1"]}),
            self.config(),
        )
        geometry = report.by_task["__geometry__:evidence_roi"]
        self.assertFalse(report.passed)
        self.assertEqual(float(geometry["pairwise_median_iou"]), 0.0)
        self.assertTrue(any("pairwise median IoU" in error for error in report.errors))
        self.assertTrue(any("adjudication median IoU" in error for error in report.errors))
        # Categorical agreement remains perfect and excludes coordinate strings;
        # only the dedicated localization gate fails.
        self.assertEqual(float(report.observed_agreement), 1.0)
        self.assertEqual(float(report.chance_corrected_agreement), 1.0)
        self.assertNotIn("evidence_roi", report.by_task)


if __name__ == "__main__":
    unittest.main()
