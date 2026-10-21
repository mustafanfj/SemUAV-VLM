import unittest

import pandas as pd

from semuav.experiments import select_experiment3_candidate_variants
from semuav.registry import StudyGateError


class Experiment3VariantSelectionTests(unittest.TestCase):
    def candidates(self):
        rows = []
        for sample in ("s1", "s2"):
            for mode, source, variants in (
                ("L2_SEMANTICS", "m", (("a", 10), ("b", 20), ("c", 30), ("d", 40))),
                ("L5_IMAGE_OR_CLIP", "ground", (("jpeg_low", 100), ("jpeg_high", 1000))),
            ):
                for variant, wire in variants:
                    rows.append({
                        "sample_id": sample, "mode": mode, "source_model": source,
                        "variant": variant, "wire_bytes": wire,
                        "correctness_must_not_drive_selection": sample == "s1",
                    })
        return pd.DataFrame(rows)

    def test_validation_selection_spans_wire_size_and_is_bounded(self):
        selected, spec = select_experiment3_candidate_variants(self.candidates(), 2)
        semantic = spec[spec["mode"].eq("L2_SEMANTICS")]
        self.assertEqual(set(semantic["variant"]), {"a", "d"})
        self.assertEqual(selected["sample_id"].nunique(), 2)
        self.assertLessEqual(spec.groupby(["mode", "source_model"]).size().max(), 2)

    def test_test_phase_applies_frozen_validation_spec(self):
        _, spec = select_experiment3_candidate_variants(self.candidates(), 2)
        changed = self.candidates().copy()
        changed["wire_bytes"] = changed["wire_bytes"].max() - changed["wire_bytes"]
        selected, applied = select_experiment3_candidate_variants(
            changed, 2, frozen_spec=spec,
        )
        self.assertEqual(
            set(selected[["mode", "source_model", "variant"]].itertuples(index=False, name=None)),
            set(spec[["mode", "source_model", "variant"]].itertuples(index=False, name=None)),
        )
        pd.testing.assert_frame_equal(applied, spec)

    def test_incomplete_frozen_variant_fails_closed(self):
        _, spec = select_experiment3_candidate_variants(self.candidates(), 2)
        incomplete = self.candidates().loc[
            ~(
                self.candidates()["sample_id"].eq("s2")
                & self.candidates()["variant"].eq("d")
            )
        ]
        with self.assertRaisesRegex(StudyGateError, "absent or incomplete"):
            select_experiment3_candidate_variants(incomplete, 2, frozen_spec=spec)


if __name__ == "__main__":
    unittest.main()
