from __future__ import annotations

import hashlib
import inspect
import json
import unittest
from pathlib import Path

import yaml

from semuav.perception import VISDRONE_GENERATION_STRATEGY_BY_MODEL
from semuav.visdrone_models import (
    OFFICIAL_COMPACT_MODEL_REPOS,
    OFFICIAL_MODEL_BACKENDS,
    OFFICIAL_RUNTIME_MODEL_IDS,
    OFFICIAL_UPPER_BOUND_ID,
    run_one_image_runtime_smoke,
)


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_MODEL_IDS = (
    "qwen3_vl_2b",
    "smolvlm2_2b",
    "qwen3_vl_8b_upper_bound",
)
AMENDMENT = ROOT / "PRE_VALIDATION_MODEL_SET_AMENDMENT_20260914_V2R6.md"
V2R5_CORRECTION = ROOT / "PRE_VALIDATION_RUNTIME_CORRECTION_20260913_V2R5.md"
V2R5_CORRECTION_SHA256 = (
    "10fc6ad3a2d4ae41855ae5fce25c66cb20f4e6393d81f0f099d7a3071e317aae"
)


class V2R6ProspectiveModelSetAmendmentTests(unittest.TestCase):
    def test_active_runtime_constants_are_exact_ordered_three_model_set(self) -> None:
        self.assertEqual(OFFICIAL_RUNTIME_MODEL_IDS, ACTIVE_MODEL_IDS)
        self.assertEqual(
            tuple(OFFICIAL_COMPACT_MODEL_REPOS), ACTIVE_MODEL_IDS[:2]
        )
        self.assertEqual(
            tuple(OFFICIAL_MODEL_BACKENDS), ACTIVE_MODEL_IDS[:2]
        )
        self.assertEqual(OFFICIAL_UPPER_BOUND_ID, ACTIVE_MODEL_IDS[2])
        self.assertNotIn("moondream2", OFFICIAL_RUNTIME_MODEL_IDS)
        self.assertEqual(
            set(VISDRONE_GENERATION_STRATEGY_BY_MODEL),
            set(ACTIVE_MODEL_IDS),
        )
        self.assertNotIn("moondream2", VISDRONE_GENERATION_STRATEGY_BY_MODEL)

    def test_checked_in_config_matches_amended_model_population(self) -> None:
        config = yaml.safe_load(
            (ROOT / "configs" / "visdrone_paper.yaml").read_text(
                encoding="utf-8"
            )
        )
        configured_ids = tuple(
            item["id"] for item in config["models"]["vlms"]
        ) + (config["models"]["upper_bound"]["id"],)
        self.assertEqual(configured_ids, ACTIVE_MODEL_IDS)
        active_models = dict(config["models"])
        amendment = active_models.pop("model_set_amendment")
        self.assertNotIn("moondream2", json.dumps(active_models))
        self.assertEqual(
            amendment["excluded_models"],
            {
                "moondream2": {
                    "reason": "one_call_exact_schema_feasibility_failure",
                    "predecessor_status": "FAIL_SCHEMA",
                    "accuracy_evaluated": False,
                    "validation_or_test_outcome_used": False,
                    "scientific_claims_permitted": False,
                },
            },
        )
        self.assertNotIn("moondream", json.dumps(config["runtime"]).lower())

    def test_active_smoke_signature_retains_only_fail_closed_legacy_guards(self) -> None:
        signature = inspect.signature(run_one_image_runtime_smoke)
        for name in (
            "moondream_tokenizer_lock_path",
            "moondream_tokenizer_offline_checker",
        ):
            self.assertIn(name, signature.parameters)
            self.assertIsNone(signature.parameters[name].default)
        source = inspect.getsource(run_one_image_runtime_smoke)
        self.assertIn("v2r13 runtime excludes Moondream", source)
        self.assertNotIn("bind_moondream_tokenizer_lock(", source)

    def test_amendment_is_explicit_prospective_and_preserves_failed_v2r5(self) -> None:
        self.assertTrue(AMENDMENT.is_file())
        amendment = AMENDMENT.read_text(encoding="utf-8")
        for token in (
            "release_20260914_visdrone_paper_v2r6",
            "semuav_visdrone_v2r6",
            "V2R5_RUNTIME_SMOKE=FAILED_3_OF_4_PASS",
            "V2R5_ALL_PASSED=false",
            "V2R5_FAILED_EVIDENCE=IMMUTABLE",
            "V2R5_VALIDATION_DATA_READ=NO",
            "V2R5_UNTOUCHED_TEST_DATA_READ=NO",
            "V2R5_EXPERIMENTS_1_TO_7_EXECUTED=NO",
            "V2R6_MODEL_SET_AMENDMENT=PROSPECTIVE_PRE_VALIDATION",
            "V2R6_REGISTERED_VLM_COUNT=3",
            "V2R6_COMPACT_SELECTION_CANDIDATE_COUNT=2",
            "V2R6_REQUIRED_RUNTIME_SMOKE=EXACT_3_OF_3_PASS",
            "MOONDREAM_ACCURACY_CONCLUSION=NOT_EVALUATED",
            "NEXT_GATE=FRESH_V2R6_THREE_MODEL_DEVELOPMENT_RUNTIME_SMOKE",
        ):
            self.assertIn(token, amendment)

        for model_id in ACTIVE_MODEL_IDS:
            self.assertIn(f"`{model_id}`", amendment)
        self.assertIn("`moondream2` is excluded", amendment)
        self.assertIn("no v2r5 PASS row is carried forward", amendment)
        self.assertIn("This is not an observed accuracy", amendment)

        self.assertEqual(
            hashlib.sha256(V2R5_CORRECTION.read_bytes()).hexdigest(),
            V2R5_CORRECTION_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
