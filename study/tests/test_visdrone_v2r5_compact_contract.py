from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from semuav.perception import (
    InferenceRequest,
    VISDRONE_GENERATION_POLICY_VERSION,
    VISDRONE_MISSION_PROMPT,
    VISDRONE_PROPOSAL_CONTEXT_SCHEMA,
    VISDRONE_PROPOSAL_COORDINATE_MAX,
    VISDRONE_PROPOSAL_DECISION_CONTRACT_VERSION,
    VISDRONE_REGISTERED_MAX_NEW_TOKENS,
    _moondream_native_query_budget,
    extract_exact_visdrone_proposal_decision,
    visdrone_proposal_decision_context,
)
from semuav.registry import StudyGateError


class _Encoding:
    def __init__(self, ids):
        self.ids = list(ids)


class _Tokenizer:
    def __init__(self, ids):
        self.ids = list(ids)

    def encode(self, question):
        if not isinstance(question, str):
            raise AssertionError("question must be text")
        return _Encoding(self.ids)


def fake_moondream(token_ids):
    native = SimpleNamespace(
        tokenizer=_Tokenizer(token_ids),
        config=SimpleNamespace(
            text=SimpleNamespace(max_context=2048, vocab_size=51200),
            tokenizer=SimpleNamespace(
                templates={"query": {"prefix": [1, 15381, 2], "suffix": [3]}},
            ),
        ),
    )
    return SimpleNamespace(model=native)


class V2R5CompactProposalContractTests(unittest.TestCase):
    def request(self):
        return InferenceRequest(
            observation_id="train-canary",
            image_path="/not-read.jpg",
            mission="visdrone_urban",
            capture_time_us=1,
            telemetry={
                "mission_profile": "heavy_vehicle",
                "registered_roi_xyxy": [0.25, 0.25, 0.75, 0.85],
            },
            detector_proposals=({
                "label": "truck",
                "bbox_normalized_xyxy": [0.1, 0.2, 0.3, 0.4],
                "confidence": 0.9,
                "track_id": 77,
                "gold_action": "must_not_leak",
            },),
        )

    def test_registered_identities_and_budget_are_prospective(self):
        self.assertEqual(
            VISDRONE_PROPOSAL_DECISION_CONTRACT_VERSION,
            "semuav.visdrone-proposal-decision.v2",
        )
        self.assertEqual(
            VISDRONE_GENERATION_POLICY_VERSION,
            "semuav.visdrone-generation-policy.v5",
        )
        self.assertEqual(VISDRONE_REGISTERED_MAX_NEW_TOKENS, 256)
        self.assertLess(len(VISDRONE_MISSION_PROMPT), 800)

    def test_context_is_compact_processed_plane_and_outcome_blind(self):
        context = visdrone_proposal_decision_context(
            self.request(),
            (100, 200),
            (768, 768),
            preprocess={
                "source_size": [100, 200],
                "model_input_size": [768, 768],
                "content_xyxy": [192, 0, 576, 768],
            },
        )
        self.assertEqual(
            context,
            {
                "v": VISDRONE_PROPOSAL_CONTEXT_SCHEMA,
                "q": VISDRONE_PROPOSAL_COORDINATE_MAX,
                "p": [[0, "truck", 1228, 819, 1638, 1639]],
            },
        )
        serialized = json.dumps(context, separators=(",", ":"))
        for forbidden in (
            "confidence", "track_id", "gold_action", "must_not_leak",
            "mission_profile", "telemetry", "observation_id",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_short_wire_is_strict_and_normalized_before_composition(self):
        observed = extract_exact_visdrone_proposal_decision(
            '{"e":true,"a":[0,2],"l":[2]}',
            proposal_count=3,
        )
        self.assertEqual(observed, {
            "sufficient_evidence": True,
            "accepted_proposal_ids": [0, 2],
            "limited_visibility_proposal_ids": [2],
        })
        invalid = (
            '{"sufficient_evidence":true,"accepted_proposal_ids":[],"limited_visibility_proposal_ids":[]}',
            'result={"e":true,"a":[],"l":[]}',
            '{"e":true,"e":false,"a":[],"l":[]}',
            '{"e":true,"a":[1,0],"l":[]}',
            '{"e":false,"a":[0],"l":[]}',
        )
        for text in invalid:
            with self.subTest(text=text), self.assertRaises((TypeError, ValueError)):
                extract_exact_visdrone_proposal_decision(text, proposal_count=3)

    def test_maximum_legal_wire_object_fits_registered_output_reserve(self):
        maximum = json.dumps(
            {"e": True, "a": list(range(32)), "l": list(range(32))},
            separators=(",", ":"),
        )
        self.assertEqual(len(maximum), 194)
        self.assertLess(len(maximum), VISDRONE_REGISTERED_MAX_NEW_TOKENS)

    def test_moondream_budget_counts_native_template_and_generation_reserve(self):
        budget = _moondream_native_query_budget(
            fake_moondream(range(1000)),
            SimpleNamespace(pos=730),
            "compact query",
            VISDRONE_REGISTERED_MAX_NEW_TOKENS,
        )
        self.assertEqual(budget["native_template_tokens"], 4)
        self.assertEqual(budget["prefill_tokens"], 1734)
        self.assertEqual(budget["required_tokens"], 1990)
        self.assertEqual(budget["effective_context_tokens"], 2048)
        self.assertEqual(budget["context_margin_tokens"], 58)

    def test_moondream_budget_rejects_overflow_and_bad_token_id_before_query(self):
        with self.assertRaisesRegex(StudyGateError, "required=2050, limit=2048"):
            _moondream_native_query_budget(
                fake_moondream(range(1060)),
                SimpleNamespace(pos=730),
                "too long",
                VISDRONE_REGISTERED_MAX_NEW_TOKENS,
            )
        with self.assertRaisesRegex(StudyGateError, "out-of-range token ID"):
            _moondream_native_query_budget(
                fake_moondream([51199, 51200]),
                SimpleNamespace(pos=730),
                "bad tokenizer route",
                VISDRONE_REGISTERED_MAX_NEW_TOKENS,
            )


if __name__ == "__main__":
    unittest.main()
