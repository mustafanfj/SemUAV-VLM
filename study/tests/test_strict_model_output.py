from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml

from semuav.perception import (
    MODEL_AUTHORED_PACKET_KEYS,
    InferenceRequest,
    VISDRONE_MISSION_PROMPT,
    VISDRONE_MISSION_PROMPT_PATH,
    VISDRONE_IMAGE_PLANE_IOU,
    VISDRONE_MAX_PROPOSALS,
    VISDRONE_MIN_PROPOSAL_CONFIDENCE,
    VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256,
    VISDRONE_PROPOSAL_DECISION_CONTRACT_VERSION,
    VISDRONE_PRIORITY_LABELS_BY_PROFILE,
    VISDRONE_REGISTERED_ROI_XYXY,
    VISDRONE_SOURCE_RULE_PROMPT,
    VISDRONE_SOURCE_RULE_PROMPT_PATH,
    canonical_model_context,
    compose_visdrone_semantic_packet,
    extract_exact_json_object,
    extract_exact_visdrone_proposal_decision,
    prompt_for_mission,
    visdrone_proposal_decision_context,
)
from semuav.protocol import validate_semantic_content_packet


class StrictModelOutputTests(unittest.TestCase):
    def packet(self) -> dict[str, object]:
        return {
            "schema_version": "semuav.v1",
            "observation_id": "o1",
            "mission": "visdrone_urban",
            "capture_time_us": 0,
            "telemetry": {
                "gps_lat": 0.0,
                "gps_lon": 0.0,
                "altitude_m": 1.0,
                "yaw_deg": 0.0,
            },
            "sufficient_evidence": True,
            "event": "no_registered_priority_visible",
            "objects": [],
            "relations": [],
            "evidence": [],
            "risk": 0.0,
            "uncertainty": 0.0,
            "recommended_action": "NO_ACTION",
        }

    def test_exact_object_is_accepted(self) -> None:
        packet = self.packet()
        self.assertEqual(frozenset(packet), MODEL_AUTHORED_PACKET_KEYS)
        self.assertEqual(extract_exact_json_object(json.dumps(packet)), packet)
        self.assertEqual(validate_semantic_content_packet(packet), packet)

    def test_surrounding_prose_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            extract_exact_json_object("result: " + json.dumps(self.packet()))

    def test_extra_field_is_rejected(self) -> None:
        packet = self.packet() | {"explanation": "unsupported"}
        with self.assertRaisesRegex(ValueError, "extra"):
            extract_exact_json_object(json.dumps(packet))

    def test_duplicate_keys_and_nonstandard_constants_are_rejected(self) -> None:
        packet_text = json.dumps(self.packet())
        duplicated_nested = packet_text.replace(
            '"gps_lat": 0.0',
            '"gps_lat": 0.0, "gps_lat": 1.0',
        )
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            extract_exact_json_object(duplicated_nested)
        nonstandard = packet_text.replace('"risk": 0.0', '"risk": NaN')
        with self.assertRaisesRegex(ValueError, "non-standard JSON constant"):
            extract_exact_json_object(nonstandard)

    def test_canonical_context_excludes_filesystem_path(self) -> None:
        request = InferenceRequest(
            observation_id="o1",
            image_path="/secret/test-dev/frame.jpg",
            mission="visdrone_urban",
            capture_time_us=4,
            telemetry={"mission_profile": "heavy_vehicle"},
            detector_proposals=({
                "label": "truck", "confidence": 0.9,
                "bbox_xyxy": [10, 20, 30, 40],
                "source_path": "/also/secret",
            },),
        )
        context = canonical_model_context(request, (100, 200), (768, 768))
        serialized = json.dumps(context, sort_keys=True)
        self.assertNotIn("secret", serialized)
        self.assertEqual(
            context["detector_proposals"][0]["bbox_normalized_xyxy"],
            [0.1, 0.1, 0.3, 0.2],
        )

    def test_visdrone_prompt_is_frozen_and_complete(self) -> None:
        self.assertEqual(
            VISDRONE_MISSION_PROMPT_PATH.read_text(encoding="utf-8").strip(),
            VISDRONE_MISSION_PROMPT,
        )
        self.assertEqual(prompt_for_mission("visdrone_urban"), VISDRONE_MISSION_PROMPT)
        self.assertEqual(
            VISDRONE_SOURCE_RULE_PROMPT_PATH.read_text(encoding="utf-8").strip(),
            VISDRONE_SOURCE_RULE_PROMPT,
        )
        self.assertNotEqual(VISDRONE_SOURCE_RULE_PROMPT_PATH, VISDRONE_MISSION_PROMPT_PATH)
        for token in (
            VISDRONE_PROPOSAL_DECISION_CONTRACT_VERSION,
            "exactly keys e, a, and l",
            "visibly supported",
            "no prose",
            "zero-based",
        ):
            self.assertIn(token, VISDRONE_MISSION_PROMPT)


class VisDroneProposalDecisionTests(unittest.TestCase):
    @staticmethod
    def decision(
        *,
        sufficient: bool = True,
        accepted: list[int] | None = None,
        limited: list[int] | None = None,
    ) -> dict[str, object]:
        return {
            "sufficient_evidence": sufficient,
            "accepted_proposal_ids": [] if accepted is None else accepted,
            "limited_visibility_proposal_ids": [] if limited is None else limited,
        }

    @staticmethod
    def wire(
        *,
        sufficient: bool = True,
        accepted: list[int] | None = None,
        limited: list[int] | None = None,
    ) -> dict[str, object]:
        return {
            "e": sufficient,
            "a": [] if accepted is None else accepted,
            "l": [] if limited is None else limited,
        }

    @staticmethod
    def telemetry(profile: str = "general_mobility") -> dict[str, object]:
        return {
            "gps_lat": 24.0,
            "gps_lon": 54.0,
            "altitude_m": 70.0,
            "yaw_deg": 0.0,
            "mission_profile": profile,
            "registered_roi_xyxy": [0.25, 0.25, 0.75, 0.85],
        }

    def request(
        self,
        proposals: tuple[dict[str, object], ...],
        *,
        profile: str = "general_mobility",
    ) -> InferenceRequest:
        return InferenceRequest(
            observation_id="visdrone:train:canary",
            image_path="/never/exposed/train-canary.jpg",
            mission="visdrone_urban",
            capture_time_us=123,
            telemetry=self.telemetry(profile),
            detector_proposals=proposals,
        )

    @staticmethod
    def proposal(
        label: str,
        box: list[float],
        *,
        confidence: float = 0.9,
        track_id: int | None = None,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "label": label,
            "bbox_normalized_xyxy": box,
            "confidence": confidence,
        }
        if track_id is not None:
            result["track_id"] = track_id
        return result

    def test_bounded_decision_accepts_only_the_exact_contract(self) -> None:
        accepted = self.decision(accepted=[0, 2], limited=[2])
        self.assertEqual(
            extract_exact_visdrone_proposal_decision(
                json.dumps(self.wire(accepted=[0, 2], limited=[2])),
                proposal_count=3,
            ),
            accepted,
        )
        self.assertEqual(
            extract_exact_visdrone_proposal_decision(
                json.dumps(self.wire()), proposal_count=0,
            ),
            self.decision(),
        )
        self.assertEqual(
            extract_exact_visdrone_proposal_decision(
                json.dumps(self.wire(sufficient=False)), proposal_count=0,
            ),
            self.decision(sufficient=False),
        )

    def test_bounded_decision_rejects_every_noncontract_shape(self) -> None:
        invalid = {
            "prose": "result: " + json.dumps(self.wire()),
            "fence": "```json\n" + json.dumps(self.wire()) + "\n```",
            "trailing": json.dumps(self.wire()) + " {}",
            "missing": '{"e":true,"a":[]}',
            "extra": json.dumps(self.wire() | {"event": "none"}),
            "duplicate_key": '{"e":true,"e":false,"a":[],"l":[]}',
            "nonstandard": '{"e":NaN,"a":[],"l":[]}',
            "numeric_boolean": json.dumps(self.wire() | {"e": 1}),
            "nonarray": json.dumps(self.wire() | {"a": 0}),
            "boolean_id": '{"e":true,"a":[true],"l":[]}',
            "negative": json.dumps(self.wire(accepted=[-1])),
            "out_of_range": json.dumps(self.wire(accepted=[3])),
            "unsorted": json.dumps(self.wire(accepted=[1, 0])),
            "duplicate_id": json.dumps(self.wire(accepted=[0, 0])),
            "limited_not_subset": json.dumps(self.wire(accepted=[0], limited=[1])),
            "false_nonempty": json.dumps(
                self.wire(sufficient=False, accepted=[0]),
            ),
        }
        for name, text in invalid.items():
            with self.subTest(name=name), self.assertRaises((TypeError, ValueError)):
                extract_exact_visdrone_proposal_decision(text, proposal_count=3)

    def test_bounded_decision_enforces_top32_id_boundary(self) -> None:
        decision = self.decision(accepted=[31])
        self.assertEqual(
            extract_exact_visdrone_proposal_decision(
                json.dumps(self.wire(accepted=[31])), proposal_count=32,
            ),
            decision,
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            extract_exact_visdrone_proposal_decision(
                json.dumps(self.wire(accepted=[32])), proposal_count=32,
            )
        with self.assertRaisesRegex(ValueError, "proposal_count"):
            extract_exact_visdrone_proposal_decision(
                json.dumps(self.wire()), proposal_count=33,
            )

    def test_compact_context_is_an_exact_outcome_blind_allowlist(self) -> None:
        proposal = self.proposal("truck", [0.1, 0.2, 0.3, 0.4], track_id=77)
        proposal.update({
            "source_path": "/secret/source.jpg",
            "gold_action": "RESPOND",
            "annotation_path": "/secret/labels.txt",
        })
        request = self.request((proposal,), profile="heavy_vehicle")
        context = visdrone_proposal_decision_context(
            request,
            (100, 200),
            (768, 768),
            preprocess={
                "source_size": [100, 200],
                "model_input_size": [768, 768],
                "content_xyxy": [192, 0, 576, 768],
            },
        )
        self.assertEqual(set(context), {"v", "q", "p"})
        self.assertEqual(
            context["p"],
            [[0, "truck", 1228, 819, 1638, 1639]],
        )
        serialized = json.dumps(context, sort_keys=True)
        for forbidden in (
            "secret", "gold_action", "annotation", "confidence", "track_id",
            "mission_profile", "registered_roi", "observation_id", "telemetry",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_context_rejects_conflicts_and_fractional_letterbox_geometry(self) -> None:
        conflicting_label = self.proposal("truck", [0.1, 0.1, 0.2, 0.2])
        conflicting_label["class"] = "bus"
        conflicting_box = self.proposal("truck", [0.1, 0.1, 0.2, 0.2])
        conflicting_box["bbox_xyxy"] = [10, 10, 30, 20]
        base_preprocess = {
            "source_size": [100, 100],
            "model_input_size": [768, 768],
            "content_xyxy": [0, 0, 768, 768],
        }
        for name, proposal in (
            ("label", conflicting_label),
            ("box", conflicting_box),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "conflicting"):
                visdrone_proposal_decision_context(
                    self.request((proposal,)),
                    (100, 100),
                    (768, 768),
                    preprocess=base_preprocess,
                )
        fractional = dict(base_preprocess) | {"content_xyxy": [0, 0.5, 768, 767]}
        with self.assertRaisesRegex(ValueError, "integers"):
            visdrone_proposal_decision_context(
                self.request((self.proposal("truck", [0.1, 0.1, 0.2, 0.2]),)),
                (100, 100),
                (768, 768),
                preprocess=fractional,
            )

    def test_module_policy_exactly_matches_the_registered_yaml(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "visdrone_paper.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["models"]["proposal_policy"]["top_k"], VISDRONE_MAX_PROPOSALS)
        self.assertEqual(
            config["models"]["proposal_policy"]["minimum_confidence"],
            VISDRONE_MIN_PROPOSAL_CONFIDENCE,
        )
        self.assertEqual(config["benchmark"]["occupancy_rule"], "center")
        self.assertEqual(
            config["benchmark"]["registered_roi_xyxy"],
            list(VISDRONE_REGISTERED_ROI_XYXY),
        )
        self.assertEqual(
            config["benchmark"]["min_image_plane_iou"],
            VISDRONE_IMAGE_PLANE_IOU,
        )
        observed_profiles = {
            name: frozenset(value["priority_categories"])
            for name, value in config["benchmark"]["mission_profiles"].items()
        }
        self.assertEqual(observed_profiles, VISDRONE_PRIORITY_LABELS_BY_PROFILE)

    def test_composer_event_action_precedence_and_authoritative_copy(self) -> None:
        proposals = (
            self.proposal("pedestrian", [0.25, 0.30, 0.35, 0.40], track_id=900),
            self.proposal("truck", [0.80, 0.80, 0.95, 0.95], track_id=0),
        )
        request = self.request(proposals)
        packet = compose_visdrone_semantic_packet(
            request,
            self.decision(accepted=[0, 1], limited=[1]),
            (100, 100),
        )
        self.assertEqual(packet["recommended_action"], "RESPOND")
        self.assertEqual(packet["event"], "visible_priority_roi_occupancy")
        self.assertEqual(packet["uncertainty"], 1.0)
        self.assertEqual(packet["observation_id"], request.observation_id)
        self.assertEqual(packet["mission"], request.mission)
        self.assertEqual(packet["capture_time_us"], request.capture_time_us)
        self.assertEqual(packet["telemetry"], request.telemetry)
        self.assertEqual(packet["objects"], list(proposals))
        self.assertEqual(
            [item["evidence_id"] for item in packet["evidence"]],
            ["proposal-0", "proposal-1"],
        )
        self.assertEqual(validate_semantic_content_packet(packet), packet)

        limited_only = compose_visdrone_semantic_packet(
            request,
            self.decision(accepted=[1], limited=[1]),
            (100, 100),
        )
        self.assertEqual(
            (limited_only["event"], limited_only["recommended_action"]),
            ("priority_target_low_visibility", "REVIEW"),
        )
        outside = compose_visdrone_semantic_packet(
            request,
            self.decision(accepted=[1]),
            (100, 100),
        )
        self.assertEqual(
            (outside["event"], outside["recommended_action"]),
            ("visible_priority_outside_roi", "REVIEW"),
        )

    def test_composer_overlap_no_priority_and_limited_nonpriority(self) -> None:
        proposals = (
            self.proposal("car", [0.05, 0.05, 0.45, 0.45]),
            self.proposal("van", [0.25, 0.05, 0.65, 0.45]),
            self.proposal("truck", [0.8, 0.8, 0.9, 0.9]),
        )
        vulnerable = self.request(proposals, profile="vulnerable_road_user")
        overlap = compose_visdrone_semantic_packet(
            vulnerable,
            self.decision(accepted=[0, 1], limited=[0]),
            (100, 100),
        )
        self.assertEqual(
            (overlap["event"], overlap["recommended_action"], overlap["uncertainty"]),
            ("visible_image_plane_overlap", "NO_ACTION", 0.0),
        )
        self.assertEqual(len(overlap["relations"]), 1)
        no_priority = compose_visdrone_semantic_packet(
            vulnerable,
            self.decision(accepted=[2]),
            (100, 100),
        )
        self.assertEqual(no_priority["event"], "no_registered_priority_visible")
        self.assertEqual(no_priority["recommended_action"], "NO_ACTION")

    def test_same_core_is_profile_conditioned_and_ids_are_not_track_ids(self) -> None:
        proposals = (
            self.proposal("truck", [0.30, 0.30, 0.40, 0.40], track_id=17),
            self.proposal("pedestrian", [0.80, 0.80, 0.90, 0.90], track_id=0),
        )
        core = self.decision(accepted=[0])
        heavy = compose_visdrone_semantic_packet(
            self.request(proposals, profile="heavy_vehicle"), core, (100, 100),
        )
        vulnerable = compose_visdrone_semantic_packet(
            self.request(proposals, profile="vulnerable_road_user"), core, (100, 100),
        )
        self.assertEqual(heavy["recommended_action"], "RESPOND")
        self.assertEqual(vulnerable["recommended_action"], "NO_ACTION")
        self.assertEqual(heavy["objects"][0]["track_id"], 17)
        self.assertEqual(heavy["evidence"][0]["evidence_id"], "proposal-0")

    def test_roi_boundary_and_iou_threshold_are_inclusive(self) -> None:
        boundary = self.request((
            self.proposal("truck", [0.20, 0.20, 0.30, 0.30]),
        ), profile="heavy_vehicle")
        packet = compose_visdrone_semantic_packet(
            boundary, self.decision(accepted=[0]), (100, 100),
        )
        self.assertEqual(packet["recommended_action"], "RESPOND")

        exact_overlap = self.request((
            self.proposal("car", [0.0, 0.0, 1.0, 1.0]),
            self.proposal("van", [0.0, 0.0, 0.05, 1.0]),
        ), profile="vulnerable_road_user")
        exact = compose_visdrone_semantic_packet(
            exact_overlap, self.decision(accepted=[0, 1]), (100, 100),
        )
        self.assertEqual(len(exact["relations"]), 1)
        self.assertEqual(exact["relations"][0]["image_plane_iou"], 0.05)

        below_overlap = self.request((
            self.proposal("car", [0.0, 0.0, 1.0, 1.0]),
            self.proposal("van", [0.0, 0.0, 0.049, 1.0]),
        ), profile="vulnerable_road_user")
        below = compose_visdrone_semantic_packet(
            below_overlap, self.decision(accepted=[0, 1]), (100, 100),
        )
        self.assertEqual(below["relations"], [])

    def test_relations_use_only_accepted_ids_and_deduplicate_label_pairs(self) -> None:
        request = self.request((
            self.proposal("car", [0.0, 0.0, 0.6, 0.6]),
            self.proposal("car", [0.1, 0.1, 0.7, 0.7]),
            self.proposal("car", [0.2, 0.2, 0.8, 0.8]),
        ), profile="vulnerable_road_user")
        all_accepted = compose_visdrone_semantic_packet(
            request, self.decision(accepted=[0, 1, 2]), (100, 100),
        )
        self.assertEqual(len(all_accepted["relations"]), 1)
        one_accepted = compose_visdrone_semantic_packet(
            request, self.decision(accepted=[1]), (100, 100),
        )
        self.assertEqual(one_accepted["relations"], [])
        self.assertEqual(one_accepted["evidence"][0]["evidence_id"], "proposal-1")

    def test_insufficient_evidence_is_a_deterministic_empty_abstention(self) -> None:
        request = self.request((self.proposal("truck", [0.3, 0.3, 0.4, 0.4]),))
        first = compose_visdrone_semantic_packet(
            request, self.decision(sufficient=False), (100, 100),
        )
        second = compose_visdrone_semantic_packet(
            request, self.decision(sufficient=False), (100, 100),
        )
        self.assertEqual(first, second)
        self.assertEqual(first["event"], "insufficient_evidence")
        self.assertEqual(first["recommended_action"], "REVIEW")
        self.assertEqual(first["objects"], [])
        self.assertEqual(first["relations"], [])
        self.assertEqual(first["evidence"], [])
        self.assertRegex(VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256, r"^[0-9a-f]{64}$")
        first["telemetry"]["registered_roi_xyxy"][0] = 0.0
        self.assertEqual(
            request.telemetry["registered_roi_xyxy"],
            [0.25, 0.25, 0.75, 0.85],
        )


if __name__ == "__main__":
    unittest.main()
