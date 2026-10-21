from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import sys
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.optimize import minimize_scalar

from .protocol import (
    MISSION_ONTOLOGY,
    VISDRONE_COMPOSED_PACKET_AUTHORSHIP,
    VISDRONE_PROPOSAL_DECISION_VERSION,
    validate_semantic_content_packet,
    validate_semantic_packet,
)
from .registry import (
    RunContext,
    StudyGateError,
    frame_stage,
    hash_file,
    hash_object,
    sha256_bytes,
)


MISSION_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "mission_packet_v1.txt"
if not MISSION_PROMPT_PATH.is_file():
    raise RuntimeError(f"Frozen mission prompt is missing: {MISSION_PROMPT_PATH}")
MISSION_PROMPT = MISSION_PROMPT_PATH.read_text(encoding="utf-8").strip()

VISDRONE_PROPOSAL_DECISION_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "visdrone_proposal_decision_v3.txt"
)
VISDRONE_MISSION_PROMPT_PATH = VISDRONE_PROPOSAL_DECISION_PROMPT_PATH
if not VISDRONE_MISSION_PROMPT_PATH.is_file():
    raise RuntimeError(
        f"Frozen VisDrone mission prompt is missing: {VISDRONE_MISSION_PROMPT_PATH}"
    )
VISDRONE_MISSION_PROMPT = VISDRONE_MISSION_PROMPT_PATH.read_text(
    encoding="utf-8"
).strip()
# Preserve the legacy full-packet source-rule artifact as a distinct identity.
VISDRONE_SOURCE_RULE_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "visdrone_source_rule_v1.txt"
)
if not VISDRONE_SOURCE_RULE_PROMPT_PATH.is_file():
    raise RuntimeError(
        f"Frozen VisDrone source-rule prompt is missing: {VISDRONE_SOURCE_RULE_PROMPT_PATH}"
    )
VISDRONE_SOURCE_RULE_PROMPT = VISDRONE_SOURCE_RULE_PROMPT_PATH.read_text(
    encoding="utf-8"
).strip()

VISDRONE_PROPOSAL_DECISION_CONTRACT_VERSION = (
    VISDRONE_PROPOSAL_DECISION_VERSION
)
QWEN3_VL_REGISTERED_MODEL_IDS = frozenset({
    "qwen3_vl_2b",
    "qwen3_vl_8b_upper_bound",
})
QWEN3_VL_JSON_ASSISTANT_PREFILL = "{"
SMOLVLM2_JSON_ASSISTANT_PREFILL = '{"'
VISDRONE_PROPOSAL_COMPOSER_VERSION = "semuav.visdrone-proposal-composer.v1"
VISDRONE_PROPOSAL_DECISION_KEYS = frozenset({
    "sufficient_evidence",
    "accepted_proposal_ids",
    "limited_visibility_proposal_ids",
})
VISDRONE_PROPOSAL_DECISION_WIRE_KEYS = frozenset({"e", "a", "l"})
VISDRONE_PROPOSAL_CONTEXT_SCHEMA = 2
VISDRONE_PROPOSAL_COORDINATE_MAX = 4095
VISDRONE_MAX_PROPOSALS = 32
VISDRONE_MIN_PROPOSAL_CONFIDENCE = 0.05
VISDRONE_IMAGE_PLANE_IOU = 0.05
VISDRONE_REGISTERED_ROI_XYXY = (0.25, 0.25, 0.75, 0.85)
DETECTOR_LABEL_ALIASES = {
    "person": "pedestrian",
    "motorcycle": "motor",
    "awning-tricycle": "awning_tricycle",
}
VISDRONE_PRIORITY_LABELS_BY_PROFILE: Mapping[str, frozenset[str]] = {
    "vulnerable_road_user": frozenset({
        "pedestrian", "people", "bicycle", "motor", "tricycle",
        "awning_tricycle",
    }),
    "heavy_vehicle": frozenset({"truck", "bus"}),
    "general_mobility": frozenset(MISSION_ONTOLOGY["visdrone_urban"]["objects"]),
}
VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256 = hash_object({
    "version": VISDRONE_PROPOSAL_COMPOSER_VERSION,
    "decision_contract": VISDRONE_PROPOSAL_DECISION_CONTRACT_VERSION,
    "proposal_id_definition": "zero_based_authenticated_proposal_position",
    "maximum_proposal_count": VISDRONE_MAX_PROPOSALS,
    "authenticated_proposal_ordering": "confidence_descending_label_geometry_track",
    "occupancy_rule": "box_center_inside_registered_roi",
    "registered_roi_xyxy": list(VISDRONE_REGISTERED_ROI_XYXY),
    "minimum_proposal_confidence": VISDRONE_MIN_PROPOSAL_CONFIDENCE,
    "min_image_plane_iou": VISDRONE_IMAGE_PLANE_IOU,
    "priority_profiles": {
        name: sorted(labels)
        for name, labels in sorted(VISDRONE_PRIORITY_LABELS_BY_PROFILE.items())
    },
    "event_precedence": [
        "visible_priority_roi_occupancy",
        "priority_target_low_visibility",
        "visible_priority_outside_roi",
        "visible_image_plane_overlap",
        "no_registered_priority_visible",
    ],
    "authoritative_fields": "copied_from_inference_request",
    "object_policy": {
        "source": "accepted_authenticated_detector_proposals",
        "fields": ["label", "bbox_normalized_xyxy", "confidence", "track_id_if_present"],
        "label_aliases": dict(sorted(DETECTOR_LABEL_ALIASES.items())),
    },
    "relation_policy": {
        "predicate": "image_plane_overlap",
        "value": "normalized_box_iou",
        "deduplication": "unique_sorted_label_pair",
    },
    "evidence_policy": {
        "records": "one_proposal_box_per_accepted_id",
        "evidence_id": "proposal-{proposal_id}",
        "linked_claim": "object:{label}",
    },
    "risk_by_action": {"RESPOND": 1.0, "REVIEW": 0.5, "NO_ACTION": 0.0},
    "uncertainty_policy": "one_iff_any_limited_priority_proposal_else_zero",
    "insufficient_evidence_policy": {
        "claims": "empty",
        "action": "REVIEW",
        "event": "insufficient_evidence",
        "risk": 0.5,
        "uncertainty": 1.0,
    },
})

VISDRONE_GENERATION_POLICY_VERSION = "semuav.visdrone-generation-policy.v5"
VISDRONE_REGISTERED_MAX_NEW_TOKENS = 256
VISDRONE_GENERATION_STRATEGY_BY_MODEL: Mapping[str, str] = {
    "qwen3_vl_2b": "qwen_open_brace_assistant_continuation",
    "qwen3_vl_8b_upper_bound": "qwen_open_brace_assistant_continuation",
    "smolvlm2_2b": "smol_native_double_quote_assistant_prefix",
}
VISDRONE_ASSISTANT_PREFILL_BY_STRATEGY: Mapping[str, str] = {
    "qwen_open_brace_assistant_continuation": QWEN3_VL_JSON_ASSISTANT_PREFILL,
    "smol_native_double_quote_assistant_prefix": SMOLVLM2_JSON_ASSISTANT_PREFILL,
}


def visdrone_assistant_prefill_for_model(model_id: str) -> str:
    """Return the registered host transport scaffold for one VisDrone model."""

    strategy = VISDRONE_GENERATION_STRATEGY_BY_MODEL.get(str(model_id))
    if strategy not in VISDRONE_ASSISTANT_PREFILL_BY_STRATEGY:
        raise StudyGateError(
            f"No registered VisDrone generation strategy exists for {model_id!r}"
        )
    return VISDRONE_ASSISTANT_PREFILL_BY_STRATEGY[strategy]


def visdrone_generation_policy(max_new_tokens: int) -> dict[str, Any]:
    """Return the complete deterministic VisDrone VLM generation policy."""

    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise ValueError("max_new_tokens must be a positive integer")
    return {
        "version": VISDRONE_GENERATION_POLICY_VERSION,
        "decision_contract": VISDRONE_PROPOSAL_DECISION_CONTRACT_VERSION,
        "strategy_by_model": dict(VISDRONE_GENERATION_STRATEGY_BY_MODEL),
        "qwen_open_brace_assistant_continuation": {
            "assistant_prefill": QWEN3_VL_JSON_ASSISTANT_PREFILL,
            "assistant_prefill_authorship": "host_transport_scaffold",
            "assistant_content_shape": "typed_text_block_list",
            "add_generation_prompt": False,
            "continue_final_message": True,
        },
        "smol_native_double_quote_assistant_prefix": {
            "assistant_prefill": SMOLVLM2_JSON_ASSISTANT_PREFILL,
            "assistant_prefill_authorship": "host_transport_scaffold",
            "assistant_content_shape": "native_user_only_render_plus_text_prefix",
            "add_generation_prompt": True,
            "continue_final_message": False,
            "chat_template_tokenize": False,
            "processor_retokenize_with_image": True,
        },
        "proposal_context": {
            "schema": VISDRONE_PROPOSAL_CONTEXT_SCHEMA,
            "coordinate_plane": "processed_model_input",
            "coordinate_max": VISDRONE_PROPOSAL_COORDINATE_MAX,
            "lower_edge_quantization": "floor",
            "upper_edge_quantization": "ceil",
            "row": ["proposal_id", "label", "x1", "y1", "x2", "y2"],
        },
        "decision_wire": {
            "keys": {
                "e": "sufficient_evidence",
                "a": "accepted_proposal_ids",
                "l": "limited_visibility_proposal_ids",
            },
            "host_normalization": (
                "expand_short_keys_before_validation_and_composition"
            ),
        },
        "generation": {
            "do_sample": False,
            "max_new_tokens": int(max_new_tokens),
            "retry_count": 0,
        },
        "decode": {
            "generated_continuation_only": True,
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        },
        "transport_validation": {
            "strict_exact_json_object": True,
            "strip_markdown_fences": False,
            "extract_substring": False,
            "repair_response": False,
        },
    }


def visdrone_generation_policy_sha256(max_new_tokens: int) -> str:
    """Hash the exact generation policy used for a VisDrone inference call."""

    return hash_object(visdrone_generation_policy(max_new_tokens))


VISDRONE_GENERATION_POLICY = visdrone_generation_policy(
    VISDRONE_REGISTERED_MAX_NEW_TOKENS
)
VISDRONE_GENERATION_POLICY_SHA256 = hash_object(VISDRONE_GENERATION_POLICY)


def prompt_for_mission(mission: str) -> str:
    """Return the only frozen prompt registered for a mission ontology."""

    normalized = str(mission).strip()
    if normalized == "visdrone_urban":
        return VISDRONE_MISSION_PROMPT
    if normalized in {"urban", "wildfire"}:
        return MISSION_PROMPT
    raise StudyGateError(f"No frozen prompt is registered for mission {mission!r}")


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    model_id: str
    repo: str
    revision: str
    role: str = "compact"
    quantization: str = "none"
    trust_remote_code: bool = False
    image_size: tuple[int, int] = (768, 768)

    def validate_full(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise StudyGateError(f"{self.model_id}: revision must be a resolved 40-character commit hash")
        if self.role not in {"compact", "upper_bound"}:
            raise ValueError("Model role must be compact or upper_bound")
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("image_size must contain two positive pixels")


OFFICIAL_MODEL_REPOS = {
    "qwen3_vl_2b": "Qwen/Qwen3-VL-2B-Instruct",
    "smolvlm2_2b": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
}

DETECTOR_RULE_VERSION = "semuav.detector-geometry-rules.v2"
DETECTOR_PROPOSAL_ORDERING = "confidence_descending_label_geometry_track"
def detector_inference_spec(
    *,
    image_size: int = 1280,
    confidence: float = 0.001,
    iou: float = 0.70,
    max_detections: int = 300,
    seed: int = 20260823,
    top_k: int = 32,
    minimum_confidence: float = 0.05,
    ordering: str = DETECTOR_PROPOSAL_ORDERING,
) -> dict[str, Any]:
    """Validate and canonicalize every output-affecting detector parameter."""

    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("Detector image_size must be a positive integer")
    if isinstance(max_detections, bool) or not isinstance(max_detections, int) or max_detections <= 0:
        raise ValueError("Detector max_detections must be a positive integer")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("Detector top_k must be a positive integer")
    if top_k > max_detections:
        raise ValueError("Detector top_k cannot exceed max_detections")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Detector seed must be a nonnegative integer")
    try:
        confidence_value = float(confidence)
        iou_value = float(iou)
        minimum_value = float(minimum_confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError("Detector thresholds must be numeric") from exc
    for name, value in (
        ("confidence", confidence_value),
        ("iou", iou_value),
        ("minimum_confidence", minimum_value),
    ):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Detector {name} must lie within [0, 1]")
    if minimum_value < confidence_value:
        raise ValueError(
            "Detector minimum_confidence cannot be below the model confidence threshold"
        )
    ordering_value = str(ordering).strip()
    if ordering_value != DETECTOR_PROPOSAL_ORDERING:
        raise ValueError(
            f"Detector ordering must be {DETECTOR_PROPOSAL_ORDERING!r}"
        )
    return {
        "image_size": image_size,
        "confidence": confidence_value,
        "iou": iou_value,
        "max_detections": max_detections,
        "seed": seed,
        "top_k": top_k,
        "minimum_confidence": minimum_value,
        "ordering": ordering_value,
        "object_ontology": sorted(MISSION_ONTOLOGY["visdrone_urban"]["objects"]),
        "label_aliases": dict(sorted(DETECTOR_LABEL_ALIASES.items())),
    }


def _model_local_files_kwargs() -> dict[str, bool]:
    """Translate the frozen-run offline flag into Hugging Face loader kwargs."""

    value = os.environ.get("SEMUAV_MODELS_LOCAL_ONLY", "").strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return {}
    if value in {"1", "true", "yes", "on"}:
        return {"local_files_only": True}
    raise StudyGateError(
        "SEMUAV_MODELS_LOCAL_ONLY must be one of 1/true/yes/on or 0/false/no/off"
    )


def _moondream_load_kwargs() -> dict[str, Any]:
    """Bind trusted-repository loading to the verified immutable cache root."""

    kwargs: dict[str, Any] = dict(_model_local_files_kwargs())
    if not kwargs.get("local_files_only"):
        return kwargs

    cache_value = os.environ.get("HF_HUB_CACHE", "").strip()
    if not cache_value:
        raise StudyGateError(
            "HF_HUB_CACHE must bind local-only Moondream loading to MODEL_LOCK"
        )
    cache = Path(cache_value).expanduser()
    if not cache.is_absolute():
        raise StudyGateError("HF_HUB_CACHE must be an absolute path")
    cache = cache.resolve()
    if not cache.is_dir():
        raise StudyGateError(f"Frozen Hugging Face cache is missing: {cache}")

    for variable in ("HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        alias_value = os.environ.get(variable, "").strip()
        if not alias_value:
            raise StudyGateError(
                f"{variable} must equal HF_HUB_CACHE for local-only Moondream loading"
            )
        alias = Path(alias_value).expanduser()
        if not alias.is_absolute() or alias.resolve() != cache:
            raise StudyGateError(
                f"{variable} must equal HF_HUB_CACHE for local-only Moondream loading"
            )

    kwargs["cache_dir"] = str(cache)
    return kwargs


def detector_execution_revision(
    checkpoint_sha256: str,
    tracker: str,
    *,
    image_size: int = 1280,
    confidence: float = 0.001,
    iou: float = 0.70,
    max_detections: int = 300,
    seed: int = 20260823,
    top_k: int = 32,
    minimum_confidence: float = 0.05,
    ordering: str = DETECTOR_PROPOSAL_ORDERING,
) -> str:
    """Bind detector provenance to weights and the complete frozen policy."""
    if not re.fullmatch(r"[0-9a-f]{64}", str(checkpoint_sha256)):
        raise ValueError("Detector checkpoint SHA-256 must be 64 lowercase hex characters")
    if not str(tracker).strip():
        raise ValueError("Detector tracker identity must be nonempty")
    inference_spec = detector_inference_spec(
        image_size=image_size,
        confidence=confidence,
        iou=iou,
        max_detections=max_detections,
        seed=seed,
        top_k=top_k,
        minimum_confidence=minimum_confidence,
        ordering=ordering,
    )
    return hash_object({
        "checkpoint_sha256": str(checkpoint_sha256),
        "tracker": str(tracker),
        "rule_version": DETECTOR_RULE_VERSION,
        "inference_spec": inference_spec,
    })[:40]


def resolve_model_revision(repo: str, token: str | None = None) -> str:
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError as exc:
        raise StudyGateError("Install huggingface_hub before resolving model revisions") from exc
    return str(HfApi(token=token).model_info(repo).sha)


def extract_first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Model response contains no JSON object")


MODEL_AUTHORED_PACKET_KEYS = frozenset({
    "schema_version", "observation_id", "mission", "capture_time_us",
    "telemetry", "sufficient_evidence", "event", "objects", "relations",
    "evidence", "risk", "uncertainty", "recommended_action",
})


def _strict_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object while rejecting duplicate member names."""

    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"Model response contains duplicate JSON key {key!r}")
        output[key] = value
    return output


def _reject_nonstandard_json_constant(value: str) -> Any:
    raise ValueError(f"Model response contains non-standard JSON constant {value!r}")


def extract_exact_json_object(
    text: str,
    *,
    expected_keys: frozenset[str] = MODEL_AUTHORED_PACKET_KEYS,
) -> dict[str, Any]:
    """Parse exactly one JSON object with no prose or undeclared fields."""

    if not isinstance(text, str):
        raise TypeError("Model response must be text")
    stripped = text.strip()
    decoder = json.JSONDecoder(
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_nonstandard_json_constant,
    )
    try:
        value, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError("Model response is not one exact JSON object") from exc
    if end != len(stripped) or not isinstance(value, dict):
        raise ValueError("Model response must contain exactly one JSON object and no prose")
    observed = frozenset(map(str, value))
    if observed != expected_keys:
        missing = sorted(expected_keys - observed)
        extra = sorted(observed - expected_keys)
        raise ValueError(
            f"Model response fields differ from the frozen contract; missing={missing}, extra={extra}"
        )
    return value


def validate_visdrone_proposal_decision(
    value: Mapping[str, Any],
    *,
    proposal_count: int,
) -> dict[str, Any]:
    """Validate the bounded model-authored VisDrone proposal decision."""

    if isinstance(proposal_count, bool) or not isinstance(proposal_count, int):
        raise TypeError("proposal_count must be an integer")
    if not 0 <= proposal_count <= VISDRONE_MAX_PROPOSALS:
        raise ValueError(
            f"proposal_count must lie within [0, {VISDRONE_MAX_PROPOSALS}]"
        )
    observed = frozenset(map(str, value))
    if observed != VISDRONE_PROPOSAL_DECISION_KEYS:
        missing = sorted(VISDRONE_PROPOSAL_DECISION_KEYS - observed)
        extra = sorted(observed - VISDRONE_PROPOSAL_DECISION_KEYS)
        raise ValueError(
            "Proposal decision fields differ from the frozen contract; "
            f"missing={missing}, extra={extra}"
        )
    sufficient_evidence = value["sufficient_evidence"]
    if type(sufficient_evidence) is not bool:
        raise ValueError("sufficient_evidence must be a JSON boolean")

    def validated_ids(name: str) -> list[int]:
        raw = value[name]
        if not isinstance(raw, list):
            raise ValueError(f"{name} must be a JSON array")
        if any(type(item) is not int for item in raw):
            raise ValueError(f"{name} must contain only JSON integers")
        if any(item < 0 or item >= proposal_count for item in raw):
            raise ValueError(
                f"{name} contains an ID outside [0, {proposal_count})"
            )
        if any(right <= left for left, right in zip(raw, raw[1:])):
            raise ValueError(f"{name} must be strictly increasing with no duplicates")
        return list(raw)

    accepted = validated_ids("accepted_proposal_ids")
    limited = validated_ids("limited_visibility_proposal_ids")
    if not set(limited).issubset(accepted):
        raise ValueError(
            "limited_visibility_proposal_ids must be a subset of accepted_proposal_ids"
        )
    if not sufficient_evidence and (accepted or limited):
        raise ValueError(
            "Insufficient evidence requires both proposal ID arrays to be empty"
        )
    return {
        "sufficient_evidence": sufficient_evidence,
        "accepted_proposal_ids": accepted,
        "limited_visibility_proposal_ids": limited,
    }


def extract_exact_visdrone_proposal_decision(
    text: str,
    *,
    proposal_count: int,
) -> dict[str, Any]:
    """Parse the compact wire object and return the canonical host decision."""

    wire = extract_exact_json_object(
        text,
        expected_keys=VISDRONE_PROPOSAL_DECISION_WIRE_KEYS,
    )
    return validate_visdrone_proposal_decision(
        {
            "sufficient_evidence": wire["e"],
            "accepted_proposal_ids": wire["a"],
            "limited_visibility_proposal_ids": wire["l"],
        },
        proposal_count=proposal_count,
    )


def aspect_preserving_letterbox(
    image: Image.Image,
    target_size: tuple[int, int],
) -> tuple[Image.Image, dict[str, Any]]:
    """Resize without geometric distortion and record the visible pixel grid."""

    width, height = (int(target_size[0]), int(target_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError("target_size must contain two positive pixels")
    contained = ImageOps.contain(image.convert("RGB"), (width, height), Image.Resampling.LANCZOS)
    left = (width - contained.width) // 2
    top = (height - contained.height) // 2
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    canvas.paste(contained, (left, top))
    return canvas, {
        "policy": "aspect_preserving_letterbox_v1",
        "source_size": [int(image.width), int(image.height)],
        "content_size": [int(contained.width), int(contained.height)],
        "content_xyxy": [left, top, left + contained.width, top + contained.height],
        "model_input_size": [width, height],
    }


def canonical_model_context(
    request: "InferenceRequest",
    source_size: tuple[int, int],
    model_input_size: tuple[int, int],
    *,
    preprocess: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build identical, path-free model context for every VLM family."""

    source_width, source_height = map(int, source_size)
    normalized_proposals: list[dict[str, Any]] = []
    for proposal in request.detector_proposals:
        normalized = {
            str(key): value
            for key, value in dict(proposal).items()
            if str(key) not in {"image_path", "media_path", "source_path"}
        }
        box = proposal.get("bbox_xyxy")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            normalized["bbox_normalized_xyxy"] = [
                float(box[0]) / source_width,
                float(box[1]) / source_height,
                float(box[2]) / source_width,
                float(box[3]) / source_height,
            ]
            normalized.pop("bbox_xyxy", None)
        normalized_proposals.append(normalized)
    return {
        "observation_id": request.observation_id,
        "mission": request.mission,
        "capture_time_us": int(request.capture_time_us),
        "telemetry": dict(request.telemetry),
        "source_image_size": [source_width, source_height],
        "model_input_size": [int(model_input_size[0]), int(model_input_size[1])],
        "preprocess": dict(preprocess or {}),
        "detector_proposals": normalized_proposals,
    }


def _canonical_visdrone_proposals(
    request: "InferenceRequest",
    source_size: tuple[int, int],
) -> list[dict[str, Any]]:
    """Validate proposals without changing their authenticated positional IDs."""

    if request.mission != "visdrone_urban":
        raise ValueError("The proposal-decision contract is only defined for visdrone_urban")
    source_width, source_height = map(int, source_size)
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source_size must contain two positive pixels")
    if len(request.detector_proposals) > VISDRONE_MAX_PROPOSALS:
        raise ValueError(
            f"VisDrone proposal count exceeds the frozen maximum {VISDRONE_MAX_PROPOSALS}"
        )
    allowed_labels = MISSION_ONTOLOGY["visdrone_urban"]["objects"]
    canonical: list[dict[str, Any]] = []
    for proposal_id, raw_proposal in enumerate(request.detector_proposals):
        if not isinstance(raw_proposal, Mapping):
            raise ValueError(f"Detector proposal {proposal_id} must be an object")
        raw_labels = [
            raw_proposal[key]
            for key in ("label", "class", "name")
            if isinstance(raw_proposal.get(key), str)
            and str(raw_proposal[key]).strip()
        ]
        if not raw_labels:
            raise ValueError(f"Detector proposal {proposal_id} lacks a label")
        normalized_labels = {
            DETECTOR_LABEL_ALIASES.get(
                str(raw_label).strip().lower().replace(" ", "_"),
                str(raw_label).strip().lower().replace(" ", "_"),
            )
            for raw_label in raw_labels
        }
        if len(normalized_labels) != 1:
            raise ValueError(
                f"Detector proposal {proposal_id} has conflicting label fields"
            )
        label = normalized_labels.pop()
        if label not in allowed_labels:
            raise ValueError(
                f"Detector proposal {proposal_id} has unsupported VisDrone label {label!r}"
            )

        confidence_raw = raw_proposal.get("confidence")
        if isinstance(confidence_raw, bool):
            raise ValueError(f"Detector proposal {proposal_id} has invalid confidence")
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Detector proposal {proposal_id} lacks numeric confidence"
            ) from exc
        if (
            not math.isfinite(confidence)
            or not VISDRONE_MIN_PROPOSAL_CONFIDENCE <= confidence <= 1
        ):
            raise ValueError(
                f"Detector proposal {proposal_id} confidence must lie within "
                f"[{VISDRONE_MIN_PROPOSAL_CONFIDENCE}, 1]"
            )

        normalized_box = raw_proposal.get("bbox_normalized_xyxy")
        pixel_box = raw_proposal.get("bbox_xyxy")
        box = normalized_box
        converted_pixel_box = None
        if pixel_box is not None and (
            not isinstance(pixel_box, (list, tuple)) or len(pixel_box) != 4
        ):
            raise ValueError(
                f"Detector proposal {proposal_id} pixel box must contain four coordinates"
            )
        if isinstance(pixel_box, (list, tuple)) and any(
            isinstance(item, bool) for item in pixel_box
        ):
            raise ValueError(f"Detector proposal {proposal_id} pixel box must be numeric")
        if isinstance(pixel_box, (list, tuple)) and len(pixel_box) == 4:
            try:
                converted_pixel_box = [
                    float(pixel_box[0]) / source_width,
                    float(pixel_box[1]) / source_height,
                    float(pixel_box[2]) / source_width,
                    float(pixel_box[3]) / source_height,
                ]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Detector proposal {proposal_id} pixel box must be numeric"
                ) from exc
        if box is None:
            box = converted_pixel_box
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(
                f"Detector proposal {proposal_id} requires one four-value bounding box"
            )
        if any(isinstance(item, bool) for item in box):
            raise ValueError(f"Detector proposal {proposal_id} box must be numeric")
        try:
            coordinates = [float(item) for item in box]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Detector proposal {proposal_id} box must be numeric"
            ) from exc
        if (
            not all(math.isfinite(item) for item in coordinates)
            or not 0 <= coordinates[0] < coordinates[2] <= 1
            or not 0 <= coordinates[1] < coordinates[3] <= 1
        ):
            raise ValueError(
                f"Detector proposal {proposal_id} has invalid normalized geometry"
            )
        if converted_pixel_box is not None and any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
            for left, right in zip(coordinates, converted_pixel_box)
        ):
            raise ValueError(
                f"Detector proposal {proposal_id} has conflicting normalized and pixel boxes"
            )
        item: dict[str, Any] = {
            "label": label,
            "bbox_normalized_xyxy": coordinates,
            "confidence": confidence,
        }
        if "track_id" in raw_proposal and raw_proposal["track_id"] is not None:
            track_id = raw_proposal["track_id"]
            if isinstance(track_id, bool) or not isinstance(track_id, (str, int)):
                raise ValueError(
                    f"Detector proposal {proposal_id} has invalid track_id"
                )
            if isinstance(track_id, str) and not track_id.strip():
                raise ValueError(
                    f"Detector proposal {proposal_id} has an empty track_id"
                )
            item["track_id"] = track_id
        canonical.append(item)
    return canonical


def visdrone_proposal_decision_context(
    request: "InferenceRequest",
    source_size: tuple[int, int],
    model_input_size: tuple[int, int],
    *,
    preprocess: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the bounded, outcome-blind context shown to every VisDrone VLM."""

    proposals = _canonical_visdrone_proposals(request, source_size)
    source_width, source_height = map(int, source_size)
    model_width, model_height = map(int, model_input_size)
    if model_width <= 0 or model_height <= 0:
        raise ValueError("model_input_size must contain two positive pixels")
    content_box = preprocess.get("content_xyxy")
    if not isinstance(content_box, (list, tuple)) or len(content_box) != 4:
        raise ValueError("preprocess.content_xyxy must contain four coordinates")
    if any(isinstance(item, bool) for item in content_box):
        raise ValueError("preprocess.content_xyxy must be numeric")
    try:
        content_values = [float(item) for item in content_box]
    except (TypeError, ValueError) as exc:
        raise ValueError("preprocess.content_xyxy must be numeric") from exc
    if not all(math.isfinite(item) and item.is_integer() for item in content_values):
        raise ValueError("preprocess.content_xyxy must contain finite integers")
    content_xyxy = [int(item) for item in content_values]
    if (
        not 0 <= content_xyxy[0] < content_xyxy[2] <= model_width
        or not 0 <= content_xyxy[1] < content_xyxy[3] <= model_height
    ):
        raise ValueError("preprocess.content_xyxy is outside the model input")
    for name, expected in (
        ("source_size", [source_width, source_height]),
        ("model_input_size", [model_width, model_height]),
    ):
        observed = preprocess.get(name)
        if observed is not None and (
            not isinstance(observed, (list, tuple)) or list(observed) != expected
        ):
            raise ValueError(f"preprocess.{name} differs from the measured geometry")
    content_left, content_top, content_right, content_bottom = content_xyxy
    content_width = content_right - content_left
    content_height = content_bottom - content_top
    coordinate_max = VISDRONE_PROPOSAL_COORDINATE_MAX

    def compact_box(box: Sequence[float]) -> list[int]:
        model_plane = (
            (content_left + box[0] * content_width) / model_width,
            (content_top + box[1] * content_height) / model_height,
            (content_left + box[2] * content_width) / model_width,
            (content_top + box[3] * content_height) / model_height,
        )
        quantized = [
            max(0, min(coordinate_max, math.floor(model_plane[0] * coordinate_max))),
            max(0, min(coordinate_max, math.floor(model_plane[1] * coordinate_max))),
            max(0, min(coordinate_max, math.ceil(model_plane[2] * coordinate_max))),
            max(0, min(coordinate_max, math.ceil(model_plane[3] * coordinate_max))),
        ]
        if not 0 <= quantized[0] < quantized[2] <= coordinate_max:
            raise ValueError("Compact proposal x coordinates are not strictly ordered")
        if not 0 <= quantized[1] < quantized[3] <= coordinate_max:
            raise ValueError("Compact proposal y coordinates are not strictly ordered")
        return quantized

    return {
        "v": VISDRONE_PROPOSAL_CONTEXT_SCHEMA,
        "q": coordinate_max,
        "p": [
            [proposal_id, item["label"], *compact_box(item["bbox_normalized_xyxy"])]
            for proposal_id, item in enumerate(proposals)
        ],
    }


def _normalized_box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def compose_visdrone_semantic_packet(
    request: "InferenceRequest",
    decision: Mapping[str, Any],
    source_size: tuple[int, int],
) -> dict[str, Any]:
    """Compose the full packet from a bounded decision and immutable proposals."""

    proposals = _canonical_visdrone_proposals(request, source_size)
    core = validate_visdrone_proposal_decision(
        decision,
        proposal_count=len(proposals),
    )
    telemetry = dict(request.telemetry)
    profile = telemetry.get("mission_profile")
    if not isinstance(profile, str) or profile not in VISDRONE_PRIORITY_LABELS_BY_PROFILE:
        raise ValueError(f"Unregistered VisDrone mission_profile {profile!r}")
    roi = telemetry.get("registered_roi_xyxy")
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        raise ValueError("telemetry.registered_roi_xyxy must contain four coordinates")
    if any(isinstance(item, bool) for item in roi):
        raise ValueError("telemetry.registered_roi_xyxy must be numeric")
    try:
        roi_xyxy = [float(item) for item in roi]
    except (TypeError, ValueError) as exc:
        raise ValueError("telemetry.registered_roi_xyxy must be numeric") from exc
    if (
        not all(math.isfinite(item) for item in roi_xyxy)
        or not 0 <= roi_xyxy[0] < roi_xyxy[2] <= 1
        or not 0 <= roi_xyxy[1] < roi_xyxy[3] <= 1
    ):
        raise ValueError("telemetry.registered_roi_xyxy must be a normalized rectangle")
    if tuple(roi_xyxy) != VISDRONE_REGISTERED_ROI_XYXY:
        raise ValueError("telemetry.registered_roi_xyxy differs from the frozen VisDrone ROI")

    if not core["sufficient_evidence"]:
        packet = {
            "schema_version": "semuav.v1",
            "observation_id": request.observation_id,
            "mission": request.mission,
            "capture_time_us": request.capture_time_us,
            "telemetry": telemetry,
            "sufficient_evidence": False,
            "event": "insufficient_evidence",
            "objects": [],
            "relations": [],
            "evidence": [],
            "risk": 0.5,
            "uncertainty": 1.0,
            "recommended_action": "REVIEW",
        }
        return validate_semantic_content_packet(packet)

    accepted_ids = core["accepted_proposal_ids"]
    limited_ids = set(core["limited_visibility_proposal_ids"])
    accepted = [(proposal_id, proposals[proposal_id]) for proposal_id in accepted_ids]
    objects = [dict(item) for _, item in accepted]

    relation_keys: set[tuple[str, str, str]] = set()
    relations: list[dict[str, Any]] = []
    for position, (_, left) in enumerate(accepted):
        for _, right in accepted[position + 1:]:
            overlap = _normalized_box_iou(
                left["bbox_normalized_xyxy"],
                right["bbox_normalized_xyxy"],
            )
            if overlap < VISDRONE_IMAGE_PLANE_IOU:
                continue
            subject, object_label = sorted((str(left["label"]), str(right["label"])))
            key = (subject, "image_plane_overlap", object_label)
            if key in relation_keys:
                continue
            relation_keys.add(key)
            relations.append({
                "subject": subject,
                "predicate": "image_plane_overlap",
                "object": object_label,
                "image_plane_iou": overlap,
            })

    priority_labels = VISDRONE_PRIORITY_LABELS_BY_PROFILE[profile]
    priority = [
        (proposal_id, item)
        for proposal_id, item in accepted
        if item["label"] in priority_labels
    ]
    inside_nonlimited = []
    for proposal_id, item in priority:
        if proposal_id in limited_ids:
            continue
        x1, y1, x2, y2 = item["bbox_normalized_xyxy"]
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if roi_xyxy[0] <= center_x <= roi_xyxy[2] and roi_xyxy[1] <= center_y <= roi_xyxy[3]:
            inside_nonlimited.append(proposal_id)
    limited_priority = [
        proposal_id for proposal_id, _ in priority if proposal_id in limited_ids
    ]
    if inside_nonlimited:
        event, action = "visible_priority_roi_occupancy", "RESPOND"
    elif limited_priority:
        event, action = "priority_target_low_visibility", "REVIEW"
    elif priority:
        event, action = "visible_priority_outside_roi", "REVIEW"
    elif relations:
        event, action = "visible_image_plane_overlap", "NO_ACTION"
    else:
        event, action = "no_registered_priority_visible", "NO_ACTION"

    evidence = [
        {
            "evidence_id": f"proposal-{proposal_id}",
            "linked_claims": [f"object:{item['label']}"],
            "bbox_normalized_xyxy": list(item["bbox_normalized_xyxy"]),
        }
        for proposal_id, item in accepted
    ]
    packet = {
        "schema_version": "semuav.v1",
        "observation_id": request.observation_id,
        "mission": request.mission,
        "capture_time_us": request.capture_time_us,
        "telemetry": telemetry,
        "sufficient_evidence": True,
        "event": event,
        "objects": objects,
        "relations": relations,
        "evidence": evidence,
        "risk": 1.0 if action == "RESPOND" else 0.5 if action == "REVIEW" else 0.0,
        "uncertainty": 1.0 if limited_priority else 0.0,
        "recommended_action": action,
    }
    return validate_semantic_content_packet(packet)


@dataclasses.dataclass(frozen=True)
class InferenceRequest:
    observation_id: str
    image_path: str
    mission: str
    capture_time_us: int
    telemetry: Mapping[str, float]
    detector_proposals: Sequence[Mapping[str, Any]] = ()


LEGACY_MOONDREAM_INFERENCE_FIELDS: tuple[str, ...] = (
    "moondream_image_positions",
    "moondream_native_template_tokens",
    "moondream_question_tokens",
    "moondream_prefill_tokens",
    "moondream_generation_reserve_tokens",
    "moondream_required_tokens",
    "moondream_effective_context_tokens",
    "moondream_context_margin_tokens",
    "moondream_tokenizer_lock_sha256",
    "moondream_tokenizer_artifact_sha256",
)


@dataclasses.dataclass
class InferenceRecord:
    observation_id: str
    model_id: str
    model_revision: str
    raw_text: str
    parsed: dict[str, Any] | None
    schema_valid: bool
    error: str | None
    input_tokens: int | None
    output_tokens: int | None
    confidence: float | None
    confidence_source: str
    latency_ms: float
    peak_memory_mb: float | None
    energy_j: float | None
    prompt_hash: str
    peak_temperature_c: float | None = None
    decision_contract_version: str | None = None
    decision_core_valid: bool | None = None
    decision_core_json: str | None = None
    decision_context_sha256: str | None = None
    decision_proposal_count: int | None = None
    decision_composer_policy_sha256: str | None = None
    terminated_by_eos: bool | None = None
    reached_token_cap: bool | None = None
    generation_policy_version: str | None = None
    generation_policy_sha256: str | None = None
    generation_strategy: str | None = None
    generation_max_new_tokens: int | None = None
    assistant_prefill_sha256: str | None = None
    assistant_prefill_characters: int | None = None
    generated_continuation_sha256: str | None = None
    moondream_image_positions: int | None = None
    moondream_native_template_tokens: int | None = None
    moondream_question_tokens: int | None = None
    moondream_prefill_tokens: int | None = None
    moondream_generation_reserve_tokens: int | None = None
    moondream_required_tokens: int | None = None
    moondream_effective_context_tokens: int | None = None
    moondream_context_margin_tokens: int | None = None
    moondream_tokenizer_lock_sha256: str | None = None
    moondream_tokenizer_artifact_sha256: str | None = None

    def require_active_evidence_boundary(self) -> None:
        """Reject historical Moondream provenance at the active v2r13 row boundary."""

        populated = [
            name
            for name in LEGACY_MOONDREAM_INFERENCE_FIELDS
            if getattr(self, name) is not None
        ]
        if populated:
            raise StudyGateError(
                "Legacy Moondream provenance is forbidden in active v2r13 evidence: "
                + ", ".join(populated)
            )

    def as_row(self) -> dict[str, Any]:
        self.require_active_evidence_boundary()
        row = dataclasses.asdict(self)
        for name in LEGACY_MOONDREAM_INFERENCE_FIELDS:
            row.pop(name)
        row["parsed_json"] = json.dumps(row.pop("parsed"), sort_keys=True) if self.parsed is not None else None
        return row


class PerceptionBackend(ABC):
    spec: ModelSpec

    @abstractmethod
    def infer(self, request: InferenceRequest, prompt: str, max_new_tokens: int) -> InferenceRecord:
        raise NotImplementedError


class DeterministicMockBackend(PerceptionBackend):
    """Integration fixture only. It is structurally blocked from the full profile."""

    def __init__(self, model_id: str = "mock_vlm"):
        self.spec = ModelSpec(model_id, "GENERATED_FIXTURE", "0" * 40)

    def infer(self, request: InferenceRequest, prompt: str, max_new_tokens: int = 384) -> InferenceRecord:
        digest = int(hash_object({"id": request.observation_id, "model": self.spec.model_id})[:8], 16)
        # A wall-clock measurement would make the synthetic acceptance fixture
        # nondeterministic.  Real backends retain measured latency; this mock
        # uses a stable, explicitly non-scientific latency value instead.
        fixture_latency_ms = 4.0 + (digest % 2_000) / 1_000
        high_risk = digest % 7 == 0
        event = (
            "visible_flame" if request.mission == "wildfire"
            else "visible_road_obstruction"
        ) if high_risk else "none_visible"
        packet = {
            "schema_version": "semuav.v1",
            "observation_id": request.observation_id,
            "mission": request.mission,
            "capture_time_us": request.capture_time_us,
            "telemetry": dict(request.telemetry),
            "sufficient_evidence": digest % 11 != 0,
            "event": event,
            "objects": list(request.detector_proposals)[:8],
            "relations": [],
            "risk": 0.88 if high_risk else 0.08,
            "uncertainty": 0.45 if digest % 11 == 0 else 0.12,
            "recommended_action": "RESPOND" if high_risk else "NO_ACTION",
        }
        if not packet["sufficient_evidence"]:
            packet["event"] = "insufficient_evidence"
            packet["recommended_action"] = "REVIEW"
        grounded_claim = packet["event"] not in {"none_visible", "insufficient_evidence"}
        packet["evidence"] = ([{
            "evidence_id": "fixture-event-0",
            "linked_claims": ["event"],
            "bbox_normalized_xyxy": [0.0, 0.0, 1.0, 1.0],
        }] if grounded_claim else [])
        raw = json.dumps(packet, separators=(",", ":"))
        return InferenceRecord(
            request.observation_id,
            self.spec.model_id,
            self.spec.revision,
            raw,
            validate_semantic_content_packet(packet),
            True,
            None,
            None,
            len(raw.split()),
            0.75,
            "deterministic_fixture_score",
            fixture_latency_ms,
            None,
            None,
            hash_object(prompt),
        )


class DetectorOnlyBackend(PerceptionBackend):
    """Frozen detector-action rules with explicit observable critical triggers."""

    def __init__(
        self,
        revision: str,
        respond_threshold: float = 0.85,
        review_threshold: float = 0.45,
        *,
        urban_policy: str = "legacy_count_obstruction",
        urban_priority_labels: Sequence[str] = (),
        urban_priority_profiles: Mapping[str, Sequence[str]] | None = None,
        urban_priority_roi_xyxy: Sequence[float] = (0.25, 0.25, 0.75, 0.75),
        urban_priority_confidence: float = 0.50,
        urban_image_plane_iou: float = 0.05,
    ):
        self.spec = ModelSpec("detector_only", "local/frozen-detector-rules", revision)
        self.respond_threshold = respond_threshold
        self.review_threshold = review_threshold
        allowed_policies = {"legacy_count_obstruction", "registered_roi_occupancy"}
        if urban_policy not in allowed_policies:
            raise ValueError(f"urban_policy must be one of {sorted(allowed_policies)}")
        priority_labels = tuple(sorted({str(label).strip() for label in urban_priority_labels if str(label).strip()}))
        if urban_policy == "registered_roi_occupancy" and not priority_labels:
            raise ValueError(
                "registered_roi_occupancy requires frozen urban_priority_labels"
            )
        profile_labels = {
            str(profile): tuple(sorted({str(label).strip() for label in labels if str(label).strip()}))
            for profile, labels in dict(urban_priority_profiles or {}).items()
        }
        all_registered_labels = set(priority_labels).union(
            *(set(labels) for labels in profile_labels.values())
        ) if profile_labels else set(priority_labels)
        unsupported_labels = all_registered_labels.difference(
            MISSION_ONTOLOGY["visdrone_urban"]["objects"]
        )
        if unsupported_labels:
            raise ValueError(
                f"urban_priority_labels contains unsupported labels: {sorted(unsupported_labels)}"
            )
        try:
            priority_roi = tuple(float(value) for value in urban_priority_roi_xyxy)
        except (TypeError, ValueError) as exc:
            raise ValueError("urban_priority_roi_xyxy must contain four numeric coordinates") from exc
        if (
            len(priority_roi) != 4
            or not all(np.isfinite(priority_roi))
            or not 0 <= priority_roi[0] < priority_roi[2] <= 1
            or not 0 <= priority_roi[1] < priority_roi[3] <= 1
        ):
            raise ValueError("urban_priority_roi_xyxy must be valid normalized xyxy geometry")
        if not 0 <= float(urban_priority_confidence) <= 1:
            raise ValueError("urban_priority_confidence must lie within [0, 1]")
        if not 0 < float(urban_image_plane_iou) <= 1:
            raise ValueError("urban_image_plane_iou must lie within (0, 1]")
        self.urban_policy = urban_policy
        self.urban_priority_labels = priority_labels
        self.urban_priority_profiles = profile_labels
        self.urban_priority_roi_xyxy = priority_roi
        self.urban_priority_confidence = float(urban_priority_confidence)
        self.urban_image_plane_iou = float(urban_image_plane_iou)
        self.rule_config_hash = hash_object({
            "urban_policy": self.urban_policy,
            "urban_priority_labels": self.urban_priority_labels,
            "urban_priority_profiles": self.urban_priority_profiles,
            "urban_priority_roi_xyxy": self.urban_priority_roi_xyxy,
            "urban_priority_confidence": self.urban_priority_confidence,
            "urban_image_plane_iou": self.urban_image_plane_iou,
            "respond_threshold": float(self.respond_threshold),
            "review_threshold": float(self.review_threshold),
        })

    def infer(self, request: InferenceRequest, prompt: str = MISSION_PROMPT, max_new_tokens: int = 384) -> InferenceRecord:
        start = time.perf_counter()
        aliases = {
            "urban": {
                "person": "pedestrian", "motorcycle": "motor",
                "awning-tricycle": "awning_tricycle",
            },
            "visdrone_urban": {
                "person": "pedestrian", "motorcycle": "motor",
                "awning-tricycle": "awning_tricycle",
            },
            "wildfire": {
                "smoke": "visible_smoke", "flame": "visible_flame", "fire": "visible_flame",
                "pedestrian": "person", "people": "person", "car": "vehicle",
                "van": "vehicle", "truck": "vehicle", "bus": "vehicle",
                "motor": "vehicle", "motorcycle": "vehicle", "bicycle": "vehicle",
            },
        }
        raw_boxes_present = any("bbox_xyxy" in proposal for proposal in request.detector_proposals)
        source_size: tuple[int, int] | None = None
        if raw_boxes_present:
            try:
                with Image.open(request.image_path) as source_image:
                    source_size = source_image.size
            except Exception as exc:
                raise StudyGateError(
                    "Pixel-coordinate detector proposals require a readable source image"
                ) from exc

        def canonical_box(proposal: Mapping[str, Any]) -> list[float] | None:
            box = proposal.get("bbox_normalized_xyxy")
            if box is None and "bbox_xyxy" in proposal:
                box = proposal.get("bbox_xyxy")
                if source_size is None:
                    raise StudyGateError("Detector proposal normalization lacks source image size")
                if isinstance(box, (list, tuple)) and len(box) == 4:
                    width, height = source_size
                    box = [
                        float(box[0]) / width, float(box[1]) / height,
                        float(box[2]) / width, float(box[3]) / height,
                    ]
            if box is None:
                return None
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise StudyGateError("Detector proposal box must contain four coordinates")
            values = [float(value) for value in box]
            if (
                not all(np.isfinite(values))
                or not 0 <= values[0] < values[2] <= 1
                or not 0 <= values[1] < values[3] <= 1
            ):
                raise StudyGateError("Detector proposal box is not valid normalized xyxy geometry")
            return values

        proposals: list[dict[str, Any]] = []
        allowed_objects = MISSION_ONTOLOGY[request.mission]["objects"]
        for proposal in request.detector_proposals:
            raw_label = next(
                (
                    proposal[key] for key in ("label", "class", "name")
                    if isinstance(proposal.get(key), str) and str(proposal[key]).strip()
                ),
                None,
            )
            if raw_label is None:
                continue
            normalized = str(raw_label).strip().lower().replace(" ", "_")
            normalized = aliases[request.mission].get(normalized, normalized)
            if normalized not in allowed_objects:
                continue
            canonical: dict[str, Any] = {
                "label": normalized,
                "confidence": float(np.clip(float(proposal.get("confidence", 0.0)), 0, 1)),
            }
            box = canonical_box(proposal)
            if box is not None:
                canonical["bbox_normalized_xyxy"] = box
            track_id = proposal.get("track_id")
            if isinstance(track_id, (str, int)) and not isinstance(track_id, bool):
                canonical["track_id"] = track_id
            proposals.append(canonical)

        def box_gap(first: Sequence[float], second: Sequence[float]) -> float:
            dx = max(float(first[0]) - float(second[2]), float(second[0]) - float(first[2]), 0.0)
            dy = max(float(first[1]) - float(second[3]), float(second[1]) - float(first[3]), 0.0)
            return float(math.hypot(dx, dy))

        def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
            ix1, iy1 = max(first[0], second[0]), max(first[1], second[1])
            ix2, iy2 = min(first[2], second[2]), min(first[3], second[3])
            intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            first_area = (first[2] - first[0]) * (first[3] - first[1])
            second_area = (second[2] - second[0]) * (second[3] - second[1])
            union = first_area + second_area - intersection
            return float(intersection / union) if union > 0 else 0.0

        # Strong geometry-aware detector comparator.  The 0.10 normalized
        # edge-gap threshold and registered VisDrone IoU threshold are frozen
        # before validation; no contextual VLM output enters these relations.
        relations: list[dict[str, str]] = []
        seen_relations: set[tuple[str, str, str]] = set()
        boxed = [item for item in proposals if "bbox_normalized_xyxy" in item]
        for left_index, left in enumerate(boxed):
            for right in boxed[left_index + 1:]:
                left_label, right_label = str(left["label"]), str(right["label"])
                left_box = left["bbox_normalized_xyxy"]
                right_box = right["bbox_normalized_xyxy"]
                predicate: str | None = None
                subject, object_label = left_label, right_label
                if request.mission in {"urban", "visdrone_urban"}:
                    # The registered source-derived VisDrone policy may use
                    # publisher-supported boxes and image-plane overlap, but it
                    # must not turn an image-plane distance into a physical
                    # proximity claim.  The legacy comparator remains available
                    # for reproduction of the original multi-dataset protocol.
                    overlap = box_iou(left_box, right_box)
                    if request.mission == "visdrone_urban":
                        predicate = (
                            "image_plane_overlap"
                            if overlap >= self.urban_image_plane_iou else None
                        )
                    elif overlap > 0:
                        predicate = "overlapping"
                    elif (
                        self.urban_policy == "legacy_count_obstruction"
                        and box_gap(left_box, right_box) <= 0.10
                    ):
                        predicate = "near"
                    if predicate is not None and subject > object_label:
                        subject, object_label = object_label, subject
                elif box_gap(left_box, right_box) <= 0.10 or box_iou(left_box, right_box) > 0:
                    labels = {left_label, right_label}
                    if labels == {"visible_smoke", "structure"}:
                        predicate, subject, object_label = "smoke_near_structure", "visible_smoke", "structure"
                    elif labels == {"visible_flame", "vegetation"}:
                        predicate, subject, object_label = "flame_near_vegetation", "visible_flame", "vegetation"
                    elif labels == {"person", "visible_flame"}:
                        predicate, subject, object_label = "person_near_visible_fire", "person", "visible_flame"
                if predicate is not None:
                    key = (subject, predicate, object_label)
                    if key not in seen_relations:
                        relations.append({
                            "subject": subject, "predicate": predicate,
                            "object": object_label,
                        })
                        seen_relations.add(key)
        confidences = [float(item.get("confidence", 0.0)) for item in proposals]
        object_count = len(proposals)
        high_confidence = sum(value >= 0.5 for value in confidences)
        evidence = object_count > 0 and (max(confidences, default=0.0) >= 0.35)
        labels = {str(item["label"]) for item in proposals}
        if request.mission == "wildfire" and labels.intersection({"visible_flame", "visible_smoke"}):
            event = "visible_flame" if "visible_flame" in labels else "visible_smoke"
            risk = float(np.clip(max(confidences, default=0.0), 0, 0.99))
            source_supported_action: str | None = None
        elif request.mission in {"urban", "visdrone_urban"} and self.urban_policy == "registered_roi_occupancy":
            x1, y1, x2, y2 = self.urban_priority_roi_xyxy
            profile = str(request.telemetry.get("mission_profile", "")).strip()
            active_priority_labels = self.urban_priority_profiles.get(
                profile, self.urban_priority_labels,
            )
            if self.urban_priority_profiles and profile not in self.urban_priority_profiles:
                raise StudyGateError(
                    f"Unregistered VisDrone mission_profile {profile!r}"
                )
            priority_candidates = [
                item for item in proposals
                if (
                    str(item["label"]) in active_priority_labels
                    and float(item["confidence"]) >= self.urban_priority_confidence
                )
            ]
            priority_roi_hits = [
                item for item in priority_candidates
                if "bbox_normalized_xyxy" in item
                and x1 <= (
                    float(item["bbox_normalized_xyxy"][0])
                    + float(item["bbox_normalized_xyxy"][2])
                ) / 2 <= x2
                and y1 <= (
                    float(item["bbox_normalized_xyxy"][1])
                    + float(item["bbox_normalized_xyxy"][3])
                ) / 2 <= y2
            ]
            # VisDrone boxes support object identity and registered image-plane
            # ROI occupancy, but not the contextual claim that the objects form
            # a road obstruction.  Keep the event at the ontology's neutral
            # value and derive only the explicitly registered ground-notification
            # action from observable object geometry.
            if request.mission == "visdrone_urban":
                event = (
                    "visible_priority_roi_occupancy" if priority_roi_hits
                    else "visible_priority_outside_roi" if priority_candidates
                    else "no_registered_priority_visible" if evidence
                    else "insufficient_evidence"
                )
            else:
                event = "none_visible" if evidence else "insufficient_evidence"
            risk = float(max(
                (float(item["confidence"]) for item in priority_roi_hits),
                default=0.0,
            ))
            if not evidence:
                source_supported_action = "REVIEW"
            elif priority_roi_hits and risk >= self.respond_threshold:
                source_supported_action = "RESPOND"
            elif priority_candidates:
                source_supported_action = "REVIEW"
            else:
                source_supported_action = "NO_ACTION"
        elif request.mission == "urban" and high_confidence >= 3:
            # This is a registered detector-count *action* rule: dense, confident
            # detections trigger a conservative obstruction alert. It makes the
            # comparator capable of RESPOND without using VLM-authored context.
            event = "visible_road_obstruction"
            risk = float(np.clip(0.60 + 0.10 * high_confidence, 0, 0.95))
            source_supported_action = None
        elif evidence:
            event = "none_visible"
            risk = float(np.clip(0.08 * high_confidence + 0.015 * max(0, object_count - 4), 0, 0.60))
            source_supported_action = None
        else:
            risk, event = 0.0, "insufficient_evidence"
            source_supported_action = None
        # Priority proposals can pass their registered threshold even when
        # the frame lacks sufficient evidence.  Preserve the frozen evidence
        # threshold and REVIEW rule, and give that state its required event.
        # Apply this after profile validation, including for legacy missions.
        if not evidence:
            event = "insufficient_evidence"
        action = (
            source_supported_action
            if source_supported_action is not None
            else "RESPOND" if evidence and risk >= self.respond_threshold
            else "REVIEW" if (not evidence or risk >= self.review_threshold)
            else "NO_ACTION"
        )
        uncertainty = float(np.clip(1 - max(confidences, default=0.0), 0, 1))
        action_confidence = (
            risk if action == "RESPOND"
            else 1.0 - risk if action == "NO_ACTION"
            else max(0.5, uncertainty)
        )
        evidence_records = [
            {
                "evidence_id": f"detector-object-{index}",
                "linked_claims": [f"object:{item['label']}"],
                "bbox_normalized_xyxy": list(item["bbox_normalized_xyxy"]),
            }
            for index, item in enumerate(proposals)
            if "bbox_normalized_xyxy" in item
        ]
        packet = {
            "schema_version": "semuav.v1", "observation_id": request.observation_id,
            "mission": request.mission, "capture_time_us": request.capture_time_us,
            "telemetry": dict(request.telemetry), "sufficient_evidence": evidence,
            "event": event, "objects": proposals, "relations": relations, "risk": risk,
            "uncertainty": uncertainty,
            "recommended_action": action, "evidence": evidence_records,
        }
        raw = json.dumps(packet, separators=(",", ":"))
        return InferenceRecord(
            request.observation_id, self.spec.model_id, self.spec.revision, raw,
            validate_semantic_content_packet(packet), True, None, None, None,
            float(np.clip(action_confidence, 0, 1)), "detector_rule_action_margin",
            (time.perf_counter() - start) * 1000, None, None, hash_object(prompt),
        )


def _hf_eos_token_ids(model: Any, processor: Any) -> frozenset[int]:
    """Resolve every configured Hugging Face end-of-sequence token ID."""

    output: set[int] = set()
    owners = (
        getattr(model, "generation_config", None),
        getattr(model, "config", None),
        getattr(processor, "tokenizer", None),
    )
    for owner in owners:
        value = getattr(owner, "eos_token_id", None) if owner is not None else None
        candidates = value if isinstance(value, (list, tuple, set, frozenset)) else (value,)
        for candidate in candidates:
            if candidate is None or isinstance(candidate, bool):
                continue
            try:
                output.add(int(candidate))
            except (TypeError, ValueError):
                continue
    return frozenset(output)


class HFMultimodalBackend(PerceptionBackend):
    """Pinned Hugging Face image-text adapter for Qwen and SmolVLM families."""

    def __init__(self, spec: ModelSpec, device: str = "cuda:0", dtype: str = "bfloat16"):
        spec.validate_full()
        self.spec, self.device, self.dtype_name = spec, device, dtype
        try:
            import torch  # type: ignore
            import transformers  # type: ignore
        except ImportError as exc:
            raise StudyGateError("Install torch and transformers for real VLM inference") from exc
        self.torch = torch
        local_files_kwargs = _model_local_files_kwargs()
        self.processor = transformers.AutoProcessor.from_pretrained(
            spec.repo,
            revision=spec.revision,
            trust_remote_code=spec.trust_remote_code,
            **local_files_kwargs,
        )
        model_kwargs: dict[str, Any] = {
            "revision": spec.revision,
            "trust_remote_code": spec.trust_remote_code,
            "torch_dtype": getattr(torch, dtype),
            **local_files_kwargs,
        }
        if spec.quantization == "int4":
            try:
                model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(load_in_4bit=True)
            except Exception as exc:
                raise StudyGateError("INT4 requires bitsandbytes and a compatible GPU") from exc
        model_class = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_class is None:
            model_class = getattr(transformers, "AutoModelForVision2Seq", None)
        if model_class is None:
            model_class = getattr(transformers, "AutoModelForMultimodalLM", None)
        if model_class is None:
            raise StudyGateError("Pinned Transformers build lacks a compatible multimodal auto class")
        if spec.quantization == "int4":
            model_kwargs["device_map"] = {"": device}
            self.model = model_class.from_pretrained(spec.repo, **model_kwargs).eval()
        else:
            self.model = model_class.from_pretrained(spec.repo, **model_kwargs).to(device).eval()

    def infer(self, request: InferenceRequest, prompt: str = MISSION_PROMPT, max_new_tokens: int = 384) -> InferenceRecord:
        torch = self.torch
        torch.cuda.reset_peak_memory_stats(self.device)
        from .hardware import profile_callable
        device_index = int(self.device.split(":")[-1]) if self.device.startswith("cuda") else 0

        def measured_path() -> dict[str, Any]:
            source_image = Image.open(request.image_path).convert("RGB")
            source_width, source_height = source_image.size
            image, preprocess = aspect_preserving_letterbox(source_image, self.spec.image_size)
            decision_contract = request.mission == "visdrone_urban"
            if decision_contract:
                context = visdrone_proposal_decision_context(
                    request, (source_width, source_height), self.spec.image_size,
                    preprocess=preprocess,
                )
            else:
                context = canonical_model_context(
                    request, (source_width, source_height), self.spec.image_size,
                    preprocess=preprocess,
                )
            user_message = {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt + "\nCONTEXT=" + json.dumps(context, separators=(",", ":"))},
            ]}
            generation_strategy = None
            generation_policy_sha256 = None
            if decision_contract:
                generation_strategy = VISDRONE_GENERATION_STRATEGY_BY_MODEL.get(
                    self.spec.model_id
                )
                if generation_strategy not in {
                    "qwen_open_brace_assistant_continuation",
                    "smol_native_double_quote_assistant_prefix",
                    "native_generation_prompt",
                }:
                    raise StudyGateError(
                        "No registered Hugging Face generation strategy exists for "
                        f"{self.spec.model_id!r}"
                    )
                generation_policy_sha256 = visdrone_generation_policy_sha256(
                    max_new_tokens
                )
            # These host-authored prefixes are transport scaffolds only: the
            # model still authors every decision value and must terminate
            # normally. Qwen uses its typed assistant-continuation path. The
            # locked SmolVLM2 processor instead needs its native user-only
            # generation render followed by a direct multimodal re-tokenization;
            # the real registered-train canary established that an exact ``{"``
            # prefix yields strict JSON without changing the parser, retrying,
            # or repairing model output.
            assistant_prefill = (
                visdrone_assistant_prefill_for_model(self.spec.model_id)
                if decision_contract else ""
            )
            if generation_strategy == "qwen_open_brace_assistant_continuation":
                messages = [
                    user_message,
                    {
                        "role": "assistant",
                        "content": [{
                            "type": "text",
                            "text": assistant_prefill,
                        }],
                    },
                ]
                template_kwargs = {
                    "add_generation_prompt": False,
                    "continue_final_message": True,
                }
                inputs = self.processor.apply_chat_template(
                    messages, tokenize=True,
                    return_dict=True, return_tensors="pt",
                    **template_kwargs,
                )
            elif generation_strategy == "smol_native_double_quote_assistant_prefix":
                messages = [user_message]
                template_kwargs = {
                    "add_generation_prompt": True,
                    "tokenize": False,
                }
                rendered_prompt = self.processor.apply_chat_template(
                    messages,
                    **template_kwargs,
                )
                if not isinstance(rendered_prompt, str) or not rendered_prompt:
                    raise StudyGateError(
                        "SmolVLM2 native chat template did not return text"
                    )
                inputs = self.processor(
                    text=rendered_prompt + assistant_prefill,
                    images=[image],
                    return_tensors="pt",
                )
            else:
                messages = [user_message]
                template_kwargs = {"add_generation_prompt": True}
                inputs = self.processor.apply_chat_template(
                    messages, tokenize=True,
                    return_dict=True, return_tensors="pt",
                    **template_kwargs,
                )
            inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            torch.cuda.synchronize(self.device)
            input_length = int(inputs["input_ids"].shape[-1])
            new_tokens = generated.sequences[:, input_length:]
            continuation = self.processor.batch_decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            raw = assistant_prefill + continuation
            output_length = int(new_tokens.shape[-1])
            eos_token_ids = _hf_eos_token_ids(self.model, self.processor)
            last_token_id = None
            if output_length:
                last_token = new_tokens[0, -1]
                if hasattr(last_token, "detach"):
                    last_token = last_token.detach()
                if hasattr(last_token, "cpu"):
                    last_token = last_token.cpu()
                if hasattr(last_token, "item"):
                    last_token = last_token.item()
                last_token_id = int(last_token)
            terminated_by_eos = (
                bool(last_token_id in eos_token_ids)
                if eos_token_ids and last_token_id is not None else None
            )
            reached_token_cap = (
                output_length >= max_new_tokens and terminated_by_eos is not True
            )
            confidence = None
            if generated.scores:
                probabilities = []
                for logits, token in zip(generated.scores, new_tokens[0]):
                    probabilities.append(float(torch.softmax(logits[0], dim=-1)[token].detach().cpu()))
                confidence = float(np.exp(np.mean(np.log(np.clip(probabilities, 1e-12, 1.0)))))
            parsed, valid, error = None, False, None
            decision_core = None
            decision_core_valid = None if not decision_contract else False
            try:
                if decision_contract:
                    decision_core = extract_exact_visdrone_proposal_decision(
                        raw,
                        proposal_count=len(context["p"]),
                    )
                    decision_core_valid = True
                    parsed = compose_visdrone_semantic_packet(
                        request,
                        decision_core,
                        (source_width, source_height),
                    )
                    if reached_token_cap:
                        raise ValueError(
                            "Proposal decision reached max_new_tokens before verified termination"
                        )
                    if terminated_by_eos is not True:
                        raise ValueError(
                            "Proposal decision did not terminate with a configured EOS token"
                        )
                else:
                    parsed = validate_semantic_content_packet(extract_exact_json_object(raw))
                valid = True
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            return {
                "raw": raw,
                "input_length": input_length,
                "output_length": output_length,
                "confidence": confidence,
                "parsed": parsed,
                "valid": valid,
                "error": error,
                "decision_core": decision_core,
                "decision_core_valid": decision_core_valid,
                "decision_context_sha256": hash_object(context) if decision_contract else None,
                "decision_proposal_count": len(context["p"]) if decision_contract else None,
                "terminated_by_eos": terminated_by_eos if decision_contract else None,
                "reached_token_cap": reached_token_cap if decision_contract else None,
                "generation_policy_sha256": generation_policy_sha256,
                "generation_strategy": generation_strategy,
                "assistant_prefill_sha256": (
                    sha256_bytes(assistant_prefill.encode("utf-8"))
                    if decision_contract else None
                ),
                "assistant_prefill_characters": (
                    len(assistant_prefill) if decision_contract else None
                ),
                "generated_continuation_sha256": (
                    sha256_bytes(continuation.encode("utf-8"))
                    if decision_contract else None
                ),
            }

        measured, profile = profile_callable(measured_path, device_index=device_index, measure_energy=True)
        return InferenceRecord(
            request.observation_id,
            self.spec.model_id,
            self.spec.revision,
            measured["raw"],
            measured["parsed"],
            measured["valid"],
            measured["error"],
            measured["input_length"],
            measured["output_length"],
            measured["confidence"],
            "generated_token_geometric_mean",
            profile["latency_ms"],
            float(torch.cuda.max_memory_allocated(self.device) / 2**20),
            profile["energy_j"],
            hash_object(prompt),
            profile["peak_temperature_c"],
            VISDRONE_PROPOSAL_DECISION_CONTRACT_VERSION if request.mission == "visdrone_urban" else None,
            measured["decision_core_valid"],
            (
                json.dumps(measured["decision_core"], sort_keys=True, separators=(",", ":"))
                if measured["decision_core"] is not None else None
            ),
            measured["decision_context_sha256"],
            measured["decision_proposal_count"],
            VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256 if request.mission == "visdrone_urban" else None,
            measured["terminated_by_eos"],
            measured["reached_token_cap"],
            generation_policy_version=(
                VISDRONE_GENERATION_POLICY_VERSION
                if request.mission == "visdrone_urban" else None
            ),
            generation_policy_sha256=measured["generation_policy_sha256"],
            generation_strategy=measured["generation_strategy"],
            generation_max_new_tokens=(
                int(max_new_tokens) if request.mission == "visdrone_urban" else None
            ),
            assistant_prefill_sha256=measured["assistant_prefill_sha256"],
            assistant_prefill_characters=measured["assistant_prefill_characters"],
            generated_continuation_sha256=measured[
                "generated_continuation_sha256"
            ],
        )


_MOONDREAM_TOKENIZER_ROUTE_LOCK = threading.RLock()
_MOONDREAM_NATIVE_GENERATION_CONTEXT_LIMIT = 2048


def _moondream_native_query_budget(
    model: Any,
    encoded_image: Any,
    question: str,
    max_new_tokens: int,
) -> dict[str, int]:
    """Measure and fail closed before Moondream indexes its fixed context."""

    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise ValueError("max_new_tokens must be a positive integer")
    native = getattr(model, "model", None)
    config = getattr(native, "config", None)
    text_config = getattr(config, "text", None)
    tokenizer_config = getattr(config, "tokenizer", None)
    tokenizer = getattr(native, "tokenizer", None)
    templates = getattr(tokenizer_config, "templates", None)
    query_template = templates.get("query") if isinstance(templates, Mapping) else None
    if (
        native is None
        or tokenizer is None
        or not isinstance(query_template, Mapping)
        or not isinstance(query_template.get("prefix"), (list, tuple))
        or not isinstance(query_template.get("suffix"), (list, tuple))
    ):
        raise StudyGateError("Locked Moondream native-query structure changed")
    try:
        image_positions = int(encoded_image.pos)
        model_context = int(text_config.max_context)
        vocab_size = int(text_config.vocab_size)
        question_token_ids = list(tokenizer.encode(question).ids)
    except (AttributeError, TypeError, ValueError) as exc:
        raise StudyGateError("Locked Moondream query budget is unreadable") from exc
    if image_positions <= 0 or model_context <= 0 or vocab_size <= 0:
        raise StudyGateError("Locked Moondream query budget contains invalid limits")
    native_prompt_token_ids = [
        *query_template["prefix"],
        *question_token_ids,
        *query_template["suffix"],
    ]
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        or token_id >= vocab_size
        for token_id in native_prompt_token_ids
    ):
        raise StudyGateError(
            "Locked Moondream native query contains an out-of-range token ID"
        )
    native_template_tokens = len(query_template["prefix"]) + len(
        query_template["suffix"]
    )
    question_tokens = len(question_token_ids)
    prefill_tokens = image_positions + native_template_tokens + question_tokens
    effective_context = min(
        model_context,
        _MOONDREAM_NATIVE_GENERATION_CONTEXT_LIMIT,
    )
    reserved_tokens = prefill_tokens + max_new_tokens
    if reserved_tokens > effective_context:
        raise StudyGateError(
            "Moondream native-query context budget exceeded before generation: "
            f"image={image_positions}, template={native_template_tokens}, "
            f"question={question_tokens}, generation={max_new_tokens}, "
            f"required={reserved_tokens}, limit={effective_context}"
        )
    return {
        "image_positions": image_positions,
        "native_template_tokens": native_template_tokens,
        "question_tokens": question_tokens,
        "prefill_tokens": prefill_tokens,
        "generation_reserve_tokens": max_new_tokens,
        "required_tokens": reserved_tokens,
        "effective_context_tokens": effective_context,
        "context_margin_tokens": effective_context - reserved_tokens,
    }


class MoondreamBackend(PerceptionBackend):
    def __init__(
        self,
        spec: ModelSpec,
        device: str = "cuda:0",
        *,
        tokenizer_artifact: Mapping[str, Any] | None = None,
    ):
        spec.validate_full()
        self.spec, self.device = spec, device
        self.tokenizer_artifact: dict[str, Any] | None = None
        try:
            import torch  # type: ignore
            from tokenizers import Tokenizer  # type: ignore
            from transformers import (  # type: ignore
                AutoConfig,
                AutoModelForCausalLM,
            )
            from transformers.dynamic_module_utils import (  # type: ignore
                get_class_from_dynamic_module,
            )
        except ImportError as exc:
            raise StudyGateError("Install torch and transformers for Moondream") from exc
        self.torch = torch
        load_kwargs = _moondream_load_kwargs()
        model_kwargs: dict[str, Any] = {
            "revision": spec.revision,
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            **load_kwargs,
        }

        if not load_kwargs.get("local_files_only"):
            self.model = AutoModelForCausalLM.from_pretrained(
                spec.repo, **model_kwargs,
            ).to(device).eval()
            return

        if tokenizer_artifact is None:
            raise StudyGateError(
                "Local-only Moondream requires its auxiliary Starmie tokenizer lock"
            )
        from .visdrone_models import (
            MOONDREAM_TOKENIZER_FILENAME,
            MOONDREAM_TOKENIZER_REPO,
            MOONDREAM_TOKENIZER_REVISION,
            verify_bound_moondream_tokenizer_artifact,
        )

        tokenizer_artifact = verify_bound_moondream_tokenizer_artifact(
            tokenizer_artifact
        )
        self.tokenizer_artifact = dict(tokenizer_artifact)

        expected_keys = {
            "repo", "revision", "filename", "path", "cache_root", "bytes",
            "sha256", "lock_path", "lock_sha256",
        }
        if set(tokenizer_artifact) != expected_keys:
            raise StudyGateError("Moondream Starmie tokenizer record changed")
        if (
            tokenizer_artifact["repo"] != MOONDREAM_TOKENIZER_REPO
            or tokenizer_artifact["revision"] != MOONDREAM_TOKENIZER_REVISION
            or tokenizer_artifact["filename"] != MOONDREAM_TOKENIZER_FILENAME
        ):
            raise StudyGateError("Moondream Starmie tokenizer identity changed")
        expected_sha256 = str(tokenizer_artifact["sha256"]).lower()
        if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
            raise StudyGateError(
                "Locked Moondream tokenizer SHA-256 is not 64 lowercase hex"
            )
        lexical_tokenizer = Path(str(tokenizer_artifact["path"])).expanduser()
        if not lexical_tokenizer.is_absolute() or not lexical_tokenizer.is_file():
            raise StudyGateError("Locked Moondream tokenizer file is missing")
        cache_root = Path(str(tokenizer_artifact["cache_root"])).resolve()
        expected_snapshot = (
            cache_root
            / "models--moondream--starmie-v1"
            / "snapshots"
            / MOONDREAM_TOKENIZER_REVISION
            / MOONDREAM_TOKENIZER_FILENAME
        )
        if lexical_tokenizer != expected_snapshot:
            raise StudyGateError(
                "Locked Moondream tokenizer is not the pinned Starmie snapshot"
            )
        resolved_tokenizer = lexical_tokenizer.resolve()
        try:
            resolved_tokenizer.relative_to(cache_root)
        except ValueError as exc:
            raise StudyGateError(
                "Locked Moondream tokenizer resolves outside the verified cache"
            ) from exc
        if hash_file(lexical_tokenizer) != expected_sha256:
            raise StudyGateError("Locked Moondream tokenizer SHA-256 changed")
        if (
            isinstance(tokenizer_artifact["bytes"], bool)
            or not isinstance(tokenizer_artifact["bytes"], int)
            or tokenizer_artifact["bytes"] != lexical_tokenizer.stat().st_size
        ):
            raise StudyGateError("Locked Moondream tokenizer byte count changed")

        # The locked Moondream implementation constructs its tokenizer through
        # Tokenizer.from_pretrained("moondream/starmie-v1"). That secondary
        # mutable repository is deliberately not authorized. Stage the exact
        # locked remote-code class first, then narrow only that constructor call
        # to tokenizer.json from the separate, immutable Starmie lock. The module
        # binding is restored even if model construction fails.
        config = AutoConfig.from_pretrained(
            spec.repo,
            revision=spec.revision,
            trust_remote_code=True,
            **load_kwargs,
        )
        auto_map = getattr(config, "auto_map", None)
        if (
            not isinstance(auto_map, Mapping)
            or auto_map.get("AutoModelForCausalLM")
            != "hf_moondream.HfMoondream"
        ):
            raise StudyGateError("Locked Moondream AutoModel route changed")
        class _LockedTokenizerProxy:
            @staticmethod
            def from_pretrained(identifier: str, *args: Any, **kwargs: Any) -> Any:
                if identifier != "moondream/starmie-v1" or args or kwargs:
                    raise StudyGateError(
                        "Moondream requested an unregistered secondary tokenizer"
                    )
                return Tokenizer.from_file(str(lexical_tokenizer))

        with _MOONDREAM_TOKENIZER_ROUTE_LOCK:
            # Resolve and validate the mutable module binding only while the
            # process-wide routing lock is held.  Otherwise a concurrent
            # Moondream construction could observe the temporary proxy and
            # either fail spuriously or restore another constructor's state.
            model_class = get_class_from_dynamic_module(
                auto_map["AutoModelForCausalLM"],
                spec.repo,
                revision=spec.revision,
                **load_kwargs,
            )
            hf_module = sys.modules.get(model_class.__module__)
            moon_class = getattr(hf_module, "MoondreamModel", None)
            moon_module = (
                sys.modules.get(getattr(moon_class, "__module__", ""))
                if moon_class is not None else None
            )
            if (
                moon_module is None
                or getattr(moon_module, "Tokenizer", None) is not Tokenizer
            ):
                raise StudyGateError(
                    "Locked Moondream tokenizer constructor route changed"
                )
            original_tokenizer = moon_module.Tokenizer
            moon_module.Tokenizer = _LockedTokenizerProxy
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    spec.repo,
                    config=config,
                    **model_kwargs,
                ).to(device).eval()
            finally:
                moon_module.Tokenizer = original_tokenizer

    def infer(self, request: InferenceRequest, prompt: str = MISSION_PROMPT, max_new_tokens: int = 384) -> InferenceRecord:
        self.torch.cuda.reset_peak_memory_stats(self.device)
        from .hardware import profile_callable
        device_index = int(self.device.split(":")[-1]) if self.device.startswith("cuda") else 0

        def measured_path() -> dict[str, Any]:
            source_image = Image.open(request.image_path).convert("RGB")
            source_width, source_height = source_image.size
            image, preprocess = aspect_preserving_letterbox(source_image, self.spec.image_size)
            decision_contract = request.mission == "visdrone_urban"
            generation_strategy = None
            generation_policy_sha256 = None
            if decision_contract:
                generation_strategy = VISDRONE_GENERATION_STRATEGY_BY_MODEL.get(
                    self.spec.model_id
                )
                if generation_strategy != "native_query":
                    raise StudyGateError(
                        "Moondream lacks its registered native-query generation policy"
                    )
                generation_policy_sha256 = visdrone_generation_policy_sha256(
                    max_new_tokens
                )
            if decision_contract:
                context = visdrone_proposal_decision_context(
                    request, (source_width, source_height), self.spec.image_size,
                    preprocess=preprocess,
                )
            else:
                context = canonical_model_context(
                    request, (source_width, source_height), self.spec.image_size,
                    preprocess=preprocess,
                )
            query = prompt + "\nCONTEXT=" + json.dumps(context, default=list, separators=(",", ":"))
            with self.torch.inference_mode():
                encoded_image = self.model.encode_image(image)
                query_budget = _moondream_native_query_budget(
                    self.model,
                    encoded_image,
                    query,
                    max_new_tokens,
                )
                answer = self.model.query(
                    image=encoded_image,
                    question=query,
                    settings={"max_tokens": max_new_tokens, "temperature": 0.0},
                )
            self.torch.cuda.synchronize(self.device)
            raw = answer["answer"] if isinstance(answer, Mapping) else str(answer)
            parsed, valid, error = None, False, None
            decision_core = None
            decision_core_valid = None if not decision_contract else False
            try:
                if decision_contract:
                    decision_core = extract_exact_visdrone_proposal_decision(
                        raw,
                        proposal_count=len(context["p"]),
                    )
                    decision_core_valid = True
                    parsed = compose_visdrone_semantic_packet(
                        request,
                        decision_core,
                        (source_width, source_height),
                    )
                else:
                    parsed = validate_semantic_content_packet(extract_exact_json_object(raw))
                valid = True
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            return {
                "raw": raw,
                "input_length": query_budget["prefill_tokens"],
                "parsed": parsed,
                "valid": valid,
                "error": error,
                "decision_core": decision_core,
                "decision_core_valid": decision_core_valid,
                "decision_context_sha256": hash_object(context) if decision_contract else None,
                "decision_proposal_count": len(context["p"]) if decision_contract else None,
                "generation_policy_sha256": generation_policy_sha256,
                "generation_strategy": generation_strategy,
                "assistant_prefill_sha256": (
                    sha256_bytes(b"") if decision_contract else None
                ),
                "assistant_prefill_characters": 0 if decision_contract else None,
                "generated_continuation_sha256": (
                    sha256_bytes(raw.encode("utf-8")) if decision_contract else None
                ),
                "query_budget": query_budget if decision_contract else None,
            }

        measured, profile = profile_callable(
            measured_path, device_index=device_index, measure_energy=True
        )
        query_budget = measured["query_budget"]
        tokenizer_artifact = getattr(self, "tokenizer_artifact", None)
        return InferenceRecord(
            request.observation_id,
            self.spec.model_id,
            self.spec.revision,
            measured["raw"],
            measured["parsed"],
            measured["valid"],
            measured["error"],
            measured["input_length"],
            None,
            None,
            "unavailable_not_self_reported",
            profile["latency_ms"],
            float(self.torch.cuda.max_memory_allocated(self.device) / 2**20),
            profile["energy_j"],
            hash_object(prompt),
            profile["peak_temperature_c"],
            VISDRONE_PROPOSAL_DECISION_CONTRACT_VERSION if request.mission == "visdrone_urban" else None,
            measured["decision_core_valid"],
            (
                json.dumps(measured["decision_core"], sort_keys=True, separators=(",", ":"))
                if measured["decision_core"] is not None else None
            ),
            measured["decision_context_sha256"],
            measured["decision_proposal_count"],
            VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256 if request.mission == "visdrone_urban" else None,
            None,
            None,
            generation_policy_version=(
                VISDRONE_GENERATION_POLICY_VERSION
                if request.mission == "visdrone_urban" else None
            ),
            generation_policy_sha256=measured["generation_policy_sha256"],
            generation_strategy=measured["generation_strategy"],
            generation_max_new_tokens=(
                int(max_new_tokens) if request.mission == "visdrone_urban" else None
            ),
            assistant_prefill_sha256=measured["assistant_prefill_sha256"],
            assistant_prefill_characters=measured["assistant_prefill_characters"],
            generated_continuation_sha256=measured[
                "generated_continuation_sha256"
            ],
            moondream_image_positions=(
                query_budget["image_positions"] if query_budget is not None else None
            ),
            moondream_native_template_tokens=(
                query_budget["native_template_tokens"]
                if query_budget is not None else None
            ),
            moondream_question_tokens=(
                query_budget["question_tokens"] if query_budget is not None else None
            ),
            moondream_prefill_tokens=(
                query_budget["prefill_tokens"] if query_budget is not None else None
            ),
            moondream_generation_reserve_tokens=(
                query_budget["generation_reserve_tokens"]
                if query_budget is not None else None
            ),
            moondream_required_tokens=(
                query_budget["required_tokens"] if query_budget is not None else None
            ),
            moondream_effective_context_tokens=(
                query_budget["effective_context_tokens"]
                if query_budget is not None else None
            ),
            moondream_context_margin_tokens=(
                query_budget["context_margin_tokens"]
                if query_budget is not None else None
            ),
            moondream_tokenizer_lock_sha256=(
                str(tokenizer_artifact["lock_sha256"])
                if tokenizer_artifact is not None and query_budget is not None
                else None
            ),
            moondream_tokenizer_artifact_sha256=(
                str(tokenizer_artifact["sha256"])
                if tokenizer_artifact is not None and query_budget is not None
                else None
            ),
        )


class UltralyticsDetectorTracker:
    def __init__(
        self,
        checkpoint: str,
        tracker: str = "botsort.yaml",
        device: str = "cuda:0",
        *,
        image_size: int = 1280,
        confidence: float = 0.001,
        iou: float = 0.70,
        max_detections: int = 300,
        seed: int = 20260823,
        top_k: int = 32,
        minimum_confidence: float = 0.05,
        ordering: str = DETECTOR_PROPOSAL_ORDERING,
    ):
        if not Path(checkpoint).is_file():
            raise StudyGateError(f"Detector checkpoint not found: {checkpoint}")
        if not str(tracker).strip():
            raise ValueError("Detector tracker identity must be nonempty")
        if not str(device).strip():
            raise ValueError("Detector device must be nonempty")
        inference_spec = detector_inference_spec(
            image_size=image_size,
            confidence=confidence,
            iou=iou,
            max_detections=max_detections,
            seed=seed,
            top_k=top_k,
            minimum_confidence=minimum_confidence,
            ordering=ordering,
        )
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise StudyGateError("Install ultralytics for the detector/tracker baseline") from exc
        self.model, self.tracker, self.device = YOLO(checkpoint), str(tracker), str(device)
        self.inference_spec = inference_spec

    def ensure_resident(self) -> dict[str, Any]:
        """Materialize the registered always-on detector on its deployment device.

        This must run even when the detector-output stage is restored from cache;
        otherwise a resumed VLM benchmark could silently omit detector residency
        from its peak-memory measurement.
        """
        if not hasattr(self.model, "to"):
            raise StudyGateError("Ultralytics detector does not expose device placement")
        self.model.to(self.device)
        underlying = getattr(self.model, "model", None)
        if underlying is not None and hasattr(underlying, "eval"):
            underlying.eval()
        torch = None
        if str(self.device).startswith("cuda"):
            try:
                import torch as torch_module  # type: ignore
            except ImportError as exc:
                raise StudyGateError("Torch is required to verify CUDA detector residency") from exc
            torch = torch_module
            if not torch.cuda.is_available():
                raise StudyGateError("CUDA detector residency was requested but CUDA is unavailable")
            torch.cuda.synchronize(self.device)
        parameters = list(underlying.parameters()) if underlying is not None and hasattr(underlying, "parameters") else []
        if not parameters:
            raise StudyGateError("Detector residency cannot be verified without model parameters")
        parameter_count = int(sum(parameter.numel() for parameter in parameters))
        if parameter_count <= 0:
            raise StudyGateError("Detector residency requires a positive model parameter count")
        observed_devices = {str(parameter.device) for parameter in parameters}
        requested_device = str(self.device)
        if requested_device == "cuda":
            requested_device = "cuda:0"
        normalized_observed = {
            "cuda:0" if device == "cuda" else device for device in observed_devices
        }
        if normalized_observed != {requested_device}:
            raise StudyGateError(
                f"Detector residency mismatch: requested {self.device}, observed {sorted(observed_devices)}"
            )
        cuda_allocated_bytes = int(torch.cuda.memory_allocated(self.device)) if torch is not None else 0
        if torch is not None and cuda_allocated_bytes <= 0:
            raise StudyGateError("CUDA detector residency has no measured allocated memory")
        return {
            "strategy": "always_on_detector_plus_one_triggered_context_model",
            "requested_device": str(self.device),
            "observed_device": sorted(observed_devices)[0],
            "parameter_count": parameter_count,
            "cuda_allocated_bytes": cuda_allocated_bytes,
        }

    def detect(self, image_path: str, persist: bool = False) -> list[dict[str, Any]]:
        spec = self.inference_spec
        results = self.model.track(
            image_path,
            tracker=self.tracker,
            persist=bool(persist),
            device=self.device,
            imgsz=spec["image_size"],
            conf=spec["confidence"],
            iou=spec["iou"],
            max_det=spec["max_detections"],
            deterministic=True,
            seed=spec["seed"],
            verbose=False,
        )
        output = []
        allowed_labels = set(spec["object_ontology"])
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            source_shape = getattr(result, "orig_shape", None)
            if (
                not isinstance(source_shape, (list, tuple))
                or len(source_shape) < 2
                or int(source_shape[0]) <= 0
                or int(source_shape[1]) <= 0
            ):
                raise StudyGateError("Ultralytics result lacks a valid original image shape")
            source_height, source_width = int(source_shape[0]), int(source_shape[1])
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
            for xyxy, cls, conf, track_id in zip(boxes.xyxy.cpu().tolist(), boxes.cls.int().cpu().tolist(), boxes.conf.cpu().tolist(), ids):
                names = getattr(result, "names", {})
                label = names.get(cls) if isinstance(names, Mapping) else names[cls]
                canonical_label = DETECTOR_LABEL_ALIASES.get(
                    str(label).strip().lower(), str(label).strip().lower(),
                )
                confidence_value = float(conf)
                if not math.isfinite(confidence_value) or not 0 <= confidence_value <= 1:
                    raise StudyGateError("Ultralytics emitted an invalid detector confidence")
                if canonical_label not in allowed_labels:
                    continue
                if confidence_value < float(spec["minimum_confidence"]):
                    continue
                normalized_box = [
                    float(xyxy[0]) / source_width, float(xyxy[1]) / source_height,
                    float(xyxy[2]) / source_width, float(xyxy[3]) / source_height,
                ]
                if (
                    not 0 <= normalized_box[0] < normalized_box[2] <= 1
                    or not 0 <= normalized_box[1] < normalized_box[3] <= 1
                ):
                    raise StudyGateError("Ultralytics emitted out-of-bounds detector geometry")
                canonical = {
                    "bbox_normalized_xyxy": normalized_box,
                    "label": canonical_label, "confidence": confidence_value,
                }
                if track_id is not None:
                    canonical["track_id"] = int(track_id)
                output.append(canonical)
        output.sort(key=lambda item: (
            -float(item["confidence"]),
            str(item["label"]),
            *tuple(float(value) for value in item["bbox_normalized_xyxy"]),
            int(item.get("track_id", 2**63 - 1)),
        ))
        return output[: int(spec["top_k"])]


def run_detector_cache(
    detector: UltralyticsDetectorTracker,
    manifest: pd.DataFrame,
    *,
    measure_energy: bool = False,
    device_index: int = 0,
) -> pd.DataFrame:
    required = {"sample_id", "image_path", "sequence_id", "frame_index"}
    if missing := required.difference(manifest.columns):
        raise ValueError(f"Detector manifest missing columns: {sorted(missing)}")
    rows = []
    ordered = manifest.copy()
    # The registered VisDrone manifest expands every source image into one row
    # per outcome-independent mission profile. Detector/tracker execution is
    # image-derived and must therefore happen once per source frame. Running
    # the persistent tracker on every profile copy would advance state multiple
    # times on the same pixels and make the paired profiles order-dependent.
    source_column = "source_sample_id" if "source_sample_id" in ordered else "sample_id"
    ordered["_detector_source_id"] = ordered[source_column].fillna("").astype(str)
    if ordered["_detector_source_id"].eq("").any():
        raise StudyGateError("Detector manifest contains an empty source-frame identity")
    invariant_columns = [
        column for column in (
            "image_path", "sequence_id", "frame_index", "dataset",
            "component", "dataset_component", "source_component",
        ) if column in ordered
    ]
    for column in invariant_columns:
        counts = ordered.groupby("_detector_source_id", sort=False)[column].nunique(dropna=False)
        if counts.gt(1).any():
            changed = counts[counts.gt(1)].index.astype(str).tolist()[:10]
            raise StudyGateError(
                f"Profile copies disagree on detector input {column!r}: {changed}"
            )
    profile_expansion = ordered[["sample_id", "_detector_source_id"]].copy()
    ordered = ordered.drop_duplicates("_detector_source_id", keep="first").copy()
    has_dataset_identity = "dataset" in ordered
    component_column = next(
        (
            column for column in ("component", "dataset_component", "source_component")
            if column in ordered
        ),
        None,
    )
    ordered["_tracking_dataset"] = (
        ordered["dataset"].fillna("").astype(str).str.strip()
        if has_dataset_identity else ""
    )
    ordered["_tracking_component"] = (
        ordered[component_column].fillna("").astype(str).str.strip()
        if component_column is not None else ""
    )
    ordered["_tracking_sequence"] = ordered["sequence_id"].fillna("").astype(str).str.strip()
    ordered["_tracking_frame"] = pd.to_numeric(ordered["frame_index"], errors="coerce")
    ordered["_tracking_input_order"] = np.arange(len(ordered))
    ordered = ordered.sort_values(
        [
            "_tracking_dataset", "_tracking_component", "_tracking_sequence",
            "_tracking_frame", "_tracking_input_order",
        ],
        na_position="last",
        kind="stable",
    )
    previous_tracking_identity: tuple[str, ...] | None = None
    for _, item in ordered.iterrows():
        dataset_id = str(item["_tracking_dataset"])
        component_id = str(item["_tracking_component"])
        sequence_id = str(item["_tracking_sequence"])
        tracking_identity = (
            (dataset_id, component_id, sequence_id)
            if component_id and sequence_id
            else (dataset_id, sequence_id)
            if sequence_id and (dataset_id or not has_dataset_identity)
            else None
        )
        persist = tracking_identity is not None and previous_tracking_identity == tracking_identity
        if measure_energy:
            from .hardware import profile_callable
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(device_index)
            except ImportError:
                torch = None  # type: ignore[assignment]
            proposals, profile = profile_callable(
                lambda: detector.detect(str(item["image_path"]), persist=persist),
                device_index=device_index, measure_energy=True,
            )
            latency_ms, energy_j = profile["latency_ms"], profile["energy_j"]
            peak_temperature_c = profile["peak_temperature_c"]
            peak_memory_mb = (
                float(torch.cuda.max_memory_allocated(device_index) / 2**20)
                if torch is not None and torch.cuda.is_available() else np.nan
            )
        else:
            start = time.perf_counter()
            proposals = detector.detect(str(item["image_path"]), persist=persist)
            latency_ms, energy_j = (time.perf_counter() - start) * 1000, np.nan
            peak_temperature_c, peak_memory_mb = np.nan, np.nan
        rows.append({
            "sample_id": item["_detector_source_id"],
            "source_sample_id": item["_detector_source_id"],
            "_detector_source_id": item["_detector_source_id"],
            "sequence_id": item["sequence_id"],
            "proposals_json": json.dumps(proposals, sort_keys=True, separators=(",", ":")),
            "latency_ms": latency_ms,
            "energy_j": energy_j,
            "peak_memory_mb": peak_memory_mb,
            "peak_temperature_c": peak_temperature_c,
        })
        previous_tracking_identity = tracking_identity
    source_results = pd.DataFrame(rows)
    expanded = profile_expansion.merge(
        source_results.drop(columns=["sample_id"]),
        on="_detector_source_id", validate="many_to_one", sort=False,
    ).drop(columns=["_detector_source_id"])
    # Profile copies must share byte-identical proposals and the exact same
    # measured detector cost. This protects paired comparisons if a future
    # refactor accidentally reintroduces profile-specific execution.
    equality_columns = [
        "proposals_json", "latency_ms", "energy_j", "peak_memory_mb",
        "peak_temperature_c",
    ]
    inconsistent = expanded.groupby("source_sample_id", sort=False)[equality_columns].nunique(
        dropna=False,
    ).gt(1).any(axis=1)
    if inconsistent.any():
        raise StudyGateError(
            "Detector proposal/cost replication differs across mission profiles"
        )
    return expanded


@dataclasses.dataclass
class TemperatureCalibrator:
    temperature: float = 1.0
    fitted: bool = False

    def fit(
        self,
        probabilities: Sequence[float],
        labels: Sequence[int],
        *,
        sample_weight: Sequence[float] | None = None,
    ) -> "TemperatureCalibrator":
        p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        y = np.asarray(labels, dtype=float)
        if p.ndim != 1 or y.ndim != 1 or len(p) != len(y) or len(p) == 0:
            raise ValueError("Calibration probabilities and labels must be equal nonempty vectors")
        if sample_weight is None:
            weights = np.ones(len(p), dtype=float)
        else:
            weights = np.asarray(sample_weight, dtype=float)
            if weights.ndim != 1 or len(weights) != len(p):
                raise ValueError("Calibration sample weights must match the observations")
            if not np.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("Calibration sample weights must be finite and positive")
        logits = np.log(p / (1 - p))
        def nll(log_temperature: float) -> float:
            q = 1 / (1 + np.exp(-logits / np.exp(log_temperature)))
            losses = -(y * np.log(q + 1e-12) + (1 - y) * np.log(1 - q + 1e-12))
            return float(np.average(losses, weights=weights))
        result = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
        self.temperature = float(np.exp(result.x))
        self.fitted = True
        return self

    def transform(self, probabilities: Sequence[float]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Calibrator is not fitted")
        p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        logits = np.log(p / (1 - p)) / self.temperature
        return 1 / (1 + np.exp(-logits))


def calibrated_semantic_packet_cache(
    records: pd.DataFrame,
    temperature: float,
    *,
    source_model: str,
    authoritative_manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Replace self-reported risk/uncertainty with a frozen observable score.

    The calibrated probability estimates whether the model's committed action
    is correct.  We derive a conservative mission-risk proxy from that
    probability and the committed action; raw model-authored confidence fields
    never reach the adaptive controller.
    """
    required = {
        "observation_id", "parsed_json", "schema_valid", "confidence",
        "model_revision", "prompt_hash", "confidence_source",
    }
    if missing := required.difference(records.columns):
        raise ValueError(f"Calibration cache missing columns: {sorted(missing)}")
    if records.empty:
        raise StudyGateError("Selected-model packet cache is empty")
    authoritative: dict[str, dict[str, Any]] = {}
    if authoritative_manifest is not None:
        needed = {"sample_id", "mission", "capture_time_us", "telemetry_json"}
        if missing := needed.difference(authoritative_manifest.columns):
            raise ValueError(f"Authoritative manifest missing columns: {sorted(missing)}")
        authoritative = authoritative_manifest.drop_duplicates("sample_id").set_index("sample_id")[
            ["mission", "capture_time_us", "telemetry_json"]
        ].to_dict("index")
    invalid = ~records["schema_valid"].astype(bool)
    if invalid.any() and not authoritative:
        raise StudyGateError(
            "Invalid model packets require authoritative mission/time/telemetry "
            "to construct an explicit abstention packet"
        )
    calibrator = TemperatureCalibrator(float(temperature), True)
    raw_confidence = pd.to_numeric(records["confidence"], errors="coerce").to_numpy(float)
    correctness_probability = np.zeros(len(records), dtype=float)
    observed = np.isfinite(raw_confidence)
    if observed.any():
        correctness_probability[observed] = calibrator.transform(raw_confidence[observed])
    rows: list[dict[str, Any]] = []
    for record, probability in zip(records.itertuples(index=False), correctness_probability):
        def optional_text(name: str) -> str | None:
            value = getattr(record, name, None)
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                return None
            normalized = str(value).strip()
            return normalized or None

        decision_contract_version = optional_text("decision_contract_version")
        decision_context_sha256 = optional_text("decision_context_sha256")
        decision_composer_policy_sha256 = optional_text(
            "decision_composer_policy_sha256"
        )
        decision_core_valid = getattr(record, "decision_core_valid", None)
        if decision_contract_version is not None:
            if decision_contract_version != VISDRONE_PROPOSAL_DECISION_CONTRACT_VERSION:
                raise StudyGateError("Unknown proposal-decision contract in inference cache")
            for name, value in (
                ("decision_context_sha256", decision_context_sha256),
                ("decision_composer_policy_sha256", decision_composer_policy_sha256),
            ):
                if value is None or not re.fullmatch(r"[0-9a-f]{64}", value):
                    raise StudyGateError(f"Inference cache lacks valid {name}")
        packet = None
        perception_valid = bool(
            record.schema_valid
            and getattr(record, "semantic_valid", True)
            and getattr(record, "authoritative_consistent", True)
            and isinstance(record.parsed_json, str)
            and (
                decision_contract_version is None
                or decision_core_valid is True
            )
        )
        if perception_valid:
            try:
                packet = validate_semantic_content_packet(json.loads(record.parsed_json))
            except Exception:
                perception_valid = False
        if perception_valid and packet is not None:
            pass
        else:
            source = authoritative.get(str(record.observation_id))
            if source is None:
                raise StudyGateError(
                    f"No authoritative record is available for invalid packet {record.observation_id}"
                )
            packet = {
                "schema_version": "semuav.v1",
                "observation_id": str(record.observation_id),
                "mission": str(source["mission"]),
                "capture_time_us": int(source["capture_time_us"]),
                "telemetry": json.loads(source["telemetry_json"]),
                "sufficient_evidence": False,
                "event": "insufficient_evidence",
                "objects": [],
                "relations": [],
                "risk": 0.5,
                "uncertainty": 1.0,
                "recommended_action": "REVIEW",
                "perception_failure": "schema_invalid_or_unparseable",
            }
        action = str(packet.get("recommended_action", "REVIEW"))
        p_correct = float(np.clip(probability, 0, 1)) if perception_valid else 0.0
        if action == "RESPOND":
            risk = p_correct
        elif action == "NO_ACTION":
            risk = 1.0 - p_correct
        else:
            risk = max(0.5, 1.0 - p_correct)
        packet["risk"] = float(np.clip(risk, 0, 1))
        packet["uncertainty"] = 1.0 - p_correct
        packet["calibrated_correctness_probability"] = p_correct
        packet["confidence_provenance"] = {
            "source": f"temperature_scaled:{record.confidence_source}",
            "source_model": source_model,
            "temperature": float(temperature),
        }
        # The strict model-output validator has already verified normalized,
        # grounded evidence. Preserve it exactly through calibration so an
        # event-only ROI cannot be silently replaced by a full-frame fallback.
        packet["evidence"] = json.loads(json.dumps(packet.get("evidence", [])))
        packet["provenance"] = {
            "source_model": source_model,
            "model_revision": str(record.model_revision),
            "prompt_hash": str(record.prompt_hash),
            "ontology_version": "semuav.ontology.v1",
            "calibration": {
                "method": "temperature_scaling",
                "temperature": float(temperature),
                "confidence_source": str(record.confidence_source),
            },
            "schema_version": "semuav.v1",
        }
        if decision_contract_version is not None:
            packet["provenance"].update({
                "semantic_packet_authorship": VISDRONE_COMPOSED_PACKET_AUTHORSHIP,
                "decision_contract_version": decision_contract_version,
                "decision_composer_policy_sha256": decision_composer_policy_sha256,
                "decision_context_sha256": decision_context_sha256,
            })
        validate_semantic_packet(packet, require_transmission_metadata=True)
        output_row = {
            "sample_id": str(record.observation_id),
            "packet_json": json.dumps(packet, sort_keys=True, separators=(",", ":")),
            "calibrated_correctness_probability": p_correct,
            "risk": packet["risk"],
            "uncertainty": packet["uncertainty"],
            "source_model": source_model,
            "perception_valid": perception_valid,
            "abstained": not bool(packet["sufficient_evidence"]),
        }
        if decision_contract_version is not None:
            output_row.update({
                "decision_contract_version": decision_contract_version,
                "decision_context_sha256": decision_context_sha256,
                "decision_composer_policy_sha256": decision_composer_policy_sha256,
            })
        rows.append(output_row)
    return pd.DataFrame(rows)


def expected_calibration_error(probabilities: Sequence[float], labels: Sequence[int], bins: int = 15) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    if len(p) == 0:
        return float("nan")
    quantiles = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 2:
        return float(abs(p.mean() - y.mean()))
    groups = np.clip(np.digitize(p, quantiles[1:-1], right=True), 0, len(quantiles) - 2)
    return float(sum(np.mean(groups == group) * abs(p[groups == group].mean() - y[groups == group].mean()) for group in np.unique(groups)))


def calibration_metrics(probabilities: Sequence[float], labels: Sequence[int]) -> dict[str, float]:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1 - 1e-8)
    y = np.asarray(labels, dtype=float)
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "ece_15_equal_mass": expected_calibration_error(p, y, 15),
    }


def run_inference_cache(
    backend: PerceptionBackend,
    requests: Iterable[InferenceRequest],
    prompt: str = MISSION_PROMPT,
    max_new_tokens: int = 384,
    *,
    scientific: bool,
) -> pd.DataFrame:
    if scientific and isinstance(backend, DeterministicMockBackend):
        raise StudyGateError("Mock perception is forbidden in the full profile")
    records = [backend.infer(request, prompt, max_new_tokens).as_row() for request in requests]
    return pd.DataFrame(records)


def run_inference_cache_sharded(
    backend: PerceptionBackend,
    requests: Sequence[InferenceRequest],
    ctx: RunContext,
    prompt: str = MISSION_PROMPT,
    max_new_tokens: int = 384,
    *,
    scientific: bool,
    shard_size: int = 32,
    execution_context: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    if scientific and not execution_context:
        raise StudyGateError(
            "Scientific VLM caching requires a detector co-residency execution context"
        )
    shards = []
    for offset in range(0, len(requests), shard_size):
        shard = list(requests[offset : offset + shard_size])
        ids = [request.observation_id for request in shard]
        name = f"01_vlm_{backend.spec.model_id}_shard_{offset // shard_size:05d}"
        key = {
            "model": backend.spec.model_id, "repo": backend.spec.repo,
            "revision": backend.spec.revision, "quantization": backend.spec.quantization,
            "image_size": list(backend.spec.image_size),
            "prompt": hash_object(prompt), "max_new_tokens": max_new_tokens,
            "execution_context": dict(execution_context or {}),
            "requests": [
                {
                    "observation_id": request.observation_id,
                    "image_sha256": hash_file(request.image_path),
                    "mission": request.mission,
                    "capture_time_us": request.capture_time_us,
                    "telemetry": dict(request.telemetry),
                    "detector_proposals": list(request.detector_proposals),
                }
                for request in shard
            ],
        }
        if os.environ.get("SEMUAV_ENABLE_V2R9_EXP1_RECOVERY") == "1":
            # The v2r13 correction may reuse only hash-verified raw v2r9
            # Experiment-1 frames.  The importer installs byte-identical gzip
            # content under the current cache key; frame_stage then follows its
            # ordinary checksum-verified resume path.  Derived metrics and all
            # downstream stages are recomputed by the caller.
            from .visdrone_recovery import import_v2r9_inference_stage

            import_v2r9_inference_stage(
                ctx,
                name,
                key,
                requests=shard,
                model_id=backend.spec.model_id,
                model_revision=backend.spec.revision,
                prompt=prompt,
            )
        shards.append(frame_stage(
            ctx, name, key,
            lambda shard=shard: run_inference_cache(
                backend, shard, prompt, max_new_tokens, scientific=scientific,
            ),
        ))
    return pd.concat(shards, ignore_index=True) if shards else pd.DataFrame()
