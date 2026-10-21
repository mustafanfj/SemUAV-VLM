from __future__ import annotations

import dataclasses
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .control import (
    AdaptiveImagePolicy,
    AdaptiveSemanticPolicy,
    CandidateProfile,
    ChannelMissionPolicy,
    ChannelOnlyPolicy,
    ControllerObservation,
    DeadlineGreedyPolicy,
    FixedModePolicy,
    FullPolicy,
    LinkEstimate,
    LookaheadContext,
    LookaheadReferencePolicy,
    MissionOnlyPolicy,
    OraclePolicy,
    QueueSnapshot,
    RepresentationPolicy,
)
from .evaluation import (
    add_primary_endpoint,
    assert_paired_design,
    minimum_sample_gate,
    pareto_front,
    resource_summary,
)
from .hardware import environment_report, import_embedded_jsonl
from .network import (
    Direction,
    EventType,
    LinkTrace,
    SelectiveRepeatSimulator,
    SerializationSpan,
    constant_trace,
    dynamic_trace_family,
    scientific_episode_coverage_audit,
    traces_to_frame,
)
from .perception import (
    DeterministicMockBackend,
    InferenceRequest,
    MISSION_PROMPT,
    VISDRONE_PROPOSAL_DECISION_CONTRACT_VERSION,
    run_inference_cache,
    run_inference_cache_sharded,
)
from .protocol import (
    AppMessage,
    PayloadMode,
    ReceiverSemanticState,
    ReliabilityProfile,
    SenderReceiverBelief,
    StateKind,
    apply_state_delta,
    deserialize_quantized_tensor,
    decode_mapping,
    encode_image,
    encode_video_clip,
    encode_mapping,
    fragment_message,
    make_state_delta,
    quantize_tensor,
    representation_wire_bytes,
    serialize_quantized_tensor,
    semantic_level_view,
    validate_semantic_content_packet,
    validate_semantic_packet,
)
from .registry import RunContext, StudyGateError, atomic_write_frame, atomic_write_json, frame_stage, hash_file, hash_object


EXPERIMENT_NAMES = {
    1: "Perception and packet validity",
    2: "Static representation frontier",
    3: "Static transport grid",
    4: "Adaptive controller on held-out traces",
    5: "Mission risk and uncertainty",
    6: "Receiver-state robustness",
    7: "Embedded end-to-end",
}


EXPERIMENT3_REGISTERED_LOSS_MODELS = ("iid", "gilbert_elliott")


COMPOSED_PIPELINE_METRIC_SCOPE = "composed_detector_verifier_pipeline"
DETECTOR_ONLY_METRIC_SCOPE = "detector_only_pipeline"
MODEL_AUTHORED_METRIC_SCOPE = "model_authored_semantic_packet"
COMPOSED_PIPELINE_CLAIM_SCOPE = (
    "detector_proposal_conditioned_end_to_end_pipeline_not_free_form_vlm_generation"
)


EXPERIMENT6_CAMPAIGN_VERSION = "semuav.exp6.v1"
EXPERIMENT6_REGISTERED_CASES = (
    "data_loss_retransmission",
    "fragment_ack_loss_application_ack_recovery",
    "application_ack_loss_safe_uncertainty",
    "data_reordering_delay",
    "application_ack_reordering_delay",
    "corruption_crc_retransmission",
    "missing_prerequisite_base_mismatch",
    "stale_state_rejected",
    "expired_state_rejected",
    "schema_version_mismatch_rejected",
    "model_version_mismatch_rejected",
    "repair_keyframe_cumulative_ack_recovery",
    "urgent_priority_bypass",
)
EXPERIMENT6_STATE_STRATEGY_VERSION = "semuav.exp6.state-strategy.v1"
EXPERIMENT6_REGISTERED_STATE_STRATEGIES = (
    "application_acked_deltas",
    "periodic_semantic_keyframes",
    "explicit_resynchronization",
)


def _stable_uniform(*parts: Any) -> float:
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).digest()
    return (int.from_bytes(digest[:8], "big") + 0.5) / 2**64


_NON_SEMANTIC_OBJECT_FIELDS = {
    "bbox", "bbox_xyxy", "bbox_normalized_xyxy", "box", "bounding_box",
    "confidence", "score", "probability", "track_id", "id", "class_id",
    "evidence_id",
}
_NON_SEMANTIC_RELATION_FIELDS = {
    "bbox", "bbox_xyxy", "bbox_normalized_xyxy", "box", "bounding_box",
    "confidence", "score", "probability", "evidence_id",
}


def _semantic_list(value: Any) -> list[Any] | None:
    """Return an annotated/predicted semantic list, preserving missingness."""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, list) else None
    return None


def _semantic_item_key(value: Any, *, kind: str) -> str:
    """Canonical semantic identity that excludes localization/confidence metadata."""
    if kind == "object":
        if isinstance(value, str):
            label = value.strip()
        elif isinstance(value, Mapping):
            label = next((
                str(value[key]).strip() for key in ("label", "class", "name")
                if isinstance(value.get(key), str) and str(value[key]).strip()
            ), "")
        else:
            label = ""
        if label:
            return json.dumps({"label": label}, sort_keys=True, separators=(",", ":"))
    if kind == "relation" and isinstance(value, Mapping):
        predicate = next((
            str(value[key]).strip() for key in ("predicate", "relation", "label", "type")
            if isinstance(value.get(key), str) and str(value[key]).strip()
        ), "")
        subject = next((
            str(value[key]).strip() for key in ("subject", "subject_id", "source")
            if isinstance(value.get(key), (str, int)) and not isinstance(value.get(key), bool)
            and str(value[key]).strip()
        ), "")
        object_value = next((
            str(value[key]).strip() for key in ("object", "object_id", "target")
            if isinstance(value.get(key), (str, int)) and not isinstance(value.get(key), bool)
            and str(value[key]).strip()
        ), "")
        if predicate:
            if predicate in {"near", "overlapping", "image_plane_overlap"} and subject and object_value:
                subject, object_value = sorted((subject, object_value))
            semantic_relation: dict[str, str] = {"predicate": predicate}
            if subject:
                semantic_relation["subject"] = subject
            if object_value:
                semantic_relation["object"] = object_value
            return json.dumps(
                semantic_relation, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False,
            )
    if isinstance(value, Mapping):
        ignored = _NON_SEMANTIC_OBJECT_FIELDS if kind == "object" else _NON_SEMANTIC_RELATION_FIELDS
        semantic = {
            str(key): item for key, item in value.items()
            if str(key).lower() not in ignored
        }
        # A geometry-only item has no semantic identity; retain the original so
        # two empty semantic projections do not become a fabricated match.
        value = semantic if semantic else dict(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _multiset_semantic_counts(
    gold: list[Any] | None,
    predicted: list[Any] | None,
    *,
    kind: str,
) -> dict[str, float]:
    if gold is None:
        return {
            "tp": np.nan, "fp": np.nan, "fn": np.nan,
            "precision": np.nan, "recall": np.nan, "f1": np.nan,
            "count_absolute_error": np.nan,
        }
    predicted = predicted if predicted is not None else []
    gold_counts = Counter(_semantic_item_key(item, kind=kind) for item in gold)
    predicted_counts = Counter(_semantic_item_key(item, kind=kind) for item in predicted)
    true_positive = float(sum((gold_counts & predicted_counts).values()))
    false_positive = float(sum((predicted_counts - gold_counts).values()))
    false_negative = float(sum((gold_counts - predicted_counts).values()))
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    f1_denominator = 2 * true_positive + false_positive + false_negative
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": true_positive / precision_denominator if precision_denominator else np.nan,
        "recall": true_positive / recall_denominator if recall_denominator else np.nan,
        "f1": 2 * true_positive / f1_denominator if f1_denominator else np.nan,
        "count_absolute_error": float(abs(len(predicted) - len(gold))),
    }


def _normalized_box(value: Any, *, explicitly_normalized: bool = False) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(box)):
        return None
    if not explicitly_normalized and not all(0.0 <= item <= 1.0 for item in box):
        # Pixel coordinates cannot be compared with normalized annotation boxes
        # without a declared source-image coordinate system.
        return None
    if not all(0.0 <= item <= 1.0 for item in box):
        return None
    x1, y1, x2, y2 = box
    return box if x2 > x1 and y2 > y1 else None


def _packet_evidence_boxes(packet: Mapping[str, Any] | None) -> list[tuple[float, float, float, float]]:
    if not isinstance(packet, Mapping):
        return []
    boxes: list[tuple[float, float, float, float]] = []
    for collection_name in ("evidence", "objects"):
        collection = packet.get(collection_name, [])
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, Mapping):
                continue
            candidate = None
            for key in ("bbox_normalized_xyxy", "normalized_bbox_xyxy", "bbox_xyxy", "bbox", "box"):
                if key in item:
                    candidate = _normalized_box(item[key], explicitly_normalized="normalized" in key)
                    if candidate is not None:
                        break
            if candidate is not None and candidate not in boxes:
                boxes.append(candidate)
    return boxes


def _box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def _group_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    if "visual_group_id" not in frame or frame["visual_group_id"].isna().any():
        return np.ones(len(frame), dtype=float)
    sizes = frame.groupby("visual_group_id", dropna=False)["visual_group_id"].transform("size").to_numpy(float)
    return 1.0 / sizes


def _weighted_mean(values: Any, weights: np.ndarray) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    valid = np.isfinite(numeric) & np.isfinite(weights) & (weights > 0)
    return float(np.average(numeric[valid], weights=weights[valid])) if valid.any() else np.nan


def _weighted_sum(values: Any, weights: np.ndarray) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    valid = np.isfinite(numeric) & np.isfinite(weights) & (weights > 0)
    return float(np.sum(numeric[valid] * weights[valid])) if valid.any() else np.nan


def _weighted_quantile(values: Any, weights: np.ndarray, quantile: float) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    valid = np.isfinite(numeric) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.nan
    numeric, valid_weights = numeric[valid], weights[valid]
    order = np.argsort(numeric, kind="mergesort")
    numeric, valid_weights = numeric[order], valid_weights[order]
    threshold = float(np.clip(quantile, 0, 1)) * valid_weights.sum()
    index = min(int(np.searchsorted(np.cumsum(valid_weights), threshold, side="left")), len(numeric) - 1)
    return float(numeric[index])


def _weighted_ece(probabilities: np.ndarray, labels: np.ndarray, weights: np.ndarray, bins: int = 15) -> float:
    valid = (
        np.isfinite(probabilities) & np.isfinite(labels) & np.isfinite(weights)
        & (weights > 0) & (probabilities >= 0) & (probabilities <= 1)
    )
    if not valid.any():
        return np.nan
    probabilities, labels, weights = probabilities[valid], labels[valid], weights[valid]
    order = np.argsort(probabilities, kind="mergesort")
    probabilities, labels, weights = probabilities[order], labels[order], weights[order]
    cumulative = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    assignments = np.minimum((cumulative * bins).astype(int), bins - 1)
    error = 0.0
    for assignment in np.unique(assignments):
        selected = assignments == assignment
        bin_weight = float(weights[selected].sum())
        error += (bin_weight / weights.sum()) * abs(
            float(np.average(probabilities[selected], weights=weights[selected]))
            - float(np.average(labels[selected], weights=weights[selected]))
        )
    return float(error)


def _weighted_average_precision(probabilities: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    valid = (
        np.isfinite(probabilities) & np.isfinite(labels) & np.isfinite(weights)
        & (weights > 0) & (probabilities >= 0) & (probabilities <= 1)
    )
    probabilities, labels, weights = probabilities[valid], labels[valid], weights[valid]
    positive_weight = float(np.sum(weights * (labels == 1)))
    negative_weight = float(np.sum(weights * (labels == 0)))
    if positive_weight <= 0 or negative_weight <= 0:
        return np.nan
    order = np.argsort(-probabilities, kind="mergesort")
    labels, weights = labels[order], weights[order]
    cumulative_positive = np.cumsum(weights * (labels == 1))
    cumulative_total = np.cumsum(weights)
    precision = cumulative_positive / cumulative_total
    return float(np.sum(precision * weights * (labels == 1)) / positive_weight)


def make_dry_run_fixtures(root: str | Path, n_observations: int = 36) -> pd.DataFrame:
    root = Path(root)
    image_root = root / "dry_run_images"
    image_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(n_observations):
        image_path = image_root / f"fixture_{index:04d}.png"
        if not image_path.exists():
            image = Image.new("RGB", (192, 128), (35 + index % 40, 60, 75))
            draw = ImageDraw.Draw(image)
            for object_index in range(1 + index % 5):
                x = 10 + 30 * object_index
                y = 25 + (13 * index + 7 * object_index) % 65
                draw.rectangle((x, y, x + 18, y + 10), outline=(240, 210, 70), width=2)
            image.save(image_path)
        critical = index % 7 == 0
        rows.append({
            "sample_id": f"dry-{index:04d}",
            "visual_group_id": f"dry-sequence-{index:04d}",
            "sequence_id": f"dry-sequence-{index:04d}",
            "event_group": f"dry-event-{index:04d}",
            "dataset": "GENERATED_FIXTURE",
            "mission": "wildfire" if index % 9 == 0 else "urban",
            "mission_profile": ["routine", "standard", "high_priority"][index % 3],
            "split": ["train", "calibration", "validation", "test"][index % 4],
            "cohort": "natural_test" if index % 4 == 3 else "development",
            "image_path": str(image_path),
            "capture_time_us": index * 1_000_000,
            "trace_time_us": index * 5_000_000,
            "critical": critical,
            "gold_action": "RESPOND" if critical else "NO_ACTION",
            "deadline_us": index * 1_000_000 + (700_000 if critical else 2_000_000),
            "risk": 0.88 if critical else 0.08,
            "uncertainty": 0.48 if index % 11 == 0 else 0.12,
            "battery_fraction": 0.85 - 0.005 * index,
            "thermal_c": 52.0 + 0.4 * (index % 12),
            "ood": index % 13 == 0,
            "telemetry_json": json.dumps({"gps_lat": 24.523, "gps_lon": 54.434, "altitude_m": 75.0, "yaw_deg": float(index * 11 % 360)}),
        })
    frame = pd.DataFrame(rows)
    all_paths = frame["image_path"].tolist()
    frame["clip_paths_json"] = [
        json.dumps(all_paths[max(0, min(index - 1, len(all_paths) - 3)) : max(0, min(index - 1, len(all_paths) - 3)) + 3])
        for index in range(len(frame))
    ]
    return frame


def experiment1_perception(
    manifest: pd.DataFrame,
    backends: Sequence[Any],
    *,
    scientific: bool,
    max_new_tokens: int = 384,
    detector_cache: pd.DataFrame | None = None,
    ctx: RunContext | None = None,
    shard_size: int = 32,
    prompt: str = MISSION_PROMPT,
    inference_cache_context: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    proposal_lookup: dict[str, Any] = {}
    detector_metric_lookup: dict[str, dict[str, Any]] = {}
    if detector_cache is not None:
        required = {"sample_id", "proposals_json"}
        if missing := required.difference(detector_cache.columns):
            raise ValueError(f"Detector cache missing columns: {sorted(missing)}")
        proposal_lookup = detector_cache.drop_duplicates("sample_id").set_index("sample_id")["proposals_json"].to_dict()
        measured_columns = [
            column for column in (
                "latency_ms", "energy_j", "peak_memory_mb", "peak_temperature_c",
            ) if column in detector_cache
        ]
        if measured_columns:
            detector_metric_lookup = (
                detector_cache.drop_duplicates("sample_id")
                .set_index("sample_id")[measured_columns]
                .to_dict("index")
            )
    if scientific and not proposal_lookup:
        raise StudyGateError("Full Experiment 1 requires the frozen detector/tracker proposal cache")
    records = []
    requests = [
        InferenceRequest(
            str(row.sample_id), str(row.image_path), str(row.mission), int(row.capture_time_us),
            json.loads(row.telemetry_json), tuple(json.loads(proposal_lookup.get(str(row.sample_id), "[]"))),
        )
        for row in manifest.itertuples(index=False)
    ]
    for backend in backends:
        frame = (
            run_inference_cache_sharded(
                backend, requests, ctx, prompt, max_new_tokens,
                scientific=scientific, shard_size=shard_size,
                execution_context=inference_cache_context,
            )
            if ctx is not None else
            run_inference_cache(backend, requests, prompt, max_new_tokens, scientific=scientific)
        )
        records.append(frame)
    cache = pd.concat(records, ignore_index=True)
    cache["peak_temperature_c"] = pd.to_numeric(
        cache.get("peak_temperature_c", pd.Series(np.nan, index=cache.index)), errors="coerce",
    )
    detector_only = cache["model_id"].eq("detector_only")
    if detector_metric_lookup and detector_only.any():
        ids = cache["observation_id"].astype(str)
        for column in ("energy_j", "peak_memory_mb", "peak_temperature_c"):
            values = ids.map(
                lambda sample_id, column=column: detector_metric_lookup.get(sample_id, {}).get(column, np.nan)
            )
            if column not in cache:
                cache[column] = np.nan
            cache.loc[detector_only, column] = values.loc[detector_only]
        detector_latency = ids.map(
            lambda sample_id: detector_metric_lookup.get(sample_id, {}).get("latency_ms", np.nan)
        )
        if scientific:
            missing_measures = []
            for column in ("latency_ms", "energy_j", "peak_memory_mb"):
                values = ids.map(
                    lambda sample_id, column=column: detector_metric_lookup.get(sample_id, {}).get(column, np.nan)
                )
                if pd.to_numeric(values.loc[detector_only], errors="coerce").isna().any():
                    missing_measures.append(column)
            if missing_measures:
                raise StudyGateError(
                    "Detector-only Experiment 1 rows lack measured "
                    + ", ".join(missing_measures)
                )
        cache.loc[detector_only, "latency_ms"] = (
            pd.to_numeric(cache.loc[detector_only, "latency_ms"], errors="coerce")
            + pd.to_numeric(detector_latency.loc[detector_only], errors="coerce")
        )
    ids = cache["observation_id"].astype(str)
    detector_stage_latency = ids.map(
        lambda sample_id: detector_metric_lookup.get(sample_id, {}).get("latency_ms", 0.0)
    )
    detector_stage_energy = ids.map(
        lambda sample_id: detector_metric_lookup.get(sample_id, {}).get("energy_j", 0.0)
    )
    cache["detector_stage_latency_ms"] = pd.to_numeric(detector_stage_latency, errors="coerce")
    cache["detector_stage_energy_j"] = pd.to_numeric(detector_stage_energy, errors="coerce")
    context_model = ~detector_only
    cache["full_path_latency_ms"] = pd.to_numeric(cache["latency_ms"], errors="coerce")
    cache.loc[context_model, "full_path_latency_ms"] += cache.loc[
        context_model, "detector_stage_latency_ms"
    ]
    cache["full_path_energy_j"] = pd.to_numeric(cache["energy_j"], errors="coerce")
    cache.loc[context_model, "full_path_energy_j"] += cache.loc[
        context_model, "detector_stage_energy_j"
    ]
    cache["full_path_peak_memory_mb"] = pd.to_numeric(
        cache.get("peak_memory_mb", pd.Series(np.nan, index=cache.index)), errors="coerce",
    )
    cache["temperature_available"] = cache["peak_temperature_c"].map(np.isfinite)
    gold_columns = [column for column in (
        "sample_id", "gold_action", "critical", "visual_group_id", "split",
        "event", "objects", "relations", "sufficient_evidence", "evidence_roi",
        "mission", "capture_time_us", "telemetry_json",
    ) if column in manifest]
    gold = manifest[gold_columns].rename(columns={
        "sample_id": "observation_id", "event": "gold_event",
        "objects": "gold_objects", "relations": "gold_relations",
        "mission": "authoritative_mission",
        "capture_time_us": "authoritative_capture_time_us",
        "telemetry_json": "authoritative_telemetry_json",
    })
    scored = cache.merge(gold, on="observation_id", validate="many_to_one")
    is_visdrone = scored["authoritative_mission"].astype(str).eq("visdrone_urban")
    is_detector_only = scored["model_id"].astype(str).eq("detector_only")
    decision_contract = scored.get(
        "decision_contract_version", pd.Series(None, index=scored.index, dtype=object),
    )
    composed_context = is_visdrone & ~is_detector_only & decision_contract.astype(str).eq(
        VISDRONE_PROPOSAL_DECISION_CONTRACT_VERSION
    )
    if scientific:
        unbound_context = is_visdrone & ~is_detector_only & ~composed_context
        if unbound_context.any():
            offenders = sorted(scored.loc[unbound_context, "model_id"].astype(str).unique())
            raise StudyGateError(
                "Scientific VisDrone context-model rows are not bound to the registered "
                f"proposal-decision contract: {offenders}"
            )
    metric_scope = np.where(
        composed_context,
        COMPOSED_PIPELINE_METRIC_SCOPE,
        np.where(
            is_visdrone & is_detector_only,
            DETECTOR_ONLY_METRIC_SCOPE,
            MODEL_AUTHORED_METRIC_SCOPE,
        ),
    )
    # These are packet/pipeline outcomes.  For the amended VisDrone path the
    # VLM authors only a bounded proposal decision; event, objects, relations,
    # evidence and action are deterministic host-composer outputs.
    for column in (
        "action_metric_scope", "event_metric_scope",
        "object_metric_scope", "relation_metric_scope",
    ):
        scored[column] = metric_scope
    scored["semantic_claim_scope"] = np.where(
        composed_context,
        COMPOSED_PIPELINE_CLAIM_SCOPE,
        np.where(
            is_visdrone & is_detector_only,
            "detector_only_pipeline_baseline",
            "model_authored_semantic_packet",
        ),
    )
    scored["detector_proposal_conditioned"] = composed_context
    scored["free_form_vlm_packet_authorship_claim_eligible"] = ~is_visdrone
    scored["objects_exact_metric_definition"] = np.where(
        composed_context | (is_visdrone & is_detector_only),
        (
            "multiset_object_label_exact_geometry_ignored_"
            "conditioned_on_frozen_detector_proposals"
        ),
        "multiset_semantic_identity_exact_geometry_ignored",
    )
    def action(value: Any) -> str:
        if not isinstance(value, str) or not value:
            return "INVALID"
        try:
            return str(json.loads(value).get("recommended_action", "INVALID"))
        except Exception:
            return "INVALID"
    scored["predicted_action"] = scored["parsed_json"].map(action)
    def packet_field(value: Any, field: str, default: Any = None) -> Any:
        try:
            return json.loads(value).get(field, default) if isinstance(value, str) and value else default
        except Exception:
            return default
    def parsed_packet(value: Any) -> Mapping[str, Any] | None:
        try:
            parsed = json.loads(value) if isinstance(value, str) and value else None
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, Mapping) else None

    parsed_packets = [parsed_packet(value) for value in scored["parsed_json"]]
    def content_valid(packet: Mapping[str, Any] | None) -> bool:
        if packet is None:
            return False
        try:
            validate_semantic_content_packet(packet)
            return True
        except Exception:
            return False
    # Recompute validity in the evaluator instead of trusting backend-authored
    # flags. This is the exact frozen-ontology criterion used for model selection.
    scored["semantic_valid"] = [content_valid(packet) for packet in parsed_packets]
    def authoritative_match(row: Any, packet: Mapping[str, Any] | None) -> bool:
        if packet is None:
            return False
        required = (
            "authoritative_mission", "authoritative_capture_time_us",
            "authoritative_telemetry_json",
        )
        if any(not hasattr(row, column) for column in required):
            return False
        try:
            expected_telemetry = json.loads(row.authoritative_telemetry_json)
            observed_telemetry = packet["telemetry"]

            def authoritative_equal(expected: Any, observed: Any) -> bool:
                if (
                    isinstance(expected, (int, float))
                    and not isinstance(expected, bool)
                    and isinstance(observed, (int, float))
                    and not isinstance(observed, bool)
                ):
                    return bool(
                        np.isfinite(float(expected))
                        and np.isfinite(float(observed))
                        and float(expected) == float(observed)
                    )
                if isinstance(expected, Mapping) and isinstance(observed, Mapping):
                    return set(expected) == set(observed) and all(
                        authoritative_equal(expected[key], observed[key])
                        for key in expected
                    )
                if (
                    isinstance(expected, Sequence)
                    and not isinstance(expected, (str, bytes))
                    and isinstance(observed, Sequence)
                    and not isinstance(observed, (str, bytes))
                ):
                    return len(expected) == len(observed) and all(
                        authoritative_equal(left, right)
                        for left, right in zip(expected, observed)
                    )
                return expected == observed

            telemetry_equal = authoritative_equal(
                expected_telemetry, observed_telemetry
            )
            return bool(
                str(packet["observation_id"]) == str(row.observation_id)
                and str(packet["mission"]) == str(row.authoritative_mission)
                and int(packet["capture_time_us"]) == int(row.authoritative_capture_time_us)
                and telemetry_equal
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
    scored["authoritative_consistent"] = [
        authoritative_match(row, packet)
        for row, packet in zip(scored.itertuples(index=False), parsed_packets)
    ]
    scored["predicted_event"] = scored["parsed_json"].map(lambda value: packet_field(value, "event", "INVALID"))
    scored["predicted_objects"] = scored["parsed_json"].map(lambda value: packet_field(value, "objects", []))
    scored["predicted_relations"] = scored["parsed_json"].map(lambda value: packet_field(value, "relations", []))
    scored["predicted_sufficient_evidence"] = scored["parsed_json"].map(lambda value: packet_field(value, "sufficient_evidence", None))
    scored["action_correct"] = scored["predicted_action"] == scored["gold_action"]
    scored["event_correct"] = scored["predicted_event"] == scored["gold_event"] if "gold_event" in scored else np.nan
    def semantic_exact(gold_value: Any, predicted_value: Any, *, kind: str) -> bool | float:
        gold_items = _semantic_list(gold_value)
        if gold_items is None:
            return np.nan
        predicted_items = _semantic_list(predicted_value) or []
        return Counter(
            _semantic_item_key(item, kind=kind) for item in gold_items
        ) == Counter(
            _semantic_item_key(item, kind=kind) for item in predicted_items
        )
    scored["objects_exact"] = (
        [
            semantic_exact(gold_value, predicted_value, kind="object")
            for gold_value, predicted_value in zip(scored["gold_objects"], scored["predicted_objects"])
        ] if "gold_objects" in scored else np.nan
    )
    scored["relations_exact"] = (
        [
            semantic_exact(gold_value, predicted_value, kind="relation")
            for gold_value, predicted_value in zip(scored["gold_relations"], scored["predicted_relations"])
        ] if "gold_relations" in scored else np.nan
    )
    scored["evidence_state_correct"] = scored["predicted_sufficient_evidence"] == scored["sufficient_evidence"] if "sufficient_evidence" in scored else np.nan

    for prefix, gold_column, predicted_column, kind in (
        ("object", "gold_objects", "predicted_objects", "object"),
        ("relation", "gold_relations", "predicted_relations", "relation"),
    ):
        if gold_column in scored:
            metric_rows = [
                _multiset_semantic_counts(
                    _semantic_list(gold_value), _semantic_list(predicted_value), kind=kind,
                )
                for gold_value, predicted_value in zip(scored[gold_column], scored[predicted_column])
            ]
        else:
            metric_rows = [
                _multiset_semantic_counts(None, _semantic_list(predicted_value), kind=kind)
                for predicted_value in scored[predicted_column]
            ]
        metrics = pd.DataFrame(metric_rows, index=scored.index)
        for name in ("tp", "fp", "fn", "precision", "recall", "f1", "count_absolute_error"):
            scored[f"{prefix}_{name}"] = metrics[name]
        scored[f"{prefix}_metrics_available"] = scored[f"{prefix}_tp"].notna()

    no_claim_events = {
        "",
        "none_visible",
        "no_registered_priority_visible",
        "insufficient_evidence",
        "invalid",
        "none",
        "nan",
    }
    event_rows = []
    for row in scored.itertuples(index=False):
        if not hasattr(row, "gold_event") or pd.isna(row.gold_event):
            event_rows.append({"tp": np.nan, "fp": np.nan, "fn": np.nan})
            continue
        gold_event = str(row.gold_event).strip().lower()
        predicted_event = str(row.predicted_event).strip().lower()
        gold_claims = [] if gold_event in no_claim_events else [gold_event]
        predicted_claims = [] if predicted_event in no_claim_events else [predicted_event]
        counts = _multiset_semantic_counts(gold_claims, predicted_claims, kind="relation")
        event_rows.append({name: counts[name] for name in ("tp", "fp", "fn")})
    event_metrics = pd.DataFrame(event_rows, index=scored.index)
    for name in ("tp", "fp", "fn"):
        scored[f"event_{name}"] = event_metrics[name]

    evaluated_components = pd.concat([
        scored["event_tp"].notna(),
        scored["object_metrics_available"],
        scored["relation_metrics_available"],
    ], axis=1).any(axis=1)
    grounded_tp = scored[["event_tp", "object_tp", "relation_tp"]].sum(axis=1, min_count=1)
    grounded_fp = scored[["event_fp", "object_fp", "relation_fp"]].sum(axis=1, min_count=1)
    grounded_fn = scored[["event_fn", "object_fn", "relation_fn"]].sum(axis=1, min_count=1)
    grounded_tp = grounded_tp.where(evaluated_components)
    grounded_fp = grounded_fp.where(evaluated_components)
    grounded_fn = grounded_fn.where(evaluated_components)
    scored["grounded_claim_tp"] = grounded_tp
    scored["grounded_claim_fp"] = grounded_fp
    scored["grounded_claim_fn"] = grounded_fn
    scored["grounded_hallucination"] = (grounded_fp > 0).where(evaluated_components)
    scored["grounded_omission"] = (grounded_fn > 0).where(evaluated_components)
    scored["grounded_hallucination_rate"] = (
        grounded_fp / (grounded_tp + grounded_fp)
    ).where((grounded_tp + grounded_fp) > 0)
    scored["grounded_omission_rate"] = (
        grounded_fn / (grounded_tp + grounded_fn)
    ).where((grounded_tp + grounded_fn) > 0)
    # Backwards-compatible name now has the proposal-defined, gold-grounded meaning.
    scored["hallucination"] = scored["grounded_hallucination"]
    scored["output_invalid"] = (
        ~scored["schema_valid"].astype(bool)
        | ~scored["semantic_valid"].astype(bool)
        | ~scored["authoritative_consistent"].astype(bool)
        | scored["predicted_action"].eq("INVALID")
    )
    # An invalid output cannot receive component-level task credit merely
    # because its parser fallbacks happen to match an empty gold list.  This is
    # especially important for the contextual-model GO/NO-GO score, which
    # averages these exact-match components.
    for correctness_column in (
        "action_correct", "event_correct", "objects_exact",
        "relations_exact", "evidence_state_correct",
    ):
        if correctness_column in scored:
            available = scored[correctness_column].notna()
            invalid_available = available & scored["output_invalid"]
            if invalid_available.any():
                if not pd.api.types.is_bool_dtype(scored[correctness_column].dtype):
                    scored[correctness_column] = scored[correctness_column].astype(object)
                scored.loc[invalid_available, correctness_column] = False

    # REVIEW is a terminal ground action, not an abstention token.  Coverage is
    # defined by the packet's explicit evidence state.
    scored["abstained"] = (
        scored["predicted_sufficient_evidence"].ne(True) | scored["output_invalid"]
    )
    scored["selective_covered"] = ~scored["abstained"]
    scored["selective_action_correct"] = scored["action_correct"].where(scored["selective_covered"])

    evidence_rows = []
    for position, row in enumerate(scored.itertuples(index=False)):
        gold_roi = _normalized_box(getattr(row, "evidence_roi", None), explicitly_normalized=True)
        predicted_boxes = _packet_evidence_boxes(parsed_packets[position])
        if gold_roi is None:
            evidence_rows.append({
                "evaluable": False, "box_available": np.nan, "best_iou": np.nan,
                "recall_at_0_5": np.nan, "recall_at_0_7": np.nan,
            })
            continue
        best_iou = max((_box_iou(gold_roi, box) for box in predicted_boxes), default=0.0)
        evidence_rows.append({
            "evaluable": True, "box_available": bool(predicted_boxes), "best_iou": best_iou,
            "recall_at_0_5": best_iou >= 0.5, "recall_at_0_7": best_iou >= 0.7,
        })
    evidence_metrics = pd.DataFrame(evidence_rows, index=scored.index)
    scored["evidence_box_evaluable"] = evidence_metrics["evaluable"].astype(bool)
    scored["evidence_box_available"] = evidence_metrics["box_available"]
    scored["evidence_box_best_iou"] = evidence_metrics["best_iou"]
    scored["evidence_box_recall_at_0_5"] = evidence_metrics["recall_at_0_5"]
    scored["evidence_box_recall_at_0_7"] = evidence_metrics["recall_at_0_7"]

    raw_confidence = pd.to_numeric(scored["confidence"], errors="coerce")
    scored["raw_confidence_valid"] = raw_confidence.notna() & raw_confidence.between(0, 1)
    scored["raw_confidence_brier"] = (
        (raw_confidence - scored["action_correct"].astype(float)) ** 2
    ).where(scored["raw_confidence_valid"])

    def summarize_model(group: pd.DataFrame) -> dict[str, Any]:
        weights = _group_equal_weights(group)
        scope_columns = (
            "action_metric_scope", "event_metric_scope",
            "object_metric_scope", "relation_metric_scope",
        )
        observed_scopes = {
            str(value)
            for column in scope_columns
            for value in group[column].dropna().astype(str).unique()
        }
        if len(observed_scopes) != 1:
            raise StudyGateError(
                f"Model {group.iloc[0]['model_id']!r} mixes semantic metric scopes: "
                f"{sorted(observed_scopes)}"
            )
        metric_scope_value = next(iter(observed_scopes))
        claim_scopes = set(group["semantic_claim_scope"].dropna().astype(str).unique())
        if len(claim_scopes) != 1:
            raise StudyGateError(
                f"Model {group.iloc[0]['model_id']!r} mixes semantic claim scopes"
            )
        def mean(column: str) -> float:
            return _weighted_mean(group[column], weights) if column in group else np.nan
        def weighted_count(column: str) -> float:
            return _weighted_sum(group[column], weights) if column in group else np.nan
        def count_metrics(prefix: str) -> tuple[float, float, float]:
            true_positive = weighted_count(f"{prefix}_tp")
            false_positive = weighted_count(f"{prefix}_fp")
            false_negative = weighted_count(f"{prefix}_fn")
            if not all(np.isfinite(value) for value in (true_positive, false_positive, false_negative)):
                return np.nan, np.nan, np.nan
            precision_denominator = true_positive + false_positive
            recall_denominator = true_positive + false_negative
            f1_denominator = 2 * true_positive + false_positive + false_negative
            return (
                true_positive / precision_denominator if precision_denominator else np.nan,
                true_positive / recall_denominator if recall_denominator else np.nan,
                2 * true_positive / f1_denominator if f1_denominator else np.nan,
            )
        object_precision, object_recall, object_f1 = count_metrics("object")
        relation_precision, relation_recall, relation_f1 = count_metrics("relation")
        confidence = pd.to_numeric(group["confidence"], errors="coerce").to_numpy(float)
        labels = group["action_correct"].astype(float).to_numpy()
        valid_confidence = np.isfinite(confidence) & (confidence >= 0) & (confidence <= 1)
        confidence_groups = (
            group.loc[valid_confidence, "visual_group_id"].nunique()
            if "visual_group_id" in group else int(valid_confidence.sum())
        )
        evidence_mask = group["evidence_box_evaluable"].astype(bool).to_numpy()
        evidence_weights = weights[evidence_mask]
        evidence_iou = group.loc[evidence_mask, "evidence_box_best_iou"]
        critical_mask = (
            group["critical"].astype(bool).to_numpy()
            if "critical" in group else np.zeros(len(group), dtype=bool)
        )
        peak_memory = pd.to_numeric(group["peak_memory_mb"], errors="coerce")
        peak_temperature = pd.to_numeric(group["peak_temperature_c"], errors="coerce")
        valid_temperature = peak_temperature.map(np.isfinite).to_numpy(dtype=bool)
        return {
            "model_id": str(group.iloc[0]["model_id"]),
            "samples": int(len(group)),
            "independent_visual_groups": int(group["visual_group_id"].nunique()) if "visual_group_id" in group else int(len(group)),
            "action_metric_scope": metric_scope_value,
            "event_metric_scope": metric_scope_value,
            "object_metric_scope": metric_scope_value,
            "relation_metric_scope": metric_scope_value,
            "semantic_claim_scope": next(iter(claim_scopes)),
            "detector_proposal_conditioned": bool(
                group["detector_proposal_conditioned"].astype(bool).all()
            ),
            "free_form_vlm_packet_authorship_claim_eligible": bool(
                group["free_form_vlm_packet_authorship_claim_eligible"].astype(bool).all()
            ),
            "objects_exact_metric_definition": str(
                group["objects_exact_metric_definition"].iloc[0]
            ),
            "schema_valid_rate": mean("schema_valid"),
            "semantic_valid_rate": mean("semantic_valid"),
            "authoritative_consistency_rate": mean("authoritative_consistent"),
            "output_invalid_rate": mean("output_invalid"),
            "action_accuracy": mean("action_correct"),
            "event_accuracy": mean("event_correct"),
            "objects_exact_accuracy": mean("objects_exact"),
            "relations_exact_accuracy": mean("relations_exact"),
            "evidence_state_accuracy": mean("evidence_state_correct"),
            "object_precision": object_precision,
            "object_recall": object_recall,
            "object_f1": object_f1,
            "object_count_mae": mean("object_count_absolute_error"),
            "object_metric_coverage": mean("object_metrics_available"),
            "relation_precision": relation_precision,
            "relation_recall": relation_recall,
            "relation_f1": relation_f1,
            "relation_count_mae": mean("relation_count_absolute_error"),
            "relation_metric_coverage": mean("relation_metrics_available"),
            "grounded_hallucination_rate": mean("grounded_hallucination"),
            "grounded_omission_rate": mean("grounded_omission"),
            "hallucination_rate": mean("grounded_hallucination"),
            "abstention_rate": mean("abstained"),
            "selective_coverage": mean("selective_covered"),
            "selective_action_accuracy": mean("selective_action_correct"),
            "evidence_roi_units": int(evidence_mask.sum()),
            "evidence_roi_visual_groups": int(group.loc[evidence_mask, "visual_group_id"].nunique()) if "visual_group_id" in group else int(evidence_mask.sum()),
            "evidence_box_coverage": _weighted_mean(group.loc[evidence_mask, "evidence_box_available"], evidence_weights) if evidence_mask.any() else np.nan,
            "evidence_box_median_iou": _weighted_quantile(evidence_iou, evidence_weights, 0.5) if evidence_mask.any() else np.nan,
            "evidence_box_recall_at_0_5": _weighted_mean(group.loc[evidence_mask, "evidence_box_recall_at_0_5"], evidence_weights) if evidence_mask.any() else np.nan,
            "evidence_box_recall_at_0_7": _weighted_mean(group.loc[evidence_mask, "evidence_box_recall_at_0_7"], evidence_weights) if evidence_mask.any() else np.nan,
            "raw_confidence_units": int(valid_confidence.sum()),
            "raw_confidence_visual_groups": int(confidence_groups),
            "raw_confidence_coverage": mean("raw_confidence_valid"),
            "raw_confidence_brier": _weighted_mean(group.loc[valid_confidence, "raw_confidence_brier"], weights[valid_confidence]) if valid_confidence.any() else np.nan,
            "raw_confidence_ece_15_equal_mass": _weighted_ece(confidence, labels, weights, 15),
            "raw_confidence_auprc": _weighted_average_precision(confidence, labels, weights),
            "critical_action_recall": _weighted_mean(group.loc[critical_mask, "action_correct"], weights[critical_mask]) if critical_mask.any() else np.nan,
            "latency_p50_ms": _weighted_quantile(group["latency_ms"], weights, 0.50),
            "latency_p95_ms": _weighted_quantile(group["latency_ms"], weights, 0.95),
            "full_path_latency_p50_ms": _weighted_quantile(group["full_path_latency_ms"], weights, 0.50),
            "full_path_latency_p95_ms": _weighted_quantile(group["full_path_latency_ms"], weights, 0.95),
            "peak_memory_mb": float(peak_memory.max()) if peak_memory.notna().any() else np.nan,
            "temperature_units": int(valid_temperature.sum()),
            "temperature_coverage": mean("temperature_available"),
            "peak_temperature_c": (
                float(peak_temperature.loc[valid_temperature].max())
                if valid_temperature.any() else np.nan
            ),
            "energy_j": mean("energy_j"),
            "full_path_energy_j": mean("full_path_energy_j"),
        }

    summary = pd.DataFrame([
        summarize_model(group)
        for _, group in scored.groupby("model_id", sort=True, dropna=False)
    ])
    return scored, summary


def _semantic_packet_from_manifest(row: Any) -> dict[str, Any]:
    sufficient_evidence = not bool(row.ood)
    event = (
        "visible_flame" if row.mission == "wildfire" else "visible_road_obstruction"
    ) if bool(row.critical) else "none_visible"
    if not sufficient_evidence:
        event = "insufficient_evidence"
    return {
        "schema_version": "semuav.v1",
        "observation_id": row.sample_id,
        "mission": row.mission,
        "capture_time_us": int(row.capture_time_us),
        "telemetry": json.loads(row.telemetry_json),
        "sufficient_evidence": sufficient_evidence,
        "event": event,
        "objects": [],
        "relations": [],
        "risk": float(row.risk),
        "uncertainty": float(row.uncertainty),
        "calibrated_correctness_probability": float(np.clip(1.0 - row.uncertainty, 0, 1)),
        "recommended_action": row.gold_action if sufficient_evidence else "REVIEW",
        "evidence": [],
        "confidence_provenance": {
            "source": "generated_fixture_probability",
            "source_model": "GENERATED_FIXTURE",
            "temperature": 1.0,
        },
        "provenance": {
            "source_model": "GENERATED_FIXTURE",
            "model_revision": "0" * 40,
            "prompt_hash": "NON_SCIENTIFIC_FIXTURE",
            "ontology_version": "semuav.ontology.v1",
            "calibration": {"method": "fixture", "temperature": 1.0},
            "schema_version": "semuav.v1",
        },
    }


def select_grounded_semantic_roi(
    packet: Mapping[str, Any], image_size: tuple[int, int],
) -> list[float]:
    """Choose the first model-grounded ROI without consulting annotations."""
    image_width, image_height = map(int, image_size)
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_size must contain positive width and height")
    geometry_records = [
        *[item for item in packet.get("evidence", []) if isinstance(item, Mapping)],
        *[item for item in packet.get("objects", []) if isinstance(item, Mapping)],
    ]
    for obj in geometry_records:
        if not isinstance(obj, Mapping):
            continue
        normalized = "bbox_normalized_xyxy" in obj
        box = obj.get(
            "bbox_normalized_xyxy",
            obj.get("bbox_xyxy", obj.get("bbox", obj.get("box"))),
        )
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            values = [float(value) for value in box]
        except (TypeError, ValueError):
            continue
        if not all(np.isfinite(values)):
            continue
        if not normalized and max(map(abs, values)) > 1.0:
            values = [
                values[0] / image_width, values[1] / image_height,
                values[2] / image_width, values[3] / image_height,
            ]
        x1, y1, x2, y2 = [float(np.clip(value, 0, 1)) for value in values]
        if x2 > x1 and y2 > y1:
            return [x1, y1, x2, y2]
    return [0.0, 0.0, 1.0, 1.0]


def experiment2_representation_frontier(
    manifest: pd.DataFrame,
    *,
    scientific: bool,
    mtu_bytes: int = 1200,
    semantic_packet_cache: pd.DataFrame | None = None,
    native_receiver_cache: pd.DataFrame | None = None,
    registered_feature_cache: pd.DataFrame | None = None,
    semantic_source_model: str = "deployed_semantic_model",
    image_codecs: Sequence[str] = ("JPEG", "WEBP", "AVIF"),
    image_qualities: Sequence[int] = (25, 50, 80),
    image_scales: Sequence[float] = (0.5, 1.0),
    video_crfs: Sequence[int] = (23, 35),
    video_fps: int = 5,
) -> pd.DataFrame:
    video_crfs = tuple(int(value) for value in video_crfs)
    if not video_crfs or len(set(video_crfs)) != len(video_crfs):
        raise ValueError("video_crfs must be a nonempty unique sequence")
    if any(value < 0 or value > 51 for value in video_crfs):
        raise ValueError("video_crfs must lie within [0, 51]")
    video_fps = int(video_fps)
    if video_fps <= 0:
        raise ValueError("video_fps must be positive")
    if scientific:
        if semantic_packet_cache is None or native_receiver_cache is None:
            raise StudyGateError(
                "Full Experiment 2 requires frozen semantic-packet and native image/ROI receiver caches; gold labels may not generate candidates"
            )
        required_semantic = {
            "sample_id", "packet_json", "calibrated_correctness_probability", "source_model",
        }
        required_receiver = {
            "sample_id", "variant", "source_model", "receiver_action",
            "generation_ms", "energy_j",
            "receiver_latency_ms", "receiver_energy_j", "hardware_id",
            "receiver_revision", "payload_sha256", "cost_scope",
        }
        if required_semantic.difference(semantic_packet_cache.columns) or required_receiver.difference(native_receiver_cache.columns):
            raise StudyGateError("Experiment 2 cache schemas are incomplete")
        if semantic_packet_cache["sample_id"].astype(str).duplicated().any():
            duplicate_rows = semantic_packet_cache[
                semantic_packet_cache["sample_id"].astype(str).duplicated(keep=False)
            ]
            if any(
                group[["packet_json", "calibrated_correctness_probability", "source_model"]]
                .astype(str).nunique(dropna=False).gt(1).any()
                for _, group in duplicate_rows.groupby(duplicate_rows["sample_id"].astype(str))
            ):
                raise StudyGateError("Conflicting duplicate rows exist in the semantic packet cache")
        semantic_sources = set(semantic_packet_cache["source_model"].dropna().astype(str))
        if semantic_sources != {str(semantic_source_model)}:
            raise StudyGateError(
                "Semantic packet cache source_model differs from the registered deployed source"
            )
        receiver_key = ["sample_id", "variant", "source_model"]
        for _, duplicate_group in native_receiver_cache.groupby(receiver_key, dropna=False):
            if len(duplicate_group) > 1 and any(
                duplicate_group[column].astype(str).nunique(dropna=False) > 1
                for column in required_receiver.difference(receiver_key)
            ):
                raise StudyGateError("Conflicting duplicate rows exist in the frozen native receiver cache")
        if registered_feature_cache is not None:
            manifest_ids = set(manifest["sample_id"].astype(str))
            feature_ids = set(registered_feature_cache["sample_id"].astype(str))
            if missing_feature_ids := sorted(manifest_ids - feature_ids):
                raise StudyGateError(
                    f"Registered L4 feature cache misses {len(missing_feature_ids)} sample(s); "
                    f"examples: {missing_feature_ids[:5]}"
                )
            variants_by_sample = registered_feature_cache.groupby(
                registered_feature_cache["sample_id"].astype(str)
            )["variant"].apply(lambda values: tuple(sorted(set(map(str, values)))))
            if variants_by_sample.nunique() != 1:
                raise StudyGateError("Registered L4 variants are not complete and identical across samples")
    semantic_lookup = (
        semantic_packet_cache.drop_duplicates("sample_id").set_index("sample_id").to_dict("index")
        if semantic_packet_cache is not None else {}
    )
    receiver_columns = [
        "receiver_action", "generation_ms", "energy_j",
        "receiver_latency_ms", "receiver_energy_j", "hardware_id",
        "receiver_revision", "payload_sha256", "cost_scope",
    ]
    receiver_lookup = (
        native_receiver_cache.drop_duplicates(["sample_id", "variant", "source_model"])
        .set_index(["sample_id", "variant", "source_model"])[receiver_columns].to_dict("index")
        if native_receiver_cache is not None else {}
    )
    def receiver_outcome(
        sample_id: str, variant: str, source_model: str, payload: bytes,
        fixture_action: str, fixture_correct: bool,
        fixture_generation_ms: float, fixture_energy_j: float,
        fixture_receiver_latency_ms: float = 1.0,
        fixture_receiver_energy_j: float = 0.001,
    ) -> tuple[str, bool, float, float, float, float]:
        if scientific:
            value = receiver_lookup.get((sample_id, variant, source_model))
            if value is None:
                raise StudyGateError(
                    f"Missing frozen native-receiver output for {sample_id}/{variant}/{source_model}"
                )
            actual_hash = hashlib.sha256(payload).hexdigest()
            if str(value["payload_sha256"]) != actual_hash:
                raise StudyGateError(
                    f"Native receiver payload hash mismatch for {sample_id}/{variant}/{source_model}"
                )
            if not str(value["hardware_id"]).strip() or not str(value["receiver_revision"]).strip():
                raise StudyGateError("Native receiver hardware/revision provenance is empty")
            if str(value["cost_scope"]) != "representation_encoding_only_excludes_perception":
                raise StudyGateError(
                    "Native sender cost_scope must exclude detector/VLM perception to prevent double charging"
                )
            return (
                str(value["receiver_action"]), False,
                float(value["generation_ms"]), float(value["energy_j"]),
                float(value["receiver_latency_ms"]), float(value["receiver_energy_j"]),
            )
        return (
            fixture_action, fixture_correct, fixture_generation_ms, fixture_energy_j,
            fixture_receiver_latency_ms, fixture_receiver_energy_j,
        )

    def add_candidate(
        *, sample_id: str, visual_group_id: str, mode: PayloadMode, variant: str,
        payload: bytes, source_model: str, fixture_action: str,
        fixture_correct: bool, gold_action: str, fixture_generation_ms: float,
        fixture_energy_j: float, fixture_receiver_latency_ms: float = 1.0,
        fixture_receiver_energy_j: float = 0.001,
        calibrated_correctness_probability: float | None = None,
    ) -> None:
        action, correct, generation_ms, sender_energy_j, receiver_latency_ms, receiver_energy_j = receiver_outcome(
            sample_id, variant, source_model, payload, fixture_action, fixture_correct,
            fixture_generation_ms, fixture_energy_j, fixture_receiver_latency_ms,
            fixture_receiver_energy_j,
        )
        if scientific:
            # Correctness is evaluation truth and is never trusted from a
            # sender/receiver cache that can affect causal policy selection.
            correct = str(action) == str(gold_action)
        rows.append({
            "sample_id": sample_id, "visual_group_id": visual_group_id,
            "mode": mode.name, "variant": variant, "source_model": source_model,
            "payload_bytes": len(payload),
            "wire_bytes": representation_wire_bytes(payload, mtu_bytes),
            "receiver_action": action, "native_receiver_correct": correct,
            "generation_ms": generation_ms, "energy_j": sender_energy_j,
            "receiver_latency_ms": receiver_latency_ms,
            "receiver_energy_j": receiver_energy_j, "payload_blob": payload.hex(),
            "calibrated_correctness_probability": calibrated_correctness_probability,
        })
    rows: list[dict[str, Any]] = []
    for item in manifest.itertuples(index=False):
        image = Image.open(item.image_path).convert("RGB")
        semantic_record = semantic_lookup.get(item.sample_id, {})
        packet = json.loads(semantic_record["packet_json"]) if scientific else _semantic_packet_from_manifest(item)
        if scientific:
            packet = validate_semantic_packet(packet, require_transmission_metadata=True)
            if str(packet["provenance"]["source_model"]) != str(semantic_source_model):
                raise StudyGateError(
                    f"Semantic packet provenance source differs for sample {item.sample_id}"
                )
        calibrated_probability = (
            float(semantic_record["calibrated_correctness_probability"])
            if scientific else float(1.0 - packet["uncertainty"])
        )
        for level in (PayloadMode.L0_HEARTBEAT, PayloadMode.L1_ALERT, PayloadMode.L2_SEMANTICS):
            if level is PayloadMode.L0_HEARTBEAT:
                content = semantic_level_view(packet, level)
                receiver_action, success = "NO_ACTION", item.gold_action == "NO_ACTION"
            elif level is PayloadMode.L1_ALERT:
                content = semantic_level_view(packet, level)
                receiver_action, success = packet["recommended_action"], packet["recommended_action"] == item.gold_action
            else:
                content = packet
                receiver_action, success = packet["recommended_action"], packet["recommended_action"] == item.gold_action
            payload, wire_format = encode_mapping(content, scientific=scientific)
            # Receiver-cache identity is (sample_id, variant, source_model),
            # so the three semantic levels must not all collapse to the same
            # bare "cbor" variant while carrying different payload hashes.
            semantic_variant = f"l{int(level)}_{wire_format}"
            add_candidate(
                sample_id=item.sample_id, visual_group_id=item.visual_group_id,
                mode=level, variant=semantic_variant, payload=payload,
                source_model=semantic_source_model, fixture_action=receiver_action,
                fixture_correct=success, gold_action=item.gold_action, fixture_generation_ms=0.08,
                fixture_energy_j=0.0002, fixture_receiver_latency_ms=0.25,
                fixture_receiver_energy_j=0.0001,
                calibrated_correctness_probability=calibrated_probability,
            )
        # L3 carries the full semantics plus an actually encoded region of interest.
        if scientific:
            # ROI generation is sender-side and may not use the annotation
            # evidence box.  Prefer the first grounded object proposal emitted
            # by the frozen detector/VLM packet; otherwise escalate to the full
            # frame as explicit insufficient evidence.
            normalized_roi = select_grounded_semantic_roi(packet, image.size)
            x1, y1, x2, y2 = normalized_roi
            roi_box = (
                int(x1 * image.width), int(y1 * image.height),
                max(int(x1 * image.width) + 1, int(x2 * image.width)),
                max(int(y1 * image.height) + 1, int(y2 * image.height)),
            )
        else:
            roi_box = (0, 0, max(1, image.width // 2), max(1, image.height // 2))
        roi = image.crop(roi_box)
        roi_bytes = encode_image(roi, "JPEG", 50, 1.0)
        l3_packet = json.loads(json.dumps(packet))
        l3_packet.setdefault("evidence", []).append({
            "evidence_id": "transmitted-roi-0",
            "linked_claims": ["event", "recommended_action"],
            "bbox_normalized_xyxy": [float(value) for value in normalized_roi]
            if scientific else [0.0, 0.0, 0.5, 0.5],
            "codec": "jpeg",
            "integrity_sha256": hashlib.sha256(roi_bytes).hexdigest(),
        })
        if scientific:
            validate_semantic_packet(l3_packet, require_transmission_metadata=True)
        semantics, wire_format = encode_mapping(l3_packet, scientific=scientific)
        l3 = len(semantics).to_bytes(4, "big") + semantics + roi_bytes
        l3_variant = f"{wire_format}+jpeg_q50_roi"
        add_candidate(
            sample_id=item.sample_id, visual_group_id=item.visual_group_id,
            mode=PayloadMode.L3_SEMANTICS_ROI, variant=l3_variant, payload=l3,
            source_model=semantic_source_model,
            fixture_action=packet["recommended_action"],
            fixture_correct=packet["recommended_action"] == item.gold_action,
            gold_action=item.gold_action,
            fixture_generation_ms=1.2, fixture_energy_j=0.004,
            fixture_receiver_latency_ms=2.0, fixture_receiver_energy_j=0.002,
            calibrated_correctness_probability=calibrated_probability,
        )
        # Registered feature fixture exercises the real INT8/INT4 packers. Full mode
        # must replace this with the frozen encoder/split/head cache.
        feature = np.asarray(image.resize((16, 16)), dtype=np.float32).transpose(2, 0, 1) / 255
        if scientific:
            # L4 is included only from a frozen, registered sender/receiver cache.
            if registered_feature_cache is not None:
                feature_rows = registered_feature_cache[registered_feature_cache["sample_id"] == item.sample_id]
                required = {
                    "sample_id", "variant", "payload_blob", "payload_bytes",
                    "payload_sha256",
                    "receiver_action", "generation_ms",
                    "energy_j", "receiver_latency_ms", "receiver_energy_j",
                    "encoder_revision", "split_layer", "receiver_head_revision",
                    "tensor_shape", "quantization", "quantization_scale",
                    "quantization_zero_point", "cost_scope",
                }
                if required.difference(feature_rows.columns):
                    raise StudyGateError("Registered feature cache schema is incomplete")
                for feature_row in feature_rows.itertuples(index=False):
                    feature_payload = bytes.fromhex(feature_row.payload_blob)
                    if len(feature_payload) != int(feature_row.payload_bytes):
                        raise StudyGateError("Registered feature payload length does not match payload_bytes")
                    if hashlib.sha256(feature_payload).hexdigest() != str(feature_row.payload_sha256):
                        raise StudyGateError("Registered feature payload hash does not match payload_blob")
                    try:
                        decoded_feature = deserialize_quantized_tensor(feature_payload)
                        declared_shape = tuple(json.loads(str(feature_row.tensor_shape)))
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise StudyGateError("Registered feature wire payload or tensor shape is invalid") from exc
                    if decoded_feature.shape != declared_shape:
                        raise StudyGateError("Registered feature tensor shape differs from its wire payload")
                    if str(feature_row.quantization) != f"int{decoded_feature.bits}":
                        raise StudyGateError("Registered feature quantization differs from its wire payload")
                    if not math.isclose(
                        float(feature_row.quantization_scale), decoded_feature.scale,
                        rel_tol=0.0, abs_tol=0.0,
                    ):
                        raise StudyGateError("Registered feature scale differs from its wire payload")
                    if int(feature_row.quantization_zero_point) != decoded_feature.zero_point:
                        raise StudyGateError("Registered feature zero point differs from its wire payload")
                    for provenance in (feature_row.encoder_revision, feature_row.split_layer, feature_row.receiver_head_revision, feature_row.tensor_shape, feature_row.quantization):
                        if not str(provenance).strip():
                            raise StudyGateError("Registered feature provenance is incomplete")
                    if str(feature_row.cost_scope) != "representation_encoding_only_excludes_perception":
                        raise StudyGateError("Registered feature cost_scope would double-charge perception")
                    rows.append({
                        "sample_id": item.sample_id, "visual_group_id": item.visual_group_id,
                        "mode": PayloadMode.L4_FEATURES.name, "variant": feature_row.variant,
                        "source_model": "registered_feature_receiver",
                        "payload_bytes": int(feature_row.payload_bytes),
                        "wire_bytes": representation_wire_bytes(feature_payload, mtu_bytes),
                        "receiver_action": feature_row.receiver_action,
                        "native_receiver_correct": str(feature_row.receiver_action) == str(item.gold_action),
                        "generation_ms": float(feature_row.generation_ms),
                        "energy_j": float(feature_row.energy_j),
                        "receiver_latency_ms": float(feature_row.receiver_latency_ms),
                        "receiver_energy_j": float(feature_row.receiver_energy_j),
                        "calibrated_correctness_probability": np.nan,
                        "payload_blob": feature_row.payload_blob,
                    })
        else:
            for bits in (8, 4):
                quantized = quantize_tensor(feature, bits)
                feature_payload = serialize_quantized_tensor(quantized)
                action = packet["recommended_action"]
                rows.append({
                    "sample_id": item.sample_id, "visual_group_id": item.visual_group_id,
                    "mode": PayloadMode.L4_FEATURES.name, "variant": f"fixture_int{bits}",
                    "source_model": "GENERATED_FIXTURE", "payload_bytes": len(feature_payload),
                    "wire_bytes": representation_wire_bytes(feature_payload, mtu_bytes),
                    "receiver_action": action, "native_receiver_correct": action == item.gold_action,
                    "generation_ms": 0.7, "energy_j": 0.003,
                    "receiver_latency_ms": 2.0, "receiver_energy_j": 0.002,
                    "calibrated_correctness_probability": np.nan,
                    "payload_blob": feature_payload.hex(),
                })
        for codec in image_codecs:
            for quality in image_qualities:
                for scale in image_scales:
                    try:
                        payload = encode_image(image, codec, quality, scale)
                    except StudyGateError:
                        if scientific:
                            raise
                        continue
                    # In the fixture the native image receiver mirrors gold with a
                    # deterministic degradation; scientific mode must provide its cache.
                    succeeds = _stable_uniform(item.sample_id, codec, quality, scale) < (0.55 + 0.004 * quality + 0.08 * scale)
                    receiver_action = item.gold_action if succeeds else "REVIEW"
                    variant = f"{codec.lower()}_q{quality}_s{scale}"
                    add_candidate(
                        sample_id=item.sample_id, visual_group_id=item.visual_group_id,
                        mode=PayloadMode.L5_IMAGE_OR_CLIP, variant=variant, payload=payload,
                        source_model="ground_visual_receiver", fixture_action=receiver_action,
                        fixture_correct=receiver_action == item.gold_action, gold_action=item.gold_action,
                        fixture_generation_ms=1.5 + 2 * scale,
                        fixture_energy_j=0.006 + 0.004 * scale,
                        fixture_receiver_latency_ms=12.0,
                        fixture_receiver_energy_j=0.02,
                    )
        lossless = io.BytesIO()
        image.save(lossless, format="PNG")
        payload = lossless.getvalue()
        add_candidate(
            sample_id=item.sample_id, visual_group_id=item.visual_group_id,
            mode=PayloadMode.L5_IMAGE_OR_CLIP, variant="lossless_png_reference",
            payload=payload, source_model="ground_visual_receiver",
            fixture_action=item.gold_action, fixture_correct=True, gold_action=item.gold_action,
            fixture_generation_ms=4.0, fixture_energy_j=0.012,
            fixture_receiver_latency_ms=12.0, fixture_receiver_energy_j=0.02,
        )
        clip_paths = json.loads(getattr(item, "clip_paths_json", "[]"))
        if len(clip_paths) >= 2:
            clip_images = [Image.open(path).convert("RGB") for path in clip_paths]
            clip_reference_payload_bytes = 0
            clip_reference_wire_bytes = 0
            for clip_image in clip_images:
                clip_lossless = io.BytesIO()
                clip_image.save(clip_lossless, format="PNG")
                clip_reference_payload = clip_lossless.getvalue()
                clip_reference_payload_bytes += len(clip_reference_payload)
                clip_reference_wire_bytes += representation_wire_bytes(
                    clip_reference_payload, mtu_bytes,
                )
            for crf in video_crfs:
                video_payload = encode_video_clip(
                    clip_images, "libx264", crf, video_fps,
                )
                variant = f"h264_crf{crf}_fps{video_fps}"
                # Fixture-only correctness degrades monotonically with CRF;
                # the scientific path obtains its outcome from the exact
                # checksum-bound native-receiver variant instead.
                fixture_success_probability = float(np.clip(
                    0.92 - 0.10 * ((crf - min(video_crfs)) / max(1, max(video_crfs) - min(video_crfs))),
                    0.0, 1.0,
                ))
                succeeds = _stable_uniform(item.sample_id, variant) < fixture_success_probability
                fixture_action = item.gold_action if succeeds else "REVIEW"
                add_candidate(
                    sample_id=item.sample_id, visual_group_id=item.visual_group_id,
                    mode=PayloadMode.L5_IMAGE_OR_CLIP, variant=variant,
                    payload=video_payload, source_model="ground_visual_receiver",
                    fixture_action=fixture_action,
                    fixture_correct=fixture_action == item.gold_action, gold_action=item.gold_action,
                    fixture_generation_ms=8.0, fixture_energy_j=0.02,
                    fixture_receiver_latency_ms=18.0,
                    fixture_receiver_energy_j=0.035,
                )
                rows[-1]["lossless_reference_payload_bytes"] = clip_reference_payload_bytes
                rows[-1]["lossless_reference_wire_bytes"] = clip_reference_wire_bytes
                rows[-1]["lossless_reference_basis"] = "sum_of_clip_frame_png_packets"
    frame = pd.DataFrame(rows)
    single_frame_reference = (
        frame.loc[
            frame["variant"].eq("lossless_png_reference"),
            ["sample_id", "payload_bytes", "wire_bytes"],
        ]
        .drop_duplicates("sample_id")
        .set_index("sample_id")
    )
    if single_frame_reference.empty:
        raise StudyGateError("Experiment 2 did not produce its registered lossless PNG reference")
    if "lossless_reference_payload_bytes" not in frame:
        frame["lossless_reference_payload_bytes"] = np.nan
    if "lossless_reference_wire_bytes" not in frame:
        frame["lossless_reference_wire_bytes"] = np.nan
    if "lossless_reference_basis" not in frame:
        frame["lossless_reference_basis"] = None
    frame["lossless_reference_payload_bytes"] = frame[
        "lossless_reference_payload_bytes"
    ].fillna(frame["sample_id"].map(single_frame_reference["payload_bytes"]))
    frame["lossless_reference_wire_bytes"] = frame[
        "lossless_reference_wire_bytes"
    ].fillna(frame["sample_id"].map(single_frame_reference["wire_bytes"]))
    frame["lossless_reference_basis"] = frame["lossless_reference_basis"].fillna(
        "single_frame_png_packet"
    )
    if frame[["lossless_reference_payload_bytes", "lossless_reference_wire_bytes"]].isna().any().any():
        raise StudyGateError("Experiment 2 could not resolve a lossless reference for every candidate")
    frame["lossless_wire_compression_ratio"] = (
        frame["lossless_reference_wire_bytes"] / frame["wire_bytes"]
    )
    frame["lossless_wire_reduction_fraction"] = 1.0 - (
        frame["wire_bytes"] / frame["lossless_reference_wire_bytes"]
    )
    if scientific and frame["variant"].str.contains("fixture", na=False).any():
        raise StudyGateError("Synthetic feature candidates are forbidden in full mode")
    if scientific:
        candidate_native_keys = set(
            frame.loc[
                ~frame["mode"].eq(PayloadMode.L4_FEATURES.name),
                ["sample_id", "variant", "source_model"],
            ].astype(str).itertuples(index=False, name=None)
        )
        cache_native_keys = set(
            native_receiver_cache[
                ["sample_id", "variant", "source_model"]
            ].astype(str).itertuples(index=False, name=None)
        )
        if candidate_native_keys != cache_native_keys:
            missing = sorted(candidate_native_keys - cache_native_keys)
            extra = sorted(cache_native_keys - candidate_native_keys)
            raise StudyGateError(
                "Experiment 2 candidate/native-receiver variant coverage differs: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
    return frame


def experiment2_representation_frontier_sharded(
    manifest: pd.DataFrame,
    ctx: RunContext,
    *,
    scientific: bool,
    mtu_bytes: int = 1200,
    semantic_packet_cache: pd.DataFrame | None = None,
    native_receiver_cache: pd.DataFrame | None = None,
    registered_feature_cache: pd.DataFrame | None = None,
    semantic_source_model: str = "deployed_semantic_model",
    image_codecs: Sequence[str] = ("JPEG", "WEBP", "AVIF"),
    image_qualities: Sequence[int] = (25, 50, 80),
    image_scales: Sequence[float] = (0.5, 1.0),
    video_crfs: Sequence[int] = (23, 35),
    video_fps: int = 5,
    shard_size: int = 16,
    stage_prefix: str = "02_representations",
) -> pd.DataFrame:
    shards = []
    for offset in range(0, len(manifest), shard_size):
        manifest_shard = manifest.iloc[offset : offset + shard_size].copy()
        sample_ids = set(manifest_shard["sample_id"].astype(str))
        semantic_shard = semantic_packet_cache[semantic_packet_cache["sample_id"].astype(str).isin(sample_ids)].copy() if semantic_packet_cache is not None else None
        receiver_shard = native_receiver_cache[native_receiver_cache["sample_id"].astype(str).isin(sample_ids)].copy() if native_receiver_cache is not None else None
        feature_shard = registered_feature_cache[registered_feature_cache["sample_id"].astype(str).isin(sample_ids)].copy() if registered_feature_cache is not None else None
        key = {
            "manifest_inputs": [
                {
                    "sample_id": str(row.sample_id),
                    "image_sha256": hash_file(row.image_path),
                    "clip_sha256": [
                        hash_file(path) for path in (
                            json.loads(row.clip_paths_json)
                            if isinstance(getattr(row, "clip_paths_json", None), str) and row.clip_paths_json
                            else []
                        )
                    ],
                    "mission": str(row.mission),
                    "capture_time_us": int(row.capture_time_us),
                    "telemetry_json": str(row.telemetry_json),
                }
                for row in manifest_shard.itertuples(index=False)
            ],
            "mtu_bytes": mtu_bytes,
            "semantic_source_model": semantic_source_model,
            "image_codecs": list(image_codecs),
            "image_qualities": list(image_qualities),
            "image_scales": list(image_scales),
            "video_crfs": list(map(int, video_crfs)),
            "video_fps": int(video_fps),
            "semantic_hash": hash_object(semantic_shard.to_dict("records")) if semantic_shard is not None else None,
            "receiver_hash": hash_object(receiver_shard.to_dict("records")) if receiver_shard is not None else None,
            "feature_hash": hash_object(feature_shard.to_dict("records")) if feature_shard is not None else None,
        }
        shards.append(frame_stage(
            ctx, f"{stage_prefix}_shard_{offset // shard_size:05d}", key,
            lambda manifest_shard=manifest_shard, semantic_shard=semantic_shard, receiver_shard=receiver_shard, feature_shard=feature_shard: experiment2_representation_frontier(
                manifest_shard, scientific=scientific, mtu_bytes=mtu_bytes,
                semantic_packet_cache=semantic_shard, native_receiver_cache=receiver_shard,
                registered_feature_cache=feature_shard,
                semantic_source_model=semantic_source_model,
                image_codecs=image_codecs,
                image_qualities=image_qualities,
                image_scales=image_scales,
                video_crfs=video_crfs,
                video_fps=video_fps,
            ),
        ))
    return pd.concat(shards, ignore_index=True) if shards else pd.DataFrame()


_STATIC_SEMANTIC_MODES = {
    PayloadMode.L1_ALERT.name,
    PayloadMode.L2_SEMANTICS.name,
    PayloadMode.L3_SEMANTICS_ROI.name,
}
_STATIC_NON_PERCEPTION_SOURCES = {
    "generated_fixture",
    "ground_visual_receiver",
    "registered_feature_receiver",
}
_STATIC_PERCEPTION_COST_COLUMNS = (
    "detector_latency_ms",
    "detector_energy_j",
    "context_model_latency_ms",
    "context_model_energy_j",
)


_EXPERIMENT3_SELECTION_VERSION = "semuav.exp3.static-selection.v1"
_EXPERIMENT3_SELECTION_COLUMNS = (
    "experiment3_selection_seed",
    "experiment3_selection_group_column",
    "experiment3_selection_stratum",
    "experiment3_selection_group_hash",
    "experiment3_selection_group_rank",
    "experiment3_selection_observation_rank",
    "experiment3_selection_original_group_rows",
    "experiment3_selection_original_source_observations",
)

_EXPERIMENT3_VARIANT_SPEC_COLUMNS = (
    "mode", "source_model", "variant", "validation_median_wire_bytes",
    "selection_rank", "selection_version",
)


def select_experiment3_candidate_variants(
    candidate_frame: pd.DataFrame,
    max_variants_per_mode_source: int,
    *,
    frozen_spec: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a bounded, wire-size-spanning representation set.

    Validation creates the spec using only representation identity and measured
    wire size; test applies that frozen spec verbatim.  Correctness, labels,
    latency, energy, and channel outcomes never enter selection.  Every chosen
    variant must cover every sample in the supplied static manifest slice.
    """

    if isinstance(max_variants_per_mode_source, (bool, np.bool_)):
        raise ValueError("max_variants_per_mode_source must be a positive integer")
    try:
        limit = int(max_variants_per_mode_source)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_variants_per_mode_source must be a positive integer") from exc
    if limit != max_variants_per_mode_source or limit <= 0:
        raise ValueError("max_variants_per_mode_source must be a positive integer")
    required = {"sample_id", "mode", "variant", "source_model", "wire_bytes"}
    if missing := required.difference(candidate_frame.columns):
        raise StudyGateError(
            f"Experiment 3 variant selection is missing columns: {sorted(missing)}"
        )
    if candidate_frame.empty:
        raise StudyGateError("Experiment 3 variant selection requires candidates")
    candidates = candidate_frame.copy()
    for column in ("sample_id", "mode", "variant", "source_model"):
        candidates[column] = candidates[column].fillna("").astype(str).str.strip()
    if candidates[list(("sample_id", "mode", "variant", "source_model"))].eq("").any().any():
        raise StudyGateError("Experiment 3 candidate identities must be nonblank")
    if candidates.duplicated(["sample_id", "mode", "variant", "source_model"]).any():
        raise StudyGateError("Experiment 3 candidates contain duplicate sample/variant rows")
    expected_samples = set(candidates["sample_id"])
    coverage = candidates.groupby(
        ["mode", "source_model", "variant"], sort=True, dropna=False,
    )["sample_id"].nunique()
    complete_keys = {
        tuple(map(str, key))
        for key, count in coverage.items()
        if int(count) == len(expected_samples)
    }
    if not complete_keys:
        raise StudyGateError("No Experiment 3 representation variant covers every selected sample")

    if frozen_spec is None:
        medians = (
            candidates.groupby(
                ["mode", "source_model", "variant"], sort=True, dropna=False,
            )["wire_bytes"]
            .median()
            .rename("validation_median_wire_bytes")
            .reset_index()
        )
        medians = medians.loc[
            medians.apply(
                lambda row: (str(row["mode"]), str(row["source_model"]), str(row["variant"]))
                in complete_keys,
                axis=1,
            )
        ]
        selected_rows: list[pd.DataFrame] = []
        for _, group in medians.groupby(["mode", "source_model"], sort=True, dropna=False):
            ordered = group.sort_values(
                ["validation_median_wire_bytes", "variant"], kind="mergesort",
            ).reset_index(drop=True)
            count = min(limit, len(ordered))
            if count == len(ordered):
                chosen_indices = list(range(len(ordered)))
            elif count == 1:
                chosen_indices = [0]
            else:
                chosen_indices = sorted({
                    int(round(index * (len(ordered) - 1) / (count - 1)))
                    for index in range(count)
                })
            chosen = ordered.iloc[chosen_indices].copy()
            chosen["selection_rank"] = range(len(chosen))
            selected_rows.append(chosen)
        spec = pd.concat(selected_rows, ignore_index=True)
        spec["selection_version"] = "semuav.exp3.variant-wire-span.v1"
        spec = spec.loc[:, _EXPERIMENT3_VARIANT_SPEC_COLUMNS].sort_values(
            ["mode", "source_model", "selection_rank", "variant"], kind="mergesort",
        ).reset_index(drop=True)
    else:
        if missing := set(_EXPERIMENT3_VARIANT_SPEC_COLUMNS).difference(frozen_spec.columns):
            raise StudyGateError(
                f"Frozen Experiment 3 variant spec is missing columns: {sorted(missing)}"
            )
        spec = frozen_spec.loc[:, _EXPERIMENT3_VARIANT_SPEC_COLUMNS].copy()
        if spec.empty or spec.duplicated(["mode", "source_model", "variant"]).any():
            raise StudyGateError("Frozen Experiment 3 variant spec is empty or duplicated")
        versions = set(spec["selection_version"].astype(str))
        if versions != {"semuav.exp3.variant-wire-span.v1"}:
            raise StudyGateError(f"Frozen Experiment 3 variant spec version is invalid: {versions}")
        requested = {
            (str(row.mode), str(row.source_model), str(row.variant))
            for row in spec.itertuples(index=False)
        }
        missing_keys = sorted(requested - complete_keys)
        if missing_keys:
            raise StudyGateError(
                "Frozen Experiment 3 variants are absent or incomplete in this phase; "
                f"first={missing_keys[0]}"
            )

    selected_keys = {
        (str(row.mode), str(row.source_model), str(row.variant))
        for row in spec.itertuples(index=False)
    }
    selected_mask = candidates.apply(
        lambda row: (str(row["mode"]), str(row["source_model"]), str(row["variant"]))
        in selected_keys,
        axis=1,
    )
    selected = candidates.loc[selected_mask].copy()
    if selected.empty:
        raise StudyGateError("Experiment 3 frozen variant selection produced no candidates")
    return selected.reset_index(drop=True), spec.reset_index(drop=True)


def select_experiment3_manifest(
    manifest: pd.DataFrame,
    max_groups_per_dataset: int,
    seed: int,
    max_observations_per_group: int | None = None,
) -> pd.DataFrame:
    """Select a deterministic, outcome-blind static-characterization subset.

    Whole visual independence groups are ranked within each dataset using only
    a stable hash of the registered seed, dataset, and group identity.  When a
    mission column is available, selection proceeds in balanced mission rounds
    before the next group from a mission can be selected.  No annotation,
    correctness, risk, payload, latency, or other outcome column is inspected.

    By default every observation in a selected group is retained.  An optional
    observation cap keeps evenly spaced source frames from the canonical
    temporal ordering (frame index, timestamp, trace/capture time, then source
    ID). If the manifest contains profile-expanded ``source_sample_id`` rows,
    every registered mission-profile copy of each retained source frame is
    kept. It never rewrites the original timestamps or deadlines.
    """
    if isinstance(max_groups_per_dataset, (bool, np.bool_)):
        raise ValueError("max_groups_per_dataset must be a positive integer")
    try:
        group_limit = int(max_groups_per_dataset)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_groups_per_dataset must be a positive integer") from exc
    if group_limit != max_groups_per_dataset or group_limit <= 0:
        raise ValueError("max_groups_per_dataset must be a positive integer")
    if max_observations_per_group is not None:
        if isinstance(max_observations_per_group, (bool, np.bool_)):
            raise ValueError(
                "max_observations_per_group must be a positive integer or None"
            )
        try:
            observation_limit = int(max_observations_per_group)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "max_observations_per_group must be a positive integer or None"
            ) from exc
        if observation_limit != max_observations_per_group or observation_limit <= 0:
            raise ValueError(
                "max_observations_per_group must be a positive integer or None"
            )
    else:
        observation_limit = None
    if not isinstance(manifest, pd.DataFrame) or manifest.empty:
        raise StudyGateError("Experiment 3 selection requires a nonempty manifest")
    required = {"sample_id", "dataset"}
    if missing := required.difference(manifest.columns):
        raise StudyGateError(
            f"Experiment 3 selection manifest is missing columns: {sorted(missing)}"
        )
    if manifest["sample_id"].astype(str).duplicated().any():
        raise StudyGateError("Experiment 3 selection manifest has duplicate sample_id rows")

    def _normalized_identity(column: str) -> pd.Series:
        return manifest[column].fillna("").astype(str).str.strip()

    dataset_identity = _normalized_identity("dataset")
    if dataset_identity.eq("").any():
        raise StudyGateError("Experiment 3 selection requires nonblank dataset values")
    complete_group_columns = [
        column for column in ("visual_group_id", "split_unit")
        if column in manifest.columns and _normalized_identity(column).ne("").all()
    ]
    if not complete_group_columns:
        raise StudyGateError(
            "Experiment 3 selection requires complete visual_group_id or split_unit values"
        )
    group_column = complete_group_columns[0]
    group_identity = _normalized_identity(group_column)
    if len(complete_group_columns) == 2:
        alternate = _normalized_identity("split_unit")
        pairs = pd.DataFrame({
            "visual_group_id": _normalized_identity("visual_group_id"),
            "split_unit": alternate,
        }).drop_duplicates()
        if (
            pairs.groupby("visual_group_id")["split_unit"].nunique().gt(1).any()
            or pairs.groupby("split_unit")["visual_group_id"].nunique().gt(1).any()
        ):
            raise StudyGateError(
                "Experiment 3 visual_group_id and split_unit do not identify "
                "the same independence groups"
            )

    mission_available = "mission" in manifest.columns
    mission_identity = (
        _normalized_identity("mission")
        if mission_available else pd.Series("", index=manifest.index, dtype=str)
    )
    work = manifest.drop(
        columns=[column for column in _EXPERIMENT3_SELECTION_COLUMNS if column in manifest],
    ).copy()
    work["__exp3_dataset"] = dataset_identity.to_numpy()
    work["__exp3_group"] = group_identity.to_numpy()
    work["__exp3_mission"] = mission_identity.to_numpy()
    natural_observation_column = (
        "source_sample_id" if "source_sample_id" in work.columns else "sample_id"
    )
    if natural_observation_column == "source_sample_id":
        source_identity = work["source_sample_id"].fillna("").astype(str).str.strip()
        if source_identity.eq("").any():
            raise StudyGateError(
                "Experiment 3 profile-expanded rows require nonblank source_sample_id"
            )
        work["__exp3_source_observation"] = source_identity.to_numpy()
        if "mission_profile" not in work.columns:
            raise StudyGateError(
                "Experiment 3 profile-expanded rows require mission_profile"
            )
        registered_profiles = tuple(sorted(set(
            work["mission_profile"].fillna("").astype(str).str.strip()
        )))
        if not registered_profiles or registered_profiles[0] == "":
            raise StudyGateError("Experiment 3 mission profiles are incomplete")
        observed_profile_sets = work.groupby(
            "__exp3_source_observation", sort=False,
        )["mission_profile"].agg(lambda values: tuple(sorted(set(map(str, values)))))
        if not observed_profile_sets.map(lambda value: value == registered_profiles).all():
            raise StudyGateError(
                "Experiment 3 source frames do not retain complete mission-profile pairs"
            )
    else:
        work["__exp3_source_observation"] = work["sample_id"].astype(str)

    group_rows: list[dict[str, Any]] = []
    for (dataset, group_id), group in work.groupby(
        ["__exp3_dataset", "__exp3_group"], sort=True, dropna=False,
    ):
        missions = tuple(sorted(set(group["__exp3_mission"].astype(str))))
        if len(missions) != 1:
            raise StudyGateError(
                f"Experiment 3 group {group_id!r} spans multiple mission strata"
            )
        group_rows.append({
            "dataset": str(dataset),
            "group_id": str(group_id),
            "mission": missions[0],
            "group_hash": hash_object({
                "selection_version": _EXPERIMENT3_SELECTION_VERSION,
                "seed": int(seed),
                "dataset": str(dataset),
                "group": str(group_id),
            }),
        })
    groups = pd.DataFrame(group_rows)
    groups = groups.sort_values(
        ["dataset", "mission", "group_hash", "group_id"], kind="mergesort",
    ).reset_index(drop=True)
    # Round-robin mission strata provide balanced mission representation while
    # the stable group hashes remain the sole within-stratum selection score.
    groups["mission_round"] = groups.groupby(
        ["dataset", "mission"], sort=False,
    ).cumcount()
    groups = groups.sort_values(
        ["dataset", "mission_round", "group_hash", "group_id"], kind="mergesort",
    )
    selected_groups = groups.groupby("dataset", sort=True).head(group_limit).copy()
    selected_groups["group_rank"] = selected_groups.groupby(
        "dataset", sort=False,
    ).cumcount()

    selected_parts: list[pd.DataFrame] = []
    temporal_sources = (
        "frame_index", "timestamp_ms", "trace_time_us", "capture_time_us",
    )
    for selection in selected_groups.itertuples(index=False):
        group = work[
            work["__exp3_dataset"].eq(selection.dataset)
            & work["__exp3_group"].eq(selection.group_id)
        ].copy()
        temporal_sort_columns: list[str] = []
        for column in temporal_sources:
            if column not in group.columns:
                continue
            internal = f"__exp3_time_{column}"
            group[internal] = pd.to_numeric(group[column], errors="coerce")
            if group[internal].notna().any():
                temporal_sort_columns.append(internal)
        group["__exp3_sample_id"] = group["sample_id"].astype(str)
        group = group.sort_values(
            [*temporal_sort_columns, "__exp3_source_observation", "__exp3_sample_id"],
            kind="mergesort", na_position="last",
        ).reset_index(drop=True)
        original_rows = len(group)
        natural = group.drop_duplicates("__exp3_source_observation", keep="first").copy()
        natural_observations = len(natural)
        natural["__exp3_natural_rank"] = np.arange(natural_observations, dtype=int)
        if observation_limit is not None and natural_observations > observation_limit:
            if observation_limit == 1:
                positions = np.array([(natural_observations - 1) // 2], dtype=int)
            else:
                positions = np.rint(np.linspace(
                    0, natural_observations - 1, observation_limit,
                )).astype(int)
            natural = natural.iloc[positions].copy()
        rank_lookup = dict(zip(
            natural["__exp3_source_observation"].astype(str),
            natural["__exp3_natural_rank"].astype(int),
        ))
        group = group[
            group["__exp3_source_observation"].astype(str).isin(rank_lookup)
        ].copy()
        group["experiment3_selection_observation_rank"] = (
            group["__exp3_source_observation"].astype(str).map(rank_lookup).astype(int)
        )
        group["experiment3_selection_seed"] = int(seed)
        group["experiment3_selection_group_column"] = group_column
        group["experiment3_selection_stratum"] = (
            f"{selection.dataset}|{selection.mission}"
            if mission_available else str(selection.dataset)
        )
        group["experiment3_selection_group_hash"] = selection.group_hash
        group["experiment3_selection_group_rank"] = int(selection.group_rank)
        group["experiment3_selection_original_group_rows"] = original_rows
        group["experiment3_selection_original_source_observations"] = natural_observations
        selected_parts.append(group)

    if not selected_parts:
        raise StudyGateError("Experiment 3 selection produced no groups")
    selected = pd.concat(selected_parts, ignore_index=True)
    internal_columns = [
        column for column in selected.columns if column.startswith("__exp3_")
    ]
    selected = selected.drop(columns=internal_columns)
    ordered_columns = [
        *[column for column in manifest.columns if column not in _EXPERIMENT3_SELECTION_COLUMNS],
        *_EXPERIMENT3_SELECTION_COLUMNS,
    ]
    selected = selected[ordered_columns].reset_index(drop=True)
    selected.attrs = dict(manifest.attrs)
    selected.attrs["experiment3_selection"] = {
        "version": _EXPERIMENT3_SELECTION_VERSION,
        "seed": int(seed),
        "group_column": group_column,
        "max_groups_per_dataset": group_limit,
        "max_observations_per_group": observation_limit,
        "observation_unit": natural_observation_column,
        "dataset_groups": selected.groupby("dataset")[group_column].nunique().to_dict(),
    }
    return selected


def _static_perception_costs(candidate: Any, require_perception_costs: bool) -> dict[str, Any]:
    """Return causal sender compute charged before a static transmission.

    Experiment-2 sender costs contain representation encoding/serialization
    only.  Semantic candidates therefore add the measured detector cost once;
    a VLM-sourced semantic candidate also adds its measured context-model cost
    once.  Image, clip, heartbeat, and registered-feature candidates do not run
    either perception stage for this representation comparison.
    """
    mode = str(getattr(candidate, "mode", ""))
    source_model = str(getattr(candidate, "source_model", ""))
    source_key = source_model.strip().lower()
    if mode not in _STATIC_SEMANTIC_MODES or source_key in _STATIC_NON_PERCEPTION_SOURCES:
        return {
            "perception_pipeline": "none",
            "detector_latency_ms": 0.0,
            "detector_energy_j": 0.0,
            "context_model_latency_ms": 0.0,
            "context_model_energy_j": 0.0,
        }
    if not source_key:
        if require_perception_costs:
            raise StudyGateError("A scientific semantic candidate is missing source_model provenance")
        return {
            "perception_pipeline": "none",
            "detector_latency_ms": 0.0,
            "detector_energy_j": 0.0,
            "context_model_latency_ms": 0.0,
            "context_model_energy_j": 0.0,
        }

    detector_only = source_key == "detector_only"
    required_columns = (
        ("detector_latency_ms", "detector_energy_j")
        if detector_only else _STATIC_PERCEPTION_COST_COLUMNS
    )
    costs: dict[str, float] = {column: 0.0 for column in _STATIC_PERCEPTION_COST_COLUMNS}
    for column in required_columns:
        raw_value = getattr(candidate, column, None)
        if raw_value is None:
            if require_perception_costs:
                raise StudyGateError(
                    f"Static semantic candidate {source_model!r} is missing measured {column}"
                )
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise StudyGateError(f"Static perception cost {column} is not numeric") from exc
        if not np.isfinite(value) or value < 0:
            raise StudyGateError(f"Static perception cost {column} must be finite and nonnegative")
        costs[column] = value
    return {
        "perception_pipeline": "detector_only" if detector_only else "detector_plus_vlm",
        **costs,
    }


def add_experiment2_total_costs(
    candidate_frame: pd.DataFrame,
    perception_manifest: pd.DataFrame,
    *,
    require_measured_costs: bool,
) -> pd.DataFrame:
    """Attach the proposal's end-to-end compute cost axes to Experiment 2.

    The representation table stores sender encoding and native receiver costs.
    This function adds the causal detector/context-model stages for semantic
    candidates exactly once. Network queueing, transport, retries, and radio
    energy remain Experiment 3 axes and are explicitly excluded here.
    """
    required_candidates = {
        "sample_id", "mode", "source_model", "generation_ms", "energy_j",
        "receiver_latency_ms", "receiver_energy_j",
    }
    if missing := required_candidates.difference(candidate_frame.columns):
        raise StudyGateError(
            f"Experiment 2 total-cost input is missing columns: {sorted(missing)}"
        )
    manifest_columns = {"sample_id", *_STATIC_PERCEPTION_COST_COLUMNS}
    if missing := manifest_columns.difference(perception_manifest.columns):
        raise StudyGateError(
            f"Experiment 2 perception-cost manifest is missing columns: {sorted(missing)}"
        )
    if perception_manifest["sample_id"].astype(str).duplicated().any():
        raise StudyGateError("Experiment 2 perception-cost manifest has duplicate sample_id rows")
    result = candidate_frame.copy()
    result["sample_id"] = result["sample_id"].astype(str)
    costs = perception_manifest[["sample_id", *_STATIC_PERCEPTION_COST_COLUMNS]].copy()
    costs["sample_id"] = costs["sample_id"].astype(str)
    result = result.merge(costs, on="sample_id", how="left", validate="many_to_one")
    if require_measured_costs and result[list(_STATIC_PERCEPTION_COST_COLUMNS)].isna().any().any():
        raise StudyGateError("Experiment 2 lacks measured perception cost for one or more samples")
    perception_rows = [
        _static_perception_costs(row, require_measured_costs)
        for row in result.itertuples(index=False)
    ]
    perception = pd.DataFrame(perception_rows, index=result.index)
    for column in _STATIC_PERCEPTION_COST_COLUMNS:
        result[column] = pd.to_numeric(perception[column], errors="raise")
    result["perception_pipeline"] = perception["perception_pipeline"].astype(str)
    for column in ("generation_ms", "energy_j", "receiver_latency_ms", "receiver_energy_j"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    required_numeric = [
        "generation_ms", "energy_j", "receiver_latency_ms", "receiver_energy_j",
        *_STATIC_PERCEPTION_COST_COLUMNS,
    ]
    if require_measured_costs and (
        result[required_numeric].isna().any().any()
        or (~np.isfinite(result[required_numeric].to_numpy(float))).any()
        or (result[required_numeric] < 0).any().any()
    ):
        raise StudyGateError("Experiment 2 total costs must be complete, finite, and nonnegative")
    result["onboard_perception_latency_ms"] = (
        result["detector_latency_ms"] + result["context_model_latency_ms"]
    )
    result["onboard_perception_energy_j"] = (
        result["detector_energy_j"] + result["context_model_energy_j"]
    )
    result["total_processing_latency_ms"] = (
        result["onboard_perception_latency_ms"]
        + result["generation_ms"]
        + result["receiver_latency_ms"]
    )
    result["total_compute_energy_j"] = (
        result["onboard_perception_energy_j"]
        + result["energy_j"]
        + result["receiver_energy_j"]
    )
    result["total_cost_scope"] = (
        "capture_to_native_receiver_decision_compute_only_excludes_network_and_radio"
    )
    return result


# v3r2 correction (receipted in v3r1 as v3r1c): the registered level variants
# ``l0_cbor``, ``l1_cbor`` and ``l2_cbor`` are CBOR on the wire.  The v3r1 prefix
# rule resolved them to ``None`` so the receiver parsed CBOR as JSON and every
# L0-L2 message failed with ``decode_error`` in Experiments 3 and 4.
REGISTERED_CBOR_LEVEL_VARIANTS = frozenset({"l0_cbor", "l1_cbor", "l2_cbor"})


def registered_wire_format(variant: Any) -> str | None:
    """Return the wire format of a semantic variant, or ``None`` for media."""
    token = str(variant).split(":", 1)[0].split("+", 1)[0].split("__", 1)[0]
    if token in {"cbor", "json-debug"}:
        return token
    if token in REGISTERED_CBOR_LEVEL_VARIANTS:
        return "cbor"
    return None


def experiment3_static_transport(
    candidate_frame: pd.DataFrame,
    manifest: pd.DataFrame,
    seed: int = 101,
    *,
    rates_mbps: Sequence[float] = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
    rtts_ms: Sequence[float] = (20.0, 80.0, 200.0),
    losses: Sequence[float] = (0.0, 0.01, 0.05, 0.15),
    loss_models: Sequence[str] = EXPERIMENT3_REGISTERED_LOSS_MODELS,
    burst_recovery_probability: float = 0.2,
    burst_slot_ms: float = 100.0,
    jitters_ms: Sequence[float] = (0.0, 10.0),
    queue_kb: Sequence[int] = (32, 128, 512),
    radio_tx_j_per_byte: float = 0.0,
    radio_rx_j_per_byte: float = 0.0,
    require_perception_costs: bool = False,
    require_registered_loss_models: bool = False,
) -> pd.DataFrame:
    """Replay each fixed representation as a sustained observation stream.

    A stream is identified by ``(mode, variant, source_model)``.  All of its
    observations share a two-stage FCFS sender flow shop (perception followed
    by representation encoding) and one packet-level simulator for each link
    condition, so compute backlog, pipelining, transport backlog, retries, and
    finite-queue admission interact across captures. Absolute dataset timestamps
    are mapped onto ``trace_time_us``; the original
    ``deadline_us - capture_time_us`` budget is preserved.
    """
    if radio_tx_j_per_byte < 0 or radio_rx_j_per_byte < 0:
        raise ValueError("Radio-energy coefficients must be nonnegative")
    normalized_loss_models = tuple(str(model).strip().lower() for model in loss_models)
    if not normalized_loss_models or len(set(normalized_loss_models)) != len(normalized_loss_models):
        raise ValueError("loss_models must be a nonempty sequence without duplicates")
    unknown_loss_models = set(normalized_loss_models) - set(EXPERIMENT3_REGISTERED_LOSS_MODELS)
    if unknown_loss_models:
        raise ValueError(f"Unknown static loss model(s): {sorted(unknown_loss_models)}")
    if require_registered_loss_models and not set(EXPERIMENT3_REGISTERED_LOSS_MODELS).issubset(
        normalized_loss_models
    ):
        raise StudyGateError(
            "Scientific Experiment 3 requires both registered IID and Gilbert-Elliott loss models"
        )
    if not 0 < burst_recovery_probability <= 1:
        raise ValueError("burst_recovery_probability must lie in (0, 1]")
    if not np.isfinite(burst_slot_ms) or burst_slot_ms <= 0:
        raise ValueError("burst_slot_ms must be finite and positive")
    burst_slot_us = int(round(float(burst_slot_ms) * 1000))
    if burst_slot_us <= 0:
        raise ValueError("burst_slot_ms is below the simulator's one-microsecond resolution")
    for configured_loss in losses:
        loss_value = float(configured_loss)
        if not 0 <= loss_value <= 1:
            raise ValueError("Static loss probabilities must lie in [0, 1]")
        if (
            "gilbert_elliott" in normalized_loss_models
            and 0 < loss_value < 1
            and burst_recovery_probability * loss_value / (1 - loss_value) > 1
        ):
            raise ValueError(
                "The configured Gilbert-Elliott recovery probability cannot realize "
                f"stationary loss={loss_value}; reduce burst_recovery_probability"
            )
    required_candidate_columns = {
        "sample_id", "mode", "variant", "source_model", "payload_blob",
        "generation_ms", "energy_j", "receiver_latency_ms", "receiver_energy_j",
        "native_receiver_correct",
    }
    if missing := required_candidate_columns.difference(candidate_frame.columns):
        raise StudyGateError(f"Static candidate table is missing columns: {sorted(missing)}")
    base_manifest_columns = [
        "sample_id", "capture_time_us", "trace_time_us", "deadline_us",
    ]
    if missing := set(base_manifest_columns).difference(manifest.columns):
        raise StudyGateError(f"Static manifest is missing columns: {sorted(missing)}")
    if require_perception_costs:
        if missing := set(_STATIC_PERCEPTION_COST_COLUMNS).difference(manifest.columns):
            raise StudyGateError(
                f"Scientific static transport is missing measured perception costs: {sorted(missing)}"
            )
    manifest_columns = [
        *base_manifest_columns,
        *[
            column for column in ("visual_group_id", "mission_profile")
            if column in manifest.columns
        ],
        *[column for column in _STATIC_PERCEPTION_COST_COLUMNS if column in manifest.columns],
    ]
    if manifest["sample_id"].astype(str).duplicated().any():
        raise StudyGateError("Static manifest has duplicate sample_id rows")
    timing = manifest[manifest_columns].copy()
    timing["sample_id"] = timing["sample_id"].astype(str)
    # Experiment 2 may already carry candidate-specific reporting columns for
    # total compute cost. Experiment 3 recomputes perception causally from the
    # authoritative manifest and candidate source, so drop those reporting
    # copies before the merge to avoid suffixes or double charging.
    candidate_input = candidate_frame.drop(
        columns=[
            column for column in _STATIC_PERCEPTION_COST_COLUMNS
            if column in candidate_frame.columns
        ],
    ).copy()
    candidate_input["sample_id"] = candidate_input["sample_id"].astype(str)
    if missing_ids := sorted(set(candidate_input["sample_id"]) - set(timing["sample_id"])):
        raise StudyGateError(
            f"Static candidates lack manifest timing for {len(missing_ids)} sample(s)"
        )
    candidates = candidate_input.merge(timing, on="sample_id", validate="many_to_one")
    candidates["deadline_budget_us"] = (
        candidates["deadline_us"].astype(np.int64)
        - candidates["capture_time_us"].astype(np.int64)
    )
    if (candidates["deadline_budget_us"] <= 0).any():
        raise StudyGateError("Static transport requires strictly positive original deadline budgets")
    stream_keys = ["mode", "variant", "source_model"]
    episode_keys = [
        column for column in ("visual_group_id", "mission_profile")
        if column in candidates.columns
    ]
    if set(episode_keys) != {"visual_group_id", "mission_profile"}:
        # Preserve the general Experiment-3 API for legacy/unit fixtures that
        # predate the paper's mission-profile expansion.  Their historical API
        # is one stream-wide FCFS episode.  The VisDrone paper runner
        # independently requires and audits both canonical fields before
        # scientific execution, so this compatibility path cannot weaken the
        # registered real-data design.
        episode_keys = []
    if candidates.duplicated([*stream_keys, "sample_id"]).any():
        raise StudyGateError("A static representation stream has duplicate observations")

    def _wire_format(variant: str) -> str | None:
        return registered_wire_format(variant)

    def _expired_generation_row(message: AppMessage, metadata: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "run_id": message.run_id,
            "observation_id": message.logical_observation_id,
            "message_id": message.message_id,
            "mode": message.mode.name,
            "capture_us": message.capture_us,
            "ready_us": message.ready_us,
            "deadline_us": message.action_deadline_us,
            "first_tx_us": None,
            "last_rx_us": None,
            "action_ready_us": None,
            "full_message_received": False,
            "timely_delivery": False,
            "apply_status": None,
            "retry_exhausted": False,
            "queue_overflow": False,
            "original_uplink_bytes": 0,
            "retry_uplink_bytes": 0,
            "downlink_control_bytes": 0,
            "total_wire_bytes": 0,
            "duplicate_payload_bytes": 0,
            "latency_ms": np.inf,
            "deadline_slack_ms": -np.inf,
            **metadata,
            "generation_deadline_drop": True,
        }

    def _add_stream_audits(
        frame: pd.DataFrame, events: pd.DataFrame, link_rate_mbps: float,
    ) -> pd.DataFrame:
        frame = frame.sort_values(["capture_us", "observation_id"], kind="mergesort").copy()
        frame["network_queue_wait_ms"] = np.where(
            frame["first_tx_us"].notna(),
            (pd.to_numeric(frame["first_tx_us"]) - frame["ready_us"]) / 1000,
            np.nan,
        )
        frame["application_valid"] = frame["apply_status"].isin(["applied", "duplicate"])
        valid_delivery = frame["full_message_received"].astype(bool) & frame["application_valid"]
        frame["correct_timely_action"] = (
            frame["correct_timely_action"].astype(bool) & frame["application_valid"]
        )
        frame["aoi_at_delivery_ms"] = np.where(
            valid_delivery,
            (pd.to_numeric(frame["action_ready_us"]) - frame["capture_us"]) / 1000,
            np.inf,
        )

        capture_times = np.sort(frame["capture_us"].astype(np.int64).unique())
        positive_intervals = np.diff(capture_times)
        positive_intervals = positive_intervals[positive_intervals > 0]
        if len(positive_intervals):
            nominal_interval_us = max(1, int(np.median(positive_intervals)))
        else:
            nominal_interval_us = max(
                1, int(np.median(frame["deadline_us"] - frame["capture_us"])),
            )
        stream_start_us = int(capture_times[0])
        offer_end_us = int(capture_times[-1]) + nominal_interval_us
        event_end_us = int(events["time_us"].max()) if len(events) else stream_start_us
        finite_action_times = pd.to_numeric(frame["action_ready_us"], errors="coerce").dropna()
        action_end_us = int(finite_action_times.max()) if len(finite_action_times) else stream_start_us
        drain_end_us = max(offer_end_us, event_end_us, action_end_us)
        offer_duration_s = max(1, offer_end_us - stream_start_us) / 1e6
        drain_duration_s = max(1, drain_end_us - stream_start_us) / 1e6
        offered_wire_bits = float(frame["offered_wire_bytes"].sum()) * 8
        delivered_payload_bits = float(
            frame.loc[valid_delivery, "payload_bytes"].sum()
        ) * 8
        delivered_wire_bits = float(
            frame.loc[valid_delivery, "offered_wire_bytes"].sum()
        ) * 8
        stream_offered_load_bps = offered_wire_bits / offer_duration_s
        frame["stream_offer_window_s"] = offer_duration_s
        frame["stream_drain_window_s"] = drain_duration_s
        frame["stream_offered_load_bps"] = stream_offered_load_bps
        frame["stream_offered_load_ratio"] = stream_offered_load_bps / (link_rate_mbps * 1e6)
        frame["stream_delivered_payload_goodput_bps"] = delivered_payload_bits / drain_duration_s
        frame["stream_delivered_wire_goodput_bps"] = delivered_wire_bits / drain_duration_s
        frame["stream_uplink_wire_rate_bps"] = float(
            (frame["original_uplink_bytes"] + frame["retry_uplink_bytes"]).sum()
        ) * 8 / drain_duration_s
        frame["stream_queue_drop_rate"] = float(frame["queue_overflow"].astype(bool).mean())
        frame["stream_generation_deadline_drop_rate"] = float(
            frame["generation_deadline_drop"].astype(bool).mean()
        )

        # Causal receiver age at each capture uses only valid deliveries that
        # arrived by that capture.  The conditional time-average AoI begins at
        # the first valid delivery; it is infinite when the stream never updates.
        deliveries = frame.loc[
            valid_delivery, ["capture_us", "action_ready_us"]
        ].sort_values(["action_ready_us", "capture_us"], kind="mergesort")
        if len(deliveries):
            delivery_arrivals = pd.to_numeric(
                deliveries["action_ready_us"], errors="raise",
            ).to_numpy(dtype=np.int64)
            freshest_delivered_capture = np.maximum.accumulate(
                deliveries["capture_us"].to_numpy(dtype=np.int64)
            )
            row_captures = frame["capture_us"].to_numpy(dtype=np.int64)
            visible_indices = np.searchsorted(
                delivery_arrivals, row_captures, side="right",
            ) - 1
            receiver_age_at_capture = np.full(len(frame), np.inf, dtype=float)
            visible_mask = visible_indices >= 0
            receiver_age_at_capture[visible_mask] = (
                row_captures[visible_mask]
                - freshest_delivered_capture[visible_indices[visible_mask]]
            ) / 1000
            frame["receiver_aoi_at_capture_ms"] = receiver_age_at_capture
        else:
            frame["receiver_aoi_at_capture_ms"] = np.inf
        if len(deliveries):
            grouped_deliveries = [
                (int(arrival), group["capture_us"].astype(np.int64).tolist())
                for arrival, group in deliveries.groupby("action_ready_us", sort=True)
            ]
            aoi_start_us = grouped_deliveries[0][0]
            clock_us = aoi_start_us
            freshest_capture_us = max(grouped_deliveries[0][1])
            area_us2 = 0.0
            for arrival_us, delivered_captures in grouped_deliveries[1:]:
                delta_us = arrival_us - clock_us
                age_us = clock_us - freshest_capture_us
                area_us2 += age_us * delta_us + 0.5 * delta_us * delta_us
                freshest_capture_us = max(freshest_capture_us, *delivered_captures)
                clock_us = arrival_us
            tail_us = max(0, drain_end_us - clock_us)
            area_us2 += (clock_us - freshest_capture_us) * tail_us + 0.5 * tail_us * tail_us
            observed_us = max(0, drain_end_us - aoi_start_us)
            mean_aoi_ms = (
                area_us2 / observed_us / 1000
                if observed_us else (aoi_start_us - freshest_capture_us) / 1000
            )
            freshest_age_at_end_ms = (drain_end_us - freshest_capture_us) / 1000
        else:
            aoi_start_us = np.nan
            mean_aoi_ms = np.inf
            freshest_age_at_end_ms = np.inf
        frame["stream_aoi_observation_start_us"] = aoi_start_us
        frame["stream_time_average_aoi_ms"] = mean_aoi_ms
        frame["stream_freshest_age_at_end_ms"] = freshest_age_at_end_ms
        return frame

    outputs: list[pd.DataFrame] = []
    for rate in rates_mbps:
        for rtt_ms in rtts_ms:
            for loss in losses:
                for loss_model in normalized_loss_models:
                    for jitter_ms, capacity_kb in (
                        (jitter, capacity)
                        for jitter in jitters_ms
                        for capacity in queue_kb
                    ):
                        trace_key = (
                            f"{rate}-{rtt_ms}-{loss}-{loss_model}-"
                            f"q{burst_recovery_probability}-slot{burst_slot_us}-"
                            f"{jitter_ms}-{capacity_kb}"
                        )
                        uplink = constant_trace(f"static-u-{trace_key}", Direction.UPLINK, rate, rtt_ms / 2, loss, jitter_ms=jitter_ms)
                        downlink = constant_trace(f"static-d-{trace_key}", Direction.DOWNLINK, max(rate, 0.5), rtt_ms / 2, loss / 2, jitter_ms=jitter_ms / 2)
                        for stream_values, stream in candidates.groupby(
                            [*stream_keys, *episode_keys], sort=True, dropna=False,
                        ):
                            values = tuple(stream_values) if isinstance(stream_values, tuple) else (stream_values,)
                            mode, variant, source_model = map(str, values[:3])
                            episode_values = tuple(map(str, values[3:]))
                            ordered = stream.sort_values(
                                ["trace_time_us", "capture_time_us", "sample_id"],
                                kind="mergesort",
                            )
                            stream_id = hashlib.sha256(
                                json.dumps(
                                    [mode, variant, source_model, *episode_values],
                                    separators=(",", ":"),
                                ).encode()
                            ).hexdigest()[:16]
                            perception_server_available_us = 0
                            representation_server_available_us = 0
                            captures = []
                            expired_rows = []
                            message_ids: set[int] = set()
                            for candidate in ordered.itertuples(index=False):
                                payload = bytes.fromhex(candidate.payload_blob)
                                capture_us = int(candidate.trace_time_us)
                                deadline_us = capture_us + int(candidate.deadline_budget_us)
                                perception = _static_perception_costs(candidate, require_perception_costs)
                                representation_generation_ms = float(candidate.generation_ms)
                                perception_service_us = int((
                                    perception["detector_latency_ms"]
                                    + perception["context_model_latency_ms"]
                                ) * 1000)
                                representation_service_us = int(
                                    representation_generation_ms * 1000
                                )
                                if perception_service_us:
                                    perception_start_us = max(
                                        capture_us, perception_server_available_us,
                                    )
                                    perception_queue_wait_us = perception_start_us - capture_us
                                    perception_done_us = perception_start_us + perception_service_us
                                    perception_server_available_us = perception_done_us
                                else:
                                    perception_start_us = capture_us
                                    perception_queue_wait_us = 0
                                    perception_done_us = capture_us
                                representation_start_us = max(
                                    perception_done_us, representation_server_available_us,
                                )
                                representation_queue_wait_us = (
                                    representation_start_us - perception_done_us
                                )
                                ready_us = representation_start_us + representation_service_us
                                representation_server_available_us = ready_us
                                sender_generation_service_ms = (
                                    perception_service_us + representation_service_us
                                ) / 1000
                                sender_generation_queue_wait_us = (
                                    perception_queue_wait_us + representation_queue_wait_us
                                )
                                sender_generation_start_us = (
                                    perception_start_us
                                    if perception_service_us else representation_start_us
                                )
                                message_id = int(hashlib.sha256(
                                    f"exp3|{candidate.sample_id}|{trace_key}|{seed}".encode()
                                ).hexdigest()[:12], 16)
                                if message_id in message_ids:
                                    raise StudyGateError("Static paired message_id collision")
                                message_ids.add(message_id)
                                message = AppMessage(
                                    "exp3", str(candidate.sample_id), message_id,
                                    PayloadMode[mode], capture_us, ready_us,
                                    deadline_us, deadline_us, 4, payload,
                                )
                                offered_wire_bytes = sum(
                                    fragment.ip_packet_bytes for fragment in fragment_message(message)
                                )
                                metadata = {
                                    "replicate_seed": int(seed),
                                    "variant": variant, "rate_mbps": rate,
                                    "rtt_ms": rtt_ms, "loss": loss,
                                    "loss_model": loss_model,
                                    "burst_recovery_probability": float(burst_recovery_probability),
                                    "burst_slot_ms": burst_slot_us / 1000,
                                    "jitter_ms": jitter_ms, "queue_kb": capacity_kb,
                                    "native_receiver_correct": bool(candidate.native_receiver_correct),
                                    "source_model": source_model,
                                    "stream_id": stream_id,
                                    **{
                                        column: value
                                        for column, value in zip(episode_keys, episode_values)
                                    },
                                    "perception_pipeline": perception["perception_pipeline"],
                                    "representation_generation_ms": representation_generation_ms,
                                    "sender_generation_service_ms": sender_generation_service_ms,
                                    "sender_generation_queue_wait_ms": sender_generation_queue_wait_us / 1000,
                                    "sender_generation_start_us": sender_generation_start_us,
                                    "compute_flow_model": "two_stage_fcfs_perception_then_representation",
                                    "perception_service_ms": perception_service_us / 1000,
                                    "perception_queue_wait_ms": perception_queue_wait_us / 1000,
                                    "perception_start_us": perception_start_us,
                                    "perception_done_us": perception_done_us,
                                    "representation_queue_wait_ms": representation_queue_wait_us / 1000,
                                    "representation_start_us": representation_start_us,
                                    "representation_energy_j": float(candidate.energy_j),
                                    "receiver_energy_j": float(candidate.receiver_energy_j),
                                    "receiver_latency_us": int(float(candidate.receiver_latency_ms) * 1000),
                                    "detector_latency_ms": perception["detector_latency_ms"],
                                    "detector_energy_j": perception["detector_energy_j"],
                                    "context_model_latency_ms": perception["context_model_latency_ms"],
                                    "context_model_energy_j": perception["context_model_energy_j"],
                                    "payload_bytes": len(payload),
                                    "offered_wire_bytes": offered_wire_bytes,
                                    "wire_format": _wire_format(variant),
                                }
                                if ready_us >= deadline_us:
                                    expired_rows.append(_expired_generation_row(message, metadata))
                                else:
                                    captures.append((ready_us, (message, metadata)))

                            simulator = SelectiveRepeatSimulator(
                                uplink, downlink, seed=seed,
                                queue_capacity_bytes=int(capacity_kb) * 1024,
                                loss_model=loss_model,
                                burst_recovery_probability=burst_recovery_probability,
                                burst_slot_us=burst_slot_us,
                            )
                            if captures:
                                result, events = simulator.run(
                                    [], captures=captures,
                                    capture_handler=lambda _sim, _now, prepared: prepared,
                                )
                                result["generation_deadline_drop"] = False
                            else:
                                result = pd.DataFrame()
                                events = pd.DataFrame()
                            if expired_rows:
                                result = pd.concat(
                                    [result, pd.DataFrame(expired_rows)], ignore_index=True,
                                    sort=False,
                                )
                            result["radio_tx_energy_j"] = (
                                result["original_uplink_bytes"] + result["retry_uplink_bytes"]
                            ) * radio_tx_j_per_byte
                            result["radio_rx_energy_j"] = (
                                result["downlink_control_bytes"] * radio_rx_j_per_byte
                            )
                            result["receiver_energy_j_charged"] = (
                                result["receiver_energy_j"]
                                * result["full_message_received"].astype(bool)
                            )
                            result["energy_j"] = (
                                result["representation_energy_j"]
                                + result["receiver_energy_j_charged"]
                                + result["detector_energy_j"]
                                + result["context_model_energy_j"]
                                + result["radio_tx_energy_j"]
                                + result["radio_rx_energy_j"]
                            )
                            result["correct_timely_action"] = (
                                result["timely_delivery"].astype(bool)
                                & result["native_receiver_correct"].astype(bool)
                            )
                            outputs.append(_add_stream_audits(result, events, float(rate)))
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


_EXPERIMENT3_SUMMARY_GROUP_COLUMNS = (
    "stream_id", "mode", "variant", "source_model",
    "replicate_seed",
    "rate_mbps", "rtt_ms", "loss", "loss_model",
    "burst_recovery_probability", "burst_slot_ms", "jitter_ms", "queue_kb",
)


_EXPERIMENT3_REPLICATE_COVERAGE_COLUMNS = (
    "observation_id", "mode", "variant", "source_model",
    "rate_mbps", "rtt_ms", "loss", "loss_model",
    "burst_recovery_probability", "burst_slot_ms", "jitter_ms", "queue_kb",
)


def assert_experiment3_exact_replicate_coverage(
    frame: pd.DataFrame,
    registered_seeds: Sequence[int],
) -> None:
    """Require every registered seed to contain the identical static design.

    Checking only the set of seed labels is insufficient: a cache shard or
    interrupted run could leave one seed partially populated while preserving
    that set.  This gate rejects duplicate design cells and any missing or
    extra design cell relative to the first registered seed.
    """
    seeds = tuple(int(seed) for seed in registered_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise StudyGateError("Experiment 3 registered seeds must be nonempty and unique")
    required = {"replicate_seed", *_EXPERIMENT3_REPLICATE_COVERAGE_COLUMNS}
    if missing := required.difference(frame.columns):
        raise StudyGateError(
            f"Experiment 3 replicate coverage is missing columns: {sorted(missing)}"
        )
    observed_seed_series = pd.to_numeric(frame["replicate_seed"], errors="raise").astype(int)
    observed_seeds = set(observed_seed_series)
    if observed_seeds != set(seeds):
        raise StudyGateError(
            "Experiment 3 replicate seeds differ from the registered set: "
            f"registered={sorted(seeds)}, observed={sorted(observed_seeds)}"
        )

    reference_seed = seeds[0]
    reference_cells: set[tuple[Any, ...]] | None = None
    for seed in seeds:
        seed_rows = frame.loc[
            observed_seed_series.eq(seed), list(_EXPERIMENT3_REPLICATE_COVERAGE_COLUMNS)
        ]
        if seed_rows.duplicated(list(_EXPERIMENT3_REPLICATE_COVERAGE_COLUMNS)).any():
            raise StudyGateError(
                f"Experiment 3 seed {seed} contains duplicate registered design cells"
            )
        cells = set(seed_rows.itertuples(index=False, name=None))
        if reference_cells is None:
            reference_cells = cells
            continue
        if cells != reference_cells:
            missing_count = len(reference_cells - cells)
            extra_count = len(cells - reference_cells)
            raise StudyGateError(
                f"Experiment 3 seed {seed} has incomplete design coverage relative "
                f"to seed {reference_seed}: missing={missing_count}, extra={extra_count}"
            )


def summarize_experiment3_static_transport(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one proposal-aligned row per representation/link stream.

    Delivery and deadline probabilities retain failures in their denominators.
    Latency quantiles use only finite, completely delivered observations. Queue
    delay is the sender-ready-to-first-uplink-transmission wait and therefore
    excludes the separately reported sender-generation queue.
    """
    metric_columns = (
        "full_message_received", "application_valid", "timely_delivery",
        "correct_timely_action", "latency_ms", "network_queue_wait_ms",
        "sender_generation_queue_wait_ms", "queue_overflow",
        "generation_deadline_drop", "stream_offered_load_bps",
        "stream_offered_load_ratio", "stream_delivered_payload_goodput_bps",
        "stream_delivered_wire_goodput_bps", "stream_uplink_wire_rate_bps",
        "stream_time_average_aoi_ms", "stream_freshest_age_at_end_ms",
    )
    output_columns = [
        *_EXPERIMENT3_SUMMARY_GROUP_COLUMNS,
        "observations", "delivered_observations", "delivery_probability",
        "valid_delivery_probability", "deadline_success_probability",
        "correct_timely_action_probability", "e2e_latency_p50_ms",
        "e2e_latency_p95_ms", "e2e_latency_p99_ms", "mean_queue_delay_ms",
        "mean_sender_generation_queue_wait_ms", "offered_load_bps",
        "offered_load_ratio", "payload_goodput_bps", "wire_goodput_bps",
        "uplink_wire_rate_bps", "queue_drop_probability",
        "generation_deadline_drop_probability", "time_average_aoi_ms",
        "freshest_age_at_end_ms",
    ]
    if frame.empty:
        return pd.DataFrame(columns=output_columns)
    required = set(_EXPERIMENT3_SUMMARY_GROUP_COLUMNS) | set(metric_columns)
    if missing := required.difference(frame.columns):
        raise StudyGateError(
            f"Experiment 3 summary input is missing columns: {sorted(missing)}"
        )

    def _constant_stream_metric(group: pd.DataFrame, column: str) -> float:
        values = pd.to_numeric(group[column], errors="raise").drop_duplicates()
        if len(values) != 1:
            raise StudyGateError(
                f"Experiment 3 stream metric {column!r} is not constant within a summary group"
            )
        return float(values.iloc[0])

    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(list(_EXPERIMENT3_SUMMARY_GROUP_COLUMNS), sort=True, dropna=False)
    for group_values, group in grouped:
        complete = group["full_message_received"].astype(bool)
        valid = complete & group["application_valid"].astype(bool)
        timely = group["timely_delivery"].astype(bool)
        correct_timely = group["correct_timely_action"].astype(bool)
        latency = pd.to_numeric(group["latency_ms"], errors="coerce")
        finite_latency = latency.loc[complete & np.isfinite(latency)]
        queue_delay = pd.to_numeric(group["network_queue_wait_ms"], errors="coerce")
        sender_queue = pd.to_numeric(
            group["sender_generation_queue_wait_ms"], errors="coerce",
        )
        row = dict(zip(_EXPERIMENT3_SUMMARY_GROUP_COLUMNS, group_values))
        row.update({
            "observations": int(len(group)),
            "delivered_observations": int(complete.sum()),
            "delivery_probability": float(complete.mean()),
            "valid_delivery_probability": float(valid.mean()),
            "deadline_success_probability": float(timely.mean()),
            "correct_timely_action_probability": float(correct_timely.mean()),
            "e2e_latency_p50_ms": float(finite_latency.quantile(0.50)) if len(finite_latency) else np.nan,
            "e2e_latency_p95_ms": float(finite_latency.quantile(0.95)) if len(finite_latency) else np.nan,
            "e2e_latency_p99_ms": float(finite_latency.quantile(0.99)) if len(finite_latency) else np.nan,
            "mean_queue_delay_ms": float(queue_delay.mean()) if queue_delay.notna().any() else np.nan,
            "mean_sender_generation_queue_wait_ms": float(sender_queue.mean()) if sender_queue.notna().any() else np.nan,
            "offered_load_bps": _constant_stream_metric(group, "stream_offered_load_bps"),
            "offered_load_ratio": _constant_stream_metric(group, "stream_offered_load_ratio"),
            "payload_goodput_bps": _constant_stream_metric(group, "stream_delivered_payload_goodput_bps"),
            "wire_goodput_bps": _constant_stream_metric(group, "stream_delivered_wire_goodput_bps"),
            "uplink_wire_rate_bps": _constant_stream_metric(group, "stream_uplink_wire_rate_bps"),
            "queue_drop_probability": float(group["queue_overflow"].astype(bool).mean()),
            "generation_deadline_drop_probability": float(
                group["generation_deadline_drop"].astype(bool).mean()
            ),
            "time_average_aoi_ms": _constant_stream_metric(group, "stream_time_average_aoi_ms"),
            "freshest_age_at_end_ms": _constant_stream_metric(group, "stream_freshest_age_at_end_ms"),
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=output_columns)


_EXPERIMENT3_PARTITION_INDEX_COLUMNS = (
    "seed", "shard", "stage", "artifact_path", "artifact_sha256",
    "row_count", "design_hash",
)


EXPERIMENT3_PUBLICATION_SUMMARY_SCHEMA = (
    "semuav.exp3.publication-summary.v1"
)
EXPERIMENT3_PUBLICATION_SUMMARY_MAX_ROWS = 200_000
_EXPERIMENT3_PUBLICATION_GROUP_COLUMNS = (
    "mode", "variant", "source_model",
    "rate_mbps", "rtt_ms", "loss", "loss_model",
    "burst_recovery_probability", "burst_slot_ms", "jitter_ms", "queue_kb",
)


def experiment3_publication_summary_row_bound(
    candidate_frame: pd.DataFrame,
    *,
    rates_mbps: Sequence[float],
    rtts_ms: Sequence[float],
    losses: Sequence[float],
    loss_models: Sequence[str],
    jitters_ms: Sequence[float],
    queue_kb: Sequence[int],
) -> int:
    """Return the exact paper-summary row count before a campaign starts.

    Registered seeds, observations, visual groups, and mission profiles are
    deliberately absent from this formula because they are reduced within a
    representation/link cell.
    """

    representation_columns = ["mode", "variant", "source_model"]
    if missing := set(representation_columns).difference(candidate_frame.columns):
        raise StudyGateError(
            f"Experiment 3 candidates lack publication keys: {sorted(missing)}"
        )
    representations = int(
        candidate_frame[representation_columns].astype(str).drop_duplicates().shape[0]
    )
    axes = (
        tuple(rates_mbps), tuple(rtts_ms), tuple(losses), tuple(loss_models),
        tuple(jitters_ms), tuple(queue_kb),
    )
    if any(not axis for axis in axes):
        raise StudyGateError("Experiment 3 publication grid has an empty axis")
    return int(representations * math.prod(len(axis) for axis in axes))


def _experiment3_publication_partial(frame: pd.DataFrame) -> pd.DataFrame:
    """Return additive sufficient statistics for one raw static partition.

    This intentionally contains no stream identifier, visual-group label, or
    seed dimension.  Those dimensions would make the paper-facing descriptive
    table grow with the number of independent groups.  H1 inference continues
    to use the separately generated, group-equal fixed-grid aggregate derived
    from the checksum-bound raw partitions.
    """

    required = {
        *_EXPERIMENT3_PUBLICATION_GROUP_COLUMNS,
        "replicate_seed", "stream_id", "full_message_received",
        "application_valid", "timely_delivery", "correct_timely_action",
        "latency_ms", "network_queue_wait_ms",
        "sender_generation_queue_wait_ms", "queue_overflow",
        "generation_deadline_drop", "stream_offered_load_bps",
        "stream_offered_load_ratio", "stream_delivered_payload_goodput_bps",
        "stream_delivered_wire_goodput_bps", "stream_uplink_wire_rate_bps",
        "stream_time_average_aoi_ms", "stream_freshest_age_at_end_ms",
        "total_wire_bytes", "energy_j",
    }
    if missing := required.difference(frame.columns):
        raise StudyGateError(
            "Experiment 3 publication reduction is missing columns: "
            f"{sorted(missing)}"
        )
    work = frame[list(required)].copy()
    complete = work["full_message_received"].astype(bool)
    work["__delivered"] = complete.astype(np.int64)
    work["__valid"] = (
        complete & work["application_valid"].astype(bool)
    ).astype(np.int64)
    work["__timely"] = work["timely_delivery"].astype(bool).astype(np.int64)
    work["__correct_timely"] = (
        work["correct_timely_action"].astype(bool).astype(np.int64)
    )
    latency = pd.to_numeric(work["latency_ms"], errors="coerce")
    finite_delivered = complete & np.isfinite(latency)
    work["__latency_sum"] = latency.where(finite_delivered, 0.0)
    work["__latency_count"] = finite_delivered.astype(np.int64)
    for source, stem in (
        ("network_queue_wait_ms", "network_queue"),
        ("sender_generation_queue_wait_ms", "sender_queue"),
    ):
        values = pd.to_numeric(work[source], errors="coerce")
        finite = np.isfinite(values)
        work[f"__{stem}_sum"] = values.where(finite, 0.0)
        work[f"__{stem}_count"] = finite.astype(np.int64)
    work["__queue_drop"] = work["queue_overflow"].astype(bool).astype(np.int64)
    work["__generation_drop"] = (
        work["generation_deadline_drop"].astype(bool).astype(np.int64)
    )
    numeric_sums = {
        "__wire_sum": "total_wire_bytes",
        "__energy_sum": "energy_j",
        "__offered_load_sum": "stream_offered_load_bps",
        "__offered_ratio_sum": "stream_offered_load_ratio",
        "__payload_goodput_sum": "stream_delivered_payload_goodput_bps",
        "__wire_goodput_sum": "stream_delivered_wire_goodput_bps",
        "__uplink_rate_sum": "stream_uplink_wire_rate_bps",
        "__aoi_sum": "stream_time_average_aoi_ms",
        "__freshest_age_sum": "stream_freshest_age_at_end_ms",
    }
    for target, source in numeric_sums.items():
        work[target] = pd.to_numeric(work[source], errors="raise")

    grouped = work.groupby(
        list(_EXPERIMENT3_PUBLICATION_GROUP_COLUMNS),
        as_index=False, sort=True, dropna=False,
    ).agg(
        observation_rows=("replicate_seed", "size"),
        delivered_observations=("__delivered", "sum"),
        valid_deliveries=("__valid", "sum"),
        timely_deliveries=("__timely", "sum"),
        correct_timely_actions=("__correct_timely", "sum"),
        finite_delivered_latency_count=("__latency_count", "sum"),
        finite_delivered_latency_sum_ms=("__latency_sum", "sum"),
        network_queue_count=("__network_queue_count", "sum"),
        network_queue_sum_ms=("__network_queue_sum", "sum"),
        sender_queue_count=("__sender_queue_count", "sum"),
        sender_queue_sum_ms=("__sender_queue_sum", "sum"),
        queue_drops=("__queue_drop", "sum"),
        generation_deadline_drops=("__generation_drop", "sum"),
        total_wire_bytes_sum=("__wire_sum", "sum"),
        total_energy_j_sum=("__energy_sum", "sum"),
        offered_load_bps_sum=("__offered_load_sum", "sum"),
        offered_load_ratio_sum=("__offered_ratio_sum", "sum"),
        payload_goodput_bps_sum=("__payload_goodput_sum", "sum"),
        wire_goodput_bps_sum=("__wire_goodput_sum", "sum"),
        uplink_wire_rate_bps_sum=("__uplink_rate_sum", "sum"),
        time_average_aoi_ms_sum=("__aoi_sum", "sum"),
        freshest_age_at_end_ms_sum=("__freshest_age_sum", "sum"),
        episode_seed_cells=("stream_id", "nunique"),
    )
    grouped["source_partition_count"] = 1
    grouped["observed_seed_count"] = 1
    return grouped


def _finalize_experiment3_publication_summary(
    accumulator: Mapping[tuple[Any, ...], Mapping[str, Any]],
    *,
    expected_coverage: Mapping[tuple[str, str, str], tuple[int, int]],
    registered_seeds: Sequence[int],
    expected_grid_conditions: int,
    partition_index: pd.DataFrame,
) -> pd.DataFrame:
    """Finalize bounded static sufficient statistics and coverage evidence."""

    rows: list[dict[str, Any]] = []
    seeds = tuple(int(seed) for seed in registered_seeds)
    partition_binding = hash_object(
        partition_index[list(_EXPERIMENT3_PARTITION_INDEX_COLUMNS)]
        .sort_values(["seed", "shard"], kind="mergesort")
        .to_dict("records")
    )
    raw_partition_rows = int(pd.to_numeric(
        partition_index["row_count"], errors="raise",
    ).sum())
    additive = (
        "observation_rows", "delivered_observations", "valid_deliveries",
        "timely_deliveries", "correct_timely_actions",
        "finite_delivered_latency_count", "finite_delivered_latency_sum_ms",
        "network_queue_count", "network_queue_sum_ms",
        "sender_queue_count", "sender_queue_sum_ms", "queue_drops",
        "generation_deadline_drops", "total_wire_bytes_sum",
        "total_energy_j_sum", "offered_load_bps_sum",
        "offered_load_ratio_sum", "payload_goodput_bps_sum",
        "wire_goodput_bps_sum", "uplink_wire_rate_bps_sum",
        "time_average_aoi_ms_sum", "freshest_age_at_end_ms_sum",
        "episode_seed_cells", "source_partition_count",
    )
    for key in sorted(accumulator, key=lambda item: tuple(map(str, item))):
        values = accumulator[key]
        row = dict(zip(_EXPERIMENT3_PUBLICATION_GROUP_COLUMNS, key))
        for column in additive:
            row[column] = values[column]
        observed_seeds = set(map(int, values["seeds"]))
        representation = tuple(map(str, key[:3]))
        if representation not in expected_coverage:
            raise StudyGateError(
                f"Experiment 3 publication row has an unknown representation: {representation}"
            )
        observations_per_seed, episodes_per_seed = expected_coverage[representation]
        expected_observations = observations_per_seed * len(seeds)
        expected_episode_cells = episodes_per_seed * len(seeds)
        observation_rows = int(row["observation_rows"])
        row.update({
            "delivery_probability": row["delivered_observations"] / observation_rows,
            "valid_delivery_probability": row["valid_deliveries"] / observation_rows,
            "deadline_success_probability": row["timely_deliveries"] / observation_rows,
            "correct_timely_action_probability": row["correct_timely_actions"] / observation_rows,
            "mean_e2e_latency_ms_delivered": (
                row["finite_delivered_latency_sum_ms"]
                / row["finite_delivered_latency_count"]
                if row["finite_delivered_latency_count"] else np.nan
            ),
            "mean_queue_delay_ms": (
                row["network_queue_sum_ms"] / row["network_queue_count"]
                if row["network_queue_count"] else np.nan
            ),
            "mean_sender_generation_queue_wait_ms": (
                row["sender_queue_sum_ms"] / row["sender_queue_count"]
                if row["sender_queue_count"] else np.nan
            ),
            "mean_wire_bytes": row["total_wire_bytes_sum"] / observation_rows,
            "mean_energy_j": row["total_energy_j_sum"] / observation_rows,
            "offered_load_bps": row["offered_load_bps_sum"] / observation_rows,
            "offered_load_ratio": row["offered_load_ratio_sum"] / observation_rows,
            "payload_goodput_bps": row["payload_goodput_bps_sum"] / observation_rows,
            "wire_goodput_bps": row["wire_goodput_bps_sum"] / observation_rows,
            "uplink_wire_rate_bps": row["uplink_wire_rate_bps_sum"] / observation_rows,
            "queue_drop_probability": row["queue_drops"] / observation_rows,
            "generation_deadline_drop_probability": (
                row["generation_deadline_drops"] / observation_rows
            ),
            "time_average_aoi_ms": row["time_average_aoi_ms_sum"] / observation_rows,
            "freshest_age_at_end_ms": row["freshest_age_at_end_ms_sum"] / observation_rows,
            "registered_seed_count": len(seeds),
            "observed_seed_count": len(observed_seeds),
            "expected_observation_rows": expected_observations,
            "expected_episode_seed_cells": expected_episode_cells,
            "exact_seed_coverage": observed_seeds == set(seeds),
            "exact_observation_coverage": observation_rows == expected_observations,
            "exact_episode_coverage": int(row["episode_seed_cells"]) == expected_episode_cells,
            "exact_partition_coverage": int(row["source_partition_count"]) == len(seeds),
            "registered_grid_condition_count": int(expected_grid_conditions),
            "publication_summary_schema": EXPERIMENT3_PUBLICATION_SUMMARY_SCHEMA,
            "aggregation_scope": (
                "representation_link_cell_across_registered_seeds_and_visual_streams"
            ),
            "inference_role": (
                "descriptive_publication_only_h1_uses_group_equal_fixed_grid_aggregate"
            ),
            "h1_estimand": "visual_group_equal_over_registered_fixed_grid_v1",
            "raw_partition_index_sha256": partition_binding,
            "raw_partition_count": int(len(partition_index)),
            "raw_partition_rows_verified": raw_partition_rows,
        })
        row["exact_registered_coverage"] = bool(
            row["exact_seed_coverage"]
            and row["exact_observation_coverage"]
            and row["exact_episode_coverage"]
            and row["exact_partition_coverage"]
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        raise StudyGateError("Experiment 3 publication summary is empty")
    representation_columns = ["mode", "variant", "source_model"]
    condition_counts = result.groupby(
        representation_columns, dropna=False,
    ).size()
    if not condition_counts.eq(int(expected_grid_conditions)).all():
        raise StudyGateError(
            "Experiment 3 publication summary lacks exact registered grid coverage"
        )
    if not result["exact_registered_coverage"].astype(bool).all():
        raise StudyGateError(
            "Experiment 3 publication summary lacks exact seed/observation/episode coverage"
        )
    # Internal additive sums are reducer state, not report-facing endpoints.
    # Retaining only denominators, counts, and final estimates materially
    # reduces the compressed artifact size while preserving auditability.
    result = result.drop(columns=[
        column for column in result.columns
        if column.endswith("_sum") or column.endswith("_sum_ms")
    ])
    return result.sort_values(
        list(_EXPERIMENT3_PUBLICATION_GROUP_COLUMNS), kind="mergesort",
    ).reset_index(drop=True)


def _experiment3_candidate_shards(
    candidate_frame: pd.DataFrame,
    shard_size: int,
) -> Iterable[tuple[int, pd.DataFrame]]:
    """Yield bounded candidate shards without splitting a transport stream."""
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    stream_columns = ["mode", "variant", "source_model"]
    ordered = candidate_frame.sort_values(
        [*stream_columns, "sample_id"], kind="mergesort",
    )
    pending_groups: list[pd.DataFrame] = []
    pending_rows = 0
    shard_index = 0
    for _, group in ordered.groupby(
        stream_columns, sort=True, dropna=False,
    ):
        group = group.copy()
        if pending_groups and pending_rows + len(group) > shard_size:
            shard = pd.concat(pending_groups, ignore_index=True)
            pending_groups, pending_rows = [], 0
            yield shard_index, shard
            shard_index += 1
        pending_groups.append(group)
        pending_rows += len(group)
    if pending_groups:
        yield shard_index, pd.concat(pending_groups, ignore_index=True)


def _experiment3_static_transport_stage(
    candidate_shard: pd.DataFrame,
    manifest: pd.DataFrame,
    ctx: RunContext,
    *,
    seed: int,
    shard_index: int,
    options: Mapping[str, Any],
) -> tuple[str, pd.DataFrame]:
    """Run or load one raw Experiment-3 cache partition."""
    ids = set(candidate_shard["sample_id"].astype(str))
    manifest_shard = manifest[
        manifest["sample_id"].astype(str).isin(ids)
    ].copy()
    manifest_cost_columns = [
        column for column in _STATIC_PERCEPTION_COST_COLUMNS
        if column in manifest_shard.columns
    ]
    key = {
        "candidate_hash": hash_object(candidate_shard.to_dict("records")),
        "manifest_timing_and_compute": hash_object(manifest_shard[[
            "sample_id", "capture_time_us", "trace_time_us", "deadline_us",
            *manifest_cost_columns,
        ]].to_dict("records")),
        "execution_model": "two_stage_fcfs_sender_shared_transport_registered_loss_v3",
        "seed": int(seed),
        "rates_mbps": list(options["rates_mbps"]),
        "rtts_ms": list(options["rtts_ms"]),
        "losses": list(options["losses"]),
        "loss_models": list(options["loss_models"]),
        "burst_recovery_probability": options["burst_recovery_probability"],
        "burst_slot_ms": options["burst_slot_ms"],
        "jitters_ms": list(options["jitters_ms"]),
        "queue_kb": list(options["queue_kb"]),
        "radio_tx_j_per_byte": options["radio_tx_j_per_byte"],
        "radio_rx_j_per_byte": options["radio_rx_j_per_byte"],
        "require_perception_costs": options["require_perception_costs"],
        "require_registered_loss_models": options["require_registered_loss_models"],
    }
    stage = f"03_static_transport_seed_{int(seed)}_shard_{shard_index:05d}"
    frame = frame_stage(
        ctx, stage, key,
        lambda: experiment3_static_transport(
            candidate_shard, manifest_shard, int(seed), **options,
        ),
    )
    return stage, frame


def _experiment3_design_cell(values: Sequence[Any]) -> tuple[Any, ...]:
    """Canonicalize a registered design cell across CSV dtype inference."""
    if len(values) != len(_EXPERIMENT3_REPLICATE_COVERAGE_COLUMNS):
        raise ValueError("Experiment 3 design-cell width is invalid")
    text_indices = {0, 1, 2, 3, 7}
    integer_indices = {11}
    normalized: list[Any] = []
    for index, value in enumerate(values):
        if index in text_indices:
            normalized.append(str(value))
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise StudyGateError(
                "Experiment 3 partition design contains a nonnumeric condition"
            ) from exc
        if not np.isfinite(number):
            raise StudyGateError(
                "Experiment 3 partition design conditions must be finite"
            )
        if index in integer_indices:
            integer = int(number)
            if number != integer:
                raise StudyGateError("Experiment 3 queue_kb design values must be integers")
            normalized.append(integer)
        else:
            # Treat signed zero identically and force Python scalars so fresh
            # frames and frames reloaded from CSV have identical identities.
            normalized.append(0.0 if number == 0 else number)
    return tuple(normalized)


def _experiment3_design_hash(
    cells: Iterable[Sequence[Any]],
    *,
    description: str,
) -> tuple[str, int]:
    """Hash an unordered partition design while rejecting duplicate cells."""
    ordered = sorted(_experiment3_design_cell(cell) for cell in cells)
    for previous, current in zip(ordered, ordered[1:]):
        if previous == current:
            raise StudyGateError(
                f"Experiment 3 {description} contains duplicate registered design cells"
            )
    digest = hashlib.sha256()
    digest.update(b"semuav.exp3.partition-design.v1\n")
    digest.update(json.dumps(
        list(_EXPERIMENT3_REPLICATE_COVERAGE_COLUMNS),
        separators=(",", ":"),
    ).encode("utf-8"))
    digest.update(b"\n")
    for cell in ordered:
        digest.update(json.dumps(
            cell, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest(), len(ordered)


def _experiment3_expected_partition_design(
    candidate_shard: pd.DataFrame,
    options: Mapping[str, Any],
) -> tuple[str, int]:
    """Build the registered Cartesian design for one candidate partition."""
    axes = {
        "rates_mbps": tuple(options["rates_mbps"]),
        "rtts_ms": tuple(options["rtts_ms"]),
        "losses": tuple(options["losses"]),
        "loss_models": tuple(
            str(model).strip().lower() for model in options["loss_models"]
        ),
        "jitters_ms": tuple(options["jitters_ms"]),
        "queue_kb": tuple(options["queue_kb"]),
    }
    empty_axes = sorted(name for name, values in axes.items() if not values)
    if empty_axes:
        raise StudyGateError(
            f"Experiment 3 registered condition axes are empty: {empty_axes}"
        )
    burst_slot_ms = int(round(float(options["burst_slot_ms"]) * 1000)) / 1000
    candidate_cells = [
        (
            str(row.sample_id), str(row.mode), str(row.variant),
            str(row.source_model),
        )
        for row in candidate_shard.itertuples(index=False)
    ]
    cells = (
        (
            observation_id, mode, variant, source_model,
            rate, rtt, loss, loss_model,
            options["burst_recovery_probability"], burst_slot_ms,
            jitter, capacity,
        )
        for rate in axes["rates_mbps"]
        for rtt in axes["rtts_ms"]
        for loss in axes["losses"]
        for loss_model in axes["loss_models"]
        for jitter in axes["jitters_ms"]
        for capacity in axes["queue_kb"]
        for observation_id, mode, variant, source_model in candidate_cells
    )
    return _experiment3_design_hash(cells, description="expected partition")


def _experiment3_observed_partition_design(
    frame: pd.DataFrame,
    *,
    seed: int,
    shard_index: int,
) -> tuple[str, int]:
    required = {"replicate_seed", *_EXPERIMENT3_REPLICATE_COVERAGE_COLUMNS}
    if missing := required.difference(frame.columns):
        raise StudyGateError(
            f"Experiment 3 seed {seed} shard {shard_index} is missing design "
            f"columns: {sorted(missing)}"
        )
    observed_seeds = set(
        pd.to_numeric(frame["replicate_seed"], errors="raise").astype(int)
    )
    if observed_seeds != {int(seed)}:
        raise StudyGateError(
            f"Experiment 3 seed {seed} shard {shard_index} contains "
            f"replicate seeds {sorted(observed_seeds)}"
        )
    cells = frame[list(_EXPERIMENT3_REPLICATE_COVERAGE_COLUMNS)].itertuples(
        index=False, name=None,
    )
    return _experiment3_design_hash(
        cells, description=f"seed {seed} shard {shard_index}",
    )


def assert_experiment3_partition_index_coverage(
    partition_index: pd.DataFrame,
    registered_seeds: Sequence[int],
    *,
    verify_artifacts: bool = True,
) -> None:
    """Verify integrity and exact cross-seed coverage of raw partitions.

    The producing API separately checks each partition against its expected
    candidate-by-condition Cartesian product.  This index-level gate then
    requires every registered seed to contain the same contiguous shard set,
    row counts, and per-shard design hashes, without loading raw artifacts.
    """
    seeds = tuple(int(seed) for seed in registered_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise StudyGateError(
            "Experiment 3 registered seeds must be nonempty and unique"
        )
    required = set(_EXPERIMENT3_PARTITION_INDEX_COLUMNS)
    if missing := required.difference(partition_index.columns):
        raise StudyGateError(
            f"Experiment 3 partition index is missing columns: {sorted(missing)}"
        )
    if partition_index.empty:
        raise StudyGateError("Experiment 3 partition index is empty")
    seed_values = pd.to_numeric(
        partition_index["seed"], errors="raise",
    ).astype(int)
    shard_values = pd.to_numeric(
        partition_index["shard"], errors="raise",
    ).astype(int)
    row_counts = pd.to_numeric(
        partition_index["row_count"], errors="raise",
    ).astype(int)
    observed_seeds = set(seed_values)
    if observed_seeds != set(seeds):
        raise StudyGateError(
            "Experiment 3 partition seeds differ from the registered set: "
            f"registered={sorted(seeds)}, observed={sorted(observed_seeds)}"
        )
    normalized = partition_index.copy()
    normalized["seed"] = seed_values
    normalized["shard"] = shard_values
    normalized["row_count"] = row_counts
    if normalized.duplicated(["seed", "shard"]).any():
        raise StudyGateError("Experiment 3 partition index has duplicate seed/shard rows")
    if normalized["stage"].astype(str).duplicated().any():
        raise StudyGateError("Experiment 3 partition index has duplicate stages")
    if normalized["artifact_path"].astype(str).duplicated().any():
        raise StudyGateError("Experiment 3 partition index has duplicate artifact paths")
    if (normalized["shard"] < 0).any() or (normalized["row_count"] <= 0).any():
        raise StudyGateError(
            "Experiment 3 partition shard indices must be nonnegative and row counts positive"
        )
    for column in ("artifact_sha256", "design_hash"):
        valid_hashes = normalized[column].astype(str).str.fullmatch(r"[0-9a-f]{64}")
        if not valid_hashes.all():
            raise StudyGateError(
                f"Experiment 3 partition index has an invalid {column}"
            )
    if verify_artifacts:
        for row in normalized.itertuples(index=False):
            path = Path(str(row.artifact_path)).expanduser().resolve()
            if not path.is_file():
                raise StudyGateError(
                    f"Experiment 3 partition artifact is missing: {path}"
                )
            observed_hash = hash_file(path)
            if observed_hash != str(row.artifact_sha256):
                raise StudyGateError(
                    f"Experiment 3 partition artifact checksum changed: {path}"
                )

    reference_seed = seeds[0]
    reference_signature: tuple[tuple[int, int, str], ...] | None = None
    for seed in seeds:
        rows = normalized.loc[normalized["seed"].eq(seed)].sort_values("shard")
        shards = tuple(rows["shard"].astype(int))
        if shards != tuple(range(len(rows))):
            raise StudyGateError(
                f"Experiment 3 seed {seed} has noncontiguous partition shards"
            )
        signature = tuple(
            (int(row.shard), int(row.row_count), str(row.design_hash))
            for row in rows.itertuples(index=False)
        )
        if len({item[2] for item in signature}) != len(signature):
            raise StudyGateError(
                f"Experiment 3 seed {seed} contains duplicate partition designs"
            )
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise StudyGateError(
                f"Experiment 3 seed {seed} has incomplete condition coverage "
                f"relative to seed {reference_seed}"
            )


def experiment3_static_transport_sharded(
    candidate_frame: pd.DataFrame,
    manifest: pd.DataFrame,
    ctx: RunContext,
    seed: int = 101,
    shard_size: int = 64,
    rates_mbps: Sequence[float] = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
    rtts_ms: Sequence[float] = (20.0, 80.0, 200.0),
    losses: Sequence[float] = (0.0, 0.01, 0.05, 0.15),
    loss_models: Sequence[str] = EXPERIMENT3_REGISTERED_LOSS_MODELS,
    burst_recovery_probability: float = 0.2,
    burst_slot_ms: float = 100.0,
    jitters_ms: Sequence[float] = (0.0, 10.0),
    queue_kb: Sequence[int] = (32, 128, 512),
    radio_tx_j_per_byte: float = 0.0,
    radio_rx_j_per_byte: float = 0.0,
    require_perception_costs: bool = False,
    require_registered_loss_models: bool = False,
) -> pd.DataFrame:
    options = {
        "rates_mbps": rates_mbps, "rtts_ms": rtts_ms, "losses": losses,
        "loss_models": loss_models,
        "burst_recovery_probability": burst_recovery_probability,
        "burst_slot_ms": burst_slot_ms, "jitters_ms": jitters_ms,
        "queue_kb": queue_kb,
        "radio_tx_j_per_byte": radio_tx_j_per_byte,
        "radio_rx_j_per_byte": radio_rx_j_per_byte,
        "require_perception_costs": require_perception_costs,
        "require_registered_loss_models": require_registered_loss_models,
    }
    shards: list[pd.DataFrame] = []
    for shard_index, candidate_shard in _experiment3_candidate_shards(
        candidate_frame, shard_size,
    ):
        _, shard = _experiment3_static_transport_stage(
            candidate_shard, manifest, ctx, seed=int(seed),
            shard_index=shard_index, options=options,
        )
        shards.append(shard)
    return pd.concat(shards, ignore_index=True) if shards else pd.DataFrame()


_EXPERIMENT3_EPISODE_KEYS = ("visual_group_id", "mission_profile")


def _drop_verified_episode_keys(
    candidate_frame: pd.DataFrame, manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Remove candidate copies of the manifest's episode keys after verifying them.

    v3r2 correction (receipted in v3r1): Experiment-2 candidates already carry
    ``visual_group_id``/``mission_profile``.  The static runner merges these
    keys from the authoritative manifest, so the duplicate copies produced
    suffixed columns and ``KeyError: visual_group_id``.  Each candidate value
    must equal the manifest value for its sample before the copy is dropped.
    """
    overlap = [
        column for column in _EXPERIMENT3_EPISODE_KEYS
        if column in candidate_frame.columns and column in manifest.columns
    ]
    if not overlap:
        return candidate_frame
    authority = manifest[["sample_id", *overlap]].copy()
    authority["sample_id"] = authority["sample_id"].astype(str)
    if authority["sample_id"].duplicated().any():
        raise StudyGateError("Static manifest has duplicate sample_id rows")
    copies = candidate_frame[["sample_id", *overlap]].copy()
    copies["sample_id"] = copies["sample_id"].astype(str)
    joined = copies.merge(
        authority, on="sample_id", how="left",
        suffixes=("_candidate", "_manifest"), validate="many_to_one",
    )
    for column in overlap:
        candidate_values = joined[f"{column}_candidate"].astype(str)
        manifest_values = joined[f"{column}_manifest"].astype(str)
        if joined[f"{column}_manifest"].isna().any() or not candidate_values.eq(manifest_values).all():
            raise StudyGateError(
                f"Static candidate {column} disagrees with the authoritative manifest"
            )
    return candidate_frame.drop(columns=overlap)


def experiment3_static_transport_partitioned(
    candidate_frame: pd.DataFrame,
    manifest: pd.DataFrame,
    ctx: RunContext,
    registered_seeds: Sequence[int] = (101,),
    shard_size: int = 64,
    rates_mbps: Sequence[float] = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
    rtts_ms: Sequence[float] = (20.0, 80.0, 200.0),
    losses: Sequence[float] = (0.0, 0.01, 0.05, 0.15),
    loss_models: Sequence[str] = EXPERIMENT3_REGISTERED_LOSS_MODELS,
    burst_recovery_probability: float = 0.2,
    burst_slot_ms: float = 100.0,
    jitters_ms: Sequence[float] = (0.0, 10.0),
    queue_kb: Sequence[int] = (32, 128, 512),
    radio_tx_j_per_byte: float = 0.0,
    radio_rx_j_per_byte: float = 0.0,
    require_perception_costs: bool = False,
    require_registered_loss_models: bool = False,
    publication_summary_max_rows: int = EXPERIMENT3_PUBLICATION_SUMMARY_MAX_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Characterize all registered seeds without combining raw partitions.

    Each stream-preserving raw cache partition is validated against its exact
    candidate-by-condition Cartesian design and reduced before the next raw
    partition is loaded.  The paper-facing summary is one row per
    representation/link cell aggregated across registered seeds and visual
    streams, so its row count cannot grow with the number of visual groups.
    H1 uses a separate group-equal reducer over the checksum-bound raw
    partitions.  The returned pair is ``(summary, partition_index)``; no
    combined raw or per-stream-summary frame is constructed or returned.
    """
    candidate_frame = _drop_verified_episode_keys(candidate_frame, manifest)
    seeds = tuple(int(seed) for seed in registered_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise StudyGateError(
            "Experiment 3 registered seeds must be nonempty and unique"
        )
    rates_mbps = tuple(rates_mbps)
    rtts_ms = tuple(rtts_ms)
    losses = tuple(losses)
    loss_models = tuple(loss_models)
    jitters_ms = tuple(jitters_ms)
    queue_kb = tuple(queue_kb)
    if (
        isinstance(publication_summary_max_rows, (bool, np.bool_))
        or int(publication_summary_max_rows) <= 0
    ):
        raise ValueError("publication_summary_max_rows must be a positive integer")
    options = {
        "rates_mbps": rates_mbps, "rtts_ms": rtts_ms, "losses": losses,
        "loss_models": loss_models,
        "burst_recovery_probability": burst_recovery_probability,
        "burst_slot_ms": burst_slot_ms, "jitters_ms": jitters_ms,
        "queue_kb": queue_kb,
        "radio_tx_j_per_byte": radio_tx_j_per_byte,
        "radio_rx_j_per_byte": radio_rx_j_per_byte,
        "require_perception_costs": require_perception_costs,
        "require_registered_loss_models": require_registered_loss_models,
    }
    representation_columns = ["mode", "variant", "source_model"]
    candidate_required = {"sample_id", *representation_columns}
    if missing := candidate_required.difference(candidate_frame.columns):
        raise StudyGateError(
            f"Experiment 3 candidates lack publication keys: {sorted(missing)}"
        )
    normalized_candidates = candidate_frame.copy()
    normalized_candidates["sample_id"] = normalized_candidates["sample_id"].astype(str)
    for column in representation_columns:
        normalized_candidates[column] = normalized_candidates[column].astype(str)
    if normalized_candidates.duplicated([*representation_columns, "sample_id"]).any():
        raise StudyGateError(
            "Experiment 3 publication candidates duplicate a representation/sample"
        )
    representation_count = int(
        normalized_candidates[representation_columns].drop_duplicates().shape[0]
    )
    expected_summary_rows = experiment3_publication_summary_row_bound(
        normalized_candidates,
        rates_mbps=rates_mbps, rtts_ms=rtts_ms, losses=losses,
        loss_models=loss_models, jitters_ms=jitters_ms, queue_kb=queue_kb,
    )
    if representation_count <= 0:
        raise StudyGateError("Experiment 3 publication candidates are empty")
    grid_condition_count = expected_summary_rows // representation_count
    if expected_summary_rows > int(publication_summary_max_rows):
        raise StudyGateError(
            "Experiment 3 paper-facing summary would exceed its registered row cap: "
            f"{expected_summary_rows} > {int(publication_summary_max_rows)}"
        )
    manifest_keys = ["sample_id"]
    has_episode_keys = {"visual_group_id", "mission_profile"}.issubset(manifest.columns)
    if has_episode_keys:
        manifest_keys.extend(["visual_group_id", "mission_profile"])
    manifest_mapping = manifest[manifest_keys].copy()
    manifest_mapping["sample_id"] = manifest_mapping["sample_id"].astype(str)
    if manifest_mapping["sample_id"].duplicated().any():
        raise StudyGateError("Experiment 3 publication manifest duplicates sample_id")
    expected_frame = normalized_candidates.merge(
        manifest_mapping, on="sample_id", how="left", validate="many_to_one",
    )
    if has_episode_keys and expected_frame[["visual_group_id", "mission_profile"]].isna().any().any():
        raise StudyGateError("Experiment 3 publication candidates lack episode labels")
    expected_coverage: dict[tuple[str, str, str], tuple[int, int]] = {}
    for representation, group in expected_frame.groupby(
        representation_columns, sort=True, dropna=False,
    ):
        representation_key = tuple(map(str, representation))
        episodes = (
            int(group[["visual_group_id", "mission_profile"]].drop_duplicates().shape[0])
            if has_episode_keys else 1
        )
        expected_coverage[representation_key] = (int(len(group)), episodes)

    additive_columns = (
        "observation_rows", "delivered_observations", "valid_deliveries",
        "timely_deliveries", "correct_timely_actions",
        "finite_delivered_latency_count", "finite_delivered_latency_sum_ms",
        "network_queue_count", "network_queue_sum_ms",
        "sender_queue_count", "sender_queue_sum_ms", "queue_drops",
        "generation_deadline_drops", "total_wire_bytes_sum",
        "total_energy_j_sum", "offered_load_bps_sum",
        "offered_load_ratio_sum", "payload_goodput_bps_sum",
        "wire_goodput_bps_sum", "uplink_wire_rate_bps_sum",
        "time_average_aoi_ms_sum", "freshest_age_at_end_ms_sum",
        "episode_seed_cells", "source_partition_count",
    )
    accumulator: dict[tuple[Any, ...], dict[str, Any]] = {}
    index_rows: list[dict[str, Any]] = []
    for seed in seeds:
        partition_count = 0
        for shard_index, candidate_shard in _experiment3_candidate_shards(
            candidate_frame, shard_size,
        ):
            partition_count += 1
            expected_hash, expected_rows = _experiment3_expected_partition_design(
                candidate_shard, options,
            )
            stage, raw_shard = _experiment3_static_transport_stage(
                candidate_shard, manifest, ctx, seed=seed,
                shard_index=shard_index, options=options,
            )
            observed_hash, observed_rows = _experiment3_observed_partition_design(
                raw_shard, seed=seed, shard_index=shard_index,
            )
            if observed_rows != expected_rows or observed_hash != expected_hash:
                raise StudyGateError(
                    f"Experiment 3 seed {seed} shard {shard_index} does not have "
                    "exact registered condition coverage: "
                    f"expected_rows={expected_rows}, observed_rows={observed_rows}, "
                    f"expected_design={expected_hash}, observed_design={observed_hash}"
                )

            # Reduce immediately into additive representation/link sufficient
            # statistics.  No per-stream summary list or campaign concat is
            # permitted here.
            partial = _experiment3_publication_partial(raw_shard)
            for row in partial.itertuples(index=False):
                key = tuple(
                    getattr(row, column)
                    for column in _EXPERIMENT3_PUBLICATION_GROUP_COLUMNS
                )
                target = accumulator.setdefault(
                    key,
                    {**{column: 0 for column in additive_columns}, "seeds": set()},
                )
                for column in additive_columns:
                    target[column] += getattr(row, column)
                target["seeds"].add(int(seed))
            record = ctx.stage_record(stage)
            if not record or record.get("status") != "COMPLETE":
                raise StudyGateError(
                    f"Experiment 3 partition stage {stage!r} is not complete"
                )
            artifact = Path(str(record.get("artifact", ""))).expanduser().resolve()
            if not artifact.is_file():
                raise StudyGateError(
                    f"Experiment 3 partition stage {stage!r} has no artifact"
                )
            artifact_sha256 = hash_file(artifact)
            if str(record.get("checksum", "")) != artifact_sha256:
                raise StudyGateError(
                    f"Experiment 3 partition stage {stage!r} checksum is inconsistent"
                )
            if int(record.get("rows", -1)) != observed_rows:
                raise StudyGateError(
                    f"Experiment 3 partition stage {stage!r} row count is inconsistent"
                )
            index_rows.append({
                "seed": seed,
                "shard": shard_index,
                "stage": stage,
                "artifact_path": str(artifact),
                "artifact_sha256": artifact_sha256,
                "row_count": observed_rows,
                "design_hash": observed_hash,
            })
            del partial, raw_shard
        if partition_count == 0:
            raise StudyGateError(
                f"Experiment 3 seed {seed} produced no candidate partitions"
            )

    partition_index = pd.DataFrame(
        index_rows, columns=_EXPERIMENT3_PARTITION_INDEX_COLUMNS,
    )
    assert_experiment3_partition_index_coverage(
        partition_index, seeds, verify_artifacts=True,
    )
    summary = _finalize_experiment3_publication_summary(
        accumulator,
        expected_coverage=expected_coverage,
        registered_seeds=seeds,
        expected_grid_conditions=grid_condition_count,
        partition_index=partition_index,
    )
    if len(summary) != expected_summary_rows:
        raise StudyGateError(
            "Experiment 3 publication summary row count differs from the registered design: "
            f"{len(summary)} != {expected_summary_rows}"
        )
    if len(summary) > int(publication_summary_max_rows):
        raise StudyGateError("Experiment 3 publication summary exceeded its row cap")
    return summary, partition_index


def build_causal_candidate_profiles(
    candidate_frame: pd.DataFrame,
    development_sample_ids: Sequence[str],
    *,
    semantic_acceptance_threshold: float = 0.80,
    gold_actions: Mapping[str, str] | None = None,
    calibration_trusted: bool = True,
) -> pd.DataFrame:
    """Estimate representation profiles using development IDs only.

    v3r2 controller revision (pre-specified before any test access):

    * ``gold_actions`` makes ``predicted_task_success`` the class-balanced
      receiver recall (mean over gold-action classes of per-class recall)
      instead of raw accuracy, which rewards guessing the majority class.
    * ``calibration_trusted=False`` (the registered calibration gate failed
      for the selected perception model) removes the maximally conservative
      high-confidence-error placeholder for candidates without
      high-confidence support, so an uncalibratable confidence signal can
      neither force escalation nor zero the success term.
    """
    if not 0.0 <= float(semantic_acceptance_threshold) <= 1.0:
        raise ValueError("semantic_acceptance_threshold must lie in [0, 1]")
    development = candidate_frame[candidate_frame["sample_id"].isin(set(development_sample_ids))].copy()
    if development.empty:
        raise StudyGateError("No development candidates are available for causal profiles")
    if "source_model" in development:
        source_counts = development.groupby(
            ["mode", "variant"], dropna=False,
        )["source_model"].nunique(dropna=False)
        collisions = source_counts[source_counts.gt(1)]
        if len(collisions):
            raise StudyGateError(
                "Causal candidate identity collision: one (mode, variant) maps "
                f"to multiple source models; examples={list(collisions.index[:5])}"
            )
    aggregate_spec: dict[str, tuple[str, str]] = {
        "observations": ("sample_id", "nunique"),
        "predicted_task_success": ("native_receiver_correct", "mean"),
        "payload_bytes": ("wire_bytes", "median"),
        "generation_ms": ("generation_ms", "median"),
        "energy_j": ("energy_j", "median"),
        "receiver_latency_ms": ("receiver_latency_ms", "median"),
        "receiver_energy_j": ("receiver_energy_j", "median"),
    }
    if "total_processing_latency_ms" in development:
        aggregate_spec["total_processing_latency_ms"] = (
            "total_processing_latency_ms", "median",
        )
    if "total_compute_energy_j" in development:
        aggregate_spec["total_compute_energy_j"] = (
            "total_compute_energy_j", "median",
        )
    profile = development.groupby(["mode", "variant"], as_index=False).agg(**aggregate_spec)
    if "total_processing_latency_ms" not in profile:
        profile["total_processing_latency_ms"] = (
            profile["generation_ms"] + profile["receiver_latency_ms"]
        )
    if "total_compute_energy_j" not in profile:
        profile["total_compute_energy_j"] = (
            profile["energy_j"] + profile["receiver_energy_j"]
        )
    if gold_actions is not None:
        gold = development["sample_id"].astype(str).map(
            {str(key): str(value) for key, value in gold_actions.items()}
        )
        if gold.isna().any():
            raise StudyGateError(
                "Class-balanced profile success lacks gold actions for development samples"
            )
        correct = development["native_receiver_correct"].map(
            lambda value: 1.0 if str(value).strip().lower() in {"1", "1.0", "true"} else 0.0
        )
        balanced = correct.groupby(
            [development["mode"], development["variant"], gold.rename("gold_action")],
        ).mean().groupby(level=[0, 1]).mean()
        profile["predicted_task_success"] = [
            float(balanced.loc[(row.mode, row.variant)])
            for row in profile.itertuples(index=False)
        ]
    # With untrusted calibration no confidence-derived estimate is used.
    missing_support_error = 1.0 if calibration_trusted else 0.0
    high_confidence_error: dict[tuple[str, str], float] = {}
    for (mode, variant), group in development.groupby(["mode", "variant"]):
        confidence = pd.to_numeric(
            group.get("calibrated_correctness_probability", pd.Series(np.nan, index=group.index)),
            errors="coerce",
        )
        eligible = group.loc[confidence.ge(float(semantic_acceptance_threshold))]
        # Missing high-confidence support is deliberately maximally
        # conservative; it cannot make evidence escalation look unnecessary.
        high_confidence_error[(str(mode), str(variant))] = (
            float((~eligible["native_receiver_correct"].astype(bool)).mean())
            if len(eligible) and calibration_trusted else missing_support_error
        )
    profile["predicted_high_confidence_error"] = [
        high_confidence_error[(str(row.mode), str(row.variant))]
        for row in profile.itertuples(index=False)
    ]
    return profile


def _candidate_profiles(rows: pd.DataFrame, now_us: int, profile_table: pd.DataFrame) -> tuple[CandidateProfile, ...]:
    profiles = []
    profile_lookup = profile_table.set_index(["mode", "variant"]).to_dict("index")
    for selected in rows.itertuples(index=False):
        key = (selected.mode, selected.variant)
        if key not in profile_lookup:
            continue
        causal = profile_lookup[key]
        mode_name = selected.mode
        profiles.append(CandidateProfile(
            mode=PayloadMode[mode_name],
            payload_bytes=int(causal["payload_bytes"]),
            generation_us=int(float(causal["generation_ms"]) * 1000),
            energy_j=float(causal["energy_j"]),
            predicted_task_success=float(causal["predicted_task_success"]),
            predicted_high_confidence_error=float(causal["predicted_high_confidence_error"]),
            evidence_strength=0.9 if mode_name in {"L3_SEMANTICS_ROI", "L5_IMAGE_OR_CLIP"} else 0.55,
            state_kind=StateKind.KEYFRAME if mode_name in {"L2_SEMANTICS", "L3_SEMANTICS_ROI"} else StateKind.NONE,
            resulting_state_seq=1 if mode_name in {"L2_SEMANTICS", "L3_SEMANTICS_ROI"} else None,
            available_us=now_us,
            variant=str(selected.variant),
            receiver_latency_us=int(float(causal["receiver_latency_ms"]) * 1000),
            receiver_energy_j=float(causal["receiver_energy_j"]),
        ))
    return tuple(profiles)


def _contiguous_outage_windows(trace: LinkTrace) -> tuple[tuple[int, int], ...]:
    """Return merged zero-service uplink intervals for recovery reporting."""
    windows: list[tuple[int, int]] = []
    for segment in sorted(trace.segments, key=lambda item: item.start_us):
        if segment.service_rate_bps > 0:
            continue
        start_us, end_us = int(segment.start_us), int(segment.end_us)
        if windows and start_us <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end_us))
        else:
            windows.append((start_us, end_us))
    return tuple(windows)


def _annotate_outage_recovery(rows: pd.DataFrame, trace: LinkTrace) -> pd.DataFrame:
    """Attach episode-level outage recovery observations without future policy input.

    This is post-hoc evaluation only. A recovery is the first correct committed
    action from a capture at or after a zero-service interval ends and before the
    next outage starts. The latency clock starts at link restoration. Ordinary
    controllers never receive these annotations.
    """
    result = rows.copy()
    windows = _contiguous_outage_windows(trace)
    result["active_outage_at_capture"] = False
    result["last_outage_end_us"] = np.nan
    result["post_outage_decision_index"] = np.nan
    result["outage_recovery_event"] = False
    result["outage_recovery_latency_ms"] = np.nan
    result["episode_outage_count"] = len(windows)
    result["episode_outages_recovered"] = 0
    result["episode_outages_unrecovered"] = len(windows)
    if result.empty or not windows:
        return result

    capture_times = pd.to_numeric(result["capture_us"], errors="raise").astype(np.int64)
    for start_us, end_us in windows:
        result.loc[capture_times.between(start_us, end_us, inclusive="left"), "active_outage_at_capture"] = True

    recovered = 0
    ordered_windows = list(windows)
    for outage_index, (_, end_us) in enumerate(ordered_windows):
        next_outage_start = (
            ordered_windows[outage_index + 1][0]
            if outage_index + 1 < len(ordered_windows) else math.inf
        )
        eligible_mask = capture_times.ge(end_us) & capture_times.lt(next_outage_start)
        eligible = result.loc[eligible_mask].sort_values(
            ["capture_us", "decision_us", "sample_id"], kind="mergesort",
        )
        if eligible.empty:
            continue
        result.loc[eligible.index, "last_outage_end_us"] = int(end_us)
        result.loc[eligible.index, "post_outage_decision_index"] = np.arange(
            1, len(eligible) + 1, dtype=np.int64,
        )
        successful = eligible[
            eligible["correct_timely_action"].astype(bool)
            & eligible["action_ready_us"].notna()
            & pd.to_numeric(eligible["action_ready_us"], errors="coerce").lt(next_outage_start)
        ].sort_values(["action_ready_us", "capture_us"], kind="mergesort")
        if successful.empty:
            continue
        first_index = successful.index[0]
        action_ready_us = int(successful.loc[first_index, "action_ready_us"])
        result.loc[first_index, "outage_recovery_event"] = True
        result.loc[first_index, "outage_recovery_latency_ms"] = max(
            0.0, (action_ready_us - end_us) / 1000.0,
        )
        recovered += 1
    result["episode_outages_recovered"] = recovered
    result["episode_outages_unrecovered"] = len(windows) - recovered
    return result


def run_paired_policy_trials(
    manifest: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    traces: Sequence[LinkTrace],
    policies: Sequence[RepresentationPolicy],
    replicate_seeds: Sequence[int] = (11, 29),
    causal_profile_table: pd.DataFrame | None = None,
    scientific: bool = False,
    radio_tx_j_per_byte: float = 0.0,
    radio_rx_j_per_byte: float = 0.0,
    radio_energy_included: bool = True,
    semantic_acceptance_threshold: float = 0.80,
) -> pd.DataFrame:
    if not 0.0 <= float(semantic_acceptance_threshold) <= 1.0:
        raise ValueError("semantic_acceptance_threshold must lie in [0, 1]")
    def policy_flag(policy: Any, name: str, default: bool) -> bool:
        return bool(getattr(policy, name, default))

    def context_model_available(sample_id: str) -> bool:
        rows = candidate_frame[candidate_frame["sample_id"].astype(str).eq(str(sample_id))]
        semantic = rows[rows["mode"].isin([
            PayloadMode.L1_ALERT.name,
            PayloadMode.L2_SEMANTICS.name,
            PayloadMode.L3_SEMANTICS_ROI.name,
        ])]
        return bool(
            semantic.get("source_model", pd.Series(dtype=object)).astype(str).map(
                lambda value: value not in {"detector_only", "GENERATED_FIXTURE"}
            ).any()
        )

    context_availability = {
        str(sample_id): context_model_available(str(sample_id))
        for sample_id in manifest["sample_id"].astype(str).unique()
    }

    def context_before_decision(policy: Any, scene: Mapping[str, Any]) -> bool:
        """Apply the registered detector trigger using only signals retained by a policy.

        The previously combined ``context_triggered`` manifest field remains an
        audit column, but cannot be reused by component ablations because it
        does not reveal whether risk, uncertainty, or mission priority fired it.
        """
        if not policy_flag(policy, "uses_context_model", False):
            return False
        if not context_availability.get(str(scene["sample_id"]), False):
            return False
        if policy_flag(policy, "force_context_model", False):
            return True
        mission_risk_trigger = policy_flag(policy, "uses_mission_risk", True) and (
            float(scene.get("detector_risk", 0.0))
            >= float(getattr(policy, "vlm_trigger_risk", 0.45))
            or str(scene.get("mission_profile", "")) in {
                "high_priority", "vulnerable_road_user", "heavy_vehicle",
            }
        )
        uncertainty_trigger = policy_flag(policy, "uses_uncertainty", True) and (
            float(scene.get("detector_uncertainty", 0.0))
            >= float(getattr(policy, "vlm_trigger_uncertainty", 0.35))
        )
        return bool(mission_risk_trigger or uncertainty_trigger)

    def semantic_perception_requirements(
        mode: PayloadMode, source_model: str,
    ) -> tuple[bool, bool]:
        """Return whether a candidate requires detector and context-model work.

        Candidate generation costs deliberately cover representation encoding
        only.  This shared classification keeps causal planning costs aligned
        with the eventual readiness/energy ledger without adding perception to
        that ledger twice.
        """
        semantic = mode in {
            PayloadMode.L1_ALERT, PayloadMode.L2_SEMANTICS,
            PayloadMode.L3_SEMANTICS_ROI,
        }
        detector_semantic = semantic and source_model == "detector_only"
        deployed_semantic = (
            semantic and source_model not in {"detector_only", "GENERATED_FIXTURE"}
        )
        return detector_semantic or deployed_semantic, deployed_semantic

    def perception_cost(
        scene: Mapping[str, Any], *, detector: bool, context: bool,
    ) -> tuple[int, float]:
        """Return serialized perception service time and active energy.

        Queue waiting has no active-stage energy in this model.  Representation
        encoding is intentionally excluded and remains a separate stage.
        """
        service_ms = (
            (float(scene.get("detector_latency_ms", 0.0)) if detector else 0.0)
            + (float(scene.get("context_model_latency_ms", 0.0)) if context else 0.0)
        )
        energy_j = (
            (float(scene.get("detector_energy_j", 0.0)) if detector else 0.0)
            + (float(scene.get("context_model_energy_j", 0.0)) if context else 0.0)
        )
        return int(service_ms * 1000), energy_j

    if causal_profile_table is None:
        if scientific:
            raise StudyGateError("Full policy evaluation requires frozen validation-only candidate profiles")
        development_ids = manifest.loc[~manifest["split"].eq("test"), "sample_id"].astype(str).tolist() if "split" in manifest else manifest["sample_id"].astype(str).tolist()
        causal_profile_table = build_causal_candidate_profiles(candidate_frame, development_ids)
    if scientific:
        required = {
            "trace_time_us", "risk", "uncertainty", "battery_fraction",
            "thermal_c", "detector_latency_ms", "detector_energy_j",
            "detector_risk", "detector_uncertainty", "context_triggered",
            "context_model_latency_ms", "context_model_energy_j",
        }
        if missing := required.difference(manifest.columns):
            raise StudyGateError(f"Full causal episodes require measured/assigned fields: {sorted(missing)}")
        if radio_energy_included:
            if radio_tx_j_per_byte <= 0 or radio_rx_j_per_byte <= 0:
                raise StudyGateError(
                    "Full-system energy accounting requires measured positive TX and RX joules/byte"
                )
        elif radio_tx_j_per_byte != 0 or radio_rx_j_per_byte != 0:
            raise StudyGateError(
                "GPU-board-only energy scope requires radio TX/RX coefficients to be exactly zero"
            )
    episode_manifest = manifest.copy().sort_values("capture_time_us")
    if "trace_time_us" not in episode_manifest:
        episode_manifest["trace_time_us"] = np.arange(len(episode_manifest), dtype=np.int64) * 500_000
    episode_manifest["deadline_budget_us"] = episode_manifest["deadline_us"].astype(np.int64) - episode_manifest["capture_time_us"].astype(np.int64)
    if (episode_manifest["deadline_budget_us"] <= 0).any():
        raise StudyGateError("Every action deadline must be strictly after capture")
    uplinks = [trace for trace in traces if trace.direction is Direction.UPLINK]
    downlinks = {trace.trace_id: trace for trace in traces if trace.direction is Direction.DOWNLINK}
    if scientific:
        if not uplinks:
            raise StudyGateError("Scientific episodes require at least one uplink trace")
        if any(trace.semantics.value != "raw_service" for trace in traces):
            raise StudyGateError("Packet-level scientific simulation requires raw_service traces")
        uncapped_uplinks = [
            trace.trace_id for trace in uplinks
            if any(segment.queue_capacity_bytes is None for segment in trace.segments)
        ]
        if uncapped_uplinks:
            raise StudyGateError(
                "Scientific dynamic trials require queue_capacity_bytes on every "
                f"uplink trace segment: {sorted(set(uncapped_uplinks))}"
            )
        coverage_audit = scientific_episode_coverage_audit(traces, episode_manifest)
        failed_coverage = coverage_audit.loc[~coverage_audit["passed"].astype(bool)]
        if len(failed_coverage):
            raise StudyGateError(
                "Scientific episode trace-coverage gate failed: "
                f"{failed_coverage[['gate', 'observed']].to_dict('records')}"
            )
    outputs: list[pd.DataFrame] = []
    for uplink in uplinks:
        downlink = downlinks.get(uplink.trace_id)
        if downlink is None:
            if scientific:
                raise StudyGateError(f"Scientific trace {uplink.trace_id!r} lacks a measured downlink")
            downlink = LinkTrace(
                uplink.trace_id + "-down", Direction.DOWNLINK, uplink.semantics,
                tuple(dataclasses.replace(segment) for segment in uplink.segments), uplink.end_behavior,
            )
        for replicate_seed in replicate_seeds:
            for policy in policies:
                policy.reset(replicate_seed)
                simulator = SelectiveRepeatSimulator(uplink, downlink, seed=replicate_seed)
                previous_mode: PayloadMode | None = None
                previous_switch_us: int | None = None
                previous_variant: str | None = None
                state_seq = 0
                generated_states: dict[int, dict[str, Any]] = {0: {}}
                # One non-preemptive FCFS perception server per paired episode.
                # Context-aware jobs are reserved below before their decision
                # events; ordinary-policy jobs are reserved causally only after
                # the selected representation is known.
                perception_server_available_us = 0
                # Representation encoding/serialization is a distinct
                # non-preemptive FCFS stage. It is downstream of perception and
                # cannot overlap encoding for another capture.
                representation_server_available_us = 0

                def on_capture(sim: SelectiveRepeatSimulator, now_us: int, scene_data: Mapping[str, Any]):
                    nonlocal previous_mode, previous_switch_us, previous_variant, state_seq
                    nonlocal perception_server_available_us, representation_server_available_us
                    sample_id = str(scene_data["sample_id"])
                    scene_candidates = candidate_frame[candidate_frame["sample_id"] == sample_id]
                    profiles = _candidate_profiles(scene_candidates, now_us, causal_profile_table)
                    by_candidate = {
                        (str(row.mode), str(row.variant)): row
                        for row in scene_candidates.itertuples(index=False)
                    }
                    use_context = context_before_decision(policy, scene_data)
                    detector_before_decision = policy_flag(
                        policy, "uses_context_model", False,
                    )
                    if policy.name == "detector_only_semantics":
                        profiles = tuple(
                            profile for profile in profiles
                            if "detector_only" in profile.variant
                        )
                    elif use_context:
                        profiles = tuple(
                            profile for profile in profiles
                            if str(getattr(
                                by_candidate[(profile.mode.name, profile.variant)],
                                "source_model", "",
                            )) != "detector_only"
                        )
                    elif policy_flag(policy, "uses_context_model", False):
                        # Detector-triggered VLM execution: when the frozen
                        # trigger is false, deployed VLM semantic candidates are
                        # causally unavailable.  Image/feature candidates and
                        # detector-only semantics remain available.
                        profiles = tuple(
                            profile for profile in profiles
                            if not semantic_perception_requirements(
                                profile.mode,
                                str(getattr(
                                    by_candidate[(profile.mode.name, profile.variant)],
                                    "source_model", "",
                                )),
                            )[1]
                        )
                    else:
                        profiles = tuple(profile for profile in profiles if "detector_only" not in profile.variant)
                    if not profiles:
                        raise StudyGateError(f"No causal candidates remain for {policy.name}/{sample_id}")
                    # A pre-decision detector/VLM stage is already represented by
                    # ``now_us``.  For a candidate whose perception has not run,
                    # expose the remaining latency and energy to the controller;
                    # otherwise deadline/energy-aware policies would reason from
                    # encoding-only costs and could choose a causally infeasible
                    # semantic packet.  The selected-row ledger below continues
                    # to charge these stages once, independently of this planning
                    # estimate.
                    planning_profiles = []
                    for profile in profiles:
                        candidate_row = by_candidate[(profile.mode.name, profile.variant)]
                        requires_detector, requires_context = semantic_perception_requirements(
                            profile.mode, str(getattr(candidate_row, "source_model", "")),
                        )
                        pending_detector = requires_detector and not detector_before_decision
                        pending_context = requires_context and not use_context
                        pending_perception = pending_detector or pending_context
                        pending_service_us, pending_energy_j = perception_cost(
                            scene_data, detector=pending_detector, context=pending_context,
                        )
                        pending_queue_wait_us = (
                            max(0, perception_server_available_us - now_us)
                            if pending_perception else 0
                        )
                        perception_ready_us = (
                            now_us + pending_queue_wait_us + pending_service_us
                        )
                        pending_encoding_queue_wait_us = max(
                            0, representation_server_available_us - perception_ready_us,
                        )
                        planning_profiles.append(dataclasses.replace(
                            profile,
                            generation_us=(
                                profile.generation_us + pending_queue_wait_us
                                + pending_service_us
                                + pending_encoding_queue_wait_us
                            ),
                            energy_j=profile.energy_j + pending_energy_j,
                        ))
                    profiles = tuple(planning_profiles)
                    mission_priority_lookup = {
                        "routine": 0.2,
                        "standard": 0.55,
                        "high_priority": 0.9,
                        "general_mobility": 0.55,
                        "heavy_vehicle": 0.80,
                        "vulnerable_road_user": 0.90,
                    }
                    profile_name = str(scene_data["mission_profile"])
                    if profile_name not in mission_priority_lookup:
                        raise StudyGateError(
                            f"Unregistered mission_profile {profile_name!r} in causal episode"
                        )
                    mission_priority = mission_priority_lookup[profile_name]
                    physical_capture_us = int(scene_data["trace_time_us"])
                    deadline_us = physical_capture_us + int(scene_data["deadline_budget_us"])
                    link_estimate = sim.link_estimate(now_us)
                    queue_snapshot = sim.queue_snapshot(now_us)
                    actual_receiver_belief = dataclasses.replace(sim.sender_belief)
                    prior_mode = previous_mode
                    prior_variant = previous_variant
                    prior_switch_us = previous_switch_us
                    actual_age = (
                        None if actual_receiver_belief.receiver_update_us is None
                        else now_us - actual_receiver_belief.receiver_update_us
                    )
                    if policy_flag(policy, "uses_receiver_state", True):
                        receiver_belief = actual_receiver_belief
                        age = actual_age
                    else:
                        receiver_belief = dataclasses.replace(
                            actual_receiver_belief,
                            status="known",
                            feedback_arrival_us=None,
                        )
                        age = None
                    mission_priority = (
                        mission_priority
                        if policy_flag(policy, "uses_mission_risk", True) else 0.0
                    )
                    observed_risk = float(
                        scene_data["risk"] if use_context else
                        scene_data.get("detector_risk", scene_data["risk"])
                        if detector_before_decision else 0.0
                    )
                    if not policy_flag(policy, "uses_mission_risk", True):
                        observed_risk = 0.0
                    observed_uncertainty = float(
                        scene_data["uncertainty"] if use_context else
                        scene_data.get("detector_uncertainty", scene_data["uncertainty"])
                        if detector_before_decision else 1.0
                    )
                    if not policy_flag(policy, "uses_uncertainty", True):
                        observed_uncertainty = 0.0
                        profiles = tuple(
                            dataclasses.replace(
                                profile, predicted_high_confidence_error=0.0,
                            )
                            for profile in profiles
                        )
                    obs = ControllerObservation(
                        now_us, physical_capture_us, deadline_us, link_estimate,
                        queue_snapshot, receiver_belief, mission_priority,
                        observed_risk, observed_uncertainty, age,
                        float(scene_data.get("battery_fraction", 0.8)), float(scene_data.get("thermal_c", 55.0)),
                        previous_mode, previous_switch_us, previous_variant,
                    )
                    if isinstance(policy, LookaheadReferencePolicy):
                        lookahead_outcomes = {}
                        for candidate in profiles:
                            candidate_row = by_candidate[(candidate.mode.name, candidate.variant)]
                            try:
                                span = uplink.serialize(
                                    now_us + candidate.generation_us,
                                    candidate.payload_bytes * 8,
                                )
                                segment = uplink.condition_at(max(span.start_us, span.end_us - 1))
                                ready = span.end_us + segment.base_one_way_delay_us + candidate.receiver_latency_us
                                timely = ready <= deadline_us
                            except (EOFError, TimeoutError, ValueError):
                                timely = False
                            lookahead_outcomes[f"{candidate.mode.name}:{candidate.variant}"] = {
                                "correct_timely": bool(
                                    timely and str(candidate_row.receiver_action) == str(scene_data["gold_action"])
                                )
                            }
                        plan = policy.select(
                            obs, profiles,
                            LookaheadContext(
                                uplink, scene_data["gold_action"], lookahead_outcomes,
                            ),
                        )
                    else:
                        plan = policy.select(obs, profiles)
                    selected = by_candidate[(plan.mode.name, plan.variant)]
                    switched = previous_mode is not None and (
                        plan.mode is not previous_mode or plan.variant != previous_variant
                    )
                    if previous_mode is None or switched:
                        previous_switch_us = now_us
                    previous_mode = plan.mode
                    previous_variant = plan.variant
                    payload = bytes.fromhex(selected.payload_blob)
                    message_id = int(hashlib.sha256(f"{sample_id}-{uplink.trace_id}-{replicate_seed}".encode()).hexdigest()[:12], 16)
                    resulting_seq = None
                    base_seq = None
                    state_kind = plan.state_kind
                    wire_format = registered_wire_format(selected.variant)
                    full_semantic_state = None
                    if plan.mode is PayloadMode.L2_SEMANTICS and wire_format is not None:
                        full_semantic_state = decode_mapping(payload, wire_format)
                        confirmed = sim.sender_belief.confirmed_seq
                        if (
                            policy_flag(policy, "uses_acknowledged_deltas", True)
                            and sim.sender_belief.status == "known"
                            and confirmed in generated_states
                            and confirmed > 0
                        ):
                            delta = make_state_delta(generated_states[confirmed], full_semantic_state)
                            payload, delta_wire_format = encode_mapping(
                                delta,
                                scientific=scientific,
                                wire_format=wire_format,
                            )
                            if delta_wire_format != wire_format:
                                raise StudyGateError(
                                    "Semantic delta encoding differs from its declared wire format"
                                )
                            state_kind = StateKind.DELTA
                            base_seq = confirmed
                        else:
                            state_kind = StateKind.KEYFRAME
                    if state_kind is not StateKind.NONE:
                        state_seq += 1
                        resulting_seq = state_seq
                        if full_semantic_state is not None:
                            generated_states[state_seq] = full_semantic_state
                    source_model = str(getattr(selected, "source_model", ""))
                    requires_detector, requires_context = semantic_perception_requirements(
                        plan.mode, source_model,
                    )
                    detector_used = detector_before_decision or requires_detector
                    context_used = use_context or requires_context
                    pending_detector = requires_detector and not detector_before_decision
                    pending_context = requires_context and not use_context
                    postdecision_perception = pending_detector or pending_context
                    postdecision_service_us, _ = perception_cost(
                        scene_data, detector=pending_detector, context=pending_context,
                    )
                    if postdecision_perception:
                        perception_start_us = max(now_us, perception_server_available_us)
                        perception_queue_wait_us = perception_start_us - now_us
                        perception_done_us = perception_start_us + postdecision_service_us
                        perception_server_available_us = perception_done_us
                        representation_start_us = perception_done_us
                        perception_service_us = postdecision_service_us
                    else:
                        perception_queue_wait_us = int(
                            scene_data.get("_predecision_perception_queue_wait_us", 0)
                        )
                        perception_service_us = int(
                            scene_data.get("_predecision_perception_service_us", 0)
                        )
                        perception_start_us = scene_data.get("_predecision_perception_start_us")
                        perception_done_us = scene_data.get("_predecision_perception_done_us")
                        representation_start_us = now_us
                    context_energy_j = float(scene_data.get("context_model_energy_j", 0.0)) if context_used else 0.0
                    detector_energy_j = float(scene_data.get("detector_energy_j", 0.0)) if detector_used else 0.0
                    representation_generation_us = int(float(selected.generation_ms) * 1000)
                    representation_encoding_start_us = max(
                        representation_start_us, representation_server_available_us,
                    )
                    representation_queue_wait_us = (
                        representation_encoding_start_us - representation_start_us
                    )
                    representation_done_us = (
                        representation_encoding_start_us + representation_generation_us
                    )
                    representation_server_available_us = representation_done_us
                    expiry_us = deadline_us if state_kind is StateKind.NONE else deadline_us + 2_000_000
                    message = AppMessage(
                        policy.name, sample_id, message_id, plan.mode, physical_capture_us,
                        representation_done_us, deadline_us,
                        expiry_us, plan.priority, payload, state_kind,
                        base_seq, resulting_seq, plan.reliability,
                    )
                    metadata = {
                        "policy": policy.name, "visual_group_id": scene_data["visual_group_id"],
                        "sample_id": sample_id, "mission_profile": scene_data["mission_profile"],
                        "trace_id": uplink.trace_id, "trace_family": uplink.trace_id.rsplit("-", 1)[0],
                        "replicate_seed": replicate_seed, "gold_action": scene_data["gold_action"],
                        "receiver_action": str(selected.receiver_action),
                        "representation_energy_j": float(selected.energy_j),
                        "representation_generation_ms": representation_generation_us / 1000,
                        "representation_queue_wait_ms": representation_queue_wait_us / 1000,
                        "representation_start_us": representation_encoding_start_us,
                        "representation_done_us": representation_done_us,
                        "receiver_energy_j": float(selected.receiver_energy_j),
                        "receiver_latency_us": int(float(selected.receiver_latency_ms) * 1000),
                        "context_model_energy_j": context_energy_j,
                        "context_model_latency_ms": float(scene_data.get("context_model_latency_ms", 0.0)) if context_used else 0.0,
                        "detector_latency_ms": float(scene_data.get("detector_latency_ms", 0.0)) if detector_used else 0.0,
                        "detector_energy_j": detector_energy_j,
                        "perception_queue_wait_ms": perception_queue_wait_us / 1000,
                        "perception_service_ms": perception_service_us / 1000,
                        "perception_start_us": perception_start_us,
                        "perception_done_us": perception_done_us,
                        "selected_mode": plan.mode.name, "variant": plan.variant,
                        "application_payload_bytes": len(payload),
                        "state_kind": state_kind.value,
                        "base_state_seq": base_seq,
                        "resulting_state_seq": resulting_seq,
                        "calibrated_correctness_probability": float(
                            getattr(selected, "calibrated_correctness_probability", np.nan)
                        ),
                        "evidence_escalated": plan.mode in {
                            PayloadMode.L3_SEMANTICS_ROI, PayloadMode.L5_IMAGE_OR_CLIP
                        },
                        "reason_code": plan.reason_code, "critical": bool(scene_data["critical"]),
                        "cohort": str(scene_data.get("cohort", "development")),
                        "sufficient_evidence": bool(scene_data.get("sufficient_evidence", True)),
                        "ood": bool(scene_data.get("ood", False)), "risk": float(scene_data["risk"]),
                        "uncertainty": float(scene_data["uncertainty"]), "switches": int(switched),
                        # Frozen, causal controller observation at decision time.
                        # These fields are audit outputs only and never feed a
                        # later decision in the same episode.
                        "decision_us": now_us,
                        "decision_candidate_count": len(profiles),
                        "decision_previous_mode": prior_mode.name if prior_mode is not None else None,
                        "decision_previous_variant": prior_variant,
                        "decision_previous_dwell_ms": (
                            (now_us - prior_switch_us) / 1000.0
                            if prior_switch_us is not None else np.nan
                        ),
                        "decision_link_as_of_us": link_estimate.as_of_us,
                        "decision_link_feedback_samples": link_estimate.sample_count,
                        "decision_estimated_service_rate_bps": link_estimate.service_rate_bps,
                        "decision_estimated_one_way_delay_us": link_estimate.one_way_delay_us,
                        "decision_estimated_loss_probability": link_estimate.loss_probability,
                        "decision_estimated_rto_us": link_estimate.rto_us,
                        "decision_queue_as_of_us": queue_snapshot.as_of_us,
                        "decision_queue_pending_bytes_by_priority_json": json.dumps(
                            list(queue_snapshot.pending_bytes_by_priority),
                            separators=(",", ":"),
                        ),
                        "decision_queue_messages_by_priority_json": json.dumps(
                            list(queue_snapshot.messages_by_priority),
                            separators=(",", ":"),
                        ),
                        "decision_queue_oldest_ready_us_by_priority_json": json.dumps(
                            list(queue_snapshot.oldest_ready_us_by_priority),
                            separators=(",", ":"),
                        ),
                        "decision_queue_pending_bytes": queue_snapshot.pending_bytes,
                        "decision_queue_inflight_bytes": queue_snapshot.inflight_bytes,
                        "decision_queue_outstanding_unacked_bytes": queue_snapshot.outstanding_unacked_bytes,
                        "decision_queue_messages": int(sum(queue_snapshot.messages_by_priority)),
                        "decision_queue_oldest_age_ms": (
                            max(
                                0, now_us - min(
                                    value for value in queue_snapshot.oldest_ready_us_by_priority
                                    if value is not None
                                ),
                            ) / 1000.0
                            if any(value is not None for value in queue_snapshot.oldest_ready_us_by_priority)
                            else 0.0
                        ),
                        "decision_backlog_bytes": (
                            queue_snapshot.pending_bytes
                            + queue_snapshot.outstanding_unacked_bytes
                        ),
                        "decision_receiver_belief_status": receiver_belief.status,
                        "decision_receiver_confirmed_seq": receiver_belief.confirmed_seq,
                        "decision_receiver_feedback_arrival_us": receiver_belief.feedback_arrival_us,
                        "decision_receiver_update_us": receiver_belief.receiver_update_us,
                        "decision_receiver_age_ms": age / 1000.0 if age is not None else np.nan,
                        "audit_actual_receiver_belief_status": actual_receiver_belief.status,
                        "audit_actual_receiver_confirmed_seq": actual_receiver_belief.confirmed_seq,
                        "audit_actual_receiver_feedback_arrival_us": actual_receiver_belief.feedback_arrival_us,
                        "audit_actual_receiver_update_us": actual_receiver_belief.receiver_update_us,
                        "audit_actual_receiver_age_ms": (
                            actual_age / 1000.0 if actual_age is not None else np.nan
                        ),
                        "decision_sender_sequence_gap": max(
                            0, state_seq - receiver_belief.confirmed_seq,
                        ),
                        "decision_mission_priority": obs.mission_priority,
                        "decision_risk": obs.risk,
                        "decision_uncertainty": obs.uncertainty,
                        "decision_battery_fraction": obs.battery_fraction,
                        "decision_thermal_c": obs.thermal_c,
                        "decision_remaining_deadline_ms": obs.remaining_deadline_us / 1000.0,
                        "decision_context_available": bool(use_context),
                        "decision_causal_input_hash": hash_object({
                            "observation": obs,
                            "candidate_profiles": profiles,
                        }),
                        "wire_format": wire_format,
                    }
                    return message, metadata

                captures = []
                predecision_server_available_us = 0
                capture_manifest = episode_manifest.sort_values(
                    ["trace_time_us", "capture_time_us", "sample_id"], kind="mergesort",
                )
                detector_before_decision = policy_flag(
                    policy, "uses_context_model", False,
                )
                for row in capture_manifest.itertuples(index=False):
                    scene = row._asdict()
                    physical_capture_us = int(row.trace_time_us)
                    if detector_before_decision:
                        use_context = context_before_decision(policy, scene)
                        service_us, _ = perception_cost(
                            scene, detector=True, context=use_context,
                        )
                        server_start_us = max(
                            physical_capture_us, predecision_server_available_us,
                        )
                        decision_us = server_start_us + service_us
                        predecision_server_available_us = decision_us
                        scene["_predecision_perception_queue_wait_us"] = (
                            server_start_us - physical_capture_us
                        )
                        scene["_predecision_perception_service_us"] = service_us
                        scene["_predecision_perception_start_us"] = server_start_us
                        scene["_predecision_perception_done_us"] = decision_us
                    else:
                        decision_us = physical_capture_us
                    captures.append((decision_us, scene))
                network_rows, _ = simulator.run(
                    [], captures=captures, capture_handler=on_capture,
                    feedback_handler=lambda feedback, now: policy.on_feedback(feedback, now),
                )
                network_rows["payload_valid"] = network_rows["full_message_received"].astype(bool)
                # A duplicate can be a valid idempotent delivery, but stale,
                # base-mismatched, expired, and undecodable states fail the
                # mission endpoint.
                network_rows["state_valid"] = network_rows["apply_status"].isin(["applied", "duplicate"])
                outputs.append(network_rows)
    combined = pd.concat(outputs, ignore_index=True)
    combined["retry_exhausted"] = combined["retry_exhausted"].astype(bool)
    combined["receiver_energy_j_charged"] = (
        combined["receiver_energy_j"]
        * combined["full_message_received"].astype(bool)
    )
    combined["energy_j"] = (
        combined["representation_energy_j"] + combined["detector_energy_j"]
        + combined["receiver_energy_j_charged"]
        + combined["context_model_energy_j"]
        + (combined["original_uplink_bytes"] + combined["retry_uplink_bytes"]) * radio_tx_j_per_byte
        + combined["downlink_control_bytes"] * radio_rx_j_per_byte
    )
    combined = add_primary_endpoint(combined)
    confidence = pd.to_numeric(combined["calibrated_correctness_probability"], errors="coerce")
    combined["high_confidence_semantic_error"] = (
        confidence.ge(float(semantic_acceptance_threshold))
        & combined["selected_mode"].isin(["L1_ALERT", "L2_SEMANTICS"])
        & ~combined["correct_action"].astype(bool)
    )
    annotated_episodes = []
    uplink_lookup = {trace.trace_id: trace for trace in uplinks}
    for (policy_name, trace_id, replicate_seed), episode in combined.groupby(
        ["policy", "trace_id", "replicate_seed"], sort=False, dropna=False,
    ):
        del policy_name, replicate_seed
        trace = uplink_lookup.get(str(trace_id))
        if trace is None:
            raise StudyGateError(f"Cannot recover outage metadata for unknown trace {trace_id!r}")
        annotated_episodes.append(_annotate_outage_recovery(episode, trace))
    combined = pd.concat(annotated_episodes, ignore_index=True) if annotated_episodes else combined
    return combined


def run_paired_policy_trials_sharded(
    manifest: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    traces: Sequence[LinkTrace],
    policies: Sequence[RepresentationPolicy],
    replicate_seeds: Sequence[int],
    causal_profile_table: pd.DataFrame,
    ctx: RunContext,
    *,
    scientific: bool,
    radio_tx_j_per_byte: float = 0.0,
    radio_rx_j_per_byte: float = 0.0,
    radio_energy_included: bool = True,
    semantic_acceptance_threshold: float = 0.80,
    return_partition_index: bool = False,
) -> pd.DataFrame:
    """Run dynamic trials in coarse, resumable policy/trace/seed partitions.

    Episode isolation remains exact: every episode is passed to
    :func:`run_paired_policy_trials` separately, so controller, queue, sender,
    receiver, and outage state are reset at the registered episode boundary.
    The durable stage granularity is deliberately coarser, however.  A stage
    contains *all* policies and episodes for one ``(trace, seed)`` tuple.  Thus
    the number of checksummed stage artifacts is O(trace x seed), not
    O(episode x policy x trace x seed).

    Completed partition frames are released immediately.  By default, their
    verified CSV artifacts are concatenated into one aggregate with streaming
    gzip I/O and read once for the historical public DataFrame API.  Scientific
    paper runners set ``return_partition_index=True`` and receive only the
    checksum-bound partition index; an out-of-core reducer then consumes those
    partitions without materializing the campaign-wide trial table.
    """

    profile_key = hash_object(causal_profile_table.to_dict("records"))
    if "episode_id" in manifest:
        episode_groups = list(manifest.groupby("episode_id", sort=True))
    elif {"visual_group_id", "mission_profile"}.issubset(manifest.columns):
        # Canonical real-data manifests use these two frozen, outcome-blind
        # control-plane fields even when an explicit episode_id is omitted.
        episode_groups = list(manifest.groupby(
            ["visual_group_id", "mission_profile"], sort=True, dropna=False,
        ))
    else:
        episode_groups = [("single", manifest)]

    episodes: list[tuple[str, pd.DataFrame, pd.DataFrame, str, str]] = []
    all_manifest_ids = set(manifest["sample_id"].astype(str))
    candidate_ids = set(candidate_frame["sample_id"].astype(str))
    if candidate_ids != all_manifest_ids:
        missing = sorted(all_manifest_ids - candidate_ids)
        extra = sorted(candidate_ids - all_manifest_ids)
        raise StudyGateError(
            "Dynamic candidate coverage differs from the registered manifest: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    for episode_id, episode_manifest in episode_groups:
        episode_ids = set(episode_manifest["sample_id"].astype(str))
        episode_candidates = candidate_frame[
            candidate_frame["sample_id"].astype(str).isin(episode_ids)
        ].copy()
        if set(episode_candidates["sample_id"].astype(str)) != episode_ids:
            raise StudyGateError(
                f"Dynamic episode {episode_id!r} lacks candidate coverage"
            )
        manifest_key = hash_object(
            episode_manifest.sort_values("sample_id").to_dict("records")
        )
        candidate_key = hash_object(
            episode_candidates.sort_values(
                ["sample_id", "mode", "variant"], kind="mergesort",
            ).to_dict("records")
        )
        episodes.append((
            str(episode_id), episode_manifest.copy(), episode_candidates,
            manifest_key, candidate_key,
        ))

    trace_pairs: list[tuple[LinkTrace, list[LinkTrace], str]] = []
    for trace in traces:
        if trace.direction is not Direction.UPLINK:
            continue
        matching_downlink = [
            item for item in traces
            if item.direction is Direction.DOWNLINK and item.trace_id == trace.trace_id
        ]
        pair = [trace, *matching_downlink]
        trace_pairs.append((
            trace, pair,
            hash_object(traces_to_frame(pair).to_dict("records")),
        ))

    partition_artifacts: list[Path] = []
    partition_records: list[dict[str, Any]] = []
    episode_design_key = hash_object([
        {"episode_id": episode_id, "manifest": manifest_key, "candidates": candidate_key}
        for episode_id, _, _, manifest_key, candidate_key in episodes
    ])
    policy_design = [
        {
            "policy": str(policy.name),
            "parameters": {
                parameter: value
                for parameter, value in vars(policy).items()
                if parameter != "run_seed"
            },
        }
        for policy in policies
    ]
    for trace, trace_pair, trace_key in trace_pairs:
        for seed in replicate_seeds:
            partition_identity = {
                "episode_design": episode_design_key,
                "profiles": profile_key,
                "trace": trace_key,
                "trace_id": str(trace.trace_id),
                "seed": int(seed),
                "policies": policy_design,
                "scientific": bool(scientific),
                "radio_tx_j_per_byte": float(radio_tx_j_per_byte),
                "radio_rx_j_per_byte": float(radio_rx_j_per_byte),
                "radio_energy_included": bool(radio_energy_included),
                "semantic_acceptance_threshold": float(
                    semantic_acceptance_threshold
                ),
            }
            partition_token = hash_object(partition_identity)[:16]
            name = f"04_policy_partition_{partition_token}"

            def compute_partition(
                trace_pair: Sequence[LinkTrace] = trace_pair,
                seed: int = int(seed),
            ) -> pd.DataFrame:
                episode_outputs: list[pd.DataFrame] = []
                for policy in policies:
                    for episode_id, episode_manifest, episode_candidates, _, _ in episodes:
                        part = run_paired_policy_trials(
                            episode_manifest, episode_candidates, trace_pair,
                            [policy], (seed,), causal_profile_table,
                            scientific=scientific,
                            radio_tx_j_per_byte=radio_tx_j_per_byte,
                            radio_rx_j_per_byte=radio_rx_j_per_byte,
                            radio_energy_included=radio_energy_included,
                            semantic_acceptance_threshold=(
                                semantic_acceptance_threshold
                            ),
                        )
                        if len(part):
                            part = part.copy()
                            if "episode_id" not in part.columns:
                                part["episode_id"] = str(episode_id)
                            episode_outputs.append(part)
                return (
                    pd.concat(episode_outputs, ignore_index=True)
                    if episode_outputs else pd.DataFrame()
                )

            # ``frame_stage`` performs checksum validation on resume.  Do not
            # retain its DataFrame: the durable artifact is the source for the
            # single final streaming concatenation below.
            partition_frame = frame_stage(
                ctx, name, partition_identity, compute_partition,
            )
            del partition_frame
            record = ctx.stage_record(name)
            if not record or record.get("status") != "COMPLETE":
                raise StudyGateError(f"Dynamic partition {name!r} did not complete")
            artifact = Path(str(record["artifact"]))
            if not artifact.is_file() or hash_file(artifact) != str(record["checksum"]):
                raise StudyGateError(f"Dynamic partition {name!r} failed checksum validation")
            partition_artifacts.append(artifact)
            partition_records.append({
                "stage": name,
                "artifact": str(artifact),
                "artifact_path": str(artifact),
                "sha256": str(record["checksum"]),
                "artifact_sha256": str(record["checksum"]),
                "rows": int(record.get("rows") or 0),
                "row_count": int(record.get("rows") or 0),
                "policies": [str(policy.name) for policy in policies],
                "trace_id": str(trace.trace_id),
                "replicate_seed": int(seed),
                "episodes": int(len(episodes)),
            })

    if not partition_artifacts:
        return pd.DataFrame()

    if return_partition_index:
        index = pd.DataFrame(partition_records).sort_values(
            ["trace_id", "replicate_seed", "stage"], kind="mergesort",
        ).reset_index(drop=True)
        if index["stage"].astype(str).duplicated().any():
            raise StudyGateError("Dynamic partition index contains duplicate stages")
        return index

    aggregate_key = hash_object({
        "version": "semuav.dynamic-coarse-partitions.v1",
        "partitions": partition_records,
    })
    aggregate_dir = ctx.stage_dir("04_policy_partition_aggregate")
    aggregate_path = aggregate_dir / f"{aggregate_key}.csv.gz"
    index_path = aggregate_dir / f"{aggregate_key}.index.json"
    reusable = False
    if aggregate_path.is_file() and index_path.is_file():
        previous = json.loads(index_path.read_text(encoding="utf-8"))
        reusable = bool(
            previous.get("aggregate_key") == aggregate_key
            and previous.get("partitions") == partition_records
            and previous.get("aggregate_sha256") == hash_file(aggregate_path)
        )
    if not reusable:
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{aggregate_path.name}.", suffix=".csv.gz", dir=aggregate_dir,
        )
        os.close(fd)
        try:
            expected_header: str | None = None
            with gzip.open(tmp_name, "wt", encoding="utf-8", newline="") as sink:
                for artifact in partition_artifacts:
                    with gzip.open(artifact, "rt", encoding="utf-8", newline="") as source:
                        header = source.readline()
                        if expected_header is None:
                            expected_header = header
                            sink.write(header)
                        elif header != expected_header:
                            raise StudyGateError(
                                "Dynamic partition columns differ; refusing lossy concatenation"
                            )
                        shutil.copyfileobj(source, sink, length=1 << 20)
            os.replace(tmp_name, aggregate_path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        atomic_write_json(index_path, {
            "version": "semuav.dynamic-coarse-partitions.v1",
            "aggregate_key": aggregate_key,
            "aggregate_artifact": str(aggregate_path),
            "aggregate_sha256": hash_file(aggregate_path),
            "partition_count": len(partition_records),
            "episode_count": len(episodes),
            "partitions": partition_records,
        })
    combined = pd.read_csv(aggregate_path, compression="gzip")
    if not combined.empty:
        assert set(combined["policy"]) == {policy.name for policy in policies}
    return combined


def experiment4_controller_behavior(
    policy_trials: pd.DataFrame,
    *,
    reference_policy: str = "heuristic_lookahead_reference",
    semantic_acceptance_threshold: float = 0.80,
    oracle_policy: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create registered controller-behavior detail and a policy summary.

    All ordinary-policy inputs were captured at decision time by
    :func:`run_paired_policy_trials`. The privileged one-step lookahead outcome
    is joined only after all episodes finish, on the exact paired
    observation/trace/seed key.  It is a heuristic reference—not an oracle or
    upper bound—because it does not replay counterfactual queues, retries,
    receiver state, or later decisions.

    ``wasted_uplink_wire_bytes`` counts every retransmitted uplink byte, plus
    original uplink bytes for a message that misses the primary endpoint.
    ``aborted_generated_payload_bytes`` counts application payload generated for
    a message that never becomes a complete valid receiver state. A completed
    but late/wrong message is wasted on wire, not classified as aborted.
    """
    if not 0.0 <= semantic_acceptance_threshold <= 1.0:
        raise ValueError("semantic_acceptance_threshold must lie in [0, 1]")
    if oracle_policy is not None:
        warnings.warn(
            "oracle_policy is deprecated; use reference_policy. The selected "
            "policy is reported only as a heuristic lookahead reference.",
            DeprecationWarning,
            stacklevel=2,
        )
        reference_policy = oracle_policy
    # Scientific VisDrone replays materialize ``episode_id`` before invoking
    # this report.  Older/generic callers have one continuous stream and must
    # retain that legacy interpretation: deriving an episode from each visual
    # group here would turn every decision into a one-row episode and erase
    # switches, dwell times, and outage-recovery accounting.
    policy_trials = policy_trials.copy()
    if "episode_id" not in policy_trials.columns:
        policy_trials["episode_id"] = "legacy_stream"
    required = {
        "policy", "sample_id", "visual_group_id", "mission_profile",
        "trace_id", "replicate_seed", "episode_id", "decision_us", "selected_mode",
        "variant", "application_payload_bytes", "full_message_received",
        "state_valid", "state_kind", "apply_status", "correct_action",
        "correct_timely_action", "original_uplink_bytes", "retry_uplink_bytes",
        "total_wire_bytes", "energy_j", "decision_queue_pending_bytes",
        "decision_queue_outstanding_unacked_bytes", "decision_queue_messages",
        "decision_queue_oldest_age_ms", "decision_backlog_bytes",
        "decision_sender_sequence_gap", "episode_outage_count",
        "episode_outages_recovered", "episode_outages_unrecovered",
        "outage_recovery_event", "outage_recovery_latency_ms",
        "calibrated_correctness_probability", "high_confidence_semantic_error",
    }
    if missing := required.difference(policy_trials.columns):
        raise ValueError(f"Experiment 4 controller report missing columns: {sorted(missing)}")
    if policy_trials.empty:
        raise ValueError("Experiment 4 controller report cannot summarize an empty frame")
    assert_paired_design(policy_trials)

    episode_keys = ["policy", "trace_id", "replicate_seed", "episode_id"]
    pair_keys = ["sample_id", "trace_id", "replicate_seed", "episode_id"]
    detail = policy_trials.copy().sort_values(
        [*episode_keys, "decision_us", "sample_id"], kind="mergesort",
    ).reset_index(drop=True)
    detail["representation_key"] = (
        detail["selected_mode"].astype(str) + ":" + detail["variant"].fillna("").astype(str)
    )
    grouped_representation = detail.groupby(episode_keys, sort=False)["representation_key"]
    previous_representation = grouped_representation.shift(1)
    two_back_representation = grouped_representation.shift(2)
    decision_order = detail.groupby(episode_keys, sort=False).cumcount()
    detail["decision_order"] = decision_order
    detail["mode_switch_event"] = (
        decision_order.gt(0) & detail["representation_key"].ne(previous_representation)
    )
    detail["oscillation_event"] = (
        decision_order.ge(2)
        & detail["representation_key"].eq(two_back_representation)
        & detail["representation_key"].ne(previous_representation)
    )
    detail["dwell_run_id"] = detail.groupby(episode_keys, sort=False)[
        "mode_switch_event"
    ].cumsum().astype(np.int64)

    run_keys = [*episode_keys, "dwell_run_id"]
    dwell_runs = detail.groupby(run_keys, sort=False).agg(
        dwell_mode=("selected_mode", "first"),
        dwell_variant=("variant", "first"),
        dwell_start_us=("decision_us", "min"),
        dwell_decisions=("decision_order", "size"),
    ).reset_index()
    dwell_runs["next_dwell_start_us"] = dwell_runs.groupby(
        episode_keys, sort=False,
    )["dwell_start_us"].shift(-1)
    dwell_runs["completed_dwell"] = dwell_runs["next_dwell_start_us"].notna()
    dwell_runs["completed_dwell_ms"] = np.where(
        dwell_runs["completed_dwell"],
        (
            pd.to_numeric(dwell_runs["next_dwell_start_us"], errors="coerce")
            - pd.to_numeric(dwell_runs["dwell_start_us"], errors="raise")
        ) / 1000.0,
        np.nan,
    )
    if dwell_runs["completed_dwell_ms"].dropna().lt(0).any():
        raise StudyGateError("Experiment 4 decision timestamps are not monotonic within an episode")
    detail = detail.merge(
        dwell_runs[[*run_keys, "dwell_decisions", "completed_dwell", "completed_dwell_ms"]],
        on=run_keys, how="left", validate="many_to_one",
    )

    if (
        reference_policy == "heuristic_lookahead_reference"
        and not detail["policy"].eq(reference_policy).any()
        and detail["policy"].eq("future_oracle").any()
    ):
        warnings.warn(
            "future_oracle is a deprecated policy label; treating it as the "
            "heuristic lookahead reference.",
            DeprecationWarning,
            stacklevel=2,
        )
        reference_policy = "future_oracle"
    reference = detail[detail["policy"].eq(reference_policy)][
        [*pair_keys, "correct_timely_action"]
    ].rename(columns={
        "correct_timely_action": "paired_lookahead_reference_correct_timely_action",
    })
    if reference.empty:
        raise StudyGateError(
            "Experiment 4 controller report requires paired heuristic "
            f"lookahead reference policy {reference_policy!r}"
        )
    if reference.duplicated(pair_keys).any():
        raise StudyGateError(
            "Heuristic-lookahead rows are duplicated on the paired trial key"
        )
    detail = detail.merge(reference, on=pair_keys, how="left", validate="many_to_one")
    if detail["paired_lookahead_reference_correct_timely_action"].isna().any():
        raise StudyGateError(
            "Some controller decisions lack a paired heuristic-lookahead outcome"
        )
    own_success = detail["correct_timely_action"].astype(bool)
    reference_success = detail[
        "paired_lookahead_reference_correct_timely_action"
    ].astype(bool)
    detail["lookahead_reference_signed_gap"] = (
        reference_success.astype(int) - own_success.astype(int)
    )
    detail["lookahead_reference_missed_opportunity"] = (
        reference_success & ~own_success
    )
    # Inversions are expected to be possible: the heuristic is expressly not
    # an upper bound.  Reporting them makes that limitation measurable.
    detail["lookahead_reference_inversion"] = own_success & ~reference_success

    complete_valid = (
        detail["full_message_received"].astype(bool)
        & detail["state_valid"].astype(bool)
    )
    detail["aborted_generated_payload_bytes"] = np.where(
        complete_valid, 0, pd.to_numeric(detail["application_payload_bytes"], errors="raise"),
    )
    detail["wasted_uplink_wire_bytes"] = (
        pd.to_numeric(detail["retry_uplink_bytes"], errors="raise")
        + np.where(
            own_success, 0,
            pd.to_numeric(detail["original_uplink_bytes"], errors="raise"),
        )
    )
    detail["stale_state_event"] = detail["apply_status"].eq("stale")
    detail["sequence_gap_rejection_event"] = detail["apply_status"].eq("base_mismatch")
    detail["stateful_decision"] = ~detail["state_kind"].eq(StateKind.NONE.value)
    detail["decision_sequence_gap_event"] = pd.to_numeric(
        detail["decision_sender_sequence_gap"], errors="raise",
    ).gt(0)
    confidence = pd.to_numeric(
        detail["calibrated_correctness_probability"], errors="coerce",
    )
    detail["semantic_prediction"] = detail["selected_mode"].isin(
        [PayloadMode.L1_ALERT.name, PayloadMode.L2_SEMANTICS.name]
    )
    detail["semantic_accepted"] = (
        detail["semantic_prediction"] & confidence.ge(semantic_acceptance_threshold)
    )
    detail["semantic_accepted_wrong"] = (
        detail["semantic_accepted"] & ~detail["correct_action"].astype(bool)
    )
    detail["semantic_accepted_mission_failure"] = (
        detail["semantic_accepted"] & ~own_success
    )

    summary_rows: list[dict[str, Any]] = []
    all_modes = tuple(mode.name for mode in PayloadMode)
    for policy_name, group in detail.groupby("policy", sort=True):
        episode_sizes = group.groupby(
            ["trace_id", "replicate_seed", "episode_id"], sort=False,
        ).size()
        switch_opportunities = int(np.maximum(episode_sizes.to_numpy() - 1, 0).sum())
        oscillation_opportunities = int(np.maximum(episode_sizes.to_numpy() - 2, 0).sum())
        policy_runs = dwell_runs[dwell_runs["policy"].eq(policy_name)]
        completed_runs = policy_runs[policy_runs["completed_dwell"].astype(bool)]
        accepted = group[group["semantic_accepted"].astype(bool)]
        semantic = group[group["semantic_prediction"].astype(bool)]
        stateful = group[group["stateful_decision"].astype(bool)]
        recovery_events = group[group["outage_recovery_event"].astype(bool)]
        outage_count = int(
            group.groupby(["trace_id", "replicate_seed", "episode_id"], sort=False)[
                "episode_outage_count"
            ].first().sum()
        )
        recovered_count = int(len(recovery_events))
        if recovered_count != int(
            group.groupby(["trace_id", "replicate_seed", "episode_id"], sort=False)[
                "episode_outages_recovered"
            ].first().sum()
        ):
            raise StudyGateError("Outage recovery annotations are internally inconsistent")
        uplink_wire_bytes = float(
            (group["original_uplink_bytes"] + group["retry_uplink_bytes"]).sum()
        )
        payload_bytes = float(group["application_payload_bytes"].sum())
        correct_count = int(group["correct_timely_action"].astype(bool).sum())
        total_energy_j = float(group["energy_j"].sum())
        row: dict[str, Any] = {
            "policy": policy_name,
            "episodes": int(len(episode_sizes)),
            "decisions": int(len(group)),
            "correct_timely_actions": correct_count,
            "total_energy_j": total_energy_j,
            "joules_per_correct_timely_action": (
                total_energy_j / correct_count if correct_count else np.nan
            ),
            "mode_switches": int(group["mode_switch_event"].sum()),
            "switch_opportunities": switch_opportunities,
            "mode_switch_rate": (
                float(group["mode_switch_event"].sum()) / switch_opportunities
                if switch_opportunities else 0.0
            ),
            "oscillations_aba": int(group["oscillation_event"].sum()),
            "oscillation_opportunities": oscillation_opportunities,
            "oscillation_rate": (
                float(group["oscillation_event"].sum()) / oscillation_opportunities
                if oscillation_opportunities else 0.0
            ),
            "dwell_runs": int(len(policy_runs)),
            "completed_dwell_runs": int(len(completed_runs)),
            "mean_dwell_decisions": float(policy_runs["dwell_decisions"].mean()),
            "median_dwell_decisions": float(policy_runs["dwell_decisions"].median()),
            "mean_completed_dwell_ms": (
                float(completed_runs["completed_dwell_ms"].mean())
                if len(completed_runs) else np.nan
            ),
            "median_completed_dwell_ms": (
                float(completed_runs["completed_dwell_ms"].median())
                if len(completed_runs) else np.nan
            ),
            "p95_completed_dwell_ms": (
                float(completed_runs["completed_dwell_ms"].quantile(0.95))
                if len(completed_runs) else np.nan
            ),
            "aborted_generated_payload_bytes": float(
                group["aborted_generated_payload_bytes"].sum()
            ),
            "aborted_generated_payload_fraction": (
                float(group["aborted_generated_payload_bytes"].sum()) / payload_bytes
                if payload_bytes else np.nan
            ),
            "wasted_uplink_wire_bytes": float(group["wasted_uplink_wire_bytes"].sum()),
            "wasted_uplink_wire_fraction": (
                float(group["wasted_uplink_wire_bytes"].sum()) / uplink_wire_bytes
                if uplink_wire_bytes else np.nan
            ),
            "mean_queue_pending_bytes_at_decision": float(
                group["decision_queue_pending_bytes"].mean()
            ),
            "p95_queue_pending_bytes_at_decision": float(
                group["decision_queue_pending_bytes"].quantile(0.95)
            ),
            "max_queue_pending_bytes_at_decision": float(
                group["decision_queue_pending_bytes"].max()
            ),
            "mean_outstanding_unacked_bytes_at_decision": float(
                group["decision_queue_outstanding_unacked_bytes"].mean()
            ),
            "mean_queue_messages_at_decision": float(
                group["decision_queue_messages"].mean()
            ),
            "mean_queue_oldest_age_ms_at_decision": float(
                group["decision_queue_oldest_age_ms"].mean()
            ),
            "mean_backlog_bytes_at_decision": float(group["decision_backlog_bytes"].mean()),
            "p95_backlog_bytes_at_decision": float(
                group["decision_backlog_bytes"].quantile(0.95)
            ),
            "stateful_decisions": int(len(stateful)),
            "stale_state_events": int(group["stale_state_event"].sum()),
            "stale_state_rate": (
                float(group["stale_state_event"].sum()) / len(stateful)
                if len(stateful) else np.nan
            ),
            "sequence_gap_rejections": int(group["sequence_gap_rejection_event"].sum()),
            "sequence_gap_rejection_rate": (
                float(group["sequence_gap_rejection_event"].sum()) / len(stateful)
                if len(stateful) else np.nan
            ),
            "decision_sequence_gap_rate": float(group["decision_sequence_gap_event"].mean()),
            "outages": outage_count,
            "outages_recovered": recovered_count,
            "outages_unrecovered": outage_count - recovered_count,
            "outage_recovery_rate": (
                recovered_count / outage_count if outage_count else np.nan
            ),
            "median_outage_recovery_ms": (
                float(recovery_events["outage_recovery_latency_ms"].median())
                if recovered_count else np.nan
            ),
            "p95_outage_recovery_ms": (
                float(recovery_events["outage_recovery_latency_ms"].quantile(0.95))
                if recovered_count else np.nan
            ),
            "mean_lookahead_reference_signed_gap": float(
                group["lookahead_reference_signed_gap"].mean()
            ),
            "lookahead_reference_signed_gap_percentage_points": 100.0 * float(
                group["lookahead_reference_signed_gap"].mean()
            ),
            "lookahead_reference_missed_opportunity_rate": float(
                group["lookahead_reference_missed_opportunity"].mean()
            ),
            "lookahead_reference_inversion_rate": float(
                group["lookahead_reference_inversion"].mean()
            ),
            "semantic_predictions": int(len(semantic)),
            "semantic_accepted": int(len(accepted)),
            "semantic_selective_coverage": float(group["semantic_accepted"].mean()),
            "semantic_acceptance_rate_within_l1_l2": (
                len(accepted) / len(semantic) if len(semantic) else np.nan
            ),
            "semantic_selective_risk": (
                float(accepted["semantic_accepted_wrong"].mean())
                if len(accepted) else np.nan
            ),
            "semantic_selective_mission_failure_rate": (
                float(accepted["semantic_accepted_mission_failure"].mean())
                if len(accepted) else np.nan
            ),
            "high_confidence_semantic_rows": int(len(accepted)),
            "high_confidence_semantic_errors": int(
                accepted["semantic_accepted_wrong"].astype(bool).sum()
            ),
            "high_confidence_semantic_error_rate": (
                float(accepted["semantic_accepted_wrong"].mean())
                if len(accepted) else np.nan
            ),
            "semantic_acceptance_threshold": float(semantic_acceptance_threshold),
        }
        mode_counts = group["selected_mode"].value_counts()
        for mode_name in all_modes:
            row[f"mode_frequency_{mode_name.lower()}"] = (
                int(mode_counts.get(mode_name, 0)) / len(group)
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("policy").reset_index(drop=True)
    return detail, summary


def experiment5_risk_uncertainty(
    policy_trials: pd.DataFrame,
    *,
    semantic_acceptance_threshold: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 <= float(semantic_acceptance_threshold) <= 1.0:
        raise ValueError("semantic_acceptance_threshold must lie in [0, 1]")
    required = {
        "critical", "ood", "uncertainty", "policy", "correct_timely_action",
        "false_alert", "critical_miss", "high_confidence_semantic_error",
        "evidence_escalated", "total_wire_bytes", "energy_j",
        "cohort", "visual_group_id", "gold_action", "receiver_action",
        "delivered_by_deadline", "risk", "selected_mode",
        "calibrated_correctness_probability", "correct_action",
    }
    missing = required.difference(policy_trials.columns)
    if missing:
        raise ValueError(f"Experiment 5 missing columns {sorted(missing)}")
    subsets = []
    cohort = policy_trials["cohort"].astype(str)
    for name, mask in {
        "natural_prevalence": cohort.eq("natural_test"),
        "safety_critical_positive": cohort.eq("safety") & policy_trials["critical"].astype(bool),
        "safety_hard_negative": cohort.eq("safety") & ~policy_trials["critical"].astype(bool),
        "ood": cohort.eq("ood") | policy_trials["ood"].astype(bool),
        "high_uncertainty": policy_trials["uncertainty"] >= 0.35,
    }.items():
        section = policy_trials.loc[mask].copy()
        if section.empty:
            continue
        section["subset"] = name
        subsets.append(section)
    detail = pd.concat(subsets, ignore_index=True) if subsets else pd.DataFrame()
    if detail.empty:
        raise StudyGateError("Experiment 5 has no rows in any registered evaluation cohort")
    confidence = pd.to_numeric(
        detail["calibrated_correctness_probability"], errors="coerce",
    )
    detail["high_confidence_semantic_eligible"] = (
        detail["selected_mode"].isin([
            PayloadMode.L1_ALERT.name, PayloadMode.L2_SEMANTICS.name,
        ])
        & confidence.ge(float(semantic_acceptance_threshold))
    )
    detail["high_confidence_semantic_error"] = (
        detail["high_confidence_semantic_eligible"]
        & ~detail["correct_action"].astype(bool)
    )
    rows = []
    for (subset, policy), group in detail.groupby(["subset", "policy"]):
        visual_sizes = group.groupby("visual_group_id")["visual_group_id"].transform("size")
        weights = 1.0 / visual_sizes.to_numpy(float)
        def weighted_rate(column: str) -> float:
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
            valid = np.isfinite(values)
            return float(np.average(values[valid], weights=weights[valid])) if valid.any() else np.nan
        noncritical = group[~group["critical"].astype(bool)]
        critical = group[group["critical"].astype(bool)]
        high_confidence = group[
            group["high_confidence_semantic_eligible"].astype(bool)
        ]
        high_confidence_group_rates = high_confidence.groupby(
            "visual_group_id", sort=False,
        )["high_confidence_semantic_error"].mean()
        predicted_respond = (
            group["delivered_by_deadline"].astype(bool)
            & group["receiver_action"].eq("RESPOND")
        ).to_numpy()
        truth_respond = group["gold_action"].eq("RESPOND").to_numpy()
        weighted_tp = float(weights[predicted_respond & truth_respond].sum())
        weighted_fp = float(weights[predicted_respond & ~truth_respond].sum())
        weighted_fn = float(weights[~predicted_respond & truth_respond].sum())
        respond_precision = weighted_tp / (weighted_tp + weighted_fp) if weighted_tp + weighted_fp else np.nan
        respond_recall = weighted_tp / (weighted_tp + weighted_fn) if weighted_tp + weighted_fn else np.nan
        observed_action = np.where(
            group["delivered_by_deadline"].astype(bool),
            group["receiver_action"].astype(str), "NO_TIMELY_ACTION",
        )
        confusion = {}
        for gold_action in ("NO_ACTION", "REVIEW", "RESPOND"):
            for received_action in ("NO_ACTION", "REVIEW", "RESPOND", "NO_TIMELY_ACTION"):
                selected = group["gold_action"].eq(gold_action).to_numpy() & (observed_action == received_action)
                confusion[f"{gold_action}->{received_action}"] = float(weights[selected].sum())
        risk_probability = pd.to_numeric(group["risk"], errors="coerce").to_numpy(float)
        risk_auprc = _weighted_average_precision(
            risk_probability, truth_respond.astype(float), weights,
        )
        rows.append({
            "subset": subset,
            "policy": policy,
            "trials": len(group),
            "independent_visual_groups": int(group["visual_group_id"].nunique()),
            "correct_timely_rate": weighted_rate("correct_timely_action"),
            "critical_miss_rate": (
                float(critical.groupby("visual_group_id")["critical_miss"].mean().mean())
                if len(critical) else np.nan
            ),
            "false_alert_rate": (
                float(noncritical.groupby("visual_group_id")["false_alert"].mean().mean())
                if len(noncritical) else np.nan
            ),
            "respond_precision": respond_precision,
            "respond_recall": respond_recall,
            "risk_auprc": risk_auprc,
            "action_confusion_group_equal_json": json.dumps(confusion, sort_keys=True),
            "high_confidence_semantic_rows": int(len(high_confidence)),
            "high_confidence_semantic_errors": int(
                high_confidence["high_confidence_semantic_error"].astype(bool).sum()
            ),
            "high_confidence_semantic_visual_groups": int(
                len(high_confidence_group_rates)
            ),
            "high_confidence_semantic_error_rate": (
                float(high_confidence_group_rates.mean())
                if len(high_confidence_group_rates) else np.nan
            ),
            "semantic_acceptance_threshold": float(
                semantic_acceptance_threshold
            ),
            "evidence_escalation_rate": weighted_rate("evidence_escalated"),
            "mean_wire_bytes": weighted_rate("total_wire_bytes"),
            "mean_energy_j": weighted_rate("energy_j"),
        })
    summary = pd.DataFrame(rows)
    return detail, summary


def experiment6_receiver_robustness() -> pd.DataFrame:
    rows = []
    def apply_case(name: str, state: ReceiverSemanticState, message: AppMessage, decoded: Mapping[str, Any], now: int, expected: str) -> None:
        status = state.apply(message, now, decoded).value
        rows.append({"case": name, "observed": status, "expected": expected, "passed": status == expected, "receiver_seq": state.applied_seq})
    reliability = ReliabilityProfile()
    state = ReceiverSemanticState()
    key = AppMessage("exp6", "o1", 1, PayloadMode.L2_SEMANTICS, 0, 0, 100, 1000, 5, b"{}", StateKind.KEYFRAME, None, 1, reliability)
    apply_case("keyframe", state, key, {"event": "none"}, 10, "applied")
    delta_payload = make_state_delta(state.state, {"event": "hazard"})
    delta = AppMessage("exp6", "o2", 2, PayloadMode.L2_SEMANTICS, 20, 20, 100, 1000, 5, b"{}", StateKind.DELTA, 1, 2, reliability)
    apply_case("valid_delta", state, delta, delta_payload, 30, "applied")
    apply_case("duplicate_delta", state, delta, delta_payload, 40, "duplicate")
    mismatch = AppMessage("exp6", "o3", 3, PayloadMode.L2_SEMANTICS, 40, 40, 100, 1000, 5, b"{}", StateKind.DELTA, 0, 3, reliability)
    apply_case("base_mismatch", state, mismatch, {"set": {"event": "x"}, "delete": []}, 50, "base_mismatch")
    stale = AppMessage("exp6", "o4", 4, PayloadMode.L2_SEMANTICS, 50, 50, 100, 1000, 5, b"{}", StateKind.KEYFRAME, None, 1, reliability)
    apply_case("stale_keyframe", state, stale, {"event": "old"}, 60, "stale")
    expired = AppMessage("exp6", "o5", 5, PayloadMode.L2_SEMANTICS, 70, 70, 80, 90, 5, b"{}", StateKind.KEYFRAME, None, 3, reliability)
    apply_case("expired_keyframe", state, expired, {"event": "late"}, 100, "expired")
    repair = AppMessage("exp6", "o6", 6, PayloadMode.L2_SEMANTICS, 100, 100, 200, 500, 9, b"{}", StateKind.KEYFRAME, None, 4, reliability)
    apply_case("repair_keyframe", state, repair, {"event": "repaired"}, 110, "applied")
    belief = SenderReceiverBelief()
    rows.append({"case": "sender_receiver_separation", "observed": belief.confirmed_seq, "expected": 0, "passed": belief.confirmed_seq == 0, "receiver_seq": state.applied_seq})
    return pd.DataFrame(rows)


class _Experiment6FaultSimulator(SelectiveRepeatSimulator):
    """Deterministic fault injection used only by the registered Exp. 6 campaign."""

    def __init__(
        self,
        *args: Any,
        drop_data_once: Iterable[tuple[str, int, int]] = (),
        drop_fragment_ack_once: Iterable[tuple[int, int, int]] = (),
        drop_application_ack_once: Iterable[tuple[int, int]] = (),
        data_delay_us: Mapping[tuple[str, int, int], int] | None = None,
        application_ack_delay_us: Mapping[tuple[int, int], int] | None = None,
        corrupt_data_once: Iterable[tuple[int, int, int]] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._exp6_drop_data_once = set(drop_data_once)
        self._exp6_drop_fragment_ack_once = set(drop_fragment_ack_once)
        self._exp6_drop_application_ack_once = set(drop_application_ack_once)
        self._exp6_data_delay_us = dict(data_delay_us or {})
        self._exp6_application_ack_delay_us = dict(application_ack_delay_us or {})
        self._exp6_corrupt_data_once = set(corrupt_data_once)

    def _loss_and_delay(
        self,
        trace: LinkTrace,
        span: SerializationSpan,
        key: tuple[Any, ...],
    ) -> tuple[bool, int]:
        lost, delay_us = super()._loss_and_delay(trace, span, key)
        if trace.direction is Direction.UPLINK and len(key) == 4 and key[-1] == "data":
            selector = (str(key[0]), int(key[1]), int(key[2]))
            if selector in self._exp6_drop_data_once:
                self._exp6_drop_data_once.remove(selector)
                return True, delay_us
            delay_us += int(self._exp6_data_delay_us.get(selector, 0))
        elif trace.direction is Direction.DOWNLINK:
            if len(key) == 4 and key[-1] == "frag_ack":
                selector = (int(key[0]), int(key[1]), int(key[2]))
                if selector in self._exp6_drop_fragment_ack_once:
                    self._exp6_drop_fragment_ack_once.remove(selector)
                    return True, delay_us
            elif len(key) == 3 and key[1] == "app_ack":
                selector = (int(key[0]), int(key[2]))
                if selector in self._exp6_drop_application_ack_once:
                    self._exp6_drop_application_ack_once.remove(selector)
                    return True, delay_us
                delay_us += int(self._exp6_application_ack_delay_us.get(selector, 0))
        return lost, delay_us

    def _schedule(self, time_us: int, kind: EventType, data: Any = None) -> None:
        if kind is EventType.DATA_ARRIVE and data is not None:
            attempt, datagram = data
            selector = (
                int(attempt.message_id), int(attempt.fragment_index), int(attempt.attempt),
            )
            if selector in self._exp6_corrupt_data_once:
                self._exp6_corrupt_data_once.remove(selector)
                corrupted = bytearray(datagram)
                if not corrupted:
                    raise AssertionError("Cannot corrupt an empty fragment datagram")
                corrupted[-1] ^= 0x01
                data = (attempt, bytes(corrupted))
        super()._schedule(time_us, kind, data)


def _experiment6_packet(
    observation_id: str,
    capture_us: int,
    *,
    schema_version: str = "semuav.v1",
    model_revision: str = "0" * 40,
    padding_bytes: int = 0,
) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema_version": schema_version,
        "observation_id": observation_id,
        "mission": "urban",
        "capture_time_us": capture_us,
        "telemetry": {
            "gps_lat": 24.523,
            "gps_lon": 54.434,
            "altitude_m": 80.0,
            "yaw_deg": 12.0,
        },
        "sufficient_evidence": True,
        "event": "visible_road_obstruction",
        "objects": [],
        "relations": [],
        "risk": 0.9,
        "uncertainty": 0.1,
        "calibrated_correctness_probability": 0.9,
        "recommended_action": "RESPOND",
        "evidence": [],
        "confidence_provenance": {
            "source": "registered_deterministic_probability",
            "source_model": "exp6-registered-fixture",
            "temperature": 1.0,
        },
        "provenance": {
            "source_model": "exp6-registered-fixture",
            "model_revision": model_revision,
            "prompt_hash": "0" * 64,
            "ontology_version": "semuav.ontology.v1",
            "calibration": {
                "method": "registered-deterministic-contract",
                "temperature": 1.0,
            },
            "schema_version": schema_version,
        },
    }
    if padding_bytes:
        packet["registered_padding"] = "x" * int(padding_bytes)
    return packet


def _experiment6_mapping_message(
    *,
    message_id: int,
    observation_id: str,
    capture_us: int,
    mapping: Mapping[str, Any],
    scientific: bool,
    reliability: ReliabilityProfile,
    state_kind: StateKind = StateKind.KEYFRAME,
    base_state_seq: int | None = None,
    resulting_state_seq: int | None = 1,
    priority: int = 5,
    deadline_after_us: int = 500_000,
    expiry_after_us: int = 1_000_000,
    mode: PayloadMode = PayloadMode.L2_SEMANTICS,
) -> tuple[AppMessage, dict[str, Any]]:
    payload, wire_format = encode_mapping(mapping, scientific=scientific)
    message = AppMessage(
        "exp6-registered",
        observation_id,
        message_id,
        mode,
        capture_us,
        capture_us,
        capture_us + deadline_after_us,
        capture_us + expiry_after_us,
        priority,
        payload,
        state_kind,
        base_state_seq,
        resulting_state_seq,
        reliability,
    )
    return message, {"wire_format": wire_format}


def _experiment6_run(
    simulator: SelectiveRepeatSimulator,
    messages: Sequence[tuple[AppMessage, Mapping[str, Any]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    captures = [
        (message.capture_us, (message, dict(metadata)))
        for message, metadata in messages
    ]
    return simulator.run(
        [], captures=captures,
        capture_handler=lambda _simulator, _now_us, value: value,
    )


def experiment6_registered_campaign(
    *,
    seeds: Sequence[int] = (601, 607, 613),
    scientific: bool,
    registered_cases: Sequence[str] = EXPERIMENT6_REGISTERED_CASES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the frozen, deterministic receiver/transport fault campaign.

    Every registered case is repeated under every registered seed. Faults are
    injected by exact packet/ACK identity, while the seed still controls any
    simulator keyed randomness. Passing therefore cannot depend on a lucky loss
    draw. The returned detail and summary tables are ledger-ready evidence.
    """

    registered_cases = tuple(str(case) for case in registered_cases)
    if registered_cases != EXPERIMENT6_REGISTERED_CASES:
        raise StudyGateError(
            "Experiment 6 registered case set/order differs from the frozen campaign"
        )
    registered_seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not registered_seeds:
        raise StudyGateError("Experiment 6 requires at least one registered seed")

    criteria = {
        "data_loss_retransmission": (
            "The selected first data attempt is lost; ARQ retries, the receiver applies "
            "the keyframe exactly once, and cumulative application feedback repairs sender belief."
        ),
        "fragment_ack_loss_application_ack_recovery": (
            "The selected fragment ACK is lost; a validated application ACK completes "
            "fragment state without an unnecessary retry."
        ),
        "application_ack_loss_safe_uncertainty": (
            "The selected application ACK is lost; receiver state advances but sender "
            "belief remains unadvanced and becomes uncertain after the registered timeout."
        ),
        "data_reordering_delay": (
            "A delayed fragment arrives after a later fragment; reassembly still applies "
            "the keyframe once with no state divergence."
        ),
        "application_ack_reordering_delay": (
            "A delayed older application ACK arrives after newer cumulative feedback and "
            "is ignored without rolling sender belief backward."
        ),
        "corruption_crc_retransmission": (
            "A one-bit payload corruption is rejected by fragment CRC; a clean retry is "
            "accepted and the keyframe is applied exactly once."
        ),
        "missing_prerequisite_base_mismatch": (
            "A delta whose base is absent is rejected as base_mismatch; receiver and sender "
            "sequences do not advance."
        ),
        "stale_state_rejected": (
            "A lower state sequence arriving after a newer keyframe is rejected as stale "
            "without overwriting receiver state."
        ),
        "expired_state_rejected": (
            "A complete keyframe arriving after its registered expiry is rejected as expired "
            "and does not advance receiver state."
        ),
        "schema_version_mismatch_rejected": (
            "An unsupported semantic schema version is rejected before state application."
        ),
        "model_version_mismatch_rejected": (
            "A semantic keyframe whose provenance revision differs from the registered "
            "receiver revision is rejected as version_mismatch."
        ),
        "repair_keyframe_cumulative_ack_recovery": (
            "After an earlier keyframe ACK timeout, a later repair keyframe and cumulative "
            "application ACK clear all pending/timed-out state and restore known belief."
        ),
        "urgent_priority_bypass": (
            "An urgent alert captured while bulk fragments are queued starts after the "
            "non-preemptible in-flight fragment but before every remaining bulk fragment."
        ),
    }
    rows: list[dict[str, Any]] = []

    def append_case(
        case: str,
        seed: int,
        passed: bool,
        *,
        expected_status: str,
        observed_status: str,
        receiver_seq: int,
        sender_seq: int,
        sender_status: str,
        retry_uplink_bytes: int = 0,
        event_evidence: Mapping[str, Any] | None = None,
        notes: str = "",
    ) -> None:
        rows.append({
            "campaign_version": EXPERIMENT6_CAMPAIGN_VERSION,
            "case": case,
            "case_index": EXPERIMENT6_REGISTERED_CASES.index(case),
            "seed": int(seed),
            "fault_supported": True,
            "pass_criterion": criteria[case],
            "expected_status": expected_status,
            "observed_status": observed_status,
            "receiver_seq": int(receiver_seq),
            "sender_confirmed_seq": int(sender_seq),
            "sender_status": str(sender_status),
            "retry_uplink_bytes": int(retry_uplink_bytes),
            "event_evidence": json.dumps(event_evidence or {}, sort_keys=True),
            "notes": notes,
            "passed": bool(passed),
            "scientific_wire_format": bool(scientific),
        })

    def simulator(
        seed: int,
        *,
        one_way_ms: float = 1.0,
        rate_mbps: float = 10.0,
        decode_us: int = 0,
        **faults: Any,
    ) -> _Experiment6FaultSimulator:
        return _Experiment6FaultSimulator(
            constant_trace(
                f"exp6-uplink-{seed}", Direction.UPLINK,
                rate_mbps, one_way_ms, 0.0,
            ),
            constant_trace(
                f"exp6-downlink-{seed}", Direction.DOWNLINK,
                rate_mbps, one_way_ms, 0.0,
            ),
            seed=seed,
            decode_us=decode_us,
            **faults,
        )

    def result_for(results: pd.DataFrame, message_id: int) -> pd.Series:
        match = results[results["message_id"].eq(message_id)]
        if len(match) != 1:
            raise AssertionError(f"Expected one Experiment 6 result for message {message_id}")
        return match.iloc[0]

    def event_mask(events: pd.DataFrame, kind: str, message_id: int | None = None) -> pd.Series:
        mask = events["kind"].eq(kind)
        if message_id is not None:
            mask &= events["message_id"].eq(message_id)
        return mask

    for seed in registered_seeds:
        reliability = ReliabilityProfile(
            window_fragments=512,
            retry_limit=2,
            min_rto_us=60_000,
            max_rto_us=60_000,
            delayed_ack_us=0,
        )

        # Data loss followed by an identity-targeted ARQ retransmission.
        message_id = 6101
        observation_id = f"exp6-data-loss-{seed}"
        message, metadata = _experiment6_mapping_message(
            message_id=message_id, observation_id=observation_id, capture_us=0,
            mapping=_experiment6_packet(observation_id, 0), scientific=scientific,
            reliability=reliability,
        )
        sim = simulator(
            seed, drop_data_once={(observation_id, 0, 1)},
        )
        results, events = _experiment6_run(sim, [(message, metadata)])
        observed = result_for(results, message_id)
        retried = bool(event_mask(events, "fragment_retry", message_id).any())
        passed = bool(
            observed.apply_status == "applied"
            and int(observed.retry_uplink_bytes) > 0
            and retried
            and sim.receiver_state.applied_seq == 1
            and sim.sender_belief.confirmed_seq == 1
            and sim.sender_belief.status == "known"
        )
        append_case(
            "data_loss_retransmission", seed, passed,
            expected_status="applied_after_retry", observed_status=str(observed.apply_status),
            receiver_seq=sim.receiver_state.applied_seq,
            sender_seq=sim.sender_belief.confirmed_seq,
            sender_status=sim.sender_belief.status,
            retry_uplink_bytes=int(observed.retry_uplink_bytes),
            event_evidence={"fragment_retry": retried},
        )

        # Fragment ACK loss is recovered by the stronger application ACK.
        message_id = 6102
        observation_id = f"exp6-fragment-ack-loss-{seed}"
        message, metadata = _experiment6_mapping_message(
            message_id=message_id, observation_id=observation_id, capture_us=0,
            mapping=_experiment6_packet(observation_id, 0), scientific=scientific,
            reliability=reliability,
        )
        sim = simulator(
            seed, drop_fragment_ack_once={(message_id, 0, 1)},
        )
        results, events = _experiment6_run(sim, [(message, metadata)])
        observed = result_for(results, message_id)
        app_ack_arrived = bool(event_mask(events, "application_ack_arrive", message_id).any())
        passed = bool(
            observed.apply_status == "applied"
            and int(observed.retry_uplink_bytes) == 0
            and app_ack_arrived
            and sim.sender_belief.confirmed_seq == 1
            and sim.sender_belief.status == "known"
        )
        append_case(
            "fragment_ack_loss_application_ack_recovery", seed, passed,
            expected_status="applied_via_application_ack",
            observed_status=str(observed.apply_status),
            receiver_seq=sim.receiver_state.applied_seq,
            sender_seq=sim.sender_belief.confirmed_seq,
            sender_status=sim.sender_belief.status,
            retry_uplink_bytes=int(observed.retry_uplink_bytes),
            event_evidence={"application_ack_arrived": app_ack_arrived},
        )

        # Application ACK loss must leave safe uncertainty rather than a false belief.
        message_id = 6103
        observation_id = f"exp6-application-ack-loss-{seed}"
        message, metadata = _experiment6_mapping_message(
            message_id=message_id, observation_id=observation_id, capture_us=0,
            mapping=_experiment6_packet(observation_id, 0), scientific=scientific,
            reliability=reliability,
        )
        sim = simulator(seed, drop_application_ack_once={(message_id, 1)})
        results, events = _experiment6_run(sim, [(message, metadata)])
        observed = result_for(results, message_id)
        timed_out = bool(event_mask(events, "application_ack_timeout", message_id).any())
        passed = bool(
            observed.apply_status == "applied"
            and sim.receiver_state.applied_seq == 1
            and sim.sender_belief.confirmed_seq == 0
            and sim.sender_belief.status == "uncertain"
            and timed_out
        )
        append_case(
            "application_ack_loss_safe_uncertainty", seed, passed,
            expected_status="receiver_applied_sender_uncertain",
            observed_status=f"{observed.apply_status}/{sim.sender_belief.status}",
            receiver_seq=sim.receiver_state.applied_seq,
            sender_seq=sim.sender_belief.confirmed_seq,
            sender_status=sim.sender_belief.status,
            retry_uplink_bytes=int(observed.retry_uplink_bytes),
            event_evidence={"application_ack_timeout": timed_out},
        )

        # Out-of-order data fragments are buffered and reassembled safely.
        ordering_reliability = dataclasses.replace(
            reliability, min_rto_us=200_000, max_rto_us=200_000,
        )
        message_id = 6104
        observation_id = f"exp6-data-reorder-{seed}"
        message, metadata = _experiment6_mapping_message(
            message_id=message_id, observation_id=observation_id, capture_us=0,
            mapping=_experiment6_packet(observation_id, 0, padding_bytes=5_000),
            scientific=scientific, reliability=ordering_reliability,
        )
        sim = simulator(
            seed, data_delay_us={(observation_id, 0, 1): 50_000},
        )
        results, events = _experiment6_run(sim, [(message, metadata)])
        observed = result_for(results, message_id)
        receives = events[event_mask(events, "fragment_receive", message_id)]
        receive_order = receives["fragment_index"].astype(int).tolist()
        reordered = bool(receive_order and receive_order[0] != 0 and 0 in receive_order)
        passed = bool(
            reordered and observed.apply_status == "applied"
            and sim.receiver_state.applied_seq == 1
            and int(observed.retry_uplink_bytes) == 0
        )
        append_case(
            "data_reordering_delay", seed, passed,
            expected_status="out_of_order_then_applied", observed_status=str(observed.apply_status),
            receiver_seq=sim.receiver_state.applied_seq,
            sender_seq=sim.sender_belief.confirmed_seq,
            sender_status=sim.sender_belief.status,
            retry_uplink_bytes=int(observed.retry_uplink_bytes),
            event_evidence={"fragment_receive_order": receive_order},
        )

        # Older application feedback delayed behind a newer cumulative ACK is ignored.
        feedback_reliability = dataclasses.replace(
            reliability, min_rto_us=400_000, max_rto_us=400_000,
        )
        first_id, second_id = 6105, 6106
        first_obs, second_obs = f"exp6-feedback-old-{seed}", f"exp6-feedback-new-{seed}"
        first, first_metadata = _experiment6_mapping_message(
            message_id=first_id, observation_id=first_obs, capture_us=0,
            mapping=_experiment6_packet(first_obs, 0), scientific=scientific,
            reliability=feedback_reliability, resulting_state_seq=1,
        )
        second, second_metadata = _experiment6_mapping_message(
            message_id=second_id, observation_id=second_obs, capture_us=40_000,
            mapping=_experiment6_packet(second_obs, 40_000), scientific=scientific,
            reliability=feedback_reliability, resulting_state_seq=2,
        )
        sim = simulator(
            seed, application_ack_delay_us={(first_id, 1): 150_000},
        )
        results, events = _experiment6_run(
            sim, [(first, first_metadata), (second, second_metadata)],
        )
        second_observed = result_for(results, second_id)
        ignored_old = bool(
            event_mask(events, "application_ack_arrive", first_id).any()
            and events.loc[
                event_mask(events, "application_ack_arrive", first_id),
                "feedback_ignored",
            ].astype(bool).all()
        )
        passed = bool(
            second_observed.apply_status == "applied" and ignored_old
            and sim.receiver_state.applied_seq == 2
            and sim.sender_belief.confirmed_seq == 2
            and sim.sender_belief.status == "known"
        )
        append_case(
            "application_ack_reordering_delay", seed, passed,
            expected_status="new_ack_retained_old_ack_ignored",
            observed_status=str(second_observed.apply_status),
            receiver_seq=sim.receiver_state.applied_seq,
            sender_seq=sim.sender_belief.confirmed_seq,
            sender_status=sim.sender_belief.status,
            retry_uplink_bytes=int(results["retry_uplink_bytes"].sum()),
            event_evidence={"older_feedback_ignored": ignored_old},
        )

        # Payload bit corruption must fail the fragment CRC and trigger a clean retry.
        message_id = 6107
        observation_id = f"exp6-corruption-{seed}"
        message, metadata = _experiment6_mapping_message(
            message_id=message_id, observation_id=observation_id, capture_us=0,
            mapping=_experiment6_packet(observation_id, 0), scientific=scientific,
            reliability=reliability,
        )
        sim = simulator(seed, corrupt_data_once={(message_id, 0, 1)})
        results, events = _experiment6_run(sim, [(message, metadata)])
        observed = result_for(results, message_id)
        crc_rejected = bool(event_mask(events, "fragment_drop_invalid", message_id).any())
        passed = bool(
            crc_rejected and observed.apply_status == "applied"
            and int(observed.retry_uplink_bytes) > 0
            and sim.receiver_state.applied_seq == 1
        )
        append_case(
            "corruption_crc_retransmission", seed, passed,
            expected_status="crc_reject_then_applied", observed_status=str(observed.apply_status),
            receiver_seq=sim.receiver_state.applied_seq,
            sender_seq=sim.sender_belief.confirmed_seq,
            sender_status=sim.sender_belief.status,
            retry_uplink_bytes=int(observed.retry_uplink_bytes),
            event_evidence={"fragment_drop_invalid": crc_rejected},
        )

        # A delta cannot be applied without its exact prerequisite state.
        message_id = 6108
        observation_id = f"exp6-base-mismatch-{seed}"
        message, metadata = _experiment6_mapping_message(
            message_id=message_id, observation_id=observation_id, capture_us=0,
            mapping={"set": {"event": "hazard"}, "delete": []},
            scientific=scientific, reliability=reliability,
            state_kind=StateKind.DELTA, base_state_seq=1, resulting_state_seq=2,
        )
        sim = simulator(seed)
        results, events = _experiment6_run(sim, [(message, metadata)])
        observed = result_for(results, message_id)
        passed = bool(
            observed.apply_status == "base_mismatch"
            and sim.receiver_state.applied_seq == 0
            and sim.sender_belief.confirmed_seq == 0
            and sim.sender_belief.status == "uncertain"
        )
        append_case(
            "missing_prerequisite_base_mismatch", seed, passed,
            expected_status="base_mismatch", observed_status=str(observed.apply_status),
            receiver_seq=sim.receiver_state.applied_seq,
            sender_seq=sim.sender_belief.confirmed_seq,
            sender_status=sim.sender_belief.status,
            retry_uplink_bytes=int(observed.retry_uplink_bytes),
        )

        # A lower state sequence cannot overwrite a newer keyframe.
        newer_id, stale_id = 6109, 6110
        newer_obs, stale_obs = f"exp6-newer-{seed}", f"exp6-stale-{seed}"
        newer, newer_metadata = _experiment6_mapping_message(
            message_id=newer_id, observation_id=newer_obs, capture_us=0,
            mapping=_experiment6_packet(newer_obs, 0), scientific=scientific,
            reliability=reliability, resulting_state_seq=2,
        )
        stale, stale_metadata = _experiment6_mapping_message(
            message_id=stale_id, observation_id=stale_obs, capture_us=40_000,
            mapping=_experiment6_packet(stale_obs, 40_000), scientific=scientific,
            reliability=reliability, resulting_state_seq=1,
        )
        sim = simulator(seed)
        results, events = _experiment6_run(
            sim, [(newer, newer_metadata), (stale, stale_metadata)],
        )
        observed = result_for(results, stale_id)
        passed = bool(
            observed.apply_status == "stale"
            and sim.receiver_state.applied_seq == 2
            and sim.sender_belief.confirmed_seq == 2
        )
        append_case(
            "stale_state_rejected", seed, passed,
            expected_status="stale", observed_status=str(observed.apply_status),
            receiver_seq=sim.receiver_state.applied_seq,
            sender_seq=sim.sender_belief.confirmed_seq,
            sender_status=sim.sender_belief.status,
            retry_uplink_bytes=int(results["retry_uplink_bytes"].sum()),
        )

        # Completion after state expiry must not mutate receiver state.
        message_id = 6111
        observation_id = f"exp6-expired-{seed}"
        message, metadata = _experiment6_mapping_message(
            message_id=message_id, observation_id=observation_id, capture_us=0,
            mapping=_experiment6_packet(observation_id, 0), scientific=scientific,
            reliability=reliability, deadline_after_us=20_000, expiry_after_us=20_000,
        )
        sim = simulator(seed, one_way_ms=30.0)
        results, events = _experiment6_run(sim, [(message, metadata)])
        observed = result_for(results, message_id)
        expiry_event = bool(event_mask(events, "message_expire", message_id).any())
        passed = bool(
            observed.apply_status == "expired" and expiry_event
            and sim.receiver_state.applied_seq == 0
            and sim.sender_belief.confirmed_seq == 0
        )
        append_case(
            "expired_state_rejected", seed, passed,
            expected_status="expired", observed_status=str(observed.apply_status),
            receiver_seq=sim.receiver_state.applied_seq,
            sender_seq=sim.sender_belief.confirmed_seq,
            sender_status=sim.sender_belief.status,
            retry_uplink_bytes=int(observed.retry_uplink_bytes),
            event_evidence={"message_expire": expiry_event},
        )

        # Unsupported semantic schema is rejected before state application.
        message_id = 6112
        observation_id = f"exp6-schema-mismatch-{seed}"
        message, metadata = _experiment6_mapping_message(
            message_id=message_id, observation_id=observation_id, capture_us=0,
            mapping=_experiment6_packet(
                observation_id, 0, schema_version="semuav.unsupported",
            ),
            scientific=scientific, reliability=reliability,
        )
        sim = simulator(seed)
        results, events = _experiment6_run(sim, [(message, metadata)])
        observed = result_for(results, message_id)
        passed = bool(
            observed.apply_status == "decode_error"
            and sim.receiver_state.applied_seq == 0
            and sim.sender_belief.confirmed_seq == 0
        )
        append_case(
            "schema_version_mismatch_rejected", seed, passed,
            expected_status="decode_error", observed_status=str(observed.apply_status),
            receiver_seq=sim.receiver_state.applied_seq,
            sender_seq=sim.sender_belief.confirmed_seq,
            sender_status=sim.sender_belief.status,
            retry_uplink_bytes=int(observed.retry_uplink_bytes),
        )

        # Receiver's registered model-revision contract is checked explicitly.
        message_id = 6113
        observation_id = f"exp6-model-mismatch-{seed}"
        message, metadata = _experiment6_mapping_message(
            message_id=message_id, observation_id=observation_id, capture_us=0,
            mapping=_experiment6_packet(
                observation_id, 0, model_revision="1" * 40,
            ),
            scientific=scientific, reliability=reliability,
        )
        metadata["expected_model_revision"] = "0" * 40
        sim = simulator(seed)
        results, events = _experiment6_run(sim, [(message, metadata)])
        observed = result_for(results, message_id)
        passed = bool(
            observed.apply_status == "version_mismatch"
            and sim.receiver_state.applied_seq == 0
            and sim.sender_belief.confirmed_seq == 0
        )
        append_case(
            "model_version_mismatch_rejected", seed, passed,
            expected_status="version_mismatch", observed_status=str(observed.apply_status),
            receiver_seq=sim.receiver_state.applied_seq,
            sender_seq=sim.sender_belief.confirmed_seq,
            sender_status=sim.sender_belief.status,
            retry_uplink_bytes=int(observed.retry_uplink_bytes),
        )

        # A later repair keyframe's cumulative ACK resolves an older lost ACK.
        first_id, repair_id = 6114, 6115
        first_obs, repair_obs = f"exp6-repair-old-{seed}", f"exp6-repair-new-{seed}"
        first, first_metadata = _experiment6_mapping_message(
            message_id=first_id, observation_id=first_obs, capture_us=0,
            mapping=_experiment6_packet(first_obs, 0), scientific=scientific,
            reliability=reliability, resulting_state_seq=1,
        )
        repair, repair_metadata = _experiment6_mapping_message(
            message_id=repair_id, observation_id=repair_obs, capture_us=100_000,
            mapping=_experiment6_packet(repair_obs, 100_000), scientific=scientific,
            reliability=reliability, resulting_state_seq=2, priority=9,
        )
        sim = simulator(seed, drop_application_ack_once={(first_id, 1)})
        results, events = _experiment6_run(
            sim, [(first, first_metadata), (repair, repair_metadata)],
        )
        repair_observed = result_for(results, repair_id)
        old_timed_out = bool(event_mask(events, "application_ack_timeout", first_id).any())
        passed = bool(
            old_timed_out and repair_observed.apply_status == "applied"
            and sim.receiver_state.applied_seq == 2
            and sim.sender_belief.confirmed_seq == 2
            and sim.sender_belief.status == "known"
            and not sim._pending_state_message_ids
            and not sim._timed_out_state_message_ids
        )
        append_case(
            "repair_keyframe_cumulative_ack_recovery", seed, passed,
            expected_status="repair_applied_belief_known",
            observed_status=f"{repair_observed.apply_status}/{sim.sender_belief.status}",
            receiver_seq=sim.receiver_state.applied_seq,
            sender_seq=sim.sender_belief.confirmed_seq,
            sender_status=sim.sender_belief.status,
            retry_uplink_bytes=int(results["retry_uplink_bytes"].sum()),
            event_evidence={
                "older_application_ack_timeout": old_timed_out,
                "pending_state_count": len(sim._pending_state_message_ids),
                "timed_out_state_count": len(sim._timed_out_state_message_ids),
            },
        )

        # Priority is non-preemptive for the active fragment, then bypasses queued bulk.
        bulk_id, urgent_id = 6116, 6117
        bulk = AppMessage(
            "exp6-registered", f"exp6-bulk-{seed}", bulk_id,
            PayloadMode.L5_IMAGE_OR_CLIP, 0, 0, 2_000_000, 3_000_000,
            1, b"bulk" * 4_000, StateKind.NONE, None, None, reliability,
        )
        urgent_observation = f"exp6-urgent-{seed}"
        urgent, urgent_metadata = _experiment6_mapping_message(
            message_id=urgent_id, observation_id=urgent_observation, capture_us=5_000,
            mapping=semantic_level_view(
                _experiment6_packet(urgent_observation, 5_000), PayloadMode.L1_ALERT,
            ),
            scientific=scientific, reliability=reliability,
            state_kind=StateKind.NONE, resulting_state_seq=None,
            priority=10, mode=PayloadMode.L1_ALERT,
            deadline_after_us=300_000, expiry_after_us=600_000,
        )
        urgent_decoded = decode_mapping(
            urgent.payload, str(urgent_metadata["wire_format"]),
        )
        urgent_required_fields = {
            "schema_version", "observation_id", "mission", "capture_time_us",
            "telemetry", "event", "risk", "uncertainty",
            "calibrated_correctness_probability", "recommended_action",
            "sufficient_evidence", "provenance", "confidence_provenance",
        }
        urgent_payload_contract_valid = not urgent_required_fields.difference(
            urgent_decoded,
        )
        sim = simulator(seed, rate_mbps=0.5)
        results, events = _experiment6_run(
            sim, [(bulk, {}), (urgent, urgent_metadata)],
        )
        starts = events[events["kind"].eq("fragment_tx_start")].copy()
        bulk_starts = starts[starts["message_id"].eq(bulk_id)]
        urgent_starts = starts[starts["message_id"].eq(urgent_id)]
        bulk_first_us = int(bulk_starts["time_us"].min())
        urgent_first_us = int(urgent_starts["time_us"].min())
        remaining_bulk = bulk_starts[bulk_starts["fragment_index"].astype(int) > 0]
        next_bulk_us = int(remaining_bulk["time_us"].min())
        urgent_observed = result_for(results, urgent_id)
        bypassed = bulk_first_us < urgent_first_us < next_bulk_us
        passed = bool(
            bypassed and len(fragment_message(bulk)) > 1
            and urgent_payload_contract_valid
            and urgent_observed.apply_status == "applied"
        )
        append_case(
            "urgent_priority_bypass", seed, passed,
            expected_status="urgent_after_inflight_before_bulk_queue",
            observed_status=str(urgent_observed.apply_status),
            receiver_seq=sim.receiver_state.applied_seq,
            sender_seq=sim.sender_belief.confirmed_seq,
            sender_status=sim.sender_belief.status,
            retry_uplink_bytes=int(results["retry_uplink_bytes"].sum()),
            event_evidence={
                "bulk_first_tx_us": bulk_first_us,
                "urgent_first_tx_us": urgent_first_us,
                "next_bulk_tx_us": next_bulk_us,
                "bulk_fragment_count": len(fragment_message(bulk)),
                "urgent_payload_contract_valid": urgent_payload_contract_valid,
            },
            notes="Fragment transmission is deliberately non-preemptive.",
        )

    detail = pd.DataFrame(rows).sort_values(["case_index", "seed"]).reset_index(drop=True)
    expected_pairs = {
        (case, seed) for case in EXPERIMENT6_REGISTERED_CASES for seed in registered_seeds
    }
    observed_pairs = set(zip(detail["case"], detail["seed"].astype(int)))
    if observed_pairs != expected_pairs or len(detail) != len(expected_pairs):
        raise StudyGateError("Experiment 6 campaign coverage is incomplete or duplicated")

    summary_rows = []
    for case in EXPERIMENT6_REGISTERED_CASES:
        group = detail[detail["case"].eq(case)]
        summary_rows.append({
            "campaign_version": EXPERIMENT6_CAMPAIGN_VERSION,
            "case": case,
            "case_index": EXPERIMENT6_REGISTERED_CASES.index(case),
            "pass_criterion": criteria[case],
            "registered_seeds": json.dumps(list(registered_seeds)),
            "runs": int(len(group)),
            "passed_runs": int(group["passed"].sum()),
            "pass_rate": float(group["passed"].mean()),
            "fault_supported": bool(group["fault_supported"].all()),
            "all_passed": bool(group["passed"].all()),
            "observed_statuses": json.dumps(
                sorted(set(group["observed_status"].astype(str))),
            ),
        })
    summary = pd.DataFrame(summary_rows)
    return detail, summary


def experiment7_readiness(
    project_root: str | Path,
    cache_root: str | Path,
    device: str,
    profile: str,
    embedded_log: str | Path | None = None,
    expected_hardware_id: str | None = None,
    expected_platform_class: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    report = environment_report(project_root, cache_root, device, profile)
    embedded = (
        import_embedded_jsonl(
            embedded_log,
            expected_hardware_id=expected_hardware_id,
            expected_platform_class=expected_platform_class,
        )
        if embedded_log else None
    )
    return report, embedded


def dry_run_all(ctx: RunContext, device: str = "cpu") -> dict[str, pd.DataFrame]:
    """Exercise all seven code paths. Outputs are marked NON_SCIENTIFIC."""
    if ctx.profile != "dry_run":
        raise StudyGateError("dry_run_all is restricted to the dry_run profile")
    root = ctx.run_root / "dry_run_NON_SCIENTIFIC"
    root.mkdir(parents=True, exist_ok=True)
    manifest = frame_stage(ctx, "00_dry_manifest", {"n": 36}, lambda: make_dry_run_fixtures(root, 36))
    backends = [DeterministicMockBackend(name) for name in ("mock_qwen", "mock_smolvlm", "mock_moondream", "mock_detector_only")]
    exp1_detail, exp1_summary = experiment1_perception(manifest, backends, scientific=False)
    exp2 = experiment2_representation_frontier(manifest, scientific=False)
    # The full static grid is intentionally sampled in integration mode; full mode
    # executes the preregistered complete grid.
    exp3_input = exp2[exp2["variant"].isin(["json-debug", "json-debug+jpeg_q50_roi", "jpeg_q50_s0.5", "lossless_png_reference"])].head(12)
    exp3 = experiment3_static_transport(
        exp3_input, manifest, seed=17,
        rates_mbps=(0.1, 1.0, 20.0), rtts_ms=(20.0, 200.0),
        losses=(0.0, 0.15), jitters_ms=(0.0,), queue_kb=(128,),
    )
    traces = [dynamic_trace_family(family, seed, 120) for family, seed in zip(("sudden_drop", "ramp", "outage", "markov", "fading", "recovery"), range(6))]
    policies: list[RepresentationPolicy] = [
        FixedModePolicy(PayloadMode.L1_ALERT),
        FixedModePolicy(PayloadMode.L2_SEMANTICS),
        AdaptiveImagePolicy(), AdaptiveSemanticPolicy(), ChannelOnlyPolicy(), MissionOnlyPolicy(),
        DeadlineGreedyPolicy(), ChannelMissionPolicy(), FullPolicy(), LookaheadReferencePolicy(),
    ]
    exp4 = run_paired_policy_trials(manifest.head(18), exp2, traces, policies, (11,))
    exp4, exp4_controller_summary = experiment4_controller_behavior(exp4)
    exp5_detail, exp5_summary = experiment5_risk_uncertainty(exp4)
    exp6 = experiment6_receiver_robustness()
    # Local import avoids a module cycle: the strategy comparison reuses the
    # registered fault simulator defined above.
    from .exp6_strategies import experiment6_state_strategy_comparison
    exp6_strategy_detail, exp6_strategy_summary = experiment6_state_strategy_comparison(
        seeds=(601,), scientific=False,
    )
    exp7, _ = experiment7_readiness(ctx.project_root, ctx.run_root, device, "dry_run")
    outputs = {
        "manifest": manifest,
        "exp1_detail": exp1_detail, "exp1_summary": exp1_summary, "exp2_frontier": exp2,
        "exp3_transport": exp3, "exp4_policy": exp4,
        "exp4_controller_summary": exp4_controller_summary, "exp5_detail": exp5_detail,
        "exp5_summary": exp5_summary, "exp6_robustness": exp6,
        "exp6_state_strategy_detail": exp6_strategy_detail,
        "exp6_state_strategy_summary": exp6_strategy_summary,
        "exp7_readiness": exp7,
    }
    for name, frame in outputs.items():
        atomic_write_frame(root / f"{name}.csv.gz", frame)
    status_rows = []
    for number, name in EXPERIMENT_NAMES.items():
        execution = (
            "UNIT_CONTRACT_EXERCISED" if number == 6 else
            "READINESS_SCAFFOLD_EXERCISED" if number == 7 else
            "SOFTWARE_PATH_EXERCISED"
        )
        status_rows.append({
            "experiment": number, "name": name, "execution_status": execution,
            "evidence_class": "synthetic_integration",
            "empirical_complete": False, "scientific_claim_eligible": False,
            "claim_status": "NOT_RUN", "scientific": False,
        })
    status = pd.DataFrame(status_rows)
    atomic_write_frame(root / "experiment_status.csv.gz", status)
    atomic_write_json(root / "SCIENTIFIC_STATUS.json", {
        "status": "NON_SCIENTIFIC_INTEGRATION_FIXTURE",
        "reason": "Generated images, mock model outputs, and synthetic traces validate software only",
        "experiments_exercised": list(EXPERIMENT_NAMES),
        "empirical_study_executed": False,
    })
    outputs["status"] = status
    return outputs


def final_completion_status(ctx: RunContext, stage_status: Mapping[int, str], gates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    gates_pass = bool(len(gates) and gates["passed"].all()) if "passed" in gates else False
    for number, name in EXPERIMENT_NAMES.items():
        raw = stage_status.get(number, "BLOCKED")
        publishable = ctx.profile == "full" and raw == "COMPLETE" and gates_pass
        rows.append({"experiment": number, "name": name, "execution_status": raw, "scientific_claim_eligible": publishable})
    return pd.DataFrame(rows)
