"""Paper-phase runner for the real-data VisDrone study.

This module is intentionally thin over the tested Experiment 1--7 machinery.
It adds the phase contracts that a notebook needs: bounded outcome-blind
sampling, frozen real-model execution, concrete native receivers and L4 head,
validation-only selection, one-time confirmatory access, hash-bound artifacts,
and claim recomputation.  It has no mock/synthetic fallback.
"""

from __future__ import annotations

import dataclasses
import gc
import itertools
import io
import json
import math
import os
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd

from .control import (
    AdaptiveFeaturePolicy,
    AdaptiveImagePolicy,
    AdaptiveSemanticPolicy,
    AlwaysVLMPolicy,
    ChannelMissionPolicy,
    ChannelOnlyPolicy,
    DeadlineGreedyPolicy,
    FixedVariantPolicy,
    FullPolicy,
    LookaheadReferencePolicy,
    MissionOnlyPolicy,
    NoEvidenceEscalationPolicy,
    NoHysteresisPolicy,
    NoMissionRiskPolicy,
    NoReceiverStatePolicy,
    RiskAwareNoUncertaintyPolicy,
    causal_schema_audit,
)
from .evaluation import (
    BootstrapEffect,
    assert_paired_design,
    budget_compliance,
    holm_adjust,
    hypothesis_report,
    two_way_clustered_paired_bootstrap,
    validation_power_simulation,
)
from .exp6_strategies import experiment6_state_strategy_comparison
from .experiments import (
    add_experiment2_total_costs,
    assert_experiment3_partition_index_coverage,
    build_causal_candidate_profiles,
    experiment1_perception,
    experiment2_representation_frontier_sharded,
    experiment3_static_transport_partitioned,
    experiment4_controller_behavior,
    experiment5_risk_uncertainty,
    experiment6_registered_campaign,
    run_paired_policy_trials_sharded,
    select_experiment3_candidate_variants,
    select_experiment3_manifest,
)
from .hardware import (
    embedded_gate,
    evaluate_embedded_retest,
    evaluate_h5_status,
    import_embedded_jsonl,
    import_embedded_retest_summary,
)
from .network import (
    EndBehavior,
    LinkTrace,
    load_trace_manifest,
    scientific_episode_coverage_audit,
)
from .perception import (
    DetectorOnlyBackend,
    HFMultimodalBackend,
    InferenceRequest,
    ModelSpec,
    TemperatureCalibrator,
    UltralyticsDetectorTracker,
    VISDRONE_REGISTERED_MAX_NEW_TOKENS,
    calibrated_semantic_packet_cache,
    calibration_metrics,
    detector_execution_revision,
    prompt_for_mission,
    run_detector_cache,
)
from .protocol import PayloadMode
from .policy_reducer import (
    PolicyReductionResult,
    compact_two_way_clustered_paired_bootstrap,
    reduce_policy_trial_partitions,
)
from .registry import (
    RunContext,
    StudyGateError,
    atomic_write_frame,
    atomic_write_json,
    atomic_write_text,
    frame_stage,
    hash_file,
    hash_object,
    matched_coverage_semantic_errors,
)
from .visdrone_models import (
    model_loader_environment,
    registered_model_load_reference,
    verify_detector_checkpoint,
    verify_huggingface_model_lock,
)
from .visdrone_measurement import (
    RepeatedMeasurementPerceptionBackend,
    run_repeated_detector_measurement,
)
from .visdrone_paper import load_untouched_test_permit
from .visdrone_workflow import registered_temporal_sampling_spec
from .visdrone_receiver_runtime import (
    RegisteredVisDroneNativeReceiver,
    ResNet50Layer3Encoder,
    load_frozen_feature_head,
    nvml_measurement_runner,
    prepare_feature_head_artifact,
    registered_receiver_context,
    train_and_freeze_feature_head,
)
from .visdrone_receivers import (
    FeatureCacheDesign,
    NativeReceiverDesign,
    build_native_receiver_cache as produce_native_receiver_cache,
    build_registered_feature_cache as produce_feature_cache,
    validate_native_receiver_cache,
    validate_registered_feature_cache,
)


RUNNER_VERSION = "semuav.visdrone-paper-runner.v7"
# The sender/receiver measurement producers are unchanged in v3r2, so
# RUNNER_VERSION (bound into measured native-receiver rows) is unchanged.  The
# controller and gate revision is identified separately.
CONTROLLER_REVISION_V3R2 = "v3r2-calibration-aware-escalation-and-balanced-profiles"
CALIBRATION_GATE_AMENDMENT_V3R2 = "PRE_TEST_CALIBRATION_GATE_AMENDMENT_20261013_V3R2"
TEST_RUNTIME_IDENTITY_SCHEMA = "semuav.visdrone-test-runtime-identity.v1"

# VisDrone VLMs emit a bounded proposal decision.  The transmitted semantic
# packet is assembled deterministically from that decision and the frozen
# detector proposals.  These labels travel with selection and hypothesis
# artifacts so pipeline-level measurements cannot later be represented as
# free-form VLM-authorship measurements.
COMPOSED_PIPELINE_METRIC_SCOPE = "composed_detector_verifier_pipeline"
DETECTOR_ONLY_METRIC_SCOPE = "detector_only_pipeline"
COMPOSED_PIPELINE_CLAIM_SCOPE = (
    "detector_proposal_conditioned_end_to_end_pipeline_not_free_form_vlm_generation"
)
TEMPORAL_SAMPLE_SCHEMA = "semuav.visdrone-temporal-controller.v1"
H5_NOT_APPLICABLE_STATUS = "NOT_APPLICABLE_NO_TARGET_EVIDENCE"


def _section(cfg: Any, name: str) -> Mapping[str, Any]:
    value = getattr(cfg, name, None)
    if value is None and isinstance(cfg, Mapping):
        value = cfg.get(name)
    # ``StudyConfig`` intentionally uses frozen dataclasses for a few core
    # sections (notably ``split``) and mappings for extensible paper sections.
    # Normalize both representations here so every scientific gate reads the
    # same registered values without a permissive fallback.
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if not isinstance(value, Mapping):
        raise StudyGateError(f"Configuration section {name!r} is missing")
    return value


def _normalize_measurement_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize measured-runtime metadata into strict native JSON values.

    Pandas materializes integral detector metadata as ``numpy.int64``.  The
    stage cache key must preserve its numeric meaning while remaining portable
    JSON, so NumPy scalars are converted recursively to their native scalar.
    Unsupported objects and non-finite floats fail closed; they are never
    silently stringified into a different cache identity.
    """

    if not isinstance(metadata, Mapping):
        raise StudyGateError("Measurement metadata must be a mapping")

    def normalize(value: Any, path: str) -> Any:
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            if value.dtype.itemsize > np.dtype(np.float64).itemsize:
                raise StudyGateError(
                    "Measurement metadata contains unsupported extended-precision "
                    f"value at {path}: {type(value).__name__}"
                )
            return normalize(float(value), path)
        if isinstance(value, np.str_):
            return str(value)
        if isinstance(value, np.generic):
            raise StudyGateError(
                "Measurement metadata contains unsupported NumPy scalar at "
                f"{path}: {type(value).__name__}"
            )
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise StudyGateError(
                    f"Measurement metadata contains non-finite value at {path}"
                )
            return value
        if isinstance(value, Mapping):
            normalized_mapping: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str) or not key:
                    raise StudyGateError(
                        f"Measurement metadata has invalid key at {path}"
                    )
                normalized_mapping[key] = normalize(item, f"{path}.{key}")
            return normalized_mapping
        if isinstance(value, (list, tuple)):
            return [
                normalize(item, f"{path}[{index}]")
                for index, item in enumerate(value)
            ]
        raise StudyGateError(
            "Measurement metadata contains unsupported value at "
            f"{path}: {type(value).__name__}"
        )

    normalized = normalize(metadata, "measurement")
    if not isinstance(normalized, dict):  # defensive: the root is a Mapping
        raise StudyGateError("Measurement metadata did not normalize to an object")
    try:
        json.dumps(
            normalized, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
    except (TypeError, ValueError) as exc:  # defensive independent verifier
        raise StudyGateError("Measurement metadata is not strict JSON") from exc
    return normalized


def _seed(cfg: Any) -> int:
    value = getattr(cfg, "seed", None)
    if value is None and isinstance(cfg, Mapping):
        value = cfg.get("seed")
    return int(value)


def _seeds(cfg: Any) -> tuple[int, ...]:
    value = getattr(cfg, "seeds", None)
    if value is None and isinstance(cfg, Mapping):
        value = cfg.get("seeds")
    seeds = tuple(int(item) for item in (value or ()))
    if not seeds or len(set(seeds)) != len(seeds):
        raise StudyGateError("Registered replicate seeds are missing or duplicated")
    return seeds


def _results(ctx: RunContext) -> Path:
    path = ctx.run_root / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_frame(ctx: RunContext, name: str, frame: pd.DataFrame) -> Path:
    return atomic_write_frame(_results(ctx) / name, frame)



ExperimentCallback = Callable[[int, str, Mapping[str, pd.DataFrame]], None]


def _emit_experiment_complete(
    callback: ExperimentCallback | None,
    experiment: int,
    phase: str,
    frames: Mapping[str, pd.DataFrame],
) -> None:
    """Synchronously publish presentation-only snapshots at experiment boundaries.

    The callback cannot influence policy selection or mutate canonical results.
    It runs before the next experiment begins.  A notebook may catch and record
    display/export errors inside its callback without rerunning scientific work.
    No callback opens the test ledger, selects test settings, or reads new data.
    """

    if callback is not None:
        if not callable(callback):
            raise TypeError("on_experiment_complete must be callable or None")
        callback(experiment, phase, {key: frame.copy(deep=True) for key, frame in frames.items()})

def _stable_rank(seed: int, *values: Any) -> str:
    return hash_object({"seed": int(seed), "identity": list(map(str, values))})


def _require_registered_phase_sample(
    manifest: pd.DataFrame,
    cfg: Any,
    *,
    expected_splits: Sequence[str],
    test_permit: Any | None = None,
    require_validation_coverage: bool = False,
) -> pd.DataFrame:
    """Validate a notebook-persisted registered sample without reselecting it."""

    if not isinstance(manifest, pd.DataFrame) or manifest.empty:
        raise StudyGateError("Registered phase sample is missing or empty")
    required = {
        "sample_id", "split", "component", "visual_group_id",
        "mission_profile", "gold_action", "image_path",
    }
    if missing := required.difference(manifest.columns):
        raise StudyGateError(f"Registered phase sample schema is incomplete: {sorted(missing)}")
    actual_splits = set(manifest["split"].astype(str))
    expected = set(map(str, expected_splits))
    if actual_splits != expected:
        raise StudyGateError(
            f"Registered phase sample has splits {sorted(actual_splits)}; expected {sorted(expected)}"
        )
    if manifest["sample_id"].astype(str).duplicated().any():
        raise StudyGateError("Registered phase sample IDs are duplicated")
    sampling = _section(cfg, "benchmark").get("analysis_sampling")
    if not isinstance(sampling, Mapping) or sampling.get("method") != "outcome_blind_group_hash_temporal_v1":
        raise StudyGateError("Registered outcome-blind analysis sampling is missing")
    group_caps = sampling.get("max_groups_per_component")
    frame_caps = sampling.get("max_frames_per_group")
    if not isinstance(group_caps, Mapping) or not isinstance(frame_caps, Mapping):
        raise StudyGateError("Registered analysis sampling caps are incomplete")
    source_column = "source_sample_id" if "source_sample_id" in manifest else (
        "source_uid" if "source_uid" in manifest else "image_path"
    )
    registered_profiles = set(map(str, _priority_profiles(cfg)))
    for split in expected:
        scoped = manifest[manifest["split"].astype(str).eq(split)]
        group_cap = int(group_caps[split])
        frame_cap = int(frame_caps[split])
        counts = scoped.groupby("component")["visual_group_id"].nunique()
        if counts.gt(group_cap).any():
            raise StudyGateError(f"{split} sample exceeds its frozen group cap")
        frame_counts = scoped.groupby(["component", "visual_group_id"])[source_column].nunique()
        if frame_counts.gt(frame_cap).any():
            raise StudyGateError(f"{split} sample exceeds its frozen per-group frame cap")
        profile_counts = scoped.groupby(source_column)["mission_profile"].nunique()
        observed_profiles = set(scoped["mission_profile"].astype(str))
        if observed_profiles != registered_profiles:
            raise StudyGateError(
                f"{split} sample profiles {sorted(observed_profiles)} do not match "
                f"the frozen profiles {sorted(registered_profiles)}"
            )
        if profile_counts.ne(len(registered_profiles)).any():
            raise StudyGateError(f"{split} sample is not mission-profile paired")
        component_counts = {
            str(component): int(value) for component, value in counts.items()
        }
        natural_groups = int(
            scoped[["component", "visual_group_id"]].astype(str).drop_duplicates().shape[0]
        )
        if split == "validation" and require_validation_coverage:
            expected_components = {"det_val", "mot_val"}
            if set(component_counts) != expected_components:
                raise StudyGateError(
                    "Validation sample must include both frozen publisher components"
                )
            if natural_groups < 15 or any(value < 1 for value in component_counts.values()):
                raise StudyGateError(
                    "Validation sample requires at least 15 natural groups total and "
                    "at least one group from each publisher component"
                )
        elif split == "test":
            required_counts = {"det_test_dev": group_cap, "mot_test_dev": group_cap}
            if component_counts != required_counts:
                raise StudyGateError(
                    "Confirmatory sample must contain exactly the frozen balanced "
                    f"component groups {required_counts}; observed {component_counts}"
                )
    if expected == {"test"}:
        if test_permit is None:
            raise StudyGateError("Test sample cannot be inspected without an OPEN permit")
        if "test_ledger_sha256" not in manifest or set(
            manifest["test_ledger_sha256"].astype(str)
        ) != {str(test_permit.ledger_sha256)}:
            raise StudyGateError("Test sample is not bound to the OPEN ledger")
        evaluation = _section(cfg, "evaluation")
        split_cfg = _section(cfg, "split")
        split_min_units = int(split_cfg.get("min_test_units", 0))
        split_min_positive = int(split_cfg.get("min_critical_test_events", 0))
        min_units = max(split_min_units, int(evaluation["natural_test_units"]))
        min_positive = max(
            split_min_positive,
            int(evaluation["mission_priority_positive_units"]),
        )
        min_negative = int(evaluation["mission_priority_negative_units"])
        independent_units = int(
            manifest[["component", "visual_group_id"]].astype(str)
            .drop_duplicates().shape[0]
        )
        positive_mask = (
            manifest["critical"].astype(bool)
            if "critical" in manifest
            else manifest["gold_action"].astype(str).eq("RESPOND")
        )
        positives = manifest.loc[positive_mask, "visual_group_id"].astype(str).nunique()
        negatives = manifest.loc[~positive_mask, "visual_group_id"].astype(str).nunique()
        if independent_units < min_units:
            raise StudyGateError(
                "Confirmatory sample lacks the registered natural independent units: "
                f"{independent_units}/{min_units}. Mission-priority positive/negative "
                f"counts ({positives}/{min_positive}, {negatives}/{min_negative}) "
                "are hypothesis-specific estimability gates and do not block execution."
            )
    return manifest.copy()


def select_paper_perception_sample(
    manifest: pd.DataFrame,
    *,
    splits: Sequence[str],
    seed: int,
    max_groups_per_component_split: int = 500,
    max_frames_per_group: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a profile-paired, outcome-blind bounded perception sample.

    Group ranks use only seed/component/split/group identity.  Frame ranks use
    temporal/source identity.  All mission-profile copies of every selected
    source frame are retained, so models see an exactly paired mission design.
    """

    required = {
        "sample_id", "split", "component", "visual_group_id",
        "mission_profile", "frame_index",
    }
    if missing := required.difference(manifest.columns):
        raise StudyGateError(f"Perception sampling manifest is missing: {sorted(missing)}")
    selected_split = {str(item) for item in splits}
    scoped = manifest[manifest["split"].astype(str).isin(selected_split)].copy()
    if scoped.empty:
        raise StudyGateError(f"No rows exist for registered perception splits {sorted(selected_split)}")
    source_column = "source_uid" if "source_uid" in scoped else "media_path" if "media_path" in scoped else "image_path"
    scoped["__group_rank"] = [
        _stable_rank(seed, row.component, row.split, row.visual_group_id)
        for row in scoped.itertuples(index=False)
    ]
    chosen_groups: set[tuple[str, str, str]] = set()
    group_table = scoped[[
        "component", "split", "visual_group_id", "__group_rank",
    ]].drop_duplicates()
    for (component, split), group in group_table.groupby(["component", "split"], sort=True):
        ranked = group.sort_values(["__group_rank", "visual_group_id"], kind="mergesort")
        for row in ranked.head(int(max_groups_per_component_split)).itertuples(index=False):
            chosen_groups.add((str(component), str(split), str(row.visual_group_id)))
    group_mask = scoped.apply(
        lambda row: (str(row["component"]), str(row["split"]), str(row["visual_group_id"]))
        in chosen_groups,
        axis=1,
    )
    scoped = scoped.loc[group_mask].copy()
    keep_source_ids: set[str] = set()
    for _, group in scoped.groupby(["component", "split", "visual_group_id"], sort=True):
        sources = group[[source_column, "frame_index"]].drop_duplicates(source_column)
        sources = sources.sort_values(["frame_index", source_column], kind="mergesort")
        if len(sources) <= int(max_frames_per_group):
            positions = list(range(len(sources)))
        else:
            positions = sorted({
                int(round(index * (len(sources) - 1) / (int(max_frames_per_group) - 1)))
                for index in range(int(max_frames_per_group))
            }) if int(max_frames_per_group) > 1 else [0]
        keep_source_ids.update(sources.iloc[positions][source_column].astype(str))
    sampled = scoped[scoped[source_column].astype(str).isin(keep_source_ids)].copy()
    expected_profiles = set(scoped["mission_profile"].astype(str))
    coverage = sampled.groupby(source_column)["mission_profile"].apply(
        lambda values: set(map(str, values))
    )
    if any(profiles != expected_profiles for profiles in coverage):
        raise StudyGateError("Outcome-blind sampling broke mission-profile pairing")
    sampled = sampled.drop(columns=["__group_rank"]).sort_values(
        ["split", "component", "visual_group_id", "frame_index", "mission_profile"],
        kind="mergesort",
    ).reset_index(drop=True)
    audit = pd.DataFrame([{
        "runner_version": RUNNER_VERSION,
        "splits": json.dumps(sorted(selected_split)),
        "source_identity_column": source_column,
        "outcome_columns_used": False,
        "max_groups_per_component_split": int(max_groups_per_component_split),
        "max_frames_per_group": int(max_frames_per_group),
        "selected_rows": int(len(sampled)),
        "selected_source_frames": int(sampled[source_column].nunique()),
        "selected_visual_groups": int(sampled["visual_group_id"].nunique()),
        "mission_profiles": json.dumps(sorted(expected_profiles)),
        "sample_manifest_sha256": hash_object(sampled["sample_id"].astype(str).tolist()),
        "passed": bool(len(sampled) and expected_profiles),
    }])
    return sampled, audit


def prepare_and_freeze_feature_head(
    ctx: RunContext,
    cfg: Any,
    manifest: pd.DataFrame,
    device: str,
) -> dict[str, Any]:
    """Train/freeze the registered L4 receiver head on TRAIN rows only."""

    if not ctx.is_scientific:
        raise StudyGateError("Feature-head preparation requires the scientific profile")
    train = _require_registered_phase_sample(
        manifest, cfg, expected_splits=("train",),
    )
    destination = _results(ctx) / "feature_head"
    lock_path = destination / "FEATURE_HEAD_LOCK.json"
    if lock_path.exists():
        raise StudyGateError(
            "Frozen feature-head artifact already exists; do not overwrite it"
        )
    artifact = dict(prepare_feature_head_artifact(
        train, output_dir=destination, device=device, seed=_seed(cfg),
    ))
    artifact.update({
        "training_sample_sha256": hash_object(
            train.sort_values("sample_id", kind="mergesort")["sample_id"].astype(str).tolist()
        ),
        "analysis_sampling_sha256": hash_object(
            _section(cfg, "benchmark")["analysis_sampling"]
        ),
        "temporal_sampling": dict(_section(cfg, "benchmark").get("temporal_sampling", {})),
        "temporal_sampling_sha256": hash_object(
            _section(cfg, "benchmark").get("temporal_sampling", {})
        ),
        "runner_version": RUNNER_VERSION,
    })
    receipt = atomic_write_json(destination / "FEATURE_HEAD_RECEIPT.json", artifact)
    artifact["receipt_path"] = str(receipt)
    artifact["receipt_sha256"] = hash_file(receipt)
    artifact["environment"] = {
        "SEMUAV_FEATURE_RECEIVER_HEAD": str(artifact["lock_path"]),
        "SEMUAV_FEATURE_HEAD_SHA256": str(artifact["lock_sha256"]),
    }
    return artifact


def _model_backend(entry: Mapping[str, Any], device: str, image_size: Sequence[int]) -> Any:
    backend = str(entry.get("backend", ""))
    if backend not in {"qwen3_vl", "smolvlm2"}:
        raise StudyGateError(
            f"v2r13 active model registry contains an unsupported backend: {backend!r}"
        )
    spec = ModelSpec(
        model_id=str(entry["model_id"]),
        repo=registered_model_load_reference(entry),
        revision=str(entry["resolved_revision"]),
        role=str(entry["role"]),
        quantization=str(entry.get("quantization", "none")),
        trust_remote_code=False,
        image_size=(int(image_size[0]), int(image_size[1])),
    )
    return HFMultimodalBackend(spec, device, str(entry.get("dtype", "bfloat16")))


def _bind_runtime_model_lock(
    model_lock: str | Path,
    cfg: Any,
) -> dict[str, Any]:
    """Verify the exact v2r13 model lock and prepare its offline bindings."""

    verified_model_lock = verify_huggingface_model_lock(
        model_lock, models=_section(cfg, "models")
    )
    for key, value in model_loader_environment(model_lock).items():
        os.environ[key] = value
    return verified_model_lock


def _development_warmup_spec(manifest: pd.DataFrame) -> dict[str, Any]:
    """Select one deterministic, frozen development image for unmeasured warmup."""

    required = {
        "sample_id", "image_path", "split", "mission", "capture_time_us",
        "telemetry_json",
    }
    if missing := required.difference(manifest.columns):
        raise StudyGateError(f"Repeated measurement warmup lacks: {sorted(missing)}")
    development = manifest[
        manifest["split"].astype(str).isin(["train", "calibration", "validation"])
    ].copy()
    if development.empty:
        raise StudyGateError("Repeated measurement requires a frozen development warmup")
    source_column = "source_sample_id" if "source_sample_id" in development else "sample_id"
    preferred = development[development["split"].astype(str).eq("calibration")]
    pool = preferred if len(preferred) else development
    row = pool.sort_values(
        [source_column, "sample_id"], kind="mergesort",
    ).iloc[0]
    image_path = Path(str(row["image_path"])).expanduser().resolve()
    if not image_path.is_file():
        raise StudyGateError(f"Frozen warmup image is missing: {image_path}")
    return {
        "sample_id": str(row["sample_id"]),
        "source_id": str(row[source_column]),
        "image_path": str(image_path),
        "image_sha256": hash_file(image_path),
        "split": str(row["split"]),
        "mission": str(row["mission"]),
        "capture_time_us": int(row["capture_time_us"]),
        "telemetry_json": str(row["telemetry_json"]),
        "selection": "lexicographic_first_frozen_calibration_source_v1",
    }


def _warmup_request(
    spec: Mapping[str, Any],
    detector_cache: pd.DataFrame | None = None,
) -> InferenceRequest:
    proposals: Sequence[Mapping[str, Any]] = ()
    if "detector_proposals_json" in spec:
        proposals = tuple(json.loads(str(spec["detector_proposals_json"])))
    elif detector_cache is not None:
        match = detector_cache[
            detector_cache["sample_id"].astype(str).eq(str(spec["sample_id"]))
        ]
        if match.empty:
            raise StudyGateError("Frozen VLM warmup lacks detector proposals")
        proposals = tuple(json.loads(str(match.iloc[0]["proposals_json"])))
    return InferenceRequest(
        observation_id=str(spec["sample_id"]),
        image_path=str(spec["image_path"]),
        mission=str(spec["mission"]),
        capture_time_us=int(spec["capture_time_us"]),
        telemetry=json.loads(str(spec["telemetry_json"])),
        detector_proposals=proposals,
    )


def _detector_and_cache(
    ctx: RunContext,
    cfg: Any,
    manifest: pd.DataFrame,
    device: str,
    *,
    stage_prefix: str,
    warmup_spec: Mapping[str, Any],
) -> tuple[UltralyticsDetectorTracker, pd.DataFrame, str, Mapping[str, Any]]:
    detector_cfg = _section(cfg, "models").get("detector")
    if not isinstance(detector_cfg, Mapping):
        raise StudyGateError("Registered detector configuration is missing")
    checkpoint = Path(str(detector_cfg.get("checkpoint", ""))).expanduser().resolve()
    verification = verify_detector_checkpoint(
        checkpoint, str(detector_cfg.get("checkpoint_sha256", "")),
        training_receipt=str(detector_cfg.get("training_receipt", "")),
    )
    proposal_cfg = _section(cfg, "models").get("proposal_policy")
    if not isinstance(proposal_cfg, Mapping):
        raise StudyGateError("Registered detector proposal policy is missing")
    tracker = str(detector_cfg["tracker"])
    inference_kwargs = {
        "image_size": int(detector_cfg["image_size"]),
        "confidence": float(detector_cfg["confidence"]),
        "iou": float(detector_cfg["iou"]),
        "max_detections": int(detector_cfg["max_detections"]),
        "seed": int(detector_cfg["seed"]),
        "top_k": int(proposal_cfg["top_k"]),
        "minimum_confidence": float(proposal_cfg["minimum_confidence"]),
        "ordering": str(proposal_cfg["ordering"]),
    }
    revision = detector_execution_revision(
        str(verification["sha256"]), tracker, **inference_kwargs,
    )

    def detector_factory() -> UltralyticsDetectorTracker:
        return UltralyticsDetectorTracker(
            str(checkpoint), tracker, device, **inference_kwargs,
        )

    runtime = _section(cfg, "runtime")
    key = {
        "checkpoint_sha256": verification["sha256"],
        "tracker": tracker,
        "detector_revision": revision,
        "inference_and_proposal_policy": inference_kwargs,
        "measurement": {
            "warmup_iterations": int(runtime["warmup_iterations"]),
            "repetitions": int(runtime["steady_state_repetitions"]),
            "warmup_source_id": str(warmup_spec["source_id"]),
            "warmup_image_sha256": str(warmup_spec["image_sha256"]),
            "warmup_split": str(warmup_spec["split"]),
        },
        "samples": [
            {"sample_id": str(row.sample_id), "image_sha256": hash_file(row.image_path)}
            for row in manifest.itertuples(index=False)
        ],
    }
    cache = frame_stage(
        ctx, f"{stage_prefix}_detector", key,
        lambda: run_repeated_detector_measurement(
            detector_factory, manifest,
            warmup_image_path=str(warmup_spec["image_path"]),
            warmup_image_sha256=str(warmup_spec["image_sha256"]),
            warmup_source_id=str(warmup_spec["source_id"]),
            warmup_split=str(warmup_spec["split"]),
            warmup_iterations=int(runtime["warmup_iterations"]),
            repetitions=int(runtime["steady_state_repetitions"]),
            device_index=int(device.split(":")[-1]) if device.startswith("cuda") else 0,
        ),
    )
    # Materialize and verify detector residency even when frame_stage restored
    # the proposals from cache. Otherwise the following one-at-a-time VLM
    # measurement could silently omit the registered always-on detector.
    detector = detector_factory()
    observed_residency = detector.ensure_resident()
    residency_payload = {
        "runner_version": RUNNER_VERSION,
        "stage_prefix": str(stage_prefix),
        "detector_revision": revision,
        "detector_checkpoint_sha256": str(verification["sha256"]),
        "inference_and_proposal_policy": inference_kwargs,
        "measurement_warmup_iterations": int(runtime["warmup_iterations"]),
        "measurement_repetitions": int(runtime["steady_state_repetitions"]),
        "measurement_warmup_image_sha256": str(warmup_spec["image_sha256"]),
        **dict(observed_residency),
    }
    residency_path = atomic_write_json(
        _results(ctx) / f"{stage_prefix}_detector_residency.json",
        residency_payload,
    )
    verified = {
        **dict(verification),
        "residency": residency_payload,
        "residency_receipt_path": str(residency_path),
        "residency_receipt_sha256": hash_file(residency_path),
    }
    return detector, cache, revision, verified


def _detector_lineage_binding(verification: Mapping[str, Any]) -> dict[str, str]:
    """Extract the immutable training lineage bound to a detector checkpoint."""

    fields = (
        "sha256", "training_receipt", "training_receipt_sha256",
        "training_plan_sha256", "training_environment_sha256",
    )
    binding = {field: str(verification.get(field, "")).strip() for field in fields}
    if any(not value for value in binding.values()):
        raise StudyGateError(
            "Scientific detector execution requires a complete verified training lineage"
        )
    return binding


def _fit_validation_selection(
    detail: pd.DataFrame,
    cfg: Any,
    locked_models: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    calibration = detail[detail["split"].astype(str).eq("calibration")].copy()
    validation = detail[detail["split"].astype(str).eq("validation")].copy()
    if calibration.empty or validation.empty:
        raise StudyGateError("Model selection requires frozen calibration and validation rows")
    calibration_payload: dict[str, Any] = {}
    gate_rows: list[dict[str, Any]] = []
    limits = _section(cfg, "evaluation")

    def weighted_quantiles(
        values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray,
    ) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ordered_values = values[order]
        ordered_weights = weights[order]
        cumulative = np.cumsum(ordered_weights)
        if cumulative[-1] <= 0:
            raise StudyGateError("Calibration weights have zero total mass")
        cumulative = (cumulative - 0.5 * ordered_weights) / cumulative[-1]
        return np.interp(quantiles, cumulative, ordered_values)

    def group_equal_metrics(
        probabilities: Sequence[float], labels: Sequence[int], groups: Sequence[str],
    ) -> dict[str, float]:
        frame = pd.DataFrame({
            "p": np.asarray(probabilities, dtype=float),
            "y": np.asarray(labels, dtype=float),
            "group": np.asarray(list(groups), dtype=str),
        })
        if frame.empty or not np.isfinite(frame[["p", "y"]].to_numpy()).all():
            return {"brier": np.nan, "nll": np.nan, "ece_15_equal_mass": np.nan}
        frame["p"] = frame["p"].clip(1e-8, 1 - 1e-8)
        counts = frame.groupby("group", sort=False)["group"].transform("size")
        group_count = int(frame["group"].nunique())
        if group_count <= 0:
            return {"brier": np.nan, "nll": np.nan, "ece_15_equal_mass": np.nan}
        weights = (1.0 / counts.to_numpy(float)) / float(group_count)
        p = frame["p"].to_numpy(float)
        y = frame["y"].to_numpy(float)
        edges = np.unique(weighted_quantiles(
            p, weights, np.linspace(0.0, 1.0, 16),
        ))
        if len(edges) < 2:
            ece = float(abs(np.average(p, weights=weights) - np.average(y, weights=weights)))
        else:
            bins = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
            ece = 0.0
            for bin_id in np.unique(bins):
                mask = bins == bin_id
                mass = float(weights[mask].sum())
                if mass <= 0:
                    continue
                ece += mass * abs(
                    float(np.average(p[mask], weights=weights[mask]))
                    - float(np.average(y[mask], weights=weights[mask]))
                )
        return {
            "brier": float(np.average((p - y) ** 2, weights=weights)),
            "nll": float(np.average(-(y * np.log(p) + (1 - y) * np.log(1 - p)), weights=weights)),
            "ece_15_equal_mass": float(ece),
        }

    def group_bootstrap_ece_upper(
        probabilities: Sequence[float], labels: Sequence[int], groups: Sequence[str],
        *, model_id: str, endpoint: str,
    ) -> tuple[float, float, int]:
        frame = pd.DataFrame({
            "p": np.asarray(probabilities, dtype=float),
            "y": np.asarray(labels, dtype=int),
            "group": np.asarray(list(groups), dtype=str),
        })
        unique_groups = tuple(sorted(frame["group"].unique()))
        if not unique_groups:
            return np.nan, np.nan, 0
        point = group_equal_metrics(frame["p"], frame["y"], frame["group"])[
            "ece_15_equal_mass"
        ]
        repetitions = int(limits.get("calibration_bootstrap_repetitions", 2000))
        upper_quantile = float(limits.get("calibration_bootstrap_upper_quantile", 0.95))
        if repetitions <= 0 or not 0.5 < upper_quantile < 1.0:
            raise StudyGateError("Registered calibration bootstrap settings are invalid")
        seed_value = int(hash_object({
            "study_seed": _seed(cfg), "model_id": str(model_id),
            "endpoint": str(endpoint), "groups": list(unique_groups),
        })[:16], 16) % (2**32)
        rng = np.random.default_rng(seed_value)
        by_group = {
            group: frame.loc[frame["group"].eq(group)].copy()
            for group in unique_groups
        }
        estimates = np.empty(repetitions, dtype=float)
        for index in range(repetitions):
            sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            parts: list[pd.DataFrame] = []
            for draw, group in enumerate(sampled):
                part = by_group[str(group)].copy()
                # Treat repeated selections as independent bootstrap clusters.
                part["bootstrap_group"] = f"{draw}:{group}"
                parts.append(part)
            replicate = pd.concat(parts, ignore_index=True)
            estimates[index] = group_equal_metrics(
                replicate["p"], replicate["y"], replicate["bootstrap_group"],
            )["ece_15_equal_mass"]
        return point, float(np.quantile(estimates, upper_quantile)), len(unique_groups)

    for model_id, group in calibration.groupby("model_id", sort=True):
        observed = group[pd.to_numeric(group["confidence"], errors="coerce").notna()]
        if observed.empty:
            continue
        calibration_fitting_estimand = str(
            limits.get("calibration_fitting_estimand", "")
        )
        if calibration_fitting_estimand != "visual_group_equal_temperature_nll_v1":
            raise StudyGateError("Registered calibration fitting estimand changed")
        group_sizes = observed.groupby("visual_group_id", sort=False)[
            "visual_group_id"
        ].transform("size").astype(float)
        calibration_weights = 1.0 / group_sizes
        calibrator = TemperatureCalibrator().fit(
            observed["confidence"], observed["action_correct"].astype(int),
            sample_weight=calibration_weights,
        )
        validate_rows = validation[
            validation["model_id"].astype(str).eq(str(model_id))
            & pd.to_numeric(validation["confidence"], errors="coerce").notna()
        ]
        if validate_rows.empty:
            continue
        before = group_equal_metrics(
            validate_rows["confidence"], validate_rows["action_correct"].astype(int),
            validate_rows["visual_group_id"].astype(str),
        )
        probabilities = calibrator.transform(validate_rows["confidence"])
        after = group_equal_metrics(
            probabilities, validate_rows["action_correct"].astype(int),
            validate_rows["visual_group_id"].astype(str),
        )
        ece_point, ece_upper, validation_groups = group_bootstrap_ece_upper(
            probabilities, validate_rows["action_correct"].astype(int),
            validate_rows["visual_group_id"].astype(str),
            model_id=str(model_id), endpoint="all_validation",
        )
        priority = validate_rows[validate_rows["critical"].astype(bool)]
        priority_ece, priority_ece_upper, priority_groups = (
            group_bootstrap_ece_upper(
                calibrator.transform(priority["confidence"]),
                priority["action_correct"].astype(int),
                priority["visual_group_id"].astype(str),
                model_id=str(model_id), endpoint="priority_validation",
            )
            if len(priority) else (np.nan, np.nan, 0)
        )
        calibration_groups = int(observed["visual_group_id"].astype(str).nunique())
        incorrect_groups = observed.loc[
            ~observed["action_correct"].astype(bool), "visual_group_id",
        ].nunique()
        minimum_groups = int(limits.get("minimum_calibration_independent_groups", 30))
        diagnostic_target = int(limits.get(
            "calibration_incorrect_group_diagnostic_target", 50,
        ))
        passed = bool(
            calibration_groups >= minimum_groups
            and validation_groups >= minimum_groups
            and np.isfinite(ece_upper)
            and ece_upper <= float(limits.get("calibration_ece_upper_limit", 0.05))
            and np.isfinite(priority_ece_upper)
            and priority_ece_upper <= float(
                limits.get("calibration_priority_ece_upper_limit", 0.10)
            )
            and after["brier"] <= before["brier"]
            and len(validate_rows) == len(validation[validation["model_id"].eq(model_id)])
        )
        calibration_payload[str(model_id)] = {
            "temperature": float(calibrator.temperature),
            "confidence_threshold": _h4_high_confidence_threshold(cfg),
            "fitting_estimand": calibration_fitting_estimand,
        }
        gate_rows.append({
            "model_id": str(model_id), "ece": float(ece_point),
            "ece_group_bootstrap_upper": float(ece_upper),
            "priority_ece": float(priority_ece),
            "priority_ece_group_bootstrap_upper": float(priority_ece_upper),
            "calibration_independent_visual_groups": calibration_groups,
            "validation_independent_visual_groups": int(validation_groups),
            "priority_validation_independent_visual_groups": int(priority_groups),
            "minimum_calibration_independent_groups": minimum_groups,
            "calibration_incorrect_group_diagnostic_target": diagnostic_target,
            "calibration_incorrect_group_diagnostic_met": bool(
                incorrect_groups >= diagnostic_target
            ),
            "calibration_incorrect_group_diagnostic_is_blocking": False,
            "bootstrap_repetitions": int(limits.get("calibration_bootstrap_repetitions", 2000)),
            "bootstrap_upper_quantile": float(limits.get("calibration_bootstrap_upper_quantile", 0.95)),
            "brier_before": float(before["brier"]),
            "brier_after": float(after["brier"]),
            "calibration_incorrect_visual_groups": int(incorrect_groups),
            "calibration_fitting_estimand": calibration_fitting_estimand,
            "passed": passed,
        })
    required_metric_scope = {
        "action_metric_scope", "event_metric_scope",
        "object_metric_scope", "relation_metric_scope",
    }
    if missing := required_metric_scope.difference(validation.columns):
        raise StudyGateError(
            "Model selection lacks composed-pipeline metric scope: "
            f"{sorted(missing)}"
        )
    context_rows = validation[~validation["model_id"].astype(str).eq("detector_only")]
    detector_rows = validation[validation["model_id"].astype(str).eq("detector_only")]
    for column in sorted(required_metric_scope):
        if not context_rows[column].astype(str).eq(COMPOSED_PIPELINE_METRIC_SCOPE).all():
            raise StudyGateError(
                f"Context-model {column} must be {COMPOSED_PIPELINE_METRIC_SCOPE!r}"
            )
        if not detector_rows[column].astype(str).eq(DETECTOR_ONLY_METRIC_SCOPE).all():
            raise StudyGateError(
                f"Detector-only {column} must be {DETECTOR_ONLY_METRIC_SCOPE!r}"
            )

    summary = validation.groupby("model_id", sort=True).agg(
        action_accuracy=("action_correct", "mean"),
        event_accuracy=("event_correct", "mean"),
        objects_exact_accuracy_frame_weighted_secondary=("objects_exact", "mean"),
        relations_exact_accuracy_frame_weighted_secondary=("relations_exact", "mean"),
        schema_valid_rate=("schema_valid", "mean"),
        semantic_valid_rate=("semantic_valid", "mean"),
        authoritative_consistency_rate=("authoritative_consistent", "mean"),
        full_path_latency_p95_ms=("full_path_latency_ms", lambda values: float(values.quantile(0.95))),
        peak_memory_mb=("peak_memory_mb", "max"),
    ).reset_index()
    priority_rows = validation[validation["critical"].astype(bool)].copy()
    frame_priority_recall = priority_rows.groupby("model_id")["action_correct"].mean()
    group_priority_recall = (
        priority_rows.groupby(["model_id", "visual_group_id"], sort=True)[
            "action_correct"
        ].mean().groupby("model_id", sort=True).mean()
    )
    summary["priority_action_recall"] = summary["model_id"].map(group_priority_recall)
    summary["priority_action_recall_frame_weighted_secondary"] = summary["model_id"].map(
        frame_priority_recall
    )
    summary["priority_action_recall_estimand"] = (
        "group_equal_critical_action_recall_v1"
    )
    selection_spec = limits.get("model_selection")
    if not isinstance(selection_spec, Mapping):
        raise StudyGateError("Frozen model-selection estimand is missing")
    expected_estimand = "group_equal_critical_action_plus_object_exact_v2"
    if str(selection_spec.get("estimand")) != expected_estimand:
        raise StudyGateError("Unregistered model-selection estimand")
    weights = selection_spec.get("weights")
    if not isinstance(weights, Mapping) or set(weights) != {
        "critical_action_accuracy", "objects_exact_accuracy",
    }:
        raise StudyGateError("Model-selection weights are incomplete")
    critical_weight = float(weights["critical_action_accuracy"])
    object_weight = float(weights["objects_exact_accuracy"])
    if (
        critical_weight < 0 or object_weight < 0
        or not math.isclose(critical_weight + object_weight, 1.0, abs_tol=1e-12)
    ):
        raise StudyGateError("Model-selection weights must be nonnegative and sum to one")
    expected_tie_break = [
        "selection_score_desc", "critical_action_accuracy_desc",
        "objects_exact_accuracy_desc", "full_path_latency_p95_ms_asc",
        "model_id_asc",
    ]
    if list(selection_spec.get("tie_break", ())) != expected_tie_break:
        raise StudyGateError("Model-selection tie-break differs from the frozen order")

    def group_equal_metric(rows: pd.DataFrame, column: str) -> pd.Series:
        usable = rows.loc[rows[column].notna(), ["model_id", "visual_group_id", column]].copy()
        usable[column] = pd.to_numeric(usable[column], errors="coerce")
        usable = usable[usable[column].notna()]
        if usable.empty:
            return pd.Series(dtype=float)
        return (
            usable.groupby(["model_id", "visual_group_id"], sort=True)[column].mean()
            .groupby("model_id", sort=True).mean()
        )

    critical_action = group_equal_metric(
        validation[validation["critical"].astype(bool)], "action_correct",
    )
    objects_exact = group_equal_metric(validation, "objects_exact")
    summary["critical_action_accuracy"] = summary["model_id"].map(critical_action)
    summary["objects_exact_accuracy"] = summary["model_id"].map(objects_exact)
    summary["selection_score"] = (
        critical_weight * summary["critical_action_accuracy"]
        + object_weight * summary["objects_exact_accuracy"]
    )
    summary["selection_metric_scope"] = COMPOSED_PIPELINE_METRIC_SCOPE
    summary["selection_claim_scope"] = COMPOSED_PIPELINE_CLAIM_SCOPE
    summary["objects_exact_metric_definition"] = (
        "multiset_object_label_exact_geometry_ignored_"
        "conditioned_on_frozen_detector_proposals"
    )
    summary["free_form_vlm_packet_authorship_claim_eligible"] = False
    gate_lookup = {row["model_id"]: bool(row["passed"]) for row in gate_rows}
    lock_lookup = {str(item["model_id"]): item for item in locked_models}
    compact = [
        model_id for model_id, item in lock_lookup.items()
        if str(item.get("role")) == "compact" and bool(item.get("calibration_eligible", True))
    ]
    eligible: list[str] = []
    for row in summary.itertuples(index=False):
        model_id = str(row.model_id)
        if model_id not in compact:
            continue
        if (
            np.isfinite(float(row.selection_score))
            and
            gate_lookup.get(model_id, False)
            and float(row.full_path_latency_p95_ms) <= float(limits.get("max_vlm_full_path_p95_latency_ms", 1000))
            and float(row.peak_memory_mb) <= float(limits.get("max_vlm_peak_memory_mb", 12000))
            and float(row.schema_valid_rate) >= float(limits.get("min_schema_valid_rate", 0.98))
            and float(row.semantic_valid_rate) >= float(limits.get("min_semantic_valid_rate", 0.98))
            and float(row.authoritative_consistency_rate) == 1.0
            and float(row.priority_action_recall) >= float(limits.get("min_priority_recall", 0.85))
        ):
            eligible.append(model_id)
    if not compact:
        raise StudyGateError("MODEL_LOCK has no registered compact candidate")
    ranked = summary.sort_values(
        [
            "selection_score", "critical_action_accuracy", "objects_exact_accuracy",
            "full_path_latency_p95_ms", "model_id",
        ],
        ascending=[False, False, False, True, True], kind="mergesort",
        na_position="last",
    )
    unconstrained_rows = ranked[ranked["model_id"].astype(str).isin(compact)]
    if unconstrained_rows.empty:
        raise StudyGateError("Registered compact candidates have no validation results")
    unconstrained_best = str(unconstrained_rows.iloc[0]["model_id"])
    eligible_rows = ranked[ranked["model_id"].astype(str).isin(eligible)]
    best_eligible = str(eligible_rows.iloc[0]["model_id"]) if len(eligible_rows) else None
    calibrated_compact_rows = ranked[
        ranked["model_id"].astype(str).isin(
            set(compact).intersection(calibration_payload)
        )
    ]
    if calibrated_compact_rows.empty:
        raise StudyGateError(
            "Registered compact candidates have no validated calibration identity"
        )
    selected_compact_model = str(
        best_eligible or calibrated_compact_rows.iloc[0]["model_id"]
    )
    score_lookup = dict(zip(
        summary["model_id"].astype(str),
        pd.to_numeric(summary["selection_score"], errors="coerce"),
    ))
    detector_score = float(score_lookup.get("detector_only", np.nan))
    candidate_score = (
        float(score_lookup.get(best_eligible, np.nan)) if best_eligible is not None else np.nan
    )
    gain = candidate_score - detector_score
    vlm_enabled = bool(
        best_eligible is not None
        and np.isfinite(gain)
        and gain >= float(limits.get("minimum_vlm_context_gain", 0.03))
    )
    selected = str(best_eligible) if vlm_enabled else "detector_only"
    if selected not in calibration_payload:
        raise StudyGateError("Frozen selected model has no usable calibration")
    return {
        "runner_version": RUNNER_VERSION,
        "selected_model": selected,
        "selected_compact_model": selected_compact_model,
        "temporal_experiment1_model_ids": [
            "detector_only", selected_compact_model,
        ],
        "temporal_experiment1_model_count": 2,
        "best_compact_vlm": selected_compact_model,
        "best_eligible_compact_vlm": best_eligible,
        "unconstrained_best_compact_vlm": unconstrained_best,
        "vlm_claim_enabled": vlm_enabled,
        "vlm_gain_over_detector": float(gain),
        "model_selection": {
            "estimand": expected_estimand,
            "weights": {
                "critical_action_accuracy": critical_weight,
                "objects_exact_accuracy": object_weight,
            },
            "tie_break": expected_tie_break,
            "metric_scope": COMPOSED_PIPELINE_METRIC_SCOPE,
            "claim_scope": COMPOSED_PIPELINE_CLAIM_SCOPE,
            "objects_exact_metric_definition": (
                "multiset_object_label_exact_geometry_ignored_"
                "conditioned_on_frozen_detector_proposals"
            ),
            "free_form_vlm_packet_authorship_claim_eligible": False,
        },
        "calibration_claim_enabled": gate_lookup.get(selected, False),
        "calibration": calibration_payload,
        "calibration_gate": gate_rows,
        "eligible_compact_models": eligible,
        "selection_summary": summary.to_dict("records"),
        "upper_bound_model": next(
            str(item["model_id"]) for item in locked_models if item.get("role") == "upper_bound"
        ),
    }


def _require_selected_compact_model(
    selection: Mapping[str, Any],
    locked_models: Sequence[Mapping[str, Any]],
) -> str:
    """Return the frozen compact comparison identity, independent of fallback.

    ``selected_model`` remains the deployment/claim decision and may validly be
    ``detector_only``.  Temporal Experiment 1 nevertheless has a distinct,
    validation-ranked compact route whose identity is frozen here.
    """

    selected_compact = str(selection.get("selected_compact_model", "")).strip()
    expected_routes = ["detector_only", selected_compact]
    matches = [
        item
        for item in locked_models
        if isinstance(item, Mapping)
        and str(item.get("model_id", "")) == selected_compact
    ]
    calibration = selection.get("calibration")
    summary = selection.get("selection_summary")
    summary_matches = [
        row
        for row in summary
        if isinstance(row, Mapping)
        and str(row.get("model_id", "")) == selected_compact
    ] if isinstance(summary, Sequence) and not isinstance(summary, (str, bytes)) else []
    if (
        not selected_compact
        or selected_compact == "detector_only"
        or len(matches) != 1
        or str(matches[0].get("role", "")) != "compact"
        or matches[0].get("calibration_eligible", True) is not True
        or str(selection.get("best_compact_vlm", "")) != selected_compact
        or not isinstance(calibration, Mapping)
        or not isinstance(calibration.get(selected_compact), Mapping)
        or len(summary_matches) != 1
        or selection.get("temporal_experiment1_model_ids") != expected_routes
        or type(selection.get("temporal_experiment1_model_count")) is not int
        or selection.get("temporal_experiment1_model_count") != 2
    ):
        raise StudyGateError(
            "Frozen selected compact model or temporal Experiment 1 identity is invalid"
        )
    selected_model = str(selection.get("selected_model", ""))
    if selected_model not in {"detector_only", selected_compact}:
        raise StudyGateError(
            "Frozen deployment selection is neither detector_only nor the selected compact model"
        )
    return selected_compact


def _require_exact_temporal_experiment1_routes(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    selected_compact_model: str,
) -> None:
    """Require exactly detector-only plus one compact temporal route."""

    expected = ("detector_only", str(selected_compact_model))
    for label, frame in (("detail", detail), ("summary", summary)):
        if (
            not isinstance(frame, pd.DataFrame)
            or frame.empty
            or "model_id" not in frame.columns
        ):
            raise StudyGateError(
                "Temporal Experiment 1 must contain exactly detector_only plus "
                f"one selected compact model; {label} evidence is missing"
            )
        observed = tuple(dict.fromkeys(frame["model_id"].astype(str)))
        duplicate_summary_route = bool(
            label == "summary"
            and (
                len(frame) != 2
                or frame["model_id"].astype(str).duplicated().any()
            )
        )
        if (
            observed != expected
            or len(set(observed)) != 2
            or duplicate_summary_route
        ):
            raise StudyGateError(
                "Temporal Experiment 1 must contain exactly detector_only plus "
                f"one selected compact model; {label} routes={list(observed)}"
            )


def _priority_profiles(cfg: Any) -> dict[str, tuple[str, ...]]:
    profiles = _section(cfg, "benchmark").get("mission_profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise StudyGateError("Registered VisDrone mission profiles are missing")
    output: dict[str, tuple[str, ...]] = {}
    for name, value in profiles.items():
        if not isinstance(value, Mapping) or not value.get("priority_categories"):
            raise StudyGateError(f"Mission profile {name!r} has no registered priority categories")
        output[str(name)] = tuple(map(str, value["priority_categories"]))
    return output


def _receiver_control_contexts(
    manifest: pd.DataFrame,
    cfg: Any,
) -> Mapping[str, Any]:
    return registered_receiver_context(
        manifest,
        registered_roi_xyxy=_section(cfg, "benchmark")["registered_roi_xyxy"],
    )


def _measured_runner(cfg: Any, device: str) -> Callable[..., Any]:
    runtime = _section(cfg, "runtime")
    device_index = int(device.split(":")[-1]) if device.startswith("cuda") else 0
    return partial(
        nvml_measurement_runner,
        device_index=device_index,
        warmup_iterations=int(runtime.get("warmup_iterations", 5)),
        steady_state_repetitions=int(runtime.get("steady_state_repetitions", 3)),
    )


def _radio_energy_arguments(cfg: Any) -> dict[str, Any]:
    evaluation = _section(cfg, "evaluation")
    hardware = _section(cfg, "hardware")
    included = bool(evaluation.get("radio_energy_included", True))
    scope = str(evaluation.get("energy_claim_scope", ""))
    tx = float(hardware.get("radio_tx_j_per_byte", 0.0))
    rx = float(hardware.get("radio_rx_j_per_byte", 0.0))
    registered_scope = (
        "combined_sender_and_ground_gpu_board_nvml_compute_only_excludes_radio_system"
    )
    if not included and scope != registered_scope:
        raise StudyGateError(
            "Radio-excluded execution must declare the registered combined sender/ground "
            "GPU-board NVML compute scope"
        )
    if not included and (tx != 0 or rx != 0):
        raise StudyGateError("GPU-board-only execution requires zero radio coefficients")
    return {
        "radio_tx_j_per_byte": tx,
        "radio_rx_j_per_byte": rx,
        "radio_energy_included": included,
    }


def _h4_high_confidence_threshold(cfg: Any) -> float:
    """Return the one frozen semantic-acceptance threshold used end to end."""

    evaluation = _section(cfg, "evaluation")
    if "h4_high_confidence_threshold" not in evaluation:
        raise StudyGateError(
            "evaluation.h4_high_confidence_threshold must be explicitly registered"
        )
    threshold = float(evaluation["h4_high_confidence_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise StudyGateError(
            "evaluation.h4_high_confidence_threshold must lie in [0, 1]"
        )
    return threshold


def _video_payload_arguments(cfg: Any) -> dict[str, Any]:
    """Resolve the exact frozen practical-video variants for cache and trials."""

    payloads = _section(cfg, "payloads")
    if "video_crf" not in payloads or "video_fps" not in payloads:
        raise StudyGateError("Video CRFs and FPS must be explicitly registered")
    crfs = tuple(int(value) for value in payloads["video_crf"])
    fps = int(payloads["video_fps"])
    if not crfs or len(set(crfs)) != len(crfs):
        raise StudyGateError("payloads.video_crf must be a nonempty unique sequence")
    if any(value < 0 or value > 51 for value in crfs) or fps <= 0:
        raise StudyGateError("Registered video CRF/FPS values are invalid")
    return {"video_crfs": list(crfs), "video_fps": fps}


def _image_payload_arguments(cfg: Any) -> dict[str, Any]:
    """Resolve the exact frozen image variants shared by cache and frontier."""

    payloads = _section(cfg, "payloads")
    required = {"image_codecs", "image_qualities", "image_scales"}
    if missing := required.difference(payloads):
        raise StudyGateError(
            f"Image payload design must be explicitly registered: {sorted(missing)}"
        )
    codecs = tuple(str(value) for value in payloads["image_codecs"])
    qualities = tuple(int(value) for value in payloads["image_qualities"])
    scales = tuple(float(value) for value in payloads["image_scales"])
    if (
        not codecs or len(set(codecs)) != len(codecs)
        or not qualities or len(set(qualities)) != len(qualities)
        or not scales or len(set(scales)) != len(scales)
    ):
        raise StudyGateError(
            "Registered image codecs, qualities, and scales must be nonempty and unique"
        )
    if any(value < 1 or value > 100 for value in qualities):
        raise StudyGateError("Registered image qualities must lie in [1, 100]")
    if any(not 0.0 < value <= 1.0 for value in scales):
        raise StudyGateError("Registered image scales must lie in (0, 1]")
    return {
        "image_codecs": list(codecs),
        "image_qualities": list(qualities),
        "image_scales": list(scales),
    }


def _resolve_native_cache(
    request: Any,
    *,
    ctx: RunContext,
    cfg: Any,
    manifest: pd.DataFrame,
    semantic_cache: pd.DataFrame,
    detector: UltralyticsDetectorTracker,
    detector_revision: str,
    semantic_source_model: str,
    device: str,
    phase: str,
) -> tuple[pd.DataFrame, Path]:
    image_design = _image_payload_arguments(cfg)
    video_design = _video_payload_arguments(cfg)
    design = NativeReceiverDesign(
        semantic_source_model=semantic_source_model,
        image_codecs=tuple(image_design["image_codecs"]),
        image_qualities=tuple(image_design["image_qualities"]),
        image_scales=tuple(image_design["image_scales"]),
        video_crfs=tuple(video_design["video_crfs"]),
        video_fps=video_design["video_fps"],
    )
    artifact = _results(ctx) / f"{phase}_{semantic_source_model}_native_receiver_cache.csv.gz"
    control_contexts = _receiver_control_contexts(manifest, cfg)
    receiver = RegisteredVisDroneNativeReceiver(
        detector,
        detector_revision=detector_revision,
        priority_profiles=_priority_profiles(cfg),
        registered_roi_xyxy=_section(cfg, "benchmark")["registered_roi_xyxy"],
        priority_confidence=float(_section(cfg, "models").get("proposal_policy", {}).get(
            "minimum_confidence", 0.05,
        )),
        image_plane_iou=float(_section(cfg, "benchmark")["min_image_plane_iou"]),
        temporary_root=ctx.run_root / "receiver_tmp",
    )
    if isinstance(request, pd.DataFrame):
        cache = request.copy()
    elif isinstance(request, (str, Path)):
        cache = pd.read_csv(request, compression="infer", float_precision="round_trip")
    elif isinstance(request, bool) and ctx.resume and artifact.is_file():
        cache = pd.read_csv(artifact, compression="infer", float_precision="round_trip")
    elif callable(request) and request is not True:
        cache = request(
            manifest=manifest, semantic_packet_cache=semantic_cache,
            semantic_source_model=semantic_source_model, design=design,
            control_contexts=control_contexts, device=device, output_path=artifact,
        )
    elif request is True:
        cache = produce_native_receiver_cache(
            manifest, semantic_cache, design=design,
            control_contexts=control_contexts, receiver=receiver,
            measurement_runner=_measured_runner(cfg, device),
            sender_hardware_id=f"gpu:{device}", receiver_hardware_id=f"gpu:{device}",
            sender_revision=RUNNER_VERSION, receiver_revision=detector_revision,
            receiver_fusion_receipt=receiver.fusion_receipt,
            temporary_root=ctx.run_root / "receiver_tmp", output_path=artifact,
            checkpoint_root=(ctx.run_root / "native_receiver_checkpoints" / phase) if ctx.resume else None,
            checkpoint_context_sha256=_native_checkpoint_context(
                ctx, cfg, phase=phase, source_model=semantic_source_model, device=device,
            ) if ctx.resume else None,
        )
    else:
        raise StudyGateError(
            "Scientific Experiment 2 requires build_native_receiver_cache=True, "
            "a verified cache path/frame, or an explicit real producer callable"
        )
    validate_native_receiver_cache(
        cache, manifest, semantic_cache, design=design,
        control_contexts=control_contexts,
        expected_fusion_receipt=receiver.fusion_receipt,
    )
    for column, expected in {
        "sender_revision": RUNNER_VERSION,
        "receiver_revision": detector_revision,
        "sender_hardware_id": f"gpu:{device}",
        "receiver_hardware_id": f"gpu:{device}",
    }.items():
        if not cache[column].astype(str).eq(str(expected)).all():
            raise StudyGateError(f"Native receiver cache {column} differs from the active runtime")
    cache = _bind_verified_cache_artifact(cache, artifact)
    return cache, artifact



def _bind_verified_cache_artifact(cache: pd.DataFrame, artifact: Path) -> pd.DataFrame:
    """Bind consumed rows to their canonical artifact; never return B with path A.

    CSV round-trip normalization accounts for pandas' serialization of dtypes
    and missing values. Numeric comparisons remain exact. Existing artifacts
    are immutable: different rows fail closed instead of silently overwriting
    evidence that a frozen selection might already reference.
    """

    expected = pd.read_csv(io.StringIO(cache.to_csv(index=False)), float_precision="round_trip")
    if artifact.is_file():
        persisted = pd.read_csv(artifact, compression="infer", float_precision="round_trip")
        try:
            pd.testing.assert_frame_equal(
                expected, persisted, check_dtype=False, check_exact=True,
            )
        except AssertionError as exc:
            raise StudyGateError(
                f"Verified cache rows differ from existing canonical artifact: {artifact}. "
                "Use the matching cache or a new immutable run."
            ) from exc
        return persisted
    atomic_write_frame(artifact, cache)
    return pd.read_csv(artifact, compression="infer", float_precision="round_trip")


def _native_checkpoint_context(ctx: RunContext, cfg: Any, *, phase: str,
                               source_model: str, device: str) -> str:
    runtime = _section(cfg, "runtime")
    # The notebook has already verified this known development/runtime seal.
    # Bind its exact bytes to checkpoint recovery as well as run and GPU identity.
    configured_paths = ctx.config.get("paths", {})
    output_root = configured_paths.get("output_root") if isinstance(configured_paths, Mapping) else None
    seal_path = (
        Path(str(output_root)).expanduser().resolve() / ctx.run_name
        / "frozen_inputs" / "INFERENCE_ENVIRONMENT_SEAL.json"
    ) if output_root else None
    seal_sha256 = runtime.get("inference_environment_seal_sha256")
    if seal_path is not None and seal_path.is_file() and not seal_path.is_symlink():
        seal_sha256 = hash_file(seal_path)
    elif runtime.get("inference_environment_seal_status"):
        raise StudyGateError("Native resume checkpoints require the verified inference environment seal")
    guard_names = sorted(runtime.get("required_environment_guards", {}))
    return hash_object({
        "config_hash": ctx.config_hash,
        "code_hash": ctx.code_hash,
        "run_root": str(ctx.run_root.resolve()),
        "run_name": ctx.run_name,
        "profile": ctx.profile,
        "phase": phase,
        "source_model": source_model,
        "device": device,
        "environment_guards": {key: os.environ.get(key, "") for key in guard_names},
        "inference_environment_id": runtime.get("inference_environment_id"),
        "inference_environment_seal_sha256": seal_sha256,
    })


def _native_receiver_fusion_binding(cache: pd.DataFrame) -> dict[str, str]:
    """Extract the single fusion receipt embedded in a verified native cache."""

    required = {
        "receiver_fusion_policy_json", "receiver_fusion_policy_version",
        "receiver_fusion_policy_sha256",
    }
    if missing := required.difference(cache.columns):
        raise StudyGateError(f"Native receiver cache lacks fusion binding: {sorted(missing)}")
    values = {
        column: set(cache[column].astype(str))
        for column in sorted(required)
    }
    if any(len(observed) != 1 for observed in values.values()):
        raise StudyGateError("Native receiver cache mixes fusion receipts")
    payload_json = next(iter(values["receiver_fusion_policy_json"]))
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as exc:
        raise StudyGateError("Native receiver fusion binding is not valid JSON") from exc
    return {
        "policy_version": next(iter(values["receiver_fusion_policy_version"])),
        "policy_sha256": next(iter(values["receiver_fusion_policy_sha256"])),
        "detector_revision": str(payload.get("detector_revision", "")),
    }


def _feature_design(cfg: Any, revision: str, device: str) -> FeatureCacheDesign:
    feature = _section(cfg, "models").get("feature_baseline")
    if not isinstance(feature, Mapping):
        raise StudyGateError("Registered L4 feature design is missing")
    quantization = tuple(int(str(item).lower().removeprefix("int")) for item in feature.get("quantization", (8, 4)))
    return FeatureCacheDesign(
        encoder_revision=str(feature.get("encoder_revision", "IMAGENET1K_V2")),
        split_layer=str(feature.get("split_layer", "layer3")),
        receiver_head_revision=revision,
        sender_hardware_id=f"gpu:{device}", receiver_hardware_id=f"gpu:{device}",
        quantization_bits=quantization,
    )


def _resolve_feature_cache(
    request: Any,
    *,
    ctx: RunContext,
    cfg: Any,
    phase_manifest: pd.DataFrame,
    device: str,
    phase: str,
    frozen_head_lock: str | Path,
) -> tuple[pd.DataFrame, Path, Path]:
    artifact = _results(ctx) / f"{phase}_registered_feature_cache.csv.gz"
    lock_path = Path(frozen_head_lock).expanduser().resolve()
    if not lock_path.is_file():
        raise StudyGateError("Registered feature-head lock is missing")
    feature_cfg = _section(cfg, "models").get("feature_baseline")
    if not isinstance(feature_cfg, Mapping):
        raise StudyGateError("Registered feature-baseline configuration is missing")
    configured_path = Path(str(feature_cfg.get("receiver_head", ""))).expanduser().resolve()
    configured_hash = str(feature_cfg.get("receiver_head_revision", ""))
    if configured_path != lock_path or hash_file(lock_path) != configured_hash:
        raise StudyGateError(
            "Feature-head path/hash are not bound to the loaded scientific configuration"
        )
    encoder = ResNet50Layer3Encoder(device, allow_download=False)
    head = load_frozen_feature_head(lock_path, encoder=encoder)
    design = _feature_design(cfg, head.revision, device)
    control_contexts = _receiver_control_contexts(phase_manifest, cfg)
    if isinstance(request, pd.DataFrame):
        cache = request.copy()
    elif isinstance(request, (str, Path)):
        cache = pd.read_csv(request, compression="infer", float_precision="round_trip")
    elif isinstance(request, bool) and ctx.resume and artifact.is_file():
        cache = pd.read_csv(artifact, compression="infer", float_precision="round_trip")
    elif callable(request) and request is not True:
        cache = request(
            manifest=phase_manifest, design=design,
            control_contexts=control_contexts, encoder=encoder,
            receiver_head=head, device=device, output_path=artifact,
        )
    elif request is True:
        cache = produce_feature_cache(
            phase_manifest, design=design, control_contexts=control_contexts,
            encoder=encoder, receiver_head=head,
            measurement_runner=_measured_runner(cfg, device),
            output_path=artifact,
        )
    else:
        raise StudyGateError(
            "Scientific Experiment 2 requires build_feature_cache=True, a verified "
            "cache path/frame, or an explicit real producer callable"
        )
    validate_registered_feature_cache(
        cache, phase_manifest, design=design,
        control_contexts=control_contexts,
    )
    cache = _bind_verified_cache_artifact(cache, artifact)
    return cache, artifact, lock_path


def _policy_manifest(
    manifest: pd.DataFrame,
    detector_cache: pd.DataFrame,
    selected_rows: pd.DataFrame,
    detector_rows: pd.DataFrame,
    selection: Mapping[str, Any],
) -> pd.DataFrame:
    selected_model = str(selection["selected_model"])
    selected_calibration = selection.get("calibration", {}).get(selected_model)
    detector_calibration = selection.get("calibration", {}).get("detector_only")
    if not isinstance(selected_calibration, Mapping) or not isinstance(detector_calibration, Mapping):
        raise StudyGateError("Frozen selected/detector calibration is missing")
    selected_semantic = calibrated_semantic_packet_cache(
        selected_rows, float(selected_calibration["temperature"]),
        source_model=selected_model, authoritative_manifest=manifest,
    )
    detector_semantic = calibrated_semantic_packet_cache(
        detector_rows, float(detector_calibration["temperature"]),
        source_model="detector_only", authoritative_manifest=manifest,
    )
    output = manifest.drop(columns=[column for column in ("risk", "uncertainty") if column in manifest]).copy()
    output = output.merge(
        selected_semantic[["sample_id", "risk", "uncertainty"]],
        on="sample_id", validate="one_to_one",
    ).merge(
        detector_semantic[["sample_id", "risk", "uncertainty"]].rename(columns={
            "risk": "detector_risk", "uncertainty": "detector_uncertainty",
        }),
        on="sample_id", validate="one_to_one",
    )
    detector_cost = detector_cache[["sample_id", "latency_ms", "energy_j"]].rename(columns={
        "latency_ms": "detector_latency_ms", "energy_j": "detector_energy_j",
    })
    output = output.merge(detector_cost, on="sample_id", validate="one_to_one")
    selected_cost = selected_rows[["observation_id", "latency_ms", "energy_j"]].rename(columns={
        "observation_id": "sample_id", "latency_ms": "context_model_latency_ms",
        "energy_j": "context_model_energy_j",
    })
    if selected_model == "detector_only":
        selected_cost["context_model_latency_ms"] = 0.0
        selected_cost["context_model_energy_j"] = 0.0
    output = output.merge(selected_cost, on="sample_id", validate="one_to_one")
    output["context_triggered"] = (
        output["detector_risk"].ge(0.45)
        | output["detector_uncertainty"].ge(0.35)
    )
    for column in (
        "detector_latency_ms", "detector_energy_j", "context_model_latency_ms",
        "context_model_energy_j", "battery_fraction", "thermal_c",
    ):
        values = pd.to_numeric(output[column], errors="coerce")
        if values.isna().any() or (~np.isfinite(values)).any() or (values < 0).any():
            raise StudyGateError(f"Policy manifest has invalid measured/assigned {column}")
    return output


def run_validation_perception_and_frontier(
    ctx: RunContext,
    cfg: Any,
    manifest: pd.DataFrame,
    model_lock: str | Path,
    device: str,
    build_native_receiver_cache: Any,
    build_feature_cache: Any,
    feature_head_lock: str | Path,
    temporal_manifest: pd.DataFrame | None = None,
    on_experiment_complete: ExperimentCallback | None = None,
) -> dict[str, Any]:
    """Execute real validation perception, selection, and Experiment 2.

    This path has no fixture backend.  Every VLM is loaded from its verified
    local snapshot, and absence of native receiver/L4 producers blocks the run.
    """

    if not ctx.is_scientific:
        raise StudyGateError("VisDrone paper validation requires the full scientific profile")
    lock = _bind_runtime_model_lock(model_lock, cfg)
    if manifest["split"].astype(str).eq("test").any():
        raise StudyGateError("Validation runner may not receive untouched test rows")
    if os.environ.get("SEMUAV_ENABLE_V3R1_VALIDATION_IMPORT") == "1":
        # v3r2: exact v3r1 native-receiver/feature caches; validated below
        # against this run's manifest, semantic cache and design.
        from .v3r1_import import import_result_caches

        import_result_caches(ctx)
    sample = _require_registered_phase_sample(
        manifest, cfg, expected_splits=("calibration", "validation"),
        require_validation_coverage=True,
    )
    temporal_sample: pd.DataFrame | None = None
    temporal_gate: pd.DataFrame | None = None
    if temporal_manifest is not None:
        temporal_sample, temporal_gate = _require_registered_temporal_manifest(
            temporal_manifest, cfg, split="validation",
        )
        general_ids = set(sample["sample_id"].astype(str))
        temporal_ids = set(temporal_sample["sample_id"].astype(str))
        overlap = general_ids & temporal_ids
        if overlap:
            identity_columns = [
                "sample_id", "source_sample_id", "image_path", "split", "component",
                "sequence_id", "frame_index", "mission_profile", "gold_action",
            ]
            general_identity = sample.loc[
                sample["sample_id"].astype(str).isin(overlap), identity_columns,
            ].sort_values("sample_id", kind="mergesort").reset_index(drop=True)
            temporal_identity = temporal_sample.loc[
                temporal_sample["sample_id"].astype(str).isin(overlap), identity_columns,
            ].sort_values("sample_id", kind="mergesort").reset_index(drop=True)
            if not general_identity.equals(temporal_identity):
                raise StudyGateError(
                    "General and temporal validation manifests conflict on shared samples"
                )
        detector_manifest = pd.concat(
            [sample, temporal_sample.loc[~temporal_sample["sample_id"].astype(str).isin(overlap)]],
            ignore_index=True,
        )
    else:
        detector_manifest = sample
    sample_gate = pd.DataFrame([{
        "splits": "calibration,validation",
        "rows": int(len(sample)),
        "independent_visual_groups": int(sample["visual_group_id"].nunique()),
        "profile_pairs_complete": True,
        "analysis_sampling_sha256": hash_object(
            _section(cfg, "benchmark")["analysis_sampling"]
        ),
        "passed": True,
    }])
    _write_frame(ctx, "validation_perception_sample_gate.csv.gz", sample_gate)
    measurement_warmup = _development_warmup_spec(sample)
    detector, detector_cache, detector_revision, detector_verification = _detector_and_cache(
        ctx, cfg, detector_manifest, device, stage_prefix="validation",
        warmup_spec=measurement_warmup,
    )
    warmup_match = detector_cache[
        detector_cache["sample_id"].astype(str).eq(str(measurement_warmup["sample_id"]))
    ]
    if warmup_match.empty:
        raise StudyGateError("Measured detector cache lacks the frozen VLM warmup row")
    measurement_warmup = {
        **measurement_warmup,
        "detector_proposals_json": str(warmup_match.iloc[0]["proposals_json"]),
        "detector_revision": detector_revision,
    }
    benchmark = _section(cfg, "benchmark")
    detector_measurement_columns = [
        column for column in detector_cache.columns if column.startswith("measurement_")
    ]

    def make_detector_backend() -> DetectorOnlyBackend:
        backend = DetectorOnlyBackend(
            detector_revision,
            urban_policy="registered_roi_occupancy",
            urban_priority_labels=sorted({
                label for labels in _priority_profiles(cfg).values() for label in labels
            }),
            urban_priority_profiles=_priority_profiles(cfg),
            urban_priority_roi_xyxy=benchmark["registered_roi_xyxy"],
            urban_priority_confidence=float(
                _section(cfg, "models").get("proposal_policy", {}).get(
                    "minimum_confidence", 0.05,
                )
            ),
            urban_image_plane_iou=float(benchmark["min_image_plane_iou"]),
        )
        backend.measurement_metadata = {
            column: detector_cache.iloc[0][column]
            for column in detector_measurement_columns
        }
        return backend

    detector_backend = make_detector_backend()
    prompt = prompt_for_mission("visdrone_urban")
    model_cfg = _section(cfg, "models")
    image_size = model_cfg.get("input_size", (768, 768))
    runtime = _section(cfg, "runtime")
    vlm_warmup_request = _warmup_request(measurement_warmup)
    details: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []

    def execute_backend(
        backend: Any,
        *,
        phase_sample: pd.DataFrame = sample,
        target_details: list[pd.DataFrame] = details,
        target_summaries: list[pd.DataFrame] = summaries,
        cache_role: str = "general",
    ) -> None:
        measurement_metadata = _normalize_measurement_metadata(
            getattr(backend, "measurement_metadata", {})
        )
        detail, summary = experiment1_perception(
            phase_sample, [backend], scientific=True, detector_cache=detector_cache,
            ctx=ctx, shard_size=32,
            max_new_tokens=int(model_cfg.get(
                "max_new_tokens", VISDRONE_REGISTERED_MAX_NEW_TOKENS,
            )),
            prompt=prompt,
            inference_cache_context={
                "detector_revision": detector_revision,
                "detector_checkpoint_sha256": detector_verification["sha256"],
                "model_lock_sha256": hash_file(model_lock),
                "measurement": measurement_metadata,
                "sample_role": str(cache_role),
            },
        )
        encoded_metadata = json.dumps(
            measurement_metadata, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        )
        detail["measurement_metadata_json"] = encoded_metadata
        summary["measurement_metadata_json"] = encoded_metadata
        detail["registered_sample_role"] = str(cache_role)
        summary["registered_sample_role"] = str(cache_role)
        target_details.append(detail)
        target_summaries.append(summary)

    execute_backend(detector_backend)
    del detector_backend
    for entry in lock["models"]:
        raw_backend = _model_backend(entry, device, image_size)
        backend = RepeatedMeasurementPerceptionBackend(
            raw_backend,
            warmup_request=vlm_warmup_request,
            warmup_image_sha256=str(measurement_warmup["image_sha256"]),
            warmup_split=str(measurement_warmup["split"]),
            warmup_prompt=prompt,
            warmup_max_new_tokens=int(model_cfg["max_new_tokens"]),
            warmup_iterations=int(runtime["warmup_iterations"]),
            repetitions=int(runtime["steady_state_repetitions"]),
        )
        execute_backend(backend)
        del backend, raw_backend
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
    exp1_detail = pd.concat(details, ignore_index=True)
    exp1_summary = pd.concat(summaries, ignore_index=True)
    selection = _fit_validation_selection(exp1_detail, cfg, lock["models"])
    selected_model = str(selection["selected_model"])
    selected_compact_model = _require_selected_compact_model(
        selection, lock["models"],
    )
    temporal_exp1_detail = pd.DataFrame()
    temporal_exp1_summary = pd.DataFrame()
    temporal_policy_manifest: pd.DataFrame | None = None
    if temporal_sample is not None:
        temporal_details: list[pd.DataFrame] = []
        temporal_summaries: list[pd.DataFrame] = []
        temporal_detector_backend = make_detector_backend()
        execute_backend(
            temporal_detector_backend,
            phase_sample=temporal_sample,
            target_details=temporal_details,
            target_summaries=temporal_summaries,
            cache_role="temporal_controller",
        )
        del temporal_detector_backend
        selected_entries = [
            entry for entry in lock["models"]
            if str(entry["model_id"]) == selected_compact_model
        ]
        if len(selected_entries) != 1:
            raise StudyGateError(
                "Frozen selected compact model is not unique in MODEL_LOCK"
            )
        temporal_raw_backend = _model_backend(selected_entries[0], device, image_size)
        temporal_backend = RepeatedMeasurementPerceptionBackend(
            temporal_raw_backend,
            warmup_request=vlm_warmup_request,
            warmup_image_sha256=str(measurement_warmup["image_sha256"]),
            warmup_split=str(measurement_warmup["split"]),
            warmup_prompt=prompt,
            warmup_max_new_tokens=int(model_cfg["max_new_tokens"]),
            warmup_iterations=int(runtime["warmup_iterations"]),
            repetitions=int(runtime["steady_state_repetitions"]),
        )
        execute_backend(
            temporal_backend,
            phase_sample=temporal_sample,
            target_details=temporal_details,
            target_summaries=temporal_summaries,
            cache_role="temporal_controller",
        )
        del temporal_backend, temporal_raw_backend
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
        temporal_exp1_detail = pd.concat(temporal_details, ignore_index=True)
        temporal_exp1_summary = pd.concat(temporal_summaries, ignore_index=True)
        _require_exact_temporal_experiment1_routes(
            temporal_exp1_detail,
            temporal_exp1_summary,
            selected_compact_model=selected_compact_model,
        )
        temporal_compact_rows = temporal_exp1_detail[
            temporal_exp1_detail["model_id"].astype(str).eq(
                selected_compact_model
            )
        ].copy()
        temporal_detector_rows = temporal_exp1_detail[
            temporal_exp1_detail["model_id"].astype(str).eq("detector_only")
        ].copy()
        temporal_selected_rows = (
            temporal_detector_rows.copy()
            if selected_model == "detector_only"
            else temporal_compact_rows.copy()
        )
        temporal_policy_manifest = _policy_manifest(
            temporal_sample, detector_cache, temporal_selected_rows,
            temporal_detector_rows, selection,
        )
    # Persist completed perception before native encoding can fail.  These are
    # the same canonical artifacts later bound into the validation selection.
    _write_frame(ctx, "validation_exp1_perception_detail.csv.gz", exp1_detail)
    _write_frame(ctx, "validation_exp1_perception_summary.csv.gz", exp1_summary)
    if temporal_sample is not None:
        _write_frame(ctx, "validation_temporal_exp1_perception_detail.csv.gz", temporal_exp1_detail)
        _write_frame(ctx, "validation_temporal_exp1_perception_summary.csv.gz", temporal_exp1_summary)
    _emit_experiment_complete(on_experiment_complete, 1, "validation", {
        "exp1_summary": exp1_summary,
        "temporal_exp1_summary": temporal_exp1_summary,
    })
    selected_rows = exp1_detail[exp1_detail["model_id"].astype(str).eq(selected_model)].copy()
    detector_rows = exp1_detail[exp1_detail["model_id"].astype(str).eq("detector_only")].copy()
    policy_manifest = _policy_manifest(
        sample, detector_cache, selected_rows, detector_rows, selection,
    )
    selected_semantic = calibrated_semantic_packet_cache(
        selected_rows, float(selection["calibration"][selected_model]["temperature"]),
        source_model=selected_model, authoritative_manifest=sample,
    )
    native_cache, native_path = _resolve_native_cache(
        build_native_receiver_cache, ctx=ctx, cfg=cfg, manifest=sample,
        semantic_cache=selected_semantic, detector=detector,
        detector_revision=detector_revision, semantic_source_model=selected_model,
        device=device, phase="validation",
    )
    feature_cache, feature_path, feature_lock = _resolve_feature_cache(
        build_feature_cache, ctx=ctx, cfg=cfg, phase_manifest=sample,
        device=device, phase="validation", frozen_head_lock=feature_head_lock,
    )
    exp2_frontier = experiment2_representation_frontier_sharded(
        sample, ctx, scientific=True, mtu_bytes=int(_section(cfg, "payloads").get("mtu_bytes", 1200)),
        semantic_packet_cache=selected_semantic, native_receiver_cache=native_cache,
        registered_feature_cache=feature_cache, semantic_source_model=selected_model,
        **_image_payload_arguments(cfg),
        **_video_payload_arguments(cfg),
        stage_prefix="validation_02_representations",
    )
    # A detector-only semantic comparator is produced from the same transmitted
    # contract. Visual and L4 rows remain the selected receiver's shared rows.
    detector_semantic = calibrated_semantic_packet_cache(
        detector_rows, float(selection["calibration"]["detector_only"]["temperature"]),
        source_model="detector_only", authoritative_manifest=sample,
    )
    detector_native, detector_native_path = _resolve_native_cache(
        build_native_receiver_cache, ctx=ctx, cfg=cfg, manifest=sample,
        semantic_cache=detector_semantic, detector=detector,
        detector_revision=detector_revision, semantic_source_model="detector_only",
        device=device, phase="validation",
    )
    detector_frontier = experiment2_representation_frontier_sharded(
        sample, ctx, scientific=True,
        mtu_bytes=int(_section(cfg, "payloads").get("mtu_bytes", 1200)),
        semantic_packet_cache=detector_semantic, native_receiver_cache=detector_native,
        registered_feature_cache=feature_cache, semantic_source_model="detector_only",
        **_image_payload_arguments(cfg),
        **_video_payload_arguments(cfg),
        stage_prefix="validation_02_detector_representations",
    )
    detector_semantic_candidates = detector_frontier[
        detector_frontier["mode"].isin(["L0_HEARTBEAT", "L1_ALERT", "L2_SEMANTICS", "L3_SEMANTICS_ROI"])
    ].copy()
    detector_semantic_candidates["variant"] = (
        detector_semantic_candidates["variant"].astype(str) + "__detector_only"
    )
    exp2_frontier = pd.concat([exp2_frontier, detector_semantic_candidates], ignore_index=True)
    perception_costs = policy_manifest[[
        "sample_id", "detector_latency_ms", "detector_energy_j",
        "context_model_latency_ms", "context_model_energy_j",
    ]]
    exp2_frontier = add_experiment2_total_costs(
        exp2_frontier, perception_costs, require_measured_costs=True,
    )
    temporal_exp2_frontier = pd.DataFrame()
    temporal_native_path: Path | None = None
    temporal_detector_native_path: Path | None = None
    temporal_feature_path: Path | None = None
    if temporal_sample is not None and temporal_policy_manifest is not None:
        temporal_selected_semantic = calibrated_semantic_packet_cache(
            temporal_selected_rows,
            float(selection["calibration"][selected_model]["temperature"]),
            source_model=selected_model,
            authoritative_manifest=temporal_sample,
        )
        temporal_native, temporal_native_path = _resolve_native_cache(
            build_native_receiver_cache, ctx=ctx, cfg=cfg,
            manifest=temporal_sample, semantic_cache=temporal_selected_semantic,
            detector=detector, detector_revision=detector_revision,
            semantic_source_model=selected_model, device=device,
            phase="validation_temporal",
        )
        temporal_feature, temporal_feature_path, temporal_feature_lock = _resolve_feature_cache(
            build_feature_cache, ctx=ctx, cfg=cfg, phase_manifest=temporal_sample,
            device=device, phase="validation_temporal",
            frozen_head_lock=feature_head_lock,
        )
        if temporal_feature_lock != feature_lock:
            raise StudyGateError("Temporal and general validation use different L4 heads")
        temporal_exp2_frontier = experiment2_representation_frontier_sharded(
            temporal_sample, ctx, scientific=True,
            mtu_bytes=int(_section(cfg, "payloads").get("mtu_bytes", 1200)),
            semantic_packet_cache=temporal_selected_semantic,
            native_receiver_cache=temporal_native,
            registered_feature_cache=temporal_feature,
            semantic_source_model=selected_model,
            **_image_payload_arguments(cfg),
            **_video_payload_arguments(cfg),
            stage_prefix="validation_temporal_02_representations",
        )
        temporal_detector_semantic = calibrated_semantic_packet_cache(
            temporal_detector_rows,
            float(selection["calibration"]["detector_only"]["temperature"]),
            source_model="detector_only", authoritative_manifest=temporal_sample,
        )
        temporal_detector_native, temporal_detector_native_path = _resolve_native_cache(
            build_native_receiver_cache, ctx=ctx, cfg=cfg,
            manifest=temporal_sample, semantic_cache=temporal_detector_semantic,
            detector=detector, detector_revision=detector_revision,
            semantic_source_model="detector_only", device=device,
            phase="validation_temporal",
        )
        temporal_detector_frontier = experiment2_representation_frontier_sharded(
            temporal_sample, ctx, scientific=True,
            mtu_bytes=int(_section(cfg, "payloads").get("mtu_bytes", 1200)),
            semantic_packet_cache=temporal_detector_semantic,
            native_receiver_cache=temporal_detector_native,
            registered_feature_cache=temporal_feature,
            semantic_source_model="detector_only",
            **_image_payload_arguments(cfg),
            **_video_payload_arguments(cfg),
            stage_prefix="validation_temporal_02_detector_representations",
        )
        temporal_detector_candidates = temporal_detector_frontier[
            temporal_detector_frontier["mode"].isin([
                "L0_HEARTBEAT", "L1_ALERT", "L2_SEMANTICS", "L3_SEMANTICS_ROI",
            ])
        ].copy()
        temporal_detector_candidates["variant"] = (
            temporal_detector_candidates["variant"].astype(str) + "__detector_only"
        )
        temporal_exp2_frontier = pd.concat(
            [temporal_exp2_frontier, temporal_detector_candidates], ignore_index=True,
        )
        temporal_exp2_frontier = add_experiment2_total_costs(
            temporal_exp2_frontier,
            temporal_policy_manifest[[
                "sample_id", "detector_latency_ms", "detector_energy_j",
                "context_model_latency_ms", "context_model_energy_j",
            ]],
            require_measured_costs=True,
        )
    _write_frame(ctx, "validation_exp2_frontier.csv.gz", exp2_frontier)
    if temporal_sample is not None:
        _write_frame(ctx, "validation_temporal_exp2_frontier.csv.gz", temporal_exp2_frontier)
    _emit_experiment_complete(on_experiment_complete, 2, "validation", {
        "exp2_frontier": exp2_frontier,
    })
    selection.update({
        "model_lock_path": str(Path(model_lock).resolve()),
        "model_lock_sha256": hash_file(model_lock),
        "detector_revision": detector_revision,
        "detector_checkpoint": str(detector_verification["checkpoint"]),
        "detector_checkpoint_sha256": str(detector_verification["sha256"]),
        "detector_training_lineage": _detector_lineage_binding(
            detector_verification
        ),
        "detector_residency_receipt": str(detector_verification["residency_receipt_path"]),
        "detector_residency_receipt_sha256": str(detector_verification["residency_receipt_sha256"]),
        "feature_head_lock": str(feature_lock),
        "feature_head_lock_sha256": hash_file(feature_lock),
        "native_receiver_cache": str(native_path),
        "detector_native_receiver_cache": str(detector_native_path),
        "native_receiver_fusion": _native_receiver_fusion_binding(native_cache),
        "detector_native_receiver_fusion": _native_receiver_fusion_binding(
            detector_native
        ),
        "registered_feature_cache": str(feature_path),
        "analysis_sampling": dict(_section(cfg, "benchmark")["analysis_sampling"]),
        "analysis_sampling_sha256": hash_object(
            _section(cfg, "benchmark")["analysis_sampling"]
        ),
        "measurement_warmup": measurement_warmup,
        "detector_inference_and_proposal_policy": dict(
            detector_verification["residency"]["inference_and_proposal_policy"]
        ),
    })
    feature_receipt = json.loads(Path(feature_lock).read_text(encoding="utf-8"))
    selection["feature_head"] = {
        "lock_path": str(feature_lock),
        "lock_sha256": hash_file(feature_lock),
        "weights_path": str(feature_receipt["weights_path"]),
        "weights_sha256": str(feature_receipt["weights_sha256"]),
        "encoder_checkpoint_path": str(feature_receipt["encoder_checkpoint_path"]),
        "encoder_checkpoint_sha256": str(feature_receipt["encoder_checkpoint_sha256"]),
        "encoder_state_sha256": str(feature_receipt["encoder_state_sha256"]),
        "revision": str(feature_receipt["revision"]),
    }
    paths = {
        "exp1_detail": _write_frame(ctx, "validation_exp1_perception_detail.csv.gz", exp1_detail),
        "exp1_summary": _write_frame(ctx, "validation_exp1_perception_summary.csv.gz", exp1_summary),
        "exp2_frontier": _write_frame(ctx, "validation_exp2_frontier.csv.gz", exp2_frontier),
        "policy_manifest": _write_frame(ctx, "validation_policy_manifest.csv.gz", policy_manifest),
    }
    if temporal_sample is not None and temporal_policy_manifest is not None:
        assert temporal_gate is not None
        paths.update({
            "temporal_sample_gate": _write_frame(
                ctx, "validation_temporal_sample_gate.csv.gz", temporal_gate,
            ),
            "temporal_exp1_detail": _write_frame(
                ctx, "validation_temporal_exp1_perception_detail.csv.gz",
                temporal_exp1_detail,
            ),
            "temporal_exp1_summary": _write_frame(
                ctx, "validation_temporal_exp1_perception_summary.csv.gz",
                temporal_exp1_summary,
            ),
            "temporal_exp2_frontier": _write_frame(
                ctx, "validation_temporal_exp2_frontier.csv.gz",
                temporal_exp2_frontier,
            ),
            "temporal_policy_manifest": _write_frame(
                ctx, "validation_temporal_policy_manifest.csv.gz",
                temporal_policy_manifest,
            ),
        })
        selection.update({
            "temporal_sample_schema": TEMPORAL_SAMPLE_SCHEMA,
            "temporal_selection_sha256": str(
                temporal_sample["temporal_selection_sha256"].iloc[0]
            ),
            "temporal_sequence_count": int(
                temporal_sample["sequence_id"].astype(str).nunique()
            ),
            "temporal_observations_per_sequence": int(
                temporal_sample.groupby(
                    ["sequence_id", "mission_profile"], sort=False,
                )["source_sample_id"].nunique().min()
            ),
            "temporal_target_rate_hz": 5,
            "temporal_native_receiver_cache": str(temporal_native_path),
            "temporal_detector_native_receiver_cache": str(
                temporal_detector_native_path
            ),
            "temporal_native_receiver_fusion": _native_receiver_fusion_binding(
                temporal_native
            ),
            "temporal_detector_native_receiver_fusion": (
                _native_receiver_fusion_binding(temporal_detector_native)
            ),
            "temporal_registered_feature_cache": str(temporal_feature_path),
        })
    selection["artifacts"] = {name: str(path) for name, path in paths.items()}
    selection_path = atomic_write_json(_results(ctx) / "validation_selection.json", selection)
    selection["selection_path"] = str(selection_path)
    return {
        "exp1_detail": exp1_detail,
        "exp1_summary": exp1_summary,
        "exp2_frontier": exp2_frontier,
        "temporal_exp1_detail": temporal_exp1_detail,
        "temporal_exp1_summary": temporal_exp1_summary,
        "temporal_exp2_frontier": temporal_exp2_frontier,
        "temporal_policy_manifest": temporal_policy_manifest,
        "selection": selection,
    }


def _load_traces_for_split(
    traces: Any,
    *,
    ctx: RunContext,
    split: str,
) -> list[LinkTrace]:
    if isinstance(traces, Sequence) and not isinstance(traces, (str, bytes, pd.DataFrame)):
        if traces and all(isinstance(item, LinkTrace) for item in traces):
            return list(traces)
    if isinstance(traces, pd.DataFrame):
        path = _results(ctx) / "registered_trace_manifest.csv"
        traces.to_csv(path, index=False)
    else:
        path = Path(traces).expanduser().resolve()
    loaded, _ = load_trace_manifest(
        str(path), required_split=split, end_behavior=EndBehavior.TERMINATE,
    )
    return loaded


def _require_registered_temporal_manifest(
    manifest: pd.DataFrame,
    cfg: Any,
    *,
    split: str,
    test_permit: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verify a separately persisted MOT-only 5 Hz controller sample.

    The bounded DET+MOT analysis sample is appropriate for Experiments 1--3,
    but it is not a temporal controller estimand.  H2--H4 therefore consume a
    distinct outcome-blind MOT sample with one contiguous 64-observation replay
    per registered sequence.  This gate validates the persisted selection; it
    never selects, ranks, or repairs rows in the paper runner.
    """

    if not isinstance(manifest, pd.DataFrame) or manifest.empty:
        raise StudyGateError("Registered temporal controller manifest is missing or empty")
    spec = registered_temporal_sampling_spec(_section(cfg, "benchmark"), str(split))
    required = {
        "sample_id", "source_sample_id", "split", "component", "split_unit",
        "visual_group_id", "sequence_id", "frame_index", "mission_profile",
        "capture_time_us", "trace_time_us", "deadline_us",
        "temporal_observation_index", "temporal_episode_id",
        "temporal_selection_sha256", "temporal_sampling_method",
        "registered_source_frame_stride", "registered_target_rate_hz",
        "gold_action", "image_path",
    }
    if missing := required.difference(manifest.columns):
        raise StudyGateError(
            f"Registered temporal controller schema is incomplete: {sorted(missing)}"
        )
    sample = manifest.copy()
    if sample["sample_id"].astype(str).duplicated().any():
        raise StudyGateError("Registered temporal controller sample IDs are duplicated")
    if set(sample["split"].astype(str)) != {str(split)}:
        raise StudyGateError("Registered temporal controller sample has the wrong split")
    if set(sample["component"].astype(str)) != {str(spec["component"])}:
        raise StudyGateError("Registered temporal controller sample is not MOT-only")
    if set(sample["temporal_sampling_method"].astype(str)) != {str(spec["method"])}:
        raise StudyGateError("Registered temporal controller sampling method changed")
    selection_hashes = set(sample["temporal_selection_sha256"].astype(str))
    if len(selection_hashes) != 1 or not all(
        len(value) == 64 and all(character in "0123456789abcdef" for character in value)
        for value in selection_hashes
    ):
        raise StudyGateError("Registered temporal controller selection hash is invalid")
    if str(split) == "test":
        if test_permit is None or "test_ledger_sha256" not in sample:
            raise StudyGateError("Temporal test sample is not bound to the active OPEN ledger")
        if set(sample["test_ledger_sha256"].astype(str)) != {
            str(test_permit.ledger_sha256)
        }:
            raise StudyGateError("Temporal test rows are bound to another test ledger")

    expected_profiles = set(map(str, _priority_profiles(cfg)))
    observed_profiles = set(sample["mission_profile"].astype(str))
    profile_sets = sample.groupby("source_sample_id", sort=True)["mission_profile"].agg(
        lambda values: set(map(str, values))
    )
    if observed_profiles != expected_profiles or not profile_sets.map(
        lambda values: values == expected_profiles
    ).all():
        raise StudyGateError("Temporal source observations lack exact mission-profile copies")
    if sample.duplicated(["source_sample_id", "mission_profile"], keep=False).any():
        raise StudyGateError("Temporal source/profile rows are duplicated")

    sequences_required = int(spec["sequences_per_split"])
    observations_required = int(spec["observations_per_sequence"])
    stride_required = int(spec["source_frame_stride"])
    period_required = int(spec["capture_period_us"])
    observed_sequences = int(sample["sequence_id"].astype(str).nunique())
    if observed_sequences != sequences_required:
        raise StudyGateError(
            "Temporal controller sample has the wrong sequence count: "
            f"{observed_sequences}/{sequences_required}"
        )
    sequence_units = sample.groupby("sequence_id", sort=True)["split_unit"].nunique(
        dropna=False
    )
    if (sequence_units != 1).any():
        raise StudyGateError("A temporal sequence crosses independent split units")
    expected_index = np.arange(observations_required, dtype=np.int64)
    for (sequence_id, profile), group in sample.groupby(
        ["sequence_id", "mission_profile"], sort=True,
    ):
        ordered = group.sort_values("temporal_observation_index", kind="mergesort")
        if len(ordered) != observations_required:
            raise StudyGateError(
                f"Temporal episode {sequence_id!r}/{profile!r} does not contain "
                f"exactly {observations_required} observations"
            )
        indices = pd.to_numeric(
            ordered["temporal_observation_index"], errors="raise",
        ).to_numpy(np.int64)
        frames = pd.to_numeric(ordered["frame_index"], errors="raise").to_numpy(np.int64)
        captures = pd.to_numeric(
            ordered["capture_time_us"], errors="raise",
        ).to_numpy(np.int64)
        trace_times = pd.to_numeric(
            ordered["trace_time_us"], errors="raise",
        ).to_numpy(np.int64)
        if not np.array_equal(indices, expected_index):
            raise StudyGateError("Temporal observation indices are not contiguous from zero")
        if not np.array_equal(np.diff(frames), np.full(observations_required - 1, stride_required)):
            raise StudyGateError("Temporal publisher frames do not follow the registered stride")
        expected_times = expected_index * period_required
        if not np.array_equal(captures, expected_times) or not np.array_equal(
            trace_times, expected_times
        ):
            raise StudyGateError("Temporal replay is not the registered 5 Hz schedule")
        if ordered["temporal_episode_id"].astype(str).nunique() != 1:
            raise StudyGateError("Temporal episode identity changes within an episode")
        deadline = pd.to_numeric(ordered["deadline_us"], errors="raise").to_numpy(np.int64)
        if np.any(deadline <= captures):
            raise StudyGateError("Temporal replay contains a nonpositive deadline budget")

    expected_rows = sequences_required * observations_required * len(expected_profiles)
    gate = pd.DataFrame([
        {
            "gate": "registered_temporal_mot_sequences",
            "observed": observed_sequences,
            "required": sequences_required,
            "passed": observed_sequences == sequences_required,
            "required_for": "temporal_execution",
        },
        {
            "gate": "registered_temporal_observations_per_sequence",
            "observed": observations_required,
            "required": observations_required,
            "passed": True,
            "required_for": "temporal_execution",
        },
        {
            "gate": "registered_temporal_profile_rows",
            "observed": int(len(sample)),
            "required": expected_rows,
            "passed": int(len(sample)) == expected_rows,
            "required_for": "temporal_execution",
        },
        {
            "gate": "registered_temporal_target_rate_hz",
            "observed": int(spec["target_rate_hz"]),
            "required": 5,
            "passed": int(spec["target_rate_hz"]) == 5,
            "required_for": "temporal_execution",
        },
    ])
    if not gate["passed"].astype(bool).all():
        raise StudyGateError("Registered temporal controller sample gate failed")
    return sample.reset_index(drop=True), gate


def _rebase_registered_episodes(
    manifest: pd.DataFrame,
    traces: Sequence[LinkTrace],
    *,
    state_expiry_grace_us: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebase each sequence/profile replay to its own frozen trace episode.

    Coverage includes the registered post-deadline receiver-state expiry grace,
    not merely the last action deadline.  This prevents a trace that ends while
    the receiver can still legally retain state from being treated as fully
    measured.
    """

    required = {
        "sample_id", "sequence_id", "mission_profile", "capture_time_us",
        "trace_time_us", "deadline_us", "frame_index",
    }
    if missing := required.difference(manifest.columns):
        raise StudyGateError(f"Episode timing fields are missing: {sorted(missing)}")
    if not traces:
        raise StudyGateError("Episode timing requires registered traces")
    if isinstance(state_expiry_grace_us, bool) or int(state_expiry_grace_us) < 0:
        raise StudyGateError("Registered temporal state-expiry grace is invalid")
    state_expiry_grace_us = int(state_expiry_grace_us)
    trace_start = max(int(trace.segments[0].start_us) for trace in traces)
    trace_end = min(int(trace.segments[-1].end_us) for trace in traces)
    if trace_start != 0 or trace_end <= 0:
        raise StudyGateError("Registered episode traces must share a positive zero-based window")
    output: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for (sequence_id, profile), group in manifest.groupby(
        ["sequence_id", "mission_profile"], sort=True,
    ):
        episode = group.sort_values(
            ["frame_index", "source_uid" if "source_uid" in group else "sample_id"],
            kind="mergesort",
        ).copy()
        original_capture = pd.to_numeric(episode["capture_time_us"], errors="raise").astype(np.int64)
        budgets = (
            pd.to_numeric(episode["deadline_us"], errors="raise").astype(np.int64)
            - original_capture
        )
        if (budgets <= 0).any():
            raise StudyGateError("Episode contains a nonpositive action-deadline budget")
        rebased = original_capture - int(original_capture.min())
        episode["trace_time_us"] = rebased
        episode["capture_time_us"] = rebased
        episode["deadline_us"] = rebased + budgets.to_numpy(np.int64)
        episode_id = hash_object({
            "sequence_id": str(sequence_id), "mission_profile": str(profile),
            "sample_ids": episode["sample_id"].astype(str).tolist(),
        })[:20]
        episode["episode_id"] = episode_id
        observed_end = int(episode["deadline_us"].max())
        observed_expiry = observed_end + state_expiry_grace_us
        passed = observed_expiry < trace_end
        audit_rows.append({
            "episode_id": episode_id,
            "sequence_id": str(sequence_id),
            "mission_profile": str(profile),
            "observations": int(len(episode)),
            "first_trace_time_us": int(episode["trace_time_us"].min()),
            "last_deadline_us": observed_end,
            "state_expiry_grace_us": state_expiry_grace_us,
            "last_state_expiry_us": observed_expiry,
            "registered_trace_end_us": trace_end,
            "passed": passed,
        })
        output.append(episode)
    audit = pd.DataFrame(audit_rows)
    if audit.empty or not audit["passed"].astype(bool).all():
        raise StudyGateError("One or more registered VisDrone episodes exceed trace coverage")
    rebased = pd.concat(output, ignore_index=True)
    scientific_coverage = scientific_episode_coverage_audit(
        traces,
        rebased,
        state_expiry_grace_us=state_expiry_grace_us,
    )
    if scientific_coverage.empty or not scientific_coverage["passed"].astype(bool).all():
        raise StudyGateError(
            "Registered VisDrone episodes fail scientific capture/deadline/state-expiry coverage"
        )
    audit["scientific_episode_coverage_passed"] = True
    return rebased, audit


def _linearize_static_stream(manifest: pd.DataFrame) -> pd.DataFrame:
    """Rebase each independent visual-group/profile static episode to zero."""

    required = {
        "visual_group_id", "mission_profile", "frame_index", "sample_id",
        "deadline_us", "capture_time_us",
    }
    if missing := required.difference(manifest.columns):
        raise StudyGateError(f"Static episode fields are missing: {sorted(missing)}")
    parts: list[pd.DataFrame] = []
    for (group_id, profile), group in manifest.groupby(
        ["visual_group_id", "mission_profile"], sort=True,
    ):
        ordered = group.sort_values(["frame_index", "sample_id"], kind="mergesort").copy()
        budget = (
            pd.to_numeric(ordered["deadline_us"], errors="raise")
            - pd.to_numeric(ordered["capture_time_us"], errors="raise")
        ).astype(np.int64)
        if (budget <= 0).any():
            raise StudyGateError("Static episode contains a nonpositive deadline budget")
        times = np.arange(len(ordered), dtype=np.int64) * 200_000
        ordered["trace_time_us"] = times
        ordered["capture_time_us"] = times
        ordered["deadline_us"] = times + budget.to_numpy(np.int64)
        ordered["static_episode_id"] = hash_object({
            "visual_group_id": str(group_id),
            "mission_profile": str(profile),
            "sample_ids": ordered["sample_id"].astype(str).tolist(),
        })[:20]
        parts.append(ordered)
    if not parts:
        raise StudyGateError("Static episode selection is empty")
    return pd.concat(parts, ignore_index=True)


def _static_sample_gate(manifest: pd.DataFrame, cfg: Any, *, phase: str) -> pd.DataFrame:
    """Require the frozen H1 static sample to support clustered inference."""

    network = _section(cfg, "network")
    required_key = (
        "validation_static_min_independent_groups"
        if str(phase) == "validation"
        else "static_min_independent_groups"
    )
    required_groups = int(network[required_key])
    observed_groups = int(manifest["visual_group_id"].astype(str).nunique())
    source_column = "source_sample_id" if "source_sample_id" in manifest else "sample_id"
    expected_profiles = set(map(str, _priority_profiles(cfg)))
    profile_sets = manifest.groupby(source_column, sort=False)["mission_profile"].agg(
        lambda values: set(map(str, values))
    )
    complete_pairs = bool(len(profile_sets) and profile_sets.map(
        lambda observed: observed == expected_profiles
    ).all())
    gate = pd.DataFrame([
        {
            "gate": "h1_static_independent_visual_groups",
            "phase": str(phase), "observed": observed_groups,
            "required": required_groups, "passed": observed_groups >= required_groups,
            "required_for": "completion",
        },
        {
            "gate": "h1_static_complete_profile_pairs",
            "phase": str(phase), "observed": int(complete_pairs),
            "required": 1, "passed": complete_pairs,
            "required_for": "completion",
        },
    ])
    if not gate["passed"].astype(bool).all():
        raise StudyGateError(
            f"{phase} H1 static sample fails independent-group/profile gates"
        )
    return gate


def _controller_policies(
    cfg: Any,
    policy_spec: Mapping[str, Any],
    exp2_frontier: pd.DataFrame,
) -> list[Any]:
    fixed = [
        FixedVariantPolicy(PayloadMode[item["mode"]], str(item["variant"]), str(item["name"]))
        for item in policy_spec["fixed"]
    ]
    fixed.append(FixedVariantPolicy(
        PayloadMode[policy_spec["best_fixed"]["mode"]],
        str(policy_spec["best_fixed"]["variant"]), "best_fixed_validation",
    ))
    control = _section(cfg, "controller")
    hardware = _section(cfg, "hardware")
    controller_kwargs = {
        "minimum_dwell_us": int(control.get("minimum_dwell_us", 2_000_000)),
        "hysteresis_margin": float(control.get("hysteresis_margin", 0.04)),
    }
    full_kwargs = {
        **controller_kwargs,
        "thermal_limit_c": float(hardware.get("thermal_limit_c", 82)),
        **{name: float(control[name]) for name in (
            "risk_escalation", "uncertainty_escalation", "byte_weight",
            "energy_weight", "evidence_weight", "deadline_weight",
            "vlm_trigger_risk", "vlm_trigger_uncertainty",
        )},
        "calibration_trusted": bool(policy_spec.get("calibration_trusted", True)),
    }
    policies: list[Any] = [
        *fixed, AdaptiveImagePolicy(), AdaptiveSemanticPolicy(),
        ChannelOnlyPolicy(**controller_kwargs), MissionOnlyPolicy(), DeadlineGreedyPolicy(),
        ChannelMissionPolicy(), RiskAwareNoUncertaintyPolicy(**full_kwargs),
        NoMissionRiskPolicy(**full_kwargs), NoEvidenceEscalationPolicy(**full_kwargs),
        NoReceiverStatePolicy(**full_kwargs),
        NoHysteresisPolicy(**{key: value for key, value in full_kwargs.items() if key not in controller_kwargs}),
        AlwaysVLMPolicy(**full_kwargs), FullPolicy(**full_kwargs), LookaheadReferencePolicy(),
    ]
    if bool(policy_spec.get("feature_baseline_available")):
        policies.append(AdaptiveFeaturePolicy())
    detector = exp2_frontier[
        exp2_frontier["mode"].eq("L2_SEMANTICS")
        & exp2_frontier["variant"].astype(str).str.endswith("__detector_only")
    ]
    if detector.empty:
        raise StudyGateError("Detector-only semantic baseline is absent")
    policies.append(FixedVariantPolicy(
        PayloadMode.L2_SEMANTICS, str(detector.iloc[0]["variant"]), "detector_only_semantics",
    ))
    if not causal_schema_audit().get("passed", False):
        raise StudyGateError("Causal controller schema audit failed")
    return policies


def _validation_policy_spec(
    cfg: Any,
    profiles: pd.DataFrame,
    validation_manifest: pd.DataFrame,
    *,
    calibration_trusted: bool = True,
    balanced_success: bool = False,
) -> dict[str, Any]:
    exposure_us = 0
    episode_columns = [
        column for column in ("sequence_id", "mission_profile")
        if column in validation_manifest
    ]
    groups = (
        validation_manifest.groupby(episode_columns, sort=True)
        if episode_columns else [("episode", validation_manifest)]
    )
    for _, episode in groups:
        capture = np.sort(pd.to_numeric(
            episode["trace_time_us"], errors="raise",
        ).unique())
        interval = max(1, int(np.median(np.diff(capture)))) if len(capture) > 1 else 200_000
        exposure_us += int(capture[-1] - capture[0] + interval) if len(capture) else interval
    rate_hz = len(validation_manifest) / max(exposure_us / 1e6, 1e-9)
    candidates = profiles.copy()
    candidates["estimated_wire_rate_bytes_per_s"] = candidates["payload_bytes"] * rate_hz
    candidates["estimated_energy_rate_j_per_s"] = candidates["total_compute_energy_j"] * rate_hz
    evaluation = _section(cfg, "evaluation")
    fixed: list[dict[str, Any]] = []
    for mode, group in candidates.groupby("mode", sort=True):
        complete = group[group["observations"].eq(validation_manifest["sample_id"].nunique())]
        if complete.empty:
            raise StudyGateError(f"No full-coverage validation candidate for {mode}")
        under = complete[
            complete["estimated_wire_rate_bytes_per_s"].le(1.05 * float(evaluation["primary_wire_budget_bytes_per_s"]))
            & complete["estimated_energy_rate_j_per_s"].le(1.05 * float(evaluation["primary_energy_budget_j_per_s"]))
        ]
        pool = under if len(under) else complete
        if mode == "L5_IMAGE_OR_CLIP":
            practical = pool[~pool["variant"].astype(str).str.contains("lossless", case=False)]
            if len(practical):
                pool = practical
        chosen = pool.sort_values(
            ["predicted_task_success", "payload_bytes", "total_compute_energy_j", "total_processing_latency_ms"],
            ascending=[False, True, True, True], kind="mergesort",
        ).iloc[0]
        fixed.append({
            "name": f"fixed_{str(mode).lower()}", "mode": str(mode),
            "variant": str(chosen.variant),
            "rate_screen_passed": bool(len(under)),
            "selection": "validation_success_then_resources_with_registered_rate_screen",
        })
    # v3r2 correction (receipted in v3r1 as v3r1c): the best fixed comparator
    # is chosen among representations that pass the registered rate screen.
    # The v3r1 rule selected an L4 variant that failed the screen and sent
    # zero bytes in Experiment 4.
    rate_screened_fixed = [item for item in fixed if bool(item["rate_screen_passed"])]
    if not rate_screened_fixed:
        raise StudyGateError(
            "No fixed representation passes the registered rate screen; "
            "best_fixed_validation is undefined"
        )
    best = max(
        rate_screened_fixed,
        key=lambda item: float(candidates.loc[
            candidates["mode"].eq(item["mode"]) & candidates["variant"].eq(item["variant"]),
            "predicted_task_success",
        ].iloc[0]),
    )
    primary_h2 = [
        "best_fixed_validation",
        "adaptive_image_codec", "adaptive_semantic_rate",
        "detector_only_semantics", "channel_only_hysteresis", "mission_only",
        "queue_deadline_greedy", "channel_plus_mission", "risk_aware_no_uncertainty",
    ]
    secondary_h2 = [
        *[item["name"] for item in fixed],
        "ablate_mission_risk", "ablate_evidence_escalation", "ablate_receiver_state",
        "ablate_hysteresis", "always_vlm_full", "heuristic_lookahead_reference",
    ]
    feature = any(item["mode"] == "L4_FEATURES" for item in fixed)
    if feature:
        primary_h2.append("adaptive_feature_rate")
    return {
        "fixed": fixed, "best_fixed": {**best, "name": "best_fixed_validation"},
        "feature_baseline_available": feature,
        "required_h2_baselines": primary_h2,
        "secondary_h2_comparators": secondary_h2,
        "h2_family_definition": (
            "primary_controller_baselines_plus_best_fixed_validation_v1"
        ),
        "controller": dict(_section(cfg, "controller")),
        "best_fixed_rule": "max_predicted_success_among_rate_screened_fixed_v3r2",
        "calibration_trusted": bool(calibration_trusted),
        "predicted_task_success_estimand": (
            "class_balanced_receiver_recall_v3r2" if balanced_success
            else "raw_receiver_accuracy"
        ),
        "controller_revision": CONTROLLER_REVISION_V3R2,
    }


def calibration_gate_row(
    *, calibration_pass: bool, selected_model: str, policy_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Registered calibration gate with the v3r2 pre-test amendment.

    The amendment (CALIBRATION_GATE_AMENDMENT_V3R2) keeps the calibration
    measurement and its FAIL status unchanged.  A failed gate no longer blocks
    the protocol freeze when the frozen controller cannot consume model
    confidence (``calibration_trusted`` is False under the v3r2 controller
    revision); calibration-dependent claims (H4) are instead NOT_ESTIMABLE.
    """
    routed = bool(
        not calibration_pass
        and policy_spec.get("calibration_trusted") is False
        and policy_spec.get("controller_revision") == CONTROLLER_REVISION_V3R2
    )
    if calibration_pass:
        detail = f"selected={selected_model}; calibration=PASS"
    elif routed:
        detail = (
            f"selected={selected_model}; calibration=FAIL; routed by "
            f"{CALIBRATION_GATE_AMENDMENT_V3R2}: confidence-dependent escalation "
            "disabled in the frozen controller and H4 NOT_ESTIMABLE"
        )
    else:
        detail = f"selected={selected_model}; calibration=FAIL"
    return {
        "gate": "selected_and_detector_calibration",
        "passed": bool(calibration_pass or routed),
        "detail": detail,
        "calibration_measurement_passed": bool(calibration_pass),
        "calibration_amendment_routed": routed,
    }


def _selected_calibration_trusted(selection: Mapping[str, Any]) -> bool:
    """Whether the registered calibration gate passed for the selected model."""
    rows = [
        row for row in selection.get("calibration_gate", [])
        if isinstance(row, Mapping)
        and str(row.get("model_id")) == str(selection.get("selected_model"))
    ]
    if len(rows) != 1:
        raise StudyGateError("Selected model lacks exactly one calibration-gate row")
    return bool(rows[0].get("passed") is True)


def run_validation_network_and_controller(
    ctx: RunContext,
    cfg: Any,
    manifest: pd.DataFrame,
    traces: Any,
    exp2_frontier: pd.DataFrame,
    selection: MutableMapping[str, Any],
    temporal_manifest: pd.DataFrame | None = None,
    on_experiment_complete: ExperimentCallback | None = None,
) -> dict[str, Any]:
    """Run validation Experiments 3--5 and freeze comparator specifications."""

    # This mutable selection becomes the protocol decision object. The v2r13
    # active model lock is its complete model-dependency identity.
    registered_validation = _require_registered_phase_sample(
        manifest, cfg, expected_splits=("validation",),
        require_validation_coverage=True,
    )
    if temporal_manifest is None:
        raise StudyGateError(
            "Validation H2--H4 require the separately registered MOT temporal manifest"
        )
    registered_temporal, temporal_gate = _require_registered_temporal_manifest(
        temporal_manifest, cfg, split="validation",
    )
    artifact_map = selection.get("artifacts", {})
    policy_path = artifact_map.get("policy_manifest")
    if not policy_path or not Path(policy_path).is_file():
        raise StudyGateError("Validation policy manifest from real perception is missing")
    general_policy_manifest = pd.read_csv(policy_path, compression="infer")
    validation_manifest = general_policy_manifest[
        general_policy_manifest["split"].astype(str).eq("validation")
    ].copy()
    if validation_manifest.empty:
        raise StudyGateError("No validation observations are available")
    if set(validation_manifest["sample_id"].astype(str)) != set(
        registered_validation["sample_id"].astype(str)
    ):
        raise StudyGateError(
            "Validation network sample differs from the registered perception sample"
        )
    temporal_policy_path = artifact_map.get("temporal_policy_manifest")
    temporal_frontier_path = artifact_map.get("temporal_exp2_frontier")
    if (
        not temporal_policy_path or not Path(temporal_policy_path).is_file()
        or not temporal_frontier_path or not Path(temporal_frontier_path).is_file()
    ):
        raise StudyGateError(
            "Validation temporal perception/frontier artifacts are missing"
        )
    temporal_policy_manifest = pd.read_csv(temporal_policy_path, compression="infer")
    temporal_exp2_frontier = pd.read_csv(temporal_frontier_path, compression="infer")
    if set(temporal_policy_manifest["sample_id"].astype(str)) != set(
        registered_temporal["sample_id"].astype(str)
    ):
        raise StudyGateError(
            "Validation temporal network sample differs from the registered temporal sample"
        )
    development_ids = temporal_policy_manifest["sample_id"].astype(str)
    if "gold_action" not in temporal_policy_manifest:
        raise StudyGateError("Temporal policy manifest lacks registered gold actions")
    gold_actions = dict(zip(
        temporal_policy_manifest["sample_id"].astype(str),
        temporal_policy_manifest["gold_action"].astype(str),
    ))
    calibration_trusted = _selected_calibration_trusted(selection)
    profiles = build_causal_candidate_profiles(
        temporal_exp2_frontier, development_ids,
        semantic_acceptance_threshold=_h4_high_confidence_threshold(cfg),
        gold_actions=gold_actions,
        calibration_trusted=calibration_trusted,
    )
    profiles_path = _write_frame(
        ctx, "validation_causal_candidate_profiles.csv.gz", profiles,
    )
    policy_spec = _validation_policy_spec(
        cfg, profiles, temporal_policy_manifest,
        calibration_trusted=calibration_trusted, balanced_success=True,
    )
    policy_spec_path = atomic_write_json(_results(ctx) / "validation_policy_spec.json", policy_spec)

    network = _section(cfg, "network")
    static_manifest = select_experiment3_manifest(
        validation_manifest,
        int(network.get("static_max_groups_per_dataset", 3)), _seed(cfg),
        int(network.get("static_max_observations_per_group", 8)),
    )
    static_manifest = _linearize_static_stream(static_manifest)
    validation_static_gate = _static_sample_gate(
        static_manifest, cfg, phase="validation",
    )
    ids = set(static_manifest["sample_id"].astype(str))
    static_candidates = exp2_frontier[exp2_frontier["sample_id"].astype(str).isin(ids)].copy()
    selected_candidates, variant_spec = select_experiment3_candidate_variants(
        static_candidates, int(network.get("static_max_variants_per_mode_source", 3)),
    )
    # Fixed comparators must enter the static characterization even when they
    # are not a wire-span point.
    selected_keys = set(variant_spec[["mode", "source_model", "variant"]].itertuples(index=False, name=None))
    for item in policy_spec["fixed"]:
        matching = static_candidates[
            static_candidates["mode"].eq(item["mode"])
            & static_candidates["variant"].eq(item["variant"])
        ]
        for source_model, source_rows in matching.groupby("source_model", sort=True):
            key = (str(item["mode"]), str(source_model), str(item["variant"]))
            if key in selected_keys:
                continue
            if source_rows["sample_id"].nunique() != len(ids):
                raise StudyGateError(f"Fixed static candidate lacks full coverage: {key}")
            variant_spec = pd.concat([variant_spec, pd.DataFrame([{
                "mode": key[0], "source_model": key[1], "variant": key[2],
                "validation_median_wire_bytes": float(source_rows["wire_bytes"].median()),
                "selection_rank": int(len(variant_spec)),
                "selection_version": "semuav.exp3.variant-wire-span.v1",
            }])], ignore_index=True)
            selected_keys.add(key)
    selected_candidates = static_candidates[static_candidates.apply(
        lambda row: (str(row["mode"]), str(row["source_model"]), str(row["variant"])) in selected_keys,
        axis=1,
    )]
    exp3_summary, exp3_detail = experiment3_static_transport_partitioned(
        selected_candidates, static_manifest, ctx, registered_seeds=_seeds(cfg),
        rates_mbps=network["static_rate_mbps"], rtts_ms=network["rtt_ms"],
        losses=network["loss"], loss_models=network["static_loss_models"],
        burst_recovery_probability=float(network["static_burst_recovery_probability"]),
        burst_slot_ms=float(network["static_burst_slot_ms"]),
        jitters_ms=network.get("static_jitter_ms", (0.0, 10.0)),
        queue_kb=network["queue_kb"],
        radio_tx_j_per_byte=float(_section(cfg, "hardware").get("radio_tx_j_per_byte", 0.0)),
        radio_rx_j_per_byte=float(_section(cfg, "hardware").get("radio_rx_j_per_byte", 0.0)),
        require_perception_costs=True, require_registered_loss_models=True,
    )
    _write_frame(ctx, "validation_exp3_partition_index.csv.gz", exp3_detail)
    _write_frame(ctx, "validation_exp3_summary.csv.gz", exp3_summary)
    _emit_experiment_complete(on_experiment_complete, 3, "validation", {
        "exp3_summary": exp3_summary,
    })
    dynamic = _load_traces_for_split(traces, ctx=ctx, split="validation")
    frozen_test_trace_design = _load_traces_for_split(
        traces, ctx=ctx, split="test",
    )
    frozen_test_uplink_traces = {
        str(trace.trace_id) for trace in frozen_test_trace_design
        if str(getattr(trace.direction, "value", trace.direction)).lower() == "uplink"
    }
    temporal_policy_manifest, episode_audit = _rebase_registered_episodes(
        temporal_policy_manifest,
        dynamic,
        state_expiry_grace_us=int(network["temporal_state_expiry_grace_us"]),
    )
    episode_audit_path = _write_frame(
        ctx, "validation_episode_trace_coverage.csv.gz", episode_audit,
    )
    policies = _controller_policies(cfg, policy_spec, temporal_exp2_frontier)
    validation_policy_partition_index = run_paired_policy_trials_sharded(
        temporal_policy_manifest, temporal_exp2_frontier, dynamic, policies,
        _seeds(cfg), profiles,
        ctx, scientific=True, return_partition_index=True,
        semantic_acceptance_threshold=_h4_high_confidence_threshold(cfg),
        **_radio_energy_arguments(cfg),
    )
    validation_cartesian = _dynamic_partition_coverage_audit(
        validation_policy_partition_index,
        temporal_policy_manifest,
        dynamic,
        policies,
        _seeds(cfg),
    )
    evaluation = _section(cfg, "evaluation")
    validation_reduction = reduce_policy_trial_partitions(
        validation_policy_partition_index,
        _results(ctx) / "validation_policy_reduction",
        h4_target_coverage=float(evaluation["h4_matched_coverage"]),
        semantic_acceptance_threshold=_h4_high_confidence_threshold(cfg),
        energy_scope=str(evaluation["energy_claim_scope"]),
    )
    compact = dict(validation_reduction.frames)
    # Experiments 4 and 5 reuse the same paired trials by design; present
    # controller outcomes first, then the separately reduced safety endpoints.
    _write_frame(ctx, "validation_exp4_summary.csv.gz", compact["controller_summary"])
    _emit_experiment_complete(on_experiment_complete, 4, "validation", {
        "exp4_summary": compact["resource_summary"],
        "exp4_controller_summary": compact["controller_summary"],
    })
    _write_frame(ctx, "validation_exp5_summary.csv.gz", compact["exp5_summary"])
    _emit_experiment_complete(on_experiment_complete, 5, "validation", {
        "exp5_summary": compact["exp5_summary"],
    })
    validation_proposed = compact["policy_pair_cells"][
        compact["policy_pair_cells"]["policy"].astype(str).eq("proposed_full")
    ]
    validation_paired_cells = int(validation_proposed[[
        "visual_group_id", "trace_id",
    ]].drop_duplicates().shape[0])
    validation_temporal_groups = int(
        validation_proposed["visual_group_id"].astype(str).nunique()
    )
    validation_trace_realizations = int(
        validation_proposed["trace_id"].astype(str).nunique()
    )
    registered_validation_temporal_groups = int(
        registered_temporal_sampling_spec(
            _section(cfg, "benchmark"), "validation",
        )["sequences_per_split"]
    )
    validation_budget = compact["resource_rate_summary"].copy()
    validation_budget["wire_cap_bytes_per_s"] = 1.05 * float(
        evaluation["primary_wire_budget_bytes_per_s"]
    )
    validation_budget["energy_cap_j_per_s"] = 1.05 * float(
        evaluation["primary_energy_budget_j_per_s"]
    )
    validation_budget["wire_pass"] = (
        validation_budget["wire_bytes_per_s"]
        <= validation_budget["wire_cap_bytes_per_s"]
    )
    validation_budget["energy_pass"] = (
        validation_budget["energy_j_per_s"]
        <= validation_budget["energy_cap_j_per_s"]
    )
    validation_budget["eligible"] = (
        validation_budget["wire_pass"] & validation_budget["energy_pass"]
    )
    baselines = tuple(map(str, policy_spec["required_h2_baselines"]))
    validation_pseudo_trials = _compact_policy_pseudo_trials(compact)
    validation_budget_match = _h2_pairwise_budget_match(
        validation_pseudo_trials, baselines, cfg,
    )
    validation_h3_gate = pd.DataFrame([
        _compact_h3_estimability_gate(
            compact["policy_pair_cells"], compact["h3_negative_episode_cells"], cfg,
        )
    ])
    power_rows: list[dict[str, Any]] = []
    for baseline in baselines:
        pair = compact["policy_pair_cells"][
            compact["policy_pair_cells"]["policy"].isin(["proposed_full", baseline])
        ].pivot(
            index=["visual_group_id", "trace_id"],
            columns="policy",
            values="correct_timely_rate",
        )
        if pair.empty or pair[["proposed_full", baseline]].isna().any().any():
            raise StudyGateError(f"Validation power screen lacks paired baseline {baseline}")
        # For paired Bernoulli outcomes, |p_A - p_B| is a valid lower bound on
        # discordance.  Using that bound keeps this nonblocking sensitivity
        # diagnostic conservative without materializing raw validation trials.
        discordant = float(
            (pair["proposed_full"] - pair[baseline]).abs().mean()
        )
        power = (
            validation_power_simulation(
                discordant,
                visual_clusters=int(pair.index.get_level_values("visual_group_id").nunique()),
                trace_clusters=int(pair.index.get_level_values("trace_id").nunique()),
                effect=float(evaluation.get("power_effect", 0.06)),
                null_margin=float(evaluation.get("primary_margin", 0.03)),
                simulations=int(evaluation.get("power_simulations", 5000)),
                seed=_seed(cfg),
            )
            if discordant > 0 else 0.0
        )
        power_rows.append({
            "method": "proposed_full", "reference": baseline,
            "discordant_rate": discordant,
            "discordance_estimand": "cell_mean_frechet_lower_bound_v1",
            "visual_clusters": int(pair.index.get_level_values("visual_group_id").nunique()),
            "trace_clusters": int(pair.index.get_level_values("trace_id").nunique()),
            "power": power,
            "required_power": float(evaluation.get("minimum_power", 0.80)),
            "assumed_effect": float(evaluation.get("power_effect", 0.06)),
            "null_margin": float(evaluation.get("primary_margin", 0.03)),
            "simulations": int(evaluation.get("power_simulations", 5000)),
            "passed": power >= float(evaluation.get("minimum_power", 0.80)),
            "blocking_protocol_gate": False,
            "role": "design_sensitivity_diagnostic",
        })
    validation_power = pd.DataFrame(power_rows)
    validation_h4_gate = compact["h4_matched_coverage_audit"].copy()
    expected_policies = {policy.name for policy in policies}
    observed_policies = set(compact["resource_summary"]["policy"].astype(str))
    h2_audit_registration = _validation_h2_resource_audit_registration(
        validation_budget_match, validation_budget,
        baselines, expected_policies,
    )
    matched_baselines = validation_budget_match.loc[
        validation_budget_match["passed"].astype(bool), "reference",
    ].astype(str).tolist()
    unmatched_baselines = validation_budget_match.loc[
        ~validation_budget_match["passed"].astype(bool), "reference",
    ].astype(str).tolist()
    policy_spec.update({
        "validation_h2_matched_baselines": matched_baselines,
        "validation_h2_unmatched_baselines": unmatched_baselines,
        "validation_h2_family_status": (
            "MATCHED" if not unmatched_baselines else "NOT_ESTIMABLE"
        ),
        "validation_h2_match_artifact": "validation_h2_matched_budget.csv.gz",
    })
    atomic_write_json(policy_spec_path, policy_spec)
    calibration_gate = pd.DataFrame(selection.get("calibration_gate", []))
    calibration_pass = all(
        bool(calibration_gate.loc[
            calibration_gate["model_id"].astype(str).eq(model), "passed"
        ].all())
        for model in {"detector_only", str(selection["selected_model"])}
        if len(calibration_gate.loc[calibration_gate["model_id"].astype(str).eq(model)])
    ) and all(
        len(calibration_gate.loc[calibration_gate["model_id"].astype(str).eq(model)]) > 0
        for model in {"detector_only", str(selection["selected_model"])}
    )
    validation_gates = pd.DataFrame([
        {"gate": "registered_temporal_controller_sample", "passed": bool(
            temporal_gate["passed"].astype(bool).all()
        ), "detail": (
            f"sequences={int(registered_temporal['sequence_id'].nunique())} "
            "observations_per_sequence=64 target_rate_hz=5"
        )},
        {"gate": "exact_policy_coverage", "passed": observed_policies == expected_policies,
         "detail": f"observed={len(observed_policies)} expected={len(expected_policies)}"},
        {"gate": "registered_feature_baseline", "passed": bool(policy_spec["feature_baseline_available"]),
         "detail": "L4 feature candidates and adaptive feature policy required"},
        h2_audit_registration,
        calibration_gate_row(
            calibration_pass=calibration_pass,
            selected_model=str(selection["selected_model"]),
            policy_spec=policy_spec,
        ),
        {"gate": "h4_denominator_audit_registered", "passed": bool(
            set(validation_h4_gate["policy"].astype(str))
            == {"proposed_full", "ablate_evidence_escalation"}
            and len(validation_h4_gate) == 2
        ), "detail": (
            f"coverage={float(evaluation['h4_matched_coverage'])}; "
            f"estimability_pass={bool(validation_h4_gate['passed'].astype(bool).all())}; "
            "denominator shortfalls remain nonblocking NOT_ESTIMABLE"
        )},
        {"gate": "h1_static_independent_groups", "passed": bool(
            validation_static_gate["passed"].astype(bool).all()
        ), "detail": (
            f"observed={int(static_manifest['visual_group_id'].nunique())} "
            f"required={int(_section(cfg, 'network')['validation_static_min_independent_groups'])}"
        )},
        {"gate": "validation_fully_paired_scene_trace_cells", "passed": (
            validation_paired_cells == int(
                _section(cfg, "network")["validation_fully_paired_scene_trace_cells"]
            )
        ), "detail": (
            f"observed={validation_paired_cells} "
            f"required={int(_section(cfg, 'network')['validation_fully_paired_scene_trace_cells'])}"
        )},
        {"gate": "exact_validation_temporal_groups", "passed": (
            validation_temporal_groups == registered_validation_temporal_groups
        ), "detail": (
            f"observed={validation_temporal_groups} "
            f"required={registered_validation_temporal_groups}"
        )},
        {"gate": "validation_trace_realizations", "passed": (
            validation_trace_realizations == int(
                _section(cfg, "network")["validation_trace_realizations"]
            )
        ), "detail": (
            f"observed={validation_trace_realizations} "
            f"required={int(_section(cfg, 'network')['validation_trace_realizations'])}"
        )},
        {"gate": "frozen_test_trace_design", "passed": (
            len(frozen_test_uplink_traces) == int(
                _section(cfg, "network")["held_out_trace_realizations"]
            )
        ), "detail": (
            f"observed={len(frozen_test_uplink_traces)} "
            f"required={int(_section(cfg, 'network')['held_out_trace_realizations'])}"
        )},
        {"gate": "exact_dynamic_cartesian_coverage", "passed": bool(
            validation_cartesian["passed"].astype(bool).all()
        ), "detail": "sample x trace x seed x policy product"},
    ])
    exp4_summary = compact["controller_summary"]
    exp5_summary = compact["exp5_summary"]
    paths = {
        "exp3_detail": _write_frame(ctx, "validation_exp3_partition_index.csv.gz", exp3_detail),
        "exp3_summary": _write_frame(ctx, "validation_exp3_summary.csv.gz", exp3_summary),
        "variant_spec": _write_frame(ctx, "validation_exp3_variant_spec.csv.gz", variant_spec),
        "exp4_summary": _write_frame(ctx, "validation_exp4_summary.csv.gz", exp4_summary),
        "exp5_summary": _write_frame(ctx, "validation_exp5_summary.csv.gz", exp5_summary),
        "candidate_profiles": profiles_path,
        "static_sample_gate": _write_frame(
            ctx, "validation_h1_static_sample_gate.csv.gz", validation_static_gate,
        ),
        "validation_power": _write_frame(ctx, "validation_power_by_baseline.csv.gz", validation_power),
        "validation_cartesian_coverage": _write_frame(
            ctx, "validation_dynamic_cartesian_coverage.csv.gz", validation_cartesian,
        ),
        "validation_h3_gate": _write_frame(
            ctx, "validation_h3_false_alert_gate.csv.gz", validation_h3_gate,
        ),
        "validation_budget": _write_frame(ctx, "validation_budget_compliance.csv.gz", validation_budget),
        "validation_h2_matched_budget": _write_frame(
            ctx, "validation_h2_matched_budget.csv.gz", validation_budget_match,
        ),
        "validation_h4_gate": _write_frame(ctx, "validation_h4_coverage_gate.csv.gz", validation_h4_gate),
        "validation_gates": _write_frame(ctx, "validation_gate_table.csv.gz", validation_gates),
        "episode_trace_coverage": episode_audit_path,
        "temporal_sample_gate": _write_frame(
            ctx, "validation_temporal_controller_sample_gate.csv.gz", temporal_gate,
        ),
        "policy_partition_index": atomic_write_text(
            _results(ctx) / "validation_policy_partition_index.csv",
            validation_policy_partition_index.to_csv(index=False, lineterminator="\n"),
        ),
        "policy_reduction_receipt": validation_reduction.receipt_path,
    }
    validation_reduction_artifacts = {
        f"policy_reduction_{name}": path
        for name, path in validation_reduction.artifacts.items()
    }
    selection.update({
        "policy_spec": policy_spec,
        "policy_spec_path": str(policy_spec_path),
        "policy_spec_sha256": hash_file(policy_spec_path),
        "exp3_variant_spec": str(paths["variant_spec"]),
        "exp3_variant_spec_sha256": hash_file(paths["variant_spec"]),
        "candidate_profiles_path": str(profiles_path),
        "candidate_profiles_sha256": hash_file(profiles_path),
        "h4_matched_coverage": float(evaluation["h4_matched_coverage"]),
        "h4_high_confidence_threshold": _h4_high_confidence_threshold(cfg),
        "representation_image_design": _image_payload_arguments(cfg),
        "representation_video_design": _video_payload_arguments(cfg),
        "h2_matched_budget_relative_tolerance": float(
            evaluation["h2_matched_budget_relative_tolerance"]
        ),
        "h2_validation_matched_baselines": matched_baselines,
        "h2_validation_unmatched_baselines": unmatched_baselines,
        "h2_validation_family_status": (
            "MATCHED" if not unmatched_baselines else "NOT_ESTIMABLE"
        ),
        "expected_not_estimable_hypotheses": list(map(
            str, evaluation.get("expected_not_estimable_hypotheses", ()),
        )),
        "expected_not_estimable_reasons": dict(
            evaluation.get("expected_not_estimable_reasons", {}),
        ),
        "h1_static_min_independent_groups": int(
            _section(cfg, "network")["static_min_independent_groups"]
        ),
        "validation_h1_static_min_independent_groups": int(
            _section(cfg, "network")["validation_static_min_independent_groups"]
        ),
        "validation_gates_passed": bool(validation_gates["passed"].all()),
        "temporal_sample_schema": TEMPORAL_SAMPLE_SCHEMA,
        "temporal_selection_sha256": str(
            registered_temporal["temporal_selection_sha256"].iloc[0]
        ),
        "temporal_policy_manifest_sha256": hash_file(
            Path(str(temporal_policy_path))
        ),
        "temporal_exp2_frontier_sha256": hash_file(
            Path(str(temporal_frontier_path))
        ),
        "gate_artifacts": {
            "validation_power_by_baseline": str(paths["validation_power"]),
            "validation_dynamic_cartesian_coverage": str(
                paths["validation_cartesian_coverage"]
            ),
            "validation_h3_false_alert_gate": str(paths["validation_h3_gate"]),
            "validation_budget_compliance": str(paths["validation_budget"]),
            "validation_h2_matched_budget": str(paths["validation_h2_matched_budget"]),
            "validation_h4_coverage_gate": str(paths["validation_h4_gate"]),
            "validation_gate_table": str(paths["validation_gates"]),
        },
        "network_artifacts": {
            **{name: str(path) for name, path in paths.items()},
            **{
                name: str(path)
                for name, path in validation_reduction_artifacts.items()
            },
        },
    })
    atomic_write_json(_results(ctx) / "validation_selection.json", selection)
    return {
        "exp3_detail": exp3_detail,
        "exp3_summary": exp3_summary,
        "exp4_summary": exp4_summary,
        "exp5_summary": exp5_summary,
        "resource_summary": compact["resource_summary"],
        "policy_partition_index": validation_policy_partition_index,
        "policy_reduction": compact,
        "policy_reduction_receipt": validation_reduction.receipt_path,
    }


def _visdrone_experiment5(
    policy_trials: pd.DataFrame,
    *,
    semantic_acceptance_threshold: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 <= float(semantic_acceptance_threshold) <= 1.0:
        raise ValueError("semantic_acceptance_threshold must lie in [0, 1]")
    required = {
        "policy", "visual_group_id", "critical", "correct_timely_action",
        "false_alert", "critical_miss", "total_wire_bytes", "energy_j",
        "selected_mode", "calibrated_correctness_probability", "correct_action",
    }
    if missing := required.difference(policy_trials.columns):
        raise StudyGateError(f"VisDrone Experiment 5 is missing: {sorted(missing)}")
    sections: list[pd.DataFrame] = []
    for name, mask in {
        "natural_prevalence": pd.Series(True, index=policy_trials.index),
        "mission_priority_positive": policy_trials["critical"].astype(bool),
        "mission_priority_negative": ~policy_trials["critical"].astype(bool),
    }.items():
        section = policy_trials.loc[mask].copy()
        if len(section):
            section["subset"] = name
            sections.append(section)
    detail = pd.concat(sections, ignore_index=True)
    confidence = pd.to_numeric(
        detail["calibrated_correctness_probability"], errors="coerce",
    )
    detail["high_confidence_semantic_eligible"] = (
        detail["selected_mode"].isin(["L1_ALERT", "L2_SEMANTICS"])
        & confidence.ge(float(semantic_acceptance_threshold))
    )
    detail["high_confidence_semantic_error"] = (
        detail["high_confidence_semantic_eligible"]
        & ~detail["correct_action"].astype(bool)
    )
    rows: list[dict[str, Any]] = []
    for (subset, policy), group in detail.groupby(["subset", "policy"], sort=True):
        group_equal = group.groupby("visual_group_id").agg(
            correct_timely_rate=("correct_timely_action", "mean"),
            mean_wire_bytes=("total_wire_bytes", "mean"),
            mean_gpu_board_energy_j=("energy_j", "mean"),
        )
        noncritical = group[~group["critical"].astype(bool)]
        critical = group[group["critical"].astype(bool)]
        high_confidence = group[
            group["high_confidence_semantic_eligible"].astype(bool)
        ]
        false_alert_rates = noncritical.groupby(
            "visual_group_id", sort=False,
        )["false_alert"].mean()
        critical_miss_rates = critical.groupby(
            "visual_group_id", sort=False,
        )["critical_miss"].mean()
        high_confidence_rates = high_confidence.groupby(
            "visual_group_id", sort=False,
        )["high_confidence_semantic_error"].mean()
        rows.append({
            "subset": subset, "policy": policy, "trials": int(len(group)),
            "independent_visual_groups": int(len(group_equal)),
            **{column: float(group_equal[column].mean()) for column in group_equal.columns},
            "noncritical_rows": int(len(noncritical)),
            "false_alerts": int(noncritical["false_alert"].astype(bool).sum()),
            "false_alert_rate": (
                float(false_alert_rates.mean()) if len(false_alert_rates) else np.nan
            ),
            "critical_rows": int(len(critical)),
            "critical_misses": int(critical["critical_miss"].astype(bool).sum()),
            "critical_miss_rate": (
                float(critical_miss_rates.mean()) if len(critical_miss_rates) else np.nan
            ),
            "high_confidence_semantic_rows": int(len(high_confidence)),
            "high_confidence_semantic_errors": int(
                high_confidence["high_confidence_semantic_error"].astype(bool).sum()
            ),
            "high_confidence_semantic_visual_groups": int(
                len(high_confidence_rates)
            ),
            "high_confidence_semantic_error_rate": (
                float(high_confidence_rates.mean())
                if len(high_confidence_rates) else np.nan
            ),
            "semantic_acceptance_threshold": float(
                semantic_acceptance_threshold
            ),
            "ood_claim_applicable": False,
            "energy_scope": (
                "combined_sender_and_ground_gpu_board_nvml_compute_only_"
                "excludes_radio_system"
            ),
        })
    return detail, pd.DataFrame(rows)


def _dynamic_cartesian_coverage_audit(
    trials: pd.DataFrame,
    manifest: pd.DataFrame,
    traces: Sequence[LinkTrace],
    policies: Sequence[Any],
    replicate_seeds: Sequence[int],
) -> pd.DataFrame:
    """Audit the exact registered sample x trace x seed x policy product.

    The audit avoids materializing the potentially large Cartesian product.
    Exact domain equality, a duplicate-key prohibition, and the exact row
    count for every sample together are logically equivalent to equality with
    that product.  This catches a missing shard even when a duplicated shard
    would otherwise preserve a headline row or scene--trace count.
    """

    key_columns = ["sample_id", "trace_id", "replicate_seed", "policy"]
    if missing := set(key_columns).difference(trials.columns):
        raise StudyGateError(
            f"Dynamic coverage audit lacks trial keys: {sorted(missing)}"
        )
    if "sample_id" not in manifest.columns:
        raise StudyGateError("Dynamic coverage audit lacks registered sample_id")
    expected_samples = set(manifest["sample_id"].astype(str))
    if len(expected_samples) != len(manifest):
        raise StudyGateError("Registered dynamic manifest has duplicate sample_id rows")
    expected_traces = {
        str(trace.trace_id) for trace in traces
        if str(getattr(trace.direction, "value", trace.direction)).lower() == "uplink"
    }
    expected_seeds = {int(seed) for seed in replicate_seeds}
    expected_policies = {str(policy.name) for policy in policies}
    if not all((expected_samples, expected_traces, expected_seeds, expected_policies)):
        raise StudyGateError("Registered dynamic Cartesian domains must all be non-empty")

    keys = trials[key_columns].copy()
    keys["sample_id"] = keys["sample_id"].astype(str)
    keys["trace_id"] = keys["trace_id"].astype(str)
    keys["replicate_seed"] = pd.to_numeric(
        keys["replicate_seed"], errors="raise",
    ).astype(np.int64)
    keys["policy"] = keys["policy"].astype(str)
    observed_samples = set(keys["sample_id"])
    observed_traces = set(keys["trace_id"])
    observed_seeds = set(map(int, keys["replicate_seed"]))
    observed_policies = set(keys["policy"])
    duplicate_rows = int(keys.duplicated(key_columns, keep=False).sum())
    expected_per_sample = (
        len(expected_traces) * len(expected_seeds) * len(expected_policies)
    )
    per_sample = keys.groupby("sample_id", sort=False).size()
    complete_samples = int(
        per_sample.reindex(sorted(expected_samples), fill_value=0)
        .eq(expected_per_sample).sum()
    )
    expected_rows = len(expected_samples) * expected_per_sample
    domain_pass = {
        "sample_domain": observed_samples == expected_samples,
        "trace_domain": observed_traces == expected_traces,
        "replicate_seed_domain": observed_seeds == expected_seeds,
        "policy_domain": observed_policies == expected_policies,
    }
    exact_product = bool(
        all(domain_pass.values())
        and duplicate_rows == 0
        and len(keys) == expected_rows
        and complete_samples == len(expected_samples)
    )
    rows = [
        {
            "gate": name,
            "observed": int(observed),
            "required": int(required),
            "passed": bool(passed),
        }
        for name, observed, required, passed in (
            ("sample_domain", len(observed_samples), len(expected_samples), domain_pass["sample_domain"]),
            ("trace_domain", len(observed_traces), len(expected_traces), domain_pass["trace_domain"]),
            ("replicate_seed_domain", len(observed_seeds), len(expected_seeds), domain_pass["replicate_seed_domain"]),
            ("policy_domain", len(observed_policies), len(expected_policies), domain_pass["policy_domain"]),
            ("no_duplicate_cartesian_keys", duplicate_rows, 0, duplicate_rows == 0),
            ("complete_samples", complete_samples, len(expected_samples), complete_samples == len(expected_samples)),
            ("exact_cartesian_rows", len(keys), expected_rows, exact_product),
        )
    ]
    output = pd.DataFrame(rows)
    output["estimand"] = "exact_sample_trace_seed_policy_cartesian_v1"
    output["expected_per_sample"] = expected_per_sample
    return output


def _dynamic_partition_coverage_audit(
    index: pd.DataFrame,
    manifest: pd.DataFrame,
    traces: Sequence[LinkTrace],
    policies: Sequence[Any],
    replicate_seeds: Sequence[int],
) -> pd.DataFrame:
    """Prove the exact dynamic product from checksum-bound raw partitions."""

    required = {
        "stage", "artifact_path", "artifact_sha256", "row_count",
        "trace_id", "replicate_seed", "episodes",
    }
    if not isinstance(index, pd.DataFrame) or index.empty:
        raise StudyGateError("Dynamic policy partition index is missing or empty")
    if missing := required.difference(index.columns):
        raise StudyGateError(f"Dynamic partition index lacks: {sorted(missing)}")
    expected_traces = {
        str(trace.trace_id) for trace in traces
        if str(getattr(trace.direction, "value", trace.direction)).lower() == "uplink"
    }
    expected_seeds = set(map(int, replicate_seeds))
    expected_policies = {str(policy.name) for policy in policies}
    expected_episodes = int(
        manifest[["sequence_id", "mission_profile"]].drop_duplicates().shape[0]
    )
    expected_rows_per_partition = int(len(manifest) * len(expected_policies))
    keys = index[["trace_id", "replicate_seed"]].copy()
    keys["trace_id"] = keys["trace_id"].astype(str)
    keys["replicate_seed"] = pd.to_numeric(
        keys["replicate_seed"], errors="raise",
    ).astype(int)
    exact_domain = bool(
        set(keys["trace_id"]) == expected_traces
        and set(keys["replicate_seed"]) == expected_seeds
        and len(keys) == len(expected_traces) * len(expected_seeds)
        and not keys.duplicated().any()
    )
    row_counts = pd.to_numeric(index["row_count"], errors="raise").astype(int)
    episode_counts = pd.to_numeric(index["episodes"], errors="raise").astype(int)
    rows = pd.DataFrame([
        {
            "gate": "trace_seed_partition_domain",
            "observed": int(len(index)),
            "required": int(len(expected_traces) * len(expected_seeds)),
            "passed": exact_domain,
        },
        {
            "gate": "exact_rows_per_partition",
            "observed": int(row_counts.eq(expected_rows_per_partition).sum()),
            "required": int(len(index)),
            "passed": bool(row_counts.eq(expected_rows_per_partition).all()),
        },
        {
            "gate": "exact_episodes_per_partition",
            "observed": int(episode_counts.eq(expected_episodes).sum()),
            "required": int(len(index)),
            "passed": bool(episode_counts.eq(expected_episodes).all()),
        },
    ])
    if not rows["passed"].astype(bool).all():
        raise StudyGateError("Dynamic partition Cartesian coverage is incomplete")
    rows["estimand"] = "exact_temporal_sample_trace_seed_policy_partition_product_v1"
    rows["expected_policies"] = int(len(expected_policies))
    rows["expected_samples"] = int(len(manifest))
    return rows


def _two_way_cluster_false_alert_upper(
    negative_rows: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    """Episode-equal FAR and one-sided two-way cluster-bootstrap upper bound."""

    episode_keys = [
        "visual_group_id", "mission_profile", "trace_id", "replicate_seed",
    ]
    if missing := set([*episode_keys, "false_alert"]).difference(negative_rows.columns):
        raise StudyGateError(f"H3 false-alert endpoint lacks: {sorted(missing)}")
    if repetitions <= 0:
        raise ValueError("H3 bootstrap repetitions must be positive")
    episodes = (
        negative_rows.groupby(episode_keys, sort=True, dropna=False)
        .agg(
            false_alert_rate=("false_alert", "mean"),
            negative_decision_exposures=("false_alert", "size"),
        )
        .reset_index()
    )
    if episodes.empty:
        return {
            "point": np.nan, "upper_95": np.nan,
            "negative_decision_exposures": 0, "episode_cells": 0,
            "visual_clusters": 0, "trace_clusters": 0,
        }
    visual = sorted(episodes["visual_group_id"].astype(str).unique())
    traces = sorted(episodes["trace_id"].astype(str).unique())
    visual_index = {value: index for index, value in enumerate(visual)}
    trace_index = {value: index for index, value in enumerate(traces)}
    sums = np.zeros((len(visual), len(traces)), dtype=float)
    counts = np.zeros_like(sums)
    for row in episodes.itertuples(index=False):
        vi = visual_index[str(row.visual_group_id)]
        ti = trace_index[str(row.trace_id)]
        sums[vi, ti] += float(row.false_alert_rate)
        counts[vi, ti] += 1.0
    point = float(episodes["false_alert_rate"].mean())
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(repetitions), dtype=float)
    visual_probability = np.full(len(visual), 1.0 / len(visual))
    trace_probability = np.full(len(traces), 1.0 / len(traces))
    for index in range(int(repetitions)):
        visual_weights = rng.multinomial(len(visual), visual_probability).astype(float)
        trace_weights = rng.multinomial(len(traces), trace_probability).astype(float)
        denominator = float(visual_weights @ counts @ trace_weights)
        draws[index] = (
            float(visual_weights @ sums @ trace_weights) / denominator
            if denominator > 0 else np.nan
        )
    finite = draws[np.isfinite(draws)]
    if len(finite) < int(repetitions) * 0.99:
        raise StudyGateError(
            "H3 two-way bootstrap produced too many empty cluster replicates"
        )
    return {
        "point": point,
        "upper_95": float(np.quantile(finite, 0.95, method="linear")),
        "negative_decision_exposures": int(
            episodes["negative_decision_exposures"].sum()
        ),
        "episode_cells": int(len(episodes)),
        "visual_clusters": int(len(visual)),
        "trace_clusters": int(len(traces)),
    }


def _h3_estimability_gate(exp4: pd.DataFrame, cfg: Any) -> dict[str, Any]:
    """Compute the registered H3 count and clustered false-alert gates."""

    required = {
        "policy", "visual_group_id", "mission_profile", "trace_id",
        "replicate_seed", "critical", "false_alert",
    }
    if missing := required.difference(exp4.columns):
        raise StudyGateError(f"H3 gate lacks trial columns: {sorted(missing)}")
    evaluation = _section(cfg, "evaluation")
    positive_groups = int(
        exp4.loc[exp4["critical"].astype(bool), "visual_group_id"].astype(str).nunique()
    )
    required_positive_groups = max(
        int(_section(cfg, "split").get("min_critical_test_events", 0)),
        int(evaluation["mission_priority_positive_units"]),
    )
    negative = exp4.loc[~exp4["critical"].astype(bool)].copy()
    proposed_rows = negative[
        negative["policy"].astype(str).eq("proposed_full")
    ]
    reference_rows = negative[
        negative["policy"].astype(str).eq("channel_only_hysteresis")
    ]
    episode_keys = [
        "visual_group_id", "mission_profile", "trace_id", "replicate_seed",
    ]
    proposed_keys = set(
        proposed_rows[episode_keys].astype(str).itertuples(index=False, name=None)
    )
    reference_keys = set(
        reference_rows[episode_keys].astype(str).itertuples(index=False, name=None)
    )
    if proposed_keys != reference_keys:
        raise StudyGateError("H3 negative episodes are not paired across policies")
    proposed_groups = set(proposed_rows["visual_group_id"].astype(str))
    reference_groups = set(reference_rows["visual_group_id"].astype(str))
    if proposed_groups != reference_groups:
        raise StudyGateError("H3 negative visual groups are not paired across policies")
    negative_groups = int(len(proposed_groups))
    required_negative_groups = int(evaluation["mission_priority_negative_units"])
    repetitions = int(evaluation["bootstrap_repetitions"])
    proposed = _two_way_cluster_false_alert_upper(
        proposed_rows, repetitions=repetitions, seed=_seed(cfg) + 3101,
    )
    reference = _two_way_cluster_false_alert_upper(
        reference_rows, repetitions=repetitions, seed=_seed(cfg) + 3109,
    )
    estimable = bool(
        positive_groups >= required_positive_groups
        and negative_groups >= required_negative_groups
    )
    false_alert_pass = bool(
        estimable
        and np.isfinite(proposed["upper_95"])
        and np.isfinite(reference["upper_95"])
        and proposed["upper_95"] <= float(evaluation["false_alert_limit"])
        and reference["upper_95"] <= float(evaluation["false_alert_limit"])
    )
    return {
        "estimand": "group_trace_equal_negative_decision_false_alert_rate_v1",
        "bootstrap_method": "one_sided_95_two_way_cluster_visual_group_trace_v1",
        "bootstrap_repetitions": repetitions,
        "positive_visual_groups": positive_groups,
        "required_positive_visual_groups": required_positive_groups,
        "negative_visual_groups": negative_groups,
        "required_negative_visual_groups": required_negative_groups,
        "proposed_false_alert_rate": float(proposed["point"]),
        "reference_false_alert_rate": float(reference["point"]),
        "proposed_false_alert_cluster_bootstrap_upper_95": float(proposed["upper_95"]),
        "reference_false_alert_cluster_bootstrap_upper_95": float(reference["upper_95"]),
        "proposed_negative_decision_exposures": int(proposed["negative_decision_exposures"]),
        "reference_negative_decision_exposures": int(reference["negative_decision_exposures"]),
        "proposed_negative_episode_cells": int(proposed["episode_cells"]),
        "reference_negative_episode_cells": int(reference["episode_cells"]),
        "negative_trace_clusters": int(proposed["trace_clusters"]),
        "estimable": estimable,
        "false_alert_gate_pass": false_alert_pass,
    }


def _compact_h3_estimability_gate(
    pair_cells: pd.DataFrame,
    negative_episode_cells: pd.DataFrame,
    cfg: Any,
) -> dict[str, Any]:
    """Compute the same H3 diagnostic from bounded reducer sufficient statistics."""

    required_pair = {
        "policy", "visual_group_id", "trace_id", "critical_trials",
    }
    required_negative = {
        "policy", "visual_group_id", "mission_profile", "trace_id",
        "replicate_seed", "episode_id", "negative_decision_exposures",
        "false_alert_rate",
    }
    if missing := required_pair.difference(pair_cells.columns):
        raise StudyGateError(f"Compact H3 pair cells lack: {sorted(missing)}")
    if missing := required_negative.difference(negative_episode_cells.columns):
        raise StudyGateError(f"Compact H3 negative cells lack: {sorted(missing)}")
    evaluation = _section(cfg, "evaluation")
    proposed_pair = pair_cells[pair_cells["policy"].astype(str).eq("proposed_full")]
    positive_groups = int(
        proposed_pair.loc[
            pd.to_numeric(proposed_pair["critical_trials"], errors="raise").gt(0),
            "visual_group_id",
        ].astype(str).nunique()
    )
    required_positive_groups = max(
        int(_section(cfg, "split").get("min_critical_test_events", 0)),
        int(evaluation["mission_priority_positive_units"]),
    )
    proposed = negative_episode_cells[
        negative_episode_cells["policy"].astype(str).eq("proposed_full")
    ].copy()
    reference = negative_episode_cells[
        negative_episode_cells["policy"].astype(str).eq("channel_only_hysteresis")
    ].copy()
    pair_keys = [
        "visual_group_id", "mission_profile", "trace_id", "replicate_seed", "episode_id",
    ]
    proposed_keys = set(proposed[pair_keys].astype(str).itertuples(index=False, name=None))
    reference_keys = set(reference[pair_keys].astype(str).itertuples(index=False, name=None))
    if proposed_keys != reference_keys:
        raise StudyGateError("Compact H3 negative episodes are not paired across policies")
    negative_groups = int(proposed["visual_group_id"].astype(str).nunique())
    required_negative_groups = int(evaluation["mission_priority_negative_units"])
    repetitions = int(evaluation["bootstrap_repetitions"])
    proposed_audit = _two_way_cluster_false_alert_upper(
        proposed.rename(columns={"false_alert_rate": "false_alert"}),
        repetitions=repetitions,
        seed=_seed(cfg) + 3101,
    )
    reference_audit = _two_way_cluster_false_alert_upper(
        reference.rename(columns={"false_alert_rate": "false_alert"}),
        repetitions=repetitions,
        seed=_seed(cfg) + 3109,
    )
    estimable = bool(
        positive_groups >= required_positive_groups
        and negative_groups >= required_negative_groups
    )
    false_alert_pass = bool(
        np.isfinite(proposed_audit["upper_95"])
        and np.isfinite(reference_audit["upper_95"])
        and proposed_audit["upper_95"] <= float(evaluation["false_alert_limit"])
        and reference_audit["upper_95"] <= float(evaluation["false_alert_limit"])
    )
    return {
        "estimand": "group_trace_equal_negative_decision_false_alert_rate_v1",
        "bootstrap_method": "one_sided_95_two_way_cluster_visual_group_trace_v1",
        "bootstrap_repetitions": repetitions,
        "positive_visual_groups": positive_groups,
        "required_positive_visual_groups": required_positive_groups,
        "negative_visual_groups": negative_groups,
        "required_negative_visual_groups": required_negative_groups,
        "proposed_false_alert_rate": float(proposed_audit["point"]),
        "reference_false_alert_rate": float(reference_audit["point"]),
        "proposed_false_alert_cluster_bootstrap_upper_95": float(
            proposed_audit["upper_95"]
        ),
        "reference_false_alert_cluster_bootstrap_upper_95": float(
            reference_audit["upper_95"]
        ),
        "proposed_negative_decision_exposures": int(
            proposed_audit["negative_decision_exposures"]
        ),
        "reference_negative_decision_exposures": int(
            reference_audit["negative_decision_exposures"]
        ),
        "proposed_negative_episode_cells": int(proposed_audit["episode_cells"]),
        "reference_negative_episode_cells": int(reference_audit["episode_cells"]),
        "negative_trace_clusters": int(proposed_audit["trace_clusters"]),
        "estimable": estimable,
        "false_alert_gate_pass": false_alert_pass,
    }


def _h2_pairwise_budget_match(
    trials: pd.DataFrame,
    baselines: Sequence[str],
    cfg: Any,
) -> pd.DataFrame:
    """Compare realized policy resource rates under the frozen matched-budget rule."""

    evaluation = _section(cfg, "evaluation")
    tolerance = float(evaluation["h2_matched_budget_relative_tolerance"])
    if not 0 <= tolerance < 1:
        raise StudyGateError("H2 matched-budget tolerance must lie within [0, 1)")
    trials = trials.copy()
    if "episode_id" not in trials.columns:
        # Backward-compatible direct callers represent one independent episode
        # per visual group.  Scientific VisDrone execution supplies the frozen,
        # hash-bound episode identifier produced by `_rebase_registered_episodes`.
        if "visual_group_id" not in trials.columns:
            raise StudyGateError("H2 matched-budget input lacks episode fields: ['episode_id']")
        trials["episode_id"] = "legacy::" + trials["visual_group_id"].astype(str)
    required_episode_columns = {
        "episode_id", "visual_group_id", "policy", "trace_id", "replicate_seed",
        "capture_us", "total_wire_bytes", "energy_j",
    }
    if missing := required_episode_columns.difference(trials.columns):
        raise StudyGateError(
            f"H2 matched-budget input lacks episode fields: {sorted(missing)}"
        )
    rates = budget_compliance(
        trials,
        float(evaluation["primary_wire_budget_bytes_per_s"]),
        float(evaluation["primary_energy_budget_j_per_s"]),
    ).set_index("policy")
    component_columns = [
        "representation_energy_j", "detector_energy_j", "context_model_energy_j",
        "receiver_energy_j_charged",
    ]
    if missing := set(component_columns).difference(trials.columns):
        raise StudyGateError(
            f"H2 combined-compute scope lacks energy components: {sorted(missing)}"
        )
    episode_rows: list[dict[str, Any]] = []
    for (policy, trace_id, seed, episode_id), group in trials.groupby(
        ["policy", "trace_id", "replicate_seed", "episode_id"], sort=True,
    ):
        captures = np.sort(pd.to_numeric(group["capture_us"], errors="raise").unique())
        if len(captures) >= 2:
            interval_us = max(1, int(np.median(np.diff(captures))))
            duration_s = float(captures[-1] - captures[0] + interval_us) / 1e6
        else:
            duration_s = 1.0
        episode_rows.append({
            "policy": str(policy), "trace_id": str(trace_id),
            "replicate_seed": int(seed), "episode_id": str(episode_id),
            "visual_group_id": str(group["visual_group_id"].iloc[0]),
            "duration_s": duration_s,
            "wire_bytes": float(pd.to_numeric(
                group["total_wire_bytes"], errors="raise",
            ).sum()),
            **{
                column: float(pd.to_numeric(group[column], errors="raise").sum())
                for column in component_columns
            },
            "total_energy_j": float(pd.to_numeric(group["energy_j"], errors="raise").sum()),
        })
    episode = pd.DataFrame(episode_rows)
    episode["sender_compute_energy_j"] = (
        episode["representation_energy_j"] + episode["detector_energy_j"]
        + episode["context_model_energy_j"]
    )
    episode["ground_receiver_compute_energy_j"] = episode["receiver_energy_j_charged"]
    combined = episode["sender_compute_energy_j"] + episode["ground_receiver_compute_energy_j"]
    if not np.allclose(
        combined.to_numpy(float), episode["total_energy_j"].to_numpy(float),
        rtol=1e-9, atol=1e-12,
    ):
        raise StudyGateError(
            "H2 energy_j differs from sender plus ground GPU-board compute under radio exclusion"
        )
    components = episode.groupby("policy", sort=True).agg(
        exposure_s=("duration_s", "sum"),
        sender_compute_energy_j=("sender_compute_energy_j", "sum"),
        ground_receiver_compute_energy_j=("ground_receiver_compute_energy_j", "sum"),
    )
    components["sender_compute_energy_j_per_s"] = (
        components["sender_compute_energy_j"] / components["exposure_s"]
    )
    components["ground_receiver_compute_energy_j_per_s"] = (
        components["ground_receiver_compute_energy_j"] / components["exposure_s"]
    )

    # Pairwise rate-ratio uncertainty is clustered over the registered visual
    # groups and held-out traces.  Episodes/profiles/seeds are nested within
    # those independent dimensions and contribute resources and exposure once.
    resource_cells = episode.groupby(
        ["policy", "visual_group_id", "trace_id"], sort=True,
    ).agg(
        exposure_s=("duration_s", "sum"),
        wire_bytes=("wire_bytes", "sum"),
        total_energy_j=("total_energy_j", "sum"),
    ).reset_index()

    def clustered_relative_delta(
        baseline: str, resource: str, *, seed_offset: int,
    ) -> tuple[float, float, float]:
        policies = resource_cells[
            resource_cells["policy"].isin(["proposed_full", baseline])
        ].copy()
        keys = ["visual_group_id", "trace_id"]
        proposed_cells = policies[policies["policy"].eq("proposed_full")].set_index(keys)
        reference_cells = policies[policies["policy"].eq(baseline)].set_index(keys)
        if set(proposed_cells.index) != set(reference_cells.index):
            raise StudyGateError(
                f"H2 resource cells are not paired for {baseline}"
            )
        visuals = sorted(policies["visual_group_id"].astype(str).unique())
        trace_ids = sorted(policies["trace_id"].astype(str).unique())
        full_index = pd.MultiIndex.from_product(
            [visuals, trace_ids], names=keys,
        )

        def matrix(rows: pd.DataFrame, column: str) -> np.ndarray:
            values = pd.to_numeric(
                rows[column].reindex(full_index), errors="coerce",
            ).to_numpy(float).reshape(len(visuals), len(trace_ids))
            if not np.isfinite(values).all():
                raise StudyGateError(
                    f"H2 resource design is not fully crossed for {baseline}"
                )
            return values

        proposed_resource = matrix(proposed_cells, resource)
        proposed_exposure = matrix(proposed_cells, "exposure_s")
        reference_resource = matrix(reference_cells, resource)
        reference_exposure = matrix(reference_cells, "exposure_s")

        def relative(
            p_resource: np.ndarray | float, p_exposure: np.ndarray | float,
            r_resource: np.ndarray | float, r_exposure: np.ndarray | float,
        ) -> np.ndarray:
            p_rate = np.asarray(p_resource, dtype=float) / np.asarray(p_exposure, dtype=float)
            r_rate = np.asarray(r_resource, dtype=float) / np.asarray(r_exposure, dtype=float)
            if np.any(r_rate <= 0):
                if np.all(p_rate == 0) and np.all(r_rate == 0):
                    return np.zeros_like(p_rate, dtype=float)
                raise StudyGateError(
                    f"H2 reference resource rate is nonpositive for {baseline}"
                )
            return p_rate / r_rate - 1.0

        point = float(relative(
            proposed_resource.sum(), proposed_exposure.sum(),
            reference_resource.sum(), reference_exposure.sum(),
        ))
        repetitions = int(evaluation["bootstrap_repetitions"])
        rng = np.random.default_rng(_seed(cfg) + int(seed_offset))
        draws = np.empty(repetitions, dtype=float)
        batch_size = min(512, repetitions)
        probabilities_v = np.full(len(visuals), 1.0 / len(visuals))
        probabilities_t = np.full(len(trace_ids), 1.0 / len(trace_ids))
        for start in range(0, repetitions, batch_size):
            stop = min(repetitions, start + batch_size)
            size = stop - start
            visual_weights = rng.multinomial(len(visuals), probabilities_v, size=size)
            trace_weights = rng.multinomial(len(trace_ids), probabilities_t, size=size)

            def weighted_sum(values: np.ndarray) -> np.ndarray:
                return np.einsum(
                    "bi,ij,bj->b", visual_weights, values, trace_weights,
                    optimize=True,
                )

            draws[start:stop] = relative(
                weighted_sum(proposed_resource), weighted_sum(proposed_exposure),
                weighted_sum(reference_resource), weighted_sum(reference_exposure),
            )
        low, high = np.quantile(draws, [0.025, 0.975])
        return point, float(low), float(high)

    def relative_delta(proposed: float, reference: float) -> float:
        if reference == 0:
            return 0.0 if proposed == 0 else math.inf
        return float((proposed - reference) / reference)

    rows: list[dict[str, Any]] = []
    for baseline_index, baseline in enumerate(map(str, baselines)):
        if "proposed_full" not in rates.index or baseline not in rates.index:
            raise StudyGateError(f"H2 matched-budget policy is missing: {baseline}")
        proposed = rates.loc["proposed_full"]
        reference = rates.loc[baseline]
        wire_delta = relative_delta(
            float(proposed["wire_bytes_per_s"]), float(reference["wire_bytes_per_s"]),
        )
        energy_delta = relative_delta(
            float(proposed["energy_j_per_s"]), float(reference["energy_j_per_s"]),
        )
        wire_bootstrap, wire_ci_low, wire_ci_high = clustered_relative_delta(
            baseline, "wire_bytes", seed_offset=4100 + 2 * baseline_index,
        )
        energy_bootstrap, energy_ci_low, energy_ci_high = clustered_relative_delta(
            baseline, "total_energy_j", seed_offset=4101 + 2 * baseline_index,
        )
        if not math.isclose(wire_delta, wire_bootstrap, rel_tol=1e-10, abs_tol=1e-12):
            raise StudyGateError("H2 wire-rate estimators disagree")
        if not math.isclose(energy_delta, energy_bootstrap, rel_tol=1e-10, abs_tol=1e-12):
            raise StudyGateError("H2 energy-rate estimators disagree")
        rows.append({
            "method": "proposed_full", "reference": baseline,
            "proposed_wire_bytes_per_s": float(proposed["wire_bytes_per_s"]),
            "reference_wire_bytes_per_s": float(reference["wire_bytes_per_s"]),
            "wire_relative_delta": wire_delta,
            "wire_relative_delta_ci_low": wire_ci_low,
            "wire_relative_delta_ci_high": wire_ci_high,
            "proposed_energy_j_per_s": float(proposed["energy_j_per_s"]),
            "reference_energy_j_per_s": float(reference["energy_j_per_s"]),
            "energy_relative_delta": energy_delta,
            "energy_relative_delta_ci_low": energy_ci_low,
            "energy_relative_delta_ci_high": energy_ci_high,
            "proposed_sender_compute_energy_j_per_s": float(
                components.loc["proposed_full", "sender_compute_energy_j_per_s"]
            ),
            "reference_sender_compute_energy_j_per_s": float(
                components.loc[baseline, "sender_compute_energy_j_per_s"]
            ),
            "proposed_ground_receiver_compute_energy_j_per_s": float(
                components.loc["proposed_full", "ground_receiver_compute_energy_j_per_s"]
            ),
            "reference_ground_receiver_compute_energy_j_per_s": float(
                components.loc[baseline, "ground_receiver_compute_energy_j_per_s"]
            ),
            "relative_tolerance": tolerance,
            "wire_match_pass": abs(wire_delta) <= tolerance,
            "energy_match_pass": abs(energy_delta) <= tolerance,
            "common_cap_pass": bool(proposed["eligible"] and reference["eligible"]),
            "energy_scope": str(evaluation["energy_claim_scope"]),
        })
    output = pd.DataFrame(rows)
    output["passed"] = (
        output["wire_match_pass"].astype(bool)
        & output["energy_match_pass"].astype(bool)
        & output["common_cap_pass"].astype(bool)
    )
    output["estimability_status"] = np.where(
        output["passed"].astype(bool), "MATCHED", "NOT_ESTIMABLE",
    )
    output["bootstrap_method"] = "two_way_visual_group_trace_resource_ratio_v1"
    return output


def _validation_h2_resource_audit_registration(
    pairwise: pd.DataFrame,
    resource_rates: pd.DataFrame,
    baselines: Sequence[str],
    expected_policies: Sequence[str],
) -> dict[str, Any]:
    """Gate audit completeness, never force H2 resource-effect matching.

    The pairwise ±tolerance outcomes are frozen claim-eligibility evidence.
    They are not a completion condition for unrelated hypotheses and cannot be
    tuned by padding traffic or energy after validation.
    """

    required_pairwise = {
        "reference", "wire_relative_delta", "energy_relative_delta",
        "wire_match_pass", "energy_match_pass", "common_cap_pass", "passed",
    }
    required_resource = {
        "policy", "wire_bytes_per_s", "energy_j_per_s",
        "wire_cap_bytes_per_s", "energy_cap_j_per_s",
        "wire_pass", "energy_pass", "eligible",
    }
    baseline_domain = set(map(str, baselines))
    policy_domain = set(map(str, expected_policies))
    pairwise_complete = bool(
        not required_pairwise.difference(pairwise.columns)
        and not pairwise["reference"].astype(str).duplicated().any()
        and set(pairwise["reference"].astype(str)) == baseline_domain
        and np.isfinite(pairwise[[
            "wire_relative_delta", "energy_relative_delta",
        ]].to_numpy(float)).all()
    )
    resource_complete = bool(
        not required_resource.difference(resource_rates.columns)
        and not resource_rates["policy"].astype(str).duplicated().any()
        and set(resource_rates["policy"].astype(str)) == policy_domain
        and np.isfinite(resource_rates[[
            "wire_bytes_per_s", "energy_j_per_s",
            "wire_cap_bytes_per_s", "energy_cap_j_per_s",
        ]].to_numpy(float)).all()
    )
    all_matched = bool(
        pairwise_complete and pairwise["passed"].astype(bool).all()
    )
    policies_within_caps = int(
        resource_rates["eligible"].astype(bool).sum()
    ) if resource_complete else 0
    return {
        "gate": "h2_comparator_and_resource_audits_registered",
        "passed": bool(pairwise_complete and resource_complete),
        "detail": (
            f"comparators={len(pairwise)}/{len(baseline_domain)} "
            f"policies={len(resource_rates)}/{len(policy_domain)} "
            f"pairwise_all_matched={all_matched} "
            f"policies_within_common_caps={policies_within_caps}; "
            "resource mismatches are nonblocking H2 claim-ineligibility"
        ),
        "pairwise_all_matched": all_matched,
        "policies_within_common_caps": policies_within_caps,
        "required_for": "artifact_integrity_not_h2_effect_matching",
    }


def _read_frozen_selection(protocol: Mapping[str, Any]) -> MutableMapping[str, Any]:
    decisions = protocol.get("decisions")
    if not isinstance(decisions, Mapping):
        raise StudyGateError("Frozen protocol has no decisions")
    selection = decisions.get("selection", decisions)
    if not isinstance(selection, Mapping):
        raise StudyGateError("Frozen protocol lacks a validation selection")
    return dict(selection)


def run_confirmatory_test_once(
    ctx: RunContext,
    cfg: Any,
    manifest: pd.DataFrame,
    traces: Any,
    protocol: str | Path | Mapping[str, Any],
    model_lock: str | Path,
    device: str,
    temporal_manifest: pd.DataFrame | None = None,
    on_experiment_complete: ExperimentCallback | None = None,
) -> dict[str, Any]:
    """Execute confirmatory work under a caller-owned OPEN one-time ledger.

    This function never opens or closes the ledger.  The notebook owns that
    irreversible lifecycle and passes an already ledger-bound registered test
    sample.  The runner returns every canonical artifact path for finalization.
    """

    if isinstance(protocol, Mapping):
        frozen = dict(protocol)
    else:
        protocol_path = Path(protocol).expanduser().resolve()
        if not protocol_path.is_file():
            raise StudyGateError("Confirmatory execution requires frozen_protocol.json")
        frozen = json.loads(protocol_path.read_text(encoding="utf-8"))
    selection = _read_frozen_selection(frozen)
    lock = _bind_runtime_model_lock(model_lock, cfg)
    if hash_file(model_lock) != str(selection.get("model_lock_sha256", "")):
        raise StudyGateError("Test MODEL_LOCK differs from frozen validation")
    selected_compact_model = _require_selected_compact_model(
        selection, lock["models"],
    )
    ledger_path = ctx.run_root / "test_access_ledger.json"
    if not ledger_path.is_file():
        raise StudyGateError("Notebook must authorize the one-time test before the runner")
    permit = load_untouched_test_permit(
        ledger_path,
        expected_protocol_hash=str(frozen["protocol_hash"]),
        expected_config_hash=ctx.config_hash,
        expected_code_hash=ctx.code_hash,
    )
    registered_sampling = _section(cfg, "benchmark")["analysis_sampling"]
    frozen_sampling = frozen.get("decisions", {}).get(
        "analysis_sampling", selection.get("analysis_sampling")
    )
    if frozen_sampling != registered_sampling:
        raise StudyGateError("Confirmatory analysis sampling differs from the frozen protocol")
    if not bool(selection.get("validation_gates_passed", False)):
        raise StudyGateError("Validation gates were not all passed before protocol freeze")
    if not math.isclose(
        float(selection.get("h4_matched_coverage", np.nan)),
        float(_section(cfg, "evaluation")["h4_matched_coverage"]),
        abs_tol=1e-12,
    ):
        raise StudyGateError("Frozen H4 matched coverage differs from configuration")
    if not math.isclose(
        float(selection.get("h4_high_confidence_threshold", np.nan)),
        _h4_high_confidence_threshold(cfg),
        rel_tol=0.0, abs_tol=0.0,
    ):
        raise StudyGateError(
            "Frozen H4 high-confidence threshold differs from configuration"
        )
    if dict(selection.get("representation_video_design", {})) != (
        _video_payload_arguments(cfg)
    ):
        raise StudyGateError(
            "Frozen practical-video representation design differs from configuration"
        )
    if dict(selection.get("representation_image_design", {})) != (
        _image_payload_arguments(cfg)
    ):
        raise StudyGateError(
            "Frozen image representation design differs from configuration"
        )
    if not math.isclose(
        float(selection.get("h2_matched_budget_relative_tolerance", np.nan)),
        float(_section(cfg, "evaluation")["h2_matched_budget_relative_tolerance"]),
        abs_tol=1e-12,
    ):
        raise StudyGateError("Frozen H2 matched-budget tolerance differs from configuration")
    expected_not_estimable = list(map(
        str, _section(cfg, "evaluation").get("expected_not_estimable_hypotheses", ()),
    ))
    if list(map(str, selection.get("expected_not_estimable_hypotheses", ()))) != expected_not_estimable:
        raise StudyGateError("Frozen expected-not-estimable hypothesis set changed")
    expected_not_estimable_reasons = dict(
        _section(cfg, "evaluation").get("expected_not_estimable_reasons", {})
    )
    if dict(selection.get("expected_not_estimable_reasons", {})) != expected_not_estimable_reasons:
        raise StudyGateError("Frozen expected-not-estimable hypothesis reasons changed")
    sample = _require_registered_phase_sample(
        manifest, cfg, expected_splits=("test",), test_permit=permit,
    )
    if temporal_manifest is None:
        raise StudyGateError(
            "Confirmatory H2--H4 require the separately registered MOT temporal manifest"
        )
    temporal_sample, temporal_sample_gate = _require_registered_temporal_manifest(
        temporal_manifest, cfg, split="test", test_permit=permit,
    )
    frozen_temporal_sampling = frozen.get("decisions", {}).get(
        "temporal_sampling", selection.get("temporal_sampling")
    )
    registered_temporal_sampling = _section(cfg, "benchmark").get("temporal_sampling")
    if frozen_temporal_sampling != registered_temporal_sampling:
        raise StudyGateError(
            "Confirmatory temporal sampling differs from the frozen protocol"
        )
    try:
        # Reuse the full real perception machinery but apply frozen choices and
        # frozen head.  Test never refits calibration, model, variant, or policy.
        positive_mask = sample["critical"].astype(bool)
        evaluation = _section(cfg, "evaluation")
        split_cfg = _section(cfg, "split")
        sampling_cfg = _section(cfg, "benchmark")["analysis_sampling"]
        test_component_cap = int(sampling_cfg["max_groups_per_component"]["test"])
        test_frame_cap = int(sampling_cfg["max_frames_per_group"]["test"])
        source_column = "source_sample_id" if "source_sample_id" in sample else (
            "source_uid" if "source_uid" in sample else "image_path"
        )
        component_group_counts = sample.groupby("component")["visual_group_id"].nunique()
        max_observed_frames = int(sample.groupby(
            ["component", "visual_group_id"], sort=False,
        )[source_column].nunique().max())
        profile_pairing_pass = bool(
            sample.groupby(source_column)["mission_profile"].nunique().eq(
                len(_priority_profiles(cfg))
            ).all()
        )
        sample_gate = pd.DataFrame([
            {"gate": "natural_independent_visual_groups",
             "observed": int(sample[["component", "visual_group_id"]].drop_duplicates().shape[0]),
             "required": max(int(split_cfg.get("min_test_units", 0)), int(evaluation["natural_test_units"])),
             "required_for": "completion"},
            {"gate": "balanced_det_test_dev_visual_groups",
             "observed": int(component_group_counts.get("det_test_dev", 0)),
             "required": test_component_cap, "required_for": "completion"},
            {"gate": "balanced_mot_test_dev_visual_groups",
             "observed": int(component_group_counts.get("mot_test_dev", 0)),
             "required": test_component_cap, "required_for": "completion"},
            {"gate": "maximum_source_frames_per_visual_group",
             "observed": max_observed_frames, "required": test_frame_cap,
             "required_for": "completion", "comparison": "at_most"},
            {"gate": "complete_registered_mission_profile_pairs",
             "observed": int(profile_pairing_pass), "required": 1,
             "required_for": "completion"},
            {"gate": "mission_priority_positive_visual_groups",
             "observed": int(sample.loc[positive_mask, "visual_group_id"].nunique()),
             "required": max(int(split_cfg.get("min_critical_test_events", 0)), int(evaluation["mission_priority_positive_units"])),
             "required_for": "H3_estimability"},
            {"gate": "mission_priority_negative_visual_groups",
             "observed": int(sample.loc[~positive_mask, "visual_group_id"].nunique()),
             "required": int(evaluation["mission_priority_negative_units"]),
             "required_for": "H3_estimability"},
        ])
        if "comparison" not in sample_gate:
            sample_gate["comparison"] = "at_least"
        sample_gate["comparison"] = sample_gate["comparison"].fillna("at_least")
        sample_gate["passed"] = np.where(
            sample_gate["comparison"].eq("at_most"),
            sample_gate["observed"] <= sample_gate["required"],
            sample_gate["observed"] >= sample_gate["required"],
        )
        if not sample_gate.loc[
            sample_gate["required_for"].eq("completion"), "passed"
        ].astype(bool).all():
            raise StudyGateError("Confirmatory natural sample-size gate failed")
        measurement_warmup = selection.get("measurement_warmup")
        if not isinstance(measurement_warmup, Mapping):
            raise StudyGateError("Frozen protocol lacks the development warmup receipt")
        warmup_image = Path(str(measurement_warmup.get("image_path", ""))).resolve()
        if (
            not warmup_image.is_file()
            or hash_file(warmup_image) != str(measurement_warmup.get("image_sha256", ""))
        ):
            raise StudyGateError("Frozen development warmup image changed before test")
        general_ids = set(sample["sample_id"].astype(str))
        temporal_ids = set(temporal_sample["sample_id"].astype(str))
        overlap = general_ids & temporal_ids
        if overlap:
            identity_columns = [
                "sample_id", "source_sample_id", "image_path", "split", "component",
                "sequence_id", "frame_index", "mission_profile", "gold_action",
            ]
            general_identity = sample.loc[
                sample["sample_id"].astype(str).isin(overlap), identity_columns,
            ].sort_values("sample_id", kind="mergesort").reset_index(drop=True)
            temporal_identity = temporal_sample.loc[
                temporal_sample["sample_id"].astype(str).isin(overlap), identity_columns,
            ].sort_values("sample_id", kind="mergesort").reset_index(drop=True)
            if not general_identity.equals(temporal_identity):
                raise StudyGateError(
                    "General and temporal test manifests conflict on shared samples"
                )
        detector_manifest = pd.concat(
            [sample, temporal_sample.loc[
                ~temporal_sample["sample_id"].astype(str).isin(overlap)
            ]], ignore_index=True,
        )
        detector, detector_cache, detector_revision, detector_verification = _detector_and_cache(
            ctx, cfg, detector_manifest, device, stage_prefix="test",
            warmup_spec=measurement_warmup,
        )
        if (
            detector_revision != selection.get("detector_revision")
            or detector_verification["sha256"] != selection.get("detector_checkpoint_sha256")
        ):
            raise StudyGateError("Test detector differs from frozen validation")
        if _detector_lineage_binding(detector_verification) != selection.get(
            "detector_training_lineage"
        ):
            raise StudyGateError("Test detector training lineage differs from frozen validation")
        prompt = prompt_for_mission("visdrone_urban")
        selected_model = str(selection["selected_model"])
        requested_ids = {"detector_only", selected_model, str(selection["upper_bound_model"])}
        detector_measurement_columns = [
            column for column in detector_cache.columns if column.startswith("measurement_")
        ]

        def make_test_detector_backend() -> DetectorOnlyBackend:
            backend = DetectorOnlyBackend(
                detector_revision,
                urban_policy="registered_roi_occupancy",
                urban_priority_labels=sorted({
                    label for labels in _priority_profiles(cfg).values() for label in labels
                }),
                urban_priority_profiles=_priority_profiles(cfg),
                urban_priority_roi_xyxy=_section(cfg, "benchmark")["registered_roi_xyxy"],
                urban_priority_confidence=float(
                    _section(cfg, "models").get("proposal_policy", {}).get(
                        "minimum_confidence", 0.05,
                    )
                ),
                urban_image_plane_iou=float(
                    _section(cfg, "benchmark")["min_image_plane_iou"]
                ),
            )
            backend.measurement_metadata = {
                column: detector_cache.iloc[0][column]
                for column in detector_measurement_columns
            }
            return backend

        detector_backend = make_test_detector_backend()
        vlm_warmup_request = _warmup_request(measurement_warmup)
        runtime = _section(cfg, "runtime")
        detail_parts: list[pd.DataFrame] = []
        summary_parts: list[pd.DataFrame] = []

        def execute_test_backend(
            backend: Any,
            *,
            phase_sample: pd.DataFrame = sample,
            target_details: list[pd.DataFrame] = detail_parts,
            target_summaries: list[pd.DataFrame] = summary_parts,
            cache_role: str = "general",
        ) -> None:
            measurement_metadata = _normalize_measurement_metadata(
                getattr(backend, "measurement_metadata", {})
            )
            detail, summary = experiment1_perception(
                phase_sample, [backend], scientific=True, detector_cache=detector_cache,
                ctx=ctx, shard_size=32,
                max_new_tokens=int(_section(cfg, "models").get(
                    "max_new_tokens", VISDRONE_REGISTERED_MAX_NEW_TOKENS,
                )),
                prompt=prompt,
                inference_cache_context={
                    "frozen_protocol_hash": frozen["protocol_hash"],
                    "test_ledger_sha256": permit.ledger_sha256,
                    "model_lock_sha256": hash_file(model_lock),
                    "measurement": measurement_metadata,
                    "sample_role": str(cache_role),
                },
            )
            encoded_metadata = json.dumps(
                measurement_metadata, sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            )
            detail["measurement_metadata_json"] = encoded_metadata
            summary["measurement_metadata_json"] = encoded_metadata
            detail["registered_sample_role"] = str(cache_role)
            summary["registered_sample_role"] = str(cache_role)
            target_details.append(detail)
            target_summaries.append(summary)

        execute_test_backend(detector_backend)
        del detector_backend
        for entry in lock["models"]:
            if str(entry["model_id"]) not in requested_ids:
                continue
            raw_backend = _model_backend(
                entry, device, _section(cfg, "models").get("input_size", (768, 768)),
            )
            backend = RepeatedMeasurementPerceptionBackend(
                raw_backend,
                warmup_request=vlm_warmup_request,
                warmup_image_sha256=str(measurement_warmup["image_sha256"]),
                warmup_split=str(measurement_warmup["split"]),
                warmup_prompt=prompt,
                warmup_max_new_tokens=int(_section(cfg, "models")["max_new_tokens"]),
                warmup_iterations=int(runtime["warmup_iterations"]),
                repetitions=int(runtime["steady_state_repetitions"]),
            )
            execute_test_backend(backend)
            del backend, raw_backend
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
        exp1_detail = pd.concat(detail_parts, ignore_index=True)
        exp1_summary = pd.concat(summary_parts, ignore_index=True)
        temporal_detail_parts: list[pd.DataFrame] = []
        temporal_summary_parts: list[pd.DataFrame] = []
        temporal_detector_backend = make_test_detector_backend()
        execute_test_backend(
            temporal_detector_backend,
            phase_sample=temporal_sample,
            target_details=temporal_detail_parts,
            target_summaries=temporal_summary_parts,
            cache_role="temporal_controller",
        )
        del temporal_detector_backend
        temporal_entries = [
            entry for entry in lock["models"]
            if str(entry["model_id"]) == selected_compact_model
        ]
        if len(temporal_entries) != 1:
            raise StudyGateError(
                "Frozen selected compact model is not unique in MODEL_LOCK"
            )
        temporal_raw_backend = _model_backend(
            temporal_entries[0], device,
            _section(cfg, "models").get("input_size", (768, 768)),
        )
        temporal_backend = RepeatedMeasurementPerceptionBackend(
            temporal_raw_backend,
            warmup_request=vlm_warmup_request,
            warmup_image_sha256=str(measurement_warmup["image_sha256"]),
            warmup_split=str(measurement_warmup["split"]),
            warmup_prompt=prompt,
            warmup_max_new_tokens=int(_section(cfg, "models")["max_new_tokens"]),
            warmup_iterations=int(runtime["warmup_iterations"]),
            repetitions=int(runtime["steady_state_repetitions"]),
        )
        execute_test_backend(
            temporal_backend,
            phase_sample=temporal_sample,
            target_details=temporal_detail_parts,
            target_summaries=temporal_summary_parts,
            cache_role="temporal_controller",
        )
        del temporal_backend, temporal_raw_backend
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
        temporal_exp1_detail = pd.concat(temporal_detail_parts, ignore_index=True)
        temporal_exp1_summary = pd.concat(temporal_summary_parts, ignore_index=True)
        _require_exact_temporal_experiment1_routes(
            temporal_exp1_detail,
            temporal_exp1_summary,
            selected_compact_model=selected_compact_model,
        )
        temporal_compact_rows = temporal_exp1_detail[
            temporal_exp1_detail["model_id"].astype(str).eq(
                selected_compact_model
            )
        ].copy()
        temporal_detector_rows = temporal_exp1_detail[
            temporal_exp1_detail["model_id"].astype(str).eq("detector_only")
        ].copy()
        temporal_selected_rows = (
            temporal_detector_rows.copy()
            if selected_model == "detector_only"
            else temporal_compact_rows.copy()
        )
        temporal_policy_manifest = _policy_manifest(
            temporal_sample, detector_cache, temporal_selected_rows,
            temporal_detector_rows, selection,
        )
        _write_frame(ctx, "test_exp1_perception_detail.csv.gz", exp1_detail)
        _write_frame(ctx, "test_exp1_perception_summary.csv.gz", exp1_summary)
        _write_frame(ctx, "test_temporal_exp1_perception_detail.csv.gz", temporal_exp1_detail)
        _write_frame(ctx, "test_temporal_exp1_perception_summary.csv.gz", temporal_exp1_summary)
        _emit_experiment_complete(on_experiment_complete, 1, "test", {
            "exp1_summary": exp1_summary,
            "temporal_exp1_summary": temporal_exp1_summary,
        })
        selected_rows = exp1_detail[exp1_detail["model_id"].astype(str).eq(selected_model)].copy()
        detector_rows = exp1_detail[exp1_detail["model_id"].eq("detector_only")].copy()
        policy_manifest = _policy_manifest(sample, detector_cache, selected_rows, detector_rows, selection)
        selected_semantic = calibrated_semantic_packet_cache(
            selected_rows, float(selection["calibration"][selected_model]["temperature"]),
            source_model=selected_model, authoritative_manifest=sample,
        )
        native_cache, _ = _resolve_native_cache(
            True, ctx=ctx, cfg=cfg, manifest=sample, semantic_cache=selected_semantic,
            detector=detector, detector_revision=detector_revision,
            semantic_source_model=selected_model, device=device, phase="test",
        )
        feature_binding = selection.get("feature_head")
        if not isinstance(feature_binding, Mapping):
            raise StudyGateError("Frozen protocol lacks the exact L4 feature-head receipt")
        frozen_head_lock = Path(str(feature_binding.get("lock_path", ""))).resolve()
        if (
            not frozen_head_lock.is_file()
            or hash_file(frozen_head_lock) != str(feature_binding.get("lock_sha256", ""))
        ):
            raise StudyGateError("Frozen L4 feature-head lock is missing or changed")
        feature_cache, _, _ = _resolve_feature_cache(
            True, ctx=ctx, cfg=cfg, phase_manifest=sample,
            device=device, phase="test", frozen_head_lock=frozen_head_lock,
        )
        exp2_frontier = experiment2_representation_frontier_sharded(
            sample, ctx, scientific=True,
            mtu_bytes=int(_section(cfg, "payloads").get("mtu_bytes", 1200)),
            semantic_packet_cache=selected_semantic, native_receiver_cache=native_cache,
            registered_feature_cache=feature_cache, semantic_source_model=selected_model,
            **_image_payload_arguments(cfg),
            **_video_payload_arguments(cfg),
            stage_prefix="test_02_representations",
        )
        detector_semantic = calibrated_semantic_packet_cache(
            detector_rows, float(selection["calibration"]["detector_only"]["temperature"]),
            source_model="detector_only", authoritative_manifest=sample,
        )
        detector_native, _ = _resolve_native_cache(
            True, ctx=ctx, cfg=cfg, manifest=sample, semantic_cache=detector_semantic,
            detector=detector, detector_revision=detector_revision,
            semantic_source_model="detector_only", device=device, phase="test",
        )
        if _native_receiver_fusion_binding(native_cache) != selection.get(
            "native_receiver_fusion"
        ) or _native_receiver_fusion_binding(detector_native) != selection.get(
            "detector_native_receiver_fusion"
        ):
            raise StudyGateError("Test native-receiver fusion policy differs from validation")
        detector_frontier = experiment2_representation_frontier_sharded(
            sample, ctx, scientific=True,
            semantic_packet_cache=detector_semantic, native_receiver_cache=detector_native,
            registered_feature_cache=feature_cache, semantic_source_model="detector_only",
            **_image_payload_arguments(cfg),
            **_video_payload_arguments(cfg),
            stage_prefix="test_02_detector_representations",
        )
        detector_candidates = detector_frontier[
            detector_frontier["mode"].isin(["L0_HEARTBEAT", "L1_ALERT", "L2_SEMANTICS", "L3_SEMANTICS_ROI"])
        ].copy()
        detector_candidates["variant"] = detector_candidates["variant"].astype(str) + "__detector_only"
        exp2_frontier = pd.concat([exp2_frontier, detector_candidates], ignore_index=True)
        exp2_frontier = add_experiment2_total_costs(
            exp2_frontier,
            policy_manifest[["sample_id", "detector_latency_ms", "detector_energy_j", "context_model_latency_ms", "context_model_energy_j"]],
            require_measured_costs=True,
        )
        temporal_selected_semantic = calibrated_semantic_packet_cache(
            temporal_selected_rows,
            float(selection["calibration"][selected_model]["temperature"]),
            source_model=selected_model, authoritative_manifest=temporal_sample,
        )
        temporal_native, _ = _resolve_native_cache(
            True, ctx=ctx, cfg=cfg, manifest=temporal_sample,
            semantic_cache=temporal_selected_semantic, detector=detector,
            detector_revision=detector_revision, semantic_source_model=selected_model,
            device=device, phase="test_temporal",
        )
        temporal_feature_cache, _, temporal_feature_lock = _resolve_feature_cache(
            True, ctx=ctx, cfg=cfg, phase_manifest=temporal_sample,
            device=device, phase="test_temporal", frozen_head_lock=frozen_head_lock,
        )
        if temporal_feature_lock != frozen_head_lock:
            raise StudyGateError("Temporal test L4 head differs from the frozen head")
        temporal_exp2_frontier = experiment2_representation_frontier_sharded(
            temporal_sample, ctx, scientific=True,
            mtu_bytes=int(_section(cfg, "payloads").get("mtu_bytes", 1200)),
            semantic_packet_cache=temporal_selected_semantic,
            native_receiver_cache=temporal_native,
            registered_feature_cache=temporal_feature_cache,
            semantic_source_model=selected_model,
            **_image_payload_arguments(cfg),
            **_video_payload_arguments(cfg),
            stage_prefix="test_temporal_02_representations",
        )
        temporal_detector_semantic = calibrated_semantic_packet_cache(
            temporal_detector_rows,
            float(selection["calibration"]["detector_only"]["temperature"]),
            source_model="detector_only", authoritative_manifest=temporal_sample,
        )
        temporal_detector_native, _ = _resolve_native_cache(
            True, ctx=ctx, cfg=cfg, manifest=temporal_sample,
            semantic_cache=temporal_detector_semantic, detector=detector,
            detector_revision=detector_revision,
            semantic_source_model="detector_only", device=device,
            phase="test_temporal",
        )
        if _native_receiver_fusion_binding(temporal_native) != selection.get(
            "temporal_native_receiver_fusion"
        ) or _native_receiver_fusion_binding(temporal_detector_native) != selection.get(
            "temporal_detector_native_receiver_fusion"
        ):
            raise StudyGateError(
                "Temporal test native-receiver fusion policy differs from validation"
            )
        temporal_detector_frontier = experiment2_representation_frontier_sharded(
            temporal_sample, ctx, scientific=True,
            mtu_bytes=int(_section(cfg, "payloads").get("mtu_bytes", 1200)),
            semantic_packet_cache=temporal_detector_semantic,
            native_receiver_cache=temporal_detector_native,
            registered_feature_cache=temporal_feature_cache,
            semantic_source_model="detector_only",
            **_image_payload_arguments(cfg),
            **_video_payload_arguments(cfg),
            stage_prefix="test_temporal_02_detector_representations",
        )
        temporal_detector_candidates = temporal_detector_frontier[
            temporal_detector_frontier["mode"].isin([
                "L0_HEARTBEAT", "L1_ALERT", "L2_SEMANTICS", "L3_SEMANTICS_ROI",
            ])
        ].copy()
        temporal_detector_candidates["variant"] = (
            temporal_detector_candidates["variant"].astype(str) + "__detector_only"
        )
        temporal_exp2_frontier = pd.concat(
            [temporal_exp2_frontier, temporal_detector_candidates], ignore_index=True,
        )
        temporal_exp2_frontier = add_experiment2_total_costs(
            temporal_exp2_frontier,
            temporal_policy_manifest[[
                "sample_id", "detector_latency_ms", "detector_energy_j",
                "context_model_latency_ms", "context_model_energy_j",
            ]], require_measured_costs=True,
        )
        _write_frame(ctx, "test_exp2_representation_frontier.csv.gz", exp2_frontier)
        _write_frame(ctx, "test_temporal_exp2_representation_frontier.csv.gz", temporal_exp2_frontier)
        _emit_experiment_complete(on_experiment_complete, 2, "test", {
            "exp2_frontier": exp2_frontier,
        })
        profiles_path = Path(str(selection.get("candidate_profiles_path", ""))).resolve()
        if (
            not profiles_path.is_file()
            or hash_file(profiles_path) != str(selection.get("candidate_profiles_sha256", ""))
        ):
            raise StudyGateError("Frozen validation causal candidate profiles changed")
        profiles = pd.read_csv(profiles_path, compression="infer")
        policy_spec_path = Path(str(selection.get("policy_spec_path", ""))).resolve()
        if (
            not policy_spec_path.is_file()
            or hash_file(policy_spec_path) != str(selection.get("policy_spec_sha256", ""))
        ):
            raise StudyGateError("Frozen validation policy specification changed")
        policy_spec = json.loads(policy_spec_path.read_text(encoding="utf-8"))
        network = _section(cfg, "network")
        static_manifest = select_experiment3_manifest(
            policy_manifest, int(network["static_max_groups_per_dataset"]), _seed(cfg),
            int(network["static_max_observations_per_group"]),
        )
        static_manifest = _linearize_static_stream(static_manifest)
        test_static_gate = _static_sample_gate(
            static_manifest, cfg, phase="test",
        )
        static_ids = set(static_manifest["sample_id"].astype(str))
        frozen_variant_path = Path(str(selection.get("exp3_variant_spec", ""))).resolve()
        if (
            not frozen_variant_path.is_file()
            or hash_file(frozen_variant_path) != str(selection.get("exp3_variant_spec_sha256", ""))
        ):
            raise StudyGateError("Frozen Experiment 3 variant specification changed")
        frozen_variant_spec = pd.read_csv(frozen_variant_path, compression="infer")
        static_candidates, _ = select_experiment3_candidate_variants(
            exp2_frontier[exp2_frontier["sample_id"].astype(str).isin(static_ids)],
            int(network["static_max_variants_per_mode_source"]),
            frozen_spec=frozen_variant_spec,
        )
        exp3_summary, exp3_detail = experiment3_static_transport_partitioned(
            static_candidates, static_manifest, ctx, registered_seeds=_seeds(cfg),
            rates_mbps=network["static_rate_mbps"], rtts_ms=network["rtt_ms"],
            losses=network["loss"], loss_models=network["static_loss_models"],
            burst_recovery_probability=float(network["static_burst_recovery_probability"]),
            burst_slot_ms=float(network["static_burst_slot_ms"]),
            jitters_ms=network["static_jitter_ms"], queue_kb=network["queue_kb"],
            radio_tx_j_per_byte=float(_section(cfg, "hardware").get("radio_tx_j_per_byte", 0.0)),
            radio_rx_j_per_byte=float(_section(cfg, "hardware").get("radio_rx_j_per_byte", 0.0)),
            require_perception_costs=True, require_registered_loss_models=True,
        )
        _write_frame(ctx, "test_exp3_static_partition_index.csv.gz", exp3_detail)
        _write_frame(ctx, "test_exp3_static_summary.csv.gz", exp3_summary)
        _emit_experiment_complete(on_experiment_complete, 3, "test", {
            "exp3_summary": exp3_summary,
        })
        test_traces = _load_traces_for_split(traces, ctx=ctx, split="test")
        temporal_policy_manifest, test_episode_audit = _rebase_registered_episodes(
            temporal_policy_manifest,
            test_traces,
            state_expiry_grace_us=int(network["temporal_state_expiry_grace_us"]),
        )
        policies = _controller_policies(cfg, policy_spec, temporal_exp2_frontier)
        policy_partition_index = run_paired_policy_trials_sharded(
            temporal_policy_manifest, temporal_exp2_frontier, test_traces,
            policies, _seeds(cfg), profiles, ctx, scientific=True,
            semantic_acceptance_threshold=_h4_high_confidence_threshold(cfg),
            return_partition_index=True, **_radio_energy_arguments(cfg),
        )
        test_cartesian_coverage = _dynamic_partition_coverage_audit(
            policy_partition_index, temporal_policy_manifest, test_traces,
            policies, _seeds(cfg),
        )
        reduction = reduce_policy_trial_partitions(
            policy_partition_index,
            _results(ctx) / "test_policy_reduction",
            h4_target_coverage=float(evaluation["h4_matched_coverage"]),
            semantic_acceptance_threshold=_h4_high_confidence_threshold(cfg),
            energy_scope=str(evaluation["energy_claim_scope"]),
        )
        compact = dict(reduction.frames)
        _write_frame(ctx, "test_controller_behavior_summary.csv.gz", compact["controller_summary"])
        _emit_experiment_complete(on_experiment_complete, 4, "test", {
            "exp4_summary": compact["resource_summary"],
            "exp4_controller_summary": compact["controller_summary"],
        })
        _write_frame(ctx, "test_exp5_risk_uncertainty_summary.csv.gz", compact["exp5_summary"])
        _emit_experiment_complete(on_experiment_complete, 5, "test", {
            "exp5_summary": compact["exp5_summary"],
        })
        restricted_raw_partitions: list[Path] = []
        for label, partition_frame in (
            ("Experiment 3", exp3_detail),
            ("dynamic policy", policy_partition_index),
        ):
            required_partition = {"artifact_path", "artifact_sha256"}
            if missing := required_partition.difference(partition_frame.columns):
                raise StudyGateError(
                    f"{label} partition index lacks restricted-evidence fields: {sorted(missing)}"
                )
            for row in partition_frame.itertuples(index=False):
                path = Path(str(row.artifact_path)).expanduser().resolve()
                if not path.is_file() or hash_file(path) != str(row.artifact_sha256):
                    raise StudyGateError(
                        f"{label} restricted partition changed before receipt registration"
                    )
                restricted_raw_partitions.append(path)
        if len(set(restricted_raw_partitions)) != len(restricted_raw_partitions):
            raise StudyGateError("Restricted raw partition paths are duplicated")
        pair_cells = compact["policy_pair_cells"]
        proposed_cells = pair_cells[
            pair_cells["policy"].astype(str).eq("proposed_full")
        ]
        paired_scene_trace_cells = int(len(proposed_cells))
        paired_trace_realizations = int(
            proposed_cells["trace_id"].astype(str).nunique()
        )
        paired_gate = pd.DataFrame([
            {
                "gate": "fully_paired_scene_trace_cells",
                "observed": paired_scene_trace_cells,
                "required": int(evaluation["fully_paired_scene_trace_cells"]),
                "passed": paired_scene_trace_cells >= int(
                    evaluation["fully_paired_scene_trace_cells"]
                ),
                "required_for": "H2_estimability",
            },
            {
                "gate": "held_out_trace_realizations",
                "observed": paired_trace_realizations,
                "required": int(_section(cfg, "network")["held_out_trace_realizations"]),
                "passed": paired_trace_realizations >= int(
                    _section(cfg, "network")["held_out_trace_realizations"]
                ),
                "required_for": "completion",
            },
        ])
        exp4_summary = compact["controller_summary"]
        exp5_summary = compact["exp5_summary"]
        exp6_detail, exp6_summary = experiment6_registered_campaign(
            seeds=_seeds(cfg), scientific=True,
        )
        exp6_strategy_detail, exp6_strategy_summary = experiment6_state_strategy_comparison(
            seeds=_seeds(cfg), scientific=True,
        )
        _write_frame(ctx, "test_receiver_robustness_detail.csv.gz", exp6_detail)
        _write_frame(ctx, "test_receiver_robustness_summary.csv.gz", exp6_summary)
        _write_frame(ctx, "test_receiver_state_strategy_detail.csv.gz", exp6_strategy_detail)
        _write_frame(ctx, "test_receiver_state_strategy_summary.csv.gz", exp6_strategy_summary)
        _emit_experiment_complete(on_experiment_complete, 6, "test", {
            "exp6_campaign_summary": exp6_summary,
            "exp6_state_strategy_summary": exp6_strategy_summary,
        })
        frames = {
            "exp1_detail": exp1_detail, "exp1_summary": exp1_summary,
            "temporal_exp1_detail": temporal_exp1_detail,
            "temporal_exp1_summary": temporal_exp1_summary,
            "exp2_frontier": exp2_frontier, "exp3_detail": exp3_detail,
            "temporal_exp2_frontier": temporal_exp2_frontier,
            "exp3_summary": exp3_summary,
            "exp4_summary": exp4_summary,
            "exp5_summary": exp5_summary, "exp6_detail": exp6_detail,
            "exp6_summary": exp6_summary,
            "exp6_state_strategy_detail": exp6_strategy_detail,
            "exp6_state_strategy_summary": exp6_strategy_summary,
        }
        reports = recompute_hypotheses_from_frozen_results(
            {
                **frames, "policy_spec": policy_spec, "manifest": policy_manifest,
                "temporal_manifest": temporal_policy_manifest,
                "policy_reduction": compact,
            }, cfg,
            cluster_columns=("visual_group_id", "trace_id"),
            bootstrap_repetitions=int(_section(cfg, "evaluation").get("bootstrap_repetitions", 10_000)),
        )
        test_budget = compact["resource_rate_summary"].copy()
        test_budget["wire_cap_bytes_per_s"] = 1.05 * float(
            evaluation["primary_wire_budget_bytes_per_s"]
        )
        test_budget["energy_cap_j_per_s"] = 1.05 * float(
            evaluation["primary_energy_budget_j_per_s"]
        )
        test_budget["wire_pass"] = (
            test_budget["wire_bytes_per_s"] <= test_budget["wire_cap_bytes_per_s"]
        )
        test_budget["energy_pass"] = (
            test_budget["energy_j_per_s"] <= test_budget["energy_cap_j_per_s"]
        )
        test_budget["eligible"] = test_budget["wire_pass"] & test_budget["energy_pass"]
        observed_policies = set(compact["resource_summary"]["policy"].astype(str))
        expected_policies = {policy.name for policy in policies}
        gate_table = pd.concat([
            sample_gate[["gate", "observed", "required", "passed", "required_for"]],
            test_static_gate[["gate", "observed", "required", "passed", "required_for"]],
            paired_gate[["gate", "observed", "required", "passed", "required_for"]],
            pd.DataFrame([
                {"gate": "frozen_validation_gates", "observed": 1,
                 "required": 1, "passed": bool(selection["validation_gates_passed"]),
                 "required_for": "completion"},
                {"gate": "exact_confirmatory_policy_coverage",
                 "observed": len(observed_policies), "required": len(expected_policies),
                 "passed": observed_policies == expected_policies,
                 "required_for": "completion"},
                {"gate": "exact_confirmatory_cartesian_coverage",
                 "observed": int(test_cartesian_coverage["passed"].astype(bool).sum()),
                 "required": int(len(test_cartesian_coverage)),
                 "passed": bool(test_cartesian_coverage["passed"].astype(bool).all()),
                 "required_for": "completion"},
                {"gate": "exp6_registered_cases",
                 "observed": int(exp6_detail["case"].nunique()), "required": 13,
                 "passed": int(exp6_detail["case"].nunique()) == 13,
                 "required_for": "completion"},
                {"gate": "exp6_registered_state_strategies",
                 "observed": int(exp6_strategy_detail["strategy"].nunique()), "required": 3,
                 "passed": int(exp6_strategy_detail["strategy"].nunique()) == 3,
                 "required_for": "completion"},
            ]),
        ], ignore_index=True)
        if not gate_table.loc[
            gate_table["required_for"].eq("completion"), "passed"
        ].astype(bool).all():
            raise StudyGateError("Confirmatory completion gates failed")
        registered_status = reports["hypothesis_status"][
            reports["hypothesis_status"]["hypothesis"].astype(str).isin(
                ["H1", "H2", "H3", "H4"]
            )
        ]
        completion = pd.DataFrame([{
            "execution_status": "COMPLETE",
            "empirical_complete": True,
            "all_required_artifacts_written": True,
            "hypotheses_supported": int(
                registered_status["claim_supported"].astype(bool).sum()
            ),
            "hypotheses_registered": int(len(registered_status)),
            "hypotheses_evaluated": int(
                registered_status["execution_status"].astype(str).eq("COMPLETE").sum()
            ),
            "hypotheses_not_estimable": int(
                registered_status["execution_status"].astype(str).eq(
                    "NOT_ESTIMABLE"
                ).sum()
            ),
            "h5_status": H5_NOT_APPLICABLE_STATUS,
            "energy_claim_scope": str(evaluation["energy_claim_scope"]),
            "runner_version": RUNNER_VERSION,
            "model_lock_sha256": hash_file(model_lock),
        }])
        dataset_input = policy_manifest.copy()
        dataset_input["critical_visual_group_id"] = dataset_input["visual_group_id"].where(
            dataset_input["critical"].astype(bool)
        )
        dataset_summary = (
            dataset_input.groupby(
                ["dataset", "component", "mission_profile"], as_index=False,
                dropna=False,
            ).agg(
                observations=("sample_id", "nunique"),
                independent_visual_groups=("visual_group_id", "nunique"),
                source_frames=("source_sample_id", "nunique"),
                mission_priority_positive_groups=("critical_visual_group_id", "nunique"),
            )
        )
        canonical_frames = {
            "test_exp1_perception_detail.csv.gz": exp1_detail,
            "test_exp1_perception_summary.csv.gz": exp1_summary,
            "test_temporal_exp1_perception_detail.csv.gz": temporal_exp1_detail,
            "test_temporal_exp1_perception_summary.csv.gz": temporal_exp1_summary,
            "test_exp2_representation_frontier.csv.gz": exp2_frontier,
            "test_temporal_exp2_representation_frontier.csv.gz": temporal_exp2_frontier,
            "test_exp3_static_partition_index.csv.gz": exp3_detail,
            "test_exp3_static_summary.csv.gz": exp3_summary,
            "test_controller_behavior_summary.csv.gz": exp4_summary,
            "test_exp5_risk_uncertainty_summary.csv.gz": exp5_summary,
            "test_budget_compliance.csv.gz": test_budget,
            "test_h2_matched_budget.csv.gz": reports["h2_matched_budget"],
            "confirmatory_gate_table.csv.gz": gate_table,
            "test_h1_fixed_grid_group_aggregates.csv.gz": reports[
                "h1_fixed_grid_aggregates"
            ],
            "test_h1_fixed_grid_coverage_audit.csv.gz": reports[
                "h1_fixed_grid_audit"
            ],
            "h1_semantic_noninferiority.csv.gz": reports["h1_report"],
            "h2_confirmatory_report.csv.gz": reports["h2_report"],
            "h3_mission_awareness.csv.gz": reports["h3_report"],
            "h4_uncertainty_escalation.csv.gz": reports["h4_report"],
            "test_h4_matched_coverage_gate.csv.gz": reports["h4_coverage_audit"],
            "test_receiver_robustness_detail.csv.gz": exp6_detail,
            "test_receiver_robustness_summary.csv.gz": exp6_summary,
            "test_receiver_state_strategy_detail.csv.gz": exp6_strategy_detail,
            "test_receiver_state_strategy_summary.csv.gz": exp6_strategy_summary,
            "hypothesis_status.csv.gz": reports["hypothesis_status"],
            "final_completion_status.csv.gz": completion,
            "test_episode_trace_coverage.csv.gz": test_episode_audit,
            "test_dynamic_cartesian_coverage_audit.csv.gz": test_cartesian_coverage,
            "test_sample_cluster_manifest.csv.gz": policy_manifest[[
                "sample_id", "source_sample_id", "source_uid", "visual_group_id",
                "mission_profile", "sequence_id", "frame_index", "component",
                "split", "critical", "gold_action",
            ]].drop_duplicates("sample_id"),
            "test_temporal_cluster_manifest.csv.gz": temporal_policy_manifest[[
                "sample_id", "source_sample_id", "source_uid", "visual_group_id",
                "mission_profile", "sequence_id", "frame_index", "component",
                "split", "critical", "gold_action", "temporal_observation_index",
                "temporal_episode_id", "temporal_selection_sha256",
            ]].drop_duplicates("sample_id"),
            "test_temporal_sample_gate.csv.gz": temporal_sample_gate,
            "test_dataset_summary.csv.gz": dataset_summary,
        }
        result_artifacts = [
            _write_frame(ctx, filename, frame)
            for filename, frame in canonical_frames.items()
        ]
        runtime_identity = {
            "schema_version": TEST_RUNTIME_IDENTITY_SCHEMA,
            "status": "VERIFIED_AND_EXECUTED",
            "phase": "test",
            "runner_version": RUNNER_VERSION,
            "frozen_protocol_hash": str(frozen["protocol_hash"]),
            "test_ledger_sha256": str(permit.ledger_sha256),
            "model_lock": {
                "path": str(Path(model_lock).expanduser().resolve()),
                "sha256": hash_file(model_lock),
            },
        }
        runtime_identity_path = atomic_write_json(
            _results(ctx) / "test_inference_runtime_identity.json",
            runtime_identity,
        )
        result_artifacts.append(runtime_identity_path)
        partition_index_path = atomic_write_text(
            _results(ctx) / "test_policy_partition_index.csv",
            policy_partition_index.to_csv(index=False, lineterminator="\n"),
        )
        result_artifacts.append(partition_index_path)
        result_artifacts.extend(
            path for _, path in sorted(reduction.artifacts.items())
        )
        residency = Path(str(detector_verification["residency_receipt_path"]))
        if (
            not residency.is_file()
            or hash_file(residency) != str(detector_verification["residency_receipt_sha256"])
        ):
            raise StudyGateError("Observed detector residency receipt changed")
        result_artifacts.insert(0, residency)
        return {
            **frames, **reports, "result_artifacts": tuple(result_artifacts),
            "policy_partition_index": policy_partition_index,
            "policy_reduction": compact,
            "policy_reduction_receipt": reduction.receipt_path,
            "restricted_raw_partition_artifacts": tuple(restricted_raw_partitions),
            "confirmatory_gate_table": gate_table,
            "test_budget_compliance": test_budget,
            "final_completion_status": completion,
            "runtime_identity": runtime_identity,
            "runtime_identity_path": runtime_identity_path,
        }
    except Exception:
        # The notebook owns ledger finalization.  Propagate the exact error so
        # it can close OPEN as BLOCKED or FAILED without an ambiguous rerun.
        raise


def _load_h1_fixed_grid_aggregates(
    index: pd.DataFrame,
    required_streams: Mapping[str, str],
    cluster_manifest: pd.DataFrame,
    cfg: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verify every raw partition and stream only frozen H1 representations.

    Raw static output can contain tens of millions of rows.  This reader never
    concatenates those rows.  It verifies the complete partition index and
    every artifact checksum, filters each bounded CSV chunk to the registered
    L2/L3/practical-L5 streams, and accumulates only group/profile totals.  The
    returned artifact is therefore the final fixed-grid estimand input, not a
    lossy re-interpretation of a compact condition summary.
    """

    required_index = {
        "seed", "shard", "stage", "artifact_path", "artifact_sha256",
        "row_count", "design_hash",
    }
    if missing := required_index.difference(index.columns):
        raise StudyGateError(
            f"Experiment 3 partition index is incomplete: {sorted(missing)}"
        )
    seeds = _seeds(cfg)
    assert_experiment3_partition_index_coverage(
        index, seeds, verify_artifacts=True,
    )
    if not isinstance(cluster_manifest, pd.DataFrame) or cluster_manifest.empty:
        raise StudyGateError("H1 requires the frozen sample-to-cluster manifest")
    mapping_columns = ["sample_id", "visual_group_id", "mission_profile"]
    if missing := set(mapping_columns).difference(cluster_manifest.columns):
        raise StudyGateError(f"H1 cluster manifest lacks: {sorted(missing)}")
    mapping = cluster_manifest[mapping_columns].copy()
    mapping["sample_id"] = mapping["sample_id"].astype(str)
    if mapping["sample_id"].duplicated().any():
        raise StudyGateError("H1 cluster manifest has duplicate sample_id rows")

    raw_required = {
        "observation_id", "mode", "variant", "source_model", "replicate_seed",
        "rate_mbps", "rtt_ms", "loss", "loss_model",
        "burst_recovery_probability", "burst_slot_ms", "jitter_ms", "queue_kb",
        "correct_timely_action", "total_wire_bytes", "latency_ms",
        "deadline_us", "capture_us", "energy_j",
    }
    optional_cluster = {"visual_group_id", "mission_profile"}
    stream_pairs = {(str(mode), str(variant)) for mode, variant in required_streams.items()}
    if len(stream_pairs) != len(required_streams):
        raise StudyGateError("Frozen H1 stream specification contains duplicates")
    network = _section(cfg, "network")
    expected_conditions = set(itertools.product(
        map(float, network["static_rate_mbps"]),
        map(float, network["rtt_ms"]),
        map(float, network["loss"]),
        map(str, network["static_loss_models"]),
        (float(network["static_burst_recovery_probability"]),),
        (float(network["static_burst_slot_ms"]),),
        map(float, network["static_jitter_ms"]),
        map(int, network["queue_kb"]),
    ))
    partials: list[pd.DataFrame] = []
    stream_seeds: dict[tuple[str, str, str], set[int]] = {}
    stream_conditions: dict[tuple[str, str, str], set[tuple[Any, ...]]] = {}
    stream_observation_counts: dict[tuple[str, str, str], dict[str, int]] = {}
    verified_partition_rows = 0
    for partition in index.sort_values(["seed", "shard"]).itertuples(index=False):
        path = Path(str(partition.artifact_path)).expanduser().resolve()
        if not path.is_file() or hash_file(path) != str(partition.artifact_sha256):
            raise StudyGateError(f"Experiment 3 raw partition is missing or changed: {path}")
        header = set(pd.read_csv(path, compression="infer", nrows=0).columns)
        if missing := raw_required.difference(header):
            raise StudyGateError(
                f"Experiment 3 raw partition lacks H1 columns {sorted(missing)}: {path}"
            )
        partition_rows = 0
        usecols = raw_required | (optional_cluster & header)
        for chunk in pd.read_csv(
            path, compression="infer", usecols=lambda column: column in usecols,
            chunksize=100_000,
        ):
            partition_rows += len(chunk)
            mode = chunk["mode"].astype(str)
            variant = chunk["variant"].astype(str)
            keep = pd.Series(False, index=chunk.index)
            for stream_mode, stream_variant in stream_pairs:
                keep |= mode.eq(stream_mode) & variant.eq(stream_variant)
            selected = chunk.loc[keep].copy()
            if selected.empty:
                continue
            selected["observation_id"] = selected["observation_id"].astype(str)
            if not optional_cluster.issubset(selected.columns):
                selected = selected.merge(
                    mapping.rename(columns={"sample_id": "observation_id"}),
                    on="observation_id", validate="many_to_one",
                )
            if selected[["visual_group_id", "mission_profile"]].isna().any().any():
                raise StudyGateError("H1 selected rows lack frozen cluster/profile labels")
            selected["replicate_seed"] = pd.to_numeric(
                selected["replicate_seed"], errors="raise",
            ).astype(int)
            for stream_values, stream_frame in selected.groupby(
                ["mode", "variant", "source_model"], sort=False, dropna=False,
            ):
                stream_key = tuple(map(str, stream_values))
                stream_seeds.setdefault(stream_key, set()).update(
                    map(int, stream_frame["replicate_seed"].unique())
                )
                normalized_conditions = zip(
                    pd.to_numeric(stream_frame["rate_mbps"], errors="raise").astype(float),
                    pd.to_numeric(stream_frame["rtt_ms"], errors="raise").astype(float),
                    pd.to_numeric(stream_frame["loss"], errors="raise").astype(float),
                    stream_frame["loss_model"].astype(str),
                    pd.to_numeric(
                        stream_frame["burst_recovery_probability"], errors="raise",
                    ).astype(float),
                    pd.to_numeric(stream_frame["burst_slot_ms"], errors="raise").astype(float),
                    pd.to_numeric(stream_frame["jitter_ms"], errors="raise").astype(float),
                    pd.to_numeric(stream_frame["queue_kb"], errors="raise").astype(int),
                )
                stream_conditions.setdefault(stream_key, set()).update(normalized_conditions)
                counts = stream_observation_counts.setdefault(stream_key, {})
                for observation_id, count in stream_frame["observation_id"].value_counts().items():
                    counts[str(observation_id)] = counts.get(str(observation_id), 0) + int(count)

            selected["utility_sum"] = selected["correct_timely_action"].astype(bool).astype(float)
            selected["wire_sum"] = pd.to_numeric(
                selected["total_wire_bytes"], errors="raise",
            ).astype(float)
            deadline_budget_ms = (
                pd.to_numeric(selected["deadline_us"], errors="raise")
                - pd.to_numeric(selected["capture_us"], errors="raise")
            ) / 1000.0
            latency = pd.to_numeric(selected["latency_ms"], errors="coerce")
            selected["latency_sum"] = np.minimum(
                latency.fillna(np.inf).to_numpy(float),
                deadline_budget_ms.to_numpy(float),
            )
            selected["energy_sum"] = pd.to_numeric(
                selected["energy_j"], errors="raise",
            ).astype(float)
            partials.append(
                selected.groupby(
                    ["visual_group_id", "mission_profile", "mode", "variant", "source_model"],
                    as_index=False, sort=False, dropna=False,
                ).agg(
                    utility_sum=("utility_sum", "sum"),
                    wire_sum=("wire_sum", "sum"),
                    latency_sum=("latency_sum", "sum"),
                    energy_sum=("energy_sum", "sum"),
                    fixed_grid_rows=("observation_id", "size"),
                )
            )
        if partition_rows != int(partition.row_count):
            raise StudyGateError(
                f"Experiment 3 partition row count changed: {path} "
                f"({partition_rows} != {int(partition.row_count)})"
            )
        verified_partition_rows += partition_rows
    if not partials:
        raise StudyGateError("Frozen Experiment 3 partitions contain no registered H1 streams")
    combined = pd.concat(partials, ignore_index=True)
    group_columns = [
        "visual_group_id", "mission_profile", "mode", "variant", "source_model",
    ]
    aggregate = combined.groupby(
        group_columns, as_index=False, sort=True, dropna=False,
    ).agg(
        utility_sum=("utility_sum", "sum"), wire_sum=("wire_sum", "sum"),
        latency_sum=("latency_sum", "sum"), energy_sum=("energy_sum", "sum"),
        fixed_grid_rows=("fixed_grid_rows", "sum"),
    )
    for output, total in (
        ("correct_timely_action", "utility_sum"),
        ("total_wire_bytes", "wire_sum"),
        ("deadline_censored_latency_ms", "latency_sum"),
        ("energy_j", "energy_sum"),
    ):
        aggregate[output] = aggregate[total] / aggregate["fixed_grid_rows"]
    aggregate["fixed_grid_condition_count"] = int(len(expected_conditions))
    aggregate["stochastic_repeats"] = int(len(seeds))
    aggregate["estimand"] = "visual_group_equal_over_registered_fixed_grid_v1"

    audit_rows: list[dict[str, Any]] = []
    reference_observations: set[str] | None = None
    expected_rows_per_observation = len(expected_conditions) * len(seeds)
    for stream_key in sorted(stream_seeds):
        mode, variant, source_model = stream_key
        observations = stream_observation_counts[stream_key]
        observation_set = set(observations)
        if reference_observations is None:
            reference_observations = observation_set
        paired_observations = observation_set == reference_observations
        exact_observation_rows = all(
            count == expected_rows_per_observation for count in observations.values()
        )
        seeds_pass = stream_seeds[stream_key] == set(seeds)
        conditions_pass = stream_conditions[stream_key] == expected_conditions
        audit_rows.append({
            "mode": mode, "variant": variant, "source_model": source_model,
            "observations": len(observation_set),
            "fixed_grid_conditions": len(stream_conditions[stream_key]),
            "required_fixed_grid_conditions": len(expected_conditions),
            "stochastic_repeats": len(stream_seeds[stream_key]),
            "required_stochastic_repeats": len(seeds),
            "exact_rows_per_observation": exact_observation_rows,
            "paired_observations": paired_observations,
            "conditions_pass": conditions_pass,
            "seeds_pass": seeds_pass,
            "passed": bool(
                exact_observation_rows and paired_observations
                and conditions_pass and seeds_pass
            ),
            "inference_scope": "visual_groups_over_registered_fixed_grid",
        })
    audit = pd.DataFrame(audit_rows)
    observed_pairs = set(zip(audit["mode"].astype(str), audit["variant"].astype(str)))
    if (
        observed_pairs != stream_pairs
        or len(audit) != len(stream_pairs)
        or not audit["passed"].astype(bool).all()
    ):
        raise StudyGateError("H1 frozen fixed-grid stream coverage is incomplete")
    audit["verified_raw_partition_rows"] = int(verified_partition_rows)
    return aggregate, audit


def _group_only_paired_bootstrap(
    group_strata: pd.DataFrame,
    method: str,
    reference: str,
    *,
    endpoint: str,
    repetitions: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    """Bootstrap independent visual groups after fixed-grid/profile averaging."""

    required = {"visual_group_id", "mission_profile", "policy", endpoint}
    if missing := required.difference(group_strata.columns):
        raise StudyGateError(f"H1 group bootstrap lacks: {sorted(missing)}")
    subset = group_strata[group_strata["policy"].isin([method, reference])]
    pivot = subset.pivot(
        index=["visual_group_id", "mission_profile"],
        columns="policy", values=endpoint,
    )
    if method not in pivot or reference not in pivot or pivot[[method, reference]].isna().any().any():
        raise StudyGateError("H1 fixed-grid outcomes are not paired by group/profile")
    delta = (pivot[method].astype(float) - pivot[reference].astype(float)).rename("delta")
    group_delta = delta.groupby(level="visual_group_id").mean()
    values = group_delta.to_numpy(float)
    if not len(values):
        raise StudyGateError("H1 has no independent visual groups")
    rng = np.random.default_rng(int(seed))
    draws = values[rng.integers(0, len(values), size=(int(repetitions), len(values)))].mean(axis=1)
    alpha = 1.0 - float(confidence)
    low, high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "estimate": float(values.mean()), "ci_low": float(low), "ci_high": float(high),
        "repetitions": int(repetitions), "visual_clusters": int(len(values)),
        "method": "visual_group_bootstrap_over_registered_fixed_grid_v1",
    }


def _recompute_hypotheses_from_frozen_results_legacy(
    report_frames: Mapping[str, Any],
    cfg: Any,
    cluster_columns: Sequence[str],
    bootstrap_repetitions: int,
) -> dict[str, pd.DataFrame]:
    """Recompute H1--H4 only from frozen detailed results.

    ``cluster_columns`` must name the registered visual and trace clusters; it
    is checked explicitly even though the underlying tested estimator uses its
    canonical column names.
    """

    if tuple(cluster_columns) != ("visual_group_id", "trace_id"):
        raise StudyGateError(
            "Paper inference is registered for visual_group_id and trace_id clustering"
        )
    exp4 = report_frames.get("exp4_policy")
    if not isinstance(exp4, pd.DataFrame) or exp4.empty:
        raise StudyGateError("Hypothesis recomputation requires detailed Experiment 4 trials")
    assert_paired_design(exp4)
    policy_spec = report_frames.get("policy_spec")
    if not isinstance(policy_spec, Mapping):
        raise StudyGateError("Hypothesis recomputation requires the frozen policy specification")
    evaluation = _section(cfg, "evaluation")
    repetitions = int(bootstrap_repetitions)
    if repetitions <= 0:
        raise ValueError("bootstrap_repetitions must be positive")
    alpha = 0.05
    baselines = tuple(map(str, policy_spec.get("required_h2_baselines", ())))
    if not baselines:
        raise StudyGateError("Frozen H2 baseline family is empty")
    if str(policy_spec.get("h2_family_definition", "")) != (
        "primary_controller_baselines_plus_best_fixed_validation_v1"
    ):
        raise StudyGateError("Frozen H2 primary comparator family is unregistered")
    observed = set(exp4["policy"].astype(str))
    if missing := sorted({"proposed_full", *baselines} - observed):
        raise StudyGateError(f"Confirmatory policy family is incomplete: {missing}")
    confidence = 1.0 - alpha / len(baselines)
    h2_effects = [
        two_way_clustered_paired_bootstrap(
            exp4, "proposed_full", baseline,
            repetitions=repetitions, seed=_seed(cfg), confidence=confidence,
        ) for baseline in baselines
    ]
    adjusted = holm_adjust({
        f"{effect.method}_vs_{effect.reference}": effect.p_two_sided
        for effect in h2_effects
    })
    h2 = hypothesis_report(
        h2_effects, adjusted, margin=float(evaluation.get("primary_margin", 0.03)),
    )
    budgets = budget_compliance(
        exp4, float(evaluation["primary_wire_budget_bytes_per_s"]),
        float(evaluation["primary_energy_budget_j_per_s"]),
    ).set_index("policy")
    h2_matched_budget = _h2_pairwise_budget_match(exp4, baselines, cfg)
    h2_match_lookup = h2_matched_budget.set_index("reference")
    h2["matched_budget_pass"] = [
        bool(h2_match_lookup.loc[str(baseline), "passed"])
        for baseline in h2["reference"].astype(str)
    ]
    validation_unmatched = tuple(map(
        str, policy_spec.get("validation_h2_unmatched_baselines", ()),
    ))
    unknown_validation = sorted(set(validation_unmatched) - set(baselines))
    if unknown_validation:
        raise StudyGateError(
            f"Frozen validation H2 unmatched family has unknown policies: {unknown_validation}"
        )
    test_unmatched = tuple(
        h2_matched_budget.loc[
            ~h2_matched_budget["passed"].astype(bool), "reference",
        ].astype(str)
    )
    h2_estimable = not validation_unmatched and not test_unmatched
    h2_not_estimable_reason = (
        "" if h2_estimable else
        "primary H2 family is not pairwise matched within the frozen 5% "
        f"wire/combined-compute tolerance; validation={list(validation_unmatched)} "
        f"test={list(test_unmatched)}"
    )
    h2["claim_supported"] = (
        h2["exceeds_margin"].astype(bool)
        & h2["matched_budget_pass"].astype(bool)
        & bool(h2_estimable)
    )
    h2["execution_status"] = "COMPLETE" if h2_estimable else "NOT_ESTIMABLE"
    h2["not_estimable_reason"] = h2_not_estimable_reason
    h2["validation_unmatched_baselines"] = json.dumps(
        list(validation_unmatched), separators=(",", ":"),
    )
    h2["test_unmatched_baselines"] = json.dumps(
        list(test_unmatched), separators=(",", ":"),
    )
    h2["energy_scope"] = str(evaluation["energy_claim_scope"])

    critical = exp4[exp4["critical"].astype(bool)]
    h3_gate = _h3_estimability_gate(exp4, cfg)
    positive_groups = int(h3_gate["positive_visual_groups"])
    required_positive_groups = int(h3_gate["required_positive_visual_groups"])
    proposed_far = float(h3_gate["proposed_false_alert_rate"])
    reference_far = float(h3_gate["reference_false_alert_rate"])
    negative_groups = int(h3_gate["negative_visual_groups"])
    required_negative_groups = int(h3_gate["required_negative_visual_groups"])
    proposed_far_upper = float(
        h3_gate["proposed_false_alert_cluster_bootstrap_upper_95"]
    )
    reference_far_upper = float(
        h3_gate["reference_false_alert_cluster_bootstrap_upper_95"]
    )
    h3_estimable = bool(h3_gate["estimable"])
    if (
        "H3" in set(map(str, evaluation.get("expected_not_estimable_hypotheses", ())))
        and h3_estimable
    ):
        raise StudyGateError(
            "The frozen bounded VisDrone design preregisters H3 as NOT_ESTIMABLE"
        )
    h3_effect = (
        two_way_clustered_paired_bootstrap(
            critical, "channel_only_hysteresis", "proposed_full",
            endpoint="critical_miss", repetitions=repetitions, seed=_seed(cfg),
        ) if h3_estimable else None
    )

    target_coverage = float(evaluation["h4_matched_coverage"])
    h4_trials, h4_audit = matched_coverage_semantic_errors(
        exp4, ("proposed_full", "ablate_evidence_escalation"), target_coverage,
    )
    h4_pass = bool(h4_audit["passed"].astype(bool).all())
    h4_effect = (
        two_way_clustered_paired_bootstrap(
            h4_trials, "ablate_evidence_escalation", "proposed_full",
            endpoint="matched_coverage_semantic_error", repetitions=repetitions,
            seed=_seed(cfg),
        ) if h4_pass else None
    )
    secondary = holm_adjust({
        "H3": h3_effect.p_two_sided if h3_effect is not None else 1.0,
        "H4": h4_effect.p_two_sided if h4_effect is not None else 1.0,
    })
    false_limit = float(evaluation.get("false_alert_limit", 0.05))
    h3 = pd.DataFrame([{
        "hypothesis": "H3", "endpoint": "mission_priority_critical_miss_reduction",
        "execution_status": (
            "COMPLETE" if h3_estimable else "NOT_ESTIMABLE"
        ),
        "not_estimable_reason": (
            "" if h3_estimable else
            "VisDrone test sample does not meet the registered independent "
            "mission-priority positive/negative group minima"
        ),
        "estimate": h3_effect.estimate if h3_effect else np.nan,
        "critical_miss_reduction": h3_effect.estimate if h3_effect else np.nan,
        "ci_low": h3_effect.ci_low if h3_effect else np.nan,
        "ci_high": h3_effect.ci_high if h3_effect else np.nan,
        "holm_adjusted_p": secondary["H3"],
        "positive_visual_groups": positive_groups,
        "required_positive_visual_groups": required_positive_groups,
        "negative_visual_groups": negative_groups,
        "required_negative_visual_groups": required_negative_groups,
        "proposed_false_alert_rate": proposed_far,
        "reference_false_alert_rate": reference_far,
        "false_alert_estimand": str(h3_gate["estimand"]),
        "false_alert_bootstrap_method": str(h3_gate["bootstrap_method"]),
        "false_alert_bootstrap_repetitions": int(h3_gate["bootstrap_repetitions"]),
        "proposed_false_alert_cluster_bootstrap_upper_95": proposed_far_upper,
        "reference_false_alert_cluster_bootstrap_upper_95": reference_far_upper,
        "proposed_negative_decision_exposures": int(
            h3_gate["proposed_negative_decision_exposures"]
        ),
        "reference_negative_decision_exposures": int(
            h3_gate["reference_negative_decision_exposures"]
        ),
        "proposed_negative_episode_cells": int(
            h3_gate["proposed_negative_episode_cells"]
        ),
        "reference_negative_episode_cells": int(
            h3_gate["reference_negative_episode_cells"]
        ),
        "negative_trace_clusters": int(h3_gate["negative_trace_clusters"]),
        "false_alert_gate_pass": bool(h3_gate["false_alert_gate_pass"]),
        "false_alert_limit": false_limit,
        "claim_supported": bool(
            h3_estimable and h3_effect is not None
            and h3_effect.ci_low > 0 and secondary["H3"] <= alpha
            and bool(h3_gate["false_alert_gate_pass"])
        ),
    }])
    h4_lookup = h4_audit.set_index("policy") if len(h4_audit) else pd.DataFrame()
    proposed_risk = float(h4_lookup.loc["proposed_full", "group_equal_selective_risk"]) if h4_pass else np.nan
    reference_risk = float(h4_lookup.loc["ablate_evidence_escalation", "group_equal_selective_risk"]) if h4_pass else np.nan
    h4 = pd.DataFrame([{
        "hypothesis": "H4", "endpoint": "matched_coverage_semantic_error_reduction",
        "claim_scope": (
            "composed_detector_verifier_pipeline_selective_action_error_"
            "not_free_form_vlm_semantic_generation"
        ),
        "semantic_packet_authorship": "deterministic_host_composer",
        "detector_proposal_conditioned": True,
        "free_form_vlm_packet_authorship_claim_eligible": False,
        "execution_status": "COMPLETE",
        "estimate": h4_effect.estimate if h4_effect else np.nan,
        "high_confidence_error_reduction": h4_effect.estimate if h4_effect else np.nan,
        "ci_low": h4_effect.ci_low if h4_effect else np.nan,
        "ci_high": h4_effect.ci_high if h4_effect else np.nan,
        "holm_adjusted_p": secondary["H4"], "coverage_gate_pass": h4_pass,
        "proposed_selective_risk": proposed_risk,
        "reference_selective_risk": reference_risk,
        "claim_supported": bool(
            h4_pass and h4_effect is not None and h4_effect.ci_low > 0
            and secondary["H4"] <= alpha and proposed_risk < reference_risk
        ),
    }])

    # H1 prefers the finalized, checksum-bound per-group fixed-grid aggregate.
    # During confirmatory execution only, that small artifact is constructed by
    # streaming and verifying every raw static partition; report recomputation
    # supplies the finalized aggregate/audit and never reopens the raw grid.
    exp3_index = report_frames.get("exp3_detail")
    if not isinstance(exp3_index, pd.DataFrame) or exp3_index.empty:
        raise StudyGateError("H1 requires the frozen Experiment 3 partition index")
    fixed_lookup = {str(item["mode"]): item for item in policy_spec["fixed"]}
    image = fixed_lookup.get("L5_IMAGE_OR_CLIP")
    if image is None:
        raise StudyGateError("H1 requires a frozen practical L5 comparator")
    h1_streams = {
        mode: str(fixed_lookup[mode]["variant"])
        for mode in ("L2_SEMANTICS", "L3_SEMANTICS_ROI")
        if mode in fixed_lookup
    }
    h1_streams["L5_IMAGE_OR_CLIP"] = str(image["variant"])
    existing_h1_aggregate = report_frames.get("h1_fixed_grid_aggregates")
    existing_h1_audit = report_frames.get("h1_fixed_grid_audit")
    if isinstance(existing_h1_aggregate, pd.DataFrame) and not existing_h1_aggregate.empty:
        h1_aggregate = existing_h1_aggregate.copy()
        if not isinstance(existing_h1_audit, pd.DataFrame) or existing_h1_audit.empty:
            raise StudyGateError(
                "Finalized H1 fixed-grid aggregate requires its coverage audit"
            )
        h1_grid_audit = existing_h1_audit.copy()
    else:
        cluster_manifest = report_frames.get("manifest")
        if not isinstance(cluster_manifest, pd.DataFrame):
            raise StudyGateError("H1 requires the frozen sample-to-cluster manifest")
        h1_aggregate, h1_grid_audit = _load_h1_fixed_grid_aggregates(
            exp3_index, h1_streams, cluster_manifest, cfg,
        )
    required_h1_aggregate = {
        "visual_group_id", "mission_profile", "mode", "variant", "source_model",
        "fixed_grid_rows", "fixed_grid_condition_count", "stochastic_repeats",
        "correct_timely_action", "total_wire_bytes",
        "deadline_censored_latency_ms", "energy_j", "estimand",
    }
    if missing := required_h1_aggregate.difference(h1_aggregate.columns):
        raise StudyGateError(f"H1 fixed-grid aggregate lacks: {sorted(missing)}")
    if not h1_aggregate["estimand"].astype(str).eq(
        "visual_group_equal_over_registered_fixed_grid_v1"
    ).all():
        raise StudyGateError("H1 fixed-grid aggregate uses an unregistered estimand")
    required_h1_audit = {
        "mode", "variant", "fixed_grid_conditions",
        "required_fixed_grid_conditions", "stochastic_repeats",
        "required_stochastic_repeats", "passed", "inference_scope",
    }
    if missing := required_h1_audit.difference(h1_grid_audit.columns):
        raise StudyGateError(f"H1 fixed-grid coverage audit lacks: {sorted(missing)}")
    if (
        not h1_grid_audit["passed"].astype(bool).all()
        or not h1_grid_audit["inference_scope"].astype(str).eq(
            "visual_groups_over_registered_fixed_grid"
        ).all()
    ):
        raise StudyGateError("H1 finalized fixed-grid coverage audit failed")
    if set(pd.to_numeric(h1_aggregate["stochastic_repeats"], errors="raise").astype(int)) != {
        len(_seeds(cfg))
    }:
        raise StudyGateError("H1 fixed-grid aggregate has the wrong stochastic repeat count")
    h1_rows: list[dict[str, Any]] = []
    for mode in ("L2_SEMANTICS", "L3_SEMANTICS_ROI"):
        semantic = fixed_lookup.get(mode)
        if semantic is None:
            continue
        subset = h1_aggregate[
            (
                h1_aggregate["mode"].eq(mode)
                & h1_aggregate["variant"].eq(semantic["variant"])
            )
            | (
                h1_aggregate["mode"].eq("L5_IMAGE_OR_CLIP")
                & h1_aggregate["variant"].eq(image["variant"])
            )
        ].copy()
        if subset.empty:
            continue
        observed_h1_groups = int(subset["visual_group_id"].astype(str).nunique())
        required_h1_groups = int(_section(cfg, "network")["static_min_independent_groups"])
        if observed_h1_groups < required_h1_groups:
            raise StudyGateError(
                "H1 static partitions lack the registered independent visual groups: "
                f"{observed_h1_groups}/{required_h1_groups}"
            )
        subset["policy"] = np.where(subset["mode"].eq(mode), "semantic", "image")
        row_coverage = subset.pivot(
            index=["visual_group_id", "mission_profile"],
            columns="policy", values="fixed_grid_rows",
        )
        if (
            {"semantic", "image"}.difference(row_coverage.columns)
            or row_coverage[["semantic", "image"]].isna().any().any()
            or not row_coverage["semantic"].eq(row_coverage["image"]).all()
        ):
            raise StudyGateError("H1 semantic/image fixed-grid rows are not exactly paired")
        joint_confidence = 1.0 - 0.05 / 8.0
        utility = _group_only_paired_bootstrap(
            subset, "semantic", "image", endpoint="correct_timely_action",
            repetitions=repetitions, seed=_seed(cfg), confidence=joint_confidence,
        )
        bytes_saved = _group_only_paired_bootstrap(
            subset, "image", "semantic", endpoint="total_wire_bytes",
            repetitions=repetitions, seed=_seed(cfg) + 1,
            confidence=joint_confidence,
        )
        latency_saved = _group_only_paired_bootstrap(
            subset, "image", "semantic", endpoint="deadline_censored_latency_ms",
            repetitions=repetitions, seed=_seed(cfg) + 2,
            confidence=joint_confidence,
        )
        energy_saved = _group_only_paired_bootstrap(
            subset, "image", "semantic", endpoint="energy_j",
            repetitions=repetitions, seed=_seed(cfg) + 3,
            confidence=joint_confidence,
        )
        margin = float(evaluation.get("primary_margin", 0.03))
        utility_pass = utility["ci_low"] > -margin
        bytes_pass = bytes_saved["ci_low"] > 0
        latency_pass = latency_saved["ci_low"] > 0
        energy_pass = energy_saved["ci_low"] > 0
        h1_rows.append({
            "hypothesis": "H1", "semantic_mode": mode,
            "execution_status": "COMPLETE",
            "claim_scope": (
                "end_to_end_transmission_utility_and_resource_effects_for_"
                "composed_semantic_packets"
            ),
            "semantic_packet_authorship": "deterministic_host_composer",
            "detector_proposal_conditioned": True,
            "free_form_vlm_packet_authorship_claim_eligible": False,
            "semantic_variant": semantic["variant"],
            "image_variant": image["variant"],
            "utility_estimate": utility["estimate"],
            "utility_difference": utility["estimate"],
            "utility_ci_low": utility["ci_low"],
            "utility_ci_high": utility["ci_high"],
            "noninferiority_margin": -margin,
            "utility_noninferiority_pass": utility_pass,
            "wire_bytes_saved_estimate": bytes_saved["estimate"],
            "wire_bytes_saved_ci_low": bytes_saved["ci_low"],
            "wire_bytes_saved_ci_high": bytes_saved["ci_high"],
            "wire_bytes_savings_pass": bytes_pass,
            "deadline_censored_latency_ms_saved_estimate": latency_saved["estimate"],
            "deadline_censored_latency_ms_saved_ci_low": latency_saved["ci_low"],
            "deadline_censored_latency_ms_saved_ci_high": latency_saved["ci_high"],
            "latency_savings_pass": latency_pass,
            "gpu_board_energy_j_saved_estimate": energy_saved["estimate"],
            "gpu_board_energy_j_saved_ci_low": energy_saved["ci_low"],
            "gpu_board_energy_j_saved_ci_high": energy_saved["ci_high"],
            "energy_savings_pass": energy_pass,
            "energy_scope": str(evaluation.get("energy_claim_scope", "")),
            "independent_visual_groups": observed_h1_groups,
            "required_independent_visual_groups": required_h1_groups,
            "fixed_grid_conditions": int(subset["fixed_grid_condition_count"].iloc[0]),
            "stochastic_repeats": int(subset["stochastic_repeats"].iloc[0]),
            "inference_scope": "visual_groups_over_registered_fixed_grid",
            "bootstrap_method": str(utility["method"]),
            "claim_supported": bool(
                utility_pass and bytes_pass and latency_pass and energy_pass
            ),
        })
    if not h1_rows:
        raise StudyGateError("Frozen static results contain no H1 comparison")
    status = pd.DataFrame([
        {"hypothesis": "H1", "execution_status": "COMPLETE",
         "claim_supported": bool(pd.DataFrame(h1_rows)["claim_supported"].all())},
        {"hypothesis": "H2", "execution_status": (
            "COMPLETE" if h2_estimable else "NOT_ESTIMABLE"
        ),
         "claim_supported": bool(h2["claim_supported"].all())},
        {"hypothesis": "H3", "execution_status": str(h3.iloc[0]["execution_status"]),
         "claim_supported": bool(h3["claim_supported"].all())},
        {"hypothesis": "H4", "execution_status": "COMPLETE",
         "claim_supported": bool(h4["claim_supported"].all())},
        {
            "hypothesis": "H5",
            "claim_supported": False,
            "execution_status": H5_NOT_APPLICABLE_STATUS,
        },
    ])
    return {
        "h1_report": pd.DataFrame(h1_rows),
        "h1_fixed_grid_aggregates": h1_aggregate,
        "h1_fixed_grid_audit": h1_grid_audit,
        "h2_report": h2,
        "h2_matched_budget": h2_matched_budget,
        "h3_report": h3, "h4_report": h4, "h4_coverage_audit": h4_audit,
        "h4_matched_coverage_trials": h4_trials,
        "hypothesis_status": status,
    }


def _compact_policy_pseudo_trials(
    reduction: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build a bounded compatibility frame from cluster/episode aggregates.

    This frame is not a trial artifact and is never written.  It exists only
    so the established H1 path and matched-resource implementation can be
    reused while all confirmatory dynamic evidence remains compact.
    """

    episodes = reduction.get("resource_episode_cells")
    pair_cells = reduction.get("policy_pair_cells")
    if not isinstance(episodes, pd.DataFrame) or episodes.empty:
        raise StudyGateError("Compact recomputation lacks resource_episode_cells")
    if not isinstance(pair_cells, pd.DataFrame) or pair_cells.empty:
        raise StudyGateError("Compact recomputation lacks policy_pair_cells")
    required_episode = {
        "policy", "trace_id", "replicate_seed", "episode_id", "visual_group_id",
        "duration_s", "wire_bytes", "total_energy_j", "representation_energy_j",
        "detector_energy_j", "context_model_energy_j", "receiver_energy_j_charged",
    }
    if missing := required_episode.difference(episodes.columns):
        raise StudyGateError(f"Compact resource episodes lack: {sorted(missing)}")
    rates = pair_cells[[
        "policy", "visual_group_id", "trace_id", "correct_timely_rate",
    ]].copy()
    base = episodes.merge(
        rates, on=["policy", "visual_group_id", "trace_id"],
        validate="many_to_one",
    )
    rows: list[pd.DataFrame] = []
    for half in (0, 1):
        part = base.copy()
        part["sample_id"] = (
            part["episode_id"].astype(str) + f"::compact-half-{half}"
        )
        part["mission_profile"] = "registered_temporal"
        half_duration_us = np.maximum(
            1,
            np.rint(pd.to_numeric(part["duration_s"], errors="raise") * 500_000)
            .astype(np.int64),
        )
        part["capture_us"] = half * half_duration_us
        part["total_wire_bytes"] = pd.to_numeric(
            part["wire_bytes"], errors="raise",
        ) / 2.0
        part["energy_j"] = pd.to_numeric(
            part["total_energy_j"], errors="raise",
        ) / 2.0
        for column in (
            "representation_energy_j", "detector_energy_j",
            "context_model_energy_j", "receiver_energy_j_charged",
        ):
            part[column] = pd.to_numeric(part[column], errors="raise") / 2.0
        part["correct_timely_action"] = pd.to_numeric(
            part["correct_timely_rate"], errors="raise",
        )
        part["critical"] = False
        part["false_alert"] = False
        part["critical_miss"] = False
        part["selected_mode"] = "L2_SEMANTICS"
        part["calibrated_correctness_probability"] = 1.0
        part["correct_action"] = True
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def _compact_dynamic_hypothesis_reports(
    reduction: Mapping[str, pd.DataFrame],
    policy_spec: Mapping[str, Any],
    temporal_manifest: pd.DataFrame,
    cfg: Any,
    *,
    repetitions: int,
) -> dict[str, pd.DataFrame]:
    """Recompute H2--H4 from bounded visual-group/trace artifacts."""

    pair_cells = reduction.get("policy_pair_cells")
    h3_cells = reduction.get("h3_negative_episode_cells")
    h4_cells = reduction.get("h4_matched_coverage_cells")
    h4_audit = reduction.get("h4_matched_coverage_audit")
    if any(
        not isinstance(frame, pd.DataFrame) or frame.empty
        for frame in (pair_cells, h4_cells, h4_audit)
    ):
        raise StudyGateError("Compact hypothesis artifacts are missing or empty")
    if not isinstance(h3_cells, pd.DataFrame):
        raise StudyGateError("Compact H3 episode artifact is missing")
    if not isinstance(temporal_manifest, pd.DataFrame) or temporal_manifest.empty:
        raise StudyGateError("Compact hypothesis recomputation lacks the temporal manifest")
    assert isinstance(pair_cells, pd.DataFrame)
    assert isinstance(h3_cells, pd.DataFrame)
    assert isinstance(h4_cells, pd.DataFrame)
    assert isinstance(h4_audit, pd.DataFrame)
    evaluation = _section(cfg, "evaluation")
    expected_not_estimable = set(map(
        str, evaluation.get("expected_not_estimable_hypotheses", ()),
    ))
    registered_reasons = evaluation.get("expected_not_estimable_reasons", {})
    if not isinstance(registered_reasons, Mapping):
        raise StudyGateError("Registered expected-NOT_ESTIMABLE reasons are invalid")

    def registered_not_estimable_reason(hypothesis: str) -> str:
        if hypothesis not in expected_not_estimable:
            raise StudyGateError(
                f"{hypothesis} became structurally NOT_ESTIMABLE without being preregistered"
            )
        reason = str(registered_reasons.get(hypothesis, "")).strip()
        if not reason:
            raise StudyGateError(
                f"{hypothesis} lacks its frozen expected-NOT_ESTIMABLE reason"
            )
        return reason
    baselines = tuple(map(str, policy_spec.get("required_h2_baselines", ())))
    if not baselines:
        raise StudyGateError("Frozen H2 baseline family is empty")
    observed = set(pair_cells["policy"].astype(str))
    if missing := sorted({"proposed_full", *baselines} - observed):
        raise StudyGateError(f"Compact H2 family is incomplete: {missing}")
    confidence = 1.0 - 0.05 / len(baselines)
    h2_effects = [
        compact_two_way_clustered_paired_bootstrap(
            pair_cells, "proposed_full", baseline,
            endpoint="correct_timely_rate", repetitions=int(repetitions),
            seed=_seed(cfg), confidence=confidence,
        )
        for baseline in baselines
    ]
    adjusted = holm_adjust({
        f"{effect.method}_vs_{effect.reference}": effect.p_two_sided
        for effect in h2_effects
    })
    h2 = hypothesis_report(
        h2_effects, adjusted, margin=float(evaluation.get("primary_margin", 0.03)),
    )
    pseudo = _compact_policy_pseudo_trials(reduction)
    h2_matched_budget = _h2_pairwise_budget_match(pseudo, baselines, cfg)
    match_lookup = h2_matched_budget.set_index("reference")
    h2["matched_budget_pass"] = [
        bool(match_lookup.loc[baseline, "passed"])
        for baseline in h2["reference"].astype(str)
    ]
    validation_unmatched = tuple(map(
        str, policy_spec.get("validation_h2_unmatched_baselines", ()),
    ))
    test_unmatched = tuple(
        h2_matched_budget.loc[
            ~h2_matched_budget["passed"].astype(bool), "reference",
        ].astype(str)
    )
    proposed = pair_cells[
        pair_cells["policy"].astype(str).eq("proposed_full")
    ]
    temporal_groups = int(proposed["visual_group_id"].astype(str).nunique())
    temporal_cells = int(proposed[["visual_group_id", "trace_id"]].drop_duplicates().shape[0])
    temporal_spec = registered_temporal_sampling_spec(
        _section(cfg, "benchmark"), "test",
    )
    required_groups = int(temporal_spec["minimum_temporal_groups"])
    required_cells = int(temporal_spec["minimum_group_trace_cells"])
    required_observations = int(temporal_spec["observations_per_sequence"])
    temporal_required = {
        "visual_group_id", "mission_profile", "temporal_observation_index",
    }
    if missing := temporal_required.difference(temporal_manifest.columns):
        raise StudyGateError(
            f"Temporal manifest lacks compact hypothesis fields: {sorted(missing)}"
        )
    manifest_groups = int(temporal_manifest["visual_group_id"].astype(str).nunique())
    if manifest_groups != temporal_groups:
        raise StudyGateError(
            "Compact policy cells and frozen temporal manifest disagree on visual groups"
        )
    observation_counts = temporal_manifest.groupby(
        ["visual_group_id", "mission_profile"], sort=True,
    )["temporal_observation_index"].nunique()
    observed_observations = int(observation_counts.min()) if len(observation_counts) else 0
    temporal_sampling_pass = bool(
        len(observation_counts)
        and observation_counts.eq(required_observations).all()
        and manifest_groups == int(temporal_spec["sequences_per_split"])
    )
    structural_pass = bool(
        temporal_sampling_pass
        and temporal_groups >= required_groups
        and temporal_cells >= required_cells
    )
    if structural_pass and "H2" in expected_not_estimable:
        raise StudyGateError(
            "Frozen H2 expected-NOT_ESTIMABLE registration contradicts observed temporal minima"
        )
    matched_pass = not validation_unmatched and not test_unmatched
    h2_estimable = bool(structural_pass and matched_pass)
    h2_reasons: list[str] = []
    if not structural_pass:
        h2_reasons.append(registered_not_estimable_reason("H2"))
    if not matched_pass:
        h2_reasons.append(
            "primary H2 family is not pairwise matched within the frozen resource tolerance"
        )
    h2_reason = "; ".join(h2_reasons)
    h2["claim_supported"] = (
        h2["exceeds_margin"].astype(bool)
        & h2["matched_budget_pass"].astype(bool)
        & h2_estimable
    )
    h2["execution_status"] = "COMPLETE" if h2_estimable else "NOT_ESTIMABLE"
    h2["not_estimable_reason"] = h2_reason
    if not h2_estimable:
        # Preserve the bounded descriptive computation under unmistakably
        # underpowered names, while leaving every registered confirmatory
        # inferential field blank.  This prevents a finite CI/p-value from
        # leaking into a structural NOT_ESTIMABLE claim.
        descriptive_columns = (
            "estimate", "ci_low", "ci_high", "p_two_sided", "holm_p",
            "exceeds_margin", "bootstrap_distribution_hash",
        )
        for column in descriptive_columns:
            if column in h2:
                h2[f"descriptive_underpowered_{column}"] = h2[column]
        for column in (
            "estimate", "ci_low", "ci_high", "p_two_sided", "holm_p",
        ):
            if column in h2:
                h2[column] = np.nan
        if "exceeds_margin" in h2:
            h2["exceeds_margin"] = np.nan
        if "bootstrap_distribution_hash" in h2:
            h2["bootstrap_distribution_hash"] = None
        h2["descriptive_inference_status"] = (
            "UNDERPOWERED_NOT_CONFIRMATORY"
        )
    h2["temporal_visual_groups"] = temporal_groups
    h2["required_temporal_visual_groups"] = required_groups
    h2["temporal_group_trace_cells"] = temporal_cells
    h2["required_temporal_group_trace_cells"] = required_cells
    h2["temporal_observations_per_group"] = observed_observations
    h2["required_temporal_observations_per_group"] = required_observations
    h2["temporal_sampling_pass"] = temporal_sampling_pass
    h2["validation_unmatched_baselines"] = json.dumps(
        list(validation_unmatched), separators=(",", ":"),
    )
    h2["test_unmatched_baselines"] = json.dumps(
        list(test_unmatched), separators=(",", ":"),
    )
    h2["energy_scope"] = str(evaluation["energy_claim_scope"])

    proposed_negative = h3_cells[
        h3_cells["policy"].astype(str).eq("proposed_full")
    ]
    reference_negative = h3_cells[
        h3_cells["policy"].astype(str).eq("channel_only_hysteresis")
    ]
    positive_groups = int(
        proposed.loc[
            pd.to_numeric(proposed["critical_trials"], errors="raise").gt(0),
            "visual_group_id",
        ].astype(str).nunique()
    )
    negative_groups = int(proposed_negative["visual_group_id"].astype(str).nunique())
    required_positive = max(
        int(_section(cfg, "split").get("min_critical_test_events", 0)),
        int(evaluation["mission_priority_positive_units"]),
    )
    required_negative = int(evaluation["mission_priority_negative_units"])

    def compact_false_alert_audit(frame: pd.DataFrame, *, seed_offset: int) -> dict[str, Any]:
        diagnostic = frame.rename(columns={"false_alert_rate": "false_alert"}).copy()
        return _two_way_cluster_false_alert_upper(
            diagnostic,
            repetitions=int(repetitions),
            seed=_seed(cfg) + int(seed_offset),
        )

    proposed_far_audit = compact_false_alert_audit(
        proposed_negative, seed_offset=701,
    )
    reference_far_audit = compact_false_alert_audit(
        reference_negative, seed_offset=907,
    )
    false_alert_limit = float(evaluation.get("false_alert_limit", 0.05))
    false_alert_gate_pass = bool(
        np.isfinite(float(proposed_far_audit["upper_95"]))
        and np.isfinite(float(reference_far_audit["upper_95"]))
        and float(proposed_far_audit["upper_95"]) <= false_alert_limit
        and float(reference_far_audit["upper_95"]) <= false_alert_limit
    )

    h3_structural_failure = bool(
        positive_groups < required_positive or negative_groups < required_negative
    )
    if not h3_structural_failure:
        if "H3" in expected_not_estimable:
            raise StudyGateError(
                "Frozen H3 expected-NOT_ESTIMABLE registration contradicts observed group minima"
            )
        raise StudyGateError(
            "Compact H3 estimator is unavailable even though registered group minima were met"
        )
    h3_reason = registered_not_estimable_reason("H3")
    h3 = pd.DataFrame([{
        "hypothesis": "H3",
        "endpoint": "mission_priority_critical_miss_reduction",
        "execution_status": "NOT_ESTIMABLE",
        "not_estimable_reason": h3_reason,
        "estimate": np.nan, "critical_miss_reduction": np.nan,
        "ci_low": np.nan, "ci_high": np.nan, "holm_adjusted_p": np.nan,
        "positive_visual_groups": positive_groups,
        "required_positive_visual_groups": required_positive,
        "negative_visual_groups": negative_groups,
        "required_negative_visual_groups": required_negative,
        "proposed_false_alert_rate": float(proposed_far_audit["point"]),
        "reference_false_alert_rate": float(reference_far_audit["point"]),
        "false_alert_estimand": "group_trace_equal_negative_decision_false_alert_rate_v1",
        "false_alert_bootstrap_method": "two_way_visual_group_trace_cluster_bootstrap_one_sided_v1",
        "false_alert_bootstrap_repetitions": int(repetitions),
        "proposed_false_alert_cluster_bootstrap_upper_95": float(
            proposed_far_audit["upper_95"]
        ),
        "reference_false_alert_cluster_bootstrap_upper_95": float(
            reference_far_audit["upper_95"]
        ),
        "proposed_negative_decision_exposures": int(
            proposed_negative["negative_decision_exposures"].sum()
        ),
        "reference_negative_decision_exposures": int(
            reference_negative["negative_decision_exposures"].sum()
        ),
        "proposed_negative_episode_cells": int(len(proposed_negative)),
        "reference_negative_episode_cells": int(len(reference_negative)),
        "negative_trace_clusters": int(proposed_negative["trace_id"].astype(str).nunique()),
        "false_alert_descriptive_limit_pass": false_alert_gate_pass,
        "false_alert_gate_pass": False,
        "false_alert_limit": false_alert_limit,
        "claim_supported": False,
    }])

    h4_policies = ("proposed_full", "ablate_evidence_escalation")
    h4_scope = h4_cells[h4_cells["policy"].astype(str).isin(h4_policies)].copy()
    required_h4_columns = {
        "matched_coverage_passed", "target_rows", "selected_rows",
        "matched_coverage_semantic_error", "high_confidence_semantic_rows",
        "high_confidence_semantic_errors",
        "high_confidence_semantic_error_rate", "high_confidence_semantic_passed",
    }
    if missing := required_h4_columns.difference(h4_scope.columns):
        raise StudyGateError(
            f"Compact H4 cells lack conjunctive endpoint fields: {sorted(missing)}"
        )
    cell_keys = ["visual_group_id", "trace_id"]
    policy_keys = {
        policy: set(map(tuple, group[cell_keys].astype(str).to_numpy()))
        for policy, group in h4_scope.groupby("policy", sort=False)
    }
    paired_h4 = set(policy_keys) == set(h4_policies) and len(set(
        frozenset(keys) for keys in policy_keys.values()
    )) == 1
    matched_coverage_pass = bool(
        paired_h4
        and h4_scope["matched_coverage_passed"].astype(bool).all()
        and "matched_coverage_passed" in h4_audit
        and h4_audit["matched_coverage_passed"].astype(bool).all()
    )
    high_confidence_denominator_pass = bool(
        paired_h4
        and h4_scope["high_confidence_semantic_passed"].astype(bool).all()
        and "high_confidence_semantic_passed" in h4_audit
        and h4_audit["high_confidence_semantic_passed"].astype(bool).all()
    )
    selected_by_policy = h4_scope.groupby("policy")["selected_rows"].sum()
    target_by_policy = h4_scope.groupby("policy")["target_rows"].sum()
    high_confidence_by_policy = h4_scope.groupby("policy")[
        "high_confidence_semantic_rows"
    ].sum()
    matched_cells_by_policy = h4_scope.groupby("policy")[
        "matched_coverage_passed"
    ].sum()
    high_confidence_cells_by_policy = h4_scope.groupby("policy")[
        "high_confidence_semantic_passed"
    ].sum()
    required_selected = int(
        target_by_policy.max()
    ) if len(target_by_policy) else 0
    selected_rows = int(selected_by_policy.min()) if len(selected_by_policy) else 0
    high_confidence_rows = int(
        high_confidence_by_policy.min()
    ) if len(high_confidence_by_policy) else 0
    required_high_confidence = int(
        h4_scope.groupby("policy").size().max()
    ) if len(h4_scope) else 0
    required_h4_cells = required_high_confidence
    matched_coverage_cells = int(
        matched_cells_by_policy.min()
    ) if len(matched_cells_by_policy) else 0
    high_confidence_cells = int(
        high_confidence_cells_by_policy.min()
    ) if len(high_confidence_cells_by_policy) else 0
    count_pass = bool(
        selected_rows >= required_selected
        and high_confidence_rows >= required_high_confidence
        and matched_coverage_cells >= required_h4_cells
        and high_confidence_cells >= required_h4_cells
        and required_selected > 0 and required_high_confidence > 0
    )
    # v3r2 amendment: H4 tests confidence-gated evidence escalation.  When the
    # registered calibration gate failed and the frozen controller therefore
    # cannot consume confidence, H4 is NOT_ESTIMABLE by registration.
    h4_calibration_pass = policy_spec.get("calibration_trusted", True) is not False
    h4_estimable = bool(
        matched_coverage_pass and high_confidence_denominator_pass and count_pass
        and h4_calibration_pass
    )
    h4_matched_effect = (
        compact_two_way_clustered_paired_bootstrap(
            h4_scope, "ablate_evidence_escalation", "proposed_full",
            endpoint="matched_coverage_semantic_error",
            repetitions=int(repetitions), seed=_seed(cfg),
        )
        if h4_estimable else None
    )
    h4_high_confidence_effect = (
        compact_two_way_clustered_paired_bootstrap(
            h4_scope, "ablate_evidence_escalation", "proposed_full",
            endpoint="high_confidence_semantic_error_rate",
            repetitions=int(repetitions), seed=_seed(cfg),
        )
        if h4_estimable else None
    )
    h4_adjusted = holm_adjust({
        "H3": 1.0,
        "H4_matched_coverage": (
            h4_matched_effect.p_two_sided if h4_matched_effect is not None else 1.0
        ),
        "H4_high_confidence": (
            h4_high_confidence_effect.p_two_sided
            if h4_high_confidence_effect is not None else 1.0
        ),
    })
    h4_lookup = h4_audit.set_index("policy")
    proposed_risk = float(
        h4_lookup.loc["proposed_full", "group_trace_equal_selective_risk"]
    ) if "proposed_full" in h4_lookup.index else np.nan
    reference_risk = float(
        h4_lookup.loc[
            "ablate_evidence_escalation", "group_trace_equal_selective_risk"
        ]
    ) if "ablate_evidence_escalation" in h4_lookup.index else np.nan
    proposed_high_confidence_risk = float(
        h4_lookup.loc[
            "proposed_full", "group_trace_equal_high_confidence_semantic_error_rate"
        ]
    ) if "proposed_full" in h4_lookup.index else np.nan
    reference_high_confidence_risk = float(
        h4_lookup.loc[
            "ablate_evidence_escalation",
            "group_trace_equal_high_confidence_semantic_error_rate",
        ]
    ) if "ablate_evidence_escalation" in h4_lookup.index else np.nan
    h4_failures: list[str] = []
    if not h4_calibration_pass:
        h4_failures.append(
            "registered calibration gate failed; confidence-dependent escalation "
            f"disabled by {CALIBRATION_GATE_AMENDMENT_V3R2}"
        )
    if not matched_coverage_pass:
        h4_failures.append("matched-coverage denominator gate failed")
    if not high_confidence_denominator_pass:
        h4_failures.append("high-confidence semantic denominator gate failed")
    if not count_pass:
        h4_failures.append("registered H4 count gate failed")
    h4_reason = "" if h4_estimable else (
        "; ".join(h4_failures) + "; H4 remains nonblocking NOT_ESTIMABLE"
    )
    matched_significant = bool(
        h4_matched_effect is not None
        and h4_matched_effect.ci_low > 0
        and h4_adjusted["H4_matched_coverage"] <= 0.05
        and proposed_risk < reference_risk
    )
    high_confidence_significant = bool(
        h4_high_confidence_effect is not None
        and h4_high_confidence_effect.ci_low > 0
        and h4_adjusted["H4_high_confidence"] <= 0.05
        and proposed_high_confidence_risk < reference_high_confidence_risk
    )
    h4 = pd.DataFrame([{
        "hypothesis": "H4",
        "endpoint": (
            "conjunctive_matched_coverage_and_high_confidence_"
            "semantic_error_reduction"
        ),
        "claim_scope": (
            "composed_detector_verifier_pipeline_selective_action_error_"
            "not_free_form_vlm_semantic_generation"
        ),
        "semantic_packet_authorship": "deterministic_host_composer",
        "detector_proposal_conditioned": True,
        "free_form_vlm_packet_authorship_claim_eligible": False,
        "execution_status": "COMPLETE" if h4_estimable else "NOT_ESTIMABLE",
        "not_estimable_reason": h4_reason,
        "estimate": (
            h4_high_confidence_effect.estimate
            if h4_high_confidence_effect else np.nan
        ),
        "matched_coverage_error_reduction": (
            h4_matched_effect.estimate if h4_matched_effect else np.nan
        ),
        "matched_coverage_ci_low": (
            h4_matched_effect.ci_low if h4_matched_effect else np.nan
        ),
        "matched_coverage_ci_high": (
            h4_matched_effect.ci_high if h4_matched_effect else np.nan
        ),
        "matched_coverage_holm_adjusted_p": (
            h4_adjusted["H4_matched_coverage"] if h4_estimable else np.nan
        ),
        "high_confidence_error_reduction": (
            h4_high_confidence_effect.estimate
            if h4_high_confidence_effect else np.nan
        ),
        "high_confidence_ci_low": (
            h4_high_confidence_effect.ci_low
            if h4_high_confidence_effect else np.nan
        ),
        "high_confidence_ci_high": (
            h4_high_confidence_effect.ci_high
            if h4_high_confidence_effect else np.nan
        ),
        "high_confidence_holm_adjusted_p": (
            h4_adjusted["H4_high_confidence"] if h4_estimable else np.nan
        ),
        "ci_low": (
            h4_high_confidence_effect.ci_low
            if h4_high_confidence_effect else np.nan
        ),
        "ci_high": (
            h4_high_confidence_effect.ci_high
            if h4_high_confidence_effect else np.nan
        ),
        "holm_adjusted_p": (
            max(
                h4_adjusted["H4_matched_coverage"],
                h4_adjusted["H4_high_confidence"],
            )
            if h4_estimable else np.nan
        ),
        "coverage_gate_pass": matched_coverage_pass,
        "high_confidence_gate_pass": high_confidence_denominator_pass,
        "matched_coverage_selected_rows": selected_rows,
        "required_matched_coverage_selected_rows": required_selected,
        "matched_coverage_group_trace_cells": matched_coverage_cells,
        "required_matched_coverage_group_trace_cells": required_h4_cells,
        "high_confidence_rows": high_confidence_rows,
        "required_high_confidence_rows": required_high_confidence,
        "high_confidence_group_trace_cells": high_confidence_cells,
        "required_high_confidence_group_trace_cells": required_h4_cells,
        "h4_count_gate_pass": count_pass,
        "proposed_selective_risk": proposed_risk,
        "reference_selective_risk": reference_risk,
        "proposed_high_confidence_semantic_error_rate": (
            proposed_high_confidence_risk
        ),
        "reference_high_confidence_semantic_error_rate": (
            reference_high_confidence_risk
        ),
        "matched_coverage_claim_pass": matched_significant,
        "high_confidence_claim_pass": high_confidence_significant,
        "claim_supported": bool(
            h4_estimable and matched_significant and high_confidence_significant
        ),
    }])
    return {
        "h2_report": h2,
        "h2_matched_budget": h2_matched_budget,
        "h3_report": h3,
        "h4_report": h4,
        "h4_coverage_audit": h4_audit.copy(),
        "h4_matched_coverage_cells": h4_scope,
    }


def recompute_hypotheses_from_frozen_results(
    report_frames: Mapping[str, Any],
    cfg: Any,
    cluster_columns: Sequence[str],
    bootstrap_repetitions: int,
) -> dict[str, pd.DataFrame]:
    """Recompute paper hypotheses from raw legacy or compact frozen inputs."""

    reduction = report_frames.get("policy_reduction")
    if not isinstance(reduction, Mapping):
        return _recompute_hypotheses_from_frozen_results_legacy(
            report_frames, cfg, cluster_columns, bootstrap_repetitions,
        )
    pseudo = _compact_policy_pseudo_trials(reduction)
    legacy_inputs = dict(report_frames)
    legacy_inputs["exp4_policy"] = pseudo
    legacy = _recompute_hypotheses_from_frozen_results_legacy(
        legacy_inputs, cfg, cluster_columns, bootstrap_repetitions,
    )
    policy_spec = report_frames.get("policy_spec")
    if not isinstance(policy_spec, Mapping):
        raise StudyGateError("Compact recomputation requires the frozen policy specification")
    temporal_manifest = report_frames.get("temporal_manifest")
    if not isinstance(temporal_manifest, pd.DataFrame) or temporal_manifest.empty:
        raise StudyGateError(
            "Compact recomputation requires the frozen temporal cluster manifest"
        )
    dynamic = _compact_dynamic_hypothesis_reports(
        reduction, policy_spec, temporal_manifest, cfg,
        repetitions=int(bootstrap_repetitions),
    )
    legacy.update(dynamic)
    legacy.pop("h4_matched_coverage_trials", None)
    status = legacy["hypothesis_status"].copy()
    for hypothesis, frame_name in (
        ("H2", "h2_report"), ("H3", "h3_report"), ("H4", "h4_report"),
    ):
        report = legacy[frame_name]
        status.loc[status["hypothesis"].astype(str).eq(hypothesis), [
            "execution_status", "claim_supported",
        ]] = [
            str(report.iloc[0]["execution_status"]),
            bool(report["claim_supported"].astype(bool).all()),
        ]
    legacy["hypothesis_status"] = status
    return legacy


def evaluate_optional_embedded_h5(
    handoff: Any,
    cfg: Any,
    raw_trial_path: str | Path | None,
) -> pd.DataFrame:
    """Return explicit N/A for this scope or verify a fully supplied Jetson retest."""

    evaluation = _section(cfg, "evaluation")
    if "H5" in set(map(str, evaluation.get("not_applicable_hypotheses", ()))):
        return pd.DataFrame([{
            "hypothesis": "H5",
            "execution_status": H5_NOT_APPLICABLE_STATUS,
            "claim_supported": False,
            "evidence_complete": False,
            "detail": "Target-device generalization was not registered for the VisDrone-only paper scope",
        }])
    if not isinstance(handoff, Mapping):
        raise StudyGateError("Embedded H5 requires a verified handoff mapping")
    required = {"original_hypothesis_status", "embedded_log", "retest_summary", "protocol"}
    if missing := required.difference(handoff):
        raise StudyGateError(f"Embedded H5 handoff is incomplete: {sorted(missing)}")
    if raw_trial_path is None or not Path(raw_trial_path).is_file():
        raise StudyGateError("Embedded H5 requires the hashed raw per-stage trial artifact")
    hardware_cfg = _section(cfg, "hardware")
    embedded = import_embedded_jsonl(
        handoff["embedded_log"],
        expected_hardware_id=str(hardware_cfg.get("hardware_id", "")),
    )
    gates = embedded_gate(
        embedded,
        deadline_ms=float(handoff.get("deadline_ms", 1000.0)),
        memory_limit_mb=float(evaluation.get("target_memory_limit_mb", 8000)),
        thermal_limit_c=float(hardware_cfg.get("thermal_limit_c", 82)),
    )
    summary = import_embedded_retest_summary(handoff["retest_summary"])
    protocol_payload = handoff["protocol"]
    if isinstance(protocol_payload, (str, Path)):
        protocol_payload = json.loads(
            Path(protocol_payload).expanduser().resolve().read_text(encoding="utf-8")
        )
    if not isinstance(protocol_payload, Mapping):
        raise StudyGateError("Embedded H5 protocol is not a mapping")
    selection = _read_frozen_selection(protocol_payload)
    policy_spec = selection.get("policy_spec")
    if not isinstance(policy_spec, Mapping):
        raise StudyGateError("Embedded H5 protocol lacks its frozen policy specification")
    retest = evaluate_embedded_retest(
        summary,
        required_h2_baselines=tuple(map(str, policy_spec["required_h2_baselines"])),
        primary_margin=float(evaluation.get("primary_margin", 0.03)),
        false_alert_limit=float(evaluation.get("false_alert_limit", 0.05)),
        protocol_hash=str(protocol_payload["protocol_hash"]),
        target_log_sha256=hash_file(handoff["embedded_log"]),
        expected_hardware_id=str(hardware_cfg.get("hardware_id", "")),
        wire_budget_bytes_per_s=float(evaluation["primary_wire_budget_bytes_per_s"]),
        energy_budget_j_per_s=float(evaluation["primary_energy_budget_j_per_s"]),
        sample_minima={
            "natural_test": int(evaluation.get("natural_test_units", 0)),
            "safety_critical": int(evaluation.get("mission_priority_positive_units", 0)),
            "safety_noncritical": int(evaluation.get("mission_priority_negative_units", 0)),
        },
        registered_calibration_gate_pass=bool(
            selection.get("validation_gates_passed", False)
            and selection.get("calibration_claim_enabled", False)
        ),
        bootstrap_repetitions=int(evaluation.get("bootstrap_repetitions", 10_000)),
        bootstrap_seed=_seed(cfg),
        h4_matched_coverage=float(evaluation["h4_matched_coverage"]),
        h4_coverage_selection_rule="top_calibrated_semantic_score_within_visual_group",
    )
    # evaluate_embedded_retest verifies the raw path/hash referenced by summary;
    # additionally bind the caller-supplied path to that exact artifact.
    supplied_hash = hash_file(raw_trial_path)
    referenced = set(summary["trial_artifact_sha256"].astype(str))
    if referenced != {supplied_hash}:
        raise StudyGateError("Caller-supplied embedded raw trials differ from the verified retest summary")
    return evaluate_h5_status(handoff["original_hypothesis_status"], retest, gates)


__all__ = [
    "RUNNER_VERSION",
    "evaluate_optional_embedded_h5",
    "prepare_and_freeze_feature_head",
    "recompute_hypotheses_from_frozen_results",
    "run_confirmatory_test_once",
    "run_validation_network_and_controller",
    "run_validation_perception_and_frontier",
    "select_paper_perception_sample",
]
