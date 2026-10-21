"""Registered repeated-measurement wrappers for real VisDrone inference.

Warmup observations are frozen development inputs and never contribute timing,
energy, memory, or temperature values. Measured repetitions may differ only in
resource measurements; any committed detector or VLM output drift fails closed.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from .perception import (
    InferenceRecord,
    InferenceRequest,
    LEGACY_MOONDREAM_INFERENCE_FIELDS,
    PerceptionBackend,
    UltralyticsDetectorTracker,
    run_detector_cache,
)
from .registry import StudyGateError, canonical_json, hash_file, hash_object


DEVELOPMENT_WARMUP_SPLITS = frozenset({"development", "train", "calibration", "validation"})
DETECTOR_MEASUREMENT_AGGREGATION = (
    "median_latency_energy_max_peak_memory_temperature_across_reset_repetitions"
)
VLM_MEASUREMENT_AGGREGATION = "median_latency_energy_max_peak_memory_temperature"
MEASUREMENT_SCHEMA_VERSION = "semuav.repeated-inference-measurement.v2"
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_frozen_warmup(
    image_path: str | Path,
    image_sha256: str,
    *,
    source_id: str,
    split: str,
) -> Path:
    if not str(source_id).strip():
        raise ValueError("warmup_source_id must be nonempty")
    normalized_split = str(split).strip().lower()
    if normalized_split not in DEVELOPMENT_WARMUP_SPLITS:
        raise ValueError("Warmup input must come from a frozen development split")
    digest = str(image_sha256).strip().lower()
    if _HEX64.fullmatch(digest) is None:
        raise ValueError("warmup_image_sha256 must be 64 lowercase hexadecimal characters")
    path = Path(image_path).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise StudyGateError(f"Frozen warmup image is missing or not a regular file: {path}")
    if hash_file(path) != digest:
        raise StudyGateError("Frozen development warmup image changed")
    return path


def _finite_metric(values: Sequence[Any], *, name: str, aggregation: str) -> float:
    numeric: list[float] = []
    for value in values:
        if value is None:
            raise StudyGateError(f"Measured {name} is missing")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise StudyGateError(f"Measured {name} is nonnumeric") from exc
        if not math.isfinite(number) or number < 0:
            raise StudyGateError(f"Measured {name} must be finite and nonnegative")
        numeric.append(number)
    if not numeric:
        raise StudyGateError(f"Measured {name} has no repetitions")
    if aggregation == "median":
        return float(statistics.median(numeric))
    if aggregation == "max":
        return float(max(numeric))
    raise ValueError(f"Unsupported measurement aggregation: {aggregation}")


def run_repeated_detector_measurement(
    detector_factory: Callable[[], UltralyticsDetectorTracker],
    manifest: pd.DataFrame,
    *,
    warmup_image_path: str | Path,
    warmup_image_sha256: str,
    warmup_source_id: str,
    warmup_split: str,
    warmup_iterations: int = 5,
    repetitions: int = 3,
    device_index: int = 0,
) -> pd.DataFrame:
    """Measure a detector/tracker stream after unmeasured frozen warmups.

    The factory is invoked once for warmup and once per measured repetition.
    Each measured tracker is therefore fresh before :func:`run_detector_cache`
    deduplicates mission-profile copies and replays the complete registered
    stream in order. Reusing the same tracker instance fails closed.
    """

    if not callable(detector_factory):
        raise ValueError("detector_factory must be callable")
    warmups = _positive_integer(warmup_iterations, name="warmup_iterations")
    repeats = _positive_integer(repetitions, name="repetitions")
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise ValueError("device_index must be a nonnegative integer")
    if not isinstance(manifest, pd.DataFrame) or manifest.empty:
        raise ValueError("A nonempty detector manifest is required")
    warmup_path = _validate_frozen_warmup(
        warmup_image_path,
        warmup_image_sha256,
        source_id=warmup_source_id,
        split=warmup_split,
    )

    def make_detector(*, role: str) -> UltralyticsDetectorTracker:
        detector = detector_factory()
        if detector is None or not callable(getattr(detector, "detect", None)):
            raise StudyGateError(f"detector_factory returned an invalid {role} detector")
        if (
            not str(getattr(detector, "tracker", "")).strip()
            or not str(getattr(detector, "device", "")).strip()
            or not isinstance(getattr(detector, "inference_spec", None), Mapping)
        ):
            raise StudyGateError(
                f"detector_factory returned an unfrozen {role} detector configuration"
            )
        return detector

    # Warmups intentionally bypass run_detector_cache/profile_callable, so
    # their measurements cannot leak into the paper statistics.  A separate
    # warmup instance verifies the factory contract, while every fresh
    # measured instance is warmed as well.  Warming only a throwaway instance
    # would leave model materialization/autotuning in the first timed frame of
    # every repetition.
    warmup_detector = make_detector(role="warmup")
    detector_commitment = canonical_json({
        "tracker": warmup_detector.tracker,
        "device": warmup_detector.device,
        "inference_spec": warmup_detector.inference_spec,
    })
    for _ in range(warmups):
        warmup_detector.detect(str(warmup_path), persist=False)

    measured: list[pd.DataFrame] = []
    detector_instances: list[UltralyticsDetectorTracker] = [warmup_detector]
    for repetition in range(1, repeats + 1):
        detector = make_detector(role=f"repetition {repetition}")
        if any(detector is previous for previous in detector_instances):
            raise StudyGateError(
                "detector_factory reused tracker state across warmup or repetitions"
            )
        detector_instances.append(detector)
        observed_commitment = canonical_json({
            "tracker": detector.tracker,
            "device": detector.device,
            "inference_spec": detector.inference_spec,
        })
        if observed_commitment != detector_commitment:
            raise StudyGateError(
                "detector_factory changed frozen configuration across repetitions"
            )
        for _ in range(warmups):
            detector.detect(str(warmup_path), persist=False)
        frame = run_detector_cache(
            detector,
            manifest,
            measure_energy=True,
            device_index=device_index,
        ).copy()
        if frame.empty:
            raise StudyGateError("Detector repetition produced no rows")
        frame["_measurement_repetition"] = repetition
        measured.append(frame)

    identity_columns = ("sample_id", "source_sample_id", "sequence_id")
    required_columns = set(identity_columns) | {
        "proposals_json", "latency_ms", "energy_j", "peak_memory_mb",
        "peak_temperature_c", "_measurement_repetition",
    }
    for frame in measured:
        missing = sorted(required_columns.difference(frame.columns))
        if missing:
            raise StudyGateError(f"Detector repetition is missing columns: {missing}")
        if frame[list(identity_columns)].astype(str).duplicated().any():
            raise StudyGateError("Detector repetition has duplicate profile/source identities")

    baseline = measured[0].sort_values(list(identity_columns), kind="stable").reset_index(drop=True)
    committed_columns = [*identity_columns, "proposals_json"]
    baseline_committed = baseline.loc[:, committed_columns].astype(str)
    for repetition, frame in enumerate(measured[1:], start=2):
        ordered = frame.sort_values(list(identity_columns), kind="stable").reset_index(drop=True)
        if not ordered.loc[:, committed_columns].astype(str).equals(baseline_committed):
            raise StudyGateError(
                "Detector identities/proposals are nondeterministic in measured "
                f"repetition {repetition}"
            )

    combined = pd.concat(measured, ignore_index=True)
    proposal_counts = combined.groupby("source_sample_id", sort=False)["proposals_json"].nunique(
        dropna=False,
    )
    if proposal_counts.gt(1).any():
        raise StudyGateError("Detector proposal replication differs across profiles/repetitions")
    source_measurements = combined.drop_duplicates(
        ["source_sample_id", "_measurement_repetition"], keep="first",
    )
    expected_per_source = source_measurements.groupby("source_sample_id", sort=False)[
        "_measurement_repetition"
    ].nunique()
    if not expected_per_source.eq(repeats).all():
        raise StudyGateError("Detector measured repetitions are incomplete for a source frame")

    aggregates: dict[str, dict[str, float]] = {}
    for source_id, group in source_measurements.groupby("source_sample_id", sort=False):
        aggregates[str(source_id)] = {
            "latency_ms": _finite_metric(group["latency_ms"].tolist(), name="latency_ms", aggregation="median"),
            "energy_j": _finite_metric(group["energy_j"].tolist(), name="energy_j", aggregation="median"),
            "peak_memory_mb": _finite_metric(
                group["peak_memory_mb"].tolist(), name="peak_memory_mb", aggregation="max",
            ),
            "peak_temperature_c": _finite_metric(
                group["peak_temperature_c"].tolist(), name="peak_temperature_c", aggregation="max",
            ),
        }
    output = measured[0].drop(columns=["_measurement_repetition"]).copy()
    for metric in ("latency_ms", "energy_j", "peak_memory_mb", "peak_temperature_c"):
        output[metric] = output["source_sample_id"].astype(str).map(
            lambda source_id, metric=metric: aggregates[source_id][metric]
        )
    output["measurement_warmup_iterations"] = warmups
    output["measurement_repetitions"] = repeats
    output["measurement_schema_version"] = MEASUREMENT_SCHEMA_VERSION
    output["measurement_aggregation"] = DETECTOR_MEASUREMENT_AGGREGATION
    output["measurement_state_reset"] = (
        "fresh_tracker_per_repetition_each_warmed_unmeasured_"
        "and_persist_false_at_every_sequence_start"
    )
    output["measurement_warmup_excluded"] = True
    output["measurement_warmup_source_id"] = str(warmup_source_id)
    output["measurement_warmup_split"] = str(warmup_split).strip().lower()
    output["measurement_warmup_image_path"] = str(warmup_path)
    output["measurement_warmup_image_sha256"] = str(warmup_image_sha256).lower()
    output["measurement_detector_spec_hash"] = hash_object(
        json.loads(detector_commitment)
    )
    output["measurement_repetition_ids_json"] = canonical_json(list(range(1, repeats + 1)))
    return output


class RepeatedMeasurementPerceptionBackend(PerceptionBackend):
    """Warm once, then aggregate deterministic repeated VLM inference records."""

    def __init__(
        self,
        backend: PerceptionBackend,
        *,
        warmup_request: InferenceRequest,
        warmup_image_sha256: str,
        warmup_split: str,
        warmup_prompt: str,
        warmup_max_new_tokens: int,
        warmup_iterations: int = 5,
        repetitions: int = 3,
    ) -> None:
        if backend is None or not hasattr(backend, "infer") or not hasattr(backend, "spec"):
            raise ValueError("A concrete PerceptionBackend with a model spec is required")
        if not isinstance(warmup_request, InferenceRequest):
            raise ValueError("warmup_request must be an InferenceRequest")
        if not str(warmup_prompt).strip():
            raise ValueError("warmup_prompt must be nonempty")
        self.warmup_iterations = _positive_integer(
            warmup_iterations, name="warmup_iterations",
        )
        self.repetitions = _positive_integer(repetitions, name="repetitions")
        self.warmup_max_new_tokens = _positive_integer(
            warmup_max_new_tokens, name="warmup_max_new_tokens",
        )
        self.backend = backend
        self.spec = backend.spec
        self.warmup_request = warmup_request
        self.warmup_prompt = str(warmup_prompt)
        self.warmup_split = str(warmup_split).strip().lower()
        self.warmup_image_sha256 = str(warmup_image_sha256).strip().lower()
        self._warmed = False
        self.measurement_metadata = {
            "measurement_schema_version": MEASUREMENT_SCHEMA_VERSION,
            "measurement_warmup_iterations": self.warmup_iterations,
            "measurement_repetitions": self.repetitions,
            "measurement_aggregation": VLM_MEASUREMENT_AGGREGATION,
            "measurement_warmup_excluded": True,
            "measurement_warmup_source_id": str(warmup_request.observation_id),
            "measurement_warmup_split": self.warmup_split,
            "measurement_warmup_image_sha256": self.warmup_image_sha256,
        }
        _validate_frozen_warmup(
            warmup_request.image_path,
            self.warmup_image_sha256,
            source_id=warmup_request.observation_id,
            split=self.warmup_split,
        )

    def _warm_once(self) -> None:
        if self._warmed:
            return
        _validate_frozen_warmup(
            self.warmup_request.image_path,
            self.warmup_image_sha256,
            source_id=self.warmup_request.observation_id,
            split=self.warmup_split,
        )
        for _ in range(self.warmup_iterations):
            self.backend.infer(
                self.warmup_request,
                self.warmup_prompt,
                self.warmup_max_new_tokens,
            )
        self._warmed = True

    @staticmethod
    def _committed_record(record: InferenceRecord) -> str:
        record.require_active_evidence_boundary()
        committed = {
            "observation_id": record.observation_id,
            "model_id": record.model_id,
            "model_revision": record.model_revision,
            "raw_text": record.raw_text,
            "parsed": record.parsed,
            "schema_valid": record.schema_valid,
            "error": record.error,
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
            "confidence": record.confidence,
            "confidence_source": record.confidence_source,
            "prompt_hash": record.prompt_hash,
            "decision_contract_version": record.decision_contract_version,
            "decision_core_valid": record.decision_core_valid,
            "decision_core_json": record.decision_core_json,
            "decision_context_sha256": record.decision_context_sha256,
            "decision_proposal_count": record.decision_proposal_count,
            "decision_composer_policy_sha256": (
                record.decision_composer_policy_sha256
            ),
            "terminated_by_eos": record.terminated_by_eos,
            "reached_token_cap": record.reached_token_cap,
            "generation_policy_version": record.generation_policy_version,
            "generation_policy_sha256": record.generation_policy_sha256,
            "generation_strategy": record.generation_strategy,
            "generation_max_new_tokens": record.generation_max_new_tokens,
            "assistant_prefill_sha256": record.assistant_prefill_sha256,
            "assistant_prefill_characters": (
                record.assistant_prefill_characters
            ),
            "generated_continuation_sha256": (
                record.generated_continuation_sha256
            ),
        }
        if any(name in committed for name in LEGACY_MOONDREAM_INFERENCE_FIELDS):
            raise StudyGateError(
                "Legacy Moondream keys entered the active measurement commitment"
            )
        return canonical_json(committed)

    def infer(
        self,
        request: InferenceRequest,
        prompt: str,
        max_new_tokens: int,
    ) -> InferenceRecord:
        if not isinstance(request, InferenceRequest):
            raise ValueError("request must be an InferenceRequest")
        if not str(prompt).strip():
            raise ValueError("prompt must be nonempty")
        token_limit = _positive_integer(max_new_tokens, name="max_new_tokens")
        self._warm_once()
        records = [
            self.backend.infer(request, prompt, token_limit)
            for _ in range(self.repetitions)
        ]
        expected_prompt_hash = hash_object(prompt)
        if any(record.prompt_hash != expected_prompt_hash for record in records):
            raise StudyGateError("Measured VLM record is not bound to the exact prompt content")
        committed = [self._committed_record(record) for record in records]
        if len(set(committed)) != 1:
            raise StudyGateError("Measured VLM output/provenance is nondeterministic")
        first = records[0]
        return dataclasses.replace(
            first,
            latency_ms=_finite_metric(
                [record.latency_ms for record in records],
                name="latency_ms",
                aggregation="median",
            ),
            energy_j=_finite_metric(
                [record.energy_j for record in records],
                name="energy_j",
                aggregation="median",
            ),
            peak_memory_mb=_finite_metric(
                [record.peak_memory_mb for record in records],
                name="peak_memory_mb",
                aggregation="max",
            ),
            peak_temperature_c=_finite_metric(
                [record.peak_temperature_c for record in records],
                name="peak_temperature_c",
                aggregation="max",
            ),
        )


__all__ = [
    "DETECTOR_MEASUREMENT_AGGREGATION",
    "DEVELOPMENT_WARMUP_SPLITS",
    "MEASUREMENT_SCHEMA_VERSION",
    "RepeatedMeasurementPerceptionBackend",
    "VLM_MEASUREMENT_AGGREGATION",
    "run_repeated_detector_measurement",
]
