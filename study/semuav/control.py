from __future__ import annotations

import dataclasses
import math
from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .protocol import PayloadMode, ReliabilityProfile, SenderReceiverBelief, StateKind


@dataclasses.dataclass(frozen=True)
class LinkEstimate:
    as_of_us: int
    sample_count: int
    service_rate_bps: float
    one_way_delay_us: float
    loss_probability: float
    rto_us: float


@dataclasses.dataclass(frozen=True)
class QueueSnapshot:
    as_of_us: int
    pending_bytes_by_priority: tuple[int, ...]
    inflight_bytes: int
    outstanding_unacked_bytes: int
    oldest_ready_us_by_priority: tuple[int | None, ...]
    messages_by_priority: tuple[int, ...]

    @property
    def pending_bytes(self) -> int:
        # Outstanding bytes have already consumed link service.  They remain a
        # reliability risk, but are not counted as queued service until an RTO
        # actually enqueues a retry.
        return int(sum(self.pending_bytes_by_priority) + self.inflight_bytes)


@dataclasses.dataclass(frozen=True)
class ControllerObservation:
    now_us: int
    capture_us: int
    action_deadline_us: int
    link: LinkEstimate
    queue: QueueSnapshot
    receiver_belief: SenderReceiverBelief
    mission_priority: float
    risk: float
    uncertainty: float
    age_of_information_us: int | None
    battery_fraction: float
    thermal_c: float
    previous_mode: PayloadMode | None = None
    previous_switch_us: int | None = None
    previous_variant: str | None = None

    def validate_causal(self) -> None:
        if self.capture_us > self.now_us:
            raise ValueError("Capture time is in the future")
        if self.link.as_of_us > self.now_us or self.queue.as_of_us > self.now_us:
            raise ValueError("Controller received future-derived state")
        if self.receiver_belief.feedback_arrival_us is not None and self.receiver_belief.feedback_arrival_us > self.now_us:
            raise ValueError("Receiver feedback has not arrived yet")
        for name in ("mission_priority", "risk", "uncertainty", "battery_fraction"):
            if not 0 <= float(getattr(self, name)) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")

    @property
    def remaining_deadline_us(self) -> int:
        return self.action_deadline_us - self.now_us


@dataclasses.dataclass(frozen=True)
class CandidateProfile:
    mode: PayloadMode
    payload_bytes: int
    generation_us: int
    energy_j: float
    predicted_task_success: float
    predicted_high_confidence_error: float
    evidence_strength: float
    state_kind: StateKind = StateKind.NONE
    base_state_seq: int | None = None
    resulting_state_seq: int | None = None
    available_us: int = 0
    variant: str = ""
    receiver_latency_us: int = 0
    receiver_energy_j: float = 0.0

    def __post_init__(self) -> None:
        if (
            self.payload_bytes < 0 or self.generation_us < 0 or self.energy_j < 0
            or self.receiver_latency_us < 0 or self.receiver_energy_j < 0
        ):
            raise ValueError("Candidate resources cannot be negative")
        for name in ("predicted_task_success", "predicted_high_confidence_error", "evidence_strength"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0,1]")


@dataclasses.dataclass(frozen=True)
class TransmissionPlan:
    mode: PayloadMode
    priority: int
    state_kind: StateKind
    reliability: ReliabilityProfile
    cancel_message_ids: tuple[int, ...] = ()
    reason_code: str = ""
    variant: str = ""


class RepresentationPolicy(ABC):
    name: str

    def reset(self, run_seed: int) -> None:
        self.run_seed = run_seed

    def on_feedback(self, feedback: Any, now_us: int) -> None:
        return None

    @abstractmethod
    def select(self, obs: ControllerObservation, candidates: tuple[CandidateProfile, ...]) -> TransmissionPlan:
        raise NotImplementedError

    @staticmethod
    def _find(candidates: Sequence[CandidateProfile], mode: PayloadMode) -> CandidateProfile:
        for candidate in candidates:
            if candidate.mode is mode:
                return candidate
        raise ValueError(f"Candidate mode {mode.name} is unavailable")

    @staticmethod
    def _plan(candidate: CandidateProfile, reason: str, priority: int = 3) -> TransmissionPlan:
        retries = 3 if priority >= 8 else 2
        return TransmissionPlan(candidate.mode, priority, candidate.state_kind, ReliabilityProfile(retry_limit=retries), (), reason, candidate.variant)


class FixedModePolicy(RepresentationPolicy):
    def __init__(self, mode: PayloadMode):
        self.mode = mode
        self.name = f"fixed_{mode.name.lower()}"

    def select(self, obs: ControllerObservation, candidates: tuple[CandidateProfile, ...]) -> TransmissionPlan:
        obs.validate_causal()
        return self._plan(self._find(candidates, self.mode), "fixed_mode")


class FixedVariantPolicy(RepresentationPolicy):
    def __init__(self, mode: PayloadMode, variant: str, name: str):
        self.mode, self.variant, self.name = mode, variant, name

    def select(self, obs: ControllerObservation, candidates: tuple[CandidateProfile, ...]) -> TransmissionPlan:
        obs.validate_causal()
        matches = [candidate for candidate in candidates if candidate.mode is self.mode and candidate.variant == self.variant]
        if not matches:
            raise ValueError(f"Registered fixed candidate {self.mode.name}/{self.variant} is unavailable")
        return self._plan(matches[0], "fixed_registered_variant")


def estimated_completion_us(obs: ControllerObservation, candidate: CandidateProfile) -> float:
    # The causal estimator is ACK-derived delivered service/goodput, so applying
    # a second (1-loss) penalty would double-count observed loss.
    rate = max(1.0, obs.link.service_rate_bps)
    queued_bits = (obs.queue.pending_bytes + candidate.payload_bytes) * 8
    return (
        obs.now_us + candidate.generation_us + queued_bits / rate * 1e6
        + obs.link.one_way_delay_us + candidate.receiver_latency_us
    )


class ChannelOnlyPolicy(RepresentationPolicy):
    name = "channel_only_hysteresis"

    def __init__(self, minimum_dwell_us: int = 2_000_000, hysteresis_margin: float = 0.10):
        self.minimum_dwell_us = minimum_dwell_us
        self.hysteresis_margin = hysteresis_margin

    def select(self, obs: ControllerObservation, candidates: tuple[CandidateProfile, ...]) -> TransmissionPlan:
        obs.validate_causal()
        feasible = [c for c in candidates if estimated_completion_us(obs, c) <= obs.action_deadline_us]
        chosen = max(feasible or list(candidates), key=lambda c: (c.predicted_task_success, -c.payload_bytes))
        if obs.previous_mode is not None and obs.previous_switch_us is not None:
            previous_matches = [
                candidate for candidate in candidates
                if candidate.mode is obs.previous_mode
                and (obs.previous_variant is None or candidate.variant == obs.previous_variant)
            ]
            previous = previous_matches[0] if previous_matches else self._find(candidates, obs.previous_mode)
            dwell_ok = obs.now_us - obs.previous_switch_us >= self.minimum_dwell_us
            improvement = chosen.predicted_task_success - previous.predicted_task_success
            previous_feasible = estimated_completion_us(obs, previous) <= obs.action_deadline_us
            if (not dwell_ok or improvement < self.hysteresis_margin) and previous_feasible:
                chosen = previous
        return self._plan(chosen, "channel_feasibility_hysteresis")


class MissionOnlyPolicy(RepresentationPolicy):
    name = "mission_only"

    def select(self, obs: ControllerObservation, candidates: tuple[CandidateProfile, ...]) -> TransmissionPlan:
        obs.validate_causal()
        target = PayloadMode.L3_SEMANTICS_ROI if obs.mission_priority >= 0.7 else PayloadMode.L2_SEMANTICS
        available = {candidate.mode for candidate in candidates}
        if target not in available:
            target = min(available, key=lambda mode: abs(int(mode) - int(target)))
        return self._plan(self._find(candidates, target), "mission_priority", priority=8 if obs.mission_priority >= 0.7 else 4)


class DeadlineGreedyPolicy(RepresentationPolicy):
    name = "queue_deadline_greedy"

    def select(self, obs: ControllerObservation, candidates: tuple[CandidateProfile, ...]) -> TransmissionPlan:
        obs.validate_causal()
        feasible = [c for c in candidates if estimated_completion_us(obs, c) <= obs.action_deadline_us]
        chosen = max(feasible, key=lambda c: (c.predicted_task_success, -c.payload_bytes)) if feasible else min(candidates, key=lambda c: c.payload_bytes)
        return self._plan(chosen, "deadline_feasible_best")


class AdaptiveImagePolicy(RepresentationPolicy):
    name = "adaptive_image_codec"

    def select(self, obs: ControllerObservation, candidates: tuple[CandidateProfile, ...]) -> TransmissionPlan:
        obs.validate_causal()
        images = [candidate for candidate in candidates if candidate.mode is PayloadMode.L5_IMAGE_OR_CLIP]
        if not images:
            raise ValueError("Adaptive image baseline requires registered L5 candidates")
        feasible = [candidate for candidate in images if estimated_completion_us(obs, candidate) <= obs.action_deadline_us]
        chosen = max(feasible, key=lambda item: (item.predicted_task_success, -item.payload_bytes)) if feasible else min(images, key=lambda item: item.payload_bytes)
        return self._plan(chosen, "within_image_rate_adaptation")


class AdaptiveSemanticPolicy(RepresentationPolicy):
    name = "adaptive_semantic_rate"

    def select(self, obs: ControllerObservation, candidates: tuple[CandidateProfile, ...]) -> TransmissionPlan:
        obs.validate_causal()
        semantics = [candidate for candidate in candidates if candidate.mode in {PayloadMode.L1_ALERT, PayloadMode.L2_SEMANTICS, PayloadMode.L3_SEMANTICS_ROI}]
        if not semantics:
            raise ValueError("Adaptive semantic baseline requires registered L1-L3 candidates")
        feasible = [candidate for candidate in semantics if estimated_completion_us(obs, candidate) <= obs.action_deadline_us]
        chosen = max(feasible, key=lambda item: (item.predicted_task_success, item.evidence_strength, -item.payload_bytes)) if feasible else min(semantics, key=lambda item: item.payload_bytes)
        return self._plan(chosen, "within_semantic_rate_adaptation")


class AdaptiveFeaturePolicy(RepresentationPolicy):
    name = "adaptive_feature_rate"

    def select(self, obs: ControllerObservation, candidates: tuple[CandidateProfile, ...]) -> TransmissionPlan:
        obs.validate_causal()
        features = [candidate for candidate in candidates if candidate.mode is PayloadMode.L4_FEATURES]
        if not features:
            raise ValueError("Adaptive feature baseline requires registered L4 candidates")
        feasible = [candidate for candidate in features if estimated_completion_us(obs, candidate) <= obs.action_deadline_us]
        chosen = max(
            feasible, key=lambda item: (item.predicted_task_success, -item.payload_bytes)
        ) if feasible else min(features, key=lambda item: item.payload_bytes)
        return self._plan(chosen, "within_feature_rate_adaptation")


class ChannelMissionPolicy(RepresentationPolicy):
    """Ablation that has channel and exogenous mission priority, but no risk/uncertainty."""

    name = "channel_plus_mission"

    def select(self, obs: ControllerObservation, candidates: tuple[CandidateProfile, ...]) -> TransmissionPlan:
        obs.validate_causal()
        feasible = [candidate for candidate in candidates if estimated_completion_us(obs, candidate) <= obs.action_deadline_us]
        pool = feasible or list(candidates)
        target_strength = 0.8 if obs.mission_priority >= 0.7 else 0.5
        chosen = max(pool, key=lambda item: item.predicted_task_success - 0.05 * abs(item.evidence_strength - target_strength) - 0.01 * math.log1p(item.payload_bytes))
        return self._plan(chosen, "channel_and_exogenous_mission")


class FullPolicy(RepresentationPolicy):
    """Full causal controller and the shared implementation for component ablations.

    The capability flags are part of the experimental contract.  The episode
    runner uses the same flags when deciding which perception work happens
    before a decision and whether ACK-derived state deltas may be generated.
    This prevents an ablation from silently receiving a removed signal through
    the VLM trigger or state encoder.
    """

    name = "proposed_full"
    uses_mission_risk = True
    uses_uncertainty = True
    uses_receiver_state = True
    uses_acknowledged_deltas = True
    uses_evidence_escalation = True
    uses_context_model = True
    force_context_model = False

    def __init__(
        self,
        minimum_dwell_us: int = 2_000_000,
        hysteresis_margin: float = 0.04,
        thermal_limit_c: float = 82.0,
        risk_escalation: float = 0.7,
        uncertainty_escalation: float = 0.35,
        byte_weight: float = 0.02,
        energy_weight: float = 0.08,
        evidence_weight: float = 0.30,
        deadline_weight: float = 1.0,
        vlm_trigger_risk: float = 0.45,
        vlm_trigger_uncertainty: float = 0.35,
        calibration_trusted: bool = True,
    ):
        # v3r2 pre-specified revision: when the registered calibration gate
        # fails for the selected perception model, model uncertainty and the
        # high-confidence-error estimate cannot trigger evidence escalation.
        self.calibration_trusted = bool(calibration_trusted)
        self.minimum_dwell_us = minimum_dwell_us
        self.hysteresis_margin = hysteresis_margin
        self.thermal_limit_c = thermal_limit_c
        self.risk_escalation = risk_escalation
        self.uncertainty_escalation = uncertainty_escalation
        self.byte_weight = byte_weight
        self.energy_weight = energy_weight
        self.evidence_weight = evidence_weight
        self.deadline_weight = deadline_weight
        self.vlm_trigger_risk = vlm_trigger_risk
        self.vlm_trigger_uncertainty = vlm_trigger_uncertainty
        for name in (
            "risk_escalation", "uncertainty_escalation",
            "vlm_trigger_risk", "vlm_trigger_uncertainty",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")

    def select(self, obs: ControllerObservation, candidates: tuple[CandidateProfile, ...]) -> TransmissionPlan:
        obs.validate_causal()
        if not self.uses_mission_risk:
            obs = dataclasses.replace(obs, mission_priority=0.0, risk=0.0)
        if not self.uses_uncertainty:
            obs = dataclasses.replace(obs, uncertainty=0.0)
            candidates = tuple(
                dataclasses.replace(candidate, predicted_high_confidence_error=0.0)
                for candidate in candidates
            )
        if not self.uses_receiver_state:
            assumed = dataclasses.replace(
                obs.receiver_belief, status="known", feedback_arrival_us=None,
            )
            obs = dataclasses.replace(
                obs, receiver_belief=assumed, age_of_information_us=None,
            )
        if not self.uses_evidence_escalation:
            # This is the registered semantic-only comparator.  L3 carries an
            # ROI, L4 is a learned feature representation, and L5 is pixels;
            # admitting any of them would retain an evidence-bearing escape
            # path and no longer isolate evidence escalation.
            candidates = tuple(
                candidate for candidate in candidates
                if candidate.mode in {
                    PayloadMode.L0_HEARTBEAT,
                    PayloadMode.L1_ALERT,
                    PayloadMode.L2_SEMANTICS,
                }
            )
            if not candidates:
                raise ValueError("Evidence-escalation ablation requires an L0-L2 semantic candidate")
        candidates = tuple(c for c in candidates if c.available_us <= obs.now_us)
        if not candidates:
            raise ValueError("No causally available candidates")
        # Receiver synchronization is a hard override, not a weighted preference.
        if obs.receiver_belief.status != "known":
            keyframes = [c for c in candidates if c.state_kind is StateKind.KEYFRAME]
            if keyframes:
                chosen = min(keyframes, key=lambda c: c.payload_bytes)
                return self._plan(chosen, "receiver_state_repair", priority=9)
        urgent = obs.risk >= self.risk_escalation or obs.mission_priority >= 0.85 or obs.remaining_deadline_us <= 300_000
        needs_evidence = self.uses_evidence_escalation and self.calibration_trusted and (
            obs.uncertainty >= self.uncertainty_escalation
            or max(c.predicted_high_confidence_error for c in candidates) > 0.05
        )
        feasible = [c for c in candidates if estimated_completion_us(obs, c) <= obs.action_deadline_us]
        pool = feasible or list(candidates)
        if urgent:
            alerts = [c for c in pool if c.mode in {PayloadMode.L1_ALERT, PayloadMode.L2_SEMANTICS, PayloadMode.L3_SEMANTICS_ROI}]
            pool = alerts or pool
        if needs_evidence:
            evidence = [c for c in pool if c.mode in {PayloadMode.L3_SEMANTICS_ROI, PayloadMode.L5_IMAGE_OR_CLIP}]
            if evidence:
                pool = evidence
        thermal_penalty = 2.0 if obs.thermal_c >= self.thermal_limit_c or obs.battery_fraction <= 0.15 else 1.0
        def score(candidate: CandidateProfile) -> float:
            reliability_value = candidate.predicted_task_success * (1 - candidate.predicted_high_confidence_error)
            evidence_bonus = (self.evidence_weight * candidate.evidence_strength) if needs_evidence else 0.0
            byte_cost = self.byte_weight * math.log1p(candidate.payload_bytes)
            # Sender representation and ground/edge receiver compute are both
            # decision-dependent. Pending detector/VLM energy, when any, is
            # added causally to candidate.energy_j by the trial runner.
            energy_cost = thermal_penalty * self.energy_weight * (
                candidate.energy_j + candidate.receiver_energy_j
            )
            deadline_penalty = self.deadline_weight * max(0.0, (estimated_completion_us(obs, candidate) - obs.action_deadline_us) / 1e6)
            return reliability_value + evidence_bonus - byte_cost - energy_cost - deadline_penalty
        chosen = max(pool, key=score)
        safety_override = urgent or needs_evidence or obs.receiver_belief.status != "known"
        if not safety_override and obs.previous_mode is not None and obs.previous_switch_us is not None:
            previous_matches = [
                c for c in pool if c.mode is obs.previous_mode
                and (obs.previous_variant is None or c.variant == obs.previous_variant)
            ]
            if previous_matches:
                previous = previous_matches[0]
                dwell_ok = obs.now_us - obs.previous_switch_us >= self.minimum_dwell_us
                if not dwell_ok or score(chosen) - score(previous) < self.hysteresis_margin:
                    chosen = previous
        reason = "urgent_minimum_sufficient" if urgent else "uncertainty_evidence" if needs_evidence else "resource_aware_minimum_sufficient"
        return self._plan(chosen, reason, priority=10 if urgent else 5)


class RiskAwareNoUncertaintyPolicy(FullPolicy):
    """Retain mission/risk while removing uncertainty and its VLM trigger."""

    name = "risk_aware_no_uncertainty"
    uses_uncertainty = False


class NoMissionRiskPolicy(FullPolicy):
    """Component ablation with channel/uncertainty/state but no mission or risk cue."""

    name = "ablate_mission_risk"
    uses_mission_risk = False


class NoEvidenceEscalationPolicy(FullPolicy):
    """Full inputs, but no evidence escalation and only L0-L2 candidates."""

    name = "ablate_evidence_escalation"
    uses_evidence_escalation = False


class NoReceiverStatePolicy(FullPolicy):
    """Ablation hiding receiver belief and disabling ACK-derived state deltas."""

    name = "ablate_receiver_state"
    uses_receiver_state = False
    uses_acknowledged_deltas = False


class NoHysteresisPolicy(FullPolicy):
    """Switching-cost ablation with no minimum dwell or score hysteresis."""

    name = "ablate_hysteresis"

    def __init__(self, **kwargs: Any):
        super().__init__(minimum_dwell_us=0, hysteresis_margin=0.0, **kwargs)


class AlwaysVLMPolicy(FullPolicy):
    """Compute ablation; the runner forces compact-VLM context on every capture."""

    name = "always_vlm_full"
    force_context_model = True


@dataclasses.dataclass(frozen=True)
class LookaheadContext:
    future_trace: Any
    true_event: Any
    cached_receiver_outcomes: Mapping[str, Any]


class LookaheadReferencePolicy:
    """Non-deployable one-step heuristic reference with privileged information.

    This selector knows the frozen receiver outcome and estimates a candidate's
    next-hop completion on the future trace.  It does *not* counterfactually
    replay queue interactions, retries, later decisions, or receiver state, so
    it is neither an oracle nor a mathematical upper bound.
    """

    name = "heuristic_lookahead_reference"

    def reset(self, run_seed: int) -> None:
        self.run_seed = run_seed

    def on_feedback(self, feedback: Any, now_us: int) -> None:
        return None

    def select(
        self,
        obs: ControllerObservation,
        candidates: tuple[CandidateProfile, ...],
        lookahead: LookaheadContext,
    ) -> TransmissionPlan:
        obs.validate_causal()
        outcome = lookahead.cached_receiver_outcomes
        chosen = max(candidates, key=lambda c: (
            bool(outcome.get(f"{c.mode.name}:{c.variant}", {}).get("correct_timely", False)),
            -c.payload_bytes,
        ))
        return TransmissionPlan(
            chosen.mode, 10, chosen.state_kind, ReliabilityProfile(retry_limit=3),
            (), "privileged_one_step_lookahead", chosen.variant,
        )


# Deprecated source-level aliases retained for callers that imported the old
# class names.  Both aliases emit the new policy name and non-oracle metrics.
OracleContext = LookaheadContext


class OraclePolicy(LookaheadReferencePolicy):
    """Deprecated alias for :class:`LookaheadReferencePolicy`."""


class LinUCBPolicy(RepresentationPolicy):
    """Optional contextual-bandit controller; fit on train/validation only."""

    name = "linucb_validation_trained"

    def __init__(self, modes: Sequence[PayloadMode], feature_names: Sequence[str], alpha: float = 0.5):
        self.modes = tuple(modes)
        self.feature_names = tuple(feature_names)
        self.alpha = alpha
        d = len(self.feature_names)
        self.A = {mode: np.eye(d) for mode in self.modes}
        self.b = {mode: np.zeros(d) for mode in self.modes}
        self.frozen = False

    def update(self, features: Mapping[str, float], mode: PayloadMode, reward: float, split: str) -> None:
        if self.frozen:
            raise RuntimeError("Bandit is frozen")
        if split not in {"train", "validation"}:
            raise ValueError("Bandit updates are forbidden outside train/validation")
        x = np.asarray([features[name] for name in self.feature_names], dtype=float)
        self.A[mode] += np.outer(x, x)
        self.b[mode] += reward * x

    def freeze(self) -> None:
        self.frozen = True

    def select(self, obs: ControllerObservation, candidates: tuple[CandidateProfile, ...]) -> TransmissionPlan:
        obs.validate_causal()
        if not self.frozen:
            raise RuntimeError("Freeze the bandit before evaluation")
        values = {
            "goodput_mbps": obs.link.service_rate_bps / 1e6,
            "loss": obs.link.loss_probability,
            "queue_mb": obs.queue.pending_bytes / 1e6,
            "risk": obs.risk,
            "uncertainty": obs.uncertainty,
            "mission_priority": obs.mission_priority,
            "deadline_s": obs.remaining_deadline_us / 1e6,
            "battery": obs.battery_fraction,
            "thermal_scaled": obs.thermal_c / 100,
        }
        x = np.asarray([values[name] for name in self.feature_names], dtype=float)
        available = {candidate.mode: candidate for candidate in candidates}
        scores = {}
        for mode in self.modes:
            if mode not in available:
                continue
            inv = np.linalg.inv(self.A[mode])
            theta = inv @ self.b[mode]
            scores[mode] = float(theta @ x + self.alpha * np.sqrt(x @ inv @ x))
        mode = max(scores, key=scores.get)
        return self._plan(available[mode], "validation_trained_linucb")


def causal_schema_audit() -> dict[str, Any]:
    forbidden = {"truth", "label", "future", "actual_completion", "receiver_outcome"}
    names = {field.name.lower() for field in dataclasses.fields(ControllerObservation)}
    violations = sorted(name for name in names if any(token in name for token in forbidden))
    return {"passed": not violations, "violations": violations, "fields": sorted(names)}
