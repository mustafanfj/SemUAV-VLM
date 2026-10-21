from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .experiments import (
    EXPERIMENT6_REGISTERED_STATE_STRATEGIES,
    EXPERIMENT6_STATE_STRATEGY_VERSION,
    _Experiment6FaultSimulator,
    _experiment6_mapping_message,
    _experiment6_packet,
)
from .network import Direction, SelectiveRepeatSimulator, constant_trace
from .protocol import (
    AppMessage,
    ReliabilityProfile,
    StateKind,
    make_state_delta,
)
from .registry import StudyGateError


def experiment6_state_strategy_comparison(
    *,
    seeds: Sequence[int] = (601, 607, 613),
    scientific: bool,
    registered_strategies: Sequence[str] = EXPERIMENT6_REGISTERED_STATE_STRATEGIES,
    periodic_keyframe_interval: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare three frozen, causally paired semantic-state strategies.

    Each episode starts with a keyframe and an application-ACKed delta. The
    application ACK for update 2 is then deterministically dropped. At update 3
    the sender therefore has only stale confirmed state. Strategies use only
    the sender's application-ACK-derived belief at each capture:

    * ``application_acked_deltas`` continues from its last confirmed base;
    * ``periodic_semantic_keyframes`` forces a keyframe every third update;
    * ``explicit_resynchronization`` sends a keyframe immediately whenever the
      receiver belief is not known.

    Fault schedule, captures, desired states, traces, and seeds are identical.
    The returned detail and summary frames are deterministic ledger artifacts.
    """
    strategies = tuple(str(strategy) for strategy in registered_strategies)
    if strategies != EXPERIMENT6_REGISTERED_STATE_STRATEGIES:
        raise StudyGateError(
            "Experiment 6 state-strategy set/order differs from the frozen comparison"
        )
    if periodic_keyframe_interval != 3:
        raise StudyGateError(
            "Experiment 6 periodic keyframe interval differs from frozen value 3"
        )
    registered_seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not registered_seeds:
        raise StudyGateError("Experiment 6 state-strategy comparison requires a seed")

    update_count = 6
    capture_interval_us = 200_000
    disruption_sequence = 3
    reliability = ReliabilityProfile(
        window_fragments=512,
        retry_limit=2,
        min_rto_us=60_000,
        max_rto_us=60_000,
        delayed_ack_us=0,
    )
    rows: list[dict[str, Any]] = []

    for seed in registered_seeds:
        desired_states = {
            sequence: _experiment6_packet(
                f"exp6-state-update-{sequence}-{seed}",
                (sequence - 1) * capture_interval_us,
            )
            for sequence in range(1, update_count + 1)
        }
        for strategy in strategies:
            message_id_base = 6200
            update_two_message_id = message_id_base + 2
            simulator = _Experiment6FaultSimulator(
                constant_trace(
                    f"exp6-state-uplink-{seed}", Direction.UPLINK,
                    2.0, 2.0, 0.0,
                ),
                constant_trace(
                    f"exp6-state-downlink-{seed}", Direction.DOWNLINK,
                    2.0, 2.0, 0.0,
                ),
                seed=seed,
                decode_us=0,
                drop_application_ack_once={(update_two_message_id, 1)},
            )

            def on_capture(
                sim: SelectiveRepeatSimulator,
                now_us: int,
                capture: Mapping[str, Any],
            ) -> tuple[AppMessage, Mapping[str, Any]]:
                desired_sequence = int(capture["desired_state_seq"])
                desired = desired_states[desired_sequence]
                confirmed_sequence = int(sim.sender_belief.confirmed_seq)
                belief_status = str(sim.sender_belief.status)
                periodic_due = (
                    desired_sequence == 1
                    or (desired_sequence - 1) % periodic_keyframe_interval == 0
                )
                explicit_due = (
                    strategy == "explicit_resynchronization"
                    and desired_sequence > 1
                    and belief_status != "known"
                )
                if desired_sequence == 1 or explicit_due or (
                    strategy == "periodic_semantic_keyframes" and periodic_due
                ):
                    state_kind = StateKind.KEYFRAME
                    base_sequence = None
                    mapping: Mapping[str, Any] = desired
                else:
                    if confirmed_sequence not in desired_states:
                        raise StudyGateError(
                            "Delta strategy lacks the application-ACKed base state"
                        )
                    state_kind = StateKind.DELTA
                    base_sequence = confirmed_sequence
                    mapping = make_state_delta(
                        desired_states[confirmed_sequence], desired,
                    )
                resync_attempt = bool(
                    desired_sequence > 1
                    and state_kind is StateKind.KEYFRAME
                    and belief_status != "known"
                )
                resync_trigger = (
                    "explicit_receiver_belief" if explicit_due
                    else "periodic_keyframe" if resync_attempt
                    else "none"
                )
                message, metadata = _experiment6_mapping_message(
                    message_id=message_id_base + desired_sequence,
                    observation_id=str(desired["observation_id"]),
                    capture_us=now_us,
                    mapping=mapping,
                    scientific=scientific,
                    reliability=reliability,
                    state_kind=state_kind,
                    base_state_seq=base_sequence,
                    resulting_state_seq=desired_sequence,
                    priority=9 if resync_attempt else 5,
                    deadline_after_us=150_000,
                    expiry_after_us=350_000,
                )
                return message, {
                    **metadata,
                    "strategy": strategy,
                    "desired_state_seq": desired_sequence,
                    "sender_confirmed_seq_at_decision": confirmed_sequence,
                    "sender_belief_status_at_decision": belief_status,
                    "transmitted_state_kind": state_kind.value,
                    "transmitted_base_state_seq": base_sequence,
                    "transmitted_resulting_state_seq": desired_sequence,
                    "application_payload_bytes": len(message.payload),
                    "resync_attempt": resync_attempt,
                    "resync_trigger": resync_trigger,
                }

            captures = [
                (
                    (sequence - 1) * capture_interval_us,
                    {"desired_state_seq": sequence},
                )
                for sequence in range(1, update_count + 1)
            ]
            results, events = simulator.run(
                [], captures=captures, capture_handler=on_capture,
            )
            timeout_mask = events["kind"].eq("application_ack_timeout")
            timeout_observed = bool(
                timeout_mask.any()
                and events.loc[timeout_mask, "message_id"].eq(
                    update_two_message_id
                ).any()
            )
            if len(results) != update_count:
                raise StudyGateError(
                    f"State strategy {strategy!r} did not transmit every frozen update"
                )

            receiver_sequence = 0
            strategy_row_indices: list[int] = []
            for desired_sequence in range(1, update_count + 1):
                match = results[results["desired_state_seq"].eq(desired_sequence)]
                if len(match) != 1:
                    raise StudyGateError(
                        f"State strategy {strategy!r} has missing/duplicate update {desired_sequence}"
                    )
                observed = match.iloc[0]
                status = str(observed.apply_status)
                if status in {"applied", "duplicate"}:
                    receiver_sequence = max(
                        receiver_sequence,
                        int(observed.transmitted_resulting_state_seq),
                    )
                sequence_gap = max(0, desired_sequence - receiver_sequence)
                valid_state_delivery = status in {"applied", "duplicate"}
                total_uplink_bytes = int(
                    observed.original_uplink_bytes + observed.retry_uplink_bytes
                )
                resync_attempt = bool(observed.resync_attempt)
                resync_success = bool(
                    resync_attempt and valid_state_delivery and sequence_gap == 0
                )
                row_index = len(rows)
                strategy_row_indices.append(row_index)
                rows.append({
                    "strategy_version": EXPERIMENT6_STATE_STRATEGY_VERSION,
                    "strategy": strategy,
                    "seed": seed,
                    "desired_state_seq": desired_sequence,
                    "capture_us": int(observed.capture_us),
                    "message_id": int(observed.message_id),
                    "sender_confirmed_seq_at_decision": int(
                        observed.sender_confirmed_seq_at_decision
                    ),
                    "sender_belief_status_at_decision": str(
                        observed.sender_belief_status_at_decision
                    ),
                    "state_kind": str(observed.transmitted_state_kind),
                    "base_state_seq": (
                        int(observed.transmitted_base_state_seq)
                        if pd.notna(observed.transmitted_base_state_seq) else np.nan
                    ),
                    "resulting_state_seq": int(
                        observed.transmitted_resulting_state_seq
                    ),
                    "apply_status": status,
                    "full_message_received": bool(observed.full_message_received),
                    "timely_delivery": bool(observed.timely_delivery),
                    "valid_state_delivery": valid_state_delivery,
                    "receiver_state_seq_after_update": receiver_sequence,
                    "receiver_sequence_gap": sequence_gap,
                    "sequence_gap_event": sequence_gap > 0,
                    "base_mismatch_error": status == "base_mismatch",
                    "stale_state_error": status == "stale",
                    "keyframe": str(observed.transmitted_state_kind) == StateKind.KEYFRAME.value,
                    "resync_attempt": resync_attempt,
                    "resync_trigger": str(observed.resync_trigger),
                    "resync_success": resync_success,
                    "application_payload_bytes": int(observed.application_payload_bytes),
                    "original_uplink_bytes": int(observed.original_uplink_bytes),
                    "retry_uplink_bytes": int(observed.retry_uplink_bytes),
                    "total_uplink_bytes": total_uplink_bytes,
                    "total_wire_bytes": int(observed.total_wire_bytes),
                    "resync_wire_bytes": int(observed.total_wire_bytes) if resync_attempt else 0,
                    "wasted_uplink_bytes": (
                        0 if valid_state_delivery else total_uplink_bytes
                    ),
                    "action_ready_us": (
                        int(observed.action_ready_us)
                        if pd.notna(observed.action_ready_us) else np.nan
                    ),
                    "restored_correct_state": bool(
                        desired_sequence >= disruption_sequence
                        and valid_state_delivery and sequence_gap == 0
                    ),
                    "ack_loss_fault_target": desired_sequence == 2,
                    "ack_loss_timeout_observed": timeout_observed,
                    "scientific_wire_format": bool(scientific),
                    "restoration_event": False,
                })

            disruption_us = (disruption_sequence - 1) * capture_interval_us
            restoration_candidates = [
                index for index in strategy_row_indices
                if rows[index]["restored_correct_state"]
                and np.isfinite(rows[index]["action_ready_us"])
            ]
            if restoration_candidates:
                first_restoration = min(
                    restoration_candidates,
                    key=lambda index: int(rows[index]["action_ready_us"]),
                )
                rows[first_restoration]["restoration_event"] = True
                restoration_ms = max(
                    0.0,
                    (int(rows[first_restoration]["action_ready_us"]) - disruption_us)
                    / 1000.0,
                )
            else:
                restoration_ms = np.nan
            for index in strategy_row_indices:
                rows[index]["time_to_restored_correct_state_ms"] = restoration_ms

    detail = pd.DataFrame(rows).sort_values(
        ["strategy", "seed", "desired_state_seq"], kind="mergesort",
    ).reset_index(drop=True)
    expected = {
        (strategy, seed, sequence)
        for strategy in strategies
        for seed in registered_seeds
        for sequence in range(1, update_count + 1)
    }
    observed_keys = set(zip(
        detail["strategy"], detail["seed"].astype(int),
        detail["desired_state_seq"].astype(int),
    ))
    if observed_keys != expected or len(detail) != len(expected):
        raise StudyGateError("Experiment 6 state-strategy coverage is incomplete")
    if not detail["ack_loss_timeout_observed"].astype(bool).all():
        raise StudyGateError("Experiment 6 state-strategy ACK-loss fault was not exercised")
    if (detail["resync_attempt"] & ~detail["keyframe"]).any():
        raise StudyGateError("A registered resynchronization attempt was not a keyframe")

    summary_rows: list[dict[str, Any]] = []
    for strategy in strategies:
        group = detail[detail["strategy"].eq(strategy)]
        transmitted = len(group)
        keyframes = int(group["keyframe"].sum())
        resync_attempts = int(group["resync_attempt"].sum())
        resync_successes = int(group["resync_success"].sum())
        per_seed_restoration = group.groupby("seed", sort=True)[
            "time_to_restored_correct_state_ms"
        ].first()
        summary_rows.append({
            "strategy_version": EXPERIMENT6_STATE_STRATEGY_VERSION,
            "strategy": strategy,
            "registered_seeds": json.dumps(list(registered_seeds)),
            "updates": transmitted,
            "stale_state_errors": int(group["stale_state_error"].sum()),
            "stale_state_error_rate": float(group["stale_state_error"].mean()),
            "base_mismatch_errors": int(group["base_mismatch_error"].sum()),
            "sequence_gap_updates": int(group["sequence_gap_event"].sum()),
            "sequence_gap_rate": float(group["sequence_gap_event"].mean()),
            "mean_sequence_gap": float(group["receiver_sequence_gap"].mean()),
            "maximum_sequence_gap": int(group["receiver_sequence_gap"].max()),
            "keyframes": keyframes,
            "keyframe_frequency": keyframes / transmitted,
            "resync_attempts": resync_attempts,
            "resync_successes": resync_successes,
            "resync_success_rate": (
                resync_successes / resync_attempts if resync_attempts else np.nan
            ),
            "resync_wire_bytes": int(group["resync_wire_bytes"].sum()),
            "wasted_uplink_bytes": int(group["wasted_uplink_bytes"].sum()),
            "wire_delivery_rate": float(group["full_message_received"].mean()),
            "timely_delivery_rate": float(group["timely_delivery"].mean()),
            "valid_state_delivery_rate": float(group["valid_state_delivery"].mean()),
            "restored_correct_state_runs": int(per_seed_restoration.notna().sum()),
            "restoration_success_rate": float(per_seed_restoration.notna().mean()),
            "median_time_to_restored_correct_state_ms": (
                float(per_seed_restoration.dropna().median())
                if per_seed_restoration.notna().any() else np.nan
            ),
            "p95_time_to_restored_correct_state_ms": (
                float(per_seed_restoration.dropna().quantile(0.95))
                if per_seed_restoration.notna().any() else np.nan
            ),
            "ack_loss_fault_exercised": bool(
                group["ack_loss_timeout_observed"].all()
            ),
            "coverage_complete": bool(
                len(group) == len(registered_seeds) * update_count
            ),
            "scientific_wire_format": bool(scientific),
        })
    return detail, pd.DataFrame(summary_rows)
