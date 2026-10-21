from __future__ import annotations

import dataclasses
import enum
import hashlib
import heapq
import json
import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .protocol import (
    ApplicationAck,
    AppMessage,
    ApplyStatus,
    Fragment,
    FragmentAck,
    PayloadMode,
    ReassemblyBuffer,
    ReceiverSemanticState,
    SenderReceiverBelief,
    StateKind,
    decode_app_envelope,
    decode_image,
    decode_mapping,
    fragment_message,
    validate_semantic_packet,
)


class Direction(str, enum.Enum):
    UPLINK = "uplink"
    DOWNLINK = "downlink"


class TraceSemantics(str, enum.Enum):
    RAW_SERVICE = "raw_service"
    EMPIRICAL_GOODPUT = "empirical_goodput"


class EndBehavior(str, enum.Enum):
    HOLD_LAST = "hold_last"
    OUTAGE = "outage"
    TERMINATE = "terminate"


@dataclasses.dataclass(frozen=True)
class LinkSegment:
    start_us: int
    end_us: int
    service_rate_bps: int
    base_one_way_delay_us: int
    jitter_parameter_us: int = 0
    packet_loss_parameter: float = 0.0
    snr_db: float | None = None
    distance_m: float | None = None
    queue_capacity_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.start_us < 0 or self.end_us <= self.start_us:
            raise ValueError("Link segments require an increasing non-negative interval")
        if self.service_rate_bps < 0:
            raise ValueError("Service rate cannot be negative")
        if self.base_one_way_delay_us < 0 or self.jitter_parameter_us < 0:
            raise ValueError("Delay and jitter cannot be negative")
        if not 0 <= self.packet_loss_parameter <= 1:
            raise ValueError("Loss must lie in [0, 1]")
        if self.queue_capacity_bytes is not None and self.queue_capacity_bytes <= 0:
            raise ValueError("Segment queue capacity must be positive when configured")


@dataclasses.dataclass(frozen=True)
class SerializationPiece:
    segment_index: int
    start_us: int
    end_us: int
    bits: int


@dataclasses.dataclass(frozen=True)
class SerializationSpan:
    start_us: int
    end_us: int
    bits: int
    pieces: tuple[SerializationPiece, ...]


@dataclasses.dataclass(frozen=True)
class LinkTrace:
    trace_id: str
    direction: Direction
    semantics: TraceSemantics
    segments: tuple[LinkSegment, ...]
    end_behavior: EndBehavior = EndBehavior.HOLD_LAST

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("A trace needs at least one segment")
        previous_end = self.segments[0].start_us
        for segment in self.segments:
            if segment.start_us != previous_end:
                raise ValueError("Trace segments must be contiguous and ordered")
            previous_end = segment.end_us

    def _index_at(self, time_us: int) -> int:
        starts = [segment.start_us for segment in self.segments]
        index = int(np.searchsorted(starts, time_us, side="right") - 1)
        if 0 <= index < len(self.segments) and time_us < self.segments[index].end_us:
            return index
        if time_us < self.segments[0].start_us:
            raise ValueError("Time precedes trace start")
        if self.end_behavior is EndBehavior.HOLD_LAST:
            return len(self.segments) - 1
        if self.end_behavior is EndBehavior.OUTAGE:
            return -1
        raise EOFError(f"Trace {self.trace_id} ended at {self.segments[-1].end_us}")

    def condition_at(self, time_us: int) -> LinkSegment:
        index = self._index_at(time_us)
        if index >= 0:
            segment = self.segments[index]
            if time_us >= segment.end_us and self.end_behavior is EndBehavior.HOLD_LAST:
                return dataclasses.replace(segment, start_us=segment.end_us, end_us=2**62)
            return segment
        last = self.segments[-1]
        return LinkSegment(
            last.end_us, 2**62, 0, last.base_one_way_delay_us,
            last.jitter_parameter_us, 1.0,
            queue_capacity_bytes=last.queue_capacity_bytes,
        )

    def serialize(self, start_us: int, bits: int) -> SerializationSpan:
        if bits < 0:
            raise ValueError("bits must be non-negative")
        if bits == 0:
            return SerializationSpan(start_us, start_us, 0, ())
        remaining = bits
        now = start_us
        pieces: list[SerializationPiece] = []
        zero_segments = 0
        while remaining > 0:
            index = self._index_at(now)
            segment = self.condition_at(now)
            if segment.service_rate_bps == 0:
                zero_segments += 1
                if segment.end_us >= 2**61 or zero_segments > len(self.segments) + 1:
                    raise TimeoutError(f"Trace {self.trace_id} has no future service")
                now = segment.end_us
                continue
            zero_segments = 0
            available_us = segment.end_us - now
            capacity_bits = (segment.service_rate_bps * available_us) // 1_000_000
            if capacity_bits <= 0:
                now = segment.end_us
                continue
            served = min(remaining, capacity_bits)
            duration_us = math.ceil(served * 1_000_000 / segment.service_rate_bps)
            end = now + duration_us
            pieces.append(SerializationPiece(index, now, end, served))
            now = end
            remaining -= served
            if remaining > 0 and now < segment.end_us:
                now = segment.end_us
        return SerializationSpan(start_us, now, bits, tuple(pieces))

    def effective_loss(self, span: SerializationSpan) -> float:
        if self.semantics is TraceSemantics.EMPIRICAL_GOODPUT:
            return 0.0
        survival = 1.0
        for piece in span.pieces:
            segment = self.condition_at(piece.start_us)
            weight = piece.bits / max(1, span.bits)
            survival *= (1.0 - segment.packet_loss_parameter) ** weight
        return float(np.clip(1.0 - survival, 0.0, 1.0))


def _typed_key_bytes(parts: Sequence[Any]) -> bytes:
    encoded = []
    for part in parts:
        blob = json.dumps({"type": type(part).__name__, "value": part}, sort_keys=True, separators=(",", ":")).encode()
        encoded.append(len(blob).to_bytes(4, "big") + blob)
    return b"".join(encoded)


def keyed_uniform(*parts: Any) -> float:
    integer = int.from_bytes(hashlib.sha256(_typed_key_bytes(parts)).digest()[:8], "big")
    return (integer + 0.5) / 2**64


def constant_trace(
    trace_id: str,
    direction: Direction,
    rate_mbps: float,
    one_way_ms: float,
    loss: float,
    duration_s: float = 3600,
    jitter_ms: float = 0.0,
) -> LinkTrace:
    return LinkTrace(
        trace_id,
        direction,
        TraceSemantics.RAW_SERVICE,
        (LinkSegment(
            0, int(duration_s * 1e6), int(rate_mbps * 1e6),
            int(one_way_ms * 1e3), int(jitter_ms * 1e3), loss,
        ),),
    )


def dynamic_trace_family(family: str, seed: int, duration_s: int = 120, step_s: int = 1) -> LinkTrace:
    rng = np.random.default_rng(seed)
    times = np.arange(0, duration_s + step_s, step_s)
    if family == "sudden_drop":
        rates = np.full(len(times) - 1, 8.0)
        rates[(times[:-1] >= 35) & (times[:-1] < 70)] = 0.25
    elif family == "ramp":
        rates = np.linspace(0.1, 20.0, len(times) - 1)
    elif family == "outage":
        rates = np.full(len(times) - 1, 4.0)
        rates[(times[:-1] >= 45) & (times[:-1] < 60)] = 0.0
    elif family == "markov":
        rates, state = [], 1
        for _ in times[:-1]:
            if rng.random() < (0.12 if state else 0.35):
                state = 1 - state
            rates.append(8.0 if state else 0.3)
        rates = np.asarray(rates)
    elif family == "fading":
        raw = np.exp(rng.normal(np.log(2.0), 0.9, len(times) - 1))
        rates = np.clip(np.convolve(raw, np.ones(5) / 5, mode="same"), 0.1, 20.0)
    elif family == "recovery":
        rates = np.full(len(times) - 1, 0.15)
        rates[times[:-1] >= 55] = 10.0
    else:
        raise ValueError(f"Unknown trace family {family}")
    segments = []
    for index, rate in enumerate(rates):
        loss = float(np.clip(0.12 / max(rate, 0.1), 0.002, 0.25)) if rate > 0 else 1.0
        rtt_ms = float(np.clip(30 + 60 / max(rate, 0.1), 30, 300))
        segments.append(
            LinkSegment(
                int(times[index] * 1e6),
                int(times[index + 1] * 1e6),
                int(rate * 1e6),
                int(rtt_ms * 500),
                int(rtt_ms * 50),
                loss,
            )
        )
    return LinkTrace(f"{family}-{seed}", Direction.UPLINK, TraceSemantics.RAW_SERVICE, tuple(segments))


class EventType(str, enum.Enum):
    CAPTURE = "capture"
    STATE_PENDING = "state_pending"
    ENQUEUE = "enqueue"
    TX_END = "tx_end"
    DATA_ARRIVE = "data_arrive"
    FRAGMENT_ACK_ARRIVE = "fragment_ack_arrive"
    APPLICATION_DONE = "application_done"
    APPLICATION_ACK_ARRIVE = "application_ack_arrive"
    APPLICATION_ACK_TIMEOUT = "application_ack_timeout"
    RTO_EXPIRE = "rto_expire"
    MESSAGE_EXPIRE = "message_expire"
    DISPATCH = "dispatch"


class _ModelVersionMismatch(ValueError):
    """Receiver-side incompatibility with the registered model revision."""


PHASE = {
    EventType.FRAGMENT_ACK_ARRIVE: 0,
    EventType.APPLICATION_ACK_ARRIVE: 0,
    # A feedback arrival at the exact timeout wins; the timeout in turn is
    # visible to a controller capture at the same integer timestamp.
    EventType.APPLICATION_ACK_TIMEOUT: 1,
    EventType.CAPTURE: 2,
    EventType.STATE_PENDING: 3,
    EventType.ENQUEUE: 4,
    EventType.DATA_ARRIVE: 5,
    EventType.APPLICATION_DONE: 6,
    EventType.TX_END: 7,
    EventType.RTO_EXPIRE: 8,
    EventType.MESSAGE_EXPIRE: 9,
    EventType.DISPATCH: 10,
}


@dataclasses.dataclass(order=True)
class Event:
    time_us: int
    phase: int
    sequence: int
    kind: EventType = dataclasses.field(compare=False)
    data: Any = dataclasses.field(compare=False)


@dataclasses.dataclass(frozen=True)
class FragmentAttempt:
    message_id: int
    fragment_index: int
    attempt: int


@dataclasses.dataclass
class _MessageRuntime:
    message: AppMessage
    fragments: tuple[Fragment, ...]
    queued: set[int] = dataclasses.field(default_factory=set)
    sent_attempts: dict[int, int] = dataclasses.field(default_factory=lambda: defaultdict(int))
    acked: set[int] = dataclasses.field(default_factory=set)
    received: set[int] = dataclasses.field(default_factory=set)
    reassembly: ReassemblyBuffer | None = None
    first_tx_us: int | None = None
    last_rx_us: int | None = None
    action_ready_us: int | None = None
    apply_status: str | None = None
    retry_exhausted: bool = False
    original_wire_bytes: int = 0
    retry_wire_bytes: int = 0
    downlink_control_bytes: int = 0
    duplicate_payload_bytes: int = 0
    sent_end_us: dict[tuple[int, int], int] = dataclasses.field(default_factory=dict)
    application_ack_sends: int = 0
    application_ack_timeout_us: int | None = None
    queue_overflow: bool = False


class SelectiveRepeatSimulator:
    """Integer-time, fragment-level, paired-randomness network simulator."""

    def __init__(self, uplink: LinkTrace, downlink: LinkTrace, mtu_bytes: int = 1200, seed: int = 0, decode_us: int = 5_000, initial_rate_bps: float = 1_000_000, initial_rtt_us: float = 200_000, queue_capacity_bytes: int | None = None, loss_model: str = "iid", burst_recovery_probability: float = 0.2, burst_slot_us: int = 100_000):
        if uplink.direction is not Direction.UPLINK or downlink.direction is not Direction.DOWNLINK:
            raise ValueError("Provide distinct uplink and downlink traces")
        self.uplink = uplink
        self.downlink = downlink
        self.mtu_bytes = mtu_bytes
        self.seed = seed
        self.decode_us = decode_us
        if queue_capacity_bytes is not None and queue_capacity_bytes <= 0:
            raise ValueError("queue_capacity_bytes must be positive when configured")
        self.queue_capacity_bytes = queue_capacity_bytes
        if loss_model not in {"iid", "gilbert_elliott"}:
            raise ValueError("loss_model must be 'iid' or 'gilbert_elliott'")
        if not 0 < burst_recovery_probability <= 1:
            raise ValueError("burst_recovery_probability must lie in (0, 1]")
        if burst_slot_us <= 0:
            raise ValueError("burst_slot_us must be positive")
        self.loss_model = loss_model
        self.burst_recovery_probability = float(burst_recovery_probability)
        self.burst_slot_us = int(burst_slot_us)
        self._burst_states: dict[str, list[bool]] = {}
        self._burst_target_loss: dict[str, float] = {}
        self._events: list[Event] = []
        self._event_seq = 0
        self._queue: list[tuple[int, int, int, int, FragmentAttempt]] = []
        self._queue_seq = 0
        self._now = 0
        self._uplink_busy_until = 0
        self._downlink_busy_until = 0
        self._uplink_inflight_bytes = 0
        self._uplink_inflight_attempt: FragmentAttempt | None = None
        self._runtimes: dict[int, _MessageRuntime] = {}
        self.receiver_state = ReceiverSemanticState()
        self.sender_belief = SenderReceiverBelief()
        self.event_log: list[dict[str, Any]] = []
        self.message_metadata: dict[int, dict[str, Any]] = {}
        self._capture_handler: Any = None
        self._feedback_handler: Any = None
        self._estimate_rate_bps = float(initial_rate_bps)
        self._estimate_rtt_us = float(initial_rtt_us)
        self._rttvar_us = float(initial_rtt_us) / 2
        self._last_rate_ack_us: int | None = None
        self._rate_ack_bits = 0
        self._estimate_loss = 0.05
        self._feedback_samples = 0
        # These sets are sender-local causal state.  A semantic message remains
        # pending until a cumulative application ACK proves that state (or a
        # later state) was applied.  Timed-out IDs stay pending so a late ACK can
        # still repair the belief, but force the public status to ``uncertain``.
        self._pending_state_message_ids: set[int] = set()
        self._timed_out_state_message_ids: set[int] = set()

    def _schedule(self, time_us: int, kind: EventType, data: Any = None) -> None:
        if time_us < self._now:
            raise AssertionError("Event scheduled in the past")
        self._event_seq += 1
        heapq.heappush(self._events, Event(time_us, PHASE[kind], self._event_seq, kind, data))

    def _log(self, kind: str, **fields: Any) -> None:
        self.event_log.append({"time_us": self._now, "kind": kind, **fields})

    def _send_application_ack(self, runtime: _MessageRuntime, *, repeat: bool) -> None:
        """Send a fresh cumulative application ACK for a completed message."""
        if runtime.action_ready_us is None or runtime.apply_status is None:
            raise AssertionError("Cannot ACK an application result before completion")
        runtime.application_ack_sends += 1
        ack = ApplicationAck(
            runtime.message.message_id,
            ApplyStatus(runtime.apply_status),
            self.receiver_state.applied_seq,
            runtime.action_ready_us,
            self._now,
            semantic_state_message=runtime.message.state_kind is not StateKind.NONE,
            state_update_us=self.receiver_state.last_update_us,
        )
        runtime.downlink_control_bytes += 64
        self._send_downlink(
            64,
            self._now,
            (runtime.message.message_id, "app_ack", runtime.application_ack_sends),
            EventType.APPLICATION_ACK_ARRIVE,
            ack,
        )
        self._log(
            "application_ack_repeat" if repeat else "application_ack_send",
            message_id=runtime.message.message_id,
            send_number=runtime.application_ack_sends,
            receiver_state_seq=ack.receiver_state_seq,
        )

    def _resolve_cumulative_application_ack(self, ack: ApplicationAck) -> None:
        """Apply an ACK to all sender-pending states it causally confirms."""
        if not ack.semantic_state_message:
            return
        for message_id in list(self._pending_state_message_ids):
            runtime = self._runtimes[message_id]
            resulting_seq = runtime.message.resulting_state_seq
            if message_id == ack.message_id or (
                resulting_seq is not None and resulting_seq <= ack.receiver_state_seq
            ):
                self._pending_state_message_ids.discard(message_id)
                self._timed_out_state_message_ids.discard(message_id)

    def _push_attempt(self, runtime: _MessageRuntime, index: int, attempt: int) -> None:
        if index in runtime.acked or self._now > runtime.message.state_expiry_us:
            return
        runtime.queued.add(index)
        self._queue_seq += 1
        key = (-runtime.message.priority, runtime.message.action_deadline_us, self._queue_seq, index)
        heapq.heappush(self._queue, (*key, FragmentAttempt(runtime.message.message_id, index, attempt)))

    def _fill_window(self, runtime: _MessageRuntime) -> None:
        in_flight = sum(1 for i, count in runtime.sent_attempts.items() if count and i not in runtime.acked)
        available = max(0, runtime.message.reliability.window_fragments - in_flight)
        for index in range(len(runtime.fragments)):
            if available <= 0:
                break
            if index not in runtime.acked and runtime.sent_attempts[index] == 0 and index not in runtime.queued:
                self._push_attempt(runtime, index, 1)
                available -= 1

    def _dispatch(self) -> None:
        if self._now < self._uplink_busy_until:
            return
        while self._queue:
            *_, attempt = heapq.heappop(self._queue)
            runtime = self._runtimes[attempt.message_id]
            runtime.queued.discard(attempt.fragment_index)
            if attempt.fragment_index in runtime.acked or self._now > runtime.message.state_expiry_us:
                continue
            fragment = runtime.fragments[attempt.fragment_index]
            if self._now < runtime.message.ready_us:
                raise AssertionError("Fragment sent before message readiness")
            bits = fragment.ip_packet_bytes * 8
            span = self.uplink.serialize(self._now, bits)
            self._uplink_busy_until = span.end_us
            self._uplink_inflight_bytes = fragment.ip_packet_bytes
            self._uplink_inflight_attempt = attempt
            runtime.sent_attempts[attempt.fragment_index] = attempt.attempt
            runtime.first_tx_us = runtime.first_tx_us if runtime.first_tx_us is not None else self._now
            if attempt.attempt == 1:
                runtime.original_wire_bytes += fragment.ip_packet_bytes
            else:
                runtime.retry_wire_bytes += fragment.ip_packet_bytes
            self._log("fragment_tx_start", message_id=attempt.message_id, fragment_index=attempt.fragment_index, attempt=attempt.attempt, bytes=fragment.ip_packet_bytes)
            self._schedule(span.end_us, EventType.TX_END, (attempt, span))
            return

    def link_estimate(self, now_us: int):
        from .control import LinkEstimate
        rto_us = float(np.clip(
            self._estimate_rtt_us + 4 * self._rttvar_us,
            200_000, 2_000_000,
        ))
        return LinkEstimate(
            now_us, self._feedback_samples, self._estimate_rate_bps,
            self._estimate_rtt_us / 2, self._estimate_loss, rto_us,
        )

    def queue_snapshot(self, now_us: int):
        from .control import QueueSnapshot
        pending = [0] * 11
        messages = [0] * 11
        oldest: list[int | None] = [None] * 11
        active_message_priorities: dict[int, int] = {}
        outstanding = 0
        for runtime in self._runtimes.values():
            if runtime.queue_overflow:
                continue
            if now_us < runtime.message.ready_us:
                # A registered future message has not reached the sender queue
                # yet and must not influence a causal controller observation or
                # the admission decision for another message.
                continue
            if now_us > runtime.message.state_expiry_us:
                continue
            priority = int(np.clip(runtime.message.priority, 0, 10))
            message_pending = False
            for index, fragment in enumerate(runtime.fragments):
                if index in runtime.acked:
                    continue
                attempt = FragmentAttempt(
                    runtime.message.message_id, index,
                    runtime.sent_attempts.get(index, 0),
                )
                is_inflight = (
                    self._uplink_inflight_attempt is not None
                    and self._uplink_inflight_attempt.message_id == attempt.message_id
                    and self._uplink_inflight_attempt.fragment_index == index
                    and self._uplink_busy_until > now_us
                )
                if is_inflight:
                    continue
                if index in runtime.queued or runtime.sent_attempts.get(index, 0) == 0:
                    # Includes the unscheduled tail beyond the current ARQ
                    # window as well as explicit first-send/retry queue entries.
                    pending[priority] += fragment.ip_packet_bytes
                    message_pending = True
                else:
                    outstanding += fragment.ip_packet_bytes
            if message_pending:
                active_message_priorities[runtime.message.message_id] = priority
                oldest[priority] = runtime.message.ready_us if oldest[priority] is None else min(oldest[priority], runtime.message.ready_us)
        for priority in active_message_priorities.values():
            messages[priority] += 1
        inflight = self._uplink_inflight_bytes if self._uplink_busy_until > now_us else 0
        return QueueSnapshot(
            now_us, tuple(pending), inflight, outstanding,
            tuple(oldest), tuple(messages),
        )

    def _register_message(self, message: AppMessage, metadata: Mapping[str, Any] | None = None) -> None:
        if message.message_id in self._runtimes:
            raise ValueError(f"Duplicate message_id {message.message_id}")
        fragments = fragment_message(message, self.mtu_bytes)
        existing_queue_bytes = self.queue_snapshot(self._now).pending_bytes if self._runtimes else 0
        runtime = _MessageRuntime(message, fragments)
        runtime.reassembly = ReassemblyBuffer(len(fragments), message.wire_payload_hash[:8])
        self._runtimes[message.message_id] = runtime
        self.message_metadata[message.message_id] = dict(metadata or {})
        admitted_bytes = sum(fragment.ip_packet_bytes for fragment in fragments)
        segment_capacity = self.uplink.condition_at(self._now).queue_capacity_bytes
        effective_capacity = (
            segment_capacity if segment_capacity is not None else self.queue_capacity_bytes
        )
        if effective_capacity is not None and existing_queue_bytes + admitted_bytes > effective_capacity:
            runtime.queue_overflow = True
            runtime.retry_exhausted = True
            self._log(
                "queue_overflow_drop", message_id=message.message_id,
                queued_bytes=existing_queue_bytes, offered_bytes=admitted_bytes,
                capacity_bytes=effective_capacity,
            )
            self._schedule(message.state_expiry_us, EventType.MESSAGE_EXPIRE, message.message_id)
            return
        if message.state_kind is not StateKind.NONE:
            # Registration can happen before a future capture when callers pass
            # an initial message list, so expose PENDING only at capture time.
            self._schedule(max(self._now, message.capture_us), EventType.STATE_PENDING, message.message_id)
            runtime.application_ack_timeout_us = min(
                message.state_expiry_us,
                message.ready_us + message.reliability.max_rto_us,
            )
            self._schedule(
                max(self._now, runtime.application_ack_timeout_us),
                EventType.APPLICATION_ACK_TIMEOUT,
                message.message_id,
            )
        self._schedule(message.ready_us, EventType.ENQUEUE, message.message_id)
        self._schedule(message.state_expiry_us, EventType.MESSAGE_EXPIRE, message.message_id)

    def _loss_and_delay(self, trace: LinkTrace, span: SerializationSpan, key: tuple[Any, ...]) -> tuple[bool, int]:
        loss = trace.effective_loss(span)
        if self.loss_model == "iid":
            lost = keyed_uniform(self.seed, trace.trace_id, *key, "loss") < loss
        else:
            # Registered Gilbert-Elliott special case: the good state has zero
            # loss, the bad state has unit loss, and the stationary bad-state
            # probability equals the configured static-grid loss.  State evolves
            # in exogenous fixed time slots, not per transmitted packet, so two
            # representations probing the same trace time observe the same state.
            target_loss = float(np.clip(loss, 0.0, 1.0))
            previous_target = self._burst_target_loss.setdefault(trace.trace_id, target_loss)
            if not math.isclose(previous_target, target_loss, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    "Gilbert-Elliott mode requires a constant loss parameter per trace"
                )
            states = self._burst_states.get(trace.trace_id)
            if states is None:
                initial_bad = (
                    keyed_uniform(
                        self.seed, trace.trace_id, "gilbert_elliott", "initial",
                    ) < target_loss
                )
                states = [initial_bad]
                self._burst_states[trace.trace_id] = states
            slot = max(span.start_us, span.end_us - 1) // self.burst_slot_us
            recovery = self.burst_recovery_probability
            if target_loss >= 1.0:
                good_to_bad = 1.0
            elif target_loss <= 0.0:
                good_to_bad = 0.0
            else:
                good_to_bad = min(1.0, recovery * target_loss / (1.0 - target_loss))
            while len(states) <= slot:
                transition_slot = len(states)
                draw = keyed_uniform(
                    self.seed, trace.trace_id, "gilbert_elliott",
                    "transition", transition_slot,
                )
                states.append(
                    draw >= recovery if states[-1] else draw < good_to_bad
                )
            lost = states[slot]
        # Propagation belongs to the segment that serialized the final bit; an
        # exact boundary must not borrow the next segment's delay/jitter.
        segment = trace.condition_at(max(span.start_us, span.end_us - 1))
        jitter = round((2 * keyed_uniform(self.seed, trace.trace_id, *key, "jitter") - 1) * segment.jitter_parameter_us)
        return lost, max(0, segment.base_one_way_delay_us + jitter)

    def _send_downlink(self, bytes_on_wire: int, ready_us: int, key: tuple[Any, ...], arrival_kind: EventType, data: Any) -> None:
        start = max(ready_us, self._downlink_busy_until)
        span = self.downlink.serialize(start, bytes_on_wire * 8)
        self._downlink_busy_until = span.end_us
        lost, delay = self._loss_and_delay(self.downlink, span, key)
        if not lost:
            self._schedule(span.end_us + delay, arrival_kind, data)
        self._log("downlink_control", bytes=bytes_on_wire, lost=lost, control=arrival_kind.value)

    def run(
        self,
        messages: Sequence[AppMessage],
        *,
        captures: Sequence[tuple[int, Any]] = (),
        capture_handler: Any = None,
        feedback_handler: Any = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        message_ids = [message.message_id for message in messages]
        if len(message_ids) != len(set(message_ids)):
            raise ValueError("message_id values must be unique")
        self._capture_handler = capture_handler
        self._feedback_handler = feedback_handler
        for message in messages:
            self._register_message(message)
        for capture_us, capture_data in captures:
            self._schedule(int(capture_us), EventType.CAPTURE, capture_data)
        last_time = -1
        while self._events:
            event = heapq.heappop(self._events)
            if event.time_us < last_time:
                raise AssertionError("Event timestamps decreased")
            self._now = event.time_us
            last_time = self._now
            kind, data = event.kind, event.data
            if kind is EventType.CAPTURE:
                if self._capture_handler is None:
                    raise AssertionError("Capture event has no handler")
                created = self._capture_handler(self, self._now, data)
                if created is not None:
                    message, metadata = created
                    self._register_message(message, metadata)
                    self._log("controller_decision", message_id=message.message_id, mode=message.mode.name, **{key: value for key, value in metadata.items() if key in {"policy", "variant", "reason_code"}})
            elif kind is EventType.STATE_PENDING:
                runtime = self._runtimes[data]
                if runtime.message.state_kind is StateKind.NONE:
                    raise AssertionError("Stateless message entered semantic pending state")
                self._pending_state_message_ids.add(runtime.message.message_id)
                self.sender_belief.status = "pending"
                self._log(
                    "application_ack_pending",
                    message_id=runtime.message.message_id,
                    resulting_state_seq=runtime.message.resulting_state_seq,
                    timeout_us=runtime.application_ack_timeout_us,
                )
            elif kind is EventType.ENQUEUE:
                runtime = self._runtimes[data]
                self._fill_window(runtime)
                self._log("message_enqueue", message_id=data, queue_fragments=len(self._queue))
                self._schedule(self._now, EventType.DISPATCH)
            elif kind is EventType.DISPATCH:
                self._dispatch()
            elif kind is EventType.TX_END:
                attempt, span = data
                runtime = self._runtimes[attempt.message_id]
                fragment = runtime.fragments[attempt.fragment_index]
                if self._uplink_inflight_attempt == attempt:
                    self._uplink_inflight_bytes = 0
                    self._uplink_inflight_attempt = None
                runtime.sent_end_us[(attempt.fragment_index, attempt.attempt)] = self._now
                lost, delay = self._loss_and_delay(self.uplink, span, (runtime.message.logical_observation_id, attempt.fragment_index, attempt.attempt, "data"))
                self._log("fragment_tx_end", message_id=attempt.message_id, fragment_index=attempt.fragment_index, attempt=attempt.attempt, lost=lost)
                if not lost:
                    # Cross the actual byte codec boundary.  Receiver-side CRC,
                    # version, length, ID, and range checks must succeed before
                    # reassembly.
                    self._schedule(
                        self._now + delay, EventType.DATA_ARRIVE,
                        (attempt, fragment.to_bytes()),
                    )
                rto = int(np.clip(
                    (self._estimate_rtt_us + 4 * self._rttvar_us) * (2 ** max(0, attempt.attempt - 1)),
                    runtime.message.reliability.min_rto_us,
                    runtime.message.reliability.max_rto_us,
                ))
                self._schedule(self._now + rto, EventType.RTO_EXPIRE, attempt)
                self._schedule(self._now, EventType.DISPATCH)
            elif kind is EventType.DATA_ARRIVE:
                attempt, datagram = data
                runtime = self._runtimes[attempt.message_id]
                try:
                    fragment = Fragment.from_bytes(datagram)
                    if fragment.message_id != attempt.message_id or fragment.fragment_index != attempt.fragment_index:
                        raise ValueError("Received fragment identity differs from scheduled attempt")
                except Exception as exc:
                    self._log(
                        "fragment_drop_invalid", message_id=attempt.message_id,
                        fragment_index=attempt.fragment_index,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    continue
                duplicate = attempt.fragment_index in runtime.received
                if duplicate:
                    runtime.duplicate_payload_bytes += len(fragment.fragment_payload)
                runtime.received.add(attempt.fragment_index)
                runtime.last_rx_us = self._now
                complete = runtime.reassembly.add(fragment) if runtime.reassembly else False
                ack = FragmentAck(attempt.message_id, (attempt.fragment_index,), self._now)
                runtime.downlink_control_bytes += 48
                self._send_downlink(48, self._now + runtime.message.reliability.delayed_ack_us, (attempt.message_id, attempt.fragment_index, attempt.attempt, "frag_ack"), EventType.FRAGMENT_ACK_ARRIVE, ack)
                self._log("fragment_receive", message_id=attempt.message_id, fragment_index=attempt.fragment_index, duplicate=duplicate)
                if duplicate and runtime.action_ready_us is not None:
                    # A retransmitted fragment is also an implicit application-
                    # status request.  Repeating a fresh cumulative ACK lets the
                    # sender recover when the original application ACK was lost.
                    self._send_application_ack(runtime, repeat=True)
                if complete and runtime.action_ready_us is None:
                    receiver_latency_us = int(
                        self.message_metadata.get(attempt.message_id, {}).get(
                            "receiver_latency_us", self.decode_us
                        )
                    )
                    if receiver_latency_us < 0:
                        raise ValueError("receiver_latency_us cannot be negative")
                    self._schedule(self._now + receiver_latency_us, EventType.APPLICATION_DONE, attempt.message_id)
            elif kind is EventType.FRAGMENT_ACK_ARRIVE:
                ack: FragmentAck = data
                runtime = self._runtimes[ack.message_id]
                newly_acked = set(ack.received_indices).difference(runtime.acked)
                runtime.acked.update(ack.received_indices)
                for index in newly_acked:
                    attempt_number = runtime.sent_attempts.get(index, 1)
                    sent_end = runtime.sent_end_us.get((index, attempt_number))
                    # Karn's algorithm: an ACK received after retransmission is
                    # not used for RTT/rate estimation because it is ambiguous.
                    if attempt_number == 1 and sent_end is not None and self._now > sent_end:
                        rtt_sample = self._now - sent_end
                        alpha, beta = 0.125, 0.25
                        self._rttvar_us = (1 - beta) * self._rttvar_us + beta * abs(self._estimate_rtt_us - rtt_sample)
                        self._estimate_rtt_us = (1 - alpha) * self._estimate_rtt_us + alpha * rtt_sample
                        ack_bits = runtime.fragments[index].ip_packet_bytes * 8
                        if self._last_rate_ack_us is None:
                            self._last_rate_ack_us = self._now
                            self._rate_ack_bits = 0
                        else:
                            self._rate_ack_bits += ack_bits
                            if self._now > self._last_rate_ack_us:
                                rate_sample = self._rate_ack_bits * 1_000_000 / (self._now - self._last_rate_ack_us)
                                self._estimate_rate_bps = 0.8 * self._estimate_rate_bps + 0.2 * rate_sample
                                self._last_rate_ack_us = self._now
                                self._rate_ack_bits = 0
                        self._estimate_loss = (1 - alpha) * self._estimate_loss
                        self._feedback_samples += 1
                self._fill_window(runtime)
                self._log("fragment_ack_arrive", message_id=ack.message_id, acked=list(ack.received_indices))
                self._schedule(self._now, EventType.DISPATCH)
            elif kind is EventType.RTO_EXPIRE:
                attempt = data
                runtime = self._runtimes[attempt.message_id]
                if attempt.fragment_index in runtime.acked or runtime.sent_attempts[attempt.fragment_index] != attempt.attempt:
                    continue
                if attempt.attempt <= runtime.message.reliability.retry_limit and self._now <= runtime.message.state_expiry_us:
                    self._push_attempt(runtime, attempt.fragment_index, attempt.attempt + 1)
                    self._log("fragment_retry", message_id=attempt.message_id, fragment_index=attempt.fragment_index, attempt=attempt.attempt + 1)
                    self._schedule(self._now, EventType.DISPATCH)
                else:
                    runtime.retry_exhausted = True
                    self._estimate_loss = min(1.0, 0.8 * self._estimate_loss + 0.2)
                    self._log("retry_exhausted", message_id=attempt.message_id, fragment_index=attempt.fragment_index)
            elif kind is EventType.APPLICATION_DONE:
                runtime = self._runtimes[data]
                if runtime.action_ready_us is not None:
                    continue
                try:
                    wire_payload = runtime.reassembly.assemble() if runtime.reassembly else b""
                    envelope = decode_app_envelope(wire_payload)
                    if (
                        envelope.mode is not runtime.message.mode
                        or envelope.state_kind is not runtime.message.state_kind
                        or envelope.priority != runtime.message.priority
                        or envelope.capture_us != runtime.message.capture_us
                        or envelope.action_deadline_us != runtime.message.action_deadline_us
                        or envelope.state_expiry_us != runtime.message.state_expiry_us
                        or envelope.base_state_seq != runtime.message.base_state_seq
                        or envelope.resulting_state_seq != runtime.message.resulting_state_seq
                    ):
                        raise ValueError("Received envelope metadata differs from sender record")
                    payload = envelope.payload
                    wire_format = self.message_metadata.get(runtime.message.message_id, {}).get("wire_format")
                    try:
                        if runtime.message.mode is PayloadMode.L3_SEMANTICS_ROI:
                            if len(payload) < 4 or wire_format not in {"cbor", "json-debug"}:
                                raise ValueError("Malformed L3 semantic+ROI payload")
                            semantic_size = int.from_bytes(payload[:4], "big")
                            if semantic_size <= 0 or 4 + semantic_size > len(payload):
                                raise ValueError("Invalid L3 semantic section length")
                            decoded = decode_mapping(payload[4 : 4 + semantic_size], wire_format)
                            validate_semantic_packet(decoded)
                            roi_payload = payload[4 + semantic_size :]
                            if not roi_payload:
                                raise ValueError("L3 evidence ROI is empty")
                            decode_image(roi_payload)
                            declared_hashes = {
                                str(evidence.get("integrity_sha256"))
                                for evidence in decoded.get("evidence", [])
                                if isinstance(evidence, Mapping) and evidence.get("integrity_sha256")
                            }
                            if declared_hashes and hashlib.sha256(roi_payload).hexdigest() not in declared_hashes:
                                raise ValueError("L3 evidence ROI integrity hash mismatch")
                        else:
                            decoded = decode_mapping(payload, wire_format) if wire_format in {"cbor", "json-debug"} else json.loads(payload.decode("utf-8"))
                        if not isinstance(decoded, dict):
                            raise ValueError("Decoded application payload is not a mapping")
                        if runtime.message.mode is PayloadMode.L2_SEMANTICS:
                            if runtime.message.state_kind is StateKind.DELTA:
                                if set(decoded) != {"set", "delete"} or not isinstance(decoded["set"], dict) or not isinstance(decoded["delete"], list):
                                    raise ValueError("Malformed semantic delta")
                            else:
                                validate_semantic_packet(decoded)
                        elif runtime.message.mode is PayloadMode.L1_ALERT:
                            required_alert = {
                                "schema_version", "observation_id", "mission",
                                "capture_time_us", "telemetry", "event", "risk",
                                "uncertainty", "calibrated_correctness_probability",
                                "recommended_action", "sufficient_evidence",
                                "provenance", "confidence_provenance",
                            }
                            if required_alert.difference(decoded):
                                raise ValueError("L1 alert is missing required fields")
                            probability = float(decoded["calibrated_correctness_probability"])
                            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                                raise ValueError("L1 calibrated probability is outside [0, 1]")
                        elif runtime.message.mode is PayloadMode.L0_HEARTBEAT:
                            required_heartbeat = {
                                "schema_version", "observation_id", "mission",
                                "capture_time_us", "telemetry",
                            }
                            if required_heartbeat.difference(decoded):
                                raise ValueError("L0 heartbeat is missing required fields")
                    except Exception:
                        if runtime.message.mode in {
                            PayloadMode.L0_HEARTBEAT, PayloadMode.L1_ALERT,
                            PayloadMode.L2_SEMANTICS, PayloadMode.L3_SEMANTICS_ROI,
                        }:
                            raise
                        # Native L4/L5 receivers are represented by frozen
                        # candidate artifacts; their opaque bytes are not JSON.
                        decoded = {}
                    expected_model_revision = self.message_metadata.get(
                        runtime.message.message_id, {}
                    ).get("expected_model_revision")
                    if expected_model_revision is not None:
                        if runtime.message.mode not in {
                            PayloadMode.L2_SEMANTICS, PayloadMode.L3_SEMANTICS_ROI,
                        } or runtime.message.state_kind is StateKind.DELTA:
                            raise ValueError(
                                "expected_model_revision is valid only for semantic keyframes"
                            )
                        validate_semantic_packet(
                            decoded, require_transmission_metadata=True,
                        )
                        observed_model_revision = decoded["provenance"]["model_revision"]
                        if str(observed_model_revision) != str(expected_model_revision):
                            raise _ModelVersionMismatch(
                                "Semantic packet model revision differs from receiver contract"
                            )
                    received_message = dataclasses.replace(
                        runtime.message, mode=envelope.mode, priority=envelope.priority,
                        capture_us=envelope.capture_us,
                        action_deadline_us=envelope.action_deadline_us,
                        state_expiry_us=envelope.state_expiry_us,
                        payload=envelope.payload, state_kind=envelope.state_kind,
                        base_state_seq=envelope.base_state_seq,
                        resulting_state_seq=envelope.resulting_state_seq,
                    )
                    status = self.receiver_state.apply(received_message, self._now, decoded)
                except _ModelVersionMismatch:
                    status = ApplyStatus.VERSION_MISMATCH
                except Exception:
                    status = ApplyStatus.DECODE_ERROR
                runtime.action_ready_us = self._now
                runtime.apply_status = status.value
                self._send_application_ack(runtime, repeat=False)
                self._log("application_done", message_id=data, status=status.value, timely=self._now <= runtime.message.action_deadline_us)
            elif kind is EventType.APPLICATION_ACK_ARRIVE:
                ack: ApplicationAck = data
                runtime = self._runtimes[ack.message_id]
                # A validated application ACK proves complete reassembly even
                # when one or more fragment ACKs were lost.
                runtime.acked.update(range(len(runtime.fragments)))
                runtime.queued.clear()
                feedback_is_stale = (
                    ack.receiver_state_seq < self.sender_belief.confirmed_seq
                    or (
                        self.sender_belief.latest_feedback_generated_us is not None
                        and ack.generated_us < self.sender_belief.latest_feedback_generated_us
                    )
                )
                if not feedback_is_stale:
                    self.sender_belief.on_application_ack(ack, self._now)
                    self._resolve_cumulative_application_ack(ack)
                    acknowledged_result = runtime.message.resulting_state_seq
                    ack_failed = ack.semantic_state_message and ack.status not in {
                        ApplyStatus.APPLIED, ApplyStatus.DUPLICATE, ApplyStatus.STALE,
                    } and (
                        acknowledged_result is None
                        or ack.receiver_state_seq < acknowledged_result
                    )
                    if ack_failed or self._timed_out_state_message_ids:
                        self.sender_belief.status = "uncertain"
                    elif self._pending_state_message_ids:
                        self.sender_belief.status = "pending"
                    elif ack.semantic_state_message:
                        # A repeated ACK for an older failed message can carry a
                        # newer cumulative state sequence.  That newer state,
                        # not the obsolete per-message failure, determines belief.
                        self.sender_belief.status = "known"
                    if self._feedback_handler is not None:
                        self._feedback_handler(ack, self._now)
                self._log(
                    "application_ack_arrive",
                    message_id=ack.message_id,
                    receiver_state_seq=ack.receiver_state_seq,
                    feedback_ignored=feedback_is_stale,
                )
            elif kind is EventType.APPLICATION_ACK_TIMEOUT:
                message_id = int(data)
                if message_id in self._pending_state_message_ids:
                    self._timed_out_state_message_ids.add(message_id)
                    self.sender_belief.status = "uncertain"
                    self._log(
                        "application_ack_timeout",
                        message_id=message_id,
                        resulting_state_seq=self._runtimes[message_id].message.resulting_state_seq,
                    )
            elif kind is EventType.MESSAGE_EXPIRE:
                runtime = self._runtimes[data]
                if runtime.action_ready_us is None:
                    self._log("message_expire", message_id=data)
            else:
                raise AssertionError(f"Unhandled event {kind}")
        rows = []
        for runtime in self._runtimes.values():
            message = runtime.message
            complete = runtime.action_ready_us is not None
            timely = bool(complete and runtime.action_ready_us <= message.action_deadline_us)
            rows.append(
                {
                    "run_id": message.run_id,
                    "observation_id": message.logical_observation_id,
                    "message_id": message.message_id,
                    "mode": message.mode.name,
                    "capture_us": message.capture_us,
                    "ready_us": message.ready_us,
                    "deadline_us": message.action_deadline_us,
                    "first_tx_us": runtime.first_tx_us,
                    "last_rx_us": runtime.last_rx_us,
                    "action_ready_us": runtime.action_ready_us,
                    "full_message_received": complete,
                    "timely_delivery": timely,
                    "apply_status": runtime.apply_status,
                    "retry_exhausted": runtime.retry_exhausted,
                    "queue_overflow": runtime.queue_overflow,
                    "original_uplink_bytes": runtime.original_wire_bytes,
                    "retry_uplink_bytes": runtime.retry_wire_bytes,
                    "downlink_control_bytes": runtime.downlink_control_bytes,
                    "total_wire_bytes": runtime.original_wire_bytes + runtime.retry_wire_bytes + runtime.downlink_control_bytes,
                    "duplicate_payload_bytes": runtime.duplicate_payload_bytes,
                    "latency_ms": (runtime.action_ready_us - message.capture_us) / 1000 if complete else np.inf,
                    "deadline_slack_ms": (message.action_deadline_us - runtime.action_ready_us) / 1000 if complete else -np.inf,
                    **self.message_metadata.get(message.message_id, {}),
                }
            )
        results = pd.DataFrame(rows).sort_values("message_id").reset_index(drop=True)
        events = pd.DataFrame(self.event_log)
        self._assert_invariants(results)
        return results, events

    def _assert_invariants(self, results: pd.DataFrame) -> None:
        if (results["retry_uplink_bytes"] < 0).any() or (results["total_wire_bytes"] < results["original_uplink_bytes"]).any():
            raise AssertionError("Byte conservation failed")
        if (results["timely_delivery"] & ~results["full_message_received"]).any():
            raise AssertionError("Timely action without complete message")
        if self.receiver_state.applied_seq < self.sender_belief.confirmed_seq:
            raise AssertionError("Sender belief advanced beyond receiver state")


def traces_to_frame(traces: Iterable[LinkTrace]) -> pd.DataFrame:
    rows = []
    for trace in traces:
        for segment in trace.segments:
            rows.append({"trace_id": trace.trace_id, "direction": trace.direction.value, "semantics": trace.semantics.value, **dataclasses.asdict(segment)})
    return pd.DataFrame(rows)


def scientific_trace_design_audit(
    frame: pd.DataFrame,
    required_test_families: Sequence[str],
) -> pd.DataFrame:
    """Audit preregistered development/test trace provenance and pairing.

    A trace ID denotes one bidirectional realization. Segment rows may repeat
    that ID, but its family, split, and seed must be invariant and it must
    resolve to exactly one contiguous raw-service series in each direction.
    """
    rows: list[dict[str, Any]] = []

    def record(gate: str, passed: bool, observed: Any, required: Any) -> None:
        rows.append({
            "gate": gate,
            "passed": bool(passed),
            "observed": json.dumps(observed, sort_keys=True, default=str),
            "required": json.dumps(required, sort_keys=True, default=str),
        })

    required_columns = {
        "trace_id", "family", "split", "seed", "direction", "semantics",
        "start_us", "end_us",
    }
    missing = sorted(required_columns.difference(frame.columns))
    record("trace_manifest_schema", not missing, missing, "no missing required columns")
    if missing:
        return pd.DataFrame(rows)
    if frame.empty:
        record("trace_manifest_nonempty", False, 0, "> 0 rows")
        return pd.DataFrame(rows)
    record("trace_manifest_nonempty", True, int(len(frame)), "> 0 rows")

    tuning_splits = {"train", "calibration", "validation"}
    development_families = set(
        frame.loc[frame["split"].astype(str).isin(tuning_splits), "family"].astype(str)
    )
    test_families = set(
        frame.loc[frame["split"].astype(str).eq("test"), "family"].astype(str)
    )
    overlap = sorted(development_families & test_families)
    record("development_test_family_disjoint", not overlap, overlap, [])
    registered_test_families = {str(family) for family in required_test_families}
    missing_test_families = sorted(registered_test_families - test_families)
    record(
        "registered_test_families_present",
        not missing_test_families,
        sorted(test_families),
        sorted(registered_test_families),
    )

    inconsistent_metadata: list[str] = []
    invalid_direction_or_semantics: list[str] = []
    noncontiguous: list[str] = []
    for trace_id, group in frame.groupby("trace_id", sort=True, dropna=False):
        trace_name = str(trace_id)
        for column in ("family", "split", "seed"):
            if group[column].isna().any() or group[column].nunique(dropna=False) != 1:
                inconsistent_metadata.append(f"{trace_name}:{column}")
        directions = set(group["direction"].astype(str))
        semantics = set(group["semantics"].astype(str))
        directional_series = group[["direction", "semantics"]].drop_duplicates()
        if (
            directions != {Direction.UPLINK.value, Direction.DOWNLINK.value}
            or semantics != {TraceSemantics.RAW_SERVICE.value}
            or len(directional_series) != 2
        ):
            invalid_direction_or_semantics.append(trace_name)
        for direction, direction_rows in group.groupby("direction", sort=True):
            try:
                ordered = direction_rows.sort_values("start_us", kind="mergesort")
                starts = pd.to_numeric(ordered["start_us"], errors="raise").to_numpy(dtype=np.int64)
                ends = pd.to_numeric(ordered["end_us"], errors="raise").to_numpy(dtype=np.int64)
                valid = bool(
                    len(starts)
                    and starts[0] >= 0
                    and np.all(ends > starts)
                    and np.all(starts[1:] == ends[:-1])
                )
            except (TypeError, ValueError):
                valid = False
            if not valid:
                noncontiguous.append(f"{trace_name}:{direction}")
    record("trace_metadata_consistent", not inconsistent_metadata, inconsistent_metadata, [])
    record(
        "one_raw_service_uplink_downlink_per_trace",
        not invalid_direction_or_semantics,
        invalid_direction_or_semantics,
        [],
    )
    record("trace_segments_contiguous", not noncontiguous, noncontiguous, [])
    return pd.DataFrame(rows)


def scientific_episode_coverage_audit(
    traces: Sequence[LinkTrace],
    manifest: pd.DataFrame,
    *,
    state_expiry_grace_us: int = 2_000_000,
) -> pd.DataFrame:
    """Prove selected capture, deadline, and worst-case state expiry are measured."""
    rows: list[dict[str, Any]] = []

    def record(gate: str, passed: bool, observed: Any, required: Any) -> None:
        rows.append({
            "gate": gate,
            "passed": bool(passed),
            "observed": json.dumps(observed, sort_keys=True, default=str),
            "required": json.dumps(required, sort_keys=True, default=str),
        })

    if state_expiry_grace_us < 0:
        raise ValueError("state_expiry_grace_us cannot be negative")
    required_columns = {"capture_time_us", "trace_time_us", "deadline_us"}
    missing = sorted(required_columns.difference(manifest.columns))
    record("episode_timing_schema", not missing, missing, "no missing required columns")
    if missing or manifest.empty or not traces:
        record(
            "episode_and_trace_nonempty",
            False,
            {"episode_rows": int(len(manifest)), "traces": int(len(traces))},
            {"episode_rows": "> 0", "traces": "> 0"},
        )
        return pd.DataFrame(rows)
    record(
        "episode_and_trace_nonempty", True,
        {"episode_rows": int(len(manifest)), "traces": int(len(traces))},
        {"episode_rows": "> 0", "traces": "> 0"},
    )
    capture = pd.to_numeric(manifest["capture_time_us"], errors="raise").to_numpy(dtype=np.int64)
    trace_capture = pd.to_numeric(manifest["trace_time_us"], errors="raise").to_numpy(dtype=np.int64)
    dataset_deadline = pd.to_numeric(manifest["deadline_us"], errors="raise").to_numpy(dtype=np.int64)
    budgets = dataset_deadline - capture
    positive_budgets = bool(np.all(budgets > 0))
    record("positive_deadline_budgets", positive_budgets, budgets.tolist(), "> 0")
    if not positive_budgets:
        return pd.DataFrame(rows)
    physical_deadlines = trace_capture + budgets
    physical_expiries = physical_deadlines + int(state_expiry_grace_us)
    required_start_us = int(trace_capture.min())
    required_end_exclusive_us = int(physical_expiries.max())

    held_last = sorted({
        f"{trace.trace_id}:{trace.direction.value}"
        for trace in traces if trace.end_behavior is EndBehavior.HOLD_LAST
    })
    record("scientific_traces_terminate", not held_last, held_last, [])
    uncovered: list[dict[str, Any]] = []
    for trace in traces:
        measured_start = int(trace.segments[0].start_us)
        measured_end = int(trace.segments[-1].end_us)
        if measured_start > required_start_us or measured_end <= required_end_exclusive_us:
            uncovered.append({
                "trace_id": trace.trace_id,
                "direction": trace.direction.value,
                "measured_start_us": measured_start,
                "measured_end_us": measured_end,
                "required_capture_start_us": required_start_us,
                "required_expiry_us": required_end_exclusive_us,
            })
    record("episode_within_measured_trace_coverage", not uncovered, uncovered, [])
    return pd.DataFrame(rows)


def load_trace_design_registry(path: str) -> pd.DataFrame:
    """Load only preregistered trace provenance, never held-out link values."""
    design_columns = {
        "trace_id", "family", "split", "seed", "direction", "semantics",
        "start_us", "end_us",
    }
    return pd.read_csv(path, usecols=lambda column: column in design_columns)


def load_trace_manifest(
    path: str,
    *,
    required_split: str | None = None,
    end_behavior: EndBehavior = EndBehavior.HOLD_LAST,
) -> tuple[list[LinkTrace], pd.DataFrame]:
    """Load measured/generated trace segments with explicit raw-service semantics."""
    frame = pd.read_csv(path)
    required = {
        "trace_id", "family", "split", "direction", "semantics", "start_us", "end_us",
        "service_rate_bps", "base_one_way_delay_us", "jitter_parameter_us", "packet_loss_parameter",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"Trace manifest missing columns: {sorted(missing)}")
    if required_split is not None:
        frame = frame[frame["split"].eq(required_split)].copy()
    traces: list[LinkTrace] = []
    for (trace_id, direction, semantics), group in frame.groupby(["trace_id", "direction", "semantics"], sort=True):
        ordered = group.sort_values("start_us")
        segments = tuple(
            LinkSegment(
                int(row.start_us), int(row.end_us), int(row.service_rate_bps),
                int(row.base_one_way_delay_us), int(row.jitter_parameter_us),
                float(row.packet_loss_parameter),
                float(row.snr_db) if hasattr(row, "snr_db") and pd.notna(row.snr_db) else None,
                float(row.distance_m) if hasattr(row, "distance_m") and pd.notna(row.distance_m) else None,
                int(row.queue_capacity_bytes)
                if hasattr(row, "queue_capacity_bytes") and pd.notna(row.queue_capacity_bytes)
                else None,
            )
            for row in ordered.itertuples(index=False)
        )
        traces.append(LinkTrace(
            str(trace_id), Direction(str(direction)), TraceSemantics(str(semantics)),
            segments, EndBehavior(end_behavior),
        ))
    if not traces:
        raise ValueError("Trace manifest selection is empty")
    return traces, frame
