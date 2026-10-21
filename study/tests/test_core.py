from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from semuav.config import Profile, load_study_config
from semuav.control import causal_schema_audit
from semuav.data import (
    assign_group_safe_splits,
    audit_split_leakage,
    build_canonical_manifest,
)
from semuav.experiments import (
    add_experiment2_total_costs,
    assert_experiment3_exact_replicate_coverage,
    build_causal_candidate_profiles,
    experiment2_representation_frontier,
    experiment3_static_transport,
    experiment6_receiver_robustness,
    make_dry_run_fixtures,
    run_paired_policy_trials,
    summarize_experiment3_static_transport,
)
from semuav.control import FixedModePolicy
from semuav.network import Direction, EndBehavior, EventType, LinkSegment, LinkTrace, PHASE, TraceSemantics, SelectiveRepeatSimulator, constant_trace, load_trace_design_registry, load_trace_manifest, scientific_episode_coverage_audit, scientific_trace_design_audit
from semuav.protocol import (
    AppMessage,
    ApplicationAck,
    ApplyStatus,
    PayloadMode,
    ReassemblyBuffer,
    ReceiverSemanticState,
    ReliabilityProfile,
    StateKind,
    SenderReceiverBelief,
    dequantize_tensor,
    decode_app_envelope,
    encode_mapping,
    encode_image,
    fragment_message,
    make_state_delta,
    quantize_tensor,
    semantic_level_view,
    validate_semantic_packet,
)
from semuav.registry import StudyGateError


ROOT = Path(__file__).resolve().parents[1]


class ConfigDataTests(unittest.TestCase):
    def setUp(self) -> None:
        # These assertions describe checked-in configuration templates, not
        # environment-resolved live-run configuration.
        clean_environment = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith("SEMUAV_")
        }
        environment_patch = patch.dict(
            os.environ, clean_environment, clear=True,
        )
        environment_patch.start()
        self.addCleanup(environment_patch.stop)

    def test_dry_config_and_group_safe_split(self) -> None:
        config = load_study_config(ROOT / "configs" / "dry_run.yaml")
        self.assertEqual(config.profile, Profile.DRY_RUN)
        manifest = build_canonical_manifest(config)
        split = assign_group_safe_splits(manifest, config.split)
        report = audit_split_leakage(split, config.split)
        self.assertTrue(report.passed, report.errors)
        self.assertEqual(len(split), 120)
        self.assertFalse(split.groupby("split_group")["split"].nunique().gt(1).any())

    def test_full_config_is_parseable_but_blocked(self) -> None:
        config = load_study_config(ROOT / "configs" / "full.yaml")
        failures = config.prerequisite_errors("test")
        self.assertEqual(config.profile, Profile.FULL)
        self.assertGreaterEqual(len(failures), 8)
        self.assertTrue(any("40-character commit" in failure for failure in failures))


class ProtocolTests(unittest.TestCase):
    def test_explicit_json_mapping_format_is_preserved(self) -> None:
        payload, wire_format = encode_mapping(
            {"value": 7}, scientific=False, wire_format="json-debug",
        )
        self.assertEqual(wire_format, "json-debug")
        self.assertEqual(json.loads(payload), {"value": 7})

        with self.assertRaises(StudyGateError):
            encode_mapping(
                {"value": 7}, scientific=True, wire_format="json-debug",
            )

    def test_scientific_semantic_packet_requires_wire_provenance(self) -> None:
        packet = {
            "schema_version": "semuav.v1", "observation_id": "p1",
            "mission": "urban", "capture_time_us": 0,
            "telemetry": {"gps_lat": 1.0, "gps_lon": 2.0, "altitude_m": 3.0, "yaw_deg": 4.0},
            "sufficient_evidence": False, "event": "insufficient_evidence",
            "objects": [], "relations": [], "risk": 0.5,
            "uncertainty": 1.0, "recommended_action": "REVIEW",
        }
        with self.assertRaises(ValueError):
            validate_semantic_packet(packet, require_transmission_metadata=True)
        packet["evidence"] = []
        packet["calibrated_correctness_probability"] = 0.25
        packet["provenance"] = {
            "source_model": "model", "model_revision": "a" * 40,
            "prompt_hash": "b" * 64, "ontology_version": "semuav.ontology.v1",
            "calibration": {"method": "temperature_scaling", "temperature": 1.0},
            "schema_version": "semuav.v1",
        }
        self.assertEqual(
            validate_semantic_packet(packet, require_transmission_metadata=True)["observation_id"],
            "p1",
        )

    def test_l1_is_minimum_calibrated_core_alert_without_evidence_records(self) -> None:
        packet = {
            "schema_version": "semuav.v1", "observation_id": "alert-1",
            "mission": "urban", "capture_time_us": 10,
            "telemetry": {"gps_lat": 24.0, "gps_lon": 54.0, "altitude_m": 80.0, "yaw_deg": 1.0},
            "sufficient_evidence": True, "event": "visible_road_obstruction",
            "objects": [{"label": "car"}], "relations": [],
            "risk": 0.8, "uncertainty": 0.2,
            "calibrated_correctness_probability": 0.85,
            "recommended_action": "RESPOND",
            "evidence": [{"linked_claims": ["event"], "bbox_normalized_xyxy": [0.1, 0.1, 0.5, 0.5]}],
            "confidence_provenance": {"source": "temperature_scaled", "source_model": "model"},
            "provenance": {
                "source_model": "model", "model_revision": "a" * 40,
                "prompt_hash": "b" * 64, "ontology_version": "semuav.ontology.v1",
                "calibration": {"method": "temperature_scaling", "temperature": 1.0},
                "schema_version": "semuav.v1",
            },
        }
        validate_semantic_packet(packet, require_transmission_metadata=True)
        alert = semantic_level_view(packet, PayloadMode.L1_ALERT)
        self.assertEqual(float(alert["calibrated_correctness_probability"]), 0.85)
        self.assertIn("sufficient_evidence", alert)
        self.assertIn("provenance", alert)
        self.assertIn("confidence_provenance", alert)
        self.assertNotIn("evidence", alert)
        self.assertNotIn("objects", alert)
        self.assertNotIn("relations", alert)
        self.assertLess(len(json.dumps(alert)), len(json.dumps(packet)))

    def test_fragment_boundaries_and_reassembly(self) -> None:
        for size in (1, 1135, 1136, 1137, 5000):
            message = AppMessage("t", "o", size, PayloadMode.L5_IMAGE_OR_CLIP, 0, 0, 10, 20, 1, bytes(range(251)) * (size // 251) + bytes(range(size % 251)))
            fragments = fragment_message(message, 1200)
            self.assertTrue(all(fragment.ip_packet_bytes <= 1200 for fragment in fragments))
            buffer = ReassemblyBuffer(len(fragments), message.wire_payload_hash[:8])
            for fragment in reversed(fragments):
                buffer.add(fragment)
            envelope = decode_app_envelope(buffer.assemble())
            self.assertEqual(envelope.payload, message.payload)

    def test_serialized_fragment_crc_and_version_are_enforced(self) -> None:
        message = AppMessage("t", "o", 55, PayloadMode.L5_IMAGE_OR_CLIP, 0, 0, 10, 20, 1, b"payload")
        fragment = fragment_message(message)[0]
        encoded = bytearray(fragment.to_bytes())
        encoded[-1] ^= 0x01
        with self.assertRaises(ValueError):
            type(fragment).from_bytes(bytes(encoded))
        wrong_version = bytearray(fragment.to_bytes())
        wrong_version[0] = 99
        decoded = type(fragment).from_bytes(bytes(wrong_version))
        buffer = ReassemblyBuffer(1, message.wire_payload_hash[:8])
        with self.assertRaises(ValueError):
            buffer.add(decoded)

    def test_reassembly_rejects_noncontiguous_offsets(self) -> None:
        message = AppMessage("t", "o", 56, PayloadMode.L5_IMAGE_OR_CLIP, 0, 0, 10, 20, 1, b"x" * 3000)
        fragments = list(fragment_message(message))
        fragments[-1] = dataclasses.replace(fragments[-1], payload_offset=4000)
        buffer = ReassemblyBuffer(len(fragments), message.wire_payload_hash[:8])
        for fragment in fragments:
            buffer.add(fragment)
        with self.assertRaises(ValueError):
            buffer.assemble()

    def test_quantization_contract(self) -> None:
        values = np.linspace(-3, 7, 1001, dtype=np.float32).reshape(7, 11, 13)
        for bits, tolerance in ((8, 0.05), (4, 0.7)):
            encoded = quantize_tensor(values, bits)
            decoded = dequantize_tensor(encoded)
            self.assertEqual(decoded.shape, values.shape)
            self.assertLess(float(np.max(np.abs(decoded - values))), tolerance)

    def test_receiver_delta_and_repair(self) -> None:
        report = experiment6_receiver_robustness()
        self.assertTrue(report["passed"].all(), report.to_dict("records"))

    def test_sender_belief_ignores_reordered_and_stateless_acks(self) -> None:
        belief = SenderReceiverBelief()
        belief.on_application_ack(
            ApplicationAck(2, ApplyStatus.APPLIED, 2, 100, 110, True, 100), 120
        )
        belief.on_application_ack(
            ApplicationAck(1, ApplyStatus.APPLIED, 1, 50, 60, True, 50), 130
        )
        belief.on_application_ack(
            ApplicationAck(3, ApplyStatus.APPLIED, 2, 200, 210, False, 200), 220
        )
        self.assertEqual(belief.confirmed_seq, 2)
        self.assertEqual(belief.receiver_update_us, 100)


class NetworkTests(unittest.TestCase):
    def test_scientific_trace_design_requires_unseen_paired_raw_families(self) -> None:
        rows = []
        for trace_id, family, split, seed in (
            ("validation-1", "validation_only", "validation", 11),
            ("calibration-1", "calibration_only", "calibration", 17),
            ("test-1", "held_out", "test", 29),
        ):
            for direction in ("uplink", "downlink"):
                rows.append({
                    "trace_id": trace_id, "family": family, "split": split,
                    "seed": seed, "direction": direction,
                    "semantics": "raw_service", "start_us": 0,
                    "end_us": 5_000_000,
                })
        valid = pd.DataFrame(rows)
        audit = scientific_trace_design_audit(valid, ("held_out",))
        self.assertTrue(audit["passed"].all(), audit.to_dict("records"))

        overlapping = valid.copy()
        overlapping.loc[overlapping["split"].eq("calibration"), "family"] = "held_out"
        overlap_audit = scientific_trace_design_audit(overlapping, ("held_out",))
        self.assertFalse(bool(overlap_audit.set_index("gate").loc[
            "development_test_family_disjoint", "passed",
        ]))

        malformed = valid.copy()
        malformed = malformed[~(
            malformed["trace_id"].eq("test-1")
            & malformed["direction"].eq("downlink")
        )]
        malformed.loc[malformed["trace_id"].eq("test-1"), "seed"] = np.nan
        malformed_audit = scientific_trace_design_audit(malformed, ("held_out",))
        indexed = malformed_audit.set_index("gate")
        self.assertFalse(bool(indexed.loc["trace_metadata_consistent", "passed"]))
        self.assertFalse(bool(indexed.loc[
            "one_raw_service_uplink_downlink_per_trace", "passed",
        ]))

    def test_scientific_episode_coverage_is_strict_and_disables_hold_last(self) -> None:
        segment = LinkSegment(0, 5_000_000, 1_000_000, 1_000)
        traces = [
            LinkTrace(
                "coverage", direction, TraceSemantics.RAW_SERVICE, (segment,),
                EndBehavior.TERMINATE,
            )
            for direction in (Direction.UPLINK, Direction.DOWNLINK)
        ]
        manifest = pd.DataFrame([{
            "capture_time_us": 10_000_000, "trace_time_us": 0,
            "deadline_us": 11_000_000,
        }])
        audit = scientific_episode_coverage_audit(traces, manifest)
        self.assertTrue(audit["passed"].all(), audit.to_dict("records"))

        hold_last = [dataclasses.replace(trace, end_behavior=EndBehavior.HOLD_LAST) for trace in traces]
        hold_audit = scientific_episode_coverage_audit(hold_last, manifest).set_index("gate")
        self.assertFalse(bool(hold_audit.loc["scientific_traces_terminate", "passed"]))

        exact_boundary = [
            dataclasses.replace(
                trace,
                segments=(dataclasses.replace(segment, end_us=3_000_000),),
            )
            for trace in traces
        ]
        boundary_audit = scientific_episode_coverage_audit(
            exact_boundary, manifest,
        ).set_index("gate")
        self.assertFalse(bool(boundary_audit.loc[
            "episode_within_measured_trace_coverage", "passed",
        ]))

        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "traces.csv"
            pd.DataFrame([
                {
                    "trace_id": "loaded", "family": "held_out", "split": "test",
                    "seed": 41, "direction": direction.value,
                    "semantics": "raw_service", "start_us": 0,
                    "end_us": 5_000_000, "service_rate_bps": 1_000_000,
                    "base_one_way_delay_us": 1_000,
                    "jitter_parameter_us": 0, "packet_loss_parameter": 0.0,
                    "queue_capacity_bytes": 16_384,
                }
                for direction in (Direction.UPLINK, Direction.DOWNLINK)
            ]).to_csv(trace_path, index=False)
            registry = load_trace_design_registry(str(trace_path))
            self.assertNotIn("service_rate_bps", registry.columns)
            self.assertNotIn("packet_loss_parameter", registry.columns)
            loaded, _ = load_trace_manifest(
                str(trace_path), required_split="test",
                end_behavior=EndBehavior.TERMINATE,
            )
        self.assertTrue(all(trace.end_behavior is EndBehavior.TERMINATE for trace in loaded))
        self.assertTrue(
            scientific_episode_coverage_audit(loaded, manifest)["passed"].all()
        )

    def test_gilbert_elliott_loss_is_deterministic_exogenous_and_bursty(self) -> None:
        uplink = constant_trace("registered-burst", Direction.UPLINK, 1.0, 1, 0.20)
        downlink = constant_trace("registered-burst-down", Direction.DOWNLINK, 1.0, 1, 0.0)

        def pattern(packet_key: str) -> list[bool]:
            simulator = SelectiveRepeatSimulator(
                uplink, downlink, seed=20260823,
                loss_model="gilbert_elliott",
                burst_recovery_probability=0.10,
                burst_slot_us=100_000,
            )
            return [
                simulator._loss_and_delay(
                    uplink,
                    uplink.serialize(slot * 100_000, 1),
                    (packet_key, slot),
                )[0]
                for slot in range(200)
            ]

        first = pattern("representation-a")
        self.assertEqual(first, pattern("representation-a"))
        # Burst loss is a trace-time process: representation-specific packet
        # keys cannot change the state observed in the same exogenous slot.
        self.assertEqual(first, pattern("representation-b"))
        longest_run = max(
            (len(run) for run in "".join("1" if item else "0" for item in first).split("0")),
            default=0,
        )
        self.assertGreaterEqual(longest_run, 2)

    def test_future_ready_message_is_not_visible_in_current_queue(self) -> None:
        uplink = constant_trace("future-q-u", Direction.UPLINK, 1.0, 10, 0.0)
        downlink = constant_trace("future-q-d", Direction.DOWNLINK, 1.0, 10, 0.0)
        simulator = SelectiveRepeatSimulator(uplink, downlink, seed=1)
        future = AppMessage(
            "r", "future", 999, PayloadMode.L5_IMAGE_OR_CLIP,
            0, 1_000_000, 2_000_000, 3_000_000, 1, b"x" * 2000,
        )
        simulator._register_message(future)
        self.assertEqual(simulator.queue_snapshot(0).pending_bytes, 0)
        self.assertGreater(simulator.queue_snapshot(1_000_000).pending_bytes, 0)

    def test_cross_segment_serialization_and_outage(self) -> None:
        trace = LinkTrace(
            "piecewise", Direction.UPLINK, TraceSemantics.RAW_SERVICE,
            (
                LinkSegment(0, 1_000_000, 1_000_000, 10_000),
                LinkSegment(1_000_000, 2_000_000, 0, 10_000, packet_loss_parameter=1.0),
                LinkSegment(2_000_000, 4_000_000, 2_000_000, 10_000),
            ),
            EndBehavior.HOLD_LAST,
        )
        span = trace.serialize(900_000, 400_000)
        self.assertEqual(span.end_us, 2_150_000)
        self.assertEqual(sum(piece.bits for piece in span.pieces), 400_000)

    def test_no_loss_closed_form_and_bytes(self) -> None:
        uplink = constant_trace("u", Direction.UPLINK, 1.0, 10, 0.0)
        downlink = constant_trace("d", Direction.DOWNLINK, 1.0, 10, 0.0)
        payload = b"x" * 2000
        message = AppMessage("r", "o", 1, PayloadMode.L5_IMAGE_OR_CLIP, 0, 0, 2_000_000, 3_000_000, 1, payload)
        results, events = SelectiveRepeatSimulator(uplink, downlink, seed=1).run([message])
        row = results.iloc[0]
        self.assertTrue(bool(row.full_message_received))
        self.assertTrue(bool(row.timely_delivery))
        self.assertEqual(row.apply_status, "applied")
        self.assertEqual(int(row.retry_uplink_bytes), 0)
        expected = sum(fragment.ip_packet_bytes for fragment in fragment_message(message))
        self.assertEqual(int(row.original_uplink_bytes), expected)
        self.assertTrue((events["time_us"].diff().fillna(0) >= 0).all())

    def test_no_spurious_retry_when_ack_precedes_adaptive_rto(self) -> None:
        uplink = constant_trace("slow-rtt-u", Direction.UPLINK, 1.0, 100, 0.0)
        downlink = constant_trace("slow-rtt-d", Direction.DOWNLINK, 1.0, 100, 0.0)
        message = AppMessage("r", "o", 91, PayloadMode.L1_ALERT, 0, 0, 1_000_000, 2_000_000, 5, b"{}")
        results, _ = SelectiveRepeatSimulator(
            uplink, downlink, seed=2, initial_rtt_us=200_000
        ).run([message])
        self.assertEqual(int(results.iloc[0].retry_uplink_bytes), 0)

    def test_l3_semantic_prefix_is_decoded_before_roi(self) -> None:
        uplink = constant_trace("l3-u", Direction.UPLINK, 5.0, 1, 0.0)
        downlink = constant_trace("l3-d", Direction.DOWNLINK, 5.0, 1, 0.0)
        packet = {
            "schema_version": "semuav.v1", "observation_id": "l3",
            "mission": "urban", "capture_time_us": 0,
            "telemetry": {"gps_lat": 1.0, "gps_lon": 2.0, "altitude_m": 3.0, "yaw_deg": 4.0},
            "sufficient_evidence": True, "event": "visible_road_obstruction",
            "objects": [], "relations": [], "risk": 0.9, "uncertainty": 0.1,
            "recommended_action": "RESPOND",
        }
        semantic, wire_format = encode_mapping(packet, scientific=False)
        roi = encode_image(Image.new("RGB", (8, 8), "red"), "JPEG", 80)
        payload = len(semantic).to_bytes(4, "big") + semantic + roi
        message = AppMessage(
            "r", "l3", 92, PayloadMode.L3_SEMANTICS_ROI,
            0, 0, 1_000_000, 2_000_000, 5, payload,
            StateKind.KEYFRAME, None, 1,
        )
        simulator = SelectiveRepeatSimulator(uplink, downlink, seed=4)
        results, _ = simulator.run(
            [], captures=[(0, {})],
            capture_handler=lambda *_: (message, {"wire_format": wire_format}),
        )
        self.assertEqual(results.iloc[0].apply_status, "applied")
        self.assertEqual(simulator.receiver_state.state["event"], "visible_road_obstruction")

    def test_l3_declared_evidence_hash_is_enforced(self) -> None:
        uplink = constant_trace("l3-hash-u", Direction.UPLINK, 5.0, 1, 0.0)
        downlink = constant_trace("l3-hash-d", Direction.DOWNLINK, 5.0, 1, 0.0)
        roi = encode_image(Image.new("RGB", (8, 8), "red"), "JPEG", 80)
        packet = {
            "schema_version": "semuav.v1", "observation_id": "l3-hash",
            "mission": "urban", "capture_time_us": 0,
            "telemetry": {"gps_lat": 1.0, "gps_lon": 2.0, "altitude_m": 3.0, "yaw_deg": 4.0},
            "sufficient_evidence": True, "event": "visible_road_obstruction",
            "objects": [], "relations": [], "risk": 0.9, "uncertainty": 0.1,
            "recommended_action": "RESPOND",
            "evidence": [{"linked_claims": ["event"], "integrity_sha256": hashlib.sha256(b"wrong").hexdigest()}],
        }
        semantic, wire_format = encode_mapping(packet, scientific=False)
        message = AppMessage(
            "r", "l3-hash", 93, PayloadMode.L3_SEMANTICS_ROI,
            0, 0, 1_000_000, 2_000_000, 5,
            len(semantic).to_bytes(4, "big") + semantic + roi,
            StateKind.KEYFRAME, None, 1,
        )
        results, _ = SelectiveRepeatSimulator(uplink, downlink, seed=4).run(
            [], captures=[(0, {})],
            capture_handler=lambda *_: (message, {"wire_format": wire_format}),
        )
        self.assertEqual(results.iloc[0].apply_status, "decode_error")

    def test_malformed_stateful_payload_cannot_advance_state(self) -> None:
        uplink = constant_trace("bad-u", Direction.UPLINK, 5.0, 1, 0.0)
        downlink = constant_trace("bad-d", Direction.DOWNLINK, 5.0, 1, 0.0)
        message = AppMessage(
            "r", "bad", 94, PayloadMode.L3_SEMANTICS_ROI,
            0, 0, 1_000_000, 2_000_000, 5,
            (999).to_bytes(4, "big") + b"truncated",
            StateKind.KEYFRAME, None, 1,
        )
        simulator = SelectiveRepeatSimulator(uplink, downlink, seed=4)
        results, _ = simulator.run(
            [], captures=[(0, {})],
            capture_handler=lambda *_: (message, {"wire_format": "json-debug"}),
        )
        self.assertEqual(results.iloc[0].apply_status, "decode_error")
        self.assertEqual(simulator.receiver_state.applied_seq, 0)

    def test_queue_snapshot_includes_unscheduled_window_tail(self) -> None:
        uplink = constant_trace("queue-u", Direction.UPLINK, 0.1, 1, 0.0)
        downlink = constant_trace("queue-d", Direction.DOWNLINK, 1.0, 1, 0.0)
        message = AppMessage(
            "r", "large", 93, PayloadMode.L5_IMAGE_OR_CLIP,
            0, 0, 20_000_000, 30_000_000, 1, b"x" * 50_000,
        )
        snapshots = []
        def capture_handler(simulator, now_us, data):
            if data == "message":
                return message, {}
            snapshots.append(simulator.queue_snapshot(now_us))
            return None
        SelectiveRepeatSimulator(uplink, downlink, seed=5).run(
            [], captures=[(0, "message"), (1, "inspect")],
            capture_handler=capture_handler,
        )
        expected = sum(fragment.ip_packet_bytes for fragment in fragment_message(message))
        snapshot = snapshots[0]
        self.assertEqual(snapshot.pending_bytes + snapshot.outstanding_unacked_bytes, expected)

    def test_queue_capacity_drops_oversized_message_without_wire_bytes(self) -> None:
        uplink = constant_trace("capacity-u", Direction.UPLINK, 1.0, 1, 0.0)
        downlink = constant_trace("capacity-d", Direction.DOWNLINK, 1.0, 1, 0.0)
        message = AppMessage(
            "r", "oversized", 95, PayloadMode.L5_IMAGE_OR_CLIP,
            0, 0, 1_000_000, 2_000_000, 1, b"x" * 5_000,
        )
        result, events = SelectiveRepeatSimulator(
            uplink, downlink, seed=8, queue_capacity_bytes=1_024,
        ).run([message])
        self.assertTrue(bool(result.iloc[0].queue_overflow))
        self.assertEqual(int(result.iloc[0].total_wire_bytes), 0)
        self.assertIn("queue_overflow_drop", set(events["kind"]))

    def test_trace_segment_queue_capacity_controls_admission(self) -> None:
        uplink = LinkTrace(
            "segment-cap", Direction.UPLINK, TraceSemantics.RAW_SERVICE,
            (LinkSegment(
                0, 3_000_000, 1_000_000, 1_000,
                queue_capacity_bytes=1_024,
            ),),
        )
        downlink = constant_trace("segment-cap-down", Direction.DOWNLINK, 1.0, 1, 0.0)
        message = AppMessage(
            "r", "segment-cap", 96, PayloadMode.L5_IMAGE_OR_CLIP,
            0, 0, 1_000_000, 2_000_000, 1, b"x" * 2_000,
        )
        result, events = SelectiveRepeatSimulator(
            uplink, downlink, seed=8,
            # The active trace-segment cap must override this looser fallback.
            queue_capacity_bytes=16_384,
        ).run([message])
        self.assertTrue(bool(result.iloc[0].queue_overflow))
        drop = events[events["kind"].eq("queue_overflow_drop")].iloc[0]
        self.assertEqual(int(drop.capacity_bytes), 1_024)

    def test_ack_is_visible_to_same_time_capture(self) -> None:
        self.assertLess(PHASE[EventType.APPLICATION_ACK_ARRIVE], PHASE[EventType.CAPTURE])
        self.assertLess(PHASE[EventType.FRAGMENT_ACK_ARRIVE], PHASE[EventType.RTO_EXPIRE])

    def test_future_trace_tail_cannot_change_early_result(self) -> None:
        common = LinkSegment(0, 1_000_000, 5_000_000, 5_000)
        fast_tail = LinkSegment(1_000_000, 3_000_000, 20_000_000, 5_000)
        outage_tail = LinkSegment(1_000_000, 3_000_000, 0, 5_000, packet_loss_parameter=1.0)
        message = AppMessage("r", "o", 9, PayloadMode.L1_ALERT, 0, 0, 500_000, 800_000, 5, b"{}")
        results = []
        for tail in (fast_tail, outage_tail):
            uplink = LinkTrace("same-prefix", Direction.UPLINK, TraceSemantics.RAW_SERVICE, (common, tail))
            downlink = constant_trace("reverse", Direction.DOWNLINK, 5.0, 5, 0.0)
            results.append(SelectiveRepeatSimulator(uplink, downlink, seed=3).run([message])[0].iloc[0])
        self.assertEqual(results[0].action_ready_us, results[1].action_ready_us)
        self.assertEqual(results[0].total_wire_bytes, results[1].total_wire_bytes)


class ControllerTests(unittest.TestCase):
    def test_causal_schema_has_no_hidden_fields(self) -> None:
        self.assertTrue(causal_schema_audit()["passed"])


class FixtureTests(unittest.TestCase):
    def test_experiment2_total_cost_axes_charge_only_required_perception(self) -> None:
        common = {
            "sample_id": "s1", "generation_ms": 3.0, "energy_j": 0.2,
            "receiver_latency_ms": 4.0, "receiver_energy_j": 0.1,
            "variant": "candidate", "payload_blob": b"payload".hex(),
            "wire_bytes": 7,
            "native_receiver_correct": True,
        }
        candidates = pd.DataFrame([
            {**common, "mode": "L2_SEMANTICS", "source_model": "compact_vlm"},
            {**common, "mode": "L2_SEMANTICS", "source_model": "detector_only"},
            {**common, "mode": "L5_IMAGE_OR_CLIP", "source_model": "ground_visual_receiver"},
        ])
        measured = pd.DataFrame([{
            "sample_id": "s1", "detector_latency_ms": 5.0,
            "detector_energy_j": 0.4, "context_model_latency_ms": 7.0,
            "context_model_energy_j": 0.6,
            "capture_time_us": 0, "trace_time_us": 0, "deadline_us": 2_000_000,
        }])
        result = add_experiment2_total_costs(
            candidates, measured, require_measured_costs=True,
        ).set_index("source_model")
        self.assertAlmostEqual(result.loc["compact_vlm", "total_processing_latency_ms"], 19.0)
        self.assertAlmostEqual(result.loc["compact_vlm", "total_compute_energy_j"], 1.3)
        self.assertAlmostEqual(result.loc["detector_only", "total_processing_latency_ms"], 12.0)
        self.assertAlmostEqual(result.loc["detector_only", "total_compute_energy_j"], 0.7)
        self.assertAlmostEqual(result.loc["ground_visual_receiver", "total_processing_latency_ms"], 7.0)
        self.assertAlmostEqual(result.loc["ground_visual_receiver", "total_compute_energy_j"], 0.3)
        transport = experiment3_static_transport(
            result.reset_index(), measured, seed=13,
            rates_mbps=(20.0,), rtts_ms=(20.0,), losses=(0.0,),
            loss_models=("iid",), jitters_ms=(0.0,), queue_kb=(512,),
            require_perception_costs=True,
        )
        self.assertEqual(len(transport), 3)
        self.assertIn("detector_latency_ms", transport.columns)
        self.assertNotIn("detector_latency_ms_x", transport.columns)
    def test_causal_profiles_preserve_total_cost_when_sender_ranking_reverses(self) -> None:
        candidates = pd.DataFrame([
            {
                "sample_id": "s1", "mode": "L2_SEMANTICS", "variant": "sender-cheap",
                "source_model": "model-a",
                "native_receiver_correct": True, "wire_bytes": 10,
                "generation_ms": 1.0, "energy_j": 0.1,
                "receiver_latency_ms": 1.0, "receiver_energy_j": 5.0,
                "total_processing_latency_ms": 2.0, "total_compute_energy_j": 5.1,
            },
            {
                "sample_id": "s1", "mode": "L2_SEMANTICS", "variant": "total-cheap",
                "source_model": "model-b",
                "native_receiver_correct": True, "wire_bytes": 10,
                "generation_ms": 1.0, "energy_j": 0.2,
                "receiver_latency_ms": 1.0, "receiver_energy_j": 0.1,
                "total_processing_latency_ms": 2.0, "total_compute_energy_j": 0.3,
            },
        ])
        profile = build_causal_candidate_profiles(candidates, ["s1"]).set_index("variant")
        self.assertLess(
            float(profile.loc["sender-cheap", "energy_j"]),
            float(profile.loc["total-cheap", "energy_j"]),
        )
        self.assertGreater(
            float(profile.loc["sender-cheap", "total_compute_energy_j"]),
            float(profile.loc["total-cheap", "total_compute_energy_j"]),
        )
        collision = candidates.copy()
        collision.loc[collision["source_model"].eq("model-b"), "variant"] = "sender-cheap"
        with self.assertRaises(StudyGateError):
            build_causal_candidate_profiles(collision, ["s1"])

    def test_fixture_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = make_dry_run_fixtures(directory, 4)
            self.assertTrue(frame["dataset"].eq("GENERATED_FIXTURE").all())
            self.assertTrue(all(Path(path).is_file() for path in frame["image_path"]))

    def test_representation_frontier_reports_lossless_wire_reduction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = make_dry_run_fixtures(directory, 2)
            frontier = experiment2_representation_frontier(manifest, scientific=False)
            required = {
                "lossless_reference_payload_bytes", "lossless_reference_wire_bytes",
                "lossless_reference_basis", "lossless_wire_compression_ratio",
                "lossless_wire_reduction_fraction",
            }
            self.assertTrue(required.issubset(frontier.columns))
            self.assertTrue(np.isfinite(frontier["lossless_wire_compression_ratio"]).all())
            reference = frontier[frontier["variant"].eq("lossless_png_reference")]
            self.assertTrue(np.allclose(reference["lossless_wire_compression_ratio"], 1.0))
            self.assertTrue(np.allclose(reference["lossless_wire_reduction_fraction"], 0.0))

    def test_policy_execution_order_does_not_change_paired_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = make_dry_run_fixtures(directory, 4)
            candidates = experiment2_representation_frontier(manifest, scientific=False)
            trace = LinkTrace(
                "order-0", Direction.UPLINK, TraceSemantics.RAW_SERVICE,
                (LinkSegment(0, 30_000_000, 2_000_000, 20_000, 0, 0.02),),
            )
            policies = [FixedModePolicy(PayloadMode.L1_ALERT), FixedModePolicy(PayloadMode.L2_SEMANTICS)]
            first = run_paired_policy_trials(manifest, candidates, [trace], policies, (7,))
            second = run_paired_policy_trials(manifest, candidates, [trace], list(reversed(policies)), (7,))
            columns = ["policy", "sample_id", "trace_id", "replicate_seed", "selected_mode", "total_wire_bytes", "correct_timely_action"]
            left = first[columns].sort_values(columns[:4]).reset_index(drop=True)
            right = second[columns].sort_values(columns[:4]).reset_index(drop=True)
            pd.testing.assert_frame_equal(left, right)

    def test_static_transport_charges_perception_once_by_candidate_source(self) -> None:
        manifest = pd.DataFrame([{
            "sample_id": "sample-1", "capture_time_us": 1_000_000,
            "trace_time_us": 1_000_000, "deadline_us": 5_000_000,
            "detector_latency_ms": 10.0,
            "detector_energy_j": 0.2, "context_model_latency_ms": 20.0,
            "context_model_energy_j": 0.3,
        }])
        common = {
            "sample_id": "sample-1", "payload_blob": b"payload".hex(),
            "generation_ms": 5.0, "energy_j": 0.1,
            "receiver_latency_ms": 2.0, "receiver_energy_j": 0.01,
            "native_receiver_correct": True,
        }
        candidates = pd.DataFrame([
            {**common, "mode": "L2_SEMANTICS", "variant": "vlm", "source_model": "qwen_vlm"},
            {**common, "mode": "L2_SEMANTICS", "variant": "detector", "source_model": "detector_only"},
            {**common, "mode": "L5_IMAGE_OR_CLIP", "variant": "image", "source_model": "ground_visual_receiver"},
            {**common, "mode": "L4_FEATURES", "variant": "feature", "source_model": "registered_feature_receiver"},
        ])
        result = experiment3_static_transport(
            candidates, manifest, seed=3,
            rates_mbps=(20.0,), rtts_ms=(20.0,), losses=(0.0,),
            loss_models=("iid",),
            jitters_ms=(0.0,), queue_kb=(512,),
            radio_tx_j_per_byte=1e-6, radio_rx_j_per_byte=2e-6,
            require_perception_costs=True,
        ).set_index("variant")

        self.assertEqual(result.loc["vlm", "perception_pipeline"], "detector_plus_vlm")
        self.assertEqual(result.loc["detector", "perception_pipeline"], "detector_only")
        self.assertTrue(result.loc[["image", "feature"], "perception_pipeline"].eq("none").all())
        self.assertEqual(int(result.loc["vlm", "ready_us"]), 1_035_000)
        self.assertEqual(int(result.loc["detector", "ready_us"]), 1_015_000)
        self.assertEqual(int(result.loc["image", "ready_us"]), 1_005_000)
        self.assertEqual(int(result.loc["feature", "ready_us"]), 1_005_000)
        self.assertAlmostEqual(
            result.loc["vlm", "latency_ms"] - result.loc["image", "latency_ms"], 30.0,
        )
        self.assertAlmostEqual(
            result.loc["detector", "latency_ms"] - result.loc["image", "latency_ms"], 10.0,
        )
        nonradio = result["energy_j"] - result["radio_tx_energy_j"] - result["radio_rx_energy_j"]
        self.assertAlmostEqual(nonradio.loc["vlm"], 0.61)
        self.assertAlmostEqual(nonradio.loc["detector"], 0.31)
        self.assertAlmostEqual(nonradio.loc["image"], 0.11)
        self.assertAlmostEqual(nonradio.loc["feature"], 0.11)

    def test_static_transport_replays_stream_backlog_and_queue_drops(self) -> None:
        observations = 6
        manifest_rows = []
        candidate_rows = []
        for index in range(observations):
            capture_us = index * 10_000
            sample_id = f"overload-{index}"
            manifest_rows.append({
                "sample_id": sample_id,
                "capture_time_us": capture_us,
                "trace_time_us": capture_us,
                "deadline_us": capture_us + 3_000_000,
            })
            candidate_rows.append({
                "sample_id": sample_id,
                "mode": "L5_IMAGE_OR_CLIP",
                "variant": "overload-image",
                "source_model": "ground_visual_receiver",
                "payload_blob": (b"x" * 4_000).hex(),
                # A 15 ms non-preemptive generation service is offered every
                # 10 ms, independently of the slower network service.
                "generation_ms": 15.0,
                "energy_j": 0.1,
                "receiver_latency_ms": 7.0,
                "receiver_energy_j": 0.01,
                "native_receiver_correct": True,
            })
        result = experiment3_static_transport(
            pd.DataFrame(candidate_rows), pd.DataFrame(manifest_rows), seed=5,
            rates_mbps=(0.1,), rtts_ms=(20.0,), losses=(0.0,),
            loss_models=("iid",),
            jitters_ms=(0.0,), queue_kb=(12,),
        ).sort_values("capture_us").reset_index(drop=True)

        self.assertEqual(len(result), observations)
        self.assertTrue(result["replicate_seed"].eq(5).all())
        self.assertEqual(
            result["sender_generation_queue_wait_ms"].tolist(),
            [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        )
        self.assertTrue(result["queue_overflow"].astype(bool).any())
        self.assertTrue(result.loc[result["queue_overflow"], "total_wire_bytes"].eq(0).all())
        self.assertGreater(float(result["network_queue_wait_ms"].max()), 0.0)
        self.assertGreater(float(result["stream_offered_load_ratio"].iloc[0]), 1.0)
        self.assertLess(
            float(result["stream_delivered_wire_goodput_bps"].iloc[0]),
            float(result["stream_offered_load_bps"].iloc[0]),
        )
        delivered = result[result["full_message_received"].astype(bool)]
        self.assertGreater(len(delivered), 0)
        self.assertTrue(
            ((delivered["action_ready_us"] - delivered["last_rx_us"]) == 7_000).all()
        )
        self.assertTrue(np.isfinite(result["stream_time_average_aoi_ms"]).all())

        summary = summarize_experiment3_static_transport(result)
        self.assertEqual(len(summary), 1)
        aggregate = summary.iloc[0]
        self.assertEqual(int(aggregate.replicate_seed), 5)
        self.assertEqual(int(aggregate.observations), observations)
        self.assertAlmostEqual(
            float(aggregate.delivery_probability), len(delivered) / observations,
        )
        self.assertAlmostEqual(
            float(aggregate.deadline_success_probability),
            float(result["timely_delivery"].astype(bool).mean()),
        )
        self.assertAlmostEqual(
            float(aggregate.e2e_latency_p50_ms),
            float(delivered["latency_ms"].median()),
        )
        self.assertAlmostEqual(
            float(aggregate.offered_load_bps),
            float(result["stream_offered_load_bps"].iloc[0]),
        )
        self.assertAlmostEqual(
            float(aggregate.time_average_aoi_ms),
            float(result["stream_time_average_aoi_ms"].iloc[0]),
        )

    def test_static_transport_uses_two_stage_perception_encoding_flow_shop(self) -> None:
        manifest_rows = []
        candidate_rows = []
        for index in range(3):
            capture_us = index * 10_000
            sample_id = f"flow-{index}"
            manifest_rows.append({
                "sample_id": sample_id, "capture_time_us": capture_us,
                "trace_time_us": capture_us, "deadline_us": capture_us + 2_000_000,
                "detector_latency_ms": 20.0, "detector_energy_j": 0.1,
                "context_model_latency_ms": 30.0, "context_model_energy_j": 0.2,
            })
            candidate_rows.append({
                "sample_id": sample_id, "mode": "L2_SEMANTICS",
                "variant": "detector-flow", "source_model": "detector_only",
                "payload_blob": b"{}".hex(), "generation_ms": 15.0,
                "energy_j": 0.01, "receiver_latency_ms": 1.0,
                "receiver_energy_j": 0.001, "native_receiver_correct": True,
            })
        result = experiment3_static_transport(
            pd.DataFrame(candidate_rows), pd.DataFrame(manifest_rows), seed=9,
            rates_mbps=(20.0,), rtts_ms=(20.0,), losses=(0.0,),
            loss_models=("iid",), jitters_ms=(0.0,), queue_kb=(128,),
            require_perception_costs=True,
        ).sort_values("capture_us").reset_index(drop=True)
        # Perception for capture n+1 overlaps representation encoding for n;
        # neither server overlaps itself.
        self.assertEqual(result["ready_us"].astype(int).tolist(), [35_000, 55_000, 75_000])
        self.assertEqual(result["perception_queue_wait_ms"].tolist(), [0.0, 10.0, 20.0])
        self.assertEqual(result["representation_queue_wait_ms"].tolist(), [0.0, 0.0, 0.0])
        self.assertTrue(result["compute_flow_model"].eq(
            "two_stage_fcfs_perception_then_representation"
        ).all())

    def test_static_transport_registers_iid_and_burst_conditions(self) -> None:
        manifest = pd.DataFrame([
            {
                "sample_id": "loss-model-1", "capture_time_us": 0,
                "trace_time_us": 0, "deadline_us": 3_000_000,
            },
            {
                "sample_id": "loss-model-2", "capture_time_us": 100_000,
                "trace_time_us": 100_000, "deadline_us": 3_100_000,
            },
        ])
        candidates = pd.DataFrame([
            {
                "sample_id": sample_id, "mode": "L5_IMAGE_OR_CLIP",
                "variant": "registered-image", "source_model": "ground_visual_receiver",
                "payload_blob": (b"x" * 400).hex(), "generation_ms": 1.0,
                "energy_j": 0.01, "receiver_latency_ms": 1.0,
                "receiver_energy_j": 0.001, "native_receiver_correct": True,
            }
            for sample_id in ("loss-model-1", "loss-model-2")
        ])
        kwargs = {
            "rates_mbps": (2.0,), "rtts_ms": (20.0,), "losses": (0.15,),
            "loss_models": ("iid", "gilbert_elliott"),
            "burst_recovery_probability": 0.2, "burst_slot_ms": 100.0,
            "jitters_ms": (0.0,), "queue_kb": (128,),
            "require_registered_loss_models": True,
        }
        first = experiment3_static_transport(candidates, manifest, seed=31, **kwargs)
        second = experiment3_static_transport(candidates, manifest, seed=31, **kwargs)
        self.assertEqual(set(first["loss_model"]), {"iid", "gilbert_elliott"})
        self.assertEqual(len(first), 2 * len(candidates))
        columns = [
            "observation_id", "loss_model", "message_id", "total_wire_bytes",
            "full_message_received", "latency_ms",
        ]
        pd.testing.assert_frame_equal(
            first[columns].sort_values(columns[:2]).reset_index(drop=True),
            second[columns].sort_values(columns[:2]).reset_index(drop=True),
        )
        with self.assertRaises(StudyGateError):
            experiment3_static_transport(
                candidates, manifest, seed=31,
                **{**kwargs, "loss_models": ("iid",)},
            )

    def test_static_transport_requires_exact_coverage_in_every_seed(self) -> None:
        manifest = pd.DataFrame([
            {
                "sample_id": sample_id, "capture_time_us": index * 100_000,
                "trace_time_us": index * 100_000,
                "deadline_us": index * 100_000 + 2_000_000,
            }
            for index, sample_id in enumerate(("seed-pair-1", "seed-pair-2"))
        ])
        candidates = pd.DataFrame([
            {
                "sample_id": sample_id, "mode": "L5_IMAGE_OR_CLIP",
                "variant": "paired-image", "source_model": "ground_visual_receiver",
                "payload_blob": b"image".hex(), "generation_ms": 1.0,
                "energy_j": 0.01, "receiver_latency_ms": 1.0,
                "receiver_energy_j": 0.001, "native_receiver_correct": True,
            }
            for sample_id in ("seed-pair-1", "seed-pair-2")
        ])
        kwargs = {
            "rates_mbps": (2.0,), "rtts_ms": (20.0,), "losses": (0.0,),
            "loss_models": ("iid",), "jitters_ms": (0.0,), "queue_kb": (128,),
        }
        paired = pd.concat([
            experiment3_static_transport(candidates, manifest, seed=1, **kwargs),
            experiment3_static_transport(candidates, manifest, seed=2, **kwargs),
        ], ignore_index=True)
        assert_experiment3_exact_replicate_coverage(paired, (1, 2))
        truncated = paired.drop(
            paired.loc[paired["replicate_seed"].eq(2)].index[0]
        ).reset_index(drop=True)
        with self.assertRaises(StudyGateError):
            assert_experiment3_exact_replicate_coverage(truncated, (1, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
