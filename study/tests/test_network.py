from __future__ import annotations

import unittest

from semuav.network import Direction, LinkTrace, SelectiveRepeatSimulator, SerializationSpan, constant_trace
from semuav.protocol import (
    AppMessage,
    PayloadMode,
    ReliabilityProfile,
    StateKind,
    encode_mapping,
)


def semantic_message(
    message_id: int,
    resulting_seq: int,
    capture_us: int,
    reliability: ReliabilityProfile,
) -> tuple[AppMessage, str]:
    packet = {
        "schema_version": "semuav.v1",
        "observation_id": f"obs-{message_id}",
        "mission": "urban",
        "capture_time_us": capture_us,
        "telemetry": {
            "gps_lat": 24.5,
            "gps_lon": 54.4,
            "altitude_m": 80.0,
            "yaw_deg": 10.0,
        },
        "sufficient_evidence": True,
        "event": "visible_road_obstruction",
        "objects": [],
        "relations": [],
        "risk": 0.9,
        "uncertainty": 0.1,
        "recommended_action": "RESPOND",
    }
    payload, wire_format = encode_mapping(packet, scientific=False)
    return (
        AppMessage(
            "network-test",
            packet["observation_id"],
            message_id,
            PayloadMode.L2_SEMANTICS,
            capture_us,
            capture_us,
            capture_us + 500_000,
            capture_us + 1_000_000,
            5,
            payload,
            StateKind.KEYFRAME,
            None,
            resulting_seq,
            reliability,
        ),
        wire_format,
    )


class SelectiveAckLossSimulator(SelectiveRepeatSimulator):
    """Deterministically drop selected ACKs without relying on hash draws."""

    def __init__(self, *args, drop_app_once=(), drop_fragment_once=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.drop_app_once = set(drop_app_once)
        self.drop_fragment_once = set(drop_fragment_once)

    def _loss_and_delay(
        self,
        trace: LinkTrace,
        span: SerializationSpan,
        key: tuple[object, ...],
    ) -> tuple[bool, int]:
        lost, delay = super()._loss_and_delay(trace, span, key)
        if trace.direction is Direction.DOWNLINK:
            if len(key) == 3 and key[1] == "app_ack":
                selector = (int(key[0]), int(key[2]))
                if selector in self.drop_app_once:
                    self.drop_app_once.remove(selector)
                    return True, delay
            if len(key) == 4 and key[-1] == "frag_ack":
                selector = (int(key[0]), int(key[1]), int(key[2]))
                if selector in self.drop_fragment_once:
                    self.drop_fragment_once.remove(selector)
                    return True, delay
        return lost, delay


class ApplicationAckStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.uplink = constant_trace("ack-state", Direction.UPLINK, 5.0, 1.0, 0.0)
        self.downlink = constant_trace("ack-state", Direction.DOWNLINK, 5.0, 1.0, 0.0)
        self.reliability = ReliabilityProfile(
            window_fragments=8,
            retry_limit=1,
            min_rto_us=100_000,
            max_rto_us=100_000,
            delayed_ack_us=0,
        )

    def test_sender_transitions_pending_to_uncertain_on_application_ack_timeout(self) -> None:
        message, wire_format = semantic_message(101, 1, 0, self.reliability)
        lossy_downlink = constant_trace("ack-state", Direction.DOWNLINK, 5.0, 1.0, 1.0)
        simulator = SelectiveRepeatSimulator(self.uplink, lossy_downlink, seed=3, decode_us=0)
        observed: dict[str, str] = {}

        def capture(sim, now_us, label):
            observed[str(label)] = sim.sender_belief.status
            return None

        _, events = simulator.run(
            [],
            captures=[
                (0, (message, {"wire_format": wire_format})),
                (50_000, "before"),
                (150_000, "after"),
            ],
            capture_handler=lambda sim, now, value: (
                value if isinstance(value, tuple) else capture(sim, now, value)
            ),
        )

        self.assertEqual(observed["before"], "pending")
        self.assertEqual(observed["after"], "uncertain")
        self.assertIn("application_ack_timeout", set(events["kind"]))
        self.assertEqual(simulator.sender_belief.confirmed_seq, 0)

    def test_duplicate_fragment_repeats_application_ack_and_repairs_belief(self) -> None:
        message, wire_format = semantic_message(102, 1, 0, self.reliability)
        simulator = SelectiveAckLossSimulator(
            self.uplink,
            self.downlink,
            seed=4,
            decode_us=0,
            drop_app_once={(102, 1)},
            drop_fragment_once={(102, 0, 1)},
        )
        results, events = simulator.run(
            [],
            captures=[(0, (message, {"wire_format": wire_format}))],
            capture_handler=lambda _sim, _now, value: value,
        )

        self.assertGreater(int(results.iloc[0].retry_uplink_bytes), 0)
        self.assertIn("application_ack_repeat", set(events["kind"]))
        self.assertEqual(simulator.sender_belief.confirmed_seq, 1)
        self.assertEqual(simulator.sender_belief.status, "known")

    def test_later_cumulative_application_ack_clears_older_pending_state(self) -> None:
        first, first_format = semantic_message(103, 1, 0, self.reliability)
        second, second_format = semantic_message(104, 2, 50_000, self.reliability)
        simulator = SelectiveAckLossSimulator(
            self.uplink,
            self.downlink,
            seed=5,
            decode_us=0,
            drop_app_once={(103, 1)},
        )
        _, events = simulator.run(
            [],
            captures=[
                (0, (first, {"wire_format": first_format})),
                (50_000, (second, {"wire_format": second_format})),
            ],
            capture_handler=lambda _sim, _now, value: value,
        )

        arrivals = events[events["kind"].eq("application_ack_arrive")]
        self.assertNotIn(103, set(arrivals["message_id"].dropna().astype(int)))
        self.assertIn(104, set(arrivals["message_id"].dropna().astype(int)))
        self.assertEqual(simulator.sender_belief.confirmed_seq, 2)
        self.assertEqual(simulator.sender_belief.status, "known")
        self.assertFalse(simulator._pending_state_message_ids)
        self.assertFalse(simulator._timed_out_state_message_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
