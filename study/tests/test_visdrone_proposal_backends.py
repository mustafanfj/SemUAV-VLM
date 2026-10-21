from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from semuav.perception import (
    HFMultimodalBackend,
    InferenceRequest,
    ModelSpec,
    MoondreamBackend,
    VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256,
    VISDRONE_PROPOSAL_DECISION_CONTRACT_VERSION,
    calibrated_semantic_packet_cache,
)


class _FakeScalar:
    def __init__(self, value: int):
        self.value = int(value)

    def detach(self) -> "_FakeScalar":
        return self

    def cpu(self) -> "_FakeScalar":
        return self

    def item(self) -> int:
        return self.value


class _FakeTensor:
    def __init__(self, values: list[list[int]] | np.ndarray):
        self.values = np.asarray(values, dtype=np.int64)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def to(self, device: str) -> "_FakeTensor":
        del device
        return self

    def __getitem__(self, key):
        value = self.values[key]
        if np.isscalar(value):
            return _FakeScalar(int(value))
        return _FakeTensor(value)


class _InferenceMode:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback
        return False


class _Cuda:
    @staticmethod
    def reset_peak_memory_stats(device):
        del device

    @staticmethod
    def synchronize(device):
        del device

    @staticmethod
    def max_memory_allocated(device):
        del device
        return 0


FAKE_TORCH = SimpleNamespace(
    cuda=_Cuda(),
    inference_mode=lambda: _InferenceMode(),
)


class ProposalDecisionBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.image_path = Path(self.temporary.name) / "train-canary.png"
        Image.new("RGB", (20, 10), "white").save(self.image_path)

    def request(self, *, mission: str = "visdrone_urban") -> InferenceRequest:
        telemetry: dict[str, object] = {
            "gps_lat": 24.0,
            "gps_lon": 54.0,
            "altitude_m": 70.0,
            "yaw_deg": 0.0,
        }
        proposals: tuple[dict[str, object], ...] = ()
        if mission == "visdrone_urban":
            telemetry.update({
                "mission_profile": "heavy_vehicle",
                "registered_roi_xyxy": [0.25, 0.25, 0.75, 0.85],
            })
            proposals = ({
                "label": "truck",
                "bbox_normalized_xyxy": [0.3, 0.3, 0.4, 0.4],
                "confidence": 0.91,
                "track_id": 41,
            },)
        return InferenceRequest(
            observation_id="train-canary",
            image_path=str(self.image_path),
            mission=mission,
            capture_time_us=5,
            telemetry=telemetry,
            detector_proposals=proposals,
        )

    @staticmethod
    def profile(operation, **kwargs):
        del kwargs
        return operation(), {
            "latency_ms": 3.0,
            "energy_j": 0.2,
            "peak_temperature_c": 42.0,
        }

    def hf_backend(self, raw: str, *, last_token_id: int, output_tokens: int = 2):
        class Processor:
            tokenizer = SimpleNamespace(eos_token_id=99)

            def __init__(self):
                self.messages = None

            def apply_chat_template(self, messages, **kwargs):
                del kwargs
                self.messages = messages
                return {"input_ids": _FakeTensor([[10, 11]])}

            def batch_decode(
                self,
                tokens,
                *,
                skip_special_tokens,
                clean_up_tokenization_spaces,
            ):
                del tokens, skip_special_tokens
                if clean_up_tokenization_spaces is not False:
                    raise AssertionError("decision decoding must preserve exact spacing")
                # Qwen's registered v2r2 transport prepends one host-authored
                # opening brace and decodes only the generated continuation.
                return [raw[1:]]

        class Model:
            generation_config = SimpleNamespace(eos_token_id=[98, 99])
            config = SimpleNamespace(eos_token_id=99)

            def generate(self, **kwargs):
                del kwargs
                generated = [20] * max(0, output_tokens - 1) + [last_token_id]
                return SimpleNamespace(
                    sequences=_FakeTensor([[10, 11, *generated]]),
                    scores=[],
                )

        backend = object.__new__(HFMultimodalBackend)
        backend.torch = FAKE_TORCH
        backend.device = "cpu"
        backend.processor = Processor()
        backend.model = Model()
        backend.spec = ModelSpec(
            "qwen3_vl_2b",
            "Qwen/Qwen3-VL-2B-Instruct",
            "a" * 40,
            image_size=(8, 8),
        )
        return backend

    def test_hf_uses_bounded_context_and_accepts_eos_on_final_allowed_token(self) -> None:
        raw = json.dumps({
            "e": True,
            "a": [0],
            "l": [],
        }, separators=(",", ":"))
        backend = self.hf_backend(raw, last_token_id=99, output_tokens=2)
        with patch("semuav.hardware.profile_callable", side_effect=self.profile):
            record = backend.infer(self.request(), prompt="BOUNDED", max_new_tokens=2)

        self.assertTrue(record.schema_valid)
        self.assertTrue(record.decision_core_valid)
        self.assertTrue(record.terminated_by_eos)
        self.assertFalse(record.reached_token_cap)
        self.assertEqual(record.raw_text, raw)
        self.assertEqual(record.parsed["recommended_action"], "RESPOND")
        self.assertEqual(record.parsed["objects"][0]["track_id"], 41)
        self.assertEqual(
            record.decision_contract_version,
            VISDRONE_PROPOSAL_DECISION_CONTRACT_VERSION,
        )
        self.assertEqual(
            record.decision_composer_policy_sha256,
            VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256,
        )
        text = backend.processor.messages[0]["content"][1]["text"]
        context = json.loads(text.split("CONTEXT=", 1)[1])
        self.assertEqual(context["p"], [[0, "truck", 1228, 1638, 1638, 1843]])
        for forbidden in ("confidence", "track_id", "mission_profile", "telemetry"):
            self.assertNotIn(forbidden, json.dumps(context, sort_keys=True))

    def test_hf_rejects_an_unterminated_decision_at_the_token_cap(self) -> None:
        raw = json.dumps({
            "e": True,
            "a": [0],
            "l": [],
        })
        backend = self.hf_backend(raw, last_token_id=97, output_tokens=2)
        with patch("semuav.hardware.profile_callable", side_effect=self.profile):
            record = backend.infer(self.request(), prompt="BOUNDED", max_new_tokens=2)
        self.assertFalse(record.schema_valid)
        self.assertTrue(record.decision_core_valid)
        self.assertFalse(record.terminated_by_eos)
        self.assertTrue(record.reached_token_cap)
        self.assertIn("max_new_tokens", record.error)

    def test_calibration_preserves_host_composer_lineage(self) -> None:
        raw = json.dumps({
            "e": True,
            "a": [0],
            "l": [],
        })
        backend = self.hf_backend(raw, last_token_id=99, output_tokens=2)
        with patch("semuav.hardware.profile_callable", side_effect=self.profile):
            record = backend.infer(self.request(), prompt="BOUNDED", max_new_tokens=2)
        row = record.as_row()
        self.assertFalse(any(key.startswith("moondream_") for key in row))
        row["confidence"] = 0.8
        cache = calibrated_semantic_packet_cache(
            pd.DataFrame([row]),
            1.0,
            source_model="qwen3_vl_2b",
        )
        packet = json.loads(cache.iloc[0]["packet_json"])
        self.assertEqual(
            packet["provenance"]["semantic_packet_authorship"],
            "deterministic_host_composer",
        )
        for name, expected in (
            ("decision_contract_version", VISDRONE_PROPOSAL_DECISION_CONTRACT_VERSION),
            ("decision_context_sha256", record.decision_context_sha256),
            (
                "decision_composer_policy_sha256",
                VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256,
            ),
        ):
            self.assertEqual(packet["provenance"][name], expected)
            self.assertEqual(cache.iloc[0][name], expected)

    def test_moondream_uses_the_same_core_and_composer(self) -> None:
        raw = json.dumps({
            "e": True,
            "a": [0],
            "l": [0],
        })

        class Model:
            def __init__(self):
                self.query_text = None
                self.model = SimpleNamespace(
                    tokenizer=SimpleNamespace(
                        encode=lambda question: SimpleNamespace(ids=[10] * 100),
                    ),
                    config=SimpleNamespace(
                        text=SimpleNamespace(max_context=2048, vocab_size=51200),
                        tokenizer=SimpleNamespace(templates={
                            "query": {"prefix": [1, 15381, 2], "suffix": [3]},
                        }),
                    ),
                )

            def encode_image(self, image):
                del image
                return SimpleNamespace(pos=730)

            def query(self, *, image, question, settings):
                self.encoded_image = image
                self.query_text = question
                self.settings = settings
                return {"answer": raw}

        backend = object.__new__(MoondreamBackend)
        backend.torch = FAKE_TORCH
        backend.device = "cpu"
        backend.model = Model()
        backend.spec = ModelSpec(
            "moondream2", "vikhyatk/moondream2", "b" * 40,
            image_size=(8, 8),
        )
        with patch.dict(
            "semuav.perception.VISDRONE_GENERATION_STRATEGY_BY_MODEL",
            {"moondream2": "native_query"},
            clear=False,
        ), patch("semuav.hardware.profile_callable", side_effect=self.profile):
            record = backend.infer(self.request(), prompt="BOUNDED", max_new_tokens=256)
        self.assertTrue(record.schema_valid)
        self.assertTrue(record.decision_core_valid)
        self.assertEqual(record.parsed["event"], "priority_target_low_visibility")
        self.assertEqual(record.parsed["recommended_action"], "REVIEW")
        self.assertIsNone(record.terminated_by_eos)
        self.assertIsNone(record.reached_token_cap)
        self.assertEqual(backend.model.settings, {"max_tokens": 256, "temperature": 0.0})
        context = json.loads(backend.model.query_text.split("CONTEXT=", 1)[1])
        self.assertEqual(context["v"], 2)

    def test_moondream_preserves_the_legacy_full_packet_path(self) -> None:
        packet = {
            "schema_version": "semuav.v1",
            "observation_id": "train-canary",
            "mission": "urban",
            "capture_time_us": 5,
            "telemetry": {
                "gps_lat": 24.0,
                "gps_lon": 54.0,
                "altitude_m": 70.0,
                "yaw_deg": 0.0,
            },
            "sufficient_evidence": True,
            "event": "none_visible",
            "objects": [],
            "relations": [],
            "evidence": [],
            "risk": 0.0,
            "uncertainty": 0.0,
            "recommended_action": "NO_ACTION",
        }

        class Model:
            def __init__(self):
                self.model = SimpleNamespace(
                    tokenizer=SimpleNamespace(
                        encode=lambda question: SimpleNamespace(ids=[10] * 100),
                    ),
                    config=SimpleNamespace(
                        text=SimpleNamespace(max_context=2048, vocab_size=51200),
                        tokenizer=SimpleNamespace(templates={
                            "query": {"prefix": [1, 15381, 2], "suffix": [3]},
                        }),
                    ),
                )

            def encode_image(self, image):
                del image
                return SimpleNamespace(pos=730)

            def query(self, *, image, question, settings):
                del image, question, settings
                return {"answer": json.dumps(packet)}

        backend = object.__new__(MoondreamBackend)
        backend.torch = FAKE_TORCH
        backend.device = "cpu"
        backend.model = Model()
        backend.spec = ModelSpec(
            "legacy", "owner/legacy", "c" * 40, image_size=(8, 8),
        )
        with patch("semuav.hardware.profile_callable", side_effect=self.profile):
            record = backend.infer(self.request(mission="urban"), prompt="LEGACY")
        self.assertTrue(record.schema_valid)
        self.assertEqual(record.parsed, packet)
        self.assertIsNone(record.decision_contract_version)
        self.assertIsNone(record.decision_core_valid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
