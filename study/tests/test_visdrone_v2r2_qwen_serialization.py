from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from semuav.perception import (
    HFMultimodalBackend,
    InferenceRequest,
    ModelSpec,
    QWEN3_VL_JSON_ASSISTANT_PREFILL,
    SMOLVLM2_JSON_ASSISTANT_PREFILL,
    VISDRONE_GENERATION_STRATEGY_BY_MODEL,
    visdrone_assistant_prefill_for_model,
    visdrone_generation_policy,
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
    def __init__(self, values):
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


class _RecordingProcessor:
    tokenizer = SimpleNamespace(eos_token_id=99)

    def __init__(self, decoded_continuation: str, input_ids: list[int]):
        self.decoded_continuation = decoded_continuation
        self.input_ids = list(input_ids)
        self.messages = None
        self.template_kwargs = None
        self.rendered_prompt = "<native-smol-generation-prompt>"
        self.direct_text = None
        self.direct_images = None
        self.direct_kwargs = None
        self.decoded_token_ids = None

    def apply_chat_template(self, messages, **kwargs):
        # Transformers 4.57.1's multimodal ProcessorMixin traverses every
        # content item as a typed mapping before rendering the native template.
        # A raw assistant string would fail here before model.generate().
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list) or not content:
                raise TypeError("multimodal message content must be typed blocks")
            for item in content:
                if not isinstance(item, dict) or "type" not in item:
                    raise TypeError("multimodal content item must be a typed mapping")
        self.messages = messages
        self.template_kwargs = dict(kwargs)
        if kwargs.get("tokenize") is False:
            return self.rendered_prompt
        return {"input_ids": _FakeTensor([self.input_ids])}

    def __call__(self, *, text, images, **kwargs):
        self.direct_text = text
        self.direct_images = images
        self.direct_kwargs = dict(kwargs)
        return {"input_ids": _FakeTensor([self.input_ids])}

    def batch_decode(
        self,
        tokens,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces,
    ):
        if not skip_special_tokens:
            raise AssertionError("special tokens must not be decoded into the candidate")
        if clean_up_tokenization_spaces is not False:
            raise AssertionError("decision decoding must preserve exact spacing")
        self.decoded_token_ids = tokens.values.tolist()
        return [self.decoded_continuation]


class _RecordingModel:
    generation_config = SimpleNamespace(eos_token_id=[98, 99])
    config = SimpleNamespace(eos_token_id=99)

    def __init__(self, input_ids: list[int], generated_ids: list[int]):
        self.input_ids = list(input_ids)
        self.generated_ids = list(generated_ids)
        self.generate_kwargs = None
        self.generate_calls = 0

    def generate(self, **kwargs):
        self.generate_calls += 1
        self.generate_kwargs = dict(kwargs)
        return SimpleNamespace(
            sequences=_FakeTensor([[*self.input_ids, *self.generated_ids]]),
            scores=[],
        )


class V2R2QwenSerializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.image_path = Path(self.temporary.name) / "registered-train-canary.png"
        Image.new("RGB", (20, 10), "white").save(self.image_path)

    def request(self) -> InferenceRequest:
        return InferenceRequest(
            observation_id="registered-train-canary",
            image_path=str(self.image_path),
            mission="visdrone_urban",
            capture_time_us=7,
            telemetry={
                "gps_lat": 24.0,
                "gps_lon": 54.0,
                "altitude_m": 70.0,
                "yaw_deg": 0.0,
                "mission_profile": "heavy_vehicle",
                "registered_roi_xyxy": [0.25, 0.25, 0.75, 0.85],
            },
            detector_proposals=({
                "label": "truck",
                "bbox_normalized_xyxy": [0.3, 0.3, 0.4, 0.4],
                "confidence": 0.91,
                "track_id": 41,
            },),
        )

    @staticmethod
    def profile(operation, **kwargs):
        del kwargs
        return operation(), {
            "latency_ms": 3.0,
            "energy_j": 0.2,
            "peak_temperature_c": 42.0,
        }

    def backend(
        self,
        model_id: str,
        decoded_continuation: str,
        *,
        input_ids: list[int] | None = None,
        generated_ids: list[int] | None = None,
    ) -> HFMultimodalBackend:
        input_ids = list(input_ids or [10, 11, 12, 13, 14])
        generated_ids = list(generated_ids or [21, 22, 99])
        backend = object.__new__(HFMultimodalBackend)
        backend.torch = FAKE_TORCH
        backend.device = "cpu"
        backend.processor = _RecordingProcessor(decoded_continuation, input_ids)
        backend.model = _RecordingModel(input_ids, generated_ids)
        backend.spec = ModelSpec(
            model_id,
            "registered/test-model",
            "a" * 40,
            image_size=(8, 8),
        )
        return backend

    def infer(self, backend: HFMultimodalBackend, *, max_new_tokens: int = 8):
        with patch("semuav.hardware.profile_callable", side_effect=self.profile):
            return backend.infer(
                self.request(),
                prompt="FROZEN DECISION PROMPT",
                max_new_tokens=max_new_tokens,
            )

    @staticmethod
    def valid_decision() -> str:
        return json.dumps({
            "e": True,
            "a": [0],
            "l": [],
        }, separators=(",", ":"))

    def test_registered_qwen_ids_continue_one_assistant_open_brace(self) -> None:
        candidate = self.valid_decision()
        continuation = candidate[1:]

        self.assertEqual(QWEN3_VL_JSON_ASSISTANT_PREFILL, "{")
        self.assertEqual(len(QWEN3_VL_JSON_ASSISTANT_PREFILL), 1)
        for model_id in (
            "qwen3_vl_2b",
            "qwen3_vl_8b_upper_bound",
        ):
            with self.subTest(model_id=model_id):
                backend = self.backend(model_id, continuation)
                record = self.infer(backend)

                self.assertEqual(
                    VISDRONE_GENERATION_STRATEGY_BY_MODEL[model_id],
                    "qwen_open_brace_assistant_continuation",
                )
                self.assertTrue(record.schema_valid)
                self.assertEqual(record.raw_text, candidate)
                self.assertEqual(backend.model.generate_calls, 1)
                self.assertEqual(len(backend.processor.messages), 2)
                self.assertEqual(
                    backend.processor.messages[1],
                    {
                        "role": "assistant",
                        "content": [{
                            "type": "text",
                            "text": QWEN3_VL_JSON_ASSISTANT_PREFILL,
                        }],
                    },
                )
                self.assertFalse(
                    backend.processor.template_kwargs["add_generation_prompt"]
                )
                self.assertTrue(
                    backend.processor.template_kwargs["continue_final_message"]
                )

    def test_smol_uses_native_user_only_render_then_exact_double_quote_prefix(self) -> None:
        candidate = self.valid_decision()
        continuation = candidate[len(SMOLVLM2_JSON_ASSISTANT_PREFILL):]
        backend = self.backend("smolvlm2_2b", continuation)

        record = self.infer(backend)

        self.assertEqual(SMOLVLM2_JSON_ASSISTANT_PREFILL, '{"')
        self.assertEqual(
            VISDRONE_GENERATION_STRATEGY_BY_MODEL["smolvlm2_2b"],
            "smol_native_double_quote_assistant_prefix",
        )
        self.assertTrue(record.schema_valid)
        self.assertEqual(record.raw_text, candidate)
        self.assertEqual(backend.model.generate_calls, 1)
        self.assertEqual(len(backend.processor.messages), 1)
        self.assertEqual(backend.processor.messages[0]["role"], "user")
        self.assertEqual(
            backend.processor.template_kwargs,
            {"add_generation_prompt": True, "tokenize": False},
        )
        self.assertEqual(
            backend.processor.direct_text,
            backend.processor.rendered_prompt + SMOLVLM2_JSON_ASSISTANT_PREFILL,
        )
        self.assertEqual(len(backend.processor.direct_images), 1)
        self.assertEqual(backend.processor.direct_images[0].size, (8, 8))
        self.assertEqual(
            backend.processor.direct_kwargs,
            {"return_tensors": "pt"},
        )

    def test_candidate_is_exactly_host_prefill_plus_generated_continuation(self) -> None:
        candidate = self.valid_decision()
        for model_id in (
            "qwen3_vl_2b",
            "smolvlm2_2b",
            "qwen3_vl_8b_upper_bound",
        ):
            with self.subTest(model_id=model_id):
                prefill = visdrone_assistant_prefill_for_model(model_id)
                continuation = candidate[len(prefill):]
                backend = self.backend(model_id, continuation)
                record = self.infer(backend)

                self.assertEqual(
                    record.raw_text,
                    prefill + continuation,
                )
                self.assertNotEqual(record.raw_text, continuation)

    def test_smolvlm_invalid_json_is_not_repaired_or_retried(self) -> None:
        continuation = (
            "e: true, a: [0], l: []}"
        )
        backend = self.backend("smolvlm2_2b", continuation)

        record = self.infer(backend)

        self.assertFalse(record.schema_valid)
        self.assertFalse(record.decision_core_valid)
        self.assertIsNone(record.parsed)
        self.assertEqual(
            record.raw_text,
            SMOLVLM2_JSON_ASSISTANT_PREFILL + continuation,
        )
        self.assertEqual(backend.model.generate_calls, 1)
        self.assertIn("not one exact JSON object", str(record.error))

        policy = visdrone_generation_policy(8)
        self.assertEqual(
            policy["version"], "semuav.visdrone-generation-policy.v5"
        )
        self.assertEqual(
            policy["smol_native_double_quote_assistant_prefix"],
            {
                "assistant_prefill": '{"',
                "assistant_prefill_authorship": "host_transport_scaffold",
                "assistant_content_shape": (
                    "native_user_only_render_plus_text_prefix"
                ),
                "add_generation_prompt": True,
                "continue_final_message": False,
                "chat_template_tokenize": False,
                "processor_retokenize_with_image": True,
            },
        )
        self.assertEqual(policy["generation"]["retry_count"], 0)
        self.assertEqual(
            policy["transport_validation"],
            {
                "strict_exact_json_object": True,
                "strip_markdown_fences": False,
                "extract_substring": False,
                "repair_response": False,
            },
        )

    def test_strict_parser_rejects_every_unregistered_transport_shape(self) -> None:
        candidate = self.valid_decision()
        invalid_continuations = {
            "markdown_fence": "```json\n" + candidate + "\n```",
            "trailing_prose": candidate[1:] + " trailing prose",
            "duplicate_key": (
                '"e":true,"e":false,"a":[0],"l":[]}'
            ),
            "second_open_brace": candidate,
        }

        for label, continuation in invalid_continuations.items():
            with self.subTest(label=label):
                backend = self.backend("qwen3_vl_2b", continuation)
                record = self.infer(backend)

                self.assertFalse(record.schema_valid)
                self.assertFalse(record.decision_core_valid)
                self.assertIsNone(record.parsed)
                self.assertEqual(
                    record.raw_text,
                    QWEN3_VL_JSON_ASSISTANT_PREFILL + continuation,
                )
                self.assertTrue(record.error)

    def test_token_counts_and_eos_apply_only_to_generated_continuation(self) -> None:
        candidate = self.valid_decision()
        continuation = candidate[1:]
        input_ids = [10, 11, 12, 13, 14]
        generated_ids = [31, 32, 99]
        backend = self.backend(
            "qwen3_vl_2b",
            continuation,
            input_ids=input_ids,
            generated_ids=generated_ids,
        )

        record = self.infer(backend, max_new_tokens=len(generated_ids))

        self.assertTrue(record.schema_valid)
        self.assertEqual(record.input_tokens, len(input_ids))
        self.assertEqual(record.output_tokens, len(generated_ids))
        self.assertEqual(backend.processor.decoded_token_ids, [generated_ids])
        self.assertTrue(record.terminated_by_eos)
        self.assertFalse(record.reached_token_cap)


if __name__ == "__main__":
    unittest.main(verbosity=2)
