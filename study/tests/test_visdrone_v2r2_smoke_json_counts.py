from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

from semuav.perception import (
    VISDRONE_GENERATION_POLICY_VERSION,
    VISDRONE_GENERATION_STRATEGY_BY_MODEL,
    VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256,
    aspect_preserving_letterbox,
    visdrone_assistant_prefill_for_model,
    visdrone_generation_policy_sha256,
    visdrone_proposal_decision_context,
)
from semuav.registry import StudyGateError, hash_file, hash_object, sha256_bytes
from semuav.visdrone_models import (
    OFFICIAL_COMPACT_MODEL_REPOS,
    OFFICIAL_MODEL_BACKENDS,
    OFFICIAL_UPPER_BOUND_ID,
    OFFICIAL_UPPER_BOUND_REPO,
    freeze_huggingface_models,
    model_loader_environment,
    run_one_image_runtime_smoke,
)


def _registered_models() -> dict[str, object]:
    return {
        "vlms": [
            {
                "id": model_id,
                "repo": repo,
                "backend": OFFICIAL_MODEL_BACKENDS[model_id],
                "revision": "REQUIRED_40_HEX_COMMIT",
                "quantization": "none",
            }
            for model_id, repo in OFFICIAL_COMPACT_MODEL_REPOS.items()
        ],
        "upper_bound": {
            "id": OFFICIAL_UPPER_BOUND_ID,
            "repo": OFFICIAL_UPPER_BOUND_REPO,
            "backend": "qwen3_vl",
            "revision": "REQUIRED_40_HEX_COMMIT",
            "quantization": "none",
        },
    }


class V2R2SmokeJsonTokenCountTests(unittest.TestCase):
    def _freeze(self, root: Path, cache: Path | None = None):
        cache = cache or (root / "hf-cache")
        snapshots: dict[tuple[str, str], Path] = {}

        def resolver(repo, requested):
            self.assertIsNone(requested)
            return hashlib.sha1(repo.encode()).hexdigest()

        def downloader(repo, revision, cache_root):
            snapshot = cache_root / "snapshots" / revision
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text(
                json.dumps({"repo": repo, "revision": revision}),
                encoding="utf-8",
            )
            (snapshot / "weights.bin").write_bytes(repo.encode() * 2)
            snapshots[(repo, revision)] = snapshot.resolve()
            return snapshot

        lock_path = root / "MODEL_LOCK.json"
        freeze_huggingface_models(
            _registered_models(),
            cache_root=cache,
            lock_path=lock_path,
            resolver=resolver,
            downloader=downloader,
        )
        return lock_path, snapshots

    def test_historical_lock_root_controls_every_transformers_cache_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            historical_cache = root / ".cache" / "huggingface"
            lock_path, _ = self._freeze(root, historical_cache)
            environment = model_loader_environment(lock_path)
            expected = str(historical_cache.resolve())

            self.assertEqual(environment["HF_HUB_CACHE"], expected)
            self.assertEqual(environment["HUGGINGFACE_HUB_CACHE"], expected)
            self.assertEqual(environment["TRANSFORMERS_CACHE"], expected)
            self.assertNotEqual(environment["HF_HUB_CACHE"], expected + "/hub")

    def _fixture(self, root: Path):
        lock_path, snapshots = self._freeze(root)
        image = root / "frame.jpg"
        Image.new("RGB", (16, 12), (1, 2, 3)).save(image)
        request = SimpleNamespace(
            observation_id="visdrone:det_train:smoke",
            image_path=str(image),
            mission="visdrone_urban",
            capture_time_us=123,
            telemetry={"battery_fraction": 0.9},
            detector_proposals=({
                "label": "car",
                "bbox_normalized_xyxy": [0.1, 0.2, 0.3, 0.4],
                "confidence": 0.8,
            },),
        )
        with Image.open(image) as opened_image:
            source_image = opened_image.convert("RGB")
        _, preprocess = aspect_preserving_letterbox(source_image, (768, 768))
        context_sha256 = hash_object(visdrone_proposal_decision_context(
            request,
            source_image.size,
            (768, 768),
            preprocess=preprocess,
        ))
        return lock_path, snapshots, request, context_sha256

    @staticmethod
    def _factory(context_sha256: str, *, invalid_count=None):
        class FakeBackend:
            def __init__(self, entry):
                self.entry = entry

            def infer(self, request, prompt, max_new_tokens):
                model_id = self.entry["model_id"]
                raw_text = (
                    '{"accepted_proposal_ids":[0],'
                    '"limited_visibility_proposal_ids":[],'
                    '"sufficient_evidence":true}'
                )
                strategy = VISDRONE_GENERATION_STRATEGY_BY_MODEL[model_id]
                prefill = visdrone_assistant_prefill_for_model(model_id)
                continuation = raw_text[len(prefill):]
                input_tokens = 100
                if invalid_count is not None and model_id == "qwen3_vl_2b":
                    input_tokens = invalid_count
                return SimpleNamespace(
                    observation_id=request.observation_id,
                    model_id=model_id,
                    model_revision=self.entry["resolved_revision"],
                    schema_valid=True,
                    error=None,
                    latency_ms=5.0,
                    raw_text=raw_text,
                    parsed={"schema_version": "semuav.v1"},
                    input_tokens=input_tokens,
                    output_tokens=20,
                    prompt_hash=hash_object(prompt),
                    decision_contract_version=(
                        "semuav.visdrone-proposal-decision.v2"
                    ),
                    decision_core_valid=True,
                    decision_core_json=raw_text,
                    decision_context_sha256=context_sha256,
                    decision_proposal_count=1,
                    decision_composer_policy_sha256=(
                        VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256
                    ),
                    terminated_by_eos=True,
                    reached_token_cap=False,
                    generation_policy_version=VISDRONE_GENERATION_POLICY_VERSION,
                    generation_policy_sha256=visdrone_generation_policy_sha256(
                        max_new_tokens
                    ),
                    generation_strategy=strategy,
                    generation_max_new_tokens=max_new_tokens,
                    assistant_prefill_sha256=sha256_bytes(prefill.encode("utf-8")),
                    assistant_prefill_characters=len(prefill),
                    generated_continuation_sha256=sha256_bytes(
                        continuation.encode("utf-8")
                    ),
                )

        return lambda entry, device, dtype: FakeBackend(entry)

    def test_three_model_hf_counts_are_json_integers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path, snapshots, request, context_sha256 = self._fixture(root)
            output_path = root / "MODEL_RUNTIME_SMOKE.json"
            with mock.patch.dict(
                os.environ, model_loader_environment(lock_path), clear=False,
            ):
                frame = run_one_image_runtime_smoke(
                    lock_path,
                    request,
                    output_path=output_path,
                    prompt="FROZEN VISDRONE PROMPT",
                    backend_factory=self._factory(context_sha256),
                    offline_checker=lambda repo, revision, cache: snapshots[(repo, revision)],
                )

            self.assertTrue(frame["status"].eq("PASS").all())
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [row["model_id"] for row in report["models"]],
                [
                    "qwen3_vl_2b",
                    "smolvlm2_2b",
                    "qwen3_vl_8b_upper_bound",
                ],
            )
            for row in report["models"]:
                self.assertIs(type(row["input_tokens"]), int)
                self.assertIs(type(row["output_tokens"]), int)

    def test_invalid_token_counts_fail_closed_before_report_write(self) -> None:
        for invalid in (True, 1.5, float("inf"), float("nan")):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                lock_path, snapshots, request, context_sha256 = self._fixture(root)
                output_path = root / "MODEL_RUNTIME_SMOKE.json"
                with mock.patch.dict(
                    os.environ, model_loader_environment(lock_path), clear=False,
                ), self.assertRaisesRegex(
                    StudyGateError, "input_tokens must be a nonnegative integer",
                ):
                    run_one_image_runtime_smoke(
                        lock_path,
                        request,
                        output_path=output_path,
                        prompt="FROZEN VISDRONE PROMPT",
                        backend_factory=self._factory(
                            context_sha256, invalid_count=invalid,
                        ),
                        offline_checker=(
                            lambda repo, revision, cache: snapshots[(repo, revision)]
                        ),
                    )
                self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
