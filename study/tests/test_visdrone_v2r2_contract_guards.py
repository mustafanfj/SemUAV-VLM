from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from PIL import Image

from semuav.config import ConfigError, load_study_config
from semuav.perception import (
    VISDRONE_GENERATION_POLICY_VERSION,
    VISDRONE_GENERATION_STRATEGY_BY_MODEL,
    VISDRONE_MISSION_PROMPT,
    VISDRONE_MISSION_PROMPT_PATH,
    VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256,
    aspect_preserving_letterbox,
    visdrone_assistant_prefill_for_model,
    visdrone_generation_policy_sha256,
    visdrone_proposal_decision_context,
)
from semuav.registry import hash_object, sha256_bytes
from semuav.visdrone_models import (
    OFFICIAL_COMPACT_MODEL_REPOS,
    OFFICIAL_MODEL_BACKENDS,
    OFFICIAL_UPPER_BOUND_ID,
    OFFICIAL_UPPER_BOUND_REPO,
    VISDRONE_RUNTIME_SMOKE_INPUT_CONTRACT,
    freeze_huggingface_models,
    run_one_image_runtime_smoke,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


class V2R2PromptAndConfigurationGuardTests(unittest.TestCase):
    def setUp(self) -> None:
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

    def test_v2_prompt_contains_no_answer_object_or_concrete_proposal_ids(self) -> None:
        prompt = VISDRONE_MISSION_PROMPT_PATH.read_text(
            encoding="utf-8"
        ).strip()

        self.assertEqual(prompt, VISDRONE_MISSION_PROMPT)
        self.assertEqual(
            VISDRONE_MISSION_PROMPT_PATH.name,
            "visdrone_proposal_decision_v3.txt",
        )
        self.assertNotIn("{", prompt)
        self.assertNotIn("}", prompt)
        self.assertNotRegex(prompt, r"```")
        self.assertNotRegex(
            prompt,
            r'"?(?:sufficient_evidence|accepted_proposal_ids|'
            r'limited_visibility_proposal_ids)"?\s*:\s*'
            r'(?:true|false|\[|\{)',
        )
        self.assertNotRegex(
            prompt,
            r"\[\s*-?\d+(?:\s*,\s*-?\d+)*\s*\]",
        )
        self.assertNotRegex(prompt.casefold(), r"\b(?:example|e\.g\.)\b")

    def test_config_rejects_self_consistent_but_unregistered_policy_tamper(self) -> None:
        source = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "visdrone_paper.yaml").read_text(
                encoding="utf-8"
            )
        )
        policy = copy.deepcopy(source["models"]["generation_policy"])
        policy.pop("sha256")
        policy["generation"]["retry_count"] = 1
        source["models"]["generation_policy"] = {
            **policy,
            "sha256": hash_object(policy),
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered-policy.yaml"
            path.write_text(
                yaml.safe_dump(source, sort_keys=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ConfigError, "generation policy differs",
            ):
                load_study_config(path)

    def test_config_rejects_generation_policy_hash_tamper(self) -> None:
        source = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "visdrone_paper.yaml").read_text(
                encoding="utf-8"
            )
        )
        source["models"]["generation_policy"]["sha256"] = "0" * 64

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered-policy-hash.yaml"
            path.write_text(
                yaml.safe_dump(source, sort_keys=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ConfigError, "generation policy differs",
            ):
                load_study_config(path)


class V2R2RuntimeSmokeClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        loader_variables = {
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "TRANSFORMERS_CACHE",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "SEMUAV_MODELS_LOCAL_ONLY",
            "SEMUAV_QWEN_REVISION",
            "SEMUAV_SMOLVLM_REVISION",
            "SEMUAV_UPPER_BOUND_REVISION",
        }
        clean_environment = {
            name: value
            for name, value in os.environ.items()
            if name not in loader_variables
        }
        environment_patch = patch.dict(
            os.environ, clean_environment, clear=True,
        )
        environment_patch.start()
        self.addCleanup(environment_patch.stop)

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.lock_path, self.snapshots = self._freeze(self.root)
        self.image_path = self.root / "registered-train-canary.jpg"
        Image.new("RGB", (16, 12), (1, 2, 3)).save(self.image_path)
        self.request = SimpleNamespace(
            observation_id="visdrone:det_train:registered-smoke",
            image_path=str(self.image_path),
            mission="visdrone_urban",
            capture_time_us=123,
            telemetry={"battery_fraction": 0.9},
            detector_proposals=({
                "label": "car",
                "bbox_normalized_xyxy": [0.1, 0.2, 0.3, 0.4],
                "confidence": 0.8,
            },),
        )
        with Image.open(self.image_path) as opened_image:
            source_image = opened_image.convert("RGB")
        _, preprocess = aspect_preserving_letterbox(source_image, (768, 768))
        self.decision_context_sha256 = hash_object(
            visdrone_proposal_decision_context(
                self.request,
                source_image.size,
                (768, 768),
                preprocess=preprocess,
            )
        )

    @staticmethod
    def _freeze(root: Path):
        cache = root / "hf-cache"
        snapshots: dict[tuple[str, str], Path] = {}

        def resolver(repo, requested):
            if requested is not None:
                raise AssertionError("fixture expects an unresolved revision")
            return hashlib.sha1(repo.encode("utf-8")).hexdigest()

        def downloader(repo, revision, cache_root):
            snapshot = cache_root / "snapshots" / revision
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text(
                json.dumps({"repo": repo, "revision": revision}),
                encoding="utf-8",
            )
            (snapshot / "weights.bin").write_bytes(repo.encode("utf-8") * 2)
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

    @staticmethod
    def _valid_decision() -> str:
        return (
            '{"accepted_proposal_ids":[0],'
            '"limited_visibility_proposal_ids":[],'
            '"sufficient_evidence":true}'
        )

    def _run_case(self, case: str):
        test = self

        class FakeBackend:
            def __init__(self, entry):
                self.entry = entry

            def infer(self, request, prompt, max_new_tokens):
                model_id = self.entry["model_id"]
                target = model_id == "qwen3_vl_2b"
                raw_text = test._valid_decision()
                schema_valid = True
                decision_core_valid = True
                parsed = {"schema_version": "semuav.v1"}
                error = None
                if target and case == "malformed":
                    raw_text = '{"sufficient_evidence":true'
                    schema_valid = False
                    decision_core_valid = False
                    parsed = None
                    error = "ValueError: Model response is not one exact JSON object"
                elif target and case == "fenced":
                    raw_text = "{```json\n" + test._valid_decision() + "\n```"
                    schema_valid = False
                    decision_core_valid = False
                    parsed = None
                    error = "ValueError: Model response is not one exact JSON object"

                strategy = VISDRONE_GENERATION_STRATEGY_BY_MODEL[model_id]
                prefill = visdrone_assistant_prefill_for_model(model_id)
                if not raw_text.startswith(prefill):
                    raise AssertionError("fixture raw text must retain its host prefill")
                continuation = raw_text[len(prefill):]
                composer_sha256 = VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256
                generation_sha256 = visdrone_generation_policy_sha256(
                    max_new_tokens
                )
                if target and case == "composer_tamper":
                    composer_sha256 = "c" * 64
                if target and case == "policy_tamper":
                    generation_sha256 = "d" * 64

                return SimpleNamespace(
                    observation_id=request.observation_id,
                    model_id=model_id,
                    model_revision=self.entry["resolved_revision"],
                    schema_valid=schema_valid,
                    error=error,
                    latency_ms=5.0,
                    raw_text=raw_text,
                    parsed=parsed,
                    input_tokens=100,
                    output_tokens=20,
                    prompt_hash=hash_object(prompt),
                    decision_contract_version=(
                        VISDRONE_RUNTIME_SMOKE_INPUT_CONTRACT
                    ),
                    decision_core_valid=decision_core_valid,
                    decision_core_json=(raw_text if decision_core_valid else None),
                    decision_context_sha256=test.decision_context_sha256,
                    decision_proposal_count=1,
                    decision_composer_policy_sha256=composer_sha256,
                    terminated_by_eos=True,
                    reached_token_cap=False,
                    generation_policy_version=VISDRONE_GENERATION_POLICY_VERSION,
                    generation_policy_sha256=generation_sha256,
                    generation_strategy=strategy,
                    generation_max_new_tokens=max_new_tokens,
                    assistant_prefill_sha256=sha256_bytes(
                        prefill.encode("utf-8")
                    ),
                    assistant_prefill_characters=len(prefill),
                    generated_continuation_sha256=sha256_bytes(
                        continuation.encode("utf-8")
                    ),
                )

        return run_one_image_runtime_smoke(
            self.lock_path,
            self.request,
            output_path=self.root / f"SMOKE-{case}.json",
            prompt="FROZEN VISDRONE PROMPT",
            backend_factory=lambda entry, device, dtype: FakeBackend(entry),
            offline_checker=lambda repo, revision, cache: self.snapshots[
                (repo, revision)
            ],
        )

    def test_exact_identity_malformed_and_fenced_outputs_are_schema_failures(self):
        for case in ("malformed", "fenced"):
            with self.subTest(case=case):
                frame = self._run_case(case)
                row = frame.set_index("model_id").loc["qwen3_vl_2b"]
                self.assertEqual(row["status"], "FAIL_SCHEMA")
                self.assertEqual(
                    row["error"],
                    "ValueError: Model response is not one exact JSON object",
                )
                self.assertEqual(
                    sum(frame["status"].eq("PASS")),
                    2,
                )

    def test_policy_and_composer_tamper_are_provenance_failures(self):
        for case in ("policy_tamper", "composer_tamper"):
            with self.subTest(case=case):
                frame = self._run_case(case)
                row = frame.set_index("model_id").loc["qwen3_vl_2b"]
                self.assertEqual(row["status"], "FAIL_PROVENANCE")
                self.assertIn("lineage differs", row["error"])
                self.assertEqual(
                    sum(frame["status"].eq("PASS")),
                    2,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
