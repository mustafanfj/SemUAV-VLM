from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import semuav.visdrone_runner as runner
import semuav.visdrone_models as models
from semuav.perception import MoondreamBackend, ModelSpec, _moondream_load_kwargs
from semuav.registry import StudyGateError


REVISION = "6b714b26eea5cbd9f31e4edb2541c170afa935ba"
STARMIE_REVISION = models.MOONDREAM_TOKENIZER_REVISION


class V2R2MoondreamLoadRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        test_bytes = b'{"locked":"starmie-v1"}\n'
        bytes_patch = mock.patch.object(
            models, "MOONDREAM_TOKENIZER_BYTES", len(test_bytes)
        )
        sha_patch = mock.patch.object(
            models,
            "MOONDREAM_TOKENIZER_SHA256",
            hashlib.sha256(test_bytes).hexdigest(),
        )
        bytes_patch.start()
        sha_patch.start()
        self.addCleanup(bytes_patch.stop)
        self.addCleanup(sha_patch.stop)

    @staticmethod
    def _locked_tokenizer_fixture(
        root: Path,
        *,
        escape_symlink: bool = False,
    ) -> tuple[dict[str, object], Path, str]:
        cache = root / "huggingface"
        tokenizer_root = cache / "models--moondream--starmie-v1"
        snapshot = tokenizer_root / "snapshots" / STARMIE_REVISION
        blob = (
            root / "outside-tokenizer.json"
            if escape_symlink else tokenizer_root / "blobs" / ("a" * 64)
        )
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(b'{"locked":"starmie-v1"}\n')
        snapshot.mkdir(parents=True, exist_ok=True)
        tokenizer = snapshot / "tokenizer.json"
        tokenizer.symlink_to(blob)
        digest = hashlib.sha256(blob.read_bytes()).hexdigest()
        lock_path = root / "MOONDREAM_TOKENIZER_LOCK.json"
        if not escape_symlink:
            models.freeze_moondream_tokenizer_lock(
                cache_root=cache,
                lock_path=lock_path,
                downloader=lambda repo, revision, filename, cache_root: tokenizer,
            )
        entry: dict[str, object] = {
            "model_id": "moondream2",
            "repo": "vikhyatk/moondream2",
            "snapshot_path": str(
                cache / "models--vikhyatk--moondream2" / "snapshots" / REVISION
            ),
            "resolved_revision": REVISION,
            "role": "compact",
            "backend": "moondream2",
            "dtype": "bfloat16",
            "quantization": "none",
        }
        if not escape_symlink:
            payload = {"models": [entry]}
            bound = models.bind_moondream_tokenizer_lock(
                payload,
                lock_path,
                offline_checker=lambda repo, revision, filename, cache_root: tokenizer,
            )
            entry = bound["models"][0]
        else:
            entry["moondream_tokenizer"] = {
                "repo": models.MOONDREAM_TOKENIZER_REPO,
                "revision": STARMIE_REVISION,
                "filename": models.MOONDREAM_TOKENIZER_FILENAME,
                "path": str(tokenizer),
                "cache_root": str(cache.resolve()),
                "bytes": blob.stat().st_size,
                "sha256": digest,
                "lock_path": str(lock_path),
                "lock_sha256": "f" * 64,
            }
        return entry, tokenizer, digest

    @staticmethod
    def _construct_with_fake_modules(
        spec: ModelSpec,
        tokenizer_artifact: dict[str, object],
        *,
        requested_identifier: str = "moondream/starmie-v1",
        requested_args: tuple[object, ...] = (),
        requested_kwargs: dict[str, object] | None = None,
        fail_after_tokenizer: bool = False,
        state: dict[str, object] | None = None,
        model_cache: Path | None = None,
    ) -> tuple[object, type, types.ModuleType, list[str], list[tuple]]:
        from_file_calls: list[str] = []
        from_pretrained_calls: list[tuple] = []
        tokenizer_result = object()

        class Tokenizer:
            @classmethod
            def from_file(cls, path: str):
                del cls
                from_file_calls.append(path)
                return tokenizer_result

            @classmethod
            def from_pretrained(cls, identifier: str, *args, **kwargs):
                del cls
                from_pretrained_calls.append((identifier, args, kwargs))
                return object()

        implementation_name = "transformers_modules.test.moondream"
        facade_name = "transformers_modules.test.hf_moondream"
        implementation_module = types.ModuleType(implementation_name)
        implementation_module.Tokenizer = Tokenizer
        facade_module = types.ModuleType(facade_name)
        MoondreamModel = type("MoondreamModel", (), {})
        MoondreamModel.__module__ = implementation_name
        HfMoondream = type("HfMoondream", (), {})
        HfMoondream.__module__ = facade_name
        HfConfig = type("HfConfig", (), {})
        HfConfig.__module__ = facade_name
        facade_module.MoondreamModel = MoondreamModel
        facade_module.HfMoondream = HfMoondream
        facade_module.HfConfig = HfConfig
        if state is not None:
            state.update({
                "original_tokenizer": Tokenizer,
                "implementation_module": implementation_module,
            })

        class Loaded:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer

            def to(self, device: str) -> "Loaded":
                self.device = device
                return self

            def eval(self) -> "Loaded":
                return self

        class AutoConfig:
            @classmethod
            def from_pretrained(cls, repo: str, **kwargs):
                del cls, repo, kwargs
                config = HfConfig()
                config.auto_map = {
                    "AutoModelForCausalLM": "hf_moondream.HfMoondream",
                }
                return config

        class AutoModelForCausalLM:
            @classmethod
            def from_pretrained(cls, repo: str, **kwargs):
                del cls, repo, kwargs
                routed = implementation_module.Tokenizer.from_pretrained(
                    requested_identifier,
                    *requested_args,
                    **dict(requested_kwargs or {}),
                )
                if fail_after_tokenizer:
                    raise RuntimeError("deliberate downstream construction failure")
                return Loaded(routed)

        dynamic_module_utils = types.ModuleType(
            "transformers.dynamic_module_utils"
        )
        dynamic_module_utils.get_class_from_dynamic_module = (
            lambda *args, **kwargs: HfMoondream
        )
        transformers = types.ModuleType("transformers")
        transformers.AutoConfig = AutoConfig
        transformers.AutoModelForCausalLM = AutoModelForCausalLM
        transformers.dynamic_module_utils = dynamic_module_utils
        tokenizers = types.ModuleType("tokenizers")
        tokenizers.Tokenizer = Tokenizer
        torch = types.ModuleType("torch")
        torch.bfloat16 = object()

        cache = model_cache or Path(str(tokenizer_artifact["cache_root"]))
        environment = {
            "SEMUAV_MODELS_LOCAL_ONLY": "1",
            "HF_HUB_CACHE": str(cache),
            "HUGGINGFACE_HUB_CACHE": str(cache),
            "TRANSFORMERS_CACHE": str(cache),
        }
        modules = {
            "torch": torch,
            "tokenizers": tokenizers,
            "transformers": transformers,
            "transformers.dynamic_module_utils": dynamic_module_utils,
            facade_name: facade_module,
            implementation_name: implementation_module,
        }
        with mock.patch.dict(sys.modules, modules), mock.patch.dict(
            os.environ, environment, clear=True,
        ):
            backend = MoondreamBackend(
                spec,
                device="cpu",
                tokenizer_artifact=tokenizer_artifact,
            )
        return (
            backend.model,
            Tokenizer,
            implementation_module,
            from_file_calls,
            from_pretrained_calls,
        )

    def test_local_only_moondream_rejects_split_or_missing_cache_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary).resolve()
            coherent = {
                "SEMUAV_MODELS_LOCAL_ONLY": "1",
                "HF_HUB_CACHE": str(cache),
                "HUGGINGFACE_HUB_CACHE": str(cache),
                "TRANSFORMERS_CACHE": str(cache),
            }
            cases = (
                {key: value for key, value in coherent.items() if key != "HUGGINGFACE_HUB_CACHE"},
                {**coherent, "HUGGINGFACE_HUB_CACHE": str(cache / "hub")},
                {**coherent, "TRANSFORMERS_CACHE": str(cache / "transformers")},
            )
            for environment in cases:
                with self.subTest(environment=environment), mock.patch.dict(
                    os.environ, environment, clear=True,
                ), self.assertRaisesRegex(
                    StudyGateError, "must equal HF_HUB_CACHE",
                ):
                    _moondream_load_kwargs()

    def test_locked_tokenizer_artifact_accepts_in_cache_snapshot_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            entry, tokenizer, digest = self._locked_tokenizer_fixture(
                Path(temporary)
            )

            actual = models.registered_moondream_tokenizer_artifact(entry)

        self.assertEqual(actual["repo"], "moondream/starmie-v1")
        self.assertEqual(actual["revision"], STARMIE_REVISION)
        self.assertEqual(actual["path"], str(tokenizer))
        self.assertEqual(actual["sha256"], digest)

    def test_locked_tokenizer_artifact_rejects_missing_unsafe_or_changed_bytes(self) -> None:
        cases = ("missing", "hash_mismatch", "unrecorded")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                entry, tokenizer, _ = self._locked_tokenizer_fixture(
                    Path(temporary),
                )
                if case == "missing":
                    tokenizer.unlink()
                elif case == "hash_mismatch":
                    entry["moondream_tokenizer"]["sha256"] = "0" * 64
                elif case == "unrecorded":
                    del entry["moondream_tokenizer"]

                with self.assertRaisesRegex(
                    StudyGateError,
                    "tokenizer|Tokenizer|artifact|bytes|hash|unsafe|missing|escape",
                ):
                    models.registered_moondream_tokenizer_artifact(entry)

    def test_local_constructor_rejects_missing_unsafe_or_changed_tokenizer(self) -> None:
        for case in ("missing", "escape", "hash_mismatch"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                entry, tokenizer, _ = self._locked_tokenizer_fixture(
                    Path(temporary), escape_symlink=case == "escape",
                )
                artifact = dict(entry["moondream_tokenizer"])
                if case == "missing":
                    tokenizer.unlink()
                elif case == "hash_mismatch":
                    artifact["sha256"] = "0" * 64
                spec = ModelSpec(
                    "moondream2",
                    "vikhyatk/moondream2",
                    REVISION,
                    trust_remote_code=True,
                )

                with self.assertRaisesRegex(
                    StudyGateError,
                    "tokenizer|Tokenizer|bytes|hash|unsafe|missing|escape",
                ):
                    self._construct_with_fake_modules(spec, artifact)

    def test_local_construction_routes_only_starmie_to_locked_tokenizer_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            entry, tokenizer, _ = self._locked_tokenizer_fixture(
                Path(temporary)
            )
            spec = ModelSpec(
                "moondream2",
                "vikhyatk/moondream2",
                REVISION,
                trust_remote_code=True,
            )

            _, original, implementation, from_file, delegated = (
                self._construct_with_fake_modules(
                    spec, dict(entry["moondream_tokenizer"]),
                )
            )

        self.assertEqual(from_file, [str(tokenizer)])
        self.assertEqual(delegated, [])
        self.assertIs(implementation.Tokenizer, original)

    def test_local_construction_allows_separate_model_and_starmie_caches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry, tokenizer, _ = self._locked_tokenizer_fixture(root)
            model_cache = root / "model-cache"
            model_cache.mkdir()
            spec = ModelSpec(
                "moondream2",
                "vikhyatk/moondream2",
                REVISION,
                trust_remote_code=True,
            )

            _, _, _, from_file, _ = self._construct_with_fake_modules(
                spec,
                dict(entry["moondream_tokenizer"]),
                model_cache=model_cache,
            )

        self.assertEqual(from_file, [str(tokenizer)])

    def test_local_construction_fails_closed_for_other_tokenizer_requests(self) -> None:
        cases = (
            ("other/model", (), {}),
            ("moondream/starmie-v1", (), {"revision": "main"}),
            ("moondream/starmie-v1", ("unexpected",), {}),
        )
        for identifier, args, kwargs in cases:
            with self.subTest(
                identifier=identifier, args=args, kwargs=kwargs,
            ), tempfile.TemporaryDirectory() as temporary:
                entry, _, _ = self._locked_tokenizer_fixture(
                    Path(temporary)
                )
                spec = ModelSpec(
                    "moondream2",
                    "vikhyatk/moondream2",
                    REVISION,
                    trust_remote_code=True,
                )
                with self.assertRaisesRegex(
                    StudyGateError, "tokenizer|Tokenizer|starmie",
                ):
                    self._construct_with_fake_modules(
                        spec,
                        dict(entry["moondream_tokenizer"]),
                        requested_identifier=identifier,
                        requested_args=args,
                        requested_kwargs=kwargs,
                    )

    def test_tokenizer_route_restores_module_global_after_loader_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            entry, _, _ = self._locked_tokenizer_fixture(Path(temporary))
            state: dict[str, object] = {}
            spec = ModelSpec(
                "moondream2",
                "vikhyatk/moondream2",
                REVISION,
                trust_remote_code=True,
            )
            with self.assertRaisesRegex(
                RuntimeError, "downstream construction failure",
            ):
                self._construct_with_fake_modules(
                    spec,
                    dict(entry["moondream_tokenizer"]),
                    fail_after_tokenizer=True,
                    state=state,
                )
            self.assertIs(
                state["implementation_module"].Tokenizer,
                state["original_tokenizer"],
            )

    def test_v2r9_scientific_factory_rejects_historical_moondream_backend(self) -> None:
        entry = {
            "model_id": "moondream2",
            "repo": "vikhyatk/moondream2",
            "snapshot_path": "/locked/cache/models--vikhyatk--moondream2/snapshots/" + REVISION,
            "resolved_revision": REVISION,
            "role": "compact",
            "backend": "moondream2",
            "dtype": "bfloat16",
            "quantization": "none",
        }
        with self.assertRaisesRegex(
            StudyGateError, "unsupported backend",
        ):
            runner._model_backend(entry, "cuda:0", (768, 640))

    def test_scientific_factory_preserves_snapshot_for_non_moondream(self) -> None:
        snapshot = "/locked/cache/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/" + REVISION
        entry = {
            "model_id": "qwen3_vl_2b",
            "repo": "Qwen/Qwen3-VL-2B-Instruct",
            "snapshot_path": snapshot,
            "resolved_revision": REVISION,
            "role": "compact",
            "backend": "qwen3_vl",
            "dtype": "bfloat16",
            "quantization": "none",
        }
        backend = object()
        with mock.patch.object(
            runner, "HFMultimodalBackend", return_value=backend,
        ) as hf_backend:
            actual = runner._model_backend(entry, "cuda:0", (512, 512))

        self.assertIs(actual, backend)
        hf_backend.assert_called_once()
        spec, device, dtype = hf_backend.call_args.args
        self.assertEqual(device, "cuda:0")
        self.assertEqual(dtype, "bfloat16")
        self.assertEqual(spec.model_id, "qwen3_vl_2b")
        self.assertEqual(spec.repo, snapshot)
        self.assertEqual(spec.revision, REVISION)
        self.assertFalse(spec.trust_remote_code)
        self.assertEqual(spec.image_size, (512, 512))

    def test_v2r9_runtime_smoke_factory_rejects_historical_moondream(self) -> None:
        entry = {
            "model_id": "moondream2",
            "repo": "vikhyatk/moondream2",
            "snapshot_path": "/locked/cache/models--vikhyatk--moondream2/snapshots/" + REVISION,
            "resolved_revision": REVISION,
            "role": "compact",
            "backend": "moondream2",
            "dtype": "bfloat16",
            "quantization": "none",
        }
        with self.assertRaisesRegex(
            StudyGateError, "Inactive or unsupported v2r13 model backend",
        ):
            models._default_backend_factory(
                entry, device="cuda:0", dtype="bfloat16",
            )


if __name__ == "__main__":
    unittest.main()
