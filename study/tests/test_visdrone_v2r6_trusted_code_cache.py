from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from semuav.registry import StudyGateError, atomic_write_json, hash_object


# Historical bootstrap contract: v3 orchestration has separate tests.
ROOT = Path(__file__).resolve().parents[1] / "legacy" / "v2r13"
BUILDER = ROOT / "build_visdrone_paper_notebook.py"
NOTEBOOK_NAME = "SemUAV_VLM_VisDrone_Paper_Study.ipynb"
ACTIVE_MODEL_IDS = (
    "qwen3_vl_2b",
    "smolvlm2_2b",
    "qwen3_vl_8b_upper_bound",
)


def _trusted_cache_function_nodes() -> list[ast.FunctionDef]:
    """Extract the authored notebook helpers from a fresh builder output."""

    with tempfile.TemporaryDirectory() as directory:
        namespace = {
            "__file__": str(Path(directory) / BUILDER.name),
            "__name__": "trusted_cache_builder_test",
            "print": lambda *args, **kwargs: None,
        }
        source = BUILDER.read_text(encoding="utf-8")
        exec(compile(source, str(BUILDER), "exec"), namespace)
        notebook = json.loads(
            (Path(directory) / NOTEBOOK_NAME).read_text(encoding="utf-8")
        )

    cell_source = next(
        str(cell.get("source", ""))
        for cell in notebook["cells"]
        if "def _trusted_code_cache_payload" in str(cell.get("source", ""))
    )
    wanted = {
        "_read_regular_file_bytes",
        "_decode_json_object_bytes",
        "_trusted_code_cache_root",
        "_require_pristine_trusted_code_cache",
        "_trusted_code_cache_payload",
        "_capture_or_verify_trusted_code_cache",
    }
    nodes = [
        node
        for node in ast.parse(cell_source).body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    found = {node.name for node in nodes}
    if found != wanted:
        raise AssertionError(
            f"Trusted-code helper extraction mismatch: {sorted(found)}"
        )
    return nodes


TRUSTED_CACHE_FUNCTION_NODES = _trusted_cache_function_nodes()


class V2R9EmptyTrustedCodeCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache_root = self.root / "modules-v2r9"
        self.cache_root.mkdir()
        self.frozen_root = self.root / "frozen"
        self.frozen_root.mkdir()
        self.model_lock = self.frozen_root / "MODEL_LOCK.json"
        self.model_lock.write_text(
            '{"schema_version":"test"}\n', encoding="utf-8"
        )
        self.model_lock_derivation = (
            self.frozen_root / "MODEL_LOCK_DERIVATION.json"
        )
        self.model_lock_derivation.write_text(
            '{"schema_version":"test-derivation"}\n', encoding="utf-8"
        )
        self.receipt = self.frozen_root / "TRUSTED_CODE_CACHE.json"

        self.namespace = {
            "Path": Path,
            "os": os,
            "json": json,
            "StudyGateError": StudyGateError,
            "atomic_write_json": atomic_write_json,
            "hash_object": hash_object,
            "hash_file": self._hash_file,
            "cfg": SimpleNamespace(runtime={
                "required_environment_guards": {
                    "HF_MODULES_CACHE": str(self.cache_root),
                },
            }),
            "MODEL_LOCK": self.model_lock,
            "MODEL_LOCK_DERIVATION": self.model_lock_derivation,
            "TRUSTED_CODE_CACHE_RECEIPT": self.receipt,
            "REGISTERED_RUNTIME_SMOKE_MODEL_IDS": ACTIVE_MODEL_IDS,
            "_assert_safe_output_directory": self._safe_output_directory,
        }
        module = ast.Module(
            body=TRUSTED_CACHE_FUNCTION_NODES, type_ignores=[]
        )
        exec(
            compile(
                ast.fix_missing_locations(module),
                "trusted-cache-cell",
                "exec",
            ),
            self.namespace,
        )

        self.previous_hf_modules_cache = os.environ.get("HF_MODULES_CACHE")
        os.environ["HF_MODULES_CACHE"] = str(self.cache_root)
        self.addCleanup(self._restore_environment)

    def _restore_environment(self) -> None:
        if self.previous_hf_modules_cache is None:
            os.environ.pop("HF_MODULES_CACHE", None)
        else:
            os.environ["HF_MODULES_CACHE"] = self.previous_hf_modules_cache

    @staticmethod
    def _safe_output_directory(path, *, label, create):
        del label
        candidate = Path(path)
        if create:
            candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    @staticmethod
    def _hash_file(path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def test_pristine_gate_requires_empty_cache_and_absent_receipt(self) -> None:
        pristine = self.namespace["_require_pristine_trusted_code_cache"]()
        self.assertEqual(pristine, self.cache_root.resolve())

        (self.cache_root / "unexpected.py").write_text(
            "unexpected\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(StudyGateError, "not pristine"):
            self.namespace["_require_pristine_trusted_code_cache"]()

        (self.cache_root / "unexpected.py").unlink()
        self.receipt.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(StudyGateError, "already exists"):
            self.namespace["_require_pristine_trusted_code_cache"]()

    def test_payload_is_exact_empty_inventory_for_three_active_models(self) -> None:
        payload = self.namespace["_trusted_code_cache_payload"]()
        self.assertEqual(
            payload,
            {
                "schema_version": "semuav.visdrone-trusted-code-cache.v2",
                "scope": "no_active_trusted_remote_code",
                "scientific_results_eligible": False,
                "cache_root": str(self.cache_root.resolve()),
                "model_lock_path": str(self.model_lock.resolve()),
                "model_lock_sha256": self._hash_file(self.model_lock),
                "model_lock_derivation_path": str(
                    self.model_lock_derivation.resolve()
                ),
                "model_lock_derivation_sha256": self._hash_file(
                    self.model_lock_derivation
                ),
                "registered_model_ids": list(ACTIVE_MODEL_IDS),
                "excluded_model_ids": ["moondream2"],
                "module_roots": [],
                "file_count": 0,
                "files": [],
                "tree_sha256": hash_object([]),
            },
        )

    def test_any_remote_code_cache_material_fails_closed(self) -> None:
        nested = self.cache_root / "transformers_modules" / "unexpected"
        nested.mkdir(parents=True)
        (nested / "module.py").write_text("unsafe\n", encoding="utf-8")
        with self.assertRaisesRegex(
            StudyGateError, "unexpectedly populated",
        ):
            self.namespace["_trusted_code_cache_payload"]()

    def test_receipt_is_created_once_reverified_and_tamper_evident(self) -> None:
        captured = self.namespace["_capture_or_verify_trusted_code_cache"](
            allow_create=True,
        )
        self.assertTrue(self.receipt.is_file())
        self.assertEqual(captured["payload"]["file_count"], 0)

        verified = self.namespace["_capture_or_verify_trusted_code_cache"](
            allow_create=False,
        )
        self.assertEqual(verified, captured)

        changed = json.loads(self.receipt.read_text(encoding="utf-8"))
        changed["registered_model_ids"].append("moondream2")
        atomic_write_json(self.receipt, changed)
        with self.assertRaisesRegex(StudyGateError, "cache changed"):
            self.namespace["_capture_or_verify_trusted_code_cache"](
                allow_create=False,
            )

    def test_receipt_is_bound_to_exact_model_lock_derivation(self) -> None:
        self.namespace["_capture_or_verify_trusted_code_cache"](
            allow_create=True,
        )
        self.model_lock_derivation.write_text(
            '{"schema_version":"tampered"}\n', encoding="utf-8"
        )
        with self.assertRaisesRegex(StudyGateError, "cache changed"):
            self.namespace["_capture_or_verify_trusted_code_cache"](
                allow_create=False,
            )

    def test_missing_or_unsafe_receipt_fails_closed(self) -> None:
        with self.assertRaisesRegex(StudyGateError, "not frozen"):
            self.namespace["_capture_or_verify_trusted_code_cache"](
                allow_create=False,
            )

        target = self.frozen_root / "receipt-target.json"
        target.write_text("{}\n", encoding="utf-8")
        self.receipt.symlink_to(target)
        with self.assertRaisesRegex(StudyGateError, "receipt is unsafe"):
            self.namespace["_capture_or_verify_trusted_code_cache"](
                allow_create=False,
            )


if __name__ == "__main__":
    unittest.main()
