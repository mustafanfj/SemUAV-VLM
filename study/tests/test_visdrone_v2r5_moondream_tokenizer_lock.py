from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import semuav.visdrone_models as models
from semuav.registry import StudyGateError
from semuav.visdrone_models import (
    MOONDREAM_TOKENIZER_AUDITED_REPO_HEAD,
    MOONDREAM_TOKENIZER_FILENAME,
    MOONDREAM_TOKENIZER_BYTES,
    MOONDREAM_TOKENIZER_SHA256,
    MOONDREAM_TOKENIZER_LOCK_SCHEMA,
    MOONDREAM_TOKENIZER_REPO,
    MOONDREAM_TOKENIZER_REVISION,
    bind_moondream_tokenizer_lock,
    freeze_moondream_tokenizer_lock,
    registered_moondream_tokenizer_artifact,
    verify_moondream_tokenizer_lock,
)


class V2R5MoondreamTokenizerLockTests(unittest.TestCase):
    _TEST_TOKENIZER_BYTES = b'{"registered":"moondream/starmie-v1"}\n'

    def setUp(self) -> None:
        expected_sha256 = hashlib.sha256(self._TEST_TOKENIZER_BYTES).hexdigest()
        bytes_patch = mock.patch.object(
            models, "MOONDREAM_TOKENIZER_BYTES", len(self._TEST_TOKENIZER_BYTES)
        )
        sha_patch = mock.patch.object(
            models, "MOONDREAM_TOKENIZER_SHA256", expected_sha256
        )
        bytes_patch.start()
        sha_patch.start()
        self.addCleanup(bytes_patch.stop)
        self.addCleanup(sha_patch.stop)

    @staticmethod
    def _cached_starmie(
        root: Path,
        *,
        bytes_: bytes = _TEST_TOKENIZER_BYTES,
        escape: bool = False,
    ) -> tuple[Path, Path]:
        cache = root / "huggingface"
        repo = cache / "models--moondream--starmie-v1"
        snapshot = repo / "snapshots" / MOONDREAM_TOKENIZER_REVISION
        blob = (
            root / "outside" / "tokenizer.json"
            if escape
            else repo / "blobs" / ("a" * 64)
        )
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(bytes_)
        snapshot.mkdir(parents=True, exist_ok=True)
        tokenizer = snapshot / MOONDREAM_TOKENIZER_FILENAME
        tokenizer.symlink_to(blob)
        return cache, tokenizer

    @staticmethod
    def _model_lock() -> dict[str, object]:
        return {
            "schema_version": 1,
            "models": [
                {
                    "model_id": "qwen3_vl_2b",
                    "backend": "qwen3_vl",
                },
                {
                    "model_id": "moondream2",
                    "repo": "vikhyatk/moondream2",
                    "snapshot_path": "/cache/models--vikhyatk--moondream2/snapshots/"
                    "6b714b26eea5cbd9f31e4edb2541c170afa935ba",
                    "resolved_revision":
                    "6b714b26eea5cbd9f31e4edb2541c170afa935ba",
                    "role": "compact",
                    "backend": "moondream2",
                    "dtype": "bfloat16",
                    "quantization": "none",
                },
            ],
        }

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    def test_official_repository_and_immutable_revisions_are_exact(self) -> None:
        self.assertEqual(MOONDREAM_TOKENIZER_REPO, "moondream/starmie-v1")
        self.assertEqual(
            MOONDREAM_TOKENIZER_REVISION,
            "360559108ac4586b96328de81f817859a72fd5ca",
        )
        self.assertEqual(
            MOONDREAM_TOKENIZER_AUDITED_REPO_HEAD,
            "35192e10a54e36eabe0a7cc57a2c1aab371cafc5",
        )
        self.assertEqual(MOONDREAM_TOKENIZER_FILENAME, "tokenizer.json")
        self.assertEqual(models.MOONDREAM_TOKENIZER_BYTES, len(self._TEST_TOKENIZER_BYTES))
        self.assertEqual(MOONDREAM_TOKENIZER_BYTES, 3_694_206)
        self.assertEqual(
            MOONDREAM_TOKENIZER_SHA256,
            "0512fdcac4a5f9e7746cbefce4a468dc93bf0f93f11e701b84f8b31bff199e9c",
        )

    def test_freeze_verify_and_bind_preserve_separate_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, tokenizer = self._cached_starmie(root)
            lock_path = root / "frozen_inputs" / "MOONDREAM_TOKENIZER_LOCK.json"
            download_calls: list[tuple[object, ...]] = []
            offline_calls: list[tuple[object, ...]] = []

            def downloader(repo: str, revision: str, filename: str, root_: Path) -> Path:
                download_calls.append((repo, revision, filename, root_))
                return tokenizer

            def checker(repo: str, revision: str, filename: str, root_: Path) -> Path:
                offline_calls.append((repo, revision, filename, root_))
                return tokenizer

            payload = freeze_moondream_tokenizer_lock(
                cache_root=cache,
                lock_path=lock_path,
                downloader=downloader,
            )
            verified = verify_moondream_tokenizer_lock(
                lock_path,
                offline_checker=checker,
            )
            original = self._model_lock()
            before = copy.deepcopy(original)
            bound = bind_moondream_tokenizer_lock(
                original,
                lock_path,
                offline_checker=checker,
            )

            self.assertEqual(original, before)
            self.assertEqual(payload["schema_version"], MOONDREAM_TOKENIZER_LOCK_SCHEMA)
            self.assertEqual(payload["requested_revision"], MOONDREAM_TOKENIZER_REVISION)
            self.assertEqual(payload["resolved_revision"], MOONDREAM_TOKENIZER_REVISION)
            self.assertEqual(payload["file_count"], 1)
            artifact = payload["artifacts"][0]
            self.assertEqual(artifact["snapshot_path"], str(tokenizer))
            self.assertEqual(artifact["bytes"], tokenizer.stat().st_size)
            self.assertEqual(
                artifact["sha256"],
                hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
            )
            self.assertEqual(verified["offline_verification"], "OFFLINE_VERIFIED")
            self.assertEqual(
                download_calls,
                [(
                    MOONDREAM_TOKENIZER_REPO,
                    MOONDREAM_TOKENIZER_REVISION,
                    MOONDREAM_TOKENIZER_FILENAME,
                    cache.resolve(),
                )],
            )
            self.assertEqual(offline_calls, download_calls * 2)

            moon_rows = [
                row for row in bound["models"]
                if row["model_id"] == "moondream2"
            ]
            self.assertEqual(len(moon_rows), 1)
            record = registered_moondream_tokenizer_artifact(moon_rows[0])
            self.assertEqual(record["repo"], MOONDREAM_TOKENIZER_REPO)
            self.assertEqual(record["revision"], MOONDREAM_TOKENIZER_REVISION)
            self.assertEqual(record["path"], str(tokenizer))
            self.assertEqual(record["lock_path"], str(lock_path.resolve()))
            self.assertEqual(
                bound["moondream_tokenizer_lock"]["artifact_sha256"],
                artifact["sha256"],
            )

    def test_freeze_refuses_mutable_or_wrong_download_location(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, tokenizer = self._cached_starmie(root)
            wrong = (
                cache
                / "models--vikhyatk--moondream2"
                / "snapshots"
                / "6b714b26eea5cbd9f31e4edb2541c170afa935ba"
                / "tokenizer.json"
            )
            wrong.parent.mkdir(parents=True)
            wrong.write_bytes(tokenizer.read_bytes())
            lock_path = root / "MOONDREAM_TOKENIZER_LOCK.json"

            with self.assertRaisesRegex(StudyGateError, "exact pinned Starmie"):
                freeze_moondream_tokenizer_lock(
                    cache_root=cache,
                    lock_path=lock_path,
                    downloader=lambda *args: wrong,
                )

            self.assertFalse(lock_path.exists())

    def test_freeze_rejects_snapshot_symlink_that_escapes_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, tokenizer = self._cached_starmie(root, escape=True)

            with self.assertRaisesRegex(StudyGateError, "outside.*cache|escapes"):
                freeze_moondream_tokenizer_lock(
                    cache_root=cache,
                    lock_path=root / "MOONDREAM_TOKENIZER_LOCK.json",
                    downloader=lambda *args: tokenizer,
                )

    def test_freeze_rejects_cross_repository_blob_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "huggingface"
            tokenizer = (
                cache
                / "models--moondream--starmie-v1"
                / "snapshots"
                / MOONDREAM_TOKENIZER_REVISION
                / MOONDREAM_TOKENIZER_FILENAME
            )
            other_blob = cache / "models--attacker--other" / "blobs" / ("a" * 64)
            other_blob.parent.mkdir(parents=True)
            other_blob.write_bytes(b'{"substitution":true}\n')
            tokenizer.parent.mkdir(parents=True)
            tokenizer.symlink_to(other_blob)

            with self.assertRaisesRegex(StudyGateError, "own repository blob"):
                freeze_moondream_tokenizer_lock(
                    cache_root=cache,
                    lock_path=root / "MOONDREAM_TOKENIZER_LOCK.json",
                    downloader=lambda *args: tokenizer,
                )

    def test_freeze_rejects_symlinked_lock_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, tokenizer = self._cached_starmie(root)
            real_receipts = root / "real-receipts"
            real_receipts.mkdir()
            alias = root / "receipt-alias"
            alias.symlink_to(real_receipts, target_is_directory=True)

            with self.assertRaisesRegex(StudyGateError, "symlinked parent"):
                freeze_moondream_tokenizer_lock(
                    cache_root=cache,
                    lock_path=alias / "MOONDREAM_TOKENIZER_LOCK.json",
                    downloader=lambda *args: tokenizer,
                )

    def test_lock_is_create_once_unless_overwrite_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, tokenizer = self._cached_starmie(root)
            lock_path = root / "MOONDREAM_TOKENIZER_LOCK.json"
            freeze_moondream_tokenizer_lock(
                cache_root=cache,
                lock_path=lock_path,
                downloader=lambda *args: tokenizer,
            )

            with self.assertRaisesRegex(StudyGateError, "already exists"):
                freeze_moondream_tokenizer_lock(
                    cache_root=cache,
                    lock_path=lock_path,
                    downloader=lambda *args: tokenizer,
                )

    def test_verify_rejects_identity_inventory_and_byte_tampering(self) -> None:
        cases = ("repo", "revision", "inventory", "file_count_type", "bytes")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                cache, tokenizer = self._cached_starmie(root)
                lock_path = root / "MOONDREAM_TOKENIZER_LOCK.json"
                freeze_moondream_tokenizer_lock(
                    cache_root=cache,
                    lock_path=lock_path,
                    downloader=lambda *args: tokenizer,
                )
                if case == "bytes":
                    tokenizer.resolve().write_bytes(b"changed after freeze\n")
                else:
                    payload = json.loads(lock_path.read_text(encoding="utf-8"))
                    if case == "repo":
                        payload["repo"] = "vikhyatk/moondream2"
                    elif case == "revision":
                        payload["resolved_revision"] = "0" * 40
                    elif case == "inventory":
                        payload["inventory_sha256"] = "0" * 64
                    elif case == "file_count_type":
                        payload["file_count"] = True
                    self._write_json(lock_path, payload)

                with self.assertRaisesRegex(
                    StudyGateError,
                    "identity|inventory|byte|path|changed",
                ):
                    verify_moondream_tokenizer_lock(
                        lock_path,
                        offline_checker=lambda *args: tokenizer,
                    )

    def test_verify_rejects_duplicate_json_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, tokenizer = self._cached_starmie(root)
            lock_path = root / "MOONDREAM_TOKENIZER_LOCK.json"
            freeze_moondream_tokenizer_lock(
                cache_root=cache,
                lock_path=lock_path,
                downloader=lambda *args: tokenizer,
            )
            raw = lock_path.read_text(encoding="utf-8")
            duplicate = raw.replace(
                '"schema_version":',
                '"schema_version":"ambiguous","schema_version":',
                1,
            )
            lock_path.write_text(duplicate, encoding="utf-8")

            with self.assertRaisesRegex(StudyGateError, "duplicate JSON key"):
                verify_moondream_tokenizer_lock(
                    lock_path,
                    offline_checker=lambda *args: tokenizer,
                )

    def test_verify_rejects_wrong_offline_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, tokenizer = self._cached_starmie(root)
            lock_path = root / "MOONDREAM_TOKENIZER_LOCK.json"
            freeze_moondream_tokenizer_lock(
                cache_root=cache,
                lock_path=lock_path,
                downloader=lambda *args: tokenizer,
            )
            other = cache / "other-tokenizer.json"
            other.write_bytes(tokenizer.read_bytes())

            with self.assertRaisesRegex(StudyGateError, "different tokenizer"):
                verify_moondream_tokenizer_lock(
                    lock_path,
                    offline_checker=lambda *args: other,
                )

    def test_bound_record_detects_later_lock_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, tokenizer = self._cached_starmie(root)
            lock_path = root / "MOONDREAM_TOKENIZER_LOCK.json"
            freeze_moondream_tokenizer_lock(
                cache_root=cache,
                lock_path=lock_path,
                downloader=lambda *args: tokenizer,
            )
            bound = bind_moondream_tokenizer_lock(
                self._model_lock(),
                lock_path,
                offline_checker=lambda *args: tokenizer,
            )
            moon = next(
                row for row in bound["models"] if row["model_id"] == "moondream2"
            )
            lock_path.write_bytes(lock_path.read_bytes() + b"\n")

            with self.assertRaisesRegex(StudyGateError, "lock|binding"):
                registered_moondream_tokenizer_artifact(moon)

    def test_bound_record_rejects_semantically_forged_lock_with_matching_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, tokenizer = self._cached_starmie(root)
            lock_path = root / "MOONDREAM_TOKENIZER_LOCK.json"
            freeze_moondream_tokenizer_lock(
                cache_root=cache,
                lock_path=lock_path,
                downloader=lambda *args: tokenizer,
            )
            bound = bind_moondream_tokenizer_lock(
                self._model_lock(),
                lock_path,
                offline_checker=lambda *args: tokenizer,
            )
            moon = next(
                row for row in bound["models"] if row["model_id"] == "moondream2"
            )
            lock_path.write_bytes(b'{"not":"a tokenizer lock"}\n')
            moon["moondream_tokenizer"]["lock_sha256"] = hashlib.sha256(
                lock_path.read_bytes()
            ).hexdigest()

            with self.assertRaisesRegex(StudyGateError, "envelope"):
                registered_moondream_tokenizer_artifact(moon)


if __name__ == "__main__":
    unittest.main()
