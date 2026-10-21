from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from semuav.perception import (
    DeterministicMockBackend,
    InferenceRequest,
    run_inference_cache_sharded,
)
from semuav.registry import RunContext, StudyGateError, hash_object
from semuav import visdrone_recovery as recovery


class V2R9InferenceRecoveryTests(unittest.TestCase):
    MODEL_ID = "qwen3_vl_2b"
    MODEL_REVISION = "1" * 40
    CURRENT_MODEL_LOCK = "a" * 64
    CONFIG_HASH = recovery.PREDECESSOR_CONFIG_PREFIX + "0" * 54
    PROMPT = "registered recovery prompt"
    STAGE_NAME = "01_vlm_qwen3_vl_2b_shard_00000"

    @staticmethod
    def _current_context(root: Path) -> RunContext:
        run_root = root / "current"
        for name in ("stages", "blobs", "checkpoints", "logs", "results", "plots"):
            (run_root / name).mkdir(parents=True, exist_ok=True)
        ctx = RunContext(
            project_root=root,
            run_root=run_root,
            profile="full",
            run_name="semuav_visdrone_v2r13",
            config={"study": "v2r13-test"},
            config_hash="b" * 64,
            code_hash="c" * 64,
            created_utc="2026-09-21T00:00:00Z",
            resume=True,
        )
        ctx._initialize()
        return ctx

    @classmethod
    def _key_material(cls) -> dict[str, object]:
        return {
            "model": cls.MODEL_ID,
            "repo": "registered/repo",
            "revision": cls.MODEL_REVISION,
            "quantization": "none",
            "image_size": [768, 768],
            "prompt": hash_object(cls.PROMPT),
            "max_new_tokens": 384,
            "execution_context": {
                "detector_revision": "detector-revision",
                "model_lock_sha256": cls.CURRENT_MODEL_LOCK,
                "sample_role": "general",
            },
            "requests": [
                {"observation_id": "obs-a"},
                {"observation_id": "obs-b"},
            ],
        }

    @classmethod
    def _frame(cls) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "observation_id": observation_id,
                "model_id": cls.MODEL_ID,
                "model_revision": cls.MODEL_REVISION,
                "prompt_hash": hash_object(cls.PROMPT),
                "parsed_json": "{}",
                "schema_valid": True,
                "raw_text": "{}",
                "latency_ms": latency,
            }
            for observation_id, latency in (("obs-a", 1.25), ("obs-b", 2.5))
        ])

    def _fixture(
        self,
        root: Path,
        *,
        record_rows: int = 2,
    ) -> dict[str, object]:
        ctx = self._current_context(root)
        key = self._key_material()
        predecessor_key_material = copy.deepcopy(key)
        predecessor_key_material["execution_context"][
            "model_lock_sha256"
        ] = recovery.PREDECESSOR_MODEL_LOCK_SHA256
        predecessor_key = hash_object({
            "name": self.STAGE_NAME,
            "key": predecessor_key_material,
            "config": self.CONFIG_HASH,
            "code": recovery.PREDECESSOR_CODE_SHA256,
        })
        run_root = root / (
            f"{recovery.PREDECESSOR_RUN_NAME}-full-"
            f"{self.CONFIG_HASH[:10]}-{recovery.PREDECESSOR_CODE_SHA256[:10]}"
        )
        source = (
            run_root / "stages" / self.STAGE_NAME / f"{predecessor_key}.csv.gz"
        )
        source.parent.mkdir(parents=True)
        self._frame().to_csv(
            source,
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        payload = source.read_bytes()
        relative = source.relative_to(run_root).as_posix()
        artifacts = [{
            "relative_path": relative,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": record_rows,
        }]
        run_manifest = run_root / "run_manifest.json"
        run_manifest.write_text(json.dumps({
            "run_name": recovery.PREDECESSOR_RUN_NAME,
            "profile": "full",
            "config_hash": self.CONFIG_HASH,
            "code_hash": recovery.PREDECESSOR_CODE_SHA256,
        }, sort_keys=True), encoding="utf-8")
        manifest_payload = run_manifest.read_bytes()
        receipt = {
            "schema_version": recovery.RECOVERY_SCHEMA,
            "status": recovery.RECOVERY_STATUS,
            "predecessor_release_id": recovery.PREDECESSOR_RELEASE_ID,
            "predecessor_run_name": recovery.PREDECESSOR_RUN_NAME,
            "predecessor_code_sha256": recovery.PREDECESSOR_CODE_SHA256,
            "predecessor_config_sha256": self.CONFIG_HASH,
            "predecessor_model_lock_sha256": (
                recovery.PREDECESSOR_MODEL_LOCK_SHA256
            ),
            "predecessor_run_root": str(run_root),
            "untouched_test_accessed": False,
            "protocol_frozen": False,
            "native_cache_completed": False,
            "artifact_count": 1,
            "inference_row_count": record_rows,
            "artifact_inventory_sha256": hash_object(artifacts),
            "artifacts": artifacts,
            "run_manifest": {
                "path": str(run_manifest),
                "bytes": len(manifest_payload),
                "sha256": hashlib.sha256(manifest_payload).hexdigest(),
            },
        }
        receipt_path = root / "V2R9_EXP1_RECOVERY_SOURCE.json"
        receipt_path.write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        receipt_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        return {
            "ctx": ctx,
            "key": key,
            "source": source,
            "receipt_path": receipt_path,
            "environment": {
                "SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT": str(receipt_path),
                "SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT_SHA256": receipt_sha256,
            },
            "requests": [
                SimpleNamespace(observation_id="obs-a"),
                SimpleNamespace(observation_id="obs-b"),
            ],
        }

    def _import(self, fixture: dict[str, object]) -> bool:
        return recovery.import_v2r9_inference_stage(
            fixture["ctx"],
            self.STAGE_NAME,
            fixture["key"],
            requests=fixture["requests"],
            model_id=self.MODEL_ID,
            model_revision=self.MODEL_REVISION,
            prompt=self.PROMPT,
        )

    def test_verified_shard_is_byte_copied_and_then_resumes_without_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            with (
                mock.patch.object(recovery, "EXPECTED_ARTIFACT_COUNT", 1),
                mock.patch.object(recovery, "EXPECTED_INFERENCE_ROWS", 2),
                mock.patch.dict(os.environ, fixture["environment"], clear=False),
            ):
                self.assertTrue(self._import(fixture))
            record = fixture["ctx"].stage_record(self.STAGE_NAME)
            self.assertEqual(record["status"], "COMPLETE")
            self.assertEqual(record["rows"], 2)
            self.assertTrue(record["message"].startswith(
                "verified_v2r9_development_inference_reuse:"
            ))
            self.assertEqual(
                Path(record["artifact"]).read_bytes(),
                fixture["source"].read_bytes(),
            )
            # A completed current-key stage is a true resume no-op and does not
            # need to reopen the predecessor receipt.
            with mock.patch.dict(os.environ, {
                "SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT": "",
                "SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT_SHA256": "",
            }, clear=False):
                self.assertFalse(self._import(fixture))

    def test_changed_predecessor_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            fixture["source"].write_bytes(fixture["source"].read_bytes() + b"changed")
            with (
                mock.patch.object(recovery, "EXPECTED_ARTIFACT_COUNT", 1),
                mock.patch.object(recovery, "EXPECTED_INFERENCE_ROWS", 2),
                mock.patch.dict(os.environ, fixture["environment"], clear=False),
                self.assertRaisesRegex(StudyGateError, "shard changed"),
            ):
                self._import(fixture)

    def test_unregistered_predecessor_cache_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            fixture["key"]["max_new_tokens"] = 385
            with (
                mock.patch.object(recovery, "EXPECTED_ARTIFACT_COUNT", 1),
                mock.patch.object(recovery, "EXPECTED_INFERENCE_ROWS", 2),
                mock.patch.dict(os.environ, fixture["environment"], clear=False),
                self.assertRaisesRegex(StudyGateError, "lacks expected shard"),
            ):
                self._import(fixture)

    def test_receipt_row_claim_must_match_authenticated_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory), record_rows=1)
            with (
                mock.patch.object(recovery, "EXPECTED_ARTIFACT_COUNT", 1),
                mock.patch.object(recovery, "EXPECTED_INFERENCE_ROWS", 1),
                mock.patch.dict(os.environ, fixture["environment"], clear=False),
                self.assertRaisesRegex(StudyGateError, "content changed"),
            ):
                self._import(fixture)

    def test_sharded_inference_does_not_consult_recovery_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.png"
            image.write_bytes(b"fixture-image-bytes")
            ctx = RunContext(
                project_root=root,
                run_root=root / "ordinary-run",
                profile="dry_run",
                run_name="ordinary",
                config={},
                config_hash="1" * 64,
                code_hash="2" * 64,
                created_utc="2026-09-21T00:00:00Z",
                resume=True,
            )
            for name in ("stages", "blobs", "checkpoints", "logs", "results", "plots"):
                (ctx.run_root / name).mkdir(parents=True, exist_ok=True)
            ctx._initialize()
            request = InferenceRequest(
                "ordinary-observation",
                str(image),
                "urban",
                0,
                {
                    "gps_lat": 24.0,
                    "gps_lon": 54.0,
                    "altitude_m": 50.0,
                    "yaw_deg": 0.0,
                },
            )
            with mock.patch.dict(
                os.environ,
                {"SEMUAV_ENABLE_V2R9_EXP1_RECOVERY": "0"},
                clear=False,
            ):
                frame = run_inference_cache_sharded(
                    DeterministicMockBackend(),
                    [request],
                    ctx,
                    scientific=False,
                    execution_context={"ordinary": True},
                )
            self.assertEqual(frame["observation_id"].tolist(), ["ordinary-observation"])

    def test_enabled_recovery_remains_restricted_at_public_cache_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.png"
            image.write_bytes(b"fixture-image-bytes")
            ctx = RunContext(
                project_root=root,
                run_root=root / "ordinary-run",
                profile="dry_run",
                run_name="ordinary",
                config={},
                config_hash="1" * 64,
                code_hash="2" * 64,
                created_utc="2026-09-21T00:00:00Z",
                resume=True,
            )
            for name in (
                "stages", "blobs", "checkpoints", "logs", "results", "plots",
            ):
                (ctx.run_root / name).mkdir(parents=True, exist_ok=True)
            ctx._initialize()
            request = InferenceRequest(
                "ordinary-observation",
                str(image),
                "urban",
                0,
                {
                    "gps_lat": 24.0,
                    "gps_lon": 54.0,
                    "altitude_m": 50.0,
                    "yaw_deg": 0.0,
                },
            )
            with (
                mock.patch.dict(
                    os.environ,
                    {"SEMUAV_ENABLE_V2R9_EXP1_RECOVERY": "1"},
                    clear=False,
                ),
                self.assertRaisesRegex(
                    StudyGateError,
                    "restricted to the authenticated v2r13/v3/v3r1/v3r2 full runs",
                ),
            ):
                run_inference_cache_sharded(
                    DeterministicMockBackend(),
                    [request],
                    ctx,
                    scientific=False,
                    execution_context={"ordinary": True},
                )


if __name__ == "__main__":
    unittest.main()
