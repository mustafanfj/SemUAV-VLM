from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import semuav.visdrone_models as models
from semuav.registry import StudyGateError, atomic_write_json, hash_file, hash_object


AMENDMENT_ID = "semuav.visdrone-v2r6-prospective-moondream-exclusion.v1"
EXCLUSION_REASON = (
    "v2r5_moondream2_runtime_smoke_fail_schema_before_validation_or_test_access"
)


class V2R6ModelLockDerivationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.cache = self.root / "huggingface"
        self.cache.mkdir()
        self.source_lock = self.root / "SOURCE_MODEL_LOCK.json"
        self.active_lock = self.root / "MODEL_LOCK.json"
        self.derivation = self.root / "MODEL_LOCK_DERIVATION.json"
        self.snapshots: dict[tuple[str, str], Path] = {}
        self.source = self._write_historical_source_lock()
        self.source_sha256 = hash_file(self.source_lock)

    def _write_historical_source_lock(self) -> dict[str, object]:
        compact_repos = models.HISTORICAL_V2R5_COMPACT_MODEL_REPOS
        backends = models.HISTORICAL_V2R5_MODEL_BACKENDS
        specs = [
            (model_id, repo, "compact", backends[model_id])
            for model_id, repo in compact_repos.items()
        ] + [(
            models.OFFICIAL_UPPER_BOUND_ID,
            models.OFFICIAL_UPPER_BOUND_REPO,
            "upper_bound",
            "qwen3_vl",
        )]
        rows = []
        for index, (model_id, repo, role, backend) in enumerate(specs, start=1):
            revision = f"{index:x}" * 40
            snapshot = self.cache / "snapshots" / revision
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text(
                json.dumps({"model_id": model_id}) + "\n",
                encoding="utf-8",
            )
            inventory = models._snapshot_inventory(snapshot, self.cache)
            self.snapshots[(repo, revision)] = snapshot.resolve()
            rows.append({
                "model_id": model_id,
                "role": role,
                "backend": backend,
                "repo": repo,
                "requested_revision": None,
                "quantization": "none",
                "dtype": "bfloat16",
                "calibration_eligible": (
                    role == "compact" and model_id != "moondream2"
                ),
                "resolved_revision": revision,
                "snapshot_path": str(snapshot.resolve()),
                "snapshot_relative_path": (
                    snapshot.resolve().relative_to(self.cache).as_posix()
                ),
                "file_count": len(inventory),
                "total_bytes": sum(item["bytes"] for item in inventory),
                "inventory_sha256": hash_object(inventory),
                "files": inventory,
            })
        payload = {
            "schema_version": models.MODEL_LOCK_SCHEMA_VERSION,
            "lock_type": models.MODEL_LOCK_TYPE,
            "created_utc": "2026-09-13T00:00:00Z",
            "cache_root": str(self.cache),
            "models": rows,
        }
        payload["model_set_sha256"] = models._model_set_fingerprint(rows)
        atomic_write_json(self.source_lock, payload)
        return payload

    def _offline(self, repo: str, revision: str, cache: Path) -> Path:
        self.assertEqual(Path(cache).resolve(), self.cache)
        return self.snapshots[(repo, revision)]

    def _derive(self, *, allow_create: bool = True) -> dict[str, object]:
        return models.derive_v2r6_three_model_lock(
            self.source_lock,
            destination=self.active_lock,
            derivation_destination=self.derivation,
            expected_source_sha256=self.source_sha256,
            amendment_id=AMENDMENT_ID,
            exclusion_reason=EXCLUSION_REASON,
            offline_checker=self._offline,
            allow_create=allow_create,
        )

    def test_derivation_filters_only_moondream_and_preserves_ordered_rows(self) -> None:
        observed = self._derive()
        active = json.loads(self.active_lock.read_text(encoding="utf-8"))
        receipt = json.loads(self.derivation.read_text(encoding="utf-8"))
        expected_rows = [
            row for row in self.source["models"]
            if row["model_id"] != "moondream2"
        ]

        self.assertEqual(active["models"], expected_rows)
        self.assertEqual(
            [row["model_id"] for row in active["models"]],
            list(models.OFFICIAL_RUNTIME_MODEL_IDS),
        )
        self.assertEqual(
            receipt["schema_version"],
            models.V2R6_MODEL_LOCK_DERIVATION_SCHEMA,
        )
        self.assertEqual(receipt["source_model_lock_sha256"], self.source_sha256)
        self.assertEqual(receipt["amendment_id"], AMENDMENT_ID)
        self.assertEqual(receipt["exclusion_reason"], EXCLUSION_REASON)
        self.assertEqual(receipt["excluded_model_ids"], ["moondream2"])
        self.assertEqual(
            receipt["derivation_operation"],
            "ordered_filter_excluding_only_moondream2",
        )
        self.assertIs(receipt["prior_runtime_smoke_rows_reused"], False)
        self.assertIs(receipt["validation_or_test_evidence_used"], False)
        self.assertNotIn("runtime_smoke", active)
        self.assertNotIn("models", receipt)
        self.assertNotIn("smoke_report", receipt)
        self.assertEqual(
            observed["derivation_mode"],
            {"model_lock": "CREATED", "receipt": "CREATED"},
        )

    def test_read_only_reverify_succeeds_but_never_recreates_missing_target(self) -> None:
        self._derive()
        observed = self._derive(allow_create=False)
        self.assertEqual(
            observed["derivation_mode"],
            {
                "model_lock": "REVERIFIED_EXISTING",
                "receipt": "REVERIFIED_EXISTING",
            },
        )

        self.derivation.unlink()
        with self.assertRaisesRegex(StudyGateError, "read-only reverify"):
            self._derive(allow_create=False)
        self.assertFalse(self.derivation.exists())

    def test_read_only_mode_refuses_initial_creation(self) -> None:
        with self.assertRaisesRegex(StudyGateError, "read-only reverify"):
            self._derive(allow_create=False)
        self.assertFalse(self.active_lock.exists())
        self.assertFalse(self.derivation.exists())

    def test_source_anchor_and_existing_derived_bytes_are_tamper_evident(self) -> None:
        self._derive()
        active = json.loads(self.active_lock.read_text(encoding="utf-8"))
        active["models"][0]["repo"] = "attacker/repack"
        atomic_write_json(self.active_lock, active)
        with self.assertRaisesRegex(StudyGateError, "exact derivation"):
            self._derive()

        self.source_lock.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(StudyGateError, "external SHA-256 anchor"):
            self._derive()


if __name__ == "__main__":
    unittest.main()
