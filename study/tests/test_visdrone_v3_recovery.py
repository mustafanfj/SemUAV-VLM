"""The v3 successor may import only the authenticated raw v2r9 shards."""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from semuav.registry import StudyGateError
from semuav import visdrone_recovery as recovery
import test_visdrone_v2r9_recovery as legacy_fixtures


class V3RecoveryAllowlistTests(unittest.TestCase):
    def test_v3_import_preserves_authenticated_raw_shard_bytes(self) -> None:
        fixture_builder = legacy_fixtures.V2R9InferenceRecoveryTests()
        with tempfile.TemporaryDirectory() as directory:
            fixture = fixture_builder._fixture(Path(directory))
            fixture["ctx"] = replace(fixture["ctx"], run_name="semuav_visdrone_v3r1")
            with (
                mock.patch.object(recovery, "EXPECTED_ARTIFACT_COUNT", 1),
                mock.patch.object(recovery, "EXPECTED_INFERENCE_ROWS", 2),
                mock.patch.dict(os.environ, fixture["environment"], clear=False),
            ):
                self.assertTrue(fixture_builder._import(fixture))
            record = fixture["ctx"].stage_record(fixture_builder.STAGE_NAME)
            self.assertIsNotNone(record)
            self.assertEqual(
                Path(record["artifact"]).read_bytes(),
                fixture["source"].read_bytes(),
            )
            self.assertFalse((fixture["ctx"].run_root / "test_access_ledger.json").exists())

    def test_new_or_similar_run_names_do_not_inherit_recovery_authority(self) -> None:
        fixture_builder = legacy_fixtures.V2R9InferenceRecoveryTests()
        for unauthorized in (
            "semuav_visdrone_v3r1_test", "semuav_visdrone_v3_test", "semuav_visdrone_v30", "ordinary",
        ):
            with self.subTest(run_name=unauthorized), tempfile.TemporaryDirectory() as directory:
                fixture = fixture_builder._fixture(Path(directory))
                fixture["ctx"] = replace(fixture["ctx"], run_name=unauthorized)
                with (
                    mock.patch.object(recovery, "EXPECTED_ARTIFACT_COUNT", 1),
                    mock.patch.object(recovery, "EXPECTED_INFERENCE_ROWS", 2),
                    mock.patch.dict(os.environ, fixture["environment"], clear=False),
                ):
                    with self.assertRaisesRegex(StudyGateError, "restricted"):
                        fixture_builder._import(fixture)
                self.assertIsNone(fixture["ctx"].stage_record(fixture_builder.STAGE_NAME))


if __name__ == "__main__":
    unittest.main()
