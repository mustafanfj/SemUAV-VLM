"""V3 raw inference recovery must not silently change measurement or inference."""
from pathlib import Path
import hashlib
import json
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy" / "v2r13"


class RawInferenceCompatibilityTests(unittest.TestCase):
    def test_raw_producer_and_measurement_sources_are_byte_identical(self):
        manifest = json.loads((LEGACY / "RELEASE_MANIFEST.json").read_text())
        inventory = {record["path"]: record for record in manifest["payload_inventory"]}
        for name in (
            "semuav/perception.py", "semuav/visdrone_models.py",
            "semuav/visdrone_measurement.py", "semuav/hardware.py",
            "prompts/visdrone_proposal_decision_v3.txt",
            "acceptance/visdrone_runtime_smoke_canary_v1.json",
            "requirements.txt",
        ):
            with self.subTest(source=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
                                 inventory[name]["sha256"])

    def test_registered_configuration_changes_only_run_namespaces(self):
        original = yaml.safe_load((LEGACY / "configs/visdrone_paper.yaml").read_text())
        current = yaml.safe_load((ROOT / "configs/visdrone_paper.yaml").read_text())
        original["study_id"] = "semuav_vlm_visdrone_paper_v3r2"
        original["paths"]["cache_root"] = "${SEMUAV_CACHE_ROOT:-.cache/visdrone_paper_v3r2}"
        original["runtime"]["pre_test_calibration_gate_amendment_record"] = (
            "PRE_TEST_CALIBRATION_GATE_AMENDMENT_20261013_V3R2.md"
        )
        original["runtime"]["required_environment_guards"]["HF_MODULES_CACHE"] = (
            "/data/Lane Detection Datasets/musta/semuav_vlm/.cache/huggingface/modules-v3r1"
        )
        self.assertEqual(current, original)


if __name__ == "__main__":
    unittest.main()
