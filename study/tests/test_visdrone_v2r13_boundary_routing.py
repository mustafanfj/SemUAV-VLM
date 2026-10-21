"""Verify the deployed-and-sealed predecessor remains at its own layer."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from test_visdrone_v2r12_boundary_routing import StudyGateError, VerifierHarness


class V2R13PredecessorBoundaryTests(unittest.TestCase):
    def setUp(self):
        previous = os.environ.get("SEMUAV_V2R13_DEPLOYMENT_RECEIPT_SHA256")
        def restore():
            if previous is None:
                os.environ.pop("SEMUAV_V2R13_DEPLOYMENT_RECEIPT_SHA256", None)
            else:
                os.environ["SEMUAV_V2R13_DEPLOYMENT_RECEIPT_SHA256"] = previous
        self.addCleanup(restore)

    def test_v2r12_evidence_and_preservation_record_mutations_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = VerifierHarness(directory)
            for field in ("active_model_lock_sha256", "runtime_smoke_report_sha256",
                          "scientific_environment_seal_sha256", "deployment_receipt_sha256"):
                with self.subTest(field=field):
                    receipt = copy.deepcopy(harness.receipt)
                    receipt["authenticated_predecessor_v2r12"][field] = "0" * 64
                    with self.assertRaises(StudyGateError):
                        harness.verify(receipt)
            for field, value in (("schema_version", "wrong"), ("output_root", "/wrong"),
                                 ("release_tree", "ABSENT"), ("cache_tree", "a" * 64 + " 0 0"),
                                 ("unchanged_after_deployment_actions", False)):
                with self.subTest(field=field):
                    receipt = copy.deepcopy(harness.receipt)
                    receipt["predecessor_v2r12_state"][field] = value
                    with self.assertRaises(StudyGateError):
                        harness.verify(receipt)

    def test_historical_v2r11_boundary_cannot_be_replaced_at_v2r12_layer(self):
        with tempfile.TemporaryDirectory() as directory:
            for replacement in ("inherited_v2r9", "inherited_v2r10", "top_boundary"):
                with self.subTest(replacement=replacement):
                    harness = VerifierHarness(str(Path(directory) / replacement))
                    sealed = harness.records["sealed"]
                    for path, digest in (
                        (Path(sealed["release_path"]) / "RELEASE_MANIFEST.json", sealed["release_manifest_sha256"]),
                        (Path(sealed["deployment_receipt"]), sealed["deployment_receipt_sha256"]),
                    ):
                        payload = json.loads(path.read_text(encoding="utf-8"))
                        payload["predecessor_observed_terminal_boundary"] = copy.deepcopy(
                            harness.records[replacement]
                        )
                        harness._put(path, payload, claimed_sha256=digest)
                    with self.assertRaises(StudyGateError):
                        harness.verify()


if __name__ == "__main__":
    unittest.main()
