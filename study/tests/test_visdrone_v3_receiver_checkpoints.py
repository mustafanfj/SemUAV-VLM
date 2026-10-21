from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from PIL import Image

from semuav.registry import StudyGateError
from semuav.visdrone_receivers import MeasuredValue, NativeReceiverDesign, ReceiverControlContext, build_native_receiver_cache
from test_visdrone_receivers import _fake_cbor_module, _fusion_receipt, _semantic_cache


class NativeReceiverCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "source.png"
        Image.new("RGB", (32, 24), "red").save(self.path)
        self.manifest = pd.DataFrame([{"sample_id": "s1", "image_path": str(self.path), "clip_paths_json": "[]"}])
        self.contexts = {"s1": ReceiverControlContext(
            mission_profile="general_mobility", registered_roi_xyxy=(0.25, 0.25, 0.75, 0.85),
            capture_time_us=0, gps_lat=24.0, gps_lon=54.0, altitude_m=70.0, yaw_deg=0.0,
        )}
        self.calls = []
        patch = mock.patch.dict(sys.modules, {"cbor2": _fake_cbor_module()})
        patch.start()
        self.addCleanup(patch.stop)

    def measure(self, context, operation):
        self.calls.append((context.variant, context.stage))
        return MeasuredValue(operation(), latency_ms=0.25, energy_j=0.01)

    def build(self, *, measure=None, digest="a" * 64):
        return build_native_receiver_cache(
            self.manifest, _semantic_cache(),
            design=NativeReceiverDesign(semantic_source_model="model_a", image_codecs=("JPEG",), image_qualities=(50,), image_scales=(1.0,)),
            control_contexts=self.contexts, receiver=lambda value: "REVIEW", measurement_runner=measure or self.measure,
            sender_hardware_id="sender", receiver_hardware_id="receiver", sender_revision="v3-source",
            receiver_revision="receiver-revision", receiver_fusion_receipt=_fusion_receipt(),
            output_path=self.root / "native.csv.gz", checkpoint_root=self.root / "checkpoints",
            checkpoint_context_sha256=digest,
        )

    def test_interrupted_run_resumes_only_completed_rows_then_atomically_writes_cache(self) -> None:
        def fail_on_third_row(context, operation):
            if context.variant == "l2_cbor":
                raise RuntimeError("interruption")
            return self.measure(context, operation)

        with self.assertRaisesRegex(RuntimeError, "interruption"):
            self.build(measure=fail_on_third_row)
        self.assertFalse((self.root / "native.csv.gz").exists())
        self.assertEqual(len(list((self.root / "checkpoints").rglob("*.json"))), 2)
        self.calls.clear()
        result = self.build()
        self.assertEqual(len(result), 6)
        self.assertEqual(len(self.calls), 8)  # four remaining rows × encode/receive
        self.assertTrue((self.root / "native.csv.gz").is_file())
        self.calls.clear()
        resumed = self.build()
        self.assertEqual(self.calls, [])
        pd.testing.assert_frame_equal(result, resumed)

    def test_modified_checkpoint_row_is_rejected(self) -> None:
        self.build()
        for path in (self.root / "checkpoints").rglob("*.json"):
            record = json.loads(path.read_text())
            if record["row"]["variant"] == "l0_cbor":
                record["row"]["generation_ms"] = 0
                path.write_text(json.dumps(record))
                break
        with self.assertRaisesRegex(StudyGateError, "checkpoint is invalid"):
            self.build()

    def test_changed_source_pixels_cannot_reuse_measurements(self) -> None:
        self.build()
        Image.new("RGB", (32, 24), "blue").save(self.path)
        with self.assertRaisesRegex(StudyGateError, "wire payload changed"):
            self.build()

    def test_changed_receiver_context_cannot_reuse_measurements(self) -> None:
        self.build()
        self.contexts["s1"] = dataclasses.replace(self.contexts["s1"], yaw_deg=1.0)
        with self.assertRaisesRegex(StudyGateError, "identity/hash mismatch"):
            self.build()

    def test_changed_sealed_run_context_starts_separate_measurements(self) -> None:
        self.build()
        self.calls.clear()
        self.build(digest="b" * 64)
        self.assertEqual(len(self.calls), 12)
        self.assertEqual(len(list((self.root / "checkpoints").iterdir())), 2)


if __name__ == "__main__":
    unittest.main()
