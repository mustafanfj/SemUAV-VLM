from __future__ import annotations

import io
import struct
import subprocess
import types
import unittest

import numpy as np
from PIL import Image, ImageDraw

from semuav.protocol import encode_video_clip, video_geometry_from_payload
from semuav.registry import StudyGateError
from semuav.runtime import probe_ffmpeg_encoder, resolve_ffmpeg_executable
from semuav.visdrone_receiver_runtime import RegisteredVisDroneNativeReceiver
from semuav.visdrone_receivers import DecodedNativePayload, ReceiverControlContext, _decode_video


FFMPEG = resolve_ffmpeg_executable()


@unittest.skipUnless(FFMPEG, "FFmpeg is unavailable")
class VideoGeometryTests(unittest.TestCase):
    def frames(self, size: tuple[int, int]) -> list[Image.Image]:
        frames = []
        for fill in (40, 190):
            image = Image.new("RGB", size, (fill, fill, fill))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 9, 9), fill="white")
            draw.rectangle((size[0] - 10, size[1] - 10, size[0] - 1, size[1] - 1), fill="black")
            frames.append(image)
        return frames

    def test_h264_and_h265_roundtrip_all_dimension_parities_and_reported_failure(self) -> None:
        for codec in ("libx264", "libx265"):
            available, _ = probe_ffmpeg_encoder(FFMPEG, codec)
            if not available:
                if codec == "libx264":
                    self.fail("The registered H264 encoder is missing")
                continue
            for size in ((64, 48), (65, 48), (64, 49), (65, 49), (1360, 765)):
                with self.subTest(codec=codec, size=size):
                    payload = encode_video_clip(self.frames(size), codec, crf=18)
                    frames = _decode_video(payload, None)
                    self.assertEqual([frame.size for frame in frames], [size, size])
                    self.assertEqual(payload[4:8], b"ftyp")  # remains a regular MP4
                    geometry = video_geometry_from_payload(payload)
                    self.assertEqual(geometry, (*size, 2) if any(value % 2 for value in size) else None)
                    # Temporal order and full-frame corner positions survive.
                    self.assertLess(float(np.asarray(frames[0])[15:25, 15:25].mean()), 80)
                    self.assertGreater(float(np.asarray(frames[1])[15:25, 15:25].mean()), 150)
                    self.assertGreater(float(np.asarray(frames[0])[2:7, 2:7].mean()), 225)
                    self.assertLess(float(np.asarray(frames[1])[-7:-2, -7:-2].mean()), 30)

    def test_padded_encoding_is_byte_identical_across_measurement_repetitions(self) -> None:
        frames = self.frames((65, 49))
        for codec in ("libx264", "libx265"):
            available, _ = probe_ffmpeg_encoder(FFMPEG, codec)
            if available:
                with self.subTest(codec=codec):
                    self.assertEqual(encode_video_clip(frames, codec), encode_video_clip(frames, codec))

    def test_original_ffmpeg_command_reproduces_1360_by_765_error(self) -> None:
        stream = io.BytesIO()
        for frame in self.frames((1360, 765)):
            frame.save(stream, format="PNG")
        process = subprocess.run([
            FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "image2pipe",
            "-vcodec", "png", "-framerate", "5", "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", "28",
            "-pix_fmt", "yuv420p", "-movflags", "frag_keyframe+empty_moov",
            "-f", "mp4", "pipe:1",
        ], input=stream.getvalue(), capture_output=True, check=False)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn(b"height not divisible by 2", process.stderr)

    def test_mixed_source_geometry_is_rejected_instead_of_resized(self) -> None:
        with self.assertRaisesRegex(StudyGateError, "dimensions differ"):
            encode_video_clip([Image.new("RGB", (65, 49)), Image.new("RGB", (64, 49))])

    def test_metadata_is_transmitted_and_validated_not_borrowed_from_source(self) -> None:
        payload = encode_video_clip(self.frames((65, 49)))
        self.assertEqual(len(payload[-40:]), 40)
        self.assertEqual(payload[-36:-32], b"uuid")
        with self.assertRaisesRegex(StudyGateError, "duplicated"):
            _decode_video(payload + payload[-40:], None)
        for field_offset, value in ((-12, 71), (-4, 3)):
            changed = bytearray(payload)
            struct.pack_into("!I", changed, len(payload) + field_offset, value)
            with self.assertRaisesRegex(StudyGateError, "disagree"):
                _decode_video(bytes(changed), None)
        with self.assertRaisesRegex(StudyGateError, "MP4 box size"):
            _decode_video(payload[:-1], None)

    def test_geometry_bytes_in_video_picture_data_are_not_metadata(self) -> None:
        payload = encode_video_clip(self.frames((65, 49)))
        fake_picture = struct.pack("!I4s", 48, b"mdat") + payload[-40:]
        self.assertIsNone(video_geometry_from_payload(fake_picture))

    def test_native_detector_observes_restored_dimensions_and_registered_roi(self) -> None:
        observed = []

        class Detector:
            def detect(self, path, *, persist=False):
                with Image.open(path) as image:
                    observed.append((image.size, persist))
                return ()

        context = ReceiverControlContext(
            mission_profile="general_mobility", registered_roi_xyxy=(0.25, 0.25, 0.75, 0.85),
            capture_time_us=1, gps_lat=24.0, gps_lon=54.0, altitude_m=70.0, yaw_deg=0.0,
        )
        receiver = RegisteredVisDroneNativeReceiver(
            Detector(), detector_revision="test-revision",
            priority_profiles={"general_mobility": ("car",)},
            registered_roi_xyxy=context.registered_roi_xyxy,
        )
        requests = []

        def infer(request):
            requests.append(request)
            return types.SimpleNamespace(schema_valid=True, parsed={"recommended_action": "REVIEW"})

        receiver.policy = types.SimpleNamespace(infer=infer)
        payload = encode_video_clip(self.frames((1360, 765)))
        result = receiver(DecodedNativePayload(
            sample_id="geometry-test", variant="h264_crf28_fps5", source_model="ground_visual_receiver",
            media_kind="video", payload_sha256="a" * 64, control_context=context,
            control_context_sha256=context.sha256, frames=_decode_video(payload, None),
        ))
        self.assertEqual(result, "REVIEW")
        self.assertEqual(observed, [((1360, 765), False), ((1360, 765), True)])
        self.assertTrue(all(tuple(request.telemetry["registered_roi_xyxy"]) == context.registered_roi_xyxy for request in requests))


if __name__ == "__main__":
    unittest.main()
