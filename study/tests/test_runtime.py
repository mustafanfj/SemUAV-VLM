from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from semuav.hardware import environment_report
from semuav.protocol import encode_video_clip
from semuav.runtime import probe_ffmpeg_encoder, resolve_ffmpeg_executable


class FFmpegRuntimeTests(unittest.TestCase):
    @staticmethod
    def _executable(root: Path, name: str) -> Path:
        path = root / name
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o700)
        return path

    def test_resolver_prefers_verified_path_executable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = self._executable(Path(directory), "ffmpeg-system")
            with (
                patch("semuav.runtime.shutil.which", return_value=str(executable)),
                patch(
                    "semuav.runtime.importlib.import_module",
                    side_effect=AssertionError("fallback must not be imported"),
                ),
            ):
                self.assertEqual(resolve_ffmpeg_executable(), str(executable.resolve()))

    def test_resolver_uses_bundled_imageio_ffmpeg_when_path_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = self._executable(Path(directory), "ffmpeg-bundled")
            module = SimpleNamespace(get_ffmpeg_exe=lambda: str(executable))
            with (
                patch("semuav.runtime.shutil.which", return_value=None),
                patch("semuav.runtime.importlib.import_module", return_value=module),
            ):
                self.assertEqual(resolve_ffmpeg_executable(), str(executable.resolve()))

    def test_resolver_rejects_non_executable_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "ffmpeg-not-executable"
            candidate.write_text("not executable", encoding="utf-8")
            candidate.chmod(0o600)
            module = SimpleNamespace(get_ffmpeg_exe=lambda: str(candidate))
            with (
                patch("semuav.runtime.shutil.which", return_value=None),
                patch("semuav.runtime.importlib.import_module", return_value=module),
            ):
                self.assertIsNone(resolve_ffmpeg_executable())

    def test_encoder_probe_requires_the_exact_registered_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = self._executable(Path(directory), "ffmpeg")
            encoder_listing = " V....D libx264rgb RGB encoder\n V....D libx264 H.264 encoder\n"
            completed = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=encoder_listing, stderr="",
            )
            with patch("semuav.runtime.subprocess.run", return_value=completed) as run:
                available, detail = probe_ffmpeg_encoder(executable, "libx264")
            self.assertTrue(available, detail)
            self.assertIn("libx264 available", detail)
            run.assert_called_once_with(
                [str(executable.resolve()), "-hide_banner", "-encoders"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10.0,
            )

            rgb_only = subprocess.CompletedProcess(
                args=[], returncode=0,
                stdout=" V....D libx264rgb RGB encoder\n", stderr="",
            )
            with patch("semuav.runtime.subprocess.run", return_value=rgb_only):
                available, _ = probe_ffmpeg_encoder(executable, "libx264")
            self.assertFalse(available)

    def test_environment_preflight_requires_ffmpeg_and_libx264(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("semuav.hardware.resolve_ffmpeg_executable", return_value="/verified/ffmpeg"),
                patch(
                    "semuav.hardware.probe_ffmpeg_encoder",
                    return_value=(True, "libx264 available in /verified/ffmpeg"),
                ),
            ):
                report = environment_report(root, root / "cache", "cpu", "dry_run")
            statuses = report.set_index("item")["status"].to_dict()
            self.assertEqual(statuses["ffmpeg"], "PASS")
            self.assertEqual(statuses["ffmpeg_encoder:libx264"], "PASS")

    def test_environment_preflight_fails_without_libx264(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("semuav.hardware.resolve_ffmpeg_executable", return_value="/verified/ffmpeg"),
                patch(
                    "semuav.hardware.probe_ffmpeg_encoder",
                    return_value=(False, "libx264 unavailable"),
                ),
            ):
                report = environment_report(root, root / "cache", "cpu", "dry_run")
            row = report.set_index("item").loc["ffmpeg_encoder:libx264"]
            self.assertEqual(row["status"], "FAIL")
            self.assertEqual(row["required_for"], "experiments_2_7")

    def test_video_encoder_launches_only_the_resolved_executable(self) -> None:
        images = [
            Image.new("RGB", (8, 8), (255, 0, 0)),
            Image.new("RGB", (8, 8), (0, 255, 0)),
        ]
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"encoded-video", stderr=b"",
        )
        with (
            patch("semuav.protocol.resolve_ffmpeg_executable", return_value="/verified/ffmpeg"),
            patch("semuav.protocol.subprocess.run", return_value=completed) as run,
        ):
            self.assertEqual(encode_video_clip(images), b"encoded-video")
        self.assertEqual(run.call_args.args[0][0], "/verified/ffmpeg")
        self.assertNotIn("shell", run.call_args.kwargs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
