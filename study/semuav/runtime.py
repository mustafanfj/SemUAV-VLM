from __future__ import annotations

import importlib
import os
import re
import shutil
import subprocess
from pathlib import Path


_FFMPEG_ENCODER_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _usable_executable(candidate: str | os.PathLike[str] | None) -> str | None:
    """Return a canonical executable file path, or ``None`` when unusable."""
    if candidate is None or not str(candidate).strip():
        return None
    try:
        resolved = Path(candidate).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        return None
    return str(resolved)


def resolve_ffmpeg_executable() -> str | None:
    """Resolve FFmpeg without downloading or modifying the environment.

    An explicitly installed executable on ``PATH`` takes precedence.  When it
    is absent or unusable, the executable bundled by the pinned
    ``imageio-ffmpeg`` wheel is used.  Every returned path is resolved to an
    existing executable file before callers may launch it.
    """
    path_executable = _usable_executable(shutil.which("ffmpeg"))
    if path_executable is not None:
        return path_executable

    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
        bundled_candidate = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    return _usable_executable(bundled_candidate)


def probe_ffmpeg_encoder(
    executable: str | os.PathLike[str],
    encoder: str,
    *,
    timeout_s: float = 10.0,
) -> tuple[bool, str]:
    """Check an exact FFmpeg encoder name without invoking a shell."""
    executable_path = _usable_executable(executable)
    if executable_path is None:
        return False, "FFmpeg executable is missing or not executable"
    if not _FFMPEG_ENCODER_NAME.fullmatch(str(encoder)):
        raise ValueError(f"Invalid FFmpeg encoder name: {encoder!r}")
    if timeout_s <= 0:
        raise ValueError("FFmpeg probe timeout must be positive")

    try:
        process = subprocess.run(
            [executable_path, "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"FFmpeg encoder probe failed: {exc}"
    if process.returncode:
        detail = (process.stderr or process.stdout or "unknown error").strip()
        return False, f"FFmpeg encoder probe exited {process.returncode}: {detail[:300]}"

    available = {
        fields[1]
        for line in process.stdout.splitlines()
        if len(fields := line.split()) >= 2
        and len(fields[0]) == 6
        and fields[0][0] in {"V", "A", "S", "."}
    }
    if encoder not in available:
        return False, f"FFmpeg encoder {encoder} is unavailable in {executable_path}"
    return True, f"{encoder} available in {executable_path}"
