from __future__ import annotations

import dataclasses
import importlib.metadata
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .evaluation import (
    assert_paired_design,
    budget_compliance,
    holm_adjust,
    two_way_clustered_paired_bootstrap,
)
from .registry import StudyGateError, hash_file, matched_coverage_semantic_errors
from .runtime import probe_ffmpeg_encoder, resolve_ffmpeg_executable


@dataclasses.dataclass(frozen=True)
class EnvironmentCheck:
    item: str
    status: str
    value: str
    required_for: str


def environment_report(project_root: str | Path, cache_root: str | Path, device: str, profile: str) -> pd.DataFrame:
    checks: list[EnvironmentCheck] = []
    def add(item: str, passed: bool, value: Any, required_for: str = "full", applicable: bool = True) -> None:
        status = "PASS" if passed else "FAIL" if applicable else "SKIP"
        checks.append(EnvironmentCheck(item, status, str(value), required_for))
    project_root, cache_root = Path(project_root), Path(cache_root)
    add("python", tuple(map(int, platform.python_version_tuple()[:2])) >= (3, 10), platform.python_version(), "all")
    add("project_root", project_root.is_dir(), project_root, "all")
    cache_root.mkdir(parents=True, exist_ok=True)
    add("cache_writable", os.access(cache_root, os.W_OK), cache_root, "all")
    add("not_login_node", not any(token in socket.gethostname().lower() for token in ("login", "head")), socket.gethostname(), "full")
    ffmpeg = resolve_ffmpeg_executable()
    add("ffmpeg", ffmpeg is not None, ffmpeg or "unavailable", "experiments_2_7")
    if ffmpeg is None:
        libx264_available, libx264_detail = False, "FFmpeg unavailable"
    else:
        libx264_available, libx264_detail = probe_ffmpeg_encoder(ffmpeg, "libx264")
    add(
        "ffmpeg_encoder:libx264", libx264_available, libx264_detail,
        "experiments_2_7",
    )
    add("tc_netem", shutil.which("tc") is not None, shutil.which("tc"), "experiment_7b")
    try:
        import torch  # type: ignore
        cuda = bool(torch.cuda.is_available())
        if cuda and device.startswith("cuda"):
            properties = torch.cuda.get_device_properties(device)
            gpu = (
                f"{properties.name}|cc={properties.major}.{properties.minor}"
                f"|memory_bytes={properties.total_memory}|visible_count={torch.cuda.device_count()}"
            )
        else:
            gpu = "CPU"
        cuda_runtime = str(torch.version.cuda or "unavailable")
        cudnn_version = str(torch.backends.cudnn.version() or "unavailable")
    except Exception as exc:
        cuda, gpu = False, f"unavailable: {exc}"
        cuda_runtime = cudnn_version = "unavailable"
    add("cuda", cuda, gpu, "real_profiles", applicable=profile != "dry_run")
    add(
        "cuda_runtime", cuda_runtime != "unavailable", cuda_runtime,
        "real_profiles", applicable=profile != "dry_run",
    )
    add(
        "cudnn", cudnn_version != "unavailable", cudnn_version,
        "real_profiles", applicable=profile != "dry_run",
    )
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip().replace("\n", ",")
        driver_available = bool(driver)
    except (FileNotFoundError, subprocess.SubprocessError):
        driver, driver_available = "unavailable", False
    add(
        "nvidia_driver", driver_available, driver,
        "real_profiles", applicable=profile != "dry_run",
    )
    packages = [
        "numpy", "pandas", "scipy", "torch", "transformers", "ultralytics",
        "cbor2", "pillow-avif-plugin", "zstandard", "imageio-ffmpeg",
    ]
    for package in packages:
        try:
            version = importlib.metadata.version(package)
            passed = True
        except importlib.metadata.PackageNotFoundError:
            version, passed = "missing", False
        core = package in {"numpy", "pandas", "scipy"}
        add(f"package:{package}", passed, version, "all" if core else "real_profiles", applicable=core or profile != "dry_run")
    return pd.DataFrame(dataclasses.asdict(check) for check in checks)


def environment_fingerprint(report: pd.DataFrame) -> dict[str, Any]:
    """Return the exact executable/runtime identity frozen by the protocol.

    Package versions alone are not sufficient for a GPU/video experiment: two
    hosts can report the same Python packages while using different Python or
    FFmpeg binaries, CUDA/cuDNN runtimes, drivers, or visible devices.  This
    fingerprint intentionally binds those scientific execution dependencies.
    It contains no volatile utilization, temperature, clock, or timestamp
    fields, so recomputation immediately before the one-time test is exact.
    """
    required = {"item", "status", "value", "required_for"}
    if missing := required.difference(report.columns):
        raise ValueError(f"Environment report is missing columns: {sorted(missing)}")
    if report.empty or report["item"].astype(str).duplicated().any():
        raise StudyGateError("Environment report must contain one row per check")
    checks = [
        {
            "item": str(row.item),
            "status": str(row.status),
            "value": str(row.value),
            "required_for": str(row.required_for),
        }
        for row in report.sort_values("item").itertuples(index=False)
    ]
    distributions = sorted({
        (
            str(distribution.metadata.get("Name") or "unknown").strip().lower(),
            str(distribution.version),
        )
        for distribution in importlib.metadata.distributions()
    })
    python_executable = Path(sys.executable).resolve()
    python_binary = {
        "path": str(python_executable),
        "sha256": hash_file(python_executable) if python_executable.is_file() else "",
        "bytes": python_executable.stat().st_size if python_executable.is_file() else 0,
        "version": sys.version,
        "version_info": list(sys.version_info[:5]),
        "implementation": platform.python_implementation(),
        "compiler": platform.python_compiler(),
        "build": list(platform.python_build()),
    }

    ffmpeg_path = resolve_ffmpeg_executable()
    ffmpeg: dict[str, Any] = {
        "path": "",
        "sha256": "",
        "bytes": 0,
        "version_output": "",
        "buildconf_output": "",
        "libx264_available": False,
        "libx264_detail": "FFmpeg unavailable",
    }
    if ffmpeg_path:
        resolved_ffmpeg = Path(ffmpeg_path).resolve()
        try:
            version_output = subprocess.run(
                [str(resolved_ffmpeg), "-version"], check=True,
                capture_output=True, text=True, timeout=20,
            ).stdout.strip()
            buildconf_output = subprocess.run(
                [str(resolved_ffmpeg), "-buildconf"], check=True,
                capture_output=True, text=True, timeout=20,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise StudyGateError(
                f"Cannot fingerprint the resolved FFmpeg executable: {resolved_ffmpeg}"
            ) from exc
        libx264_available, libx264_detail = probe_ffmpeg_encoder(
            str(resolved_ffmpeg), "libx264",
        )
        ffmpeg = {
            "path": str(resolved_ffmpeg),
            "sha256": hash_file(resolved_ffmpeg),
            "bytes": resolved_ffmpeg.stat().st_size,
            "version_output": version_output,
            "buildconf_output": buildconf_output,
            "libx264_available": bool(libx264_available),
            "libx264_detail": str(libx264_detail),
        }

    torch_runtime: dict[str, Any] = {
        "available": False,
        "version": "unavailable",
        "compiled_cuda": "unavailable",
        "cudnn": "unavailable",
        "cuda_available": False,
        "visible_devices": [],
    }
    try:
        import torch  # type: ignore

        visible_devices: list[dict[str, Any]] = []
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                visible_devices.append({
                    "logical_index": index,
                    "name": str(properties.name),
                    "compute_capability": [
                        int(properties.major), int(properties.minor),
                    ],
                    "total_memory_bytes": int(properties.total_memory),
                    "uuid": _normalized_device_identifier(
                        getattr(properties, "uuid", ""),
                    ),
                    "pci_bus_id": _normalized_device_identifier(
                        getattr(
                            properties, "pci_bus_id",
                            getattr(properties, "pciBusID", ""),
                        ),
                    ),
                })
        torch_runtime = {
            "available": True,
            "version": str(torch.__version__),
            "compiled_cuda": str(torch.version.cuda or "unavailable"),
            "cudnn": str(torch.backends.cudnn.version() or "unavailable"),
            "cuda_available": bool(torch.cuda.is_available()),
            "visible_devices": visible_devices,
        }
    except ImportError:
        pass

    return {
        "schema_version": "semuav.environment.v2",
        "python": python_binary,
        # Retained as a convenience field for older audit readers.
        "python_executable": str(python_executable),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "execution_environment": {
            key: os.environ.get(key, "")
            for key in (
                "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG", "HF_HOME", "TORCH_HOME",
            )
        },
        "torch_runtime": torch_runtime,
        "ffmpeg": ffmpeg,
        "installed_distributions": [
            {"name": name, "version": version} for name, version in distributions
        ],
        "checks": checks,
    }


def assert_environment_ready(report: pd.DataFrame, required_scopes: Sequence[str]) -> None:
    """Fail before a scientific stage when an applicable runtime check failed."""
    required = {"item", "status", "required_for"}
    if missing := required.difference(report.columns):
        raise ValueError(f"Environment report is missing columns: {sorted(missing)}")
    scopes = {str(scope) for scope in required_scopes}
    applicable = report[report["required_for"].astype(str).isin(scopes)]
    failures = applicable[~applicable["status"].astype(str).eq("PASS")]
    if len(failures):
        detail = "; ".join(
            f"{row.item}={row.status} ({getattr(row, 'value', '')})"
            for row in failures.itertuples(index=False)
        )
        raise StudyGateError(f"Scientific environment gate failed: {detail}")


def require_exclusive_physical_gpu(
    physical_gpu_index: int,
    *,
    allowed_pids: Sequence[int] | None = None,
    query_timeout_s: float = 10.0,
) -> dict[str, Any]:
    """Fail closed unless a physical GPU has no unexpected compute process.

    ``nvidia-smi`` addresses the physical device directly, so this check is not
    confused by a visibility mapping such as ``CUDA_VISIBLE_DEVICES=1`` making
    physical GPU 1 appear to Python as logical ``cuda:0``.  The successful
    record intentionally contains no PID or process-count field: occupancy is a
    just-in-time mutual-exclusion gate, not part of the immutable environment
    fingerprint.
    """

    if (
        isinstance(physical_gpu_index, bool)
        or not isinstance(physical_gpu_index, int)
        or physical_gpu_index < 0
    ):
        raise ValueError("physical_gpu_index must be a nonnegative integer")
    if not isinstance(query_timeout_s, (int, float)) or query_timeout_s <= 0:
        raise ValueError("query_timeout_s must be positive")

    permitted = {os.getpid()} if allowed_pids is None else set()
    if allowed_pids is not None:
        for pid in allowed_pids:
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
                raise ValueError("allowed_pids must contain only positive integers")
            permitted.add(pid)

    command = [
        "nvidia-smi",
        "-i",
        str(physical_gpu_index),
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=float(query_timeout_s),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise StudyGateError(
            f"Physical GPU {physical_gpu_index} occupancy query failed closed"
        ) from exc

    if not isinstance(completed.stdout, str):
        raise StudyGateError(
            f"Physical GPU {physical_gpu_index} occupancy query returned malformed output"
        )
    observed: set[int] = set()
    for line_number, raw_line in enumerate(completed.stdout.splitlines(), start=1):
        value = raw_line.strip()
        if not value:
            continue
        if not value.isascii() or not value.isdecimal():
            raise StudyGateError(
                f"Physical GPU {physical_gpu_index} occupancy query returned "
                f"malformed PID data on line {line_number}"
            )
        pid = int(value)
        if pid <= 0:
            raise StudyGateError(
                f"Physical GPU {physical_gpu_index} occupancy query returned "
                "a nonpositive PID"
            )
        observed.add(pid)

    unexpected = sorted(observed.difference(permitted))
    if unexpected:
        pid_text = ",".join(map(str, unexpected))
        raise StudyGateError(
            f"Physical GPU {physical_gpu_index} is not exclusive; unexpected "
            f"compute PID(s): {pid_text}. Inspect process identity and ownership "
            "before stopping any process"
        )
    return {
        "schema_version": "semuav.physical-gpu-exclusive-gate.v1",
        "status": "PASS",
        "physical_gpu_index": physical_gpu_index,
        "allowed_process_scope": (
            "current_scientific_kernel_only"
            if permitted == {os.getpid()}
            else "explicit_allowed_process_set"
        ),
        "query_backend": "nvidia-smi",
    }


class NVMLPowerSampler:
    """Integrates GPU power at a fixed monotonic sampling interval.

    ``device_index`` is the logical CUDA index used by the caller.  NVML, in
    contrast, normally enumerates physical devices and does not apply
    ``CUDA_VISIBLE_DEVICES``.  Resolving the handle explicitly is therefore
    necessary when (for example) physical GPU 1 is exposed to torch as
    ``cuda:0``.
    """

    def __init__(self, device_index: int = 0, interval_s: float = 0.02):
        self.device_index, self.interval_s = device_index, interval_s
        self.samples: list[tuple[float, float]] = []
        self.temperature_samples_c: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _read_temperature_c(self) -> float:
        try:
            return float(
                self.pynvml.nvmlDeviceGetTemperature(
                    self.handle, self.pynvml.NVML_TEMPERATURE_GPU,
                )
            )
        except Exception:
            return float("nan")

    def __enter__(self) -> "NVMLPowerSampler":
        try:
            import pynvml  # type: ignore
        except ImportError as exc:
            raise StudyGateError("Install nvidia-ml-py for GPU energy measurement") from exc
        self.pynvml = pynvml
        pynvml.nvmlInit()
        try:
            self.handle, self.device_resolution = _resolve_nvml_device_handle(
                pynvml, self.device_index,
            )
        except Exception:
            pynvml.nvmlShutdown()
            raise
        self.samples.append((time.perf_counter(), pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000))
        self.temperature_samples_c.append(self._read_temperature_c())
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self) -> None:
        while not self._stop.is_set():
            timestamp = time.perf_counter()
            power_w = self.pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000
            self.samples.append((timestamp, power_w))
            self.temperature_samples_c.append(self._read_temperature_c())
            self._stop.wait(self.interval_s)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.samples.append((time.perf_counter(), self.pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000))
        self.temperature_samples_c.append(self._read_temperature_c())
        self.pynvml.nvmlShutdown()

    @property
    def energy_j(self) -> float:
        if len(self.samples) < 2:
            return float("nan")
        times, powers = zip(*self.samples)
        return float(np.trapz(powers, times))

    @property
    def peak_temperature_c(self) -> float:
        values = np.asarray(self.temperature_samples_c, dtype=float)
        finite = values[np.isfinite(values)]
        return float(finite.max()) if len(finite) else float("nan")


def _normalized_device_identifier(value: Any) -> str:
    """Return a stable text form for CUDA/NVML UUID and PCI identifiers."""

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="strict")
    return str(value).strip()


def _torch_device_identifiers(logical_device_index: int) -> tuple[str, str]:
    """Best-effort UUID/PCI identity for a logical torch CUDA device."""

    try:
        import torch  # type: ignore
    except ImportError:
        return "", ""
    try:
        if not torch.cuda.is_available() or logical_device_index >= torch.cuda.device_count():
            return "", ""
        properties = torch.cuda.get_device_properties(logical_device_index)
    except Exception:
        return "", ""
    uuid = _normalized_device_identifier(getattr(properties, "uuid", ""))
    pci_bus_id = _normalized_device_identifier(
        getattr(properties, "pci_bus_id", getattr(properties, "pciBusID", ""))
    )
    return uuid, pci_bus_id


def _resolve_nvml_uuid_handle(pynvml: Any, identifier: str) -> Any:
    """Resolve a complete or unambiguous abbreviated NVML UUID."""

    try:
        return pynvml.nvmlDeviceGetHandleByUUID(identifier)
    except Exception as direct_error:
        matches: list[Any] = []
        try:
            count = int(pynvml.nvmlDeviceGetCount())
            for index in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                observed = _normalized_device_identifier(
                    pynvml.nvmlDeviceGetUUID(handle),
                )
                if observed.lower().startswith(identifier.lower()):
                    matches.append(handle)
        except Exception:
            raise StudyGateError(
                f"Unable to resolve CUDA device UUID {identifier!r} through NVML"
            ) from direct_error
        if len(matches) != 1:
            raise StudyGateError(
                f"CUDA device UUID {identifier!r} did not identify exactly one NVML device"
            ) from direct_error
        return matches[0]


def _resolve_nvml_device_handle(
    pynvml: Any,
    logical_device_index: int,
) -> tuple[Any, str]:
    """Map a logical torch CUDA index to its physical NVML device handle.

    UUID or PCI identity reported by torch is authoritative when available.
    Otherwise the function interprets ``CUDA_VISIBLE_DEVICES`` using CUDA's
    logical ordering rules.  With no visibility override, the logical and
    NVML indices are identical, preserving the historical behavior.
    """

    if isinstance(logical_device_index, bool) or int(logical_device_index) < 0:
        raise ValueError("logical CUDA device index must be a nonnegative integer")
    logical_device_index = int(logical_device_index)

    torch_uuid, torch_pci_bus_id = _torch_device_identifiers(logical_device_index)
    if torch_uuid:
        return _resolve_nvml_uuid_handle(pynvml, torch_uuid), f"torch_uuid:{torch_uuid}"
    if torch_pci_bus_id and hasattr(pynvml, "nvmlDeviceGetHandleByPciBusId"):
        try:
            return (
                pynvml.nvmlDeviceGetHandleByPciBusId(torch_pci_bus_id),
                f"torch_pci_bus_id:{torch_pci_bus_id}",
            )
        except Exception as exc:
            raise StudyGateError(
                f"Unable to resolve torch PCI bus ID {torch_pci_bus_id!r} through NVML"
            ) from exc

    visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visibility is not None:
        selectors = [item.strip() for item in visibility.split(",") if item.strip()]
        if logical_device_index >= len(selectors):
            raise StudyGateError(
                f"Logical CUDA device {logical_device_index} is outside CUDA_VISIBLE_DEVICES"
            )
        selector = selectors[logical_device_index]
        if selector.isdigit():
            physical_index = int(selector)
            return (
                pynvml.nvmlDeviceGetHandleByIndex(physical_index),
                f"cuda_visible_devices_index:{physical_index}",
            )
        if selector.upper().startswith(("GPU-", "MIG-")):
            return (
                _resolve_nvml_uuid_handle(pynvml, selector),
                f"cuda_visible_devices_uuid:{selector}",
            )
        raise StudyGateError(
            f"Unsupported CUDA_VISIBLE_DEVICES selector {selector!r} for energy metering"
        )

    return (
        pynvml.nvmlDeviceGetHandleByIndex(logical_device_index),
        f"nvml_index:{logical_device_index}",
    )


def profile_callable(function: Callable[[], Any], device_index: int = 0, measure_energy: bool = True) -> tuple[Any, dict[str, float]]:
    start = time.perf_counter()
    if measure_energy:
        with NVMLPowerSampler(device_index) as meter:
            output = function()
        energy = meter.energy_j
        peak_temperature_c = meter.peak_temperature_c
    else:
        output, energy, peak_temperature_c = function(), float("nan"), float("nan")
    elapsed_ms = (time.perf_counter() - start) * 1000
    return output, {
        "latency_ms": elapsed_ms,
        "energy_j": energy,
        "peak_temperature_c": peak_temperature_c,
    }


def tegrastats_command(interval_ms: int = 100) -> list[str]:
    if interval_ms < 20:
        raise ValueError("tegrastats sampling below 20 ms is not supported")
    return ["tegrastats", "--interval", str(interval_ms)]


def netem_command_plan(
    interface: str,
    rate_mbps: float,
    one_way_delay_ms: float,
    jitter_ms: float,
    loss_percent: float,
    queue_packets: int,
    seed: int,
) -> list[list[str]]:
    """Return commands for manual review; never executes privileged traffic shaping."""
    if not interface or any(character.isspace() for character in interface):
        raise ValueError("Invalid interface name")
    if rate_mbps <= 0 or one_way_delay_ms < 0 or jitter_ms < 0 or not 0 <= loss_percent <= 100:
        raise ValueError("Invalid netem parameters")
    return [
        ["sudo", "tc", "qdisc", "replace", "dev", interface, "root", "handle", "1:", "tbf", "rate", f"{rate_mbps}mbit", "burst", "32kbit", "latency", "400ms"],
        ["sudo", "tc", "qdisc", "replace", "dev", interface, "parent", "1:1", "handle", "10:", "netem", "limit", str(queue_packets), "delay", f"{one_way_delay_ms}ms", f"{jitter_ms}ms", "loss", f"{loss_percent}%", "seed", str(seed)],
        ["sudo", "tc", "-s", "qdisc", "show", "dev", interface],
    ]


def netem_cleanup_command(interface: str) -> list[str]:
    return ["sudo", "tc", "qdisc", "del", "dev", interface, "root"]


def render_command(command: list[str]) -> str:
    import shlex
    return shlex.join(command)


EMBEDDED_STAGE_CHAIN = (
    "capture", "detector_done", "representation_ready", "enqueue",
    "tx_done", "receive", "decode_done", "decision",
)
EMBEDDED_STAGE_ROLES = (
    "sender", "sender", "sender", "sender", "sender",
    "receiver", "receiver", "receiver",
)
EMBEDDED_RAW_TRIAL_REQUIRED_COLUMNS = {
    "trial_id", "sample_id", "visual_group_id", "mission_profile", "cohort",
    "trace_id", "trace_family", "replicate_seed", "policy", "event",
    "stage_index", "monotonic_ns", "clock_domain_id", "process_role",
    "link_profile_id", "hardware_id", "platform_class", "absolute_deadline_ns",
    "gold_action", "receiver_action", "critical", "payload_valid", "state_valid",
    "selected_mode", "semantic_prediction", "calibrated_correctness_probability",
    "stage_compute_energy_j", "stage_radio_energy_j", "wire_bytes_delta",
    "trial_total_energy_j", "trial_total_wire_bytes", "memory_mb", "temperature_c",
    "protocol_hash", "target_log_sha256", "sender_revision", "receiver_revision",
    "analysis_plan_version",
}
EMBEDDED_TARGET_LOG_REQUIRED_COLUMNS = (
    EMBEDDED_RAW_TRIAL_REQUIRED_COLUMNS - {"target_log_sha256"}
)
_EMBEDDED_CHAIN_KEY = ("trial_id", "policy")
_EMBEDDED_PAIR_KEY = (
    "sample_id", "visual_group_id", "mission_profile", "trace_id", "replicate_seed",
)
_EMBEDDED_CONSTANT_COLUMNS = (
    "sample_id", "visual_group_id", "mission_profile", "cohort", "trace_id",
    "trace_family", "replicate_seed", "policy", "clock_domain_id", "link_profile_id",
    "hardware_id", "platform_class", "absolute_deadline_ns", "gold_action",
    "receiver_action", "critical", "payload_valid", "state_valid",
    "selected_mode", "semantic_prediction", "calibrated_correctness_probability",
    "trial_total_energy_j", "trial_total_wire_bytes", "protocol_hash",
    "target_log_sha256", "sender_revision", "receiver_revision", "analysis_plan_version",
)


def _strict_boolean(series: pd.Series, column: str) -> pd.Series:
    mapping = {
        True: True, False: False, 1: True, 0: False,
        "true": True, "false": False, "1": True, "0": False,
    }
    normalized = series.map(lambda value: value.strip().lower() if isinstance(value, str) else value)
    converted = normalized.map(mapping)
    if converted.isna().any():
        raise StudyGateError(f"Embedded evidence column {column!r} must contain strict booleans")
    return converted.astype(bool)


def _read_embedded_trial_artifact(path: Path) -> pd.DataFrame:
    lower_name = path.name.lower()
    if lower_name.endswith(".csv") or lower_name.endswith(".csv.gz"):
        return pd.read_csv(path)
    if lower_name.endswith(".jsonl") or lower_name.endswith(".ndjson"):
        return pd.read_json(path, lines=True)
    raise StudyGateError(
        "Embedded raw trials must be CSV, CSV.GZ, JSONL, or NDJSON"
    )


def _validate_embedded_stage_chains(
    frame: pd.DataFrame,
    *,
    expected_hardware_id: str | None = None,
    expected_platform_class: str | None = None,
    expected_protocol_hash: str | None = None,
    expected_target_log_sha256: str | None = None,
    expected_analysis_plan_version: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate event-level evidence and derive one immutable row per paired trial.

    `latency_ms`, endpoint outcomes, and resource totals are never accepted from
    a summary.  They are derived here from the ordered raw stage timestamps and
    per-stage resource deltas.
    """
    if frame.empty:
        raise StudyGateError("Embedded raw trial artifact is empty")
    required_columns = (
        EMBEDDED_RAW_TRIAL_REQUIRED_COLUMNS
        if expected_target_log_sha256 is not None
        else EMBEDDED_TARGET_LOG_REQUIRED_COLUMNS
    )
    if missing := required_columns.difference(frame.columns):
        raise StudyGateError(f"Embedded raw trials are missing columns: {sorted(missing)}")
    data = frame.copy()
    string_columns = {
        "trial_id", "sample_id", "visual_group_id", "mission_profile", "cohort",
        "trace_id", "trace_family", "policy", "event", "clock_domain_id",
        "process_role", "link_profile_id", "hardware_id", "platform_class",
        "gold_action", "receiver_action", "selected_mode", "protocol_hash", "target_log_sha256",
        "sender_revision", "receiver_revision", "analysis_plan_version",
    }
    for column in string_columns.intersection(data.columns):
        values = data[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise StudyGateError(f"Embedded raw trial column {column!r} cannot be blank")
        data[column] = values.astype(str)
    for column in ("critical", "payload_valid", "state_valid", "semantic_prediction"):
        data[column] = _strict_boolean(data[column], column)
    integer_columns = (
        "replicate_seed", "stage_index", "monotonic_ns", "absolute_deadline_ns",
        "wire_bytes_delta", "trial_total_wire_bytes",
    )
    for column in integer_columns:
        numeric = pd.to_numeric(data[column], errors="coerce")
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise StudyGateError(f"Embedded raw trial column {column!r} must contain finite integers")
        data[column] = numeric.astype(np.int64)
    numeric_columns = (
        "stage_compute_energy_j", "stage_radio_energy_j", "trial_total_energy_j",
        "memory_mb", "temperature_c",
    )
    for column in numeric_columns:
        numeric = pd.to_numeric(data[column], errors="coerce")
        if not np.isfinite(numeric).all():
            raise StudyGateError(f"Embedded raw trial column {column!r} must be finite")
        data[column] = numeric.astype(float)
    confidence = pd.to_numeric(data["calibrated_correctness_probability"], errors="coerce")
    nonmissing_confidence = confidence.notna()
    if not np.isfinite(confidence.loc[nonmissing_confidence]).all():
        raise StudyGateError("Embedded calibrated correctness probabilities must be finite when present")
    finite_confidence = nonmissing_confidence
    if not confidence.loc[finite_confidence].between(0, 1).all():
        raise StudyGateError("Embedded calibrated correctness probabilities must lie in [0,1]")
    data["calibrated_correctness_probability"] = confidence.astype(float)
    nonnegative = (
        "stage_compute_energy_j", "stage_radio_energy_j", "wire_bytes_delta",
        "trial_total_energy_j", "trial_total_wire_bytes", "memory_mb",
    )
    if any((data[column] < 0).any() for column in nonnegative):
        raise StudyGateError("Embedded raw resource measurements cannot be negative")
    if not set(data["gold_action"]).issubset({"NO_ACTION", "REVIEW", "RESPOND"}):
        raise StudyGateError("Embedded raw gold_action is outside the registered action ontology")
    if not set(data["receiver_action"]).issubset({"NO_ACTION", "REVIEW", "RESPOND"}):
        raise StudyGateError("Embedded raw receiver_action is outside the registered action ontology")
    semantic_modes = {"L1_ALERT", "L2_SEMANTICS"}
    if not set(data["selected_mode"]).issubset({
        "L0_HEARTBEAT", "L1_ALERT", "L2_SEMANTICS", "L3_SEMANTICS_ROI",
        "L4_FEATURES", "L5_IMAGE_OR_CLIP",
    }):
        raise StudyGateError("Embedded raw selected_mode is outside the registered representation ontology")
    semantic_from_mode = data["selected_mode"].isin(semantic_modes)
    if not semantic_from_mode.eq(data["semantic_prediction"]).all():
        raise StudyGateError("Embedded semantic_prediction disagrees with selected_mode")
    if not finite_confidence.loc[semantic_from_mode].all():
        raise StudyGateError("Embedded semantic modes require a finite calibrated correctness score")

    singleton_expectations = {
        "hardware_id": expected_hardware_id,
        "platform_class": expected_platform_class,
        "protocol_hash": expected_protocol_hash,
        "target_log_sha256": expected_target_log_sha256,
        "analysis_plan_version": expected_analysis_plan_version,
    }
    for column, expected in singleton_expectations.items():
        if column not in data and expected is None:
            continue
        if data[column].nunique(dropna=False) != 1:
            raise StudyGateError(f"Embedded raw trials must contain exactly one {column}")
        if expected is not None and str(data[column].iloc[0]) != str(expected):
            raise StudyGateError(
                f"Embedded raw trial {column} {str(data[column].iloc[0])!r} "
                f"differs from registered {str(expected)!r}"
            )
    for column in ("clock_domain_id", "sender_revision", "receiver_revision"):
        if data[column].nunique(dropna=False) != 1:
            raise StudyGateError(f"Embedded raw trials must contain exactly one {column}")

    derived_rows: list[dict[str, Any]] = []
    latency_by_chain: dict[tuple[str, str], float] = {}
    for key, group in data.groupby(list(_EMBEDDED_CHAIN_KEY), sort=False, dropna=False):
        ordered = group.sort_values("stage_index", kind="stable")
        if len(ordered) != len(EMBEDDED_STAGE_CHAIN):
            raise StudyGateError(f"Embedded trial {key!r} does not have exactly eight stages")
        if ordered["stage_index"].tolist() != list(range(len(EMBEDDED_STAGE_CHAIN))):
            raise StudyGateError(f"Embedded trial {key!r} has invalid or duplicate stage_index values")
        if tuple(ordered["event"]) != EMBEDDED_STAGE_CHAIN:
            raise StudyGateError(f"Embedded trial {key!r} does not follow the registered stage chain")
        if tuple(ordered["process_role"]) != EMBEDDED_STAGE_ROLES:
            raise StudyGateError(f"Embedded trial {key!r} has the wrong sender/receiver stage roles")
        timestamps = ordered["monotonic_ns"].to_numpy(np.int64)
        if not np.all(np.diff(timestamps) > 0):
            raise StudyGateError(f"Embedded trial {key!r} timestamps are not strictly increasing")
        for column in _EMBEDDED_CONSTANT_COLUMNS:
            if column not in ordered:
                continue
            if ordered[column].nunique(dropna=False) != 1:
                raise StudyGateError(f"Embedded trial {key!r} changes trial-level column {column!r}")
        if (ordered.loc[~ordered["event"].eq("tx_done"), "wire_bytes_delta"] != 0).any():
            raise StudyGateError(f"Embedded trial {key!r} charges uplink wire bytes outside tx_done")
        measured_compute = float(ordered["stage_compute_energy_j"].sum())
        measured_radio = float(ordered["stage_radio_energy_j"].sum())
        measured_total = measured_compute + measured_radio
        declared_total = float(ordered["trial_total_energy_j"].iloc[0])
        measured_wire = int(ordered["wire_bytes_delta"].sum())
        declared_wire = int(ordered["trial_total_wire_bytes"].iloc[0])
        if measured_compute <= 0:
            raise StudyGateError(f"Embedded trial {key!r} has no measured compute energy")
        if measured_wire > 0 and measured_radio <= 0:
            raise StudyGateError(f"Embedded trial {key!r} transmits bytes without measured radio energy")
        if not np.isclose(declared_total, measured_total, rtol=1e-5, atol=1e-8):
            raise StudyGateError(f"Embedded trial {key!r} energy total disagrees with stage deltas")
        if declared_wire != measured_wire:
            raise StudyGateError(f"Embedded trial {key!r} wire total disagrees with stage deltas")
        capture_ns, decision_ns = int(timestamps[0]), int(timestamps[-1])
        deadline_ns = int(ordered["absolute_deadline_ns"].iloc[0])
        if deadline_ns <= capture_ns:
            raise StudyGateError(f"Embedded trial {key!r} deadline is not after capture")
        latency_ms = (decision_ns - capture_ns) / 1e6
        latency_by_chain[(str(key[0]), str(key[1]))] = latency_ms
        decision = ordered.iloc[-1].to_dict()
        decision.update({
            "capture_ns": capture_ns,
            "decision_ns": decision_ns,
            "latency_ms": float(latency_ms),
            "capture_us": capture_ns // 1000,
            "action_ready_us": decision_ns // 1000,
            "deadline_us": deadline_ns // 1000,
            "energy_j": float(measured_total),
            "compute_energy_j": float(measured_compute),
            "radio_energy_j": float(measured_radio),
            "total_wire_bytes": int(measured_wire),
        })
        delivered = bool(
            decision["payload_valid"] and decision["state_valid"] and decision_ns <= deadline_ns
        )
        correct = str(decision["receiver_action"]) == str(decision["gold_action"])
        decision["delivered_by_deadline"] = delivered
        decision["correct_action"] = correct
        decision["correct_timely_action"] = bool(delivered and correct)
        decision["false_alert"] = bool(
            delivered and decision["receiver_action"] == "RESPOND"
            and decision["gold_action"] != "RESPOND"
        )
        decision["critical_miss"] = bool(
            decision["gold_action"] == "RESPOND" and not decision["correct_timely_action"]
        )
        derived_rows.append(decision)

    trials = pd.DataFrame(derived_rows)
    if trials.duplicated(list(_EMBEDDED_CHAIN_KEY)).any():
        raise StudyGateError("Embedded raw trials contain duplicate trial/policy chains")
    paired_exogenous = [
        *_EMBEDDED_PAIR_KEY, "cohort", "trace_family", "clock_domain_id",
        "link_profile_id", "hardware_id", "platform_class", "absolute_deadline_ns",
        "gold_action", "critical", "protocol_hash", "sender_revision",
        "receiver_revision", "analysis_plan_version", "capture_ns",
    ]
    if "target_log_sha256" in trials:
        paired_exogenous.append("target_log_sha256")
    trial_identity = trials.groupby("trial_id", dropna=False)[paired_exogenous].nunique()
    if (trial_identity > 1).any().any():
        raise StudyGateError("Embedded trial_id maps to more than one paired trial identity")
    pair_to_trial = trials.groupby(list(_EMBEDDED_PAIR_KEY), dropna=False)["trial_id"].nunique()
    if not pair_to_trial.eq(1).all():
        raise StudyGateError("Embedded paired trial identity maps to multiple trial_id values")
    enriched = data.copy()
    if "latency_ms" in enriched:
        if "reported_latency_ms" in enriched:
            enriched = enriched.drop(columns="latency_ms")
        else:
            enriched = enriched.rename(columns={"latency_ms": "reported_latency_ms"})
    enriched["latency_ms"] = [
        latency_by_chain[(str(row.trial_id), str(row.policy))]
        for row in enriched.itertuples(index=False)
    ]
    return enriched, trials


def import_embedded_jsonl(
    path: str | Path,
    expected_hardware_id: str | None = None,
    expected_platform_class: str | None = None,
) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise StudyGateError(f"Embedded log not found: {path}")
    frame = pd.read_json(path, lines=True)
    enriched, _ = _validate_embedded_stage_chains(
        frame,
        expected_hardware_id=expected_hardware_id,
        expected_platform_class=expected_platform_class,
    )
    enriched["source_log"] = str(path)
    return enriched


def embedded_gate(frame: pd.DataFrame, deadline_ms: float, memory_limit_mb: float, thermal_limit_c: float) -> pd.DataFrame:
    """Evaluate target-hardware gates from raw stages, never reported latency/totals."""
    if deadline_ms <= 0 or memory_limit_mb <= 0:
        raise ValueError("Embedded deadline and memory limit must be positive")
    enriched, trials = _validate_embedded_stage_chains(frame)
    latency = trials["latency_ms"].astype(float)
    memory = enriched["memory_mb"].astype(float)
    thermal = enriched["temperature_c"].astype(float)
    transmitted = trials["total_wire_bytes"].gt(0)
    radio_complete = bool(
        trials["radio_energy_j"].ge(0).all()
        and (trials.loc[transmitted, "radio_energy_j"].gt(0).all() if transmitted.any() else True)
    )
    checks = {
        "measured_target_hardware": (True, enriched["hardware_id"].nunique() == 1),
        "deadline": (True, float(latency.quantile(0.95)) <= deadline_ms),
        "memory": (True, float(memory.max()) <= memory_limit_mb),
        "thermal": (True, float(thermal.max()) <= thermal_limit_c),
        "energy_measured": (True, trials["compute_energy_j"].gt(0).all()),
        "radio_energy_measured": (True, radio_complete),
        "complete_stage_breakdown": (True, True),
        "ordered_stage_chains": (True, True),
        "energy_accounting_consistent": (True, True),
        "controlled_link_profile": (True, enriched["link_profile_id"].nunique() >= 1),
        "separate_sender_receiver": (
            True, {"sender", "receiver"}.issubset(set(enriched["process_role"])),
        ),
        "wire_bytes_measured": (True, trials["total_wire_bytes"].ge(0).all()),
    }
    return pd.DataFrame([
        {"gate": gate, "evidence_complete": bool(complete), "passed": bool(passed)}
        for gate, (complete, passed) in checks.items()
    ])


EMBEDDED_RETEST_REQUIRED_COLUMNS = {
    "hypothesis", "comparison", "endpoint", "estimate", "ci_low", "ci_high",
    "adjusted_p", "margin", "paired_design_pass", "matched_budget_pass",
    "sample_gate_pass", "calibration_gate_pass", "false_alert_rate",
    "reference_false_alert_rate", "method_selective_risk", "reference_selective_risk",
    "method_selective_coverage", "reference_selective_coverage",
    "h4_matched_coverage", "h4_coverage_selection_rule", "h4_coverage_gate_pass",
    "protocol_hash", "target_log_sha256", "hardware_id", "sender_revision",
    "receiver_revision", "analysis_plan_version", "trial_artifact_path",
    "trial_artifact_sha256",
}
_EMBEDDED_RETEST_BOOLEAN_COLUMNS = (
    "paired_design_pass",
    "matched_budget_pass",
    "sample_gate_pass",
    "calibration_gate_pass",
)


def import_embedded_retest_summary(path: str | Path) -> pd.DataFrame:
    """Load the registered target-hardware H2--H4 retest summary."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise StudyGateError(f"Embedded retest summary not found: {path}")
    lower_name = path.name.lower()
    if lower_name.endswith(".csv") or lower_name.endswith(".csv.gz"):
        frame = pd.read_csv(path)
    elif lower_name.endswith(".jsonl") or lower_name.endswith(".ndjson"):
        frame = pd.read_json(path, lines=True)
    else:
        raise StudyGateError("Embedded retest summary must be CSV, CSV.GZ, JSONL, or NDJSON")
    if frame.empty:
        raise StudyGateError("Embedded retest summary is empty")
    if missing := EMBEDDED_RETEST_REQUIRED_COLUMNS.difference(frame.columns):
        raise StudyGateError(f"Embedded retest summary is missing columns: {sorted(missing)}")
    frame.attrs["source_path"] = str(path)
    return frame


def _group_equal_boolean_rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return float("nan")
    return float(frame.groupby("visual_group_id")[column].mean().mean())


def _assert_summary_close(row: pd.Series, column: str, expected: float) -> None:
    observed = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    if np.isnan(expected):
        if not pd.isna(observed):
            raise StudyGateError(
                f"Embedded retest reported {column} but raw evidence has no estimand"
            )
        return
    if not np.isfinite(observed) or not np.isclose(float(observed), expected, rtol=1e-6, atol=1e-8):
        raise StudyGateError(
            f"Embedded retest {column}={observed!r} disagrees with raw recomputation {expected!r}"
        )


def evaluate_embedded_retest(
    frame: pd.DataFrame,
    *,
    required_h2_baselines: Sequence[str],
    primary_margin: float,
    false_alert_limit: float,
    protocol_hash: str,
    target_log_sha256: str,
    expected_hardware_id: str | None = None,
    alpha: float = 0.05,
    analysis_plan_version: str = "semuav.analysis.v1",
    wire_budget_bytes_per_s: float | None = None,
    energy_budget_j_per_s: float | None = None,
    sample_minima: Mapping[str, int] | None = None,
    registered_calibration_gate_pass: bool | None = None,
    bootstrap_repetitions: int = 10_000,
    bootstrap_seed: int = 20260823,
    h4_matched_coverage: float | None = None,
    h4_coverage_selection_rule: str | None = None,
) -> pd.DataFrame:
    """Recompute target-hardware H2--H4 support from hashed raw stage trials.

    The supplied summary is only a provenance/reconciliation document.  Its
    effect estimates, confidence intervals, p-values, and pass flags never drive
    a claim.  Every endpoint and resource total is reconstructed from the raw
    ordered event chains, then compared with the summary to detect corruption.
    """
    if frame.empty or (missing := EMBEDDED_RETEST_REQUIRED_COLUMNS.difference(frame.columns)):
        raise StudyGateError(
            "Embedded retest summary is empty or incomplete"
            + (f": {sorted(missing)}" if frame.empty is False and missing else "")
        )
    if (
        not 0 < alpha < 1 or primary_margin < 0 or not 0 <= false_alert_limit <= 1
        or bootstrap_repetitions < 100
    ):
        raise ValueError("Invalid embedded retest decision thresholds")
    data = frame.copy()
    for column in _EMBEDDED_RETEST_BOOLEAN_COLUMNS:
        data[column] = _strict_boolean(data[column], column)
    data["h4_coverage_gate_pass"] = _strict_boolean(
        data["h4_coverage_gate_pass"], "h4_coverage_gate_pass",
    )
    for column in ("estimate", "ci_low", "ci_high", "adjusted_p", "margin"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if not np.isfinite(data[column]).all():
            raise StudyGateError(f"Embedded retest column {column!r} must be finite")
    if (data["ci_low"] > data["ci_high"]).any():
        raise StudyGateError("Embedded retest confidence interval has ci_low > ci_high")
    frozen_h4_rule = "top_calibrated_semantic_score_within_visual_group"
    h4_inputs_available = bool(
        h4_matched_coverage is not None
        and 0 < float(h4_matched_coverage) <= 1
        and h4_coverage_selection_rule == frozen_h4_rule
    )
    reported_h4_coverage = pd.to_numeric(data["h4_matched_coverage"], errors="coerce")
    if h4_inputs_available:
        if not np.isclose(reported_h4_coverage.to_numpy(float), float(h4_matched_coverage)).all():
            raise StudyGateError("Embedded retest H4 matched coverage differs from the frozen protocol")
        if not data["h4_coverage_selection_rule"].astype(str).eq(frozen_h4_rule).all():
            raise StudyGateError("Embedded retest H4 selection rule differs from the frozen protocol")
    elif data["h4_coverage_gate_pass"].any():
        raise StudyGateError(
            "Embedded H4 cannot claim a coverage pass without the frozen matched-coverage inputs"
        )
    if set(data["hypothesis"].astype(str)) != {"H2", "H3", "H4"}:
        raise StudyGateError("Embedded retest must contain exactly hypotheses H2, H3, and H4")
    if data.duplicated(["hypothesis", "comparison"]).any():
        raise StudyGateError("Embedded retest contains duplicate hypothesis/comparison rows")

    expected_provenance = {
        "protocol_hash": protocol_hash,
        "target_log_sha256": target_log_sha256,
        "analysis_plan_version": analysis_plan_version,
    }
    for column, expected in expected_provenance.items():
        if not data[column].astype(str).eq(str(expected)).all():
            raise StudyGateError(f"Embedded retest {column} does not match the registered handoff")
    for column in ("hardware_id", "sender_revision", "receiver_revision"):
        values = data[column].dropna().astype(str).str.strip()
        if len(values) != len(data) or values.eq("").any() or values.nunique() != 1:
            raise StudyGateError(f"Embedded retest must have one nonempty {column}")
    if expected_hardware_id is not None:
        observed_hardware_id = str(data["hardware_id"].iloc[0]).strip()
        if observed_hardware_id != str(expected_hardware_id):
            raise StudyGateError(
                f"Embedded retest hardware_id {observed_hardware_id!r} differs from "
                f"the registered target log {expected_hardware_id!r}"
            )
    artifact_rows = data[["trial_artifact_path", "trial_artifact_sha256"]].drop_duplicates()
    if len(artifact_rows) != 1:
        raise StudyGateError("Embedded retest must reference exactly one immutable raw trial artifact")
    artifact_row = artifact_rows.iloc[0]
    artifact = Path(str(artifact_row["trial_artifact_path"])).expanduser()
    if not artifact.is_absolute() and frame.attrs.get("source_path"):
        artifact = Path(str(frame.attrs["source_path"])).parent / artifact
    artifact = artifact.resolve()
    expected_hash = str(artifact_row["trial_artifact_sha256"]).strip().lower()
    if not artifact.is_file() or len(expected_hash) != 64 or hash_file(artifact) != expected_hash:
        raise StudyGateError(f"Embedded retest trial artifact is missing or changed: {artifact}")

    h2 = data[data["hypothesis"].eq("H2")].copy()
    required_baselines = tuple(map(str, required_h2_baselines))
    if set(h2["comparison"].astype(str)) != set(required_baselines) or len(h2) != len(required_baselines):
        raise StudyGateError("Embedded H2 retest does not cover the exact frozen baseline family")
    expected_singletons = {
        "H3": ("proposed_full_vs_channel_only_hysteresis", "critical_miss_reduction"),
        "H4": ("proposed_full_vs_ablate_evidence_escalation", "high_confidence_semantic_error_reduction"),
    }
    for hypothesis, (comparison, endpoint) in expected_singletons.items():
        rows = data[data["hypothesis"].eq(hypothesis)]
        if len(rows) != 1 or str(rows.iloc[0]["comparison"]) != comparison or str(rows.iloc[0]["endpoint"]) != endpoint:
            raise StudyGateError(f"Embedded {hypothesis} retest has the wrong comparison or endpoint")
    if not h2["endpoint"].astype(str).eq("correct_timely_action").all():
        raise StudyGateError("Embedded H2 retest endpoint must be correct_timely_action")
    if not np.isclose(h2["margin"].to_numpy(float), primary_margin).all():
        raise StudyGateError("Embedded H2 retest does not use the frozen primary margin")
    if not np.isclose(
        data.loc[data["hypothesis"].isin(["H3", "H4"]), "margin"].to_numpy(float), 0.0
    ).all():
        raise StudyGateError("Embedded H3/H4 retest margins must be zero")

    raw = _read_embedded_trial_artifact(artifact)
    raw_stages, trials = _validate_embedded_stage_chains(
        raw,
        expected_hardware_id=(expected_hardware_id or str(data["hardware_id"].iloc[0])),
        expected_platform_class=None,
        expected_protocol_hash=protocol_hash,
        expected_target_log_sha256=target_log_sha256,
        expected_analysis_plan_version=analysis_plan_version,
    )
    for column in ("hardware_id", "sender_revision", "receiver_revision"):
        if str(raw_stages[column].iloc[0]) != str(data[column].iloc[0]).strip():
            raise StudyGateError(f"Embedded summary {column} disagrees with the raw trial artifact")

    expected_policies = {
        "proposed_full", *required_baselines,
        "channel_only_hysteresis", "ablate_evidence_escalation",
    }
    if set(trials["policy"].astype(str)) != expected_policies:
        raise StudyGateError("Embedded raw trials do not cover the exact frozen policy family")
    assert_paired_design(trials)

    default_minima = {
        "natural_test_units": 500,
        "safety_critical_positive_units": 200,
        "safety_noncritical_units": 500,
        "ood_units": 150,
        "held_out_trace_realizations": 30,
        "held_out_trace_families": 6,
    }
    minima = {**default_minima, **({str(k): int(v) for k, v in sample_minima.items()} if sample_minima else {})}
    if any(value < 0 for value in minima.values()):
        raise ValueError("Embedded registered sample minima cannot be negative")
    unique_trials = trials.drop_duplicates("trial_id")
    natural = unique_trials[unique_trials["cohort"].eq("natural_test")]
    safety = unique_trials[unique_trials["cohort"].eq("safety")]
    ood = unique_trials[unique_trials["cohort"].eq("ood")]
    coverage_observed = {
        "natural_test_units": int(natural["visual_group_id"].nunique()),
        "safety_critical_positive_units": int(
            safety.loc[safety["critical"] & safety["gold_action"].eq("RESPOND"), "visual_group_id"].nunique()
        ),
        "safety_noncritical_units": int(
            safety.loc[~safety["critical"], "visual_group_id"].nunique()
        ),
        "ood_units": int(ood["visual_group_id"].nunique()),
        "held_out_trace_realizations": int(unique_trials["trace_id"].nunique()),
        "held_out_trace_families": int(unique_trials["trace_family"].nunique()),
    }
    sample_gate_pass = all(coverage_observed[name] >= minima[name] for name in default_minima)

    budgets_available = bool(
        wire_budget_bytes_per_s is not None and energy_budget_j_per_s is not None
        and float(wire_budget_bytes_per_s) > 0 and float(energy_budget_j_per_s) > 0
    )
    if budgets_available:
        budget = budget_compliance(
            trials,
            float(wire_budget_bytes_per_s),
            float(energy_budget_j_per_s),
            tolerance=0.05,
        ).set_index("policy")
        budget_eligible = {policy: bool(budget.loc[policy, "eligible"]) for policy in expected_policies}
    else:
        budget_eligible = {policy: False for policy in expected_policies}
    calibration_gate_pass = registered_calibration_gate_pass is True
    if not h4_inputs_available:
        raise StudyGateError(
            "Embedded H4 recomputation requires the frozen matched coverage and selection rule"
        )

    family_confidence = 1.0 - alpha / max(1, len(required_baselines))
    h2_effects = {
        baseline: two_way_clustered_paired_bootstrap(
            trials, "proposed_full", baseline,
            endpoint="correct_timely_action", repetitions=bootstrap_repetitions,
            seed=bootstrap_seed, confidence=family_confidence,
        )
        for baseline in required_baselines
    }
    h2_adjusted = holm_adjust({baseline: effect.p_two_sided for baseline, effect in h2_effects.items()})

    critical_pair = trials[
        trials["critical"]
        & trials["policy"].isin(["proposed_full", "channel_only_hysteresis"])
    ]
    if critical_pair.empty:
        raise StudyGateError("Embedded H3 raw trials contain no registered critical events")
    h3_effect = two_way_clustered_paired_bootstrap(
        critical_pair, "channel_only_hysteresis", "proposed_full",
        endpoint="critical_miss", repetitions=bootstrap_repetitions, seed=bootstrap_seed,
    )
    h4_pair, h4_coverage_audit = matched_coverage_semantic_errors(
        trials,
        ("proposed_full", "ablate_evidence_escalation"),
        float(h4_matched_coverage),
    )
    h4_coverage_pass = bool(h4_coverage_audit["passed"].astype(bool).all())
    if not h4_coverage_pass:
        raise StudyGateError(
            "Embedded H4 matched-coverage gate failed; empirical H4/H5 support is blocked"
        )
    h4_effect = two_way_clustered_paired_bootstrap(
        h4_pair, "ablate_evidence_escalation", "proposed_full",
        endpoint="matched_coverage_semantic_error", repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    secondary_adjusted = holm_adjust({
        "H3": h3_effect.p_two_sided,
        "H4": h4_effect.p_two_sided,
    })

    proposed_noncritical = trials[
        trials["policy"].eq("proposed_full") & ~trials["critical"]
    ]
    channel_noncritical = trials[
        trials["policy"].eq("channel_only_hysteresis") & ~trials["critical"]
    ]
    proposed_far = _group_equal_boolean_rate(proposed_noncritical, "false_alert")
    channel_far = _group_equal_boolean_rate(channel_noncritical, "false_alert")

    h4_audit = h4_coverage_audit.set_index("policy")
    proposed_risk = float(h4_audit.loc["proposed_full", "group_equal_selective_risk"])
    proposed_coverage = float(h4_audit.loc["proposed_full", "realized_group_equal_coverage"])
    ablation_risk = float(h4_audit.loc[
        "ablate_evidence_escalation", "group_equal_selective_risk",
    ])
    ablation_coverage = float(h4_audit.loc[
        "ablate_evidence_escalation", "realized_group_equal_coverage",
    ])
    h4_coverage_exactly_matched = bool(np.isclose(
        proposed_coverage, ablation_coverage, rtol=0.0, atol=1e-12,
    ))

    recomputed: dict[tuple[str, str], dict[str, float | bool]] = {}
    for baseline, effect in h2_effects.items():
        recomputed[("H2", baseline)] = {
            "estimate": effect.estimate, "ci_low": effect.ci_low, "ci_high": effect.ci_high,
            "adjusted_p": h2_adjusted[baseline],
            "matched_budget_pass": budget_eligible["proposed_full"] and budget_eligible[baseline],
        }
    recomputed[("H3", "proposed_full_vs_channel_only_hysteresis")] = {
        "estimate": h3_effect.estimate, "ci_low": h3_effect.ci_low, "ci_high": h3_effect.ci_high,
        "adjusted_p": secondary_adjusted["H3"], "false_alert_rate": proposed_far,
        "reference_false_alert_rate": channel_far,
        "matched_budget_pass": budget_eligible["proposed_full"] and budget_eligible["channel_only_hysteresis"],
    }
    recomputed[("H4", "proposed_full_vs_ablate_evidence_escalation")] = {
        "estimate": h4_effect.estimate, "ci_low": h4_effect.ci_low, "ci_high": h4_effect.ci_high,
        "adjusted_p": secondary_adjusted["H4"], "method_selective_risk": proposed_risk,
        "reference_selective_risk": ablation_risk,
        "method_selective_coverage": proposed_coverage,
        "reference_selective_coverage": ablation_coverage,
        "h4_coverage_gate_pass": h4_coverage_pass and h4_coverage_exactly_matched,
        "matched_budget_pass": budget_eligible["proposed_full"] and budget_eligible["ablate_evidence_escalation"],
    }

    comparison_rows: list[dict[str, Any]] = []
    for _, summary_row in data.iterrows():
        key = (str(summary_row["hypothesis"]), str(summary_row["comparison"]))
        raw_result = recomputed[key]
        for column in ("estimate", "ci_low", "ci_high", "adjusted_p"):
            _assert_summary_close(summary_row, column, float(raw_result[column]))
        for column in (
            "false_alert_rate", "reference_false_alert_rate", "method_selective_risk",
            "reference_selective_risk", "method_selective_coverage", "reference_selective_coverage",
        ):
            if column in raw_result:
                _assert_summary_close(summary_row, column, float(raw_result[column]))
        expected_flags = {
            "paired_design_pass": True,
            "matched_budget_pass": bool(raw_result["matched_budget_pass"]),
            "sample_gate_pass": sample_gate_pass,
            "calibration_gate_pass": calibration_gate_pass,
        }
        if key[0] == "H4":
            expected_flags["h4_coverage_gate_pass"] = bool(
                raw_result["h4_coverage_gate_pass"]
            )
        for column, expected in expected_flags.items():
            if bool(summary_row[column]) != expected:
                raise StudyGateError(
                    f"Embedded retest {column} disagrees with raw/frozen recomputation for {key}"
                )
        supported = bool(all(expected_flags.values()))
        if key[0] == "H2":
            supported &= bool(float(raw_result["ci_low"]) > primary_margin and float(raw_result["adjusted_p"]) <= alpha)
        elif key[0] == "H3":
            supported &= bool(
                float(raw_result["ci_low"]) > 0
                and float(raw_result["adjusted_p"]) <= alpha
                and np.isfinite(proposed_far) and proposed_far <= false_alert_limit
                and np.isfinite(channel_far) and channel_far <= false_alert_limit
            )
        else:
            supported &= bool(
                float(raw_result["ci_low"]) > 0
                and float(raw_result["adjusted_p"]) <= alpha
                and np.isfinite(proposed_risk) and np.isfinite(ablation_risk)
                and proposed_risk < ablation_risk
                and proposed_coverage > 0 and ablation_coverage > 0
                and h4_coverage_exactly_matched
            )
        comparison_rows.append({
            "hypothesis": key[0], "comparison": key[1],
            "comparison_supported": supported,
            "prerequisite_evidence_complete": bool(all(expected_flags.values())),
            **raw_result,
        })

    comparisons = pd.DataFrame(comparison_rows)
    common_prerequisites = bool(
        sample_gate_pass and calibration_gate_pass and budgets_available
        and all(budget_eligible.values())
    )
    rows = []
    for hypothesis in ("H2", "H3", "H4"):
        group = comparisons[comparisons["hypothesis"].eq(hypothesis)]
        hypothesis_prerequisites = bool(
            common_prerequisites
            and (hypothesis != "H4" or (h4_coverage_pass and h4_coverage_exactly_matched))
        )
        rows.append({
            "hypothesis": hypothesis,
            "execution_status": (
                "EMBEDDED_RETEST_COMPLETE" if hypothesis_prerequisites
                else "BLOCKED_EMBEDDED_PREREQUISITE_EVIDENCE"
            ),
            "comparison_count": len(group),
            "claim_supported": bool(hypothesis_prerequisites and group["comparison_supported"].all()),
            "evidence_complete": hypothesis_prerequisites,
            "raw_recomputed": True,
            "source_kind": "hashed_raw_stage_trial_recomputation",
            "raw_trial_rows": int(len(raw_stages)),
            "raw_paired_trials": int(unique_trials["trial_id"].nunique()),
            "raw_visual_groups": int(unique_trials["visual_group_id"].nunique()),
            "raw_trace_realizations": coverage_observed["held_out_trace_realizations"],
            "raw_trace_families": coverage_observed["held_out_trace_families"],
            "h4_matched_coverage": float(h4_matched_coverage),
            "h4_coverage_selection_rule": frozen_h4_rule,
            "h4_coverage_gate_pass": bool(h4_coverage_pass and h4_coverage_exactly_matched),
            "hardware_id": str(data["hardware_id"].iloc[0]),
            "protocol_hash": protocol_hash,
            "target_log_sha256": target_log_sha256,
            "trial_artifact_sha256": expected_hash,
        })
    return pd.DataFrame(rows)


def evaluate_h5_status(
    original_hypothesis_status: pd.DataFrame,
    embedded_retest_status: pd.DataFrame,
    embedded_gates: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate H5 without conflating completed evidence with support."""
    original_required = {"hypothesis", "claim_supported"}
    retest_required = {
        "hypothesis", "claim_supported", "evidence_complete", "raw_recomputed",
        "source_kind", "trial_artifact_sha256",
    }
    if missing := original_required.difference(original_hypothesis_status.columns):
        raise StudyGateError(f"Original hypothesis status is missing columns: {sorted(missing)}")
    if missing := retest_required.difference(embedded_retest_status.columns):
        raise StudyGateError(f"Embedded retest status is missing columns: {sorted(missing)}")
    if not {"gate", "passed", "evidence_complete"}.issubset(embedded_gates.columns):
        raise StudyGateError("Embedded hardware gate table is incomplete")
    original = original_hypothesis_status[
        original_hypothesis_status["hypothesis"].isin(["H2", "H3", "H4"])
    ]
    retest = embedded_retest_status[
        embedded_retest_status["hypothesis"].isin(["H2", "H3", "H4"])
    ]
    if (
        len(original) != 3 or len(retest) != 3
        or set(original["hypothesis"]) != {"H2", "H3", "H4"}
        or set(retest["hypothesis"]) != {"H2", "H3", "H4"}
    ):
        raise StudyGateError("H5 requires exactly one original and embedded result for H2, H3, and H4")
    raw_recomputed = _strict_boolean(retest["raw_recomputed"], "embedded raw_recomputed").all()
    if not raw_recomputed or not retest["source_kind"].astype(str).eq(
        "hashed_raw_stage_trial_recomputation"
    ).all():
        raise StudyGateError("H5 requires H2--H4 statuses recomputed from hashed raw stage trials")
    artifact_hashes = retest["trial_artifact_sha256"].astype(str).str.strip().str.lower()
    if artifact_hashes.nunique() != 1 or not artifact_hashes.str.fullmatch(r"[0-9a-f]{64}").all():
        raise StudyGateError("H5 embedded retest status has invalid raw-trial provenance")
    original_supported = _strict_boolean(original["claim_supported"], "original claim_supported").all()
    retest_supported = _strict_boolean(retest["claim_supported"], "embedded claim_supported").all()
    retest_evidence_complete = _strict_boolean(
        retest["evidence_complete"], "embedded retest evidence_complete",
    ).all()
    hardware_evidence_complete = _strict_boolean(
        embedded_gates["evidence_complete"], "embedded evidence_complete",
    ).all()
    evidence_complete = bool(
        raw_recomputed and retest_evidence_complete and hardware_evidence_complete
    )
    hardware_passed = _strict_boolean(embedded_gates["passed"], "embedded gate passed").all()
    return pd.DataFrame([{
        "hypothesis": "H5",
        "execution_status": "EMPIRICAL_COMPLETE" if evidence_complete else "BLOCKED_INCOMPLETE_TARGET_EVIDENCE",
        "claim_supported": bool(evidence_complete and original_supported and retest_supported and hardware_passed),
        "embedded_evidence_complete": bool(evidence_complete),
        "embedded_raw_recomputed": bool(raw_recomputed),
        "original_h2_h4_supported": bool(original_supported),
        "embedded_h2_h4_supported": bool(retest_supported),
        "embedded_hardware_gates_passed": bool(hardware_passed),
    }])
