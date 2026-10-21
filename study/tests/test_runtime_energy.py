from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import patch

from semuav.hardware import NVMLPowerSampler, _resolve_nvml_device_handle


class NVMLLogicalDeviceResolutionTests(unittest.TestCase):
    def test_cuda_visible_devices_numeric_selector_maps_logical_zero_to_physical_one(self) -> None:
        calls: list[int] = []
        fake_nvml = types.SimpleNamespace(
            nvmlDeviceGetHandleByIndex=lambda index: calls.append(index) or f"gpu-{index}",
        )
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            is_available=lambda: False,
        ))
        with patch.dict(sys.modules, {"torch": fake_torch}), patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False,
        ):
            handle, resolution = _resolve_nvml_device_handle(fake_nvml, 0)
        self.assertEqual(handle, "gpu-1")
        self.assertEqual(calls, [1])
        self.assertEqual(resolution, "cuda_visible_devices_index:1")

    def test_torch_uuid_is_preferred_over_nvml_index(self) -> None:
        calls: list[tuple[str, object]] = []
        fake_nvml = types.SimpleNamespace(
            nvmlDeviceGetHandleByUUID=lambda uuid: calls.append(("uuid", uuid)) or "uuid-handle",
            nvmlDeviceGetHandleByIndex=lambda index: calls.append(("index", index)) or "index-handle",
        )
        properties = types.SimpleNamespace(uuid="GPU-registered-device", pci_bus_id="0000:61:00.0")
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda index: properties,
        ))
        with patch.dict(sys.modules, {"torch": fake_torch}), patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False,
        ):
            handle, resolution = _resolve_nvml_device_handle(fake_nvml, 0)
        self.assertEqual(handle, "uuid-handle")
        self.assertEqual(calls, [("uuid", "GPU-registered-device")])
        self.assertEqual(resolution, "torch_uuid:GPU-registered-device")

    def test_sampler_uses_resolved_physical_handle(self) -> None:
        index_calls: list[int] = []
        fake_nvml = types.SimpleNamespace(
            NVML_TEMPERATURE_GPU=0,
            nvmlInit=lambda: None,
            nvmlShutdown=lambda: None,
            nvmlDeviceGetHandleByIndex=lambda index: index_calls.append(index) or f"gpu-{index}",
            nvmlDeviceGetPowerUsage=lambda handle: 30_000,
            nvmlDeviceGetTemperature=lambda handle, sensor: 40,
        )
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            is_available=lambda: False,
        ))
        with patch.dict(sys.modules, {"pynvml": fake_nvml, "torch": fake_torch}), patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False,
        ):
            with NVMLPowerSampler(device_index=0, interval_s=1.0) as sampler:
                self.assertEqual(sampler.handle, "gpu-1")
                self.assertEqual(
                    sampler.device_resolution,
                    "cuda_visible_devices_index:1",
                )
        self.assertEqual(index_calls, [1])


if __name__ == "__main__":
    unittest.main()
