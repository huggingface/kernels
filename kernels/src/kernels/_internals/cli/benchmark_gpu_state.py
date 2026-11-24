import json
import shutil
import subprocess
from typing import Any

# Record fields any vendor can report. Vendor-specific values go into `hardware`.
CONFIG_FIELDS = ("driver_version", "gpu_max_clock_mhz", "gpu_power_limit_w")
STATE_FIELDS = ("gpu_clock_mhz", "gpu_mem_clock_mhz", "gpu_power_draw_w", "gpu_temperature_c", "gpu_throttled")

# nvidia-smi query field -> (record field, vendor-specific)
_NVIDIA_CONFIG = {
    "driver_version": ("driver_version", False),
    "clocks.max.sm": ("gpu_max_clock_mhz", False),
    "power.limit": ("gpu_power_limit_w", False),
    "clocks.max.mem": ("max_mem_clock_mhz", True),
    "persistence_mode": ("persistence_mode", True),
}
_NVIDIA_STATE = {
    "clocks.sm": "gpu_clock_mhz",
    "clocks.mem": "gpu_mem_clock_mhz",
    "power.draw": "gpu_power_draw_w",
    "temperature.gpu": "gpu_temperature_c",
}

# Older drivers use the `clocks_throttle_reasons` name.
_EVENT_REASON_FIELDS = ("clocks_event_reasons.active", "clocks_throttle_reasons.active")
_EVENT_REASONS = {
    0x1: "gpu_idle",
    0x2: "applications_clocks_setting",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal_slowdown",
    0x40: "hw_thermal_slowdown",
    0x80: "hw_power_brake_slowdown",
    0x100: "display_clock_setting",
}
_THROTTLE_REASONS = {
    "sw_power_cap",
    "hw_slowdown",
    "sw_thermal_slowdown",
    "hw_thermal_slowdown",
    "hw_power_brake_slowdown",
}


class GpuTelemetry:
    """Reads GPU settings and state. This base class is used when no vendor tool is available."""

    def read_config(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Settings fixed for a run, as (record fields, vendor-specific fields)."""
        return dict.fromkeys(CONFIG_FIELDS), {}

    def sample_state(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Current clocks, power, temperature and throttling, as (record fields, vendor-specific fields)."""
        return dict.fromkeys(STATE_FIELDS), {}


class NvidiaTelemetry(GpuTelemetry):
    def __init__(self, uuid: str):
        self.uuid = uuid
        # Which clock event reasons field the driver supports, found on first use.
        self._event_reason_field: str | None = None

    def read_config(self) -> tuple[dict[str, Any], dict[str, Any]]:
        core, hardware = super().read_config()
        values = self._query(list(_NVIDIA_CONFIG))
        if values is None:
            return core, hardware
        hardware = {"vendor": "nvidia"}
        for field, (name, vendor_specific) in _NVIDIA_CONFIG.items():
            (hardware if vendor_specific else core)[name] = _parse(values[field])
        core["driver_version"] = values["driver_version"] or None
        return core, hardware

    def sample_state(self) -> tuple[dict[str, Any], dict[str, Any]]:
        core, hardware = super().sample_state()
        values = self._query(list(_NVIDIA_STATE))
        if values is not None:
            for field, name in _NVIDIA_STATE.items():
                core[name] = _parse(values[field])
        mask = self._event_reasons()
        if mask is not None:
            reasons = [name for bit, name in _EVENT_REASONS.items() if mask & bit]
            hardware["clock_event_reasons"] = reasons
            core["gpu_throttled"] = any(r in _THROTTLE_REASONS for r in reasons)
        return core, hardware

    def _event_reasons(self) -> int | None:
        fields = (self._event_reason_field,) if self._event_reason_field else _EVENT_REASON_FIELDS
        for field in fields:
            values = self._query([field])
            if values is None:
                continue
            self._event_reason_field = field
            try:
                return int(values[field], 16)
            except ValueError:  # E.g. `[N/A]`
                return None
        return None

    def _query(self, fields: list[str]) -> dict[str, str] | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.uuid}",
                    f"--query-gpu={','.join(fields)}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        values = [v.strip() for v in result.stdout.strip().split(",")]
        if result.returncode != 0 or len(values) != len(fields):
            return None
        return dict(zip(fields, values))


class AmdTelemetry(GpuTelemetry):
    def __init__(self, pci_address: str):
        self.pci_address = pci_address

    def read_config(self) -> tuple[dict[str, Any], dict[str, Any]]:
        core, hardware = super().read_config()
        static, metric = _amd_smi("static", self.pci_address), _amd_smi("metric", self.pci_address)
        if static is None or metric is None:
            return core, hardware
        core["gpu_max_clock_mhz"] = _amd_value(metric.get("clock", {}).get("gfx_0", {}).get("max_clk"))
        asic = static.get("asic", {})
        hardware = {
            "vendor": "amd",
            "target_graphics_version": asic.get("target_graphics_version"),
            "num_compute_units": asic.get("num_compute_units"),
        }
        return core, hardware

    def sample_state(self) -> tuple[dict[str, Any], dict[str, Any]]:
        core, hardware = super().sample_state()
        metric = _amd_smi("metric", self.pci_address)
        if metric is None:
            return core, hardware
        clock, power = metric.get("clock", {}), metric.get("power", {})
        core["gpu_clock_mhz"] = _amd_value(clock.get("gfx_0", {}).get("clk"))
        core["gpu_mem_clock_mhz"] = _amd_value(clock.get("mem_0", {}).get("clk"))
        core["gpu_power_draw_w"] = _amd_value(power.get("socket_power"))
        core["gpu_temperature_c"] = _amd_value(metric.get("temperature", {}).get("hotspot"))
        if power.get("throttle_status") in ("THROTTLED", "UNTHROTTLED"):
            core["gpu_throttled"] = power["throttle_status"] == "THROTTLED"
        return core, {"clk_locked": clock.get("gfx_0", {}).get("clk_locked")}


def gpu_telemetry(backend_name: str) -> GpuTelemetry:
    """Telemetry for the current torch device, using the vendor's SMI tool."""
    import torch

    if not torch.cuda.is_available():
        return GpuTelemetry()
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if backend_name == "cuda" and shutil.which("nvidia-smi"):
        # The UUID identifies the device regardless of CUDA_VISIBLE_DEVICES.
        uuid = str(props.uuid)
        return NvidiaTelemetry(uuid if uuid.startswith("GPU-") else f"GPU-{uuid}")
    if backend_name == "rocm" and shutil.which("amd-smi"):
        # torch's UUID differs from amd-smi's on ROCm, the PCI address does not.
        domain, bus, dev = (getattr(props, f"pci_{part}_id", None) for part in ("domain", "bus", "device"))
        if domain is not None and bus is not None and dev is not None:
            return AmdTelemetry(f"{domain:04x}:{bus:02x}:{dev:02x}.0")
    return GpuTelemetry()


def _amd_smi(command: str, bdf: str) -> dict[str, Any] | None:
    try:
        result = subprocess.run(["amd-smi", command, "-g", bdf, "--json"], capture_output=True, text=True, timeout=10)
        return json.loads(result.stdout)["gpu_data"][0] if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, IndexError):
        return None


def _amd_value(value: Any) -> Any:
    # amd-smi reports values as {"value": 145, "unit": "W"}, or "N/A".
    return value.get("value") if isinstance(value, dict) else None


def _parse(value: str) -> Any:
    # Unsupported values look like `[N/A]` or `[Requested functionality has been deprecated]`.
    if value in ("", "N/A") or value.startswith("["):
        return None
    for convert in (int, float):
        try:
            return convert(value)
        except ValueError:
            pass
    return value
