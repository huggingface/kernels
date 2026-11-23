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


def gpu_config(device: tuple[str, str] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Settings fixed for a run, as (record fields, vendor-specific fields)."""
    core: dict[str, Any] = dict.fromkeys(CONFIG_FIELDS)
    if device and device[0] == "amd":
        return _amd_config(device[1], core)
    values = _query(device[1], list(_NVIDIA_CONFIG)) if device else None
    if values is None:
        return core, {}
    hardware: dict[str, Any] = {"vendor": "nvidia"}
    for field, (name, vendor_specific) in _NVIDIA_CONFIG.items():
        (hardware if vendor_specific else core)[name] = _parse(values[field])
    core["driver_version"] = values["driver_version"] or None
    return core, hardware


def gpu_state(device: tuple[str, str] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Current clocks, power, temperature and throttling, as (record fields, vendor-specific fields)."""
    core: dict[str, Any] = dict.fromkeys(STATE_FIELDS)
    hardware: dict[str, Any] = {}
    if not device:
        return core, hardware
    if device[0] == "amd":
        return _amd_state(device[1], core)
    device_id = device[1]
    values = _query(device_id, list(_NVIDIA_STATE))
    if values is not None:
        for field, name in _NVIDIA_STATE.items():
            core[name] = _parse(values[field])
    for field in _EVENT_REASON_FIELDS:
        values = _query(device_id, [field])
        if values is not None:
            try:
                mask = int(values[field], 16)
            except ValueError:  # E.g. `[N/A]`
                break
            reasons = [name for bit, name in _EVENT_REASONS.items() if mask & bit]
            hardware["clock_event_reasons"] = reasons
            core["gpu_throttled"] = any(r in _THROTTLE_REASONS for r in reasons)
            break
    return core, hardware


def gpu_device(backend_name: str) -> tuple[str, str] | None:
    """The current torch device as (vendor, id for the vendor's SMI tool)."""
    import torch

    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if backend_name == "cuda" and shutil.which("nvidia-smi"):
        # The UUID identifies the device regardless of CUDA_VISIBLE_DEVICES.
        uuid = str(props.uuid)
        return "nvidia", uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"
    if backend_name == "rocm" and shutil.which("amd-smi"):
        # torch's UUID differs from amd-smi's on ROCm, the PCI address does not.
        domain, bus, dev = (getattr(props, f"pci_{part}_id", None) for part in ("domain", "bus", "device"))
        if domain is None or bus is None or dev is None:
            return None
        return "amd", f"{domain:04x}:{bus:02x}:{dev:02x}.0"
    return None


def _amd_config(bdf: str, core: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    static, metric = _amd_smi("static", bdf), _amd_smi("metric", bdf)
    if static is None or metric is None:
        return core, {}
    core["gpu_max_clock_mhz"] = _amd_value(metric.get("clock", {}).get("gfx_0", {}).get("max_clk"))
    asic = static.get("asic", {})
    hardware = {
        "vendor": "amd",
        "target_graphics_version": asic.get("target_graphics_version"),
        "num_compute_units": asic.get("num_compute_units"),
    }
    return core, hardware


def _amd_state(bdf: str, core: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    metric = _amd_smi("metric", bdf)
    if metric is None:
        return core, {}
    clock, power = metric.get("clock", {}), metric.get("power", {})
    core["gpu_clock_mhz"] = _amd_value(clock.get("gfx_0", {}).get("clk"))
    core["gpu_mem_clock_mhz"] = _amd_value(clock.get("mem_0", {}).get("clk"))
    core["gpu_power_draw_w"] = _amd_value(power.get("socket_power"))
    core["gpu_temperature_c"] = _amd_value(metric.get("temperature", {}).get("hotspot"))
    if power.get("throttle_status") in ("THROTTLED", "UNTHROTTLED"):
        core["gpu_throttled"] = power["throttle_status"] == "THROTTLED"
    return core, {"clk_locked": clock.get("gfx_0", {}).get("clk_locked")}


def _amd_smi(command: str, bdf: str) -> dict[str, Any] | None:
    try:
        result = subprocess.run(["amd-smi", command, "-g", bdf, "--json"], capture_output=True, text=True, timeout=10)
        return json.loads(result.stdout)["gpu_data"][0] if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, IndexError):
        return None


def _amd_value(value: Any) -> Any:
    # amd-smi reports values as {"value": 145, "unit": "W"}, or "N/A".
    return value.get("value") if isinstance(value, dict) else None


def _query(device_id: str, fields: list[str]) -> dict[str, str] | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--id={device_id}", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
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
