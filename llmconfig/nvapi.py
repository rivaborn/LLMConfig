"""Minimal NVAPI wrapper for GPU hotspot + memory-junction temperatures.

NVML / nvidia-smi expose only the GPU core temperature on consumer GeForce
cards. The hotspot (hottest point on the die) and GDDR6X memory junction —
the readings that actually predict throttling on the RTX 3090 — come only from
the undocumented but widely-used NvAPI_GPU_ClientThermalSensors_GetValues call
(id 0x65FE3AAD), the same entry point HWiNFO / GPU-Z use. Ported from the nmon
project (the original TUI this Monitor tab grew out of).

Windows-only, pure ctypes (no extra dependency). Every failure path degrades to
returning None so the sampler treats the sensors as simply unavailable — e.g.
when the control app runs without the NVIDIA display driver on PATH.
"""
from __future__ import annotations

import ctypes
import logging
import sys
import threading

log = logging.getLogger(__name__)

# NVAPI function ids. Initialize / EnumPhysicalGPUs are public; the client
# thermal sensors id is undocumented but stable across Ampere / Ada / Blackwell.
_NVAPI_INITIALIZE = 0x0150E828
_NVAPI_ENUM_PHYSICAL_GPUS = 0xE5AC921F
_NVAPI_GPU_CLIENT_THERMAL_SENSORS_GET_VALUES = 0x65FE3AAD

_NVAPI_MAX_PHYSICAL_GPUS = 64
_NVAPI_OK = 0

# Channel indices on Ampere / Ada consumer cards: 0 = GPU core, 1 = hotspot,
# 9 = GDDR6X memory junction (on cards that expose it: 3080/3090/4080/4090).
_SENSOR_INDEX_HOTSPOT = 1
_SENSOR_INDEX_MEMORY = 9

# Sensor-channel masks to try, widest first; the driver rejects masks asking
# for channels the card lacks, so we fall through to narrower ones.
_SENSOR_MASKS = (0xFFFF, 0x3FF, 0x1FF, 0xFF, 0x1F, 0x0F, 0x03, 0x01)

# Temperatures are Q8.8 fixed point (raw / 256 = degrees C).
_TEMP_DIVISOR = 256.0


class _NvGpuClientThermalSensors(ctypes.Structure):
    """NV_GPU_CLIENT_THERMAL_SENSORS v2 (168 bytes). Temps in Q8.8 °C."""
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("mask", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 8),
        ("temperatures", ctypes.c_int32 * 32),
    ]


_V2_STRUCT_SIZE = ctypes.sizeof(_NvGpuClientThermalSensors)
_V2_VERSION = _V2_STRUCT_SIZE | (2 << 16)

_lock = threading.Lock()
_state = {
    "init_tried": False,
    "initialized": False,
    "query_iface": None,
    "gpu_handles": [],
    "fn_cache": {},
    "unsupported_gpus": set(),
}


def _load_and_init() -> bool:
    """Lazily load nvapi64.dll and call NvAPI_Initialize. Idempotent."""
    if _state["initialized"]:
        return True
    if _state["init_tried"]:
        return False
    _state["init_tried"] = True

    if sys.platform != "win32":
        log.debug("NVAPI: not on Windows, skipping")
        return False

    try:
        dll = ctypes.WinDLL("nvapi64.dll")
    except (OSError, AttributeError) as e:
        log.debug("NVAPI: nvapi64.dll not available: %s", e)
        return False

    try:
        query = dll.nvapi_QueryInterface
    except AttributeError:
        log.debug("NVAPI: nvapi_QueryInterface export missing")
        return False

    query.restype = ctypes.c_void_p
    query.argtypes = [ctypes.c_uint32]
    _state["query_iface"] = query

    addr = query(_NVAPI_INITIALIZE)
    if not addr:
        log.debug("NVAPI: could not resolve NvAPI_Initialize")
        return False
    init_fn = ctypes.CFUNCTYPE(ctypes.c_int32)(addr)
    status = init_fn()
    if status != _NVAPI_OK:
        log.debug("NVAPI: NvAPI_Initialize returned %d", status)
        return False

    _state["initialized"] = True
    return True


def _resolve(fn_id: int, prototype):
    cache = _state["fn_cache"]
    if fn_id in cache:
        return cache[fn_id]
    addr = _state["query_iface"](fn_id)
    fn = prototype(addr) if addr else None
    cache[fn_id] = fn
    return fn


def _enum_gpus() -> bool:
    if _state["gpu_handles"]:
        return True
    proto = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32))
    fn = _resolve(_NVAPI_ENUM_PHYSICAL_GPUS, proto)
    if fn is None:
        return False
    handles = (ctypes.c_void_p * _NVAPI_MAX_PHYSICAL_GPUS)()
    count = ctypes.c_uint32(0)
    if fn(handles, ctypes.byref(count)) != _NVAPI_OK:
        return False
    _state["gpu_handles"] = [ctypes.c_void_p(handles[i]) for i in range(count.value)]
    return True


def _read_thermal_sensors(gpu_index: int):
    if not _load_and_init() or not _enum_gpus():
        return None
    handles = _state["gpu_handles"]
    if gpu_index >= len(handles):
        return None
    proto = ctypes.CFUNCTYPE(
        ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(_NvGpuClientThermalSensors)
    )
    fn = _resolve(_NVAPI_GPU_CLIENT_THERMAL_SENSORS_GET_VALUES, proto)
    if fn is None:
        return None
    for mask in _SENSOR_MASKS:
        data = _NvGpuClientThermalSensors()
        data.version = _V2_VERSION
        data.mask = mask
        try:
            if fn(handles[gpu_index], ctypes.byref(data)) == _NVAPI_OK:
                return data
        except OSError as e:
            log.debug("NVAPI: thermal sensors call raised: %s", e)
            return None
    return None


def read_thermal_channels(gpu_index: int) -> dict[str, float] | None:
    """Hotspot + memory-junction temps (°C) for a GPU by driver index.

    Returns a dict with some subset of {"hotspot", "memory"}, or None when
    NVAPI is unavailable or the card exposes neither sensor. A GPU that fails
    outright is cached as unsupported so we stop paying for the round trip.
    Thread-safe; the underlying call is blocking, so callers should run it off
    the event loop (e.g. asyncio.to_thread).
    """
    with _lock:
        if gpu_index in _state["unsupported_gpus"]:
            return None
        data = _read_thermal_sensors(gpu_index)
        if data is None:
            _state["unsupported_gpus"].add(gpu_index)
            return None
        result: dict[str, float] = {}
        hotspot_raw = data.temperatures[_SENSOR_INDEX_HOTSPOT]
        if hotspot_raw > 0:
            result["hotspot"] = hotspot_raw / _TEMP_DIVISOR
        memory_raw = data.temperatures[_SENSOR_INDEX_MEMORY]
        if memory_raw > 0:
            result["memory"] = memory_raw / _TEMP_DIVISOR
        if not result:
            _state["unsupported_gpus"].add(gpu_index)
            return None
        return result
