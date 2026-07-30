from __future__ import annotations

import os
import subprocess
from functools import lru_cache


def get_hardware_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return os.uname().machine


@lru_cache(maxsize=1)
def _nvml():
    """Return an initialized ``pynvml`` module, or ``None`` if unavailable.

    Cached because ``nvmlInit`` is once-per-process and the fallback path is
    expensive enough that we do not want to retry it on every sample.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


def _power_via_nvidia_smi() -> float | None:
    """Board power via ``nvidia-smi``, the fallback when NVML is unavailable.

    Costs tens of milliseconds per call because it spawns a process, so it is
    unsuitable for sampling inside a timed region.
    """
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip().splitlines()

        if result:
            return float(result[0])
        return None
    except Exception:
        return None


def get_gpu_power_watts(device: int = 0) -> float | None:
    """Instantaneous board power draw in watts.

    Prefers NVML, which is a library call costing microseconds. The previous
    ``nvidia-smi`` subprocess cost ~30 ms per reading, which both capped the
    achievable sampling rate and stole CPU from the work being measured.

    Note this is whole-board power, including any other process on the GPU, so
    it is only meaningful on an otherwise idle device.

    Args:
        device: CUDA device index to read.

    Returns:
        Watts, or ``None`` if no reading is available.
    """
    nvml = _nvml()
    if nvml is not None:
        try:
            handle = nvml.nvmlDeviceGetHandleByIndex(device)
            return nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except Exception:
            pass
    return _power_via_nvidia_smi()


def cuda_device_indices() -> list[int]:
    """Indices of visible CUDA devices, empty when CUDA is unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except Exception:
        pass
    return []


def reset_peak_memory_stats() -> None:
    """Reset the allocator's peak counters on every visible CUDA device.

    All devices, not just the current one: a model sharded by
    ``device_map="auto"`` holds memory on each of them.
    """
    try:
        import torch

        for index in cuda_device_indices():
            torch.cuda.reset_peak_memory_stats(index)
    except Exception:
        pass


def get_peak_vram_reserved_gb() -> float | None:
    """Peak VRAM *reserved* by the caching allocator, summed over all devices.

    Reserved rather than allocated: the allocator's high-water mark is what the
    process actually holds from the driver, so it includes fragmentation, while
    ``max_memory_allocated`` counts only live tensors and understates the
    footprint. Summed over devices because ``max_memory_*`` defaults to the
    current device and would report a single shard of a sharded model.

    Still excludes the CUDA context and any non-PyTorch allocation -- see
    :func:`get_device_vram_used_gb` for the true device figure.

    Returns:
        Gibibytes, or ``None`` if CUDA is unavailable.
    """
    try:
        import torch

        indices = cuda_device_indices()
        if not indices:
            return None
        total = sum(torch.cuda.max_memory_reserved(i) for i in indices)
        return total / (1024**3)
    except Exception:
        return None


def get_device_vram_used_gb() -> float | None:
    """VRAM in use device-wide, summed over all visible CUDA devices.

    Derived from ``mem_get_info`` (``total - free``), so unlike the allocator
    counters this includes the CUDA context (a few hundred MB), driver
    overhead, and any other process on the card. This is the number to size a
    deployment GPU against.

    Instantaneous, not a high-water mark: sample it while the work is running.

    Returns:
        Gibibytes, or ``None`` if CUDA is unavailable.
    """
    try:
        import torch

        indices = cuda_device_indices()
        if not indices:
            return None
        used = 0
        for index in indices:
            free, total = torch.cuda.mem_get_info(index)
            used += total - free
        return used / (1024**3)
    except Exception:
        return None
