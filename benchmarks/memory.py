"""VRAM and system RAM profiling during inference.

Samples peak memory usage on both NVIDIA GPUs (via nvidia-smi) and
Apple Silicon unified memory (via psutil) during benchmark runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class MemorySnapshot:
    """A point-in-time memory measurement."""

    system_used_mb: float
    system_total_mb: float
    gpu_used_mb: float | None = None
    gpu_total_mb: float | None = None


def get_system_memory() -> MemorySnapshot:
    """Capture current system memory usage.

    Returns:
        Memory snapshot with system RAM usage. GPU fields are None
        if nvidia-smi is not available.
    """
    mem = psutil.virtual_memory()
    return MemorySnapshot(
        system_used_mb=mem.used / (1024 * 1024),
        system_total_mb=mem.total / (1024 * 1024),
    )
