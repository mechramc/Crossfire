"""ANE and per-component power measurement on Apple Silicon.

Uses macOS powermetrics to capture per-component wattage during inference,
enabling power-performance analysis of ANE vs GPU vs CPU contributions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PowerSnapshot:
    """Per-component power measurement on Apple Silicon."""

    ane_watts: float | None = None
    gpu_watts: float | None = None
    cpu_watts: float | None = None
    total_system_watts: float | None = None

    @property
    def ane_fraction(self) -> float | None:
        """ANE power as fraction of total system power."""
        if self.ane_watts is not None and self.total_system_watts:
            return self.ane_watts / self.total_system_watts
        return None


# Expected ANE power range based on maderix characterization
ANE_IDLE_WATTS = 0.5
ANE_ACTIVE_WATTS_MIN = 2.0
ANE_ACTIVE_WATTS_MAX = 5.0
ANE_TFLOPS_FP16 = 19.0
ANE_EFFICIENCY_TFLOPS_PER_WATT = 6.6  # 80x more efficient than A100
ANE_SRAM_CLIFF_MB = 32  # 30% throughput drop beyond this
