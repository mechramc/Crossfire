"""Benchmark metric collection and reporting.

Collects perplexity, throughput, memory, and power metrics from inference runs
and formats them for comparison tables. Supports the CROSSFIRE v2 ablation
matrix (C0-C6) with per-target power tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    model: str
    quant_type: str
    context_size: int
    ablation_config: str = "c0"
    perplexity: float | None = None
    tokens_per_second: float | None = None
    peak_memory_mb: float | None = None
    kv_compression: str | None = None
    distributed: bool = False
    ane_active: bool = False
    ane_role: str | None = None
    total_power_watts: float | None = None
    ane_power_watts: float | None = None
    rdma_active: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())

    def to_row(self) -> list[str]:
        """Format as a table row for display.

        Returns:
            List of string values for tabular output.
        """
        return [
            self.ablation_config.upper(),
            self.model,
            self.quant_type,
            str(self.context_size),
            f"{self.perplexity:.2f}" if self.perplexity else "—",
            f"{self.tokens_per_second:.1f}" if self.tokens_per_second else "—",
            f"{self.peak_memory_mb:.0f}" if self.peak_memory_mb else "—",
            self.kv_compression or "none",
            self.ane_role or "idle",
            f"{self.total_power_watts:.0f}" if self.total_power_watts else "—",
        ]


TABLE_HEADERS = [
    "Config",
    "Model",
    "Quant",
    "Context",
    "PPL",
    "tok/s",
    "Peak MB",
    "KV Comp",
    "ANE",
    "Power (W)",
]
