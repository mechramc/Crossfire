"""Benchmark metric collection and reporting.

Collects perplexity, throughput, and memory metrics from inference runs
and formats them for comparison tables.
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
    perplexity: float | None = None
    tokens_per_second: float | None = None
    peak_memory_mb: float | None = None
    kv_compression: str | None = None
    distributed: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())

    def to_row(self) -> list[str]:
        """Format as a table row for display.

        Returns:
            List of string values for tabular output.
        """
        return [
            self.model,
            self.quant_type,
            str(self.context_size),
            f"{self.perplexity:.2f}" if self.perplexity else "—",
            f"{self.tokens_per_second:.1f}" if self.tokens_per_second else "—",
            f"{self.peak_memory_mb:.0f}" if self.peak_memory_mb else "—",
            self.kv_compression or "none",
            "yes" if self.distributed else "no",
        ]


TABLE_HEADERS = [
    "Model",
    "Quant",
    "Context",
    "PPL",
    "tok/s",
    "Peak MB",
    "KV Comp",
    "Distributed",
]
