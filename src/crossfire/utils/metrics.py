"""Benchmark metric collection and reporting.

Collects perplexity, throughput, memory, and power metrics from inference runs
and formats them for comparison tables. Schema is aligned with the unified spec
execution-policy model (P0-P6) and the AutoPilot decision log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run.

    Attributes:
        model: Model identifier (e.g. "qwen3.5-27b", "qwen3.5-35b-a3b").
        quant_type: Weight quantization type ("Q8_0", "TQ4_1S", etc.).
        context_size: Context window size in tokens.
        execution_policy: AutoPilot policy used (P0-P6). Default "P0".
        ablation_config: Ablation config label (C0-C7). Optional -- used when
            running the full ablation matrix rather than AutoPilot.
        perplexity: Measured perplexity (wikitext-2-raw-v1, 20 chunks at c=512).
        tokens_per_second: Decode throughput in tok/s.
        prefill_tokens_per_second: Prefill throughput in tok/s (T1 CUDA GPU).
        ttft_ms: Time to first token in milliseconds.
        tokens_per_watt: System-level efficiency (tok/s / total_power_watts).
        acceptance_rate: Speculative decode acceptance rate (0.0-1.0). None
            if speculative decode was not active.
        flash_moe_hit_rate: Flash-MoE slot-bank hit rate (0.0-1.0). None
            if Flash-MoE was not active.
        peak_memory_mb: Peak memory usage in MB.
        kv_compression: KV compression strategy ("triattention", "turbo3", "none").
        distributed: True if EXO distributed pipeline was active.
        ane_active: True if ANE (T3) was active.
        ane_role: ANE role if active ("draft", "policy_model", etc.).
        total_power_watts: Total system power in watts.
        ane_power_watts: ANE-specific power in watts.
        interconnect: Physical interconnect label used for distributed runs
            (e.g. "usb4", "5gbe", "1gbe", "wifi"). None for single-node runs.
        interconnect_bytes: Bytes transferred across the interconnect during
            the run. Used to quantify compression savings.
        flash_moe_active: True if Flash-MoE (T5 NVMe SSD) was active.
    """

    model: str
    quant_type: str
    context_size: int
    execution_policy: str = "P0"
    ablation_config: str | None = None
    perplexity: float | None = None
    tokens_per_second: float | None = None
    prefill_tokens_per_second: float | None = None
    ttft_ms: float | None = None
    tokens_per_watt: float | None = None
    acceptance_rate: float | None = None
    flash_moe_hit_rate: float | None = None
    peak_memory_mb: float | None = None
    kv_compression: str | None = None
    distributed: bool = False
    ane_active: bool = False
    ane_role: str | None = None
    total_power_watts: float | None = None
    ane_power_watts: float | None = None
    interconnect: str | None = None
    interconnect_bytes: int | None = None
    flash_moe_active: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())

    def to_row(self) -> list[str]:
        """Format as a table row for display.

        Returns:
            List of string values for tabular output matching TABLE_HEADERS.
        """
        label = self.ablation_config.upper() if self.ablation_config else self.execution_policy
        return [
            label,
            self.model,
            self.quant_type,
            str(self.context_size),
            f"{self.perplexity:.2f}" if self.perplexity else "—",
            f"{self.tokens_per_second:.1f}" if self.tokens_per_second else "—",
            f"{self.ttft_ms:.0f}" if self.ttft_ms else "—",
            f"{self.tokens_per_watt:.3f}" if self.tokens_per_watt else "—",
            f"{self.peak_memory_mb:.0f}" if self.peak_memory_mb else "—",
            self.kv_compression or "none",
            self.ane_role or "idle",
            f"{self.acceptance_rate:.2f}" if self.acceptance_rate is not None else "—",
            f"{self.flash_moe_hit_rate:.2f}" if self.flash_moe_hit_rate is not None else "—",
            f"{self.total_power_watts:.0f}" if self.total_power_watts else "—",
        ]


TABLE_HEADERS = [
    "Policy",
    "Model",
    "Quant",
    "Context",
    "PPL",
    "tok/s",
    "TTFT (ms)",
    "tok/W",
    "Peak MB",
    "KV Comp",
    "ANE",
    "Accept Rate",
    "MoE Hit%",
    "Power (W)",
]
