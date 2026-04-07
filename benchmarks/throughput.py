"""Token throughput (tok/s) benchmarking.

Measures prompt processing (prefill) and generation (decode) speeds
for both single-node and distributed configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ThroughputConfig:
    """Configuration for a throughput benchmark run."""

    model_path: Path
    prompt: str = "The meaning of life is"
    n_predict: int = 128
    context_size: int = 2048
    threads: int = 8
    gpu_layers: int = 0


@dataclass(frozen=True)
class ThroughputResult:
    """Results from a throughput benchmark."""

    prefill_tok_per_sec: float
    decode_tok_per_sec: float
    total_tokens: int
    elapsed_seconds: float


def run_throughput(config: ThroughputConfig) -> ThroughputResult:
    """Run throughput benchmark and return timing results.

    Args:
        config: Benchmark parameters.

    Returns:
        Throughput measurements.

    Raises:
        FileNotFoundError: If model not found.
        RuntimeError: If inference binary fails.
    """
    if not config.model_path.exists():
        msg = f"Model not found: {config.model_path}"
        raise FileNotFoundError(msg)

    raise NotImplementedError("Throughput benchmark not yet implemented")
