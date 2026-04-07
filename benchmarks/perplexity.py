"""Perplexity measurement using llama.cpp perplexity binary.

Runs the llama-perplexity tool against a model with a standard
evaluation dataset and extracts the final PPL score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PerplexityConfig:
    """Configuration for a perplexity benchmark run."""

    model_path: Path
    dataset_path: Path
    context_size: int = 2048
    batch_size: int = 512
    threads: int = 8
    gpu_layers: int = 0


def run_perplexity(config: PerplexityConfig) -> float:
    """Run perplexity evaluation and return the PPL score.

    Args:
        config: Benchmark parameters.

    Returns:
        Final perplexity value.

    Raises:
        FileNotFoundError: If model or dataset not found.
        RuntimeError: If perplexity binary fails.
    """
    if not config.model_path.exists():
        msg = f"Model not found: {config.model_path}"
        raise FileNotFoundError(msg)
    if not config.dataset_path.exists():
        msg = f"Dataset not found: {config.dataset_path}"
        raise FileNotFoundError(msg)

    raise NotImplementedError("Perplexity benchmark not yet implemented")
