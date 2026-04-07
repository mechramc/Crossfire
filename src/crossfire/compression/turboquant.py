"""TQ4_1S weight compression via llama-cpp-turboquant.

Wraps the TurboQuant+ quantization pipeline for converting models
from standard GGUF formats (Q8_0, F16) to TQ4_1S compressed format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QuantConfig:
    """Configuration for TQ4_1S quantization."""

    input_path: Path
    output_path: Path
    quant_type: str = "TQ4_1S"
    threads: int = 8


def quantize_model(config: QuantConfig) -> Path:
    """Quantize a model using TurboQuant+ TQ4_1S format.

    Args:
        config: Quantization parameters.

    Returns:
        Path to the quantized model file.

    Raises:
        FileNotFoundError: If input model does not exist.
        RuntimeError: If quantization subprocess fails.
    """
    if not config.input_path.exists():
        msg = f"Input model not found: {config.input_path}"
        raise FileNotFoundError(msg)

    raise NotImplementedError("Quantization pipeline not yet implemented")
