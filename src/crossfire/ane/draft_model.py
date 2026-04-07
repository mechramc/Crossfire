"""ANE draft model for speculative decoding.

Runs a small model (0.6B-1B) on the Apple Neural Engine via ANEMLL
to generate draft tokens for speculative verification against the
main model running on Metal GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ANEBackend(Enum):
    """Backend for ANE inference."""

    ANEMLL = "anemll"
    RUSTANE = "rustane"
    COREML = "coreml"


@dataclass(frozen=True)
class DraftModelConfig:
    """Configuration for an ANE draft model."""

    model_path: Path
    backend: ANEBackend = ANEBackend.ANEMLL
    max_tokens: int = 5
    context_size: int = 2048

    def validate(self) -> None:
        """Validate draft model configuration.

        Raises:
            ValueError: If context exceeds ANE limits.
            FileNotFoundError: If model path does not exist.
        """
        max_ane_context = 4096
        if self.context_size > max_ane_context:
            msg = (
                f"ANE context {self.context_size} exceeds ANEMLL limit of {max_ane_context}. "
                "Consider reducing context or using Metal GPU."
            )
            raise ValueError(msg)

        if not self.model_path.exists():
            msg = f"Draft model not found: {self.model_path}"
            raise FileNotFoundError(msg)


@dataclass(frozen=True)
class DraftResult:
    """Result from a draft model forward pass on ANE."""

    tokens: list[int]
    logits_shape: tuple[int, ...]
    elapsed_ms: float
    power_watts: float | None = None
