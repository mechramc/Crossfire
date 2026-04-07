"""KV cache compression using turbo3/turbo4 algorithms.

Manages KV cache compression configuration for llama.cpp inference,
enabling extended context lengths with minimal quality degradation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KVCacheConfig:
    """Configuration for KV cache compression."""

    algorithm: str = "turbo3"
    context_size: int = 8192
    cache_type_k: str = "q4_0"
    cache_type_v: str = "q4_0"

    def to_llama_args(self) -> list[str]:
        """Convert config to llama.cpp CLI arguments.

        Returns:
            List of CLI argument strings.
        """
        return [
            f"--ctx-size={self.context_size}",
            f"--cache-type-k={self.cache_type_k}",
            f"--cache-type-v={self.cache_type_v}",
        ]
