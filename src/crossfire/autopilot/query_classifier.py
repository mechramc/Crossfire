"""Query classification for CROSSFIRE-X AutoPilot."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QueryClass(Enum):
    """Query classes used as AutoPilot bandit contexts."""

    BATCH = "batch"
    SHORT_GEN = "short_gen"
    LONG_GEN = "long_gen"
    VERY_LONG_PROMPT = "very_long_prompt"
    LONG_PROMPT = "long_prompt"
    MEDIUM_PROMPT = "medium_prompt"
    SHORT_PROMPT = "short_prompt"


@dataclass(frozen=True)
class QueryFeatures:
    """Request features used for execution-policy classification."""

    prompt_tokens: int
    max_gen_tokens: int
    context_used: int
    model_size_b: float
    available_vram_mb: float
    concurrent_requests: int

    def __post_init__(self) -> None:
        """Validate feature values are physically meaningful."""

        if self.prompt_tokens < 0:
            msg = "prompt_tokens must be non-negative"
            raise ValueError(msg)
        if self.max_gen_tokens < 0:
            msg = "max_gen_tokens must be non-negative"
            raise ValueError(msg)
        if self.context_used < 0:
            msg = "context_used must be non-negative"
            raise ValueError(msg)
        if self.model_size_b <= 0:
            msg = "model_size_b must be positive"
            raise ValueError(msg)
        if self.available_vram_mb < 0:
            msg = "available_vram_mb must be non-negative"
            raise ValueError(msg)
        if self.concurrent_requests < 0:
            msg = "concurrent_requests must be non-negative"
            raise ValueError(msg)


def classify_query(features: QueryFeatures) -> QueryClass:
    """Classify a request using the spec-defined priority order."""

    if features.concurrent_requests > 1:
        return QueryClass.BATCH

    if features.max_gen_tokens <= 64:
        return QueryClass.SHORT_GEN
    if features.max_gen_tokens > 256:
        return QueryClass.LONG_GEN

    if features.prompt_tokens >= 16384:
        return QueryClass.VERY_LONG_PROMPT
    if features.prompt_tokens >= 4096:
        return QueryClass.LONG_PROMPT
    if features.prompt_tokens >= 512:
        return QueryClass.MEDIUM_PROMPT
    return QueryClass.SHORT_PROMPT
