"""Reward calculation for CROSSFIRE-X AutoPilot."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardWeights:
    """Weights for the multi-objective AutoPilot reward."""

    throughput: float = 0.4
    efficiency: float = 0.3
    latency: float = 0.2
    quality: float = 0.1

    def validate(self) -> None:
        """Validate weight values and normalization."""

        values = (self.throughput, self.efficiency, self.latency, self.quality)
        if any(weight < 0 for weight in values):
            msg = "Reward weights must be non-negative"
            raise ValueError(msg)
        if abs(sum(values) - 1.0) > 1e-9:
            msg = "Reward weights must sum to 1.0"
            raise ValueError(msg)


@dataclass(frozen=True)
class RewardInputs:
    """Measured execution metrics and calibration baselines."""

    tokens_per_second: float
    baseline_tokens_per_second: float
    tokens_per_watt: float
    baseline_tokens_per_watt: float
    ttft_ms: float
    baseline_ttft_ms: float
    perplexity_delta: float

    def validate(self) -> None:
        """Validate reward inputs are physically meaningful."""

        positive_fields = (
            ("tokens_per_second", self.tokens_per_second),
            ("baseline_tokens_per_second", self.baseline_tokens_per_second),
            ("tokens_per_watt", self.tokens_per_watt),
            ("baseline_tokens_per_watt", self.baseline_tokens_per_watt),
            ("ttft_ms", self.ttft_ms),
            ("baseline_ttft_ms", self.baseline_ttft_ms),
        )
        for name, value in positive_fields:
            if value <= 0:
                msg = f"{name} must be positive"
                raise ValueError(msg)
        if self.perplexity_delta < 0:
            msg = "perplexity_delta must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True)
class RewardBreakdown:
    """Component scores and final reward."""

    throughput: float
    efficiency: float
    latency: float
    quality: float
    total: float


DEFAULT_REWARD_WEIGHTS = RewardWeights()


def compute_reward(
    inputs: RewardInputs,
    *,
    weights: RewardWeights = DEFAULT_REWARD_WEIGHTS,
) -> RewardBreakdown:
    """Compute the normalized multi-objective reward."""

    weights.validate()
    inputs.validate()

    throughput = min(inputs.tokens_per_second / inputs.baseline_tokens_per_second, 2.0) / 2.0
    efficiency = min(inputs.tokens_per_watt / inputs.baseline_tokens_per_watt, 2.0) / 2.0
    latency = max(0.0, 1.0 - (inputs.ttft_ms / inputs.baseline_ttft_ms))
    if inputs.perplexity_delta < 0.5:
        quality = 1.0
    else:
        quality = max(0.0, 1.0 - (inputs.perplexity_delta / 5.0))

    total = (
        weights.throughput * throughput
        + weights.efficiency * efficiency
        + weights.latency * latency
        + weights.quality * quality
    )

    return RewardBreakdown(
        throughput=throughput,
        efficiency=efficiency,
        latency=latency,
        quality=quality,
        total=total,
    )
