"""Top-level AutoPilot orchestration for CROSSFIRE-X."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from crossfire.autopilot.bandit import ThompsonBandit, UCB1Bandit
from crossfire.autopilot.logger import DecisionLogger, DecisionRecord
from crossfire.autopilot.policy import (
    POLICY_REGISTRY,
    ExecutionPolicy,
    HardwareAvailability,
    available_policies,
)
from crossfire.autopilot.query_classifier import QueryClass, QueryFeatures, classify_query
from crossfire.autopilot.reward import RewardBreakdown, RewardInputs, RewardWeights, compute_reward


class BanditType(Enum):
    """Supported bandit backends for AutoPilot."""

    UCB1 = "ucb1"
    THOMPSON = "thompson"


@dataclass(frozen=True)
class AutoPilotConfig:
    """Runtime configuration for AutoPilot orchestration."""

    bandit_type: BanditType = BanditType.UCB1
    exploration_weight: float = 2.0
    success_threshold: float = 0.5
    seed: int | None = None


@dataclass(frozen=True)
class AutoPilotSelection:
    """Selected policy plus the decision context used to choose it."""

    query_class: QueryClass
    selected_policy: ExecutionPolicy
    available_policies: tuple[ExecutionPolicy, ...]
    was_exploration: bool
    scores: dict[str, float | None]


@dataclass(frozen=True)
class AutoPilotOutcome:
    """Observed metrics from one executed policy decision."""

    tokens_per_second: float
    tokens_per_watt: float
    ttft_ms: float
    perplexity_delta: float
    acceptance_rate: float
    execution_time_ms: float


@dataclass(frozen=True)
class AutoPilotBaselines:
    """Calibration baselines used for reward normalization."""

    tokens_per_second: float
    tokens_per_watt: float
    ttft_ms: float


DEFAULT_AUTOPILOT_CONFIG = AutoPilotConfig()
DEFAULT_REWARD_WEIGHTS = RewardWeights()


class AutoPilot:
    """Coordinate query classification, policy choice, reward, and logging."""

    def __init__(
        self,
        *,
        hardware: HardwareAvailability,
        config: AutoPilotConfig = DEFAULT_AUTOPILOT_CONFIG,
        reward_weights: RewardWeights = DEFAULT_REWARD_WEIGHTS,
        decision_logger: DecisionLogger | None = None,
    ) -> None:
        """Initialize AutoPilot state for all query classes."""

        reward_weights.validate()
        self.hardware = hardware
        self.config = config
        self.reward_weights = reward_weights
        self.decision_logger = decision_logger
        self._bandits = {query_class: self._make_bandit() for query_class in QueryClass}

    def select_policy(self, features: QueryFeatures) -> AutoPilotSelection:
        """Select an execution policy for one request."""

        query_class = classify_query(features)
        policies = tuple(available_policies(self.hardware))
        if not policies:
            msg = "No execution policies are available for the current hardware"
            raise ValueError(msg)

        bandit = self._bandits[query_class]
        selected_policy = bandit.select_arm(list(policies))
        scores = self._score_policies(query_class, list(policies))
        was_exploration = self._is_exploration(query_class, selected_policy)

        return AutoPilotSelection(
            query_class=query_class,
            selected_policy=selected_policy,
            available_policies=policies,
            was_exploration=was_exploration,
            scores=scores,
        )

    def record_outcome(
        self,
        selection: AutoPilotSelection,
        *,
        outcome: AutoPilotOutcome,
        baselines: AutoPilotBaselines,
    ) -> RewardBreakdown:
        """Update the selected bandit arm and optionally log the decision."""

        reward_inputs = RewardInputs(
            tokens_per_second=outcome.tokens_per_second,
            baseline_tokens_per_second=baselines.tokens_per_second,
            tokens_per_watt=outcome.tokens_per_watt,
            baseline_tokens_per_watt=baselines.tokens_per_watt,
            ttft_ms=outcome.ttft_ms,
            baseline_ttft_ms=baselines.ttft_ms,
            perplexity_delta=outcome.perplexity_delta,
        )
        reward = compute_reward(reward_inputs, weights=self.reward_weights)

        bandit = self._bandits[selection.query_class]
        bandit.update(selection.selected_policy, reward.total)

        if self.decision_logger is not None:
            self.decision_logger.log(
                DecisionRecord(
                    query_class=selection.query_class,
                    selected_policy=selection.selected_policy,
                    was_exploration=selection.was_exploration,
                    ucb_scores=selection.scores,
                    tokens_per_second=outcome.tokens_per_second,
                    tokens_per_watt=outcome.tokens_per_watt,
                    ttft_ms=outcome.ttft_ms,
                    acceptance_rate=outcome.acceptance_rate,
                    reward=reward.total,
                    execution_time_ms=outcome.execution_time_ms,
                )
            )

        return reward

    def get_policy_stats(self, query_class: QueryClass) -> dict[str, dict[str, float | int]]:
        """Return a serializable view of policy stats for one query class."""

        bandit = self._bandits[query_class]
        stats_by_policy: dict[str, dict[str, float | int]] = {}

        if isinstance(bandit, UCB1Bandit):
            for policy in POLICY_REGISTRY:
                stats = bandit.stats_for(policy)
                stats_by_policy[policy.name] = {
                    "pulls": stats.pulls,
                    "mean_reward": stats.mean_reward,
                }
            return stats_by_policy

        for policy in POLICY_REGISTRY:
            stats = bandit.stats_for(policy)
            stats_by_policy[policy.name] = {
                "pulls": stats.pulls,
                "alpha": stats.alpha,
                "beta": stats.beta,
            }
        return stats_by_policy

    def _make_bandit(self) -> UCB1Bandit | ThompsonBandit:
        """Construct the configured bandit backend."""

        arms = list(POLICY_REGISTRY)
        if self.config.bandit_type is BanditType.UCB1:
            return UCB1Bandit(arms, exploration_weight=self.config.exploration_weight)
        return ThompsonBandit(
            arms,
            success_threshold=self.config.success_threshold,
            seed=self.config.seed,
        )

    def _score_policies(
        self,
        query_class: QueryClass,
        policies: list[ExecutionPolicy],
    ) -> dict[str, float | None]:
        """Return a serializable score view for the candidate policies."""

        bandit = self._bandits[query_class]
        if isinstance(bandit, UCB1Bandit):
            return {policy.name: score for policy, score in bandit.score_arms(policies).items()}
        return {policy.name: score for policy, score in bandit.sample_scores(policies).items()}

    def _is_exploration(self, query_class: QueryClass, policy: ExecutionPolicy) -> bool:
        """Determine whether the current selection is an exploration step."""

        bandit = self._bandits[query_class]
        return bandit.stats_for(policy).pulls == 0
