"""Top-level AutoPilot orchestration for CROSSFIRE-X.

Supports two selection engines:

  DECISION_TREE  -- deterministic rule-based selector (unified spec default).
                   Use for predictable, zero-cold-start policy selection.
                   See crossfire.autopilot.decision_tree for the implementation.

  UCB1_BANDIT    -- adaptive UCB1 multi-armed bandit per query class.
                   Learns optimal policy per workload pattern over time.

  THOMPSON_BANDIT -- Thompson sampling alternative to UCB1.

Both bandit modes still log decisions and compute rewards for analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from crossfire.autopilot.bandit import ThompsonBandit, UCB1Bandit
from crossfire.autopilot.decision_tree import DecisionContext
from crossfire.autopilot.decision_tree import select_policy as dt_select
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
    """Supported bandit backends for AutoPilot (legacy name; use AutoPilotEngine)."""

    UCB1 = "ucb1"
    THOMPSON = "thompson"


class AutoPilotEngine(Enum):
    """AutoPilot selection engine.

    Attributes:
        DECISION_TREE: Deterministic rule-based tree (unified spec default).
            No cold-start, no exploration overhead. Predictable policy
            selection from the first request.
        UCB1_BANDIT: Adaptive UCB1 multi-armed bandit. Learns the optimal
            policy per query class over N requests. Requires cold-start.
        THOMPSON_BANDIT: Thompson sampling alternative to UCB1.
    """

    DECISION_TREE = "decision_tree"
    UCB1_BANDIT = "ucb1"
    THOMPSON_BANDIT = "thompson"


@dataclass(frozen=True)
class AutoPilotConfig:
    """Runtime configuration for AutoPilot orchestration."""

    engine: AutoPilotEngine = AutoPilotEngine.DECISION_TREE
    exploration_weight: float = 2.0
    success_threshold: float = 0.5
    seed: int | None = None

    # Backwards-compatible alias -- if set, overrides engine
    bandit_type: BanditType | None = None

    def resolved_engine(self) -> AutoPilotEngine:
        """Resolve the effective engine, respecting the legacy bandit_type field."""
        if self.bandit_type is not None:
            return (
                AutoPilotEngine.UCB1_BANDIT
                if self.bandit_type is BanditType.UCB1
                else AutoPilotEngine.THOMPSON_BANDIT
            )
        return self.engine


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
    """Coordinate query classification, policy choice, reward, and logging.

    Supports two selection engines configured via AutoPilotConfig.engine:
      - DECISION_TREE: deterministic, zero-cold-start (unified spec default)
      - UCB1_BANDIT / THOMPSON_BANDIT: adaptive, learns per workload

    Both engines produce AutoPilotSelection objects with the same shape,
    so callers do not need to branch on the engine type.
    """

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
        """Select an execution policy for one request.

        Routes to the decision tree or bandit engine based on config.engine.
        For the decision tree, DecisionContext is constructed from QueryFeatures.
        """

        query_class = classify_query(features)
        policies = tuple(available_policies(self.hardware))
        if not policies:
            msg = "No execution policies are available for the current hardware"
            raise ValueError(msg)

        engine = self.config.resolved_engine()

        if engine is AutoPilotEngine.DECISION_TREE:
            return self._select_via_decision_tree(features, query_class, policies)

        return self._select_via_bandit(query_class, policies)

    def _select_via_decision_tree(
        self,
        features: QueryFeatures,
        query_class: QueryClass,
        policies: tuple[ExecutionPolicy, ...],
    ) -> AutoPilotSelection:
        """Select policy using the deterministic decision tree."""

        ctx = DecisionContext(
            prompt_len=features.prompt_tokens,
            output_len=features.max_gen_tokens,
            context_len=features.context_used,
            model_size_gb=features.model_size_b * 2.0,  # rough Q8_0 GB estimate
            model_is_moe=features.model_is_moe,
            decode_is_bottleneck=False,  # heuristic: true for long-gen classes
        )
        selected_policy = dt_select(ctx)

        # Clamp to available policies (hardware may not support all)
        if selected_policy not in policies:
            selected_policy = policies[0]  # P0 always available as fallback

        return AutoPilotSelection(
            query_class=query_class,
            selected_policy=selected_policy,
            available_policies=policies,
            was_exploration=False,  # decision tree is deterministic
            scores={p.name: None for p in policies},
        )

    def _select_via_bandit(
        self,
        query_class: QueryClass,
        policies: tuple[ExecutionPolicy, ...],
    ) -> AutoPilotSelection:
        """Select policy using the UCB1 or Thompson bandit."""

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
