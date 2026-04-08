"""Bandit algorithms for CROSSFIRE-X AutoPilot."""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import log, sqrt
from typing import TypeVar

Arm = TypeVar("Arm")


@dataclass
class ArmStats:
    """Running statistics for one bandit arm."""

    pulls: int = 0
    cumulative_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        """Average reward for the arm, or zero if unseen."""

        if self.pulls == 0:
            return 0.0
        return self.cumulative_reward / self.pulls


@dataclass
class ThompsonArmStats:
    """Beta-distribution parameters for one Thompson-sampling arm."""

    alpha: float = 1.0
    beta: float = 1.0

    @property
    def pulls(self) -> int:
        """Observed update count excluding the prior."""

        return int((self.alpha - 1.0) + (self.beta - 1.0))


class UCB1Bandit:
    """UCB1 multi-armed bandit with forced exploration for unseen arms."""

    def __init__(self, arms: list[Arm], *, exploration_weight: float = 2.0) -> None:
        """Initialize the bandit state for a fixed arm set."""

        if not arms:
            msg = "UCB1Bandit requires at least one arm"
            raise ValueError(msg)
        if exploration_weight <= 0:
            msg = "exploration_weight must be positive"
            raise ValueError(msg)

        self.exploration_weight = exploration_weight
        self._arms = list(arms)
        self._stats: dict[Arm, ArmStats] = {arm: ArmStats() for arm in self._arms}

    @property
    def total_pulls(self) -> int:
        """Total number of observed arm pulls."""

        return sum(stats.pulls for stats in self._stats.values())

    def stats_for(self, arm: Arm) -> ArmStats:
        """Return current stats for an arm."""

        return self._stats[arm]

    def score(self, arm: Arm) -> float:
        """Compute the UCB1 score for a seen arm."""

        stats = self._stats[arm]
        if stats.pulls == 0:
            msg = "Cannot score an arm with zero pulls; select it via forced exploration"
            raise ValueError(msg)

        return stats.mean_reward + self.exploration_weight * sqrt(
            log(self.total_pulls) / stats.pulls
        )

    def select_arm(self, available_arms: list[Arm] | None = None) -> Arm:
        """Select the next arm, forcing unseen arms first."""

        candidate_arms = self._resolve_available_arms(available_arms)

        for arm in candidate_arms:
            if self._stats[arm].pulls == 0:
                return arm

        return max(candidate_arms, key=self.score)

    def update(self, arm: Arm, reward: float) -> None:
        """Record an observation for a selected arm."""

        if arm not in self._stats:
            msg = f"Unknown arm: {arm!r}"
            raise KeyError(msg)
        if not 0.0 <= reward <= 1.0:
            msg = "reward must be normalized to [0, 1]"
            raise ValueError(msg)

        stats = self._stats[arm]
        stats.pulls += 1
        stats.cumulative_reward += reward

    def score_arms(self, available_arms: list[Arm] | None = None) -> dict[Arm, float | None]:
        """Return UCB1 scores for the currently available arms."""

        candidate_arms = self._resolve_available_arms(available_arms)
        return {
            arm: None if self._stats[arm].pulls == 0 else self.score(arm) for arm in candidate_arms
        }

    def _resolve_available_arms(self, available_arms: list[Arm] | None) -> list[Arm]:
        """Normalize and validate the candidate arm set."""

        if available_arms is None:
            return list(self._arms)
        if not available_arms:
            msg = "available_arms must not be empty"
            raise ValueError(msg)
        for arm in available_arms:
            if arm not in self._stats:
                msg = f"Unknown arm: {arm!r}"
                raise KeyError(msg)
        return list(available_arms)


class ThompsonBandit:
    """Thompson-sampling bandit with Beta priors and binary rewards."""

    def __init__(self, arms: list[Arm], *, success_threshold: float = 0.5, seed: int | None = None):
        """Initialize Thompson-sampling state for a fixed arm set."""

        if not arms:
            msg = "ThompsonBandit requires at least one arm"
            raise ValueError(msg)
        if not 0.0 <= success_threshold <= 1.0:
            msg = "success_threshold must be in [0, 1]"
            raise ValueError(msg)

        self.success_threshold = success_threshold
        self._arms = list(arms)
        self._stats: dict[Arm, ThompsonArmStats] = {arm: ThompsonArmStats() for arm in self._arms}
        self._rng = random.Random(seed)

    @property
    def total_pulls(self) -> int:
        """Total number of Thompson updates across all arms."""

        return sum(stats.pulls for stats in self._stats.values())

    def stats_for(self, arm: Arm) -> ThompsonArmStats:
        """Return current Beta parameters for an arm."""

        return self._stats[arm]

    def sample_score(self, arm: Arm) -> float:
        """Draw a Thompson score for an arm."""

        stats = self._stats[arm]
        return self._rng.betavariate(stats.alpha, stats.beta)

    def select_arm(self, available_arms: list[Arm] | None = None) -> Arm:
        """Select the next arm via Thompson sampling."""

        candidate_arms = self._resolve_available_arms(available_arms)
        return max(candidate_arms, key=self.sample_score)

    def update(self, arm: Arm, reward: float) -> None:
        """Update Beta priors using a binarized reward observation."""

        if arm not in self._stats:
            msg = f"Unknown arm: {arm!r}"
            raise KeyError(msg)
        if not 0.0 <= reward <= 1.0:
            msg = "reward must be normalized to [0, 1]"
            raise ValueError(msg)

        stats = self._stats[arm]
        if reward > self.success_threshold:
            stats.alpha += 1.0
        else:
            stats.beta += 1.0

    def sample_scores(self, available_arms: list[Arm] | None = None) -> dict[Arm, float]:
        """Draw Thompson scores for all currently available arms."""

        candidate_arms = self._resolve_available_arms(available_arms)
        return {arm: self.sample_score(arm) for arm in candidate_arms}

    def _resolve_available_arms(self, available_arms: list[Arm] | None) -> list[Arm]:
        """Normalize and validate the candidate arm set."""

        if available_arms is None:
            return list(self._arms)
        if not available_arms:
            msg = "available_arms must not be empty"
            raise ValueError(msg)
        for arm in available_arms:
            if arm not in self._stats:
                msg = f"Unknown arm: {arm!r}"
                raise KeyError(msg)
        return list(available_arms)
