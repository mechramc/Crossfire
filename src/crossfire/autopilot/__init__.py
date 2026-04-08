"""AutoPilot package for CROSSFIRE-X runtime policy selection."""

from crossfire.autopilot.autopilot import (
    DEFAULT_AUTOPILOT_CONFIG,
    AutoPilot,
    AutoPilotBaselines,
    AutoPilotConfig,
    AutoPilotOutcome,
    AutoPilotSelection,
    BanditType,
)
from crossfire.autopilot.bandit import ArmStats, ThompsonArmStats, ThompsonBandit, UCB1Bandit
from crossfire.autopilot.logger import DecisionLogger, DecisionRecord
from crossfire.autopilot.policy import (
    POLICY_REGISTRY,
    ExecutionPolicy,
    HardwareAvailability,
    PolicyConfig,
    available_policies,
    get_policy_config,
)
from crossfire.autopilot.query_classifier import QueryClass, QueryFeatures, classify_query
from crossfire.autopilot.reward import RewardBreakdown, RewardInputs, RewardWeights, compute_reward

__all__ = [
    "DEFAULT_AUTOPILOT_CONFIG",
    "POLICY_REGISTRY",
    "ArmStats",
    "AutoPilot",
    "AutoPilotBaselines",
    "AutoPilotConfig",
    "AutoPilotOutcome",
    "AutoPilotSelection",
    "BanditType",
    "DecisionLogger",
    "DecisionRecord",
    "ExecutionPolicy",
    "HardwareAvailability",
    "PolicyConfig",
    "QueryClass",
    "QueryFeatures",
    "RewardBreakdown",
    "RewardInputs",
    "RewardWeights",
    "ThompsonArmStats",
    "ThompsonBandit",
    "UCB1Bandit",
    "available_policies",
    "classify_query",
    "compute_reward",
    "get_policy_config",
]
