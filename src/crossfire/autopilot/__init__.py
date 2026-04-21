"""AutoPilot package for CROSSFIRE-X runtime policy selection."""

from crossfire.autopilot.autopilot import (
    DEFAULT_AUTOPILOT_CONFIG,
    AutoPilot,
    AutoPilotBaselines,
    AutoPilotConfig,
    AutoPilotEngine,
    AutoPilotOutcome,
    AutoPilotSelection,
    BanditType,
)
from crossfire.autopilot.bandit import ArmStats, ThompsonArmStats, ThompsonBandit, UCB1Bandit
from crossfire.autopilot.config_loader import (
    AutoPilotYamlConfig,
    build_autopilot_from_yaml,
    load_autopilot_yaml,
)
from crossfire.autopilot.decision_tree import (
    DEFAULT_DECISION_TREE_THRESHOLDS,
    DecisionContext,
    DecisionTreeThresholds,
)
from crossfire.autopilot.decision_tree import select_policy as decision_tree_select
from crossfire.autopilot.logger import DecisionLogger, DecisionRecord
from crossfire.autopilot.pipeline_integration import (
    apply_selection_to_pipeline,
    policy_requires_flash_moe,
    run_autopilot_cycle,
)
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
    "DEFAULT_DECISION_TREE_THRESHOLDS",
    "POLICY_REGISTRY",
    "ArmStats",
    "AutoPilot",
    "AutoPilotBaselines",
    "AutoPilotConfig",
    "AutoPilotEngine",
    "AutoPilotOutcome",
    "AutoPilotSelection",
    "AutoPilotYamlConfig",
    "BanditType",
    "DecisionContext",
    "DecisionLogger",
    "DecisionRecord",
    "DecisionTreeThresholds",
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
    "apply_selection_to_pipeline",
    "available_policies",
    "build_autopilot_from_yaml",
    "classify_query",
    "compute_reward",
    "decision_tree_select",
    "get_policy_config",
    "load_autopilot_yaml",
    "policy_requires_flash_moe",
    "run_autopilot_cycle",
]
