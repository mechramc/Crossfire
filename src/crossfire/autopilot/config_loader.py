"""YAML configuration loader for CROSSFIRE-X AutoPilot.

Parses `configs/autopilot.yaml` into the dataclasses consumed by AutoPilot:

  - AutoPilotConfig (engine, exploration_weight, success_threshold)
  - RewardWeights
  - DecisionTreeThresholds
  - DecisionLogger (if a log_path is configured)
  - AutoPilotBaselines (optional; returned as None if baselines are unset)

The loader is strict about schema typos — unknown engines or malformed values
raise ValueError — but permissive about missing optional sections.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from crossfire.autopilot.autopilot import (
    AutoPilot,
    AutoPilotBaselines,
    AutoPilotConfig,
    AutoPilotEngine,
)
from crossfire.autopilot.decision_tree import (
    DEFAULT_DECISION_TREE_THRESHOLDS,
    DecisionTreeThresholds,
)
from crossfire.autopilot.logger import DecisionLogger
from crossfire.autopilot.policy import HardwareAvailability
from crossfire.autopilot.reward import RewardWeights


@dataclass(frozen=True)
class AutoPilotYamlConfig:
    """Parsed view of `configs/autopilot.yaml`."""

    autopilot: AutoPilotConfig
    reward_weights: RewardWeights
    decision_tree_thresholds: DecisionTreeThresholds
    log_path: Path | None
    baselines: AutoPilotBaselines | None


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        msg = f"'{field}' must be a mapping"
        raise ValueError(msg)
    return value


def _parse_engine(name: str) -> AutoPilotEngine:
    try:
        return AutoPilotEngine(name)
    except ValueError as exc:
        valid = ", ".join(e.value for e in AutoPilotEngine)
        msg = f"Unknown autopilot engine {name!r}; expected one of: {valid}"
        raise ValueError(msg) from exc


def _parse_autopilot_section(section: dict[str, Any]) -> AutoPilotConfig:
    engine_name = section.get("engine", AutoPilotEngine.DECISION_TREE.value)
    engine = _parse_engine(str(engine_name))

    bandit = _require_mapping(section.get("bandit"), "autopilot.bandit")
    exploration_weight = float(bandit.get("exploration_weight", 2.0))
    success_threshold = float(bandit.get("success_threshold", 0.5))

    return AutoPilotConfig(
        engine=engine,
        exploration_weight=exploration_weight,
        success_threshold=success_threshold,
    )


def _parse_decision_tree(section: dict[str, Any]) -> DecisionTreeThresholds:
    if not section:
        return DEFAULT_DECISION_TREE_THRESHOLDS
    defaults = DEFAULT_DECISION_TREE_THRESHOLDS
    return DecisionTreeThresholds(
        short_token_threshold=int(
            section.get("short_token_threshold", defaults.short_token_threshold)
        ),
        long_context_threshold=int(
            section.get("long_context_threshold", defaults.long_context_threshold)
        ),
        memory_threshold_gb=float(section.get("memory_threshold_gb", defaults.memory_threshold_gb)),
        node_memory_gb=float(section.get("node_memory_gb", defaults.node_memory_gb)),
    )


def _parse_reward_weights(section: dict[str, Any]) -> RewardWeights:
    if not section:
        return RewardWeights()
    weights = RewardWeights(
        throughput=float(section.get("throughput", 0.4)),
        efficiency=float(section.get("efficiency", 0.3)),
        latency=float(section.get("latency", 0.2)),
        quality=float(section.get("quality", 0.1)),
    )
    weights.validate()
    return weights


def _parse_baselines(section: dict[str, Any]) -> AutoPilotBaselines | None:
    """Parse the `baselines.p1_8k` section if fully populated."""

    p1 = _require_mapping(section.get("p1_8k"), "baselines.p1_8k")
    tps = p1.get("tokens_per_second")
    tpw = p1.get("tokens_per_watt")
    ttft = p1.get("ttft_ms")
    if tps is None or tpw is None or ttft is None:
        return None
    return AutoPilotBaselines(
        tokens_per_second=float(tps),
        tokens_per_watt=float(tpw),
        ttft_ms=float(ttft),
    )


def load_autopilot_yaml(path: str | Path) -> AutoPilotYamlConfig:
    """Parse an autopilot YAML file into structured config objects."""

    yaml_path = Path(path)
    if not yaml_path.is_file():
        msg = f"AutoPilot config not found: {yaml_path}"
        raise FileNotFoundError(msg)

    with yaml_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        msg = f"AutoPilot config root must be a mapping, got {type(raw).__name__}"
        raise ValueError(msg)

    autopilot_section = _require_mapping(raw.get("autopilot"), "autopilot")
    decision_tree_section = _require_mapping(
        autopilot_section.get("decision_tree"), "autopilot.decision_tree"
    )
    reward_section = _require_mapping(raw.get("reward_weights"), "reward_weights")
    logging_section = _require_mapping(raw.get("logging"), "logging")
    baselines_section = _require_mapping(raw.get("baselines"), "baselines")

    log_path_value = logging_section.get("log_path")
    log_path = Path(log_path_value) if log_path_value else None

    return AutoPilotYamlConfig(
        autopilot=_parse_autopilot_section(autopilot_section),
        reward_weights=_parse_reward_weights(reward_section),
        decision_tree_thresholds=_parse_decision_tree(decision_tree_section),
        log_path=log_path,
        baselines=_parse_baselines(baselines_section),
    )


def build_autopilot_from_yaml(
    path: str | Path,
    *,
    hardware: HardwareAvailability,
    enable_logging: bool = True,
) -> AutoPilot:
    """Construct a fully-wired AutoPilot from `configs/autopilot.yaml`.

    Args:
        path: Path to the YAML config file.
        hardware: Current HardwareAvailability state (drives policy filtering).
        enable_logging: When True (default), attach a DecisionLogger at the
            `logging.log_path` from the YAML if one is configured. Set to
            False for tests or when the caller wants to manage logging.

    Returns:
        An AutoPilot ready to select policies and record outcomes.
    """

    parsed = load_autopilot_yaml(path)
    decision_logger = (
        DecisionLogger(parsed.log_path) if (enable_logging and parsed.log_path) else None
    )
    return AutoPilot(
        hardware=hardware,
        config=parsed.autopilot,
        reward_weights=parsed.reward_weights,
        decision_logger=decision_logger,
        decision_tree_thresholds=parsed.decision_tree_thresholds,
    )
