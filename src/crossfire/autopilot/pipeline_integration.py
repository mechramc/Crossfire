"""Bridge between AutoPilot selections and PipelineConfig.

Provides the glue for the three integration points called out in the
task ledger:

    T-0410: Wire AutoPilot selection into pipeline execution.
    T-0411: Wire outcome reporting back into reward + bandit updates.

The pipeline receives a plain-string execution_policy (e.g. "P6") plus a
flash_moe_enabled flag. AutoPilot returns a typed AutoPilotSelection. This
module maps the latter onto the former without importing AutoPilot into
the pipeline package (keeps distributed free of autopilot as a dependency).
"""

from __future__ import annotations

from dataclasses import replace

from crossfire.autopilot.autopilot import (
    AutoPilot,
    AutoPilotBaselines,
    AutoPilotOutcome,
    AutoPilotSelection,
)
from crossfire.autopilot.policy import ExecutionPolicy
from crossfire.autopilot.query_classifier import QueryFeatures
from crossfire.autopilot.reward import RewardBreakdown
from crossfire.distributed.pipeline import PipelineConfig


def policy_requires_flash_moe(policy: ExecutionPolicy) -> bool:
    """Return True if `policy` needs the Flash-MoE slot-bank runtime."""

    return policy is ExecutionPolicy.P6


def apply_selection_to_pipeline(
    config: PipelineConfig,
    selection: AutoPilotSelection,
) -> PipelineConfig:
    """Return a PipelineConfig reflecting the AutoPilot selection.

    Sets `execution_policy` to the policy name (e.g. "P6") and toggles
    `flash_moe_enabled` for Flash-MoE policies. Other pipeline fields
    (nodes, interconnect, speculative_decode) are left untouched.

    Raises ValueError if the resulting config fails PipelineConfig.validate.
    """

    updated = replace(
        config,
        execution_policy=selection.selected_policy.name,
        flash_moe_enabled=policy_requires_flash_moe(selection.selected_policy),
    )
    updated.validate()
    return updated


def run_autopilot_cycle(
    autopilot: AutoPilot,
    *,
    pipeline_config: PipelineConfig,
    features: QueryFeatures,
    outcome: AutoPilotOutcome,
    baselines: AutoPilotBaselines,
) -> tuple[PipelineConfig, AutoPilotSelection, RewardBreakdown]:
    """Execute one full select → apply → record AutoPilot cycle.

    Intended for the benchmark harness and the end-to-end integration test.
    The caller is responsible for actually running the selected pipeline
    between `apply_selection_to_pipeline` and `record_outcome`; this helper
    wraps both in one call when the outcome is known up front (e.g. in
    tests or replays from logged runs).

    Returns the updated PipelineConfig, the AutoPilotSelection, and the
    RewardBreakdown computed from the outcome.
    """

    selection = autopilot.select_policy(features)
    applied = apply_selection_to_pipeline(pipeline_config, selection)
    reward = autopilot.record_outcome(selection, outcome=outcome, baselines=baselines)
    return applied, selection, reward
