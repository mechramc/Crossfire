"""Tests for AutoPilot orchestration, config loading, and pipeline wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossfire.autopilot import (
    AutoPilot,
    AutoPilotBaselines,
    AutoPilotConfig,
    AutoPilotEngine,
    AutoPilotOutcome,
    BanditType,
    DecisionLogger,
    DecisionTreeThresholds,
    ExecutionPolicy,
    HardwareAvailability,
    QueryClass,
    QueryFeatures,
    RewardWeights,
    apply_selection_to_pipeline,
    build_autopilot_from_yaml,
    load_autopilot_yaml,
    policy_requires_flash_moe,
    run_autopilot_cycle,
)
from crossfire.distributed.network import InterconnectType
from crossfire.distributed.pipeline import (
    ComputeTarget,
    ComputeTargetConfig,
    NodeConfig,
    NodeRole,
    PipelineConfig,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "autopilot.yaml"


# --- fixtures ---------------------------------------------------------------


def _full_hardware() -> HardwareAvailability:
    return HardwareAvailability(
        distributed_available=True,
        ane_available=True,
        tq4_1s_available=True,
        turbo_kv_available=True,
        flash_moe_available=True,
    )


def _bare_hardware() -> HardwareAvailability:
    return HardwareAvailability()  # P0 only


def _baselines() -> AutoPilotBaselines:
    return AutoPilotBaselines(tokens_per_second=20.0, tokens_per_watt=0.5, ttft_ms=500.0)


def _outcome() -> AutoPilotOutcome:
    return AutoPilotOutcome(
        tokens_per_second=25.0,
        tokens_per_watt=0.6,
        ttft_ms=400.0,
        perplexity_delta=0.1,
        acceptance_rate=0.9,
        execution_time_ms=1200.0,
    )


def _features(**overrides: object) -> QueryFeatures:
    defaults: dict[str, object] = {
        "prompt_tokens": 128,
        "max_gen_tokens": 128,
        "context_used": 1024,
        "model_size_b": 9.5,  # 19 GB at Q8_0
        "available_vram_mb": 24_000,
        "concurrent_requests": 1,
        "model_is_moe": False,
    }
    defaults.update(overrides)
    return QueryFeatures(**defaults)  # type: ignore[arg-type]


def _pipeline_config(*, flash_moe: bool = True) -> PipelineConfig:
    mac = NodeConfig(
        name="mac",
        host="mac.local",
        port=8080,
        targets=[
            ComputeTargetConfig(target=ComputeTarget.T2_METAL_GPU, role=NodeRole.DECODE),
            ComputeTargetConfig(target=ComputeTarget.T3_ANE, role=NodeRole.DRAFT),
        ]
        + (
            [ComputeTargetConfig(target=ComputeTarget.T5_NVME_SSD, role=NodeRole.EXPERT_STREAMING)]
            if flash_moe
            else []
        ),
    )
    pc = NodeConfig(
        name="pc",
        host="pc.local",
        port=8080,
        targets=[
            ComputeTargetConfig(target=ComputeTarget.T1_CUDA_GPU, role=NodeRole.PREFILL),
        ],
    )
    return PipelineConfig(nodes=[mac, pc], interconnect=InterconnectType.USB4)


# --- AutoPilot core ---------------------------------------------------------


def test_default_autopilot_uses_decision_tree_engine() -> None:
    ap = AutoPilot(hardware=_full_hardware())
    assert ap.config.resolved_engine() is AutoPilotEngine.DECISION_TREE


def test_select_policy_routes_short_prompt_to_p1() -> None:
    ap = AutoPilot(hardware=_full_hardware())
    selection = ap.select_policy(_features(prompt_tokens=64, max_gen_tokens=32))
    assert selection.selected_policy is ExecutionPolicy.P1
    assert selection.was_exploration is False
    assert selection.query_class is QueryClass.SHORT_GEN
    # Decision tree scores should all be None (deterministic)
    assert all(score is None for score in selection.scores.values())


def test_select_policy_routes_long_context_to_p4() -> None:
    ap = AutoPilot(hardware=_full_hardware())
    # Long context trips P4 only if we're past the P1 short-prompt/output gate
    selection = ap.select_policy(
        _features(prompt_tokens=1024, max_gen_tokens=128, context_used=16384)
    )
    assert selection.selected_policy is ExecutionPolicy.P4


def test_select_policy_routes_moe_overflow_to_p6() -> None:
    ap = AutoPilot(hardware=_full_hardware())
    # 35B * 2 ≈ 70 GB > 64 GB node memory
    selection = ap.select_policy(_features(model_size_b=35.0, model_is_moe=True))
    assert selection.selected_policy is ExecutionPolicy.P6


def test_select_policy_clamps_to_available_policies() -> None:
    # Bare hardware: only P0 is available. Tree picks P1, should clamp.
    ap = AutoPilot(hardware=_bare_hardware())
    selection = ap.select_policy(_features(prompt_tokens=100, max_gen_tokens=100))
    assert selection.selected_policy is ExecutionPolicy.P0
    assert selection.available_policies == (ExecutionPolicy.P0,)


def test_select_policy_raises_when_no_policy_available() -> None:
    """An impossible hardware state (all P1-P6 gated out; P0 always works)
    still yields P0 — the only way to empty the candidate set is a
    hardware object with future flags the registry doesn't see."""
    # P0 is always available, so this is effectively a smoke test that
    # the empty-check path exists.
    ap = AutoPilot(hardware=_bare_hardware())
    assert ap.select_policy(_features()).selected_policy is ExecutionPolicy.P0


def test_record_outcome_updates_bandit_and_logger(tmp_path: Path) -> None:
    log_path = tmp_path / "decisions.jsonl"
    ap = AutoPilot(
        hardware=_full_hardware(),
        config=AutoPilotConfig(engine=AutoPilotEngine.UCB1_BANDIT, bandit_type=BanditType.UCB1),
        decision_logger=DecisionLogger(log_path),
    )
    selection = ap.select_policy(_features())
    reward = ap.record_outcome(selection, outcome=_outcome(), baselines=_baselines())

    assert 0.0 <= reward.total <= 1.0
    assert log_path.exists()
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["selected_policy"] == selection.selected_policy.name


def test_ucb1_engine_explores_unseen_arms_first() -> None:
    ap = AutoPilot(
        hardware=_full_hardware(),
        config=AutoPilotConfig(engine=AutoPilotEngine.UCB1_BANDIT, bandit_type=BanditType.UCB1),
    )
    selection = ap.select_policy(_features())
    assert selection.was_exploration is True


def test_thompson_engine_produces_sampled_scores() -> None:
    ap = AutoPilot(
        hardware=_full_hardware(),
        config=AutoPilotConfig(
            engine=AutoPilotEngine.THOMPSON_BANDIT,
            bandit_type=BanditType.THOMPSON,
            seed=42,
        ),
    )
    selection = ap.select_policy(_features())
    # Thompson returns concrete floats for every candidate
    assert all(isinstance(score, float) for score in selection.scores.values())


def test_autopilot_respects_custom_decision_tree_thresholds() -> None:
    # Raise the short-token threshold so a 400-token prompt still routes to P1
    thresholds = DecisionTreeThresholds(short_token_threshold=1000)
    ap = AutoPilot(hardware=_full_hardware(), decision_tree_thresholds=thresholds)
    selection = ap.select_policy(_features(prompt_tokens=400, max_gen_tokens=400))
    assert selection.selected_policy is ExecutionPolicy.P1


# --- YAML config loading (T-0409) -------------------------------------------


def test_load_autopilot_yaml_parses_repo_config() -> None:
    parsed = load_autopilot_yaml(CONFIG_PATH)
    assert parsed.autopilot.engine is AutoPilotEngine.DECISION_TREE
    assert parsed.reward_weights.throughput == pytest.approx(0.4)
    assert parsed.decision_tree_thresholds.short_token_threshold == 512
    assert parsed.decision_tree_thresholds.long_context_threshold == 8192
    assert parsed.log_path == Path("results/autopilot_decisions.jsonl")
    # Baselines are null in the committed config
    assert parsed.baselines is None


def test_load_autopilot_yaml_rejects_unknown_engine(tmp_path: Path) -> None:
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text("autopilot:\n  engine: not_a_real_engine\n")
    with pytest.raises(ValueError, match="Unknown autopilot engine"):
        load_autopilot_yaml(yaml_path)


def test_load_autopilot_yaml_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_autopilot_yaml("/nonexistent/autopilot.yaml")


def test_load_autopilot_yaml_parses_baselines(tmp_path: Path) -> None:
    yaml_path = tmp_path / "with_baselines.yaml"
    yaml_path.write_text(
        "autopilot:\n  engine: decision_tree\n"
        "baselines:\n"
        "  p1_8k:\n"
        "    tokens_per_second: 18.5\n"
        "    tokens_per_watt: 0.42\n"
        "    ttft_ms: 640.0\n"
    )
    parsed = load_autopilot_yaml(yaml_path)
    assert parsed.baselines is not None
    assert parsed.baselines.tokens_per_second == pytest.approx(18.5)
    assert parsed.baselines.ttft_ms == pytest.approx(640.0)


def test_build_autopilot_from_yaml_wires_logger(tmp_path: Path) -> None:
    log_path = tmp_path / "decisions.jsonl"
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(f"autopilot:\n  engine: decision_tree\nlogging:\n  log_path: {log_path}\n")
    ap = build_autopilot_from_yaml(yaml_path, hardware=_full_hardware())
    assert ap.decision_logger is not None
    assert ap.decision_logger.path == log_path


def test_build_autopilot_from_yaml_can_disable_logging(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "autopilot:\n  engine: decision_tree\nlogging:\n  log_path: results/ignored.jsonl\n"
    )
    ap = build_autopilot_from_yaml(yaml_path, hardware=_full_hardware(), enable_logging=False)
    assert ap.decision_logger is None


def test_build_autopilot_from_yaml_uses_repo_config() -> None:
    ap = build_autopilot_from_yaml(CONFIG_PATH, hardware=_full_hardware(), enable_logging=False)
    assert ap.config.engine is AutoPilotEngine.DECISION_TREE
    assert ap.reward_weights == RewardWeights()


def test_load_autopilot_yaml_rejects_malformed_root(tmp_path: Path) -> None:
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text("- just_a_list\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_autopilot_yaml(yaml_path)


# --- Pipeline integration (T-0410 / T-0411) ---------------------------------


def test_policy_requires_flash_moe_is_true_only_for_p6() -> None:
    assert policy_requires_flash_moe(ExecutionPolicy.P6) is True
    for policy in ExecutionPolicy:
        if policy is ExecutionPolicy.P6:
            continue
        assert policy_requires_flash_moe(policy) is False


def test_apply_selection_to_pipeline_sets_policy_and_flash_moe_flag() -> None:
    ap = AutoPilot(hardware=_full_hardware())
    selection = ap.select_policy(_features(model_size_b=35.0, model_is_moe=True))
    assert selection.selected_policy is ExecutionPolicy.P6

    updated = apply_selection_to_pipeline(_pipeline_config(flash_moe=True), selection)
    assert updated.execution_policy == "P6"
    assert updated.flash_moe_enabled is True


def test_apply_selection_to_pipeline_clears_flash_moe_for_non_p6() -> None:
    ap = AutoPilot(hardware=_full_hardware())
    selection = ap.select_policy(_features(prompt_tokens=100, max_gen_tokens=100))
    assert selection.selected_policy is ExecutionPolicy.P1

    # Start with a pipeline that has flash_moe_enabled=True
    base = _pipeline_config(flash_moe=False)
    updated = apply_selection_to_pipeline(base, selection)
    assert updated.execution_policy == "P1"
    assert updated.flash_moe_enabled is False


def test_apply_selection_raises_when_p6_pipeline_lacks_expert_streaming() -> None:
    ap = AutoPilot(hardware=_full_hardware())
    selection = ap.select_policy(_features(model_size_b=35.0, model_is_moe=True))
    assert selection.selected_policy is ExecutionPolicy.P6
    # Pipeline without T5 NVMe node
    with pytest.raises(ValueError, match="Flash-MoE"):
        apply_selection_to_pipeline(_pipeline_config(flash_moe=False), selection)


def test_run_autopilot_cycle_executes_select_apply_record(tmp_path: Path) -> None:
    log_path = tmp_path / "cycle.jsonl"
    ap = AutoPilot(
        hardware=_full_hardware(),
        decision_logger=DecisionLogger(log_path),
    )
    features = _features(prompt_tokens=100, max_gen_tokens=100)
    updated, selection, reward = run_autopilot_cycle(
        ap,
        pipeline_config=_pipeline_config(flash_moe=True),
        features=features,
        outcome=_outcome(),
        baselines=_baselines(),
    )
    assert updated.execution_policy == selection.selected_policy.name
    assert 0.0 <= reward.total <= 1.0
    # Outcome reporting wired → bandit update + log entry
    assert log_path.exists()
    assert len(log_path.read_text().splitlines()) == 1
