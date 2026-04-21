"""Tests for the execution-policy registry and hardware-availability gating."""

from __future__ import annotations

import pytest

from crossfire.autopilot.policy import (
    POLICY_REGISTRY,
    ExecutionPolicy,
    HardwareAvailability,
    PolicyConfig,
    available_policies,
    get_policy_config,
)

# --- Registry shape ---------------------------------------------------------


def test_registry_covers_every_execution_policy() -> None:
    assert set(POLICY_REGISTRY) == set(ExecutionPolicy)


def test_get_policy_config_returns_registered_entry() -> None:
    for policy in ExecutionPolicy:
        config = get_policy_config(policy)
        assert isinstance(config, PolicyConfig)
        assert config.policy is policy


def test_p0_policy_has_no_hardware_requirements() -> None:
    p0 = get_policy_config(ExecutionPolicy.P0)
    assert p0.requires_distributed is False
    assert p0.requires_ane is False
    assert p0.requires_tq4_1s is False
    assert p0.requires_turbo_kv is False
    assert p0.requires_flash_moe is False


def test_p6_policy_requires_flash_moe_and_distributed() -> None:
    p6 = get_policy_config(ExecutionPolicy.P6)
    assert p6.requires_flash_moe is True
    assert p6.requires_distributed is True
    assert p6.uses_flash_moe is True


def test_p5_full_stack_requires_all_non_flash_moe_features() -> None:
    p5 = get_policy_config(ExecutionPolicy.P5)
    assert p5.requires_distributed is True
    assert p5.requires_ane is True
    assert p5.requires_tq4_1s is True
    assert p5.requires_turbo_kv is True
    assert p5.requires_flash_moe is False


# --- is_available gating ----------------------------------------------------


def _hw(**overrides: bool) -> HardwareAvailability:
    return HardwareAvailability(**overrides)


def test_p0_always_available_even_with_no_hardware() -> None:
    assert get_policy_config(ExecutionPolicy.P0).is_available(_hw())


def test_p1_requires_distributed() -> None:
    config = get_policy_config(ExecutionPolicy.P1)
    assert not config.is_available(_hw())
    assert config.is_available(_hw(distributed_available=True))


def test_p2_requires_distributed_and_ane() -> None:
    config = get_policy_config(ExecutionPolicy.P2)
    assert not config.is_available(_hw(distributed_available=True))
    assert not config.is_available(_hw(ane_available=True))
    assert config.is_available(_hw(distributed_available=True, ane_available=True))


def test_p3_requires_distributed_and_tq4_1s() -> None:
    config = get_policy_config(ExecutionPolicy.P3)
    assert not config.is_available(_hw(distributed_available=True))
    assert not config.is_available(_hw(tq4_1s_available=True))
    assert config.is_available(_hw(distributed_available=True, tq4_1s_available=True))


def test_p4_requires_distributed_and_turbo_kv() -> None:
    config = get_policy_config(ExecutionPolicy.P4)
    assert not config.is_available(_hw(distributed_available=True))
    assert not config.is_available(_hw(turbo_kv_available=True))
    assert config.is_available(_hw(distributed_available=True, turbo_kv_available=True))


def test_p5_requires_every_non_flash_moe_feature() -> None:
    config = get_policy_config(ExecutionPolicy.P5)
    full = _hw(
        distributed_available=True,
        ane_available=True,
        tq4_1s_available=True,
        turbo_kv_available=True,
    )
    assert config.is_available(full)
    # Drop each requirement one at a time
    required_flags = (
        "distributed_available",
        "ane_available",
        "tq4_1s_available",
        "turbo_kv_available",
    )
    for field in required_flags:
        overrides = {f: True for f in full.__dict__ if full.__dict__[f]}
        overrides[field] = False
        assert not config.is_available(_hw(**overrides))


def test_p6_requires_distributed_and_flash_moe() -> None:
    config = get_policy_config(ExecutionPolicy.P6)
    assert not config.is_available(_hw(distributed_available=True))
    assert not config.is_available(_hw(flash_moe_available=True))
    assert config.is_available(_hw(distributed_available=True, flash_moe_available=True))


# --- available_policies() ---------------------------------------------------


def test_available_policies_bare_hardware_returns_only_p0() -> None:
    assert available_policies(_hw()) == [ExecutionPolicy.P0]


def test_available_policies_distributed_only_returns_p0_and_p1() -> None:
    assert available_policies(_hw(distributed_available=True)) == [
        ExecutionPolicy.P0,
        ExecutionPolicy.P1,
    ]


def test_available_policies_full_hardware_returns_every_policy() -> None:
    full = _hw(
        distributed_available=True,
        ane_available=True,
        tq4_1s_available=True,
        turbo_kv_available=True,
        flash_moe_available=True,
    )
    assert available_policies(full) == list(ExecutionPolicy)


def test_available_policies_preserves_registry_ordering() -> None:
    policies = available_policies(
        _hw(distributed_available=True, ane_available=True, turbo_kv_available=True)
    )
    # Should be ordered P0, P1, P2, P4 (P3/P5/P6 gated out); check monotonic
    values = [p.value for p in policies]
    assert values == sorted(values)


@pytest.mark.parametrize(
    "policy,required_flag",
    [
        (ExecutionPolicy.P1, "distributed_available"),
        (ExecutionPolicy.P2, "ane_available"),
        (ExecutionPolicy.P3, "tq4_1s_available"),
        (ExecutionPolicy.P4, "turbo_kv_available"),
        (ExecutionPolicy.P6, "flash_moe_available"),
    ],
)
def test_disabling_required_flag_removes_policy(
    policy: ExecutionPolicy, required_flag: str
) -> None:
    full = {
        "distributed_available": True,
        "ane_available": True,
        "tq4_1s_available": True,
        "turbo_kv_available": True,
        "flash_moe_available": True,
    }
    full[required_flag] = False
    assert policy not in available_policies(_hw(**full))
