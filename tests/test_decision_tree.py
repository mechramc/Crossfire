"""Tests for the deterministic AutoPilot decision-tree policy selector."""

from __future__ import annotations

import pytest

from crossfire.autopilot.decision_tree import (
    LONG_CONTEXT_THRESHOLD,
    MEMORY_THRESHOLD_GB,
    SHORT_TOKEN_THRESHOLD,
    DecisionContext,
    select_policy,
)
from crossfire.autopilot.policy import ExecutionPolicy

# --- DecisionContext validation ----------------------------------------------


def _valid_ctx(**overrides: object) -> DecisionContext:
    defaults: dict[str, object] = {
        "prompt_len": 128,
        "output_len": 128,
        "context_len": 1024,
        "model_size_gb": 19.0,
        "model_is_moe": False,
        "decode_is_bottleneck": False,
    }
    defaults.update(overrides)
    return DecisionContext(**defaults)  # type: ignore[arg-type]


def test_decision_context_defaults():
    """Defaults: non-MoE, decode not bottlenecked, 64 GB node memory."""
    ctx = _valid_ctx()
    assert ctx.model_is_moe is False
    assert ctx.decode_is_bottleneck is False
    assert ctx.node_memory_gb == 64.0


@pytest.mark.parametrize("field", ["prompt_len", "output_len", "context_len"])
def test_decision_context_rejects_negative_lengths(field: str):
    with pytest.raises(ValueError, match=f"{field} must be non-negative"):
        _valid_ctx(**{field: -1})


def test_decision_context_rejects_non_positive_model_size():
    with pytest.raises(ValueError, match="model_size_gb must be positive"):
        _valid_ctx(model_size_gb=0.0)


def test_decision_context_rejects_non_positive_node_memory():
    with pytest.raises(ValueError, match="node_memory_gb must be positive"):
        _valid_ctx(node_memory_gb=0.0)


# --- Priority 1: P6 (MoE exceeding node memory) -------------------------------


def test_moe_exceeding_node_memory_routes_to_p6():
    """MoE model larger than node memory -> P6 (Flash-MoE slot-bank)."""
    ctx = _valid_ctx(model_is_moe=True, model_size_gb=120.0, node_memory_gb=64.0)
    assert select_policy(ctx) is ExecutionPolicy.P6


def test_moe_fitting_in_node_memory_does_not_route_to_p6():
    """MoE model that fits falls through to later rules, not P6."""
    ctx = _valid_ctx(
        model_is_moe=True,
        model_size_gb=20.0,
        node_memory_gb=64.0,
        prompt_len=128,
        output_len=128,
    )
    assert select_policy(ctx) is not ExecutionPolicy.P6


def test_non_moe_over_node_memory_does_not_route_to_p6():
    """Dense model, even if huge, never goes to P6."""
    ctx = _valid_ctx(model_is_moe=False, model_size_gb=120.0, node_memory_gb=64.0)
    assert select_policy(ctx) is not ExecutionPolicy.P6


# --- Priority 2: P1 (short prompt + short output) ----------------------------


def test_short_prompt_and_short_output_routes_to_p1():
    """Cheap request -> simple EXO split (P1)."""
    ctx = _valid_ctx(prompt_len=64, output_len=64)
    assert select_policy(ctx) is ExecutionPolicy.P1


def test_p1_boundary_short_token_threshold():
    """Exactly at the threshold does NOT count as short (strict `<`)."""
    ctx = _valid_ctx(
        prompt_len=SHORT_TOKEN_THRESHOLD,
        output_len=SHORT_TOKEN_THRESHOLD,
    )
    assert select_policy(ctx) is not ExecutionPolicy.P1


def test_long_prompt_short_output_does_not_route_to_p1():
    """One side being long disqualifies P1."""
    ctx = _valid_ctx(prompt_len=SHORT_TOKEN_THRESHOLD + 1, output_len=64)
    assert select_policy(ctx) is not ExecutionPolicy.P1


# --- Priority 3: P4 (long context, TriAttention) -----------------------------


def test_long_context_routes_to_p4():
    """context_len > 8192 -> P4 (TriAttention KV compression)."""
    ctx = _valid_ctx(
        prompt_len=1024,
        output_len=1024,
        context_len=LONG_CONTEXT_THRESHOLD + 1,
    )
    assert select_policy(ctx) is ExecutionPolicy.P4


def test_p4_boundary_long_context_threshold():
    """Exactly at the threshold is NOT long (strict `>`)."""
    ctx = _valid_ctx(
        prompt_len=1024,
        output_len=1024,
        context_len=LONG_CONTEXT_THRESHOLD,
    )
    assert select_policy(ctx) is not ExecutionPolicy.P4


# --- Priority 4: P3 (large model, TQ4_1S) ------------------------------------


def test_large_model_routes_to_p3():
    """model_size_gb > 32 -> P3 (TQ4_1S weight compression)."""
    ctx = _valid_ctx(
        prompt_len=1024,
        output_len=1024,
        context_len=1024,
        model_size_gb=MEMORY_THRESHOLD_GB + 1.0,
    )
    assert select_policy(ctx) is ExecutionPolicy.P3


def test_p3_boundary_memory_threshold():
    """Exactly at the threshold is NOT oversized (strict `>`)."""
    ctx = _valid_ctx(
        prompt_len=1024,
        output_len=1024,
        context_len=1024,
        model_size_gb=MEMORY_THRESHOLD_GB,
    )
    assert select_policy(ctx) is not ExecutionPolicy.P3


# --- Priority 5: P2 (decode-bound, ANE speculative) --------------------------


def test_decode_bottleneck_routes_to_p2():
    """decode_is_bottleneck=True and no earlier rule fires -> P2."""
    ctx = _valid_ctx(
        prompt_len=1024,
        output_len=1024,
        context_len=1024,
        model_size_gb=19.0,
        decode_is_bottleneck=True,
    )
    assert select_policy(ctx) is ExecutionPolicy.P2


# --- Priority 6: P5 default --------------------------------------------------


def test_default_routes_to_p5():
    """No rule fires -> P5 (full stack default)."""
    ctx = _valid_ctx(
        prompt_len=1024,
        output_len=1024,
        context_len=1024,
        model_size_gb=19.0,
        model_is_moe=False,
        decode_is_bottleneck=False,
    )
    assert select_policy(ctx) is ExecutionPolicy.P5


# --- Priority ordering -------------------------------------------------------


def test_moe_beats_short_prompt():
    """P6 fires before P1 even when prompt/output are short."""
    ctx = _valid_ctx(
        prompt_len=64,
        output_len=64,
        model_is_moe=True,
        model_size_gb=120.0,
    )
    assert select_policy(ctx) is ExecutionPolicy.P6


def test_short_prompt_beats_long_context():
    """P1 wins over P4 when both could match."""
    ctx = _valid_ctx(
        prompt_len=64,
        output_len=64,
        context_len=LONG_CONTEXT_THRESHOLD + 1,
    )
    assert select_policy(ctx) is ExecutionPolicy.P1


def test_long_context_beats_large_model():
    """P4 wins over P3 when both could match."""
    ctx = _valid_ctx(
        prompt_len=1024,
        output_len=1024,
        context_len=LONG_CONTEXT_THRESHOLD + 1,
        model_size_gb=MEMORY_THRESHOLD_GB + 1.0,
    )
    assert select_policy(ctx) is ExecutionPolicy.P4


def test_large_model_beats_decode_bottleneck():
    """P3 wins over P2 when both could match."""
    ctx = _valid_ctx(
        prompt_len=1024,
        output_len=1024,
        context_len=1024,
        model_size_gb=MEMORY_THRESHOLD_GB + 1.0,
        decode_is_bottleneck=True,
    )
    assert select_policy(ctx) is ExecutionPolicy.P3
