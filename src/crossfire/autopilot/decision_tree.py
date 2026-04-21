"""Decision-tree execution policy selector for CROSSFIRE-X AutoPilot.

The unified spec defines AutoPilot as "not ML-based -- a decision tree
informed by prompt characteristics and hardware state." This module
implements that deterministic selector.

For the adaptive UCB1/Thompson bandit alternative, see autopilot.py.
Both engines share the same ExecutionPolicy enum and PolicyConfig registry.
"""

from __future__ import annotations

from dataclasses import dataclass

from crossfire.autopilot.policy import ExecutionPolicy

# ── Threshold constants (from unified spec) ──────────────────────────────────

# Prompt / output length cutoffs (tokens)
SHORT_TOKEN_THRESHOLD: int = 512
LONG_CONTEXT_THRESHOLD: int = 8192

# Model size requiring weight compression to fit in node VRAM (GB)
# RTX 5090 has 32 GB; Gemma 4 31B at Q8_0 is ~33 GB -- needs TQ4_1S (~23 GB)
# to fit single-node, or distributed split to span nodes.
MEMORY_THRESHOLD_GB: float = 32.0

# Node memory for MoE slot-bank decision (Mac Studio unified memory)
NODE_MEMORY_GB: float = 64.0


@dataclass(frozen=True)
class DecisionTreeThresholds:
    """Tunable thresholds for the deterministic decision-tree selector.

    Defaults come from the unified spec constants above. Callers may override
    them (typically via `configs/autopilot.yaml`) after Phase 2 calibration.
    """

    short_token_threshold: int = SHORT_TOKEN_THRESHOLD
    long_context_threshold: int = LONG_CONTEXT_THRESHOLD
    memory_threshold_gb: float = MEMORY_THRESHOLD_GB
    node_memory_gb: float = NODE_MEMORY_GB

    def __post_init__(self) -> None:
        if self.short_token_threshold <= 0:
            msg = "short_token_threshold must be positive"
            raise ValueError(msg)
        if self.long_context_threshold <= 0:
            msg = "long_context_threshold must be positive"
            raise ValueError(msg)
        if self.memory_threshold_gb <= 0:
            msg = "memory_threshold_gb must be positive"
            raise ValueError(msg)
        if self.node_memory_gb <= 0:
            msg = "node_memory_gb must be positive"
            raise ValueError(msg)


DEFAULT_DECISION_TREE_THRESHOLDS = DecisionTreeThresholds()


@dataclass(frozen=True)
class DecisionContext:
    """Request context used by the decision-tree policy selector.

    Unlike QueryFeatures (which feeds the bandit classifier), DecisionContext
    captures the hardware-aware attributes needed by the deterministic
    decision tree.

    Attributes:
        prompt_len: Number of tokens in the prompt.
        output_len: Requested maximum generation length in tokens.
        context_len: Total context window currently in use (tokens).
        model_size_gb: Model size in GB (as loaded -- Q8_0 or TQ4_1S).
        model_is_moe: True if the model is a Mixture-of-Experts architecture
            (e.g., Gemma 4 26B-A4B). Triggers P6 / Flash-MoE routing.
        decode_is_bottleneck: True if decode throughput (not prefill) is
            the limiting factor for this request class. Enables P2 (ANE
            speculative draft).
        node_memory_gb: Available node memory in GB. Defaults to 64 GB
            (Mac Studio M4 Max unified memory).
    """

    prompt_len: int
    output_len: int
    context_len: int
    model_size_gb: float
    model_is_moe: bool = False
    decode_is_bottleneck: bool = False
    node_memory_gb: float = NODE_MEMORY_GB

    def __post_init__(self) -> None:
        if self.prompt_len < 0:
            msg = "prompt_len must be non-negative"
            raise ValueError(msg)
        if self.output_len < 0:
            msg = "output_len must be non-negative"
            raise ValueError(msg)
        if self.context_len < 0:
            msg = "context_len must be non-negative"
            raise ValueError(msg)
        if self.model_size_gb <= 0:
            msg = "model_size_gb must be positive"
            raise ValueError(msg)
        if self.node_memory_gb <= 0:
            msg = "node_memory_gb must be positive"
            raise ValueError(msg)


def select_policy(
    ctx: DecisionContext,
    *,
    thresholds: DecisionTreeThresholds = DEFAULT_DECISION_TREE_THRESHOLDS,
) -> ExecutionPolicy:
    """Select the execution policy for a single request.

    Implements the decision tree from the CROSSFIRE-X unified spec
    (Section 9.2). Priority order:

        1. MoE model exceeding node memory → P6 (Flash-MoE slot-bank)
        2. Short prompt + short output → P1 (simple EXO split)
        3. Long context → P4 (TriAttention KV compression)
        4. Large model → P3 (TQ4_1S weight compression)
        5. Decode is bottleneck → P2 (ANE speculative draft)
        6. Default → P5 (full stack)

    Args:
        ctx: The request context with prompt, model, and hardware state.
        thresholds: Tunable decision-tree thresholds (defaults to unified
            spec constants). Override via `configs/autopilot.yaml`.

    Returns:
        The selected ExecutionPolicy for this request.
    """
    # P6: MoE model that won't fit in resident GPU-bank mode
    if ctx.model_is_moe and ctx.model_size_gb > ctx.node_memory_gb:
        return ExecutionPolicy.P6

    # P1: Cheap request -- overhead of full stack outweighs benefit
    if (
        ctx.prompt_len < thresholds.short_token_threshold
        and ctx.output_len < thresholds.short_token_threshold
    ):
        return ExecutionPolicy.P1

    # P4: Long context -- KV cache growth needs TriAttention compression
    if ctx.context_len > thresholds.long_context_threshold:
        return ExecutionPolicy.P4

    # P3: Model too large for node VRAM without weight compression
    if ctx.model_size_gb > thresholds.memory_threshold_gb:
        return ExecutionPolicy.P3

    # P2: Decode is the bottleneck -- ANE speculative draft helps
    if ctx.decode_is_bottleneck:
        return ExecutionPolicy.P2

    # P5: Default -- run the full stack
    return ExecutionPolicy.P5
