"""Execution-policy registry for CROSSFIRE-X AutoPilot.

Policies follow crossfire_x_final.docx Section 9. All distributed policies
(P1-P6) run over TCP/IP between nodes -- the physical link is USB4 at
40 Gbps in production and WiFi/5GbE during bring-up. No policy requires
RDMA; composed TriAttention + TurboQuant compression (6.8x KV reduction)
makes the TCP/IP bandwidth envelope sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExecutionPolicy(Enum):
    """Execution policies available to AutoPilot.

    P0-P5 operate on dense models. P6 is the Flash-MoE slot-bank policy
    for MoE models (Gemma 4 26B-A4B, Orion Forge) that exceed node memory.
    """

    P0 = "p0"
    P1 = "p1"
    P2 = "p2"
    P3 = "p3"
    P4 = "p4"
    P5 = "p5"
    P6 = "p6"


@dataclass(frozen=True)
class HardwareAvailability:
    """Runtime hardware and artifact availability for policy filtering."""

    distributed_available: bool = False  # EXO peer reachable over TCP/IP
    ane_available: bool = False
    tq4_1s_available: bool = False
    turbo_kv_available: bool = False
    flash_moe_available: bool = False  # anemll-flash-llama.cpp built and verified


@dataclass(frozen=True)
class PolicyConfig:
    """Definition of a single execution policy."""

    policy: ExecutionPolicy
    description: str
    distributed: bool
    uses_ane: bool
    uses_tq4_1s: bool
    uses_turbo_kv: bool
    uses_flash_moe: bool = False
    requires_distributed: bool = False
    requires_ane: bool = False
    requires_tq4_1s: bool = False
    requires_turbo_kv: bool = False
    requires_flash_moe: bool = False

    def is_available(self, hardware: HardwareAvailability) -> bool:
        """Return whether the policy is runnable with the current hardware."""

        if self.requires_distributed and not hardware.distributed_available:
            return False
        if self.requires_ane and not hardware.ane_available:
            return False
        if self.requires_tq4_1s and not hardware.tq4_1s_available:
            return False
        if self.requires_turbo_kv and not hardware.turbo_kv_available:
            return False
        return not (self.requires_flash_moe and not hardware.flash_moe_available)


POLICY_REGISTRY: dict[ExecutionPolicy, PolicyConfig] = {
    ExecutionPolicy.P0: PolicyConfig(
        policy=ExecutionPolicy.P0,
        description="Single-node 5090 only; short prompt and short output, model fits in 32 GB",
        distributed=False,
        uses_ane=False,
        uses_tq4_1s=False,
        uses_turbo_kv=False,
    ),
    ExecutionPolicy.P1: PolicyConfig(
        policy=ExecutionPolicy.P1,
        description="EXO split (5090 prefill + Mac decode) over TCP/IP",
        distributed=True,
        uses_ane=False,
        uses_tq4_1s=False,
        uses_turbo_kv=False,
        requires_distributed=True,
    ),
    ExecutionPolicy.P2: PolicyConfig(
        policy=ExecutionPolicy.P2,
        description="EXO split plus ANE speculative draft (decode bottleneck)",
        distributed=True,
        uses_ane=True,
        uses_tq4_1s=False,
        uses_turbo_kv=False,
        requires_distributed=True,
        requires_ane=True,
    ),
    ExecutionPolicy.P3: PolicyConfig(
        policy=ExecutionPolicy.P3,
        description="EXO split plus TQ4_1S compressed weights (reduce cross-node transfer)",
        distributed=True,
        uses_ane=False,
        uses_tq4_1s=True,
        uses_turbo_kv=False,
        requires_distributed=True,
        requires_tq4_1s=True,
    ),
    ExecutionPolicy.P4: PolicyConfig(
        policy=ExecutionPolicy.P4,
        description="EXO split plus TriAttention KV compression (long context, >8K)",
        distributed=True,
        uses_ane=False,
        uses_tq4_1s=False,
        uses_turbo_kv=True,
        requires_distributed=True,
        requires_turbo_kv=True,
    ),
    ExecutionPolicy.P5: PolicyConfig(
        policy=ExecutionPolicy.P5,
        description="Full stack: EXO + ANE + TQ4_1S + TriAttention (maximum capability)",
        distributed=True,
        uses_ane=True,
        uses_tq4_1s=True,
        uses_turbo_kv=True,
        requires_distributed=True,
        requires_ane=True,
        requires_tq4_1s=True,
        requires_turbo_kv=True,
    ),
    ExecutionPolicy.P6: PolicyConfig(
        policy=ExecutionPolicy.P6,
        description=(
            "Flash-MoE slot-bank + EXO for MoE models exceeding node memory; "
            "experts streamed from NVMe SSD (T5)"
        ),
        distributed=True,
        uses_ane=False,
        uses_tq4_1s=False,
        uses_turbo_kv=False,
        uses_flash_moe=True,
        requires_distributed=True,
        requires_flash_moe=True,
    ),
}


def get_policy_config(policy: ExecutionPolicy) -> PolicyConfig:
    """Look up configuration for an execution policy."""

    return POLICY_REGISTRY[policy]


def available_policies(hardware: HardwareAvailability) -> list[ExecutionPolicy]:
    """Return policies available for the current hardware state."""

    return [policy for policy, config in POLICY_REGISTRY.items() if config.is_available(hardware)]
