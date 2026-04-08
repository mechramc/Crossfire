"""Execution-policy registry for CROSSFIRE-X AutoPilot."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExecutionPolicy(Enum):
    """Execution policies available to AutoPilot."""

    P0 = "p0"
    P1 = "p1"
    P2 = "p2"
    P3 = "p3"
    P4 = "p4"
    P5 = "p5"


@dataclass(frozen=True)
class HardwareAvailability:
    """Runtime hardware and artifact availability for policy filtering."""

    rdma_available: bool = False
    ane_available: bool = False
    tq4_1s_available: bool = False
    turbo_kv_available: bool = False


@dataclass(frozen=True)
class PolicyConfig:
    """Definition of a single execution policy."""

    policy: ExecutionPolicy
    description: str
    distributed: bool
    uses_ane: bool
    uses_tq4_1s: bool
    uses_turbo_kv: bool
    requires_rdma: bool = False
    requires_ane: bool = False
    requires_tq4_1s: bool = False
    requires_turbo_kv: bool = False

    def is_available(self, hardware: HardwareAvailability) -> bool:
        """Return whether the policy is runnable with the current hardware."""

        if self.requires_rdma and not hardware.rdma_available:
            return False
        if self.requires_ane and not hardware.ane_available:
            return False
        if self.requires_tq4_1s and not hardware.tq4_1s_available:
            return False
        return not (self.requires_turbo_kv and not hardware.turbo_kv_available)


POLICY_REGISTRY: dict[ExecutionPolicy, PolicyConfig] = {
    ExecutionPolicy.P0: PolicyConfig(
        policy=ExecutionPolicy.P0,
        description="Single best node fallback with no distributed overhead",
        distributed=False,
        uses_ane=False,
        uses_tq4_1s=False,
        uses_turbo_kv=False,
    ),
    ExecutionPolicy.P1: PolicyConfig(
        policy=ExecutionPolicy.P1,
        description="Distributed baseline with 5090 prefill and Mac decode",
        distributed=True,
        uses_ane=False,
        uses_tq4_1s=False,
        uses_turbo_kv=False,
        requires_rdma=True,
    ),
    ExecutionPolicy.P2: PolicyConfig(
        policy=ExecutionPolicy.P2,
        description="Distributed baseline plus ANE speculative decode",
        distributed=True,
        uses_ane=True,
        uses_tq4_1s=False,
        uses_turbo_kv=False,
        requires_rdma=True,
        requires_ane=True,
    ),
    ExecutionPolicy.P3: PolicyConfig(
        policy=ExecutionPolicy.P3,
        description="Distributed baseline plus TQ4_1S compressed weights",
        distributed=True,
        uses_ane=False,
        uses_tq4_1s=True,
        uses_turbo_kv=False,
        requires_rdma=True,
        requires_tq4_1s=True,
    ),
    ExecutionPolicy.P4: PolicyConfig(
        policy=ExecutionPolicy.P4,
        description="Distributed compressed run with turbo KV cache",
        distributed=True,
        uses_ane=False,
        uses_tq4_1s=True,
        uses_turbo_kv=True,
        requires_rdma=True,
        requires_tq4_1s=True,
        requires_turbo_kv=True,
    ),
    ExecutionPolicy.P5: PolicyConfig(
        policy=ExecutionPolicy.P5,
        description="Full-stack policy with RDMA, ANE, TQ4_1S, and turbo KV",
        distributed=True,
        uses_ane=True,
        uses_tq4_1s=True,
        uses_turbo_kv=True,
        requires_rdma=True,
        requires_ane=True,
        requires_tq4_1s=True,
        requires_turbo_kv=True,
    ),
}


def get_policy_config(policy: ExecutionPolicy) -> PolicyConfig:
    """Look up configuration for an execution policy."""

    return POLICY_REGISTRY[policy]


def available_policies(hardware: HardwareAvailability) -> list[ExecutionPolicy]:
    """Return policies available for the current hardware state."""

    return [policy for policy, config in POLICY_REGISTRY.items() if config.is_available(hardware)]
