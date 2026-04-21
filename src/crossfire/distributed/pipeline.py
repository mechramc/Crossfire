"""EXO-orchestrated pipeline across heterogeneous compute targets.

Manages the five compute targets in a CROSSFIRE-X pipeline:
  T1 (CUDA GPU)   -- RTX 5090 prefill
  T2 (Metal GPU)  -- M4 Max decode
  T3 (ANE)        -- draft model / speculative decode
  T4 (CPU/SME)    -- scheduling, KV management, speculative verification
  T5 (NVMe SSD)   -- Flash-MoE slot-bank expert streaming (P6 policy)

The interconnect (USB4 / 5GbE / 1GbE / WiFi) is tracked in
distributed.network.InterconnectType; it is not a compute target.

EXO handles topology-aware auto-parallel and disaggregated prefill/decode.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from crossfire.distributed.network import InterconnectType


class ComputeTarget(Enum):
    """Compute targets in the CROSSFIRE-X pipeline.

    Five targets across two machines:
      T1-T2: GPU compute (prefill on 5090, decode on Mac GPU)
      T3-T4: Mac auxiliary (ANE draft, CPU scheduling)
      T5:    NVMe SSD expert streaming (Flash-MoE slot-bank)
    """

    T1_CUDA_GPU = "cuda_gpu"
    T2_METAL_GPU = "metal_gpu"
    T3_ANE = "ane"
    T4_CPU_SME = "cpu_sme"
    T5_NVME_SSD = "nvme_ssd"  # Flash-MoE slot-bank expert streaming


class NodeRole(Enum):
    """Role assignment for each node in the EXO pipeline."""

    PREFILL = "prefill"
    DECODE = "decode"
    DRAFT = "draft"
    EXPERT_STREAMING = "expert_streaming"


@dataclass(frozen=True)
class ComputeTargetConfig:
    """Configuration for a single compute target."""

    target: ComputeTarget
    role: NodeRole
    enabled: bool = True
    power_watts: float | None = None


@dataclass(frozen=True)
class NodeConfig:
    """Configuration for a single machine in the EXO cluster."""

    name: str
    host: str
    port: int
    targets: list[ComputeTargetConfig]


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the EXO-orchestrated distributed pipeline."""

    nodes: list[NodeConfig]
    framework: str = "exo"
    interconnect: InterconnectType = InterconnectType.WIFI
    speculative_decode: bool = False
    execution_policy: str = "P0"  # AutoPilot policy (P0-P6)
    flash_moe_enabled: bool = False  # True when running P6 (slot-bank)

    def validate(self) -> None:
        """Validate pipeline configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        all_targets = [t for node in self.nodes for t in node.targets if t.enabled]
        roles = {t.role for t in all_targets}

        if NodeRole.PREFILL not in roles:
            msg = "Pipeline requires at least one PREFILL target"
            raise ValueError(msg)
        if NodeRole.DECODE not in roles:
            msg = "Pipeline requires at least one DECODE target"
            raise ValueError(msg)
        if self.speculative_decode and NodeRole.DRAFT not in roles:
            msg = "Speculative decode requires at least one DRAFT target (ANE)"
            raise ValueError(msg)
        if self.flash_moe_enabled and NodeRole.EXPERT_STREAMING not in roles:
            msg = "Flash-MoE (P6) requires at least one EXPERT_STREAMING target (T5 NVMe SSD)"
            raise ValueError(msg)
