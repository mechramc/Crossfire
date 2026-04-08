"""EXO-orchestrated pipeline across heterogeneous compute targets.

Manages the five compute targets in a CROSSFIRE-X pipeline:
T1 (CUDA GPU), T2 (Metal GPU), T3 (ANE), T4 (CPU/SME), T5 (RDMA).
EXO handles topology-aware auto-parallel and disaggregated prefill/decode.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ComputeTarget(Enum):
    """Compute targets in the CROSSFIRE-X pipeline."""

    T1_CUDA_GPU = "cuda_gpu"
    T2_METAL_GPU = "metal_gpu"
    T3_ANE = "ane"
    T4_CPU_SME = "cpu_sme"
    T5_RDMA = "rdma"


class NodeRole(Enum):
    """Role assignment for each node in the EXO pipeline."""

    PREFILL = "prefill"
    DECODE = "decode"
    DRAFT = "draft"


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
    rdma_enabled: bool = True
    speculative_decode: bool = False

    def validate(self) -> None:
        """Validate pipeline configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        all_targets = [t for node in self.nodes for t in node.targets if t.enabled]
        # RDMA is an interconnect, not a compute target — exclude from role checks
        compute_targets = [t for t in all_targets if t.target != ComputeTarget.T5_RDMA]
        roles = {t.role for t in compute_targets}

        if NodeRole.PREFILL not in roles:
            msg = "Pipeline requires at least one PREFILL target"
            raise ValueError(msg)
        if NodeRole.DECODE not in roles:
            msg = "Pipeline requires at least one DECODE target"
            raise ValueError(msg)
        if self.speculative_decode and NodeRole.DRAFT not in roles:
            msg = "Speculative decode requires at least one DRAFT target (ANE)"
            raise ValueError(msg)

        if self.rdma_enabled:
            rdma_targets = [t for t in all_targets if t.target == ComputeTarget.T5_RDMA]
            if not rdma_targets:
                msg = "RDMA enabled but no T5_RDMA target configured"
                raise ValueError(msg)
