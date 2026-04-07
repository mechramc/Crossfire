"""Pipeline-parallel orchestration across heterogeneous nodes.

Manages the split between prefill (GPU-bound) and decode (memory-bound)
phases across NVIDIA GPU and Apple Silicon hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NodeRole(Enum):
    """Role assignment for each node in the pipeline."""

    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class NodeConfig:
    """Configuration for a single compute node."""

    name: str
    host: str
    port: int
    role: NodeRole
    gpu_layers: int = 0


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the distributed pipeline."""

    nodes: list[NodeConfig]
    framework: str = "parallax"

    def validate(self) -> None:
        """Validate pipeline configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        roles = {node.role for node in self.nodes}
        if NodeRole.PREFILL not in roles:
            msg = "Pipeline requires at least one PREFILL node"
            raise ValueError(msg)
        if NodeRole.DECODE not in roles:
            msg = "Pipeline requires at least one DECODE node"
            raise ValueError(msg)
