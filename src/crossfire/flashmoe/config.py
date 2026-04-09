"""Configuration types for the Flash-MoE slot-bank runtime.

Models the anemll-flash-llama.cpp build flags and runtime modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class FlashMoEMode(Enum):
    """Execution mode for the Flash-MoE runtime.

    Attributes:
        STOCK: All experts resident in GPU/unified memory. No streaming.
            Use when the full expert pool fits in node memory.
        RESIDENT_BANK: All experts resident but indexed via bank structure.
            Use when all experts fit but bank-level placement is desired.
        SLOT_BANK: Hot experts cached in slots, cold experts streamed from
            NVMe via pread(). Use when expert pool exceeds node memory.
        ORACLE_ALL_HIT: Benchmarking ceiling -- all experts pre-loaded to
            simulate a 100% cache-hit rate. Not for production use.
    """

    STOCK = "stock"
    RESIDENT_BANK = "resident-bank"
    SLOT_BANK = "slot-bank"
    ORACLE_ALL_HIT = "oracle-all-hit"


@dataclass(frozen=True)
class SidecarConfig:
    """Sidecar directory for per-specialist binary weights.

    The sidecar is created by the Flash-MoE extraction tool from a fused
    GGUF. For Orion Forge models, the extraction converts KALAVAI adapter
    weights into the Flash-MoE binary-per-specialist format.
    """

    sidecar_path: Path
    manifest_filename: str = "manifest.json"

    def __post_init__(self) -> None:
        if not isinstance(self.sidecar_path, Path):
            object.__setattr__(self, "sidecar_path", Path(self.sidecar_path))

    @property
    def manifest_path(self) -> Path:
        return self.sidecar_path / self.manifest_filename


@dataclass(frozen=True)
class SlotBankConfig:
    """Slot-bank sizing and prefetch configuration.

    Attributes:
        slots_per_layer: Number of resident expert slots per transformer
            layer. Size to 5-15% of node RAM to balance hit rate and
            memory cost. Default 64 covers typical 20-specialist pools.
        topk: Number of experts activated per token. Must match the model
            router's top-k value (KALAVAI default k=3).
        prefetch_temporal: Enable one-step temporal prefetch. Achieves
            ~100% hit rate after warmup for sequential generation.
        trace_path: Optional path for per-request routing diagnostics
            (JSONL). Used for hit-rate analysis and oracle calibration.
    """

    slots_per_layer: int = 64
    topk: int = 3
    prefetch_temporal: bool = True
    trace_path: Path | None = None

    def __post_init__(self) -> None:
        if self.slots_per_layer <= 0:
            msg = "slots_per_layer must be positive"
            raise ValueError(msg)
        if self.topk <= 0:
            msg = "topk must be positive"
            raise ValueError(msg)
        if self.trace_path is not None and not isinstance(self.trace_path, Path):
            object.__setattr__(self, "trace_path", Path(self.trace_path))


@dataclass(frozen=True)
class FlashMoEBuildConfig:
    """Build flags for anemll-flash-llama.cpp.

    Encodes the cmake flags required for each target platform.
    """

    # Metal (Mac Studio) flags
    METAL_FLAGS: tuple[str, ...] = field(
        default=("-DGGML_METAL=ON", "-DLLAMA_FLASH_MOE_GPU_BANK=ON"),
        init=False,
    )

    # CUDA (RTX 5090) flags
    CUDA_FLAGS: tuple[str, ...] = field(
        default=("-DGGML_CUDA=ON", "-DLLAMA_FLASH_MOE_GPU_BANK=ON"),
        init=False,
    )

    # Required inference flags (apply to both platforms)
    REQUIRED_INFERENCE_FLAGS: tuple[str, ...] = field(
        default=("-ub", "1", "-ngl", "99", "-fa", "on"),
        init=False,
    )
