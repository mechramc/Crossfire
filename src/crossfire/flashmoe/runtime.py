"""Flash-MoE runtime interface for CROSSFIRE-X.

Provides the interface to anemll-flash-llama.cpp for MoE expert streaming.
Execution paths are stubs pending the hardware bring-up phase where
anemll-flash-llama.cpp is built and validated on both nodes.

Build targets:
  Mac Studio (Metal):
    cmake -S . -B build -DGGML_METAL=ON -DLLAMA_FLASH_MOE_GPU_BANK=ON
    cmake --build build --config Release -j$(sysctl -n hw.ncpu)

  RTX 5090 (CUDA):
    cmake -S . -B build -DGGML_CUDA=ON -DLLAMA_FLASH_MOE_GPU_BANK=ON
    cmake --build build --config Release -j$(nproc)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from crossfire.flashmoe.config import FlashMoEMode, SidecarConfig, SlotBankConfig


@dataclass(frozen=True)
class FlashMoEStats:
    """Per-request statistics from a Flash-MoE inference pass."""

    hit_rate: float  # fraction of slot-bank hits (0.0-1.0)
    miss_count: int  # number of NVMe pread() calls
    expert_loads: int  # total expert weight loads this request
    decode_tok_s: float | None = None  # decode throughput in tok/s


class FlashMoERuntime:
    """Interface to the anemll-flash-llama.cpp slot-bank runtime.

    Wraps the subprocess-level llama-cli invocation with Flash-MoE flags.
    Not runnable until anemll-flash-llama.cpp is built on both nodes.
    """

    def __init__(
        self,
        *,
        binary_path: Path,
        mode: FlashMoEMode = FlashMoEMode.STOCK,
        sidecar: SidecarConfig | None = None,
        slot_bank: SlotBankConfig | None = None,
    ) -> None:
        self.binary_path = Path(binary_path)
        self.mode = mode
        self.sidecar = sidecar
        self.slot_bank = slot_bank or SlotBankConfig()

        if mode == FlashMoEMode.SLOT_BANK and sidecar is None:
            msg = "slot-bank mode requires a SidecarConfig"
            raise ValueError(msg)

    def build_cli_args(self, model_path: Path, context_size: int = 8192) -> list[str]:
        """Construct the llama-cli argument list for this configuration.

        Args:
            model_path: Path to the base GGUF file (dense weights).
            context_size: Context window size in tokens.

        Returns:
            List of CLI argument strings ready for subprocess.run().
        """
        args: list[str] = [
            str(self.binary_path),
            "-m",
            str(model_path),
            "--ctx-size",
            str(context_size),
            # Required Flash-MoE inference flags
            "-ub",
            "1",
            "-ngl",
            "99",
            "-fa",
            "on",
            f"--moe-mode={self.mode.value}",
            f"--moe-topk={self.slot_bank.topk}",
        ]

        if self.sidecar is not None:
            args += ["--moe-sidecar", str(self.sidecar.sidecar_path)]

        if self.mode == FlashMoEMode.SLOT_BANK:
            args += ["--moe-slot-bank", str(self.slot_bank.slots_per_layer)]
            if self.slot_bank.prefetch_temporal:
                args.append("--moe-prefetch-temporal")
            if self.slot_bank.trace_path is not None:
                args += ["--moe-trace", str(self.slot_bank.trace_path)]

        return args

    def run_inference(
        self,
        model_path: Path,
        prompt: str,
        *,
        context_size: int = 8192,
        max_tokens: int = 256,
    ) -> FlashMoEStats:
        """Run a single inference pass through the Flash-MoE runtime.

        Not implemented -- requires anemll-flash-llama.cpp built on the
        target node. Will be implemented during Phase 2 hardware bring-up.

        Args:
            model_path: Path to the base GGUF file.
            prompt: Input prompt string.
            context_size: Context window in tokens.
            max_tokens: Maximum tokens to generate.

        Raises:
            NotImplementedError: Always. Pending hardware bring-up.
        """
        raise NotImplementedError(
            "Flash-MoE inference requires anemll-flash-llama.cpp built on the target node. "
            "See Phase 2 of the experiment plan for build instructions."
        )

    def extract_sidecar(self, gguf_path: Path, output_dir: Path) -> SidecarConfig:
        """Extract per-specialist weights from a fused MoE GGUF into sidecar format.

        For standard MoE GGUFs (Qwen3.5-35B, Kimi-K2.5): uses the
        flashmoe_sidecar.py extract tool.

        For Orion Forge fused models: converts KALAVAI adapter weights
        into the Flash-MoE binary-per-specialist + manifest.json format.

        Raises:
            NotImplementedError: Always. Pending hardware bring-up.
        """
        raise NotImplementedError(
            "Sidecar extraction requires anemll-flash-llama.cpp tools built on the target node. "
            "See Phase 2 of the experiment plan."
        )
