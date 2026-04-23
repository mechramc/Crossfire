"""Flash-MoE runtime interface for CROSSFIRE-X.

Provides the interface to anemll-flash-llama.cpp for MoE expert streaming.
The repo vendors both the `llama-cli` binary target and the Python sidecar
tooling, so this module wraps those real execution paths rather than leaving
them as manual shell-only steps.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from crossfire.flashmoe.config import FlashMoEMode, SidecarConfig, SlotBankConfig

_HIT_RATE_RE = re.compile(r"slot-bank cached expert hit rate:\s+(?P<hit>[0-9.]+)%")
_ROUTED_SUMMARY_RE = re.compile(
    r"Flash-MoE routed src=.*?\brefs=(?P<refs>\d+).*?\bpread=(?P<pread>\d+)",
)
_PREFETCH_SUMMARY_RE = re.compile(r"Flash-MoE .*?\bmiss=(?P<miss>\d+).*?\bpread=(?P<pread>\d+)")
_TOK_S_RE = re.compile(
    r"eval time = .*?\([ ]*[0-9.]+ ms per token, [ ]*(?P<tok_s>[0-9.]+) tokens per second\)",
)


@dataclass(frozen=True)
class FlashMoEStats:
    """Per-request statistics from a Flash-MoE inference pass."""

    hit_rate: float  # fraction of slot-bank hits (0.0-1.0)
    miss_count: int  # number of NVMe pread() calls
    expert_loads: int  # total expert weight loads this request
    decode_tok_s: float | None = None  # decode throughput in tok/s


class FlashMoERuntime:
    """Interface to the anemll-flash-llama.cpp slot-bank runtime."""

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

    @staticmethod
    def _sidecar_tool_path() -> Path:
        return (
            Path(__file__).resolve().parents[3]
            / "vendor"
            / "anemll-flash-llama.cpp"
            / "tools"
            / "flashmoe-sidecar"
            / "flashmoe_sidecar.py"
        )

    @staticmethod
    def _run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
        )

    @classmethod
    def _run_sidecar_tool(cls, args: list[str]) -> subprocess.CompletedProcess[str]:
        tool_path = cls._sidecar_tool_path()
        if not tool_path.is_file():
            raise FileNotFoundError(f"Flash-MoE sidecar tool not found: {tool_path}")
        return cls._run_command([sys.executable, str(tool_path), *args])

    @staticmethod
    def _require_existing_path(path: Path, *, kind: str) -> Path:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"{kind} not found: {resolved}")
        return resolved

    @staticmethod
    def _parse_inference_output(output: str) -> FlashMoEStats:
        hit_rate = 0.0
        miss_count = 0
        expert_loads = 0
        decode_tok_s: float | None = None

        hit_match = _HIT_RATE_RE.search(output)
        if hit_match is not None:
            hit_rate = float(hit_match.group("hit")) / 100.0

        routed_match = _ROUTED_SUMMARY_RE.search(output)
        if routed_match is not None:
            expert_loads = int(routed_match.group("refs"))
            miss_count = int(routed_match.group("pread"))
        else:
            prefetch_match = _PREFETCH_SUMMARY_RE.search(output)
            if prefetch_match is not None:
                miss_count = int(prefetch_match.group("pread"))

        tok_s_match = _TOK_S_RE.search(output)
        if tok_s_match is not None:
            decode_tok_s = float(tok_s_match.group("tok_s"))

        return FlashMoEStats(
            hit_rate=hit_rate,
            miss_count=miss_count,
            expert_loads=expert_loads,
            decode_tok_s=decode_tok_s,
        )

    def build_cli_args(self, model_path: Path, context_size: int = 8192) -> list[str]:
        """Construct the llama-cli argument list for this configuration."""
        args: list[str] = [
            str(self.binary_path),
            "-m",
            str(model_path),
            "--ctx-size",
            str(context_size),
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
        """Run a single inference pass through the Flash-MoE runtime."""
        self._require_existing_path(self.binary_path, kind="Flash-MoE binary")
        model = self._require_existing_path(model_path, kind="model")

        args = self.build_cli_args(model, context_size=context_size)
        args += [
            "--simple-io",
            "--color",
            "off",
            "--perf",
            "-p",
            prompt,
            "-n",
            str(max_tokens),
        ]

        proc = self._run_command(args)
        if proc.returncode != 0:
            raise RuntimeError(
                "Flash-MoE inference failed\n"
                f"command: {' '.join(args)}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        return self._parse_inference_output(f"{proc.stdout}\n{proc.stderr}")

    def inspect_sidecar(
        self,
        gguf_path: Path,
        *,
        sidecar_path: Path | None = None,
        include_shared: bool = False,
        layers: str | None = None,
        families: str | None = None,
    ) -> dict[str, object]:
        """Inspect the GGUF MoE tensor layout and optional sidecar parity."""
        model = self._require_existing_path(gguf_path, kind="model")

        args = ["inspect", "--model", str(model), "--json"]
        if sidecar_path is not None:
            args += ["--sidecar", str(Path(sidecar_path).expanduser().resolve())]
        if include_shared:
            args.append("--include-shared")
        if layers is not None:
            args += ["--layers", layers]
        if families is not None:
            args += ["--families", families]

        proc = self._run_sidecar_tool(args)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Flash-MoE sidecar inspect failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        return json.loads(proc.stdout)

    def verify_sidecar(
        self,
        gguf_path: Path,
        sidecar_path: Path,
        *,
        metadata_only: bool = False,
        layers: str | None = None,
        families: str | None = None,
    ) -> SidecarConfig:
        """Verify a sidecar against the source GGUF."""
        model = self._require_existing_path(gguf_path, kind="model")
        sidecar = Path(sidecar_path).expanduser().resolve()
        if not sidecar.exists():
            raise FileNotFoundError(f"sidecar not found: {sidecar}")

        args = ["verify", "--model", str(model), "--sidecar", str(sidecar)]
        if metadata_only:
            args.append("--metadata-only")
        if layers is not None:
            args += ["--layers", layers]
        if families is not None:
            args += ["--families", families]

        proc = self._run_sidecar_tool(args)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Flash-MoE sidecar verify failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        return SidecarConfig(sidecar_path=sidecar if sidecar.is_dir() else sidecar.parent)

    def extract_sidecar(
        self,
        gguf_path: Path,
        output_dir: Path,
        *,
        include_shared: bool = False,
        force: bool = False,
        layers: str | None = None,
        families: str | None = None,
        verify: bool = True,
    ) -> SidecarConfig:
        """Extract per-specialist weights from a fused MoE GGUF into sidecar format."""
        model = self._require_existing_path(gguf_path, kind="model")
        out_dir = Path(output_dir).expanduser().resolve()

        args = ["extract", "--model", str(model), "--out-dir", str(out_dir)]
        if include_shared:
            args.append("--include-shared")
        if force:
            args.append("--force")
        if layers is not None:
            args += ["--layers", layers]
        if families is not None:
            args += ["--families", families]

        proc = self._run_sidecar_tool(args)
        if proc.returncode != 0:
            raise RuntimeError(
                "Flash-MoE sidecar extraction failed\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

        sidecar = SidecarConfig(sidecar_path=out_dir)
        if verify:
            self.verify_sidecar(
                model,
                sidecar.sidecar_path,
                metadata_only=False,
                layers=layers,
                families=families,
            )
        return sidecar
