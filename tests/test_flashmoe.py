"""Tests for the Flash-MoE slot-bank runtime configuration and interface."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossfire.flashmoe.config import (
    FlashMoEBuildConfig,
    FlashMoEMode,
    SidecarConfig,
    SlotBankConfig,
)
from crossfire.flashmoe.runtime import FlashMoERuntime, FlashMoEStats

# --- FlashMoEMode -------------------------------------------------------------


def test_flash_moe_mode_values():
    """Enum string values match the anemll-flash-llama.cpp CLI flags."""
    assert FlashMoEMode.STOCK.value == "stock"
    assert FlashMoEMode.RESIDENT_BANK.value == "resident-bank"
    assert FlashMoEMode.SLOT_BANK.value == "slot-bank"
    assert FlashMoEMode.ORACLE_ALL_HIT.value == "oracle-all-hit"


# --- SidecarConfig ------------------------------------------------------------


def test_sidecar_config_coerces_str_to_path():
    """String sidecar paths are normalized to pathlib.Path."""
    cfg = SidecarConfig(sidecar_path="/tmp/sidecar")  # type: ignore[arg-type]
    assert isinstance(cfg.sidecar_path, Path)
    assert cfg.manifest_filename == "manifest.json"


def test_sidecar_manifest_path():
    """manifest_path joins sidecar_path and manifest_filename."""
    cfg = SidecarConfig(sidecar_path=Path("/tmp/sidecar"), manifest_filename="meta.json")
    assert cfg.manifest_path == Path("/tmp/sidecar/meta.json")


# --- SlotBankConfig -----------------------------------------------------------


def test_slot_bank_defaults():
    """Default slot-bank config uses documented k=3, 64 slots, prefetch on."""
    cfg = SlotBankConfig()
    assert cfg.slots_per_layer == 64
    assert cfg.topk == 3
    assert cfg.prefetch_temporal is True
    assert cfg.trace_path is None


def test_slot_bank_rejects_non_positive_slots():
    with pytest.raises(ValueError, match="slots_per_layer must be positive"):
        SlotBankConfig(slots_per_layer=0)


def test_slot_bank_rejects_non_positive_topk():
    with pytest.raises(ValueError, match="topk must be positive"):
        SlotBankConfig(topk=0)


def test_slot_bank_coerces_trace_path():
    """String trace_path is normalized to pathlib.Path."""
    cfg = SlotBankConfig(trace_path="/tmp/trace.jsonl")  # type: ignore[arg-type]
    assert isinstance(cfg.trace_path, Path)


# --- FlashMoEBuildConfig ------------------------------------------------------


def test_build_config_metal_flags():
    """Metal build includes GGML_METAL and LLAMA_FLASH_MOE_GPU_BANK."""
    cfg = FlashMoEBuildConfig()
    assert "-DGGML_METAL=ON" in cfg.METAL_FLAGS
    assert "-DLLAMA_FLASH_MOE_GPU_BANK=ON" in cfg.METAL_FLAGS


def test_build_config_cuda_flags():
    """CUDA build includes GGML_CUDA and LLAMA_FLASH_MOE_GPU_BANK."""
    cfg = FlashMoEBuildConfig()
    assert "-DGGML_CUDA=ON" in cfg.CUDA_FLAGS
    assert "-DLLAMA_FLASH_MOE_GPU_BANK=ON" in cfg.CUDA_FLAGS


def test_build_config_required_inference_flags():
    """Required inference flags match anemll-flash-llama.cpp runtime contract."""
    cfg = FlashMoEBuildConfig()
    assert cfg.REQUIRED_INFERENCE_FLAGS == ("-ub", "1", "-ngl", "99", "-fa", "on")


# --- FlashMoEStats ------------------------------------------------------------


def test_flash_moe_stats_optional_decode():
    """decode_tok_s is optional and defaults to None."""
    stats = FlashMoEStats(hit_rate=0.95, miss_count=12, expert_loads=256)
    assert stats.hit_rate == 0.95
    assert stats.miss_count == 12
    assert stats.expert_loads == 256
    assert stats.decode_tok_s is None


# --- FlashMoERuntime construction --------------------------------------------


def test_runtime_default_mode_stock():
    """Default mode is STOCK with no sidecar required."""
    rt = FlashMoERuntime(binary_path=Path("/opt/llama-cli"))
    assert rt.mode == FlashMoEMode.STOCK
    assert rt.sidecar is None
    assert isinstance(rt.slot_bank, SlotBankConfig)


def test_runtime_slot_bank_requires_sidecar():
    """SLOT_BANK mode raises without a SidecarConfig."""
    with pytest.raises(ValueError, match="slot-bank mode requires a SidecarConfig"):
        FlashMoERuntime(
            binary_path=Path("/opt/llama-cli"),
            mode=FlashMoEMode.SLOT_BANK,
        )


def test_runtime_coerces_binary_path():
    """String binary_path is normalized to pathlib.Path."""
    rt = FlashMoERuntime(binary_path="/opt/llama-cli")  # type: ignore[arg-type]
    assert isinstance(rt.binary_path, Path)


# --- FlashMoERuntime.build_cli_args ------------------------------------------


def _stock_runtime() -> FlashMoERuntime:
    return FlashMoERuntime(binary_path=Path("/opt/llama-cli"), mode=FlashMoEMode.STOCK)


def test_cli_args_include_required_flags():
    """build_cli_args always includes -ub 1 -ngl 99 -fa on."""
    rt = _stock_runtime()
    args = rt.build_cli_args(Path("/models/gemma.gguf"))
    assert "-ub" in args
    assert "-ngl" in args
    assert "-fa" in args
    assert "on" in args


def test_cli_args_include_mode_and_topk():
    """--moe-mode and --moe-topk reflect the runtime configuration."""
    rt = FlashMoERuntime(
        binary_path=Path("/opt/llama-cli"),
        mode=FlashMoEMode.RESIDENT_BANK,
        slot_bank=SlotBankConfig(topk=5),
    )
    args = rt.build_cli_args(Path("/models/gemma.gguf"))
    assert "--moe-mode=resident-bank" in args
    assert "--moe-topk=5" in args


def test_cli_args_context_size_passthrough():
    """--ctx-size reflects the caller's requested context window."""
    rt = _stock_runtime()
    args = rt.build_cli_args(Path("/models/gemma.gguf"), context_size=16384)
    assert "--ctx-size" in args
    assert "16384" in args


def test_cli_args_slot_bank_adds_streaming_flags():
    """SLOT_BANK mode adds --moe-slot-bank and --moe-prefetch-temporal."""
    sidecar = SidecarConfig(sidecar_path=Path("/tmp/sidecar"))
    slot_bank = SlotBankConfig(slots_per_layer=128, prefetch_temporal=True)
    rt = FlashMoERuntime(
        binary_path=Path("/opt/llama-cli"),
        mode=FlashMoEMode.SLOT_BANK,
        sidecar=sidecar,
        slot_bank=slot_bank,
    )
    args = rt.build_cli_args(Path("/models/gemma.gguf"))
    assert "--moe-slot-bank" in args
    assert "128" in args
    assert "--moe-prefetch-temporal" in args
    assert "--moe-sidecar" in args


def test_cli_args_slot_bank_trace_path_optional():
    """--moe-trace is only emitted when trace_path is configured."""
    sidecar = SidecarConfig(sidecar_path=Path("/tmp/sidecar"))

    no_trace = FlashMoERuntime(
        binary_path=Path("/opt/llama-cli"),
        mode=FlashMoEMode.SLOT_BANK,
        sidecar=sidecar,
        slot_bank=SlotBankConfig(trace_path=None),
    )
    assert "--moe-trace" not in no_trace.build_cli_args(Path("/models/gemma.gguf"))

    with_trace = FlashMoERuntime(
        binary_path=Path("/opt/llama-cli"),
        mode=FlashMoEMode.SLOT_BANK,
        sidecar=sidecar,
        slot_bank=SlotBankConfig(trace_path=Path("/tmp/trace.jsonl")),
    )
    args = with_trace.build_cli_args(Path("/models/gemma.gguf"))
    assert "--moe-trace" in args
    assert "/tmp/trace.jsonl" in args or str(Path("/tmp/trace.jsonl")) in args


def test_cli_args_stock_mode_omits_slot_bank_flags():
    """STOCK mode does not emit --moe-slot-bank or prefetch flags."""
    rt = _stock_runtime()
    args = rt.build_cli_args(Path("/models/gemma.gguf"))
    assert "--moe-slot-bank" not in args
    assert "--moe-prefetch-temporal" not in args


# --- FlashMoERuntime hardware-gated paths ------------------------------------


def test_run_inference_raises_not_implemented():
    """Inference is gated until anemll-flash-llama.cpp is built."""
    rt = _stock_runtime()
    with pytest.raises(NotImplementedError, match=r"anemll-flash-llama\.cpp"):
        rt.run_inference(Path("/models/gemma.gguf"), prompt="hello")


def test_extract_sidecar_raises_not_implemented():
    """Sidecar extraction is gated until tools are built."""
    rt = _stock_runtime()
    with pytest.raises(NotImplementedError, match="Sidecar extraction"):
        rt.extract_sidecar(Path("/models/gemma.gguf"), Path("/tmp/out"))
