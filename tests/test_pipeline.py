"""Tests for EXO-orchestrated pipeline configuration."""

import pytest

from crossfire.distributed.network import InterconnectType
from crossfire.distributed.pipeline import (
    ComputeTarget,
    ComputeTargetConfig,
    NodeConfig,
    NodeRole,
    PipelineConfig,
)


def _make_pipeline(
    *,
    prefill: bool = True,
    decode: bool = True,
    draft: bool = False,
    speculative: bool = False,
    interconnect: InterconnectType = InterconnectType.WIFI,
) -> PipelineConfig:
    """Helper to build pipeline configs for testing."""
    targets_pc = []
    targets_mac = []

    if prefill:
        targets_pc.append(
            ComputeTargetConfig(target=ComputeTarget.T1_CUDA_GPU, role=NodeRole.PREFILL)
        )
    if decode:
        targets_mac.append(
            ComputeTargetConfig(target=ComputeTarget.T2_METAL_GPU, role=NodeRole.DECODE)
        )
    if draft:
        targets_mac.append(ComputeTargetConfig(target=ComputeTarget.T3_ANE, role=NodeRole.DRAFT))

    nodes = [
        NodeConfig(name="PC", host="192.168.1.100", port=8080, targets=targets_pc),
        NodeConfig(name="Mac", host="192.168.1.101", port=8080, targets=targets_mac),
    ]
    return PipelineConfig(nodes=nodes, interconnect=interconnect, speculative_decode=speculative)


def test_valid_pipeline():
    config = _make_pipeline()
    config.validate()  # Should not raise


def test_missing_prefill_raises():
    config = _make_pipeline(prefill=False)
    with pytest.raises(ValueError, match="PREFILL"):
        config.validate()


def test_missing_decode_raises():
    config = _make_pipeline(decode=False)
    with pytest.raises(ValueError, match="DECODE"):
        config.validate()


def test_speculative_without_draft_raises():
    config = _make_pipeline(speculative=True, draft=False)
    with pytest.raises(ValueError, match="DRAFT"):
        config.validate()


def test_speculative_with_draft_passes():
    config = _make_pipeline(speculative=True, draft=True)
    config.validate()  # Should not raise


def test_compute_target_values():
    assert ComputeTarget.T1_CUDA_GPU.value == "cuda_gpu"
    assert ComputeTarget.T2_METAL_GPU.value == "metal_gpu"
    assert ComputeTarget.T3_ANE.value == "ane"
    assert ComputeTarget.T4_CPU_SME.value == "cpu_sme"
    assert ComputeTarget.T5_NVME_SSD.value == "nvme_ssd"


def test_default_interconnect_is_wifi():
    """Default interconnect reflects current dev setup (TCP/IP over WiFi), not USB4."""
    config = _make_pipeline()
    assert config.interconnect == InterconnectType.WIFI


def test_pipeline_with_usb4():
    config = _make_pipeline(interconnect=InterconnectType.USB4)
    config.validate()
    assert config.interconnect == InterconnectType.USB4


def test_pipeline_execution_policy_default():
    config = _make_pipeline()
    assert config.execution_policy == "P0"
    assert config.flash_moe_enabled is False


def test_pipeline_flash_moe_policy():
    targets_pc = [
        ComputeTargetConfig(target=ComputeTarget.T1_CUDA_GPU, role=NodeRole.PREFILL),
    ]
    targets_mac = [
        ComputeTargetConfig(target=ComputeTarget.T2_METAL_GPU, role=NodeRole.DECODE),
        ComputeTargetConfig(target=ComputeTarget.T5_NVME_SSD, role=NodeRole.EXPERT_STREAMING),
    ]
    config = PipelineConfig(
        nodes=[
            NodeConfig(name="PC", host="192.168.1.100", port=8080, targets=targets_pc),
            NodeConfig(name="Mac", host="192.168.1.101", port=8080, targets=targets_mac),
        ],
        interconnect=InterconnectType.USB4,
        execution_policy="P6",
        flash_moe_enabled=True,
    )
    config.validate()
    assert config.execution_policy == "P6"
    assert config.flash_moe_enabled is True


def test_flash_moe_without_expert_streaming_raises():
    """P6 (flash_moe_enabled=True) requires a T5 NVMe expert-streaming target."""
    config = _make_pipeline()  # has only PREFILL + DECODE, no expert-streaming
    # Rebuild with flash_moe_enabled=True and no T5
    config = PipelineConfig(
        nodes=config.nodes,
        interconnect=config.interconnect,
        execution_policy="P6",
        flash_moe_enabled=True,
    )
    with pytest.raises(ValueError, match="EXPERT_STREAMING"):
        config.validate()
