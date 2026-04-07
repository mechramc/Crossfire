"""Tests for EXO-orchestrated pipeline configuration."""

import pytest

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
    rdma: bool = True,
    speculative: bool = False,
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
        targets_mac.append(
            ComputeTargetConfig(target=ComputeTarget.T3_ANE, role=NodeRole.DRAFT)
        )
    if rdma:
        targets_pc.append(
            ComputeTargetConfig(target=ComputeTarget.T5_RDMA, role=NodeRole.PREFILL)
        )

    nodes = [
        NodeConfig(name="PC", host="192.168.1.100", port=8080, targets=targets_pc),
        NodeConfig(name="Mac", host="192.168.1.101", port=8080, targets=targets_mac),
    ]
    return PipelineConfig(
        nodes=nodes, rdma_enabled=rdma, speculative_decode=speculative
    )


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
    assert ComputeTarget.T3_ANE.value == "ane"
    assert ComputeTarget.T5_RDMA.value == "rdma"
