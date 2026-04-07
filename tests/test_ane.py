"""Tests for ANE compute target configuration."""

import pytest

from crossfire.ane.draft_model import ANEBackend, DraftModelConfig, DraftResult
from crossfire.ane.power import (
    ANE_ACTIVE_WATTS_MAX,
    ANE_ACTIVE_WATTS_MIN,
    ANE_SRAM_CLIFF_MB,
    ANE_TFLOPS_FP16,
    PowerSnapshot,
)


def test_draft_model_config_validates_context():
    config = DraftModelConfig(
        model_path=__import__("pathlib").Path("/fake/model"),
        context_size=8192,
    )
    with pytest.raises(ValueError, match="exceeds ANEMLL limit"):
        config.validate()


def test_draft_model_config_validates_path():
    config = DraftModelConfig(
        model_path=__import__("pathlib").Path("/nonexistent/model.mlmodelc"),
    )
    with pytest.raises(FileNotFoundError, match="Draft model not found"):
        config.validate()


def test_ane_backend_values():
    assert ANEBackend.ANEMLL.value == "anemll"
    assert ANEBackend.RUSTANE.value == "rustane"
    assert ANEBackend.COREML.value == "coreml"


def test_draft_result_fields():
    result = DraftResult(
        tokens=[1, 2, 3],
        logits_shape=(3, 32000),
        elapsed_ms=15.2,
        power_watts=3.1,
    )
    assert len(result.tokens) == 3
    assert result.power_watts == 3.1


def test_power_snapshot_ane_fraction():
    snap = PowerSnapshot(
        ane_watts=3.0,
        gpu_watts=50.0,
        cpu_watts=12.0,
        total_system_watts=65.0,
    )
    assert snap.ane_fraction is not None
    assert abs(snap.ane_fraction - 3.0 / 65.0) < 1e-6


def test_power_snapshot_no_total():
    snap = PowerSnapshot(ane_watts=3.0)
    assert snap.ane_fraction is None


def test_ane_constants():
    assert ANE_TFLOPS_FP16 == 19.0
    assert ANE_SRAM_CLIFF_MB == 32
    assert ANE_ACTIVE_WATTS_MIN < ANE_ACTIVE_WATTS_MAX
