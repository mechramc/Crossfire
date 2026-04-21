"""Tests for TriAttention KV cache compression scaffold."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossfire.compression.triattention import (
    KVCompressionStrategy,
    TriAttentionConfig,
    apply,
    calibrate,
)

# --- KVCompressionStrategy ---------------------------------------------------


def test_strategy_values():
    """Enum string values are stable -- configs reference them by string."""
    assert KVCompressionStrategy.TRIATTENTION.value == "triattention"
    assert KVCompressionStrategy.TURBO3.value == "turbo3"
    assert KVCompressionStrategy.NONE.value == "none"


def test_strategy_triattention_is_primary():
    """TriAttention is the primary KV strategy (non-regression guard)."""
    # This asserts that the TRIATTENTION member exists and is selectable;
    # prevents accidental rename or removal that would break configs.
    strategy = KVCompressionStrategy("triattention")
    assert strategy is KVCompressionStrategy.TRIATTENTION


# --- TriAttentionConfig ------------------------------------------------------


def test_config_defaults():
    """Defaults: 4096 budget, no calibration file, no inter-iter pruning."""
    cfg = TriAttentionConfig()
    assert cfg.kv_budget == 4096
    assert cfg.calibration_path is None
    assert cfg.prune_between_iters is False


def test_config_rejects_non_positive_budget():
    with pytest.raises(ValueError, match="kv_budget must be positive"):
        TriAttentionConfig(kv_budget=0)


def test_config_rejects_negative_budget():
    with pytest.raises(ValueError, match="kv_budget must be positive"):
        TriAttentionConfig(kv_budget=-128)


def test_config_coerces_calibration_str_to_path():
    """String calibration_path is normalized to pathlib.Path."""
    cfg = TriAttentionConfig(calibration_path="/tmp/calib.safetensors")  # type: ignore[arg-type]
    assert isinstance(cfg.calibration_path, Path)


def test_is_calibrated_false_when_unset():
    """is_calibrated is False when no calibration file is configured."""
    cfg = TriAttentionConfig()
    assert cfg.is_calibrated is False


def test_is_calibrated_false_when_missing(tmp_path: Path):
    """is_calibrated is False when the calibration path doesn't exist."""
    cfg = TriAttentionConfig(calibration_path=tmp_path / "missing.bin")
    assert cfg.is_calibrated is False


def test_is_calibrated_true_when_present(tmp_path: Path):
    """is_calibrated is True when the calibration path exists on disk."""
    calib = tmp_path / "calib.bin"
    calib.write_bytes(b"placeholder")
    cfg = TriAttentionConfig(calibration_path=calib)
    assert cfg.is_calibrated is True


def test_config_preserves_prune_flag():
    """prune_between_iters is stored verbatim for Verifier Loop mode."""
    cfg = TriAttentionConfig(prune_between_iters=True)
    assert cfg.prune_between_iters is True


# --- calibrate ---------------------------------------------------------------


def test_calibrate_raises_not_implemented(tmp_path: Path):
    """calibrate() is gated until the arXiv:2604.04921 reference lands."""
    with pytest.raises(NotImplementedError, match=r"arXiv:2604\.04921"):
        calibrate(
            model_path=tmp_path / "model.safetensors",
            output_path=tmp_path / "centers.bin",
        )


# --- apply -------------------------------------------------------------------


def test_apply_raises_not_implemented():
    """apply() is gated until the reference implementation ships."""
    cfg = TriAttentionConfig()
    with pytest.raises(NotImplementedError, match="TriAttention runtime scoring"):
        apply(kv_cache=object(), config=cfg)
