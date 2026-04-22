"""Tests for the Gemma 4 E2B chunked CoreML inference engine (T-0609a)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from crossfire.ane.gemma4_assets import (
    COS_FULL_FILENAME,
    COS_SLIDING_FILENAME,
    EMBED_DATA_FILENAME,
    EMBED_SCALES_FILENAME,
    INT8_MAX,
    MODEL_CONFIG_FILENAME,
    PER_LAYER_NORM_FILENAME,
    PROJECTION_FILENAME,
    SIN_FULL_FILENAME,
    SIN_SLIDING_FILENAME,
    TOKENIZER_FILENAME,
    TOKENIZER_SUBDIR,
    Gemma4Config,
    QuantizedEmbedding,
    load_per_layer_embedding,
    load_per_layer_norm_weight,
    load_per_layer_projection,
    load_rope_tables,
    load_token_embedding,
    load_tokenizer,
)
from crossfire.ane.gemma4_chunked import (
    Gemma4ChunkedEngine,
    GenerationResult,
    _compute_unit_from_string,
    _extract_context_width,
    _parse_shape,
    _read_chunk_metadata,
)
from crossfire.ane.gemma4_masks import (
    FP16_ALLOW_FILL,
    FP16_BLOCK_FILL,
    causal_mask_full,
    causal_mask_sliding,
    update_mask,
)


def _write_model_config(bundle: Path, overrides: dict | None = None) -> dict:
    """Write a Gemma 4 E2B-shaped model_config.json; return the dict."""
    data = {
        "model_name": "gemma4-e2b-swa-2k",
        "architecture": "gemma4",
        "hidden_size": 1536,
        "num_hidden_layers": 35,
        "context_length": 2048,
        "sliding_window": 512,
        "vocab_size": 262144,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "per_layer_dim": 256,
        "embed_scale": 39.191835884530846,
        "per_layer_model_projection_scale": 0.02551551815399144,
        "per_layer_input_scale": 0.7071067811865476,
        "per_layer_embed_scale": 16.0,
    }
    if overrides:
        data.update(overrides)
    (bundle / MODEL_CONFIG_FILENAME).write_text(json.dumps(data), encoding="utf-8")
    return data


def test_gemma4_config_from_bundle_populates_all_fields(tmp_path: Path):
    _write_model_config(tmp_path)
    cfg = Gemma4Config.from_bundle(tmp_path)

    assert cfg.hidden_size == 1536
    assert cfg.num_layers == 35
    assert cfg.vocab_size == 262144
    assert cfg.context_length == 2048
    assert cfg.per_layer_dim == 256
    assert cfg.bos_token_id == 2
    assert cfg.eos_token_id == 1
    assert cfg.embed_scale == pytest.approx(39.191835884530846)
    assert cfg.per_layer_embed_scale == pytest.approx(16.0)


def test_gemma4_config_matches_real_bundle_if_present():
    """If the real bundle is downloaded, ensure from_bundle parses it."""
    real_bundle = Path("models/gemma-4-E2B-coreml")
    if not (real_bundle / MODEL_CONFIG_FILENAME).is_file():
        pytest.skip("real bundle not present; run download first")
    cfg = Gemma4Config.from_bundle(real_bundle)
    # Values we know from Session 17 scout + checkpoint.md
    assert cfg.num_layers == 35
    assert cfg.hidden_size == 1536
    assert cfg.vocab_size == 262144
    assert cfg.bos_token_id == 2
    assert cfg.eos_token_id == 1
    assert cfg.context_length == 2048
    assert cfg.per_layer_dim == 256


def test_gemma4_config_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match=r"model_config\.json"):
        Gemma4Config.from_bundle(tmp_path)


def test_gemma4_config_missing_key_raises(tmp_path: Path):
    # Use full config then remove a required key
    data = {
        "model_name": "x",
        "architecture": "gemma4",
        "hidden_size": 1536,
        # missing: num_hidden_layers
        "context_length": 2048,
        "sliding_window": 512,
        "vocab_size": 262144,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "per_layer_dim": 256,
        "embed_scale": 1.0,
        "per_layer_model_projection_scale": 1.0,
        "per_layer_input_scale": 1.0,
        "per_layer_embed_scale": 1.0,
    }
    (tmp_path / MODEL_CONFIG_FILENAME).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(KeyError, match="num_hidden_layers"):
        Gemma4Config.from_bundle(tmp_path)


def test_gemma4_config_is_frozen(tmp_path: Path):
    import dataclasses

    _write_model_config(tmp_path)
    cfg = Gemma4Config.from_bundle(tmp_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.hidden_size = 999  # type: ignore[misc]


def test_load_tokenizer_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="tokenizer"):
        load_tokenizer(tmp_path)


def test_load_tokenizer_from_real_bundle_roundtrips():
    """If bundle is present, ensure tokenizer loads and BOS/EOS ids match config."""
    real_bundle = Path("models/gemma-4-E2B-coreml")
    tok_path = real_bundle / TOKENIZER_SUBDIR / TOKENIZER_FILENAME
    if not tok_path.is_file():
        pytest.skip("real bundle tokenizer not present")
    tok = load_tokenizer(real_bundle)
    encoded = tok.encode("The capital of France is")
    # Non-empty id list, all ints in vocab range
    assert len(encoded.ids) > 0
    assert all(isinstance(i, int) and 0 <= i < 262144 for i in encoded.ids)
    # Roundtrip
    decoded = tok.decode(encoded.ids)
    assert "capital" in decoded.lower()


def _write_synthetic_qembed(
    bundle: Path, *, vocab: int, dim: int, filename: str, scales_filename: str
) -> tuple[np.ndarray, np.ndarray]:
    """Write a synthetic int8 embedding table + fp16 scales; return arrays."""
    rng = np.random.default_rng(42)
    data = rng.integers(-128, 128, size=(vocab, dim), dtype=np.int8)
    scales = rng.uniform(0.01, 1.0, size=(vocab,)).astype(np.float16)
    (bundle / filename).write_bytes(data.tobytes())
    (bundle / scales_filename).write_bytes(scales.tobytes())
    return data, scales


def test_quantized_embedding_dequant_matches_formula(tmp_path: Path):
    vocab, dim, gscale = 8, 4, 39.191835884530846
    data, scales = _write_synthetic_qembed(
        tmp_path,
        vocab=vocab,
        dim=dim,
        filename=EMBED_DATA_FILENAME,
        scales_filename=EMBED_SCALES_FILENAME,
    )
    emb = QuantizedEmbedding(
        data_path=tmp_path / EMBED_DATA_FILENAME,
        scales_path=tmp_path / EMBED_SCALES_FILENAME,
        vocab_size=vocab,
        dim=dim,
        global_scale=gscale,
    )

    for tid in range(vocab):
        out = emb.lookup(tid)
        assert out.dtype == np.float16
        assert out.shape == (dim,)
        # Reference: (int8 * scale/127 * gscale) in fp32, cast fp16
        row_scale = float(scales[tid]) / INT8_MAX * gscale
        expected = (data[tid].astype(np.float32) * row_scale).astype(np.float16)
        np.testing.assert_array_equal(out, expected)


def test_quantized_embedding_size_mismatch_raises(tmp_path: Path):
    # Wrong data size
    (tmp_path / EMBED_DATA_FILENAME).write_bytes(b"\x00" * 10)
    (tmp_path / EMBED_SCALES_FILENAME).write_bytes(b"\x00" * 4)
    with pytest.raises(ValueError, match="bytes"):
        QuantizedEmbedding(
            data_path=tmp_path / EMBED_DATA_FILENAME,
            scales_path=tmp_path / EMBED_SCALES_FILENAME,
            vocab_size=2,
            dim=8,
            global_scale=1.0,
        )


def test_quantized_embedding_token_out_of_range(tmp_path: Path):
    _write_synthetic_qembed(
        tmp_path,
        vocab=4,
        dim=2,
        filename=EMBED_DATA_FILENAME,
        scales_filename=EMBED_SCALES_FILENAME,
    )
    emb = QuantizedEmbedding(
        data_path=tmp_path / EMBED_DATA_FILENAME,
        scales_path=tmp_path / EMBED_SCALES_FILENAME,
        vocab_size=4,
        dim=2,
        global_scale=1.0,
    )
    with pytest.raises(ValueError, match="out of range"):
        emb.lookup(4)
    with pytest.raises(ValueError, match="out of range"):
        emb.lookup(-1)


def _write_synthetic_npy(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Write a synthetic fp16 .npy at path with the given shape."""
    arr = np.random.default_rng(0).standard_normal(shape).astype(np.float16)
    np.save(path, arr)
    return arr


def test_load_rope_tables(tmp_path: Path):
    cos_s = _write_synthetic_npy(tmp_path / COS_SLIDING_FILENAME, (64, 128))
    sin_s = _write_synthetic_npy(tmp_path / SIN_SLIDING_FILENAME, (64, 128))
    cos_f = _write_synthetic_npy(tmp_path / COS_FULL_FILENAME, (64, 256))
    sin_f = _write_synthetic_npy(tmp_path / SIN_FULL_FILENAME, (64, 256))

    tables = load_rope_tables(tmp_path)
    assert tables.max_positions == 64
    assert tables.sliding_head_dim == 128
    assert tables.full_head_dim == 256
    np.testing.assert_array_equal(tables.cos_sliding, cos_s)
    np.testing.assert_array_equal(tables.sin_sliding, sin_s)
    np.testing.assert_array_equal(tables.cos_full, cos_f)
    np.testing.assert_array_equal(tables.sin_full, sin_f)


def test_load_rope_tables_mismatched_positions(tmp_path: Path):
    _write_synthetic_npy(tmp_path / COS_SLIDING_FILENAME, (64, 128))
    _write_synthetic_npy(tmp_path / SIN_SLIDING_FILENAME, (64, 128))
    _write_synthetic_npy(tmp_path / COS_FULL_FILENAME, (128, 256))
    _write_synthetic_npy(tmp_path / SIN_FULL_FILENAME, (128, 256))
    with pytest.raises(ValueError, match="positions"):
        load_rope_tables(tmp_path)


def test_load_per_layer_projection(tmp_path: Path):
    shape = (32, 16)
    arr = np.random.default_rng(1).standard_normal(shape).astype(np.float16)
    (tmp_path / PROJECTION_FILENAME).write_bytes(arr.tobytes())
    loaded = load_per_layer_projection(tmp_path, expected_shape=shape)
    np.testing.assert_array_equal(loaded, arr)


def test_load_per_layer_projection_size_mismatch(tmp_path: Path):
    (tmp_path / PROJECTION_FILENAME).write_bytes(b"\x00" * 10)
    with pytest.raises(ValueError, match="bytes"):
        load_per_layer_projection(tmp_path, expected_shape=(8, 4))


def test_load_per_layer_norm_weight(tmp_path: Path):
    arr = np.random.default_rng(2).standard_normal(64).astype(np.float32)
    (tmp_path / PER_LAYER_NORM_FILENAME).write_bytes(arr.tobytes())
    loaded = load_per_layer_norm_weight(tmp_path, expected_dim=64)
    np.testing.assert_array_equal(loaded, arr)


def test_causal_mask_full_position_zero():
    m = causal_mask_full(0, context_length=8)
    assert m.shape == (1, 1, 1, 8)
    assert m.dtype == np.float16
    assert m[0, 0, 0, 0] == FP16_ALLOW_FILL
    assert (m[0, 0, 0, 1:] == FP16_BLOCK_FILL).all()


def test_causal_mask_full_mid_position():
    m = causal_mask_full(3, context_length=8)
    assert (m[0, 0, 0, :4] == FP16_ALLOW_FILL).all()
    assert (m[0, 0, 0, 4:] == FP16_BLOCK_FILL).all()


def test_causal_mask_full_last_position():
    m = causal_mask_full(7, context_length=8)
    assert (m[0, 0, 0, :] == FP16_ALLOW_FILL).all()


def test_causal_mask_full_raises_on_out_of_range():
    import pytest as _pt

    with _pt.raises(ValueError, match="context_length"):
        causal_mask_full(8, context_length=8)
    with _pt.raises(ValueError, match=">= 0"):
        causal_mask_full(-1, context_length=8)


def test_causal_mask_sliding_early_positions():
    # At position 0, only slot W-1 is valid
    m = causal_mask_sliding(0, sliding_window=4)
    assert m.shape == (1, 1, 1, 4)
    assert m.dtype == np.float16
    assert (m[0, 0, 0, :3] == FP16_BLOCK_FILL).all()
    assert m[0, 0, 0, 3] == FP16_ALLOW_FILL


def test_causal_mask_sliding_at_window_boundary():
    # position = W-1: all slots valid
    m = causal_mask_sliding(3, sliding_window=4)
    assert (m[0, 0, 0, :] == FP16_ALLOW_FILL).all()


def test_causal_mask_sliding_past_window():
    # position = W: cache still full (W valid, right-aligned)
    m = causal_mask_sliding(4, sliding_window=4)
    assert (m[0, 0, 0, :] == FP16_ALLOW_FILL).all()
    # position = 100: same
    m = causal_mask_sliding(100, sliding_window=4)
    assert (m[0, 0, 0, :] == FP16_ALLOW_FILL).all()


def test_update_mask_first_position():
    m = update_mask(0, context_length=8)
    assert m.shape == (1, 1, 8, 1)
    assert m.dtype == np.float16
    assert m[0, 0, 0, 0] == np.float16(1.0)
    assert (m[0, 0, 1:, 0] == 0.0).all()


def test_update_mask_mid_position():
    m = update_mask(5, context_length=8)
    assert m[0, 0, 5, 0] == np.float16(1.0)
    # All other slots are zero
    nonzero = np.nonzero(m)
    assert len(nonzero[0]) == 1


def test_update_mask_clamps_at_ctx():
    # position = ctx: should clamp to ctx-1
    m = update_mask(8, context_length=8)
    assert m[0, 0, 7, 0] == np.float16(1.0)


def test_parse_shape():
    assert _parse_shape("[1, 1, 1, 512]") == (1, 1, 1, 512)
    assert _parse_shape("[1]") == (1,)
    assert _parse_shape("[]") == ()


def test_compute_unit_from_string_valid():
    import coremltools as ct

    assert _compute_unit_from_string("cpu_and_ne") == ct.ComputeUnit.CPU_AND_NE
    assert _compute_unit_from_string("CPU_ONLY") == ct.ComputeUnit.CPU_ONLY
    assert _compute_unit_from_string("all") == ct.ComputeUnit.ALL


def test_compute_unit_from_string_invalid():
    with pytest.raises(ValueError, match="unknown compute_units"):
        _compute_unit_from_string("magic")


def test_extract_context_width():
    meta = {
        "inputSchema": [
            {"name": "causal_mask", "shape": "[1, 1, 1, 512]"},
            {"name": "other", "shape": "[1]"},
        ]
    }
    assert _extract_context_width(meta) == 512


def test_extract_context_width_missing_raises():
    with pytest.raises(ValueError, match="causal_mask"):
        _extract_context_width({"inputSchema": []})


def test_read_chunk_metadata_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="metadata"):
        _read_chunk_metadata(tmp_path / "chunk1.mlmodelc")


def test_read_chunk_metadata_from_real_bundle():
    bundle = Path("models/gemma-4-E2B-coreml")
    chunk = bundle / "chunk1.mlmodelc"
    if not chunk.is_dir():
        pytest.skip("real bundle not present")
    meta = _read_chunk_metadata(chunk)
    assert "inputSchema" in meta
    assert "outputSchema" in meta
    # Effective context derived from mask shape
    assert _extract_context_width(meta) == 512


# ---- End-to-end engine tests (require real bundle + ANE) ----


def _real_bundle_or_skip() -> Path:
    bundle = Path("models/gemma-4-E2B-coreml")
    if not (bundle / "model_config.json").is_file():
        pytest.skip("real bundle not present")
    return bundle


def test_engine_load_prewarm_and_properties():
    bundle = _real_bundle_or_skip()
    engine = Gemma4ChunkedEngine.load(bundle, compute_units="cpu_and_ne")
    assert engine.num_chunks == 3
    assert engine.effective_context == 512
    assert engine.current_position == 0  # reset after prewarm
    assert engine.config.bos_token_id == 2
    assert engine.config.eos_token_id == 1


def test_engine_predict_step_advances_state():
    bundle = _real_bundle_or_skip()
    engine = Gemma4ChunkedEngine.load(bundle)
    # First prediction at position 0 with BOS; should return a valid token id
    out = engine.predict_step(engine.config.bos_token_id, 0)
    assert 0 <= out < engine.config.vocab_size


def test_engine_generate_produces_coherent_paris_answer():
    bundle = _real_bundle_or_skip()
    engine = Gemma4ChunkedEngine.load(bundle)
    result = engine.generate(
        "The capital of France is",
        max_tokens=8,
        stop_on_eos=False,
    )
    assert isinstance(result, GenerationResult)
    assert result.prompt_tokens >= 6  # BOS + at least 5 prompt tokens
    assert result.generated_tokens == 8
    assert result.ttft_ms > 0.0
    assert result.decode_tok_s > 0.0
    # Acceptance bar: first token should be " Paris" (token 9079) — verified
    # from the model's argmax on this prompt. A change here means the model
    # is broken, not that the test is wrong.
    assert "paris" in result.text.lower(), (
        f"expected 'Paris' in generated text, got {result.text!r}"
    )


def test_engine_reset_zeroes_position_and_state():
    bundle = _real_bundle_or_skip()
    engine = Gemma4ChunkedEngine.load(bundle)
    engine.predict_step(engine.config.bos_token_id, 0)
    engine._position = 5  # simulate advancement
    engine.reset()
    assert engine.current_position == 0
    # Post-reset, should be able to predict at position 0 again
    out = engine.predict_step(engine.config.bos_token_id, 0)
    assert 0 <= out < engine.config.vocab_size


def test_engine_predict_step_rejects_out_of_range_position():
    bundle = _real_bundle_or_skip()
    engine = Gemma4ChunkedEngine.load(bundle)
    with pytest.raises(ValueError, match="out of range"):
        engine.predict_step(2, engine.effective_context)
    with pytest.raises(ValueError, match="out of range"):
        engine.predict_step(2, -1)


def test_real_bundle_asset_load_roundtrip():
    """Verify every asset loader opens the real bundle without error."""
    bundle = Path("models/gemma-4-E2B-coreml")
    if not (bundle / MODEL_CONFIG_FILENAME).is_file():
        pytest.skip("real bundle not present")
    cfg = Gemma4Config.from_bundle(bundle)
    tok_embed = load_token_embedding(bundle, cfg)
    ple_embed = load_per_layer_embedding(bundle, cfg)
    rope = load_rope_tables(bundle)
    proj = load_per_layer_projection(
        bundle, expected_shape=(cfg.num_layers * cfg.per_layer_dim, cfg.hidden_size)
    )
    norm = load_per_layer_norm_weight(bundle, expected_dim=cfg.per_layer_dim)

    # Shapes
    assert tok_embed.dim == cfg.hidden_size
    assert tok_embed.vocab_size == cfg.vocab_size
    assert ple_embed.dim == cfg.num_layers * cfg.per_layer_dim
    assert rope.sliding_head_dim == 256
    assert rope.full_head_dim == 512
    assert proj.shape == (cfg.num_layers * cfg.per_layer_dim, cfg.hidden_size)
    assert norm.shape == (cfg.per_layer_dim,)

    # BOS lookup returns finite values
    bos_vec = tok_embed.lookup(cfg.bos_token_id)
    assert bos_vec.shape == (cfg.hidden_size,)
    assert np.isfinite(bos_vec).all()
    ple_vec = ple_embed.lookup(cfg.bos_token_id)
    assert ple_vec.shape == (cfg.num_layers * cfg.per_layer_dim,)
    assert np.isfinite(ple_vec).all()
