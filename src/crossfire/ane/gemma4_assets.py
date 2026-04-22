"""Asset loaders for the Gemma 4 E2B CoreML bundle.

Handles parsing of `model_config.json`, loading of the HuggingFace tokenizer,
memory-mapped int8 embedding dequantization, per-layer embedding (PLE)
lookup, `per_layer_projection.bin`/`per_layer_norm_weight.bin`, and the
precomputed RoPE tables (`cos_sliding.npy`, `sin_sliding.npy`,
`cos_full.npy`, `sin_full.npy`).

All functions here are pure data-loading: no CoreML models are opened, no
ANE compute is invoked. This module is safely importable on any platform.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tokenizers import Tokenizer

MODEL_CONFIG_FILENAME = "model_config.json"
TOKENIZER_SUBDIR = "hf_model"
TOKENIZER_FILENAME = "tokenizer.json"

EMBED_DATA_FILENAME = "embed_tokens_q8.bin"
EMBED_SCALES_FILENAME = "embed_tokens_scales.bin"
PLE_DATA_FILENAME = "embed_tokens_per_layer_q8.bin"
PLE_SCALES_FILENAME = "embed_tokens_per_layer_scales.bin"
PROJECTION_FILENAME = "per_layer_projection.bin"
PER_LAYER_NORM_FILENAME = "per_layer_norm_weight.bin"
COS_SLIDING_FILENAME = "cos_sliding.npy"
SIN_SLIDING_FILENAME = "sin_sliding.npy"
COS_FULL_FILENAME = "cos_full.npy"
SIN_FULL_FILENAME = "sin_full.npy"

# Int8 -> fp16 dequant divisor (Swift: rowScale = scale / 127.0 * embed_scale)
INT8_MAX: float = 127.0


@dataclass(frozen=True)
class Gemma4Config:
    """Configuration parsed from the bundle's `model_config.json`.

    Every field has a real consumer per project CLAUDE.md §3
    (Schema->Consumer Audit). Fields present in `model_config.json` but
    NOT consumed by the Python chunked path are intentionally omitted
    so their presence in the dataclass would not falsely suggest they
    affect inference:

    - `model_name`, `architecture`, `sliding_window`, `bundle_path`:
      informational only; add back only when a real consumer exists.
    - `per_layer_model_projection_scale`, `per_layer_input_scale`:
      scales are baked into `chunk*.mlmodelc` weights; Python does not
      apply them externally.
    """

    hidden_size: int  # consumed: embed_token reshape
    num_layers: int  # consumed: PLE dim = num_layers * per_layer_dim
    vocab_size: int  # consumed: QuantizedEmbedding bounds + token validation
    context_length: int  # consumed: compared against effective_context at load
    per_layer_dim: int  # consumed: PLE dim computation
    bos_token_id: int  # consumed: generate() prepends BOS; _prewarm uses it
    eos_token_id: int  # consumed: generate() EOS stopping
    embed_scale: float  # consumed: QuantizedEmbedding global_scale for tokens
    per_layer_embed_scale: float  # consumed: QuantizedEmbedding global_scale for PLE

    @classmethod
    def from_bundle(cls, bundle_path: Path) -> Gemma4Config:
        """Load Gemma4Config from `<bundle_path>/model_config.json`."""
        config_path = bundle_path / MODEL_CONFIG_FILENAME
        if not config_path.is_file():
            raise FileNotFoundError(f"Gemma 4 bundle missing model_config.json at {config_path}")
        with config_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        try:
            return cls(
                hidden_size=int(data["hidden_size"]),
                num_layers=int(data["num_hidden_layers"]),
                vocab_size=int(data["vocab_size"]),
                context_length=int(data["context_length"]),
                per_layer_dim=int(data["per_layer_dim"]),
                bos_token_id=int(data["bos_token_id"]),
                eos_token_id=int(data["eos_token_id"]),
                embed_scale=float(data["embed_scale"]),
                per_layer_embed_scale=float(data["per_layer_embed_scale"]),
            )
        except KeyError as err:
            raise KeyError(
                f"model_config.json at {config_path} missing required key {err}"
            ) from err


def load_tokenizer(bundle_path: Path) -> Tokenizer:
    """Load the HuggingFace tokenizer from `<bundle_path>/hf_model/tokenizer.json`.

    Returns a `tokenizers.Tokenizer` instance. Import is deferred so that
    this module stays importable without the `[ane]` extras installed.
    """
    from tokenizers import Tokenizer as _Tokenizer

    tokenizer_path = bundle_path / TOKENIZER_SUBDIR / TOKENIZER_FILENAME
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Gemma 4 bundle missing tokenizer at {tokenizer_path}")
    return _Tokenizer.from_file(str(tokenizer_path))


class QuantizedEmbedding:
    """Memory-mapped int8 embedding table with per-row fp16 scale.

    Dequant: `fp16_out = int8_val * (scale_fp16 / 127.0) * global_scale`.
    Matches Swift `EmbeddingLookup` (EmbeddingLookup.swift:24-84) without
    the Accelerate vectorization — numpy's broadcast is already SIMD on
    Apple Silicon via BLAS.
    """

    def __init__(
        self,
        data_path: Path,
        scales_path: Path,
        *,
        vocab_size: int,
        dim: int,
        global_scale: float,
    ) -> None:
        if not data_path.is_file():
            raise FileNotFoundError(f"Embedding data missing: {data_path}")
        if not scales_path.is_file():
            raise FileNotFoundError(f"Embedding scales missing: {scales_path}")

        expected_data_bytes = vocab_size * dim
        expected_scales_bytes = vocab_size * 2  # fp16
        actual_data = data_path.stat().st_size
        actual_scales = scales_path.stat().st_size
        if actual_data != expected_data_bytes:
            raise ValueError(
                f"{data_path}: expected {expected_data_bytes} bytes "
                f"(vocab={vocab_size} x dim={dim} int8), got {actual_data}"
            )
        if actual_scales != expected_scales_bytes:
            raise ValueError(
                f"{scales_path}: expected {expected_scales_bytes} bytes "
                f"(vocab={vocab_size} fp16), got {actual_scales}"
            )

        self._data = np.memmap(data_path, dtype=np.int8, mode="r").reshape(vocab_size, dim)
        self._scales = np.memmap(scales_path, dtype=np.float16, mode="r")
        self._vocab_size = vocab_size
        self._dim = dim
        self._global_scale = np.float32(global_scale)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def lookup(self, token_id: int) -> np.ndarray:
        """Return dequantized embedding for token_id as fp16 (dim,)."""
        if not 0 <= token_id < self._vocab_size:
            raise ValueError(f"token_id {token_id} out of range [0, {self._vocab_size})")
        row_int8 = self._data[token_id]  # (dim,) int8
        row_scale = np.float32(self._scales[token_id]) / np.float32(INT8_MAX)
        row_scale = row_scale * self._global_scale
        row_fp32 = row_int8.astype(np.float32) * row_scale
        return row_fp32.astype(np.float16)


@dataclass(frozen=True)
class RoPETables:
    """Precomputed cos/sin tables for sliding (head_dim=256) and full (head_dim=512)
    attention paths. Stored as fp16 [positions, head_dim]. The on-disk tables
    cover up to 1024 positions even though `context_length=2048`; positions
    beyond `positions` will raise in the engine's mask/RoPE slicing step.
    """

    cos_sliding: np.ndarray
    sin_sliding: np.ndarray
    cos_full: np.ndarray
    sin_full: np.ndarray

    @property
    def max_positions(self) -> int:
        return int(self.cos_sliding.shape[0])

    @property
    def sliding_head_dim(self) -> int:
        return int(self.cos_sliding.shape[1])

    @property
    def full_head_dim(self) -> int:
        return int(self.cos_full.shape[1])


def load_rope_tables(bundle_path: Path) -> RoPETables:
    """Memory-map cos/sin .npy tables for both sliding and full attention paths."""
    files = {
        "cos_sliding": bundle_path / COS_SLIDING_FILENAME,
        "sin_sliding": bundle_path / SIN_SLIDING_FILENAME,
        "cos_full": bundle_path / COS_FULL_FILENAME,
        "sin_full": bundle_path / SIN_FULL_FILENAME,
    }
    for label, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"RoPE table missing ({label}): {path}")

    tables = {k: np.load(p, mmap_mode="r") for k, p in files.items()}
    for label, arr in tables.items():
        if arr.dtype != np.float16:
            raise ValueError(f"RoPE table {label} has dtype {arr.dtype}, expected float16")
        if arr.ndim != 2:
            raise ValueError(f"RoPE table {label} has shape {arr.shape}, expected 2D")

    cos_s, sin_s = tables["cos_sliding"], tables["sin_sliding"]
    cos_f, sin_f = tables["cos_full"], tables["sin_full"]
    if cos_s.shape != sin_s.shape:
        raise ValueError(f"cos_sliding {cos_s.shape} != sin_sliding {sin_s.shape}")
    if cos_f.shape != sin_f.shape:
        raise ValueError(f"cos_full {cos_f.shape} != sin_full {sin_f.shape}")
    if cos_s.shape[0] != cos_f.shape[0]:
        raise ValueError(f"sliding positions {cos_s.shape[0]} != full positions {cos_f.shape[0]}")

    return RoPETables(cos_sliding=cos_s, sin_sliding=sin_s, cos_full=cos_f, sin_full=sin_f)


def load_per_layer_projection(bundle_path: Path, *, expected_shape: tuple[int, int]) -> np.ndarray:
    """Load `per_layer_projection.bin` as fp16 and reshape to `expected_shape`.

    Swift loads this as fp16 then converts to fp32 for matmul; we keep fp16
    on disk and promote at use site.
    """
    path = bundle_path / PROJECTION_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"per_layer_projection missing: {path}")
    expected_bytes = expected_shape[0] * expected_shape[1] * 2
    actual = path.stat().st_size
    if actual != expected_bytes:
        raise ValueError(
            f"{path}: expected {expected_bytes} bytes (shape {expected_shape} fp16), got {actual}"
        )
    arr = np.memmap(path, dtype=np.float16, mode="r").reshape(expected_shape)
    return arr


def load_per_layer_norm_weight(bundle_path: Path, *, expected_dim: int) -> np.ndarray:
    """Load `per_layer_norm_weight.bin` as fp32 [expected_dim]."""
    path = bundle_path / PER_LAYER_NORM_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"per_layer_norm_weight missing: {path}")
    expected_bytes = expected_dim * 4
    actual = path.stat().st_size
    if actual != expected_bytes:
        raise ValueError(
            f"{path}: expected {expected_bytes} bytes ({expected_dim} fp32), got {actual}"
        )
    return np.memmap(path, dtype=np.float32, mode="r").reshape((expected_dim,))


def load_token_embedding(bundle_path: Path, cfg: Gemma4Config) -> QuantizedEmbedding:
    """Load the primary token embedding table (int8 [vocab, hidden]).

    Applies `embed_scale` from config as the global multiplier.
    """
    return QuantizedEmbedding(
        data_path=bundle_path / EMBED_DATA_FILENAME,
        scales_path=bundle_path / EMBED_SCALES_FILENAME,
        vocab_size=cfg.vocab_size,
        dim=cfg.hidden_size,
        global_scale=cfg.embed_scale,
    )


def load_per_layer_embedding(bundle_path: Path, cfg: Gemma4Config) -> QuantizedEmbedding:
    """Load the per-layer embedding (PLE) table (int8 [vocab, num_layers * per_layer_dim]).

    Applies `per_layer_embed_scale` from config as the global multiplier.
    """
    return QuantizedEmbedding(
        data_path=bundle_path / PLE_DATA_FILENAME,
        scales_path=bundle_path / PLE_SCALES_FILENAME,
        vocab_size=cfg.vocab_size,
        dim=cfg.num_layers * cfg.per_layer_dim,
        global_scale=cfg.per_layer_embed_scale,
    )
