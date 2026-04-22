"""Causal mask and update mask builders for Gemma 4 chunked inference.

Pure-numpy utilities, no CoreML dependency. Fp16 convention:
- Allowed positions: 0.0 (additive mask)
- Blocked positions: -65504.0 (fp16 min, not -1e9 which overflows fp16)

The Gemma 4 E2B chunked models consume three masks per decode step:
- Full-attention causal mask over the full context window
  Shape (1, 1, 1, ctx), blocks positions > current.
- Sliding-window causal mask over the local window (W=512 default)
  Shape (1, 1, 1, W), blocks padding slots (cache is right-aligned).
- Update mask for full-attention KV scatter
  Shape (1, 1, ctx, 1), one-hot at current position.

References: Swift `ChunkedEngine.makeCausalMask` / `makeSlidingCausalMask`
/ `makeUpdateMask` (`vendor/coreml-llm/Sources/CoreMLLLM/ChunkedEngine.swift`
lines 1692-1729).
"""

from __future__ import annotations

import numpy as np

FP16_BLOCK_FILL: float = -65504.0
FP16_ALLOW_FILL: float = 0.0


def causal_mask_full(position: int, *, context_length: int) -> np.ndarray:
    """Full-attention causal mask for single-token decode.

    Returns fp16 (1, 1, 1, context_length).
    - mask[..., j] = 0.0 if j <= position else -65504.0.
    """
    if position < 0:
        raise ValueError(f"position must be >= 0, got {position}")
    if position >= context_length:
        raise ValueError(f"position {position} >= context_length {context_length}")
    mask = np.full((1, 1, 1, context_length), FP16_BLOCK_FILL, dtype=np.float16)
    mask[..., : position + 1] = FP16_ALLOW_FILL
    return mask


def causal_mask_sliding(position: int, *, sliding_window: int) -> np.ndarray:
    """Sliding-window causal mask for single-token decode.

    The sliding cache holds W slots, right-aligned — valid tokens occupy
    the last `min(position+1, W)` slots, leading slots are padding.
    Returns fp16 (1, 1, 1, sliding_window).
    """
    if position < 0:
        raise ValueError(f"position must be >= 0, got {position}")
    if sliding_window <= 0:
        raise ValueError(f"sliding_window must be > 0, got {sliding_window}")
    valid = min(position + 1, sliding_window)
    mask = np.full((1, 1, 1, sliding_window), FP16_BLOCK_FILL, dtype=np.float16)
    mask[..., sliding_window - valid :] = FP16_ALLOW_FILL
    return mask


def update_mask(position: int, *, context_length: int) -> np.ndarray:
    """Full-attention KV scatter mask: one-hot fp16 at `min(position, ctx-1)`.

    Returns fp16 (1, 1, context_length, 1).
    """
    if position < 0:
        raise ValueError(f"position must be >= 0, got {position}")
    if context_length <= 0:
        raise ValueError(f"context_length must be > 0, got {context_length}")
    mask = np.zeros((1, 1, context_length, 1), dtype=np.float16)
    mask[0, 0, min(position, context_length - 1), 0] = np.float16(1.0)
    return mask
