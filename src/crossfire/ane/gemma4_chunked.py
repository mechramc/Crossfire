"""Gemma 4 E2B chunked CoreML inference engine for Apple Neural Engine.

Python port of the Gemma 4 E2B chunked on-device inference path, adapted from
`vendor/coreml-llm/Sources/CoreMLLLM/ChunkedEngine.swift` (MIT,
john-rocky/CoreML-LLM). Targets the root `swa-2k` variant of the pre-converted
bundle at `models/gemma-4-E2B-coreml/` (three `chunk*.mlmodelc` models).

Unlike the Swift reference (which manually manages KV buffers via IOSurface),
these `.mlmodelc` artifacts are STATEFUL: chunk1 and chunk2 own internal
`kv_cache_0` state via Apple's MLState API. `CompiledMLModel.make_state()`
returns an opaque state handle that is passed to `predict(..., state=...)`
and mutated in-place by the ANE. No Python-side KV buffer management is
required.

Effective context width is determined by the on-disk `causal_mask` and
`update_mask` input shapes (observed: 512 slots for the swa-2k variant).
Positions beyond `effective_context` cannot be decoded and will raise.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from crossfire.ane.gemma4_assets import (
    Gemma4Config,
    QuantizedEmbedding,
    RoPETables,
    load_per_layer_embedding,
    load_rope_tables,
    load_token_embedding,
    load_tokenizer,
)
from crossfire.ane.gemma4_masks import causal_mask_full, update_mask

if TYPE_CHECKING:
    import coremltools as ct
    from tokenizers import Tokenizer

CHUNK_FILE_PATTERN = "chunk*.mlmodelc"
CHUNK_METADATA_FILENAME = "metadata.json"
CAUSAL_MASK_INPUT = "causal_mask"
UPDATE_MASK_INPUT = "update_mask"

COMPUTE_UNIT_ALIASES = {
    "all": "ALL",
    "cpu_only": "CPU_ONLY",
    "cpu_and_gpu": "CPU_AND_GPU",
    "cpu_and_ne": "CPU_AND_NE",
}


@dataclass(frozen=True)
class GenerationResult:
    """Output of `Gemma4ChunkedEngine.generate()`.

    All timing fields are wallclock, measured on the calling thread.
    `ttft_ms` is time from start of prefill to the first generated token
    being sampled (a.k.a. "time to first token").
    """

    text: str
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float
    decode_tok_s: float
    total_tok_s: float


def _read_chunk_metadata(chunk_path: Path) -> dict[str, Any]:
    """Read `metadata.json` from a compiled `.mlmodelc` directory.

    coremltools' `CompiledMLModel` does not expose `get_spec()`, so we read
    the sibling `metadata.json` (written by the converter) to enumerate the
    chunk's input/output/state schemas.
    """
    meta_path = chunk_path / CHUNK_METADATA_FILENAME
    if not meta_path.is_file():
        raise FileNotFoundError(f"chunk metadata missing: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        if not data:
            raise ValueError(f"{meta_path} is an empty list")
        return data[0]
    if not isinstance(data, dict):
        raise ValueError(f"{meta_path} has unexpected shape: {type(data).__name__}")
    return data


def _parse_shape(raw_shape: str) -> tuple[int, ...]:
    """Parse a metadata.json shape string like '[1, 1, 1, 512]'."""
    stripped = raw_shape.strip().strip("[]")
    if not stripped:
        return ()
    return tuple(int(tok.strip()) for tok in stripped.split(","))


def _discover_chunks(bundle_path: Path) -> list[Path]:
    """Return chunk*.mlmodelc paths sorted lexicographically."""
    chunks = sorted(bundle_path.glob(CHUNK_FILE_PATTERN))
    if not chunks:
        raise FileNotFoundError(f"No {CHUNK_FILE_PATTERN} found under {bundle_path}")
    return chunks


def _extract_context_width(chunk_meta: dict[str, Any]) -> int:
    """Extract the effective context width from the causal_mask input shape.

    The causal_mask input has shape (1, 1, 1, ctx); we return ctx.
    """
    for inp in chunk_meta.get("inputSchema", []):
        if inp.get("name") == CAUSAL_MASK_INPUT:
            shape = _parse_shape(inp["shape"])
            if len(shape) != 4:
                raise ValueError(f"causal_mask has unexpected rank: {shape}")
            return shape[-1]
    raise ValueError(
        f"chunk metadata has no '{CAUSAL_MASK_INPUT}' input; "
        f"found {[i['name'] for i in chunk_meta.get('inputSchema', [])]}"
    )


def _compute_unit_from_string(name: str) -> ct.ComputeUnit:
    import coremltools as ct

    try:
        enum_name = COMPUTE_UNIT_ALIASES[name.lower()]
    except KeyError as err:
        raise ValueError(
            f"unknown compute_units {name!r}; expected one of {sorted(COMPUTE_UNIT_ALIASES)}"
        ) from err
    return getattr(ct.ComputeUnit, enum_name)


class Gemma4ChunkedEngine:
    """Chunked CoreML inference engine for Gemma 4 E2B on Apple Neural Engine.

    Not thread-safe; one engine per inference thread. Construct via `load()`;
    the constructor is internal and expects fully-initialized collaborators.
    """

    def __init__(
        self,
        *,
        config: Gemma4Config,
        tokenizer: Tokenizer,
        token_embedding: QuantizedEmbedding,
        per_layer_embedding: QuantizedEmbedding,
        rope_tables: RoPETables,
        chunks: list[Any],  # list[CompiledMLModel]
        states: list[Any | None],
        effective_context: int,
    ) -> None:
        self._config = config
        self._tokenizer = tokenizer
        self._embed = token_embedding
        self._ple = per_layer_embedding
        self._rope = rope_tables
        self._chunks = chunks
        self._states = states
        self._effective_context = effective_context
        self._position = 0

    @classmethod
    def load(
        cls,
        bundle_path: Path,
        *,
        compute_units: str = "cpu_and_ne",
    ) -> Gemma4ChunkedEngine:
        """Load a bundle from disk and prewarm the ANE.

        Parameters
        ----------
        bundle_path : Path
            Directory containing `model_config.json`, `chunk*.mlmodelc`, the
            quantized embedding tables, RoPE .npy tables, and `hf_model/tokenizer.json`.
        compute_units : str, default "cpu_and_ne"
            One of "all", "cpu_only", "cpu_and_gpu", "cpu_and_ne".
        """
        import coremltools as ct

        cfg = Gemma4Config.from_bundle(bundle_path)
        tokenizer = load_tokenizer(bundle_path)
        tok_emb = load_token_embedding(bundle_path, cfg)
        ple_emb = load_per_layer_embedding(bundle_path, cfg)
        rope = load_rope_tables(bundle_path)

        chunk_paths = _discover_chunks(bundle_path)
        cu = _compute_unit_from_string(compute_units)

        chunks: list[Any] = []
        states: list[Any | None] = []
        effective_context: int | None = None

        for chunk_path in chunk_paths:
            meta = _read_chunk_metadata(chunk_path)
            ctx = _extract_context_width(meta)
            if effective_context is None:
                effective_context = ctx
            elif ctx != effective_context:
                raise ValueError(
                    f"chunk {chunk_path.name} has ctx={ctx} but earlier chunks "
                    f"have ctx={effective_context}; mixed variants are unsupported"
                )

            model = ct.models.CompiledMLModel(str(chunk_path), compute_units=cu)
            chunks.append(model)
            # Chunks with a state schema get a fresh state; others get None.
            state = model.make_state() if meta.get("stateSchema") else None
            states.append(state)

        assert effective_context is not None
        # Cross-validate: the chunks' on-disk mask width must not exceed what
        # model_config.json advertises as context_length. If the config claims
        # support for more context than the chunks can actually serve, the
        # config is misleading and decode will silently truncate. Raise rather
        # than silently downgrading.
        if effective_context > cfg.context_length:
            raise ValueError(
                f"chunk effective_context={effective_context} exceeds "
                f"model_config.context_length={cfg.context_length}; "
                f"bundle is inconsistent"
            )

        engine = cls(
            config=cfg,
            tokenizer=tokenizer,
            token_embedding=tok_emb,
            per_layer_embedding=ple_emb,
            rope_tables=rope,
            chunks=chunks,
            states=states,
            effective_context=effective_context,
        )
        engine._prewarm()
        return engine

    @property
    def config(self) -> Gemma4Config:
        return self._config

    @property
    def current_position(self) -> int:
        return self._position

    @property
    def effective_context(self) -> int:
        return self._effective_context

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def reset(self) -> None:
        """Zero KV state and reset position to 0.

        Reallocates each chunk's `MLState` so decode can start fresh.
        """
        new_states: list[Any | None] = []
        for chunk, prev in zip(self._chunks, self._states, strict=True):
            new_states.append(chunk.make_state() if prev is not None else None)
        self._states = new_states
        self._position = 0

    def embed_token(self, token_id: int) -> np.ndarray:
        """Return the scaled fp16 token embedding as (1, 1, hidden_size)."""
        return self._embed.lookup(token_id).reshape(1, 1, self._config.hidden_size)

    def predict_step(self, token_id: int, position: int) -> int:
        """Advance one decode step at `position` on `token_id`.

        Returns the argmax next-token id from the final chunk.
        Mutates internal KV state for stateful chunks.
        """
        if position < 0 or position >= self._effective_context:
            raise ValueError(f"position {position} out of range [0, {self._effective_context})")
        if position >= self._rope.max_positions:
            raise ValueError(f"position {position} >= rope max {self._rope.max_positions}")

        hidden = (
            self._embed.lookup(token_id).reshape(1, 1, self._config.hidden_size).astype(np.float16)
        )
        per_layer = (
            self._ple.lookup(token_id)
            .reshape(1, 1, self._config.num_layers * self._config.per_layer_dim)
            .astype(np.float16)
        )
        cos_s, sin_s = self._rope_slice(position, sliding=True)
        cos_f, sin_f = self._rope_slice(position, sliding=False)
        causal = causal_mask_full(position, context_length=self._effective_context)
        upd = update_mask(position, context_length=self._effective_context)

        base_inputs = {
            "per_layer_combined": per_layer,
            "cos_s": cos_s,
            "sin_s": sin_s,
            "cos_f": cos_f,
            "sin_f": sin_f,
            CAUSAL_MASK_INPUT: causal,
            UPDATE_MASK_INPUT: upd,
        }

        # Sequential per-chunk forward; state is mutated in place for stateful chunks.
        hidden_current = hidden
        shared_kv: dict[str, np.ndarray] = {}
        token_out: int | None = None
        last_chunk_idx = len(self._chunks) - 1
        for idx, (chunk, state) in enumerate(zip(self._chunks, self._states, strict=True)):
            inputs: dict[str, Any] = {"hidden_states": hidden_current}
            if idx == last_chunk_idx:
                # Final chunk: consumes shared_kv, doesn't need update_mask
                for k in (
                    "per_layer_combined",
                    "cos_s",
                    "sin_s",
                    "cos_f",
                    "sin_f",
                    CAUSAL_MASK_INPUT,
                ):
                    inputs[k] = base_inputs[k]
                inputs.update(shared_kv)
            else:
                inputs.update(base_inputs)

            if state is not None:
                outputs = chunk.predict(inputs, state=state)
            else:
                outputs = chunk.predict(inputs)

            if "hidden_states_out" in outputs:
                hidden_current = outputs["hidden_states_out"].astype(np.float16)

            # Collect shared-KV outputs (kv13_k/v, kv14_k/v) for downstream chunks
            for k, v in outputs.items():
                if k.startswith("kv") and hasattr(v, "dtype"):
                    shared_kv[k] = v.astype(np.float16)

            if "token_id" in outputs:
                token_out = int(outputs["token_id"][0])

        if token_out is None:
            raise RuntimeError("no chunk produced a 'token_id' output; check chunk schemas")
        return token_out

    def run_prefill(self, token_ids: Sequence[int]) -> int:
        """Prefill `token_ids` starting at `current_position`, return next token id.

        Implementation note: iterates `predict_step` per token. The Swift
        reference ships batched `prefill_chunk*.mlmodelc` models for N=512
        batched prefill; those are not consumed here. This is a correctness-
        first path, not a latency-optimized one. See T-0609a.1 follow-up.
        """
        if not token_ids:
            raise ValueError("token_ids must be non-empty")
        if self._position + len(token_ids) > self._effective_context:
            raise ValueError(
                f"prefill of {len(token_ids)} tokens from position "
                f"{self._position} exceeds context {self._effective_context}"
            )

        next_token: int | None = None
        for tid in token_ids:
            next_token = self.predict_step(int(tid), self._position)
            self._position += 1
        assert next_token is not None
        return next_token

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        stop_on_eos: bool = True,
    ) -> GenerationResult:
        """Tokenize `prompt`, prefill, decode up to `max_tokens`, return result.

        BOS token is prepended to the prompt tokenization.
        Decoding stops at EOS (if `stop_on_eos`) or after `max_tokens`.
        """
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {max_tokens}")

        self.reset()
        prompt_ids: list[int] = [self._config.bos_token_id]
        prompt_ids.extend(self._tokenizer.encode(prompt).ids)

        t_total_start = time.perf_counter()
        t_prefill_start = time.perf_counter()
        first_token = self.run_prefill(prompt_ids)
        t_prefill_end = time.perf_counter()

        generated: list[int] = [first_token]
        if not (stop_on_eos and first_token == self._config.eos_token_id):
            next_tid = first_token
            while len(generated) < max_tokens:
                next_tid = self.predict_step(next_tid, self._position)
                self._position += 1
                generated.append(next_tid)
                if stop_on_eos and next_tid == self._config.eos_token_id:
                    break
                if self._position >= self._effective_context:
                    break

        t_total_end = time.perf_counter()

        text = self._tokenizer.decode(generated)
        ttft_ms = (t_prefill_end - t_prefill_start) * 1000.0
        total_elapsed = t_total_end - t_total_start
        decode_elapsed = t_total_end - t_prefill_end
        decode_tok_s = (
            (len(generated) - 1) / decode_elapsed
            if decode_elapsed > 0 and len(generated) > 1
            else 0.0
        )
        total_tok_s = (
            (len(prompt_ids) + len(generated)) / total_elapsed if total_elapsed > 0 else 0.0
        )

        return GenerationResult(
            text=text,
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(generated),
            ttft_ms=ttft_ms,
            decode_tok_s=decode_tok_s,
            total_tok_s=total_tok_s,
        )

    def _rope_slice(self, position: int, *, sliding: bool) -> tuple[np.ndarray, np.ndarray]:
        """Extract RoPE cos/sin at `position` reshaped to (1, 1, 1, head_dim)."""
        if sliding:
            cos_src = self._rope.cos_sliding
            sin_src = self._rope.sin_sliding
        else:
            cos_src = self._rope.cos_full
            sin_src = self._rope.sin_full
        cos = cos_src[position : position + 1].reshape(1, 1, 1, -1).copy()
        sin = sin_src[position : position + 1].reshape(1, 1, 1, -1).copy()
        return cos.astype(np.float16), sin.astype(np.float16)

    def _prewarm(self, *, steps: int = 4) -> None:
        """Run `steps` dummy decodes then reset, matching Swift lines 616-621.

        This finalizes ANE compile schedules and weight layouts so the first
        real predict call doesn't pay that cost.
        """
        bos = self._config.bos_token_id
        try:
            for i in range(steps):
                self.predict_step(bos, i)
        finally:
            self.reset()
