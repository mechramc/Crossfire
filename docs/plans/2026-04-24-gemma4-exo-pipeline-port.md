# Gemma 4 EXO Pipeline Port — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Lift EXO's hard ban on Pipeline parallelism for Gemma 4 by making `Gemma4TextModel.make_cache()` and forward-pass cache indexing shard-aware, so a 2-rank pipeline split (PC RTX 5090 + Mac M4 Max over WiFi) becomes a viable distributed execution mode for `mlx-community/gemma-4-31b-it-4bit`.

**Architecture:** Three coordinated edits across two repos. (1) In EXO `placement.py`, replace the blanket Gemma 4 Pipeline ban with a model-aware constraint check that enforces `K ≤ min(previous_kvs[M:N])` so KV-shared layers stay co-located with their KV producers. (2) In EXO `auto_parallel.py:pipeline_auto_parallel`, add a `Gemma4TextModel` special-case (mirroring the existing `Step35InnerModel` block) that slices `layer_types` and `previous_kvs` to the local shard and re-bases all global indices to local. (3) In the same EXO function, monkey-patch `Model.make_cache` (mirroring `_patch_hybrid_cache`) to size caches against the local shard's first-kv-shared count rather than the global one. No mlx-lm fork required.

**Tech Stack:** Python 3.13, mlx 0.31.1, mlx-lm 0.31.3, EXO 1.0 @ commit `49670c8`, MLX distributed (TCP) over WiFi.

**Validation principle:** After each code task, type-check + run the existing EXO test suite. After implementation, run a single-shard regression to prove we didn't break Gemma 4 for everyone, then a two-node Pipeline load + chat, then a side-by-side dmon capture (Tensor vs Pipeline) for the writeup.

**Constraint analysis (load-bearing):** For Gemma 4 31B with `num_hidden_layers=60`, `num_kv_shared_layers=20`, `sliding_window_pattern=5` → layer_types is `SSSSF` × 12 → last full-attention in [0, M=40) is index 39, last sliding in [0, M=40) is index 38 → `min_producer = 38`. Therefore for a 2-rank split, the boundary K must satisfy `1 ≤ K ≤ 38`. EXO's default even split (K = 30) satisfies this. 3-rank splits would require the second boundary K₂ at ~40 > 38 — not supported in V1.

**Out of scope (deferred):**
- Upstream mlx-lm PR — file separately as a follow-up after this lands. Path B ships against EXO only.
- 3+ rank Gemma 4 Pipeline splits — raise `NotImplementedError` with clear message.
- Generalizing the constraint to other KV-sharing models — leave a TODO.

**Risk notes:**
- After 10+ messages, Rule 6 applies: re-read `gemma4_text.py`, `placement.py`, and `auto_parallel.py` chunks before each Edit.
- The Edit tool fails silently on stale `old_string`. Always re-read the exact target span before editing.

---

## Task 1: Set up isolated worktree and branch

**Files:** none yet.

**Step 1.1: Create worktree off `main`.**

Run:
```bash
cd C:/Github/Crossfire
git worktree add ../Crossfire-gemma4-pipeline -b feat/gemma4-exo-pipeline
```
Expected: `Preparing worktree (new branch 'feat/gemma4-exo-pipeline')` followed by checkout output.

**Step 1.2: Confirm clean tree and matching commit.**

Run: `cd ../Crossfire-gemma4-pipeline && git status && git log --oneline -1`
Expected: `nothing to commit, working tree clean`, current commit matches `89b23f7`.

**Step 1.3: Locate the EXO checkout that the WSL daemon uses.**

The runtime EXO is at WSL path `/home/mechramc/crossfire/exo`. Edits to its source affect the live daemon. Confirm we are editing the SAME tree:
```bash
wsl -d Ubuntu -- bash -c 'cd /home/mechramc/crossfire/exo && git rev-parse HEAD && git status -s'
```
Expected: a commit hash + clean (or known-clean) working tree. **All edits in this plan target the WSL EXO checkout.** The `Crossfire-gemma4-pipeline` worktree on the Windows side is for the writeup, dmon traces, and plan tracking.

**Step 1.4: Create a checkpoint branch in the WSL EXO checkout.**

```bash
wsl -d Ubuntu -- bash -c 'cd /home/mechramc/crossfire/exo && git checkout -b feat/gemma4-pipeline-port'
```
Expected: `Switched to a new branch 'feat/gemma4-pipeline-port'`.

**Step 1.5: Commit (no code change yet — branch marker).**

```bash
wsl -d Ubuntu -- bash -c 'cd /home/mechramc/crossfire/exo && git commit --allow-empty -m "wip: start gemma4 pipeline port"'
```

---

## Task 2: Add `compute_gemma4_max_pipeline_split` helper

**Files:**
- Modify: `/home/mechramc/crossfire/exo/src/exo/master/placement.py` (add helper near top, before `set_instance`)

**Goal:** Pure function that returns the maximum legal pipeline split position K for a given Gemma 4 config, computed from the same logic that `Gemma4TextModel.__init__` uses to build `previous_kvs`.

**Step 2.1: Re-read the relevant `Gemma4TextModel.__init__` block.**

Read `gemma4_text.py:425-433` (the `previous_kvs` construction). Confirm the formula:
```python
M = N - num_kv_shared_layers
kvs_by_type[layer_types[i]] = i  for i in range(M)   # last-occurrence per type
previous_kvs[j] = kvs_by_type[layer_types[j]]        for j in range(M, N)
```

**Step 2.2: Re-read `placement.py` import block + lines 156-166** (the existing ban).

**Step 2.3: Add the helper function above `set_instance` in `placement.py`.**

Exact code to insert (place after the imports, before the first function definition):

```python
def compute_gemma4_max_pipeline_split(
    num_hidden_layers: int,
    num_kv_shared_layers: int,
    sliding_window_pattern: int,
    layer_types: list[str] | None = None,
) -> int:
    """Return the largest legal 2-rank pipeline boundary K for a Gemma 4 model.

    A 2-rank split places layers [0, K) on rank 0 and [K, N) on rank 1.
    Gemma 4's KV-shared tail layers reference earlier "producer" layers via
    previous_kvs. To avoid shipping KV state across the pipeline boundary,
    every shared layer's producer must be on the same rank as the consumer.

    For 2 ranks with the shared tail entirely on rank 1, this reduces to
    K <= min(previous_kvs[M:N]) where M = N - num_kv_shared_layers.

    Returns N-1 if there are no shared layers (any non-degenerate split works).
    Raises ValueError if no legal split exists (degenerate config).
    """
    N = num_hidden_layers
    if num_kv_shared_layers <= 0:
        return N - 1

    M = N - num_kv_shared_layers

    if layer_types is None:
        # Reproduce gemma4_text.ModelArgs.__post_init__ default
        pattern = ["sliding_attention"] * (sliding_window_pattern - 1) + [
            "full_attention"
        ]
        layer_types = (pattern * (N // len(pattern) + 1))[:N]

    # Reproduce previous_kvs construction (last-occurrence per type in [0, M))
    kvs_by_type: dict[str, int] = {}
    for i in range(M):
        kvs_by_type[layer_types[i]] = i

    producer_indices: list[int] = []
    for j in range(M, N):
        producer = kvs_by_type.get(layer_types[j])
        if producer is None:
            raise ValueError(
                f"Gemma 4 layer {j} has type {layer_types[j]!r} with no producer "
                f"in [0, {M}); model config is malformed."
            )
        producer_indices.append(producer)

    return min(producer_indices)
```

**Step 2.4: Verify the helper at the REPL.**

```bash
wsl -d Ubuntu -- /home/mechramc/crossfire/exo/.venv/bin/python -c "
from exo.master.placement import compute_gemma4_max_pipeline_split
# Gemma 4 31B
print('31B:', compute_gemma4_max_pipeline_split(60, 20, 5))   # expect 38
# Hypothetical Gemma 4 small with no shared layers
print('no-shared:', compute_gemma4_max_pipeline_split(20, 0, 5))   # expect 19
"
```
Expected:
```
31B: 38
no-shared: 19
```

**Step 2.5: Commit.**

```bash
wsl -d Ubuntu -- bash -c 'cd /home/mechramc/crossfire/exo && git add src/exo/master/placement.py && git commit -m "feat(placement): add compute_gemma4_max_pipeline_split helper"'
```

---

## Task 3: Lift Gemma 4 Pipeline ban with constraint check

**Files:**
- Modify: `/home/mechramc/crossfire/exo/src/exo/master/placement.py` lines 156-166

**Step 3.1: Re-read lines 130-170 of placement.py** to confirm exact span and surrounding context.

**Step 3.2: Replace the existing ban with model-aware filtering.**

Old span (lines 156-166):

```python
    if (
        command.sharding == Sharding.Pipeline
        and command.model_card.base_model.startswith("Gemma 4")
    ):
        cycles_with_sufficient_memory = [
            cycle for cycle in cycles_with_sufficient_memory if len(cycle) == 1
        ]
        if not cycles_with_sufficient_memory:
            raise ValueError(
                "Pipeline parallelism is not supported for Gemma 4; use tensor parallelism instead."
            )
```

New span (replace exactly):

```python
    if (
        command.sharding == Sharding.Pipeline
        and command.model_card.base_model.startswith("Gemma 4")
    ):
        # Gemma 4 KV-shared layers reference earlier "producer" layers via
        # previous_kvs. A pipeline split is legal only if the shared tail
        # stays co-located with its producers on the last rank. For V1 we
        # support 2-rank splits where K <= min(previous_kvs[M:N]); 3+ rank
        # splits would force K_2 > the producer boundary and are rejected.
        # TODO: generalize KV-co-location check beyond Gemma 4.
        max_split = compute_gemma4_max_pipeline_split(
            num_hidden_layers=command.model_card.num_hidden_layers,
            num_kv_shared_layers=command.model_card.num_kv_shared_layers,
            sliding_window_pattern=command.model_card.sliding_window_pattern,
        )
        legal_cycles: list[Cycle] = []
        for cycle in cycles_with_sufficient_memory:
            n_ranks = len(cycle)
            if n_ranks == 1:
                legal_cycles.append(cycle)
                continue
            if n_ranks > 2:
                # 3+ ranks not supported for Gemma 4 in V1
                continue
            # 2 ranks: EXO's pure-pipeline split is roughly even.
            # Use the same formula as get_shard_assignments_for_pipeline_parallel.
            approx_K = (command.model_card.num_hidden_layers + 1) // 2
            if approx_K <= max_split:
                legal_cycles.append(cycle)
        cycles_with_sufficient_memory = legal_cycles
        if not cycles_with_sufficient_memory:
            raise ValueError(
                f"No Gemma 4 pipeline-legal cycles found. The KV-shared tail "
                f"requires the boundary K <= {max_split} for "
                f"num_hidden_layers={command.model_card.num_hidden_layers}, "
                f"num_kv_shared_layers={command.model_card.num_kv_shared_layers}. "
                f"For 2-rank splits this means rank 0 holds layers [0, K) and "
                f"rank 1 holds layers [K, N). 3+ rank splits are not yet supported."
            )
```

**Step 3.3: Verify ModelCard exposes the required fields.**

```bash
wsl -d Ubuntu -- /home/mechramc/crossfire/exo/.venv/bin/python -c "
from exo.shared.types.master.model_card import ModelCard
import inspect
print(inspect.getsource(ModelCard))
" | head -80
```
Expected: confirms `num_hidden_layers`, `num_kv_shared_layers`, `sliding_window_pattern` are dataclass fields. **If any are missing**, add them to `ModelCard` (separate dataclass file under `src/exo/shared/types/master/`) and ensure the Gemma 4 model card definitions in `src/exo/shared/model_cards/` populate them. Make this a sub-step before continuing.

**Step 3.4: Type-check.**

```bash
wsl -d Ubuntu -- bash -c 'cd /home/mechramc/crossfire/exo && .venv/bin/python -m mypy src/exo/master/placement.py 2>&1 | tail -10'
```
If no mypy in the project, fall back to running ruff + a syntax sanity check:
```bash
wsl -d Ubuntu -- bash -c 'cd /home/mechramc/crossfire/exo && .venv/bin/ruff check src/exo/master/placement.py'
wsl -d Ubuntu -- bash -c 'cd /home/mechramc/crossfire/exo && .venv/bin/python -c "from exo.master import placement"'
```
Expected: clean import, no ruff errors.

**Step 3.5: Commit.**

```bash
wsl -d Ubuntu -- bash -c 'cd /home/mechramc/crossfire/exo && git add src/exo/master/placement.py && git commit -m "feat(placement): allow Gemma 4 Pipeline with KV-producer co-location check"'
```

---

## Task 4: Make `pipeline_auto_parallel` shard-aware for Gemma 4

**Files:**
- Modify: `/home/mechramc/crossfire/exo/src/exo/worker/engines/mlx/auto_parallel.py`

**Step 4.1: Re-read auto_parallel.py lines 14-60 (imports) and lines 275-400 (`pipeline_auto_parallel`).**

**Step 4.2: Add the inner-model import.**

Find the existing `from mlx_lm.models.gemma4 import Model as Gemma4Model` at line 22. Add immediately after it:

```python
from mlx_lm.models.gemma4_text import Gemma4TextModel
```

**Step 4.3: Add the Gemma 4 special-case block in `pipeline_auto_parallel`.**

Locate the existing `Step35InnerModel` block (auto_parallel.py:332-343). Insert the following block immediately AFTER the `Step35InnerModel` block and BEFORE the `Qwen3_5TextModelInner` block:

```python
    if isinstance(inner_model_instance, Gemma4TextModel):
        # Gemma 4 keeps two pieces of layer-indexed state that we must slice
        # to the local shard: layer_types (drives mask selection) and
        # previous_kvs (drives KV sharing for the tail layers).
        global_layer_types: list[str] = list(inner_model_instance.config.layer_types)
        global_N = inner_model_instance.num_hidden_layers
        global_num_shared = inner_model_instance.config.num_kv_shared_layers
        global_M = global_N - global_num_shared
        global_previous_kvs: list[int] = list(inner_model_instance.previous_kvs)

        # Validate: every previous_kv referenced by the local shard must be
        # an index that is also in the local shard. This is the placement
        # constraint check restated at the runtime layer.
        local_layer_indices = range(start_layer, end_layer)
        for j in local_layer_indices:
            prev_global = global_previous_kvs[j]
            if not (start_layer <= prev_global < end_layer):
                raise ValueError(
                    f"Gemma 4 pipeline shard [{start_layer}, {end_layer}) on rank "
                    f"{device_rank} requires layer {j} to read KV from layer "
                    f"{prev_global}, but that producer is on a different shard. "
                    f"This means placement.py's constraint check is incorrect; "
                    f"file a bug. (max legal K for this model is the min of "
                    f"previous_kvs[M:N] across the global model.)"
                )

        # Slice + rebase
        inner_model_instance.config.layer_types = global_layer_types[
            start_layer:end_layer
        ]
        inner_model_instance.num_hidden_layers = len(layers)
        # previous_kvs[i_local] = global_previous_kvs[i_local + start_layer] - start_layer
        inner_model_instance.previous_kvs = [
            global_previous_kvs[start_layer + i] - start_layer
            for i in range(len(layers))
        ]

        # Patch make_cache to use the LOCAL first_kv_shared count.
        # Compute how many of the local layers fall in the global shared tail.
        local_num_shared = max(0, end_layer - max(start_layer, global_M))

        original_make_cache = inner_model_instance.config  # placeholder; see below
        model_to_patch = model  # outer Model object that owns make_cache
        local_layer_types = inner_model_instance.config.layer_types

        def _gemma4_make_cache_local(
            _self,
            _local_N: int = len(layers),
            _local_num_shared: int = local_num_shared,
            _local_layer_types: list[str] = local_layer_types,
            _sliding_window: int = inner_model_instance.config.sliding_window,
        ) -> list[KVCache | RotatingKVCache]:
            from mlx_lm.models.cache import KVCache, RotatingKVCache
            first_kv_shared_local = _local_N - _local_num_shared
            caches: list[KVCache | RotatingKVCache] = []
            for i in range(first_kv_shared_local):
                if _local_layer_types[i] == "full_attention":
                    caches.append(KVCache())
                else:
                    caches.append(
                        RotatingKVCache(
                            max_size=_sliding_window,
                            keep=0,
                        )
                    )
            return caches

        # Bind as bound method on the outer Model
        import types as _types
        model_to_patch.make_cache = _types.MethodType(
            _gemma4_make_cache_local, model_to_patch
        )
```

**Step 4.4: Confirm the imports are still correct.**

`KVCache` is already imported at line 17 (`from mlx_lm.models.cache import ArraysCache, KVCache`). Need to add `RotatingKVCache` to that import. Edit line 17:

Old:
```python
from mlx_lm.models.cache import ArraysCache, KVCache
```

New:
```python
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache
```

(The local `from mlx_lm.models.cache import ...` inside `_gemma4_make_cache_local` is defensive; removing it would also work since the outer import is in scope.)

**Step 4.5: Run ruff + import smoke test.**

```bash
wsl -d Ubuntu -- bash -c 'cd /home/mechramc/crossfire/exo && .venv/bin/ruff check src/exo/worker/engines/mlx/auto_parallel.py && .venv/bin/python -c "from exo.worker.engines.mlx.auto_parallel import pipeline_auto_parallel; print(pipeline_auto_parallel)"'
```
Expected: no ruff errors, function reference printed.

**Step 4.6: Commit.**

```bash
wsl -d Ubuntu -- bash -c 'cd /home/mechramc/crossfire/exo && git add src/exo/worker/engines/mlx/auto_parallel.py && git commit -m "feat(auto_parallel): shard-aware Gemma 4 layer_types, previous_kvs, make_cache"'
```

---

## Task 5: Single-shard regression test (whole-model path unchanged)

**Goal:** Prove that when there is only 1 rank (start=0, end=num_hidden_layers, world_size=1), the patched code path produces identical behavior to upstream — same caches, same forward output.

**Files:**
- Create: `/home/mechramc/crossfire/exo/tests/worker/engines/mlx/test_gemma4_pipeline_shard.py`

**Step 5.1: Locate the existing test layout.**

```bash
wsl -d Ubuntu -- bash -c 'find /home/mechramc/crossfire/exo/tests -type d | head -20'
```
Pick a sensible path mirroring the source layout. Confirm the path above exists or create the directory.

**Step 5.2: Write the regression test.**

```python
# tests/worker/engines/mlx/test_gemma4_pipeline_shard.py
"""Regression: ensure single-shard (whole-model) Gemma 4 path is byte-identical
to the unpatched path. Catches accidental breakage of the most common case."""

from __future__ import annotations

import pytest


def _make_tiny_gemma4_config():
    """Build a minimal Gemma 4 config small enough to instantiate quickly."""
    from mlx_lm.models.gemma4_text import ModelArgs

    return ModelArgs(
        hidden_size=64,
        num_hidden_layers=10,
        intermediate_size=128,
        num_attention_heads=2,
        head_dim=32,
        global_head_dim=32,
        num_key_value_heads=1,
        num_kv_shared_layers=4,
        sliding_window=16,
        sliding_window_pattern=5,
        vocab_size=1024,
        vocab_size_per_layer_input=1024,
        hidden_size_per_layer_input=0,  # disable per-layer-input path for simplicity
        max_position_embeddings=128,
    )


def test_single_shard_make_cache_unchanged():
    """When end_layer == num_hidden_layers, our patched make_cache must
    produce a cache list identical in length and entry types to upstream."""
    from mlx_lm.models.gemma4 import Model as Gemma4Model

    cfg = _make_tiny_gemma4_config()
    model_baseline = Gemma4Model(cfg)
    baseline_cache = model_baseline.make_cache()

    # Apply pipeline_auto_parallel with single-rank metadata
    from exo.shared.types.worker.shards import PipelineShardMetadata
    from exo.worker.engines.mlx.auto_parallel import pipeline_auto_parallel
    import mlx.core as mx

    model_patched = Gemma4Model(cfg)
    meta = PipelineShardMetadata(
        start_layer=0,
        end_layer=cfg.num_hidden_layers,
        device_rank=0,
        world_size=1,
    )
    # Build a trivial single-rank group
    group = mx.distributed.init(strict=False)
    pipeline_auto_parallel(model_patched, group, meta, on_layer_loaded=None)

    patched_cache = model_patched.make_cache()

    assert len(patched_cache) == len(baseline_cache), (
        f"cache length changed: {len(patched_cache)} vs {len(baseline_cache)}"
    )
    for i, (b, p) in enumerate(zip(baseline_cache, patched_cache)):
        assert type(b) is type(p), (
            f"cache[{i}] type changed: {type(b).__name__} vs {type(p).__name__}"
        )


def test_two_shard_make_cache_lengths():
    """For a 2-shard split at K=6 on a 10-layer / 4-shared model:
    rank 0 has layers [0, 6) with 0 shared → 6 caches.
    rank 1 has layers [6, 10) with 4 shared → 0 caches (only the shared tail).
    """
    from mlx_lm.models.gemma4 import Model as Gemma4Model
    from exo.shared.types.worker.shards import PipelineShardMetadata
    from exo.worker.engines.mlx.auto_parallel import pipeline_auto_parallel
    import mlx.core as mx

    cfg = _make_tiny_gemma4_config()  # N=10, num_kv_shared=4 → M=6

    # Rank 0
    model_r0 = Gemma4Model(cfg)
    meta_r0 = PipelineShardMetadata(start_layer=0, end_layer=6, device_rank=0, world_size=2)
    group = mx.distributed.init(strict=False)
    pipeline_auto_parallel(model_r0, group, meta_r0, on_layer_loaded=None)
    cache_r0 = model_r0.make_cache()
    assert len(cache_r0) == 6, f"rank 0 expected 6 caches, got {len(cache_r0)}"

    # Rank 1
    model_r1 = Gemma4Model(cfg)
    meta_r1 = PipelineShardMetadata(start_layer=6, end_layer=10, device_rank=1, world_size=2)
    pipeline_auto_parallel(model_r1, group, meta_r1, on_layer_loaded=None)
    cache_r1 = model_r1.make_cache()
    # All 4 layers on rank 1 are in the shared tail → 0 own caches
    assert len(cache_r1) == 0, f"rank 1 expected 0 caches (all shared), got {len(cache_r1)}"


def test_max_pipeline_split_helper_31b():
    from exo.master.placement import compute_gemma4_max_pipeline_split
    assert compute_gemma4_max_pipeline_split(60, 20, 5) == 38


def test_max_pipeline_split_helper_no_shared():
    from exo.master.placement import compute_gemma4_max_pipeline_split
    assert compute_gemma4_max_pipeline_split(20, 0, 5) == 19


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Step 5.3: Run the tests.**

```bash
wsl -d Ubuntu -- bash -c 'cd /home/mechramc/crossfire/exo && .venv/bin/python -m pytest tests/worker/engines/mlx/test_gemma4_pipeline_shard.py -v 2>&1 | tail -30'
```
Expected: 4 passed. **If `mx.distributed.init` requires a specific env, document the workaround in the test and re-run. If `PipelineShardMetadata` constructor signature differs, adjust.**

**Step 5.4: Commit.**

```bash
wsl -d Ubuntu -- bash -c 'cd /home/mechramc/crossfire/exo && git add tests/ && git commit -m "test: add single-shard + 2-shard regression tests for gemma4 pipeline"'
```

---

## Task 6: Two-node Pipeline load test (no inference yet)

**Status (Session 27, 2026-04-24):** Load path works end-to-end; Mac shards
30 layers, PC shards 30 layers, model loads in ~46 s cold on the PC rank.
**The runtime tuple-passthrough bug surfaced and was fixed inline:**
`Gemma4TextModel.DecoderLayer.__call__` returns `(h, shared_kv, offset)`,
but the original `PipelineLastLayer.__call__` treated the return as a bare
`mx.array` and fed the tuple straight into `mx.distributed.send`, raising
`ValueError: Invalid type tuple received in array initialization` at
`auto_parallel.py:175`.

Fix landed in EXO branch `feat/gemma4-pipeline-port` at commit
`94c0ce6` (bundle:
`C:\Users\mechr\Downloads\gemma4-pipeline-fix-94c0ce6.bundle`,
SHA256 `6ac9f06075e8da9650b7fbcdff584be480c71583b42d82bd83e723e17502d83f`,
parent `d9959da`). `PipelineLastLayer.__call__` now detects a tuple return,
sends only `output[0]` across the rank boundary, and rebuilds the tuple
shape so the outer model loop's `h, kvs, offset = layer(...)` unpacking
keeps working. `PipelineFirstLayer.__call__` had its return annotation
relaxed to `mx.array | tuple[object, ...]` so the same path round-trips
through rank 0. Two contract tests added in
`src/exo/worker/engines/mlx/tests/test_gemma4_pipeline_shard.py`:
`test_pipeline_last_layer_tuple_passthrough` and
`test_pipeline_last_layer_bare_array_unchanged` (7/7 pass; pre-existing
ruff B905/I001 issues fixed in the same commit).

**Inference now blocks at the first `mx_barrier` collective** (`generate.py:326`,
called from `prefill`). py-spy on the PC runner at PID 314489 caught the
stack pinned at `mx_barrier (utils_mlx.py:774)` for 7+ minutes with no
"Starting prefill" log line ever emitted. Symptoms across three attempts:
attempt 1 hung at the same barrier, attempt 2 cleared it and crashed at
the tuple bug (root cause for the fix above), attempt 3 hung again.
Pattern is consistent with a transport handshake race in `mx.distributed`
ring init over WiFi on the first collective after a fresh
`mx.distributed.init`. There is no timeout / diagnostic in the current
`mx_barrier` implementation, so failures are silent hangs.

**Goal:** Prove the 2-node Pipeline path can complete load without raising. Don't run prefill.

**Files:** none modified; runs the existing daemon + Mac client.

**Step 6.1: Restart PC daemon clean.**

```bash
wsl -d Ubuntu -- bash -c 'pkill -f "[v]env/bin/python -m exo" 2>/dev/null; sleep 1; bash /tmp/start_exo_clean.sh'
```
Expected: daemon up, Mac peer detected via election.

**Step 6.2: From the Mac, request a Pipeline placement for `mlx-community/gemma-4-31b-it-4bit`.**

This requires the Mac dashboard to send a placement command with `sharding=Pipeline`. The default sharding in `api/main.py:get_placement` is already `Sharding.Pipeline`, so this should be the default. **If the dashboard hard-codes Tensor for Gemma 4 anywhere, find and remove that hard-code.**

**Step 6.3: Watch PC daemon log for the load.**

```bash
wsl -d Ubuntu -- bash -c 'tail -f /tmp/exo_pc.log | grep -iE "loading|sharding|pipeline|tensor|shard and load|rank|error"'
```
Expected: log shows `with pipeline parallelism` (or equivalent), then `Time taken to shard and load model: …`. **No** ValueError about KV co-location.

**Step 6.4: Verify VRAM occupancy.**

```bash
wsl -d Ubuntu -- nvidia-smi --query-gpu=memory.used --format=csv,noheader
```
Expected: For a 50/50 split, ~9 GB used (half of 18.4 GB).

**Step 6.5: If load fails, capture the exception and stop.**

Don't proceed to inference if load doesn't complete. Capture the full traceback to `/home/mechramc/exo_pipeline_load_FAIL_<timestamp>.log`, drop back to the planning loop, and diagnose with `superpowers:systematic-debugging`.

**Step 6.6: Commit a note marker.**

```bash
wsl -d Ubuntu -- bash -c 'cd /home/mechramc/crossfire/exo && git commit --allow-empty -m "checkpoint: two-node pipeline load succeeds for gemma-4-31b-it-4bit"'
```

---

## Task 7: Two-node Pipeline inference + correctness check

**Status (Session 27, 2026-04-24):** Blocked on the `mx_barrier` hang
described under Task 6. No tokens generated yet. Next-session
unblock candidates (rank-ordered):

1. Add a watchdog + diagnostic logging around the `mx_barrier` call in
   `generate.py:326` (timeout, peer-state dump, retry on first call only).
2. Add per-layer `logger.debug` markers at each `PipelineFirstLayer.recv`
   and `PipelineLastLayer.send` so a hang inside the layer ring is
   distinguishable from a hang at the prefill barrier.
3. Restart only the PC runner (not the master) on each fresh attempt to
   avoid re-running the libp2p handshake — cheaper than a full cluster
   restart and isolates whether it's a per-process vs per-cluster init race.

**Step 7.1: From Mac, send the prompt `what is 2+2`.**

Watch PC daemon log:
```bash
wsl -d Ubuntu -- bash -c 'tail -f /tmp/exo_pc.log'
```
Expected sequence: `Starting prefill` → `KV cache added: <N> tokens` → tokens stream back to Mac.

**Step 7.2: Verify the answer is sensible.**

The model should respond with "4" or a short equivalent. **If output is gibberish, our cache slicing or `previous_kvs` rebasing has a bug.** Capture the dump and diagnose.

**Step 7.3: Capture nvidia-smi dmon during decode (60 s window).**

```bash
wsl -d Ubuntu -- bash -c '> /tmp/nvidia_dmon_pipeline.log; nvidia-smi dmon -s pucm -c 60 -d 1 > /tmp/nvidia_dmon_pipeline.log 2>&1' &
# (use Bash tool with run_in_background=true so dmon survives)
```
Then send a longer prompt from Mac (~50 tokens) so decode runs for the full window.

Expected: SM% noticeably higher than the 7 % baseline measured in tensor-parallel mode; longer steady-state intervals between collective stalls (collectives reduced from ~120/token to ~1/token).

**Step 7.4: Save the trace.**

```bash
wsl -d Ubuntu -- bash -c 'TS=$(date +%H%M%S); cp /tmp/nvidia_dmon_pipeline.log /home/mechramc/exo_dmon_PIPELINE_DECODE_${TS}.log; cp /tmp/exo_pc.log /home/mechramc/exo_daemon_PIPELINE_DECODE_${TS}.log; ls -la /home/mechramc/exo_*PIPELINE*.log'
```

**Step 7.5: Commit a note marker.**

```bash
wsl -d Ubuntu -- bash -c 'cd /home/mechramc/crossfire/exo && git commit --allow-empty -m "checkpoint: two-node pipeline inference produces correct output"'
```

---

## Task 8: Side-by-side dmon capture (Tensor vs Pipeline) for the writeup

**Goal:** Money chart — same prompt, same cluster, same session, two runs, one in each mode, with SM% trace and tok/s.

**Step 8.1: Pick a deterministic prompt.**

Use a prompt with a fixed `max_output_tokens=64` and `seed=1` to make tok/s comparable.

**Step 8.2: Run Tensor pass.**

a. Set `sharding=Tensor` in the chat request from Mac.
b. Start dmon: `nvidia-smi dmon -s pucm -c 180 -d 1 > /tmp/dmon_tensor.log` (background).
c. Send prompt.
d. Record wall-clock from `Starting prefill` to last token.

**Step 8.3: Run Pipeline pass.**

Same prompt, same `max_output_tokens=64`, `sharding=Pipeline`.
Capture to `/tmp/dmon_pipeline.log`.

**Step 8.4: Compute summary stats.**

```bash
wsl -d Ubuntu -- bash -c '
for mode in tensor pipeline; do
    echo "=== $mode ==="
    # mean, p50, p95 of SM% column (col 5)
    awk "NR>2 {print \$5}" /tmp/dmon_${mode}.log | python3 -c "
import sys, statistics
vals = [int(l.strip()) for l in sys.stdin if l.strip()]
print(f\"  samples={len(vals)} mean={statistics.mean(vals):.1f}% median={statistics.median(vals):.0f}% p95={sorted(vals)[int(len(vals)*0.95)]:.0f}% max={max(vals)}%\")"
done
'
```

**Step 8.5: Save numbers + traces under `results/`.**

```bash
mkdir -p C:/Github/Crossfire/results/2026-04-24-pipeline-vs-tensor/
# copy from WSL via UNC
cp \\wsl.localhost\Ubuntu\tmp\dmon_tensor.log    C:/Github/Crossfire/results/2026-04-24-pipeline-vs-tensor/
cp \\wsl.localhost\Ubuntu\tmp\dmon_pipeline.log  C:/Github/Crossfire/results/2026-04-24-pipeline-vs-tensor/
cp \\wsl.localhost\Ubuntu\tmp\exo_pc.log         C:/Github/Crossfire/results/2026-04-24-pipeline-vs-tensor/exo_pc_pipeline.log
```

**Step 8.6: Commit.**

```bash
cd C:/Github/Crossfire
git add results/2026-04-24-pipeline-vs-tensor/
git commit -m "results: side-by-side dmon traces for tensor vs pipeline gemma 4 31b"
```

---

## Task 9: Land the EXO change + writeup hooks

**Step 9.1: Push branch.**

```bash
wsl -d Ubuntu -- bash -c 'cd /home/mechramc/crossfire/exo && git push -u origin feat/gemma4-pipeline-port'
```
**STOP — confirm with user before opening any PR upstream.** This is a behavioral change to upstream EXO; the user may want to land in a fork first.

**Step 9.2: Update CROSSFIRE-X tasks.md / status.md / checkpoint.md.**

Mark the relevant T-06xx task as complete. Add the dmon comparison to the writeup notes.

**Step 9.3: Write up the contribution paragraph in the writeup draft.**

Use the language from the prior session brainstorm:

> *"Gemma 4 31B in EXO tensor-parallel mode across 5090+M4 Max over WiFi: GPU holds an 11.1 GB shard but observed SM utilization pinned at 7 % during decode — matching the predicted 7.2 % from 120 cross-node AllReduce round-trips per token. Pipeline mode reduces this to 1 activation handoff per token (~120× fewer cross-node exchanges), but EXO banned Pipeline placement for Gemma 4 because `gemma4_text.make_cache()` iterated layers globally without per-rank sharding. Contribution: lifted the placement ban with a KV-producer co-location constraint, made `pipeline_auto_parallel` shard-aware for Gemma 4 (slicing `layer_types` and `previous_kvs`, re-basing global indices to local), and patched `make_cache` from EXO without forking mlx-lm. Result: pipeline decode achieves <X> tok/s with <Y> % SM utilization, vs <Z> tok/s and 7 % SM in tensor-parallel mode."*

Fill in <X>, <Y>, <Z> from Task 8.

**Step 9.4: Commit writeup updates.**

```bash
cd C:/Github/Crossfire
git add docs/ tasks.md status.md checkpoint.md
git commit -m "docs: T-06xx complete — gemma 4 pipeline port landed; pipeline vs tensor results"
```

---

## Deferred (post-Path-B)

- **Upstream mlx-lm PR:** open separately. The change required is an additional optional `start_layer`/`end_layer` argument to `Model.make_cache` so the patching can move from EXO into mlx-lm proper. EXO's monkey-patch becomes a thin shim or vanishes.
- **3+ rank Gemma 4 Pipeline:** generalize the constraint to multi-cut splits where the shared tail is constrained to the last rank.
- **General KV-sharing co-location check:** factor out the constraint check into a model-agnostic helper that any future model with KV sharing can opt into.
