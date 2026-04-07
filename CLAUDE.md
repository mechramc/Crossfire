# CROSSFIRE-X

Heterogeneous distributed LLM inference with ANE compute integration and stacked compression.

## Project Overview

CROSSFIRE-X lights up the Apple Neural Engine (ANE) as a compute target inside an
EXO-orchestrated distributed inference pipeline spanning NVIDIA CUDA and Apple Silicon.

**Core thesis:** Every Mac in an EXO cluster leaves ~19 TFLOPS of dedicated fp16 ANE
compute completely dark. CROSSFIRE-X asks: can we add the ANE as a compute target
inside an EXO pipeline, and does it improve throughput, latency, or power efficiency?

**Secondary question:** Does stacking TurboQuant+ compression (TQ4_1S weights + turbo3
KV cache) on top of the EXO + ANE pipeline compound the gains?

### Five Compute Targets

| Target | Hardware | Managed By | Role | Power |
|--------|----------|------------|------|-------|
| T1: CUDA GPU | RTX 5090 32GB | EXO (CUDA) | Prefill | ~350W |
| T2: Metal GPU | M4 Max 40-core | EXO (MLX) | Primary decode | ~40-60W |
| T3: ANE | M4 Max 16-core ANE | ANEMLL / Rustane | Draft model / speculative decode | ~2-5W |
| T4: CPU/SME | M4 Max CPU | EXO scheduler | KV cache mgmt, speculative verification | ~10-15W |
| T5: RDMA | Thunderbolt 5 link | EXO RDMA (IBV) | KV cache streaming (3us latency) | ~2W |

Primary models: Qwen 3.5 27B (primary), Qwen3.5 0.6B (ANE draft), Qwen 2.5 72B (stretch).

## Tech Stack

- **Distributed orchestration:** EXO 1.0 (RDMA over Thunderbolt 5, topology-aware auto-parallel)
- **ANE inference:** ANEMLL (CoreML path), Rustane (direct API path)
- **Weight compression:** TurboQuant+ (TheTom/turboquant_plus fork) — TQ4_1S format
- **KV cache compression:** llama.cpp turbo3/turbo4 (asymmetric q8_0-K / turbo3-V)
- **Inference engine:** llama.cpp (TurboQuant+ fork for TQ4_1S), MLX (via EXO)
- **Benchmarking:** Python 3.10+, psutil, tabulate
- **Config:** YAML-based model and hardware definitions
- **Linting:** ruff
- **Testing:** pytest

## Directory Structure

```
src/crossfire/           # Core library
  ane/                   # ANE compute target (ANEMLL/Rustane integration)
  compression/           # TQ4_1S quantization + KV cache compression
  distributed/           # EXO orchestration + RDMA networking
  utils/                 # Metric collection and reporting
benchmarks/              # Perplexity, throughput, memory, power profiling
configs/                 # Model and hardware YAML definitions
scripts/                 # Setup and experiment runner shell scripts
results/                 # Benchmark outputs (raw/ is gitignored)
tests/                   # pytest test suite
```

## Commands

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .
ruff format --check .

# Run a benchmark (placeholder — will be implemented)
python -m benchmarks.perplexity --model models/qwen3.5-27b-tq4_1s.gguf
```

## Code Conventions

- **Python 3.10+** — use `from __future__ import annotations` in every module
- **Type hints** on all public function signatures
- **Dataclasses** (`frozen=True`) for configuration objects
- **ruff** for linting and formatting (line length 100)
- **No `any` types** — be explicit
- **Constants** as module-level `UPPER_SNAKE_CASE`
- **Errors:** raise specific exceptions with descriptive messages, not bare `Exception`
- **Shell scripts:** bash, set `-euo pipefail`, quote all variables

## Do's

- Use `pathlib.Path` for all file path handling
- Use `subprocess.run(..., check=True)` for external commands
- Keep benchmark scripts stateless — config in, results out
- Store processed results in `results/`, raw data in `results/raw/` (gitignored)
- Use YAML configs for anything that varies between runs
- Measure power per-target (use `powermetrics` on Mac, `nvidia-smi` on PC)

## Don'ts

- Don't commit model files (*.gguf, *.bin, *.safetensors, *.mlmodelc) — they're multi-GB
- Don't commit spec docs, tasks.md, checkpoint.md — internal only
- Don't hardcode paths to models or binaries — use configs
- Don't add dependencies without justification
- Don't use `os.path` — use `pathlib`
- Don't use CoreML `compute_units=ALL` — macOS 26.3 routes to GPU, not ANE. Use direct API.

## Hardware Context

| Node | Hardware | Role | Key Specs |
|------|----------|------|-----------|
| PC | RTX 5090 32GB + 64GB DDR5 | Prefill (T1) | ~200 TFLOPS FP16, ~1,792 GB/s |
| Mac | M4 Max 64GB Unified | Decode (T2) + ANE (T3) | 546 GB/s, 40-core GPU, 16-core ANE (~19 TFLOPS) |

Network: Thunderbolt 5 (80 Gbps bidirectional, RDMA-capable, 3us latency via EXO).

Key config: `sudo sysctl iogpu.wired_limit_mb=58982` on Mac (unlocks ~90% of 64GB for Metal).
RDMA enable via Recovery mode: `rdma_ctl enable`.

## Known Constraints

- EXO 1.0 demos use DGX Spark (ARM + Blackwell), not standard Linux PC with RTX 5090
- TQ4_1S requires TheTom/turboquant_plus fork, not mainline llama.cpp
- ANE has 32 MB SRAM cliff (30% throughput drop beyond it)
- ANE dimension efficiency cliff at dim=5120 (4.7x penalty — keep <=4096)
- ANEMLL max model size ~8B, context caps 2048-4096
- macOS 26.3 routes CoreML `compute_units=ALL` to GPU, not ANE — use direct private API
- KV cache FP16 ANE -> GGML/MLX bridge costs ~11-16% decode degradation
- Blackwell tensor cores have structural mismatch with turbo dequant (25-38% penalty)
- Model files must be downloaded separately (not in repo)

## Experiment Tiers

- **Tier 0 (Day 1-2):** EXO baseline — confirm distributed inference over TB5 RDMA
- **Tier 1 (Day 3-6):** ANE integration — zero-interference, speculative draft, prefill offload
- **Tier 2 (Day 7-9):** TurboQuant+ compression stacked on EXO + ANE pipeline
- **Day 10:** Analysis, charts, write-up

## Key References

- EXO Labs: github.com/exo-explore/exo
- Orion (Murai Labs): arXiv:2603.06728
- ANEMLL: github.com/Anemll/Anemll
- Rustane: github.com/ncdrone/rustane
- AtomGradient: github.com/AtomGradient/hybrid-ane-mlx-bench
- TurboQuant+: github.com/TheTom/turboquant_plus
