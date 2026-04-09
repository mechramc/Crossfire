# CROSSFIRE-X

Heterogeneous distributed LLM inference with ANE compute integration, Flash-MoE expert streaming, and stacked compression.

## Project Overview

CROSSFIRE-X integrates the Apple Neural Engine (ANE) as a compute target inside an
EXO-orchestrated distributed inference pipeline spanning NVIDIA CUDA and Apple Silicon,
with Flash-MoE slot-bank expert streaming for MoE models that exceed node memory.

**Core thesis:** Every Mac in an EXO cluster leaves ~19 TFLOPS of dedicated fp16 ANE
compute completely dark, and MoE models that exceed node memory are completely unservable
without expert streaming. CROSSFIRE-X lights up both.

**Secondary questions:**
- Does stacking TriAttention KV compression (10.7x reduction) + TQ4_1S weight compression
  on top of EXO + ANE compound the throughput/efficiency gains?
- Can Flash-MoE slot-bank streaming make Orion Forge (fused specialist MoE) servable at
  real-time inference speeds on the T6 NVMe target?

### Six Compute Targets

| Target | Hardware | Managed By | Role | Power |
|--------|----------|------------|------|-------|
| T1: CUDA GPU | RTX 5090 32GB | EXO (CUDA) | Prefill | ~350W |
| T2: Metal GPU | M4 Max 40-core | EXO (MLX) | Primary decode | ~40-60W |
| T3: ANE | M4 Max 16-core ANE | ANEMLL / Rustane | Draft model / speculative decode | ~2-5W |
| T4: CPU/SME | M4 Max CPU | EXO scheduler | KV cache mgmt, speculative verification | ~10-15W |
| T5: RDMA | Thunderbolt 5 link | EXO RDMA (IBV) | KV cache streaming (3us latency) | ~2W |
| T6: NVMe SSD | Mac internal SSD | anemll-flash-llama.cpp | Flash-MoE expert streaming (P6) | ~5W |

Primary models: Qwen 3.5 27B (dense, primary), Qwen3.5-35B-A3B (MoE, P6 target),
Qwen3.5 0.6B (ANE draft), Orion Forge (fused specialist MoE), Qwen 2.5 72B (stretch).

## Tech Stack

- **Distributed orchestration:** EXO 1.0 (RDMA over Thunderbolt 5, topology-aware auto-parallel)
- **ANE inference:** ANEMLL (CoreML path), Rustane (direct API path)
- **Flash-MoE expert streaming:** anemll-flash-llama.cpp (slot-bank runtime, pread from NVMe)
- **Weight compression:** TurboQuant+ (TheTom/turboquant_plus fork) -- TQ4_1S format
- **KV cache compression:** TriAttention (arXiv:2604.04921, 10.7x KV reduction, primary strategy)
  and llama.cpp turbo3 (legacy fallback)
- **Inference engine:** anemll-flash-llama.cpp (primary for P6), llama.cpp TurboQuant+ fork (P0-P5)
- **AutoPilot:** Deterministic decision tree (default) or configurable UCB1/Thompson bandit
- **Benchmarking:** Python 3.10+, psutil, tabulate
- **Config:** YAML-based model and hardware definitions
- **Linting:** ruff
- **Testing:** pytest

## Directory Structure

```
src/crossfire/           # Core library
  ane/                   # ANE compute target (ANEMLL/Rustane integration)
  autopilot/             # AutoPilot policy engine (decision tree + bandits)
  compression/           # TQ4_1S quantization + TriAttention/turbo3 KV compression
  distributed/           # EXO orchestration + RDMA networking
  flashmoe/              # Flash-MoE slot-bank runtime (anemll-flash-llama.cpp)
  utils/                 # Metric collection and reporting
benchmarks/              # Perplexity, throughput, memory, power profiling
configs/                 # Model and hardware YAML definitions
docs/archive/            # Superseded spec documents
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

# Run a benchmark (placeholder -- will be implemented)
python -m benchmarks.perplexity --model models/qwen3.5-27b-tq4_1s.gguf
```

## Code Conventions

- **Python 3.10+** -- use `from __future__ import annotations` in every module
- **Type hints** on all public function signatures
- **Dataclasses** (`frozen=True`) for configuration objects
- **ruff** for linting and formatting (line length 100)
- **No `any` types** -- be explicit
- **Constants** as module-level `UPPER_SNAKE_CASE`
- **Errors:** raise specific exceptions with descriptive messages, not bare `Exception`
- **Shell scripts:** bash, set `-euo pipefail`, quote all variables
- **NotImplementedError stubs**: hardware-gated paths use `raise NotImplementedError` with
  a clear message stating what bring-up is required before implementation

## Do's

- Use `pathlib.Path` for all file path handling
- Use `subprocess.run(..., check=True)` for external commands
- Keep benchmark scripts stateless -- config in, results out
- Store processed results in `results/`, raw data in `results/raw/` (gitignored)
- Use YAML configs for anything that varies between runs
- Measure power per-target (use `powermetrics` on Mac, `nvidia-smi` on PC)
- Gate Flash-MoE paths behind `flash_moe_available` in `HardwareAvailability` --
  P6 must not be selectable until anemll-flash-llama.cpp is built and verified

## Don'ts

- Don't commit model files (*.gguf, *.bin, *.safetensors, *.mlmodelc) -- they're multi-GB
- Don't commit spec docs, tasks.md, checkpoint.md -- internal only
- Don't hardcode paths to models or binaries -- use configs
- Don't add dependencies without justification
- Don't use `os.path` -- use `pathlib`
- Don't use CoreML `compute_units=ALL` -- macOS 26.3 routes to GPU, not ANE. Use direct API.
- Don't use turbo3/turbo4 KV as the primary KV strategy -- TriAttention is the new primary.
  turbo3 remains in `kvcache.py` as a legacy fallback only.

## Hardware Context

| Node | Hardware | Role | Key Specs |
|------|----------|------|-----------|
| PC | RTX 5090 32GB + 64GB DDR5 | Prefill (T1) | ~200 TFLOPS FP16, ~1,792 GB/s |
| Mac | M4 Max 64GB Unified + NVMe | Decode (T2) + ANE (T3) + T6 | 546 GB/s, 40-core GPU, 16-core ANE (~19 TFLOPS), ~4.7 GB/s NVMe |

Network: Thunderbolt 5 (80 Gbps bidirectional, RDMA-capable, 3us latency via EXO).

Key config: `sudo sysctl iogpu.wired_limit_mb=58982` on Mac (unlocks ~90% of 64GB for Metal).
RDMA enable via Recovery mode: `rdma_ctl enable`.

## Known Constraints

- EXO 1.0 demos use DGX Spark (ARM + Blackwell), not standard Linux PC with RTX 5090
- TQ4_1S requires TheTom/turboquant_plus fork, not mainline llama.cpp
- ANE has 32 MB SRAM cliff (30% throughput drop beyond it)
- ANE dimension efficiency cliff at dim=5120 (4.7x penalty -- keep <=4096)
- ANEMLL max model size ~8B, context caps 2048-4096
- macOS 26.3 routes CoreML `compute_units=ALL` to GPU, not ANE -- use direct private API
- KV cache FP16 ANE -> GGML/MLX bridge costs ~11-16% decode degradation
- Blackwell tensor cores have structural mismatch with turbo dequant (25-38% penalty)
- Flash-MoE slot-bank requires anemll-flash-llama.cpp built with LLAMA_FLASH_MOE_GPU_BANK=ON;
  mainline llama.cpp does not support slot-bank mode
- TriAttention requires calibration artifacts per model; not yet publicly released --
  implementation stubs are present, paper author collaboration pending
- Orion Forge model artifacts require separate acquisition (KALAVAI training pipeline)
- Model files must be downloaded separately (not in repo)
- T6 effective bandwidth ~4.7 GB/s (NVMe pread); slot-bank sizing must keep hot experts
  within 5-15% of node RAM to maintain acceptable hit rate

## Experiment Tiers

- **Tier 0 (Day 1-2):** EXO baseline -- confirm distributed inference over TB5 RDMA
- **Tier 1 (Day 3-6):** ANE integration -- zero-interference, speculative draft, prefill offload
- **Tier 2 (Day 7-9):** TurboQuant+ + TriAttention compression stacked on EXO + ANE pipeline
- **Tier 3 (Day 10-12):** Flash-MoE slot-bank bring-up + Orion Forge serving (P6)
- **Write-up (Day 13-14):** Analysis, charts, blog post, community submissions

## Key References

- EXO Labs: github.com/exo-explore/exo
- anemll-flash-llama.cpp: github.com/Anemll/anemll-flash-llama.cpp
- Orion (Murai Labs): arXiv:2603.06728
- ANEMLL: github.com/Anemll/Anemll
- Rustane: github.com/ncdrone/rustane
- AtomGradient: github.com/AtomGradient/hybrid-ane-mlx-bench
- TurboQuant+: github.com/TheTom/turboquant_plus
- TriAttention: arXiv:2604.04921
