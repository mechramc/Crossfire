# CROSSFIRE-X

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Heterogeneous distributed LLM inference with ANE compute, Flash-MoE expert streaming, and stacked compression.**

CROSSFIRE-X integrates the Apple Neural Engine (ANE) as a compute target inside an
[EXO](https://github.com/exo-explore/exo)-orchestrated distributed inference pipeline
spanning NVIDIA CUDA and Apple Silicon — with Flash-MoE slot-bank streaming, TriAttention
KV compression, and TurboQuant+ weight compression stacked on top.

### Six Compute Targets, One Pipeline

| Target | Hardware | Role | Power |
|--------|----------|------|-------|
| T1: CUDA GPU | RTX 5090 32GB | Prefill | ~350W |
| T2: Metal GPU | M4 Max 40-core | Primary decode | ~40-60W |
| T3: ANE | M4 Max 16-core ANE | Draft model (speculative decode) | ~2-5W |
| T4: CPU/SME | M4 Max CPU | KV cache mgmt, speculative verification | ~10-15W |
| T5: RDMA | Thunderbolt 5 | KV cache streaming (3us latency) | ~2W |
| T6: NVMe SSD | Mac internal SSD | Flash-MoE expert streaming (P6) | ~5W |

**Core thesis:** Every Mac in an EXO cluster leaves ~19 TFLOPS of ANE compute dark, and MoE
models that exceed node memory are completely unservable without expert streaming. CROSSFIRE-X
lights up both — and measures whether the combination yields better throughput, latency, and
power efficiency than any single technique alone.

**Secondary question:** Does stacking TriAttention KV compression + TQ4_1S weight compression
on top of the EXO + ANE pipeline compound the gains? And can Flash-MoE slot-bank streaming
make Orion Forge (fused specialist MoE) servable at real-time inference speeds?

## Hardware Requirements

| Node | Hardware | Role |
|------|----------|------|
| PC | NVIDIA RTX 5090 (32GB) + 64GB DDR5 | EXO compute node: prefill (T1) |
| Mac | Apple M4 Max (64GB Unified) + NVMe | EXO compute node: decode (T2) + ANE (T3) + expert streaming (T6) |

**Network:** Thunderbolt 5 (80 Gbps bidirectional, RDMA-capable, 3us latency).

## Models

| Model | Parameters | Use Case |
|-------|-----------|----------|
| Qwen 3.5 27B | 27B dense | Primary benchmark target |
| Qwen 3.5 0.6B | 0.6B dense | ANE draft model (speculative decode, T3) |
| Qwen 3.5 35B-A3B | 35B MoE (3B active) | Flash-MoE slot-bank target (P6) |
| Orion Forge | MoE fused specialist | Primary Flash-MoE + EXO serving target |
| Qwen 2.5 72B | 72B dense | Stretch goal (distributed, TQ4_1S) |

## Execution Policies (P0-P6)

AutoPilot selects the appropriate policy at runtime using a deterministic decision tree
(default) or a configurable UCB1/Thompson bandit engine.

| Policy | Pipeline | ANE | KV Strategy | Weights | Use Case |
|--------|----------|-----|-------------|---------|----------|
| P0 | Single best node | Idle | None | FP16/Q8 | Fallback, no distributed overhead |
| P1 | EXO (5090 prefill + Mac decode) | Idle | None | FP16/Q8 | Distributed baseline |
| P2 | EXO | Draft 0.6B | None | FP16/Q8 | Speculative decode via ANE |
| P3 | EXO | Idle | None | TQ4_1S | Compressed weights |
| P4 | EXO | Idle | TriAttention | TQ4_1S | Long-context, max KV compression |
| P5 | EXO | Draft 0.6B | TriAttention | TQ4_1S | Full-stack stacked compression |
| P6 | EXO | Idle | TriAttention | Flash-MoE | MoE models exceeding node memory |

## Project Structure

```
src/crossfire/           # Core library
  ane/                   # ANE compute target (ANEMLL, Rustane integration)
  autopilot/             # AutoPilot policy engine (decision tree + bandits)
  compression/           # TQ4_1S quantization, TriAttention + turbo3 KV compression
  distributed/           # EXO orchestration, RDMA networking
  flashmoe/              # Flash-MoE slot-bank runtime (anemll-flash-llama.cpp)
  utils/                 # Metric collection, power profiling, reporting
benchmarks/              # Perplexity, throughput, memory, power profiling
configs/                 # Model and hardware YAML definitions
docs/archive/            # Superseded spec documents
scripts/                 # Setup and experiment runner scripts
results/                 # Benchmark outputs
tests/                   # Test suite
```

## Quick Start

```bash
# Clone
git clone https://github.com/murai-labs/crossfire.git
cd crossfire

# Install
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .
```

> Full hardware setup (EXO + RDMA + Flash-MoE build) documented in `scripts/setup_mac.sh`
> and `scripts/setup_pc.sh`.

## Ablation Matrix (C0-C7)

Controlled offline benchmarking configs that isolate each component's contribution:

| Config | Pipeline | llama Runtime | ANE | KV Compression | Weights |
|--------|----------|---------------|-----|----------------|---------|
| C0 | Single-node | stock | off | none | Q8_0 |
| C1 | EXO (T1+T2) | stock | off | none | Q8_0 |
| C2 | EXO | stock | T3 draft | none | Q8_0 |
| C3 | EXO | stock | off | TriAttention | Q8_0 |
| C4 | EXO | stock | off | none | TQ4_1S |
| C5 | EXO | stock | T3 draft | TriAttention | TQ4_1S |
| C6 | EXO | Flash-MoE slot-bank | off | TriAttention | Flash-MoE |
| C7 | Single-node | Flash-MoE slot-bank | off | none | Flash-MoE |

## Experiment Tiers

| Tier | Focus |
|------|-------|
| Tier 0 (Day 1-2) | EXO baseline -- confirm distributed inference over TB5 RDMA |
| Tier 1 (Day 3-6) | ANE integration -- zero-interference, speculative draft, prefill offload |
| Tier 2 (Day 7-9) | TurboQuant+ + TriAttention stacked on EXO + ANE pipeline |
| Tier 3 (Day 10-12) | Flash-MoE slot-bank bring-up + Orion Forge serving (P6) |
| Write-up (Day 13-14) | Analysis, charts, blog post, community submissions |

## Results

> Benchmark results will be published here as experiments complete.

| Policy | Model | Quant | PPL | tok/s | TTFT (ms) | tok/W | MoE Hit% | Power (W) |
|--------|-------|-------|-----|-------|-----------|-------|----------|-----------|
| P0 | qwen3.5-27b | Q8_0 | -- | -- | -- | -- | -- | -- |
| P1 | qwen3.5-27b | Q8_0 | -- | -- | -- | -- | -- | -- |
| P2 | qwen3.5-27b | Q8_0 | -- | -- | -- | -- | -- | -- |
| P5 | qwen3.5-27b | TQ4_1S | -- | -- | -- | -- | -- | -- |
| P6 | qwen3.5-35b-a3b | Flash-MoE | -- | -- | -- | -- | -- | -- |

## Key Dependencies

- [EXO](https://github.com/exo-explore/exo) -- distributed inference with RDMA over TB5
- [anemll-flash-llama.cpp](https://github.com/Anemll/anemll-flash-llama.cpp) -- Flash-MoE slot-bank expert streaming (T6)
- [ANEMLL](https://github.com/Anemll/Anemll) -- ANE inference pipeline (T3)
- [Rustane](https://github.com/ncdrone/rustane) -- Rust-native ANE training/inference
- [TurboQuant+](https://github.com/TheTom/turboquant_plus) -- TQ4_1S weight compression
- [TriAttention](https://arxiv.org/abs/2604.04921) -- Trigonometric pre-RoPE KV scoring (10.7x KV reduction)
- [Orion](https://arxiv.org/abs/2603.06728) -- ANE programming system (Murai Labs)

## License

Apache 2.0 -- see [LICENSE](LICENSE) for details.

## Credits

Built on the work of: EXO Labs (Alex Cheema), maderix (Manjeet Singh), ANEMLL team,
Daniel Isaac (Rustane), AtomGradient, SqueezeBits (Yetter), Tom Turney (TurboQuant+),
TriAttention authors (arXiv:2604.04921), Jeff Geerling (RDMA benchmarks), and the
ANE research community.

## Citation

If you use CROSSFIRE-X in your research, please cite:

```bibtex
@software{crossfire2026,
  title = {CROSSFIRE-X: Heterogeneous Distributed LLM Inference with ANE, Flash-MoE, and Stacked Compression},
  author = {Murai Labs},
  year = {2026},
  url = {https://github.com/murai-labs/crossfire}
}
```
