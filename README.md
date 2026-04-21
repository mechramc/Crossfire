# CROSSFIRE-X

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Heterogeneous distributed LLM inference with ANE compute, Flash-MoE expert streaming, and stacked compression.**

CROSSFIRE-X integrates the Apple Neural Engine (ANE) as a compute target inside an
[EXO](https://github.com/exo-explore/exo)-orchestrated distributed inference pipeline
spanning NVIDIA CUDA and Apple Silicon, with Flash-MoE slot-bank streaming, TriAttention
KV compression, and TurboQuant+ weight compression stacked on top.

### Five Compute Targets, One Pipeline

| Target | Hardware | Role | Power |
|--------|----------|------|-------|
| T1: CUDA GPU | RTX 5090 32GB | Prefill | ~350W |
| T2: Metal GPU | M4 Max 40-core | Primary decode | ~40-60W |
| T3: ANE | M4 Max 16-core ANE | Draft model (speculative decode) | ~2-5W |
| T4: CPU/SME | M4 Max CPU | KV cache mgmt, speculative verification | ~10-15W |
| T5: NVMe SSD | Mac internal SSD | Flash-MoE expert streaming (P6) | ~5W |

**Core thesis:** Every Mac in an EXO cluster leaves ~19 TFLOPS of ANE compute dark, and MoE
models that exceed node memory are completely unservable without expert streaming. CROSSFIRE-X
lights up both, then uses composed compression to make a USB4 interconnect viable enough that
cable speed stops being the limiting story.

**Secondary questions:**
- Does composing TriAttention + TurboQuant reduce cross-node KV transfer enough to make USB4
  effectively invisible for practical inference?
- Can Flash-MoE slot-bank streaming make Orion Forge (fused specialist MoE) servable at
  real-time inference speeds?

### Interconnect

| Link | Speed | Latency | Role |
|------|-------|---------|------|
| USB4 active cable | 40 Gbps (~4-5 GB/s effective) | ~300 us | Primary EXO data path |
| 5GbE Ethernet | 5 Gbps | ~500 us | Discovery, control plane, fallback |

The final build spec assumes USB4 with Thunderbolt IP bridging, not TB5 RDMA. The practical
claim is that composed compression makes the slower, more common consumer interconnect good
enough for the target workload.

## Hardware Requirements

| Node | Hardware | Role |
|------|----------|------|
| PC | NVIDIA RTX 5090 (32GB) + 64GB DDR5 | EXO compute node: prefill (T1) |
| Mac | Apple M4 Max (64GB Unified) + NVMe | EXO compute node: decode (T2) + ANE (T3) + expert streaming (T5) |

**Network:** USB4 active cable as the primary EXO path, with 5GbE as the fallback and control link.

## Models

The target family is **Gemma 4** (Apache 2.0) — dense primary, ANE draft, and
Flash-MoE all draw from the same tokenizer family for coherent speculative
decode pairing.

| Model | Parameters | Use Case |
|-------|-----------|----------|
| Gemma 4 31B | 33B dense | Primary benchmark target |
| Gemma 4 E2B | 5.1B stored / 2.3B effective (PLE) | ANE draft model (speculative decode, T3) |
| Gemma 4 26B-A4B | 25.2B total / 3.8B active, 128 experts | Flash-MoE slot-bank target (P6) |
| Orion Forge | MoE fused specialist | Secondary Flash-MoE + EXO serving target |
| Gemma 4 31B @ 256K ctx | Same weights, long-context | Stretch / impossible-scenario (distributed + TriAttention) |

## Execution Policies (P0-P6)

AutoPilot selects the appropriate policy at runtime using a deterministic decision tree
(default) or a configurable UCB1/Thompson bandit engine.

| Policy | Pipeline | ANE | KV Strategy | Weights | Use Case |
|--------|----------|-----|-------------|---------|----------|
| P0 | Single best node | Idle | None | FP16/Q8 | Fallback, no distributed overhead |
| P1 | EXO over USB4 (5090 prefill + Mac decode) | Idle | None | FP16/Q8 | Distributed baseline |
| P2 | EXO over USB4 | Draft E2B | None | FP16/Q8 | Speculative decode via ANE |
| P3 | EXO over USB4 | Idle | None | TQ4_1S | Reduce cross-node transfer cost |
| P4 | EXO over USB4 | Idle | TriAttention | TQ4_1S | Long-context, compressed KV path |
| P5 | EXO over USB4 | Draft E2B | TriAttention | TQ4_1S | Full-stack stacked compression |
| P6 | EXO over USB4 | Idle | TriAttention | Flash-MoE | MoE models exceeding node memory |

## Project Structure

```text
src/crossfire/           # Core library
  ane/                   # ANE compute target (ANEMLL, Rustane integration)
  autopilot/             # AutoPilot policy engine (decision tree + bandits)
  compression/           # TQ4_1S quantization, TriAttention + turbo3 KV compression
  distributed/           # EXO orchestration and cross-node networking
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
ruff format --check .
```

> The final build spec assumes EXO over USB4 plus a 5GbE fallback. Setup scripts and code
> scaffolds are aligned with that spec; see `status.md` for remaining bring-up work.

## Ablation Matrix (C0-C7)

Controlled offline benchmarking configs that isolate each component's contribution:

| Config | Pipeline | llama Runtime | ANE | KV Compression | Weights |
|--------|----------|---------------|-----|----------------|---------|
| C0 | Single-node | stock | off | none | Q8_0 |
| C1 | EXO (T1+T2) | stock | off | none | Q8_0 |
| C2 | EXO | stock | T3 draft | none | Q8_0 |
| C3 | EXO | stock | off | TriAttention | Q8_0 |
| C4 | EXO | stock | off | TriAttention | TQ4_1S |
| C5 | EXO | stock | T3 draft | TriAttention | TQ4_1S |
| C6 | EXO | Flash-MoE slot-bank | off | TriAttention | Flash-MoE |
| C7 | Single-node | Flash-MoE slot-bank | off | none | Flash-MoE |

## Experiment Tiers

| Tier | Focus |
|------|-------|
| Tier 0 (Day 1-2) | EXO baseline over USB4 + Thunderbolt IP bridge |
| Tier 1 (Day 3-4) | Flash-MoE integration and slot-bank validation |
| Tier 2 (Day 5-9) | ANE integration + composed TriAttention/TurboQuant compression |
| Tier 3 (Day 10-12) | AutoPilot routing, USB4-aware policy selection, Orion Forge serving |
| Write-up (Day 13-15) | Analysis, charts, blog post, community submissions |

## Results

> Benchmark results will be published here as experiments complete.

| Policy | Model | Quant | PPL | tok/s | TTFT (ms) | tok/W | MoE Hit% | Power (W) |
|--------|-------|-------|-----|-------|-----------|-------|----------|-----------|
| P0 | gemma-4-31b | Q8_0 | -- | -- | -- | -- | -- | -- |
| P1 | gemma-4-31b | Q8_0 | -- | -- | -- | -- | -- | -- |
| P2 | gemma-4-31b | Q8_0 | -- | -- | -- | -- | -- | -- |
| P5 | gemma-4-31b | TQ4_1S | -- | -- | -- | -- | -- | -- |
| P6 | gemma-4-26b-a4b | Flash-MoE | -- | -- | -- | -- | -- | -- |

## Key Dependencies

- [EXO](https://github.com/exo-explore/exo) -- distributed inference over a consumer interconnect
- [anemll-flash-llama.cpp](https://github.com/Anemll/anemll-flash-llama.cpp) -- Flash-MoE slot-bank expert streaming (T5)
- [ANEMLL](https://github.com/Anemll/Anemll) -- ANE inference pipeline (T3)
- [Rustane](https://github.com/ncdrone/rustane) -- Rust-native ANE training/inference
- [TurboQuant+](https://github.com/TheTom/turboquant_plus) -- TQ4_1S weight compression
- [TriAttention](https://arxiv.org/abs/2604.04921) -- Trigonometric pre-RoPE KV scoring (10.7x KV reduction; composes to ~6.8x with TurboQuant on the current thesis)
- [Orion](https://arxiv.org/abs/2603.06728) -- ANE programming system (Murai Labs)

## License

Apache 2.0 -- see [LICENSE](LICENSE) for details.

## Credits

Built on the work of: EXO Labs (Alex Cheema), maderix (Manjeet Singh), ANEMLL team,
Daniel Isaac (Rustane), AtomGradient, SqueezeBits (Yetter), Tom Turney (TurboQuant+),
TriAttention authors and downstream ports, Jeff Geerling, and the ANE research community.

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