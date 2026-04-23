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
lights up both, then uses composed compression to make a consumer interconnect viable enough
that link speed stops being the limiting story. The current cluster path is WiFi; TB4/USB4
remains an optional future optimization if the bandwidth budget demands it.

**Secondary questions:**
- Does composing TriAttention + TurboQuant reduce cross-node KV transfer enough to make WiFi
  practical today and TB4/USB4 effectively invisible if a faster link is added later?
- Can Flash-MoE slot-bank streaming make Orion Forge (fused specialist MoE) servable at
  real-time inference speeds?

### Interconnect

| Link | Speed | Latency | Role |
|------|-------|---------|------|
| WiFi | Environment-dependent | Environment-dependent | Current EXO data path and discovery path |
| TB4/USB4 cable | 40 Gbps (~4-5 GB/s effective) | ~300 us | Optional future optimization |
| 5GbE Ethernet | 5 Gbps | ~500 us | Optional fallback / control path |

The repository currently runs over WiFi rather than TB5 RDMA or Thunderbolt bridging. The
practical claim is that composed compression makes a slower, more common consumer interconnect
good enough for the target workload, with TB4/USB4 reserved for future optimization if needed.

## Hardware Requirements

| Node | Hardware | Role |
|------|----------|------|
| PC | NVIDIA RTX 5090 (32GB) + 64GB DDR5 | EXO compute node: prefill (T1) |
| Mac | Apple M4 Max (64GB Unified) + NVMe | EXO compute node: decode (T2) + ANE (T3) + expert streaming (T5) |

**Network:** WiFi is the current EXO path. TB4/USB4 and 5GbE remain optional future interconnect work.

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
| P1 | EXO over current interconnect (WiFi today) | Idle | None | FP16/Q8 | Distributed baseline |
| P2 | EXO over current interconnect | Draft E2B | None | FP16/Q8 | Speculative decode via ANE |
| P3 | EXO over current interconnect | Idle | None | TQ4_1S | Reduce cross-node transfer cost |
| P4 | EXO over current interconnect | Idle | TriAttention | TQ4_1S | Long-context, compressed KV path |
| P5 | EXO over current interconnect | Draft E2B | TriAttention | TQ4_1S | Full-stack stacked compression |
| P6 | EXO over current interconnect | Idle | TriAttention | Flash-MoE | MoE models exceeding node memory |

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

> The current repo reality is EXO over WiFi. USB4 and 5GbE remain optional future interconnect
> work; see `status.md` for the current bring-up and calibration state.

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
| Tier 0 (Day 1-2) | EXO baseline over the active interconnect |
| Tier 1 (Day 3-4) | Flash-MoE integration and slot-bank validation |
| Tier 2 (Day 5-9) | ANE integration + composed TriAttention/TurboQuant compression |
| Tier 3 (Day 10-12) | AutoPilot routing, interconnect-aware policy selection, Orion Forge serving |
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
