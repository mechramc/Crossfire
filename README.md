# CROSSFIRE v2

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Lighting up the Neural Engine in a distributed inference pipeline.**

CROSSFIRE v2 integrates the Apple Neural Engine (ANE) as a compute target inside an
[EXO](https://github.com/exo-explore/exo)-orchestrated distributed inference pipeline
spanning NVIDIA CUDA and Apple Silicon — with TurboQuant+ compression stacked on top.

### Five Compute Targets, One Pipeline

| Target | Hardware | Role | Power |
|--------|----------|------|-------|
| T1: CUDA GPU | RTX 5090 32GB | Prefill | ~350W |
| T2: Metal GPU | M4 Max 40-core | Primary decode | ~40-60W |
| T3: ANE | M4 Max 16-core ANE | Draft model (speculative decode) | ~2-5W |
| T4: CPU/SME | M4 Max CPU | KV cache mgmt, speculative verification | ~10-15W |
| T5: RDMA | Thunderbolt 5 | KV cache streaming (3us latency) | ~2W |

**Core thesis:** Every Mac in an EXO cluster leaves ~19 TFLOPS of ANE compute dark.
CROSSFIRE v2 lights it up — and measures whether it helps.

## Hardware Requirements

| Node | Hardware | Role |
|------|----------|------|
| PC | NVIDIA RTX 5090 (32GB) + 64GB DDR5 | EXO compute node: prefill (T1) |
| Mac | Apple M4 Max (64GB Unified) | EXO compute node: decode (T2) + ANE target (T3) |

**Network:** Thunderbolt 5 (80 Gbps bidirectional, RDMA-capable).

## Models

| Model | Parameters | Use Case |
|-------|-----------|----------|
| Qwen 3.5 27B | 27B | Primary benchmark target |
| Qwen 3.5 0.6B | 0.6B | ANE draft model (speculative decode) |
| Qwen 2.5 72B | 72B | Stretch goal (distributed, TQ4_1S compressed) |

## Project Structure

```
src/crossfire/           # Core library
  ane/                   # ANE compute target (ANEMLL, Rustane integration)
  compression/           # TQ4_1S quantization, KV cache compression
  distributed/           # EXO orchestration, RDMA networking
  utils/                 # Metric collection, power profiling, reporting
benchmarks/              # Perplexity, throughput, memory, power profiling
configs/                 # Model and hardware YAML definitions
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

> Full setup instructions for EXO + ANE distributed inference coming soon.

## Experiment Tiers

| Tier | Days | Focus |
|------|------|-------|
| T0: EXO Baseline | 1-2 | Confirm EXO distributed inference over TB5 RDMA |
| T1: ANE Integration | 3-6 | Zero-interference, speculative draft, prefill offload |
| T2: TurboQuant+ | 7-9 | Stack TQ4_1S + turbo3 compression on EXO + ANE |
| Write-up | 10 | Analysis, blog post, community submissions |

## Results

> Benchmark results will be published here as experiments complete.

| Config | EXO Pipeline | ANE | TQ+ | tok/s | Power (W) |
|--------|-------------|-----|-----|-------|-----------|
| C0: EXO baseline | 5090 + Mac GPU | Idle | None | — | — |
| C1: ANE speculative | 5090 + Mac GPU | Draft 0.6B | None | — | — |
| C5: Full stack | 5090 + Mac GPU | Draft 0.6B | TQ4_1S + turbo3 | — | — |
| C6: 72B stretch | 5090 + Mac GPU (72B) | Draft 0.6B | TQ4_1S + turbo3 | — | — |

## Key Dependencies

- [EXO](https://github.com/exo-explore/exo) — distributed inference with RDMA over TB5
- [ANEMLL](https://github.com/Anemll/Anemll) — ANE inference pipeline
- [Rustane](https://github.com/ncdrone/rustane) — Rust-native ANE training/inference
- [TurboQuant+](https://github.com/TheTom/turboquant_plus) — TQ4_1S weight compression
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — inference engine
- [Orion](https://arxiv.org/abs/2603.06728) — ANE programming system (Murai Labs)

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

## Credits

Built on the work of: EXO Labs (Alex Cheema), maderix (Manjeet Singh), ANEMLL team,
Daniel Isaac (Rustane), AtomGradient, SqueezeBits (Yetter), Tom Turney (TurboQuant+),
Jeff Geerling (RDMA benchmarks), and the ANE research community.

## Citation

If you use CROSSFIRE in your research, please cite:

```bibtex
@software{crossfire2026,
  title = {CROSSFIRE v2: Lighting Up the Neural Engine in a Distributed Inference Pipeline},
  author = {Murai Labs},
  year = {2026},
  url = {https://github.com/murai-labs/crossfire}
}
```
