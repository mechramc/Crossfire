# CROSSFIRE

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Heterogeneous distributed LLM inference with stacked weight and KV cache compression.**

CROSSFIRE is the first project to combine three cutting-edge optimization techniques simultaneously:

1. **Heterogeneous distributed inference** across NVIDIA GPU and Apple Silicon
2. **TurboQuant+ (TQ4_1S) weight compression** for aggressive model size reduction
3. **turbo3 KV cache compression** for extended context lengths

By splitting inference across an RTX 5090 (prefill) and M4 Max (decode), each chip handles the phase it's architecturally strongest at — achieving better combined performance than either alone.

## Hardware Requirements

| Node | Hardware | Role |
|------|----------|------|
| PC | NVIDIA RTX 5090 (32GB) + 64GB DDR5 | Prefill (compute-bound) |
| Mac | Apple M4 Max (64GB Unified) | Decode (memory-bandwidth-bound) |

**Network:** Thunderbolt 4 (40 Gbps) or 10GbE between nodes.

## Models

| Model | Parameters | Use Case |
|-------|-----------|----------|
| Qwen 3.5 27B | 27B | Primary benchmark target |
| Qwen 2.5 72B | 72B | Stretch goal (requires TQ4_1S to fit) |
| Phi-4 14B | 14B | Small control model |

## Project Structure

```
src/crossfire/           # Core library
  compression/           # TQ4_1S quantization, KV cache compression
  distributed/           # Pipeline-parallel orchestration, networking
  utils/                 # Metric collection and reporting
benchmarks/              # Perplexity, throughput, memory profiling
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

> Full setup instructions for distributed inference coming soon.

## Results

> Benchmark results will be published here as experiments complete.

| Model | Quant | Context | PPL | tok/s | Peak MB | KV Comp | Distributed |
|-------|-------|---------|-----|-------|---------|---------|-------------|
| — | — | — | — | — | — | — | — |

## Key Dependencies

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — inference engine
- [TurboQuant+](https://github.com/TheTom/turboquant_plus) — TQ4_1S weight compression
- [Parallax](https://github.com/nickthecook/parallax) — P2P pipeline-parallel inference

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

## Citation

If you use CROSSFIRE in your research, please cite:

```
@software{crossfire2025,
  title = {CROSSFIRE: Heterogeneous Distributed LLM Inference with Stacked Compression},
  author = {Murai Labs},
  year = {2025},
  url = {https://github.com/murai-labs/crossfire}
}
```
