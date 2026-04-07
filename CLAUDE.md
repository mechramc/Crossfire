# CROSSFIRE

Heterogeneous distributed LLM inference with stacked weight and KV cache compression.

## Project Overview

CROSSFIRE combines three optimization techniques that have never been stacked together:

1. **Heterogeneous distributed inference** — RTX 5090 (prefill) + M4 Max (decode)
2. **TQ4_1S weight compression** — via llama-cpp-turboquant
3. **turbo3 KV cache compression** — via llama.cpp

Primary models: Qwen 3.5 27B (primary), Qwen 2.5 72B (stretch goal).

## Tech Stack

- **Inference engine:** llama.cpp (TheTom/turboquant_plus fork for TQ4_1S)
- **Distributed frameworks:** Parallax (primary), EXO 1.0 (secondary), prima.cpp (fallback)
- **Benchmarking:** Python 3.10+, psutil, tabulate
- **Config:** YAML-based model and hardware definitions
- **Linting:** ruff
- **Testing:** pytest

## Directory Structure

```
src/crossfire/           # Core library
  compression/           # TQ4_1S quantization + KV cache compression
  distributed/           # Pipeline-parallel orchestration + networking
  utils/                 # Metric collection and reporting
benchmarks/              # Perplexity, throughput, memory profiling scripts
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

## Don'ts

- Don't commit model files (*.gguf, *.bin, *.safetensors) — they're multi-GB
- Don't commit spec docs, tasks.md, checkpoint.md — internal only
- Don't hardcode paths to models or binaries — use configs
- Don't add dependencies without justification
- Don't use `os.path` — use `pathlib`

## Hardware Context

| Node | Hardware | Role | Key Specs |
|------|----------|------|-----------|
| PC | RTX 5090 32GB + 64GB DDR5 | Prefill | 209 TFLOPS FP16, 1.8 TB/s |
| Mac | M4 Max 64GB Unified | Decode | 546 GB/s bandwidth |

Network: Thunderbolt 4 (40 Gbps) or 10GbE.

## Known Constraints

- TQ4_1S requires the TheTom/turboquant_plus fork (PR #45), not mainline llama.cpp
- turbo3 KV cache is available in recent llama.cpp builds (check `--cache-type-k` flag)
- Parallax requires compatible llama.cpp builds on both nodes
- Model files must be downloaded separately (not in repo)
