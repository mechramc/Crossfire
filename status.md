# CROSSFIRE-X Status

Last updated: 2026-04-09
Branch: main
Latest commit: 9558acf (`feat: add autopilot primitives and orchestrator`)
Tracker state: reconciled against repository contents after unified spec migration (Session 11)

## Summary

The repository has been migrated to the unified spec. All spec documents have been
archived under `docs/archive/`; `crossfire_x_unified.docx` is the sole canonical spec.
Core library modules now reflect the six compute targets (T1-T6), seven policies (P0-P6),
Flash-MoE slot-bank runtime, TriAttention KV compression, configurable AutoPilot engine
(decision tree + bandit), and C0-C7 ablation matrix. The project is not yet in a runnable
experimental state: benchmark execution paths are stubs and hardware bring-up has not started.

## Completed In Repo

### Infrastructure
- Project scaffold: `src/`, `tests/`, `configs/`, `scripts/`, `benchmarks/`, `results/`
- `pyproject.toml`, `.gitignore`, `LICENSE`, `CLAUDE.md`, `AGENTS.md`
- `docs/archive/` created; all superseded specs moved there

### Core Modules
- ANE module: `draft_model.py`, `power.py`, `speculative.py`
- Distributed module: `pipeline.py` (T1-T6 targets, execution_policy, flash_moe_enabled), `network.py`
- Compression module: `turboquant.py`, `kvcache.py`, `triattention.py` (scaffolded stubs)
- Flash-MoE module: `flashmoe/__init__.py`, `flashmoe/config.py`, `flashmoe/runtime.py`
- Metrics: `utils/metrics.py` (policy-native schema, 14-column table, P6/Flash-MoE fields)

### AutoPilot
- `autopilot/query_classifier.py` (QueryClass, QueryFeatures with model_is_moe field)
- `autopilot/policy.py` (ExecutionPolicy P0-P6, HardwareAvailability with flash_moe_available)
- `autopilot/bandit.py` (UCB1Bandit, ThompsonBandit)
- `autopilot/reward.py`
- `autopilot/logger.py`
- `autopilot/autopilot.py` (AutoPilotEngine enum, configurable decision tree + bandit paths)
- `autopilot/decision_tree.py` (deterministic rule tree from unified spec Section 9.2)

### Configs
- `configs/autopilot.yaml` (decision_tree default, bandit settings, reward weights)
- `configs/models.yaml` (qwen3.5-35b-a3b MoE added, ablation matrix updated C0-C7)
- `configs/hardware.yaml` (T6 NVMe SSD added, Flash-MoE build flags)

### Tests
- `tests/test_ane.py` (11 tests)
- `tests/test_pipeline.py` (includes T6, execution_policy, flash_moe P6 tests)
- `tests/test_metrics.py` (includes policy-label, P6/flash_moe fields)
- All 25 tests pass; ruff lint and format clean

## Partially Implemented

- `src/crossfire/compression/turboquant.py`: config exists, execution path is NotImplementedError
- `src/crossfire/compression/triattention.py`: KVCompressionStrategy enum + config exists, calibrate/apply are NotImplementedError stubs
- `src/crossfire/flashmoe/runtime.py`: FlashMoERuntime interface defined, run_inference/extract_sidecar are NotImplementedError (pending anemll-flash-llama.cpp bring-up)
- `benchmarks/perplexity.py`: config and validation exist, execution path is NotImplementedError
- `benchmarks/throughput.py`: config and validation exist, execution path is NotImplementedError

## Not Started

- Tests for new modules: `tests/test_flashmoe.py`, `tests/test_triattention.py`, `tests/test_decision_tree.py`
- AutoPilot unit tests (classifier, bandit, reward, logger, orchestrator)
- Dashboard / TUI layer
- Hardware bring-up and calibration artifacts under `results/`
- Flash-MoE binary build (`scripts/build_flash_moe.sh`)
- TriAttention calibration artifacts (pending paper author collaboration)

## Verification

- `pytest`: 25 passed
- `ruff check .`: clean
- `ruff format --check .`: clean

## Immediate Next Work

1. Add unit tests for new modules (flash_moe, triattention, decision_tree) + AutoPilot components
2. Wire `configs/autopilot.yaml` loading into `AutoPilot.__init__()`
3. Add `scripts/build_flash_moe.sh` for anemll-flash-llama.cpp build automation
4. Hardware bring-up: Tier 0 EXO baseline (T-0601 through T-0614)
