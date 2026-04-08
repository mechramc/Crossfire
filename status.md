# CROSSFIRE-X Status

Last updated: 2026-04-08
Branch: main
Latest commit: cbb7adc (`add status, checkpoint, tasks files`)
Tracker state: reconciled against the current repository contents on disk after initial AutoPilot primitives were added

## Summary

The repository has a solid scaffold for the ANE, distributed pipeline, metrics, benchmark, config, script, and test layers. The project is not yet in a runnable experimental state: benchmark execution paths are still placeholders and the pipeline/metrics layers have not yet been migrated to execution-policy-native schemas. Phase 1 rename/release alignment is complete, and the first AutoPilot layer now includes both primitives and a top-level orchestrator.

## Completed In Repo

- Project scaffold exists for `src/`, `tests/`, `configs/`, `scripts/`, `benchmarks/`, and `results/`
- ANE module scaffold exists: `draft_model.py`, `power.py`, `speculative.py`
- Distributed module scaffold exists: `pipeline.py`, `network.py`
- Compression module scaffold exists: `turboquant.py`, `kvcache.py`
- Metrics model exists in `src/crossfire/utils/metrics.py`
- Benchmark scaffolds exist: `perplexity.py`, `throughput.py`, `memory.py`
- Three test modules exist: `test_ane.py`, `test_pipeline.py`, `test_metrics.py`
- Tracking files now reflect repo reality instead of roadmap-only assumptions
- Phase 1 rename/release alignment is complete for `README.md`, `pyproject.toml`, and `src/crossfire/__init__.py`
- Initial AutoPilot layer now exists: query classifier, execution-policy registry, UCB1 bandit, Thompson bandit, reward calculation, decision logger, and top-level orchestrator

## Partially Implemented

- `src/crossfire/ane/speculative.py`: bounded speculative step logic is implemented and tested
- `src/crossfire/compression/turboquant.py`: config exists, execution path still raises `NotImplementedError`
- `benchmarks/perplexity.py`: config and validation exist, execution path still raises `NotImplementedError`
- `benchmarks/throughput.py`: config and validation exist, execution path still raises `NotImplementedError`
- `src/crossfire/utils/metrics.py`: current schema supports ablation-style reporting, not the planned `P0-P5` policy schema
- `src/crossfire/autopilot/`: orchestrator exists, but persistence/config wiring and dedicated unit tests are still missing

## Not Started

- `configs/autopilot.yaml`
- Dashboard/TUI layer
- Hardware bring-up and calibration artifacts under `results/`

## Current Mismatches To Fix

- `src/crossfire/utils/metrics.py` still uses the ablation-style `ablation_config` schema instead of execution policies
- `README.md` results are now policy-named, but benchmark code and metrics are not yet policy-native
- `src/crossfire/distributed/pipeline.py` has no `execution_policy` concept yet
- No dedicated AutoPilot unit tests exist yet; current verification is smoke-based plus existing repo checks
- `configs/autopilot.yaml` does not exist yet, so the orchestrator is still code-configured only
- The package remains named `crossfire`; no package/module rename has been attempted

## Verification

- `.venv` created locally for isolated verification
- `pytest`: passed (`21 passed`)
- `ruff check .`: passed
- `ruff format --check .`: passed

## Immediate Next Work

1. Update pipeline and metrics schemas from ablation naming to execution-policy naming.
2. Add the first dedicated AutoPilot unit tests for classifier, bandits, reward, logging, and orchestration.
3. Add `configs/autopilot.yaml` and wire configuration loading into the new AutoPilot layer.
