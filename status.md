# CROSSFIRE-X Status

Last updated: 2026-04-07
Branch: main
Latest commit: ee67abb (`feat: add CROSSFIRE-X specs and ANE speculative scaffolding`)
Tracker state: reconciled against the current repository contents on disk

## Summary

The repository has a solid scaffold for the ANE, distributed pipeline, metrics, benchmark, config, script, and test layers. The project is not yet in a runnable experimental state: the benchmark execution paths are still placeholders, the AutoPilot/policy layer does not exist yet, and the repo is still split between `CROSSFIRE-X` planning docs and `CROSSFIRE v2` public/package identifiers.

## Completed In Repo

- Project scaffold exists for `src/`, `tests/`, `configs/`, `scripts/`, `benchmarks/`, and `results/`
- ANE module scaffold exists: `draft_model.py`, `power.py`, `speculative.py`
- Distributed module scaffold exists: `pipeline.py`, `network.py`
- Compression module scaffold exists: `turboquant.py`, `kvcache.py`
- Metrics model exists in `src/crossfire/utils/metrics.py`
- Benchmark scaffolds exist: `perplexity.py`, `throughput.py`, `memory.py`
- Three test modules exist: `test_ane.py`, `test_pipeline.py`, `test_metrics.py`
- Tracking files now reflect repo reality instead of roadmap-only assumptions

## Partially Implemented

- `src/crossfire/ane/speculative.py`: bounded speculative step logic is implemented and tested
- `src/crossfire/compression/turboquant.py`: config exists, execution path still raises `NotImplementedError`
- `benchmarks/perplexity.py`: config and validation exist, execution path still raises `NotImplementedError`
- `benchmarks/throughput.py`: config and validation exist, execution path still raises `NotImplementedError`
- `src/crossfire/utils/metrics.py`: current schema supports ablation-style reporting, not the planned `P0-P5` policy schema

## Not Started

- `src/crossfire/autopilot/` package
- `configs/autopilot.yaml`
- Dashboard/TUI layer
- Hardware bring-up and calibration artifacts under `results/`
- Final public rename / release alignment

## Current Mismatches To Fix

- `README.md` still identifies the project as `CROSSFIRE v2`
- `pyproject.toml` is still version `0.1.0`
- `src/crossfire/__init__.py`, `pipeline.py`, and `metrics.py` still carry `v2` wording
- Public/docs naming and code naming are not yet aligned with the `CROSSFIRE-X` plan

## Verification

- `pytest`: passed (`21 passed`)
- `ruff check .`: passed
- `ruff format --check .`: passed
- Note: pytest and ruff emitted cache write warnings because `.pytest_cache` / `.ruff_cache` were not writable in this environment; the checks themselves still passed

## Immediate Next Work

1. Finish the rename/release alignment tasks in Phase 1.
2. Update the public/package/source identifiers so the repository is consistently `CROSSFIRE-X`.
3. Start the AutoPilot/policy refactor only after the naming split is resolved.
