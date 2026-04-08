# CROSSFIRE-X Task Ledger

Last updated: 2026-04-07
Purpose: Atomic project task list grounded in the current repository state.
Rule: Only mark a task done when the code, file, or artifact exists in this repo or the required hardware action has been executed and recorded.

## Legend

- [x] Done
- [ ] Pending
- [~] In progress / partially implemented
- [!] Blocked by hardware, external tooling, or prerequisite work

## Phase 0 - Repo Baseline And Tracking

- [x] T-0001 Create Python project scaffold in `src/`, `tests/`, `configs/`, `scripts/`, `benchmarks/`, and `results/`
- [x] T-0002 Create `pyproject.toml` with package metadata, pytest config, and ruff config
- [x] T-0003 Create repository `.gitignore`
- [x] T-0004 Add Apache 2.0 `LICENSE`
- [x] T-0005 Create public `README.md`
- [x] T-0006 Create internal implementation spec `CROSSFIRE-X_Implementation_Spec.md`
- [x] T-0007 Create and maintain `tasks.md`
- [x] T-0008 Create and maintain `status.md`
- [x] T-0009 Create and maintain `checkpoint.md`
- [x] T-0010 Create and maintain `AGENTS.md` with push-gate tracker rules

## Phase 1 - Rename And Release Alignment

- [x] T-0101 Rename internal planning docs to `CROSSFIRE-X` in `CLAUDE.md`
- [ ] T-0102 Rename public `README.md` from `CROSSFIRE v2` to `CROSSFIRE-X`
- [ ] T-0103 Update `README.md` result table from `C0-C6` naming to `P0-P5` policy naming
- [ ] T-0104 Update `pyproject.toml` version from `0.1.0` to the intended next release
- [ ] T-0105 Update `pyproject.toml` description to include the self-optimizing / AutoPilot direction if still intended
- [ ] T-0106 Rename source docstrings that still say `CROSSFIRE v2`
- [ ] T-0107 Update `src/crossfire/__init__.py` package docstring/version to match release plan

## Phase 2 - Core Library Scaffolds Already In Repo

- [x] T-0201 Create `src/crossfire/ane/draft_model.py`
- [x] T-0202 Implement `ANEBackend` enum
- [x] T-0203 Implement `DraftModelConfig` dataclass with path and context validation
- [x] T-0204 Implement `DraftResult` dataclass
- [x] T-0205 Create `src/crossfire/ane/power.py`
- [x] T-0206 Implement `PowerSnapshot` dataclass and ANE power constants
- [x] T-0207 Create `src/crossfire/ane/speculative.py`
- [x] T-0208 Implement bounded speculative decode step orchestration
- [x] T-0209 Create `src/crossfire/distributed/pipeline.py`
- [x] T-0210 Implement `ComputeTarget` enum for T1-T5
- [x] T-0211 Implement `NodeRole` enum for PREFILL / DECODE / DRAFT
- [x] T-0212 Implement pipeline dataclasses (`ComputeTargetConfig`, `NodeConfig`, `PipelineConfig`)
- [x] T-0213 Implement `PipelineConfig.validate()` checks for PREFILL / DECODE / DRAFT / RDMA
- [x] T-0214 Create `src/crossfire/distributed/network.py`
- [x] T-0215 Implement `InterconnectType` enum and network constants
- [x] T-0216 Implement `NetworkStats` dataclass
- [x] T-0217 Implement TCP connectivity probe helper
- [x] T-0218 Create `src/crossfire/utils/metrics.py`
- [x] T-0219 Implement `BenchmarkResult` dataclass for current ablation-style metrics
- [x] T-0220 Implement table row formatting and headers for current metrics table
- [x] T-0221 Create `src/crossfire/compression/turboquant.py`
- [x] T-0222 Implement `QuantConfig` dataclass
- [~] T-0223 Implement actual TurboQuant+ subprocess execution in `quantize_model()`
- [x] T-0224 Create `src/crossfire/compression/kvcache.py`
- [x] T-0225 Implement `KVCacheConfig` dataclass
- [x] T-0226 Implement llama.cpp CLI arg rendering for KV cache config

## Phase 3 - Benchmark And Script Scaffolds Already In Repo

- [x] T-0301 Create `benchmarks/perplexity.py`
- [~] T-0302 Implement actual perplexity execution path in `run_perplexity()`
- [x] T-0303 Create `benchmarks/throughput.py`
- [~] T-0304 Implement actual throughput execution path in `run_throughput()`
- [x] T-0305 Create `benchmarks/memory.py`
- [x] T-0306 Implement system memory snapshot collection
- [x] T-0307 Create `scripts/setup_pc.sh`
- [x] T-0308 Create `scripts/setup_mac.sh`
- [x] T-0309 Create `scripts/run_experiment.sh`
- [x] T-0310 Create `configs/models.yaml`
- [x] T-0311 Create `configs/hardware.yaml`

## Phase 4 - Current Test And Quality Coverage

- [x] T-0401 Create `tests/test_pipeline.py`
- [x] T-0402 Cover valid distributed pipeline configuration
- [x] T-0403 Cover missing PREFILL / DECODE validation failures
- [x] T-0404 Cover speculative decode DRAFT validation path
- [x] T-0405 Create `tests/test_metrics.py`
- [x] T-0406 Cover current `BenchmarkResult.to_row()` formatting
- [x] T-0407 Cover benchmark timestamp/default fields
- [x] T-0408 Create `tests/test_ane.py`
- [x] T-0409 Cover draft model validation failures
- [x] T-0410 Cover ANE enum/constants and power snapshot behavior
- [x] T-0411 Cover speculative decode accept / reject / empty-result paths
- [x] T-0412 Run `pytest` after the current tracking-doc update
- [x] T-0413 Run `ruff check .` after the current tracking-doc update
- [x] T-0414 Run `ruff format --check .` after the current tracking-doc update

## Phase 5 - AutoPilot / Policy Refactor Roadmap

- [ ] T-0501 Create `src/crossfire/autopilot/__init__.py`
- [ ] T-0502 Implement query classifier module
- [ ] T-0503 Implement execution policy registry for `P0-P5`
- [ ] T-0504 Implement UCB1 bandit
- [ ] T-0505 Implement Thompson sampling alternative
- [ ] T-0506 Implement reward calculation
- [ ] T-0507 Implement decision logger
- [ ] T-0508 Implement `AutoPilot` orchestrator
- [ ] T-0509 Add `execution_policy` to pipeline configuration
- [ ] T-0510 Replace `ablation_config` metrics with `execution_policy`
- [ ] T-0511 Add new benchmark fields: tok/W, acceptance rate, TTFT, prefill tok/s, decode tok/s
- [ ] T-0512 Add AutoPilot benchmark fields: query class, reward, exploration flag
- [ ] T-0513 Add unit tests for query classification
- [ ] T-0514 Add unit tests for UCB1 bandit
- [ ] T-0515 Add unit tests for Thompson sampling
- [ ] T-0516 Add unit tests for reward calculation
- [ ] T-0517 Add unit tests for decision logging
- [ ] T-0518 Update metrics tests for the policy-based schema
- [ ] T-0519 Update pipeline tests for execution-policy validation
- [ ] T-0520 Create `configs/autopilot.yaml`
- [ ] T-0521 Wire AutoPilot selection into pipeline execution
- [ ] T-0522 Wire outcome reporting back into reward + bandit updates
- [ ] T-0523 Persist bandit state across runs
- [ ] T-0524 Warm-start AutoPilot from calibration data
- [ ] T-0525 Add an end-to-end AutoPilot integration test

## Phase 6 - Hardware Bring-Up And Calibration

- [!] T-0601 Verify PC environment and run `scripts/setup_pc.sh`
- [!] T-0602 Verify Mac environment and run `scripts/setup_mac.sh`
- [!] T-0603 Acquire and verify Thunderbolt 5 cable
- [!] T-0604 Confirm macOS version and RDMA prerequisites on Mac
- [!] T-0605 Enable RDMA and confirm EXO node discovery
- [!] T-0606 Download primary 27B model artifacts
- [!] T-0607 Download 0.6B draft model artifacts
- [!] T-0608 Convert 0.6B draft model into ANE-ready format
- [!] T-0609 Build Rustane
- [!] T-0610 Record P0 single-node PC baseline
- [!] T-0611 Record P0 single-node Mac baseline
- [!] T-0612 Record baseline perplexity runs
- [!] T-0613 Record baseline power measurements
- [!] T-0614 Record distributed P1 baseline at 8K / 16K / 32K
- [!] T-0615 Lock reward normalization constants from P1 baseline
- [!] T-0616 Run ANE zero-interference gate
- [!] T-0617 Run P2 ANE speculative calibration
- [!] T-0618 Compress 27B to TQ4_1S
- [!] T-0619 Verify EXO can load TQ4_1S artifacts
- [!] T-0620 Run P3 compressed calibration
- [!] T-0621 Run P4 compressed + turbo KV calibration
- [!] T-0622 Run P5 full-stack calibration
- [!] T-0623 Compile calibration matrix artifact under `results/`

## Phase 7 - Dashboard, Evaluation, And Deliverables

- [ ] T-0701 Create Textual dashboard package scaffold
- [ ] T-0702 Add dashboard dependencies if approved
- [ ] T-0703 Implement utilization panel
- [ ] T-0704 Implement AutoPilot status panel
- [ ] T-0705 Implement live metrics panel
- [ ] T-0706 Implement decision history panel
- [ ] T-0707 Create dashboard entry point
- [ ] T-0708 Define mixed-workload evaluation config
- [ ] T-0709 Run AutoPilot workload evaluation
- [ ] T-0710 Run oracle-policy comparison
- [ ] T-0711 Run random-policy comparison
- [ ] T-0712 Compute regret and convergence analysis
- [ ] T-0713 Compile final results matrix
- [ ] T-0714 Generate charts
- [ ] T-0715 Update public README with final results
- [ ] T-0716 Prepare demo / blog / community deliverables

## Immediate Priorities

1. Finish rename/release alignment in Phase 1.
2. Finish the public/package/source rename so the repository is consistently `CROSSFIRE-X`.
3. Decide whether the AutoPilot / policy refactor is still the active direction, then execute Phase 5 against a consistent naming baseline.
