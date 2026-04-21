# CROSSFIRE-X Task Ledger

Last updated: 2026-04-21
Purpose: Atomic project task list grounded in the current repository state.
Rule: Only mark a task done when the code, file, or artifact exists in this repo or the required hardware action has been executed and recorded.

## Legend

- [x] Done
- [ ] Pending
- [~] In progress / partially implemented
- [!] Blocked by hardware, external tooling, or prerequisite work

## Phase 0 - Repo Baseline And Tracking

- [x] T-0001 Create Python project scaffold
- [x] T-0002 Create `pyproject.toml`
- [x] T-0003 Create `.gitignore`
- [x] T-0004 Add Apache 2.0 `LICENSE`
- [x] T-0005 Create public `README.md`
- [x] T-0006 Archive superseded spec docs to `docs/archive/`
- [x] T-0007 Create and maintain `tasks.md`
- [x] T-0008 Create and maintain `status.md`
- [x] T-0009 Create and maintain `checkpoint.md`
- [x] T-0010 Create and maintain `AGENTS.md`

## Phase 1 - Unified Spec Migration (Session 11)

- [x] T-0101 Archive all superseded specs; `crossfire_x_unified.docx` became the Session 11 canonical spec
- [x] T-0102 Update `README.md` for unified spec (6 targets, 7 policies, Flash-MoE, TriAttention, C0-C7)
- [x] T-0103 Update `CLAUDE.md` for unified spec (T6, P6, Flash-MoE, TriAttention, Orion Forge)
- [x] T-0104 Update `status.md` for unified spec state
- [x] T-0105 Update `tasks.md` for unified spec phases
- [x] T-0106 Update `checkpoint.md` with Session 11 entry
- [x] T-0107 Add `T6_NVME_SSD` to `ComputeTarget` enum in `pipeline.py`
- [x] T-0108 Add `execution_policy` and `flash_moe_enabled` to `PipelineConfig`
- [x] T-0109 Add `P6` to `ExecutionPolicy` enum and `POLICY_REGISTRY`
- [x] T-0110 Add `flash_moe_available` to `HardwareAvailability`
- [x] T-0111 Migrate `BenchmarkResult` to policy-native schema (execution_policy primary)
- [x] T-0112 Add 14-column metrics table with P6/Flash-MoE/TriAttention fields
- [x] T-0113 Create `src/crossfire/flashmoe/` module (config, runtime, __init__)
- [x] T-0114 Create `src/crossfire/compression/triattention.py` stub
- [x] T-0115 Create `src/crossfire/autopilot/decision_tree.py`
- [x] T-0116 Add `AutoPilotEngine` enum and configurable engine selection to `autopilot.py`
- [x] T-0117 Add `model_is_moe` field to `QueryFeatures`
- [x] T-0118 Create `configs/autopilot.yaml`
- [x] T-0119 Update `configs/models.yaml` (MoE model + C0-C7 ablation matrix; model family migrated to Gemma 4 in Session 16)
- [x] T-0120 Update `configs/hardware.yaml` (T6 NVMe SSD, Flash-MoE build flags)
- [x] T-0121 Update pipeline tests for T6 and P6
- [x] T-0122 Update metrics tests for policy-native schema and P6 fields

## Phase 1A - Final Build Spec Reconciliation (Session 12)

- [x] T-0123 Review `crossfire_x_final.docx` against current repo docs and trackers
- [x] T-0124 Update `README.md` for USB4/TCP-IP interconnect, 5GbE fallback, and composed compression thesis
- [x] T-0125 Update `CLAUDE.md` for final build spec framing and current implementation mismatch notes
- [x] T-0126 Update `status.md` and `checkpoint.md` for final build spec state
- [x] T-0127 Canonicalize spec docs around `crossfire_x_final.docx`; `crossfire_x_unified.docx` archived to `docs/archive/`
- [x] T-0128 Reconcile code/config/test naming from RDMA/T5/T6 model to USB4 interconnect + T5 NVMe final spec model
- [x] T-0129 Update setup scripts and hardware/config docs from TB5 RDMA prerequisites to USB4/Thunderbolt IP bridge + 5GbE fallback

## Phase 2 - Core Library Scaffolds

- [x] T-0201 `src/crossfire/ane/draft_model.py` -- ANEBackend, DraftModelConfig, DraftResult
- [x] T-0202 `src/crossfire/ane/power.py` -- PowerSnapshot, ANE power constants
- [x] T-0203 `src/crossfire/ane/speculative.py` -- bounded speculative decode step
- [x] T-0204 `src/crossfire/distributed/pipeline.py` -- T1-T6, NodeRole, PipelineConfig
- [x] T-0205 `src/crossfire/distributed/network.py` -- InterconnectType, NetworkStats, TCP probe
- [x] T-0206 `src/crossfire/utils/metrics.py` -- BenchmarkResult (14-column policy schema)
- [x] T-0207 `src/crossfire/compression/turboquant.py` -- QuantConfig (execution stub pending)
- [x] T-0208 `src/crossfire/compression/kvcache.py` -- KVCacheConfig, llama.cpp CLI arg rendering
- [x] T-0209 `src/crossfire/compression/triattention.py` -- KVCompressionStrategy, TriAttentionConfig (stubs)
- [x] T-0210 `src/crossfire/flashmoe/config.py` -- FlashMoEMode, SidecarConfig, SlotBankConfig, FlashMoEBuildConfig
- [x] T-0211 `src/crossfire/flashmoe/runtime.py` -- FlashMoEStats, FlashMoERuntime (stubs)

## Phase 3 - Benchmark And Script Scaffolds

- [x] T-0301 `benchmarks/perplexity.py` (execution stub pending)
- [x] T-0302 `benchmarks/throughput.py` (execution stub pending)
- [x] T-0303 `benchmarks/memory.py`
- [x] T-0304 `scripts/setup_pc.sh`
- [x] T-0305 `scripts/setup_mac.sh`
- [x] T-0306 `scripts/run_experiment.sh`
- [x] T-0307 `configs/models.yaml` (includes MoE models, C0-C7 ablation matrix)
- [x] T-0308 `configs/hardware.yaml` (includes T6 NVMe SSD)
- [x] T-0309 `scripts/build_flash_moe.sh` -- automate anemll-flash-llama.cpp cmake build

## Phase 4 - AutoPilot

- [x] T-0401 `src/crossfire/autopilot/query_classifier.py`
- [x] T-0402 `src/crossfire/autopilot/policy.py` (P0-P6 registry)
- [x] T-0403 `src/crossfire/autopilot/bandit.py` (UCB1, Thompson)
- [x] T-0404 `src/crossfire/autopilot/reward.py`
- [x] T-0405 `src/crossfire/autopilot/logger.py`
- [x] T-0406 `src/crossfire/autopilot/autopilot.py` (AutoPilotEngine, decision tree + bandit paths)
- [x] T-0407 `src/crossfire/autopilot/decision_tree.py` (deterministic rule tree)
- [x] T-0408 `configs/autopilot.yaml`
- [x] T-0409 Wire `configs/autopilot.yaml` loading into `AutoPilot.__init__()` (via `build_autopilot_from_yaml`)
- [x] T-0410 Wire AutoPilot selection into pipeline execution (via `apply_selection_to_pipeline`)
- [x] T-0411 Wire outcome reporting back into reward + bandit updates (via `run_autopilot_cycle`)
- [ ] T-0412 Persist bandit state across runs
- [ ] T-0413 Warm-start AutoPilot from calibration data
- [ ] T-0414 Add end-to-end AutoPilot integration test

## Phase 5 - Test Coverage

- [x] T-0501 `tests/test_ane.py` (11 tests: draft model, power, speculative)
- [x] T-0502 `tests/test_pipeline.py` (T6, execution_policy, P6 Flash-MoE)
- [x] T-0503 `tests/test_metrics.py` (policy schema, P6 fields, 14-column table)
- [x] T-0504 `tests/test_flashmoe.py` -- FlashMoEMode, SlotBankConfig, FlashMoEBuildConfig, FlashMoERuntime
- [x] T-0505 `tests/test_triattention.py` -- KVCompressionStrategy, TriAttentionConfig
- [x] T-0506 `tests/test_decision_tree.py` -- all branches of `select_policy()`
- [x] T-0507 `tests/test_autopilot.py` -- classifier, bandit, reward, logger, orchestrator
- [x] T-0508 `tests/test_policy.py` -- P0-P6 availability filtering with `HardwareAvailability`

## Phase 6 - Hardware Bring-Up And Calibration

- [x] T-0601 Verify PC environment and run `scripts/setup_pc.sh` (Session 15; CUDA 13.2 toolkit + Node 22 prerequisites installed in WSL; EXO source-install + dashboard + CUDA llama.cpp build all green; `~/crossfire/exo/.venv/bin/exo -v` launches cleanly, discovers Mac peer at `192.168.4.41:52415` over WiFi mDNS; API live on `localhost:52415`)
- [x] T-0602 Verify Mac environment and run `scripts/setup_mac.sh`
- [!] T-0603 Acquire and verify USB4 40 Gbps active cable
- [!] T-0604 Configure Thunderbolt IP bridge / TCP-IP networking between nodes
- [!] T-0605 Measure USB4 throughput between nodes and record baseline
- [!] T-0606 Validate 5GbE fallback link and discovery path
- [!] T-0607 Download Gemma 4 31B (`google/gemma-4-31B-it`) model artifacts (Q8_0 + TQ4_1S)
- [!] T-0608 Download Gemma 4 E2B (`google/gemma-4-E2B-it`) draft model artifacts
- [!] T-0609 Convert Gemma 4 E2B draft into ANE-ready CoreML format (SCOUT FIRST: ANEMLL has no published E2B benchmark; PLE architecture may not round-trip through CoreML cleanly)
- [!] T-0610 Build Rustane
- [!] T-0611 Build anemll-flash-llama.cpp with Metal flags (Mac) and CUDA flags (PC)
- [!] T-0612 Download Gemma 4 26B-A4B (`google/gemma-4-26B-A4B-it`) and run Flash-MoE sidecar extraction (SCOUT FIRST: 128-expert + 1-shared-expert topology is not what the extractor was built for)
- [!] T-0613 Record P0 single-node PC baseline (C0 reference)
- [!] T-0614 Record P0 single-node Mac baseline (C0 reference)
- [!] T-0615 Record baseline perplexity runs (wikitext-2-raw-v1, 20 chunks, c=512)
- [!] T-0616 Record baseline power measurements
- [!] T-0617 Record distributed P1 baseline at 8K / 16K / 32K (C1)
- [!] T-0618 Lock reward normalization constants from P1 baseline
- [!] T-0619 Run ANE zero-interference gate (T3 load with no GPU regression)
- [!] T-0620 Run P2 ANE speculative calibration (C2)
- [!] T-0621 Run P3 cross-node compression calibration (C3/C4 prework)
- [!] T-0622 Run P4 TriAttention KV calibration (C4)
- [!] T-0623 Run P5 full-stack calibration (C5)
- [!] T-0624 Run long-context stretch calibration (C6) -- Gemma 4 31B at 256K ctx, distributed + TriAttention
- [!] T-0625 Run P6 Flash-MoE single-node slot-bank calibration with Gemma 4 26B-A4B (C7)
- [!] T-0626 Compile C0-C7 ablation matrix artifact under `results/`

## Phase 7 - Orion Forge Serving

- [!] T-0701 Obtain or generate Orion Forge model artifacts
- [!] T-0702 Run Flash-MoE sidecar extraction for Orion Forge (KALAVAI adapter conversion)
- [!] T-0703 Verify slot-bank hit rate >= 90% after warmup
- [!] T-0704 Benchmark Orion Forge P6 throughput vs P0 baseline
- [!] T-0705 Document Orion Forge serving configuration in `configs/`

## Phase 8 - Dashboard, Evaluation, And Deliverables

- [ ] T-0801 Create Textual dashboard package scaffold
- [ ] T-0802 Implement utilization panel
- [ ] T-0803 Implement AutoPilot status panel (engine, current policy, decision log)
- [ ] T-0804 Implement live metrics panel
- [ ] T-0805 Implement Flash-MoE hit rate panel
- [ ] T-0806 Implement decision history panel
- [ ] T-0807 Create dashboard entry point
- [ ] T-0808 Define mixed-workload evaluation config (P0-P6)
- [ ] T-0809 Run AutoPilot workload evaluation
- [ ] T-0810 Run oracle-policy comparison
- [ ] T-0811 Run random-policy comparison
- [ ] T-0812 Compute regret and convergence analysis
- [ ] T-0813 Compile final C0-C7 + P0-P6 results matrix
- [ ] T-0814 Generate charts (throughput, power, KV hit rate, acceptance rate)
- [ ] T-0815 Update public README with final results
- [ ] T-0816 Prepare demo / blog / community deliverables

## Immediate Priorities

1. Hardware bring-up: USB4 cable, Thunderbolt IP bridge, iperf3 baseline, 5GbE fallback (T-0601 through T-0606)
2. Model downloads + ANE draft conversion + Flash-MoE sidecar build (T-0607 through T-0612)
3. Calibration baselines: P0/P1 reference runs, reward normalization lock-in (T-0613 through T-0618)
4. Policy calibration runs P2-P6 (T-0619 through T-0625) and the C0-C7 ablation matrix (T-0626)
5. Software-side follow-ups that don't block hardware: persist bandit state (T-0412),
   warm-start from calibration data (T-0413), end-to-end AutoPilot integration test (T-0414)