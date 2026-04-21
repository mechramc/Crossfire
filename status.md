# CROSSFIRE-X Status

Last updated: 2026-04-21
Branch: main
Latest commit: (Session 16 Gemma pivot pending push)
Tracker state: software-layer tasks closed; Phase 6 hardware bring-up in progress. T-0601 (PC) and T-0602 (Mac) done; EXO nodes discovering each other over WiFi. USB4 tasks T-0603/T-0604/T-0605 deferred until cable is acquired. Model family switched from Qwen to Gemma 4 (see Session 16 checkpoint entry).

## Summary

Repository docs, code, configs, scripts, and tests are aligned with
`crossfire_x_final.docx`. The AutoPilot orchestrator now loads from
`configs/autopilot.yaml`, routes selections into `PipelineConfig`, and feeds
outcomes back into the reward + bandit layer. The anemll-flash-llama.cpp
build is automated by `scripts/build_flash_moe.sh`. The only `.docx` spec in
the repo root is `crossfire_x_final.docx`; the prior unified spec is archived.

## Completed In Repo

### Documentation And Trackers
- `README.md`, `CLAUDE.md`, `tasks.md`, `status.md`, `checkpoint.md` reflect final build spec
- `crossfire_x_final.docx` is the active spec reference; `crossfire_x_unified.docx`
  now lives under `docs/archive/`

### Implementation Layer
- `src/crossfire/distributed/pipeline.py` -- ComputeTarget uses T1-T5 (T5 is NVMe SSD)
- `src/crossfire/distributed/network.py` -- InterconnectType is USB4/5GbE/WiFi; no RDMA path
- `src/crossfire/autopilot/policy.py` -- `distributed_available` / `requires_distributed`;
  P4 is TriAttention-only; P0-P6 descriptions match final spec Section 9
- `src/crossfire/autopilot/decision_tree.py` -- `DecisionTreeThresholds` dataclass with
  tunable thresholds; `select_policy` accepts overrides from YAML
- `src/crossfire/autopilot/config_loader.py` -- `load_autopilot_yaml` and
  `build_autopilot_from_yaml` parse `configs/autopilot.yaml` into a wired AutoPilot
- `src/crossfire/autopilot/pipeline_integration.py` -- `apply_selection_to_pipeline`,
  `policy_requires_flash_moe`, and `run_autopilot_cycle` bridge AutoPilot and PipelineConfig
- `src/crossfire/utils/metrics.py` -- `interconnect` label + `interconnect_bytes` counter
- `configs/hardware.yaml` -- `interconnect` block (usb4 / 5gbe / wifi)
- `scripts/setup_mac.sh`, `scripts/setup_pc.sh` -- Thunderbolt IP bridge guidance,
  iperf3 probe, nc reachability check; EXO installed from source via uv
- `scripts/build_flash_moe.sh` -- cross-platform anemll-flash-llama.cpp build automation

### Tests
- `tests/test_ane.py`, `tests/test_pipeline.py`, `tests/test_metrics.py`
- `tests/test_flashmoe.py`, `tests/test_triattention.py`, `tests/test_decision_tree.py`
- `tests/test_autopilot.py` -- classifier, bandit, reward, logger, orchestrator, YAML loader,
  pipeline integration (select → apply → record cycle)
- `tests/test_policy.py` -- P0-P6 availability filtering across every HardwareAvailability combo

## Not Started

- USB4 hardware path: cable acquisition, Thunderbolt IP bridge, iperf3 baseline
  (T-0603 through T-0605) -- deferred; active interconnect is WiFi per `memory/interconnect.md`
- Model downloads, ANE conversion, Flash-MoE build (T-0607 through T-0612)
- Calibration runs for every policy (T-0613 through T-0626)
- Orion Forge serving (Phase 7)
- Textual dashboard and final evaluation deliverables (Phase 8)
- Software follow-ups that are not on the critical path: persist bandit state (T-0412),
  warm-start from calibration data (T-0413), end-to-end AutoPilot integration test (T-0414)

## Verification

- `pytest`: 129 passed
- `ruff check .`: clean
- `ruff format --check .`: clean

## Immediate Next Work

Phase 6 (Hardware Bring-Up And Calibration), Gemma 4 family:
1. Download Gemma 4 31B / E2B / 26B-A4B to both nodes (T-0607, T-0608, T-0612).
   SCOUT FIRST for T-0609 (E2B ANE conversion) and T-0612 (26B-A4B Flash-MoE
   sidecar extraction) -- both paths are unvalidated against Gemma 4 topology.
2. Convert Gemma 4 E2B draft to ANE CoreML (T-0609); build Rustane (T-0610);
   build anemll-flash-llama.cpp with CUDA on PC and Metal on Mac (T-0611)
3. Record P0 single-node baselines on PC and Mac (T-0613, T-0614). Note:
   Gemma 4 31B at Q8_0 (~33 GB) does not fit RTX 5090 single-node; PC P0
   must run TQ4_1S (~23 GB) or skip to distributed.
4. Record P1 distributed baseline over WiFi at 8K/16K/32K (T-0617)
5. Lock reward normalization constants from P1 baseline (T-0618)
6. Policy calibrations P2-P6 (T-0619 through T-0625) -> C0-C7 matrix (T-0626)
   with C6 now Gemma 4 31B @ 256K ctx (was Qwen 2.5 72B).

## Known unknowns to resolve during Phase 6

- **ANE conversion path for Gemma 4 E2B.** ANEMLL has no E2B benchmark;
  PLE architecture may not round-trip through CoreML cleanly. T-0609 must
  scout before committing to the 50 tok/s draft target.
- **Flash-MoE sidecar extraction for Gemma 4 26B-A4B.** anemll-flash-llama.cpp
  was built around Qwen / Kimi topology. Gemma's 128-expert + 1-shared-expert
  layout may need extractor patches. T-0612 scout before full calibration.

Deferred (hardware):
- USB4 40 Gbps cable + Thunderbolt IP bridge + throughput baseline (T-0603/T-0604/T-0605).
  Active interconnect is WiFi; composed TriAttention + TurboQuant compression is the
  bandwidth-hiding strategy. Revisit once the cable is acquired.
