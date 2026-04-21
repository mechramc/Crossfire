# CROSSFIRE-X Status

Last updated: 2026-04-21
Branch: main
Latest commit: 12b233e (merged with origin/main; setup script fixes pending this session)
Tracker state: software-layer tasks closed; Phase 6 hardware bring-up in progress. T-0602 (Mac) done; T-0601 (PC) unblocked.

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

- Hardware bring-up: USB4 cable, Thunderbolt IP bridge, iperf3 baselines,
  model downloads, ANE model conversion (T-0601, T-0603 through T-0612)
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

Phase 6 (Hardware Bring-Up And Calibration):
1. Run `scripts/setup_pc.sh` on the PC (T-0601); EXO binary, llama.cpp CUDA build, dashboard assets
2. Install iperf3 on Mac (`brew install iperf3`) and PC for T-0605 baseline
3. Acquire USB4 40 Gbps active cable; configure Thunderbolt IP bridge between nodes
4. Record USB4 iperf3 baseline; validate 5GbE fallback
5. Download 27B / 0.6B / 35B-A3B models; convert 0.6B draft to ANE CoreML
6. Build Rustane and anemll-flash-llama.cpp; extract MoE sidecar
7. Record P0/P1 baselines; lock reward normalization constants
