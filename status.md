# CROSSFIRE-X Status

Last updated: 2026-04-21
Branch: main
Latest commit: d3c78c2 (`refactor: complete USB4/TCP-IP migration across configs, autopilot, metrics, setup (T-0128 P2)`)
Tracker state: implementation layer reconciled with `crossfire_x_final.docx` (T-0128, T-0129 closed)

## Summary

Repository docs, code, configs, scripts, and tests are now aligned with
`crossfire_x_final.docx`. The public framing (USB4/TCP-IP primary, 5GbE fallback,
composed TriAttention + TurboQuant compression thesis) is also the model used in
`ComputeTarget`, `ExecutionPolicy`, `HardwareAvailability`, `BenchmarkResult`,
`configs/hardware.yaml`, and the setup scripts. There is no remaining `T5_RDMA`
or `T6_NVME_SSD` naming in code, configs, scripts, or tests.

## Completed In Repo

### Documentation And Trackers
- `README.md`, `CLAUDE.md`, `tasks.md`, `status.md`, `checkpoint.md` reflect final build spec
- `crossfire_x_final.docx` is the active spec reference

### Implementation Layer (Session 13 migration)
- `src/crossfire/distributed/pipeline.py` -- ComputeTarget uses T1-T5 (T5 is NVMe SSD)
- `src/crossfire/distributed/network.py` -- InterconnectType is USB4/5GbE/WiFi; no RDMA path
- `src/crossfire/autopilot/policy.py` -- `distributed_available` / `requires_distributed`;
  P4 is TriAttention-only; P0-P6 descriptions match final spec Section 9
- `src/crossfire/utils/metrics.py` -- `interconnect` label + `interconnect_bytes` counter
  replace the legacy `rdma_active` boolean
- `configs/hardware.yaml` -- `interconnect` block (usb4 / 5gbe / wifi)
- `scripts/setup_mac.sh`, `scripts/setup_pc.sh` -- Thunderbolt IP bridge guidance,
  iperf3 probe, nc reachability check; RDMA enablement steps removed
- `tests/test_metrics.py`, `tests/test_pipeline.py` -- updated fixtures and round-trip
  coverage for the new interconnect fields

## Partially Implemented

- Spec canonicalization: `crossfire_x_unified.docx` still remains in the repo root (T-0127)
- AutoPilot orchestrator is scaffolded but not yet wired to load `configs/autopilot.yaml`
  or to drive pipeline execution (T-0409, T-0410, T-0411)

## Not Started

- Unit tests for Flash-MoE, TriAttention, decision tree, and AutoPilot components
  (T-0504 through T-0508)
- `scripts/build_flash_moe.sh` for anemll-flash-llama.cpp build automation (T-0309)
- Hardware bring-up: USB4 cable procurement, Thunderbolt IP bridge configuration,
  iperf3 baselines, model downloads, ANE model conversion (T-0601 through T-0612)
- Calibration runs for every policy (T-0613 through T-0626)
- Orion Forge serving (Phase 7)
- Textual dashboard and final evaluation deliverables (Phase 8)

## Verification

- `pytest`: 29 passed
- `ruff check .`: clean
- `ruff format --check .`: clean

## Immediate Next Work

1. Add unit tests for the remaining scaffolded modules: `test_flashmoe.py`,
   `test_triattention.py`, `test_decision_tree.py`
2. Wire `configs/autopilot.yaml` loading into `AutoPilot.__init__()`
3. Add `scripts/build_flash_moe.sh` for anemll-flash-llama.cpp automation
4. Archive `crossfire_x_unified.docx` so `crossfire_x_final.docx` is the only root spec
5. Begin hardware bring-up with USB4 baseline and 5GbE fallback measurements
