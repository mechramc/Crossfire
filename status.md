# CROSSFIRE-X Status

Last updated: 2026-04-11
Branch: main
Latest commit: 9558acf (`feat: add autopilot primitives and orchestrator`)
Tracker state: reconciled against repository contents after final build spec review (Session 12)

## Summary

Repository-facing docs and trackers now reflect `crossfire_x_final.docx` as the current
reference spec. The public framing has shifted from TB5/RDMA to USB4/TCP-IP with 5GbE
fallback, and the thesis now treats composed TriAttention + TurboQuant compression as the
reason the slower consumer interconnect is viable. The implementation layer has not yet been
migrated to match that framing: code, configs, scripts, and tests still encode the older
RDMA/T5/T6 model from Session 11.

## Completed In Repo

### Documentation And Trackers
- `README.md` updated for final build spec framing: USB4 primary link, 5GbE fallback,
  composed compression thesis, and revised tier ordering
- `CLAUDE.md` updated for final build spec framing and explicit implementation mismatch notes
- `tasks.md`, `status.md`, and `checkpoint.md` updated for Session 12
- `crossfire_x_final.docx` reviewed and treated as the current spec reference

### Current Scaffolds Still Present
- Core library modules still scaffold six-target / RDMA-oriented concepts from Session 11
- AutoPilot, Flash-MoE, TriAttention, and metrics scaffolds remain present as previously built
- Test suite still passes against the current scaffolded implementation

## Partially Implemented

- Final spec canonicalization is incomplete: `crossfire_x_unified.docx` still remains in the repo root
- `src/crossfire/distributed/pipeline.py` still models the interconnect as `T5_RDMA` and SSD as `T6_NVME_SSD`
- `src/crossfire/distributed/network.py`, `configs/hardware.yaml`, `scripts/setup_mac.sh`,
  `scripts/setup_pc.sh`, and related tests still describe TB5/RDMA prerequisites
- `src/crossfire/autopilot/policy.py` still describes the full-stack policy in RDMA-oriented terms
- Benchmark and hardware plans in code/config artifacts have not yet been rewritten around USB4

## Not Started

- Code/config/test reconciliation to the final USB4 build spec
- Setup-script migration from RDMA/TB5 bring-up to USB4/Thunderbolt IP bridge + 5GbE fallback
- Archival/removal of superseded `crossfire_x_unified.docx` from the repo root
- Hardware bring-up and calibration artifacts under `results/`
- Flash-MoE binary build (`scripts/build_flash_moe.sh`)

## Naming Mismatches

The repo still has a live naming mismatch between the final build spec and the implementation:
- Final spec: five compute targets plus USB4 interconnect
- Current code/config/tests: `T5_RDMA` interconnect plus `T6_NVME_SSD`

That mismatch is documented, not resolved. It must remain visible in trackers until the
implementation layer is migrated.

## Verification

- `pytest`: 25 passed (with a non-blocking `.pytest_cache` permission warning)
- `ruff check .`: clean
- `ruff format --check .`: clean

## Immediate Next Work

1. Reconcile the implementation layer with the final USB4 build spec: code, configs, scripts, and tests
2. Archive or relocate `crossfire_x_unified.docx` so `crossfire_x_final.docx` is the only active root spec
3. Add missing unit tests for Flash-MoE, TriAttention, decision tree, and AutoPilot components
4. Start hardware bring-up with USB4 baseline and 5GbE fallback measurements