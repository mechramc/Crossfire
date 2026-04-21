# CROSSFIRE-X Checkpoint Log

Purpose: durable work log for meaningful project sessions.
Rule: update this file before every `git push`.

---

## Session 14 - 2026-04-21: Phase 6 Mac bring-up (T-0602) and setup-script fixes

### What was done

**T-0602 executed end-to-end on Mac:**
- `sudo sysctl iogpu.wired_limit_mb=58982` set
- EXO cloned to `~/crossfire/exo`, `uv sync` completed (172 packages resolved)
- EXO dashboard built (`npm install && npm run build`, 980 modules transformed)
- llama.cpp TurboQuant+ fork cloned to `vendor/llama.cpp`, built with `GGML_METAL=ON` +
  `GGML_ACCELERATE=ON`; `llama-cli` loads with `turbo3 using 4-mag LUT` and sparse V
  dequant enabled
- ANEMLL cloned to `vendor/anemll`; Rustane cloned to `vendor/rustane`
- EXO binary `~/crossfire/exo/.venv/bin/exo --help` runs without import error

**Setup-script bugs found and fixed:**
- `scripts/setup_mac.sh:94` -- wrong llama.cpp fork URL; was `TheTom/llama.cpp.git` (404),
  corrected to `TheTom/llama-cpp-turboquant.git`
- `scripts/setup_pc.sh:70` -- identical wrong URL fixed the same way
- Both scripts: added EXO dashboard build step (`npm install && npm run build` in
  `$EXO_DIR/dashboard`) after `uv sync`. Without this, EXO fails at import time because
  `exo.shared.constants` resolves dashboard assets on module load. Includes an npm
  presence check that matches the existing `uv` / `cargo` check style.
- `CLAUDE.md` Key References: added the llama.cpp fork pointer
  (`github.com/TheTom/llama-cpp-turboquant`) alongside the existing research-workspace
  pointer (`github.com/TheTom/turboquant_plus`), and labeled which is which.

**Tracker updates:**
- `tasks.md` -- T-0602 marked done
- `status.md` -- Latest commit, Phase 6 progress, iperf3 prerequisite, and Immediate
  Next Work reordered (T-0601 now first)
- `checkpoint.md` -- this entry

### Verification
- `~/crossfire/exo/.venv/bin/exo --help` -- full usage output, no traceback
- `vendor/llama.cpp/build/bin/llama-cli --version` -- Metal backend loads cleanly
- `vendor/llama.cpp` origin = `TheTom/llama-cpp-turboquant` @ `4d24ad87b`
- `sysctl -n iogpu.wired_limit_mb` -- 58982
- Build log: `results/raw/setup_mac_20260421_*.log` -- zero errors, benign warnings only

### State at end of session
- Mac bring-up complete; T-0602 closed
- T-0601 (PC) unblocked by both setup-script fixes (URL + dashboard build)
- iperf3 still missing on both nodes; required before T-0605 USB4 throughput baseline
- `results/raw/` contains the setup and dashboard build logs (gitignored)

---

## Session 13 - 2026-04-21: Implementation-layer USB4/TCP-IP migration (T-0128, T-0129)

### What was done

**Code / config / test migration (commits e8c1698, d3c78c2):**
- `src/crossfire/distributed/pipeline.py` -- T5_RDMA removed; T6_NVME_SSD renumbered to T5;
  ComputeTarget enum now T1-T5 matching the final spec
- `src/crossfire/distributed/network.py` -- InterconnectType now usb4 / 5gbe / wifi; RDMA
  path deleted; explanatory note retained
- `src/crossfire/autopilot/policy.py` -- renamed `rdma_available` ->
  `distributed_available`, `requires_rdma` -> `requires_distributed`; P0-P6 descriptions
  rewritten to match final-spec Section 9 (P4 is TriAttention KV only, not TQ4_1S)
- `src/crossfire/utils/metrics.py` -- replaced boolean `rdma_active` with `interconnect`
  string label ("usb4"/"5gbe"/"wifi") plus `interconnect_bytes` counter to quantify
  compression savings on the data path
- `configs/hardware.yaml` -- `network: thunderbolt5_rdma` block replaced with
  `interconnect:` block containing `primary` (usb4), `fallback` (5gbe), `dev` (wifi)
- `scripts/setup_mac.sh`, `scripts/setup_pc.sh` -- RDMA/rdma_ctl enablement removed;
  Thunderbolt IP bridge guidance, iperf3 throughput probe, and nc reachability check added
- `tests/test_metrics.py` -- fixtures updated for new interconnect fields; added a
  round-trip test for `interconnect` + `interconnect_bytes`

**Tracker reconciliation:**
- `tasks.md` -- T-0128 and T-0129 marked done; Immediate Priorities updated
- `status.md` -- rewritten to reflect the completed migration; stale RDMA/T5/T6
  mismatch callouts removed
- `checkpoint.md` -- this entry added

### Verification
- `pytest`: 29 passed
- `ruff check .`: clean
- `ruff format --check .`: clean
- `grep -rn "T5_RDMA|T6_NVME_SSD"` in src/, configs/, scripts/, tests/: no matches
- Remaining `rdma` mentions in code are explanatory ("no RDMA" / "RDMA is not supported")

### State at end of session
- Implementation layer matches `crossfire_x_final.docx`
- No stale RDMA/T5/T6 naming in code, configs, scripts, or tests
- Outstanding work: unit tests for Flash-MoE / TriAttention / decision tree, AutoPilot
  yaml wiring, Flash-MoE build script, spec-doc canonicalization, hardware bring-up

---

## Session 12 - 2026-04-11: Final build spec doc and tracker reconciliation

### What was done

**Spec review:**
- Reviewed `crossfire_x_final.docx` against the repo's current public docs and trackers
- Identified the major spec delta: the project now assumes USB4 at 40 Gbps over TCP/IP with a
  5GbE fallback, not TB5 RDMA as the primary interconnect story
- Identified the new framing change: composed TriAttention + TurboQuant compression is now part of
  the central thesis because it makes the slower consumer interconnect practical

**Documentation updated:**
- Rewrote `README.md` to match the final build spec framing:
  - five compute targets instead of six-target RDMA/T6 public framing
  - USB4 primary data path and 5GbE fallback
  - composed compression thesis and revised experiment tiers
- Rewrote `CLAUDE.md` to match the same framing and to call out that the implementation layer still
  uses RDMA/T5/T6 naming from the prior scaffold session
- Reworked `tasks.md` to add Session 12 final-spec reconciliation tasks and to replace hardware
  bring-up tracker items that assumed TB5 RDMA with USB4 / Thunderbolt IP bridge / 5GbE tasks
- Rewrote `status.md` to describe the repo's actual state after the doc-only migration

**State documented but not changed in code:**
- The implementation still models the interconnect as `T5_RDMA` and the SSD as `T6_NVME_SSD`
- `configs/hardware.yaml`, `scripts/setup_mac.sh`, `scripts/setup_pc.sh`, and several tests still
  reflect the older TB5/RDMA assumptions
- `crossfire_x_unified.docx` still remains in the repo root, so spec canonicalization is partial

### Verification
- `pytest`: 25 passed (with a non-blocking `.pytest_cache` permission warning)
- `ruff check .`: clean
- `ruff format --check .`: clean

### State at end of session
- Public docs and trackers now reflect the final build spec
- Repo code/config/script/test layers still reflect the earlier unified-spec implementation model
- Follow-up work is required to reconcile naming, setup flow, and spec canonicalization

---

## Session 11 - 2026-04-09: Unified spec migration

### What was done

**Spec archival:**
- Moved `project_crossfire_spec.docx`, `crossfire_v2_spec.docx`, `CROSSFIRE-X_Spec.docx`,
  `CROSSFIRE_v2.1_Addendum.docx`, `Orion_Forge_v1.1_Addendum.docx`, and
  `CROSSFIRE-X_Implementation_Spec.md` to `docs/archive/`
- `crossfire_x_unified.docx` is now the sole canonical spec in the repository root

**New modules added:**
- `src/crossfire/flashmoe/__init__.py` -- Flash-MoE runtime integration package
- `src/crossfire/flashmoe/config.py` -- FlashMoEMode enum, SidecarConfig, SlotBankConfig, FlashMoEBuildConfig
- `src/crossfire/flashmoe/runtime.py` -- FlashMoEStats, FlashMoERuntime interface (stubs pending hardware)
- `src/crossfire/compression/triattention.py` -- KVCompressionStrategy enum, TriAttentionConfig (stubs)
- `src/crossfire/autopilot/decision_tree.py` -- deterministic policy selection tree (unified spec Section 9.2)
- `configs/autopilot.yaml` -- engine config (decision_tree default), bandit settings, reward weights

**Existing modules updated:**
- `src/crossfire/autopilot/policy.py` -- P6 policy added; flash_moe_available in HardwareAvailability;
  uses_flash_moe / requires_flash_moe in PolicyConfig
- `src/crossfire/autopilot/query_classifier.py` -- model_is_moe field added to QueryFeatures
- `src/crossfire/autopilot/autopilot.py` -- AutoPilotEngine enum; configurable decision tree + bandit paths;
  AutoPilotConfig.resolved_engine() for backwards compatibility
- `src/crossfire/distributed/pipeline.py` -- T6_NVME_SSD added to ComputeTarget; execution_policy and
  flash_moe_enabled added to PipelineConfig; all six targets documented
- `src/crossfire/utils/metrics.py` -- execution_policy as primary field (replaces ablation_config as
  primary); prefill_tok/s, ttft_ms, tok/W, acceptance_rate, flash_moe_hit_rate, flash_moe_active added;
  14-column TABLE_HEADERS
- `configs/models.yaml` -- qwen3.5-35b-a3b MoE model added with flash_moe_config; ablation matrix
  expanded from C0-C6 to C0-C7 with new dimensions (llama_runtime, triattention)
- `configs/hardware.yaml` -- T6 NVMe SSD target added to mac node; Flash-MoE build flags added
- `tests/test_pipeline.py` -- T6 enum value, execution_policy default, P6 flash_moe_enabled tests
- `tests/test_metrics.py` -- policy-label format, P6/flash_moe fields, 14-column table

**Documentation updated:**
- `README.md` -- full rewrite: 6 targets, 7 policies P0-P6, Flash-MoE, TriAttention, C0-C7, Orion Forge
- `CLAUDE.md` -- T6, P6, Flash-MoE, TriAttention, Orion Forge added to tech stack / constraints / tiers
- `status.md` -- reflects all Session 11 changes
- `tasks.md` -- complete overhaul: unified spec phases, new Flash-MoE / TriAttention / Orion Forge tasks

**Lint fixes:**
- Unicode characters (en-dash, em-dash, multiplication sign, micro sign) in docstrings/comments
  replaced with ASCII equivalents to satisfy ruff RUF002/RUF003
- Display placeholder em-dash strings in to_row() (u"\u2014") preserved correctly
- 3us latency string preserved after aggressive script fixed

### Verification
- `pytest`: 25 passed
- `ruff check .`: clean
- `ruff format --check .`: clean

### State at end of session
- Unified spec migration complete
- 6 compute targets, 7 policies, Flash-MoE, TriAttention, Orion Forge all scaffolded in code
- Hardware bring-up has not started; all T6/P6/Flash-MoE execution paths are stubs
- Immediate next work: unit tests for new modules, autopilot.yaml config wiring, build_flash_moe.sh

---

## Session 10 - 2026-04-08: AutoPilot orchestrator completed

### What was done
- Completed `T-0508`
- Added `src/crossfire/autopilot/autopilot.py`
- Implemented `AutoPilot` with per-query-class bandit instances, query classification,
  hardware-aware policy filtering, UCB1 or Thompson backend selection, reward computation,
  bandit updates, optional JSONL decision logging, and serializable policy-stat reporting
- Added supporting types: `BanditType`, `AutoPilotConfig`, `AutoPilotSelection`,
  `AutoPilotOutcome`, `AutoPilotBaselines`
- Exported the new orchestrator types from `src/crossfire/autopilot/__init__.py`

### Verification
- End-to-end smoke check: classify -> select policy -> record outcome -> decision log write
- `pytest`: passed, `ruff check .`: clean, `ruff format --check .`: clean

---

## Session 9 - 2026-04-08: AutoPilot primitives batch 1

### What was done
- Completed `T-0501` through `T-0507`
- Created `src/crossfire/autopilot/__init__.py`
- Added `query_classifier.py`, `policy.py` (P0-P5), `bandit.py` (UCB1 + Thompson),
  `reward.py`, `logger.py`

### Verification
- Targeted smoke checks after each task
- `pytest`: passed, `ruff check .`: clean, `ruff format --check .`: clean

---

## Session 8 - 2026-04-07: Phase 1 rename/release alignment completed

### What was done
- Completed T-0102 through T-0107
- Renamed public README from `CROSSFIRE v2` to `CROSSFIRE-X`
- Updated README results table from ablation config labels to P0-P5 policy labels
- Bumped package version from 0.1.0 to 0.2.0
- Updated package metadata description, source docstrings, `src/crossfire/__init__.py`

### Verification
- `pytest` (21 passed), `ruff check .`, `ruff format --check .` all clean in `.venv`

---

## Session 7 - 2026-04-07: Tracker reconciliation and push-gate rules

### What was done
- Re-audited repo contents; rebuilt `tasks.md` as atomic ledger grounded in disk state
- Rewrote `status.md` to describe actual repo state: scaffolded but not experimentally runnable
- Rewrote `AGENTS.md` with push-gate tracker rules

---

## Session 6 - 2026-04-07: Phase 1 batch 1

### What was done
- Completed T-0101: renamed `CLAUDE.md` from `CROSSFIRE v2` references to `CROSSFIRE-X`

---

## Session 5 - 2026-04-07: CROSSFIRE-X spec + tasks rewrite

### What was done
- Reviewed CROSSFIRE-X spec; wrote `CROSSFIRE-X_Implementation_Spec.md`
- Reworked project plan around P0-P5, AutoPilot, dashboard, and impossible-scenario experiments

---

## Session 4 - 2026-04-07: Speculative harness prep

### What was done
- Added `src/crossfire/ane/speculative.py`
- Implemented bounded speculative decoding step (draft proposes, verifier accepts/rejects)
- Added focused tests for accept, reject, and empty-result paths

---

## Session 3 - 2026-04-07: First tracker pass

### What was done
- Created initial versions of `tasks.md`, `status.md`, `checkpoint.md`

---

## Session 2 - 2026-04-07: v2 spec alignment

### What was done
- Updated docs and code toward EXO + ANE architecture
- Added ANE module scaffolds, distributed pipeline/network scaffolds
- Updated configs and scripts for EXO, ANEMLL, Rustane, RDMA

---

## Session 1 - 2026-04-06: Project initialization

### Committed as
- `4611d21` -- `Initialize project structure with scaffolded source, benchmarks, and configs`