# CROSSFIRE-X Status

Last updated: 2026-04-21
Branch: main
Latest commit: (Session 17 T-0609 scout pending push)
Tracker state: software-layer tasks closed; Phase 6 hardware bring-up in progress. T-0601 (PC), T-0602 (Mac), T-0606 (WiFi discovery) done; EXO PC is cluster Master, Mac is Worker. T-0609 Gemma 4 E2B -> ANE scout complete: viability proven on M4 Max ANE via `john-rocky/CoreML-LLM` pre-converted bundle; correct-output harness port tracked as T-0609a. USB4 tasks T-0603/T-0604/T-0605 deferred until cable is acquired.

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
- Model downloads for dense primary and MoE (T-0607, T-0612); ANE conversion
  harness port (T-0609a); Rustane + Flash-MoE builds (T-0610, T-0611)
- Calibration runs for every policy (T-0613 through T-0626)
- Orion Forge serving (Phase 7)
- Textual dashboard and final evaluation deliverables (Phase 8)
- Software follow-ups that are not on the critical path: persist bandit state (T-0412),
  warm-start from calibration data (T-0413), end-to-end AutoPilot integration test (T-0414)
- Stretch: upstream Gemma 4 support PR to ANEMLL (T-0609b)

## Session 17 scout artifacts (local, gitignored)

- `models/gemma-4-E2B-it/` -- 9.6 GB full multimodal checkpoint from `google/gemma-4-E2B-it`
- `models/gemma-4-E2B-coreml/` -- 25 GB pre-converted CoreML bundle from
  `mlboydaisuke/gemma-4-E2B-coreml` (chunk1/2/3.mlmodelc + monolith model.mlpackage +
  external embed/PLE/RoPE .bin/.npy files + vision.mlpackage + audio.mlmodelc)
- `vendor/coreml-llm/` -- clone of `github.com/john-rocky/CoreML-LLM` (MIT; reference
  conversion pipeline + Swift ChunkedEngine to port to Python for T-0609a)
- `vendor/anemll/env-anemll/` -- ANEMLL Python 3.9 venv with coremltools 9.0, torch 2.5,
  transformers 4.57.6 (cannot load `model_type: gemma4` on its own)
- `/tmp/crossfire_gemma4_scout.py` -- viability test script (not committed; logic will
  be reworked into T-0609a chunked harness)

## Verification

- `pytest`: 129 passed
- `ruff check .`: clean
- `ruff format --check .`: clean

## Immediate Next Work

Phase 6 (Hardware Bring-Up And Calibration), Gemma 4 family:
1. T-0609a: port CoreML-LLM `ChunkedEngine.swift` to Python for the M4 Max
   chunked inference harness -- unblocks correct-output Gemma 4 E2B on ANE
   (viability already proven at 22.5 tok/s monolith floor). Reference code in
   `vendor/coreml-llm/conversion/collect_eagle_hidden_states_w4a8.py`.
2. T-0610 (Rustane build) and T-0611 (anemll-flash-llama.cpp build, Metal on
   Mac / CUDA on PC) -- both local, fast, no downloads
3. Download Gemma 4 31B / 26B-A4B to both nodes (T-0607, T-0612). E2B already
   downloaded at `models/gemma-4-E2B-it/`. Scout-first still applies for T-0612
   (26B-A4B Flash-MoE sidecar extraction; 128-expert + 1-shared topology is
   not what the extractor was built for).
4. Record P0 single-node baselines on PC and Mac (T-0613, T-0614). Note:
   Gemma 4 31B at Q8_0 (~33 GB) does not fit RTX 5090 single-node; PC P0
   must run TQ4_1S (~23 GB) or skip to distributed.
5. Record P1 distributed baseline over WiFi at 8K/16K/32K (T-0617)
6. Lock reward normalization constants from P1 baseline (T-0618)
7. Policy calibrations P2-P6 (T-0619 through T-0625) -> C0-C7 matrix (T-0626)
   with C6 now Gemma 4 31B @ 256K ctx (was Qwen 2.5 72B).
8. Stretch: upstream Gemma 4 support PR to ANEMLL (T-0609b); non-blocking for
   Phase 6 deliverables.

## Known unknowns to resolve during Phase 6

- **Chunked harness correctness on M4 Max (T-0609a).** Monolith
  `model.mlpackage` loads and runs on ANE but outputs garbage because PLE
  weights are external -- correct inference requires orchestrating chunk1/2/3
  plus external embed/PLE/RoPE lookups. Port from CoreML-LLM's Swift
  ChunkedEngine. No architectural unknowns remain; it's an engineering task.
- **Flash-MoE sidecar extraction for Gemma 4 26B-A4B.** anemll-flash-llama.cpp
  was built around Qwen / Kimi topology. Gemma's 128-expert + 1-shared-expert
  layout may need extractor patches. T-0612 scout before full calibration.
- **ANEMLL Gemma 4 upstream PR (T-0609b, stretch).** Architectural deltas
  enumerated in tasks.md; non-blocking for Phase 6 if T-0609a lands.

Deferred (hardware):
- USB4 40 Gbps cable + Thunderbolt IP bridge + throughput baseline (T-0603/T-0604/T-0605).
  Active interconnect is WiFi; composed TriAttention + TurboQuant compression is the
  bandwidth-hiding strategy. Revisit once the cable is acquired.
