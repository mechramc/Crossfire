# CROSSFIRE-X Status

Last updated: 2026-04-23
Branch: main
Tracker state: software-layer tasks closed; Phase 6 calibration and model-prep work remain. T-0601 (PC), T-0602 (Mac), and T-0606 (active WiFi discovery path) are done; EXO PC is cluster Master, Mac is Worker. T-0607.mac and T-0607.pc are both done — Mac has fp16 safetensors + Q8_0 GGUF, PC has Config-I (TQ4_1S/Q4_K/Q8_0 mixed) GGUF and a working `tqp-v0.1.0` CUDA toolchain at `~/llama-cpp-v010/`. PC smoke (Session 23): Gemma 4 31B Config-I on RTX 5090 measured 95.7 tok/s prefill / 37.7 tok/s decode, 30,064 MiB VRAM, 61/61 layers offloaded. T-0609a Gemma 4 E2B chunked CoreML engine DONE: `src/crossfire/ane/gemma4_chunked.py` loads 3 stateful chunks (MLState API), generates coherent text ("Paris" for "The capital of France is"), measured 42.98 tok/s decode / 138.9 ms TTFT on M4 Max ANE. T-0610 Rustane and the Mac half of T-0611 are already complete. Project venv migrated to Python 3.13.12 (coremltools 9.0 has no working native wheel for 3.14). WiFi is the active interconnect; T-0603/T-0604/T-0605 remain optional future TB4/USB4 work if WiFi throughput proves insufficient, but they are not current blockers.

## Summary

Repository docs, code, configs, scripts, and tests are aligned with
`crossfire_x_final.docx`. The AutoPilot orchestrator now loads from
`configs/autopilot.yaml`, routes selections into `PipelineConfig`, and feeds
outcomes back into the reward + bandit layer. The anemll-flash-llama.cpp
build is automated by `scripts/build_flash_moe.sh`. The only `.docx` spec in
the repo root is `crossfire_x_final.docx`; the prior unified spec is archived.
There is still a naming mismatch to keep visible: planning docs and user-facing
docs use `CROSSFIRE-X`, while some code/history still refer to `CROSSFIRE v2`.

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
- `src/crossfire/flashmoe/runtime.py` -- wraps the vendored Flash-MoE sidecar tool
  (`inspect` / `extract` / `verify`) and parses `llama-cli --perf` output into
  `FlashMoEStats` for smoke runs
- `scripts/run_flashmoe_scout.py` -- repo entrypoint for T-0612 scout work
  against Gemma 4 26B-A4B once the GGUF is present locally
- `models/gemma-4-26B-A4B-it/` -- local HF download now complete; both
  safetensor shards and tokenizer/config assets are present for Mac-side
  Flash-MoE extractor scouting

### Tests
- `tests/test_ane.py`, `tests/test_pipeline.py`, `tests/test_metrics.py`
- `tests/test_flashmoe.py`, `tests/test_triattention.py`, `tests/test_decision_tree.py`
- `tests/test_autopilot.py` -- classifier, bandit, reward, logger, orchestrator, YAML loader,
  pipeline integration (select → apply → record cycle)
- `tests/test_policy.py` -- P0-P6 availability filtering across every HardwareAvailability combo

## Not Started

- PC-side MoE model prep (T-0612.pc, vanilla TQ4_1S of Gemma 4 26B-A4B for
  RTX 5090 single-node baselines) and Mac-side MoE extraction work
  (T-0612 Flash-MoE sidecar). Repo-side T-0612 scout tooling is implemented
  and the Gemma 4 26B-A4B HF weights are downloaded locally on the Mac;
  remaining blockers are the actual Mac extractor validation run and the
  PC-side download + quantize for T-0612.pc.
- Calibration runs for every policy (T-0613 through T-0626)
- Orion Forge serving (Phase 7)
- Textual dashboard and final evaluation deliverables (Phase 8)
- Software follow-ups that are not on the critical path: persist bandit state (T-0412),
  warm-start from calibration data (T-0413), end-to-end AutoPilot integration test (T-0414)
- T-0609a follow-ups (none blocking Phase 6 calibration): batched prefill (a.1),
  prefix cache (a.2), speculative/verify (a.3), multimodal (a.4), top-k/p sampler (a.5)
- Stretch: upstream Gemma 4 support PR to ANEMLL (T-0609b)
- Optional future interconnect optimization: TB4/USB4 cable, Thunderbolt IP bridge,
  and throughput baseline (T-0603 through T-0605). These are no longer required
  for current Phase 6 execution because the active cluster path is WiFi, but
  they stay on the roadmap if WiFi throughput is not enough.

## Session 18 chunked engine artifacts

- `src/crossfire/ane/gemma4_chunked.py` — `Gemma4ChunkedEngine` (load/generate/predict_step/run_prefill/reset)
- `src/crossfire/ane/gemma4_assets.py` — `Gemma4Config`, tokenizer loader, `QuantizedEmbedding`, RoPE table loaders
- `src/crossfire/ane/gemma4_masks.py` — fp16 causal_mask_full / causal_mask_sliding / update_mask builders
- `tests/test_gemma4_chunked.py` — 38 tests (unit + real-bundle end-to-end)
- `scripts/run_gemma4_scout.py` — CLI; replaces `/tmp/crossfire_gemma4_scout.py`
- `pyproject.toml` — adds `[project.optional-dependencies].ane = [coremltools, numpy, tokenizers]`
- `configs/models.yaml` — adds `gemma-4-e2b.ane_config.coreml_bundle_path` + notes on
  effective context 512 and 40 tok/s target
- `.venv` migrated: Python 3.13.12 (coremltools 9.0 cp313 native wheel) — 3.14 has no
  working native wheel; documented in checkpoint Session 18

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

- `pytest`: clean for current batch in the Codex sandbox: 165 passed, 5 skipped.
  The skipped tests are the real-bundle `tests/test_gemma4_chunked.py` cases,
  which now detect sandbox-limited CoreML execution-plan availability and skip
  instead of failing. The same file still passes unsandboxed on this Mac
  (`38 passed` when run directly outside the sandbox).
- `ruff check .`: clean
- `ruff format --check .`: clean
- Scout CLI: not rerun in this session; last known good Session 18 run was
  `python scripts/run_gemma4_scout.py --prompt "The capital of France is"
  --max-tokens 24` -> generated " Paris." + continuation, reports TTFT 138.9 ms +
  decode 42.98 tok/s on M4 Max at `cpu_and_ne`

## Immediate Next Work

Phase 6 (Hardware Bring-Up And Calibration), Gemma 4 family:
1. Finish the remaining remote-node model prep: T-0612.pc (PC vanilla TQ4_1S of
   26B-A4B) and T-0612 (Mac Flash-MoE sidecar extraction). E2B already
   downloaded at `models/gemma-4-E2B-it/`; 31B is done on both nodes.
   Scout-first still applies for T-0612 (128-expert + 1-shared topology is
   not what the extractor was built for).
2. T-0613 (PC P0 single-node baseline) is now DONE Session 23: Gemma 4 31B
   Config-I on RTX 5090 measured 139.45 tok/s prefill / 42.76 tok/s decode
   with proper Gemma 4 chat template (`--jinja`); results JSON at
   `results/t0613_pc_p0_baseline.json`. Vs Mac C0: 2.2x prefill, 2.9x decode
   on a 37%-smaller model. Gemma 4 31B at Q8_0 (~33 GB) does not fit RTX
   5090 single-node; PC P0 runs the 19 GB Config-I GGUF (~30 GB on GPU
   including KV at ctx 4096). Consider re-running with a prompt set that
   elicits longer output for tighter decode confidence intervals before
   freezing C0 for the calibration matrix.
3. Record P1 distributed baseline over WiFi at 8K/16K/32K (T-0617)
4. Lock reward normalization constants from P1 baseline (T-0618)
5. Policy calibrations P2-P6 (T-0619 through T-0625) -> C0-C7 matrix (T-0626)
   with C6 now Gemma 4 31B @ 256K ctx (was Qwen 2.5 72B).
6. Stretch: upstream Gemma 4 support PR to ANEMLL (T-0609b); non-blocking for
   Phase 6 deliverables.
7. Sampler improvement (T-0609a.5) — current argmax decode drifts after ~4 tokens;
   plug in top-p + temperature once speculative-decode integration needs better draft quality.

## Known unknowns to resolve during Phase 6

- **Chunked harness correctness on M4 Max (T-0609a).** RESOLVED in Session 18:
  chunks are stateful via MLState API (simpler than Swift's manual IOSurface KV);
  effective context is 512 (not 2048 — determined from on-disk `causal_mask` input
  shape, config.json's 2048 is informational); engine generates coherent Gemma 4
  output at 42.98 tok/s decode. Remaining quality issue (drift after ~4 decode
  tokens) tracked as T-0609a.5 (sampler improvement).
- **Flash-MoE sidecar extraction for Gemma 4 26B-A4B.** anemll-flash-llama.cpp
  was built around Qwen / Kimi topology. Gemma's 128-expert + 1-shared-expert
  layout may need extractor patches. T-0612 scout before full calibration.
- **ANEMLL Gemma 4 upstream PR (T-0609b, stretch).** Architectural deltas
  enumerated in tasks.md; non-blocking for Phase 6 now that T-0609a is done.

Optional future interconnect work:
- TB4/USB4 40 Gbps cable + Thunderbolt IP bridge + throughput baseline (T-0603/T-0604/T-0605).
  Active interconnect is WiFi; composed TriAttention + TurboQuant compression is the
  bandwidth-hiding strategy. Revisit if WiFi throughput is not sufficient for the
  target workload.
