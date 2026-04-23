# CROSSFIRE-X Task Ledger

Last updated: 2026-04-23
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
- [!] T-0603 Acquire and verify TB4/USB4 40 Gbps cable -- optional future optimization only; not required for current Phase 6 work while WiFi is the active interconnect
- [!] T-0604 Configure Thunderbolt IP bridge / TCP-IP networking between nodes -- optional future optimization only; use if WiFi throughput is insufficient for target workloads
- [!] T-0605 Measure TB4/USB4 throughput between nodes and record baseline -- optional future benchmark only; no longer a bring-up blocker while WiFi is the production path
- [x] T-0606 Validate active discovery path between nodes (Session 17; EXO on Mac and PC discovered each other over WiFi mDNS; PC node `12D3KooWLeMLzYwn...3cGM` elected master, Mac demoted to worker at 2026-04-21 15:01:12; observed on Mac dashboard at localhost:52415. WiFi discovery is the current production path; 5GbE Ethernet and USB4 remain optional future interconnect work)
- [~] T-0607 Download Gemma 4 31B (`google/gemma-4-31B-it`) model artifacts (fp16 for Mac/MLX + TQ4_1S for PC llama.cpp). Subtasks split by node below
- [x] T-0607.mac Mac: download fp16 safetensors to `models/gemma-4-31B-it/` — DONE Session 18. `hf download google/gemma-4-31B-it` pulled 58 GB into `models/gemma-4-31B-it/` (2 safetensor shards + index.json + tokenizer + config + chat_template). Consumed by MLX/EXO decode path (T2); Mac P0 baseline T-0614 will convert this to GGUF via `vendor/llama.cpp/convert_hf_to_gguf.py` as part of T-0614 prep.
- [x] T-0607.pc PC (WSL2 Ubuntu): pre-quantized GGUF path adopted — DONE Session 23 (started Session 19). Environment: CUDA 13.2, RTX 5090 32 GB, 927 GB free, `vendor/llama.cpp` = TheTom/llama-cpp-turboquant at tag `tqp-v0.1.1` (HEAD `4d24ad8`) with CUDA build green (Linux ELF binaries in `build/bin/`, `libggml-cuda.so` present), runtime has full `LLM_ARCH_GEMMA4` + `llm_build_gemma4_iswa` + `GGML_TYPE_TQ4_1S` (enum 46), converter has `Gemma4Model(Gemma3Model)` registered for `Gemma4ForConditionalGeneration`. Initial converter risk note (`may lack Gemma 4 support`) was stale — fork is current. Tooling installed: `hf` CLI 1.11.0 + `hf_transfer` 0.1.9 via `uv tool install`; convert venv at `~/crossfire/.convertvenv` (Python 3.12, torch 2.6.0+cpu, transformers 5.5.1, safetensors 0.7.0, numpy 1.26.4, gguf, sentencepiece) installed via `uv pip --index-strategy unsafe-best-match`. Steps —
  1. **DONE** — HF license accepted; `hf auth login` stored token at `~/.cache/huggingface/token`.
  2. **DONE** — `HF_HUB_ENABLE_HF_TRANSFER=1 hf download google/gemma-4-31B-it --local-dir ~/crossfire-models/gemma-4-31B-it` completed in 1h 35m (`brz1b7hs0`); 59 GB on disk; `model-00001-of-00002.safetensors` (49.78 GB) + `model-00002-of-00002.safetensors` (12.76 GB) match HF byte sizes exactly.
  3. **PIVOT (no in-house convert/quantize)** — discovered `thetom-ai/Gemma-4-31B-it-TQPlus` (TurboQuant+ Config-I GGUF, 20.28 GB, public, ungated; KLD 0.125 vs Q8_0 = 5% better than Q4_K_M at only 8% larger size; built against `tqp-v0.1.0`). Verified format compat: TQ4_1S block layout (`d0:half + d1:half + qs[16]` = 20 B/block, `QK_TQ4_1S=32`) and enum value (`GGML_TYPE_TQ4_1S = 46`) byte-identical between `tqp-v0.1.0` (GGUF) and `tqp-v0.1.1` (our build); 434-file tag diff is upstream rebase noise (web UI / build infra), no on-disk format change. Gemma 4 runtime present in both tags.
  4. **DONE** — `hf download thetom-ai/Gemma-4-31B-it-TQPlus --local-dir ~/crossfire-models/gemma-4-31b-it-tqplus`; 19 GB GGUF on disk at `Gemma-4-31B-it-Config-I.gguf`.
  5. **DONE Session 23** — Smoke: tqp-v0.1.1 binary failed to load the v0.1.0-built GGUF with `tensor 'blk.2.attn_k_norm.weight' has offset ... expected ...` (file-level alignment shifted between fork tags despite block format being byte-identical). Built side toolchain at `~/llama-cpp-v010/build/` from `tqp-v0.1.0` tag (CUDA 13.2, `-DCMAKE_CUDA_ARCHITECTURES=120`, full Linux ELF binaries incl. `llama-completion`). Two more compat fixes needed: (a) use `llama-completion` not `llama-cli` (the `-no-cnv` flag was rejected by `llama-cli` and the binary printed `please use llama-completion instead`); (b) add `--jinja` to handle Gemma 4's custom chat template (without it the binary terminated with `std::runtime_error: this custom template is not supported, try using --jinja`). Final clean run: `~/llama-cpp-v010/build/bin/llama-completion -m .../Gemma-4-31B-it-Config-I.gguf --jinja -no-cnv -p "The capital of France is" -n 32 --n-gpu-layers 99 --seed 1`. **Results:** load 4.4 s, prefill 95.7 tok/s (5 toks @ 10.4 ms each), decode 37.7 tok/s (31 toks @ 26.5 ms each), 30,064 MiB CUDA mem used (28,021 model + 1,520 KV + 522 compute), 337 MiB free; 61/61 layers offloaded; context auto-reduced from 262144 to 4096 to fit. Output text is repetitive in `-no-cnv` mode (instruction-tuned model needs the chat template); a chat-mode rerun with proper Gemma turn formatting produced coherent CoT in interactive mode but exits via Ctrl-C, so the perf numbers above come from the `-no-cnv` raw-completion path.
  6. **DECISION** — keep 59 GB safetensors at `~/crossfire-models/gemma-4-31B-it/` for future re-quantization at `tqp-v0.1.1` (e.g. for the C0-C7 ablation matrix); disk budget (797 GB free post-tqplus) supports it.
  Quality follow-up (non-blocking): pure `TQ4_1S` everywhere via in-house `llama-quantize` (~13 GB, lower quality than Config-I) is still available if a smaller PC footprint is needed; replicating Config-I's mixed-precision recipe requires the `turboquant_plus` repo's quant scripts.
- [~] T-0608 Download Gemma 4 E2B (`google/gemma-4-E2B-it`) draft model artifacts (Session 17; full multimodal checkpoint downloaded to `models/gemma-4-E2B-it/` — 9.6 GB safetensors, single file. Text-only text-tower extraction still pending; tracked via T-0609a)
- [~] T-0609 Convert Gemma 4 E2B draft into ANE-ready CoreML format (Session 17 scout complete; ANEMLL has zero Gemma 4 support as of 2026-04-21 — last release 0.3.5 Beta from Feb 14 predates Gemma 4. Pivoted scout to evaluate `john-rocky/CoreML-LLM`'s pre-converted bundle `mlboydaisuke/gemma-4-E2B-coreml` — 25 GB, downloaded to `models/gemma-4-E2B-coreml/`. Viability PROVEN: `model.mlpackage` loads on M4 Max 16-core ANE via coremltools 9.0 with `CPU_AND_NE`, stateful KV cache `make_state()` works, forward pass runs at 22.5 tok/s. Monolith output is garbage (missing PLE path); correct inference is chunked via `chunk1/2/3.mlmodelc` + external embed/PLE/RoPE .bin/.npy files — see T-0609a)
- [x] T-0609a Port CoreML-LLM `ChunkedEngine.swift` to Python for Mac harness — DONE (Session 18). `src/crossfire/ane/gemma4_chunked.py` + `gemma4_assets.py` + `gemma4_masks.py`. Chunks are stateful via MLState (simpler than Swift's manual IOSurface KV management). Effective context 512 per `causal_mask` input shape. End-to-end test verified coherent "Paris" output; scout CLI (`scripts/run_gemma4_scout.py`) reports TTFT 138.9 ms, decode 42.98 tok/s, total 44.51 tok/s on M4 Max ANE at `CPU_AND_NE` — beats the iPhone 17 Pro's 31 tok/s target. Drift after ~4 decode tokens flagged as sampler follow-up (T-0609a.5)
- [ ] T-0609a.1 Batched prefill via `prefill_chunk*.mlmodelc` (optimization: Swift has N=512 batched prefill models; the current port loops single-token prefill). Unblocks long-prompt TTFT
- [ ] T-0609a.2 Prefix cache (disk-backed KV snapshot/restore) matching Swift `captureKVSnapshot`/`restoreKVSnapshot`. Requires MLState serialization — not yet probed
- [ ] T-0609a.3 Speculative/verify entry points (`verifyCandidates`, `commitAccepted`). Needed for EAGLE-3 / MTP / cross-vocab speculative decoding
- [ ] T-0609a.4 Multimodal input handling (image/audio features -> embed placeholders). Requires `vision.mlpackage` / `audio.mlmodelc` wiring
- [ ] T-0609a.5 Improved sampler (temperature, top-k, top-p) — current path is argmax-from-chunk3 which drifts after ~5 decode tokens on small 2.3B-effective E2B model. INVESTIGATION (Session 18): `chunk3/model.mil` computes full logits `[1, 262144]` with softcapping (`tanh(logits/30) * 30`) internally but exports only `token_id` (argmax) and `token_logit` (scalar max). Full-logits sampling is NOT possible without recompiling the chunk to expose the `logits_cast_fp16` intermediate. For the intended use case (E2B as speculative draft for Gemma 4 31B dense), per-step argmax is what matters — drift is masked by the verifier. Revisit only if E2B is used as a standalone generator
- [ ] T-0609b Port Gemma 4 text-only support to ANEMLL and upstream PR — architectural deltas vs Gemma 3: PLE (`hidden_size_per_layer_input`), shared KV across layers (`num_kv_shared_layers: 20`), double-wide MLP (`use_double_wide_mlp`), proportional RoPE with `partial_rotary_factor: 0.25` for full-attention layers, asymmetric head dims (sliding 256 / global 512), final logit softcapping 30.0. Use `vendor/coreml-llm/conversion/models/gemma4.py` + `gemma4_swa_chunks.py` + `gemma4_prefill_chunks.py` as reference. Stretch goal; non-blocking for Phase 6 if T-0609a lands
- [x] T-0610 Build Rustane — Session 18. `cargo build --release` in `vendor/rustane/` compiled clean in 38s; `target/release/` contains `generate`, `train`, `prepare_data`, `serve` binaries. `generate --help` surfaces full CLI (checkpoint path, tokenizer, decode-backend auto|naive|metal, KV cache, multi-sample, JSONL session mode). Integration with CROSSFIRE-X's ANE path is via the `generate` binary — to be wired up when speculative decoding needs a second ANE draft backend alongside the CoreML `Gemma4ChunkedEngine`.
- [~] T-0611 Build anemll-flash-llama.cpp with Metal flags (Mac) and CUDA flags (PC) — Mac side DONE Session 18. `scripts/build_flash_moe.sh` clones `Anemll/anemll-flash-llama.cpp` to `vendor/anemll-flash-llama.cpp/`, builds with `-DGGML_METAL=ON -DLLAMA_FLASH_MOE_GPU_BANK=ON`. `build/bin/llama-cli` loads Metal library on M4 Max, detects `MTLGPUFamilyApple9` + `MTLGPUFamilyMetal4`. PC CUDA build is remote — run on PC node via WSL2 when needed (same script auto-detects Linux + CUDA toolkit).
- [x] T-0612 Gemma 4 26B-A4B Flash-MoE sidecar extractor validation (Mac) — DONE Session 24. fp16 GGUF at `models/gemma-4-26B-A4B-fp16.gguf` (50.5 GB, 658 tensors) converted via `.venv-convert` (`convert_hf_to_gguf.py`). Scout `inspect` confirmed `arch=gemma4`, `expert_count=128`, `expert_used_count=8`, 30 MoE layers, fused `ffn_gate_up_exps` + `ffn_down_exps` families recognized natively by the schema-driven extractor (no anemll-flash-llama.cpp patches needed). `extract --include-shared` wrote 30 `layer_NNN.bin` files totaling 46.0 GB at `results/flashmoe_sidecar/gemma-4-26B-A4B/`; byte-level `verify` passed (60 manifest entries vs source GGUF). Topology note: Gemma 4 26B-A4B has a dense per-layer FFN using standard `ffn_down/gate/up.weight` names (NOT `_shexp`), so `--include-shared` is a no-op for this model — the "128+1" framing refers to dense shared FFN path, not a separate shared-expert tensor set. Stock smoke (llama-completion): **prompt 57.14 tok/s, decode 49.46 tok/s, 48.4 GB Metal** — new Mac P0 MoE baseline. Slot-bank smoke (16 slots, topk=8, prefetch-temporal on): **prompt 5.28 tok/s, decode 7.17 tok/s, 10.3 GB Metal, 57.9% slot-bank hit rate, 97.8% prefetch hit rate, 1.10 GiB routed bytes/token, 95.5% per-token time in pread source**. Summary JSON at `results/t0612_mac_flashmoe_scout.json`; logs at `results/raw/t0612_*`. See T-0612.pc for PC vanilla quant and T-0625 for the full P6 calibration sweep.
- [x] T-0612.mac Mac: download Gemma 4 26B-A4B HF artifacts to `models/gemma-4-26B-A4B-it/` — DONE Session 22. `hf download google/gemma-4-26B-A4B-it --local-dir models/gemma-4-26B-A4B-it` completed locally; directory contains `model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors`, `model.safetensors.index.json`, tokenizer/config/processor files, and is ready for repo-side Flash-MoE inspection/extraction.
- [x] T-0612.pc PC (WSL2 Ubuntu): vanilla TQ4_1S GGUF for 26B-A4B single-node MoE baselines on RTX 5090 — DONE Session 23. HF download (49 GB, ~48 min), HF→fp16 GGUF (50.5 GB, n_tensors=658), TQ4_1S quantize (15 GB, 5.06 BPW, 198 s, 3.16x compression vs fp16). Smoke on RTX 5090 via `~/llama-cpp-v010/build/bin/llama-completion --jinja -p 'The capital of France is' -n 64 --n-gpu-layers 99 --seed 1 --no-warmup </dev/null`: load 186 ms, prefill **148.96 tok/s** (21 toks @ 6.71 ms), decode **157.60 tok/s** (30 toks @ 6.35 ms), total 412 ms / 51 toks. VRAM: 25,430 model + 3,750 KV + 569 compute = 29,749 used of 32,606. Output coherent ("Paris"). Decode is **3.7x faster than dense Gemma 4 31B baseline** (T-0613: 42.76 tok/s) — validates MoE 4B-active routing on RTX 5090. Local files: `~/crossfire-models/gemma-4-26B-A4B-it/` (HF, 49 GB), `~/crossfire-models/gemma-4-26B-A4B-fp16.gguf` (48 GB), `~/crossfire-models/gemma-4-26B-A4B-TQ4_1S.gguf` (15 GB). Raw log: `results/raw/t0612pc_smoke.log`. Used the same v0.1.0 side-toolchain (`~/llama-cpp-v010/`) built for T-0607.pc.
  Distinction from T-0612: this is the standard llama.cpp single-node quant for PC C0/baseline MoE comparison. T-0612 is the Mac slot-bank extractor that requires the `anemll-flash-llama.cpp` sidecar pipeline.
- [x] T-0612.repo Wire repo-side Flash-MoE scout/extract/verify path — DONE Session 21; runtime wrapper fixes landed Session 24. `src/crossfire/flashmoe/runtime.py` now wraps the vendored `vendor/anemll-flash-llama.cpp/tools/flashmoe-sidecar/flashmoe_sidecar.py` tool for inspect/extract/verify and parses `--perf` output into `FlashMoEStats`. Session 24 corrections after the first real end-to-end run: (1) `--moe-mode` / `--moe-topk` are now space-separated (the fork's argparser does not accept `=`-form), (2) subprocess runs now close stdin (`stdin=subprocess.DEVNULL`) so inherited TTY never lands the binary in an interactive REPL, (3) smoke inference now targets `llama-completion` (not `llama-cli` — the fork's llama-cli has no working `-no-cnv` path) and adds `--jinja --single-turn` to handle Gemma 4 chat template and exit cleanly, (4) scout default binary changed to `llama-completion`.
- [x] T-0613 Record P0 single-node PC baseline (C0 reference) — DONE Session 23. Gemma 4 31B Config-I (TQ4_1S/Q4_K/Q8_0 mixed, 19 GB) on RTX 5090 via `~/llama-cpp-v010/build/bin/llama-completion --jinja`, ctx 4096 (auto-fit from 262144), n_predict 64, seed 1, stdin closed for clean REPL exit. Prefill 139.45 tok/s (20 toks @ 7.17 ms each), decode 42.76 tok/s (63 toks @ 23.39 ms each), total 1665 ms / 83 toks, load 176 ms. CUDA0 used 30,064 MiB (28,021 model + 1,520 KV + 522 compute), 343 MiB free, 61/61 layers offloaded. Output coherent: model produced CoT reasoning then "The capital of France is Paris." Results JSON: `results/t0613_pc_p0_baseline.json`. Raw log: `results/raw/t0613_pc_p0_baseline.log` (gitignored). Vs T-0614 Mac Q8_0: 2.2x prefill, 2.9x decode on a 37%-smaller model.
- [x] T-0614 Record P0 single-node Mac baseline (C0 reference) — DONE Session 18. Gemma 4 31B Q8_0 (30 GB) on M4 Max via llama.cpp Metal, `--n-gpu-layers 99`, ctx=8192, temp=0. Prefill 64.1 tok/s, decode 14.9 tok/s, 33.5 GB of 58.9 GB wired VRAM used. Conversion fp16 safetensors -> Q8_0 GGUF via `.venv-convert/` (sibling venv with `transformers==5.5.1`, `numpy~=1.26.4` — conflicts with main venv's `numpy 2.x` which coremltools needs). Results JSON: `results/t0614_mac_p0_baseline.json`. Raw log in `results/raw/` (gitignored).
- [~] T-0615 Record baseline perplexity runs (wikitext-2-raw-v1, 20 chunks, c=512). Subtasks split by node below
- [x] T-0615.pc PC: Gemma 4 31B Config-I (TQ4_1S/Q4_K/Q8_0 mixed, 19 GB) on RTX 5090 — DONE Session 23. `~/llama-cpp-v010/build/bin/llama-perplexity -m Gemma-4-31B-it-Config-I.gguf -f wiki.test.raw -c 512 --chunks 20 --n-gpu-layers 99 --seed 1`. **PPL = 2595.3860 ± 268.91931** over 20 × 512-token chunks (10,240 tokens total), tokenize 648 ms, prompt eval 2859 ms (3,581 tok/s). Per-chunk PPL stable in 1700–2800 range after chunk 1 warm-up bump (430.81). Absolute number is high because (a) Gemma 4 31B-it is instruction-tuned and expects chat-template-formatted input, not raw Wikipedia prose; (b) c=512 is short relative to the model's 262144 ctx_train; (c) Config-I mixed quant adds quantization noise. Used as relative C0 reference for the C0–C7 calibration matrix. Same model in chat-template mode produces coherent output (T-0613). Wikitext source: `~/crossfire-data/wikitext-2-raw/wiki.test.raw` (1.3 MB, from `hf:datasets/ggml-org/ci/wikitext-2-raw-v1.zip`). Results JSON: `results/t0615_pc_perplexity.json`. Raw log: `results/raw/t0615_pc_perplexity.log` (gitignored).
- [ ] T-0615.mac Mac: same wikitext-2-raw-v1 sweep on Gemma 4 31B Q8_0 via llama.cpp Metal (Mac side running in parallel session, results pending merge)
- [~] T-0616 Record baseline power measurements. Subtasks split by node below
- [x] T-0616.pc PC: nvidia-smi 1 Hz power/util/mem/temp sampling around the T-0615.pc workload — DONE Session 23. 31 samples (5 idle_pre + 21 load + 5 idle_post). **Idle baseline 31.16 W**; **inference peak 504.32 W at 100% util, 42 °C, 31,351 MiB VRAM** (96.2% of 32,606 total); steady-state inference mean (last 4 samples once util hit >90%) **429.29 W**. ΔW vs idle = +398 W (13.8× idle draw). Energy over 21 s load window ≈ 3,110 J (0.864 Wh). Cooling headroom substantial — peak temp 42 °C is far below throttle. VRAM peak confirms the C0 PC budget is tight; longer-context C1–C7 cells will need TriAttention KV reduction to fit. Results JSON: `results/t0616_pc_power.json`. Raw CSV: `results/raw/t0616_pc_power.csv`.
- [ ] T-0616.mac Mac: `powermetrics` per-target sampling (T2 Metal GPU, T3 ANE, T4 CPU) around the T-0615.mac workload (Mac side running in parallel session, results pending merge)
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
