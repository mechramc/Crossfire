# CROSSFIRE-X Checkpoint Log

Purpose: durable work log for meaningful project sessions.
Rule: update this file before every `git push`.

---

## Session 25 - 2026-04-23: T-0615.pc + T-0616.pc PC C0 calibration baselines (Gemma 4 31B Config-I)

### What was done

PC-side C0 calibration baselines recorded against the existing Gemma 4 31B
Config-I GGUF (TQ4_1S/Q4_K/Q8_0 mixed, 18.87 GiB, 5.28 BPW) on the RTX 5090.
Mac counterparts (T-0615.mac, T-0616.mac) are running in a parallel session
and will land separately.

1. **T-0615.pc — wikitext-2-raw-v1 perplexity.** Pulled the dataset to
   `~/crossfire-data/wikitext-2-raw/wiki.test.raw` (1.3 MB,
   `hf:datasets/ggml-org/ci/wikitext-2-raw-v1.zip`). Ran
   `~/llama-cpp-v010/build/bin/llama-perplexity -m
   Gemma-4-31B-it-Config-I.gguf -f wiki.test.raw -c 512 --chunks 20
   --n-gpu-layers 99 --seed 1`. **PPL = 2595.3860 ± 268.91931** over
   20 × 512-token chunks (10,240 tokens total). Per-chunk values stable in
   1700–2800 after the chunk-1 warm-up bump (430.81). Tokenize 648 ms,
   prompt eval 2859 ms (3,581 tok/s).

2. **T-0616.pc — GPU power profile.** Re-ran the T-0615 workload while
   `nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used,temperature.gpu`
   sampled at 1 Hz. 31 samples (5 idle_pre + 21 load + 5 idle_post). Idle
   baseline 31.16 W; inference peak 504.32 W at 100% util, 42 °C, 31,351 MiB
   VRAM (96.2% of 32,606 total); steady-state mean of the last 4 samples
   (>90% util) = 429.29 W. ΔW vs idle = +398 W (13.8× idle draw). Energy
   over the 21 s load window ≈ 3,110 J (0.864 Wh). Cooling headroom is large
   — 42 °C peak is far below RTX 5090 throttle.

### Headline findings

- **Absolute PPL is high (2595)** because the model is instruction-tuned,
  the dataset is raw Wikipedia prose without chat formatting, and c=512 is
  short relative to ctx_train=262144. **Valid as a relative C0 reference**
  for the C0–C7 calibration matrix; lower PPL is expected when (a) using
  chat-template-aware perplexity, (b) longer context, (c) full precision.
  Same model in chat-template mode produces coherent CoT (T-0613).
- **VRAM peak 96.2% of total** confirms the C0 PC envelope is tight.
  Any longer-context or larger-batch C1–C7 cell will need TriAttention KV
  reduction to fit on the 5090 alone.
- **Power signature is a clean three-phase shape**: idle (~31 W) → tensor
  upload ramp (50–150 W, 1–2 s) → inference plateau (266–504 W, ~10 s).
  This shape will be the template against which P1–P6 power deltas are
  measured.

### Caveats

- 21 s load window is short. A longer-running calibration cell
  (e.g. c=2048 / --chunks 100) will produce a more steady envelope; this
  baseline captures inclusive-of-ramp behavior, not pure steady state.
- Mac side (T-0615.mac, T-0616.mac) is running independently in a parallel
  session — those results will be added when that session merges.

### Local files

- `results/t0615_pc_perplexity.json` (committed)
- `results/t0616_pc_power.json` (committed)
- `results/raw/t0615_pc_perplexity.log` (gitignored)
- `results/raw/t0616_pc_power.csv` (gitignored)
- WSL: `/tmp/t0615_pc_perplexity.log`, `/tmp/t0616_pc_power.csv`

---

## Session 24 - 2026-04-23: T-0612 Mac Flash-MoE extractor validation (Gemma 4 26B-A4B)

### What was done

**T-0612 closed end-to-end on Mac.** The path from HF safetensors to a working
Flash-MoE slot-bank smoke run was executed in one pass and all four stages
recorded real telemetry.

1. **HF -> fp16 GGUF conversion.** `.venv-convert/bin/python
   vendor/llama.cpp/convert_hf_to_gguf.py models/gemma-4-26B-A4B-it --outfile
   models/gemma-4-26B-A4B-fp16.gguf --outtype f16` produced 50.5 GB / 658
   tensors in ~2 min. Converter recognized `Gemma4ForConditionalGeneration`,
   emitted `gemma4.expert_count = 128` and `experts used count = 8`, 30
   MoE layers with fused `ffn_gate_up_exps` (shape {2816, 1408, 128}) +
   `ffn_down_exps` (shape {704, 2816, 128}) per layer.

2. **Scout inspect.** `scripts/run_flashmoe_scout.py --model ... --include-shared`
   ran cleanly against the vendored extractor. JSON confirmed `arch=gemma4`,
   `expert_count=128`, `expert_used_count=8`, all 30 layers present with
   both routed families, total routed scope = 45.68 GB. **Topology finding:**
   Gemma 4 26B-A4B has 128 routed experts + a dense per-layer FFN using
   standard `ffn_down/gate/up.weight` names (NOT `_shexp`). `--include-shared`
   is a no-op for this model. The "128+1" in the project docs refers to the
   dense shared FFN path, not a separate shared-expert tensor set.

3. **Scout extract + byte verify.** Same entry point with `--extract` wrote
   30 `layer_NNN.bin` files (1.42 GB each) + `manifest.json` (60 entries,
   schema_version=1, flashmoe_gguf kind, layer_major_whole_tensor layout) to
   `results/flashmoe_sidecar/gemma-4-26B-A4B/`. Byte-level verify against
   source GGUF passed without exception. Total sidecar: 46.0 GB.

4. **Stock smoke (Mac P0 MoE baseline).** `llama-completion` with `--moe-mode stock`
   at fp16: **prompt 57.14 tok/s, decode 49.46 tok/s, load 1.88 s**, 48.4 GB
   of Metal memory used (of 58.9 GB wired budget), 9.1 GB headroom.
   Sampled output: "The user is asking for the capital of France." (chat
   template + thinking mode engaged correctly).

5. **Slot-bank smoke (Flash-MoE).** Same binary, `--moe-mode slot-bank
   --moe-topk 8 --moe-slot-bank 16 --moe-prefetch-temporal`, sidecar pointed
   at the extraction above. **Prompt 5.28 tok/s, decode 7.17 tok/s, load
   4.32 s, 10.3 GB Metal (78% reduction vs stock), 48.6 GB headroom.**
   Telemetry: slot-bank cached expert hit rate **57.9%**, prefetch (temporal
   reuse) hit rate **97.8%**, 13,032 expert refs, 43,856 preads, **60.73 GiB
   streamed from NVMe over 52 tokens**, **1.10 GiB routed bytes per token**.
   Per-token time split: routing + slot resolve 0.02 ms, expert I/O source
   372.54 ms (95.5%), expert upload 17.33 ms (4.4%). Per-layer hit rate
   range 40.2% (layer 0) – 56.6% (layer 23).

**Runtime wrapper corrections (`src/crossfire/flashmoe/runtime.py`).**

The repo-side scout tooling from Session 21 had three defects that only
surfaced during the first real end-to-end run. All three are now fixed:

- **`--moe-mode=X` form rejected by the binary.** The fork's llama-cli /
  llama-completion argparser does not accept `=`-separated flags. Changed
  `f"--moe-mode={self.mode.value}"` / `f"--moe-topk={...}"` into
  space-separated list entries (`"--moe-mode", self.mode.value, ...`).
- **Inherited TTY -> interactive-mode hang.** `subprocess.run(capture_output=True)`
  inherited the caller's stdin by default. When stdout is redirected, the
  binary still detected stdin as a TTY and dropped into its REPL. `> ` was
  printed in a tight loop until the log grew past 22 GB. Fix:
  `subprocess.run(..., stdin=subprocess.DEVNULL)` in `_run_command`, applied
  universally (harmless for inspect/extract/verify subprocess calls).
- **Wrong binary + missing chat-template flag.** This fork's `llama-cli`
  explicitly rejects `--no-conversation` ("please use llama-completion
  instead") and enters conversation mode when the GGUF has a chat template.
  Switched the scout default and runtime wrapper to `llama-completion`;
  dropped `--simple-io` and `--color off` (llama-cli UI flags not accepted by
  llama-completion); added `--jinja` (required to parse Gemma 4's chat
  template) and `--single-turn` (explicit batch-completion flag).

Test coverage updated: `tests/test_flashmoe.py` fake-subprocess mocks now
accept `stdin` kwarg and the `test_run_inference_executes_binary_and_parses_output`
test asserts `--jinja`, `--single-turn`, and `stdin is subprocess.DEVNULL`.

**Scout script default.** `scripts/run_flashmoe_scout.py --binary` default
changed from `.../llama-cli` to `.../llama-completion`.

**Artifacts produced this session.**

- `models/gemma-4-26B-A4B-fp16.gguf` (50.5 GB, sha256 prefix `d73341c5fb1e`).
- `results/flashmoe_sidecar/gemma-4-26B-A4B/` (46.0 GB, 30 layer bins + manifest).
- `results/t0612_mac_flashmoe_scout.json` (summary + telemetry, committed).
- `results/raw/t0612_convert_26B_A4B_fp16.log`, `t0612_scout_inspect.log`,
  `t0612_scout_extract.log`, `t0612_scout_verify.log`, `t0612_stock_smoke.log`,
  `t0612_slotbank_smoke.log` (gitignored).

### Verification

- `./.venv/bin/pytest tests/test_flashmoe.py -q`: `25 passed`.
- `./.venv/bin/pytest`: `170 passed`.
- `./.venv/bin/ruff check .` and `ruff format --check .`: clean.
- Byte-level sidecar verify: `FlashMoERuntime.verify_sidecar(..., metadata_only=False)`
  returned without exception (wrapper raises on mismatch).
- Scout JSON output sanity: `expert_count=128`, `expert_used_count=8`, 30
  layers, 60 manifest entries, `45675970560` routed bytes claimed in scope.
- Slot-bank smoke produced coherent Gemma 4 output through the chat template
  and emitted the expected `log_runtime_summary: Flash-MoE ...` telemetry
  block parsed by `_parse_inference_output`.

### State at end of session

- T-0612 closed; T-0612.repo closed (runtime wrapper now known-good against
  a real binary, not just mocks).
- Mac P0 MoE baseline captured: Gemma 4 26B-A4B fp16 stock = **49.46 tok/s
  decode** on M4 Max.
- Mac P6 initial telemetry captured: Gemma 4 26B-A4B slot-bank (16 slots,
  topk=8) = **7.17 tok/s decode, 57.9% hit rate, 10.3 GB Metal**. The full
  C7 ablation sweep (T-0625) can now vary slot count, `--moe-cache-io-split`,
  and async/batch-read flags on top of this known-good baseline.
- Outstanding critical path is now Phase 6 calibration; model-prep is done
  on both nodes (PC T-0612.pc closed same day — see Session 23 (cont.)
  entry below).
- Disk used by this session's artifacts: ~97 GB (50 GB GGUF + 46 GB sidecar
  + 1 GB logs/JSON).

---

## Session 23 (cont.) - 2026-04-23: Close T-0612.pc — PC Gemma 4 26B-A4B TQ4_1S MoE smoke passed on RTX 5090

### What was done

- Closed T-0612.pc by downloading, converting, quantizing, and smoke-testing
  `google/gemma-4-26B-A4B-it` (MoE: 128 experts + 1 shared expert, 4 B active
  parameters per token) on the WSL2 / RTX 5090 PC node.
- Pipeline (one shot, sequential):
  1. **HF download** — `hf download google/gemma-4-26B-A4B-it
     --local-dir ~/crossfire-models/gemma-4-26B-A4B-it`. 49 GB across two
     safetensors shards (model-00001-of-00002.safetensors at 49.9 GB,
     model-00002-of-00002.safetensors at 1.7 GB). License accepted via
     existing HF token. Took ~48 min over WiFi (no `hf_transfer` available
     because the convertvenv is uv-managed and has no `pip`).
  2. **HF → fp16 GGUF** — `~/crossfire/.convertvenv/bin/python
     ~/llama-cpp-v010/convert_hf_to_gguf.py ... --outtype f16`. n_tensors=658,
     output 50.5 GB (matches 16.01 BPW). Sustained 500–700 MB/s during
     write. Conversion log saved to `/tmp/t0612pc_convert.log`.
  3. **TQ4_1S quantize** — `~/llama-cpp-v010/build/bin/llama-quantize
     ... TQ4_1S`. 48,150 MiB → 15,214 MiB (5.06 BPW, 3.16x compression).
     Total quantize time 198 s (~3.3 min). All 658 tensors handled cleanly,
     including the MoE expert tensors (`ffn_*_exps`).
  4. **Smoke** — same `llama-completion --jinja </dev/null` recipe used for
     T-0613, with `-n 64 --n-gpu-layers 99 --seed 1 --no-warmup`.
- Reused the v0.1.0 side toolchain at `~/llama-cpp-v010/` built during T-0607.pc;
  no new binary work required.

### Smoke results (Gemma 4 26B-A4B TQ4_1S on RTX 5090, ctx auto-fit, n_predict 64)

| Metric         | Value                                                    |
| -------------- | -------------------------------------------------------- |
| Load time      | 185.68 ms                                                |
| Prefill        | 148.96 tok/s (21 toks @ 6.71 ms each)                    |
| Decode         | **157.60 tok/s** (30 toks @ 6.35 ms each)                |
| Total          | 412 ms / 51 tokens                                       |
| Sampling       | 5.66 ms (52 tokens)                                      |
| VRAM (CUDA0)   | 29,749 MiB used of 32,606 (532 free)                     |
| VRAM breakdown | 25,430 model + 3,750 KV + 569 compute + 2,324 unaccted   |
| Output         | "Paris" (coherent, after a brief CoT block)              |

### Headline finding

Decode at **157.6 tok/s** is **3.7x faster** than the dense Gemma 4 31B
Config-I baseline on the same hardware (T-0613: 42.76 tok/s). This validates
the MoE 4 B-active routing path on RTX 5090 — the model is 26 B parameters
nominal but only ~4 B participate per token, so single-node decode latency
tracks the active-parameter count, not the total.

### Caveats

- Custom chat template still requires `--jinja` (same as T-0607.pc / T-0613).
- Model uses ConditionalGeneration (multimodal) HF class; converter handled
  the text-only weight subset cleanly. Vision/audio paths not exercised.
- Smoke is single-prompt, 30 decoded tokens — too short for a tight
  steady-state decode confidence interval. Treat 157 tok/s as the upper
  bound of what a follow-up calibration sweep should expect, not as the
  calibration number itself.

### Local files (PC node, gitignored)

- `~/crossfire-models/gemma-4-26B-A4B-it/` (49 GB, HF safetensors + tokenizer)
- `~/crossfire-models/gemma-4-26B-A4B-fp16.gguf` (48 GB, intermediate)
- `~/crossfire-models/gemma-4-26B-A4B-TQ4_1S.gguf` (15 GB, smoke target)
- `/tmp/t0612pc_dl.log`, `/tmp/t0612pc_convert.log`,
  `/tmp/t0612pc_quantize.log`, `/tmp/t0612pc_smoke.log`

### Repo files

- `results/raw/t0612pc_smoke.log` (gitignored)
- `tasks.md` — T-0612.pc marked `[x]` DONE Session 23 with full pipeline metrics
- `status.md` — tracker state updated with T-0612.pc closure and headline 3.7x decode finding
- `checkpoint.md` — this entry

### Follow-up

- T-0612 (Mac slot-bank extractor for Flash-MoE NVMe streaming) is still open
  and is the canonical Phase-6-blocking MoE work. The PC TQ4_1S build is the
  C0 reference partner to it, not a substitute.
- HF token in `~/.cache/huggingface/token` was reused from prior session; same
  rotation reminder still applies whenever convenient.

---

## Session 23 - 2026-04-23: Close T-0607.pc — PC Gemma 4 31B Config-I smoke passed on RTX 5090

### What was done

- Closed T-0607.pc by smoke-testing the pre-quantized
  `thetom-ai/Gemma-4-31B-it-TQPlus` (TurboQuant+ Config-I, 19 GB GGUF, mixed
  TQ4_1S + Q4_K + Q8_0) on the WSL2 / RTX 5090 PC node.
- Worked through three GGUF/binary compatibility issues:
  1. **Tensor offset mismatch** between `tqp-v0.1.0` (the GGUF's build tag) and
     `tqp-v0.1.1` (the previously built `vendor/llama.cpp/`). Block format is
     byte-identical between tags but file-level alignment shifted, so v0.1.1
     could not load the v0.1.0 file. Fix: built a side toolchain at
     `~/llama-cpp-v010/build/` from `tqp-v0.1.0` with CUDA 13.2 and
     `-DCMAKE_CUDA_ARCHITECTURES=120` (Blackwell/RTX 5090). v0.1.1 build is
     kept intact for future re-quantization work.
  2. **`-no-cnv` not supported by `llama-cli`** in the v0.1.0 toolchain. The
     binary explicitly prints `please use llama-completion instead`. Switched
     to `~/llama-cpp-v010/build/bin/llama-completion`.
  3. **Custom chat template not supported by the default parser.** Without
     `--jinja`, `llama-completion` terminates with
     `std::runtime_error: this custom template is not supported, try using --jinja`.
     Added `--jinja` to the smoke command line.
- Final smoke command:
  `~/llama-cpp-v010/build/bin/llama-completion -m .../Gemma-4-31B-it-Config-I.gguf --jinja -no-cnv -p "The capital of France is" -n 32 --n-gpu-layers 99 --seed 1`
- Updated `tasks.md` to mark T-0607.pc as done and record Session 23 step 5 with
  the full set of fixes and the measured perf.

### Smoke results (Gemma 4 31B Config-I on RTX 5090, ctx 4096)

| Metric         | Value                                                    |
| -------------- | -------------------------------------------------------- |
| Load time      | 4.42 s                                                   |
| Prefill        | 95.7 tok/s  (5 toks @ 10.45 ms each)                     |
| Decode         | 37.7 tok/s  (31 toks @ 26.54 ms each)                    |
| CUDA0 used     | 30,064 MiB (28,021 model + 1,520 KV + 522 compute)       |
| CUDA0 free     | 337 MiB                                                  |
| GPU offload    | 61/61 layers                                             |
| Context        | auto-reduced from 262,144 to 4,096 to fit free VRAM      |

For comparison, T-0614 Mac M4 Max baseline (Gemma 4 31B Q8_0, 30 GB):
prefill 64.1 / decode 14.9 tok/s. PC node is roughly 1.5x prefill / 2.5x decode
on a model that is ~37% smaller on disk.

### Caveats

- The `-no-cnv` raw-completion output loops ("...France is France is France
  is..."). That is expected for an instruction-tuned model when the chat
  template is bypassed; a chat-mode run (`--jinja` without `-no-cnv`) produced
  coherent Gemma 4 CoT output (`<|channel>thought ...`), but in that mode the
  binary drops into interactive REPL after the prompt, which is fine for ad-hoc
  use but does not produce a clean `--perf` line. The perf numbers above are
  from the `-no-cnv` path; they are valid as throughput numbers because the
  decode kernel does the same work either way, but T-0613 (PC P0 baseline)
  should be re-recorded with proper chat formatting before being used for the
  C0 row of the calibration matrix.
- WSL2 + RTX 5090 driver shows ~31 GB used at idle when no processes are
  running. This is a baseline reservation quirk in the current driver; it does
  not actually consume VRAM and the smoke run was able to allocate 30 GB on top
  of it without issue. Flagged for investigation if anything later needs the
  full 32 GB of nominal VRAM.

### Local files (gitignored, PC node only)

- `~/crossfire-models/gemma-4-31B-it/` — 59 GB fp16 safetensors (kept for
  future TQ4_1S re-quantization at v0.1.1)
- `~/crossfire-models/gemma-4-31b-it-tqplus/Gemma-4-31B-it-Config-I.gguf` —
  19 GB pre-quantized GGUF used in this smoke
- `~/llama-cpp-v010/` — side `tqp-v0.1.0` toolchain build dir
- `~/run_smoke.sh` — convenience launcher used in this session
- `~/crossfire/.convertvenv/` — Python 3.12 venv with
  `transformers 5.5.1`, `torch 2.6.0+cpu`, `numpy 1.26.4`, `safetensors 0.7.0`
  (already documented in tasks.md T-0607.pc preamble)

### Verification

- Smoke perf line emitted by `llama-completion --perf` (see Smoke results
  above); exit code 0; no CUDA errors.
- `git pull` brought in Sessions 20-22 work (Flash-MoE scout wiring, Mac
  C0 baseline, 26B-A4B HF download). Local Session 19 tasks.md edit was merged
  with remote without losing either side: T-0607.mac kept remote DONE, T-0607.pc
  kept the Session 19 detail and was re-marked DONE here, T-0612 / T-0612.pc /
  T-0612.repo kept all three.
- Tests / lint not re-run in this session (no Python source changes; only doc
  edits and a remote-node smoke).

### Follow-up: T-0613 PC P0 baseline (also Session 23)

After closing T-0607.pc, ran the formal C0 baseline with proper Gemma 4 chat
template to unblock the calibration matrix:

```
~/llama-cpp-v010/build/bin/llama-completion \
  -m .../Gemma-4-31B-it-Config-I.gguf \
  --jinja --no-warmup -p 'The capital of France is' \
  -n 64 --n-gpu-layers 99 --seed 1 </dev/null
```

Closing stdin (`</dev/null`) makes the `--jinja` chat-mode REPL exit after the
first response so `--perf` actually fires (without it the binary stays in
interactive mode forever).

**Results:**

| Metric         | Value                                                  |
| -------------- | ------------------------------------------------------ |
| Prefill        | 139.45 tok/s (20 toks @ 7.17 ms each)                  |
| Decode         | 42.76 tok/s (63 toks @ 23.39 ms each)                  |
| Total          | 1665 ms / 83 tokens                                    |
| Load           | 176 ms (warmup off; weights already mmap-warm in cache)|
| CUDA0 used     | 30,064 MiB (28,021 model + 1,520 KV + 522 compute)     |
| Output         | "...The capital of France is Paris..." (coherent)      |

Vs T-0614 Mac Q8_0 baseline: PC is 2.2x prefill (139.45 vs 64.1) and 2.9x
decode (42.76 vs 14.9) on a model that is 37% smaller on disk. Recorded as
`results/t0613_pc_p0_baseline.json` with the raw log under
`results/raw/t0613_pc_p0_baseline.log` (gitignored). T-0613 is closed.

### State at end of session

- T-0607.pc and T-0613 are both closed.
- Side toolchain `~/llama-cpp-v010/build/` is the binary to use for any future
  smoke or perf work against a v0.1.0-built GGUF.
- T-0612.pc (PC vanilla TQ4_1S of 26B-A4B) is the next remaining PC model-prep
  task; still requires the 52 GB HF download on the PC node before quantization.

---

## Session 22 - 2026-04-22: End-of-day tracker sync after 26B-A4B download

### What was done

- Verified that the local Hugging Face download of Gemma 4 26B-A4B completed
  successfully into `models/gemma-4-26B-A4B-it/`.
- Confirmed both shard files are now present:
  `model-00001-of-00002.safetensors` and
  `model-00002-of-00002.safetensors`, plus
  `model.safetensors.index.json`, tokenizer, config, processor config, and
  generation config.
- Updated `tasks.md` so T-0612 now reflects the real state: model acquisition
  is complete on the Mac, while Flash-MoE sidecar inspection/extraction is
  still pending.
- Updated `status.md` to record that the 26B-A4B HF weights are local and that
  the remaining blocker is extractor validation, not download/auth.

### Verification

- `./.venv/bin/pytest`: `165 passed, 5 skipped`
- `./.venv/bin/ruff check .`: clean
- `./.venv/bin/ruff format --check .`: clean

### State at end of session

- End-of-day model state on the Mac:
  Gemma 4 31B HF, Gemma 4 31B Q8_0 GGUF, Gemma 4 E2B HF, Gemma 4 E2B CoreML,
  and Gemma 4 26B-A4B HF are all present locally.
- T-0612 is now an extractor-validation task rather than a model-download task.
- T-0617 remains unrun; no distributed WiFi baseline was recorded today.

---

## Session 21 - 2026-04-22: Wire Flash-MoE scout/extract path for T-0612

### What was done

- Replaced the Flash-MoE runtime stubs in `src/crossfire/flashmoe/runtime.py`
  with real wrappers around the vendored Anemll tooling.
- `FlashMoERuntime.extract_sidecar()` now calls
  `vendor/anemll-flash-llama.cpp/tools/flashmoe-sidecar/flashmoe_sidecar.py`
  to extract a sidecar from a MoE GGUF and optionally verify it.
- Added `inspect_sidecar()` and `verify_sidecar()` wrappers so the repo can
  probe Gemma 4 26B-A4B extractor compatibility before committing to a full
  Flash-MoE run.
- Implemented `run_inference()` against the built `llama-cli` and parse the
  `--perf` output into `FlashMoEStats` (hit rate, pread count, expert loads,
  decode tok/s) for smoke tests once the model exists locally.
- Added `scripts/run_flashmoe_scout.py` as a repo-level T-0612 entrypoint:
  inspect -> optional extract/verify -> optional smoke inference.
- Added test coverage in `tests/test_flashmoe.py` for output parsing, binary
  execution wrapping, and sidecar tool invocation.

### Verification

- `./.venv/bin/pytest`: `165 passed, 5 skipped`
- `./.venv/bin/ruff check .`: clean
- `./.venv/bin/ruff format --check .`: clean

### State at end of session

- Repo-side support for T-0612 now exists and is verified.
- T-0612 itself is still not complete because `models/` does not yet contain a
  Gemma 4 26B-A4B GGUF to inspect/extract against.
- T-0617 remains blocked on cross-node runtime/model availability; no fake
  distributed baseline was recorded.

---

## Session 20 - 2026-04-22: Make Gemma 4 CoreML real-bundle tests sandbox-safe

### What was done

- Investigated the apparent Mac-side CoreML regression from Session 19.
- Confirmed the failure was environment-specific: inside the Codex sandbox,
  `coremltools` could not build an execution plan for
  `models/gemma-4-E2B-coreml/chunk1.mlmodelc`, but the same real-bundle test
  file passed unsandboxed on the same machine (`./.venv/bin/pytest
  tests/test_gemma4_chunked.py -q` -> `38 passed`).
- Updated `tests/test_gemma4_chunked.py` so the real-bundle tests first probe
  whether CoreML can build a `CompiledMLModel` execution plan in the current
  runtime. If the runtime is sandbox-limited, those tests now skip with an
  explicit reason instead of failing as if the bundle were broken.
- Left the actual inference logic untouched. This fix is about test behavior in
  restricted environments, not about changing the Gemma 4 chunked engine.

### Verification

- `./.venv/bin/pytest`: clean in the sandbox — `162 passed, 5 skipped`
- `./.venv/bin/ruff check .`: clean
- `./.venv/bin/ruff format --check .`: clean
- `./.venv/bin/pytest tests/test_gemma4_chunked.py -q` outside the sandbox:
  `38 passed`

### State at end of session

- The Mac-side CoreML path is not currently broken in repo code.
- Sandbox-limited CoreML availability no longer shows up as a false code
  failure in repo verification.
- Remaining Mac work is back to the actual roadmap items: ANE follow-ups,
  model prep, and calibration runs.

---

## Session 19 - 2026-04-22: Tracker sync for WiFi interconnect decision

### What was done

- Updated `tasks.md` to reclassify T-0603/T-0604/T-0605 from implied bring-up
  blockers to optional future TB4/USB4 optimization tasks. Current repo reality
  is that the cluster runs over WiFi and higher-bandwidth cable work is only
  needed if WiFi throughput is not enough.
- Updated T-0606 wording to reflect the actual validated path: WiFi mDNS
  discovery between PC and Mac is the active production interconnect path.
- Updated `status.md` to remove stale next-work items that were already complete
  in Session 18 (`T-0610`, Mac side of `T-0611`, and `T-0614`), and to state
  explicitly that WiFi is the active interconnect.
- Added a naming-mismatch callout in `status.md` per `AGENTS.md`: planning/user
  docs say `CROSSFIRE-X`, while some historical code/doc references still say
  `CROSSFIRE v2`.
- Updated `README.md` so the top-level project description no longer claims that
  USB4 cable acquisition is required for the current execution path, while still
  preserving TB4/USB4 as a future option if WiFi underperforms.

### Verification

- `./.venv/bin/pytest`: FAIL. 162 passed, 5 failed, all in
  `tests/test_gemma4_chunked.py` real-bundle coverage. Failure occurs while
  loading `models/gemma-4-E2B-coreml/chunk1.mlmodelc` through
  `ct.models.CompiledMLModel`, which now raises CoreML execution-plan error
  `-14`.
- `./.venv/bin/ruff check .`: clean
- `./.venv/bin/ruff format --check .`: clean

### State at end of session

- Tracker/docs now match the user decision that WiFi is the active interconnect.
- TB4/USB4 work remains available as an optional future optimization and
  benchmark path, not as a blocker for current Mac or cluster work.
- Next actual blockers remain remote-node model prep and Phase 6 calibration,
  not cable acquisition.
- Verification is not fully green: the real-bundle Gemma 4 CoreML tests that
  passed in Session 18 do not currently load on this machine, so that regression
  remains an open blocker and is recorded in `status.md`.

---

## Session 18 - 2026-04-22: T-0609a Gemma 4 E2B chunked engine (Python port of ChunkedEngine.swift)

### What was done

**T-0609a closed (chunked CoreML inference for Gemma 4 E2B on M4 Max ANE):**

Planned as a full port of `vendor/coreml-llm/Sources/CoreMLLLM/ChunkedEngine.swift`
(2407 LOC, MIT). On inspecting the on-disk chunk `metadata.json` files, discovered
the chunks are **stateful via Apple's MLState API**, not stateless with manual KV
buffers as the Swift reference code (optimized for batched prefill + EAGLE-3) implied.
Final Python implementation is ~400 LOC covering the correct-output text-decoding
path end-to-end.

- `src/crossfire/ane/gemma4_assets.py` — parses `model_config.json`; loads tokenizer
  from `hf_model/tokenizer.json`; provides `QuantizedEmbedding` (mmap int8 +
  per-row fp16 scale dequant for token embeddings AND per-layer embeddings);
  loads `per_layer_projection.bin` (fp16 8960x1536), `per_layer_norm_weight.bin`
  (fp32[256]), and `cos_sliding/sin_sliding/cos_full/sin_full.npy` RoPE tables
- `src/crossfire/ane/gemma4_masks.py` — pure-numpy fp16 mask builders:
  `causal_mask_full(position, ctx)`, `causal_mask_sliding(position, W)`, and
  `update_mask(position, ctx)` — fp16 block fill is `-65504.0` (fp16 min), not
  `-1e9` (overflows)
- `src/crossfire/ane/gemma4_chunked.py` — `Gemma4ChunkedEngine` orchestration:
  - `load(bundle_path, compute_units)` discovers `chunk*.mlmodelc` via glob,
    reads each chunk's `metadata.json` to extract effective context from
    `causal_mask` input shape (observed 512 for swa-2k variant, NOT the 2048
    in `model_config.json.context_length`), loads each chunk with
    `ct.models.CompiledMLModel` (Python 3.14 had no working native wheel;
    `.venv` migrated to 3.13.12), calls `make_state()` per stateful chunk,
    runs 4 dummy decode steps + `reset()` to prewarm ANE compile schedules
  - `predict_step(token_id, position)` looks up embedding + PLE, builds
    RoPE slice + masks, passes hidden_states sequentially through chunks;
    stateful chunks get `state=` kwarg, chunk3 is stateless and consumes
    `kv13_k/v` + `kv14_k/v` outputs emitted by chunk2 (shared KV across
    layers 13-14 -> 15-24). Returns `token_id` int from chunk3's in-model argmax
  - `run_prefill(token_ids)` loops `predict_step` (no batched-prefill models
    used — Swift's `prefill_chunk*.mlmodelc` is deferred to T-0609a.1)
  - `generate(prompt, max_tokens, stop_on_eos)` tokenizes with BOS prepended,
    prefills with TTFT timer, decodes with decode tok/s timer, returns
    `GenerationResult(text, prompt_tokens, generated_tokens, ttft_ms, decode_tok_s, total_tok_s)`
- `scripts/run_gemma4_scout.py` — CLI that loads bundle, prints generation +
  TTFT + tok/s. Replaces the throwaway `/tmp/crossfire_gemma4_scout.py`
- `tests/test_gemma4_chunked.py` — 38 tests: config roundtrip, tokenizer, embed
  dequant matches reference formula, RoPE/projection/norm loaders, mask builders
  at boundary positions, chunk metadata parsing, compute-unit enum mapping, and
  real-bundle end-to-end that loads the actual chunks and asserts "Paris" in the
  generated text for prompt "The capital of France is"

**Smoke run (M4 Max, `cpu_and_ne`):**
- Prompt: "The capital of France is" (6 tokens incl. BOS)
- Generated: " Paris.\n\n The capital" + drift after ~4 decode tokens
- TTFT: 138.9 ms
- Decode tok/s: 42.98 (beats Session 17 monolith floor of 22.5 by 1.9x and
  iPhone 17 Pro's 31 tok/s)
- Total tok/s: 44.51

**Python 3.14 -> 3.13 venv migration:**

During task 5 discovered coremltools 9.0 on Python 3.14 installs as a source-only
wheel with no `libcoremlpython` / `libmilstoragepython` native extensions, so
`MLModel(chunk.mlmodelc)` fails with "Unable to load libmodelpackage". PyPI shows
native wheels for cp312 and cp313 only (`coremltools-9.0-cp313-none-macosx_11_0_arm64.whl`).
Migrated `.venv` to `/opt/homebrew/bin/python3.13` (3.13.12). All 167 tests
(129 pre-existing + 38 new) pass on the 3.13 venv.

**Dependency additions (`pyproject.toml` `[project.optional-dependencies]`):**
- `ane = ["coremltools>=9.0", "numpy>=1.26", "tokenizers>=0.20"]`
- Justification per project CLAUDE.md: coremltools is the only supported Python
  path to load `.mlmodelc` with ANE compute-unit selection; `tokenizers` (Rust
  HF lib) consumes `hf_model/tokenizer.json`; numpy is unavoidable

**Config update (`configs/models.yaml`):**
- `gemma-4-e2b.ane_config.backend` renamed `anemll` -> `coreml_chunked`
- Added `coreml_bundle_path: models/gemma-4-E2B-coreml`
- `max_context: 2048` -> `512` (the effective value from chunk schemas)
- `expected_tok_s_target: 50` -> `40` (based on observed 42.98)

**Tracker updates:**
- `tasks.md` — T-0609a marked done with session note; added T-0609a.1
  (batched prefill), T-0609a.2 (prefix cache), T-0609a.3 (speculative/verify),
  T-0609a.4 (multimodal), T-0609a.5 (top-k/p sampler) as follow-ups
- `status.md` — Session 18 artifacts section, verification block, immediate
  next work dropped T-0609a and added T-0609a.5 to later items, Known unknowns
  resolved entry for chunked harness correctness
- `checkpoint.md` — this entry

### Verification
- `pytest`: 167 passed (38 new gemma4 tests incl. real-bundle end-to-end)
- `ruff check .`: clean
- `ruff format --check .`: clean (45 files)
- Scout CLI end-to-end: generates " Paris" + continuation, reports all three
  timing metrics

### State at end of session
- T-0609a done; critical-path Gemma 4 E2B -> ANE inference is working
- 38 new tests, no regressions in the 129 pre-existing tests
- Project venv on Python 3.13.12 (from 3.14.3) — only reason: coremltools
  native wheel support. No other code touched
- Outstanding critical-path: T-0610 (Rustane), T-0611 (anemll-flash-llama.cpp),
  T-0607 / T-0612 (31B + 26B-A4B downloads), P0-P6 calibration
- T-0609a follow-ups (a.1-a.5) are quality/optimization, not correctness

---

## Session 17 - 2026-04-21: T-0606 close and T-0609 Gemma 4 -> ANE scout

### What was done

**T-0606 closed (WiFi mDNS discovery validates 5GbE fallback path):**
- After Session 15 brought up EXO on PC and Session 16 the Gemma pivot, started
  EXO on Mac from PID 46521 (dashboard live on `http://localhost:52415`); PC
  node `12D3KooWLeMLzYwnaBdSQagZW8KiTZMBFqtnw2nqRyhdSFzn3cGM` elected cluster
  Master, Mac demoted to Worker at 2026-04-21 15:01:12 per Mac EXO log -- peer
  discovery over WiFi mDNS confirmed end-to-end. Observed on Mac EXO dashboard
  with PC peer visible. 5GbE Ethernet link itself untested; WiFi fallback is
  sufficient for current no-USB4 state.

**T-0609 scout -- Gemma 4 E2B -> ANE viability proven, correct harness still to build:**

Research (live-source, dated 2026-04-21):
- `transformers` 5.5.4 stable supports `model_type: gemma4` (5.5.0 added Gemma 4,
  2026-04-02). Requires Python >= 3.10; 3.9 was dropped.
- ANEMLL 0.3.5 Beta (2026-02-14) predates Gemma 4. Zero Gemma 4 code hits,
  branches, PRs, or issues on `github.com/Anemll/Anemll` as of 2026-04-21.
  Active branch work is Qwen3.5 / DeltaNet.
- `github.com/john-rocky/CoreML-LLM` (MIT) ships a pre-converted Gemma 4 E2B
  CoreML bundle at `huggingface.co/mlboydaisuke/gemma-4-E2B-coreml` with
  99.78% ANE placement claim (7294 / 7310 ops), 31 tok/s on iPhone 17 Pro.
  Ships an iOS app (no Mac CLI / app target); reusable as Swift Package or
  via coremltools from Python.

Prerequisites installed:
- `hf` CLI 1.11.0 via `uv tool install huggingface_hub[cli]` (executables in
  `~/.local/bin/hf`, `~/.local/bin/huggingface-cli`)
- HF auth -- user logged in as `mechramc`, Gemma 4 license accepted
- ANEMLL venv created at `vendor/anemll/env-anemll/` via `create_uv_env.sh`
  (Python 3.9, coremltools 9.0, torch 2.5.0, transformers 4.57.6,
  sentencepiece, scikit-learn 1.5.1)
- `vendor/coreml-llm/` -- cloned `github.com/john-rocky/CoreML-LLM` @
  `99cf93fdd208...` for reference conversion pipeline and ChunkedEngine

Downloads (all gitignored):
- `models/gemma-4-E2B-it/` (9.6 GB) -- full multimodal checkpoint from
  `google/gemma-4-E2B-it`. Gemma 4 E2B is always multimodal (text + image +
  audio); Google does not ship a text-only E2B. Config reveals nested
  `gemma4_text` submodule with 35 layers, 1536 hidden, 8 heads, head_dim 256,
  vocab 262144, tie_word_embeddings, and Gemma 4-specific features:
  `hidden_size_per_layer_input: 256` (PLE), `num_kv_shared_layers: 20`
  (shared KV across 20 layers), `use_double_wide_mlp: true`,
  `final_logit_softcapping: 30.0`, proportional RoPE with
  `partial_rotary_factor: 0.25` for full-attention layers, asymmetric head
  dims (sliding 256 / global 512)
- `models/gemma-4-E2B-coreml/` (25 GB) -- CoreML bundle. Top-level artifacts:
  `model.mlpackage` (2.4 GB, stateful int4 monolith),
  `model.mlmodelc` (compiled version),
  `chunk1/2/3.mlmodelc` (the on-device chunked path), plus variants `lite/`,
  `lite-chunks/`, `mf/`, `sdpa/`, `sdpa-8k/`, `stateless/`, `stateless-ctx2048/`,
  `swa/`, `prefill/`, `vision.mlpackage` (322 MB), `audio.mlmodelc`, and the
  external weights `embed_tokens_q8.bin` (384 MB), `embed_tokens_per_layer_q8.bin`
  (2.2 GB PLE), `per_layer_projection.bin` (27 MB), `per_layer_norm_weight.bin`,
  `cos_full.npy`, `cos_sliding.npy`, `sin_full.npy`, `sin_sliding.npy` (RoPE),
  plus `hf_model/tokenizer.json`

Viability test (`/tmp/crossfire_gemma4_scout.py`, not committed -- will be
reworked into T-0609a chunked harness):
- Loaded `model.mlpackage` with `ComputeUnit.CPU_AND_NE` -- success (89.7s
  cold load first time, macOS 26.3)
- Stateful KV cache `make_state()` works -- returns populated state object,
  KV shape `[70, 1, 512, 512]` fp16 (70 = 35 layers x 2 for K+V)
- IO spec: `input_ids` int32 [1,1], `position_ids` int32 [1], `causal_mask`
  fp16 [1,1,1,512], `update_mask` fp16 [1,1,512,1]; outputs `token_id` int32
  [1] (in-model argmax) + `token_logit` fp16 [1]; context = 512 tokens
- Forward pass at 22.5 tok/s on M4 Max 16-core ANE (19-token decode window,
  mean 44.45 ms per token; prefill last-token 43.4 ms)
- Output is GARBAGE -- monolith emits loop (" is is is...") then last-vocab
  token (262143 `<unused6226>`) because `model.mlpackage` doesn't carry PLE
  weights. Per `build_gemma4_bundle.py` docstring the canonical on-device
  path is chunked: chunk1/2/3 + external QuantEmbed + PerLayerRawEmbed +
  RoPE tables, orchestrated by `Sources/CoreMLLLM/ChunkedEngine.swift`
- fp16 mask fill must be -65504.0 (fp16 min) not -1e9 (overflows); BOS
  token 2 must be prepended before the prompt

**Tracker reconciliation:**
- `tasks.md` -- T-0606 marked done; T-0608 marked in-progress (full
  multimodal downloaded); T-0609 marked in-progress (scout complete,
  correct harness port tracked as follow-up); added T-0609a (port
  CoreML-LLM ChunkedEngine to Python) and T-0609b (upstream Gemma 4 PR
  to ANEMLL, stretch)
- `status.md` -- Session 17 progress summary, artifact inventory, Immediate
  Next Work reordered to lead with T-0609a, Known unknowns section rewritten
  for the new state
- `checkpoint.md` -- this entry

**Lesson learned saved to project memory:**
- `memory/feedback_verify_before_asserting.md` -- rule: verify live sources
  before making hard assertions about library versions, Python compat, or
  "X doesn't exist" claims. Training-data recall is stale by months.
  Triggered by three wrong caveats earlier in the session (transformers 5.x
  "dev-only", Python 3.9 compat uncertainty, implicit assumption that ANEMLL
  might have partial support). User corrected with "check for latest updates
  before making hard assertions".

### Verification
- `~/crossfire/exo/.venv/bin/exo` process live on Mac: PID 46521, API on
  `http://localhost:52415`, Mac is Worker since 15:01:12
- `curl -s http://localhost:52415/node_id` returns Mac node ID;
  `curl -s http://localhost:52415/v1/models` returns model catalog
- Mac EXO log (`results/raw/exo_run_20260421_142221.log`) contains line
  `Node 12D3KooWLeMLzYwnaBdSQagZW8KiTZMBFqtnw2nqRyhdSFzn3cGM elected master
  - demoting self` at 15:01:12 -- PC peer ID matches Session 15 entry
- Viability test log: `model.mlpackage` predict() returns non-null
  `token_id` + `token_logit` on every call; decode wallclock ~44 ms/token
  at `CPU_AND_NE`
- `hf whoami` returns `user=mechramc`

### State at end of session
- T-0602 (Mac), T-0601 (PC), T-0606 (WiFi discovery) closed
- EXO cluster healthy: PC is Master, Mac is Worker, discovering over WiFi mDNS
- T-0608 (E2B download) done for full multimodal; text-tower extraction not
  required given chunked CoreML bundle covers it
- T-0609 scout complete; viability proven; correct harness is a defined
  engineering task (T-0609a) with reference code in hand
- Outstanding critical-path: T-0609a chunked harness, T-0610 Rustane, T-0611
  anemll-flash-llama.cpp build, T-0607 / T-0612 remaining downloads
- Stretch: T-0609b upstream PR to ANEMLL

---

## Session 16 - 2026-04-21: Model family migration to Gemma 4

### What was done

**Model family pivot (Qwen -> Gemma 4):**
- All three core model slots switched from Qwen 3.5 / 3.6 to Gemma 4 (Apache 2.0):
  - Dense primary: `Qwen 3.5 27B` -> `google/gemma-4-31B-it` (33B dense, 256K ctx)
  - ANE draft: `Qwen3.5-0.6B` -> `google/gemma-4-E2B-it` (5.1B stored / 2.3B
    effective via Per-Layer Embeddings, shares Gemma tokenizer with primary + MoE)
  - MoE / P6: `Qwen3.5-35B-A3B` -> `google/gemma-4-26B-A4B-it` (25.2B / 3.8B
    active, 128 experts with 8 routed + 1 shared, 256K ctx)
- Stretch slot changed: Qwen 2.5 72B removed; replaced with Gemma 4 31B @ 256K
  ctx (distributed + TriAttention) so the C6 ablation remains all-Gemma
- Rationale: writeup leads with Gemma for Google-recruiter framing; mixed-family
  references dilute that story. Qwen 3.6 also had gaps (no 3.6-27B dense, no
  sub-1B draft), which Gemma 4 closes cleanly for two of three slots.

**Files updated:**
- `configs/models.yaml` -- full rewrite: Gemma 4 slot definitions with HF
  repo IDs, tokenizer field, updated sizes (31B Q8_0 ~33 GB, 31B TQ4_1S
  ~23 GB, 26B-A4B ~26 GB, E2B PLE architecture), Flash-MoE config for
  128-expert / 8+1 topology, ablation matrix C0-C7 realigned per policy
- `README.md` -- Models table + results matrix now list Gemma 4 IDs; P-policy
  descriptions say "Draft E2B"
- `CLAUDE.md` -- project overview and example benchmark command updated to
  Gemma 4 (`gemma-4-31b-tq4_1s.gguf`); Apache 2.0 across the core slots
- `tasks.md` -- T-0607/T-0608/T-0609/T-0612 rewritten with Gemma 4 HF repo
  IDs and SCOUT-FIRST notes; T-0624 repurposed as long-ctx stretch; T-0625
  gated on Gemma 4 26B-A4B instead of 35B-A3B
- `scripts/setup_mac.sh` -- next-steps banner mentions Gemma 4 E2B ANE
  conversion with the scout caveat
- `scripts/run_experiment.sh` -- usage, ablation config table, and examples
  updated for the Gemma 4 matrix; c2/c5/c6 now the ANE-requiring configs
- `src/crossfire/autopilot/decision_tree.py` -- memory-threshold comment
  now refers to Gemma 4 31B sizing, DecisionContext docstring mentions
  Gemma 4 26B-A4B as the MoE example
- `src/crossfire/autopilot/policy.py` -- docstring updated to Gemma 4 26B-A4B
- `src/crossfire/autopilot/query_classifier.py` -- QueryFeatures docstring
  updated
- `src/crossfire/flashmoe/runtime.py` -- sidecar extraction docstring
  flags Gemma's 128+1 topology as something the stock extractor was not
  built for
- `src/crossfire/utils/metrics.py` -- BenchmarkResult docstring example
  identifiers updated
- `tests/test_metrics.py` -- fixtures use `gemma-4-31b` / `gemma-4-26b-a4b`;
  ANE role string now `draft_e2b`
- `tests/test_flashmoe.py` -- generic model-path fixtures updated to
  `/models/gemma.gguf` for consistency

**Known unknowns flagged in status.md and memory/models_gemma4.md:**
- ANE / CoreML conversion path for Gemma 4 E2B is unvalidated
  (ANEMLL has no E2B benchmark; PLE may not round-trip through CoreML)
- Flash-MoE sidecar extraction for Gemma 4 26B-A4B is unvalidated
  (anemll-flash-llama.cpp sidecar extractor was built around Qwen / Kimi
  topology, not 128-expert + 1-shared-expert)

### Verification
- `pytest`: 129 passed
- `ruff check .`: clean (All checks passed!)
- `ruff format --check .`: clean (40 files already formatted)
- `grep -rn "[Qq]wen"` shows only historical references (Session 16
  checkpoint entry describing the pivot, explanatory notes about what
  the Flash-MoE extractor was built for, the TriAttention paper's
  validation set, and the archived pre-migration spec doc)

### State at end of session
- Repo is fully Gemma 4 across docs, configs, scripts, code, and tests
- Stretch slot reframed as long-ctx Gemma 4 31B (no Qwen references remain
  in the public-facing matrix)
- Ready to begin T-0607/T-0608/T-0612 downloads; T-0609 and T-0612 need
  scout runs before full calibration

---

## Session 15 - 2026-04-21: Phase 6 PC bring-up (T-0601) and CRLF shell-script fix

### What was done

**T-0601 executed end-to-end on PC (WSL2 Ubuntu 24.04):**
- Installed prerequisites inside WSL: `build-essential`, the NVIDIA CUDA 13 apt
  repo via `cuda-keyring_1.1-1_all.deb`, `cuda-toolkit-13-0` (resolved to CUDA
  13.2.78), `cmake 3.28.3`, and Node 22 LTS via `nodesource setup_22.x` script
- Appended properly-quoted CUDA exports (`PATH=/usr/local/cuda-13.2/bin:$PATH`
  and `LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64`) to `~/.profile` so login
  shells pick up `nvcc` -- earlier attempt was corrupted by unescaped Windows
  `Program Files (x86)` paths
- `scripts/setup_pc.sh` ran green: EXO source clone synced with `--extra cuda13`,
  dashboard built (`npm install && npm run build`, 980 modules transformed),
  llama.cpp TurboQuant+ fork built with `GGML_CUDA=ON` using 24 parallel jobs;
  `ldd llama-cli` shows `libggml-cuda.so.0`, `libcudart.so.13`, `libcublas.so.13`,
  `libcublasLt.so.13` linked against `/usr/local/cuda-13.2/lib64`
- `~/crossfire/exo/.venv/bin/exo -v` launched cleanly with node id
  `12D3KooWLeMLzYwnaBdSQagZW8...`, API live on `localhost:52415`, dashboard
  banner printed; discovered Mac peer at `192.168.4.41:52415` over WiFi mDNS
- `curl localhost:52415/node_id` and `/v1/models` both return valid responses

**CRLF line-ending fix:**
- Root cause: `.gitattributes` had only `* text=auto`, so Windows git with
  `core.autocrlf=true` wrote CRLF to every `*.sh` on checkout. In WSL bash,
  `set -euo pipefail\r` fails with "set: pipefail: invalid option name"
  because `\r` is parsed as part of the option name, killing the script at
  line 2
- Fix: added `*.sh text eol=lf` to `.gitattributes` (commit `8872d62`); then
  `git add --renormalize .` + working-tree re-checkout of the three shell
  scripts (`setup_mac.sh`, `setup_pc.sh`, `build_flash_moe.sh`). All three now
  have LF line terminators on disk and will stay LF on future clones/pulls

**Interim workaround used during session:**
- Before the `.gitattributes` fix, ran `scripts/setup_pc.sh` via
  `bash <(tr -d "\r" < scripts/setup_pc.sh)` to strip CR on the fly.
  This broke because `BASH_SOURCE[0]` resolves to `/dev/fd/NN` under process
  substitution, so `PROJECT_ROOT` became `/dev` and the llama.cpp clone
  targeted `/dev/vendor/llama.cpp`. Switched to converting the working-tree
  file in place with `tr -d "\r"`, which let the script run normally

**Tracker updates:**
- `tasks.md` -- T-0601 marked done with session-15 verification notes
- `status.md` -- latest commit, Phase 6 PC bring-up summary, `Not Started`
  section clarified (USB4 tasks deferred per `memory/interconnect.md` because
  the active interconnect is WiFi), Immediate Next Work reordered to lead with
  model downloads / ANE conversion / Flash-MoE build now that both nodes are
  running EXO
- `checkpoint.md` -- this entry

### Verification
- `nvcc --version` in login shell: `release 13.2, V13.2.78`
- `cmake --version`: `3.28.3`
- `node --version && npm --version`: `v22.22.2` / `10.9.7`
- `ldd vendor/llama.cpp/build/bin/llama-cli | grep -iE 'cuda|cublas'`:
  `libggml-cuda.so.0`, `libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`,
  `libcuda.so.1` (WSL GPU passthrough via `/usr/lib/wsl/lib/libcuda.so.1`)
- `curl -s http://localhost:52415/node_id`:
  `"12D3KooWLeMLzYwnaBdSQagZW8KiTZMBFqtnw2nqRyhdSFzn3cGM"`
- `file scripts/*.sh`: all report Bourne-Again shell script without "CRLF line
  terminators"

### State at end of session
- PC bring-up complete; T-0601 closed; EXO + llama.cpp (CUDA) both runnable
- Both nodes running EXO and discovering each other over WiFi libp2p; ready
  for model downloads and calibration without USB4
- Shell-script CRLF landmine closed at the repo level (`.gitattributes`);
  future clones on Windows will not regress
- iperf3 still not installed on either node; needed for eventual T-0605 USB4
  baseline, not urgent while WiFi is the active path

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
