# CROSSFIRE-X: Implementation Specification

**Version:** 1.0
**Date:** April 2026
**Author:** Murai Labs (Ramchand Easwar)
**Status:** INTERNAL — DO NOT COMMIT

---

## 1. Executive Summary

CROSSFIRE-X is the third iteration of the CROSSFIRE project:

- **v1** asked: can we combine heterogeneous distributed inference, TQ4_1S weight compression, and KV cache compression? (Answer: yes, but EXO already solved the distributed part.)
- **v2** asked: can we light up the ANE as a compute target inside an EXO pipeline? (Answer: validated in scaffolding, pending hardware experiments.)
- **X** asks: can a runtime intelligence layer automatically select the optimal execution strategy per request across GPU, ANE, and compression?

The headline contribution is **AutoPilot** (Layer 5) — a UCB1 multi-armed bandit that classifies each incoming query, filters policies by hardware availability, selects the execution policy with the highest upper confidence bound, executes it, observes the result, and updates its model. Over time, AutoPilot converges to the oracle policy for each query class.

**Success criteria (any one):**
- +20% tok/s over the EXO-only baseline (P1)
- +30% tokens-per-watt over P1
- Ability to run a model or context length impossible on either machine alone

---

## 2. One-Line Claim (Locked)

> A local AI cluster that automatically decides how to use GPU, ANE, and compression in real-time to maximize performance, efficiency, or capability — per prompt.

---

## 3. Architecture

Five layers, each independently testable:

```
Layer 5: AutoPilot          (runtime policy selection — UCB1 bandit)
Layer 4: Compression Plane  (TQ4_1S weights, turbo3 KV cache)
Layer 3: Inter-node Orchestration  (EXO — RDMA over TB5)
Layer 2: Compute Substrate  (5 targets across 2 machines)
Layer 1: Model              (Qwen 3.5 27B / 0.6B / 2.5 72B)
```

### Compute Targets (Layer 2)

| Target | Hardware | Managed By | Role | Power |
|--------|----------|------------|------|-------|
| T1: CUDA GPU | RTX 5090 32GB | EXO (CUDA) | Prefill | ~350W |
| T2: Metal GPU | M4 Max 40-core | EXO (MLX) | Primary decode | ~40-60W |
| T3: ANE | M4 Max 16-core ANE | ANEMLL / Rustane | Draft model (speculative) | ~2-5W |
| T4: CPU/SME | M4 Max CPU | EXO scheduler | KV cache mgmt, verification | ~10-15W |
| T5: RDMA | Thunderbolt 5 link | EXO RDMA (IBV) | KV cache streaming | ~2W |

### Data Flow

```
Prompt received
  -> AutoPilot classifies query, selects policy
  -> EXO initializes pipeline per policy
  -> T1 (5090) performs prefill [if distributed]
  -> KV cache streamed via T5 (RDMA) [if distributed]
  -> T2 (Metal GPU) performs decode
  -> T3 (ANE) runs draft model + T4 (CPU) verifies [if P2/P5]
  -> Compression applied at load (TQ4_1S) and runtime (turbo3 KV) [if P3-P5]
  -> Metrics collected, AutoPilot receives reward, bandit updated
```

---

## 4. Execution Policies

Six policies, ordered by complexity. AutoPilot selects one per request.

| Policy | Nodes | ANE | TQ4_1S | turbo KV | Requires | Description |
|--------|-------|-----|--------|----------|----------|-------------|
| **P0** | Single best | Idle | No | No | — | Lowest latency, no distribution overhead |
| **P1** | Distributed | Idle | No | No | RDMA | EXO split: 5090 prefill, Mac decode |
| **P2** | Distributed | Draft 0.6B | No | No | RDMA + ANE | P1 + speculative decode via ANE |
| **P3** | Distributed | Idle | Yes | No | RDMA + TQ4_1S file | P1 + compressed weights |
| **P4** | Distributed | Idle | Yes | Yes | RDMA + TQ4_1S + turbo | P3 + KV cache compression |
| **P5** | Distributed | Draft 0.6B | Yes | Yes | All | Full stack — everything active |

### Policy selection intuition

| Scenario | Best policy | Why |
|----------|-------------|-----|
| Short prompt, short generation | P0 | RDMA overhead exceeds benefit |
| Medium prompt, medium generation | P1 | Prefill benefits from 5090 TFLOPS |
| Long generation (>256 tokens) | P2 | Speculative decode amortizes ANE overhead |
| Large model (>27B) | P3 | TQ4_1S needed to fit in memory |
| Long context (>16K) | P4 | turbo KV prevents OOM |
| All of the above | P5 | Stacking everything for max throughput |

### Hardware requirements per policy

| Policy | RDMA link | ANE available | TQ4_1S model | turbo KV build |
|--------|-----------|---------------|--------------|----------------|
| P0 | — | — | — | — |
| P1 | Required | — | — | — |
| P2 | Required | Required | — | — |
| P3 | Required | — | Required | — |
| P4 | Required | — | Required | Required |
| P5 | Required | Required | Required | Required |

P0 is always available as a fallback. If RDMA is down, AutoPilot only offers P0.

---

## 5. AutoPilot Design

### 5.1 Query Classification

Seven query classes determine the bandit context:

| Class | Condition | Priority |
|-------|-----------|----------|
| BATCH | concurrent_requests > 1 | Highest |
| SHORT_GEN | max_gen_tokens <= 64 | High |
| LONG_GEN | max_gen_tokens > 256 | High |
| VERY_LONG_PROMPT | prompt_tokens >= 16384 | Medium |
| LONG_PROMPT | 4096 <= prompt_tokens < 16384 | Medium |
| MEDIUM_PROMPT | 512 <= prompt_tokens < 4096 | Low |
| SHORT_PROMPT | prompt_tokens < 512 | Lowest |

Classification priority resolves ties: BATCH > generation-length > prompt-length.

Input features for classification:

```
QueryFeatures:
  prompt_tokens: int          # tokenized prompt length
  max_gen_tokens: int         # requested generation length
  context_used: int           # current KV cache occupancy
  model_size_b: float         # model parameter count (billions)
  available_vram_mb: float    # free VRAM at request time
  concurrent_requests: int    # number of in-flight requests
```

### 5.2 UCB1 Multi-Armed Bandit

One bandit per QueryClass. Each arm is an ExecutionPolicy. Total state: 7 bandits x 6 arms = 42 arm-stat entries.

**Selection formula:**

```
UCB1_score(arm) = mean_reward(arm) + c * sqrt(ln(total_pulls) / arm_pulls)
```

Where:
- `mean_reward(arm)` = cumulative reward / number of pulls
- `c` = exploration weight (default 2.0, tunable)
- `total_pulls` = total decisions for this QueryClass
- `arm_pulls` = times this arm was selected for this QueryClass

Arms with 0 pulls are selected first (forced exploration).

**Cold start strategy:**
- First 42 requests cycle through all arms via round-robin (7 classes x 6 policies)
- This seeds every arm with at least 1 observation before UCB1 takes over
- If a policy is hardware-unavailable, it gets skipped (no forced exploration)

**Thompson sampling alternative:**
- Beta(alpha, beta) priors per arm
- Reward binarized: r > 0.5 -> success (alpha += 1), else failure (beta += 1)
- Available as fallback if UCB1 converges too slowly

**Persistence:**
- Bandit state serialized to `results/autopilot_state.json` every 10 decisions
- Load on startup if file exists (warm start across sessions)

### 5.3 Reward Function

Multi-objective reward normalized to [0, 1]:

```
reward = w_throughput * R_throughput
       + w_efficiency * R_efficiency
       + w_latency    * R_latency
       + w_quality    * R_quality
```

**Default weights:**
- throughput_weight = 0.4
- efficiency_weight = 0.3
- latency_weight = 0.2
- quality_weight = 0.1

**Component normalization (all against P1 baseline from calibration):**

| Component | Formula | Range |
|-----------|---------|-------|
| Throughput | min(tok_s / baseline_tok_s, 2.0) / 2.0 | [0, 1] |
| Efficiency | min(tok_watt / baseline_tok_watt, 2.0) / 2.0 | [0, 1] |
| Latency | max(0, 1.0 - ttft_ms / baseline_ttft_ms) | [0, 1] |
| Quality | 1.0 if ppl_delta < 0.5, else max(0, 1 - ppl_delta/5.0) | [0, 1] |

### 5.4 Decision Logging

Every decision logged as JSONL to `results/autopilot_decisions.jsonl`:

```json
{
  "timestamp": "2026-04-15T14:32:01Z",
  "query_class": "MEDIUM_PROMPT",
  "selected_policy": "P5",
  "was_exploration": false,
  "ucb_scores": {"P0": 0.62, "P1": 0.71, "P2": 0.83, "P3": 0.77, "P4": 0.80, "P5": 0.91},
  "tokens_per_second": 47.2,
  "tokens_per_watt": 0.31,
  "ttft_ms": 245,
  "acceptance_rate": 0.72,
  "reward": 0.82,
  "execution_time_ms": 3400
}
```

---

## 6. Models

| Model | Parameters | Role | Q8_0 Size | TQ4_1S Size | Constraints |
|-------|-----------|------|-----------|-------------|-------------|
| Qwen 3.5 27B | 27B | Primary | 26.6 GB | 19.1 GB | Fits both machines independently |
| Qwen 3.5 0.6B | 0.6B | ANE draft | — | — | Convert via ANEMLL, max ctx 2048, expected 47-62 tok/s |
| Qwen 2.5 72B | 72B | Stretch | 72.0 GB | 45.8 GB | Requires distributed; fits neither alone at Q8_0 |

**Evaluation dataset:** wikitext-2-raw-v1 (PPL measurement, 20 chunks at c=512)

**Evaluation context lengths:** 8192, 16384, 32768 (+ 131072 for impossible scenario)

---

## 7. Hardware Constraints

### Critical constraints (must not violate)

| Constraint | Impact | Mitigation |
|-----------|--------|------------|
| ANE 32 MB SRAM cliff | 30% throughput drop beyond 32 MB | Keep draft model activations < 32 MB |
| ANE dim efficiency cliff at 5120 | 4.7x penalty | Keep hidden dim <= 4096 |
| ANEMLL max model: ~8B | Cannot run 27B on ANE | ANE only runs 0.6B draft model |
| ANEMLL max context: 4096 | Draft model limited to 4096 ctx | Draft speculates within 4096 window |
| macOS 26.3 CoreML routing | `compute_units=ALL` routes to GPU, not ANE | Use direct private API (Orion/Rustane path) |
| Blackwell turbo dequant | 25-38% decode penalty with symmetric config | Use asymmetric: q8_0-K / turbo3-V only |
| KV cache FP16 ANE->GGML bridge | 11-16% decode degradation | ANE for draft only (no KV bridge needed — independent caches) |
| EXO 1.0 RTX 5090 support | Demos use DGX Spark (ARM), not standard Linux | Fallback: llama.cpp RPC for distributed layer |

### Hardware specs

| Node | GPU | Memory | Bandwidth | Compute | Power |
|------|-----|--------|-----------|---------|-------|
| PC | RTX 5090 | 32 GB GDDR7 | ~1,792 GB/s | ~200 TFLOPS FP16 | ~350W |
| Mac | M4 Max 40-core | 64 GB unified | 546 GB/s | ~54 TFLOPS FP16 | ~50W |
| Mac ANE | 16-core | Shared unified | — | ~19 TFLOPS FP16 | ~3W |

**Network:** Thunderbolt 5 (80 Gbps bidirectional, RDMA via EXO, 3us latency)

**Mac config:** `sudo sysctl iogpu.wired_limit_mb=58982` (90% of 64GB for Metal)

**RDMA enable:** Recovery mode -> `rdma_ctl enable` (requires macOS Tahoe 26.2+)

---

## 8. Evaluation Methodology

### 8.1 Phase 2: Calibration (Days 3-6)

Run every policy at 3 context lengths. Produces a 6x3 matrix of 18 benchmark runs:

| Policy | 8K ctx | 16K ctx | 32K ctx |
|--------|--------|---------|---------|
| P0 | tok/s, TTFT, PPL, power | ... | ... |
| P1 | tok/s, TTFT, PPL, power | ... | ... |
| P2 | tok/s, TTFT, PPL, power | ... | ... |
| P3 | tok/s, TTFT, PPL, power | ... | ... |
| P4 | tok/s, TTFT, PPL, power | ... | ... |
| P5 | tok/s, TTFT, PPL, power | ... | ... |

**P1 at 8K context becomes the normalization baseline for all reward calculations.**

### 8.2 Phase 3: AutoPilot Evaluation (Days 7-8)

**Mixed workload:** 200 queries sampled from 7 query classes:
- 40 SHORT_PROMPT (< 512 tokens)
- 30 MEDIUM_PROMPT (512-4096)
- 30 LONG_PROMPT (4096-16384)
- 20 VERY_LONG_PROMPT (>= 16384)
- 30 SHORT_GEN (<= 64 tokens output)
- 30 LONG_GEN (> 256 tokens output)
- 20 BATCH (concurrent)

Three runs on the same workload:
1. **AutoPilot**: bandit selects policy per query
2. **Oracle**: best fixed policy per query class (from calibration)
3. **Random**: uniform random policy selection

**Metrics:**
- Cumulative regret: sum(oracle_reward - autopilot_reward) over 200 queries
- Convergence: after query 100, is AutoPilot within 5% of oracle?
- Policy distribution: which policies does AutoPilot favor per class?
- Total throughput: aggregate tok/s across workload
- Total efficiency: aggregate tok/watt across workload

### 8.3 Success Verification

| Criterion | Measurement | Threshold |
|-----------|-------------|-----------|
| Throughput gain | AutoPilot avg tok/s vs P1 avg tok/s | >= +20% |
| Efficiency gain | AutoPilot avg tok/watt vs P1 avg tok/watt | >= +30% |
| Capability | 72B runs OR 128K context works | Binary yes |
| Convergence | Regret growth rate after 100 queries | Sublinear (flattening) |
| Quality | Max PPL delta across all policies | < +5% vs Q8_0 baseline |

---

## 9. Demo Dashboard

### 9.1 TUI (Primary) — Rich / Textual

Four-quadrant layout, refreshing every 500ms:

```
+-------------------------------+---------------------+
|    Compute Targets (5 bars)   |   AutoPilot Status  |
|    T1: CUDA [====    ] 62%    |   Policy: P5        |
|    T2: Metal[========] 95%    |   Class:  MEDIUM    |
|    T3: ANE  [======= ] 88%   |   Reward: 0.82      |
|    T4: CPU  [==      ] 23%   |   Explore: no       |
|    T5: RDMA [===     ] 38%   |   Pulls: 142        |
+-------------------------------+---------------------+
|         Live Metrics          |   Decision History  |
|   tok/s: 47.2 (+23% vs P1)   |   #142 P5 r=0.82   |
|   tok/W: 0.31 (+35% vs P1)   |   #141 P2 r=0.71   |
|   TTFT:  245ms               |   #140 P5 r=0.85   |
|   Accept: 72%                |   #139 P1 r=0.55   |
|   Memory: 54.2/64 GB         |   #138 P5 r=0.88   |
+-------------------------------+---------------------+
```

**Data sources:**
- Compute utilization: `powermetrics` (Mac), `nvidia-smi` (PC via SSH/RPC)
- AutoPilot status: AutoPilot.get_policy_stats()
- Live metrics: BenchmarkResult from latest inference
- Decision history: tail of `results/autopilot_decisions.jsonl`

**Keyboard controls:** `p` = pause, `r` = reset bandit, `q` = quit, `1-6` = force policy

### 9.2 Web (Stretch) — FastAPI + htmx

Same 4-panel layout rendered as HTML. SSE endpoint pushes updates. No JavaScript framework — htmx swaps panels on SSE events.

**Optional dependencies:** `fastapi`, `uvicorn`, `jinja2`

---

## 10. Impossible Scenarios

### 10.1 72B on Consumer Hardware

**Setup:** Qwen 2.5 72B compressed from Q8_0 (72G) to TQ4_1S (45.8G). Pipeline split: early layers on RTX 5090 (32GB VRAM), remaining layers on Mac Studio (64GB unified). turbo3 KV on decode side. ANE runs 0.6B draft model for speculative decode.

**This is impossible without distribution** — 72B at Q8_0 fits neither machine. Even at TQ4_1S (45.8G), it exceeds the 5090's 32GB VRAM. Only the combined memory pool (32 + 64 = 96 GB) can hold it.

**Success:** Binary — does it generate coherent text? If yes, measure tok/s.

### 10.2 128K Context on 64GB

**Setup:** Qwen 3.5 27B at TQ4_1S (19.1G) with context=131072. Without turbo3 KV compression, the KV cache at 128K context for a 27B model would require ~16-24 GB (depending on head count and precision). Combined with model weights, this risks OOM on the Mac's 64GB.

**turbo3 KV compresses the KV cache by ~4x**, making 128K feasible:
- Model weights (TQ4_1S): ~19.1 GB
- KV cache (turbo3, 128K ctx): ~4-6 GB (vs ~16-24 GB at FP16)
- Total: ~23-25 GB, well within 64GB with headroom

**This is impossible without turbo KV compression** at this context length on 64GB.

**Success:** Binary — no OOM, generates text at 128K? If yes, measure decode tok/s.

---

## 11. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | EXO doesn't support RTX 5090 (demos use DGX Spark) | Medium | High | Fallback to llama.cpp RPC + EXO Mac-only. Still tests ANE + AutoPilot. |
| R2 | ANE degrades GPU decode > 2% (interference) | Low | High | **Gate G1.** If fail: disable P2/P5, AutoPilot uses P0-P4 only. |
| R3 | EXO rejects TQ4_1S GGUF type IDs (44/45) | Medium | Medium | **Gate G3.** Load-time TQ4_1S->Q8_0 conversion (retains file size benefit). |
| R4 | Blackwell turbo dequant penalty on 5090 | Confirmed | Medium | Use asymmetric config only (q8_0-K / turbo3-V). Never symmetric on Blackwell. |
| R5 | macOS 26.3 CoreML routes to GPU | Confirmed | Medium | Use direct private API (Orion/Rustane). Pin macOS version. |
| R6 | 72B doesn't fit after OS overhead | Medium | Medium | iogpu.wired_limit_mb=58982 + turbo3 KV to reduce runtime memory. |
| R7 | UCB1 bandit converges too slowly | Low | Medium | Switch to Thompson sampling. Reduce exploration_weight. Warm-start from calibration. |
| R8 | Private ANE API instability | Medium | Low | ANEMLL CoreML path as stable fallback. This is research, not production. |

---

## 12. Timeline (14 Days, 6 Phases)

### Phase 1: Foundation (Day 1-2)

| Day | Activity | Deliverable |
|-----|----------|-------------|
| 1 | Rename project v2->X. Create AutoPilot module scaffolds. Update configs (C0-C6 -> P0-P5). | All scaffolds exist, tests compile |
| 2 | Implement bandit algorithms (UCB1, Thompson). Implement reward function. Write all Phase 1 tests. | 30+ new tests passing |

### Phase 2: Hardware + Calibration (Day 3-6)

| Day | Activity | Deliverable |
|-----|----------|-------------|
| 3 | Download models. Setup PC (CUDA + EXO). Setup Mac (Metal + EXO + ANEMLL + Rustane). | Both machines ready |
| 4 | Enable TB5 RDMA. Verify EXO auto-discovery. Run P0 baselines (single-node). | Baseline numbers recorded |
| 5 | Run P1 (EXO distributed). **Set P1 as reward normalization baseline.** ANE gate test (G1). | P1 baseline locked. G1 pass/fail. |
| 6 | Calibrate P2-P5 at 8K/16K/32K context. Compress to TQ4_1S (G3). | 6x3 calibration matrix complete |

### Phase 3: AutoPilot Integration (Day 7-8)

| Day | Activity | Deliverable |
|-----|----------|-------------|
| 7 | Wire AutoPilot into pipeline. Warm-start bandits from calibration data. Integration test. | AutoPilot selects policies live |
| 8 | Run 200-query mixed workload (AutoPilot vs oracle vs random). Measure regret. Tune if needed. | Convergence analysis, regret curve |

### Phase 4: Dashboard (Day 9-10)

| Day | Activity | Deliverable |
|-----|----------|-------------|
| 9 | Build TUI dashboard (4 panels: compute, policy, metrics, history). | TUI runs alongside inference |
| 10 | Polish TUI. Start web dashboard (stretch). | Demo-ready TUI |

### Phase 5: Impossible Scenarios (Day 11-12)

| Day | Activity | Deliverable |
|-----|----------|-------------|
| 11 | Download 72B. Compress to TQ4_1S. Run via P5 across both nodes (G4). | Binary: 72B runs or doesn't |
| 12 | 128K context test (G5). Memory breakdown analysis. | Binary: 128K works or doesn't |

### Phase 6: Analysis + Deliverables (Day 13-14)

| Day | Activity | Deliverable |
|-----|----------|-------------|
| 13 | Compile results matrix. Generate charts. Record demo video. | All data finalized |
| 14 | Blog post draft. X thread draft. Update README with results. Submit TurboQuant+ community PR. | All deliverables ready |

---

## 13. Deliverables

1. **GitHub repo** — all code, configs, benchmark data (processed), AutoPilot implementation
2. **AutoPilot proof** — regret curve, convergence analysis, policy distribution per query class
3. **Five-target pipeline** — CUDA + Metal GPU + ANE + CPU/SME + RDMA in a single inference request
4. **Power-performance matrix** — per-target wattage and throughput contribution
5. **Impossible scenarios** — 72B on consumer hardware + 128K context
6. **Demo video** — TUI dashboard showing AutoPilot switching policies live
7. **TurboQuant+ community PR** — results submitted using standard template
8. **X/Twitter thread** — visual thread with 5 compute targets, AutoPilot, impossible scenarios
9. **Blog post (murailabs.com)** — connecting KALAVAI, Orion, and CROSSFIRE-X

---

## 14. Credits

Built on the work of:

- **Alex Cheema & EXO Labs** — EXO distributed inference, RDMA over TB5
- **maderix (Manjeet Singh)** — foundational ANE reverse-engineering
- **ANEMLL team** — production ANE inference pipeline
- **Daniel Isaac (@danpacary)** — Rustane: Rust-native ANE training/inference
- **AtomGradient** — disaggregated ANE+GPU benchmarks, macOS 26.3 routing discovery
- **SqueezeBits** — Yetter disaggregated inference engine
- **Tom Turney (TheTom)** — TurboQuant+ weight and KV cache compression
- **Jeff Geerling** — Mac Studio RDMA cluster benchmarks
- **Ahmad** — inference engine taxonomy that prompted the CROSSFIRE-X architecture
- **Orion (Murai Labs, arXiv:2603.06728)** — first open ANE programming system

---

## Appendix A: Module File Map

### New files (CROSSFIRE-X)

```
src/crossfire/autopilot/
  __init__.py              # Package exports
  query_classifier.py      # QueryClass enum, QueryFeatures, classify_query()
  policy.py                # ExecutionPolicy enum, PolicyConfig, hardware reqs
  bandit.py                # ArmStats, UCB1Bandit, ThompsonBandit
  reward.py                # RewardWeights, compute_reward()
  logger.py                # DecisionRecord, DecisionLogger (JSONL)
  autopilot.py             # AutoPilot orchestrator class

src/crossfire/dashboard/
  __init__.py
  tui.py                   # Textual TUI entry point
  panels/
    __init__.py
    compute_panel.py        # 5-target utilization bars
    policy_panel.py         # AutoPilot status display
    metrics_panel.py        # Live metrics with % vs baseline
    decision_log.py         # Decision history scrollback

configs/
  autopilot.yaml            # Bandit config (weights, thresholds, paths)

tests/
  test_query_classifier.py  # 7+ tests
  test_bandit.py            # 12+ tests (UCB1 + Thompson)
  test_reward.py            # 5+ tests
  test_autopilot.py         # Integration tests
```

### Modified files

```
src/crossfire/__init__.py         # Version bump
src/crossfire/utils/metrics.py    # ablation_config -> execution_policy, new fields
src/crossfire/distributed/pipeline.py  # Add execution_policy, update validation
configs/models.yaml               # ablation: C0-C6 -> policies: P0-P5
configs/hardware.yaml             # Minor docstring updates
pyproject.toml                    # Version, textual/rich deps
CLAUDE.md                         # Rename, add AutoPilot section
README.md                         # Rename, add AutoPilot and demo sections
.gitignore                        # Add spec file
tests/test_metrics.py             # Update for renamed fields
tests/test_pipeline.py            # Update for execution_policy
```

### Appendix B: AutoPilot Configuration (`configs/autopilot.yaml`)

```yaml
autopilot:
  bandit_type: ucb1            # ucb1 or thompson
  exploration_weight: 2.0      # c parameter for UCB1
  cold_start_rounds: 42        # 7 classes x 6 policies
  persist_interval: 10         # save state every N decisions
  state_path: results/autopilot_state.json
  log_path: results/autopilot_decisions.jsonl

reward_weights:
  throughput: 0.4
  efficiency: 0.3
  latency: 0.2
  quality: 0.1

thresholds:
  short_prompt_tokens: 512
  medium_prompt_tokens: 4096
  long_prompt_tokens: 16384
  short_gen_tokens: 64
  long_gen_tokens: 256

success_criteria:
  throughput_gain: 0.20        # +20%
  efficiency_gain: 0.30        # +30%
  ppl_max_delta: 0.5
  convergence_queries: 100     # within 5% of oracle after this many
```
