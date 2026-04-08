# CROSSFIRE-X Checkpoint Log

Purpose: durable work log for meaningful project sessions.
Rule: update this file before every `git push`.

---

## Session 7 - 2026-04-07: Tracker reconciliation and push-gate rules

### What was done
- Re-audited the repository contents instead of trusting the existing tracker files
- Rebuilt `tasks.md` as an atomic ledger grounded in what actually exists on disk
- Marked scaffolded modules, configs, scripts, and existing tests as done only where the files and implementations are present
- Left hardware, calibration, AutoPilot, dashboard, and public rename work pending because the repo does not yet contain those deliverables
- Rewrote `status.md` to describe the actual repo state: scaffolded but not yet experimentally runnable
- Rewrote `AGENTS.md` so every future agent must update `tasks.md`, `status.md`, and `checkpoint.md` before any `git push`

### State at end of session
- Tracking docs are now aligned with the current repository state
- Key mismatch remains: planning docs say `CROSSFIRE-X`, while public/package/source identifiers still contain `CROSSFIRE v2`
- Verification completed: `pytest`, `ruff check .`, and `ruff format --check .` all passed; cache-write warnings were non-fatal

---

## Session 6 - 2026-04-07: Phase 1 batch 1

### What was done
- Completed `TASK-101`
  - Renamed `CLAUDE.md` from `CROSSFIRE v2` references to `CROSSFIRE-X`
- Updated trackers to reflect the first completed Phase 1 task
- Corrected tracker state to acknowledge the current local workspace changes

### State at end of session
- `TASK-101` is complete
- `TASK-102` to `TASK-104` remain pending
- Verification pending for this batch

---

## Session 5 - 2026-04-07: CROSSFIRE-X spec + tasks rewrite

### What was done
- Reviewed the new `CROSSFIRE-X` spec replacing the earlier v2 plan
- Wrote `CROSSFIRE-X_Implementation_Spec.md`
- Reworked the project plan around `P0-P5` execution policies, AutoPilot, dashboard work, and impossible-scenario experiments
- Updated the earlier tracker files to match that plan

### State at end of session
- Strategy docs were updated
- Repo implementation still depended on follow-on code work
- Tests and lint had last been reported clean at that point

---

## Session 4 - 2026-04-07: speculative harness prep

### What was done
- Added `src/crossfire/ane/speculative.py`
- Implemented a bounded speculative decoding step:
  - draft model proposes tokens
  - verifier returns authoritative tokens
  - accepted prefix and rejection path are computed
- Added focused tests for accept, reject, and empty-result paths

### State at end of session
- Speculative harness logic existed in repo
- Hardware-gated speculative experiments remained pending
- Verification was still pending for that session's tracker entry

---

## Session 3 - 2026-04-07: first tracker pass

### What was done
- Created the earlier versions of `tasks.md`, `status.md`, and `checkpoint.md`
- Captured the then-current v2-oriented roadmap and repository status

### State at end of session
- Tracking infrastructure existed
- Later sessions superseded portions of that tracker content

---

## Session 2 - 2026-04-07: v2 spec alignment

### What was done
- Updated docs and code toward the v2 EXO + ANE architecture
- Added ANE module scaffolds
- Updated distributed pipeline/network scaffolds
- Updated configs and scripts for EXO, ANEMLL, Rustane, and RDMA assumptions
- Added and updated tests around the new scaffolds

### State at end of session
- Repo had moved to the EXO/ANE architecture direction
- Public naming still reflected `CROSSFIRE v2`

---

## Session 1 - 2026-04-06: project initialization

### What was done
- Created the initial repository scaffold
- Added package metadata, baseline docs, configs, scripts, tests, and ignore rules
- Fixed packaging and ruff issues in the initial setup

### Committed as
- `4611d21` - `Initialize project structure with scaffolded source, benchmarks, and configs`

### State at end of session
- Initial scaffold was committed and pushed
