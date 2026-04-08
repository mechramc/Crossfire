# AGENTS.md

These instructions are mandatory for every coding agent working in this repository.

## Primary Rule

Before any `git push`, the agent must bring the project trackers up to date:

- `tasks.md`
- `status.md`
- `checkpoint.md`

If those files do not reflect the actual repository state, the work is not ready to push.

## Tracker Responsibilities

### `tasks.md`

Maintain `tasks.md` as the atomic source of truth.

Rules:
- Only mark a task done when the file, code path, test, artifact, or hardware action actually exists and has been verified in the current repo or environment.
- Do not mark roadmap items done because a plan says they should be done.
- Use partial status only when code exists but the implementation is still incomplete, placeholder, or blocked on follow-up work.
- Add new atomic tasks when work appears that is not already represented.
- Split vague tasks into concrete tasks before marking progress.

### `status.md`

Maintain `status.md` as the concise executive snapshot.

Rules:
- Reflect current repo reality, not aspiration.
- Call out mismatches between planning docs and code.
- State what is complete, partial, not started, and blocked.
- Include current verification status for the latest edit batch.
- Include the current branch and latest commit when practical.

### `checkpoint.md`

Maintain `checkpoint.md` as the durable session log.

Rules:
- Add a new top entry for every meaningful work session.
- Record what was actually changed, not what was intended.
- Record the ending state, open risks, and whether verification was run.
- Do not delete prior session history unless it is factually wrong and replaced by a corrected entry.

## Push Gate

Before any `git push`, the agent must complete all of the following:

1. Re-read `tasks.md`, `status.md`, and `checkpoint.md`.
2. Update them to match the current repo state.
3. Run the required verification for the current batch.
4. Reflect the verification result in `status.md` and `checkpoint.md`.
5. Confirm there is no obvious mismatch between:
   - tracker files
   - code on disk
   - git diff

If any of those are out of sync, do not push yet.

## Verification Rule

For code or config changes, run the project checks that apply to this repository:

```bash
pytest
ruff check .
ruff format --check .
```

If a command fails:
- do not claim the work is complete
- record the failure in `status.md` and `checkpoint.md`
- either fix it or explicitly leave it as an open blocker

## Completion Standard

A task is only complete when all of the following are true:
- the implementation exists in the repository
- any required tests or verification for that work have been run
- the trackers reflect that state
- the checkpoint log records the session

## Naming And Scope Discipline

This repo currently has a live mismatch between `CROSSFIRE-X` planning docs and `CROSSFIRE v2` code/public docs.

Rules:
- call out naming mismatches in `status.md` until they are fixed
- do not silently mark rename work done while old identifiers remain in code or docs
- keep hardware tasks blocked until the hardware step was actually executed and recorded
- keep placeholder implementations marked partial until the real execution path exists

## Agent Behavior

- Prefer updating the trackers in the same change set as the code change they describe.
- Do not leave tracker updates for later if a push may happen first.
- Do not trust earlier tracker claims without rechecking the actual files.
- If you are unsure whether something is complete, mark it pending or partial and explain why.
