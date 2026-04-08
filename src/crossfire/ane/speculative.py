"""Speculative decoding harness for ANE draft models.

This module provides one bounded speculative decoding step: the ANE draft model
proposes tokens, the verifier returns authoritative tokens, and the harness
computes which tokens can be committed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from crossfire.ane.draft_model import DraftResult


class DraftTokenGenerator(Protocol):
    """Protocol for a draft model used during speculative decoding."""

    def generate_draft(
        self,
        prompt_tokens: Sequence[int],
        *,
        max_tokens: int | None = None,
    ) -> DraftResult:
        """Generate a draft token sequence for the current prompt."""


class TokenVerifier(Protocol):
    """Protocol for the authoritative verification path."""

    def verify_tokens(
        self,
        prompt_tokens: Sequence[int],
        draft_tokens: Sequence[int],
    ) -> Sequence[int]:
        """Return authoritative tokens for the current speculative step."""


@dataclass(frozen=True)
class SpeculativeStepResult:
    """Outcome of a single speculative decoding step."""

    draft_tokens: list[int]
    verified_tokens: list[int]
    accepted_tokens: list[int]
    committed_tokens: list[int]
    rejected: bool
    draft_elapsed_ms: float
    verifier_generated_token: int | None

    @property
    def accepted_count(self) -> int:
        """Count of accepted draft tokens."""

        return len(self.accepted_tokens)


def run_speculative_step(
    prompt_tokens: Sequence[int],
    *,
    draft_model: DraftTokenGenerator,
    verifier: TokenVerifier,
    max_draft_tokens: int | None = None,
) -> SpeculativeStepResult:
    """Run one speculative decoding step.

    Raises:
        ValueError: If the draft model or verifier returns no tokens.
    """

    draft_result = draft_model.generate_draft(prompt_tokens, max_tokens=max_draft_tokens)
    draft_tokens = list(draft_result.tokens)
    if not draft_tokens:
        msg = "Draft model returned no tokens for speculative decoding"
        raise ValueError(msg)

    verified_tokens = list(verifier.verify_tokens(prompt_tokens, draft_tokens))
    if not verified_tokens:
        msg = "Verifier returned no authoritative tokens"
        raise ValueError(msg)

    accepted_count = _count_accepted_prefix(draft_tokens, verified_tokens)
    accepted_tokens = draft_tokens[:accepted_count]
    rejected = accepted_count < len(draft_tokens)

    if rejected:
        committed_tokens = verified_tokens[: accepted_count + 1]
        verifier_generated_token = committed_tokens[-1]
    else:
        committed_tokens = list(draft_tokens)
        verifier_generated_token = None
        if len(verified_tokens) > len(draft_tokens):
            verifier_generated_token = verified_tokens[len(draft_tokens)]
            committed_tokens.append(verifier_generated_token)

    return SpeculativeStepResult(
        draft_tokens=draft_tokens,
        verified_tokens=verified_tokens,
        accepted_tokens=accepted_tokens,
        committed_tokens=committed_tokens,
        rejected=rejected,
        draft_elapsed_ms=draft_result.elapsed_ms,
        verifier_generated_token=verifier_generated_token,
    )


def _count_accepted_prefix(
    draft_tokens: Sequence[int],
    verified_tokens: Sequence[int],
) -> int:
    """Count the shared prefix between draft and verified tokens."""

    accepted_count = 0
    for draft_token, verified_token in zip(draft_tokens, verified_tokens, strict=False):
        if draft_token != verified_token:
            break
        accepted_count += 1
    return accepted_count
