"""Tests for ANE compute target configuration."""

import pytest

from crossfire.ane.draft_model import ANEBackend, DraftModelConfig, DraftResult
from crossfire.ane.power import (
    ANE_ACTIVE_WATTS_MAX,
    ANE_ACTIVE_WATTS_MIN,
    ANE_SRAM_CLIFF_MB,
    ANE_TFLOPS_FP16,
    PowerSnapshot,
)
from crossfire.ane.speculative import run_speculative_step


def test_draft_model_config_validates_context():
    config = DraftModelConfig(
        model_path=__import__("pathlib").Path("/fake/model"),
        context_size=8192,
    )
    with pytest.raises(ValueError, match="exceeds ANEMLL limit"):
        config.validate()


def test_draft_model_config_validates_path():
    config = DraftModelConfig(
        model_path=__import__("pathlib").Path("/nonexistent/model.mlmodelc"),
    )
    with pytest.raises(FileNotFoundError, match="Draft model not found"):
        config.validate()


def test_ane_backend_values():
    assert ANEBackend.ANEMLL.value == "anemll"
    assert ANEBackend.RUSTANE.value == "rustane"
    assert ANEBackend.COREML.value == "coreml"


def test_draft_result_fields():
    result = DraftResult(
        tokens=[1, 2, 3],
        logits_shape=(3, 32000),
        elapsed_ms=15.2,
        power_watts=3.1,
    )
    assert len(result.tokens) == 3
    assert result.power_watts == 3.1


def test_power_snapshot_ane_fraction():
    snap = PowerSnapshot(
        ane_watts=3.0,
        gpu_watts=50.0,
        cpu_watts=12.0,
        total_system_watts=65.0,
    )
    assert snap.ane_fraction is not None
    assert abs(snap.ane_fraction - 3.0 / 65.0) < 1e-6


def test_power_snapshot_no_total():
    snap = PowerSnapshot(ane_watts=3.0)
    assert snap.ane_fraction is None


def test_ane_constants():
    assert ANE_TFLOPS_FP16 == 19.0
    assert ANE_SRAM_CLIFF_MB == 32
    assert ANE_ACTIVE_WATTS_MIN < ANE_ACTIVE_WATTS_MAX


class StubDraftModel:
    def __init__(self, tokens: list[int]) -> None:
        self._tokens = tokens

    def generate_draft(
        self,
        prompt_tokens: list[int],
        *,
        max_tokens: int | None = None,
    ) -> DraftResult:
        del prompt_tokens
        tokens = self._tokens if max_tokens is None else self._tokens[:max_tokens]
        return DraftResult(tokens=tokens, logits_shape=(len(tokens), 32000), elapsed_ms=4.5)


class StubVerifier:
    def __init__(self, verified_tokens: list[int]) -> None:
        self._verified_tokens = verified_tokens

    def verify_tokens(self, prompt_tokens: list[int], draft_tokens: list[int]) -> list[int]:
        del prompt_tokens
        del draft_tokens
        return self._verified_tokens


def test_run_speculative_step_accepts_full_draft_and_appends_verifier_token():
    result = run_speculative_step(
        [101, 102],
        draft_model=StubDraftModel([201, 202]),
        verifier=StubVerifier([201, 202, 203]),
    )

    assert result.accepted_tokens == [201, 202]
    assert result.accepted_count == 2
    assert result.committed_tokens == [201, 202, 203]
    assert result.verifier_generated_token == 203
    assert result.rejected is False


def test_run_speculative_step_rejects_on_first_mismatch():
    result = run_speculative_step(
        [101, 102],
        draft_model=StubDraftModel([201, 202, 203]),
        verifier=StubVerifier([201, 999, 1000]),
    )

    assert result.accepted_tokens == [201]
    assert result.accepted_count == 1
    assert result.committed_tokens == [201, 999]
    assert result.verifier_generated_token == 999
    assert result.rejected is True


def test_run_speculative_step_requires_draft_tokens():
    with pytest.raises(ValueError, match="Draft model returned no tokens"):
        run_speculative_step(
            [101],
            draft_model=StubDraftModel([]),
            verifier=StubVerifier([301]),
        )


def test_run_speculative_step_requires_verifier_tokens():
    with pytest.raises(ValueError, match="Verifier returned no authoritative tokens"):
        run_speculative_step(
            [101],
            draft_model=StubDraftModel([201]),
            verifier=StubVerifier([]),
        )
