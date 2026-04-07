"""Tests for benchmark metric collection."""

from crossfire.utils.metrics import TABLE_HEADERS, BenchmarkResult


def test_benchmark_result_to_row_complete():
    result = BenchmarkResult(
        model="qwen3.5-27b",
        quant_type="TQ4_1S",
        context_size=8192,
        ablation_config="c5",
        perplexity=6.42,
        tokens_per_second=35.7,
        peak_memory_mb=18432.0,
        kv_compression="turbo3",
        distributed=True,
        ane_active=True,
        ane_role="draft_0.6b",
        total_power_watts=415.0,
        ane_power_watts=3.2,
        rdma_active=True,
    )
    row = result.to_row()
    assert len(row) == len(TABLE_HEADERS)
    assert row[0] == "C5"
    assert row[1] == "qwen3.5-27b"
    assert row[4] == "6.42"
    assert row[8] == "draft_0.6b"
    assert row[9] == "415"


def test_benchmark_result_to_row_baseline():
    result = BenchmarkResult(
        model="qwen3.5-27b",
        quant_type="Q8_0",
        context_size=8192,
        ablation_config="c0",
    )
    row = result.to_row()
    assert row[0] == "C0"
    assert row[4] == "—"
    assert row[8] == "idle"
    assert row[9] == "—"


def test_benchmark_result_has_timestamp():
    result = BenchmarkResult(
        model="test",
        quant_type="Q8_0",
        context_size=2048,
    )
    assert result.timestamp is not None
    assert "T" in result.timestamp


def test_benchmark_result_defaults():
    result = BenchmarkResult(
        model="test",
        quant_type="Q8_0",
        context_size=2048,
    )
    assert result.ablation_config == "c0"
    assert result.ane_active is False
    assert result.rdma_active is False
    assert result.distributed is False
