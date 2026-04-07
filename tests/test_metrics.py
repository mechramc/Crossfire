"""Tests for benchmark metric collection."""

from crossfire.utils.metrics import TABLE_HEADERS, BenchmarkResult


def test_benchmark_result_to_row_complete():
    result = BenchmarkResult(
        model="qwen3.5-27b",
        quant_type="TQ4_1S",
        context_size=8192,
        perplexity=6.42,
        tokens_per_second=35.7,
        peak_memory_mb=18432.0,
        kv_compression="turbo3",
        distributed=True,
    )
    row = result.to_row()
    assert len(row) == len(TABLE_HEADERS)
    assert row[0] == "qwen3.5-27b"
    assert row[3] == "6.42"
    assert row[7] == "yes"


def test_benchmark_result_to_row_missing_values():
    result = BenchmarkResult(
        model="phi-4-14b",
        quant_type="Q8_0",
        context_size=2048,
    )
    row = result.to_row()
    assert row[3] == "\u2014"
    assert row[4] == "\u2014"
    assert row[7] == "no"


def test_benchmark_result_has_timestamp():
    result = BenchmarkResult(
        model="test",
        quant_type="Q8_0",
        context_size=2048,
    )
    assert result.timestamp is not None
    assert "T" in result.timestamp
