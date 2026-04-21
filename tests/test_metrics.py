"""Tests for benchmark metric collection."""

from crossfire.utils.metrics import TABLE_HEADERS, BenchmarkResult


def test_benchmark_result_to_row_complete():
    result = BenchmarkResult(
        model="qwen3.5-27b",
        quant_type="TQ4_1S",
        context_size=8192,
        execution_policy="P5",
        ablation_config="c5",
        perplexity=6.42,
        tokens_per_second=35.7,
        ttft_ms=245.0,
        tokens_per_watt=0.31,
        peak_memory_mb=18432.0,
        kv_compression="triattention",
        distributed=True,
        ane_active=True,
        ane_role="draft_0.6b",
        acceptance_rate=0.72,
        total_power_watts=415.0,
        ane_power_watts=3.2,
        interconnect="usb4",
        interconnect_bytes=63_000_000,
    )
    row = result.to_row()
    assert len(row) == len(TABLE_HEADERS)
    # Label comes from ablation_config when set (uppercase)
    assert row[0] == "C5"
    assert row[1] == "qwen3.5-27b"
    assert row[4] == "6.42"
    assert row[5] == "35.7"
    assert row[6] == "245"
    assert row[7] == "0.310"
    assert row[10] == "draft_0.6b"
    assert row[11] == "0.72"
    assert row[13] == "415"


def test_benchmark_result_to_row_policy_label():
    """When no ablation_config is set, label comes from execution_policy."""
    result = BenchmarkResult(
        model="qwen3.5-27b",
        quant_type="Q8_0",
        context_size=8192,
        execution_policy="P1",
    )
    row = result.to_row()
    assert row[0] == "P1"
    assert row[4] == "—"
    assert row[10] == "idle"
    assert row[13] == "—"


def test_benchmark_result_to_row_baseline():
    result = BenchmarkResult(
        model="qwen3.5-27b",
        quant_type="Q8_0",
        context_size=8192,
        execution_policy="P0",
        ablation_config="c0",
    )
    row = result.to_row()
    assert row[0] == "C0"
    assert row[4] == "—"
    assert row[10] == "idle"
    assert row[13] == "—"


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
    assert result.execution_policy == "P0"
    assert result.ablation_config is None
    assert result.ane_active is False
    assert result.interconnect is None
    assert result.interconnect_bytes is None
    assert result.distributed is False
    assert result.flash_moe_active is False
    assert result.acceptance_rate is None
    assert result.flash_moe_hit_rate is None


def test_flash_moe_result():
    result = BenchmarkResult(
        model="qwen3.5-35b-a3b",
        quant_type="Q8_0",
        context_size=16384,
        execution_policy="P6",
        ablation_config="c1",
        tokens_per_second=53.0,
        flash_moe_hit_rate=0.82,
        flash_moe_active=True,
        distributed=True,
        interconnect="wifi",
    )
    row = result.to_row()
    assert row[0] == "C1"
    assert row[5] == "53.0"
    assert row[12] == "0.82"


def test_interconnect_field():
    """interconnect and interconnect_bytes round-trip correctly."""
    result = BenchmarkResult(
        model="qwen3.5-27b",
        quant_type="TQ4_1S",
        context_size=8192,
        execution_policy="P5",
        distributed=True,
        interconnect="usb4",
        interconnect_bytes=63_000_000,
    )
    assert result.interconnect == "usb4"
    assert result.interconnect_bytes == 63_000_000
