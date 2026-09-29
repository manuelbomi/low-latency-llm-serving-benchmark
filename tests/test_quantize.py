"""Tests for serving/quantize.py -- correctness of the measurement, not a
claim that quantization always wins (it doesn't, on this hardware; see the
README and docs/adr/003-quantization-tradeoffs.md for the real, measured,
and slightly counterintuitive result).
"""
from __future__ import annotations

from serving.quantize import run_comparison


def test_run_comparison_produces_valid_measurements():
    result = run_comparison(max_new_tokens=6)
    assert result["fp32_size_mb"] > 0
    assert result["int8_size_mb"] > 0
    # INT8 dynamic quantization must shrink Linear-layer weights -- this is
    # a structural guarantee of the technique, unlike the latency direction,
    # which is workload/hardware dependent and NOT asserted here.
    assert result["int8_size_mb"] < result["fp32_size_mb"]
    assert result["fp32_median_latency_s"] > 0
    assert result["int8_median_latency_s"] > 0
