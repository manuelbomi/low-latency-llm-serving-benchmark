"""Unit tests for bench/load_test.py's pure statistics logic.

Deliberately does NOT spin up a live server here (that's what the README's
"reproduce it yourself" walkthrough and the CI smoke-test job are for) --
this test suite verifies the percentile/throughput math is correct in
isolation, using a LevelResult built by hand.
"""
from __future__ import annotations

from bench.load_test import LevelResult


def test_percentile_p50_and_p99_on_known_data():
    result = LevelResult(concurrency=1, n_requests=10, latencies_s=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], wall_clock_s=10)
    assert result.percentile(0.50) == 6  # index int(10*0.5)=5 -> sorted[5]=6
    assert result.percentile(0.99) == 10  # clamps to the last element


def test_throughput_rps():
    result = LevelResult(concurrency=4, n_requests=8, latencies_s=[1] * 8, wall_clock_s=4.0)
    assert result.throughput_rps == 2.0


def test_throughput_rps_zero_wall_clock_is_safe():
    result = LevelResult(concurrency=1, n_requests=0, latencies_s=[], wall_clock_s=0.0)
    assert result.throughput_rps == 0.0
