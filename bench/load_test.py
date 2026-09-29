"""A small, dependency-light concurrency-sweep load generator.

We deliberately did not reach for Locust here: Locust is a fine choice for a
long-lived load-testing setup with a web UI, but for a repo whose whole point
is "here is exactly how these numbers were produced, go reproduce them
yourself," a ~100-line asyncio script that you can read start to finish in
two minutes is more honest and more portable (no extra service to run).

What it measures, at each concurrency level: p50/p95/p99 end-to-end request
latency and achieved throughput (completed requests / wall-clock second). It
talks to the *non-streaming* `/v1/chat/completions` endpoint (`stream:
false`) so that "request latency" has an unambiguous single meaning (time to
full response) -- streaming's time-to-first-token is a different, also
important, metric, which serving/server.py already reports per-request in
its SSE `usage` field.

Run it against a running server:
    uvicorn serving.server:app --port 8000 &
    python -m bench.load_test --base-url http://127.0.0.1:8000 --concurrency 1 2 4 8 --requests-per-level 12
"""
from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass, field

import httpx

DEFAULT_PROMPT = "In one short sentence, what does a load balancer do?"


@dataclass
class LevelResult:
    concurrency: int
    n_requests: int
    latencies_s: list[float] = field(default_factory=list)
    wall_clock_s: float = 0.0

    @property
    def throughput_rps(self) -> float:
        return self.n_requests / self.wall_clock_s if self.wall_clock_s > 0 else 0.0

    def percentile(self, pct: float) -> float:
        if not self.latencies_s:
            return 0.0
        ordered = sorted(self.latencies_s)
        idx = min(int(len(ordered) * pct), len(ordered) - 1)
        return ordered[idx]


async def _one_request(client: httpx.AsyncClient, base_url: str, max_tokens: int) -> float:
    payload = {
        "messages": [{"role": "user", "content": DEFAULT_PROMPT}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    start = time.perf_counter()
    resp = await client.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120.0)
    resp.raise_for_status()
    return time.perf_counter() - start


async def run_level(base_url: str, concurrency: int, n_requests: int, max_tokens: int) -> LevelResult:
    result = LevelResult(concurrency=concurrency, n_requests=n_requests)
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(client: httpx.AsyncClient) -> float:
        async with semaphore:
            return await _one_request(client, base_url, max_tokens)

    async with httpx.AsyncClient() as client:
        start = time.perf_counter()
        latencies = await asyncio.gather(*[bounded_request(client) for _ in range(n_requests)])
        result.wall_clock_s = time.perf_counter() - start
    result.latencies_s = list(latencies)
    return result


async def run_sweep(base_url: str, concurrency_levels: list[int], requests_per_level: int, max_tokens: int) -> list[LevelResult]:
    results = []
    for level in concurrency_levels:
        result = await run_level(base_url, level, requests_per_level, max_tokens)
        results.append(result)
    return results


def print_results(results: list[LevelResult]) -> None:
    header = f"{'concurrency':>11} | {'p50 (s)':>8} | {'p95 (s)':>8} | {'p99 (s)':>8} | {'throughput (req/s)':>19}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.concurrency:>11} | {r.percentile(0.50):>8.3f} | {r.percentile(0.95):>8.3f} | "
            f"{r.percentile(0.99):>8.3f} | {r.throughput_rps:>19.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--requests-per-level", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=24)
    args = parser.parse_args()

    results = asyncio.run(run_sweep(args.base_url, args.concurrency, args.requests_per_level, args.max_tokens))
    print_results(results)


if __name__ == "__main__":
    main()
