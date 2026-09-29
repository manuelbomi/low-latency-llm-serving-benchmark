"""The production GPU serving path: vLLM with continuous batching.

serving/server.py (the baseline in this repo) generates for one request at a
time on one thread. That is fine for a CPU demo at low concurrency, and it is
*exactly* the design that falls over in real production traffic: every extra
concurrent request adds a full generation's worth of queueing delay, because
nothing is shared across requests.

vLLM's core idea -- continuous batching with PagedAttention -- is different:
instead of "one generate() call per request," the scheduler maintains a
single running batch and inserts/evicts individual requests's *token steps*
into it on every iteration, so GPU compute is shared across concurrently
in-flight requests instead of serialized. That is what lets a real deployment
serve dozens of concurrent users off one GPU at a small fraction of the
per-request latency a naive loop would give you. See docs/PRODUCTION.md for a
cited external throughput comparison -- this repo does not fabricate a GPU
number it never measured.

This module is written to be *correct*, not to be run in this sandbox: vLLM
requires a CUDA GPU and is not installed here (see requirements-gpu.txt). The
import is guarded so the rest of the repo (tests, CI, the CPU server) never
depends on vLLM being present.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - only for type checkers, never imported at runtime
    from vllm import LLM, SamplingParams

try:
    from vllm import LLM, SamplingParams  # type: ignore

    VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover - expected on CPU-only dev/CI machines
    VLLM_AVAILABLE = False


def build_engine(model_name: str, *, max_num_seqs: int = 256, gpu_memory_utilization: float = 0.90):
    """Builds a vLLM engine configured for continuous batching.

    max_num_seqs is the key production knob: it caps how many sequences the
    scheduler will keep in the running batch at once. Set it too low and you
    under-utilize the GPU under load; set it too high and per-token latency
    for every in-flight request degrades because the batch is doing more
    total work per step. In production this is tuned against a real traffic
    replay (see docs/PRODUCTION.md's batch-size/latency table), not guessed.
    """
    if not VLLM_AVAILABLE:
        raise RuntimeError(
            "vllm is not installed. This module documents the GPU production "
            "serving path; install `vllm` on a CUDA machine to actually run "
            "it (see requirements-gpu.txt)."
        )
    return LLM(
        model=model_name,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        # Continuous batching + PagedAttention are vLLM's defaults; no extra
        # flag is needed to enable them, which is precisely why vLLM is the
        # low-effort/high-payoff production upgrade over a hand-rolled batching
        # loop.
    )


def generate_batch(engine, prompts: list[str], *, max_tokens: int = 128, temperature: float = 0.0) -> list[str]:
    """Submits many prompts at once; vLLM's scheduler interleaves them internally.

    Note this is the *offline batch* API for illustration. The production
    online-serving path is `vllm.entrypoints.openai.api_server`, which exposes
    the same OpenAI-compatible /v1/chat/completions shape that
    serving/server.py hand-rolls here -- so a real migration from this repo's
    CPU baseline to vLLM in production is a client base-URL change, not a
    client rewrite.
    """
    if not VLLM_AVAILABLE:
        raise RuntimeError("vllm is not installed; see build_engine() docstring.")
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
    outputs = engine.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]
