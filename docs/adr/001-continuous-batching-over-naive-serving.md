# ADR 001: Continuous batching (vLLM) over naive per-request serving at scale

## Status
Accepted (for the GPU production path; the CPU baseline in this repo intentionally does NOT do this -- see "What we did instead" below).

## Context
`serving/server.py` handles one generation request at a time: a request comes
in, a background thread calls `model.generate()`, and the next request has to
wait its turn (or run on its own thread and fight for CPU). Its own load-test
results (see the README) show this directly: throughput stays essentially
flat as concurrency increases from 1 to 8 -- each additional concurrent
request just adds queueing delay, because nothing about the serving loop
shares work across requests.

In a real deployment this design breaks down badly. LLM decoding is
memory-bandwidth-bound per step, not compute-bound -- a GPU running one
request's `generate()` call is leaving most of its throughput on the table,
because it could be doing the same memory-bound attention/matmul work for
several requests' next-token step in the same pass, for barely more cost
than doing it for one.

## Decision
For real GPU production serving, use vLLM (`serving/server_vllm.py`), which
implements continuous batching: instead of "one `generate()` call, start to
finish, per request," the scheduler maintains one running batch and
inserts/evicts individual requests' next-token steps into it on every
iteration. A request that arrives mid-batch doesn't wait for the whole batch
to finish -- it joins the next iteration. This is combined with
PagedAttention, which manages each request's KV cache in fixed-size,
non-contiguous "pages" (the same idea as OS virtual memory paging) instead of
one large contiguous pre-allocated buffer per request, which is what lets a
GPU hold far more concurrent sequences in memory at once.

The original PagedAttention paper (Kwon et al., SOSP 2023) reports 2-4x
throughput improvement over FasterTransformer/Orca at the same latency; see
docs/PRODUCTION.md for the citation. We do not restate a bigger number here
without a link, because a number without a reproducible source is not
evidence.

## What we did instead, in this repo's CPU baseline
`serving/server.py` deliberately does NOT implement batching. It exists to
give the rest of this repo (load testing, quantization measurement) a real,
runnable target on a laptop, and to make the *problem* (flat throughput under
concurrency) directly observable rather than asserted. Building a
hand-rolled batching scheduler would be reinventing a worse, unmaintained
version of what vLLM already does well -- so the production answer is "adopt
vLLM," not "improve this file."

## Consequences
- Migrating from the CPU baseline to vLLM in production is mostly a
  deployment change, not a client rewrite: `serving/server_vllm.py`'s
  `generate_batch` and vLLM's own OpenAI-compatible server expose the same
  request/response shape as `serving/server.py`'s `/v1/chat/completions`.
- vLLM requires a CUDA GPU; it is listed in `requirements-gpu.txt`, not
  `requirements.txt`, and is not installed in CI.
- `max_num_seqs` and `gpu_memory_utilization` become real production tuning
  knobs that need to be set against a real traffic profile, not left at
  defaults (see docs/PRODUCTION.md).
