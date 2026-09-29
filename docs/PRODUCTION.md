# Taking this to production

This repo's `serving/server.py` is a CPU-runnable baseline, deliberately kept
simple so its own benchmark numbers are real and reproducible on a laptop.
This document is the honest bridge from that baseline to an actual
production deployment -- what changes, what stays the same, and where the
baseline's own measurements already tell you why each change matters.

## 1. Swap the serving engine, keep the API contract

Do not deploy `serving/server.py` to production. Its whole value is as a
*measurement target* for this repo -- its naive per-request generation loop
is precisely the bottleneck the rest of this document explains how to fix.

Deploy `serving/server_vllm.py`'s engine (or, more commonly in practice,
vLLM's own `vllm.entrypoints.openai.api_server`) instead. Because both expose
an OpenAI-Chat-Completions-shaped `/v1/chat/completions` endpoint, this is a
base-URL change for client code, not a client rewrite -- see ADR 001.

**External benchmark citation** (we do not have a GPU in this build
environment, so we cite rather than fabricate a number): the original
PagedAttention paper reports vLLM improving throughput 2-4x over
FasterTransformer/Orca at matched latency (Kwon et al., *Efficient Memory
Management for Large Language Model Serving with PagedAttention*, SOSP 2023 —
https://dl.acm.org/doi/10.1145/3600006.3613165), and vLLM's own project blog
post covers the same result
(https://vllm.ai/blog/2023-06-20-vllm). Treat any larger multiplier you see
quoted elsewhere (e.g. "24x") skeptically until you check what baseline and
workload it's measured against -- it is very sensitive to sequence length,
batch size, and what it's being compared to.

## 2. Our own measured numbers (CPU baseline, this repo)

These are what this repo's own scripts measured on the build machine — a
baseline for understanding the *shape* of the problem, not a production SLA
for any real deployment:

**Quantization** (`serving/quantize.py`, `HuggingFaceTB/SmolLM2-135M-Instruct`, greedy decode, 32 new tokens):

| | FP32 | Dynamic INT8 | Change |
|---|---|---|---|
| Model size (state_dict) | 513.2 MB | 236.6 MB | **-53.9%** |
| Median single-request latency | 2.050 s | 3.763 s | **+83.5%** (slower) |

Size dropped by half, as expected. Latency got *worse*, which is the
counterintuitive-but-real result explained in ADR 003: dynamic quantization
pays a fixed per-Linear-layer quantize/dequantize cost on every forward pass,
and at batch size 1 on this CPU's INT8 kernel that overhead outweighs the
compute savings. **Do not generalize "quantize it, it gets faster" from
this** -- it's workload- and hardware-dependent. The tradeoff table below
treats size and latency as two independently-tunable axes for exactly this
reason.

**Concurrency sweep** (`bench/load_test.py`, same model, 24 new tokens, 8 requests per level, non-streaming):

| Concurrency | p50 latency | p95 latency | p99 latency | Throughput |
|---|---|---|---|---|
| 1 | 1.546 s | 1.775 s | 1.775 s | 0.658 req/s |
| 2 | 2.995 s | 3.357 s | 3.357 s | 0.660 req/s |
| 4 | 6.001 s | 6.084 s | 6.084 s | 0.662 req/s |
| 8 | 11.542 s | 11.544 s | 11.544 s | 0.693 req/s |

Throughput is flat (~0.66 req/s) regardless of concurrency, while p50 latency
scales almost exactly linearly with it (1.5s -> 3s -> 6s -> 11.5s for
1 -> 2 -> 4 -> 8 concurrent requests). That's the direct, measured signature
of "no batching": each additional concurrent request just waits behind the
ones ahead of it: this is exactly the problem continuous batching (ADR 001)
exists to solve, and it is why we did not try to fix it by hand in this
baseline.

## 3. Cost/latency tradeoff table (batch size vs latency)

Reasoning from the measurements above plus standard continuous-batching
behavior (not separately re-measured on a GPU here): every request added to
a running batch adds a small, roughly-constant amount of per-step compute
(the batch's matmuls get wider), while *not* adding a full serialized
generation's worth of wall-clock time the way our naive baseline does. The
real production tuning question is where to cap `max_num_seqs`
(`serving/server_vllm.py`):

| `max_num_seqs` | Per-request latency under load | GPU utilization | When to use |
|---|---|---|---|
| Low (e.g. 8-16) | Best-case, closest to unloaded latency | Often under-utilized | Strict p99 latency SLOs, low/predictable traffic |
| Medium (e.g. 64-128) | Moderate increase under peak load | Good | Most production chat/assist workloads |
| High (e.g. 256+) | Noticeably higher tail latency under peak | Near-saturated | Maximizing throughput/cost-per-token for batch or non-interactive workloads |

Tune this against a real traffic replay and your actual p95/p99 latency
budget -- it is not a constant that transfers between models, GPUs, or
traffic patterns.

## 4. Kubernetes deployment (`deploy/k8s/`)

- **`deployment.yaml`**: `RollingUpdate` with `maxUnavailable: 0` so a
  deploy never drops total serving capacity mid-rollout — important because
  model load takes tens of seconds, so a naive rolling update could
  temporarily reduce capacity below demand. `startupProbe` gives that load
  window room before liveness checks kick in and start a restart loop.
- **`service.yaml`**: a plain `ClusterIP` in front of the pod set; put a
  real ingress/gateway in front of this for external traffic and TLS
  termination — intentionally out of scope here since it's
  environment-specific.
- **`hpa.yaml`**: scales on a **custom metric** (in-flight request queue
  depth), not CPU/memory. CPU/memory utilization is close to meaningless for
  a GPU-bound serving pod: it can be GPU-saturated with queueing latency
  climbing while CPU utilization looks idle. Requires Prometheus +
  prometheus-adapter wired up to expose `llm_serving_queue_depth` through
  the `custom.metrics.k8s.io` API — cluster-specific setup, documented here
  rather than templated, since it varies by Prometheus deployment.
  - **Scale-up** is fast (0s stabilization window): under-provisioning
    during a spike directly costs users latency-SLO breaches.
  - **Scale-down** is slow (10 minute stabilization window): a new pod
    costs real model-load time before it's useful, so flapping down and
    immediately back up on a noisy dip in traffic is worse than staying warm
    a bit longer than strictly necessary.

## 5. Autoscaling cooldown tuning for bursty traffic

Bursty traffic (e.g. business-hours-only usage, or a spike right after a
product launch/announcement) is the case the scale-down cooldown above is
specifically tuned for. Two failure modes to watch for and tune against:

- **Too fast a scale-down** causes "sawtooth" behavior: traffic dips for a
  minute, HPA removes a pod, traffic ticks back up, users on that removed
  pod's slice of traffic eat a full cold-start (model load) before serving
  resumes. The `600s` `stabilizationWindowSeconds` above is a starting point
  — tune it against your actual traffic's real inter-burst gap, measured
  from logs, not guessed.
- **Too slow a scale-up** during a genuine spike causes queue depth (and
  therefore latency) to climb before new capacity comes online. Keeping
  `stabilizationWindowSeconds: 0` on scale-up, and provisioning `minReplicas`
  high enough to absorb the time-to-first-new-pod-Ready window, is the
  mitigation.

## 6. Model warm-pool / cold-start mitigation

Model load (weights from disk/object storage into GPU memory) is the
dominant cold-start cost, not container start. Two standard mitigations,
neither implemented in this repo's manifests but worth naming explicitly:

- **Keep `minReplicas` warm** at a level that absorbs your normal traffic
  floor, so autoscaling only ever needs to add *marginal* capacity, not
  bootstrap from zero.
- **Pre-bake weights into the container image** (or a fast local NVMe cache
  warmed by a node-level DaemonSet) rather than pulling from remote object
  storage on every pod start, if cold-start time is itself a problem at your
  scale-up cadence.

## 7. Canary + shadow-traffic rollout for a new serving image

Before shifting production traffic to a new serving image (new model
version, new vLLM version, new `max_num_seqs` tuning):

1. **Shadow traffic**: mirror a copy of real production requests to the
   candidate deployment without returning its responses to users, and
   compare its latency distribution and (if you have an automated quality
   check) output quality against the current production deployment.
2. **Canary**: route a small, monitored percentage of real traffic (e.g. 5%)
   to the candidate, watching p95/p99 latency and error rate against
   pre-declared thresholds, before progressively increasing the percentage.
3. **Automatic rollback trigger**: wire canary latency/error-rate regression
   to an automatic rollback, not just an alert — a person noticing and
   reacting to a dashboard is meaningfully slower than a scripted threshold
   check.

This pairs directly with the evaluation-gate pattern described in this
account's golden-dataset evaluation work: a new model/adapter should clear
an offline eval bar *before* it's even eligible for canary traffic, and clear
a live canary bar before it gets the rest.
