# ADR 003: Quantization is a real tradeoff, not a free win -- and our own measurement proves it

## Status
Accepted, with an intentionally counterintuitive result kept in, not smoothed over.

## Context
`serving/quantize.py` runs PyTorch's native CPU dynamic INT8 quantization
(`torch.ao.quantization.quantize_dynamic`) against
`HuggingFaceTB/SmolLM2-135M-Instruct` on this repo's build machine and
measures both model size and single-request generation latency, before and
after. The measured result (see README for the exact numbers) is:

- Model size: roughly halved (INT8 Linear-layer weights vs FP32).
- Single-request latency: **slower**, not faster, after quantization.

This is a real, reproduced result on this hardware -- not a bug we're
covering up. It happens because dynamic quantization pays a real, fixed
per-call cost (quantizing the FP32 activations to INT8 immediately before
each Linear layer's matmul, then dequantizing the result) on every forward
pass. For a decode-bound autoregressive loop generating one token at a time
(batch size 1), that per-call overhead is paid on every single token step,
and on this CPU's INT8 GEMM kernel it outweighs the compute savings from the
smaller weight representation. The technique's real win -- memory bandwidth
reduction -- shows up more reliably at larger batch sizes or on hardware
with quantization-aware kernels tuned for it, neither of which describes
"one request at a time on a laptop CPU."

## Decision
We report the measured result honestly, including the direction that
contradicts the "quantization = faster" intuition, and treat model **size**
reduction (real, structural, and unconditionally true for this technique) as
the trustworthy part of this specific measurement, while treating **latency**
direction as hardware- and workload-dependent and NOT something to assert
generally from one CPU/batch-size-1 data point.

For real production quantization wins, the right tools are GPU-native
quantized-kernel approaches -- AWQ, GPTQ, or bitsandbytes 4-bit/8-bit -- which
require a CUDA GPU and are documented (not benchmarked, since this repo has
no GPU available) in docs/PRODUCTION.md as the production path, alongside a
citation instead of a made-up number.

## Consequences
- `tests/test_quantize.py` asserts the one universally-true property (size
  reduction), and deliberately does NOT assert a latency direction, because
  that assertion would be hardware-dependent and could make the test suite
  lie about what the code guarantees.
- Anyone reading this repo's numbers and expecting "quantize it and it gets
  faster" as a universal law will instead see a concrete counter-example and
  the reasoning for why it happened -- which is a more useful lesson than a
  clean number would have been.
