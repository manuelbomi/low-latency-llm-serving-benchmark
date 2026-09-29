"""Real, CPU-runnable dynamic INT8 quantization, with real before/after measurements.

We deliberately do NOT use bitsandbytes 4-bit/8-bit here: bitsandbytes'
quantized kernels are CUDA-only (its CPU support is experimental/incomplete
at the time of writing), so "running" it on a CPU dev box would either fail
outright or silently fall back to full precision -- either way, any number
we reported would not be a real measurement of quantization's effect. See
docs/PRODUCTION.md for where bitsandbytes/AWQ/GPTQ *do* apply (GPU serving).

Instead this module uses `torch.ao.quantization.quantize_dynamic`, PyTorch's
native CPU dynamic quantization: it replaces `nn.Linear` weights with INT8
tensors (activations stay float, quantized per-inference) and is fully
supported on stock CPU PyTorch, which means the size and latency numbers this
module prints are genuinely measured, not asserted.

Run it directly:
    python -m serving.quantize
"""
from __future__ import annotations

import io
import time

import torch

from serving.model_registry import DEFAULT_MODEL_NAME, load_model


def model_size_mb(model: torch.nn.Module) -> float:
    """Serializes the model's state_dict to an in-memory buffer and measures its size.

    This is a more honest measurement than "count parameters * 4 bytes,"
    because it reflects what quantize_dynamic actually produced (INT8 weight
    tensors for Linear layers, float elsewhere) rather than an assumption
    about the dtype.
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024 * 1024)


def measure_single_request_latency(model, tokenizer, prompt: str, max_new_tokens: int, n_runs: int = 5) -> float:
    """Median wall-clock latency (seconds) for a single greedy generation, over n_runs."""
    chat_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model.generate(**chat_input, max_new_tokens=max_new_tokens, do_sample=False)
        latencies.append(time.perf_counter() - start)
    latencies.sort()
    return latencies[len(latencies) // 2]


def run_comparison(
    model_name: str = DEFAULT_MODEL_NAME, prompt: str = "Explain what caching does in one sentence.", max_new_tokens: int = 32
) -> dict:
    fp32_model, tokenizer = load_model(model_name)

    fp32_size = model_size_mb(fp32_model)
    fp32_latency = measure_single_request_latency(fp32_model, tokenizer, prompt, max_new_tokens)

    # quantize_dynamic returns a NEW module; it does not mutate fp32_model in
    # place, so both models remain independently usable/comparable.
    int8_model = torch.ao.quantization.quantize_dynamic(fp32_model, {torch.nn.Linear}, dtype=torch.qint8)
    int8_size = model_size_mb(int8_model)
    int8_latency = measure_single_request_latency(int8_model, tokenizer, prompt, max_new_tokens)

    return {
        "model_name": model_name,
        "fp32_size_mb": round(fp32_size, 2),
        "int8_size_mb": round(int8_size, 2),
        "size_reduction_pct": round(100 * (1 - int8_size / fp32_size), 1),
        "fp32_median_latency_s": round(fp32_latency, 4),
        "int8_median_latency_s": round(int8_latency, 4),
        "latency_change_pct": round(100 * (int8_latency - fp32_latency) / fp32_latency, 1),
        "max_new_tokens": max_new_tokens,
    }


if __name__ == "__main__":
    result = run_comparison()
    for key, value in result.items():
        print(f"{key}: {value}")
