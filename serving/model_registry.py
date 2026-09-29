"""Loads and caches the local generation model used by the CPU baseline server.

Kept in its own module (rather than inline in server.py) so bench/load_test.py
and tests/ can import and warm the same singleton without booting a whole
FastAPI app, and so serving/quantize.py can reuse the exact same loading path
that the server uses -- otherwise a "quantized model is faster" measurement
would really just be measuring two different loading code paths.
"""
from __future__ import annotations

import os
import threading

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# A small (~135M param) instruction-tuned model. It is genuinely CPU-runnable
# in well under a second per short reply, which is the whole point: this repo
# benchmarks the *serving layer* (batching, streaming, quantization, scaling),
# and a small model lets every measurement here be real and reproducible on a
# laptop instead of asserted from a GPU cluster we don't have access to.
DEFAULT_MODEL_NAME = os.environ.get("SERVING_MODEL_NAME", "HuggingFaceTB/SmolLM2-135M-Instruct")

_lock = threading.Lock()
_cache: dict[str, tuple[PreTrainedModel, PreTrainedTokenizerBase]] = {}


def load_model(model_name: str = DEFAULT_MODEL_NAME) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Loads (or returns the cached) model + tokenizer pair.

    Thread-safe and idempotent: FastAPI's startup event and pytest fixtures
    can both call this without racing to load the model twice.
    """
    with _lock:
        if model_name in _cache:
            return _cache[model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Force float32: the checkpoint's default dtype (often bfloat16) is a
        # GPU-oriented choice that (a) PyTorch's native CPU dynamic
        # quantization in serving/quantize.py cannot operate on, and (b) has
        # no throughput benefit on CPU anyway, since CPUs don't have
        # bfloat16 tensor cores the way GPUs do.
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        model.eval()
        _cache[model_name] = (model, tokenizer)
        return model, tokenizer
