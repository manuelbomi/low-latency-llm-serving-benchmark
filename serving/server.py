"""CPU-runnable baseline LLM serving server.

This is deliberately the *simple* baseline in this repo: one process, one
model, naive per-request generation, no continuous batching. It exists so
that (a) the rest of the benchmark harness has something real to load-test,
and (b) the README's latency/throughput numbers are honest, reproducible
measurements against real running code -- not numbers copied from a vendor
blog post.

It exposes an OpenAI-Chat-Completions-*shaped* endpoint (not a full
implementation of the spec) so that bench/load_test.py, and any real OpenAI
client library, can talk to it with minimal adaptation. Responses stream as
Server-Sent Events (SSE), token by token, which is what makes a chat UI feel
fast even when the *total* generation time is unchanged -- see the "why
streaming" ADR in docs/adr/ for the reasoning.

Run it with:
    uvicorn serving.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import TextIteratorStreamer

from serving.model_registry import DEFAULT_MODEL_NAME, load_model

app = FastAPI(title="low-latency-llm-serving-benchmark", version="0.1.0")

# Loaded once at import time (module-level singleton), not per-request. This
# mirrors real serving practice -- model load is a multi-second, memory-heavy
# operation that must happen exactly once at process startup, never inside a
# request handler.
_model, _tokenizer = load_model()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL_NAME)
    messages: list[ChatMessage]
    max_tokens: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stream: bool = Field(default=True)


def _generate_stream(messages: list[ChatMessage], max_tokens: int, temperature: float):
    """Runs generation in a background thread and yields tokens as they land.

    TextIteratorStreamer is the standard transformers primitive for this: the
    generate() call itself is blocking, so it has to run off the main thread
    or the whole server would stall for every other request while it runs.
    This is exactly the naive-concurrency problem that a real production
    server (see serving/server_vllm.py and docs/PRODUCTION.md) solves with
    continuous batching instead of one-thread-per-request.
    """
    chat_input = _tokenizer.apply_chat_template(
        [m.model_dump() for m in messages],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    streamer = TextIteratorStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **chat_input,
        max_new_tokens=max_tokens,
        do_sample=temperature > 0.0,
        temperature=max(temperature, 1e-5),
        streamer=streamer,
    )
    thread = threading.Thread(target=_model.generate, kwargs=generation_kwargs)
    thread.start()
    yield from streamer
    thread.join()


async def _sse_event_generator(
    messages: list[ChatMessage], max_tokens: int, temperature: float, request_id: str
) -> AsyncGenerator[str, None]:
    start = time.perf_counter()
    first_token_at: float | None = None
    token_count = 0
    for token_text in _generate_stream(messages, max_tokens, temperature):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        token_count += 1
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "choices": [{"delta": {"content": token_text}, "index": 0, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    total_time = time.perf_counter() - start
    ttft = (first_token_at - start) if first_token_at else total_time
    final = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "completion_tokens": token_count,
            "time_to_first_token_s": round(ttft, 4),
            "total_time_s": round(total_time, 4),
        },
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    request_id = f"chatcmpl-{time.time_ns()}"
    if not req.stream:
        # Non-streaming path: drain the generator server-side and return one
        # JSON body, matching what a batch-oriented client (or a load test
        # that wants a single wall-clock number instead of a token stream)
        # expects.
        start = time.perf_counter()
        text = "".join(_generate_stream(req.messages, req.max_tokens, req.temperature))
        elapsed = time.perf_counter() - start
        return {
            "id": request_id,
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": text}, "index": 0, "finish_reason": "stop"}],
            "usage": {"total_time_s": round(elapsed, 4)},
        }
    return StreamingResponse(
        _sse_event_generator(req.messages, req.max_tokens, req.temperature, request_id),
        media_type="text/event-stream",
    )


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "model": DEFAULT_MODEL_NAME}
