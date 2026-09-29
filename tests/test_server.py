"""Tests for the FastAPI serving layer.

These run against the real small model (no mocking of generation) -- at
~135M parameters it loads and generates fast enough that "real" and "fast"
aren't in tension here, which is the whole reason this repo picked that
model size. What IS deliberately kept tiny is max_tokens, to keep CI fast.
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from serving.server import app

client = TestClient(app)


def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model" in body


def test_chat_completions_non_streaming():
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Say OK."}],
            "max_tokens": 4,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(body["choices"][0]["message"]["content"], str)
    assert body["usage"]["total_time_s"] >= 0


def test_chat_completions_streaming_emits_sse_chunks_and_done():
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Say OK."}],
            "max_tokens": 4,
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        lines = [line for line in resp.iter_lines() if line.startswith("data: ")]

    assert lines, "expected at least one SSE data line"
    assert lines[-1] == "data: [DONE]"

    # Every non-terminal line before [DONE] must be valid JSON with the
    # OpenAI-chunk shape our server promises.
    for line in lines[:-1]:
        payload = json.loads(line[len("data: ") :])
        assert payload["object"] == "chat.completion.chunk"
        assert "choices" in payload

    # The second-to-last chunk carries usage/timing metadata.
    final_chunk = json.loads(lines[-2][len("data: ") :])
    assert "usage" in final_chunk
    assert final_chunk["usage"]["time_to_first_token_s"] >= 0
    assert final_chunk["usage"]["total_time_s"] >= final_chunk["usage"]["time_to_first_token_s"]


def test_max_tokens_is_bounded():
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 99999, "stream": False},
    )
    assert resp.status_code == 422  # exceeds Field(le=512)
