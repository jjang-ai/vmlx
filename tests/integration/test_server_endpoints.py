"""
Integration tests for vmlx-engine server endpoints.

Tests the full HTTP → server → response pipeline against a live model.
Requires --model flag to specify the test model.

Run:
    pytest tests/integration/ --model <path-to-small-model> -v
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests


# ─── Health & Models ──────────────────────────────────────────────────


def test_health(base_url):
    """Health endpoint returns status and model info."""
    r = requests.get(f"{base_url}/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_name"] is not None
    assert "last_request_time" in data


def test_models(base_url):
    """Models endpoint lists loaded model."""
    r = requests.get(f"{base_url}/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert len(data["data"]) >= 1
    assert data["data"][0]["id"] is not None


# ─── Chat Completions ────────────────────────────────────────────────


def test_chat_completion_nonstream(base_url, model_name):
    """Non-streaming chat completion returns valid response."""
    r = requests.post(f"{base_url}/v1/chat/completions", json={
        "model": model_name,
        "messages": [{"role": "user", "content": "Say hello in exactly 3 words."}],
        "max_tokens": 20,
        "stream": False,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["choices"][0]["message"]["content"]
    assert data["usage"]["completion_tokens"] > 0


def test_chat_completion_stream(base_url, model_name):
    """Streaming chat completion returns SSE chunks."""
    r = requests.post(f"{base_url}/v1/chat/completions", json={
        "model": model_name,
        "messages": [{"role": "user", "content": "Count to 3."}],
        "max_tokens": 30,
        "stream": True,
        "stream_options": {"include_usage": True},
    }, stream=True)
    assert r.status_code == 200

    chunks = []
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            chunks.append(json.loads(data_str))

    assert len(chunks) > 0
    # First chunk should have role
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    # Some chunks should have content
    content_chunks = [c for c in chunks if c.get("choices") and c["choices"][0]["delta"].get("content")]
    assert len(content_chunks) > 0


def test_chat_with_system(base_url, model_name):
    """Chat completion with system message works."""
    r = requests.post(f"{base_url}/v1/chat/completions", json={
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You only respond with the word 'yes'."},
            {"role": "user", "content": "Is the sky blue?"},
        ],
        "max_tokens": 10,
        "stream": False,
    })
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"]


# ─── Chat with Tools ────────────────────────────────────────────────


def test_chat_tools(base_url, model_name):
    """Chat completion with tool definitions should work (may or may not call tools)."""
    r = requests.post(f"{base_url}/v1/chat/completions", json={
        "model": model_name,
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            },
        }],
        "max_tokens": 100,
        "stream": False,
    })
    assert r.status_code == 200
    data = r.json()
    # Model should produce either content or tool_calls
    msg = data["choices"][0]["message"]
    assert msg.get("content") is not None or msg.get("tool_calls") is not None


# ─── Responses API ────────────────────────────────────────────────────


def test_responses_api_nonstream(base_url, model_name):
    """Responses API non-streaming works."""
    r = requests.post(f"{base_url}/v1/responses", json={
        "model": model_name,
        "input": "What is 2+2?",
        "stream": False,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["output"]
    # Should have at least one output item with text
    text_items = [o for o in data["output"] if o.get("type") == "message"]
    assert len(text_items) > 0


def test_responses_api_stream(base_url, model_name):
    """Responses API streaming works."""
    r = requests.post(f"{base_url}/v1/responses", json={
        "model": model_name,
        "input": "Say 'test'.",
        "stream": True,
    }, stream=True)
    assert r.status_code == 200

    events = []
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("event: "):
            events.append(line[7:])

    assert "response.created" in events
    assert "response.completed" in events or "response.done" in events


# ─── Anthropic Messages API ─────────────────────────────────────────


def test_anthropic_messages_nonstream(base_url, model_name):
    """Anthropic Messages API non-streaming works."""
    r = requests.post(f"{base_url}/v1/messages", json={
        "model": model_name,
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 20,
        "stream": False,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert len(data["content"]) > 0
    assert data["content"][0]["type"] == "text"
    assert data["usage"]["input_tokens"] > 0
    assert data["usage"]["output_tokens"] > 0


def test_anthropic_messages_stream(base_url, model_name):
    """Anthropic Messages API streaming returns proper SSE events."""
    r = requests.post(f"{base_url}/v1/messages", json={
        "model": model_name,
        "messages": [{"role": "user", "content": "Count to 2."}],
        "max_tokens": 30,
        "stream": True,
    }, stream=True)
    assert r.status_code == 200

    events = []
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("event: "):
            events.append(line[7:])

    assert "message_start" in events
    assert "content_block_start" in events
    assert "content_block_delta" in events
    assert "message_stop" in events


def test_anthropic_messages_with_system(base_url, model_name):
    """Anthropic Messages API with top-level system prompt."""
    r = requests.post(f"{base_url}/v1/messages", json={
        "model": model_name,
        "system": "Respond only with 'OK'.",
        "messages": [{"role": "user", "content": "Confirm."}],
        "max_tokens": 10,
        "stream": False,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["content"][0]["text"]


def test_anthropic_messages_with_tools(base_url, model_name):
    """Anthropic Messages API with tool definitions."""
    r = requests.post(f"{base_url}/v1/messages", json={
        "model": model_name,
        "messages": [{"role": "user", "content": "What time is it?"}],
        "tools": [{
            "name": "get_time",
            "description": "Get current time",
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        }],
        "max_tokens": 100,
        "stream": False,
    })
    assert r.status_code == 200
    data = r.json()
    # Should have content (either text or tool_use blocks)
    assert len(data["content"]) > 0


# ─── Cache ────────────────────────────────────────────────────────────


def test_cache_stats(base_url):
    """Cache stats endpoint works."""
    r = requests.get(f"{base_url}/v1/cache/stats")
    assert r.status_code == 200


# ─── SSE Keep-alive ──────────────────────────────────────────────────


def test_sse_keepalive_comment_format(base_url, model_name):
    """Streaming response should not include malformed SSE lines.
    (Keep-alive comments are ': keep-alive\\n\\n' — valid SSE.)
    """
    r = requests.post(f"{base_url}/v1/chat/completions", json={
        "model": model_name,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5,
        "stream": True,
    }, stream=True)
    assert r.status_code == 200

    for line in r.iter_lines(decode_unicode=True):
        if line:
            # Every non-empty line should be either:
            # - "data: ..." (SSE data)
            # - ": ..." (SSE comment, e.g., ": keep-alive")
            assert line.startswith("data: ") or line.startswith(": "), f"Unexpected SSE line: {line!r}"


# ─── Concurrent Requests ─────────────────────────────────────────────


def test_concurrent_requests(base_url, model_name):
    """Server handles 4 concurrent requests without errors."""
    def make_request(i):
        r = requests.post(f"{base_url}/v1/chat/completions", json={
            "model": model_name,
            "messages": [{"role": "user", "content": f"Say the number {i}."}],
            "max_tokens": 10,
            "stream": False,
        })
        return r.status_code

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(make_request, range(4)))

    assert all(code == 200 for code in results), f"Some requests failed: {results}"


# ─── Error Handling ──────────────────────────────────────────────────


def test_invalid_endpoint(base_url):
    """Unknown endpoint returns 404."""
    r = requests.get(f"{base_url}/v1/nonexistent")
    assert r.status_code in (404, 405)


def test_empty_messages(base_url, model_name):
    """Empty messages list returns 422."""
    r = requests.post(f"{base_url}/v1/chat/completions", json={
        "model": model_name,
        "messages": [],
        "max_tokens": 10,
    })
    assert r.status_code == 422
