# SPDX-License-Identifier: Apache-2.0
"""Ollama adapter parity tests."""

from __future__ import annotations

import json


def test_ollama_generate_default_uses_chat_template_request_shape():
    from vmlx_engine.api.ollama_adapter import ollama_generate_to_openai_chat

    req = ollama_generate_to_openai_chat(
        {
            "model": "zaya",
            "system": "Be terse.",
            "prompt": "What is the capital of France?",
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "num_predict": 16,
                "temperature": 0,
                "top_p": 1,
                "repeat_penalty": 1.1,
            },
        }
    )

    assert req["messages"] == [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    assert req["stream"] is False
    assert req["max_tokens"] == 16
    assert req["temperature"] == 0
    assert req["top_p"] == 1
    assert req["repetition_penalty"] == 1.1
    assert req["enable_thinking"] is False
    assert req["response_format"] == {"type": "json_object"}


def test_ollama_chat_accepts_enable_thinking_extension():
    from vmlx_engine.api.ollama_adapter import ollama_chat_to_openai

    req = ollama_chat_to_openai(
        {
            "model": "zaya",
            "messages": [{"role": "user", "content": "hi"}],
            "enable_thinking": False,
        }
    )

    assert req["enable_thinking"] is False


def test_ollama_native_think_beats_enable_thinking_extension():
    from vmlx_engine.api.ollama_adapter import ollama_chat_to_openai

    req = ollama_chat_to_openai(
        {
            "model": "zaya",
            "messages": [{"role": "user", "content": "hi"}],
            "think": True,
            "enable_thinking": False,
        }
    )

    assert req["enable_thinking"] is True


def test_ollama_chat_drops_reasoning_effort_when_thinking_off():
    from vmlx_engine.api.ollama_adapter import ollama_chat_to_openai

    req = ollama_chat_to_openai(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}],
            "think": False,
            "reasoning_effort": "max",
        }
    )

    assert req["enable_thinking"] is False
    assert "reasoning_effort" not in req


def test_ollama_generate_chat_accepts_enable_thinking_extension():
    from vmlx_engine.api.ollama_adapter import ollama_generate_to_openai_chat

    req = ollama_generate_to_openai_chat(
        {"model": "zaya", "prompt": "hi", "enable_thinking": False}
    )

    assert req["enable_thinking"] is False


def test_ollama_generate_chat_drops_reasoning_effort_when_template_kwargs_disable_thinking():
    from vmlx_engine.api.ollama_adapter import ollama_generate_to_openai_chat

    req = ollama_generate_to_openai_chat(
        {
            "model": "qwen",
            "prompt": "hi",
            "reasoning_effort": "high",
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )

    assert req["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in req


def test_ollama_generate_raw_keeps_completion_request_shape():
    from vmlx_engine.api.ollama_adapter import ollama_generate_to_openai

    req = ollama_generate_to_openai(
        {
            "model": "base",
            "prompt": "raw text",
            "stream": False,
            "options": {"num_predict": 4, "temperature": 0},
        }
    )

    assert req["prompt"] == "raw text"
    assert "messages" not in req
    assert req["max_tokens"] == 4
    assert req["temperature"] == 0


def test_chat_response_converts_to_ollama_generate_shape():
    from vmlx_engine.api.ollama_adapter import (
        openai_chat_response_to_ollama_generate,
    )

    out = openai_chat_response_to_ollama_generate(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "Paris",
                        "reasoning_content": "I know this.",
                    },
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
        "zaya",
    )

    assert out["response"] == "Paris"
    assert out["thinking"] == "I know this."
    assert out["done"] is True
    assert out["done_reason"] == "stop"
    assert out["prompt_eval_count"] == 3
    assert out["eval_count"] == 2


def test_chat_stream_chunk_converts_to_ollama_generate_ndjson():
    from vmlx_engine.api.ollama_adapter import (
        openai_chat_chunk_to_ollama_generate_ndjson,
    )

    line = "data: " + json.dumps(
        {
            "choices": [
                {
                    "delta": {"content": "Pa"},
                    "finish_reason": None,
                }
            ]
        }
    )

    out = json.loads(openai_chat_chunk_to_ollama_generate_ndjson(line, "zaya"))

    assert out["response"] == "Pa"
    assert out["done"] is False
