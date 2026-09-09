# SPDX-License-Identifier: Apache-2.0
"""Ollama adapter parity tests."""

from __future__ import annotations

import json

import pytest


def test_ollama_generate_default_uses_chat_template_request_shape():
    from vmlx_engine.api.ollama_adapter import ollama_generate_to_openai_chat

    req = ollama_generate_to_openai_chat(
        {
            "model": "zaya",
            "system": "Be terse.",
            "prompt": "What is the capital of France?",
            "stream": False,
            "format": "json",
            "options": {
                "num_predict": 16,
                "temperature": 0,
                "top_p": 1,
                "top_k": 40,
                "min_p": 0.02,
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
    assert req["top_k"] == 40
    assert req["min_p"] == 0.02
    assert req["repetition_penalty"] == 1.1
    assert "enable_thinking" not in req
    assert req["response_format"] == {"type": "json_object"}


def test_ollama_chat_preserves_top_k_absent_zero_positive_contract():
    from vmlx_engine.api.ollama_adapter import ollama_chat_to_openai

    base = {
        "model": "hy3",
        "messages": [{"role": "user", "content": "hi"}],
    }
    assert "top_k" not in ollama_chat_to_openai(base)
    assert ollama_chat_to_openai(
        {**base, "options": {"top_k": 0}}
    )["top_k"] == 0
    assert ollama_chat_to_openai(
        {**base, "options": {"top_k": 40}}
    )["top_k"] == 40


@pytest.mark.parametrize("invalid_top_k", [-1, 1.5])
def test_ollama_chat_preserves_invalid_top_k_for_request_validation(invalid_top_k):
    from pydantic import ValidationError

    from vmlx_engine.api.models import ChatCompletionRequest
    from vmlx_engine.api.ollama_adapter import ollama_chat_to_openai

    req = ollama_chat_to_openai(
        {
            "model": "hy3",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"top_k": invalid_top_k},
        }
    )
    assert req["top_k"] == invalid_top_k
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(req)


def test_ollama_chat_translates_video_and_audio_extensions():
    from vmlx_engine.api.ollama_adapter import ollama_chat_to_openai

    req = ollama_chat_to_openai(
        {
            "model": "omni",
            "messages": [{
                "role": "user",
                "content": "Inspect both.",
                "videos": ["AAAAIGZ0eXA="],
                "audio": "SUQz",
            }],
        }
    )

    assert req["messages"] == [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Inspect both."},
            {"type": "video_url", "video_url": {
                "url": "data:video/mp4;base64,AAAAIGZ0eXA=",
            }},
            {"type": "audio_url", "audio_url": {
                "url": "data:audio/wav;base64,SUQz",
            }},
        ],
    }]


def test_ollama_generate_images_agree_with_chat_translation():
    """/api/generate takes `images` as a TOP-LEVEL base64 array alongside
    `prompt`. It must produce the exact same internal multimodal message
    /api/chat produces for the equivalent per-message request — previously
    the array was silently dropped and vision requests became text-only."""
    from vmlx_engine.api.models import ChatCompletionRequest
    from vmlx_engine.api.ollama_adapter import (
        ollama_chat_to_openai,
        ollama_generate_to_openai_chat,
    )

    raw_b64 = "aW1hZ2U="  # raw base64, NOT a data URL — Ollama's convention
    data_url = "data:image/jpeg;base64,/9j/4AAQ"
    gen_req = ollama_generate_to_openai_chat(
        {
            "model": "qwen-vl",
            "system": "Be terse.",
            "prompt": "What is in this picture?",
            "images": [raw_b64, data_url],
        }
    )
    chat_req = ollama_chat_to_openai(
        {
            "model": "qwen-vl",
            "messages": [
                {"role": "system", "content": "Be terse."},
                {
                    "role": "user",
                    "content": "What is in this picture?",
                    "images": [raw_b64, data_url],
                },
            ],
        }
    )

    # The regression that matters: both Ollama entry points must agree.
    assert gen_req["messages"] == chat_req["messages"]
    # And the media must actually be present in the shared shape: raw base64
    # gets the data-URL prefix, existing data URLs pass through untouched.
    assert gen_req["messages"][-1]["content"] == [
        {"type": "text", "text": "What is in this picture?"},
        {"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{raw_b64}",
        }},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    # The converted request must still validate as a chat request end-to-end.
    ChatCompletionRequest.model_validate(gen_req)

    # Bare-string leniency mirrors /api/chat's per-message coercion.
    bare = ollama_generate_to_openai_chat(
        {"model": "qwen-vl", "prompt": "look", "images": raw_b64}
    )
    assert bare["messages"] == ollama_chat_to_openai(
        {
            "model": "qwen-vl",
            "messages": [{"role": "user", "content": "look", "images": raw_b64}],
        }
    )["messages"]


def test_ollama_generate_without_images_keeps_plain_string_prompt():
    """No `images` key (or an explicitly empty array — /api/chat's "no media"
    contract) must leave the prompt as a plain string: no content-part
    wrapping that would change a text-only prompt's tokenization."""
    from vmlx_engine.api.ollama_adapter import ollama_generate_to_openai_chat

    req = ollama_generate_to_openai_chat({"model": "hy3", "prompt": "hi"})
    assert req["messages"] == [{"role": "user", "content": "hi"}]

    req_empty = ollama_generate_to_openai_chat(
        {"model": "hy3", "prompt": "hi", "images": []}
    )
    assert req_empty["messages"] == [{"role": "user", "content": "hi"}]


def test_ollama_generate_malformed_images_agree_with_chat_handling():
    """/api/chat does not validate base64: strings pass through (raw base64
    is data-URL prefixed, the downstream content-part decoder surfaces the
    malformed payload) and non-string entries are skipped. /api/generate
    must do exactly the same — no second convention, no adapter-level
    validation, no exception."""
    from vmlx_engine.api.ollama_adapter import (
        ollama_chat_to_openai,
        ollama_generate_to_openai_chat,
    )

    malformed = ["%%%not-base64%%%", "", 42, None, "data:image/jpeg;base64,///"]
    gen_req = ollama_generate_to_openai_chat(
        {"model": "vl", "prompt": "look", "images": malformed}
    )
    chat_req = ollama_chat_to_openai(
        {
            "model": "vl",
            "messages": [{"role": "user", "content": "look", "images": malformed}],
        }
    )

    assert gen_req["messages"] == chat_req["messages"]
    parts = gen_req["messages"][-1]["content"]
    assert parts[0] == {"type": "text", "text": "look"}
    assert [p["image_url"]["url"] for p in parts[1:]] == [
        "data:image/png;base64,%%%not-base64%%%",
        "data:image/png;base64,",
        "data:image/jpeg;base64,///",
    ]


def test_ollama_chat_normalizes_private_thinking_before_text_and_media_history():
    from vmlx_engine.api.ollama_adapter import ollama_chat_to_openai

    req = ollama_chat_to_openai(
        {
            "model": "omni",
            "messages": [
                {"role": "user", "content": "first"},
                {
                    "role": "assistant",
                    "thinking": "PRIVATE-PLAN-TEXT",
                    "content": "visible text",
                },
                {
                    "role": "assistant",
                    "thinking": "PRIVATE-PLAN-MEDIA",
                    "content": "visible media",
                    "images": ["aW1hZ2U="],
                },
                {
                    "role": "assistant",
                    "thinking": "",
                    "content": "empty private rail",
                },
                {
                    "role": "assistant",
                    "thinking": "ALIAS-MUST-NOT-WIN",
                    "reasoning_content": "CANONICAL-PRIVATE-PLAN",
                    "content": "canonical wins",
                },
                {"role": "user", "content": "second"},
            ],
        }
    )

    assert req["messages"] == [
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "reasoning_content": "PRIVATE-PLAN-TEXT",
            "content": "visible text",
        },
        {
            "role": "assistant",
            "reasoning_content": "PRIVATE-PLAN-MEDIA",
            "content": [
                {"type": "text", "text": "visible media"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,aW1hZ2U=",
                    },
                },
            ],
        },
        {
            "role": "assistant",
            "content": "empty private rail",
        },
        {
            "role": "assistant",
            "reasoning_content": "CANONICAL-PRIVATE-PLAN",
            "content": "canonical wins",
        },
        {"role": "user", "content": "second"},
    ]
    assert '"thinking"' not in json.dumps(req["messages"])


def test_ollama_generate_preserves_top_k_absent_zero_positive_contract():
    from vmlx_engine.api.ollama_adapter import (
        ollama_generate_to_openai,
        ollama_generate_to_openai_chat,
    )

    for convert in (ollama_generate_to_openai, ollama_generate_to_openai_chat):
        base = {"model": "hy3", "prompt": "hi"}
        assert "top_k" not in convert(base)
        assert convert({**base, "options": {"top_k": 0}})["top_k"] == 0
        assert convert({**base, "options": {"top_k": 40}})["top_k"] == 40


@pytest.mark.parametrize("invalid_top_k", [-1, 1.5])
def test_ollama_generate_preserves_invalid_top_k_for_request_validation(
    invalid_top_k,
):
    from pydantic import ValidationError

    from vmlx_engine.api.models import ChatCompletionRequest, CompletionRequest
    from vmlx_engine.api.ollama_adapter import (
        ollama_generate_to_openai,
        ollama_generate_to_openai_chat,
    )

    for convert, request_model in (
        (ollama_generate_to_openai, CompletionRequest),
        (ollama_generate_to_openai_chat, ChatCompletionRequest),
    ):
        req = convert(
            {
                "model": "hy3",
                "prompt": "hi",
                "options": {"top_k": invalid_top_k},
            }
        )
        assert req["top_k"] == invalid_top_k
        with pytest.raises(ValidationError):
            request_model.model_validate(req)


def test_ollama_chat_omits_non_positive_num_predict_sentinels():
    from vmlx_engine.api.ollama_adapter import ollama_chat_to_openai

    for sentinel in (0, -1, -2):
        req = ollama_chat_to_openai(
            {
                "model": "hy3",
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"num_predict": sentinel},
            }
        )
        assert "max_tokens" not in req


def test_ollama_output_limits_reach_shared_validation_without_truncation():
    from pydantic import ValidationError
    import pytest
    from vmlx_engine.api.models import ChatCompletionRequest, CompletionRequest
    from vmlx_engine.api.ollama_adapter import (
        ollama_chat_to_openai, ollama_generate_to_openai, ollama_generate_to_openai_chat,
    )
    for convert, request_model in (
        (ollama_chat_to_openai, ChatCompletionRequest),
        (ollama_generate_to_openai, CompletionRequest),
        (ollama_generate_to_openai_chat, ChatCompletionRequest),
    ):
        for value in ("not-a-number", "Infinity", "", 12.9, -0.5, [], {}):
            result = convert({
                "model": "diagnostic", "prompt": "hi",
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"num_predict": value},
            })
            assert result["max_tokens"] == value
            with pytest.raises(ValidationError):
                request_model.model_validate(result)


def test_ollama_generate_omits_non_positive_num_predict_sentinels():
    from vmlx_engine.api.ollama_adapter import (
        ollama_generate_to_openai,
        ollama_generate_to_openai_chat,
    )

    for convert in (ollama_generate_to_openai, ollama_generate_to_openai_chat):
        for sentinel in (0, -1, -2):
            req = convert(
                {
                    "model": "hy3",
                    "prompt": "hi",
                    "options": {"num_predict": sentinel},
                }
            )
            assert "max_tokens" not in req


def test_ollama_chat_omits_enable_thinking_when_think_is_omitted():
    from vmlx_engine.api.ollama_adapter import ollama_chat_to_openai

    req = ollama_chat_to_openai(
        {
            "model": "zaya",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )

    assert "enable_thinking" not in req


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


def test_ollama_native_think_false_disables_reasoning():
    from vmlx_engine.api.ollama_adapter import ollama_chat_to_openai

    req = ollama_chat_to_openai(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}],
            "think": False,
        }
    )

    assert req["enable_thinking"] is False


def test_ollama_chat_drops_reasoning_effort_when_native_think_false():
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


def test_ollama_generate_chat_native_think_false_disables_reasoning():
    from vmlx_engine.api.ollama_adapter import ollama_generate_to_openai_chat

    req = ollama_generate_to_openai_chat(
        {"model": "qwen", "prompt": "hi", "think": False}
    )

    assert req["enable_thinking"] is False


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
            "options": {
                "num_predict": 4,
                "temperature": 0,
                "top_p": 0.9,
                "top_k": 20,
                "min_p": 0.01,
                "repeat_penalty": 1.05,
            },
        }
    )

    assert req["prompt"] == "raw text"
    assert "messages" not in req
    assert req["max_tokens"] == 4
    assert req["temperature"] == 0
    assert req["top_p"] == 0.9
    assert req["top_k"] == 20
    assert req["min_p"] == 0.01
    assert req["repetition_penalty"] == 1.05


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


def test_ollama_chat_preserves_two_tool_results_for_continuation():
    from vmlx_engine.api.ollama_adapter import ollama_chat_to_openai

    messages = [
        {"role": "user", "content": "use both tools"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "file_info",
                        "arguments": {"path": "panel/package.json"},
                    }
                },
                {
                    "function": {
                        "name": "run_command",
                        "arguments": {"command": "pwd"},
                    }
                },
            ],
        },
        {"role": "tool", "tool_name": "file_info", "content": "Size: 5.2 KB"},
        {
            "role": "tool",
            "tool_name": "run_command",
            "content": "/Users/example/mlx/vllm-mlx",
        },
    ]

    converted = ollama_chat_to_openai(
        {"model": "minimax", "messages": messages, "tools": [{"type": "function"}]}
    )

    assert converted["messages"] == messages
    assert converted["tools"] == [{"type": "function"}]


def test_ollama_chat_preserves_prior_thinking_privately_for_tool_continuation():
    from vmlx_engine.api.ollama_adapter import ollama_chat_to_openai

    converted = ollama_chat_to_openai(
        {
            "model": "laguna",
            "messages": [
                {"role": "user", "content": "inspect the file"},
                {
                    "role": "assistant",
                    "thinking": "I should call file_info.",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "file_info",
                                "arguments": {"path": "panel/package.json"},
                            }
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_name": "file_info",
                    "content": "Size: 5.2 KB",
                },
            ],
        }
    )

    assistant = converted["messages"][1]
    assert "thinking" not in assistant
    assert assistant["reasoning_content"] == "I should call file_info."
    assert assistant["content"] == ""


def test_ollama_stream_terminal_preserves_two_object_argument_tool_calls():
    from vmlx_engine.api.ollama_adapter import openai_chat_chunk_to_ollama_ndjson

    line = "data: " + json.dumps(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_file",
                                "function": {
                                    "name": "file_info",
                                    "arguments": '{"path":"panel/package.json"}',
                                },
                            },
                            {
                                "index": 1,
                                "id": "call_pwd",
                                "function": {
                                    "name": "run_command",
                                    "arguments": '{"command":"pwd"}',
                                },
                            },
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 304, "completion_tokens": 67},
        }
    )

    out = json.loads(openai_chat_chunk_to_ollama_ndjson(line, "minimax"))

    assert out["done"] is True
    assert out["done_reason"] == "tool_calls"
    assert out["prompt_eval_count"] == 304
    assert out["eval_count"] == 67
    assert out["message"]["tool_calls"] == [
        {
            "function": {
                "name": "file_info",
                "arguments": {"path": "panel/package.json"},
            }
        },
        {"function": {"name": "run_command", "arguments": {"command": "pwd"}}},
    ]


def test_hy3_answer_pass_streams_final_text_as_ollama_content():
    """Hy3's bounded retry must leave the high-effort thinking rail.

    If ``reasoning_effort=high`` survives from the first pass, the parser sends
    the final answer as ``reasoning_content`` and Ollama misroutes it to
    ``message.thinking``.  The direct-rail retry produces a content delta; the
    adapter must preserve that channel while keeping genuine reasoning separate.
    """
    from vmlx_engine import server
    from vmlx_engine.api.ollama_adapter import openai_chat_chunk_to_ollama_ndjson

    answer_kwargs = {
        "enable_thinking": True,
        "reasoning_effort": "high",
        "chat_template_kwargs": {"reasoning_effort": "high"},
    }
    assert "hy_v3" in server._REASONING_ANSWER_PASS_FAMILIES
    server._force_answer_pass_direct_rail(
        answer_kwargs,
        family_name="hy_v3",
    )
    assert answer_kwargs == {
        "enable_thinking": False,
        "reasoning_effort": "no_think",
        "chat_template_kwargs": {"reasoning_effort": "no_think"},
    }

    reasoning_line = "data: " + json.dumps(
        {
            "choices": [
                {
                    "delta": {"reasoning_content": "internal rail"},
                    "finish_reason": None,
                }
            ]
        }
    )
    answer_line = "data: " + json.dumps(
        {
            "choices": [
                {
                    "delta": {"content": "FINAL-CHECK"},
                    "finish_reason": None,
                }
            ]
        }
    )
    reasoning = json.loads(openai_chat_chunk_to_ollama_ndjson(reasoning_line, "hy3"))
    answer = json.loads(openai_chat_chunk_to_ollama_ndjson(answer_line, "hy3"))

    assert reasoning["message"] == {
        "role": "assistant",
        "content": "",
        "thinking": "internal rail",
    }
    assert answer["message"] == {
        "role": "assistant",
        "content": "FINAL-CHECK",
    }


def test_ollama_terminal_merge_defers_length_until_after_answer_and_usage():
    from vmlx_engine.api.ollama_adapter import merge_ollama_stream_terminal

    provisional = {
        "model": "hy3",
        "message": {"role": "assistant", "content": "", "thinking": "reason"},
        "done": True,
        "done_reason": "length",
    }
    answer_stop = {
        "model": "hy3",
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
    }
    usage = {
        "model": "hy3",
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "eval_count": 428,
        "prompt_eval_count": 97,
    }

    merged = merge_ollama_stream_terminal(provisional, answer_stop)
    merged = merge_ollama_stream_terminal(merged, usage)

    assert merged["done"] is True
    assert merged["done_reason"] == "stop"
    assert merged["eval_count"] == 428
    assert merged["prompt_eval_count"] == 97
    assert merged["message"]["thinking"] == "reason"


def test_ollama_terminal_merge_preserves_truncated_answer_after_usage():
    from vmlx_engine.api.ollama_adapter import merge_ollama_stream_terminal

    answer_length = {
        "model": "laguna",
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "length",
    }
    usage = {
        "model": "laguna",
        "message": {"role": "assistant", "content": ""},
        "done": True,
        # Adapter-created usage terminals default to stop; they are accounting,
        # not a second model finish event.
        "done_reason": "stop",
        "eval_count": 560,
        "prompt_eval_count": 73,
    }

    merged = merge_ollama_stream_terminal(answer_length, usage)

    assert merged["done"] is True
    assert merged["done_reason"] == "length"
    assert merged["eval_count"] == 560
    assert merged["prompt_eval_count"] == 73


def test_ollama_generate_terminal_merge_retains_usage_on_single_done_row():
    from vmlx_engine.api.ollama_adapter import (
        merge_ollama_generate_stream_terminal,
    )

    finish = {
        "model": "minimax",
        "created_at": "2026-07-19T00:00:00.000Z",
        "response": "",
        "done": True,
        "done_reason": "stop",
    }
    usage = {
        "model": "minimax",
        "created_at": "2026-07-19T00:00:01.000Z",
        "response": "",
        "done": True,
        "done_reason": "stop",
        "eval_count": 215,
        "prompt_eval_count": 78,
    }

    merged = merge_ollama_generate_stream_terminal(finish, usage)

    assert merged["done"] is True
    assert merged["done_reason"] == "stop"
    assert merged["eval_count"] == 215
    assert merged["prompt_eval_count"] == 78


def test_ollama_generate_terminal_merge_preserves_length_after_usage():
    from vmlx_engine.api.ollama_adapter import (
        merge_ollama_generate_stream_terminal,
    )

    finish = {
        "model": "laguna",
        "created_at": "2026-07-22T00:00:00.000Z",
        "response": "",
        "done": True,
        "done_reason": "length",
    }
    usage = {
        "model": "laguna",
        "created_at": "2026-07-22T00:00:01.000Z",
        "response": "",
        "done": True,
        "done_reason": "stop",
        "eval_count": 560,
        "prompt_eval_count": 73,
    }

    merged = merge_ollama_generate_stream_terminal(finish, usage)

    assert merged["done"] is True
    assert merged["done_reason"] == "length"
    assert merged["eval_count"] == 560
    assert merged["prompt_eval_count"] == 73


def test_ollama_chat_stream_maps_upstream_error_to_native_error_row():
    from vmlx_engine.api.ollama_adapter import openai_chat_chunk_to_ollama_ndjson

    line = "data: " + json.dumps(
        {
            "error": {
                "type": "server_error",
                "message": "CHAT MIDSTREAM PROBE FAILURE",
                "code": "internal_error",
            }
        }
    )

    assert json.loads(openai_chat_chunk_to_ollama_ndjson(line, "probe")) == {
        "error": "CHAT MIDSTREAM PROBE FAILURE"
    }


def test_ollama_templated_generate_stream_preserves_native_error_row():
    from vmlx_engine.api.ollama_adapter import (
        openai_chat_chunk_to_ollama_generate_ndjson,
    )

    line = "data: " + json.dumps(
        {"error": {"type": "server_error", "message": "TEMPLATE FAILURE"}}
    )

    assert json.loads(
        openai_chat_chunk_to_ollama_generate_ndjson(line, "probe")
    ) == {"error": "TEMPLATE FAILURE"}


def test_ollama_raw_generate_stream_maps_upstream_error_to_native_error_row():
    from vmlx_engine.api.ollama_adapter import (
        openai_completion_chunk_to_ollama_ndjson,
    )

    line = "data: " + json.dumps(
        {"error": {"type": "server_error", "message": "RAW FAILURE"}}
    )

    assert json.loads(openai_completion_chunk_to_ollama_ndjson(line, "probe")) == {
        "error": "RAW FAILURE"
    }


def test_ollama_think_string_levels_enable_and_select_effort():
    """Modern Ollama passes an effort *level* through `think` ("low"/"high").
    A level must both enable thinking and select the reasoning effort, forwarded
    through the shared reasoning_effort passthrough. Previously a level string
    normalized to None and was silently dropped — thinking never even engaged."""
    from vmlx_engine.api.ollama_adapter import (
        _apply_ollama_thinking,
        _should_forward_reasoning_effort,
    )

    for level in ("minimal", "low", "medium", "high", "xhigh", "max"):
        body = {"think": level}
        req: dict = {}
        _apply_ollama_thinking(body, req)
        if _should_forward_reasoning_effort(body, req):
            req["reasoning_effort"] = body["reasoning_effort"]
        assert req.get("enable_thinking") is True, level
        assert req.get("reasoning_effort") == level, level

    # Booleans still behave exactly as before.
    for raw, expected in ((True, True), (False, False), ("none", False), ("off", False)):
        body = {"think": raw}
        req = {}
        _apply_ollama_thinking(body, req)
        assert req.get("enable_thinking") is expected, raw

    # An explicit body-level reasoning_effort is not clobbered by the level.
    body = {"think": "high", "reasoning_effort": "low"}
    req = {}
    _apply_ollama_thinking(body, req)
    assert body["reasoning_effort"] == "low"
