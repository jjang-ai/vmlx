from types import SimpleNamespace

from tests.cross_matrix import run_mimo_v2_cache_vs_nocache_next_token as runner


def test_cache_vs_nocache_runner_builds_chat_payload_for_mllm_rows():
    args = SimpleNamespace(
        served_model_name="n2-pro-jangtq2-cache-proof",
        prompt="Return exactly one word: ACK",
    )

    payload = runner.request_payload(args, skip_prefix_cache=True, endpoint="chat")

    assert payload["model"] == "n2-pro-jangtq2-cache-proof"
    assert payload["messages"] == [
        {"role": "user", "content": "Return exactly one word: ACK"}
    ]
    assert payload["max_tokens"] == 1
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 10
    assert payload["skip_prefix_cache"] is True
    assert "prompt" not in payload


def test_cache_vs_nocache_runner_extracts_chat_logprobs_and_usage():
    response = {
        "code": 200,
        "elapsed_s": 0.25,
        "error": None,
        "body": {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ACK"},
                    "finish_reason": "length",
                    "logprobs": {
                        "content": [
                            {
                                "token": "ACK",
                                "logprob": -0.01,
                                "top_logprobs": [
                                    {"token": "ACK", "logprob": -0.01},
                                    {"token": "OK", "logprob": -2.0},
                                ],
                            }
                        ]
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 1,
                "prompt_tokens_details": {
                    "cached_tokens": 8,
                    "cache_detail": "paged+ssm",
                },
            },
        },
    }

    row = runner.row_from_response("cache_hit", response, endpoint="chat")

    assert row["status_code"] == 200
    assert row["text"] == "ACK"
    assert row["token"] == "ACK"
    assert row["token_logprob"] == -0.01
    assert row["top10"] == [
        {"id": None, "text": "ACK", "logprob": -0.01},
        {"id": None, "text": "OK", "logprob": -2.0},
    ]
    assert row["usage"]["prompt_tokens_details"]["cached_tokens"] == 8


def test_cache_vs_nocache_runner_classifies_mllm_logprobs_as_unsupported():
    rows = [
        {
            "mode": mode,
            "status_code": 400,
            "body": {
                "detail": (
                    "Chat Completions logprobs currently require text-only LLM "
                    "requests; multimodal/VLM logprobs are not implemented."
                )
            },
            "top10": [],
        }
        for mode in ("no_cache_bypass", "cache_warm_store", "cache_hit")
    ]

    unsupported = runner.unsupported_logprobs_boundary(rows, endpoint="chat")

    assert unsupported == {
        "status": "skipped",
        "reason": "mllm_logprobs_unsupported",
        "detail": (
            "Chat Completions logprobs currently require text-only LLM "
            "requests; multimodal/VLM logprobs are not implemented."
        ),
    }
