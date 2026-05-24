from tests.cross_matrix.run_runtime_memory_stress_probe import (
    add_speed_metrics,
    build_request,
    classify_http_stage_status,
    extract_usage,
    probe_status_from_results,
    redact_large_payloads,
    Row,
)


def test_extract_usage_accepts_chat_and_responses_shapes():
    chat_usage = extract_usage(
        {
            "usage": {
                "prompt_tokens": 10086,
                "completion_tokens": 16,
                "prompt_tokens_details": {
                    "cached_tokens": 10085,
                    "cache_detail": "paged+dsv4",
                },
            }
        }
    )
    responses_usage = extract_usage(
        {
            "usage": {
                "input_tokens": 5994,
                "output_tokens": 16,
                "input_tokens_details": {
                    "cached_tokens": 512,
                    "cache_detail": "paged+ssm",
                },
            }
        }
    )

    assert chat_usage == {
        "prompt_tokens": 10086,
        "completion_tokens": 16,
        "cached_tokens": 10085,
        "cache_detail": "paged+dsv4",
    }
    assert responses_usage == {
        "prompt_tokens": 5994,
        "completion_tokens": 16,
        "cached_tokens": 512,
        "cache_detail": "paged+ssm",
    }


def test_add_speed_metrics_reports_uncached_prompt_rate_for_cache_hits():
    stage = {
        "elapsed_s": 1.002,
        "response": {
            "usage": {
                "prompt_tokens": 10086,
                "completion_tokens": 16,
                "prompt_tokens_details": {
                    "cached_tokens": 10085,
                    "cache_detail": "paged+dsv4",
                },
            }
        },
    }

    add_speed_metrics(stage)

    assert stage["usage_summary"]["cached_tokens"] == 10085
    assert stage["speed"]["prompt_tok_s_wall"] == 10065.868
    assert stage["speed"]["uncached_prompt_tok_s_wall"] == 0.998
    assert stage["speed"]["decode_tok_s_wall"] == 15.968


def test_redact_large_payloads_hides_data_urls_and_long_text():
    value = {
        "image_url": "data:image/png;base64," + ("a" * 5000),
        "text": "b" * 5001,
    }

    redacted = redact_large_payloads(value)

    assert redacted["image_url"].startswith("data:image/png;base64,<redacted chars=")
    assert "<redacted chars=5001>" in redacted["text"]


def test_probe_status_from_results_rejects_http_error_stage():
    status, reason = probe_status_from_results(
        [
            {"status": "ok", "http_code": 200},
            {"status": "http_error", "http_code": 400},
        ]
    )

    assert status == "fail"
    assert "http_error" in reason


def test_probe_status_from_results_accepts_expected_http_error_stage():
    status, reason = probe_status_from_results(
        [
            {"status": "expected_http_error", "http_code": 413},
        ]
    )

    assert status == "pass"
    assert reason is None


def test_classify_http_stage_requires_expected_error_code_when_supplied():
    response = {
        "error": {
            "message": "too large",
            "type": "invalid_request_error",
            "code": "vlm_image_prefill_too_large",
        }
    }

    assert classify_http_stage_status(413, response, 413) == "expected_http_error"
    assert (
        classify_http_stage_status(
            413,
            response,
            413,
            "vlm_image_prefill_too_large",
        )
        == "expected_http_error"
    )
    assert classify_http_stage_status(413, response, 413, "other_error") == "http_error"
    assert (
        classify_http_stage_status(500, response, 413, "vlm_image_prefill_too_large")
        == "http_error"
    )


def test_build_request_can_disable_thinking_for_chat_and_responses():
    row = Row("qwen", "/models/Qwen3.6-27B-MXFP4-MTP")

    chat = build_request(
        row,
        "Describe the image.",
        64,
        "chat",
        None,
        retained_image_turns=1,
        enable_thinking=False,
    )
    responses = build_request(
        row,
        "Describe the image.",
        64,
        "responses",
        None,
        retained_image_turns=1,
        enable_thinking=False,
    )

    assert chat["enable_thinking"] is False
    assert chat["chat_template_kwargs"] == {"enable_thinking": False}
    assert responses["enable_thinking"] is False
    assert responses["chat_template_kwargs"] == {"enable_thinking": False}


def test_build_request_can_set_explicit_max_thinking_tokens_without_changing_output_cap():
    row = Row("gemma", "/models/Gemma-4-26B")

    chat = build_request(
        row,
        "Think briefly, then answer.",
        128,
        "chat",
        None,
        retained_image_turns=1,
        enable_thinking=True,
        max_thinking_tokens=16,
    )
    responses = build_request(
        row,
        "Think briefly, then answer.",
        128,
        "responses",
        None,
        retained_image_turns=1,
        enable_thinking=True,
        max_thinking_tokens=16,
    )

    assert chat["max_tokens"] == 128
    assert chat["max_thinking_tokens"] == 16
    assert chat["chat_template_kwargs"] == {
        "enable_thinking": True,
        "thinking_budget": 16,
    }
    assert responses["max_output_tokens"] == 128
    assert responses["max_thinking_tokens"] == 16
    assert responses["chat_template_kwargs"] == {
        "enable_thinking": True,
        "thinking_budget": 16,
    }
