from types import SimpleNamespace

from tests.cross_matrix.run_runtime_memory_stress_probe import (
    ROWS,
    add_response_contract_metrics,
    add_speed_metrics,
    add_stream_speed_metrics,
    build_request,
    classify_http_stage_status,
    extract_usage,
    add_memory_metrics,
    add_cache_capacity_projection,
    extract_block_disk_max_gb,
    make_prompt,
    probe_status_from_results,
    redact_large_payloads,
    run_probe,
    Row,
    http_json_stream_timed,
    summarize_response_rails,
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


def test_make_prompt_speed_floor_asks_for_sustained_decode_and_identifiers():
    prompt = make_prompt(64, mode="speed_floor", max_tokens=192)

    assert "exactly 192 numbered audit lines" in prompt
    assert "Do not stop early" in prompt
    assert "DSV4CacheBudget" in prompt
    assert "qwen_mtp_media_history" in prompt
    assert "GemmaCacheProbe" in prompt


def test_runtime_memory_probe_includes_minimax_small_jangtq_row():
    row = ROWS["minimax_m27_small_jangtq"]

    assert row.path == "/Users/example/models/JANGQ/MiniMax-M2.7-Small-JANGTQ"
    assert row.tool_parser == "minimax"
    assert row.reasoning_parser == "minimax_m2"
    assert "--use-paged-cache" in row.cache_args
    assert "--enable-block-disk-cache" in row.cache_args


def test_http_json_timed_splits_open_body_and_parse_time(monkeypatch):
    import tests.cross_matrix.run_runtime_memory_stress_probe as probe

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"ok": true}'

    ticks = iter([10.0, 10.25, 10.75, 10.8])

    monkeypatch.setattr(probe.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(probe.urllib.request, "urlopen", lambda req, timeout=0: FakeResponse())

    code, body, timing = probe.http_json_timed("POST", "http://127.0.0.1/test", {"x": 1})

    assert code == 200
    assert body == {"ok": True}
    assert timing == {
        "open_seconds": 0.25,
        "read_seconds": 0.5,
        "json_seconds": 0.05,
        "total_seconds": 0.8,
    }


def test_http_json_stream_timed_records_ttft_and_usage(monkeypatch):
    import tests.cross_matrix.run_runtime_memory_stress_probe as probe

    class FakeStreamResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter(
                [
                    b'data: {"choices":[{"delta":{"content":"A"}}],"usage":{"prompt_tokens":10,"completion_tokens":1}}\n',
                    b"\n",
                    b'data: {"choices":[{"delta":{"content":"B"}}],"usage":{"prompt_tokens":10,"completion_tokens":2}}\n',
                    b"data: [DONE]\n",
                ]
            )

    ticks = iter([10.0, 10.1, 10.4, 10.5, 10.6, 10.7, 10.7])

    monkeypatch.setattr(probe.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(probe.urllib.request, "urlopen", lambda req, timeout=0: FakeStreamResponse())

    code, body, timing = http_json_stream_timed(
        "POST",
        "http://127.0.0.1/v1/chat/completions",
        {"stream": True},
    )

    assert code == 200
    assert body["choices"][0]["message"]["content"] == "AB"
    assert body["usage"] == {"prompt_tokens": 10, "completion_tokens": 2}
    assert timing == {
        "open_seconds": 0.1,
        "ttft_seconds": 0.4,
        "stream_read_seconds": 0.6,
        "post_first_token_seconds": 0.3,
        "total_seconds": 0.7,
        "chunk_count": 2,
    }


def test_add_stream_speed_metrics_uses_post_first_token_time():
    stage = {
        "http_timing": {
            "ttft_seconds": 0.6,
            "post_first_token_seconds": 2.5,
        },
        "response": {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 251,
                "prompt_tokens_details": {"cached_tokens": 0},
            }
        },
    }

    add_stream_speed_metrics(stage)

    assert stage["stream_speed"] == {
        "ttft_seconds": 0.6,
        "post_first_token_seconds": 2.5,
        "decode_tok_s_stream": 100.0,
        "completion_tokens": 251,
        "cached_tokens": 0,
    }


def test_add_memory_metrics_summarizes_process_and_metal_deltas():
    stage = {
        "before": {
            "process": {"rss_mb": 8192.0},
            "health": {
                "body": {
                    "memory": {
                        "active_mb": 16384.0,
                        "peak_mb": 18432.0,
                        "cache_mb": 256.0,
                    }
                }
            },
        },
        "after": {
            "process": {"rss_mb": 9216.0},
            "health": {
                "body": {
                    "memory": {
                        "active_mb": 17408.0,
                        "peak_mb": 20480.0,
                        "cache_mb": 384.0,
                    }
                }
            },
        },
    }

    add_memory_metrics(stage)

    assert stage["memory"] == {
        "process_rss_before_gb": 8.0,
        "process_rss_after_gb": 9.0,
        "process_rss_delta_gb": 1.0,
        "metal_active_before_gb": 16.0,
        "metal_active_after_gb": 17.0,
        "metal_active_delta_gb": 1.0,
        "metal_peak_after_gb": 20.0,
        "metal_cache_after_gb": 0.375,
    }


def test_add_cache_capacity_projection_uses_observed_l2_bytes_per_token():
    stage = {
        "after": {
            "cache_stats": {
                "body": {
                    "block_disk_cache": {
                        "disk_size_bytes": 144_837_947,
                        "total_tokens_on_disk": 5037,
                        "blocks_on_disk": 20,
                    }
                }
            }
        }
    }

    add_cache_capacity_projection(stage, block_disk_max_gb=10.0)

    assert stage["cache_capacity_projection"] == {
        "basis": "observed_block_disk_l2",
        "tokens_on_disk": 5037,
        "blocks_on_disk": 20,
        "disk_size_gb": 0.135,
        "bytes_per_token": 28754.8,
        "projected_tokens_at_l2_max_gb": 373_413,
        "projected_gb_at_1m_tokens": 26.78,
        "block_disk_max_gb": 10.0,
    }


def test_cache_capacity_projection_marks_bypassed_rows_invalid_for_capacity_proof():
    stage = {
        "after": {
            "cache_stats": {
                "body": {
                    "block_disk_cache": {
                        "disk_size_bytes": 144_837_947,
                        "total_tokens_on_disk": 5037,
                        "blocks_on_disk": 20,
                    }
                }
            }
        }
    }

    add_cache_capacity_projection(stage, block_disk_max_gb=10.0, cache_bypassed=True)

    assert stage["cache_capacity_projection"] == {
        "basis": "invalid_for_capacity_proof",
        "reason": "cache_salt_or_skip_prefix_cache_bypasses_prefix_paged_l2",
        "tokens_on_disk": 5037,
        "blocks_on_disk": 20,
        "disk_size_gb": 0.135,
    }


def test_extract_block_disk_max_gb_reads_last_cli_value():
    assert (
        extract_block_disk_max_gb(
            [
                "--block-disk-cache-max-gb",
                "10",
                "--other",
                "--block-disk-cache-max-gb",
                "6.5",
            ]
        )
        == 6.5
    )


def test_redact_large_payloads_hides_data_urls_and_long_text():
    value = {
        "image_url": "data:image/png;base64," + ("a" * 5000),
        "text": "b" * 5001,
    }

    redacted = redact_large_payloads(value)

    assert redacted["image_url"].startswith("data:image/png;base64,<redacted chars=")
    assert "<redacted chars=5001>" in redacted["text"]


def test_summarize_response_rails_handles_responses_reasoning_only():
    summary = summarize_response_rails(
        {
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "content": [
                        {"type": "reasoning", "text": "thinking through it"},
                    ],
                }
            ],
            "usage": {"output_tokens": 16},
        }
    )

    assert summary["status"] == "completed"
    assert summary["has_visible_content"] is False
    assert summary["has_reasoning_content"] is True
    assert summary["visible_chars"] == 0
    assert summary["reasoning_chars"] == len("thinking through it")


def test_response_contract_can_fail_ok_http_when_visible_content_is_required():
    stage = {
        "status": "ok",
        "response": {
            "output": [
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning", "text": "private thought"}],
                }
            ]
        },
    }

    add_response_contract_metrics(stage, expect_visible_content=True)

    assert stage["status"] == "response_contract_failed"
    assert stage["response_contract_failures"] == ["missing_visible_content"]
    assert stage["response_rails"]["has_reasoning_content"] is True


def test_response_contract_accepts_visible_responses_text():
    stage = {
        "status": "ok",
        "response": {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "final answer"}],
                }
            ]
        },
    }

    add_response_contract_metrics(stage, expect_visible_content=True)

    assert stage["status"] == "ok"
    assert "response_contract_failures" not in stage
    assert stage["response_rails"]["visible_preview"] == "final answer"


def test_probe_status_from_results_rejects_http_error_stage():
    status, reason = probe_status_from_results(
        [
            {"status": "ok", "http_code": 200},
            {"status": "response_contract_failed", "http_code": 200},
        ]
    )

    assert status == "fail"
    assert "response_contract_failed" in reason


def test_probe_status_from_results_rejects_response_contract_failure_stage():
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


def test_build_request_can_bypass_prefix_cache_for_cold_speed_rows():
    row = Row("gemma", "/models/Gemma-4-26B")

    chat = build_request(
        row,
        "Cold run.",
        64,
        "chat",
        None,
        retained_image_turns=1,
        cache_salt="cold-gemma",
        skip_prefix_cache=True,
    )
    responses = build_request(
        row,
        "Cold run.",
        64,
        "responses",
        None,
        retained_image_turns=1,
        cache_salt="cold-gemma",
        skip_prefix_cache=True,
    )

    assert chat["cache_salt"] == "cold-gemma"
    assert chat["skip_prefix_cache"] is True
    assert responses["cache_salt"] == "cold-gemma"
    assert responses["skip_prefix_cache"] is True


def test_build_request_can_enable_streaming_usage_for_app_style_timing():
    row = Row("gemma", "/models/Gemma-4-26B")

    chat = build_request(
        row,
        "Stream this.",
        64,
        "chat",
        None,
        retained_image_turns=1,
        stream=True,
    )
    responses = build_request(
        row,
        "Stream this.",
        64,
        "responses",
        None,
        retained_image_turns=1,
        stream=True,
    )

    assert chat["stream"] is True
    assert chat["stream_options"] == {"include_usage": True}
    assert responses["stream"] is True
    assert responses["stream_options"] == {"include_usage": True}


def test_run_probe_artifact_marks_cache_salt_as_not_capacity_proof(monkeypatch):
    import tests.cross_matrix.run_runtime_memory_stress_probe as probe

    monkeypatch.setitem(
        probe.ROWS,
        "missing_for_policy_test",
        Row(
            "missing_for_policy_test",
            "/definitely/missing/model",
            cache_args=["--enable-block-disk-cache", "--block-disk-cache-max-gb", "10"],
        ),
    )

    result = run_probe(
        SimpleNamespace(
            row="missing_for_policy_test",
            python=".venv/bin/python",
            route="chat",
            prompt_tokens=[128],
            max_tokens=16,
            max_thinking_tokens=None,
            image_path=None,
            retained_image_turns=1,
            abort_rss_gb=0.0,
            abort_metal_active_gb=0.0,
            serve_extra_arg=[],
            cache_salt="capacity-calibration",
            skip_prefix_cache=False,
        )
    )

    assert result["status"] == "skipped"
    assert result["cache_proof_policy"] == {
        "cache_salt": "capacity-calibration",
        "skip_prefix_cache": False,
        "prefix_paged_l2_bypassed": True,
        "capacity_projection_valid": False,
    }
