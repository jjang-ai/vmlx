from tests.cross_matrix import run_mimo_v2_no_source_exactness_classifier as classifier
from tests.cross_matrix.run_mimo_v2_no_source_exactness_classifier import (
    build_classification,
)


def test_mimo_no_source_classifier_defaults_point_at_live_refresh_artifacts():
    assert str(classifier.DEFAULT_AUDIT) == (
        "build/current-mimo-v2-jang2l-current-audit-after-live-mimo-jangtq2-runtime-proof-20260609.json"
    )
    assert str(classifier.DEFAULT_SMOKE) == (
        "build/current-all-local-model-smoke-mimo-v25-jangtq2-live-refresh-20260608/"
        "summary.json"
    )
    assert str(classifier.DEFAULT_OUT) == (
        "build/current-mimo-v2-no-source-exactness-classifier-after-lossless-token-trace-20260609.json"
    )
    assert str(classifier.DEFAULT_JANG2L_JSON_SENTINEL_ISOLATION) == (
        "build/current-all-local-model-smoke-mimo-v25-jang2l-live-refresh-20260608/"
        "summary.json"
    )
    assert str(classifier.DEFAULT_JANGTQ2_CURRENT_ALL_LOCAL) == (
        "build/current-all-local-model-smoke-mimo-v25-jangtq2-live-refresh-20260608/"
        "JANGQ_MiMo-V2.5-JANGTQ_2/result.json"
    )


def test_mimo_no_source_classifier_keeps_exactness_open_without_fake_fix():
    audit = {
        "component_ok": {
            "api_cache_responses_contract": True,
            "tool_protocol": True,
            "exactness_cache_kv_quant_excluded": True,
            "decode_speed_target": True,
            "source_vs_quant_first_divergence": False,
            "long_prompt_coherence": False,
            "cb_system_prompt_working_set_pressure": False,
            "mimo_media_wired": False,
        }
    }
    smoke = {
        "results": [
            {
                "server_log_tail": [
                    "kwargs={'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 64}"
                ],
                "failures": [
                    {
                        "label": "tool_required",
                        "reason": "expected_tool_argument_missing",
                        "expected": {"value": "blue-cat"},
                        "actual": {"value": "blue-123"},
                    },
                    {
                        "label": "structured_json_exact",
                        "reason": "json_exact_object_mismatch",
                        "expected": {"status": "ok", "value": "blue-cat", "count": 3},
                        "actual": {"status": "ok", "value": "bluecat", "count": 3},
                    },
                ],
            }
        ]
    }

    artifact = build_classification(audit, smoke)

    assert artifact["status"] == "open"
    assert artifact["release_ready"] is False
    assert (
        artifact["classification"]
        == "model_generated_literal_mutation_after_valid_parser_structure"
    )
    assert artifact["source_vs_quant_load_performed"] is False
    assert artifact["excluded_surfaces"]["parser_argument_rewrite"] is True
    assert (
        artifact["excluded_surfaces"]["prefix_paged_l2_or_kv_quant_primary_cause"]
        is True
    )
    assert artifact["excluded_surfaces"]["hidden_stochastic_sampling_primary_cause"] is True
    assert artifact["unresolved_surfaces"]["artifact_quantization_or_decode_logits_quality"] is True
    assert artifact["unresolved_surfaces"]["source_vs_quant_first_divergence"] is True


def test_mimo_no_source_classifier_refuses_parser_claim_without_actual_args():
    audit = {"component_ok": {"tool_protocol": True}}
    smoke = {
        "results": [
            {
                "failures": [
                    {
                        "label": "tool_required",
                        "reason": "expected_tool_argument_missing",
                        "expected": {"value": "blue-cat"},
                        "actual": "blue-123",
                    }
                ]
            }
        ]
    }

    artifact = build_classification(audit, smoke)

    assert artifact["status"] == "open"
    assert artifact["classification"] == "insufficient_evidence"
    assert artifact["excluded_surfaces"]["parser_argument_rewrite"] is False


def test_mimo_no_source_classifier_records_greedy_top1_literal_drift():
    audit = {
        "component_ok": {
            "api_cache_responses_contract": True,
            "tool_protocol": True,
            "exactness_cache_kv_quant_excluded": True,
            "decode_speed_target": True,
            "source_vs_quant_first_divergence": False,
            "long_prompt_coherence": False,
            "cb_system_prompt_working_set_pressure": False,
            "mimo_media_wired": True,
        }
    }
    smoke = {
        "results": [
            {
                "server_log_tail": [
                    "kwargs={'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 24}"
                ],
                "failures": [
                    {
                        "label": "structured_json_exact",
                        "reason": "json_exact_object_mismatch",
                        "expected": {"status": "ok", "value": "blue-cat", "count": 3},
                        "actual": {"status": "ok", "value": "blue", "count": 3},
                    }
                ],
            }
        ]
    }
    logprob_diagnostic = {
        "cases": [
            {
                "label": "completion_exact_blue_cat",
                "route": "completions",
                "expected": "blue-cat",
                "text": "blue",
                "logprobs": {
                    "tokens": ["blue", "<|im_end|>"],
                    "top_logprobs": [
                        {"blue": -0.01, "Blue": -6.0},
                        {"<|im_end|>": -1.6, " cat": -1.9},
                    ],
                },
            },
            {
                "label": "chat_exact_blue_cat",
                "route": "chat",
                "expected": "blue-cat",
                "content": "blue grass",
                "logprobs": {
                    "content": [
                        {
                            "token": "blue",
                            "top_logprobs": [{"token": "blue", "logprob": -0.01}],
                        },
                        {
                            "token": " grass",
                            "top_logprobs": [{"token": " grass", "logprob": -1.4}],
                        },
                    ]
                },
            },
        ]
    }

    artifact = build_classification(
        audit,
        smoke,
        jangtq2_logprob_diagnostic=logprob_diagnostic,
    )

    assert artifact["status"] == "open"
    assert artifact["excluded_surfaces"]["hidden_stochastic_sampling_primary_cause"] is True
    assert artifact["excluded_surfaces"]["api_sampler_non_top1_selection"] is True
    assert artifact["jangtq2_logprob_summary"]["wrong_literal_outputs_are_top1"] is True
    assert artifact["model_upload_action_required"] is True
    assert any(
        "JANGTQ_2 served artifact emits wrong compact-hyphen/sentinel literals"
        in item
        for item in artifact["model_upload_action_reasons"]
    )
    assert artifact["jangtq2_logprob_summary"]["failed_literal_cases"][0]["tokens"] == [
        "blue",
        "<|im_end|>",
    ]
    assert any(
        "wrong literal outputs are greedy top-1" in item
        for item in artifact["required_next_evidence"]
    )


def test_mimo_no_source_classifier_preserves_overridden_literal_variant_pointer():
    audit = {"component_ok": {"tool_protocol": True}}
    smoke = {"results": []}

    artifact = build_classification(
        audit,
        smoke,
        literal_variants={"status": "open", "requests": []},
        artifact_paths={
            "jangtq2_literal_variant_artifact": (
                "build/current-mimo-v25-jangtq2-exactness-rerun-20260608/result.json"
            )
        },
    )

    assert artifact["jangtq2_literal_variant_artifact"] == (
        "build/current-mimo-v25-jangtq2-exactness-rerun-20260608/result.json"
    )


def test_mimo_no_source_classifier_consumes_jangtq_and_jang2l_isolation_artifacts():
    audit = {
        "component_ok": {
            "api_cache_responses_contract": True,
            "tool_protocol": True,
            "exactness_cache_kv_quant_excluded": True,
            "decode_speed_target": True,
            "source_vs_quant_first_divergence": False,
            "long_prompt_coherence": False,
            "cb_system_prompt_working_set_pressure": False,
            "mimo_media_wired": True,
        }
    }
    smoke = {"results": []}
    jangtq2 = {
        "cases": {
            "completion_copy_b7": {
                "body": {"choices": [{"text": "B7C9099"}]},
            },
            "chat_tool_b7": {
                "body": {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "arguments": '{"value":"B7CAT-09"}'
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            },
        }
    }
    jang2l = {
        "cases": {
            "completion_copy_b7": {
                "body": {"choices": [{"text": "B7-CAT-09"}]},
            },
            "chat_tool_b7": {
                "body": {"choices": [{"message": {"content": "prose, no tool"}}]},
            },
        }
    }

    no_switchglu_fastpath = {
        "cases": {
            "completion_copy_b7": {
                "body": {"choices": [{"text": "B7C9099"}]},
            }
        }
    }
    no_router_no_switchglu_fastpath = {
        "cases": {
            "completion_copy_b7": {
                "body": {"choices": [{"text": "B7C9099"}]},
            }
        }
    }
    tq_kernel_parity = {
        "status": "pass",
        "reports": [{"max_abs_diff": 2.4e-5}],
    }

    artifact = build_classification(
        audit,
        smoke,
        jangtq2=jangtq2,
        jang2l=jang2l,
        no_switchglu_fastpath=no_switchglu_fastpath,
        no_router_no_switchglu_fastpath=no_router_no_switchglu_fastpath,
        tq_kernel_parity=tq_kernel_parity,
    )

    assert artifact["status"] == "open"
    assert artifact["classification"] == (
        "jangtq2_literal_corruption_persists_without_cache_fastpath_router_"
        "and_tq_kernel_parity_passes"
    )
    assert artifact["no_source_exactness"]["jangtq2_raw_completion_literal_preserved"] is False
    assert artifact["no_source_exactness"]["jang2l_raw_completion_literal_preserved"] is True
    assert (
        artifact["no_source_runtime_diagnostics"][
            "jangtq2_switchglu_fastpath_primary_cause_excluded"
        ]
        is True
    )
    assert (
        artifact["no_source_runtime_diagnostics"][
            "jangtq2_compiled_router_primary_cause_excluded"
        ]
        is True
    )
    assert (
        artifact["no_source_runtime_diagnostics"][
            "jangtq2_tq_gather_kernel_primary_cause_excluded"
        ]
        is True
    )
    assert artifact["unresolved_surfaces"]["jangtq2_raw_decode_or_artifact_quality"] is True
    assert artifact["unresolved_surfaces"]["jang2l_chat_tool_quality"] is True
    assert artifact["unresolved_surfaces"]["jangtq2_switchglu_fastpath"] is False
    assert artifact["unresolved_surfaces"]["jangtq2_compiled_router"] is False
    assert artifact["unresolved_surfaces"]["jangtq2_tq_gather_kernel"] is False


def test_mimo_no_source_classifier_promotes_plain_literal_copy_failure():
    audit = {
        "component_ok": {
            "api_cache_responses_contract": True,
            "tool_protocol": True,
            "exactness_cache_kv_quant_excluded": True,
            "decode_speed_target": True,
            "source_vs_quant_first_divergence": False,
            "long_prompt_coherence": True,
            "cb_system_prompt_working_set_pressure": True,
            "mimo_media_wired": True,
        }
    }
    smoke = {
        "results": [
            {
                "server_log_tail": [
                    "kwargs={'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 64}"
                ],
                "failures": [
                    {
                        "label": "structured_json_exact",
                        "reason": "json_exact_object_mismatch",
                        "expected": {"status": "ok", "value": "blue-cat", "count": 3},
                        "actual": {"status": "ok", "value": "bluecat", "count": 3},
                    },
                ],
            }
        ]
    }
    literal_variants = {
        "status": "open",
        "requests": [
            {
                "label": "plain_exact_blue_cat",
                "route": "completions",
                "pass": False,
                "content": "blue cat",
                "expected": "blue-cat",
            },
            {
                "label": "plain_exact_chat_blue_cat",
                "route": "chat",
                "pass": False,
                "content": "blue cat",
                "expected": "blue-cat",
            },
            {
                "label": "json_blue_cat",
                "route": "chat",
                "pass": False,
                "content": '{"status":"ok","value":"bluecat","count":3}',
                "parsed": {"status": "ok", "value": "bluecat", "count": 3},
                "expected": {"status": "ok", "value": "blue-cat", "count": 3},
            },
            {
                "label": "tool_blue_cat",
                "route": "chat",
                "pass": False,
                "parsed": {"value": "blue-123"},
                "expected": {"value": "blue-cat"},
            },
        ],
    }
    jang2l_literal_variants = {
        "status": "open",
        "requests": [
            {
                "label": "plain_exact_blue_cat",
                "route": "completions",
                "pass": True,
                "content": "blue-cat",
                "expected": "blue-cat",
            },
            {
                "label": "plain_exact_sentinel",
                "route": "completions",
                "pass": True,
                "content": "B7-CAT-09",
                "expected": "B7-CAT-09",
            },
            {
                "label": "plain_exact_chat_sentinel",
                "route": "chat",
                "pass": True,
                "content": "B7-CAT-09",
                "expected": "B7-CAT-09",
            },
            {
                "label": "tool_blue_cat",
                "route": "chat",
                "pass": False,
                "code": 0,
                "expected": {"value": "blue-cat"},
            },
        ],
    }

    artifact = build_classification(
        audit,
        smoke,
        literal_variants=literal_variants,
        jang2l_literal_variants=jang2l_literal_variants,
    )

    assert artifact["classification"] == (
        "jangtq2_plain_literal_copy_regression_jang2l_plain_copy_passes"
    )
    assert artifact["literal_variant_summary"]["plain_literal_copy_pass"] is False
    assert (
        artifact["literal_variant_summary"]["plain_completion_literal_copy_pass"]
        is False
    )
    assert artifact["literal_variant_summary"]["plain_chat_literal_copy_pass"] is False
    assert artifact["jang2l_literal_variant_summary"]["plain_literal_copy_pass"] is True
    assert (
        artifact["jang2l_literal_variant_summary"]["plain_completion_literal_copy_pass"]
        is True
    )
    assert artifact["jang2l_literal_variant_summary"]["plain_chat_literal_copy_pass"] is True
    assert artifact["literal_variant_summary"]["structured_literal_pass"] is False
    assert artifact["literal_variant_summary"]["tool_literal_pass"] is False
    assert artifact["unresolved_surfaces"]["jangtq2_plain_literal_copy"] is True
    assert artifact["unresolved_surfaces"]["jang2l_tool_memory_or_protocol"] is True


def test_mimo_no_source_classifier_consumes_compact_hyphen_distribution_probe():
    audit = {
        "component_ok": {
            "api_cache_responses_contract": True,
            "tool_protocol": True,
            "exactness_cache_kv_quant_excluded": True,
            "decode_speed_target": True,
            "source_vs_quant_first_divergence": False,
            "long_prompt_coherence": True,
            "cb_system_prompt_working_set_pressure": True,
            "mimo_media_wired": True,
        }
    }
    smoke = {"results": []}
    hyphen_distribution = {
        "classification": "compact_hyphen_exactness_fails_in_served_artifact",
        "not_caused_by": [
            "prefix_cache",
            "l2_disk_cache",
            "tool_parser",
            "chat_template_only",
            "tokenizer_roundtrip",
        ],
        "requests": {
            "chat_exact_blue_cat": {"code": 200, "text": "blue cat"},
            "completion_repeat_blue_cat": {"code": 200, "text": "blue"},
            "chat_hyphen_spaces": {"code": 200, "text": "blue - cat"},
        },
    }

    artifact = build_classification(
        audit,
        smoke,
        jangtq2_hyphen_distribution=hyphen_distribution,
    )

    assert artifact["classification"] == (
        "jangtq2_compact_hyphen_decode_quality_open_not_cache_parser_template_tokenizer"
    )
    assert artifact["jangtq2_hyphen_distribution_summary"] == {
        "exists": True,
        "classification": "compact_hyphen_exactness_fails_in_served_artifact",
        "chat_compact_hyphen_fails": True,
        "completion_compact_hyphen_fails": True,
        "spaced_hyphen_preserved": True,
        "not_caused_by": [
            "prefix_cache",
            "l2_disk_cache",
            "tool_parser",
            "chat_template_only",
            "tokenizer_roundtrip",
        ],
    }
    assert (
        artifact["excluded_surfaces"]["chat_template_only_primary_cause"] is True
    )
    assert (
        artifact["excluded_surfaces"]["tokenizer_roundtrip_primary_cause"] is True
    )
    assert (
        artifact["unresolved_surfaces"]["jangtq2_compact_hyphen_decode_quality"]
        is True
    )


def test_mimo_no_source_classifier_consumes_conservative_no_cb_no_prefix_probe():
    audit = {
        "component_ok": {
            "api_cache_responses_contract": True,
            "tool_protocol": True,
            "exactness_cache_kv_quant_excluded": True,
            "decode_speed_target": True,
            "source_vs_quant_first_divergence": False,
            "long_prompt_coherence": True,
            "cb_system_prompt_working_set_pressure": True,
            "mimo_media_wired": True,
        }
    }
    smoke = {
        "results": [
            {
                "failures": [
                    {
                        "label": "tool_required",
                        "reason": "expected_tool_argument_missing",
                        "actual": {"value": "blue cat"},
                    }
                ],
                "server_log_tail": [
                    "Resolved sampling kwargs {'temperature': 0.0, 'top_p': 1.0}"
                ],
            }
        ]
    }
    conservative = {
        "status": "fail",
        "launch": "no-continuous-batching + disable-prefix-cache + is-mllm",
        "failures": [
            {"label": "json_blue_cat", "content": '{"value":"blue"}'},
            {"label": "tool_b7_cat_09", "tool_calls": []},
        ],
    }

    artifact = build_classification(
        audit,
        smoke,
        jangtq2_conservative_no_cb_no_prefix=conservative,
    )

    assert artifact["status"] == "open"
    assert artifact["jangtq2_conservative_no_cb_no_prefix_summary"] == {
        "exists": True,
        "status": "fail",
        "literal_exactness_failed": True,
        "failed_labels": ["json_blue_cat", "tool_b7_cat_09"],
        "continuous_batching_disabled": True,
        "prefix_cache_disabled": True,
        "launch": "no-continuous-batching + disable-prefix-cache + is-mllm",
    }
    assert artifact["excluded_surfaces"]["continuous_batching_primary_cause"] is True
    assert artifact["excluded_surfaces"]["prefix_cache_primary_cause"] is True
    assert (
        artifact["unresolved_surfaces"][
            "jangtq2_conservative_no_cb_no_prefix_literal_mutation"
        ]
        is True
    )


def test_mimo_no_source_classifier_tracks_jang2l_json_sentinel_separately():
    audit = {
        "component_ok": {
            "api_cache_responses_contract": True,
            "tool_protocol": True,
            "exactness_cache_kv_quant_excluded": True,
            "decode_speed_target": True,
            "source_vs_quant_first_divergence": False,
            "long_prompt_coherence": True,
            "cb_system_prompt_working_set_pressure": True,
            "mimo_media_wired": True,
        }
    }
    smoke = {"results": []}
    literal_variants = {
        "status": "open",
        "requests": [
            {
                "label": "plain_exact_blue_cat",
                "pass": False,
                "content": "blue cat",
                "expected": "blue-cat",
            },
            {
                "label": "json_blue_cat",
                "pass": False,
                "content": '{"status":"ok","value":"bluecat","count":3}',
                "parsed": {"status": "ok", "value": "bluecat", "count": 3},
                "expected": {"status": "ok", "value": "blue-cat", "count": 3},
            },
            {
                "label": "tool_blue_cat",
                "pass": False,
                "parsed": {"value": "blue-123"},
                "expected": {"value": "blue-cat"},
            },
        ],
    }
    jang2l_literal_variants = {
        "status": "open",
        "requests": [
            {
                "label": "plain_exact_blue_cat",
                "pass": True,
                "content": "blue-cat",
                "expected": "blue-cat",
            },
            {
                "label": "plain_exact_sentinel",
                "pass": True,
                "content": "B7-CAT-09",
                "expected": "B7-CAT-09",
            },
            {
                "label": "tool_blue_cat",
                "pass": True,
                "parsed": {"value": "blue-cat"},
                "expected": {"value": "blue-cat"},
            },
            {
                "label": "tool_sentinel_json_call",
                "pass": True,
                "parsed": {"value": "B7-CAT-09"},
                "expected": {"value": "B7-CAT-09"},
            },
            {
                "label": "json_sentinel",
                "pass": False,
                "content": "",
                "parsed": None,
                "expected": {"status": "ok", "value": "B7-CAT-09", "count": 3},
            },
        ],
    }
    jang2l_json_sentinel = {
        "status": "open",
        "requests": [
            {
                "label": "original",
                "pass": False,
                "content": "",
                "parsed": None,
                "expected": {"status": "ok", "value": "B7-CAT-09", "count": 3},
                "usage": {"completion_tokens": 1},
            },
            {
                "label": "lower_value_control",
                "pass": False,
                "content": '{"status":"ok","value":"b7-cat-09","readcount":3}',
                "parsed": {"status": "ok", "value": "b7-cat-09", "readcount": 3},
                "expected": {"status": "ok", "value": "b7-cat-09", "count": 3},
                "usage": {"completion_tokens": 20},
            },
        ],
    }
    stale_jang2l_isolation = {
        "cases": {
            "completion_copy_b7": {
                "body": {"choices": [{"text": "B7-CAT-09"}]},
            },
            "chat_tool_b7": {
                "body": {"choices": [{"message": {"content": "prose, no tool"}}]},
            },
        }
    }

    artifact = build_classification(
        audit,
        smoke,
        jang2l=stale_jang2l_isolation,
        literal_variants=literal_variants,
        jang2l_literal_variants=jang2l_literal_variants,
        jang2l_json_sentinel=jang2l_json_sentinel,
    )

    assert artifact["classification"] == (
        "jangtq2_plain_literal_copy_regression_jang2l_plain_copy_passes"
    )
    assert artifact["secondary_classification"] == "jang2l_json_sentinel_empty_output_open"
    assert artifact["unresolved_surfaces"]["jang2l_tool_memory_or_protocol"] is False
    assert artifact["unresolved_surfaces"]["jang2l_json_sentinel_exactness"] is True
    assert artifact["jang2l_json_sentinel_summary"]["empty_output_labels"] == ["original"]
    assert artifact["jang2l_json_sentinel_summary"]["schema_mutation_labels"] == [
        "lower_value_control"
    ]


def test_mimo_no_source_classifier_tracks_current_jang2l_json_semantic_mismatch():
    audit = {
        "component_ok": {
            "api_cache_responses_contract": True,
            "tool_protocol": True,
            "exactness_cache_kv_quant_excluded": True,
            "decode_speed_target": True,
            "source_vs_quant_first_divergence": False,
            "long_prompt_coherence": True,
            "cb_system_prompt_working_set_pressure": True,
            "mimo_media_wired": True,
        }
    }
    smoke = {"results": []}
    literal_variants = {
        "status": "open",
        "requests": [
            {
                "label": "plain_exact_blue_cat",
                "pass": False,
                "content": "blue cat",
                "expected": "blue-cat",
            },
        ],
    }
    jang2l_literal_variants = {
        "status": "open",
        "requests": [
            {
                "label": "plain_exact_blue_cat",
                "pass": True,
                "content": "blue-cat",
                "expected": "blue-cat",
            },
            {
                "label": "tool_blue_cat",
                "pass": True,
                "parsed": {"value": "blue-cat"},
                "expected": {"value": "blue-cat"},
            },
        ],
    }
    jang2l_json_sentinel = {
        "http_status": 200,
        "finish_reason": "stop",
        "content": '{"status":"ok","value":"B7-CAT-09","count":1}',
        "parsed_content": {"status": "ok", "value": "B7-CAT-09", "count": 1},
        "expected": {"status": "ok", "value": "B7-CAT-09", "count": 3},
        "usage": {"completion_tokens": 20},
        "pass": False,
    }

    artifact = build_classification(
        audit,
        smoke,
        literal_variants=literal_variants,
        jang2l_literal_variants=jang2l_literal_variants,
        jang2l_json_sentinel=jang2l_json_sentinel,
    )

    assert artifact["secondary_classification"] == (
        "jang2l_json_sentinel_semantic_mismatch_open"
    )
    assert artifact["unresolved_surfaces"]["jang2l_json_sentinel_exactness"] is True
    assert artifact["jang2l_json_sentinel_summary"]["empty_output_labels"] == []
    assert artifact["jang2l_json_sentinel_summary"]["schema_mutation_labels"] == []
    assert artifact["jang2l_json_sentinel_summary"]["semantic_mismatch_labels"] == [
        "current_json_sentinel"
    ]


def test_mimo_no_source_classifier_reads_jang2l_all_local_summary_json_sentinel():
    audit = {
        "component_ok": {
            "api_cache_responses_contract": True,
            "tool_protocol": True,
            "exactness_cache_kv_quant_excluded": True,
            "decode_speed_target": True,
            "source_vs_quant_first_divergence": False,
            "long_prompt_coherence": True,
            "cb_system_prompt_working_set_pressure": True,
            "mimo_media_wired": True,
        }
    }
    smoke = {"results": []}
    jang2l_all_local_summary = {
        "status": "fail",
        "results": [
            {
                "requests": [
                    {
                        "label": "mimo_structured_json_sentinel",
                        "content": "```json\n{\"status\":\"ok\",\"value\":\"B7-CAT-09\",\"count\":1}\n```",
                        "validation_failures": [
                            {
                                "reason": "json_exact_object_mismatch",
                                "actual": {
                                    "status": "ok",
                                    "value": "B7-CAT-09",
                                    "count": 1,
                                },
                                "expected": {
                                    "status": "ok",
                                    "value": "B7-CAT-09",
                                    "count": 3,
                                },
                            }
                        ],
                        "usage": {"completion_tokens": 24},
                    }
                ]
            }
        ],
    }

    artifact = build_classification(
        audit,
        smoke,
        jang2l_json_sentinel=jang2l_all_local_summary,
    )

    assert artifact["secondary_classification"] == (
        "jang2l_json_sentinel_semantic_mismatch_open"
    )
    assert artifact["jang2l_json_sentinel_summary"]["semantic_mismatch_labels"] == [
        "mimo_structured_json_sentinel"
    ]
    assert artifact["jang2l_json_sentinel_artifact"] == (
        "build/current-all-local-model-smoke-mimo-v25-jang2l-live-refresh-20260608/summary.json"
    )


def test_mimo_no_source_classifier_records_current_jangtq2_all_local_boundary():
    audit = {
        "component_ok": {
            "api_cache_responses_contract": True,
            "tool_protocol": True,
            "exactness_cache_kv_quant_excluded": True,
            "decode_speed_target": True,
            "source_vs_quant_first_divergence": False,
            "long_prompt_coherence": True,
            "cb_system_prompt_working_set_pressure": True,
            "mimo_media_wired": True,
        }
    }
    smoke = {"results": []}
    hyphen_distribution = {
        "classification": "compact_hyphen_exactness_fails_in_served_artifact",
        "not_caused_by": [
            "prefix_cache",
            "l2_disk_cache",
            "tool_parser",
            "chat_template_only",
            "tokenizer_roundtrip",
        ],
        "requests": {
            "chat_exact_blue_cat": {"code": 200, "text": "blue cat"},
            "completion_repeat_blue_cat": {"code": 200, "text": "blue"},
            "chat_hyphen_spaces": {"code": 200, "text": "blue - cat"},
        },
    }
    current_all_local = {
        "result": "probe_failed",
        "requests": [
            {
                "label": "text_cache_repeat_1",
                "content": "ACK",
                "validation": [],
                "usage": {"cached_tokens": 0},
            },
            {
                "label": "text_cache_repeat_2",
                "content": "ACK",
                "validation": [],
                "usage": {"cached_tokens": 67, "cache_detail": "paged"},
            },
            {
                "label": "text_cache_repeat_l2_restart",
                "content": "ACK",
                "validation": [],
                "usage": {"cached_tokens": 67, "cache_detail": "paged+disk"},
                "cache_stats": {"disk_hits": 4},
            },
            {"label": "vl_blue_image", "content": "Blue.", "validation": []},
            {"label": "video_blue_image", "content": "Blue.", "validation": []},
            {"label": "audio_blue_wave", "content": "Blue.", "validation": []},
            {
                "label": "tool_required",
                "parsed": {"value": "blue-1"},
                "expected": {"value": "blue-cat"},
                "validation": ["expected_tool_argument_missing"],
            },
            {
                "label": "mimo_tool_required_sentinel",
                "parsed": {"value": "B7CAT-09"},
                "expected": {"value": "B7-CAT-09"},
                "validation": ["expected_tool_argument_missing"],
            },
            {
                "label": "structured_json_exact",
                "parsed": {"status": "ok", "value": "bluecat", "count": 3},
                "expected": {"status": "ok", "value": "blue-cat", "count": 3},
                "validation": ["json_exact_object_mismatch"],
            },
        ],
    }

    artifact = build_classification(
        audit,
        smoke,
        jangtq2_hyphen_distribution=hyphen_distribution,
        jangtq2_current_all_local=current_all_local,
    )

    assert artifact["classification"] == (
        "jangtq2_compact_hyphen_decode_quality_open_not_cache_parser_template_tokenizer"
    )
    assert artifact["jangtq2_current_all_local_summary"]["exists"] is True
    assert artifact["jangtq2_current_all_local_summary"]["cache_l2_transport_green"] is True
    assert artifact["jangtq2_current_all_local_summary"]["media_transport_green"] is True
    assert artifact["jangtq2_current_all_local_summary"]["literal_mutation_labels"] == [
        "tool_required",
        "mimo_tool_required_sentinel",
        "structured_json_exact",
    ]
    assert (
        artifact["unresolved_surfaces"]["jangtq2_current_all_local_literal_mutation"]
        is True
    )
