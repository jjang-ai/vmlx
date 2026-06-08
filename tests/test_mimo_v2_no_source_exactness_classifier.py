from tests.cross_matrix.run_mimo_v2_no_source_exactness_classifier import (
    build_classification,
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

    artifact = build_classification(audit, smoke, literal_variants=literal_variants)

    assert artifact["classification"] == (
        "jangtq2_plain_literal_copy_fails_before_parser_or_json_repair"
    )
    assert artifact["literal_variant_summary"]["plain_literal_copy_pass"] is False
    assert artifact["literal_variant_summary"]["structured_literal_pass"] is False
    assert artifact["literal_variant_summary"]["tool_literal_pass"] is False
    assert artifact["unresolved_surfaces"]["jangtq2_plain_literal_copy"] is True
