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
