from __future__ import annotations


def _policy_row(
    *,
    target_temp: float,
    draft_temp: float,
    acceptances: list[float],
    top_k: int = 20,
) -> dict:
    return {
        "target_settings": {
            "temperature": target_temp,
            "top_p": 0.95,
            "min_p": 0.0,
            "min_tokens_to_keep": 1,
            "top_k": top_k,
        },
        "draft_settings": {
            "temperature": draft_temp,
            "top_p": 0.95,
            "min_p": 0.0,
            "min_tokens_to_keep": 1,
            "top_k": top_k,
        },
        "depths": [
            {
                "depth": index + 1,
                "expected_acceptance_rate": acceptance,
                "target_support": 20,
                "draft_support": 20,
            }
            for index, acceptance in enumerate(acceptances)
        ],
        "mean_acceptance": sum(acceptances) / len(acceptances),
        "worst_acceptance": min(acceptances),
        "mean_target_support": 20,
        "mean_draft_support": 20,
        "min_target_support": 20,
        "min_draft_support": 20,
    }


def test_score_policy_cost_penalizes_low_value_third_depth():
    from vmlx_engine.native_mtp_policy_suite import score_policy_cost

    good = _policy_row(target_temp=0.2, draft_temp=0.2, acceptances=[1.0, 1.0, 1.0])
    collapsed = _policy_row(
        target_temp=0.2,
        draft_temp=0.2,
        acceptances=[0.01, 0.66, 0.0],
    )

    good_score = score_policy_cost(good, verify_ms=60.0, draft_ms_per_depth=5.0)
    collapsed_d2 = score_policy_cost(
        collapsed,
        verify_ms=60.0,
        draft_ms_per_depth=5.0,
        depth_limit=2,
    )
    collapsed_d3 = score_policy_cost(
        collapsed,
        verify_ms=60.0,
        draft_ms_per_depth=5.0,
        depth_limit=3,
    )

    assert good_score["expected_output_tokens"] == 4.0
    assert good_score["cost_ms"] == 75.0
    assert round(good_score["tokens_per_ms"], 6) == round(4.0 / 75.0, 6)
    assert collapsed_d3["marginal_last_depth_tokens"] == 0.0
    assert collapsed_d2["tokens_per_ms"] > collapsed_d3["tokens_per_ms"]


def test_aggregate_policy_suite_ranks_common_policy_by_cost():
    from vmlx_engine.native_mtp_policy_suite import aggregate_policy_suite

    stable = _policy_row(target_temp=0.2, draft_temp=0.2, acceptances=[0.95, 0.90, 0.85])
    flashy_bad = _policy_row(
        target_temp=0.8,
        draft_temp=0.9,
        acceptances=[1.0, 1.0, 0.0],
    )
    suite = [
        {
            "prompt_class": "deterministic",
            "artifact": "det.json",
            "result": {
                "all_argmax_equal": True,
                "real_logit_policy_sweep": {"rows": [stable, flashy_bad]},
            },
        },
        {
            "prompt_class": "creative",
            "artifact": "creative.json",
            "result": {
                "all_argmax_equal": True,
                "real_logit_policy_sweep": {
                    "rows": [
                        _policy_row(
                            target_temp=0.2,
                            draft_temp=0.2,
                            acceptances=[0.70, 0.60, 0.50],
                        ),
                        _policy_row(
                            target_temp=0.8,
                            draft_temp=0.9,
                            acceptances=[0.30, 0.20, 0.0],
                        ),
                    ]
                },
            },
        },
    ]

    aggregate = aggregate_policy_suite(
        suite,
        verify_ms=60.0,
        draft_ms_per_depth=5.0,
        top_n=2,
    )

    assert aggregate["prompt_count"] == 2
    assert aggregate["cost_model"]["verify_ms"] == 60.0
    assert aggregate["top"][0]["target_settings"]["temperature"] == 0.2
    assert aggregate["top"][0]["prompt_count"] == 2
    assert aggregate["coverage"]["present_classes"] == ["creative", "deterministic"]
    assert "vl_image" in aggregate["coverage"]["missing_classes"]
    assert "video" in aggregate["coverage"]["missing_classes"]
    assert "cache_repeat" in aggregate["coverage"]["missing_classes"]
    assert aggregate["readiness"]["d3_shadow_ready"] is False
    assert "missing_required_classes" in aggregate["readiness"]["reasons"]
    assert aggregate["top"][0]["worst_d3_acceptance"] == 0.50
    assert aggregate["top"][0]["mean_tokens_per_ms_d3"] > aggregate["top"][1]["mean_tokens_per_ms_d3"]


def test_read_model_policy_metadata_separates_trained_and_sampler_top_k(tmp_path):
    import json

    from vmlx_engine.native_mtp_policy_suite import read_model_policy_metadata

    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_experts": 256,
            "num_experts_per_tok": 8,
        },
    }))
    (tmp_path / "generation_config.json").write_text(json.dumps({
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.02,
        "do_sample": True,
    }))

    metadata = read_model_policy_metadata(tmp_path)

    assert metadata["trained_routing_top_k"] == {
        "active_experts": 8,
        "source": "config.text_config.num_experts_per_tok",
        "n_routed_experts": 256,
        "n_routed_experts_source": "config.text_config.num_experts",
    }
    assert metadata["generation_config"]["exists"] is True
    assert metadata["generation_config"]["sampling"] == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.02,
        "repetition_penalty": None,
        "do_sample": True,
    }


def test_read_model_policy_metadata_audits_qwen_stop_token_set(tmp_path):
    import json

    from vmlx_engine.native_mtp_policy_suite import read_model_policy_metadata

    added_tokens = {
        "248044": {"content": "<|endoftext|>", "special": True},
        "248046": {"content": "<|im_end|>", "special": True},
    }
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "eos_token_id": 248046,
            "mtp_num_hidden_layers": 1,
        },
    }))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "added_tokens_decoder": added_tokens,
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
    }))
    (tmp_path / "generation_config.json").write_text(json.dumps({
        "eos_token_id": [248046, 248046],
        "pad_token_id": 248044,
        "top_k": 20,
    }))

    metadata = read_model_policy_metadata(tmp_path)

    assert metadata["stop_token_audit"]["required_token_ids"] == [248046, 248044]
    assert metadata["stop_token_audit"]["generation_eos_token_ids"] == [
        248046,
        248046,
    ]
    assert metadata["stop_token_audit"]["deduplicated_generation_eos_token_ids"] == [
        248046,
    ]
    assert metadata["stop_token_audit"]["missing_required_token_ids"] == [248044]
    assert metadata["stop_token_audit"]["duplicate_token_ids"] == [248046]
    assert metadata["stop_token_audit"]["stop_config_clean"] is False
    assert metadata["metadata_readiness"]["ready"] is False
    assert "duplicate_generation_eos_token_id" in metadata["metadata_readiness"]["reasons"]
    assert (
        "missing_required_generation_eos_token_id"
        in metadata["metadata_readiness"]["reasons"]
    )


def test_policy_suite_markdown_surfaces_stop_token_audit(tmp_path):
    from bench.native_mtp_policy_suite_summary import _write_markdown

    summary = {
        "model_metadata": {
            "model_path": "/models/qwen",
            "model_type": "qwen3_5",
            "text_model_type": "qwen3_5_text",
            "trained_routing_top_k": {},
            "generation_config": {
                "sampling": {
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 20,
                    "min_p": None,
                    "do_sample": True,
                }
            },
            "stop_token_audit": {
                "stop_config_clean": False,
                "generation_eos_token_ids": [248046, 248046],
                "deduplicated_generation_eos_token_ids": [248046],
                "required_token_ids": [248046, 248044],
                "missing_required_token_ids": [248044],
                "duplicate_token_ids": [248046],
                "issues": [
                    "duplicate_generation_eos_token_id",
                    "missing_required_generation_eos_token_id",
                ],
            },
            "metadata_readiness": {
                "ready": False,
                "reasons": [
                    "duplicate_generation_eos_token_id",
                    "missing_required_generation_eos_token_id",
                ],
            },
        },
        "aggregate": {
            "prompt_count": 1,
            "coverage": {
                "present_classes": ["deterministic"],
                "missing_classes": ["vl_image"],
                "has_vl_image": False,
                "has_video": False,
                "has_cache_repeat": False,
            },
            "readiness": {
                "d3_shadow_ready": False,
                "reasons": ["missing_required_classes"],
            },
            "cost_model": {
                "verify_ms": 60.0,
                "draft_ms_per_depth": 5.0,
                "fixed_ms": 0.0,
            },
            "top": [],
        },
    }
    path = tmp_path / "SUMMARY.md"

    _write_markdown(summary, path)

    text = path.read_text()
    assert "Stop-token config clean: `False`" in text
    assert "generation EOS token IDs: `[248046, 248046]`" in text
    assert "missing required EOS token IDs: `[248044]`" in text
    assert "duplicate EOS token IDs: `[248046]`" in text
