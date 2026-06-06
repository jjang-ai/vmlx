from __future__ import annotations

import json
from pathlib import Path

from tests.cross_matrix import run_mimo_v2_jang2l_current_audit as audit


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


def test_mimo_current_audit_separates_clean_artifact_from_runtime_blockers(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(audit, "STALE_TARGETS", [])
    model_parent = tmp_path / "models" / "JANGQ-AI"
    model_path = model_parent / "MiMo-V2.5-JANG_2L"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text("{}")
    manifest = tmp_path / "build" / "current-mimo-http-tb5-manifest-20260606.tsv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        f"{(model_path / 'config.json').stat().st_size}\tMiMo-V2.5-JANG_2L/config.json\n"
    )

    _write_json(
        tmp_path / "build/current-mimo-jang2l-local-structural-verify-20260606.json",
        {"status": "pass"},
    )
    _write_json(
        tmp_path / "build/current-mimo-jang2l-live-text-cache-smoke-20260606.json",
        {
            "requests": [
                {
                    "response": {
                        "http_status": 200,
                        "json": {
                            "choices": [
                                {"message": {"content": "cache ok"}, "finish_reason": "stop"}
                            ]
                        },
                    }
                }
            ]
        },
    )
    _write_json(
        tmp_path / "build/current-mimo-v2-jang2l-quantized-switchglu-parity-20260606.json",
        {
            "got_shape": [2, 3, 4096],
            "manual_shape": [2, 3, 4096],
            "max_abs_diff": 0.0007,
            "mean_abs_diff": 0.00009,
        },
    )
    _write_json(
        tmp_path / "build/current-mimo-v2-jang2l-cache-vs-nocache-next-token-20260606.json",
        {
            "rows": [
                {"mode": "no_cache", "top10": [{"id": 1, "text": "A"}]},
                {"mode": "cache_prefill", "top10": [{"id": 1, "text": "A"}]},
            ]
        },
    )
    _write_json(
        tmp_path / "build/current-mimo-v2-jang2l-direct-length-sweep-20260606.json",
        {"status": "fail", "cases": [{"status": "fail_corrupt"}]},
    )
    _write_json(
        tmp_path / "build/current-mimo-v2-jang2l-tool-dialect-failure-20260606.json",
        {"status": "fail", "runtime_observations": [{"result": "fail_incomplete_marker"}]},
    )
    _write_json(
        tmp_path
        / "build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-after-metadata-truth-20260606/summary.json",
        {
            "status": "fail",
            "results": [
                {
                    "status": "probe_failed",
                    "failures": [
                        {"label": "text_cache_repeat_1", "reason": "empty_visible"},
                    ],
                    "requests": [
                        {
                            "label": "tool_required",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "record_fact",
                                        "arguments": "{\"value\":\"blue-cat\"}",
                                    },
                                }
                            ],
                            "validation_failures": [],
                        }
                    ],
                    "cache_after": {
                        "body": {
                            "scheduler_stats": {
                                "batch_generator": {"generation_tps": 1.8}
                            }
                        }
                    },
                }
            ],
        },
    )
    _write_json(
        tmp_path / "build/current-mimo-v2-jang2l-sink-mode-length-diagnostic-20260606.json",
        {
            "status": "fail",
            "cases": [
                {"manual_sink": False, "passes_length_ok": False},
                {"manual_sink": True, "passes_length_ok": False},
            ],
        },
    )
    _write_json(
        tmp_path
        / "build/current-mimo-v2-jang2l-disable-sink-length-diagnostic-20260606.json",
        {
            "status": "fail",
            "cases": [{"passes_length_ok": False, "contains_corrupt_repetition": True}],
        },
    )
    _write_json(
        tmp_path / "build/current-mimo-v2-jang2l-prompt-shape-sweep-20260606.json",
        {
            "status": "fail",
            "results": [
                {
                    "name": "long_cache_system_exact",
                    "content": "",
                    "finish_reason": "stop",
                    "prompt_tokens": 60,
                    "completion_tokens": 1,
                },
                {
                    "name": "long_cache_no_system_exact",
                    "content": "ACK",
                    "finish_reason": "stop",
                    "prompt_tokens": 66,
                    "completion_tokens": 2,
                },
                {
                    "name": "short_system_exact",
                    "content": "ACK",
                    "finish_reason": "stop",
                    "prompt_tokens": 23,
                    "completion_tokens": 2,
                },
            ],
        },
    )
    _write_json(
        tmp_path / "build/current-mimo-v2-jang2l-rendered-prompt-compare-20260606.json",
        {
            "rows": [
                {
                    "name": "long_cache_system_exact",
                    "contains_empty_think_generation_prefix": True,
                    "rendered": "<|im_start|>assistant\n<think></think>",
                }
            ]
        },
    )
    _write_json(
        tmp_path
        / "build/current-mimo-v2-jang2l-first-token-probe-registered-20260606.json",
        {
            "status": "pass",
            "rows": [
                {
                    "name": "failing_system_long",
                    "top": [
                        {
                            "token_id": 151645,
                            "text": "<|im_end|>",
                            "is_special": True,
                        },
                        {"token_id": 4032, "text": "ACK", "is_special": False},
                    ],
                },
                {
                    "name": "working_folded_long",
                    "top": [{"token_id": 4032, "text": "ACK", "is_special": False}],
                },
                {
                    "name": "working_short_system",
                    "top": [{"token_id": 4032, "text": "ACK", "is_special": False}],
                },
            ],
        },
    )

    result = audit.build_audit(tmp_path, model_path, manifest)

    assert result["status"] == "open"
    assert result["local_release_clearance"] is False
    assert result["component_ok"]["manifest_integrity"] is True
    assert result["component_ok"]["text_cache_narrow"] is True
    assert result["component_ok"]["cache_vs_nocache_next_token"] is True
    assert result["component_ok"]["long_prompt_coherence"] is False
    assert result["component_ok"]["tool_protocol"] is True
    assert result["component_ok"]["exact_cache_prompt_following"] is False
    assert result["component_ok"]["decode_speed_target"] is False
    assert result["component_ok"]["system_prompt_first_token_stop"] is False
    assert result["component_ok"]["source_vs_quant_first_divergence"] is False
    assert result["component_ok"]["manual_sink_does_not_clear_length_generation"] is True
    assert result["component_ok"]["disable_sink_does_not_clear_length_generation"] is True
    assert result["diagnostics"]["all_local_smoke"]["tool_protocol_pass"] is True
    assert result["diagnostics"]["prompt_shape_first_token"]["blocked"] is True
    assert (
        result["diagnostics"]["prompt_shape_first_token"]["classification"]
        == "decode_loop_or_model_artifact_first_token_stop"
    )
    assert result["diagnostics"]["prompt_shape_first_token"]["failing_top_token"][
        "text"
    ] == "<|im_end|>"
    assert result["diagnostics"]["manual_sink_sdpa_clears_length_generation"] is False
    assert result["diagnostics"]["disable_sink_clears_length_generation"] is False
    assert "sink-kernel-only" in result["diagnostics"]["sink_boundary"]
    assert result["blockers"] == [
        "mimo_long_prompt_coherence_blocked",
        "mimo_exact_cache_prompt_following_blocked",
        "mimo_decode_speed_below_release_target",
        "mimo_system_prompt_first_token_stop_blocked",
        "mimo_source_vs_quant_first_divergence_missing_or_failed",
    ]
