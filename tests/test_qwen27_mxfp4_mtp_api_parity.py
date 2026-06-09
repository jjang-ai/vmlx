from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tests.cross_matrix import run_qwen27_mxfp4_mtp_api_parity as gate


def test_build_command_uses_current_mllm_flag() -> None:
    args = SimpleNamespace(
        python=Path(".venv/bin/python"),
        model=Path("/tmp/model"),
        port=8902,
        served_model_name="qwen-test",
        prefill_batch_size=512,
        prefill_step_size=512,
        completion_batch_size=128,
        ssm_state_cache_mb=8192,
        server_max_tokens=128,
        paged_cache_block_size=64,
        max_cache_blocks=1000,
        cache_dir=Path("/tmp/cache"),
        block_disk_cache_max_gb=2.0,
    )

    cmd = gate.build_command(args)

    assert "--is-mllm" in cmd
    assert "--mllm" not in cmd


def test_text_extractors_and_checks_recognize_api_surfaces() -> None:
    rows = {
        "responses_text": {
            "code": 200,
            "body": {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ACK"}],
                    }
                ]
            },
        },
        "responses_tool_required": {
            "code": 200,
            "body": {"output": [{"type": "function_call", "name": "record_fact"}]},
        },
        "anthropic_messages": {
            "code": 200,
            "body": {"content": [{"type": "text", "text": "ACK"}]},
        },
        "ollama_chat": {"code": 200, "body": {"message": {"content": "ACK"}}},
        "chat_stream_sse": {"code": 200},
    }
    events = [
        {"data": {"choices": [{"delta": {"content": "A"}}]}},
        {"data": {"choices": [{"delta": {"content": "CK"}}]}},
    ]

    checks = gate.build_checks(rows, events)

    assert checks["responses_text"] == {"code": 200, "text_head": "ACK"}
    assert checks["responses_tool_required"] == {
        "code": 200,
        "has_record_fact": True,
    }
    assert checks["anthropic_messages"] == {"code": 200, "text_head": "ACK"}
    assert checks["ollama_chat"] == {"code": 200, "text_head": "ACK"}
    assert checks["chat_stream_sse"]["has_ack"] is True


def test_health_rollup_preserves_observed_l2_writes_and_restart_hits() -> None:
    primary = {
        "native_cache": {"schema": "hybrid_ssm_v1"},
        "mtp": {"runtime_active": True},
        "scheduler": {"cache_hit_tokens": 2},
        "cache": {
            "block_disk_cache": {"disk_writes": 3, "disk_hits": 0},
            "ssm_companion": {"disk": {"stores": 5, "hits": 0}},
            "totals": {"l2_ssm_tokens_on_disk": 144},
        },
    }
    restart = {
        "native_cache": {"schema": "hybrid_ssm_v1"},
        "mtp": {"runtime_active": True},
        "scheduler": {"cache_hit_tokens": 0},
        "cache": {
            "block_disk_cache": {"disk_writes": 0, "disk_hits": 4},
            "ssm_companion": {"disk": {"stores": 0, "hits": 6}},
            "totals": {"l2_ssm_tokens_on_disk": 144},
        },
    }

    rollup = gate.health_cache_rollup(
        primary_health=primary,
        restart_health=restart,
        cache_hit_tokens=9,
    )

    assert rollup["scheduler"]["cache_hit_tokens"] == 9
    assert rollup["cache"]["block_disk_cache"]["disk_writes"] == 3
    assert rollup["cache"]["block_disk_cache"]["disk_hits"] == 4
    assert rollup["cache"]["ssm_companion"]["disk"]["stores"] == 5
    assert rollup["cache"]["ssm_companion"]["disk"]["hits"] == 6
    assert rollup["cache"]["ssm_companion"]["disk"]["total_tokens_on_disk"] == 144


def test_api_parity_passed_matches_release_checklist_contract() -> None:
    summary = {
        "checks": {
            "responses_text": {"code": 200, "text_head": "ACK"},
            "responses_tool_required": {"code": 200, "has_record_fact": True},
            "anthropic_messages": {"code": 200, "text_head": "ACK"},
            "ollama_chat": {"code": 200, "text_head": "ACK"},
            "chat_stream_sse": {"has_ack": True},
        },
        "health_after": {
            "native_cache": {
                "schema": "hybrid_ssm_v1",
                "cache_type": "hybrid_ssm_typed",
                "prefix": True,
                "paged": True,
                "block_disk_l2": True,
                "generic_turboquant_kv": {"enabled": True},
                "attention_kv_storage_quantization": {"enabled": True, "bits": 4},
            },
            "mtp": {
                "runtime_active": True,
                "runtime_available": True,
                "runtime_supported": True,
                "index_has_mtp_tensors": True,
                "status": "native_runtime_active",
                "effective_depth": 2,
            },
            "scheduler": {"cache_hit_tokens": 8},
            "cache": {
                "block_disk_cache": {"disk_writes": 1, "disk_hits": 1},
                "ssm_companion": {"disk": {"stores": 1, "total_tokens_on_disk": 12}},
            },
        },
    }

    assert gate.api_parity_passed(summary) is True
    summary["checks"]["responses_tool_required"]["has_record_fact"] = False
    assert gate.api_parity_passed(summary) is False
