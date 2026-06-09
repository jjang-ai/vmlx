from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tests.cross_matrix import run_qwen27_jang4m_mtp_long_context_cache_tail as gate


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        out=Path("build/current-qwen27-jang4m-mtp-installed-long-context-cache-tail-20260607.json"),
        model=Path("/models/qwen"),
        served_model_name="qwen27",
        words=5200,
        min_input_tokens=30000,
        python=Path(".venv/bin/python"),
        port=8903,
        prefill_batch_size=512,
        prefill_step_size=512,
        completion_batch_size=128,
        ssm_state_cache_mb=8192,
        max_prompt_tokens=65536,
        paged_cache_block_size=64,
        max_cache_blocks=2000,
        cache_dir=Path("/tmp/cache"),
        block_disk_cache_max_gb=8.0,
    )


def test_build_command_uses_current_mllm_and_cache_flags() -> None:
    cmd = gate.build_command(_args())

    assert "--is-mllm" in cmd
    assert "--mllm" not in cmd
    assert "--kv-cache-quantization" not in cmd
    assert "--enable-block-disk-cache" in cmd
    assert "--max-prompt-tokens" in cmd


def test_summarize_accepts_restart_backed_long_context_l2() -> None:
    args = _args()
    native = {
        "schema": "hybrid_ssm_v1",
        "cache_type": "hybrid_ssm_typed",
        "prefix": True,
        "paged": True,
        "block_disk_l2": True,
        "generic_turboquant_kv": {"enabled": True},
        "attention_kv_storage_quantization": {"enabled": True, "bits": 4},
    }
    mtp = {
        "runtime_active": True,
        "runtime_available": True,
        "runtime_supported": True,
        "index_has_mtp_tensors": True,
        "status": "native_runtime_active",
        "effective_depth": 3,
    }
    cold = {
        "text": "LONGCTX-OK",
        "response": {"usage": {"input_tokens": 31647}},
    }
    warm = {
        "text": "LONGCTX-OK",
        "response": {
            "usage": {
                "input_tokens": 31647,
                "prompt_tokens_details": {
                    "cached_tokens": 31646,
                    "cache_detail": "paged+ssm+disk",
                },
            }
        },
        "health": {
            "native_cache": native,
            "mtp": mtp,
            "turboquant_kv_cache": {"enabled": True},
        },
        "cache_stats": {
            "block_disk_cache": {
                "disk_writes": 495,
                "disk_hits": 1485,
                "total_tokens_on_disk": 31646,
            },
            "ssm_companion": {
                "disk": {"stores": 2, "total_tokens_on_disk": 63262}
            },
        },
    }

    summary = gate.summarize(args, cold, warm)

    assert summary["status"] == "pass"
    assert all(summary["checks"].values())
    assert summary["phases"]["warm"]["cache_stats"]["block_disk_cache"]["disk_hits"] == 1485


def test_restart_cache_stats_aggregate_cold_writes_and_warm_hits() -> None:
    cold = {
        "block_disk_cache": {"disk_writes": 7, "disk_hits": 0, "total_tokens_on_disk": 11},
        "ssm_companion": {
            "disk": {"stores": 3, "hits": 0, "total_tokens_on_disk": 22}
        },
    }
    warm = {
        "block_disk_cache": {"disk_writes": 0, "disk_hits": 7, "total_tokens_on_disk": 11},
        "ssm_companion": {
            "disk": {"stores": 0, "hits": 1, "total_tokens_on_disk": 22}
        },
    }

    merged = gate.aggregate_restart_cache_stats(cold, warm)

    assert merged["block_disk_cache"]["disk_writes"] == 7
    assert merged["block_disk_cache"]["disk_hits"] == 7
    assert merged["ssm_companion"]["disk"]["stores"] == 3
    assert merged["ssm_companion"]["disk"]["hits"] == 1


def test_summarize_fails_when_tail_marker_missing() -> None:
    args = _args()
    cold = {"text": "", "response": {"usage": {"input_tokens": 31647}}}
    warm = {"text": "", "response": {"usage": {"input_tokens": 31647}}}

    summary = gate.summarize(args, cold, warm)

    assert summary["status"] == "fail"
    assert summary["checks"]["cold_visible_tail_markers"] is False
