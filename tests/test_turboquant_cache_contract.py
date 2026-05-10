from __future__ import annotations

import json
import os
from argparse import Namespace

import pytest


class _StopServe(RuntimeError):
    pass


def _serve_args(model_path: str, *, kv_cache_quantization):
    return Namespace(
        model=model_path,
        host="127.0.0.1",
        port=8099,
        timeout=300,
        rate_limit=0,
        api_key=None,
        enable_auto_tool_choice=False,
        tool_call_parser=None,
        reasoning_parser=None,
        default_temperature=None,
        default_top_p=None,
        default_repetition_penalty=None,
        default_enable_thinking=None,
        chat_template=None,
        chat_template_kwargs=None,
        kv_cache_quantization=kv_cache_quantization,
        kv_cache_group_size=64,
        continuous_batching=False,
        max_num_seqs=5,
        prefill_batch_size=1024,
        prefill_step_size=2048,
        completion_batch_size=1024,
        disable_prefix_cache=False,
        enable_prefix_cache=True,
        use_paged_cache=False,
        cache_memory_mb=None,
        cache_memory_percent=0.20,
        no_memory_aware_cache=False,
        prefix_cache_size=100,
        prefix_cache_max_bytes=None,
        cache_ttl_minutes=0,
        ssm_state_cache_size=8,
        ssm_state_cache_mb=512,
        paged_cache_block_size=64,
        max_cache_blocks=1000,
        enable_block_disk_cache=False,
        block_disk_cache_dir=None,
        block_disk_cache_max_gb=10.0,
        enable_disk_cache=False,
        disk_cache_dir=None,
        disk_cache_max_gb=10.0,
        enable_pld=False,
        pld_summary_interval=200,
        max_tokens=128,
        stream_interval=1,
        mcp_config=None,
        embedding_model=None,
        smelt=False,
        smelt_experts=50,
        flash_moe=False,
        flash_moe_slot_bank=64,
        flash_moe_prefetch="none",
        flash_moe_io_split=4,
        distributed=False,
        distributed_mode="pipeline",
        cluster_secret="",
        worker_nodes=None,
        speculative_model=None,
        num_draft_tokens=3,
        is_mllm=False,
        served_model_name=None,
        enable_jit=False,
        log_level="INFO",
        allowed_origins="*",
    )


def _run_serve_until_uvicorn(monkeypatch, args):
    import uvicorn
    from vmlx_engine import cli, server

    # Other endpoint tests may have already driven the global FastAPI app
    # through TestClient, which materializes Starlette's middleware stack.
    # These CLI contract tests do not exercise middleware behavior; reset the
    # stack so serve_command can add CORS just like a fresh process would.
    server.app.user_middleware.clear()
    server.app.middleware_stack = None
    monkeypatch.setattr(server, "load_model", lambda *a, **kw: None)
    monkeypatch.setattr(server, "load_embedding_model", lambda *a, **kw: None)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: (_ for _ in ()).throw(_StopServe()))

    try:
        with pytest.raises(_StopServe):
            cli.serve_command(args)
    finally:
        server.app.user_middleware.clear()
        server.app.middleware_stack = None


def test_omitted_kv_quantization_keeps_loader_turboquant_auto_enabled(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    monkeypatch.delenv("VMLX_DISABLE_TQ_KV", raising=False)
    monkeypatch.delenv("VMLINUX_FORCE_TQ_AUTO", raising=False)

    args = _serve_args(str(tmp_path), kv_cache_quantization=None)

    _run_serve_until_uvicorn(monkeypatch, args)

    assert args.kv_cache_quantization == "q4"
    assert args.kv_cache_quantization_explicit is False
    assert os.environ.get("VMLX_FORCE_TQ_AUTO") == "1"
    assert os.environ.get("VMLX_DISABLE_TQ_KV") is None


def test_plain_qwen3_moe_auto_mode_keeps_loader_turboquant_enabled(tmp_path, monkeypatch):
    """Plain KV MoE families must keep auto TQ-KV enabled."""

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_moe"}))
    monkeypatch.delenv("VMLX_DISABLE_TQ_KV", raising=False)
    monkeypatch.delenv("VMLX_FORCE_TQ_AUTO", raising=False)

    args = _serve_args(str(tmp_path), kv_cache_quantization=None)

    _run_serve_until_uvicorn(monkeypatch, args)

    assert args.kv_cache_quantization == "q4"
    assert args.kv_cache_quantization_explicit is False
    assert os.environ.get("VMLX_FORCE_TQ_AUTO") == "1"
    assert os.environ.get("VMLX_DISABLE_TQ_KV") is None


def test_qwen3_5_moe_linear_attention_disables_kv_only_turboquant(tmp_path, monkeypatch):
    """Qwen3.5/3.6 MoE with GatedDelta ArraysCache is not plain KV MoE."""

    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "layer_types": ["linear_attention", "full_attention"],
        },
    }))
    monkeypatch.delenv("VMLX_DISABLE_TQ_KV", raising=False)
    monkeypatch.delenv("VMLX_FORCE_TQ_AUTO", raising=False)
    monkeypatch.delenv("VMLX_ALLOW_HYBRID_KV_QUANT", raising=False)

    args = _serve_args(str(tmp_path), kv_cache_quantization=None)

    _run_serve_until_uvicorn(monkeypatch, args)

    assert args.kv_cache_quantization == "none"
    assert args.kv_cache_quantization_explicit is True
    assert os.environ.get("VMLX_DISABLE_TQ_KV") == "1"
    assert os.environ.get("VMLX_FORCE_TQ_AUTO") is None


def test_explicit_kv_quantization_disables_loader_turboquant(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    monkeypatch.delenv("VMLX_DISABLE_TQ_KV", raising=False)
    monkeypatch.setenv("VMLX_FORCE_TQ_AUTO", "1")

    args = _serve_args(str(tmp_path), kv_cache_quantization="q4")

    _run_serve_until_uvicorn(monkeypatch, args)

    assert args.kv_cache_quantization == "q4"
    assert args.kv_cache_quantization_explicit is True
    assert os.environ.get("VMLX_DISABLE_TQ_KV") == "1"
    assert os.environ.get("VMLINUX_FORCE_TQ_AUTO") is None


def test_hybrid_ssm_auto_mode_disables_kv_only_turboquant(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bailing_hybrid"}))
    monkeypatch.delenv("VMLX_DISABLE_TQ_KV", raising=False)
    monkeypatch.delenv("VMLINUX_FORCE_TQ_AUTO", raising=False)
    monkeypatch.delenv("VMLINUX_ALLOW_HYBRID_KV_QUANT", raising=False)

    args = _serve_args(str(tmp_path), kv_cache_quantization=None)

    _run_serve_until_uvicorn(monkeypatch, args)

    assert args.kv_cache_quantization == "none"
    assert args.kv_cache_quantization_explicit is True
    assert os.environ.get("VMLX_DISABLE_TQ_KV") == "1"
    assert os.environ.get("VMLINUX_FORCE_TQ_AUTO") is None


def test_paged_cache_warns_memory_aware_budget_flags_are_ignored(
    tmp_path,
    monkeypatch,
    caplog,
):
    import logging

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    args = _serve_args(str(tmp_path), kv_cache_quantization=None)
    args.continuous_batching = True
    args.use_paged_cache = True
    args.cache_memory_mb = 4096
    args.cache_memory_percent = 0.35

    caplog.set_level(logging.WARNING, logger="vmlx_engine.cli")

    _run_serve_until_uvicorn(monkeypatch, args)

    assert "--cache-memory-mb/--cache-memory-percent" in caplog.text
    assert "Use --max-cache-blocks" in caplog.text
