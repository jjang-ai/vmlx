from __future__ import annotations

import json
import os
from argparse import Namespace

import pytest


class _StopServe(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _restore_cli_policy_environment():
    """Keep in-process CLI policy tests from leaking serve-time env state.

    ``serve_command`` intentionally mutates these variables for the lifetime
    of a real server process.  This module stops the command at ``uvicorn.run``
    and then continues in the same pytest process, so direct mutations made by
    the CLI are not automatically tracked by ``monkeypatch``.
    """
    names = (
        "VMLX_DISABLE_TQ_KV",
        "VMLX_FORCE_TQ_AUTO",
        "VMLX_ALLOW_HYBRID_KV_QUANT",
        "VMLX_DISABLE_SSM_DISK_RESTORE",
        "VMLX_ALLOW_UNSAFE_QWEN_SSM_DISK_RESTORE",
    )
    original = {name: os.environ.get(name) for name in names}
    try:
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


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
    monkeypatch.delenv("VMLX_FORCE_TQ_AUTO", raising=False)

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


def test_qwen3_5_moe_linear_attention_keeps_selective_live_tq_and_ssm_restore(tmp_path, monkeypatch):
    """Qwen hybrid Auto keeps selective TQ and does not disable SSM L2."""

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
    monkeypatch.delenv("VMLX_DISABLE_SSM_DISK_RESTORE", raising=False)
    monkeypatch.delenv("VMLX_ALLOW_UNSAFE_QWEN_SSM_DISK_RESTORE", raising=False)

    args = _serve_args(str(tmp_path), kv_cache_quantization=None)

    _run_serve_until_uvicorn(monkeypatch, args)

    assert args.kv_cache_quantization == "none"
    assert args.kv_cache_quantization_explicit is False
    assert os.environ.get("VMLX_DISABLE_TQ_KV") is None
    assert os.environ.get("VMLX_FORCE_TQ_AUTO") == "1"
    assert os.environ.get("VMLX_DISABLE_SSM_DISK_RESTORE") is None


def test_minimax_m3_native_msa_disables_tq_for_live_and_persisted_cache(
    tmp_path, monkeypatch
):
    """M3 idx_keys tuples must never advertise or admit generic TQ blocks."""

    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "minimax_m3_vl",
        "text_config": {
            "model_type": "minimax_m3",
            "num_hidden_layers": 60,
            "sparse_attention_config": {"use_sparse_attention": True},
        },
    }))
    monkeypatch.delenv("VMLX_DISABLE_TQ_KV", raising=False)
    monkeypatch.setenv("VMLX_FORCE_TQ_AUTO", "1")

    args = _serve_args(str(tmp_path), kv_cache_quantization=None)

    _run_serve_until_uvicorn(monkeypatch, args)

    assert args.kv_cache_quantization == "none"
    assert args.kv_cache_quantization_explicit is False
    assert os.environ.get("VMLX_DISABLE_TQ_KV") == "1"
    assert os.environ.get("VMLX_FORCE_TQ_AUTO") is None


def test_minicpm_requires_native_raw_kv(monkeypatch):
    from types import SimpleNamespace

    from vmlx_engine.cli import MiniCPMCachePolicyError, _apply_minicpm_cache_policy

    config = SimpleNamespace(
        family_name="minicpm",
        architecture_hints={
            "blocked_kv_cache_storage_quantizations": ["q4", "q8"]
        },
    )
    auto = SimpleNamespace(
        kv_cache_quantization="q4",
        kv_cache_quantization_explicit=False,
    )
    monkeypatch.setenv("VMLX_FORCE_TQ_AUTO", "1")

    assert _apply_minicpm_cache_policy(
        auto,
        config,
        __import__("logging").getLogger("test"),
    )
    assert auto.kv_cache_quantization == "none"
    assert os.environ["VMLX_DISABLE_TQ_KV"] == "1"
    assert "VMLX_FORCE_TQ_AUTO" not in os.environ

    for codec in ("q4", "q8"):
        explicit = SimpleNamespace(
            kv_cache_quantization=codec,
            kv_cache_quantization_explicit=True,
        )
        with pytest.raises(MiniCPMCachePolicyError, match=codec):
            _apply_minicpm_cache_policy(
                explicit,
                config,
                __import__("logging").getLogger("test"),
            )


def test_mimo_v2_auto_mode_keeps_prefix_cache_lossless_by_default(tmp_path, monkeypatch):
    """MiMo mixed-SWA cache must not use lossy stored q4 in auto mode."""

    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "mimo_v2",
        "num_hidden_layers": 48,
        "head_dim": 192,
        "v_head_dim": 128,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
    }))
    (tmp_path / "jang_config.json").write_text(json.dumps({
        "architecture": {
            "type": "mimo_v2",
            "text_model_type": "mimo_v2",
            "has_vision": True,
            "has_audio": True,
            "has_mtp_tensors": False,
        }
    }))
    monkeypatch.delenv("VMLX_DISABLE_TQ_KV", raising=False)
    monkeypatch.delenv("VMLX_FORCE_TQ_AUTO", raising=False)

    args = _serve_args(str(tmp_path), kv_cache_quantization=None)

    _run_serve_until_uvicorn(monkeypatch, args)

    assert args.kv_cache_quantization == "none"
    assert args.kv_cache_quantization_explicit is False
    assert os.environ.get("VMLX_FORCE_TQ_AUTO") == "1"
    assert os.environ.get("VMLX_DISABLE_TQ_KV") is None


def test_explicit_kv_quantization_disables_loader_turboquant(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    monkeypatch.delenv("VMLX_DISABLE_TQ_KV", raising=False)
    monkeypatch.setenv("VMLX_FORCE_TQ_AUTO", "1")

    args = _serve_args(str(tmp_path), kv_cache_quantization="q4")

    _run_serve_until_uvicorn(monkeypatch, args)

    assert args.kv_cache_quantization == "q4"
    assert args.kv_cache_quantization_explicit is True
    assert os.environ.get("VMLX_DISABLE_TQ_KV") == "1"
    assert os.environ.get("VMLX_FORCE_TQ_AUTO") is None


def test_hybrid_ssm_auto_mode_disables_live_tq_but_keeps_stored_kv_q4(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bailing_hybrid"}))
    monkeypatch.delenv("VMLX_DISABLE_TQ_KV", raising=False)
    monkeypatch.delenv("VMLX_FORCE_TQ_AUTO", raising=False)
    monkeypatch.delenv("VMLX_ALLOW_HYBRID_KV_QUANT", raising=False)

    args = _serve_args(str(tmp_path), kv_cache_quantization=None)

    _run_serve_until_uvicorn(monkeypatch, args)

    assert args.kv_cache_quantization == "q4"
    assert args.kv_cache_quantization_explicit is False
    assert os.environ.get("VMLX_DISABLE_TQ_KV") == "1"
    assert os.environ.get("VMLX_FORCE_TQ_AUTO") is None


def test_mimo_v2_jang_loader_skips_generic_turboquant_kv_auto_mode(monkeypatch, caplog):
    """MiMo asymmetric full/SWA cache must not be flattened to generic TQ-KV."""

    import logging

    from vmlx_engine.utils.jang_loader import _patch_turboquant_make_cache

    class FakeModel:
        layers = [object(), object()]

        def make_cache(self):
            return ["native-full", "native-swa"]

    model = FakeModel()
    original_make_cache = model.make_cache
    monkeypatch.delenv("VMLX_DISABLE_TQ_KV", raising=False)
    monkeypatch.setenv("VMLX_FORCE_TQ_AUTO", "1")
    caplog.set_level(logging.INFO, logger="vmlx_engine.utils.jang_loader")

    _patch_turboquant_make_cache(
        model,
        jang_cfg={},
        model_config={
            "model_type": "mimo_v2",
            "cache_subtype": "mimo_v2_asymmetric_swa",
        },
    )

    assert model.make_cache == original_make_cache
    assert "TurboQuant KV skipped: MiMo-V2 uses native asymmetric full/SWA" in caplog.text


def test_paged_cache_reports_cache_memory_sets_l1_byte_ceiling(
    tmp_path,
    monkeypatch,
    caplog,
):
    # Truthful-log reconciliation (paged-default-ON campaign): since the Wave-18
    # byte-ceiling work, the paged path DOES honor --cache-memory-mb/percent — it
    # computes _paged_resident_budget and passes it to PagedCacheManager as
    # max_resident_bytes (scheduler.py). The old warning claiming the flags are
    # ignored was false; the engine now logs an INFO explaining they set the L1
    # RAM byte ceiling.
    import logging

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    args = _serve_args(str(tmp_path), kv_cache_quantization=None)
    args.continuous_batching = True
    args.use_paged_cache = True
    args.cache_memory_mb = 4096
    args.cache_memory_percent = 0.35

    caplog.set_level(logging.INFO, logger="vmlx_engine.cli")

    _run_serve_until_uvicorn(monkeypatch, args)

    assert "--cache-memory-mb/--cache-memory-percent" in caplog.text
    assert "L1 RAM byte ceiling" in caplog.text
    # The stale "ignored" claim must NOT resurface.
    assert "apply only to" not in caplog.text
