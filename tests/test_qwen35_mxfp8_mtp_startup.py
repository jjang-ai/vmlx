from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tests.cross_matrix import run_qwen35_mxfp8_mtp_startup as gate


def test_build_command_uses_current_mllm_flag_and_cache() -> None:
    args = SimpleNamespace(
        python=Path(".venv/bin/python"),
        model=Path("/models/qwen35"),
        port=8904,
        served_model_name="qwen35",
        ssm_state_cache_mb=8192,
        cache_dir=Path("/tmp/cache"),
        block_disk_cache_max_gb=4.0,
    )

    cmd = gate.build_command(args)

    assert "--is-mllm" in cmd
    assert "--mllm" not in cmd
    assert "--enable-block-disk-cache" in cmd
    assert "--use-paged-cache" in cmd
    assert "--kv-cache-quantization" not in cmd


def test_startup_checks_match_release_checklist_contract() -> None:
    health = {
        "model_loaded": True,
        "model_name": "JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP",
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
            "effective_depth": 3,
        },
        "routing": {"trained_active_experts": 8, "effective_active_experts": 8},
    }

    checks = gate.startup_checks(health)

    assert checks == {
        "model_loaded": True,
        "mtp_runtime_active": True,
        "mtp_depth_three": True,
        "native_hybrid_cache": True,
        "trained_k_preserved": True,
    }


def test_startup_checks_fail_when_router_k_is_changed() -> None:
    health = {
        "model_loaded": True,
        "model_name": "JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP",
        "native_cache": {},
        "mtp": {},
        "routing": {"trained_active_experts": 8, "effective_active_experts": 4},
    }

    checks = gate.startup_checks(health)

    assert checks["trained_k_preserved"] is False
