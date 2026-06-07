# SPDX-License-Identifier: Apache-2.0
"""Contracts for live installed-app runtime evidence on issues #175-#177."""

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_issue175_177_live_runtime_audit_accepts_qwen_installed_cache_hit(tmp_path):
    from tests.cross_matrix import run_issue175_177_live_runtime_audit as gate

    qwen_source = Path("qwen-live.json")
    minimax_source = Path("minimax-live.json")
    minimax_restart_source = Path("minimax-restart.json")
    admin_sleep_source = Path("admin-sleep.json")
    installed_python = (
        "/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3"
    )

    _write_json(
        tmp_path / qwen_source,
        {
            "status": "pass",
            "python": installed_python,
            "cache_proof_policy": {"capacity_projection_valid": True},
            "results": [
                {
                    "http_timing": {"ttft_seconds": 2.0},
                    "memory": {"metal_active_before_gb": 50.0},
                    "response_rails": {"has_visible_content": True},
                },
                {
                    "http_timing": {"ttft_seconds": 0.5},
                    "memory": {"metal_active_after_gb": 49.5},
                    "response_rails": {"has_visible_content": True},
                    "usage_summary": {
                        "cached_tokens": 128,
                        "cache_detail": "paged+ssm",
                    },
                    "cache_capacity_projection": {
                        "basis": "observed_block_disk_l2",
                    },
                    "after": {
                        "cache_stats": {
                            "body": {"block_disk_cache": {"disk_hits": 3}}
                        }
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path / minimax_source,
        {
            "status": "pass",
            "python": installed_python,
            "results": [
                {
                    "http_timing": {"ttft_seconds": 2.0},
                    "response_rails": {"has_visible_content": True},
                },
                {
                    "http_timing": {"ttft_seconds": 0.4},
                    "response_rails": {"has_visible_content": True},
                    "usage_summary": {
                        "cached_tokens": 96,
                        "cache_detail": "paged+tq",
                    },
                    "after": {
                        "cache_stats": {
                            "body": {"block_disk_cache": {"disk_hits": 2}}
                        },
                        "health": {
                            "body": {
                                "scheduler": {
                                    "last_cache_selection": {"selected": "paged"},
                                    "last_cache_execution": {
                                        "cache_detail": "paged+tq",
                                        "cached_tokens": 96,
                                        "reconstruction_seconds": 0.01,
                                        "dequantization_seconds": 0.02,
                                        "total_worker_cache_seconds": 0.03,
                                    },
                                },
                                "turboquant_kv_cache": {
                                    "enabled": True,
                                    "batch_api": "turboquant_kv_v1",
                                },
                            }
                        },
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path / minimax_restart_source,
        {
            "status": "pass",
            "python": installed_python,
            "results": [
                {
                    "http_timing": {"ttft_seconds": 0.8},
                    "response_rails": {"has_visible_content": True},
                    "usage_summary": {
                        "cached_tokens": 80,
                        "cache_detail": "paged+disk+tq",
                    },
                    "after": {
                        "cache_stats": {
                            "body": {"block_disk_cache": {"disk_hits": 4}}
                        },
                        "health": {
                            "body": {
                                "scheduler": {
                                    "last_cache_selection": {
                                        "selected": "paged",
                                        "paged_cold_tokens": 80,
                                    },
                                    "last_cache_execution": {
                                        "cache_detail": "paged",
                                        "cached_tokens": 80,
                                        "reconstruction_seconds": 0.01,
                                        "dequantization_seconds": 0.02,
                                        "total_worker_cache_seconds": 0.03,
                                    },
                                }
                            }
                        },
                    },
                }
            ],
        },
    )
    _write_json(
        tmp_path / admin_sleep_source,
        {
            "status": "pass",
            "classification": {"status": "pass"},
            "health_deep_sleep": {
                "body": {
                    "model_loaded": False,
                    "status": "standby_deep",
                }
            },
        },
    )

    audit = gate.build_audit(
        tmp_path,
        source=qwen_source,
        minimax_source=minimax_source,
        minimax_restart_reader_source=minimax_restart_source,
        admin_sleep_source=admin_sleep_source,
    )

    assert audit["status"] == "pass"
    assert audit["checks"]["installed_app_qwen_live_probe_passed"] is True
    assert audit["checks"]["cache_hit_ttft_improved"] is True
    assert audit["checks"]["l2_block_disk_hits_observed"] is True
    assert audit["checks"]["paged_ssm_cache_hit_observed"] is True
    assert audit["checks"]["visible_content_observed"] is True
    assert audit["checks"]["minimax_installed_live_probe_passed"] is True
    assert audit["checks"]["turboquant_live_decode_observed"] is True
    assert audit["checks"]["scheduler_cache_selection_observed"] is True
    assert audit["checks"]["scheduler_cache_execution_timing_observed"] is True
    assert audit["checks"]["paged_tq_cache_hit_observed"] is True
    assert audit["checks"]["minimax_restart_reader_probe_passed"] is True
    assert audit["checks"]["cold_paged_tq_restart_hit_observed"] is True
    assert audit["checks"]["cold_paged_cache_selection_observed"] is True
    assert audit["checks"]["cold_paged_stream_ttft_observed"] is True
    assert audit["checks"]["admin_sleep_lifecycle_probe_passed"] is True
    assert audit["checks"]["admin_sleep_deep_unload_observed"] is True
    assert audit["remaining_blockers"] == []


def test_issue175_177_live_runtime_audit_writes_json_artifact(tmp_path):
    from tests.cross_matrix import run_issue175_177_live_runtime_audit as gate

    out = tmp_path / "issue175-177-live-runtime.json"
    audit = gate.write_audit(Path("."), out)

    assert out.exists()
    assert audit["artifact"] == str(out)
    assert '"checks"' in out.read_text(encoding="utf-8")
