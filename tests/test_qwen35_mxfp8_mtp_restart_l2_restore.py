from pathlib import Path

from tests.cross_matrix import run_qwen35_mxfp8_mtp_restart_l2_restore as runner


def _native_cache() -> dict:
    return {
        "schema": "hybrid_ssm_v1",
        "cache_type": "hybrid_ssm_typed",
        "prefix": True,
        "paged": True,
        "block_disk_l2": True,
    }


def _mtp() -> dict:
    return {
        "runtime_active": True,
        "status": "native_runtime_active",
        "effective_depth": 3,
    }


def test_qwen35_restart_l2_runner_defaults_to_checklist_artifact():
    args = runner.parse_args([])
    assert args.out == runner.REPO / Path(
        "build/current-qwen35-mxfp8-mtp-restart-l2-restore-20260607/summary.json"
    )
    assert args.served_model_name == "qwen3.6-35b-mxfp8-mtp-restart-l2"


def test_qwen35_restart_l2_runner_requires_block_ssm_and_disk_cache_detail():
    summary = {
        "phases": {
            "phase1": {
                "response": {"code": 200},
                "block_disk_cache": {"disk_writes": 3},
                "ssm_companion_disk": {"stores": 2},
            },
            "phase2": {
                "response": {"code": 200},
                "usage": {
                    "prompt_tokens_details": {
                        "cached_tokens": 128,
                        "cache_detail": "paged+ssm+disk",
                    }
                },
                "block_disk_cache": {"disk_hits": 3},
                "ssm_companion_disk": {"hits": 1},
                "native_cache": _native_cache(),
                "mtp": _mtp(),
            },
        }
    }
    assert runner._passed(summary) is True

    summary["phases"]["phase2"]["usage"]["prompt_tokens_details"][
        "cache_detail"
    ] = "paged+ssm"
    assert runner._passed(summary) is False


def test_qwen35_restart_l2_runner_rejects_missing_mtp():
    summary = {
        "phases": {
            "phase1": {
                "response": {"code": 200},
                "block_disk_cache": {"disk_writes": 1},
                "ssm_companion_disk": {"stores": 1},
            },
            "phase2": {
                "response": {"code": 200},
                "usage": {
                    "prompt_tokens_details": {
                        "cached_tokens": 64,
                        "cache_detail": "paged+ssm+disk",
                    }
                },
                "block_disk_cache": {"disk_hits": 1},
                "ssm_companion_disk": {"hits": 1},
                "native_cache": _native_cache(),
                "mtp": {"runtime_active": False},
            },
        }
    }
    assert runner._passed(summary) is False
