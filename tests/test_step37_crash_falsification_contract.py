from __future__ import annotations

import json
from pathlib import Path

from tests.cross_matrix import run_step37_crash_falsification_contract as contract


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _native_cache() -> dict[str, object]:
    return {
        "family": "step3p7",
        "schema": "mixed_swa_kv_v1",
        "cache_type": "mixed_swa_kv",
        "cache_subtype": "step3p7_full_sliding_kv",
        "prefix": True,
        "paged": True,
        "block_disk_l2": True,
    }


def _passing_summary() -> dict[str, object]:
    labels = sorted(contract.REQUIRED_SMOKE_LABELS)
    return {
        "status": "pass",
        "completed": 1,
        "failed": 0,
        "results": [
            {
                "status": "pass",
                "row": {"name": "Step-3.7-Flash-JANG_2L"},
                "health_before": {"native_cache": _native_cache()},
                "requests": [
                    {
                        "label": label,
                        "code": 200,
                        "validation_failures": [],
                        "cache_summary": {"has_cache_hit": label == "text_cache_repeat_2"},
                    }
                    for label in labels
                ],
            }
        ],
    }


def _passing_endpoint() -> dict[str, object]:
    return {
        "health_before": {"native_cache": _native_cache()},
        "rows": [
            {
                "label": label,
                "path": path,
                "result": {"code": 200, "server_returncode": None},
                "server_alive_after": True,
            }
            for label, path in contract.REQUIRED_ENDPOINT_ROWS.items()
        ],
    }


def test_step37_crash_falsification_contract_accepts_current_bundled_proof(tmp_path):
    _write_json(tmp_path / contract.BUNDLED_SMOKE_SUMMARY, _passing_summary())
    _write_json(tmp_path / contract.BUNDLED_ENDPOINT_RESULT, _passing_endpoint())

    payload = contract.build_contract(tmp_path)

    assert payload["status"] == "pass"
    assert payload["failures"] == []
    assert payload["evidence"]["bundled_smoke"]["cache_hit_observed"] is True
    assert payload["evidence"]["bundled_endpoint"]["rows"]["legacy_completions"] == {
        "path": "/v1/completions",
        "code": 200,
        "server_alive_after": True,
        "server_returncode": None,
    }


def test_step37_crash_falsification_contract_rejects_silent_mid_request_exit(tmp_path):
    endpoint = _passing_endpoint()
    endpoint["rows"][1]["server_alive_after"] = False
    endpoint["rows"][1]["result"]["server_returncode"] = -9
    _write_json(tmp_path / contract.BUNDLED_SMOKE_SUMMARY, _passing_summary())
    _write_json(tmp_path / contract.BUNDLED_ENDPOINT_RESULT, endpoint)

    payload = contract.build_contract(tmp_path)

    assert payload["status"] == "fail"
    assert "endpoint_server_not_alive_after:chat_on" in payload["failures"]
    assert "endpoint_server_exited_during:chat_on" in payload["failures"]


def test_step37_crash_falsification_contract_rejects_missing_mixed_swa_l2_cache(tmp_path):
    summary = _passing_summary()
    summary["results"][0]["health_before"]["native_cache"]["block_disk_l2"] = False
    _write_json(tmp_path / contract.BUNDLED_SMOKE_SUMMARY, summary)
    _write_json(tmp_path / contract.BUNDLED_ENDPOINT_RESULT, _passing_endpoint())

    payload = contract.build_contract(tmp_path)

    assert payload["status"] == "fail"
    assert "bundled_smoke_native_cache_not_step37_mixed_swa_l2" in payload["failures"]


def test_step37_crash_falsification_contract_accepts_packaged_textonly_guard_proof(tmp_path):
    _write_json(
        tmp_path / contract.PACKAGED_TEXTONLY_GUARD_PROOF,
        {
            "pass": True,
            "assertions": {
                "chat_text_before_media_http_200": True,
                "chat_media_rejected_http_400": True,
                "chat_media_rejection_mentions_text_only": True,
                "chat_text_after_media_http_200": True,
                "responses_media_rejected_http_400": True,
                "responses_media_rejection_mentions_text_only": True,
                "responses_text_after_media_http_200": True,
                "server_health_after_http_200": True,
            },
        },
    )

    payload = contract.build_contract(tmp_path)

    assert payload["status"] == "pass"
    assert payload["failures"] == []
    assert payload["evidence"]["packaged_textonly_guard"]["pass"] is True
