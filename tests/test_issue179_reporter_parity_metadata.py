# SPDX-License-Identifier: Apache-2.0
"""Contracts for the MiniMax-K issue #179 reporter parity metadata package."""

import json

from tests.cross_matrix import run_issue179_reporter_parity_metadata as parity


def test_issue179_reporter_parity_metadata_collects_required_fields(tmp_path):
    server = tmp_path / "server.py"
    server.write_text(
        '@app.post("/v1/responses/{response_id}/cancel")\n'
        "async def cancel_response():\n"
        "    await _engine.abort_request(response_id)\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "pass",
                "files": [
                    {"path": "config.json", "sha256": "cfg"},
                    {"path": "model-00001-of-00067.safetensors", "sha256": "shard"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    session = tmp_path / "session.json"
    session.write_text(
        json.dumps(
            {
                "wireApi": "responses",
                "detectedFamily": "minimax",
                "route": "/v1/responses",
                "stream": True,
                "sessionHasReasoningParser": True,
                "sampling": {
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_tokens": 4096,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    lifecycle = tmp_path / "lifecycle.json"
    lifecycle.write_text(
        json.dumps(
            {
                "request_error": {
                    "code": "ECONNRESET",
                    "fullContentLen": 0,
                    "readerAcquired": True,
                },
                "request_error_before_visible_content": True,
                "response_active_at_cancel": False,
                "responses_cancel_404_after_econnreset_same_response_id": True,
                "responses_cancel_404_after_request_error": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out = parity.build_metadata(
        installed_server_path=server,
        model_manifest_path=manifest,
        capture_provenance="reporter_machine",
        chat_id="33a744d8",
        session_settings_path=session,
        response_id="resp_66d7e36b833e",
        response_active_at_cancel=False,
        raw_sse_cancel_lifecycle_path=lifecycle,
    )

    assert out["status"] == "pass"
    assert out["capture_provenance"] == "reporter_machine"
    assert out["server_has_responses_cancel_route"] is True
    assert out["server_cancel_calls_engine_abort"] is True
    assert out["model_manifest_sha256"]
    assert out["model_file_hashes"] == [
        {"path": "config.json", "sha256": "cfg"},
        {"path": "model-00001-of-00067.safetensors", "sha256": "shard"},
    ]
    assert out["chat_id"] == "33a744d8"
    assert out["session_settings"] == {
        "wireApi": "responses",
        "detectedFamily": "minimax",
        "route": "/v1/responses",
        "stream": True,
        "sessionHasReasoningParser": True,
        "sampling": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 40,
            "max_tokens": 4096,
        },
    }
    assert out["response_id"] == "resp_66d7e36b833e"
    assert out["response_active_at_cancel"] is False
    assert out["raw_sse_cancel_lifecycle"] == {
        "request_error": {
            "code": "ECONNRESET",
            "fullContentLen": 0,
            "readerAcquired": True,
        },
        "request_error_before_visible_content": True,
        "response_active_at_cancel": False,
        "responses_cancel_404_after_econnreset_same_response_id": True,
        "responses_cancel_404_after_request_error": True,
    }
    assert out["missing_fields"] == []


def test_issue179_reporter_parity_metadata_marks_missing_required_fields(tmp_path):
    server = tmp_path / "server.py"
    server.write_text("print('no cancel route')\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"status": "missing"}) + "\n", encoding="utf-8")

    out = parity.build_metadata(
        installed_server_path=server,
        model_manifest_path=manifest,
        capture_provenance="local_template",
        chat_id="",
        session_settings_path=None,
        response_id="",
        response_active_at_cancel=None,
        raw_sse_cancel_lifecycle_path=None,
    )

    assert out["status"] == "open"
    assert "capture_provenance" in out["missing_fields"]
    assert "server_has_responses_cancel_route" in out["missing_fields"]
    assert "server_cancel_calls_engine_abort" in out["missing_fields"]
    assert "model_file_hashes" in out["missing_fields"]
    assert "chat_id" in out["missing_fields"]
    assert "session_settings" in out["missing_fields"]
    assert "response_id" in out["missing_fields"]
    assert "response_active_at_cancel" in out["missing_fields"]
    assert "raw_sse_cancel_lifecycle" in out["missing_fields"]


def test_issue179_reporter_parity_metadata_rejects_wrong_session_shape(tmp_path):
    server = tmp_path / "server.py"
    server.write_text(
        '@app.post("/v1/responses/{response_id}/cancel")\n'
        "async def cancel_response():\n"
        "    await _engine.abort_request(response_id)\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"files": [{"path": "config.json", "sha256": "cfg"}]}) + "\n",
        encoding="utf-8",
    )
    session = tmp_path / "session.json"
    session.write_text(
        json.dumps(
            {
                "wireApi": "chat",
                "detectedFamily": "qwen",
                "route": "/v1/chat/completions",
                "stream": False,
                "sessionHasReasoningParser": False,
                "sampling": {"temperature": 0.2, "top_p": 0.9, "top_k": 20},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    lifecycle = tmp_path / "lifecycle.json"
    lifecycle.write_text(
        json.dumps(
            {
                "request_error": {
                    "code": "ECONNRESET",
                    "fullContentLen": 0,
                    "readerAcquired": True,
                },
                "request_error_before_visible_content": True,
                "responses_cancel_404_after_econnreset_same_response_id": True,
                "responses_cancel_404_after_request_error": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out = parity.build_metadata(
        installed_server_path=server,
        model_manifest_path=manifest,
        capture_provenance="reporter_machine",
        chat_id="33a744d8",
        session_settings_path=session,
        response_id="resp_66d7e36b833e",
        response_active_at_cancel=False,
        raw_sse_cancel_lifecycle_path=lifecycle,
    )

    assert out["status"] == "open"
    assert "session_settings_shape" in out["missing_fields"]


def test_issue179_reporter_parity_metadata_rejects_wrong_cancel_lifecycle_shape(tmp_path):
    server = tmp_path / "server.py"
    server.write_text(
        '@app.post("/v1/responses/{response_id}/cancel")\n'
        "async def cancel_response():\n"
        "    await _engine.abort_request(response_id)\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"files": [{"path": "config.json", "sha256": "cfg"}]}) + "\n",
        encoding="utf-8",
    )
    session = tmp_path / "session.json"
    session.write_text(
        json.dumps(
            {
                "wireApi": "responses",
                "detectedFamily": "minimax",
                "route": "/v1/responses",
                "stream": True,
                "sessionHasReasoningParser": True,
                "sampling": {
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_tokens": 4096,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    lifecycle = tmp_path / "lifecycle.json"
    lifecycle.write_text(
        json.dumps(
            {
                "request_error": {
                    "code": "ETIMEOUT",
                    "fullContentLen": 12,
                    "readerAcquired": False,
                },
                "request_error_before_visible_content": False,
                "responses_cancel_404_after_econnreset_same_response_id": False,
                "responses_cancel_404_after_request_error": False,
                "response_active_at_cancel": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out = parity.build_metadata(
        installed_server_path=server,
        model_manifest_path=manifest,
        capture_provenance="reporter_machine",
        chat_id="33a744d8",
        session_settings_path=session,
        response_id="resp_66d7e36b833e",
        response_active_at_cancel=False,
        raw_sse_cancel_lifecycle_path=lifecycle,
    )

    assert out["status"] == "open"
    assert "raw_sse_cancel_lifecycle_shape" in out["missing_fields"]
    assert "response_active_at_cancel_consistency" in out["missing_fields"]
