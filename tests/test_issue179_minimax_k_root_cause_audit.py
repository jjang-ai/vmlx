# SPDX-License-Identifier: Apache-2.0
"""Contracts for the MiniMax-K issue #179 no-heavy root-cause audit."""

import json
from pathlib import Path

from tests.cross_matrix import run_issue179_minimax_k_root_cause_audit as gate
from tests.cross_matrix import run_issue179_public_dmg_contract as dmg_gate


def test_issue179_audit_keeps_reporter_cancel_404_boundary_open():
    audit = gate.build_audit(Path("."))

    assert audit["status"] == "open"
    assert audit["reporter"]["responses_cancel_404_seen"] is True
    assert audit["reporter"]["request_error_before_visible_content"] is True
    assert audit["reporter"]["responses_cancel_404_after_request_error"] is True
    assert audit["reporter"]["installed_bundled_python_seen"] is True
    assert audit["proven"]["reporter_log_installed_app_bundled_python_seen"] is True
    assert audit["reporter"]["request_shape"]["wireApi"] == "responses"
    assert audit["reporter"]["request_shape"]["body"]["route"] == "/v1/responses"
    assert audit["reporter"]["request_shape"]["body"]["stream"] is True
    assert audit["reporter"]["request_shape"]["detectedFamily"] == "minimax"
    assert audit["reporter"]["request_shape"]["sessionHasReasoningParser"] is True
    assert audit["reporter"]["resolved_sampling_kwargs"] == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "max_tokens": 4096,
    }
    assert audit["proven"]["reporter_log_request_shape_and_sampling_kwargs_seen"] is True
    assert audit["reporter"]["launch_config"] == {
        "executable": "/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3",
        "module": "vmlx_engine.cli",
        "subcommand": "serve",
        "model_path": "/Users/example/.omlx/models/MiniMax-M2.7-JANGTQ_K",
        "host": "127.0.0.1",
        "port": 8000,
        "timeout": 300,
        "max_num_seqs": 1,
        "prefill_batch_size": 512,
        "prefill_step_size": 2048,
        "completion_batch_size": 512,
        "continuous_batching": True,
        "tool_call_parser": "minimax",
        "enable_auto_tool_choice": True,
        "reasoning_parser": "minimax_m2",
        "use_paged_cache": True,
        "paged_cache_block_size": 64,
        "max_cache_blocks": 1000,
        "enable_block_disk_cache": True,
        "block_disk_cache_max_gb": 10,
        "stream_interval": 1,
    }
    assert audit["reporter"]["model_config"] == {
        "detection_source": "jang_stamped",
        "family": "minimax",
        "reasoning_parser": "minimax_m2",
        "tool_parser": "minimax",
        "think_in_template": True,
        "cache_type": "kv",
        "cache_subtype": None,
        "is_mllm": False,
    }
    assert audit["reporter"]["runtime_config"] == {
        "native_tool_format": "minimax",
        "reasoning_parser": "minimax_m2",
        "paged_cache": {"block_size": 64, "max_blocks": 1000},
        "block_disk_cache_max_gb": 10.0,
        "kv_cache_quantization": {"bits": 4, "group_size": 64},
        "runtime_cache_all_turboquant": True,
        "single_active_scheduler": True,
        "default_max_tokens_fallback": 4096,
    }
    assert audit["proven"]["reporter_log_launch_parser_cache_flags_seen"] is True
    assert audit["reporter"]["request_error"] == {
        "chatId": "33a744d8",
        "message": "aborted",
        "name": "Error",
        "code": "ECONNRESET",
        "timedOut": False,
        "fullContentLen": 0,
        "readerAcquired": True,
    }
    assert (
        audit["reporter"]["responses_cancel_404_after_econnreset_same_response_id"]
        is True
    )
    assert audit["reporter"]["responses_cancel_404_response_id"] == "resp_66d7e36b833e"
    assert audit["reporter"]["request_error_response_id"] == "resp_66d7e36b833e"
    assert audit["reporter"]["bad_text_captured_in_log"] is False
    assert audit["reporter"]["prompt_chars"] == 6
    assert audit["reporter_screenshot"]["manual_observation"]["surface"] == "reasoning_panel"
    assert audit["reporter_screenshot"]["manual_observation"]["interrupted"] is True
    assert audit["reporter_screenshot"]["manual_observation"]["looks_like_numeric_sequence_garbage"] is True
    assert audit["local_real_ui"]["all_required_clean"] is True
    assert audit["local_real_ui"]["clean_count"] == audit["local_real_ui"]["required_count"]
    assert audit["local_real_ui"]["installed_session_settings_parity"] == {
        "installed_proof_count": 4,
        "all_installed_use_bundled_python": True,
        "all_installed_use_responses": True,
        "all_installed_disable_builtin_tools": True,
        "all_installed_server_cache_controls_disabled": True,
        "all_installed_have_cache_flags": True,
        "has_thinking_512_code_clean": True,
        "has_hi_auto_clean_repeat": True,
    }
    assert audit["proven"]["local_installed_issue179_session_settings_parity"] is True
    assert any(
        proof["path"].endswith("issue179-hi-thinking-20260527-proof.json")
        and proof["clean"]
        and "reasoning_display" in proof["surfaces"]
        and proof["reasoning_leaks"] == {
            "raw": False,
            "cjk": 0,
            "korean": 0,
            "numeric": 0,
        }
        for proof in audit["local_real_ui"]["proofs"]
    )
    assert audit["local_installed_bundle_contract"]["server_has_responses_cancel_route"] is True
    assert audit["bundle_hash_parity"] == {
        "source_server_sha256": audit["source_contract"]["source_hashes"][
            "vmlx_engine/server.py"
        ],
        "local_installed_server_sha256": audit["local_installed_bundle_contract"][
            "sha256"
        ],
        "public_v1549_tahoe_server_sha256": audit["public_release_dmg_contract"][
            "server_sha256"
        ],
        "source_matches_local_installed": True,
        "source_matches_public_v1549_tahoe": False,
        "local_installed_matches_public_v1549_tahoe": False,
    }
    assert audit["reporter_parity_artifact"]["path"] == (
        "build/issue-179/reporter-parity-metadata-20260527.json"
    )
    assert audit["reporter_parity_artifact"]["exists"] is True
    assert audit["reporter_parity_artifact"]["status"] == "pass"
    assert audit["reporter_parity_artifact"]["capture_provenance"] == "reporter_machine"
    assert audit["reporter_parity_artifact"]["missing_fields"] == []
    assert audit["reporter_parity_artifact"]["collector_missing_fields"] == []
    assert (
        audit["reporter_parity_artifact"]["comparison_status"]
        == "ready_for_direct_comparison"
    )
    assert audit["reporter_parity_comparison"] == {
        "status": "open",
        "capture_provenance_is_reporter_machine": True,
        "server_hash_matches_local_installed": False,
        "server_route_markers_match": True,
        "model_manifest_sha256_matches_local": True,
        "model_file_hashes_match_local": True,
        "chat_id_matches_reporter_log": True,
        "response_id_matches_reporter_log": True,
        "response_active_at_cancel_recorded": True,
        "raw_sse_cancel_lifecycle_present": True,
        "failures": ["server_hash_matches_local_installed"],
    }
    assert audit["reporter_server_hash_parity"]["status"] == "open"
    assert (
        audit["reporter_server_hash_parity"]["failure"]
        == "reporter_installed_server_hash_drift"
    )
    assert audit["reporter_server_hash_parity"]["route_markers_match"] is True
    assert (
        audit["reporter_server_hash_parity"]["provenance"]["status"]
        == "open"
    )
    assert (
        audit["reporter_server_hash_parity"]["provenance"]["failure"]
        == "reporter_server_hash_provenance_unknown"
    )
    assert (
        audit["reporter_server_hash_parity"]["provenance"]["direct_matches"]
        == {
            "source": False,
            "local_installed": False,
            "public_v1549_tahoe": False,
        }
    )
    assert (
        audit["reporter_server_hash_parity"]["provenance"]["git_history"]["match"]
        is False
    )
    assert (
        audit["reporter_server_hash_parity"]["provenance"][
            "public_release_checked_count"
        ]
        >= 19
    )
    checked_public_dmg_assets = {
        (row.get("release_tag"), row.get("asset"))
        for row in audit["reporter_server_hash_parity"]["provenance"][
            "public_release_checked"
        ]
    }
    assert {
        ("v1.5.43", "vMLX-1.5.43-tahoe-arm64.dmg"),
        ("v1.5.44", "vMLX-1.5.44-tahoe-arm64.dmg"),
        ("v1.5.46", "vMLX-1.5.46-tahoe-arm64.dmg"),
        ("v1.5.47", "vMLX-1.5.47-tahoe-arm64.dmg"),
        ("v1.5.48", "vMLX-1.5.48-tahoe-arm64.dmg"),
        ("v1.5.49", "vMLX-1.5.49-tahoe-arm64.dmg"),
    }.issubset(checked_public_dmg_assets)
    assert audit["proven"]["local_installed_bundle_has_responses_cancel_route"] is True
    assert audit["public_release_dmg_contract"]["server_has_responses_cancel_route"] is True
    assert audit["proven"]["public_v1549_tahoe_dmg_has_responses_cancel_route"] is True
    assert audit["proven"]["local_installed_responses_cancel_live_probe"] is True
    assert (
        audit["proven"]["reporter_cancel_404_after_stream_abort_order_proven"]
        is True
    )
    assert (
        audit["proven"][
            "current_source_responses_cancel_inactive_404_contract_proven"
        ]
        is True
    )
    assert audit["local_responses_cancel_probe"]["status"] == "pass"
    assert audit["local_responses_cancel_probe"]["probe"]["abort_boundary"] == (
        "controlled_cancel_after_response_id"
    )
    assert audit["local_responses_cancel_probe"]["probe"]["bad_text_captured"] is False
    assert audit["local_reporter_prompt_reproduction"]["clean"] is True
    assert audit["local_reporter_prompt_reproduction"]["request_matches_reporter"] is True
    assert audit["local_reporter_prompt_reproduction"]["bad_text_captured"] is False
    assert audit["proven"]["local_reporter_prompt_reproduction_clean"] is True
    assert audit["local_model_manifest"]["local_full_k_artifact_shape_recorded"] is True
    assert audit["local_model_manifest"]["model_shard_count"] == 67
    assert audit["local_model_manifest"]["checks"]["has_jangtq_runtime"] is True
    assert audit["source_contract"]["server_has_responses_cancel_route"] is True
    assert audit["source_contract"]["panel_routes_resp_ids_to_responses_cancel"] is True
    assert audit["source_contract"]["test_proves_successful_responses_cancel_route"] is True
    assert (
        audit["source_contract"][
            "test_proves_inactive_responses_cancel_404_after_engine_lookup"
        ]
        is True
    )
    discriminators = {row["id"]: row for row in audit["root_cause_discriminators"]}
    assert discriminators["reporter_bundle_cancel_route_parity"]["status"] == "open"
    assert "public_v1549_tahoe_dmg_cancel_route_present" in discriminators[
        "reporter_bundle_cancel_route_parity"
    ]["observed"]
    assert "local_installed_hash_differs_from_public_v1549_tahoe" in discriminators[
        "reporter_bundle_cancel_route_parity"
    ]["observed"]
    assert discriminators["reporter_stream_abort_vs_model_text"]["status"] == "open"
    assert "reporter_cancel_404_after_stream_abort" in discriminators[
        "reporter_stream_abort_vs_model_text"
    ]["observed"]
    assert discriminators["local_runtime_cache_math_parity"]["status"] == "partially_proven"
    assert discriminators["local_runtime_cache_math_parity"]["observed"] == [
        "single_active_scheduler",
        "paged_prefix_cache_hits",
        "block_disk_l2_writes_and_hits",
        "turboquant_kv_live_decode",
    ]
    assert discriminators["prompt_template_parser_parity"]["status"] == "open"
    assert discriminators["model_artifact_hash_parity"]["status"] == "partially_proven"
    assert "local_full_k_artifact_manifest_recorded" in discriminators[
        "model_artifact_hash_parity"
    ]["observed"]
    assert "local_full_k_shard_hashes_recorded" in discriminators[
        "model_artifact_hash_parity"
    ]["observed"]
    assert "reporter model shard/codebook hashes match local full K artifact" not in audit["not_proven"]
    assert "reporter model artifact manifest is available for direct local comparison" not in audit["not_proven"]
    assert "reporter installed app bundle hash matches public/local server.py route proof" in audit["not_proven"]
    assert "reporter chat/session/settings database state matches local diagnostic state" not in audit["not_proven"]
    assert "the reporter 404 happened before the client stream abort" not in audit["not_proven"]
    assert (
        "the 404 cancel response caused the screenshot rather than followed the stream abort"
        not in audit["not_proven"]
    )
    assert audit["not_proven"] == [
        "reporter installed app bundle hash matches public/local server.py route proof"
    ]
    assert "reporter screenshot garbage is an interrupted reasoning-panel stream, not proven final visible answer text" in audit["release_boundary"]


def test_issue179_audit_writes_json_artifact(tmp_path):
    out = tmp_path / "issue179-audit.json"

    audit = gate.write_audit(Path("."), out)

    assert out.exists()
    assert '"status": "open"' in out.read_text(encoding="utf-8")
    assert audit["issue"]["id"] == 179


def test_issue179_public_dmg_contract_extracts_server_route_and_hash(tmp_path):
    app_server = (
        tmp_path
        / "mount/vMLX.app/Contents/Resources/bundled-python/python/lib/"
        "python3.12/site-packages/vmlx_engine/server.py"
    )
    app_server.parent.mkdir(parents=True)
    app_server.write_text(
        '@app.post("/v1/responses/{response_id}/cancel")\n'
        "async def cancel_response():\n"
        '    await _engine.abort_request(response_id)\n',
        encoding="utf-8",
    )
    out = tmp_path / "public-v1.5.48-sequoia-dmg-contract.json"

    contract = dmg_gate.build_contract_from_mount(
        mountpoint=tmp_path / "mount",
        dmg=tmp_path / "vMLX-1.5.48-sequoia-arm64.dmg",
        release_tag="v1.5.48",
        asset="vMLX-1.5.48-sequoia-arm64.dmg",
        asset_size_bytes=123,
        out=out,
    )

    assert contract["release_tag"] == "v1.5.48"
    assert contract["asset"] == "vMLX-1.5.48-sequoia-arm64.dmg"
    assert contract["asset_size_bytes"] == 123
    assert contract["server_has_responses_cancel_route"] is True
    assert contract["server_cancel_calls_engine_abort"] is True
    assert contract["server_sha256"] == dmg_gate.sha256_file(app_server)


def test_issue179_reporter_parity_comparison_marks_matching_reporter_metadata_pass():
    comparison = gate.build_reporter_parity_comparison(
        reporter_parity_artifact={
            "status": "pass",
            "capture_provenance": "reporter_machine",
            "installed_server_sha256": "server-sha",
            "server_has_responses_cancel_route": True,
            "server_cancel_calls_engine_abort": True,
            "model_manifest_sha256": "manifest-sha",
            "model_file_hashes": [
                {"path": "config.json", "sha256": "cfg"},
                {"path": "model-00001-of-00067.safetensors", "sha256": "shard"},
            ],
            "chat_id": "33a744d8",
            "session_settings": {"wireApi": "responses", "enableThinking": None},
            "response_id": "resp_66d7e36b833e",
            "response_active_at_cancel": False,
            "raw_sse_cancel_lifecycle": {"cancel_status": 404},
        },
        reporter={
            "request_error": {"chatId": "33a744d8"},
            "request_error_response_id": "resp_66d7e36b833e",
        },
        installed_bundle={"sha256": "server-sha"},
        local_model_manifest={
            "sha256": "manifest-sha",
            "model_file_hashes": [
                {"path": "model-00001-of-00067.safetensors", "sha256": "shard"},
                {"path": "config.json", "sha256": "cfg"},
            ],
        },
    )

    assert comparison == {
        "status": "pass",
        "capture_provenance_is_reporter_machine": True,
        "server_hash_matches_local_installed": True,
        "server_route_markers_match": True,
        "model_manifest_sha256_matches_local": True,
        "model_file_hashes_match_local": True,
        "chat_id_matches_reporter_log": True,
        "response_id_matches_reporter_log": True,
        "response_active_at_cancel_recorded": True,
        "raw_sse_cancel_lifecycle_present": True,
        "failures": [],
    }


def test_issue179_reporter_server_hash_parity_names_drift():
    parity = gate.build_reporter_server_hash_parity(
        reporter_parity_artifact={
            "installed_server_sha256": "reporter-sha",
            "server_has_responses_cancel_route": True,
            "server_cancel_calls_engine_abort": True,
        },
        source={"source_hashes": {"vmlx_engine/server.py": "source-sha"}},
        installed_bundle={"sha256": "source-sha"},
        public_dmg={"server_sha256": "public-sha"},
    )

    assert parity == {
        "reporter_installed_server_sha256": "reporter-sha",
        "source_server_sha256": "source-sha",
        "local_installed_server_sha256": "source-sha",
        "public_v1549_tahoe_server_sha256": "public-sha",
        "reporter_matches_source": False,
        "reporter_matches_local_installed": False,
        "reporter_matches_public_v1549_tahoe": False,
        "route_markers_match": True,
        "status": "open",
        "failure": "reporter_installed_server_hash_drift",
    }


def test_issue179_reporter_server_hash_provenance_names_unknown_hash(tmp_path):
    backup = (
        tmp_path
        / "build/installed-app-backups/old.app/Contents/Resources/bundled-python/"
        "python/lib/python3.12/site-packages/vmlx_engine/server.py"
    )
    backup.parent.mkdir(parents=True)
    backup.write_text("different server\n", encoding="utf-8")

    provenance = gate.build_reporter_server_hash_provenance(
        root=tmp_path,
        reporter_sha="reporter-sha",
        source_sha="source-sha",
        local_sha="local-sha",
        public_sha="public-sha",
        check_git_history=False,
    )

    assert provenance == {
        "status": "open",
        "checked_sources": [
            "source_contract",
            "local_installed_bundle",
            "public_v1549_tahoe_dmg",
            "public_release_dmg_contracts",
            "local_installed_app_backups",
            "git_history",
        ],
        "direct_matches": {
            "source": False,
            "local_installed": False,
            "public_v1549_tahoe": False,
        },
        "public_release_matches": [],
        "public_release_checked": [],
        "public_release_checked_count": 0,
        "local_backup_matches": [],
        "local_backup_checked_count": 1,
        "git_history": {
            "checked": False,
            "match": False,
            "commit": None,
            "error": "skipped",
        },
        "failure": "reporter_server_hash_provenance_unknown",
    }


def test_issue179_reporter_server_hash_provenance_checks_all_public_dmg_contracts(tmp_path):
    provenance = gate.build_reporter_server_hash_provenance(
        root=tmp_path,
        reporter_sha="reporter-sha",
        source_sha="source-sha",
        local_sha="local-sha",
        public_sha="tahoe-sha",
        public_dmg_contracts=[
            {
                "asset": "vMLX-1.5.49-tahoe-arm64.dmg",
                "server_sha256": "tahoe-sha",
            },
            {
                "asset": "vMLX-1.5.49-sequoia-arm64.dmg",
                "server_sha256": "reporter-sha",
            },
        ],
        check_git_history=False,
    )

    assert provenance["status"] == "pass"
    assert provenance["direct_matches"] == {
        "source": False,
        "local_installed": False,
        "public_v1549_tahoe": False,
    }
    assert provenance["public_release_checked_count"] == 2
    assert provenance["public_release_matches"] == [
        {
            "asset": "vMLX-1.5.49-sequoia-arm64.dmg",
            "path": None,
            "release_tag": None,
            "server_sha256": "reporter-sha",
        }
    ]
    assert "failure" not in provenance


def test_issue179_reporter_server_hash_provenance_finds_backup_match(tmp_path):
    content = b"matched server\n"
    reporter_sha = gate.hashlib.sha256(content).hexdigest()
    backup = (
        tmp_path
        / "build/installed-app-backups/matched.app/Contents/Resources/"
        "bundled-python/python/lib/python3.12/site-packages/vmlx_engine/server.py"
    )
    backup.parent.mkdir(parents=True)
    backup.write_bytes(content)

    provenance = gate.build_reporter_server_hash_provenance(
        root=tmp_path,
        reporter_sha=reporter_sha,
        source_sha="source-sha",
        local_sha="local-sha",
        public_sha="public-sha",
        check_git_history=False,
    )

    assert provenance["status"] == "pass"
    assert provenance["local_backup_checked_count"] == 1
    assert provenance["local_backup_matches"] == [
        {
            "path": (
                "build/installed-app-backups/matched.app/Contents/Resources/"
                "bundled-python/python/lib/python3.12/site-packages/"
                "vmlx_engine/server.py"
            ),
            "sha256": reporter_sha,
            "matches_reporter": True,
        }
    ]
    assert "failure" not in provenance


def test_issue179_not_proven_removes_reporter_parity_gaps_when_comparison_passes():
    not_proven = gate.build_not_proven_items(
        reporter_parity_comparison={
            "status": "pass",
            "server_hash_matches_local_installed": True,
            "model_manifest_sha256_matches_local": True,
            "model_file_hashes_match_local": True,
            "response_active_at_cancel_recorded": True,
            "raw_sse_cancel_lifecycle_present": True,
        },
        reporter_parity_artifact={
            "status": "pass",
            "session_settings": {"wireApi": "responses"},
        },
        reporter={
            "responses_cancel_404_after_econnreset_same_response_id": True,
            "responses_cancel_404_after_request_error": True,
        },
        local_reporter_prompt_reproduction={"clean": False},
    )

    assert "reporter model shard/codebook hashes match local full K artifact" not in not_proven
    assert "reporter model artifact manifest is available for direct local comparison" not in not_proven
    assert "reporter installed app bundle hash matches public/local server.py route proof" not in not_proven
    assert "reporter response id was still active when the cancel request was sent" not in not_proven
    assert "reporter chat/session/settings database state matches local diagnostic state" not in not_proven
    assert (
        "a concrete prompt reproduces screenshot-shaped wrong-language or numeric garbage"
        in not_proven
    )
    assert (
        "the 404 cancel response caused the screenshot rather than followed the stream abort"
        not in not_proven
    )


def test_issue179_not_proven_keeps_only_server_hash_when_other_parity_checks_pass():
    not_proven = gate.build_not_proven_items(
        reporter_parity_comparison={
            "status": "open",
            "server_hash_matches_local_installed": False,
            "model_manifest_sha256_matches_local": True,
            "model_file_hashes_match_local": True,
            "response_active_at_cancel_recorded": True,
            "raw_sse_cancel_lifecycle_present": True,
        },
        reporter_parity_artifact={
            "status": "pass",
            "session_settings": {"wireApi": "responses"},
        },
        reporter={
            "responses_cancel_404_after_econnreset_same_response_id": True,
            "responses_cancel_404_after_request_error": True,
        },
        local_reporter_prompt_reproduction={"clean": True},
    )

    assert not_proven == [
        "reporter installed app bundle hash matches public/local server.py route proof"
    ]


def test_issue179_reporter_parity_artifact_keeps_local_template_open(tmp_path):
    artifact = tmp_path / "build/issue-179/reporter-parity-metadata-20260527.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "status": "open",
                "capture_provenance": "local_template",
                "installed_server_sha256": "server-sha",
                "server_has_responses_cancel_route": True,
                "server_cancel_calls_engine_abort": True,
                "model_manifest_sha256": "manifest-sha",
                "model_file_hashes": [{"path": "config.json", "sha256": "cfg"}],
                "chat_id": "33a744d8",
                "session_settings": {"wireApi": "responses"},
                "response_id": "resp_66d7e36b833e",
                "response_active_at_cancel": False,
                "raw_sse_cancel_lifecycle": {"cancel_status": 404},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = gate.analyze_reporter_parity_artifact(tmp_path)

    assert result["exists"] is True
    assert result["status"] == "open"
    assert result["comparison_status"] == "collector_status_open"
    assert result["missing_fields"] == []


def test_issue179_reporter_parity_artifact_rejects_collector_missing_fields(tmp_path):
    artifact = tmp_path / "build/issue-179/reporter-parity-metadata-20260527.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "status": "pass",
                "capture_provenance": "reporter_machine",
                "installed_server_sha256": "server-sha",
                "server_has_responses_cancel_route": True,
                "server_cancel_calls_engine_abort": True,
                "model_manifest_sha256": "manifest-sha",
                "model_file_hashes": [{"path": "config.json", "sha256": "cfg"}],
                "chat_id": "33a744d8",
                "session_settings": {"wireApi": "chat"},
                "response_id": "resp_66d7e36b833e",
                "response_active_at_cancel": False,
                "raw_sse_cancel_lifecycle": {"request_error": {"code": "ETIMEOUT"}},
                "missing_fields": [
                    "session_settings_shape",
                    "raw_sse_cancel_lifecycle_shape",
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = gate.analyze_reporter_parity_artifact(tmp_path)

    assert result["status"] == "open"
    assert result["collector_missing_fields"] == [
        "session_settings_shape",
        "raw_sse_cancel_lifecycle_shape",
    ]
    assert result["comparison_status"] == "collector_missing_fields"
