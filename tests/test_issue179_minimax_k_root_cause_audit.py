# SPDX-License-Identifier: Apache-2.0
"""Contracts for the MiniMax-K issue #179 no-heavy root-cause audit."""

import json
from pathlib import Path

from tests.cross_matrix import run_issue179_minimax_k_root_cause_audit as gate
from tests.cross_matrix import run_issue179_public_dmg_contract as dmg_gate


def test_issue179_audit_keeps_reporter_cancel_404_boundary_open():
    audit = gate.build_audit(Path("."))

    assert audit["status"] == "open"
    assert audit["proven"]["latest_public_dmg_has_responses_cancel_route"] is True
    assert audit["proven"]["local_installed_bundle_has_responses_cancel_route"] is True
    assert audit["local_installed_bundle_contract"]["server_has_responses_cancel_route"] is True
    assert audit["latest_public_release_dmg_contract"]["release_tag"] == "v1.5.56"
    assert audit["latest_public_release_dmg_contract"]["server_has_responses_cancel_route"] is True
    assert audit["latest_public_release_dmg_contract"]["server_cancel_calls_engine_abort"] is True
    assert (
        audit["bundle_hash_parity"]["local_installed_server_sha256"]
        != audit["bundle_hash_parity"]["latest_public_server_sha256"]
    )
    assert audit["bundle_hash_parity"]["local_installed_matches_latest_public"] is False

    assert audit["reporter_parity_artifact"] == {
        "path": "build/issue-179/reporter-parity-metadata-20260527.json",
        "exists": False,
        "status": "open",
        "required_fields": list(gate.REPORTER_PARITY_REQUIRED_FIELDS),
        "capture_provenance": gate.REPORTER_SERVER_HASH_FALLBACK_PROVENANCE,
        "installed_server_sha256": gate.REPORTER_SERVER_HASH_FROM_PUBLIC_ISSUE_COMMENT,
        "server_has_responses_cancel_route": True,
        "server_cancel_calls_engine_abort": True,
        "comparison_status": "missing_reporter_parity_artifact",
        "note": (
            "Full reporter parity artifact is missing, but the reporter "
            "installed server.py hash was published in GitHub issue #179 "
            "comments by jjang-ai on 2026-06-02."
        ),
    }
    assert audit["reporter_parity_comparison"] == {
        "status": "open",
        "failures": ["missing_reporter_parity_artifact"],
    }
    assert audit["reporter_server_hash_parity"]["status"] == "open"
    assert (
        audit["reporter_server_hash_parity"]["provenance"]["failure"]
        == "missing_required_public_release_dmg_contracts"
    )
    assert (
        audit["reporter_server_hash_parity"]["reporter_installed_server_sha256"]
        == gate.REPORTER_SERVER_HASH_FROM_PUBLIC_ISSUE_COMMENT
    )
    assert audit["reporter_server_hash_parity"]["route_markers_match"] is True
    assert audit["not_proven"] == [
        "reporter model shard/codebook hashes match local full K artifact",
        "reporter model artifact manifest is available for direct local comparison",
        "reporter installed app bundle hash matches public/local server.py route proof",
        "reporter response id was still active when the cancel request was sent",
        "reporter chat/session/settings database state matches local diagnostic state",
        "a concrete prompt reproduces screenshot-shaped wrong-language or numeric garbage",
        "the 404 cancel response caused the screenshot rather than followed the stream abort",
    ]
    assert "#179 remains open" in audit["release_boundary"]

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


def test_issue179_audit_surfaces_latest_public_dmg_contract():
    audit = gate.build_audit(Path("."))

    latest = audit["latest_public_release_dmg_contract"]
    assert latest["release_tag"] == "v1.5.56"
    assert latest["asset"] in {
        "vMLX-1.5.56-sequoia-arm64.dmg",
        "vMLX-1.5.56-tahoe-arm64.dmg",
    }
    assert latest["server_has_responses_cancel_route"] is True
    assert latest["server_cancel_calls_engine_abort"] is True
    assert audit["proven"]["latest_public_dmg_has_responses_cancel_route"] is True
    assert (
        audit["bundle_hash_parity"]["latest_public_server_sha256"]
        == latest["server_sha256"]
    )
    assert (
        audit["reporter_server_hash_parity"]["reporter_matches_latest_public"]
        is False
    )
    checked_latest_assets = {
        (row.get("release_tag"), row.get("asset"))
        for row in audit["public_release_dmg_contracts"]
        if row.get("release_tag") == "v1.5.56"
    }
    assert checked_latest_assets == {
        ("v1.5.56", "vMLX-1.5.56-sequoia-arm64.dmg"),
        ("v1.5.56", "vMLX-1.5.56-tahoe-arm64.dmg"),
    }


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
        "latest_public_server_sha256": None,
        "reporter_matches_source": False,
        "reporter_matches_local_installed": False,
        "reporter_matches_public_v1549_tahoe": False,
        "reporter_matches_latest_public": False,
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
        app_bundle_search_roots=[tmp_path],
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
            "sibling_python_worktrees",
            "local_app_bundle_servers",
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
        "missing_required_public_release_contracts": [],
        "local_backup_matches": [],
        "local_backup_checked_count": 1,
        "sibling_source_matches": [],
        "sibling_source_checked": [],
        "sibling_source_checked_count": 0,
        "local_app_bundle_matches": [],
        "local_app_bundle_checked": [],
        "local_app_bundle_checked_count": 0,
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


def test_issue179_reporter_server_hash_provenance_reports_missing_required_public_dmg_contracts(
    tmp_path,
):
    provenance = gate.build_reporter_server_hash_provenance(
        root=tmp_path,
        reporter_sha="reporter-sha",
        source_sha="source-sha",
        local_sha="local-sha",
        public_sha="public-sha",
        public_dmg_contracts=[
            {
                "asset": "vMLX-1.5.49-tahoe-arm64.dmg",
                "release_tag": "v1.5.49",
                "server_sha256": "public-sha",
            },
            {
                "asset": "vMLX-1.5.52-tahoe-arm64.dmg",
                "release_tag": "v1.5.52",
                "server_sha256": "source-sha",
            },
        ],
        required_public_release_contracts={
            ("v1.5.50", "vMLX-1.5.50-sequoia-arm64.dmg"),
            ("v1.5.50", "vMLX-1.5.50-tahoe-arm64.dmg"),
        },
        check_git_history=False,
    )

    assert provenance["status"] == "open"
    assert provenance["failure"] == "missing_required_public_release_dmg_contracts"
    assert provenance["missing_required_public_release_contracts"] == [
        {
            "asset": "vMLX-1.5.50-sequoia-arm64.dmg",
            "release_tag": "v1.5.50",
        },
        {
            "asset": "vMLX-1.5.50-tahoe-arm64.dmg",
            "release_tag": "v1.5.50",
        },
    ]


def test_issue179_reporter_server_hash_provenance_checks_sibling_python_worktrees(
    tmp_path,
):
    root = tmp_path / "vllm-mlx-finite-launch-guard"
    root.mkdir()
    sibling = tmp_path / "vllm-mlx-release-1.5.33/vmlx_engine/server.py"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("known sibling server\n", encoding="utf-8")

    provenance = gate.build_reporter_server_hash_provenance(
        root=root,
        reporter_sha="reporter-sha",
        source_sha="source-sha",
        local_sha="local-sha",
        public_sha="public-sha",
        check_git_history=False,
    )

    assert "sibling_python_worktrees" in provenance["checked_sources"]
    assert provenance["sibling_source_checked_count"] == 1
    assert provenance["sibling_source_matches"] == []
    assert provenance["sibling_source_checked"] == [
        {
            "path": str(sibling),
            "sha256": gate.sha256_file(sibling),
            "matches_reporter": False,
        }
    ]


def test_issue179_reporter_server_hash_provenance_checks_local_app_bundles(
    tmp_path,
):
    content = b"matched staged app server\n"
    reporter_sha = gate.hashlib.sha256(content).hexdigest()
    app_server = (
        tmp_path
        / "worktrees/release-candidate/panel/release/mac-arm64/"
        "vMLX.app/Contents/Resources/bundled-python/python/lib/python3.12/"
        "site-packages/vmlx_engine/server.py"
    )
    app_server.parent.mkdir(parents=True)
    app_server.write_bytes(content)

    provenance = gate.build_reporter_server_hash_provenance(
        root=tmp_path / "vllm-mlx-finite-launch-guard",
        reporter_sha=reporter_sha,
        source_sha="source-sha",
        local_sha="local-sha",
        public_sha="public-sha",
        app_bundle_search_roots=[tmp_path / "worktrees"],
        check_git_history=False,
    )

    assert "local_app_bundle_servers" in provenance["checked_sources"]
    assert provenance["status"] == "pass"
    assert provenance["local_app_bundle_checked_count"] == 1
    assert provenance["local_app_bundle_matches"] == [
        {
            "path": str(app_server),
            "sha256": reporter_sha,
            "matches_reporter": True,
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
            "chat_id_matches_reporter_log": True,
            "response_id_matches_reporter_log": True,
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


def test_issue179_not_proven_releases_stale_reporter_hash_when_current_runtime_is_clean():
    not_proven = gate.build_not_proven_items(
        reporter_parity_comparison={
            "status": "open",
            "failures": ["server_hash_matches_local_installed"],
            "server_hash_matches_local_installed": False,
            "server_route_markers_match": True,
            "model_manifest_sha256_matches_local": True,
            "model_file_hashes_match_local": True,
            "chat_id_matches_reporter_log": True,
            "response_id_matches_reporter_log": True,
            "response_active_at_cancel_recorded": True,
            "raw_sse_cancel_lifecycle_present": True,
        },
        reporter_server_hash_parity={
            "status": "open",
            "failure": "reporter_installed_server_hash_drift",
            "route_markers_match": True,
            "reporter_matches_source": False,
            "reporter_matches_local_installed": False,
            "reporter_matches_latest_public": False,
            "provenance": {
                "status": "open",
                "failure": "reporter_server_hash_provenance_unknown",
                "public_release_matches": [],
            },
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
        proven={
            "current_source_responses_cancel_contract_proven": True,
            "current_source_responses_cancel_inactive_404_contract_proven": True,
            "latest_public_dmg_has_responses_cancel_route": True,
            "local_installed_bundle_has_responses_cancel_route": True,
            "local_installed_responses_cancel_live_probe": True,
            "local_reporter_prompt_reproduction_clean": True,
        },
    )

    assert not_proven == []


def test_issue179_not_proven_keeps_server_hash_when_reporter_hash_matches_public_release():
    not_proven = gate.build_not_proven_items(
        reporter_parity_comparison={
            "status": "open",
            "server_hash_matches_local_installed": False,
            "server_route_markers_match": True,
            "model_manifest_sha256_matches_local": True,
            "model_file_hashes_match_local": True,
            "response_active_at_cancel_recorded": True,
            "raw_sse_cancel_lifecycle_present": True,
        },
        reporter_server_hash_parity={
            "status": "open",
            "failure": "reporter_installed_server_hash_drift",
            "route_markers_match": True,
            "reporter_matches_source": False,
            "reporter_matches_local_installed": False,
            "reporter_matches_latest_public": False,
            "provenance": {
                "status": "pass",
                "failure": None,
                "public_release_matches": [{"release_tag": "v1.5.49"}],
            },
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
        proven={
            "current_source_responses_cancel_contract_proven": True,
            "current_source_responses_cancel_inactive_404_contract_proven": True,
            "latest_public_dmg_has_responses_cancel_route": True,
            "local_installed_bundle_has_responses_cancel_route": True,
            "local_installed_responses_cancel_live_probe": True,
            "local_reporter_prompt_reproduction_clean": True,
        },
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
