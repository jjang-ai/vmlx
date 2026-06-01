# SPDX-License-Identifier: Apache-2.0
"""Contracts for installed-app runtime parity audit."""

from pathlib import Path


def test_installed_app_runtime_parity_records_known_stale_surface():
    from tests.cross_matrix import run_installed_app_runtime_parity_audit as gate

    audit = gate.build_audit(Path("."))

    assert audit["status"] in {"pass", "open"}
    assert "responses_cancel_route" in audit["checks"]
    assert "image_lora_cli_flags" in audit["checks"]
    assert "text_lora_flags_rejected" in audit["checks"]
    assert "xml_function_tool_parser_cli" in audit["checks"]
    assert "plain_attention_kv_native_cache_status" in audit["checks"]
    assert "installed_panel_gateway_epipe_guard" in audit["checks"]
    assert "installed_panel_gateway_epipe_aggregate_guard" in audit["checks"]
    assert "installed_panel_chat_ipc_epipe_aggregate_guard" in audit["checks"]
    assert "installed_panel_image_ipc_epipe_aggregate_guard" in audit["checks"]
    assert "installed_panel_cache_ipc_epipe_aggregate_guard" in audit["checks"]
    assert "installed_panel_child_process_stdio_epipe_guard" in audit["checks"]
    assert "installed_panel_child_process_stdio_epipe_aggregate_guard" in audit["checks"]
    assert "installed_panel_renderer_chat_epipe_toast_normalized" in audit["checks"]
    assert "installed_panel_gateway_guarded_proxy_forwarding" in audit["checks"]
    assert "installed_panel_gateway_write_once_behavior_marker" in audit["checks"]
    assert "installed_panel_gateway_response_socket_destroyed_guard" in audit["checks"]
    assert "installed_panel_gateway_request_socket_destroyed_guard" in audit["checks"]
    assert "installed_panel_gateway_single_model_cache_endpoint_routing" in audit["checks"]
    assert "installed_panel_gateway_single_model_streaming_routes" in audit["checks"]
    assert "installed_panel_max_output_context_settings_wired" in audit["checks"]
    assert "installed_panel_model_owned_generation_defaults_wired" in audit["checks"]
    assert "installed_panel_request_builders_keep_output_context_separate" in audit["checks"]
    assert "installed_panel_ollama_streaming_json_writes_guarded" in audit["checks"]
    assert "installed_panel_ollama_proxy_response_errors_guarded" in audit["checks"]
    assert "installed_vmlx_user_data_no_raw_disconnect_errors" in audit["checks"]
    assert "installed_vmlx_diagnostic_reports_no_raw_disconnect_errors" in audit["checks"]
    assert "vmlx_user_data_disconnect_errors" in audit
    assert "vmlx_diagnostic_disconnect_errors" in audit
    if audit["status"] == "open":
        assert audit["missing_or_stale"]


def test_installed_app_runtime_parity_requires_versioned_python_entrypoint():
    from tests.cross_matrix import run_installed_app_runtime_parity_audit as gate

    audit = gate.build_audit(Path("."))

    assert "installed_versioned_python_exists" in audit["checks"]
    assert "installed_versioned_python_runs" in audit["checks"]
    assert "installed_bundled_python_launch_crash_reports_classified" in audit["checks"]
    assert (
        "installed_bundled_python_launch_crashes_not_reproduced" in audit["checks"]
    )
    assert "vmlx_bundled_python_launch_crash_reports" in audit
    assert audit["installed_versioned_python"].endswith(
        "Contents/Resources/bundled-python/python/bin/python3.12"
    )


def test_installed_app_runtime_parity_default_out_tracks_manifest():
    from tests.cross_matrix import release_regression_manifest as manifest
    from tests.cross_matrix import run_installed_app_runtime_parity_audit as gate

    assert str(gate.DEFAULT_OUT) == (
        manifest.CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    )


def test_installed_app_runtime_parity_scopes_child_stdio_aggregate_guard():
    from tests.cross_matrix import run_installed_app_runtime_parity_audit as gate

    old_child_guard_with_unrelated_nested_guard = """
function isExpectedChildProcessStreamDisconnectError$4(err) {
  const code = err?.code;
  const message = String(err?.message || "").toLowerCase();
  return code === "EPIPE" || message.includes("write EPIPE");
}
function isExpectedChatBackendDisconnectError(err) {
  const nestedErrors = Array.isArray(err?.errors) ? err.errors : [];
  return nestedErrors.some((nested) => isExpectedChatBackendDisconnectError(nested));
}
"""
    compiled_child_guard = """
function isExpectedChildProcessStreamDisconnectError$4(err) {
  const code = err?.code;
  const message = String(err?.message || "").toLowerCase();
  const cause = err?.cause;
  const nestedErrors = Array.isArray(err?.errors) ? err.errors : [];
  return code === "EPIPE" || message.includes("write EPIPE") ||
    (cause ? isExpectedChildProcessStreamDisconnectError$4(cause) : false) ||
    nestedErrors.some((nested) => isExpectedChildProcessStreamDisconnectError$4(nested));
}
"""
    source_child_guard = """
function isExpectedChildProcessStreamDisconnectError(err: unknown): boolean {
  const code = (err as NodeJS.ErrnoException)?.code
  const message = String((err as Error)?.message || '').toLowerCase()
  const cause = (err as any)?.cause
  const nestedErrors = Array.isArray((err as any)?.errors) ? (err as any).errors : []
  return nestedErrors.some((nested) => isExpectedChildProcessStreamDisconnectError(nested))
}
"""

    assert not gate._has_child_stdio_aggregate_disconnect_guard(
        old_child_guard_with_unrelated_nested_guard
    )
    assert gate._has_child_stdio_aggregate_disconnect_guard(compiled_child_guard)
    assert gate._has_child_stdio_aggregate_disconnect_guard(source_child_guard)


def test_installed_app_runtime_parity_writes_json_artifact(tmp_path):
    from tests.cross_matrix import run_installed_app_runtime_parity_audit as gate

    out = tmp_path / "installed-parity.json"
    audit = gate.write_audit(Path("."), out)

    assert out.exists()
    assert audit["artifact"] == str(out)
    assert '"checks"' in out.read_text(encoding="utf-8")


def test_installed_app_runtime_parity_can_target_staged_app_path(tmp_path):
    from tests.cross_matrix import run_installed_app_runtime_parity_audit as gate

    staged_app = tmp_path / "Staged.app"
    staged_python = staged_app / "Contents/Resources/bundled-python/python/bin/python3"
    staged_asar = staged_app / "Contents/Resources/app.asar"

    audit = gate.build_audit(
        Path("."),
        app_path=staged_app,
        python_path=staged_python,
        app_asar=staged_asar,
        user_data=tmp_path / "user-data",
        diagnostic_reports=tmp_path / "reports",
    )

    assert audit["installed_app"] == str(staged_app)
    assert audit["installed_python"] == str(staged_python)
    assert audit["installed_versioned_python"] == str(
        staged_python.with_name("python3.12")
    )
    assert audit["installed_app_asar"] == str(staged_asar)
    assert audit["serve_help_returncode"] == 127
    assert audit["panel_main_stderr"] == f"missing installed app.asar: {staged_asar}"


def test_installed_app_runtime_parity_write_audit_accepts_staged_paths(tmp_path):
    from tests.cross_matrix import run_installed_app_runtime_parity_audit as gate

    staged_app = tmp_path / "Staged.app"
    staged_python = staged_app / "Contents/Resources/bundled-python/python/bin/python3"
    staged_asar = staged_app / "Contents/Resources/app.asar"
    out = tmp_path / "staged-parity.json"

    audit = gate.write_audit(
        Path("."),
        out,
        app_path=staged_app,
        python_path=staged_python,
        app_asar=staged_asar,
        user_data=tmp_path / "user-data",
        diagnostic_reports=tmp_path / "reports",
    )

    assert audit["artifact"] == str(out)
    assert audit["installed_app"] == str(staged_app)
    assert audit["installed_python"] == str(staged_python)
    assert audit["installed_app_asar"] == str(staged_asar)
    assert out.exists()


def test_installed_app_runtime_parity_scans_only_vmlx_user_data_disconnect_logs(tmp_path):
    from tests.cross_matrix import run_installed_app_runtime_parity_audit as gate

    user_data = tmp_path / "vMLX"
    session_storage = user_data / "Session Storage"
    session_storage.mkdir(parents=True)
    (session_storage / "LOG").write_text("ok\nwrite EPIPE after disconnect\n")
    (user_data / "ignored.txt").write_text("EPIPE outside allowed log names\n")

    scan = gate._scan_vmlx_user_data_disconnect_errors(user_data)

    assert scan["exists"] is True
    assert scan["scanned"] == ["Session Storage/LOG"]
    assert scan["hits"] == [
        {
            "file": "Session Storage/LOG",
            "line": 2,
            "text": "write EPIPE after disconnect",
        }
    ]


def test_installed_app_runtime_parity_scans_vmlx_diagnostic_disconnect_reports(
    tmp_path,
):
    from tests.cross_matrix import run_installed_app_runtime_parity_audit as gate

    reports = tmp_path / "DiagnosticReports"
    retired = reports / "Retired"
    retired.mkdir(parents=True)
    (retired / "vMLX-2026-05-28-120000.ips").write_text(
        "ok\nError: write EPIPE\n",
        encoding="utf-8",
    )
    (retired / "Slack-2026-05-28-120000.ips").write_text(
        "Error: write EPIPE\n",
        encoding="utf-8",
    )

    scan = gate._scan_vmlx_diagnostic_disconnect_errors(reports)

    assert scan["exists"] is True
    assert scan["scanned"] == ["Retired/vMLX-2026-05-28-120000.ips"]
    assert scan["hits"] == [
        {
            "file": "Retired/vMLX-2026-05-28-120000.ips",
            "line": 2,
            "text": "Error: write EPIPE",
        }
    ]


def test_installed_app_runtime_parity_classifies_bundled_python_launch_crashes(
    tmp_path,
):
    from tests.cross_matrix import run_installed_app_runtime_parity_audit as gate

    reports = tmp_path / "DiagnosticReports"
    reports.mkdir()
    (reports / "python3.12-2026-05-31-194414.ips").write_text(
        '{"app_name":"python3.12","timestamp":"2026-05-31 19:44:14.00 -0700"}\n'
        '  "procPath" : "\\/Applications\\/vMLX.app\\/Contents\\/Resources\\/'
        'bundled-python\\/python\\/bin\\/python3.12",\n'
        '  "termination" : {"namespace":"DYLD","indicator":"Library missing",'
        '"reasons":["Library not loaded: @rpath\\/libpython3.12.dylib"]}\n',
        encoding="utf-8",
    )
    (reports / "python3.12-2026-05-31-200000.ips").write_text(
        "unrelated python crash\n",
        encoding="utf-8",
    )

    scan = gate._scan_vmlx_bundled_python_launch_crashes(reports)

    assert scan["exists"] is True
    assert scan["scanned"] == ["python3.12-2026-05-31-194414.ips"]
    assert scan["hits"] == [
        {
            "file": "python3.12-2026-05-31-194414.ips",
            "reason": "Library not loaded: @rpath/libpython3.12.dylib",
        }
    ]
