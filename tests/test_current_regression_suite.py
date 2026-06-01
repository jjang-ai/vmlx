import json
import os
from pathlib import Path
import hashlib
import sys
import time


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _write_known_open_objective_digest(tmp_path: Path) -> None:
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_json(
        tmp_path / suite.CURRENT_OBJECTIVE_DIGEST_ARTIFACT,
        {
            "requirements": [
                {"requirement": requirement, "status": "open"}
                for requirement in suite.EXPECTED_OPEN_REQUIREMENTS
            ]
        },
    )


def test_current_regression_suite_keeps_declared_known_blockers_open(tmp_path, monkeypatch):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_json(
        tmp_path / suite.CURRENT_OBJECTIVE_DIGEST_ARTIFACT,
        {
            "requirements": [
                {"requirement": "Ling/Bailing multilingual output quality is release-cleared", "status": "pass"},
                {"requirement": "DSV4 Flash prefix/paged/L2 cache is enabled by default from app launch", "status": "pass"},
                {"requirement": "DSV4 default-cache multi-tool agent loop is proven", "status": "pass"},
                {"requirement": "Gemma4 26B CRACK mixed-SWA app-engine speed floor is release-cleared", "status": "pass"},
                *(
                    {"requirement": requirement, "status": "open"}
                    for requirement in suite.EXPECTED_OPEN_REQUIREMENTS
                ),
            ]
        },
    )

    def fake_run_step(name, cmd, cwd):
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=True)

    assert artifact["status"] == "open"
    assert artifact["known_open_requirements"] == suite.EXPECTED_OPEN_REQUIREMENTS
    assert artifact["unexpected_open_requirements"] == []
    assert artifact["missing_expected_open_requirements"] == []
    assert artifact["steps"]["release_gate_skip_app"]["returncode"] == 0


def test_current_regression_suite_does_not_keep_cleared_unblocked_non_mimo_gap_open():
    from tests.cross_matrix import run_current_regression_suite as suite

    assert (
        "Real Electron UI unblocked non-MiMo live model matrix is proven"
        not in suite.EXPECTED_OPEN_REQUIREMENTS
    )


def test_current_regression_suite_preserves_expected_open_requirement_details(
    tmp_path,
    monkeypatch,
):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_json(
        tmp_path / suite.CURRENT_OBJECTIVE_DIGEST_ARTIFACT,
        {
            "requirements": [
                {
                    "requirement": "DSV4 default-cache multi-tool agent loop is proven",
                    "status": "pass",
                    "caveat": None,
                    "evidence": ["build/current-dsv4-default-cache-tool-loop/result.json"],
                    "details": {
                        "failed_required_tool_loop_checks": [],
                        "tool_loop_checks": {"code_file_written_exact": False},
                    },
                },
                {
                    "requirement": "Gemma4 26B CRACK mixed-SWA app-engine speed floor is release-cleared",
                    "status": "pass",
                    "details": {
                        "app_streaming_speed_floor_clears": True,
                    },
                },
                {
                    "requirement": "Cross-family live multi-turn smoke matrix is release-cleared",
                    "status": "pass",
                    "details": {
                        "missing_required_family_keys": [],
                    },
                },
                {
                    "requirement": "Real Electron UI cross-family live model matrix is release-cleared",
                    "status": "open",
                    "details": {
                        "release_boundary": "mock_ui_plus_server_smoke_is_not_real_ui_live_model_clearance",
                    },
                },
                {
                    "requirement": "DSV4 long-output/code/file-generation quality is release-cleared",
                    "status": "open",
                    "details": {
                        "direct_off_exactness_boundary": {
                            "hidden_force_on_would_be_false_clearance": True,
                        }
                    },
                },
            ]
        },
    )

    monkeypatch.setattr(
        suite,
        "_run_step",
        lambda name, cmd, cwd: {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []},
    )

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    details = artifact["open_requirement_details"]
    assert "DSV4 default-cache multi-tool agent loop is proven" not in details
    assert (
        "Gemma4 26B CRACK mixed-SWA app-engine speed floor is release-cleared"
        not in details
    )
    quality = details["DSV4 long-output/code/file-generation quality is release-cleared"]
    assert quality["details"]["direct_off_exactness_boundary"][
        "hidden_force_on_would_be_false_clearance"
    ] is True


def test_current_regression_suite_records_source_hashes_for_stale_proof_detection(
    tmp_path,
    monkeypatch,
):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)
    source_path = tmp_path / "tests/cross_matrix/summarize_objective_proof.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("current digest source\n", encoding="utf-8")
    monkeypatch.setattr(
        suite,
        "CURRENT_SUITE_SOURCE_HASH_FILES",
        ("tests/cross_matrix/summarize_objective_proof.py",),
    )
    monkeypatch.setattr(
        suite,
        "_run_step",
        lambda name, cmd, cwd: {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []},
    )

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["source_hashes"] == {
        "tests/cross_matrix/summarize_objective_proof.py": hashlib.sha256(
            b"current digest source\n"
        ).hexdigest()
    }


def test_current_regression_suite_hashes_dsv4_generation_boundary_sources():
    from tests.cross_matrix import run_current_regression_suite as suite

    required = {
        "vmlx_engine/scheduler.py",
        "vmlx_engine/utils/dsv4_batch_generator.py",
        "tests/cross_matrix/run_dsv4_route_mode_code_exactness.py",
        "tests/cross_matrix/run_dsv4_default_cache_tool_loop_gate.py",
        "tests/test_dsv4_route_mode_code_exactness.py",
        "tests/test_dsv4_default_cache_tool_loop_gate.py",
        "tests/test_objective_proof_digest.py",
    }

    assert required.issubset(set(suite.CURRENT_SUITE_SOURCE_HASH_FILES))
    assert all(Path(path).exists() for path in required)


def test_current_regression_suite_source_hash_list_matches_release_manifest():
    from tests.cross_matrix import release_regression_manifest as manifest
    from tests.cross_matrix import run_current_regression_suite as suite

    assert suite.CURRENT_SUITE_SOURCE_HASH_FILES == manifest.CURRENT_SUITE_SOURCE_HASH_FILES


def test_current_regression_suite_contract_outputs_track_runner_defaults():
    from tests.cross_matrix import run_api_surface_contract
    from tests.cross_matrix import run_cache_architecture_contract
    from tests.cross_matrix import run_generation_defaults_contract
    from tests.cross_matrix import run_installed_app_runtime_parity_audit
    from tests.cross_matrix import run_issue175_179_release_boundary_audit
    from tests.cross_matrix import run_issue181_183_runtime_audit
    from tests.cross_matrix import run_jang_model_compat_contract
    from tests.cross_matrix import run_max_output_context_contract
    from tests.cross_matrix import run_mcp_policy_contract
    from tests.cross_matrix import run_model_artifact_format_contract
    from tests.cross_matrix import run_model_family_detection_contract
    from tests.cross_matrix import run_native_mtp_contract
    from tests.cross_matrix import run_noheavy_api_cache_contract
    from tests.cross_matrix import run_noheavy_panel_settings_contract
    from tests.cross_matrix import run_packaged_integrity_contract
    from tests.cross_matrix import run_panel_tool_security_contract
    from tests.cross_matrix import run_parser_registry_contract
    from tests.cross_matrix import run_public_app_issue_audit
    from tests.cross_matrix import run_reasoning_template_contract
    from tests.cross_matrix import run_real_ui_dsv4_memory_preflight
    from tests.cross_matrix import run_release_surface_contract
    from tests.cross_matrix import run_tool_call_contract
    from tests.cross_matrix import run_vl_media_cache_contract
    from tests.cross_matrix import run_current_regression_suite as suite

    expected = {
        "noheavy_api_cache_contract": run_noheavy_api_cache_contract.DEFAULT_OUT,
        "cache_architecture_contracts": run_cache_architecture_contract.DEFAULT_OUT,
        "noheavy_panel_settings_contract": run_noheavy_panel_settings_contract.DEFAULT_OUT,
        "max_output_context_contracts": run_max_output_context_contract.DEFAULT_OUT,
        "parser_registry_contracts": run_parser_registry_contract.DEFAULT_OUT,
        "generation_defaults_contracts": run_generation_defaults_contract.DEFAULT_OUT,
        "reasoning_template_contracts": run_reasoning_template_contract.DEFAULT_OUT,
        "api_surface_contracts": run_api_surface_contract.DEFAULT_OUT,
        "panel_tool_security_contracts": run_panel_tool_security_contract.DEFAULT_OUT,
        "tool_call_contracts": run_tool_call_contract.DEFAULT_OUT,
        "mcp_policy_contracts": run_mcp_policy_contract.DEFAULT_OUT,
        "real_ui_dsv4_memory_preflight": run_real_ui_dsv4_memory_preflight.DEFAULT_OUT,
        "release_surface_contracts": run_release_surface_contract.DEFAULT_OUT,
        "jang_model_compat_contracts": run_jang_model_compat_contract.DEFAULT_OUT,
        "model_artifact_format_contracts": run_model_artifact_format_contract.DEFAULT_OUT,
        "model_family_detection_contracts": run_model_family_detection_contract.DEFAULT_OUT,
        "native_mtp_contracts": run_native_mtp_contract.DEFAULT_OUT,
        "vl_media_cache_contracts": run_vl_media_cache_contract.DEFAULT_OUT,
        "packaged_integrity_contracts": run_packaged_integrity_contract.DEFAULT_OUT,
        "installed_app_runtime_parity_audit": (
            run_installed_app_runtime_parity_audit.DEFAULT_OUT
        ),
        "issue175_179_release_boundary_audit": (
            run_issue175_179_release_boundary_audit.DEFAULT_OUT
        ),
        "issue181_183_runtime_audit": run_issue181_183_runtime_audit.DEFAULT_OUT,
        "public_app_issue_audit": run_public_app_issue_audit.DEFAULT_OUT,
    }

    for name, expected_out in expected.items():
        cmd = suite.CURRENT_SUITE_COMMANDS[name]
        assert "--out" in cmd
        assert Path(cmd[cmd.index("--out") + 1]) == expected_out


def test_current_regression_suite_executes_declared_command_table(monkeypatch, tmp_path):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)
    seen_steps = {}

    def fake_run_step(name, cmd, cwd):
        seen_steps[name] = cmd
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    for name, cmd in suite.CURRENT_SUITE_COMMANDS.items():
        assert seen_steps[name] == cmd


def test_current_regression_suite_checkpoints_each_started_step(
    monkeypatch,
    tmp_path,
):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)
    out = tmp_path / "current-suite.json"
    monkeypatch.setattr(
        suite,
        "CURRENT_SUITE_COMMANDS",
        {
            "first_contract": [sys.executable, "-c", "print('first')"],
            "second_contract": [sys.executable, "-c", "print('second')"],
        },
    )
    snapshots = {}

    def fake_run_step(name, cmd, cwd):
        if name == "second_contract":
            snapshots["before_second"] = json.loads(
                out.read_text(encoding="utf-8")
            )
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    suite.build_suite_artifact(
        tmp_path,
        include_release_gate=False,
        current_suite_artifact_path=out,
    )

    before_second = snapshots["before_second"]
    assert before_second["current_step"] == "second_contract"
    assert before_second["completed_steps"] == ["first_contract"]


def test_current_regression_suite_refreshes_release_boundary_artifacts():
    from tests.cross_matrix import run_current_regression_suite as suite
    from tests.cross_matrix import release_regression_manifest as manifest

    installed_cmd = suite.CURRENT_SUITE_COMMANDS["installed_app_runtime_parity_audit"]
    assert "--out" in installed_cmd
    assert (
        installed_cmd[installed_cmd.index("--out") + 1]
        == manifest.CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    )

    staged_cmd = suite.CURRENT_SUITE_COMMANDS["staged_app_runtime_parity_audit"]
    assert "--app" in staged_cmd
    assert staged_cmd[staged_cmd.index("--app") + 1] == "panel/release/mac-arm64/vMLX.app"
    assert "--user-data" not in staged_cmd
    assert "--diagnostic-reports" not in staged_cmd
    assert "--out" in staged_cmd
    assert (
        staged_cmd[staged_cmd.index("--out") + 1]
        == manifest.CURRENT_STAGED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    )

    issue_cmd = suite.CURRENT_SUITE_COMMANDS["issue175_179_release_boundary_audit"]
    assert "--out" in issue_cmd
    assert (
        issue_cmd[issue_cmd.index("--out") + 1]
        == manifest.CURRENT_ISSUE175_179_RELEASE_BOUNDARY_AUDIT_ARTIFACT
    )

    ordered = list(suite.CURRENT_SUITE_COMMANDS)
    assert ordered.index("installed_app_runtime_parity_audit") < ordered.index(
        "issue175_179_release_boundary_audit"
    )


def test_current_regression_suite_does_not_hang_on_pipe_inheriting_descendants(tmp_path):
    from tests.cross_matrix import run_current_regression_suite as suite

    script = tmp_path / "pipe_holder.py"
    script.write_text(
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)'])\n"
        "print('parent exited')\n",
        encoding="utf-8",
    )

    started = time.monotonic()
    step = suite._run_step(
        "pipe_holder",
        [sys.executable, str(script)],
        tmp_path,
        timeout_sec=0.2,
    )

    assert time.monotonic() - started < 2.0
    assert step["returncode"] == 0
    assert step["timed_out"] is False
    assert "parent exited" in "\n".join(step["stdout_tail"])


def test_current_regression_suite_hashes_model_matrix_contract_sources():
    from tests.cross_matrix import run_current_regression_suite as suite

    required = {
        "tests/cross_matrix/run_cache_architecture_contract.py",
        "tests/cross_matrix/run_generation_defaults_contract.py",
        "tests/cross_matrix/run_jang_model_compat_contract.py",
        "tests/cross_matrix/run_mcp_policy_contract.py",
        "tests/cross_matrix/run_model_family_detection_contract.py",
        "tests/cross_matrix/run_model_artifact_format_contract.py",
        "tests/cross_matrix/run_native_mtp_contract.py",
        "tests/cross_matrix/run_noheavy_api_cache_contract.py",
        "tests/cross_matrix/run_parser_registry_contract.py",
        "tests/cross_matrix/run_panel_tool_security_contract.py",
        "tests/cross_matrix/run_production_family_audit.py",
        "tests/cross_matrix/run_reasoning_template_contract.py",
        "tests/cross_matrix/run_release_regression_manifest.py",
        "tests/cross_matrix/run_release_surface_contract.py",
        "tests/cross_matrix/run_tool_call_contract.py",
        "tests/cross_matrix/run_vl_media_cache_contract.py",
        "tests/test_all_local_model_smoke.py",
    }

    assert required.issubset(set(suite.CURRENT_SUITE_SOURCE_HASH_FILES))


def test_current_regression_suite_hashes_release_blocker_boundary_sources():
    from tests.cross_matrix import run_current_regression_suite as suite

    required = {
        "tests/cross_matrix/run_installed_app_runtime_parity_audit.py",
        "tests/cross_matrix/run_issue175_177_installed_runtime_audit.py",
        "tests/cross_matrix/run_issue175_177_live_runtime_audit.py",
        "tests/cross_matrix/run_issue175_admin_sleep_probe.py",
        "tests/cross_matrix/run_issue175_179_release_boundary_audit.py",
        "tests/cross_matrix/run_issue181_183_runtime_audit.py",
        "tests/cross_matrix/run_public_app_issue_audit.py",
        "tests/cross_matrix/run_issue179_minimax_k_model_manifest.py",
        "tests/cross_matrix/run_issue179_reporter_parity_metadata.py",
        "tests/cross_matrix/run_issue179_minimax_k_root_cause_audit.py",
        "tests/cross_matrix/run_issue179_responses_cancel_probe.py",
        "tests/cross_matrix/run_real_ui_dsv4_memory_preflight.py",
        "tests/test_installed_app_runtime_parity_audit.py",
        "tests/test_issue175_177_installed_runtime_audit.py",
        "tests/test_issue175_177_live_runtime_audit.py",
        "tests/test_issue175_admin_sleep_probe.py",
        "tests/test_issue175_179_release_boundary_audit.py",
        "tests/test_issue181_183_runtime_audit.py",
        "tests/test_public_app_issue_audit.py",
        "tests/test_issue179_minimax_k_model_manifest.py",
        "tests/test_issue179_reporter_parity_metadata.py",
        "tests/test_issue179_minimax_k_root_cause_audit.py",
        "tests/test_issue179_responses_cancel_probe.py",
        "tests/test_real_ui_dsv4_memory_preflight.py",
    }

    assert required.issubset(set(suite.CURRENT_SUITE_SOURCE_HASH_FILES))
    assert all(Path(path).exists() for path in required)


def test_current_regression_suite_hashes_runtime_launch_and_mllm_cache_sources():
    from tests.cross_matrix import run_current_regression_suite as suite

    required = {
        "bench/native_mtp_speed_ab.py",
        "tests/cross_matrix/run_decode_speed_gate.py",
        "vmlx_engine/block_disk_store.py",
        "vmlx_engine/cli.py",
        "vmlx_engine/engine/batched.py",
        "vmlx_engine/engine/simple.py",
        "vmlx_engine/mllm_scheduler.py",
        "vmlx_engine/models/mllm.py",
        "vmlx_engine/paged_cache.py",
        "vmlx_engine/reranker.py",
    }

    assert required.issubset(set(suite.CURRENT_SUITE_SOURCE_HASH_FILES))
    assert all(Path(path).exists() for path in required)


def test_current_regression_suite_hashes_panel_api_settings_sources():
    from tests.cross_matrix import run_current_regression_suite as suite

    required = {
        "panel/src/main/api-gateway.ts",
        "panel/src/main/engine-manager.ts",
        "panel/scripts/live-chat-tools-reasoning-proof.mjs",
        "panel/src/main/index.ts",
        "panel/src/main/server.ts",
        "panel/src/main/sessions.ts",
        "panel/src/main/ipc/chat.ts",
        "panel/src/main/ipc/developer.ts",
        "panel/src/main/ipc/image.ts",
        "panel/src/main/ipc/imageGenerationState.ts",
        "panel/src/main/ipc/models.ts",
        "panel/src/main/model-config-registry.ts",
        "panel/src/main/process-manager.ts",
        "panel/src/main/tools/executor.ts",
        "panel/src/renderer/src/components/chat/MessageBubble.tsx",
        "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx",
        "panel/src/renderer/src/components/sessions/SessionSettings.tsx",
        "panel/src/shared/reasoningParserAliases.ts",
        "panel/tests/api-gateway-ollama-behavior.test.ts",
        "panel/tests/api-gateway-ollama.test.ts",
        "panel/tests/api-gateway-single-model.behavior.test.ts",
        "panel/tests/generation-defaults.test.ts",
        "panel/tests/image-system.test.ts",
        "panel/tests/interleaved-reasoning-render.test.ts",
        "panel/tests/model-config-registry.test.ts",
        "panel/tests/settings-flow.test.ts",
        "tests/cross_matrix/run_api_surface_contract.py",
        "tests/cross_matrix/run_max_output_context_contract.py",
        "tests/cross_matrix/run_noheavy_panel_settings_contract.py",
    }

    assert required.issubset(set(suite.CURRENT_SUITE_SOURCE_HASH_FILES))
    assert all(Path(path).exists() for path in required)


def test_current_regression_suite_hashes_focused_pytest_gate_sources():
    from tests.cross_matrix import run_current_regression_suite as suite

    required = {
        "tests/test_objective_proof_digest.py",
        "tests/test_dsv4_default_cache_tool_loop_gate.py",
        "tests/test_release_gate_python_app.py",
        "tests/test_current_regression_suite.py",
        "tests/test_release_regression_manifest.py",
        "tests/test_issue179_minimax_k_root_cause_audit.py",
        "tests/test_model_family_detection_contract.py",
        "tests/test_mcp_policy_contract.py",
        "tests/test_vl_media_cache_contract.py",
        "tests/test_batching.py",
        "tests/test_dsv4_batch_generator_speed.py",
        "tests/test_scheduler_repetition_context.py",
        "tests/test_dsv4_paged_cache.py",
    }

    assert required.issubset(set(suite.CURRENT_SUITE_SOURCE_HASH_FILES))
    assert all(Path(path).exists() for path in required)


def test_current_regression_suite_runs_issue179_root_cause_pytest():
    from tests.cross_matrix import run_current_regression_suite as suite

    command = suite.CURRENT_SUITE_COMMANDS["focused_regression_pytest"]
    joined = " ".join(command)

    assert "tests/test_issue179_minimax_k_root_cause_audit.py" in command
    assert "reporter_server_hash_parity" in joined


def test_current_regression_suite_runs_issue179_cancel_probe_memory_preflight_pytest():
    from tests.cross_matrix import run_current_regression_suite as suite

    command = suite.CURRENT_SUITE_COMMANDS["focused_regression_pytest"]
    joined = " ".join(command)

    assert "tests/test_issue179_responses_cancel_probe.py" in command
    assert "issue179_memory_preflight" in joined


def test_current_regression_suite_hashes_dirty_contract_unit_sources():
    from tests.cross_matrix import run_current_regression_suite as suite

    required = {
        "tests/test_api_surface_contract.py",
        "tests/test_cache_architecture_contract.py",
        "tests/test_dsml_tool_parser.py",
        "tests/test_engine_audit.py",
        "tests/test_generation_defaults_contract.py",
        "tests/test_image_api.py",
        "tests/test_image_gen.py",
        "tests/test_local_generation_metadata_audit.py",
        "tests/test_mllm_continuous_batching.py",
        "tests/test_mllm_scheduler_cache.py",
        "tests/test_model_config_registry.py",
        "tests/test_mlx_memory_cleanup.py",
        "tests/test_packaged_integrity_contract.py",
        "tests/test_paged_cache_unit.py",
        "tests/test_panel_cli_flag_contract.py",
        "tests/test_reasoning_modes.py",
        "tests/test_runtime_memory_stress_probe.py",
        "tests/test_server.py",
        "tests/test_tool_format.py",
        "tests/test_tool_parsers.py",
        "vmlx_engine/image_gen.py",
        "vmlx_engine/model_config_registry.py",
        "vmlx_engine/model_configs.py",
        "vmlx_engine/mlx_memory.py",
        "vmlx_engine/reasoning/__init__.py",
        "vmlx_engine/reasoning/think_xml_parser.py",
        "vmlx_engine/tool_parsers/__init__.py",
        "vmlx_engine/tool_parsers/xml_function_tool_parser.py",
        "vmlx_engine/tool_parsers/zaya_tool_parser.py",
    }

    assert required.issubset(set(suite.CURRENT_SUITE_SOURCE_HASH_FILES))
    assert all(Path(path).exists() for path in required)


def test_current_regression_suite_fails_on_new_unexpected_open_requirement(tmp_path, monkeypatch):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_json(
        tmp_path / suite.CURRENT_OBJECTIVE_DIGEST_ARTIFACT,
        {
            "requirements": [
                {"requirement": "Ling/Bailing multilingual output quality is release-cleared", "status": "pass"},
                {"requirement": "DSV4 default-cache multi-tool agent loop is proven", "status": "pass"},
                {"requirement": "Gemma4 26B CRACK mixed-SWA app-engine speed floor is release-cleared", "status": "pass"},
                {"requirement": "Real Electron UI cross-family live model matrix is release-cleared", "status": "open"},
                {"requirement": "DSV4 long-output/code/file-generation quality is release-cleared", "status": "open"},
                {"requirement": "New untracked blocker", "status": "open"},
            ]
        },
    )

    monkeypatch.setattr(
        suite,
        "_run_step",
        lambda name, cmd, cwd: {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []},
    )

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    assert artifact["unexpected_open_requirements"] == [
        "New untracked blocker"
    ]


def test_current_regression_suite_fails_on_step_failure_even_if_digest_is_expected(tmp_path, monkeypatch):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_json(
        tmp_path / suite.CURRENT_OBJECTIVE_DIGEST_ARTIFACT,
        {
            "requirements": [
                {"requirement": "Ling/Bailing multilingual output quality is release-cleared", "status": "pass"},
                {"requirement": "DSV4 default-cache multi-tool agent loop is proven", "status": "pass"},
                {"requirement": "Gemma4 26B CRACK mixed-SWA app-engine speed floor is release-cleared", "status": "pass"},
                {"requirement": "Cross-family live multi-turn smoke matrix is release-cleared", "status": "open"},
                {"requirement": "Real Electron UI cross-family live model matrix is release-cleared", "status": "open"},
                {"requirement": "DSV4 long-output/code/file-generation quality is release-cleared", "status": "open"},
            ]
        },
    )

    def fake_run_step(name, cmd, cwd):
        return {
            "name": name,
            "command": cmd,
            "returncode": 1 if name == "noheavy_api_cache_contract" else 0,
            "stdout_tail": ["failed"],
        }

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    assert artifact["failed_steps"] == ["noheavy_api_cache_contract"]


def test_current_regression_suite_writes_provisional_artifact_before_packaged_and_manifest(
    tmp_path,
    monkeypatch,
):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)
    current_suite_artifact = tmp_path / "build/current-regression-suite-current.json"
    _write_json(
        current_suite_artifact,
        {
            "status": "open",
            "open_requirements": suite.EXPECTED_OPEN_REQUIREMENTS,
            "failed_steps": ["release_regression_manifest"],
        },
    )
    observed_failed_steps = {}

    def fake_run_step(name, cmd, cwd):
        if name in {"packaged_integrity_contracts", "release_regression_manifest"}:
            observed_failed_steps[name] = json.loads(
                current_suite_artifact.read_text(encoding="utf-8")
            )["failed_steps"]
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(
        tmp_path,
        include_release_gate=False,
        current_suite_artifact_path=current_suite_artifact,
    )

    assert artifact["status"] == "open"
    assert observed_failed_steps == {
        "packaged_integrity_contracts": [],
        "release_regression_manifest": [],
    }


def test_current_regression_suite_rejects_release_gate_crash_as_expected_failure():
    from tests.cross_matrix import run_current_regression_suite as suite

    step = {
        "name": "release_gate_skip_app",
        "returncode": 1,
        "stdout_tail": [
            "[FAIL] objective proof digest: "
            + "; ".join(suite.EXPECTED_OPEN_REQUIREMENTS),
            "Traceback (most recent call last):",
            "ModuleNotFoundError: No module named 'tests.cross_matrix'",
            "[FAIL] release-ready manifest: exit=1; log=/tmp/release-ready.log",
        ],
    }

    assert suite._release_gate_failure_is_expected(step) is False


def test_noheavy_api_cache_contract_includes_live_dsv4_nested_name_repair():
    from tests.cross_matrix.run_noheavy_api_cache_contract import COMMANDS

    command = " ".join(COMMANDS["dsv4_dsml_tool_contracts"])

    assert "dsml_parser_repairs_dsv4_live_degraded_dsml_params" in command
    assert "dsml_parser_repairs_partial_invoke_with_malformed_value_attr" in command
    assert "dsml_parser_repairs_htmlish_invoke_degradation" in command
    assert "server_repairs_dsv4_partial_tool_intent_from_request_args" in command
    assert "dsv4_encoder_preserves_code_identifiers_on_direct_chat_rail" in command


def test_noheavy_api_cache_contract_includes_output_context_precedence_gate():
    from tests.cross_matrix.run_noheavy_api_cache_contract import COMMANDS

    command = " ".join(COMMANDS["api_route_contracts"])

    assert "request_output_caps_override_server_default_without_touching_context_cap" in command


def test_noheavy_api_cache_contract_default_out_tracks_current_suite_artifact():
    from tests.cross_matrix import run_noheavy_api_cache_contract as gate

    assert gate.DEFAULT_OUT == Path(
        "build/current-api-cache-contract-proof-20260531-post-step-lfm-refresh.json"
    )


def test_noheavy_panel_settings_contract_parses_vitest_test_count():
    from tests.cross_matrix.run_noheavy_panel_settings_contract import _parse_counts

    output = """
     Test Files  3 passed (3)
          Tests  267 passed (267)
       Start at  13:52:21
    """

    counts = _parse_counts(output)

    assert counts["test_files_passed"] == 3
    assert counts["tests_passed"] == 267


def test_noheavy_panel_settings_contract_default_out_tracks_current_release_proof_artifact():
    from pathlib import Path

    from tests.cross_matrix import run_noheavy_panel_settings_contract as gate

    assert gate.DEFAULT_OUT == Path(
        "build/current-panel-settings-contract-proof-20260601-cache-ui-storage-quant.json"
    )


def test_current_regression_suite_runs_panel_settings_contract_to_current_artifact(monkeypatch, tmp_path):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    seen_steps = []

    def fake_run_step(name, cmd, cwd):
        seen_steps.append((name, cmd))
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    assert any(
        name == "noheavy_panel_settings_contract"
        and "current-panel-settings-contract-proof-20260601-cache-ui-storage-quant.json" in " ".join(cmd)
        for name, cmd in seen_steps
    )


def test_noheavy_panel_settings_contract_hashes_parser_mtp_and_migration_sources():
    from tests.cross_matrix.run_noheavy_panel_settings_contract import SOURCE_HASH_FILES

    required = {
        "panel/src/shared/sessionConfigMigrations.ts",
        "panel/src/shared/reasoningParserAliases.ts",
        "panel/src/main/model-config-registry.ts",
        "panel/src/renderer/src/components/sessions/CreateSession.tsx",
        "panel/src/renderer/src/components/sessions/ServerSettingsDrawer.tsx",
        "panel/src/renderer/src/components/sessions/CachePanel.tsx",
        "panel/src/renderer/src/components/sessions/PerformancePanel.tsx",
        "panel/src/renderer/src/i18n/locales/en.json",
        "panel/src/renderer/src/i18n/locales/es.json",
        "panel/src/renderer/src/i18n/locales/ja.json",
        "panel/src/renderer/src/i18n/locales/ko.json",
        "panel/src/renderer/src/i18n/locales/zh.json",
        "panel/tests/generation-defaults.test.ts",
        "panel/tests/i18n-consistency.test.ts",
    }

    assert required.issubset(set(SOURCE_HASH_FILES))


def test_noheavy_panel_settings_contract_runs_generation_defaults_and_i18n_tests():
    from tests.cross_matrix.run_noheavy_panel_settings_contract import COMMANDS

    command = " ".join(COMMANDS["panel_settings_contracts"])

    assert "tests/generation-defaults.test.ts" in command
    assert "tests/i18n-consistency.test.ts" in command


def test_noheavy_panel_settings_contract_runs_model_family_registry_tests():
    from tests.cross_matrix.run_noheavy_panel_settings_contract import COMMANDS

    panel_command = " ".join(COMMANDS["panel_model_config_registry"])
    engine_command = " ".join(COMMANDS["engine_model_config_registry"])

    assert "tests/model-config-registry.test.ts" in panel_command
    assert "tests/test_model_config_registry.py" in engine_command
    assert "-k" not in COMMANDS["engine_model_config_registry"]


def test_current_regression_suite_refreshes_release_regression_manifest(monkeypatch, tmp_path):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    seen_steps = []

    def fake_run_step(name, cmd, cwd):
        seen_steps.append((name, cmd))
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    assert any(name == "release_regression_manifest" for name, _cmd in seen_steps)
    assert any(
        "run_release_regression_manifest.py" in " ".join(cmd)
        for _name, cmd in seen_steps
    )
    assert any(
        name == "release_regression_manifest"
        and "--require-current-proof-sweep" in cmd
        for name, cmd in seen_steps
    )
    assert any(
        name == "release_regression_manifest"
        and "--require-release-ready" in cmd
        for name, cmd in seen_steps
    )
    assert any(
        name == "dsv4_source_exactness_memory_preflight"
        and "tests/cross_matrix/run_dsv4_route_mode_code_exactness.py" in " ".join(cmd)
        and "--memory-preflight-only" in cmd
        and "--cases" not in cmd
        and "build/current-dsv4-route-mode-code-exactness-memory-preflight-20260601-post-epipe-fix.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert any(
        name == "real_ui_dsv4_memory_preflight"
        and "tests/cross_matrix/run_real_ui_dsv4_memory_preflight.py" in " ".join(cmd)
        and "build/current-real-ui-dsv4-memory-preflight-20260601-local-recheck.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260601-pipe-safe-runner.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260601-developer-id-dmg-assertions-refresh.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260601-after-adhoc-reseal.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260601-qwen3vl-minicpm-mpp-refresh.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260528-dsv4-continue-refresh.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260528-installed-aggregate-stale.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260528-epipe-aggregate-guard.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260528-nonmimo-zaya-direct.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260528-ollama-chat-epipe-dsv4-preflight.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260528-dsv4-0625-preflight-caseguard.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260528-dsv4-0625-preflight.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260528-current-live-smoke-pointers.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260528-dsv4-0600-preflight.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260528-userdata-epipe-scan.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260528-ipc-epipe-request-guard.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260527-zaya-vl-point-leak-proof-tightening.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260527-zaya-vl-cachecontrols.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260527-gemma4-speed-overclaim-fixed.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260527-real-ui-family-expansion.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260527-real-ui-zaya-vl-image.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260526-real-ui-live-model-slice.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260526-dev-ui-wiring.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260526-settings-audit.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260525-smoke-reasoning-loop-guard.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260525-dsv4-copy-block-boundary.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260525-dsv4-prompt-guard-boundary.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260525-dsv4-nocache-ab-boundary.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260525-dsv4-default-cache-raw-boundary.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260525-dsv4-bundle-defaults-cjk-wide.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_regression_manifest"
        and "build/current-release-regression-manifest-20260524-openai-single-model-streaming-audit.json"
        in cmd
        for name, cmd in seen_steps
    )


def test_current_regression_suite_allows_release_manifest_not_ready_for_known_open_requirements(
    tmp_path,
):
    from tests.cross_matrix import run_current_regression_suite as suite

    manifest_path = tmp_path / "build/current-release-regression-manifest.json"
    _write_json(
        manifest_path,
        {
            "status": "fail",
            "release_ready": False,
            "current_proof_sweep": {"status": "fail"},
            "release_clearance": {
                "release_ready": False,
                "open_requirements": suite.EXPECTED_OPEN_REQUIREMENTS,
                "blockers": [
                    {"id": "dsv4_long_output_code_exactness_open"},
                    {"id": "issue179_minimax_k_root_cause_audit"},
                    {"id": "issue175_179_release_boundary_audit"},
                    {"id": "installed_app_runtime_parity_audit"},
                    {"id": "real_ui_dsv4_memory_blocked"},
                    {"id": "real_ui_request_contract_proofs_stale"},
                    {"id": "real_ui_step37_vlm_runtime_missing"},
                    {"id": "real_ui_unblocked_non_mimo_partial"},
                    {"id": "packaged_app_developer_id_signing_blocked"},
                ],
            },
        },
    )

    step = {
        "name": "release_regression_manifest",
        "command": [
            "python",
            "tests/cross_matrix/run_release_regression_manifest.py",
            "--require-release-ready",
            "--out",
            str(manifest_path.relative_to(tmp_path)),
        ],
        "returncode": 1,
        "stdout_tail": [
            str(manifest_path.relative_to(tmp_path)),
            "current_proof_sweep=fail",
            "release_ready=false",
        ],
    }

    assert suite._step_is_ok("release_regression_manifest", step, tmp_path) is True


def test_current_regression_suite_rejects_release_manifest_with_unexpected_open_requirement(
    tmp_path,
):
    from tests.cross_matrix import run_current_regression_suite as suite

    manifest_path = tmp_path / "build/current-release-regression-manifest.json"
    _write_json(
        manifest_path,
        {
            "status": "fail",
            "release_ready": False,
            "release_clearance": {
                "release_ready": False,
                "open_requirements": [
                    *suite.EXPECTED_OPEN_REQUIREMENTS,
                    "Unexpected quality row is still open",
                ],
                "blockers": [],
            },
        },
    )

    step = {
        "name": "release_regression_manifest",
        "command": ["python", "--out", str(manifest_path.relative_to(tmp_path))],
        "returncode": 1,
        "stdout_tail": [],
    }

    assert suite._step_is_ok("release_regression_manifest", step, tmp_path) is False


def test_current_regression_suite_refreshes_current_packaged_integrity_artifact(
    monkeypatch,
    tmp_path,
):
    from tests.cross_matrix import release_regression_manifest as manifest
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    seen_steps = []

    def fake_run_step(name, cmd, cwd):
        seen_steps.append((name, cmd))
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    current_artifact = manifest.CURRENT_POST_BUDGET_EDGE_ARTIFACTS[
        "packaged-release-integrity"
    ]
    assert any(
        name == "packaged_integrity_contracts"
        and current_artifact in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "packaged_integrity_contracts"
        and "build/current-packaged-integrity-contract-20260531-live-signing-refresh.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "packaged_integrity_contracts"
        and "build/current-packaged-integrity-contract-20260528-installed-aggregate-stale.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "packaged_integrity_contracts"
        and "build/current-packaged-integrity-contract-20260528-prepackage-gate.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert any(
        name == "api_surface_contracts"
        and "build/current-api-surface-contract-20260601-cache-ipc-epipe-refresh.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "api_surface_contracts"
        and "build/current-api-surface-contract-20260531-nested-epipe-childstream-refresh.json"
        in cmd
        for name, cmd in seen_steps
    )
    step_names = [name for name, _cmd in seen_steps]
    assert step_names.index("api_surface_contracts") < step_names.index(
        "packaged_integrity_contracts"
    )
    assert step_names.index("generation_defaults_contracts") < step_names.index(
        "packaged_integrity_contracts"
    )
    assert step_names.index("native_mtp_contracts") < step_names.index(
        "packaged_integrity_contracts"
    )
    assert step_names.index("vl_media_cache_contracts") < step_names.index(
        "packaged_integrity_contracts"
    )
    assert not any(
        name == "packaged_integrity_contracts"
        and "build/current-packaged-integrity-contract-20260528-ollama-chat-epipe-guard.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "packaged_integrity_contracts"
        and "build/current-packaged-integrity-contract-20260528-ipc-epipe-request-guard.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "packaged_integrity_contracts"
        and "build/current-packaged-integrity-contract-20260527-after-think-xml-registry-fix-rerun.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "packaged_integrity_contracts"
        and "build/current-packaged-integrity-contract-20260525-additional-args-guard.json"
        in cmd
        for name, cmd in seen_steps
    )


def test_current_regression_suite_runs_panel_tool_security_contracts(monkeypatch, tmp_path):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    seen_steps = []

    def fake_run_step(name, cmd, cwd):
        seen_steps.append((name, cmd))
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    assert any(name == "panel_tool_security_contracts" for name, _cmd in seen_steps)
    assert any(
        "run_panel_tool_security_contract.py" in " ".join(cmd)
        for _name, cmd in seen_steps
    )
    assert any(
        name == "panel_tool_security_contracts"
        and "build/current-panel-tool-security-contract-20260528-tool-loop-security-matrix.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "panel_tool_security_contracts"
        and "build/current-panel-tool-security-contract-20260521.json" in cmd
        for name, cmd in seen_steps
    )


def test_current_regression_suite_refreshes_current_objective_digest_artifact(
    monkeypatch,
    tmp_path,
):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    seen_steps = []

    def fake_run_step(name, cmd, cwd):
        seen_steps.append((name, cmd))
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    assert suite.CURRENT_OBJECTIVE_DIGEST_ARTIFACT == (
        "build/current-objective-proof-audit-20260531-nemotron-exact-finalizer-ledger.json"
    )
    assert any(
        name == "objective_digest"
        and suite.CURRENT_OBJECTIVE_DIGEST_ARTIFACT in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "objective_digest"
        and "build/current-objective-proof-audit-20260521.json" in cmd
        for name, cmd in seen_steps
    )


def test_current_regression_suite_runs_release_surface_contract(monkeypatch, tmp_path):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    seen_steps = []

    def fake_run_step(name, cmd, cwd):
        seen_steps.append((name, cmd))
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    assert any(name == "release_surface_contracts" for name, _cmd in seen_steps)
    assert any(
        "run_release_surface_contract.py" in " ".join(cmd)
        for _name, cmd in seen_steps
    )
    assert any(
        name == "release_surface_contracts"
        and "build/current-release-surface-contract-20260528-release-surface-matrix.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "release_surface_contracts"
        and "build/current-release-surface-contract-20260521.json" in cmd
        for name, cmd in seen_steps
    )


def test_current_regression_suite_runs_cli_release_contracts(monkeypatch, tmp_path):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    seen_steps = []

    def fake_run_step(name, cmd, cwd):
        seen_steps.append((name, cmd))
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    assert any(name == "cli_release_contracts" for name, _cmd in seen_steps)
    assert any(
        "TestServeCommandSocketBinding" in " ".join(cmd)
        for _name, cmd in seen_steps
    )


def test_current_regression_suite_runs_jang_model_compat_contracts(monkeypatch, tmp_path):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    seen_steps = []

    def fake_run_step(name, cmd, cwd):
        seen_steps.append((name, cmd))
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    assert any(name == "jang_model_compat_contracts" for name, _cmd in seen_steps)
    assert any(
        "run_jang_model_compat_contract.py" in " ".join(cmd)
        for _name, cmd in seen_steps
    )
    assert any(
        name == "jang_model_compat_contracts"
        and "build/current-jang-model-compat-contract-20260528-pr155-runtime-boundary.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "jang_model_compat_contracts"
        and "build/current-jang-model-compat-contract-20260521.json" in cmd
        for name, cmd in seen_steps
    )


def test_current_suite_child_runner_defaults_track_current_artifacts():
    from tests.cross_matrix import run_jang_model_compat_contract
    from tests.cross_matrix import run_panel_tool_security_contract

    assert run_panel_tool_security_contract.DEFAULT_OUT == Path(
        "build/current-panel-tool-security-contract-20260528-tool-loop-security-matrix.json"
    )
    assert run_jang_model_compat_contract.DEFAULT_OUT == Path(
        "build/current-jang-model-compat-contract-20260528-pr155-runtime-boundary.json"
    )


def test_current_regression_suite_runs_model_artifact_format_contracts(monkeypatch, tmp_path):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    seen_steps = []

    def fake_run_step(name, cmd, cwd):
        seen_steps.append((name, cmd))
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    assert any(name == "model_artifact_format_contracts" for name, _cmd in seen_steps)
    assert any(
        "run_model_artifact_format_contract.py" in " ".join(cmd)
        for _name, cmd in seen_steps
    )


def test_current_regression_suite_runs_model_family_detection_contracts(monkeypatch, tmp_path):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    seen_steps = []

    def fake_run_step(name, cmd, cwd):
        seen_steps.append((name, cmd))
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    assert any(name == "model_family_detection_contracts" for name, _cmd in seen_steps)
    assert any(
        "run_model_family_detection_contract.py" in " ".join(cmd)
        for _name, cmd in seen_steps
    )


def test_current_regression_suite_runs_parser_registry_contracts(monkeypatch, tmp_path):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    seen_steps = []

    def fake_run_step(name, cmd, cwd):
        seen_steps.append((name, cmd))
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    assert any(name == "parser_registry_contracts" for name, _cmd in seen_steps)
    assert any(
        "run_parser_registry_contract.py" in " ".join(cmd)
        for _name, cmd in seen_steps
    )


def test_current_regression_suite_runs_max_output_context_contracts(monkeypatch, tmp_path):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    seen_steps = []

    def fake_run_step(name, cmd, cwd):
        seen_steps.append((name, cmd))
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    assert any(name == "max_output_context_contracts" for name, _cmd in seen_steps)
    assert any(
        "run_max_output_context_contract.py" in " ".join(cmd)
        for _name, cmd in seen_steps
    )


def test_current_regression_suite_runs_vl_media_cache_contracts(monkeypatch, tmp_path):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    seen_steps = []

    def fake_run_step(name, cmd, cwd):
        seen_steps.append((name, cmd))
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    assert any(name == "vl_media_cache_contracts" for name, _cmd in seen_steps)
    assert any(
        "run_vl_media_cache_contract.py" in " ".join(cmd)
        for _name, cmd in seen_steps
    )


def test_current_regression_suite_runs_cache_architecture_contracts(monkeypatch, tmp_path):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    seen_steps = []

    def fake_run_step(name, cmd, cwd):
        seen_steps.append((name, cmd))
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    assert any(name == "cache_architecture_contracts" for name, _cmd in seen_steps)
    assert any(
        "run_cache_architecture_contract.py" in " ".join(cmd)
        for _name, cmd in seen_steps
    )


def test_current_regression_suite_runs_native_mtp_contracts(monkeypatch, tmp_path):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    seen_steps = []

    def fake_run_step(name, cmd, cwd):
        seen_steps.append((name, cmd))
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    assert any(name == "native_mtp_contracts" for name, _cmd in seen_steps)
    assert any(
        "run_native_mtp_contract.py" in " ".join(cmd)
        for _name, cmd in seen_steps
    )


def test_current_regression_suite_runs_generation_defaults_contracts(monkeypatch, tmp_path):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    seen_steps = []

    def fake_run_step(name, cmd, cwd):
        seen_steps.append((name, cmd))
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    assert any(name == "generation_defaults_contracts" for name, _cmd in seen_steps)
    assert any(
        "run_generation_defaults_contract.py" in " ".join(cmd)
        for _name, cmd in seen_steps
    )


def test_current_regression_suite_runs_reasoning_template_contracts(monkeypatch, tmp_path):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    seen_steps = []

    def fake_run_step(name, cmd, cwd):
        seen_steps.append((name, cmd))
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    assert any(name == "reasoning_template_contracts" for name, _cmd in seen_steps)
    assert any(
        "run_reasoning_template_contract.py" in " ".join(cmd)
        for _name, cmd in seen_steps
    )
    assert any(
        name == "reasoning_template_contracts"
        and "build/current-reasoning-template-contract-20260526-settings-audit.json"
        in cmd
        for name, cmd in seen_steps
    )
    assert not any(
        name == "reasoning_template_contracts"
        and "build/current-reasoning-template-contract-20260521.json" in cmd
        for name, cmd in seen_steps
    )


def test_current_regression_suite_runs_api_surface_contracts(monkeypatch, tmp_path):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    seen_steps = []

    def fake_run_step(name, cmd, cwd):
        seen_steps.append((name, cmd))
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    assert any(name == "api_surface_contracts" for name, _cmd in seen_steps)
    assert any(
        "run_api_surface_contract.py" in " ".join(cmd)
        for _name, cmd in seen_steps
    )


def test_current_regression_suite_runs_tool_call_contracts(monkeypatch, tmp_path):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    seen_steps = []

    def fake_run_step(name, cmd, cwd):
        seen_steps.append((name, cmd))
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    assert any(name == "tool_call_contracts" for name, _cmd in seen_steps)
    assert any(
        "run_tool_call_contract.py" in " ".join(cmd)
        and "current-tool-call-contract-20260528-tool-parser-loop-matrix.json" in " ".join(cmd)
        for _name, cmd in seen_steps
    )


def test_current_regression_suite_runs_mcp_policy_contracts(monkeypatch, tmp_path):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    seen_steps = []

    def fake_run_step(name, cmd, cwd):
        seen_steps.append((name, cmd))
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    assert any(name == "mcp_policy_contracts" for name, _cmd in seen_steps)
    assert any(
        "run_mcp_policy_contract.py" in " ".join(cmd)
        and "current-mcp-policy-contract-20260531-post-step-lfm-refresh.json" in " ".join(cmd)
        for _name, cmd in seen_steps
    )


def test_current_regression_suite_runs_mcp_policy_marker_contract(monkeypatch, tmp_path):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    seen_steps = []

    def fake_run_step(name, cmd, cwd):
        seen_steps.append((name, cmd))
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    focused = next(cmd for name, cmd in seen_steps if name == "focused_regression_pytest")
    joined = " ".join(focused)
    assert "tests/test_mcp_policy_contract.py" in joined
    assert "mcp_policy_contract" in joined


def test_current_regression_suite_runs_dsv4_processor_context_regressions(
    monkeypatch, tmp_path
):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    seen_steps = []

    def fake_run_step(name, cmd, cwd):
        seen_steps.append((name, cmd))
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    focused = next(cmd for name, cmd in seen_steps if name == "focused_regression_pytest")
    joined = " ".join(focused)
    assert "tests/test_batching.py" in joined
    assert "tests/test_scheduler_repetition_context.py" in joined
    assert "tests/test_dsv4_paged_cache.py" in joined
    assert "dsv4_cache_hit_repetition_processor" in joined
    assert "generated_only_logits_processor" in joined
    assert "dsv4_repetition_penalty_uses_generated_only_prompt_context" in joined


def test_current_regression_suite_runs_packaged_integrity_contracts(monkeypatch, tmp_path):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    seen_steps = []

    def fake_run_step(name, cmd, cwd):
        seen_steps.append((name, cmd))
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "open"
    assert any(name == "packaged_integrity_contracts" for name, _cmd in seen_steps)
    assert any(
        "run_packaged_integrity_contract.py" in " ".join(cmd)
        for _name, cmd in seen_steps
    )


def test_current_regression_suite_sets_clean_jang_source_env_for_release_children(monkeypatch, tmp_path):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    clean_jang = tmp_path / "clean-jang" / "jang-tools"
    seen_env = {}

    def fake_run_step(name, cmd, cwd):
        if name in {"packaged_integrity_contracts", "release_gate_skip_app"}:
            seen_env[name] = (
                os.environ.get("VMLX_JANG_TOOLS_SOURCE"),
                os.environ.get("VMLINUX_JANG_TOOLS_SOURCE"),
            )
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(
        tmp_path,
        include_release_gate=True,
        jang_tools_source=clean_jang,
    )

    assert artifact["status"] == "open"
    assert seen_env["packaged_integrity_contracts"] == (str(clean_jang), str(clean_jang))
    assert seen_env["release_gate_skip_app"] == (str(clean_jang), str(clean_jang))


def test_api_surface_contract_parses_nested_runner_counts():
    from tests.cross_matrix.run_api_surface_contract import _parse_counts

    output = """
    build/current-api-cache-contract-api-surface-check-20260521.json
    status=pass
    api_route_contracts: rc=0 passed=15 deselected=397
    """

    counts = _parse_counts(output)

    assert counts["passed"] == 15
    assert counts["deselected"] == 397
