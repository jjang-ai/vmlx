import json
import os
from pathlib import Path


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _write_known_open_objective_digest(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "build/current-objective-proof-audit-20260521.json",
        {
            "requirements": [
                {
                    "requirement": "DSV4 long-output/code/file-generation quality is release-cleared",
                    "status": "open",
                },
            ]
        },
    )


def test_current_regression_suite_allows_only_declared_known_blockers(tmp_path, monkeypatch):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_json(
        tmp_path / "build/current-objective-proof-audit-20260521.json",
        {
            "requirements": [
                {"requirement": "DSV4 Flash prefix/paged/L2 cache is enabled by default from app launch", "status": "pass"},
                {"requirement": "DSV4 long-output/code/file-generation quality is release-cleared", "status": "open"},
            ]
        },
    )

    def fake_run_step(name, cmd, cwd):
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=True)

    assert artifact["status"] == "pass"
    assert artifact["known_open_requirements"] == suite.EXPECTED_OPEN_REQUIREMENTS
    assert artifact["unexpected_open_requirements"] == []
    assert artifact["missing_expected_open_requirements"] == []
    assert artifact["steps"]["release_gate_skip_app"]["returncode"] == 0


def test_current_regression_suite_fails_on_new_unexpected_open_requirement(tmp_path, monkeypatch):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_json(
        tmp_path / "build/current-objective-proof-audit-20260521.json",
        {
            "requirements": [
                {"requirement": "DSV4 long-output/code/file-generation quality is release-cleared", "status": "open"},
                {"requirement": "Server default max output and max context are distinct and map to correct CLI flags", "status": "open"},
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
        "Server default max output and max context are distinct and map to correct CLI flags"
    ]


def test_current_regression_suite_fails_on_step_failure_even_if_digest_is_expected(tmp_path, monkeypatch):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_json(
        tmp_path / "build/current-objective-proof-audit-20260521.json",
        {
            "requirements": [
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


def test_noheavy_panel_settings_contract_hashes_parser_mtp_and_migration_sources():
    from tests.cross_matrix.run_noheavy_panel_settings_contract import SOURCE_HASH_FILES

    required = {
        "panel/src/shared/sessionConfigMigrations.ts",
        "panel/src/shared/reasoningParserAliases.ts",
        "panel/src/main/model-config-registry.ts",
        "panel/src/renderer/src/components/sessions/CreateSession.tsx",
        "panel/src/renderer/src/components/sessions/ServerSettingsDrawer.tsx",
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

    assert artifact["status"] == "pass"
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


def test_current_regression_suite_runs_panel_tool_security_contracts(monkeypatch, tmp_path):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    seen_steps = []

    def fake_run_step(name, cmd, cwd):
        seen_steps.append((name, cmd))
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "pass"
    assert any(name == "panel_tool_security_contracts" for name, _cmd in seen_steps)
    assert any(
        "run_panel_tool_security_contract.py" in " ".join(cmd)
        for _name, cmd in seen_steps
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

    assert artifact["status"] == "pass"
    assert any(name == "release_surface_contracts" for name, _cmd in seen_steps)
    assert any(
        "run_release_surface_contract.py" in " ".join(cmd)
        for _name, cmd in seen_steps
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

    assert artifact["status"] == "pass"
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

    assert artifact["status"] == "pass"
    assert any(name == "jang_model_compat_contracts" for name, _cmd in seen_steps)
    assert any(
        "run_jang_model_compat_contract.py" in " ".join(cmd)
        for _name, cmd in seen_steps
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

    assert artifact["status"] == "pass"
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

    assert artifact["status"] == "pass"
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

    assert artifact["status"] == "pass"
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

    assert artifact["status"] == "pass"
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

    assert artifact["status"] == "pass"
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

    assert artifact["status"] == "pass"
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

    assert artifact["status"] == "pass"
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

    assert artifact["status"] == "pass"
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

    assert artifact["status"] == "pass"
    assert any(name == "reasoning_template_contracts" for name, _cmd in seen_steps)
    assert any(
        "run_reasoning_template_contract.py" in " ".join(cmd)
        for _name, cmd in seen_steps
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

    assert artifact["status"] == "pass"
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

    assert artifact["status"] == "pass"
    assert any(name == "tool_call_contracts" for name, _cmd in seen_steps)
    assert any(
        "run_tool_call_contract.py" in " ".join(cmd)
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

    assert artifact["status"] == "pass"
    assert any(name == "mcp_policy_contracts" for name, _cmd in seen_steps)
    assert any(
        "run_mcp_policy_contract.py" in " ".join(cmd)
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

    assert artifact["status"] == "pass"
    focused = next(cmd for name, cmd in seen_steps if name == "focused_regression_pytest")
    joined = " ".join(focused)
    assert "tests/test_mcp_policy_contract.py" in joined
    assert "mcp_policy_contract" in joined


def test_current_regression_suite_runs_packaged_integrity_contracts(monkeypatch, tmp_path):
    from tests.cross_matrix import run_current_regression_suite as suite

    _write_known_open_objective_digest(tmp_path)

    seen_steps = []

    def fake_run_step(name, cmd, cwd):
        seen_steps.append((name, cmd))
        return {"name": name, "command": cmd, "returncode": 0, "stdout_tail": []}

    monkeypatch.setattr(suite, "_run_step", fake_run_step)

    artifact = suite.build_suite_artifact(tmp_path, include_release_gate=False)

    assert artifact["status"] == "pass"
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

    assert artifact["status"] == "pass"
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
