from __future__ import annotations

import os
import json
from pathlib import Path

from tests.cross_matrix import run_packaged_integrity_contract as runner
from tests.cross_matrix import run_current_regression_suite as suite


def test_packaged_integrity_known_open_rows_match_current_suite():
    assert runner.EXPECTED_OPEN_REQUIREMENTS == suite.EXPECTED_OPEN_REQUIREMENTS


def test_packaged_integrity_default_out_tracks_current_release_proof_artifact():
    assert runner.DEFAULT_OUT == Path(
        "build/current-packaged-integrity-contract-after-checkpoint-app-parity-20260609.json"
    )


def test_packaged_integrity_hashes_cache_ipc_guard_source():
    assert "panel/src/main/ipc/cache.ts" in runner.SOURCE_HASH_FILES
    assert "tests/test_packaged_integrity_contract.py" in runner.SOURCE_HASH_FILES


def _result(name: str, returncode: int, stdout_tail: list[str], passed: int | None = None):
    return {
        "name": name,
        "command": [name],
        "cwd": ".",
        "returncode": returncode,
        "elapsed_sec": 0.0,
        "counts": {"passed": passed, "skipped": None, "deselected": None},
        "stdout_tail": stdout_tail,
    }


def _expected_open_digest_line() -> str:
    return "[FAIL] objective proof digest: " + "; ".join(
        item
        for item in runner.EXPECTED_OPEN_REQUIREMENTS
        if item not in runner.SUITE_DEFERRED_RELEASE_OPEN_REQUIREMENTS
    )


def _current_objective_open_digest_line() -> str:
    artifact = json.loads(
        Path(runner.CURRENT_OBJECTIVE_DIGEST_ARTIFACT).read_text(encoding="utf-8")
    )
    open_requirements = [
        item["requirement"]
        for item in artifact["requirements"]
        if item["status"] != "pass"
        and item["requirement"] not in runner.SUITE_DEFERRED_RELEASE_OPEN_REQUIREMENTS
    ]
    return "[FAIL] objective proof digest: " + "; ".join(open_requirements)


def _expected_release_ready_line() -> str:
    return "[FAIL] release-ready manifest: exit=1; log=/tmp/release-ready.log"


def _expected_release_gate_failure_tail() -> list[str]:
    return [_expected_open_digest_line()]


def _signed_preflight() -> dict[str, object]:
    return {
        "status": "pass",
        "signing_blocker_reason": None,
        "signing_blocker_reasons": [],
    }


def test_packaged_integrity_accepts_release_gate_objective_digest_fallback_failure(
    tmp_path,
):
    step = _result(
        "release_gate_skip_app",
        1,
        _expected_release_gate_failure_tail(),
    )

    assert runner.release_gate_failure_is_expected(step, tmp_path)


def test_packaged_integrity_accepts_current_objective_digest_only_failure():
    step = _result(
        "release_gate_skip_app",
        1,
        [_current_objective_open_digest_line()],
    )

    assert runner.release_gate_failure_is_expected(step)


def test_packaged_integrity_accepts_release_gate_known_objectives_plus_manifest_failure(
    tmp_path,
):
    step = _result(
        "release_gate_skip_app",
        1,
        [
            _expected_open_digest_line(),
            _expected_release_ready_line(),
        ],
    )

    assert runner.release_gate_failure_is_expected(step, tmp_path)


def test_packaged_integrity_rejects_release_ready_manifest_crash_as_expected_failure(
    tmp_path,
):
    step = _result(
        "release_gate_skip_app",
        1,
        [
            _expected_open_digest_line(),
            "Traceback (most recent call last):",
            "ModuleNotFoundError: No module named 'tests.cross_matrix'",
        ],
    )

    assert runner.release_gate_failure_is_expected(step, tmp_path) is False


def test_packaged_integrity_rejects_stale_objective_digest_refresh_log(tmp_path):
    log = tmp_path / "objective_proof_digest_refresh.log"
    log.write_text(
        str(Path.cwd() / "build/current-objective-proof-audit-20260521.json") + "\n",
        encoding="utf-8",
    )
    step = _result(
        "release_gate_skip_app",
        1,
        [
            f"[PASS] objective proof digest refresh: 0.1s; log={log}",
            _expected_open_digest_line(),
        ],
    )

    assert runner.dry_release_gate_used_current_objective_digest(step) is False


def test_packaged_integrity_accepts_current_objective_digest_refresh_log(tmp_path):
    log = tmp_path / "objective_proof_digest_refresh.log"
    log.write_text(
        str(Path.cwd() / runner.CURRENT_OBJECTIVE_DIGEST_ARTIFACT) + "\n",
        encoding="utf-8",
    )
    step = _result(
        "release_gate_skip_app",
        1,
        [
            f"[PASS] objective proof digest refresh: 0.1s; log={log}",
            _expected_open_digest_line(),
        ],
    )

    assert runner.dry_release_gate_used_current_objective_digest(step) is True


def test_packaged_integrity_accepts_current_release_gate_unit_count(monkeypatch, tmp_path):
    def fake_run(_root: Path, name: str, _cwd_rel: Path, _cmd: list[str]):
        if name == "release_gate_unit_contracts":
            return _result(name, 0, ["34 passed in 0.07s"], passed=runner.MIN_RELEASE_GATE_UNIT_TESTS)
        if name == "bundled_python_verifier":
            return _result(
                name,
                0,
                [
                    "  ok   bundled vmlx_engine version matches package.json",
                    "  ok   bundled critical vmlx_engine files match source content",
                    "  ok   bundled critical jang_tools files match source content",
                    "  ok   bundled-python console-script shebangs are relocatable",
                    "bundled-python: all critical imports ok",
                ],
            )
        if name == "release_gate_skip_app":
            return _result(
                name,
                1,
                _expected_release_gate_failure_tail(),
            )
        raise AssertionError(name)

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(runner, "_sha256", lambda _path: "hash")
    monkeypatch.setattr(runner, "_check_packaged_renderer_dsv4_cache_ui", lambda _root: True)
    monkeypatch.setattr(runner, "_check_packaged_renderer_max_thinking_tokens", lambda _root: True)
    monkeypatch.setattr(runner, "_check_packaged_epipe_closed_stream_guards", lambda _root: True)
    monkeypatch.setattr(runner, "_check_packaged_user_data_isolation_bootstrap", lambda _root: True)
    monkeypatch.setattr(runner, "_check_packaged_python_has_no_pycache", lambda _root: True)
    monkeypatch.setattr(runner, "_check_staged_app_engine_hash_parity", lambda _root: True)
    monkeypatch.setattr(runner, "_check_staged_app_engine_source_hash_parity", lambda _root: True)
    monkeypatch.setattr(runner, "_check_release_dmg_hardened_runtime_contract", lambda _root: True)
    monkeypatch.setattr(runner, "_check_release_dmg_notarization_verifier_contract", lambda _root: True)
    monkeypatch.setattr(runner, "_check_release_dmg_notarization_submit_contract", lambda _root: True)
    monkeypatch.setattr(runner, "dry_release_gate_used_current_objective_digest", lambda _step: True)
    monkeypatch.setattr(runner, "_package_signing_preflight", lambda _root: _signed_preflight())

    artifact = runner.build_artifact(tmp_path)

    assert artifact["checks"]["release_gate_unit_contracts_pass"] is True
    assert artifact["checks"]["dry_release_gate_fails_only_on_known_objectives"] is True
    assert artifact["checks"]["dry_release_gate_uses_current_objective_digest"] is True
    assert artifact["known_expected_release_gate_open_requirements"] == (
        runner.EXPECTED_OPEN_REQUIREMENTS
    )
    assert artifact["status"] == "pass"


def test_packaged_integrity_sets_clean_jang_source_env_for_bundle_checks(monkeypatch, tmp_path):
    clean_jang = tmp_path / "clean-jang" / "jang-tools"
    seen_env = {}

    def fake_run(_root: Path, name: str, _cwd_rel: Path, _cmd: list[str]):
        seen_env[name] = (
            os.environ.get("VMLX_JANG_TOOLS_SOURCE"),
            os.environ.get("VMLINUX_JANG_TOOLS_SOURCE"),
        )
        if name == "release_gate_unit_contracts":
            return _result(name, 0, ["34 passed in 0.07s"], passed=runner.MIN_RELEASE_GATE_UNIT_TESTS)
        if name == "bundled_python_verifier":
            return _result(
                name,
                0,
                [
                    "  ok   bundled vmlx_engine version matches package.json",
                    "  ok   bundled critical vmlx_engine files match source content",
                    "  ok   bundled critical jang_tools files match source content",
                    "  ok   bundled-python console-script shebangs are relocatable",
                    "bundled-python: all critical imports ok",
                ],
            )
        if name == "release_gate_skip_app":
            return _result(
                name,
                1,
                _expected_release_gate_failure_tail(),
            )
        raise AssertionError(name)

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(runner, "_sha256", lambda _path: "hash")
    monkeypatch.setattr(runner, "_check_packaged_renderer_dsv4_cache_ui", lambda _root: True)
    monkeypatch.setattr(runner, "_check_packaged_renderer_max_thinking_tokens", lambda _root: True)
    monkeypatch.setattr(runner, "_check_packaged_epipe_closed_stream_guards", lambda _root: True)
    monkeypatch.setattr(runner, "_check_packaged_user_data_isolation_bootstrap", lambda _root: True)
    monkeypatch.setattr(runner, "_check_packaged_python_has_no_pycache", lambda _root: True)
    monkeypatch.setattr(runner, "_check_staged_app_engine_hash_parity", lambda _root: True)
    monkeypatch.setattr(runner, "_check_staged_app_engine_source_hash_parity", lambda _root: True)
    monkeypatch.setattr(runner, "_check_release_dmg_hardened_runtime_contract", lambda _root: True)
    monkeypatch.setattr(runner, "_check_release_dmg_notarization_verifier_contract", lambda _root: True)
    monkeypatch.setattr(runner, "_check_release_dmg_notarization_submit_contract", lambda _root: True)
    monkeypatch.setattr(runner, "dry_release_gate_used_current_objective_digest", lambda _step: True)
    monkeypatch.setattr(runner, "_package_signing_preflight", lambda _root: _signed_preflight())

    artifact = runner.build_artifact(tmp_path, jang_tools_source=clean_jang)

    assert artifact["status"] == "pass"
    assert seen_env["bundled_python_verifier"] == (str(clean_jang), str(clean_jang))
    assert seen_env["release_gate_skip_app"] == (str(clean_jang), str(clean_jang))


def test_packaged_integrity_checks_packaged_dsv4_cache_ui_labels(monkeypatch, tmp_path):
    def fake_run(_root: Path, name: str, _cwd_rel: Path, _cmd: list[str]):
        if name == "release_gate_unit_contracts":
            return _result(name, 0, ["34 passed in 0.07s"], passed=runner.MIN_RELEASE_GATE_UNIT_TESTS)
        if name == "bundled_python_verifier":
            return _result(
                name,
                0,
                [
                    "  ok   bundled vmlx_engine version matches package.json",
                    "  ok   bundled critical vmlx_engine files match source content",
                    "  ok   bundled critical jang_tools files match source content",
                    "  ok   bundled-python console-script shebangs are relocatable",
                    "bundled-python: all critical imports ok",
                ],
            )
        if name == "release_gate_skip_app":
            return _result(name, 1, _expected_release_gate_failure_tail())
        raise AssertionError(name)

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(runner, "_sha256", lambda _path: "hash")
    monkeypatch.setattr(runner, "_check_packaged_renderer_dsv4_cache_ui", lambda _root: True)
    monkeypatch.setattr(runner, "_check_packaged_renderer_max_thinking_tokens", lambda _root: True)
    monkeypatch.setattr(runner, "_check_packaged_epipe_closed_stream_guards", lambda _root: True)

    artifact = runner.build_artifact(tmp_path)

    assert "packaged_renderer_dsv4_cache_ui_deduped" in artifact["checks"]


def test_packaged_integrity_checks_packaged_max_thinking_tokens_wiring(monkeypatch, tmp_path):
    def fake_run(_root: Path, name: str, _cwd_rel: Path, _cmd: list[str]):
        if name == "release_gate_unit_contracts":
            return _result(name, 0, ["34 passed in 0.07s"], passed=runner.MIN_RELEASE_GATE_UNIT_TESTS)
        if name == "bundled_python_verifier":
            return _result(
                name,
                0,
                [
                    "  ok   bundled vmlx_engine version matches package.json",
                    "  ok   bundled critical vmlx_engine files match source content",
                    "  ok   bundled critical jang_tools files match source content",
                    "  ok   bundled-python console-script shebangs are relocatable",
                    "bundled-python: all critical imports ok",
                ],
            )
        if name == "release_gate_skip_app":
            return _result(name, 1, _expected_release_gate_failure_tail())
        raise AssertionError(name)

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(runner, "_sha256", lambda _path: "hash")
    monkeypatch.setattr(runner, "_check_packaged_renderer_dsv4_cache_ui", lambda _root: True)
    monkeypatch.setattr(runner, "_check_packaged_renderer_max_thinking_tokens", lambda _root: True)
    monkeypatch.setattr(runner, "_check_packaged_epipe_closed_stream_guards", lambda _root: True)

    artifact = runner.build_artifact(tmp_path)

    assert "packaged_renderer_max_thinking_tokens_wired" in artifact["checks"]


def test_packaged_integrity_checks_packaged_user_data_isolation_bootstrap(monkeypatch, tmp_path):
    def fake_run(_root: Path, name: str, _cwd_rel: Path, _cmd: list[str]):
        if name == "release_gate_unit_contracts":
            return _result(name, 0, ["34 passed in 0.07s"], passed=runner.MIN_RELEASE_GATE_UNIT_TESTS)
        if name == "bundled_python_verifier":
            return _result(
                name,
                0,
                [
                    "  ok   bundled vmlx_engine version matches package.json",
                    "  ok   bundled critical vmlx_engine files match source content",
                    "  ok   bundled critical jang_tools files match source content",
                    "  ok   bundled-python console-script shebangs are relocatable",
                    "bundled-python: all critical imports ok",
                ],
            )
        if name == "release_gate_skip_app":
            return _result(name, 1, _expected_release_gate_failure_tail())
        raise AssertionError(name)

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(runner, "_sha256", lambda _path: "hash")
    monkeypatch.setattr(runner, "_check_packaged_renderer_dsv4_cache_ui", lambda _root: True)
    monkeypatch.setattr(runner, "_check_packaged_renderer_max_thinking_tokens", lambda _root: True)
    monkeypatch.setattr(runner, "_check_packaged_epipe_closed_stream_guards", lambda _root: True)
    monkeypatch.setattr(runner, "_check_packaged_user_data_isolation_bootstrap", lambda _root: True)

    artifact = runner.build_artifact(tmp_path)

    assert "packaged_user_data_isolation_bootstrap" in artifact["checks"]


def test_packaged_epipe_guard_rejects_missing_wrapped_disconnect_recursion(tmp_path):
    app_asar = tmp_path / runner.PACKAGED_RENDERER_ASAR
    app_asar.parent.mkdir(parents=True)
    app_asar.write_bytes(
        b"responseWritable(res)\n"
        b"!anyRes.closed\n"
        b"requestWritable(req)\n"
        b"!anyReq.closed\n"
        b"function chatBackendRequestWritable(req)\n"
        b"function imageServerRequestWritable(req)\n"
        b"!req.closed\n"
        b"isExpectedChildProcessStreamDisconnectError\n"
        b"isExpectedImageServerDisconnectError\n"
        b"nestedErrors.some((nested) => isExpectedChildProcessStreamDisconnectError(nested))\n"
        b"nestedErrors.some((nested) => isExpectedImageServerDisconnectError(nested))\n"
    )

    assert runner._check_packaged_epipe_closed_stream_guards(tmp_path) is False


def test_packaged_epipe_guard_accepts_wrapped_disconnect_recursion(tmp_path):
    app_asar = tmp_path / runner.PACKAGED_RENDERER_ASAR
    app_asar.parent.mkdir(parents=True)
    app_asar.write_bytes(
        b"responseWritable(res)\n"
        b"!anyRes.closed\n"
        b"requestWritable(req)\n"
        b"!anyReq.closed\n"
        b"function chatBackendRequestWritable(req)\n"
        b"function imageServerRequestWritable(req)\n"
        b"!req.closed\n"
        b"isExpectedChildProcessStreamDisconnectError\n"
        b"isExpectedImageServerDisconnectError\n"
        b"wrappedDisconnects\n"
        b"reason\n"
        b"detail\n"
        b"wrappedDisconnects.some((nested) => isExpectedChildProcessStreamDisconnectError(nested))\n"
        b"nestedErrors.some((nested) => isExpectedChildProcessStreamDisconnectError(nested))\n"
        b"wrappedDisconnects.some((nested) => isExpectedImageServerDisconnectError(nested))\n"
        b"nestedErrors.some((nested) => isExpectedImageServerDisconnectError(nested))\n"
        b"function isExpectedCacheEndpointDisconnectError\n"
        b"function fetchCacheJson\n"
        b"Cache stats\n"
        b"connection lost. The model server may have stopped or restarted\n"
        b"wrappedDisconnects.some((nested) => isExpectedCacheEndpointDisconnectError(nested))\n"
        b"nestedErrors.some((nested) => isExpectedCacheEndpointDisconnectError(nested))\n"
        b"function isExpectedPerformanceEndpointDisconnectError\n"
        b"Performance health connection lost. The model server may have stopped or restarted\n"
        b"wrappedDisconnects.some((nested) => isExpectedPerformanceEndpointDisconnectError(nested))\n"
        b"nestedErrors.some((nested) => isExpectedPerformanceEndpointDisconnectError(nested))\n"
    )

    assert runner._check_packaged_epipe_closed_stream_guards(tmp_path) is True


def test_packaged_epipe_guard_rejects_missing_cache_endpoint_disconnect_guard(tmp_path):
    app_asar = tmp_path / runner.PACKAGED_RENDERER_ASAR
    app_asar.parent.mkdir(parents=True)
    app_asar.write_bytes(
        b"responseWritable(res)\n"
        b"!anyRes.closed\n"
        b"requestWritable(req)\n"
        b"!anyReq.closed\n"
        b"function chatBackendRequestWritable(req)\n"
        b"function imageServerRequestWritable(req)\n"
        b"!req.closed\n"
        b"isExpectedChildProcessStreamDisconnectError\n"
        b"isExpectedImageServerDisconnectError\n"
        b"wrappedDisconnects\n"
        b"reason\n"
        b"detail\n"
        b"wrappedDisconnects.some((nested) => isExpectedChildProcessStreamDisconnectError(nested))\n"
        b"nestedErrors.some((nested) => isExpectedChildProcessStreamDisconnectError(nested))\n"
        b"wrappedDisconnects.some((nested) => isExpectedImageServerDisconnectError(nested))\n"
        b"nestedErrors.some((nested) => isExpectedImageServerDisconnectError(nested))\n"
    )

    assert runner._check_packaged_epipe_closed_stream_guards(tmp_path) is False


def test_packaged_integrity_checks_packaged_python_has_no_pycache(monkeypatch, tmp_path):
    def fake_run(_root: Path, name: str, _cwd_rel: Path, _cmd: list[str]):
        if name == "release_gate_unit_contracts":
            return _result(name, 0, ["34 passed in 0.07s"], passed=runner.MIN_RELEASE_GATE_UNIT_TESTS)
        if name == "bundled_python_verifier":
            return _result(
                name,
                0,
                [
                    "  ok   bundled vmlx_engine version matches package.json",
                    "  ok   bundled critical vmlx_engine files match source content",
                    "  ok   bundled critical jang_tools files match source content",
                    "  ok   bundled-python console-script shebangs are relocatable",
                    "bundled-python: all critical imports ok",
                ],
            )
        if name == "release_gate_skip_app":
            return _result(name, 1, _expected_release_gate_failure_tail())
        raise AssertionError(name)

    app_python = (
        tmp_path
        / "panel/release/sequoia-app/mac-arm64/vMLX.app/Contents/Resources/bundled-python/python"
    )
    pycache = app_python / "lib/python3.12/json/__pycache__"
    pycache.mkdir(parents=True)
    (pycache / "decoder.cpython-312.pyc").write_bytes(b"pyc")

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(runner, "_sha256", lambda _path: "hash")
    monkeypatch.setattr(runner, "_check_packaged_renderer_dsv4_cache_ui", lambda _root: True)
    monkeypatch.setattr(runner, "_check_packaged_renderer_max_thinking_tokens", lambda _root: True)
    monkeypatch.setattr(runner, "_check_packaged_epipe_closed_stream_guards", lambda _root: True)
    monkeypatch.setattr(runner, "_check_packaged_user_data_isolation_bootstrap", lambda _root: True)

    artifact = runner.build_artifact(tmp_path)

    assert artifact["checks"]["packaged_python_has_no_pycache"] is False
    assert artifact["status"] == "fail"


def test_staged_app_engine_hash_parity_rejects_stale_packaged_runtime(tmp_path):
    assert "models/step3p7_mlx_vlm.py" in runner.STAGED_APP_ENGINE_HASH_FILES
    assert "models/gemma4_unified_register.py" in runner.STAGED_APP_ENGINE_HASH_FILES
    assert "models/gemma4_unified/__init__.py" in runner.STAGED_APP_ENGINE_HASH_FILES
    assert "models/gemma4_unified/config.py" in runner.STAGED_APP_ENGINE_HASH_FILES
    assert "models/gemma4_unified/gemma4_unified.py" in runner.STAGED_APP_ENGINE_HASH_FILES
    assert (
        "models/gemma4_unified/processing_gemma4_unified.py"
        in runner.STAGED_APP_ENGINE_HASH_FILES
    )
    assert "patches/mlx_vlm_mtp/qwen35_vl.py" in runner.STAGED_APP_ENGINE_HASH_FILES
    assert "cache_record_validator.py" in runner.STAGED_APP_ENGINE_HASH_FILES
    assert "native_mtp.py" in runner.STAGED_APP_ENGINE_HASH_FILES
    assert "patches/mlx_lm_mtp/__init__.py" in runner.STAGED_APP_ENGINE_HASH_FILES
    assert "patches/mlx_lm_mtp/batch_generator.py" in runner.STAGED_APP_ENGINE_HASH_FILES
    assert "patches/mlx_lm_mtp/cache_rollback.py" in runner.STAGED_APP_ENGINE_HASH_FILES
    assert "patches/mlx_lm_mtp/deepseek_v4_model.py" in runner.STAGED_APP_ENGINE_HASH_FILES
    assert "patches/mlx_lm_mtp/qwen35_model.py" in runner.STAGED_APP_ENGINE_HASH_FILES
    assert "tq_disk_store.py" in runner.STAGED_APP_ENGINE_HASH_FILES
    assert "utils/hybrid_tq_cache.py" in runner.STAGED_APP_ENGINE_HASH_FILES
    assert "utils/mlx_vlm_compat.py" in runner.STAGED_APP_ENGINE_HASH_FILES
    assert "runtime_patches/deepseek_v4_register.py" in runner.STAGED_APP_ENGINE_HASH_FILES
    assert "runtime_patches/gemma4_vision.py" in runner.STAGED_APP_ENGINE_HASH_FILES
    assert "runtime_patches/kimi_k25_mla.py" in runner.STAGED_APP_ENGINE_HASH_FILES
    assert _expected_tool_parser_hash_files().issubset(
        set(runner.STAGED_APP_ENGINE_HASH_FILES)
    )
    assert _expected_reasoning_parser_hash_files().issubset(
        set(runner.STAGED_APP_ENGINE_HASH_FILES)
    )

    source = tmp_path / "vmlx_engine/server.py"
    staged = (
        tmp_path
        / runner.PACKAGED_PYTHON_ROOT
        / "lib/python3.12/site-packages/vmlx_engine/server.py"
    )
    source.parent.mkdir(parents=True)
    staged.parent.mkdir(parents=True)
    source.write_text("current\n", encoding="utf-8")
    staged.write_text("stale\n", encoding="utf-8")

    assert runner._check_staged_app_engine_hash_parity(tmp_path) is False

    staged.write_text("current\n", encoding="utf-8")

    assert runner._check_staged_app_engine_hash_parity(tmp_path) is True


def _expected_tool_parser_hash_files() -> set[str]:
    return {
        str(path.relative_to("vmlx_engine"))
        for path in Path("vmlx_engine/tool_parsers").glob("*.py")
    }


def _expected_reasoning_parser_hash_files() -> set[str]:
    return {
        str(path.relative_to("vmlx_engine"))
        for path in Path("vmlx_engine/reasoning").glob("*.py")
    }


def test_staged_app_engine_hash_parity_rejects_stale_packaged_source_mirror(
    tmp_path,
):
    source = tmp_path / "vmlx_engine/server.py"
    staged_runtime = (
        tmp_path
        / runner.PACKAGED_PYTHON_ROOT
        / "lib/python3.12/site-packages/vmlx_engine/server.py"
    )
    staged_source = (
        tmp_path
        / runner.PACKAGED_APP
        / "Contents/Resources/vmlx-engine-source/vmlx_engine/server.py"
    )
    source.parent.mkdir(parents=True)
    staged_runtime.parent.mkdir(parents=True)
    staged_source.parent.mkdir(parents=True)
    source.write_text("current\n", encoding="utf-8")
    staged_runtime.write_text("current\n", encoding="utf-8")
    staged_source.write_text("stale\n", encoding="utf-8")

    assert runner._check_staged_app_engine_source_hash_parity(tmp_path) is False

    staged_source.write_text("current\n", encoding="utf-8")

    assert runner._check_staged_app_engine_source_hash_parity(tmp_path) is True


def test_packaged_user_data_isolation_check_rejects_missing_user_data_override(tmp_path):
    app_asar = tmp_path / runner.PACKAGED_RENDERER_ASAR
    app_asar.parent.mkdir(parents=True)
    app_asar.write_bytes(b"requestSingleInstanceLock\n")

    assert runner._check_packaged_user_data_isolation_bootstrap(tmp_path) is False


def test_packaged_user_data_isolation_check_accepts_early_user_data_override(tmp_path):
    app_asar = tmp_path / runner.PACKAGED_RENDERER_ASAR
    app_asar.parent.mkdir(parents=True)
    app_asar.write_bytes(
        b"--vmlx-user-data-dir\n"
        b"VMLX_USER_DATA_DIR\n"
        b"VMLINUX_USER_DATA_DIR\n"
        b"setPath(\"userData\"\n"
        b"requestSingleInstanceLock\n"
    )

    assert runner._check_packaged_user_data_isolation_bootstrap(tmp_path) is True


def test_packaged_python_pycache_check_rejects_sealed_resource_drift(tmp_path):
    pycache = (
        tmp_path
        / "panel/release/sequoia-app/mac-arm64/vMLX.app/Contents/Resources/bundled-python/python/lib/python3.12/encodings/__pycache__"
    )
    pycache.mkdir(parents=True)
    (pycache / "utf_8.cpython-312.pyc").write_bytes(b"pyc")

    assert runner._check_packaged_python_has_no_pycache(tmp_path) is False


def test_packaged_python_pycache_check_accepts_clean_bundle(tmp_path):
    python_root = (
        tmp_path
        / "panel/release/sequoia-app/mac-arm64/vMLX.app/Contents/Resources/bundled-python/python"
    )
    python_root.mkdir(parents=True)

    assert runner._check_packaged_python_has_no_pycache(tmp_path) is True


def test_packaged_integrity_source_hashes_cover_release_dmg_signing_script():
    assert "panel/scripts/build-release-dmgs.sh" in runner.SOURCE_HASH_FILES


def test_packaged_integrity_source_hashes_cover_release_dmg_verifier_script():
    assert "panel/scripts/verify-release-dmgs.sh" in runner.SOURCE_HASH_FILES


def test_packaged_integrity_source_hashes_cover_release_dmg_notary_script():
    assert "panel/scripts/notarize-release-dmgs.sh" in runner.SOURCE_HASH_FILES


def test_release_dmg_hardened_runtime_contract_rejects_weak_final_codesign(tmp_path):
    script = tmp_path / "panel/scripts/build-release-dmgs.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        'codesign --force --deep --sign "$identity" "$app_path"\n',
        encoding="utf-8",
    )

    assert runner._check_release_dmg_hardened_runtime_contract(tmp_path) is False


def test_release_dmg_hardened_runtime_contract_accepts_runtime_entitlements(tmp_path):
    script = tmp_path / "panel/scripts/build-release-dmgs.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        "finalize_release_app_signature() {\n"
        'local entitlements="$PANEL_DIR/build/entitlements.mac.plist"\n'
        'codesign --force --deep --options runtime --entitlements "$entitlements" --sign "$identity" "$app_path"\n'
        'codesign --verify --deep --strict --verbose=2 "$app_path"\n'
        "}\n"
        "find_staged_app() {\n"
        "}\n",
        encoding="utf-8",
    )
    entitlements = tmp_path / "panel/build/entitlements.mac.plist"
    entitlements.parent.mkdir(parents=True)
    entitlements.write_text("<plist/>", encoding="utf-8")

    assert runner._check_release_dmg_hardened_runtime_contract(tmp_path) is True


def test_release_dmg_notarization_verifier_contract_rejects_missing_script(tmp_path):
    assert runner._check_release_dmg_notarization_verifier_contract(tmp_path) is False


def test_release_dmg_notarization_verifier_contract_accepts_final_dmg_checks(tmp_path):
    script = tmp_path / "panel/scripts/verify-release-dmgs.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        "#!/usr/bin/env bash\n"
        "require_developer_id_signature() {\n"
        '  codesign -dv --verbose=4 "$1" 2>&1 | grep -F "Authority=Developer ID Application: ShieldStack LLC (55KGF2S5AY)"\n'
        '  codesign -dv --verbose=4 "$1" 2>&1 | grep -F "TeamIdentifier=55KGF2S5AY"\n'
        '  if codesign -dv --verbose=4 "$1" 2>&1 | grep -F "Signature=adhoc"; then exit 1; fi\n'
        "}\n"
        "for flavor in sequoia tahoe; do\n"
        '  dmg="release/vMLX-${VERSION}-${flavor}-arm64.dmg"\n'
        '  hdiutil verify "$dmg"\n'
        '  codesign --verify --verbose=2 "$dmg"\n'
        '  codesign -dv --verbose=4 "$dmg"\n'
        '  require_developer_id_signature "$dmg"\n'
        '  xcrun stapler validate "$dmg"\n'
        '  spctl --assess --type open --context context:primary-signature --verbose=4 "$dmg"\n'
        '  shasum -a 256 "$dmg"\n'
        "done\n",
        encoding="utf-8",
    )

    assert runner._check_release_dmg_notarization_verifier_contract(tmp_path) is True


def test_release_dmg_notarization_verifier_contract_rejects_no_developer_id_assertion(
    tmp_path,
):
    script = tmp_path / "panel/scripts/verify-release-dmgs.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        "#!/usr/bin/env bash\n"
        "for flavor in sequoia tahoe; do\n"
        '  dmg="release/vMLX-${VERSION}-${flavor}-arm64.dmg"\n'
        '  hdiutil verify "$dmg"\n'
        '  codesign --verify --verbose=2 "$dmg"\n'
        '  codesign -dv --verbose=4 "$dmg"\n'
        '  xcrun stapler validate "$dmg"\n'
        '  spctl --assess --type open --context context:primary-signature --verbose=4 "$dmg"\n'
        '  shasum -a 256 "$dmg"\n'
        "done\n",
        encoding="utf-8",
    )

    assert runner._check_release_dmg_notarization_verifier_contract(tmp_path) is False


def test_release_dmg_notarization_submit_contract_rejects_missing_script(tmp_path):
    assert runner._check_release_dmg_notarization_submit_contract(tmp_path) is False


def test_release_dmg_notarization_submit_contract_accepts_signed_dmg_workflow(tmp_path):
    script = tmp_path / "panel/scripts/notarize-release-dmgs.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        "#!/usr/bin/env bash\n"
        "NOTARY_PROFILE=\"${VMLINUX_NOTARY_KEYCHAIN_PROFILE:-vmlx-notary}\"\n"
        "NOTARY_KEYCHAIN=\"${VMLINUX_NOTARY_KEYCHAIN:-}\"\n"
        "notarytool_args=(--keychain-profile \"$NOTARY_PROFILE\")\n"
        'if [[ -n "$NOTARY_KEYCHAIN" ]]; then notarytool_args=(--keychain "$NOTARY_KEYCHAIN" "${notarytool_args[@]}"); fi\n'
        "regenerate_blockmap() {\n"
        '  ./node_modules/app-builder-bin/mac/app-builder_arm64 blockmap --input "$1" --output "$1.blockmap"\n'
        "}\n"
        "require_developer_id_signature() {\n"
        '  codesign -dv --verbose=4 "$1" 2>&1 | grep -F "Authority=Developer ID Application: ShieldStack LLC (55KGF2S5AY)"\n'
        '  codesign -dv --verbose=4 "$1" 2>&1 | grep -F "TeamIdentifier=55KGF2S5AY"\n'
        '  if codesign -dv --verbose=4 "$1" 2>&1 | grep -F "Signature=adhoc"; then exit 1; fi\n'
        "}\n"
        "for flavor in sequoia tahoe; do\n"
        '  dmg="release/vMLX-${VERSION}-${flavor}-arm64.dmg"\n'
        '  codesign --verify --verbose=2 "$dmg"\n'
        '  codesign -dv --verbose=4 "$dmg"\n'
        '  require_developer_id_signature "$dmg"\n'
        '  xcrun notarytool submit "$dmg" "${notarytool_args[@]}" --wait --output-format json\n'
        '  xcrun stapler staple "$dmg"\n'
        '  xcrun stapler validate "$dmg"\n'
        '  regenerate_blockmap "$dmg"\n'
        '  spctl --assess --type open --context context:primary-signature --verbose=4 "$dmg"\n'
        '  shasum -a 256 "$dmg"\n'
        "done\n",
        encoding="utf-8",
    )

    assert runner._check_release_dmg_notarization_submit_contract(tmp_path) is True


def test_release_dmg_notarization_submit_contract_rejects_default_keychain_only(
    tmp_path,
):
    script = tmp_path / "panel/scripts/notarize-release-dmgs.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        "#!/usr/bin/env bash\n"
        "NOTARY_PROFILE=\"${VMLINUX_NOTARY_KEYCHAIN_PROFILE:-vmlx-notary}\"\n"
        "require_developer_id_signature() {\n"
        '  codesign -dv --verbose=4 "$1" 2>&1 | grep -F "Authority=Developer ID Application: ShieldStack LLC (55KGF2S5AY)"\n'
        '  codesign -dv --verbose=4 "$1" 2>&1 | grep -F "TeamIdentifier=55KGF2S5AY"\n'
        '  if codesign -dv --verbose=4 "$1" 2>&1 | grep -F "Signature=adhoc"; then exit 1; fi\n'
        "}\n"
        "for flavor in sequoia tahoe; do\n"
        '  dmg="release/vMLX-${VERSION}-${flavor}-arm64.dmg"\n'
        '  codesign --verify --verbose=2 "$dmg"\n'
        '  codesign -dv --verbose=4 "$dmg"\n'
        '  require_developer_id_signature "$dmg"\n'
        '  xcrun notarytool submit "$dmg" --keychain-profile "$NOTARY_PROFILE" --wait --output-format json\n'
        '  xcrun stapler staple "$dmg"\n'
        '  xcrun stapler validate "$dmg"\n'
        '  spctl --assess --type open --context context:primary-signature --verbose=4 "$dmg"\n'
        '  shasum -a 256 "$dmg"\n'
        "done\n",
        encoding="utf-8",
    )

    assert runner._check_release_dmg_notarization_submit_contract(tmp_path) is False


def test_release_dmg_notarization_submit_contract_rejects_stale_prestaple_blockmaps(
    tmp_path,
):
    script = tmp_path / "panel/scripts/notarize-release-dmgs.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        "#!/usr/bin/env bash\n"
        "NOTARY_PROFILE=\"${VMLINUX_NOTARY_KEYCHAIN_PROFILE:-vmlx-notary}\"\n"
        "NOTARY_KEYCHAIN=\"${VMLINUX_NOTARY_KEYCHAIN:-}\"\n"
        "notarytool_args=(--keychain-profile \"$NOTARY_PROFILE\")\n"
        'if [[ -n "$NOTARY_KEYCHAIN" ]]; then notarytool_args=(--keychain "$NOTARY_KEYCHAIN" "${notarytool_args[@]}"); fi\n'
        "require_developer_id_signature() {\n"
        '  codesign -dv --verbose=4 "$1" 2>&1 | grep -F "Authority=Developer ID Application: ShieldStack LLC (55KGF2S5AY)"\n'
        '  codesign -dv --verbose=4 "$1" 2>&1 | grep -F "TeamIdentifier=55KGF2S5AY"\n'
        '  if codesign -dv --verbose=4 "$1" 2>&1 | grep -F "Signature=adhoc"; then exit 1; fi\n'
        "}\n"
        "for flavor in sequoia tahoe; do\n"
        '  dmg="release/vMLX-${VERSION}-${flavor}-arm64.dmg"\n'
        '  codesign --verify --verbose=2 "$dmg"\n'
        '  codesign -dv --verbose=4 "$dmg"\n'
        '  require_developer_id_signature "$dmg"\n'
        '  xcrun notarytool submit "$dmg" "${notarytool_args[@]}" --wait --output-format json\n'
        '  xcrun stapler staple "$dmg"\n'
        '  xcrun stapler validate "$dmg"\n'
        '  spctl --assess --type open --context context:primary-signature --verbose=4 "$dmg"\n'
        '  shasum -a 256 "$dmg"\n'
        "done\n",
        encoding="utf-8",
    )

    assert runner._check_release_dmg_notarization_submit_contract(tmp_path) is False


def test_release_dmg_notarization_submit_contract_rejects_no_developer_id_assertion(
    tmp_path,
):
    script = tmp_path / "panel/scripts/notarize-release-dmgs.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        "#!/usr/bin/env bash\n"
        "NOTARY_PROFILE=\"${VMLINUX_NOTARY_KEYCHAIN_PROFILE:-vmlx-notary}\"\n"
        "for flavor in sequoia tahoe; do\n"
        '  dmg="release/vMLX-${VERSION}-${flavor}-arm64.dmg"\n'
        '  codesign --verify --verbose=2 "$dmg"\n'
        '  xcrun notarytool submit "$dmg" --keychain-profile "$NOTARY_PROFILE" --wait --output-format json\n'
        '  xcrun stapler staple "$dmg"\n'
        '  xcrun stapler validate "$dmg"\n'
        '  spctl --assess --type open --context context:primary-signature --verbose=4 "$dmg"\n'
        '  shasum -a 256 "$dmg"\n'
        "done\n",
        encoding="utf-8",
    )

    assert runner._check_release_dmg_notarization_submit_contract(tmp_path) is False


def test_package_signing_preflight_records_missing_signed_app(tmp_path):
    preflight = runner._package_signing_preflight(tmp_path)

    assert preflight["status"] == "open"
    assert preflight["app_exists"] is False
    assert preflight["developer_id_signed"] is False
    assert preflight["developer_id_identity_count"] == 0
    assert preflight["simple_developer_id_sign_rc"] is None
    assert preflight["signing_blocker_reason"] == "packaged_app_missing"


def test_package_signing_preflight_records_keychain_private_key_failure(
    monkeypatch, tmp_path
):
    app = tmp_path / runner.PACKAGED_APP
    app.mkdir(parents=True)

    def fake_run(cmd, **_kwargs):
        command = " ".join(str(part) for part in cmd)
        if cmd[:3] == ["codesign", "-dv", "--verbose=4"]:
            return runner.subprocess.CompletedProcess(
                cmd,
                0,
                "Signature=adhoc\nTeamIdentifier=not set\n",
            )
        if cmd[:2] == ["codesign", "--verify"]:
            return runner.subprocess.CompletedProcess(
                cmd,
                1,
                "file modified: /tmp/vMLX.app/Contents/Resources/runtime.py\n",
            )
        if cmd[:3] == ["security", "find-identity", "-v"]:
            return runner.subprocess.CompletedProcess(
                cmd,
                0,
                '  1) D4DBBCB52F666D03F0A5154BFFEA2227BEE8FC7C "Developer ID Application: ShieldStack LLC (55KGF2S5AY)"\n',
            )
        if cmd[:2] == ["security", "show-keychain-info"]:
            return runner.subprocess.CompletedProcess(
                cmd,
                1,
                "security: SecKeychainCopySettings: User interaction is not allowed.\n",
            )
        if "codesign --force --sign" in command:
            return runner.subprocess.CompletedProcess(
                cmd,
                1,
                "probe: errSecInternalComponent\n",
            )
        raise AssertionError(command)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    preflight = runner._package_signing_preflight(tmp_path)

    assert preflight["status"] == "open"
    assert preflight["developer_id_signed"] is False
    assert preflight["signature_is_adhoc"] is True
    assert preflight["team_identifier"] == "not set"
    assert preflight["developer_id_identity_count"] == 1
    assert preflight["simple_developer_id_sign_rc"] == 1
    assert "errSecInternalComponent" in "\n".join(
        preflight["simple_developer_id_sign_tail"]
    )
    assert preflight["keychain_info_statuses"] == [
        {
            "keychain": str(Path("~/Library/Keychains/vmlx-build.keychain-db").expanduser()),
            "returncode": 1,
            "tail": [
                "security: SecKeychainCopySettings: User interaction is not allowed."
            ],
        },
        {
            "keychain": str(Path("~/Library/Keychains/build.keychain-db").expanduser()),
            "returncode": 1,
            "tail": [
                "security: SecKeychainCopySettings: User interaction is not allowed."
            ],
        },
        {
            "keychain": str(Path("~/Library/Keychains/login.keychain-db").expanduser()),
            "returncode": 1,
            "tail": [
                "security: SecKeychainCopySettings: User interaction is not allowed."
            ],
        },
    ]
    assert preflight["signing_blocker_reason"] == "developer_id_keychain_user_interaction_not_allowed"
    assert preflight["packaged_app_modified_after_signing"] is True
    assert preflight["modified_after_signing_file_count"] == 1
    assert preflight["missing_after_signing_file_count"] == 0
    assert preflight["modified_after_signing_tail"] == [
        "file modified: /tmp/vMLX.app/Contents/Resources/runtime.py"
    ]
    assert preflight["missing_after_signing_tail"] == []
    assert "packaged_app_modified_after_signing" in preflight["signing_blocker_reasons"]
    assert preflight["manual_remediation_required"] is True
    assert preflight["remediation_summary"] == (
        "Developer ID identities are visible, but codesign cannot use the private "
        "key from this non-interactive process."
    )
    assert preflight["remediation_steps"] == [
        "Unlock the signing keychain in an interactive macOS session.",
        "Grant codesign access to the Developer ID private key, for example with security set-key-partition-list using the keychain password outside Codex logs.",
        "Rebuild or reseal the packaged app after bundled runtime sync so codesign --verify --deep --strict passes before notarization.",
        "Rerun the packaged integrity contract and require package_signing_preflight.status=pass before notarization.",
    ]


def test_package_signing_preflight_classifies_codesign_user_interaction_failure(
    monkeypatch, tmp_path
):
    app = tmp_path / runner.PACKAGED_APP
    app.mkdir(parents=True)

    def fake_run(cmd, **_kwargs):
        command = " ".join(str(part) for part in cmd)
        if cmd[:3] == ["codesign", "-dv", "--verbose=4"]:
            return runner.subprocess.CompletedProcess(
                cmd,
                0,
                "Signature=adhoc\nTeamIdentifier=not set\n",
            )
        if cmd[:2] == ["codesign", "--verify"]:
            return runner.subprocess.CompletedProcess(cmd, 1, "adhoc verify failed\n")
        if cmd[:3] == ["security", "find-identity", "-v"]:
            return runner.subprocess.CompletedProcess(
                cmd,
                0,
                '  1) D4DBBCB52F666D03F0A5154BFFEA2227BEE8FC7C "Developer ID Application: ShieldStack LLC (55KGF2S5AY)"\n',
            )
        if cmd[:2] == ["security", "show-keychain-info"]:
            return runner.subprocess.CompletedProcess(cmd, 0, "no-timeout\n")
        if "codesign --force --sign" in command:
            return runner.subprocess.CompletedProcess(
                cmd,
                1,
                "codesign-probe: User interaction is not allowed.\n",
            )
        raise AssertionError(command)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    preflight = runner._package_signing_preflight(tmp_path)

    assert (
        preflight["signing_blocker_reason"]
        == "developer_id_keychain_user_interaction_not_allowed"
    )
    assert preflight["manual_remediation_required"] is True


def test_package_signing_preflight_rejects_signed_old_app_when_new_signing_unusable(
    monkeypatch, tmp_path
):
    app = tmp_path / runner.PACKAGED_APP
    app.mkdir(parents=True)

    def fake_run(cmd, **_kwargs):
        command = " ".join(str(part) for part in cmd)
        if cmd[:3] == ["codesign", "-dv", "--verbose=4"]:
            return runner.subprocess.CompletedProcess(
                cmd,
                0,
                "\n".join(
                    [
                        "CodeDirectory v=20500 size=472 flags=0x10000(runtime) hashes=4+7 location=embedded",
                        "Authority=Developer ID Application: ShieldStack LLC (55KGF2S5AY)",
                        "Authority=Developer ID Certification Authority",
                        "Authority=Apple Root CA",
                        "TeamIdentifier=55KGF2S5AY",
                    ]
                ),
            )
        if cmd[:2] == ["codesign", "--verify"]:
            return runner.subprocess.CompletedProcess(cmd, 0, "valid on disk\n")
        if cmd[:3] == ["security", "find-identity", "-v"]:
            return runner.subprocess.CompletedProcess(
                cmd,
                0,
                '  1) D4DBBCB52F666D03F0A5154BFFEA2227BEE8FC7C "Developer ID Application: ShieldStack LLC (55KGF2S5AY)"\n',
            )
        if cmd[:2] == ["security", "show-keychain-info"]:
            return runner.subprocess.CompletedProcess(
                cmd,
                36,
                "security: SecKeychainCopySettings: User interaction is not allowed.\n",
            )
        if "codesign --force --sign" in command:
            return runner.subprocess.CompletedProcess(
                cmd,
                1,
                "codesign-probe: errSecInternalComponent\n",
            )
        raise AssertionError(command)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    preflight = runner._package_signing_preflight(tmp_path)

    assert preflight["developer_id_signed"] is True
    assert preflight["hardened_runtime_enabled"] is True
    assert preflight["codesign_verify_rc"] == 0
    assert preflight["simple_developer_id_sign_rc"] == 1
    assert preflight["status"] == "open"
    assert (
        preflight["signing_blocker_reason"]
        == "developer_id_keychain_user_interaction_not_allowed"
    )
    assert preflight["manual_remediation_required"] is True


def test_package_signing_preflight_rejects_developer_id_without_hardened_runtime(
    monkeypatch, tmp_path
):
    app = tmp_path / runner.PACKAGED_APP
    app.mkdir(parents=True)

    def fake_run(cmd, **_kwargs):
        command = " ".join(str(part) for part in cmd)
        if cmd[:3] == ["codesign", "-dv", "--verbose=4"]:
            return runner.subprocess.CompletedProcess(
                cmd,
                0,
                "\n".join(
                    [
                        "CodeDirectory v=20500 size=325 flags=0x0(none) hashes=4+3 location=embedded",
                        "Authority=Developer ID Application: ShieldStack LLC (55KGF2S5AY)",
                        "Authority=Developer ID Certification Authority",
                        "Authority=Apple Root CA",
                        "TeamIdentifier=55KGF2S5AY",
                    ]
                ),
            )
        if cmd[:2] == ["codesign", "--verify"]:
            return runner.subprocess.CompletedProcess(cmd, 0, "valid on disk\n")
        if cmd[:3] == ["security", "find-identity", "-v"]:
            return runner.subprocess.CompletedProcess(
                cmd,
                0,
                '  1) D4DBBCB52F666D03F0A5154BFFEA2227BEE8FC7C "Developer ID Application: ShieldStack LLC (55KGF2S5AY)"\n',
            )
        if cmd[:2] == ["security", "show-keychain-info"]:
            return runner.subprocess.CompletedProcess(cmd, 0, "no-timeout\n")
        if "codesign --force --sign" in command:
            return runner.subprocess.CompletedProcess(cmd, 0, "probe signed\n")
        raise AssertionError(command)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    preflight = runner._package_signing_preflight(tmp_path)

    assert preflight["status"] == "open"
    assert preflight["developer_id_signed"] is True
    assert preflight["hardened_runtime_enabled"] is False
    assert (
        preflight["signing_blocker_reason"]
        == "packaged_app_missing_hardened_runtime"
    )


def test_package_signing_preflight_detects_runtime_flag_with_adhoc_signature(
    monkeypatch, tmp_path
):
    app = tmp_path / runner.PACKAGED_APP
    app.mkdir(parents=True)

    def fake_run(cmd, **_kwargs):
        command = " ".join(str(part) for part in cmd)
        if cmd[:3] == ["codesign", "-dv", "--verbose=4"]:
            return runner.subprocess.CompletedProcess(
                cmd,
                0,
                "\n".join(
                    [
                        "CodeDirectory v=20500 size=461 flags=0x10002(adhoc,runtime) hashes=4+7 location=embedded",
                        "Signature=adhoc",
                        "TeamIdentifier=not set",
                    ]
                ),
            )
        if cmd[:2] == ["codesign", "--verify"]:
            return runner.subprocess.CompletedProcess(cmd, 0, "valid on disk\n")
        if cmd[:3] == ["security", "find-identity", "-v"]:
            return runner.subprocess.CompletedProcess(
                cmd,
                0,
                '  1) D4DBBCB52F666D03F0A5154BFFEA2227BEE8FC7C "Developer ID Application: ShieldStack LLC (55KGF2S5AY)"\n',
            )
        if cmd[:2] == ["security", "show-keychain-info"]:
            return runner.subprocess.CompletedProcess(cmd, 0, "no-timeout\n")
        if "codesign --force --sign" in command:
            return runner.subprocess.CompletedProcess(cmd, 0, "probe signed\n")
        raise AssertionError(command)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    preflight = runner._package_signing_preflight(tmp_path)

    assert preflight["signature_is_adhoc"] is True
    assert preflight["hardened_runtime_enabled"] is True
    assert (
        preflight["signing_blocker_reason"]
        == "packaged_app_not_developer_id_signed"
    )


def test_packaged_integrity_fails_when_signing_blocker_is_open(
    monkeypatch, tmp_path
):
    def fake_run(_root: Path, name: str, _cwd_rel: Path, _cmd: list[str]):
        if name == "release_gate_unit_contracts":
            return _result(name, 0, ["34 passed in 0.07s"], passed=runner.MIN_RELEASE_GATE_UNIT_TESTS)
        if name == "bundled_python_verifier":
            return _result(
                name,
                0,
                [
                    "  ok   bundled vmlx_engine version matches package.json",
                    "  ok   bundled critical vmlx_engine files match source content",
                    "  ok   bundled critical jang_tools files match source content",
                    "  ok   bundled-python console-script shebangs are relocatable",
                    "bundled-python: all critical imports ok",
                ],
            )
        if name == "release_gate_skip_app":
            return _result(name, 1, _expected_release_gate_failure_tail())
        raise AssertionError(name)

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(runner, "_sha256", lambda _path: "hash")
    monkeypatch.setattr(runner, "_check_packaged_renderer_dsv4_cache_ui", lambda _root: True)
    monkeypatch.setattr(runner, "_check_packaged_renderer_max_thinking_tokens", lambda _root: True)
    monkeypatch.setattr(runner, "_check_packaged_epipe_closed_stream_guards", lambda _root: True)
    monkeypatch.setattr(runner, "_check_packaged_user_data_isolation_bootstrap", lambda _root: True)
    monkeypatch.setattr(runner, "_check_packaged_python_has_no_pycache", lambda _root: True)
    monkeypatch.setattr(runner, "_check_staged_app_engine_hash_parity", lambda _root: True)
    monkeypatch.setattr(runner, "_check_staged_app_engine_source_hash_parity", lambda _root: True)
    monkeypatch.setattr(runner, "_check_release_dmg_hardened_runtime_contract", lambda _root: True)
    monkeypatch.setattr(runner, "_check_release_dmg_notarization_verifier_contract", lambda _root: True)
    monkeypatch.setattr(runner, "_check_release_dmg_notarization_submit_contract", lambda _root: True)
    monkeypatch.setattr(runner, "dry_release_gate_used_current_objective_digest", lambda _step: True)

    artifact = runner.build_artifact(tmp_path)

    assert artifact["status"] == "fail"
    assert artifact["failed"] == ["packaged_app_developer_id_signing_blocked"]
    assert artifact["package_signing_preflight"]["status"] == "open"
    assert artifact["release_blockers"] == [
        {
            "id": "packaged_app_developer_id_signing_blocked",
            "status": "open",
            "evidence": "package_signing_preflight",
            "next_proof": "Build and verify a Developer ID signed vMLX.app before notarization.",
        }
    ]


def test_packaged_renderer_max_thinking_tokens_check_rejects_missing_request_wiring(tmp_path):
    app_asar = tmp_path / runner.PACKAGED_RENDERER_ASAR
    app_asar.parent.mkdir(parents=True)
    app_asar.write_bytes(b"Max Thinking Tokens\nmaxThinkingTokens\n")

    assert runner._check_packaged_renderer_max_thinking_tokens(tmp_path) is False


def test_packaged_renderer_max_thinking_tokens_check_accepts_ui_and_request_wiring(tmp_path):
    app_asar = tmp_path / runner.PACKAGED_RENDERER_ASAR
    app_asar.parent.mkdir(parents=True)
    app_asar.write_bytes(
        b"Max Thinking Tokens\n"
        b"maxThinkingTokens\n"
        b"max_thinking_tokens\n"
        b"thinking_budget\n"
    )

    assert runner._check_packaged_renderer_max_thinking_tokens(tmp_path) is True


def test_packaged_epipe_closed_stream_guard_check_rejects_stale_asar(tmp_path):
    app_asar = tmp_path / runner.PACKAGED_RENDERER_ASAR
    app_asar.parent.mkdir(parents=True)
    app_asar.write_bytes(
        b"responseWritable(res){return !anyRes.destroyed && "
        b"!anyRes.writableEnded && !anyRes.writableDestroyed}"
        b"requestWritable(req){return !anyReq.destroyed && "
        b"!anyReq.writableEnded && !anyReq.writableDestroyed}"
        b"function chatBackendRequestWritable(req){return !anyReq.destroyed && "
        b"!anyReq.writableEnded && !anyReq.writableDestroyed}"
        b"function imageServerRequestWritable(req){return !req.destroyed && "
        b"!req.writableEnded && !req.writableDestroyed}"
    )

    assert runner._check_packaged_epipe_closed_stream_guards(tmp_path) is False


def test_packaged_epipe_closed_stream_guard_check_accepts_current_asar(tmp_path):
    app_asar = tmp_path / runner.PACKAGED_RENDERER_ASAR
    app_asar.parent.mkdir(parents=True)
    app_asar.write_bytes(
        b"responseWritable(res){return !anyRes.closed && "
        b"!anyRes.destroyed && !anyRes.writableEnded && !anyRes.writableDestroyed}"
        b"requestWritable(req){return !anyReq.closed && "
        b"!anyReq.destroyed && !anyReq.writableEnded && !anyReq.writableDestroyed}"
        b"function chatBackendRequestWritable(req){return !anyReq.closed && "
        b"!anyReq.destroyed && !anyReq.writableEnded && !anyReq.writableDestroyed}"
        b"function imageServerRequestWritable(req){return !req.closed && "
        b"!req.destroyed && !req.writableEnded && !req.writableDestroyed}"
        b"isExpectedChildProcessStreamDisconnectError\n"
        b"isExpectedImageServerDisconnectError\n"
        b"wrappedDisconnects\n"
        b"reason\n"
        b"detail\n"
        b"wrappedDisconnects.some((nested) => isExpectedChildProcessStreamDisconnectError(nested))\n"
        b"nestedErrors.some((nested) => isExpectedChildProcessStreamDisconnectError(nested))\n"
        b"wrappedDisconnects.some((nested) => isExpectedImageServerDisconnectError(nested))\n"
        b"nestedErrors.some((nested) => isExpectedImageServerDisconnectError(nested))\n"
        b"function isExpectedCacheEndpointDisconnectError\n"
        b"function fetchCacheJson\n"
        b"Cache stats\n"
        b"connection lost. The model server may have stopped or restarted\n"
        b"wrappedDisconnects.some((nested) => isExpectedCacheEndpointDisconnectError(nested))\n"
        b"nestedErrors.some((nested) => isExpectedCacheEndpointDisconnectError(nested))\n"
        b"function isExpectedPerformanceEndpointDisconnectError\n"
        b"Performance health connection lost. The model server may have stopped or restarted\n"
        b"wrappedDisconnects.some((nested) => isExpectedPerformanceEndpointDisconnectError(nested))\n"
        b"nestedErrors.some((nested) => isExpectedPerformanceEndpointDisconnectError(nested))\n"
    )

    assert runner._check_packaged_epipe_closed_stream_guards(tmp_path) is True


def test_packaged_renderer_dsv4_cache_ui_check_rejects_stale_duplicate_labels(tmp_path):
    app_asar = tmp_path / runner.PACKAGED_RENDERER_ASAR
    app_asar.parent.mkdir(parents=True)
    app_asar.write_bytes(
        b"DSV4 Native Cache\n"
        b"DSV4 Composite Prefix Cache\n"
        b"DSV4 Pool Quantization\n"
        b"DSV4 Flash composite prefix cache is disabled by default\n"
    )

    assert runner._check_packaged_renderer_dsv4_cache_ui(tmp_path) is False


def test_packaged_renderer_dsv4_cache_ui_check_accepts_deduped_labels(tmp_path):
    app_asar = tmp_path / runner.PACKAGED_RENDERER_ASAR
    app_asar.parent.mkdir(parents=True)
    app_asar.write_bytes(
        b"DSV4 Native Composite Prefix Cache\n"
        b"DSV4 CSA/HCA Pool Codec\n"
        b"DSV4_POOL_QUANT=1 native CSA/HCA pool codec\n"
    )

    assert runner._check_packaged_renderer_dsv4_cache_ui(tmp_path) is True
