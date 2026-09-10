from __future__ import annotations

import hashlib
import json
import os
import plistlib
import shutil
import stat
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from tests.cross_matrix import run_current_regression_suite as suite
from tests.cross_matrix import run_packaged_integrity_contract as runner


def test_packaged_integrity_known_open_rows_match_current_suite():
    assert runner.EXPECTED_OPEN_REQUIREMENTS == suite.EXPECTED_OPEN_REQUIREMENTS


def test_packaged_integrity_default_out_tracks_current_release_proof_artifact():
    assert Path(
        "build/current-packaged-integrity-contract-after-bundled-python-sync-20260608.json"
    ) == runner.DEFAULT_OUT


def test_packaged_integrity_release_version_is_explicit_and_valid(monkeypatch):
    monkeypatch.delenv("VMLX_R20_EXPECTED_VERSION", raising=False)
    assert runner._configured_r20_artifact_chain_version() == "1.6.20"

    monkeypatch.setenv("VMLX_R20_EXPECTED_VERSION", "1.6.41")
    assert runner._configured_r20_artifact_chain_version() == "1.6.41"

    monkeypatch.setenv("VMLX_R20_EXPECTED_VERSION", "not-a-release")
    with pytest.raises(RuntimeError, match="invalid vMLX production version"):
        runner._configured_r20_artifact_chain_version()


def test_packaged_integrity_hashes_cache_ipc_guard_source():
    assert "panel/src/main/ipc/cache.ts" in runner.SOURCE_HASH_FILES
    assert "tests/test_packaged_integrity_contract.py" in runner.SOURCE_HASH_FILES
    assert "panel/scripts/electron-builder-before-pack.cjs" in runner.SOURCE_HASH_FILES
    assert "panel/tests/release-packaging.test.ts" in runner.SOURCE_HASH_FILES


def test_tree_parity_and_payload_records_accept_empty_regular_files(tmp_path):
    source = tmp_path / "source"
    packaged = tmp_path / "packaged"
    source.mkdir()
    packaged.mkdir()
    (source / "__init__.py").write_bytes(b"")
    (packaged / "__init__.py").write_bytes(b"")

    parity = runner._require_tree_parity(
        source,
        packaged,
        label="empty package marker",
    )
    payload = runner._tree_payload_records(
        packaged,
        label="empty package payload",
    )

    assert parity["file_count"] == 1
    assert payload["entries"]["__init__.py"]["size"] == 0
    assert payload["entries"]["__init__.py"]["sha256"] == hashlib.sha256(b"").hexdigest()


def test_tree_parity_excludes_only_named_source_only_files(tmp_path):
    source = tmp_path / "source"
    packaged = tmp_path / "packaged"
    source.mkdir()
    packaged.mkdir()
    (source / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    (packaged / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "diagnostic.txt").write_text("source only\n", encoding="utf-8")

    parity = runner._require_tree_parity(
        source,
        packaged,
        label="runtime package",
        source_only_files=("diagnostic.txt",),
    )
    assert parity["file_count"] == 1

    (source / "unexpected.txt").write_text("not allowlisted\n", encoding="utf-8")
    with pytest.raises(runner.ArtifactChainError, match="tree differs"):
        runner._require_tree_parity(
            source,
            packaged,
            label="runtime package",
            source_only_files=("diagnostic.txt",),
        )

    (source / "unexpected.txt").unlink()
    (packaged / "diagnostic.txt").write_text("source only\n", encoding="utf-8")
    with pytest.raises(runner.ArtifactChainError, match="contains source-only files"):
        runner._require_tree_parity(
            source,
            packaged,
            label="runtime package",
            source_only_files=("diagnostic.txt",),
        )


def test_runtime_tree_parity_applies_source_only_contract_at_every_stage(tmp_path):
    source = tmp_path / "source"
    packaged = tmp_path / "packaged"
    diagnostic = Path(runner.R20_RUNTIME_SOURCE_ONLY_FILES[0])
    source.mkdir()
    packaged.mkdir()
    (source / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    (packaged / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / diagnostic).parent.mkdir(parents=True)
    (source / diagnostic).write_text("source only\n", encoding="utf-8")

    parity = runner._require_runtime_tree_parity(
        source,
        packaged,
        label="mounted bundled runtime",
    )
    assert parity["file_count"] == 1

    (packaged / diagnostic).parent.mkdir(parents=True)
    (packaged / diagnostic).write_text("must not ship\n", encoding="utf-8")
    with pytest.raises(runner.ArtifactChainError, match="contains source-only files"):
        runner._require_runtime_tree_parity(
            source,
            packaged,
            label="mounted bundled runtime",
        )


def test_extracted_asar_payload_ignores_only_ambient_finder_metadata(tmp_path):
    extracted = tmp_path / "extracted"
    nested = extracted / "node_modules"
    nested.mkdir(parents=True)
    (nested / "module.js").write_text("export default 1;\n", encoding="utf-8")
    (nested / ".DS_Store").write_bytes(b"ambient Finder metadata")
    (nested / ".other-hidden").write_text("covered\n", encoding="utf-8")

    strict = runner._tree_payload_records(extracted, label="strict ASAR")
    filtered = runner._tree_payload_records(
        extracted,
        label="Finder-safe ASAR",
        ignore_ambient_finder_metadata=True,
    )

    assert "node_modules/.DS_Store" in strict["entries"]
    assert "node_modules/.DS_Store" not in filtered["entries"]
    assert "node_modules/.other-hidden" in filtered["entries"]
    assert "node_modules/module.js" in filtered["entries"]


def test_wheel_manifest_covers_runtime_diagnostics_and_native_mtp_docs():
    pyproject = tomllib.loads((runner.REPO_ROOT / "pyproject.toml").read_text())
    package_data = pyproject["tool"]["setuptools"]["package-data"]["vmlx_engine"]

    assert "native_mtp_examples/*.md" in package_data
    assert "tests/*.py" in package_data


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
    """The line the REAL release gate prints when objectives are open.

    Deliberately does NOT subtract DEFERRED_RELEASE_OPEN_REQUIREMENTS. The gate
    names every open objective; it knows nothing about the suite's deferral
    list. This helper used to apply that subtraction, which made it agree with
    the production code's equally-wrong prediction and disagree with reality —
    so the reconciliation bug was invisible to these tests.
    """
    return "[FAIL] objective proof digest: " + "; ".join(
        runner.EXPECTED_OPEN_REQUIREMENTS
    )


def _current_objective_open_digest_line(root=None) -> str:
    open_requirements = runner.current_objective_open_requirements(root or Path("."))
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


def test_packaged_integrity_accepts_current_objective_digest_only_failure(tmp_path):
    artifact = tmp_path / runner.CURRENT_OBJECTIVE_DIGEST_ARTIFACT
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "requirements": [
                    {
                        "requirement": runner.EXPECTED_OPEN_REQUIREMENTS[0],
                        "status": "open",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    step = _result(
        "release_gate_skip_app",
        1,
        [_current_objective_open_digest_line(tmp_path)],
    )

    assert runner.release_gate_failure_is_expected(step, tmp_path)


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


def test_packaged_integrity_rejects_a_new_unknown_open_objective(tmp_path):
    """The reconciliation must not launder an objective nobody has signed off.

    This is the property that makes membership-based judging safe: a gate
    failure naming something outside EXPECTED_OPEN_REQUIREMENTS is NOT an
    expected failure, however many known objectives sit beside it.
    """
    step = _result(
        "release_gate_skip_app",
        1,
        [
            "[FAIL] objective proof digest: "
            + "; ".join(
                [
                    *runner.EXPECTED_OPEN_REQUIREMENTS,
                    "Brand new objective nobody has proven",
                ]
            )
        ],
    )

    assert runner.release_gate_failure_is_expected(step, tmp_path) is False


def test_packaged_integrity_rejects_a_non_objective_first_failure(tmp_path):
    step = _result(
        "release_gate_skip_app",
        1,
        ["[FAIL] panel typecheck: 2 errors"],
    )

    assert runner.release_gate_failure_is_expected(step, tmp_path) is False


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
    monkeypatch.setattr(runner, "dry_release_gate_used_current_objective_digest", lambda _step: True)
    monkeypatch.setattr(runner, "_package_signing_preflight", lambda _root: _signed_preflight())
    # 2026-08-16 (ledger row 151): the hardened-runtime entitlements check is
    # now emitted by build_artifact; stub it like the other packaged checks.
    monkeypatch.setattr(
        runner, "_check_release_dmg_hardened_runtime_contract", lambda _root: True
    )

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
    monkeypatch.setattr(runner, "dry_release_gate_used_current_objective_digest", lambda _step: True)
    monkeypatch.setattr(runner, "_package_signing_preflight", lambda _root: _signed_preflight())
    # 2026-08-16 (ledger row 151): the hardened-runtime entitlements check is
    # now emitted by build_artifact; stub it like the other packaged checks.
    monkeypatch.setattr(
        runner, "_check_release_dmg_hardened_runtime_contract", lambda _root: True
    )

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
    assert "models/llm.py" in runner.STAGED_APP_ENGINE_HASH_FILES
    assert "speculative.py" in runner.STAGED_APP_ENGINE_HASH_FILES
    assert "utils/nanbeige_runtime.py" in runner.STAGED_APP_ENGINE_HASH_FILES
    assert "patches/mlx_vlm_mtp/qwen35_vl.py" in runner.STAGED_APP_ENGINE_HASH_FILES
    assert "utils/mlx_vlm_compat.py" in runner.STAGED_APP_ENGINE_HASH_FILES

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
    # 2026-08-16 (ledger row 151): the contract now matches the pinned
    # $APPLE_CODESIGN invocation form (unshadowed-tool hardening, --timestamp).
    script = tmp_path / "panel/scripts/build-release-dmgs.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        "finalize_release_app_signature() {\n"
        'local entitlements="$PANEL_DIR/build/entitlements.mac.plist"\n'
        '"$APPLE_CODESIGN" --force --deep --timestamp --options runtime --entitlements "$entitlements" --sign "$identity" "$app_path"\n'
        '"$APPLE_CODESIGN" --verify --deep --strict --verbose=2 "$app_path"\n'
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
        b"DSV4 Native Composite Prefix Cache\n"
        b"DSV4 Native Cache\n"
        b"DSV4 Composite Prefix Cache\n"
        b"DSV4 Pool Quantization\n"
        b"DSV4 CSA/HCA Pool Codec\n"
    )

    assert runner._check_packaged_renderer_dsv4_cache_ui(tmp_path) is False


def test_packaged_renderer_dsv4_cache_ui_check_accepts_deduped_labels(tmp_path):
    app_asar = tmp_path / runner.PACKAGED_RENDERER_ASAR
    app_asar.parent.mkdir(parents=True)
    # 2026-08-16 (ledger row 159): fixture follows the repointed markers —
    # the old sentences exist in neither the packaged app nor panel source.
    app_asar.write_bytes(
        b"\n".join(runner.PACKAGED_RENDERER_REQUIRED_DSV4_CACHE_UI_STRINGS) + b"\n"
    )

    assert runner._check_packaged_renderer_dsv4_cache_ui(tmp_path) is True

    # ... and the de-duplication assertion still bites: a renderer that ALSO
    # carries an old duplicated DSV4 cache heading must fail.
    app_asar.write_bytes(
        b"\n".join(runner.PACKAGED_RENDERER_REQUIRED_DSV4_CACHE_UI_STRINGS)
        + b"\n"
        + runner.PACKAGED_RENDERER_FORBIDDEN_DSV4_CACHE_UI_STRINGS[0]
        + b"\n"
    )
    assert runner._check_packaged_renderer_dsv4_cache_ui(tmp_path) is False


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _write_r20_runtime_fixture(
    bundle_root: Path,
    *,
    flavor: str,
    source_commit: str,
) -> None:
    contract = runner.R20_FLAVOR_RUNTIME_CONTRACTS[flavor]
    site_packages = bundle_root / "python/lib/python3.12/site-packages"
    tags = {
        "mlx": f"cp312-cp312-{contract['mlx_wheel_platform']}",
        "mlx-metal": f"py3-none-{contract['mlx_wheel_platform']}",
    }
    for distribution, normalized in (
        ("mlx", "mlx"),
        ("mlx-metal", "mlx_metal"),
    ):
        dist_info = site_packages / f"{normalized}-0.31.2.dist-info"
        dist_info.mkdir(parents=True)
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.1\n"
            f"Name: {distribution}\n"
            "Version: 0.31.2\n",
            encoding="utf-8",
        )
        (dist_info / "WHEEL").write_text(
            "Wheel-Version: 1.0\n"
            "Generator: vmlx-r20-fixture\n"
            "Root-Is-Purelib: false\n"
            f"Tag: {tags[distribution]}\n",
            encoding="utf-8",
        )
    (bundle_root / "vmlx-bundle-provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "vmlx": {"commit": source_commit, "version": "1.6.20"},
                "jang": {"commit": "f" * 40, "version": "2.5.36"},
                "mlx_wheel_platform": contract["mlx_wheel_platform"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _r20_artifact_chain_fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    _git(root, "init")
    _git(root, "config", "user.name", "Artifact Chain Test")
    _git(root, "config", "user.email", "artifact-chain@example.invalid")
    (root / ".gitignore").write_text("build/\ndist/\n", encoding="utf-8")
    (root / "source.txt").write_text("release source\n", encoding="utf-8")
    source_engine = root / "vmlx_engine"
    panel_out = root / "panel/out"
    source_engine.mkdir()
    panel_out.mkdir(parents=True)
    (source_engine / "__init__.py").write_text('VERSION = "test"\n', encoding="utf-8")
    (panel_out / "main.js").write_text("console.log('r20')\n", encoding="utf-8")
    (root / "panel/package.json").write_text(
        '{"name":"vmlx","version":"1.6.20","main":"out/main.js","type":"module"}\n',
        encoding="utf-8",
    )
    _git(root, "add", ".gitignore", "source.txt", "vmlx_engine", "panel")
    _git(root, "commit", "-m", "fixture")
    source_commit = _git(root, "rev-parse", "HEAD")

    dist = root / "dist"
    dist.mkdir()
    extracted_root = tmp_path / "extracted-asars"
    staged_outputs: dict[str, Path] = {}
    extracted_asars: dict[str, Path] = {}
    for flavor in runner.R20_ARTIFACT_CHAIN_FLAVORS:
        stem = f"vMLX-1.6.20-{flavor}-arm64.dmg"
        (dist / stem).write_bytes(f"{flavor}-dmg-before-notary".encode())
        (dist / f"{stem}.blockmap").write_bytes(
            f"{flavor}-blockmap-before-notary".encode()
        )
        staged_output = dist / f"{flavor}-app"
        app = staged_output / "mac-arm64/vMLX.app"
        resources = app / "Contents/Resources"
        mirror = resources / "vmlx-engine-source/vmlx_engine"
        bundled = (
            resources
            / "bundled-python/python/lib/python3.12/site-packages/vmlx_engine"
        )
        extracted = extracted_root / flavor
        for directory in (mirror, bundled, extracted / "out"):
            directory.mkdir(parents=True)
        for packaged in (mirror / "__init__.py", bundled / "__init__.py"):
            packaged.write_text('VERSION = "test"\n', encoding="utf-8")
        (extracted / "out/main.js").write_text(
            "console.log('r20')\n",
            encoding="utf-8",
        )
        (extracted / "package.json").write_text(
            '{"name":"vmlx","version":"1.6.20","main":"out/main.js","type":"module"}\n',
            encoding="utf-8",
        )
        _write_r20_runtime_fixture(
            resources / "bundled-python",
            flavor=flavor,
            source_commit=source_commit,
        )
        (resources / "app.asar").write_bytes(
            f"fixture renderer source commit {source_commit}\n".encode()
        )
        electron = app / "Contents/MacOS/vMLX"
        electron.parent.mkdir(parents=True, exist_ok=True)
        electron.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        electron.chmod(0o755)
        python312 = resources / "bundled-python/python/bin/python3.12"
        python312.parent.mkdir(parents=True, exist_ok=True)
        python312.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        python312.chmod(0o755)
        (python312.parent / "python3").symlink_to("python3.12")
        helper = (
            app
            / "Contents/Frameworks/vMLX Helper.app/Contents/MacOS/vMLX Helper"
        )
        helper.parent.mkdir(parents=True)
        helper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        helper.chmod(0o755)
        info = app / "Contents/Info.plist"
        info.parent.mkdir(parents=True, exist_ok=True)
        with info.open("wb") as handle:
            plistlib.dump(
                {
                    "CFBundleIdentifier": "net.vmlx.app",
                    "CFBundleShortVersionString": "1.6.20",
                    "CFBundleVersion": "1.6.20",
                    "LSMinimumSystemVersion": runner.R20_FLAVOR_RUNTIME_CONTRACTS[
                        flavor
                    ]["minimum_system_version"],
                },
                handle,
            )
        staged_outputs[flavor] = staged_output
        extracted_asars[flavor] = extracted
    build = root / "build"
    build.mkdir()
    preflight = build / "r20-preflight.json"
    preflight.write_text('{"status":"pass"}\n', encoding="utf-8")
    return {
        "root": root,
        "dist": dist,
        "preflight": preflight,
        "private_root": tmp_path / "private-evidence",
        # The pytest process is the fixture's build driver.  Do not use its
        # ambient parent: detached full-suite runners can legitimately reparent
        # pytest to launchd (PID 1) before these late-running tests execute.
        "driver_pid": os.getpid(),
        "pre_manifest": tmp_path
        / "private-evidence"
        / "handoffs"
        / "r20-release-artifact-chain-pre-notary.json",
        "build_attestation": tmp_path
        / "private-evidence"
        / "handoffs"
        / "r20-build-driver.json",
        "staged_outputs": staged_outputs,
        "extracted_asars": extracted_asars,
        "hook_attestations": {
            flavor: tmp_path
            / "private-evidence"
            / "hook-completions"
            / f"{flavor}.completion.json"
            for flavor in runner.R20_ARTIFACT_CHAIN_FLAVORS
        },
        "bundle_attestations": {
            flavor: tmp_path
            / "private-evidence"
            / "hook-completions"
            / f"{flavor}.bundle-runtime.json"
            for flavor in runner.R20_ARTIFACT_CHAIN_FLAVORS
        },
        "dmg_parity_attestations": {
            flavor: tmp_path
            / "private-evidence"
            / "hook-completions"
            / f"{flavor}.dmg-parity.json"
            for flavor in runner.R20_ARTIFACT_CHAIN_FLAVORS
        },
    }


def _prepare_installed_manifest_fixture(
    paths: dict[str, Path],
    *,
    flavor: str = "sequoia",
) -> Path:
    return runner.find_exact_staged_app(paths["staged_outputs"][flavor])


def _sealed_installed_manifest_fixture(
    tmp_path: Path,
    *,
    flavor: str = "sequoia",
) -> tuple[dict[str, Path], Path, Path, dict[str, object]]:
    paths = _r20_artifact_chain_fixture(tmp_path)
    return _seal_installed_manifest_paths(paths, flavor=flavor)


def _seal_installed_manifest_paths(
    paths: dict[str, Path],
    *,
    flavor: str = "sequoia",
) -> tuple[dict[str, Path], Path, Path, dict[str, object]]:
    pre = _write_pre_manifest(paths)
    snapshots = _create_snapshots(paths, pre)
    submission_ids = _submission_ids()
    for current_flavor in runner.R20_ARTIFACT_CHAIN_FLAVORS:
        stem = f"vMLX-1.6.20-{current_flavor}-arm64.dmg"
        (paths["dist"] / stem).write_bytes(
            f"{current_flavor}-stapled-dmg".encode()
        )
        (paths["dist"] / f"{stem}.blockmap").write_bytes(
            f"{current_flavor}-post-staple-blockmap".encode()
        )
    snapshot_records = snapshots["snapshots"]
    assert isinstance(snapshot_records, dict)
    snapshot_paths = {
        current_flavor: Path(str(snapshot_records[current_flavor]["dmg_path"]))
        for current_flavor in runner.R20_ARTIFACT_CHAIN_FLAVORS
    }
    final_manifest = (
        paths["private_root"] / "handoffs/r20-post-notary-manifest.json"
    )
    final = _write_final(
        paths,
        pre,
        submission_ids,
        snapshot_paths,
        final_manifest,
    )
    handoff = _pre_handoff(paths, pre)
    app = _prepare_installed_manifest_fixture(paths, flavor=flavor)
    arguments: dict[str, object] = {
        "root": paths["root"],
        "app": app,
        "dist_dir": paths["dist"],
        "version": "1.6.20",
        "flavor": flavor,
        "private_root": paths["private_root"],
        "final_manifest_path": final_manifest,
        "expected_final_manifest_sha256": str(final["sha256"]),
        "expected_pre_manifest_sha256": handoff["expected_manifest_sha256"],
        "expected_source_commit": handoff["expected_source_commit"],
        "expected_source_tree": handoff["expected_source_tree"],
        "expected_preflight_sha256": handoff["expected_preflight_sha256"],
        "extracted_asar": paths["extracted_asars"][flavor],
    }
    return paths, app, final_manifest, arguments


def _installed_manifest_cli_arguments(
    arguments: dict[str, object],
    *,
    output_path: Path,
) -> list[str]:
    return [
        "write-installed-release-manifest",
        "--root",
        str(arguments["root"]),
        "--app",
        str(arguments["app"]),
        "--dist",
        str(arguments["dist_dir"]),
        "--version",
        str(arguments["version"]),
        "--flavor",
        str(arguments["flavor"]),
        "--private-root",
        str(arguments["private_root"]),
        "--out",
        str(output_path),
        "--final-manifest",
        str(arguments["final_manifest_path"]),
        "--expected-final-manifest-sha256",
        str(arguments["expected_final_manifest_sha256"]),
        "--expected-pre-manifest-sha256",
        str(arguments["expected_pre_manifest_sha256"]),
        "--expected-source-commit",
        str(arguments["expected_source_commit"]),
        "--expected-source-tree",
        str(arguments["expected_source_tree"]),
        "--expected-preflight-sha256",
        str(arguments["expected_preflight_sha256"]),
        "--extracted-asar",
        str(arguments["extracted_asar"]),
    ]


@pytest.mark.parametrize("flavor", runner.R20_ARTIFACT_CHAIN_FLAVORS)
def test_r20_installed_manifest_producer_writes_exact_deterministic_private_schema(
    tmp_path,
    capsys,
    monkeypatch,
    flavor,
):
    paths, app, _, producer_arguments = _sealed_installed_manifest_fixture(
        tmp_path,
        flavor=flavor,
    )
    invoked_python = app / runner.INSTALLED_BUNDLED_PYTHON_RELATIVE_PATH
    canonical_python = invoked_python.resolve(strict=True)
    monkeypatch.setattr(runner.sys, "executable", str(canonical_python))
    first = paths["private_root"] / f"installed/{flavor}-first.json"
    second = paths["private_root"] / f"installed/{flavor}-second.json"

    assert runner.artifact_chain_main(
        _installed_manifest_cli_arguments(
            producer_arguments,
            output_path=first,
        )
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    direct = runner.write_installed_release_manifest(
        **producer_arguments,
        output_path=second,
    )
    manifest = json.loads(first.read_text(encoding="utf-8"))

    assert first.read_bytes() == second.read_bytes()
    assert summary["manifest"] == str(first)
    assert summary["sha256"] == direct["sha256"]
    assert set(manifest) == runner.INSTALLED_RELEASE_MANIFEST_FIELDS
    assert manifest["schema"] == runner.INSTALLED_RELEASE_MANIFEST_SCHEMA
    assert manifest["source_commit"] == _git(paths["root"], "rev-parse", "HEAD")
    assert manifest["source_tree"] == _git(
        paths["root"], "rev-parse", "HEAD^{tree}"
    )
    assert manifest["bundled_python_executable_fingerprint_sha256"] == (
        hashlib.sha256(str(canonical_python).encode("utf-8")).hexdigest()
    )
    assert manifest["bundled_python_executable_sha256"] == runner._sha256(
        canonical_python
    )
    assert stat.S_IMODE(first.stat().st_mode) == 0o600
    assert first.stat().st_nlink == 1


def test_r20_installed_manifest_producer_rejects_mismatched_app_identity(tmp_path):
    paths, app, _, producer_arguments = _sealed_installed_manifest_fixture(tmp_path)
    info_path = app / "Contents/Info.plist"
    with info_path.open("rb") as handle:
        info = plistlib.load(handle)
    info["CFBundleIdentifier"] = "invalid.example.app"
    with info_path.open("wb") as handle:
        plistlib.dump(info, handle)

    with pytest.raises(runner.ArtifactChainError, match="app identity/version"):
        runner.write_installed_release_manifest(
            **producer_arguments,
            output_path=paths["private_root"] / "installed/manifest.json",
            producer_executable=(
                app / runner.INSTALLED_BUNDLED_PYTHON_RELATIVE_PATH
            ),
        )


def test_r20_installed_manifest_producer_rejects_mismatched_source_provenance(
    tmp_path,
):
    paths, app, _, producer_arguments = _sealed_installed_manifest_fixture(tmp_path)
    provenance_path = (
        app
        / "Contents/Resources/bundled-python/vmlx-bundle-provenance.json"
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["vmlx"]["commit"] = "0" * 40
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        runner.ArtifactChainError,
        match="runtime/minimum-OS contract differs from final manifest",
    ):
        runner.write_installed_release_manifest(
            **producer_arguments,
            output_path=paths["private_root"] / "installed/manifest.json",
            producer_executable=(
                app / runner.INSTALLED_BUNDLED_PYTHON_RELATIVE_PATH
            ),
        )


def test_r20_installed_manifest_producer_rejects_mismatched_bundled_python_source(
    tmp_path,
):
    paths, app, _, producer_arguments = _sealed_installed_manifest_fixture(tmp_path)
    bundled_engine = (
        app
        / "Contents/Resources/bundled-python/python/lib/python3.12/"
        "site-packages/vmlx_engine/__init__.py"
    )
    bundled_engine.write_text('VERSION = "stale"\n', encoding="utf-8")

    with pytest.raises(runner.ArtifactChainError, match="bundled-runtime.*differs"):
        runner.write_installed_release_manifest(
            **producer_arguments,
            output_path=paths["private_root"] / "installed/manifest.json",
            producer_executable=(
                app / runner.INSTALLED_BUNDLED_PYTHON_RELATIVE_PATH
            ),
        )


def test_r20_installed_manifest_producer_rejects_unbound_renderer_asar(tmp_path):
    paths, app, _, producer_arguments = _sealed_installed_manifest_fixture(tmp_path)
    (app / "Contents/Resources/app.asar").write_bytes(b"stale renderer bytes\n")

    with pytest.raises(runner.ArtifactChainError, match="app payload/modes differ"):
        runner.write_installed_release_manifest(
            **producer_arguments,
            output_path=paths["private_root"] / "installed/manifest.json",
            producer_executable=(
                app / runner.INSTALLED_BUNDLED_PYTHON_RELATIVE_PATH
            ),
        )


def test_r20_installed_manifest_producer_cli_rejects_checkout_python(
    tmp_path,
    capsys,
):
    paths, _, _, producer_arguments = _sealed_installed_manifest_fixture(tmp_path)
    output = paths["private_root"] / "installed/manifest.json"

    assert runner.artifact_chain_main(
        _installed_manifest_cli_arguments(
            producer_arguments,
            output_path=output,
        )
    ) == 1
    assert "must run under the exact bundled Python" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("mutated_surface", ["app-payload", "extracted-asar"])
def test_r20_installed_manifest_producer_rejects_payload_swaps(
    tmp_path,
    mutated_surface,
):
    paths, app, _, producer_arguments = _sealed_installed_manifest_fixture(tmp_path)
    if mutated_surface == "app-payload":
        (app / "Contents/Resources/rogue.bin").write_bytes(b"swapped app payload\n")
        expected = "app payload/modes differ"
    else:
        (Path(producer_arguments["extracted_asar"]) / "out/main.js").write_text(
            "console.log('swapped asar')\n",
            encoding="utf-8",
        )
        expected = "renderer/main output.*differs|ASAR payload/modes differ"

    with pytest.raises(runner.ArtifactChainError, match=expected):
        runner.write_installed_release_manifest(
            **producer_arguments,
            output_path=paths["private_root"] / "installed/manifest.json",
            producer_executable=(
                app / runner.INSTALLED_BUNDLED_PYTHON_RELATIVE_PATH
            ),
        )


def test_r20_installed_manifest_producer_rejects_versioned_python_mismatch(tmp_path):
    paths = _r20_artifact_chain_fixture(tmp_path)
    app = _prepare_installed_manifest_fixture(paths)
    other_python = app / "Contents/Resources/bundled-python/python/bin/python3.11"
    other_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    other_python.chmod(0o755)
    paths, app, _, producer_arguments = _seal_installed_manifest_paths(paths)

    with pytest.raises(runner.ArtifactChainError, match="exact bundled Python"):
        runner.write_installed_release_manifest(
            **producer_arguments,
            output_path=paths["private_root"] / "installed/manifest.json",
            producer_executable=other_python,
        )


def test_r20_installed_manifest_producer_rechecks_provenance_before_write(
    tmp_path,
    monkeypatch,
):
    paths, app, _, producer_arguments = _sealed_installed_manifest_fixture(tmp_path)
    provenance = (
        app
        / "Contents/Resources/bundled-python/vmlx-bundle-provenance.json"
    )
    real_tree_payload_records = runner._tree_payload_records

    def mutate_after_first_app_observation(path, *, label, **kwargs):
        result = real_tree_payload_records(path, label=label, **kwargs)
        if label == "installed release-manifest application payload":
            provenance.write_bytes(provenance.read_bytes() + b" ")
        return result

    monkeypatch.setattr(
        runner,
        "_tree_payload_records",
        mutate_after_first_app_observation,
    )
    with pytest.raises(runner.ArtifactChainError, match="changed during manifest production"):
        runner.write_installed_release_manifest(
            **producer_arguments,
            output_path=paths["private_root"] / "installed/manifest.json",
            producer_executable=(
                app / runner.INSTALLED_BUNDLED_PYTHON_RELATIVE_PATH
            ),
        )


@pytest.mark.parametrize("late_surface", ["app-tree", "extracted-asar"])
def test_r20_installed_manifest_producer_rejects_late_full_tree_mutation(
    tmp_path,
    monkeypatch,
    late_surface,
):
    paths, app, _, producer_arguments = _sealed_installed_manifest_fixture(tmp_path)
    extracted_asar = Path(producer_arguments["extracted_asar"])
    output = paths["private_root"] / "installed/manifest.json"
    real_tree_payload_records = runner._tree_payload_records

    def mutate_after_penultimate_tree_observation(path, *, label, **kwargs):
        result = real_tree_payload_records(path, label=label, **kwargs)
        if (
            late_surface == "app-tree"
            and label == "installed release-manifest application payload recheck"
        ):
            (app / "Contents/Resources/late-rogue.bin").write_bytes(
                b"late app mutation\n"
            )
        elif (
            late_surface == "extracted-asar"
            and label
            == "installed release-manifest extracted ASAR payload recheck"
        ):
            (extracted_asar / "late-rogue.js").write_text(
                "console.log('late asar mutation')\n",
                encoding="utf-8",
            )
        return result

    monkeypatch.setattr(
        runner,
        "_tree_payload_records",
        mutate_after_penultimate_tree_observation,
    )
    with pytest.raises(
        runner.ArtifactChainError,
        match="changed before manifest serialization",
    ):
        runner.write_installed_release_manifest(
            **producer_arguments,
            output_path=output,
            producer_executable=(
                app / runner.INSTALLED_BUNDLED_PYTHON_RELATIVE_PATH
            ),
        )
    assert not output.exists()


def _write_hook_and_parity_attestations(
    paths: dict[str, object],
) -> tuple[dict[str, tuple[Path, str]], dict[str, tuple[Path, str]]]:
    nonce = "a" * 64
    driver_pid = int(paths["driver_pid"])
    private_root = runner.ensure_private_evidence_root(paths["private_root"])
    hook_dir = private_root / "hook-completions"
    if hook_dir.exists():
        return (
            {
                flavor: (
                    paths["hook_attestations"][flavor],
                    runner._sha256(paths["hook_attestations"][flavor]),
                )
                for flavor in runner.R20_ARTIFACT_CHAIN_FLAVORS
            },
            {
                flavor: (
                    paths["dmg_parity_attestations"][flavor],
                    runner._sha256(paths["dmg_parity_attestations"][flavor]),
                )
                for flavor in runner.R20_ARTIFACT_CHAIN_FLAVORS
            },
        )
    hook_dir.mkdir(mode=0o700)
    source = runner._git_source_identity(paths["root"])
    preflight_sha256 = runner._sha256(paths["preflight"])
    checkout_root = Path(__file__).resolve().parents[1]
    tool_paths = {
        "git": "/usr/bin/git",
        "node": "/opt/homebrew/bin/node",
        "npm": "/opt/homebrew/bin/npm",
        "npx": "/opt/homebrew/bin/npx",
        "shasum": "/usr/bin/shasum",
        "awk": "/usr/bin/awk",
        "file": "/usr/bin/file",
        "find": "/usr/bin/find",
        "asar": str(
            checkout_root / "panel/node_modules/@electron/asar/bin/asar.js"
        ),
        "app_builder": str(
            checkout_root
            / "panel/node_modules/app-builder-bin/mac/app-builder_arm64"
        ),
        "electron_builder": str(
            checkout_root / "panel/node_modules/electron-builder/cli.js"
        ),
    }
    tools = {}
    for name, invocation in tool_paths.items():
        realpath = Path(invocation).resolve(strict=True)
        tool_record, _ = runner._read_regular_file(
            realpath,
            label=f"fixture pinned {name}",
            require_single_link=False,
        )
        tools[name] = {
            "path": invocation,
            "realpath": str(realpath),
            "sha256": tool_record["sha256"],
        }
    hook_results: dict[str, tuple[Path, str]] = {}
    parity_results: dict[str, tuple[Path, str]] = {}
    for flavor in runner.R20_ARTIFACT_CHAIN_FLAVORS:
        staged_output = paths["staged_outputs"][flavor]
        app = runner.find_exact_staged_app(staged_output)
        extracted = paths["extracted_asars"][flavor]
        bundle_attestation = runner.write_bundle_runtime_attestation(
            root=paths["root"],
            bundle_root=app / "Contents/Resources/bundled-python",
            version="1.6.20",
            private_root=private_root,
            flavor=flavor,
            output_path=paths["bundle_attestations"][flavor],
        )
        stem = f"vMLX-1.6.20-{flavor}-arm64.dmg"
        artifact_payload = {}
        for kind, artifact_path in {
            "dmg": paths["dist"] / stem,
            "blockmap": paths["dist"] / f"{stem}.blockmap",
        }.items():
            record = runner._safe_regular_file(
                artifact_path,
                label=f"fixture {flavor} {kind}",
            )
            artifact_payload[kind] = {
                "path": record["path"],
                "sha256": record["sha256"],
                "size": record["size"],
                "mode": record["mode"],
            }
        hook_path = paths["hook_attestations"][flavor]
        hook_payload = {
            "schema_version": 1,
            "scope": "r20_production",
            "stage": "electron_builder_completion",
            "version": "1.6.20",
            "flavor": flavor,
            "source": {"commit": source["commit"], "tree": source["tree"]},
            "preflight_sha256": preflight_sha256,
            "plan": {
                "path": str(paths["root"] / "build/r20-release-driver-plan.json"),
                "sha256": ("b" if flavor == "sequoia" else "c") * 64,
                "nonce": nonce,
                "driver_pid": driver_pid,
            },
            "fixed_path": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "tools": tools,
            "bundle_runtime": {
                "path": bundle_attestation["attestation"],
                "sha256": bundle_attestation["sha256"],
            },
            "runtime_contract": runner._inspect_app_runtime_contract(
                app=app,
                flavor=flavor,
                version="1.6.20",
            ),
            "staged_app": {
                "path": str(app),
                "payload": runner._tree_payload_records(
                    app,
                    label=f"fixture {flavor} app",
                ),
            },
            "extracted_asar": {
                "payload": runner._tree_payload_records(
                    extracted,
                    label=f"fixture {flavor} ASAR",
                ),
            },
            "artifacts": artifact_payload,
        }
        hook_sha256 = runner._write_private_json_with_digest(
            hook_path,
            hook_payload,
        )
        parity_path = paths["dmg_parity_attestations"][flavor]
        parity = runner.write_dmg_payload_parity_attestation(
            root=paths["root"],
            dist_dir=paths["dist"],
            version="1.6.20",
            private_root=private_root,
            flavor=flavor,
            hook_attestation_path=hook_path,
            expected_hook_sha256=hook_sha256,
            expected_nonce=nonce,
            expected_driver_pid=driver_pid,
            mounted_app=app,
            extracted_asar=extracted,
            output_path=parity_path,
        )
        hook_results[flavor] = (hook_path, hook_sha256)
        parity_results[flavor] = (parity_path, str(parity["sha256"]))
    return hook_results, parity_results


def _write_build_attestation(paths: dict[str, Path]) -> dict[str, object]:
    nonce = "a" * 64
    hooks, parity = _write_hook_and_parity_attestations(paths)
    driver_pid = int(paths["driver_pid"])
    command = [
        sys.executable,
        str(Path(runner.__file__).resolve()),
        "artifact-chain",
        "write-build-attestation",
        "--root",
        str(paths["root"]),
        "--dist",
        str(paths["dist"]),
        "--version",
        "1.6.20",
        "--preflight",
        str(paths["preflight"]),
        "--private-root",
        str(paths["private_root"]),
        "--out",
        str(paths["build_attestation"]),
        "--nonce",
        nonce,
        "--driver-pid",
        str(driver_pid),
    ]
    for flavor in runner.R20_ARTIFACT_CHAIN_FLAVORS:
        hook_path, hook_sha256 = hooks[flavor]
        parity_path, parity_sha256 = parity[flavor]
        command.extend(
            [
                f"--{flavor}-staged-output",
                str(paths["staged_outputs"][flavor]),
                f"--{flavor}-extracted-asar",
                str(paths["extracted_asars"][flavor]),
                f"--{flavor}-hook-attestation",
                str(hook_path),
                f"--{flavor}-hook-attestation-sha256",
                hook_sha256,
                f"--{flavor}-dmg-parity-attestation",
                str(parity_path),
                f"--{flavor}-dmg-parity-attestation-sha256",
                parity_sha256,
            ]
        )
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip()
        if message.startswith("ERROR: "):
            message = message.removeprefix("ERROR: ")
        raise runner.ArtifactChainError(message or "build-driver attestation failed")
    payload = runner._read_json_object(
        paths["build_attestation"],
        label="fixture build-driver attestation",
    )
    return {
        "attestation": str(paths["build_attestation"]),
        "sha256": runner._sha256(paths["build_attestation"]),
        "payload": payload,
    }


def _write_pre_manifest(paths: dict[str, Path]) -> dict[str, object]:
    nonce = "a" * 64
    attestation = _write_build_attestation(paths)
    return runner.write_pre_notary_artifact_manifest(
        root=paths["root"],
        dist_dir=paths["dist"],
        version="1.6.20",
        private_root=paths["private_root"],
        output_path=paths["pre_manifest"],
        build_attestation_path=paths["build_attestation"],
        expected_build_attestation_sha256=str(attestation["sha256"]),
        expected_nonce=nonce,
        expected_driver_pid=int(paths["driver_pid"]),
    )


def _pre_handoff(paths: dict[str, Path], pre: dict[str, object]) -> dict[str, str]:
    payload = pre["payload"]
    assert isinstance(payload, dict)
    source = payload["source"]
    preflight = payload["preflight"]
    assert isinstance(source, dict)
    assert isinstance(preflight, dict)
    return {
        "expected_manifest_sha256": str(pre["sha256"]),
        "expected_source_commit": str(source["commit"]),
        "expected_source_tree": str(source["tree"]),
        "expected_preflight_sha256": str(preflight["sha256"]),
    }


def _validate_pre(
    paths: dict[str, Path],
    pre: dict[str, object],
) -> dict[str, object]:
    return runner.validate_pre_notary_artifact_manifest(
        root=paths["root"],
        dist_dir=paths["dist"],
        version="1.6.20",
        private_root=paths["private_root"],
        manifest_path=paths["pre_manifest"],
        **_pre_handoff(paths, pre),
    )


def _create_snapshots(
    paths: dict[str, Path],
    pre: dict[str, object],
) -> dict[str, object]:
    return runner.create_pre_notary_snapshots(
        root=paths["root"],
        dist_dir=paths["dist"],
        version="1.6.20",
        private_root=paths["private_root"],
        manifest_path=paths["pre_manifest"],
        snapshot_dir=paths["private_root"] / "snapshots",
        **_pre_handoff(paths, pre),
    )


def _seal_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    runner._exclusive_sealed_write(path, encoded)


def _write_accepted_notary_records(
    paths: dict[str, Path],
    snapshots: dict[str, object],
) -> dict[str, dict[str, Path]]:
    private_root = runner.ensure_private_evidence_root(paths["private_root"])
    result_dir = private_root / "notary-results"
    result_dir.mkdir(mode=0o700)
    records: dict[str, dict[str, Path]] = {}
    snapshot_records = snapshots["snapshots"]
    assert isinstance(snapshot_records, dict)
    ids = {
        "sequoia": "11111111-1111-4111-8111-111111111111",
        "tahoe": "22222222-2222-4222-8222-222222222222",
    }
    for flavor, submission_id in ids.items():
        snapshot = snapshot_records[flavor]
        assert isinstance(snapshot, dict)
        flavor_records = {
            kind: result_dir / f"{flavor}.{kind}.json"
            for kind in ("submit", "info", "log")
        }
        _seal_json(
            flavor_records["submit"],
            {"id": submission_id, "status": "Accepted"},
        )
        _seal_json(
            flavor_records["info"],
            {"id": submission_id, "status": "Accepted"},
        )
        _seal_json(
            flavor_records["log"],
            {
                "jobId": submission_id,
                "status": "Accepted",
                "sha256": snapshot["dmg_sha256"],
                "archiveFilename": Path(str(snapshot["dmg_path"])).name,
                "ticketContents": [{"path": "vMLX.app"}],
            },
        )
        records[flavor] = flavor_records
    return records


def _submission_ids() -> dict[str, str]:
    return {
        "sequoia": "11111111-1111-4111-8111-111111111111",
        "tahoe": "22222222-2222-4222-8222-222222222222",
    }


def test_r20_production_cli_has_no_direct_self_certifying_write_pre():
    with pytest.raises(SystemExit):
        runner.artifact_chain_main(["write-pre"])
    assert (
        "tests/cross_matrix/run_packaged_integrity_contract.py"
        in runner.SOURCE_HASH_FILES
    )


def test_r20_fixture_uses_a_live_child_when_pytest_is_orphaned(tmp_path, monkeypatch):
    paths = _r20_artifact_chain_fixture(tmp_path)
    monkeypatch.setattr(os, "getppid", lambda: 1)

    result = _write_pre_manifest(paths)

    assert result["payload"]["build_attestation"]["driver_pid"] == os.getpid()


def test_r20_build_driver_accepts_live_non_direct_release_driver_ancestor():
    inner = (
        "import sys; "
        "from tests.cross_matrix.run_packaged_integrity_contract "
        "import _require_live_driver_ancestor; "
        "_require_live_driver_ancestor(int(sys.argv[1]))"
    )
    outer = (
        "import subprocess,sys; "
        "result=subprocess.run([sys.executable,'-c',sys.argv[1],sys.argv[2]]); "
        "raise SystemExit(result.returncode)"
    )
    result = subprocess.run(
        [sys.executable, "-c", outer, inner, str(os.getpid())],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_r20_pre_notary_manifest_binds_source_preflight_and_exact_artifacts(tmp_path):
    paths = _r20_artifact_chain_fixture(tmp_path)
    result = _write_pre_manifest(paths)
    manifest = paths["pre_manifest"]

    assert result["payload"]["source"]["commit"] == _git(paths["root"], "rev-parse", "HEAD")
    assert result["payload"]["source"]["tree"] == _git(
        paths["root"], "rev-parse", "HEAD^{tree}"
    )
    assert set(result["payload"]["artifacts"]) == {"sequoia", "tahoe"}
    assert stat.S_IMODE(manifest.stat().st_mode) == 0o400
    assert not Path(f"{manifest}.sha256").exists()
    assert _validate_pre(paths, result)["sha256"] == result["sha256"]


def test_r20_v4_rejects_same_version_mounted_dmg_payload_mismatch(tmp_path):
    paths = _r20_artifact_chain_fixture(tmp_path)
    hooks, _ = _write_hook_and_parity_attestations(paths)
    flavor = "sequoia"
    staged_app = runner.find_exact_staged_app(paths["staged_outputs"][flavor])
    mounted_app = tmp_path / "mounted-copy/vMLX.app"
    extracted_asar = tmp_path / "mounted-copy/extracted-asar"
    shutil.copytree(staged_app, mounted_app)
    shutil.copytree(paths["extracted_asars"][flavor], extracted_asar)
    with (mounted_app / "Contents/Info.plist").open("rb") as handle:
        assert plistlib.load(handle)["CFBundleShortVersionString"] == "1.6.20"
    (mounted_app / "Contents/Resources/payload-drift.bin").write_bytes(
        b"same-version altered payload"
    )

    hook_path, hook_sha256 = hooks[flavor]
    with pytest.raises(
        runner.ArtifactChainError,
        match="mounted sequoia DMG app differs from hook-completed staged app",
    ):
        runner.write_dmg_payload_parity_attestation(
            root=paths["root"],
            dist_dir=paths["dist"],
            version="1.6.20",
            private_root=paths["private_root"],
            flavor=flavor,
            hook_attestation_path=hook_path,
            expected_hook_sha256=hook_sha256,
            expected_nonce="a" * 64,
            expected_driver_pid=int(paths["driver_pid"]),
            mounted_app=mounted_app,
            extracted_asar=extracted_asar,
            output_path=paths["private_root"]
            / "hook-completions/sequoia.altered.dmg-parity.json",
        )


def test_r20_pre_notary_manifest_rejects_extra_or_symlinked_artifacts(tmp_path):
    paths = _r20_artifact_chain_fixture(tmp_path)
    extra = paths["dist"] / "vMLX-1.6.20-rogue-arm64.dmg"
    extra.write_bytes(b"rogue")
    with pytest.raises(runner.ArtifactChainError, match="exact Sequoia/Tahoe"):
        _write_pre_manifest(paths)
    extra.unlink()

    blockmap = paths["dist"] / "vMLX-1.6.20-sequoia-arm64.dmg.blockmap"
    blockmap.unlink()
    blockmap.symlink_to(paths["dist"] / "vMLX-1.6.20-tahoe-arm64.dmg.blockmap")
    with pytest.raises(runner.ArtifactChainError, match="symlinked path component"):
        _write_pre_manifest(paths)


def test_r20_pre_notary_manifest_rejects_artifact_and_manifest_drift(tmp_path):
    paths = _r20_artifact_chain_fixture(tmp_path)
    pre = _write_pre_manifest(paths)
    dmg = paths["dist"] / "vMLX-1.6.20-sequoia-arm64.dmg"
    dmg.write_bytes(dmg.read_bytes() + b"-changed")
    with pytest.raises(runner.ArtifactChainError, match="changed after|artifacts changed"):
        _validate_pre(paths, pre)

    paths = _r20_artifact_chain_fixture(tmp_path / "second")
    pre = _write_pre_manifest(paths)
    original_digest = str(pre["sha256"])
    paths["pre_manifest"].chmod(0o600)
    paths["pre_manifest"].write_text("{}\n", encoding="utf-8")
    paths["pre_manifest"].chmod(0o400)
    with pytest.raises(runner.ArtifactChainError, match="digest mismatch"):
        runner.validate_pre_notary_artifact_manifest(
            root=paths["root"],
            dist_dir=paths["dist"],
            version="1.6.20",
            private_root=paths["private_root"],
            manifest_path=paths["pre_manifest"],
            expected_manifest_sha256=original_digest,
            expected_source_commit=_git(paths["root"], "rev-parse", "HEAD"),
            expected_source_tree=_git(paths["root"], "rev-parse", "HEAD^{tree}"),
            expected_preflight_sha256=runner._sha256(paths["preflight"]),
        )


def test_r20_pre_notary_manifest_rejects_source_and_preflight_drift(tmp_path):
    paths = _r20_artifact_chain_fixture(tmp_path)
    pre = _write_pre_manifest(paths)
    (paths["root"] / "source.txt").write_text("dirty release source\n", encoding="utf-8")
    with pytest.raises(runner.ArtifactChainError, match="source is not clean"):
        _validate_pre(paths, pre)

    paths = _r20_artifact_chain_fixture(tmp_path / "preflight")
    pre = _write_pre_manifest(paths)
    paths["preflight"].write_text('{"status":"changed"}\n', encoding="utf-8")
    with pytest.raises(runner.ArtifactChainError, match="digest mismatch|preflight changed"):
        _validate_pre(paths, pre)


def test_r20_private_evidence_root_rejects_every_git_worktree(tmp_path):
    paths = _r20_artifact_chain_fixture(tmp_path)
    with pytest.raises(runner.ArtifactChainError, match="outside every Git worktree"):
        runner.ensure_private_evidence_root(paths["root"] / "build/private-release")

    outside = runner.ensure_private_evidence_root(paths["private_root"])
    assert outside == paths["private_root"]
    assert stat.S_IMODE(outside.stat().st_mode) == 0o700


def _post_notary_fixture(tmp_path: Path):
    paths = _r20_artifact_chain_fixture(tmp_path)
    pre = _write_pre_manifest(paths)
    snapshots = _create_snapshots(paths, pre)
    submission_ids = _submission_ids()
    for flavor in runner.R20_ARTIFACT_CHAIN_FLAVORS:
        stem = f"vMLX-1.6.20-{flavor}-arm64.dmg"
        (paths["dist"] / stem).write_bytes(f"{flavor}-stapled-dmg".encode())
        (paths["dist"] / f"{stem}.blockmap").write_bytes(
            f"{flavor}-post-staple-blockmap".encode()
        )
    snapshot_records = snapshots["snapshots"]
    assert isinstance(snapshot_records, dict)
    snapshot_paths = {
        flavor: Path(str(snapshot_records[flavor]["dmg_path"]))
        for flavor in runner.R20_ARTIFACT_CHAIN_FLAVORS
    }
    return paths, pre, snapshots, submission_ids, snapshot_paths


def _write_final(paths, pre, submission_ids, snapshot_paths, final_manifest):
    handoff = _pre_handoff(paths, pre)
    return runner.write_final_notary_artifact_manifest(
        root=paths["root"],
        dist_dir=paths["dist"],
        version="1.6.20",
        pre_notary_manifest_path=paths["pre_manifest"],
        expected_pre_manifest_sha256=handoff["expected_manifest_sha256"],
        expected_source_commit=handoff["expected_source_commit"],
        expected_source_tree=handoff["expected_source_tree"],
        expected_preflight_sha256=handoff["expected_preflight_sha256"],
        private_root=paths["private_root"],
        output_path=final_manifest,
        submission_ids=submission_ids,
        submitted_snapshot_paths=snapshot_paths,
    )


def _validate_final(paths, pre, final, final_manifest):
    handoff = _pre_handoff(paths, pre)
    return runner.validate_final_notary_artifact_manifest(
        root=paths["root"],
        dist_dir=paths["dist"],
        version="1.6.20",
        private_root=paths["private_root"],
        manifest_path=final_manifest,
        expected_manifest_sha256=str(final["sha256"]),
        expected_pre_manifest_sha256=handoff["expected_manifest_sha256"],
        expected_source_commit=handoff["expected_source_commit"],
        expected_source_tree=handoff["expected_source_tree"],
        expected_preflight_sha256=handoff["expected_preflight_sha256"],
    )


def test_r20_production_cli_rejects_caller_authored_apple_json(tmp_path):
    paths = _r20_artifact_chain_fixture(tmp_path)
    pre = _write_pre_manifest(paths)
    snapshots = _create_snapshots(paths, pre)
    records = _write_accepted_notary_records(paths, snapshots)
    with pytest.raises(SystemExit):
        runner.artifact_chain_main(
            [
                "check-apple-records",
                "--private-root",
                str(paths["private_root"]),
                "--submit",
                str(records["sequoia"]["submit"]),
                "--info",
                str(records["sequoia"]["info"]),
                "--log",
                str(records["sequoia"]["log"]),
            ]
        )


def test_r20_final_manifest_binds_post_notary_hashes_and_rejects_drift(tmp_path):
    paths, pre, _, submission_ids, snapshot_paths = _post_notary_fixture(tmp_path)
    final_manifest = paths["private_root"] / "r20-post-notary-manifest.json"
    final = _write_final(paths, pre, submission_ids, snapshot_paths, final_manifest)

    assert final["payload"]["pre_notary_manifest"]["sha256"] == pre["sha256"]
    assert stat.S_IMODE(final_manifest.stat().st_mode) == 0o400
    assert _validate_final(paths, pre, final, final_manifest)["sha256"] == final["sha256"]

    final_manifest.chmod(0o600)
    final_manifest.write_text("{}\n", encoding="utf-8")
    final_manifest.chmod(0o400)
    with pytest.raises(runner.ArtifactChainError, match="digest mismatch"):
        _validate_final(paths, pre, final, final_manifest)


def test_r20_final_manifest_refuses_stale_output_and_duplicate_submission_id(tmp_path):
    paths, pre, _, submission_ids, snapshot_paths = _post_notary_fixture(tmp_path)
    duplicate_id = "11111111-1111-4111-8111-111111111111"
    submission_ids["tahoe"] = duplicate_id
    final_manifest = paths["private_root"] / "r20-post-notary-manifest.json"
    with pytest.raises(runner.ArtifactChainError, match="distinct Apple submission IDs"):
        _write_final(paths, pre, submission_ids, snapshot_paths, final_manifest)

    final_manifest.write_text("stale\n", encoding="utf-8")
    final_manifest.chmod(0o400)
    with pytest.raises(runner.ArtifactChainError, match="refusing to overwrite"):
        _write_final(paths, pre, submission_ids, snapshot_paths, final_manifest)


def test_r20_artifact_chain_rejects_hardlinks_and_symlinked_ancestors(tmp_path):
    paths = _r20_artifact_chain_fixture(tmp_path)
    dmg = paths["dist"] / "vMLX-1.6.20-sequoia-arm64.dmg"
    os.link(dmg, paths["dist"] / "hardlink.keep")
    with pytest.raises(runner.ArtifactChainError, match="exactly one hard link"):
        _write_pre_manifest(paths)

    paths = _r20_artifact_chain_fixture(tmp_path / "symlink")
    private_root = runner.ensure_private_evidence_root(paths["private_root"])
    real = private_root / "real"
    real.mkdir()
    alias = private_root / "alias"
    alias.symlink_to(real, target_is_directory=True)
    paths["build_attestation"] = alias / "build-driver.json"
    with pytest.raises(runner.ArtifactChainError, match="symlinked path component"):
        _write_pre_manifest(paths)


def test_r20_private_outputs_are_no_clobber_and_independent_digest_bound(
    tmp_path,
    monkeypatch,
):
    paths = _r20_artifact_chain_fixture(tmp_path)
    pre = _write_pre_manifest(paths)
    with pytest.raises(runner.ArtifactChainError, match="refusing to overwrite"):
        _write_pre_manifest(paths)

    race_paths = _r20_artifact_chain_fixture(tmp_path / "race")
    original_link = runner.os.link

    def race_link(source, destination, **kwargs):
        concurrent_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o400,
            dir_fd=kwargs["dst_dir_fd"],
        )
        try:
            os.write(concurrent_fd, b"concurrent")
        finally:
            os.close(concurrent_fd)
        raise FileExistsError(destination)

    monkeypatch.setattr(runner.os, "link", race_link)
    with pytest.raises(runner.ArtifactChainError, match="concurrent private output"):
        _write_pre_manifest(race_paths)
    monkeypatch.setattr(runner.os, "link", original_link)

    manifest = paths["pre_manifest"]
    manifest.unlink()
    _seal_json(manifest, {"rewritten": True})
    with pytest.raises(runner.ArtifactChainError, match="digest mismatch"):
        runner.validate_pre_notary_artifact_manifest(
            root=paths["root"],
            dist_dir=paths["dist"],
            version="1.6.20",
            private_root=paths["private_root"],
            manifest_path=manifest,
            **_pre_handoff(paths, pre),
        )


def test_r20_bound_tool_action_rechecks_document_and_complete_toolchain(tmp_path, monkeypatch):
    marker = tmp_path / "action-ran"
    umask_marker = tmp_path / "action-umask"
    monkeypatch.setenv("VMLX_TEST_ACTION_MARKER", str(marker))
    monkeypatch.setenv("VMLX_TEST_UMASK_MARKER", str(umask_marker))
    toolchain = {}
    for name in runner.R20_PINNED_TOOL_NAMES:
        tool = tmp_path / name
        tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        tool.chmod(0o700)
        record, _ = runner._read_regular_file(
            tool,
            label=f"fake pinned {name}",
            require_single_link=False,
        )
        toolchain[name] = {
            "path": str(tool),
            "realpath": str(tool.resolve(strict=True)),
            "sha256": record["sha256"],
        }

    app_builder = Path(toolchain["app_builder"]["realpath"])
    app_builder.write_text(
        "#!/bin/sh\n"
        'printf "ran\\n" >"$VMLX_TEST_ACTION_MARKER"\n'
        'umask >"$VMLX_TEST_UMASK_MARKER"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    app_builder.chmod(0o700)
    app_builder_record, _ = runner._read_regular_file(
        app_builder,
        label="fake pinned app-builder",
        require_single_link=False,
    )
    toolchain["app_builder"]["sha256"] = app_builder_record["sha256"]

    manifest = tmp_path / "tool-manifest.json"
    _seal_json(manifest, {"toolchain": toolchain})
    manifest_record, _ = runner._read_regular_file(
        manifest,
        label="tool manifest",
    )
    original_umask = os.umask(0o022)
    try:
        result = runner.run_bound_tool_action(
            document_path=manifest,
            expected_document_sha256=manifest_record["sha256"],
            binding_kind="manifest",
            action="app-builder",
            arguments=["blockmap", "--input", "unused"],
            cwd=tmp_path,
        )
    finally:
        restored_umask = os.umask(original_umask)
    assert result["returncode"] == 0
    assert marker.read_text(encoding="utf-8") == "ran\n"
    assert umask_marker.read_text(encoding="utf-8").strip() in {"0077", "077"}
    assert restored_umask == 0o022

    fixed_path = "/usr/bin:/bin"
    monkeypatch.setenv("PATH", fixed_path)
    monkeypatch.setenv("VMLX_R20_FIXED_PATH", fixed_path)
    plan = tmp_path / "tool-plan.json"
    _seal_json(
        plan,
        {
            "fixed_path": fixed_path,
            "tools": toolchain,
        },
    )
    plan_record, _ = runner._read_regular_file(
        plan,
        label="tool plan",
    )
    result = runner.run_bound_tool_action(
        document_path=plan,
        expected_document_sha256=plan_record["sha256"],
        binding_kind="plan",
        action="app-builder",
        arguments=[],
        cwd=tmp_path,
    )
    assert result["returncode"] == 0

    marker.unlink()
    substituted = json.loads(manifest.read_text(encoding="utf-8"))
    substituted["toolchain"]["app_builder"]["sha256"] = "0" * 64
    manifest.chmod(0o600)
    manifest.write_text(
        json.dumps(substituted, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest.chmod(0o400)
    with pytest.raises(runner.ArtifactChainError, match="digest mismatch"):
        runner.run_bound_tool_action(
            document_path=manifest,
            expected_document_sha256=manifest_record["sha256"],
            binding_kind="manifest",
            action="app-builder",
            arguments=[],
            cwd=tmp_path,
        )
    assert not marker.exists()

    manifest.chmod(0o600)
    manifest.unlink()
    app_builder.write_text(
        "#!/bin/sh\n"
        'printf "ran\\n" >"$VMLX_TEST_ACTION_MARKER"\n'
        'printf "# mutated\\n" >>"$0"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    app_builder.chmod(0o700)
    mutated_record, _ = runner._read_regular_file(
        app_builder,
        label="mutating app-builder",
        require_single_link=False,
    )
    toolchain["app_builder"]["sha256"] = mutated_record["sha256"]
    _seal_json(manifest, {"toolchain": toolchain})
    manifest_record, _ = runner._read_regular_file(
        manifest,
        label="tool manifest",
    )
    with pytest.raises(runner.ArtifactChainError, match="pinned app_builder changed"):
        runner.run_bound_tool_action(
            document_path=manifest,
            expected_document_sha256=manifest_record["sha256"],
            binding_kind="manifest",
            action="app-builder",
            arguments=[],
            cwd=tmp_path,
        )
    assert marker.read_text(encoding="utf-8") == "ran\n"


def test_r20_bound_tool_action_preserves_non_utf8_output_through_json(tmp_path):
    toolchain = {}
    for name in runner.R20_PINNED_TOOL_NAMES:
        tool = tmp_path / name
        output = (r"printf 'Mach-O \251 signed\n'" + "\n") if name == "file" else ""
        tool.write_text(f"#!/bin/sh\n{output}exit 0\n", encoding="utf-8")
        tool.chmod(0o700)
        record, _ = runner._read_regular_file(
            tool,
            label=f"fake pinned {name}",
            require_single_link=False,
        )
        toolchain[name] = {
            "path": str(tool),
            "realpath": str(tool.resolve(strict=True)),
            "sha256": record["sha256"],
        }

    manifest = tmp_path / "tool-manifest.json"
    _seal_json(manifest, {"toolchain": toolchain})
    manifest_record, _ = runner._read_regular_file(
        manifest,
        label="tool manifest",
    )
    result = runner.run_bound_tool_action(
        document_path=manifest,
        expected_document_sha256=manifest_record["sha256"],
        binding_kind="manifest",
        action="file",
        arguments=[],
        cwd=tmp_path,
        capture_output=True,
    )

    expected = b"Mach-O \xa9 signed\n"
    assert result["stdout"].encode("utf-8", "surrogateescape") == expected
    encoded = json.dumps(result, sort_keys=True)
    assert encoded.isascii()
    assert (
        json.loads(encoded)["stdout"].encode("utf-8", "surrogateescape")
        == expected
    )
    restored = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                'import json,sys; sys.stdout.buffer.write('
                'json.load(sys.stdin)["stdout"].encode("utf-8", "surrogateescape"))'
            ),
        ],
        input=encoded,
        text=True,
        capture_output=True,
        check=True,
        errors="surrogateescape",
    )
    assert restored.stdout.encode("utf-8", "surrogateescape") == expected


def test_r20_git_status_excludes_only_its_verified_release_python_action(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    scripts = repo / "tests" / "cross_matrix"
    scripts.mkdir(parents=True)
    original = scripts / "run_packaged_integrity_contract.py"
    original.write_text("# verified runner\n", encoding="utf-8")
    action = scripts / (
        ".run_packaged_integrity_contract.py.vmlx-r20-"
        "0123456789abcdef0123456789abcdef"
    )
    os.link(original, action)
    monkeypatch.setattr(runner, "__file__", str(action))

    action_row = f"?? {action.relative_to(repo).as_posix()}\n"
    dirty_row = "?? actual-untracked.txt\n"
    filtered = runner._strip_verified_release_python_action_from_git_status(
        arguments=[
            "-C",
            str(repo),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        stdout=action_row + dirty_row,
    )

    assert filtered == dirty_row


def test_r20_git_source_identity_excludes_only_its_verified_release_python_action(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    scripts = repo / "tests" / "cross_matrix"
    scripts.mkdir(parents=True)
    original = scripts / "run_packaged_integrity_contract.py"
    original.write_text("# verified runner\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.name", "Artifact Chain Test")
    _git(repo, "config", "user.email", "artifact-chain@example.invalid")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")

    action = scripts / (
        ".run_packaged_integrity_contract.py.vmlx-r20-"
        "0123456789abcdef0123456789abcdef"
    )
    os.link(original, action)
    monkeypatch.setattr(runner, "__file__", str(action))

    identity = runner._git_source_identity(repo)
    assert identity["commit"] == _git(repo, "rev-parse", "HEAD")
    assert identity["tree"] == _git(repo, "rev-parse", "HEAD^{tree}")

    (repo / "actual-untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(runner.ArtifactChainError, match="source is not clean"):
        runner._git_source_identity(repo)


def test_r20_git_status_refuses_to_hide_an_unverified_lookalike(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    scripts = repo / "tests" / "cross_matrix"
    scripts.mkdir(parents=True)
    original = scripts / "run_packaged_integrity_contract.py"
    original.write_text("# verified runner\n", encoding="utf-8")
    action = scripts / (
        ".run_packaged_integrity_contract.py.vmlx-r20-"
        "fedcba9876543210fedcba9876543210"
    )
    action.write_text("# lookalike, not a hardlink\n", encoding="utf-8")
    monkeypatch.setattr(runner, "__file__", str(action))

    with pytest.raises(
        runner.ArtifactChainError,
        match="not the verified runner hardlink",
    ):
        runner._strip_verified_release_python_action_from_git_status(
            arguments=[
                "-C",
                str(repo),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            stdout=f"?? {action.relative_to(repo).as_posix()}\n",
        )


@pytest.mark.parametrize("action", ("git", "shasum", "awk", "file", "find"))
def test_r20_bound_tool_action_detects_each_residual_tool_swap(
    tmp_path,
    monkeypatch,
    action,
):
    marker = tmp_path / f"{action}-ran"
    monkeypatch.setenv("VMLX_TEST_ACTION_MARKER", str(marker))
    fixed_path = "/usr/bin:/bin"
    monkeypatch.setenv("PATH", fixed_path)
    monkeypatch.setenv("VMLX_R20_FIXED_PATH", fixed_path)
    toolchain = {}
    for name in runner.R20_PINNED_TOOL_NAMES:
        tool = tmp_path / name
        if name == action:
            tool.write_text(
                "#!/bin/sh\n"
                'printf "ran\\n" >"$VMLX_TEST_ACTION_MARKER"\n'
                'printf "# swapped\\n" >>"$0"\n'
                "exit 0\n",
                encoding="utf-8",
            )
        else:
            tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        tool.chmod(0o700)
        record, _ = runner._read_regular_file(
            tool,
            label=f"fake pinned {name}",
            require_single_link=False,
        )
        toolchain[name] = {
            "path": str(tool),
            "realpath": str(tool.resolve(strict=True)),
            "sha256": record["sha256"],
        }

    plan = tmp_path / "tool-plan.json"
    _seal_json(plan, {"fixed_path": fixed_path, "tools": toolchain})
    plan_record, _ = runner._read_regular_file(plan, label="tool plan")

    with pytest.raises(
        runner.ArtifactChainError,
        match=rf"pinned {action} changed",
    ):
        runner.run_bound_tool_action(
            document_path=plan,
            expected_document_sha256=plan_record["sha256"],
            binding_kind="plan",
            action=action,
            arguments=[],
            cwd=tmp_path,
        )
    assert marker.read_text(encoding="utf-8") == "ran\n"


def test_r20_private_directory_creation_rejects_symlink_without_chmod(tmp_path):
    private_root = runner.ensure_private_evidence_root(tmp_path / "private")
    parent = private_root / "notary-records"
    parent.mkdir(mode=0o700)
    victim = tmp_path / "victim"
    victim.mkdir(mode=0o755)
    result_dir = parent / "vMLX-1.6.20-deadbeefcafe"
    result_dir.symlink_to(victim, target_is_directory=True)

    with pytest.raises(runner.ArtifactChainError, match="refusing to reuse"):
        runner.create_private_directory(
            private_root=private_root,
            directory=result_dir,
            label="notary result directory",
        )
    assert stat.S_IMODE(victim.stat().st_mode) == 0o755

    result_dir.unlink()
    created = runner.create_private_directory(
        private_root=private_root,
        directory=result_dir,
        label="notary result directory",
    )
    assert created == result_dir
    assert stat.S_IMODE(result_dir.stat().st_mode) == 0o700
    with pytest.raises(runner.ArtifactChainError, match="refusing to reuse"):
        runner.create_private_directory(
            private_root=private_root,
            directory=result_dir,
            label="notary result directory",
        )


def test_r20_private_root_alias_swap_never_chmods_victim(tmp_path, monkeypatch):
    private_root = tmp_path / "private"
    private_root.mkdir(mode=0o755)
    moved_root = tmp_path / "private-moved"
    victim = tmp_path / "victim"
    victim.mkdir(mode=0o755)
    swapped = False

    def swapping_git(*args, **kwargs):
        nonlocal swapped
        if not swapped:
            private_root.rename(moved_root)
            private_root.symlink_to(victim, target_is_directory=True)
            swapped = True
        return runner.subprocess.CompletedProcess(args, 1, "", "")

    monkeypatch.setattr(runner, "_run_git", swapping_git)
    with pytest.raises(
        runner.ArtifactChainError,
        match="secure no-follow open failed|identity changed",
    ):
        runner.ensure_private_evidence_root(private_root)

    assert private_root.is_symlink()
    assert stat.S_IMODE(victim.stat().st_mode) == 0o755
    assert list(victim.iterdir()) == []
    assert stat.S_IMODE(moved_root.stat().st_mode) == 0o700


def test_r20_exclusive_sealed_write_rejects_parent_swap_without_victim_write(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "private-output"
    output_dir.mkdir(mode=0o700)
    moved_dir = tmp_path / "private-output-moved"
    victim = tmp_path / "victim"
    victim.mkdir(mode=0o755)
    original_write = runner.os.write
    swapped = False

    def swapping_write(fd, data):
        nonlocal swapped
        if not swapped:
            output_dir.rename(moved_dir)
            output_dir.symlink_to(victim, target_is_directory=True)
            swapped = True
        return original_write(fd, data)

    monkeypatch.setattr(runner.os, "write", swapping_write)
    with pytest.raises(
        runner.ArtifactChainError,
        match="secure no-follow open failed|identity changed",
    ):
        runner._exclusive_sealed_write(output_dir / "record.json", b"private\n")

    assert output_dir.is_symlink()
    assert stat.S_IMODE(victim.stat().st_mode) == 0o755
    assert list(victim.iterdir()) == []
    assert list(moved_dir.iterdir()) == []


def test_r20_private_command_capture_is_fd_owned_and_immutable(tmp_path, capsys):
    private_root = runner.ensure_private_evidence_root(tmp_path / "private")
    result_dir = runner.create_private_directory(
        private_root=private_root,
        directory=private_root / "notary-records" / "positive",
        label="notary result directory",
    )

    record = runner.capture_private_command(
        private_root=private_root,
        result_dir=result_dir,
        output_name="sequoia.submit.json",
        stderr_name="sequoia.submit.stderr.log",
        label="sequoia-submit",
        command=[
            runner.sys.executable,
            "-c",
            (
                "import sys;"
                "sys.stdout.write('{\"id\":\"submission\"}\\n');"
                "sys.stderr.write('PRIVATE_PROVIDER_STDERR_SENTINEL\\n')"
            ),
        ],
    )

    output = result_dir / "sequoia.submit.json"
    stderr_output = result_dir / "sequoia.submit.stderr.log"
    assert output.read_text(encoding="utf-8") == '{"id":"submission"}\n'
    assert (
        stderr_output.read_text(encoding="utf-8")
        == "PRIVATE_PROVIDER_STDERR_SENTINEL\n"
    )
    assert "PRIVATE_PROVIDER_STDERR_SENTINEL" not in capsys.readouterr().err
    assert record["path"] == str(output)
    assert record["mode"] == 0o400
    assert record["stderr"]["path"] == str(stderr_output)
    assert record["stderr"]["mode"] == 0o400
    assert stat.S_IMODE(output.stat().st_mode) == 0o400
    assert stat.S_IMODE(stderr_output.stat().st_mode) == 0o400


def test_r20_private_command_capture_rejects_result_dir_swap_without_victim_write(
    tmp_path,
):
    private_root = runner.ensure_private_evidence_root(tmp_path / "private")
    result_dir = runner.create_private_directory(
        private_root=private_root,
        directory=private_root / "notary-records" / "swap",
        label="notary result directory",
    )
    moved_result_dir = result_dir.with_name("swap-moved")
    victim = tmp_path / "victim"
    victim.mkdir(mode=0o755)

    with pytest.raises(
        runner.ArtifactChainError,
        match="secure no-follow open failed|identity changed",
    ):
        runner.capture_private_command(
            private_root=private_root,
            result_dir=result_dir,
            output_name="sequoia.submit.json",
            stderr_name="sequoia.submit.stderr.log",
            label="sequoia-submit",
            command=[
                runner.sys.executable,
                "-c",
                (
                    "import os,sys;"
                    "os.rename(sys.argv[1],sys.argv[2]);"
                    "os.symlink(sys.argv[3],sys.argv[1],target_is_directory=True);"
                    "sys.stdout.write('{\"id\":\"attacker\"}\\n')"
                ),
                str(result_dir),
                str(moved_result_dir),
                str(victim),
            ],
        )

    assert result_dir.is_symlink()
    assert stat.S_IMODE(victim.stat().st_mode) == 0o755
    assert list(victim.iterdir()) == []
    assert list(moved_result_dir.iterdir()) == []


def test_r20_snapshot_copy_and_operation_identity_detect_mutation(tmp_path, monkeypatch):
    paths = _r20_artifact_chain_fixture(tmp_path)
    pre = _write_pre_manifest(paths)
    source = paths["dist"] / "vMLX-1.6.20-sequoia-arm64.dmg"
    destination = paths["private_root"] / "mutation-snapshot.dmg"
    original_read = runner.os.read
    mutated = False

    def mutating_read(fd, size):
        nonlocal mutated
        chunk = original_read(fd, size)
        if chunk and not mutated:
            source.write_bytes(source.read_bytes() + b"-raced")
            mutated = True
        return chunk

    monkeypatch.setattr(runner.os, "read", mutating_read)
    with pytest.raises(runner.ArtifactChainError, match="changed during|changed while"):
        runner._copy_immutable_file(
            source,
            destination,
            expected_sha256=pre["payload"]["artifacts"]["sequoia"]["dmg_sha256"],
            label="sequoia DMG",
        )
    monkeypatch.setattr(runner.os, "read", original_read)

    target = tmp_path / "identity.dmg"
    target.write_bytes(b"same bytes")
    record = runner._safe_regular_file(target, label="identity test")
    target.unlink()
    target.write_bytes(b"same bytes")
    with pytest.raises(runner.ArtifactChainError, match="inode changed"):
        runner.validate_file_identity(
            path=target,
            expected_sha256=record["sha256"],
            expected_device=record["device"],
            expected_inode=record["inode"],
            expected_size=record["size"],
        )


def test_r20_staged_app_requires_exact_one_and_full_source_parity(tmp_path):
    import plistlib

    paths = _r20_artifact_chain_fixture(tmp_path)
    root = paths["root"]

    staged = tmp_path / "staged"
    app = staged / "mac-arm64/vMLX.app"
    resources = app / "Contents/Resources"
    mirror = resources / "vmlx-engine-source/vmlx_engine"
    bundled = resources / "bundled-python/python/lib/python3.12/site-packages/vmlx_engine"
    extracted = tmp_path / "extracted-asar"
    for directory in (mirror, bundled, extracted / "out"):
        directory.mkdir(parents=True)
    for packaged in (mirror / "__init__.py", bundled / "__init__.py"):
        packaged.write_text('VERSION = "test"\n', encoding="utf-8")
    (extracted / "out/main.js").write_text(
        "console.log('r20')\n",
        encoding="utf-8",
    )
    (extracted / "package.json").write_text(
        '{"name":"vmlx","version":"1.6.20","main":"out/main.js","type":"module"}\n',
        encoding="utf-8",
    )
    info = app / "Contents/Info.plist"
    info.parent.mkdir(parents=True, exist_ok=True)
    with info.open("wb") as handle:
        plistlib.dump(
            {
                "CFBundleIdentifier": "net.vmlx.app",
                "CFBundleShortVersionString": "1.6.20",
                "CFBundleVersion": "1.6.20",
                "LSMinimumSystemVersion": "14.5.0",
            },
            handle,
        )
    _write_r20_runtime_fixture(
        resources / "bundled-python",
        flavor="sequoia",
        source_commit=_git(root, "rev-parse", "HEAD"),
    )
    helper = (
        app
        / "Contents/Frameworks/vMLX Helper.app/Contents/MacOS/vMLX Helper"
    )
    helper.parent.mkdir(parents=True)
    helper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    helper.chmod(0o755)
    assert runner.validate_staged_app_parity(
        root=root,
        staged_output=staged,
        extracted_asar=extracted,
        version="1.6.20",
        flavor="sequoia",
    )["app"] == str(app)

    (bundled / "__init__.py").write_text('VERSION = "stale"\n', encoding="utf-8")
    with pytest.raises(runner.ArtifactChainError, match="tree differs"):
        runner.validate_staged_app_parity(
            root=root,
            staged_output=staged,
            extracted_asar=extracted,
            version="1.6.20",
            flavor="sequoia",
        )
    (bundled / "__init__.py").write_text('VERSION = "test"\n', encoding="utf-8")
    (staged / "other/vMLX.app").mkdir(parents=True)
    with pytest.raises(runner.ArtifactChainError, match="exactly one application"):
        runner.find_exact_staged_app(staged)


@pytest.mark.parametrize("flavor", ["sequoia", "tahoe"])
@pytest.mark.parametrize("pin", ["f" * 40, "e" * 40, "", None, 123])
def test_r20_runtime_contract_validates_optional_jang_release_pin(tmp_path, flavor, pin):
    bundle = tmp_path / "bundle"
    _write_r20_runtime_fixture(bundle, flavor=flavor, source_commit="a" * 40)
    path = bundle / "vmlx-bundle-provenance.json"
    provenance = json.loads(path.read_text())
    provenance["jang"]["release_pin"] = pin
    path.write_text(json.dumps(provenance))
    if pin == "f" * 40:
        result = runner.inspect_bundle_runtime_contract(
            bundle_root=bundle, flavor=flavor, version="1.6.20")
        assert result["bundle_provenance"]["jang"]["release_pin"] == pin
    else:
        with pytest.raises(runner.ArtifactChainError, match="JANG release pin"):
            runner.inspect_bundle_runtime_contract(
                bundle_root=bundle, flavor=flavor, version="1.6.20")


def test_r20_runtime_contract_rejects_swapped_or_identical_wrong_platforms(
    tmp_path,
):
    paths = _r20_artifact_chain_fixture(tmp_path)
    apps = {
        flavor: runner.find_exact_staged_app(paths["staged_outputs"][flavor])
        for flavor in runner.R20_ARTIFACT_CHAIN_FLAVORS
    }
    bundles = {
        flavor: app / "Contents/Resources/bundled-python"
        for flavor, app in apps.items()
    }

    for actual, asserted in (("sequoia", "tahoe"), ("tahoe", "sequoia")):
        with pytest.raises(
            runner.ArtifactChainError,
            match="bundle provenance does not declare the exact runtime contract",
        ):
            runner.inspect_bundle_runtime_contract(
                bundle_root=bundles[actual],
                flavor=asserted,
                version="1.6.20",
            )

    tahoe_bundle = bundles["tahoe"]
    metal_metadata = (
        tahoe_bundle
        / "python/lib/python3.12/site-packages/mlx_metal-0.31.2.dist-info/METADATA"
    )
    original_metal_metadata = metal_metadata.read_text(encoding="utf-8")
    metal_metadata.write_text(
        original_metal_metadata.replace("Version: 0.31.2", "Version: 0.31.3"),
        encoding="utf-8",
    )
    with pytest.raises(runner.ArtifactChainError, match="mlx and mlx-metal versions differ"):
        runner.inspect_bundle_runtime_contract(
            bundle_root=tahoe_bundle,
            flavor="tahoe",
            version="1.6.20",
        )
    metal_metadata.write_text(original_metal_metadata, encoding="utf-8")

    provenance_path = tahoe_bundle / "vmlx-bundle-provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["mlx_wheel_platform"] = "macosx_14_0_arm64"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for wheel in (
        tahoe_bundle / "python/lib/python3.12/site-packages"
    ).glob("mlx*.dist-info/WHEEL"):
        wheel.write_text(
            wheel.read_text(encoding="utf-8").replace(
                "macosx_26_0_arm64",
                "macosx_14_0_arm64",
            ),
            encoding="utf-8",
        )
    with pytest.raises(
        runner.ArtifactChainError,
        match="bundle provenance does not declare the exact runtime contract",
    ):
        runner.inspect_bundle_runtime_contract(
            bundle_root=tahoe_bundle,
            flavor="tahoe",
            version="1.6.20",
        )


def test_r20_tahoe_minimum_os_is_an_attested_app_contract(tmp_path):
    paths = _r20_artifact_chain_fixture(tmp_path)
    app = runner.find_exact_staged_app(paths["staged_outputs"]["tahoe"])
    assert runner._inspect_app_runtime_contract(
        app=app,
        flavor="tahoe",
        version="1.6.20",
    )["minimum_system_version"] == "26.0.0"

    info_path = app / "Contents/Info.plist"
    with info_path.open("rb") as handle:
        info = plistlib.load(handle)
    info["LSMinimumSystemVersion"] = "14.5.0"
    with info_path.open("wb") as handle:
        plistlib.dump(info, handle)
    with pytest.raises(
        runner.ArtifactChainError,
        match="minimum-system contract is not exact",
    ):
        runner._inspect_app_runtime_contract(
            app=app,
            flavor="tahoe",
            version="1.6.20",
        )


def test_r20_online_apple_query_pins_xcrun_and_sanitizes_path(
    tmp_path,
    monkeypatch,
    capsys,
):
    private_root = runner.ensure_private_evidence_root(tmp_path / "private")
    submission_id = "11111111-1111-4111-8111-111111111111"
    dmg_sha256 = "a" * 64
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command, **kwargs):
        if command[0] == "/usr/bin/git":
            return runner.subprocess.CompletedProcess(command, 1, "", "")
        calls.append((command, kwargs))
        if command[2] == "info":
            payload = {"id": submission_id, "status": "Accepted"}
        else:
            payload = {
                "jobId": submission_id,
                "status": "Accepted",
                "sha256": dmg_sha256,
                "archiveFilename": "vMLX-1.6.20-sequoia-arm64.dmg",
                "ticketContents": [{"path": "vMLX.app"}],
            }
        os.write(kwargs["stdout"], json.dumps(payload).encode())
        os.write(kwargs["stderr"], b"PRIVATE_APPLE_STDERR_SENTINEL\n")
        return runner.subprocess.CompletedProcess(command, 0)

    monkeypatch.setenv("PATH", f"{tmp_path}/fake-bin:/usr/bin:/bin")
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner.query_apple_notary_fresh(
        private_root=private_root,
        capture_dir=private_root / "fresh-query",
        submission_id=submission_id,
        expected_dmg_sha256=dmg_sha256,
        expected_archive_name="vMLX-1.6.20-sequoia-arm64.dmg",
        expected_team_id="55KGF2S5AY",
        keychain_profile="vmlx-notary",
        keychain=None,
    )

    assert result["status"] == "Accepted"
    assert len(calls) == 2
    assert all(command[0] == "/usr/bin/xcrun" for command, _ in calls)
    assert all(
        kwargs["env"]["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"
        for _, kwargs in calls
    )
    capture_dir = private_root / "fresh-query"
    for kind in ("info", "log"):
        stderr_path = capture_dir / f"{kind}.stderr.log"
        assert stderr_path.read_text() == "PRIVATE_APPLE_STDERR_SENTINEL\n"
        assert stat.S_IMODE(stderr_path.stat().st_mode) == 0o400
    assert "PRIVATE_APPLE_STDERR_SENTINEL" not in capsys.readouterr().err


def test_r20_caller_cannot_pair_substituted_attestation_with_own_digest(tmp_path):
    paths = _r20_artifact_chain_fixture(tmp_path)
    attestation = _write_build_attestation(paths)
    with pytest.raises(runner.ArtifactChainError, match="digest mismatch"):
        runner.write_pre_notary_artifact_manifest(
            root=paths["root"],
            dist_dir=paths["dist"],
            version="1.6.20",
            private_root=paths["private_root"],
            output_path=paths["pre_manifest"],
            build_attestation_path=paths["build_attestation"],
            expected_build_attestation_sha256="f" * 64,
            expected_nonce="a" * 64,
            expected_driver_pid=int(paths["driver_pid"]),
        )
    assert attestation["sha256"] != "f" * 64


def test_r20_fd_bound_read_is_not_substituted_by_swap_and_restore(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "artifact.dmg"
    original = b"bound-original-bytes"
    target.write_bytes(original)
    expected = runner._sha256(target)
    original_read = runner.os.read
    swapped = False

    def swap_restore_read(fd, size):
        nonlocal swapped
        if not swapped:
            backup = target.with_suffix(".bound")
            target.rename(backup)
            target.write_bytes(b"caller-substitution")
            target.unlink()
            backup.rename(target)
            swapped = True
        return original_read(fd, size)

    monkeypatch.setattr(runner.os, "read", swap_restore_read)
    record = runner._safe_regular_file(target, label="swap-restore artifact")
    assert record["sha256"] == expected
    assert target.read_bytes() == original


def test_r20_wrong_named_extra_app_is_rejected(tmp_path):
    paths = _r20_artifact_chain_fixture(tmp_path)
    staged = paths["staged_outputs"]["sequoia"]
    (staged / "mac-arm64/Other.app").mkdir()
    with pytest.raises(runner.ArtifactChainError, match="exactly one application"):
        runner.find_exact_staged_app(staged)


def test_r20_mounted_parity_rejects_extra_asar_and_stale_output(tmp_path):
    paths, pre, _, submission_ids, snapshot_paths = _post_notary_fixture(tmp_path)
    final_manifest = paths["private_root"] / "r20-post-notary-manifest.json"
    final = _write_final(
        paths,
        pre,
        submission_ids,
        snapshot_paths,
        final_manifest,
    )
    handoff = _pre_handoff(paths, pre)
    app = (
        paths["staged_outputs"]["sequoia"]
        / "mac-arm64/vMLX.app"
    )
    extracted = paths["extracted_asars"]["sequoia"]
    arguments = {
        "root": paths["root"],
        "dist_dir": paths["dist"],
        "version": "1.6.20",
        "private_root": paths["private_root"],
        "final_manifest_path": final_manifest,
        "expected_final_manifest_sha256": str(final["sha256"]),
        "expected_pre_manifest_sha256": handoff["expected_manifest_sha256"],
        "expected_source_commit": handoff["expected_source_commit"],
        "expected_source_tree": handoff["expected_source_tree"],
        "expected_preflight_sha256": handoff["expected_preflight_sha256"],
        "flavor": "sequoia",
        "mounted_app": app,
        "extracted_asar": extracted,
    }
    runner.validate_mounted_app_against_final_manifest(**arguments)

    rogue = extracted / "node_modules/rogue.js"
    rogue.parent.mkdir()
    rogue.write_text("malicious extra payload\n", encoding="utf-8")
    with pytest.raises(runner.ArtifactChainError, match="ASAR payload"):
        runner.validate_mounted_app_against_final_manifest(**arguments)
    rogue.unlink()
    rogue.parent.rmdir()

    (extracted / "out/main.js").write_text("stale renderer\n", encoding="utf-8")
    with pytest.raises(runner.ArtifactChainError, match="tree differs"):
        runner.validate_mounted_app_against_final_manifest(**arguments)


def test_r20_unrelated_recomputed_blockmap_is_rejected(tmp_path):
    expected = tmp_path / "release.dmg.blockmap"
    recomputed = tmp_path / "recomputed.dmg.blockmap"
    expected.write_bytes(b"blockmap-for-final-dmg")
    recomputed.write_bytes(b"blockmap-for-another-dmg")
    with pytest.raises(runner.ArtifactChainError, match="unrelated"):
        runner.validate_recomputed_blockmap(
            expected_blockmap=expected,
            recomputed_blockmap=recomputed,
            expected_sha256=runner._sha256(expected),
        )
