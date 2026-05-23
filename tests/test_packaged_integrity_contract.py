from __future__ import annotations

import os
from pathlib import Path

from tests.cross_matrix import run_packaged_integrity_contract as runner


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
        runner.EXPECTED_OPEN_REQUIREMENTS
    )


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
                [_expected_open_digest_line()],
            )
        raise AssertionError(name)

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(runner, "_sha256", lambda _path: "hash")
    monkeypatch.setattr(runner, "_check_packaged_renderer_dsv4_cache_ui", lambda _root: True)

    artifact = runner.build_artifact(tmp_path)

    assert artifact["checks"]["release_gate_unit_contracts_pass"] is True
    assert artifact["checks"]["dry_release_gate_fails_only_on_known_objectives"] is True
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
                [_expected_open_digest_line()],
            )
        raise AssertionError(name)

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(runner, "_sha256", lambda _path: "hash")
    monkeypatch.setattr(runner, "_check_packaged_renderer_dsv4_cache_ui", lambda _root: True)

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
            return _result(name, 1, [_expected_open_digest_line()])
        raise AssertionError(name)

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(runner, "_sha256", lambda _path: "hash")
    monkeypatch.setattr(runner, "_check_packaged_renderer_dsv4_cache_ui", lambda _root: True)

    artifact = runner.build_artifact(tmp_path)

    assert "packaged_renderer_dsv4_cache_ui_deduped" in artifact["checks"]


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
        b"actually emit DSV4_POOL_QUANT=1\n"
    )

    assert runner._check_packaged_renderer_dsv4_cache_ui(tmp_path) is True
