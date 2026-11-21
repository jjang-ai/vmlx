# SPDX-License-Identifier: Apache-2.0
"""Pure contracts for the reusable four-protocol agentic matrix runner."""

import copy
import hashlib
import json
import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from tests.cross_matrix import run_agentic_protocol_matrix as matrix

TEST_PYTHON_EXECUTABLE_PATH = str(Path(sys.executable).absolute())
TEST_PYTHON_EXECUTABLE_FINGERPRINT = matrix._sha256(
    TEST_PYTHON_EXECUTABLE_PATH
)
TEST_PYTHON_PREFIX_PATH = str(Path(sys.prefix).resolve())
TEST_PYTHON_PREFIX_FINGERPRINT = matrix._sha256(TEST_PYTHON_PREFIX_PATH)


def test_python_canonical_json_preserves_integral_float_bytes():
    canonical = matrix._canonical_json({"float": 10.0, "integer": 10})

    assert canonical == '{"float":10.0,"integer":10}'
    assert matrix._canonical_sha256({"float": 10.0, "integer": 10}) == (
        hashlib.sha256(canonical.encode()).hexdigest()
    )


def test_wait_for_idle_reads_engine_owned_lifecycle_without_scheduler(monkeypatch):
    idle = {
        "schema": "vmlx-request-lifecycle-v1",
        "available": True,
        "active_request_count": 0,
        "engine_collector_count": 0,
        "scheduler_waiting_count": 0,
        "scheduler_running_count": 0,
        "terminal_cleanup_pending": False,
    }
    states = iter(
        [
            {**idle, "active_request_count": 1, "engine_collector_count": 1},
            {**idle, "terminal_cleanup_pending": True},
            idle,
        ]
    )
    monkeypatch.setattr(
        matrix.requests,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(
            status_code=200,
            json=lambda: {"status": "healthy", "request_lifecycle": next(states)},
        ),
    )
    result = matrix.wait_for_idle("http://127.0.0.1:1/health", timeout=2)

    assert result["idle"] is True
    assert len(result["samples"]) == 3
    assert result["samples"][-1]["num_running"] is None
    assert result["samples"][-1]["request_lifecycle"] == idle


def _identity_repo(tmp_path: Path) -> tuple[Path, dict]:
    repo_root = tmp_path / "repo"
    package_json = repo_root / matrix.FILE_INFO_PATH
    package_json.parent.mkdir(parents=True)
    package_json.write_text("{}\n")
    engine = repo_root / "vmlx_engine"
    engine.mkdir()
    (engine / "__init__.py").write_text('__version__ = "test"\n')
    (engine / "server.py").write_text("SERVER = True\n")
    subprocess.run(
        ["git", "init", "-q", str(repo_root)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_root), "add", "."],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )
    return repo_root, matrix.observe_source_checkout(repo_root)


def _identity_bundle(tmp_path: Path) -> tuple[Path, dict]:
    bundle_root = tmp_path / "served-org" / "served-model"
    bundle_root.mkdir(parents=True)
    (bundle_root / "config.json").write_text('{"model_type":"test"}\n')
    (bundle_root / "tokenizer_config.json").write_text(
        '{"chat_template":"{{ messages }}"}\n'
    )
    return bundle_root, matrix.observe_bundle_configuration(bundle_root)


def _identity_runner() -> dict:
    return {
        "repo_venv": True,
        "repo_python": True,
        "python_executable_path": TEST_PYTHON_EXECUTABLE_PATH,
        "python_executable_fingerprint_sha256":
            TEST_PYTHON_EXECUTABLE_FINGERPRINT,
        "checkout_python_invocation_fingerprints_sha256": [
            TEST_PYTHON_EXECUTABLE_FINGERPRINT
        ],
        "python_prefix_path": TEST_PYTHON_PREFIX_PATH,
        "python_prefix_fingerprint_sha256": TEST_PYTHON_PREFIX_FINGERPRINT,
        "producer_pid": 2468,
        "producer_executable_path": str(Path(sys.executable).resolve()),
        "producer_executable_sha256": "1" * 64,
        "producer_executable_size_bytes": 4096,
        "producer_harness_relative_path":
            "tests/cross_matrix/run_agentic_protocol_matrix.py",
        "producer_harness_path":
            "/private/repo/tests/cross_matrix/run_agentic_protocol_matrix.py",
        "producer_harness_sha256": "2" * 64,
        "producer_harness_size_bytes": 8192,
    }


def _installed_runner_fixture(
    monkeypatch,
    tmp_path: Path,
) -> tuple[Path, dict, Path, dict]:
    repo_root, _ = _identity_repo(tmp_path)
    harness_path = (
        repo_root / "tests/cross_matrix/run_agentic_protocol_matrix.py"
    )
    harness_path.parent.mkdir(parents=True)
    harness_path.write_bytes(Path(matrix.__file__).read_bytes())
    subprocess.run(
        ["git", "-C", str(repo_root), "add", "."],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--amend",
            "--no-edit",
            "-q",
        ],
        check=True,
        capture_output=True,
    )
    source = matrix.observe_source_checkout(repo_root)

    app_root = tmp_path / "installed" / "vMLX.app"
    resources = app_root / "Contents/Resources"
    python_root = resources / "bundled-python/python"
    python_invocation = python_root / "bin/python3"
    python_executable = python_root / "bin/python3.12"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_bytes(b"installed-python-binary\n")
    python_executable.chmod(0o755)
    python_invocation.symlink_to("python3.12")
    electron_executable = app_root / "Contents/MacOS/vMLX"
    electron_executable.parent.mkdir(parents=True)
    electron_executable.write_bytes(b"installed-electron-binary\n")
    electron_executable.chmod(0o755)
    app_asar = resources / "app.asar"
    app_asar.write_bytes(b"installed-renderer-asar\n")
    bundled_provenance = resources / "bundled-python/vmlx-bundle-provenance.json"
    bundled_provenance.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "vmlx": {"commit": source["head"], "version": "test"},
                "jang": {"commit": "a" * 40, "version": "test"},
                "mlx_wheel_platform": "test",
            }
        )
        + "\n"
    )
    bundled_source = resources / "vmlx-engine-source/vmlx_engine"
    bundled_source.mkdir(parents=True)
    for name in ("__init__.py", "server.py"):
        (bundled_source / name).write_bytes(
            (repo_root / "vmlx_engine" / name).read_bytes()
        )

    manifest = {
        "schema": matrix.INSTALLED_RELEASE_MANIFEST_SCHEMA,
        "source_commit": source["head"],
        "source_tree": source["tree"],
        "app_asar_sha256": hashlib.sha256(app_asar.read_bytes()).hexdigest(),
        "electron_executable_sha256": hashlib.sha256(
            electron_executable.read_bytes()
        ).hexdigest(),
        "bundled_provenance_sha256": hashlib.sha256(
            bundled_provenance.read_bytes()
        ).hexdigest(),
        "bundled_python_executable_sha256": hashlib.sha256(
            python_executable.read_bytes()
        ).hexdigest(),
        "bundled_python_executable_fingerprint_sha256": matrix._sha256(
            str(python_executable.resolve(strict=True))
        ),
    }
    manifest_path = tmp_path / "private" / "installed-release.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps(manifest) + "\n")
    manifest_path.chmod(0o600)

    monkeypatch.setattr(
        matrix.sys,
        "executable",
        str(python_executable.resolve(strict=True)),
    )
    monkeypatch.setattr(matrix.sys, "prefix", str(python_root))
    runner = matrix.observe_runner_environment(
        repo_root,
        manifest_path,
        source,
    )
    return repo_root, source, manifest_path, runner


def _identity_health(
    source: dict,
    *,
    pid: int = 1234,
    model_name: str = "served-org/served-model",
    loaded_model_name: str | None = None,
    bundle: dict | None = None,
    cache_fingerprint: str = "b" * 64,
) -> dict:
    if bundle is None:
        files = {
            name: {"state": "missing"}
            for name in matrix.BUNDLE_ATTESTATION_FILENAMES
        }
        for name in ("config.json", "tokenizer_config.json"):
            files[name] = {
                "state": "present",
                "size_bytes": 2,
                "sha256": hashlib.sha256(b"{}\n").hexdigest(),
            }
        bundle_observed = {
            "schema": "vmlx-bundle-config-v1",
            "directory_state": "available",
            "files": files,
        }
        bundle_fingerprint = matrix._canonical_sha256(bundle_observed)
        bundle = {
            **bundle_observed,
            "aggregate_sha256": bundle_fingerprint,
            "fingerprint_sha256": bundle_fingerprint,
        }
    if loaded_model_name is None:
        loaded_model_name = str(bundle.get("model_name") or model_name)
    cache_configuration = {
        "schema": "test-cache-topology-v1",
        "nonce": cache_fingerprint,
    }
    cache_digest = matrix._canonical_sha256(cache_configuration)
    model_bundle_provenance = {
        key: copy.deepcopy(value)
        for key, value in bundle.items()
        if key != "model_name"
    }
    cache_topology_provenance = {
        "schema": "vmlx-cache-topology-attestation-v1",
        "configuration": cache_configuration,
        "canonical_sha256": cache_digest,
        "fingerprint_sha256": cache_digest,
    }
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": model_name,
        "loaded_model_name": loaded_model_name,
        "runtime_provenance": {
            "pid": pid,
            "server_module_sha256": source["server_module_sha256"],
            "package_init_sha256": source["package_init_sha256"],
            "python_source_tree_sha256": source[
                "python_source_tree_sha256"
            ],
            "python_source_file_count": source["python_source_file_count"],
            "python_source_read_error_count": source[
                "python_source_read_error_count"
            ],
            "python_executable_fingerprint_sha256":
                TEST_PYTHON_EXECUTABLE_FINGERPRINT,
            "model_bundle_provenance": copy.deepcopy(model_bundle_provenance),
            "cache_topology_provenance": copy.deepcopy(
                cache_topology_provenance
            ),
        },
        "model_bundle_provenance": model_bundle_provenance,
        "cache_topology_provenance": cache_topology_provenance,
        "scheduler": {"num_waiting": 0, "num_running": 0},
        "cache": {"scheduler_cache": {"cache_hits": 0}},
    }


def _round(call_id: str = "call_1", name: str = "file_info", arguments=None):
    return {
        "content": "",
        "reasoning": "private",
        "tool_calls": [
            {
                "index": 0,
                "id": call_id,
                "name": name,
                "arguments": arguments or {"path": matrix.FILE_INFO_PATH},
            }
        ],
    }


def _execution(call_id: str = "call_1", name: str = "file_info"):
    arguments = (
        {"path": matrix.FILE_INFO_PATH}
        if name == "file_info"
        else {"command": matrix.PWD_COMMAND}
    )
    return {
        "name": name,
        "call_id": call_id,
        "arguments": arguments,
        "result": {"ok": True},
        "output": '{"ok":true}',
    }


def test_runner_identity_requires_real_producer_pid_executable_and_harness_bytes():
    runner = _identity_runner()
    assert matrix._runner_environment_failures(runner) == []

    runner["producer_pid"] = 0
    runner["producer_executable_sha256"] = "not-a-hash"
    runner["producer_harness_relative_path"] = "different.py"
    assert matrix._runner_environment_failures(runner) == [
        "proof producer PID is invalid",
        "proof producer executable bytes fingerprint is invalid",
        "proof producer harness relative path is invalid",
    ]

    runner = _identity_runner()
    runner["python_executable_path"] = "/private/not-the-runner"
    assert matrix._runner_environment_failures(runner) == [
        "proof runner Python executable path binding is invalid",
        "proof producer executable path cannot be resolved",
    ]


def test_installed_runner_is_bound_to_external_manifest_and_bundled_python(
    monkeypatch,
    tmp_path: Path,
):
    _, source, manifest_path, runner = _installed_runner_fixture(
        monkeypatch,
        tmp_path,
    )

    assert manifest_path.is_file()
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
    assert runner["execution_mode"] == "installed-runtime"
    assert runner["repo_venv"] is False
    assert runner["repo_python"] is False
    assert runner["installed_runtime"]["manifest_opened_nofollow"] is True
    assert runner["installed_runtime"]["source_binding"]["head"] == source["head"]
    assert matrix._runner_environment_failures(runner) == []

    installed = runner["installed_runtime"]
    packaged_invocation = (
        Path(installed["app_path"])
        / matrix.INSTALLED_BUNDLED_PYTHON_RELATIVE_PATH
    )
    assert packaged_invocation.is_symlink()
    assert Path(runner["python_executable_path"]).name == "python3.12"
    assert packaged_invocation.resolve(strict=True) == Path(
        runner["python_executable_path"]
    )
    assert (
        installed["bundled_python"]["sha256"]
        == runner["producer_executable_sha256"]
    )
    assert (
        installed["manifest"][
            "bundled_python_executable_sha256"
        ]
        == runner["producer_executable_sha256"]
    )
    assert (
        installed["manifest"][
            "bundled_python_executable_fingerprint_sha256"
        ]
        == runner["python_executable_fingerprint_sha256"]
    )


def test_installed_runner_health_binding_uses_manifest_attested_python(
    monkeypatch,
    tmp_path: Path,
):
    _, source, _, runner = _installed_runner_fixture(monkeypatch, tmp_path)
    bundle_root, bundle = _identity_bundle(tmp_path)
    del bundle_root
    health = _identity_health(source, bundle=bundle)
    health["runtime_provenance"][
        "python_executable_fingerprint_sha256"
    ] = runner["python_executable_fingerprint_sha256"]
    identity, failures = matrix._health_identity(health)

    assert failures == []
    assert matrix._validate_health_source_binding(
        identity,
        source,
        runner,
        bundle,
        bundle["model_name"],
    ) == []

    # Electron starts the backend through the fixed in-app `python3` symlink,
    # while the proof producer binds the same binary through its canonical
    # versioned path. Both exact paths are accepted; neither broadens the
    # trust boundary beyond the manifest-hashed packaged interpreter.
    health["runtime_provenance"][
        "python_executable_fingerprint_sha256"
    ] = runner["installed_runtime"][
        "packaged_python_invocation_fingerprint_sha256"
    ]
    identity, failures = matrix._health_identity(health)
    assert failures == []
    assert matrix._validate_health_source_binding(
        identity,
        source,
        runner,
        bundle,
        bundle["model_name"],
    ) == []

    health["runtime_provenance"][
        "python_executable_fingerprint_sha256"
    ] = matrix._sha256("/private/tmp/foreign-python")
    identity, failures = matrix._health_identity(health)
    assert failures == []
    assert matrix._validate_health_source_binding(
        identity,
        source,
        runner,
        bundle,
        bundle["model_name"],
    ) == ["observed backend Python executable does not match the proof runner"]


@pytest.mark.parametrize(
    ("failure_mode", "expected"),
    [
        ("manifest-symlink", "regular non-symlink"),
        ("manifest-hardlink", "exactly one filesystem link"),
        ("manifest-in-worktree", "outside every Git worktree"),
        ("manifest-oversized", "safety limit"),
        ("wrong-source", "source commit does not match"),
        ("mutated-asar", "app asar does not match"),
        ("mutated-python", "bundled Python does not match"),
        (
            "python-symlink-escape",
            "manifest-attested bundled Python|resolves outside the installed app",
        ),
        ("python-versioned-mismatch", "exact packaged bundled-Python target"),
    ],
)
def test_installed_runner_rejects_unbound_release_material(
    monkeypatch,
    tmp_path: Path,
    failure_mode: str,
    expected: str,
):
    repo_root, source, manifest_path, runner = _installed_runner_fixture(
        monkeypatch,
        tmp_path,
    )
    installed = runner["installed_runtime"]
    requested_manifest = manifest_path
    if failure_mode == "manifest-symlink":
        requested_manifest = manifest_path.with_name("manifest-link.json")
        requested_manifest.symlink_to(manifest_path)
    elif failure_mode == "manifest-hardlink":
        requested_manifest = manifest_path.with_name("manifest-hardlink.json")
        os.link(manifest_path, requested_manifest)
    elif failure_mode == "manifest-in-worktree":
        requested_manifest = repo_root / "private-manifest.json"
        requested_manifest.write_bytes(manifest_path.read_bytes())
        requested_manifest.chmod(0o600)
    elif failure_mode == "manifest-oversized":
        manifest_path.write_bytes(
            manifest_path.read_bytes() + b" " * (1024 * 1024)
        )
        manifest_path.chmod(0o600)
    elif failure_mode == "wrong-source":
        manifest = json.loads(manifest_path.read_text())
        manifest["source_commit"] = "0" * len(source["head"])
        manifest_path.write_text(json.dumps(manifest) + "\n")
        manifest_path.chmod(0o600)
    elif failure_mode == "mutated-asar":
        Path(installed["artifacts"]["app_asar"]["path"]).write_bytes(
            b"mutated-after-manifest\n"
        )
    elif failure_mode == "mutated-python":
        Path(installed["invoked_python_path"]).write_bytes(
            b"mutated-python-after-manifest\n"
        )
    elif failure_mode == "python-symlink-escape":
        python_path = Path(installed["invoked_python_path"])
        escaped = tmp_path / "escaped-python3"
        escaped.write_bytes(python_path.read_bytes())
        escaped.chmod(0o755)
        python_path.unlink()
        python_path.symlink_to(escaped)
    elif failure_mode == "python-versioned-mismatch":
        wrong_versioned_python = (
            Path(installed["app_path"])
            / "Contents/Resources/bundled-python/python/bin/python3.11"
        )
        wrong_versioned_python.write_bytes(b"other-versioned-python\n")
        wrong_versioned_python.chmod(0o755)
        monkeypatch.setattr(
            matrix.sys,
            "executable",
            str(wrong_versioned_python),
        )

    with pytest.raises(ValueError, match=expected):
        matrix.observe_runner_environment(
            repo_root,
            requested_manifest,
            source,
        )


def test_installed_runner_failure_contract_rejects_attestation_mutation(
    monkeypatch,
    tmp_path: Path,
):
    _, _, _, runner = _installed_runner_fixture(monkeypatch, tmp_path)
    runner["installed_runtime"]["bundled_source"][
        "server_module_sha256"
    ] = "f" * 64
    runner["installed_runtime"]["bundled_python"]["sha256"] = "e" * 64
    runner["installed_runtime"]["manifest"][
        "bundled_python_executable_sha256"
    ] = "d" * 64

    failures = matrix._runner_environment_failures(runner)

    assert "installed bundled Python identity is invalid" in failures
    assert "installed runtime source provenance is invalid" in failures


def test_backend_identity_accepts_equivalent_checkout_python_alias(tmp_path: Path):
    repo_root, source = _identity_repo(tmp_path)
    bundle_root, bundle = _identity_bundle(tmp_path)
    del repo_root, bundle_root
    health = _identity_health(source, bundle=bundle)
    identity, failures = matrix._health_identity(health)
    assert failures == []
    alias_fingerprint = "a" * 64
    identity["runtime_source_hashes"][
        "python_executable_fingerprint_sha256"
    ] = alias_fingerprint
    runner = _identity_runner()
    runner["checkout_python_invocation_fingerprints_sha256"].append(
        alias_fingerprint
    )
    assert matrix._validate_health_source_binding(
        identity,
        source,
        runner,
        bundle,
        bundle["model_name"],
    ) == []


def test_backend_identity_accepts_served_alias_separate_from_loaded_bundle(
    tmp_path: Path,
):
    _, source = _identity_repo(tmp_path)
    _, bundle = _identity_bundle(tmp_path)
    served_alias = "DeepSeek-V4-Flash-0731-JANG"
    health = _identity_health(
        source,
        model_name=served_alias,
        loaded_model_name=bundle["model_name"],
        bundle=bundle,
    )

    identity, failures = matrix._health_identity(health)

    assert failures == []
    assert identity["model_name"] == served_alias
    assert identity["loaded_model_name"] == bundle["model_name"]
    assert (
        matrix._validate_health_source_binding(
            identity,
            source,
            _identity_runner(),
            bundle,
            served_alias,
        )
        == []
    )


def test_backend_identity_rejects_served_alias_substituted_for_loaded_bundle(
    tmp_path: Path,
):
    _, source = _identity_repo(tmp_path)
    _, bundle = _identity_bundle(tmp_path)
    served_alias = "DeepSeek-V4-Flash-0731-JANG"
    health = _identity_health(
        source,
        model_name=served_alias,
        loaded_model_name=served_alias,
        bundle=bundle,
    )

    identity, failures = matrix._health_identity(health)

    assert failures == []
    assert matrix._validate_health_source_binding(
        identity,
        source,
        _identity_runner(),
        bundle,
        served_alias,
    ) == ["/health loaded_model_name does not match the independently observed bundle"]


def test_backend_identity_fingerprint_retains_bundle_file_attestation(
    tmp_path: Path,
):
    _, source = _identity_repo(tmp_path)
    _, bundle = _identity_bundle(tmp_path)
    identity, failures = matrix._health_identity(
        _identity_health(source, bundle=bundle)
    )
    assert failures == []
    assert identity["model_bundle_files"] == bundle["files"]
    core = {
        key: value
        for key, value in identity.items()
        if key != "fingerprint_sha256"
    }
    assert identity["fingerprint_sha256"] == matrix._canonical_sha256(core)


def test_public_request_metadata_retains_exact_tool_contracts_and_safe_history_links():
    payload = {
        "stream": True,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "file_info",
                    "parameters": matrix.TOOL_PARAMETERS["file_info"],
                },
            },
            {
                "name": "run_command",
                "input_schema": matrix.TOOL_PARAMETERS["run_command"],
            },
        ],
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_file",
                        "function": {"name": "file_info", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_file",
                "name": "file_info",
                "content": '{"size_human":"5.2 KB"}',
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_pwd",
                        "name": "run_command",
                        "input": {"command": "pwd"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_pwd",
                        "content": '{"stdout":"/private/repo"}',
                    }
                ],
            },
        ],
    }

    public = matrix._request_public(2, payload)

    assert public["tool_contracts"] == [
        {
            "name": "file_info",
            "parameters": matrix.TOOL_PARAMETERS["file_info"],
        },
        {
            "name": "run_command",
            "parameters": matrix.TOOL_PARAMETERS["run_command"],
        },
    ]
    assert public["tool_history_linkage"] == [
        {
            "kind": "assistant_tool_call",
            "role": "assistant",
            "call_id": "call_file",
            "name": "file_info",
        },
        {
            "kind": "tool_result",
            "role": "tool",
            "call_id": "call_file",
            "name": "file_info",
            "output_chars": len('{"size_human":"5.2 KB"}'),
            "output_sha256": matrix._sha256('{"size_human":"5.2 KB"}'),
        },
        {
            "kind": "assistant_tool_call",
            "role": "assistant",
            "call_id": "call_pwd",
            "name": "run_command",
        },
        {
            "kind": "tool_result",
            "role": "user",
            "call_id": "call_pwd",
            "output_chars": len('{"stdout":"/private/repo"}'),
            "output_sha256": matrix._sha256('{"stdout":"/private/repo"}'),
        },
    ]
    serialized = json.dumps(public, sort_keys=True)
    assert "5.2 KB" not in serialized
    assert "/private/repo" not in serialized


def test_public_request_metadata_links_responses_function_output_without_payload():
    output = '{"stdout":"/private/repo"}'
    public = matrix._request_public(
        3,
        {
            "previous_response_id": "response-2",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_pwd",
                    "output": output,
                }
            ]
        },
    )

    assert public["tool_history_linkage"] == [
        {
            "kind": "tool_result",
            "role": "function_call_output",
            "call_id": "call_pwd",
            "output_chars": len(output),
            "output_sha256": matrix._sha256(output),
        }
    ]
    assert public["previous_response_id"] == "response-2"
    assert output not in json.dumps(public, sort_keys=True)


def test_fragmented_tool_assembler_reconstructs_split_name_and_arguments():
    assembler = matrix.FragmentedToolAssembler()
    assembler.add(
        0,
        call_id="call_split",
        name="file_",
        arguments='{"path":"panel/',
    )
    assembler.add(0, name="info", arguments='package.json"}')

    assert assembler.calls() == [
        {
            "index": 0,
            "id": "call_split",
            "name": "file_info",
            "arguments": {"path": "panel/package.json"},
        }
    ]


def test_anthropic_empty_tool_start_does_not_mask_later_json_fragments():
    collector = matrix.EventCollector(protocol="anthropic", started=0.0)
    matrix._parse_stream_object(
        "anthropic",
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_split",
                "name": "file_info",
                "input": {},
            },
        },
        "content_block_start",
        collector,
        1.0,
    )
    matrix._parse_stream_object(
        "anthropic",
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"path":"panel/package.json"}',
            },
        },
        "content_block_delta",
        collector,
        2.0,
    )

    assert collector.tools.calls() == [
        {
            "index": 2,
            "id": "toolu_split",
            "name": "file_info",
            "arguments": {"path": "panel/package.json"},
        }
    ]


def test_responses_argument_delta_never_replaces_call_id_with_item_id():
    collector = matrix.EventCollector(protocol="responses", started=0.0)
    matrix._parse_stream_object(
        "responses",
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "id": "fc_item",
                "type": "function_call",
                "call_id": "call_real",
                "name": "file_info",
                "arguments": "",
            },
        },
        "response.output_item.added",
        collector,
        1.0,
    )
    matrix._parse_stream_object(
        "responses",
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_item",
            "output_index": 1,
            "delta": '{"path":"panel/package.json"}',
        },
        "response.function_call_arguments.delta",
        collector,
        2.0,
    )
    matrix._parse_stream_object(
        "responses",
        {
            "type": "response.function_call_arguments.done",
            "item_id": "fc_item",
            "output_index": 1,
            "arguments": '{"path":"panel/package.json"}',
        },
        "response.function_call_arguments.done",
        collector,
        3.0,
    )
    assert collector.tools.calls()[0]["id"] == "call_real"


@pytest.mark.parametrize("protocol", ["chat", "anthropic", "ollama"])
def test_history_after_tool_preserves_native_call_result_adjacency(protocol):
    history = [{"role": "user", "content": "use the tool"}]
    updated = matrix.history_after_tool(
        protocol,
        history,
        _round(),
        _execution(),
        "continue",
    )

    assert updated[0] == history[0]
    assert updated[1]["role"] == "assistant"
    if protocol == "chat":
        assert updated[1]["tool_calls"][0]["id"] == "call_1"
        assert updated[2] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "file_info",
            "content": '{"ok":true}',
        }
    elif protocol == "anthropic":
        thinking = updated[1]["content"][0]
        tool_use = updated[1]["content"][1]
        tool_result = updated[2]["content"][0]
        assert thinking == {
            "type": "thinking",
            "thinking": "private",
            "signature": "dm1seA==",
        }
        assert tool_use["type"] == "tool_use"
        assert tool_use["id"] == "call_1"
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "call_1"
    else:
        assert updated[1]["tool_calls"][0]["function"]["arguments"] == {
            "path": "panel/package.json"
        }
        assert updated[2] == {
            "role": "tool",
            "tool_name": "file_info",
            "content": '{"ok":true}',
        }


def test_responses_history_carries_real_output_then_latest_user_instruction():
    updated = matrix.history_after_tool(
        "responses",
        [],
        _round(),
        _execution(),
        "continue",
    )

    assert updated == [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"ok":true}',
        },
        {"role": "user", "content": "continue"},
    ]


@pytest.mark.parametrize(
    ("protocol", "terminals", "stream", "expect_tool", "expected"),
    [
        ("chat", ["tool_calls", "DONE"], True, True, True),
        ("chat", ["stop", "DONE"], True, False, True),
        ("chat", ["DONE", "stop"], True, False, False),
        ("chat", ["stop", "DONE", "DONE"], True, False, False),
        ("responses", ["response.completed"], True, True, True),
        ("responses", ["response.incomplete"], True, False, False),
        ("anthropic", ["tool_use", "message_stop"], True, True, True),
        ("anthropic", ["end_turn", "message_stop"], True, False, True),
        ("anthropic", ["message_stop", "end_turn"], True, False, False),
        ("ollama", ["tool_calls"], True, True, True),
        ("ollama", ["stop"], True, True, False),
        ("ollama", ["stop"], False, False, True),
    ],
)
def test_terminal_classification(protocol, terminals, stream, expect_tool, expected):
    result = matrix.classify_terminal(
        protocol,
        terminals,
        stream=stream,
        expect_tool=expect_tool,
    )

    assert result["pass"] is expected


def test_terminal_classification_rejects_nonterminal_event_after_terminal():
    result = matrix.classify_terminal(
        "chat",
        ["stop", "DONE"],
        stream=True,
        expect_tool=False,
        events=[
            {"channel": "content", "kind": "chat.content.delta"},
            {"channel": "terminal", "kind": "stop"},
            {"channel": "content", "kind": "chat.content.delta"},
            {"channel": "terminal", "kind": "DONE"},
        ],
    )

    assert result["pass"] is False
    assert result["post_terminal_events"] == 1


def test_responses_nonstream_requires_explicit_completed_status():
    missing = matrix.parse_nonstream(
        "responses",
        {
            "id": "response-without-status",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "visible"}],
                }
            ],
        },
        200,
        1.0,
    )
    explicit = matrix.parse_nonstream(
        "responses",
        {
            "id": "response-completed",
            "status": "completed",
            "output": [],
        },
        200,
        1.0,
    )

    assert missing["terminals"] == []
    assert [error["kind"] for error in missing["errors"]] == [
        "responses.missing_status"
    ]
    assert matrix.classify_terminal(
        "responses",
        missing["terminals"],
        stream=False,
        expect_tool=False,
        events=missing["events"],
    )["pass"] is False
    assert explicit["terminals"] == ["response.completed"]


@pytest.mark.parametrize(
    ("protocol", "mode", "payload", "expected"),
    [
        (
            "chat",
            "stream",
            {
                "delta_events_before_abort": 3,
                "cancel_status": 200,
                "terminals_before_abort": [],
                "idle_after_abort": {"idle": True},
            },
            True,
        ),
        (
            "responses",
            "stream",
            {
                "delta_events_before_abort": 3,
                "cancel_status": 404,
                "terminals_before_abort": [],
                "idle_after_abort": {"idle": True},
            },
            False,
        ),
        (
            "anthropic",
            "stream",
            {
                "delta_events_before_abort": 3,
                "cancel_status": None,
                "terminals_before_abort": [],
                "idle_after_abort": {"idle": True},
            },
            True,
        ),
        (
            "ollama",
            "stream",
            {
                "delta_events_before_abort": 3,
                "cancel_status": None,
                "terminals_before_abort": ["stop"],
                "idle_after_abort": {"idle": True},
            },
            False,
        ),
        (
            "chat",
            "nonstream",
            {"idle_after_disconnect": {"idle": True}},
            True,
        ),
    ],
)
def test_abort_classification_requires_real_cancel_and_no_false_terminal(
    protocol, mode, payload, expected
):
    assert matrix.classify_abort(protocol, mode, payload, 3)["pass"] is expected


def test_allowlist_rejects_path_command_and_extra_argument_variants():
    valid_file = {
        "id": "call_file",
        "name": "file_info",
        "arguments": {"path": "panel/package.json"},
    }
    valid_pwd = {
        "id": "call_pwd",
        "name": "run_command",
        "arguments": {"command": "pwd"},
    }
    assert matrix.validate_allowlisted_call(valid_file, "file_info") == (True, "")
    assert matrix.validate_allowlisted_call(valid_pwd, "run_command") == (True, "")

    for invalid, name in [
        ({**valid_file, "arguments": {"path": "pyproject.toml"}}, "file_info"),
        (
            {**valid_file, "arguments": {"path": "panel/package.json", "extra": 1}},
            "file_info",
        ),
        ({**valid_pwd, "arguments": {"command": "pwd; env"}}, "run_command"),
    ]:
        assert matrix.validate_allowlisted_call(invalid, name)[0] is False


def test_execute_allowlisted_tools_uses_real_repo_state(tmp_path: Path):
    package = tmp_path / "panel" / "package.json"
    package.parent.mkdir()
    package.write_text('{"name":"fixture"}\n')

    file_result = matrix.execute_allowlisted_tool(
        tmp_path,
        {
            "id": "call_file",
            "name": "file_info",
            "arguments": {"path": "panel/package.json"},
        },
    )
    pwd_result = matrix.execute_allowlisted_tool(
        tmp_path,
        {
            "id": "call_pwd",
            "name": "run_command",
            "arguments": {"command": "pwd"},
        },
    )

    assert file_result["result"]["size_bytes"] == package.stat().st_size
    assert file_result["result"]["path"] == "panel/package.json"
    assert pwd_result["result"] == {
        "command": "pwd",
        "stdout": str(tmp_path),
        "exit_code": 0,
    }
    assert matrix._human_size(5284) == "5.2 KB"


def test_sanitized_round_keeps_timing_and_hashes_but_drops_private_reasoning():
    private = "PRIVATE-REASONING-MUST-NOT-BE-SERIALIZED"
    row = {
        "status_code": 200,
        "elapsed_ms": 12.5,
        "response_id": "resp_safe",
        "reasoning": private,
        "content": "VISIBLE-DONE",
        "tool_calls": [],
        "terminals": ["response.completed"],
        "events": [
            {
                "at_ms": 4.0,
                "channel": "reasoning",
                "kind": "response.reasoning_summary_text.delta",
                "chars": len(private),
                "sha256": matrix._sha256(private),
            }
        ],
    }

    sanitized = matrix._sanitized_round(row)
    encoded = json.dumps(sanitized)
    assert private not in encoded
    assert sanitized["reasoning_chars"] == len(private)
    assert sanitized["reasoning_sha256"] == matrix._sha256(private)
    assert sanitized["events"][0]["at_ms"] == 4.0
    assert sanitized["content"] == "VISIBLE-DONE"


@pytest.mark.parametrize(
    "visible",
    [
        "<think>private</think>",
        '<tool_call>{"name":"file_info"}</tool_call>',
        "<function=file_info><parameter=path>panel/package.json",
        "<|tool_call|>file_info",
        "[TOOL_CALLS] file_info",
        "<｜tool▁calls▁begin｜>",
        '<｜DSML｜invoke name="file_info">',
        "<minimax:tool_call>",
        "```tool_code",
    ],
)
def test_visible_control_markup_rejects_reasoning_and_tool_protocol_leaks(visible):
    assert matrix._contains_control_markup(visible)


@pytest.mark.parametrize(
    ("protocol", "payload", "event_name", "error_kind"),
    [
        (
            "responses",
            {"type": "error", "error": {"message": "boom"}},
            "error",
            "error",
        ),
        (
            "anthropic",
            {"type": "error", "error": {"message": "boom"}},
            "error",
            "error",
        ),
        ("ollama", {"error": "boom"}, None, "ollama.error"),
    ],
)
def test_protocol_error_events_are_retained_as_fail_closed_evidence(
    protocol, payload, event_name, error_kind
):
    collector = matrix.EventCollector(protocol=protocol, started=0.0)
    matrix._parse_stream_object(protocol, payload, event_name, collector, 1.0)
    assert collector.errors[0]["kind"] == error_kind


def test_final_synthesis_instruction_does_not_leak_expected_result_values():
    prompt = matrix.final_synthesis_instruction("direct", "responses", "stream")
    assert "SIZE=<copy size_human" in prompt
    assert "PWD=<copy stdout" in prompt
    assert "5.2 KB" not in prompt
    assert "/Users/example" not in prompt


def test_direct_and_gateway_synthesis_prompts_are_byte_identical():
    direct = matrix.final_synthesis_instruction("direct", "anthropic", "stream")
    gateway = matrix.final_synthesis_instruction("gateway", "anthropic", "stream")

    assert direct == gateway
    assert "DIRECT" not in direct
    assert "GATEWAY" not in direct


def test_agentic_flow_turns_thinking_off_for_final_synthesis():
    class RecordingClient:
        def __init__(self):
            self.payloads = []

        def send(self, protocol, payload, stream, *, capture_label):
            assert protocol == "chat"
            assert stream is False
            self.payloads.append(copy.deepcopy(payload))
            if capture_label.endswith("round1"):
                call = {
                    "id": "call-file",
                    "name": "file_info",
                    "arguments": {"path": matrix.FILE_INFO_PATH},
                }
                terminal = "tool_calls"
                return {
                    "status_code": 200,
                    "response_id": "response-1",
                    "reasoning": "reason one",
                    "content": "",
                    "tool_calls": [call],
                    "terminals": [terminal],
                    "events": [{"at_ms": 1, "channel": "terminal", "kind": terminal}],
                    "errors": [],
                }
            if capture_label.endswith("round2"):
                call = {
                    "id": "call-pwd",
                    "name": "run_command",
                    "arguments": {"command": matrix.PWD_COMMAND},
                }
                terminal = "tool_calls"
                return {
                    "status_code": 200,
                    "response_id": "response-2",
                    "reasoning": "reason two",
                    "content": "",
                    "tool_calls": [call],
                    "terminals": [terminal],
                    "events": [{"at_ms": 1, "channel": "terminal", "kind": terminal}],
                    "errors": [],
                }
            terminal = "stop"
            return {
                "status_code": 200,
                "response_id": "response-3",
                "reasoning": "",
                "content": "final",
                "tool_calls": [],
                "terminals": [terminal],
                "events": [{"at_ms": 1, "channel": "terminal", "kind": terminal}],
                "errors": [],
            }

    client = RecordingClient()
    repo_root = Path(matrix.__file__).resolve().parents[2]
    result = matrix.run_flow(
        client,
        base_label="direct",
        protocol="chat",
        mode="nonstream",
        model="served-model",
        repo_root=repo_root,
        max_tokens=128,
        enable_thinking=True,
    )

    assert [payload["enable_thinking"] for payload in client.payloads] == [
        True,
        True,
        False,
    ]
    assert result["requests"][2]["enable_thinking"] is False


def test_responses_agentic_flow_can_keep_developer_instructions_stable():
    class RecordingClient:
        def __init__(self):
            self.payloads = []

        def send(self, protocol, payload, stream, *, capture_label):
            assert protocol == "responses"
            assert stream is True
            self.payloads.append(copy.deepcopy(payload))
            if capture_label.endswith("round1"):
                call = {
                    "id": "call-file",
                    "name": "file_info",
                    "arguments": {"path": matrix.FILE_INFO_PATH},
                }
                content = ""
            elif capture_label.endswith("round2"):
                call = {
                    "id": "call-pwd",
                    "name": "run_command",
                    "arguments": {"command": matrix.PWD_COMMAND},
                }
                content = ""
            else:
                call = None
                content = "final"
            return {
                "status_code": 200,
                "response_id": f"resp-{len(self.payloads)}",
                "reasoning": "reason" if call else "",
                "content": content,
                "tool_calls": [call] if call else [],
                "terminals": ["response.completed"],
                "events": [
                    {
                        "at_ms": 1,
                        "channel": "terminal",
                        "kind": "response.completed",
                    }
                ],
                "errors": [],
            }

    client = RecordingClient()
    repo_root = Path(matrix.__file__).resolve().parents[2]
    matrix.run_flow(
        client,
        base_label="direct",
        protocol="responses",
        mode="stream",
        model="served-model",
        repo_root=repo_root,
        max_tokens=128,
        enable_thinking=True,
        first_tool_choice="auto",
        second_tool_choice="auto",
        stable_responses_instructions=True,
    )

    assert {payload["instructions"] for payload in client.payloads} == {
        matrix.STABLE_RESPONSES_INSTRUCTIONS
    }
    assert client.payloads[0]["input"] != client.payloads[1]["input"]


def test_request_metadata_hashes_full_body_and_normalizes_only_tool_ids():
    base = {
        "model": "served-model",
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_one",
                        "type": "function",
                        "function": {"name": "file_info", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_one", "content": "ok"},
        ],
        "stream": True,
        "max_tokens": 512,
        "enable_thinking": True,
        "tool_choice": {"type": "function", "function": {"name": "run_command"}},
    }
    other = copy.deepcopy(base)
    other["messages"][0]["tool_calls"][0]["id"] = "call_two"
    other["messages"][1]["tool_call_id"] = "call_two"

    first = matrix._request_public(2, base)
    second = matrix._request_public(2, other)

    assert first["body_sha256"] != second["body_sha256"]
    assert first["canonical_body_sha256"] == second["canonical_body_sha256"]
    assert first["tool_choice"] == base["tool_choice"]
    assert first["max_output_tokens"] == 512


def test_request_metadata_normalizes_only_responses_continuation_identity():
    first_request = {
        "model": "served-model",
        "input": [{"role": "user", "content": "continue"}],
        "stream": True,
        "previous_response_id": "response-one",
    }
    second_request = {
        **first_request,
        "previous_response_id": "response-two",
    }

    first = matrix._request_public(2, first_request, protocol="responses")
    second = matrix._request_public(2, second_request, protocol="responses")

    assert first["previous_response_id"] == "response-one"
    assert second["previous_response_id"] == "response-two"
    assert first["body_sha256"] != second["body_sha256"]
    assert first["canonical_body_sha256"] == second["canonical_body_sha256"]


def test_request_metadata_does_not_normalize_noncontinuation_response_id_fields():
    nested_first = {
        "model": "served-model",
        "input": [
            {
                "role": "user",
                "content": {"previous_response_id": "nested-one"},
            }
        ],
        "stream": True,
    }
    nested_second = {
        **nested_first,
        "input": [
            {
                "role": "user",
                "content": {"previous_response_id": "nested-two"},
            }
        ],
    }
    chat_first = {
        "model": "served-model",
        "messages": [{"role": "user", "content": "continue"}],
        "stream": False,
        "previous_response_id": "chat-one",
    }
    chat_second = {
        **chat_first,
        "previous_response_id": "chat-two",
    }

    nested_public_first = matrix._request_public(
        2,
        nested_first,
        protocol="responses",
    )
    nested_public_second = matrix._request_public(
        2,
        nested_second,
        protocol="responses",
    )
    chat_public_first = matrix._request_public(2, chat_first, protocol="chat")
    chat_public_second = matrix._request_public(2, chat_second, protocol="chat")

    assert (
        nested_public_first["canonical_body_sha256"]
        != nested_public_second["canonical_body_sha256"]
    )
    assert (
        chat_public_first["canonical_body_sha256"]
        != chat_public_second["canonical_body_sha256"]
    )


def test_first_tool_prompt_is_base_independent():
    prompt = matrix.first_tool_instruction("anthropic", "stream")

    assert prompt == matrix.first_tool_instruction("anthropic", "stream")
    assert "direct" not in prompt.lower()
    assert "gateway" not in prompt.lower()
    assert "agentic/" not in prompt.lower()


def test_build_request_uses_native_tool_choice_shapes_without_hosts():
    chat = matrix.build_request(
        "chat",
        "served-model",
        "stream",
        1,
        history=[{"role": "user", "content": "x"}],
        instructions="x",
    )
    responses = matrix.build_request(
        "responses",
        "served-model",
        "stream",
        1,
        history="x",
        instructions="x",
    )
    anthropic = matrix.build_request(
        "anthropic",
        "served-model",
        "nonstream",
        1,
        history=[{"role": "user", "content": "x"}],
        instructions="x",
    )
    ollama = matrix.build_request(
        "ollama",
        "served-model",
        "stream",
        1,
        history=[{"role": "user", "content": "x"}],
        instructions="x",
    )

    assert chat["tool_choice"] == {
        "type": "function",
        "function": {"name": "file_info"},
    }
    assert responses["tool_choice"] == {"type": "function", "name": "file_info"}
    assert anthropic["tool_choice"] == {"type": "any"}
    assert "tool_choice" not in ollama
    assert [tool["function"]["name"] for tool in ollama["tools"]] == ["file_info"]

    assert matrix.tool_choice("chat", "stream", 2, "explicit") == {
        "type": "function",
        "function": {"name": "run_command"},
    }
    assert matrix.tool_choice("responses", "stream", 2, "explicit") == {
        "type": "function",
        "name": "run_command",
    }
    assert matrix.tool_choice("anthropic", "stream", 2, "explicit") == {
        "type": "tool",
        "name": "run_command",
    }
    assert matrix.tool_choice("chat", "stream", 1, "auto", "auto") == "auto"
    assert matrix.tool_choice("responses", "stream", 1, "auto", "auto") == "auto"
    assert matrix.tool_choice("anthropic", "stream", 1, "auto", "auto") == {
        "type": "auto"
    }


def test_import_safe_parser_requires_caller_supplied_model_and_base(tmp_path: Path):
    parser = matrix.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--output", str(tmp_path / "out.json")])

    args = parser.parse_args(
        [
            "--base-url",
            "direct=http://127.0.0.1:8000",
            "--base-url",
            "gateway=http://127.0.0.1:8088",
            "--model",
            "served-model",
            "--bundle-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "out.json"),
            "--source-head",
            "observed-at-run-time",
            "--installed-release-manifest",
            str(tmp_path / "installed-release.json"),
            "--run-id",
            "paired-proof-run",
        ]
    )
    assert args.model == "served-model"
    assert args.bundle_root == tmp_path
    assert args.base_url == [
        "direct=http://127.0.0.1:8000",
        "gateway=http://127.0.0.1:8088",
    ]
    assert args.raw_artifact_dir is None
    assert args.source_head == "observed-at-run-time"
    assert args.installed_release_manifest == tmp_path / "installed-release.json"
    assert args.run_id == "paired-proof-run"


@pytest.mark.parametrize("run_id", ["", "bad id", "x" * 81])
def test_run_matrix_rejects_non_binding_run_ids(run_id):
    with pytest.raises(ValueError, match="--run-id must be a nonempty"):
        matrix.run_matrix(SimpleNamespace(run_id=run_id))


def test_raw_capture_rejects_worktree_and_symlink_escape(
    tmp_path: Path,
):
    worktree = tmp_path / "repo"
    worktree.mkdir()

    with pytest.raises(ValueError, match="outside every Git worktree"):
        matrix.DecompressedParserInputCaptureRecorder(
            worktree / "captures",
            worktree,
            run_id="rejected",
        )
    assert not (worktree / "captures").exists()

    link = tmp_path / "repo-link"
    link.symlink_to(worktree, target_is_directory=True)
    with pytest.raises(ValueError, match="outside every Git worktree"):
        matrix.DecompressedParserInputCaptureRecorder(
            link / "captures",
            worktree,
            run_id="rejected-link",
        )


def test_raw_capture_rejects_apfs_case_alias_and_sibling_git_worktree(
    tmp_path: Path,
):
    worktree = tmp_path / "GuardedRepo"
    worktree.mkdir()
    case_alias = tmp_path / "guardedrepo"
    if case_alias.exists() and case_alias.samefile(worktree):
        with pytest.raises(ValueError, match="outside every Git worktree"):
            matrix.DecompressedParserInputCaptureRecorder(
                case_alias / "captures",
                worktree,
                run_id="rejected-case-alias",
            )
        assert not (worktree / "captures").exists()

    sibling_repo = tmp_path / "sibling-public-repo"
    subprocess.run(
        ["git", "init", "-q", str(sibling_repo)],
        check=True,
        capture_output=True,
    )
    with pytest.raises(ValueError, match="outside every Git worktree"):
        matrix.DecompressedParserInputCaptureRecorder(
            sibling_repo / "captures",
            worktree,
            run_id="rejected-sibling",
        )
    assert not (sibling_repo / "captures").exists()


def test_opt_in_capture_preserves_parser_input_bytes_and_allowlists_metadata(
    monkeypatch,
    tmp_path: Path,
):
    secret = "SECRET-AUTH-VALUE"
    worktree = tmp_path / "repo"
    worktree.mkdir()
    capture_root = tmp_path / "private-captures"
    response_body = (
        b'data: {"id":"chat_1","choices":[{"delta":'
        b'{"reasoning_content":"private"},"finish_reason":null}]}\r\n\r\n'
        b'data: {"id":"chat_1","choices":[{"delta":'
        b'{"content":"$43 \\\\times 2"},"finish_reason":"stop"}]}\r\n\r\n'
        b"data: [DONE]\r\n\r\n"
    )

    class FakeRaw:
        def __init__(self):
            self.headers = {
                "Content-Type": "text/event-stream",
                "Set-Cookie": f"session={secret}",
                "X-Trace-Id": "trace-safe",
                "Location": f"https://example.invalid/callback?token={secret}",
            }
            self.closed = False

        def stream(self, _amount, decode_content=None):
            assert decode_content is True
            yield response_body[:37]
            yield response_body[37:91]
            yield response_body[91:]

        def close(self):
            self.closed = True

    payload = {
        "model": "served-model",
        "messages": [{"role": "user", "content": "private prompt"}],
        "stream": True,
        "max_tokens": 32,
    }
    prepared = requests.Request(
        "POST",
        "http://user:password@127.0.0.1:8000/v1/chat/completions?key=secret",
        headers={
            "Authorization": f"Bearer {secret}",
            "X-Api-Key": secret,
            "Content-Type": "application/json",
        },
        json=payload,
    ).prepare()
    response = requests.Response()
    response.status_code = 200
    response.reason = "OK"
    response.raw = FakeRaw()
    response.request = prepared
    response.encoding = "utf-8"

    def fake_post(*_args, **_kwargs):
        return response

    monkeypatch.setattr(matrix.requests, "post", fake_post)
    recorder = matrix.DecompressedParserInputCaptureRecorder(
        capture_root,
        worktree,
        run_id="unit-test",
    )
    recorder.configure_expected([("direct", "chat", "stream-round1")])
    client = matrix.ProtocolClient(
        "http://127.0.0.1:8000",
        secret,
        30,
        base_label="direct",
        raw_recorder=recorder,
    )

    result = client.send(
        "chat",
        payload,
        True,
        capture_label="stream-round1",
    )

    assert result["reasoning"] == "private"
    assert result["content"] == "$43 \\times 2"
    capture_summary = recorder.finalize()
    body_path = next(recorder.run_dir.glob("*.decompressed-parser-input.bin"))
    metadata_path = next(recorder.run_dir.glob("*.metadata.json"))
    assert body_path.read_bytes() == response_body
    metadata_text = metadata_path.read_text()
    assert secret not in metadata_text
    assert "password" not in metadata_text
    assert "?key=secret" not in metadata_text

    metadata = json.loads(metadata_text)
    assert metadata["capture_layer"] == matrix.CAPTURE_LAYER
    assert metadata["capture_semantics"] == matrix.CAPTURE_SEMANTICS
    assert metadata["request"]["url"] == ("http://127.0.0.1:8000/v1/chat/completions")
    assert metadata["request"]["body_bytes"] == len(prepared.body)
    assert (
        metadata["request"]["body_sha256"] == hashlib.sha256(prepared.body).hexdigest()
    )
    assert metadata["request"]["prepared_payload_body_sha256"] == (
        metadata["request"]["payload"]["body_sha256"]
    )
    assert metadata["request"][
        "prepared_payload_canonical_body_sha256"
    ] == metadata["request"]["payload"]["canonical_body_sha256"]
    assert metadata["request"]["payload"]["top_level_fields"] == [
        "max_tokens",
        "messages",
        "model",
        "stream",
    ]
    assert {
        row["name"].lower(): row["value"] for row in metadata["request"]["headers"]
    }["authorization"] == "<redacted>"
    assert {
        row["name"].lower(): row["value"] for row in metadata["response"]["headers"]
    }["set-cookie"] == "<redacted>"
    assert {
        row["name"].lower(): row["value"] for row in metadata["response"]["headers"]
    }["location"] == "<redacted>"
    assert {
        row["name"].lower(): row["value"] for row in metadata["response"]["headers"]
    }["x-trace-id"] == "<redacted>"
    assert {
        row["name"].lower(): row["value"] for row in metadata["response"]["headers"]
    }["content-type"] == "text/event-stream"
    assert metadata["response"]["status_code"] == 200
    route = capture_summary["routes"][0]
    assert (
        route["base_label"],
        route["protocol"],
        route["capture_label"],
    ) == ("direct", "chat", "stream-round1")
    artifact = route["artifacts"][0]
    assert artifact["verified"] is True
    assert artifact["route_bound"] is True
    assert artifact["body_sha256"] == hashlib.sha256(response_body).hexdigest()
    assert artifact["body_bytes"] == len(response_body)
    assert artifact["metadata_sha256"] == hashlib.sha256(
        metadata_path.read_bytes()
    ).hexdigest()
    assert artifact["request_body_sha256"] == hashlib.sha256(
        prepared.body
    ).hexdigest()
    assert metadata["response"]["body_bytes"] == len(response_body)
    assert (
        metadata["response"]["body_sha256"] == hashlib.sha256(response_body).hexdigest()
    )
    assert metadata["response"]["first_byte_ms"] is not None
    assert (
        metadata["response"]["completed_ms"] >= (metadata["response"]["first_byte_ms"])
    )
    assert stat.S_IMODE(body_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(metadata_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(recorder.manifest_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(recorder.run_dir.stat().st_mode) == 0o700
    assert capture_summary["expected"] == 1
    assert capture_summary["started"] == 1
    assert capture_summary["finished"] == 1
    assert capture_summary["errors"] == 0
    assert capture_summary["complete"] is True


def test_responses_nonstream_capture_retains_exact_missing_status_bytes(
    monkeypatch,
    tmp_path: Path,
):
    worktree = tmp_path / "repo"
    worktree.mkdir()
    response_body = json.dumps(
        {
            "id": "response-without-status",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "visible"}],
                }
            ],
        },
        separators=(",", ":"),
    ).encode()
    payload = {
        "model": "served-model",
        "input": "private prompt",
        "stream": False,
    }
    prepared = requests.Request(
        "POST",
        "http://127.0.0.1:8000/v1/responses",
        json=payload,
    ).prepare()
    response = requests.Response()
    response.status_code = 200
    response.request = prepared
    response.encoding = "utf-8"
    response.raw = SimpleNamespace(
        headers={"Content-Type": "application/json"},
        close=lambda: None,
    )
    response._content = response_body
    response._content_consumed = True
    monkeypatch.setattr(matrix.requests, "post", lambda *_args, **_kwargs: response)

    recorder = matrix.DecompressedParserInputCaptureRecorder(
        tmp_path / "private-captures",
        worktree,
        run_id="responses-nonstream-missing-status",
    )
    recorder.configure_expected(
        [("direct", "responses", "nonstream-flow-round1")]
    )
    client = matrix.ProtocolClient(
        "http://127.0.0.1:8000",
        None,
        30,
        base_label="direct",
        raw_recorder=recorder,
    )

    result = client.send(
        "responses",
        payload,
        False,
        capture_label="nonstream-flow-round1",
    )
    summary = recorder.finalize()
    body_path = next(recorder.run_dir.glob("*.decompressed-parser-input.bin"))
    metadata = json.loads(
        next(recorder.run_dir.glob("*.metadata.json")).read_text()
    )

    assert body_path.read_bytes() == response_body
    assert result["terminals"] == []
    assert [error["kind"] for error in result["errors"]] == [
        "responses.missing_status"
    ]
    assert metadata["request"]["prepared_payload_body_sha256"] == (
        metadata["request"]["payload"]["body_sha256"]
    )
    assert metadata["request"][
        "prepared_payload_canonical_body_sha256"
    ] == metadata["request"]["payload"]["canonical_body_sha256"]
    assert summary["complete"] is True


def test_capture_rejects_prepared_body_that_differs_from_declared_payload(
    tmp_path: Path,
):
    worktree = tmp_path / "repo"
    worktree.mkdir()
    recorder = matrix.DecompressedParserInputCaptureRecorder(
        tmp_path / "private-captures",
        worktree,
        run_id="prepared-body-mismatch",
    )
    recorder.configure_expected([("direct", "chat", "stream-round1")])
    response = requests.Response()
    response.status_code = 200
    response.raw = SimpleNamespace(headers={"Content-Type": "text/event-stream"})
    response.request = requests.Request(
        "POST",
        "http://127.0.0.1:8000/v1/chat/completions",
        json={"model": "served-model", "stream": True, "temperature": 0.9},
    ).prepare()

    with pytest.raises(
        ValueError,
        match="prepared request body does not match",
    ):
        recorder.begin(
            base_label="direct",
            protocol="chat",
            capture_label="stream-round1",
            payload={
                "model": "served-model",
                "stream": True,
                "temperature": 0.1,
            },
            response=response,
            started=time.monotonic(),
            started_at="2026-07-24T00:00:00+00:00",
        )

    assert recorder.finalize()["complete"] is False


@pytest.mark.parametrize("base_label", ["direct", "gateway"])
@pytest.mark.parametrize(
    ("protocol", "response_body"),
    [
        (
            "chat",
            b'data: {"choices":[{"delta":{"content":"ok"},'
            b'"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
        ),
        (
            "responses",
            b"event: response.output_text.delta\n"
            b'data: {"type":"response.output_text.delta","delta":"ok"}\n\n'
            b"event: response.completed\n"
            b'data: {"type":"response.completed","response":{"id":"r1"}}\n\n',
        ),
        (
            "anthropic",
            b"event: content_block_delta\n"
            b'data: {"type":"content_block_delta","index":0,"delta":'
            b'{"type":"text_delta","text":"ok"}}\n\n'
            b"event: message_delta\n"
            b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\n'
            b"event: message_stop\n"
            b'data: {"type":"message_stop"}\n\n',
        ),
        (
            "ollama",
            b'{"message":{"content":"ok"},"done":true,"done_reason":"stop"}\n',
        ),
    ],
)
def test_parser_input_capture_covers_each_protocol_and_base(
    monkeypatch,
    tmp_path: Path,
    base_label: str,
    protocol: str,
    response_body: bytes,
):
    worktree = tmp_path / "repo"
    worktree.mkdir()

    class FakeRaw:
        headers = {"Content-Type": "application/octet-stream"}

        def stream(self, _amount, decode_content=None):
            assert decode_content is True
            midpoint = len(response_body) // 2
            yield response_body[:midpoint]
            yield response_body[midpoint:]

        def close(self):
            return None

    payload = {"model": "served-model", "stream": True}
    prepared = requests.Request(
        "POST",
        f"http://127.0.0.1:8000{matrix.ProtocolClient.route(protocol)}",
        json=payload,
    ).prepare()
    response = requests.Response()
    response.status_code = 200
    response.reason = "OK"
    response.raw = FakeRaw()
    response.request = prepared
    response.encoding = "utf-8"
    monkeypatch.setattr(matrix.requests, "post", lambda *_args, **_kwargs: response)

    recorder = matrix.DecompressedParserInputCaptureRecorder(
        tmp_path / "private-captures",
        worktree,
        run_id=f"{base_label}-{protocol}",
    )
    recorder.configure_expected([(base_label, protocol, "stream-flow-round1")])
    client = matrix.ProtocolClient(
        "http://127.0.0.1:8000",
        None,
        30,
        base_label=base_label,
        raw_recorder=recorder,
    )

    result = client.send(
        protocol,
        payload,
        True,
        capture_label="stream-flow-round1",
    )

    assert result["content"] == "ok"
    summary = recorder.finalize()
    body_path = next(recorder.run_dir.glob("*.decompressed-parser-input.bin"))
    metadata_path = next(recorder.run_dir.glob("*.metadata.json"))
    assert body_path.read_bytes() == response_body
    metadata = json.loads(metadata_path.read_text())
    assert metadata["base_label"] == base_label
    assert metadata["protocol"] == protocol
    assert metadata["request"]["payload"]["model"] == "served-model"
    assert summary["complete"] is True


def test_capture_setup_failure_closes_response(monkeypatch):
    class FailingRecorder:
        def begin(self, **_kwargs):
            raise OSError("capture unavailable")

    response = requests.Response()
    response.status_code = 200
    response.raw = SimpleNamespace()
    response.request = requests.Request(
        "POST",
        "http://127.0.0.1:8000/v1/chat/completions",
        json={"model": "served-model", "stream": True},
    ).prepare()
    closed = False

    def close_response():
        nonlocal closed
        closed = True

    response.close = close_response
    monkeypatch.setattr(
        matrix.requests,
        "post",
        lambda *_args, **_kwargs: response,
    )
    client = matrix.ProtocolClient(
        "http://127.0.0.1:8000",
        None,
        30,
        raw_recorder=FailingRecorder(),
    )

    with pytest.raises(OSError, match="capture unavailable"):
        client.send(
            "chat",
            {"model": "served-model", "stream": True},
            True,
        )
    assert closed is True


def test_abort_capture_setup_failure_closes_response(monkeypatch):
    class FailingRecorder:
        def begin(self, **_kwargs):
            raise OSError("capture unavailable")

    response = requests.Response()
    response.status_code = 200
    response.raw = SimpleNamespace()
    response.request = requests.Request(
        "POST",
        "http://127.0.0.1:8000/v1/chat/completions",
        json={"model": "served-model", "stream": True},
    ).prepare()
    closed = False

    def close_response():
        nonlocal closed
        closed = True

    response.close = close_response
    monkeypatch.setattr(
        matrix.requests,
        "post",
        lambda *_args, **_kwargs: response,
    )
    client = matrix.ProtocolClient(
        "http://127.0.0.1:8000",
        None,
        30,
        raw_recorder=FailingRecorder(),
    )

    with pytest.raises(OSError, match="capture unavailable"):
        matrix.abort_stream_after_deltas(
            client,
            "chat",
            {
                "model": "served-model",
                "messages": [{"role": "user", "content": "final"}],
                "stream": True,
            },
            health_url="http://127.0.0.1:8000/health",
            minimum_deltas=1,
        )
    assert closed is True


def test_capture_session_setup_failure_removes_partial_body(tmp_path: Path):
    class ExplodingHeaders:
        def items(self):
            raise RuntimeError("header enumeration failed")

    worktree = tmp_path / "repo"
    worktree.mkdir()
    recorder = matrix.DecompressedParserInputCaptureRecorder(
        tmp_path / "private-captures",
        worktree,
        run_id="setup-failure",
    )
    recorder.configure_expected([("direct", "chat", "stream-round1")])
    response = requests.Response()
    response.status_code = 200
    response.raw = SimpleNamespace(headers=ExplodingHeaders())
    response.request = requests.Request(
        "POST",
        "http://127.0.0.1:8000/v1/chat/completions",
        json={"model": "served-model", "stream": True},
    ).prepare()

    with pytest.raises(RuntimeError, match="header enumeration failed"):
        recorder.begin(
            base_label="direct",
            protocol="chat",
            capture_label="stream-round1",
            payload={"model": "served-model", "stream": True},
            response=response,
            started=time.monotonic(),
            started_at="2026-07-24T00:00:00+00:00",
        )

    assert not list(recorder.run_dir.glob("*.decompressed-parser-input.bin"))
    assert not list(recorder.run_dir.glob("*.metadata.json"))
    summary = recorder.finalize()
    assert summary["started"] == 1
    assert summary["finished"] == 0
    assert summary["errors"] == 1
    assert summary["complete"] is False


def test_capture_finish_failure_removes_body_and_partial_metadata(
    monkeypatch,
    tmp_path: Path,
):
    worktree = tmp_path / "repo"
    worktree.mkdir()
    recorder = matrix.DecompressedParserInputCaptureRecorder(
        tmp_path / "private-captures",
        worktree,
        run_id="finish-failure",
    )
    recorder.configure_expected([("direct", "chat", "stream-round1")])
    response = requests.Response()
    response.status_code = 200
    response.raw = SimpleNamespace(headers={"Content-Type": "text/event-stream"})
    response.request = requests.Request(
        "POST",
        "http://127.0.0.1:8000/v1/chat/completions",
        json={"model": "served-model", "stream": True},
    ).prepare()
    capture = recorder.begin(
        base_label="direct",
        protocol="chat",
        capture_label="stream-round1",
        payload={"model": "served-model", "stream": True},
        response=response,
        started=time.monotonic(),
        started_at="2026-07-24T00:00:00+00:00",
    )
    capture.write(b"sensitive decompressed parser input")
    artifact = recorder.current_summary()["routes"][0]["artifacts"][0]
    body_path = recorder.run_dir / artifact["body_file"]
    metadata_path = recorder.run_dir / artifact["metadata_file"]
    real_fdopen = matrix.os.fdopen

    class FailingMetadataFile:
        def __init__(self, descriptor: int, mode: str) -> None:
            self._file = real_fdopen(descriptor, mode)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self._file.close()

        def write(self, data: bytes) -> None:
            self._file.write(data[:8])
            self._file.flush()
            raise OSError("metadata write failed")

    monkeypatch.setattr(matrix.os, "fdopen", FailingMetadataFile)
    with pytest.raises(OSError, match="metadata write failed"):
        capture.finish(completed_ms=1.0)

    assert not body_path.exists()
    assert not metadata_path.exists()
    monkeypatch.setattr(matrix.os, "fdopen", real_fdopen)
    summary = recorder.finalize()
    assert summary["started"] == 1
    assert summary["finished"] == 0
    assert summary["errors"] == 1
    assert summary["complete"] is False


def _synthetic_terminals(protocol: str, *, expect_tool: bool) -> list[str]:
    if protocol == "chat":
        return ["tool_calls" if expect_tool else "stop", "DONE"]
    if protocol == "responses":
        return ["response.completed"]
    if protocol == "anthropic":
        return ["tool_use" if expect_tool else "end_turn", "message_stop"]
    return ["tool_calls" if expect_tool else "stop"]


def _finish_synthetic_capture(
    client,
    protocol: str,
    capture_label: str,
    payload: dict,
    *,
    omitted_label: str | None,
) -> None:
    recorder = client.raw_recorder
    if recorder is None or capture_label == omitted_label:
        return
    response = requests.Response()
    response.status_code = 200
    response.raw = SimpleNamespace(headers={"Content-Type": "text/event-stream"})
    response.request = requests.Request(
        "POST",
        client.base_url + client.route(protocol),
        json=payload,
    ).prepare()
    capture = recorder.begin(
        base_label=client.base_label,
        protocol=protocol,
        capture_label=capture_label,
        payload=payload,
        response=response,
        started=time.monotonic(),
        started_at="2026-07-24T00:00:00+00:00",
    )
    capture.write(f"{client.base_label}/{protocol}/{capture_label}\n".encode())
    capture.finish(completed_ms=1.0)


@pytest.mark.parametrize("failure_mode", ["declared-head", "dirty"])
def test_run_matrix_fails_closed_before_generation_on_unobserved_source_identity(
    monkeypatch,
    tmp_path: Path,
    failure_mode: str,
):
    repo_root, source = _identity_repo(tmp_path)
    bundle_root, bundle = _identity_bundle(tmp_path)
    if failure_mode == "dirty":
        (repo_root / matrix.FILE_INFO_PATH).write_text('{"dirty":true}\n')
        source = matrix.observe_source_checkout(repo_root)
    health = _identity_health(source, bundle=bundle)
    monkeypatch.setattr(
        matrix,
        "observe_runner_environment",
        lambda _repo_root: copy.deepcopy(_identity_runner()),
    )
    monkeypatch.setattr(
        matrix,
        "_get_full_health",
        lambda _url, _timeout: copy.deepcopy(health),
    )

    def generation_must_not_run(*_args, **_kwargs):
        raise AssertionError("generation ran before provenance passed")

    monkeypatch.setattr(matrix.ProtocolClient, "send", generation_must_not_run)
    declared_head = (
        "0" * len(source["head"])
        if failure_mode == "declared-head"
        else source["head"]
    )
    args = matrix.build_parser().parse_args(
        [
            "--base-url",
            "direct=http://direct.invalid",
            "--base-url",
            "gateway=http://gateway.invalid",
            "--model",
            bundle["model_name"],
            "--bundle-root",
            str(bundle_root),
            "--repo-root",
            str(repo_root),
            "--output",
            str(tmp_path / "result.json"),
            "--raw-artifact-dir",
            str(tmp_path / "private-captures"),
            "--source-head",
            declared_head,
            "--run-id",
            "identity-failure-run",
            "--protocol",
            "chat",
            "--mode",
            "nonstream",
            "--skip-cancellation",
        ]
    )

    result = matrix.run_matrix(args)

    assert result["schema"] == "vmlx-agentic-protocol-matrix-v2"
    assert result["pass"] is False
    assert result["checks"]["identity_provenance_pass"] is False
    assert result["backend_identity_fingerprint_sha256"] is None
    assert result["flows"] == {}
    joined = "\n".join(result["identity"]["failures"])
    expected = (
        "--source-head does not match"
        if failure_mode == "declared-head"
        else "source checkout is dirty"
    )
    assert expected in joined


def test_run_matrix_fails_closed_when_direct_and_gateway_are_different_backends(
    monkeypatch,
    tmp_path: Path,
):
    repo_root, source = _identity_repo(tmp_path)
    bundle_root, bundle = _identity_bundle(tmp_path)
    direct_health = _identity_health(source, pid=111, bundle=bundle)
    gateway_health = _identity_health(source, pid=222, bundle=bundle)
    monkeypatch.setattr(
        matrix,
        "observe_runner_environment",
        lambda _repo_root: copy.deepcopy(_identity_runner()),
    )

    def fake_health(url, _timeout):
        return copy.deepcopy(
            direct_health if "direct.invalid" in url else gateway_health
        )

    monkeypatch.setattr(matrix, "_get_full_health", fake_health)
    monkeypatch.setattr(
        matrix.ProtocolClient,
        "send",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("generation ran against mismatched backends")
        ),
    )
    args = matrix.build_parser().parse_args(
        [
            "--base-url",
            "direct=http://direct.invalid",
            "--base-url",
            "gateway=http://gateway.invalid",
            "--model",
            bundle["model_name"],
            "--bundle-root",
            str(bundle_root),
            "--repo-root",
            str(repo_root),
            "--output",
            str(tmp_path / "result.json"),
            "--raw-artifact-dir",
            str(tmp_path / "private-captures"),
            "--source-head",
            source["head"],
            "--run-id",
            "backend-mismatch-run",
            "--protocol",
            "chat",
            "--mode",
            "nonstream",
            "--skip-cancellation",
        ]
    )

    result = matrix.run_matrix(args)

    assert result["pass"] is False
    assert result["checks"]["identity_provenance_pass"] is False
    assert result["backend_identity_fingerprint_sha256"] is None
    assert any(
        "backend runtime/model/cache identity differs" in failure
        for failure in result["identity"]["failures"]
    )
    assert (
        result["identity"]["health"]["direct"]["before"]["identity"][
            "backend_pid"
        ]
        == 111
    )
    assert (
        result["identity"]["health"]["gateway"]["before"]["identity"][
            "backend_pid"
        ]
        == 222
    )


@pytest.mark.parametrize(
    "failure_mode",
    ["wrong-model", "wrong-bundle", "wrong-python", "same-origin"],
)
def test_run_matrix_rejects_cross_surface_identity_substitution(
    monkeypatch,
    tmp_path: Path,
    failure_mode: str,
):
    repo_root, source = _identity_repo(tmp_path)
    bundle_root, bundle = _identity_bundle(tmp_path)
    health = _identity_health(source, bundle=bundle)
    runner = _identity_runner()
    requested_model = bundle["model_name"]
    direct_url = "http://direct.invalid"
    gateway_url = "http://gateway.invalid"

    if failure_mode == "wrong-model":
        requested_model = "served-org/different-model"
    elif failure_mode == "wrong-bundle":
        forged = copy.deepcopy(health["model_bundle_provenance"])
        forged["files"]["config.json"]["sha256"] = "9" * 64
        observed = {
            "schema": forged["schema"],
            "directory_state": forged["directory_state"],
            "files": forged["files"],
        }
        forged["aggregate_sha256"] = matrix._canonical_sha256(observed)
        forged["fingerprint_sha256"] = forged["aggregate_sha256"]
        health["model_bundle_provenance"] = forged
        health["runtime_provenance"]["model_bundle_provenance"] = copy.deepcopy(
            forged
        )
    elif failure_mode == "wrong-python":
        runner["python_executable_fingerprint_sha256"] = "7" * 64
        runner["checkout_python_invocation_fingerprints_sha256"] = ["7" * 64]
    elif failure_mode == "same-origin":
        gateway_url = direct_url

    monkeypatch.setattr(
        matrix,
        "observe_runner_environment",
        lambda _repo_root: copy.deepcopy(runner),
    )
    monkeypatch.setattr(
        matrix,
        "_get_full_health",
        lambda _url, _timeout: copy.deepcopy(health),
    )
    monkeypatch.setattr(
        matrix.ProtocolClient,
        "send",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("generation ran before identity substitution failed")
        ),
    )
    args = matrix.build_parser().parse_args(
        [
            "--base-url",
            f"direct={direct_url}",
            "--base-url",
            f"gateway={gateway_url}",
            "--model",
            requested_model,
            "--bundle-root",
            str(bundle_root),
            "--repo-root",
            str(repo_root),
            "--output",
            str(tmp_path / f"{failure_mode}.json"),
            "--raw-artifact-dir",
            str(tmp_path / f"{failure_mode}-private-captures"),
            "--source-head",
            source["head"],
            "--run-id",
            f"identity-substitution-{failure_mode}",
            "--protocol",
            "chat",
            "--mode",
            "nonstream",
            "--skip-cancellation",
        ]
    )

    result = matrix.run_matrix(args)

    assert result["pass"] is False
    assert result["flows"] == {}
    failures = "\n".join(result["identity"]["failures"])
    expected = {
        "wrong-model": "requested --model does not match",
        "wrong-bundle": "bundle fingerprint does not match",
        "wrong-python": "Python executable does not match",
        "same-origin": "direct and gateway must be distinct",
    }[failure_mode]
    assert expected in failures


def test_health_capture_rejects_cross_origin_substitution_before_fetch(
    monkeypatch,
):
    fetched = False

    def unexpected_fetch(_url, _timeout):
        nonlocal fetched
        fetched = True
        raise AssertionError("cross-origin health must not be fetched")

    monkeypatch.setattr(matrix, "_get_full_health", unexpected_fetch)
    evidence, failures = matrix._capture_health_evidence(
        {"gateway": "http://127.0.0.1:8080"},
        {"gateway": "http://127.0.0.1:8000/health"},
        30,
        "served-org/served-model",
    )

    assert fetched is False
    assert evidence["gateway"]["error_type"] == "HealthOriginMismatch"
    assert failures == [
        "gateway: /health URL origin does not match its corresponding request base"
    ]


def test_gateway_health_binds_topology_to_direct_backend_identity(
    monkeypatch,
    tmp_path: Path,
):
    _, source = _identity_repo(tmp_path)
    _, bundle = _identity_bundle(tmp_path)
    backend_health = _identity_health(source, bundle=bundle)
    gateway_health = {
        "status": "ok",
        "backends": [
            {
                "model": bundle["model_name"],
                "port": 8000,
                "status": "running",
            }
        ],
    }

    def fake_health(url, _timeout):
        if url == "http://127.0.0.1:8080/health":
            return copy.deepcopy(gateway_health)
        if url == "http://127.0.0.1:8000/health":
            return copy.deepcopy(backend_health)
        raise AssertionError(f"unexpected health URL: {url}")

    monkeypatch.setattr(matrix, "_get_full_health", fake_health)
    evidence, failures = matrix._capture_health_evidence(
        {
            "direct": "http://127.0.0.1:8000",
            "gateway": "http://127.0.0.1:8080",
        },
        {
            "direct": "http://127.0.0.1:8000/health",
            "gateway": "http://127.0.0.1:8080/health",
        },
        30,
        bundle["model_name"],
    )

    assert failures == []
    assert evidence["gateway"]["full"] == gateway_health
    assert evidence["gateway"]["identity_full"] == backend_health
    assert (
        evidence["gateway"]["identity"]
        == evidence["direct"]["identity"]
    )


def test_run_matrix_rejects_result_output_inside_git_worktree(
    tmp_path: Path,
):
    repo_root, source = _identity_repo(tmp_path)
    bundle_root, bundle = _identity_bundle(tmp_path)
    output = repo_root / "build" / "private-result.json"
    args = matrix.build_parser().parse_args(
        [
            "--base-url",
            "direct=http://direct.invalid",
            "--base-url",
            "gateway=http://gateway.invalid",
            "--model",
            bundle["model_name"],
            "--bundle-root",
            str(bundle_root),
            "--repo-root",
            str(repo_root),
            "--output",
            str(output),
            "--raw-artifact-dir",
            str(tmp_path / "private-captures"),
            "--source-head",
            source["head"],
            "--run-id",
            "inside-worktree-output",
            "--protocol",
            "chat",
            "--mode",
            "nonstream",
            "--skip-cancellation",
        ]
    )

    with pytest.raises(ValueError, match="outside every Git worktree"):
        matrix.run_matrix(args)
    assert not output.exists()


def test_identity_comparison_detects_runtime_model_cache_and_source_drift(
    tmp_path: Path,
):
    repo_root, source_before = _identity_repo(tmp_path)
    source_after = copy.deepcopy(source_before)
    source_after["tree"] = "f" * len(source_after["tree"])
    before_health = _identity_health(source_before)
    before_identity, assert_no_failure = matrix._health_identity(before_health)
    assert assert_no_failure == []
    bundle_before = {
        **before_health["model_bundle_provenance"],
        "model_name": before_health["model_name"],
    }
    bundle_after = copy.deepcopy(bundle_before)
    bundle_after["files"]["config.json"]["sha256"] = "c" * 64
    bundle_observed = {
        "schema": bundle_after["schema"],
        "directory_state": bundle_after["directory_state"],
        "files": bundle_after["files"],
    }
    bundle_after["aggregate_sha256"] = matrix._canonical_sha256(bundle_observed)
    bundle_after["fingerprint_sha256"] = bundle_after["aggregate_sha256"]
    after_health = _identity_health(
        source_before,
        pid=5678,
        bundle=bundle_after,
        cache_fingerprint="d" * 64,
    )
    after_identity, assert_no_failure = matrix._health_identity(after_health)
    assert assert_no_failure == []

    failures = matrix._compare_identity_evidence(
        source_before,
        source_after,
        _identity_runner(),
        _identity_runner(),
        bundle_before,
        bundle_after,
        before_health["model_name"],
        {
            "direct": {"identity": before_identity},
            "gateway": {"identity": before_identity},
        },
        {
            "direct": {"identity": after_identity},
            "gateway": {"identity": after_identity},
        },
    )

    assert "source identity changed during the matrix: tree" in failures
    assert any(
        "backend runtime/model/cache identity differs" in failure
        for failure in failures
    )
    assert repo_root.is_dir()


@pytest.mark.parametrize(
    (
        "capture_enabled",
        "omitted_label",
        "expected_pass",
        "expected_started",
    ),
    [
        (True, None, True, 40),
        (True, "stream-flow-round2", False, 32),
    ],
)
def test_run_matrix_binds_full_stream_capture_manifest_into_pass(
    monkeypatch,
    tmp_path: Path,
    capture_enabled: bool,
    omitted_label: str | None,
    expected_pass: bool,
    expected_started: int,
):
    repo_root, source = _identity_repo(tmp_path)
    bundle_root, bundle = _identity_bundle(tmp_path)
    package_json = repo_root / matrix.FILE_INFO_PATH
    raw_root = tmp_path / "private-captures"
    size_human = matrix._human_size(package_json.stat().st_size)
    health = _identity_health(source, bundle=bundle)
    monkeypatch.setattr(
        matrix,
        "observe_runner_environment",
        lambda _repo_root: copy.deepcopy(_identity_runner()),
    )

    def fake_send(
        client,
        protocol,
        payload,
        stream,
        *,
        capture_label="request",
    ):
        assert stream is True
        _finish_synthetic_capture(
            client,
            protocol,
            capture_label,
            payload,
            omitted_label=omitted_label,
        )
        response_id = f"{client.base_label}-{protocol}-{capture_label}"
        result = {
            "status_code": 200,
            "elapsed_ms": 1.0,
            "response_id": response_id,
            "reasoning": "",
            "content": "",
            "tool_calls": [],
            "terminals": [],
            "errors": [],
            "events": [],
        }
        if capture_label == "stream-flow-round1":
            result["tool_calls"] = [
                {
                    "index": 0,
                    "id": f"{response_id}-call",
                    "name": "file_info",
                    "arguments": {"path": matrix.FILE_INFO_PATH},
                }
            ]
            result["terminals"] = _synthetic_terminals(
                protocol,
                expect_tool=True,
            )
        elif capture_label == "stream-flow-round2":
            result["tool_calls"] = [
                {
                    "index": 0,
                    "id": f"{response_id}-call",
                    "name": "run_command",
                    "arguments": {"command": matrix.PWD_COMMAND},
                }
            ]
            result["terminals"] = _synthetic_terminals(
                protocol,
                expect_tool=True,
            )
        elif capture_label == "stream-flow-round3":
            result["content"] = (
                f"AGENTIC-{protocol.upper()}-STREAM-DONE "
                f"SIZE={size_human} PWD={repo_root}"
            )
            result["terminals"] = _synthetic_terminals(
                protocol,
                expect_tool=False,
            )
            result["events"] = [
                {
                    "at_ms": 1.0,
                    "channel": "content",
                    "kind": "synthetic.delta.1",
                },
                {
                    "at_ms": 2.0,
                    "channel": "content",
                    "kind": "synthetic.delta.2",
                },
            ]
        elif capture_label == "stream-recovery":
            result["content"] = (
                f"RECOVERY-{client.base_label.upper()}-{protocol.upper()}-STREAM-DONE"
            )
            result["terminals"] = _synthetic_terminals(
                protocol,
                expect_tool=False,
            )
        else:
            raise AssertionError(f"unexpected capture label: {capture_label}")

        if protocol == "responses" and capture_label in {
            "stream-flow-round1",
            "stream-flow-round2",
        }:
            result["events"] = [
                {
                    "at_ms": 1.0,
                    "channel": "tool",
                    "kind": "response.output_item.added",
                },
                {
                    "at_ms": 2.0,
                    "channel": "tool",
                    "kind": "response.function_call_arguments.done",
                },
                {
                    "at_ms": 3.0,
                    "channel": "tool",
                    "kind": "response.output_item.done",
                },
            ]
        next_at_ms = max(
            (float(event.get("at_ms") or 0) for event in result["events"]),
            default=0.0,
        )
        for terminal in result["terminals"]:
            next_at_ms += 1.0
            result["events"].append(
                {
                    "at_ms": next_at_ms,
                    "channel": "terminal",
                    "kind": terminal,
                }
            )
        return result

    def fake_abort(
        client,
        protocol,
        payload,
        *,
        health_url,
        minimum_deltas,
    ):
        del health_url
        _finish_synthetic_capture(
            client,
            protocol,
            "stream-abort",
            payload,
            omitted_label=omitted_label,
        )
        return {
            "status_code": 200,
            "closed_at_ms": 1.0,
            "response_id": f"{client.base_label}-{protocol}-abort",
            "delta_events_before_abort": minimum_deltas,
            "cancel_status": (200 if protocol in {"chat", "responses"} else None),
            "cancel_body_sha256": "",
            "terminals_before_abort": [],
            "events": [],
            "idle_after_abort": {"idle": True},
        }

    monkeypatch.setattr(matrix.ProtocolClient, "send", fake_send)
    monkeypatch.setattr(
        matrix,
        "abort_stream_after_deltas",
        fake_abort,
    )
    monkeypatch.setattr(
        matrix,
        "_get_full_health",
        lambda _url, _timeout: copy.deepcopy(health),
    )
    argv = [
        "--base-url",
        "direct=http://direct.invalid",
        "--base-url",
        "gateway=http://gateway.invalid",
        "--model",
        bundle["model_name"],
        "--bundle-root",
        str(bundle_root),
        "--repo-root",
        str(repo_root),
        "--output",
        str(tmp_path / "result.json"),
        "--mode",
        "stream",
        "--no-enable-thinking",
        "--source-head",
        source["head"],
        "--run-id",
        "full-stream-matrix",
    ]
    if capture_enabled:
        argv.extend(("--raw-artifact-dir", str(raw_root)))
    args = matrix.build_parser().parse_args(argv)

    result = matrix.run_matrix(args)

    assert result["schema"] == matrix.OUTPUT_SCHEMA
    assert result["schema_version"] == 2
    assert result["run_id"] == "full-stream-matrix"
    assert result["checks"]["identity_provenance_pass"] is True
    expected_backend_identity, failures = matrix._health_identity(health)
    assert failures == []
    assert result["backend_identity_fingerprint_sha256"] == (
        expected_backend_identity["fingerprint_sha256"]
    )
    assert result["identity"]["source"]["before"] == source
    assert result["identity"]["source"]["after"] == source
    assert result["identity"]["health"]["direct"]["before"]["full"] == health
    assert result["identity"]["health"]["gateway"]["after"]["full"] == health
    assert result["checks"]["all_flows_pass"] is True
    assert result["checks"]["all_abort_recovery_pass"] is True
    assert result["raw_capture"]["enabled"] is capture_enabled
    assert result["raw_capture"]["started"] == expected_started
    assert result["pass"] is expected_pass
    assert result["raw_capture"]["run_id"] == result["run_id"]
    assert result["raw_capture"]["expected"] == 40
    assert result["raw_capture"]["finished"] == expected_started
    assert result["raw_capture"]["errors"] == 0
    assert result["checks"]["raw_capture_complete"] is expected_pass
    manifest_path = next(raw_root.glob("*/manifest.json"))
    assert result["raw_capture"]["manifest_path"] == str(manifest_path.resolve())
    assert result["raw_capture"]["run_directory"] == str(
        manifest_path.parent.resolve()
    )
    manifest_bytes = manifest_path.read_bytes()
    assert (
        hashlib.sha256(manifest_bytes).hexdigest()
        == (result["raw_capture"]["manifest_sha256"])
    )
    manifest = json.loads(manifest_bytes)
    assert manifest["expected"] == 40
    assert manifest["started"] == expected_started
    assert manifest["finished"] == expected_started
    assert manifest["complete"] is expected_pass
    if expected_pass:
        assert len(manifest["routes"]) == 40
        assert all(row["expected"] == 1 for row in manifest["routes"])
        assert all(row["started"] == 1 for row in manifest["routes"])
        assert all(row["finished"] == 1 for row in manifest["routes"])
        assert {row["capture_label"] for row in manifest["routes"]} == {
            "stream-flow-round1",
            "stream-flow-round2",
            "stream-flow-round3",
            "stream-abort",
            "stream-recovery",
        }
        artifacts = [
            artifact
            for route in result["raw_capture"]["routes"]
            for artifact in route["artifacts"]
        ]
        assert all(artifact["verified"] is True for artifact in artifacts)
        assert all(len(artifact["body_sha256"]) == 64 for artifact in artifacts)
        assert all(
            len(artifact["metadata_sha256"]) == 64 for artifact in artifacts
        )
        assert all(
            len(artifact["request_body_sha256"]) == 64
            for artifact in artifacts
        )


def test_stream_matrix_requires_private_raw_capture(tmp_path: Path):
    package_json = tmp_path / matrix.FILE_INFO_PATH
    package_json.parent.mkdir(parents=True)
    package_json.write_text("{}\n")
    args = SimpleNamespace(
        run_id="missing-raw-stream",
        base_url=[
            "direct=http://direct.invalid",
            "gateway=http://gateway.invalid",
        ],
        allow_single_base=False,
        health_url=[],
        repo_root=str(tmp_path),
        protocol=["chat"],
        mode=["stream"],
        raw_artifact_dir=None,
    )

    with pytest.raises(
        ValueError,
        match="--raw-artifact-dir is required whenever stream mode",
    ):
        matrix.run_matrix(args)


def test_nonstream_matrix_requires_private_raw_capture(tmp_path: Path):
    package_json = tmp_path / matrix.FILE_INFO_PATH
    package_json.parent.mkdir(parents=True)
    package_json.write_text("{}\n")
    args = SimpleNamespace(
        run_id="missing-raw-responses-nonstream",
        base_url=[
            "direct=http://direct.invalid",
            "gateway=http://gateway.invalid",
        ],
        allow_single_base=False,
        health_url=[],
        repo_root=str(tmp_path),
        protocol=["chat"],
        mode=["nonstream"],
        raw_artifact_dir=None,
    )

    with pytest.raises(
        ValueError,
        match="nonstream mode is requested",
    ):
        matrix.run_matrix(args)


def test_expected_parser_input_routes_include_nonresponses_nonstream():
    routes = matrix.expected_parser_input_capture_routes(
        ["direct", "gateway"],
        ["chat"],
        ["nonstream"],
        skip_cancellation=True,
    )
    assert routes == [
        ("direct", "chat", "nonstream-flow-round1"),
        ("direct", "chat", "nonstream-flow-round2"),
        ("direct", "chat", "nonstream-flow-round3"),
        ("gateway", "chat", "nonstream-flow-round1"),
        ("gateway", "chat", "nonstream-flow-round2"),
        ("gateway", "chat", "nonstream-flow-round3"),
    ]


def test_expected_parser_input_routes_include_nonstream_recovery():
    routes = matrix.expected_parser_input_capture_routes(
        ["direct", "gateway"],
        ["chat"],
        ["nonstream"],
        skip_cancellation=False,
    )
    assert routes == [
        ("direct", "chat", "nonstream-flow-round1"),
        ("direct", "chat", "nonstream-flow-round2"),
        ("direct", "chat", "nonstream-flow-round3"),
        ("direct", "chat", "nonstream-recovery"),
        ("gateway", "chat", "nonstream-flow-round1"),
        ("gateway", "chat", "nonstream-flow-round2"),
        ("gateway", "chat", "nonstream-flow-round3"),
        ("gateway", "chat", "nonstream-recovery"),
    ]


def test_gateway_abort_lifecycle_uses_bound_direct_backend_health():
    bases = {
        "direct": "http://127.0.0.1:8000",
        "gateway": "http://127.0.0.1:8080",
    }
    health_urls = {
        "direct": "http://127.0.0.1:8000/health/full",
        "gateway": "http://127.0.0.1:8080/health",
    }

    assert matrix.backend_lifecycle_health_url("direct", bases, health_urls) == (
        "http://127.0.0.1:8000/health/full"
    )
    assert matrix.backend_lifecycle_health_url("gateway", bases, health_urls) == (
        "http://127.0.0.1:8000/health/full"
    )


def test_nonstream_chat_capture_preserves_parser_input_bytes(
    monkeypatch,
    tmp_path: Path,
):
    worktree = tmp_path / "repo"
    worktree.mkdir()
    raw_root = tmp_path / "private-captures"
    payload = {
        "model": "served-model",
        "messages": [{"role": "user", "content": "private prompt"}],
        "stream": False,
        "max_tokens": 32,
    }
    response_body = (
        b'{"id":"chatcmpl_1","choices":[{"message":{"role":"assistant",'
        b'"reasoning_content":"private","content":"done"},'
        b'"finish_reason":"stop"}],"usage":{"completion_tokens":7}}'
    )
    prepared = requests.Request(
        "POST",
        "http://127.0.0.1:8000/v1/chat/completions",
        json=payload,
    ).prepare()
    response = requests.Response()
    response.status_code = 200
    response.reason = "OK"
    response._content = response_body
    response.request = prepared
    response.encoding = "utf-8"

    monkeypatch.setattr(matrix.requests, "post", lambda *_args, **_kwargs: response)
    recorder = matrix.DecompressedParserInputCaptureRecorder(
        raw_root,
        worktree,
        run_id="nonstream-chat",
    )
    recorder.configure_expected([("direct", "chat", "nonstream-round1")])
    client = matrix.ProtocolClient(
        "http://127.0.0.1:8000",
        None,
        30,
        base_label="direct",
        raw_recorder=recorder,
    )

    result = client.send("chat", payload, False, capture_label="nonstream-round1")

    assert result["content"] == "done"
    manifest = recorder.finalize()
    body_path = next(recorder.run_dir.glob("*.decompressed-parser-input.bin"))
    assert body_path.read_bytes() == response_body
    assert manifest["complete"] is True
    assert manifest["routes"][0]["protocol"] == "chat"
    assert manifest["routes"][0]["capture_label"] == "nonstream-round1"


def test_prepared_body_transport_is_byte_exact(monkeypatch):
    payload = {
        "model": "served-model",
        "messages": [{"role": "user", "content": "frozen"}],
        "stream": False,
    }
    body = matrix.frozen_request_body(payload)
    observed = {}

    def fake_post(url, **kwargs):
        observed.update(kwargs)
        response = requests.Response()
        response.status_code = 200
        response._content = (
            b'{"choices":[{"message":{"content":"done"},'
            b'"finish_reason":"stop"}]}'
        )
        response.encoding = "utf-8"
        response.request = requests.Request(
            "POST", url, headers=kwargs["headers"], data=kwargs["data"]
        ).prepare()
        return response

    monkeypatch.setattr(matrix.requests, "post", fake_post)
    result = matrix.ProtocolClient("http://direct.invalid", None, 30).send(
        "chat", payload, False, prepared_body=body
    )

    assert observed["data"] == body
    assert observed["stream"] is True
    assert result["content"] == "done"
    assert result["_prepared_request_body_sha256"] == hashlib.sha256(body).hexdigest()

    def mutating_post(url, **kwargs):
        response = fake_post(url, **kwargs)
        response.request = requests.Request(
            "POST", url, headers=kwargs["headers"], data=kwargs["data"] + b" "
        ).prepare()
        return response

    monkeypatch.setattr(matrix.requests, "post", mutating_post)
    with pytest.raises(ValueError, match="changed before transport"):
        matrix.ProtocolClient("http://direct.invalid", None, 30).send(
            "chat", payload, False, prepared_body=body
        )


def _paired_replay(
    monkeypatch,
    *,
    protocol="chat",
    mode="nonstream",
    stage=2,
    direct=("same", "same"),
    gateway="same",
    body_fault=None,
    lifecycle="exact",
    identity="match",
    payload=None,
    canary="",
    raw_override=None,
):
    lock = threading.Lock()
    response_id_b = "" if protocol == "ollama" else "response-b"
    backend_request_id = (
        "opaque-ollama-backend-request"
        if protocol == "ollama"
        else response_id_b
    )
    request_hash = matrix._sha256(backend_request_id)
    child_hash = matrix._sha256("response-b:visible-answer")
    foreign_hash = matrix._sha256("foreign-request")
    preexisting_hash = matrix._sha256("direct-a1-terminal-cleanup")
    activity = {
        "collector": [],
        "waiting": [],
        "running": [],
        "cleanup": False,
        "gateway_started": False,
        "probe_count": 0,
        "last_cache_execution_request_id": "response-a1",
    }
    if lifecycle in {"preexisting", "settling"}:
        activity.update(
            collector=[preexisting_hash],
            cleanup=True,
        )
    calls = []
    direct_index = 0

    def set_activity(
        *,
        collector=(),
        waiting=(),
        running=(),
        cleanup=False,
    ):
        with lock:
            activity.update(
                collector=list(collector),
                waiting=list(waiting),
                running=list(running),
                cleanup=cleanup,
            )

    class Client:
        def __init__(self, label):
            self.label = label

        def send(
            self,
            sent_protocol,
            _payload,
            sent_stream,
            *,
            capture_label,
            prepared_body,
        ):
            nonlocal direct_index
            leg = "b" if self.label == "gateway" else ("a1", "a2")[direct_index]
            if self.label == "gateway":
                with lock:
                    activity["gateway_started"] = True
                active_hash = (
                    child_hash
                    if lifecycle == "child_suffix"
                    else foreign_hash
                    if lifecycle == "foreign"
                    else request_hash
                )
                if lifecycle == "fast":
                    # The request completes between polls. The observer must
                    # not invent activity from the successful response alone.
                    pass
                elif lifecycle == "fast_completed":
                    # No active sample is observable, but the backend retains
                    # the exact admitted request ID after terminal cleanup.
                    with lock:
                        activity["last_cache_execution_request_id"] = (
                            backend_request_id
                        )
                elif lifecycle == "fast_foreign_completed":
                    with lock:
                        activity["last_cache_execution_request_id"] = (
                            "foreign-request"
                        )
                elif lifecycle == "inactive":
                    time.sleep(0.04)
                elif lifecycle == "non_atomic":
                    for phase in (
                        {"collector": [active_hash]},
                        {"waiting": [active_hash]},
                        {"running": [active_hash]},
                        {
                            "collector": [active_hash],
                            "running": [active_hash],
                        },
                    ):
                        set_activity(**phase)
                        time.sleep(0.012)
                elif lifecycle == "foreign_sequential":
                    set_activity(
                        collector=[active_hash],
                        running=[active_hash],
                    )
                    time.sleep(0.02)
                    set_activity(
                        collector=[foreign_hash],
                        running=[foreign_hash],
                    )
                    time.sleep(0.02)
                elif lifecycle == "collector_only":
                    set_activity(collector=[active_hash])
                    time.sleep(0.04)
                elif lifecycle == "running_only":
                    set_activity(running=[active_hash])
                    time.sleep(0.04)
                elif lifecycle == "concurrent":
                    set_activity(
                        collector=[active_hash, foreign_hash],
                        running=[active_hash, foreign_hash],
                    )
                    time.sleep(0.04)
                else:
                    set_activity(
                        collector=[active_hash],
                        running=[active_hash],
                    )
                    time.sleep(0.04)
                if lifecycle not in {
                    "inactive",
                    "fast",
                    "fast_completed",
                    "fast_foreign_completed",
                }:
                    set_activity(collector=[active_hash], cleanup=True)
                    time.sleep(0.012)
                set_activity()
                content = gateway
            else:
                content = direct[direct_index]
                direct_index += 1
            calls.append(
                (self.label, sent_protocol, sent_stream, capture_label, prepared_body)
            )
            digest = hashlib.sha256(prepared_body).hexdigest()
            if body_fault == leg:
                digest = "" if leg == "a2" else "0" * 64
            event = {
                "at_ms": 7,
                "channel": "tool",
                "kind": (
                    "ollama.tool" if sent_protocol == "ollama" else "chat.tool.complete"
                ),
                "call_id": f"volatile-{leg}",
            }
            error = {"channel": "error", "kind": "json_parse_error"}
            if canary:
                event["private_event"] = canary
                error["private_error"] = canary
            response = {
                "status_code": 200,
                "elapsed_ms": 7,
                "response_id": (
                    ""
                    if sent_protocol == "ollama"
                    else response_id_b
                    if leg == "b"
                    else f"response-{leg}"
                ),
                "reasoning": canary or "private reasoning",
                "content": canary or content,
                "notices": [canary or f"notice-{content}"],
                "tool_calls": [
                    {
                        "index": 0,
                        "id": f"volatile-{leg}",
                        "name": "run_command",
                        "arguments": {"command": canary or "pwd"},
                    }
                ],
                "terminals": ["stop"],
                "errors": [error],
                "events": [event],
                "_prepared_request_body_sha256": digest,
            }
            if raw_override:
                response.update(raw_override)
            return response

    expected = "f" * 64

    def fake_identity(health):
        if identity == "failure":
            return {}, ["identity invalid"]
        mismatch = identity == "mismatch" or (
            identity == "drift" and health.get("_gateway_active") is True
        )
        return {
            "fingerprint_sha256": "e" * 64 if mismatch else expected
        }, []

    def health_probe():
        if lifecycle == "exception":
            raise RuntimeError("probe failed")
        with lock:
            activity["probe_count"] += 1
            if (
                lifecycle == "settling"
                and not activity["gateway_started"]
                and activity["probe_count"] >= 2
            ):
                activity.update(
                    collector=[],
                    waiting=[],
                    running=[],
                    cleanup=False,
                )
            collector = list(activity["collector"])
            waiting = list(activity["waiting"])
            running = list(activity["running"])
            cleanup = bool(activity["cleanup"])
            gateway_active = bool(collector or waiting or running)
            last_cache_execution_request_id = activity[
                "last_cache_execution_request_id"
            ]
        active = sorted(set(collector) | set(waiting) | set(running))
        running_rows = [
            {"request_id_sha256": item, "status": "running"}
            for item in sorted(running)
        ]
        if lifecycle == "missing_running_rows":
            running_rows = []
        result = {
            "schema": "vmlx-request-lifecycle-v1",
            "request_id_encoding": "sha256-utf8-lowerhex",
            "available": True,
            "engine_collector_count": len(collector),
            "engine_collector_request_ids_sha256": sorted(collector),
            "scheduler_waiting_count": len(waiting),
            "scheduler_waiting_request_ids_sha256": sorted(waiting),
            "scheduler_running_count": len(running),
            "scheduler_running_request_ids_sha256": sorted(running),
            "scheduler_running_requests": running_rows,
            "active_request_count": len(active),
            "active_request_ids_sha256": active,
            "terminal_cleanup_pending": cleanup,
        }
        if lifecycle == "malformed":
            result["active_request_count"] = "one"
        elif lifecycle == "count_mismatch":
            result["engine_collector_count"] = len(collector) + 1
        elif lifecycle == "invalid_id":
            result["active_request_ids_sha256"] = ["not-a-digest"]
        elif lifecycle == "invalid_encoding":
            result["request_id_encoding"] = "raw"
        return {
            # These intentionally disagree with lifecycle v1. The observer
            # must not use cache-table or legacy scheduler counts.
            "scheduler": {
                "num_running": 99,
                "last_cache_execution": {
                    "request_id": last_cache_execution_request_id,
                    "cache_outcome": "hit",
                },
            },
            "cache": {"scheduler_cache": {"active_requests": 99}},
            "request_lifecycle": result,
            "_gateway_active": gateway_active,
        }

    monkeypatch.setattr(matrix, "_health_identity", fake_identity)
    result = matrix.run_paired_replay_discriminator(
        direct_client=Client("direct"),
        gateway_client=Client("gateway"),
        protocol=protocol,
        mode=mode,
        stage=stage,
        payload=payload
        or {
            "model": "served-model",
            "messages": [{"role": "user", "content": "frozen"}],
            "stream": mode == "stream",
        },
        expected_backend_identity_fingerprint=expected,
        gateway_direct_health_probe=health_probe,
        health_timeout_s=0.2,
        health_poll_interval_s=0.002,
    )
    return result, calls


@pytest.mark.parametrize(
    (
        "protocol",
        "mode",
        "stage",
        "direct",
        "gateway",
        "classification",
        "expected_pass",
    ),
    [
        (
            "chat",
            "nonstream",
            2,
            ("same", "same"),
            "different",
            "gateway_owned_difference",
            True,
        ),
        (
            "ollama",
            "stream",
            3,
            ("same", "same"),
            "same",
            "unverified",
            False,
        ),
        (
            "chat",
            "nonstream",
            2,
            ("first", "second"),
            "gateway",
            "shared_backend_model_or_cache_nondeterminism",
            True,
        ),
        (
            "chat",
            "nonstream",
            3,
            ("same", "same"),
            "same",
            "all_equal_prior_history_variance",
            True,
        ),
    ],
)
def test_paired_replay_classification_and_frozen_order(
    monkeypatch,
    protocol,
    mode,
    stage,
    direct,
    gateway,
    classification,
    expected_pass,
):
    result, calls = _paired_replay(
        monkeypatch,
        protocol=protocol,
        mode=mode,
        stage=stage,
        direct=direct,
        gateway=gateway,
    )
    assert result["pass"] is expected_pass
    assert result["classification"] == classification
    assert [row[0] for row in calls] == ["direct", "gateway", "direct"]
    assert len({row[4] for row in calls}) == 1
    assert result["checks"] == {
        "exact_body_sha_equal": True,
        "gateway_backend_lifecycle_pass": expected_pass,
    }


@pytest.mark.parametrize(
    ("lifecycle", "expected_counts"),
    [
        ("collector_only", (0, 1)),
        ("running_only", (1, 0)),
        ("non_atomic", None),
        ("settling", None),
        ("child_suffix", None),
    ],
)
def test_paired_replay_accepts_bounded_non_atomic_lifecycle_transitions(
    monkeypatch,
    lifecycle,
    expected_counts,
):
    result, calls = _paired_replay(monkeypatch, lifecycle=lifecycle)

    assert result["pass"] is True
    assert [row[0] for row in calls] == ["direct", "gateway", "direct"]
    evidence = result["gateway_in_flight_direct_health"]
    assert evidence["baseline_idle_settled"] is True
    assert evidence["gateway_activity_observed"] is True
    assert evidence["final_idle_settled"] is True
    assert evidence["gateway_action_executed"] is True
    assert evidence["request_id_correlation_status"] == "matched"
    assert evidence["request_id_correlation_pass"] is True
    assert evidence["foreign_request_ids_sha256"] == []
    assert evidence["worker_stopped"] is True
    assert all(
        (row["num_running"] or 0) <= 1
        and (row["active_requests"] or 0) <= 1
        for row in evidence["samples"]
    )
    if expected_counts is not None:
        assert any(
            (row["num_running"], row["active_requests"]) == expected_counts
            for row in evidence["samples"]
            if row["phase"] in {"during", "after"}
        )
    if lifecycle == "settling":
        assert any(
            row["phase"] == "before" and row["active_request_count"] == 1
            for row in evidence["samples"]
        )
    if lifecycle == "child_suffix":
        assert evidence["observed_action_request_ids_sha256"] == [
            matrix._sha256("response-b:visible-answer")
        ]


def test_paired_replay_ollama_keeps_exclusivity_without_fake_id_correlation(
    monkeypatch,
):
    result, _ = _paired_replay(
        monkeypatch,
        protocol="ollama",
        mode="stream",
        stage=3,
        lifecycle="running_only",
    )

    assert result["pass"] is False
    assert result["classification"] == "unverified"
    evidence = result["gateway_in_flight_direct_health"]
    assert evidence["bounded_exclusive_idle_active_idle"] is True
    assert evidence["exclusive_idle_active_idle"] is False
    assert evidence["request_owned_exclusive_idle_active_idle"] is False
    assert evidence["request_id_correlation_available"] is False
    assert (
        evidence["request_id_correlation_status"]
        == "unavailable_ollama_gateway_translation"
    )
    assert evidence["request_id_correlation_pass"] is None
    assert evidence["gateway_response_id_sha256"] is None
    assert evidence["observed_action_request_ids_sha256"]
    assert evidence["foreign_request_ids_sha256"] == []


@pytest.mark.parametrize(
    ("body_fault", "lifecycle", "identity"),
    [
        ("b", "exact", "match"),
        ("a2", "exact", "match"),
        (None, "inactive", "match"),
        (None, "fast", "match"),
        (None, "concurrent", "match"),
        (None, "preexisting", "match"),
        (None, "foreign", "match"),
        (None, "foreign_sequential", "match"),
        (None, "malformed", "match"),
        (None, "count_mismatch", "match"),
        (None, "invalid_id", "match"),
        (None, "invalid_encoding", "match"),
        (None, "missing_running_rows", "match"),
        (None, "exact", "mismatch"),
        (None, "exact", "drift"),
        (None, "exact", "failure"),
        (None, "exception", "match"),
    ],
)
def test_paired_replay_suppresses_unattested_classification(
    monkeypatch, body_fault, lifecycle, identity
):
    result, _ = _paired_replay(
        monkeypatch,
        body_fault=body_fault,
        lifecycle=lifecycle,
        identity=identity,
    )
    assert result["pass"] is False
    assert result["classification"] == "unverified"


def test_paired_replay_rejects_fast_unsampled_gateway_action(monkeypatch):
    result, calls = _paired_replay(monkeypatch, lifecycle="fast")

    assert [row[0] for row in calls] == ["direct", "gateway", "direct"]
    assert result["pass"] is False
    evidence = result["gateway_in_flight_direct_health"]
    assert evidence["gateway_action_executed"] is True
    assert evidence["gateway_activity_observed"] is False
    assert evidence["durable_completion_observed"] is False
    assert evidence["observed_action_request_ids_sha256"] == []
    assert evidence["request_id_correlation_status"] == "unobserved_request_id"
    assert any(
        "exclusive gateway activity or durable completion was not observed"
        in failure
        for failure in evidence["failures"]
    )


def test_paired_replay_accepts_correlated_fast_durable_completion(monkeypatch):
    result, calls = _paired_replay(monkeypatch, lifecycle="fast_completed")

    assert [row[0] for row in calls] == ["direct", "gateway", "direct"]
    assert result["pass"] is True
    evidence = result["gateway_in_flight_direct_health"]
    assert evidence["gateway_activity_observed"] is False
    assert evidence["durable_completion_observed"] is True
    assert evidence["action_evidence_observed"] is True
    assert evidence["request_id_correlation_status"] == "matched"
    assert evidence["request_id_correlation_pass"] is True
    assert evidence["observed_action_request_ids_sha256"] == [
        matrix._sha256("response-b")
    ]
    assert evidence["foreign_request_ids_sha256"] == []


def test_paired_replay_rejects_foreign_fast_durable_completion(monkeypatch):
    result, _ = _paired_replay(
        monkeypatch,
        lifecycle="fast_foreign_completed",
    )

    assert result["pass"] is False
    evidence = result["gateway_in_flight_direct_health"]
    assert evidence["gateway_activity_observed"] is False
    assert evidence["durable_completion_observed"] is True
    assert evidence["request_id_correlation_status"] == "foreign_request_ids"
    assert evidence["request_id_correlation_pass"] is False
    assert evidence["foreign_request_ids_sha256"] == [
        matrix._sha256("foreign-request")
    ]


def test_paired_replay_rejects_concurrent_and_foreign_lifecycle_ids(
    monkeypatch,
):
    concurrent, _ = _paired_replay(monkeypatch, lifecycle="concurrent")
    assert concurrent["pass"] is False
    concurrent_evidence = concurrent["gateway_in_flight_direct_health"]
    assert any(
        "concurrent or malformed engine activity" in failure
        for failure in concurrent_evidence["failures"]
    )

    foreign, _ = _paired_replay(
        monkeypatch,
        lifecycle="foreign_sequential",
    )
    assert foreign["pass"] is False
    foreign_evidence = foreign["gateway_in_flight_direct_health"]
    assert foreign_evidence["request_id_correlation_status"] == "foreign_request_ids"
    assert foreign_evidence["request_id_correlation_pass"] is False
    assert foreign_evidence["foreign_request_ids_sha256"] == [
        matrix._sha256("foreign-request")
    ]
    assert foreign_evidence["observed_action_request_ids_sha256"] == sorted(
        [
            matrix._sha256("response-b"),
            matrix._sha256("foreign-request"),
        ]
    )
    assert not any(
        "concurrent or malformed engine activity" in failure
        for failure in foreign_evidence["failures"]
    )


@pytest.mark.parametrize(
    "lifecycle",
    [
        "malformed",
        "count_mismatch",
        "invalid_id",
        "invalid_encoding",
        "missing_running_rows",
    ],
)
def test_paired_replay_rejects_malformed_lifecycle_attestation(
    monkeypatch,
    lifecycle,
):
    result, _ = _paired_replay(monkeypatch, lifecycle=lifecycle)

    assert result["pass"] is False
    evidence = result["gateway_in_flight_direct_health"]
    assert any(
        "request lifecycle capture failed" in failure
        for failure in evidence["failures"]
    )
    assert any(row["lifecycle_failures"] for row in evidence["samples"])


@pytest.mark.parametrize(
    "field",
    ["tool_calls", "notices", "terminals", "errors", "events"],
)
@pytest.mark.parametrize("invalid_container", [{}, (), ""])
def test_paired_replay_pipeline_rejects_nonlist_raw_collections(
    monkeypatch,
    field,
    invalid_container,
):
    with pytest.raises(ValueError, match=rf"raw {field} is not a list"):
        _paired_replay(
            monkeypatch,
            raw_override={field: invalid_container},
        )


def test_paired_replay_rejects_invalid_target_prerequisites():
    kwargs = {
        "direct_client": None,
        "gateway_client": None,
        "payload": {"model": "served-model", "stream": True},
        "expected_backend_identity_fingerprint": "f" * 64,
        "gateway_direct_health_probe": lambda: {},
    }
    with pytest.raises(ValueError, match="limited to Chat nonstream"):
        matrix.run_paired_replay_discriminator(
            protocol="responses", mode="stream", stage=3, **kwargs
        )
    with pytest.raises(ValueError, match="stream mode"):
        matrix.run_paired_replay_discriminator(
            protocol="chat", mode="nonstream", stage=2, **kwargs
        )
    kwargs["payload"]["stream"] = False
    kwargs["gateway_direct_health_probe"] = None
    with pytest.raises(ValueError, match="direct-health"):
        matrix.run_paired_replay_discriminator(
            protocol="chat", mode="nonstream", stage=2, **kwargs
        )


def test_paired_replay_whole_result_redacts_request_and_response_canaries(
    monkeypatch,
):
    canaries = [
        "CANARY-PROMPT",
        "CANARY-CONTRACT-NAME",
        "CANARY-DESCRIPTION",
        "CANARY-SCHEMA-KEY",
        "CANARY-CHOICE",
        "CANARY-CALL-ID",
        "CANARY-HISTORY-NAME",
        "CANARY-RESULT-NAME",
        "CANARY-RESULT",
        "CANARY-PREVIOUS-ID",
        "CANARY-RESPONSE",
    ]
    payload = {
        "model": "served-model",
        "messages": [
            {"role": "user", "content": canaries[0]},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": canaries[5],
                        "type": "function",
                        "function": {"name": canaries[6], "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": canaries[5],
                "name": canaries[7],
                "content": canaries[8],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": canaries[1],
                    "description": canaries[2],
                    "parameters": {
                        "type": "object",
                        "properties": {canaries[3]: {"type": "string"}},
                    },
                },
            }
        ],
        "tool_choice": {
            "type": "function",
            "function": {"name": canaries[4]},
        },
        "previous_response_id": canaries[9],
        "stream": False,
    }
    result, _ = _paired_replay(
        monkeypatch, payload=payload, canary=canaries[10]
    )
    serialized = json.dumps(result, sort_keys=True)
    assert all(canary not in serialized for canary in canaries)
    linkage = result["request"]["tool_history_linkage"]
    assert linkage[0]["call_id"] == linkage[1]["call_id"] == "tool_call_1"
    response = result["responses"]["a1"]
    assert response["tool_calls"][0]["call_id"] == response["events"][0]["call_id"]


def _private_response(**overrides):
    value = {
        "reasoning": "",
        "content": "",
        "status_code": 200,
        "notices": [],
        "tool_calls": [],
        "terminals": [],
        "errors": [],
        "events": [],
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize(
    "overrides",
    [
        {"tool_calls": [["nested"]]},
        {
            "tool_calls": [
                {"index": 0, "name": "run_command", "arguments": []}
            ]
        },
        {
            "tool_calls": [
                {
                    "index": 0,
                    "name": "run_command",
                    "arguments": {},
                    "unknown": {"nested": "value"},
                }
            ]
        },
        {"notices": [{"nested": "value"}]},
        {"terminals": [{"nested": "value"}]},
        {"terminals": ["not-a-terminal"]},
        {"errors": {}},
        {"errors": ()},
        {"errors": ""},
        {"events": {}},
        {"events": ()},
        {"events": ""},
        {
            "events": [
                {
                    "channel": "content",
                    "kind": "chat.content.complete",
                    "unknown": {"nested": "value"},
                }
            ]
        },
        {
            "errors": [
                {
                    "channel": "error",
                    "kind": "json_parse_error",
                    "unknown": {"nested": "value"},
                }
            ]
        },
    ],
)
def test_paired_replay_public_response_rejects_unsupported_shapes(overrides):
    with pytest.raises(ValueError):
        matrix._paired_public_response(_private_response(**overrides))


def test_paired_replay_path_free_event_contract():
    event = {
        "channel": "content",
        "kind": "chat.content.complete",
        "chars": 7,
        "sha256": matrix._sha256("visible"),
    }
    assert matrix._paired_public_response(
        _private_response(events=[event])
    )["events"] == [event]
    for invalid in (
        {**event, "chars": -1},
        {**event, "sha256": "invalid"},
        {**event, "kind": "delta"},
    ):
        with pytest.raises(ValueError):
            matrix._paired_public_response(
                _private_response(events=[invalid])
            )


def test_paired_replay_accepts_real_collector_result_without_notices():
    canary = "CANARY-REAL-EVENT-COLLECTOR"
    collector = matrix.EventCollector(protocol="chat", started=time.monotonic())
    collector.text(
        "content",
        canary,
        "chat.content.complete",
        at_ms=1.0,
    )
    collector.terminal("stop", at_ms=2.0)
    raw = collector.result(status_code=200, elapsed_ms=3.0)

    assert "notices" not in raw
    normalized = matrix._normalized_replay_response(raw)
    public = matrix._paired_public_response(normalized)
    assert normalized["notices"] == []
    assert canary not in json.dumps(public)
    assert public["content_chars"] == len(canary)
    assert public["content_sha256"] == matrix._sha256(canary)
    assert public["events"][0] == {
        "channel": "content",
        "kind": "chat.content.complete",
        "chars": len(canary),
        "sha256": matrix._sha256(canary),
    }


@pytest.mark.parametrize(
    "mutation",
    [
        {"tools": {}},
        {"tools": ()},
        {"tools": ""},
        {"messages": {}},
        {"messages": ()},
        {"messages": ""},
        {"messages": ["not-an-object"]},
        {
            "messages": [
                {"role": "assistant", "content": "", "tool_calls": {}}
            ]
        },
        {
            "messages": [
                {"role": "assistant", "content": "", "tool_calls": ()}
            ]
        },
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": ["not-an-object"],
                }
            ]
        },
        {"messages": [{"role": "user", "content": {}}]},
        {"messages": [{"role": "user", "content": ()}]},
        {"messages": [{"role": "user", "content": ["not-an-object"]}]},
    ],
)
def test_paired_replay_request_rejects_malformed_collection_shapes(mutation):
    payload = {
        "model": "served-model",
        "messages": [{"role": "user", "content": "valid"}],
        "stream": False,
        **mutation,
    }
    with pytest.raises(ValueError):
        matrix._paired_public_request(2, payload)


@pytest.mark.parametrize("invalid_id", [{}, [], 7])
def test_paired_replay_request_rejects_nontext_tool_ids(invalid_id):
    payload = {
        "model": "served-model",
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": invalid_id,
                        "type": "function",
                        "function": {
                            "name": "run_command",
                            "arguments": "{}",
                        },
                    }
                ],
            }
        ],
        "stream": False,
    }
    with pytest.raises(ValueError, match="call id is not text"):
        matrix._paired_public_request(2, payload)


@pytest.mark.parametrize("location", ["tool", "event"])
@pytest.mark.parametrize("invalid_id", [{}, [], 7])
def test_paired_replay_response_rejects_nontext_tool_ids(
    location,
    invalid_id,
):
    raw = {
        "status_code": 200,
        "reasoning": "",
        "content": "",
        "notices": [],
        "tool_calls": [],
        "terminals": [],
        "errors": [],
        "events": [],
    }
    if location == "tool":
        raw["tool_calls"] = [
            {
                "index": 0,
                "id": invalid_id,
                "name": "run_command",
                "arguments": {},
            }
        ]
    else:
        raw["events"] = [
            {
                "channel": "tool",
                "kind": "chat.tool.complete",
                "call_id": invalid_id,
            }
        ]
    with pytest.raises(ValueError, match="call id is not text"):
        matrix._normalized_replay_response(raw)
