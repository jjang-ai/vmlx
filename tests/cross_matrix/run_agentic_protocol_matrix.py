#!/usr/bin/env python3
"""Run a real two-tool coding-harness matrix against existing vMLX endpoints.

The runner is intentionally import-safe and does not launch, stop, or mutate a
server.  Callers provide the model and both direct/gateway base URLs.  It runs
the same bounded conversation through Chat Completions, Responses, Anthropic,
and Ollama, in streaming and non-streaming modes:

    reasoning -> file_info -> real result -> reasoning -> pwd -> real result
    -> final visible synthesis

Only two allowlisted read-only tools exist.  Private reasoning and exact
decompressed parser-input response bodies are not written to the output
artifact; the artifact retains hashes, lengths, timestamps, visible text, tool
metadata, and terminal classifications. Streaming mode and Responses
non-streaming mode require private parser-input capture with
``--raw-artifact-dir``. That directory must resolve outside every Git
worktree, and capture metadata retains values only for an explicit safe-header
allowlist.

The v2 result fails closed unless the runner observes one clean Git commit/tree
and matching immutable runtime, model-bundle, and cache-topology attestations
from full direct and gateway health snapshots before and after the matrix.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import json
import os
import stat
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

PROTOCOLS = ("chat", "responses", "anthropic", "ollama")
MODES = ("stream", "nonstream")
FILE_INFO_PATH = "panel/package.json"
PWD_COMMAND = "pwd"
CONTROL_MARKERS = (
    "<think",
    "</think",
    "[think]",
    "[/think]",
    "<mm:think",
    "<tool_call",
    "</tool_call",
    "<tool_calls",
    "</tool_calls",
    "<tool_sep>",
    "<arg_key>",
    "<arg_value>",
    "<zyphra_tool_call",
    "<function=",
    "<parameter=",
    "<|tool_call",
    "<|tool_calls",
    "[tool_calls]",
    "[tool]",
    "[calling tool:",
    "<minimax:tool_call>",
    "]<]minimax[>[",
    "<|recipient|>",
    "<|tool_calls_section_begin|>",
    "<|tool_call_begin|>",
    "<｜tool▁calls▁begin｜>",
    "<｜tool▁call▁begin｜>",
    "<｜dsml｜tool",
    "<｜dsml｜invoke",
    "<|python_tag|>",
    "```tool_code",
)
SAFE_CAPTURE_HEADER_NAMES = {
    "content-encoding",
    "content-length",
    "content-type",
    "transfer-encoding",
}
CAPTURE_LAYER = "requests.decompressed_response_parser_input"
CAPTURE_SEMANTICS = (
    "Exact decompressed response-body bytes delivered to protocol parsers: "
    "streaming bytes before requests.iter_lines line splitting or Unicode "
    "decoding, and nonstream response bytes before JSON decoding; excludes "
    "HTTP transfer framing and compressed transport octets."
)
OUTPUT_SCHEMA = "vmlx-agentic-protocol-matrix-v2"
OUTPUT_SCHEMA_VERSION = 2
INSTALLED_RELEASE_MANIFEST_SCHEMA = "vmlx-installed-release-manifest-v1"
INSTALLED_RELEASE_MANIFEST_FIELDS = {
    "schema",
    "source_commit",
    "source_tree",
    "app_asar_sha256",
    "electron_executable_sha256",
    "bundled_provenance_sha256",
    "bundled_python_executable_sha256",
    "bundled_python_executable_fingerprint_sha256",
}
INSTALLED_BUNDLED_PYTHON_RELATIVE_PATH = Path(
    "Contents/Resources/bundled-python/python/bin/python3"
)
INSTALLED_APP_ARTIFACTS = {
    "app_asar": (
        Path("Contents/Resources/app.asar"),
        "app_asar_sha256",
    ),
    "electron_executable": (
        Path("Contents/MacOS/vMLX"),
        "electron_executable_sha256",
    ),
    "bundled_provenance": (
        Path("Contents/Resources/bundled-python/vmlx-bundle-provenance.json"),
        "bundled_provenance_sha256",
    ),
}
BUNDLE_ATTESTATION_FILENAMES = (
    "config.json",
    "generation_config.json",
    "jang_config.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
RUNTIME_SOURCE_HASH_FIELDS = (
    "server_module_sha256",
    "package_init_sha256",
    "python_source_tree_sha256",
)
RUNTIME_HASH_FIELDS = (
    *RUNTIME_SOURCE_HASH_FIELDS,
    "python_executable_fingerprint_sha256",
)
PAIRED_REPLAY_TARGETS = {
    ("chat", "nonstream", 2),
    ("ollama", "stream", 3),
    ("chat", "nonstream", 3),
}
PAIRED_REPLAY_EVENT_CHANNELS = {
    "reasoning",
    "content",
    "tool",
    "terminal",
    "error",
}
PAIRED_REPLAY_EVENT_KINDS = {
    "chat.reasoning.complete",
    "chat.content.complete",
    "chat.tool.complete",
    "ollama.thinking",
    "ollama.content",
    "ollama.tool",
    "json_parse_error",
    "ollama.error",
    "stop",
    "length",
    "tool_calls",
    "DONE",
}
PAIRED_REPLAY_TOOL_NAMES = {"file_info", "run_command"}
PAIRED_REPLAY_TERMINALS = {"stop", "length", "tool_calls", "DONE"}

TOOL_PARAMETERS: dict[str, dict[str, Any]] = {
    "file_info": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    },
    "run_command": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
        "additionalProperties": False,
    },
}

TOOL_DESCRIPTIONS = {
    "file_info": "Return current filesystem metadata for the one allowed path.",
    "run_command": "Run the one allowed read-only command in the repository.",
}

STABLE_RESPONSES_INSTRUCTIONS = (
    "Act as a coding agent. Follow the current user input, use only the supplied "
    "tools, never fabricate tool results, and continue from real function outputs."
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _valid_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text.lower()
    )


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _opened_regular_file_identity(path: Path, label: str) -> dict[str, Any]:
    """Hash one resolved regular file through a no-follow descriptor."""
    resolved = path.expanduser().resolve(strict=True)
    path_stat = resolved.lstat()
    if not stat.S_ISREG(path_stat.st_mode) or stat.S_ISLNK(path_stat.st_mode):
        raise ValueError(f"{label} is not a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{label} is not a regular file")
        if (
            opened.st_dev != path_stat.st_dev
            or opened.st_ino != path_stat.st_ino
            or opened.st_size != path_stat.st_size
        ):
            raise ValueError(f"{label} changed identity while opening")
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    finally:
        os.close(descriptor)
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "size_bytes": size,
    }


def _opened_nofollow_regular_file(
    path: Path,
    label: str,
    *,
    retain_bytes: bool = False,
    max_bytes: int | None = None,
    require_single_link: bool = False,
) -> tuple[dict[str, Any], bytes]:
    """Open a caller-named regular file without following its final component."""
    requested = path.expanduser().absolute()
    path_stat = requested.lstat()
    if not stat.S_ISREG(path_stat.st_mode) or stat.S_ISLNK(path_stat.st_mode):
        raise ValueError(f"{label} is not a regular non-symlink file")
    if require_single_link and path_stat.st_nlink != 1:
        raise ValueError(f"{label} must have exactly one filesystem link")
    if max_bytes is not None and path_stat.st_size > max_bytes:
        raise ValueError(f"{label} exceeds the {max_bytes}-byte safety limit")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(requested, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{label} is not a regular file")
        if require_single_link and opened.st_nlink != 1:
            raise ValueError(f"{label} opened object must have exactly one link")
        if (
            opened.st_dev != path_stat.st_dev
            or opened.st_ino != path_stat.st_ino
            or opened.st_size != path_stat.st_size
        ):
            raise ValueError(f"{label} changed identity while opening")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            if retain_bytes:
                chunks.append(chunk)
            digest.update(chunk)
            size += len(chunk)
            if max_bytes is not None and size > max_bytes:
                raise ValueError(
                    f"{label} exceeds the {max_bytes}-byte safety limit"
                )
    finally:
        os.close(descriptor)
    return (
        {
            "path": str(requested.resolve(strict=True)),
            "requested_path": str(requested),
            "sha256": digest.hexdigest(),
            "size_bytes": size,
            "nlink": opened.st_nlink,
            "opened_nofollow": True,
        },
        b"".join(chunks),
    )


def _git_text(repo_root: Path, *arguments: str) -> str:
    environment = dict(os.environ)
    for name in ("GIT_DIR", "GIT_WORK_TREE"):
        environment.pop(name, None)
    environment["LC_ALL"] = "C"
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
    except OSError as exc:
        raise ValueError("cannot observe source Git identity") from exc
    if proc.returncode != 0:
        raise ValueError("cannot observe source Git identity")
    return proc.stdout.strip()


def _python_source_tree_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    count = 0
    read_errors = 0
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root.parent).as_posix().encode()
        try:
            content = path.read_bytes()
        except OSError:
            digest.update(relative)
            digest.update(b"\0UNREADABLE\0")
            read_errors += 1
            continue
        digest.update(relative)
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
        count += 1
    return digest.hexdigest(), count, read_errors


def observe_source_checkout(repo_root: Path) -> dict[str, Any]:
    """Observe the exact checkout rather than trusting a caller label."""
    requested_root = repo_root.resolve(strict=True)
    git_root = Path(
        _git_text(requested_root, "rev-parse", "--show-toplevel")
    ).resolve(strict=True)
    if git_root != requested_root:
        raise ValueError("--repo-root must be the observed Git worktree root")
    head = _git_text(git_root, "rev-parse", "HEAD")
    tree = _git_text(git_root, "rev-parse", "HEAD^{tree}")
    status = _git_text(
        git_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    source_tree_sha256, source_file_count, source_read_error_count = (
        _python_source_tree_digest(git_root / "vmlx_engine")
    )
    return {
        "git_root": str(git_root),
        "head": head,
        "tree": tree,
        "clean": not bool(status),
        "status_sha256": _sha256(status),
        "python_source_tree_sha256": source_tree_sha256,
        "python_source_file_count": source_file_count,
        "python_source_read_error_count": source_read_error_count,
        "server_module_sha256": hashlib.sha256(
            (git_root / "vmlx_engine" / "server.py").read_bytes()
        ).hexdigest(),
        "package_init_sha256": hashlib.sha256(
            (git_root / "vmlx_engine" / "__init__.py").read_bytes()
        ).hexdigest(),
    }


def _installed_app_root_from_python(executable: Path) -> Path:
    requested = executable.expanduser().absolute()
    try:
        canonical_executable = requested.resolve(strict=True)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        raise ValueError("installed proof runner executable is unavailable") from exc
    candidate = requested
    for _ in INSTALLED_BUNDLED_PYTHON_RELATIVE_PATH.parts:
        candidate = candidate.parent
    packaged_invocation = candidate / INSTALLED_BUNDLED_PYTHON_RELATIVE_PATH
    try:
        canonical_packaged_invocation = packaged_invocation.resolve(strict=True)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        raise ValueError(
            "installed app exact bundled-Python invocation is unavailable"
        ) from exc
    if canonical_packaged_invocation != canonical_executable:
        raise ValueError(
            "installed proof runner is not the exact packaged bundled-Python target"
        )
    if candidate.name != "vMLX.app":
        raise ValueError("installed proof runner is not inside vMLX.app")
    app_stat = candidate.lstat()
    if not stat.S_ISDIR(app_stat.st_mode) or stat.S_ISLNK(app_stat.st_mode):
        raise ValueError("installed vMLX.app is not a real non-symlink directory")
    return candidate


def _observe_installed_runtime(
    *,
    manifest_path: Path,
    repo_root: Path,
    source: dict[str, Any],
    invoked_executable: Path,
) -> dict[str, Any]:
    manifest_requested = manifest_path.expanduser().absolute()
    if _is_within_directory(manifest_requested, repo_root) or _is_in_git_context(
        manifest_requested
    ):
        raise ValueError(
            "--installed-release-manifest must resolve outside every Git "
            "worktree and Git metadata directory"
        )
    if stat.S_IMODE(manifest_requested.stat().st_mode) != 0o600:
        raise ValueError("installed release manifest permissions must be 0600")
    manifest_record, manifest_bytes = _opened_nofollow_regular_file(
        manifest_requested,
        "installed release manifest",
        retain_bytes=True,
        max_bytes=1024 * 1024,
        require_single_link=True,
    )
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("installed release manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("installed release manifest must be a JSON object")
    if set(manifest) != INSTALLED_RELEASE_MANIFEST_FIELDS:
        raise ValueError("installed release manifest fields are not exact")
    if manifest.get("schema") != INSTALLED_RELEASE_MANIFEST_SCHEMA:
        raise ValueError("installed release manifest schema is invalid")
    if manifest.get("source_commit") != source.get("head"):
        raise ValueError(
            "installed release manifest source commit does not match the checkout"
        )
    if manifest.get("source_tree") != source.get("tree"):
        raise ValueError(
            "installed release manifest source tree does not match the checkout"
        )
    for field in sorted(INSTALLED_RELEASE_MANIFEST_FIELDS - {
        "schema",
        "source_commit",
        "source_tree",
    }):
        if not _valid_sha256(manifest.get(field)):
            raise ValueError(f"installed release manifest {field} is invalid")

    requested_executable = invoked_executable.expanduser().absolute()
    app_root = _installed_app_root_from_python(requested_executable)
    canonical_executable = requested_executable.resolve(strict=True)
    invoked_fingerprint = _sha256(str(canonical_executable))
    # Electron launches the packaged interpreter through the stable `python3`
    # path, while CPython may report the canonical versioned target (for
    # example `python3.12`) to the proof producer. Both invocation spellings
    # resolve to the same manifest-hashed executable; retain the exact pair so
    # health can attest the app-owned launcher without accepting arbitrary
    # interpreters from the same prefix.
    packaged_invocation_path = (
        app_root / INSTALLED_BUNDLED_PYTHON_RELATIVE_PATH
    ).absolute()
    if packaged_invocation_path.resolve(strict=True) != canonical_executable:
        raise ValueError(
            "installed app exact bundled-Python invocation does not resolve "
            "to the manifest-attested runner"
        )
    packaged_invocation_fingerprint = _sha256(str(packaged_invocation_path))
    if (
        invoked_fingerprint
        != manifest["bundled_python_executable_fingerprint_sha256"]
    ):
        raise ValueError(
            "installed proof runner path does not match the manifest-attested "
            "bundled Python"
        )
    bundled_python = _opened_regular_file_identity(
        canonical_executable,
        "installed bundled Python",
    )
    if not _is_within_directory(Path(bundled_python["path"]), app_root):
        raise ValueError(
            "installed bundled Python resolves outside the installed app"
        )
    if (
        bundled_python["sha256"]
        != manifest["bundled_python_executable_sha256"]
    ):
        raise ValueError(
            "installed bundled Python does not match the release manifest"
        )

    artifacts: dict[str, dict[str, Any]] = {}
    artifact_bytes: dict[str, bytes] = {}
    for label, (relative_path, manifest_field) in INSTALLED_APP_ARTIFACTS.items():
        record, content = _opened_nofollow_regular_file(
            app_root / relative_path,
            f"installed {label.replace('_', ' ')}",
            retain_bytes=label == "bundled_provenance",
        )
        if record["sha256"] != manifest[manifest_field]:
            raise ValueError(
                f"installed {label.replace('_', ' ')} does not match the "
                "release manifest"
            )
        artifacts[label] = record
        artifact_bytes[label] = content
    if not os.access(artifacts["electron_executable"]["path"], os.X_OK):
        raise ValueError("installed Electron executable is not executable")

    try:
        provenance = json.loads(artifact_bytes["bundled_provenance"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "installed bundled provenance is not valid UTF-8 JSON"
        ) from exc
    if (
        not isinstance(provenance, dict)
        or provenance.get("schema_version") != 1
        or not isinstance(provenance.get("vmlx"), dict)
        or provenance["vmlx"].get("commit") != source.get("head")
    ):
        raise ValueError(
            "installed bundled provenance does not match the source checkout"
        )

    bundled_source_root = (
        app_root / "Contents/Resources/vmlx-engine-source/vmlx_engine"
    )
    bundled_source_stat = bundled_source_root.lstat()
    if (
        not stat.S_ISDIR(bundled_source_stat.st_mode)
        or stat.S_ISLNK(bundled_source_stat.st_mode)
    ):
        raise ValueError("installed bundled source is not a real directory")
    for bundled_python_source in bundled_source_root.rglob("*.py"):
        bundled_python_source_stat = bundled_python_source.lstat()
        if (
            not stat.S_ISREG(bundled_python_source_stat.st_mode)
            or stat.S_ISLNK(bundled_python_source_stat.st_mode)
        ):
            raise ValueError(
                "installed bundled source contains a non-regular Python file"
            )
    bundled_tree_sha256, bundled_file_count, bundled_read_errors = (
        _python_source_tree_digest(bundled_source_root)
    )
    bundled_source = {
        "python_source_tree_sha256": bundled_tree_sha256,
        "python_source_file_count": bundled_file_count,
        "python_source_read_error_count": bundled_read_errors,
        "server_module_sha256": hashlib.sha256(
            (bundled_source_root / "server.py").read_bytes()
        ).hexdigest(),
        "package_init_sha256": hashlib.sha256(
            (bundled_source_root / "__init__.py").read_bytes()
        ).hexdigest(),
    }
    for field in (
        *RUNTIME_SOURCE_HASH_FIELDS,
        "python_source_file_count",
        "python_source_read_error_count",
    ):
        if bundled_source.get(field) != source.get(field):
            raise ValueError(
                f"installed bundled source does not match the checkout: {field}"
            )

    source_binding = {
        field: source[field]
        for field in (
            "head",
            "tree",
            *RUNTIME_SOURCE_HASH_FIELDS,
            "python_source_file_count",
            "python_source_read_error_count",
        )
    }
    return {
        "schema": "vmlx-agentic-installed-runtime-v1",
        "manifest": manifest,
        "manifest_path": manifest_record["path"],
        "manifest_sha256": manifest_record["sha256"],
        "manifest_size_bytes": manifest_record["size_bytes"],
        "manifest_nlink": manifest_record["nlink"],
        "manifest_opened_nofollow": manifest_record["opened_nofollow"],
        "app_path": str(app_root),
        "invoked_python_path": str(canonical_executable),
        "invoked_python_fingerprint_sha256": invoked_fingerprint,
        "packaged_python_invocation_path": str(packaged_invocation_path),
        "packaged_python_invocation_fingerprint_sha256": (
            packaged_invocation_fingerprint
        ),
        "python_prefix_path": str(
            canonical_executable.parent.parent.resolve(strict=True)
        ),
        "bundled_python": bundled_python,
        "artifacts": artifacts,
        "bundled_provenance": provenance,
        "bundled_source": bundled_source,
        "source_binding": source_binding,
    }


def observe_runner_environment(
    repo_root: Path,
    installed_release_manifest: Path | None = None,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the proof producer to source Python or a manifest-bound install."""
    git_root = repo_root.resolve(strict=True)
    expected_prefix_path = git_root / ".venv"
    expected_prefix = (
        expected_prefix_path.absolute()
        if installed_release_manifest is not None
        else expected_prefix_path.resolve(strict=True)
    )
    actual_prefix = Path(sys.prefix).resolve(strict=True)
    executable = Path(sys.executable).absolute()
    executable_file = _opened_regular_file_identity(
        Path(sys.executable),
        "proof runner executable",
    )
    harness_relative_path = "tests/cross_matrix/run_agentic_protocol_matrix.py"
    harness_file = _opened_regular_file_identity(
        git_root / harness_relative_path,
        "proof runner harness",
    )
    expected_executables = tuple(
        candidate.absolute()
        for candidate in (
            git_root / ".venv" / "bin" / "python",
            git_root / ".venv" / "bin" / "python3",
        )
        if candidate.is_file()
    )
    producer_executable = Path(executable_file["path"]).resolve(strict=True)
    checkout_python_invocation_fingerprints = sorted(
        {
            _sha256(str(candidate))
            for candidate in expected_executables
            if candidate.resolve(strict=True) == producer_executable
        }
    )
    installed_runtime = None
    if installed_release_manifest is not None:
        installed_runtime = _observe_installed_runtime(
            manifest_path=installed_release_manifest,
            repo_root=git_root,
            source=source or observe_source_checkout(git_root),
            invoked_executable=executable,
        )
        if actual_prefix != Path(installed_runtime["python_prefix_path"]):
            raise ValueError(
                "installed proof runner sys.prefix is not the bundled Python root"
            )
    execution_mode = (
        "installed-runtime" if installed_runtime is not None else "source-checkout-venv"
    )
    installed_python_invocation_fingerprints = (
        sorted(
            {
                installed_runtime["invoked_python_fingerprint_sha256"],
                installed_runtime[
                    "packaged_python_invocation_fingerprint_sha256"
                ],
            }
        )
        if installed_runtime is not None
        else []
    )
    accepted_python_invocation_fingerprints = (
        installed_python_invocation_fingerprints
        if installed_runtime is not None
        else checkout_python_invocation_fingerprints
    )
    effective_executable = (
        Path(installed_runtime["invoked_python_path"])
        if installed_runtime is not None
        else executable
    )
    effective_executable_fingerprint = _sha256(str(effective_executable))
    return {
        "execution_mode": execution_mode,
        "repo_venv": actual_prefix == expected_prefix,
        "repo_python": executable in expected_executables,
        "python_executable_path": str(effective_executable),
        "python_executable_fingerprint_sha256": effective_executable_fingerprint,
        "checkout_python_invocation_fingerprints_sha256": (
            checkout_python_invocation_fingerprints
        ),
        "installed_python_invocation_fingerprints_sha256": (
            installed_python_invocation_fingerprints
        ),
        "accepted_python_invocation_fingerprints_sha256": (
            accepted_python_invocation_fingerprints
        ),
        "python_prefix_path": str(actual_prefix),
        "python_prefix_fingerprint_sha256": hashlib.sha256(
            str(actual_prefix).encode()
        ).hexdigest(),
        "producer_pid": os.getpid(),
        "producer_executable_path": executable_file["path"],
        "producer_executable_sha256": executable_file["sha256"],
        "producer_executable_size_bytes": executable_file["size_bytes"],
        "producer_harness_relative_path": harness_relative_path,
        "producer_harness_path": harness_file["path"],
        "producer_harness_sha256": harness_file["sha256"],
        "producer_harness_size_bytes": harness_file["size_bytes"],
        "installed_runtime": installed_runtime,
    }


def _normalized_bundle_model_name(bundle_root: Path) -> str:
    parts = bundle_root.resolve(strict=True).parts
    for part in parts:
        if part.startswith("models--") and "--" in part[len("models--") :]:
            organization, repository = part[len("models--") :].split("--", 1)
            if organization and repository:
                return f"{organization}/{repository}"
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return parts[-1]


def observe_bundle_configuration(bundle_root: Path) -> dict[str, Any]:
    """Independently hash the exact local bundle selected for this proof."""
    root = bundle_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("--bundle-root must resolve to a local model directory")
    files: dict[str, dict[str, Any]] = {}
    for name in BUNDLE_ATTESTATION_FILENAMES:
        candidate = root / name
        if not candidate.exists():
            files[name] = {"state": "missing"}
            continue
        if not candidate.is_file() or candidate.is_symlink():
            raise ValueError(
                f"--bundle-root contains a non-regular attestation file: {name}"
            )
        try:
            data = candidate.read_bytes()
        except OSError as exc:
            raise ValueError(f"cannot read bundle attestation file: {name}") from exc
        files[name] = {
            "state": "present",
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    observed = {
        "schema": "vmlx-bundle-config-v1",
        "directory_state": "available",
        "files": files,
    }
    aggregate = _canonical_sha256(observed)
    return {
        **observed,
        "aggregate_sha256": aggregate,
        "fingerprint_sha256": aggregate,
        "model_name": _normalized_bundle_model_name(root),
    }


def _get_full_health(url: str, timeout: int) -> dict[str, Any]:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        health = response.json()
    except (OSError, requests.RequestException, ValueError) as exc:
        raise ValueError("could not capture full /health evidence") from exc
    if not isinstance(health, dict):
        raise ValueError("full /health evidence is not a JSON object")
    return health


def _health_identity(health: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Extract immutable backend identity while retaining the full health body."""
    failures: list[str] = []
    runtime = health.get("runtime_provenance")
    bundle = health.get("model_bundle_provenance")
    topology = health.get("cache_topology_provenance")
    if not isinstance(runtime, dict):
        failures.append("/health runtime_provenance is missing")
        runtime = {}
    if not isinstance(bundle, dict):
        failures.append("/health model_bundle_provenance is missing")
        bundle = {}
    if not isinstance(topology, dict):
        failures.append("/health cache_topology_provenance is missing")
        topology = {}
    if runtime.get("model_bundle_provenance") != bundle:
        failures.append(
            "/health nested and top-level model bundle provenance differ"
        )
    if runtime.get("cache_topology_provenance") != topology:
        failures.append(
            "/health nested and top-level cache topology provenance differ"
        )

    try:
        backend_pid = int(runtime.get("pid") or 0)
    except (TypeError, ValueError):
        backend_pid = 0
    if backend_pid <= 0:
        failures.append("/health runtime_provenance.pid is invalid")

    runtime_hashes = {
        hash_field: str(runtime.get(hash_field) or "")
        for hash_field in RUNTIME_HASH_FIELDS
    }
    for hash_field, value in runtime_hashes.items():
        if not _valid_sha256(value):
            failures.append(
                f"/health runtime_provenance.{hash_field} is invalid"
            )

    bundle_fingerprint = str(bundle.get("fingerprint_sha256") or "")
    topology_fingerprint = str(topology.get("fingerprint_sha256") or "")
    if not _valid_sha256(bundle_fingerprint):
        failures.append(
            "/health model_bundle_provenance.fingerprint_sha256 is invalid"
        )
    if not _valid_sha256(topology_fingerprint):
        failures.append(
            "/health cache_topology_provenance.fingerprint_sha256 is invalid"
        )
    topology_configuration = topology.get("configuration")
    if topology.get("schema") != "vmlx-cache-topology-attestation-v1":
        failures.append("/health cache_topology_provenance.schema is invalid")
    if not isinstance(topology_configuration, dict):
        failures.append(
            "/health cache_topology_provenance.configuration is missing"
        )
        topology_configuration = {}
    computed_topology_fingerprint = _canonical_sha256(topology_configuration)
    if topology.get("canonical_sha256") != computed_topology_fingerprint:
        failures.append(
            "/health cache_topology_provenance.canonical_sha256 is not self-consistent"
        )
    if topology_fingerprint != computed_topology_fingerprint:
        failures.append(
            "/health cache_topology_provenance.fingerprint_sha256 is not self-consistent"
        )
    if health.get("model_loaded") is not True:
        failures.append("/health does not attest model_loaded=true")
    model_name = str(health.get("model_name") or "")
    if not model_name:
        failures.append("/health model_name is empty")
    loaded_model_name = str(health.get("loaded_model_name") or "")
    if not loaded_model_name:
        failures.append("/health loaded_model_name is empty")

    bundle_files = bundle.get("files")
    if bundle.get("schema") != "vmlx-bundle-config-v1":
        failures.append("/health model_bundle_provenance.schema is invalid")
    if bundle.get("directory_state") != "available":
        failures.append(
            "/health model_bundle_provenance does not attest an available local bundle"
        )
    if not isinstance(bundle_files, dict) or set(bundle_files) != set(
        BUNDLE_ATTESTATION_FILENAMES
    ):
        failures.append("/health model_bundle_provenance.files is incomplete")
        bundle_files = {}
    observed_bundle = {
        "schema": bundle.get("schema"),
        "directory_state": bundle.get("directory_state"),
        "files": bundle_files,
    }
    computed_bundle_fingerprint = _canonical_sha256(observed_bundle)
    if bundle.get("aggregate_sha256") != computed_bundle_fingerprint:
        failures.append(
            "/health model_bundle_provenance.aggregate_sha256 is not self-consistent"
        )
    if bundle_fingerprint != computed_bundle_fingerprint:
        failures.append(
            "/health model_bundle_provenance.fingerprint_sha256 is not self-consistent"
        )
    for required_name in ("config.json", "tokenizer_config.json"):
        row = bundle_files.get(required_name)
        if not isinstance(row, dict) or row.get("state") != "present":
            failures.append(
                f"/health model bundle does not contain required {required_name}"
            )

    identity = {
        "backend_pid": backend_pid,
        "runtime_source_hashes": runtime_hashes,
        "python_source_file_count": runtime.get("python_source_file_count"),
        "python_source_read_error_count": runtime.get(
            "python_source_read_error_count"
        ),
        "model_name": model_name,
        "loaded_model_name": loaded_model_name,
        "model_bundle_fingerprint_sha256": bundle_fingerprint,
        "model_bundle_files": bundle_files,
        "cache_topology_fingerprint_sha256": topology_fingerprint,
    }
    identity["fingerprint_sha256"] = _canonical_sha256(identity)
    return identity, failures


def _validate_health_source_binding(
    identity: dict[str, Any],
    source: dict[str, Any],
    runner: dict[str, Any],
    bundle: dict[str, Any],
    requested_model: str,
) -> list[str]:
    failures: list[str] = []
    runtime_hashes = identity.get("runtime_source_hashes")
    if not isinstance(runtime_hashes, dict):
        return ["observed backend runtime source hashes are missing"]
    for source_field in RUNTIME_SOURCE_HASH_FIELDS:
        if runtime_hashes.get(source_field) != source.get(source_field):
            failures.append(
                f"observed backend {source_field} does not match the source checkout"
            )
    for count_field in ("python_source_file_count", "python_source_read_error_count"):
        if identity.get(count_field) != source.get(count_field):
            failures.append(
                f"observed backend {count_field} does not match the source checkout"
            )
    observed_python_fingerprint = identity.get("runtime_source_hashes", {}).get(
        "python_executable_fingerprint_sha256"
    )
    accepted_python_fingerprints = runner.get(
        "accepted_python_invocation_fingerprints_sha256"
    )
    if accepted_python_fingerprints is None:
        accepted_python_fingerprints = runner.get(
            "checkout_python_invocation_fingerprints_sha256"
        )
    if (
        not isinstance(accepted_python_fingerprints, list)
        or observed_python_fingerprint not in accepted_python_fingerprints
    ):
        failures.append(
            "observed backend Python executable does not match the proof runner"
        )
    if identity.get("model_name") != requested_model:
        failures.append(
            "requested --model does not match the served /health model_name"
        )
    if identity.get("loaded_model_name") != bundle.get("model_name"):
        failures.append(
            "/health loaded_model_name does not match the independently observed bundle"
        )
    if identity.get("model_bundle_fingerprint_sha256") != bundle.get(
        "fingerprint_sha256"
    ):
        failures.append(
            "loaded model bundle fingerprint does not match --bundle-root"
        )
    if identity.get("model_bundle_files") != bundle.get("files"):
        failures.append(
            "loaded model bundle configuration files do not match --bundle-root"
        )
    return failures


def _canonical_request_payload(
    value: Any,
    *,
    normalize_previous_response_id: bool = False,
    _depth: int = 0,
) -> Any:
    """Normalize only protocol-owned volatile IDs for body comparison."""
    if isinstance(value, list):
        return [
            _canonical_request_payload(
                item,
                normalize_previous_response_id=normalize_previous_response_id,
                _depth=_depth + 1,
            )
            for item in value
        ]
    if not isinstance(value, dict):
        return value

    result: dict[str, Any] = {}
    object_type = str(value.get("type") or "")
    for key, item in value.items():
        if (
            key == "previous_response_id"
            and normalize_previous_response_id
            and _depth == 0
        ):
            result[key] = "<response-id>" if item else item
        elif key in {"call_id", "tool_call_id", "tool_use_id"} or (
            key == "id" and object_type in {"function", "function_call", "tool_use"}
        ):
            result[key] = "<tool-call-id>"
        else:
            result[key] = _canonical_request_payload(
                item,
                normalize_previous_response_id=normalize_previous_response_id,
                _depth=_depth + 1,
            )
    return result


def _public_tool_contracts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for tool in payload.get("tools") or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict):
            name = str(function.get("name") or "")
            parameters = function.get("parameters")
        elif isinstance(tool, dict):
            name = str(tool.get("name") or "")
            parameters = tool.get("parameters", tool.get("input_schema"))
        else:
            continue
        contracts.append(
            {
                "name": name,
                "parameters": copy.deepcopy(parameters),
            }
        )
    return contracts


def _public_tool_history_linkage(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose role/order/call linkage without retaining full prompt text."""
    history = payload.get("input", payload.get("messages"))
    if not isinstance(history, list):
        return []
    linkage: list[dict[str, Any]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        item_type = str(item.get("type") or "")
        if item_type == "function_call_output":
            output = str(item.get("output") or "")
            linkage.append(
                {
                    "kind": "tool_result",
                    "role": "function_call_output",
                    "call_id": str(item.get("call_id") or ""),
                    "output_chars": len(output),
                    "output_sha256": _sha256(output),
                }
            )
        for call in item.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            linkage.append(
                {
                    "kind": "assistant_tool_call",
                    "role": role,
                    "call_id": str(call.get("id") or ""),
                    "name": str(function.get("name") or ""),
                }
            )
        content = item.get("content")
        if role == "tool":
            output = str(content or "")
            linkage.append(
                {
                    "kind": "tool_result",
                    "role": role,
                    "call_id": str(
                        item.get("tool_call_id")
                        or item.get("tool_use_id")
                        or ""
                    ),
                    "name": str(item.get("name") or item.get("tool_name") or ""),
                    "output_chars": len(output),
                    "output_sha256": _sha256(output),
                }
            )
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "")
            if block_type == "tool_use":
                linkage.append(
                    {
                        "kind": "assistant_tool_call",
                        "role": role,
                        "call_id": str(block.get("id") or ""),
                        "name": str(block.get("name") or ""),
                    }
                )
            elif block_type == "tool_result":
                output = str(block.get("content") or "")
                linkage.append(
                    {
                        "kind": "tool_result",
                        "role": role,
                        "call_id": str(block.get("tool_use_id") or ""),
                        "output_chars": len(output),
                        "output_sha256": _sha256(output),
                    }
                )
    return linkage


def _request_public(
    stage: int,
    payload: dict[str, Any],
    *,
    protocol: str = "",
) -> dict[str, Any]:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    canonical = json.dumps(
        _canonical_request_payload(
            payload,
            normalize_previous_response_id=protocol == "responses",
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return {
        "stage": int(stage),
        "body_chars": len(raw),
        "body_sha256": _sha256(raw),
        "canonical_body_sha256": _sha256(canonical),
        "tool_choice": copy.deepcopy(payload.get("tool_choice")),
        "stream": bool(payload.get("stream")),
        "enable_thinking": payload.get("enable_thinking", payload.get("think")),
        "previous_response_id": (
            str(payload.get("previous_response_id"))
            if payload.get("previous_response_id")
            else None
        ),
        "max_output_tokens": payload.get(
            "max_output_tokens",
            payload.get(
                "max_tokens", (payload.get("options") or {}).get("num_predict")
            ),
        ),
        "tool_contracts": _public_tool_contracts(payload),
        "tool_history_linkage": _public_tool_history_linkage(payload),
    }


def _milliseconds(start: float) -> float:
    return round((time.monotonic() - start) * 1000, 3)


def _human_size(size: int) -> str:
    value = float(size)
    units = ("B", "KB", "MB", "GB", "TB")
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{int(value)} B"
    return f"{value:.1f} {unit}"


def _nearest_existing_directory(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            return candidate
        candidate = parent
    return candidate if candidate.is_dir() else candidate.parent


def _is_within_directory(path: Path, parent: Path) -> bool:
    """Use filesystem identity so APFS case aliases and symlinks cannot escape."""
    try:
        guarded_parent = parent.resolve(strict=True)
    except OSError:
        guarded_parent = parent.resolve()
    existing = _nearest_existing_directory(path)
    for candidate in (existing, *existing.parents):
        try:
            if candidate.samefile(guarded_parent):
                return True
        except OSError:
            continue
    return False


def _is_in_git_context(path: Path) -> bool:
    """Return whether an existing ancestor is in any worktree or Git metadata."""
    directory = _nearest_existing_directory(path)
    environment = dict(os.environ)
    for name in ("GIT_CEILING_DIRECTORIES", "GIT_DIR", "GIT_WORK_TREE"):
        environment.pop(name, None)
    environment["LC_ALL"] = "C"
    try:
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(directory),
                "rev-parse",
                "--is-inside-work-tree",
                "--is-inside-git-dir",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
    except OSError as exc:
        raise ValueError(
            "cannot safely inspect --raw-artifact-dir for Git containment"
        ) from exc
    if proc.returncode != 0:
        if "not a git repository" in proc.stderr.lower():
            return False
        raise ValueError("cannot safely inspect --raw-artifact-dir for Git containment")
    return any(line.strip().lower() == "true" for line in proc.stdout.splitlines())


def _reject_capture_destination(path: Path, guarded_worktree: Path) -> None:
    if _is_within_directory(path, guarded_worktree) or _is_in_git_context(path):
        raise ValueError(
            "--raw-artifact-dir must resolve outside every Git worktree "
            "and Git metadata directory"
        )


def _validate_private_result_destination(
    path: Path,
    guarded_worktree: Path,
) -> Path:
    """Require a new private result file outside every Git worktree."""
    output = path.expanduser().absolute()
    if output.exists():
        raise ValueError("--output must be a new file; stale result reuse is forbidden")
    _reject_capture_destination(output, guarded_worktree)
    return output


def _write_private_result_exclusive(
    path: Path,
    result: dict[str, Any],
    guarded_worktree: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _reject_capture_destination(path, guarded_worktree)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        stat.S_IRUSR | stat.S_IWUSR,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            json.dump(result, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
    except BaseException:
        with suppress(OSError):
            path.unlink(missing_ok=True)
        raise


def _git_worktree_root(repo_root: Path) -> Path:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise ValueError(
            "cannot enable raw capture without identifying the Git worktree"
        )
    return Path(proc.stdout.strip()).resolve()


def _sanitized_headers(headers: Any) -> list[dict[str, str]]:
    if headers is None:
        return []
    iterator = getattr(headers, "iteritems", None)
    pairs = iterator() if callable(iterator) else headers.items()
    return [
        {
            "name": str(name),
            "value": (
                str(value)
                if str(name).strip().lower().replace("_", "-")
                in SAFE_CAPTURE_HEADER_NAMES
                else "<redacted>"
            ),
        }
        for name, value in pairs
    ]


def _safe_request_url(value: str) -> str:
    parsed = urlparse(value)
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return parsed._replace(netloc=netloc, query="", fragment="").geturl()


def _prepared_body_bytes(body: Any) -> bytes:
    if body is None:
        return b""
    if isinstance(body, bytes):
        return body
    if isinstance(body, bytearray):
        return bytes(body)
    return str(body).encode("utf-8", errors="replace")


def _prepared_request_public(
    request_body: bytes,
    expected_payload: dict[str, Any],
    protocol: str,
) -> dict[str, Any]:
    """Bind the exact prepared request body to the public payload evidence."""
    try:
        decoded = json.loads(request_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("prepared request body is not a UTF-8 JSON object") from exc
    if not isinstance(decoded, dict):
        raise ValueError("prepared request body is not a JSON object")
    prepared_public = _request_public(0, decoded, protocol=protocol)
    prepared_public.pop("stage", None)
    expected_public = _request_public(0, expected_payload, protocol=protocol)
    expected_public.pop("stage", None)
    if prepared_public != expected_public:
        raise ValueError(
            "prepared request body does not match the declared request payload"
        )
    return prepared_public


def _safe_capture_label(value: str) -> str:
    label = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in value
    )
    return label.strip("-")[:80] or "request"


def _capture_route(
    base_label: str,
    protocol: str,
    capture_label: str,
) -> dict[str, str]:
    return {
        "base_label": str(base_label),
        "protocol": str(protocol),
        "capture_label": str(capture_label),
    }


def _capture_route_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("base_label") or ""),
        str(row.get("protocol") or ""),
        str(row.get("capture_label") or ""),
    )


class DecompressedParserInputCaptureSession:
    """Write decompressed bytes before streaming or JSON parsing."""

    def __init__(
        self,
        *,
        body_path: Path,
        metadata_path: Path,
        base_label: str,
        protocol: str,
        capture_label: str,
        payload: dict[str, Any],
        response: requests.Response,
        started: float,
        started_at: str,
        on_finished: Callable[[str | None], None],
        on_finish_error: Callable[[str], None],
    ) -> None:
        descriptor = os.open(
            body_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            stat.S_IRUSR | stat.S_IWUSR,
        )
        try:
            body_file = os.fdopen(descriptor, "wb")
        except BaseException:
            with suppress(OSError):
                os.close(descriptor)
            with suppress(OSError):
                body_path.unlink(missing_ok=True)
            raise
        try:
            self._body_file = body_file
            self._body_path = body_path
            self._metadata_path = metadata_path
            self._started = started
            self._digest = hashlib.sha256()
            self._bytes = 0
            self._first_byte_ms: float | None = None
            self._last_byte_ms: float | None = None
            self._finished = False
            self._on_finished = on_finished
            self._on_finish_error = on_finish_error

            prepared = response.request
            request_body = _prepared_body_bytes(
                prepared.body if prepared is not None else None
            )
            prepared_request_public = _prepared_request_public(
                request_body,
                payload,
                protocol,
            )
            request_public = _request_public(0, payload, protocol=protocol)
            request_public.pop("stage", None)
            response_headers = getattr(response.raw, "headers", None)
            if response_headers is None:
                response_headers = response.headers
            self._metadata: dict[str, Any] = {
                "schema_version": 1,
                "capture_layer": CAPTURE_LAYER,
                "capture_semantics": CAPTURE_SEMANTICS,
                "base_label": base_label,
                "protocol": protocol,
                "capture_label": capture_label,
                "started_at": started_at,
                "request": {
                    "method": str(prepared.method if prepared is not None else "POST"),
                    "url": _safe_request_url(
                        str(prepared.url if prepared is not None else "")
                    ),
                    "headers": _sanitized_headers(
                        prepared.headers if prepared is not None else {}
                    ),
                    "body_bytes": len(request_body),
                    "body_sha256": hashlib.sha256(request_body).hexdigest(),
                    "prepared_payload_body_sha256": prepared_request_public[
                        "body_sha256"
                    ],
                    "prepared_payload_canonical_body_sha256":
                        prepared_request_public["canonical_body_sha256"],
                    "payload": {
                        **request_public,
                        "model": str(payload.get("model") or ""),
                        "top_level_fields": sorted(str(key) for key in payload),
                    },
                },
                "response": {
                    "status_code": int(response.status_code),
                    "headers": _sanitized_headers(response_headers),
                    "headers_received_ms": _milliseconds(started),
                    "body_file": body_path.name,
                },
            }
        except BaseException:
            with suppress(Exception):
                body_file.close()
            with suppress(OSError):
                body_path.unlink(missing_ok=True)
            raise

    def write(self, chunk: bytes) -> None:
        if not chunk:
            return
        now_ms = _milliseconds(self._started)
        if self._first_byte_ms is None:
            self._first_byte_ms = now_ms
        self._last_byte_ms = now_ms
        self._body_file.write(chunk)
        self._digest.update(chunk)
        self._bytes += len(chunk)

    def finish(
        self,
        *,
        completed_ms: float,
        error_type: str | None = None,
    ) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            try:
                self._body_file.flush()
            finally:
                self._body_file.close()
            response = self._metadata["response"]
            response.update(
                {
                    "first_byte_ms": self._first_byte_ms,
                    "last_byte_ms": self._last_byte_ms,
                    "completed_ms": completed_ms,
                    "body_bytes": self._bytes,
                    "body_sha256": self._digest.hexdigest(),
                }
            )
            if error_type:
                response["capture_error_type"] = error_type
            metadata_bytes = (
                json.dumps(self._metadata, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            descriptor = os.open(
                self._metadata_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                stat.S_IRUSR | stat.S_IWUSR,
            )
            try:
                with os.fdopen(descriptor, "wb") as metadata_file:
                    metadata_file.write(metadata_bytes)
            except BaseException:
                self._metadata_path.unlink(missing_ok=True)
                raise
        except BaseException as exc:
            cleanup_error: OSError | None = None
            for artifact_path in (self._metadata_path, self._body_path):
                try:
                    artifact_path.unlink(missing_ok=True)
                except OSError as cleanup_exc:
                    if cleanup_error is None:
                        cleanup_error = cleanup_exc
            self._on_finish_error(type(exc).__name__)
            if cleanup_error is not None:
                raise cleanup_error from exc
            raise
        self._on_finished(error_type)


class DecompressedParserInputCaptureRecorder:
    """Allocate private parser-input captures outside every Git worktree."""

    def __init__(
        self,
        artifact_root: Path,
        git_worktree: Path,
        *,
        run_id: str | None = None,
    ) -> None:
        worktree = git_worktree.resolve(strict=True)
        root = artifact_root.expanduser().resolve()
        if root.exists() and not root.is_dir():
            raise ValueError("--raw-artifact-dir must name a directory")
        _reject_capture_destination(root, worktree)
        root_existed = root.exists()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root = root.resolve(strict=True)
        try:
            _reject_capture_destination(root, worktree)
        except BaseException:
            if not root_existed:
                with suppress(OSError):
                    root.rmdir()
            raise
        identifier = run_id or (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + f"-pid{os.getpid()}"
        )
        self.run_id = _safe_capture_label(identifier)
        self.run_dir = root / self.run_id
        self.run_dir.mkdir(mode=0o700)
        self.manifest_path = self.run_dir / "manifest.json"
        self._counter = 0
        self._expected_configured = False
        self._expected: list[dict[str, str]] = []
        self._started: list[dict[str, Any]] = []
        self._finished: list[dict[str, Any]] = []
        self._errors: list[dict[str, Any]] = []
        self._final_summary: dict[str, Any] | None = None

    def configure_expected(
        self,
        routes: Iterable[tuple[str, str, str]],
    ) -> None:
        if self._expected_configured or self._counter:
            raise RuntimeError(
                "capture expectations must be configured once before use"
            )
        self._expected = [
            _capture_route(base_label, protocol, capture_label)
            for base_label, protocol, capture_label in routes
        ]
        keys = [_capture_route_key(row) for row in self._expected]
        if len(keys) != len(set(keys)):
            raise ValueError("capture expectations contain duplicate route labels")
        self._expected_configured = True

    def _record_error(
        self,
        route: dict[str, str],
        *,
        phase: str,
        error_type: str,
        sequence: int | None = None,
    ) -> None:
        row: dict[str, Any] = {
            **route,
            "phase": phase,
            "error_type": error_type,
        }
        if sequence is not None:
            row["sequence"] = sequence
        self._errors.append(row)

    def _record_finished(
        self,
        started_row: dict[str, Any],
        error_type: str | None,
    ) -> None:
        self._finished.append(dict(started_row))
        if error_type:
            self._record_error(
                _capture_route(
                    started_row["base_label"],
                    started_row["protocol"],
                    started_row["capture_label"],
                ),
                phase="stream_or_parse",
                error_type=error_type,
                sequence=int(started_row["sequence"]),
            )

    def _record_finish_error(
        self,
        started_row: dict[str, Any],
        error_type: str,
    ) -> None:
        self._record_error(
            _capture_route(
                started_row["base_label"],
                started_row["protocol"],
                started_row["capture_label"],
            ),
            phase="finish",
            error_type=error_type,
            sequence=int(started_row["sequence"]),
        )

    def begin(
        self,
        *,
        base_label: str,
        protocol: str,
        capture_label: str,
        payload: dict[str, Any],
        response: requests.Response,
        started: float,
        started_at: str,
    ) -> DecompressedParserInputCaptureSession:
        if self._final_summary is not None:
            raise RuntimeError("capture recorder is already finalized")
        self._counter += 1
        route = _capture_route(base_label, protocol, capture_label)
        stem = "-".join(
            (
                f"{self._counter:04d}",
                _safe_capture_label(base_label),
                _safe_capture_label(protocol),
                _safe_capture_label(capture_label),
            )
        )
        started_row: dict[str, Any] = {
            **route,
            "sequence": self._counter,
            "body_file": f"{stem}.decompressed-parser-input.bin",
            "metadata_file": f"{stem}.metadata.json",
        }
        self._started.append(started_row)
        if self._expected_configured:
            expected = Counter(_capture_route_key(row) for row in self._expected)
            actual = Counter(_capture_route_key(row) for row in self._started)
            route_key = _capture_route_key(route)
            if route_key not in expected or actual[route_key] > expected[route_key]:
                self._record_error(
                    route,
                    phase="setup",
                    error_type="UnexpectedCaptureRoute",
                    sequence=self._counter,
                )
                raise ValueError(
                    "capture route was not expected or was started more than once"
                )
        try:
            return DecompressedParserInputCaptureSession(
                body_path=self.run_dir / started_row["body_file"],
                metadata_path=self.run_dir / started_row["metadata_file"],
                base_label=base_label,
                protocol=protocol,
                capture_label=capture_label,
                payload=payload,
                response=response,
                started=started,
                started_at=started_at,
                on_finished=lambda error_type: self._record_finished(
                    started_row,
                    error_type,
                ),
                on_finish_error=lambda error_type: self._record_finish_error(
                    started_row,
                    error_type,
                ),
            )
        except BaseException as exc:
            self._record_error(
                route,
                phase="setup",
                error_type=type(exc).__name__,
                sequence=self._counter,
            )
            raise

    def _summary(self) -> dict[str, Any]:
        expected_counts = Counter(_capture_route_key(row) for row in self._expected)
        started_counts = Counter(_capture_route_key(row) for row in self._started)
        finished_counts = Counter(_capture_route_key(row) for row in self._finished)
        ordered_keys = list(expected_counts)
        ordered_keys.extend(key for key in started_counts if key not in expected_counts)
        for error in self._errors:
            key = _capture_route_key(error)
            if key not in ordered_keys:
                ordered_keys.append(key)
        routes = []
        for key in ordered_keys:
            base_label, protocol, capture_label = key
            route_errors = [
                {
                    "phase": row["phase"],
                    "error_type": row["error_type"],
                    **({"sequence": row["sequence"]} if "sequence" in row else {}),
                }
                for row in self._errors
                if _capture_route_key(row) == key
            ]
            artifacts: list[dict[str, Any]] = []
            for row in self._started:
                if _capture_route_key(row) != key:
                    continue
                body_path = self.run_dir / row["body_file"]
                metadata_path = self.run_dir / row["metadata_file"]
                artifact: dict[str, Any] = {
                    "sequence": row["sequence"],
                    "body_file": row["body_file"],
                    "metadata_file": row["metadata_file"],
                    "verified": False,
                }
                if body_path.is_file():
                    try:
                        body_sha256, body_bytes = _sha256_file(body_path)
                    except OSError:
                        pass
                    else:
                        artifact["body_bytes"] = body_bytes
                        artifact["body_sha256"] = body_sha256
                if metadata_path.is_file():
                    try:
                        metadata_bytes = metadata_path.read_bytes()
                        artifact["metadata_sha256"] = hashlib.sha256(
                            metadata_bytes
                        ).hexdigest()
                        metadata = json.loads(metadata_bytes)
                    except (
                        OSError,
                        UnicodeDecodeError,
                        json.JSONDecodeError,
                    ):
                        metadata = {}
                    if not isinstance(metadata, dict):
                        metadata = {}
                    response = metadata.get("response")
                    request = metadata.get("request")
                    route_bound = all(
                        (
                            metadata.get("base_label") == base_label,
                            metadata.get("protocol") == protocol,
                            metadata.get("capture_label") == capture_label,
                        )
                    )
                    artifact["route_bound"] = route_bound
                    if isinstance(request, dict):
                        artifact["request_body_sha256"] = str(
                            request.get("body_sha256") or ""
                        )
                        artifact["prepared_payload_body_sha256"] = str(
                            request.get("prepared_payload_body_sha256") or ""
                        )
                        artifact[
                            "prepared_payload_canonical_body_sha256"
                        ] = str(
                            request.get(
                                "prepared_payload_canonical_body_sha256"
                            )
                            or ""
                        )
                    request_payload = (
                        request.get("payload")
                        if isinstance(request, dict)
                        else None
                    )
                    prepared_payload_matches = (
                        isinstance(request_payload, dict)
                        and artifact.get("prepared_payload_body_sha256")
                        == request_payload.get("body_sha256")
                        and artifact.get(
                            "prepared_payload_canonical_body_sha256"
                        )
                        == request_payload.get("canonical_body_sha256")
                    )
                    response_matches = (
                        isinstance(response, dict)
                        and response.get("body_file") == row["body_file"]
                        and response.get("body_bytes") == artifact.get("body_bytes")
                        and response.get("body_sha256")
                        == artifact.get("body_sha256")
                    )
                    artifact["verified"] = bool(
                        route_bound
                        and response_matches
                        and _valid_sha256(artifact.get("metadata_sha256"))
                        and _valid_sha256(artifact.get("request_body_sha256"))
                        and _valid_sha256(
                            artifact.get("prepared_payload_body_sha256")
                        )
                        and _valid_sha256(
                            artifact.get(
                                "prepared_payload_canonical_body_sha256"
                            )
                        )
                        and prepared_payload_matches
                    )
                artifacts.append(artifact)
            routes.append(
                {
                    "base_label": base_label,
                    "protocol": protocol,
                    "capture_label": capture_label,
                    "expected": expected_counts[key],
                    "started": started_counts[key],
                    "finished": finished_counts[key],
                    "errors": route_errors,
                    "artifacts": artifacts,
                }
            )
        complete = (
            self._expected_configured
            and started_counts == expected_counts
            and finished_counts == expected_counts
            and not self._errors
            and all(
                artifact.get("verified") is True
                for route in routes
                for artifact in route["artifacts"]
            )
        )
        return {
            "schema_version": 1,
            "enabled": True,
            "capture_layer": CAPTURE_LAYER,
            "capture_semantics": CAPTURE_SEMANTICS,
            "run_id": self.run_id,
            "expected": len(self._expected),
            "started": len(self._started),
            "finished": len(self._finished),
            "errors": len(self._errors),
            "complete": complete,
            "routes": routes,
        }

    def current_summary(self) -> dict[str, Any]:
        return self._summary()

    def finalize(self) -> dict[str, Any]:
        if self._final_summary is not None:
            return dict(self._final_summary)
        manifest = self._summary()
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        try:
            descriptor = os.open(
                self.manifest_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                stat.S_IRUSR | stat.S_IWUSR,
            )
            try:
                with os.fdopen(descriptor, "wb") as manifest_file:
                    manifest_file.write(manifest_bytes)
            except BaseException:
                self.manifest_path.unlink(missing_ok=True)
                raise
        except BaseException as exc:
            self.manifest_path.unlink(missing_ok=True)
            self._record_error(
                _capture_route("", "", ""),
                phase="manifest",
                error_type=type(exc).__name__,
            )
            raise
        self._final_summary = {
            **manifest,
            "manifest_file": self.manifest_path.name,
            "manifest_path": str(self.manifest_path.resolve(strict=True)),
            "run_directory": str(self.run_dir.resolve(strict=True)),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        }
        return dict(self._final_summary)


class _CapturingDecompressedStream:
    """Tee decompressed bytes before requests splits them into logical lines."""

    def __init__(
        self,
        raw: Any,
        capture: DecompressedParserInputCaptureSession,
    ) -> None:
        self._raw = raw
        self._capture = capture

    def stream(
        self,
        amount: int | None = None,
        decode_content: bool | None = None,
    ) -> Iterable[bytes]:
        for chunk in self._raw.stream(
            amount,
            decode_content=decode_content,
        ):
            capture_bytes = (
                chunk.encode("utf-8", errors="replace")
                if isinstance(chunk, str)
                else bytes(chunk)
            )
            self._capture.write(capture_bytes)
            yield chunk

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


def _parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _merge_name_fragment(existing: str, fragment: str) -> str:
    """Merge split names without duplicating full-name retransmissions."""
    if not fragment:
        return existing
    if not existing:
        return fragment
    if fragment.startswith(existing):
        return fragment
    if existing.endswith(fragment):
        return existing
    return existing + fragment


class FragmentedToolAssembler:
    """Accumulate OpenAI/Anthropic streamed function-call fragments by index."""

    def __init__(self) -> None:
        self._parts: dict[int, dict[str, Any]] = {}

    def add(
        self,
        index: int,
        *,
        call_id: str = "",
        name: str = "",
        arguments: Any = None,
        complete: bool = False,
    ) -> None:
        target = self._parts.setdefault(
            int(index),
            {"id": "", "name": "", "arguments_text": "", "arguments_object": None},
        )
        if call_id:
            target["id"] = call_id
        target["name"] = _merge_name_fragment(str(target["name"]), str(name or ""))
        if isinstance(arguments, dict):
            target["arguments_object"] = dict(arguments)
            if complete:
                target["arguments_text"] = ""
        elif arguments is not None:
            fragment = str(arguments)
            if complete:
                target["arguments_text"] = fragment
                target["arguments_object"] = None
            else:
                target["arguments_text"] += fragment

    def calls(self) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for index in sorted(self._parts):
            part = self._parts[index]
            raw: Any = part["arguments_object"]
            if raw is None:
                raw = part["arguments_text"] or "{}"
            try:
                arguments = _parse_arguments(raw)
                parse_error = None
            except Exception as exc:
                arguments = {}
                parse_error = str(exc)
            call = {
                "index": index,
                "id": str(part["id"]),
                "name": str(part["name"]),
                "arguments": arguments,
            }
            if parse_error:
                call["arguments_parse_error"] = parse_error
                call["arguments_sha256"] = _sha256(str(raw))
            calls.append(call)
        return calls


@dataclass
class EventCollector:
    protocol: str
    started: float
    events: list[dict[str, Any]] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)
    content_parts: list[str] = field(default_factory=list)
    terminals: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    response_id: str = ""
    tools: FragmentedToolAssembler = field(default_factory=FragmentedToolAssembler)

    def text(
        self, channel: str, text: str, kind: str, at_ms: float | None = None
    ) -> None:
        if not text:
            return
        if channel == "reasoning":
            self.reasoning_parts.append(text)
        elif channel == "content":
            self.content_parts.append(text)
        else:
            raise ValueError(f"unsupported text channel: {channel}")
        self.events.append(
            {
                "at_ms": _milliseconds(self.started) if at_ms is None else at_ms,
                "channel": channel,
                "kind": kind,
                "chars": len(text),
                "sha256": _sha256(text),
            }
        )

    def tool_fragment(
        self,
        index: int,
        *,
        call_id: str = "",
        name: str = "",
        arguments: Any = None,
        kind: str,
        complete: bool = False,
        at_ms: float | None = None,
    ) -> None:
        self.tools.add(
            index,
            call_id=call_id,
            name=name,
            arguments=arguments,
            complete=complete,
        )
        argument_text = (
            ""
            if arguments is None
            else json.dumps(arguments, sort_keys=True)
            if isinstance(arguments, dict)
            else str(arguments)
        )
        self.events.append(
            {
                "at_ms": _milliseconds(self.started) if at_ms is None else at_ms,
                "channel": "tool",
                "kind": kind,
                "index": int(index),
                "call_id": call_id,
                "name_fragment": name,
                "argument_chars": len(argument_text),
                "argument_sha256": _sha256(argument_text),
            }
        )

    def terminal(self, value: str, at_ms: float | None = None) -> None:
        if not value:
            return
        self.terminals.append(value)
        self.events.append(
            {
                "at_ms": _milliseconds(self.started) if at_ms is None else at_ms,
                "channel": "terminal",
                "kind": value,
            }
        )

    def error(self, kind: str, detail: str = "", at_ms: float | None = None) -> None:
        row = {
            "at_ms": _milliseconds(self.started) if at_ms is None else at_ms,
            "channel": "error",
            "kind": kind,
        }
        if detail:
            row["detail_chars"] = len(detail)
            row["detail_sha256"] = _sha256(detail)
        self.errors.append(dict(row))
        self.events.append(row)

    def result(self, status_code: int, elapsed_ms: float) -> dict[str, Any]:
        return {
            "status_code": int(status_code),
            "elapsed_ms": elapsed_ms,
            "response_id": self.response_id,
            "reasoning": "".join(self.reasoning_parts),
            "content": "".join(self.content_parts),
            "tool_calls": self.tools.calls(),
            "terminals": list(self.terminals),
            "errors": list(self.errors),
            "events": list(self.events),
        }


def tool_schemas(protocol: str, names: Iterable[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name in names:
        parameters = TOOL_PARAMETERS[name]
        if protocol == "responses":
            result.append(
                {
                    "type": "function",
                    "name": name,
                    "description": TOOL_DESCRIPTIONS[name],
                    "parameters": parameters,
                }
            )
        elif protocol == "anthropic":
            result.append(
                {
                    "name": name,
                    "description": TOOL_DESCRIPTIONS[name],
                    "input_schema": parameters,
                }
            )
        else:
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": TOOL_DESCRIPTIONS[name],
                        "parameters": parameters,
                    },
                }
            )
    return result


def tool_choice(
    protocol: str,
    mode: str,
    stage: int,
    second_tool_choice: str = "auto",
    first_tool_choice: str = "current",
) -> Any:
    """Return native choice for tool stage 1/2 or final stage 3."""
    if protocol == "ollama":
        return None
    if stage == 3:
        return {"type": "none"} if protocol == "anthropic" else "none"
    if stage == 1 and first_tool_choice == "auto":
        return {"type": "auto"} if protocol == "anthropic" else "auto"
    if stage == 1 and mode == "stream":
        if protocol == "chat":
            return {"type": "function", "function": {"name": "file_info"}}
        if protocol == "responses":
            return {"type": "function", "name": "file_info"}
        return {"type": "tool", "name": "file_info"}
    if stage == 2:
        if second_tool_choice == "explicit":
            if protocol == "chat":
                return {"type": "function", "function": {"name": "run_command"}}
            if protocol == "responses":
                return {"type": "function", "name": "run_command"}
            return {"type": "tool", "name": "run_command"}
        if second_tool_choice == "required":
            return {"type": "any"} if protocol == "anthropic" else "required"
        return {"type": "auto"} if protocol == "anthropic" else "auto"
    return {"type": "any"} if protocol == "anthropic" else "required"


def validate_allowlisted_call(
    call: dict[str, Any], expected_name: str
) -> tuple[bool, str]:
    if call.get("arguments_parse_error"):
        return False, "arguments were not valid JSON"
    if call.get("name") != expected_name:
        return False, f"expected {expected_name}, got {call.get('name')!r}"
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        return False, "arguments are not an object"
    expected = (
        {"path": FILE_INFO_PATH}
        if expected_name == "file_info"
        else {"command": PWD_COMMAND}
    )
    if arguments != expected:
        return False, f"arguments must equal {expected!r}, got {arguments!r}"
    if not call.get("id"):
        return False, "tool call has no id"
    return True, ""


def execute_allowlisted_tool(repo_root: Path, call: dict[str, Any]) -> dict[str, Any]:
    ok, error = validate_allowlisted_call(call, str(call.get("name") or ""))
    if not ok:
        raise ValueError(error)
    name = str(call["name"])
    if name == "file_info":
        target = (repo_root / FILE_INFO_PATH).resolve()
        expected = (repo_root.resolve() / FILE_INFO_PATH).resolve()
        if target != expected or not target.is_file():
            raise ValueError(f"allowed file is unavailable: {expected}")
        info = target.stat()
        result = {
            "path": FILE_INFO_PATH,
            "type": "file",
            "size_bytes": info.st_size,
            "size_human": _human_size(info.st_size),
            "modified_utc": datetime.fromtimestamp(info.st_mtime, tz=UTC).isoformat(),
            "permissions": f"{stat.S_IMODE(info.st_mode):04o}",
        }
    elif name == "run_command":
        completed = subprocess.run(
            [PWD_COMMAND],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"pwd failed with exit code {completed.returncode}")
        result = {
            "command": PWD_COMMAND,
            "stdout": completed.stdout.strip(),
            "exit_code": completed.returncode,
        }
    else:
        raise ValueError(f"tool is not allowlisted: {name}")
    output = json.dumps(result, sort_keys=True, separators=(",", ":"))
    return {
        "name": name,
        "call_id": call["id"],
        "arguments": call["arguments"],
        "result": result,
        "output": output,
    }


def assistant_message(protocol: str, round_result: dict[str, Any]) -> dict[str, Any]:
    calls = round_result.get("tool_calls") or []
    if protocol == "chat":
        message: dict[str, Any] = {
            "role": "assistant",
            "content": round_result.get("content") or "",
            "tool_calls": [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(
                            call["arguments"], separators=(",", ":")
                        ),
                    },
                }
                for call in calls
            ],
        }
        if round_result.get("reasoning"):
            message["reasoning_content"] = round_result["reasoning"]
        return message
    if protocol == "anthropic":
        blocks: list[dict[str, Any]] = []
        if round_result.get("reasoning"):
            blocks.append(
                {
                    "type": "thinking",
                    "thinking": round_result["reasoning"],
                    "signature": "dm1seA==",
                }
            )
        if round_result.get("content"):
            blocks.append({"type": "text", "text": round_result["content"]})
        blocks.extend(
            {
                "type": "tool_use",
                "id": call["id"],
                "name": call["name"],
                "input": call["arguments"],
            }
            for call in calls
        )
        return {"role": "assistant", "content": blocks}
    if protocol == "ollama":
        message = {
            "role": "assistant",
            "content": round_result.get("content") or "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"],
                    },
                }
                for call in calls
            ],
        }
        if round_result.get("reasoning"):
            message["thinking"] = round_result["reasoning"]
        return message
    raise ValueError(
        f"Responses uses function_call_output, not assistant_message: {protocol}"
    )


def history_after_tool(
    protocol: str,
    history: list[dict[str, Any]],
    round_result: dict[str, Any],
    execution: dict[str, Any],
    next_instruction: str,
) -> list[dict[str, Any]]:
    """Return protocol-native history after one real tool result."""
    if protocol == "responses":
        return [
            {
                "type": "function_call_output",
                "call_id": execution["call_id"],
                "output": execution["output"],
            },
            {"role": "user", "content": next_instruction},
        ]
    result = [*history, assistant_message(protocol, round_result)]
    if protocol == "chat":
        result.extend(
            [
                {
                    "role": "tool",
                    "tool_call_id": execution["call_id"],
                    "name": execution["name"],
                    "content": execution["output"],
                },
                {"role": "user", "content": next_instruction},
            ]
        )
    elif protocol == "anthropic":
        result.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": execution["call_id"],
                        "content": execution["output"],
                    },
                    {"type": "text", "text": next_instruction},
                ],
            }
        )
    elif protocol == "ollama":
        result.extend(
            [
                {
                    "role": "tool",
                    "tool_name": execution["name"],
                    "content": execution["output"],
                },
                {"role": "user", "content": next_instruction},
            ]
        )
    else:
        raise ValueError(f"unknown protocol: {protocol}")
    return result


def classify_terminal(
    protocol: str,
    terminals: list[str],
    *,
    stream: bool,
    expect_tool: bool,
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Require the exact protocol-native terminal suffix and event ordering."""
    values = [str(value) for value in terminals if value]
    if protocol == "chat":
        expected_semantic = ["tool_calls" if expect_tool else "stop"]
        semantic = [value for value in values if value != "DONE"]
        expected_values = [*expected_semantic, *(["DONE"] if stream else [])]
    elif protocol == "responses":
        expected_semantic = ["response.completed"]
        semantic = list(values)
        expected_values = list(expected_semantic)
    elif protocol == "anthropic":
        expected_semantic = ["tool_use" if expect_tool else "end_turn"]
        semantic = [value for value in values if value != "message_stop"]
        expected_values = [
            *expected_semantic,
            *(["message_stop"] if stream else []),
        ]
    elif protocol == "ollama":
        expected_semantic = ["tool_calls" if expect_tool else "stop"]
        semantic = list(values)
        expected_values = list(expected_semantic)
    else:
        raise ValueError(f"unknown protocol: {protocol}")

    event_values: list[str] | None = None
    post_terminal_events = 0
    event_order_pass = True
    if events is not None:
        event_values = [
            str(event.get("kind") or "")
            for event in events
            if event.get("channel") == "terminal"
        ]
        first_terminal = next(
            (
                index
                for index, event in enumerate(events)
                if event.get("channel") == "terminal"
            ),
            -1,
        )
        post_terminal_events = (
            0
            if first_terminal < 0
            else sum(
                1
                for event in events[first_terminal:]
                if event.get("channel") != "terminal"
            )
        )
        event_order_pass = (
            first_terminal >= 0
            and event_values == expected_values
            and post_terminal_events == 0
            and len(events) - first_terminal == len(expected_values)
        )

    return {
        "pass": values == expected_values and event_order_pass,
        "values": values,
        "expected": expected_values,
        "expected_semantic": expected_semantic,
        "semantic": semantic,
        "event_values": event_values,
        "post_terminal_events": post_terminal_events,
    }


def _parse_stream_object(
    protocol: str,
    data: dict[str, Any],
    event_name: str | None,
    collector: EventCollector,
    at_ms: float,
) -> None:
    if not collector.response_id:
        response = data.get("response")
        message = data.get("message")
        collector.response_id = str(
            data.get("id")
            or data.get("response_id")
            or (response.get("id") if isinstance(response, dict) else "")
            or (message.get("id") if isinstance(message, dict) else "")
            or ""
        )
    if protocol == "chat":
        for choice in data.get("choices") or []:
            delta = choice.get("delta") or {}
            collector.text(
                "reasoning",
                str(delta.get("reasoning_content") or delta.get("reasoning") or ""),
                "chat.reasoning.delta",
                at_ms,
            )
            collector.text(
                "content", str(delta.get("content") or ""), "chat.content.delta", at_ms
            )
            for fragment in delta.get("tool_calls") or []:
                function = fragment.get("function") or {}
                collector.tool_fragment(
                    int(fragment.get("index") or 0),
                    call_id=str(fragment.get("id") or ""),
                    name=str(function.get("name") or ""),
                    arguments=function.get("arguments"),
                    kind="chat.tool.delta",
                    at_ms=at_ms,
                )
            if choice.get("finish_reason") is not None:
                collector.terminal(str(choice["finish_reason"]), at_ms)
        return

    kind = str(data.get("type") or event_name or "")
    if protocol == "responses":
        if kind in {
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
            "response.reasoning.delta",
            "response.output_text.reasoning_delta",
        }:
            collector.text("reasoning", str(data.get("delta") or ""), kind, at_ms)
        elif kind == "response.output_text.delta":
            collector.text("content", str(data.get("delta") or ""), kind, at_ms)
        elif kind in {"response.output_item.added", "response.output_item.done"}:
            item = data.get("item") or {}
            if item.get("type") == "function_call":
                collector.tool_fragment(
                    int(data.get("output_index") or 0),
                    call_id=str(item.get("call_id") or item.get("id") or ""),
                    name=str(item.get("name") or ""),
                    arguments=item.get("arguments"),
                    kind=kind,
                    complete=kind == "response.output_item.done",
                    at_ms=at_ms,
                )
        elif kind == "response.function_call_arguments.delta":
            collector.tool_fragment(
                int(data.get("output_index") or 0),
                call_id=str(data.get("call_id") or ""),
                arguments=data.get("delta"),
                kind=kind,
                at_ms=at_ms,
            )
        elif kind == "response.function_call_arguments.done":
            collector.tool_fragment(
                int(data.get("output_index") or 0),
                call_id=str(data.get("call_id") or ""),
                arguments=data.get("arguments"),
                kind=kind,
                complete=True,
                at_ms=at_ms,
            )
        elif kind == "error":
            collector.error(kind, json.dumps(data, sort_keys=True), at_ms)
        elif kind in {
            "response.completed",
            "response.incomplete",
            "response.failed",
            "response.cancelled",
        }:
            collector.terminal(kind, at_ms)
        return

    if protocol == "anthropic":
        delta = data.get("delta") or {}
        if kind == "message_start":
            message = data.get("message") or {}
            collector.response_id = collector.response_id or str(
                message.get("id") or ""
            )
        elif kind == "content_block_start":
            block = data.get("content_block") or {}
            if block.get("type") == "tool_use":
                collector.tool_fragment(
                    int(data.get("index") or 0),
                    call_id=str(block.get("id") or ""),
                    name=str(block.get("name") or ""),
                    arguments=block.get("input") if block.get("input") else None,
                    kind=kind,
                    complete=bool(block.get("input")),
                    at_ms=at_ms,
                )
        elif kind == "content_block_delta" and delta.get("type") in {
            "thinking_delta",
            "reasoning_delta",
        }:
            collector.text(
                "reasoning",
                str(delta.get("thinking") or delta.get("reasoning") or ""),
                str(delta.get("type")),
                at_ms,
            )
        elif kind == "content_block_delta" and delta.get("type") == "text_delta":
            collector.text("content", str(delta.get("text") or ""), "text_delta", at_ms)
        elif kind == "content_block_delta" and delta.get("type") == "input_json_delta":
            collector.tool_fragment(
                int(data.get("index") or 0),
                arguments=delta.get("partial_json"),
                kind="input_json_delta",
                at_ms=at_ms,
            )
        elif kind == "message_delta" and delta.get("stop_reason"):
            collector.terminal(str(delta["stop_reason"]), at_ms)
        elif kind == "error":
            collector.error(kind, json.dumps(data, sort_keys=True), at_ms)
            collector.terminal(kind, at_ms)
        elif kind == "message_stop":
            collector.terminal(kind, at_ms)
        return

    if protocol == "ollama":
        if data.get("error"):
            collector.error("ollama.error", str(data.get("error")), at_ms)
        message = data.get("message") or {}
        collector.text(
            "reasoning",
            str(message.get("thinking") or message.get("reasoning") or ""),
            "ollama.thinking",
            at_ms,
        )
        collector.text(
            "content", str(message.get("content") or ""), "ollama.content", at_ms
        )
        for index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            collector.tool_fragment(
                index,
                call_id=str(call.get("id") or f"ollama_call_{index}"),
                name=str(function.get("name") or ""),
                arguments=function.get("arguments") or {},
                kind="ollama.tool",
                complete=True,
                at_ms=at_ms,
            )
        if data.get("done"):
            collector.terminal(str(data.get("done_reason") or "stop"), at_ms)
        return

    raise ValueError(f"unknown protocol: {protocol}")


def parse_nonstream(
    protocol: str, body: dict[str, Any], status_code: int, elapsed_ms: float
) -> dict[str, Any]:
    started = time.monotonic()
    collector = EventCollector(protocol=protocol, started=started)
    collector.response_id = str(body.get("id") or "")
    if protocol == "chat":
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        collector.text(
            "reasoning",
            str(message.get("reasoning_content") or message.get("reasoning") or ""),
            "chat.reasoning.complete",
            elapsed_ms,
        )
        collector.text(
            "content",
            str(message.get("content") or ""),
            "chat.content.complete",
            elapsed_ms,
        )
        for index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            collector.tool_fragment(
                index,
                call_id=str(call.get("id") or ""),
                name=str(function.get("name") or ""),
                arguments=function.get("arguments"),
                kind="chat.tool.complete",
                complete=True,
                at_ms=elapsed_ms,
            )
        collector.terminal(str(choice.get("finish_reason") or ""), elapsed_ms)
    elif protocol == "responses":
        for item_index, item in enumerate(body.get("output") or []):
            if item.get("type") == "reasoning":
                for summary in item.get("summary") or []:
                    collector.text(
                        "reasoning",
                        str(summary.get("text") or ""),
                        "responses.reasoning.complete",
                        elapsed_ms,
                    )
            elif item.get("type") == "message":
                for part in item.get("content") or []:
                    if part.get("type") in {"output_text", "text"}:
                        collector.text(
                            "content",
                            str(part.get("text") or ""),
                            "responses.content.complete",
                            elapsed_ms,
                        )
            elif item.get("type") == "function_call":
                collector.tool_fragment(
                    item_index,
                    call_id=str(item.get("call_id") or item.get("id") or ""),
                    name=str(item.get("name") or ""),
                    arguments=item.get("arguments"),
                    kind="responses.tool.complete",
                    complete=True,
                    at_ms=elapsed_ms,
                )
        response_status = body.get("status")
        if isinstance(response_status, str) and response_status:
            collector.terminal(f"response.{response_status}", elapsed_ms)
        else:
            collector.error(
                "responses.missing_status",
                "Responses nonstream payload omitted explicit status",
                elapsed_ms,
            )
    elif protocol == "anthropic":
        for index, block in enumerate(body.get("content") or []):
            if block.get("type") in {"thinking", "reasoning"}:
                collector.text(
                    "reasoning",
                    str(
                        block.get("thinking")
                        or block.get("reasoning")
                        or block.get("text")
                        or ""
                    ),
                    "anthropic.reasoning.complete",
                    elapsed_ms,
                )
            elif block.get("type") == "text":
                collector.text(
                    "content",
                    str(block.get("text") or ""),
                    "anthropic.content.complete",
                    elapsed_ms,
                )
            elif block.get("type") == "tool_use":
                collector.tool_fragment(
                    index,
                    call_id=str(block.get("id") or ""),
                    name=str(block.get("name") or ""),
                    arguments=block.get("input") or {},
                    kind="anthropic.tool.complete",
                    complete=True,
                    at_ms=elapsed_ms,
                )
        collector.terminal(str(body.get("stop_reason") or ""), elapsed_ms)
    elif protocol == "ollama":
        _parse_stream_object(protocol, body, None, collector, elapsed_ms)
    else:
        raise ValueError(f"unknown protocol: {protocol}")
    return collector.result(status_code, elapsed_ms)


class ProtocolClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        timeout: int,
        *,
        base_label: str = "",
        raw_recorder: DecompressedParserInputCaptureRecorder | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.base_label = base_label
        self.raw_recorder = raw_recorder
        self.headers = {"content-type": "application/json"}
        if api_key:
            self.headers["authorization"] = f"Bearer {api_key}"

    @staticmethod
    def route(protocol: str) -> str:
        return {
            "chat": "/v1/chat/completions",
            "responses": "/v1/responses",
            "anthropic": "/v1/messages",
            "ollama": "/api/chat",
        }[protocol]

    def _attach_parser_input_capture(
        self,
        *,
        protocol: str,
        capture_label: str,
        payload: dict[str, Any],
        response: requests.Response,
        started: float,
        started_at: str,
    ) -> DecompressedParserInputCaptureSession | None:
        if self.raw_recorder is None:
            return None
        capture = self.raw_recorder.begin(
            base_label=self.base_label,
            protocol=protocol,
            capture_label=capture_label,
            payload=payload,
            response=response,
            started=started,
            started_at=started_at,
        )
        try:
            response.raw = _CapturingDecompressedStream(response.raw, capture)
        except BaseException as exc:
            capture.finish(
                completed_ms=_milliseconds(started),
                error_type=type(exc).__name__,
            )
            raise
        return capture

    def send(
        self,
        protocol: str,
        payload: dict[str, Any],
        stream: bool,
        *,
        capture_label: str = "request",
        prepared_body: bytes | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        started_at = datetime.now(UTC).isoformat()
        prepared_sha256 = ""
        if prepared_body is None:
            response = requests.post(
                self.base_url + self.route(protocol),
                headers=self.headers,
                json=payload,
                stream=stream,
                timeout=(15, self.timeout),
            )
        else:
            response = requests.post(
                self.base_url + self.route(protocol),
                headers=self.headers,
                data=prepared_body,
                stream=True,
                timeout=(15, self.timeout),
            )
            observed = _prepared_body_bytes(
                response.request.body if response.request is not None else None
            )
            if observed != prepared_body:
                with suppress(Exception):
                    response.close()
                raise ValueError("prepared request body changed before transport")
            prepared_sha256 = hashlib.sha256(observed).hexdigest()
        if not stream:
            capture: DecompressedParserInputCaptureSession | None = None
            capture_error_type: str | None = None
            completed_ms = 0.0
            try:
                if self.raw_recorder is not None:
                    capture = self.raw_recorder.begin(
                        base_label=self.base_label,
                        protocol=protocol,
                        capture_label=capture_label,
                        payload=payload,
                        response=response,
                        started=started,
                        started_at=started_at,
                    )
                raw_body = bytes(response.content)
                if capture is not None:
                    capture.write(raw_body)
                try:
                    decoded = raw_body.decode(response.encoding or "utf-8")
                    body = json.loads(decoded)
                    if not isinstance(body, dict):
                        raise ValueError("nonstream response JSON is not an object")
                except Exception:
                    body = {
                        "error": raw_body.decode(
                            response.encoding or "utf-8",
                            errors="replace",
                        )[:2000]
                    }
                completed_ms = _milliseconds(started)
                result = parse_nonstream(
                    protocol,
                    body,
                    response.status_code,
                    completed_ms,
                )
            except BaseException as exc:
                capture_error_type = type(exc).__name__
                raise
            finally:
                try:
                    response.close()
                except BaseException as exc:
                    if capture_error_type is None:
                        capture_error_type = type(exc).__name__
                    raise
                finally:
                    completed_ms = _milliseconds(started)
                    if capture is not None:
                        capture.finish(
                            completed_ms=completed_ms,
                            error_type=capture_error_type,
                        )
            if prepared_sha256:
                result["_prepared_request_body_sha256"] = prepared_sha256
            return result

        collector = EventCollector(protocol=protocol, started=started)
        event_name: str | None = None
        capture: DecompressedParserInputCaptureSession | None = None
        capture_error_type: str | None = None
        completed_ms = 0.0
        try:
            capture = self._attach_parser_input_capture(
                protocol=protocol,
                capture_label=capture_label,
                payload=payload,
                response=response,
                started=started,
                started_at=started_at,
            )
            for raw in response.iter_lines(decode_unicode=True, chunk_size=1):
                if raw is None:
                    continue
                line = (
                    raw.decode("utf-8", errors="replace")
                    if isinstance(raw, bytes)
                    else raw
                )
                line = line.strip()
                if not line:
                    event_name = None
                    continue
                at_ms = _milliseconds(started)
                if protocol != "ollama" and line.startswith("event: "):
                    event_name = line[7:]
                    continue
                if protocol != "ollama":
                    if not line.startswith("data: "):
                        continue
                    raw_data = line[6:]
                    if raw_data == "[DONE]":
                        collector.terminal("DONE", at_ms)
                        continue
                else:
                    raw_data = line
                try:
                    data = json.loads(raw_data)
                except json.JSONDecodeError:
                    collector.error("json_parse_error", raw_data, at_ms)
                    continue
                _parse_stream_object(
                    protocol,
                    data,
                    event_name,
                    collector,
                    at_ms,
                )
        except BaseException as exc:
            capture_error_type = type(exc).__name__
            raise
        finally:
            try:
                response.close()
            except BaseException as exc:
                if capture_error_type is None:
                    capture_error_type = type(exc).__name__
                raise
            finally:
                completed_ms = _milliseconds(started)
                if capture is not None:
                    capture.finish(
                        completed_ms=completed_ms,
                        error_type=capture_error_type,
                    )
        result = collector.result(response.status_code, completed_ms)
        if prepared_sha256:
            result["_prepared_request_body_sha256"] = prepared_sha256
        return result


def frozen_request_body(payload: dict[str, Any]) -> bytes:
    """Serialize one request once for byte-identical A1/B/A2 replay."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _paired_digest(value: Any, label: str, *, json_value: bool = False) -> dict[str, Any]:
    if json_value:
        try:
            text = json.dumps(
                value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"paired replay {label} is not JSON") from exc
    else:
        if not isinstance(value, str):
            raise ValueError(f"paired replay {label} is not text")
        text = value
    return {"chars": len(text), "sha256": _sha256(text)}


def _prefixed_digest(prefix: str, digest: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in digest.items()}


def _paired_stable_id(
    value: Any,
    id_map: dict[str, str],
    label: str,
) -> str:
    if value is None or value == "":
        return ""
    if not isinstance(value, str):
        raise ValueError(f"paired replay {label} is not text")
    if value not in id_map:
        id_map[value] = f"tool_call_{len(id_map) + 1}"
    return id_map[value]


def _normalized_replay_response(result: dict[str, Any]) -> dict[str, Any]:
    """Retain response semantics while removing timing/envelope IDs."""
    call_ids: dict[str, str] = {}
    collections = {}
    for collection_name in (
        "tool_calls",
        "notices",
        "terminals",
        "errors",
        "events",
    ):
        value = (
            []
            if collection_name == "notices" and collection_name not in result
            else result.get(collection_name)
        )
        if not isinstance(value, list):
            raise ValueError(
                f"paired replay raw {collection_name} is not a list"
            )
        collections[collection_name] = value

    tools = []
    for source in collections["tool_calls"]:
        if not isinstance(source, dict):
            tools.append(copy.deepcopy(source))
            continue
        call = {
            key: copy.deepcopy(value)
            for key, value in source.items()
            if key not in {"id", "call_id"}
        }
        raw_call_id = source.get("id")
        if raw_call_id is None or raw_call_id == "":
            raw_call_id = source.get("call_id")
        call_id = _paired_stable_id(
            raw_call_id,
            call_ids,
            "response tool call id",
        )
        if call_id:
            call["call_id"] = call_id
        tools.append(call)
    events = []
    for source in collections["events"]:
        if not isinstance(source, dict):
            events.append(copy.deepcopy(source))
            continue
        event = {
            key: copy.deepcopy(value)
            for key, value in source.items()
            if key
            not in {"at_ms", "elapsed_ms", "started_at", "completed_ms", "response_id"}
        }
        if "call_id" in source:
            call_id = _paired_stable_id(
                source["call_id"],
                call_ids,
                "response event call id",
            )
            if call_id:
                event["call_id"] = call_id
        events.append(event)
    def strip_timing(row: dict[str, Any]) -> dict[str, Any]:
        return {
            key: copy.deepcopy(value)
            for key, value in row.items()
            if key not in {"at_ms", "elapsed_ms", "started_at", "completed_ms"}
        }
    return {
        "status_code": result.get("status_code"),
        "reasoning": result.get("reasoning"),
        "content": result.get("content"),
        "notices": copy.deepcopy(collections["notices"]),
        "tool_calls": tools,
        "terminals": copy.deepcopy(collections["terminals"]),
        "errors": [
            strip_timing(row) if isinstance(row, dict) else copy.deepcopy(row)
            for row in collections["errors"]
        ],
        "events": events,
    }


def _paired_public_event(source: Any, label: str) -> dict[str, Any]:
    if not isinstance(source, dict):
        raise ValueError(f"paired replay {label} row is not an object")
    allowed = {
        "channel",
        "kind",
        "index",
        "call_id",
        "name_fragment",
        "chars",
        "sha256",
        "argument_chars",
        "argument_sha256",
        "detail_chars",
        "detail_sha256",
    }
    public: dict[str, Any] = {}
    unknown_text = []
    for key, value in source.items():
        if not isinstance(key, str):
            raise ValueError(f"paired replay {label} field name is not text")
        if key not in allowed:
            if not isinstance(value, str):
                raise ValueError(f"paired replay {label} has an unsupported field")
            unknown_text.append(
                {
                    **_prefixed_digest("field_name", _paired_digest(key, "field name")),
                    **_prefixed_digest("value", _paired_digest(value, "field value")),
                }
            )
        elif key == "channel":
            if value not in PAIRED_REPLAY_EVENT_CHANNELS:
                raise ValueError(f"paired replay {label} channel is unsupported")
            public[key] = value
        elif key == "kind":
            if value not in PAIRED_REPLAY_EVENT_KINDS:
                raise ValueError(f"paired replay {label} kind is unsupported")
            public[key] = value
        elif key == "name_fragment":
            if value not in PAIRED_REPLAY_TOOL_NAMES:
                raise ValueError(f"paired replay {label} tool name is unsupported")
            public[key] = value
        elif key == "call_id":
            if not (
                isinstance(value, str)
                and value.startswith("tool_call_")
                and value.removeprefix("tool_call_").isdigit()
            ):
                raise ValueError(f"paired replay {label} linkage is invalid")
            public[key] = value
        elif key.endswith("_sha256") or key == "sha256":
            if not _valid_sha256(value):
                raise ValueError(f"paired replay {label} digest is invalid")
            public[key] = value
        elif not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"paired replay {label} count is invalid")
        else:
            public[key] = value
    if unknown_text:
        public["unknown_text_fields"] = unknown_text
    return public


def _paired_public_response(value: dict[str, Any]) -> dict[str, Any]:
    status = value.get("status_code")
    if not isinstance(status, int) or isinstance(status, bool) or status < 0:
        raise ValueError("paired replay response status is invalid")
    reasoning = _paired_digest(value.get("reasoning"), "reasoning")
    content = _paired_digest(value.get("content"), "content")
    notices = value.get("notices")
    tools = value.get("tool_calls")
    terminals = value.get("terminals")
    errors = value.get("errors")
    events = value.get("events")
    if (
        not isinstance(notices, list)
        or not isinstance(tools, list)
        or not isinstance(errors, list)
        or not isinstance(events, list)
    ):
        raise ValueError("paired replay response lists are invalid")
    if not isinstance(terminals, list) or any(
        not isinstance(item, str) or item not in PAIRED_REPLAY_TERMINALS
        for item in terminals
    ):
        raise ValueError("paired replay terminal is unsupported")
    public_tools = []
    for source in tools:
        if not isinstance(source, dict):
            raise ValueError("paired replay tool call is not an object")
        if set(source) - {
            "index",
            "call_id",
            "name",
            "arguments",
            "arguments_parse_error",
            "arguments_sha256",
        }:
            raise ValueError("paired replay tool call has unsupported fields")
        index, name, arguments = (
            source.get("index"),
            source.get("name"),
            source.get("arguments"),
        )
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("paired replay tool index is invalid")
        if name not in PAIRED_REPLAY_TOOL_NAMES or not isinstance(arguments, dict):
            raise ValueError("paired replay tool name or arguments are invalid")
        call_id = source.get("call_id")
        if call_id and not (
            isinstance(call_id, str)
            and call_id.startswith("tool_call_")
            and call_id.removeprefix("tool_call_").isdigit()
        ):
            raise ValueError("paired replay tool linkage is invalid")
        tool = {
            "index": index,
            "call_id": call_id or None,
            "name": name,
            **_prefixed_digest(
                "arguments", _paired_digest(arguments, "tool arguments", json_value=True)
            ),
        }
        parse_error = source.get("arguments_parse_error")
        if parse_error is not None:
            tool.update(
                _prefixed_digest(
                    "arguments_parse_error",
                    _paired_digest(parse_error, "tool parse error"),
                )
            )
        raw_arguments_sha = source.get("arguments_sha256")
        if raw_arguments_sha is not None:
            if not _valid_sha256(raw_arguments_sha):
                raise ValueError("paired replay raw arguments digest is invalid")
            tool["raw_arguments_sha256"] = raw_arguments_sha
        public_tools.append(tool)
    return {
        "semantic_sha256": _canonical_sha256(value),
        "status_code": status,
        **_prefixed_digest("reasoning", reasoning),
        **_prefixed_digest("content", content),
        "notices": [_paired_digest(item, "notice") for item in notices],
        "tool_calls": public_tools,
        "terminals": list(terminals),
        "errors": [
            _paired_public_event(row, "error") for row in errors
        ],
        "events": [
            _paired_public_event(row, "event") for row in events
        ],
    }


def _paired_public_history(payload: dict[str, Any]) -> list[dict[str, Any]]:
    history = payload["input"] if "input" in payload else payload.get("messages")
    if not isinstance(history, list):
        raise ValueError("paired replay history is not a list")
    call_ids: dict[str, str] = {}
    linkage: list[dict[str, Any]] = []

    def append_link(
        kind: str,
        role: Any,
        *,
        call_id: Any = None,
        name: Any = None,
        output: Any = None,
    ) -> None:
        if role not in {"assistant", "tool", "function_call_output", "user"}:
            raise ValueError("paired replay history tool role is unsupported")
        row = {"kind": kind, "role": role}
        stable_id = _paired_stable_id(call_id, call_ids, "history tool call id")
        if stable_id:
            row.update(
                {
                    "call_id": stable_id,
                    **_prefixed_digest(
                        "call_id", _paired_digest(call_id, "history tool call id")
                    ),
                }
            )
        if name is not None:
            row.update(
                _prefixed_digest(
                    "name", _paired_digest(name, "history tool name")
                )
            )
        if output is not None:
            row.update(
                _prefixed_digest(
                    "output", _paired_digest(output, "history tool output")
                )
            )
        linkage.append(row)

    for item in history:
        if not isinstance(item, dict):
            raise ValueError("paired replay history item is not an object")
        role = item.get("role")
        item_type = item.get("type")
        if not isinstance(role, str) or (
            item_type is not None and not isinstance(item_type, str)
        ):
            raise ValueError("paired replay history role/type is invalid")
        content = item.get("content")
        if content is not None and not isinstance(content, (str, list)):
            raise ValueError("paired replay history content has an invalid shape")
        if "tool_calls" in item:
            calls = item["tool_calls"]
            if not isinstance(calls, list):
                raise ValueError("paired replay history tool_calls is not a list")
            for call in calls:
                if not isinstance(call, dict):
                    raise ValueError(
                        "paired replay history tool call is not an object"
                    )
                function = call.get("function")
                if not isinstance(function, dict):
                    raise ValueError(
                        "paired replay history tool function is not an object"
                    )
                arguments = function.get("arguments")
                if arguments is not None and not isinstance(arguments, (str, dict)):
                    raise ValueError(
                        "paired replay history tool arguments are invalid"
                    )
                append_link(
                    "assistant_tool_call",
                    role,
                    call_id=call.get("id"),
                    name=function.get("name"),
                )
        if item_type == "function_call_output":
            append_link(
                "tool_result",
                "function_call_output",
                call_id=item.get("call_id"),
                output=item.get("output"),
            )
        if role == "tool":
            if not isinstance(content, str):
                raise ValueError("paired replay tool result content is not text")
            call_id = item.get("tool_call_id")
            if call_id is None or call_id == "":
                call_id = item.get("tool_use_id")
            name = item.get("name")
            if name is None:
                name = item.get("tool_name")
            append_link(
                "tool_result",
                role,
                call_id=call_id,
                name=name,
                output=content,
            )
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                raise ValueError("paired replay history block is not an object")
            block_type = block.get("type")
            if block_type is not None and not isinstance(block_type, str):
                raise ValueError("paired replay history block type is invalid")
            if block_type == "tool_use":
                tool_input = block.get("input")
                if tool_input is not None and not isinstance(tool_input, dict):
                    raise ValueError("paired replay history tool input is invalid")
                append_link(
                    "assistant_tool_call",
                    role,
                    call_id=block.get("id"),
                    name=block.get("name"),
                )
            elif block_type == "tool_result":
                append_link(
                    "tool_result",
                    role,
                    call_id=block.get("tool_use_id"),
                    output=block.get("content"),
                )
    return linkage


def _paired_public_request(
    stage: int,
    payload: dict[str, Any],
    protocol: str = "",
) -> dict[str, Any]:
    raw = frozen_request_body(payload).decode("utf-8")
    canonical = json.dumps(
        _canonical_request_payload(
            payload,
            normalize_previous_response_id=protocol == "responses",
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    choice = payload.get("tool_choice")
    if isinstance(choice, str):
        if choice not in {"auto", "none", "required", "any"}:
            raise ValueError("paired replay tool choice is unsupported")
        public_choice: dict[str, Any] | None = {"kind": choice}
    elif choice is None:
        public_choice = None
    elif isinstance(choice, dict) and choice.get("type") in {"function", "tool"}:
        function = choice.get("function", choice)
        if not isinstance(function, dict):
            raise ValueError("paired replay named tool choice is invalid")
        public_choice = {
            "kind": "named_tool",
            "type": choice["type"],
            **_prefixed_digest(
                "name", _paired_digest(function.get("name"), "tool choice name")
            ),
            **_prefixed_digest(
                "choice", _paired_digest(choice, "tool choice", json_value=True)
            ),
        }
    else:
        raise ValueError("paired replay tool choice has an invalid shape")
    contracts = []
    tools = payload.get("tools") if "tools" in payload else []
    if not isinstance(tools, list):
        raise ValueError("paired replay tool contracts are not a list")
    for index, source in enumerate(tools):
        if not isinstance(source, dict):
            raise ValueError("paired replay tool contract is not an object")
        function = source.get("function", source)
        if not isinstance(function, dict) or source.get("type", "function") != "function":
            raise ValueError("paired replay tool contract is invalid")
        contracts.append(
            {
                "index": index,
                "kind": "function",
                **_prefixed_digest(
                    "name", _paired_digest(function.get("name"), "contract name")
                ),
                **_prefixed_digest(
                    "description",
                    _paired_digest(
                        function.get("description", source.get("description", "")),
                        "contract description",
                    ),
                ),
                **_prefixed_digest(
                    "schema",
                    _paired_digest(
                        function.get(
                            "parameters",
                            source.get("parameters", source.get("input_schema")),
                        ),
                        "contract schema",
                        json_value=True,
                    ),
                ),
                **_prefixed_digest(
                    "contract",
                    _paired_digest(source, "tool contract", json_value=True),
                ),
            }
        )
    linkage = _paired_public_history(payload)
    previous = payload.get("previous_response_id")
    previous_public = (
        None
        if previous is None or previous == ""
        else {
            "linkage": "response_id_1",
            **_paired_digest(previous, "previous response id"),
        }
    )
    thinking = payload.get("enable_thinking", payload.get("think"))
    if (
        thinking is not None
        and not isinstance(thinking, bool)
        and thinking not in {"auto", "on", "off"}
    ):
        raise ValueError("paired replay thinking mode is unsupported")
    options = payload.get("options")
    if options is not None and not isinstance(options, dict):
        raise ValueError("paired replay options are invalid")
    max_output = payload.get(
        "max_output_tokens",
        payload.get("max_tokens", (options or {}).get("num_predict")),
    )
    if max_output is not None and (
        not isinstance(max_output, int)
        or isinstance(max_output, bool)
        or max_output < 0
    ):
        raise ValueError("paired replay max output token count is invalid")
    return {
        "stage": int(stage),
        "body_chars": len(raw),
        "body_sha256": _sha256(raw),
        "canonical_body_sha256": _sha256(canonical),
        "tool_choice": public_choice,
        "stream": payload["stream"],
        "enable_thinking": thinking,
        "previous_response_id": previous_public,
        "max_output_tokens": max_output,
        "tool_contracts": contracts,
        "tool_history_linkage": linkage,
    }


def _request_lifecycle_view(
    health: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Validate the engine-owned lifecycle attestation from ``/health``.

    Prefix-cache request tables are deliberately excluded: their admission and
    cleanup boundaries differ from the engine collector and scheduler.
    """

    lifecycle = health.get("request_lifecycle")
    if not isinstance(lifecycle, dict):
        return {}, ["/health request_lifecycle is missing"]
    failures: list[str] = []
    if lifecycle.get("schema") != "vmlx-request-lifecycle-v1":
        failures.append("/health request_lifecycle schema is invalid")
    if lifecycle.get("request_id_encoding") != "sha256-utf8-lowerhex":
        failures.append(
            "/health request_lifecycle request_id_encoding is invalid"
        )
    if lifecycle.get("available") is not True:
        failures.append("/health request_lifecycle is unavailable")

    def request_id_hashes(field: str) -> list[str]:
        value = lifecycle.get(field)
        if (
            not isinstance(value, list)
            or any(
                not isinstance(item, str)
                or item != item.lower()
                or not _valid_sha256(item)
                for item in value
            )
            or value != sorted(set(value))
        ):
            failures.append(f"/health request_lifecycle.{field} is invalid")
            return []
        return list(value)

    def count(field: str, expected: int) -> int | None:
        value = lifecycle.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            failures.append(f"/health request_lifecycle.{field} is invalid")
            return None
        if value != expected:
            failures.append(
                f"/health request_lifecycle.{field} does not match its ID list"
            )
        return value

    collector_ids = request_id_hashes(
        "engine_collector_request_ids_sha256"
    )
    waiting_ids = request_id_hashes(
        "scheduler_waiting_request_ids_sha256"
    )
    running_ids = request_id_hashes(
        "scheduler_running_request_ids_sha256"
    )
    active_ids = request_id_hashes("active_request_ids_sha256")
    collector_count = count("engine_collector_count", len(collector_ids))
    waiting_count = count("scheduler_waiting_count", len(waiting_ids))
    running_count = count("scheduler_running_count", len(running_ids))
    active_count = count("active_request_count", len(active_ids))
    cleanup_pending = lifecycle.get("terminal_cleanup_pending")
    if not isinstance(cleanup_pending, bool):
        failures.append(
            "/health request_lifecycle.terminal_cleanup_pending is invalid"
        )

    expected_active_ids = sorted(
        set(collector_ids) | set(waiting_ids) | set(running_ids)
    )
    if active_ids != expected_active_ids:
        failures.append(
            "/health request_lifecycle.active_request_ids_sha256 is inconsistent"
        )

    running_rows = lifecycle.get("scheduler_running_requests")
    running_row_ids: list[str] = []
    if not isinstance(running_rows, list):
        failures.append(
            "/health request_lifecycle.scheduler_running_requests is invalid"
        )
    else:
        for row in running_rows:
            if (
                not isinstance(row, dict)
                or set(row) - {"request_id_sha256", "status"}
                or (
                    "status" in row
                    and (
                        not isinstance(row["status"], str)
                        or not row["status"]
                    )
                )
            ):
                failures.append(
                    "/health request_lifecycle.scheduler_running_requests is invalid"
                )
                continue
            request_id = row.get("request_id_sha256")
            if (
                not isinstance(request_id, str)
                or request_id != request_id.lower()
                or not _valid_sha256(request_id)
                or request_id not in running_ids
                or request_id in running_row_ids
            ):
                failures.append(
                    "/health request_lifecycle.scheduler_running_requests is invalid"
                )
                continue
            running_row_ids.append(request_id)
        if running_row_ids != running_ids:
            failures.append(
                "/health request_lifecycle.scheduler_running_requests "
                "does not match scheduler_running_request_ids_sha256"
            )

    return {
        "engine_collector_count": collector_count,
        "engine_collector_request_ids_sha256": collector_ids,
        "scheduler_waiting_count": waiting_count,
        "scheduler_waiting_request_ids_sha256": waiting_ids,
        "scheduler_running_count": running_count,
        "scheduler_running_request_ids_sha256": running_ids,
        "scheduler_running_requests": running_rows,
        "active_request_count": active_count,
        "active_request_ids_sha256": active_ids,
        "terminal_cleanup_pending": cleanup_pending,
    }, failures


def _response_request_id_hashes(response_id: str) -> set[str]:
    """Return public hashes for every documented Chat engine pass."""

    if not response_id:
        return set()
    return {
        _sha256(response_id),
        _sha256(f"{response_id}:visible-answer"),
        _sha256(f"{response_id}-xml-retry"),
        _sha256(f"{response_id}-json-retry"),
    }


def _last_cache_execution_request_id_sha256(
    health: dict[str, Any],
) -> tuple[str | None, list[str]]:
    """Return the durable admitted-request ID without retaining raw IDs.

    A fully cached non-streaming replay can finish between two health polls.
    The scheduler's last-cache-execution record is updated at admission for
    misses as well as hits and remains visible after terminal cleanup, so it
    closes that observation gap without treating a successful HTTP response as
    proof by itself.  The raw request ID never enters the private artifact.
    """

    scheduler = health.get("scheduler")
    if not isinstance(scheduler, dict):
        return None, []
    execution = scheduler.get("last_cache_execution")
    if execution is None:
        return None, []
    if not isinstance(execution, dict):
        return None, ["/health scheduler.last_cache_execution is invalid"]
    request_id = execution.get("request_id")
    if request_id is None:
        return None, []
    if not isinstance(request_id, str) or not request_id:
        return None, [
            "/health scheduler.last_cache_execution.request_id is invalid"
        ]
    return _sha256(request_id), []


def _paired_gateway_lifecycle(
    action: Callable[[], dict[str, Any]],
    probe: Callable[[], dict[str, Any]],
    expected_fingerprint: str,
    timeout_s: float,
    poll_interval_s: float,
    *,
    protocol: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ready = threading.Event()
    action_started = threading.Event()
    action_finished = threading.Event()
    stop = threading.Event()
    samples: list[dict[str, Any]] = []
    failures: list[str] = []
    observed_action_request_ids: set[str] = set()
    state = {
        "baseline_settled": False,
        "active_seen": False,
        "durable_completion_seen": False,
        "baseline_last_cache_execution_request_id_sha256": None,
        "final_idle_settled": False,
        "action_executed": False,
    }
    started = time.monotonic()

    def observe(phase: str) -> dict[str, Any]:
        health = probe()
        identity, identity_failures = _health_identity(health)
        lifecycle, lifecycle_failures = _request_lifecycle_view(health)
        (
            last_cache_execution_request_id_sha256,
            cache_execution_failures,
        ) = _last_cache_execution_request_id_sha256(health)
        lifecycle_failures = [*lifecycle_failures, *cache_execution_failures]
        fingerprint = identity.get("fingerprint_sha256")
        row = {
            "phase": phase,
            "at_ms": _milliseconds(started),
            "full_sha256": _canonical_sha256(health),
            "identity_fingerprint_sha256": (
                fingerprint if _valid_sha256(fingerprint) else None
            ),
            "identity_failures": identity_failures,
            "lifecycle_failures": lifecycle_failures,
            "last_cache_execution_request_id_sha256": (
                last_cache_execution_request_id_sha256
            ),
            **lifecycle,
            # Compatibility aliases for readers of the original V5 artifact.
            "num_running": lifecycle.get("scheduler_running_count"),
            "active_requests": lifecycle.get("engine_collector_count"),
        }
        samples.append(row)
        if identity_failures:
            failures.append(f"{phase}: backend identity capture failed")
        elif fingerprint != expected_fingerprint:
            failures.append(f"{phase}: backend identity fingerprint mismatch")
        if lifecycle_failures:
            failures.append(f"{phase}: request lifecycle capture failed")
        return row

    def is_idle(row: dict[str, Any]) -> bool:
        return (
            row.get("active_request_count") == 0
            and row.get("terminal_cleanup_pending") is False
        )

    def bounded_single_owner(row: dict[str, Any], phase: str) -> bool:
        counts = (
            row.get("engine_collector_count"),
            row.get("scheduler_waiting_count"),
            row.get("scheduler_running_count"),
            row.get("active_request_count"),
        )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or value > 1
            for value in counts
        ):
            failures.append(f"{phase}: concurrent or malformed engine activity")
            return False
        active_ids = row.get("active_request_ids_sha256")
        if not isinstance(active_ids, list):
            failures.append(f"{phase}: active request IDs are unavailable")
            return False
        if active_ids and phase != "before":
            observed_action_request_ids.update(active_ids)
            state["active_seen"] = True
        durable_id = row.get("last_cache_execution_request_id_sha256")
        if (
            phase != "before"
            and isinstance(durable_id, str)
            and durable_id
            and durable_id
            != state["baseline_last_cache_execution_request_id_sha256"]
        ):
            observed_action_request_ids.add(durable_id)
            state["durable_completion_seen"] = True
        return True

    def poll() -> None:
        baseline_deadline = time.monotonic() + timeout_s
        idle_streak = 0
        while time.monotonic() < baseline_deadline and not stop.is_set():
            try:
                baseline = observe("before")
            except Exception as exc:
                failures.append(
                    f"before: health probe raised {type(exc).__name__}"
                )
                ready.set()
                return
            if baseline["identity_failures"] or baseline["lifecycle_failures"]:
                ready.set()
                return
            if not bounded_single_owner(baseline, "before"):
                ready.set()
                return
            if is_idle(baseline):
                idle_streak += 1
                if idle_streak >= 2:
                    state["baseline_settled"] = True
                    state[
                        "baseline_last_cache_execution_request_id_sha256"
                    ] = baseline.get(
                        "last_cache_execution_request_id_sha256"
                    )
                    ready.set()
                    break
            else:
                idle_streak = 0
            stop.wait(poll_interval_s)
        if not state["baseline_settled"]:
            failures.append(
                "before: direct backend did not settle to exclusive idle"
            )
            ready.set()
            return
        if not action_started.wait(timeout_s):
            failures.append("gateway action did not start before timeout")
            return

        while not action_finished.is_set() and not stop.is_set():
            try:
                row = observe("during")
            except Exception as exc:
                failures.append(
                    f"during: health probe raised {type(exc).__name__}"
                )
                return
            if row["identity_failures"] or row["lifecycle_failures"]:
                return
            if not bounded_single_owner(row, "during"):
                return
            stop.wait(poll_interval_s)

        after_deadline = time.monotonic() + timeout_s
        idle_streak = 0
        while time.monotonic() < after_deadline and not stop.is_set():
            try:
                row = observe("after")
            except Exception as exc:
                failures.append(
                    f"after: health probe raised {type(exc).__name__}"
                )
                return
            if row["identity_failures"] or row["lifecycle_failures"]:
                return
            if not bounded_single_owner(row, "after"):
                return
            if is_idle(row):
                idle_streak += 1
                if idle_streak >= 2:
                    state["final_idle_settled"] = True
                    return
            else:
                idle_streak = 0
            stop.wait(poll_interval_s)
        failures.append(
            "after: exclusive idle state did not settle before timeout"
        )

    worker = threading.Thread(target=poll, name="paired-replay-health", daemon=True)
    worker.start()
    if not ready.wait(timeout_s + max(0.05, poll_interval_s * 2)):
        failures.append("before: health probe did not become ready before timeout")
        stop.set()
    action_error: BaseException | None = None
    result: dict[str, Any] = {
        "status_code": 0,
        "response_id": "",
        "reasoning": "",
        "content": "",
        "notices": [],
        "tool_calls": [],
        "terminals": [],
        "errors": [],
        "events": [],
        "_prepared_request_body_sha256": "",
    }
    if state["baseline_settled"]:
        action_started.set()
        state["action_executed"] = True
        try:
            result = action()
        except BaseException as exc:
            action_error = exc
        finally:
            action_finished.set()
    else:
        failures.append(
            "gateway action skipped because exclusive idle was unattested"
        )
        stop.set()
        action_finished.set()
    worker.join(timeout_s)
    if worker.is_alive():
        failures.append("health lifecycle poller did not terminate")
        stop.set()
        worker.join(min(timeout_s, 1.0))

    response_id = str(result.get("response_id") or "")
    correlation_available = protocol == "chat"
    expected_request_ids = (
        _response_request_id_hashes(response_id)
        if correlation_available
        else set()
    )
    foreign_request_ids = (
        sorted(observed_action_request_ids - expected_request_ids)
        if correlation_available
        else []
    )
    correlation_failures: list[str] = []
    if not correlation_available:
        correlation_status = "unavailable_ollama_gateway_translation"
        correlation_pass: bool | None = None
    elif not response_id:
        correlation_status = "missing_gateway_response_id"
        correlation_pass = False
        correlation_failures.append("gateway action did not return a response ID")
    elif not observed_action_request_ids:
        correlation_status = "unobserved_request_id"
        correlation_pass = False
        correlation_failures.append(
            "gateway action produced no active or durable request ID evidence"
        )
    elif foreign_request_ids:
        correlation_status = "foreign_request_ids"
        correlation_pass = False
        correlation_failures.append(
            "gateway lifecycle observed foreign request IDs"
        )
    else:
        correlation_status = "matched"
        correlation_pass = True
    action_evidence_seen = (
        state["active_seen"] or state["durable_completion_seen"]
    )
    if not action_evidence_seen:
        failures.append(
            "during: exclusive gateway activity or durable completion "
            "was not observed"
        )

    bounded_complete = (
        not failures
        and state["baseline_settled"]
        and action_evidence_seen
        and state["final_idle_settled"]
        and state["action_executed"]
        and bool(observed_action_request_ids)
        and not worker.is_alive()
    )
    request_owned_complete = (
        bounded_complete
        and correlation_pass is True
        and not correlation_failures
    )
    if action_error is not None:
        raise action_error
    return result, {
        "samples": samples,
        "failures": list(dict.fromkeys([*failures, *correlation_failures])),
        # The bounded observer remains useful even when a protocol cannot
        # expose enough response identity to prove ownership.
        "bounded_exclusive_idle_active_idle": bounded_complete,
        "bounded_exclusive_idle_action_idle": bounded_complete,
        # Retain the original key as the strict request-owned verdict consumed
        # by the paired replay release gate.
        "exclusive_idle_active_idle": request_owned_complete,
        "request_owned_exclusive_idle_active_idle": request_owned_complete,
        "baseline_idle_settled": state["baseline_settled"],
        "gateway_activity_observed": state["active_seen"],
        "durable_completion_observed": state["durable_completion_seen"],
        "action_evidence_observed": action_evidence_seen,
        "baseline_last_cache_execution_request_id_sha256": state[
            "baseline_last_cache_execution_request_id_sha256"
        ],
        "final_idle_settled": state["final_idle_settled"],
        "gateway_action_executed": state["action_executed"],
        "gateway_response_id_sha256": (
            _sha256(response_id) if response_id else None
        ),
        "request_id_correlation_available": correlation_available,
        "request_id_correlation_status": correlation_status,
        "request_id_correlation_pass": correlation_pass,
        "expected_request_ids_sha256": sorted(expected_request_ids),
        "observed_action_request_ids_sha256": sorted(
            observed_action_request_ids
        ),
        "foreign_request_ids_sha256": foreign_request_ids,
        "worker_stopped": not worker.is_alive(),
    }


def run_paired_replay_discriminator(
    *,
    direct_client: ProtocolClient,
    gateway_client: ProtocolClient,
    protocol: str,
    mode: str,
    stage: int,
    payload: dict[str, Any],
    expected_backend_identity_fingerprint: str,
    gateway_direct_health_probe: Callable[[], dict[str, Any]] | None,
    health_timeout_s: float = 5.0,
    # Cached tool-call replays can finish inside the old 25 ms sampling gap.
    # Poll frequently, while also accepting the scheduler's durable admitted-
    # request ID when it changes from the settled baseline and matches the
    # gateway response. A successful response alone never satisfies the gate.
    health_poll_interval_s: float = 0.005,
) -> dict[str, Any]:
    """Run one frozen-body direct A1 / gateway B / direct A2 discriminator."""
    if (protocol, mode, int(stage)) not in PAIRED_REPLAY_TARGETS:
        raise ValueError(
            "paired replay is limited to Chat nonstream rounds 2/3 and "
            "Ollama stream round 3"
        )
    if gateway_direct_health_probe is None:
        raise ValueError("paired replay requires an in-flight direct-health probe")
    if not _valid_sha256(expected_backend_identity_fingerprint):
        raise ValueError("paired replay expected backend fingerprint is invalid")
    if not isinstance(payload.get("stream"), bool) or payload["stream"] != (
        mode == "stream"
    ):
        raise ValueError("paired replay request stream mode does not match target")
    body = frozen_request_body(payload)
    request_public = _paired_public_request(stage, payload, protocol)
    expected_sha = hashlib.sha256(body).hexdigest()
    stem = f"paired-{protocol}-{mode}-round{stage}"

    def send(client: ProtocolClient, leg: str) -> dict[str, Any]:
        return client.send(
            protocol,
            payload,
            mode == "stream",
            capture_label=f"{stem}-{leg}",
            prepared_body=body,
        )

    a1 = send(direct_client, "a1")
    b, lifecycle = _paired_gateway_lifecycle(
        lambda: send(gateway_client, "b"),
        gateway_direct_health_probe,
        expected_backend_identity_fingerprint,
        health_timeout_s,
        health_poll_interval_s,
        protocol=protocol,
    )
    a2 = send(direct_client, "a2")
    body_hashes = {
        leg: str(result.get("_prepared_request_body_sha256") or "")
        for leg, result in {"a1": a1, "b": b, "a2": a2}.items()
    }
    normalized = {
        leg: _normalized_replay_response(result)
        for leg, result in {"a1": a1, "b": b, "a2": a2}.items()
    }
    direct_stable = normalized["a1"] == normalized["a2"]
    checks = {
        "exact_body_sha_equal": set(body_hashes.values()) == {expected_sha},
        "gateway_backend_lifecycle_pass": (
            lifecycle["exclusive_idle_active_idle"] is True
        ),
    }
    if not all(checks.values()):
        classification = "unverified"
    elif not direct_stable:
        classification = "shared_backend_model_or_cache_nondeterminism"
    elif normalized["a1"] == normalized["b"]:
        classification = "all_equal_prior_history_variance"
    else:
        classification = "gateway_owned_difference"
    return {
        "schema": "vmlx-agentic-protocol-paired-replay-v1",
        "target": {"protocol": protocol, "mode": mode, "stage": int(stage)},
        "request": {
            **request_public,
            "prepared_body_bytes": len(body),
            "prepared_body_sha256": expected_sha,
            "leg_body_sha256": body_hashes,
        },
        "responses": {
            leg: _paired_public_response(value) for leg, value in normalized.items()
        },
        "direct_replay_stable": direct_stable,
        "classification": classification,
        "gateway_in_flight_direct_health": lifecycle,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _request_common(
    model: str, max_tokens: int, enable_thinking: bool
) -> dict[str, Any]:
    return {
        "model": model,
        "temperature": 0,
        "enable_thinking": enable_thinking,
        "_max_tokens": max_tokens,
    }


def build_request(
    protocol: str,
    model: str,
    mode: str,
    stage: int,
    *,
    history: list[dict[str, Any]] | str,
    instructions: str,
    previous_response_id: str = "",
    max_tokens: int = 512,
    enable_thinking: bool = True,
    second_tool_choice: str = "auto",
    first_tool_choice: str = "current",
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    stream = mode == "stream"
    common = _request_common(model, max_tokens, enable_thinking)
    common.pop("_max_tokens")
    names = ("file_info", "run_command")
    if protocol == "chat":
        body: dict[str, Any] = {
            **common,
            "messages": history,
            "stream": stream,
            "max_tokens": max_tokens,
            "tools": tool_schemas(protocol, names),
            "tool_choice": tool_choice(
                protocol, mode, stage, second_tool_choice, first_tool_choice
            ),
        }
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body
    if protocol == "responses":
        body = {
            **common,
            "input": history,
            "instructions": instructions,
            "stream": stream,
            "store": True,
            "max_output_tokens": max_tokens,
            "tools": tool_schemas(protocol, names),
            "tool_choice": tool_choice(
                protocol, mode, stage, second_tool_choice, first_tool_choice
            ),
        }
        if reasoning_effort is not None:
            body["reasoning"] = {"effort": reasoning_effort}
        if previous_response_id:
            body["previous_response_id"] = previous_response_id
        return body
    if protocol == "anthropic":
        return {
            **common,
            "messages": history,
            "stream": stream,
            "max_tokens": max_tokens,
            "tools": tool_schemas(protocol, names),
            "tool_choice": tool_choice(
                protocol, mode, stage, second_tool_choice, first_tool_choice
            ),
        }
    if protocol == "ollama":
        body = {
            "model": model,
            "messages": history,
            "stream": stream,
            "think": enable_thinking,
            "options": {"temperature": 0, "num_predict": max_tokens},
        }
        if stage < 3:
            expected = "file_info" if stage == 1 else "run_command"
            body["tools"] = tool_schemas(protocol, (expected,))
        return body
    raise ValueError(f"unknown protocol: {protocol}")


def _sanitized_round(round_result: dict[str, Any]) -> dict[str, Any]:
    reasoning = str(round_result.get("reasoning") or "")
    content = str(round_result.get("content") or "")
    return {
        "status_code": round_result.get("status_code"),
        "elapsed_ms": round_result.get("elapsed_ms"),
        "response_id": round_result.get("response_id"),
        "reasoning_chars": len(reasoning),
        "reasoning_sha256": _sha256(reasoning),
        "reasoning_delta_count": sum(
            1
            for event in round_result.get("events") or []
            if event.get("channel") == "reasoning"
        ),
        "content": content,
        "content_chars": len(content),
        "content_sha256": _sha256(content),
        "content_delta_count": sum(
            1
            for event in round_result.get("events") or []
            if event.get("channel") == "content"
        ),
        "tool_calls": round_result.get("tool_calls") or [],
        "terminals": round_result.get("terminals") or [],
        "errors": round_result.get("errors") or [],
        "events": round_result.get("events") or [],
    }


def _execution_public(execution: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": execution["name"],
        "call_id": execution["call_id"],
        "arguments": execution["arguments"],
        "result": execution["result"],
        "output_chars": len(execution["output"]),
        "output_sha256": _sha256(execution["output"]),
    }


def _contains_control_markup(text: str) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in CONTROL_MARKERS)


def final_synthesis_instruction(base_label: str, protocol: str, mode: str) -> str:
    """Ask for result-derived fields without leaking their expected values."""
    # Direct and gateway must receive byte-identical conversations. Keep the
    # base label only in the result artifact; embedding it in the user prompt
    # changes tokenization/cache keys and invalidates a parity comparison.
    _ = base_label
    prefix = f"AGENTIC-{protocol.upper()}-{mode.upper()}-DONE"
    return (
        "Both real tool results are now present. Call no tools. Reply with exactly "
        f"one line in this format: {prefix} SIZE=<copy size_human from the file_info "
        "result> PWD=<copy stdout from the run_command result>. Replace both angle-"
        "bracket placeholders with the real result values; output no other text."
    )


def first_tool_instruction(protocol: str, mode: str) -> str:
    """Return the base-independent first turn used for parity comparison."""
    # Protocol/mode labels belong in the evidence artifact, not the model's
    # instruction. A diagnostic prefix changed Qwen's native tool-call shape
    # and made the supposed parity probe test two artificial conversations.
    _ = (protocol, mode)
    return (
        f"Call the built-in file_info tool exactly once with path {FILE_INFO_PATH}. "
        "You must use the tool and must not answer from memory. Do not call "
        "run_command yet and do not answer yet."
    )


def run_flow(
    client: ProtocolClient,
    *,
    base_label: str,
    protocol: str,
    mode: str,
    model: str,
    repo_root: Path,
    max_tokens: int,
    enable_thinking: bool,
    second_tool_choice: str = "explicit",
    first_tool_choice: str = "current",
    stable_responses_instructions: bool = False,
    native_reasoning_effort: str | None = None,
) -> dict[str, Any]:
    first_prompt = first_tool_instruction(protocol, mode)
    second_prompt = (
        "The real file_info result is now present. Call run_command exactly once "
        f"with command {PWD_COMMAND}. Do not repeat file_info and do not answer yet."
    )
    initial_history: list[dict[str, Any]] | str
    if protocol == "responses":
        initial_history = first_prompt
    else:
        initial_history = [{"role": "user", "content": first_prompt}]
    first_request_instructions = (
        STABLE_RESPONSES_INSTRUCTIONS
        if protocol == "responses" and stable_responses_instructions
        else first_prompt
    )

    request1 = build_request(
        protocol,
        model,
        mode,
        1,
        history=initial_history,
        instructions=first_request_instructions,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        second_tool_choice=second_tool_choice,
        first_tool_choice=first_tool_choice,
        reasoning_effort=native_reasoning_effort,
    )
    request_records = [_request_public(1, request1, protocol=protocol)]
    round1 = client.send(
        protocol,
        request1,
        mode == "stream",
        capture_label=f"{mode}-flow-round1",
    )
    calls1 = round1.get("tool_calls") or []
    check1 = len(calls1) == 1
    error1 = "expected exactly one tool call"
    if check1:
        check1, error1 = validate_allowlisted_call(calls1[0], "file_info")
    if not check1:
        return {
            "pass": False,
            "failure": f"round1: {error1}",
            "requests": request_records,
            "rounds": [_sanitized_round(round1)],
        }
    execution1 = execute_allowlisted_tool(repo_root, calls1[0])

    if protocol == "responses":
        history2 = history_after_tool(protocol, [], round1, execution1, second_prompt)
    else:
        history2 = history_after_tool(
            protocol, list(initial_history), round1, execution1, second_prompt
        )
    request2 = build_request(
        protocol,
        model,
        mode,
        2,
        history=history2,
        instructions=(
            STABLE_RESPONSES_INSTRUCTIONS
            if protocol == "responses" and stable_responses_instructions
            else second_prompt
        ),
        previous_response_id=str(round1.get("response_id") or ""),
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        second_tool_choice=second_tool_choice,
        first_tool_choice=first_tool_choice,
        reasoning_effort=native_reasoning_effort,
    )
    request_records.append(_request_public(2, request2, protocol=protocol))
    round2 = client.send(
        protocol,
        request2,
        mode == "stream",
        capture_label=f"{mode}-flow-round2",
    )
    calls2 = round2.get("tool_calls") or []
    check2 = len(calls2) == 1
    error2 = "expected exactly one tool call"
    if check2:
        check2, error2 = validate_allowlisted_call(calls2[0], "run_command")
    if not check2:
        return {
            "pass": False,
            "failure": f"round2: {error2}",
            "requests": request_records,
            "rounds": [_sanitized_round(round1), _sanitized_round(round2)],
            "executions": [_execution_public(execution1)],
        }
    execution2 = execute_allowlisted_tool(repo_root, calls2[0])
    final_marker = (
        f"AGENTIC-{protocol.upper()}-{mode.upper()}-DONE "
        f"SIZE={execution1['result']['size_human']} PWD={execution2['result']['stdout']}"
    )
    final_prompt = final_synthesis_instruction(base_label, protocol, mode)
    if protocol == "responses":
        history3 = history_after_tool(protocol, [], round2, execution2, final_prompt)
    else:
        history3 = history_after_tool(
            protocol, history2, round2, execution2, final_prompt
        )
    request3 = build_request(
        protocol,
        model,
        mode,
        3,
        history=history3,
        instructions=(
            STABLE_RESPONSES_INSTRUCTIONS
            if protocol == "responses" and stable_responses_instructions
            else final_prompt
        ),
        previous_response_id=str(round2.get("response_id") or ""),
        max_tokens=max_tokens,
        # Turn off optional reasoning for the final-only contract. A native
        # effort-only model continues to use its declared reasoning control.
        enable_thinking=native_reasoning_effort is not None,
        second_tool_choice=second_tool_choice,
        first_tool_choice=first_tool_choice,
        reasoning_effort=native_reasoning_effort,
    )
    request_records.append(_request_public(3, request3, protocol=protocol))
    round3 = client.send(
        protocol,
        request3,
        mode == "stream",
        capture_label=f"{mode}-flow-round3",
    )

    stream = mode == "stream"
    terminals = [
        classify_terminal(
            protocol,
            round1.get("terminals") or [],
            stream=stream,
            expect_tool=True,
            events=round1.get("events") or [],
        ),
        classify_terminal(
            protocol,
            round2.get("terminals") or [],
            stream=stream,
            expect_tool=True,
            events=round2.get("events") or [],
        ),
        classify_terminal(
            protocol,
            round3.get("terminals") or [],
            stream=stream,
            expect_tool=False,
            events=round3.get("events") or [],
        ),
    ]
    reasoning1 = str(round1.get("reasoning") or "")
    reasoning2 = str(round2.get("reasoning") or "")
    reasoning3 = str(round3.get("reasoning") or "")
    reasoning_values = [
        value for value in (reasoning1, reasoning2, reasoning3) if value
    ]
    response_ids = [
        str(row.get("response_id") or "") for row in (round1, round2, round3)
    ]
    response_tool_lifecycles = []
    if protocol == "responses" and stream:
        for row in (round1, round2):
            event_kinds = [
                str(event.get("kind") or "")
                for event in row.get("events") or []
                if event.get("channel") == "tool"
            ]
            response_tool_lifecycles.append(
                "response.output_item.added" in event_kinds
                and "response.function_call_arguments.done" in event_kinds
                and "response.output_item.done" in event_kinds
            )
    checks = {
        "status_200": all(
            int(row.get("status_code") or 0) == 200 for row in (round1, round2, round3)
        ),
        "round1_exact_tool": check1,
        "round2_exact_tool": check2,
        "final_no_tool": not (round3.get("tool_calls") or []),
        "final_exact": str(round3.get("content") or "").strip() == final_marker,
        # OpenAI Responses permits an assistant message and a function_call in
        # the same response. MiniMax-M3 can truthfully narrate "I'll call …"
        # before emitting the exact required tool item; that is ordinary model
        # verbosity, not inline reasoning or a parser leak. Exact tool
        # validation plus the control-markup checks below are the protocol
        # contract. Do not reject a valid tool lifecycle solely for prose.
        "tool_rounds_valid_with_optional_visible_prose": check1 and check2,
        "no_stream_or_protocol_errors": all(
            not (row.get("errors") or []) for row in (round1, round2, round3)
        ),
        "no_visible_control_markup": all(
            not _contains_control_markup(str(row.get("content") or ""))
            for row in (round1, round2, round3)
        ),
        "terminals_truthful": all(item["pass"] for item in terminals),
        "stream_final_progressive": (
            mode != "stream"
            or sum(
                1
                for event in round3.get("events") or []
                if event.get("channel") == "content"
            )
            > 1
        ),
        "reasoning_present": (
            not enable_thinking
            # Reasoning ON permits, but does not require, a private chain before
            # every individual tool call. Require at least one real reasoning
            # rail across the three-turn flow; the separation/duplication checks
            # below still apply independently to every turn that emits it.
            or bool(reasoning_values)
        ),
        "reasoning_not_stale_when_present": len(reasoning_values)
        == len(set(reasoning_values)),
        "reasoning_not_duplicated_as_content": all(
            not (
                str(row.get("reasoning") or "").strip()
                and str(row.get("reasoning") or "").strip()
                == str(row.get("content") or "").strip()
            )
            for row in (round1, round2, round3)
        ),
        "responses_chain_ids_forwarded": (
            protocol != "responses"
            or (
                all(response_ids)
                and len(set(response_ids)) == 3
                and request2.get("previous_response_id") == response_ids[0]
                and request3.get("previous_response_id") == response_ids[1]
            )
        ),
        "responses_stream_tool_lifecycle_complete": (
            protocol != "responses" or not stream or all(response_tool_lifecycles)
        ),
        "timestamps_monotonic": all(
            all(
                float(events[index].get("at_ms") or 0)
                <= float(events[index + 1].get("at_ms") or 0)
                for index in range(len(events) - 1)
            )
            for events in [row.get("events") or [] for row in (round1, round2, round3)]
        ),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "expected_final": final_marker,
        "requests": request_records,
        "rounds": [
            _sanitized_round(round1),
            _sanitized_round(round2),
            _sanitized_round(round3),
        ],
        "executions": [_execution_public(execution1), _execution_public(execution2)],
        "terminal_classification": terminals,
        "_paired_replay_payloads": (
            {2: copy.deepcopy(request2), 3: copy.deepcopy(request3)}
            if protocol == "chat" and mode == "nonstream"
            else {}
        ),
        "_prefinal_payload": request3,
    }


def _replace_final_instruction(
    protocol: str, payload: dict[str, Any], instruction: str
) -> dict[str, Any]:
    body = copy.deepcopy(payload)
    if protocol == "responses":
        body["instructions"] = instruction
    else:
        messages = body.get("messages") or []
        if not messages:
            raise ValueError("final payload has no messages")
        last = messages[-1]
        if protocol == "anthropic" and isinstance(last.get("content"), list):
            replaced = False
            for part in reversed(last["content"]):
                if part.get("type") == "text":
                    part["text"] = instruction
                    replaced = True
                    break
            if not replaced:
                last["content"].append({"type": "text", "text": instruction})
        else:
            last["content"] = instruction
    if protocol == "ollama":
        body["options"]["num_predict"] = 1024
    elif protocol == "responses":
        body["max_output_tokens"] = 1024
    else:
        body["max_tokens"] = 1024
    return body


def _idle_values(health: dict[str, Any]) -> tuple[int | None, int | None]:
    scheduler = health.get("scheduler") or {}
    cache = health.get("cache") or {}
    scheduler_cache = cache.get("scheduler_cache") or {}
    running = scheduler.get("num_running")
    active = scheduler_cache.get("active_requests")
    return (
        int(running) if isinstance(running, (int, float)) else None,
        int(active) if isinstance(active, (int, float)) else None,
    )


def wait_for_idle(health_url: str, timeout: float = 20.0) -> dict[str, Any]:
    started = time.monotonic()
    samples: list[dict[str, Any]] = []
    while time.monotonic() - started < timeout:
        try:
            response = requests.get(health_url, timeout=5)
            health = response.json()
            running, active = _idle_values(health)
            sample = {
                "at_ms": _milliseconds(started),
                "status_code": response.status_code,
                "status": health.get("status"),
                "num_running": running,
                "active_requests": active,
            }
            lifecycle = health.get("request_lifecycle")
            if isinstance(lifecycle, dict):
                sample["request_lifecycle"] = lifecycle
                counts = (
                    lifecycle.get("active_request_count"),
                    lifecycle.get("engine_collector_count"),
                    lifecycle.get("scheduler_waiting_count"),
                    lifecycle.get("scheduler_running_count"),
                )
                idle = (
                    lifecycle.get("schema") == "vmlx-request-lifecycle-v1"
                    and lifecycle.get("available") is True
                    and all(type(count) is int and count == 0 for count in counts)
                    and lifecycle.get("terminal_cleanup_pending") is False
                )
            else:
                idle = lifecycle is None and running == 0 and active in (0, None)
            samples.append(sample)
            if response.status_code == 200 and idle:
                return {"idle": True, "elapsed_ms": sample["at_ms"], "samples": samples}
        except Exception as exc:
            samples.append(
                {
                    "at_ms": _milliseconds(started),
                    "error": type(exc).__name__,
                }
            )
        time.sleep(0.1)
    return {"idle": False, "elapsed_ms": round(timeout * 1000, 3), "samples": samples}


def _cancel_route(protocol: str, response_id: str) -> str | None:
    if protocol == "chat":
        return f"/v1/chat/completions/{response_id}/cancel"
    if protocol == "responses":
        return f"/v1/responses/{response_id}/cancel"
    return None


def abort_stream_after_deltas(
    client: ProtocolClient,
    protocol: str,
    payload: dict[str, Any],
    *,
    health_url: str,
    minimum_deltas: int,
) -> dict[str, Any]:
    body = _replace_final_instruction(
        protocol,
        payload,
        "Using the completed real tool results, output 500 numbered lines beginning "
        "with POST-TOOL-ABORT. Begin immediately and do not call tools.",
    )
    body["stream"] = True
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    response = requests.post(
        client.base_url + client.route(protocol),
        headers=client.headers,
        json=body,
        stream=True,
        timeout=(15, client.timeout),
    )
    collector = EventCollector(protocol=protocol, started=started)
    event_name: str | None = None
    cancel_status: int | None = None
    cancel_body_hash = ""
    capture: DecompressedParserInputCaptureSession | None = None
    capture_error_type: str | None = None
    closed_at_ms = 0.0
    try:
        capture = client._attach_parser_input_capture(
            protocol=protocol,
            capture_label="stream-abort",
            payload=body,
            response=response,
            started=started,
            started_at=started_at,
        )
        for raw in response.iter_lines(decode_unicode=True, chunk_size=1):
            if raw is None:
                continue
            line = (
                raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
            )
            line = line.strip()
            if not line:
                event_name = None
                continue
            at_ms = _milliseconds(started)
            if protocol != "ollama" and line.startswith("event: "):
                event_name = line[7:]
                continue
            if protocol != "ollama":
                if not line.startswith("data: "):
                    continue
                raw_data = line[6:]
                if raw_data == "[DONE]":
                    collector.terminal("DONE", at_ms)
                    continue
            else:
                raw_data = line
            data = json.loads(raw_data)
            _parse_stream_object(protocol, data, event_name, collector, at_ms)
            delta_count = len(collector.reasoning_parts) + len(collector.content_parts)
            if delta_count < minimum_deltas:
                continue
            route = _cancel_route(protocol, collector.response_id)
            if route:
                cancelled = requests.post(
                    client.base_url + route,
                    headers=client.headers,
                    json={},
                    timeout=10,
                )
                cancel_status = cancelled.status_code
                cancel_body_hash = _sha256(cancelled.text)
            break
    except BaseException as exc:
        capture_error_type = type(exc).__name__
        raise
    finally:
        try:
            response.close()
        except BaseException as exc:
            if capture_error_type is None:
                capture_error_type = type(exc).__name__
            raise
        finally:
            closed_at_ms = _milliseconds(started)
            if capture is not None:
                capture.finish(
                    completed_ms=closed_at_ms,
                    error_type=capture_error_type,
                )
    return {
        "status_code": response.status_code,
        "closed_at_ms": closed_at_ms,
        "response_id": collector.response_id,
        "delta_events_before_abort": len(collector.reasoning_parts)
        + len(collector.content_parts),
        "cancel_status": cancel_status,
        "cancel_body_sha256": cancel_body_hash,
        "terminals_before_abort": collector.terminals,
        "events": collector.events,
        "idle_after_abort": wait_for_idle(health_url),
    }


def disconnect_nonstream(
    client: ProtocolClient,
    protocol: str,
    payload: dict[str, Any],
    *,
    health_url: str,
    delay_ms: int,
) -> dict[str, Any]:
    body = _replace_final_instruction(
        protocol,
        payload,
        "Using the completed real tool results, output 500 numbered lines beginning "
        "with POST-TOOL-DISCONNECT. Begin immediately and do not call tools.",
    )
    body["stream"] = False
    parsed = urlparse(client.base_url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError(
            "non-stream disconnect hook currently requires an http base URL"
        )
    connection = http.client.HTTPConnection(
        parsed.hostname,
        parsed.port or 80,
        timeout=client.timeout,
    )
    route = (parsed.path.rstrip("/") if parsed.path else "") + client.route(protocol)
    headers = dict(client.headers)
    started = time.monotonic()
    connection.request(
        "POST", route, body=json.dumps(body).encode("utf-8"), headers=headers
    )
    time.sleep(max(delay_ms, 0) / 1000.0)
    connection.close()
    return {
        "disconnected_at_ms": _milliseconds(started),
        "delay_ms": delay_ms,
        "idle_after_disconnect": wait_for_idle(health_url),
    }


def classify_abort(
    protocol: str, mode: str, aborted: dict[str, Any], minimum_deltas: int
) -> dict[str, Any]:
    """Require a real cancel route when available and never accept a false terminal."""
    idle = bool(
        (
            aborted.get("idle_after_abort")
            or aborted.get("idle_after_disconnect")
            or {}
        ).get("idle")
    )
    if mode == "nonstream":
        return {"pass": idle, "idle": idle, "kind": "client_disconnect"}
    deltas = int(aborted.get("delta_events_before_abort") or 0)
    terminals = list(aborted.get("terminals_before_abort") or [])
    cancel_route_ok = (
        aborted.get("cancel_status") in {200, 202}
        if protocol in {"chat", "responses"}
        else True
    )
    passed = idle and deltas >= minimum_deltas and not terminals and cancel_route_ok
    return {
        "pass": passed,
        "idle": idle,
        "delta_events": deltas,
        "no_terminal_before_abort": not terminals,
        "cancel_route_ok": cancel_route_ok,
        "kind": "explicit_cancel"
        if protocol in {"chat", "responses"}
        else "client_disconnect",
    }


def build_recovery_request(
    protocol: str,
    model: str,
    mode: str,
    marker: str,
    max_tokens: int,
    *,
    native_reasoning_effort: str | None = None,
) -> dict[str, Any]:
    prompt = f"Call no tools. Reply exactly {marker} and nothing else."
    history: list[dict[str, Any]] | str = (
        prompt if protocol == "responses" else [{"role": "user", "content": prompt}]
    )
    return build_request(
        protocol,
        model,
        mode,
        3,
        history=history,
        instructions=prompt,
        max_tokens=max_tokens,
        enable_thinking=native_reasoning_effort is not None,
        reasoning_effort=native_reasoning_effort,
    )


def run_recovery(
    client: ProtocolClient,
    protocol: str,
    mode: str,
    model: str,
    marker: str,
    max_tokens: int,
    *,
    native_reasoning_effort: str | None = None,
) -> dict[str, Any]:
    payload = build_recovery_request(
        protocol, model, mode, marker, max_tokens,
        native_reasoning_effort=native_reasoning_effort,
    )
    result = client.send(
        protocol,
        payload,
        mode == "stream",
        capture_label=f"{mode}-recovery",
    )
    classification = classify_terminal(
        protocol,
        result.get("terminals") or [],
        stream=mode == "stream",
        expect_tool=False,
        events=result.get("events") or [],
    )
    public = _sanitized_round(result)
    public["expected"] = marker
    public["exact"] = str(result.get("content") or "").strip() == marker
    public["terminal_classification"] = classification
    public["pass"] = (
        public["exact"] and classification["pass"] and not result.get("tool_calls")
    )
    return public


def parse_named_urls(values: list[str], option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} requires NAME=URL, got {value!r}")
        name, url = value.split("=", 1)
        name = name.strip()
        url = url.strip().rstrip("/")
        if not name or not urlparse(url).scheme or not urlparse(url).netloc:
            raise ValueError(f"invalid {option} value: {value!r}")
        if name in parsed:
            raise ValueError(f"duplicate {option} name: {name}")
        parsed[name] = url
    return parsed


def backend_lifecycle_health_url(
    base_label: str,
    bases: dict[str, str],
    health_urls: dict[str, str],
) -> str:
    """Return the engine health endpoint that owns request cleanup.

    The gateway's own /health is required for topology/identity evidence, but
    abort recovery must observe the bound backend scheduler rather than the
    gateway process. Direct traffic uses its corresponding engine health URL.
    """
    lifecycle_label = "direct" if base_label == "gateway" else base_label
    base_url = bases.get(lifecycle_label)
    if not base_url:
        raise ValueError(
            f"missing {lifecycle_label} base URL for lifecycle observation"
        )
    return health_urls.get(lifecycle_label, base_url + "/health")


def _validate_base_origins(bases: dict[str, str]) -> list[str]:
    failures: list[str] = []
    origins: dict[str, tuple[str, str, int | None, str]] = {}
    for label, value in bases.items():
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"}:
            failures.append(f"{label}: endpoint scheme must be http or https")
        if parsed.username is not None or parsed.password is not None:
            failures.append(f"{label}: endpoint URL must not contain credentials")
        if parsed.query or parsed.fragment:
            failures.append(f"{label}: endpoint URL must not contain query or fragment")
        origins[label] = (
            parsed.scheme.lower(),
            (parsed.hostname or "").lower(),
            parsed.port,
            parsed.path.rstrip("/"),
        )
    if (
        "direct" in origins
        and "gateway" in origins
        and origins["direct"] == origins["gateway"]
    ):
        failures.append(
            "direct and gateway must be distinct observed network origins"
        )
    return failures


def _network_origin(value: str) -> tuple[str, str, int | None]:
    parsed = urlparse(value)
    scheme = parsed.scheme.lower()
    port = parsed.port
    if port is None:
        port = 80 if scheme == "http" else 443 if scheme == "https" else None
    return (scheme, (parsed.hostname or "").lower(), port)


def expected_parser_input_capture_routes(
    bases: Iterable[str],
    protocols: Iterable[str],
    modes: Iterable[str],
    *,
    skip_cancellation: bool,
) -> list[tuple[str, str, str]]:
    mode_set = set(modes)
    protocol_list = list(protocols)
    routes: list[tuple[str, str, str]] = []
    for base_label in bases:
        if "stream" in mode_set:
            labels = [
                "stream-flow-round1",
                "stream-flow-round2",
                "stream-flow-round3",
            ]
            if not skip_cancellation:
                labels.extend(("stream-abort", "stream-recovery"))
            routes.extend(
                (base_label, protocol, capture_label)
                for protocol in protocol_list
                for capture_label in labels
            )
        if "nonstream" in mode_set:
            labels = [
                f"nonstream-flow-round{round_number}"
                for round_number in (1, 2, 3)
            ]
            if not skip_cancellation:
                # Non-stream cancellation is a client disconnect followed by
                # the same recovery request that run_matrix() records below.
                labels.append("nonstream-recovery")
            routes.extend(
                (base_label, protocol, capture_label)
                for protocol in protocol_list
                for capture_label in labels
            )
    return routes


def _disabled_capture_summary() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "enabled": False,
        "capture_layer": CAPTURE_LAYER,
        "capture_semantics": CAPTURE_SEMANTICS,
        "reason": "--raw-artifact-dir not supplied",
        "expected": 0,
        "started": 0,
        "finished": 0,
        "errors": 0,
        "complete": True,
        "routes": [],
    }


def _source_identity_failures(
    source: dict[str, Any],
    declared_head: str,
) -> list[str]:
    failures: list[str] = []
    if not declared_head:
        failures.append("--source-head is required")
    elif source.get("head") != declared_head:
        failures.append(
            "--source-head does not match the observed current Git HEAD"
        )
    if source.get("clean") is not True:
        failures.append("observed source checkout is dirty")
    if int(source.get("python_source_read_error_count") or 0):
        failures.append("observed source tree contains unreadable Python files")
    return failures


def _runner_environment_failures(runner: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    execution_mode = runner.get("execution_mode") or "source-checkout-venv"
    if execution_mode not in {"source-checkout-venv", "installed-runtime"}:
        failures.append("proof runner execution mode is invalid")
    if not _valid_sha256(runner.get("python_executable_fingerprint_sha256")):
        failures.append("proof runner Python executable fingerprint is invalid")
    if execution_mode == "source-checkout-venv":
        if runner.get("repo_venv") is not True:
            failures.append("proof runner sys.prefix is not the source checkout .venv")
        if runner.get("repo_python") is not True:
            failures.append("proof runner executable is not the source checkout Python")
        checkout_python_fingerprints = runner.get(
            "checkout_python_invocation_fingerprints_sha256"
        )
        accepted_python_fingerprints = runner.get(
            "accepted_python_invocation_fingerprints_sha256"
        )
        if (
            not isinstance(checkout_python_fingerprints, list)
            or not checkout_python_fingerprints
            or any(
                not _valid_sha256(fingerprint)
                for fingerprint in checkout_python_fingerprints
            )
            or runner.get("python_executable_fingerprint_sha256")
            not in checkout_python_fingerprints
        ):
            failures.append(
                "proof runner checkout Python invocation fingerprints are invalid"
            )
        if (
            accepted_python_fingerprints is not None
            and accepted_python_fingerprints != checkout_python_fingerprints
        ):
            failures.append(
                "proof runner accepted Python fingerprints do not match the "
                "source checkout"
            )
    elif execution_mode == "installed-runtime":
        if (
            runner.get("repo_venv") is not False
            or runner.get("repo_python") is not False
        ):
            failures.append(
                "installed proof runner unexpectedly aliases source-checkout Python"
            )
        installed_python_fingerprints = runner.get(
            "installed_python_invocation_fingerprints_sha256"
        )
        accepted_python_fingerprints = runner.get(
            "accepted_python_invocation_fingerprints_sha256"
        )
        installed_runtime = runner.get("installed_runtime")
        expected_installed_python_fingerprints = (
            sorted(
                {
                    installed_runtime.get(
                        "invoked_python_fingerprint_sha256"
                    ),
                    installed_runtime.get(
                        "packaged_python_invocation_fingerprint_sha256"
                    ),
                }
            )
            if isinstance(installed_runtime, dict)
            else []
        )
        if (
            not isinstance(installed_python_fingerprints, list)
            or len(installed_python_fingerprints) != 2
            or any(
                not _valid_sha256(fingerprint)
                for fingerprint in installed_python_fingerprints
            )
            or installed_python_fingerprints
            != expected_installed_python_fingerprints
            or installed_python_fingerprints
            != accepted_python_fingerprints
            or runner.get("python_executable_fingerprint_sha256")
            not in installed_python_fingerprints
        ):
            failures.append(
                "installed proof runner Python invocation binding is invalid"
            )
        if not isinstance(installed_runtime, dict):
            failures.append("installed proof runner release attestation is missing")
        else:
            manifest = installed_runtime.get("manifest")
            artifacts = installed_runtime.get("artifacts")
            bundled_python = installed_runtime.get("bundled_python")
            bundled_source = installed_runtime.get("bundled_source")
            source_binding = installed_runtime.get("source_binding")
            provenance = installed_runtime.get("bundled_provenance")
            manifest_value = manifest if isinstance(manifest, dict) else {}
            if (
                installed_runtime.get("schema")
                != "vmlx-agentic-installed-runtime-v1"
                or installed_runtime.get("manifest_opened_nofollow") is not True
                or not Path(
                    str(installed_runtime.get("manifest_path") or "")
                ).is_absolute()
                or not _valid_sha256(installed_runtime.get("manifest_sha256"))
                or int(installed_runtime.get("manifest_size_bytes") or 0) <= 0
                or int(installed_runtime.get("manifest_size_bytes") or 0)
                > 1024 * 1024
                or installed_runtime.get("manifest_nlink") != 1
                or not isinstance(manifest, dict)
                or set(manifest_value) != INSTALLED_RELEASE_MANIFEST_FIELDS
                or manifest_value.get("schema")
                != INSTALLED_RELEASE_MANIFEST_SCHEMA
            ):
                failures.append("installed release manifest attestation is invalid")
            app_path = Path(str(installed_runtime.get("app_path") or ""))
            invoked_python_path = Path(
                str(installed_runtime.get("invoked_python_path") or "")
            )
            packaged_python_path = (
                app_path / INSTALLED_BUNDLED_PYTHON_RELATIVE_PATH
            )
            try:
                canonical_invoked_python_path = invoked_python_path.resolve(
                    strict=True
                )
                packaged_python_matches = (
                    packaged_python_path.resolve(strict=True)
                    == canonical_invoked_python_path
                )
            except (FileNotFoundError, RuntimeError, OSError):
                canonical_invoked_python_path = None
                packaged_python_matches = False
            if (
                not app_path.is_absolute()
                or app_path.name != "vMLX.app"
                or not packaged_python_matches
            ):
                failures.append("installed app and bundled Python paths are invalid")
            if (
                not isinstance(artifacts, dict)
                or set(artifacts) != set(INSTALLED_APP_ARTIFACTS)
                or any(
                    not isinstance(artifacts.get(label), dict)
                    or artifacts[label].get("opened_nofollow") is not True
                    or artifacts[label].get("requested_path")
                    != str(app_path / relative_path)
                    or artifacts[label].get("sha256")
                    != manifest_value.get(manifest_field)
                    for label, (
                        relative_path,
                        manifest_field,
                    ) in INSTALLED_APP_ARTIFACTS.items()
                )
            ):
                failures.append("installed app artifact manifest binding is invalid")
            if (
                not isinstance(bundled_python, dict)
                or bundled_python.get("sha256")
                != runner.get("producer_executable_sha256")
                or bundled_python.get("sha256")
                != manifest_value.get("bundled_python_executable_sha256")
                or bundled_python.get("size_bytes")
                != runner.get("producer_executable_size_bytes")
                or bundled_python.get("path")
                != str(canonical_invoked_python_path)
                or installed_runtime.get("invoked_python_path")
                != runner.get("python_executable_path")
                or installed_runtime.get("invoked_python_fingerprint_sha256")
                != runner.get("python_executable_fingerprint_sha256")
                or installed_runtime.get("python_prefix_path")
                != runner.get("python_prefix_path")
                or manifest_value.get(
                    "bundled_python_executable_fingerprint_sha256"
                )
                != runner.get("python_executable_fingerprint_sha256")
            ):
                failures.append("installed bundled Python identity is invalid")
            expected_source = {
                "head": manifest_value.get("source_commit"),
                "tree": manifest_value.get("source_tree"),
            }
            if isinstance(source_binding, dict):
                expected_source.update(
                    {
                        field: source_binding.get(field)
                        for field in (
                            *RUNTIME_SOURCE_HASH_FIELDS,
                            "python_source_file_count",
                            "python_source_read_error_count",
                        )
                    }
                )
            if (
                not isinstance(source_binding, dict)
                or source_binding != expected_source
                or not isinstance(bundled_source, dict)
                or any(
                    bundled_source.get(field) != source_binding.get(field)
                    for field in (
                        *RUNTIME_SOURCE_HASH_FIELDS,
                        "python_source_file_count",
                        "python_source_read_error_count",
                    )
                )
                or not isinstance(provenance, dict)
                or provenance.get("schema_version") != 1
                or not isinstance(provenance.get("vmlx"), dict)
                or provenance["vmlx"].get("commit")
                != source_binding.get("head")
            ):
                failures.append("installed runtime source provenance is invalid")
    if not _valid_sha256(runner.get("python_prefix_fingerprint_sha256")):
        failures.append("proof runner Python prefix fingerprint is invalid")
    executable_path = Path(str(runner.get("python_executable_path") or ""))
    prefix_path = Path(str(runner.get("python_prefix_path") or ""))
    if (
        not executable_path.is_absolute()
        or _sha256(str(executable_path))
        != runner.get("python_executable_fingerprint_sha256")
    ):
        failures.append("proof runner Python executable path binding is invalid")
    if (
        not prefix_path.is_absolute()
        or _sha256(str(prefix_path))
        != runner.get("python_prefix_fingerprint_sha256")
    ):
        failures.append("proof runner Python prefix path binding is invalid")
    try:
        producer_pid = int(runner.get("producer_pid") or 0)
    except (TypeError, ValueError):
        producer_pid = 0
    if producer_pid <= 0:
        failures.append("proof producer PID is invalid")
    if not Path(str(runner.get("producer_executable_path") or "")).is_absolute():
        failures.append("proof producer executable path is invalid")
    if not _valid_sha256(runner.get("producer_executable_sha256")):
        failures.append("proof producer executable bytes fingerprint is invalid")
    if int(runner.get("producer_executable_size_bytes") or 0) <= 0:
        failures.append("proof producer executable byte count is invalid")
    try:
        if (
            executable_path.resolve(strict=True)
            != Path(str(runner.get("producer_executable_path") or "")).resolve(
                strict=True
            )
        ):
            failures.append(
                "proof producer executable is not the bound runner Python"
            )
    except OSError:
        failures.append("proof producer executable path cannot be resolved")
    if (
        runner.get("producer_harness_relative_path")
        != "tests/cross_matrix/run_agentic_protocol_matrix.py"
    ):
        failures.append("proof producer harness relative path is invalid")
    if not Path(str(runner.get("producer_harness_path") or "")).is_absolute():
        failures.append("proof producer harness path is invalid")
    if not _valid_sha256(runner.get("producer_harness_sha256")):
        failures.append("proof producer harness bytes fingerprint is invalid")
    if int(runner.get("producer_harness_size_bytes") or 0) <= 0:
        failures.append("proof producer harness byte count is invalid")
    return failures


def _capture_health_evidence(
    bases: dict[str, str],
    health_urls: dict[str, str],
    timeout: int,
    requested_model: str,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    evidence: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for base_label, base_url in bases.items():
        health_url = health_urls.get(base_label, base_url + "/health")
        if _network_origin(health_url) != _network_origin(base_url):
            evidence[base_label] = {
                "url": _safe_request_url(health_url),
                "error_type": "HealthOriginMismatch",
            }
            failures.append(
                f"{base_label}: /health URL origin does not match its "
                "corresponding request base"
            )
            continue
        try:
            transport_full = _get_full_health(health_url, timeout)
        except ValueError as exc:
            evidence[base_label] = {
                "url": _safe_request_url(health_url),
                "error_type": type(exc).__name__,
            }
            failures.append(f"{base_label}: full /health capture failed")
            continue
        identity_url = health_url
        full = transport_full
        if (
            base_label == "gateway"
            and not isinstance(transport_full.get("runtime_provenance"), dict)
        ):
            direct_base = bases.get("direct")
            if not direct_base:
                evidence[base_label] = {
                    "url": _safe_request_url(health_url),
                    "full": transport_full,
                    "full_canonical_json": _canonical_json(transport_full),
                    "full_sha256": _canonical_sha256(transport_full),
                    "error_type": "GatewayBackendIdentityMissing",
                }
                failures.append(
                    "gateway: topology /health cannot bind a missing direct backend"
                )
                continue
            direct_origin = _network_origin(direct_base)
            gateway_origin = _network_origin(bases[base_label])
            if direct_origin[:2] != gateway_origin[:2]:
                evidence[base_label] = {
                    "url": _safe_request_url(health_url),
                    "full": transport_full,
                    "full_canonical_json": _canonical_json(transport_full),
                    "full_sha256": _canonical_sha256(transport_full),
                    "error_type": "GatewayBackendOriginMismatch",
                }
                failures.append(
                    "gateway: topology and direct backend do not share "
                    "scheme and hostname"
                )
                continue
            direct_port = direct_origin[2]
            matching_backends = []
            for backend in transport_full.get("backends") or []:
                if not isinstance(backend, dict):
                    continue
                try:
                    backend_port = int(backend.get("port") or 0)
                except (TypeError, ValueError):
                    continue
                if (
                    backend_port == direct_port
                    and backend.get("model") == requested_model
                    and backend.get("status") == "running"
                ):
                    matching_backends.append(backend)
            if len(matching_backends) != 1:
                evidence[base_label] = {
                    "url": _safe_request_url(health_url),
                    "full": transport_full,
                    "full_canonical_json": _canonical_json(transport_full),
                    "full_sha256": _canonical_sha256(transport_full),
                    "error_type": "GatewayBackendIdentityMissing",
                }
                failures.append(
                    "gateway: topology /health does not name exactly one running "
                    "requested model on the direct backend port"
                )
                continue
            identity_url = health_urls.get("direct", direct_base + "/health")
            try:
                full = _get_full_health(identity_url, timeout)
            except ValueError:
                evidence[base_label] = {
                    "url": _safe_request_url(health_url),
                    "full": transport_full,
                    "full_canonical_json": _canonical_json(transport_full),
                    "full_sha256": _canonical_sha256(transport_full),
                    "identity_url": _safe_request_url(identity_url),
                    "error_type": "GatewayBackendHealthCaptureFailed",
                }
                failures.append(
                    "gateway: bound backend /health capture failed"
                )
                continue
        identity, identity_failures = _health_identity(full)
        evidence[base_label] = {
            "url": _safe_request_url(health_url),
            "full": transport_full,
            "full_canonical_json": _canonical_json(transport_full),
            "full_sha256": _canonical_sha256(transport_full),
            "identity": identity,
        }
        if base_label == "gateway" and identity_url != health_url:
            evidence[base_label].update(
                {
                    "identity_url": _safe_request_url(identity_url),
                    "identity_full": full,
                    "identity_full_canonical_json": _canonical_json(full),
                    "identity_full_sha256": _canonical_sha256(full),
                }
            )
        failures.extend(
            f"{base_label}: {failure}" for failure in identity_failures
        )
    return evidence, failures


def _compare_identity_evidence(
    source_before: dict[str, Any],
    source_after: dict[str, Any],
    runner_before: dict[str, Any],
    runner_after: dict[str, Any],
    bundle_before: dict[str, Any],
    bundle_after: dict[str, Any],
    requested_model: str,
    health_before: dict[str, dict[str, Any]],
    health_after: dict[str, dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    for source_field in (
        "git_root",
        "head",
        "tree",
        "clean",
        "status_sha256",
        "python_source_tree_sha256",
        "python_source_file_count",
        "python_source_read_error_count",
        "server_module_sha256",
        "package_init_sha256",
    ):
        if source_before.get(source_field) != source_after.get(source_field):
            failures.append(
                f"source identity changed during the matrix: {source_field}"
            )
    if runner_before != runner_after:
        failures.append("proof runner environment changed during the matrix")
    if bundle_before != bundle_after:
        failures.append("model bundle configuration changed during the matrix")

    observed_identities: list[tuple[str, str, dict[str, Any]]] = []
    for phase, evidence in (("before", health_before), ("after", health_after)):
        for base_label, row in evidence.items():
            identity = row.get("identity")
            if not isinstance(identity, dict):
                failures.append(
                    f"{base_label}: observed backend identity missing {phase}"
                )
                continue
            observed_identities.append((phase, base_label, identity))
            failures.extend(
                f"{base_label}: {failure}"
                for failure in _validate_health_source_binding(
                    identity,
                    source_before if phase == "before" else source_after,
                    runner_before if phase == "before" else runner_after,
                    bundle_before if phase == "before" else bundle_after,
                    requested_model,
                )
            )
    if observed_identities:
        reference = observed_identities[0][2]
        for phase, base_label, identity in observed_identities[1:]:
            if identity != reference:
                failures.append(
                    f"{base_label}: backend runtime/model/cache identity differs "
                    f"at {phase}"
                )
    return failures


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    run_id = str(args.run_id or "")
    if not run_id or _safe_capture_label(run_id) != run_id:
        raise ValueError(
            "--run-id must be a nonempty 80-character-or-shorter identifier "
            "containing only letters, digits, hyphens, or underscores"
        )
    bases = parse_named_urls(args.base_url, "--base-url")
    required_bases = {"direct", "gateway"}
    if not args.allow_single_base and not required_bases.issubset(bases):
        raise ValueError("--base-url must include direct=... and gateway=...")
    base_origin_failures = _validate_base_origins(bases)
    health_urls = parse_named_urls(args.health_url or [], "--health-url")
    unknown_health_labels = sorted(set(health_urls) - set(bases))
    if unknown_health_labels:
        raise ValueError(
            "--health-url names an unknown base: "
            + ", ".join(unknown_health_labels)
        )
    repo_root = Path(args.repo_root).resolve()
    if not (repo_root / FILE_INFO_PATH).is_file():
        raise ValueError(f"repo root does not contain {FILE_INFO_PATH}: {repo_root}")
    protocols = args.protocol or list(PROTOCOLS)
    modes = args.mode or list(MODES)
    raw_artifact_dir = getattr(args, "raw_artifact_dir", None)
    capture_required = "stream" in modes or "nonstream" in modes
    if capture_required and raw_artifact_dir is None:
        raise ValueError(
            "--raw-artifact-dir is required whenever stream mode or "
            "nonstream mode is requested"
        )
    if raw_artifact_dir is not None and not capture_required:
        raise ValueError(
            "--raw-artifact-dir captures streaming parser-input bytes and "
            "nonstream JSON parser-input bytes; request --mode stream or "
            "--mode nonstream"
        )
    git_worktree = _git_worktree_root(repo_root)
    _validate_private_result_destination(Path(args.output), git_worktree)
    installed_release_manifest = getattr(
        args,
        "installed_release_manifest",
        None,
    )
    source_before = observe_source_checkout(repo_root)
    runner_before = (
        observe_runner_environment(
            repo_root,
            installed_release_manifest,
            source_before,
        )
        if installed_release_manifest is not None
        else observe_runner_environment(repo_root)
    )
    bundle_root = Path(args.bundle_root)
    native_reasoning_effort = getattr(args, "native_reasoning_effort", None)
    if native_reasoning_effort is not None:
        config = json.loads((bundle_root / "config.json").read_text())
        template = (bundle_root / "chat_template.jinja").read_text()
        if (
            config.get("model_type") not in {"inkling2", "inkling2_loop"}
            or "reasoning_effort" not in template
            or "enable_thinking" in template
            or not args.enable_thinking
        ):
            raise ValueError("--native-reasoning-effort requires an Inkling effort-only template")
    bundle_before = observe_bundle_configuration(bundle_root)
    identity_failures = _source_identity_failures(
        source_before,
        args.source_head,
    )
    identity_failures.extend(base_origin_failures)
    identity_failures.extend(_runner_environment_failures(runner_before))
    health_before, health_before_failures = _capture_health_evidence(
        bases,
        health_urls,
        args.timeout,
        args.model,
    )
    identity_failures.extend(health_before_failures)
    identity_failures.extend(
        _compare_identity_evidence(
            source_before,
            source_before,
            runner_before,
            runner_before,
            bundle_before,
            bundle_before,
            args.model,
            health_before,
            health_before,
        )
    )
    expected_capture_routes = expected_parser_input_capture_routes(
        bases,
        protocols,
        modes,
        skip_cancellation=args.skip_cancellation,
    )
    raw_recorder: DecompressedParserInputCaptureRecorder | None = None
    if raw_artifact_dir is not None and not identity_failures:
        raw_recorder = DecompressedParserInputCaptureRecorder(
            Path(raw_artifact_dir),
            git_worktree,
            run_id=run_id,
        )
        raw_recorder.configure_expected(expected_capture_routes)
    output: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "requested_model": args.model,
        "repo_root": str(repo_root),
        "bases": bases,
        "protocols": protocols,
        "modes": modes,
        "first_tool_choice": args.first_tool_choice,
        "second_tool_choice": args.second_tool_choice,
        "native_reasoning_effort": native_reasoning_effort,
        "stable_responses_instructions": args.stable_responses_instructions,
        "backend_identity_fingerprint_sha256": None,
        "identity": {
            "source": {
                "declared_head": args.source_head,
                "before": source_before,
                "after": None,
            },
            "runner": {
                "before": runner_before,
                "after": None,
            },
            "bundle": {
                "before": bundle_before,
                "after": None,
            },
            "health": {
                base_label: {
                    "before": health_before.get(base_label),
                    "after": None,
                }
                for base_label in bases
            },
            "failures": list(dict.fromkeys(identity_failures)),
        },
        "raw_capture": (
            {
                **_disabled_capture_summary(),
                "enabled": True,
                "reason": "capture active; manifest pending",
                "expected": len(expected_capture_routes),
                "complete": False,
            }
            if raw_recorder is not None
            else (
                {
                    **_disabled_capture_summary(),
                    "enabled": True,
                    "reason": (
                        "identity preflight failed before private capture setup"
                    ),
                    "expected": len(expected_capture_routes),
                    "complete": False,
                }
                if raw_artifact_dir is not None
                else _disabled_capture_summary()
            )
        ),
        "flows": {},
        "abort_recovery": {},
        "paired_replays": {},
    }
    if identity_failures:
        output["checks"] = {
            "identity_provenance_pass": False,
            "all_requested_flows_present": False,
            "all_flows_pass": False,
            "abort_recovery_skipped": bool(args.skip_cancellation),
            "all_abort_recovery_pass": False,
            "raw_capture_complete": bool(output["raw_capture"]["complete"]),
        }
        output["pass"] = False
        return output

    paired_replay_payloads: dict[
        tuple[str, str, str, int], dict[str, Any]
    ] = {}
    for base_label, base_url in bases.items():
        client = ProtocolClient(
            base_url,
            args.api_key,
            args.timeout,
            base_label=base_label,
            raw_recorder=raw_recorder,
        )
        output["flows"][base_label] = {}
        output["abort_recovery"][base_label] = {}
        health_url = backend_lifecycle_health_url(
            base_label,
            bases,
            health_urls,
        )
        for protocol in protocols:
            output["flows"][base_label][protocol] = {}
            output["abort_recovery"][base_label][protocol] = {}
            for mode in modes:
                try:
                    flow = run_flow(
                        client,
                        base_label=base_label,
                        protocol=protocol,
                        mode=mode,
                        model=args.model,
                        repo_root=repo_root,
                        max_tokens=args.max_output_tokens,
                        enable_thinking=args.enable_thinking,
                        second_tool_choice=args.second_tool_choice,
                        first_tool_choice=args.first_tool_choice,
                        native_reasoning_effort=native_reasoning_effort,
                        stable_responses_instructions=(
                            args.stable_responses_instructions
                        ),
                    )
                    flow_replay_payloads = flow.pop(
                        "_paired_replay_payloads", {}
                    )
                    prefinal = flow.pop("_prefinal_payload", None)
                    if isinstance(flow_replay_payloads, dict):
                        for replay_stage, replay_payload in (
                            flow_replay_payloads.items()
                        ):
                            if isinstance(replay_payload, dict):
                                paired_replay_payloads[
                                    (
                                        base_label,
                                        protocol,
                                        mode,
                                        int(replay_stage),
                                    )
                                ] = replay_payload
                    output["flows"][base_label][protocol][mode] = flow
                    if (
                        args.skip_cancellation
                        or not flow.get("pass")
                        or prefinal is None
                    ):
                        continue
                    try:
                        if mode == "stream":
                            aborted = abort_stream_after_deltas(
                                client,
                                protocol,
                                prefinal,
                                health_url=health_url,
                                minimum_deltas=args.minimum_abort_deltas,
                            )
                        else:
                            aborted = disconnect_nonstream(
                                client,
                                protocol,
                                prefinal,
                                health_url=health_url,
                                delay_ms=args.disconnect_delay_ms,
                            )
                        abort_classification = classify_abort(
                            protocol,
                            mode,
                            aborted,
                            args.minimum_abort_deltas,
                        )
                        recovery_marker = f"RECOVERY-{base_label.upper()}-{protocol.upper()}-{mode.upper()}-DONE"
                        recovered = run_recovery(
                            client,
                            protocol,
                            mode,
                            args.model,
                            recovery_marker,
                            args.recovery_max_tokens,
                            native_reasoning_effort=native_reasoning_effort,
                        )
                        output["abort_recovery"][base_label][protocol][mode] = {
                            "abort": aborted,
                            "abort_classification": abort_classification,
                            "recovery": recovered,
                            "pass": abort_classification["pass"]
                            and recovered.get("pass") is True,
                        }
                    except Exception as exc:
                        output["abort_recovery"][base_label][protocol][mode] = {
                            "pass": False,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                except Exception as exc:
                    output["flows"][base_label][protocol][mode] = {
                        "pass": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
    paired_chat_required = (
        "chat" in protocols
        and "nonstream" in modes
        and {"direct", "gateway"}.issubset(bases)
    )
    if paired_chat_required:
        direct_flow = output["flows"].get("direct", {}).get("chat", {}).get(
            "nonstream", {}
        )
        gateway_flow = output["flows"].get("gateway", {}).get("chat", {}).get(
            "nonstream", {}
        )
        for replay_stage in (2, 3):
            replay_key = f"chat_nonstream_round{replay_stage}"
            replay_payload = paired_replay_payloads.get(
                ("direct", "chat", "nonstream", replay_stage)
            )
            if (
                not isinstance(replay_payload, dict)
                or direct_flow.get("pass") is not True
                or gateway_flow.get("pass") is not True
            ):
                output["paired_replays"][replay_key] = {
                    "pass": False,
                    "error_type": "PrerequisiteFlowFailure",
                }
                continue
            try:
                output["paired_replays"][replay_key] = (
                    run_paired_replay_discriminator(
                        direct_client=ProtocolClient(
                            bases["direct"],
                            args.api_key,
                            args.timeout,
                            base_label="direct",
                        ),
                        gateway_client=ProtocolClient(
                            bases["gateway"],
                            args.api_key,
                            args.timeout,
                            base_label="gateway",
                        ),
                        protocol="chat",
                        mode="nonstream",
                        stage=replay_stage,
                        payload=replay_payload,
                        expected_backend_identity_fingerprint=health_before[
                            "direct"
                        ]["identity"]["fingerprint_sha256"],
                        gateway_direct_health_probe=lambda: _get_full_health(
                            health_urls.get(
                                "direct", bases["direct"] + "/health"
                            ),
                            args.timeout,
                        ),
                    )
                )
            except Exception as exc:
                output["paired_replays"][replay_key] = {
                    "pass": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
    flow_rows = [
        row
        for base in output["flows"].values()
        for protocol in base.values()
        for row in protocol.values()
    ]
    abort_rows = [
        row
        for base in output["abort_recovery"].values()
        for protocol in base.values()
        for row in protocol.values()
        if row
    ]
    if raw_recorder is not None:
        try:
            output["raw_capture"] = raw_recorder.finalize()
        except Exception as exc:
            output["raw_capture"] = {
                **raw_recorder.current_summary(),
                "complete": False,
                "manifest_error_type": type(exc).__name__,
            }
    source_after: dict[str, Any] = {}
    runner_after: dict[str, Any] = {}
    bundle_after: dict[str, Any] = {}
    try:
        source_after = observe_source_checkout(repo_root)
    except ValueError as exc:
        identity_failures.append(
            f"source identity capture failed after matrix: {type(exc).__name__}"
        )
    try:
        runner_after = (
            observe_runner_environment(
                repo_root,
                installed_release_manifest,
                source_after or None,
            )
            if installed_release_manifest is not None
            else observe_runner_environment(repo_root)
        )
    except (OSError, ValueError) as exc:
        identity_failures.append(
            f"runner identity capture failed after matrix: {type(exc).__name__}"
        )
    try:
        bundle_after = observe_bundle_configuration(Path(args.bundle_root))
    except (OSError, ValueError) as exc:
        identity_failures.append(
            f"bundle identity capture failed after matrix: {type(exc).__name__}"
        )
    health_after, health_after_failures = _capture_health_evidence(
        bases,
        health_urls,
        args.timeout,
        args.model,
    )
    identity_failures.extend(health_after_failures)
    if source_after and runner_after and bundle_after:
        identity_failures.extend(
            _compare_identity_evidence(
                source_before,
                source_after,
                runner_before,
                runner_after,
                bundle_before,
                bundle_after,
                args.model,
                health_before,
                health_after,
            )
        )
    output["identity"]["source"]["after"] = source_after or None
    output["identity"]["runner"]["after"] = runner_after or None
    output["identity"]["bundle"]["after"] = bundle_after or None
    for base_label in bases:
        output["identity"]["health"][base_label]["after"] = health_after.get(
            base_label
        )
    output["identity"]["failures"] = list(dict.fromkeys(identity_failures))
    if not identity_failures:
        first_base = next(iter(bases))
        output["backend_identity_fingerprint_sha256"] = health_before[
            first_base
        ]["identity"]["fingerprint_sha256"]
    output["checks"] = {
        "identity_provenance_pass": not bool(identity_failures),
        "all_requested_flows_present": len(flow_rows)
        == len(bases) * len(protocols) * len(modes),
        "all_flows_pass": bool(flow_rows)
        and all(row.get("pass") is True for row in flow_rows),
        "abort_recovery_skipped": bool(args.skip_cancellation),
        "all_abort_recovery_pass": (
            True
            if args.skip_cancellation
            else bool(abort_rows) and all(row.get("pass") is True for row in abort_rows)
        ),
        "paired_replay_chat_nonstream_rounds_pass": (
            not paired_chat_required
            or all(
                output["paired_replays"]
                .get(f"chat_nonstream_round{stage}", {})
                .get("pass")
                is True
                for stage in (2, 3)
            )
        ),
        "raw_capture_complete": bool(output["raw_capture"]["complete"]),
    }
    output["pass"] = all(
        value
        for key, value in output["checks"].items()
        if key != "abort_recovery_skipped"
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        action="append",
        required=True,
        metavar="NAME=URL",
        help="Endpoint base; provide direct=... and gateway=...",
    )
    parser.add_argument(
        "--health-url",
        action="append",
        metavar="NAME=URL",
        help="Optional backend health URL per base label for idle checks",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--bundle-root",
        type=Path,
        required=True,
        help=(
            "Exact local model bundle whose fixed configuration files must "
            "match the loaded backend attestation"
        ),
    )
    parser.add_argument("--repo-root", default=os.getcwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--raw-artifact-dir",
        type=Path,
        help=(
            "Private root required for exact decompressed streaming and "
            "nonstream parser-input response bytes with safe-allowlisted "
            "metadata; requires --mode stream or --mode nonstream and must "
            "resolve outside every Git "
            "worktree"
        ),
    )
    parser.add_argument(
        "--source-head",
        required=True,
        help="Expected Git HEAD; the runner must observe an exact clean match",
    )
    parser.add_argument(
        "--installed-release-manifest",
        type=Path,
        help=(
            "External vmlx-installed-release-manifest-v1 for an installed-app "
            "run. The producer must be that app's exact bundled Python and the "
            "manifest must resolve outside every Git worktree."
        ),
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Stable paired Electron/API proof run identifier",
    )
    parser.add_argument("--api-key")
    parser.add_argument("--protocol", action="append", choices=PROTOCOLS)
    parser.add_argument("--mode", action="append", choices=MODES)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--recovery-max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--minimum-abort-deltas", type=int, default=3)
    parser.add_argument("--disconnect-delay-ms", type=int, default=1000)
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--stable-responses-instructions",
        action="store_true",
        help=(
            "Keep the Responses developer instructions constant across the "
            "agent loop so appended input/tool results remain prefix-cacheable."
        ),
    )
    parser.add_argument(
        "--first-tool-choice",
        choices=("current", "auto"),
        default="current",
        help=(
            "Tool-choice policy for the first tool turn. 'current' preserves "
            "the historical explicit-stream/required-nonstream contract; "
            "'auto' enables an append-only auto-to-auto cache proof."
        ),
    )
    parser.add_argument(
        "--second-tool-choice",
        choices=("explicit", "required", "auto"),
        default="explicit",
        help="Tool-choice policy for the second tool turn; run a separate auto row for release proof",
    )
    parser.add_argument(
        "--native-reasoning-effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        help="Use the exact Inkling bundle's effort selector throughout the flow and recovery.",
    )
    parser.add_argument("--skip-cancellation", action="store_true")
    parser.add_argument(
        "--allow-single-base",
        action="store_true",
        help="Diagnostic-only override; release proof requires direct and gateway",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        guarded_worktree = _git_worktree_root(Path(args.repo_root))
        output_path = _validate_private_result_destination(
            Path(args.output),
            guarded_worktree,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "pass": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
            )
        )
        return 1
    try:
        result = run_matrix(args)
    except Exception as exc:
        result = {
            "schema": OUTPUT_SCHEMA,
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "run_id": args.run_id,
            "backend_identity_fingerprint_sha256": None,
            "pass": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    _write_private_result_exclusive(
        output_path,
        result,
        guarded_worktree,
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "pass": result.get("pass"),
                "checks": result.get("checks"),
            },
            indent=2,
        )
    )
    return 0 if result.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
