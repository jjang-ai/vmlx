#!/usr/bin/env python3
"""Run no-publish packaged integrity contracts.

This gate does not build, tag, upload, notarize, or update release feeds. It
checks the release-gate script contracts, bundled Python verifier, and the dry
release gate behavior. The current dry gate is allowed to fail only for the
known objective digest rows.
"""

from __future__ import annotations

import argparse
import email.parser
import email.policy
import hashlib
import json
import os
import plistlib
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

try:
    from tests.cross_matrix.output_counts import parse_counts
except ModuleNotFoundError:  # direct script execution
    from output_counts import parse_counts

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.cross_matrix.run_current_regression_suite import (  # noqa: E402
    CURRENT_OBJECTIVE_DIGEST_ARTIFACT as SUITE_CURRENT_OBJECTIVE_DIGEST_ARTIFACT,
)
from tests.cross_matrix.run_current_regression_suite import (  # noqa: E402
    DEFERRED_RELEASE_OPEN_REQUIREMENTS as SUITE_DEFERRED_RELEASE_OPEN_REQUIREMENTS,
)
from tests.cross_matrix.run_current_regression_suite import (  # noqa: E402
    EXPECTED_OPEN_REQUIREMENTS as SUITE_EXPECTED_OPEN_REQUIREMENTS,
)
from tests.cross_matrix.run_current_regression_suite import (  # noqa: E402
    release_gate_digest_failure_is_known,
)

DEFAULT_OUT = Path(
    "build/current-packaged-integrity-contract-after-bundled-python-sync-20260608.json"
)
EXPECTED_OPEN_REQUIREMENTS = SUITE_EXPECTED_OPEN_REQUIREMENTS
CURRENT_OBJECTIVE_DIGEST_ARTIFACT = Path(
    SUITE_CURRENT_OBJECTIVE_DIGEST_ARTIFACT
)
MIN_RELEASE_GATE_UNIT_TESTS = 34
PACKAGED_RENDERER_ASAR = Path(
    "panel/release/sequoia-app/mac-arm64/vMLX.app/Contents/Resources/app.asar"
)
PACKAGED_PYTHON_ROOT = Path(
    "panel/release/sequoia-app/mac-arm64/vMLX.app/Contents/Resources/bundled-python/python"
)
PACKAGED_APP = Path("panel/release/sequoia-app/mac-arm64/vMLX.app")
DEVELOPER_ID_IDENTITY = "D4DBBCB52F666D03F0A5154BFFEA2227BEE8FC7C"
SIGNING_KEYCHAINS = (
    Path("~/Library/Keychains/vmlx-build.keychain-db").expanduser(),
    Path("~/Library/Keychains/build.keychain-db").expanduser(),
    Path("~/Library/Keychains/login.keychain-db").expanduser(),
)
# 2026-08-16 (ledger row 159): repointed to the CURRENT DSV4 cache-UI copy.
# The two sentences pinned before ("restored SWA+CSA/HCA state has not proven
# output-equivalent", "unsafe hits are still rejected") exist in NEITHER the
# freshly staged app, NOR the previously released /Applications app, NOR
# panel source — the copy was reworded, so the check could never pass again.
# These markers carry the same guarantee: the packaged renderer still tells
# the user DSV4 keeps its native composite path and that prefix reuse
# preserves the native SWA+CSA/HCA state. The FORBIDDEN list below is the
# actual de-duplication assertion and is unchanged.
PACKAGED_RENDERER_REQUIRED_DSV4_CACHE_UI_STRINGS = (
    b"DSV4 Flash stays on its native DSV4BatchGenerator path",
    b"DSV4 prefix reuse preserves the native SWA plus CSA/HCA composite state",
)
PACKAGED_RENDERER_FORBIDDEN_DSV4_CACHE_UI_STRINGS = (
    b"DSV4 Native Composite Prefix Cache",
    b"DSV4 Native Cache",
    b"DSV4 Composite Prefix Cache",
    b"DSV4 Pool Quantization",
    b"DSV4 CSA/HCA Pool Codec",
)
PACKAGED_RENDERER_REQUIRED_MAX_THINKING_STRINGS = (
    b"Max Thinking Tokens",
    b"maxThinkingTokens",
    b"max_thinking_tokens",
    b"thinking_budget",
)
PACKAGED_USER_DATA_ISOLATION_STRINGS = (
    b"--vmlx-user-data-dir",
    b"VMLX_USER_DATA_DIR",
    b"VMLINUX_USER_DATA_DIR",
    b"requestSingleInstanceLock",
)
PACKAGED_EPIPE_CLOSED_STREAM_GUARD_STRINGS = (
    b"responseWritable(res)",
    b"!anyRes.closed",
    b"requestWritable(req)",
    b"!anyReq.closed",
    b"function chatBackendRequestWritable(req)",
    b"function imageServerRequestWritable(req)",
    b"!req.closed",
    b"wrappedDisconnects",
    b"reason",
    b"detail",
    b"wrappedDisconnects.some((nested) => isExpectedChildProcessStreamDisconnectError(nested))",
    b"nestedErrors.some((nested) => isExpectedChildProcessStreamDisconnectError(nested))",
    b"wrappedDisconnects.some((nested) => isExpectedImageServerDisconnectError(nested))",
    b"nestedErrors.some((nested) => isExpectedImageServerDisconnectError(nested))",
    b"function isExpectedCacheEndpointDisconnectError",
    b"function fetchCacheJson",
    b"Cache stats",
    b"connection lost. The model server may have stopped or restarted",
    b"wrappedDisconnects.some((nested) => isExpectedCacheEndpointDisconnectError(nested))",
    b"nestedErrors.some((nested) => isExpectedCacheEndpointDisconnectError(nested))",
    b"function isExpectedPerformanceEndpointDisconnectError",
    b"Performance health connection lost. The model server may have stopped or restarted",
    b"wrappedDisconnects.some((nested) => isExpectedPerformanceEndpointDisconnectError(nested))",
    b"nestedErrors.some((nested) => isExpectedPerformanceEndpointDisconnectError(nested))",
)

STAGED_APP_ENGINE_HASH_FILES = (
    "server.py",
    "api/utils.py",
    "api/tool_calling.py",
    "api/anthropic_adapter.py",
    "api/ollama_adapter.py",
    "block_disk_store.py",
    "cli.py",
    "disk_cache.py",
    "engine/batched.py",
    "engine/simple.py",
    "loaders/load_jangtq_dsv4.py",
    "mllm_batch_generator.py",
    "mllm_scheduler.py",
    "model_configs.py",
    "model_config_registry.py",
    "speculative.py",
    "models/llm.py",
    "models/mllm.py",
    "models/step3p7_mlx_vlm.py",
    "omni_multimodal.py",
    "paged_cache.py",
    "prefix_cache.py",
    "runtime_patches/gemma4_processing.py",
    "scheduler.py",
    "tool_parsers/dsml_tool_parser.py",
    "patches/mlx_vlm_mtp/qwen35_vl.py",
    "utils/single_batch_generator.py",
    "utils/head_dim_detection.py",
    "utils/mlx_vlm_compat.py",
    "utils/ssm_companion_cache.py",
    "utils/ssm_companion_disk_store.py",
    "utils/jang_loader.py",
    "utils/nanbeige_runtime.py",
    "utils/tokenizer.py",
    "chat_templates/gemma4.jinja",
    "config/defaults.yaml",
    "metal/codebook_matvec.metal",
    "metal/codebook_moe.metal",
)
R20_RUNTIME_SOURCE_ONLY_FILES = (
    "models/minimax_m3/MODEL-MATRIX-AUTODETECT.txt",
)

SOURCE_HASH_FILES = (
    "tests/cross_matrix/run_packaged_integrity_contract.py",
    "tests/cross_matrix/output_counts.py",
    "panel/scripts/release-gate-python-app.py",
    "panel/scripts/verify-bundled-python.sh",
    "panel/scripts/bundle-python.sh",
    "panel/scripts/build-release-dmgs.sh",
    "panel/scripts/electron-builder-before-pack.cjs",
    "panel/scripts/notarize-release-dmgs.sh",
    "panel/scripts/verify-release-dmgs.sh",
    "panel/src/main/index.ts",
    "panel/src/main/engine-manager.ts",
    "panel/src/main/process-manager.ts",
    "panel/src/main/ipc/developer.ts",
    "panel/src/main/ipc/cache.ts",
    "panel/src/main/ipc/image.ts",
    "panel/src/main/ipc/imageGenerationState.ts",
    "panel/src/main/ipc/models.ts",
    "panel/src/main/tools/executor.ts",
    "panel/src/main/user-data-dir.ts",
    "panel/package.json",
    "pyproject.toml",
    "tests/test_release_gate_python_app.py",
    "tests/test_packaged_integrity_contract.py",
    "panel/tests/release-packaging.test.ts",
    "tests/cross_matrix/summarize_objective_proof.py",
    "tests/test_objective_proof_digest.py",
)

R20_ARTIFACT_CHAIN_SCHEMA_VERSION = 4
R20_ARTIFACT_CHAIN_SCOPE = "r20_production"


def _configured_r20_artifact_chain_version() -> str:
    version = os.environ.get("VMLX_R20_EXPECTED_VERSION", "1.6.20")
    if re.fullmatch(r"\d+\.\d+\.\d+", version) is None:
        raise RuntimeError(f"invalid vMLX production version: {version}")
    return version


R20_ARTIFACT_CHAIN_VERSION = _configured_r20_artifact_chain_version()
R20_ARTIFACT_CHAIN_FLAVORS = ("sequoia", "tahoe")
INSTALLED_RELEASE_MANIFEST_SCHEMA = "vmlx-installed-release-manifest-v1"
INSTALLED_RELEASE_MANIFEST_FIELDS = {
    "app_asar_sha256",
    "bundled_provenance_sha256",
    "bundled_python_executable_fingerprint_sha256",
    "bundled_python_executable_sha256",
    "electron_executable_sha256",
    "schema",
    "source_commit",
    "source_tree",
}
INSTALLED_BUNDLED_PYTHON_RELATIVE_PATH = Path(
    "Contents/Resources/bundled-python/python/bin/python3"
)
R20_FLAVOR_RUNTIME_CONTRACTS = {
    "sequoia": {
        "mlx_wheel_platform": "macosx_14_0_arm64",
        "minimum_system_version": "14.5.0",
    },
    "tahoe": {
        "mlx_wheel_platform": "macosx_26_0_arm64",
        "minimum_system_version": "26.0.0",
    },
}
R20_PINNED_TOOL_NAMES = (
    "git",
    "node",
    "npm",
    "npx",
    "shasum",
    "awk",
    "file",
    "find",
    "asar",
    "app_builder",
    "electron_builder",
)

COMMANDS: dict[str, tuple[Path, list[str]]] = {
    "release_gate_unit_contracts": (
        Path("."),
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_release_gate_python_app.py",
        ],
    ),
    "bundled_python_verifier": (
        Path("panel"),
        [
            "npm",
            "run",
            "verify-bundled",
        ],
    ),
    "release_gate_skip_app": (
        Path("."),
        [
            "panel/scripts/release-gate-python-app.py",
            "--skip-app",
            "--skip-gui",
            "--skip-release-manifest",
        ],
    ),
}


@contextmanager
def _scoped_jang_tools_source(jang_tools_source: Path | None):
    if jang_tools_source is None:
        yield
        return

    keys = ("VMLX_JANG_TOOLS_SOURCE", "VMLINUX_JANG_TOOLS_SOURCE")
    old = {key: os.environ.get(key) for key in keys}
    value = str(jang_tools_source)
    try:
        for key in keys:
            os.environ[key] = value
        yield
    finally:
        for key, previous in old.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


class ArtifactChainError(ValueError):
    """A fail-closed release artifact-chain contract violation."""


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _open_nofollow(path: Path, flags: int) -> int:
    """Open one absolute path without following any path-component symlink."""
    path = _absolute_path(path)
    parts = path.parts
    if not parts or parts[0] != "/":
        raise ArtifactChainError(f"secure open requires an absolute path: {path}")
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
        flags |= os.O_NOFOLLOW
    directory_fd = os.open("/", directory_flags)
    try:
        for component in parts[1:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return os.open(parts[-1], flags, dir_fd=directory_fd)
    except OSError as exc:
        raise ArtifactChainError(f"secure no-follow open failed for {path}: {exc}") from exc
    finally:
        os.close(directory_fd)


def _read_regular_file(
    path: Path,
    *,
    label: str,
    require_nonempty: bool = True,
    capture_bytes: bool = False,
    require_single_link: bool = True,
) -> tuple[dict[str, Any], bytes | None]:
    """Read and describe one regular file through the same no-follow fd."""
    path = _absolute_path(path)
    fd = _open_nofollow(path, os.O_RDONLY)
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if capture_bytes else None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ArtifactChainError(f"{label} is not a regular file: {path}")
        if require_single_link and before.st_nlink != 1:
            raise ArtifactChainError(
                f"{label} must have exactly one hard link, found {before.st_nlink}: {path}"
            )
        if require_nonempty and before.st_size <= 0:
            raise ArtifactChainError(f"{label} is empty: {path}")
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_nlink,
        before.st_mode,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_nlink,
        after.st_mode,
    )
    if identity_before != identity_after:
        raise ArtifactChainError(f"{label} changed while it was read: {path}")
    data = b"".join(chunks) if chunks is not None else None
    if data is not None and len(data) != after.st_size:
        raise ArtifactChainError(f"{label} changed size while it was read: {path}")
    return (
        {
            "path": str(path),
            "sha256": digest.hexdigest(),
            "size": after.st_size,
            "device": after.st_dev,
            "inode": after.st_ino,
            "mtime_ns": after.st_mtime_ns,
            "nlink": after.st_nlink,
            "mode": stat.S_IMODE(after.st_mode),
        },
        data,
    )


def _sha256(path: Path) -> str:
    record, _ = _read_regular_file(path, label=f"SHA-256 input {path}")
    return str(record["sha256"])


def _reject_symlinked_ancestors(path: Path, *, label: str) -> None:
    path = _absolute_path(path)
    chain = list(reversed(path.parents)) + [path]
    for component in chain:
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactChainError(
                f"{label} has a symlinked path component: {component}"
            )


def _safe_regular_file(
    path: Path,
    *,
    label: str,
    require_nonempty: bool = True,
) -> dict[str, Any]:
    path = _absolute_path(path)
    _reject_symlinked_ancestors(path, label=label)
    record, _ = _read_regular_file(
        path,
        label=label,
        require_nonempty=require_nonempty,
    )
    return record


def _directory_open_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _assert_bound_directory_alias(
    path: Path,
    expected: os.stat_result,
    *,
    label: str,
) -> None:
    check_fd = _open_nofollow(path, _directory_open_flags())
    try:
        current = os.fstat(check_fd)
    finally:
        os.close(check_fd)
    if not stat.S_ISDIR(current.st_mode) or (
        current.st_dev,
        current.st_ino,
    ) != (
        expected.st_dev,
        expected.st_ino,
    ):
        raise ArtifactChainError(f"{label} directory identity changed")


def _open_or_create_directory_nofollow(
    path: Path,
    *,
    label: str,
    leaf_mode: int | None = None,
) -> tuple[int, os.stat_result]:
    """Open/create a directory chain without path-based chmod or symlink follow."""
    path = _absolute_path(path)
    parts = path.parts
    if not parts or parts[0] != "/":
        raise ArtifactChainError(f"{label} must be absolute: {path}")
    flags = _directory_open_flags()
    directory_fd = os.open("/", flags)
    try:
        for index, component in enumerate(parts[1:]):
            final = index == len(parts[1:]) - 1
            try:
                next_fd = os.open(component, flags, dir_fd=directory_fd)
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o700, dir_fd=directory_fd)
                except FileExistsError:
                    pass
                try:
                    next_fd = os.open(component, flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise ArtifactChainError(
                        f"{label} has an unsafe created component: {component}"
                    ) from exc
            except OSError as exc:
                raise ArtifactChainError(
                    f"{label} has an unsafe component: {component}"
                ) from exc
            metadata = os.fstat(next_fd)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(next_fd)
                raise ArtifactChainError(
                    f"{label} component is not a directory: {component}"
                )
            if final and leaf_mode is not None:
                os.fchmod(next_fd, leaf_mode)
                os.fsync(next_fd)
                metadata = os.fstat(next_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        metadata = os.fstat(directory_fd)
        _assert_bound_directory_alias(path, metadata, label=label)
        return directory_fd, metadata
    except Exception:
        os.close(directory_fd)
        raise


def _safe_output_basename(name: str, *, label: str) -> str:
    if (
        not name
        or name in {".", ".."}
        or Path(name).name != name
        or "/" in name
        or "\x00" in name
    ):
        raise ArtifactChainError(f"{label} output name is not a safe basename")
    return name


def _record_file_at(
    directory_fd: int,
    name: str,
    *,
    label: str,
    expected_mode: int,
    require_nonempty: bool,
) -> dict[str, Any]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(name, flags, dir_fd=directory_fd)
    digest = hashlib.sha256()
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != expected_mode
            or (require_nonempty and before.st_size <= 0)
        ):
            raise ArtifactChainError(f"{label} is not a sealed private regular file")
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ArtifactChainError(f"{label} changed while read")
    return {
        "sha256": digest.hexdigest(),
        "size": after.st_size,
        "mode": stat.S_IMODE(after.st_mode),
        "device": after.st_dev,
        "inode": after.st_ino,
    }


def _link_sealed_temporary_at(
    directory_fd: int,
    temporary_name: str,
    output_name: str,
    *,
    label: str,
) -> None:
    try:
        os.link(
            temporary_name,
            output_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileExistsError as exc:
        raise ArtifactChainError("concurrent private output already exists") from exc
    os.unlink(temporary_name, dir_fd=directory_fd)
    os.fsync(directory_fd)


def _write_sealed_file_at(
    directory_fd: int,
    output_name: str,
    data: bytes,
    *,
    label: str,
    mode: int,
    require_nonempty: bool,
) -> dict[str, Any]:
    output_name = _safe_output_basename(output_name, label=label)
    try:
        os.stat(output_name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise ArtifactChainError("refusing to overwrite existing private output")
    temporary_name = f".{output_name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(temporary_name, flags, 0o600, dir_fd=directory_fd)
    temporary_created = True
    output_created = False
    try:
        remaining = memoryview(data)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise ArtifactChainError(f"short write for private {label}")
            remaining = remaining[written:]
        os.fsync(fd)
        metadata = os.fstat(fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or (require_nonempty and metadata.st_size <= 0)
        ):
            raise ArtifactChainError(f"private {label} temporary output is invalid")
        os.fchmod(fd, mode)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        _link_sealed_temporary_at(
            directory_fd,
            temporary_name,
            output_name,
            label=label,
        )
        temporary_created = False
        output_created = True
        return _record_file_at(
            directory_fd,
            output_name,
            label=label,
            expected_mode=mode,
            require_nonempty=require_nonempty,
        )
    except Exception:
        if output_created:
            with suppress(FileNotFoundError):
                os.unlink(output_name, dir_fd=directory_fd)
        raise
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary_created:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=directory_fd)


def _exclusive_sealed_write(path: Path, data: bytes, *, mode: int = 0o400) -> None:
    path = _absolute_path(path)
    directory_fd, directory_metadata = _open_or_create_directory_nofollow(
        path.parent,
        label="private output directory",
    )
    output_created = False
    try:
        record = _write_sealed_file_at(
            directory_fd,
            path.name,
            data,
            label="sealed private output",
            mode=mode,
            require_nonempty=True,
        )
        output_created = True
        _assert_bound_directory_alias(
            path.parent,
            directory_metadata,
            label="private output directory",
        )
    except Exception:
        if output_created:
            with suppress(FileNotFoundError):
                os.unlink(path.name, dir_fd=directory_fd)
        raise
    finally:
        os.close(directory_fd)
    if record["size"] != len(data):
        raise ArtifactChainError("sealed private output size changed after creation")


def _write_private_json_with_digest(path: Path, payload: dict[str, Any]) -> str:
    path = _absolute_path(path)
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    _exclusive_sealed_write(path, encoded)
    return digest


def _require_private_mode(
    path: Path,
    *,
    label: str,
    expected: int = 0o400,
    record: dict[str, Any] | None = None,
) -> None:
    path = _absolute_path(path)
    if record is None:
        record = _safe_regular_file(path, label=label)
    mode = int(record["mode"])
    if mode != expected:
        raise ArtifactChainError(
            f"{label} must have mode {expected:04o}, found {mode:04o}: {path}"
        )


def _validate_independent_digest(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
) -> str:
    path = _absolute_path(path)
    record = _safe_regular_file(path, label=label)
    _require_private_mode(path, label=label, record=record)
    expected = expected_sha256.strip()
    if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ArtifactChainError(f"{label} independently supplied digest is malformed")
    actual = str(record["sha256"])
    if actual != expected:
        raise ArtifactChainError(
            f"{label} digest mismatch: expected {expected}, found {actual}"
        )
    return actual


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/usr/bin/git", "-C", str(root), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def _git_source_identity(root: Path, *, require_clean: bool = True) -> dict[str, str]:
    root = _absolute_path(root)
    _reject_symlinked_ancestors(root, label="release source root")
    if root.is_symlink() or not root.is_dir():
        raise ArtifactChainError(f"release source root is invalid: {root}")
    top = _run_git(root, "rev-parse", "--show-toplevel")
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != root.resolve():
        raise ArtifactChainError(f"release source is not the exact Git root: {root}")
    commit = _run_git(root, "rev-parse", "HEAD")
    tree = _run_git(root, "rev-parse", "HEAD^{tree}")
    if commit.returncode != 0 or tree.returncode != 0:
        raise ArtifactChainError("release source Git identity is unavailable")
    if require_clean:
        status_arguments = [
            "-C",
            str(root),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ]
        status_result = _run_git(
            root,
            "status",
            "--porcelain",
            "--untracked-files=all",
        )
        status_stdout = status_result.stdout
        if status_result.returncode == 0:
            status_stdout = _strip_verified_release_python_action_from_git_status(
                arguments=status_arguments,
                stdout=status_stdout,
            )
        if status_result.returncode != 0 or status_stdout:
            raise ArtifactChainError("release source is not clean")
    return {
        "root": str(root),
        "commit": commit.stdout.strip(),
        "tree": tree.stdout.strip(),
    }


def ensure_private_evidence_root(private_root: Path) -> Path:
    private_root = _absolute_path(private_root)
    root_fd, root_metadata = _open_or_create_directory_nofollow(
        private_root,
        label="private evidence root",
        leaf_mode=0o700,
    )
    try:
        if stat.S_IMODE(root_metadata.st_mode) != 0o700:
            raise ArtifactChainError("private evidence root mode is not 0700")
        inside_worktree = _run_git(
            private_root,
            "rev-parse",
            "--is-inside-work-tree",
        )
        inside_git_dir = _run_git(
            private_root,
            "rev-parse",
            "--is-inside-git-dir",
        )
        _assert_bound_directory_alias(
            private_root,
            root_metadata,
            label="private evidence root",
        )
        if (
            inside_worktree.returncode == 0
            and inside_worktree.stdout.strip() == "true"
        ) or (
            inside_git_dir.returncode == 0
            and inside_git_dir.stdout.strip() == "true"
        ):
            raise ArtifactChainError(
                f"private evidence root must be outside every Git worktree: {private_root}"
            )
    finally:
        os.close(root_fd)
    return private_root


def create_private_directory(
    *,
    private_root: Path,
    directory: Path,
    label: str,
) -> Path:
    """Create one fresh private directory without following path symlinks."""
    private_root = ensure_private_evidence_root(private_root)
    directory = _absolute_path(directory)
    try:
        relative = directory.relative_to(private_root)
    except ValueError as exc:
        raise ArtifactChainError(
            f"{label} must stay under the private evidence root: {directory}"
        ) from exc
    if not relative.parts:
        raise ArtifactChainError(f"{label} must not replace the private evidence root")

    directory_flags = _directory_open_flags()
    parent_fd = _open_nofollow(private_root, directory_flags)
    try:
        for index, component in enumerate(relative.parts):
            final = index == len(relative.parts) - 1
            try:
                os.mkdir(component, 0o700, dir_fd=parent_fd)
            except FileExistsError as exc:
                if final:
                    raise ArtifactChainError(
                        f"refusing to reuse prior {label}: {directory}"
                    ) from exc
            try:
                next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            except OSError as exc:
                raise ArtifactChainError(
                    f"{label} has an unsafe directory component: {component}"
                ) from exc
            metadata = os.fstat(next_fd)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(next_fd)
                raise ArtifactChainError(
                    f"{label} path component is not a directory: {component}"
                )
            os.fchmod(next_fd, 0o700)
            os.close(parent_fd)
            parent_fd = next_fd
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return directory


def _assert_within_private_root(private_root: Path, path: Path, *, label: str) -> Path:
    private_root = ensure_private_evidence_root(private_root)
    path = _absolute_path(path)
    _reject_symlinked_ancestors(path, label=label)
    if path.is_symlink():
        raise ArtifactChainError(f"{label} must not be a symlink: {path}")
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(private_root)
    except ValueError as exc:
        raise ArtifactChainError(
            f"{label} must stay under the private evidence root: {resolved}"
        ) from exc
    return resolved


def _expected_release_artifact_paths(
    dist_dir: Path,
    version: str,
) -> dict[str, dict[str, Path]]:
    dist_dir = _absolute_path(dist_dir)
    if dist_dir.is_symlink():
        raise ArtifactChainError(f"release output directory is symlinked: {dist_dir}")
    if not dist_dir.is_dir():
        raise ArtifactChainError(f"release output directory is missing: {dist_dir}")
    return {
        flavor: {
            "dmg": dist_dir / f"vMLX-{version}-{flavor}-arm64.dmg",
            "blockmap": dist_dir / f"vMLX-{version}-{flavor}-arm64.dmg.blockmap",
        }
        for flavor in R20_ARTIFACT_CHAIN_FLAVORS
    }


def _release_artifact_records(
    dist_dir: Path,
    version: str,
) -> dict[str, dict[str, Any]]:
    dist_dir = _absolute_path(dist_dir)
    expected = _expected_release_artifact_paths(dist_dir, version)
    expected_names = {
        path.name
        for paths in expected.values()
        for path in paths.values()
    }
    actual_names = {
        path.name
        for path in dist_dir.iterdir()
        if path.name.lower().endswith(".dmg")
        or path.name.lower().endswith(".dmg.blockmap")
    }
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ArtifactChainError(
            "release output must contain the exact Sequoia/Tahoe DMG and blockmap "
            f"set; missing={missing}, extra={extra}"
        )
    records: dict[str, dict[str, Any]] = {}
    for flavor, paths in expected.items():
        dmg = _safe_regular_file(paths["dmg"], label=f"{flavor} DMG")
        blockmap = _safe_regular_file(
            paths["blockmap"],
            label=f"{flavor} blockmap",
        )
        records[flavor] = {
            "flavor": flavor,
            "dmg_path": dmg["path"],
            "dmg_sha256": dmg["sha256"],
            "dmg_size": dmg["size"],
            "blockmap_path": blockmap["path"],
            "blockmap_sha256": blockmap["sha256"],
            "blockmap_size": blockmap["size"],
        }
    return records


def _require_exact_dict_keys(
    value: Any,
    expected: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactChainError(f"{label} must be a JSON object")
    actual = set(value)
    if actual != expected:
        raise ArtifactChainError(
            f"{label} fields do not match schema; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return value


def _read_json_object_with_record(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    require_private_mode: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        record, encoded = _read_regular_file(
            path,
            label=label,
            capture_bytes=True,
        )
        assert encoded is not None
        value = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactChainError(f"{label} is not valid JSON: {path}") from exc
    if expected_sha256 is not None:
        expected = expected_sha256.strip()
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ArtifactChainError(f"{label} independently supplied digest is malformed")
        if record["sha256"] != expected:
            raise ArtifactChainError(
                f"{label} digest mismatch: expected {expected}, found {record['sha256']}"
            )
    if require_private_mode:
        _require_private_mode(
            _absolute_path(path),
            label=label,
            record=record,
        )
    if not isinstance(value, dict):
        raise ArtifactChainError(f"{label} must be a JSON object: {path}")
    return value, record


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    value, _ = _read_json_object_with_record(path, label=label)
    return value


def _process_parent_pid(pid: int) -> int:
    completed = subprocess.run(
        ["/bin/ps", "-o", "ppid=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or re.fullmatch(r"[0-9]+", value) is None:
        raise ArtifactChainError(
            f"could not establish the live parent of release process {pid}"
        )
    return int(value)


def _require_live_driver_ancestor(driver_pid: int) -> None:
    if driver_pid <= 1:
        raise ArtifactChainError("build-driver process identity is invalid")
    current_pid = os.getpid()
    seen = {current_pid}
    for _ in range(16):
        parent_pid = (
            os.getppid()
            if current_pid == os.getpid()
            else _process_parent_pid(current_pid)
        )
        if parent_pid == driver_pid:
            try:
                os.kill(driver_pid, 0)
            except OSError as exc:
                raise ArtifactChainError(
                    "build-driver attestation requires a live release-driver ancestor"
                ) from exc
            return
        if parent_pid <= 1 or parent_pid in seen:
            break
        seen.add(parent_pid)
        current_pid = parent_pid
    raise ArtifactChainError(
        "build-driver attestation must be written by its live release-driver ancestor"
    )


def write_build_driver_attestation(
    *,
    root: Path,
    dist_dir: Path,
    version: str,
    preflight_path: Path,
    private_root: Path,
    output_path: Path,
    nonce: str,
    driver_pid: int,
    staged_outputs: dict[str, Path],
    extracted_asars: dict[str, Path],
    hook_attestations: dict[str, tuple[Path, str]],
    dmg_parity_attestations: dict[str, tuple[Path, str]],
) -> dict[str, Any]:
    if version != R20_ARTIFACT_CHAIN_VERSION:
        raise ArtifactChainError(
            f"r20 artifact chain requires version {R20_ARTIFACT_CHAIN_VERSION}"
        )
    if re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
        raise ArtifactChainError("build-driver nonce must be 256 unpredictable bits")
    _require_live_driver_ancestor(driver_pid)
    if any(
        set(value) != set(R20_ARTIFACT_CHAIN_FLAVORS)
        for value in (
            staged_outputs,
            extracted_asars,
            hook_attestations,
            dmg_parity_attestations,
        )
    ):
        raise ArtifactChainError("build-driver attestation requires both exact flavors")
    private_root = ensure_private_evidence_root(private_root)
    output_path = _assert_within_private_root(
        private_root,
        output_path,
        label="build-driver attestation",
    )
    source = _git_source_identity(root)
    preflight_payload, preflight = _read_json_object_with_record(
        preflight_path,
        label="r20 preflight manifest",
    )
    if preflight_payload.get("status") != "pass":
        raise ArtifactChainError("r20 preflight manifest is not a passing result")
    artifacts = _release_artifact_records(dist_dir, version)
    staged = {
        flavor: validate_staged_app_parity(
            root=root,
            staged_output=staged_outputs[flavor],
            extracted_asar=extracted_asars[flavor],
            version=version,
            flavor=flavor,
        )
        for flavor in R20_ARTIFACT_CHAIN_FLAVORS
    }
    hook_completion: dict[str, dict[str, Any]] = {}
    dmg_payload_parity: dict[str, dict[str, Any]] = {}
    runtime_contracts: dict[str, dict[str, Any]] = {}
    toolchain: dict[str, Any] | None = None
    for flavor in R20_ARTIFACT_CHAIN_FLAVORS:
        hook_path, hook_sha256 = hook_attestations[flavor]
        hook = _validate_hook_completion_attestation(
            root=root,
            dist_dir=dist_dir,
            version=version,
            private_root=private_root,
            flavor=flavor,
            attestation_path=hook_path,
            expected_sha256=hook_sha256,
            expected_nonce=nonce,
            expected_driver_pid=driver_pid,
        )
        parity_path, parity_sha256 = dmg_parity_attestations[flavor]
        parity = _validate_dmg_payload_parity_attestation(
            private_root=private_root,
            flavor=flavor,
            attestation_path=parity_path,
            expected_sha256=parity_sha256,
            expected_hook_sha256=hook["sha256"],
            expected_nonce=nonce,
            expected_driver_pid=driver_pid,
        )
        if (
            hook["payload"]["preflight_sha256"] != preflight["sha256"]
            or hook["payload"]["staged_app"]["payload"]
            != staged[flavor]["app_payload"]
            or hook["payload"]["extracted_asar"]["payload"]
            != staged[flavor]["asar_payload"]
            or parity["payload"]["app_payload"] != staged[flavor]["app_payload"]
            or parity["payload"]["asar_payload"] != staged[flavor]["asar_payload"]
            or hook["payload"]["runtime_contract"]
            != staged[flavor]["runtime_contract"]
            or parity["payload"]["runtime_contract"]
            != staged[flavor]["runtime_contract"]
        ):
            raise ArtifactChainError(
                f"{flavor} hook/mounted payload differs from staged source parity"
            )
        hook_completion[flavor] = {
            "path": hook["path"],
            "sha256": hook["sha256"],
            "plan_sha256": hook["payload"]["plan"]["sha256"],
        }
        dmg_payload_parity[flavor] = {
            "path": parity["path"],
            "sha256": parity["sha256"],
            "hook_sha256": hook["sha256"],
        }
        runtime_contracts[flavor] = staged[flavor]["runtime_contract"]
        if toolchain is None:
            toolchain = hook["payload"]["tools"]
        elif toolchain != hook["payload"]["tools"]:
            raise ArtifactChainError(
                "Sequoia and Tahoe hook toolchains are not identical"
            )
    assert toolchain is not None
    payload = {
        "schema_version": R20_ARTIFACT_CHAIN_SCHEMA_VERSION,
        "scope": R20_ARTIFACT_CHAIN_SCOPE,
        "stage": "build_driver",
        "version": version,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "nonce": nonce,
        "driver_pid": driver_pid,
        "source": source,
        "preflight": {
            "path": preflight["path"],
            "sha256": preflight["sha256"],
        },
        "staged": staged,
        "runtime_contracts": runtime_contracts,
        "toolchain": toolchain,
        "hook_completion": hook_completion,
        "dmg_payload_parity": dmg_payload_parity,
        "artifacts": artifacts,
    }
    digest = _write_private_json_with_digest(output_path, payload)
    return {"attestation": str(output_path), "sha256": digest, "payload": payload}


def _validate_build_driver_attestation(
    *,
    root: Path,
    dist_dir: Path,
    version: str,
    private_root: Path,
    attestation_path: Path,
    expected_attestation_sha256: str,
    expected_nonce: str | None,
    expected_driver_pid: int | None,
    require_current_artifacts: bool,
) -> dict[str, Any]:
    private_root = ensure_private_evidence_root(private_root)
    attestation_path = _assert_within_private_root(
        private_root,
        attestation_path,
        label="build-driver attestation",
    )
    payload, record = _read_json_object_with_record(
        attestation_path,
        label="build-driver attestation",
        expected_sha256=expected_attestation_sha256,
        require_private_mode=True,
    )
    _require_exact_dict_keys(
        payload,
        {
            "schema_version",
            "scope",
            "stage",
            "version",
            "created_at",
            "nonce",
            "driver_pid",
            "source",
            "preflight",
            "staged",
            "runtime_contracts",
            "toolchain",
            "hook_completion",
            "dmg_payload_parity",
            "artifacts",
        },
        label="build-driver attestation",
    )
    if (
        payload["schema_version"] != R20_ARTIFACT_CHAIN_SCHEMA_VERSION
        or payload["scope"] != R20_ARTIFACT_CHAIN_SCOPE
        or payload["stage"] != "build_driver"
        or payload["version"] != version
        or re.fullmatch(r"[0-9a-f]{64}", str(payload["nonce"])) is None
    ):
        raise ArtifactChainError("build-driver attestation identity is invalid")
    if expected_nonce is not None and payload["nonce"] != expected_nonce:
        raise ArtifactChainError("build-driver nonce does not match the live driver")
    if expected_driver_pid is not None and payload["driver_pid"] != expected_driver_pid:
        raise ArtifactChainError("build-driver PID does not match the live driver")
    current_source = _git_source_identity(root)
    if payload["source"] != current_source:
        raise ArtifactChainError("build-driver source no longer matches clean Git source")
    preflight = _require_exact_dict_keys(
        payload["preflight"],
        {"path", "sha256"},
        label="build-driver preflight",
    )
    current_preflight_payload, current_preflight = _read_json_object_with_record(
        Path(str(preflight["path"])),
        label="build-driver preflight manifest",
        expected_sha256=str(preflight["sha256"]),
    )
    if current_preflight_payload.get("status") != "pass":
        raise ArtifactChainError("build-driver preflight is no longer passing")
    if set(payload["staged"]) != set(R20_ARTIFACT_CHAIN_FLAVORS):
        raise ArtifactChainError("build-driver staged evidence is incomplete")
    if set(payload["runtime_contracts"]) != set(R20_ARTIFACT_CHAIN_FLAVORS):
        raise ArtifactChainError("build-driver runtime-contract evidence is incomplete")
    _validate_pinned_toolchain(
        payload["toolchain"],
        label="build-driver toolchain",
    )
    if set(payload["hook_completion"]) != set(R20_ARTIFACT_CHAIN_FLAVORS) or set(
        payload["dmg_payload_parity"]
    ) != set(R20_ARTIFACT_CHAIN_FLAVORS):
        raise ArtifactChainError("build-driver completion evidence is incomplete")
    for flavor in R20_ARTIFACT_CHAIN_FLAVORS:
        staged = payload["staged"][flavor]
        expected_app = _absolute_path(Path(str(staged["app"])))
        expected_parent = _absolute_path(dist_dir) / f"{flavor}-app/mac-arm64/vMLX.app"
        if expected_app != expected_parent:
            raise ArtifactChainError(
                f"build-driver {flavor} app path is not the exact staged path"
            )
        if not isinstance(staged.get("app_payload"), dict) or not isinstance(
            staged.get("asar_payload"), dict
        ):
            raise ArtifactChainError(f"build-driver {flavor} payload evidence is missing")
        hook_reference = _require_exact_dict_keys(
            payload["hook_completion"][flavor],
            {"path", "sha256", "plan_sha256"},
            label=f"build-driver {flavor} hook completion",
        )
        hook = _validate_hook_completion_attestation(
            root=root,
            dist_dir=dist_dir,
            version=version,
            private_root=private_root,
            flavor=flavor,
            attestation_path=Path(str(hook_reference["path"])),
            expected_sha256=str(hook_reference["sha256"]),
            expected_nonce=str(payload["nonce"]),
            expected_driver_pid=int(payload["driver_pid"]),
            require_current_artifacts=require_current_artifacts,
        )
        if hook["payload"]["plan"]["sha256"] != hook_reference["plan_sha256"]:
            raise ArtifactChainError(
                f"build-driver {flavor} hook plan digest changed"
            )
        parity_reference = _require_exact_dict_keys(
            payload["dmg_payload_parity"][flavor],
            {"path", "sha256", "hook_sha256"},
            label=f"build-driver {flavor} DMG parity",
        )
        parity = _validate_dmg_payload_parity_attestation(
            private_root=private_root,
            flavor=flavor,
            attestation_path=Path(str(parity_reference["path"])),
            expected_sha256=str(parity_reference["sha256"]),
            expected_hook_sha256=str(parity_reference["hook_sha256"]),
            expected_nonce=str(payload["nonce"]),
            expected_driver_pid=int(payload["driver_pid"]),
        )
        if (
            hook["sha256"] != parity_reference["hook_sha256"]
            or hook["payload"]["staged_app"]["payload"] != staged["app_payload"]
            or hook["payload"]["extracted_asar"]["payload"]
            != staged["asar_payload"]
            or parity["payload"]["app_payload"] != staged["app_payload"]
            or parity["payload"]["asar_payload"] != staged["asar_payload"]
            or hook["payload"]["runtime_contract"] != staged["runtime_contract"]
            or parity["payload"]["runtime_contract"] != staged["runtime_contract"]
            or payload["runtime_contracts"][flavor] != staged["runtime_contract"]
            or hook["payload"]["tools"] != payload["toolchain"]
        ):
            raise ArtifactChainError(
                f"build-driver {flavor} mounted payload lineage changed"
            )
    if require_current_artifacts:
        current_artifacts = _release_artifact_records(dist_dir, version)
        if payload["artifacts"] != current_artifacts:
            raise ArtifactChainError("build-driver artifacts changed after attestation")
    return {
        "attestation": str(attestation_path),
        "sha256": record["sha256"],
        "payload": payload,
    }


def write_pre_notary_artifact_manifest(
    *,
    root: Path,
    dist_dir: Path,
    version: str,
    private_root: Path,
    output_path: Path,
    build_attestation_path: Path,
    expected_build_attestation_sha256: str,
    expected_nonce: str,
    expected_driver_pid: int,
) -> dict[str, Any]:
    if version != R20_ARTIFACT_CHAIN_VERSION:
        raise ArtifactChainError(
            f"r20 artifact chain requires version {R20_ARTIFACT_CHAIN_VERSION}"
        )
    root = _absolute_path(root)
    private_root = ensure_private_evidence_root(private_root)
    output_path = _assert_within_private_root(
        private_root,
        output_path,
        label="pre-notary artifact handoff",
    )
    attestation = _validate_build_driver_attestation(
        root=root,
        dist_dir=dist_dir,
        version=version,
        private_root=private_root,
        attestation_path=build_attestation_path,
        expected_attestation_sha256=expected_build_attestation_sha256,
        expected_nonce=expected_nonce,
        expected_driver_pid=expected_driver_pid,
        require_current_artifacts=True,
    )
    payload = {
        "schema_version": R20_ARTIFACT_CHAIN_SCHEMA_VERSION,
        "scope": R20_ARTIFACT_CHAIN_SCOPE,
        "stage": "pre_notary",
        "version": version,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": attestation["payload"]["source"],
        "preflight": attestation["payload"]["preflight"],
        "build_attestation": {
            "path": attestation["attestation"],
            "sha256": attestation["sha256"],
            "nonce": attestation["payload"]["nonce"],
            "driver_pid": attestation["payload"]["driver_pid"],
        },
        "staged": attestation["payload"]["staged"],
        "runtime_contracts": attestation["payload"]["runtime_contracts"],
        "toolchain": attestation["payload"]["toolchain"],
        "hook_completion": attestation["payload"]["hook_completion"],
        "dmg_payload_parity": attestation["payload"]["dmg_payload_parity"],
        "artifacts": attestation["payload"]["artifacts"],
    }
    digest = _write_private_json_with_digest(output_path, payload)
    return {
        "manifest": str(output_path),
        "sha256": digest,
        "payload": payload,
    }


def _validate_pre_notary_artifact_manifest_metadata(
    *,
    root: Path,
    dist_dir: Path,
    version: str,
    private_root: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
    expected_source_commit: str,
    expected_source_tree: str,
    expected_preflight_sha256: str,
    require_current_artifact_hashes: bool,
) -> dict[str, Any]:
    if version != R20_ARTIFACT_CHAIN_VERSION:
        raise ArtifactChainError(
            f"r20 artifact chain requires version {R20_ARTIFACT_CHAIN_VERSION}"
        )
    root = _absolute_path(root)
    private_root = ensure_private_evidence_root(private_root)
    manifest_path = _assert_within_private_root(
        private_root,
        manifest_path,
        label="pre-notary artifact handoff",
    )
    payload, manifest_record = _read_json_object_with_record(
        manifest_path,
        label="pre-notary artifact manifest",
        expected_sha256=expected_manifest_sha256,
        require_private_mode=True,
    )
    manifest_digest = str(manifest_record["sha256"])
    _require_exact_dict_keys(
        payload,
        {
            "schema_version",
            "scope",
            "stage",
            "version",
            "created_at",
            "source",
            "preflight",
            "build_attestation",
            "staged",
            "runtime_contracts",
            "toolchain",
            "hook_completion",
            "dmg_payload_parity",
            "artifacts",
        },
        label="pre-notary artifact manifest",
    )
    if (
        payload["schema_version"] != R20_ARTIFACT_CHAIN_SCHEMA_VERSION
        or payload["scope"] != R20_ARTIFACT_CHAIN_SCOPE
        or payload["stage"] != "pre_notary"
        or payload["version"] != version
    ):
        raise ArtifactChainError("pre-notary artifact manifest identity is invalid")
    source = _require_exact_dict_keys(
        payload["source"],
        {"root", "commit", "tree"},
        label="pre-notary source",
    )
    current_source = _git_source_identity(root)
    if source != current_source:
        raise ArtifactChainError(
            "pre-notary artifact manifest source does not match current clean Git source"
        )
    if (
        expected_source_commit != current_source["commit"]
        or expected_source_tree != current_source["tree"]
    ):
        raise ArtifactChainError(
            "independently supplied source commit/tree do not match current release source"
        )
    preflight = _require_exact_dict_keys(
        payload["preflight"],
        {"path", "sha256"},
        label="pre-notary preflight",
    )
    current_preflight_payload, current_preflight = _read_json_object_with_record(
        Path(str(preflight["path"])),
        label="r20 preflight manifest",
        expected_sha256=str(preflight["sha256"]),
    )
    if current_preflight_payload.get("status") != "pass":
        raise ArtifactChainError("r20 preflight manifest is no longer passing")
    if expected_preflight_sha256 != preflight["sha256"]:
        raise ArtifactChainError(
            "independently supplied preflight digest does not match the build handoff"
        )
    build_attestation = _require_exact_dict_keys(
        payload["build_attestation"],
        {"path", "sha256", "nonce", "driver_pid"},
        label="pre-notary build attestation",
    )
    attestation = _validate_build_driver_attestation(
        root=root,
        dist_dir=dist_dir,
        version=version,
        private_root=private_root,
        attestation_path=Path(str(build_attestation["path"])),
        expected_attestation_sha256=str(build_attestation["sha256"]),
        expected_nonce=str(build_attestation["nonce"]),
        expected_driver_pid=int(build_attestation["driver_pid"]),
        require_current_artifacts=require_current_artifact_hashes,
    )
    if payload["source"] != attestation["payload"]["source"]:
        raise ArtifactChainError("pre-notary source differs from build-driver attestation")
    if payload["preflight"] != attestation["payload"]["preflight"]:
        raise ArtifactChainError("pre-notary preflight differs from build-driver attestation")
    if payload["staged"] != attestation["payload"]["staged"]:
        raise ArtifactChainError("pre-notary staged evidence differs from build-driver attestation")
    if payload["runtime_contracts"] != attestation["payload"]["runtime_contracts"]:
        raise ArtifactChainError(
            "pre-notary runtime contracts differ from build-driver attestation"
        )
    if payload["toolchain"] != attestation["payload"]["toolchain"]:
        raise ArtifactChainError(
            "pre-notary toolchain differs from build-driver attestation"
        )
    _validate_pinned_toolchain(
        payload["toolchain"],
        label="pre-notary toolchain",
    )
    if payload["hook_completion"] != attestation["payload"]["hook_completion"]:
        raise ArtifactChainError(
            "pre-notary hook completion differs from build-driver attestation"
        )
    if payload["dmg_payload_parity"] != attestation["payload"]["dmg_payload_parity"]:
        raise ArtifactChainError(
            "pre-notary mounted DMG parity differs from build-driver attestation"
        )
    if payload["artifacts"] != attestation["payload"]["artifacts"]:
        raise ArtifactChainError("pre-notary artifacts differ from build-driver attestation")
    artifacts = _require_exact_dict_keys(
        payload["artifacts"],
        set(R20_ARTIFACT_CHAIN_FLAVORS),
        label="pre-notary artifacts",
    )
    expected_paths = _expected_release_artifact_paths(dist_dir, version)
    current_records = (
        _release_artifact_records(dist_dir, version)
        if require_current_artifact_hashes
        else None
    )
    artifact_keys = {
        "flavor",
        "dmg_path",
        "dmg_sha256",
        "dmg_size",
        "blockmap_path",
        "blockmap_sha256",
        "blockmap_size",
    }
    for flavor in R20_ARTIFACT_CHAIN_FLAVORS:
        record = _require_exact_dict_keys(
            artifacts[flavor],
            artifact_keys,
            label=f"pre-notary {flavor} artifact",
        )
        if record["flavor"] != flavor:
            raise ArtifactChainError(f"pre-notary flavor mismatch for {flavor}")
        if record["dmg_path"] != str(expected_paths[flavor]["dmg"]) or record[
            "blockmap_path"
        ] != str(expected_paths[flavor]["blockmap"]):
            raise ArtifactChainError(
                f"pre-notary {flavor} artifact paths do not match the exact release set"
            )
        if current_records is not None and record != current_records[flavor]:
            raise ArtifactChainError(
                f"pre-notary {flavor} DMG or blockmap changed after build"
            )
    return {
        "manifest": str(manifest_path),
        "sha256": manifest_digest,
        "payload": payload,
    }


def validate_pre_notary_artifact_manifest(
    *,
    root: Path,
    dist_dir: Path,
    version: str,
    private_root: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
    expected_source_commit: str,
    expected_source_tree: str,
    expected_preflight_sha256: str,
) -> dict[str, Any]:
    return _validate_pre_notary_artifact_manifest_metadata(
        root=root,
        dist_dir=dist_dir,
        version=version,
        private_root=private_root,
        manifest_path=manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_source_commit=expected_source_commit,
        expected_source_tree=expected_source_tree,
        expected_preflight_sha256=expected_preflight_sha256,
        require_current_artifact_hashes=True,
    )


def seal_private_capture(
    *,
    private_root: Path,
    temporary_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    private_root = ensure_private_evidence_root(private_root)
    temporary_path = _assert_within_private_root(
        private_root,
        temporary_path,
        label="temporary private capture",
    )
    output_path = _assert_within_private_root(
        private_root,
        output_path,
        label="sealed private capture",
    )
    temporary = _safe_regular_file(
        temporary_path,
        label="temporary private capture",
    )
    _require_private_mode(
        temporary_path,
        label="temporary private capture",
        expected=0o600,
        record=temporary,
    )
    sealed = _copy_immutable_file(
        temporary_path,
        output_path,
        expected_sha256=temporary["sha256"],
        label="temporary private capture",
    )
    temporary_path.unlink()
    return sealed


def capture_private_command(
    *,
    private_root: Path,
    result_dir: Path,
    output_name: str,
    stderr_name: str,
    label: str,
    command: list[str],
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one command with both streams captured through a bound directory fd.

    The result-directory pathname is used only to establish and recheck the
    directory identity. Creation, provider stdout/stderr, sealing, and cleanup
    are all relative to the already-open descriptor; provider stderr is never
    inherited by a terminal, build log, or caller.
    """
    private_root = ensure_private_evidence_root(private_root)
    result_dir = _absolute_path(result_dir)
    try:
        relative = result_dir.relative_to(private_root)
    except ValueError as exc:
        raise ArtifactChainError(
            f"{label} result directory must stay under the private evidence root"
        ) from exc
    if not relative.parts:
        raise ArtifactChainError(
            f"{label} result directory must not be the private evidence root"
        )
    output_name = _safe_output_basename(output_name, label=label)
    stderr_name = _safe_output_basename(stderr_name, label=f"{label} stderr")
    if output_name == stderr_name:
        raise ArtifactChainError(f"{label} stdout/stderr outputs must be distinct")
    if not command or not Path(command[0]).is_absolute():
        raise ArtifactChainError(f"{label} command must use an absolute executable")

    directory_flags = _directory_open_flags()
    result_fd = _open_nofollow(result_dir, directory_flags)
    temporary_names = {
        "stdout": f".{output_name}.{uuid.uuid4().hex}.tmp",
        "stderr": f".{stderr_name}.{uuid.uuid4().hex}.tmp",
    }
    temporary_created: set[str] = set()
    final_created: set[str] = set()
    preserve_outputs = False
    try:
        result_metadata = os.fstat(result_fd)
        if not stat.S_ISDIR(result_metadata.st_mode):
            raise ArtifactChainError(f"{label} result directory is not a directory")
        if stat.S_IMODE(result_metadata.st_mode) != 0o700:
            raise ArtifactChainError(
                f"{label} result directory must have mode 0700"
            )
        for name in (output_name, stderr_name):
            try:
                os.stat(name, dir_fd=result_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise ArtifactChainError(
                    f"refusing to overwrite prior private {label}: "
                    f"{result_dir / name}"
                )

        temporary_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            temporary_flags |= os.O_NOFOLLOW
        stream_fds: dict[str, int] = {}
        try:
            for kind, temporary_name in temporary_names.items():
                stream_fd = os.open(
                    temporary_name,
                    temporary_flags,
                    0o600,
                    dir_fd=result_fd,
                )
                stream_fds[kind] = stream_fd
                temporary_created.add(temporary_name)
                metadata = os.fstat(stream_fd)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise ArtifactChainError(
                        f"{label} temporary {kind} capture is invalid"
                    )
            completed = subprocess.run(
                [str(value) for value in command],
                stdout=stream_fds["stdout"],
                stderr=stream_fds["stderr"],
                check=False,
                env=env,
            )
            for kind, stream_fd in stream_fds.items():
                os.fsync(stream_fd)
                metadata = os.fstat(stream_fd)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                    or (
                        kind == "stdout"
                        and completed.returncode == 0
                        and metadata.st_size <= 0
                    )
                ):
                    raise ArtifactChainError(
                        f"{label} produced an invalid {kind} capture"
                    )
                os.fchmod(stream_fd, 0o400)
                os.fsync(stream_fd)
        finally:
            for stream_fd in stream_fds.values():
                os.close(stream_fd)

        _assert_bound_directory_alias(
            result_dir,
            result_metadata,
            label=f"{label} result",
        )
        records: dict[str, dict[str, Any]] = {}
        for kind, output in (("stdout", output_name), ("stderr", stderr_name)):
            temporary = temporary_names[kind]
            _link_sealed_temporary_at(
                result_fd,
                temporary,
                output,
                label=f"{label} {kind}",
            )
            temporary_created.discard(temporary)
            final_created.add(output)
            records[kind] = _record_file_at(
                result_fd,
                output,
                label=f"{label} sealed {kind}",
                expected_mode=0o400,
                require_nonempty=(kind == "stdout" and completed.returncode == 0),
            )
            records[kind]["path"] = str(result_dir / output)
        _assert_bound_directory_alias(
            result_dir,
            result_metadata,
            label=f"{label} result",
        )
        if completed.returncode != 0:
            preserve_outputs = True
            raise ArtifactChainError(
                f"{label} failed with exit {completed.returncode}; "
                f"stderr sealed at {result_dir / stderr_name}"
            )
        return {
            **records["stdout"],
            "stderr": records["stderr"],
            "returncode": completed.returncode,
        }
    except Exception:
        if not preserve_outputs:
            for output in final_created:
                with suppress(FileNotFoundError):
                    os.unlink(output, dir_fd=result_fd)
        raise
    finally:
        for temporary_name in temporary_created:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=result_fd)
        os.close(result_fd)


def validate_apple_notary_records(
    *,
    private_root: Path,
    submit_path: Path,
    info_path: Path,
    log_path: Path,
    expected_dmg_sha256: str,
    expected_archive_name: str,
) -> dict[str, Any]:
    paths = {
        "submit": _assert_within_private_root(
            private_root,
            submit_path,
            label="Apple submit record",
        ),
        "info": _assert_within_private_root(
            private_root,
            info_path,
            label="Apple info record",
        ),
        "log": _assert_within_private_root(
            private_root,
            log_path,
            label="Apple log record",
        ),
    }
    if len({str(path) for path in paths.values()}) != 3:
        raise ArtifactChainError("Apple submit/info/log records must be distinct files")
    records: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for kind, path in paths.items():
        payloads[kind], records[kind] = _read_json_object_with_record(
            path,
            label=f"Apple {kind} record",
            require_private_mode=True,
        )
    submit = payloads["submit"]
    info = payloads["info"]
    log = payloads["log"]
    submission_id = submit.get("id")
    if not isinstance(submission_id, str):
        raise ArtifactChainError("Apple submit record has an invalid submission id")
    try:
        parsed_submission_id = uuid.UUID(submission_id)
    except ValueError as exc:
        raise ArtifactChainError("Apple submit record has an invalid submission UUID") from exc
    if str(parsed_submission_id) != submission_id.lower():
        raise ArtifactChainError("Apple submit record UUID is not canonical")
    if submit.get("status") != "Accepted":
        raise ArtifactChainError("Apple submit record is not Accepted")
    if info.get("id") != submission_id or info.get("status") != "Accepted":
        raise ArtifactChainError(
            "Apple info record does not independently confirm the Accepted submission"
        )
    if log.get("jobId") != submission_id or log.get("status") != "Accepted":
        raise ArtifactChainError(
            "Apple log record does not independently confirm the Accepted submission"
        )
    if log.get("sha256") != expected_dmg_sha256:
        raise ArtifactChainError(
            "Apple log SHA-256 does not match the immutable submitted DMG"
        )
    if log.get("archiveFilename") != expected_archive_name:
        raise ArtifactChainError(
            "Apple log archive filename does not match the immutable submitted DMG"
        )
    ticket_contents = log.get("ticketContents")
    if not isinstance(ticket_contents, list) or not ticket_contents:
        raise ArtifactChainError("Apple log record has no notarization ticket contents")
    return {
        "submission_id": submission_id,
        "status": "Accepted",
        "submitted_dmg_sha256": expected_dmg_sha256,
        "records": records,
    }


def validate_fresh_apple_notary_query(
    *,
    private_root: Path,
    info_path: Path,
    log_path: Path,
    expected_submission_id: str,
    expected_dmg_sha256: str,
    expected_archive_name: str,
    expected_team_id: str,
) -> dict[str, Any]:
    """Validate fresh pinned-notarytool info/log captures owned by final verify."""
    try:
        parsed_submission_id = uuid.UUID(expected_submission_id)
    except ValueError as exc:
        raise ArtifactChainError("expected Apple submission id is not a UUID") from exc
    canonical_id = str(parsed_submission_id)
    if canonical_id != expected_submission_id.lower():
        raise ArtifactChainError("expected Apple submission UUID is not canonical")
    if re.fullmatch(r"[A-Z0-9]{10}", expected_team_id) is None:
        raise ArtifactChainError("expected Apple team identifier is malformed")
    paths = {
        "info": _assert_within_private_root(
            private_root, info_path, label="fresh Apple info record"
        ),
        "log": _assert_within_private_root(
            private_root, log_path, label="fresh Apple log record"
        ),
    }
    if paths["info"] == paths["log"]:
        raise ArtifactChainError("fresh Apple info/log records must be distinct files")
    payloads: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    for kind, path in paths.items():
        payloads[kind], records[kind] = _read_json_object_with_record(
            path,
            label=f"fresh Apple {kind} record",
            require_private_mode=True,
        )
    info = payloads["info"]
    log = payloads["log"]
    if str(info.get("id", "")).lower() != canonical_id or info.get("status") != "Accepted":
        raise ArtifactChainError("fresh Apple info query is not the expected Accepted job")
    if (
        str(log.get("jobId", "")).lower() != canonical_id
        or log.get("status") != "Accepted"
    ):
        raise ArtifactChainError("fresh Apple log query is not the expected Accepted job")
    if log.get("sha256") != expected_dmg_sha256:
        raise ArtifactChainError("fresh Apple log digest does not match submitted snapshot")
    if log.get("archiveFilename") != expected_archive_name:
        raise ArtifactChainError("fresh Apple log archive name does not match snapshot")
    tickets = log.get("ticketContents")
    if not isinstance(tickets, list) or not tickets:
        raise ArtifactChainError("fresh Apple log has no notarization ticket contents")
    return {
        "submission_id": canonical_id,
        "status": "Accepted",
        "team_id": expected_team_id,
        "records": records,
    }


def query_apple_notary_fresh(
    *,
    private_root: Path,
    capture_dir: Path,
    submission_id: str,
    expected_dmg_sha256: str,
    expected_archive_name: str,
    expected_team_id: str,
    keychain_profile: str,
    keychain: str | None,
) -> dict[str, Any]:
    """Own a fresh Apple query; no caller-authored response file is accepted."""
    try:
        canonical_id = str(uuid.UUID(submission_id))
    except ValueError as exc:
        raise ArtifactChainError("expected Apple submission id is not a UUID") from exc
    if canonical_id != submission_id.lower():
        raise ArtifactChainError("expected Apple submission UUID is not canonical")
    if re.fullmatch(r"[0-9a-f]{64}", expected_dmg_sha256) is None:
        raise ArtifactChainError("expected submitted DMG digest is malformed")
    if (
        not expected_archive_name
        or expected_archive_name != Path(expected_archive_name).name
    ):
        raise ArtifactChainError("expected submitted archive name is invalid")
    if re.fullmatch(r"[A-Z0-9]{10}", expected_team_id) is None:
        raise ArtifactChainError("expected Apple team identifier is malformed")
    if not keychain_profile.strip():
        raise ArtifactChainError("Apple notary keychain profile is empty")
    private_root = ensure_private_evidence_root(private_root)
    capture_dir = create_private_directory(
        private_root=private_root,
        directory=capture_dir,
        label="fresh Apple query directory",
    )
    base = [
        "/usr/bin/xcrun",
        "notarytool",
    ]
    credentials: list[str] = []
    if keychain:
        credentials.extend(["--keychain", keychain])
    credentials.extend(["--keychain-profile", keychain_profile])
    outputs: dict[str, Path] = {}
    sanitized_env = dict(os.environ)
    sanitized_env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
    for kind in ("info", "log"):
        command = [
            *base,
            kind,
            submission_id,
            *credentials,
            "--output-format",
            "json",
        ]
        record = capture_private_command(
            private_root=private_root,
            result_dir=capture_dir,
            output_name=f"{kind}.json",
            stderr_name=f"{kind}.stderr.log",
            label=f"fresh Apple {kind} query",
            command=command,
            env=sanitized_env,
        )
        outputs[kind] = Path(str(record["path"]))
    result = validate_fresh_apple_notary_query(
        private_root=private_root,
        info_path=outputs["info"],
        log_path=outputs["log"],
        expected_submission_id=submission_id,
        expected_dmg_sha256=expected_dmg_sha256,
        expected_archive_name=expected_archive_name,
        expected_team_id=expected_team_id,
    )
    result["command"] = "/usr/bin/xcrun notarytool info/log"
    return result


def _copy_immutable_file(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    label: str,
    mode: int = 0o400,
) -> dict[str, Any]:
    source = _absolute_path(source)
    destination = _absolute_path(destination)
    source_record = _safe_regular_file(source, label=label)
    if source_record["sha256"] != expected_sha256:
        raise ArtifactChainError(f"{label} does not match its build handoff digest")
    _reject_symlinked_ancestors(destination.parent, label=f"{label} snapshot directory")
    if destination.exists() or destination.is_symlink():
        raise ArtifactChainError(f"immutable snapshot already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _reject_symlinked_ancestors(destination.parent, label=f"{label} snapshot directory")

    source_fd = _open_nofollow(source, os.O_RDONLY)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    destination_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    digest = hashlib.sha256()
    try:
        source_before = os.fstat(source_fd)
        if source_before.st_nlink != 1:
            raise ArtifactChainError(f"{label} became hard-linked during snapshot")
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            remaining = memoryview(chunk)
            while remaining:
                written = os.write(destination_fd, remaining)
                if written <= 0:
                    raise ArtifactChainError(
                        f"short write while copying immutable {label} snapshot"
                    )
                remaining = remaining[written:]
        os.fsync(destination_fd)
        source_after = os.fstat(source_fd)
        if (
            source_before.st_dev,
            source_before.st_ino,
            source_before.st_size,
            source_before.st_mtime_ns,
            source_before.st_nlink,
        ) != (
            source_after.st_dev,
            source_after.st_ino,
            source_after.st_size,
            source_after.st_mtime_ns,
            source_after.st_nlink,
        ):
            raise ArtifactChainError(f"{label} changed during immutable snapshot")
        if digest.hexdigest() != expected_sha256:
            raise ArtifactChainError(
                f"{label} changed while its immutable snapshot was copied"
            )
        os.fchmod(destination_fd, mode)
    except BaseException:
        with suppress(OSError):
            os.close(source_fd)
        with suppress(OSError):
            os.close(destination_fd)
        with suppress(FileNotFoundError):
            temporary.unlink()
        raise
    else:
        os.close(source_fd)
        os.close(destination_fd)
    try:
        os.link(temporary, destination, follow_symlinks=False)
    except FileExistsError as exc:
        raise ArtifactChainError(
            f"concurrent immutable snapshot already exists: {destination}"
        ) from exc
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    copied = _safe_regular_file(destination, label=f"immutable {label} snapshot")
    _require_private_mode(
        destination,
        label=f"immutable {label} snapshot",
        expected=mode,
        record=copied,
    )
    if copied["sha256"] != expected_sha256:
        raise ArtifactChainError(f"immutable {label} snapshot hash mismatch")
    return copied


def create_private_operation_copy(
    *,
    private_root: Path,
    source: Path,
    destination: Path,
    expected_sha256: str,
    writable: bool,
    label: str,
) -> dict[str, Any]:
    private_root = ensure_private_evidence_root(private_root)
    destination = _assert_within_private_root(
        private_root,
        destination,
        label=f"{label} private operation copy",
    )
    return _copy_immutable_file(
        source,
        destination,
        expected_sha256=expected_sha256,
        label=label,
        mode=0o600 if writable else 0o400,
    )


def install_private_operation_result(
    *,
    private_root: Path,
    source: Path,
    destination: Path,
    expected_source_sha256: str,
    expected_destination_sha256: str | None,
    label: str,
) -> dict[str, Any]:
    """Atomically install a verified private result over one expected public file."""
    private_root = ensure_private_evidence_root(private_root)
    source = _assert_within_private_root(
        private_root,
        source,
        label=f"{label} private install source",
    )
    source_record = _safe_regular_file(
        source,
        label=f"{label} private install source",
    )
    if source_record["sha256"] != expected_source_sha256:
        raise ArtifactChainError(f"{label} private result digest changed")
    destination = _absolute_path(destination)
    _reject_symlinked_ancestors(destination.parent, label=f"{label} destination")
    if expected_destination_sha256 is not None:
        current = _safe_regular_file(destination, label=f"{label} current destination")
        if current["sha256"] != expected_destination_sha256:
            raise ArtifactChainError(f"{label} destination changed before install")
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.install"
    )
    source_fd = _open_nofollow(source, os.O_RDONLY)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    digest = hashlib.sha256()
    try:
        source_before = os.fstat(source_fd)
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            remaining = memoryview(chunk)
            while remaining:
                written = os.write(fd, remaining)
                if written <= 0:
                    raise ArtifactChainError(
                        f"short write while installing {label}"
                    )
                remaining = remaining[written:]
        os.fsync(fd)
        source_after = os.fstat(source_fd)
        if (
            source_before.st_dev,
            source_before.st_ino,
            source_before.st_size,
            source_before.st_mtime_ns,
            source_before.st_nlink,
            source_before.st_mode,
        ) != (
            source_after.st_dev,
            source_after.st_ino,
            source_after.st_size,
            source_after.st_mtime_ns,
            source_after.st_nlink,
            source_after.st_mode,
        ):
            raise ArtifactChainError(f"{label} private result changed during install")
        if digest.hexdigest() != expected_source_sha256:
            raise ArtifactChainError(f"{label} private result changed during install")
        os.close(fd)
        fd = -1
        os.close(source_fd)
        source_fd = -1
        os.replace(temporary, destination)
    finally:
        if fd >= 0:
            os.close(fd)
        if source_fd >= 0:
            os.close(source_fd)
        with suppress(FileNotFoundError):
            temporary.unlink()
    installed = _safe_regular_file(destination, label=f"{label} installed destination")
    if installed["sha256"] != expected_source_sha256:
        raise ArtifactChainError(f"{label} installed digest mismatch")
    return installed


def validate_recomputed_blockmap(
    *,
    expected_blockmap: Path,
    recomputed_blockmap: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    expected = _safe_regular_file(
        expected_blockmap,
        label="final release blockmap",
    )
    recomputed = _safe_regular_file(
        recomputed_blockmap,
        label="independently recomputed blockmap",
    )
    if expected["sha256"] != expected_sha256:
        raise ArtifactChainError("final blockmap differs from final manifest")
    if recomputed["sha256"] != expected_sha256:
        raise ArtifactChainError("final blockmap is unrelated to its bound DMG")
    if recomputed["size"] != expected["size"]:
        raise ArtifactChainError("recomputed blockmap size differs from final blockmap")
    return {
        "sha256": expected_sha256,
        "size": expected["size"],
        "expected_path": expected["path"],
        "recomputed_path": recomputed["path"],
    }


def validate_file_identity(
    *,
    path: Path,
    expected_sha256: str,
    expected_device: int | None = None,
    expected_inode: int | None = None,
    expected_size: int | None = None,
    label: str = "release artifact",
) -> dict[str, Any]:
    record = _safe_regular_file(path, label=label)
    if record["sha256"] != expected_sha256:
        raise ArtifactChainError(
            f"{label} digest mismatch: expected {expected_sha256}, "
            f"found {record['sha256']}"
        )
    expected_identity = (expected_device, expected_inode, expected_size)
    actual_identity = (record["device"], record["inode"], record["size"])
    for expected, actual, field in zip(
        expected_identity,
        actual_identity,
        ("device", "inode", "size"),
        strict=True,
    ):
        if expected is not None and expected != actual:
            raise ArtifactChainError(
                f"{label} {field} changed: expected {expected}, found {actual}"
            )
    return record


def create_pre_notary_snapshots(
    *,
    root: Path,
    dist_dir: Path,
    version: str,
    private_root: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
    expected_source_commit: str,
    expected_source_tree: str,
    expected_preflight_sha256: str,
    snapshot_dir: Path,
) -> dict[str, Any]:
    pre = validate_pre_notary_artifact_manifest(
        root=root,
        dist_dir=dist_dir,
        version=version,
        private_root=private_root,
        manifest_path=manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_source_commit=expected_source_commit,
        expected_source_tree=expected_source_tree,
        expected_preflight_sha256=expected_preflight_sha256,
    )
    private_root = ensure_private_evidence_root(private_root)
    snapshot_dir = _assert_within_private_root(
        private_root,
        snapshot_dir,
        label="immutable notary snapshot directory",
    )
    if snapshot_dir.exists() or snapshot_dir.is_symlink():
        raise ArtifactChainError(
            f"immutable notary snapshot directory already exists: {snapshot_dir}"
        )
    snapshot_dir.mkdir(parents=True, mode=0o700)
    snapshots: dict[str, dict[str, Any]] = {}
    for flavor in R20_ARTIFACT_CHAIN_FLAVORS:
        record = pre["payload"]["artifacts"][flavor]
        dmg_destination = snapshot_dir / Path(record["dmg_path"]).name
        blockmap_destination = snapshot_dir / Path(record["blockmap_path"]).name
        dmg = _copy_immutable_file(
            Path(record["dmg_path"]),
            dmg_destination,
            expected_sha256=record["dmg_sha256"],
            label=f"{flavor} DMG",
        )
        blockmap = _copy_immutable_file(
            Path(record["blockmap_path"]),
            blockmap_destination,
            expected_sha256=record["blockmap_sha256"],
            label=f"{flavor} blockmap",
        )
        snapshots[flavor] = {
            "dmg_path": dmg["path"],
            "dmg_sha256": dmg["sha256"],
            "dmg_size": dmg["size"],
            "dmg_device": dmg["device"],
            "dmg_inode": dmg["inode"],
            "blockmap_path": blockmap["path"],
            "blockmap_sha256": blockmap["sha256"],
            "blockmap_size": blockmap["size"],
            "blockmap_device": blockmap["device"],
            "blockmap_inode": blockmap["inode"],
        }
    snapshot_manifest_path = snapshot_dir / "snapshot-manifest.json"
    payload = {
        "schema_version": R20_ARTIFACT_CHAIN_SCHEMA_VERSION,
        "source": pre["payload"]["source"],
        "pre_notary_manifest_sha256": pre["sha256"],
        "snapshots": snapshots,
    }
    digest = _write_private_json_with_digest(snapshot_manifest_path, payload)
    return {
        "snapshot_dir": str(snapshot_dir),
        "manifest": str(snapshot_manifest_path),
        "sha256": digest,
        "snapshots": snapshots,
    }


def _expected_runtime_contract(flavor: str) -> dict[str, str]:
    try:
        return dict(R20_FLAVOR_RUNTIME_CONTRACTS[flavor])
    except KeyError as exc:
        raise ArtifactChainError(f"unsupported runtime flavor: {flavor}") from exc


def _wheel_distribution_record(
    site_packages: Path,
    *,
    distribution: str,
    expected_platform: str,
) -> dict[str, Any]:
    pattern = "mlx-*.dist-info" if distribution == "mlx" else "mlx_metal-*.dist-info"
    candidates = sorted(
        path
        for path in site_packages.glob(pattern)
        if path.is_dir() and not path.is_symlink()
    )
    if len(candidates) != 1:
        raise ArtifactChainError(
            f"bundled runtime must contain exactly one {distribution} dist-info; "
            f"found {[path.name for path in candidates]}"
        )
    dist_info = candidates[0]
    metadata_record, metadata_bytes = _read_regular_file(
        dist_info / "METADATA",
        label=f"bundled {distribution} METADATA",
        capture_bytes=True,
    )
    wheel_record, wheel_bytes = _read_regular_file(
        dist_info / "WHEEL",
        label=f"bundled {distribution} WHEEL",
        capture_bytes=True,
    )
    assert metadata_bytes is not None
    assert wheel_bytes is not None
    metadata = email.parser.BytesParser(policy=email.policy.default).parsebytes(
        metadata_bytes
    )
    canonical_name = str(metadata.get("Name", "")).strip().lower().replace("_", "-")
    version = str(metadata.get("Version", "")).strip()
    if canonical_name != distribution or not version:
        raise ArtifactChainError(
            f"bundled {distribution} METADATA identity/version is invalid"
        )
    tags = sorted(
        {
            line.split(":", 1)[1].strip()
            for line in wheel_bytes.decode("utf-8").splitlines()
            if line.startswith("Tag:")
        }
    )
    if (
        not tags
        or any(tag.count("-") < 2 for tag in tags)
        or {tag.rsplit("-", 1)[-1] for tag in tags} != {expected_platform}
    ):
        raise ArtifactChainError(
            f"bundled {distribution} wheel tags do not exactly target "
            f"{expected_platform}: {tags}"
        )
    return {
        "distribution": distribution,
        "version": version,
        "dist_info": dist_info.name,
        "tags": tags,
        "metadata_sha256": metadata_record["sha256"],
        "wheel_sha256": wheel_record["sha256"],
    }


def inspect_bundle_runtime_contract(
    *,
    bundle_root: Path,
    flavor: str,
    version: str,
) -> dict[str, Any]:
    expected = _expected_runtime_contract(flavor)
    bundle_root = _absolute_path(bundle_root)
    _reject_symlinked_ancestors(bundle_root, label=f"{flavor} bundled runtime")
    if bundle_root.is_symlink() or not bundle_root.is_dir():
        raise ArtifactChainError(f"{flavor} bundled runtime is missing or invalid")
    python_lib = bundle_root / "python/lib"
    site_candidates = sorted(
        path / "site-packages"
        for path in python_lib.glob("python*")
        if (path / "site-packages").is_dir()
        and not (path / "site-packages").is_symlink()
    )
    if len(site_candidates) != 1:
        raise ArtifactChainError(
            f"{flavor} bundled runtime must contain exactly one site-packages"
        )
    site_packages = site_candidates[0]
    provenance_path = bundle_root / "vmlx-bundle-provenance.json"
    provenance, provenance_record = _read_json_object_with_record(
        provenance_path,
        label=f"{flavor} vMLX bundle provenance",
    )
    _require_exact_dict_keys(
        provenance,
        {"schema_version", "vmlx", "jang", "mlx_wheel_platform"},
        label=f"{flavor} vMLX bundle provenance",
    )
    vmlx = _require_exact_dict_keys(
        provenance["vmlx"],
        {"commit", "version"},
        label=f"{flavor} vMLX bundle provenance source",
    )
    jang_fields = {"commit", "version"}
    if isinstance(provenance["jang"], dict) and "release_pin" in provenance["jang"]:
        jang_fields.add("release_pin")
    jang = _require_exact_dict_keys(
        provenance["jang"],
        jang_fields,
        label=f"{flavor} JANG bundle provenance source",
    )
    if "release_pin" in jang:
        pin = jang["release_pin"]
        if (
            not isinstance(pin, str)
            or re.fullmatch(r"[0-9a-f]{40}", pin) is None
            or pin != jang["commit"]
        ):
            raise ArtifactChainError(
                f"{flavor} JANG release pin must match the exact source commit"
            )
    if (
        provenance["schema_version"] != 1
        or vmlx["version"] != version
        or provenance["mlx_wheel_platform"] != expected["mlx_wheel_platform"]
    ):
        raise ArtifactChainError(
            f"{flavor} bundle provenance does not declare the exact runtime contract"
        )
    distributions = {
        distribution: _wheel_distribution_record(
            site_packages,
            distribution=distribution,
            expected_platform=expected["mlx_wheel_platform"],
        )
        for distribution in ("mlx", "mlx-metal")
    }
    versions = {
        record["version"]
        for record in distributions.values()
    }
    if len(versions) != 1:
        raise ArtifactChainError(
            f"{flavor} mlx and mlx-metal versions differ: {sorted(versions)}"
        )
    return {
        "flavor": flavor,
        "mlx_wheel_platform": expected["mlx_wheel_platform"],
        "minimum_system_version": expected["minimum_system_version"],
        "mlx_version": versions.pop(),
        "bundle_provenance_sha256": provenance_record["sha256"],
        "bundle_provenance": provenance,
        "distributions": distributions,
    }


def _inspect_app_runtime_contract(
    *,
    app: Path,
    flavor: str,
    version: str,
) -> dict[str, Any]:
    app = _absolute_path(app)
    info_record, info_bytes = _read_regular_file(
        app / "Contents/Info.plist",
        label=f"{flavor} app Info.plist",
        capture_bytes=True,
    )
    assert info_bytes is not None
    info = plistlib.loads(info_bytes)
    expected = _expected_runtime_contract(flavor)
    if (
        info.get("CFBundleIdentifier") != "net.vmlx.app"
        or info.get("CFBundleShortVersionString") != version
        or info.get("CFBundleVersion") != version
        or info.get("LSMinimumSystemVersion")
        != expected["minimum_system_version"]
    ):
        raise ArtifactChainError(
            f"{flavor} app identity/version/minimum-system contract is not exact"
        )
    runtime = inspect_bundle_runtime_contract(
        bundle_root=app / "Contents/Resources/bundled-python",
        flavor=flavor,
        version=version,
    )
    return {
        **runtime,
        "info_plist_sha256": info_record["sha256"],
    }


def write_installed_release_manifest(
    *,
    root: Path,
    app: Path,
    dist_dir: Path,
    version: str,
    flavor: str,
    private_root: Path,
    output_path: Path,
    final_manifest_path: Path,
    expected_final_manifest_sha256: str,
    expected_pre_manifest_sha256: str,
    expected_source_commit: str,
    expected_source_tree: str,
    expected_preflight_sha256: str,
    extracted_asar: Path,
    producer_executable: Path | None = None,
) -> dict[str, Any]:
    """Write the external manifest consumed by installed UI/API proof.

    The manifest intentionally contains only stable source and installed-byte
    identities.  Its authority comes from the existing sealed post-notary
    chain and mounted app/ASAR parity validator; there is no standalone weaker
    app self-attestation path.
    """
    if version != R20_ARTIFACT_CHAIN_VERSION:
        raise ArtifactChainError(
            f"r20 installed manifest requires version {R20_ARTIFACT_CHAIN_VERSION}"
        )
    if flavor not in R20_ARTIFACT_CHAIN_FLAVORS:
        raise ArtifactChainError(f"unsupported installed app flavor: {flavor}")

    private_root = ensure_private_evidence_root(private_root)
    output_path = _assert_within_private_root(
        private_root,
        output_path,
        label="installed release manifest",
    )
    app = _absolute_path(app)
    extracted_asar = _absolute_path(extracted_asar)
    _reject_symlinked_ancestors(app, label="installed vMLX.app")
    if app.name != "vMLX.app" or app.is_symlink() or not app.is_dir():
        raise ArtifactChainError(
            "installed application must be a real directory named vMLX.app"
        )
    mounted = validate_mounted_app_against_final_manifest(
        root=root,
        dist_dir=dist_dir,
        version=version,
        private_root=private_root,
        final_manifest_path=final_manifest_path,
        expected_final_manifest_sha256=expected_final_manifest_sha256,
        expected_pre_manifest_sha256=expected_pre_manifest_sha256,
        expected_source_commit=expected_source_commit,
        expected_source_tree=expected_source_tree,
        expected_preflight_sha256=expected_preflight_sha256,
        flavor=flavor,
        mounted_app=app,
        extracted_asar=extracted_asar,
    )
    source = _git_source_identity(root)
    resources = app / "Contents/Resources"
    app_asar_path = resources / "app.asar"
    electron_path = app / "Contents/MacOS/vMLX"
    provenance_path = resources / "bundled-python/vmlx-bundle-provenance.json"
    invoked_python_path = app / INSTALLED_BUNDLED_PYTHON_RELATIVE_PATH

    def observe_bound_artifacts() -> dict[str, Any]:
        try:
            invoked_python_metadata = invoked_python_path.lstat()
            resolved_python_path = invoked_python_path.resolve(strict=True)
            resolved_python_path.relative_to(app.resolve(strict=True))
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
            raise ArtifactChainError(
                "installed bundled Python does not resolve inside vMLX.app"
            ) from exc
        if not (
            stat.S_ISREG(invoked_python_metadata.st_mode)
            or stat.S_ISLNK(invoked_python_metadata.st_mode)
        ):
            raise ArtifactChainError(
                "installed bundled Python invocation is not a file or symlink"
            )
        electron = _safe_regular_file(
            electron_path,
            label="installed Electron executable",
        )
        bundled_python = _safe_regular_file(
            resolved_python_path,
            label="installed bundled Python executable",
        )
        if not os.access(electron_path, os.X_OK):
            raise ArtifactChainError("installed Electron executable is not executable")
        if not os.access(resolved_python_path, os.X_OK):
            raise ArtifactChainError("installed bundled Python is not executable")
        return {
            "app_asar": _safe_regular_file(
                app_asar_path,
                label="installed app.asar",
            ),
            "electron": electron,
            "provenance": _safe_regular_file(
                provenance_path,
                label="installed bundled provenance",
            ),
            "bundled_python": bundled_python,
            "python_invocation_lstat": {
                "device": invoked_python_metadata.st_dev,
                "inode": invoked_python_metadata.st_ino,
                "mode": invoked_python_metadata.st_mode,
                "mtime_ns": invoked_python_metadata.st_mtime_ns,
                "size": invoked_python_metadata.st_size,
            },
            "resolved_python_path": str(resolved_python_path),
        }

    first = observe_bound_artifacts()
    if (
        first["provenance"]["sha256"]
        != mounted["runtime_contract"]["bundle_provenance_sha256"]
    ):
        raise ArtifactChainError(
            "installed bundled provenance differs from mounted release attestation"
        )
    if producer_executable is None:
        producer_executable = Path(sys.executable)
    else:
        producer_executable = Path(producer_executable)
    try:
        resolved_producer_executable = producer_executable.resolve(strict=True)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        raise ArtifactChainError(
            "installed manifest producer executable is unavailable"
        ) from exc
    if str(resolved_producer_executable) != first["resolved_python_path"]:
        raise ArtifactChainError(
            "installed manifest producer must run under the exact bundled Python"
        )

    first_app_payload = _tree_payload_records(
        app,
        label="installed release-manifest application payload",
    )
    first_asar_payload = _tree_payload_records(
        extracted_asar,
        label="installed release-manifest extracted ASAR payload",
        ignore_ambient_finder_metadata=True,
    )
    if (
        first_app_payload["tree_sha256"] != mounted["app_tree_sha256"]
        or first_asar_payload["tree_sha256"] != mounted["asar_tree_sha256"]
    ):
        raise ArtifactChainError(
            "installed app/ASAR payload changed after mounted release validation"
        )

    second_app_payload = _tree_payload_records(
        app,
        label="installed release-manifest application payload recheck",
    )
    second_asar_payload = _tree_payload_records(
        extracted_asar,
        label="installed release-manifest extracted ASAR payload recheck",
        ignore_ambient_finder_metadata=True,
    )
    second = observe_bound_artifacts()
    if (
        second_app_payload != first_app_payload
        or second_asar_payload != first_asar_payload
        or second != first
        or second["provenance"]["sha256"]
        != mounted["runtime_contract"]["bundle_provenance_sha256"]
    ):
        raise ArtifactChainError(
            "installed app/ASAR/provenance changed during manifest production"
        )

    # The final complete payload observation must come after the last
    # core-artifact observation.  Otherwise a non-core app or extracted-ASAR
    # mutation in that interval would not affect any field serialized below.
    final_app_payload = _tree_payload_records(
        app,
        label="installed release-manifest final application payload",
    )
    final_asar_payload = _tree_payload_records(
        extracted_asar,
        label="installed release-manifest final extracted ASAR payload",
        ignore_ambient_finder_metadata=True,
    )
    if (
        final_app_payload != first_app_payload
        or final_asar_payload != first_asar_payload
    ):
        raise ArtifactChainError(
            "installed app/ASAR payload changed before manifest serialization"
        )

    payload = {
        "schema": INSTALLED_RELEASE_MANIFEST_SCHEMA,
        "source_commit": source["commit"],
        "source_tree": source["tree"],
        "app_asar_sha256": second["app_asar"]["sha256"],
        "electron_executable_sha256": second["electron"]["sha256"],
        "bundled_provenance_sha256": second["provenance"]["sha256"],
        "bundled_python_executable_sha256": second["bundled_python"]["sha256"],
        "bundled_python_executable_fingerprint_sha256": hashlib.sha256(
            second["resolved_python_path"].encode("utf-8")
        ).hexdigest(),
    }
    if set(payload) != INSTALLED_RELEASE_MANIFEST_FIELDS:
        raise ArtifactChainError("installed release manifest fields are not exact")
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    _exclusive_sealed_write(output_path, encoded, mode=0o600)
    return {
        "manifest": str(output_path),
        "sha256": digest,
        "payload": payload,
    }


def write_bundle_runtime_attestation(
    *,
    root: Path,
    bundle_root: Path,
    version: str,
    private_root: Path,
    flavor: str,
    output_path: Path,
) -> dict[str, Any]:
    if version != R20_ARTIFACT_CHAIN_VERSION:
        raise ArtifactChainError(
            f"r20 artifact chain requires version {R20_ARTIFACT_CHAIN_VERSION}"
        )
    private_root = ensure_private_evidence_root(private_root)
    output_path = _assert_within_private_root(
        private_root,
        output_path,
        label=f"{flavor} bundle runtime attestation",
    )
    source = _git_source_identity(root)
    runtime = inspect_bundle_runtime_contract(
        bundle_root=bundle_root,
        flavor=flavor,
        version=version,
    )
    if runtime["bundle_provenance"]["vmlx"]["commit"] != source["commit"]:
        raise ArtifactChainError(
            f"{flavor} bundle provenance commit differs from clean Git"
        )
    payload = {
        "schema_version": 1,
        "scope": R20_ARTIFACT_CHAIN_SCOPE,
        "stage": "bundle_runtime",
        "version": version,
        "flavor": flavor,
        "source": source,
        "runtime_contract": runtime,
    }
    digest = _write_private_json_with_digest(output_path, payload)
    return {"attestation": str(output_path), "sha256": digest, "payload": payload}


def _validate_bundle_runtime_attestation(
    *,
    root: Path,
    version: str,
    private_root: Path,
    flavor: str,
    attestation_path: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    path = _assert_within_private_root(
        private_root,
        attestation_path,
        label=f"{flavor} bundle runtime attestation",
    )
    payload, record = _read_json_object_with_record(
        path,
        label=f"{flavor} bundle runtime attestation",
        expected_sha256=expected_sha256,
        require_private_mode=True,
    )
    _require_exact_dict_keys(
        payload,
        {
            "schema_version",
            "scope",
            "stage",
            "version",
            "flavor",
            "source",
            "runtime_contract",
        },
        label=f"{flavor} bundle runtime attestation",
    )
    if (
        payload["schema_version"] != 1
        or payload["scope"] != R20_ARTIFACT_CHAIN_SCOPE
        or payload["stage"] != "bundle_runtime"
        or payload["version"] != version
        or payload["flavor"] != flavor
        or payload["source"] != _git_source_identity(root)
        or payload["runtime_contract"]["flavor"] != flavor
        or payload["runtime_contract"]["mlx_wheel_platform"]
        != _expected_runtime_contract(flavor)["mlx_wheel_platform"]
        or payload["runtime_contract"]["minimum_system_version"]
        != _expected_runtime_contract(flavor)["minimum_system_version"]
    ):
        raise ArtifactChainError(f"{flavor} bundle runtime attestation is invalid")
    return {"path": str(path), "sha256": record["sha256"], "payload": payload}


def find_exact_staged_app(staged_output: Path) -> Path:
    staged_output = _absolute_path(staged_output)
    _reject_symlinked_ancestors(staged_output, label="staged app output")
    if staged_output.is_symlink() or not staged_output.is_dir():
        raise ArtifactChainError(f"staged app output is invalid: {staged_output}")
    mac_output = staged_output / "mac-arm64"
    _reject_symlinked_ancestors(mac_output, label="staged mac-arm64 output")
    if mac_output.is_symlink() or not mac_output.is_dir():
        raise ArtifactChainError("staged output is missing the exact mac-arm64 directory")
    expected = mac_output / "vMLX.app"
    candidates: list[Path] = []
    for parent in (staged_output, *(
        path
        for path in staged_output.iterdir()
        if path.is_dir() and not path.is_symlink()
    )):
        for candidate in parent.iterdir():
            if candidate.name.endswith(".app"):
                candidates.append(candidate)
    candidates = sorted(set(candidates))
    if candidates != [expected]:
        raise ArtifactChainError(
            "staged output must contain exactly one application at the top level, "
            "with the canonical application at "
            f"{expected}; found {[str(path) for path in candidates]}"
        )
    _reject_symlinked_ancestors(expected, label="staged vMLX.app")
    if expected.is_symlink() or not expected.is_dir():
        raise ArtifactChainError(f"staged application must be a real directory: {expected}")
    return expected


def _tree_file_records(
    root: Path,
    *,
    label: str,
    exclude_python_bytecode: bool = False,
) -> dict[str, dict[str, Any]]:
    root = _absolute_path(root)
    _reject_symlinked_ancestors(root, label=label)
    if root.is_symlink() or not root.is_dir():
        raise ArtifactChainError(f"{label} is missing or invalid: {root}")
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if exclude_python_bytecode and (
            "__pycache__" in relative.parts or path.suffix == ".pyc"
        ):
            continue
        if path.is_symlink():
            raise ArtifactChainError(f"{label} contains a symlink: {relative}")
        if path.is_dir():
            continue
        # Empty tracked files (notably package __init__.py markers) are valid
        # tree members. Tree parity records their size and SHA-256, so allowing
        # them does not weaken equality or substitution checks.
        record = _safe_regular_file(
            path,
            label=f"{label}/{relative}",
            require_nonempty=False,
        )
        records[relative.as_posix()] = {
            "sha256": record["sha256"],
            "size": record["size"],
            "mode": record["mode"],
        }
    if not records:
        raise ArtifactChainError(f"{label} has no files")
    return records


def _require_tree_parity(
    source: Path,
    packaged: Path,
    *,
    label: str,
    exclude_python_bytecode: bool = False,
    source_only_files: tuple[str, ...] = (),
) -> dict[str, Any]:
    source_records = _tree_file_records(
        source,
        label=f"{label} source",
        exclude_python_bytecode=exclude_python_bytecode,
    )
    packaged_records = _tree_file_records(
        packaged,
        label=f"{label} packaged",
        exclude_python_bytecode=exclude_python_bytecode,
    )
    packaged_source_only = sorted(set(source_only_files) & set(packaged_records))
    if packaged_source_only:
        raise ArtifactChainError(
            f"{label} contains source-only files: {packaged_source_only}"
        )
    for relative in source_only_files:
        source_records.pop(relative, None)
    if source_records != packaged_records:
        source_names = set(source_records)
        packaged_names = set(packaged_records)
        mismatched = sorted(
            name
            for name in source_names & packaged_names
            if source_records[name] != packaged_records[name]
        )
        raise ArtifactChainError(
            f"{label} tree differs from the release source; "
            f"missing={sorted(source_names - packaged_names)[:10]}, "
            f"extra={sorted(packaged_names - source_names)[:10]}, "
            f"mismatched={mismatched[:10]}"
        )
    digest = hashlib.sha256(
        json.dumps(source_records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"file_count": len(source_records), "tree_sha256": digest}


def _require_runtime_tree_parity(
    source: Path,
    packaged: Path,
    *,
    label: str,
) -> dict[str, Any]:
    """Require installed-runtime parity under the single source-only contract."""
    return _require_tree_parity(
        source,
        packaged,
        label=label,
        exclude_python_bytecode=True,
        source_only_files=R20_RUNTIME_SOURCE_ONLY_FILES,
    )


def _tree_payload_records(
    root: Path,
    *,
    label: str,
    ignore_ambient_finder_metadata: bool = False,
) -> dict[str, Any]:
    """Record a complete app/ASAR payload, including modes and symlink targets."""
    root = _absolute_path(root)
    _reject_symlinked_ancestors(root, label=label)
    if root.is_symlink() or not root.is_dir():
        raise ArtifactChainError(f"{label} is missing or invalid: {root}")
    root_mode = stat.S_IMODE(root.stat().st_mode)
    entries: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if ignore_ambient_finder_metadata and path.name == ".DS_Store":
            continue
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISLNK(metadata.st_mode):
            entries[relative] = {
                "kind": "symlink",
                "target": os.readlink(path),
                "mode": mode,
            }
        elif stat.S_ISDIR(metadata.st_mode):
            entries[relative] = {"kind": "directory", "mode": mode}
        elif stat.S_ISREG(metadata.st_mode):
            record = _safe_regular_file(
                path,
                label=f"{label}/{relative}",
                require_nonempty=False,
            )
            entries[relative] = {
                "kind": "file",
                "sha256": record["sha256"],
                "size": record["size"],
                "mode": record["mode"],
            }
        else:
            raise ArtifactChainError(f"{label} has unsupported entry: {relative}")
    if not entries:
        raise ArtifactChainError(f"{label} has no payload entries")
    encoded = json.dumps(
        {"root_mode": root_mode, "entries": entries},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "root_mode": root_mode,
        "entry_count": len(entries),
        "tree_sha256": hashlib.sha256(encoded).hexdigest(),
        "entries": entries,
    }


def validate_staged_app_parity(
    *,
    root: Path,
    staged_output: Path,
    extracted_asar: Path,
    version: str,
    flavor: str,
) -> dict[str, Any]:
    root = _absolute_path(root)
    source = _git_source_identity(root)
    app = find_exact_staged_app(staged_output)
    resources = app / "Contents/Resources"
    runtime_contract = _inspect_app_runtime_contract(
        app=app,
        flavor=flavor,
        version=version,
    )
    if runtime_contract["bundle_provenance"]["vmlx"]["commit"] != source["commit"]:
        raise ArtifactChainError(
            f"{flavor} staged bundle provenance commit differs from clean Git"
        )

    source_engine = root / "vmlx_engine"
    mirror_engine = resources / "vmlx-engine-source/vmlx_engine"
    bundled_candidates = list(
        (resources / "bundled-python/python/lib").glob(
            "python*/site-packages/vmlx_engine"
        )
    )
    if len(bundled_candidates) != 1:
        raise ArtifactChainError(
            "staged app must contain exactly one bundled vmlx_engine package, "
            f"found {len(bundled_candidates)}"
        )
    mirror_parity = _require_tree_parity(
        source_engine,
        mirror_engine,
        label="staged source-mirror vmlx_engine",
        exclude_python_bytecode=True,
    )
    runtime_parity = _require_runtime_tree_parity(
        source_engine,
        bundled_candidates[0],
        label="staged bundled-runtime vmlx_engine",
    )
    renderer_parity = _require_tree_parity(
        root / "panel/out",
        _absolute_path(extracted_asar) / "out",
        label="staged Electron renderer/main output",
    )
    source_package = _safe_regular_file(
        root / "panel/package.json",
        label="release package.json",
    )
    asar_package = _safe_regular_file(
        _absolute_path(extracted_asar) / "package.json",
        label="staged app.asar package.json",
    )
    source_package_json, source_package_bound = _read_json_object_with_record(
        Path(source_package["path"]),
        label="release package.json",
    )
    asar_package_json, asar_package_bound = _read_json_object_with_record(
        Path(asar_package["path"]),
        label="staged app.asar package.json",
    )
    for field in ("name", "version", "main", "type"):
        if source_package_json.get(field) != asar_package_json.get(field):
            raise ArtifactChainError(
                f"staged app.asar package.json {field} differs from release source"
            )
    return {
        "app": str(app),
        "source": source,
        "version": version,
        "flavor": flavor,
        "runtime_contract": runtime_contract,
        "source_mirror": mirror_parity,
        "bundled_runtime": runtime_parity,
        "renderer": renderer_parity,
        "source_package_json_sha256": source_package_bound["sha256"],
        "asar_package_json_sha256": asar_package_bound["sha256"],
        "app_payload": _tree_payload_records(app, label="staged application payload"),
        "asar_payload": _tree_payload_records(
            _absolute_path(extracted_asar),
            label="staged extracted ASAR payload",
            ignore_ambient_finder_metadata=True,
        ),
    }


def _validate_pinned_toolchain(
    value: Any,
    *,
    label: str,
    require_current: bool = True,
) -> dict[str, Any]:
    tools = _require_exact_dict_keys(
        value,
        set(R20_PINNED_TOOL_NAMES),
        label=label,
    )
    for name, raw_tool in tools.items():
        tool = _require_exact_dict_keys(
            raw_tool,
            {"path", "realpath", "sha256"},
            label=f"{label} {name}",
        )
        if (
            re.fullmatch(r"[0-9a-f]{64}", str(tool["sha256"])) is None
            or os.path.realpath(str(tool["path"])) != str(tool["realpath"])
        ):
            raise ArtifactChainError(f"{label} {name} identity is invalid")
        if require_current:
            current, _ = _read_regular_file(
                Path(str(tool["realpath"])),
                label=f"{label} pinned {name}",
                require_single_link=False,
            )
            if current["sha256"] != tool["sha256"] or not (
                current["mode"] & 0o111
            ):
                raise ArtifactChainError(f"{label} pinned {name} changed")
    return tools


def _bound_release_toolchain(
    *,
    document_path: Path,
    expected_document_sha256: str,
    binding_kind: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one hash-bound release document and validate its complete toolchain."""
    if binding_kind not in {"manifest", "plan"}:
        raise ArtifactChainError(f"unsupported release tool binding: {binding_kind}")
    label = f"release tool {binding_kind}"
    payload, record = _read_json_object_with_record(
        document_path,
        label=label,
        expected_sha256=expected_document_sha256,
        require_private_mode=True,
    )
    if binding_kind == "plan":
        fixed_path = payload.get("fixed_path")
        if (
            not isinstance(fixed_path, str)
            or os.environ.get("PATH") != fixed_path
            or os.environ.get("VMLX_R20_FIXED_PATH") != fixed_path
        ):
            raise ArtifactChainError(
                "release tool plan fixed PATH does not match the action environment"
            )
    toolchain_key = "toolchain" if binding_kind == "manifest" else "tools"
    toolchain = _validate_pinned_toolchain(
        payload.get(toolchain_key),
        label=f"{label} toolchain",
    )
    return toolchain, record


def _strip_verified_release_python_action_from_git_status(
    *,
    arguments: list[str],
    stdout: str,
) -> str:
    """Remove only this runner's verified source-adjacent action hardlink.

    The release Python launcher executes a hash-bound hardlink beside a Python
    script so imports and ``__file__`` keep their source-tree semantics.  A
    pinned ``git status`` launched by this script can therefore observe that
    one action hardlink while it is running.  Exclude it from the measurement
    only after proving that the path is the expected random action name and is
    the same inode as the original runner.  Every other status row remains
    visible to the source-identity fence.
    """
    if (
        len(arguments) != 5
        or arguments[0] != "-C"
        or arguments[2:] != [
            "status",
            "--porcelain",
            "--untracked-files=all",
        ]
    ):
        return stdout

    repo = _absolute_path(Path(arguments[1]))
    action_path = Path(__file__).absolute()
    match = re.fullmatch(
        r"\.(?P<original>[^/]+\.py)\.vmlx-r20-(?P<nonce>[0-9a-f]{32})",
        action_path.name,
    )
    if match is None:
        return stdout
    try:
        relative_action = action_path.relative_to(repo).as_posix()
    except ValueError:
        return stdout

    original_path = action_path.with_name(match.group("original"))
    try:
        action_stat = action_path.stat(follow_symlinks=False)
        original_stat = original_path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ArtifactChainError(
            "release Python script action disappeared during git status"
        ) from exc
    if (
        not stat.S_ISREG(action_stat.st_mode)
        or not stat.S_ISREG(original_stat.st_mode)
        or action_stat.st_nlink < 2
        or (
            action_stat.st_dev,
            action_stat.st_ino,
            action_stat.st_size,
        )
        != (
            original_stat.st_dev,
            original_stat.st_ino,
            original_stat.st_size,
        )
    ):
        raise ArtifactChainError(
            "release Python git-status action is not the verified runner hardlink"
        )

    expected_row = f"?? {relative_action}"
    rows = stdout.splitlines(keepends=True)
    matching = [
        index
        for index, row in enumerate(rows)
        if row.rstrip("\r\n") == expected_row
    ]
    if not matching:
        return stdout
    if len(matching) != 1:
        raise ArtifactChainError(
            "release Python git-status action appeared more than once"
        )
    del rows[matching[0]]
    return "".join(rows)


def run_bound_tool_action(
    *,
    document_path: Path,
    expected_document_sha256: str,
    binding_kind: str,
    action: str,
    arguments: list[str],
    cwd: Path | None = None,
    capture_output: bool = False,
) -> dict[str, Any]:
    """Validate, invoke, then revalidate one restricted pinned-tool action.

    The executable path never leaves this action boundary.  Both the independent
    document digest and the complete pinned toolchain are revalidated after the
    child exits, including when the child fails.
    """
    toolchain, before = _bound_release_toolchain(
        document_path=document_path,
        expected_document_sha256=expected_document_sha256,
        binding_kind=binding_kind,
    )
    argv = [str(value) for value in arguments]
    if argv[:1] == ["--"]:
        argv = argv[1:]
    action_argv: dict[str, list[str]] = {
        "node": [str(toolchain["node"]["realpath"]), *argv],
        "npm": [
            str(toolchain["node"]["realpath"]),
            str(toolchain["npm"]["realpath"]),
            *argv,
        ],
        "npx": [
            str(toolchain["node"]["realpath"]),
            str(toolchain["npx"]["realpath"]),
            *argv,
        ],
        "asar": [
            str(toolchain["node"]["realpath"]),
            str(toolchain["asar"]["realpath"]),
            *argv,
        ],
        "app-builder": [str(toolchain["app_builder"]["realpath"]), *argv],
        "electron-builder": [
            str(toolchain["node"]["realpath"]),
            str(toolchain["electron_builder"]["realpath"]),
            *argv,
        ],
        "git": [str(toolchain["git"]["realpath"]), *argv],
        "shasum": [str(toolchain["shasum"]["realpath"]), *argv],
        "awk": [str(toolchain["awk"]["realpath"]), *argv],
        "file": [str(toolchain["file"]["realpath"]), *argv],
        "find": [str(toolchain["find"]["realpath"]), *argv],
    }
    if action not in action_argv:
        raise ArtifactChainError(f"unsupported pinned-tool action: {action}")

    action_cwd: Path | None = None
    if cwd is not None:
        action_cwd = _absolute_path(cwd)
        _reject_symlinked_ancestors(action_cwd, label="pinned-tool action cwd")
        if action_cwd.is_symlink() or not action_cwd.is_dir():
            raise ArtifactChainError(
                f"pinned-tool action cwd is not a real directory: {action_cwd}"
            )

    completed: subprocess.CompletedProcess[str] | None = None
    invocation_error: OSError | None = None
    previous_umask = os.umask(0o077)
    try:
        try:
            completed = subprocess.run(
                action_argv[action],
                cwd=action_cwd,
                text=True,
                errors="surrogateescape",
                capture_output=capture_output,
                check=False,
            )
        except OSError as exc:
            invocation_error = exc
    finally:
        os.umask(previous_umask)
        _, after = _bound_release_toolchain(
            document_path=document_path,
            expected_document_sha256=expected_document_sha256,
            binding_kind=binding_kind,
        )
        if (
            before["device"],
            before["inode"],
            before["sha256"],
        ) != (
            after["device"],
            after["inode"],
            after["sha256"],
        ):
            raise ArtifactChainError(
                f"release tool {binding_kind} identity changed across action"
            )

    if invocation_error is not None:
        raise ArtifactChainError(
            f"pinned-tool action {action} could not start: {invocation_error}"
        ) from invocation_error
    assert completed is not None
    if completed.returncode != 0:
        safe_stderr = (completed.stderr or "").encode(
            "utf-8",
            errors="backslashreplace",
        ).decode("utf-8")
        detail = (
            f": {safe_stderr.strip()}"
            if capture_output and safe_stderr
            else ""
        )
        raise ArtifactChainError(
            f"pinned-tool action {action} failed with exit "
            f"{completed.returncode}{detail}"
        )
    result: dict[str, Any] = {
        "action": action,
        "document_sha256": str(after["sha256"]),
        "returncode": completed.returncode,
    }
    if capture_output:
        stdout = completed.stdout or ""
        if action == "git":
            stdout = _strip_verified_release_python_action_from_git_status(
                arguments=argv,
                stdout=stdout,
            )
        result["stdout"] = stdout
        result["stderr"] = completed.stderr or ""
    return result


def _validate_hook_completion_attestation(
    *,
    root: Path,
    dist_dir: Path,
    version: str,
    private_root: Path,
    flavor: str,
    attestation_path: Path,
    expected_sha256: str,
    expected_nonce: str,
    expected_driver_pid: int,
    require_current_artifacts: bool = True,
) -> dict[str, Any]:
    if flavor not in R20_ARTIFACT_CHAIN_FLAVORS:
        raise ArtifactChainError(f"unsupported hook-completion flavor: {flavor}")
    path = _assert_within_private_root(
        private_root,
        attestation_path,
        label=f"{flavor} electron-builder completion attestation",
    )
    payload, record = _read_json_object_with_record(
        path,
        label=f"{flavor} electron-builder completion attestation",
        expected_sha256=expected_sha256,
        require_private_mode=True,
    )
    _require_exact_dict_keys(
        payload,
        {
            "schema_version",
            "scope",
            "stage",
            "version",
            "flavor",
            "source",
            "preflight_sha256",
            "plan",
            "fixed_path",
            "tools",
            "bundle_runtime",
            "runtime_contract",
            "staged_app",
            "extracted_asar",
            "artifacts",
        },
        label=f"{flavor} electron-builder completion attestation",
    )
    if (
        payload["schema_version"] != 1
        or payload["scope"] != R20_ARTIFACT_CHAIN_SCOPE
        or payload["stage"] != "electron_builder_completion"
        or payload["version"] != version
        or payload["flavor"] != flavor
    ):
        raise ArtifactChainError(f"{flavor} hook-completion identity is invalid")
    source = _require_exact_dict_keys(
        payload["source"],
        {"commit", "tree"},
        label=f"{flavor} hook source",
    )
    current_source = _git_source_identity(root)
    if (
        source["commit"] != current_source["commit"]
        or source["tree"] != current_source["tree"]
    ):
        raise ArtifactChainError(f"{flavor} hook source differs from clean Git")
    if re.fullmatch(r"[0-9a-f]{64}", str(payload["preflight_sha256"])) is None:
        raise ArtifactChainError(f"{flavor} hook preflight digest is malformed")
    plan = _require_exact_dict_keys(
        payload["plan"],
        {"path", "sha256", "nonce", "driver_pid"},
        label=f"{flavor} hook plan",
    )
    if (
        plan["nonce"] != expected_nonce
        or plan["driver_pid"] != expected_driver_pid
        or re.fullmatch(r"[0-9a-f]{64}", str(plan["sha256"])) is None
    ):
        raise ArtifactChainError(f"{flavor} hook plan does not match the build driver")
    expected_app = (
        _absolute_path(dist_dir) / f"{flavor}-app/mac-arm64/vMLX.app"
    )
    staged = _require_exact_dict_keys(
        payload["staged_app"],
        {"path", "payload"},
        label=f"{flavor} hook staged app",
    )
    if _absolute_path(Path(str(staged["path"]))) != expected_app:
        raise ArtifactChainError(f"{flavor} hook staged app path is not exact")
    if not isinstance(staged["payload"], dict):
        raise ArtifactChainError(f"{flavor} hook staged app payload is missing")
    extracted = _require_exact_dict_keys(
        payload["extracted_asar"],
        {"payload"},
        label=f"{flavor} hook extracted ASAR",
    )
    if not isinstance(extracted["payload"], dict):
        raise ArtifactChainError(f"{flavor} hook ASAR payload is missing")
    tools = _validate_pinned_toolchain(
        payload["tools"],
        label=f"{flavor} hook toolchain",
    )
    if payload["fixed_path"] != "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin":
        raise ArtifactChainError(f"{flavor} hook fixed PATH is invalid")
    del tools
    bundle_reference = _require_exact_dict_keys(
        payload["bundle_runtime"],
        {"path", "sha256"},
        label=f"{flavor} hook bundle runtime",
    )
    bundle = _validate_bundle_runtime_attestation(
        root=root,
        version=version,
        private_root=private_root,
        flavor=flavor,
        attestation_path=Path(str(bundle_reference["path"])),
        expected_sha256=str(bundle_reference["sha256"]),
    )
    runtime_contract = payload["runtime_contract"]
    if (
        not isinstance(runtime_contract, dict)
        or runtime_contract.get("flavor") != flavor
        or runtime_contract.get("mlx_wheel_platform")
        != _expected_runtime_contract(flavor)["mlx_wheel_platform"]
        or runtime_contract.get("minimum_system_version")
        != _expected_runtime_contract(flavor)["minimum_system_version"]
        or {
            key: value
            for key, value in runtime_contract.items()
            if key != "info_plist_sha256"
        }
        != bundle["payload"]["runtime_contract"]
    ):
        raise ArtifactChainError(
            f"{flavor} hook runtime differs from sealed bundle provenance"
        )
    artifacts = _require_exact_dict_keys(
        payload["artifacts"],
        {"dmg", "blockmap"},
        label=f"{flavor} hook artifacts",
    )
    expected_paths = _expected_release_artifact_paths(dist_dir, version)[flavor]
    for kind in ("dmg", "blockmap"):
        artifact = _require_exact_dict_keys(
            artifacts[kind],
            {"path", "sha256", "size", "mode"},
            label=f"{flavor} hook {kind}",
        )
        if artifact["path"] != str(expected_paths[kind]):
            raise ArtifactChainError(
                f"{flavor} hook {kind} path differs from the generated artifact"
            )
        if require_current_artifacts:
            current = _safe_regular_file(
                expected_paths[kind],
                label=f"{flavor} hook-bound {kind}",
            )
            if (
                artifact["sha256"] != current["sha256"]
                or artifact["size"] != current["size"]
                or artifact["mode"] != current["mode"]
            ):
                raise ArtifactChainError(
                    f"{flavor} hook {kind} changed after completion"
                )
    return {"path": str(path), "sha256": record["sha256"], "payload": payload}


def write_dmg_payload_parity_attestation(
    *,
    root: Path,
    dist_dir: Path,
    version: str,
    private_root: Path,
    flavor: str,
    hook_attestation_path: Path,
    expected_hook_sha256: str,
    expected_nonce: str,
    expected_driver_pid: int,
    mounted_app: Path,
    extracted_asar: Path,
    output_path: Path,
) -> dict[str, Any]:
    hook = _validate_hook_completion_attestation(
        root=root,
        dist_dir=dist_dir,
        version=version,
        private_root=private_root,
        flavor=flavor,
        attestation_path=hook_attestation_path,
        expected_sha256=expected_hook_sha256,
        expected_nonce=expected_nonce,
        expected_driver_pid=expected_driver_pid,
    )
    mounted_app = _absolute_path(mounted_app)
    extracted_asar = _absolute_path(extracted_asar)
    app_payload = _tree_payload_records(
        mounted_app,
        label=f"mounted pre-notary {flavor} application",
    )
    asar_payload = _tree_payload_records(
        extracted_asar,
        label=f"mounted pre-notary {flavor} ASAR",
        ignore_ambient_finder_metadata=True,
    )
    if app_payload != hook["payload"]["staged_app"]["payload"]:
        raise ArtifactChainError(
            f"mounted {flavor} DMG app differs from hook-completed staged app"
        )
    if asar_payload != hook["payload"]["extracted_asar"]["payload"]:
        raise ArtifactChainError(
            f"mounted {flavor} DMG ASAR differs from hook-completed staged ASAR"
        )
    runtime_contract = _inspect_app_runtime_contract(
        app=mounted_app,
        flavor=flavor,
        version=version,
    )
    if runtime_contract != hook["payload"]["runtime_contract"]:
        raise ArtifactChainError(
            f"mounted {flavor} DMG runtime differs from hook-completed runtime"
        )
    resources = mounted_app / "Contents/Resources"
    bundled_candidates = list(
        (resources / "bundled-python/python/lib").glob(
            "python*/site-packages/vmlx_engine"
        )
    )
    if len(bundled_candidates) != 1:
        raise ArtifactChainError(
            f"mounted {flavor} DMG must contain exactly one vmlx_engine"
        )
    source_engine = _absolute_path(root) / "vmlx_engine"
    source_mirror = _require_tree_parity(
        source_engine,
        resources / "vmlx-engine-source/vmlx_engine",
        label=f"mounted {flavor} source-mirror vmlx_engine",
        exclude_python_bytecode=True,
    )
    bundled_runtime = _require_runtime_tree_parity(
        source_engine,
        bundled_candidates[0],
        label=f"mounted {flavor} bundled-runtime vmlx_engine",
    )
    renderer = _require_tree_parity(
        _absolute_path(root) / "panel/out",
        extracted_asar / "out",
        label=f"mounted {flavor} Electron output",
    )
    output_path = _assert_within_private_root(
        private_root,
        output_path,
        label=f"{flavor} mounted DMG parity attestation",
    )
    payload = {
        "schema_version": 1,
        "scope": R20_ARTIFACT_CHAIN_SCOPE,
        "stage": "mounted_dmg_payload_parity",
        "version": version,
        "flavor": flavor,
        "source": hook["payload"]["source"],
        "hook_completion": {
            "path": hook["path"],
            "sha256": hook["sha256"],
            "plan_sha256": hook["payload"]["plan"]["sha256"],
            "nonce": hook["payload"]["plan"]["nonce"],
            "driver_pid": hook["payload"]["plan"]["driver_pid"],
        },
        "artifacts": hook["payload"]["artifacts"],
        "runtime_contract": runtime_contract,
        "app_payload": app_payload,
        "asar_payload": asar_payload,
        "source_mirror": source_mirror,
        "bundled_runtime": bundled_runtime,
        "renderer": renderer,
    }
    digest = _write_private_json_with_digest(output_path, payload)
    return {"attestation": str(output_path), "sha256": digest, "payload": payload}


def _validate_dmg_payload_parity_attestation(
    *,
    private_root: Path,
    flavor: str,
    attestation_path: Path,
    expected_sha256: str,
    expected_hook_sha256: str,
    expected_nonce: str,
    expected_driver_pid: int,
) -> dict[str, Any]:
    path = _assert_within_private_root(
        private_root,
        attestation_path,
        label=f"{flavor} mounted DMG parity attestation",
    )
    payload, record = _read_json_object_with_record(
        path,
        label=f"{flavor} mounted DMG parity attestation",
        expected_sha256=expected_sha256,
        require_private_mode=True,
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("scope") != R20_ARTIFACT_CHAIN_SCOPE
        or payload.get("stage") != "mounted_dmg_payload_parity"
        or payload.get("version") != R20_ARTIFACT_CHAIN_VERSION
        or payload.get("flavor") != flavor
    ):
        raise ArtifactChainError(f"{flavor} mounted DMG parity identity is invalid")
    hook = payload.get("hook_completion")
    if (
        not isinstance(hook, dict)
        or hook.get("sha256") != expected_hook_sha256
        or hook.get("nonce") != expected_nonce
        or hook.get("driver_pid") != expected_driver_pid
    ):
        raise ArtifactChainError(
            f"{flavor} mounted DMG parity does not bind the hook completion"
        )
    for field in (
        "runtime_contract",
        "app_payload",
        "asar_payload",
        "source_mirror",
        "bundled_runtime",
        "renderer",
    ):
        if not isinstance(payload.get(field), dict):
            raise ArtifactChainError(
                f"{flavor} mounted DMG parity is missing {field}"
            )
    return {"path": str(path), "sha256": record["sha256"], "payload": payload}


def write_final_notary_artifact_manifest(
    *,
    root: Path,
    dist_dir: Path,
    version: str,
    pre_notary_manifest_path: Path,
    expected_pre_manifest_sha256: str,
    expected_source_commit: str,
    expected_source_tree: str,
    expected_preflight_sha256: str,
    private_root: Path,
    output_path: Path,
    submission_ids: dict[str, str],
    submitted_snapshot_paths: dict[str, Path],
) -> dict[str, Any]:
    private_root = ensure_private_evidence_root(private_root)
    output_path = _assert_within_private_root(
        private_root,
        output_path,
        label="final post-notary manifest",
    )
    if output_path.exists() or output_path.is_symlink():
        raise ArtifactChainError(
            "refusing to overwrite an existing final post-notary manifest"
        )
    pre = _validate_pre_notary_artifact_manifest_metadata(
        root=root,
        dist_dir=dist_dir,
        version=version,
        private_root=private_root,
        manifest_path=pre_notary_manifest_path,
        expected_manifest_sha256=expected_pre_manifest_sha256,
        expected_source_commit=expected_source_commit,
        expected_source_tree=expected_source_tree,
        expected_preflight_sha256=expected_preflight_sha256,
        require_current_artifact_hashes=False,
    )
    post_records = _release_artifact_records(dist_dir, version)
    if set(submission_ids) != set(R20_ARTIFACT_CHAIN_FLAVORS):
        raise ArtifactChainError("Apple submission set must contain Sequoia and Tahoe")
    if set(submitted_snapshot_paths) != set(R20_ARTIFACT_CHAIN_FLAVORS):
        raise ArtifactChainError("submitted snapshot set must contain Sequoia and Tahoe")
    artifacts: dict[str, dict[str, Any]] = {}
    seen_submission_ids: set[str] = set()
    for flavor in R20_ARTIFACT_CHAIN_FLAVORS:
        pre_record = pre["payload"]["artifacts"][flavor]
        post_record = post_records[flavor]
        submitted_path = _assert_within_private_root(
            private_root,
            submitted_snapshot_paths[flavor],
            label=f"immutable submitted {flavor} DMG",
        )
        submitted = _safe_regular_file(
            submitted_path,
            label=f"immutable submitted {flavor} DMG",
        )
        _require_private_mode(
            Path(submitted["path"]),
            label=f"immutable submitted {flavor} DMG",
            record=submitted,
        )
        if submitted["sha256"] != pre_record["dmg_sha256"]:
            raise ArtifactChainError(
                f"immutable submitted {flavor} DMG does not match the build handoff"
            )
        try:
            submission_id = str(uuid.UUID(submission_ids[flavor]))
        except ValueError as exc:
            raise ArtifactChainError(
                f"{flavor} Apple submission id is not a canonical UUID"
            ) from exc
        if submission_id != submission_ids[flavor].lower():
            raise ArtifactChainError(
                f"{flavor} Apple submission UUID is not canonical"
            )
        if submission_id in seen_submission_ids:
            raise ArtifactChainError(
                "Sequoia and Tahoe must have distinct Apple submission IDs"
            )
        seen_submission_ids.add(submission_id)
        if post_record["dmg_sha256"] == pre_record["dmg_sha256"]:
            raise ArtifactChainError(
                f"{flavor} DMG hash did not change after stapling"
            )
        if post_record["blockmap_sha256"] == pre_record["blockmap_sha256"]:
            raise ArtifactChainError(
                f"{flavor} blockmap hash did not change after post-staple regeneration"
            )
        artifacts[flavor] = {
            "flavor": flavor,
            "dmg_path": post_record["dmg_path"],
            "dmg_pre_notary_sha256": pre_record["dmg_sha256"],
            "dmg_post_notary_sha256": post_record["dmg_sha256"],
            "dmg_post_notary_size": post_record["dmg_size"],
            "blockmap_path": post_record["blockmap_path"],
            "blockmap_pre_notary_sha256": pre_record["blockmap_sha256"],
            "blockmap_post_notary_sha256": post_record["blockmap_sha256"],
            "blockmap_post_notary_size": post_record["blockmap_size"],
            "submitted_dmg_path": submitted["path"],
            "submitted_dmg_sha256": submitted["sha256"],
            "submitted_dmg_size": submitted["size"],
            "submitted_dmg_device": submitted["device"],
            "submitted_dmg_inode": submitted["inode"],
            "notary_status": "requires_fresh_final_query",
            "notary_submission_id": submission_id,
        }
    payload = {
        "schema_version": R20_ARTIFACT_CHAIN_SCHEMA_VERSION,
        "scope": R20_ARTIFACT_CHAIN_SCOPE,
        "stage": "post_notary",
        "version": version,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": pre["payload"]["source"],
        "preflight": pre["payload"]["preflight"],
        "pre_notary_manifest": {
            "path": pre["manifest"],
            "sha256": pre["sha256"],
        },
        "runtime_contracts": pre["payload"]["runtime_contracts"],
        "toolchain": pre["payload"]["toolchain"],
        "artifacts": artifacts,
    }
    digest = _write_private_json_with_digest(output_path, payload)
    return {
        "manifest": str(output_path),
        "sha256": digest,
        "payload": payload,
    }


def validate_final_notary_artifact_manifest(
    *,
    root: Path,
    dist_dir: Path,
    version: str,
    private_root: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
    expected_pre_manifest_sha256: str,
    expected_source_commit: str,
    expected_source_tree: str,
    expected_preflight_sha256: str,
) -> dict[str, Any]:
    private_root = ensure_private_evidence_root(private_root)
    manifest_path = _assert_within_private_root(
        private_root,
        manifest_path,
        label="final post-notary manifest",
    )
    payload, manifest_record = _read_json_object_with_record(
        manifest_path,
        label="final post-notary manifest",
        expected_sha256=expected_manifest_sha256,
        require_private_mode=True,
    )
    manifest_digest = str(manifest_record["sha256"])
    _require_exact_dict_keys(
        payload,
        {
            "schema_version",
            "scope",
            "stage",
            "version",
            "created_at",
            "source",
            "preflight",
            "pre_notary_manifest",
            "runtime_contracts",
            "toolchain",
            "artifacts",
        },
        label="final post-notary manifest",
    )
    if (
        payload["schema_version"] != R20_ARTIFACT_CHAIN_SCHEMA_VERSION
        or payload["scope"] != R20_ARTIFACT_CHAIN_SCOPE
        or payload["stage"] != "post_notary"
        or payload["version"] != version
    ):
        raise ArtifactChainError("final post-notary manifest identity is invalid")
    current_source = _git_source_identity(root)
    if payload["source"] != current_source:
        raise ArtifactChainError(
            "final post-notary manifest source does not match current clean Git source"
        )
    if (
        expected_source_commit != current_source["commit"]
        or expected_source_tree != current_source["tree"]
    ):
        raise ArtifactChainError(
            "independently supplied source commit/tree do not match final release source"
        )
    pre_manifest = _require_exact_dict_keys(
        payload["pre_notary_manifest"],
        {"path", "sha256"},
        label="final pre-notary manifest reference",
    )
    pre = _validate_pre_notary_artifact_manifest_metadata(
        root=root,
        dist_dir=dist_dir,
        version=version,
        private_root=private_root,
        manifest_path=Path(str(pre_manifest["path"])),
        expected_manifest_sha256=expected_pre_manifest_sha256,
        expected_source_commit=expected_source_commit,
        expected_source_tree=expected_source_tree,
        expected_preflight_sha256=expected_preflight_sha256,
        require_current_artifact_hashes=False,
    )
    if pre["sha256"] != pre_manifest["sha256"]:
        raise ArtifactChainError("pre-notary manifest reference hash changed")
    if payload["preflight"] != pre["payload"]["preflight"]:
        raise ArtifactChainError("final manifest preflight does not match build manifest")
    if payload["runtime_contracts"] != pre["payload"]["runtime_contracts"]:
        raise ArtifactChainError(
            "final runtime contracts do not match the pre-notary manifest"
        )
    if payload["toolchain"] != pre["payload"]["toolchain"]:
        raise ArtifactChainError(
            "final toolchain does not match the pre-notary manifest"
        )
    _validate_pinned_toolchain(
        payload["toolchain"],
        label="final manifest toolchain",
    )
    artifacts = _require_exact_dict_keys(
        payload["artifacts"],
        set(R20_ARTIFACT_CHAIN_FLAVORS),
        label="final post-notary artifacts",
    )
    post_records = _release_artifact_records(dist_dir, version)
    final_artifact_keys = {
        "flavor",
        "dmg_path",
        "dmg_pre_notary_sha256",
        "dmg_post_notary_sha256",
        "dmg_post_notary_size",
        "blockmap_path",
        "blockmap_pre_notary_sha256",
        "blockmap_post_notary_sha256",
        "blockmap_post_notary_size",
        "submitted_dmg_path",
        "submitted_dmg_sha256",
        "submitted_dmg_size",
        "submitted_dmg_device",
        "submitted_dmg_inode",
        "notary_status",
        "notary_submission_id",
    }
    submission_ids: set[str] = set()
    for flavor in R20_ARTIFACT_CHAIN_FLAVORS:
        record = _require_exact_dict_keys(
            artifacts[flavor],
            final_artifact_keys,
            label=f"final {flavor} artifact",
        )
        current = post_records[flavor]
        before = pre["payload"]["artifacts"][flavor]
        submitted_path = _assert_within_private_root(
            private_root,
            Path(str(record["submitted_dmg_path"])),
            label=f"immutable submitted {flavor} DMG",
        )
        submitted = _safe_regular_file(
            submitted_path,
            label=f"immutable submitted {flavor} DMG",
        )
        _require_private_mode(
            Path(submitted["path"]),
            label=f"immutable submitted {flavor} DMG",
            record=submitted,
        )
        if submitted["sha256"] != before["dmg_sha256"]:
            raise ArtifactChainError(
                f"immutable submitted {flavor} DMG does not match the build handoff"
            )
        if current["dmg_sha256"] == before["dmg_sha256"]:
            raise ArtifactChainError(f"{flavor} DMG hash did not change after stapling")
        if current["blockmap_sha256"] == before["blockmap_sha256"]:
            raise ArtifactChainError(
                f"{flavor} blockmap hash did not change after post-staple regeneration"
            )
        try:
            submission_id = str(uuid.UUID(str(record["notary_submission_id"])))
        except ValueError as exc:
            raise ArtifactChainError(
                f"final {flavor} Apple submission id is not a UUID"
            ) from exc
        if submission_id != str(record["notary_submission_id"]).lower():
            raise ArtifactChainError(
                f"final {flavor} Apple submission UUID is not canonical"
            )
        if record["notary_status"] != "requires_fresh_final_query":
            raise ArtifactChainError(
                f"final {flavor} must not claim locally attested Apple acceptance"
            )
        if submission_id in submission_ids:
            raise ArtifactChainError(
                "Sequoia and Tahoe final records reuse one Apple submission ID"
            )
        submission_ids.add(submission_id)
        expected = {
            "flavor": flavor,
            "dmg_path": current["dmg_path"],
            "dmg_pre_notary_sha256": before["dmg_sha256"],
            "dmg_post_notary_sha256": current["dmg_sha256"],
            "dmg_post_notary_size": current["dmg_size"],
            "blockmap_path": current["blockmap_path"],
            "blockmap_pre_notary_sha256": before["blockmap_sha256"],
            "blockmap_post_notary_sha256": current["blockmap_sha256"],
            "blockmap_post_notary_size": current["blockmap_size"],
            "submitted_dmg_path": submitted["path"],
            "submitted_dmg_sha256": submitted["sha256"],
            "submitted_dmg_size": submitted["size"],
            "submitted_dmg_device": submitted["device"],
            "submitted_dmg_inode": submitted["inode"],
            "notary_status": "requires_fresh_final_query",
            "notary_submission_id": submission_id,
        }
        if record != expected:
            raise ArtifactChainError(
                f"final {flavor} manifest does not match current notarized artifacts"
            )
    return {
        "manifest": str(manifest_path),
        "sha256": manifest_digest,
        "payload": payload,
    }


def validate_mounted_app_against_final_manifest(
    *,
    root: Path,
    dist_dir: Path,
    version: str,
    private_root: Path,
    final_manifest_path: Path,
    expected_final_manifest_sha256: str,
    expected_pre_manifest_sha256: str,
    expected_source_commit: str,
    expected_source_tree: str,
    expected_preflight_sha256: str,
    flavor: str,
    mounted_app: Path,
    extracted_asar: Path,
) -> dict[str, Any]:
    if flavor not in R20_ARTIFACT_CHAIN_FLAVORS:
        raise ArtifactChainError(f"unsupported mounted flavor: {flavor}")
    final = validate_final_notary_artifact_manifest(
        root=root,
        dist_dir=dist_dir,
        version=version,
        private_root=private_root,
        manifest_path=final_manifest_path,
        expected_manifest_sha256=expected_final_manifest_sha256,
        expected_pre_manifest_sha256=expected_pre_manifest_sha256,
        expected_source_commit=expected_source_commit,
        expected_source_tree=expected_source_tree,
        expected_preflight_sha256=expected_preflight_sha256,
    )
    pre_reference = final["payload"]["pre_notary_manifest"]
    pre = _validate_pre_notary_artifact_manifest_metadata(
        root=root,
        dist_dir=dist_dir,
        version=version,
        private_root=private_root,
        manifest_path=Path(str(pre_reference["path"])),
        expected_manifest_sha256=expected_pre_manifest_sha256,
        expected_source_commit=expected_source_commit,
        expected_source_tree=expected_source_tree,
        expected_preflight_sha256=expected_preflight_sha256,
        require_current_artifact_hashes=False,
    )
    mounted_app = _absolute_path(mounted_app)
    _reject_symlinked_ancestors(mounted_app, label="mounted application")
    if mounted_app.is_symlink() or not mounted_app.is_dir():
        raise ArtifactChainError("mounted application is not a real directory")
    runtime_contract = _inspect_app_runtime_contract(
        app=mounted_app,
        flavor=flavor,
        version=version,
    )
    if runtime_contract != final["payload"]["runtime_contracts"][flavor]:
        raise ArtifactChainError(
            f"mounted {flavor} runtime/minimum-OS contract differs from final manifest"
        )
    resources = mounted_app / "Contents/Resources"
    bundled_candidates = list(
        (resources / "bundled-python/python/lib").glob(
            "python*/site-packages/vmlx_engine"
        )
    )
    if len(bundled_candidates) != 1:
        raise ArtifactChainError("mounted app must contain exactly one vmlx_engine")
    source_engine = _absolute_path(root) / "vmlx_engine"
    mirror = _require_tree_parity(
        source_engine,
        resources / "vmlx-engine-source/vmlx_engine",
        label="mounted source-mirror vmlx_engine",
        exclude_python_bytecode=True,
    )
    runtime = _require_runtime_tree_parity(
        source_engine,
        bundled_candidates[0],
        label="mounted bundled-runtime vmlx_engine",
    )
    renderer = _require_tree_parity(
        _absolute_path(root) / "panel/out",
        _absolute_path(extracted_asar) / "out",
        label="mounted Electron renderer/main output",
    )
    staged = pre["payload"]["staged"][flavor]
    app_payload = _tree_payload_records(
        mounted_app,
        label="mounted application payload",
    )
    asar_payload = _tree_payload_records(
        _absolute_path(extracted_asar),
        label="mounted extracted ASAR payload",
        ignore_ambient_finder_metadata=True,
    )
    if app_payload != staged["app_payload"]:
        raise ArtifactChainError(
            f"mounted {flavor} app payload/modes differ from build-driver attestation"
        )
    if asar_payload != staged["asar_payload"]:
        raise ArtifactChainError(
            f"mounted {flavor} ASAR payload/modes differ from build-driver attestation"
        )
    return {
        "flavor": flavor,
        "app": str(mounted_app),
        "source_mirror": mirror,
        "bundled_runtime": runtime,
        "renderer": renderer,
        "runtime_contract": runtime_contract,
        "app_tree_sha256": app_payload["tree_sha256"],
        "asar_tree_sha256": asar_payload["tree_sha256"],
    }


def _parse_counts(output: str) -> dict[str, int | None]:
    return parse_counts(output)

def _check_packaged_renderer_dsv4_cache_ui(root: Path) -> bool:
    app_asar = root / PACKAGED_RENDERER_ASAR
    if not app_asar.exists():
        return False
    data = app_asar.read_bytes()
    return all(
        marker in data for marker in PACKAGED_RENDERER_REQUIRED_DSV4_CACHE_UI_STRINGS
    ) and not any(
        marker in data for marker in PACKAGED_RENDERER_FORBIDDEN_DSV4_CACHE_UI_STRINGS
    )


def _check_packaged_renderer_max_thinking_tokens(root: Path) -> bool:
    app_asar = root / PACKAGED_RENDERER_ASAR
    if not app_asar.exists():
        return False
    data = app_asar.read_bytes()
    return all(marker in data for marker in PACKAGED_RENDERER_REQUIRED_MAX_THINKING_STRINGS)


def _check_packaged_epipe_closed_stream_guards(root: Path) -> bool:
    app_asar = root / PACKAGED_RENDERER_ASAR
    if not app_asar.exists():
        return False
    data = app_asar.read_bytes()
    return all(marker in data for marker in PACKAGED_EPIPE_CLOSED_STREAM_GUARD_STRINGS)


def _check_packaged_user_data_isolation_bootstrap(root: Path) -> bool:
    app_asar = root / PACKAGED_RENDERER_ASAR
    if not app_asar.exists():
        return False
    data = app_asar.read_bytes()
    if not all(marker in data for marker in PACKAGED_USER_DATA_ISOLATION_STRINGS):
        return False
    set_path_index = data.find(b"setPath")
    lock_index = data.find(b"requestSingleInstanceLock")
    return set_path_index >= 0 and lock_index >= 0 and set_path_index < lock_index


def _check_packaged_python_has_no_pycache(root: Path) -> bool:
    python_root = root / PACKAGED_PYTHON_ROOT
    if not python_root.exists():
        return False
    return not any(
        path.is_file()
        for path in python_root.rglob("*.pyc")
        if "__pycache__" in path.parts
    )


def _check_staged_app_engine_hash_parity(root: Path) -> bool:
    source_engine_dir = root / "vmlx_engine"
    staged_engine_dir = (
        root
        / PACKAGED_PYTHON_ROOT
        / "lib/python3.12/site-packages/vmlx_engine"
    )
    checked = 0
    for rel in STAGED_APP_ENGINE_HASH_FILES:
        source = source_engine_dir / rel
        if not source.exists():
            continue
        staged = staged_engine_dir / rel
        if not staged.exists() or _sha256(source) != _sha256(staged):
            return False
        checked += 1
    return checked > 0


def _check_staged_app_engine_source_hash_parity(root: Path) -> bool:
    source_engine_dir = root / "vmlx_engine"
    staged_engine_dir = (
        root
        / PACKAGED_APP
        / "Contents/Resources/vmlx-engine-source/vmlx_engine"
    )
    checked = 0
    for rel in STAGED_APP_ENGINE_HASH_FILES:
        source = source_engine_dir / rel
        if not source.exists():
            continue
        staged = staged_engine_dir / rel
        if not staged.exists() or _sha256(source) != _sha256(staged):
            return False
        checked += 1
    return checked > 0


def _check_release_dmg_hardened_runtime_contract(root: Path) -> bool:
    script = root / "panel/scripts/build-release-dmgs.sh"
    entitlements = root / "panel/build/entitlements.mac.plist"
    if not script.exists() or not entitlements.exists():
        return False
    text = script.read_text(encoding="utf-8", errors="replace")
    try:
        final_sign_block = text[
            text.index("finalize_release_app_signature()") : text.index("find_staged_app()")
        ]
    except ValueError:
        return False
    # 2026-08-16 (ledger row 151): the script signs through the pinned
    # $APPLE_CODESIGN tool path (unshadowed-tool hardening), so the contract
    # matches that invocation form — same semantics, stricter provenance.
    return (
        '"$APPLE_CODESIGN" --force --deep' in final_sign_block
        and "--timestamp" in final_sign_block
        and "--options runtime" in final_sign_block
        and "--entitlements" in final_sign_block
        and "entitlements.mac.plist" in text
        and '"$APPLE_CODESIGN" --verify --deep --strict' in final_sign_block
    )


def _check_release_dmg_notarization_verifier_contract(root: Path) -> bool:
    script = root / "panel/scripts/verify-release-dmgs.sh"
    if not script.exists():
        return False
    text = script.read_text(encoding="utf-8", errors="replace")
    required_fragments = (
        "sequoia tahoe",
        "hdiutil verify",
        "codesign --verify",
        "codesign -dv",
        "require_developer_id_signature",
        "Signature=adhoc",
        "xcrun stapler validate",
        "spctl --assess --type open --context context:primary-signature",
    )
    has_final_hash_verification = (
        "shasum -a 256" in text
        or "artifact_chain check-final" in text
    )
    has_exact_identity_assertion = (
        "Authority=Developer ID Application: ShieldStack LLC (55KGF2S5AY)" in text
        and "TeamIdentifier=55KGF2S5AY" in text
    ) or (
        'EXPECTED_CODESIGN_IDENTITY="Developer ID Application: ShieldStack LLC (55KGF2S5AY)"'
        in text
        and 'EXPECTED_APPLE_TEAM_ID="55KGF2S5AY"' in text
        and 'Authority=$EXPECTED_CODESIGN_IDENTITY' in text
        and 'TeamIdentifier=$EXPECTED_APPLE_TEAM_ID' in text
    )
    return (
        all(fragment in text for fragment in required_fragments)
        and has_final_hash_verification
        and has_exact_identity_assertion
    )


def _check_release_dmg_notarization_submit_contract(root: Path) -> bool:
    script = root / "panel/scripts/notarize-release-dmgs.sh"
    if not script.exists():
        return False
    text = script.read_text(encoding="utf-8", errors="replace")
    verifier = root / "panel/scripts/verify-release-dmgs.sh"
    combined = text
    if verifier.exists():
        combined += "\n" + verifier.read_text(encoding="utf-8", errors="replace")
    required_fragments = (
        "sequoia tahoe",
        "VMLINUX_NOTARY_KEYCHAIN_PROFILE",
        "VMLINUX_NOTARY_KEYCHAIN",
        "notarytool_args",
        "--keychain",
        "codesign --verify",
        "codesign -dv",
        "require_developer_id_signature",
        "Signature=adhoc",
        "xcrun notarytool submit",
        "--keychain-profile",
        "--wait",
        "xcrun stapler staple",
        "xcrun stapler validate",
        "regenerate_blockmap",
        "app-builder",
        "blockmap",
        "spctl --assess --type open --context context:primary-signature",
    )
    has_final_hash_verification = (
        "shasum -a 256" in text
        or "artifact_chain check-final" in text
    )
    has_exact_identity_assertion = (
        "Authority=Developer ID Application: ShieldStack LLC (55KGF2S5AY)"
        in combined
        and "TeamIdentifier=55KGF2S5AY" in combined
    ) or (
        'EXPECTED_CODESIGN_IDENTITY="Developer ID Application: ShieldStack LLC (55KGF2S5AY)"'
        in combined
        and 'EXPECTED_APPLE_TEAM_ID="55KGF2S5AY"' in combined
        and 'Authority=$EXPECTED_CODESIGN_IDENTITY' in combined
        and 'TeamIdentifier=$EXPECTED_APPLE_TEAM_ID' in combined
    )
    return (
        all(fragment in combined for fragment in required_fragments)
        and has_final_hash_verification
        and has_exact_identity_assertion
    )


def _package_signing_preflight(root: Path) -> dict[str, Any]:
    app_path = root / PACKAGED_APP
    result: dict[str, Any] = {
        "status": "open",
        "app": str(PACKAGED_APP),
        "app_exists": app_path.exists(),
        "developer_id_signed": False,
        "developer_id_identity_count": 0,
        "keychain_info_statuses": [],
        "simple_developer_id_sign_rc": None,
        "simple_developer_id_sign_tail": [],
        "signing_blocker_reason": "packaged_app_missing",
        "signing_blocker_reasons": ["packaged_app_missing"],
        "manual_remediation_required": False,
        "remediation_summary": None,
        "remediation_steps": [],
        "signature_is_adhoc": False,
        "hardened_runtime_enabled": False,
        "team_identifier": None,
        "codesign_display_rc": None,
        "codesign_verify_rc": None,
        "packaged_app_modified_after_signing": False,
        "modified_after_signing_file_count": 0,
        "missing_after_signing_file_count": 0,
        "modified_after_signing_tail": [],
        "missing_after_signing_tail": [],
        "signature_summary_tail": [],
        "verify_tail": [],
    }
    if not app_path.exists():
        return result

    identity = subprocess.run(
        ["security", "find-identity", "-v", "-p", "codesigning"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    identity_lines = identity.stdout.splitlines()
    developer_identity_lines = [
        line for line in identity_lines if "Developer ID Application:" in line
    ]
    result["developer_id_identity_count"] = len(developer_identity_lines)
    result["developer_id_identity_tail"] = developer_identity_lines[-20:]

    keychain_statuses: list[dict[str, Any]] = []
    for keychain in SIGNING_KEYCHAINS:
        info = subprocess.run(
            ["security", "show-keychain-info", str(keychain)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        keychain_statuses.append(
            {
                "keychain": str(keychain),
                "returncode": info.returncode,
                "tail": info.stdout.splitlines()[-20:],
            }
        )
    result["keychain_info_statuses"] = keychain_statuses

    with tempfile.TemporaryDirectory(prefix="vmlx-codesign-preflight.") as tmpdir:
        probe = Path(tmpdir) / "codesign-probe"
        shutil.copyfile("/bin/echo", probe)
        probe.chmod(0o755)
        simple_sign = subprocess.run(
            [
                "codesign",
                "--force",
                "--sign",
                DEVELOPER_ID_IDENTITY,
                "--timestamp=none",
                str(probe),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result["simple_developer_id_sign_rc"] = simple_sign.returncode
    result["simple_developer_id_sign_tail"] = simple_sign.stdout.splitlines()[-40:]

    display = subprocess.run(
        ["codesign", "-dv", "--verbose=4", str(app_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    verify = subprocess.run(
        ["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    signature_text = display.stdout
    team_identifier = None
    for line in signature_text.splitlines():
        if line.startswith("TeamIdentifier="):
            team_identifier = line.split("=", 1)[1].strip()
            break
    signature_is_adhoc = "Signature=adhoc" in signature_text
    hardened_runtime_enabled = any(
        line.startswith("CodeDirectory ")
        and re.search(r"\([^)]*\bruntime\b[^)]*\)", line) is not None
        for line in signature_text.splitlines()
    )
    developer_id_signed = (
        display.returncode == 0
        and "Authority=Developer ID Application:" in signature_text
        and "TeamIdentifier=55KGF2S5AY" in signature_text
        and not signature_is_adhoc
    )
    verify_lines = verify.stdout.splitlines()
    modified_after_signing_lines = [
        line for line in verify_lines if line.startswith("file modified:")
    ]
    missing_after_signing_lines = [
        line for line in verify_lines if line.startswith("file missing:")
    ]
    modified_after_signing = bool(
        modified_after_signing_lines or missing_after_signing_lines
    )
    result.update(
        {
            "developer_id_signed": developer_id_signed,
            "signature_is_adhoc": signature_is_adhoc,
            "hardened_runtime_enabled": hardened_runtime_enabled,
            "team_identifier": team_identifier,
            "codesign_display_rc": display.returncode,
            "codesign_verify_rc": verify.returncode,
            "packaged_app_modified_after_signing": modified_after_signing,
            "modified_after_signing_file_count": len(modified_after_signing_lines),
            "missing_after_signing_file_count": len(missing_after_signing_lines),
            "modified_after_signing_tail": modified_after_signing_lines[-40:],
            "missing_after_signing_tail": missing_after_signing_lines[-40:],
            "signature_summary_tail": signature_text.splitlines()[-40:],
            "verify_tail": verify_lines[-40:],
        }
    )
    keychain_user_interaction_blocked = any(
        "User interaction is not allowed" in line
        for status in keychain_statuses
        for line in status.get("tail", [])
    )
    codesign_user_interaction_blocked = any(
        "User interaction is not allowed" in line
        for line in result["simple_developer_id_sign_tail"]
    )
    if simple_sign.returncode != 0 and (
        keychain_user_interaction_blocked or codesign_user_interaction_blocked
    ):
        result["signing_blocker_reason"] = (
            "developer_id_keychain_user_interaction_not_allowed"
        )
        result["signing_blocker_reasons"] = [
            "developer_id_keychain_user_interaction_not_allowed"
        ]
        if modified_after_signing:
            result["signing_blocker_reasons"].append(
                "packaged_app_modified_after_signing"
            )
        result["manual_remediation_required"] = True
        result["remediation_summary"] = (
            "Developer ID identities are visible, but codesign cannot use the private "
            "key from this non-interactive process."
        )
        result["remediation_steps"] = [
            "Unlock the signing keychain in an interactive macOS session.",
            "Grant codesign access to the Developer ID private key, for example with security set-key-partition-list using the keychain password outside Codex logs.",
            "Rebuild or reseal the packaged app after bundled runtime sync so codesign --verify --deep --strict passes before notarization.",
            "Rerun the packaged integrity contract and require package_signing_preflight.status=pass before notarization.",
        ]
    elif developer_id_signed and hardened_runtime_enabled and verify.returncode == 0:
        result["status"] = "pass"
        result["signing_blocker_reason"] = None
        result["signing_blocker_reasons"] = []
    elif simple_sign.returncode != 0:
        result["signing_blocker_reason"] = "developer_id_private_key_unusable"
        result["signing_blocker_reasons"] = ["developer_id_private_key_unusable"]
    elif not developer_identity_lines:
        result["signing_blocker_reason"] = "developer_id_identity_missing"
        result["signing_blocker_reasons"] = ["developer_id_identity_missing"]
    elif not developer_id_signed:
        result["signing_blocker_reason"] = "packaged_app_not_developer_id_signed"
        result["signing_blocker_reasons"] = ["packaged_app_not_developer_id_signed"]
    elif not hardened_runtime_enabled:
        result["signing_blocker_reason"] = "packaged_app_missing_hardened_runtime"
        result["signing_blocker_reasons"] = ["packaged_app_missing_hardened_runtime"]
    elif verify.returncode != 0:
        result["signing_blocker_reason"] = (
            "packaged_app_modified_after_signing"
            if modified_after_signing
            else "packaged_app_codesign_verify_failed"
        )
        result["signing_blocker_reasons"] = [result["signing_blocker_reason"]]
    return result


def _run(root: Path, name: str, cwd_rel: Path, cmd: list[str]) -> dict[str, Any]:
    started = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=root / cwd_rel,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "name": name,
        "command": cmd,
        "cwd": str(cwd_rel),
        "returncode": proc.returncode,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "counts": _parse_counts(proc.stdout),
        "stdout_tail": proc.stdout.splitlines()[-100:],
    }


def current_objective_open_requirements(root: Path = Path(".")) -> list[str]:
    fallback = [
        item
        for item in EXPECTED_OPEN_REQUIREMENTS
        if item not in SUITE_DEFERRED_RELEASE_OPEN_REQUIREMENTS
    ]
    try:
        artifact = json.loads(
            (root / CURRENT_OBJECTIVE_DIGEST_ARTIFACT).read_text(encoding="utf-8")
        )
    except Exception:
        return fallback

    open_requirements = []
    for item in artifact.get("requirements", []):
        requirement = item.get("requirement")
        if not requirement or item.get("status") == "pass":
            continue
        if requirement in SUITE_DEFERRED_RELEASE_OPEN_REQUIREMENTS:
            continue
        open_requirements.append(requirement)
    return open_requirements or fallback


def release_gate_failure_is_expected(
    step: dict[str, Any],
    root: Path = Path("."),
) -> bool:
    if step["returncode"] == 0:
        return True
    text = "\n".join(step.get("stdout_tail", []))
    fail_lines = [line for line in text.splitlines() if line.startswith("[FAIL]")]
    forbidden = (
        "bundled python import gate: FAIL",
        "[FAIL] bundled python import gate:",
        "panel typecheck: FAIL",
        "[FAIL] panel typecheck:",
        "panel request/type tests: FAIL",
        "[FAIL] panel request/type tests:",
        "version triple: FAIL",
        "[FAIL] version triple:",
        "[FAIL] twine check dist:",
        "[FAIL] packaged app checks:",
        "Traceback (most recent call last):",
        "ModuleNotFoundError:",
        "FileNotFoundError:",
        "No such file or directory",
    )
    # Shared with the regression suite: judged by membership in the known
    # objective set, not by string-equality against a predicted line that
    # subtracted the deferred objectives and could never match.
    return release_gate_digest_failure_is_known(
        fail_lines,
        text,
        forbidden,
        require_release_ready_line=False,
    )


def dry_release_gate_used_current_objective_digest(step: dict[str, Any]) -> bool:
    text = "\n".join(step.get("stdout_tail", []))
    match = re.search(r"objective proof digest refresh: .*?log=([^\s]+)", text)
    if not match:
        return False
    log_path = Path(match.group(1))
    if not log_path.exists():
        return False
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    expected_abs = str((Path.cwd() / CURRENT_OBJECTIVE_DIGEST_ARTIFACT).resolve())
    expected_rel = str(CURRENT_OBJECTIVE_DIGEST_ARTIFACT)
    return expected_abs in log_text or expected_rel in log_text


def build_artifact(
    root: Path,
    *,
    jang_tools_source: Path | None = None,
) -> dict[str, Any]:
    with _scoped_jang_tools_source(jang_tools_source):
        results = {
            name: _run(root, name, cwd_rel, cmd)
            for name, (cwd_rel, cmd) in COMMANDS.items()
        }
    release_gate_ok = release_gate_failure_is_expected(
        results["release_gate_skip_app"],
        root,
    )
    unit_passed = results["release_gate_unit_contracts"]["counts"]["passed"] or 0
    verifier_output = "\n".join(results["bundled_python_verifier"]["stdout_tail"])
    package_signing_preflight = _package_signing_preflight(root)
    dry_gate_uses_current_objective_digest = dry_release_gate_used_current_objective_digest(
        results["release_gate_skip_app"]
    )
    release_blockers = []
    if package_signing_preflight["status"] != "pass":
        release_blockers.append(
            {
                "id": "packaged_app_developer_id_signing_blocked",
                "status": "open",
                "evidence": "package_signing_preflight",
                "next_proof": "Build and verify a Developer ID signed vMLX.app before notarization.",
            }
        )
    checks = {
        "release_gate_unit_contracts_pass": (
            results["release_gate_unit_contracts"]["returncode"] == 0
            and unit_passed >= MIN_RELEASE_GATE_UNIT_TESTS
        ),
        "bundled_python_verify_passes": results["bundled_python_verifier"]["returncode"] == 0,
        "bundled_engine_version_matches_package_json": (
            results["bundled_python_verifier"]["returncode"] == 0
            and "bundled vmlx_engine version matches package.json" in verifier_output
        ),
        "bundled_engine_hash_parity": (
            results["bundled_python_verifier"]["returncode"] == 0
            and "bundled critical vmlx_engine files match source content" in verifier_output
        ),
        "bundled_jang_tools_hash_parity": (
            results["bundled_python_verifier"]["returncode"] == 0
            and "bundled critical jang_tools files match source content" in verifier_output
        ),
        "bundled_console_scripts_relocatable": (
            results["bundled_python_verifier"]["returncode"] == 0
            and "console-script shebangs are relocatable" in verifier_output
        ),
        "bundled_media_and_jang_dependencies_import": (
            results["bundled_python_verifier"]["returncode"] == 0
            and "bundled-python: all critical imports ok" in verifier_output
        ),
        "packaged_renderer_dsv4_cache_ui_deduped": (
            _check_packaged_renderer_dsv4_cache_ui(root)
        ),
        "packaged_renderer_max_thinking_tokens_wired": (
            _check_packaged_renderer_max_thinking_tokens(root)
        ),
        "packaged_epipe_closed_stream_guards": (
            _check_packaged_epipe_closed_stream_guards(root)
        ),
        "packaged_user_data_isolation_bootstrap": (
            _check_packaged_user_data_isolation_bootstrap(root)
        ),
        "packaged_python_has_no_pycache": _check_packaged_python_has_no_pycache(root),
        "staged_app_engine_hash_parity": _check_staged_app_engine_hash_parity(root),
        "staged_app_engine_source_hash_parity": (
            _check_staged_app_engine_source_hash_parity(root)
        ),
        # 2026-08-16 (ledger row 151): this check was defined but never
        # emitted, which made the manifest's packaged_integrity_matrix
        # unpassable (it expects the key). Static contract on the release
        # scripts — no DMG required.
        "release_dmg_hardened_runtime_entitlements": (
            _check_release_dmg_hardened_runtime_contract(root)
        ),
        "dry_release_gate_fails_only_on_known_objectives": release_gate_ok,
        "dry_release_gate_uses_current_objective_digest": (
            dry_gate_uses_current_objective_digest
        ),
    }
    failed = [
        name
        for name, result in results.items()
        if result["returncode"] != 0 and not (name == "release_gate_skip_app" and release_gate_ok)
    ]
    failed.extend(blocker["id"] for blocker in release_blockers)
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": (
            "pass" if all(checks.values()) and not failed and not release_blockers else "fail"
        ),
        "checks": checks,
        "failed": failed,
        "known_expected_release_gate_open_requirements": EXPECTED_OPEN_REQUIREMENTS,
        "current_effective_release_gate_open_requirements": (
            current_objective_open_requirements(root)
        ),
        "package_signing_preflight": package_signing_preflight,
        "release_blockers": release_blockers,
        "source_hashes": {
            rel: _sha256(root / rel)
            for rel in SOURCE_HASH_FILES
            if (root / rel).exists()
        },
        "results": results,
    }


def artifact_chain_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="run_packaged_integrity_contract.py artifact-chain",
        description="Write or validate the fail-closed vMLX r20 DMG artifact chain.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    private_root_parser = subparsers.add_parser("check-private-root")
    private_root_parser.add_argument("--private-root", type=Path, required=True)

    private_path_parser = subparsers.add_parser("check-private-path")
    private_path_parser.add_argument("--private-root", type=Path, required=True)
    private_path_parser.add_argument("--path", type=Path, required=True)

    private_directory_parser = subparsers.add_parser("create-private-directory")
    private_directory_parser.add_argument("--private-root", type=Path, required=True)
    private_directory_parser.add_argument("--path", type=Path, required=True)
    private_directory_parser.add_argument("--label", required=True)

    file_record_parser = subparsers.add_parser("file-record")
    file_record_parser.add_argument("--path", type=Path, required=True)
    file_record_parser.add_argument("--label", default="release artifact")

    check_file_parser = subparsers.add_parser("check-file")
    check_file_parser.add_argument("--path", type=Path, required=True)
    check_file_parser.add_argument("--expected-sha256", required=True)
    check_file_parser.add_argument("--expected-device", type=int)
    check_file_parser.add_argument("--expected-inode", type=int)
    check_file_parser.add_argument("--expected-size", type=int)
    check_file_parser.add_argument("--label", default="release artifact")

    bundle_runtime_parser = subparsers.add_parser("write-bundle-runtime")
    bundle_runtime_parser.add_argument("--root", type=Path, required=True)
    bundle_runtime_parser.add_argument("--bundle-root", type=Path, required=True)
    bundle_runtime_parser.add_argument("--version", required=True)
    bundle_runtime_parser.add_argument("--private-root", type=Path, required=True)
    bundle_runtime_parser.add_argument(
        "--flavor", choices=R20_ARTIFACT_CHAIN_FLAVORS, required=True
    )
    bundle_runtime_parser.add_argument("--out", type=Path, required=True)

    build_attestation_parser = subparsers.add_parser("write-build-attestation")
    build_attestation_parser.add_argument("--root", type=Path, required=True)
    build_attestation_parser.add_argument("--dist", type=Path, required=True)
    build_attestation_parser.add_argument("--version", required=True)
    build_attestation_parser.add_argument("--preflight", type=Path, required=True)
    build_attestation_parser.add_argument("--private-root", type=Path, required=True)
    build_attestation_parser.add_argument("--out", type=Path, required=True)
    build_attestation_parser.add_argument("--nonce", required=True)
    build_attestation_parser.add_argument("--driver-pid", type=int, required=True)
    for flavor in R20_ARTIFACT_CHAIN_FLAVORS:
        build_attestation_parser.add_argument(
            f"--{flavor}-staged-output", type=Path, required=True
        )
        build_attestation_parser.add_argument(
            f"--{flavor}-extracted-asar", type=Path, required=True
        )
        build_attestation_parser.add_argument(
            f"--{flavor}-hook-attestation", type=Path, required=True
        )
        build_attestation_parser.add_argument(
            f"--{flavor}-hook-attestation-sha256", required=True
        )
        build_attestation_parser.add_argument(
            f"--{flavor}-dmg-parity-attestation", type=Path, required=True
        )
        build_attestation_parser.add_argument(
            f"--{flavor}-dmg-parity-attestation-sha256", required=True
        )

    dmg_parity_parser = subparsers.add_parser("write-dmg-payload-parity")
    dmg_parity_parser.add_argument("--root", type=Path, required=True)
    dmg_parity_parser.add_argument("--dist", type=Path, required=True)
    dmg_parity_parser.add_argument("--version", required=True)
    dmg_parity_parser.add_argument("--private-root", type=Path, required=True)
    dmg_parity_parser.add_argument(
        "--flavor", choices=R20_ARTIFACT_CHAIN_FLAVORS, required=True
    )
    dmg_parity_parser.add_argument("--hook-attestation", type=Path, required=True)
    dmg_parity_parser.add_argument("--expected-hook-sha256", required=True)
    dmg_parity_parser.add_argument("--expected-nonce", required=True)
    dmg_parity_parser.add_argument("--expected-driver-pid", type=int, required=True)
    dmg_parity_parser.add_argument("--mounted-app", type=Path, required=True)
    dmg_parity_parser.add_argument("--extracted-asar", type=Path, required=True)
    dmg_parity_parser.add_argument("--out", type=Path, required=True)

    write_pre_parser = subparsers.add_parser("write-pre-from-driver")
    write_pre_parser.add_argument("--root", type=Path, required=True)
    write_pre_parser.add_argument("--dist", type=Path, required=True)
    write_pre_parser.add_argument("--version", required=True)
    write_pre_parser.add_argument("--private-root", type=Path, required=True)
    write_pre_parser.add_argument("--out", type=Path, required=True)
    write_pre_parser.add_argument("--build-attestation", type=Path, required=True)
    write_pre_parser.add_argument("--expected-build-attestation-sha256", required=True)
    write_pre_parser.add_argument("--expected-nonce", required=True)
    write_pre_parser.add_argument("--expected-driver-pid", type=int, required=True)

    check_pre_parser = subparsers.add_parser("check-pre")
    check_pre_parser.add_argument("--root", type=Path, required=True)
    check_pre_parser.add_argument("--dist", type=Path, required=True)
    check_pre_parser.add_argument("--version", required=True)
    check_pre_parser.add_argument("--private-root", type=Path, required=True)
    check_pre_parser.add_argument("--manifest", type=Path, required=True)
    check_pre_parser.add_argument("--expected-manifest-sha256", required=True)
    check_pre_parser.add_argument("--expected-source-commit", required=True)
    check_pre_parser.add_argument("--expected-source-tree", required=True)
    check_pre_parser.add_argument("--expected-preflight-sha256", required=True)

    snapshot_parser = subparsers.add_parser("create-snapshots")
    snapshot_parser.add_argument("--root", type=Path, required=True)
    snapshot_parser.add_argument("--dist", type=Path, required=True)
    snapshot_parser.add_argument("--version", required=True)
    snapshot_parser.add_argument("--private-root", type=Path, required=True)
    snapshot_parser.add_argument("--manifest", type=Path, required=True)
    snapshot_parser.add_argument("--expected-manifest-sha256", required=True)
    snapshot_parser.add_argument("--expected-source-commit", required=True)
    snapshot_parser.add_argument("--expected-source-tree", required=True)
    snapshot_parser.add_argument("--expected-preflight-sha256", required=True)
    snapshot_parser.add_argument("--snapshot-dir", type=Path, required=True)

    seal_capture_parser = subparsers.add_parser("seal-capture")
    seal_capture_parser.add_argument("--private-root", type=Path, required=True)
    seal_capture_parser.add_argument("--temporary", type=Path, required=True)
    seal_capture_parser.add_argument("--out", type=Path, required=True)

    capture_command_parser = subparsers.add_parser("capture-private-command")
    capture_command_parser.add_argument("--private-root", type=Path, required=True)
    capture_command_parser.add_argument("--result-dir", type=Path, required=True)
    capture_command_parser.add_argument("--output-name", required=True)
    capture_command_parser.add_argument("--stderr-name", required=True)
    capture_command_parser.add_argument("--label", required=True)
    capture_command_parser.add_argument(
        "capture_command",
        nargs=argparse.REMAINDER,
    )

    fresh_apple_parser = subparsers.add_parser("query-apple-online")
    fresh_apple_parser.add_argument("--private-root", type=Path, required=True)
    fresh_apple_parser.add_argument("--capture-dir", type=Path, required=True)
    fresh_apple_parser.add_argument("--submission-id", required=True)
    fresh_apple_parser.add_argument("--expected-dmg-sha256", required=True)
    fresh_apple_parser.add_argument("--expected-archive-name", required=True)
    fresh_apple_parser.add_argument("--expected-team-id", required=True)
    fresh_apple_parser.add_argument("--keychain-profile", required=True)
    fresh_apple_parser.add_argument("--keychain")

    operation_copy_parser = subparsers.add_parser("copy-operation-file")
    operation_copy_parser.add_argument("--private-root", type=Path, required=True)
    operation_copy_parser.add_argument("--source", type=Path, required=True)
    operation_copy_parser.add_argument("--out", type=Path, required=True)
    operation_copy_parser.add_argument("--expected-sha256", required=True)
    operation_copy_parser.add_argument("--writable", action="store_true")
    operation_copy_parser.add_argument("--label", default="release operation")

    operation_install_parser = subparsers.add_parser("install-operation-file")
    operation_install_parser.add_argument("--private-root", type=Path, required=True)
    operation_install_parser.add_argument("--source", type=Path, required=True)
    operation_install_parser.add_argument("--destination", type=Path, required=True)
    operation_install_parser.add_argument("--expected-source-sha256", required=True)
    operation_install_parser.add_argument("--expected-destination-sha256")
    operation_install_parser.add_argument("--label", default="release operation")

    blockmap_parser = subparsers.add_parser("check-recomputed-blockmap")
    blockmap_parser.add_argument("--expected-blockmap", type=Path, required=True)
    blockmap_parser.add_argument("--recomputed-blockmap", type=Path, required=True)
    blockmap_parser.add_argument("--expected-sha256", required=True)

    run_tool_parser = subparsers.add_parser("run-bound-tool-action")
    run_tool_parser.add_argument(
        "--binding-kind",
        choices=("manifest", "plan"),
        required=True,
    )
    run_tool_parser.add_argument("--document", type=Path, required=True)
    run_tool_parser.add_argument("--expected-document-sha256", required=True)
    run_tool_parser.add_argument(
        "--action",
        choices=(
            "node",
            "npm",
            "npx",
            "asar",
            "app-builder",
            "electron-builder",
            "git",
            "shasum",
            "awk",
            "file",
            "find",
        ),
        required=True,
    )
    run_tool_parser.add_argument("--cwd", type=Path)
    run_tool_parser.add_argument("--capture-output", action="store_true")
    run_tool_parser.add_argument("action_arguments", nargs=argparse.REMAINDER)

    find_staged_parser = subparsers.add_parser("find-staged-app")
    find_staged_parser.add_argument("--staged-output", type=Path, required=True)

    check_staged_parser = subparsers.add_parser("check-staged-app")
    check_staged_parser.add_argument("--root", type=Path, required=True)
    check_staged_parser.add_argument("--staged-output", type=Path, required=True)
    check_staged_parser.add_argument("--extracted-asar", type=Path, required=True)
    check_staged_parser.add_argument("--version", required=True)
    check_staged_parser.add_argument(
        "--flavor", choices=R20_ARTIFACT_CHAIN_FLAVORS, required=True
    )

    installed_manifest_parser = subparsers.add_parser(
        "write-installed-release-manifest"
    )
    installed_manifest_parser.add_argument("--root", type=Path, required=True)
    installed_manifest_parser.add_argument("--app", type=Path, required=True)
    installed_manifest_parser.add_argument("--dist", type=Path, required=True)
    installed_manifest_parser.add_argument("--version", required=True)
    installed_manifest_parser.add_argument(
        "--flavor", choices=R20_ARTIFACT_CHAIN_FLAVORS, required=True
    )
    installed_manifest_parser.add_argument(
        "--private-root", type=Path, required=True
    )
    installed_manifest_parser.add_argument("--out", type=Path, required=True)
    installed_manifest_parser.add_argument(
        "--final-manifest", type=Path, required=True
    )
    installed_manifest_parser.add_argument(
        "--expected-final-manifest-sha256", required=True
    )
    installed_manifest_parser.add_argument(
        "--expected-pre-manifest-sha256", required=True
    )
    installed_manifest_parser.add_argument("--expected-source-commit", required=True)
    installed_manifest_parser.add_argument("--expected-source-tree", required=True)
    installed_manifest_parser.add_argument(
        "--expected-preflight-sha256", required=True
    )
    installed_manifest_parser.add_argument(
        "--extracted-asar", type=Path, required=True
    )

    write_final_parser = subparsers.add_parser("write-final")
    write_final_parser.add_argument("--root", type=Path, required=True)
    write_final_parser.add_argument("--dist", type=Path, required=True)
    write_final_parser.add_argument("--version", required=True)
    write_final_parser.add_argument("--pre-manifest", type=Path, required=True)
    write_final_parser.add_argument("--expected-pre-manifest-sha256", required=True)
    write_final_parser.add_argument("--expected-source-commit", required=True)
    write_final_parser.add_argument("--expected-source-tree", required=True)
    write_final_parser.add_argument("--expected-preflight-sha256", required=True)
    write_final_parser.add_argument("--private-root", type=Path, required=True)
    write_final_parser.add_argument("--out", type=Path, required=True)
    for flavor in R20_ARTIFACT_CHAIN_FLAVORS:
        write_final_parser.add_argument(
            f"--{flavor}-submission-id", required=True
        )
        write_final_parser.add_argument(
            f"--{flavor}-snapshot-dmg", type=Path, required=True
        )

    check_final_parser = subparsers.add_parser("check-final")
    check_final_parser.add_argument("--root", type=Path, required=True)
    check_final_parser.add_argument("--dist", type=Path, required=True)
    check_final_parser.add_argument("--version", required=True)
    check_final_parser.add_argument("--private-root", type=Path, required=True)
    check_final_parser.add_argument("--manifest", type=Path, required=True)
    check_final_parser.add_argument("--expected-manifest-sha256", required=True)
    check_final_parser.add_argument("--expected-pre-manifest-sha256", required=True)
    check_final_parser.add_argument("--expected-source-commit", required=True)
    check_final_parser.add_argument("--expected-source-tree", required=True)
    check_final_parser.add_argument("--expected-preflight-sha256", required=True)

    mounted_parser = subparsers.add_parser("check-mounted-app")
    mounted_parser.add_argument("--root", type=Path, required=True)
    mounted_parser.add_argument("--dist", type=Path, required=True)
    mounted_parser.add_argument("--version", required=True)
    mounted_parser.add_argument("--private-root", type=Path, required=True)
    mounted_parser.add_argument("--final-manifest", type=Path, required=True)
    mounted_parser.add_argument("--expected-final-manifest-sha256", required=True)
    mounted_parser.add_argument("--expected-pre-manifest-sha256", required=True)
    mounted_parser.add_argument("--expected-source-commit", required=True)
    mounted_parser.add_argument("--expected-source-tree", required=True)
    mounted_parser.add_argument("--expected-preflight-sha256", required=True)
    mounted_parser.add_argument(
        "--flavor", choices=R20_ARTIFACT_CHAIN_FLAVORS, required=True
    )
    mounted_parser.add_argument("--mounted-app", type=Path, required=True)
    mounted_parser.add_argument("--extracted-asar", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "check-private-root":
            result: Any = {
                "private_root": str(ensure_private_evidence_root(args.private_root)),
            }
        elif args.command == "check-private-path":
            result = {
                "private_path": str(
                    _assert_within_private_root(
                        args.private_root,
                        args.path,
                        label="private release path",
                    )
                ),
            }
        elif args.command == "create-private-directory":
            result = {
                "private_directory": str(
                    create_private_directory(
                        private_root=args.private_root,
                        directory=args.path,
                        label=args.label,
                    )
                ),
            }
        elif args.command == "file-record":
            result = _safe_regular_file(args.path, label=args.label)
        elif args.command == "check-file":
            result = validate_file_identity(
                path=args.path,
                expected_sha256=args.expected_sha256,
                expected_device=args.expected_device,
                expected_inode=args.expected_inode,
                expected_size=args.expected_size,
                label=args.label,
            )
        elif args.command == "write-bundle-runtime":
            result = write_bundle_runtime_attestation(
                root=args.root,
                bundle_root=args.bundle_root,
                version=args.version,
                private_root=args.private_root,
                flavor=args.flavor,
                output_path=args.out,
            )
        elif args.command == "write-build-attestation":
            result = write_build_driver_attestation(
                root=args.root,
                dist_dir=args.dist,
                version=args.version,
                preflight_path=args.preflight,
                private_root=args.private_root,
                output_path=args.out,
                nonce=args.nonce,
                driver_pid=args.driver_pid,
                staged_outputs={
                    flavor: getattr(args, f"{flavor}_staged_output")
                    for flavor in R20_ARTIFACT_CHAIN_FLAVORS
                },
                extracted_asars={
                    flavor: getattr(args, f"{flavor}_extracted_asar")
                    for flavor in R20_ARTIFACT_CHAIN_FLAVORS
                },
                hook_attestations={
                    flavor: (
                        getattr(args, f"{flavor}_hook_attestation"),
                        getattr(args, f"{flavor}_hook_attestation_sha256"),
                    )
                    for flavor in R20_ARTIFACT_CHAIN_FLAVORS
                },
                dmg_parity_attestations={
                    flavor: (
                        getattr(args, f"{flavor}_dmg_parity_attestation"),
                        getattr(args, f"{flavor}_dmg_parity_attestation_sha256"),
                    )
                    for flavor in R20_ARTIFACT_CHAIN_FLAVORS
                },
            )
        elif args.command == "write-dmg-payload-parity":
            result = write_dmg_payload_parity_attestation(
                root=args.root,
                dist_dir=args.dist,
                version=args.version,
                private_root=args.private_root,
                flavor=args.flavor,
                hook_attestation_path=args.hook_attestation,
                expected_hook_sha256=args.expected_hook_sha256,
                expected_nonce=args.expected_nonce,
                expected_driver_pid=args.expected_driver_pid,
                mounted_app=args.mounted_app,
                extracted_asar=args.extracted_asar,
                output_path=args.out,
            )
        elif args.command == "write-pre-from-driver":
            result = write_pre_notary_artifact_manifest(
                root=args.root,
                dist_dir=args.dist,
                version=args.version,
                private_root=args.private_root,
                output_path=args.out,
                build_attestation_path=args.build_attestation,
                expected_build_attestation_sha256=(
                    args.expected_build_attestation_sha256
                ),
                expected_nonce=args.expected_nonce,
                expected_driver_pid=args.expected_driver_pid,
            )
        elif args.command == "check-pre":
            result = validate_pre_notary_artifact_manifest(
                root=args.root,
                dist_dir=args.dist,
                version=args.version,
                private_root=args.private_root,
                manifest_path=args.manifest,
                expected_manifest_sha256=args.expected_manifest_sha256,
                expected_source_commit=args.expected_source_commit,
                expected_source_tree=args.expected_source_tree,
                expected_preflight_sha256=args.expected_preflight_sha256,
            )
        elif args.command == "create-snapshots":
            result = create_pre_notary_snapshots(
                root=args.root,
                dist_dir=args.dist,
                version=args.version,
                private_root=args.private_root,
                manifest_path=args.manifest,
                expected_manifest_sha256=args.expected_manifest_sha256,
                expected_source_commit=args.expected_source_commit,
                expected_source_tree=args.expected_source_tree,
                expected_preflight_sha256=args.expected_preflight_sha256,
                snapshot_dir=args.snapshot_dir,
            )
        elif args.command == "seal-capture":
            result = seal_private_capture(
                private_root=args.private_root,
                temporary_path=args.temporary,
                output_path=args.out,
            )
        elif args.command == "capture-private-command":
            capture_command = list(args.capture_command)
            if capture_command[:1] == ["--"]:
                capture_command = capture_command[1:]
            result = capture_private_command(
                private_root=args.private_root,
                result_dir=args.result_dir,
                output_name=args.output_name,
                stderr_name=args.stderr_name,
                label=args.label,
                command=capture_command,
            )
        elif args.command == "query-apple-online":
            result = query_apple_notary_fresh(
                private_root=args.private_root,
                capture_dir=args.capture_dir,
                submission_id=args.submission_id,
                expected_dmg_sha256=args.expected_dmg_sha256,
                expected_archive_name=args.expected_archive_name,
                expected_team_id=args.expected_team_id,
                keychain_profile=args.keychain_profile,
                keychain=args.keychain,
            )
        elif args.command == "copy-operation-file":
            result = create_private_operation_copy(
                private_root=args.private_root,
                source=args.source,
                destination=args.out,
                expected_sha256=args.expected_sha256,
                writable=args.writable,
                label=args.label,
            )
        elif args.command == "install-operation-file":
            result = install_private_operation_result(
                private_root=args.private_root,
                source=args.source,
                destination=args.destination,
                expected_source_sha256=args.expected_source_sha256,
                expected_destination_sha256=args.expected_destination_sha256,
                label=args.label,
            )
        elif args.command == "check-recomputed-blockmap":
            result = validate_recomputed_blockmap(
                expected_blockmap=args.expected_blockmap,
                recomputed_blockmap=args.recomputed_blockmap,
                expected_sha256=args.expected_sha256,
            )
        elif args.command == "run-bound-tool-action":
            result = run_bound_tool_action(
                document_path=args.document,
                expected_document_sha256=args.expected_document_sha256,
                binding_kind=args.binding_kind,
                action=args.action,
                arguments=args.action_arguments,
                cwd=args.cwd,
                capture_output=args.capture_output,
            )
        elif args.command == "find-staged-app":
            result = {"app": str(find_exact_staged_app(args.staged_output))}
        elif args.command == "check-staged-app":
            result = validate_staged_app_parity(
                root=args.root,
                staged_output=args.staged_output,
                extracted_asar=args.extracted_asar,
                version=args.version,
                flavor=args.flavor,
            )
        elif args.command == "write-installed-release-manifest":
            result = write_installed_release_manifest(
                root=args.root,
                app=args.app,
                dist_dir=args.dist,
                version=args.version,
                flavor=args.flavor,
                private_root=args.private_root,
                output_path=args.out,
                final_manifest_path=args.final_manifest,
                expected_final_manifest_sha256=(
                    args.expected_final_manifest_sha256
                ),
                expected_pre_manifest_sha256=args.expected_pre_manifest_sha256,
                expected_source_commit=args.expected_source_commit,
                expected_source_tree=args.expected_source_tree,
                expected_preflight_sha256=args.expected_preflight_sha256,
                extracted_asar=args.extracted_asar,
            )
        elif args.command == "write-final":
            result = write_final_notary_artifact_manifest(
                root=args.root,
                dist_dir=args.dist,
                version=args.version,
                pre_notary_manifest_path=args.pre_manifest,
                expected_pre_manifest_sha256=args.expected_pre_manifest_sha256,
                expected_source_commit=args.expected_source_commit,
                expected_source_tree=args.expected_source_tree,
                expected_preflight_sha256=args.expected_preflight_sha256,
                private_root=args.private_root,
                output_path=args.out,
                submission_ids={
                    flavor: getattr(args, f"{flavor}_submission_id")
                    for flavor in R20_ARTIFACT_CHAIN_FLAVORS
                },
                submitted_snapshot_paths={
                    flavor: getattr(args, f"{flavor}_snapshot_dmg")
                    for flavor in R20_ARTIFACT_CHAIN_FLAVORS
                },
            )
        elif args.command == "check-final":
            result = validate_final_notary_artifact_manifest(
                root=args.root,
                dist_dir=args.dist,
                version=args.version,
                private_root=args.private_root,
                manifest_path=args.manifest,
                expected_manifest_sha256=args.expected_manifest_sha256,
                expected_pre_manifest_sha256=args.expected_pre_manifest_sha256,
                expected_source_commit=args.expected_source_commit,
                expected_source_tree=args.expected_source_tree,
                expected_preflight_sha256=args.expected_preflight_sha256,
            )
        elif args.command == "check-mounted-app":
            result = validate_mounted_app_against_final_manifest(
                root=args.root,
                dist_dir=args.dist,
                version=args.version,
                private_root=args.private_root,
                final_manifest_path=args.final_manifest,
                expected_final_manifest_sha256=(
                    args.expected_final_manifest_sha256
                ),
                expected_pre_manifest_sha256=args.expected_pre_manifest_sha256,
                expected_source_commit=args.expected_source_commit,
                expected_source_tree=args.expected_source_tree,
                expected_preflight_sha256=args.expected_preflight_sha256,
                flavor=args.flavor,
                mounted_app=args.mounted_app,
                extracted_asar=args.extracted_asar,
            )
        else:  # pragma: no cover - argparse makes this unreachable
            raise ArtifactChainError(f"unsupported artifact-chain command: {args.command}")
    except ArtifactChainError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = {
        key: value
        for key, value in result.items()
        if key != "payload"
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--jang-tools-source",
        type=Path,
        default=None,
        help=(
            "Clean jang-tools source checkout used for bundled hash checks. "
            "Sets both VMLX_JANG_TOOLS_SOURCE and legacy "
            "VMLINUX_JANG_TOOLS_SOURCE while running child gates."
        ),
    )
    args = parser.parse_args()

    artifact = build_artifact(args.root, jang_tools_source=args.jang_tools_source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    print(f"status={artifact['status']}")
    print("failed=" + json.dumps(artifact["failed"]))
    for name, result in artifact["results"].items():
        counts = result["counts"]
        print(
            f"{name}: rc={result['returncode']} "
            f"passed={counts['passed']} skipped={counts['skipped']} "
            f"deselected={counts['deselected']}"
        )
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "artifact-chain":
        raise SystemExit(artifact_chain_main(sys.argv[2:]))
    raise SystemExit(main())
