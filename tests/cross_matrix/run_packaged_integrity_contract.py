#!/usr/bin/env python3
"""Run no-publish packaged integrity contracts.

This gate does not build, tag, upload, notarize, or update release feeds. It
checks the release-gate script contracts, bundled Python verifier, and the dry
release gate behavior. The current dry gate is allowed to fail only for the
known objective digest rows.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path(
    "build/current-packaged-integrity-contract-20260602-developer-id-staged-signing.json"
)
EXPECTED_OPEN_REQUIREMENTS = [
    "MiniMax-M2.7-JANGTQ_K reporter parity/root cause is release-cleared",
    "Real Electron UI cross-family live model matrix is release-cleared",
    "DSV4 long-output/code/file-generation quality is release-cleared",
]
CURRENT_OBJECTIVE_DIGEST_ARTIFACT = Path(
    "build/current-objective-proof-audit-20260602-cache-detail-zero-cached.json"
)
MIN_RELEASE_GATE_UNIT_TESTS = 34
PACKAGED_RENDERER_ASAR = Path(
    "panel/release/mac-arm64/vMLX.app/Contents/Resources/app.asar"
)
PACKAGED_PYTHON_ROOT = Path(
    "panel/release/mac-arm64/vMLX.app/Contents/Resources/bundled-python/python"
)
PACKAGED_APP = Path("panel/release/mac-arm64/vMLX.app")
DEVELOPER_ID_IDENTITY = "D4DBBCB52F666D03F0A5154BFFEA2227BEE8FC7C"
SIGNING_KEYCHAINS = (
    Path("~/Library/Keychains/vmlx-build.keychain-db").expanduser(),
    Path("~/Library/Keychains/build.keychain-db").expanduser(),
    Path("~/Library/Keychains/login.keychain-db").expanduser(),
)
PACKAGED_RENDERER_REQUIRED_DSV4_CACHE_UI_STRINGS = (
    b"DSV4 Native Composite Prefix Cache",
    b"DSV4 CSA/HCA Pool Codec",
    b"DSV4_POOL_QUANT=1 native CSA/HCA pool codec",
)
PACKAGED_RENDERER_FORBIDDEN_DSV4_CACHE_UI_STRINGS = (
    b"DSV4 Native Cache",
    b"DSV4 Composite Prefix Cache",
    b"DSV4 Pool Quantization",
    b"DSV4 Flash composite prefix cache is disabled",
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
    "utils/tokenizer.py",
    "chat_templates/gemma4.jinja",
    "config/defaults.yaml",
    "metal/codebook_matvec.metal",
    "metal/codebook_moe.metal",
)

SOURCE_HASH_FILES = (
    "panel/scripts/release-gate-python-app.py",
    "panel/scripts/verify-bundled-python.sh",
    "panel/scripts/bundle-python.sh",
    "panel/scripts/build-release-dmgs.sh",
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
    "tests/cross_matrix/summarize_objective_proof.py",
    "tests/test_objective_proof_digest.py",
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_counts(output: str) -> dict[str, int | None]:
    passed = None
    skipped = None
    deselected = None
    match = re.search(r"Tests\s+(\d+) passed", output)
    if match:
        passed = int(match.group(1))
    match = re.search(r"(\d+) passed", output)
    if match and passed is None:
        passed = int(match.group(1))
    match = re.search(r"(\d+) skipped", output)
    if match:
        skipped = int(match.group(1))
    match = re.search(r"(\d+) deselected", output)
    if match:
        deselected = int(match.group(1))
    return {"passed": passed, "skipped": skipped, "deselected": deselected}


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
    return (
        "codesign --force --deep" in final_sign_block
        and "--options runtime" in final_sign_block
        and "--entitlements" in final_sign_block
        and "entitlements.mac.plist" in final_sign_block
        and "codesign --verify --deep --strict" in final_sign_block
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
        "Authority=Developer ID Application: ShieldStack LLC (55KGF2S5AY)",
        "TeamIdentifier=55KGF2S5AY",
        "Signature=adhoc",
        "xcrun stapler validate",
        "spctl --assess --type open --context context:primary-signature",
        "shasum -a 256",
    )
    return all(fragment in text for fragment in required_fragments)


def _check_release_dmg_notarization_submit_contract(root: Path) -> bool:
    script = root / "panel/scripts/notarize-release-dmgs.sh"
    if not script.exists():
        return False
    text = script.read_text(encoding="utf-8", errors="replace")
    required_fragments = (
        "sequoia tahoe",
        "VMLINUX_NOTARY_KEYCHAIN_PROFILE",
        "VMLINUX_NOTARY_KEYCHAIN",
        "notarytool_args",
        "--keychain",
        "codesign --verify",
        "codesign -dv",
        "require_developer_id_signature",
        "Authority=Developer ID Application: ShieldStack LLC (55KGF2S5AY)",
        "TeamIdentifier=55KGF2S5AY",
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
        "shasum -a 256",
    )
    return all(fragment in text for fragment in required_fragments)


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
    if developer_id_signed and hardened_runtime_enabled and verify.returncode == 0:
        result["status"] = "pass"
        result["signing_blocker_reason"] = None
        result["signing_blocker_reasons"] = []
    elif simple_sign.returncode != 0 and (
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


def release_gate_failure_is_expected(step: dict[str, Any]) -> bool:
    if step["returncode"] == 0:
        return True
    text = "\n".join(step.get("stdout_tail", []))
    fail_lines = [line for line in text.splitlines() if line.startswith("[FAIL]")]
    expected_digest = (
        "[FAIL] objective proof digest: "
        + "; ".join(EXPECTED_OPEN_REQUIREMENTS)
    )
    expected_release_ready_prefix = "[FAIL] release-ready manifest: exit=1;"
    forbidden = (
        "bundled python import gate: FAIL",
        "panel typecheck: FAIL",
        "panel request/type tests: FAIL",
        "version triple: FAIL",
        "Traceback (most recent call last):",
        "ModuleNotFoundError:",
        "FileNotFoundError:",
        "No such file or directory",
    )
    return (
        len(fail_lines) == 2
        and fail_lines[0] == expected_digest
        and fail_lines[1].startswith(expected_release_ready_prefix)
        and not any(item in text for item in forbidden)
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
    release_gate_ok = release_gate_failure_is_expected(results["release_gate_skip_app"])
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
        "release_dmg_hardened_runtime_entitlements": (
            _check_release_dmg_hardened_runtime_contract(root)
        ),
        "release_dmg_notarization_verifier_contract": (
            _check_release_dmg_notarization_verifier_contract(root)
        ),
        "release_dmg_notarization_submit_contract": (
            _check_release_dmg_notarization_submit_contract(root)
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
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "pass" if all(checks.values()) and not failed else "fail",
        "checks": checks,
        "failed": failed,
        "known_expected_release_gate_open_requirements": EXPECTED_OPEN_REQUIREMENTS,
        "package_signing_preflight": package_signing_preflight,
        "release_blockers": release_blockers,
        "source_hashes": {
            rel: _sha256(root / rel)
            for rel in SOURCE_HASH_FILES
            if (root / rel).exists()
        },
        "results": results,
    }


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
    raise SystemExit(main())
