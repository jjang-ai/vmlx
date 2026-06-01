#!/usr/bin/env python3
"""No-heavy audit for open public app/runtime issues.

This indexes the current source/proof boundary for the public issue set that
does not require launching heavy models. It is not a release-ready signal by
itself; packaging, signing, DSV4, and the full real UI matrix remain owned by
the main release manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path(
    "build/current-public-app-issue-audit-20260601-sequoia-download-minimax-gemma.json"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - audit reports false checks, not tracebacks
        return {}
    return payload if isinstance(payload, dict) else {}


def _surfaces(path: Path) -> set[str]:
    payload = _load_json(path)
    surfaces = payload.get("provenSurfaces")
    if not isinstance(surfaces, list):
        return set()
    return {str(item) for item in surfaces}


def _issue169_checks(root: Path) -> dict[str, bool]:
    bundle = _read(root / "panel/scripts/bundle-python.sh")
    release = _read(root / "panel/scripts/build-release-dmgs.sh")
    pkg = _load_json(root / "panel/package.json")
    engine_tests = _read(root / "tests/test_engine_audit.py")
    return {
        "dual_public_dmg_flavors": (
            'build_one "sequoia" "compat"' in release
            and 'build_one "tahoe" "native"' in release
            and 'sequoia) wheel_tag="macosx_14_0_arm64"' in release
            and 'tahoe) wheel_tag="macosx_26_0_arm64"' in release
        ),
        "compat_wheel_default": (
            'auto|compat|sonoma|sequoia|"")' in bundle
            and 'echo "macosx_14_0_arm64"' in bundle
            and 'tahoe|native|m5)' in bundle
            and 'echo "macosx_26_0_arm64"' in bundle
        ),
        "minimum_system_version_allows_sequoia": (
            str(pkg.get("build", {}).get("mac", {}).get("minimumSystemVersion"))
            == "14.5.0"
        ),
        "wrong_flavor_error_tested": (
            "test_macos_compat_error_points_to_sequoia_dmg_for_tahoe_wheels"
            in engine_tests
            and "Failed to load the default metallib" in engine_tests
        ),
    }


def _issue117_checks(root: Path) -> dict[str, bool]:
    issue179 = _load_json(
        root / "build/current-issue179-minimax-k-root-cause-audit-20260527.json"
    )
    release_manifest = _read(root / "tests/cross_matrix/release_regression_manifest.py")
    parser_contract = _read(root / "tests/cross_matrix/run_parser_registry_contract.py")
    minimax_real_ui = (
        root
        / "docs/internal/agent-notes/current-real-ui-live-model-minimax-m27-small-responses-stricttools-cachecontrols-20260530-proof.json"
    )
    surfaces = _surfaces(minimax_real_ui)
    return {
        "issue179_root_cause_audit_passes": issue179.get("status") == "pass",
        "issue179_cancel_probe_present": (
            root
            / "build/current-issue179-minimax-k-responses-cancel-probe-installed-20260527.json"
        ).exists(),
        "minimax_live_ui_artifacts_indexed": (
            minimax_real_ui.exists()
            and "MiniMax-M2.7-Small-JANGTQ" in _read(minimax_real_ui)
            and "cache_hit_telemetry" in surfaces
            and "parser_leak_check" in surfaces
            and "language_leak_check" in surfaces
            and "current-real-ui-live-model-minimax-m27-small" in release_manifest
        ),
        "minimax_reasoning_parser_guarded": (
            "passes MiniMax through the registered minimax_m2 reasoning parser"
            in parser_contract
            and "uses the registered MiniMax reasoning parser even when bundle sidecars say qwen3"
            in parser_contract
        ),
    }


def _issue180_checks(root: Path) -> dict[str, bool]:
    release_manifest = _read(root / "tests/cross_matrix/release_regression_manifest.py")
    minimax_real_ui = (
        root
        / "docs/internal/agent-notes/current-real-ui-live-model-minimax-m27-small-responses-stricttools-cachecontrols-20260530-proof.json"
    )
    payload = _load_json(minimax_real_ui)
    surfaces = _surfaces(minimax_real_ui)
    chat = payload.get("chat")
    chat = chat if isinstance(chat, dict) else {}
    reasoning_numeric = chat.get("reasoningNumericRunCount")
    reasoning_cjk = chat.get("reasoningCjkLeakCount")
    reasoning_korean = chat.get("reasoningKoreanLeakCount")
    return {
        "minimax_small_stricttools_real_ui_indexed": (
            minimax_real_ui.exists()
            and payload.get("modelName") == "MiniMax-M2.7-Small-JANGTQ"
            and payload.get("servedModel") == "MiniMax-M2.7-Small-JANGTQ"
            and "responses_api" in surfaces
            and "server_cache_controls" in surfaces
            and "parser_leak_check" in surfaces
            and "language_leak_check" in surfaces
            and "current-real-ui-live-model-minimax-m27-small-responses-stricttools-cachecontrols-20260530"
            in release_manifest
        ),
        "minimax_small_numeric_garbage_guarded": (
            "reasoning_language_or_numeric_leak" in release_manifest
            and "numeric/list-like garbage leaked into reasoning segments"
            in _read(root / "tests/test_release_regression_manifest.py")
            and reasoning_numeric == 0
            and reasoning_cjk == 0
            and reasoning_korean == 0
        ),
    }


def _issue118_checks(root: Path) -> dict[str, bool]:
    models = _read(root / "panel/src/main/ipc/models.ts")
    hf_settings = _read(root / "panel/src/shared/hfSettings.ts")
    hf_tests = _read(root / "panel/tests/hf-settings.test.ts")
    engine_tests = _read(root / "tests/test_engine_audit.py")
    return {
        "hf_endpoint_and_token_normalized": (
            "normalizeHfEndpointSetting" in hf_settings
            and "normalizeHfTokenSetting" in hf_settings
            and "canonicalizes valid HF-compatible endpoints" in hf_tests
        ),
        "download_worker_clears_raw_endpoint": (
            "HF_ENDPOINT: undefined" in models
            and "do not let a stale shell or persisted HF_ENDPOINT" in models
            and "const hfEndpoint = getConfiguredHfEndpoint() ?? \"\"" in models
        ),
        "api_retries_without_stale_auth": (
            "retrying ${baseUrl} without auth" in models
            and "retryWithoutAuth" in models
            and "isHfAuthStatus(response.status)" in models
        ),
        "download_script_falls_back_endpoint_and_auth": (
            "def _attempts():" in models
            and "for active_endpoint, active_token in _attempts():" in models
            and "print(json.dumps({'type':'fallback'" in models
        ),
        "focused_regression_tests_present": (
            "Issue #118: stale saved HF mirrors or tokens" in engine_tests
            and "test_panel_hf_api_retries_public_requests_without_stale_auth"
            in engine_tests
        ),
    }


def _issue119_checks(root: Path) -> dict[str, bool]:
    stress = _load_json(
        root
        / "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-installed-app-after-mixed-swa-telemetry-20260524.json"
    )
    gemma_responses = (
        root
        / "docs/internal/agent-notes/current-real-ui-live-model-gemma4-responses-stricttools-max768-visible-20260531-proof.json"
    )
    gemma_cache = (
        root
        / "docs/internal/agent-notes/current-real-ui-live-model-gemma4-cachecontrols-l2storage-20260531-proof.json"
    )
    response_surfaces = _surfaces(gemma_responses)
    cache_surfaces = _surfaces(gemma_cache)
    objective = _read(root / "tests/cross_matrix/summarize_objective_proof.py")
    production = _read(root / "tests/cross_matrix/run_production_family_audit.py")
    return {
        "gemma26_memory_stress_artifact_present": stress.get("status") == "pass",
        "gemma26_real_ui_artifacts_indexed": (
            gemma_responses.exists()
            and gemma_cache.exists()
            and "Gemma-4-26B-A4B-it-JANG_4M-CRACK" in _read(gemma_responses)
            and "Gemma-4-26B-A4B-it-JANG_4M-CRACK" in _read(gemma_cache)
            and "responses_api" in response_surfaces
            and "language_leak_check" in response_surfaces
            and "l2_disk_storage" in cache_surfaces
            and "cache_hit_telemetry" in cache_surfaces
        ),
        "gemma26_mixed_swa_cache_guarded": (
            "Gemma4 mixed-SWA" in objective
            and "Gemma4/mixed-SWA must stay on typed KV/RotatingKV cache paths"
            in _read(root / "tests/cross_matrix/run_cache_architecture_contract.py")
        ),
        "gemma26_production_family_row_present": (
            "Gemma-4-26B-A4B-it JANG_4M CRACK" in production
            and "mixed-SWA app-engine speed floor" in objective
        ),
    }


def build_audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    issues: dict[str, dict[str, Any]] = {
        "169": {
            "repo": "jjang-ai/vmlx",
            "title": (
                "Session fails on macOS Sequoia: bundled metallib requires "
                "Metal language version 4.0"
            ),
            "checks": _issue169_checks(root),
            "release_clearance": (
                "source_dual_dmg_metal_compat_route_guarded_packaging_still_gated"
            ),
        },
        "117": {
            "repo": "jjang-ai/mlxstudio",
            "title": (
                "MiniMax-M2.7 JANGTQ_K gibberish output regression introduced "
                "between v1.5.32 and v1.5.33"
            ),
            "checks": _issue117_checks(root),
            "release_clearance": (
                "mapped_to_minimax_k_issue179_live_reporter_prompt_boundary"
            ),
        },
        "180": {
            "repo": "jjang-ai/vmlx",
            "title": (
                "MiniMax-M2.7-Small-JANGTQ produces gibberish output "
                "(thinking block shows numeric sequences)"
            ),
            "checks": _issue180_checks(root),
            "release_clearance": (
                "mapped_to_minimax_small_real_ui_language_numeric_guard"
            ),
        },
        "118": {
            "repo": "jjang-ai/mlxstudio",
            "title": "Studio 1.5.42 cannot download models from the Hub",
            "checks": _issue118_checks(root),
            "release_clearance": (
                "source_gui_download_endpoint_and_stale_auth_fallback_guarded"
            ),
        },
        "119": {
            "repo": "jjang-ai/mlxstudio",
            "title": "v1.5.3 Python crashes",
            "checks": _issue119_checks(root),
            "release_clearance": (
                "source_and_live_gemma26_memory_runtime_guarded_release_package_pending"
            ),
        },
    }
    focused_failures: list[str] = []
    for number, issue in issues.items():
        checks = issue["checks"]
        issue["focused_source_slice"] = "pass" if all(checks.values()) else "fail"
        if issue["focused_source_slice"] != "pass":
            focused_failures.append(number)

    return {
        "artifact": "",
        "status": "fail" if focused_failures else "pass",
        "issues": issues,
        "focused_failures": focused_failures,
        "release_boundary": (
            "Public issue slices #169, #180, and mlxstudio #117-#119 have "
            "focused source/proof coverage here. This does not clear the full "
            "release; Developer ID signing/notarization, DSV4 exactness, and "
            "real UI matrix blockers remain owned by the release manifest."
        ),
    }


def write_audit(root: Path, out: Path) -> dict[str, Any]:
    audit = build_audit(root)
    audit["artifact"] = str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    audit = write_audit(args.root, args.out)
    print(json.dumps({"status": audit["status"], "out": str(args.out)}, sort_keys=True))
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
