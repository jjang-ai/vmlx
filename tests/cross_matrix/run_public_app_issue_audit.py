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
import plistlib
import subprocess
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path(
    "build/current-public-app-issue-audit-after-public-v1556-scan-20260606.json"
)
INSTALLED_APP_ASAR = Path("/Applications/vMLX.app/Contents/Resources/app.asar")
INSTALLED_APP = Path("/Applications/vMLX.app")
INSTALLED_APP_PYTHON = Path(
    "/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3"
)
STAGED_SEQUOIA_APP = Path("panel/release/sequoia-app/mac-arm64/vMLX.app")
STAGED_TAHOE_APP = Path("panel/release/tahoe-app/mac-arm64/vMLX.app")
TOOL_CALL_CONTRACT = Path(
    "build/current-tool-call-contract-20260528-tool-parser-loop-matrix.json"
)
INSTALLED_APP_RUNTIME_PARITY = Path(
    "build/current-installed-app-runtime-parity-audit-after-installed-app-rebuild-20260606.json"
)
REASONING_TEMPLATE_CONTRACT = Path(
    "build/current-reasoning-template-contract-20260526-settings-audit.json"
)
PACKAGED_INTEGRITY_CONTRACT = Path(
    "build/current-packaged-integrity-contract-after-noheavy-pointer-refresh-20260606.json"
)
GEMMA4_INSTALLED_SPEED_ARTIFACTS = (
    Path(
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-20260525.json"
    ),
    Path(
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-repeat-20260525.json"
    ),
    Path(
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-trace-20260525.json"
    ),
    Path(
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-triple-nocache-256-streaming-20260525.json"
    ),
)
GEMMA4_CURRENT_INSTALLED_SPEED_ARTIFACT = Path(
    "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-issue115-installed-app-20260601.json"
)
QWEN35_INSTALLED_SPEED_ARTIFACT = Path(
    "build/current-decode-speed-live-qwen35-4bit-issue115-installed-app-after-decode-position-20260601.json"
)
ISSUE179_ROOT_CAUSE_AUDIT = Path(
    "build/current-issue179-minimax-k-root-cause-audit-after-reporter-hash-refresh-20260606.json"
)
ISSUE179_RESPONSES_CANCEL_PROOF = Path(
    "build/current-issue179-minimax-k-responses-cancel-probe-20260606-live-refresh.json"
)
ISSUE179_RESPONSES_CANCEL_PREFLIGHT = Path(
    "build/current-issue179-minimax-k-responses-cancel-probe-memory-preflight-20260602-local-ready-check.json"
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


def _app_minimum_system_version(app: Path) -> str:
    info_plist = app / "Contents/Info.plist"
    if not info_plist.exists():
        return ""
    try:
        with info_plist.open("rb") as handle:
            payload = plistlib.load(handle)
    except Exception:  # noqa: BLE001 - audit reports false checks, not tracebacks
        return ""
    return str(payload.get("LSMinimumSystemVersion", ""))


def _wheel_tags(app: Path) -> dict[str, str]:
    site_packages = (
        app
        / "Contents/Resources/bundled-python/python/lib/python3.12/site-packages"
    )
    tags: dict[str, str] = {}
    for package, pattern in (
        ("mlx", "mlx-*.dist-info"),
        ("mlx_metal", "mlx_metal-*.dist-info"),
    ):
        matches = sorted(site_packages.glob(pattern))
        if not matches:
            tags[package] = ""
            continue
        wheel = matches[0] / "WHEEL"
        tag = ""
        for line in _read(wheel).splitlines():
            if line.startswith("Tag: "):
                tag = line.removeprefix("Tag: ").strip()
                break
        tags[package] = tag
    return tags


def _app_runtime_flavor_matches(
    app: Path,
    *,
    minimum_system_version: str,
    mlx_tag: str,
    mlx_metal_tag: str,
) -> bool:
    tags = _wheel_tags(app)
    return (
        _app_minimum_system_version(app) == minimum_system_version
        and tags.get("mlx") == mlx_tag
        and tags.get("mlx_metal") == mlx_metal_tag
    )


def _surfaces(path: Path) -> set[str]:
    payload = _load_json(path)
    surfaces = payload.get("provenSurfaces")
    if not isinstance(surfaces, list):
        return set()
    return {str(item) for item in surfaces}


def _read_installed_asar_file(root: Path, relpath: str) -> str:
    if not INSTALLED_APP_ASAR.exists():
        return ""
    panel_dir = root / "panel"
    proc = subprocess.run(
        [
            "node",
            "-e",
            (
                "const asar=require('@electron/asar');"
                "process.stdout.write(asar.extractFile(process.argv[1], process.argv[2])"
                ".toString('utf8'));"
            ),
            str(INSTALLED_APP_ASAR),
            relpath,
        ],
        cwd=panel_dir if panel_dir.exists() else root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else ""


def _installed_gemma4_assistant_alias_imports() -> bool:
    if not INSTALLED_APP_PYTHON.exists():
        return False
    try:
        proc = subprocess.run(
            [
                str(INSTALLED_APP_PYTHON),
                "-s",
                "-c",
                (
                    "import importlib, vmlx_engine; "
                    "vmlx_engine._install_mlx_vlm_registry_patches(); "
                    "m=importlib.import_module('mlx_vlm.models.gemma4_assistant'); "
                    "assert m.__name__ == "
                    "'mlx_vlm.speculative.drafters.gemma4_assistant'; "
                    "assert hasattr(m, 'Model'); "
                    "assert hasattr(m, 'ModelConfig')"
                ),
            ],
            cwd="/",
            env={
                "PYTHONPATH": "",
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception:  # noqa: BLE001 - audit reports false checks, not tracebacks
        return False
    return proc.returncode == 0


def _issue166_checks(root: Path) -> dict[str, bool]:
    engine_init = _read(root / "vmlx_engine/__init__.py")
    verify_script = _read(root / "panel/scripts/verify-bundled-python.sh")
    engine_tests = _read(root / "tests/test_engine_audit.py")
    return {
        "source_gemma4_assistant_alias": (
            "mlx_vlm.speculative.drafters.gemma4_assistant" in engine_init
            and "mlx_vlm.models.gemma4_assistant" in engine_init
            and "gemma4_assistant" in engine_init
            and "test_mlx_vlm_registry_patch_aliases_gemma4_assistant_model_path"
            in engine_tests
        ),
        "bundled_verify_gemma4_assistant_alias": (
            "Gemma 4 assistant mlx_vlm.models alias" in verify_script
            and "mlx_vlm.models.gemma4_assistant" in verify_script
            and '"__init__.py"' in verify_script
        ),
        "installed_app_gemma4_assistant_alias": (
            _installed_gemma4_assistant_alias_imports()
        ),
    }


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
        "installed_app_supported_runtime_flavor": (
            _app_runtime_flavor_matches(
                INSTALLED_APP,
                minimum_system_version="14.5.0",
                mlx_tag="cp312-cp312-macosx_14_0_arm64",
                mlx_metal_tag="py3-none-macosx_14_0_arm64",
            )
            or _app_runtime_flavor_matches(
                INSTALLED_APP,
                minimum_system_version="14.5.0",
                mlx_tag="cp312-cp312-macosx_26_0_arm64",
                mlx_metal_tag="py3-none-macosx_26_0_arm64",
            )
        ),
        "staged_sequoia_app_compat_runtime_flavor": _app_runtime_flavor_matches(
            root / STAGED_SEQUOIA_APP,
            minimum_system_version="14.5.0",
            mlx_tag="cp312-cp312-macosx_14_0_arm64",
            mlx_metal_tag="py3-none-macosx_14_0_arm64",
        ),
        "staged_tahoe_app_native_runtime_flavor": _app_runtime_flavor_matches(
            root / STAGED_TAHOE_APP,
            minimum_system_version="14.5.0",
            mlx_tag="cp312-cp312-macosx_26_0_arm64",
            mlx_metal_tag="py3-none-macosx_26_0_arm64",
        ),
    }


def _issue165_checks(root: Path) -> dict[str, bool]:
    tool_contract = _load_json(root / TOOL_CALL_CONTRACT)
    tool_checks = tool_contract.get("checks")
    if not isinstance(tool_checks, dict):
        tool_checks = {}
    installed_parity = _load_json(root / INSTALLED_APP_RUNTIME_PARITY)
    installed_checks = installed_parity.get("checks")
    if not isinstance(installed_checks, dict):
        installed_checks = {}
    engine_hash = installed_parity.get("bundled_engine_hash_parity")
    if not isinstance(engine_hash, dict):
        engine_hash = {}
    engine_files = engine_hash.get("files")
    if not isinstance(engine_files, dict):
        engine_files = {}
    dsml_file = engine_files.get("tool_parsers/dsml_tool_parser.py")
    if not isinstance(dsml_file, dict):
        dsml_file = {}
    engine_tests = _read(root / "tests/test_engine_audit.py")
    tool_runner = _read(root / "tests/cross_matrix/run_tool_call_contract.py")
    return {
        "tool_call_contract_source_checks_pass": (
            not tool_contract.get("failed")
            and not tool_contract.get("missing_markers")
            and tool_checks.get("schema_valid_dsml_tool_call_preserved") is True
            and tool_checks.get(
                "dsv4_default_cache_degraded_dsml_shapes_repaired_when_schema_valid"
            )
            is True
            and tool_checks.get("tool_choice_none_does_not_fallback_to_raw_dsml")
            is True
        ),
        "tool_call_contract_passes": (
            tool_contract.get("status") == "pass"
            and not tool_contract.get("failed")
            and tool_checks.get("schema_valid_dsml_tool_call_preserved") is True
            and tool_checks.get(
                "dsv4_default_cache_degraded_dsml_shapes_repaired_when_schema_valid"
            )
            is True
            and tool_checks.get("tool_choice_none_does_not_fallback_to_raw_dsml")
            is True
        ),
        "dsml_issue_165_regression_present": (
            "test_dsml_issue_165_server_tool_call_arguments_are_not_empty_or_raw"
            in engine_tests
            and "dsml_issue_165_server_tool_call_arguments_are_not_empty_or_raw"
            in tool_runner
        ),
        "installed_app_dsml_parser_hash_guarded": (
            installed_parity.get("status") == "pass"
            and installed_checks.get("installed_bundled_engine_hash_parity") is True
            and dsml_file.get("match") is True
        ),
    }


def _issue117_checks(root: Path) -> dict[str, bool]:
    issue179 = _load_json(root / ISSUE179_ROOT_CAUSE_AUDIT)
    cancel_preflight = _load_json(root / ISSUE179_RESPONSES_CANCEL_PREFLIGHT)
    release_manifest = _read(root / "tests/cross_matrix/release_regression_manifest.py")
    parser_contract = _read(root / "tests/cross_matrix/run_parser_registry_contract.py")
    minimax_real_ui = (
        root
        / "docs/internal/agent-notes/current-real-ui-live-model-minimax-m27-small-responses-stricttools-cachecontrols-20260530-proof.json"
    )
    surfaces = _surfaces(minimax_real_ui)
    return {
        "issue179_root_cause_audit_passes": issue179.get("status") == "pass",
        "issue179_root_cause_audit_open": issue179.get("status") == "open",
        "issue179_cancel_probe_present": (
            root / ISSUE179_RESPONSES_CANCEL_PROOF
        ).exists(),
        "issue179_cancel_probe_memory_preflight_present": (
            cancel_preflight.get("status")
            in {"ready_to_launch", "skipped", "skipped_active_heavy_process"}
            and cancel_preflight.get("did_not_launch") is True
            and cancel_preflight.get("launch_decision")
            in {"launch_allowed", "do_not_launch"}
        ),
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


def _mistral_small4_detection_probe() -> dict[str, bool]:
    try:
        from vmlx_engine.api import utils

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "mistral3",
                        "architectures": ["Mistral3ForConditionalGeneration"],
                        "text_config": {"model_type": "mistral4"},
                        "vision_config": {"model_type": "pixtral"},
                    }
                ),
                encoding="utf-8",
            )
            utils._IS_MLLM_CACHE.clear()
            auto = utils.is_mllm_model(str(model_dir))
            forced = utils.is_mllm_model(str(model_dir), force_mllm=True)
            return {"auto": auto is True, "forced": forced is True}
    except Exception:  # noqa: BLE001 - issue audit should report false checks
        return {"auto": False, "forced": False}


def _issue111_checks(root: Path) -> dict[str, bool]:
    installed_parity = _load_json(root / INSTALLED_APP_RUNTIME_PARITY)
    installed_checks = installed_parity.get("checks")
    if not isinstance(installed_checks, dict):
        installed_checks = {}
    engine_hash = installed_parity.get("bundled_engine_hash_parity")
    if not isinstance(engine_hash, dict):
        engine_hash = {}
    engine_files = engine_hash.get("files")
    if not isinstance(engine_files, dict):
        engine_files = {}
    api_utils = engine_files.get("api/utils.py")
    if not isinstance(api_utils, dict):
        api_utils = {}
    probe = _mistral_small4_detection_probe()
    engine_tests = _read(root / "tests/test_engine_audit.py")
    registry_tests = _read(root / "tests/test_model_config_registry.py")
    api_utils_source = _read(root / "vmlx_engine/api/utils.py")
    return {
        "mistral_small4_wrapper_stays_mllm": (
            probe.get("auto") is True
            and probe.get("forced") is True
            and "Do not apply this to Mistral Small 4" in api_utils_source
            and 'text_mt == "ministral3"' in api_utils_source
            and "tier=force_mllm result=True" in api_utils_source
        ),
        "mistral_small4_parser_metadata_preserved": (
            "test_mistral3_mistral4_vlm_wrapper_preserves_inner_reasoning"
            in registry_tests
            and "test_mistral4_detected_via_wrapper" in engine_tests
        ),
        "installed_app_mllm_hash_guarded": (
            installed_parity.get("status") == "pass"
            and installed_checks.get("installed_bundled_engine_hash_parity") is True
            and api_utils.get("match") is True
        ),
    }


def _issue116_checks(root: Path) -> dict[str, bool]:
    reasoning = _load_json(root / REASONING_TEMPLATE_CONTRACT)
    reasoning_checks = reasoning.get("checks")
    if not isinstance(reasoning_checks, dict):
        reasoning_checks = {}
    packaged = _load_json(root / PACKAGED_INTEGRITY_CONTRACT)
    packaged_checks = packaged.get("checks")
    if not isinstance(packaged_checks, dict):
        packaged_checks = {}
    reasoning_stdout = json.dumps(reasoning.get("results", {}))
    chat_settings = _read(root / "panel/src/renderer/src/components/chat/ChatSettings.tsx")
    database = _read(root / "panel/src/main/database.ts")
    gateway_tests = _read(root / "panel/tests/api-gateway-ollama.test.ts")
    reasoning_tests = _read(root / "tests/test_reasoning_modes.py")
    return {
        "reasoning_template_contract_passes": (
            reasoning.get("status") == "pass"
            and reasoning_checks.get("reasoning_on_off_request_wiring_explicit") is True
            and reasoning_checks.get("no_hidden_family_sampling_or_reasoning_forcing")
            is True
        ),
        "explicit_thinking_off_request_wired": (
            "test_reasoning_effort_none_disables_thinking_for_chat_and_responses"
            in reasoning_tests
            and "test_enable_thinking_explicit_values_still_win_for_reasoning_families"
            in reasoning_tests
            and "Completions: Explicit Off" in reasoning_stdout
            and "enable_thinking=false" in reasoning_stdout
            and "thinking=Off: no think tags generated" in reasoning_stdout
            and "openaiBody?.enable_thinking === false" in gateway_tests
        ),
        "panel_thinking_off_control_present": (
            "updateThinkingMode(false, undefined)" in chat_settings
            and "displayedEnableThinking === false" in chat_settings
            and "enable_thinking tri-state" in database
            and "0 → false (Off)" in database
        ),
        "packaged_renderer_thinking_controls_present": (
            packaged_checks.get("packaged_renderer_max_thinking_tokens_wired") is True
        ),
    }


def _installed_gemma4_speed_samples_below_floor(root: Path) -> list[float]:
    below: list[float] = []
    for relpath in GEMMA4_INSTALLED_SPEED_ARTIFACTS:
        payload = _load_json(root / relpath)
        for result in payload.get("results") or []:
            if not isinstance(result, dict):
                continue
            speed = result.get("speed")
            if not isinstance(speed, dict):
                continue
            decode_tok_s_wall = speed.get("decode_tok_s_wall")
            if isinstance(decode_tok_s_wall, int | float) and decode_tok_s_wall < 80.0:
                below.append(float(decode_tok_s_wall))
    return below


def _gemma4_current_installed_ui_speed_gate(root: Path) -> dict[str, bool]:
    payload = _load_json(root / GEMMA4_CURRENT_INSTALLED_SPEED_ARTIFACT)
    results = payload.get("results")
    if not isinstance(results, list) or len(results) < 3:
        return {
            "ui_speed_passes": False,
            "cold_wall_ttft_tracked": False,
        }

    stream_speeds: list[float] = []
    cached_wall_speeds: list[float] = []
    cold_wall_below_floor = False
    cold_stream_above_floor = False
    mixed_swa_cache_hits = 0
    for idx, result in enumerate(results):
        if not isinstance(result, dict) or result.get("status") != "ok":
            return {
                "ui_speed_passes": False,
                "cold_wall_ttft_tracked": False,
            }
        stream_speed = result.get("stream_speed")
        speed = result.get("speed")
        if not isinstance(stream_speed, dict) or not isinstance(speed, dict):
            return {
                "ui_speed_passes": False,
                "cold_wall_ttft_tracked": False,
            }
        decode_stream = stream_speed.get("decode_tok_s_stream")
        decode_wall = speed.get("decode_tok_s_wall")
        if not isinstance(decode_stream, int | float):
            return {
                "ui_speed_passes": False,
                "cold_wall_ttft_tracked": False,
            }
        stream_speeds.append(float(decode_stream))
        cached_tokens = int(stream_speed.get("cached_tokens") or 0)
        if idx == 0:
            cold_stream_above_floor = float(decode_stream) >= 80.0
            cold_wall_below_floor = (
                isinstance(decode_wall, int | float) and float(decode_wall) < 80.0
            )
        if cached_tokens > 0:
            if speed.get("cache_detail") == "paged+mixed_swa":
                mixed_swa_cache_hits += 1
            if isinstance(decode_wall, int | float):
                cached_wall_speeds.append(float(decode_wall))

    return {
        "ui_speed_passes": (
            min(stream_speeds) >= 80.0
            and len(cached_wall_speeds) >= 2
            and min(cached_wall_speeds) >= 80.0
            and mixed_swa_cache_hits >= 2
            and payload.get("status") == "pass"
        ),
        "cold_wall_ttft_tracked": cold_wall_below_floor and cold_stream_above_floor,
    }


def _qwen35_installed_speed_gate_passes(root: Path) -> bool:
    payload = _load_json(root / QWEN35_INSTALLED_SPEED_ARTIFACT)
    for result in payload.get("results") or []:
        if not isinstance(result, dict) or result.get("name") != "qwen35_4bit":
            continue
        if result.get("status") != "pass":
            return False
        bundle = result.get("bundle_sampling")
        if not isinstance(bundle, dict):
            bundle = result.get("bundle")
        if not isinstance(bundle, dict):
            return False
        speed = bundle.get("decode_tps_wall")
        if not isinstance(speed, int | float) or speed < 80.0:
            return False
        text = " ".join(
            str(bundle.get(key) or "") for key in ("content_head", "content_tail")
        )
        if "Generation failed:" in text:
            return False
        return True
    return False


def _issue115_checks(root: Path) -> dict[str, bool]:
    release_manifest = _read(root / "tests/cross_matrix/release_regression_manifest.py")
    release_tests = _read(root / "tests/test_release_regression_manifest.py")
    gemma_current = _gemma4_current_installed_ui_speed_gate(root)
    return {
        "gemma4_installed_speed_risk_tracked": (
            "Gemma4 26B mixed-SWA UI/app-engine speed is not cleared" in release_manifest
            and "stream UI TPS above 80 tok/s for cold and cache-hit rows" in release_manifest
            and "cold-wall TTFT tracking" in release_manifest
            and "does not supersede older sub-floor rows until the current issue115 speed-floor artifact exists and passes"
            in release_manifest
            and "test_release_regression_manifest_clears_gemma4_ui_speed_without_hiding_cold_wall_ttft"
            in release_tests
        ),
        "gemma4_current_installed_ui_speed_gate_passes": (
            gemma_current["ui_speed_passes"]
        ),
        "gemma4_cold_wall_includes_ttft_tracked": (
            gemma_current["cold_wall_ttft_tracked"]
        ),
        "qwen36_speed_review_tracked": (
            "qwen-jang-mx-live-speed-review" in release_manifest
            and "Qwen3.6-27B JANG_4M live decode speed" in release_manifest
            and "This row prevents broad JANG/MX matmul speed claims from hiding prompt-processing regressions"
            in release_manifest
            and "test_release_regression_manifest_tracks_qwen_jang_live_speed_review"
            in release_tests
        ),
        "qwen35_installed_app_speed_gate_passes": (
            _qwen35_installed_speed_gate_passes(root)
        ),
    }


def _issue118_checks(root: Path) -> dict[str, bool]:
    models = _read(root / "panel/src/main/ipc/models.ts")
    hf_settings = _read(root / "panel/src/shared/hfSettings.ts")
    hf_tests = _read(root / "panel/tests/hf-settings.test.ts")
    engine_tests = _read(root / "tests/test_engine_audit.py")
    installed_panel_main = _read_installed_asar_file(root, "dist/main/index.mjs")
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
        "installed_app_download_fallback_guarded": (
            "normalizeHfEndpointSetting" in installed_panel_main
            and "normalizeHfTokenSetting" in installed_panel_main
            and "HF_ENDPOINT: void 0" in installed_panel_main
            and "const hfEndpoint = getConfiguredHfEndpoint() ?? \"\"" in installed_panel_main
            and "endpoint = sys.argv[3] or None" in installed_panel_main
            and "endpoint=active_endpoint" in installed_panel_main
            and "https://huggingface.co" in installed_panel_main
            and "retrying ${baseUrl} without auth" in installed_panel_main
            and "retryWithoutAuth" in installed_panel_main
            and "isHfAuthStatus(response.status)" in installed_panel_main
            and "print(json.dumps({'type':'fallback'" in installed_panel_main
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
        "165": {
            "repo": "jjang-ai/vmlx",
            "title": (
                "DSV4-Flash JANGTQ2 + dsml parser: tool_calls.function.arguments "
                "empty / contains raw markup"
            ),
            "checks": _issue165_checks(root),
            "release_clearance": (
                "mapped_to_dsv4_dsml_tool_call_arguments_guard"
            ),
        },
        "166": {
            "repo": "jjang-ai/vmlx",
            "title": "Gemma 4 assistant support",
            "checks": _issue166_checks(root),
            "release_clearance": (
                "mapped_to_gemma4_assistant_mlx_vlm_alias_guard"
            ),
        },
        "169": {
            "repo": "jjang-ai/vmlx",
            "title": (
                "Session fails on macOS Sequoia: bundled metallib requires "
                "Metal language version 4.0"
            ),
            "checks": _issue169_checks(root),
            "release_clearance": (
                "installed_supported_runtime_flavor_and_dual_staged_dmgs_guarded_packaging_still_gated"
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
        "111": {
            "repo": "jjang-ai/mlxstudio",
            "title": "VLM mode on Mistral Small 4 broken in vMLX",
            "checks": _issue111_checks(root),
            "release_clearance": (
                "mapped_to_mistral_small4_vlm_wrapper_detection_guard"
            ),
        },
        "116": {
            "repo": "jjang-ai/mlxstudio",
            "title": "Disable thinking ?",
            "checks": _issue116_checks(root),
            "release_clearance": "mapped_to_thinking_off_ui_api_request_guard",
        },
        "115": {
            "repo": "jjang-ai/mlxstudio",
            "title": "Performance regression in v1.5.32 compared to v1.3.53",
            "checks": _issue115_checks(root),
            "release_clearance": "mapped_to_current_installed_app_gemma_qwen_speed_gate",
        },
        "118": {
            "repo": "jjang-ai/mlxstudio",
            "title": "Studio 1.5.42 cannot download models from the Hub",
            "checks": _issue118_checks(root),
            "release_clearance": (
                "installed_gui_download_endpoint_and_stale_auth_fallback_guarded"
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
    focused_open: list[str] = []
    for number, issue in issues.items():
        checks = issue["checks"]
        if number == "117" and checks.get("issue179_root_cause_audit_passes") is True:
            required_pass_checks = {
                key: value
                for key, value in checks.items()
                if key
                not in {
                    "issue179_root_cause_audit_open",
                    "issue179_cancel_probe_memory_preflight_present",
                }
            }
            issue["focused_source_slice"] = (
                "pass" if all(required_pass_checks.values()) else "fail"
            )
            issue["release_clearance"] = (
                "mapped_to_minimax_k_issue179_live_reporter_prompt_boundary"
            )
        elif number == "117" and checks.get("issue179_root_cause_audit_open") is True:
            required_open_checks = {
                key: value
                for key, value in checks.items()
                if key
                not in {
                    "issue179_root_cause_audit_passes",
                    "issue179_root_cause_audit_open",
                    "issue179_cancel_probe_present",
                }
            }
            issue["focused_source_slice"] = (
                "open" if all(required_open_checks.values()) else "fail"
            )
            issue["release_clearance"] = "open_minimax_k_issue179_reporter_parity_required"
        elif number in {"111", "165"}:
            installed_hash_checks = {
                "111": "installed_app_mllm_hash_guarded",
                "165": "installed_app_dsml_parser_hash_guarded",
            }
            installed_key = installed_hash_checks[number]
            open_keys = {installed_key}
            if number == "165":
                open_keys.add("tool_call_contract_passes")
            required_source_checks = {
                key: value for key, value in checks.items() if key not in open_keys
            }
            issue["focused_source_slice"] = (
                "open"
                if all(required_source_checks.values())
                and any(checks.get(key) is not True for key in open_keys)
                else "pass"
                if all(checks.values())
                else "fail"
            )
        elif number == "117":
            open_keys = {
                "issue179_root_cause_audit_passes",
                "issue179_root_cause_audit_open",
                "issue179_cancel_probe_present",
                "issue179_cancel_probe_memory_preflight_present",
            }
            required_source_checks = {
                key: value for key, value in checks.items() if key not in open_keys
            }
            issue["focused_source_slice"] = (
                "open"
                if all(required_source_checks.values())
                and any(checks.get(key) is not True for key in open_keys)
                else "pass"
                if all(checks.values())
                else "fail"
            )
            if issue["focused_source_slice"] == "open":
                issue["release_clearance"] = (
                    "open_minimax_k_issue179_reporter_parity_required"
                )
        elif number in {"115", "119"}:
            open_keys = {
                "115": {
                    "gemma4_current_installed_ui_speed_gate_passes",
                    "gemma4_cold_wall_includes_ttft_tracked",
                    "qwen35_installed_app_speed_gate_passes",
                },
                "119": {
                    "gemma26_memory_stress_artifact_present",
                },
            }[number]
            required_source_checks = {
                key: value for key, value in checks.items() if key not in open_keys
            }
            issue["focused_source_slice"] = (
                "open"
                if all(required_source_checks.values())
                and any(checks.get(key) is not True for key in open_keys)
                else "pass"
                if all(checks.values())
                else "fail"
            )
        else:
            issue["focused_source_slice"] = "pass" if all(checks.values()) else "fail"
        if issue["focused_source_slice"] == "fail":
            focused_failures.append(number)
        elif issue["focused_source_slice"] == "open":
            focused_open.append(number)

    return {
        "artifact": "",
        "status": "fail" if focused_failures else "open" if focused_open else "pass",
        "issues": issues,
        "focused_failures": focused_failures,
        "focused_open": focused_open,
        "release_boundary": (
            "Public issue slices #165, #166, #169, #180, and mlxstudio #111, #115-#119 "
            "have focused source/proof coverage here, and #118 includes "
            "installed app download fallback proof. This does not clear the "
            "full release; Developer ID signing/notarization, DSV4 exactness, "
            "and real UI matrix blockers remain owned by the release manifest."
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
    return 0 if audit["status"] in {"open", "pass"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
