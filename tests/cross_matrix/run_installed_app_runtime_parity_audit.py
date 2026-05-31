#!/usr/bin/env python3
"""Audit installed /Applications/vMLX.app runtime parity against release lanes.

This is a no-heavy installed-app probe. It imports the bundled runtime in an
isolated subprocess from /tmp so source-checkout modules cannot satisfy imports.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path(
    "build/current-installed-app-runtime-parity-audit-20260531-live-epipe-refresh.json"
)
INSTALLED_PYTHON = Path(
    "/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3"
)
INSTALLED_APP_ASAR = Path("/Applications/vMLX.app/Contents/Resources/app.asar")
INSTALLED_APP_USER_DATA = Path.home() / "Library/Application Support/vMLX"
INSTALLED_APP_DIAGNOSTIC_REPORTS = Path.home() / "Library/Logs/DiagnosticReports"
DISCONNECT_ERROR_RE = re.compile(
    r"\b(?:EPIPE|ECONNRESET|ERR_STREAM_DESTROYED|write EPIPE)\b",
    re.IGNORECASE,
)


def _run_installed_python(
    code: str,
    python_path: Path = INSTALLED_PYTHON,
) -> dict[str, Any]:
    if not python_path.exists():
        return {
            "returncode": 127,
            "stdout": "",
            "stderr": f"missing installed python: {python_path}",
        }
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    env["PYTHONNOUSERSITE"] = "1"
    proc = subprocess.run(
        [str(python_path), "-B", "-s", "-c", code],
        cwd="/tmp",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _serve_help(python_path: Path = INSTALLED_PYTHON) -> dict[str, Any]:
    if not python_path.exists():
        return {"returncode": 127, "stdout": "", "stderr": "missing installed python"}
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    env["PYTHONNOUSERSITE"] = "1"
    proc = subprocess.run(
        [str(python_path), "-B", "-s", "-m", "vmlx_engine.cli", "serve", "--help"],
        cwd="/tmp",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _read_installed_panel_main(
    root: Path,
    app_asar: Path = INSTALLED_APP_ASAR,
) -> dict[str, Any]:
    return _read_installed_asar_file(root, "dist/main/index.mjs", app_asar=app_asar)


def _read_installed_asar_file(
    root: Path,
    relpath: str,
    app_asar: Path = INSTALLED_APP_ASAR,
) -> dict[str, Any]:
    if not app_asar.exists():
        return {
            "returncode": 127,
            "stdout": "",
            "stderr": f"missing installed app.asar: {app_asar}",
            "text": "",
        }
    panel_dir = root / "panel"
    proc = subprocess.run(
        [
            "node",
            "-e",
            (
                "const asar=require('@electron/asar');"
                "const text=asar.extractFile(process.argv[1], process.argv[2])"
                ".toString('utf8');"
                "process.stdout.write(text);"
            ),
            str(app_asar),
            relpath,
        ],
        cwd=panel_dir if panel_dir.exists() else root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "text": proc.stdout if proc.returncode == 0 else "",
    }


def _read_installed_panel_renderer(
    root: Path,
    app_asar: Path = INSTALLED_APP_ASAR,
) -> dict[str, Any]:
    if not app_asar.exists():
        return {
            "returncode": 127,
            "stdout": "",
            "stderr": f"missing installed app.asar: {app_asar}",
            "text": "",
            "files": [],
        }
    panel_dir = root / "panel"
    proc = subprocess.run(
        [
            "node",
            "-e",
            (
                "const asar=require('@electron/asar');"
                "const archive=process.argv[1];"
                "const files=asar.listPackage(archive)"
                ".filter(f=>/\\/out\\/renderer\\/assets\\/index-.*\\.js$/.test(f));"
                "for (const file of files) {"
                "process.stdout.write(asar.extractFile(archive, file.replace(/^\\//,''))"
                ".toString('utf8'));"
                "process.stdout.write('\\n');"
                "}"
            ),
            str(app_asar),
        ],
        cwd=panel_dir if panel_dir.exists() else root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    files_proc = subprocess.run(
        [
            "node",
            "-e",
            (
                "const asar=require('@electron/asar');"
                "process.stdout.write(JSON.stringify(asar.listPackage(process.argv[1])"
                ".filter(f=>/\\/out\\/renderer\\/assets\\/index-.*\\.js$/.test(f))))"
            ),
            str(app_asar),
        ],
        cwd=panel_dir if panel_dir.exists() else root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    files: list[str] = []
    if files_proc.returncode == 0 and files_proc.stdout.strip():
        try:
            files = list(json.loads(files_proc.stdout))
        except Exception:
            files = []
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "text": proc.stdout if proc.returncode == 0 else "",
        "files": files,
    }


def _scan_vmlx_user_data_disconnect_errors(
    user_data: Path = INSTALLED_APP_USER_DATA,
) -> dict[str, Any]:
    """Scan only vMLX user-data text logs for raw disconnect errors."""
    candidate_names = {
        "LOG",
        "LOG.old",
        "000003.log",
        "Preferences",
    }
    hits: list[dict[str, Any]] = []
    scanned: list[str] = []
    if not user_data.exists():
        return {
            "user_data": str(user_data),
            "exists": False,
            "scanned": scanned,
            "hits": hits,
        }
    for path in sorted(user_data.rglob("*")):
        if not path.is_file() or path.name not in candidate_names:
            continue
        try:
            rel = str(path.relative_to(user_data))
        except ValueError:
            rel = str(path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        scanned.append(rel)
        for lineno, line in enumerate(text.splitlines(), start=1):
            if DISCONNECT_ERROR_RE.search(line):
                hits.append(
                    {
                        "file": rel,
                        "line": lineno,
                        "text": line[:500],
                    }
                )
    return {
        "user_data": str(user_data),
        "exists": True,
        "scanned": scanned,
        "hits": hits,
    }


def _scan_vmlx_diagnostic_disconnect_errors(
    diagnostic_reports: Path = INSTALLED_APP_DIAGNOSTIC_REPORTS,
) -> dict[str, Any]:
    """Scan vMLX crash/diagnostic reports for raw disconnect errors."""
    hits: list[dict[str, Any]] = []
    scanned: list[str] = []
    if not diagnostic_reports.exists():
        return {
            "diagnostic_reports": str(diagnostic_reports),
            "exists": False,
            "scanned": scanned,
            "hits": hits,
        }
    for path in sorted(diagnostic_reports.rglob("*")):
        if not path.is_file():
            continue
        if not (
            path.name.startswith("vMLX-")
            or path.name.startswith("vmlxctl-")
            or path.name.startswith("vmlx-")
        ):
            continue
        try:
            rel = str(path.relative_to(diagnostic_reports))
        except ValueError:
            rel = str(path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        scanned.append(rel)
        for lineno, line in enumerate(text.splitlines(), start=1):
            if DISCONNECT_ERROR_RE.search(line):
                hits.append(
                    {
                        "file": rel,
                        "line": lineno,
                        "text": line[:500],
                    }
                )
    return {
        "diagnostic_reports": str(diagnostic_reports),
        "exists": True,
        "scanned": scanned,
        "hits": hits,
    }


def build_audit(
    root: Path,
    *,
    app_path: Path = Path("/Applications/vMLX.app"),
    python_path: Path = INSTALLED_PYTHON,
    app_asar: Path = INSTALLED_APP_ASAR,
    user_data: Path = INSTALLED_APP_USER_DATA,
    diagnostic_reports: Path = INSTALLED_APP_DIAGNOSTIC_REPORTS,
) -> dict[str, Any]:
    help_result = _serve_help(python_path)
    help_text = help_result["stdout"] + "\n" + help_result["stderr"]
    panel_result = _read_installed_panel_main(root, app_asar=app_asar)
    panel_main = panel_result["text"]
    renderer_result = _read_installed_panel_renderer(root, app_asar=app_asar)
    panel_renderer = renderer_result["text"]
    user_data_disconnect_errors = _scan_vmlx_user_data_disconnect_errors(user_data)
    diagnostic_disconnect_errors = _scan_vmlx_diagnostic_disconnect_errors(
        diagnostic_reports
    )
    import_result = _run_installed_python(
        "import inspect, json\n"
        "from types import SimpleNamespace\n"
        "from vmlx_engine import server\n"
        "from vmlx_engine import cli\n"
        "from vmlx_engine.image_gen import ImageGenEngine\n"
        "plain_cache = server._native_cache_status(SimpleNamespace(\n"
        "    _model_type_for_runtime='minimax',\n"
        "    _tq_active=True,\n"
        "    _kv_cache_bits=3,\n"
        "    _kv_cache_group_size=64,\n"
        "    block_aware_cache=object(),\n"
        "    paged_cache_manager=SimpleNamespace(_disk_store=object()),\n"
        "))\n"
        "text_lora_rejected = False\n"
        "text_lora_error = ''\n"
        "try:\n"
        "    cli._validate_lora_args_for_model_type(\n"
        "        SimpleNamespace(\n"
        "            model='/tmp/text-model',\n"
        "            lora_paths=['/tmp/style.safetensors'],\n"
        "            lora_scales=None,\n"
        "        ),\n"
        "        is_image=False,\n"
        "    )\n"
        "except SystemExit as exc:\n"
        "    text_lora_error = str(exc)\n"
        "    text_lora_rejected = '--lora-paths' in text_lora_error\n"
        "payload = {\n"
        "  'server_file': server.__file__,\n"
        "  'image_load_signature': str(inspect.signature(ImageGenEngine.load)),\n"
        "  'has_image_lora_globals': hasattr(server, '_image_lora_paths') and hasattr(server, '_image_lora_scales'),\n"
        "  'cancel_routes': sorted(f'{','.join(sorted(route.methods or []))} {route.path}' for route in server.app.routes if '/cancel' in getattr(route, 'path', '')),\n"
        "  'plain_attention_kv_native_cache': plain_cache,\n"
        "  'text_lora_rejected': text_lora_rejected,\n"
        "  'text_lora_error': text_lora_error,\n"
        "}\n"
        "print(json.dumps(payload, sort_keys=True))\n"
        ,
        python_path=python_path,
    )
    import_payload: dict[str, Any] = {}
    if import_result["returncode"] == 0 and import_result["stdout"].strip():
        try:
            import_payload = json.loads(import_result["stdout"].splitlines()[-1])
        except Exception:
            import_payload = {}
    installed_ollama_json_writes = len(
        re.findall(r"this\.writeJsonLine\(\s*res,", panel_main)
    )
    installed_guarded_ollama_json_writes = len(
        re.findall(r"if\s*\(\s*!this\.writeJsonLine\(\s*res,", panel_main)
    )
    installed_ollama_proxy_handlers = re.findall(
        r"const proxyReq = [^(]+\(proxyOpts, \(proxyRes\) => \{[\s\S]*?proxyReq\.on\(\"error\"",
        panel_main,
    )
    installed_ollama_guarded_proxy_handlers = [
        handler
        for handler in installed_ollama_proxy_handlers
        if 'proxyRes.on("error"' in handler
    ]

    checks = {
        "installed_python_exists": python_path.exists(),
        "serve_help_runs": help_result["returncode"] == 0,
        "responses_cancel_route": any(
            "/v1/responses/{response_id}/cancel" in route
            for route in import_payload.get("cancel_routes", [])
        ),
        "image_lora_cli_flags": "--lora-paths" in help_text
        and "--lora-scales" in help_text,
        "image_lora_load_signature": "lora_paths" in str(
            import_payload.get("image_load_signature", "")
        )
        and "lora_scales" in str(import_payload.get("image_load_signature", "")),
        "image_lora_server_globals": import_payload.get("has_image_lora_globals") is True,
        "text_lora_flags_rejected": import_payload.get("text_lora_rejected") is True,
        "xml_function_tool_parser_cli": "xml_function" in help_text,
        "plain_attention_kv_native_cache_status": (
            import_payload.get("plain_attention_kv_native_cache", {}).get("schema")
            == "plain_kv_v1"
            and import_payload.get("plain_attention_kv_native_cache", {}).get("cache_type")
            == "paged_kv"
            and import_payload.get("plain_attention_kv_native_cache", {}).get(
                "generic_turboquant_kv", {}
            ).get("enabled")
            is True
            and import_payload.get("plain_attention_kv_native_cache", {}).get(
                "block_disk_l2"
            )
            is True
        ),
        "installed_panel_gateway_epipe_guard": (
            panel_result["returncode"] == 0
            and "isClientDisconnectError" in panel_main
            and "EPIPE" in panel_main
            and "ECONNRESET" in panel_main
            and "ERR_STREAM_DESTROYED" in panel_main
            and "write EPIPE" in panel_main
        ),
        "installed_panel_gateway_epipe_aggregate_guard": (
            panel_result["returncode"] == 0
            and "isClientDisconnectError" in panel_main
            and "ERR_STREAM_WRITE_AFTER_END" in panel_main
            and "nestedErrors" in panel_main
            and "Array.isArray" in panel_main
            and "isExpectedClientDisconnectError" in panel_main
            and "write EPIPE" in panel_main
        ),
        "installed_panel_chat_ipc_epipe_aggregate_guard": (
            panel_result["returncode"] == 0
            and "isExpectedChatBackendDisconnectError" in panel_main
            and "nestedErrors" in panel_main
            and "Array.isArray" in panel_main
            and "nestedErrors.some((nested) => isExpectedChatBackendDisconnectError(nested))"
            in panel_main
            and "write EPIPE" in panel_main
        ),
        "installed_panel_image_ipc_epipe_aggregate_guard": (
            panel_result["returncode"] == 0
            and "isExpectedImageServerDisconnectError" in panel_main
            and "nestedErrors" in panel_main
            and "Array.isArray" in panel_main
            and "nestedErrors.some((nested) => isExpectedImageServerDisconnectError(nested))"
            in panel_main
            and "write EPIPE" in panel_main
        ),
        "installed_panel_child_process_stdio_epipe_guard": (
            panel_result["returncode"] == 0
            and "isExpectedChildProcessStreamDisconnectError" in panel_main
            and "attachChildProcessStreamErrorGuard" in panel_main
            and "ERR_STREAM_WRITE_AFTER_END" in panel_main
            and "write EPIPE" in panel_main
            and "[DOWNLOADS] stdout stream error" in panel_main
            and "Stdout stream error" in panel_main
        ),
        "installed_panel_gateway_guarded_proxy_forwarding": (
            panel_result["returncode"] == 0
            and "proxyRes.pipe" not in panel_main
            and 'proxyRes.on("data"' in panel_main
            and "writeResponse(clientRes" in panel_main
        ),
        "installed_panel_gateway_write_once_behavior_marker": (
            panel_result["returncode"] == 0
            and "writeResponse(res, chunk)" in panel_main
            and "if (this.isClientDisconnectError(err)) return false" in panel_main
        ),
        "installed_panel_gateway_response_socket_destroyed_guard": (
            panel_result["returncode"] == 0
            and "responseWritable(res)" in panel_main
            and "writableDestroyed" in panel_main
            and "socket" in panel_main
            and "destroyed" in panel_main
        ),
        "installed_panel_gateway_request_socket_destroyed_guard": (
            panel_result["returncode"] == 0
            and "requestWritable(req)" in panel_main
            and "writableDestroyed" in panel_main
            and "socket" in panel_main
            and "destroyed" in panel_main
        ),
        "installed_panel_gateway_single_model_cache_endpoint_routing": (
            panel_result["returncode"] == 0
            and "gateway_single_model_mode" in panel_main
            and "/v1/cache/stats" in panel_main
            and "params.get(\"model\")" in panel_main
            and "enforceSingleModelMode" in panel_main
            and "sessionManager.stopSession" in panel_main
            and "sessionManager.wakeSession" in panel_main
            and "single_model_unload_failed" in panel_main
        ),
        "installed_panel_gateway_single_model_streaming_routes": (
            panel_result["returncode"] == 0
            and "gateway_single_model_mode" in panel_main
            and "/v1/chat/completions" in panel_main
            and "/v1/responses" in panel_main
            and "/api/chat" in panel_main
            and "/api/generate" in panel_main
            and "handleOllamaRoute" in panel_main
            and "enforceSingleModelMode" in panel_main
            and "sessionManager.stopSession" in panel_main
            and "sessionManager.wakeSession" in panel_main
            and "single_model_unload_failed" in panel_main
            and "writeResponse(clientRes" in panel_main
        ),
        "installed_panel_max_output_context_settings_wired": (
            renderer_result["returncode"] == 0
            and "Max Output Tokens" in panel_renderer
            and "Max Context Tokens" in panel_renderer
            and "--max-tokens" in panel_renderer
            and "--max-prompt-tokens" in panel_renderer
            and "does not change prompt/context length" in panel_renderer
            and "does not cap generated output" in panel_renderer
            and "Model-owned" in panel_renderer
        ),
        "installed_panel_model_owned_generation_defaults_wired": (
            renderer_result["returncode"] == 0
            and "generation_config.json/jang_config" in panel_renderer
            and "does not synthesize missing sampling values" in panel_renderer
            and "model-declared values" in panel_renderer
            and "max_tokens/max_output_tokens" in panel_renderer
            and "max_thinking_tokens" in panel_renderer
        ),
        "installed_panel_request_builders_keep_output_context_separate": (
            panel_result["returncode"] == 0
            and panel_main.count("max_output_tokens") > 0
            and panel_main.count("max_prompt_tokens") > 0
            and "num_predict" in panel_main
            and "num_ctx" in panel_main
            and "thinking_budget" in panel_main
        ),
        "installed_panel_ollama_streaming_json_writes_guarded": (
            panel_result["returncode"] == 0
            and installed_ollama_json_writes > 0
            and installed_guarded_ollama_json_writes == installed_ollama_json_writes
        ),
        "installed_panel_ollama_proxy_response_errors_guarded": (
            panel_result["returncode"] == 0
            and len(installed_ollama_proxy_handlers) >= 3
            and len(installed_ollama_guarded_proxy_handlers)
            == len(installed_ollama_proxy_handlers)
        ),
        "installed_vmlx_user_data_no_raw_disconnect_errors": (
            user_data_disconnect_errors["exists"] is True
            and not user_data_disconnect_errors["hits"]
        ),
        "installed_vmlx_diagnostic_reports_no_raw_disconnect_errors": (
            diagnostic_disconnect_errors["exists"] is True
            and not diagnostic_disconnect_errors["hits"]
        ),
    }
    missing_or_stale = [name for name, ok in checks.items() if not ok]
    return {
        "artifact": "",
        "status": "pass" if not missing_or_stale else "open",
        "installed_app": str(app_path),
        "installed_python": str(python_path),
        "checks": checks,
        "missing_or_stale": missing_or_stale,
        "serve_help_returncode": help_result["returncode"],
        "serve_help_relevant": [
            line
            for line in help_text.splitlines()
            if "lora" in line.lower()
            or "tool-call-parser" in line
            or "xml_function" in line
        ][:40],
        "import_returncode": import_result["returncode"],
        "import_payload": import_payload,
        "installed_app_asar": str(app_asar),
        "panel_main_returncode": panel_result["returncode"],
        "panel_main_stderr": panel_result["stderr"],
        "panel_main_markers": {
            "has_disconnect_guard": "isClientDisconnectError" in panel_main,
            "has_epipe": "EPIPE" in panel_main,
            "has_write_after_end_guard": "ERR_STREAM_WRITE_AFTER_END" in panel_main,
            "has_aggregate_disconnect_guard": "nestedErrors" in panel_main
            and "Array.isArray" in panel_main,
            "has_chat_ipc_aggregate_disconnect_guard": (
                "isExpectedChatBackendDisconnectError" in panel_main
                and "nestedErrors.some((nested) => isExpectedChatBackendDisconnectError(nested))"
                in panel_main
            ),
            "has_image_ipc_aggregate_disconnect_guard": (
                "isExpectedImageServerDisconnectError" in panel_main
                and "nestedErrors.some((nested) => isExpectedImageServerDisconnectError(nested))"
                in panel_main
            ),
            "has_child_stdio_disconnect_guard": (
                "isExpectedChildProcessStreamDisconnectError" in panel_main
                and "attachChildProcessStreamErrorGuard" in panel_main
            ),
            "has_download_stdio_guard": "[DOWNLOADS] stdout stream error"
            in panel_main,
            "has_tool_executor_stdio_guard": "Stdout stream error" in panel_main,
            "has_raw_proxy_pipe": "proxyRes.pipe" in panel_main,
            "has_guarded_proxy_data": 'proxyRes.on("data"' in panel_main
            and "writeResponse(clientRes" in panel_main,
            "has_write_response": "writeResponse(res, chunk)" in panel_main,
            "has_writable_destroyed_guard": "writableDestroyed" in panel_main,
            "has_socket_destroyed_guard": "socket" in panel_main
            and "destroyed" in panel_main,
            "has_single_model_setting": "gateway_single_model_mode" in panel_main,
            "has_cache_stats_route": "/v1/cache/stats" in panel_main,
            "has_query_model_routing": "params.get(\"model\")" in panel_main,
            "has_single_model_enforcer": "enforceSingleModelMode" in panel_main,
            "has_stop_and_wake": "sessionManager.stopSession" in panel_main
            and "sessionManager.wakeSession" in panel_main,
            "has_chat_completions_route": "/v1/chat/completions" in panel_main,
            "has_responses_route": "/v1/responses" in panel_main,
            "has_ollama_chat_route": "/api/chat" in panel_main,
            "has_ollama_generate_route": "/api/generate" in panel_main,
            "has_ollama_route_handler": "handleOllamaRoute" in panel_main,
            "has_request_max_output_tokens": "max_output_tokens" in panel_main,
            "has_request_max_prompt_tokens": "max_prompt_tokens" in panel_main,
            "has_ollama_num_predict": "num_predict" in panel_main,
            "has_ollama_num_ctx": "num_ctx" in panel_main,
            "has_thinking_budget": "thinking_budget" in panel_main,
            "ollama_json_write_count": installed_ollama_json_writes,
            "guarded_ollama_json_write_count": installed_guarded_ollama_json_writes,
        },
        "panel_renderer_returncode": renderer_result["returncode"],
        "panel_renderer_stderr": renderer_result["stderr"],
        "panel_renderer_files": renderer_result["files"],
        "panel_renderer_markers": {
            "has_max_output_tokens_label": "Max Output Tokens" in panel_renderer,
            "has_max_context_tokens_label": "Max Context Tokens" in panel_renderer,
            "has_max_tokens_cli": "--max-tokens" in panel_renderer,
            "has_max_prompt_tokens_cli": "--max-prompt-tokens" in panel_renderer,
            "has_output_context_separation_text": (
                "does not change prompt/context length" in panel_renderer
                and "does not cap generated output" in panel_renderer
            ),
            "has_model_owned_label": "Model-owned" in panel_renderer,
            "has_generation_config_note": "generation_config.json/jang_config"
            in panel_renderer,
            "has_no_synthesized_sampling_note": (
                "does not synthesize missing sampling values" in panel_renderer
            ),
            "has_model_declared_values_note": "model-declared values" in panel_renderer,
            "has_chat_output_override_note": "max_tokens/max_output_tokens"
            in panel_renderer,
            "has_chat_thinking_budget_note": "max_thinking_tokens" in panel_renderer,
        },
        "vmlx_user_data_disconnect_errors": user_data_disconnect_errors,
        "vmlx_diagnostic_disconnect_errors": diagnostic_disconnect_errors,
        "boundary": (
            "Installed app parity is required before claiming packaged/app "
            "coverage for #178 LoRA flags, MiMo xml_function launch, or "
            "Electron gateway EPIPE/AggregateError disconnect handling, or ALGateway "
            "single-model cache endpoint/routing behavior, or model-owned "
            "generation/max-token settings behavior."
        ),
    }


def write_audit(
    root: Path,
    out: Path,
    *,
    app_path: Path = Path("/Applications/vMLX.app"),
    python_path: Path = INSTALLED_PYTHON,
    app_asar: Path = INSTALLED_APP_ASAR,
    user_data: Path = INSTALLED_APP_USER_DATA,
    diagnostic_reports: Path = INSTALLED_APP_DIAGNOSTIC_REPORTS,
) -> dict[str, Any]:
    audit = build_audit(
        root,
        app_path=app_path,
        python_path=python_path,
        app_asar=app_asar,
        user_data=user_data,
        diagnostic_reports=diagnostic_reports,
    )
    audit["artifact"] = str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--app",
        type=Path,
        default=Path("/Applications/vMLX.app"),
        help="App bundle to audit; defaults to the installed /Applications app.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=None,
        help="Bundled Python executable to audit; defaults under --app.",
    )
    parser.add_argument(
        "--app-asar",
        type=Path,
        default=None,
        help="app.asar to audit; defaults under --app.",
    )
    parser.add_argument(
        "--user-data",
        type=Path,
        default=INSTALLED_APP_USER_DATA,
        help="vMLX user-data directory to scan for disconnect logs.",
    )
    parser.add_argument(
        "--diagnostic-reports",
        type=Path,
        default=INSTALLED_APP_DIAGNOSTIC_REPORTS,
        help="DiagnosticReports directory to scan for vMLX disconnect crashes.",
    )
    args = parser.parse_args()

    app_path = args.app.expanduser().resolve()
    python_path = (
        args.python.expanduser().resolve()
        if args.python
        else app_path / "Contents/Resources/bundled-python/python/bin/python3"
    )
    app_asar = (
        args.app_asar.expanduser().resolve()
        if args.app_asar
        else app_path / "Contents/Resources/app.asar"
    )
    audit = write_audit(
        args.root,
        args.out,
        app_path=app_path,
        python_path=python_path,
        app_asar=app_asar,
        user_data=args.user_data.expanduser().resolve(),
        diagnostic_reports=args.diagnostic_reports.expanduser().resolve(),
    )
    print(json.dumps({"status": audit["status"], "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
