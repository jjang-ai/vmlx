#!/usr/bin/env python3
"""Summarize the current MiniMax-K issue #179 root-cause boundary.

This is intentionally a no-heavy gate. It does not claim to reproduce the
reporter failure; it records what the submitted log proves, what local live
diagnostics prove, and what evidence is still required before closing the issue.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path(
    "build/current-issue179-minimax-k-root-cause-audit-after-parser-settings-parity-20260608.json"
)
REPORTER_LOG = Path("build/issue-179/vmlx-logs-490f58c0-2026-05-27.log")
REPORTER_SCREENSHOT = Path("build/issue-179/minimax-garbage-screenshot.png")
REPORTER_PARITY_ARTIFACT = Path(
    "build/issue-179/reporter-parity-metadata-20260527.json"
)
REPORTER_PARITY_REQUIRED_FIELDS = (
    "capture_provenance",
    "installed_server_sha256",
    "server_has_responses_cancel_route",
    "server_cancel_calls_engine_abort",
    "model_manifest_sha256",
    "model_file_hashes",
    "chat_id",
    "session_settings",
    "response_id",
    "response_active_at_cancel",
    "raw_sse_cancel_lifecycle",
)
REPORTER_SERVER_HASH_FROM_PUBLIC_ISSUE_COMMENT = (
    "a2c4aed88d69b0537de9c7bb7ecb68e4d97f40e92e9fec781e9f79b9b157e8be"
)
REPORTER_SERVER_HASH_FALLBACK_PROVENANCE = (
    "github_issue_179_comment_jjang-ai_2026-06-02"
)

LOCAL_REAL_UI_PROOFS = (
    Path(
        "docs/internal/agent-notes/"
        "diagnostic-real-ui-live-model-minimax-m27-jangtq-k-issue179-20260527-proof.json"
    ),
    Path(
        "docs/internal/agent-notes/"
        "diagnostic-installed-ui-live-model-minimax-m27-jangtq-k-issue179-20260527-proof.json"
    ),
    Path(
        "docs/internal/agent-notes/"
        "diagnostic-installed-ui-live-model-minimax-m27-jangtq-k-issue179-hi-auto-20260527-proof.json"
    ),
    Path(
        "docs/internal/agent-notes/"
        "diagnostic-installed-ui-live-model-minimax-m27-jangtq-k-issue179-hi-thinking-20260527-proof.json"
    ),
    Path(
        "docs/internal/agent-notes/"
        "diagnostic-installed-ui-live-model-minimax-m27-jangtq-k-issue179-512-20260527-proof.json"
    ),
)
LOCAL_RESPONSES_CANCEL_PROOF = Path(
    "build/current-issue179-minimax-k-responses-cancel-probe-current-source-parser-settings-parity-20260608.json"
)
LOCAL_LIVE_PROBE_MEMORY_PREFLIGHT = Path(
    "build/current-issue179-minimax-k-responses-cancel-probe-memory-preflight-20260602-local-ready-check.json"
)
LOCAL_REPORTER_PROMPT_REPRODUCTION_PROOF = Path(
    "build/current-issue179-minimax-k-responses-cancel-probe-current-source-parser-settings-parity-20260608.json"
)
LOCAL_REPORTER_PROMPT_REPRODUCTION_FALLBACKS = (
    Path("build/current-issue179-minimax-k-responses-cancel-probe-after-mimo-dsv4-ledger-20260607.json"),
    Path("build/current-issue179-minimax-k-responses-cancel-probe-20260606-live-refresh.json"),
)
LOCAL_MODEL_MANIFEST = Path(
    "build/current-issue179-minimax-k-local-model-manifest-20260527.json"
)
PUBLIC_RELEASE_DMG_CONTRACT = Path(
    "build/issue-179/public-v1.5.49-tahoe-dmg-contract.json"
)
PUBLIC_RELEASE_DMG_CONTRACT_GLOB = "public-v*-*-dmg-contract.json"
PUBLIC_RELEASE_DMG_CONTRACT_FALLBACK_AUDITS = (
    Path("build/current-issue179-minimax-k-root-cause-audit-after-public-v1556-scan-20260606.json"),
)
REQUIRED_PUBLIC_RELEASE_CONTRACTS = {
    ("v1.5.56", "vMLX-1.5.56-sequoia-arm64.dmg"),
    ("v1.5.56", "vMLX-1.5.56-tahoe-arm64.dmg"),
}
PUBLIC_RELEASE_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")

SOURCE_CONTRACT_FILES = (
    Path("vmlx_engine/server.py"),
    Path("panel/src/main/ipc/chat.ts"),
    Path("tests/test_cancellation.py"),
)
LOCAL_INSTALLED_BUNDLE_SERVER = Path(
    "/Applications/vMLX.app/Contents/Resources/bundled-python/python/lib/"
    "python3.12/site-packages/vmlx_engine/server.py"
)

RAW_REAL_UI_PARSER_LEAK_RE = re.compile(
    r"<think>|</think>|<tool_call>|</tool_call>|<function>|<invoke>|"
    r"<minimax:tool_call>|<zyphra_tool_call>|<\|point_start\|>|"
    r"<\|point_end\|>|<\|box_start\|>|<\|box_end\|>"
)
REASONING_NUMERIC_GARBAGE_RE = re.compile(
    r"(?:^|[\s([{,;:])(?:\d{1,4}[\s,;:|\-/.]+){8,}\d{1,4}(?=$|[\s)\]},;:.])",
    re.MULTILINE,
)

_FLAG_KEY_MAP = {
    "--host": "host",
    "--port": "port",
    "--timeout": "timeout",
    "--max-num-seqs": "max_num_seqs",
    "--prefill-batch-size": "prefill_batch_size",
    "--prefill-step-size": "prefill_step_size",
    "--completion-batch-size": "completion_batch_size",
    "--tool-call-parser": "tool_call_parser",
    "--reasoning-parser": "reasoning_parser",
    "--paged-cache-block-size": "paged_cache_block_size",
    "--max-cache-blocks": "max_cache_blocks",
    "--block-disk-cache-max-gb": "block_disk_cache_max_gb",
    "--stream-interval": "stream_interval",
}

_BOOL_FLAGS = {
    "--continuous-batching": "continuous_batching",
    "--enable-auto-tool-choice": "enable_auto_tool_choice",
    "--use-paged-cache": "use_paged_cache",
    "--enable-block-disk-cache": "enable_block_disk_cache",
}

_INT_FIELDS = {
    "port",
    "timeout",
    "max_num_seqs",
    "prefill_batch_size",
    "prefill_step_size",
    "completion_batch_size",
    "paged_cache_block_size",
    "max_cache_blocks",
    "block_disk_cache_max_gb",
    "stream_interval",
    "default_max_tokens_fallback",
}


def _coerce_log_value(value: str) -> Any:
    if value == "True":
        return True
    if value == "False":
        return False
    if value == "None":
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def _parse_launch_config(line: str) -> dict[str, Any]:
    _, command = line.split("$ ", 1)
    try:
        argv = shlex.split(command)
    except ValueError:
        return {}
    if len(argv) < 6 or argv[1:4] != ["-B", "-s", "-m"]:
        return {}
    out: dict[str, Any] = {
        "executable": argv[0],
        "module": argv[4],
        "subcommand": argv[5],
    }
    if len(argv) > 6:
        out["model_path"] = argv[6]
    idx = 7
    while idx < len(argv):
        flag = argv[idx]
        if flag in _BOOL_FLAGS:
            out[_BOOL_FLAGS[flag]] = True
            idx += 1
            continue
        key = _FLAG_KEY_MAP.get(flag)
        if key and idx + 1 < len(argv):
            out[key] = _coerce_log_value(argv[idx + 1])
            idx += 2
            continue
        idx += 1
    return out


def _parse_model_config(line: str) -> dict[str, Any]:
    match = re.search(r"Model config: (.*)$", line)
    if not match:
        return {}
    out: dict[str, Any] = {}
    for item in match.group(1).split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        out[key] = _coerce_log_value(value)
    return out


def _runtime_config_proven(runtime: dict[str, Any]) -> bool:
    return (
        runtime.get("native_tool_format") == "minimax"
        and runtime.get("reasoning_parser") == "minimax_m2"
        and runtime.get("paged_cache") == {"block_size": 64, "max_blocks": 1000}
        and runtime.get("block_disk_cache_max_gb") == 10.0
        and runtime.get("kv_cache_quantization") == {"bits": 4, "group_size": 64}
        and runtime.get("runtime_cache_all_turboquant") is True
        and runtime.get("single_active_scheduler") is True
        and runtime.get("default_max_tokens_fallback") == 4096
    )


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_reporter_parity_artifact(root: Path) -> dict[str, Any]:
    path = root / REPORTER_PARITY_ARTIFACT
    if not path.exists():
        return {
            "path": str(REPORTER_PARITY_ARTIFACT),
            "exists": False,
            "status": "open",
            "required_fields": list(REPORTER_PARITY_REQUIRED_FIELDS),
            "capture_provenance": REPORTER_SERVER_HASH_FALLBACK_PROVENANCE,
            "installed_server_sha256": REPORTER_SERVER_HASH_FROM_PUBLIC_ISSUE_COMMENT,
            "server_has_responses_cancel_route": True,
            "server_cancel_calls_engine_abort": True,
            "comparison_status": "missing_reporter_parity_artifact",
            "note": (
                "Full reporter parity artifact is missing, but the reporter "
                "installed server.py hash was published in GitHub issue #179 "
                "comments by jjang-ai on 2026-06-02."
            ),
        }

    data = read_json(path)
    missing = [
        field for field in REPORTER_PARITY_REQUIRED_FIELDS if field not in data
    ]
    collector_missing = data.get("missing_fields")
    if not isinstance(collector_missing, list):
        collector_missing = []
    collector_status = data.get("status")
    comparison_status = "ready_for_direct_comparison"
    if missing:
        comparison_status = "missing_required_fields"
    elif collector_missing:
        comparison_status = "collector_missing_fields"
    elif collector_status != "pass":
        comparison_status = f"collector_status_{collector_status or 'missing'}"
    return {
        "path": str(REPORTER_PARITY_ARTIFACT),
        "exists": True,
        "status": (
            "pass"
            if not missing and not collector_missing and collector_status == "pass"
            else "open"
        ),
        "required_fields": list(REPORTER_PARITY_REQUIRED_FIELDS),
        "missing_fields": missing,
        "collector_missing_fields": collector_missing,
        "comparison_status": comparison_status,
        "sha256": sha256_file(path),
        **{key: data.get(key) for key in REPORTER_PARITY_REQUIRED_FIELDS},
    }


def _hash_rows(rows: Any) -> set[tuple[str, str]]:
    if not isinstance(rows, list):
        return set()
    out: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        path = row.get("path")
        sha = row.get("sha256")
        if isinstance(path, str) and isinstance(sha, str):
            out.add((path, sha))
    return out


def build_reporter_parity_comparison(
    *,
    reporter_parity_artifact: dict[str, Any],
    reporter: dict[str, Any],
    installed_bundle: dict[str, Any],
    local_model_manifest: dict[str, Any],
) -> dict[str, Any]:
    if reporter_parity_artifact.get("status") != "pass":
        return {
            "status": "open",
            "failures": [
                str(
                    reporter_parity_artifact.get(
                        "comparison_status", "reporter_parity_artifact_not_ready"
                    )
                )
            ],
        }

    checks = {
        "capture_provenance_is_reporter_machine": (
            reporter_parity_artifact.get("capture_provenance") == "reporter_machine"
        ),
        "server_hash_matches_local_installed": (
            reporter_parity_artifact.get("installed_server_sha256")
            == installed_bundle.get("sha256")
        ),
        "server_route_markers_match": (
            reporter_parity_artifact.get("server_has_responses_cancel_route") is True
            and reporter_parity_artifact.get("server_cancel_calls_engine_abort") is True
        ),
        "model_manifest_sha256_matches_local": (
            reporter_parity_artifact.get("model_manifest_sha256")
            == local_model_manifest.get("sha256")
        ),
        "model_file_hashes_match_local": (
            bool(_hash_rows(reporter_parity_artifact.get("model_file_hashes")))
            and _hash_rows(reporter_parity_artifact.get("model_file_hashes"))
            == _hash_rows(local_model_manifest.get("model_file_hashes"))
        ),
        "chat_id_matches_reporter_log": (
            reporter_parity_artifact.get("chat_id")
            == (reporter.get("request_error") or {}).get("chatId")
        ),
        "response_id_matches_reporter_log": (
            reporter_parity_artifact.get("response_id")
            == reporter.get("request_error_response_id")
        ),
        "response_active_at_cancel_recorded": isinstance(
            reporter_parity_artifact.get("response_active_at_cancel"), bool
        ),
        "raw_sse_cancel_lifecycle_present": isinstance(
            reporter_parity_artifact.get("raw_sse_cancel_lifecycle"), dict
        ),
    }
    failures = [key for key, ok in checks.items() if not ok]
    return {
        "status": "pass" if not failures else "open",
        **checks,
        "failures": failures,
    }


def reporter_hash_mismatch_is_stale_release_nonblocker(
    *,
    reporter_parity_comparison: dict[str, Any],
    reporter_server_hash_parity: dict[str, Any] | None,
    proven: dict[str, Any] | None,
) -> bool:
    reporter_server_hash_parity = reporter_server_hash_parity or {}
    proven = proven or {}
    provenance = reporter_server_hash_parity.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    public_release_matches = provenance.get("public_release_matches")
    if not isinstance(public_release_matches, list):
        public_release_matches = []
    comparison_failures = reporter_parity_comparison.get("failures")
    if not isinstance(comparison_failures, list):
        comparison_failures = []
    required_clean_current_proofs = (
        "current_source_responses_cancel_contract_proven",
        "current_source_responses_cancel_inactive_404_contract_proven",
        "latest_public_dmg_has_responses_cancel_route",
        "local_installed_bundle_has_responses_cancel_route",
        "local_installed_responses_cancel_live_probe",
        "local_reporter_prompt_reproduction_clean",
    )
    return (
        comparison_failures == ["server_hash_matches_local_installed"]
        and reporter_parity_comparison.get("server_hash_matches_local_installed") is False
        and reporter_parity_comparison.get("server_route_markers_match") is True
        and reporter_parity_comparison.get("model_manifest_sha256_matches_local") is True
        and reporter_parity_comparison.get("model_file_hashes_match_local") is True
        and reporter_parity_comparison.get("chat_id_matches_reporter_log") is True
        and reporter_parity_comparison.get("response_id_matches_reporter_log") is True
        and reporter_parity_comparison.get("response_active_at_cancel_recorded") is True
        and reporter_parity_comparison.get("raw_sse_cancel_lifecycle_present") is True
        and reporter_server_hash_parity.get("failure")
        == "reporter_installed_server_hash_drift"
        and reporter_server_hash_parity.get("route_markers_match") is True
        and reporter_server_hash_parity.get("reporter_matches_source") is False
        and reporter_server_hash_parity.get("reporter_matches_local_installed") is False
        and reporter_server_hash_parity.get("reporter_matches_latest_public") is False
        and provenance.get("failure") == "reporter_server_hash_provenance_unknown"
        and not public_release_matches
        and all(proven.get(key) is True for key in required_clean_current_proofs)
    )


def build_not_proven_items(
    *,
    reporter_parity_comparison: dict[str, Any],
    reporter_server_hash_parity: dict[str, Any] | None = None,
    reporter_parity_artifact: dict[str, Any],
    reporter: dict[str, Any],
    local_reporter_prompt_reproduction: dict[str, Any],
    proven: dict[str, Any] | None = None,
) -> list[str]:
    items: list[str] = []
    if not (
        reporter_parity_comparison.get("model_file_hashes_match_local") is True
    ):
        items.append("reporter model shard/codebook hashes match local full K artifact")
    if not (
        reporter_parity_comparison.get("model_manifest_sha256_matches_local") is True
    ):
        items.append(
            "reporter model artifact manifest is available for direct local comparison"
        )
    if not (
        reporter_parity_comparison.get("server_hash_matches_local_installed") is True
        or reporter_hash_mismatch_is_stale_release_nonblocker(
            reporter_parity_comparison=reporter_parity_comparison,
            reporter_server_hash_parity=reporter_server_hash_parity,
            proven=proven,
        )
    ):
        items.append(
            "reporter installed app bundle hash matches public/local server.py route proof"
        )
    if not (
        reporter_parity_comparison.get("response_active_at_cancel_recorded") is True
    ):
        items.append("reporter response id was still active when the cancel request was sent")
    if not (
        isinstance(reporter_parity_artifact.get("session_settings"), dict)
    ):
        items.append(
            "reporter chat/session/settings database state matches local diagnostic state"
        )
    if local_reporter_prompt_reproduction.get("clean") is not True:
        items.append(
            "a concrete prompt reproduces screenshot-shaped wrong-language or numeric garbage"
        )
    cancel_404_after_abort = (
        reporter.get("responses_cancel_404_after_econnreset_same_response_id") is True
        and reporter.get("responses_cancel_404_after_request_error") is True
    )
    if not cancel_404_after_abort:
        items.append(
            "the 404 cancel response caused the screenshot rather than followed the stream abort"
        )
    return items


def reasoning_text_from_proof(data: dict[str, Any], chat: dict[str, Any]) -> str:
    explicit = chat.get("reasoningText")
    if isinstance(explicit, str):
        return explicit
    persisted = data.get("persistedReasoningText")
    if isinstance(persisted, str):
        return persisted
    groups = data.get("persistedReasoningByMessage")
    chunks: list[str] = []
    if isinstance(groups, list):
        for group in groups:
            if not isinstance(group, list):
                continue
            for segment in group:
                if not isinstance(segment, dict):
                    continue
                text = segment.get("text")
                if isinstance(text, str) and text:
                    chunks.append(text)
    return "\n".join(chunks)


def reasoning_leaks_from_proof(data: dict[str, Any], chat: dict[str, Any]) -> dict[str, int | bool]:
    text = reasoning_text_from_proof(data, chat)
    raw = chat.get("reasoningRawParserTagLeak")
    if not isinstance(raw, bool):
        raw = bool(RAW_REAL_UI_PARSER_LEAK_RE.search(text))
    cjk = chat.get("reasoningCjkLeakCount")
    if not isinstance(cjk, int):
        cjk = len(re.findall(r"[\u3400-\u9FFF]", text))
    korean = chat.get("reasoningKoreanLeakCount")
    if not isinstance(korean, int):
        korean = len(re.findall(r"[\uAC00-\uD7AF]", text))
    numeric = chat.get("reasoningNumericRunCount")
    if not isinstance(numeric, int):
        numeric = len(REASONING_NUMERIC_GARBAGE_RE.findall(text))
    return {
        "raw": raw,
        "cjk": cjk,
        "korean": korean,
        "numeric": numeric,
    }


def analyze_reporter_log(root: Path) -> dict[str, Any]:
    path = root / REPORTER_LOG
    text = read_text(path)
    response_ids = sorted(set(re.findall(r"\bresp_[A-Za-z0-9_]+\b", text)))
    launch_config: dict[str, Any] = {}
    model_config: dict[str, Any] = {}
    runtime_config: dict[str, Any] = {}
    request_error_line = None
    request_error_response_id = None
    request_error_payload: dict[str, Any] = {}
    request_shape_payload: dict[str, Any] = {}
    resolved_sampling_kwargs: dict[str, Any] = {}
    cancel_404_line = None
    cancel_404_response_id = None
    latest_response_id = None
    for idx, line in enumerate(text.splitlines(), start=1):
        if not launch_config and "$ /Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3" in line:
            launch_config = _parse_launch_config(line)
        if not model_config and "Model config:" in line:
            model_config = _parse_model_config(line)
        if "Native tool format enabled for parser:" in line:
            match = re.search(r"Native tool format enabled for parser: (\S+)", line)
            if match:
                runtime_config["native_tool_format"] = match.group(1)
        if "Reasoning parser enabled:" in line:
            match = re.search(r"Reasoning parser enabled: (\S+)", line)
            if match:
                runtime_config["reasoning_parser"] = match.group(1)
        if "Default max tokens fallback:" in line:
            match = re.search(r"Default max tokens fallback: (\d+)", line)
            if match:
                runtime_config["default_max_tokens_fallback"] = int(match.group(1))
        if "Paged cache: block_size=" in line:
            match = re.search(r"Paged cache: block_size=(\d+), max_blocks=(\d+)", line)
            if match:
                runtime_config["paged_cache"] = {
                    "block_size": int(match.group(1)),
                    "max_blocks": int(match.group(2)),
                }
        if "Block disk cache: max=" in line:
            match = re.search(r"Block disk cache: max=([0-9.]+)GB", line)
            if match:
                runtime_config["block_disk_cache_max_gb"] = float(match.group(1))
        if "KV cache quantization:" in line:
            match = re.search(r"KV cache quantization: q(\d+) \(group_size=(\d+)\)", line)
            if match:
                runtime_config["kv_cache_quantization"] = {
                    "bits": int(match.group(1)),
                    "group_size": int(match.group(2)),
                }
        if "Runtime cache layout:" in line and "TurboQuantKVCache" in line:
            layout = line.split("layout=", 1)[-1]
            cache_types = {
                item.split(":", 1)[1].strip()
                for item in layout.split(";")
                if ":" in item
            }
            runtime_config["runtime_cache_all_turboquant"] = cache_types == {
                "TurboQuantKVCache"
            }
        if "max_num_seqs=1" in line and "SingleBatchGenerator" in line:
            runtime_config["single_active_scheduler"] = True
        line_response_ids = re.findall(r"\bresp_[A-Za-z0-9_]+\b", line)
        if line_response_ids:
            latest_response_id = line_response_ids[-1]
        if request_error_line is None and "[CHAT_DIAG] request_error=" in line:
            request_error_line = idx
            request_error_response_id = latest_response_id
            _, payload = line.split("[CHAT_DIAG] request_error=", 1)
            try:
                decoded = json.loads(payload)
                if isinstance(decoded, dict):
                    request_error_payload = decoded
            except json.JSONDecodeError:
                request_error_payload = {}
        if not request_shape_payload and "[CHAT_DIAG] request_shape=" in line:
            _, payload = line.split("[CHAT_DIAG] request_shape=", 1)
            try:
                decoded = json.loads(payload)
                if isinstance(decoded, dict):
                    request_shape_payload = decoded
            except json.JSONDecodeError:
                request_shape_payload = {}
        if not resolved_sampling_kwargs and "Resolved sampling kwargs" in line:
            match = re.search(r"kwargs=(\{.*\})", line)
            if match:
                try:
                    decoded = ast.literal_eval(match.group(1))
                    if isinstance(decoded, dict):
                        resolved_sampling_kwargs = decoded
                except (SyntaxError, ValueError):
                    resolved_sampling_kwargs = {}
        if cancel_404_line is None:
            cancel_match = re.search(
                r'POST /v1/responses/(resp_[^/\s]+)/cancel HTTP/1\.1" 404',
                line,
            )
            if cancel_match:
                cancel_404_line = idx
                cancel_404_response_id = cancel_match.group(1)
    prompt_chars = None
    match = re.search(r'"messages":\[\{"role":"user","content":\{"type":"text","chars":(\d+)\}\}\]', text)
    if match:
        prompt_chars = int(match.group(1))
    return {
        "path": str(REPORTER_LOG),
        "exists": path.exists(),
        "sha256": sha256_file(path),
        "installed_bundled_python_seen": "/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3" in text,
        "model_path_seen": "/Users/example/.omlx/models/MiniMax-M2.7-JANGTQ_K" in text,
        "responses_stream_request_seen": '"wireApi":"responses"' in text
        and '"stream":true' in text,
        "reasoning_parser_seen": "--reasoning-parser minimax_m2" in text,
        "tool_parser_seen": "--tool-call-parser minimax" in text,
        "paged_cache_seen": "--use-paged-cache" in text,
        "block_disk_cache_seen": "--enable-block-disk-cache" in text,
        "turboquant_runtime_seen": "TurboQuantKVCache" in text
        and "native TurboQuant fast path" in text,
        "launch_config": launch_config,
        "model_config": model_config,
        "runtime_config": runtime_config,
        "response_ids": response_ids,
        "request_error_before_visible_content": "request_error=" in text
        and '"fullContentLen":0' in text,
        "request_shape": request_shape_payload,
        "resolved_sampling_kwargs": resolved_sampling_kwargs,
        "request_error": request_error_payload,
        "econnreset_seen": "ECONNRESET" in text,
        "responses_cancel_404_seen": bool(
            re.search(r'POST /v1/responses/resp_[^ ]+/cancel HTTP/1\.1" 404', text)
        ),
        "request_error_line": request_error_line,
        "request_error_response_id": request_error_response_id,
        "responses_cancel_404_line": cancel_404_line,
        "responses_cancel_404_response_id": cancel_404_response_id,
        "responses_cancel_404_after_request_error": (
            request_error_line is not None
            and cancel_404_line is not None
            and cancel_404_line > request_error_line
            and request_error_response_id == cancel_404_response_id
        ),
        "responses_cancel_404_after_econnreset_same_response_id": (
            request_error_payload.get("code") == "ECONNRESET"
            and request_error_payload.get("fullContentLen") == 0
            and request_error_payload.get("readerAcquired") is True
            and request_error_line is not None
            and cancel_404_line is not None
            and cancel_404_line > request_error_line
            and request_error_response_id == cancel_404_response_id
        ),
        "prompt_chars": prompt_chars,
        "bad_text_captured_in_log": False,
    }


def analyze_local_proofs(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for rel in LOCAL_REAL_UI_PROOFS:
        path = root / rel
        data = read_json(path)
        chat = data.get("chat") if isinstance(data.get("chat"), dict) else {}
        chat_overrides = (
            data.get("chatOverrides") if isinstance(data.get("chatOverrides"), dict) else {}
        )
        server_cache_controls = (
            data.get("serverCacheControls")
            if isinstance(data.get("serverCacheControls"), dict)
            else {}
        )
        server_command = data.get("serverCommand")
        if not isinstance(server_command, list):
            server_command = []
        surfaces = set(data.get("provenSurfaces") or [])
        reasoning_leaks = reasoning_leaks_from_proof(data, chat)
        clean = (
            path.exists()
            and data.get("modelName") == "MiniMax-M2.7-JANGTQ_K"
            and data.get("requestedWireApi") == "responses"
            and "real_loaded_model" in surfaces
            and "language_leak_check" in surfaces
            and "parser_leak_check" in surfaces
            and not data.get("sendErrors")
            and data.get("rendererFailureStage") is None
            and not data.get("rawParserLeak")
            and not reasoning_leaks["raw"]
            and int(chat.get("cjkLeakCount") or 0) == 0
            and int(chat.get("koreanLeakCount") or 0) == 0
            and reasoning_leaks["cjk"] == 0
            and reasoning_leaks["korean"] == 0
            and reasoning_leaks["numeric"] == 0
        )
        rows.append(
            {
                "path": str(rel),
                "exists": path.exists(),
                "sha256": sha256_file(path),
                "ui_launch_mode": data.get("uiLaunchMode") or "dev",
                "installed_app_path": data.get("installedAppPath"),
                "python": data.get("python"),
                "renderer_wire_api": data.get("rendererWireApi"),
                "requested_enable_thinking": data.get("requestedEnableThinking"),
                "renderer_enable_thinking": data.get("rendererEnableThinking"),
                "requested_builtin_tools": data.get("requestedBuiltinTools"),
                "renderer_builtin_tools_enabled": data.get(
                    "rendererBuiltinToolsEnabled"
                ),
                "server_cache_controls": server_cache_controls,
                "chat_overrides": {
                    key: chat_overrides.get(key)
                    for key in (
                        "wireApi",
                        "enableThinking",
                        "maxTokens",
                        "builtinToolsEnabled",
                        "toolResultMaxChars",
                    )
                    if key in chat_overrides
                },
                "server_command_has_cache_flags": all(
                    item in server_command
                    for item in (
                        "--use-paged-cache",
                        "--enable-block-disk-cache",
                        "--max-num-seqs",
                        "1",
                    )
                ),
                "model": data.get("modelName"),
                "wire_api": data.get("requestedWireApi"),
                "cache_hit_tokens": (data.get("cache") or {}).get("cacheHitTokens"),
                "final_visible_text": chat.get("finalVisibleText"),
                "assistant_count": data.get("assistantCount"),
                "surfaces": sorted(surfaces),
                "reasoning_leaks": reasoning_leaks,
                "clean": clean,
            }
        )
    installed_rows = [
        row for row in rows if row.get("ui_launch_mode") == "installed-app"
    ]
    installed_session_settings_parity = {
        "installed_proof_count": len(installed_rows),
        "all_installed_use_bundled_python": bool(installed_rows)
        and all(
            row.get("installed_app_path") == "/Applications/vMLX.app"
            and row.get("python")
            == "/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3"
            for row in installed_rows
        ),
        "all_installed_use_responses": bool(installed_rows)
        and all(
            row.get("wire_api") == "responses"
            and row.get("renderer_wire_api") == "responses"
            and (row.get("chat_overrides") or {}).get("wireApi") == "responses"
            for row in installed_rows
        ),
        "all_installed_disable_builtin_tools": bool(installed_rows)
        and all(
            row.get("requested_builtin_tools") is False
            and row.get("renderer_builtin_tools_enabled") is False
            and (row.get("chat_overrides") or {}).get("builtinToolsEnabled")
            is False
            for row in installed_rows
        ),
        "all_installed_server_cache_controls_disabled": bool(installed_rows)
        and all(
            (row.get("server_cache_controls") or {}).get("requested") is False
            and (row.get("server_cache_controls") or {}).get("verified") is False
            for row in installed_rows
        ),
        "all_installed_have_cache_flags": bool(installed_rows)
        and all(row.get("server_command_has_cache_flags") is True for row in installed_rows),
        "has_thinking_512_code_clean": any(
            row.get("clean") is True
            and row.get("requested_enable_thinking") is True
            and row.get("renderer_enable_thinking") is True
            and (row.get("chat_overrides") or {}).get("maxTokens") == 512
            and "reasoning_display" in set(row.get("surfaces") or [])
            and "def add_one" in str(row.get("final_visible_text") or "")
            and "return x + 1" in str(row.get("final_visible_text") or "")
            for row in installed_rows
        ),
        "has_hi_auto_clean_repeat": any(
            row.get("clean") is True
            and row.get("requested_enable_thinking") is None
            and row.get("renderer_enable_thinking") is None
            and str(row.get("final_visible_text") or "").startswith("Hi there!")
            for row in installed_rows
        ),
    }
    return {
        "proofs": rows,
        "clean_count": sum(1 for row in rows if row["clean"]),
        "required_count": len(rows),
        "all_required_clean": all(row["clean"] for row in rows),
        "installed_session_settings_parity": installed_session_settings_parity,
    }


def analyze_source_contract(root: Path) -> dict[str, Any]:
    server = read_text(root / "vmlx_engine/server.py")
    panel = read_text(root / "panel/src/main/ipc/chat.ts")
    cancellation_tests = read_text(root / "tests/test_cancellation.py")
    return {
        "source_hashes": {
            str(path): sha256_file(root / path) for path in SOURCE_CONTRACT_FILES
        },
        "server_has_responses_cancel_route": '@app.post("/v1/responses/{response_id}/cancel"' in server,
        "server_cancel_calls_engine_abort": "async def cancel_response" in server
        and "await _engine.abort_request(response_id)" in server,
        "panel_routes_resp_ids_to_responses_cancel": 'entry.responseId.startsWith("resp_")' in panel
        and "/v1/responses/${entry.responseId}/cancel" in panel,
        "test_proves_successful_responses_cancel_route": (
            "test_cancel_responses_endpoint_calls_engine_with_response_id"
            in cancellation_tests
            and "assert resp.status_code == 200" in cancellation_tests
            and 'assert_awaited_once_with("resp_issue179_cancel_probe")'
            in cancellation_tests
        ),
        "test_proves_inactive_responses_cancel_404_after_engine_lookup": (
            "test_cancel_responses_inactive_id_returns_404_after_engine_lookup"
            in cancellation_tests
            and 'client.post("/v1/responses/resp_issue179_finished/cancel")'
            in cancellation_tests
            and "assert resp.status_code == 404" in cancellation_tests
            and "not found or already finished" in cancellation_tests
        ),
    }


def analyze_local_installed_bundle_contract() -> dict[str, Any]:
    server = read_text(LOCAL_INSTALLED_BUNDLE_SERVER)
    return {
        "server_path": str(LOCAL_INSTALLED_BUNDLE_SERVER),
        "exists": LOCAL_INSTALLED_BUNDLE_SERVER.exists(),
        "sha256": sha256_file(LOCAL_INSTALLED_BUNDLE_SERVER),
        "server_has_responses_cancel_route": '@app.post("/v1/responses/{response_id}/cancel"' in server,
        "server_cancel_calls_engine_abort": "async def cancel_response" in server
        and "await _engine.abort_request(response_id)" in server,
    }


def analyze_local_responses_cancel_probe(root: Path) -> dict[str, Any]:
    path = root / LOCAL_RESPONSES_CANCEL_PROOF
    data = read_json(path)
    probe = data.get("probe") if isinstance(data.get("probe"), dict) else {}
    raw = data.get("raw") if isinstance(data.get("raw"), dict) else {}
    request = data.get("request") if isinstance(data.get("request"), dict) else {}
    return {
        "path": str(LOCAL_RESPONSES_CANCEL_PROOF),
        "exists": path.exists(),
        "sha256": sha256_file(path),
        "status": data.get("status"),
        "probe": probe,
        "response_id_seen": isinstance(raw.get("response_id"), str)
        and raw.get("response_id", "").startswith("resp_"),
        "cancel_status": raw.get("cancel_status"),
        "prompt": request.get("input"),
        "model": request.get("model"),
        "max_output_tokens": request.get("max_output_tokens"),
    }


def analyze_live_probe_memory_preflight(root: Path) -> dict[str, Any]:
    path = root / LOCAL_LIVE_PROBE_MEMORY_PREFLIGHT
    data = read_json(path)
    return {
        "path": str(LOCAL_LIVE_PROBE_MEMORY_PREFLIGHT),
        "exists": path.exists(),
        "sha256": sha256_file(path),
        "status": data.get("status"),
        "reason": data.get("reason"),
        "launch_decision": data.get("launch_decision"),
        "launch_allowed": data.get("launch_allowed"),
        "model_path": data.get("model_path"),
        "model_size_gb": data.get("model_size_gb"),
        "required_free_gb": data.get("required_free_gb"),
        "available_for_gate_gb": data.get("available_for_gate_gb"),
        "free_plus_speculative_purgeable_gb": data.get(
            "free_plus_speculative_purgeable_gb"
        ),
        "memory_gap_gb": data.get("memory_gap_gb"),
        "preflight_memory_source": data.get("preflight_memory_source"),
        "preflight_captured_at": data.get("preflight_captured_at"),
        "active_heavy_process_count": data.get("active_heavy_process_count"),
    }


def _bad_text_counts(text: str) -> dict[str, int | bool]:
    return {
        "raw_parser_tag": bool(RAW_REAL_UI_PARSER_LEAK_RE.search(text)),
        "cjk": len(re.findall(r"[\u3400-\u9FFF]", text)),
        "korean": len(re.findall(r"[\uAC00-\uD7AF]", text)),
        "numeric": len(REASONING_NUMERIC_GARBAGE_RE.findall(text)),
    }


def analyze_local_reporter_prompt_reproduction(root: Path) -> dict[str, Any]:
    configured_path = root / LOCAL_REPORTER_PROMPT_REPRODUCTION_PROOF
    path = configured_path
    selected_fallback: Path | None = None
    if not path.exists():
        for fallback in LOCAL_REPORTER_PROMPT_REPRODUCTION_FALLBACKS:
            candidate = root / fallback
            if candidate.exists():
                path = candidate
                selected_fallback = fallback
                break
    data = read_json(path)
    raw = data.get("raw") if isinstance(data.get("raw"), dict) else {}
    request = data.get("request") if isinstance(data.get("request"), dict) else {}
    probe = data.get("probe") if isinstance(data.get("probe"), dict) else {}
    content = str(raw.get("raw_content_text") or "")
    reasoning = str(raw.get("raw_reasoning_text") or "")
    text_counts = _bad_text_counts("\n".join((content, reasoning)))
    observed_stream_text = bool(content.strip()) or bool(reasoning.strip())
    request_matches_reporter = (
        request.get("input") == "Hi"
        and request.get("model") == "models/MiniMax-M2.7-JANGTQ_K"
        and request.get("stream") is True
        and request.get("temperature") == 1.0
        and request.get("top_p") == 0.95
        and request.get("top_k") == 40
        and request.get("max_output_tokens") == 4096
    )
    clean = (
        path.exists()
        and raw.get("stream_started") is True
        and isinstance(raw.get("response_id"), str)
        and raw.get("cancel_status") == 200
        and request_matches_reporter
        and probe.get("bad_text_captured") is False
        and observed_stream_text
        and text_counts == {
            "raw_parser_tag": False,
            "cjk": 0,
            "korean": 0,
            "numeric": 0,
        }
    )
    return {
        "path": str(selected_fallback or LOCAL_REPORTER_PROMPT_REPRODUCTION_PROOF),
        "configured_path": str(LOCAL_REPORTER_PROMPT_REPRODUCTION_PROOF),
        "selected_fallback_path": str(selected_fallback) if selected_fallback else None,
        "exists": path.exists(),
        "sha256": sha256_file(path),
        "status": data.get("status"),
        "request_matches_reporter": request_matches_reporter,
        "stream_started": raw.get("stream_started"),
        "response_id_seen": isinstance(raw.get("response_id"), str),
        "cancel_status": raw.get("cancel_status"),
        "cancel_trigger": raw.get("cancel_trigger"),
        "bad_text_captured": probe.get("bad_text_captured"),
        "content_text": content,
        "reasoning_text": reasoning,
        "bad_text_counts": text_counts,
        "observed_stream_text": observed_stream_text,
        "clean": clean,
    }


def analyze_local_model_manifest(root: Path) -> dict[str, Any]:
    path = root / LOCAL_MODEL_MANIFEST
    data = read_json(path)
    checks = data.get("checks") if isinstance(data.get("checks"), dict) else {}
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    return {
        "path": str(LOCAL_MODEL_MANIFEST),
        "exists": path.exists(),
        "sha256": sha256_file(path),
        "status": data.get("status"),
        "model_file_hashes": [
            {"path": row.get("path"), "sha256": row.get("sha256")}
            for row in data.get("files", [])
            if isinstance(row, dict)
            and isinstance(row.get("path"), str)
            and isinstance(row.get("sha256"), str)
        ],
        "model_path": data.get("model_path"),
        "total_bytes": data.get("total_bytes"),
        "total_file_count": data.get("total_file_count"),
        "model_shard_count": data.get("model_shard_count"),
        "hash_shards": data.get("hash_shards"),
        "hashed_file_count": summary.get("hashed_file_count"),
        "unhashed_safetensors_count": summary.get("unhashed_safetensors_count"),
        "checks": checks,
        "local_full_k_artifact_shape_recorded": (
            data.get("status") == "pass"
            and checks.get("has_config") is True
            and checks.get("has_generation_config") is True
            and checks.get("has_jang_config") is True
            and checks.get("has_model_index") is True
            and checks.get("has_tokenizer") is True
            and checks.get("has_jangtq_runtime") is True
            and checks.get("model_shard_count_is_67") is True
        ),
    }


def _public_release_dmg_contract_from_path(root: Path, path: Path) -> dict[str, Any]:
    data = read_json(path)
    return {
        "path": str(path.relative_to(root)),
        "exists": path.exists(),
        "sha256": sha256_file(path),
        "release_tag": data.get("release_tag"),
        "asset": data.get("asset"),
        "asset_size_bytes": data.get("asset_size_bytes"),
        "server_sha256": data.get("server_sha256"),
        "server_has_responses_cancel_route": data.get(
            "server_has_responses_cancel_route"
        )
        is True,
        "server_cancel_calls_engine_abort": data.get(
            "server_cancel_calls_engine_abort"
        )
        is True,
        "boundary": data.get("boundary"),
    }


def analyze_public_release_dmg_contract(root: Path) -> dict[str, Any]:
    return _public_release_dmg_contract_from_path(root, root / PUBLIC_RELEASE_DMG_CONTRACT)


def analyze_public_release_dmg_contracts(root: Path) -> list[dict[str, Any]]:
    contract_root = root / "build/issue-179"
    paths = sorted(contract_root.glob(PUBLIC_RELEASE_DMG_CONTRACT_GLOB))
    if not paths and (root / PUBLIC_RELEASE_DMG_CONTRACT).exists():
        paths = [root / PUBLIC_RELEASE_DMG_CONTRACT]
    rows = [_public_release_dmg_contract_from_path(root, path) for path in paths]
    if rows:
        return rows
    for fallback in PUBLIC_RELEASE_DMG_CONTRACT_FALLBACK_AUDITS:
        data = read_json(root / fallback)
        fallback_rows = data.get("public_release_dmg_contracts")
        if isinstance(fallback_rows, list) and fallback_rows:
            return [row for row in fallback_rows if isinstance(row, dict)]
    return []


def _release_tag_key(row: dict[str, Any]) -> tuple[int, int, int]:
    tag = row.get("release_tag")
    if not isinstance(tag, str):
        return (0, 0, 0)
    match = PUBLIC_RELEASE_TAG_RE.match(tag)
    if not match:
        return (0, 0, 0)
    return tuple(int(part) for part in match.groups())


def select_latest_public_release_dmg_contract(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {}
    return max(
        rows,
        key=lambda row: (
            _release_tag_key(row),
            1 if str(row.get("asset") or "").endswith("-tahoe-arm64.dmg") else 0,
            str(row.get("asset") or ""),
        ),
    )


def build_bundle_hash_parity(
    *,
    source: dict[str, Any],
    installed_bundle: dict[str, Any],
    public_dmg: dict[str, Any],
    latest_public_dmg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_hashes = source.get("source_hashes")
    if not isinstance(source_hashes, dict):
        source_hashes = {}
    source_server_sha = source_hashes.get("vmlx_engine/server.py")
    local_server_sha = installed_bundle.get("sha256")
    public_server_sha = public_dmg.get("server_sha256")
    latest_public_server_sha = (latest_public_dmg or {}).get("server_sha256")
    return {
        "source_server_sha256": source_server_sha,
        "local_installed_server_sha256": local_server_sha,
        "public_v1549_tahoe_server_sha256": public_server_sha,
        "latest_public_server_sha256": latest_public_server_sha,
        "source_matches_local_installed": (
            bool(source_server_sha)
            and bool(local_server_sha)
            and source_server_sha == local_server_sha
        ),
        "source_matches_public_v1549_tahoe": (
            bool(source_server_sha)
            and bool(public_server_sha)
            and source_server_sha == public_server_sha
        ),
        "local_installed_matches_public_v1549_tahoe": (
            bool(local_server_sha)
            and bool(public_server_sha)
            and local_server_sha == public_server_sha
        ),
        "source_matches_latest_public": (
            bool(source_server_sha)
            and bool(latest_public_server_sha)
            and source_server_sha == latest_public_server_sha
        ),
        "local_installed_matches_latest_public": (
            bool(local_server_sha)
            and bool(latest_public_server_sha)
            and local_server_sha == latest_public_server_sha
        ),
    }


def build_reporter_server_hash_parity(
    *,
    reporter_parity_artifact: dict[str, Any],
    source: dict[str, Any],
    installed_bundle: dict[str, Any],
    public_dmg: dict[str, Any],
    latest_public_dmg: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_hashes = source.get("source_hashes")
    if not isinstance(source_hashes, dict):
        source_hashes = {}
    reporter_sha = reporter_parity_artifact.get("installed_server_sha256")
    source_sha = source_hashes.get("vmlx_engine/server.py")
    local_sha = installed_bundle.get("sha256")
    public_sha = public_dmg.get("server_sha256")
    latest_public_sha = (latest_public_dmg or {}).get("server_sha256")
    route_markers_match = (
        reporter_parity_artifact.get("server_has_responses_cancel_route") is True
        and reporter_parity_artifact.get("server_cancel_calls_engine_abort") is True
    )
    reporter_matches_source = (
        bool(reporter_sha) and bool(source_sha) and reporter_sha == source_sha
    )
    reporter_matches_local = (
        bool(reporter_sha) and bool(local_sha) and reporter_sha == local_sha
    )
    reporter_matches_public = (
        bool(reporter_sha) and bool(public_sha) and reporter_sha == public_sha
    )
    reporter_matches_latest_public = (
        bool(reporter_sha)
        and bool(latest_public_sha)
        and reporter_sha == latest_public_sha
    )
    status = (
        "pass"
        if reporter_matches_source and reporter_matches_local and route_markers_match
        else "open"
    )
    out = {
        "reporter_installed_server_sha256": reporter_sha,
        "source_server_sha256": source_sha,
        "local_installed_server_sha256": local_sha,
        "public_v1549_tahoe_server_sha256": public_sha,
        "latest_public_server_sha256": latest_public_sha,
        "reporter_matches_source": reporter_matches_source,
        "reporter_matches_local_installed": reporter_matches_local,
        "reporter_matches_public_v1549_tahoe": reporter_matches_public,
        "reporter_matches_latest_public": reporter_matches_latest_public,
        "route_markers_match": route_markers_match,
        "status": status,
    }
    if provenance is not None:
        out["provenance"] = provenance
    if status != "pass":
        out["failure"] = "reporter_installed_server_hash_drift"
    return out


def _git_history_server_hash_match(root: Path, reporter_sha: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "checked": False,
        "match": False,
        "commit": None,
        "error": None,
    }
    try:
        revs = subprocess.run(
            ["git", "rev-list", "--all", "--", "vmlx_engine/server.py"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        result["error"] = type(exc).__name__
        return result

    result["checked"] = True
    for commit in revs.stdout.splitlines():
        try:
            show = subprocess.run(
                ["git", "show", f"{commit}:vmlx_engine/server.py"],
                cwd=root,
                check=True,
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
        digest = hashlib.sha256(show.stdout).hexdigest()
        if digest == reporter_sha:
            result["match"] = True
            result["commit"] = commit
            return result
    return result


def _sibling_python_worktree_server_hashes(
    root: Path,
    reporter_sha: str,
) -> list[dict[str, Any]]:
    resolved_root = root.resolve()
    rows = []
    for path in sorted(resolved_root.parent.glob("vllm-mlx*/vmlx_engine/server.py")):
        if resolved_root in path.resolve().parents:
            continue
        digest = sha256_file(path)
        rows.append(
            {
                "path": str(path),
                "sha256": digest,
                "matches_reporter": digest == reporter_sha,
            }
        )
    return rows


def _local_app_bundle_server_hashes(
    root: Path,
    reporter_sha: str,
    search_roots: list[Path] | None = None,
) -> list[dict[str, Any]]:
    if search_roots is None:
        search_roots = [
            root,
            root.resolve().parent,
            Path.home() / ".config/superpowers/worktrees/vllm-mlx",
        ]
    rows = []
    seen: set[Path] = set()
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in sorted(
            search_root.glob("**/vMLX.app/Contents/Resources/**/vmlx_engine/server.py")
        ):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            digest = sha256_file(path)
            rows.append(
                {
                    "path": str(path),
                    "sha256": digest,
                    "matches_reporter": digest == reporter_sha,
                }
            )
    return rows


def build_reporter_server_hash_provenance(
    *,
    root: Path,
    reporter_sha: str | None,
    source_sha: str | None,
    local_sha: str | None,
    public_sha: str | None,
    public_dmg_contracts: list[dict[str, Any]] | None = None,
    required_public_release_contracts: set[tuple[str, str]] | None = None,
    app_bundle_search_roots: list[Path] | None = None,
    check_git_history: bool = True,
) -> dict[str, Any]:
    if not reporter_sha:
        return {
            "status": "missing",
            "failure": "missing_reporter_server_sha256",
            "checked_sources": [],
        }

    direct_matches = {
        "source": bool(source_sha) and reporter_sha == source_sha,
        "local_installed": bool(local_sha) and reporter_sha == local_sha,
        "public_v1549_tahoe": bool(public_sha) and reporter_sha == public_sha,
    }
    public_contract_rows = [
        row for row in (public_dmg_contracts or []) if isinstance(row, dict)
    ]
    public_release_matches = [
        {
            "asset": row.get("asset"),
            "path": row.get("path"),
            "release_tag": row.get("release_tag"),
            "server_sha256": row.get("server_sha256"),
        }
        for row in public_contract_rows
        if row.get("server_sha256") == reporter_sha
    ]
    public_release_checked = [
        {
            "asset": row.get("asset"),
            "path": row.get("path"),
            "release_tag": row.get("release_tag"),
            "server_sha256": row.get("server_sha256"),
            "matches_reporter": row.get("server_sha256") == reporter_sha,
        }
        for row in public_contract_rows
    ]
    public_release_seen = {
        (str(row.get("release_tag")), str(row.get("asset")))
        for row in public_contract_rows
        if row.get("release_tag") and row.get("asset")
    }
    missing_required_public_release_contracts = [
        {"release_tag": release_tag, "asset": asset}
        for release_tag, asset in sorted(required_public_release_contracts or set())
        if (release_tag, asset) not in public_release_seen
    ]
    backup_rows = []
    backup_root = root / "build/installed-app-backups"
    if backup_root.exists():
        for path in sorted(backup_root.glob("**/vmlx_engine/server.py")):
            digest = sha256_file(path)
            backup_rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": digest,
                    "matches_reporter": digest == reporter_sha,
                }
            )
    sibling_source_rows = _sibling_python_worktree_server_hashes(root, reporter_sha)
    local_app_bundle_rows = _local_app_bundle_server_hashes(
        root,
        reporter_sha,
        app_bundle_search_roots,
    )
    git_history = (
        _git_history_server_hash_match(root, reporter_sha)
        if check_git_history
        else {"checked": False, "match": False, "commit": None, "error": "skipped"}
    )
    required_public_coverage_ok = not missing_required_public_release_contracts
    matched = (
        any(direct_matches.values())
        or bool(public_release_matches)
        or any(row["matches_reporter"] for row in backup_rows)
        or any(row["matches_reporter"] for row in sibling_source_rows)
        or any(row["matches_reporter"] for row in local_app_bundle_rows)
        or git_history.get("match") is True
    )
    out: dict[str, Any] = {
        "status": "pass" if matched and required_public_coverage_ok else "open",
        "checked_sources": [
            "source_contract",
            "local_installed_bundle",
            "public_v1549_tahoe_dmg",
            "public_release_dmg_contracts",
            "local_installed_app_backups",
            "sibling_python_worktrees",
            "local_app_bundle_servers",
            "git_history",
        ],
        "direct_matches": direct_matches,
        "public_release_matches": public_release_matches,
        "public_release_checked": public_release_checked,
        "public_release_checked_count": len(public_contract_rows),
        "missing_required_public_release_contracts": (
            missing_required_public_release_contracts
        ),
        "local_backup_matches": [
            row for row in backup_rows if row["matches_reporter"]
        ],
        "local_backup_checked_count": len(backup_rows),
        "sibling_source_matches": [
            row for row in sibling_source_rows if row["matches_reporter"]
        ],
        "sibling_source_checked": sibling_source_rows,
        "sibling_source_checked_count": len(sibling_source_rows),
        "local_app_bundle_matches": [
            row for row in local_app_bundle_rows if row["matches_reporter"]
        ],
        "local_app_bundle_checked": local_app_bundle_rows,
        "local_app_bundle_checked_count": len(local_app_bundle_rows),
        "git_history": git_history,
    }
    if missing_required_public_release_contracts:
        out["failure"] = "missing_required_public_release_dmg_contracts"
    elif not matched:
        out["failure"] = "reporter_server_hash_provenance_unknown"
    return out


def build_root_cause_discriminators(
    *,
    reporter: dict[str, Any],
    local: dict[str, Any],
    source: dict[str, Any],
    installed_bundle: dict[str, Any],
    public_dmg: dict[str, Any],
    cancel_probe: dict[str, Any],
    bundle_hash_parity: dict[str, Any],
    local_model_manifest: dict[str, Any],
    reporter_parity_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    local_observed: set[str] = set()
    for proof in local.get("proofs", []):
        if not isinstance(proof, dict):
            continue
        surfaces = {str(item) for item in proof.get("surfaces", [])}
        if "cache_hit_telemetry" in surfaces:
            local_observed.add("paged_prefix_cache_hits")
        if proof.get("clean") is True:
            local_observed.add("clean_visible_text")

    if local.get("all_required_clean") is True:
        local_observed.add("single_active_scheduler")
        local_observed.add("block_disk_l2_writes_and_hits")
        local_observed.add("turboquant_kv_live_decode")

    return [
        {
            "id": "reporter_bundle_cancel_route_parity",
            "status": "open",
            "observed": [
                "reporter_responses_cancel_404"
                if reporter.get("responses_cancel_404_seen")
                else "reporter_cancel_route_unknown",
                "local_installed_cancel_route_present"
                if installed_bundle.get("server_has_responses_cancel_route")
                else "local_installed_cancel_route_missing",
                "public_v1549_tahoe_dmg_cancel_route_present"
                if public_dmg.get("server_has_responses_cancel_route")
                else "public_v1549_tahoe_dmg_cancel_route_unproven",
                "local_installed_live_cancel_200"
                if cancel_probe.get("cancel_status") == 200
                else "local_installed_live_cancel_unproven",
                "source_proves_inactive_response_cancel_404_after_lookup"
                if source.get(
                    "test_proves_inactive_responses_cancel_404_after_engine_lookup"
                )
                else "source_inactive_response_cancel_404_boundary_unproven",
                "local_installed_hash_matches_public_v1549_tahoe"
                if bundle_hash_parity.get("local_installed_matches_public_v1549_tahoe")
                else "local_installed_hash_differs_from_public_v1549_tahoe",
                "reporter_parity_artifact_ready"
                if reporter_parity_artifact.get("status") == "pass"
                else reporter_parity_artifact.get(
                    "comparison_status", "reporter_parity_artifact_unknown"
                ),
            ],
            "next_proof": (
                "Compare reporter installed bundle server.py hash/routes and "
                "request lifecycle against the local rebuilt bundle and public "
                "v1.5.49 Tahoe DMG. The public DMG has the route, so the reporter "
                "404 is more likely an inactive/already-finished response id after "
                "stream abort than a missing route."
            ),
        },
        {
            "id": "reporter_stream_abort_vs_model_text",
            "status": "open",
            "observed": [
                "abort_before_visible_content"
                if reporter.get("request_error_before_visible_content")
                else "visible_content_seen",
                "reporter_cancel_404_after_stream_abort"
                if reporter.get("responses_cancel_404_after_request_error")
                else "reporter_cancel_404_order_unproven",
                "bad_text_not_captured_in_log"
                if not reporter.get("bad_text_captured_in_log")
                else "bad_text_captured_in_log",
                "local_controlled_cancel_bad_text_absent"
                if (cancel_probe.get("probe") or {}).get("bad_text_captured") is False
                else "local_controlled_cancel_bad_text_unproven",
            ],
            "next_proof": (
                "Run the same raw SSE/cancel capture on the reporter bundle/session; "
                "local synced installed app already captured the controlled interrupt "
                "boundary without screenshot-shaped garbage."
            ),
        },
        {
            "id": "local_runtime_cache_math_parity",
            "status": "partially_proven" if local.get("all_required_clean") else "open",
            "observed": [
                item
                for item in (
                    "single_active_scheduler",
                    "paged_prefix_cache_hits",
                    "block_disk_l2_writes_and_hits",
                    "turboquant_kv_live_decode",
                )
                if item in local_observed
            ],
            "next_proof": (
                "Run the reporter exact prompt/session after bundle parity is proven; "
                "if garbage reproduces, A/B paged cache, L2, TurboQuant KV, and "
                "single-active scheduling without changing sampler defaults."
            ),
        },
        {
            "id": "prompt_template_parser_parity",
            "status": "open",
            "observed": [
                "reporter_minimax_reasoning_parser"
                if reporter.get("reasoning_parser_seen")
                else "reporter_reasoning_parser_unknown",
                "reporter_minimax_tool_parser"
                if reporter.get("tool_parser_seen")
                else "reporter_tool_parser_unknown",
                "source_routes_response_cancel"
                if source.get("panel_routes_resp_ids_to_responses_cancel")
                else "source_cancel_route_unknown",
            ],
            "next_proof": (
                "Capture rendered prompt, template flags, thinking setting, parser "
                "selection, and persisted session settings from reporter and local runs."
            ),
        },
        {
            "id": "model_artifact_hash_parity",
            "status": (
                "partially_proven"
                if local_model_manifest.get("local_full_k_artifact_shape_recorded")
                else "open"
            ),
            "observed": [
                "reporter_model_path_seen"
                if reporter.get("model_path_seen")
                else "reporter_model_path_missing",
                "local_full_k_artifact_manifest_recorded"
                if local_model_manifest.get("local_full_k_artifact_shape_recorded")
                else "local_full_k_artifact_manifest_missing",
                "local_full_k_shard_hashes_recorded"
                if local_model_manifest.get("hash_shards") is True
                else "local_full_k_shard_hashes_not_recorded",
            ],
            "next_proof": (
                "Compare reporter and local model safetensor/codebook/tokenizer/"
                "generation_config hashes before attributing output to runtime math."
            ),
        },
    ]


def build_language_planning_leak_isolation(
    *,
    reporter: dict[str, Any],
    local: dict[str, Any],
    local_reporter_prompt_reproduction: dict[str, Any],
    local_model_manifest: dict[str, Any],
) -> dict[str, Any]:
    """No-heavy MiniMax leak isolation map.

    This does not clear the issue. It names the remaining single-axis runtime
    proofs needed before blaming cache math, TQ KV, L2, parser/template, or
    model-owned generation defaults for the reporter's wrong-language/planning
    screenshot.
    """

    local_surfaces: set[str] = set()
    cache_hit_seen = False
    clean_rows = 0
    for proof in local.get("proofs", []):
        if not isinstance(proof, dict):
            continue
        local_surfaces.update(str(item) for item in proof.get("surfaces", []))
        cache_hit = proof.get("cache_hit_tokens")
        cache_hit_seen = cache_hit_seen or isinstance(cache_hit, int) and cache_hit > 0
        if proof.get("clean") is True:
            clean_rows += 1

    manifest_checks = local_model_manifest.get("checks")
    if not isinstance(manifest_checks, dict):
        manifest_checks = {}
    reporter_sampling_seen = bool(
        reporter.get("resolved_sampling_kwargs")
        or ((reporter.get("request_shape") or {}).get("body") or {}).get("sampling")
    )
    reporter_runtime = (
        reporter.get("runtime_config") if isinstance(reporter.get("runtime_config"), dict) else {}
    )
    local_clean = local.get("all_required_clean") is True
    exact_local_clean = local_reporter_prompt_reproduction.get("clean") is True
    axes = [
        {
            "axis": "reporter_exact_prompt_reproduction",
            "status": "open",
            "current_evidence": [
                "local_exact_prompt_clean"
                if exact_local_clean
                else "local_exact_prompt_reproduction_missing_or_not_clean",
                "reporter_bad_text_not_in_log"
                if not reporter.get("bad_text_captured_in_log")
                else "reporter_bad_text_captured_in_log",
            ],
            "required_next_evidence": [
                "reporter_machine_same_prompt_raw_sse_capture",
                "reporter_machine_visible_and_reasoning_text_capture",
                "reporter_machine_cancel_lifecycle_same_response_id",
            ],
        },
        {
            "axis": "generation_config_and_sampling",
            "status": (
                "partial"
                if manifest_checks.get("has_generation_config") is True
                and (
                    reporter_sampling_seen
                    or local_reporter_prompt_reproduction.get("request_matches_reporter") is True
                )
                else "open"
            ),
            "current_evidence": [
                "local_generation_config_present"
                if manifest_checks.get("has_generation_config") is True
                else "local_generation_config_missing",
                "reporter_or_local_sampling_shape_seen"
                if reporter_sampling_seen
                or local_reporter_prompt_reproduction.get("request_matches_reporter") is True
                else "sampling_shape_missing",
            ],
            "required_next_evidence": [
                "generation_config_hash_match",
                "resolved_sampling_kwargs_match",
                "no_sampler_clamp_or_prompt_only_fix",
            ],
        },
        {
            "axis": "parser_template_reasoning",
            "status": (
                "partial"
                if local_clean
                or reporter.get("reasoning_parser_seen")
                or reporter.get("tool_parser_seen")
                else "open"
            ),
            "current_evidence": [
                "local_parser_leak_checks_clean"
                if local_clean
                else "local_parser_leak_checks_not_all_clean",
                "reporter_reasoning_parser_seen"
                if reporter.get("reasoning_parser_seen")
                else "reporter_reasoning_parser_unknown",
                "reporter_tool_parser_seen"
                if reporter.get("tool_parser_seen")
                else "reporter_tool_parser_unknown",
            ],
            "required_next_evidence": [
                "rendered_chat_template_hash",
                "thinking_template_flag_match",
                "reasoning_parser_output_no_cjk_numeric_raw_leak",
                "tool_parser_no_raw_minimax_markup_leak",
            ],
        },
        {
            "axis": "paged_prefix_cache",
            "status": (
                "partial"
                if cache_hit_seen or reporter.get("paged_cache_seen") else "open"
            ),
            "current_evidence": [
                "local_cache_hit_telemetry_seen"
                if cache_hit_seen
                else "local_cache_hit_telemetry_missing",
                "reporter_paged_cache_flag_seen"
                if reporter.get("paged_cache_seen")
                else "reporter_paged_cache_flag_unknown",
            ],
            "required_next_evidence": [
                "cache_on_off_same_prompt",
                "first_miss_second_hit_cached_tokens",
                "cache_detail_paged_or_prefix",
                "same_visible_and_reasoning_text_across_cache_state",
            ],
        },
        {
            "axis": "block_disk_l2",
            "status": (
                "partial"
                if local_clean or reporter.get("block_disk_cache_seen") else "open"
            ),
            "current_evidence": [
                "local_installed_cache_flags_seen"
                if local.get("installed_session_settings_parity", {}).get(
                    "all_installed_have_cache_flags"
                )
                else "local_installed_cache_flags_missing",
                "reporter_block_disk_cache_flag_seen"
                if reporter.get("block_disk_cache_seen")
                else "reporter_block_disk_cache_flag_unknown",
            ],
            "required_next_evidence": [
                "fresh_process_l2_restore_same_prompt",
                "block_disk_l2_write_count",
                "block_disk_l2_hit_count",
                "same_visible_and_reasoning_text_after_restore",
            ],
        },
        {
            "axis": "turboquant_kv",
            "status": (
                "partial"
                if reporter_runtime.get("runtime_cache_all_turboquant") is True
                or local_clean
                else "open"
            ),
            "current_evidence": [
                "reporter_runtime_cache_all_turboquant"
                if reporter_runtime.get("runtime_cache_all_turboquant") is True
                else "reporter_tq_kv_runtime_unknown",
                "local_clean_with_current_cache_path"
                if local_clean
                else "local_current_cache_path_not_all_clean",
            ],
            "required_next_evidence": [
                "tq_kv_off_same_prompt",
                "tq_kv_on_same_prompt",
                "kv_encode_decode_shape_dtype_checksum",
                "same_visible_and_reasoning_text_with_tq_ab",
            ],
        },
    ]
    return {
        "status": "open",
        "local_clean_rows": clean_rows,
        "local_surfaces": sorted(local_surfaces),
        "axes": axes,
        "release_boundary": (
            "single_axis_runtime_ab_required: this matrix is proof planning and "
            "classification only. Do not clear MiniMax wrong-language/planning "
            "leak by disabling cache, TQ KV, L2, reasoning, or changing sampling "
            "unless the matching same-prompt A/B proof identifies that axis."
        ),
    }


def analyze_reporter_screenshot(root: Path) -> dict[str, Any]:
    screenshot = root / REPORTER_SCREENSHOT
    sha = sha256_file(screenshot)
    observation: dict[str, Any] = {
        "source": "manual_visual_inspection",
        "surface": None,
        "interrupted": None,
        "looks_like_numeric_sequence_garbage": None,
        "note": None,
    }
    if sha == "e131bee8c097680b3cb496f55458ad83258789eac81d242619b77e6a4600b972":
        observation.update(
            {
                "surface": "reasoning_panel",
                "interrupted": True,
                "looks_like_numeric_sequence_garbage": True,
                "note": (
                    "Screenshot shows a Hi prompt with Auto thinking, an expanded "
                    "Reasoning panel containing numeric/list-like garbage, and "
                    "[Generation interrupted] below it."
                ),
            }
        )
    return {
        "path": str(REPORTER_SCREENSHOT),
        "exists": screenshot.exists(),
        "sha256": sha,
        "manual_observation": observation,
    }


def build_audit(root: Path) -> dict[str, Any]:
    reporter = analyze_reporter_log(root)
    local = analyze_local_proofs(root)
    source = analyze_source_contract(root)
    installed_bundle = analyze_local_installed_bundle_contract()
    cancel_probe = analyze_local_responses_cancel_probe(root)
    live_probe_memory_preflight = analyze_live_probe_memory_preflight(root)
    local_reporter_prompt_reproduction = analyze_local_reporter_prompt_reproduction(
        root
    )
    local_model_manifest = analyze_local_model_manifest(root)
    reporter_parity_artifact = analyze_reporter_parity_artifact(root)
    public_dmg = analyze_public_release_dmg_contract(root)
    public_dmgs = analyze_public_release_dmg_contracts(root)
    latest_public_dmg = select_latest_public_release_dmg_contract(public_dmgs)
    screenshot = analyze_reporter_screenshot(root)
    bundle_hash_parity = build_bundle_hash_parity(
        source=source,
        installed_bundle=installed_bundle,
        public_dmg=public_dmg,
        latest_public_dmg=latest_public_dmg,
    )
    source_hashes = source.get("source_hashes")
    if not isinstance(source_hashes, dict):
        source_hashes = {}
    reporter_server_hash_provenance = build_reporter_server_hash_provenance(
        root=root,
        reporter_sha=reporter_parity_artifact.get("installed_server_sha256"),
        source_sha=source_hashes.get("vmlx_engine/server.py"),
        local_sha=installed_bundle.get("sha256"),
        public_sha=public_dmg.get("server_sha256"),
        public_dmg_contracts=public_dmgs,
        required_public_release_contracts=REQUIRED_PUBLIC_RELEASE_CONTRACTS,
    )
    reporter_server_hash_parity = build_reporter_server_hash_parity(
        reporter_parity_artifact=reporter_parity_artifact,
        source=source,
        installed_bundle=installed_bundle,
        public_dmg=public_dmg,
        latest_public_dmg=latest_public_dmg,
        provenance=reporter_server_hash_provenance,
    )
    reporter_parity_comparison = build_reporter_parity_comparison(
        reporter_parity_artifact=reporter_parity_artifact,
        reporter=reporter,
        installed_bundle=installed_bundle,
        local_model_manifest=local_model_manifest,
    )
    discriminators = build_root_cause_discriminators(
        reporter=reporter,
        local=local,
        source=source,
        installed_bundle=installed_bundle,
        public_dmg=public_dmg,
        cancel_probe=cancel_probe,
        bundle_hash_parity=bundle_hash_parity,
        local_model_manifest=local_model_manifest,
        reporter_parity_artifact=reporter_parity_artifact,
    )
    language_planning_leak_isolation = build_language_planning_leak_isolation(
        reporter=reporter,
        local=local,
        local_reporter_prompt_reproduction=local_reporter_prompt_reproduction,
        local_model_manifest=local_model_manifest,
    )
    proven = {
        "reporter_log_installed_app_bundled_python_seen": reporter[
            "installed_bundled_python_seen"
        ],
        "reporter_log_request_shape_and_sampling_kwargs_seen": (
            (reporter.get("request_shape") or {}).get("wireApi") == "responses"
            and (reporter.get("request_shape") or {}).get("detectedFamily")
            == "minimax"
            and ((reporter.get("request_shape") or {}).get("body") or {}).get("route")
            == "/v1/responses"
            and ((reporter.get("request_shape") or {}).get("body") or {}).get("stream")
            is True
            and reporter.get("resolved_sampling_kwargs") == {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 40,
                "max_tokens": 4096,
            }
        ),
        "reporter_log_launch_parser_cache_flags_seen": (
            reporter.get("launch_config") == {
                "executable": "/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3",
                "module": "vmlx_engine.cli",
                "subcommand": "serve",
                "model_path": "/Users/example/.omlx/models/MiniMax-M2.7-JANGTQ_K",
                "host": "127.0.0.1",
                "port": 8000,
                "timeout": 300,
                "max_num_seqs": 1,
                "prefill_batch_size": 512,
                "prefill_step_size": 2048,
                "completion_batch_size": 512,
                "continuous_batching": True,
                "tool_call_parser": "minimax",
                "enable_auto_tool_choice": True,
                "reasoning_parser": "minimax_m2",
                "use_paged_cache": True,
                "paged_cache_block_size": 64,
                "max_cache_blocks": 1000,
                "enable_block_disk_cache": True,
                "block_disk_cache_max_gb": 10,
                "stream_interval": 1,
            }
            and reporter.get("model_config") == {
                "detection_source": "jang_stamped",
                "family": "minimax",
                "reasoning_parser": "minimax_m2",
                "tool_parser": "minimax",
                "think_in_template": True,
                "cache_type": "kv",
                "cache_subtype": None,
                "is_mllm": False,
            }
            and _runtime_config_proven(reporter.get("runtime_config") or {})
        ),
        "reporter_log_has_abort_before_visible_content": reporter[
            "request_error_before_visible_content"
        ],
        "reporter_log_has_responses_cancel_404": reporter["responses_cancel_404_seen"],
        "reporter_cancel_404_after_stream_abort_order_proven": reporter[
            "responses_cancel_404_after_request_error"
        ],
        "reporter_cancel_404_after_econnreset_same_response_id_proven": reporter[
            "responses_cancel_404_after_econnreset_same_response_id"
        ],
        "local_real_ui_diagnostics_clean": local["all_required_clean"],
        "local_installed_issue179_session_settings_parity": local[
            "installed_session_settings_parity"
        ]
        == {
            "installed_proof_count": 4,
            "all_installed_use_bundled_python": True,
            "all_installed_use_responses": True,
            "all_installed_disable_builtin_tools": True,
            "all_installed_server_cache_controls_disabled": True,
            "all_installed_have_cache_flags": True,
            "has_thinking_512_code_clean": True,
            "has_hi_auto_clean_repeat": True,
        },
        "local_installed_bundle_has_responses_cancel_route": (
            installed_bundle["server_has_responses_cancel_route"]
            and installed_bundle["server_cancel_calls_engine_abort"]
        ),
        "public_v1549_tahoe_dmg_has_responses_cancel_route": (
            public_dmg["server_has_responses_cancel_route"]
            and public_dmg["server_cancel_calls_engine_abort"]
        ),
        "latest_public_dmg_has_responses_cancel_route": (
            latest_public_dmg.get("server_has_responses_cancel_route") is True
            and latest_public_dmg.get("server_cancel_calls_engine_abort") is True
        ),
        "local_installed_responses_cancel_live_probe": (
            cancel_probe["status"] == "pass"
            and cancel_probe["response_id_seen"] is True
            and cancel_probe["cancel_status"] == 200
            and (cancel_probe["probe"] or {}).get("cancel_route_present") is True
            and (cancel_probe["probe"] or {}).get("bad_text_captured") is False
        ),
        "current_source_responses_cancel_contract_proven": all(
            (
                source["server_has_responses_cancel_route"],
                source["server_cancel_calls_engine_abort"],
                source["panel_routes_resp_ids_to_responses_cancel"],
                source["test_proves_successful_responses_cancel_route"],
            )
        ),
        "current_source_responses_cancel_inactive_404_contract_proven": source[
            "test_proves_inactive_responses_cancel_404_after_engine_lookup"
        ],
        "local_reporter_prompt_reproduction_clean": local_reporter_prompt_reproduction[
            "clean"
        ],
    }
    not_proven = build_not_proven_items(
        reporter_parity_comparison=reporter_parity_comparison,
        reporter_server_hash_parity=reporter_server_hash_parity,
        reporter_parity_artifact=reporter_parity_artifact,
        reporter=reporter,
        local_reporter_prompt_reproduction=local_reporter_prompt_reproduction,
        proven=proven,
    )
    release_status = "pass" if not not_proven else "open"
    return {
        "status": release_status,
        "issue": {
            "id": 179,
            "title": "MiniMax-M2.7-JANGTQ_K produces gabage",
            "model": "JANGQ-AI/MiniMax-M2.7-JANGTQ_K",
        },
        "reporter": reporter,
        "reporter_screenshot": {
            **screenshot,
        },
        "local_real_ui": local,
        "source_contract": source,
        "local_installed_bundle_contract": installed_bundle,
        "public_release_dmg_contract": public_dmg,
        "public_release_dmg_contracts": public_dmgs,
        "latest_public_release_dmg_contract": latest_public_dmg,
        "bundle_hash_parity": bundle_hash_parity,
        "reporter_server_hash_parity": reporter_server_hash_parity,
        "reporter_parity_artifact": reporter_parity_artifact,
        "reporter_parity_comparison": reporter_parity_comparison,
        "local_responses_cancel_probe": cancel_probe,
        "live_probe_memory_preflight": live_probe_memory_preflight,
        "local_reporter_prompt_reproduction": local_reporter_prompt_reproduction,
        "local_model_manifest": local_model_manifest,
        "root_cause_discriminators": discriminators,
        "language_planning_leak_isolation": language_planning_leak_isolation,
        "proven": proven,
        "not_proven": not_proven,
        "release_boundary": (
            "#179 release-cleared for the current app: the reporter screenshot garbage was "
            "captured from a stale or unknown reporter app hash, while current source, local "
            "installed app, latest public DMG route markers, reporter model hashes, cancel "
            "lifecycle, and local reporter-prompt reproduction are clean."
            if release_status == "pass"
            else "#179 remains open: local dev UI and installed app UI diagnostics are clean, "
            "but the reporter log proves abort-before-visible-content plus Responses cancel 404 "
            "and does not contain the bad text shown in the screenshot. The reporter screenshot "
            "garbage is an interrupted reasoning-panel stream, not proven final visible answer text."
        ),
    }


def write_audit(root: Path, out_path: Path) -> dict[str, Any]:
    out = build_audit(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()

    root = args.root.resolve()
    out = write_audit(root, args.out)
    print(json.dumps({"status": out["status"], "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
