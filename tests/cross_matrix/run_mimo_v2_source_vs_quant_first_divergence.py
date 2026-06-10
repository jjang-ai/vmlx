#!/usr/bin/env python3
"""Live MiMo V2.5 source-vs-quant first-divergence probe.

This script compares already-running source and quantized vMLX-compatible
servers. It does not launch the 113G quant bundle or a source model by itself.

The output artifact is a release-gate proof boundary: it can prove whether the
current MiMo failures are shared by source or introduced by the quant/runtime
path, but it does not itself make MiMo release-cleared.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path(
    "build/current-mimo-v2-jangtq2-source-vs-quant-first-divergence-preflight-20260607.json"
)
DEFAULT_SOURCE_BASE_URL = "http://erics-m5-max2.local:8126"
DEFAULT_QUANT_BASE_URL = "http://127.0.0.1:8897"
DEFAULT_SOURCE_HOST = "erics-m5-max2.local"
DEFAULT_SOURCE_MODEL_PATH = "/Volumes/EricsLLMDrive/jangq-ai/sources/MiMo-V2.5"
DEFAULT_QUANT_MODEL_PATH = "/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2"

PROMPTS: tuple[dict[str, Any], ...] = (
    {
        "name": "plain_exact_blue_cat",
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly blue-cat and no other text.",
            },
        ],
        "expected": "blue-cat",
        "max_tokens": 32,
    },
    {
        "name": "plain_exact_sentinel",
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly B7-CAT-09 and no other text.",
            },
        ],
        "expected": "B7-CAT-09",
        "max_tokens": 32,
    },
    {
        "name": "json_blue_cat",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Return only this JSON exactly: "
                    "{\"status\":\"ok\",\"value\":\"blue-cat\",\"count\":3}"
                ),
            },
        ],
        "expected": "{\"status\":\"ok\",\"value\":\"blue-cat\",\"count\":3}",
        "max_tokens": 64,
    },
    {
        "name": "json_sentinel",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Return only this JSON exactly: "
                    "{\"status\":\"ok\",\"value\":\"B7-CAT-09\",\"count\":3}"
                ),
            },
        ],
        "expected": "{\"status\":\"ok\",\"value\":\"B7-CAT-09\",\"count\":3}",
        "max_tokens": 64,
    },
    {
        "name": "short_system_exact_ack",
        "messages": [
            {"role": "system", "content": "You answer exact probes."},
            {"role": "user", "content": "Reply with exactly ACK."},
        ],
        "expected": "ACK",
        "max_tokens": 16,
    },
    {
        "name": "long_cache_system_exact_ack",
        "messages": [
            {
                "role": "system",
                "content": "You answer exact cache probes. Do not explain.",
            },
            {
                "role": "user",
                "content": (
                    "Cache probe fixed prefix. Read to the stable word list: "
                    "alpha beta gamma delta epsilon zeta eta theta iota kappa. "
                    "Ignore all previous words and reply with exactly ACK."
                ),
            },
        ],
        "expected": "ACK",
        "max_tokens": 16,
    },
    {
        "name": "long_cache_no_system_exact_ack",
        "messages": [
            {
                "role": "user",
                "content": (
                    "You answer exact cache probes. Do not explain.\n"
                    "Cache probe fixed prefix. Read to the stable word list: "
                    "alpha beta gamma delta epsilon zeta eta theta iota kappa. "
                    "Ignore all previous words and reply with exactly ACK."
                ),
            },
        ],
        "expected": "ACK",
        "max_tokens": 16,
    },
    {
        "name": "xml_tool_required_record_fact",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use the record_fact tool exactly once with value blue-cat. "
                    "Do not answer in visible text."
                ),
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "record_fact",
                    "description": "Record a fact.",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                },
            },
        ],
        "tool_choice": "required",
        "expected_tool": "record_fact",
        "expected_arguments": {"value": "blue-cat"},
        "max_tokens": 96,
    },
)


def _post_json(
    url: str,
    body: dict[str, Any],
    *,
    timeout: float,
) -> tuple[int, dict[str, Any], float]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
            return int(resp.status), payload, time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"error": raw}
        return int(exc.code), payload, time.perf_counter() - started
    except urllib.error.URLError as exc:
        return (
            0,
            {"error": f"{type(exc.reason).__name__}: {exc.reason}"},
            time.perf_counter() - started,
        )


def _get_json(url: str, *, timeout: float) -> tuple[int | None, dict[str, Any] | None, str | None]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
            return int(resp.status), payload, None
    except Exception as exc:  # noqa: BLE001 - diagnostic preflight
        return None, None, f"{type(exc).__name__}: {exc}"


def _extract_chat_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _extract_finish_reason(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    finish = first.get("finish_reason")
    return finish if isinstance(finish, str) else None


def _extract_tool_calls(payload: dict[str, Any]) -> list[dict[str, Any]]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    first = choices[0]
    if not isinstance(first, dict):
        return []
    message = first.get("message")
    if not isinstance(message, dict):
        return []
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return []
    return [call for call in calls if isinstance(call, dict)]


def _tool_signature(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signature: list[dict[str, Any]] = []
    for call in tool_calls:
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        signature.append(
            {
                "name": function.get("name"),
                "arguments": function.get("arguments"),
            }
        )
    return signature


def _tool_call_matches(
    tool_calls: list[dict[str, Any]],
    *,
    expected_tool: str,
    expected_arguments: dict[str, Any] | None = None,
) -> bool:
    for call in tool_calls:
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        if function.get("name") != expected_tool:
            continue
        if expected_arguments is None:
            return True
        raw_args = function.get("arguments")
        try:
            parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            return False
        return parsed == expected_arguments
    return False


def first_divergence(a: str, b: str) -> dict[str, Any] | None:
    if a == b:
        return None
    limit = min(len(a), len(b))
    index = 0
    while index < limit and a[index] == b[index]:
        index += 1
    return {
        "char_index": index,
        "source_suffix": a[index : index + 80],
        "quant_suffix": b[index : index + 80],
    }


def classify_row(
    *,
    source_output: str,
    quant_output: str,
    expected: str | None,
    source_status: int,
    quant_status: int,
) -> str:
    if source_status != 200 or quant_status != 200:
        return "request_failed"
    if expected is not None and source_output != expected:
        return "source_also_fails"
    if source_output == quant_output:
        return "source_and_quant_match"
    return "quant_diverges_from_source"


def classify_tool_row(
    *,
    source_tool_calls: list[dict[str, Any]],
    quant_tool_calls: list[dict[str, Any]],
    expected_tool: str,
    expected_arguments: dict[str, Any] | None,
    source_status: int,
    quant_status: int,
) -> str:
    if source_status != 200 or quant_status != 200:
        return "request_failed"
    source_ok = _tool_call_matches(
        source_tool_calls,
        expected_tool=expected_tool,
        expected_arguments=expected_arguments,
    )
    quant_ok = _tool_call_matches(
        quant_tool_calls,
        expected_tool=expected_tool,
        expected_arguments=expected_arguments,
    )
    if not source_ok:
        return "source_also_fails"
    if quant_ok and _tool_signature(source_tool_calls) == _tool_signature(quant_tool_calls):
        return "source_and_quant_match"
    return "quant_diverges_from_source"


def run_probe(
    *,
    source_base_url: str,
    quant_base_url: str,
    source_model: str,
    quant_model: str,
    source_model_path: str,
    quant_model_path: str,
    timeout: float,
    prompts: tuple[dict[str, Any], ...] = PROMPTS,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        max_tokens = int(prompt.get("max_tokens") or 32)
        common = {
            "messages": prompt["messages"],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if isinstance(prompt.get("tools"), list):
            common["tools"] = prompt["tools"]
        if prompt.get("tool_choice") is not None:
            common["tool_choice"] = prompt["tool_choice"]
        source_status, source_payload, source_elapsed = _post_json(
            source_base_url.rstrip("/") + "/v1/chat/completions",
            {"model": source_model, **common},
            timeout=timeout,
        )
        quant_status, quant_payload, quant_elapsed = _post_json(
            quant_base_url.rstrip("/") + "/v1/chat/completions",
            {"model": quant_model, **common},
            timeout=timeout,
        )
        source_output = _extract_chat_content(source_payload)
        quant_output = _extract_chat_content(quant_payload)
        source_tool_calls = _extract_tool_calls(source_payload)
        quant_tool_calls = _extract_tool_calls(quant_payload)
        expected = prompt.get("expected")
        expected_tool = prompt.get("expected_tool")
        expected_arguments = prompt.get("expected_arguments")
        if isinstance(expected_tool, str):
            classification = classify_tool_row(
                source_tool_calls=source_tool_calls,
                quant_tool_calls=quant_tool_calls,
                expected_tool=expected_tool,
                expected_arguments=(
                    expected_arguments if isinstance(expected_arguments, dict) else None
                ),
                source_status=source_status,
                quant_status=quant_status,
            )
            divergence = (
                None
                if _tool_signature(source_tool_calls) == _tool_signature(quant_tool_calls)
                else {
                    "source_tool_calls": _tool_signature(source_tool_calls),
                    "quant_tool_calls": _tool_signature(quant_tool_calls),
                }
            )
        else:
            classification = classify_row(
                source_output=source_output,
                quant_output=quant_output,
                expected=expected if isinstance(expected, str) else None,
                source_status=source_status,
                quant_status=quant_status,
            )
            divergence = first_divergence(source_output, quant_output)
        rows.append(
            {
                "prompt_name": prompt["name"],
                "expected": expected,
                "expected_tool": expected_tool,
                "expected_arguments": expected_arguments,
                "classification": classification,
                "first_divergence": divergence,
                "source_output": source_output,
                "quant_output": quant_output,
                "source_tool_calls": _tool_signature(source_tool_calls),
                "quant_tool_calls": _tool_signature(quant_tool_calls),
                "source_http_status": source_status,
                "quant_http_status": quant_status,
                "source_finish_reason": _extract_finish_reason(source_payload),
                "quant_finish_reason": _extract_finish_reason(quant_payload),
                "source_elapsed_s": round(source_elapsed, 3),
                "quant_elapsed_s": round(quant_elapsed, 3),
                "source_usage": source_payload.get("usage"),
                "quant_usage": quant_payload.get("usage"),
            }
        )

    failed = any(row["classification"] == "request_failed" for row in rows)
    return {
        "status": "fail" if failed else "pass",
        "remote_evidence_only": False,
        "source_model": source_model,
        "quant_model": quant_model,
        "source_model_path": source_model_path,
        "quant_model_path": quant_model_path,
        "source_base_url": source_base_url,
        "quant_base_url": quant_base_url,
        "rows": rows,
        "summary": {
            "source_and_quant_match": sum(
                row["classification"] == "source_and_quant_match" for row in rows
            ),
            "quant_diverges_from_source": sum(
                row["classification"] == "quant_diverges_from_source" for row in rows
            ),
            "source_also_fails": sum(
                row["classification"] == "source_also_fails" for row in rows
            ),
            "request_failed": sum(row["classification"] == "request_failed" for row in rows),
        },
    }


def preflight(
    *,
    source_base_url: str | None,
    quant_base_url: str | None,
    source_model_path: str | None,
    quant_model_path: str | None,
    source_host: str | None = None,
    quant_host: str | None = None,
    timeout: float,
) -> dict[str, Any]:
    endpoints: dict[str, Any] = {}
    for name, base_url in (("source", source_base_url), ("quant", quant_base_url)):
        if not base_url:
            endpoints[name] = {
                "base_url": None,
                "health_http_status": None,
                "healthy": False,
                "error": "missing_base_url",
            }
            continue
        status, payload, error = _get_json(base_url.rstrip("/") + "/health", timeout=timeout)
        endpoints[name] = {
            "base_url": base_url,
            "health_http_status": status,
            "healthy": status == 200,
            "health": payload,
            "error": error,
        }

    paths: dict[str, Any] = {}
    for name, value, host in (
        ("source", source_model_path, source_host),
        ("quant", quant_model_path, quant_host),
    ):
        if not value:
            paths[name] = {
                "path": None,
                "host": host,
                "exists": False,
                "error": "missing_model_path",
            }
            continue
        if host:
            quoted = shlex.quote(value)
            cmd = [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                f"ConnectTimeout={max(1, int(timeout))}",
                host,
                (
                    f"if test -d {quoted}; then echo '{{\"exists\":true,\"is_dir\":true}}'; "
                    f"elif test -e {quoted}; then echo '{{\"exists\":true,\"is_dir\":false}}'; "
                    "else echo '{\"exists\":false,\"is_dir\":false}'; fi"
                ),
            ]
            try:
                raw = subprocess.check_output(
                    cmd,
                    text=True,
                    stderr=subprocess.STDOUT,
                    timeout=timeout + 2,
                )
                remote = json.loads(raw.strip().splitlines()[-1])
                paths[name] = {
                    "path": value,
                    "host": host,
                    "exists": remote.get("exists") is True,
                    "is_dir": remote.get("is_dir") is True,
                    "remote": True,
                }
            except Exception as exc:  # noqa: BLE001 - diagnostic preflight
                paths[name] = {
                    "path": value,
                    "host": host,
                    "exists": False,
                    "is_dir": False,
                    "remote": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            path = Path(value)
            paths[name] = {
                "path": value,
                "host": None,
                "exists": path.exists(),
                "is_dir": path.is_dir() if path.exists() else False,
                "remote": False,
            }

    blockers: list[str] = []
    if not endpoints["source"]["healthy"]:
        blockers.append("missing_or_unhealthy_source_endpoint")
    if not endpoints["quant"]["healthy"]:
        blockers.append("missing_or_unhealthy_quant_endpoint")
    if not paths["source"]["exists"]:
        blockers.append("missing_source_model_path")
    if not paths["quant"]["exists"]:
        blockers.append("missing_quant_model_path")

    return {
        "status": "pass" if not blockers else "missing_prerequisites",
        "remote_evidence_only": False,
        "source_model_path": source_model_path,
        "quant_model_path": quant_model_path,
        "source_base_url": source_base_url,
        "quant_base_url": quant_base_url,
        "source_host": source_host,
        "quant_host": quant_host,
        "blockers": blockers,
        "endpoints": endpoints,
        "paths": paths,
        "rows": [],
        "summary": {
            "source_and_quant_match": 0,
            "quant_diverges_from_source": 0,
            "source_also_fails": 0,
            "request_failed": 0,
        },
        "release_boundary": (
            "This is preflight evidence only. It cannot clear MiMo release "
            "until source and quant endpoints both run and prompt rows execute."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-base-url", default=DEFAULT_SOURCE_BASE_URL)
    parser.add_argument("--quant-base-url", default=DEFAULT_QUANT_BASE_URL)
    parser.add_argument("--source-model", default="mimo-v2-source")
    parser.add_argument("--quant-model", default="mimo-v2-jangtq2")
    parser.add_argument("--source-model-path", default=DEFAULT_SOURCE_MODEL_PATH)
    parser.add_argument("--quant-model-path", default=DEFAULT_QUANT_MODEL_PATH)
    parser.add_argument("--source-host", default=DEFAULT_SOURCE_HOST)
    parser.add_argument("--quant-host")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    if args.preflight_only:
        result = preflight(
            source_base_url=args.source_base_url,
            quant_base_url=args.quant_base_url,
            source_model_path=args.source_model_path,
            quant_model_path=args.quant_model_path,
            source_host=args.source_host,
            quant_host=args.quant_host,
            timeout=args.timeout,
        )
    else:
        missing = [
            name
            for name, value in (
                ("--source-base-url", args.source_base_url),
                ("--quant-base-url", args.quant_base_url),
                ("--source-model-path", args.source_model_path),
                ("--quant-model-path", args.quant_model_path),
            )
            if not value
        ]
        if missing:
            parser.error("missing required arguments unless --preflight-only: " + ", ".join(missing))
        result = run_probe(
            source_base_url=args.source_base_url,
            quant_base_url=args.quant_base_url,
            source_model=args.source_model,
            quant_model=args.quant_model,
            source_model_path=args.source_model_path,
            quant_model_path=args.quant_model_path,
            timeout=args.timeout,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"pass", "missing_prerequisites"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
