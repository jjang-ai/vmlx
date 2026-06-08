#!/usr/bin/env python3
"""Live MiMo V2.5 exactness variant probe.

This runner targets the current no-source MiMo release blocker. It does not
load the source model and it does not repair semantic values. It sends a small
set of exact literal, JSON, and required-tool probes to a running vMLX server
and writes the ``requests`` shape consumed by
``run_mimo_v2_no_source_exactness_classifier.py``.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("build/current-mimo-v25-exactness-variant-probe/result.json")
DEFAULT_EXPECTED_BLUE = "blue-cat"
DEFAULT_EXPECTED_SENTINEL = "B7-CAT-09"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "record_fact",
            "description": "Record one exact fact value.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        },
    }
]


def _json_request(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
            return {
                "code": response.status,
                "elapsed_s": round(time.monotonic() - started, 3),
                "raw": raw,
                "body": json.loads(raw) if raw else None,
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            body = json.loads(raw) if raw else None
        except Exception:
            body = None
        return {
            "code": exc.code,
            "elapsed_s": round(time.monotonic() - started, 3),
            "raw": raw,
            "body": body,
            "error": repr(exc),
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic artifact records type
        return {
            "code": None,
            "elapsed_s": round(time.monotonic() - started, 3),
            "raw": "",
            "body": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _message_content(body: dict[str, Any] | None) -> str:
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    return str(message.get("content") or "")


def _completion_text(body: dict[str, Any] | None) -> str:
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0] if isinstance(choices[0], dict) else {}
    return str(choice.get("text") or "")


def _tool_arguments(body: dict[str, Any] | None) -> dict[str, Any] | None:
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    first = tool_calls[0] if isinstance(tool_calls[0], dict) else {}
    function = first.get("function") if isinstance(first.get("function"), dict) else {}
    raw_arguments = function.get("arguments")
    if not isinstance(raw_arguments, str):
        return None
    try:
        parsed = json.loads(raw_arguments)
    except Exception:
        return {"_raw_arguments": raw_arguments}
    return parsed if isinstance(parsed, dict) else {"_parsed_arguments": parsed}


def _parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        parsed = json.loads(stripped)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def build_cases(
    *,
    model: str,
    expected_blue: str = DEFAULT_EXPECTED_BLUE,
    expected_sentinel: str = DEFAULT_EXPECTED_SENTINEL,
    max_tokens: int = 48,
) -> list[dict[str, Any]]:
    return [
        {
            "label": "plain_exact_blue_cat",
            "route": "completions",
            "expected": expected_blue,
            "payload": {
                "model": model,
                "prompt": (
                    f"Output exactly this string and nothing else:\n{expected_blue}\n"
                ),
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 16,
            },
        },
        {
            "label": "plain_exact_sentinel",
            "route": "completions",
            "expected": expected_sentinel,
            "payload": {
                "model": model,
                "prompt": (
                    f"Output exactly this string and nothing else:\n{expected_sentinel}\n"
                ),
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 20,
            },
        },
        {
            "label": "json_blue_cat",
            "route": "chat",
            "expected": {"status": "ok", "value": expected_blue, "count": 3},
            "payload": {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Return exactly this JSON object, no markdown, no prose: "
                            '{"status":"ok","value":"blue-cat","count":3}'
                        ),
                    }
                ],
                "temperature": 0,
                "top_p": 1,
                "max_tokens": max_tokens,
                "enable_thinking": False,
            },
        },
        {
            "label": "json_sentinel",
            "route": "chat",
            "expected": {"status": "ok", "value": expected_sentinel, "count": 3},
            "payload": {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Return exactly this JSON object, no markdown, no prose: "
                            '{"status":"ok","value":"B7-CAT-09","count":3}'
                        ),
                    }
                ],
                "temperature": 0,
                "top_p": 1,
                "max_tokens": max_tokens,
                "enable_thinking": False,
            },
        },
        {
            "label": "tool_blue_cat",
            "route": "chat",
            "expected": {"value": expected_blue},
            "payload": {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Call record_fact exactly once with value "
                            '"blue-cat". Do not use visible prose.'
                        ),
                    }
                ],
                "tools": TOOLS,
                "tool_choice": "required",
                "temperature": 0,
                "top_p": 1,
                "max_tokens": max_tokens,
                "enable_thinking": False,
            },
        },
        {
            "label": "tool_sentinel_json_call",
            "route": "chat",
            "expected": {"value": expected_sentinel},
            "payload": {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Call record_fact exactly once with value "
                            '"B7-CAT-09". Preserve both hyphens exactly. '
                            "Do not use visible prose."
                        ),
                    }
                ],
                "tools": TOOLS,
                "tool_choice": "required",
                "temperature": 0,
                "top_p": 1,
                "max_tokens": max_tokens,
                "enable_thinking": False,
            },
        },
    ]


def _finish_reason(body: dict[str, Any] | None) -> str | None:
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0] if isinstance(choices[0], dict) else {}
    finish = choice.get("finish_reason")
    return str(finish) if finish is not None else None


def classify_case(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    body = response.get("body") if isinstance(response, dict) else None
    expected = case["expected"]
    label = str(case["label"])
    content = ""
    parsed: Any = None

    if label.startswith("plain_exact"):
        content = _completion_text(body)
        passed = response.get("code") == 200 and content.strip() == expected
    elif label.startswith("json_"):
        content = _message_content(body)
        parsed = _parse_json_object(content)
        passed = response.get("code") == 200 and parsed == expected
    elif label.startswith("tool_"):
        content = _message_content(body)
        parsed = _tool_arguments(body)
        passed = response.get("code") == 200 and parsed == expected and content == ""
    else:
        passed = False

    return {
        "label": label,
        "route": case["route"],
        "code": response.get("code"),
        "elapsed_s": response.get("elapsed_s"),
        "pass": passed,
        "content": content,
        "parsed": parsed,
        "expected": expected,
        "usage": body.get("usage") if isinstance(body, dict) else None,
        "finish_reason": _finish_reason(body),
        "error": response.get("error"),
        "raw": response.get("raw"),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.base_url.rstrip("/")
    rows: list[dict[str, Any]] = []
    for case in build_cases(model=args.model, max_tokens=args.max_tokens):
        endpoint = (
            "/v1/completions"
            if case["route"] == "completions"
            else "/v1/chat/completions"
        )
        response = _json_request(
            f"{base_url}{endpoint}",
            case["payload"],
            timeout=args.timeout,
        )
        rows.append(classify_case(case, response))
    failed = [row for row in rows if row.get("pass") is not True]
    return {
        "schema": "vmlx-mimo-v2-exactness-variant-probe-v1",
        "status": "pass" if not failed else "open",
        "base_url": base_url,
        "model": args.model,
        "requests": rows,
        "failed_labels": [str(row.get("label")) for row in failed],
        "notes": [
            "no source-vs-quant load",
            "no parser-side semantic repair",
            "no JSON repair scoring",
            "expected values must be model-emitted exactly",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--max-tokens", type=int, default=48)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact = run_probe(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(args.out)
    print(f"status={artifact['status']} failed_labels={artifact['failed_labels']}")
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
