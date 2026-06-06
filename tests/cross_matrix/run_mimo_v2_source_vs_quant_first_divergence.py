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
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path(
    "build/current-mimo-v2-jang2l-source-vs-quant-first-divergence-20260606.json"
)

PROMPTS: tuple[dict[str, Any], ...] = (
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
        expected = prompt.get("expected")
        rows.append(
            {
                "prompt_name": prompt["name"],
                "expected": expected,
                "classification": classify_row(
                    source_output=source_output,
                    quant_output=quant_output,
                    expected=expected if isinstance(expected, str) else None,
                    source_status=source_status,
                    quant_status=quant_status,
                ),
                "first_divergence": first_divergence(source_output, quant_output),
                "source_output": source_output,
                "quant_output": quant_output,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-base-url", required=True)
    parser.add_argument("--quant-base-url", required=True)
    parser.add_argument("--source-model", default="mimo-v2-source")
    parser.add_argument("--quant-model", default="mimo-v2-jang2l")
    parser.add_argument("--source-model-path", required=True)
    parser.add_argument("--quant-model-path", required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

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
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
