#!/usr/bin/env python3
"""Classify current MiMo V2.5 exactness failures without source-vs-quant load.

This runner is intentionally no-heavy. It consumes existing MiMo audit/smoke
artifacts and records what the current evidence excludes. It must not turn
wrong model-emitted literals into a pass and must not replace the missing
source-vs-quant proof.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_AUDIT = Path(
    "build/current-mimo-v2-jang2l-current-audit-after-live-mimo-jangtq2-runtime-proof-20260609.json"
)
DEFAULT_SMOKE = Path(
    "build/current-all-local-model-smoke-mimo-v25-jangtq2-live-refresh-20260608/summary.json"
)
DEFAULT_OUT = Path(
    "build/current-mimo-v2-no-source-exactness-classifier-after-lossless-token-trace-20260609.json"
)
DEFAULT_JANGTQ2_EXACTNESS = Path(
    "build/current-mimo-v25-jangtq2-exactness-isolation-20260608/result.json"
)
DEFAULT_JANG2L_EXACTNESS = Path(
    "build/current-mimo-v25-jang2l-exactness-isolation-20260608/result.json"
)
DEFAULT_JANGTQ2_NO_SWITCHGLU_FASTPATH = Path(
    "build/current-mimo-v25-jangtq2-exactness-no-switchglu-fastpath-diagnostic-20260608/result.json"
)
DEFAULT_JANGTQ2_NO_ROUTER_NO_SWITCHGLU_FASTPATH = Path(
    "build/current-mimo-v25-jangtq2-exactness-no-router-no-switchglu-fastpath-diagnostic-20260608/result.json"
)
DEFAULT_JANGTQ2_TQ_KERNEL_PARITY = Path(
    "build/current-mimo-v25-jangtq2-tq-kernel-parity-no-source-20260608.json"
)
DEFAULT_JANGTQ2_LITERAL_VARIANTS = Path(
    "build/current-mimo-v25-jangtq2-exactness-variant-probe-live-after-lossless-token-trace-20260609/result.json"
)
DEFAULT_JANG2L_LITERAL_VARIANTS = Path(
    "build/current-mimo-v25-jang2l-exactness-variant-probe-live-20260608/result.json"
)
DEFAULT_JANG2L_JSON_SENTINEL_ISOLATION = Path(
    "build/current-all-local-model-smoke-mimo-v25-jang2l-live-refresh-20260608/summary.json"
)
DEFAULT_JANGTQ2_LOGPROB_DIAGNOSTIC = Path(
    "build/current-mimo-v25-jangtq2-logprob-literal-diagnostic-20260608.json"
)
DEFAULT_JANGTQ2_TEMPLATE_DIAGNOSTIC = Path(
    "build/current-mimo-v25-jangtq2-tokenizer-template-literal-diagnostic-20260608.json"
)
DEFAULT_JANGTQ2_HYPHEN_DISTRIBUTION = Path(
    "build/current-mimo-v25-jangtq2-exactness-classification-20260608.json"
)
DEFAULT_JANGTQ2_CURRENT_ALL_LOCAL = Path(
    "build/current-all-local-model-smoke-mimo-v25-jangtq2-live-refresh-20260608/JANGQ_MiMo-V2.5-JANGTQ_2/result.json"
)
DEFAULT_JANGTQ2_CONSERVATIVE_NO_CB_NO_PREFIX = Path(
    "build/current-mimo-v25-jangtq2-conservative-no-cb-no-prefix-exactness-20260608/summary.json"
)

EXACTNESS_REASONS = {
    "expected_tool_argument_missing",
    "json_exact_object_mismatch",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _load(path)


def _probe_rows(artifact: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(artifact, dict):
        return []
    rows = artifact.get("requests")
    if not isinstance(rows, list):
        rows = artifact.get("cases")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    if isinstance(rows, dict):
        out: list[dict[str, Any]] = []
        for label, row in rows.items():
            if isinstance(row, dict):
                item = dict(row)
                item.setdefault("label", str(label))
                out.append(item)
        return out
    return []


def _probe_case_map(artifact: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("label")): row
        for row in _probe_rows(artifact)
        if row.get("label") is not None
    }


def _failures(smoke: dict[str, Any]) -> list[dict[str, Any]]:
    rows = smoke.get("results")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("failures"), list):
            out.extend(f for f in row["failures"] if isinstance(f, dict))
    return out


def _server_log_text(smoke: dict[str, Any]) -> str:
    chunks: list[str] = []
    for row in smoke.get("results", []):
        if not isinstance(row, dict):
            continue
        tail = row.get("server_log_tail")
        if isinstance(tail, list):
            chunks.extend(str(item) for item in tail)
        elif isinstance(tail, str):
            chunks.append(tail)
    return "\n".join(chunks)


def _choice_text(case: dict[str, Any], *, completion: bool = False) -> str:
    if isinstance(case.get("content"), str):
        return str(case.get("content") or "")
    body = case.get("body") if isinstance(case, dict) else None
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0] if isinstance(choices[0], dict) else {}
    if completion:
        return str(choice.get("text") or "")
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    return str(message.get("content") or "")


def _first_tool_arguments(case: dict[str, Any]) -> str:
    parsed = case.get("parsed") if isinstance(case, dict) else None
    if isinstance(parsed, dict):
        return json.dumps(parsed, sort_keys=True)
    body = case.get("body") if isinstance(case, dict) else None
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return ""
    first = tool_calls[0] if isinstance(tool_calls[0], dict) else {}
    function = first.get("function") if isinstance(first.get("function"), dict) else {}
    return str(function.get("arguments") or "")


def _exactness_probe_summary(
    jangtq2: dict[str, Any] | None,
    jang2l: dict[str, Any] | None,
) -> dict[str, Any]:
    expected = "B7-CAT-09"
    tq_cases = _probe_case_map(jangtq2)
    j2_cases = _probe_case_map(jang2l)
    tq_copy_case = tq_cases.get("completion_copy_b7") or tq_cases.get(
        "plain_exact_sentinel", {}
    )
    j2_copy_case = j2_cases.get("completion_copy_b7") or j2_cases.get(
        "plain_exact_sentinel", {}
    )
    tq_tool_case = tq_cases.get("chat_tool_b7") or tq_cases.get(
        "tool_sentinel_json_call", {}
    )
    j2_tool_case = j2_cases.get("chat_tool_b7") or j2_cases.get(
        "tool_sentinel_json_call", {}
    )
    tq_copy = _choice_text(tq_copy_case, completion=True)
    j2_copy = _choice_text(j2_copy_case, completion=True)
    tq_tool_args = _first_tool_arguments(tq_tool_case)
    j2_tool_args = _first_tool_arguments(j2_tool_case)
    j2_tool_text = _choice_text(j2_tool_case)
    return {
        "expected_literal": expected,
        "jangtq2_raw_completion_text": tq_copy,
        "jangtq2_raw_completion_literal_preserved": tq_copy.strip() == expected,
        "jangtq2_tool_arguments": tq_tool_args,
        "jangtq2_tool_literal_preserved": expected in tq_tool_args,
        "jang2l_raw_completion_text": j2_copy,
        "jang2l_raw_completion_literal_preserved": j2_copy.strip() == expected,
        "jang2l_tool_arguments": j2_tool_args,
        "jang2l_tool_content_head": j2_tool_text[:240],
        "jang2l_tool_call_parsed": bool(j2_tool_args),
    }


def _diagnostic_copy_preserved(artifact: dict[str, Any] | None) -> bool | None:
    if not isinstance(artifact, dict):
        return None
    cases = _probe_case_map(artifact)
    if not cases:
        return None
    text = _choice_text(
        cases.get("completion_copy_b7") or cases.get("plain_exact_sentinel", {}),
        completion=True,
    )
    if not text:
        return None
    return text.strip() == "B7-CAT-09"


def _tq_kernel_parity_passed(artifact: dict[str, Any] | None) -> bool | None:
    if not isinstance(artifact, dict):
        return None
    if artifact.get("status") != "pass":
        return False
    reports = artifact.get("reports")
    if not isinstance(reports, list) or not reports:
        return False
    return all(
        isinstance(report, dict) and float(report.get("max_abs_diff", 1.0)) < 2e-2
        for report in reports
    )


def _literal_variant_summary(artifact: dict[str, Any] | None) -> dict[str, Any]:
    requests = _probe_rows(artifact)
    if not requests:
        return {
            "exists": False,
            "plain_literal_copy_pass": None,
            "plain_completion_literal_copy_pass": None,
            "plain_chat_literal_copy_pass": None,
            "structured_literal_pass": None,
            "tool_literal_pass": None,
            "failed_labels": [],
            "failures": [],
        }

    failures = [
        {
            "label": str(request.get("label")),
            "content": _choice_text(
                request, completion=str(request.get("route") or "") == "completions"
            ),
            "parsed": request.get("parsed"),
            "expected": request.get("expected"),
        }
        for request in requests
        if isinstance(request, dict) and request.get("pass") is not True
    ]

    def _labels(prefix: str) -> list[dict[str, Any]]:
        return [
            request
            for request in requests
            if isinstance(request, dict)
            and str(request.get("label", "")).startswith(prefix)
        ]

    def _route_labels(prefix: str, route: str) -> list[dict[str, Any]]:
        return [
            request
            for request in _labels(prefix)
            if str(request.get("route") or "") == route
        ]

    def _all_pass(rows: list[dict[str, Any]]) -> bool | None:
        if not rows:
            return None
        return all(row.get("pass") is True for row in rows)

    return {
        "exists": True,
        "status": artifact.get("status"),
        "plain_literal_copy_pass": _all_pass(_labels("plain_exact")),
        "plain_completion_literal_copy_pass": _all_pass(
            _route_labels("plain_exact", "completions")
        ),
        "plain_chat_literal_copy_pass": _all_pass(_route_labels("plain_exact", "chat")),
        "structured_literal_pass": _all_pass(_labels("json_")),
        "tool_literal_pass": _all_pass(_labels("tool_")),
        "failed_labels": [failure["label"] for failure in failures],
        "failures": failures,
    }


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


def _json_sentinel_summary(artifact: dict[str, Any] | None) -> dict[str, Any]:
    rows = None
    if isinstance(artifact, dict) and isinstance(artifact.get("results"), list):
        for result in artifact["results"]:
            if isinstance(result, dict) and isinstance(result.get("requests"), list):
                rows = result["requests"]
                break
    if isinstance(rows, list):
        requests = [
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("label") or "") in {"mimo_structured_json_sentinel"}
        ]
        normalized_requests = []
        for row in requests:
            normalized = dict(row)
            failures = row.get("validation_failures")
            if isinstance(failures, list):
                for failure in failures:
                    if not isinstance(failure, dict):
                        continue
                    if (
                        failure.get("reason") == "json_exact_object_mismatch"
                        and "expected" not in normalized
                    ):
                        normalized["expected"] = failure.get("expected")
                    if (
                        failure.get("reason") == "json_exact_object_mismatch"
                        and "parsed" not in normalized
                    ):
                        normalized["parsed"] = failure.get("actual")
                    break
            if "parsed" not in normalized and isinstance(normalized.get("content"), str):
                parsed = _parse_json_object(str(normalized.get("content") or ""))
                if parsed is not None:
                    normalized["parsed"] = parsed
            normalized_requests.append(normalized)
        requests = normalized_requests
        artifact = {"status": artifact.get("status"), "requests": requests}

    if isinstance(artifact, dict) and "parsed_content" in artifact:
        content = artifact.get("content")
        parsed = artifact.get("parsed_content")
        expected = artifact.get("expected")
        failed = artifact.get("pass") is not True
        empty = str(content or "").strip() == ""
        schema_mutation = (
            isinstance(parsed, dict)
            and isinstance(expected, dict)
            and set(parsed.keys()) != set(expected.keys())
        )
        semantic_mismatch = (
            isinstance(parsed, dict)
            and isinstance(expected, dict)
            and set(parsed.keys()) == set(expected.keys())
            and parsed != expected
        )
        failure = {
            "label": "current_json_sentinel",
            "content": content,
            "parsed": parsed,
            "expected": expected,
            "completion_tokens": (
                artifact.get("usage", {}).get("completion_tokens")
                if isinstance(artifact.get("usage"), dict)
                else None
            ),
        }
        return {
            "exists": True,
            "status": "pass" if artifact.get("pass") is True else "open",
            "exact_json_pass": artifact.get("pass") is True,
            "empty_output_labels": ["current_json_sentinel"] if failed and empty else [],
            "schema_mutation_labels": (
                ["current_json_sentinel"] if failed and schema_mutation else []
            ),
            "semantic_mismatch_labels": (
                ["current_json_sentinel"] if failed and semantic_mismatch else []
            ),
            "failed_labels": ["current_json_sentinel"] if failed else [],
            "failures": [failure] if failed else [],
        }

    requests = artifact.get("requests") if isinstance(artifact, dict) else None
    if not isinstance(requests, list):
        return {
            "exists": False,
            "exact_json_pass": None,
            "empty_output_labels": [],
            "schema_mutation_labels": [],
            "semantic_mismatch_labels": [],
            "failed_labels": [],
            "failures": [],
        }

    failures = [
        {
            "label": str(request.get("label")),
            "content": request.get("content"),
            "parsed": request.get("parsed"),
            "expected": request.get("expected"),
            "completion_tokens": (
                request.get("usage", {}).get("completion_tokens")
                if isinstance(request.get("usage"), dict)
                else None
            ),
        }
        for request in requests
        if isinstance(request, dict) and request.get("pass") is not True
    ]
    empty_output_labels = [
        failure["label"]
        for failure in failures
        if str(failure.get("content") or "").strip() == ""
    ]
    schema_mutation_labels = [
        failure["label"]
        for failure in failures
        if isinstance(failure.get("parsed"), dict)
        and isinstance(failure.get("expected"), dict)
        and set(failure["parsed"].keys()) != set(failure["expected"].keys())
    ]
    semantic_mismatch_labels = [
        failure["label"]
        for failure in failures
        if isinstance(failure.get("parsed"), dict)
        and isinstance(failure.get("expected"), dict)
        and set(failure["parsed"].keys()) == set(failure["expected"].keys())
        and failure["parsed"] != failure["expected"]
    ]
    return {
        "exists": True,
        "status": artifact.get("status"),
        "exact_json_pass": all(
            isinstance(request, dict) and request.get("pass") is True
            for request in requests
        ),
        "empty_output_labels": empty_output_labels,
        "schema_mutation_labels": schema_mutation_labels,
        "semantic_mismatch_labels": semantic_mismatch_labels,
        "failed_labels": [failure["label"] for failure in failures],
        "failures": failures,
    }


def _case_generated_tokens_and_top1(case: dict[str, Any]) -> tuple[list[str], list[str]]:
    logprobs = case.get("logprobs") if isinstance(case, dict) else None
    if not isinstance(logprobs, dict):
        return [], []
    if case.get("route") == "completions":
        tokens = [str(token) for token in (logprobs.get("tokens") or [])]
        top_entries = logprobs.get("top_logprobs") or []
        top1: list[str] = []
        for entry in top_entries:
            if isinstance(entry, dict) and entry:
                top1.append(str(next(iter(entry.keys()))))
            else:
                top1.append("")
        return tokens, top1

    content = logprobs.get("content")
    if not isinstance(content, list):
        return [], []
    tokens = []
    top1 = []
    for entry in content:
        if not isinstance(entry, dict):
            continue
        tokens.append(str(entry.get("token") or ""))
        top = entry.get("top_logprobs")
        first = top[0] if isinstance(top, list) and top else {}
        top1.append(str(first.get("token") or "") if isinstance(first, dict) else "")
    return tokens, top1


def _logprob_literal_summary(artifact: dict[str, Any] | None) -> dict[str, Any]:
    cases = artifact.get("cases") if isinstance(artifact, dict) else None
    if not isinstance(cases, list):
        return {
            "exists": False,
            "failed_literal_cases": [],
            "generated_tokens_are_top1": None,
            "wrong_literal_outputs_are_top1": None,
        }

    failed_cases = []
    generated_tokens_are_top1 = True
    wrong_literal_outputs_are_top1 = True
    for case in cases:
        if not isinstance(case, dict):
            continue
        output = (
            case.get("text")
            if case.get("route") == "completions"
            else case.get("content")
        )
        expected = case.get("expected")
        tokens, top1 = _case_generated_tokens_and_top1(case)
        token_top1_match = bool(tokens) and len(tokens) == len(top1) and all(
            token == top for token, top in zip(tokens, top1)
        )
        if not token_top1_match:
            generated_tokens_are_top1 = False
        if isinstance(expected, str) and str(output or "").strip() != expected.strip():
            failed_cases.append(
                {
                    "label": case.get("label"),
                    "route": case.get("route"),
                    "expected": expected,
                    "output": output,
                    "tokens": tokens,
                    "top1": top1,
                    "generated_tokens_are_top1": token_top1_match,
                }
            )
            if not token_top1_match:
                wrong_literal_outputs_are_top1 = False

    if not failed_cases:
        wrong_literal_outputs_are_top1 = None
    return {
        "exists": True,
        "failed_literal_cases": failed_cases,
        "generated_tokens_are_top1": generated_tokens_are_top1,
        "wrong_literal_outputs_are_top1": wrong_literal_outputs_are_top1,
    }


def _template_literal_summary(artifact: dict[str, Any] | None) -> dict[str, Any]:
    rows = artifact.get("rows") if isinstance(artifact, dict) else None
    if not isinstance(rows, list):
        return {
            "exists": False,
            "literal_preserved_in_raw_prompt": None,
            "literal_preserved_in_chat_template": None,
            "failed_literals": [],
        }
    failed_literals = []
    raw_ok = True
    rendered_ok = True
    for row in rows:
        if not isinstance(row, dict):
            continue
        literal = row.get("literal")
        if row.get("raw_decode_contains_literal") is not True:
            raw_ok = False
            failed_literals.append({"literal": literal, "surface": "raw_prompt"})
        if (
            row.get("rendered_contains_literal") is not True
            or row.get("rendered_decode_contains_literal") is not True
        ):
            rendered_ok = False
            failed_literals.append({"literal": literal, "surface": "chat_template"})
    return {
        "exists": True,
        "literal_preserved_in_raw_prompt": raw_ok,
        "literal_preserved_in_chat_template": rendered_ok,
        "failed_literals": failed_literals,
    }


def _hyphen_distribution_summary(artifact: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        return {
            "exists": False,
            "classification": None,
            "chat_compact_hyphen_fails": None,
            "completion_compact_hyphen_fails": None,
            "spaced_hyphen_preserved": None,
            "not_caused_by": [],
        }
    requests = artifact.get("requests")
    if not isinstance(requests, dict):
        return {
            "exists": False,
            "classification": artifact.get("classification"),
            "chat_compact_hyphen_fails": None,
            "completion_compact_hyphen_fails": None,
            "spaced_hyphen_preserved": None,
            "not_caused_by": [
                str(item) for item in artifact.get("not_caused_by", [])
            ]
            if isinstance(artifact.get("not_caused_by"), list)
            else [],
        }

    def _text(label: str) -> str:
        row = requests.get(label)
        return str(row.get("text") or "") if isinstance(row, dict) else ""

    not_caused_by = artifact.get("not_caused_by")
    if not isinstance(not_caused_by, list):
        not_caused_by = []
    return {
        "exists": True,
        "classification": artifact.get("classification"),
        "chat_compact_hyphen_fails": _text("chat_exact_blue_cat").strip()
        not in {"", "blue-cat"},
        "completion_compact_hyphen_fails": _text("completion_repeat_blue_cat").strip()
        not in {"", "blue-cat"},
        "spaced_hyphen_preserved": _text("chat_hyphen_spaces").strip()
        == "blue - cat",
        "not_caused_by": [str(item) for item in not_caused_by],
    }


def _validation_failures(row: dict[str, Any]) -> list[dict[str, Any]]:
    failures = row.get("validation_failures")
    if not isinstance(failures, list):
        failures = row.get("validation")
    if not isinstance(failures, list):
        return []
    normalized: list[dict[str, Any]] = []
    for failure in failures:
        if isinstance(failure, dict):
            normalized.append(failure)
        elif isinstance(failure, str):
            normalized.append({"reason": failure})
    return normalized


def _cached_tokens(row: dict[str, Any]) -> int:
    usage = row.get("usage")
    if not isinstance(usage, dict):
        return 0
    prompt_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        return int(prompt_details.get("cached_tokens") or 0)
    return int(usage.get("cached_tokens") or 0)


def _cache_detail(row: dict[str, Any]) -> str:
    usage = row.get("usage")
    if not isinstance(usage, dict):
        return ""
    prompt_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        return str(prompt_details.get("cache_detail") or "")
    return str(usage.get("cache_detail") or "")


def _disk_hits(row: dict[str, Any]) -> int:
    cache_summary = row.get("cache_summary")
    if isinstance(cache_summary, dict):
        return int(cache_summary.get("disk_hits") or 0)
    cache_stats = row.get("cache_stats")
    if isinstance(cache_stats, dict):
        return int(cache_stats.get("disk_hits") or 0)
    return 0


def _current_all_local_summary(artifact: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        return {
            "exists": False,
            "status": None,
            "cache_l2_transport_green": None,
            "media_transport_green": None,
            "literal_mutation_labels": [],
            "context_recall_failure_labels": [],
            "literal_mutation_failures": [],
        }

    request_rows = artifact.get("requests")
    requests = request_rows if isinstance(request_rows, list) else []
    by_label = {
        str(row.get("label")): row
        for row in requests
        if isinstance(row, dict) and row.get("label") is not None
    }
    l2_restart = artifact.get("l2_restart")
    if not isinstance(l2_restart, dict):
        l2_restart = by_label.get("text_cache_repeat_l2_restart", {})
    if not isinstance(l2_restart, dict):
        l2_restart = {}

    first_cache = by_label.get("text_cache_repeat_1", {})
    second_cache = by_label.get("text_cache_repeat_2", {})
    cache_l2_transport_green = (
        isinstance(first_cache, dict)
        and isinstance(second_cache, dict)
        and str(first_cache.get("content") or "").strip() == "ACK"
        and str(second_cache.get("content") or "").strip() == "ACK"
        and str(l2_restart.get("content") or "").strip() == "ACK"
        and not _validation_failures(first_cache)
        and not _validation_failures(second_cache)
        and not _validation_failures(l2_restart)
        and _cached_tokens(second_cache) > 0
        and _cached_tokens(l2_restart) > 0
        and "disk" in _cache_detail(l2_restart)
        and _disk_hits(l2_restart) > 0
    )

    media_labels = [
        label
        for label in by_label
        if label.startswith("vl_")
        or label.startswith("video_")
        or label.startswith("audio_")
    ]
    media_transport_green = bool(media_labels) and all(
        not _validation_failures(by_label[label])
        and str(by_label[label].get("content") or "").strip()
        for label in media_labels
    )

    literal_reasons = EXACTNESS_REASONS
    literal_mutation_failures = []
    context_recall_failure_labels = []
    for row in requests:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label") or "")
        failures = _validation_failures(row)
        if any(failure.get("reason") in literal_reasons for failure in failures):
            literal_mutation_failures.append(
                {
                    "label": label,
                    "content": row.get("content"),
                    "tool_calls": row.get("tool_calls"),
                    "failures": failures,
                }
            )
        elif any(
            failure.get("reason") == "expected_recall_missing" for failure in failures
        ):
            context_recall_failure_labels.append(label)

    return {
        "exists": True,
        "status": artifact.get("status") or artifact.get("result"),
        "cache_l2_transport_green": cache_l2_transport_green,
        "media_transport_green": media_transport_green,
        "literal_mutation_labels": [
            failure["label"] for failure in literal_mutation_failures
        ],
        "context_recall_failure_labels": context_recall_failure_labels,
        "literal_mutation_failures": literal_mutation_failures,
    }


def _conservative_no_cb_no_prefix_summary(
    artifact: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        return {
            "exists": False,
            "status": None,
            "literal_exactness_failed": None,
            "failed_labels": [],
            "continuous_batching_disabled": None,
            "prefix_cache_disabled": None,
            "launch": None,
        }

    failures = artifact.get("failures")
    failures = failures if isinstance(failures, list) else []
    failed_labels = [
        str(failure.get("label"))
        for failure in failures
        if isinstance(failure, dict) and failure.get("label")
    ]
    launch = str(artifact.get("launch") or "")
    return {
        "exists": True,
        "status": artifact.get("status"),
        "literal_exactness_failed": artifact.get("status") == "fail"
        and bool(failed_labels),
        "failed_labels": failed_labels,
        "continuous_batching_disabled": "no-continuous-batching" in launch,
        "prefix_cache_disabled": "disable-prefix-cache" in launch,
        "launch": launch,
    }


def build_classification(
    audit: dict[str, Any],
    smoke: dict[str, Any],
    *,
    jangtq2: dict[str, Any] | None = None,
    jang2l: dict[str, Any] | None = None,
    no_switchglu_fastpath: dict[str, Any] | None = None,
    no_router_no_switchglu_fastpath: dict[str, Any] | None = None,
    tq_kernel_parity: dict[str, Any] | None = None,
    literal_variants: dict[str, Any] | None = None,
    jang2l_literal_variants: dict[str, Any] | None = None,
    jang2l_json_sentinel: dict[str, Any] | None = None,
    jangtq2_logprob_diagnostic: dict[str, Any] | None = None,
    jangtq2_template_diagnostic: dict[str, Any] | None = None,
    jangtq2_hyphen_distribution: dict[str, Any] | None = None,
    jangtq2_current_all_local: dict[str, Any] | None = None,
    jangtq2_conservative_no_cb_no_prefix: dict[str, Any] | None = None,
    artifact_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    component_ok = audit.get("component_ok") if isinstance(audit.get("component_ok"), dict) else {}
    failures = _failures(smoke)
    exactness_failures = [
        failure for failure in failures if failure.get("reason") in EXACTNESS_REASONS
    ]
    log_text = _server_log_text(smoke)

    parser_structure_valid = bool(component_ok.get("tool_protocol")) and all(
        isinstance(failure.get("actual"), dict) for failure in exactness_failures
    )
    cache_kv_l2_excluded = bool(component_ok.get("exactness_cache_kv_quant_excluded"))
    deterministic_sampling_seen = (
        "temperature': 0.0" in log_text
        or '"temperature": 0.0' in log_text
        or "temperature=0.0" in log_text
    ) and (
        "top_p': 1.0" in log_text
        or '"top_p": 1.0' in log_text
        or "top_p=1.0" in log_text
    )

    no_source_exactness = _exactness_probe_summary(jangtq2, jang2l)
    literal_variant_summary = _literal_variant_summary(literal_variants)
    jang2l_literal_variant_summary = _literal_variant_summary(jang2l_literal_variants)
    jang2l_json_sentinel_summary = _json_sentinel_summary(jang2l_json_sentinel)
    jangtq2_logprob_summary = _logprob_literal_summary(jangtq2_logprob_diagnostic)
    jangtq2_template_summary = _template_literal_summary(jangtq2_template_diagnostic)
    jangtq2_hyphen_summary = _hyphen_distribution_summary(jangtq2_hyphen_distribution)
    jangtq2_current_all_local_summary = _current_all_local_summary(
        jangtq2_current_all_local
    )
    conservative_no_cb_no_prefix_summary = _conservative_no_cb_no_prefix_summary(
        jangtq2_conservative_no_cb_no_prefix
    )

    excluded_surfaces = {
        "prompt_template_literal_corruption": (
            jangtq2_template_summary["literal_preserved_in_raw_prompt"] is True
            and jangtq2_template_summary["literal_preserved_in_chat_template"] is True
        ),
        "chat_template_only_primary_cause": (
            "chat_template_only" in jangtq2_hyphen_summary["not_caused_by"]
        ),
        "tokenizer_roundtrip_primary_cause": (
            "tokenizer_roundtrip" in jangtq2_hyphen_summary["not_caused_by"]
        ),
        "parser_argument_rewrite": parser_structure_valid,
        "prefix_paged_l2_or_kv_quant_primary_cause": cache_kv_l2_excluded,
        "hidden_stochastic_sampling_primary_cause": deterministic_sampling_seen,
        "continuous_batching_primary_cause": (
            conservative_no_cb_no_prefix_summary["continuous_batching_disabled"] is True
            and conservative_no_cb_no_prefix_summary["literal_exactness_failed"] is True
        ),
        "prefix_cache_primary_cause": (
            conservative_no_cb_no_prefix_summary["prefix_cache_disabled"] is True
            and conservative_no_cb_no_prefix_summary["literal_exactness_failed"] is True
        ),
        "api_sampler_non_top1_selection": (
            jangtq2_logprob_summary["wrong_literal_outputs_are_top1"] is True
        ),
        "tool_protocol_structure": bool(component_ok.get("tool_protocol")),
        "api_cache_responses_plumbing": bool(component_ok.get("api_cache_responses_contract")),
        "decode_speed_target": bool(component_ok.get("decode_speed_target")),
    }
    no_switchglu_copy_preserved = _diagnostic_copy_preserved(no_switchglu_fastpath)
    no_router_no_switchglu_copy_preserved = _diagnostic_copy_preserved(
        no_router_no_switchglu_fastpath
    )
    tq_kernel_parity_passed = _tq_kernel_parity_passed(tq_kernel_parity)
    no_source_runtime_diagnostics = {
        "jangtq2_no_switchglu_fastpath_literal_preserved": no_switchglu_copy_preserved,
        "jangtq2_no_router_no_switchglu_fastpath_literal_preserved": (
            no_router_no_switchglu_copy_preserved
        ),
        "jangtq2_tq_kernel_parity_passed": tq_kernel_parity_passed,
        "jangtq2_switchglu_fastpath_primary_cause_excluded": (
            no_switchglu_copy_preserved is False
        ),
        "jangtq2_compiled_router_primary_cause_excluded": (
            no_router_no_switchglu_copy_preserved is False
        ),
        "jangtq2_tq_gather_kernel_primary_cause_excluded": (
            tq_kernel_parity_passed is True
        ),
    }

    unresolved_surfaces = {
        "artifact_quantization_or_decode_logits_quality": bool(exactness_failures),
        "jangtq2_raw_decode_or_artifact_quality": (
            no_source_exactness["jangtq2_raw_completion_text"] != ""
            and not no_source_exactness["jangtq2_raw_completion_literal_preserved"]
        ),
        "jang2l_chat_tool_quality": (
            no_source_exactness["jang2l_raw_completion_text"] != ""
            and no_source_exactness["jang2l_raw_completion_literal_preserved"]
            and not no_source_exactness["jang2l_tool_call_parsed"]
            and jang2l_literal_variant_summary["tool_literal_pass"] is not True
        ),
        "source_vs_quant_first_divergence": not bool(
            component_ok.get("source_vs_quant_first_divergence")
        ),
        "long_prompt_coherence": not bool(component_ok.get("long_prompt_coherence")),
        "continuous_batching_system_prompt_pressure": not bool(
            component_ok.get("cb_system_prompt_working_set_pressure")
        ),
        "mimo_media_runtime": not bool(component_ok.get("mimo_media_wired")),
        "jangtq2_switchglu_fastpath": (
            no_source_runtime_diagnostics[
                "jangtq2_switchglu_fastpath_primary_cause_excluded"
            ]
            is not True
        ),
        "jangtq2_compiled_router": (
            no_source_runtime_diagnostics[
                "jangtq2_compiled_router_primary_cause_excluded"
            ]
            is not True
        ),
        "jangtq2_tq_gather_kernel": (
            no_source_runtime_diagnostics[
                "jangtq2_tq_gather_kernel_primary_cause_excluded"
            ]
            is not True
        ),
        "jangtq2_plain_literal_copy": (
            literal_variant_summary["plain_literal_copy_pass"] is False
        ),
        "jang2l_tool_memory_or_protocol": (
            jang2l_literal_variant_summary["exists"] is True
            and jang2l_literal_variant_summary["tool_literal_pass"] is not True
        ),
        "jang2l_json_sentinel_exactness": (
            jang2l_json_sentinel_summary["exists"] is True
            and jang2l_json_sentinel_summary["exact_json_pass"] is not True
        ),
        "jangtq2_compact_hyphen_decode_quality": (
            jangtq2_hyphen_summary["classification"]
            == "compact_hyphen_exactness_fails_in_served_artifact"
            and jangtq2_hyphen_summary["chat_compact_hyphen_fails"] is True
            and jangtq2_hyphen_summary["completion_compact_hyphen_fails"] is True
            and jangtq2_hyphen_summary["spaced_hyphen_preserved"] is True
        ),
        "jangtq2_current_all_local_literal_mutation": (
            jangtq2_current_all_local_summary["exists"] is True
            and bool(jangtq2_current_all_local_summary["literal_mutation_labels"])
        ),
        "jangtq2_conservative_no_cb_no_prefix_literal_mutation": (
            conservative_no_cb_no_prefix_summary["exists"] is True
            and conservative_no_cb_no_prefix_summary["literal_exactness_failed"] is True
        ),
    }

    release_ready = False
    status = "open" if exactness_failures or any(unresolved_surfaces.values()) else "pass"

    if unresolved_surfaces["jangtq2_compact_hyphen_decode_quality"]:
        classification = (
            "jangtq2_compact_hyphen_decode_quality_open_not_cache_parser_template_tokenizer"
        )
    elif (
        literal_variant_summary["plain_literal_copy_pass"] is False
        and jang2l_literal_variant_summary["plain_literal_copy_pass"] is True
    ):
        classification = (
            "jangtq2_plain_literal_copy_regression_jang2l_plain_copy_passes"
        )
    elif (
        literal_variant_summary["plain_literal_copy_pass"] is False
        and literal_variant_summary["structured_literal_pass"] is False
        and literal_variant_summary["tool_literal_pass"] is False
    ):
        classification = (
            "jangtq2_plain_literal_copy_fails_before_parser_or_json_repair"
        )
    elif (
        unresolved_surfaces["jangtq2_raw_decode_or_artifact_quality"]
        and unresolved_surfaces["jang2l_chat_tool_quality"]
        and no_source_runtime_diagnostics[
            "jangtq2_switchglu_fastpath_primary_cause_excluded"
        ]
        and no_source_runtime_diagnostics[
            "jangtq2_compiled_router_primary_cause_excluded"
        ]
        and no_source_runtime_diagnostics[
            "jangtq2_tq_gather_kernel_primary_cause_excluded"
        ]
    ):
        classification = (
            "jangtq2_literal_corruption_persists_without_cache_fastpath_router_"
            "and_tq_kernel_parity_passes"
        )
    elif (
        unresolved_surfaces["jangtq2_raw_decode_or_artifact_quality"]
        and unresolved_surfaces["jang2l_chat_tool_quality"]
    ):
        classification = "jangtq2_raw_decode_literal_corruption_jang2l_chat_tool_quality_open"
    elif exactness_failures and parser_structure_valid:
        classification = "model_generated_literal_mutation_after_valid_parser_structure"
    else:
        classification = "insufficient_evidence"

    secondary_classification = None
    if unresolved_surfaces["jang2l_json_sentinel_exactness"]:
        if jang2l_json_sentinel_summary["empty_output_labels"]:
            secondary_classification = "jang2l_json_sentinel_empty_output_open"
        elif jang2l_json_sentinel_summary["schema_mutation_labels"]:
            secondary_classification = "jang2l_json_sentinel_schema_mutation_open"
        elif jang2l_json_sentinel_summary["semantic_mismatch_labels"]:
            secondary_classification = "jang2l_json_sentinel_semantic_mismatch_open"
        else:
            secondary_classification = "jang2l_json_sentinel_exactness_open"

    artifact_only_logit_diagnosis_present = (
        jangtq2_logprob_summary["wrong_literal_outputs_are_top1"] is True
    )
    required_next_evidence = [
        "do not repair semantic values in parser or JSON repair layer",
    ]
    if artifact_only_logit_diagnosis_present:
        required_next_evidence.append(
            "artifact-only logprob diagnostic already shows current JANGTQ_2 wrong literal outputs are greedy top-1; treat as artifact/logit quality unless a runtime decode bug is proven"
        )
    else:
        required_next_evidence.append(
            "either rerun source-vs-quant when RAM is authorized or provide a stronger artifact-only logits/quantization diagnosis"
        )
    required_next_evidence.extend([
        "conservative live JANGTQ_2 proof with continuous batching off and prefix cache disabled still fails literal exactness; do not keep chasing cache/CB as the primary exactness cause without contrary logits evidence",
        "if artifact/quantization is confirmed, rebuild or reupload MiMo with a corrected quantization contract",
        "if runtime decode is confirmed, fix the decode/kernel path and rerun the exact tool/JSON sentinel rows",
    ])
    if unresolved_surfaces["mimo_media_runtime"]:
        required_next_evidence.append(
            "MiMo media remains unwired until real VL/audio/video runtime and cache proof exist"
        )
    else:
        required_next_evidence.append(
            "MiMo media runtime wiring is no longer the exactness blocker; keep live media E2E, audio depth, and cache/UI proof as separate release gates"
        )
    if unresolved_surfaces["jang2l_json_sentinel_exactness"]:
        required_next_evidence.append(
            "JANG_2L remains blocked on JSON sentinel exactness: exact semantic values must match after runtime thinking-off forwarding; do not repair semantic values or prompt-fold a fake pass"
        )

    model_upload_action_reasons: list[str] = []
    if artifact_only_logit_diagnosis_present:
        model_upload_action_reasons.append(
            "JANGTQ_2 served artifact emits wrong compact-hyphen/sentinel literals as greedy top-1; rebuild/reupload with corrected quantization unless a runtime decode bug is proven"
        )
    if unresolved_surfaces["jang2l_json_sentinel_exactness"]:
        model_upload_action_reasons.append(
            "JANG_2L served artifact/runtime still fails JSON sentinel exactness; provide corrected artifact or runtime decode proof that preserves value and count"
        )

    paths = {
        "audit_artifact": str(DEFAULT_AUDIT),
        "smoke_artifact": str(DEFAULT_SMOKE),
        "jangtq2_exactness_artifact": str(DEFAULT_JANGTQ2_EXACTNESS),
        "jang2l_exactness_artifact": str(DEFAULT_JANG2L_EXACTNESS),
        "jangtq2_no_switchglu_fastpath_artifact": str(
            DEFAULT_JANGTQ2_NO_SWITCHGLU_FASTPATH
        ),
        "jangtq2_no_router_no_switchglu_fastpath_artifact": str(
            DEFAULT_JANGTQ2_NO_ROUTER_NO_SWITCHGLU_FASTPATH
        ),
        "jangtq2_tq_kernel_parity_artifact": str(DEFAULT_JANGTQ2_TQ_KERNEL_PARITY),
        "jangtq2_literal_variant_artifact": str(DEFAULT_JANGTQ2_LITERAL_VARIANTS),
        "jang2l_literal_variant_artifact": str(DEFAULT_JANG2L_LITERAL_VARIANTS),
        "jang2l_json_sentinel_artifact": str(DEFAULT_JANG2L_JSON_SENTINEL_ISOLATION),
        "jangtq2_logprob_diagnostic_artifact": str(DEFAULT_JANGTQ2_LOGPROB_DIAGNOSTIC),
        "jangtq2_template_diagnostic_artifact": str(DEFAULT_JANGTQ2_TEMPLATE_DIAGNOSTIC),
        "jangtq2_current_all_local_artifact": str(DEFAULT_JANGTQ2_CURRENT_ALL_LOCAL),
        "jangtq2_conservative_no_cb_no_prefix_artifact": str(
            DEFAULT_JANGTQ2_CONSERVATIVE_NO_CB_NO_PREFIX
        ),
    }
    if artifact_paths:
        paths.update(artifact_paths)

    return {
        "status": status,
        "release_ready": release_ready,
        "classification": classification,
        "secondary_classification": secondary_classification,
        "model_upload_action_required": bool(model_upload_action_reasons),
        "model_upload_action_reasons": model_upload_action_reasons,
        "source_vs_quant_load_performed": False,
        "source_vs_quant_load_skipped_reason": "user_disallowed_source_vs_quant_due_ram",
        **paths,
        "no_source_exactness": no_source_exactness,
        "literal_variant_summary": literal_variant_summary,
        "jang2l_literal_variant_summary": jang2l_literal_variant_summary,
        "jang2l_json_sentinel_summary": jang2l_json_sentinel_summary,
        "jangtq2_logprob_summary": jangtq2_logprob_summary,
        "jangtq2_template_summary": jangtq2_template_summary,
        "jangtq2_hyphen_distribution_summary": jangtq2_hyphen_summary,
        "jangtq2_current_all_local_summary": jangtq2_current_all_local_summary,
        "jangtq2_conservative_no_cb_no_prefix_summary": (
            conservative_no_cb_no_prefix_summary
        ),
        "no_source_runtime_diagnostics": no_source_runtime_diagnostics,
        "exactness_failures": exactness_failures,
        "excluded_surfaces": excluded_surfaces,
        "unresolved_surfaces": unresolved_surfaces,
        "required_next_evidence": required_next_evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--smoke", type=Path, default=DEFAULT_SMOKE)
    parser.add_argument("--jangtq2-exactness", type=Path, default=DEFAULT_JANGTQ2_EXACTNESS)
    parser.add_argument("--jang2l-exactness", type=Path, default=DEFAULT_JANG2L_EXACTNESS)
    parser.add_argument(
        "--jangtq2-no-switchglu-fastpath",
        type=Path,
        default=DEFAULT_JANGTQ2_NO_SWITCHGLU_FASTPATH,
    )
    parser.add_argument(
        "--jangtq2-no-router-no-switchglu-fastpath",
        type=Path,
        default=DEFAULT_JANGTQ2_NO_ROUTER_NO_SWITCHGLU_FASTPATH,
    )
    parser.add_argument(
        "--jangtq2-tq-kernel-parity",
        type=Path,
        default=DEFAULT_JANGTQ2_TQ_KERNEL_PARITY,
    )
    parser.add_argument(
        "--jangtq2-literal-variants",
        type=Path,
        default=DEFAULT_JANGTQ2_LITERAL_VARIANTS,
    )
    parser.add_argument(
        "--jang2l-literal-variants",
        type=Path,
        default=DEFAULT_JANG2L_LITERAL_VARIANTS,
    )
    parser.add_argument(
        "--jang2l-json-sentinel",
        type=Path,
        default=DEFAULT_JANG2L_JSON_SENTINEL_ISOLATION,
    )
    parser.add_argument(
        "--jangtq2-logprob-diagnostic",
        type=Path,
        default=DEFAULT_JANGTQ2_LOGPROB_DIAGNOSTIC,
    )
    parser.add_argument(
        "--jangtq2-template-diagnostic",
        type=Path,
        default=DEFAULT_JANGTQ2_TEMPLATE_DIAGNOSTIC,
    )
    parser.add_argument(
        "--jangtq2-hyphen-distribution",
        type=Path,
        default=DEFAULT_JANGTQ2_HYPHEN_DISTRIBUTION,
    )
    parser.add_argument(
        "--jangtq2-current-all-local",
        type=Path,
        default=DEFAULT_JANGTQ2_CURRENT_ALL_LOCAL,
    )
    parser.add_argument(
        "--jangtq2-conservative-no-cb-no-prefix",
        type=Path,
        default=DEFAULT_JANGTQ2_CONSERVATIVE_NO_CB_NO_PREFIX,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    jangtq2 = _load(args.jangtq2_exactness) if args.jangtq2_exactness.exists() else None
    jang2l = _load(args.jang2l_exactness) if args.jang2l_exactness.exists() else None
    no_switchglu_fastpath = (
        _load(args.jangtq2_no_switchglu_fastpath)
        if args.jangtq2_no_switchglu_fastpath.exists()
        else None
    )
    no_router_no_switchglu_fastpath = (
        _load(args.jangtq2_no_router_no_switchglu_fastpath)
        if args.jangtq2_no_router_no_switchglu_fastpath.exists()
        else None
    )
    tq_kernel_parity = (
        _load(args.jangtq2_tq_kernel_parity)
        if args.jangtq2_tq_kernel_parity.exists()
        else None
    )
    literal_variants = (
        _load(args.jangtq2_literal_variants)
        if args.jangtq2_literal_variants.exists()
        else None
    )
    jang2l_literal_variants = (
        _load(args.jang2l_literal_variants)
        if args.jang2l_literal_variants.exists()
        else None
    )
    jang2l_json_sentinel = (
        _load(args.jang2l_json_sentinel)
        if args.jang2l_json_sentinel.exists()
        else None
    )
    jangtq2_logprob_diagnostic = (
        _load(args.jangtq2_logprob_diagnostic)
        if args.jangtq2_logprob_diagnostic.exists()
        else None
    )
    jangtq2_template_diagnostic = (
        _load(args.jangtq2_template_diagnostic)
        if args.jangtq2_template_diagnostic.exists()
        else None
    )
    jangtq2_hyphen_distribution = (
        _load(args.jangtq2_hyphen_distribution)
        if args.jangtq2_hyphen_distribution.exists()
        else None
    )
    jangtq2_current_all_local = (
        _load(args.jangtq2_current_all_local)
        if args.jangtq2_current_all_local.exists()
        else None
    )
    jangtq2_conservative_no_cb_no_prefix = (
        _load(args.jangtq2_conservative_no_cb_no_prefix)
        if args.jangtq2_conservative_no_cb_no_prefix.exists()
        else None
    )
    artifact = build_classification(
        _load(args.audit),
        _load_optional(args.smoke),
        jangtq2=jangtq2,
        jang2l=jang2l,
        no_switchglu_fastpath=no_switchglu_fastpath,
        no_router_no_switchglu_fastpath=no_router_no_switchglu_fastpath,
        tq_kernel_parity=tq_kernel_parity,
        literal_variants=literal_variants,
        jang2l_literal_variants=jang2l_literal_variants,
        jang2l_json_sentinel=jang2l_json_sentinel,
        jangtq2_logprob_diagnostic=jangtq2_logprob_diagnostic,
        jangtq2_template_diagnostic=jangtq2_template_diagnostic,
        jangtq2_hyphen_distribution=jangtq2_hyphen_distribution,
        jangtq2_current_all_local=jangtq2_current_all_local,
        jangtq2_conservative_no_cb_no_prefix=(
            jangtq2_conservative_no_cb_no_prefix
        ),
        artifact_paths={
            "audit_artifact": str(args.audit),
            "smoke_artifact": str(args.smoke),
            "jangtq2_exactness_artifact": str(args.jangtq2_exactness),
            "jang2l_exactness_artifact": str(args.jang2l_exactness),
            "jangtq2_no_switchglu_fastpath_artifact": str(
                args.jangtq2_no_switchglu_fastpath
            ),
            "jangtq2_no_router_no_switchglu_fastpath_artifact": str(
                args.jangtq2_no_router_no_switchglu_fastpath
            ),
            "jangtq2_tq_kernel_parity_artifact": str(args.jangtq2_tq_kernel_parity),
            "jangtq2_literal_variant_artifact": str(args.jangtq2_literal_variants),
            "jang2l_literal_variant_artifact": str(args.jang2l_literal_variants),
            "jang2l_json_sentinel_artifact": str(args.jang2l_json_sentinel),
            "jangtq2_logprob_diagnostic_artifact": str(
                args.jangtq2_logprob_diagnostic
            ),
            "jangtq2_template_diagnostic_artifact": str(
                args.jangtq2_template_diagnostic
            ),
            "jangtq2_current_all_local_artifact": str(args.jangtq2_current_all_local),
            "jangtq2_conservative_no_cb_no_prefix_artifact": str(
                args.jangtq2_conservative_no_cb_no_prefix
            ),
        },
    )
    artifact["jangtq2_hyphen_distribution_artifact"] = str(
        args.jangtq2_hyphen_distribution
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)
    print(f"status={artifact['status']}")
    print(f"classification={artifact['classification']}")
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
