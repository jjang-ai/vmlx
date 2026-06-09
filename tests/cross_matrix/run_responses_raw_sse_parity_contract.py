#!/usr/bin/env python3
"""Classify raw Responses SSE tool-argument parity across route surfaces.

This intentionally does not launch a model or tunnel. It consumes captured raw
SSE text from direct local server, panel gateway, and optional tunnel paths so
the release board can distinguish missing evidence from argument corruption.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("build/current-responses-raw-sse-parity-20260609.json")


def _event_payloads(sse_text: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in sse_text.replace("\r\n", "\n").split("\n\n"):
        event_type = "message"
        data_lines: list[str] = []
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if line.startswith("event:"):
                event_type = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())
        if not data_lines:
            continue
        raw_data = "\n".join(data_lines)
        if raw_data == "[DONE]":
            continue
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError:
            events.append((event_type, {"_raw": raw_data, "_parse_error": True}))
            continue
        if isinstance(payload, dict):
            events.append((event_type, payload))
    return events


def classify_sse_capture(sse_text: str) -> dict[str, Any]:
    events = _event_payloads(sse_text)
    arg_deltas: list[str] = []
    arg_done: list[str] = []
    function_items: list[dict[str, Any]] = []
    item_output_indexes: dict[str, dict[str, Any]] = {}
    models: list[str] = []
    reasoning_events = 0
    heartbeats = 0
    parse_errors = 0

    for event_type, payload in events:
        payload_type = str(payload.get("type") or event_type)
        if payload.get("_parse_error"):
            parse_errors += 1
        if payload_type in {
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
        }:
            reasoning_events += 1
        if payload_type == "response.heartbeat" or payload.get("tool_call_generating"):
            heartbeats += 1
        if payload_type in {"response.created", "response.completed"}:
            response = payload.get("response")
            if isinstance(response, dict) and response.get("model"):
                model_name = str(response.get("model"))
                if model_name not in models:
                    models.append(model_name)
        if payload_type == "response.function_call_arguments.delta":
            arg_deltas.append(str(payload.get("delta", "")))
        elif payload_type == "response.function_call_arguments.done":
            arg_done.append(str(payload.get("arguments", "")))
        elif payload_type in {"response.output_item.added", "response.output_item.done"}:
            item = payload.get("item")
            if isinstance(item, dict):
                item_type = str(item.get("type") or "")
                item_id = str(item.get("id") or f"{item_type}:{len(item_output_indexes)}")
                item_output_indexes[item_id] = {
                    "type": item_type,
                    "output_index": payload.get("output_index"),
                }
                if item_type == "function_call":
                    function_items.append(item)

    reconstructed = "".join(arg_deltas)
    final_done = arg_done[-1] if arg_done else ""
    final_item_args = ""
    final_item_name = ""
    if function_items:
        final_item = function_items[-1]
        final_item_args = str(final_item.get("arguments", ""))
        final_item_name = str(final_item.get("name", ""))

    # The done event is authoritative for Responses streaming. Some runtimes
    # emit an empty final item.arguments while preserving done/delta args.
    authoritative_args = final_done or reconstructed or final_item_args
    empty_final_item_with_done_args = bool(final_done and final_item_args == "")
    output_indices_by_type: dict[str, list[int]] = {}
    for item in item_output_indexes.values():
        item_type = str(item.get("type") or "")
        output_index = item.get("output_index")
        if item_type and isinstance(output_index, int):
            output_indices_by_type.setdefault(item_type, [])
            if output_index not in output_indices_by_type[item_type]:
                output_indices_by_type[item_type].append(output_index)
    message_like_indices = set(output_indices_by_type.get("message", [])) | set(
        output_indices_by_type.get("reasoning", [])
    )
    function_indices = set(output_indices_by_type.get("function_call", []))
    conflicting_output_indices = sorted(message_like_indices & function_indices)

    return {
        "event_count": len(events),
        "parse_errors": parse_errors,
        "reasoning_events": reasoning_events,
        "heartbeat_count": heartbeats,
        "argument_delta_count": len(arg_deltas),
        "argument_done_count": len(arg_done),
        "function_item_count": len(function_items),
        "function_name": final_item_name,
        "reconstructed_arguments": reconstructed,
        "done_arguments": final_done,
        "final_item_arguments": final_item_args,
        "authoritative_arguments": authoritative_args,
        "has_authoritative_arguments": bool(authoritative_args),
        "empty_final_item_with_done_args": empty_final_item_with_done_args,
        "output_indices_by_type": output_indices_by_type,
        "conflicting_output_indices": conflicting_output_indices,
        "output_item_indices_valid": not conflicting_output_indices,
        "models": models,
        "model": models[-1] if models else "",
    }


def build_artifact(
    *,
    direct_sse: Path | None,
    gateway_sse: Path | None,
    tunnel_sse: Path | None,
    expected_function_name: str | None = None,
    expected_arguments: str | None = None,
    expected_model: str | None = None,
    require_reasoning_events: bool = False,
    require_same_model: bool = False,
) -> dict[str, Any]:
    captures: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for name, path in (
        ("direct", direct_sse),
        ("gateway", gateway_sse),
        ("tunnel", tunnel_sse),
    ):
        if path is None or not path.exists():
            captures[name] = {"present": False}
            missing.append(name)
            continue
        classified = classify_sse_capture(path.read_text())
        classified["expected_function_name_match"] = (
            True
            if expected_function_name is None
            else classified.get("function_name") == expected_function_name
        )
        classified["expected_arguments_match"] = (
            True
            if expected_arguments is None
            else classified.get("authoritative_arguments") == expected_arguments
        )
        classified["expected_model_match"] = (
            True
            if expected_model is None
            else classified.get("model") == expected_model
        )
        classified["has_required_reasoning_events"] = (
            True
            if not require_reasoning_events
            else int(classified.get("reasoning_events") or 0) > 0
        )
        captures[name] = {
            "present": True,
            "path": str(path),
            **classified,
        }

    comparable = [name for name, row in captures.items() if row.get("present")]
    args_by_surface = {
        name: row.get("authoritative_arguments", "")
        for name, row in captures.items()
        if row.get("present")
    }
    present_with_args = {
        name: bool(row.get("has_authoritative_arguments"))
        for name, row in captures.items()
        if row.get("present")
    }
    present_without_parse_errors = {
        name: int(row.get("parse_errors") or 0) == 0
        for name, row in captures.items()
        if row.get("present")
    }
    present_with_valid_output_indices = {
        name: bool(row.get("output_item_indices_valid"))
        for name, row in captures.items()
        if row.get("present")
    }
    expected_function_matches = {
        name: bool(row.get("expected_function_name_match"))
        for name, row in captures.items()
        if row.get("present")
    }
    expected_arguments_matches = {
        name: bool(row.get("expected_arguments_match"))
        for name, row in captures.items()
        if row.get("present")
    }
    expected_model_matches = {
        name: bool(row.get("expected_model_match"))
        for name, row in captures.items()
        if row.get("present")
    }
    required_reasoning_present = {
        name: bool(row.get("has_required_reasoning_events"))
        for name, row in captures.items()
        if row.get("present")
    }
    mismatch = len(set(args_by_surface.values())) > 1 if args_by_surface else False
    all_required_present = not missing
    all_present_have_args = bool(comparable) and all(present_with_args.values())
    all_present_parse_clean = bool(comparable) and all(
        present_without_parse_errors.values()
    )
    all_present_output_indices_valid = bool(comparable) and all(
        present_with_valid_output_indices.values()
    )
    all_expected_functions_match = bool(comparable) and all(
        expected_function_matches.values()
    )
    all_expected_arguments_match = bool(comparable) and all(
        expected_arguments_matches.values()
    )
    all_expected_models_match = bool(comparable) and all(
        expected_model_matches.values()
    )
    all_required_reasoning_present = bool(comparable) and all(
        required_reasoning_present.values()
    )
    models_by_surface = {
        name: str(row.get("model") or "")
        for name, row in captures.items()
        if row.get("present")
    }
    known_models = {model for model in models_by_surface.values() if model}
    all_present_surfaces_report_model = bool(comparable) and all(
        models_by_surface.values()
    )
    all_present_surfaces_same_model = (
        True
        if not require_same_model
        else all_present_surfaces_report_model and len(known_models) == 1
    )

    if (
        mismatch
        or (
            comparable
            and not (
                all_present_have_args
                and all_present_parse_clean
                and all_present_output_indices_valid
                and all_expected_functions_match
                and all_expected_arguments_match
                and all_expected_models_match
                and all_required_reasoning_present
                and all_present_surfaces_same_model
            )
        )
    ):
        status = "fail"
    elif all_required_present and all_present_have_args:
        status = "pass"
    else:
        status = "open"

    return {
        "status": status,
        "missing_captures": missing,
        "captures": captures,
        "checks": {
            "direct_capture_present": captures["direct"].get("present") is True,
            "gateway_capture_present": captures["gateway"].get("present") is True,
            "tunnel_capture_present": captures["tunnel"].get("present") is True,
            "all_present_surfaces_have_authoritative_args": all_present_have_args,
            "all_present_surfaces_parse_cleanly": all_present_parse_clean,
            "all_present_surfaces_have_valid_output_item_indices": all_present_output_indices_valid,
            "authoritative_arguments_match_across_present_surfaces": not mismatch,
            "all_present_surfaces_match_expected_function_name": all_expected_functions_match,
            "all_present_surfaces_match_expected_arguments": all_expected_arguments_match,
            "all_present_surfaces_match_expected_model": all_expected_models_match,
            "all_present_surfaces_have_required_reasoning": all_required_reasoning_present,
            "all_present_surfaces_report_model": all_present_surfaces_report_model,
            "all_present_surfaces_same_model": all_present_surfaces_same_model,
            "all_required_surfaces_present": all_required_present,
        },
        "expected": {
            "model": expected_model,
            "function_name": expected_function_name,
            "arguments": expected_arguments,
            "require_reasoning_events": require_reasoning_events,
            "require_same_model": require_same_model,
        },
        "boundary": (
            "pass requires direct local server, panel gateway, and tunnel raw SSE "
            "captures for the same model with valid output item indices, required "
            "reasoning events when requested, and matching expected authoritative "
            "function_call arguments"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-sse", type=Path)
    parser.add_argument("--gateway-sse", type=Path)
    parser.add_argument("--tunnel-sse", type=Path)
    parser.add_argument("--expected-function-name")
    parser.add_argument("--expected-arguments")
    parser.add_argument("--expected-model")
    parser.add_argument("--require-reasoning-events", action="store_true")
    parser.add_argument("--require-same-model", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    artifact = build_artifact(
        direct_sse=args.direct_sse,
        gateway_sse=args.gateway_sse,
        tunnel_sse=args.tunnel_sse,
        expected_function_name=args.expected_function_name,
        expected_arguments=args.expected_arguments,
        expected_model=args.expected_model,
        require_reasoning_events=args.require_reasoning_events,
        require_same_model=args.require_same_model,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(args.out)
    print(f"status={artifact['status']}")
    print(f"missing_captures={artifact['missing_captures']}")


if __name__ == "__main__":
    main()
