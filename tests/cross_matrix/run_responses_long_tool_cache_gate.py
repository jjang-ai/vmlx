#!/usr/bin/env python3
"""Responses long-context tool/cache gate.

Runs a real multi-turn coding conversation against a live vMLX server and
preserves evidence under a release-gate directory. The gate is intentionally
API-level: it exercises /v1/responses, previous_response_id history, function
tool calls, /v1/cache/stats, and /health rather than source-only assertions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "inspect_symbol",
        "description": "Inspect a named function, class, or cache-related symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "symbol": {"type": "string"},
                "context_lines": {"type": "integer", "default": 30},
            },
            "required": ["path", "symbol"],
        },
    },
    {
        "type": "function",
        "name": "grep_repo",
        "description": "Search repo source for a literal or regex pattern.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "default": "."},
                "max_matches": {"type": "integer", "default": 20},
            },
            "required": ["pattern"],
        },
    },
]

FIXTURE_FILES = [
    "vmlx_engine/server.py",
    "vmlx_engine/scheduler.py",
    "vmlx_engine/mllm_batch_generator.py",
    "vmlx_engine/prefix_cache.py",
    "vmlx_engine/tq_disk_store.py",
]

GENERATED_TOOL_SEARCH_PARTS = {
    ".agents",
    ".venv",
    "build",
    "dist",
    "docs",
    "node_modules",
    "bundled-python",
    "__pycache__",
}


def _tool_search_sort_key(repo_root: Path, path: Path) -> tuple[int, int, str]:
    try:
        rel = path.relative_to(repo_root)
    except ValueError:
        rel = path
    parts = rel.parts
    root = parts[0] if parts else ""
    root_priority = {
        "vmlx_engine": 0,
        "tests": 1,
        "panel": 2,
    }.get(root, 3)
    suffix_priority = 1 if path.suffix == ".md" else 0
    return (root_priority, suffix_priority, str(rel))


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    return headers


def _get_json(base_url: str, route: str, api_key: str | None, timeout: int = 60) -> dict[str, Any]:
    resp = requests.get(f"{base_url}{route}", headers=_headers(api_key), timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _post_json(base_url: str, route: str, api_key: str | None, body: dict[str, Any], timeout: int) -> dict[str, Any]:
    resp = requests.post(f"{base_url}{route}", headers=_headers(api_key), json=body, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"{route} HTTP {resp.status_code}: {resp.text[:2000]}")
    return resp.json()


def _read_chunk(path: Path, target_chars: int, offset: int) -> str:
    text = path.read_text(errors="replace")
    if not text:
        return ""
    start = offset % max(len(text), 1)
    chunk = text[start : start + target_chars]
    if len(chunk) < target_chars:
        chunk += "\n" + text[: target_chars - len(chunk)]
    if "\n" in chunk:
        chunk = chunk[: chunk.rfind("\n")]
    return chunk


def _build_prompt(repo_root: Path, turn: int, target_chars_per_turn: int) -> str:
    file_rel = FIXTURE_FILES[(turn - 1) % len(FIXTURE_FILES)]
    file_path = repo_root / file_rel
    offset = ((turn - 1) // len(FIXTURE_FILES)) * target_chars_per_turn
    chunk = _read_chunk(file_path, target_chars_per_turn, offset)
    return (
        f"Turn {turn} of a long coding review. Keep track of prior turns.\n"
        "Task: review this cache/scheduler slice for correctness, identify one "
        "specific risk, and when tools are available you must call exactly one "
        "provided tool before answering.\n\n"
        f"FILE={file_rel} OFFSET={offset}\n"
        f"```python\n{chunk}\n```\n\n"
        "Return: (1) one risk or confirmation, (2) the function most relevant "
        "to prefix/L2/SSM cache reuse, (3) whether the answer depends on a tool result."
    )


def _instructions() -> str:
    return (
        "You are doing a real coding review of vMLX cache behavior. "
        "When tools are available, you must call exactly one provided tool "
        "before final output. When tools are not available, answer directly. "
        "When a tool result is provided, cite one exact file:line marker from "
        "that result as TOOL_EVIDENCE: <path:line>. Keep final output concise "
        "and do not loop."
    )


def _tools_enabled_for_turn(turn: int, total_turns: int, final_turn_no_tools: bool) -> bool:
    return not (final_turn_no_tools and turn == total_turns)


def _enable_thinking_for_turn(
    turn: int,
    total_turns: int,
    base_enable_thinking: bool,
    final_turn_disable_thinking: bool,
) -> bool:
    if final_turn_disable_thinking and turn == total_turns:
        return False
    return bool(base_enable_thinking)


def _extract_output_text(resp: dict[str, Any]) -> str:
    if isinstance(resp.get("output_text"), str):
        return resp["output_text"]
    parts: list[str] = []
    for item in resp.get("output") or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "output_text":
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
            continue
        if item_type != "message":
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict):
                content_type = content.get("type")
                if content_type not in {"output_text", "text"}:
                    continue
                text = content.get("text") or content.get("content")
                if isinstance(text, str):
                    parts.append(text)
    return "\n".join(parts)


def _extract_reasoning(resp: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in resp.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"reasoning", "reasoning_summary"}:
            for key in ("summary", "text", "content"):
                value = item.get(key)
                if isinstance(value, str):
                    parts.append(value)
                elif isinstance(value, list):
                    for entry in value:
                        if isinstance(entry, dict) and isinstance(entry.get("text"), str):
                            parts.append(entry["text"])
    return "\n".join(parts)


def _extract_function_calls(resp: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in resp.get("output") or []:
        if isinstance(item, dict) and item.get("type") == "function_call":
            calls.append(item)
    return calls


def _extract_warnings(resp: dict[str, Any]) -> list[str]:
    """Extract Responses warnings from object fields and warning SSE items."""
    seen: set[str] = set()
    warnings: list[str] = []

    def add(value: Any) -> None:
        if not isinstance(value, str):
            return
        text = value.strip()
        if not text or text in seen:
            return
        seen.add(text)
        warnings.append(text)

    for value in resp.get("warnings") or []:
        add(value)
    for item in resp.get("output") or []:
        if isinstance(item, dict) and item.get("type") == "response.warning":
            add(item.get("message"))
    return warnings


def _tool_output(repo_root: Path, call: dict[str, Any]) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    name = str(call.get("name") or "")
    call_id = str(call.get("call_id") or call.get("id") or f"call_{int(time.time() * 1000)}")
    try:
        args = json.loads(call.get("arguments") or "{}")
    except Exception:
        args = {}
    if name == "inspect_symbol":
        rel = str(args.get("path") or "vmlx_engine/server.py")
        symbol = str(args.get("symbol") or "_resolve_enable_thinking")
        context = int(args.get("context_lines") or 30)
        target = (repo_root / rel).resolve()
        try:
            target.relative_to(repo_root)
        except ValueError:
            output = f"{rel} is outside repo_root and cannot be inspected"
        else:
            source_files: list[Path]
            if target.is_file():
                source_files = [target]
            elif target.is_dir():
                source_files = sorted(
                    (
                    p
                    for p in target.rglob("*")
                    if p.is_file()
                    and p.suffix in {".py", ".ts", ".tsx", ".md"}
                    and GENERATED_TOOL_SEARCH_PARTS.isdisjoint(p.parts)
                    )
                    ,
                    key=lambda p: _tool_search_sort_key(repo_root, p),
                )
            else:
                source_files = []
            if not source_files:
                output = f"{rel} is not a readable file"
            else:
                match_path: Path | None = None
                match_lines: list[str] = []
                match_index = 0
                for candidate in source_files:
                    text = candidate.read_text(errors="replace").splitlines()
                    for i, line in enumerate(text):
                        if symbol in line:
                            match_path = candidate
                            match_lines = text
                            match_index = i
                            break
                    if match_path is not None:
                        break
                if match_path is None:
                    output = f"{symbol} not found under {rel}"
                else:
                    start = max(0, match_index - context)
                    end = min(len(match_lines), match_index + context + 1)
                    rel_match = match_path.relative_to(repo_root)
                    output = "\n".join(
                        f"{rel_match}:{idx + 1}: {line}"
                        for idx, line in enumerate(match_lines[start:end], start)
                    )
    elif name == "grep_repo":
        pattern = str(args.get("pattern") or "cache")
        rel = str(args.get("path") or ".")
        max_matches = int(args.get("max_matches") or 20)
        regex = re.compile(pattern)
        output_lines: list[str] = []
        for path in (repo_root / rel).rglob("*"):
            if len(output_lines) >= max_matches:
                break
            if not path.is_file() or path.suffix not in {".py", ".ts", ".tsx", ".md"}:
                continue
            try:
                for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                    if regex.search(line):
                        output_lines.append(f"{path.relative_to(repo_root)}:{lineno}: {line[:240]}")
                        if len(output_lines) >= max_matches:
                            break
            except Exception:
                continue
        output = "\n".join(output_lines) or "no matches"
    else:
        output = f"unknown tool {name}; available tools are inspect_symbol and grep_repo"
    return {"type": "function_call_output", "call_id": call_id, "output": output[:12000]}


def _usage_cached_tokens(resp: dict[str, Any]) -> int:
    usage = resp.get("usage") or {}
    for key in ("input_tokens_details", "prompt_tokens_details"):
        details = usage.get(key) or {}
        value = details.get("cached_tokens")
        if isinstance(value, int):
            return value
    return 0


def _max_cached_tokens(responses: list[dict[str, Any]]) -> int:
    """Maximum cached token count seen across one logical turn."""
    if not responses:
        return 0
    return max(_usage_cached_tokens(resp) for resp in responses)


def _cache_acceptance(
    rows: list[dict[str, Any]],
    *,
    require_cache_each_turn_after_first: bool,
) -> tuple[bool, bool]:
    """Return (any_cache_reuse_seen, strict_cache_requirement_satisfied)."""
    cached_after_first = [int(r.get("cached_tokens") or 0) for r in rows[1:]]
    cache_reuse_seen = any(v > 0 for v in cached_after_first)
    if not require_cache_each_turn_after_first:
        return cache_reuse_seen, True
    return cache_reuse_seen, all(v > 0 for v in cached_after_first)


def _tool_acceptance(
    rows: list[dict[str, Any]],
    *,
    require_tool_call: bool,
    require_tool_call_each_turn: bool,
) -> tuple[bool, bool]:
    """Return (any_tool_call_seen, every_tools_enabled_turn_called_tool)."""
    if not require_tool_call:
        return True, True
    tool_call_seen = any(r.get("function_calls") for r in rows)
    if not require_tool_call_each_turn:
        return tool_call_seen, True
    each_required_turn = all(
        (not r.get("tools_enabled")) or bool(r.get("function_calls"))
        for r in rows
    )
    return tool_call_seen, each_required_turn


def _loop_score(text: str) -> dict[str, Any]:
    tail = text[-1200:]
    if not tail:
        return {"loop_like": False, "tail_review": ""}
    chunks = [tail[i : i + 80] for i in range(0, max(len(tail) - 79, 0), 80)]
    repeated = len(chunks) - len(set(chunks))
    return {"loop_like": repeated >= 4, "tail_review": tail[-800:]}


def _tool_markup_leak(text: str) -> bool:
    """Return True when visible content still contains raw tool-call markup."""
    if not text:
        return False
    needles = (
        "<minimax:tool_call",
        "</minimax:tool_call",
        "<invoke name=",
        "</invoke>",
        "<｜DSML｜invoke",
        "</｜DSML｜invoke",
    )
    return any(needle in text for needle in needles)


_TOOL_FILE_LINE_RE = re.compile(
    r"(?m)^((?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.(?:py|ts|tsx|md):\d+):"
)


def _tool_grounding(visible: str, tool_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    """Check whether visible output cites a concrete file:line from tool output.

    The live gate previously accepted any visible text after a function call.
    That can hide a model that called a tool but ignored an empty or irrelevant
    result. For coding-review rows, require a real file:line marker from the
    tool output to appear in the final answer when the strict flag is enabled.
    """
    markers: list[str] = []
    seen: set[str] = set()
    for output in tool_outputs:
        text = output.get("output") if isinstance(output, dict) else None
        if not isinstance(text, str):
            continue
        for match in _TOOL_FILE_LINE_RE.finditer(text):
            marker = match.group(1)
            if marker not in seen:
                seen.add(marker)
                markers.append(marker)

    if not markers:
        return {
            "grounded": False,
            "marker": None,
            "markers": [],
            "reason": "no_file_line_tool_evidence",
        }

    visible_text = visible or ""
    for marker in markers:
        if marker in visible_text:
            return {
                "grounded": True,
                "marker": marker,
                "markers": markers[:20],
                "reason": "exact_file_line_mentioned",
            }
    return {
        "grounded": False,
        "marker": None,
        "markers": markers[:20],
        "reason": "no_exact_file_line_mentioned",
    }


def _resolution_tools_enabled(resolution_tool_choice: str) -> bool:
    return resolution_tool_choice in {"auto", "required"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("VMLINUX_BASE_URL", "http://127.0.0.1:8001"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=os.environ.get("VMLINUX_API_KEY"))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--target-chars-per-turn", type=int, default=10000)
    parser.add_argument("--max-output-tokens", type=int, default=1536)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--require-tool-call", action="store_true")
    parser.add_argument(
        "--tool-choice",
        choices=("auto", "required"),
        default="auto",
        help=(
            "tool_choice value to send when tools are enabled. Use required "
            "to verify the server's required-tool contract instead of relying "
            "on model self-selection under auto."
        ),
    )
    parser.add_argument(
        "--resolution-tool-choice",
        choices=("none", "auto", "required"),
        default="none",
        help=(
            "tool_choice to send on in-turn function_call_output follow-up "
            "rounds. Use auto for realistic multi-tool coding conversations; "
            "none forces a visible answer and treats raw tool markup as a leak."
        ),
    )
    parser.add_argument(
        "--require-tool-call-each-turn",
        dest="require_tool_call_each_turn",
        action="store_true",
        help=(
            "When --require-tool-call is set, also fail unless every "
            "tools-enabled logical turn produced at least one function_call."
        ),
    )
    parser.add_argument(
        "--require-tool-evidence",
        action="store_true",
        help=(
            "Fail any tools-enabled turn with function calls unless the final "
            "visible answer cites an exact file:line marker from the tool output."
        ),
    )
    parser.add_argument(
        "--final-turn-no-tools",
        action="store_true",
        help=(
            "Omit tools from the final turn so the gate must produce a visible "
            "answer after earlier tool-call turns instead of looping on calls."
        ),
    )
    parser.add_argument(
        "--final-turn-disable-thinking",
        action="store_true",
        help=(
            "Send enable_thinking=false on the final turn. This explicitly "
            "tests the supported reasoning-off escape hatch after earlier "
            "reasoning/tool turns."
        ),
    )
    parser.add_argument(
        "--resolve-tool-calls-in-turn",
        dest="resolve_tool_calls_in_turn",
        action="store_true",
        help=(
            "When a Responses turn emits function_call items, submit the "
            "function_call_output immediately under the same logical turn. "
            "This is the realistic tool-call flow and avoids changing the "
            "final turn's tool schema just to force visible text."
        ),
    )
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=2,
        help="Maximum in-turn function-call resolution rounds per logical turn.",
    )
    parser.add_argument(
        "--require-cache-each-turn-after-first",
        action="store_true",
        help=(
            "Fail unless every chained turn after turn 1 reports cached_tokens "
            "> 0. Use for strict prefix/L2 cache proof; default mode only "
            "requires at least one post-turn-1 hit."
        ),
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    startup = {
        "health": _get_json(args.base_url, "/health", args.api_key),
        "cache_stats": _get_json(args.base_url, "/v1/cache/stats", args.api_key),
    }
    (artifact_dir / "00_startup.json").write_text(json.dumps(startup, indent=2))

    previous_response_id: str | None = None
    rows: list[dict[str, Any]] = []
    tool_call_seen = False
    pending_tool_outputs: list[dict[str, Any]] = []

    for turn in range(1, args.turns + 1):
        input_payload: str | list[dict[str, Any]]
        prompt = _build_prompt(repo_root, turn, args.target_chars_per_turn)
        tools_enabled = _tools_enabled_for_turn(
            turn,
            args.turns,
            bool(args.final_turn_no_tools),
        )
        enable_thinking = _enable_thinking_for_turn(
            turn,
            args.turns,
            bool(args.enable_thinking),
            bool(args.final_turn_disable_thinking),
        )
        if not tools_enabled:
            prompt += (
                "\n\nFinal synthesis turn: do not request tools. Produce a visible "
                "answer that states whether the prior cache/tool conversation stayed "
                "coherent and whether cached_tokens were observed."
            )
        if pending_tool_outputs:
            input_payload = [*pending_tool_outputs, {"role": "user", "content": prompt}]
        else:
            input_payload = prompt
        pending_tool_outputs = []

        body: dict[str, Any] = {
            "model": args.model,
            "input": input_payload,
            "instructions": _instructions(),
            "store": True,
            "stream": False,
            "max_output_tokens": args.max_output_tokens,
            "enable_thinking": enable_thinking,
        }
        if tools_enabled:
            body["tools"] = TOOL_DEFINITIONS
            body["tool_choice"] = args.tool_choice
        if previous_response_id:
            body["previous_response_id"] = previous_response_id

        (artifact_dir / f"turn_{turn:02d}_request.json").write_text(
            json.dumps(body, indent=2)
        )

        turn_responses: list[dict[str, Any]] = []
        round_rows: list[dict[str, Any]] = []
        total_elapsed_s = 0.0
        current_body = body
        round_index = 0
        final_calls: list[dict[str, Any]] = []
        turn_tool_outputs: list[dict[str, Any]] = []

        while True:
            (artifact_dir / f"turn_{turn:02d}_request_round_{round_index:02d}.json").write_text(
                json.dumps(current_body, indent=2)
            )
            t0 = time.time()
            response = _post_json(
                args.base_url,
                "/v1/responses",
                args.api_key,
                current_body,
                args.timeout,
            )
            elapsed_s = time.time() - t0
            total_elapsed_s += elapsed_s
            turn_responses.append(response)
            previous_response_id = response.get("id") or previous_response_id
            (artifact_dir / f"turn_{turn:02d}_response_round_{round_index:02d}.json").write_text(
                json.dumps(response, indent=2)
            )
            calls_this_round = _extract_function_calls(response)
            final_calls = calls_this_round
            round_rows.append(
                {
                    "round": round_index,
                    "elapsed_s": round(elapsed_s, 3),
                    "response_id": previous_response_id,
                    "cached_tokens": _usage_cached_tokens(response),
                    "visible_chars": len(_extract_output_text(response)),
                    "reasoning_chars": len(_extract_reasoning(response)),
                    "function_calls": [
                        {
                            "name": c.get("name"),
                            "call_id": c.get("call_id") or c.get("id"),
                        }
                        for c in calls_this_round
                    ],
                    "warnings": _extract_warnings(response),
                }
            )
            if not (
                args.resolve_tool_calls_in_turn
                and tools_enabled
                and calls_this_round
                and round_index < max(0, int(args.max_tool_rounds))
            ):
                break

            tool_outputs = [_tool_output(repo_root, call) for call in calls_this_round]
            turn_tool_outputs.extend(tool_outputs)
            (artifact_dir / f"turn_{turn:02d}_tool_outputs_round_{round_index:02d}.json").write_text(
                json.dumps(tool_outputs, indent=2)
            )
            round_index += 1
            current_body = {
                "model": args.model,
                "input": tool_outputs,
                "instructions": (
                    _instructions()
                    + " Tool results are provided. Produce a visible answer "
                    "now; do not call another tool in this resolution round."
                ),
                "store": True,
                "stream": False,
                "max_output_tokens": args.max_output_tokens,
                "enable_thinking": enable_thinking,
                "previous_response_id": previous_response_id,
            }
            current_body["tool_choice"] = args.resolution_tool_choice
            if _resolution_tools_enabled(args.resolution_tool_choice):
                current_body["tools"] = TOOL_DEFINITIONS

        response = turn_responses[-1]
        visible = "\n".join(
            text for text in (_extract_output_text(r) for r in turn_responses) if text
        )
        reasoning = "\n".join(
            text for text in (_extract_reasoning(r) for r in turn_responses) if text
        )
        all_calls = [
            call for turn_response in turn_responses
            for call in _extract_function_calls(turn_response)
        ]
        warnings = []
        for turn_response in turn_responses:
            warnings.extend(_extract_warnings(turn_response))
        tool_call_seen = tool_call_seen or bool(all_calls)
        if args.resolve_tool_calls_in_turn:
            pending_tool_outputs = []
        else:
            pending_tool_outputs = [_tool_output(repo_root, call) for call in final_calls]
            turn_tool_outputs.extend(pending_tool_outputs)
        cache_stats = _get_json(args.base_url, "/v1/cache/stats", args.api_key)
        health = _get_json(args.base_url, "/health", args.api_key)
        loop = _loop_score(visible or reasoning)
        grounding = _tool_grounding(visible, turn_tool_outputs)

        row = {
            "turn": turn,
            "elapsed_s": round(total_elapsed_s, 3),
            "response_id": previous_response_id,
            "previous_response_id": body.get("previous_response_id"),
            "cached_tokens": _max_cached_tokens(turn_responses),
            "input_tokens": (response.get("usage") or {}).get("input_tokens"),
            "output_tokens": (response.get("usage") or {}).get("output_tokens"),
            "tools_enabled": tools_enabled,
            "enable_thinking": enable_thinking,
            "rounds": round_rows,
            "function_calls": [
                {"name": c.get("name"), "call_id": c.get("call_id") or c.get("id")}
                for c in all_calls
            ],
            "function_call_outputs_next_turn": len(pending_tool_outputs),
            "warnings": warnings,
            "visible_chars": len(visible),
            "reasoning_chars": len(reasoning),
            "loop_like": loop["loop_like"],
            "tool_markup_leak": _tool_markup_leak(visible),
            "tool_grounded": grounding["grounded"],
            "tool_evidence_marker": grounding["marker"],
            "tool_evidence_markers": grounding["markers"],
            "tool_evidence_reason": grounding["reason"],
            "tail_review": loop["tail_review"],
            "cache_totals": cache_stats.get("cache_totals"),
            "scheduler_stats": cache_stats.get("scheduler_stats"),
            "scheduler_cache": cache_stats.get("scheduler_cache"),
            "block_disk_cache": cache_stats.get("block_disk_cache"),
            "ssm_companion": cache_stats.get("ssm_companion"),
            "health_cache": health.get("cache"),
        }
        rows.append(row)
        (artifact_dir / f"turn_{turn:02d}_response.json").write_text(json.dumps(response, indent=2))
        (artifact_dir / f"turn_{turn:02d}_visible.txt").write_text(visible)
        (artifact_dir / f"turn_{turn:02d}_reasoning.txt").write_text(reasoning)
        (artifact_dir / f"turn_{turn:02d}_cache_stats.json").write_text(json.dumps(cache_stats, indent=2))
        (artifact_dir / f"turn_{turn:02d}_health.json").write_text(json.dumps(health, indent=2))
        (artifact_dir / f"turn_{turn:02d}_tail_review.txt").write_text(loop["tail_review"])

        print(
            f"turn={turn} elapsed={total_elapsed_s:.1f}s cached_tokens={row['cached_tokens']} "
            f"visible={len(visible)} reasoning={len(reasoning)} calls={len(all_calls)}"
        )

    cache_reuse_seen, cache_requirement_satisfied = _cache_acceptance(
        rows,
        require_cache_each_turn_after_first=bool(
            args.require_cache_each_turn_after_first
        ),
    )
    tool_call_seen, tool_call_each_required_turn = _tool_acceptance(
        rows,
        require_tool_call=bool(args.require_tool_call),
        require_tool_call_each_turn=bool(args.require_tool_call_each_turn),
    )
    no_loop = all(not r["loop_like"] for r in rows)
    no_tool_markup_leak = all(not r["tool_markup_leak"] for r in rows)
    tool_grounding_satisfied = (
        True
        if not args.require_tool_evidence
        else all(
            (not r.get("tools_enabled"))
            or not r.get("function_calls")
            or bool(r.get("tool_grounded"))
            for r in rows
        )
    )
    outputs_visible_or_tools = all(r["visible_chars"] > 0 or r["function_calls"] for r in rows)
    visible_output_observed = any(r["visible_chars"] > 0 for r in rows)
    final_turn_visible_output = bool(rows and rows[-1]["visible_chars"] > 0)
    final_turn_tools_disabled = bool(
        not args.final_turn_no_tools or (rows and rows[-1].get("tools_enabled") is False)
    )
    final_turn_thinking_disabled = bool(
        not args.final_turn_disable_thinking
        or (rows and rows[-1].get("enable_thinking") is False)
    )
    prior_id_used = all(r["previous_response_id"] for r in rows[1:]) if len(rows) > 1 else True
    acceptance = {
        "turns_completed": len(rows) == args.turns,
        "previous_response_id_used": prior_id_used,
        "cache_reuse_observed": cache_reuse_seen,
        "require_cache_each_turn_after_first": bool(
            args.require_cache_each_turn_after_first
        ),
        "cache_reuse_each_turn_after_first": cache_requirement_satisfied,
        "tool_call_observed": tool_call_seen,
        "require_tool_call_each_turn": bool(args.require_tool_call_each_turn),
        "tool_call_each_required_turn": tool_call_each_required_turn,
        "require_tool_evidence": bool(args.require_tool_evidence),
        "tool_evidence_each_required_turn": tool_grounding_satisfied,
        "final_turn_tools_disabled": final_turn_tools_disabled,
        "final_turn_thinking_disabled": final_turn_thinking_disabled,
        "visible_or_tool_output_each_turn": outputs_visible_or_tools,
        "visible_output_observed": visible_output_observed,
        "final_turn_visible_output": final_turn_visible_output,
        "no_loop_like_tail": no_loop,
        "no_tool_markup_leak": no_tool_markup_leak,
    }
    overall_pass = all(acceptance.values())
    summary = {
        "base_url": args.base_url,
        "model": args.model,
        "turns": args.turns,
        "target_chars_per_turn": args.target_chars_per_turn,
        "final_turn_no_tools": bool(args.final_turn_no_tools),
        "final_turn_disable_thinking": bool(args.final_turn_disable_thinking),
        "resolve_tool_calls_in_turn": bool(args.resolve_tool_calls_in_turn),
        "tool_choice_mode": args.tool_choice,
        "resolution_tool_choice": args.resolution_tool_choice,
        "require_tool_call_each_turn": bool(args.require_tool_call_each_turn),
        "require_tool_evidence": bool(args.require_tool_evidence),
        "require_cache_each_turn_after_first": bool(
            args.require_cache_each_turn_after_first
        ),
        "overall_pass": overall_pass,
        "rows": rows,
        "acceptance": acceptance,
    }
    (artifact_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    md_rows = "\n".join(
        f"| {r['turn']} | {r['elapsed_s']} | {r['cached_tokens']} | "
        f"{r['visible_chars']} | {r['reasoning_chars']} | "
        f"{len(r['function_calls'])} | {len(r.get('warnings') or [])} | "
        f"{r['loop_like']} | {r['tool_markup_leak']} | {r['tool_grounded']} |"
        for r in rows
    )
    (artifact_dir / "SUMMARY.md").write_text(
        "# Responses Long Tool Cache Gate\n\n"
        f"- model: `{args.model}`\n"
        f"- target_chars_per_turn: `{args.target_chars_per_turn}`\n"
        f"- turns: `{args.turns}`\n\n"
        f"- final_turn_no_tools: `{bool(args.final_turn_no_tools)}`\n\n"
        f"- final_turn_disable_thinking: `{bool(args.final_turn_disable_thinking)}`\n\n"
        f"- resolve_tool_calls_in_turn: `{bool(args.resolve_tool_calls_in_turn)}`\n\n"
        f"- tool_choice_mode: `{args.tool_choice}`\n\n"
        f"- resolution_tool_choice: `{args.resolution_tool_choice}`\n\n"
        f"- require_tool_call_each_turn: `{bool(args.require_tool_call_each_turn)}`\n\n"
        f"- require_tool_evidence: `{bool(args.require_tool_evidence)}`\n\n"
        f"- require_cache_each_turn_after_first: `{bool(args.require_cache_each_turn_after_first)}`\n\n"
        "## Acceptance\n\n"
        f"- overall_pass: `{overall_pass}`\n"
        + "\n".join(f"- {k}: `{v}`" for k, v in acceptance.items())
        + "\n\n## Rows\n\n"
        "| turn | elapsed_s | cached_tokens | visible_chars | reasoning_chars | function_calls | warnings | loop_like | tool_markup_leak | tool_grounded |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |\n"
        f"{md_rows}\n\n"
        "Raw response, cache, health, and tail_review files are preserved next to this summary.\n"
    )
    result = "PASS" if overall_pass else "FAIL"
    print(f"overall_pass={overall_pass} result={result}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
