#!/usr/bin/env python3
"""DSV4 interleaved reasoning/tool prefix probe — encoder only, no model load.

DSV4 keeps every ``<think>`` block once tools are present: the bundle encoder
force-disables ``drop_thinking`` when any message carries ``tools``. An agentic
loop is therefore a chain of intra-turn prefills

    reason -> tool_call -> tool_result -> reason -> tool_call -> ... -> answer

and each link must extend the previous one *exactly*, or the scheduler
re-prefills the whole transcript on every iteration.

This probe renders the cache identity at each intra-turn iteration under two
history-replay policies and reports the longest common prefix between
consecutive iterations:

  KEEP  — assistant turns replay their ``reasoning_content`` (correct)
  DROP  — assistant turns replay without ``reasoning_content`` (what a client
          that does not echo reasoning back would send)

Usage:
    VMLX_SRC=/path/to/checkout python3 bench/dsv4_interleaved_probe.py
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(
    0,
    os.environ.get("VMLX_SRC", str(Path(__file__).resolve().parents[1])),
)

from vmlx_engine.loaders import dsv4_chat_encoder as E  # noqa: E402, N812

MODEL_PATH = os.environ.get(
    "DSV4_BUNDLE", "/path/to/DeepSeek-V4-Flash-0731-JANG"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command in the working directory.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file and return its content.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write UTF-8 text content to a path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
]

SYS = {"role": "system", "content": "You are a helpful coding agent."}
USER = {
    "role": "user",
    "content": "Create r21_probe_one.txt containing R21_TOOL_ONE, then verify it.",
}


def call(cid, name, args, reasoning):
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": reasoning,
        "tool_calls": [
            {
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }


def result(cid, content):
    return {"role": "tool", "tool_call_id": cid, "content": content}


# One user turn, four agentic iterations: the prefill chain the scheduler sees.
ITERATIONS = [
    [SYS, USER],
    [
        SYS,
        USER,
        call("c1", "write_file", {"path": "r21_probe_one.txt", "content": "R21_TOOL_ONE"},
             "I need to create the file first."),
        result("c1", "wrote 12 bytes"),
    ],
    [
        SYS,
        USER,
        call("c1", "write_file", {"path": "r21_probe_one.txt", "content": "R21_TOOL_ONE"},
             "I need to create the file first."),
        result("c1", "wrote 12 bytes"),
        call("c2", "read_file", {"path": "r21_probe_one.txt"},
             "Now verify by reading it back."),
        result("c2", "R21_TOOL_ONE"),
    ],
    [
        SYS,
        USER,
        call("c1", "write_file", {"path": "r21_probe_one.txt", "content": "R21_TOOL_ONE"},
             "I need to create the file first."),
        result("c1", "wrote 12 bytes"),
        call("c2", "read_file", {"path": "r21_probe_one.txt"},
             "Now verify by reading it back."),
        result("c2", "R21_TOOL_ONE"),
        call("c3", "run_command", {"command": "wc -c r21_probe_one.txt"},
             "Confirm the byte count too."),
        result("c3", "12 r21_probe_one.txt"),
    ],
]


def strip_reasoning(messages):
    out = copy.deepcopy(messages)
    for msg in out:
        if msg.get("role") == "assistant":
            msg.pop("reasoning_content", None)
    return out


def render(messages):
    return E.apply_chat_template(
        messages,
        enable_thinking=True,
        reasoning_effort="low",
        tools=TOOLS,
        add_generation_prompt=False,
        model_path=MODEL_PATH,
    )


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def run(label, transform):
    print(f"\n=== {label} ===")
    ids = []
    for i, msgs in enumerate(ITERATIONS):
        try:
            ids.append(render(transform(msgs)))
        except Exception as exc:  # noqa: BLE001
            print(f"iter {i}: RENDER FAILED: {type(exc).__name__}: {exc}")
            ids.append(None)
    ok = True
    for i in range(1, len(ids)):
        a, b = ids[i - 1], ids[i]
        if a is None or b is None:
            ok = False
            continue
        n = lcp(a, b)
        verdict = "PREFIX-OK" if n == len(a) else "*** DIVERGES ***"
        if n != len(a):
            ok = False
        print(f"iter {i-1}->{i}: reuse={n}/{len(a)} chars "
              f"({100.0 * n / max(len(a), 1):5.1f}%)  {verdict}")
        if n != len(a):
            print(f"    prev @{n}: {a[n:n+140]!r}")
            print(f"    next @{n}: {b[n:n+140]!r}")
    return ids, ok


def main():
    print(f"bundle={MODEL_PATH}")
    keep_ids, keep_ok = run("KEEP reasoning_content (correct replay)", lambda m: m)
    drop_ids, drop_ok = run("DROP reasoning_content (client does not echo)", strip_reasoning)

    print("\n=== KEEP vs DROP at the same iteration ===")
    for i, (a, b) in enumerate(zip(keep_ids, drop_ids)):
        if a is None or b is None:
            continue
        n = lcp(a, b)
        same = "identical" if a == b else f"share {n} chars, keep={len(a)} drop={len(b)}"
        print(f"iter {i}: {same}")
        if a != b:
            print(f"    keep @{n}: {a[n:n+140]!r}")
            print(f"    drop @{n}: {b[n:n+140]!r}")

    print(f"\nsummary: keep_chain_ok={keep_ok} drop_chain_ok={drop_ok}")
    return 0 if keep_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
