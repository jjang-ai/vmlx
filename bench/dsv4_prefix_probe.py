#!/usr/bin/env python3
"""DSV4 prefix-cache identity probe — no model load, encoder only.

Renders the cache identity (add_generation_prompt=False) for each turn of a
realistic agentic tool transcript and reports where consecutive turns diverge.
A healthy prefix cache requires identity(turn N) to be a strict prefix of
identity(turn N+1).
"""
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
            "description": "Run a shell command in the workspace and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command"},
                    "timeout": {"type": "number", "description": "Seconds"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write UTF-8 text content to a path, creating parent dirs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
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
            "name": "list_dir",
            "description": "List entries in a directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]

SYS = {"role": "system", "content": "You are a helpful coding agent."}


def tc(cid, name, args):
    return {
        "id": cid,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


# A realistic agent loop: explicit tool request, then follow-ups that stop
# naming the tool (exactly the pattern the panel produces).
TURNS = [
    # turn 0 — first user, explicitly names run_command
    [SYS, {"role": "user", "content": "Please use run_command to list the workspace."}],
    # turn 1 — after tool call + result
    [
        SYS,
        {"role": "user", "content": "Please use run_command to list the workspace."},
        {"role": "assistant", "content": "", "reasoning_content": "List first.",
         "tool_calls": [tc("c1", "run_command", {"command": "ls -la"})]},
        {"role": "tool", "tool_call_id": "c1", "content": "total 0\ndrwxr-xr-x app.py"},
    ],
    # turn 2 — assistant answered, new user turn that does NOT name a tool
    [
        SYS,
        {"role": "user", "content": "Please use run_command to list the workspace."},
        {"role": "assistant", "content": "", "reasoning_content": "List first.",
         "tool_calls": [tc("c1", "run_command", {"command": "ls -la"})]},
        {"role": "tool", "tool_call_id": "c1", "content": "total 0\ndrwxr-xr-x app.py"},
        {"role": "assistant", "content": "The workspace has app.py.",
         "reasoning_content": "Report it."},
        {"role": "user", "content": "Now build me a minesweeper.html game."},
    ],
    # turn 3 — agent calls write_file
    [
        SYS,
        {"role": "user", "content": "Please use run_command to list the workspace."},
        {"role": "assistant", "content": "", "reasoning_content": "List first.",
         "tool_calls": [tc("c1", "run_command", {"command": "ls -la"})]},
        {"role": "tool", "tool_call_id": "c1", "content": "total 0\ndrwxr-xr-x app.py"},
        {"role": "assistant", "content": "The workspace has app.py.",
         "reasoning_content": "Report it."},
        {"role": "user", "content": "Now build me a minesweeper.html game."},
        {"role": "assistant", "content": "", "reasoning_content": "Write the file.",
         "tool_calls": [tc("c2", "write_file", {"path": "minesweeper.html", "content": "<html>...</html>"})]},
        {"role": "tool", "tool_call_id": "c2", "content": "wrote 7168 bytes"},
    ],
    # turn 4 — next agent iteration, another tool call
    [
        SYS,
        {"role": "user", "content": "Please use run_command to list the workspace."},
        {"role": "assistant", "content": "", "reasoning_content": "List first.",
         "tool_calls": [tc("c1", "run_command", {"command": "ls -la"})]},
        {"role": "tool", "tool_call_id": "c1", "content": "total 0\ndrwxr-xr-x app.py"},
        {"role": "assistant", "content": "The workspace has app.py.",
         "reasoning_content": "Report it."},
        {"role": "user", "content": "Now build me a minesweeper.html game."},
        {"role": "assistant", "content": "", "reasoning_content": "Write the file.",
         "tool_calls": [tc("c2", "write_file", {"path": "minesweeper.html", "content": "<html>...</html>"})]},
        {"role": "tool", "tool_call_id": "c2", "content": "wrote 7168 bytes"},
        {"role": "assistant", "content": "", "reasoning_content": "Verify it.",
         "tool_calls": [tc("c3", "read_file", {"path": "minesweeper.html"})]},
        {"role": "tool", "tool_call_id": "c3", "content": "<html>...</html>"},
    ],
]


def render(msgs, tools=TOOLS, gen_prompt=False):
    return E.apply_chat_template(
        msgs,
        enable_thinking=True,
        reasoning_effort="low",
        tools=tools,
        add_generation_prompt=gen_prompt,
        model_path=MODEL_PATH,
    )


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def main():
    print(f"bundle={MODEL_PATH}")
    ids = []
    for i, msgs in enumerate(TURNS):
        try:
            p = render(msgs)
        except Exception as exc:  # noqa: BLE001
            print(f"turn {i}: RENDER FAILED: {type(exc).__name__}: {exc}")
            ids.append(None)
            continue
        ids.append(p)
        sel = E.prompt_tool_catalog(TOOLS)
        names = [t["function"]["name"] for t in (sel or [])]
        print(f"turn {i}: identity_chars={len(p):6d}  prompt_tools={names}")

    print("\n=== consecutive prefix reuse (chars) ===")
    for i in range(1, len(ids)):
        a, b = ids[i - 1], ids[i]
        if a is None or b is None:
            print(f"turn {i-1}->{i}: SKIP (render failure)")
            continue
        n = lcp(a, b)
        ok = "PREFIX-OK" if n == len(a) else "*** DIVERGES ***"
        pct = 100.0 * n / max(len(a), 1)
        print(f"turn {i-1}->{i}: reuse={n}/{len(a)} chars ({pct:5.1f}%)  {ok}")
        if n != len(a):
            print(f"    prev @{n}: {a[n:n+160]!r}")
            print(f"    next @{n}: {b[n:n+160]!r}")

    print("\n=== control: explicit full-catalog render ===")
    if True:
        ids2 = []
        for msgs in TURNS:
            try:
                ids2.append(render(msgs))
            except Exception as exc:  # noqa: BLE001
                print(f"  render failed: {exc}")
                ids2.append(None)
        for i in range(1, len(ids2)):
            a, b = ids2[i - 1], ids2[i]
            if a is None or b is None:
                continue
            n = lcp(a, b)
            ok = "PREFIX-OK" if n == len(a) else "*** DIVERGES ***"
            pct = 100.0 * n / max(len(a), 1)
            print(f"turn {i-1}->{i}: reuse={n}/{len(a)} chars ({pct:5.1f}%)  {ok}")
            if n != len(a):
                print(f"    prev @{n}: {a[n:n+160]!r}")
                print(f"    next @{n}: {b[n:n+160]!r}")


if __name__ == "__main__":
    main()
