#!/usr/bin/env python3
"""R21 raw streaming API proof for DSV4-Flash: tools + reasoning + cache reuse.

Pairs with the Electron CDP proof. Exercises /v1/chat/completions streaming with
a broad tool catalog, then a tool-result continuation, and reports the engine's
cache counters around each request.
"""
import json
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "DeepSeek-V4-Flash-0731-JANG"

TOOLS = [
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Execute a shell command in the working directory.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file and return its content.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write UTF-8 text content to a path.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
]


def cache_stats():
    with urllib.request.urlopen(f"{BASE}/health", timeout=10) as r:
        d = json.load(r)
    c = d.get("cache", {}).get("scheduler_cache", {})
    return {"hits": c.get("cache_hits"), "misses": c.get("cache_misses"),
            "saved": c.get("tokens_saved")}


def stream(messages, label):
    body = json.dumps({
        "model": MODEL, "messages": messages, "tools": TOOLS,
        "tool_choice": "auto", "stream": True, "max_tokens": 512,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})

    reasoning, content, chunks = [], [], 0
    tool_calls, finish, usage = {}, None, None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            evt = json.loads(payload)
            if evt.get("usage"):
                usage = evt["usage"]
            for ch in evt.get("choices") or []:
                chunks += 1
                delta = ch.get("delta") or {}
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                if delta.get("content"):
                    content.append(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = tool_calls.setdefault(idx, {"name": "", "args": ""})
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]

    print(f"\n--- {label} ---")
    print(f"chunks={chunks} finish_reason={finish}")
    print(f"reasoning_deltas={len(reasoning)} chars={sum(len(x) for x in reasoning)}")
    print(f"content_deltas={len(content)} chars={sum(len(x) for x in content)}")
    print(f"tool_calls={json.dumps(tool_calls)[:400]}")
    print(f"usage={json.dumps(usage)}")
    if reasoning:
        print(f"reasoning[:200]={''.join(reasoning)[:200]!r}")
    if content:
        print(f"content[:300]={''.join(content)[:300]!r}")
    return {"reasoning": "".join(reasoning), "content": "".join(content),
            "tool_calls": tool_calls, "finish": finish}


def main():
    base_msgs = [
        {"role": "system", "content": "You are a helpful coding agent."},
        {"role": "user",
         "content": "Read the file /workspace/r21_probe_one.txt "
                    "and tell me exactly what it contains."},
    ]

    pre = cache_stats()
    print(f"cache before: {pre}")
    r1 = stream(base_msgs, "request 1 (expect a tool call)")
    mid = cache_stats()
    print(f"cache after r1: {mid}")

    if not r1["tool_calls"]:
        print("\nNo tool call emitted; cannot run the continuation leg.")
        return 1

    slot = r1["tool_calls"][sorted(r1["tool_calls"])[0]]
    cont = base_msgs + [
        {"role": "assistant", "content": r1["content"] or "",
         "reasoning_content": r1["reasoning"],
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": slot["name"],
                                      "arguments": slot["args"]}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "R21_TOOL_ONE"},
    ]
    r2 = stream(cont, "request 2 (tool-result continuation)")
    post = cache_stats()
    print(f"cache after r2: {post}")

    print("\n=== cache deltas ===")
    print(f"r1: hits +{mid['hits'] - pre['hits']} misses +{mid['misses'] - pre['misses']} "
          f"saved +{mid['saved'] - pre['saved']}")
    print(f"r2: hits +{post['hits'] - mid['hits']} misses +{post['misses'] - mid['misses']} "
          f"saved +{post['saved'] - mid['saved']}")
    ok = bool(r2["content"]) and r2["finish"] == "stop"
    print(f"\ncontinuation produced visible answer: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
