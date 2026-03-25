#!/usr/bin/env python3
"""
Stress test: Tool call loops on 122B model.

Simulates agentic tool-call loops with multiple iterations per conversation,
monitoring memory and compute after each round. Tests both streaming and
non-streaming paths.
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:2310")
API_URL = f"{BASE_URL}/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}
PID = os.environ.get("VLLM_PID", "85152")

# Tools definition for the model
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files matching a pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to search for",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Directory to search in",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]

# Safe calculator that only handles basic math
SAFE_OPERATORS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b if b != 0 else float("inf"),
    "**": lambda a, b: a ** b,
    "^": lambda a, b: a ** b,
}


def safe_calc(expression: str) -> str:
    """Safely evaluate simple math expressions without eval()."""
    import re
    # Handle simple "a op b" expressions
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(\*\*|\^|[+\-*/])\s*(\d+(?:\.\d+)?)\s*$", expression)
    if match:
        a, op, b = float(match.group(1)), match.group(2), float(match.group(3))
        if op in SAFE_OPERATORS:
            result = SAFE_OPERATORS[op](a, b)
            return str(int(result) if result == int(result) else result)
    return f"Cannot evaluate: {expression}"


# Simulated tool results
TOOL_RESULTS = {
    "get_weather": lambda args: json.dumps(
        {
            "temperature": 72,
            "unit": args.get("unit", "fahrenheit"),
            "condition": "sunny",
            "humidity": 45,
            "location": args.get("location", "unknown"),
        }
    ),
    "search_files": lambda args: json.dumps(
        {
            "files": [
                f"/home/user/projects/{args.get('pattern', '*.py').replace('*', 'main')}.bak",
                f"/home/user/docs/{args.get('pattern', '*.py').replace('*', 'readme')}.bak",
            ],
            "count": 2,
        }
    ),
    "read_file": lambda args: json.dumps(
        {
            "content": f"# File: {args.get('path', 'unknown')}\n\nThis is the content of the file.\n"
            + "It contains some important data.\n" * 10,
            "size": 512,
        }
    ),
    "calculator": lambda args: json.dumps(
        {"result": safe_calc(args.get("expression", "0")), "expression": args.get("expression", "0")}
    ),
}


def get_memory_gb():
    """Get RSS of the vllm process in GB."""
    try:
        result = subprocess.run(
            ["ssh", "macstudio", f"ps -o rss= -p {PID}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        rss_kb = int(result.stdout.strip())
        return rss_kb / 1024 / 1024
    except Exception:
        return -1


def get_cpu_percent():
    """Get CPU% of the vllm process."""
    try:
        result = subprocess.run(
            ["ssh", "macstudio", f"ps -o %cpu= -p {PID}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return float(result.stdout.strip())
    except Exception:
        return -1


def send_request(messages, stream=False, max_tokens=500, enable_thinking=False):
    """Send a chat completion request and return the response."""
    data = {
        "model": "test",
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
        "tools": TOOLS,
        "tool_choice": "auto",
        "enable_thinking": enable_thinking,
    }
    if stream:
        data["stream_options"] = {"include_usage": True}

    r = requests.post(API_URL, json=data, headers=HEADERS, timeout=300, stream=stream)

    if not stream:
        return r.json()

    # Parse SSE stream
    content = ""
    reasoning = ""
    tool_calls = []
    usage = {}
    current_tool_calls = {}  # index -> {id, function: {name, arguments}}

    for line in r.iter_lines():
        if not line:
            continue
        text = line.decode("utf-8")
        if text.startswith("data: "):
            text = text[6:]
        if text == "[DONE]":
            break
        try:
            chunk = json.loads(text)
            delta = chunk.get("choices", [{}])[0].get("delta", {})

            if delta.get("content"):
                content += delta["content"]
            if delta.get("reasoning_content"):
                reasoning += delta["reasoning_content"]

            # Accumulate tool calls
            if delta.get("tool_calls"):
                for tc in delta["tool_calls"]:
                    idx = tc.get("index", 0)
                    if idx not in current_tool_calls:
                        current_tool_calls[idx] = {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc.get("id"):
                        current_tool_calls[idx]["id"] = tc["id"]
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        current_tool_calls[idx]["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        current_tool_calls[idx]["function"]["arguments"] += fn[
                            "arguments"
                        ]

            if chunk.get("usage"):
                usage = chunk["usage"]
        except Exception:
            pass

    tool_calls = [current_tool_calls[i] for i in sorted(current_tool_calls.keys())]

    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content or None,
                    "reasoning_content": reasoning or None,
                    "tool_calls": tool_calls or None,
                }
            }
        ],
        "usage": usage,
    }


def execute_tool(tool_call):
    """Simulate executing a tool call."""
    fn_name = tool_call["function"]["name"]
    try:
        args = json.loads(tool_call["function"]["arguments"])
    except json.JSONDecodeError:
        args = {}

    handler = TOOL_RESULTS.get(fn_name)
    if handler:
        return handler(args)
    return json.dumps({"error": f"Unknown tool: {fn_name}"})


def run_agentic_loop(
    task_prompt, max_iterations=5, stream=True, enable_thinking=False, label=""
):
    """Run a multi-turn agentic loop with tool calls."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant with access to tools. Use them when needed. Be concise.",
        },
        {"role": "user", "content": task_prompt},
    ]

    print(f"\n{'='*70}")
    print(f"AGENTIC LOOP: {label}")
    print(f"  Task: {task_prompt[:80]}...")
    print(f"  Stream: {stream} | Thinking: {enable_thinking} | Max iters: {max_iterations}")
    print(f"{'='*70}")

    start_mem = get_memory_gb()
    start_time = time.time()

    for iteration in range(max_iterations):
        iter_start = time.time()

        resp = send_request(
            messages,
            stream=stream,
            max_tokens=800,
            enable_thinking=enable_thinking,
        )

        msg = resp.get("choices", [{}])[0].get("message", {})
        usage = resp.get("usage", {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        tool_calls = msg.get("tool_calls") or []

        iter_time = time.time() - iter_start
        cur_mem = get_memory_gb()

        # Summary line
        tc_names = [tc["function"]["name"] for tc in tool_calls] if tool_calls else []
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)

        print(
            f"  Iter {iteration+1}: {pt}pt/{ct}ct | {iter_time:.1f}s | "
            f"Mem: {cur_mem:.1f}GB | "
            f"Tools: {tc_names if tc_names else 'none'} | "
            f"Content: {content[:60].strip()!r}{'...' if len(content) > 60 else ''}"
        )

        if reasoning:
            print(f"    Reasoning: {reasoning[:80].strip()!r}...")

        if not tool_calls:
            print(f"  -> No more tool calls, loop complete after {iteration+1} iterations")
            break

        # Add assistant message with tool calls
        assistant_msg = {"role": "assistant", "content": content or None}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        # Execute each tool and add results
        for tc in tool_calls:
            result = execute_tool(tc)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{iteration}"),
                    "name": tc["function"]["name"],
                    "content": result,
                }
            )

    end_mem = get_memory_gb()
    total_time = time.time() - start_time
    mem_delta = end_mem - start_mem

    print(f"  SUMMARY: {total_time:.1f}s total | Mem: {start_mem:.1f} -> {end_mem:.1f} GB (delta: {mem_delta:+.2f}GB)")
    return {
        "start_mem": start_mem,
        "end_mem": end_mem,
        "mem_delta": mem_delta,
        "total_time": total_time,
        "iterations": iteration + 1,
    }


def run_concurrent_tool_loops(n_concurrent=2, stream=True):
    """Run multiple agentic loops concurrently."""
    tasks = [
        ("What's the weather in Tokyo, London, and New York? Compare them.", "weather-compare"),
        ("Find Python files, read the main one, and calculate 2**10 + 3**5.", "file-calc"),
        ("Get weather in Paris, search for *.txt files, and read one of them.", "multi-tool"),
    ]

    print(f"\n{'#'*70}")
    print(f"CONCURRENT TOOL LOOPS: {n_concurrent} parallel (stream={stream})")
    print(f"{'#'*70}")

    start_mem = get_memory_gb()
    results = []

    with ThreadPoolExecutor(max_workers=n_concurrent) as executor:
        futures = {}
        for i in range(n_concurrent):
            task_prompt, label = tasks[i % len(tasks)]
            future = executor.submit(
                run_agentic_loop,
                task_prompt,
                max_iterations=4,
                stream=stream,
                enable_thinking=False,
                label=f"concurrent-{i+1}-{label}",
            )
            futures[future] = label

        for future in as_completed(futures):
            label = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"  ERROR in {label}: {e}")
                results.append({"error": str(e)})

    end_mem = get_memory_gb()
    print(f"\nCONCURRENT SUMMARY: Mem: {start_mem:.1f} -> {end_mem:.1f} GB (delta: {end_mem - start_mem:+.2f}GB)")
    return results


def main():
    print("=" * 70)
    print("122B MODEL TOOL CALL STRESS TEST")
    print(f"API: {API_URL}")
    print(f"PID: {PID}")
    print("=" * 70)

    initial_mem = get_memory_gb()
    initial_cpu = get_cpu_percent()
    print(f"\nInitial state: {initial_mem:.1f}GB RSS | {initial_cpu:.1f}% CPU")

    all_results = []

    # Test 1: Single streaming agentic loop (weather comparison)
    r = run_agentic_loop(
        "What's the weather in San Francisco, Tokyo, and Berlin? "
        "Compare the temperatures and tell me which is warmest.",
        max_iterations=5,
        stream=True,
        enable_thinking=False,
        label="1-stream-weather",
    )
    all_results.append(("stream-weather", r))

    # Test 2: Single non-streaming agentic loop (file operations)
    r = run_agentic_loop(
        "Search for Python files in the current directory, then read the first one. "
        "After reading it, calculate the number of lines times 42.",
        max_iterations=5,
        stream=False,
        enable_thinking=False,
        label="2-nostream-files",
    )
    all_results.append(("nostream-files", r))

    # Test 3: Streaming with thinking enabled
    r = run_agentic_loop(
        "I need you to check the weather in 3 cities and do some math. "
        "Get weather for NYC, LA, and Chicago. Then calculate the average temperature.",
        max_iterations=5,
        stream=True,
        enable_thinking=True,
        label="3-stream-thinking-tools",
    )
    all_results.append(("stream-thinking-tools", r))

    # Test 4: Long multi-tool chain
    r = run_agentic_loop(
        "Step by step: 1) Search for *.py files, 2) Read the first result, "
        "3) Get weather in the location mentioned in the file (or default to Seattle), "
        "4) Calculate 2**20. Do each step one at a time.",
        max_iterations=6,
        stream=True,
        enable_thinking=False,
        label="4-long-chain",
    )
    all_results.append(("long-chain", r))

    # Test 5: Concurrent tool loops (2 parallel)
    concurrent_results = run_concurrent_tool_loops(n_concurrent=2, stream=True)
    all_results.append(("concurrent-2", concurrent_results))

    # Test 6: Concurrent tool loops (3 parallel, non-streaming)
    concurrent_results = run_concurrent_tool_loops(n_concurrent=3, stream=False)
    all_results.append(("concurrent-3-nostream", concurrent_results))

    # Test 7: Rapid sequential requests (no tool calls, just throughput)
    print(f"\n{'='*70}")
    print("RAPID SEQUENTIAL: 10 quick requests")
    print(f"{'='*70}")
    rapid_start = get_memory_gb()
    for i in range(10):
        resp = send_request(
            [{"role": "user", "content": f"What is {i*7+3}? One word answer."}],
            stream=True,
            max_tokens=50,
        )
        content = resp["choices"][0]["message"].get("content", "")[:40]
        print(f"  Quick {i+1}: {content.strip()!r}")
    rapid_end = get_memory_gb()
    print(f"  Rapid mem: {rapid_start:.1f} -> {rapid_end:.1f} GB (delta: {rapid_end - rapid_start:+.2f}GB)")

    # Final summary
    final_mem = get_memory_gb()
    final_cpu = get_cpu_percent()
    total_delta = final_mem - initial_mem

    print(f"\n{'='*70}")
    print("FINAL STRESS TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Initial: {initial_mem:.1f}GB | Final: {final_mem:.1f}GB | Delta: {total_delta:+.2f}GB")
    print(f"CPU: {final_cpu:.1f}%")

    if total_delta > 2.0:
        print("WARNING: Memory grew more than 2GB — possible leak!")
    elif total_delta > 0.5:
        print("NOTE: Minor memory growth detected — may be normal cache expansion")
    else:
        print("OK: Memory stable — no leaks detected")

    # Per-test summary
    print("\nPer-test results:")
    for name, result in all_results:
        if isinstance(result, dict) and "mem_delta" in result:
            print(f"  {name}: {result['mem_delta']:+.2f}GB in {result.get('total_time', 0):.1f}s ({result.get('iterations', '?')} iters)")
        elif isinstance(result, list):
            for i, r in enumerate(result):
                if isinstance(r, dict) and "mem_delta" in r:
                    print(f"  {name}[{i}]: {r['mem_delta']:+.2f}GB in {r.get('total_time', 0):.1f}s")


if __name__ == "__main__":
    main()
