#!/usr/bin/env python3
"""
PLD impact benchmark — OpenClaw workload simulation.

Sends prompts representative of OpenClaw's actual usage (coding tools, memory
search, reasoning) and measures latency and output quality with PLD on vs off.

The server must already be running. Run twice:
  # PLD on (default):
  python tests/benchmark/test_pld_openclaw.py --port 8080

  # PLD off (baseline):
  VMLX_PLD_DISABLED=1 vmlx-providers.sh restart lg
  python tests/benchmark/test_pld_openclaw.py --port 8080 --label baseline

Then compare saved outputs:
  diff outputs/pld_on/ outputs/baseline/

Usage:
  python tests/benchmark/test_pld_openclaw.py [--port 8080] [--label pld_on]
                                               [--save-dir outputs]
                                               [--temperature 0.3]
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterator, Optional


PROMPTS = [
    # --- Group A: Structured output (highest PLD risk at T=0.3) ---
    # These produce JSON/code directly without requiring tool infrastructure.
    {
        "group": "A",
        "name": "json_schema",
        "description": "Produce structured JSON (structured output, PLD risk)",
        "prompt": (
            "Return a JSON array of 5 tasks for a software engineer today. "
            "Each task must have: id (int), title (string), priority (high/medium/low), "
            "estimated_minutes (int). Output only the JSON array, no explanation."
        ),
        "max_tokens": 256,
    },
    {
        "group": "A",
        "name": "code_generation",
        "description": "Write Python code (structured output, PLD risk)",
        "prompt": (
            "Write a Python function that reverses a linked list. "
            "Include the Node class definition and a short example in __main__."
        ),
        "max_tokens": 512,
    },
    {
        "group": "A",
        "name": "code_review",
        "description": "Structured code review with section headers (PLD risk)",
        "prompt": (
            "Review this Python function and give feedback in exactly three sections: "
            "CORRECTNESS, PERFORMANCE, STYLE.\n\n"
            "def find_dupes(lst):\n"
            "    dupes = []\n"
            "    for i in range(len(lst)):\n"
            "        for j in range(i+1, len(lst)):\n"
            "            if lst[i] == lst[j] and lst[i] not in dupes:\n"
            "                dupes.append(lst[i])\n"
            "    return dupes"
        ),
        "max_tokens": 384,
    },
    # --- Group B: Reasoning / planning (moderate risk, longer output) ---
    {
        "group": "B",
        "name": "pr_checklist",
        "description": "Pre-PR checklist (reasoning, no tools)",
        "prompt": (
            "I'm about to open a PR for a big feature. "
            "What should I check before I do?"
        ),
        "max_tokens": 768,
    },
    {
        "group": "B",
        "name": "temperature_tradeoff",
        "description": "Technical explanation (reasoning, no tools)",
        "prompt": (
            "Explain the tradeoff between temperature 0 and temperature 0.3 "
            "for an LLM inference server."
        ),
        "max_tokens": 768,
    },
    # --- Group C: Short factual (latency baseline) ---
    # /no_think suppresses Qwen3 reasoning preamble so max_tokens isn't
    # consumed before visible output begins.
    {
        "group": "C",
        "name": "kv_vs_prefix_cache",
        "description": "Short factual (no tools)",
        "prompt": "What's the difference between a KV cache and a prefix cache?",
        "max_tokens": 512,
    },
    {
        "group": "C",
        "name": "last_topic",
        "description": "Memory recall — last topic (no tools)",
        "prompt": "What were we last working on together?",
        "max_tokens": 512,
    },
]


def stream_completion(
    port: int,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.3,
) -> Iterator[tuple[str, float]]:
    """
    Stream a chat completion. Yields (text_chunk, elapsed_seconds) tuples.
    First yield is the first token (use elapsed for TTFT).
    """
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    first_token = True

    with urllib.request.urlopen(req, timeout=180) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                data = json.loads(line[6:])
                delta = data.get("choices", [{}])[0].get("delta", {})
                # Capture both visible content and reasoning tokens.
                # Qwen3 thinking mode sends reasoning via reasoning_content;
                # both count toward tok/s and TTFT since PLD fires on both.
                chunk = delta.get("content", "") or delta.get("reasoning_content", "")
                if chunk:
                    elapsed = time.perf_counter() - t0
                    yield chunk, elapsed
                    first_token = False
            except json.JSONDecodeError:
                pass


def run_prompt(
    port: int,
    task: dict,
    temperature: float,
    save_dir: Optional[Path],
    label: str,
) -> dict:
    name = task["name"]
    group = task["group"]
    description = task["description"]

    print(f"\n{'=' * 60}")
    print(f"Group {group} — {name}")
    print(f"  {description}")
    print(f"  Prompt: {task['prompt'][:80]}{'...' if len(task['prompt']) > 80 else ''}")

    ttft = None
    total_time = None
    text = ""
    sse_events = 0

    try:
        for chunk, elapsed in stream_completion(
            port, task["prompt"], task["max_tokens"], temperature
        ):
            if ttft is None:
                ttft = elapsed
            text += chunk
            sse_events += 1
        total_time = elapsed if text else None
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "name": name,
            "group": group,
            "error": str(e),
        }

    if not text:
        print("  ERROR: empty response (no tokens received)")
        return {"name": name, "group": group, "error": "empty response"}

    tok_estimate = max(sse_events, len(text) // 4)
    rate = tok_estimate / total_time if total_time else 0

    print(f"  TTFT:  {ttft * 1000:.0f}ms")
    print(f"  Total: {total_time:.1f}s")
    print(f"  Tokens: ~{tok_estimate}  ({rate:.1f} tok/s)")
    print(f"  Output preview: {text[:120].strip()}...")

    if save_dir:
        out_path = save_dir / label / f"{name}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
        print(f"  Saved → {out_path}")

    return {
        "name": name,
        "group": group,
        "description": description,
        "ttft_ms": round(ttft * 1000) if ttft else None,
        "total_s": round(total_time, 2) if total_time else None,
        "tok_estimate": tok_estimate,
        "tok_per_s": round(rate, 1),
        "sse_events": sse_events,
        "output_len": len(text),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature (default: 0.3, matches OpenClaw)")
    parser.add_argument("--label", default="pld_on",
                        help="Label for this run, used as output subdirectory (default: pld_on)")
    parser.add_argument("--save-dir", default="tests/benchmark/outputs/openclaw",
                        help="Directory to save outputs for diffing")
    parser.add_argument("--groups", default="ABC",
                        help="Which groups to run: A, B, C or any combo (default: ABC)")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)

    print(f"PLD OpenClaw Benchmark — port={args.port}  temp={args.temperature}  label={args.label}")
    print(f"Saving outputs to: {save_dir / args.label}")
    print(f"Groups: {args.groups}")
    print()
    print("To compare runs:")
    print(f"  diff -r {save_dir}/pld_on {save_dir}/baseline")

    results = []
    for task in PROMPTS:
        if task["group"] not in args.groups:
            continue
        result = run_prompt(
            port=args.port,
            task=task,
            temperature=args.temperature,
            save_dir=save_dir,
            label=args.label,
        )
        results.append(result)

    # Summary table
    print(f"\n{'=' * 60}")
    print(f"Summary — label: {args.label}  (temp={args.temperature})")
    print(f"{'Task':<30} {'TTFT':>8} {'Total':>8} {'tok/s':>7}")
    print("-" * 60)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<30} {'ERROR':>8}")
            continue
        ttft = f"{r['ttft_ms']}ms" if r["ttft_ms"] else "n/a"
        total = f"{r['total_s']}s" if r["total_s"] else "n/a"
        print(f"{r['name']:<30} {ttft:>8} {total:>8} {r['tok_per_s']:>6.1f}")

    # Save JSON results
    json_path = save_dir / f"{args.label}_results.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {json_path}")
    print()
    print("For PLD stats: vmlx-providers.sh logs lg 500 | grep 'PLD-spec'")
    print("For token counts: vmlx-providers.sh logs lg 500 | grep 'finished:'")


if __name__ == "__main__":
    main()
