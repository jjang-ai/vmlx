#!/usr/bin/env python3
"""PLD throughput comparison: PLD off vs PLD on.

Sends identical prompts to two servers (PLD off / PLD on) and compares
generation tok/s. Prompts cover low/high PLD acceptance workloads.

Usage:
    python tests/benchmark/bench_pld_throughput.py --port-off 8080 --port-on 8081
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

PROMPTS = [
    ("short_factual", "Reply with the single word: ok", 5),
    ("repetitive", "Repeat exactly 5 times: AAA BBB CCC. AAA BBB CCC. AAA BBB CCC.", 50),
    ("code", "Write a Python function fibonacci(n) using recursion. Just the function, no explanation.", 100),
    ("json", 'Generate a JSON array of 3 product records, each with fields "id" (int), "name" (string), "price" (float). Just the JSON array.', 120),
    ("long_code", "Write a Python class LinkedList with methods: insert, delete, search, reverse, __len__, __repr__. Include docstrings.", 300),
]


def bench_prompt(port: int, prompt: str, max_tokens: int) -> dict:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    usage = body.get("usage", {})
    comp_tokens = usage.get("completion_tokens", 0)
    tps = comp_tokens / elapsed if elapsed > 0 else 0
    return {"tokens": comp_tokens, "time": elapsed, "tps": tps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port-off", type=int, required=True)
    parser.add_argument("--port-on", type=int, required=True)
    parser.add_argument("--warmup", action="store_true", help="Send warmup request first")
    args = parser.parse_args()

    if args.warmup:
        for port in [args.port_off, args.port_on]:
            bench_prompt(port, "Say hi", 5)

    print(f"{'Prompt':<15} {'PLD off tok/s':>14} {'PLD on tok/s':>13} {'Δ%':>7} {'tokens':>7}")
    print("-" * 60)

    total_off = total_on = 0
    total_tok_off = total_tok_on = 0
    results = []

    for label, prompt, max_tokens in PROMPTS:
        off = bench_prompt(args.port_off, prompt, max_tokens)
        on = bench_prompt(args.port_on, prompt, max_tokens)
        delta = ((on["tps"] / off["tps"]) - 1) * 100 if off["tps"] > 0 else 0
        sign = "+" if delta >= 0 else ""
        print(f"{label:<15} {off['tps']:>11.1f}    {on['tps']:>10.1f}   {sign}{delta:>5.1f}%  {on['tokens']:>5}")
        total_off += off["time"]
        total_on += on["time"]
        total_tok_off += off["tokens"]
        total_tok_on += on["tokens"]
        results.append((label, off, on, delta))

    agg_off = total_tok_off / total_off if total_off > 0 else 0
    agg_on = total_tok_on / total_on if total_on > 0 else 0
    agg_delta = ((agg_on / agg_off) - 1) * 100 if agg_off > 0 else 0
    sign = "+" if agg_delta >= 0 else ""
    print("-" * 60)
    print(f"{'AGGREGATE':<15} {agg_off:>11.1f}    {agg_on:>10.1f}   {sign}{agg_delta:>5.1f}%  {total_tok_on:>5}")

    # Health telemetry
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{args.port_on}/health", timeout=5) as r:
            h = json.loads(r.read())
        spec = h.get("speculative_decoding", {})
        if isinstance(spec, dict):
            batched = spec.get("batched", {})
            print(f"\nPLD telemetry: steps={batched.get('steps')}, "
                  f"acceptance={batched.get('acceptance_rate', 0):.2%}, "
                  f"tokens_emitted={batched.get('tokens_emitted')}")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
