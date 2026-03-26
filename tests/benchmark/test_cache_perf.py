#!/usr/bin/env python3
"""Cache performance test for hybrid SSM models (e.g. Qwen3.5-35B-A3B).

Measures time-to-first-token and throughput across a multi-turn conversation
to demonstrate block cache hit speedups on hybrid (KV + SSM) architectures.

Usage: python test_cache_perf.py [--base-url http://127.0.0.1:8081] [--turns 4]
"""
import argparse
import json
import time
import urllib.request


def chat(base_url, messages, max_tokens=64):
    """Send a streaming chat request. Returns a result dict."""
    body = json.dumps({
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    ttft = None
    content_tokens = 0
    reasoning_tokens = 0
    content_text = ""
    reasoning_text = ""
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    model_name = ""

    with urllib.request.urlopen(req) as resp:
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)

            if not model_name:
                model_name = chunk.get("model", "")

            choice = chunk["choices"][0] if chunk.get("choices") else {}
            delta = choice.get("delta", {})

            # Qwen3 reasoning models emit reasoning_content before content
            rc = delta.get("reasoning_content") or ""
            if rc:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                reasoning_tokens += 1
                reasoning_text += rc

            ct = delta.get("content") or ""
            if ct:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                content_tokens += 1
                content_text += ct

            # Usage in final chunk (requires stream_options.include_usage)
            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                details = usage.get("prompt_tokens_details") or {}
                cached_tokens = details.get("cached_tokens", 0)

    total = time.perf_counter() - t0

    # Fallback token counts from streaming
    if not completion_tokens:
        completion_tokens = content_tokens + reasoning_tokens

    return {
        "ttft": ttft or total,
        "total": total,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "content": content_text,
        "reasoning": reasoning_text,
        "model": model_name,
    }


def fmt_time(s):
    if s < 1:
        return f"{s*1000:.0f}ms"
    return f"{s:.2f}s"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--turns", type=int, default=4,
                        help="Number of conversation turns (default: 4)")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Max tokens per response (default: 64)")
    args = parser.parse_args()

    # System prompt: ~4K tokens — enough to show meaningful cache hits.
    # Real workloads (tool-use agents) are typically 30-60K tokens.
    system = (
        "You are a concise geography expert. "
        "Answer in exactly one sentence. "
        "Do not use unnecessary detail.\n\n"
        + "Context: " + (
            "The world has 195 countries, each with unique geography, culture, "
            "and history. Major cities serve as economic and cultural hubs. "
        ) * 150
    )

    questions = [
        "What is the capital of France?",
        "And what about Germany?",
        "Which of those two cities has a larger population?",
        "What river runs through the more populated one?",
        "Is that river longer than the Seine?",
        "What country does that river originate in?",
    ]

    # Detect model
    print("Connecting...", end=" ", flush=True)
    r0 = chat(args.base_url, [{"role": "user", "content": "hi"}], max_tokens=1)
    model = r0["model"] or "unknown"
    print(f"{model}\n")

    print("=" * 64)
    print(f"  Hybrid Model Block Cache — Performance Test")
    print(f"  Model: {model}")
    print(f"  Turns: {args.turns}  |  Max tokens/turn: {args.max_tokens}")
    print("=" * 64)

    messages = [{"role": "system", "content": system}]
    results = []

    for turn in range(min(args.turns, len(questions))):
        messages.append({"role": "user", "content": questions[turn]})

        label = "cold (no cache)" if turn == 0 else f"warm (cache hit)"
        print(f"\n  Turn {turn+1}: {questions[turn]}")
        print(f"  {'—'*56}")

        r = chat(args.base_url, messages, max_tokens=args.max_tokens)
        results.append(r)

        # Show answer (strip think blocks for display)
        answer = r["content"].strip() or "(thinking only)"
        if len(answer) > 100:
            answer = answer[:100] + "..."

        print(f"  Answer:    {answer}")
        print(f"  TTFT:      {fmt_time(r['ttft']):>8}   ({label})")
        print(f"  Total:     {fmt_time(r['total']):>8}")
        print(f"  Prompt:    {r['prompt_tokens']:>6} tokens"
              + (f"  (cached: {r['cached_tokens']})" if r['cached_tokens'] else ""))
        print(f"  Output:    {r['completion_tokens']:>6} tokens")

        # Append assistant response to conversation
        full_reply = r["content"]
        if r["reasoning"]:
            full_reply = f"<think>{r['reasoning']}</think>{r['content']}"
        messages.append({"role": "assistant", "content": full_reply})

    # Summary table
    print("\n" + "=" * 64)
    print("  Summary")
    print("=" * 64)
    print(f"  {'Turn':<6} {'TTFT':>8} {'Total':>8} {'Prompt':>8} {'Cached':>8} {'Speedup':>8}")
    print(f"  {'—'*6} {'—'*8} {'—'*8} {'—'*8} {'—'*8} {'—'*8}")

    cold_ttft = results[0]["ttft"] if results else 1
    for i, r in enumerate(results):
        speedup = f"{cold_ttft / r['ttft']:.1f}x" if i > 0 else "—"
        print(
            f"  {i+1:<6} {fmt_time(r['ttft']):>8} {fmt_time(r['total']):>8} "
            f"{r['prompt_tokens']:>8} {r['cached_tokens']:>8} {speedup:>8}"
        )

    print()


if __name__ == "__main__":
    main()
