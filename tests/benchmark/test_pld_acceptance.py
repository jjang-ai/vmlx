#!/usr/bin/env python3
"""
Prompt Lookup Decoding — acceptance rate benchmark.

Sends workloads designed to stress-test PLD hit rates:
  1. Code generation  — high repetition, should yield high hit rate
  2. Tool call JSON   — structured, predictable continuations
  3. Summarisation    — reads back prompt text, moderate hit rate
  4. Open-ended chat  — low repetition, expected low hit rate

The server must be started with the target model BEFORE running this.
PLD stats are emitted to the server log every 200 tokens — grep for "[PLD]".

Usage:
    python tests/benchmark/test_pld_acceptance.py [--port 8080]

Output:
    Per-task token rate and a reminder to check the server log for PLD stats.
"""

import argparse
import json
import sys
import time
import urllib.request
from typing import Iterator, Optional


def stream_completion(
    port: int,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> Iterator[str]:
    """Stream tokens from the server via chat/completions, yield text chunks."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
                delta = chunk["choices"][0].get("delta", {})
                text = delta.get("content") or delta.get("reasoning_content") or ""
                if text:
                    yield text
            except (json.JSONDecodeError, KeyError, IndexError):
                continue



def run_task(
    port: int,
    name: str,
    prompt: str,
    max_tokens: int = 400,
    temperature: float = 0.0,
    save_dir: Optional[str] = None,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Task: {name}")
    print(f"Prompt length: {len(prompt)} chars")
    print("Generating", end="", flush=True)

    t0 = time.monotonic()
    sse_events = 0
    text = ""
    for chunk in stream_completion(port, prompt, max_tokens=max_tokens,
                                   temperature=temperature):
        text += chunk
        sse_events += 1
        if sse_events % 10 == 0:
            print(".", end="", flush=True)

    elapsed = time.monotonic() - t0

    # Count actual tokens from the text length heuristic (≈4 chars/tok) as a
    # fallback; server logs are the authoritative source for exact counts.
    # With PLD Phase 2, one SSE event often carries multiple tokens, so
    # sse_events is not a useful token count.
    approx_tokens = max(sse_events, len(text) // 4)
    rate = approx_tokens / elapsed if elapsed > 0 else 0

    print(f"\n  sse_events={sse_events}  approx_tokens≈{approx_tokens}  "
          f"time={elapsed:.1f}s  rate≈{rate:.0f} tok/s")
    print(f"  output preview: {text[:120].replace(chr(10), ' ')}...")

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        slug = name.lower().replace(" ", "_")
        path = os.path.join(save_dir, f"{slug}.txt")
        with open(path, "w") as f:
            f.write(text)
        print(f"  saved → {path}")

    return {"name": name, "sse_events": sse_events, "approx_tokens": approx_tokens,
            "elapsed": elapsed, "rate": rate, "text": text}


def run_partial_accept_stress_task(port: int, temperature: float = 0.0,
                                   save_dir: Optional[str] = None) -> dict:
    """Task designed to trigger hybrid SSM partial-accept replay (issue #134).

    Uses high-repetition prompts where n-gram matches are very likely but the
    model is likely to accept some (not all) draft tokens on hybrid SSM/ATT
    models.  The repeating pattern gives PLD many K=2 candidates; the midpoint
    mismatch potential forces partial-accept paths.

    On hybrid models, check /health?pld_ssm_replay.attempts > 0 after this task.
    """
    # Prompt engineered for partial-accept: exact repetition at start,
    # then forced semantic divergence in the middle of the repeated unit.
    # K=2 means d0 often accepted (exact n-gram), d1 sometimes rejected (model
    # diverges at midpoint of the repeated unit).
    repeat_unit = (
        "The function returns a list of tuples. "
        "Each tuple contains an integer index and a string value. "
    )
    partial_prompt = (
        repeat_unit * 6
        + "\n\nContinue the pattern:\n"
        + repeat_unit * 2
        + "Each tuple contains an integer index and "
    )
    return run_task(
        port,
        "Partial-accept stress (hybrid SSM replay)",
        partial_prompt,
        max_tokens=400,
        temperature=temperature,
        save_dir=save_dir,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default 0.0 = greedy)")
    parser.add_argument("--save-outputs", metavar="DIR",
                        help="Save each task's full output text to DIR for diffing")
    args = parser.parse_args()

    print(f"PLD Acceptance Rate Benchmark — target: port {args.port}  "
          f"temperature={args.temperature}")
    if args.save_outputs:
        print(f"Saving outputs to: {args.save_outputs}")
    print("Reminder: watch server log for [PLD] lines, e.g.:")
    print("  vmlx-providers logs lg 200 | grep '\\[PLD\\]'")

    results = []

    # ── Task 1: Code generation ────────────────────────────────────────────
    # Long prompt with repeated patterns → high n-gram coverage expected
    code_prompt = """\
Write a Python class for a binary search tree with insert, search, and
in-order traversal methods. Include docstrings for each method.

```python
class BinarySearchTree:
    \"\"\"A simple binary search tree implementation.\"\"\"

    def __init__(self):
        self.root = None

    class Node:
        def __init__(self, value):
            self.value = value
            self.left = None
            self.right = None

    def insert(self, value):
"""
    results.append(run_task(args.port, "Code generation", code_prompt, max_tokens=500,
                            temperature=args.temperature, save_dir=args.save_outputs))

    # ── Task 2: Structured JSON / tool call ───────────────────────────────
    # Structured output with repeated keys → high hit rate expected
    json_prompt = """\
Generate a JSON array of 10 product records. Each record must have these
exact fields: "id" (integer), "name" (string), "price" (float), "in_stock"
(boolean), "category" (string), "tags" (array of strings).

Example record:
{"id": 1, "name": "Widget A", "price": 9.99, "in_stock": true, "category": "tools", "tags": ["sale", "new"]}

Full array:
[
{"id": 1, "name": """
    results.append(run_task(args.port, "Structured JSON", json_prompt, max_tokens=600,
                            temperature=args.temperature, save_dir=args.save_outputs))

    # ── Task 3: Summarisation (reads back prompt tokens) ──────────────────
    article = (
        "The quick brown fox jumps over the lazy dog. " * 8
        + "Scientists at the university discovered that regular exercise "
        "improves cognitive function significantly. The study, conducted "
        "over three years with 500 participants, found that just 30 minutes "
        "of aerobic exercise three times per week led to measurable "
        "improvements in memory, attention, and executive function. "
        "Participants who exercised regularly also reported better sleep "
        "quality and reduced stress levels compared to the control group. "
        "The researchers plan to expand the study to include older adults "
        "and individuals with pre-existing health conditions. "
    )
    summary_prompt = f"{article}\n\nSummarize the above passage in three sentences:\n"
    results.append(run_task(args.port, "Summarisation", summary_prompt, max_tokens=200,
                            temperature=args.temperature, save_dir=args.save_outputs))

    # ── Task 4: Open-ended chat ────────────────────────────────────────────
    chat_prompt = """\
<|im_start|>user
Explain the philosophical implications of the Ship of Theseus paradox for
modern identity theory, touching on mereological essentialism and four-dimensionalism.
<|im_end|>
<|im_start|>assistant
"""
    results.append(run_task(args.port, "Open-ended reasoning", chat_prompt, max_tokens=400,
                            temperature=args.temperature, save_dir=args.save_outputs))

    # ── Task 5: Partial-accept stress (hybrid SSM replay — issue #134) ────
    results.append(run_partial_accept_stress_task(
        args.port, temperature=args.temperature, save_dir=args.save_outputs
    ))

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Benchmark complete. Approximate token rates by task:")
    print("(approx_tokens = max(sse_events, len(text)//4); exact counts in server log)")
    for r in results:
        print(f"  {r['name']:40s}  {r['rate']:6.0f} tok/s  "
              f"(≈{r['approx_tokens']} tokens, {r['sse_events']} SSE events, "
              f"{r['elapsed']:.1f}s)")

    total_approx = sum(r["approx_tokens"] for r in results)
    total_elapsed = sum(r["elapsed"] for r in results)
    print(f"\nTotal ≈{total_approx} tokens in {total_elapsed:.1f}s  "
          f"(≈{total_approx/total_elapsed:.0f} tok/s overall)")
    print("\nFor exact token counts check server log:")
    print(f"  vmlx-providers logs lg 500 | grep 'finished:'")
    print("\nFor PLD speculative decode stats:")
    print(f"  vmlx-providers logs lg 500 | grep 'PLD-spec' | awk -F'=' '{{print $2}}' | sort | uniq -c")
    print("\nFor hybrid SSM replay stats (issue #134):")
    print(f"  curl -s http://127.0.0.1:{args.port}/health | python3 -m json.tool | grep -A5 pld_ssm_replay")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
    except OSError as e:
        print(f"\nError connecting to server: {e}")
        print("Is the vmlx server running? Try: vmlx-providers start lg")
        sys.exit(1)
