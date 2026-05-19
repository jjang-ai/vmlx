#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


PROMPTS: list[dict[str, Any]] = [
    {
        "label": "Short generation",
        "messages": [{"role": "user", "content": "Write a haiku about silicon."}],
        "max_tokens": 64,
    },
    {
        "label": "Medium generation",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Explain how a transformer neural network processes a sentence, "
                    "step by step."
                ),
            }
        ],
        "max_tokens": 256,
    },
    {
        "label": "Long generation",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a detailed technical blog post about the advantages and "
                    "challenges of running large language models on Apple Silicon. "
                    "Cover memory bandwidth, unified memory architecture, and the "
                    "role of quantization."
                ),
            }
        ],
        "max_tokens": 512,
    },
    {
        "label": "Long prompt / prefill test",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes text concisely.",
            },
            {
                "role": "user",
                "content": (
                    "Summarize the following passage in 2 sentences:\n\n"
                    + (
                        "The development of artificial intelligence has progressed "
                        "through several distinct phases. In the early days, "
                        "researchers focused on symbolic AI, attempting to encode "
                        "human knowledge into explicit rules and logical frameworks. "
                        "This approach showed promise in narrow domains but struggled "
                        "with the complexity and ambiguity of real-world problems. "
                        "The emergence of machine learning shifted the paradigm, "
                        "allowing systems to learn patterns from data rather than "
                        "following hand-crafted rules. Deep learning, powered by "
                        "neural networks with many layers, further revolutionized the "
                        "field by enabling automatic feature extraction from raw data. "
                        "The introduction of the transformer architecture in 2017 "
                        "marked another watershed moment, leading to large language "
                        "models that could generate coherent text, translate "
                        "languages, and answer questions with unprecedented accuracy. "
                        "Today, the focus has shifted to making these models more "
                        "efficient, more aligned with human values, and more "
                        "accessible to a broader range of users and applications. "
                    )
                    * 3
                ),
            },
        ],
        "max_tokens": 128,
    },
]


def stream_chat(base_url: str, model: str, prompt: dict[str, Any]) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": prompt["messages"],
        "max_tokens": prompt["max_tokens"],
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    first_token_time: float | None = None
    completion_tokens = 0
    prompt_tokens = 0
    text_chars = 0
    raw_lines = 0
    last_usage: dict[str, Any] | None = None

    with urllib.request.urlopen(req, timeout=300) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                continue
            raw_lines += 1
            parsed = json.loads(payload)
            usage = parsed.get("usage")
            if usage:
                last_usage = usage
                prompt_tokens = usage.get("prompt_tokens") or prompt_tokens
                completion_tokens = (
                    usage.get("completion_tokens")
                    if usage.get("completion_tokens") is not None
                    else completion_tokens
                )
            delta = ((parsed.get("choices") or [{}])[0].get("delta") or {})
            token_text = delta.get("content") or delta.get("reasoning_content") or ""
            if token_text:
                text_chars += len(token_text)
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                if not usage:
                    completion_tokens += 1

    end = time.perf_counter()
    total_time = end - start
    ttft = (first_token_time - start) if first_token_time is not None else total_time
    generation_time = (end - first_token_time) if first_token_time is not None else total_time
    tps = completion_tokens / generation_time if generation_time > 0.01 else 0.0
    pp_speed = prompt_tokens / ttft if ttft > 0.001 and prompt_tokens else 0.0
    return {
        "label": prompt["label"],
        "ttft_sec": ttft,
        "tps": tps,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_time_sec": total_time,
        "pp_speed": pp_speed,
        "text_chars": text_chars,
        "raw_sse_events": raw_lines,
        "usage": last_usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("model")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = [stream_chat(args.base_url, args.model, prompt) for prompt in PROMPTS]
    avg_tps = sum(row["tps"] for row in rows) / len(rows)
    avg_ttft = sum(row["ttft_sec"] for row in rows) / len(rows)
    total_time = sum(row["total_time_sec"] for row in rows)
    result = {
        "base_url": args.base_url,
        "model": args.model,
        "avg_tps": avg_tps,
        "avg_ttft_sec": avg_ttft,
        "total_time_sec": total_time,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
