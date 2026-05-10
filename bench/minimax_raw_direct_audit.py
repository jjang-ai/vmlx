#!/usr/bin/env python3
"""Raw MiniMax JANGTQ decode audit using the vMLX loader and prompt policy.

This bypasses the HTTP server, SimpleEngine, Scheduler, BatchGenerator, prefix
cache, paged cache, and block-L2. It is meant to answer one narrow question:

    How fast is the current MiniMax runtime itself on the same prompt shape?

Use this row to compare against ``bench/minimax_counting_audit.py``. If raw
direct, SimpleEngine API, and full-stack BatchedEngine all cluster around the
same tok/s, the remaining speed gap is not a scheduler/settings issue.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from vmlx_engine.utils.chat_template_kwargs import (
    build_chat_template_kwargs,
    ensure_thinking_off_sentinel,
)
from vmlx_engine.utils.tokenizer import load_model_with_fallback


DEFAULT_TURNS = [
    "List five everyday uses of large language models. Number them.",
    "For item 3, give a concrete example with a specific user persona.",
    "Now restate item 3 in one sentence under 25 words.",
    "Translate that one sentence into Spanish.",
    "Finally, give the Spanish version with each word's English gloss in parens.",
]


def _render_prompt(tokenizer: Any, messages: list[dict[str, str]], model_path: str) -> str:
    template_kwargs = build_chat_template_kwargs(
        enable_thinking=False,
        tokenize=False,
        add_generation_prompt=True,
    )
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    else:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        prompt += "\nassistant:"
    return ensure_thinking_off_sentinel(
        str(prompt),
        family_name="minimax",
        model_name=Path(model_path).name,
    )


def _generate_one(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    sampler = make_sampler(temp=temperature, top_p=1.0)
    t_start = time.perf_counter()
    t_first: float | None = None
    t_last: float | None = None
    pieces: list[str] = []
    token_ids: list[int] = []
    finish_reason = None

    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        now = time.perf_counter()
        if t_first is None:
            t_first = now
        t_last = now
        token_ids.append(int(response.token))
        pieces.append(response.text or "")
        finish_reason = response.finish_reason or finish_reason

    t_end = time.perf_counter()
    if t_first is None:
        t_first = t_end
    if t_last is None:
        t_last = t_first
    visible_window = max(t_last - t_first, 1e-9)
    stream_closed = max(t_end - t_first, 1e-9)
    text = "".join(pieces)
    return {
        "prompt_tokens": len(tokenizer.encode(prompt)),
        "generated_tokens": len(token_ids),
        "ttft_ms": (t_first - t_start) * 1000.0,
        "tail_ms": (t_end - t_last) * 1000.0,
        "decode_s_visible_window": visible_window,
        "decode_s_stream_closed": stream_closed,
        "tps_visible_window": len(token_ids) / visible_window,
        "tps_stream_closed": len(token_ids) / stream_closed,
        "finish_reason": finish_reason,
        "text_chars": len(text),
        "sample": text.replace("\n", " ")[:240],
        "token_ids_tail": token_ids[-8:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--artifact", default="")
    args = parser.parse_args()

    load_t0 = time.perf_counter()
    model, tokenizer = load_model_with_fallback(args.model_path)
    load_s = time.perf_counter() - load_t0

    messages: list[dict[str, str]] = []
    turns: list[dict[str, Any]] = []
    for idx, user_text in enumerate(DEFAULT_TURNS[: args.turns], 1):
        messages.append({"role": "user", "content": user_text})
        prompt = _render_prompt(tokenizer, messages, args.model_path)
        row = _generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        row["turn"] = idx
        turns.append(row)
        messages.append({"role": "assistant", "content": row["sample"]})

    def _mean(key: str, rows: list[dict[str, Any]]) -> float:
        return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0

    warm = turns[1:] if len(turns) > 1 else turns
    report = {
        "model_path": args.model_path,
        "load_seconds": load_s,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "turns": turns,
        "mean": {
            "tps_visible_window": _mean("tps_visible_window", turns),
            "tps_stream_closed": _mean("tps_stream_closed", turns),
            "ttft_ms": _mean("ttft_ms", turns),
        },
        "warm_mean": {
            "tps_visible_window": _mean("tps_visible_window", warm),
            "tps_stream_closed": _mean("tps_stream_closed", warm),
            "ttft_ms": _mean("ttft_ms", warm),
        },
    }

    print(
        "turn  ttft_ms  tail_ms  gen_tokens  tps_visible  tps_closed  prompt_tokens"
    )
    for row in turns:
        print(
            f"{row['turn']:>4}  {row['ttft_ms']:>7.1f}  {row['tail_ms']:>7.1f}  "
            f"{row['generated_tokens']:>10}  {row['tps_visible_window']:>11.2f}  "
            f"{row['tps_stream_closed']:>10.2f}  {row['prompt_tokens']:>13}"
        )
    print("\nmean:")
    print(f"  tps_visible_window={report['mean']['tps_visible_window']:.2f}")
    print(f"  tps_stream_closed={report['mean']['tps_stream_closed']:.2f}")
    print(f"  warm_tps_visible_window={report['warm_mean']['tps_visible_window']:.2f}")
    print(f"  warm_tps_stream_closed={report['warm_mean']['tps_stream_closed']:.2f}")
    print(f"  load_seconds={load_s:.1f}")

    if args.artifact:
        out = Path(args.artifact)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
