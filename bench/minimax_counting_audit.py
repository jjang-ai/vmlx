#!/usr/bin/env python3
"""MiniMax full-stack streaming counter audit.

This is an HTTP-only probe. It does not load weights. Point it at a running
vmlx-engine server that was launched with the desired full stack, for example:

    vmlx-engine serve /Users/example/models/JANGQ/MiniMax-M2.7-JANGTQ \
      --port 8092 --host 127.0.0.1 \
      --continuous-batching --max-num-seqs 5 \
      --prefill-batch-size 1024 --completion-batch-size 1024 \
      --use-paged-cache --enable-prefix-cache --enable-block-disk-cache \
      --served-model-name minimax

The goal is to separate measurement accounting from real decode speed:

* visible_chunk_tps: counts streamed content/reasoning chunks.
* usage_tps_from_first_delta: counts server usage.completion_tokens from the
  first streamed visible/reasoning delta. This matches the optimistic legacy
  harness style.
* usage_tps_from_first_sse: counts usage.completion_tokens from the first SSE
  frame, catching streams that emit non-visible frames before user-visible text.
* usage_tps_from_send: includes TTFT; useful for end-to-end cache impact.

If usage_tps_from_first_delta is much higher than visible_chunk_tps, the old
"tok/s" number is likely an accounting artifact. If they agree, the high
number is probably real.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator


DEFAULT_TURNS = [
    "List five everyday uses of large language models. Number them.",
    "For item 3, give a concrete example with a specific user persona.",
    "Now restate item 3 in one sentence under 25 words.",
    "Translate that one sentence into Spanish.",
    "Finally, give the Spanish version with each word's English gloss in parens.",
]


@dataclass
class TurnAudit:
    turn: int
    ttft_first_sse_ms: float
    ttft_first_delta_ms: float
    tail_after_last_delta_ms: float
    visible_chunks: int
    usage_completion_tokens: int
    prompt_tokens: int
    cached_tokens: int
    cache_detail: str
    decode_s_visible_window: float
    decode_s_from_first_delta: float
    decode_s_from_first_sse: float
    elapsed_s_from_send: float
    visible_chunk_tps_visible_window: float
    usage_tps_visible_window: float
    visible_chunk_tps: float
    usage_tps_from_first_delta: float
    usage_tps_from_first_sse: float
    usage_tps_from_send: float
    text_chars: int
    sample: str


def _http_json(url: str, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        return {"_error": f"{type(exc).__name__}: {exc}"}


def _sse_iter(resp) -> Iterator[dict[str, Any]]:
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            return
        try:
            yield json.loads(body)
        except json.JSONDecodeError:
            continue


def _post_stream(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
) -> tuple[TurnAudit, str]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t_send = time.perf_counter()
    t_first_sse: float | None = None
    t_first_delta: float | None = None
    t_last_delta: float | None = None
    visible_chunks = 0
    text_parts: list[str] = []
    usage: dict[str, Any] = {}

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for chunk in _sse_iter(resp):
            now = time.perf_counter()
            if t_first_sse is None:
                t_first_sse = now

            chunk_usage = chunk.get("usage")
            if isinstance(chunk_usage, dict):
                usage = chunk_usage

            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content") or delta.get("reasoning_content")
            if piece:
                if t_first_delta is None:
                    t_first_delta = now
                t_last_delta = now
                visible_chunks += 1
                text_parts.append(str(piece))

    t_end = time.perf_counter()
    if t_first_sse is None:
        t_first_sse = t_end
    if t_first_delta is None:
        t_first_delta = t_end
    if t_last_delta is None:
        t_last_delta = t_first_delta

    usage_completion = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached_tokens = int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0
    cache_detail = ""
    if isinstance(details, dict) and isinstance(details.get("cache_detail"), str):
        cache_detail = details["cache_detail"]

    text = "".join(text_parts)
    visible_window = max(t_last_delta - t_first_delta, 1e-9)
    from_delta = max(t_end - t_first_delta, 1e-9)
    from_sse = max(t_end - t_first_sse, 1e-9)
    from_send = max(t_end - t_send, 1e-9)
    audit = TurnAudit(
        turn=-1,
        ttft_first_sse_ms=(t_first_sse - t_send) * 1000.0,
        ttft_first_delta_ms=(t_first_delta - t_send) * 1000.0,
        tail_after_last_delta_ms=(t_end - t_last_delta) * 1000.0,
        visible_chunks=visible_chunks,
        usage_completion_tokens=usage_completion,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        cache_detail=cache_detail,
        decode_s_visible_window=visible_window,
        decode_s_from_first_delta=from_delta,
        decode_s_from_first_sse=from_sse,
        elapsed_s_from_send=from_send,
        visible_chunk_tps_visible_window=visible_chunks / visible_window,
        usage_tps_visible_window=usage_completion / visible_window if usage_completion else 0.0,
        visible_chunk_tps=visible_chunks / from_delta,
        usage_tps_from_first_delta=usage_completion / from_delta if usage_completion else 0.0,
        usage_tps_from_first_sse=usage_completion / from_sse if usage_completion else 0.0,
        usage_tps_from_send=usage_completion / from_send if usage_completion else 0.0,
        text_chars=len(text),
        sample=text.replace("\n", " ")[:220],
    )
    return audit, text


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write_artifact(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8092")
    parser.add_argument("--model", default="minimax")
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--artifact-dir", default="")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    health_before = _http_json(f"{base_url}/health", timeout=5.0)
    stats_before = _http_json(f"{base_url}/v1/cache/stats", timeout=5.0)

    messages: list[dict[str, str]] = []
    audits: list[TurnAudit] = []
    for i, prompt in enumerate(DEFAULT_TURNS[: args.turns], 1):
        messages.append({"role": "user", "content": prompt})
        audit, assistant_text = _post_stream(
            base_url=base_url,
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        audit.turn = i
        audits.append(audit)
        messages.append({"role": "assistant", "content": assistant_text or "(empty)"})

    health_after = _http_json(f"{base_url}/health", timeout=5.0)
    stats_after = _http_json(f"{base_url}/v1/cache/stats", timeout=5.0)

    print(
        "turn  ttft_delta_ms  tail_ms  visible_chunks  usage_tokens  "
        "chunk_tps_visible  chunk_tps_closed  usage_tps_closed  cached/prompt  cache_detail"
    )
    for a in audits:
        print(
            f"{a.turn:>4}  {a.ttft_first_delta_ms:>13.1f}  "
            f"{a.tail_after_last_delta_ms:>7.1f}  "
            f"{a.visible_chunks:>14}  {a.usage_completion_tokens:>12}  "
            f"{a.visible_chunk_tps_visible_window:>17.2f}  "
            f"{a.visible_chunk_tps:>16.2f}  "
            f"{a.usage_tps_from_first_delta:>16.2f}  "
            f"{a.cached_tokens:>5}/{a.prompt_tokens:<5}  {a.cache_detail or '-'}"
        )

    warm = audits[1:] if len(audits) > 1 else audits
    print("\nmean:")
    print(
        "  visible_chunk_tps_visible_window="
        f"{_mean([a.visible_chunk_tps_visible_window for a in audits]):.2f}"
    )
    print(f"  visible_chunk_tps_stream_closed={_mean([a.visible_chunk_tps for a in audits]):.2f}")
    print(
        "  usage_tps_from_first_delta_to_close="
        f"{_mean([a.usage_tps_from_first_delta for a in audits]):.2f}"
    )
    print(
        "  warm_visible_chunk_tps_visible_window="
        f"{_mean([a.visible_chunk_tps_visible_window for a in warm]):.2f}"
    )
    print(
        "  warm_visible_chunk_tps_stream_closed="
        f"{_mean([a.visible_chunk_tps for a in warm]):.2f}"
    )
    print(
        "  warm_usage_tps_from_first_delta_to_close="
        f"{_mean([a.usage_tps_from_first_delta for a in warm]):.2f}"
    )
    print("\ninterpretation:")
    print("  visible_window is first visible delta through last visible delta.")
    print("  stream_closed includes any final usage/cache-store tail before HTTP closes.")
    print("  If closed tps is much lower than visible_window tps, investigate finalization/cache store.")
    print("  If both are low with cache_detail populated, the gap is in full-stack cache/decode.")

    if args.artifact_dir:
        out_dir = Path(args.artifact_dir)
        _write_artifact(out_dir / "health_before.json", health_before)
        _write_artifact(out_dir / "cache_stats_before.json", stats_before)
        _write_artifact(out_dir / "health_after.json", health_after)
        _write_artifact(out_dir / "cache_stats_after.json", stats_after)
        _write_artifact(
            out_dir / "turns.json",
            {
                "base_url": base_url,
                "model": args.model,
                "turns": [asdict(a) for a in audits],
            },
        )
        print(f"\nartifacts: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
