#!/usr/bin/env python3
"""Cache-safe streamed speed harness shared by vMLX and MTPLX servers."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

EXACT_TOKENIZER = None
SUSTAINED_TASK = False

FILLER = (
    "Apple silicon uses high-bandwidth unified memory for local inference. "
    "A fair coding benchmark separates prompt prefill, streamed decode, cache "
    "state, model residency, memory pressure, and speculative acceptance. "
)


def _command(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception:
        return None


def host_snapshot(server_pid: int | None) -> dict:
    rss_kib = None
    if server_pid:
        raw = _command("/bin/ps", "-o", "rss=", "-p", str(server_pid))
        try:
            rss_kib = int(raw or 0)
        except ValueError:
            rss_kib = None
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "server_pid": server_pid,
        "server_rss_kib": rss_kib,
        "swapusage": _command("/usr/sbin/sysctl", "-n", "vm.swapusage"),
        "pressure_level": _command(
            "/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"
        ),
        "memory_pressure": _command("/usr/bin/memory_pressure", "-Q"),
    }


def prompt(nominal_tokens: int, trial: int, prompt_seed: str) -> str:
    nonce = hashlib.sha256(
        f"{prompt_seed}:{nominal_tokens}:{trial}".encode()
    ).hexdigest()[:24]
    # Empirically ~43 tokenizer tokens per repetition on this checkpoint.
    body = FILLER * max(2, nominal_tokens // 43)
    text = (
        f"Unique uncached benchmark nonce {nonce}. Trial {trial}; nominal context "
        f"{nominal_tokens}.\n\n{body}\n\n"
        "Write a technically precise Python implementation of an LRU cache, "
        "including type hints and a short correctness explanation. Continue "
        "until the response token budget is reached."
    )
    if SUSTAINED_TASK:
        text = (
            text.rsplit("\n\n", 1)[0]
            + "\n\n"
            + (
                "Write a complete Python cache library with LRU, LFU, TTL and bounded "
                "thread-safe variants. Include type annotations, distinct implementations, "
                "comprehensive unit tests for each variant, usage examples, and complexity "
                "analysis. Develop each implementation fully and continue until the output "
                "token budget is exhausted."
            )
        )
    if EXACT_TOKENIZER is not None:
        prefix, _, task = text.split("\n\n", 2)

        def render(repeats, padding=0):
            return prefix + "\n\n" + FILLER * repeats + " x" * padding + "\n\n" + task

        def count(value):
            return len(EXACT_TOKENIZER.encode(value, add_special_tokens=False).ids)

        # Live API receipts establish five template tokens above plain text.
        target = nominal_tokens - 5
        low, high = 0, nominal_tokens
        while low < high:
            mid = (low + high + 1) // 2
            if count(render(mid)) <= target:
                low = mid
            else:
                high = mid - 1
        text = render(low)
        if target - count(text) == 1 and low > 0:
            low -= 1
            text = render(low)
        remainder = target - count(text)
        if remainder < 0:
            raise ValueError("Requested context smaller than task")
        for padding in range(max(0, remainder - 8), remainder + 9):
            text = render(low, padding)
            if count(text) == target:
                break
        else:
            raise ValueError("Exact prompt padding failed")
    return text


def request_stream(
    *,
    url: str,
    model: str,
    text: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    generation_mode: str | None,
    depth: int | None,
    timeout: float,
) -> dict:
    sampling_seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    body = {
        "seed": sampling_seed,
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stream": True,
        "stream_options": {"include_usage": True},
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if generation_mode:
        body["generation_mode"] = generation_mode
    if depth is not None:
        body["depth"] = int(depth)
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "X-MTPLX-Allow-Client-Controls": "1",
        },
        method="POST",
    )
    started = time.monotonic()
    first = None
    request_id = None
    usage: dict = {}
    server_stats: dict = {}
    chunks = 0
    chars = 0
    fragments = []
    reasoning_fragments = []
    raw_sse_lines = []
    protocol_errors = []
    done_events = 0
    finish_events = 0
    finish_reason = None
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
            raw_sse_lines.append(raw.decode("utf-8", "replace"))
            try:
                line = raw.decode("utf-8").strip()
            except UnicodeDecodeError:
                protocol_errors.append("invalid UTF-8")
                continue
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done_events += 1
                continue
            if done_events:
                protocol_errors.append("data event after DONE")
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                protocol_errors.append("malformed JSON data event")
                continue
            if not isinstance(event, dict):
                protocol_errors.append("non-object data event")
                continue
            if event.get("error") or event.get("type") == "error":
                protocol_errors.append("server error: " + json.dumps(event))
            event_id = event.get("id")
            if event_id:
                if request_id and request_id != event_id:
                    protocol_errors.append("request ID changed")
                request_id = request_id or event_id
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            for key in ("mtplx_stats", "vmlx_stats", "stats"):
                if isinstance(event.get(key), dict):
                    server_stats.update(event[key])
            choices = event.get("choices") or []
            if not isinstance(choices, list) or len(choices) > 1:
                protocol_errors.append("unexpected choices shape")
                continue
            if choices:
                choice = choices[0]
                if not isinstance(choice, dict):
                    protocol_errors.append("non-object choice")
                    continue
                if choice.get("finish_reason"):
                    finish_events += 1
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    protocol_errors.append("non-object delta")
                    continue
                content = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or ""
                if not isinstance(content, str) or not isinstance(reasoning, str):
                    protocol_errors.append("non-string content or reasoning")
                    continue
                fragments.append(content)
                reasoning_fragments.append(reasoning)
                fragment = content + reasoning
                if fragment:
                    chunks += 1
                    chars += len(fragment)
                    if first is None:
                        first = time.monotonic()
    ended = time.monotonic()
    ttft = None if first is None else first - started
    total = ended - started
    prompt_tokens = int(
        usage.get("prompt_tokens") or server_stats.get("prompt_tokens") or 0
    )
    completion_tokens = int(
        usage.get("completion_tokens") or server_stats.get("generated_tokens") or 0
    )
    decode_elapsed = None if ttft is None else max(0.0, total - ttft)
    if done_events != 1:
        protocol_errors.append(f"expected one DONE, got {done_events}")
    if finish_events != 1:
        protocol_errors.append(f"expected one finish, got {finish_events}")
    if not request_id:
        protocol_errors.append("missing request ID")
    if any(type(usage.get(key)) is not int or usage[key] <= 0
           for key in ("prompt_tokens", "completion_tokens")):
        protocol_errors.append("missing positive raw token usage")
    if finish_reason not in ("stop", "length"):
        protocol_errors.append("unexpected finish reason for prose speed task")
    if not "".join(fragments).strip():
        protocol_errors.append("empty visible answer")
    return {
        "protocol_valid": not protocol_errors,
        "protocol_errors": protocol_errors,
        "raw_sse_lines": raw_sse_lines,
        "done_events": done_events,
        "finish_events": finish_events,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": usage.get(
            "cached_tokens",
            (usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens", server_stats.get("cached_tokens")
            ),
        ),
        "request_id": request_id,
        "sampling_seed": sampling_seed,
        "raw_usage": usage,
        "output_text": "".join(fragments),
        "reasoning_text": "".join(reasoning_fragments),
        "request": body,
        "ttft_s": ttft,
        "total_s": total,
        "client_prefill_tok_s": (
            prompt_tokens / ttft if ttft and prompt_tokens else None
        ),
        "client_decode_tok_s": (
            completion_tokens / decode_elapsed
            if decode_elapsed and completion_tokens
            else None
        ),
        "client_decode_metric_scope": (
            "legacy client estimate: all completion tokens divided by time after "
            "first received text; first-burst tokens are not subtracted and "
            "terminal durability/transport time is included; not engine decode TPS"
        ),
        "client_end_to_end_tok_s": completion_tokens / total if total else None,
        "sse_nonempty_chunks": chunks,
        "output_chars": chars,
        "finish_reason": finish_reason,
        "usage_present": bool(usage),
        "server_stats": server_stats,
    }


def request_json(url: str, timeout: float) -> dict:
    """Fetch a JSON endpoint without making benchmark failure depend on it."""

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
        return payload if isinstance(payload, dict) else {"value": payload}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def default_health_url(completions_url: str) -> str:
    parsed = urllib.parse.urlsplit(completions_url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/health", "", ""))


def summarize(rows: list[dict]) -> dict:
    out = {}
    for target in sorted({int(row["nominal_context"]) for row in rows}):
        subset = [row for row in rows if int(row["nominal_context"]) == target]
        metrics = {}
        for key in (
            "ttft_s",
            "total_s",
            "client_prefill_tok_s",
            "client_decode_tok_s",
            "client_end_to_end_tok_s",
        ):
            values = [float(row[key]) for row in subset if row.get(key) is not None]
            metrics[key] = {
                "mean": statistics.mean(values) if values else None,
                "median": statistics.median(values) if values else None,
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
        out[str(target)] = {
            "trials": len(subset),
            "actual_prompt_tokens_mean": statistics.mean(
                int(row["prompt_tokens"]) for row in subset
            ),
            "completion_tokens": [int(row["completion_tokens"]) for row in subset],
            "cached_tokens": [row.get("cached_tokens") for row in subset],
            "metrics": metrics,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--health-url",
        help="Optional server health endpoint; defaults to /health on --url host",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--prompt-seed",
        required=True,
        help="Stable seed used to make prompt bytes identical across A/B arms",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", default="1024,2048,4096,8192")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--generation-mode", choices=("mtp", "ar"))
    parser.add_argument("--depth", type=int)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--exact-contexts", action="store_true")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Tokenizer JSON required with --exact-contexts; this Qwen4 protocol assumes five template tokens",
    )
    parser.add_argument("--sustained-task", action="store_true")
    parser.add_argument("--warmup-context", type=int, default=256)
    args = parser.parse_args()
    global EXACT_TOKENIZER, SUSTAINED_TASK
    SUSTAINED_TASK = args.sustained_task
    if args.exact_contexts:
        from tokenizers import Tokenizer

        if not args.tokenizer:
            parser.error("--exact-contexts requires --tokenizer")
        EXACT_TOKENIZER = Tokenizer.from_file(str(args.tokenizer))
    health_url = args.health_url or default_health_url(args.url)
    contexts = [int(item) for item in args.contexts.split(",") if item.strip()]
    result = {
        "protocol": {
            "label": args.label,
            "url": args.url,
            "health_url": health_url,
            "model": args.model,
            "contexts": contexts,
            "trials": args.trials,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "thinking": False,
            "stream": True,
            "unique_prompts": True,
            "prompt_seed": args.prompt_seed,
            "paired_prompt_bytes": True,
            "sampling_seed_policy": "first 32 bits of SHA-256(prompt text)",
            "cache_claim": "service cache settings unchanged; unique nonce; verify cached_tokens per result",
            "prefill_metric": "usage.prompt_tokens / streamed TTFT",
            "decode_metric": "usage.completion_tokens / (wall time - TTFT)",
            "generation_mode": args.generation_mode,
            "depth": args.depth,
            "exact_contexts": args.exact_contexts,
            "sustained_task": args.sustained_task,
            "warmup_context": args.warmup_context,
            "fixed_output_required": True,
            "max_attempts_per_trial": args.max_attempts,
        },
        "started_at": datetime.now(UTC).isoformat(),
        "host_before": host_snapshot(args.server_pid),
        "rows": [],
    }
    # Compile/warm without contaminating a measured prefix.
    result["warmup"] = request_stream(
        url=args.url,
        model=args.model,
        text=prompt(args.warmup_context, 0, args.prompt_seed + ":warmup"),
        max_tokens=32,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        generation_mode=args.generation_mode,
        depth=args.depth,
        timeout=args.timeout,
    )
    if not result["warmup"]["protocol_valid"]:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        raise RuntimeError("Invalid warmup stream; raw events retained in output")
    for context in contexts:
        for trial in range(1, args.trials + 1):
            rejected_short_attempts = []
            row = None
            for attempt in range(1, max(1, args.max_attempts) + 1):
                candidate = request_stream(
                    url=args.url,
                    model=args.model,
                    text=prompt(
                        context,
                        trial,
                        f"{args.prompt_seed}:attempt-{attempt}",
                    ),
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    generation_mode=args.generation_mode,
                    depth=args.depth,
                    timeout=args.timeout,
                )
                if not candidate["protocol_valid"]:
                    result.setdefault("invalid_attempts", []).append(candidate)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
                    raise RuntimeError("Invalid measured stream; raw events retained in output")
                if args.exact_contexts and candidate["prompt_tokens"] != context:
                    raise RuntimeError(
                        f"API prompt count {candidate['prompt_tokens']} differs from requested {context}; template overhead differs"
                    )
                candidate["health_after"] = request_json(
                    health_url, min(args.timeout, 30.0)
                )
                cache = (
                    candidate["health_after"]
                    .get("scheduler", {})
                    .get("batch_generator", {})
                    .get("last_cache_execution", {})
                )
                candidate["health_cache_matches_request"] = (
                    bool(candidate.get("request_id"))
                    and cache.get("request_id") == candidate["request_id"]
                )
                if (
                    candidate["cached_tokens"] is None
                    and candidate["health_cache_matches_request"]
                ):
                    candidate["cached_tokens"] = cache.get("cached_tokens")
                    candidate["cached_tokens_source"] = "request-matched health"
                if candidate["completion_tokens"] >= args.max_tokens:
                    row = candidate
                    break
                rejected_short_attempts.append(candidate)
                print(
                    f"{args.label} {context} trial={trial} attempt={attempt} "
                    f"rejected_short={candidate['completion_tokens']}",
                    flush=True,
                )
            if row is None:
                raise RuntimeError(
                    f"{args.label} context={context} trial={trial} failed to "
                    f"reach {args.max_tokens} completion tokens in "
                    f"{max(1, args.max_attempts)} attempts"
                )
            row["rejected_short_attempts"] = rejected_short_attempts
            row.update(
                {
                    "nominal_context": context,
                    "trial": trial,
                    "host_after": host_snapshot(args.server_pid),
                }
            )
            result["rows"].append(row)
            result["summary"] = summarize(result["rows"])
            result["host_after"] = row["host_after"]
            result["finished_at"] = datetime.now(UTC).isoformat()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(
                f"{args.label} {context} trial={trial} "
                f"prefill={row['client_prefill_tok_s']:.2f} "
                f"decode={row['client_decode_tok_s']:.2f} "
                f"tokens={row['completion_tokens']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
