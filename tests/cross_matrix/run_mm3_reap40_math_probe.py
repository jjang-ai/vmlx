#!/usr/bin/env python3
"""MiniMax-M3 REAP40 arithmetic exactness probe.

This is a diagnostic harness, not a fix. It exists to keep the arithmetic
failure boundary honest:

* API mode records what the shipped server returns for the exact math prompts.
* Runtime mode loads the raw MiniMax-M3 runtime, greedy-decodes the same prompt,
  and records token IDs plus top log-prob alternatives at each generated step.

If the raw runtime logits prefer the same wrong arithmetic token/path, the
failure is model/logit-path evidence. If API output diverges from raw runtime
under the same prompt and deterministic settings, investigate server/runtime
or template/cache orchestration before blaming the model.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
SYSTEM_PROMPT = "Solve the problem. Put your final numeric answer inside \\boxed{} at the end."
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
M3_THINKING_MODE = {
    "off": "disabled",
    "on": "enabled",
    "auto": "adaptive",
}


@dataclass(frozen=True)
class MathTask:
    task_id: str
    prompt: str
    expected: str


TASKS: tuple[MathTask, ...] = (
    MathTask("R01", "Compute 17 + 25.", "42"),
    MathTask("R02", "Compute (12 * 8) - 19.", "77"),
    MathTask("R04", "Compute 3/4 + 1/6. Give the answer as a decimal.", "0.9166666666666666"),
    MathTask("R05", "A train leaves at 2:15pm and arrives at 5:45pm. How many minutes did the trip take?", "210"),
    MathTask("R09", "Circumference of a circle with radius 7 (use pi=3.14159).", "43.9823"),
    MathTask("R11", "$1000 at 5% annual interest for 2 years, compounded yearly. Final balance?", "1102.5"),
    MathTask("R12", "Average of 4, 8, 12, 16, 20?", "12"),
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def extract_boxed(text: str) -> str | None:
    matches = BOXED_RE.findall(text or "")
    return matches[-1].strip() if matches else None


def normalize_answer(text: str | None) -> str | None:
    if text is None:
        return None
    return re.sub(r"\s+", "", text.strip().strip("$"))


def answer_matches(actual: str | None, expected: str) -> bool:
    actual_norm = normalize_answer(actual)
    expected_norm = normalize_answer(expected)
    if actual_norm is None:
        return False
    if actual_norm == expected_norm:
        return True
    try:
        return math.isclose(float(actual_norm), float(expected_norm), rel_tol=1e-6, abs_tol=1e-6)
    except Exception:
        return False


def selected_tasks(spec: str) -> list[MathTask]:
    if spec == "all":
        return list(TASKS)
    requested = {part.strip() for part in spec.split(",") if part.strip()}
    rows = [task for task in TASKS if task.task_id in requested]
    missing = sorted(requested - {task.task_id for task in rows})
    if missing:
        raise SystemExit(f"Unknown task id(s): {', '.join(missing)}")
    return rows


def post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))


def run_api_task(
    task: MathTask,
    *,
    server_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    thinking_mode: str,
    timeout: float,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task.prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if thinking_mode != "auto":
        body["enable_thinking"] = thinking_mode == "on"
    body["chat_template_kwargs"] = {
        "thinking_mode": M3_THINKING_MODE[thinking_mode],
    }
    t0 = time.time()
    try:
        data = post_json(f"{server_url.rstrip('/')}/v1/chat/completions", body, timeout=timeout)
        elapsed = time.time() - t0
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        boxed = extract_boxed(content)
        return {
            "task": asdict(task),
            "status": "ok",
            "elapsed_s": round(elapsed, 3),
            "content": content,
            "reasoning_content": reasoning,
            "boxed": boxed,
            "pass": answer_matches(boxed, task.expected),
            "raw_response": data,
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "task": asdict(task),
            "status": "error",
            "error": repr(exc),
            "pass": False,
        }


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id])
    except Exception:
        return f"<decode-error:{token_id}>"


def _top_logprobs(logits: Any, tokenizer: Any, *, top_k: int) -> list[dict[str, Any]]:
    import mlx.core as mx
    import numpy as np

    row = logits.astype(mx.float32)
    log_probs = row - mx.logsumexp(row, axis=-1)
    order = mx.argsort(-log_probs)[:top_k]
    mx.eval(order, log_probs)
    ids = np.array(order).astype(int).tolist()
    probs = np.array(log_probs[order]).astype(float).tolist()
    return [
        {
            "token_id": int(token_id),
            "token": _decode_token(tokenizer, int(token_id)),
            "logprob": float(logprob),
        }
        for token_id, logprob in zip(ids, probs)
    ]


def _chat_prompt_ids(tokenizer: Any, task: MathTask, *, thinking_mode: str) -> tuple[str, list[int]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            thinking_mode=M3_THINKING_MODE[thinking_mode],
        )
        return rendered, tokenizer.encode(rendered, add_special_tokens=False)
    rendered = f"{SYSTEM_PROMPT}\n\nUser: {task.prompt}\nAssistant:"
    return rendered, tokenizer.encode(rendered)


def _cache_snapshot(cache: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    mismatches: list[int] = []
    for i, layer in enumerate(cache or []):
        class_name = type(layer).__name__
        offset = getattr(layer, "offset", None)
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        idx_keys = getattr(layer, "idx_keys", None)
        key_len = getattr(keys, "shape", [None, None, None])[2] if keys is not None else None
        value_len = getattr(values, "shape", [None, None, None])[2] if values is not None else None
        idx_len = getattr(idx_keys, "shape", [None, None, None])[2] if idx_keys is not None else None
        if class_name == "MiniMaxM3SparseCache":
            if not (offset == key_len == value_len == idx_len):
                mismatches.append(i)
        rows.append(
            {
                "layer": i,
                "class": class_name,
                "offset": offset,
                "keys_len": key_len,
                "values_len": value_len,
                "idx_keys_len": idx_len,
            }
        )
    return {
        "layers": rows,
        "mismatch_layers": mismatches,
        "m3_sparse_invariants_ok": not mismatches,
    }


def run_runtime_task(
    task: MathTask,
    *,
    model: Any,
    tokenizer: Any,
    eos_ids: set[int],
    max_tokens: int,
    thinking_mode: str,
    top_logprobs: int,
) -> dict[str, Any]:
    import mlx.core as mx

    rendered, ids = _chat_prompt_ids(tokenizer, task, thinking_mode=thinking_mode)
    cache = model.make_cache()
    logits = model(mx.array([ids]), cache=cache)
    mx.eval(logits)

    output_ids: list[int] = []
    token_trace: list[dict[str, Any]] = []
    prefill_cache_snapshot = _cache_snapshot(cache)
    t0 = time.time()
    for step in range(max_tokens):
        current = logits[0, -1].astype(mx.float32)
        next_id = int(mx.argmax(current))
        alternatives = _top_logprobs(current, tokenizer, top_k=top_logprobs)
        token_trace.append(
            {
                "step": step,
                "selected_token_id": next_id,
                "selected_token": _decode_token(tokenizer, next_id),
                "top_logprobs": alternatives,
                "cache_snapshot_before_next_forward": _cache_snapshot(cache),
            }
        )
        if next_id in eos_ids:
            break
        output_ids.append(next_id)
        logits = model(mx.array([[next_id]]), cache=cache)
        mx.eval(logits)

    elapsed = time.time() - t0
    content = tokenizer.decode(output_ids)
    boxed = extract_boxed(content)
    return {
        "task": asdict(task),
        "status": "ok",
        "elapsed_s": round(elapsed, 3),
        "rendered_prompt_prefix": rendered[:1200],
        "prompt_tokens": len(ids),
        "output_token_ids": output_ids,
        "content": content,
        "boxed": boxed,
        "pass": answer_matches(boxed, task.expected),
        "prefill_cache_snapshot": prefill_cache_snapshot,
        "final_cache_snapshot": _cache_snapshot(cache),
        "token_trace": token_trace,
    }


def load_runtime(model_path: str) -> tuple[Any, Any, set[int]]:
    sys.path.insert(0, str(REPO))
    from vmlx_engine.models.minimax_m3.runtime import load_minimax_m3

    return load_minimax_m3(model_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("api", "runtime", "both"), default="api")
    ap.add_argument("--tasks", default="all", help="Comma-separated task ids or 'all'")
    ap.add_argument("--server-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="minimax-m3-reap40-vmlx-clean")
    ap.add_argument("--model-path", default="")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--thinking-mode", choices=("off", "on", "auto"), default="off")
    ap.add_argument("--request-timeout", type=float, default=600.0)
    ap.add_argument("--top-logprobs", type=int, default=8)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    tasks = selected_tasks(args.tasks)
    result: dict[str, Any] = {
        "schema": "vmlx-mm3-reap40-math-exactness-probe-v1",
        "created_at": _now(),
        "mode": args.mode,
        "settings": {
            "system_prompt": SYSTEM_PROMPT,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_logprobs": args.top_logprobs,
            "thinking_mode": args.thinking_mode,
            "server_url": args.server_url,
            "model": args.model,
            "model_path": args.model_path,
        },
        "api": [],
        "runtime": [],
        "summary": {},
    }

    if args.mode in {"api", "both"}:
        for task in tasks:
            result["api"].append(
                run_api_task(
                    task,
                    server_url=args.server_url,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    thinking_mode=args.thinking_mode,
                    timeout=args.request_timeout,
                )
            )

    if args.mode in {"runtime", "both"}:
        if not args.model_path:
            raise SystemExit("--model-path is required for runtime/both mode")
        model, tokenizer, eos_ids = load_runtime(args.model_path)
        for task in tasks:
            result["runtime"].append(
                run_runtime_task(
                    task,
                    model=model,
                    tokenizer=tokenizer,
                    eos_ids=eos_ids,
                    max_tokens=args.max_tokens,
                    thinking_mode=args.thinking_mode,
                    top_logprobs=args.top_logprobs,
                )
            )

    for key in ("api", "runtime"):
        rows = result[key]
        if rows:
            result["summary"][key] = {
                "passed": sum(1 for row in rows if row.get("pass") is True),
                "total": len(rows),
                "failures": [
                    {
                        "task_id": row.get("task", {}).get("task_id"),
                        "expected": row.get("task", {}).get("expected"),
                        "boxed": row.get("boxed"),
                        "status": row.get("status"),
                    }
                    for row in rows
                    if row.get("pass") is not True
                ],
            }

    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print(out)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
