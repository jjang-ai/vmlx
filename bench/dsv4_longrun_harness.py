#!/usr/bin/env python3
"""Long-run DSV4 decode and prefill harness.

This complements the 30-token target gate with an exact 1,000-token stream
length check. The persistent generator excludes its one-time warmup from the
reported prefill/decode timings and records the output-token hash for repeat
comparison. Pass ``--expected-token-hash`` to make the run an exact-reference
gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dsv4_target_harness import _parse_targets, _PersistentRunner, _prompt

DEFAULT_OUTPUT = Path("dsv4-longrun-report.json")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def _request(runner: _PersistentRunner, prompt: Any, max_tokens: int) -> dict[str, Any]:
    prompt_ids = [int(value) for value in prompt.tolist()]
    uid = runner.generator.insert(
        [prompt_ids],
        max_tokens=[max_tokens],
        capture_prompt_snapshots=[False],
    )[0]
    started = time.perf_counter()
    first_token_at: float | None = None
    output_tokens: list[int] = []
    try:
        while len(output_tokens) < max_tokens:
            prompt_responses, generated_responses = runner.generator.next()
            responses = list(prompt_responses) + list(generated_responses)
            if not responses:
                break
            if first_token_at is None:
                first_token_at = time.perf_counter()
            output_tokens.extend(int(response.token) for response in responses)
            if any(response.finish_reason is not None for response in responses):
                break
    finally:
        runner.generator.remove([uid])
    finished = time.perf_counter()
    decode_seconds = finished - first_token_at if first_token_at is not None else None
    return {
        "prompt_tokens": len(prompt_ids),
        "new_tokens": len(output_tokens),
        "ttft_s": (first_token_at - started if first_token_at is not None else None),
        "prefill_tok_s": (
            len(prompt_ids) / (first_token_at - started)
            if first_token_at is not None and first_token_at > started
            else None
        ),
        "decode_tok_s": (
            (len(output_tokens) - 1) / decode_seconds
            if decode_seconds and len(output_tokens) > 1
            else None
        ),
        "token_ids_sha256": hashlib.sha256(
            ",".join(map(str, output_tokens)).encode()
        ).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt-targets", default="256")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--warmup-tokens", type=int, default=4)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--expected-token-hash")
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    if args.max_tokens < 1000:
        parser.error("this long-run harness requires at least 1000 output tokens")
    if args.repeats <= 0 or args.warmup_tokens <= 0:
        parser.error("repeats and warmup tokens must be positive")
    try:
        targets = _parse_targets(args.prompt_targets)
    except ValueError as exc:
        parser.error(str(exc))

    os.environ.setdefault("VMLX_DSV4_AFFINE_MOE_FASTPATH", "0")
    import mlx.core as mx

    import vmlx_engine
    from vmlx_engine.loaders.load_jangtq_dsv4 import load_jangtq_dsv4_model

    model, tokenizer = load_jangtq_dsv4_model(str(args.model))
    from jang_tools.dsv4.mlx_model import MoE as _Dsv4MoE

    compiled_moe_installed = bool(getattr(_Dsv4MoE, "_vmlx_dsv4_compiled_moe", False))
    prompts = {target: _prompt(tokenizer, target, mx) for target in targets}
    mx.eval(*prompts.values())
    runner = _PersistentRunner(
        model,
        step_size=args.prefill_step_size,
        max_tokens=args.max_tokens,
        mx=mx,
    )
    first_prompt = next(iter(prompts.values()))
    runner.request(first_prompt, max_tokens=args.warmup_tokens)

    results: dict[str, Any] = {}
    for target, prompt in prompts.items():
        rows = [_request(runner, prompt, args.max_tokens) for _ in range(args.repeats)]
        hashes = {row["token_ids_sha256"] for row in rows}
        results[str(target)] = {
            "passed": bool(rows)
            and all(row["new_tokens"] == args.max_tokens for row in rows)
            and len(hashes) == 1
            and (
                args.expected_token_hash is None
                or all(
                    row["token_ids_sha256"] == args.expected_token_hash for row in rows
                )
            ),
            "stable_token_stream": len(hashes) == 1,
            "reference_token_hash": args.expected_token_hash,
            "reference_token_hash_matches": args.expected_token_hash is None
            or all(row["token_ids_sha256"] == args.expected_token_hash for row in rows),
            "median_prefill_tok_s": (
                statistics.median(
                    float(row["prefill_tok_s"])
                    for row in rows
                    if row.get("prefill_tok_s") is not None
                )
                if any(row.get("prefill_tok_s") is not None for row in rows)
                else None
            ),
            "median_decode_tok_s": (
                statistics.median(
                    float(row["decode_tok_s"])
                    for row in rows
                    if row.get("decode_tok_s") is not None
                )
                if any(row.get("decode_tok_s") is not None for row in rows)
                else None
            ),
            "samples": rows,
        }

    report = {
        "schema": "dsv4_longrun_harness.v1",
        "finished_at": _now(),
        "model": str(args.model),
        "runtime": "vmlx_dsv4_batch_generator",
        "vmlx_engine": vmlx_engine.__version__,
        "jang": importlib.metadata.version("jang"),
        "mlx": importlib.metadata.version("mlx"),
        "mlx_lm": importlib.metadata.version("mlx-lm"),
        "lm_head_mode": os.environ.get("VMLX_DSV4_LM_HEAD_MODE", "quantized")
        .strip()
        .lower(),
        "lm_head_fastpath_installed": bool(
            getattr(type(model), "_vmlx_dsv4_lm_head_cache_installed", False)
        ),
        "compiled_moe_installed": compiled_moe_installed,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "prefill_step_size": args.prefill_step_size,
        "expected_token_hash": args.expected_token_hash,
        "results": results,
        "passed": bool(results) and all(row["passed"] for row in results.values()),
    }
    _atomic_write(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {args.out}")
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
