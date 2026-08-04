#!/usr/bin/env python3
"""Stable DSV4 target harness for 30-token output and cold prefill.

The generator is intentionally reused across requests. DSV4BatchGenerator
performs a one-time kernel warmup on its first request; constructing a fresh
generator for every sample would charge that fixed cost to every prefill
measurement and under-report the production path.

The default gate requires every measured sample to produce exactly 30 tokens,
the greedy token stream to remain stable, and the measured prefill rate to be
at least 200 tokens/s. Pass ``--expected-token-hash`` when the run is being
used as an exact-reference gate rather than a repeatability probe.
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

SCHEMA = "dsv4_target_harness.v1"
DEFAULT_OUTPUT = Path("dsv4-target-report.json")
DEFAULT_PROMPT_TARGETS = (256,)

PROMPT = (
    "We are reviewing a production Apple-Silicon inference service. A recent "
    "optimization changed the request loop to reuse a mutable KV cache and to "
    "batch prompt tokens in chunks. Users report slow first answers and "
    "occasional repetition from a previous request. Review this simplified "
    "code: cache = shared_cache; for request in requests: prompt = "
    "tokenizer.encode(request.text); for start in range(0, len(prompt), 2048): "
    "logits = model(prompt[start:start + 2048], cache=cache); token = "
    "sample(logits[:, -1, :]); while token not in stop_tokens: logits = "
    "model(token, cache=cache); token = sample(logits[:, -1, :]); yield "
    "tokenizer.decode(token). Give a concise diagnosis, corrected pseudocode, "
    "and three tests. State assumptions."
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_targets(raw: str) -> tuple[int, ...]:
    values = tuple(
        dict.fromkeys(int(item.strip()) for item in raw.split(",") if item.strip())
    )
    if not values or any(value <= 0 for value in values):
        raise ValueError("prompt targets must contain positive integers")
    return values


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def _prompt(tokenizer: Any, target_tokens: int, mx: Any) -> Any:
    body = PROMPT
    while len(tokenizer.encode(body)) < target_tokens:
        body += " Preserve request isolation and exact timing receipts."
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": body}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return mx.array(tokenizer.encode(rendered), dtype=mx.uint32)


class _PersistentRunner:
    def __init__(self, model: Any, *, step_size: int, max_tokens: int, mx: Any):
        # Use vMLX's production sampler wrapper. Its greedy sampler is marked
        # as accepting raw logits, which lets DSV4BatchGenerator skip an
        # unnecessary full-vocabulary log-softmax on every decode token.
        from vmlx_engine.sampling import make_sampler
        from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator

        os.environ["DSV4_PREFILL_STEP_SIZE"] = str(step_size)
        self.generator = DSV4BatchGenerator(
            model,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0, top_p=0.0),
            prefill_step_size=step_size,
            capture_prompt_snapshot=False,
        )
        self.max_tokens = max_tokens
        self.mx = mx

    def request(self, prompt: Any, *, max_tokens: int | None = None) -> dict[str, Any]:
        token_limit = max_tokens or self.max_tokens
        prompt_ids = [int(value) for value in prompt.tolist()]
        uid = self.generator.insert(
            [prompt_ids],
            max_tokens=[token_limit],
            capture_prompt_snapshots=[False],
        )[0]
        started = time.perf_counter()
        first_token_at: float | None = None
        output_tokens: list[int] = []
        try:
            while len(output_tokens) < token_limit:
                prompt_responses, generated_responses = self.generator.next()
                responses = list(prompt_responses) + list(generated_responses)
                if not responses:
                    break
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                output_tokens.extend(int(response.token) for response in responses)
                if any(response.finish_reason is not None for response in responses):
                    break
        finally:
            self.generator.remove([uid])
        finished = time.perf_counter()
        ttft = first_token_at - started if first_token_at is not None else None
        decode_seconds = (
            finished - first_token_at if first_token_at is not None else None
        )
        return {
            "prompt_tokens": len(prompt_ids),
            "new_tokens": len(output_tokens),
            "ttft_s": ttft,
            "prefill_tok_s": len(prompt_ids) / ttft if ttft else None,
            "decode_tok_s": (
                max(len(output_tokens) - 1, 0) / decode_seconds
                if decode_seconds and len(output_tokens) > 1
                else None
            ),
            "token_ids_sha256": hashlib.sha256(
                ",".join(map(str, output_tokens)).encode()
            ).hexdigest(),
        }


def _summarize(
    rows: list[dict[str, Any]],
    minimum_prefill: float,
    expected_token_hash: str | None,
) -> dict[str, Any]:
    hashes = {row["token_ids_sha256"] for row in rows}
    prefill = [
        float(row["prefill_tok_s"])
        for row in rows
        if row.get("prefill_tok_s") is not None
    ]
    decode = [float(row["decode_tok_s"]) for row in rows if row.get("decode_tok_s")]
    reference_matches = expected_token_hash is None or all(
        row["token_ids_sha256"] == expected_token_hash for row in rows
    )
    passed = bool(
        rows
        and all(row["new_tokens"] == 30 for row in rows)
        and len(hashes) == 1
        and bool(prefill)
        and min(prefill) >= minimum_prefill
        and reference_matches
    )
    return {
        "passed": passed,
        "stable_30_tokens": bool(rows) and all(row["new_tokens"] == 30 for row in rows),
        "stable_token_stream": len(hashes) == 1,
        "reference_token_hash": expected_token_hash,
        "reference_token_hash_matches": reference_matches,
        "min_prefill_tok_s": min(prefill) if prefill else None,
        "median_prefill_tok_s": statistics.median(prefill) if prefill else None,
        "median_decode_tok_s": statistics.median(decode) if decode else None,
        "samples": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt-targets", default="256")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--warmup-tokens", type=int, default=4)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--min-prefill-tok-s", type=float, default=200.0)
    parser.add_argument("--expected-token-hash")
    parser.add_argument(
        "--affine",
        action="store_true",
        help="opt into upstream's guarded affine decode path for comparison",
    )
    args = parser.parse_args()
    if args.max_tokens != 30:
        parser.error("this target harness requires --max-tokens 30")
    if args.repeats <= 0 or args.warmup_tokens <= 0:
        parser.error("repeats and warmup tokens must be positive")
    if args.prefill_step_size <= 0 or args.min_prefill_tok_s <= 0:
        parser.error("prefill step and minimum throughput must be positive")
    try:
        targets = _parse_targets(args.prompt_targets)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")

    os.environ["VMLX_DSV4_AFFINE_MOE_FASTPATH"] = "1" if args.affine else "0"
    import mlx.core as mx

    import vmlx_engine
    from vmlx_engine.loaders.load_jangtq_dsv4 import load_jangtq_dsv4_model

    model, tokenizer = load_jangtq_dsv4_model(str(args.model))
    lm_head_cache_installed = bool(
        getattr(type(model), "_vmlx_dsv4_lm_head_cache_installed", False)
    )
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
    results: dict[str, Any] = {}
    for target, prompt in prompts.items():
        runner.request(prompt, max_tokens=args.warmup_tokens)
        rows = [runner.request(prompt) for _ in range(args.repeats)]
        results[str(target)] = _summarize(
            rows,
            args.min_prefill_tok_s,
            args.expected_token_hash,
        )

    report = {
        "schema": SCHEMA,
        "finished_at": _now(),
        "model": str(args.model),
        "runtime": "vmlx_dsv4_batch_generator",
        "vmlx_engine": vmlx_engine.__version__,
        "jang": importlib.metadata.version("jang"),
        "mlx": importlib.metadata.version("mlx"),
        "mlx_lm": importlib.metadata.version("mlx-lm"),
        "affine_opt_in": args.affine,
        "lm_head_mode": os.environ.get("VMLX_DSV4_LM_HEAD_MODE", "quantized")
        .strip()
        .lower(),
        "lm_head_fastpath_installed": lm_head_cache_installed,
        "compiled_moe_installed": compiled_moe_installed,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "prefill_step_size": args.prefill_step_size,
        "minimum_prefill_tok_s": args.min_prefill_tok_s,
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
