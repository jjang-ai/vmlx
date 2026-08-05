#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproducible native-versus-optimized DeepSeek-V4 performance suite.

Run the native profile first, then pass its report to the optimized profile.
Acceptance requires exact output hashes for every case and bounded throughput
ratios; repeatability without a native report is never called exactness.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
with contextlib.suppress(ValueError):
    sys.path.remove(str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))

SCHEMA = "dsv4_performance_suite.v1"
DEFAULT_CASES = "256:30:3,400:30:3,512:30:3,1024:30:3,2048:30:3,256:1000:1"
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


@dataclass(frozen=True)
class Case:
    prompt_target: int
    generation_target: int
    repeats: int

    @property
    def key(self) -> str:
        return f"p{self.prompt_target}-g{self.generation_target}"


def _parse_cases(raw: str) -> tuple[Case, ...]:
    cases: list[Case] = []
    seen: set[str] = set()
    for item in raw.split(","):
        parts = item.strip().split(":")
        if len(parts) != 3:
            raise ValueError(f"invalid case {item!r}; expected prompt:generation:repeats")
        case = Case(*(int(value) for value in parts))
        if min(case.prompt_target, case.generation_target, case.repeats) <= 0:
            raise ValueError("case values must be positive")
        if case.key in seen:
            raise ValueError(f"duplicate case {case.key}")
        seen.add(case.key)
        cases.append(case)
    if not cases:
        raise ValueError("at least one case is required")
    return tuple(cases)


def _configure_profile(profile: str, features: str = "all") -> dict[str, str]:
    controls = {
        "VMLX_DSV4_AFFINE_MOE_FASTPATH": "0",
        "VMLX_DSV4_COMPILED_MOE": "0",
        "VMLX_DSV4_SKIP_UNUSED_INDEXER": "0",
        "VMLX_DSV4_ROPE_NATIVE": "0",
        "DSV4_LAYERWISE_PREFILL_MIN_TOKENS": "256",
    }
    if profile == "native":
        controls.update(
            {
                "VMLX_DSV4_LM_HEAD_MODE": "native",
                "VMLX_DSV4_ROPE_CACHE": "0",
            }
        )
    else:
        controls.update(
            {
                "VMLX_DSV4_LM_HEAD_MODE": (
                    "exact-cache" if features in {"all", "lm-head"} else "native"
                ),
                "VMLX_DSV4_ROPE_CACHE": (
                    "1" if features in {"all", "rope-cache"} else "0"
                ),
            }
        )
    os.environ.update(controls)
    return controls


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _model_identity(model_path: Path) -> dict[str, Any]:
    """Return a public, reproducible identity without exposing a local path."""

    digest = hashlib.sha256()
    metadata_names = (
        "config.json",
        "jang_config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    )
    for name in metadata_names:
        path = model_path / name
        if not path.is_file():
            continue
        digest.update(name.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)

    index_path = model_path / "model.safetensors.index.json"
    shard_names: list[str]
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(index.get("weight_map", {}).values()))
    except (OSError, TypeError, ValueError):
        shard_names = sorted(path.name for path in model_path.glob("model*.safetensors"))

    shard_bytes = 0
    present_shards = 0
    for name in shard_names:
        path = model_path / name
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        if size >= 0:
            present_shards += 1
            shard_bytes += size
        digest.update(f"{name}\0{size}\0".encode())

    return {
        "bundle_name": model_path.name,
        "manifest_sha256": digest.hexdigest(),
        "shard_count": present_shards,
        "shard_bytes": shard_bytes,
    }


def _swap_used_bytes() -> int | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"],
            check=True,
            capture_output=True,
            text=True,
        )
        used = result.stdout.split("used =")[1].split()[0]
        scale = {"K": 1024, "M": 1024**2, "G": 1024**3}.get(used[-1], 1)
        return int(float(used.rstrip("KMG")) * scale)
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        return None


def _memory_snapshot(mx: Any) -> dict[str, Any]:
    import resource

    from vmlx_engine.utils.memory_limits import (
        get_effective_metal_working_set_bytes,
        metal_resource_limit,
    )

    try:
        active_bytes, max_working_set = get_effective_metal_working_set_bytes(mx)
    except Exception:
        active_bytes, max_working_set = None, None
    return {
        "mlx_active_bytes": int(mx.get_active_memory()),
        "mlx_peak_bytes": int(mx.get_peak_memory()),
        "mlx_cache_bytes": int(mx.get_cache_memory()),
        "effective_working_set_active_bytes": active_bytes,
        "effective_working_set_max_bytes": max_working_set,
        "metal_resource_limit": metal_resource_limit(),
        "process_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "swap_used_bytes": _swap_used_bytes(),
    }


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


class _Runner:
    def __init__(self, model: Any, *, max_tokens: int):
        from vmlx_engine.sampling import make_sampler
        from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator

        self.generator = DSV4BatchGenerator(
            model,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0, top_p=0.0),
            prefill_step_size=2048,
            capture_prompt_snapshot=False,
        )

    def request(self, prompt: Any, generation_target: int) -> dict[str, Any]:
        import mlx.core as mx

        mx.reset_peak_memory()
        prompt_ids = [int(value) for value in prompt.tolist()]
        uid = self.generator.insert(
            [prompt_ids],
            max_tokens=[generation_target],
            capture_prompt_snapshots=[False],
        )[0]
        started = time.perf_counter()
        first_token_at: float | None = None
        output_tokens: list[int] = []
        finish_reason: str | None = None
        try:
            while len(output_tokens) < generation_target:
                prompt_responses, generated_responses = self.generator.next()
                responses = [*prompt_responses, *generated_responses]
                if not responses:
                    break
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                output_tokens.extend(int(response.token) for response in responses)
                reasons = [response.finish_reason for response in responses]
                finish_reason = next((reason for reason in reasons if reason), None)
                if finish_reason is not None:
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
            "finish_reason": finish_reason,
            "ttft_s": ttft,
            "prefill_tok_s": len(prompt_ids) / ttft if ttft else None,
            "decode_tok_s": (
                (len(output_tokens) - 1) / decode_seconds
                if decode_seconds and len(output_tokens) > 1
                else None
            ),
            "token_ids": output_tokens,
            "token_ids_sha256": hashlib.sha256(
                ",".join(map(str, output_tokens)).encode()
            ).hexdigest(),
            "memory": _memory_snapshot(mx),
        }


def _summarize(
    case: Case,
    samples: list[dict[str, Any]],
    *,
    min_within_profile_decode_ratio: float = 0.75,
) -> dict[str, Any]:
    hashes = {sample["token_ids_sha256"] for sample in samples}
    decode = [float(sample["decode_tok_s"]) for sample in samples if sample["decode_tok_s"]]
    prefill = [
        float(sample["prefill_tok_s"]) for sample in samples if sample["prefill_tok_s"]
    ]
    complete = bool(samples) and all(
        sample["new_tokens"] == case.generation_target for sample in samples
    )
    median_decode = statistics.median(decode) if decode else None
    minimum_decode = min(decode) if decode else None
    tail_ratio = (
        minimum_decode / median_decode
        if minimum_decode is not None and median_decode
        else 0.0
    )
    stable_decode_tail = tail_ratio >= min_within_profile_decode_ratio
    return {
        "passed": bool(
            complete
            and len(hashes) == 1
            and decode
            and prefill
            and stable_decode_tail
        ),
        "complete": complete,
        "stable_token_stream": len(hashes) == 1,
        "stable_decode_tail": stable_decode_tail,
        "token_ids_sha256": next(iter(hashes)) if len(hashes) == 1 else None,
        "median_decode_tok_s": median_decode,
        "min_decode_tok_s": minimum_decode,
        "min_to_median_decode_ratio": tail_ratio,
        "median_prefill_tok_s": statistics.median(prefill) if prefill else None,
        "samples": samples,
    }


def _compare(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    *,
    min_decode_ratio: float,
    min_prefill_ratio: float,
    min_sample_decode_ratio: float = 0.75,
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    passed = True
    candidate_results = candidate.get("results", {})
    baseline_results = baseline.get("results", {})
    for key in sorted(set(candidate_results) | set(baseline_results)):
        candidate_case = candidate_results.get(key)
        baseline_case = baseline_results.get(key)
        if candidate_case is None:
            comparisons[key] = {"passed": False, "reason": "missing candidate case"}
            passed = False
            continue
        if baseline_case is None:
            comparisons[key] = {"passed": False, "reason": "missing baseline case"}
            passed = False
            continue
        exact = (
            candidate_case.get("token_ids_sha256")
            == baseline_case.get("token_ids_sha256")
            and candidate_case.get("token_ids_sha256") is not None
        )
        candidate_decode = float(candidate_case.get("median_decode_tok_s") or 0.0)
        baseline_decode = float(baseline_case.get("median_decode_tok_s") or 0.0)
        candidate_min_decode = float(candidate_case.get("min_decode_tok_s") or 0.0)
        baseline_min_decode = float(
            baseline_case.get("min_decode_tok_s")
            or baseline_case.get("median_decode_tok_s")
            or 0.0
        )
        candidate_prefill = float(candidate_case.get("median_prefill_tok_s") or 0.0)
        baseline_prefill = float(baseline_case.get("median_prefill_tok_s") or 0.0)
        decode_ratio = candidate_decode / baseline_decode if baseline_decode else 0.0
        sample_decode_ratio = (
            candidate_min_decode / baseline_min_decode if baseline_min_decode else 0.0
        )
        prefill_ratio = candidate_prefill / baseline_prefill if baseline_prefill else 0.0
        case_passed = bool(
            candidate_case.get("passed")
            and baseline_case.get("passed")
            and exact
            and decode_ratio >= min_decode_ratio
            and sample_decode_ratio >= min_sample_decode_ratio
            and prefill_ratio >= min_prefill_ratio
        )
        comparisons[key] = {
            "passed": case_passed,
            "exact_token_hash": exact,
            "decode_ratio": decode_ratio,
            "min_sample_decode_ratio": sample_decode_ratio,
            "prefill_ratio": prefill_ratio,
        }
        passed = passed and case_passed
    return {"passed": passed and bool(comparisons), "cases": comparisons}


def _baseline_compatibility(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    checks = {
        "schema": baseline.get("schema") == candidate.get("schema") == SCHEMA,
        "native_profile": baseline.get("profile") == "native",
        "native_passed": baseline.get("passed") is True,
        "native_feature_state": baseline.get("feature_state_passed") is True,
        "model": baseline.get("model") == candidate.get("model"),
        "source": baseline.get("source") == candidate.get("source"),
        "git_revision": baseline.get("git_revision") == candidate.get("git_revision"),
        "hardware": baseline.get("hardware") == candidate.get("hardware"),
        "runtime": baseline.get("runtime") == candidate.get("runtime"),
        "cases": baseline.get("cases") == candidate.get("cases"),
        "benchmark_controls": baseline.get("benchmark_controls")
        == candidate.get("benchmark_controls"),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _decode_floor_sanity(
    results: dict[str, Any],
    *,
    anchor_key: str = "p256-g1000",
    min_anchor_ratio: float = 0.40,
) -> dict[str, Any]:
    anchor = results.get(anchor_key)
    anchor_decode = float(anchor.get("median_decode_tok_s") or 0.0) if anchor else 0.0
    if anchor_decode <= 0:
        return {
            "passed": True,
            "applied": False,
            "reason": f"anchor case {anchor_key} was not requested",
            "cases": {},
        }
    checks = {}
    passed = True
    for key, result in sorted(results.items()):
        decode = float(result.get("median_decode_tok_s") or 0.0)
        ratio = decode / anchor_decode
        case_passed = ratio >= min_anchor_ratio
        checks[key] = {"passed": case_passed, "anchor_decode_ratio": ratio}
        passed = passed and case_passed
    return {"passed": passed, "applied": True, "cases": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--profile", choices=("native", "optimized"), required=True)
    parser.add_argument(
        "--features",
        choices=("all", "lm-head", "rope-cache"),
        default="all",
        help="optimized components to enable; native always disables all",
    )
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--acceptance", action="store_true")
    parser.add_argument(
        "--expect-lm-head-denied",
        action="store_true",
        help=(
            "optimized profile only: assert the lm_head cache was requested "
            "but declined admission, proving the fail-closed path stays native"
        ),
    )
    parser.add_argument("--min-decode-ratio", type=float, default=0.98)
    parser.add_argument("--min-prefill-ratio", type=float, default=0.95)
    parser.add_argument("--min-sample-decode-ratio", type=float, default=0.75)
    parser.add_argument("--min-within-profile-decode-ratio", type=float, default=0.75)
    parser.add_argument("--min-anchor-decode-ratio", type=float, default=0.40)
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    model_path = args.model.resolve()
    try:
        cases = _parse_cases(args.cases)
    except ValueError as exc:
        parser.error(str(exc))
    if args.acceptance and (args.profile != "optimized" or args.baseline_report is None):
        parser.error("acceptance requires --profile optimized and --baseline-report")
    required_acceptance_cases = {case.key for case in _parse_cases(DEFAULT_CASES)}
    if args.acceptance and not required_acceptance_cases.issubset(
        {case.key for case in cases}
    ):
        parser.error("acceptance requires the complete default case matrix")
    if min(
        args.min_decode_ratio,
        args.min_prefill_ratio,
        args.min_sample_decode_ratio,
        args.min_within_profile_decode_ratio,
        args.min_anchor_decode_ratio,
    ) <= 0:
        parser.error("throughput ratios must be positive")

    controls = _configure_profile(args.profile, args.features)
    import mlx.core as mx

    import vmlx_engine
    from vmlx_engine.loaders.load_jangtq_dsv4 import load_jangtq_dsv4_model
    from vmlx_engine.models.dsv4_lm_head_fastpath import (
        dsv4_lm_head_fastpath_status,
    )
    from vmlx_engine.models.dsv4_rope_cache import dsv4_rope_cache_status

    # Resolve candidate modules before the external JANG loader registers its
    # model implementation. This makes the benchmark's source boundary
    # explicit and prevents an installed package from silently satisfying a
    # checkout-under-test import later in the run.
    load_started = time.perf_counter()
    model, tokenizer = load_jangtq_dsv4_model(str(model_path))
    load_seconds = time.perf_counter() - load_started
    post_load_memory = _memory_snapshot(mx)
    source_root = Path(vmlx_engine.__file__).resolve().parents[1]
    if source_root != REPOSITORY_ROOT:
        raise RuntimeError(
            f"benchmark imported vmlx_engine from {source_root}, "
            f"expected checkout {REPOSITORY_ROOT}"
        )

    prompts = {
        target: _prompt(tokenizer, target, mx)
        for target in {case.prompt_target for case in cases}
    }
    mx.eval(*prompts.values())
    runner = _Runner(model, max_tokens=max(case.generation_target for case in cases))
    # Warm the full decode path before the first measured case. A 4-token
    # warmup leaves kernel compilation, page-in, and allocator steady-state to
    # the first case, whose samples then carry order-of-magnitude jitter (seen
    # as 0.7 tok/s first samples on a 128 GB M4 Max). Two full-length warm
    # requests on the smallest prompt make case one measurable.
    warmup_prompt = prompts[min(prompts)]
    warmup_tokens = min(30, max(case.generation_target for case in cases))
    for _warmup in range(2):
        runner.request(warmup_prompt, warmup_tokens)

    results: dict[str, Any] = {}
    for case in cases:
        samples = [
            runner.request(prompts[case.prompt_target], case.generation_target)
            for _repeat in range(case.repeats)
        ]
        results[case.key] = _summarize(
            case,
            samples,
            min_within_profile_decode_ratio=args.min_within_profile_decode_ratio,
        )

    feature_state = {
        "lm_head": dsv4_lm_head_fastpath_status(model),
        "rope_cache": dsv4_rope_cache_status(model),
    }
    expected_head = (
        args.profile == "optimized"
        and args.features in {"all", "lm-head"}
        and not args.expect_lm_head_denied
    )
    expected_rope = args.profile == "optimized" and args.features in {"all", "rope-cache"}
    head_active = bool(
        feature_state["lm_head"]["installed"]
        and feature_state["lm_head"]["validated"]
        and feature_state["lm_head"]["disabled_reason"] is None
    )
    rope_active = bool(
        feature_state["rope_cache"]["registered_instances"] > 0
        and feature_state["rope_cache"]["table_entries"] > 0
    )
    feature_state_passed = head_active == expected_head and rope_active == expected_rope
    if args.expect_lm_head_denied:
        feature_state_passed = bool(
            feature_state_passed and not feature_state["lm_head"]["installed"]
        )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "finished_at": datetime.now(UTC).isoformat(),
        "profile": args.profile,
        "features": "none" if args.profile == "native" else args.features,
        "model": _model_identity(model_path),
        "git_revision": _git_revision(),
        "source": {"checkout_verified": True, "repository_root": "."},
        "hardware": {"machine": platform.machine(), "macos": platform.mac_ver()[0]},
        "runtime": {
            "vmlx": vmlx_engine.__version__,
            "jang": importlib.metadata.version("jang"),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
        },
        "controls": controls,
        "benchmark_controls": {
            "min_within_profile_decode_ratio": args.min_within_profile_decode_ratio,
            "min_anchor_decode_ratio": args.min_anchor_decode_ratio,
        },
        "feature_state": feature_state,
        "feature_state_passed": feature_state_passed,
        "telemetry": {
            "load_to_ready_seconds": load_seconds,
            "post_load_memory": post_load_memory,
            "final_memory": _memory_snapshot(mx),
            "expect_lm_head_denied": bool(args.expect_lm_head_denied),
        },
        "cases": [case.__dict__ | {"key": case.key} for case in cases],
        "results": results,
        "decode_floor_sanity": _decode_floor_sanity(
            results,
            min_anchor_ratio=args.min_anchor_decode_ratio,
        ),
    }
    report["passed"] = bool(
        feature_state_passed
        and results
        and all(result["passed"] for result in results.values())
        and report["decode_floor_sanity"]["passed"]
    )
    if args.baseline_report is not None:
        baseline = json.loads(args.baseline_report.read_text())
        report["baseline_compatibility"] = _baseline_compatibility(report, baseline)
        report["comparison"] = _compare(
            report,
            baseline,
            min_decode_ratio=args.min_decode_ratio,
            min_prefill_ratio=args.min_prefill_ratio,
            min_sample_decode_ratio=args.min_sample_decode_ratio,
        )
        if args.acceptance:
            report["passed"] = bool(
                report["passed"]
                and report["baseline_compatibility"]["passed"]
                and report["comparison"]["passed"]
            )

    _atomic_write(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {args.out}")
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
