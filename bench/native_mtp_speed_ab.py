#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("VMLINUX_BENCH_PYTHON", sys.executable))


DEFAULT_PROMPT = (
    "Count from 1 to 120, separated by commas. Do not write any words before "
    "or after the comma-separated numbers."
)


def request_json(
    method: str,
    url: str,
    body: Any | None = None,
    timeout: float = 300.0,
) -> tuple[Any, float]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw.decode("utf-8")), time.perf_counter() - t0


def wait_ready(base_url: str, deadline_s: float) -> dict[str, Any]:
    end = time.monotonic() + deadline_s
    last_error: str | None = None
    while time.monotonic() < end:
        try:
            health, _ = request_json("GET", f"{base_url}/health", timeout=5.0)
            if health.get("status") == "healthy" and health.get("model_loaded"):
                return health
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            last_error = repr(exc)
        time.sleep(1.0)
    raise TimeoutError(f"server did not become healthy before timeout: {last_error}")


def terminate_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.terminate()
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    proc.wait(timeout=10)


def log_tail(path: Path, limit: int = 120) -> list[str]:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return []
    return lines[-limit:]


def log_lines(path: Path) -> list[str]:
    try:
        return path.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return []


def coerce_native_mtp_depth(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        return max(1, min(3, int(raw)))
    except (TypeError, ValueError):
        return None


def effective_native_mtp_depth(
    depth_override: int | None,
    health: dict[str, Any] | None,
) -> tuple[int | None, str | None]:
    mtp = ((health or {}).get("mtp") or {})
    if depth_override is not None:
        return (
            coerce_native_mtp_depth(depth_override),
            mtp.get("effective_depth_source") or "bench_depth_override",
        )
    return (
        coerce_native_mtp_depth(mtp.get("effective_depth")),
        mtp.get("effective_depth_source"),
    )


def build_native_mtp_env(
    base_env: dict[str, str],
    *,
    mtp_enabled: bool,
    args: argparse.Namespace,
    depth_override: int | None = None,
) -> dict[str, str]:
    env = dict(base_env)
    enabled = "1" if mtp_enabled else "0"
    depth_raw = depth_override if depth_override is not None else getattr(args, "depth", None)
    depth = coerce_native_mtp_depth(depth_raw)

    env["PYTHONUNBUFFERED"] = "1"
    env["VMLINUX_NATIVE_MTP"] = enabled
    env["VMLX_NATIVE_MTP"] = enabled
    if depth is not None:
        env["VMLINUX_NATIVE_MTP_DEPTH"] = str(depth)
        env["VMLX_NATIVE_MTP_DEPTH"] = str(depth)
    env["VMLINUX_NATIVE_MTP_TRACE"] = (
        "1"
        if (
            bool(getattr(args, "trace_mtp", False))
            or (mtp_enabled and bool(getattr(args, "mtp_cost_fallback", False)))
        )
        else "0"
    )
    env["VMLINUX_NATIVE_MTP_DEBUG_TOKENS"] = (
        "1" if bool(getattr(args, "debug_mtp_tokens", False)) else "0"
    )
    env["VMLINUX_NATIVE_MTP_BURST"] = (
        "1" if bool(getattr(args, "mtp_burst", True)) else "0"
    )
    if mtp_enabled and bool(getattr(args, "mtp_cost_fallback", False)):
        env["VMLINUX_NATIVE_MTP_COST_FALLBACK"] = "1"
        ar_step_ms = getattr(args, "mtp_cost_ar_step_ms", None)
        if ar_step_ms is not None:
            env["VMLINUX_NATIVE_MTP_AR_STEP_MS"] = f"{float(ar_step_ms):.6f}"
        threshold = getattr(args, "mtp_cost_ratio_threshold", None)
        if threshold is not None:
            env["VMLINUX_NATIVE_MTP_COST_RATIO_THRESHOLD"] = f"{float(threshold):.6f}"
    return env


def calibrate_mtp_cost_fallback(
    args: argparse.Namespace,
    baseline_row: dict[str, Any],
) -> float | None:
    if not bool(getattr(args, "mtp_cost_fallback", False)):
        return None
    explicit = getattr(args, "mtp_cost_ar_step_ms", None)
    if explicit is not None:
        try:
            explicit_value = float(explicit)
        except (TypeError, ValueError):
            explicit_value = 0.0
        if explicit_value > 0.0:
            args.mtp_cost_ar_step_ms = explicit_value
            return explicit_value
    mean_tps = ((baseline_row.get("summary") or {}).get("mean_wall_tok_s"))
    try:
        mean_tps_value = float(mean_tps)
    except (TypeError, ValueError):
        return None
    if mean_tps_value <= 0.0:
        return None
    ar_step_ms = 1000.0 / mean_tps_value
    args.mtp_cost_ar_step_ms = ar_step_ms
    return ar_step_ms


def parse_depth_sweep(raw: str | None) -> list[int]:
    if raw is None:
        return []
    depths: list[int] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        depth = coerce_native_mtp_depth(item)
        if depth is None:
            continue
        if depth not in depths:
            depths.append(depth)
    return depths


_MTP_FINISH_RE = re.compile(
    r"MLLM MTP\[(?P<request>[^\]]+)\] finish=(?P<finish>\S+) "
    r"cycles=(?P<cycles>\d+) accepted=(?P<accepted>\d+)/(?P<drafted>\d+) "
    r"\((?P<percent>[0-9.]+)%\) "
    r"emits\[init=(?P<init>\d+),draft=(?P<draft>\d+),"
    r"bonus=(?P<bonus>\d+),verify=(?P<verify>\d+)\]"
)
_MTP_TIMINGS_RE = re.compile(
    r"MLLM MTP\[(?P<request>[^\]]+)\] timings_ms\[(?P<body>[^\]]+)\]"
)
_MTP_DEPTH_RE = re.compile(
    r"MLLM MTP\[(?P<request>[^\]]+)\] "
    r"accept_by_depth\[d1=(?P<a1>\d+)/(?P<d1>\d+),"
    r"d2=(?P<a2>\d+)/(?P<d2>\d+),d3=(?P<a3>\d+)/(?P<d3>\d+)\] "
    r"forwards\[seed_main=(?P<seed>\d+),verify_main=(?P<verify>\d+),"
    r"replay_main=(?P<replay>\d+),mtp=(?P<mtp>\d+)\]"
)


def _float_map(body: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in body.split():
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        try:
            values[key] = float(raw.rstrip(","))
        except ValueError:
            continue
    return values


def _rates(accepted: list[int], drafted: list[int]) -> list[float | None]:
    return [
        (a / d if d else None)
        for a, d in zip(accepted, drafted)
    ]


def _sum_lists(rows: list[list[int]], size: int = 3) -> list[int]:
    totals = [0] * size
    for row in rows:
        for idx, value in enumerate(row[:size]):
            totals[idx] += int(value)
    return totals


def _sum_float_maps(rows: list[dict[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            if key == "avg_cycle":
                continue
            totals[key] = totals.get(key, 0.0) + float(value)
    if totals:
        totals["trace_phase_total"] = sum(totals.values())
        for key, value in list(totals.items()):
            totals[key] = round(value, 2)
    return totals


def parse_mtp_log(lines: list[str]) -> dict[str, Any]:
    requests: dict[str, dict[str, Any]] = {}
    for line in lines:
        finish = _MTP_FINISH_RE.search(line)
        if finish:
            request_id = finish.group("request")
            accepted = int(finish.group("accepted"))
            drafted = int(finish.group("drafted"))
            row = requests.setdefault(request_id, {})
            row.update(
                {
                    "finish": finish.group("finish"),
                    "cycles": int(finish.group("cycles")),
                    "accepted_tokens": accepted,
                    "drafted_tokens": drafted,
                    "acceptance_rate": accepted / drafted if drafted else None,
                    "emits": {
                        "init": int(finish.group("init")),
                        "draft": int(finish.group("draft")),
                        "bonus": int(finish.group("bonus")),
                        "verify": int(finish.group("verify")),
                    },
                }
            )
            continue

        timings = _MTP_TIMINGS_RE.search(line)
        if timings:
            request_id = timings.group("request")
            row = requests.setdefault(request_id, {})
            row["timings_ms"] = _float_map(timings.group("body"))
            continue

        depth = _MTP_DEPTH_RE.search(line)
        if depth:
            request_id = depth.group("request")
            row = requests.setdefault(request_id, {})
            accepted_by_depth = [
                int(depth.group("a1")),
                int(depth.group("a2")),
                int(depth.group("a3")),
            ]
            drafted_by_depth = [
                int(depth.group("d1")),
                int(depth.group("d2")),
                int(depth.group("d3")),
            ]
            row["accepted_by_depth"] = accepted_by_depth
            row["drafted_by_depth"] = drafted_by_depth
            row["acceptance_by_depth"] = _rates(accepted_by_depth, drafted_by_depth)
            row["forwards"] = {
                "seed_main": int(depth.group("seed")),
                "verify_main": int(depth.group("verify")),
                "replay_main": int(depth.group("replay")),
                "mtp": int(depth.group("mtp")),
            }

    total_cycles = sum(int(row.get("cycles") or 0) for row in requests.values())
    total_accepted = sum(int(row.get("accepted_tokens") or 0) for row in requests.values())
    total_drafted = sum(int(row.get("drafted_tokens") or 0) for row in requests.values())
    accepted_by_depth = _sum_lists(
        [row.get("accepted_by_depth") or [] for row in requests.values()]
    )
    drafted_by_depth = _sum_lists(
        [row.get("drafted_by_depth") or [] for row in requests.values()]
    )
    timings_ms = _sum_float_maps(
        [row.get("timings_ms") or {} for row in requests.values()]
    )
    forward_totals = {
        "seed_main": sum(int((row.get("forwards") or {}).get("seed_main") or 0) for row in requests.values()),
        "verify_main": sum(int((row.get("forwards") or {}).get("verify_main") or 0) for row in requests.values()),
        "replay_main": sum(int((row.get("forwards") or {}).get("replay_main") or 0) for row in requests.values()),
        "mtp": sum(int((row.get("forwards") or {}).get("mtp") or 0) for row in requests.values()),
    }
    return {
        "requests": requests,
        "totals": {
            "cycles": total_cycles,
            "accepted_tokens": total_accepted,
            "drafted_tokens": total_drafted,
            "acceptance_rate": (
                total_accepted / total_drafted if total_drafted else None
            ),
            "accepted_by_depth": accepted_by_depth,
            "drafted_by_depth": drafted_by_depth,
            "acceptance_by_depth": _rates(accepted_by_depth, drafted_by_depth),
            "forwards": forward_totals,
            "timings_ms": timings_ms,
        },
    }


def hot_path_accounting(row: dict[str, Any], *, warmup: int) -> dict[str, Any]:
    """Separate full request wall time from scheduler and traced MTP phases."""
    generations = row.get("generations") or []
    wall_ms = sum(float(gen.get("elapsed_sec") or 0.0) for gen in generations) * 1000.0
    completion_tokens = sum(int(gen.get("completion_tokens") or 0) for gen in generations)

    batch_stats = (
        ((row.get("health_after") or {}).get("scheduler") or {}).get("batch_generator")
        or {}
    )
    scheduler_ms = None
    if batch_stats.get("generation_time") is not None:
        scheduler_ms = float(batch_stats.get("generation_time") or 0.0) * 1000.0

    mtp_totals = ((row.get("mtp_stats") or {}).get("totals") or {})
    timings = mtp_totals.get("timings_ms") or {}
    trace_ms = timings.get("trace_phase_total")
    if trace_ms is None and timings:
        trace_ms = sum(
            float(value)
            for key, value in timings.items()
            if key not in {"avg_cycle", "trace_phase_total"}
        )

    cycles = int(mtp_totals.get("cycles") or 0)
    accepted = int(mtp_totals.get("accepted_tokens") or 0)
    drafted = int(mtp_totals.get("drafted_tokens") or 0)

    def _per_cycle(name: str) -> float | None:
        if cycles <= 0:
            return None
        if timings.get(name) is None:
            return None
        return round(float(timings[name]) / cycles, 2)

    accounting: dict[str, Any] = {
        "wall_ms": round(wall_ms, 2),
        "completion_tokens": completion_tokens,
        "scheduler_generation_ms": (
            round(scheduler_ms, 2) if scheduler_ms is not None else None
        ),
        "trace_phase_ms": round(float(trace_ms), 2) if trace_ms is not None else None,
        "scheduler_generation_includes_warmup": bool(int(warmup or 0) > 0),
        "cycles": cycles,
        "accepted_tokens": accepted,
        "drafted_tokens": drafted,
        "verify_ms_per_cycle": _per_cycle("verify"),
        "draft_ms_per_cycle": _per_cycle("draft"),
        "accepted_tokens_per_cycle": (
            round(accepted / cycles, 2) if cycles > 0 else None
        ),
    }
    if scheduler_ms is not None:
        accounting["wall_minus_scheduler_ms"] = round(wall_ms - scheduler_ms, 2)
    else:
        accounting["wall_minus_scheduler_ms"] = None
    if scheduler_ms is not None and trace_ms is not None:
        accounting["scheduler_minus_trace_ms"] = round(
            scheduler_ms - float(trace_ms), 2
        )
    else:
        accounting["scheduler_minus_trace_ms"] = None
    return accounting


def _full_generation_text(generation: dict[str, Any]) -> str:
    return (generation.get("reasoning_content") or "") + (
        generation.get("content") or ""
    )


def summarize_repeat_output_equivalence(result: dict[str, Any]) -> dict[str, Any]:
    """Compare repeats inside each row so cache-hit drift is not hidden."""
    row_summaries: list[dict[str, Any]] = []
    for row in result.get("rows") or []:
        generations = row.get("generations") or []
        comparisons: list[dict[str, Any]] = []
        if len(generations) >= 2:
            reference = generations[0]
            for index, repeat in enumerate(generations[1:], start=1):
                token_count_equal = (
                    reference.get("completion_tokens") == repeat.get("completion_tokens")
                )
                finish_reason_equal = (
                    reference.get("finish_reason") == repeat.get("finish_reason")
                )
                comparisons.append(
                    {
                        "index": index,
                        "reference_index": 0,
                        "content_equal": reference.get("content")
                        == repeat.get("content"),
                        "reasoning_equal": reference.get("reasoning_content")
                        == repeat.get("reasoning_content"),
                        "full_text_equal": _full_generation_text(reference)
                        == _full_generation_text(repeat),
                        "reference_tokens": reference.get("completion_tokens"),
                        "repeat_tokens": repeat.get("completion_tokens"),
                        "token_count_equal": token_count_equal,
                        "reference_finish": reference.get("finish_reason"),
                        "repeat_finish": repeat.get("finish_reason"),
                        "finish_reason_equal": finish_reason_equal,
                    }
                )
        has_comparisons = bool(comparisons)
        row_summaries.append(
            {
                "label": row.get("label"),
                "depth": row.get("native_mtp_depth"),
                "has_repeat_comparisons": has_comparisons,
                "all_content_equal": has_comparisons
                and all(item["content_equal"] for item in comparisons),
                "all_reasoning_equal": has_comparisons
                and all(item["reasoning_equal"] for item in comparisons),
                "all_full_text_equal": has_comparisons
                and all(item["full_text_equal"] for item in comparisons),
                "all_token_count_equal": has_comparisons
                and all(item["token_count_equal"] for item in comparisons),
                "all_finish_reason_equal": has_comparisons
                and all(item["finish_reason_equal"] for item in comparisons),
                "comparisons": comparisons,
            }
        )

    rows_with_repeats = [
        row for row in row_summaries if row["has_repeat_comparisons"]
    ]
    has_repeat_comparisons = bool(rows_with_repeats)
    return {
        "has_repeat_comparisons": has_repeat_comparisons,
        "all_rows_content_equal": has_repeat_comparisons
        and all(row["all_content_equal"] for row in rows_with_repeats),
        "all_rows_reasoning_equal": has_repeat_comparisons
        and all(row["all_reasoning_equal"] for row in rows_with_repeats),
        "all_rows_full_text_equal": has_repeat_comparisons
        and all(row["all_full_text_equal"] for row in rows_with_repeats),
        "all_rows_token_count_equal": has_repeat_comparisons
        and all(row["all_token_count_equal"] for row in rows_with_repeats),
        "all_rows_finish_reason_equal": has_repeat_comparisons
        and all(row["all_finish_reason_equal"] for row in rows_with_repeats),
        "row_summaries": row_summaries,
    }


def select_best_depth(result: dict[str, Any]) -> dict[str, Any]:
    """Pick the fastest output-equivalent native-MTP depth from a result."""
    rows = result.get("rows") or []
    baseline_tps = None
    for row in rows:
        if row.get("label") == "baseline_no_mtp":
            baseline_tps = (row.get("summary") or {}).get("mean_wall_tok_s")
            break

    equivalent_by_label: dict[str, bool] = {}
    comparisons = ((result.get("output_equivalence") or {}).get("comparisons") or [])
    for comparison in comparisons:
        label = comparison.get("label")
        if not label:
            continue
        is_equivalent = bool(comparison.get("full_text_equal"))
        if comparison.get("baseline_tokens") != comparison.get("mtp_tokens"):
            is_equivalent = False
        equivalent_by_label[label] = equivalent_by_label.get(label, True) and is_equivalent

    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    candidate_count = 0
    for row in rows:
        depth = row.get("native_mtp_depth")
        if depth is None:
            continue
        candidate_count += 1
        label = str(row.get("label") or f"native_mtp_d{depth}")
        if equivalent_by_label and not equivalent_by_label.get(label, False):
            rejected.append(
                {
                    "depth": int(depth),
                    "label": label,
                    "reason": "output_not_equivalent",
                }
            )
            continue
        tps = (row.get("summary") or {}).get("mean_wall_tok_s")
        if tps is None:
            rejected.append(
                {
                    "depth": int(depth),
                    "label": label,
                    "reason": "missing_tps",
                }
            )
            continue
        candidates.append(
            {
                "depth": int(depth),
                "label": label,
                "mean_wall_tok_s": float(tps),
            }
        )

    selected = max(candidates, key=lambda item: item["mean_wall_tok_s"], default=None)
    if selected is None:
        return {
            "best_depth": None,
            "best_label": None,
            "best_mean_wall_tok_s": None,
            "speedup_vs_baseline": None,
            "candidate_count": candidate_count,
            "rejected_depths": rejected,
        }

    speedup = None
    if baseline_tps:
        speedup = selected["mean_wall_tok_s"] / float(baseline_tps)
    return {
        "best_depth": selected["depth"],
        "best_label": selected["label"],
        "best_mean_wall_tok_s": selected["mean_wall_tok_s"],
        "speedup_vs_baseline": speedup,
        "candidate_count": candidate_count,
        "rejected_depths": rejected,
    }


def cache_mode_cli_args(
    cache_mode: str,
    *,
    block_cache_dir: Path,
    kv_cache_quantization: str = "auto",
) -> list[str]:
    if cache_mode == "off":
        return ["--disable-prefix-cache"]
    if cache_mode != "on":
        raise ValueError(f"unknown cache mode: {cache_mode}")

    args = [
        "--use-paged-cache",
        "--enable-block-disk-cache",
        "--block-disk-cache-dir",
        str(block_cache_dir),
        "--block-disk-cache-max-gb",
        "2",
        "--ssm-state-cache-mb",
        "1024",
    ]
    quant = (kv_cache_quantization or "auto").strip().lower()
    if quant not in {"auto", "none", "q4", "q8"}:
        raise ValueError(f"unknown kv cache quantization: {kv_cache_quantization}")
    if quant not in {"auto", "none"}:
        args.extend(["--kv-cache-quantization", quant])
    return args


def chat_template_kwargs_for_enable_thinking(
    enable_thinking: str,
) -> dict[str, bool] | None:
    mode = (enable_thinking or "off").strip().lower()
    if mode == "off":
        return {"enable_thinking": False}
    if mode == "on":
        return {"enable_thinking": True}
    if mode == "auto":
        return None
    raise ValueError(f"unknown enable_thinking mode: {enable_thinking}")


def build_chat_completion_body(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    repeat_index: int,
    disable_prompt_reuse: bool,
    enable_thinking: str,
) -> dict[str, Any]:
    content_prompt = prompt
    if disable_prompt_reuse:
        content_prompt = f"{prompt}\nRun nonce: {repeat_index}."
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content_prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    chat_template_kwargs = chat_template_kwargs_for_enable_thinking(enable_thinking)
    if chat_template_kwargs is not None:
        body["chat_template_kwargs"] = chat_template_kwargs
    return body


def start_server(
    *,
    model_path: Path,
    served_name: str,
    port: int,
    mtp_enabled: bool,
    cache_mode: str,
    block_cache_dir: Path,
    log_path: Path,
    max_num_seqs: int,
    load_timeout_s: float,
    enable_jit: bool,
    args: argparse.Namespace,
    depth_override: int | None = None,
) -> tuple[subprocess.Popen[Any], dict[str, Any]]:
    env = build_native_mtp_env(
        os.environ.copy(),
        mtp_enabled=mtp_enabled,
        args=args,
        depth_override=depth_override,
    )

    cmd = [
        str(PYTHON),
        "-m",
        "vmlx_engine.cli",
        "serve",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        served_name,
        "--is-mllm",
        "--max-num-seqs",
        str(max_num_seqs),
        "--max-tokens",
        "512",
        "--log-level",
        "INFO",
    ]
    if enable_jit:
        cmd.append("--enable-jit")
    cmd.extend(
        cache_mode_cli_args(
            cache_mode,
            block_cache_dir=block_cache_dir,
            kv_cache_quantization=getattr(args, "kv_cache_quantization", "q4"),
        )
    )

    log_file = log_path.open("w")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_file.close()
    try:
        health = wait_ready(f"http://127.0.0.1:{port}", load_timeout_s)
        return proc, health
    except Exception:
        terminate_process(proc)
        tail = "\n".join(log_tail(log_path, 80))
        raise RuntimeError(f"server failed for mtp={mtp_enabled}; log tail:\n{tail}")


def run_generation(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    repeat_index: int,
    disable_prompt_reuse: bool,
    enable_thinking: str,
) -> dict[str, Any]:
    body = build_chat_completion_body(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        repeat_index=repeat_index,
        disable_prompt_reuse=disable_prompt_reuse,
        enable_thinking=enable_thinking,
    )
    resp, elapsed = request_json("POST", f"{base_url}/v1/chat/completions", body)
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = resp.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    content = message.get("content") or ""
    reasoning_content = (
        message.get("reasoning_content")
        or message.get("reasoning")
        or message.get("reasoning_text")
        or ""
    )
    return {
        "elapsed_sec": elapsed,
        "completion_tokens": completion_tokens,
        "wall_tok_s": completion_tokens / elapsed if elapsed > 0 else None,
        "finish_reason": choice.get("finish_reason"),
        "content": content,
        "content_head": content[:320],
        "content_tail": content[-320:],
        "reasoning_content": reasoning_content,
        "reasoning_head": reasoning_content[:320],
        "reasoning_tail": reasoning_content[-320:],
        "usage": usage,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rates = [float(r["wall_tok_s"]) for r in rows if r.get("wall_tok_s")]
    tokens = [int(r.get("completion_tokens") or 0) for r in rows]
    return {
        "runs": len(rows),
        "tokens": tokens,
        "mean_wall_tok_s": statistics.fmean(rates) if rates else None,
        "median_wall_tok_s": statistics.median(rates) if rates else None,
        "min_wall_tok_s": min(rates) if rates else None,
        "max_wall_tok_s": max(rates) if rates else None,
    }


def run_row(
    *,
    label: str,
    mtp_enabled: bool,
    args: argparse.Namespace,
    out_dir: Path,
    port: int,
    depth_override: int | None = None,
) -> dict[str, Any]:
    model_path = Path(args.model_path)
    served_name = args.served_name or model_path.name.lower().replace("/", "-")
    block_cache_dir = out_dir / f"{label}_block_cache"
    log_path = out_dir / f"{label}.server.log"
    proc, health_before = start_server(
        model_path=model_path,
        served_name=served_name,
        port=port,
        mtp_enabled=mtp_enabled,
        cache_mode=args.cache,
        block_cache_dir=block_cache_dir,
        log_path=log_path,
        max_num_seqs=args.max_num_seqs,
        load_timeout_s=args.load_timeout_s,
        enable_jit=args.enable_jit,
        args=args,
        depth_override=depth_override,
    )
    rows: list[dict[str, Any]] = []
    base_url = f"http://127.0.0.1:{port}"
    try:
        for i in range(args.warmup):
            run_generation(
                base_url=base_url,
                model=served_name,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                repeat_index=-(i + 1),
                disable_prompt_reuse=args.disable_prompt_reuse,
                enable_thinking=args.enable_thinking,
            )
        for i in range(args.repeats):
            rows.append(
                run_generation(
                    base_url=base_url,
                    model=served_name,
                    prompt=args.prompt,
                    max_tokens=args.max_tokens,
                    repeat_index=i,
                    disable_prompt_reuse=args.disable_prompt_reuse,
                    enable_thinking=args.enable_thinking,
                )
            )
        health_after, _ = request_json("GET", f"{base_url}/health", timeout=30.0)
        cache_stats, _ = request_json("GET", f"{base_url}/v1/cache/stats", timeout=30.0)
    finally:
        terminate_process(proc)

    effective_depth, effective_depth_source = effective_native_mtp_depth(
        depth_override if mtp_enabled else None,
        health_before,
    )
    full_log = log_lines(log_path)
    tail = full_log[-args.log_tail_lines:]
    mtp_lines = [
        line for line in full_log if "MLLM MTP[" in line or "native MTP" in line
    ]
    result = {
        "label": label,
        "mtp_enabled": mtp_enabled,
        "native_mtp_depth": effective_depth if mtp_enabled else None,
        "native_mtp_depth_override": depth_override if mtp_enabled else None,
        "native_mtp_depth_source": effective_depth_source if mtp_enabled else None,
        "cache_mode": args.cache,
        "port": port,
        "health_before": health_before,
        "health_after": health_after,
        "cache_stats": cache_stats,
        "generations": rows,
        "summary": summarize_rows(rows),
        "server_log": str(log_path),
        "mtp_stats": parse_mtp_log(full_log),
        "mtp_log_lines_tail": mtp_lines[-30:],
    }
    result["hot_path_accounting"] = hot_path_accounting(
        result,
        warmup=int(getattr(args, "warmup", 0) or 0),
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Live server A/B for native MTP vs autoregressive baseline on the "
            "same vMLX model artifact."
        )
    )
    ap.add_argument("model_path")
    ap.add_argument("--served-name", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--port", type=int, default=8130)
    ap.add_argument("--cache", choices=["off", "on"], default="off")
    ap.add_argument("--max-num-seqs", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument(
        "--kv-cache-quantization",
        choices=["auto", "none", "q4", "q8"],
        default="auto",
        help=(
            "Storage-boundary KV quantization used by cache-on rows. "
            "auto omits the CLI flag so the engine chooses per architecture; "
            "use q4/q8 for explicit lossy storage-codec stress rows."
        ),
    )
    ap.add_argument(
        "--depth",
        type=int,
        default=None,
        help=(
            "Explicit native-MTP depth to benchmark. When omitted without "
            "--depth-sweep, the server uses model-local tuning or its runtime "
            "default and the harness records the effective /health depth."
        ),
    )
    ap.add_argument(
        "--depth-sweep",
        default=None,
        help=(
            "Comma-separated native-MTP depths to run after the AR baseline, "
            "for example 1,2,3. When omitted, only --depth is tested."
        ),
    )
    ap.add_argument("--trace-mtp", action="store_true")
    ap.add_argument("--debug-mtp-tokens", action="store_true")
    ap.add_argument(
        "--mtp-cost-fallback",
        action="store_true",
        help=(
            "Research flag: enable measured MTP->AR fallback using traced MTP "
            "cost and an AR-step calibration from the baseline row."
        ),
    )
    ap.add_argument(
        "--mtp-cost-ar-step-ms",
        type=float,
        default=None,
        help="Override AR ms/token calibration used by --mtp-cost-fallback.",
    )
    ap.add_argument(
        "--mtp-cost-ratio-threshold",
        type=float,
        default=1.0,
        help="Disable MTP for the request when measured MTP cost ratio reaches this value.",
    )
    ap.add_argument(
        "--no-mtp-burst",
        action="store_false",
        dest="mtp_burst",
        help="Disable verified-token burst draining while benchmarking MTP.",
    )
    ap.set_defaults(mtp_burst=True)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument(
        "--enable-thinking",
        choices=["off", "on", "auto"],
        default="off",
        help=(
            "Chat-template thinking rail for benchmark requests. off is the "
            "legacy deterministic text gate, on exercises reasoning_content, "
            "and auto omits chat_template_kwargs so model config/template "
            "defaults decide."
        ),
    )
    ap.add_argument(
        "--disable-prompt-reuse",
        action="store_true",
        help="Append a nonce to each request so cache-on rows measure fresh prefill/decode.",
    )
    ap.add_argument("--load-timeout-s", type=float, default=600.0)
    ap.add_argument("--log-tail-lines", type=int, default=240)
    ap.add_argument("--enable-jit", action="store_true")
    args = ap.parse_args()

    if not PYTHON.exists():
        raise SystemExit(f"missing venv python: {PYTHON}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out)
        if args.out
        else ROOT / "docs/internal/release-gates" / f"{stamp}_native_mtp_speed_ab"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    depth_sweep = parse_depth_sweep(args.depth_sweep)
    explicit_single_depth = coerce_native_mtp_depth(args.depth)
    if not depth_sweep and explicit_single_depth is not None:
        depth_sweep = [explicit_single_depth]

    result: dict[str, Any] = {
        "created_at": stamp,
        "model_path": str(Path(args.model_path)),
        "served_name": args.served_name or Path(args.model_path).name.lower().replace("/", "-"),
        "cache_mode": args.cache,
        "kv_cache_quantization": args.kv_cache_quantization,
        "enable_jit": args.enable_jit,
        "native_mtp_depth": explicit_single_depth,
        "native_mtp_depths": depth_sweep,
        "native_mtp_depth_policy": (
            "explicit_sweep"
            if args.depth_sweep is not None
            else ("explicit_single" if explicit_single_depth is not None else "model_tuning_or_runtime_default")
        ),
        "depth_sweep": args.depth_sweep,
        "trace_mtp": args.trace_mtp,
        "debug_mtp_tokens": args.debug_mtp_tokens,
        "mtp_cost_fallback": args.mtp_cost_fallback,
        "mtp_cost_ar_step_ms": args.mtp_cost_ar_step_ms,
        "mtp_cost_ratio_threshold": args.mtp_cost_ratio_threshold,
        "mtp_burst": args.mtp_burst,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "enable_thinking": args.enable_thinking,
        "prompt": args.prompt,
        "rows": [],
    }

    rows: list[tuple[str, bool, int, int | None]] = [
        ("baseline_no_mtp", False, args.port, None),
    ]
    if args.depth_sweep is None:
        rows.append(("native_mtp", True, args.port + 1, explicit_single_depth))
    else:
        for offset, depth in enumerate(depth_sweep, start=1):
            rows.append((f"native_mtp_d{depth}", True, args.port + offset, depth))

    for label, enabled, port, depth_override in rows:
        row = run_row(
            label=label,
            mtp_enabled=enabled,
            args=args,
            out_dir=out_dir,
            port=port,
            depth_override=depth_override,
        )
        result["rows"].append(row)
        if label == "baseline_no_mtp":
            ar_step_ms = calibrate_mtp_cost_fallback(args, row)
            result["mtp_cost_ar_step_ms"] = ar_step_ms
        (out_dir / "partial.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(json.dumps({"label": label, "summary": row["summary"]}, indent=2), flush=True)

    if len(result["rows"]) >= 2:
        base = result["rows"][0]["summary"].get("mean_wall_tok_s")
        base_generations = result["rows"][0].get("generations", [])
        comparisons = []
        for mtp_row in result["rows"][1:]:
            mtp = mtp_row["summary"].get("mean_wall_tok_s")
            if base and mtp:
                mtp_row["speedup_vs_baseline"] = mtp / base
            mtp_generations = mtp_row.get("generations", [])
            for idx, (base_gen, mtp_gen) in enumerate(
                zip(base_generations, mtp_generations)
            ):
                comparisons.append(
                    {
                        "index": idx,
                        "label": mtp_row.get("label"),
                        "depth": mtp_row.get("native_mtp_depth"),
                        "content_equal": base_gen.get("content") == mtp_gen.get("content"),
                        "reasoning_equal": base_gen.get("reasoning_content") == mtp_gen.get("reasoning_content"),
                        "full_text_equal": (
                            (base_gen.get("reasoning_content") or "")
                            + (base_gen.get("content") or "")
                        )
                        == (
                            (mtp_gen.get("reasoning_content") or "")
                            + (mtp_gen.get("content") or "")
                        ),
                        "baseline_tokens": base_gen.get("completion_tokens"),
                        "mtp_tokens": mtp_gen.get("completion_tokens"),
                        "baseline_finish": base_gen.get("finish_reason"),
                        "mtp_finish": mtp_gen.get("finish_reason"),
                    }
                )
        result["output_equivalence"] = {
            "all_content_equal": bool(comparisons) and all(
                row["content_equal"] for row in comparisons
            ),
            "all_full_text_equal": bool(comparisons) and all(
                row["full_text_equal"] for row in comparisons
            ),
            "comparisons": comparisons,
        }
        result["best_native_mtp_depth"] = select_best_depth(result)
        if args.depth_sweep is None and result["rows"][1].get("summary"):
            result["speedup_vs_baseline"] = result["rows"][1].get("speedup_vs_baseline")
    result["repeat_output_equivalence"] = summarize_repeat_output_equivalence(result)

    out_path = out_dir / "result.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
