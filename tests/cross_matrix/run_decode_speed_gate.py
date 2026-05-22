#!/usr/bin/env python3
"""Live M5 PP/decode speed and coherency gate for local vMLX model rows.

This is intentionally a live harness: it starts the bundled vMLX engine for one
model at a time, sends real OpenAI-compatible chat requests, records generated
text, finish reason, cache/acceleration health, prompt-processing wall tok/s,
decode wall tok/s, and loop heuristics, then stops the server before the next
row. It does not pass any JANGTQ acceleration flag; packaged-app behavior should
come from the engine's default auto policy.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_PY = (
    REPO
    / "panel/release/mac-arm64/vMLX.app/Contents/Resources/bundled-python/python/bin/python3"
)
DEFAULT_OUT = REPO / "docs/internal/release-gates/decode_speed_gate_latest.json"
DSV4_AFFINE_MODEL_CANDIDATES = (
    "/Users/example/models/JANGQ/"
    "DeepSeek-V4-Flash-JANG_DQ2-Token8-DownG32-Gate3Math6-NoMTP",
    "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANG",
)


def resolve_default_model(candidates: tuple[str, ...] = DSV4_AFFINE_MODEL_CANDIDATES) -> str:
    for candidate in candidates:
        if Path(candidate).is_dir():
            return candidate
    return candidates[0]


@dataclass
class Row:
    name: str
    path: str
    is_mllm: bool = False
    tool_parser: str | None = None
    reasoning_parser: str | None = None
    max_tokens: int = 320
    expected_min_tps: float | None = None
    expected_min_pp: float | None = None
    pp_targets: list[int] = field(default_factory=lambda: [1024, 4096, 16384])
    extra_args: list[str] = field(default_factory=list)


ROWS: dict[str, Row] = {
    "minimax": Row(
        "minimax",
        "/Users/example/models/dealign.ai/MiniMax-M2.7-JANGTQ-CRACK",
        tool_parser="minimax",
        reasoning_parser="minimax_m2",
        expected_min_tps=40.0,
        expected_min_pp=600.0,
    ),
    "minimax_k": Row(
        "minimax_k",
        "/Users/example/models/dealign.ai/MiniMax-M2.7-JANGTQ_K-CRACK",
        tool_parser="minimax",
        reasoning_parser="minimax_m2",
        expected_min_tps=40.0,
        expected_min_pp=600.0,
    ),
    "minimax_jang2l_crack": Row(
        "minimax_jang2l_crack",
        "/Users/example/.cache/huggingface/hub/dealignai/MiniMax-M2.7-JANG_2L-CRACK",
        tool_parser="minimax",
        reasoning_parser="minimax_m2",
        expected_min_tps=40.0,
        expected_min_pp=600.0,
    ),
    "zaya_text_k": Row(
        "zaya_text_k",
        "/Users/example/models/JANGQ/ZAYA1-8B-JANGTQ_K",
        reasoning_parser="qwen3",
        expected_min_tps=45.0,
        expected_min_pp=1000.0,
    ),
    "zaya_vl_jangtq4": Row(
        "zaya_vl_jangtq4",
        "/Users/example/models/JANGQ/ZAYA1-VL-8B-JANGTQ4",
        is_mllm=True,
        reasoning_parser="qwen3",
        expected_min_tps=35.0,
        expected_min_pp=1000.0,
    ),
    "zaya_vl_k": Row(
        "zaya_vl_k",
        "/Users/example/models/JANGQ/ZAYA1-VL-8B-JANGTQ_K",
        is_mllm=True,
        reasoning_parser="qwen3",
        expected_min_tps=35.0,
        expected_min_pp=1000.0,
    ),
    "zaya_text_mxfp4": Row(
        "zaya_text_mxfp4",
        "/Users/example/models/JANGQ/ZAYA1-8B-MXFP4",
        reasoning_parser="qwen3",
        expected_min_tps=35.0,
        expected_min_pp=600.0,
    ),
    "zaya_vl_mxfp4": Row(
        "zaya_vl_mxfp4",
        "/Users/example/models/JANGQ/ZAYA1-VL-8B-MXFP4",
        is_mllm=True,
        reasoning_parser="qwen3",
        expected_min_tps=30.0,
        expected_min_pp=600.0,
    ),
    "ling": Row(
        "ling",
        "/Users/example/models/JANGQ/Ling-2.6-flash-JANGTQ",
        expected_min_tps=30.0,
        expected_min_pp=400.0,
    ),
    "ling_crack": Row(
        "ling_crack",
        "/Users/example/models/dealign.ai/Ling-2.6-flash-JANGTQ2-CRACK",
        expected_min_tps=30.0,
        expected_min_pp=400.0,
    ),
    "ling_mxfp4": Row(
        "ling_mxfp4",
        "/Users/example/models/dealign.ai/Ling-2.6-flash-MXFP4-CRACK",
        expected_min_tps=25.0,
        expected_min_pp=400.0,
    ),
    "qwen27_jang4m": Row(
        "qwen27_jang4m",
        "/Users/example/models/dealign.ai/Qwen3.6-27B-JANG_4M-CRACK",
        is_mllm=True,
        tool_parser="qwen",
        reasoning_parser="qwen3",
        expected_min_tps=18.0,
        expected_min_pp=600.0,
    ),
    "qwen27_mxfp4": Row(
        "qwen27_mxfp4",
        "/Users/example/models/dealign.ai/Qwen3.6-27B-MXFP4-CRACK",
        is_mllm=True,
        tool_parser="qwen",
        reasoning_parser="qwen3",
        expected_min_tps=25.0,
        expected_min_pp=600.0,
    ),
    "qwen27_mxfp8_mtp": Row(
        "qwen27_mxfp8_mtp",
        "/Users/example/models/JANGQ/Qwen3.6-27B-MXFP8-MTP",
        is_mllm=True,
        tool_parser="qwen",
        reasoning_parser="qwen3",
        expected_min_tps=25.0,
        expected_min_pp=600.0,
    ),
    "qwen27_jang4m_mtp": Row(
        "qwen27_jang4m_mtp",
        "/Users/example/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP",
        is_mllm=True,
        tool_parser="qwen",
        reasoning_parser="qwen3",
        expected_min_tps=18.0,
        expected_min_pp=600.0,
    ),
    "qwen35_jangtq": Row(
        "qwen35_jangtq",
        "/Users/example/models/dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK",
        tool_parser="qwen",
        reasoning_parser="qwen3",
        expected_min_tps=35.0,
        expected_min_pp=1000.0,
    ),
    "qwen35_4bit": Row(
        "qwen35_4bit",
        "/Users/example/models/Qwen3.6-35B-A3B-4bit",
        tool_parser="qwen",
        reasoning_parser="qwen3",
        expected_min_tps=30.0,
        expected_min_pp=600.0,
    ),
    "qwen35_jang4k_ext": Row(
        "qwen35_jang4k_ext",
        "/Volumes/ModelDrive/jangq-ai/Qwen3.5-35B-A3B-JANG_4K",
        tool_parser="qwen",
        reasoning_parser="qwen3",
        expected_min_tps=30.0,
        expected_min_pp=600.0,
    ),
    "qwen36_35_jangtq4_ext": Row(
        "qwen36_35_jangtq4_ext",
        "/Volumes/ModelDrive/jangq-ai/Qwen3.6-35B-A3B-JANGTQ4",
        tool_parser="qwen",
        reasoning_parser="qwen3",
        expected_min_tps=35.0,
        expected_min_pp=1000.0,
    ),
    "qwen35_mxfp8_mtp": Row(
        "qwen35_mxfp8_mtp",
        "/Users/example/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP",
        is_mllm=True,
        tool_parser="qwen",
        reasoning_parser="qwen3",
        expected_min_tps=25.0,
        expected_min_pp=600.0,
    ),
    "gemma4": Row(
        "gemma4",
        "/Users/example/models/dealign.ai/Gemma-4-26B-A4B-it-JANG_4M-CRACK",
        is_mllm=True,
        tool_parser="gemma4",
        reasoning_parser="qwen3",
        expected_min_tps=15.0,
        expected_min_pp=600.0,
    ),
    "hy3": Row(
        "hy3",
        "/Users/example/models/JANGQ/Hy3-preview-JANGTQ2",
        tool_parser="hunyuan",
        reasoning_parser="qwen3",
        max_tokens=192,
        expected_min_tps=18.0,
        expected_min_pp=400.0,
    ),
    "nemotron_jangtq": Row(
        "nemotron_jangtq",
        "/Users/example/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK",
        tool_parser="nemotron",
        reasoning_parser="qwen3",
        expected_min_tps=20.0,
        expected_min_pp=400.0,
    ),
    "nemotron_omni_nano_jangtq4": Row(
        "nemotron_omni_nano_jangtq4",
        "/Users/example/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ4-CRACK",
        tool_parser="nemotron",
        reasoning_parser="qwen3",
        expected_min_tps=20.0,
        expected_min_pp=400.0,
    ),
    "nemotron_mxfp4": Row(
        "nemotron_mxfp4",
        "/Users/example/models/dealign.ai/Nemotron-Omni-Nano-MXFP4-CRACK",
        tool_parser="nemotron",
        reasoning_parser="qwen3",
        expected_min_tps=18.0,
        expected_min_pp=400.0,
    ),
    "nemotron3_jangtq2_ext": Row(
        "nemotron3_jangtq2_ext",
        "/Volumes/ModelDrive/jangq-ai/Nemotron-3-Nano-Omni-30B-A3B-JANGTQ2",
        tool_parser="nemotron",
        reasoning_parser="deepseek_r1",
        expected_min_tps=20.0,
        expected_min_pp=400.0,
    ),
    "nemotron3_mxfp4_ext": Row(
        "nemotron3_mxfp4_ext",
        "/Volumes/ModelDrive/jangq-ai/Nemotron-3-Nano-Omni-30B-A3B-MXFP4",
        tool_parser="nemotron",
        reasoning_parser="deepseek_r1",
        expected_min_tps=18.0,
        expected_min_pp=400.0,
    ),
    "laguna": Row(
        "laguna",
        "/Users/example/models/JANGQ/Laguna-XS.2-JANGTQ",
        reasoning_parser="qwen3",
        max_tokens=192,
        expected_min_tps=15.0,
        expected_min_pp=400.0,
    ),
    "mistral_medium_jangtq_ext": Row(
        "mistral_medium_jangtq_ext",
        "/Volumes/ModelDrive/jangq-ai/Mistral-Medium-3.5-128B-JANGTQ",
        is_mllm=True,
        tool_parser="mistral",
        reasoning_parser="mistral",
        max_tokens=192,
        expected_min_tps=12.0,
        expected_min_pp=400.0,
    ),
    "mistral_medium_mxfp4_ext": Row(
        "mistral_medium_mxfp4_ext",
        "/Volumes/ModelDrive/jangq-ai/Mistral-Medium-3.5-128B-mxfp4",
        is_mllm=True,
        tool_parser="mistral",
        reasoning_parser="mistral",
        max_tokens=192,
        expected_min_tps=10.0,
        expected_min_pp=250.0,
        pp_targets=[1024, 4096],
    ),
    "gpt_oss_ext": Row(
        "gpt_oss_ext",
        "/Volumes/ModelDrive/dealignai/GPT-OSS-120B-MLX-CRACK",
        tool_parser="glm47",
        reasoning_parser="openai_gptoss",
        max_tokens=192,
        expected_min_tps=12.0,
        expected_min_pp=250.0,
    ),
    "dsv4_k": Row(
        "dsv4_k",
        "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K",
        tool_parser="dsml",
        reasoning_parser="deepseek_r1",
        max_tokens=128,
        expected_min_tps=12.0,
        expected_min_pp=250.0,
        pp_targets=[1024, 4096],
    ),
    "dsv4_jang_dq2_gate3math6": Row(
        "dsv4_jang_dq2_gate3math6",
        resolve_default_model(),
        tool_parser="dsml",
        reasoning_parser="deepseek_r1",
        max_tokens=128,
        expected_min_tps=20.0,
        expected_min_pp=250.0,
        pp_targets=[1024, 4096],
    ),
}


PROMPT = "Write the numbers 1 through 180 separated by spaces. No explanation."
WARM_PROMPT = "Reply with exactly WARM."
COHERENCY_PROMPT = (
    "Reply in exactly three short lines: READY, then 17+28=45, then CERULEAN."
)


def make_long_prompt(target_tokens: int) -> str:
    # Approximate tokens with repeated simple words. The actual prompt token
    # count is taken from API usage and written to the artifact.
    words = max(1, int(target_tokens * 1.35))
    filler = " ".join(["calm", "plain", "context", "token"] * ((words // 4) + 1))
    return (
        "Read the context, ignore the repeated filler, and answer only with "
        "READY / 45 / CERULEAN.\n\n"
        f"{filler}\n\n"
        "Now answer exactly:\nREADY\n45\nCERULEAN"
    )


def post_json(url: str, payload: dict[str, Any], timeout: int = 300) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def get_json(url: str, timeout: int = 5) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def wait_health(port: int, proc: subprocess.Popen, timeout_s: int) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            return get_json(url, timeout=2)
        except Exception as exc:  # noqa: BLE001 - diagnostics gate
            last_error = exc
            time.sleep(1)
    raise TimeoutError(f"health timeout on port {port}: {last_error!r}")


def loopish(text: str) -> bool:
    lower = text.lower()
    if any(
        marker in lower
        for marker in (
            "there are no further comments",
            "testing process has been completed",
            "result result",
        )
    ):
        return True
    return re.search(r"(.{16,96})\1\1", text, flags=re.S) is not None


def run_case(
    port: int,
    model_name: str,
    url: str,
    max_tokens: int,
    payload_extra: dict[str, Any],
    *,
    prompt: str = PROMPT,
) -> dict[str, Any]:
    body = {
        "model": model_name,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "skip_prefix_cache": True,
        "cache_salt": f"speed-gate-{model_name}-{time.time_ns()}",
    }
    body.update(payload_extra)
    t0 = time.perf_counter()
    obj = post_json(url, body)
    wall = time.perf_counter() - t0
    choice = (obj.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = obj.get("usage") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    completion_tokens = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    return {
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "wall_seconds": wall,
        "decode_tps_wall": completion_tokens / wall if wall else 0.0,
        "content_head": content[:700],
        "content_tail": content[-350:],
        "reasoning_chars": len(reasoning),
        "loopish": loopish(content + "\n" + reasoning),
        "usage": usage,
    }


def run_pp_case(url: str, model_name: str, target_tokens: int) -> dict[str, Any]:
    prompt = make_long_prompt(target_tokens)
    t0 = time.perf_counter()
    obj = post_json(
        url,
        {
            "model": model_name,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8,
            "skip_prefix_cache": True,
            "cache_salt": f"pp-gate-{model_name}-{target_tokens}-{time.time_ns()}",
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=900,
    )
    wall = time.perf_counter() - t0
    choice = (obj.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = obj.get("usage") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    return {
        "target_tokens": target_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "wall_seconds": wall,
        "pp_wall_tok_s": prompt_tokens / wall if wall else 0.0,
        "content_excerpt": content[:400],
        "reasoning_chars": len(reasoning),
        "finish_reason": choice.get("finish_reason"),
        "loopish": loopish(content + "\n" + reasoning),
        "usage": usage,
    }


def run_row(
    row: Row,
    *,
    port: int,
    python: Path,
    keep_logs: Path,
    timeout_s: int,
    prefill_step_size: int,
) -> dict[str, Any]:
    if not Path(row.path).exists():
        return {"name": row.name, "path": row.path, "status": "missing"}

    log_path = keep_logs / f"{row.name}-{int(time.time())}.log"
    cmd = [
        str(python),
        "-B",
        "-s",
        "-m",
        "vmlx_engine.cli",
        "serve",
        row.path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--timeout",
        "300",
        "--max-num-seqs",
        "1",
        "--prefill-batch-size",
        "512",
        "--prefill-step-size",
        str(prefill_step_size),
        "--completion-batch-size",
        "512",
        "--continuous-batching",
        "--use-paged-cache",
        "--paged-cache-block-size",
        "64",
        "--max-cache-blocks",
        "1000",
        "--enable-block-disk-cache",
        "--block-disk-cache-max-gb",
        "10",
        "--stream-interval",
        "1",
        "--served-model-name",
        row.name,
    ]
    if row.is_mllm:
        cmd.append("--is-mllm")
    if row.tool_parser:
        cmd.extend(["--tool-call-parser", row.tool_parser, "--enable-auto-tool-choice"])
    if row.reasoning_parser:
        cmd.extend(["--reasoning-parser", row.reasoning_parser])
    cmd.extend(row.extra_args)

    env = dict(os.environ)
    env.pop("JANGTQ_MPP_NAX", None)
    env.pop("JANGTQ_MPP_NAX_DISABLE", None)
    env.pop("JANGTQ_MPP_NAX_STRICT", None)
    env.pop("JANGTQ_MPP_DENSE", None)
    env.pop("JANGTQ_MPP_DENSE_STRICT", None)
    env.pop("JANGTQ_DISABLE_DSV4_STREAM_LOAD", None)
    env.pop("JANGTQ_DISABLE_DSV4_FAST_LOAD", None)
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"

    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        health0 = wait_health(port, proc, timeout_s)
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        warm = post_json(
            url,
            {
                "model": row.name,
                "stream": False,
                "messages": [{"role": "user", "content": WARM_PROMPT}],
                "max_tokens": 32,
                "skip_prefix_cache": True,
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        coherency = run_case(
            port,
            row.name,
            url,
            96,
            {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            prompt=COHERENCY_PROMPT,
        )
        pp_rows = [run_pp_case(url, row.name, target) for target in row.pp_targets]
        bundle = run_case(
            port,
            row.name,
            url,
            row.max_tokens,
            {
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        greedy = run_case(
            port,
            row.name,
            url,
            row.max_tokens,
            {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        health1 = get_json(f"http://127.0.0.1:{port}/health", timeout=5)
        status = "pass"
        notes: list[str] = []
        if bundle["loopish"] or greedy["loopish"]:
            status = "fail"
            notes.append("loopish output detected")
        if coherency["loopish"] or any(pp["loopish"] for pp in pp_rows):
            status = "fail"
            notes.append("coherency or PP row loopish output detected")
        if not bundle["content_head"] and bundle["reasoning_chars"]:
            status = "fail"
            notes.append("bundle-sampling row was reasoning-only")
        if row.expected_min_pp:
            below_pp = [
                pp["pp_wall_tok_s"]
                for pp in pp_rows
                if pp["prompt_tokens"] >= 512 and pp["pp_wall_tok_s"] < row.expected_min_pp
            ]
            if below_pp and status == "pass":
                status = "review"
            if below_pp:
                notes.append(
                    f"PP below expected {row.expected_min_pp:.2f}: "
                    + ", ".join(f"{v:.2f}" for v in below_pp)
                )
        if row.expected_min_tps and bundle["decode_tps_wall"] < row.expected_min_tps:
            status = "review"
            notes.append(
                f"bundle decode {bundle['decode_tps_wall']:.2f} < expected {row.expected_min_tps:.2f}"
            )
        return {
            "name": row.name,
            "path": row.path,
            "status": status,
            "notes": notes,
            "cmd": cmd,
            "prefill_step_size": prefill_step_size,
            "log_path": str(log_path),
            "health_before": health0,
            "health_after": health1,
            "warm_usage": warm.get("usage"),
            "coherency": coherency,
            "pp_rows": pp_rows,
            "bundle_sampling": bundle,
            "greedy_topk0": greedy,
        }
    except Exception as exc:  # noqa: BLE001 - live diagnostic gate
        return {
            "name": row.name,
            "path": row.path,
            "status": "error",
            "error": repr(exc),
            "cmd": cmd,
            "log_path": str(log_path),
            "log_tail": log_path.read_text(errors="replace")[-12000:] if log_path.exists() else "",
        }
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=20)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="*", default=["minimax"])
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--python", type=Path, default=DEFAULT_PY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    args = parser.parse_args()

    logs = args.out.parent / "decode-speed-logs"
    logs.mkdir(parents=True, exist_ok=True)
    results = []
    for offset, name in enumerate(args.rows):
        row = ROWS.get(name)
        if row is None:
            results.append({"name": name, "status": "unknown-row"})
            continue
        print(f"[decode-gate] row={name} path={row.path}", flush=True)
        result = run_row(
            row,
            port=args.port + offset,
            python=args.python,
            keep_logs=logs,
            timeout_s=args.timeout,
            prefill_step_size=args.prefill_step_size,
        )
        results.append(result)
        bundle = result.get("bundle_sampling") or {}
        print(
            f"[decode-gate] {name} status={result.get('status')} "
            f"bundle_tps={bundle.get('decode_tps_wall')} notes={result.get('notes')}",
            flush=True,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"created_at": time.time(), "results": results}, indent=2))

    args.out.write_text(json.dumps({"created_at": time.time(), "results": results}, indent=2))
    print(f"[decode-gate] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
