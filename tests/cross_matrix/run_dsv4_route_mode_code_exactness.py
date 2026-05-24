#!/usr/bin/env python3
"""Live DSV4 route/mode code exactness probe.

This is a deliberately narrow live diagnostic for the DSV4 long-output/code
quality release row. It starts one real DSV4 server, runs the same exact-code
task through chat, responses, and legacy completions, and records whether the
required Three.js identifiers survive without corrupt repeated subpieces.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = REPO / "panel/bundled-python/python/bin/python3"
DEFAULT_MODEL = "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K"
SAFE_CWD = Path("/tmp")

REQUIRED_IDENTIFIERS = [
    "THREE.Scene",
    "THREE.WebGLRenderer",
    "THREE.PerspectiveCamera",
    "THREE.BoxGeometry",
    "THREE.MeshBasicMaterial",
]
CORRUPT_PATTERNS = [
    "WebWebGLRenderer",
    "WebScriptRenderer",
    "PPerspectiveCamera",
    "PerscpectiveCamera",
    "BBoxGeometry",
    "MMeshBasicMaterial",
    "Memesh",
    "ScScene",
    "Three.MeshBasicMaterial",
]

EXACT_CODE = (
    "const scene = new THREE.Scene();\n"
    "const renderer = new THREE.WebGLRenderer();\n"
    "const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);\n"
    "const cube = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial());\n"
    "scene.add(cube);\n"
    "renderer.render(scene, camera);"
)


def clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = ""
    if extra:
        env.update(extra)
    return env


def http_json(method: str, url: str, body: dict[str, Any] | None = None, timeout: float = 30.0) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        return exc.code, parsed


def wait_health(port: int, proc: subprocess.Popen[str], timeout_s: int) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last: Any = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited before health: rc={proc.returncode}")
        try:
            code, body = http_json("GET", f"http://127.0.0.1:{port}/health", timeout=2)
            if code == 200 and isinstance(body, dict):
                return body
            last = body
        except Exception as exc:  # noqa: BLE001 - diagnostic polling
            last = repr(exc)
        time.sleep(1)
    raise TimeoutError(f"health timeout: {last!r}")


def terminate(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
        proc.wait(timeout=10)


def extract_content(route: str, response: Any) -> tuple[str, str | None, int, int]:
    if not isinstance(response, dict):
        return "", None, 0, 0
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    if route == "responses":
        text = response.get("output_text")
        if not isinstance(text, str):
            chunks: list[str] = []
            for item in response.get("output") or []:
                if not isinstance(item, dict):
                    continue
                for part in item.get("content") or []:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        chunks.append(part["text"])
            text = "".join(chunks)
        return text or "", response.get("status"), prompt_tokens, completion_tokens
    choices = response.get("choices") if isinstance(response.get("choices"), list) else []
    first = choices[0] if choices and isinstance(choices[0], dict) else {}
    if route == "completion":
        return str(first.get("text") or ""), first.get("finish_reason"), prompt_tokens, completion_tokens
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    return str(message.get("content") or ""), first.get("finish_reason"), prompt_tokens, completion_tokens


def analyze_content(content: str) -> dict[str, Any]:
    normalized = normalize_code_content(content)
    missing = [item for item in REQUIRED_IDENTIFIERS if item not in content]
    corrupt = [item for item in CORRUPT_PATTERNS if item in content]
    return {
        "exact": content.strip() == EXACT_CODE.strip(),
        "normalized_exact": normalized.strip() == EXACT_CODE.strip(),
        "missing": missing,
        "corrupt_patterns": corrupt,
        "has_markdown_fence": "```" in content,
    }


def normalize_code_content(content: str) -> str:
    """Remove one ordinary markdown code fence wrapper for diagnostic analysis."""
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or not lines[-1].strip().startswith("```"):
        return stripped
    return "\n".join(lines[1:-1]).strip()


def _render_dsv4_prompt(
    messages: list[dict[str, Any]],
    *,
    enable_thinking: bool | None,
    reasoning_effort: str | None,
    model_path: str,
) -> str:
    from vmlx_engine.loaders.dsv4_chat_encoder import apply_chat_template

    return apply_chat_template(
        messages,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        model_path=model_path,
    )


def prompt_diagnostics(
    route: str,
    body: dict[str, Any],
    *,
    model_path: str,
    renderer: Any | None = None,
) -> dict[str, Any]:
    """Record the concrete DSV4 prompt rail used by an exactness case."""
    if route == "completion":
        prompt = str(body.get("prompt") or "")
        enable_thinking = None
        reasoning_effort = None
    else:
        messages = body.get("messages") if route == "chat" else body.get("input")
        if not isinstance(messages, list):
            messages = []
        template_kwargs = body.get("chat_template_kwargs")
        if not isinstance(template_kwargs, dict):
            template_kwargs = {}
        enable_thinking = template_kwargs.get("enable_thinking", body.get("enable_thinking"))
        reasoning_effort = template_kwargs.get("reasoning_effort", body.get("reasoning_effort"))
        render = renderer or _render_dsv4_prompt
        prompt = render(
            messages,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            model_path=model_path,
        )

    assistant_think_open = "<｜Assistant｜><think>"
    assistant_think_close = "<｜Assistant｜></think>"
    prompt_tail = prompt[-320:]
    if prompt.endswith(assistant_think_open):
        suffix_kind = "thinking_open"
    elif prompt.endswith(assistant_think_close):
        suffix_kind = "thinking_closed"
    elif "<think>" in prompt_tail:
        suffix_kind = "contains_think_open"
    elif "</think>" in prompt_tail:
        suffix_kind = "contains_think_close"
    else:
        suffix_kind = "none"
    return {
        "route": route,
        "enable_thinking": enable_thinking,
        "reasoning_effort": reasoning_effort,
        "prompt_chars": len(prompt),
        "prompt_tail": prompt_tail,
        "assistant_suffix_kind": suffix_kind,
        "prompt_endswith_assistant_think_open": prompt.endswith(assistant_think_open),
        "prompt_endswith_assistant_think_close": prompt.endswith(assistant_think_close),
    }


def build_cmd(args: argparse.Namespace) -> list[str]:
    return [
        str(Path(args.python).expanduser().resolve()),
        "-B",
        "-s",
        "-P",
        "-m",
        "vmlx_engine.cli",
        "serve",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--timeout",
        str(args.timeout),
        "--max-num-seqs",
        "1",
        "--continuous-batching",
        "--tool-call-parser",
        "dsml",
        "--enable-auto-tool-choice",
        "--reasoning-parser",
        "deepseek_r1",
        "--use-paged-cache",
        "--paged-cache-block-size",
        "256",
        "--max-cache-blocks",
        "1000",
        "--enable-block-disk-cache",
        "--block-disk-cache-max-gb",
        "10",
        "--stream-interval",
        "1",
        "--max-tokens",
        str(args.max_tokens),
        "--served-model-name",
        "dsv4-route-code-probe",
    ]


def case_body(name: str) -> tuple[str, dict[str, Any]]:
    user = (
        "Return exactly this JavaScript code and no markdown fences:\n"
        f"{EXACT_CODE}"
    )
    if name == "chat_max":
        user = (
            "Copy the following JavaScript exactly. Preserve identifier spelling. "
            "No explanation, no markdown.\n"
            f"{EXACT_CODE}"
        )
    if name.startswith("chat"):
        body = {
            "model": "dsv4-route-code-probe",
            "messages": [{"role": "user", "content": user}],
            "stream": False,
            "max_tokens": 512,
            "temperature": 0,
            "top_p": 1,
            "skip_prefix_cache": True,
        }
        if name in {"chat_on", "chat_on_rep1"}:
            body["enable_thinking"] = True
            body["chat_template_kwargs"] = {"enable_thinking": True}
        else:
            body["enable_thinking"] = False
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if name in {"chat_off_rep1", "chat_on_rep1"}:
            body["repetition_penalty"] = 1.0
        return "chat", body
    if name in {"responses_off", "responses_off_rep1", "responses_on", "responses_on_rep1"}:
        body = {
            "model": "dsv4-route-code-probe",
            "input": [{"role": "user", "content": user}],
            "stream": False,
            "max_output_tokens": 512,
            "temperature": 0,
            "top_p": 1,
            "skip_prefix_cache": True,
        }
        if name in {"responses_on", "responses_on_rep1"}:
            body["enable_thinking"] = True
            body["chat_template_kwargs"] = {"enable_thinking": True}
        else:
            body["enable_thinking"] = False
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if name in {"responses_off_rep1", "responses_on_rep1"}:
            body["repetition_penalty"] = 1.0
        return "responses", body
    return "completion", {
        "model": "dsv4-route-code-probe",
        "prompt": user + "\n",
        "stream": False,
        "max_tokens": 220,
        "temperature": 0,
        "top_p": 1,
    }


CASE_NAMES = (
    "chat_off",
    "chat_off_rep1",
    "chat_on",
    "chat_on_rep1",
    "chat_max",
    "responses_off",
    "responses_off_rep1",
    "responses_on",
    "responses_on_rep1",
    "legacy_completion_raw",
)


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    cmd = build_cmd(args)
    cases: list[dict[str, Any]] = []
    for name in CASE_NAMES:
        route, body = case_body(name)
        cases.append(
            {
                "name": name,
                "route": route,
                "prompt_diagnostics": prompt_diagnostics(
                    route,
                    body,
                    model_path=args.model,
                ),
                "request_overrides": {
                    key: body[key]
                    for key in (
                        "enable_thinking",
                        "chat_template_kwargs",
                        "repetition_penalty",
                        "skip_prefix_cache",
                        "max_tokens",
                        "max_output_tokens",
                        "temperature",
                        "top_p",
                    )
                    if key in body
                },
            }
        )
    return {
        "schema": "vmlx-dsv4-route-mode-code-exactness-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "dry_run",
        "dry_run": True,
        "model": args.model,
        "cmd": cmd,
        "case_count": len(cases),
        "cases": cases,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "dry_run", False):
        return dry_run(args)

    cmd = build_cmd(args)
    proc = subprocess.Popen(
        cmd,
        cwd=str(SAFE_CWD),
        env=clean_env(
            {
                "DSV4_POOL_QUANT": "0",
                "VMLINUX_DSV4_ENABLE_PREFIX_CACHE": "0",
                "VMLX_DSV4_ENABLE_PREFIX_CACHE": "0",
            }
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_lines: list[str] = []

    def read_log() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            log_lines.append(line.rstrip("\n"))

    threading.Thread(target=read_log, daemon=True).start()
    try:
        health = wait_health(args.port, proc, args.timeout)
        cases: list[dict[str, Any]] = []
        for name in CASE_NAMES:
            route, body = case_body(name)
            endpoint = {
                "chat": "/v1/chat/completions",
                "responses": "/v1/responses",
                "completion": "/v1/completions",
            }[route]
            started = time.time()
            code, response = http_json(
                "POST",
                f"http://127.0.0.1:{args.port}{endpoint}",
                body,
                timeout=args.request_timeout,
            )
            elapsed = time.time() - started
            content, finish, prompt_tokens, completion_tokens = extract_content(route, response)
            analysis = analyze_content(content)
            cases.append(
                {
                    "name": name,
                    "route": route,
                    "http_code": code,
                    "finish": finish,
                    "elapsed_sec": round(elapsed, 3),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "decode_tps_wall": round(completion_tokens / elapsed, 3)
                    if elapsed > 0 and completion_tokens
                    else 0.0,
                    "content": content,
                    "prompt_diagnostics": prompt_diagnostics(
                        route,
                        body,
                        model_path=args.model,
                    ),
                    **analysis,
                    "raw_keys": sorted(response.keys()) if isinstance(response, dict) else [],
                }
            )
        return {
            "schema": "vmlx-dsv4-route-mode-code-exactness-v1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model": args.model,
            "cmd": cmd,
            "env_overrides": {
                "DSV4_POOL_QUANT": "0",
                "VMLINUX_DSV4_ENABLE_PREFIX_CACHE": "0",
                "VMLX_DSV4_ENABLE_PREFIX_CACHE": "0",
                "PYTHONPATH": "",
                "PYTHONNOUSERSITE": "1",
            },
            "health0": health,
            "cases": cases,
            "status": "pass" if all(case.get("exact") is True for case in cases) else "fail",
            "log_tail": log_lines[-240:],
        }
    finally:
        terminate(proc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=8861)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true", help="Record command/request prompt diagnostics without launching the model")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{out} status={result.get('status')}")
    return 0 if result.get("status") in {"pass", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
