#!/usr/bin/env python3
"""Live DSV4 default-cache multi-tool loop gate.

This gate exists because a no-cache two-tool proof is not enough for release:
the requested production path is DSV4 Flash with native composite
prefix/paged/L2 cache enabled. The artifact emitted here is consumed by
``summarize_objective_proof.py`` as:

``build/current-dsv4-default-cache-tool-loop/result.json``

The script is safe to dry-run and has a memory preflight so it does not launch a
97G+ DSV4 server on a busy host by accident.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_PY = (
    REPO
    / "panel/release/mac-arm64/vMLX.app/Contents/Resources/bundled-python/python/bin/python3"
)
DEFAULT_MODEL_CANDIDATES = (
    "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANG",
    "/Users/example/models/JANGQ/"
    "DeepSeek-V4-Flash-JANG_DQ2-Token8-DownG32-Gate3Math6-NoMTP",
)
DEFAULT_OUT = REPO / "build/current-dsv4-default-cache-tool-loop/result.json"
DEFAULT_MIN_FREE_GB = 120.0


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "list_directory",
        "description": "List files in a directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Write a UTF-8 text file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
]

EXPECTED_HTML_TOOL_PATH = "landing-p/proof.html"
EXPECTED_HTML_TOOL_CONTENT = "<html><body>dsv4-default-cache-tool-ok</body></html>"
EXPECTED_CODE_TOOL_PATH = "landing-p/scene.js"
EXPECTED_CODE_TOOL_CONTENT = (
    "const scene = new THREE.Scene();\n"
    "const renderer = new THREE.WebGLRenderer();\n"
    "const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);\n"
    "const cube = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial());\n"
    "scene.add(cube);\n"
    "renderer.render(scene, camera);"
)
EXPECTED_CODE_FRAGMENTS = (
    "THREE.WebGLRenderer",
    "THREE.PerspectiveCamera",
    "THREE.MeshBasicMaterial",
    "renderer.render(scene, camera);",
)
CORRUPT_CODE_PATTERNS = (
    "THREE.WebRenderer",
    "THREE.PPerspectiveCamera",
    "THREE.BBoxGeometry",
    "scene.add(c)",
)


def resolve_default_model() -> str:
    for candidate in DEFAULT_MODEL_CANDIDATES:
        if Path(candidate).is_dir():
            return candidate
    return DEFAULT_MODEL_CANDIDATES[0]


def resource_snapshot(name: str, proc: subprocess.Popen | None = None) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "name": name,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        snap["system_memory"] = {
            "total_gb": round(vm.total / (1024**3), 2),
            "available_gb": round(vm.available / (1024**3), 2),
            "percent": vm.percent,
        }
        if proc is not None and proc.poll() is None:
            p = psutil.Process(proc.pid)
            mem = p.memory_info()
            snap["process"] = {
                "pid": proc.pid,
                "rss_gb": round(mem.rss / (1024**3), 3),
                "num_threads": p.num_threads(),
                "status": p.status(),
            }
    except Exception as exc:  # noqa: BLE001 - diagnostic helper
        snap["error"] = f"{type(exc).__name__}: {exc}"
    return snap


def _headers() -> dict[str, str]:
    return {"Content-Type": "application/json"}


def post_json(url: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=_headers(),
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def get_json(url: str, timeout: int = 10) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def wait_health(port: int, proc: subprocess.Popen, timeout_s: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            return get_json(f"http://127.0.0.1:{port}/health", timeout=2)
        except Exception as exc:  # noqa: BLE001 - live diagnostic script
            last_error = exc
            time.sleep(1)
    raise TimeoutError(f"health timeout on port {port}: {last_error!r}")


def build_command(args: argparse.Namespace) -> list[str]:
    max_output_tokens = int(getattr(args, "max_output_tokens", 320) or 320)
    cmd = [
        str(args.python),
        "-B",
        "-s",
        "-m",
        "vmlx_engine.cli",
        "serve",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--timeout",
        "900",
        "--max-num-seqs",
        "1",
        "--continuous-batching",
        "--tool-call-parser",
        "dsml",
        "--enable-auto-tool-choice",
        "--reasoning-parser",
        "deepseek_r1",
    ]
    if bool(getattr(args, "disable_prefix_cache", False)):
        cmd.append("--disable-prefix-cache")
    else:
        cmd.extend(
            [
                "--dsv4-enable-prefix-cache",
                "--use-paged-cache",
                "--paged-cache-block-size",
                "256",
                "--max-cache-blocks",
                "1000",
                "--enable-block-disk-cache",
                "--block-disk-cache-max-gb",
                "10",
            ]
        )
    cmd.extend(
        [
            "--stream-interval",
            "1",
            "--max-tokens",
            str(max_output_tokens),
            "--served-model-name",
            "dsv4-default-cache-tools",
        ]
    )
    return cmd


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "VMLX_DSV4_ENABLE_PREFIX_CACHE",
        "VMLINUX_DSV4_ENABLE_PREFIX_CACHE",
        "VMLX_DSV4_FORCE_DIRECT_RAIL",
        "VMLINUX_DSV4_FORCE_DIRECT_RAIL",
        "VMLX_DSV4_RAW_MAX",
    ):
        env.pop(key, None)
    env["DSV4_LONG_CTX"] = "1"
    env["DSV4_POOL_QUANT"] = "1" if args.pool_quant else "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    return env


def blocked_by_memory_preflight(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.min_free_gb <= 0:
        return None
    snap = resource_snapshot("preflight")
    available = (snap.get("system_memory") or {}).get("available_gb")
    if isinstance(available, (int, float)) and available < args.min_free_gb:
        return {
            "status": "skipped",
            "reason": "insufficient_free_memory",
            "required_available_gb": args.min_free_gb,
            "telemetry": [snap],
        }
    return None


def extract_function_calls(resp: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in resp.get("output") or []
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]


def output_text(resp: dict[str, Any]) -> str:
    if isinstance(resp.get("output_text"), str):
        return resp["output_text"]
    parts: list[str] = []
    for item in resp.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict):
                text = content.get("text") or content.get("content")
                if isinstance(text, str):
                    parts.append(text)
    return "\n".join(parts)


def usage_details(resp: dict[str, Any]) -> dict[str, Any]:
    usage = resp.get("usage") or {}
    details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    return details if isinstance(details, dict) else {}


def code_round_request_controls(
    *,
    request_max_output_tokens: int,
    request_thinking_mode: dict[str, Any],
) -> dict[str, Any]:
    return {
        "round": 2,
        "model": "dsv4-default-cache-tools",
        "store": True,
        "stream": False,
        "max_output_tokens": request_max_output_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        **request_thinking_mode,
        "tools_enabled": True,
        "tool_choice": "auto",
    }


def build_code_prompt(variant: str) -> str:
    if variant == "copy_block":
        return (
            f"Copy this exact JavaScript into {EXPECTED_CODE_TOOL_PATH} using write_file. "
            "Preserve every character, newline, capitalization, punctuation, and identifier spelling. "
            "Do not answer in prose.\n\n"
            f"{EXPECTED_CODE_TOOL_CONTENT}"
        )
    return (
        f"Use write_file to create {EXPECTED_CODE_TOOL_PATH} with content exactly "
        f"{EXPECTED_CODE_TOOL_CONTENT!r}. Preserve identifier spelling and do not answer in prose."
    )


def first_difference(expected: str, actual: str | None) -> dict[str, Any] | None:
    if actual is None:
        return {
            "index": 0,
            "expected_char": expected[:1] or None,
            "actual_char": None,
            "expected_tail": expected[:80],
            "actual_tail": None,
        }
    for index, (expected_char, actual_char) in enumerate(zip(expected, actual)):
        if expected_char != actual_char:
            return {
                "index": index,
                "expected_char": expected_char,
                "actual_char": actual_char,
                "expected_tail": expected[index : index + 80],
                "actual_tail": actual[index : index + 80],
            }
    if len(expected) != len(actual):
        index = min(len(expected), len(actual))
        return {
            "index": index,
            "expected_char": expected[index : index + 1] or None,
            "actual_char": actual[index : index + 1] or None,
            "expected_tail": expected[index : index + 80],
            "actual_tail": actual[index : index + 80],
        }
    return None


def build_code_tool_probe(actual_content: str | None) -> dict[str, Any]:
    return {
        "path": EXPECTED_CODE_TOOL_PATH,
        "expected_content": EXPECTED_CODE_TOOL_CONTENT,
        "actual_content": actual_content,
        "exact": actual_content == EXPECTED_CODE_TOOL_CONTENT,
        "expected_length": len(EXPECTED_CODE_TOOL_CONTENT),
        "actual_length": len(actual_content) if actual_content is not None else None,
        "first_difference": first_difference(
            EXPECTED_CODE_TOOL_CONTENT,
            actual_content,
        ),
        "missing_expected_fragments": [
            item
            for item in EXPECTED_CODE_FRAGMENTS
            if actual_content is None or item not in actual_content
        ],
        "corrupt_identifier_patterns": [
            item
            for item in CORRUPT_CODE_PATTERNS
            if actual_content is not None and item in actual_content
        ],
    }


def response_diagnostics(resp: dict[str, Any]) -> dict[str, Any]:
    output_items = resp.get("output") if isinstance(resp.get("output"), list) else []
    return {
        "response_status": resp.get("status"),
        "incomplete_details": resp.get("incomplete_details"),
        "output_item_statuses": [
            {
                "type": item.get("type"),
                "id": item.get("id"),
                "status": item.get("status"),
                "finish_reason": item.get("finish_reason"),
                "name": item.get("name"),
            }
            for item in output_items
            if isinstance(item, dict)
        ],
    }


def execute_tool(workspace: Path, call: dict[str, Any]) -> dict[str, Any]:
    name = str(call.get("name") or "")
    call_id = str(call.get("call_id") or call.get("id") or "")
    try:
        args = json.loads(call.get("arguments") or "{}")
    except json.JSONDecodeError:
        args = {}
    if name == "list_directory":
        rel = str(args.get("path") or ".")
        target = (workspace / rel).resolve()
        if not str(target).startswith(str(workspace.resolve())):
            output = "ERROR: path outside workspace"
        elif not target.exists() or not target.is_dir():
            output = f"ERROR: directory not found: {rel}"
        else:
            output = "\n".join(sorted(p.name for p in target.iterdir())) or "(empty)"
    elif name == "write_file":
        rel = str(args.get("path") or "")
        target = (workspace / rel).resolve()
        if not str(target).startswith(str(workspace.resolve())):
            output = "ERROR: path outside workspace"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            text = str(args.get("content") or "")
            target.write_text(text, encoding="utf-8")
            output = f"Wrote {rel} ({len(text)} chars)"
    else:
        output = f"ERROR: unknown tool {name}"
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def run(args: argparse.Namespace) -> dict[str, Any]:
    cmd = build_command(args)
    env = build_env(args)
    request_max_output_tokens = int(getattr(args, "max_output_tokens", 320) or 320)
    enable_thinking = bool(getattr(args, "enable_thinking", False))
    request_thinking_mode = {
        "enable_thinking": enable_thinking,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    code_controls = code_round_request_controls(
        request_max_output_tokens=request_max_output_tokens,
        request_thinking_mode=request_thinking_mode,
    )
    code_prompt_variant = str(
        getattr(args, "code_prompt_variant", "exact_inline") or "exact_inline"
    )
    code_prompt = build_code_prompt(code_prompt_variant)
    env_summary = {
        "DSV4_LONG_CTX": env["DSV4_LONG_CTX"],
        "DSV4_POOL_QUANT": env["DSV4_POOL_QUANT"],
        "PYTHONNOUSERSITE": env["PYTHONNOUSERSITE"],
    }
    if args.dry_run:
        return {
            "status": "dry_run",
            "model": args.model,
            "cmd": cmd,
            "env": env_summary,
            "diagnostic_cache_mode": "disabled"
            if bool(getattr(args, "disable_prefix_cache", False))
            else "default_native_prefix_paged_l2",
            "request_max_output_tokens": request_max_output_tokens,
            "request_thinking_mode": request_thinking_mode,
            "diagnostic_code_prompt_variant": code_prompt_variant,
            "diagnostic_code_prompt": code_prompt,
            "code_round_request_controls": code_controls,
            "code_tool_probe": build_code_tool_probe(None),
            "telemetry": [resource_snapshot("dry_run")],
        }

    preflight_block = blocked_by_memory_preflight(args)
    if preflight_block is not None:
        return {**preflight_block, "model": args.model, "cmd": cmd, "env": env_summary}

    out = Path(args.out)
    log_dir = out.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"dsv4-default-cache-tool-loop-{int(time.time())}.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)

    try:
        telemetry = [resource_snapshot("server_spawned", proc)]
        health0 = wait_health(args.port, proc, args.timeout)
        telemetry.append(resource_snapshot("health_ready", proc))
        url = f"http://127.0.0.1:{args.port}/v1/responses"
        rounds: list[dict[str, Any]] = []
        previous_response_id: str | None = None
        pending_outputs: list[dict[str, Any]] = []

        with tempfile.TemporaryDirectory(prefix="vmlx-dsv4-tools-") as tmp:
            workspace = Path(tmp)
            (workspace / "README.md").write_text("dsv4 default cache tool loop\n", encoding="utf-8")
            prompts = [
                "Use list_directory for path '.' and do not answer in prose.",
                f"Now use write_file to create {EXPECTED_HTML_TOOL_PATH} with content exactly {EXPECTED_HTML_TOOL_CONTENT!r}. Do not answer in prose.",
                code_prompt,
                "No more tools. Reply exactly DONE.",
            ]
            for index, prompt in enumerate(prompts):
                body: dict[str, Any] = {
                    "model": "dsv4-default-cache-tools",
                    "input": [*pending_outputs, {"role": "user", "content": prompt}]
                    if pending_outputs
                    else prompt,
                    "store": True,
                    "stream": False,
                    "max_output_tokens": request_max_output_tokens,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "repetition_penalty": 1.0,
                    **request_thinking_mode,
                }
                if index < 3:
                    body["tools"] = TOOLS
                    body["tool_choice"] = "auto"
                if previous_response_id:
                    body["previous_response_id"] = previous_response_id

                t0 = time.perf_counter()
                resp = post_json(url, body, timeout=args.request_timeout)
                elapsed = time.perf_counter() - t0
                previous_response_id = resp.get("id") or previous_response_id
                calls = extract_function_calls(resp)
                executed = [execute_tool(workspace, call) for call in calls]
                pending_outputs = executed
                rounds.append(
                    {
                        "round": index,
                        "elapsed_sec": round(elapsed, 3),
                        "response_id": previous_response_id,
                        "previous_response_id": body.get("previous_response_id"),
                        "function_calls": calls,
                        "executed_tools": [
                            {
                                "name": call.get("name"),
                                "args": json.loads(call.get("arguments") or "{}"),
                                "call_id": call.get("call_id") or call.get("id"),
                            }
                            for call in calls
                        ],
                        "tool_outputs": executed,
                        "output_text": output_text(resp),
                        "response_diagnostics": response_diagnostics(resp),
                        "request_thinking_mode": request_thinking_mode,
                        "request_controls": code_controls if index == 2 else None,
                        "usage": resp.get("usage") or {},
                    }
                )
                telemetry.append(resource_snapshot(f"after_round_{index}", proc))

            written = workspace / EXPECTED_HTML_TOOL_PATH
            file_content = written.read_text(encoding="utf-8") if written.exists() else None
            code_written = workspace / EXPECTED_CODE_TOOL_PATH
            code_file_content = (
                code_written.read_text(encoding="utf-8") if code_written.exists() else None
            )

        health1 = get_json(f"http://127.0.0.1:{args.port}/health", timeout=10)
        cache_stats = get_json(f"http://127.0.0.1:{args.port}/v1/cache/stats", timeout=10)
        native = health1.get("native_cache") or {}
        tool_names = [
            tool.get("name")
            for row in rounds
            for tool in row.get("executed_tools") or []
        ]
        cached_total = sum(
            int(usage_details({"usage": row.get("usage") or {}}).get("cached_tokens") or 0)
            for row in rounds
        )
        details = [
            str(usage_details({"usage": row.get("usage") or {}}).get("cache_detail") or "")
            for row in rounds
        ]
        checks = {
            "tool_sequence_ordered": tool_names
            == ["list_directory", "write_file", "write_file"],
            "final_done": bool(rounds and rounds[-1].get("output_text") == "DONE"),
            "file_written": file_content == EXPECTED_HTML_TOOL_CONTENT,
            "code_file_written_exact": code_file_content == EXPECTED_CODE_TOOL_CONTENT,
            "native_cache": native.get("cache_type") == "native_composite",
            "native_prefix": native.get("prefix") is True,
            "native_paged": native.get("paged") is True,
            "native_l2": native.get("block_disk_l2") is True,
            "generic_tq_kv_off": (native.get("generic_turboquant_kv") or {}).get("enabled")
            is False,
            "cached_tokens_seen": cached_total > 0,
            "dsv4_cache_detail_seen": any("dsv4" in item for item in details),
        }
        return {
            "status": "pass" if all(checks.values()) else "review",
            "model": args.model,
            "cmd": cmd,
            "env": env_summary,
            "diagnostic_cache_mode": "disabled"
            if bool(getattr(args, "disable_prefix_cache", False))
            else "default_native_prefix_paged_l2",
            "request_max_output_tokens": request_max_output_tokens,
            "request_thinking_mode": request_thinking_mode,
            "diagnostic_code_prompt_variant": code_prompt_variant,
            "code_round_request_controls": code_controls,
            "server_pid": proc.pid,
            "log_path": str(log_path),
            "health": health1,
            "health_before": health0,
            "cache_stats": cache_stats,
            "rounds": rounds,
            "checks": checks,
            "code_tool_probe": build_code_tool_probe(code_file_content),
            "telemetry": telemetry,
            "tool_loop_cached_tokens": cached_total,
            "tool_loop_cache_details": details,
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic artifact
        return {
            "status": "error",
            "error": repr(exc),
            "model": args.model,
            "cmd": cmd,
            "env": env_summary,
            "diagnostic_cache_mode": "disabled"
            if bool(getattr(args, "disable_prefix_cache", False))
            else "default_native_prefix_paged_l2",
            "request_max_output_tokens": request_max_output_tokens,
            "request_thinking_mode": request_thinking_mode,
            "diagnostic_code_prompt_variant": code_prompt_variant,
            "log_path": str(log_path),
            "log_tail": log_path.read_text(errors="replace")[-12000:]
            if log_path.exists()
            else "",
        }
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=25)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=25)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=resolve_default_model())
    parser.add_argument("--python", type=Path, default=DEFAULT_PY)
    parser.add_argument("--port", type=int, default=8854)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    parser.add_argument("--max-output-tokens", type=int, default=320)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pool-quant", action="store_true")
    parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="Diagnostic A/B only: disable prefix/paged/L2 cache for this tool loop.",
    )
    parser.add_argument(
        "--code-prompt-variant",
        choices=("exact_inline", "copy_block"),
        default="exact_inline",
        help="Diagnostic A/B only: vary the code-round prompt with the same expected file content.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Diagnostic only: request DSV4 thinking mode for each tool-loop turn.",
    )
    args = parser.parse_args()

    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"created_at": time.time(), **result}, indent=2))
    print(
        f"[dsv4-default-cache-tool-loop] status={result.get('status')} "
        f"out={args.out} log={result.get('log_path')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
