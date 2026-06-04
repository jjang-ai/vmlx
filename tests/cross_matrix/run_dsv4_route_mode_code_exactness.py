#!/usr/bin/env python3
"""Live DSV4 route/mode code exactness probe.

This is a deliberately narrow live diagnostic for the DSV4 long-output/code
quality release row. It starts one real DSV4 server, runs the same exact-code
task through chat, responses, and legacy completions, and records whether the
required Three.js identifiers survive without corrupt repeated subpieces.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
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
DEFAULT_MIN_FREE_GB = 120.0
DEFAULT_REQUIRED_MODEL_MARGIN_GB = 40.0
TOP_MEMORY_PROCESS_COUNT = 10
HEAVY_PROCESS_PATTERNS = (
    "vmlx_engine.cli serve",
    "mlx_lm.server",
    "run_runtime_memory_stress_probe",
    "run_decode_speed_gate",
    "live-real-ui-model-proof",
    "run_dsv4_route_mode_code_exactness",
    "run_dsv4_default_cache_tool_loop_gate",
)
TOP_MEMORY_PROCESS_COMMAND = "ps -axo pid,rss,command -r | head -n 11"
ACTIVE_HEAVY_PROCESS_COMMAND = (
    "pgrep -af 'vmlx_engine.cli serve|mlx_lm.server|"
    "run_runtime_memory_stress_probe|run_decode_speed_gate|"
    "live-real-ui-model-proof|run_dsv4_route_mode_code_exactness|"
    "run_dsv4_default_cache_tool_loop_gate'"
)

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


def _run_text(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def parse_vm_stat(text: str) -> dict[str, Any]:
    page_size_match = re.search(r"page size of (\d+) bytes", text)
    page_size = int(page_size_match.group(1)) if page_size_match else 16384
    pages: dict[str, int] = {}
    for line in text.splitlines():
        match = re.match(r"Pages ([^:]+):\s+([0-9]+)\.", line.strip())
        if not match:
            continue
        pages[match.group(1).strip().replace(" ", "_")] = int(match.group(2))
    return {"page_size": page_size, "pages": pages}


def gb_from_pages(count: int, page_size: int) -> float:
    return round((count * page_size) / (1024**3), 2)


def vm_stat_memory_breakdown(vm_stat_text: str | None = None) -> dict[str, Any]:
    vm = parse_vm_stat(vm_stat_text if vm_stat_text is not None else _run_text(["vm_stat"]))
    page_size = int(vm["page_size"])
    pages = vm["pages"]
    free_gb = gb_from_pages(int(pages.get("free", 0)), page_size)
    speculative_gb = gb_from_pages(int(pages.get("speculative", 0)), page_size)
    purgeable_gb = gb_from_pages(int(pages.get("purgeable", 0)), page_size)
    inactive_gb = gb_from_pages(int(pages.get("inactive", 0)), page_size)
    return {
        "free": free_gb,
        "speculative": speculative_gb,
        "purgeable": purgeable_gb,
        "inactive": inactive_gb,
        "free_plus_speculative_purgeable": round(
            free_gb + speculative_gb + purgeable_gb,
            2,
        ),
    }


def model_size_gb(model_path: Path) -> float | None:
    if not model_path.exists():
        return None
    total = 0
    for path in model_path.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return round(total / (1024**3), 2)


def memory_pressure_snapshot() -> dict[str, Any]:
    try:
        text = _run_text(["memory_pressure"])
    except Exception as exc:  # noqa: BLE001 - diagnostic helper
        return {"free_percent": None, "error": f"{type(exc).__name__}: {exc}"}
    match = re.search(r"System-wide memory free percentage:\s*([0-9]+)%", text)
    return {
        "free_percent": int(match.group(1)) if match else None,
        "raw_tail": text.splitlines()[-20:],
    }


def parse_top_memory_processes(
    text: str,
    limit: int = TOP_MEMORY_PROCESS_COUNT,
) -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("PID "):
            continue
        parts = stripped.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            rss_kb = int(parts[1])
        except ValueError:
            continue
        processes.append(
            {
                "pid": pid,
                "rss_gb": round(rss_kb / (1024**2), 2),
                "command": parts[2],
            }
        )
    processes.sort(key=lambda item: item["rss_gb"], reverse=True)
    return processes[:limit]


def parse_active_heavy_processes(
    text: str,
    *,
    ignore_pids: set[int] | None = None,
) -> list[dict[str, Any]]:
    ignored = ignore_pids or set()
    processes: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 1)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid in ignored:
            continue
        command = parts[1]
        if "pgrep -af" in command:
            continue
        if not any(pattern in command for pattern in HEAVY_PROCESS_PATTERNS):
            continue
        processes.append({"pid": pid, "command": command})
    return processes


def process_context_snapshot() -> dict[str, Any]:
    try:
        top_processes_text = _run_text(["sh", "-c", TOP_MEMORY_PROCESS_COMMAND])
    except Exception:
        top_processes_text = ""
    try:
        active_processes_text = _run_text(["sh", "-c", ACTIVE_HEAVY_PROCESS_COMMAND])
    except subprocess.CalledProcessError:
        active_processes_text = ""
    except Exception:
        active_processes_text = ""
    active_heavy_processes = parse_active_heavy_processes(
        active_processes_text,
        ignore_pids={os.getpid()},
    )
    return {
        "top_memory_processes": parse_top_memory_processes(top_processes_text),
        "active_heavy_processes": active_heavy_processes,
        "active_heavy_process_count": len(active_heavy_processes),
        "commands": {
            "memory": "vm_stat",
            "top_memory_processes": TOP_MEMORY_PROCESS_COMMAND,
            "active_heavy_processes": ACTIVE_HEAVY_PROCESS_COMMAND,
        },
    }


def resource_snapshot(name: str, proc: subprocess.Popen[str] | None = None) -> dict[str, Any]:
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


def memory_preflight_artifact(
    args: argparse.Namespace,
    *,
    force_no_launch: bool = False,
) -> dict[str, Any] | None:
    min_free_gb = float(getattr(args, "min_free_gb", 0.0) or 0.0)
    user_ram_override = bool(getattr(args, "allow_user_ram_override", False))
    if min_free_gb <= 0:
        return None
    snap = resource_snapshot("preflight")
    available = (snap.get("system_memory") or {}).get("available_gb")
    try:
        vm_breakdown = vm_stat_memory_breakdown()
    except Exception as exc:  # noqa: BLE001 - keep psutil-only fallback diagnostic
        vm_breakdown = {"error": f"{type(exc).__name__}: {exc}"}
    pressure_snapshot = memory_pressure_snapshot()
    vm_available = vm_breakdown.get("free_plus_speculative_purgeable")
    if user_ram_override and isinstance(available, (int, float)):
        gate_available = float(available)
        memory_source = "psutil_available_user_ram_override"
    elif isinstance(vm_available, (int, float)):
        gate_available = float(vm_available)
        memory_source = "vm_stat_free_plus_speculative_purgeable"
    elif isinstance(available, (int, float)):
        gate_available = float(available)
        memory_source = "psutil_available"
    else:
        gate_available = None
        memory_source = "unknown"
    process_context = process_context_snapshot()
    active_heavy_processes = process_context["active_heavy_processes"]
    size_gb = model_size_gb(Path(str(args.model)))
    safety_margin_gb = (
        round(min_free_gb - size_gb, 2) if size_gb is not None else None
    )
    conservative_floor_valid = (
        safety_margin_gb is None
        or safety_margin_gb >= DEFAULT_REQUIRED_MODEL_MARGIN_GB
    )
    floor_valid = conservative_floor_valid or user_ram_override
    if gate_available is not None and (
        force_no_launch
        or not floor_valid
        or active_heavy_processes
        or gate_available < min_free_gb
    ):
        selected_cases = selected_case_names(args)
        memory_gap_gb = round(max(0.0, min_free_gb - gate_available), 2)
        psutil_available_gap_gb = (
            round(max(0.0, min_free_gb - float(available)), 2)
            if isinstance(available, (int, float))
            else None
        )
        status = (
            "invalid_preflight_floor"
            if not floor_valid
            else "skipped_active_heavy_process"
            if active_heavy_processes
            else "ready_to_launch"
            if gate_available >= min_free_gb
            else "skipped"
        )
        used_vm_stat = memory_source == "vm_stat_free_plus_speculative_purgeable"
        launch_blockers = [
            blocker
            for blocker, blocked in (
                ("invalid_preflight_floor", not floor_valid),
                ("active_heavy_process", bool(active_heavy_processes)),
                ("insufficient_memory", gate_available < min_free_gb),
            )
            if blocked
        ]
        artifact = {
            "schema": "vmlx-dsv4-route-mode-code-exactness-v1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": status,
            "reason": (
                "memory_preflight_floor_met"
                if status == "ready_to_launch"
                else "DSV4 preflight floor is below model-size plus safety headroom; do not use this artifact as a launch/skip decision"
                if status == "invalid_preflight_floor"
                else "another heavy model/proof process is active; do not launch DSV4 concurrently"
                if status == "skipped_active_heavy_process"
                else "insufficient_vm_stat_memory"
                if used_vm_stat
                else "insufficient_free_memory"
            ),
            "required_available_gb": min_free_gb,
            "required_free_gb": min_free_gb,
            "min_free_gb": min_free_gb,
            "required_model_margin_gb": DEFAULT_REQUIRED_MODEL_MARGIN_GB,
            "model_size_gb": size_gb,
            "safety_margin_gb": safety_margin_gb,
            "floor_valid": floor_valid,
            "conservative_floor_valid": conservative_floor_valid,
            "user_ram_override": user_ram_override,
            "available_gb": (
                round(float(available), 2)
                if isinstance(available, (int, float))
                else None
            ),
            "preflight_available_gb": round(gate_available, 2),
            "available_for_gate_gb": round(gate_available, 2),
            "preflight_memory_source": memory_source,
            "strict_vm_stat_memory_gap_gb": memory_gap_gb if used_vm_stat else None,
            "psutil_available_gap_gb": psutil_available_gap_gb,
            "free_plus_speculative_purgeable_gb": round(float(vm_available), 2)
            if isinstance(vm_available, (int, float))
            else None,
            "memory_pressure_free_percent": pressure_snapshot.get("free_percent"),
            "memory_pressure_raw_tail": pressure_snapshot.get("raw_tail"),
            "memory_pressure_error": pressure_snapshot.get("error"),
            "memory_breakdown_gb": {
                key: vm_breakdown.get(key)
                for key in ("free", "speculative", "purgeable", "inactive")
                if key in vm_breakdown
            },
            "memory_gap_gb": memory_gap_gb,
            "did_not_launch": True,
            "launch_decision": (
                "launch_allowed" if status == "ready_to_launch" else "do_not_launch"
            ),
            "launch_allowed": status == "ready_to_launch",
            "launch_blockers": launch_blockers,
            "model": args.model,
            "cmd": build_cmd(args),
            "selected_cases": list(selected_cases),
            "case_count": len(selected_cases),
            "telemetry": [snap],
        }
        artifact["memory_breakdown"] = artifact["memory_breakdown_gb"]
        artifact.update(process_context)
        artifact["commands"]["memory_pressure"] = "memory_pressure"
        return artifact
    return None


def blocked_by_memory_preflight(args: argparse.Namespace) -> dict[str, Any] | None:
    return memory_preflight_artifact(args, force_no_launch=False)


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
        "identifier_corruption_candidates": identifier_corruption_candidates(
            content,
            missing,
        ),
        "has_markdown_fence": "```" in content,
    }


def identifier_corruption_candidates(
    content: str,
    missing_identifiers: list[str],
) -> list[dict[str, str]]:
    """Record near-miss Three.js identifiers for root-cause artifacts."""
    seen = set(re.findall(r"\bTHREE\.[A-Za-z_][A-Za-z0-9_]*", content))
    seen.update(re.findall(r"\b[A-Z][A-Za-z0-9_]*(?:Renderer|Camera|Geometry|Material|Scene)\b", content))
    candidates: list[dict[str, str]] = []
    for required in missing_identifiers:
        required_tail = required.rsplit(".", 1)[-1]
        for candidate in sorted(seen):
            candidate_tail = candidate.rsplit(".", 1)[-1]
            if candidate == required or candidate_tail == required_tail:
                continue
            ratio = difflib.SequenceMatcher(
                None,
                required_tail.lower(),
                candidate_tail.lower(),
            ).ratio()
            if ratio >= 0.72:
                candidates.append({"required": required, "candidate": candidate})
    return candidates


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


def _encode_prompt_token_ids(
    prompt: str,
    *,
    model_path: str,
    tokenizer: Any | None = None,
) -> list[int]:
    tok = tokenizer
    if tok is None:
        tok_file = Path(model_path).expanduser() / "tokenizer.json"
        if not tok_file.exists():
            return []
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(tok_file))
    try:
        encoded = tok.encode(prompt, add_special_tokens=False)
    except TypeError:
        encoded = tok.encode(prompt)
    if hasattr(encoded, "ids"):
        encoded = encoded.ids
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return [int(token_id) for token_id in encoded]


def prompt_token_diagnostics(
    prompt: str,
    *,
    model_path: str,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    try:
        ids = _encode_prompt_token_ids(
            prompt,
            model_path=model_path,
            tokenizer=tokenizer,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic artifact only
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not ids:
        return {"available": False, "reason": "tokenizer_unavailable_or_empty"}
    tail = ids[-24:]
    return {
        "available": True,
        "token_count": len(ids),
        "tail_ids": tail,
        "tail_contains_assistant_id": 128804 in tail,
        "tail_contains_think_open_id": 128821 in tail,
        "tail_contains_think_close_id": 128822 in tail,
        "endswith_think_open_id": ids[-1] == 128821,
        "endswith_think_close_id": ids[-1] == 128822,
    }


def prompt_diagnostics(
    route: str,
    body: dict[str, Any],
    *,
    model_path: str,
    renderer: Any | None = None,
    tokenizer: Any | None = None,
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
        "token_diagnostics": prompt_token_diagnostics(
            prompt,
            model_path=model_path,
            tokenizer=tokenizer,
        ),
    }


def effective_prompt_diagnostics(
    route: str,
    body: dict[str, Any],
    *,
    model_path: str,
    renderer: Any | None = None,
) -> dict[str, Any]:
    """Record the DSV4 prompt rail after server-side policy resolution."""
    if route == "completion":
        from vmlx_engine.server import (
            _jang_chat_default_mode,
            _resolve_dsv4_thinking_policy,
        )

        decision = _resolve_dsv4_thinking_policy(
            requested_enable_thinking=None,
            effort_requested=False,
            tools_present=False,
            tool_choice=None,
            default_mode=_jang_chat_default_mode(model_path),
        )
        prompt = str(body.get("prompt") or "")
        render = renderer or _render_dsv4_prompt
        rendered = render(
            [{"role": "user", "content": prompt}],
            enable_thinking=decision.enable_thinking,
            reasoning_effort=None,
            model_path=model_path,
        )
        assistant_think_open = "<｜Assistant｜><think>"
        assistant_think_close = "<｜Assistant｜></think>"
        prompt_tail = rendered[-320:]
        if rendered.endswith(assistant_think_open):
            suffix_kind = "thinking_open"
        elif rendered.endswith(assistant_think_close):
            suffix_kind = "thinking_closed"
        elif "<think>" in prompt_tail:
            suffix_kind = "contains_think_open"
        elif "</think>" in prompt_tail:
            suffix_kind = "contains_think_close"
        else:
            suffix_kind = "none"
        diag = {
            "route": route,
            "enable_thinking": decision.enable_thinking,
            "reasoning_effort": None,
            "prompt_chars": len(rendered),
            "prompt_tail": prompt_tail,
            "assistant_suffix_kind": suffix_kind,
            "prompt_endswith_assistant_think_open": rendered.endswith(
                assistant_think_open
            ),
            "prompt_endswith_assistant_think_close": rendered.endswith(
                assistant_think_close
            ),
            "requested_enable_thinking": None,
            "requested_reasoning_effort": None,
        }
        diag["dsv4_policy_reason"] = decision.reason
        diag["dsv4_reasoning_effort_allowed"] = decision.reasoning_effort_allowed
        return diag

    template_kwargs = body.get("chat_template_kwargs")
    if not isinstance(template_kwargs, dict):
        template_kwargs = {}
    requested_enable = body.get("enable_thinking")
    if requested_enable is None and "enable_thinking" in template_kwargs:
        requested_enable = template_kwargs.get("enable_thinking")
    requested_effort = template_kwargs.get("reasoning_effort", body.get("reasoning_effort"))

    from vmlx_engine.server import (
        _jang_chat_default_mode,
        _normalize_dsv4_reasoning_effort,
        _resolve_dsv4_thinking_policy,
    )

    decision = _resolve_dsv4_thinking_policy(
        requested_enable_thinking=requested_enable,
        effort_requested=bool(requested_effort),
        tools_present=bool(body.get("tools")),
        tool_choice=body.get("tool_choice"),
        default_mode=_jang_chat_default_mode(model_path),
    )
    effective_effort = (
        _normalize_dsv4_reasoning_effort(requested_effort)
        if decision.reasoning_effort_allowed
        else None
    )

    effective_body = dict(body)
    effective_kwargs = dict(template_kwargs)
    effective_kwargs["enable_thinking"] = decision.enable_thinking
    if effective_effort:
        effective_kwargs["reasoning_effort"] = effective_effort
        effective_body["reasoning_effort"] = effective_effort
    else:
        effective_kwargs.pop("reasoning_effort", None)
        effective_body.pop("reasoning_effort", None)
    effective_body["enable_thinking"] = decision.enable_thinking
    effective_body["chat_template_kwargs"] = effective_kwargs

    diag = prompt_diagnostics(
        route,
        effective_body,
        model_path=model_path,
        renderer=renderer,
    )
    diag["requested_enable_thinking"] = requested_enable
    diag["requested_reasoning_effort"] = requested_effort
    diag["dsv4_policy_reason"] = decision.reason
    diag["dsv4_reasoning_effort_allowed"] = decision.reasoning_effort_allowed
    return diag


def normalize_python_executable(value: str | Path, cwd: Path | None = None) -> Path:
    """Return an absolute Python path without resolving venv symlinks."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (cwd or REPO) / path


def build_cmd(args: argparse.Namespace) -> list[str]:
    return [
        str(normalize_python_executable(args.python)),
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


def _exact_code_user_prompt(name: str) -> str:
    if name in {"chat_off_no_punct_rep1", "responses_off_no_punct_rep1"}:
        return (
            "Return exactly this JavaScript code and no markdown fences\n"
            f"{EXACT_CODE}"
        )
    return (
        "Return exactly this JavaScript code and no markdown fences:\n"
        f"{EXACT_CODE}"
    )


def case_body(name: str) -> tuple[str, dict[str, Any]]:
    user = _exact_code_user_prompt(name)
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
        if name in {"chat_off_rep1", "chat_on_rep1", "chat_off_no_punct_rep1"}:
            body["repetition_penalty"] = 1.0
        if name == "chat_off_bundle_defaults":
            body.pop("temperature", None)
            body.pop("top_p", None)
            body.pop("repetition_penalty", None)
        return "chat", body
    if name in {"responses_off", "responses_off_rep1", "responses_off_no_punct_rep1", "responses_on", "responses_on_rep1", "responses_off_bundle_defaults"}:
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
        if name in {"responses_off_rep1", "responses_on_rep1", "responses_off_no_punct_rep1"}:
            body["repetition_penalty"] = 1.0
        if name == "responses_off_bundle_defaults":
            body.pop("temperature", None)
            body.pop("top_p", None)
            body.pop("repetition_penalty", None)
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
    "chat_off_no_punct_rep1",
    "chat_off_bundle_defaults",
    "chat_on",
    "chat_on_rep1",
    "chat_max",
    "responses_off",
    "responses_off_rep1",
    "responses_off_no_punct_rep1",
    "responses_off_bundle_defaults",
    "responses_on",
    "responses_on_rep1",
    "legacy_completion_raw",
)


def selected_case_names(args: argparse.Namespace) -> tuple[str, ...]:
    raw = str(getattr(args, "cases", "") or "").strip()
    if not raw:
        return CASE_NAMES
    names = tuple(name.strip() for name in raw.split(",") if name.strip())
    unknown = sorted(set(names) - set(CASE_NAMES))
    if unknown:
        raise ValueError(f"Unknown case names: {', '.join(unknown)}")
    return names


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    cmd = build_cmd(args)
    cases: list[dict[str, Any]] = []
    selected_cases = selected_case_names(args)
    for name in selected_cases:
        route, body = case_body(name)
        requested_diag = prompt_diagnostics(
            route,
            body,
            model_path=args.model,
        )
        effective_diag = effective_prompt_diagnostics(
            route,
            body,
            model_path=args.model,
        )
        cases.append(
            {
                "name": name,
                "route": route,
                "prompt_diagnostics": requested_diag,
                "requested_prompt_diagnostics": requested_diag,
                "effective_prompt_diagnostics": effective_diag,
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
        "base_url": getattr(args, "base_url", None),
        "case_count": len(cases),
        "selected_cases": list(selected_cases),
        "cases": cases,
    }


def run_cases_against_base_url(args: argparse.Namespace, base_url: str) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    selected_cases = selected_case_names(args)
    health_code, health = http_json("GET", f"{base_url.rstrip('/')}/health", timeout=5)
    for name in selected_cases:
        route, body = case_body(name)
        endpoint = {
            "chat": "/v1/chat/completions",
            "responses": "/v1/responses",
            "completion": "/v1/completions",
        }[route]
        started = time.time()
        code, response = http_json(
            "POST",
            f"{base_url.rstrip('/')}{endpoint}",
            body,
            timeout=args.request_timeout,
        )
        elapsed = time.time() - started
        content, finish, prompt_tokens, completion_tokens = extract_content(route, response)
        analysis = analyze_content(content)
        requested_diag = prompt_diagnostics(
            route,
            body,
            model_path=args.model,
        )
        effective_diag = effective_prompt_diagnostics(
            route,
            body,
            model_path=args.model,
        )
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
                "prompt_diagnostics": requested_diag,
                "requested_prompt_diagnostics": requested_diag,
                "effective_prompt_diagnostics": effective_diag,
                **analysis,
                "raw_keys": sorted(response.keys()) if isinstance(response, dict) else [],
            }
        )
    return {
        "schema": "vmlx-dsv4-route-mode-code-exactness-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "base_url": base_url,
        "health_code": health_code,
        "health0": health,
        "selected_cases": list(selected_cases),
        "case_count": len(cases),
        "cases": cases,
        "status": "pass" if all(case.get("exact") is True for case in cases) else "fail",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "dry_run", False):
        return dry_run(args)
    if getattr(args, "base_url", None):
        return run_cases_against_base_url(args, args.base_url)
    if getattr(args, "memory_preflight_only", False):
        preflight = memory_preflight_artifact(args, force_no_launch=True)
        if preflight is None:
            selected_cases = selected_case_names(args)
            return {
                "schema": "vmlx-dsv4-route-mode-code-exactness-v1",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "status": "preflight_disabled",
                "reason": "min_free_gb_not_positive",
                "required_available_gb": float(
                    getattr(args, "min_free_gb", 0.0) or 0.0
                ),
                "did_not_launch": True,
                "launch_decision": "do_not_launch",
                "model": args.model,
                "cmd": build_cmd(args),
                "selected_cases": list(selected_cases),
                "case_count": len(selected_cases),
            }
        return preflight

    cmd = build_cmd(args)
    preflight_block = blocked_by_memory_preflight(args)
    if preflight_block is not None:
        return preflight_block

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
        selected_cases = selected_case_names(args)
        for name in selected_cases:
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
            requested_diag = prompt_diagnostics(
                route,
                body,
                model_path=args.model,
            )
            effective_diag = effective_prompt_diagnostics(
                route,
                body,
                model_path=args.model,
            )
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
                    "prompt_diagnostics": requested_diag,
                    "requested_prompt_diagnostics": requested_diag,
                    "effective_prompt_diagnostics": effective_diag,
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
            "selected_cases": list(selected_cases),
            "case_count": len(cases),
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
    parser.add_argument("--base-url", help="Reuse an already-running server instead of launching one")
    parser.add_argument("--cases", help="Comma-separated subset of cases to run")
    parser.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    parser.add_argument(
        "--allow-user-ram-override",
        action="store_true",
        help=(
            "Use psutil available memory and allow a below-conservative DSV4 "
            "safety margin when an operator explicitly accepts the RAM risk."
        ),
    )
    parser.add_argument(
        "--memory-preflight-only",
        action="store_true",
        help="Refresh memory/case inventory without launching DSV4 even when the memory floor is met.",
    )
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
    return 0 if result.get("status") in {"pass", "dry_run", "skipped", "ready_to_launch"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
