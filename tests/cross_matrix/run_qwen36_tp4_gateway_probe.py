#!/usr/bin/env python3
"""Bounded live probe for the Max2 Qwen3.6 27B MXFP8 TP4 gateway.

The ADLab TP4 gateway is bound to Max2 localhost. This probe runs HTTP checks
through SSH, applies a hard timeout to generation, and captures rank target
directory diagnostics when generation stalls. It must not be treated as a
release pass unless exact output/tool/cache rows complete and validate.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("build/current-qwen36-27b-tp4-gateway-bounded-probe-20260606.json")
DEFAULT_HOST = "test-host.local"
DEFAULT_PORT = 8124
DEFAULT_MODEL = "qwen36-27b-mxfp8-tp4-pod1-shardedvocab"


def run_ssh(host: str, command: str, *, timeout: float) -> dict[str, Any]:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={max(1, int(min(timeout, 10)))}",
        host,
        command,
    ]
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return {
            "timed_out": False,
            "returncode": proc.returncode,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "stdout": proc.stdout,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "timed_out": True,
            "returncode": None,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "stdout": exc.stdout if isinstance(exc.stdout, str) else "",
            "error": f"TimeoutExpired: {timeout}s",
        }


def _remote_json_get(host: str, port: int, path: str, *, timeout: float) -> dict[str, Any]:
    command = (
        "python3 -c "
        + shlex.quote(
            "import json, urllib.request, sys\n"
            f"url='http://127.0.0.1:{port}{path}'\n"
            "try:\n"
            f" r=urllib.request.urlopen(url, timeout={timeout!r}); "
            "print(json.dumps({'http_status': r.status, 'body': json.loads(r.read().decode())}))\n"
            "except Exception as e:\n"
            " print(json.dumps({'http_status': None, 'error': type(e).__name__ + ': ' + str(e)}))\n"
        )
    )
    result = run_ssh(host, command, timeout=timeout + 5)
    payload = _parse_last_json_line(result.get("stdout", ""))
    return {"ssh": result, "json": payload}


def _remote_chat_post(
    host: str,
    port: int,
    body: dict[str, Any],
    *,
    request_timeout: float,
    process_timeout: float,
) -> dict[str, Any]:
    body_arg = shlex.quote(json.dumps(body))
    command = (
        "python3 -c "
        + shlex.quote(
            "import json, sys, time, urllib.request, urllib.error\n"
            f"url='http://127.0.0.1:{port}/v1/chat/completions'\n"
            "body=json.loads(sys.argv[1]); data=json.dumps(body).encode()\n"
            "req=urllib.request.Request(url, data=data, method='POST', headers={'Content-Type':'application/json'})\n"
            "t=time.perf_counter()\n"
            "try:\n"
            f" r=urllib.request.urlopen(req, timeout={request_timeout!r}); raw=r.read().decode('utf-8','replace');\n"
            " print(json.dumps({'http_status': r.status, 'elapsed_s': round(time.perf_counter()-t,3), 'body': json.loads(raw)}))\n"
            "except urllib.error.HTTPError as e:\n"
            " raw=e.read().decode('utf-8','replace');\n"
            " print(json.dumps({'http_status': e.code, 'elapsed_s': round(time.perf_counter()-t,3), 'body': raw[:2000]}))\n"
            "except Exception as e:\n"
            " print(json.dumps({'http_status': None, 'elapsed_s': round(time.perf_counter()-t,3), 'error': type(e).__name__ + ': ' + str(e)}))\n"
        )
        + " "
        + body_arg
    )
    result = run_ssh(host, command, timeout=process_timeout)
    return {"ssh": result, "json": _parse_last_json_line(result.get("stdout", ""))}


def _parse_last_json_line(text: str) -> dict[str, Any] | None:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def rank_targets_from_health(health: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(health, dict):
        return []
    body = health.get("body") if isinstance(health.get("body"), dict) else health
    targets = body.get("rank_status") if isinstance(body, dict) else None
    if not isinstance(targets, list):
        targets = body.get("rank_targets") if isinstance(body, dict) else None
    if not isinstance(targets, list):
        return []
    parsed: list[dict[str, Any]] = []
    for item in targets:
        if not isinstance(item, dict):
            continue
        parsed.append(
            {
                "rank": item.get("rank"),
                "host": item.get("host"),
                "path": item.get("path"),
                "requests_dir": item.get("requests_dir"),
                "responses_dir": item.get("responses_dir"),
                "reachable": item.get("reachable"),
                "ready": item.get("ready"),
            }
        )
    return parsed


def _rank_dir_snapshot(host: str, targets: list[dict[str, Any]], *, timeout: float) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for target in targets:
        req_dir = target.get("requests_dir")
        resp_dir = target.get("responses_dir")
        remote_host = target.get("host") or host
        if not req_dir or not resp_dir:
            snapshots.append({**target, "snapshot_error": "missing_request_or_response_dir"})
            continue
        script = (
            "python3 -c "
            + shlex.quote(
                "import json, pathlib, sys\n"
                "req=pathlib.Path(sys.argv[1]); resp=pathlib.Path(sys.argv[2])\n"
                "def snap(p):\n"
                " files=sorted([x for x in p.glob('*') if x.is_file()], key=lambda x: x.stat().st_mtime, reverse=True)[:5] if p.exists() else []\n"
                " return {'exists': p.exists(), 'file_count': len(list(p.glob('*'))) if p.exists() else 0, 'newest': [{'name': f.name, 'size': f.stat().st_size} for f in files]}\n"
                "print(json.dumps({'requests': snap(req), 'responses': snap(resp)}))\n"
            )
            + " "
            + shlex.quote(str(req_dir))
            + " "
            + shlex.quote(str(resp_dir))
        )
        result = run_ssh(str(remote_host), script, timeout=timeout)
        snapshots.append({**target, "ssh": result, "snapshot": _parse_last_json_line(result.get("stdout", ""))})
    return snapshots


def classify_probe(result: dict[str, Any]) -> str:
    chat = result.get("chat", {})
    ssh = chat.get("ssh") if isinstance(chat, dict) else {}
    payload = chat.get("json") if isinstance(chat, dict) else None
    if isinstance(ssh, dict) and ssh.get("timed_out"):
        return "gateway_generation_timeout"
    if isinstance(payload, dict) and payload.get("http_status") is None:
        error = str(payload.get("error") or "").lower()
        if "timeout" in error or "timed out" in error:
            return "gateway_generation_timeout"
    if not isinstance(payload, dict) or payload.get("http_status") != 200:
        return "gateway_generation_failed"
    content = ""
    try:
        content = payload["body"]["choices"][0]["message"].get("content") or ""
    except Exception:
        content = ""
    if "QWEN-OK" not in content:
        return "exact_output_failed"
    return "pass"


def run_probe(
    *,
    host: str,
    port: int,
    model: str,
    request_timeout: float,
    process_timeout: float,
    rank_snapshot_timeout: float,
) -> dict[str, Any]:
    health = _remote_json_get(host, port, "/health", timeout=10)
    models = _remote_json_get(host, port, "/v1/models", timeout=10)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly QWEN-OK."}],
        "temperature": 0,
        "max_tokens": 16,
    }
    chat = _remote_chat_post(
        host,
        port,
        body,
        request_timeout=request_timeout,
        process_timeout=process_timeout,
    )
    rank_targets = rank_targets_from_health(health.get("json"))
    rank_snapshots = _rank_dir_snapshot(
        host,
        rank_targets,
        timeout=rank_snapshot_timeout,
    )
    result = {
        "status": "open",
        "host": host,
        "port": port,
        "model": model,
        "health": health,
        "models": models,
        "chat": chat,
        "rank_targets": rank_targets,
        "rank_snapshots": rank_snapshots,
    }
    classification = classify_probe(result)
    result["classification"] = classification
    result["status"] = "pass" if classification == "pass" else "open"
    result["release_boundary"] = (
        "This is a narrow TP4 gateway probe. It does not clear Qwen36 release "
        "without repeat cache, L2 disk hit, tool, multi-turn, and streaming rows."
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--request-timeout", type=float, default=45.0)
    parser.add_argument("--process-timeout", type=float, default=60.0)
    parser.add_argument("--rank-snapshot-timeout", type=float, default=10.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    result = run_probe(
        host=args.host,
        port=args.port,
        model=args.model,
        request_timeout=args.request_timeout,
        process_timeout=args.process_timeout,
        rank_snapshot_timeout=args.rank_snapshot_timeout,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"pass", "open"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
