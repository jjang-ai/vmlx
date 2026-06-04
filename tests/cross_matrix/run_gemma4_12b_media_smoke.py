#!/usr/bin/env python3
"""Run live Gemma 4 12B media smoke rows through vMLX HTTP APIs."""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import struct
import subprocess
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "build/current-gemma4-12b-media-smoke.json"
SAFE_SERVER_CWD = Path("/tmp")


@dataclass(frozen=True)
class Row:
    name: str
    path: str
    served_name: str


ROWS: dict[str, Row] = {
    "mxfp4": Row(
        "mxfp4",
        "/Users/example/models/OsaurusAI/gemma-4-12B-it-MXFP4",
        "gemma4-12b-mxfp4-media-smoke",
    ),
    "mxfp8": Row(
        "mxfp8",
        "/Users/example/models/OsaurusAI/gemma-4-12B-it-MXFP8",
        "gemma4-12b-mxfp8-media-smoke",
    ),
    "mxfp8_attnfp16": Row(
        "mxfp8_attnfp16",
        "/Users/example/models/OsaurusAI/gemma-4-12B-it-MXFP8-ATTNFP16-CANDIDATE",
        "gemma4-12b-mxfp8-attnfp16-media-smoke",
    ),
    "mxfp8_attnfp16_l024": Row(
        "mxfp8_attnfp16_l024",
        "/Users/example/models/OsaurusAI/gemma-4-12B-it-MXFP8-ATTNFP16-L0-24-FP16-CANDIDATE",
        "gemma4-12b-mxfp8-attnfp16-l024-media-smoke",
    ),
    "jang4m": Row(
        "jang4m",
        "/Users/example/models/JANGQ-AI/gemma-4-12B-it-JANG_4M",
        "gemma4-12b-jang4m-media-smoke",
    ),
}


def _request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 60,
) -> tuple[int, Any, str]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = raw
            return resp.status, body, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = raw
        return exc.code, body, raw
    except Exception as exc:  # noqa: BLE001 - artifact captures transport failures
        return 0, {"error": f"{type(exc).__name__}: {exc}"}, ""


def _png_data_url(width: int, height: int, rgb: tuple[int, int, int]) -> str:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    row = b"\x00" + bytes(rgb) * width
    raw = row * height
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _content_from_chat(body: Any) -> str:
    try:
        return str(body["choices"][0]["message"].get("content") or "")
    except Exception:
        return ""


def _wait_for_health(base_url: str, proc: subprocess.Popen[str], timeout_s: float) -> dict[str, Any]:
    start = time.monotonic()
    last: dict[str, Any] = {}
    while time.monotonic() - start < timeout_s:
        if proc.poll() is not None:
            return {"ok": False, "returncode": proc.returncode, "last": last}
        code, body, _ = _request_json("GET", f"{base_url}/health", timeout=5)
        last = {"code": code, "body": body}
        if code == 200 and isinstance(body, dict):
            return {"ok": True, "body": body, "elapsed_sec": round(time.monotonic() - start, 3)}
        time.sleep(2)
    return {"ok": False, "last": last, "elapsed_sec": round(time.monotonic() - start, 3)}


def _build_command(python: Path, row: Row, port: int) -> list[str]:
    return [
        str(python.absolute()),
        "-B",
        "-s",
        "-P",
        "-m",
        "vmlx_engine.cli",
        "serve",
        row.path,
        "--served-model-name",
        row.served_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--reasoning-parser",
        "gemma4",
        "--tool-call-parser",
        "gemma4",
        "--enable-auto-tool-choice",
        "--default-enable-thinking",
        "false",
        "--log-level",
        "INFO",
        "--no-continuous-batching",
        "--disable-prefix-cache",
        "--kv-cache-quantization",
        "none",
        "--disable-native-mtp",
    ]


def run_image_row(
    *,
    row: Row,
    python: Path,
    port: int,
    load_timeout_s: float,
    artifact_dir: Path,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifact_dir / f"{row.name}_image_server.log"
    result: dict[str, Any] = {
        "row": row.name,
        "media": "image",
        "model_path": row.path,
        "served_name": row.served_name,
        "status": "open",
        "checks": {},
        "failures": [],
        "server_log": str(log_path),
    }
    if not Path(row.path).is_dir():
        result["status"] = "missing"
        result["failures"].append(f"model path missing: {row.path}")
        return result

    cmd = _build_command(python, row, port)
    result["command"] = cmd
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(SAFE_SERVER_CWD),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    base_url = f"http://127.0.0.1:{port}"
    try:
        result["health"] = _wait_for_health(base_url, proc, load_timeout_s)
        if not result["health"].get("ok"):
            result["failures"].append("server_did_not_become_healthy")
            return result

        cap_code, caps, raw_caps = _request_json(
            "GET",
            f"{base_url}/v1/models/{row.served_name}/capabilities",
            timeout=20,
        )
        result["capabilities"] = {"code": cap_code, "body": caps, "raw_tail": raw_caps[-4000:]}

        payload = {
            "model": row.served_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "The image is a solid color. Answer with one color word only.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": _png_data_url(64, 64, (255, 0, 0))},
                        },
                    ],
                }
            ],
            "max_tokens": 32,
        }
        started = time.monotonic()
        code, body, raw = _request_json(
            "POST",
            f"{base_url}/v1/chat/completions",
            payload=payload,
            timeout=180,
        )
        content = _content_from_chat(body)
        result["chat"] = {
            "code": code,
            "elapsed_sec": round(time.monotonic() - started, 3),
            "body": body,
            "raw_tail": raw[-4000:],
            "content": content,
        }
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=20)
        result["server_returncode"] = proc.returncode
        if log_path.exists():
            result["server_log_tail"] = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]

    caps_body = result.get("capabilities", {}).get("body")
    modalities = caps_body.get("modalities", []) if isinstance(caps_body, dict) else []
    text = str(result.get("chat", {}).get("content") or "")
    checks = result["checks"]
    checks["capabilities_ok"] = result.get("capabilities", {}).get("code") == 200
    checks["vision_advertised"] = "vision" in modalities
    checks["chat_ok"] = result.get("chat", {}).get("code") == 200
    checks["visible_answer"] = bool(text.strip())
    checks["red_detected"] = "red" in text.lower()
    checks["no_channel_leak"] = "<|channel>" not in text and "<channel|>" not in text
    for name, ok in checks.items():
        if not ok:
            result["failures"].append(name)
    result["status"] = "pass" if not result["failures"] else "fail"
    return result


def build_artifact(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = args.out.parent / (args.out.stem + "_artifacts")
    results: dict[str, Any] = {}
    port = args.port
    for row_name in args.rows:
        row = ROWS[row_name]
        key = f"{row.name}_image"
        results[key] = run_image_row(
            row=row,
            python=args.python,
            port=port,
            load_timeout_s=args.load_timeout,
            artifact_dir=artifact_dir,
        )
        port += 1
        if results[key].get("status") != "pass" and args.stop_on_failure:
            break
    checks = {
        "all_rows_passed": all(result.get("status") == "pass" for result in results.values()),
        "at_least_one_row_ran": bool(results),
    }
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "pass" if all(checks.values()) else "open",
        "checks": checks,
        "rows": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=REPO / ".venv/bin/python")
    parser.add_argument("--rows", nargs="+", choices=sorted(ROWS), default=["mxfp4"])
    parser.add_argument("--port", type=int, default=8954)
    parser.add_argument("--load-timeout", type=float, default=420)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    artifact = build_artifact(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    print(f"status={artifact['status']}")
    for key, result in artifact["rows"].items():
        print(f"{key}: status={result.get('status')} failures={result.get('failures')}")
        print(f"  content={result.get('chat', {}).get('content')!r}")
        if result.get("server_log"):
            print(f"  log={result['server_log']}")
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
