#!/usr/bin/env python3
"""Release gate for the legacy Python/Electron vMLX app.

This script is intentionally stdlib-only. It checks the built source artifacts,
the packaged Electron app, the bundled Python runtime, optional GUI launch, and
optional live model/API/cache/sleep-wake behavior. It writes private logs under
docs/internal/release-gates/ so release evidence is preserved without shipping
local machine details in public release notes.
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "panel"
INTERNAL = ROOT / "docs" / "internal" / "release-gates"


class Gate:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.rows: list[tuple[str, str, str]] = []
        self.failures: list[str] = []
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def record(self, name: str, status: str, detail: str = "") -> None:
        self.rows.append((name, status, detail))
        if status == "FAIL":
            self.failures.append(name)
        print(f"[{status}] {name}{': ' + detail if detail else ''}")

    def run(
        self,
        name: str,
        cmd: list[str],
        *,
        cwd: Path | None = None,
        timeout: int = 120,
        env: dict[str, str] | None = None,
        allow_fail: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        log_path = self.log_dir / f"{slug(name)}.log"
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        started = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd or ROOT),
                env=merged_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            log_path.write_text(output + f"\nTIMEOUT after {timeout}s\n")
            if allow_fail:
                self.record(name, "WARN", f"timeout after {timeout}s; log={log_path}")
                return subprocess.CompletedProcess(cmd, 124, output, "")
            self.record(name, "FAIL", f"timeout after {timeout}s; log={log_path}")
            raise
        log_path.write_text(proc.stdout)
        elapsed = time.time() - started
        if proc.returncode == 0 or allow_fail:
            status = "PASS" if proc.returncode == 0 else "WARN"
            self.record(name, status, f"{elapsed:.1f}s; log={log_path}")
        else:
            self.record(name, "FAIL", f"exit={proc.returncode}; log={log_path}")
        return proc

    def write_summary(self, args: argparse.Namespace) -> Path:
        summary = self.log_dir / "SUMMARY.md"
        lines = [
            "# vMLX Python/Electron Release Gate",
            "",
            f"- Timestamp: `{self.log_dir.name}`",
            f"- Repo: `{ROOT}`",
            f"- App: `{args.app}`",
            f"- Model: `{args.model or 'not provided'}`",
            "",
            "| Check | Result | Detail |",
            "|---|---|---|",
        ]
        for name, status, detail in self.rows:
            lines.append(f"| {escape_md(name)} | {status} | {escape_md(detail)} |")
        lines.extend(
            [
                "",
                "## Release Rule",
                "",
                "- A GitHub/PyPI/DMG release is not production-ready until this gate is PASS for the packaged app.",
                "- If a live model is relevant to the issue, the release evidence must include multi-turn output, cache stats, and memory counters.",
                "- For architecture-specific work, run this script once per local model family and keep each SUMMARY.md row in the internal ledger.",
                "",
            ]
        )
        summary.write_text("\n".join(lines))
        return summary


def slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def request_json(method: str, url: str, body: dict[str, Any] | None = None, timeout: int = 60) -> Any:
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read().decode()
    return json.loads(payload) if payload else {}


def wait_health(base_url: str, timeout: int = 240) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            data = request_json("GET", f"{base_url}/health", timeout=5)
            if data.get("status") in {"healthy", "ok"} or data.get("model_loaded"):
                return data
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"server did not become healthy: {last_error}")


def extract_text(resp: Any) -> str:
    if not isinstance(resp, dict):
        return ""
    if "choices" in resp:
        choice = resp["choices"][0]
        msg = choice.get("message") or {}
        return msg.get("content") or choice.get("text") or ""
    if "output_text" in resp:
        return resp.get("output_text") or ""
    if "content" in resp and isinstance(resp["content"], list):
        return "\n".join(part.get("text", "") for part in resp["content"] if isinstance(part, dict))
    if "message" in resp:
        msg = resp["message"]
        if isinstance(msg, dict):
            return msg.get("content") or ""
    return ""


def assert_visible_text(label: str, resp: Any, gate: Gate) -> str:
    text = extract_text(resp).strip()
    if not text:
        gate.record(label, "FAIL", "empty visible content")
        raise AssertionError(f"{label}: empty content")
    if obvious_loop(text):
        gate.record(label, "FAIL", f"loop-like output: {text[:160]!r}")
        raise AssertionError(f"{label}: loop-like output")
    gate.record(label, "PASS", text[:180].replace("\n", " "))
    return text


def obvious_loop(text: str) -> bool:
    words = [w.strip(".,;:!?()[]{}\"'").lower() for w in text.split()]
    words = [w for w in words if w]
    if len(words) < 32:
        return False
    unique_ratio = len(set(words[-64:])) / min(len(words), 64)
    return unique_ratio < 0.18


def version_from_pyproject() -> str:
    for line in (ROOT / "pyproject.toml").read_text().splitlines():
        if line.startswith("version = "):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("pyproject.toml version not found")


def packaged_python(app: Path) -> Path:
    return app / "Contents" / "Resources" / "bundled-python" / "python" / "bin" / "python3"


def check_static(gate: Gate, app: Path, skip_app: bool) -> None:
    version = version_from_pyproject()
    panel_pkg = json.loads((PANEL / "package.json").read_text())
    init_version = None
    for line in (ROOT / "vmlx_engine" / "__init__.py").read_text().splitlines():
        if line.startswith("__version__"):
            init_version = line.split("=", 1)[1].strip().strip('"')
            break
    if panel_pkg["version"] == version == init_version:
        gate.record("version triple", "PASS", version)
    else:
        gate.record("version triple", "FAIL", f"pyproject={version}, panel={panel_pkg['version']}, init={init_version}")

    gate.run("twine check dist", ["/Users/eric/.local/bin/twine", "check", *map(str, sorted((ROOT / "dist").glob("vmlx-*")))], timeout=120)
    gate.run("panel request/type tests", ["npm", "test", "--", "request-builder.test.ts", "reasoning-display.test.ts", "audit-fixes.test.ts"], cwd=PANEL, timeout=180)
    gate.run("panel typecheck", ["npm", "run", "typecheck"], cwd=PANEL, timeout=180)
    gate.run("bundled python import gate", ["npm", "run", "verify-bundled"], cwd=PANEL, timeout=180)

    if skip_app:
        gate.record("packaged app checks", "WARN", "skipped by --skip-app")
        return
    if not app.exists():
        gate.record("packaged app exists", "FAIL", str(app))
        return
    gate.record("packaged app exists", "PASS", str(app))
    info = app / "Contents" / "Info.plist"
    plist = plistlib.loads(info.read_bytes())
    gate.record("packaged app version", "PASS" if plist.get("CFBundleShortVersionString") == version else "FAIL", str(plist.get("CFBundleShortVersionString")))
    gate.run("codesign strict verify", ["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app)], timeout=180, allow_fail=False)
    gate.run("spctl assessment", ["spctl", "--assess", "--type", "execute", "--verbose=4", str(app)], timeout=120, allow_fail=True)
    py = packaged_python(app)
    gate.run(
        "packaged bundled imports",
        [
            str(py),
            "-B",
            "-s",
            "-c",
            "import vmlx_engine, mflux, mlx_lm, mlx_vlm, jang_tools; print(vmlx_engine.__version__)",
        ],
        timeout=180,
    )


def launch_app_smoke(gate: Gate, app: Path) -> None:
    if not app.exists():
        gate.record("GUI open smoke", "FAIL", f"missing app {app}")
        return
    gate.run("GUI open app", ["open", "-n", str(app)], timeout=60)
    time.sleep(6)
    proc = gate.run("GUI process present", ["pgrep", "-fl", f"{app}/Contents/MacOS/vMLX"], timeout=30, allow_fail=True)
    if proc.returncode != 0:
        proc = gate.run("GUI process present fallback", ["pgrep", "-fl", "vMLX"], timeout=30, allow_fail=True)
    gate.run(
        "GUI window count",
        [
            "osascript",
            "-e",
            'tell application "System Events" to count windows of first process whose name contains "vMLX"',
        ],
        timeout=10,
        allow_fail=True,
    )
    if proc.returncode == 0:
        gate.record("GUI open smoke verdict", "PASS", "process started")
    else:
        gate.record("GUI open smoke verdict", "FAIL", "no vMLX process found")


def live_engine_gate(gate: Gate, app: Path, model: str, port: int, skip_sleep_wake: bool) -> None:
    py = packaged_python(app) if app.exists() else Path(sys.executable)
    if not Path(model).exists() and "/" not in model:
        gate.record("live model path", "FAIL", model)
        return
    log = gate.log_dir / "live_engine_server.log"
    cmd = [
        str(py),
        "-B",
        "-s",
        "-m",
        "vmlx_engine.cli",
        "serve",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-num-seqs",
        "1",
        "--enable-prefix-cache",
        "--use-paged-cache",
        "--enable-block-disk-cache",
        "--block-disk-cache-max-gb",
        "1",
        "--default-temperature",
        "0",
        "--default-top-p",
        "1",
        "--max-tokens",
        "512",
    ]
    with log.open("w") as fp:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fp, stderr=subprocess.STDOUT, text=True)
    base = f"http://127.0.0.1:{port}"
    try:
        health = wait_health(base)
        gate.record("live server health", "PASS", json.dumps(health)[:240])

        chat = request_json(
            "POST",
            f"{base}/v1/chat/completions",
            {
                "model": "local",
                "messages": [{"role": "user", "content": "Answer with exactly: Paris"}],
                "temperature": 0,
                "max_tokens": 64,
            },
        )
        assert_visible_text("OpenAI chat visible output", chat, gate)

        mt = request_json(
            "POST",
            f"{base}/v1/chat/completions",
            {
                "model": "local",
                "messages": [
                    {"role": "user", "content": "Remember this word: teal. Reply OK."},
                    {"role": "assistant", "content": "OK."},
                    {"role": "user", "content": "What word did I ask you to remember?"},
                ],
                "temperature": 0,
                "max_tokens": 96,
            },
        )
        text = assert_visible_text("OpenAI multi-turn recall", mt, gate)
        if "teal" not in text.lower():
            gate.record("OpenAI multi-turn recall exact", "FAIL", text[:180])
        else:
            gate.record("OpenAI multi-turn recall exact", "PASS", text[:180])

        responses = request_json(
            "POST",
            f"{base}/v1/responses",
            {
                "model": "local",
                "input": "Answer with exactly: 4",
                "temperature": 0,
                "max_output_tokens": 128,
            },
        )
        assert_visible_text("Responses visible output", responses, gate)

        anthropic = request_json(
            "POST",
            f"{base}/v1/messages",
            {
                "model": "local",
                "max_tokens": 96,
                "messages": [{"role": "user", "content": "Answer with exactly: blue"}],
            },
        )
        assert_visible_text("Anthropic visible output", anthropic, gate)

        ollama = request_json(
            "POST",
            f"{base}/api/chat",
            {
                "model": "local",
                "messages": [{"role": "user", "content": "Answer with exactly: green"}],
                "stream": False,
                "options": {"temperature": 0, "num_predict": 96},
            },
        )
        assert_visible_text("Ollama visible output", ollama, gate)

        stats = request_json("GET", f"{base}/v1/cache/stats")
        gate.record("cache stats after API matrix", "PASS", json.dumps(stats)[:600])

        if not skip_sleep_wake:
            request_json("POST", f"{base}/admin/soft-sleep", {})
            time.sleep(2)
            wake_resp = request_json(
                "POST",
                f"{base}/v1/chat/completions",
                {
                    "model": "local",
                    "messages": [{"role": "user", "content": "After wake, answer with exactly: awake"}],
                    "temperature": 0,
                    "max_tokens": 96,
                },
                timeout=240,
            )
            assert_visible_text("JIT soft-wake inference", wake_resp, gate)
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        gate.record("live server log", "PASS", str(log))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", default=str(PANEL / "release" / "mac-arm64" / "vMLX.app"))
    parser.add_argument("--model", help="Optional local model path or HF id for live API/cache checks")
    parser.add_argument("--port", type=int, default=18380)
    parser.add_argument("--skip-app", action="store_true", help="Skip packaged app signature/import checks")
    parser.add_argument("--skip-gui", action="store_true", help="Skip opening the GUI app")
    parser.add_argument("--skip-sleep-wake", action="store_true", help="Skip /admin/soft-sleep + JIT wake")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    gate = Gate(INTERNAL / ts)
    app = Path(args.app).expanduser().resolve()

    try:
        check_static(gate, app, args.skip_app)
        if not args.skip_gui and not args.skip_app:
            launch_app_smoke(gate, app)
        if args.model:
            live_engine_gate(gate, app, args.model, args.port, args.skip_sleep_wake)
    except Exception as exc:  # noqa: BLE001
        gate.record("release gate exception", "FAIL", repr(exc))
    summary = gate.write_summary(args)
    print(f"\nSummary: {summary}")
    return 1 if gate.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
