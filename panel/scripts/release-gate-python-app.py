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
import hashlib
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
CACHE_WARM_PAD_TERMS = tuple(f"cache-pad-{idx:03d}" for idx in range(96))
CACHE_WARM_PROMPT = (
    "Remember this cache probe word: cobalt. Reply with exactly: OK. "
    "The following deterministic padding exists only to make the first warm "
    "request cross paged-cache block thresholds on models with 64-token blocks; "
    "ignore it when answering. "
    + " ".join(CACHE_WARM_PAD_TERMS)
)
REQUIRED_RELEASE_ENTITLEMENTS = (
    "com.apple.security.cs.allow-jit",
    "com.apple.security.cs.allow-unsigned-executable-memory",
    "com.apple.security.cs.disable-library-validation",
    "com.apple.security.network.client",
    "com.apple.security.files.user-selected.read-write",
)


def twine_command() -> list[str]:
    override = os.environ.get("TWINE")
    if override:
        return [override]
    repo_venv_python = ROOT / ".venv" / "bin" / "python"
    if python_has_module(repo_venv_python, "twine"):
        return [str(repo_venv_python), "-m", "twine"]
    if python_has_module(Path(sys.executable), "twine"):
        return [sys.executable, "-m", "twine"]
    found = shutil.which("twine")
    if found:
        return [found]
    return [sys.executable, "-m", "twine"]


def python_has_module(py: Path, module_name: str) -> bool:
    if not py.exists():
        return False
    proc = subprocess.run(
        [str(py), "-B", "-s", "-c", f"import {module_name}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={
            **os.environ,
            "PYTHONPATH": "",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        timeout=30,
    )
    return proc.returncode == 0


def twine_env(gate: Gate) -> dict[str, str]:
    """Run twine without letting stale console-script shebangs mutate the app."""
    pycache_prefix = writable_probe_cache_dir(gate, "twine-pycache")
    return {
        "PYTHONPATH": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(pycache_prefix),
    }


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


def write_json_log(gate: Gate, name: str, value: Any) -> Path:
    path = gate.log_dir / f"{slug(name)}.json"
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    return path


def apply_thinking(payload: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "auto":
        return payload
    enabled = mode == "on"
    payload = dict(payload)
    payload["enable_thinking"] = enabled
    kwargs = dict(payload.get("chat_template_kwargs") or {})
    kwargs["enable_thinking"] = enabled
    payload["chat_template_kwargs"] = kwargs
    return payload


def apply_anthropic_thinking(payload: dict[str, Any], mode: str) -> dict[str, Any]:
    """Apply the Anthropic-native thinking contract to release-gate probes."""
    if mode == "auto":
        return payload
    payload = dict(payload)
    if mode == "on":
        payload["thinking"] = {"type": "enabled", "budget_tokens": 1024}
    else:
        payload["thinking"] = {"type": "disabled"}
    return payload


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


def cached_tokens_from_usage(resp: Any) -> int:
    if not isinstance(resp, dict):
        return 0
    usage = resp.get("usage")
    if not isinstance(usage, dict):
        return 0
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        return 0
    try:
        return int(details.get("cached_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def obvious_loop(text: str) -> bool:
    words = [w.strip(".,;:!?()[]{}\"'").lower() for w in text.split()]
    words = [w for w in words if w]
    if len(words) >= 32:
        unique_ratio = len(set(words[-64:])) / min(len(words), 64)
        if unique_ratio < 0.18:
            return True

    # No-space token loops (CJK strings, emoji runs, repeated byte-pair
    # fragments) do not show up as many whitespace-delimited words. Check the
    # tail for dominant character or short character n-gram repetition so the
    # gate catches the exact "eyes forever" / Chinese phrase spam failures.
    compact = "".join(ch for ch in text if not ch.isspace())
    if len(compact) < 48:
        return False
    tail = compact[-512:]
    dominant_char_ratio = max(tail.count(ch) for ch in set(tail)) / len(tail)
    if dominant_char_ratio > 0.35:
        return True
    for period in range(2, min(64, len(tail) // 3) + 1):
        pattern = tail[:period]
        if len(set(pattern)) < 2:
            continue
        expected = (pattern * ((len(tail) // period) + 1))[: len(tail)]
        matches = sum(1 for a, b in zip(tail, expected) if a == b)
        if matches / len(tail) > 0.88:
            return True
    for n in (2, 3, 4, 6, 8, 12):
        if len(tail) < n * 12:
            continue
        grams = [tail[i : i + n] for i in range(0, len(tail) - n + 1)]
        counts: dict[str, int] = {}
        for gram in grams:
            counts[gram] = counts.get(gram, 0) + 1
        if max(counts.values()) / max(1, len(grams)) > 0.18:
            return True
    return False


def version_from_pyproject() -> str:
    for line in (ROOT / "pyproject.toml").read_text().splitlines():
        if line.startswith("version = "):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("pyproject.toml version not found")


def packaged_python(app: Path) -> Path:
    return app / "Contents" / "Resources" / "bundled-python" / "python" / "bin" / "python3"


def writable_probe_cache_dir(gate: Gate, name: str) -> Path:
    """Return a writable cache dir for subprocess probes.

    Release probes run against a signed app bundle, so Python bytecode/cache
    writes must never land inside the app. The normal evidence directory is
    writable, but unit tests intentionally use isolated/read-only paths to
    prove the helper does not require mutating the packaged root.
    """
    preferred = gate.log_dir / name
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "vmlx-release-gate" / slug(
            str(gate.log_dir)
        ) / name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def packaged_python_env(gate: Gate) -> dict[str, str]:
    """Environment for probing packaged Python without mutating the app seal."""
    pycache_prefix = writable_probe_cache_dir(gate, "packaged-python-pycache")
    return {
        "PYTHONPATH": "",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(pycache_prefix),
    }


def check_no_packaged_pycache(gate: Gate, app: Path) -> None:
    python_root = app / "Contents" / "Resources" / "bundled-python" / "python"
    if not python_root.exists():
        gate.record("packaged app pycache clean", "FAIL", f"missing {python_root}")
        return
    hits = sorted(p for p in python_root.rglob("*.pyc") if "__pycache__" in p.parts)
    if hits:
        head = ", ".join(str(p.relative_to(python_root)) for p in hits[:8])
        more = f" (+{len(hits) - 8} more)" if len(hits) > 8 else ""
        gate.record("packaged app pycache clean", "FAIL", f"{head}{more}")
        return
    gate.record("packaged app pycache clean", "PASS", str(python_root))


def _last_nonempty_stdout_line(output: str) -> str:
    for line in reversed((output or "").splitlines()):
        line = line.strip()
        if line:
            return line
    return ""


def check_packaged_bundled_import_version(
    gate: Gate,
    py: Path,
    expected_version: str,
    app_version: str,
) -> None:
    proc = gate.run(
        "packaged bundled imports",
        [
            str(py),
            "-B",
            "-s",
            "-c",
            "import vmlx_engine, mflux, mlx_lm, mlx_vlm, jang_tools; print(vmlx_engine.__version__)",
        ],
        cwd=gate.log_dir,
        env=packaged_python_env(gate),
        timeout=180,
    )
    bundled_version = _last_nonempty_stdout_line(proc.stdout)
    status = "PASS" if bundled_version == expected_version else "FAIL"
    gate.record(
        "packaged bundled version",
        status,
        f"app={app_version}, bundled={bundled_version or '<none>'}, expected={expected_version}",
    )


REMOVED_ENV_VAR_FORCE_FLIPS = (
    "VMLX_DSV4_ALLOW_CHAT",
    "VMLX_DSV4_ALLOW_THINKING",
    "VMLX_DSV4_FORCE_DIRECT_RAIL",
)

BUNDLED_SOURCE_HASH_PATHS = (
    "server.py",
    "api/anthropic_adapter.py",
    "api/ollama_adapter.py",
    "block_disk_store.py",
    "cli.py",
    "disk_cache.py",
    "engine/batched.py",
    "loaders/load_jangtq_dsv4.py",
    "mllm_batch_generator.py",
    "mllm_scheduler.py",
    "model_configs.py",
    "model_config_registry.py",
    "models/mllm.py",
    "models/step3p7_mlx_vlm.py",
    "omni_multimodal.py",
    "paged_cache.py",
    "prefix_cache.py",
    "runtime_patches/gemma4_processing.py",
    "scheduler.py",
    "utils/single_batch_generator.py",
    "utils/head_dim_detection.py",
    "utils/ssm_companion_cache.py",
    "utils/ssm_companion_disk_store.py",
    "utils/jang_loader.py",
    "utils/tokenizer.py",
    "chat_templates/gemma4.jinja",
    "config/defaults.yaml",
    "metal/codebook_matvec.metal",
    "metal/codebook_moe.metal",
)

JANG_TOOLS_SOURCE_HASH_PATHS = (
    "capabilities.py",
    "convert.py",
    "convert_hy3_jangtq.py",
    "loader.py",
    "load_jangtq.py",
    "load_jangtq_vlm.py",
    "load_jangtq_kimi_vlm.py",
    "dsv4/mlx_model.py",
    "dsv4/pool_quant_cache.py",
    "hy3/__init__.py",
    "hy3/model.py",
    "hy3/runtime.py",
    "kimi_prune/generate_vl.py",
    "kimi_prune/runtime_patch.py",
    "mimo_v2/mlx_model.py",
    "step37/__init__.py",
    "step37/nvfp4_codec.py",
    "step37/step3p7_mlx.py",
    "topk_override.py",
    "turboquant/fused_gate_up_kernel.py",
    "turboquant/gather_tq_kernel.py",
    "turboquant/hadamard_kernel.py",
    "turboquant/mpp_nax_kernel.py",
    "turboquant/tq_kernel.py",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_bundled_source_file_hashes(
    gate: Gate,
    engine_dir: Path,
    *,
    source_dir: Path | None = None,
    rel_paths: tuple[str, ...] = BUNDLED_SOURCE_HASH_PATHS,
) -> None:
    """Compare critical bundled engine source files against this checkout."""
    source_dir = source_dir or (ROOT / "vmlx_engine")
    mismatches: list[str] = []
    for rel in rel_paths:
        src = source_dir / rel
        bundled = engine_dir / rel
        if not src.exists() or not bundled.exists():
            missing = []
            if not src.exists():
                missing.append(f"source missing {src}")
            if not bundled.exists():
                missing.append(f"bundled missing {bundled}")
            gate.record("bundled source content hash", "FAIL", "; ".join(missing))
            return
        src_hash = _sha256(src)
        bundled_hash = _sha256(bundled)
        if src_hash != bundled_hash:
            mismatches.append(f"{rel}: source={src_hash[:12]} bundled={bundled_hash[:12]}")
    if mismatches:
        gate.record(
            "bundled source content hash",
            "FAIL",
            ", ".join(mismatches),
        )
        return
    gate.record("bundled source content hash", "PASS", ", ".join(rel_paths))


def check_bundled_package_file_hashes(
    gate: Gate,
    package_name: str,
    bundled_dir: Path,
    source_dir: Path,
    *,
    rel_paths: tuple[str, ...],
) -> None:
    """Compare critical bundled package files against local release sources."""
    if not source_dir.exists():
        gate.record(
            f"bundled {package_name} content hash",
            "SKIP",
            f"source package not present: {source_dir}",
        )
        return

    mismatches: list[str] = []
    for rel in rel_paths:
        src = source_dir / rel
        bundled = bundled_dir / rel
        if not src.exists() or not bundled.exists():
            missing = []
            if not src.exists():
                missing.append(f"source missing {src}")
            if not bundled.exists():
                missing.append(f"bundled missing {bundled}")
            gate.record(f"bundled {package_name} content hash", "FAIL", "; ".join(missing))
            return
        src_hash = _sha256(src)
        bundled_hash = _sha256(bundled)
        if src_hash != bundled_hash:
            mismatches.append(f"{rel}: source={src_hash[:12]} bundled={bundled_hash[:12]}")
    if mismatches:
        gate.record(
            f"bundled {package_name} content hash",
            "FAIL",
            ", ".join(mismatches),
        )
        return
    gate.record(f"bundled {package_name} content hash", "PASS", ", ".join(rel_paths))


def check_no_removed_env_var_force_flips(gate: Gate, engine_dir: Path) -> None:
    """Block stale bundled engines that still contain removed DSV4 force-flips."""
    hits: list[str] = []
    for path in sorted(engine_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError as exc:
            gate.record(
                "bundled removed env-var gate",
                "FAIL",
                f"cannot read {path}: {exc}",
            )
            return
        for forbidden in REMOVED_ENV_VAR_FORCE_FLIPS:
            if forbidden in text:
                rel = path.relative_to(engine_dir)
                hits.append(f"{rel}:{forbidden}")
    if hits:
        head = ", ".join(hits[:8])
        more = f" (+{len(hits) - 8} more)" if len(hits) > 8 else ""
        gate.record(
            "bundled removed env-var gate",
            "FAIL",
            f"{head}{more}",
        )
        return
    gate.record("bundled removed env-var gate", "PASS", str(engine_dir))


def packaged_engine_dir(gate: Gate, py: Path) -> Path | None:
    proc = gate.run(
        "packaged bundled engine path",
        [
            str(py),
            "-B",
            "-s",
            "-c",
            "import pathlib, vmlx_engine; print(pathlib.Path(vmlx_engine.__file__).resolve().parent)",
        ],
        cwd=gate.log_dir,
        env=packaged_python_env(gate),
        timeout=60,
    )
    value = _last_nonempty_stdout_line(proc.stdout)
    if not value:
        gate.record("packaged bundled engine path", "FAIL", "empty vmlx_engine path")
        return None
    return Path(value)


def packaged_package_dir(gate: Gate, py: Path, module_name: str) -> Path | None:
    proc = gate.run(
        f"packaged bundled {module_name} path",
        [
            str(py),
            "-B",
            "-s",
            "-c",
            f"import pathlib, {module_name}; print(pathlib.Path({module_name}.__file__).resolve().parent)",
        ],
        cwd=gate.log_dir,
        env=packaged_python_env(gate),
        timeout=60,
    )
    value = _last_nonempty_stdout_line(proc.stdout)
    if not value:
        gate.record(f"packaged bundled {module_name} path", "FAIL", f"empty {module_name} path")
        return None
    return Path(value)


def check_packaged_console_script_shebangs(gate: Gate, app: Path) -> None:
    bin_dir = app / "Contents" / "Resources" / "bundled-python" / "python" / "bin"
    if not bin_dir.is_dir():
        gate.record("packaged console-script shebangs", "FAIL", f"missing {bin_dir}")
        return

    offenders: list[str] = []
    for script in sorted(bin_dir.iterdir()):
        if not script.is_file():
            continue
        try:
            first = script.open("rb").readline(4096).decode("utf-8", "replace").strip()
        except OSError:
            continue
        if not first.startswith("#!"):
            continue
        if (
            "python" in first
            or "/Users/" in first
            or "panel/bundled-python" in first
            or "/Applications/vMLX.app" in first
        ):
            offenders.append(f"{script.name}: {first}")

    if offenders:
        gate.record(
            "packaged console-script shebangs",
            "FAIL",
            "; ".join(offenders[:12]),
        )
        return
    gate.record("packaged console-script shebangs", "PASS", str(bin_dir))


def expected_developer_id_team_id() -> str:
    panel_pkg = json.loads((PANEL / "package.json").read_text())
    return str(panel_pkg.get("build", {}).get("mac", {}).get("notarize", {}).get("teamId", ""))


def jang_tools_source_root() -> Path:
    configured = os.environ.get("VMLX_JANG_TOOLS_SOURCE") or os.environ.get("VMLINUX_JANG_TOOLS_SOURCE")
    return Path(configured or (Path.home() / "jang" / "jang-tools"))


def check_objective_proof_digest(
    gate: Gate,
    *,
    digest_path: Path | None = None,
) -> None:
    """Refresh and enforce the objective proof digest before release checks pass."""
    digest_path = digest_path or (
        ROOT / "build/current-objective-proof-audit-20260531-nemotron-exact-finalizer-ledger.json"
    )
    proc = gate.run(
        "objective proof digest refresh",
        [
            sys.executable,
            "tests/cross_matrix/summarize_objective_proof.py",
            "--out",
            str(digest_path),
        ],
        timeout=180,
    )
    if proc.returncode != 0:
        return
    try:
        digest = json.loads(digest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        gate.record("objective proof digest", "FAIL", f"cannot read {digest_path}: {exc}")
        return

    open_requirements = [
        str(item.get("requirement"))
        for item in digest.get("requirements", [])
        if item.get("status") != "pass"
    ]
    if open_requirements:
        gate.record("objective proof digest", "FAIL", "; ".join(open_requirements))
        return
    gate.record("objective proof digest", "PASS", str(digest_path))


def check_release_ready_manifest(gate: Gate) -> None:
    """Block release/notarization when the current release ledger is still open."""
    out = gate.log_dir / "release-ready-manifest.json"
    gate.run(
        "release-ready manifest",
        [
            sys.executable,
            "tests/cross_matrix/run_release_regression_manifest.py",
            "--require-release-ready",
            "--out",
            str(out),
        ],
        timeout=240,
    )


def check_packaged_developer_id_signature(
    gate: Gate,
    app: Path,
    *,
    expected_team_id: str | None = None,
) -> None:
    """Block ad-hoc app signatures from being treated as release signatures."""
    expected_team_id = expected_team_id if expected_team_id is not None else expected_developer_id_team_id()
    proc = gate.run(
        "packaged signature details",
        ["codesign", "-dv", "--verbose=4", str(app)],
        timeout=60,
        allow_fail=True,
    )
    output = proc.stdout or ""
    if proc.returncode != 0:
        gate.record(
            "packaged Developer ID signature",
            "FAIL",
            f"codesign -dv failed with exit={proc.returncode}",
        )
        return
    if "Signature=adhoc" in output:
        gate.record(
            "packaged Developer ID signature",
            "FAIL",
            "ad-hoc signature is not a release Developer ID signature",
        )
        return

    authorities = [
        line.split("=", 1)[1].strip()
        for line in output.splitlines()
        if line.startswith("Authority=")
    ]
    team_id = ""
    for line in output.splitlines():
        if line.startswith("TeamIdentifier="):
            team_id = line.split("=", 1)[1].strip()
            break
    has_hardened_runtime = any(
        line.startswith("CodeDirectory ") and "(runtime)" in line
        for line in output.splitlines()
    )

    has_developer_id = any(auth.startswith("Developer ID Application:") for auth in authorities)
    if not has_developer_id:
        gate.record(
            "packaged Developer ID signature",
            "FAIL",
            "missing Developer ID Application authority",
        )
        return
    if expected_team_id and team_id != expected_team_id:
        gate.record(
            "packaged Developer ID signature",
            "FAIL",
            f"team={team_id or '<none>'}, expected={expected_team_id}",
        )
        return
    if not has_hardened_runtime:
        gate.record(
            "packaged Developer ID signature",
            "FAIL",
            "missing hardened runtime signature flag required for notarization",
        )
        return

    entitlements_proc = gate.run(
        "packaged entitlements details",
        ["codesign", "--display", "--entitlements", ":-", str(app)],
        timeout=60,
        allow_fail=True,
    )
    entitlements_output = entitlements_proc.stdout or ""
    if entitlements_proc.returncode != 0:
        gate.record(
            "packaged Developer ID signature",
            "FAIL",
            f"codesign entitlements failed with exit={entitlements_proc.returncode}",
        )
        return
    try:
        entitlements = plistlib.loads(entitlements_output.encode("utf-8"))
    except Exception as exc:
        gate.record(
            "packaged Developer ID signature",
            "FAIL",
            f"could not parse packaged release entitlements: {type(exc).__name__}",
        )
        return
    missing_entitlements = [
        key for key in REQUIRED_RELEASE_ENTITLEMENTS if not entitlements.get(key)
    ]
    if missing_entitlements:
        gate.record(
            "packaged Developer ID signature",
            "FAIL",
            "missing release entitlements: " + ", ".join(missing_entitlements),
        )
        return

    gate.record("packaged Developer ID signature", "PASS", f"team={team_id}")


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

    gate.run(
        "twine check dist",
        [*twine_command(), "check", *map(str, sorted((ROOT / "dist").glob("vmlx-*")))],
        env=twine_env(gate),
        timeout=120,
    )
    gate.run("panel request/type tests", ["npm", "test", "--", "request-builder.test.ts", "reasoning-display.test.ts", "audit-fixes.test.ts"], cwd=PANEL, timeout=180)
    gate.run("panel typecheck", ["npm", "run", "typecheck"], cwd=PANEL, timeout=180)
    gate.run("bundled python import gate", ["npm", "run", "verify-bundled"], cwd=PANEL, timeout=180)
    check_objective_proof_digest(gate)
    check_release_ready_manifest(gate)

    if skip_app:
        gate.record("packaged app checks", "WARN", "skipped by --skip-app")
        return
    if not app.exists():
        gate.record("packaged app exists", "FAIL", str(app))
        return
    gate.record("packaged app exists", "PASS", str(app))
    info = app / "Contents" / "Info.plist"
    plist = plistlib.loads(info.read_bytes())
    app_version = str(plist.get("CFBundleShortVersionString"))
    gate.record("packaged app version", "PASS" if app_version == version else "FAIL", app_version)
    py = packaged_python(app)
    check_packaged_bundled_import_version(gate, py, version, app_version)
    engine_dir = packaged_engine_dir(gate, py)
    if engine_dir is not None:
        check_bundled_source_file_hashes(gate, engine_dir)
        check_no_removed_env_var_force_flips(gate, engine_dir)
    check_packaged_console_script_shebangs(gate, app)
    jang_tools_dir = packaged_package_dir(gate, py, "jang_tools")
    jang_tools_source = jang_tools_source_root() / "jang_tools"
    if jang_tools_dir is not None:
        check_bundled_package_file_hashes(
            gate,
            "jang_tools",
            jang_tools_dir,
            jang_tools_source,
            rel_paths=JANG_TOOLS_SOURCE_HASH_PATHS,
        )
    check_no_packaged_pycache(gate, app)
    check_packaged_developer_id_signature(gate, app)
    gate.run("codesign strict verify", ["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app)], timeout=180, allow_fail=False)
    gate.run("spctl assessment", ["spctl", "--assess", "--type", "execute", "--verbose=4", str(app)], timeout=120, allow_fail=True)


def launch_app_smoke(gate: Gate, app: Path) -> None:
    if not app.exists():
        gate.record("GUI open smoke", "FAIL", f"missing app {app}")
        return
    exe = app / "Contents" / "MacOS" / "vMLX"
    if not exe.exists():
        gate.record("GUI open smoke", "FAIL", f"missing executable {exe}")
        return

    # Do not use `open`: the packaged dev app and installed app intentionally
    # share the production bundle id, so LaunchServices can activate
    # /Applications/vMLX.app and produce a false pass. Direct-launch the exact
    # executable with isolated user data instead.
    user_data = gate.log_dir / "gui-user-data"
    log_path = gate.log_dir / "gui_direct_launch.log"
    shutil.rmtree(user_data, ignore_errors=True)
    started = time.time()
    with log_path.open("w") as fp:
        proc = subprocess.Popen(
            [str(exe), f"--user-data-dir={user_data}"],
            cwd=str(ROOT),
            stdout=fp,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    try:
        time.sleep(8)
        if proc.poll() is not None:
            gate.record(
                "GUI direct launch",
                "FAIL",
                f"exit={proc.returncode}; log={log_path}",
            )
            return
        gate.record(
            "GUI direct launch",
            "PASS",
            f"pid={proc.pid}; {time.time() - started:.1f}s; log={log_path}",
        )
        exact = gate.run(
            "GUI process exact path",
            ["pgrep", "-fl", str(exe)],
            timeout=30,
            allow_fail=True,
        )
        if exact.returncode == 0:
            gate.record("GUI open smoke verdict", "PASS", "repo-local executable is running")
        else:
            gate.record("GUI open smoke verdict", "FAIL", "repo-local executable not found")
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
        shutil.rmtree(user_data, ignore_errors=True)


def live_engine_gate(gate: Gate, app: Path, model: str, port: int, skip_sleep_wake: bool, thinking: str) -> None:
    py = packaged_python(app) if app.exists() else Path(sys.executable)
    if not Path(model).exists() and "/" not in model:
        gate.record("live model path", "FAIL", model)
        return
    log = gate.log_dir / "live_engine_server.log"
    prompt_cache_dir = gate.log_dir / "prompt-cache"
    block_cache_dir = gate.log_dir / "block-cache"
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
        "--continuous-batching",
        "--max-num-seqs",
        "1",
        "--enable-prefix-cache",
        "--use-paged-cache",
        "--enable-disk-cache",
        "--disk-cache-dir",
        str(gate.log_dir / "prompt-cache"),
        "--enable-block-disk-cache",
        "--block-disk-cache-dir",
        str(gate.log_dir / "block-cache"),
        "--block-disk-cache-max-gb",
        "1",
        "--default-temperature",
        "0",
        "--default-top-p",
        "1",
        "--max-tokens",
        "512",
    ]
    live_env = os.environ.copy()
    live_env.update(packaged_python_env(gate))
    prompt_cache_dir.mkdir(parents=True, exist_ok=True)
    block_cache_dir.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fp:
        proc = subprocess.Popen(
            cmd,
            cwd=str(gate.log_dir),
            env=live_env,
            stdout=fp,
            stderr=subprocess.STDOUT,
            text=True,
        )
    base = f"http://127.0.0.1:{port}"
    try:
        health = wait_health(base)
        write_json_log(gate, "Live server health", health)
        gate.record("live server health", "PASS", json.dumps(health)[:240])

        chat = request_json(
            "POST",
            f"{base}/v1/chat/completions",
            apply_thinking({
                "model": "local",
                "messages": [{"role": "user", "content": "Answer with exactly: Paris"}],
                "temperature": 0,
                "max_tokens": 64,
            }, thinking),
        )
        write_json_log(gate, "OpenAI chat response", chat)
        assert_visible_text("OpenAI chat visible output", chat, gate)

        mt = request_json(
            "POST",
            f"{base}/v1/chat/completions",
            apply_thinking({
                "model": "local",
                "messages": [
                    {"role": "user", "content": "Remember this word: teal. Reply OK."},
                    {"role": "assistant", "content": "OK."},
                    {"role": "user", "content": "What word did I ask you to remember?"},
                ],
                "temperature": 0,
                "max_tokens": 96,
            }, thinking),
        )
        write_json_log(gate, "OpenAI multi turn response", mt)
        text = assert_visible_text("OpenAI multi-turn recall", mt, gate)
        if "teal" not in text.lower():
            gate.record("OpenAI multi-turn recall exact", "FAIL", text[:180])
        else:
            gate.record("OpenAI multi-turn recall exact", "PASS", text[:180])

        warm1_messages = [
            {"role": "user", "content": CACHE_WARM_PROMPT},
        ]
        warm1 = request_json(
            "POST",
            f"{base}/v1/chat/completions",
            apply_thinking({
                "model": "local",
                "messages": warm1_messages,
                "temperature": 0,
                "max_tokens": 32,
            }, thinking),
        )
        write_json_log(gate, "OpenAI cache warm turn1 response", warm1)
        warm1_text = assert_visible_text("OpenAI cache warm turn1", warm1, gate)

        warm2_messages = [
            *warm1_messages,
            {"role": "assistant", "content": warm1_text},
            {"role": "user", "content": "What cache probe word did I ask you to remember?"},
        ]
        warm2 = request_json(
            "POST",
            f"{base}/v1/chat/completions",
            apply_thinking({
                "model": "local",
                "messages": warm2_messages,
                "temperature": 0,
                "max_tokens": 96,
            }, thinking),
        )
        write_json_log(gate, "OpenAI cache warm turn2 response", warm2)
        warm2_text = assert_visible_text("OpenAI cache warm turn2", warm2, gate)
        if "cobalt" not in warm2_text.lower():
            gate.record("OpenAI cache warm turn2 recall exact", "FAIL", warm2_text[:180])
        else:
            gate.record("OpenAI cache warm turn2 recall exact", "PASS", warm2_text[:180])
        warm2_cached = cached_tokens_from_usage(warm2)
        if warm2_cached <= 0:
            gate.record(
                "OpenAI cross-request cache hit",
                "FAIL",
                json.dumps(warm2.get("usage") if isinstance(warm2, dict) else warm2)[:240],
            )
            raise AssertionError("OpenAI cross-request cache hit: cached_tokens=0")
        gate.record("OpenAI cross-request cache hit", "PASS", f"cached_tokens={warm2_cached}")

        responses = request_json(
            "POST",
            f"{base}/v1/responses",
            apply_thinking({
                "model": "local",
                "input": "Answer with exactly: 4",
                "temperature": 0,
                "max_output_tokens": 128,
            }, thinking),
        )
        write_json_log(gate, "Responses API response", responses)
        assert_visible_text("Responses visible output", responses, gate)

        anthropic = request_json(
            "POST",
            f"{base}/v1/messages",
            apply_anthropic_thinking({
                "model": "local",
                "max_tokens": 96,
                "messages": [{"role": "user", "content": "Answer with exactly: blue"}],
            }, thinking),
        )
        write_json_log(gate, "Anthropic response", anthropic)
        assert_visible_text("Anthropic visible output", anthropic, gate)

        ollama_body = {
            "model": "local",
            "messages": [{"role": "user", "content": "Answer with exactly: green"}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 96},
        }
        if thinking != "auto":
            # Native Ollama field. The server also accepts enable_thinking as
            # a vMLX extension, but the release gate should exercise the real
            # public Ollama contract.
            ollama_body["think"] = thinking == "on"
        ollama = request_json(
            "POST",
            f"{base}/api/chat",
            ollama_body,
        )
        write_json_log(gate, "Ollama response", ollama)
        assert_visible_text("Ollama visible output", ollama, gate)

        stats = request_json("GET", f"{base}/v1/cache/stats")
        write_json_log(gate, "Cache stats after API matrix", stats)
        gate.record("cache stats after API matrix", "PASS", json.dumps(stats)[:600])

        if not skip_sleep_wake:
            request_json("POST", f"{base}/admin/soft-sleep", {})
            time.sleep(2)
            wake_resp = request_json(
                "POST",
                f"{base}/v1/chat/completions",
                apply_thinking({
                    "model": "local",
                    "messages": [{"role": "user", "content": "After wake, answer with exactly: awake"}],
                    "temperature": 0,
                    "max_tokens": 96,
                }, thinking),
                timeout=240,
            )
            write_json_log(gate, "JIT soft wake response", wake_resp)
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
    parser.add_argument("--thinking", choices=["auto", "off", "on"], default="auto", help="Per-request thinking mode for live API checks")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    gate = Gate(INTERNAL / ts)
    app = Path(args.app).expanduser().resolve()

    try:
        check_static(gate, app, args.skip_app)
        if not args.skip_gui and not args.skip_app:
            launch_app_smoke(gate, app)
        if args.model:
            live_engine_gate(gate, app, args.model, args.port, args.skip_sleep_wake, args.thinking)
    except Exception as exc:  # noqa: BLE001
        gate.record("release gate exception", "FAIL", repr(exc))
    summary = gate.write_summary(args)
    print(f"\nSummary: {summary}")
    return 1 if gate.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
