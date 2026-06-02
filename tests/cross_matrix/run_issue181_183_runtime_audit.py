#!/usr/bin/env python3
"""No-heavy runtime audit for GitHub issues #181-#183.

This gate indexes recent source/runtime issue fixes so the release sweep can
name them directly. It does not replace live model/UI proof for broader release
clearance.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


DEFAULT_OUT = Path(
    "build/current-issue181-183-runtime-audit-20260602-v1554-installed-tahoe-refresh.json"
)
INSTALLED_PYTHON = Path(
    "/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _run_installed_minicpm_v46_probe(
    python_path: Path = INSTALLED_PYTHON,
) -> dict[str, object]:
    if not python_path.exists():
        return {
            "returncode": 127,
            "error": f"missing installed python: {python_path}",
            "checks": {},
        }
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = ""
    code = r"""
import json
import vmlx_engine
from mlx_vlm.prompt_utils import MODEL_CONFIG
from mlx_vlm.utils import MODEL_REMAPPING, get_model_and_args

module, model_type = get_model_and_args({"model_type": "minicpmv4_6"})
payload = {
    "model_remapping": MODEL_REMAPPING.get("minicpmv4_6"),
    "model_config_present": "minicpmv4_6" in MODEL_CONFIG,
    "model_config_matches_minicpmo": MODEL_CONFIG.get("minicpmv4_6") == MODEL_CONFIG.get("minicpmo"),
    "resolved_model_type": model_type,
    "resolved_module": getattr(module, "__name__", ""),
    "resolved_has_model": hasattr(module, "Model"),
}
print(json.dumps(payload, sort_keys=True))
"""
    proc = subprocess.run(
        [str(python_path), "-B", "-s", "-c", code],
        cwd="/tmp",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    payload: dict[str, object] = {}
    if proc.returncode == 0 and proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout.splitlines()[-1])
        except Exception as exc:  # noqa: BLE001 - report malformed probe output
            payload = {"parse_error": f"{type(exc).__name__}: {exc}"}
    checks = {
        "installed_model_remapping": payload.get("model_remapping") == "minicpmo",
        "installed_prompt_config_present": payload.get("model_config_present") is True,
        "installed_prompt_config_matches_minicpmo": (
            payload.get("model_config_matches_minicpmo") is True
        ),
        "installed_get_model_and_args_resolves_minicpmo": (
            payload.get("resolved_model_type") == "minicpmo"
            and payload.get("resolved_has_model") is True
            and str(payload.get("resolved_module", "")).endswith(".minicpmo")
        ),
    }
    return {
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout.splitlines()[-5:],
        "stderr_tail": proc.stderr.splitlines()[-20:],
        "payload": payload,
        "checks": checks,
    }


def _run_installed_mpp_policy_probe(
    python_path: Path = INSTALLED_PYTHON,
) -> dict[str, object]:
    if not python_path.exists():
        return {
            "returncode": 127,
            "error": f"missing installed python: {python_path}",
            "checks": {},
        }
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = ""
    code = r"""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

from vmlx_engine.cli import _apply_jangtq_mpp_nax_policy


class Logger:
    def __init__(self):
        self.messages = []

    def info(self, *args):
        self.messages.append(("info", " ".join(str(item) for item in args)))

    def warning(self, *args):
        self.messages.append(("warning", " ".join(str(item) for item in args)))

    def debug(self, *args):
        self.messages.append(("debug", " ".join(str(item) for item in args)))


logger = Logger()
os.environ.pop("JANGTQ_MPP_NAX", None)
os.environ.pop("JANGTQ_TOPK_OVERRIDE", None)
with tempfile.TemporaryDirectory(prefix="MiniMax-M2.7-JANGTQ_K-") as model_dir:
    Path(model_dir, "config.json").write_text(json.dumps({"mxtq_bits": 4}))
    auto_mode = _apply_jangtq_mpp_nax_policy(SimpleNamespace(model=model_dir), logger)
    auto_env = os.environ.get("JANGTQ_MPP_NAX")
    os.environ["JANGTQ_MPP_NAX"] = "on"
    explicit_mode = _apply_jangtq_mpp_nax_policy(
        SimpleNamespace(model=model_dir),
        logger,
    )
    explicit_env = os.environ.get("JANGTQ_MPP_NAX")

payload = {
    "auto_mode": auto_mode,
    "auto_env": auto_env,
    "explicit_mode": explicit_mode,
    "explicit_env": explicit_env,
    "messages": logger.messages,
}
print(json.dumps(payload, sort_keys=True))
"""
    proc = subprocess.run(
        [str(python_path), "-B", "-s", "-c", code],
        cwd="/tmp",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    payload: dict[str, object] = {}
    if proc.returncode == 0 and proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout.splitlines()[-1])
        except Exception as exc:  # noqa: BLE001 - report malformed probe output
            payload = {"parse_error": f"{type(exc).__name__}: {exc}"}
    checks = {
        "installed_auto_mxtq_jangtq_disables_mpp": (
            payload.get("auto_mode") == "off" and payload.get("auto_env") == "off"
        ),
        "installed_explicit_mpp_on_still_allowed": (
            payload.get("explicit_mode") == "on" and payload.get("explicit_env") == "1"
        ),
    }
    return {
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout.splitlines()[-5:],
        "stderr_tail": proc.stderr.splitlines()[-20:],
        "payload": payload,
        "checks": checks,
    }


def _run_installed_qwen_vl_patch_probe(
    python_path: Path = INSTALLED_PYTHON,
) -> dict[str, object]:
    if not python_path.exists():
        return {
            "returncode": 127,
            "error": f"missing installed python: {python_path}",
            "checks": {},
        }
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = ""
    code = r"""
import json

import mlx.core as mx
from vmlx_engine.utils import mlx_vlm_compat
from vmlx_engine.patches.mlx_vlm_mtp import qwen35_vl

value = mx.zeros((2, 3, 2, 4, 4), dtype=mx.float32)
normal = mlx_vlm_compat._qwen35_patch_embed_to_mlx_layout(
    "vision_tower.patch_embed.proj.weight",
    value,
)
native = qwen35_vl._qwen35_patch_embed_to_mlx_layout(
    "model.visual.patch_embed.proj.weight",
    value,
)
already_mlx = mx.zeros((2, 2, 4, 4, 3), dtype=mx.float32)
unchanged = mlx_vlm_compat._qwen35_patch_embed_to_mlx_layout(
    "vision_tower.patch_embed.proj.weight",
    already_mlx,
)
payload = {
    "normal_shape": list(normal.shape),
    "native_shape": list(native.shape),
    "already_mlx_shape": list(unchanged.shape),
}
print(json.dumps(payload, sort_keys=True))
"""
    proc = subprocess.run(
        [str(python_path), "-B", "-s", "-c", code],
        cwd="/tmp",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    payload: dict[str, object] = {}
    if proc.returncode == 0 and proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout.splitlines()[-1])
        except Exception as exc:  # noqa: BLE001 - report malformed probe output
            payload = {"parse_error": f"{type(exc).__name__}: {exc}"}
    expected = [2, 2, 4, 4, 3]
    checks = {
        "installed_normal_vlm_patch_embed_transposes": (
            payload.get("normal_shape") == expected
        ),
        "installed_native_mtp_patch_embed_transposes": (
            payload.get("native_shape") == expected
        ),
        "installed_mlx_layout_stays_unchanged": (
            payload.get("already_mlx_shape") == expected
        ),
    }
    return {
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout.splitlines()[-5:],
        "stderr_tail": proc.stderr.splitlines()[-20:],
        "payload": payload,
        "checks": checks,
    }


def _issue181_checks(
    root: Path,
    installed_mpp_probe: dict[str, object],
) -> dict[str, bool]:
    cli = _read(root / "vmlx_engine/cli.py")
    server = _read(root / "vmlx_engine/server.py")
    engine_tests = _read(root / "tests/test_engine_audit.py")
    installed_checks = installed_mpp_probe.get("checks")
    if not isinstance(installed_checks, dict):
        installed_checks = {}
    return {
        "mpp_auto_policy_function_exists": "def _apply_jangtq_mpp_nax_policy" in cli,
        "mpp_auto_disabled_for_mxtq": "MXTQ/JANGTQ bundle detected" in cli
        and 'os.environ["JANGTQ_MPP_NAX"] = "1" if mode == "on" else mode' in cli,
        "jangtq_repo_id_disables_auto": (
            "test_jangtq_mpp_nax_cli_policy_disables_auto_for_jangtq_repo_id"
            in engine_tests
        ),
        "explicit_mpp_on_still_allowed": (
            "test_jangtq_mpp_nax_cli_policy_allows_explicit_on_for_kernel_diagnostics"
            in engine_tests
            and "test_jangtq_mpp_nax_cli_policy_can_force_on" in engine_tests
        ),
        "server_health_reports_mpp_status": "def _jangtq_mpp_nax_runtime_status" in server
        and '"jangtq_acceleration"' in server,
        "installed_app_mpp_auto_policy_disables_mxtq": (
            installed_checks.get("installed_auto_mxtq_jangtq_disables_mpp") is True
            and installed_checks.get("installed_explicit_mpp_on_still_allowed") is True
        ),
    }


def _issue182_checks(
    root: Path,
    installed_qwen_vl_probe: dict[str, object],
) -> dict[str, bool]:
    compat = _read(root / "vmlx_engine/utils/mlx_vlm_compat.py")
    mtp_qwen = _read(root / "vmlx_engine/patches/mlx_vlm_mtp/qwen35_vl.py")
    mllm_tests = _read(root / "tests/test_mllm.py")
    mtp_tests = _read(root / "tests/test_native_mtp_autodetect.py")
    verify_bundled = _read(root / "panel/scripts/verify-bundled-python.sh")
    release_gate = _read(root / "panel/scripts/release-gate-python-app.py")
    packaged_contract = _read(root / "tests/cross_matrix/run_packaged_integrity_contract.py")
    installed_checks = installed_qwen_vl_probe.get("checks")
    if not isinstance(installed_checks, dict):
        installed_checks = {}
    return {
        "normal_vlm_patch_embed_transpose": (
            "def _qwen35_patch_embed_to_mlx_layout" in compat
            and "value.transpose(0, 2, 3, 4, 1)" in compat
            and "_patch_qwen35_patch_embed_layout" in compat
        ),
        "native_mtp_patch_embed_transpose": (
            "def _qwen35_patch_embed_to_mlx_layout" in mtp_qwen
            and "value.transpose(0, 2, 3, 4, 1)" in mtp_qwen
            and "_patch_moe_outer_model" in mtp_qwen
        ),
        "focused_shape_regression_test_present": (
            "test_qwen35_vlm_compat_transposes_patch_embed_to_mlx_conv3d_layout"
            in mllm_tests
            and "(1152, 3, 2, 16, 16)" in mllm_tests
            and "dense_fixed[\"vision_tower.patch_embed.proj.weight\"].shape"
            in mllm_tests
        ),
        "native_mtp_shape_regression_test_present": (
            "(1152, 3, 2, 16, 16)" in mtp_tests
            and "model.visual.patch_embed.proj.weight" in mtp_tests
            and "sanitized[\"vision_tower.patch_embed.proj.weight\"].shape"
            in mtp_tests
        ),
        "bundled_hash_gate_covers_runtime": (
            "utils/mlx_vlm_compat.py" in verify_bundled
            and "patches/mlx_vlm_mtp/qwen35_vl.py" in verify_bundled
            and "utils/mlx_vlm_compat.py" in release_gate
            and "patches/mlx_vlm_mtp/qwen35_vl.py" in release_gate
            and "utils/mlx_vlm_compat.py" in packaged_contract
            and "patches/mlx_vlm_mtp/qwen35_vl.py" in packaged_contract
        ),
        "installed_app_qwen_vl_patch_embed_layout": all(
            installed_checks.get(key) is True
            for key in (
                "installed_normal_vlm_patch_embed_transposes",
                "installed_native_mtp_patch_embed_transposes",
                "installed_mlx_layout_stays_unchanged",
            )
        ),
    }


def _issue183_checks(
    root: Path,
    installed_minicpm_probe: dict[str, object],
) -> dict[str, bool]:
    verify_bundled = _read(root / "panel/scripts/verify-bundled-python.sh")
    engine_tests = _read(root / "tests/test_engine_audit.py")
    installed_checks = installed_minicpm_probe.get("checks")
    if not isinstance(installed_checks, dict):
        installed_checks = {}
    return {
        "minicpm_v46_registry_remap": (
            'MODEL_REMAPPING.get("minicpmv4_6") != "minicpmo"' in verify_bundled
            and 'get_model_and_args({"model_type": "minicpmv4_6"})' in verify_bundled
        ),
        "minicpm_v46_prompt_config_remap": (
            '"minicpmv4_6" not in MODEL_CONFIG' in verify_bundled
            and "MiniCPM-V-4.6 mlx_vlm remap + prompt_utils config" in verify_bundled
        ),
        "bundled_import_gate_covers_runtime": (
            "test_bundled_python_import_gate_covers_minicpm_v46_runtime"
            in engine_tests
            and "test_mlx_vlm_registry_patch_remaps_minicpm_v46_to_minicpmo"
            in engine_tests
        ),
        "installed_app_minicpm_v46_runtime_remap": all(
            installed_checks.get(key) is True
            for key in (
                "installed_model_remapping",
                "installed_prompt_config_present",
                "installed_prompt_config_matches_minicpmo",
                "installed_get_model_and_args_resolves_minicpmo",
            )
        ),
    }


def build_audit(
    root: Path,
    *,
    installed_python: Path = INSTALLED_PYTHON,
) -> dict:
    root = root.resolve()
    installed_minicpm_probe = _run_installed_minicpm_v46_probe(installed_python)
    installed_mpp_probe = _run_installed_mpp_policy_probe(installed_python)
    installed_qwen_vl_probe = _run_installed_qwen_vl_patch_probe(installed_python)
    issues = {
        "181": {
            "title": (
                "JANGTQ_MPP_NAX=auto appears to produce incorrect prefill logits "
                "for MXTQ/hybrid JANGTQ models in vmlx serve"
            ),
            "checks": _issue181_checks(root, installed_mpp_probe),
            "release_clearance": "installed_mpp_auto_policy_guarded",
        },
        "182": {
            "title": (
                "升级到新版本后，运行无审查版本模型一直失败 "
                "(Qwen VL patch-embed Conv3D layout mismatch)"
            ),
            "checks": _issue182_checks(root, installed_qwen_vl_probe),
            "release_clearance": (
                "installed_qwen_vl_patch_embed_layout_guarded"
            ),
        },
        "183": {
            "title": "MiniCPM-V-4.6 model_type must remap to mlx-vlm minicpmo",
            "checks": _issue183_checks(root, installed_minicpm_probe),
            "release_clearance": "source_and_packaged_minicpm_v46_load_guarded",
        },
    }
    focused_failures: list[str] = []
    for number, issue in issues.items():
        checks = issue["checks"]
        issue["focused_source_slice"] = "pass" if all(checks.values()) else "fail"
        if issue["focused_source_slice"] != "pass":
            focused_failures.append(number)

    return {
        "artifact": "",
        "status": "fail" if focused_failures else "pass",
        "issues": issues,
        "installed_mpp_policy_probe": installed_mpp_probe,
        "installed_qwen_vl_patch_probe": installed_qwen_vl_probe,
        "installed_minicpm_v46_probe": installed_minicpm_probe,
        "focused_failures": focused_failures,
        "release_boundary": (
            "Issues #181-#183 have focused no-heavy source/packaged guard "
            "coverage; #181, #182, and #183 also include installed app runtime probes. "
            "Broader release clearance still depends on the full live "
            "model/UI/cache/parser matrix."
        ),
    }


def write_audit(root: Path, out: Path) -> dict:
    audit = build_audit(root)
    audit["artifact"] = str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    audit = write_audit(args.root, args.out)
    print(json.dumps({"status": audit["status"], "out": str(args.out)}, sort_keys=True))
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
