#!/usr/bin/env python3
"""No-heavy runtime audit for GitHub issues #181-#183.

This gate indexes recent source/runtime issue fixes so the release sweep can
name them directly. It does not replace live model/UI proof for broader release
clearance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_OUT = Path(
    "build/current-issue181-183-runtime-audit-20260601-qwen3vl-minicpm-mpp.json"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _issue181_checks(root: Path) -> dict[str, bool]:
    cli = _read(root / "vmlx_engine/cli.py")
    server = _read(root / "vmlx_engine/server.py")
    engine_tests = _read(root / "tests/test_engine_audit.py")
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
    }


def _issue182_checks(root: Path) -> dict[str, bool]:
    compat = _read(root / "vmlx_engine/utils/mlx_vlm_compat.py")
    mtp_qwen = _read(root / "vmlx_engine/patches/mlx_vlm_mtp/qwen35_vl.py")
    mllm_tests = _read(root / "tests/test_mllm.py")
    mtp_tests = _read(root / "tests/test_native_mtp_autodetect.py")
    verify_bundled = _read(root / "panel/scripts/verify-bundled-python.sh")
    release_gate = _read(root / "panel/scripts/release-gate-python-app.py")
    packaged_contract = _read(root / "tests/cross_matrix/run_packaged_integrity_contract.py")
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
    }


def _issue183_checks(root: Path) -> dict[str, bool]:
    verify_bundled = _read(root / "panel/scripts/verify-bundled-python.sh")
    engine_tests = _read(root / "tests/test_engine_audit.py")
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
    }


def build_audit(root: Path) -> dict:
    root = root.resolve()
    issues = {
        "181": {
            "title": (
                "JANGTQ_MPP_NAX=auto appears to produce incorrect prefill logits "
                "for MXTQ/hybrid JANGTQ models in vmlx serve"
            ),
            "checks": _issue181_checks(root),
            "release_clearance": "source_and_packaged_mpp_auto_policy_guarded",
        },
        "182": {
            "title": "Qwen VL patch-embed layout must load in MLX Conv3D format",
            "checks": _issue182_checks(root),
            "release_clearance": (
                "source_and_packaged_qwen_vl_patch_embed_layout_guarded"
            ),
        },
        "183": {
            "title": "MiniCPM-V-4.6 model_type must remap to mlx-vlm minicpmo",
            "checks": _issue183_checks(root),
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
        "focused_failures": focused_failures,
        "release_boundary": (
            "Issues #181-#183 have focused no-heavy source/packaged guard "
            "coverage. Broader release clearance still depends on the full live "
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
