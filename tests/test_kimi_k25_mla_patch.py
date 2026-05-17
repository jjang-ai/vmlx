# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Kimi K2.6 MLA runtime patch guards."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


def _load_patch_module():
    path = Path("vmlx_engine/runtime_patches/kimi_k25_mla.py").resolve()
    spec = importlib.util.spec_from_file_location("kimi_k25_mla_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_uv_tool_vmlx_venv_is_not_treated_as_packaged_app_bundle():
    mod = _load_patch_module()
    path = Path(
        "/Users/u/.local/share/uv/tools/vmlx/lib/python3.13/site-packages/"
        "mlx_lm/models/deepseek_v3.py"
    )

    assert mod._is_vmlx_bundle(path) is False


def test_pipx_vmlx_venv_is_not_treated_as_packaged_app_bundle():
    mod = _load_patch_module()
    path = Path(
        "/Users/u/.local/pipx/venvs/vmlx/lib/python3.13/site-packages/"
        "mlx_lm/models/deepseek_v3.py"
    )

    assert mod._is_vmlx_bundle(path) is False


def test_packaged_vmlx_app_contents_is_treated_as_bundle():
    mod = _load_patch_module()
    path = Path(
        "/Applications/vMLX.app/Contents/Resources/bundled-python/python/lib/"
        "python3.12/site-packages/mlx_lm/models/deepseek_v3.py"
    )

    assert mod._is_vmlx_bundle(path) is True


def test_kimi_patch_module_does_not_claim_class_monkeypatch_fallback():
    source = Path("vmlx_engine/runtime_patches/kimi_k25_mla.py").read_text()

    assert "monkey-patched into the class" not in source


def test_install_skips_missing_optional_patch_source_without_error_output(
    monkeypatch,
    capsys,
    tmp_path,
):
    mod = _load_patch_module()
    target = tmp_path / "deepseek_v3.py"
    target.write_text("# unpatched\n")

    runtime_patch = types.ModuleType("jang_tools.kimi_prune.runtime_patch")
    runtime_patch.PATCH_SRC = tmp_path / "missing_deepseek_v3_patched.py"

    def noisy_apply(dry_run=False):
        print(f"[patch] ERROR: patched source not found: {runtime_patch.PATCH_SRC}", file=sys.stderr)
        return 2

    runtime_patch.apply = noisy_apply
    monkeypatch.setitem(sys.modules, "jang_tools.kimi_prune.runtime_patch", runtime_patch)
    monkeypatch.setattr(mod, "is_patched", lambda: False)
    monkeypatch.setattr(mod, "_locate_target", lambda: target)
    monkeypatch.setattr(mod, "_is_vmlx_bundle", lambda path: False)

    assert mod.install() is False
    captured = capsys.readouterr()
    assert "patched source not found" not in captured.err
    assert "patched source not found" not in captured.out
