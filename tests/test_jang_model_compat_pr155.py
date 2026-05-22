"""Regression contracts for PR #155 MiniMax/JANG compatibility findings.

The PR identified two real failure classes:

* MiniMax JANGTQ bundles with packed expert tensors must not fall through to
  upstream sanitize paths that assume every expert has affine weight keys.
* MoEGate-like modules that have no ``to_quantized`` method must not be handed
  to ``nn.quantize`` by broad predicates.

These tests pin the current loader-shaped fixes without adding a global runtime
monkeypatch.
"""

from pathlib import Path
import inspect


def test_jangtq_packed_bundles_use_native_turboquant_loader_before_fallback():
    from vmlx_engine.utils import jang_loader

    source = inspect.getsource(jang_loader._load_jang_v2)

    packed_detection = source.index('k.endswith(".tq_packed")')
    native_import = source.index("from jang_tools.load_jangtq import load_jangtq_model")
    native_call = source.index("_load_jangtq(path, skip_params_eval=skip_eval)")
    native_return = source.index("return model, tokenizer", native_call)
    fallback_skeleton = source.index("_load_model_skeleton(", native_return)

    assert packed_detection < native_import < native_call < native_return
    assert native_return < fallback_skeleton


def test_jang_quantize_predicates_skip_modules_without_to_quantized():
    from vmlx_engine.utils import jang_loader

    source = inspect.getsource(jang_loader)

    assert source.count('if not hasattr(m, "to_quantized")') >= 3


def test_pr155_global_jang_model_compat_monkeypatch_is_not_auto_installed():
    runtime_patch_dir = Path("vmlx_engine/runtime_patches")
    init_source = (runtime_patch_dir / "__init__.py").read_text(encoding="utf-8")

    assert "jang_model_compat" not in init_source
    assert not (runtime_patch_dir / "jang_model_compat.py").exists()
