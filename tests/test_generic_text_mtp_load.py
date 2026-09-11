"""Generic affine/MLX loads must activate MTP before constructing the model."""
import json

import pytest


@pytest.fixture
def loader(monkeypatch, tmp_path):
    import mlx_lm
    from vmlx_engine import mlx_memory, model_bundle_integrity, native_mtp
    from vmlx_engine.models import spark2_5
    from vmlx_engine.utils import jang_loader, nanbeige_runtime, tokenizer

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    monkeypatch.setattr(model_bundle_integrity, "prepare_model_bundle_for_load",
                        lambda *a, **kw: (str(tmp_path), {}))
    monkeypatch.setattr(tokenizer, "_register_mimo_v2_runtime_for_mlx_lm", lambda: False)
    monkeypatch.setattr(tokenizer, "_needs_tokenizer_fallback", lambda p: False)
    monkeypatch.setattr(tokenizer, "_inject_chat_template_if_missing", lambda *a: None)
    monkeypatch.setattr(jang_loader, "is_jang_model", lambda p: False)
    monkeypatch.setattr(nanbeige_runtime, "ensure_nanbeige_runtime_registered", lambda p: None)
    monkeypatch.setattr(nanbeige_runtime, "validate_nanbeige_loop_cache_contract", lambda *a: None)
    monkeypatch.setattr(spark2_5, "ensure_spark2_5_runtime_registered", lambda p: None)
    monkeypatch.setattr(mlx_memory, "maybe_harmonize_quant_metadata_dtypes", lambda *a, **kw: None)
    events = []
    status = {"runtime_active": True, "status": "native_runtime_ready"}

    class HeadModel:
        mtp = object()
        def mtp_forward(self): pass
        def make_mtp_cache(self): pass

    models = [HeadModel()]
    def activate(*a, **kw):
        events.append("activate")
        return status.copy()
    def load(*a, **kw):
        events.append("load")
        return models[0], object()
    monkeypatch.setattr(native_mtp, "maybe_apply_native_mtp", activate)
    monkeypatch.setattr(native_mtp, "deactivate_native_mtp", lambda: events.append("deactivate"))
    monkeypatch.setattr(mlx_lm, "load", load)
    monkeypatch.setattr(tokenizer, "_load_with_tokenizer_fallback", load)
    return tokenizer, str(tmp_path), events, status, models


def test_activation_before_generic_constructor(loader):
    module, path, events, _, _ = loader
    module.load_model_with_fallback(path, skip_turboquant=True)
    assert events == ["activate", "load"]


def test_missing_enabled_head_is_not_silent_ar(loader):
    module, path, events, _, models = loader
    models[0] = object()
    with pytest.raises(RuntimeError, match="no attached draft head"):
        module.load_model_with_fallback(path, skip_turboquant=True)
    assert events == ["activate", "load", "deactivate"]


def test_explicit_off_does_not_require_head(loader):
    module, path, events, status, models = loader
    status.update(runtime_active=False, status="runtime_disabled")
    models[0] = object()
    module.load_model_with_fallback(path, skip_turboquant=True)
    assert events == ["activate", "load"]


def test_patch_failure_aborts_before_construction(loader):
    module, path, events, status, _ = loader
    status.update(runtime_active=False, status="runtime_patch_failed")
    with pytest.raises(RuntimeError, match="activation failed"):
        module.load_model_with_fallback(path, skip_turboquant=True)
    assert events == ["activate"]


def test_tokenizer_fallback_also_activates_before_construction(loader, monkeypatch):
    module, path, events, _, _ = loader
    monkeypatch.setattr(module, "_needs_tokenizer_fallback", lambda p: True)
    module.load_model_with_fallback(path, skip_turboquant=True)
    assert events == ["activate", "load"]
