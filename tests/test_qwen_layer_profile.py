from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path("bench/qwen_layer_profile.py")
    spec = importlib.util.spec_from_file_location("qwen_layer_profile", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_layer_profile_rejects_generation_kwargs():
    module = _load_module()

    try:
        module.assert_no_generation_kwargs({"top_k": 20})
    except ValueError as exc:
        assert "generation/sampler" in str(exc)
        assert "top_k" in str(exc)
    else:
        raise AssertionError("expected generation kwargs to be rejected")


def test_layer_profile_summary_groups_layer_types():
    module = _load_module()

    records = [
        {"route": "text", "kind": "linear", "elapsed_s": 0.10},
        {"route": "text", "kind": "attention", "elapsed_s": 0.40},
        {"route": "text", "kind": "linear", "elapsed_s": 0.20},
    ]

    summary = module.summarize_records(records)

    assert summary["total_s"] == 0.70
    assert summary["by_kind"]["linear"]["count"] == 2
    assert summary["by_kind"]["linear"]["total_s"] == 0.30
    assert summary["by_kind"]["attention"]["count"] == 1
    assert summary["by_kind"]["attention"]["total_s"] == 0.40


def test_layer_profile_layer_kind_uses_is_linear_flag():
    module = _load_module()

    class LinearLayer:
        is_linear = True

    class AttentionLayer:
        is_linear = False

    assert module.layer_kind(LinearLayer()) == "linear"
    assert module.layer_kind(AttentionLayer()) == "attention"
