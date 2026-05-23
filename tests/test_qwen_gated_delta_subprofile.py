from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path("bench/qwen_gated_delta_subprofile.py")
    spec = importlib.util.spec_from_file_location("qwen_gated_delta_subprofile", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_subprofile_rejects_generation_kwargs():
    module = _load_module()

    try:
        module.assert_no_generation_kwargs({"repetition_penalty": 1.05})
    except ValueError as exc:
        assert "generation/sampler" in str(exc)
        assert "repetition_penalty" in str(exc)
    else:
        raise AssertionError("expected generation kwargs to be rejected")


def test_subprofile_summary_groups_operations():
    module = _load_module()

    records = [
        {"route": "text", "op": "norm", "elapsed_s": 0.25},
        {"route": "text", "op": "in_proj_qkv", "elapsed_s": 0.50},
        {"route": "text", "op": "norm", "elapsed_s": 0.75},
    ]

    summary = module.summarize_records(records)

    assert summary["norm"]["count"] == 2
    assert summary["norm"]["total_s"] == 1.0
    assert summary["norm"]["avg_s"] == 0.5
    assert summary["in_proj_qkv"]["count"] == 1
    assert summary["in_proj_qkv"]["total_s"] == 0.5


def test_subprofile_installs_only_linear_layer_operations():
    module = _load_module()

    class Op:
        def __call__(self, value):
            return value

    class LinearAttn:
        in_proj_qkv = Op()
        norm = Op()

    class LinearLayer:
        is_linear = True
        linear_attn = LinearAttn()

    class AttentionLayer:
        is_linear = False
        linear_attn = LinearAttn()

    class Target:
        layers = [LinearLayer(), AttentionLayer()]

    records, restorers = module.install_timers("route", Target())

    try:
        assert len(restorers) == 2
        assert isinstance(Target.layers[0].linear_attn.in_proj_qkv, module.TimedModule)
        assert isinstance(Target.layers[0].linear_attn.norm, module.TimedModule)
        assert not isinstance(Target.layers[1].linear_attn.in_proj_qkv, module.TimedModule)
    finally:
        module.restore_timers(restorers)

    assert records == []
    assert isinstance(Target.layers[0].linear_attn.in_proj_qkv, Op)
