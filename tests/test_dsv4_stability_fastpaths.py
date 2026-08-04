# SPDX-License-Identifier: Apache-2.0
"""Fail-closed contracts for the optional DSV4 stability fast paths."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _projection(
    *, bits: int = 2, group_size: int = 64, input_dims: int = 128, output_dims: int = 64
):
    return SimpleNamespace(
        bits=bits,
        group_size=group_size,
        mode="affine",
        scales=object(),
        biases=object(),
        weight=object(),
        input_dims=input_dims,
        output_dims=output_dims,
    )


def _switch(*, gate_bits: int = 3):
    return SimpleNamespace(
        gate_proj=_projection(bits=gate_bits),
        up_proj=_projection(bits=2),
        down_proj=_projection(bits=2, group_size=32, input_dims=64, output_dims=128),
        activation=SimpleNamespace(swiglu_limit=10.0),
    )


def test_compiled_moe_installer_rejects_malformed_layout_without_raising(monkeypatch):
    import vmlx_engine.models.dsv4_compiled_moe as mod

    class Owner:
        def __init__(self, switch):
            self.switch_mlp = switch

        def _weighted_routed_experts(self, *_args):
            return "stock"

    class Model:
        def named_modules(self):
            return [("layer.0", Owner(_switch(gate_bits="broken")))]

    monkeypatch.setattr(mod, "mx", object())
    monkeypatch.setenv("VMLX_DSV4_COMPILED_MOE", "1")
    mod._SUPPORTED_MOES.clear()
    mod._DISABLED_MOES.clear()

    assert mod.install_dsv4_compiled_moe(Model()) == 0
    assert not mod._SUPPORTED_MOES


def test_compiled_moe_installation_replaces_stale_module_selection(monkeypatch):
    import vmlx_engine.models.dsv4_compiled_moe as mod

    class Owner:
        def __init__(self, switch):
            self.switch_mlp = switch

        def _weighted_routed_experts(self, *_args):
            return "stock"

    class Model:
        def __init__(self, owner):
            self.owner = owner

        def named_modules(self):
            return [("layer.0", self.owner)]

    first = Owner(_switch())
    second = Owner(_switch())
    monkeypatch.setattr(mod, "mx", object())
    monkeypatch.setenv("VMLX_DSV4_COMPILED_MOE", "1")
    mod._PATCHED_CLASS = None
    mod._SUPPORTED_MOES.clear()
    mod._DISABLED_MOES.clear()

    assert mod.install_dsv4_compiled_moe(Model(first)) == 1
    assert mod.install_dsv4_compiled_moe(Model(second)) == 1
    assert id(second) in mod._SUPPORTED_MOES
    assert id(first) not in mod._SUPPORTED_MOES
    mod._DISABLED_MOES.add(id(second))
    assert mod.install_dsv4_compiled_moe(Model(second)) == 1
    assert id(second) in mod._DISABLED_MOES

    monkeypatch.setenv("VMLX_DSV4_COMPILED_MOE", "0")
    assert first._weighted_routed_experts(None, None, None) == "stock"
    assert second._weighted_routed_experts(None, None, None) == "stock"


def test_indexer_bypass_falls_back_for_short_or_malformed_state(monkeypatch):
    import vmlx_engine.models.dsv4_indexer_skip as mod

    class FakeMx:
        @staticmethod
        def zeros(shape, dtype=None):
            return ("zeros", tuple(shape), dtype)

    class Indexer:
        def __init__(self):
            self.head_dim = 4
            self.index_topk = 4
            self.compressor = object()

        def update_pool(self, _x, _rope, _cache, start_pos):
            return ("stock", start_pos)

    class Model:
        def __init__(self, indexer):
            self.indexer = indexer

        def named_modules(self):
            return [("layer.0.indexer", self.indexer)]

    class Cache:
        _vmlx_dsv4_indexer_skip_decode = True

        def __init__(self, malformed=False):
            self.states = {
                "compressor_state": {
                    "pooled": SimpleNamespace(ndim=3, shape=(1, 3, 4))
                },
                "indexer_state": {},
            }
            if malformed:
                self.states["compressor_state"] = {"pooled": object()}

        def _branch_state(self, name):
            return self.states[name]

    monkeypatch.setattr(mod, "mx", FakeMx)
    monkeypatch.setattr(mod, "_PATCHED_CLASS", None)
    monkeypatch.setenv("VMLX_DSV4_SKIP_UNUSED_INDEXER", "1")
    indexer = Indexer()
    assert mod.install_dsv4_indexer_skip(Model(indexer)) == 1

    short_x = SimpleNamespace(ndim=1, shape=(4,), dtype="float16")
    assert indexer.update_pool(short_x, None, Cache(), 1) == ("stock", 1)

    decode_x = SimpleNamespace(ndim=2, shape=(1, 1), dtype="float16")
    bypassed = indexer.update_pool(decode_x, None, Cache(), 2)
    assert bypassed == ("zeros", (1, 3, 4), "float16")

    fallback = indexer.update_pool(decode_x, None, Cache(malformed=True), 3)
    assert fallback == ("stock", 3)


def test_lm_head_fastpath_falls_back_once_after_runtime_failure(monkeypatch):
    import vmlx_engine.models.dsv4_lm_head_cache as mod

    class FakeMx:
        @staticmethod
        def quantized_matmul(*_args, **_kwargs):
            raise RuntimeError("synthetic quantized matmul failure")

    class Head:
        weight = SimpleNamespace(ndim=2, shape=(16, 4))
        scales = object()
        bits = 8
        group_size = 64
        mode = "affine"

    class Model:
        def __init__(self):
            self.lm_head = Head()
            self.model = lambda input_ids, cache=None, mask=None: "hidden"

        def __call__(self, input_ids, cache=None, mask=None):
            return ("stock", input_ids, cache, mask)

    monkeypatch.setattr(mod, "mx", FakeMx)
    monkeypatch.setenv("VMLX_DSV4_LM_HEAD_MODE", "quantized")
    model = Model()
    assert mod.install_dsv4_lm_head_cache(model)

    first = model("ids", cache="cache", mask="mask")
    second = model("ids2")
    assert first[0] == second[0] == "stock"
    assert model._vmlx_dsv4_lm_head_fastpath_disabled is True


def test_rope_table_cache_is_bounded(monkeypatch):
    import mlx.core as mx
    from jang_tools.dsv4.mlx_model import DeepseekV4RoPE

    import vmlx_engine.models.dsv4_rope_cache as mod

    original_call = DeepseekV4RoPE.__call__
    original_cache_flag = getattr(DeepseekV4RoPE, "_vmlx_dsv4_rope_cache", None)
    original_original_call = getattr(
        DeepseekV4RoPE, "_vmlx_dsv4_rope_original_call", None
    )
    mod._TABLES.clear()
    mod._PATCHED_CLASS = None
    monkeypatch.setenv("VMLX_DSV4_ROPE_CACHE_MAX_ENTRIES", "1")

    rope = DeepseekV4RoPE(4, 10_000)

    class Model:
        def named_modules(self):
            return [("rope", rope)]

    try:
        assert mod.install_dsv4_rope_cache(Model())
        x = mx.ones((1, 2, 4), dtype=mx.float16)
        mx.eval(rope(x, offset=0), rope(x, offset=2))
        assert len(mod._TABLES) <= 1
    finally:
        DeepseekV4RoPE.__call__ = original_call
        if original_cache_flag is None:
            with __import__("contextlib").suppress(AttributeError):
                delattr(DeepseekV4RoPE, "_vmlx_dsv4_rope_cache")
        else:
            DeepseekV4RoPE._vmlx_dsv4_rope_cache = original_cache_flag
        if original_original_call is None:
            with __import__("contextlib").suppress(AttributeError):
                delattr(DeepseekV4RoPE, "_vmlx_dsv4_rope_original_call")
        else:
            DeepseekV4RoPE._vmlx_dsv4_rope_original_call = original_original_call
        mod._TABLES.clear()
        mod._PATCHED_CLASS = None


def test_target_harness_requires_reference_match_when_supplied():
    path = Path(__file__).parents[1] / "bench" / "dsv4_target_harness.py"
    spec = importlib.util.spec_from_file_location("dsv4_target_harness", path)
    assert spec is not None and spec.loader is not None
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)

    rows = [
        {
            "new_tokens": 30,
            "token_ids_sha256": "actual",
            "prefill_tok_s": 250.0,
            "decode_tok_s": 27.0,
        }
    ]
    assert harness._summarize(rows, 200.0, "actual")["passed"] is True
    mismatch = harness._summarize(rows, 200.0, "reference")
    assert mismatch["passed"] is False
    assert mismatch["reference_token_hash_matches"] is False
    assert Path("dsv4-target-report.json") == harness.DEFAULT_OUTPUT
