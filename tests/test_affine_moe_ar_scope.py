"""AR-only fusion must never leak into MTP seed, draft or verification."""

import importlib
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.switch_layers import SwiGLU, SwitchGLU

from vmlx_engine.metal import affine_moe_pair_decode as pair


@pytest.fixture
def ar_registration(monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_FUSED_MOE_PAIR", raising=False)
    monkeypatch.setenv("VMLX_QWEN4_FUSED_MOE_PAIR_AR_ONLY", "1")
    monkeypatch.setenv("VMLX_QWEN4_FUSED_MOE_PAIR_MIXED", "1")
    monkeypatch.setitem(pair._FAMILY_CONTRACTS, "qwen4_exp", {
        "hidden": 128, "top_k": 4, "layouts": {(2, 64), (4, 64)},
        "mixed_layouts": {(2, 32), (2, 64), (3, 64), (4, 64), (6, 64), (8, 64)},
        "clamp_limit": None,
    })

    def register(gate_bits=4, gate_gs=64, up_bits=6, up_gs=64, **kwargs):
        switch = SwitchGLU(128, 64, 8, activation=SwiGLU(), bias=False)
        for name, bits, gs in (("gate_proj", gate_bits, gate_gs),
                               ("up_proj", up_bits, up_gs), ("down_proj", 4, 32)):
            projection = getattr(switch, name).to_quantized(group_size=gs, bits=bits)
            projection.scales = projection.scales.astype(mx.float16)
            projection.biases = projection.biases.astype(mx.float16)
            setattr(switch, name, projection)
        switch.eval()
        model = SimpleNamespace(named_modules=lambda: [
            ("model.layers.0.mlp.switch_mlp", switch),
        ])
        installed = pair.install_affine_moe_pair_decode(
            model, family="qwen4_exp", native_mtp_active=True, **kwargs,
        )
        return switch, installed

    return register


@pytest.mark.parametrize("layout", [(2, 64, 2, 64), (2, 32, 3, 64),
                                      (4, 64, 4, 64), (4, 64, 6, 64), (8, 64, 6, 64)])
def test_ar_only_registration_preserves_actual_projection_layouts(ar_registration, layout):
    switch, installed = ar_registration(*layout)
    assert installed == 1
    config = getattr(switch, pair._CONFIG_ATTR)
    assert config.ar_only
    assert (config.bits, config.group_size, config.up_bits_eff,
            config.up_group_size_eff) == layout
    assert pair.affine_moe_pair_status("qwen4_exp")["scope"] == "ar_only"


def test_ar_only_scope_is_required_and_does_not_bypass_shape_or_dtype_guards(
    ar_registration, monkeypatch,
):
    switch, installed = ar_registration()
    assert installed == 1
    calls = []
    monkeypatch.setattr(pair, "_run_pair", lambda *args: calls.append(args) or mx.zeros((1,)))
    x = mx.zeros((1, 1, 128), dtype=mx.float16)
    indices = mx.array([0, 2, 5, 7]).reshape(1, 1, 4)
    assert pair.affine_moe_pair_activation(switch, x, indices) == (None, False)
    with pair.affine_moe_ar_scope():
        assert pair.affine_moe_pair_activation(switch, x, indices)[1]
        for shape in ((1, 2, 128), (2, 1, 128)):
            rows = mx.zeros(shape, dtype=mx.float16)
            routes = mx.zeros((*shape[:-1], 4), dtype=mx.uint32)
            assert pair.affine_moe_pair_activation(switch, rows, routes) == (None, False)
        assert pair.affine_moe_pair_activation(switch, x.astype(mx.float32), indices) == (None, False)
    assert pair.affine_moe_pair_activation(switch, x, indices) == (None, False)
    assert len(calls) == 1


def test_ar_scope_restores_after_nested_exception_and_does_not_leak_to_thread():
    assert not pair._AR_SCOPE.get()
    with pair.affine_moe_ar_scope():
        assert pair._AR_SCOPE.get()
        with pytest.raises(RuntimeError, match="inner"):
            with pair.affine_moe_ar_scope():
                raise RuntimeError("inner")
        assert pair._AR_SCOPE.get()
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(pair._AR_SCOPE.get).result() is False
    assert not pair._AR_SCOPE.get()


@pytest.mark.parametrize("override,installed,ar_only", [("0", 0, None), ("1", 1, False)])
def test_explicit_family_override_precedes_ar_only(
    ar_registration, monkeypatch, override, installed, ar_only,
):
    monkeypatch.setenv("VMLX_QWEN4_FUSED_MOE_PAIR", override)
    switch, actual = ar_registration()
    assert actual == installed
    if installed:
        assert getattr(switch, pair._CONFIG_ATTR).ar_only is ar_only


def test_ar_only_never_registers_mtp_head_even_when_head_override_is_set(
    ar_registration, monkeypatch,
):
    switch, _ = ar_registration()
    monkeypatch.setenv("VMLX_QWEN4_FUSED_MOE_PAIR_MTP_HEAD", "1")
    model = SimpleNamespace(named_modules=lambda: [
        ("language_model.mtp.layers.0.mlp.switch_mlp", switch),
    ])
    # Use a fresh module: a previous registration must not be the oracle.
    delattr(switch, pair._CONFIG_ATTR)
    assert pair.install_affine_moe_pair_decode(
        model, family="qwen4_exp", native_mtp_active=True,
    ) == 0
    assert not hasattr(switch, pair._CONFIG_ATTR)


@pytest.mark.parametrize("fail", [False, True])
def test_mllm_ar_forward_owns_scope_and_releases_it_on_error(monkeypatch, fail):
    from vmlx_engine import mllm_batch_generator as lane

    seen = []

    class Model:
        def __call__(self, inputs, cache):
            seen.append(pair._AR_SCOPE.get())
            if fail:
                raise RuntimeError("AR forward failed")
            return mx.zeros((1, 1, 8))

    gen = lane.MLLMBatchGenerator.__new__(lane.MLLMBatchGenerator)
    gen._model_type = "qwen4_exp"
    gen._decode_trace = False
    gen.language_model = Model()
    gen.active_batch = None
    gen.sampler = None
    monkeypatch.setattr(lane, "_lm_supports_position_ids", lambda _: False)
    monkeypatch.setattr(lane, "_sample_mllm_prefill_logits",
                        lambda *args: (mx.array([1]), None))
    monkeypatch.setattr(lane, "_mimo_v2_token_trace_enabled", lambda: False)
    if fail:
        with pytest.raises(RuntimeError, match="AR forward failed"):
            gen._step(mx.array([1]), [])
    else:
        gen._step(mx.array([1]), [])
    assert seen == [True]
    assert not pair._AR_SCOPE.get()


def test_text_generation_scopes_stock_ar_but_not_mtp_resume_or_verify(monkeypatch):
    from vmlx_engine.patches.mlx_lm_mtp import batch_generator as lane

    generate = importlib.import_module("mlx_lm.generate")
    seen = []

    class Batch:
        def __init__(self):
            self.uids = ["a"]

        def next(self):
            seen.append(("ar", pair._AR_SCOPE.get()))
            return []

        filter = extend = lambda *args, **kwargs: None

    monkeypatch.setattr(generate, "GenerationBatch", Batch)
    monkeypatch.setattr(generate, "BatchGenerator", type("Generator", (), {
        "_next": lambda *a: None, "remove": lambda *a: None,
        "_make_batch": lambda *a: None,
    }))
    monkeypatch.setattr(generate, "PromptProcessingBatch", type("Prompt", (), {
        "prompt": lambda *a: None,
    }))
    monkeypatch.setattr(lane, "_PATCHED", False)
    monkeypatch.setattr(lane, "_is_mtp_eligible", lambda _: True)
    monkeypatch.setattr(lane, "_text_recovery_for_batch", lambda _: SimpleNamespace(uid="a"))
    monkeypatch.setattr(lane, "_text_mtp_maybe_resume",
                        lambda *a: seen.append(("resume", pair._AR_SCOPE.get())))
    monkeypatch.setattr(lane, "_text_mtp_calibration_due", lambda *a: False)
    monkeypatch.setattr(lane, "_mtp_next",
                        lambda *a: seen.append(("verify", pair._AR_SCOPE.get())) or [])
    assert lane.apply()
    batch = Batch.__new__(Batch)
    batch.uids = ["a"]
    batch.next()
    batch._omlx_mtp_state = SimpleNamespace(ar_fallback_pending=False)
    batch.next()
    assert seen == [("resume", False), ("ar", True), ("resume", False), ("verify", False)]
    assert not pair._AR_SCOPE.get()


@pytest.mark.parametrize("fail_step", [False, True])
def test_text_handoff_scope_ends_at_forward_and_releases_on_error(monkeypatch, fail_step):
    from vmlx_engine.patches.mlx_lm_mtp import batch_generator as lane

    seen = []
    batch = SimpleNamespace(tokens=[[1, 2]])
    state = SimpleNamespace(next_main=mx.array([2]))
    monkeypatch.setattr(lane, "_mtp_ar_handoff_ready", lambda *a: (True, "ready"))

    def step():
        seen.append(pair._AR_SCOPE.get())
        if fail_step:
            raise RuntimeError("step")
        batch.tokens[0].append(2)
        return [2], None

    def after_step(_):
        assert not pair._AR_SCOPE.get()
        raise RuntimeError("after step")

    batch._step = step
    monkeypatch.setattr(lane, "_text_recovery_for_batch", after_step)
    with pytest.raises(RuntimeError, match="step"):
        lane._prepare_mtp_ar_handoff(batch, state, None)
    assert seen == [True]
    assert batch.tokens == [[1, 2]]
    assert not pair._AR_SCOPE.get()
