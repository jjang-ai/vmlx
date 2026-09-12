# SPDX-License-Identifier: Apache-2.0
"""Qualify view-based HC splitting separately from projection/rounding changes."""

import json
import os
import statistics
import time
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from vmlx_engine.models.qwen4_exp.language import (
    GatedResidual,
    _hyper_column_major_rows,
    compile_hyper_connections,
    fuse_hyper_connection_projections,
)


@pytest.fixture(autouse=True)
def _isolate_hc_switch(monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_HC_VIEW_SPLIT", raising=False)


def _gather_layout(value):
    """Match take(axis=-1)'s column-major rows without integer gathering."""
    return _hyper_column_major_rows(value)


def _view_forward(module, hyper_input):
    """Prototype: only the two mx.take operations differ from _forward."""
    normed = module.hc_norm(hyper_input)
    combined = module.input_inject_weight(normed)
    mix = combined[..., :module.hc_lowrank]
    block_injection = combined[..., module.hc_lowrank:module.hc_lowrank + module.hc_count]
    mix = nn.silu(mix / module.hc_count)
    # MLX take(axis=-1) returns column-major rows. A normal contiguous copy
    # would change the subsequent GEMM traversal and its FP16 rounding.
    mix = _gather_layout(mix)
    mix = mx.sigmoid(module.input_mix_weight_up(mix))
    mix = mix.astype(normed.dtype)
    mix = mix.reshape(*mix.shape[:-1], module.hc_count, module.hidden_size)
    mixed = (
        mix * normed.reshape(*normed.shape[:-1], module.hc_count, module.hidden_size)
    ).mean(-2).astype(hyper_input.dtype)
    inject_w = (2.0 * mx.sigmoid(block_injection / module.hc_count)).astype(hyper_input.dtype)
    return mixed, hyper_input, inject_w


def _module(dtype, quant):
    # Actual Flash-Next geometry; synthetic, reproducible component weights.
    args = SimpleNamespace(hidden_size=2560, hc_count=4, hc_lowrank=320, rms_norm_eps=1e-6)
    mx.random.seed(7191)
    module = GatedResidual(args)
    for linear in (module.input_mix_weight_down, module.input_mix_weight_up,
                   module.block_inject_weight):
        linear.weight = linear.weight.astype(dtype)
    module.hc_norm.weight = mx.random.uniform(0.5, 1.5, (10240,)).astype(dtype)
    if quant is not None:
        bits, group = quant
        module.input_mix_weight_down = module.input_mix_weight_down.to_quantized(
            bits=bits, group_size=group)
        module.block_inject_weight = module.block_inject_weight.to_quantized(
            bits=bits, group_size=group)
        module.input_mix_weight_up = module.input_mix_weight_up.to_quantized(
            bits=bits, group_size=32 if group == 128 else group)
    module.eval()
    mx.eval(module.parameters())
    assert fuse_hyper_connection_projections(module) == 1
    return module


def _exact(left, right):
    assert len(left) == len(right)
    for expected, actual in zip(left, right):
        assert expected.shape == actual.shape and expected.dtype == actual.dtype
        # Viewing BF16 as raw bits avoids silently comparing promoted values.
        cast = lambda value: np.asarray(value.view(mx.uint16)) if value.dtype == mx.bfloat16 else np.asarray(value)
        np.testing.assert_array_equal(cast(actual), cast(expected))


def _split_trace(module, x, use_view):
    """Diagnostic eager boundaries; evaluating them changes the lazy graph."""
    stages = {}
    layouts = {}
    stride_kernel = mx.fast.metal_kernel(
        name="hc_probe_strides", input_names=["inp"], output_names=["out"],
        source="out[thread_position_in_grid.x] = int(inp_strides[thread_position_in_grid.x]);",
        ensure_row_contiguous=False,
    )

    def retain(name, value):
        mx.eval(value)
        stages[name] = np.asarray(value.astype(mx.float32)).copy()
        stride = stride_kernel(inputs=[value], grid=(value.ndim, 1, 1),
                               threadgroup=(value.ndim, 1, 1),
                               output_shapes=[(value.ndim,)], output_dtypes=[mx.int32])[0]
        layouts[name] = stride.tolist()
        return value

    normed = retain("normed", module.hc_norm(x))
    combined = retain("combined", module.input_inject_weight(normed))
    if use_view:
        mix = combined[..., :module.hc_lowrank]
        injection = combined[..., module.hc_lowrank:]
    else:
        mix = mx.take(combined, mx.arange(module.hc_lowrank), axis=-1)
        injection = mx.take(combined, mx.arange(module.hc_lowrank, module.hc_lowrank + module.hc_count), axis=-1)
    mix = retain("split", mix)
    retain("injection", injection)
    mix = retain("divided", mix / module.hc_count)
    mix = retain("silu", nn.silu(mix))
    if use_view:
        mix = retain("gather_layout", _gather_layout(mix))
    mix = retain("up", module.input_mix_weight_up(mix))
    mix = retain("sigmoid", mx.sigmoid(mix))
    mix = retain("cast", mix.astype(normed.dtype))
    mix = mix.reshape(*mix.shape[:-1], module.hc_count, module.hidden_size)
    product = retain("product", mix * normed.reshape(*normed.shape[:-1], module.hc_count, module.hidden_size))
    retain("mixed", product.mean(-2).astype(x.dtype))
    return stages, layouts


def _diagnose(module, x, rows, cycle):
    repeated = []
    for function in (module._forward, module._forward,
                     lambda value: _view_forward(module, value),
                     lambda value: _view_forward(module, value)):
        result = function(x)
        mx.eval(*result)
        repeated.append(tuple(np.asarray(value.astype(mx.float32)).copy() for value in result))
    baseline, baseline_layout = _split_trace(module, x, False)
    candidate, candidate_layout = _split_trace(module, x, True)
    print("HC_SPLIT_DIVERGENCE " + json.dumps({
        "rows": rows, "cycle": cycle,
        "baseline_repeat_equal": all(np.array_equal(a, b) for a, b in zip(repeated[0], repeated[1])),
        "candidate_repeat_equal": all(np.array_equal(a, b) for a, b in zip(repeated[2], repeated[3])),
        "cross_equal": all(np.array_equal(a, b) for a, b in zip(repeated[0], repeated[2])),
        "baseline_layout": baseline_layout, "candidate_layout": candidate_layout,
        "staged_deltas": {key: {"mismatches": int(np.count_nonzero(baseline[key] != candidate[key])),
                                "max_abs": float(np.max(np.abs(baseline[key] - candidate[key])))}
                          for key in baseline},
    }))


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("rows", [1, 2, 4, 16, 65])
@pytest.mark.parametrize("quant", [None, (2, 32), (4, 64), (6, 32), (8, 128)])
@pytest.mark.parametrize("batch", [1, 2])
def test_real_shape_hc_view_split_exact(dtype, rows, quant, batch):
    module = _module(dtype, quant)
    inputs = [mx.random.normal((batch, rows, 10240)).astype(dtype) for _ in range(3)]
    with mx.stream(mx.new_stream(mx.gpu)):
        compiled_base = mx.compile(module._forward)
        compiled_view = mx.compile(lambda value: _view_forward(module, value))
        # Compare like-for-like compilation contracts, not eager vs an already
        # approximate compiled reference. Each arm is independently evaluated.
        for x in inputs:
            eager = module._forward(x)
            mx.eval(*eager)
            view = _view_forward(module, x)
            mx.eval(*view)
            base_jit = compiled_base(x)
            mx.eval(*base_jit)
            view_jit = compiled_view(x)
            mx.eval(*view_jit)
            _exact(eager, view)
            _exact(base_jit, view_jit)


@pytest.mark.skipif(os.environ.get("VMLX_HC_SPLIT_TIMING") != "1", reason="explicit M5 component timing only")
def test_hc_view_complete_operation_cost():
    """Cost includes downstream matmul, sigmoid, reduction and completion."""
    module = _module(mx.float16, None)
    results = []
    for rows in (1, 2, 4, 16, 65):
        inputs = [mx.random.normal((1, rows, 10240)).astype(mx.float16) for _ in range(14)]
        mx.eval(*inputs)
        functions = {
            "eager_take": module._forward,
            "eager_view": lambda value: _view_forward(module, value),
            "compiled_take": mx.compile(module._forward),
            "compiled_view": mx.compile(lambda value: _view_forward(module, value)),
        }
        times = {key: [] for key in functions}
        for cycle, x in enumerate(inputs):
            names = list(functions)
            if cycle % 2:
                names.reverse()
            outputs = {}
            for name in names:
                started = time.perf_counter()
                outputs[name] = functions[name](x)
                mx.eval(*outputs[name])
                times[name].append((time.perf_counter() - started) * 1000)
            try:
                _exact(outputs["eager_take"], outputs["eager_view"])
                _exact(outputs["compiled_take"], outputs["compiled_view"])
            except AssertionError:
                _diagnose(module, x, rows, cycle)
                raise
        results.append({"rows": rows, "samples_ms": times,
                        "steady_median_ms": {key: statistics.median(values[2:])
                                             for key, values in times.items()}})
        print("HC_SPLIT_ROW_COST " + json.dumps(results[-1]))
    print("HC_SPLIT_COST " + json.dumps(results))


@pytest.mark.parametrize("rows", [1, 2, 4, 5, 65])
@pytest.mark.parametrize("quant", [None, (6, 32)])
def test_production_hc_view_split_gate(monkeypatch, rows, quant, caplog):
    monkeypatch.delenv("VMLX_QWEN4_HC_VIEW_SPLIT", raising=False)
    reference = _module(mx.float16, quant)
    assert reference._hc_view_split is False
    monkeypatch.setenv("VMLX_QWEN4_HC_VIEW_SPLIT", "1")
    candidate = _module(mx.float16, quant)
    assert candidate._hc_view_split is True
    # Actual source path, not merely the wider prototype above.
    caplog.set_level("INFO", logger="vmlx_engine.models.qwen4_exp.language")
    assert compile_hyper_connections(reference) == 1
    assert compile_hyper_connections(candidate) == 1
    assert "HC view split installed: modules=1 max_rows=4" in caplog.text
    inputs = [mx.random.normal((2, rows, 10240)).astype(mx.float16) for _ in range(14)]
    for x in inputs:
        baseline = reference(x)
        mx.eval(*baseline)
        actual = candidate(x)
        mx.eval(*actual)
        _exact(baseline, actual)
    # Eager instrumentation proves disabled/unqualified widths retain takes;
    # the compiled function can legitimately elide Python on later calls.
    takes = []
    original = mx.take
    def observe_take(*args, **kwargs):
        takes.append(kwargs.get("axis"))
        return original(*args, **kwargs)
    monkeypatch.setattr(mx, "take", observe_take)
    mx.eval(*candidate._forward(inputs[0]))
    assert takes == ([] if rows <= 4 else [-1, -1])


@pytest.mark.parametrize("use_combine", [False, True])
def test_hc_view_split_unfused_fallback(monkeypatch, use_combine):
    args = SimpleNamespace(hidden_size=2560, hc_count=4, hc_lowrank=320, rms_norm_eps=1e-6)
    monkeypatch.delenv("VMLX_QWEN4_HC_VIEW_SPLIT", raising=False)
    mx.random.seed(919)
    reference = GatedResidual(args, use_combine=use_combine)
    monkeypatch.setenv("VMLX_QWEN4_HC_VIEW_SPLIT", "1")
    mx.random.seed(919)
    candidate = GatedResidual(args, use_combine=use_combine)
    x = mx.random.normal((1, 2, 10240))
    baseline = reference(x)
    actual = candidate(x)
    if not use_combine:
        baseline, actual = (baseline,), (actual,)
    mx.eval(*baseline, *actual)
    _exact(baseline, actual)


def test_hc_view_split_preserves_gather_gemm_layout_regression(monkeypatch):
    reference = _module(mx.float16, None)
    monkeypatch.setenv("VMLX_QWEN4_HC_VIEW_SPLIT", "1")
    candidate = _module(mx.float16, None)
    # The first tiny fixture missed this input. Keep the independent-eval
    # multi-row rounding case in ordinary tests, not only opt-in timing.
    for _ in range(14):
        mx.random.normal((1, 1, 10240)).astype(mx.float16)
    inputs = [mx.random.normal((1, 2, 10240)).astype(mx.float16) for _ in range(14)]
    for x in inputs:
        baseline = reference._forward(x)
        mx.eval(*baseline)
        actual = candidate._forward(x)
        mx.eval(*actual)
        _exact(baseline, actual)
