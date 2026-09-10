"""Qwen4-Exp q4/g64 routed-MoE decode kernels.

The stock single-token ``SwitchGLU`` path launches separate selected-expert
gather-QMM operations for gate, up, and down. Qwen3.8-Flash-Next repeats that
path in all 48 layers with ten routed experts, making it the measured decode
bottleneck. These kernels fuse the gate/up packed-weight walks, then apply the
router scores while reducing all ten down projections directly into one hidden
vector. The fused reduction avoids materializing a 10 x 2560 routed output.

Only the exact Qwen4-Exp 2560 -> 640 -> 2560, q4 affine, group-64, top-10
single-token layout is eligible. Prefill, other shapes, JANG codebooks, and
unregistered modules remain on mlx-lm's stock implementation.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from vmlx_engine.metal.affine_moe_pair_decode import affine_moe_pair_activation
from vmlx_engine.metal.qwen4_aligned_moe_prefill import aligned_switchglu

logger = logging.getLogger(__name__)

_D = 2560
_M = 640
_K = 10
_BITS = 4
_GROUP_SIZE = 64
_W_GU = _D // 8
_W_DN = _M // 8
_G_GU = _D // _GROUP_SIZE
_G_DN = _M // _GROUP_SIZE
_SELF_TEST_MAX_REL = 3.0e-2
_OK_ATTR = "_vmlx_qwen4_q4g64_fused_ok"
_EXACT_OK_ATTR = "_vmlx_qwen4_exact_gate_up_ok"
_EXACT_PROJ_ATTR = "_vmlx_qwen4_exact_gate_up_proj"

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

_PAIR_SRC = f"""
    uint tid = thread_position_in_grid.x;
    uint sgid = tid / 32u;
    uint lane = thread_index_in_simdgroup;
    uint k = sgid / {_M}u;
    uint m = sgid % {_M}u;
    if (k >= {_K}u) return;
    uint e = (uint)inds[k];
    size_t rowoff = ((size_t)e * {_M}u + m);
    const device uint32_t* grow = gw + rowoff * {_W_GU}u;
    const device uint32_t* urow = uw + rowoff * {_W_GU}u;
    size_t soff = rowoff * {_G_GU}u;
    float gacc = 0.0f;
    float uacc = 0.0f;
    for (uint w_idx = lane; w_idx < {_W_GU}u; w_idx += 32u) {{
        uint grp = w_idx >> 3;  // q4/g64: eight packed uint32 words per group
        float gsc = (float)gs[soff + grp];
        float gbi = (float)gb[soff + grp];
        float usc = (float)us[soff + grp];
        float ubi = (float)ub[soff + grp];
        uint32_t gwrd = grow[w_idx];
        uint32_t uwrd = urow[w_idx];
        uint xbase = w_idx * 8u;
        float qsg = 0.0f;
        float qsu = 0.0f;
        float xs = 0.0f;
        for (uint j = 0; j < 8u; j++) {{
            float xv = (float)x[xbase + j];
            xs += xv;
            qsg += xv * (float)((gwrd >> (4u * j)) & 15u);
            qsu += xv * (float)((uwrd >> (4u * j)) & 15u);
        }}
        gacc += gsc * qsg + gbi * xs;
        uacc += usc * qsu + ubi * xs;
    }}
    gacc = simd_sum(gacc);
    uacc = simd_sum(uacc);
    if (lane == 0u) {{
        float g = (float)((T)gacc);
        float u = (float)((T)uacc);
        float a = g / (1.0f + metal::exp(-g)) * u;
        act[(size_t)k * {_M}u + m] = (T)a;
    }}
"""

_DOWN_SRC = f"""
    uint tid = thread_position_in_grid.x;
    uint sgid = tid / 32u;
    uint lane = thread_index_in_simdgroup;
    uint d = sgid;
    if (d >= {_D}u) return;
    float weighted = 0.0f;
    for (uint k = 0; k < {_K}u; k++) {{
        uint e = (uint)inds[k];
        size_t rowoff = ((size_t)e * {_D}u + d);
        const device uint32_t* row = dw + rowoff * {_W_DN}u;
        size_t soff = rowoff * {_G_DN}u;
        float acc = 0.0f;
        for (uint w_idx = lane; w_idx < {_W_DN}u; w_idx += 32u) {{
            uint grp = w_idx >> 3;
            float sc = (float)ds[soff + grp];
            float bi = (float)db[soff + grp];
            uint32_t wrd = row[w_idx];
            uint abase = k * {_M}u + w_idx * 8u;
            float qs = 0.0f;
            float xs = 0.0f;
            for (uint j = 0; j < 8u; j++) {{
                float av = (float)act[abase + j];
                xs += av;
                qs += av * (float)((wrd >> (4u * j)) & 15u);
            }}
            acc += sc * qs + bi * xs;
        }}
        acc = simd_sum(acc);
        weighted += (float)scores[k] * acc;
    }}
    if (lane == 0u) {{
        out[d] = (T)weighted;
    }}
"""

_KERNELS: tuple[Any, Any] | None = None
_STATUS: dict[str, Any] = {"installed": 0, "reason": None}


def _enabled() -> bool:
    # The first live q4/g64 prototype is diagnostic-only until it demonstrates
    # an answer-preserving median improvement over MLX's selected-expert QMM.
    value = os.environ.get(
        "VMLX_QWEN4_AFFINE_MOE",
        os.environ.get("VMLINUX_QWEN4_AFFINE_MOE", "0"),
    ).strip().lower()
    return value not in {"0", "false", "off", "no"}


def _exact_enabled() -> bool:
    value = os.environ.get(
        "VMLX_QWEN4_EXACT_GATE_UP",
        os.environ.get("VMLINUX_QWEN4_EXACT_GATE_UP", "0"),
    ).strip().lower()
    return value not in {"0", "false", "off", "no"}


class _ExactGateUpProjection(nn.Module):
    """One selected-expert QMM containing the original up then gate rows."""

    def __init__(self, up: Any, gate: Any):
        super().__init__()
        self.weight = mx.concatenate([up.weight, gate.weight], axis=1)
        self.scales = mx.concatenate([up.scales, gate.scales], axis=1)
        self.biases = mx.concatenate([up.biases, gate.biases], axis=1)
        self.group_size = int(up.group_size)
        self.bits = int(up.bits)
        self.mode = str(up.mode)
        self.split = int(up.output_dims)

    def __call__(self, x: mx.array, indices: mx.array, *, sorted_indices=False) -> mx.array:
        return mx.gather_qmm(
            x,
            self.weight,
            self.scales,
            self.biases,
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )


def _exact_gate_up_switchglu(
    switch: Any,
    x: mx.array,
    indices: mx.array,
    scores: mx.array,
) -> mx.array:
    """Run the stock affine math with one combined up+gate dispatch."""
    from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

    projection = getattr(switch, _EXACT_PROJ_ATTR)
    expanded = mx.expand_dims(x, (-2, -3))
    # Match SwitchGLU's route layout and kernel selection on wide prefill.
    # Skipping this changes gather-QMM arithmetic even with identical weights.
    do_sort = indices.size >= 64
    idx = indices
    if do_sort:
        expanded, idx, inverse = _gather_sort(expanded, indices)
    if switch.training:
        idx = mx.stop_gradient(idx)
    pair = projection(expanded, idx, sorted_indices=do_sort)
    x_up, x_gate = mx.split(pair, [projection.split], axis=-1)
    selected = switch.down_proj(
        switch.activation(x_up, x_gate), idx, sorted_indices=do_sort
    )
    if do_sort:
        selected = _scatter_unsort(selected, inverse, indices.shape)
    selected = selected.squeeze(-2)
    return (selected * scores[..., None]).sum(axis=-2)


def _kernels() -> tuple[Any, Any]:
    global _KERNELS
    if _KERNELS is None:
        pair = mx.fast.metal_kernel(
            name="vmlx_qwen4_q4g64_pair_swiglu",
            input_names=["x", "gw", "gs", "gb", "uw", "us", "ub", "inds"],
            output_names=["act"],
            header=_HEADER,
            source=_PAIR_SRC,
        )
        down = mx.fast.metal_kernel(
            name="vmlx_qwen4_q4g64_weighted_down10",
            input_names=["act", "dw", "ds", "db", "inds", "scores"],
            output_names=["out"],
            header=_HEADER,
            source=_DOWN_SRC,
        )
        _KERNELS = pair, down
    return _KERNELS


def _fused(
    switch: Any,
    x: mx.array,
    indices: mx.array,
    scores: mx.array,
) -> mx.array:
    pair, down = _kernels()
    gate = switch.gate_proj
    up = switch.up_proj
    down_proj = switch.down_proj
    ids = indices.reshape(-1).astype(mx.uint32)
    token = x.reshape(-1)
    act = pair(
        inputs=[
            token,
            gate.weight,
            gate.scales,
            gate.biases,
            up.weight,
            up.scales,
            up.biases,
            ids,
        ],
        template=[("T", x.dtype)],
        grid=(32 * _M * _K, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(_K, _M)],
        output_dtypes=[x.dtype],
    )[0]
    out = down(
        inputs=[
            act,
            down_proj.weight,
            down_proj.scales,
            down_proj.biases,
            ids,
            scores.reshape(-1),
        ],
        template=[("T", x.dtype)],
        grid=(32 * _D, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(_D,)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(*x.shape[:-1], _D)


def _projection_reason(projection: Any, input_dims: int, output_dims: int) -> str | None:
    if projection.__class__.__name__ != "QuantizedSwitchLinear":
        return f"class={projection.__class__.__name__}"
    if int(getattr(projection, "bits", -1)) != _BITS:
        return f"bits={getattr(projection, 'bits', None)}"
    if int(getattr(projection, "group_size", -1)) != _GROUP_SIZE:
        return f"group_size={getattr(projection, 'group_size', None)}"
    if getattr(projection, "mode", "affine") != "affine":
        return f"mode={getattr(projection, 'mode', None)}"
    if int(projection.input_dims) != input_dims or int(projection.output_dims) != output_dims:
        return f"shape={projection.input_dims}->{projection.output_dims}"
    if projection.weight.dtype != mx.uint32:
        return f"weight_dtype={projection.weight.dtype}"
    if projection.biases is None:
        return "missing affine biases"
    return None


def _module_reason(switch: Any) -> str | None:
    checks = (
        ("gate", switch.gate_proj, _D, _M),
        ("up", switch.up_proj, _D, _M),
        ("down", switch.down_proj, _M, _D),
    )
    for name, projection, input_dims, output_dims in checks:
        reason = _projection_reason(projection, input_dims, output_dims)
        if reason is not None:
            return f"{name}:{reason}"
    return None


def qwen4_affine_switchglu(
    switch: Any,
    x: mx.array,
    indices: mx.array,
    scores: mx.array,
) -> tuple[mx.array, bool]:
    """Return the weighted routed output and whether the fused path owned it."""
    aligned = aligned_switchglu(switch, x, indices)
    if aligned is not None:
        return (aligned * scores[..., None]).sum(axis=-2), True
    full_fused_eligible = (
        getattr(switch, _OK_ATTR, False)
        and x.ndim in (2, 3)
        and int(x.size) == int(x.shape[-1])
        and int(x.shape[-1]) == _D
        and int(indices.shape[-1]) == _K
        and tuple(indices.shape[:-1]) == tuple(x.shape[:-1])
        and x.dtype in (mx.float16, mx.bfloat16)
    )
    # When both diagnostic flags are active, the complete q4/g64 kernel owns
    # its exact shape.  Falling into the generic pair hook first would retain
    # stock down/reduction and silently defeat the larger measured win.
    if full_fused_eligible:
        output = _fused(switch, x, indices, scores)
        if not _STATUS.get("observed_calls"):
            _STATUS["observed_calls"] = 1
        return output, True
    activated, pair_fused = affine_moe_pair_activation(switch, x, indices)
    if pair_fused:
        selected = switch.down_proj(activated, indices).squeeze(-2)
        return (selected * scores[..., None]).sum(axis=-2), True
    if getattr(switch, _EXACT_OK_ATTR, False):
        output = _exact_gate_up_switchglu(switch, x, indices, scores)
        if not _STATUS.get("observed_calls"):
            _STATUS["observed_calls"] = 1
        return output, True
    if not getattr(switch, _OK_ATTR, False):
        routed = switch(x, indices)
        return (routed * scores[..., None]).sum(axis=-2), False
    if not full_fused_eligible:
        routed = switch(x, indices)
        return (routed * scores[..., None]).sum(axis=-2), False
    raise AssertionError("unreachable Qwen4 affine MoE dispatch state")


def _self_test(switch: Any) -> str | None:
    dtype = switch.gate_proj.scales.dtype
    if dtype not in (mx.float16, mx.bfloat16):
        return f"activation dtype {dtype} is unsupported"
    x = (mx.random.normal((1, 1, _D)) * 0.125).astype(dtype)
    experts = int(switch.gate_proj.weight.shape[0])
    indices = mx.array(
        [(index * max(1, experts // _K)) % experts for index in range(_K)],
        dtype=mx.uint32,
    ).reshape(1, 1, _K)
    scores = mx.arange(1, _K + 1, dtype=dtype).reshape(1, 1, _K)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    try:
        selected = switch(x, indices)
        reference = (selected * scores[..., None]).sum(axis=-2).astype(mx.float32)
        candidate = _fused(switch, x, indices, scores).astype(mx.float32)
        mx.eval(reference, candidate)
    except Exception as exc:
        return f"self-test execution failed: {type(exc).__name__}: {exc}"
    denom = max(float(mx.abs(reference).max()), 1e-6)
    rel = float(mx.abs(candidate - reference).max()) / denom
    _STATUS["self_test_rel"] = rel
    if rel > _SELF_TEST_MAX_REL:
        return f"self-test rel diff {rel:.3e} > {_SELF_TEST_MAX_REL:.1e}"
    return None


def _exact_self_test(switch: Any, projection: _ExactGateUpProjection) -> str | None:
    dtype = switch.gate_proj.scales.dtype
    if dtype not in (mx.float16, mx.bfloat16):
        return f"activation dtype {dtype} is unsupported"
    x = (mx.random.normal((1, 1, _D)) * 0.125).astype(dtype)
    experts = int(switch.gate_proj.weight.shape[0])
    indices = mx.array(
        [(index * max(1, experts // _K)) % experts for index in range(_K)],
        dtype=mx.uint32,
    ).reshape(1, 1, _K)
    scores = mx.arange(1, _K + 1, dtype=dtype).reshape(1, 1, _K)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    try:
        reference = (switch(x, indices) * scores[..., None]).sum(axis=-2)
        setattr(switch, _EXACT_PROJ_ATTR, projection)
        candidate = _exact_gate_up_switchglu(switch, x, indices, scores)
        mx.eval(reference, candidate)
    except Exception as exc:
        return f"exact gate/up self-test failed: {type(exc).__name__}: {exc}"
    max_abs = float(mx.abs(candidate.astype(mx.float32) - reference.astype(mx.float32)).max())
    denom = max(float(mx.abs(reference.astype(mx.float32)).max()), 1e-6)
    rel = max_abs / denom
    _STATUS.update(exact_self_test_max_abs=max_abs, exact_self_test_rel=rel)
    if max_abs != 0.0:
        return f"exact gate/up self-test was not bit-identical (max_abs={max_abs:.3e})"
    return None


def _install_exact_gate_up(modules: list[Any]) -> int:
    """Replace separate up/gate storage after a bit-exact stock oracle."""
    first_projection = _ExactGateUpProjection(
        modules[0].up_proj, modules[0].gate_proj
    )
    mx.eval(
        first_projection.weight,
        first_projection.scales,
        first_projection.biases,
    )
    failure = _exact_self_test(modules[0], first_projection)
    if failure is not None:
        _STATUS.update(installed=0, reason=failure, mode="exact_gate_up")
        logger.warning("Qwen4 exact gate/up fusion refused: %s", failure)
        return 0

    for index, module in enumerate(modules):
        projection = (
            first_projection
            if index == 0
            else _ExactGateUpProjection(module.up_proj, module.gate_proj)
        )
        if index != 0:
            mx.eval(projection.weight, projection.scales, projection.biases)
        setattr(module, _EXACT_PROJ_ATTR, projection)
        setattr(module, _EXACT_OK_ATTR, True)
        # The combined projection owns exactly the same packed rows. Drop the
        # old references so this dispatch optimization does not duplicate the
        # roughly 40 GB routed up/gate payload.
        module.up_proj = None
        module.gate_proj = None
        mx.clear_cache()

    _STATUS.update(installed=len(modules), reason=None, mode="exact_gate_up")
    logger.info(
        "Qwen4 exact affine gate/up gather-QMM fusion installed for %d modules "
        "(stock oracle max_abs=0)",
        len(modules),
    )
    return len(modules)


def install_qwen4_affine_moe(model: Any) -> int:
    """Validate and register every exact Qwen4 routed SwitchGLU atomically."""
    if not _enabled() and not _exact_enabled():
        _STATUS.update(installed=0, reason="disabled via env")
        return 0
    from mlx_lm.models.switch_layers import SwitchGLU

    modules = [module for _name, module in model.named_modules() if isinstance(module, SwitchGLU)]
    if not modules:
        _STATUS.update(installed=0, reason="no SwitchGLU modules")
        return 0
    rejected = [reason for module in modules if (reason := _module_reason(module))]
    if rejected:
        _STATUS.update(installed=0, reason=f"incompatible modules: {rejected[:3]}")
        return 0
    if _exact_enabled():
        return _install_exact_gate_up(modules)
    failure = _self_test(modules[0])
    if failure is not None:
        _STATUS.update(installed=0, reason=failure)
        logger.warning("Qwen4 affine MoE decode refused: %s", failure)
        return 0
    for module in modules:
        setattr(module, _OK_ATTR, True)
    _STATUS.update(installed=len(modules), reason=None)
    logger.info(
        "Qwen4 affine q4/g64 fused MoE decode registered for %d modules "
        "(self-test rel=%.3e)",
        len(modules),
        float(_STATUS.get("self_test_rel", 0.0)),
    )
    return len(modules)


def qwen4_affine_moe_status() -> dict[str, Any]:
    return dict(_STATUS)


__all__ = [
    "install_qwen4_affine_moe",
    "qwen4_affine_moe_status",
    "qwen4_affine_switchglu",
]
