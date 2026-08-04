# SPDX-License-Identifier: Apache-2.0
"""Guarded affine-MoE decode kernels for DeepSeek-V4.

The fused dequantize-plus-matvec kernel design and initial implementation were
contributed by Andrew Hornsby (@Hornsan1) in jjang-ai/vmlx PR #248.  This
version keeps that work behind the exact DSV4 affine layout it was measured on
and preserves the stock ``SwitchGLU`` path for every other model, shape, dtype,
or quantization layout.
"""

from __future__ import annotations

import logging
import os
import weakref
from dataclasses import dataclass, field
from typing import Any

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - non-Apple package inspection
    mx = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

N_CHUNKS = 2
THREADGROUP_Y = 128


_KERNEL_B2_G64 = r"""
    uint expert_local = thread_position_in_grid.x;
    uint out_d = thread_position_in_grid.y;
    uint chunk_id = thread_position_in_grid.z;
    if (expert_local >= (uint)k || out_d >= (uint)out_dim) return;
    uint expert_id = (uint)expert_ids[expert_local];
    uint packed_per_row = (uint)n_groups * 4u;
    uint w_row_base = expert_id * (uint)out_dim * packed_per_row + out_d * packed_per_row;
    uint s_row_base = expert_id * (uint)out_dim * (uint)n_groups + out_d * (uint)n_groups;
    uint groups_per_chunk = (uint)n_groups / (uint)n_chunks;
    uint g_start = chunk_id * groups_per_chunk;
    uint g_end = g_start + groups_per_chunk;
    float sum = 0.0f;
    for (uint g = g_start; g < g_end; g++) {
        float scale = (float)scales[s_row_base + g];
        float bias = (float)biases[s_row_base + g];
        uint x_base = g * 64u;
        uint w_base = w_row_base + g * 4u;
        for (uint w = 0; w < 4u; w++) {
            uint packed = weights[w_base + w];
            uint x_off = x_base + w * 16u;
            for (uint i = 0; i < 16u; i++) {
                uint code = (packed >> (i * 2u)) & 0x3u;
                sum += (float)x[x_off + i] * ((float)code * scale + bias);
            }
        }
    }
    partial[expert_local * (uint)out_dim * (uint)n_chunks + out_d * (uint)n_chunks + chunk_id] = sum;
"""


_KERNEL_B3_G64 = r"""
    uint expert_local = thread_position_in_grid.x;
    uint out_d = thread_position_in_grid.y;
    uint chunk_id = thread_position_in_grid.z;
    if (expert_local >= (uint)k || out_d >= (uint)out_dim) return;
    uint expert_id = (uint)expert_ids[expert_local];
    uint packed_per_row = (uint)n_groups * 6u;
    uint w_row_base = expert_id * (uint)out_dim * packed_per_row + out_d * packed_per_row;
    uint s_row_base = expert_id * (uint)out_dim * (uint)n_groups + out_d * (uint)n_groups;
    uint groups_per_chunk = (uint)n_groups / (uint)n_chunks;
    uint g_start = chunk_id * groups_per_chunk;
    uint g_end = g_start + groups_per_chunk;
    float sum = 0.0f;
    for (uint g = g_start; g < g_end; g++) {
        float scale = (float)scales[s_row_base + g];
        float bias = (float)biases[s_row_base + g];
        uint x_base = g * 64u;
        uint w_base = w_row_base + g * 6u;
        for (uint i = 0; i < 64u; i++) {
            uint bit_start = i * 3u;
            uint uint_idx = bit_start / 32u;
            uint bit_off = bit_start % 32u;
            uint packed = weights[w_base + uint_idx];
            uint code;
            if (bit_off <= 29u) {
                code = (packed >> bit_off) & 0x7u;
            } else {
                uint lo = packed >> bit_off;
                uint hi = weights[w_base + uint_idx + 1u];
                code = (lo | (hi << (32u - bit_off))) & 0x7u;
            }
            sum += (float)x[x_base + i] * ((float)code * scale + bias);
        }
    }
    partial[expert_local * (uint)out_dim * (uint)n_chunks + out_d * (uint)n_chunks + chunk_id] = sum;
"""


_KERNEL_B2_G32 = r"""
    uint expert_local = thread_position_in_grid.x;
    uint out_d = thread_position_in_grid.y;
    uint chunk_id = thread_position_in_grid.z;
    if (expert_local >= (uint)k || out_d >= (uint)out_dim) return;
    uint expert_id = (uint)expert_ids[expert_local];
    uint packed_per_row = (uint)n_groups * 2u;
    uint w_row_base = expert_id * (uint)out_dim * packed_per_row + out_d * packed_per_row;
    uint s_row_base = expert_id * (uint)out_dim * (uint)n_groups + out_d * (uint)n_groups;
    uint x_row_base = expert_local * (uint)hidden;
    uint groups_per_chunk = (uint)n_groups / (uint)n_chunks;
    uint g_start = chunk_id * groups_per_chunk;
    uint g_end = g_start + groups_per_chunk;
    float sum = 0.0f;
    for (uint g = g_start; g < g_end; g++) {
        float scale = (float)scales[s_row_base + g];
        float bias = (float)biases[s_row_base + g];
        uint x_base = x_row_base + g * 32u;
        uint w_base = w_row_base + g * 2u;
        for (uint w = 0; w < 2u; w++) {
            uint packed = weights[w_base + w];
            uint x_off = x_base + w * 16u;
            for (uint i = 0; i < 16u; i++) {
                uint code = (packed >> (i * 2u)) & 0x3u;
                sum += (float)x[x_off + i] * ((float)code * scale + bias);
            }
        }
    }
    partial[expert_local * (uint)out_dim * (uint)n_chunks + out_d * (uint)n_chunks + chunk_id] = sum;
"""


class AffineMoEDecodeManager:
    """Lazily compile and reuse Hornsan1's three exact affine kernels."""

    def __init__(self, n_chunks: int = N_CHUNKS, threadgroup_y: int = THREADGROUP_Y):
        self.n_chunks = int(n_chunks)
        self.threadgroup_y = int(threadgroup_y)
        self._kernels: dict[str, Any] = {}

    def kernel_for(self, bits: int, group_size: int):
        if mx is None:
            raise RuntimeError("MLX is unavailable")
        key = f"b{int(bits)}_g{int(group_size)}"
        sources = {
            "b2_g64": _KERNEL_B2_G64,
            "b3_g64": _KERNEL_B3_G64,
            "b2_g32": _KERNEL_B2_G32,
        }
        if key not in sources:
            raise ValueError(f"unsupported DSV4 affine projection {key}")
        if key not in self._kernels:
            self._kernels[key] = mx.fast.metal_kernel(
                name=f"dsv4_affine_moe_decode_{key}",
                input_names=[
                    "x", "weights", "scales", "biases", "expert_ids",
                    "hidden", "out_dim", "n_groups", "k", "n_chunks",
                ],
                output_names=["partial"],
                source=sources[key],
                ensure_row_contiguous=False,
            )
        return self._kernels[key]

    def run_projection(
        self,
        x: mx.array,
        projection: _Projection,
        expert_ids: mx.array,
        k: int,
    ) -> mx.array:
        kernel = self.kernel_for(projection.bits, projection.group_size)
        partial = kernel(
            inputs=[
                x,
                projection.weight,
                projection.scales,
                projection.biases,
                expert_ids,
                projection.hidden_scalar,
                projection.out_scalar,
                projection.groups_scalar,
                mx.array(k, dtype=mx.uint32),
                projection.chunks_scalar,
            ],
            output_shapes=[(k, projection.output_dims, self.n_chunks)],
            output_dtypes=[mx.float32],
            grid=(k, projection.output_dims, self.n_chunks),
            threadgroup=(1, self.threadgroup_y, 1),
        )[0]
        return mx.sum(partial, axis=-1).astype(mx.float16)


@dataclass
class _Projection:
    weight: mx.array
    scales: mx.array
    biases: mx.array
    input_dims: int
    output_dims: int
    bits: int
    group_size: int
    n_groups: int
    hidden_scalar: mx.array = field(init=False)
    out_scalar: mx.array = field(init=False)
    groups_scalar: mx.array = field(init=False)
    chunks_scalar: mx.array = field(init=False)

    def __post_init__(self):
        self.hidden_scalar = mx.array(self.input_dims, dtype=mx.uint32)
        self.out_scalar = mx.array(self.output_dims, dtype=mx.uint32)
        self.groups_scalar = mx.array(self.n_groups, dtype=mx.uint32)
        self.chunks_scalar = mx.array(N_CHUNKS, dtype=mx.uint32)


@dataclass
class _SwitchConfig:
    gate: _Projection
    up: _Projection
    down: _Projection
    disabled_reason: str | None = None
    first_call_evaluated: bool = False


_MANAGER = AffineMoEDecodeManager()


class _WeakIdentityMap:
    """Weak module lookup that also accepts unhashable ``mlx.nn.Module`` objects."""

    def __init__(self):
        self._items: dict[int, tuple[weakref.ReferenceType[Any], _SwitchConfig]] = {}

    def __contains__(self, owner: Any) -> bool:
        entry = self._items.get(id(owner))
        return entry is not None and entry[0]() is owner

    def __setitem__(self, owner: Any, config: _SwitchConfig) -> None:
        identity = id(owner)
        items = self._items

        def _remove(reference: weakref.ReferenceType[Any]) -> None:
            current = items.get(identity)
            if current is not None and current[0] is reference:
                items.pop(identity, None)

        self._items[identity] = (weakref.ref(owner, _remove), config)

    def __getitem__(self, owner: Any) -> _SwitchConfig:
        missing = object()
        config = self.get(owner, missing)
        if config is missing:
            raise KeyError(owner)
        return config

    def get(self, owner: Any, default: Any = None) -> Any:
        entry = self._items.get(id(owner))
        if entry is None or entry[0]() is not owner:
            return default
        return entry[1]

    def discard(self, owner: Any) -> None:
        identity = id(owner)
        entry = self._items.get(identity)
        if entry is not None and entry[0]() is owner:
            self._items.pop(identity, None)


_CONFIGS = _WeakIdentityMap()
_PATCHED_CLASS: type | None = None
_ORIGINAL_CALL: Any = None
_WRAPPER: Any = None
_PATCHED_WEIGHTED_CLASS: type | None = None
_ORIGINAL_WEIGHTED_CALL: Any = None
_WEIGHTED_WRAPPER: Any = None
_FIRST_FAST_CALL_LOGGED = False
_FIRST_REGISTERED_FALLBACK_LOGGED = False


def _projection_config(module: Any, *, role: str) -> _Projection:
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    if not isinstance(module, QuantizedSwitchLinear):
        raise ValueError(f"{role} is not QuantizedSwitchLinear")
    expected = {
        "gate": ({2, 3}, 64),
        "up": ({2}, 64),
        "down": ({2}, 32),
    }[role]
    bits = int(module.bits)
    group_size = int(module.group_size)
    if bits not in expected[0] or group_size != expected[1]:
        raise ValueError(f"unsupported {role} bits/group_size={bits}/{group_size}")
    if getattr(module, "mode", "affine") != "affine":
        raise ValueError(f"{role} is not affine mode")
    if "bias" in module:
        raise ValueError(f"{role} has an unsupported post-matmul bias")
    weight = module.weight
    scales = module.scales
    biases = module.biases
    if biases is None:
        raise ValueError(f"{role} is missing affine biases")
    if weight.dtype != mx.uint32 or scales.dtype != mx.float16 or biases.dtype != mx.float16:
        raise ValueError(
            f"{role} dtype mismatch weight/scales/biases="
            f"{weight.dtype}/{scales.dtype}/{biases.dtype}"
        )
    if weight.ndim != 3 or scales.ndim != 3 or biases.shape != scales.shape:
        raise ValueError(f"{role} has invalid affine tensor ranks")
    input_dims = int(module.input_dims)
    output_dims = int(module.output_dims)
    n_groups = int(scales.shape[2])
    if n_groups % N_CHUNKS:
        raise ValueError(f"{role} group count {n_groups} is not divisible by {N_CHUNKS}")
    if input_dims != n_groups * group_size:
        raise ValueError(f"{role} input/group geometry mismatch")
    packed_cols = input_dims * bits // 32
    if input_dims * bits % 32 or int(weight.shape[2]) != packed_cols:
        raise ValueError(f"{role} packed-column geometry mismatch")
    if tuple(scales.shape[:2]) != tuple(weight.shape[:2]):
        raise ValueError(f"{role} scale/weight geometry mismatch")
    return _Projection(
        weight=weight,
        scales=scales,
        biases=biases,
        input_dims=input_dims,
        output_dims=output_dims,
        bits=bits,
        group_size=group_size,
        n_groups=n_groups,
    )


def _switch_config(module: Any) -> _SwitchConfig:
    activation = getattr(module, "activation", None)
    if float(getattr(activation, "swiglu_limit", -1.0)) != 10.0:
        raise ValueError("DSV4 affine SwitchGLU requires swiglu_limit=10")
    gate = _projection_config(module.gate_proj, role="gate")
    up = _projection_config(module.up_proj, role="up")
    down = _projection_config(module.down_proj, role="down")
    if gate.input_dims != up.input_dims or gate.output_dims != up.output_dims:
        raise ValueError("gate/up projection geometry differs")
    if down.input_dims != up.output_dims or down.output_dims != up.input_dims:
        raise ValueError("down projection does not invert gate/up geometry")
    if len({int(gate.weight.shape[0]), int(up.weight.shape[0]), int(down.weight.shape[0])}) != 1:
        raise ValueError("projection expert counts differ")
    if int(gate.weight.shape[0]) > 65535:
        raise ValueError("expert ids do not fit uint16")
    return _SwitchConfig(gate=gate, up=up, down=down)


def _infer_model_type(model: Any) -> str:
    for owner in (model, getattr(model, "model", None)):
        if owner is None:
            continue
        config = getattr(owner, "config", None)
        if isinstance(config, dict) and config.get("model_type"):
            return str(config["model_type"])
        args = getattr(owner, "args", None)
        if args is not None and getattr(args, "model_type", None):
            return str(args.model_type)
    return ""


def _install_class_wrapper(switch_class: type) -> None:
    global _PATCHED_CLASS, _ORIGINAL_CALL, _WRAPPER
    if _PATCHED_CLASS is switch_class and switch_class.__call__ is _WRAPPER:
        return
    if _PATCHED_CLASS is not None:
        raise RuntimeError("DSV4 affine fast path was already installed on a different SwitchGLU class")
    original_call = switch_class.__call__

    def _guarded_call(self: Any, x: mx.array, indices: mx.array) -> mx.array:
        global _FIRST_FAST_CALL_LOGGED, _FIRST_REGISTERED_FALLBACK_LOGGED
        config = _CONFIGS.get(self)
        if (
            config is None
            or config.disabled_reason is not None
            or bool(getattr(self, "training", False))
        ):
            return original_call(self, x, indices)
        leading = tuple(x.shape[:-1])
        # Keep multi-token prefill on mlx_lm's stock SwitchGLU. It sorts routed
        # rows once and issues one sorted gather_qmm per projection. Splitting
        # that dynamic routing into per-expert quantized_matmul calls requires a
        # host-visible ragged partition (or large padding), adds many launches,
        # and is not numerically identical to gather_qmm on MLX 0.31.2. This
        # wrapper therefore owns only the exact single-token decode shape.
        fallback_reason = None
        if x.dtype != mx.float16:
            fallback_reason = f"activation_dtype={x.dtype}"
        elif x.ndim not in (2, 3):
            fallback_reason = f"activation_rank={x.ndim}"
        elif int(x.size) != int(x.shape[-1]):
            fallback_reason = f"multi_token_shape={tuple(x.shape)}"
        elif indices.ndim != x.ndim or tuple(indices.shape[:-1]) != leading:
            fallback_reason = (
                f"route_shape={tuple(indices.shape)} activation_shape={tuple(x.shape)}"
            )
        elif int(indices.shape[-1]) <= 0 or int(indices.shape[-1]) >= 64:
            fallback_reason = f"route_top_k={int(indices.shape[-1])}"
        if fallback_reason is not None:
            if not _FIRST_REGISTERED_FALLBACK_LOGGED:
                logger.info(
                    "DSV4 affine MoE registered module used stock SwitchGLU: %s",
                    fallback_reason,
                )
                _FIRST_REGISTERED_FALLBACK_LOGGED = True
            return original_call(self, x, indices)
        k = int(indices.shape[-1])
        try:
            ids = indices.reshape(-1).astype(mx.uint16)
            token = x.reshape(-1)
            gate = _MANAGER.run_projection(token, config.gate, ids, k)
            up = _MANAGER.run_projection(token, config.up, ids, k)
            activated = self.activation(up, gate)
            down = _MANAGER.run_projection(activated, config.down, ids, k)
            result = down.reshape(leading + (k, config.down.output_dims))
            if not config.first_call_evaluated:
                mx.eval(result)
                config.first_call_evaluated = True
            if not _FIRST_FAST_CALL_LOGGED:
                logger.info(
                    "DSV4 affine MoE Metal decode path active: activation_dtype=%s "
                    "route_top_k=%d hidden=%d",
                    x.dtype,
                    k,
                    config.gate.input_dims,
                )
                _FIRST_FAST_CALL_LOGGED = True
            return result
        except Exception as exc:
            config.disabled_reason = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "DSV4 affine MoE fast path failed; disabling it for this SwitchGLU and falling back"
            )
            return original_call(self, x, indices)

    switch_class._vmlx_dsv4_affine_original_call = original_call
    switch_class._vmlx_dsv4_affine_decode_fastpath = True
    switch_class.__call__ = _guarded_call
    _PATCHED_CLASS = switch_class
    _ORIGINAL_CALL = original_call
    _WRAPPER = _guarded_call


def _weighted_decode(
    switch: Any,
    config: _SwitchConfig,
    x: mx.array,
    indices: mx.array,
    scores: mx.array,
) -> mx.array:
    """Run the official DSV4 score-before-down decode order.

    Current DSV4 packages can own this boundary instead of calling
    ``SwitchGLU.__call__``. They project the selected gate/up rows, apply
    limited SwiGLU in FP32, multiply the FP32 route score, cast once to the
    hidden dtype, and only then run down_proj. Keep that ordering while
    replacing only the three affine matvecs.
    """

    leading = tuple(x.shape[:-1])
    k = int(indices.shape[-1])
    ids = indices.reshape(-1).astype(mx.uint16)
    token = x.reshape(-1)
    gate = _MANAGER.run_projection(token, config.gate, ids, k)
    up = _MANAGER.run_projection(token, config.up, ids, k)

    gate_fp32 = gate.astype(mx.float32)
    up_fp32 = up.astype(mx.float32)
    swiglu_limit = float(switch.activation.swiglu_limit)
    if swiglu_limit > 0:
        up_fp32 = mx.clip(up_fp32, a_min=-swiglu_limit, a_max=swiglu_limit)
        gate_fp32 = mx.clip(gate_fp32, a_min=None, a_max=swiglu_limit)
    activated = mx.sigmoid(gate_fp32) * gate_fp32 * up_fp32
    activated = (
        activated * scores.reshape(-1).astype(mx.float32)[..., None]
    ).astype(x.dtype)
    down = _MANAGER.run_projection(activated, config.down, ids, k)
    return down.reshape(leading + (k, config.down.output_dims))


def _install_weighted_moe_wrapper(moe_class: type) -> None:
    """Patch the DSV4-owned weighted route boundary when it is present."""

    global _PATCHED_WEIGHTED_CLASS, _ORIGINAL_WEIGHTED_CALL, _WEIGHTED_WRAPPER
    if (
        _PATCHED_WEIGHTED_CLASS is moe_class
        and moe_class._weighted_routed_experts is _WEIGHTED_WRAPPER
    ):
        return
    if _PATCHED_WEIGHTED_CLASS is not None:
        raise RuntimeError(
            "DSV4 affine fast path was already installed on a different MoE class"
        )
    original_call = moe_class._weighted_routed_experts

    def _guarded_weighted_call(
        self: Any,
        x: mx.array,
        indices: mx.array,
        scores: mx.array,
    ) -> mx.array:
        global _FIRST_FAST_CALL_LOGGED, _FIRST_REGISTERED_FALLBACK_LOGGED
        switch = getattr(self, "switch_mlp", None)
        config = _CONFIGS.get(switch)
        if (
            config is None
            or config.disabled_reason is not None
            or bool(getattr(switch, "training", False))
        ):
            return original_call(self, x, indices, scores)

        leading = tuple(x.shape[:-1])
        fallback_reason = None
        if x.dtype != mx.float16:
            fallback_reason = f"activation_dtype={x.dtype}"
        elif x.ndim not in (2, 3):
            fallback_reason = f"activation_rank={x.ndim}"
        elif int(x.size) != int(x.shape[-1]):
            fallback_reason = f"multi_token_shape={tuple(x.shape)}"
        elif indices.ndim != x.ndim or tuple(indices.shape[:-1]) != leading:
            fallback_reason = (
                f"route_shape={tuple(indices.shape)} activation_shape={tuple(x.shape)}"
            )
        elif tuple(scores.shape) != tuple(indices.shape):
            fallback_reason = (
                f"score_shape={tuple(scores.shape)} route_shape={tuple(indices.shape)}"
            )
        elif int(indices.shape[-1]) <= 0 or int(indices.shape[-1]) >= 64:
            fallback_reason = f"route_top_k={int(indices.shape[-1])}"
        if fallback_reason is not None:
            if not _FIRST_REGISTERED_FALLBACK_LOGGED:
                logger.info(
                    "DSV4 affine MoE registered module used stock weighted route: %s",
                    fallback_reason,
                )
                _FIRST_REGISTERED_FALLBACK_LOGGED = True
            return original_call(self, x, indices, scores)

        try:
            result = _weighted_decode(switch, config, x, indices, scores)
            if not config.first_call_evaluated:
                mx.eval(result)
                config.first_call_evaluated = True
            if not _FIRST_FAST_CALL_LOGGED:
                logger.info(
                    "DSV4 affine MoE Metal weighted decode path active: "
                    "activation_dtype=%s route_top_k=%d hidden=%d",
                    x.dtype,
                    int(indices.shape[-1]),
                    config.gate.input_dims,
                )
                _FIRST_FAST_CALL_LOGGED = True
            return result
        except Exception as exc:
            config.disabled_reason = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "DSV4 affine MoE weighted fast path failed; disabling it for "
                "this SwitchGLU and falling back"
            )
            return original_call(self, x, indices, scores)

    moe_class._vmlx_dsv4_affine_original_weighted_call = original_call
    moe_class._vmlx_dsv4_affine_weighted_decode_fastpath = True
    moe_class._weighted_routed_experts = _guarded_weighted_call
    _PATCHED_WEIGHTED_CLASS = moe_class
    _ORIGINAL_WEIGHTED_CALL = original_call
    _WEIGHTED_WRAPPER = _guarded_weighted_call


def install_dsv4_affine_moe_fastpath(model: Any, *, model_type: str | None = None) -> int:
    """Enable the fused path only for validated DSV4 affine SwitchGLU modules.

    Unsupported modules remain completely owned by the stock call. Repeated
    installation is idempotent and only refreshes validated weak references.
    """
    if mx is None:
        logger.info("DSV4 affine MoE fast path skipped: MLX is unavailable")
        return 0
    # PR #248 measured a large win over an older M4 stock runtime. Current MLX
    # on M5 Max is already faster on native gather_qmm (18.7 vs 17.1 tok/s in
    # the exact 400-token keeper A/B), so this remains an explicit hardware
    # evaluation opt-in instead of regressing the production default.
    enabled = os.environ.get("VMLX_DSV4_AFFINE_MOE_FASTPATH", "0").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        logger.info(
            "DSV4 affine MoE fast path disabled by VMLX_DSV4_AFFINE_MOE_FASTPATH=%s",
            enabled,
        )
        return 0
    if (model_type or _infer_model_type(model)) != "deepseek_v4":
        logger.info("DSV4 affine MoE fast path skipped: model_type is not deepseek_v4")
        return 0
    from mlx_lm.models.switch_layers import SwitchGLU

    accepted: list[tuple[Any, _SwitchConfig]] = []
    rejected: list[str] = []
    switch_modules: list[Any] = []
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        logger.warning("DSV4 affine MoE fast path skipped: model has no named_modules()")
        return 0
    modules = list(named_modules())
    for name, module in modules:
        if not isinstance(module, SwitchGLU):
            continue
        switch_modules.append(module)
        try:
            accepted.append((module, _switch_config(module)))
        except (AttributeError, TypeError, ValueError) as exc:
            rejected.append(f"{name}: {exc}")
    if rejected:
        for module in switch_modules:
            _CONFIGS.discard(module)
        logger.warning(
            "DSV4 affine MoE fast path rejected the model atomically: "
            "%d compatible SwitchGLU modules, %d incompatible (%s)",
            len(accepted),
            len(rejected),
            "; ".join(rejected[:3]),
        )
        return 0
    if not accepted:
        logger.warning(
            "DSV4 affine MoE fast path found no exact affine SwitchGLU modules%s",
            f" ({'; '.join(rejected[:3])})" if rejected else "",
        )
        return 0
    accepted_ids = {id(module) for module, _config in accepted}
    weighted_owners = [
        module
        for _name, module in modules
        if id(getattr(module, "switch_mlp", None)) in accepted_ids
        and callable(getattr(module, "_weighted_routed_experts", None))
    ]
    weighted_classes = {type(module) for module in weighted_owners}
    weighted_switch_ids = [
        id(getattr(module, "switch_mlp", None)) for module in weighted_owners
    ]
    if weighted_owners and (
        len(weighted_classes) != 1
        or len(weighted_switch_ids) != len(accepted_ids)
        or len(set(weighted_switch_ids)) != len(weighted_switch_ids)
        or set(weighted_switch_ids) != accepted_ids
    ):
        for module in switch_modules:
            _CONFIGS.discard(module)
        logger.warning(
            "DSV4 affine MoE fast path rejected mixed owner topology: "
            "accepted_switches=%d weighted_owners=%d owner_classes=%d",
            len(accepted_ids),
            len(weighted_owners),
            len(weighted_classes),
        )
        return 0
    # Install the class boundary only after every module and owner relationship
    # has passed. This keeps registration atomic if wrapper topology collides.
    if weighted_classes:
        try:
            _install_weighted_moe_wrapper(next(iter(weighted_classes)))
        except RuntimeError as exc:
            for module in switch_modules:
                _CONFIGS.discard(module)
            logger.warning(
                "DSV4 affine MoE weighted owner install rejected: %s",
                exc,
            )
            return 0
    else:
        # Older DSV4 packages delegate directly to SwitchGLU and apply route
        # scores after it returns. Current packages own score-before-down in
        # ``_weighted_routed_experts`` and must never enter this generic hook.
        try:
            _install_class_wrapper(SwitchGLU)
        except RuntimeError as exc:
            for module in switch_modules:
                _CONFIGS.discard(module)
            logger.warning(
                "DSV4 affine MoE SwitchGLU install rejected: %s",
                exc,
            )
            return 0
    for module, config in accepted:
        # Hydration/reload can replace projection tensors while retaining the
        # enclosing SwitchGLU object. Always refresh the validated tensor refs.
        _CONFIGS[module] = config
    logger.info(
        "DSV4 affine MoE fast path enabled for %d exact modules; rejected=%d; "
        "weighted_owner=%s",
        len(accepted),
        len(rejected),
        bool(weighted_classes),
    )
    return len(accepted)


__all__ = ["AffineMoEDecodeManager", "install_dsv4_affine_moe_fastpath"]
