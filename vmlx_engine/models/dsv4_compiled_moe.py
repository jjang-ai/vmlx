# SPDX-License-Identifier: Apache-2.0
"""Decode-only compiled native-gather path for DeepSeek-V4 routed MoE.

The DSV4 model's weighted route boundary issues native ``gather_qmm`` for
gate, up, and down projections, but leaves the sequence as a fresh Python/MLX
graph on every token. This patch compiles that exact sequence for the two
validated gate layouts (2-bit and 3-bit, both group-64) and only accepts the
single-token decode shape. Multi-token prefill and unsupported modules retain
the original implementation.
"""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - package inspection off Apple Silicon
    mx = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)
_SUPPORTED_MOES: set[int] = set()
_DISABLED_MOES: set[int] = set()
_COMPILED: dict[int, Any] = {}
_PATCHED_CLASS: type | None = None
_ORIGINAL_CALL: Any = None
_WRAPPER: Any = None


def _enabled() -> bool:
    value = os.environ.get("VMLX_DSV4_COMPILED_MOE", "1")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_affine_projection(projection: Any, bits: set[int], group_size: int) -> bool:
    try:
        return (
            int(getattr(projection, "bits", -1)) in bits
            and int(getattr(projection, "group_size", -1)) == group_size
            and getattr(projection, "mode", "affine") == "affine"
            and getattr(projection, "scales", None) is not None
            and getattr(projection, "biases", None) is not None
            and getattr(projection, "weight", None) is not None
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _is_supported_candidate(module: Any) -> bool:
    switch = getattr(module, "switch_mlp", None)
    if switch is None or not callable(
        getattr(module, "_weighted_routed_experts", None)
    ):
        return False
    gate = getattr(switch, "gate_proj", None)
    up = getattr(switch, "up_proj", None)
    down = getattr(switch, "down_proj", None)
    try:
        return (
            _is_affine_projection(gate, {2, 3}, 64)
            and _is_affine_projection(up, {2}, 64)
            and _is_affine_projection(down, {2}, 32)
            and int(gate.output_dims) == int(up.output_dims)
            and int(down.input_dims) == int(up.output_dims)
            and int(down.output_dims) == int(gate.input_dims)
            and float(
                getattr(getattr(switch, "activation", None), "swiglu_limit", -1.0)
            )
            == 10.0
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _clear_runtime_selection() -> None:
    _SUPPORTED_MOES.clear()
    _DISABLED_MOES.clear()


def _compiled_for(gate_bits: int):
    if gate_bits in _COMPILED:
        return _COMPILED[gate_bits]
    if mx is None:
        raise RuntimeError("MLX is unavailable")

    def _weighted_decode(
        x: mx.array,
        indices: mx.array,
        scores: mx.array,
        gate_weight: mx.array,
        gate_scales: mx.array,
        gate_biases: mx.array,
        up_weight: mx.array,
        up_scales: mx.array,
        up_biases: mx.array,
        down_weight: mx.array,
        down_scales: mx.array,
        down_biases: mx.array,
    ) -> mx.array:
        expanded = mx.expand_dims(x, (-2, -3))
        up = mx.gather_qmm(
            expanded,
            up_weight,
            up_scales,
            up_biases,
            rhs_indices=indices,
            transpose=True,
            group_size=64,
            bits=2,
            mode="affine",
            sorted_indices=False,
        )
        gate = mx.gather_qmm(
            expanded,
            gate_weight,
            gate_scales,
            gate_biases,
            rhs_indices=indices,
            transpose=True,
            group_size=64,
            bits=gate_bits,
            mode="affine",
            sorted_indices=False,
        )
        gate_fp32 = gate.astype(mx.float32)
        up_fp32 = up.astype(mx.float32)
        up_fp32 = mx.clip(up_fp32, a_min=-10.0, a_max=10.0)
        gate_fp32 = mx.clip(gate_fp32, a_min=None, a_max=10.0)
        activated = mx.sigmoid(gate_fp32) * gate_fp32 * up_fp32
        activated = (activated * scores.astype(mx.float32)[..., None, None]).astype(
            x.dtype
        )
        routed = mx.gather_qmm(
            activated,
            down_weight,
            down_scales,
            down_biases,
            rhs_indices=indices,
            transpose=True,
            group_size=32,
            bits=2,
            mode="affine",
            sorted_indices=False,
        )
        return routed.squeeze(-2)

    compiled = mx.compile(_weighted_decode)
    _COMPILED[gate_bits] = compiled
    return compiled


def install_dsv4_compiled_moe(model: Any) -> int:
    """Install the guarded native-gather decode graph on exact DSV4 MoEs."""

    if mx is None or not _enabled():
        _clear_runtime_selection()
        return 0
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        _clear_runtime_selection()
        return 0

    candidates: list[Any] = []
    rejected = 0
    for _name, module in list(named_modules()):
        switch = getattr(module, "switch_mlp", None)
        if switch is None or not callable(
            getattr(module, "_weighted_routed_experts", None)
        ):
            continue
        if _is_supported_candidate(module):
            candidates.append(module)
        else:
            rejected += 1
    if not candidates or rejected:
        _clear_runtime_selection()
        if rejected:
            logger.info(
                "DSV4 compiled MoE path skipped atomically: exact=%d rejected=%d",
                len(candidates),
                rejected,
            )
        return 0

    global _PATCHED_CLASS, _ORIGINAL_CALL, _WRAPPER
    moe_class = type(candidates[0])
    if getattr(moe_class, "_vmlx_dsv4_affine_weighted_decode_fastpath", False):
        logger.info(
            "DSV4 compiled native-gather path skipped: affine weighted path is explicitly selected"
        )
        return 0
    if any(type(module) is not moe_class for module in candidates):
        _clear_runtime_selection()
        return 0
    if _PATCHED_CLASS is not None and _PATCHED_CLASS is not moe_class:
        _clear_runtime_selection()
        return 0
    if _PATCHED_CLASS is None:
        original_call = moe_class._weighted_routed_experts

        def _guarded_weighted_call(
            self: Any,
            x: mx.array,
            indices: mx.array,
            scores: mx.array,
        ) -> mx.array:
            module_id = id(self)
            if (
                not _enabled()
                or module_id not in _SUPPORTED_MOES
                or module_id in _DISABLED_MOES
            ):
                return original_call(self, x, indices, scores)
            if (
                x.dtype != mx.float16
                or int(x.size) != int(x.shape[-1])
                or indices.ndim != x.ndim
                or tuple(indices.shape) != tuple(scores.shape)
                or int(indices.shape[-1]) <= 0
                or int(indices.shape[-1]) >= 64
            ):
                return original_call(self, x, indices, scores)
            switch = self.switch_mlp
            gate = switch.gate_proj
            up = switch.up_proj
            down = switch.down_proj
            try:
                compiled = _compiled_for(int(gate.bits))
                return compiled(
                    x,
                    indices,
                    scores,
                    gate.weight,
                    gate.scales,
                    gate.biases,
                    up.weight,
                    up.scales,
                    up.biases,
                    down.weight,
                    down.scales,
                    down.biases,
                )
            except Exception as exc:
                # A future MLX/JANG shape or dispatch change must not turn a
                # production request into a hard failure. Disable this exact
                # module for the process and preserve the reference path.
                _DISABLED_MOES.add(module_id)
                logger.warning(
                    "DSV4 compiled MoE disabled for layer=%s after dispatch "
                    "failure; falling back to stock path: %s",
                    getattr(self, "layer_id", "?"),
                    exc,
                )
                return original_call(self, x, indices, scores)

        moe_class._vmlx_dsv4_compiled_moe_original_call = original_call
        moe_class._vmlx_dsv4_compiled_moe = True
        moe_class._weighted_routed_experts = _guarded_weighted_call
        _PATCHED_CLASS = moe_class
        _ORIGINAL_CALL = original_call
        _WRAPPER = _guarded_weighted_call
    elif moe_class._weighted_routed_experts is not _WRAPPER:
        _clear_runtime_selection()
        return 0

    current_ids = {id(module) for module in candidates}
    _SUPPORTED_MOES.clear()
    _DISABLED_MOES.intersection_update(current_ids)
    _SUPPORTED_MOES.update(current_ids)
    logger.info(
        "DSV4 compiled native-gather MoE decode path enabled for %d modules; "
        "prefill remains stock",
        len(candidates),
    )
    return len(candidates)


__all__ = ["install_dsv4_compiled_moe"]
