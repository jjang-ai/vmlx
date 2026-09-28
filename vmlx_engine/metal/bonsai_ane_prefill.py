"""Optional Bonsai prefill split using oMLX's built Qwen ANE kernels.

The adapter does not bundle a native binary. It imports the public oMLX
custom-kernel wrapper only when requested, checks runtime capabilities before
changing the model, and keeps prepared plans outside MLX module parameters.
"""

from __future__ import annotations

import logging
import math
import os
import time
import weakref
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from vmlx_engine.utils.jang_hadamard import HadamardQuantizedLinear

_LOGGER = logging.getLogger(__name__)
_TILE = 4096
_PLANS: dict[int, tuple[weakref.ReferenceType, _Plan]] = {}
_LOGGED_ROLES: set[str] = set()


@dataclass(frozen=True)
class _Plan:
    fast: Any
    role: str
    kind: str
    ane_model: Any | None
    cpu_weight: mx.array
    gpu_weight: mx.array
    gpu_scales: mx.array
    gpu_biases: mx.array
    cpu_threads: int


def _q2_to_q4(weight: mx.array) -> mx.array:
    words = np.asarray(weight)
    n, packed_k = words.shape
    codes = (words[:, :, None] >> (np.arange(16, dtype=np.uint32) * 2)) & 3
    codes = codes.reshape(n, packed_k * 16)
    if int(codes.max()) > 2:
        raise ValueError("Bonsai ANE prefill requires ternary Q2 codes")
    q4 = codes.reshape(n, -1, 8).astype(np.uint32)
    return mx.array(np.bitwise_or.reduce(q4 << (np.arange(8, dtype=np.uint32) * 4), axis=2))


def _validate_projection(linear: HadamardQuantizedLinear, role: str) -> int:
    if not isinstance(linear, HadamardQuantizedLinear):
        raise TypeError(f"Bonsai ANE {role} requires a signed-Hadamard projection")
    if linear.bits != 2 or linear.group_size != 128:
        raise ValueError(f"Bonsai ANE {role} requires 2-bit/group-128 affine weights")
    if "bias" in linear:
        raise ValueError(f"Bonsai ANE {role} does not support a separate projection bias")
    if linear.scales.dtype != mx.float16 or linear.biases.dtype != mx.float16:
        raise ValueError(f"Bonsai ANE {role} requires FP16 affine metadata")
    n = int(linear.weight.shape[0])
    if n < 192 or n % 64 or int(linear.weight.shape[1]) * 16 != linear.hadamard_width:
        raise ValueError(f"Bonsai ANE {role} has unsupported projection geometry")
    return n


def _register(linear: HadamardQuantizedLinear, plan: _Plan) -> None:
    ident = id(linear)
    if ident in _PLANS:
        raise RuntimeError("Bonsai ANE projection is already installed")

    def forget(reference: weakref.ReferenceType) -> None:
        current = _PLANS.get(ident)
        if current is not None and current[0] is reference:
            _PLANS.pop(ident, None)

    _PLANS[ident] = (weakref.ref(linear, forget), plan)
    linear._bonsai_ane_hybrid = True


def maybe_apply(linear: HadamardQuantizedLinear, x: mx.array) -> mx.array | None:
    """Apply a prepared prefill split to an already rotated FP16 activation."""
    rows = math.prod(x.shape[:-1])
    if x.dtype != mx.float16 or not 2048 <= rows <= _TILE:
        return None
    entry = _PLANS.get(id(linear))
    if entry is None or entry[0]() is not linear:
        return None
    plan = entry[1]
    if plan.role not in _LOGGED_ROLES:
        _LOGGER.info("Bonsai ANE first dispatch role=%s rows=%d", plan.role, rows)
        _LOGGED_ROLES.add(plan.role)
    flat = mx.contiguous(x.reshape(rows, x.shape[-1]))
    if plan.kind == "ane":
        if rows < _TILE:
            flat = mx.contiguous(mx.pad(flat, [(0, _TILE - rows), (0, 0)]))
        out = plan.fast.qwen35_ane_cpu_fp16_affine_qmm_t(
            flat, plan.cpu_weight, plan.gpu_weight, plan.gpu_scales,
            plan.gpu_biases, plan.ane_model, bits=4, variant=8,
            group_size=128, profile_category=0, cpu_threads=plan.cpu_threads,
            cpu_shared_resource=True,
        )[:rows]
    else:
        out = plan.fast.qwen35_cpu_fp16_affine_qmm_t(
            flat, plan.cpu_weight, plan.gpu_weight, plan.gpu_scales,
            plan.gpu_biases, 4, 8, 128, plan.cpu_threads, True,
        )
    return out.reshape((*x.shape[:-1], int(linear.weight.shape[0])))


def install(model) -> dict[str, int | float]:
    """Prepare every requested projection before registering any live plan."""
    if os.environ.get("VMLX_BONSAI_FP16_PREFILL_ONLY") != "1":
        raise ValueError("Bonsai ANE prefill requires FP16 prefill-only policy")
    if os.environ.get("VMLX_BONSAI_FP16_QMM") != "1":
        raise ValueError("Bonsai ANE prefill requires FP16 QMM")
    try:
        from omlx.custom_kernels.qwen35_prefill import fast
    except ImportError as exc:
        raise RuntimeError("Bonsai ANE prefill requires oMLX with built Qwen custom kernels") from exc
    if not (fast.qwen35_ane_available() and fast.qwen35_cpu_shared_resource_available()
            and fast.qwen35_ane_bank_compiler_available()):
        raise RuntimeError("Built oMLX ANE procedure-bank and shared-resource kernels are unavailable")
    required = (
        "qwen35_ane_linear_bank_builder",
        "qwen35_ane_cpu_fp16_affine_qmm_t",
        "qwen35_cpu_fp16_affine_qmm_t",
    )
    if any(not callable(getattr(fast, name, None)) for name in required):
        raise RuntimeError("Built oMLX package lacks the required Bonsai ANE calls")

    cpu_fraction = float(os.environ.get("VMLX_BONSAI_ANE_CPU_FRACTION", "0.1"))
    down_fraction = float(os.environ.get("VMLX_BONSAI_ANE_DOWN_CPU_FRACTION", "0.2"))
    cpu_threads = int(os.environ.get("VMLX_BONSAI_ANE_CPU_THREADS", "8"))
    bank_size = int(os.environ.get("VMLX_BONSAI_ANE_BANK_SIZE", "8"))
    gdn_enabled = os.environ.get("VMLX_BONSAI_ANE_GDN", "1") == "1"
    if not (0 < cpu_fraction < 0.3 and 0 < down_fraction < 0.4
            and 1 <= cpu_threads <= 32 and 1 <= bank_size <= 8):
        raise ValueError("Bonsai ANE split settings are out of bounds")

    trunk = getattr(getattr(model, "language_model", None), "model", None)
    layers = getattr(trunk, "layers", None)
    if layers is None or not layers:
        raise ValueError("Bonsai ANE prefill requires a Qwen language trunk")
    started = time.perf_counter()
    prepared: list[tuple[HadamardQuantizedLinear, _Plan]] = []
    pending: list[tuple[HadamardQuantizedLinear, str, mx.array, mx.array, mx.array, mx.array]] = []
    builder = fast.qwen35_ane_linear_bank_builder(_TILE)
    banks = 0
    gdn_count = 0

    def finish_bank() -> None:
        nonlocal builder, banks
        if not pending:
            return
        compiled = builder.compile(0, 0, builder.size)
        if len(compiled) != len(pending):
            raise RuntimeError("Bonsai ANE procedure bank returned an unexpected count")
        for (linear, role, cpu, gpu, scales, biases), ane_model in zip(pending, compiled):
            prepared.append((linear, _Plan(
                fast, role, "ane", ane_model, cpu, gpu, scales, biases, cpu_threads,
            )))
        pending.clear()
        builder = fast.qwen35_ane_linear_bank_builder(_TILE)
        banks += 1
        _LOGGER.info("Bonsai ANE prefill compiled bank %d", banks)

    def prepare_ane(linear: HadamardQuantizedLinear, role: str) -> None:
        n = _validate_projection(linear, role)
        ane_n = (int(n * 0.4) // 64) * 64
        cpu_n = (int(n * cpu_fraction) // 64) * 64
        if min(ane_n, cpu_n, n - ane_n - cpu_n) <= 0:
            raise ValueError(f"Bonsai ANE {role} split has an empty branch")
        ane_weight = mx.contiguous(mx.dequantize(
            linear.weight[:ane_n], linear.scales[:ane_n], linear.biases[:ane_n],
            group_size=128, bits=2, mode="affine", dtype=mx.float32,
        ))
        cpu_weight = mx.contiguous(mx.dequantize(
            linear.weight[ane_n:ane_n + cpu_n],
            linear.scales[ane_n:ane_n + cpu_n],
            linear.biases[ane_n:ane_n + cpu_n],
            group_size=128, bits=2, mode="affine", dtype=mx.float16,
        ))
        gpu_weight = _q2_to_q4(linear.weight[ane_n + cpu_n:])
        gpu_scales = mx.contiguous(linear.scales[ane_n + cpu_n:])
        gpu_biases = mx.contiguous(linear.biases[ane_n + cpu_n:])
        mx.eval(ane_weight, cpu_weight, gpu_weight, gpu_scales, gpu_biases)
        builder.add(ane_weight)
        pending.append((linear, role, cpu_weight, gpu_weight, gpu_scales, gpu_biases))
        if len(pending) == bank_size:
            finish_bank()

    def prepare_down(linear: HadamardQuantizedLinear) -> None:
        n = _validate_projection(linear, "mlp_down_proj")
        cpu_n = (int(n * down_fraction) // 64) * 64
        if not 0 < cpu_n < n:
            raise ValueError("Bonsai ANE down split has an empty branch")
        cpu_weight = mx.contiguous(mx.dequantize(
            linear.weight[:cpu_n], linear.scales[:cpu_n], linear.biases[:cpu_n],
            group_size=128, bits=2, mode="affine", dtype=mx.float16,
        ))
        gpu_weight = _q2_to_q4(linear.weight[cpu_n:])
        gpu_scales = mx.contiguous(linear.scales[cpu_n:])
        gpu_biases = mx.contiguous(linear.biases[cpu_n:])
        mx.eval(cpu_weight, gpu_weight, gpu_scales, gpu_biases)
        prepared.append((linear, _Plan(
            fast, "mlp_down_proj", "down", None, cpu_weight,
            gpu_weight, gpu_scales, gpu_biases, cpu_threads,
        )))

    for index, layer in enumerate(layers, 1):
        for name in ("gate_proj", "up_proj"):
            prepare_ane(getattr(layer.mlp, name), f"mlp_{name}")
        gdn = getattr(layer, "linear_attn", None)
        if gdn_enabled and gdn is not None:
            for name in ("in_proj_qkv", "in_proj_z", "out_proj"):
                prepare_ane(getattr(gdn, name), f"gdn_{name}")
                gdn_count += 1
        prepare_down(layer.mlp.down_proj)
        _LOGGER.info("Bonsai ANE prefill prepared layer %d/%d", index, len(layers))
    finish_bank()

    if any(id(linear) in _PLANS for linear, _ in prepared):
        raise RuntimeError("Bonsai ANE prefill is already installed on this model")
    for linear, plan in prepared:
        _register(linear, plan)
    elapsed = time.perf_counter() - started
    receipt = {
        "ane_projections": sum(plan.kind == "ane" for _, plan in prepared),
        "gdn_projections": gdn_count,
        "down_projections": sum(plan.kind == "down" for _, plan in prepared),
        "banks": banks,
        "prepare_seconds": round(elapsed, 2),
    }
    _LOGGER.info("Bonsai ANE prefill installed: %s", receipt)
    return receipt
