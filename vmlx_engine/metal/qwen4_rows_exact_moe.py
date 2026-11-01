"""Multi-row exact routed MoE for native-MTP verify rows (Qwen3.8 Flash-Next, affine experts).

Why
---
Greedy native MTP must emit exactly the AR text, so each verify row must reproduce its single-token decode
step bit for bit. For the routed MoE the row-exact verify branch therefore ran the *whole single-row decode
dispatch once per row* (`qwen4_affine_switchglu`: slice x/indices/scores, the selected-expert gate/up/SwiGLU
"pair" kernel, the expert down projection, the weighted top-10 reduction, then a concatenate).  That is ~10
small ops per row per layer.  Measured on M5 Max (CRACK-4M, R2-07): each extra verify row cost ~4.4 ms, and
once a layer's op count crossed MLX's command-buffer limits every layer paid an extra commit (W=5: +17 ms).

How
---
Two kernels replace the R per-row dispatches, each a *row-indexed copy* of the kernel the single-row path
already runs, with the token row taken from the grid's y coordinate:

* pair-rows: `affine_moe_pair_decode._pair_source` with x / expert_ids / output offset by the row
  (grid (32*I*K, R)).  Every thread executes the identical instruction sequence as the single-row launch for
  that row: same lane-strided word walk, same per-word dot order, same `simd_sum`, same SwiGLU epilogue.
* exact-down-rows: `qwen4_exact_down._SOURCE` (MLX gather_qmv lane arithmetic + MLX col_reduce_small top-10
  tree) with activated / expert_ids / route_scores / output offset by the threadgroup's y (grid (320*320, R)).

Because only addressing changes, row r of the multi-row launch is bit-identical to the single-row launch on
row r -- by construction, and checked against W single-token decode steps on the real bundle.  The expert
weight bytes are unchanged (each row still reads its own 10 experts); what goes away is ~10*(R-1) ops/layer.

Scope: verify rows only (2..16 rows), q2-8 affine experts in any pair layout the decode path already admits
(uniform or mixed gate/up bits/groups), the exact-down qualified shape (K640 -> N2560, top-10, fp16).  Anything else returns
None and the caller keeps the per-row path.  VMLX_QWEN4_ROWS_EXACT_MOE=0 disables.
"""
from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)
_STATE = {"failed": False, "observed": False, "calls": 0}
MAX_ROWS = 16


def rows_exact_moe_enabled() -> bool:
    return os.environ.get("VMLX_QWEN4_ROWS_EXACT_MOE", "1").strip().lower() not in {"0", "false", "off", "no"}


def _rowify_pair(src: str, hidden: int, top_k: int, inter: int) -> str:
    out = src.replace("uint tid = thread_position_in_grid.x;",
                      "uint tid = thread_position_in_grid.x;\n        uint trow = thread_position_in_grid.y;", 1)
    out = out.replace("expert_ids[route]", f"expert_ids[trow * {top_k}u + route]")
    out = re.sub(r"\bx\[", f"x[trow * {hidden}u + ", out)  # both _pair_source and _pair_source_mixed
    out = out.replace("output[(size_t)route *", f"output[(size_t)trow * {top_k * inter}u + (size_t)route *")
    if out.count("trow") < 4:
        raise ValueError("pair source layout changed; refusing to build a row-indexed copy")
    return out


@lru_cache(maxsize=64)  # mixed-bit bundles carry several per-layer pair configs
def _pair_rows_kernel(config):
    from .affine_moe_pair_decode import _pair_source
    src = _rowify_pair(_pair_source(config), config.hidden, config.top_k, config.intermediate)
    return mx.fast.metal_kernel(
        name=(f"vmlx_{config.family}_q{config.bits}g{config.group_size}_u{config.up_bits_eff}"
              f"g{config.up_group_size_eff}_selected_pair_swiglu_rows"),
        input_names=["x", "gate_weight", "gate_scales", "gate_biases", "up_weight", "up_scales", "up_biases",
                     "expert_ids"],
        output_names=["output"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=src,
    )


@lru_cache(maxsize=1)
def _down_rows_kernel():
    from .qwen4_exact_down import _kernel_header, _SOURCE
    src = _SOURCE.replace("uint tile=threadgroup_position_in_grid.x;",
                          "uint tile=threadgroup_position_in_grid.x;\n    uint trow=threadgroup_position_in_grid.y;", 1)
    src = src.replace("expert_ids[route]", "expert_ids[trow*10u+route]")
    src = src.replace("activated+route*640u+start", "activated+trow*6400u+route*640u+start")
    src = src.replace("route_scores[route]", "route_scores[trow*10u+route]")
    src = src.replace("output[first+lane]", "output[trow*2560u+first+lane]")
    if src.count("trow") < 5:
        raise ValueError("exact-down source layout changed; refusing to build a row-indexed copy")
    return mx.fast.metal_kernel(
        name="vmlx_qwen4_exact_down_v2_rows", header=_kernel_header(), source=src,
        input_names=["activated", "weight", "scales", "biases", "expert_ids", "route_scores"],
        output_names=["output"], ensure_row_contiguous=True)


def rows_exact_switchglu(switch: Any, x: mx.array, indices: mx.array, scores: mx.array):
    """Weighted routed output [1, R, 2560] for R verify rows, or None (caller keeps the per-row path)."""
    if not rows_exact_moe_enabled() or _STATE["failed"]:
        return None
    from .affine_moe_pair_decode import _CONFIG_ATTR, _projection_reason
    from .qwen4_exact_down import _compatible_runtime
    config = getattr(switch, _CONFIG_ATTR, None)
    # mixed gate/up layouts (JANG_4S/6S) use _pair_source_mixed, whose addressing the same
    # substitution rewrites; clamp_limit is the GLM-only epilogue and is not admitted here.
    if config is None or config.clamp_limit is not None or bool(getattr(switch, "training", False)):
        return None
    rows = int(x.shape[1]) if x.ndim == 3 else 0
    down = switch.down_proj
    if not (x.ndim == 3 and x.shape[0] == 1 and 2 <= rows <= MAX_ROWS and x.dtype == mx.float16
            and int(x.shape[-1]) == config.hidden == 2560 and config.intermediate == 640 and config.top_k == 10
            and tuple(indices.shape) == (1, rows, 10) and tuple(scores.shape) == (1, rows, 10)
            and scores.dtype == x.dtype and indices.dtype in (mx.int32, mx.uint32)
            and _compatible_runtime() and mx.default_device() == mx.gpu):
        return None
    if (_projection_reason(down, hidden=640, intermediate=2560) or down.bits not in (2, 3, 4, 6, 8)
            or down.group_size not in (32, 64) or down.scales.dtype != x.dtype):
        return None
    try:
        ids = indices.reshape(-1).astype(mx.uint32)
        activated = _pair_rows_kernel(config)(
            inputs=[mx.contiguous(x.reshape(-1)), switch.gate_proj.weight, switch.gate_proj.scales,
                    switch.gate_proj.biases, switch.up_proj.weight, switch.up_proj.scales,
                    switch.up_proj.biases, ids],
            template=[("T", x.dtype)],
            grid=(32 * config.intermediate * config.top_k, rows, 1), threadgroup=(128, 1, 1),
            output_shapes=[(rows, config.top_k, config.intermediate)], output_dtypes=[x.dtype])[0]
        out = _down_rows_kernel()(
            inputs=[activated, down.weight, down.scales, down.biases, ids, mx.contiguous(scores.reshape(-1))],
            template=[("T", x.dtype), ("BITS", down.bits), ("GS", down.group_size),
                      ("EXPERTS", down.weight.shape[0]), ("ROWS", 8)],
            grid=(320 * 320, rows, 1), threadgroup=(320, 1, 1),
            output_shapes=[(rows, 2560)], output_dtypes=[x.dtype])[0]
        _STATE["calls"] += 1
        if not _STATE["observed"]:
            mx.eval(out)
            _STATE["observed"] = True
            logger.info("Qwen4 rows-exact MoE active: rows<=%d q%d/g%d down q%d/g%d (verify rows, 2 launches/layer)",
                        MAX_ROWS, config.bits, config.group_size, down.bits, down.group_size)
        return out.reshape(1, rows, 2560)
    except (RuntimeError, ValueError):
        _STATE["failed"] = True
        logger.exception("Qwen4 rows-exact MoE failed; keeping the per-row verify path")
        return None
