# SPDX-License-Identifier: Apache-2.0
# The Metal kernel source below is taken verbatim from mlx-lm 0.32.0
# (mlx_lm/models/gated_delta.py, _make_gated_delta_packed_kernel), Copyright (c) 2023 Apple Inc., MIT License:
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#   documentation files (the "Software"), to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
#   to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above
#   copyright notice and this permission notice shall be included in all copies or substantial portions of the
#   Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.
"""Packed gated-delta recurrence for Dk=128 GDN layers (Qwen3.5 / Qwen3.5-MoE / Qwen3.8 / MiniCPM-V-4.7 trunks).

WHY
---
The pinned mlx-lm (0.31.3) recurrence kernel gives every value row its own 32-lane SIMD group: at Dk=128 each lane
holds 4 state values and every token pays two full-SIMD reductions per row, and the same k/q vectors are re-read by
all Dv row groups. mlx-lm 0.32.0 adds a packed specialization: four lanes per value row, eight rows per SIMD group,
32 contiguous state values per lane in registers.

EXACTNESS
---------
Each lane keeps the stock kernel's per-lane 4-element sequential partials; the three in-lane pairwise levels plus two
``simd_shuffle_xor`` levels reproduce an ascending xor-1/2/4/8/16 butterfly. That equals the stock ``simd_sum``
result exactly when ``simd_sum`` lowers to that tree, which mlx-lm states for current Apple GPUs. vMLX does not take
that on trust: ``tests/test_gdn_packed.py`` compares y and the state BITWISE against the stock 0.31.3 kernel at the
production shapes on the running device, and the dispatcher below keeps the stock kernel for every shape the packed
kernel does not cover (masks, vector gates, Dk != 128, non-fp32 state). ``VMLX_GDN_PACKED=0`` restores the stock
kernel everywhere.

g and beta are computed by the same mlx-lm 0.31.3 ops the stock path uses (``compute_g``, ``mx.sigmoid``), so the
two paths see identical kernel inputs.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional, Tuple

import mlx.core as mx

_SOURCE = r"""
        constexpr int lanes_per_row = 4;
        constexpr int rows_per_simdgroup = 32 / lanes_per_row;
        constexpr int values_per_lane = Dk / lanes_per_row;
        constexpr int partials_per_lane = values_per_lane / 4;

        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);

        auto lane = thread_index_in_simdgroup;
        auto row_in_simdgroup = lane / lanes_per_row;
        auto lane_in_row = lane & (lanes_per_row - 1);
        auto row_group = thread_position_in_grid.y;
        auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
        auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
        y += (b_idx * T * Hv + hv_idx) * Dv;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;

        float state[values_per_lane];
        for (int i = 0; i < values_per_lane; ++i) {
          state[i] = static_cast<float>(i_state[i]);
        }

        // g, beta: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float gt = static_cast<float>(g_[hv_idx]);

          // Partials mirror the generic kernel: each 4-element chain is one
          // original lane's sequential accumulation.
          float part[partials_per_lane];
          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] * gt;
              acc += state[e] * static_cast<float>(k_[e]);
            }
            part[pb] = acc;
          }
          // Butterfly levels xor 1,2,4 stay inside this lane (commutative
          // pairwise tree); levels xor 8,16 become the row-group shuffles.
          float kv_mem =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          kv_mem += simd_shuffle_xor(kv_mem, 1);
          kv_mem += simd_shuffle_xor(kv_mem, 2);

          auto delta =
              (static_cast<float>(v_[dv_idx]) - kv_mem) *
              static_cast<float>(beta_[hv_idx]);

          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] + static_cast<float>(k_[e]) * delta;
              acc += state[e] * static_cast<float>(q_[e]);
            }
            part[pb] = acc;
          }
          float out =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          out += simd_shuffle_xor(out, 1);
          out += simd_shuffle_xor(out, 2);
          if (lane_in_row == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }

        for (int i = 0; i < values_per_lane; ++i) {
          o_state[i] = static_cast<StT>(state[i]);
        }
"""

STATS = {"packed_calls": 0, "stock_calls": 0}


def packed_enabled() -> bool:
    return os.environ.get("VMLX_GDN_PACKED", "1").strip().lower() not in ("0", "false", "off", "no")


@lru_cache(maxsize=1)
def _kernel():
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name="vmlx_gated_delta_step_packed_btree",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=_SOURCE,
    )


def packed_eligible(q, k, v, g, state, mask) -> bool:
    return (
        mask is None
        and q.ndim == 4 and k.ndim == 4 and v.ndim == 4
        and g.ndim == 3
        and int(k.shape[-1]) == 128
        and int(v.shape[-1]) % 8 == 0
        and g.dtype == mx.float32
        and state is not None and state.dtype == mx.float32
        and q.dtype == k.dtype == v.dtype
        and int(v.shape[2]) % int(k.shape[2]) == 0
    )


def packed_gated_delta(q, k, v, g, beta, state) -> Tuple[mx.array, mx.array]:
    """Run the packed kernel on already-prepared g/beta (shapes checked by ``packed_eligible``)."""
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    y, state_out = _kernel()(
        inputs=[q, k, v, g, beta, state, T],
        template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(32, Dv // 8, B * Hv),
        threadgroup=(32, 2, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )
    return y, state_out


def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    use_kernel: bool = True,
) -> Tuple[mx.array, mx.array]:
    """Drop-in for ``mlx_lm.models.gated_delta.gated_delta_update`` (0.31.3 signature) with the packed kernel."""
    from mlx_lm.models import gated_delta as _gd

    if (
        use_kernel
        and packed_enabled()
        and mx.default_device() == mx.gpu
        and _kernel() is not None
    ):
        st = state
        if st is None:
            B, _, Hk, Dk = q.shape
            Hv, Dv = v.shape[-2:]
            st = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
        beta = mx.sigmoid(b)
        g = _gd.compute_g(A_log, a, dt_bias)
        if packed_eligible(q, k, v, g, st, mask):
            STATS["packed_calls"] += 1
            return packed_gated_delta(q, k, v, g, beta, st)
    STATS["stock_calls"] += 1
    return _gd.gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=use_kernel)
