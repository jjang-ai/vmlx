"""Blocked-sequential Gated DeltaNet prefill kernel.

Ported from omlx (jundot/omlx, omlx/custom_kernels/qwen35_prefill/gdn.py,
kernel S / ``gated_delta_blocked_seq``, v0.5.4rc1) — attribution per house
rules; the algorithm is the exact mlx-lm sequential recurrence, restructured
for Apple-GPU memory traffic:

- The stock mlx-lm ``gated_delta_kernel`` launches ``grid=(32, Dv, B*Hv)``
  threadgroups that each re-read the same k/q rows from device memory once
  per Dv-slice — ~32x redundant traffic (omlx measured ~13 GB per 16k-token
  layer on Qwen3.5/3.6 shapes).
- Here k/q/v/g/beta are cooperatively staged into threadgroup memory in
  TB-token blocks (coalesced), each v-head is split into Dv/32 row blocks so
  8x fewer threadgroups touch the same k/q rows, and each row is read from
  device exactly once per threadgroup.
- Each SIMD group computes four value rows. Its 32 lanes own four
  consecutive key dimensions each, using the same accumulation order and
  ``simd_sum`` tree as stock mlx-lm. ``float4 st[4]`` vectorizes those four
  value rows without changing their arithmetic. No threadgroup barriers
  are needed inside the token loop.
- omlx's negative result adopted with the port: the chunked WY/FLA
  reformulation costs ~2x the FLOPs and LOSES to blocked-sequential on
  Apple GPUs; do not resurrect it here.

Contract (identical to mlx-lm's scalar-gating kernel path):
  q, k: [B, T, Hk, Dk] (input dtype), v: [B, T, Hv, Dv],
  g, beta: [B, T, Hv] (cast to fp32 here), state: [B, Hv, Dv, Dk] fp32.
  Returns y [B, T, Hv, Dv] in the input dtype and the fp32 final state.

Structural requirements (checked by ``blocked_prefill_eligible``):
  Dk == 128 (32 lanes x 4 fp32 state elements per value row), Dv % 32 == 0,
  Hv % Hk == 0, scalar gating only, no mask. Anything else must stay on the
  stock path.
"""

# Adapted from MTPLX 21be78b3f51820eecef020e5e4855c0715eaf9a5
# https://github.com/youssofal/MTPLX/blob/21be78b3f51820eecef020e5e4855c0715eaf9a5/mtplx/kernels/gdn_blocked_prefill.py
# Apache-2.0; original oMLX provenance is retained above.
# vMLX changes: explicit opt-in dispatch, complete shape/dtype guards,
# threadgroup-memory bound, stock-order SIMD reductions vectorized across
# four value rows, no process-wide monkeypatch.

from __future__ import annotations

import os
import logging
from typing import Optional, Tuple

import mlx.core as mx

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

_KERNEL_S_SRC = """
    constexpr int TB = 32;                             // time block
    constexpr int DB = 32;                             // dv rows per threadgroup
    const int tid = thread_position_in_threadgroup.x;  // 0..255
    const int blk = threadgroup_position_in_grid.x;    // Dv/DB block
    const int hv  = threadgroup_position_in_grid.y;
    const int b   = threadgroup_position_in_grid.z;
    const int hk  = hv / (Hv / Hk);
    const int dv0 = blk * DB;

    // Each SIMD group computes four dv rows. Its 32 lanes partition Dk
    // into four consecutive elements, matching stock mlx-lm summation.
    const int dv  = (tid / 32) * 4;            // first of four dv rows
    const int seg = tid % 32;            // 0..31
    const int d0  = seg * 4;

    threadgroup InT k_s[TB][Dk + 8];
    threadgroup InT q_s[TB][Dk + 8];
    threadgroup InT v_s[TB][DB + 8];
    threadgroup float g_s[TB];
    threadgroup float b_s[TB];

    const device InT* k_base = k + ((size_t)b * T * Hk + hk) * Dk;
    const device InT* q_base = q + ((size_t)b * T * Hk + hk) * Dk;
    const device InT* v_base = v + ((size_t)b * T * Hv + hv) * Dv + dv0;
    const size_t krow = (size_t)Hk * Dk;

    // Four stock-order Dk elements, vectorized across four value rows.
    float4 st[4];
    {
        const device float* S_in = state_in + (((size_t)b * Hv + hv) * Dv + dv0 + dv) * Dk + d0;
        for (int i=0;i<4;++i) st[i] = float4(S_in[i], S_in[Dk+i], S_in[2*Dk+i], S_in[3*Dk+i]);
    }

    device InT* y_base = y + ((size_t)b * T * Hv + hv) * Dv + dv0;

    for (int t0 = 0; t0 < T; t0 += TB) {
        const int tt = min(TB, T - t0);
        // cooperative staging (coalesced): k/q rows, v slice, g/beta
        for (int p = tid; p < tt * Dk; p += 256) {
            const int r = p / Dk, d = p % Dk;
            k_s[r][d] = k_base[(size_t)(t0 + r) * krow + d];
            q_s[r][d] = q_base[(size_t)(t0 + r) * krow + d];
        }
        for (int p = tid; p < tt * DB; p += 256) {
            const int r = p / DB, d = p % DB;
            v_s[r][d] = v_base[(size_t)(t0 + r) * Hv * Dv + d];
        }
        for (int p = tid; p < tt; p += 256) {
            g_s[p] = g[((size_t)b * T + t0 + p) * Hv + hv];
            b_s[p] = beta[((size_t)b * T + t0 + p) * Hv + hv];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int t = 0; t < tt; ++t) {
            const float gt = g_s[t];
            const float bt = b_s[t];
            float4 kv_mem = 0.0f;
            for (int i=0;i<4;++i) {
                st[i] = st[i] * gt;
                kv_mem += st[i] * k_s[t][d0+i];
            }
            kv_mem = simd_sum(kv_mem);
            float4 values = float4(v_s[t][dv],v_s[t][dv+1],v_s[t][dv+2],v_s[t][dv+3]);
            auto delta = (values-kv_mem)*bt;
            float4 out = 0.0f;
            for (int i=0;i<4;++i) {
                st[i] = st[i] + k_s[t][d0+i] * delta;
                out += st[i] * q_s[t][d0+i];
            }
            out = simd_sum(out);
            if (seg==0) for(int r=0;r<4;++r) y_base[(size_t)(t0+t)*Hv*Dv+dv+r]=(InT)out[r];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    {
        device float* S_out = state_out + (((size_t)b * Hv + hv) * Dv + dv0 + dv) * Dk + d0;
        for (int i=0;i<4;++i) for(int r=0;r<4;++r) S_out[r*Dk+i] = st[i][r];
    }
"""

_SUPPORTED_BLOCK_T = (16, 32, 48)
_kernel_by_tb: dict = {}
_attribution_logged = False


def _normalize_block_t(block_t, input_dtype=None) -> int:
    if block_t is None:
        configured = os.environ.get("VMLX_QWEN4_GDN_BLOCKED_PREFILL_TB")
        if configured is not None:
            block_t = configured
        else:
            # fp32 inputs at TB=32 need 40,192 bytes of threadgroup memory on
            # the 128/128 layout, over Metal's 32 KiB limit; TB=16 fits
            # (20,096 bytes). bf16/fp16 fit at TB=32. (omlx measurement.)
            block_t = 16 if input_dtype == mx.float32 else 32
    block_t = int(block_t)
    if block_t not in _SUPPORTED_BLOCK_T:
        raise ValueError(
            f"VMLX_QWEN4_GDN_BLOCKED_PREFILL_TB must be one of {_SUPPORTED_BLOCK_T}, got {block_t}"
        )
    if input_dtype == mx.float32 and block_t > 16:
        raise ValueError("float32 requires TB=16 to fit Metal threadgroup memory")
    return block_t


def _get_kernel(block_t=None, input_dtype=None):
    block_t = _normalize_block_t(block_t, input_dtype)
    global _attribution_logged
    if not _attribution_logged:
        logging.getLogger(__name__).info(
            "Experimental blocked GDN prefill: Powered by MTPLX "
            "https://github.com/youssofal/mtplx (oMLX-derived kernel)"
        )
        _attribution_logged = True
    kernel = _kernel_by_tb.get(block_t)
    if kernel is None:
        source = _KERNEL_S_SRC.replace(
            "constexpr int TB = 32;", f"constexpr int TB = {block_t};"
        )
        kernel = mx.fast.metal_kernel(
            name=f"vmlx_qwen4_gdn_blocked_prefill_tb{block_t}",
            input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
            output_names=["y", "state_out"],
            source=source,
            header=_HEADER,
        )
        _kernel_by_tb[block_t] = kernel
    return kernel


def blocked_prefill_eligible(
    q: mx.array,
    v: mx.array,
    g: mx.array,
    mask,
    state,
) -> bool:
    """Structural gate: route only shapes the kernel is written for."""
    if mask is not None or g.ndim != 3:
        return False
    if q.ndim != 4 or v.ndim != 4:
        return False
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[2:]
    if Dk != 128 or Dv % 32 != 0 or Hk <= 0 or Hv % Hk != 0:
        return False
    if q.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return False
    if state is not None and state.dtype != mx.float32:
        return False
    return True


def gated_delta_blocked_prefill(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
    block_t=None,
) -> Tuple[mx.array, mx.array]:
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[2:]
    in_dtype = q.dtype
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)
    kernel = _get_kernel(block_t, in_dtype)
    y, state_out = kernel(
        inputs=[q, k, v, g, beta, state, T],
        template=[("InT", in_dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(256 * (Dv // 32), Hv, B),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[in_dtype, mx.float32],
    )
    return y, state_out




def qwen4_blocked_gated_delta_update(
    q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True
):
    """Opt-in prefill dispatch; decode, verification and masked calls stay stock."""
    from mlx_lm.models.gated_delta import compute_g, gated_delta_update

    enabled = os.environ.get("VMLX_QWEN4_GDN_BLOCKED_PREFILL", "0").lower() in {
        "1", "true", "yes", "on"
    }
    eligible = (
        enabled and use_kernel and mask is None
        and mx.default_device() == mx.gpu and mx.metal.is_available()
        and q.ndim == 4 and v.ndim == 4 and q.shape[1] >= 16
        and k.shape == q.shape and v.shape[:2] == q.shape[:2]
        and q.dtype == k.dtype == v.dtype
        and a.shape == b.shape == v.shape[:3]
        and A_log.shape == dt_bias.shape == (v.shape[2],)
    )
    if eligible:
        g = compute_g(A_log, a, dt_bias)
        expected_state = (q.shape[0], v.shape[2], v.shape[3], q.shape[3])
        if (blocked_prefill_eligible(q, v, g, mask, state)
                and v.shape[2] > 0 and v.shape[3] > 0
                and (state is None or state.shape == expected_state)):
            return gated_delta_blocked_prefill(q, k, v, g, mx.sigmoid(b), state)
    return gated_delta_update(
        q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=use_kernel
    )
