"""Fused routed-expert combine for expert-sorted MoE prefill (vMLX).

The sorted MoE prefill (jangt/switch.JTMixedSwitchGLU._prefill) produces y_s = one output row per (token, slot) pair
in expert order. Combining it took three passes: unsort (gather y_s[argsort(order)], (T*k, D)), an fp32 widening of
that copy, and a (1 x k) @ (k x D) batched matmul per token -- 2.96 ms per MiMo layer for a 2048-token chunk
(top-8, D 4096; 0.85 ms of it the unsort gather), 10.6 ms at 8192 tokens.

Here one kernel reads each token's k rows straight from y_s through the inverse permutation, weights them and sums
in fp32 (slot order 0..k-1), and stores the token row once in the output dtype. Same arithmetic class as the fp32
batched matmul it replaces (fp32 products and sums; only the summation order of the k terms may differ).
"""
from __future__ import annotations

import functools

import mlx.core as mx

_SRC = r"""
  const uint t = thread_position_in_grid.y;
  const uint c = thread_position_in_grid.x * 4;
  const uint D = y_s_shape[1];
  if (c >= D) return;
  float4 acc = float4(0.0f);
  for (uint j = 0; j < K; j++) {
    const uint r = uint(inv[t * K + j]);
    const float wj = w[t * K + j];
    const device InT* row = y_s + size_t(r) * D + c;
    acc += wj * float4(float(row[0]), float(row[1]), float(row[2]), float(row[3]));
  }
  device OutT* o = out + size_t(t) * D + c;
  o[0] = OutT(acc.x); o[1] = OutT(acc.y); o[2] = OutT(acc.z); o[3] = OutT(acc.w);
"""


@functools.lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(name="vmlx_moe_sorted_combine", input_names=["y_s", "inv", "w"], output_names=["out"],
                                source=_SRC)


def sorted_combine(y_s: mx.array, order: mx.array, w2: mx.array, out_dtype=mx.float32):
    """y_s (T*k, D) expert-sorted rows (bf16/fp16/fp32), order = argsort of the flat (token, slot) expert ids
    (sorted position p holds pair order[p]), w2 (T, k) routing weights. Returns (T, D) = sum_j w2[t, j] * row(t, j)
    in out_dtype, or None when D is not a multiple of 4."""
    M, D = int(y_s.shape[0]), int(y_s.shape[1])
    T, k = int(w2.shape[0]), int(w2.shape[1])
    if D % 4 or M != T * k or y_s.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return None
    inv = mx.argsort(order).astype(mx.uint32)          # pair (t, j) -> its sorted row
    tg = min(256, D // 4)
    return _kernel()(inputs=[y_s, inv, w2.astype(mx.float32)], template=[("InT", y_s.dtype), ("OutT", out_dtype),
                                                                         ("K", k)],
                     grid=(D // 4, T, 1), threadgroup=(tg, 1, 1), output_shapes=[(T, D)], output_dtypes=[out_dtype])[0]
