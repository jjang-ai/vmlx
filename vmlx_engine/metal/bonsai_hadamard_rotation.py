# SPDX-License-Identifier: Apache-2.0
# Adapted from MTPLX, Copyright 2026 Youssof Altoukhi.
"""Opt-in fused signed Hadamard rotation adapted from MTPLX (Apache-2.0).

Source: https://github.com/youssofal/MTPLX/blob/v2.12.0/mtplx/kernels/hadamard_rotate.py
The upstream license and notice are in third_party/mtplx_hadamard.
The 1024-wide kernel is bit-exact with MLX's signed Hadamard chain on the
tested Dealign Bonsai activation shapes. A first-use self-check guards use on
the current GPU; an unsupported layout falls back to the stock chain.
"""

from __future__ import annotations

import math
from functools import lru_cache

import mlx.core as mx

BLOCK = 1024

_SOURCE = """
  constexpr int B = 1024;
  const uint tid = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const int K = x_shape[x_ndim - 1];
  const size_t base = (size_t)threadgroup_position_in_grid.x * B;
  const int col0 = (int)(base % (size_t)K);
  const int e0 = (int)tid * 8;
  float v[8];
  for (int p = 0; p < 8; ++p) {
    float e = float(x[base + e0 + p]);
    v[p] = INV ? e : e * signs[col0 + e0 + p];
  }
  for (int h = 1; h < 8; h <<= 1) {
    for (int p = 0; p < 8; ++p) {
      if ((p & h) == 0) {
        float a = v[p];
        float b = v[p + h];
        v[p] = a + b;
        v[p + h] = a - b;
      }
    }
  }
  for (uint m = 1; m < 32; m <<= 1) {
    const bool upper = (lane & m) != 0;
    for (int p = 0; p < 8; ++p) {
      float o = simd_shuffle_xor(v[p], (ushort)m);
      v[p] = upper ? (o - v[p]) : (v[p] + o);
    }
  }
  threadgroup float tg[B];
  for (int p = 0; p < 8; ++p) {
    tg[e0 + p] = v[p];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float scale = 0.03125f;
  for (int g = (int)tid; g < 256; g += 128) {
    float w0 = tg[g];
    float w1 = tg[g + 256];
    float w2 = tg[g + 512];
    float w3 = tg[g + 768];
    float a0 = w0 + w1;
    float a1 = w0 - w1;
    float a2 = w2 + w3;
    float a3 = w2 - w3;
    float r[4];
    r[0] = a0 + a2;
    r[2] = a0 - a2;
    r[1] = a1 + a3;
    r[3] = a1 - a3;
    for (int q = 0; q < 4; ++q) {
      const int e = g + 256 * q;
      float o = r[q] * scale;
      if (INV) {
        o = o * signs[col0 + e];
      }
      y[base + e] = static_cast<T>(o);
    }
  }
"""


_SELF_CHECK_PASSED: bool | None = None
_ATTRIBUTION_PRINTED = False


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="mtplx_prism_hadamard_rotate_1024",
        input_names=["x", "signs"],
        output_names=["y"],
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def rotate(x: mx.array, signs: mx.array, block: int, *, inverse: bool = False) -> mx.array | None:
    """``H(s * x)`` (or ``s * H(x)`` when ``inverse``), or None when out of contract."""

    if (
        block != BLOCK
        or x.dtype not in (mx.float16, mx.float32)
        or signs.dtype != mx.float32
        or x.ndim < 1
        or int(x.shape[-1]) % BLOCK
        or tuple(signs.shape) != (int(x.shape[-1]),)
        or x.size == 0
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
    ):
        return None
    if not self_check():
        return None
    global _ATTRIBUTION_PRINTED
    if not _ATTRIBUTION_PRINTED:
        print("Powered by MTPLX\nhttps://github.com/youssofal/mtplx", flush=True)
        _ATTRIBUTION_PRINTED = True
    return _launch(x, signs, inverse)


def self_check() -> bool:
    """Check both transform directions and supported dtypes once per process."""
    global _SELF_CHECK_PASSED
    if _SELF_CHECK_PASSED is not None:
        return _SELF_CHECK_PASSED
    try:
        signs = mx.where(mx.arange(5120) % 3 == 0, 1, -1).astype(mx.float32)
        for dtype in (mx.float16, mx.float32):
            x = (mx.arange(2 * 5120, dtype=mx.float32) % 103 - 51).reshape(2, 5120)
            x = (x * 0.03125).astype(dtype)
            for inverse in (False, True):
                z = x.astype(mx.float32)
                if not inverse:
                    z = z * signs
                z = mx.hadamard_transform(
                    z.reshape(-1, BLOCK), scale=1 / math.sqrt(BLOCK)
                ).reshape(x.shape)
                if inverse:
                    z = z * signs
                expected = z.astype(dtype)
                actual = _launch(x, signs, inverse)
                mx.eval(expected, actual)
                if not bool(mx.array_equal(expected, actual).item()):
                    _SELF_CHECK_PASSED = False
                    return False
    except Exception:
        _SELF_CHECK_PASSED = False
        return False
    _SELF_CHECK_PASSED = True
    return True


def _launch(x: mx.array, signs: mx.array, inverse: bool) -> mx.array:
    """The kernel itself, for an input the caller has checked (the self-check probes this)."""

    blocks = x.size // BLOCK
    return _kernel()(
        inputs=[x, signs],
        template=[("T", x.dtype), ("INV", bool(inverse))],
        grid=(128 * blocks, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]
