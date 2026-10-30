# SPDX-License-Identifier: Apache-2.0
"""Fused prefill attention on the M5 tensor units (NAX) for q/k head 192 with v head 128 (MiMo-V2 family).

MLX fuses prefill attention only when the query/key and value head widths are equal; for 192/128 its SDPA builds the
[H, L, T] score matrix. MiMo's full-attention layers worked around that by zero-padding q/k/v to 256 and its
sliding-window layers (window 128) materialized scores. This module runs one fused kernel for both
(``nax_prefill_attention.metal``) on top of MLX's own NAX tile types, whose header text is read from the installed
MLX package (``mlx/include/.../steel``), so the fragments, loads and matrix products are MLX's.

Long key ranges run as several dispatches over consecutive key blocks; each hands the fp32 row state (output
accumulator, running max, running sum) to the next, so a row sees exactly the operations of one dispatch. Those
calls also split the head dims over simdgroup pairs (the scores become the sum of two fp32 halves).

Fail-closed: bf16/fp16 (192, 128) with more than 8 query rows on a NAX GPU, after a once-per-process check against
fp32 (causal, window + sinks, bool mask). Anything else returns None and the caller keeps its existing route.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

SUPPORTED_HEAD_DIMS = frozenset({(192, 128)})

KEY_BLOCK = 32                 # keys per block (the kernel's BK)
ROW_GROUPS = 4                 # simdgroups per threadgroup
THREADS = 32 * ROW_GROUPS
# Key-range dispatches: keys per dispatch, and the threadgroup count (B * H * ceil(L / 64)) from which they pay off.
KEYS_PER_DISPATCH = 8192
MIN_GROUPS_FOR_PASSES = 512

_DTYPE_NAMES = {mx.bfloat16: "bfloat16_t", mx.float16: "half"}
_HERE = Path(__file__).resolve().parent

# MLX headers that define the NAX tiles, in dependency order (paths under mlx/include/mlx/backend/metal/kernels).
_MLX_TILE_HEADERS = ("steel/defines.h", "steel/utils/type_traits.h", "steel/utils/integral_constant.h",
                     "steel/attn/nax.h")

_ENTRY = """
  threadgroup float exchange[WN == 2 ? WM * 2 * 16 * 32 : 1];
  vmlx_attn::split_value_attention<T, 16 * WM, BD, BDV, WM, WN, ALIGN_Q, ALIGN_K, HAS_MASK, CAUSAL, WINDOW,
                                   HAS_SINKS, FIRST, LAST>(
      q, k, v, out, geom, scale[0], q_strides, k_strides, v_strides, mask_strides, mask, sinks, carry,
      carry_out, exchange, simdgroup_index_in_threadgroup, thread_index_in_simdgroup,
      threadgroup_position_in_grid);
"""


def _mlx_kernel_dir() -> Path:
    return Path(mx.__file__).resolve().parent / "include" / "mlx" / "backend" / "metal" / "kernels"


@lru_cache(maxsize=1)
def _header() -> str:
    """MLX's NAX tile headers inlined (their own includes and include guards dropped), then our kernel."""
    base = _mlx_kernel_dir()
    parts = ["#include <metal_stdlib>\n#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n"]
    for rel in _MLX_TILE_HEADERS:
        text = (base / rel).read_text()
        kept = [line for line in text.splitlines()
                if not line.lstrip().startswith('#include "mlx/') and line.strip() != "#pragma once"]
        parts.append(f"// ---- MLX {rel} ----\n" + "\n".join(kept) + "\n")
    parts.append((_HERE / "nax_prefill_attention.metal").read_text())
    return "\n".join(parts)


@lru_cache(maxsize=None)
def _kernel(dtype, bd: int, bdv: int, wn: int, align_q: bool, align_k: bool, has_mask: bool, causal: bool,
            window: int, has_sinks: bool, first: bool, last: bool):
    flags = "".join(str(int(f)) for f in (align_q, align_k, has_mask, causal, has_sinks, first, last))
    consts = (f"  constexpr int BD = {bd};\n  constexpr int BDV = {bdv};\n  constexpr int WN = {wn};\n"
              f"  constexpr int WM = {ROW_GROUPS // wn};\n  constexpr int WINDOW = {int(window)};\n"
              f"  using T = {_DTYPE_NAMES[dtype]};\n")
    for name, value in (("ALIGN_Q", align_q), ("ALIGN_K", align_k), ("HAS_MASK", has_mask), ("CAUSAL", causal),
                        ("HAS_SINKS", has_sinks), ("FIRST", first), ("LAST", last)):
        consts += f"  constexpr bool {name} = {'true' if value else 'false'};\n"
    return mx.fast.metal_kernel(
        name=f"vmlx_prefill_attn_{bd}x{bdv}_wn{wn}_w{int(window)}_{flags}",
        input_names=["q", "k", "v", "mask", "sinks", "carry", "geom", "scale"],
        output_names=["out", "carry_out"],
        header=_header(),
        source=consts + _ENTRY,
        ensure_row_contiguous=False,
    )


def _pass_edges(n_groups: int, nk: int, window: int) -> list[int]:
    """Key-block boundaries of the dispatches. One dispatch for a window layer (each tile reads only its band) or a
    small grid; otherwise about KEYS_PER_DISPATCH keys each."""
    if window > 0 or n_groups < MIN_GROUPS_FOR_PASSES:
        return [0, nk]
    n = max(1, min(nk, round(nk * KEY_BLOCK / KEYS_PER_DISPATCH)))
    return [(i * nk) // n for i in range(n + 1)]


def _dispatch(q, k, v, scale, mask, sinks, window, edges, wn) -> mx.array:
    B, H, L, D = q.shape
    T, DV = int(k.shape[2]), int(v.shape[3])
    bq = 16 * (ROW_GROUPS // wn)
    nq, nk = -(-L // bq), -(-T // KEY_BLOCK)
    full_q, full_k = L // bq, T // KEY_BLOCK
    causal = isinstance(mask, str) or window > 0
    has_mask = isinstance(mask, mx.array)
    has_sinks = sinks is not None
    mask_arr = mx.broadcast_to(mask, (B, H, L, T)) if has_mask else mx.zeros((1,), dtype=mx.bool_)
    sinks_arr = mx.contiguous(sinks.astype(q.dtype)) if has_sinks else mx.zeros((1,), dtype=q.dtype)
    scale_arr = mx.array([scale], dtype=mx.float32)
    carry = mx.zeros((1,), dtype=mx.float32)
    carry_floats = B * H * nq * bq * (DV + 2)
    out = None
    for i in range(len(edges) - 1):
        first, last = i == 0, i == len(edges) - 2
        geom = mx.array([B, H, L, T, H // int(k.shape[1]), nq, nk, full_q, full_k, L - full_q * bq,
                         T - full_k * KEY_BLOCK, T - L, edges[i], edges[i + 1]], dtype=mx.int32)
        run = _kernel(q.dtype, D, DV, wn, L % bq == 0, T % KEY_BLOCK == 0, has_mask, causal, window, has_sinks,
                      first, last)
        out, carry = run(
            inputs=[q, k, v, mask_arr, sinks_arr, carry, geom, scale_arr],
            grid=(nq * THREADS, H, B),
            threadgroup=(THREADS, 1, 1),
            output_shapes=[(B, L, H, DV) if last else (1,), (1,) if last else (carry_floats,)],
            output_dtypes=[q.dtype, mx.float32],
        )
    # Rows were written as [B, L, H, DV]; the [B, H, L, DV] view makes the caller's transpose + reshape free.
    return out.transpose(0, 2, 1, 3)


def _run(q, k, v, scale, mask, sinks, window) -> mx.array:
    B, H, L, _ = q.shape
    nk = -(-int(k.shape[2]) // KEY_BLOCK)
    edges = _pass_edges(B * H * -(-L // 64), nk, window)
    wn = 2 if len(edges) > 2 else 1
    if wn == 2 and not passes_check_passed(wn):
        edges, wn = [0, nk], 1
    return _dispatch(q, k, v, scale, mask, sinks, window, edges, wn)


def reference_attention(q, k, v, scale, mask=None, sinks=None, window=0):
    """fp32 attention with bottom-right causal alignment, optional window (i - j < window), bool mask and sinks."""
    B, H, L, _ = q.shape
    Hk, T = int(k.shape[1]), int(k.shape[2])
    g = H // Hk
    s = (q.astype(mx.float32).reshape(B, Hk, g, L, -1) * scale) @ k.astype(mx.float32)[:, :, None].swapaxes(-1, -2)
    s = s.reshape(B, H, L, T)
    rows = mx.arange(L)[:, None] + (T - L)
    cols = mx.arange(T)[None]
    allowed = mx.ones((L, T), dtype=mx.bool_)
    if isinstance(mask, str) or window > 0:
        allowed = allowed & (rows >= cols)
    if window > 0:
        allowed = allowed & (rows - cols < window)
    if isinstance(mask, mx.array):
        allowed = allowed & mask
    s = mx.where(allowed, s, -mx.inf)
    if sinks is not None:
        sk = mx.broadcast_to(sinks.astype(mx.float32).reshape(1, H, 1, 1), (B, H, L, 1))
        p = mx.softmax(mx.concatenate([s, sk], axis=-1), axis=-1)[..., :-1]
    else:
        p = mx.softmax(s, axis=-1)
    return (p.reshape(B, Hk, g, L, T) @ v.astype(mx.float32)[:, :, None]).reshape(B, H, L, -1)


def _probe(seed):
    kq, kk, kv, ks = mx.random.split(mx.random.key(seed), 4)
    q = (0.5 * mx.random.normal((1, 4, 100, 192), key=kq)).astype(mx.bfloat16)
    k = (0.5 * mx.random.normal((1, 2, 300, 192), key=kk)).astype(mx.bfloat16)
    v = (0.5 * mx.random.normal((1, 2, 300, 128), key=kv)).astype(mx.bfloat16)
    sinks = mx.random.normal((4,), key=ks).astype(mx.bfloat16)
    return q, k, v, sinks, 192 ** -0.5


@lru_cache(maxsize=1)
def self_check_passed() -> bool:
    """Once per process: unaligned causal, window 37 + sinks, and a bool mask, each within 2e-2 of fp32."""
    try:
        q, k, v, sinks, scale = _probe(192128)
        bmask = (mx.arange(300)[None, :] % 3 != 1) | (mx.arange(100)[:, None] + 200 == mx.arange(300)[None, :])
        err = 0.0
        for mask, sk, window in (("causal", None, 0), ("causal", sinks, 37), (bmask, None, 0)):
            out = _dispatch(q, k, v, scale, mask, sk, window, [0, 10], 1)
            ref = reference_attention(q, k, v, scale, mask, sk, window)
            err = max(err, mx.abs(out.astype(mx.float32) - ref).max().item())
    except Exception as exc:  # noqa: BLE001 - any failure disables the route
        logger.warning("NAX 192/128 attention disabled: self-check failed (%s)", exc)
        return False
    if err >= 2e-2:
        logger.warning("NAX 192/128 attention disabled: self-check error %.3g vs fp32", err)
        return False
    logger.info("NAX 192/128 attention enabled (self-check max error %.3g vs fp32)", err)
    return True


@lru_cache(maxsize=None)
def passes_check_passed(wn: int) -> bool:
    """Several key-range dispatches must reproduce one dispatch bit for bit and stay within 2e-2 of fp32."""
    try:
        q, k, v, _, scale = _probe(192129)
        one = _dispatch(q, k, v, scale, "causal", None, 0, [0, 10], wn)
        many = _dispatch(q, k, v, scale, "causal", None, 0, [0, 3, 6, 10], wn)
        err = mx.abs(many.astype(mx.float32) - reference_attention(q, k, v, scale, "causal")).max().item()
        ok = err < 2e-2 and bool(mx.array_equal(many, one).item())
    except Exception as exc:  # noqa: BLE001
        logger.warning("NAX 192/128 attention key-range dispatches disabled (%s)", exc)
        return False
    if not ok:
        logger.warning("NAX 192/128 attention key-range dispatches disabled: check failed")
    return ok


@lru_cache(maxsize=1)
def available() -> bool:
    try:
        from ..jangh.kernels import nax_available

        return bool(nax_available()) and all((_mlx_kernel_dir() / rel).is_file() for rel in _MLX_TILE_HEADERS)
    except Exception:  # noqa: BLE001
        return False


def nax_attention_bdv(queries, keys, values, *, scale: float, mask=None, sinks: Optional[mx.array] = None,
                      window: int = 0) -> Optional[mx.array]:
    """q [B, H, L, 192], k [B, Hk, T, 192], v [B, Hk, T, 128] (bf16/fp16, contiguous head dim, any other strides);
    mask None, "causal" (bottom-right) or a bool array broadcastable to [B, H, L, T]; window > 0 adds the causal
    sliding band (row p sees keys p - window < j <= p, bottom-right positions) on top of any mask; sinks [H] or None.
    Returns [B, H, L, 128] in q's dtype, or None when this kernel does not apply."""
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return None
    B, H, L, qk_dim = queries.shape
    _, Hk, T, k_dim = keys.shape
    v_dim = values.shape[-1]
    if ((qk_dim, v_dim) not in SUPPORTED_HEAD_DIMS or k_dim != qk_dim or queries.dtype not in _DTYPE_NAMES
            or keys.dtype != queries.dtype or values.dtype != queries.dtype or keys.shape[0] != B
            or tuple(values.shape[:3]) != (B, Hk, T) or Hk == 0 or H % Hk or L <= 8 or T < L or window < 0):
        return None
    if isinstance(mask, str):
        if mask != "causal":
            return None
    elif mask is not None:
        if not isinstance(mask, mx.array) or mask.dtype != mx.bool_ or not 1 <= mask.ndim <= 4:
            return None
        try:
            if mx.broadcast_shapes(mask.shape, (B, H, L, T)) != (B, H, L, T):
                return None
        except ValueError:
            return None
    if sinks is not None and (sinks.ndim != 1 or sinks.shape[0] != H):
        return None
    if not available() or not self_check_passed():
        return None
    return _run(queries, keys, values, scale, mask, sinks, int(window))
