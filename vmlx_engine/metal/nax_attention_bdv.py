# SPDX-License-Identifier: Apache-2.0
"""Fused tensor-unit (NAX, M5) prefill attention with a narrower value head: q/k 192, v 128 (MiMo-V2 family).

MLX 0.32.3 fuses prefill attention only for equal q/k and v head dims in {64, 80, 96, 128, 256}; for 192/128 it
materializes the [H, L, T] score matrix. MiMo's full-attention layers worked around that by zero-padding q/k/v to 256
(a third of the multiply-adds on zero columns plus a q/k/v copy per layer, models/mimo_v26_janght._full_attention),
and its sliding-window layers (window 128, explicit array mask) still materialized scores (~570 MB per layer for a
2048-row chunk). This module JIT-compiles MLX's own NAX attention kernel with the value head dim as a separate
template parameter (kernel: nax_attention_bdv.metal; ported from oMLX omlx/utils/nax_attention.py, Apache-2.0) plus
a sliding-window band (mlx-serve, Apache-2.0): a window tile reads only the key blocks its rows can see.

Exactness: the kernel is MLX's (same tiles, fp32 online softmax, same MPP matmuls); long key ranges run as
consecutive key-range passes handing the fp32 row state on (bit-identical to one dispatch) with the head-dim split
(a summation-order change of Q @ K.T: last-bit differences). Fail-closed: only bf16/fp16 (192, 128) with L > 8 on NAX
GPUs; a self-check against fp32 runs once per process (causal, window+sinks, bool mask); any unsupported input returns
None and the caller keeps its existing route.
"""

from __future__ import annotations

import logging
import struct
from functools import lru_cache
from pathlib import Path
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

SUPPORTED_HEAD_DIMS = frozenset({(192, 128)})

_BQ = 64
_BK = 32
_WM = 4
_THREADS = 32 * _WM
# Key-range passes (oMLX's measured defaults): keys per dispatch, and the grid size from which passes pay.
_PASS_KEYS = 8192
_PASS_MIN_GROUPS = 512
# Head-dim split over simdgroup pairs: 1 = only for calls that run in key-range passes (3-4 % faster there).
_DSPLIT = 1

_METAL_TYPES = {mx.bfloat16: "bfloat16_t", mx.float16: "half"}
_HERE = Path(__file__).resolve().parent

_SOURCE = r"""
  threadgroup float xchg[{WN} == 2 ? {WM} * 2 * 16 * 32 : 1];
  omlx_nax::attention_nax_bdv<
      {T}, {BQ}, {BK}, {BD}, {BDV}, {WM}, {WN},
      {ALIGN_Q}, {ALIGN_K}, {HAS_MASK}, {DO_CAUSAL}, {WINDOW}, {HAS_SINKS},
      {FIRST}, {LAST}, bool, float>(
      q, k, v, out,
      reinterpret_cast<const device omlx_nax::AttnParams*>(params),
      q_strides, k_strides, v_strides, mask_strides,
      mask, sinks, state, state_out, xchg,
      simdgroup_index_in_threadgroup,
      thread_index_in_simdgroup,
      threadgroup_position_in_grid);
"""


@lru_cache(maxsize=1)
def _header() -> str:
    return (_HERE / "nax_tiles.metal").read_text() + (_HERE / "nax_attention_bdv.metal").read_text()


def _flag(value: bool) -> str:
    return "true" if value else "false"


@lru_cache(maxsize=None)
def _kernel(dtype, bd, bdv, align_q, align_k, has_mask, do_causal, window, has_sinks, first, last, wn):
    wm = _WM // wn
    subs = {"{T}": _METAL_TYPES[dtype], "{BQ}": str(16 * wm), "{BK}": str(_BK), "{BD}": str(bd), "{BDV}": str(bdv),
            "{WM}": str(wm), "{WN}": str(wn), "{ALIGN_Q}": _flag(align_q), "{ALIGN_K}": _flag(align_k),
            "{HAS_MASK}": _flag(has_mask), "{DO_CAUSAL}": _flag(do_causal), "{WINDOW}": str(int(window)),
            "{HAS_SINKS}": _flag(has_sinks), "{FIRST}": _flag(first), "{LAST}": _flag(last)}
    source = _SOURCE
    for k, v in subs.items():
        source = source.replace(k, v)
    tag = "".join("1" if f else "0" for f in (align_q, align_k, has_mask, do_causal, has_sinks, first, last))
    return mx.fast.metal_kernel(
        name=f"vmlx_nax_attention_bd{bd}_bdv{bdv}_wn{wn}_w{int(window)}_{tag}",
        input_names=["q", "k", "v", "mask", "sinks", "state", "params"],
        output_names=["out", "state_out"],
        header=_header(),
        source=source,
        ensure_row_contiguous=False,
    )


def _pass_edges(n_groups: int, nk: int, window: int) -> list[int]:
    """Key-block boundaries of the dispatches: one for a window (a tile reads only its band) or a small grid."""
    if window > 0 or n_groups < _PASS_MIN_GROUPS:
        return [0, nk]
    n_pass = max(1, min(nk, round(nk * _BK / _PASS_KEYS)))
    return [(i * nk) // n_pass for i in range(n_pass + 1)]


def _run_edges(q, k, v, scale, mask, sinks, window, edges, wn) -> mx.array:
    B, H, qL, D = q.shape
    kL, DV = int(k.shape[2]), int(v.shape[3])
    do_causal = isinstance(mask, str) or window > 0
    has_mask = isinstance(mask, mx.array)
    NK = (kL + _BK - 1) // _BK
    NK_aligned = kL // _BK
    n_pass = len(edges) - 1
    bq = 16 * (_WM // wn)
    NQ = (qL + bq - 1) // bq
    NQ_aligned = qL // bq
    has_sinks = sinks is not None
    mask = mx.broadcast_to(mask, (B, H, qL, kL)) if has_mask else mx.zeros((1,), dtype=mx.bool_)
    sinks = mx.contiguous(sinks.astype(q.dtype)) if has_sinks else mx.zeros((1,), dtype=q.dtype)
    state_size = B * H * NQ * bq * (DV + 2)
    state = mx.zeros((1,), dtype=mx.float32)
    out = None
    for i in range(n_pass):
        first, last = i == 0, i == n_pass - 1
        params = struct.pack("<6if9i", B, H, D, qL, kL, H // int(k.shape[1]), float(scale), NQ, NK, NQ_aligned,
                             NK_aligned, qL - NQ_aligned * bq, kL - NK_aligned * _BK, kL - qL, edges[i], edges[i + 1])
        params = mx.array(memoryview(params), dtype=mx.uint8)
        kernel = _kernel(q.dtype, D, DV, qL % bq == 0, kL % _BK == 0, has_mask, do_causal, window, has_sinks,
                         first, last, wn)
        out, state = kernel(
            inputs=[q, k, v, mask, sinks, state, params],
            grid=(NQ * _THREADS, H, B),
            threadgroup=(_THREADS, 1, 1),
            output_shapes=[(B, qL, H, DV) if last else (1,), (1,) if last else (state_size,)],
            output_dtypes=[q.dtype, mx.float32],
        )
    # [B, qL, H, DV] rows viewed as [B, H, qL, DV] (MLX's SDPA output layout: the caller's transpose+reshape is free).
    return out.transpose(0, 2, 1, 3)


def _run(q, k, v, scale, mask, sinks, window) -> mx.array:
    B, H, qL, _ = q.shape
    NK = (int(k.shape[2]) + _BK - 1) // _BK
    edges = _pass_edges(B * H * ((qL + _BQ - 1) // _BQ), NK, window)
    wn = 2 if (_DSPLIT == 2 or (_DSPLIT == 1 and len(edges) > 2)) else 1
    if len(edges) > 2 and not _passes_check_passed(wn):
        edges, wn = [0, NK], 1
    return _run_edges(q, k, v, scale, mask, sinks, window, edges, wn)


def reference_attention(q, k, v, scale, mask=None, sinks=None, window=0):
    """fp32 attention with bottom-right causal alignment, optional window (i - j < window), bool mask and sinks."""
    B, H, qL, _ = q.shape
    Hk, kL = int(k.shape[1]), int(k.shape[2])
    g = H // Hk
    s = (q.astype(mx.float32).reshape(B, Hk, g, qL, -1) * scale) @ k.astype(mx.float32)[:, :, None].swapaxes(-1, -2)
    s = s.reshape(B, H, qL, kL)
    rows = mx.arange(qL)[:, None] + (kL - qL)
    cols = mx.arange(kL)[None]
    allowed = mx.ones((qL, kL), dtype=mx.bool_)
    if isinstance(mask, str) or window > 0:
        allowed = allowed & (rows >= cols)
    if window > 0:
        allowed = allowed & (rows - cols < window)
    if isinstance(mask, mx.array):
        allowed = allowed & mask
    s = mx.where(allowed, s, -mx.inf)
    if sinks is not None:
        sk = mx.broadcast_to(sinks.astype(mx.float32).reshape(1, H, 1, 1), (B, H, qL, 1))
        p = mx.softmax(mx.concatenate([s, sk], axis=-1), axis=-1)[..., :-1]
    else:
        p = mx.softmax(s, axis=-1)
    return (p.reshape(B, Hk, g, qL, kL) @ v.astype(mx.float32)[:, :, None]).reshape(B, H, qL, -1)


def _check_inputs(seed):
    kq, kk, kv, ks = mx.random.split(mx.random.key(seed), 4)
    q = (0.5 * mx.random.normal((1, 4, 100, 192), key=kq)).astype(mx.bfloat16)
    k = (0.5 * mx.random.normal((1, 2, 300, 192), key=kk)).astype(mx.bfloat16)
    v = (0.5 * mx.random.normal((1, 2, 300, 128), key=kv)).astype(mx.bfloat16)
    sinks = mx.random.normal((4,), key=ks).astype(mx.bfloat16)
    return q, k, v, sinks, 192 ** -0.5


@lru_cache(maxsize=1)
def _self_check_passed() -> bool:
    """Once per process: unaligned causal, window 37 + sinks, and a bool mask against fp32."""
    try:
        q, k, v, sinks, scale = _check_inputs(192128)
        bmask = (mx.arange(300)[None, :] % 3 != 1) | (mx.arange(100)[:, None] + 200 == mx.arange(300)[None, :])
        cases = [("causal", None, 0), ("causal", sinks, 37), (bmask, None, 0)]
        err = 0.0
        for mask, sk, window in cases:
            out = _run_edges(q, k, v, scale, mask, sk, window, [0, 10], 1)
            ref = reference_attention(q, k, v, scale, mask, sk, window)
            err = max(err, mx.abs(out.astype(mx.float32) - ref).max().item())
        ok = err < 2e-2
    except Exception as exc:  # noqa: BLE001 - any failure disables the route
        logger.warning("NAX 192/128 attention disabled: self-check failed (%s)", exc)
        return False
    if not ok:
        logger.warning("NAX 192/128 attention disabled: self-check error %.3g vs fp32", err)
        return False
    logger.info("NAX 192/128 attention enabled (self-check max error %.3g vs fp32)", err)
    return True


@lru_cache(maxsize=None)
def _passes_check_passed(wn: int) -> bool:
    """Several key-range dispatches must reproduce one dispatch bit for bit and match fp32."""
    try:
        q, k, v, _, scale = _check_inputs(192129)
        one = _run_edges(q, k, v, scale, "causal", None, 0, [0, 10], wn)
        many = _run_edges(q, k, v, scale, "causal", None, 0, [0, 3, 6, 10], wn)
        err = mx.abs(many.astype(mx.float32) - reference_attention(q, k, v, scale, "causal")).max().item()
        ok = err < 2e-2 and bool(mx.array_equal(many, one).item())
    except Exception as exc:  # noqa: BLE001
        logger.warning("NAX 192/128 attention key-range passes disabled (%s)", exc)
        return False
    if not ok:
        logger.warning("NAX 192/128 attention key-range passes disabled: check failed")
    return ok


@lru_cache(maxsize=1)
def available() -> bool:
    try:
        from ..jangh.kernels import nax_available

        return bool(nax_available())
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
    B, H, qL, qk_dim = queries.shape
    _, Hk, kL, k_dim = keys.shape
    v_dim = values.shape[-1]
    if ((qk_dim, v_dim) not in SUPPORTED_HEAD_DIMS or k_dim != qk_dim or queries.dtype not in _METAL_TYPES
            or keys.dtype != queries.dtype or values.dtype != queries.dtype or keys.shape[0] != B
            or tuple(values.shape[:3]) != (B, Hk, kL) or Hk == 0 or H % Hk or qL <= 8 or kL < qL or window < 0):
        return None
    if isinstance(mask, str):
        if mask != "causal":
            return None
    elif mask is not None:
        if not isinstance(mask, mx.array) or mask.dtype != mx.bool_ or not 1 <= mask.ndim <= 4:
            return None
        try:
            if mx.broadcast_shapes(mask.shape, (B, H, qL, kL)) != (B, H, qL, kL):
                return None
        except ValueError:
            return None
    if sinks is not None and (sinks.ndim != 1 or sinks.shape[0] != H):
        return None
    if not available() or not _self_check_passed():
        return None
    return _run(queries, keys, values, scale, mask, sinks, int(window))
