"""JANGTQ v2 format: Gaussian Lloyd-Max codebooks, MLX-compatible LSB bitstream packing,
Walsh-Hadamard rotation with explicit signs, per-row solved scales.

Invariants (tests/test_format.py):
  * pack_bitstream(q, b) is byte-identical to MLX affine packing for b in {2,3,4,6,8}
  * unpack(pack(q)) == q
  * rotate/unrotate are exact inverses (orthonormal)
"""
from __future__ import annotations

import math
import numpy as np
import mlx.core as mx

SUPPORTED_BITS = (2, 3, 4, 6, 8)


# ---------------------------------------------------------------- codebooks
def _phi(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _Phi(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def lloyd_max_gaussian(bits: int, iters: int = 2000) -> np.ndarray:
    """MSE-optimal scalar quantizer levels for N(0,1), sorted ascending, float64."""
    n = 1 << bits
    # init: uniform quantiles
    levels = np.array([-_inv_Phi((i + 0.5) / n) for i in range(n)])[::-1].copy()
    for _ in range(iters):
        b = np.concatenate([[-np.inf], 0.5 * (levels[1:] + levels[:-1]), [np.inf]])
        pa, pb = _phi(b[:-1]), _phi(b[1:])
        Pa, Pb = _Phi(b[:-1]), _Phi(b[1:])
        new = (pa - pb) / np.maximum(Pb - Pa, 1e-300)
        if np.max(np.abs(new - levels)) < 1e-12:
            levels = new
            break
        levels = new
    levels = 0.5 * (levels - levels[::-1])  # enforce exact symmetry
    return levels


def _inv_Phi(p):
    # Acklam-style via bisection (only used for init)
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# v2 codebook family: odd cubic in the centered code index, levels(q) = u * (alpha + beta * u^2), u = q - (2^b - 1)/2.
# (alpha, beta) minimize N(0,1) MSE. 2-bit is EXACTLY Lloyd-Max; 3-bit +0.2% MSE; 4-bit +2% MSE (measured 2026-09-25).
# Kernels evaluate it arithmetically (no table): the decode stays ALU-cheap at every width.
from .contract import CUBIC_PARAMS as _RUNTIME_CUBIC_PARAMS

CUBIC_PARAMS = dict(_RUNTIME_CUBIC_PARAMS)

_CB_CACHE: dict[int, np.ndarray] = {}


def cubic_params(bits: int) -> tuple[float, float]:
    if bits in CUBIC_PARAMS:
        return CUBIC_PARAMS[bits]
    # 6/8-bit: uniform (beta=0) with the MSE-optimal step for N(0,1), fitted once
    n = 1 << bits
    u = np.arange(n) - (n - 1) / 2
    best = min((_mse_levels(u * s), s) for s in np.linspace(0.2 / n * 16, 8.0 / n, 400))
    CUBIC_PARAMS[bits] = (float(best[1]), 0.0)
    return CUBIC_PARAMS[bits]


def codebook(bits: int) -> np.ndarray:
    """Levels in code order (ascending) for the v2 cubic family."""
    if bits not in _CB_CACHE:
        a, b = cubic_params(bits)
        n = 1 << bits
        u = np.arange(n, dtype=np.float64) - (n - 1) / 2
        _CB_CACHE[bits] = (u * (a + b * u * u)).astype(np.float32)
    return _CB_CACHE[bits]


def _mse_levels(lv) -> float:
    lv = np.sort(np.asarray(lv, np.float64))
    b = np.concatenate([[-np.inf], 0.5 * (lv[1:] + lv[:-1]), [np.inf]])
    x = np.linspace(-12, 12, 200001)
    idx = np.searchsorted(b, x) - 1
    return float(np.trapezoid((x - lv[idx]) ** 2 * _phi(x), x))


def gaussian_mse(bits: int) -> float:
    """Expected MSE of the v2 codebook on N(0,1) (2-bit 0.11748 = Lloyd-Max; 3-bit 0.03462; 4-bit 0.00969)."""
    return _mse_levels(codebook(bits))


# ---------------------------------------------------------------- packing
def pack_bitstream(q: mx.array, bits: int) -> mx.array:
    """q: uint (..., K) values < 2**bits  ->  uint32 (..., K*bits/32), continuous LSB-first bitstream per row.
    Requires K % 32 == 0 (every 32 values occupy exactly `bits` words)."""
    assert bits in SUPPORTED_BITS, bits
    *lead, K = q.shape
    assert K % 32 == 0, f"K={K} must be a multiple of 32"
    g = q.astype(mx.uint32).reshape(*lead, K // 32, 32)
    words = []
    for w in range(bits):
        acc = mx.zeros(g.shape[:-1], dtype=mx.uint32)
        for j in range(32):
            start = j * bits
            end = start + bits
            if end <= w * 32 or start >= (w + 1) * 32:
                continue
            off = start - w * 32
            v = g[..., j]
            if off >= 0:
                acc = acc | (v << off)
            else:
                acc = acc | (v >> (-off))
        words.append(acc)
    out = mx.stack(words, axis=-1)  # (..., K/32, bits)
    return out.reshape(*lead, K * bits // 32)


def unpack_bitstream(p: mx.array, bits: int, K: int) -> mx.array:
    *lead, W = p.shape
    assert W == K * bits // 32
    g = p.reshape(*lead, K // 32, bits)
    mask = (1 << bits) - 1
    vals = []
    for j in range(32):
        start = j * bits
        w, off = divmod(start, 32)
        v = g[..., w] >> off
        if off + bits > 32:
            v = v | (g[..., w + 1] << (32 - off))
        vals.append(v & mask)
    return mx.stack(vals, axis=-1).reshape(*lead, K).astype(mx.uint8)


# ---------------------------------------------------------------- rotation
def make_signs(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=dim)


def hadamard_ok(dim: int) -> bool:
    for m in (1, 12, 20, 28):
        if dim % m == 0 and (dim // m) & (dim // m - 1) == 0:
            return True
    return False


def rotate_in(W: mx.array, signs: mx.array) -> mx.array:
    """W (..., K) -> W @ R^T where R = H diag(signs) (orthonormal). Computed in float32."""
    return mx.hadamard_transform(W.astype(mx.float32) * signs.astype(mx.float32))


def rotate_act(x: mx.array, signs: mx.array) -> mx.array:
    """x (..., K) -> R x. Same operation as rotate_in (H symmetric): H (s * x)."""
    return mx.hadamard_transform(x.astype(mx.float32) * signs.astype(mx.float32))


def unrotate_in(Wr: mx.array, signs: mx.array) -> mx.array:
    return mx.hadamard_transform(Wr.astype(mx.float32)) * signs.astype(mx.float32)


# ---------------------------------------------------------------- encode
def nearest_level(u: mx.array, cb: mx.array) -> mx.array:
    """u float (..., K) in codebook units -> uint8 index of nearest level (cb sorted)."""
    bnd = (cb[1:] + cb[:-1]) * 0.5
    q = mx.zeros(u.shape, dtype=mx.uint8)
    for i in range(bnd.shape[0]):
        q = q + (u > bnd[i]).astype(mx.uint8)
    return q


def encode_rows(Wr: mx.array, bits: int, col_weight: mx.array | None = None, iters: int = 4):
    """Quantize rotated rows Wr (N, K) float32 -> (q uint8 (N,K), scale float32 (N,)).
    Alternates nearest-level assignment and the (weighted) least-squares scale.
    col_weight: optional (K,) or (N,K) nonnegative weights for the LS scale (rotated-basis importance)."""
    cb = mx.array(codebook(bits))
    s = mx.sqrt(mx.mean(Wr * Wr, axis=-1, keepdims=True))  # N(0, s^2) assumption
    s = mx.maximum(s, 1e-12)
    wgt = 1.0 if col_weight is None else col_weight
    for _ in range(iters):
        q = nearest_level(Wr / s, cb)
        c = cb[q]
        num = mx.sum(wgt * Wr * c, axis=-1, keepdims=True)
        den = mx.maximum(mx.sum(wgt * c * c, axis=-1, keepdims=True), 1e-20)
        s = mx.maximum(num / den, 1e-12)
    q = nearest_level(Wr / s, cb)
    return q, s.squeeze(-1)


def dequant_rows(q: mx.array, scale: mx.array, bits: int) -> mx.array:
    cb = mx.array(codebook(bits))
    return cb[q] * scale[..., None].astype(mx.float32)


# ---------------------------------------------------------------- v2 blockwise rotation (hadamard32)
def h32(x: mx.array) -> mx.array:
    """Blockwise normalized Walsh-Hadamard over the LAST axis in 32-wide blocks (self-inverse), float32."""
    shp = x.shape
    return mx.hadamard_transform(x.astype(mx.float32).reshape(*shp[:-1], shp[-1] // 32, 32)).reshape(shp)


def h32_both(H: mx.array) -> mx.array:
    """R H R^T for a (..., K, K) matrix with R = blockwise H32 (symmetric): rotate rows and columns."""
    return mx.swapaxes(h32(mx.swapaxes(h32(H), -1, -2)), -1, -2)
