"""DSA indexer pool scores on the M5 matrix units (vmlx_engine/metal/dsa_index_scores_mpp.py).

out[s, p] = sum_h w[s, h] * relu(scale * q[s, h] . k[p]) from bf16 q / k with fp32 accumulation. For bf16-valued
operands the products are exact in fp32, so the result must match an fp64 reference of the same bf16 values to fp32
accumulation accuracy, keep clamped row / column tails right, select the same top-512 pools as the fp32 kernel, and be
reachable from dsa_pool_index_scores only when the caller states bf16 operands.
"""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.metal import dsa_index_scores_mpp as M  # noqa: E402
from vmlx_engine.metal.dsa_index_scores import dsa_pool_index_scores  # noqa: E402

pytestmark = [
    pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal"),
    pytest.mark.skipif(not (mx.metal.is_available() and M.available()), reason="needs Metal 4 tensor ops (M5)"),
]

H, D = 32, 128


def _inputs(S, P, seed=0, dtype=mx.bfloat16):
    mx.random.seed(seed)
    q = mx.random.normal((1, S, H, D)).astype(dtype)
    k = mx.random.normal((1, P, D)).astype(dtype)
    w = (mx.random.normal((1, S, H)) * 0.2).astype(mx.bfloat16).astype(mx.float32)
    mx.eval(q, k, w)
    return q, k, w


def _ref64(q, k, w, scale):
    qn, kn, wn = (np.array(x.astype(mx.float32)).astype(np.float64) for x in (q, k, w))
    s = np.einsum("shd,pd->shp", qn[0], kn[0]) * scale
    return np.einsum("sh,shp->sp", wn[0], np.maximum(s, 0.0))


def _max_rel(out, ref):
    return float(np.abs(np.array(out)[0].astype(np.float64) - ref).max() / np.abs(ref).max())


@pytest.mark.parametrize("S,P", [
    (16, 256),        # exactly one tile
    (64, 1024),       # whole tiles
    (100, 999),       # row and column tails (clamped, overlapping last tiles)
    (17, 257),        # one past a tile in both axes
    (300, 4097),
    (5, 100),         # below one tile: zero-padded rows and keys
    (1, 3000),
    (2048, 2500),     # a full prefill chunk
])
def test_matches_fp64_reference_of_the_bf16_values(S, P):
    q, k, w = _inputs(S, P, seed=S * 7 + P)
    scale = 128 ** -0.5
    out = M.dsa_pool_index_scores_mpp(q, k, w, scale)
    assert out is not None and out.shape == (1, S, P) and out.dtype == mx.float32
    err = _max_rel(out, _ref64(q, k, w, scale))
    print(f"S={S} P={P} max rel err {err:.3e}")
    assert err <= 2e-6


def test_fp16_operands_and_bf16_head_weights():
    q, k, w = _inputs(100, 999, seed=11, dtype=mx.float16)
    out = M.dsa_pool_index_scores_mpp(q, k, w.astype(mx.bfloat16), 1.0)
    assert out is not None
    assert _max_rel(out, _ref64(q, k, w, 1.0)) <= 2e-6


def test_every_tail_cell_is_written():
    """Clamped last tiles overlap their neighbours; every output cell (the output buffer is uninitialized) must be
    written with its own value, including the last row and the last column."""
    q, k, w = _inputs(33, 513, seed=5)
    out = np.array(M.dsa_pool_index_scores_mpp(q, k, w, 0.5))[0]
    ref = _ref64(q, k, w, 0.5)
    assert np.isfinite(out).all()
    assert np.abs(out - ref).max(axis=1).max() <= 2e-6 * np.abs(ref).max()
    assert np.abs(out[-1] - ref[-1]).max() <= 2e-6 * np.abs(ref).max()
    assert np.abs(out[:, -1] - ref[:, -1]).max() <= 2e-6 * np.abs(ref).max()


@pytest.mark.parametrize("S,P", [(256, 8192), (2048, 6000)])
def test_top512_matches_the_fp32_kernel(S, P):
    q, k, w = _inputs(S, P, seed=S + P)
    fast = M.dsa_pool_index_scores_mpp(q, k, w, 1.0)
    f32 = dsa_pool_index_scores(q.astype(mx.float32), k.astype(mx.float32), w, 1.0)
    assert f32 is not None
    top = lambda x: np.argpartition(-np.array(x)[0], 511, axis=-1)[:, :512]  # noqa: E731
    a, b = top(fast), top(f32)
    overlap = np.mean([len(np.intersect1d(a[i], b[i])) / 512 for i in range(S)])
    print(f"S={S} P={P} top-512 overlap {overlap:.6f}, max rel diff vs fp32 kernel "
          f"{float(mx.abs(fast - f32).max() / mx.abs(f32).max()):.3e}")
    assert overlap >= 0.999


def test_admission():
    q, k, w = _inputs(64, 512)
    assert M.dsa_pool_index_scores_mpp(q.astype(mx.float32), k.astype(mx.float32), w, 1.0) is None  # fp32 operands
    assert M.dsa_pool_index_scores_mpp(q, k.astype(mx.float16), w, 1.0) is None                  # mixed dtypes
    assert M.dsa_pool_index_scores_mpp(q, k, w, 0.0) is None                                      # scale <= 0
    assert M.dsa_pool_index_scores_mpp(q, k, w, 1.0, enabled=False) is None
    assert M.dsa_pool_index_scores_mpp(q[..., :120], k[..., :120], w, 1.0) is None               # D % 32
    assert M.dsa_pool_index_scores_mpp(mx.concatenate([q, q]), k, w, 1.0) is None                # batch 2


def test_pool_index_scores_prefers_mpp_only_for_bf16_operands(monkeypatch):
    q, k, w = _inputs(100, 999, seed=3)
    qf, kf = q.astype(mx.float32), k.astype(mx.float32)
    mpp = M.dsa_pool_index_scores_mpp(q, k, w, 1.0)
    fp32 = dsa_pool_index_scores(qf, kf, w, 1.0)
    # default (GLM-5.3): the fp32 kernel, bit for bit
    assert mx.array_equal(dsa_pool_index_scores(qf, kf, w, 1.0), fp32).item()
    # bf16 operands, as fp32 or bf16 arrays: the matrix-unit kernel
    assert mx.array_equal(dsa_pool_index_scores(qf, kf, w, 1.0, bf16_operands=True), mpp).item()
    assert mx.array_equal(dsa_pool_index_scores(q, k, w, 1.0, bf16_operands=True), mpp).item()
    # VMLX_DSA_MPP_INDEX=0: the fp32 kernel (bf16 arrays upcast exactly)
    monkeypatch.setenv("VMLX_DSA_MPP_INDEX", "0")
    assert M.dsa_pool_index_scores_mpp(q, k, w, 1.0) is None
    assert mx.array_equal(dsa_pool_index_scores(q, k, w, 1.0, bf16_operands=True), fp32).item()
    assert mx.array_equal(dsa_pool_index_scores(qf, kf, w, 1.0, bf16_operands=True), fp32).item()
    # disabled: the caller's stock path
    assert dsa_pool_index_scores(q, k, w, 1.0, enabled=False, bf16_operands=True) is None


def test_fp32_operands_rounded_to_bf16_stay_within_bf16_input_tolerance():
    """Arbitrary fp32 operands rounded to bf16 (2^-9 relative per element): the scores stay within bf16-input error of
    the fp32 function and the top-512 sets mostly agree (near-ties may swap)."""
    mx.random.seed(21)
    S, P = 256, 8192
    q = mx.random.normal((1, S, H, D))
    k = mx.random.normal((1, P, D))
    w = mx.random.normal((1, S, H)) * 0.2
    out = M.dsa_pool_index_scores_mpp(q.astype(mx.bfloat16), k.astype(mx.bfloat16), w, 1.0)
    ref = _ref64(q, k, w, 1.0)
    err = _max_rel(out, ref)
    top = lambda x: np.argpartition(-x, 511, axis=-1)[:, :512]  # noqa: E731
    a, b = top(np.array(out)[0]), top(ref)
    overlap = np.mean([len(np.intersect1d(a[i], b[i])) / 512 for i in range(S)])
    print(f"fp32 operands via bf16: max rel err {err:.3e}, top-512 overlap {overlap:.4f}")
    assert err <= 1e-2
    assert overlap >= 0.95
