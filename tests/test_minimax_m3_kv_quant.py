# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 KV cache formats (models/minimax_m3/kv_quant.py, kernels/kv_quant.metal; levers VMLX_M3_KV_DTYPE /
VMLX_M3_INDEX_KV_DTYPE).

* Encoders / decoders: the Metal encoder == an independent numpy reference bit for bit (every format, head dims 64 /
  128 / 256, values spanning 2^-14 .. 2^9 plus zeros, ties and saturating inputs); every e4m3 / e2m1 code round trips;
  mxfp8 rows are byte for byte MLX's own MX encoding and never saturate; nvfp4 blocks decode like MLX's nvfp4, and the
  row exponent keeps small-magnitude rows (control: NVFP4 without a second-level scale flushes them to zero).
* Row independence: a row's bytes do not depend on the other rows encoded with it.
* Cache: append / trim / restore round trips at lengths that are not multiples of 128 / 256 / 2048 (lane invariants
  asserted directly: logical rows == offset, capacity >= offset, index lane in step); partial prefix hits (same
  prefix, different suffix) == cold prefill bit for bit; a perturbed byte in a restored lane changes the logits
  (control: the comparison bites).
* Fused decode / verify kernels: K2's packed rows == the unfused encoder on K2's own rows; K3 / K4 on packed lanes ==
  K3 / K4 on plain lanes holding the decoded rows; in-place appends never write under a held view; a lazy chain of
  steps across a lane growth == eager steps.
* Model level (tiny synthetic M3): DSpark verify rows == sequential decode steps bit for bit with every format, fused
  attention on and off; greedy DSpark == greedy AR past the dense threshold at the released block geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
if not mx.metal.is_available():  # pragma: no cover - the kernels are Metal only
    pytest.skip("Metal GPU required", allow_module_level=True)

from tests import test_minimax_m3_dspark as TD  # noqa: E402
from tests import test_minimax_m3_fused_attn as TF  # noqa: E402
from vmlx_engine.dspark import generate_ar, generate_dspark, load_drafter  # noqa: E402
from vmlx_engine.dspark.generate import ARFallback  # noqa: E402
from vmlx_engine.dspark.minimax_target import MiniMaxM3DSparkTarget  # noqa: E402
from vmlx_engine.models.minimax_m3 import fused_attn as F  # noqa: E402
from vmlx_engine.models.minimax_m3 import kv_quant as Q  # noqa: E402
from vmlx_engine.models.minimax_m3.cache import (  # noqa: E402
    BatchMiniMaxM3SparseCache,
    MiniMaxM3SparseCache,
    clone_minimax_m3_sparse,
    restore_minimax_m3_sparse,
)

PACKED = ("mxfp8", "nvfp4", "affine8", "affine4")
# (K / V format, index-key format) combinations the model-level tests run
COMBOS = [("mxfp8", "mxfp8"), ("nvfp4", "nvfp4"), ("affine8", "bf16"), ("affine4", "mxfp8"), ("bf16", "nvfp4"),
          ("mxfp8", "nvfp4")]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in (Q.KV_ENV, Q.INDEX_ENV, F.ENV, "VMLX_ROW_EXACT_VERIFY_QMV", "VMLX_DSPARK", "VMLX_DSPARK_MAX_DRAFT",
                 "VMLX_DSPARK_WIDTH", "VMLX_DSPARK_SAMPLED", "VMLX_DSPARK_PIPELINE", "VMLX_DSPARK_AR_FALLBACK",
                 "VMLX_MINIMAX_MSA", "VMLX_MINIMAX_MSA_SELECT"):
        monkeypatch.delenv(name, raising=False)


def _eq(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a, b).item())


def _rows(shape, seed: int, spread: float = 8.0) -> np.ndarray:
    """float32 rows whose magnitudes vary per row over 2^-spread .. 2^spread (rounded to bf16 later)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape).astype(np.float32)
    x *= np.exp2(rng.uniform(-spread, spread, shape[:-1] + (1,))).astype(np.float32)
    return x


def _host(x):
    """A copy of ``x`` that went through host memory (what a disk tier hands back): a fresh buffer, same bytes."""
    if x.dtype == mx.bfloat16:
        return mx.array(np.array(x.view(mx.uint16))).view(mx.bfloat16)
    return mx.array(np.array(x))


def _bf16(x: np.ndarray) -> tuple:
    a = mx.array(x).astype(mx.bfloat16)
    return a, np.array(a.astype(mx.float32))


# ------------------------------------------------------------------------------------------- encoders / decoders
@pytest.mark.parametrize("fmt", PACKED)
@pytest.mark.parametrize("d", [64, 128, 256])
def test_encoder_and_decoder_match_the_reference(fmt, d):
    f = Q.get(fmt)
    x = _rows((2, 4, 45, d), seed=d + f.code, spread=7.0)
    x[0, 0, 0] = 0.0                                                   # an all-zero row
    x[0, 0, 1, :8] = [448.0 * 3, -448.0 * 3, 1e-4, -0.0, 0.0, 2.0 ** -14, 6.0, -5.0]
    x[0, 1, 2] = 1.0                                                   # constant row (affine scale 0)
    x[1, 2, 3, :] = np.linspace(-3, 3, d, dtype=np.float32)            # exact e4m3 / e2m1 ties
    xb, xf = _bf16(x)
    packed = Q.quantize(xb, f)
    ref = Q.ref_quantize(xf, f)
    got = np.array(packed)
    assert got.shape == ref.shape == x.shape[:-1] + (f.row_bytes(d),)
    assert np.array_equal(got, ref), np.argwhere(got != ref)[:5].tolist()
    dec = np.array(Q.dequantize(packed, x.shape[2], f, mx.bfloat16).astype(mx.float32))
    assert np.array_equal(dec, Q.ref_dequantize(ref, f, d))
    # error budget per format (relative RMS per row, rows with any magnitude)
    nz = np.abs(xf).max(-1) > 0
    rel = np.sqrt(((dec - xf) ** 2).sum(-1) / np.maximum((xf ** 2).sum(-1), 1e-30))[nz]
    assert np.median(rel) < {"mxfp8": 0.035, "nvfp4": 0.12, "affine8": 0.008, "affine4": 0.12}[fmt]


def test_every_code_round_trips():
    codes = np.arange(256, dtype=np.uint8)
    finite = ((codes & 0x7F) != 0x7F) & (codes != 0x80)                # 0x7f / 0xff NaN, 0x80 = -0
    vals = Q.ref_fp8(codes)
    assert np.array_equal(Q.ref_e4m3(vals)[finite], codes[finite])
    nib = np.arange(16, dtype=np.uint8)
    assert np.array_equal(Q.ref_e2m1(Q.ref_fp4(nib))[nib != 8], nib[nib != 8])
    # The Metal encoder on every finite e4m3 value and every midpoint between neighbours (ties to even): rows whose
    # every 32-dim group holds 448 keep the group scale at 2^0, so the codes are e4m3(value) directly.
    pos = np.sort(vals[(codes < 0x7F)])
    probe = np.concatenate([pos, (pos[:-1].astype(np.float64) + pos[1:]) / 2]).astype(np.float32)
    probe = np.concatenate([probe, -probe])
    per_row = 4 * 31
    n_rows = -(-probe.size // per_row)
    rows = np.zeros((n_rows, 4, 32), dtype=np.float32)
    rows[:, :, 0] = 448.0
    flat = np.zeros(n_rows * per_row, dtype=np.float32)
    flat[:probe.size] = probe
    rows[:, :, 1:] = flat.reshape(n_rows, 4, 31)
    x = rows.reshape(1, 1, n_rows, 128)
    packed = np.array(Q.quantize(mx.array(x), Q.MXFP8))
    assert np.array_equal(packed[..., :128], Q.ref_e4m3(x))
    assert (packed[..., 128:] == 127).all()                            # every group scale is 2^0
    dec = np.array(Q.dequantize(mx.array(packed), n_rows, Q.MXFP8, mx.float32))
    assert np.array_equal(dec, Q.ref_fp8(Q.ref_e4m3(x)))


def test_mxfp8_rows_are_mlx_mx_encodings_and_never_saturate():
    """Our mxfp8 rows are byte for byte MLX's own mx.quantize(mode="mxfp8") codes + e8m0 scales (standard MX:
    MLX's dequantize decodes them identically), and no element is clipped: the group exponent is the smallest with
    amax * 2^-n <= 448, so the worst error is half an e4m3 step at the top of the group."""
    x = _rows((1, 4, 300, 128), seed=3, spread=9.0)
    x[0, 0, :, 5] = 1.3 * 448.0 * np.exp2(np.arange(300) % 7 - 3)     # group maxima just above 448 * 2^n
    xb, xf = _bf16(x)
    packed = Q.quantize(xb, Q.MXFP8)
    wq, ws = mx.quantize(xb.reshape(-1, 128), mode="mxfp8", group_size=32, bits=8)
    assert _eq(mx.contiguous(packed[..., :128]).reshape(-1, 128), wq.view(mx.uint8).reshape(-1, 128))
    assert _eq(mx.contiguous(packed[..., 128:]).reshape(-1, 4), ws)
    mlx_dec = mx.dequantize(wq, ws, mode="mxfp8", group_size=32, bits=8, dtype=mx.float32).reshape(1, 4, 300, 128)
    ours = Q.dequantize(packed, 300, Q.MXFP8, mx.float32)
    assert _eq(mlx_dec, ours)
    g = np.abs(xf).reshape(1, 4, 300, 4, 32).max(-1)
    err = np.abs(np.array(ours) - xf).reshape(1, 4, 300, 4, 32).max(-1)
    assert (err <= g / 16 + 1e-30).all()


def test_nvfp4_blocks_are_standard_nvfp4_blocks_and_the_row_exponent_keeps_small_rows():
    """Strip the row exponent and our nvfp4 codes + e4m3 block scales decode like MLX's nvfp4 dequantize; times
    2^k they are our values. Control: NVFP4 without a second-level scale (MLX's nvfp4 with no global scale) flushes
    rows of small magnitude (here 2^-12 .. 2^-9) to zero; the row exponent keeps them at the format's usual error."""
    x = _rows((1, 4, 77, 128), seed=4)
    xb, _ = _bf16(x)
    packed = Q.quantize(xb, Q.NVFP4)
    p = packed.reshape(-1, 76)
    k = np.array(p[:, 72]).astype(np.int32) - 127
    blocks = mx.dequantize(mx.contiguous(p[:, :64]).view(mx.uint32), mx.contiguous(p[:, 64:72]), mode="nvfp4",
                           group_size=16, bits=4, dtype=mx.float32)
    ours = Q.dequantize(packed, 77, Q.NVFP4, mx.float32).reshape(-1, 128)
    assert _eq(blocks * mx.array(np.exp2(k).astype(np.float32))[:, None], ours)
    rng = np.random.default_rng(8)
    small = (rng.standard_normal((64, 128)) * np.exp2(rng.uniform(-12, -9, (64, 1)))).astype(np.float32)
    sb, sf = _bf16(small)
    wq, ws = mx.quantize(sb, mode="nvfp4", group_size=16, bits=4)
    plain = np.array(mx.dequantize(wq, ws, mode="nvfp4", group_size=16, bits=4, dtype=mx.float32))
    mine = np.array(Q.dequantize(Q.quantize(sb.reshape(1, 1, 64, 128), Q.NVFP4), 64, Q.NVFP4, mx.float32))[0, 0]

    def rel(v):
        return np.sqrt(((v - sf) ** 2).sum(-1) / (sf ** 2).sum(-1))

    assert np.median(rel(plain)) > 0.5                                 # mostly flushed to zero
    assert np.median(rel(mine)) < 0.12


@pytest.mark.parametrize("fmt", PACKED)
def test_a_rows_bytes_do_not_depend_on_the_other_rows(fmt):
    f = Q.get(fmt)
    xb, _ = _bf16(_rows((1, 4, 37, 128), seed=9))
    whole = Q.quantize(xb, f)
    for t in (0, 1, 17, 36):
        assert _eq(whole[:, :, t:t + 1], Q.quantize(xb[:, :, t:t + 1], f))
    # a transposed (strided) input encodes the same rows
    xt = mx.array(np.array(xb.astype(mx.float32)).transpose(0, 2, 1, 3)).astype(mx.bfloat16).transpose(0, 2, 1, 3)
    assert _eq(Q.quantize(xt, f), whole)


def test_levers_and_namespace(monkeypatch):
    assert Q.configured() == (Q.BF16, Q.BF16) and Q.cache_identity() == ""
    monkeypatch.setenv(Q.KV_ENV, "fp8")
    monkeypatch.setenv(Q.INDEX_ENV, "nvfp4")
    assert Q.configured() == (Q.MXFP8, Q.NVFP4)
    ident = Q.cache_identity()
    assert "minimax_m3_kv=mxfp8" in ident and "minimax_m3_index_kv=nvfp4" in ident
    from vmlx_engine import prefix_cache as PC

    assert ident in PC._resolve_runtime_cache_fingerprint()
    monkeypatch.setenv(Q.INDEX_ENV, "affine4")                         # not an index-lane format: warn, keep bf16
    assert Q.configured()[1] is Q.BF16
    monkeypatch.delenv(Q.KV_ENV)
    monkeypatch.delenv(Q.INDEX_ENV)
    assert "minimax_m3_kv" not in PC._resolve_runtime_cache_fingerprint()


# ------------------------------------------------------------------------------------------- cache
def _invariants(c: MiniMaxM3SparseCache):
    keys, values, idx = c.fused_lanes()
    assert int(keys.shape[2]) >= int(c.offset) and int(values.shape[2]) >= int(c.offset)
    assert int(idx.shape[2]) >= int(c.offset) and int(c._idx_offset) == int(c.offset)
    k, v, i = c.state
    assert int(k.shape[2]) == int(v.shape[2]) == int(i.shape[2]) == int(c.offset)


def _append(c, rows_k, rows_v, rows_i, a, b):
    c.update_and_fetch(rows_k[:, :, a:b], rows_v[:, :, a:b])
    return c.update_index(rows_i[:, :, a:b])


@pytest.mark.parametrize("kf, idf", COMBOS + [("affine8", "nvfp4")])
def test_append_trim_restore_round_trip(kf, idf):
    """Chunks of 137 / 1 / 1 / 161 / 2047 rows, a trim of 7 and appends after it: the lanes hold exactly the
    encoder's bytes of the surviving rows; reads return their decoded values; state -> restore and clone give
    byte-identical lanes in the same formats; lane invariants hold at every step (lengths not multiples of 128 /
    256 / 2048)."""
    kfmt, ifmt = Q.get(kf), Q.get(idf)
    rk, _ = _bf16(_rows((1, 4, 2400, 128), 1, 3.0))
    rv, _ = _bf16(_rows((1, 4, 2400, 128), 2, 3.0))
    ri, _ = _bf16(_rows((1, 1, 2400, 128), 3, 3.0))
    c = MiniMaxM3SparseCache(kf, idf)
    cuts = [0, 137, 138, 139, 300, 2347]
    for a, b in zip(cuts, cuts[1:]):
        hist = _append(c, rk, rv, ri, a, b)
        _invariants(c)
        assert int(hist.shape[2]) == b
    assert c.trim(7) == 7
    _invariants(c)
    nk, _ = _bf16(_rows((1, 4, 53, 128), 11, 3.0))                     # new rows overwrite the trimmed tail
    nv, _ = _bf16(_rows((1, 4, 53, 128), 12, 3.0))
    ni, _ = _bf16(_rows((1, 1, 53, 128), 13, 3.0))
    _append(c, nk, nv, ni, 0, 53)
    _invariants(c)
    assert int(c.offset) == 2393
    want = [mx.concatenate([r[:, :, :2340], n], axis=2) for r, n in ((rk, nk), (rv, nv), (ri, ni))]
    k, v, i = c.state
    for lane, rows, fmt in ((k, want[0], kfmt), (v, want[1], kfmt), (i, want[2], ifmt)):
        if fmt.quantized:
            assert lane.dtype == mx.uint8 and _eq(lane, Q.quantize(rows, fmt))
        else:
            assert _eq(lane, rows)
    rk_, rv_ = c.read_kv()
    ri_ = c.read_index()
    for got, rows, fmt in ((rk_, want[0], kfmt), (rv_, want[1], kfmt), (ri_, want[2], ifmt)):
        expect = Q.dequantize(Q.quantize(rows, fmt), rows.shape[2], fmt) if fmt.quantized else rows
        assert _eq(got, expect)
    with pytest.MonkeyPatch.context() as mp:                            # a tier restore reads the configured formats
        mp.setenv(Q.KV_ENV, kf)
        mp.setenv(Q.INDEX_ENV, idf)
        restored = restore_minimax_m3_sparse(*[_host(x) for x in c.state])   # a disk tier's arrays
        assert restored.kv_format() is kfmt and restored.index_format() is ifmt
    for a, b in zip(restored.state, c.state):
        assert _eq(a, b)
    clone = clone_minimax_m3_sparse(c, 1001)
    assert clone.kv_format() is kfmt and clone.index_format() is ifmt and int(clone.offset) == 1001
    for a, b in zip(clone.state, c.state):
        assert _eq(a, b[:, :, :1001])
    # nbytes reports the packed lanes
    assert c.nbytes == int(c.fused_lanes()[0].nbytes + c.fused_lanes()[1].nbytes)


def test_a_restore_under_another_layout_fails_closed():
    c = MiniMaxM3SparseCache("mxfp8", "mxfp8")
    rk, _ = _bf16(_rows((1, 4, 10, 128), 1))
    ri, _ = _bf16(_rows((1, 1, 10, 128), 2))
    c.update_and_fetch(rk, rk)
    c.update_index(ri)
    state = c.state
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(Q.KV_ENV, "nvfp4")                                    # 76-byte rows expected, 132 found
        mp.setenv(Q.INDEX_ENV, "nvfp4")
        with pytest.raises(ValueError):
            restore_minimax_m3_sparse(*state)
        with pytest.raises(ValueError):                                 # no quantized format configured at all
            mp.delenv(Q.KV_ENV)
            restore_minimax_m3_sparse(*state)


@pytest.mark.parametrize("kf, idf", [("mxfp8", "nvfp4"), ("affine4", "mxfp8")])
def test_batch_cache_merge_append_extract(kf, idf):
    rows = [_bf16(_rows((1, 4, n, 128), n, 2.0))[0] for n in (131, 259)]
    idx = [_bf16(_rows((1, 1, n, 128), n + 1, 2.0))[0] for n in (131, 259)]
    singles = []
    for r, i in zip(rows, idx):
        c = MiniMaxM3SparseCache(kf, idf)
        c.update_and_fetch(r, r)
        c.update_index(i)
        singles.append(c)
    merged = BatchMiniMaxM3SparseCache.merge(singles)
    assert merged.keys.dtype == mx.uint8 and merged.kv_format().name == kf and merged.index_format().name == idf
    new, _ = _bf16(_rows((2, 4, 1, 128), 5, 2.0))
    newi, _ = _bf16(_rows((2, 1, 1, 128), 6, 2.0))
    k, v = merged.update_and_fetch(new, new)
    hist = merged.update_index(newi)
    assert k.dtype == mx.bfloat16 and int(k.shape[2]) == 260 and int(hist.shape[2]) == 260
    for b, (r, n) in enumerate(zip(rows, (131, 259))):
        ext = merged.extract(b)
        assert ext.kv_format().name == kf and int(ext.offset) == n + 1
        ref = MiniMaxM3SparseCache(kf, idf)
        ref.update_and_fetch(mx.concatenate([r, new[b:b + 1]], axis=2), mx.concatenate([r, new[b:b + 1]], axis=2))
        ref.update_index(mx.concatenate([idx[b], newi[b:b + 1]], axis=2))
        for a, w in zip(ext.state, ref.state):
            assert _eq(a, w)
    # extending with an empty batch keeps packed lanes packed (the stock extend would promote them to float32)
    empty = BatchMiniMaxM3SparseCache([0], kf, idf)
    merged.extend(empty)
    assert merged.keys.dtype == mx.uint8 and merged.idx_keys.dtype == mx.uint8


# ------------------------------------------------------------------------------------------- fused kernels
def _packed_lane_cache(ctx: int, seed: int, kf: str, idf: str, spare: int = 0) -> MiniMaxM3SparseCache:
    """A sparse cache at ``ctx`` rows of random K / V / index keys in the given formats, capacity ctx + spare."""
    rng = np.random.default_rng(seed)
    lanes = [mx.array(rng.standard_normal((1, h, ctx + spare, 128), dtype=np.float32)).astype(mx.bfloat16)
             for h in (4, 4, 1)]
    kfmt, ifmt = Q.get(kf), Q.get(idf)
    enc = [Q.quantize(lanes[0], kfmt) if kfmt.quantized else lanes[0],
           Q.quantize(lanes[1], kfmt) if kfmt.quantized else lanes[1],
           Q.quantize(lanes[2], ifmt) if ifmt.quantized else lanes[2]]
    c = MiniMaxM3SparseCache(kf, idf)
    c.state = tuple(enc)
    c.trim(spare)
    mx.eval(c.fused_lanes())
    return c


@pytest.fixture(scope="module")
def layer():
    return TF.Layer()


@pytest.mark.parametrize("kf, idf", COMBOS)
@pytest.mark.parametrize("ctx, rows", [(700, 3), (3001, 1), (3001, 5), (9000, 16)])
def test_fused_kernels_read_and_write_the_packed_rows(layer, monkeypatch, kf, idf, ctx, rows):
    """K2's packed rows are the unfused encoder's bytes of K2's own rows; K3's scores and K4's attention over packed
    lanes equal theirs over plain lanes holding the decoded rows, bit for bit."""
    monkeypatch.setenv(F.ENV, "1")
    if True:
        x = TF.rows_input(rows, layer.hidden, ctx + rows)
        A = _packed_lane_cache(ctx, 1, kf, idf)
        p = F.plan(layer)
        y = F.qkv(layer, p, x.reshape(rows, -1))
        dts = F.lane_dtypes(layer, A, x.dtype)
        nb = -(-(ctx + rows) // p.block)
        meta = mx.array([ctx, rows, nb, 0, 0, 0, 0, 0], dtype=mx.int32)
        fm = F._fmts(A)
        _, _, _, packed = F.heads(layer, p, y, meta, None, dts, False, fm)
        _, _, _, plain = F.heads(layer, p, y, meta, None, dts, False, (0, 0))
        kfmt, ifmt = Q.get(kf), Q.get(idf)
        for got, want, fmt in ((packed[0], plain[0], kfmt), (packed[1], plain[1], kfmt), (packed[2], plain[2], ifmt)):
            assert _eq(got, Q.quantize(want[None], fmt)[0] if fmt.quantized else want)
        # append the rows (copy-on-write path: the lanes were assigned from outside), then compare the readers
        h = F.attention_block(layer, x, A)
        mx.eval(h)
        assert int(A.offset) == ctx + rows and int(A._idx_offset) == ctx + rows
        B = MiniMaxM3SparseCache()
        k_dec, v_dec = A.read_kv(mx.bfloat16)
        B.state = (k_dec, v_dec, A.read_index(mx.bfloat16))
        mx.eval(B.fused_lanes())
        q, iq, cnt, _ = F.heads(layer, p, y, meta, None, dts, False, fm)
        q2, iq2, cnt2, _ = F.heads(layer, p, y, meta, None, dts, False, fm)
        kA, vA, iA = A.fused_lanes()
        kB, vB, iB = B.fused_lanes()
        if nb > p.keep:
            scA = F.scores(p, iq, iA, meta, nb, fm[1], dts[2])
            scB = F.scores(p, iq, iB, meta, nb)
            assert _eq(scA, scB)
        else:
            scA = scB = F._dummy()
        oA = F.msa(p, q, kA, vA, scA, cnt, meta, fm[0], dts[0], dts[1])
        oB = F.msa(p, q2, kB, vB, scB, cnt2, meta)
        assert _eq(oA, oB)
        # control: one flipped code byte in a selected-or-dense K row changes the packed read
        if kfmt.quantized:
            kA2 = mx.array(np.array(kA))
            arr = np.array(kA2)
            arr[0, 0, ctx + rows - 1, 3] ^= 0x41
            q3, _, cnt3, _ = F.heads(layer, p, y, meta, None, dts, False, fm)
            oC = F.msa(p, q3, mx.array(arr), vA, scA, cnt3, meta, fm[0], dts[0], dts[1])
            assert not _eq(oA, oC)


@pytest.mark.parametrize("kf, idf", [("mxfp8", "nvfp4"), ("nvfp4", "mxfp8"), ("affine4", "bf16")])
def test_fused_packed_appends_in_place_and_never_under_a_held_view(layer, kf, idf, monkeypatch):
    monkeypatch.setenv(F.ENV, "1")
    c = _packed_lane_cache(2500, 3, kf, idf)
    for s in range(6):
        mx.eval(F.attention_block(layer, TF.rows_input(1, layer.hidden, s), c))
    before = dict(F.STATS)
    for s in range(5):
        mx.eval(F.attention_block(layer, TF.rows_input(1, layer.hidden, 10 + s), c))
    assert F.STATS["inplace"] - before["inplace"] == 5                # steady decode: in place
    held = [x[:, :, :2400] for x in c.state]                           # a zero-copy store view (marks the lanes)
    snap = [_host(x) for x in held]
    assert c.trim(200) == 200                                          # new rows now land below the view's extent
    for s in range(3):
        mx.eval(F.attention_block(layer, TF.rows_input(1, layer.hidden, 20 + s), c))
    for a, b in zip(held, snap):
        assert _eq(a, b)
    _invariants(c)


@pytest.mark.parametrize("kf, idf", [("mxfp8", "mxfp8"), ("nvfp4", "nvfp4")])
def test_lazily_chained_packed_steps_across_a_lane_growth(layer, kf, idf, monkeypatch):
    monkeypatch.setenv(F.ENV, "1")
    xs = [TF.rows_input(1, layer.hidden, 40 + s) for s in range(12)]
    lazy = _packed_lane_cache(2040, 4, kf, idf, spare=6)              # capacity 2046: step 7 grows the lanes
    outs = [F.attention_block(layer, x, lazy) for x in xs]            # and step 9 crosses 2,048 keys, unevaluated
    mx.eval(outs)
    eager = _packed_lane_cache(2040, 4, kf, idf, spare=6)
    ref = []
    for x in xs:
        ref.append(F.attention_block(layer, x, eager))
        mx.eval(ref[-1], eager.fused_lanes())
    assert all(_eq(a, b) for a, b in zip(outs, ref))
    for a, b in zip(lazy.state, eager.state):
        assert _eq(a, b)


# ------------------------------------------------------------------------------------------- model level
@pytest.fixture(scope="module")
def target():
    return MiniMaxM3DSparkTarget(TF._model())


def _set(monkeypatch, kf, idf, fused):
    monkeypatch.setenv(Q.KV_ENV, kf)
    monkeypatch.setenv(Q.INDEX_ENV, idf)
    if fused:
        monkeypatch.setenv(F.ENV, "1")
    else:
        monkeypatch.delenv(F.ENV, raising=False)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("kf, idf", COMBOS)
@pytest.mark.parametrize("prompt_len, width", [(301, 5), (765, 17)])
def test_verify_rows_equal_decode_steps_with_packed_lanes(target, monkeypatch, kf, idf, fused, prompt_len, width):
    """Row r of a T-row DSpark verify == the r-th of T one-token decode steps, bit for bit (logits, tap features,
    every lane byte), with packed lanes, fused attention on and off."""
    _set(monkeypatch, kf, idf, fused)
    prompt = mx.array([TD._ids(prompt_len, prompt_len)])
    rows = mx.array([TD._ids(width, 7 * width)])
    taps = TD.DRAFT_TAPS
    cache = target.make_cache()
    target.prefill(prompt, cache, taps, 256)
    sparse = [c for c in cache if isinstance(c, MiniMaxM3SparseCache)]
    assert sparse and all(c.kv_format().name == kf and c.index_format().name == idf for c in sparse)
    if Q.get(kf).quantized:
        assert all(c.fused_lanes()[0].dtype == mx.uint8 for c in sparse)
    logits, feats, _ = target.verify(rows, cache, taps)
    mx.eval(logits, feats)
    ref = target.make_cache()
    target.prefill(prompt, ref, taps, 256)
    F.STATS.update(calls=0)
    ref_logits, ref_feats = [], []
    for r in range(width):
        lg, f = target.ar_step(rows[0, r:r + 1], ref, taps)
        ref_logits.append(lg)
        ref_feats.append(f)
    mx.eval(ref_logits)
    assert (F.STATS["calls"] == 3 * width) if fused else F.STATS["calls"] == 0
    bad = [r for r in range(width) if not _eq(logits[r], ref_logits[r])]
    assert not bad, f"verify rows {bad} differ from decode steps"
    assert _eq(feats, mx.concatenate(ref_feats, axis=1))
    assert not TD._cache_problems(cache, ref)


@pytest.mark.parametrize("kf, idf", [("mxfp8", "mxfp8"), ("nvfp4", "nvfp4"), ("affine8", "mxfp8")])
def test_formats_change_the_numbers_and_stay_close(target, monkeypatch, kf, idf):
    """Control: the packed formats really are in the computation (logits differ from bf16 lanes) and stay near them
    on this random model (teacher-forced along bf16's greedy tokens)."""
    prompt = TD._ids(773, 11)
    tokens = generate_ar(target, prompt, 12, set(), prefill_step=256)
    base = TF._decode_logits(target, prompt, tokens)
    _set(monkeypatch, kf, idf, False)
    got = TF._decode_logits(target, prompt, tokens)
    diffs = [float(mx.abs(a - b).max()) for a, b in zip(base, got)]
    assert max(diffs) > 0                                              # not silently bf16
    scale = max(float(mx.abs(a).max()) for a in base)
    assert np.median(diffs) < 0.5 * scale


@pytest.mark.parametrize("kf, idf", [("mxfp8", "nvfp4"), ("nvfp4", "nvfp4"), ("affine4", "mxfp8")])
def test_partial_prefix_hit_equals_cold_prefill(target, monkeypatch, kf, idf):
    """A producer prefills P = A + B, its cache is cut to len(A) (a stored prefix), restored through the tier
    arrays and continued with a different suffix C: logits == a cold prefill of A + C cut at len(A), bit for bit.
    Lengths are not multiples of 128 / 256. Control: one flipped byte in the restored K lane changes the logits."""
    _set(monkeypatch, kf, idf, False)
    a_len = 701
    A = TD._ids(a_len, 1)
    Bs, Cs = TD._ids(203, 2), TD._ids(149, 3)

    def run(prefix_cache, tokens):
        logits = target.model(mx.array([tokens]), cache=prefix_cache)
        mx.eval(logits)
        return logits[0].astype(mx.float32)

    cold = target.make_cache()
    run(cold, A)
    cold_logits = run(cold, Cs)
    producer = target.make_cache()
    run(producer, A)
    run(producer, Bs)
    stored = []
    for layer_cache in producer:
        if isinstance(layer_cache, MiniMaxM3SparseCache):
            clone = clone_minimax_m3_sparse(layer_cache, a_len)
            stored.append(("m3", [_host(x) for x in clone.state]))
        else:
            stored.append(("kv", [_host(x[:, :, :a_len]) for x in layer_cache.state]))

    def restore(flip=False):
        out = []
        for li, (tag, arrays) in enumerate(stored):
            if tag == "m3":
                arrs = [_host(x) for x in arrays]
                if flip and li == 2 and arrs[0].dtype == mx.uint8:
                    k = np.array(arrs[0])
                    k[0, 0, a_len - 3, 7] ^= 0x22
                    arrs[0] = mx.array(k)
                out.append(restore_minimax_m3_sparse(*arrs))
            else:
                from mlx_lm.models.cache import KVCache

                kv = KVCache()
                kv.state = tuple(arrays)
                out.append(kv)
        return out

    warm = restore()
    assert all(c.kv_format().name == kf for c in warm if isinstance(c, MiniMaxM3SparseCache))
    assert _eq(run(warm, Cs), cold_logits)
    if Q.get(kf).quantized:
        assert not _eq(run(restore(flip=True), Cs), cold_logits)


@pytest.fixture(scope="module")
def drafter_dir(tmp_path_factory):
    return TD._save_release_drafter(tmp_path_factory.mktemp("m31_kvq") / "dspark")


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("kf, idf", [("mxfp8", "mxfp8"), ("nvfp4", "nvfp4")])
def test_released_geometry_dspark_equals_ar_with_packed_lanes(drafter_dir, monkeypatch, kf, idf, fused):
    """Blocks of 128 keys, top-16, a 2,111-token prompt (past the 2,048-key dense threshold, not a multiple of 128):
    greedy DSpark == greedy AR with packed lanes (drafts wrong every fifth position)."""
    _set(monkeypatch, kf, idf, fused)
    t = MiniMaxM3DSparkTarget(TF._model(seed=1, block=128, topk=16))
    prompt = TD._ids(2111, 5)
    reference = generate_ar(t, prompt, 24, set(), prefill_step=512)
    drafter = TD._Scripted(load_drafter(drafter_dir), prompt + reference, TD._every_fifth)
    out = generate_dspark(t, drafter, prompt, 24, set(), prefill_step=512, max_draft=2 * TD.DRAFT_BLOCK,
                          ar_fallback=ARFallback())
    assert out == reference
