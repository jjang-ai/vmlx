"""LiLiCorr drafter port (models/nemotron_lilicorr.py) against direct transcriptions of SGLang's reference formulas
(sgl-project/sglang#37462): grouped dynamic conv (dflash.py _grouped_conv), the greedy selector walk
(kernels/ops/speculative/lilicorr.py) and the NVFP4 global scale. The head and drafter end to end are qualified by the
offline acceptance run on the real bundle (tau 2.8-4.9 greedy)."""
import numpy as np
import mlx.core as mx

from vmlx_engine.models import nemotron_lilicorr as L


def _ref_grouped_conv(x, delta, base, block, groups, gs, taps):
    # transcription of sglang _grouped_conv (numpy)
    blocks = x.reshape(x.shape[0], groups, gs)
    coeff = base.reshape(1, taps, groups, gs) + delta[..., None]
    out = coeff[:, 0] * blocks
    pos = np.arange(x.shape[0]) % block
    for t in range(1, taps):
        shifted = np.concatenate([np.zeros((t, groups, gs), x.dtype), blocks[:-t]], axis=0)
        out = out + coeff[:, t] * shifted * (pos >= t).reshape(-1, 1, 1)
    return out.reshape(x.shape[0], -1)


def test_grouped_conv_matches_reference_two_blocks():
    rng = np.random.default_rng(0)
    D, gs, taps, block, N = 32, 8, 2, 4, 8                    # two blocks: position 0 of block 2 must not see block 1
    w = {"c.base_kernel": mx.array(rng.normal(size=(2, taps, D)).astype(np.float32)),
         "c.kernel_projection.weight": mx.array(rng.normal(size=(2 * taps * (D // gs), D)).astype(np.float32))}
    conv = L.GroupedConv(w, "c", block, taps, gs)
    x = rng.normal(size=(N, D)).astype(np.float32)
    got_in, kern = conv.prepare(mx.array(x))
    # same (MLX) kernel projection on both sides: the test is about the conv arithmetic, not GEMM precision
    c = np.array(conv.kernel_projection(mx.array(x))).reshape(N, 2, taps, D // gs)
    want_in = _ref_grouped_conv(x, c[:, 0], np.array(w["c.base_kernel"])[0], block, D // gs, gs, taps)
    assert np.allclose(np.array(got_in), want_in, atol=1e-4)
    y = rng.normal(size=(N, D)).astype(np.float32)
    want_out = _ref_grouped_conv(y, c[:, 1], np.array(w["c.base_kernel"])[1], block, D // gs, gs, taps)
    assert np.allclose(np.array(conv.finish(mx.array(y), kern)), want_out, atol=1e-4)


def test_greedy_walk_follows_pair_scores():
    # slot0 picks argmax(start); each next slot picks argmax over pair[s][previous]
    start = np.array([0.1, 0.9, 0.2])
    pair = np.zeros((2, 3, 3)); pair[0, 1, 2] = 5.0; pair[1, 2, 0] = 3.0
    cand = np.array([[10, 11, 12], [20, 21, 22], [30, 31, 32]])
    assert L.greedy_walk(start, pair, cand) == [11, 22, 30]


def test_nvfp4_linear_applies_global_scale():
    w = mx.random.normal((32, 64)).astype(mx.bfloat16)
    q, s = mx.quantize(w, group_size=16, bits=4, mode="nvfp4")
    lin = L.NVFP4Linear(q, s, mx.array([0.5]), mx.ones((32,), dtype=mx.bfloat16))
    x = mx.random.normal((3, 64)).astype(mx.bfloat16)
    deq = mx.dequantize(q, s, None, group_size=16, bits=4, mode="nvfp4")
    want = (x @ deq.T) * 0.5 + 1.0
    assert mx.allclose(lin(x).astype(mx.float32), want.astype(mx.float32), atol=2e-2).item()
