"""Lane matmul (metal/lane_qmm.py): row-independent, accurate at every JANG_4D width, tiled == untiled, safe routing."""
import mlx.core as mx
import mlx.nn as nn
import pytest

from vmlx_engine.metal import lane_qmm as L

pytestmark = pytest.mark.skipif(not L.available(), reason="needs Metal 4 tensor ops (M5-class GPU)")


def _weight(n, k, bits):
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=64, bits=bits)
    return q, s.astype(mx.bfloat16), b.astype(mx.bfloat16)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_rows_match_solo_calls_and_reference(bits):
    mx.random.seed(bits)
    q, s, b = _weight(256, 512, bits)
    sbt = L.pack_scales(s, b)
    x = mx.random.normal((16, 512)).astype(mx.bfloat16)
    many = L.lane_matmul(x, q, sbt, bits=bits)
    solo = mx.concatenate([L.lane_matmul(x[i:i + 1], q, sbt, bits=bits) for i in range(16)], axis=0)
    assert mx.array_equal(many, solo)                       # bits never depend on the row count
    for m in (5, 8, 12):
        assert mx.array_equal(L.lane_matmul(x[:m], q, sbt, bits=bits), many[:m])
    ref = x.astype(mx.float32) @ mx.dequantize(q, s, b, group_size=64, bits=bits).astype(mx.float32).T
    mlx = mx.quantized_matmul(x, q, s, b, transpose=True, group_size=64, bits=bits).astype(mx.float32)
    lane_err = mx.abs(many.astype(mx.float32) - ref).max().item()
    mlx_err = mx.abs(mlx - ref).max().item()
    assert lane_err <= 2 * mlx_err + 1e-3


@pytest.mark.parametrize("bits", [4, 8])
def test_tiled_layout_gives_identical_bits(bits):
    q, s, b = _weight(128, 256, bits)
    sbt = L.pack_scales(s, b)
    x = mx.random.normal((7, 256)).astype(mx.bfloat16)
    tiled = L.tile_weight(q, bits=bits)
    assert mx.array_equal(L.untile_weight(tiled, bits=bits), q)
    assert mx.array_equal(L.lane_matmul(x, tiled, sbt, bits=bits, tiled=True), L.lane_matmul(x, q, sbt, bits=bits))


def test_install_routes_only_plain_affine_layers_and_wide_calls_stay_mlx_exact():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(256, 128, bias=True)
            self.f = nn.Linear(256, 64, bias=False)            # left unquantized
    m = M()
    nn.quantize(m, group_size=64, bits=4, class_predicate=lambda p, mod: p == "a")
    m.a.scales = m.a.scales.astype(mx.bfloat16)
    m.a.biases = m.a.biases.astype(mx.bfloat16)
    wide = mx.random.normal((1, 300, 256)).astype(mx.bfloat16)  # > MAX_ROWS: prefill chunk
    ref_wide = m.a(wide)
    x = mx.random.normal((1, 6, 256)).astype(mx.bfloat16)
    counts = L.install(m)
    assert counts["lane"] == 1 and counts["tiled"] == 1
    assert isinstance(m.a, L.LaneQuantizedLinear) and type(m.f) is nn.Linear
    # the packed pairs are the only resident copy of the bf16 scales/biases (RAM at the model's size)
    assert "scales" not in m.a and "biases" not in m.a
    assert {k for k, _ in nn.utils.tree_flatten(m.a.parameters())} == {"weight", "bias"}
    assert mx.array_equal(m.a(wide), ref_wide)                 # untiled + scales rebuilt on the fly, MLX's kernel
    y = m.a(x)
    assert y.shape == (1, 6, 128)
    assert L.install(m)["lane"] == 0                           # idempotent


def test_install_keeps_vocabulary_heads_on_mlx(monkeypatch):
    """lm_head-sized projections (N >= VOCAB_HEAD_MIN_N) stay on MLX: the lane kernel measured slower at 1/5/16 rows
    (N=248,320 K=5,120 8-bit: 3.46 vs 2.54 ms at 1 row). VMLX_LANE_VOCAB_HEAD=1 restores the lane for A/B."""
    def model():
        m = nn.Sequential(nn.Linear(256, 256, bias=False), nn.Linear(256, L.VOCAB_HEAD_MIN_N, bias=False))
        nn.quantize(m, group_size=64, bits=8)
        for layer in m.layers:
            layer.scales = layer.scales.astype(mx.bfloat16)
            layer.biases = layer.biases.astype(mx.bfloat16)
        return m

    m = model()
    counts = L.install(m)
    assert counts["lane"] == 1 and counts["skipped"].get("vocab-sized head") == 1
    assert type(m.layers[1]) is nn.QuantizedLinear and isinstance(m.layers[0], L.LaneQuantizedLinear)
    monkeypatch.setenv("VMLX_LANE_VOCAB_HEAD", "1")
    m2 = model()
    assert L.install(m2)["lane"] == 2


def test_install_keeps_non_bf16_biases_and_the_keep_switch(monkeypatch):
    """Only an exact copy is dropped: fp32 biases (packed pairs round them to bf16) keep the originals for the
    MLX-exact wide fallback; VMLX_LANE_KEEP_SCALES=1 keeps them for A/B."""
    def layer(bias_dtype):
        m = nn.Sequential(nn.Linear(256, 128, bias=False))
        nn.quantize(m, group_size=64, bits=4)
        m.layers[0].scales = m.layers[0].scales.astype(mx.bfloat16)
        m.layers[0].biases = m.layers[0].biases.astype(bias_dtype)
        return m
    wide = mx.random.normal((1, 200, 256)).astype(mx.bfloat16)
    m = layer(mx.float32)
    ref = m.layers[0](wide)
    L.install(m)
    assert "scales" in m.layers[0] and "biases" in m.layers[0]
    assert mx.array_equal(m.layers[0](wide), ref)
    monkeypatch.setenv("VMLX_LANE_KEEP_SCALES", "1")
    m2 = layer(mx.bfloat16)
    L.install(m2)
    assert "scales" in m2.layers[0]


def test_projections_fed_the_same_activation_share_one_xsum(monkeypatch):
    """q/k/v, gate/up and the GDN in-projections pass the same x object back to back: the XS row sums are computed once
    and reused, bit-identically; a different array (even with equal shape and values) recomputes them."""
    mx.random.seed(11)
    weights = [_weight(n, 512, bits) for n, bits in ((256, 4), (64, 8), (128, 4))]
    packed = [(q, L.pack_scales(s, b), bits) for (q, s, b), bits in zip(weights, (4, 8, 4))]
    x = mx.random.normal((1, 3, 512)).astype(mx.bfloat16)
    monkeypatch.setattr(L, "_SHARE_XSUM", False)
    unshared = [L.lane_matmul(x, q, sbt, bits=bits) for q, sbt, bits in packed]
    monkeypatch.setattr(L, "_SHARE_XSUM", True)
    L._XS_LAST.slot = None
    shared = []
    for q, sbt, bits in packed:
        shared.append(L.lane_matmul(x, q, sbt, bits=bits))
        if len(shared) == 1:
            first_xs = L._XS_LAST.slot[2]
        assert L._XS_LAST.slot[0] is x and L._XS_LAST.slot[2] is first_xs      # computed once for the three calls
    for a, b in zip(shared, unshared):
        assert mx.array_equal(a, b)
    twin = mx.array(x)                                                         # equal values, different object
    q, sbt, bits = packed[0]
    assert mx.array_equal(L.lane_matmul(twin, q, sbt, bits=bits), unshared[0])
    assert L._XS_LAST.slot[0] is twin and L._XS_LAST.slot[2] is not first_xs
    other = mx.random.normal((1, 3, 512)).astype(mx.bfloat16)
    ref = L.lane_matmul(other, q, sbt, bits=bits)
    monkeypatch.setattr(L, "_SHARE_XSUM", False)
    assert mx.array_equal(ref, L.lane_matmul(other, q, sbt, bits=bits))


def test_installed_model_output_is_identical_with_and_without_xsum_sharing(monkeypatch):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q, self.k, self.v = nn.Linear(256, 256, bias=False), nn.Linear(256, 64, bias=False), nn.Linear(256, 64, bias=False)
            self.gate, self.up, self.down = nn.Linear(256, 512, bias=False), nn.Linear(256, 512, bias=False), nn.Linear(512, 256, bias=False)

        def __call__(self, x):
            h = x + self.q(x)[..., :256] + mx.concatenate([self.k(x), self.v(x)], axis=-1).sum(-1, keepdims=True)
            return self.down(nn.silu(self.gate(h)) * self.up(h))

    mx.random.seed(5)
    m = Block()
    nn.quantize(m, group_size=64, bits=4)
    for p in ("q", "k", "v", "gate", "up", "down"):
        mod = getattr(m, p)
        mod.scales, mod.biases = mod.scales.astype(mx.bfloat16), mod.biases.astype(mx.bfloat16)
    L.install(m)
    x = mx.random.normal((1, 5, 256)).astype(mx.bfloat16)
    monkeypatch.setattr(L, "_SHARE_XSUM", False)
    off = m(x)
    monkeypatch.setattr(L, "_SHARE_XSUM", True)
    assert mx.array_equal(m(x), off)
