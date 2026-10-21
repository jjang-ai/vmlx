"""Row-exact native-MTP verification.

A verify forward over S rows must give every row the logits that the same
token gets from an ordinary single-token decode step, bit for bit, so greedy
native MTP reproduces AR output exactly. MLX switches projection, dense
matmul, rotary and SDPA kernels between one row and several rows; the verify
scope keeps every row on the single-row arithmetic.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from vmlx_engine.metal import row_exact_qmv as R


def _quantized(out_dim, in_dim, bits, dtype, group_size=64):
    weight = mx.random.normal((out_dim, in_dim), key=mx.random.key(out_dim * 31 + in_dim)).astype(dtype)
    w, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)
    return w, scales.astype(dtype), biases.astype(dtype)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("shape", [(1024, 2048), (2560, 640), (1028, 1344), (12, 640), (96, 64), (997, 128), (5, 64), (4, 256), (1, 512)])
@pytest.mark.parametrize("rows", [2, 3, 4])
def test_each_row_equals_its_single_row_quantized_matmul(bits, dtype, shape, rows):
    out_dim, in_dim = shape
    group_size = 64 if in_dim % 64 == 0 else 32
    w, scales, biases = _quantized(out_dim, in_dim, bits, dtype, group_size)
    x = mx.random.normal((1, rows, in_dim), key=mx.random.key(7)).astype(dtype)
    got = R.row_exact_qmv(x, w, scales, biases, group_size=group_size, bits=bits)
    if got is None:
        # Only K=64/128 packs without a transcribed MLX M=1 kernel may decline (stock MLX keeps them).
        assert in_dim in (64, 128) and bits not in (2, 4, 8)
        pytest.skip("no row-exact transcription for this MLX M=1 kernel")
    want = mx.concatenate([
        mx.quantized_matmul(x[:, r:r + 1], w, scales, biases, transpose=True, group_size=group_size, bits=bits)
        for r in range(rows)
    ], axis=1)
    assert got.shape == want.shape
    assert bool(mx.array_equal(got, want).item())


@pytest.mark.parametrize("bits", [4, 6, 8])
@pytest.mark.parametrize("shape", [(512, 1024), (3584, 1024), (1024, 2048), (1024, 3584), (4096, 768)])
@pytest.mark.parametrize("rows", [5, 6, 7, 8])
def test_rows_stay_exact_on_activation_like_inputs_across_weights(bits, shape, rows):
    """Wide windows (5-8 rows, single launch or row blocks) on shifted, outlier-bearing rows."""
    out_dim, in_dim = shape
    for seed in range(3):
        w = mx.random.normal((out_dim, in_dim), key=mx.random.key(40 + seed)) * 0.05
        wq, scales, biases = mx.quantize(w.astype(mx.bfloat16), group_size=64, bits=bits)
        x = (mx.random.normal((1, rows, in_dim), key=mx.random.key(80 + seed))
             + mx.random.uniform(-2.0, 2.0, (1, rows, 1), key=mx.random.key(90 + seed))).astype(mx.bfloat16)
        got = R.row_exact_qmv(x, wq, scales, biases, group_size=64, bits=bits)
        assert got is not None
        want = mx.concatenate([
            mx.quantized_matmul(x[:, r:r + 1], wq, scales, biases, transpose=True, group_size=64, bits=bits)
            for r in range(rows)
        ], axis=1)
        assert bool(mx.array_equal(got, want).item())


@pytest.mark.parametrize("bits", [4, 6, 8])
@pytest.mark.parametrize("in_dim", [768, 1024, 1280, 3584])
def test_offered_kernels_are_the_mlx_single_row_kernel_family(bits, in_dim):
    """Only the transcription of MLX's own M=1 kernel may be offered for admission.

    qmv_fast and qmv_impl share most arithmetic, so the other family matches
    some probes by coincidence; admitted that way (per shape) it differed from
    M=1 on real rows of other weights of the same shape.
    """
    out_dim, rows = 4096, 2
    fast_ok = in_dim % (R._values_per_thread(bits) * 32) == 0
    variants = R._variants(rows, in_dim, out_dim, bits, 64, mx.bfloat16)
    assert {v.split("_")[0] for v in variants if v != "quad"} == ({"fast"} if fast_ok else {"generic"})
    w = mx.random.normal((out_dim, in_dim), key=mx.random.key(3)) * 0.05
    wq, scales, biases = mx.quantize(w.astype(mx.bfloat16), group_size=64, bits=bits)
    for seed in range(4):
        x = (mx.random.normal((rows, in_dim), key=mx.random.key(200 + seed))
             + mx.random.uniform(-3.0, 3.0, (rows, 1), key=mx.random.key(300 + seed))).astype(mx.bfloat16)
        want = mx.concatenate([
            mx.quantized_matmul(x[r:r + 1], wq, scales, biases, transpose=True, group_size=64, bits=bits)
            for r in range(rows)
        ])
        for variant in variants:
            assert bool(mx.array_equal(R._launch(x, wq, scales, biases, bits, 64, variant), want).item()), variant


def test_unsupported_inputs_fall_back_to_stock():
    w, scales, biases = _quantized(256, 512, 4, mx.float16)
    one_row = mx.zeros((1, 1, 512), dtype=mx.float16)
    too_many = mx.zeros((1, R.MAX_ROWS + 1, 512), dtype=mx.float16)
    assert R.row_exact_qmv(one_row, w, scales, biases, group_size=64, bits=4) is None
    assert R.row_exact_qmv(too_many, w, scales, biases, group_size=64, bits=4) is None
    assert R.row_exact_qmv(mx.zeros((1, 2, 512), dtype=mx.float16), w, scales, None,
                           group_size=64, bits=4, mode="mxfp4") is None


def test_missing_mlx_header_disables_the_kernel(monkeypatch):
    monkeypatch.setattr(R, "_mlx_qmv_header", lambda: None)
    w, scales, biases = _quantized(256, 512, 4, mx.float16)
    assert R.row_exact_qmv(mx.zeros((1, 2, 512), dtype=mx.float16), w, scales, biases,
                           group_size=64, bits=4) is None


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("shape", [(36, 10240), (10240, 32), (512, 2560), (4, 1024)])
def test_dense_linear_rows_equal_single_row_gemv(dtype, shape):
    out_dim, in_dim = shape
    weight = (mx.random.normal((out_dim, in_dim), key=mx.random.key(3)) * 0.02).astype(dtype)
    x = mx.random.normal((1, 4, in_dim), key=mx.random.key(5)).astype(dtype)
    got = R.row_exact_linear(x, weight)
    assert got is not None
    want = mx.concatenate([x[:, r:r + 1] @ weight.T for r in range(4)], axis=1)
    assert bool(mx.array_equal(got, want).item())


def test_scope_routes_quantized_and_dense_linears_only_inside_verify(monkeypatch):
    # process-global dense row-invariant mode (set by set_row_exact_family in other tests)
    # intercepts nn.Linear before the verify routing this test counts
    monkeypatch.setitem(R._DENSE_ROW_INVARIANT, "on", False)
    monkeypatch.delenv("VMLX_ROW_EXACT_VERIFY_QMV", raising=False)
    q = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=4)
    q.set_dtype(mx.float16)  # the kernel serves fp16/bf16 activations
    d = nn.Linear(512, 256, bias=False)
    x = mx.random.normal((1, 3, 512), key=mx.random.key(9))
    xq = x.astype(mx.float16)
    calls = dict(R.STATS)
    with R.row_exact_verify_scope() as active:
        assert active and R.row_exact_scope_active()
        q(xq), d(x)
    assert not R.row_exact_scope_active()
    assert R.STATS["calls"] == calls.get("calls", 0) + 1
    assert R.STATS.get("dense_calls", 0) == calls.get("dense_calls", 0) + 1
    q(xq), d(x)  # outside the scope: stock MLX, no new kernel calls
    assert R.STATS["calls"] == calls.get("calls", 0) + 1


def test_scope_opt_out(monkeypatch):
    monkeypatch.setenv("VMLX_ROW_EXACT_VERIFY_QMV", "0")
    with R.row_exact_verify_scope() as active:
        assert active is False and not R.row_exact_scope_active()


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("width", [2, 3, 4])
def test_tiny_qwen4_verify_rows_equal_decode_steps(bits, width):
    from tests.test_qwen4_exp_runtime import _randomize, _tiny_args
    from vmlx_engine.models.qwen4_exp.language import LanguageModel

    model = LanguageModel(_tiny_args())
    _randomize(model)
    nn.quantize(model, group_size=32, bits=bits,
                class_predicate=lambda _p, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % 32 == 0)
    model.set_dtype(mx.float16)
    mx.eval(model.parameters())
    prompt = mx.array([[5, 17, 300, 41, 2, 999 % 997, 63, 128, 7, 11, 501, 33, 90]])

    def prefill():
        cache = model.make_cache()
        logits = model(prompt, cache=cache).logits
        nxt = int(mx.argmax(logits[:, -1], -1).item())
        return cache, nxt

    cache, token = prefill()
    chain, steps = [token], []
    for _ in range(width):
        logits = model(mx.array([[chain[-1]]]), cache=cache).logits[:, -1]
        mx.eval(logits)
        steps.append(logits)
        chain.append(int(mx.argmax(logits, -1).item()))

    cache, again = prefill()
    assert again == token
    with R.row_exact_verify_scope():
        verify = model(mx.array([chain[:width]]), cache=cache, n_confirmed=1).logits
    mx.eval(verify)
    for r in range(width):
        assert bool(mx.array_equal(verify[:, r], steps[r]).item()), f"row {r} differs from its decode step"


@pytest.mark.parametrize("positions", ["text", "mrope"])
@pytest.mark.parametrize("width", [2, 3, 4, 16])
def test_qwen35_attention_verify_rows_equal_decode_steps(positions, width):
    """Qwen3.5/3.8 dense attention (27B path): fused text RoPE per row, hoisted
    M-RoPE angles formed like decode, and paired single-query SDPA."""
    from mlx_lm.models.cache import KVCache
    from vmlx_engine.patches.mlx_vlm_mtp import qwen35_vl

    qwen35_vl.apply()
    from mlx_vlm.models.qwen3_5 import config as qcfg
    from mlx_vlm.models.qwen3_5 import language as qlang

    cfg = qcfg.TextConfig(
        model_type="qwen3_5_text", hidden_size=256, intermediate_size=512,
        linear_num_value_heads=2, linear_num_key_heads=2, linear_key_head_dim=32,
        linear_value_head_dim=32, linear_conv_kernel_dim=4, num_hidden_layers=4,
        num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=97, num_key_value_heads=2,
        max_position_embeddings=4096, head_dim=256,
    )
    mx.random.seed(23)
    attn = qlang.Qwen3_5Attention(cfg)
    nn.quantize(attn, group_size=64, bits=8)
    attn.set_dtype(mx.bfloat16)
    prefix = (mx.random.normal((1, 9, 256)) * 0.5).astype(mx.bfloat16)
    rows = (mx.random.normal((1, width, 256)) * 0.5).astype(mx.bfloat16)

    def pos(start, count):
        return mx.broadcast_to(mx.arange(start, start + count)[None, None], (3, 1, count))

    def call(x, cache, start, verify=False):
        if positions == "text":
            return attn(x, mask="causal" if x.shape[1] > 1 else None, cache=cache)
        p = pos(start, x.shape[1])
        if verify:
            pe = R.rotary_rows_like_decode(attn.rotary_emb, p, x)
        else:
            pe = attn.rotary_emb(x, p)
        return attn(x, mask="causal" if x.shape[1] > 1 else None, cache=cache,
                    position_ids=p, position_embeddings=pe)

    reference = KVCache()
    mx.eval(call(prefix, reference, 0))
    steps = []
    for r in range(width):
        out = call(rows[:, r:r + 1], reference, 9 + r)
        mx.eval(out)
        steps.append(out)
    candidate = KVCache()
    mx.eval(call(prefix, candidate, 0))
    with R.row_exact_verify_scope(
        rows=max(R.MAX_ROWS, width), required=width > R.MAX_ROWS
    ):
        verify = call(rows, candidate, 9, verify=True)
    mx.eval(verify)
    for r in range(width):
        assert bool(mx.array_equal(verify[:, r:r + 1], steps[r]).item()), f"row {r} differs"


def test_row_exact_default_is_per_family_and_env_wins(monkeypatch):
    monkeypatch.delenv("VMLX_ROW_EXACT_VERIFY_QMV", raising=False)
    try:
        R.set_row_exact_family("qwen4_exp")
        assert R.row_exact_qmv_requested()
        R.set_row_exact_family("qwen3_5")  # dense 27B: speed first by default
        assert not R.row_exact_qmv_requested()
        monkeypatch.setenv("VMLX_ROW_EXACT_VERIFY_QMV", "1")
        assert R.row_exact_qmv_requested()
        R.set_row_exact_family("qwen4_exp")
        monkeypatch.setenv("VMLX_ROW_EXACT_VERIFY_QMV", "0")
        assert not R.row_exact_qmv_requested()
    finally:
        R.set_row_exact_family(None)


@pytest.mark.parametrize("shape", [(1, 2560), (4, 1024), (512, 2560), (324, 10240)])
@pytest.mark.parametrize("rows", [2, 3, 4])
def test_dense_rows_form_is_identical_for_one_row_and_many(shape, rows):
    """Decode (1 row) and verify (rows) use the same row-invariant dense form."""
    out_dim, in_dim = shape
    weight = (mx.random.normal(shape, key=mx.random.key(11)) * 0.02).astype(mx.float16)
    for trial in range(40):
        x = mx.random.normal((1, rows, in_dim), key=mx.random.key(100 + trial)).astype(mx.float16)
        many = R._dense_rows(x, weight)
        for r in range(rows):
            one = R._dense_rows(x[:, r:r + 1], weight)
            assert bool(mx.array_equal(many[:, r:r + 1], one).item()), f"trial {trial} row {r}"
