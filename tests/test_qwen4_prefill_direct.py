"""Sparse-score QSA prefill: readiness rules and bit parity with stock masked SDPA."""

import mlx.core as mx
import pytest

from vmlx_engine.metal import qwen4_qsa_direct_prefill as direct
from vmlx_engine.metal import qwen4_qsa_sparse_kernels as kernels


@pytest.fixture
def fresh_lane(monkeypatch):
    monkeypatch.setattr(direct, "LANE", direct._LaneState())
    return direct.LANE


def test_off_by_default(monkeypatch, fresh_lane):
    monkeypatch.delenv("VMLX_QWEN4_PREFILL_DIRECT", raising=False)
    assert not direct.direct_prefill_requested()
    assert not direct.direct_ready()


def test_gpu_without_matrix_units_is_unsupported(monkeypatch, fresh_lane):
    monkeypatch.setenv("VMLX_QWEN4_PREFILL_DIRECT", "1")
    monkeypatch.setattr(kernels, "matrix_units_available", lambda: False)
    assert "NAX" in direct.lane_unavailable_reason()
    assert not direct.direct_ready()


@pytest.mark.parametrize("arch,expected", [
    ("applegpu_g17s", True), ("applegpu_g17d", True), ("applegpu_g18p", True),
    ("applegpu_g17p", False), ("applegpu_g16s", False), ("applegpu_g15g", False), ("", False),
])
def test_matrix_unit_architecture_gate(monkeypatch, arch, expected):
    monkeypatch.setattr(mx, "device_info", lambda: {"architecture": arch})
    assert kernels.matrix_units_available() is expected


def test_preflight_rejects_executable_but_wrong_math(monkeypatch, fresh_lane):
    monkeypatch.setattr(direct, "direct_attention",
                        lambda q, *a, **kw: mx.ones(q.shape, dtype=q.dtype))
    assert not direct._preflight_one(mx.bfloat16)
    assert fresh_lane.status == "failed"
    assert not fresh_lane.proven
    # terminal: a later success cannot reopen the lane
    fresh_lane.prove(mx.float16)
    assert fresh_lane.status == "failed" and not fresh_lane.proven


def test_dispatch_execution_does_not_publish_numeric_readiness(fresh_lane):
    direct._first_dispatch_check(mx.zeros((1,), mx.bfloat16), mx.bfloat16)
    assert fresh_lane.status == "unproven"
    assert not fresh_lane.proven


def test_preflight_accepts_stock_math_without_changing_sampling_rng(monkeypatch, fresh_lane):
    def stock(q, k, v, ids, valid, *, pos_start, total_tokens, scale, **kw):
        positions = mx.arange(pos_start, total_tokens)
        keys = mx.arange(total_tokens)
        keep = ((keys[None] < 2048) | (keys[None] >= ((positions + 1) // 4)[:, None] * 4))
        keep &= keys[None] <= positions[:, None]
        return mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=mx.where(keep, 0, -mx.inf).astype(q.dtype)[None, None])

    monkeypatch.setattr(direct, "direct_attention", stock)
    before = list(mx.random.state)
    assert direct._preflight_one(mx.bfloat16)
    after = list(mx.random.state)
    assert len(after) == len(before)
    assert all(bool(mx.array_equal(a, b)) for a, b in zip(after, before))
    assert fresh_lane.has(mx.bfloat16)
    assert fresh_lane.status == "unproven"      # float16 not proven yet
    assert direct._preflight_one(mx.float16)
    assert fresh_lane.status == "proven"


def test_selection_contract_validation_rejects_holes():
    ids = mx.broadcast_to(mx.arange(512, dtype=mx.int32)[None], (16, 512))
    valid = mx.ones((16, 512), dtype=mx.bool_)
    buffer = direct.selected_block_buffer(ids, valid, pos_start=4096, validate=True)
    assert buffer.shape == (1, 1, 16, 512) and buffer.dtype == mx.uint32
    holed = mx.where(mx.arange(512)[None] == 3, False, valid)
    with pytest.raises(ValueError, match="contract violated"):
        direct.selected_block_buffer(ids, holed, pos_start=4096, validate=True)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_model_indexer_and_attention_integration(dtype, monkeypatch):
    from copy import deepcopy

    from vmlx_engine.models.qwen4_exp import language

    monkeypatch.setenv("VMLX_QWEN4_PREFILL_DIRECT", "1")
    if not direct.direct_ready():
        pytest.skip("sparse-score lane unavailable on this GPU/runtime")
    mx.random.seed(37)
    attention = language.QSAAttention(language.Qwen4ExpTextArgs(hidden_size=64))
    attention.set_dtype(dtype)
    attention.eval()
    cache = language.QSACache()
    monkeypatch.setenv("VMLX_QWEN4_PREFILL_DIRECT", "0")
    for _ in range(8):
        mx.eval(attention(mx.random.normal((1, 1024, 64)).astype(dtype), cache=cache))
    old, new = deepcopy(cache), deepcopy(cache)
    x = mx.random.normal((1, 256, 64)).astype(dtype)
    ref = attention(x, cache=old)
    mx.eval(ref)
    monkeypatch.setenv("VMLX_QWEN4_PREFILL_DIRECT", "1")
    got = attention(x, cache=new)
    mx.eval(got)
    assert getattr(attention, "_direct_prefill_logged", False)
    assert old.offset == new.offset == 8448
    assert bool(
        mx.allclose(got.astype(mx.float32), ref.astype(mx.float32), atol=0.005, rtol=0.01)
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "rows,total",
    [(256, 2051), (256, 2052), (257, 4096),
     pytest.param(4096, 4096, marks=pytest.mark.xfail(
         strict=False,
         reason="MLX 0.32.4 stock SDPA takes a different kernel when the chunk is the whole "
                "context (pos_start 0); origin/main bd2972e46 fails this case identically")),
     (256, 8195), (256, 32768), (256, 131072)],
)
def test_sparse_scores_match_dense_sparse_mask(dtype, rows, total, monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_PREFILL_DIRECT", "1")
    if not direct.direct_ready():
        pytest.skip("sparse-score lane unavailable on this GPU/runtime")
    mx.random.seed(total + rows)
    offset = total - rows
    q = mx.random.normal((1, 24, rows, 256)).astype(dtype)
    # Logical views into padded allocations exercise nontrivial head strides.
    k = mx.random.normal((1, 2, total + 19, 256)).astype(dtype)[:, :, :total]
    v = mx.random.normal((1, 2, total + 19, 256)).astype(dtype)[:, :, :total]
    complete = mx.arange(offset + 1, total + 1) // 4
    scores = mx.random.uniform(shape=(rows, total // 4))
    scores = mx.where(mx.arange(total // 4)[None] < complete[:, None], scores, -mx.inf)
    ids = mx.sort(mx.argpartition(-scores, kth=511, axis=-1)[:, :512], axis=-1).astype(mx.int32)
    valid = ids < complete[:, None]
    selected = mx.put_along_axis(mx.zeros(scores.shape, dtype=mx.bool_), ids, mx.array(True), axis=-1)
    keep = mx.repeat(selected, 4, axis=-1)
    if total % 4:
        keep = mx.concatenate([keep, mx.zeros((rows, total % 4), dtype=mx.bool_)], axis=-1)
    positions = mx.arange(total)[None]
    mask = mx.where(
        (keep | (positions >= complete[:, None] * 4)) & (positions <= mx.arange(offset, total)[:, None]),
        0, -mx.inf,
    ).astype(dtype)[None, None]
    if total == 2051:
        with pytest.raises(ValueError, match="dense/sparse boundary"):
            direct.direct_attention(q, k, v, ids, valid, pos_start=offset, total_tokens=total, scale=0.0625)
        return
    got = direct.direct_attention(q, k, v, ids, valid, pos_start=offset, total_tokens=total, scale=0.0625)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625, mask=mask)
    mx.eval(ref, got)
    assert bool(mx.all(mx.isfinite(got)))
    assert bool(mx.array_equal(got, ref))
