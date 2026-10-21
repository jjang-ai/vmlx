"""DFlash windows preserve target decode arithmetic without a depth cap."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from vmlx_engine.dflash2_runtime import _TargetAdapter
from vmlx_engine.jangh import dense
from vmlx_engine.metal import row_exact_qmv as exact


def test_required_window_is_local_and_restores_native_policy(monkeypatch):
    monkeypatch.setattr(exact, "row_exact_qmv_requested", lambda: False)
    assert not exact.row_exact_scope_active()
    assert exact.row_exact_scope_rows() == exact.MAX_ROWS
    with exact.row_exact_verify_scope() as active:
        assert not active
    with pytest.raises(RuntimeError):
        with exact.row_exact_verify_scope(rows=16, required=True):
            assert exact.row_exact_scope_active()
            assert exact.row_exact_scope_rows() == 16
            with exact.row_exact_verify_scope(rows=24, required=True):
                assert exact.row_exact_scope_rows() == 24
            assert exact.row_exact_scope_rows() == 16
            raise RuntimeError("unwind a cancelled verify")
    assert not exact.row_exact_scope_active()
    assert exact.row_exact_scope_rows() == exact.MAX_ROWS


def test_adapter_scopes_only_target_verification(monkeypatch):
    monkeypatch.setattr(exact, "row_exact_qmv_requested", lambda: False)
    projection = SimpleNamespace(is_jangtq2_dense=True, row_exact_verify=False)
    model = SimpleNamespace(
        model=object(), named_modules=lambda: [("mlp.gate_proj", projection)]
    )
    adapter = _TargetAdapter(model)
    assert projection.row_exact_verify
    monkeypatch.setattr(
        adapter, "_forward",
        lambda inputs, cache: (
            exact.row_exact_scope_active(), exact.row_exact_scope_rows()
        ),
    )
    assert adapter(mx.zeros((1, 16))) == (True, 16)
    assert adapter(mx.zeros((1, 24))) == (True, 24)
    assert adapter(mx.zeros((1, 1))) == (False, exact.MAX_ROWS)
    adapter.last_row_logits = True
    assert adapter(mx.zeros((1, 16))) == (False, exact.MAX_ROWS)


@pytest.mark.parametrize("rows", [9, 16, 24])
@pytest.mark.parametrize("bits", [4, 8])
def test_wide_affine_window_equals_single_row_decode(rows, bits):
    weights = mx.random.normal((256, 512), key=mx.random.key(19)).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(weights, group_size=64, bits=bits)
    x = mx.random.normal((1, rows, 512), key=mx.random.key(23)).astype(mx.bfloat16)
    want = mx.concatenate([
        mx.quantized_matmul(
            x[:, i:i + 1], packed, scales, biases,
            transpose=True, group_size=64, bits=bits,
        ) for i in range(rows)
    ], axis=1)
    with exact.row_exact_verify_scope(rows=rows, required=True):
        got = exact.row_exact_qmv(
            x, packed, scales, biases, group_size=64, bits=bits,
        )
    assert got is not None
    assert mx.array_equal(got, want).item()


def test_dense_codebook_wide_verify_equals_each_decode_row():
    projection = dense.TQLinear(512, 256, 2)
    projection.tq2_packed = mx.random.randint(
        0, 2**31 - 1, projection.tq2_packed.shape, key=mx.random.key(31)
    ).astype(mx.uint32)
    projection.tq2_scales = mx.full((1, 256), 0.01, dtype=mx.float16)
    projection.row_exact_verify = True
    x = mx.random.normal((1, 16, 512), key=mx.random.key(32)).astype(mx.bfloat16)
    want = mx.concatenate([projection(x[:, i:i + 1]) for i in range(16)], axis=1)
    with exact.row_exact_verify_scope(rows=16, required=True):
        got = projection(x)
    assert mx.array_equal(got, want).item()
