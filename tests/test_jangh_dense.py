"""Dense JANGH (jangtq2 on a plain Linear, e.g. Qwen3.8-27B JANGH2 MLP): entries are recognised, installed as TQLinear
before quantization, validated by the payload contract, and ignored by the routed-expert contract."""
import mlx.core as mx
import mlx.nn as nn
import pytest

from vmlx_engine.jangh import dense as D
from vmlx_engine.jangh.payload import expected_payload


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(64, 96, bias=False)
        self.up_proj = nn.Linear(64, 96, bias=False)
        self.down_proj = nn.Linear(96, 64, bias=False)


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [_Mlp(), _Mlp()]


def _config():
    q = {"group_size": 64, "bits": 8}
    for i in range(2):
        for p in ("gate_proj", "up_proj", "down_proj"):
            q[f"layers.{i}.{p}"] = {"mode": "jangtq2", "bits": 2, "rotation": "hadamard32"}
    q["layers.0.switch_mlp.gate_proj"] = {"mode": "jangtq2", "bits": 2}       # routed entry: not dense
    q["layers.0.self_attn.q_proj"] = {"group_size": 64, "bits": 8}            # affine: not jangtq2
    return {"quantization": q}


def test_dense_entries_select_only_non_routed_jangtq2_paths():
    entries = D.dense_entries(_config())
    assert set(entries) == {f"layers.{i}.{p}" for i in range(2) for p in ("gate_proj", "up_proj", "down_proj")}


def test_install_replaces_each_dense_entry_with_tqlinear_of_matching_shape():
    model = _Toy()
    assert D.install_jangh_dense(model, _config()) == 6
    down = model.layers[1].down_proj
    assert isinstance(down, D.TQLinear) and down.is_jangtq2_dense
    assert (down.input_dims, down.output_dims, down.bits) == (96, 64, 2)
    assert down.tq2_packed.shape == (1, 64, 96 * 2 // 32) and down.tq2_scales.shape == (1, 64)
    # installing twice is a no-op (already TQLinear)
    assert D.install_jangh_dense(model, _config()) == 0


def test_tqlinear_accepts_its_own_quantize_entry_and_rejects_others():
    t = D.TQLinear(64, 32, 2)
    assert t.to_quantized(mode="jangtq2", bits=2) is t
    with pytest.raises(ValueError):
        t.to_quantized(mode="affine", bits=4)


def test_entry_matching_no_linear_fails_closed():
    with pytest.raises(ValueError):
        D.install_jangh_dense(_Toy(), {"quantization": {"layers.0.missing_proj": {"mode": "jangtq2", "bits": 2}}})


def test_payload_contract_expects_dense_tensors():
    model = _Toy()
    D.install_jangh_dense(model, _config())
    expected = expected_payload(model)
    assert expected["layers.0.gate_proj.tq2_packed"] == ((1, 96, 64 * 2 // 32), "uint32")
    assert expected["layers.0.gate_proj.tq2_scales"] == ((1, 96), "float16")


def test_tqlinear_forward_shapes_decode_and_prefill_paths():
    t = D.TQLinear(64, 32, 2)
    for rows in (1, 3, D.SORT_THRESHOLD + 5):            # gather_qmv below the threshold, sorted qmm above
        y = t(mx.ones((1, rows, 64), dtype=mx.float16))
        mx.eval(y)
        assert y.shape == (1, rows, 32) and y.dtype == mx.float16


def test_tqlinear_row_exact_verify_keeps_verify_rows_on_the_decode_kernel(monkeypatch):
    """Opt-in flag: inside a row-exact verify scope every row takes the per-row
    decode kernel (bit-equal to its one-row call); outside it, or without the
    flag, wide windows keep the wide-window kernel."""
    from vmlx_engine.metal.row_exact_qmv import row_exact_verify_scope

    monkeypatch.setenv("VMLX_ROW_EXACT_VERIFY_QMV", "1")
    sorted_calls = []
    real_sorted = D.K.gather_qmm_dense
    monkeypatch.setattr(D.K, "gather_qmm_dense", lambda *a, **k: sorted_calls.append(1) or real_sorted(*a, **k))
    t = D.TQLinear(256, 64, 2)
    t.tq2_packed = mx.random.randint(0, 2**31 - 1, t.tq2_packed.shape, key=mx.random.key(1)).astype(mx.uint32)
    t.tq2_scales = mx.random.uniform(0.01, 0.05, t.tq2_scales.shape, key=mx.random.key(2)).astype(mx.float16)
    x = (mx.random.normal((1, 8, 256), key=mx.random.key(3)) + 1.5).astype(mx.float16)
    singles = mx.concatenate([t(x[:, r:r + 1]) for r in range(8)], axis=1)

    with row_exact_verify_scope():
        t(x)                                   # flag off: sorted path even in scope
        assert len(sorted_calls) == 1
        t.row_exact_verify = True
        for rows in range(D.SORT_THRESHOLD, 9):
            assert bool(mx.array_equal(t(x[:, :rows]), singles[:, :rows]).item())
        assert len(sorted_calls) == 1
    t(x)                                       # flag on, outside the scope: sorted path
    assert len(sorted_calls) == 2
