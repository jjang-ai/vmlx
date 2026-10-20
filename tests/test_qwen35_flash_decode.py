# SPDX-License-Identifier: Apache-2.0
"""Long-context Qwen3.5 decode / verify attention routing (vmlx_engine/metal/qwen35_flash_decode.py).

Greedy native MTP must equal AR, so every verify row has to be computed exactly like the one-row decode step at the
same position: same kernel (chosen from the row's own key count) and, on the flash kernel, the same chunk size.
"""

import mlx.core as mx
import pytest

from vmlx_engine.metal import qwen35_flash_decode as fd
from vmlx_engine.metal.row_exact_qmv import attend_rows_in_pairs

D = 256
SCALE = D ** -0.5


def _sdpa(q, k, v, m):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE, mask=m)


def _pairs(q, k, v):
    return attend_rows_in_pairs(_sdpa, q, k, v)


def _decode(q, k, v):
    out = fd.decode_attention(q, k, v, SCALE, None)
    return _sdpa(q, k, v, None) if out is None else out


def _qkv(n, rows, heads=32, kv=2, dim=D, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((1, heads, rows, dim)).astype(mx.bfloat16)
    k = mx.random.normal((1, kv, n, dim)).astype(mx.bfloat16)
    v = mx.random.normal((1, kv, n, dim)).astype(mx.bfloat16)
    mx.eval(q, k, v)
    return q, k, v


def test_decode_routes_only_long_ornith_geometry(monkeypatch):
    monkeypatch.setenv("VMLX_QWEN35_FLASH_DECODE_MIN_KEYS", "512")
    q, k, v = _qkv(600, 1)
    assert fd.decode_attention(q, k, v, SCALE, None) is not None
    assert fd.decode_attention(q, k, v, SCALE, "causal") is not None
    assert fd.decode_attention(q, k[:, :, :500], v[:, :, :500], SCALE, None) is None   # below threshold
    assert fd.decode_attention(q, k, v, SCALE, mx.zeros((1, 1, 1, 600))) is None         # array mask
    q8, k8, v8 = _qkv(600, 1, heads=16, kv=2)                                            # GQA 8 (MiniCPM / 35B)
    assert fd.decode_attention(q8, k8, v8, SCALE, None) is None
    q6, k6, v6 = _qkv(600, 1, heads=24, kv=4)                                            # GQA 6 (27B)
    assert fd.decode_attention(q6, k6, v6, SCALE, None) is None
    monkeypatch.setenv("VMLX_QWEN35_FLASH_DECODE_MIN_KEYS", "0")
    assert fd.decode_attention(q, k, v, SCALE, None) is None


def test_decode_flash_matches_mlx_closely(monkeypatch):
    monkeypatch.setenv("VMLX_QWEN35_FLASH_DECODE_MIN_KEYS", "512")
    q, k, v = _qkv(4099, 1)
    got = fd.decode_attention(q, k, v, SCALE, None).astype(mx.float32)
    ref = _sdpa(q, k, v, None).astype(mx.float32)
    assert float(mx.max(mx.abs(got - ref)).item()) < 2e-3


@pytest.mark.parametrize("n, rows, threshold", [
    (1027, 4, 1025),     # rows straddle the threshold: 1024 / 1025 / 1026 / 1027 keys
    (1030, 8, 1025),
    (1000, 3, 1025),     # all below: pairs
    (2050, 5, 1025),     # all above, one chunk class
    (32770, 4, 1025),    # chunk-size step 256 -> 512 between 32,768 and 32,769 keys
    (32772, 8, 1025),
])
def test_every_verify_row_equals_its_decode_step(monkeypatch, n, rows, threshold):
    monkeypatch.setenv("VMLX_QWEN35_FLASH_DECODE_MIN_KEYS", str(threshold))
    q, k, v = _qkv(n, rows, seed=n + rows)
    out = fd.verify_attention(q, k, v, SCALE, _pairs)
    mx.eval(out)
    start = n - rows
    for r in range(rows):
        end = start + r + 1
        one = _decode(q[:, :, r:r + 1], k[:, :, :end], v[:, :, :end])
        assert mx.array_equal(out[:, :, r:r + 1], one).item(), f"row {r} ({end} keys) differs from its decode step"


def test_verify_off_and_other_geometries_keep_pairs(monkeypatch):
    q, k, v = _qkv(1030, 4, heads=16, kv=2)
    monkeypatch.setenv("VMLX_QWEN35_FLASH_DECODE_MIN_KEYS", "512")
    assert mx.array_equal(fd.verify_attention(q, k, v, SCALE, _pairs), _pairs(q, k, v)).item()
    q, k, v = _qkv(1030, 4)
    monkeypatch.setenv("VMLX_QWEN35_FLASH_DECODE_MIN_KEYS", "off")
    assert mx.array_equal(fd.verify_attention(q, k, v, SCALE, _pairs), _pairs(q, k, v)).item()
