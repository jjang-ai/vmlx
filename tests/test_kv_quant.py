# SPDX-License-Identifier: Apache-2.0
"""Tests for KV cache quantization round-trip.

Tests the quantize/dequantize operations used by the scheduler's
_wrap_make_cache_quantized, _quantize_cache_for_storage, and
_dequantize_cache_for_use methods.

These tests use mlx directly (not the scheduler) to verify that
quantization round-trips are within acceptable tolerance.
"""

import pytest

# Check availability before all tests
try:
    from mlx_lm.models.cache import QuantizedKVCache
    HAS_QUANTIZED_KV_CACHE = True
except ImportError:
    HAS_QUANTIZED_KV_CACHE = False

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


needs_mlx = pytest.mark.skipif(not HAS_MLX, reason="mlx not available")
needs_quantized_kv = pytest.mark.skipif(
    not HAS_QUANTIZED_KV_CACHE,
    reason="QuantizedKVCache not available in this mlx-lm version",
)


def _evaluate(tensor):
    """Materialize an MLX lazy tensor."""
    mx.eval(tensor)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestKVQuantization:
    """KV cache quantization availability and round-trip tests."""

    @needs_mlx
    def test_quantized_kv_cache_available(self):
        """Check if QuantizedKVCache class exists in mlx_lm."""
        # This test documents whether the feature is available.
        # If it passes, the other tests can run.
        from mlx_lm.models.cache import QuantizedKVCache as QKV
        assert QKV is not None
        # Verify it can be instantiated
        qkv = QKV(group_size=64, bits=8)
        assert qkv is not None

    @needs_mlx
    @needs_quantized_kv
    def test_q8_round_trip_tolerance(self):
        """8-bit quantize/dequantize round-trip stays within tolerance."""
        bits = 8
        group_size = 64
        head_dim = 128

        # Create a realistic KV tensor: (batch, heads, seq_len, head_dim)
        shape = (1, 4, 16, head_dim)
        original = mx.random.normal(shape)
        _evaluate(original)

        # Quantize
        quantized_data, scales, biases = mx.quantize(
            original, group_size=group_size, bits=bits
        )

        # Dequantize
        restored = mx.dequantize(
            quantized_data, scales, biases, group_size=group_size, bits=bits
        )
        _evaluate(restored)

        # Shape must match
        assert restored.shape == original.shape

        # 8-bit should be quite accurate -- max abs error < 0.5
        diff = mx.abs(original - restored)
        _evaluate(diff)
        max_error = mx.max(diff).item()
        assert max_error < 0.5, f"8-bit max error {max_error} exceeds 0.5 tolerance"

    @needs_mlx
    @needs_quantized_kv
    def test_q4_round_trip_tolerance(self):
        """4-bit quantize/dequantize round-trip stays within tolerance."""
        bits = 4
        group_size = 64
        head_dim = 128

        shape = (1, 4, 16, head_dim)
        original = mx.random.normal(shape)
        _evaluate(original)

        quantized_data, scales, biases = mx.quantize(
            original, group_size=group_size, bits=bits
        )

        restored = mx.dequantize(
            quantized_data, scales, biases, group_size=group_size, bits=bits
        )
        _evaluate(restored)

        assert restored.shape == original.shape

        # 4-bit has more quantization error -- max abs error < 2.0
        diff = mx.abs(original - restored)
        _evaluate(diff)
        max_error = mx.max(diff).item()
        assert max_error < 2.0, f"4-bit max error {max_error} exceeds 2.0 tolerance"

    @needs_mlx
    @needs_quantized_kv
    def test_group_size_validation(self):
        """Verify head_dim % group_size == 0 check works.

        mx.quantize requires group_size to evenly divide the last dimension.
        If it doesn't, mlx raises an error.
        """
        head_dim = 96  # Not divisible by 64

        shape = (1, 4, 16, head_dim)
        tensor = mx.random.normal(shape)

        # group_size=64 should fail because 96 % 64 != 0
        with pytest.raises(Exception):
            result = mx.quantize(tensor, group_size=64, bits=8)
            # Force evaluation in case quantize is lazy
            _evaluate(result[0])

        # group_size=32 should work because 96 % 32 == 0
        quantized_data, scales, biases = mx.quantize(
            tensor, group_size=32, bits=8
        )
        _evaluate(quantized_data)
        assert quantized_data is not None
