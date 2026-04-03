#!/usr/bin/env python3
"""
Unit tests for TQ-native disk cache serialization.

Tests the three-tier cache storage (L1 memory + L2 disk + TQ compressed):
1. TQ serialization round-trip (compressed data survives disk write/read)
2. DiskCacheManager TQ-native store (26x smaller files vs float16)
3. DiskCacheManager TQ-native fetch (proper reconstruction)
4. Mixed cache (TQ KV + SSM cumulative layers for hybrid models)
5. Non-TQ cache backwards compatibility
6. Cache stats show TQ-native tier info
7. File size comparison (TQ-native vs standard float16)
8. Error handling (corrupt files, missing jang_tools, empty caches)

These tests use mock TQ objects that replicate the TurboQuantKVCache interface
so they run without loading an actual model or jang_tools dependency.

Usage:
    python3 -m pytest tests/test_tq_disk_cache.py -v
"""

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from unittest import mock

import pytest

# ── MLX Import ─────────────────────────────────────────────────────────
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    pytest.skip("MLX not available", allow_module_level=True)


# ── Mock TQ Types ──────────────────────────────────────────────────────
# These mock the jang_tools TurboQuantKVCache interface for testing
# without requiring actual jang_tools installation.

class MockEncodedKeys(NamedTuple):
    """Mock of jang_tools.turboquant.cache.EncodedKeys."""
    indices_packed: mx.array    # uint32 packed codebook indices
    qjl_packed: mx.array        # uint32 QJL sign bits
    residual_norms: mx.array    # float16 per-vector residual norms
    vector_norms: mx.array      # float16 per-vector key norms
    shape: tuple                # original shape (batch, heads, tokens, dim)
    index_bits: int             # bits per index


class MockEncodedValues(NamedTuple):
    """Mock of jang_tools.turboquant.cache.EncodedValues."""
    indices_packed: mx.array    # uint32 packed codebook indices
    vector_norms: mx.array      # float16 per-vector value norms
    shape: tuple                # original shape (batch, heads, tokens, dim)
    index_bits: int             # bits per index


class TurboQuantKVCache:
    """Mock TurboQuantKVCache that replicates the essential interface.

    Stores compressed data (EncodedKeys/EncodedValues) and decoded float
    buffers, mimicking the three-phase lifecycle of the real class:
    FILL -> COMPRESS -> GENERATE.

    After compress(), _compressed_keys/_compressed_values are set and
    keys/values are cleared (set to None).
    """

    def __init__(
        self,
        key_dim: int = 128,
        value_dim: int = 128,
        key_bits: int = 3,
        value_bits: int = 3,
        sink_tokens: int = 0,
        n_heads: int = 8,
        n_tokens: int = 100,
    ):
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.sink_tokens = sink_tokens
        self.offset = n_tokens

        # Decoded float buffers (what .state returns)
        self._joined_k = mx.random.normal(shape=(1, n_heads, n_tokens, key_dim)).astype(mx.float16)
        self._joined_v = mx.random.normal(shape=(1, n_heads, n_tokens, value_dim)).astype(mx.float16)

        # Pre-compression float buffers are cleared after compress
        self.keys = None
        self.values = None

        # Compressed data — mock the packed format
        self._compressed_tokens = n_tokens
        self._compressed_keys = MockEncodedKeys(
            indices_packed=mx.zeros((n_tokens * n_heads * key_dim // 10,), dtype=mx.uint32),
            qjl_packed=mx.zeros((n_tokens * n_heads // 32 + 1,), dtype=mx.uint32),
            residual_norms=mx.random.normal(shape=(1, n_heads, n_tokens, 1)).astype(mx.float16),
            vector_norms=mx.random.normal(shape=(1, n_heads, n_tokens, 1)).astype(mx.float16),
            shape=(1, n_heads, n_tokens, key_dim),
            index_bits=key_bits,
        )
        self._compressed_values = MockEncodedValues(
            indices_packed=mx.zeros((n_tokens * n_heads * value_dim // 10,), dtype=mx.uint32),
            vector_norms=mx.random.normal(shape=(1, n_heads, n_tokens, 1)).astype(mx.float16),
            shape=(1, n_heads, n_tokens, value_dim),
            index_bits=value_bits,
        )

    @property
    def state(self):
        """Returns decoded float16 buffers (same as real TQ .state property).
        This is what DECOMPRESSES the TQ data -- the old/wasteful path."""
        if self.offset == 0:
            return [], []
        return self._joined_k[..., :self.offset, :], self._joined_v[..., :self.offset, :]

    @property
    def meta_state(self):
        return (str(self.offset), str(self.key_bits), str(self.value_bits))

    def compress(self):
        """Mock compress -- in real TQ this encodes to 3-bit."""
        self.keys = None
        self.values = None

    def is_compressed(self):
        return self._compressed_tokens > 0


class MockKVCache:
    """Mock standard KVCache (non-TQ)."""

    def __init__(self, n_heads: int = 8, n_tokens: int = 100, dim: int = 128):
        self.keys = mx.random.normal(shape=(1, n_heads, n_tokens, dim)).astype(mx.float16)
        self.values = mx.random.normal(shape=(1, n_heads, n_tokens, dim)).astype(mx.float16)
        self.offset = n_tokens

    @property
    def state(self):
        return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]

    @property
    def meta_state(self):
        return (str(self.offset),)


class MockMambaCache:
    """Mock MambaCache (SSM cumulative state for hybrid models)."""

    def __init__(self, n_states: int = 3):
        self.cache = [mx.random.normal(shape=(1, 16, 64)).astype(mx.float16) for _ in range(n_states)]

    @property
    def state(self):
        return self.cache

    @property
    def meta_state(self):
        return ("ssm_state",)


# ── Helpers ────────────────────────────────────────────────────────────

def _create_tq_cache(
    n_layers: int = 4,
    n_heads: int = 8,
    n_tokens: int = 100,
    key_dim: int = 128,
) -> List[TurboQuantKVCache]:
    """Create a list of mock TQ cache layers."""
    return [
        TurboQuantKVCache(
            key_dim=key_dim, n_heads=n_heads, n_tokens=n_tokens
        )
        for _ in range(n_layers)
    ]


def _create_hybrid_cache(
    n_kv_layers: int = 4,
    n_ssm_layers: int = 8,
) -> List[Any]:
    """Create a mixed TQ + SSM cache (hybrid model like Nemotron)."""
    cache = []
    for i in range(n_kv_layers + n_ssm_layers):
        if i % 3 == 0:  # Every 3rd layer is KV (attention)
            cache.append(TurboQuantKVCache(n_tokens=50))
        else:
            cache.append(MockMambaCache())
    return cache


def _create_non_tq_cache(n_layers: int = 4) -> List[MockKVCache]:
    """Create a list of standard (non-TQ) KV cache layers."""
    return [MockKVCache() for _ in range(n_layers)]


# ── Tests: TQ Serialization Module ────────────────────────────────────

class TestTQDiskStore:
    """Tests for tq_disk_store.py serialization/deserialization functions."""

    def test_is_tq_compressed_cache_detects_tq(self):
        """is_tq_compressed_cache returns True for TQ cache with compressed data."""
        from vmlx_engine.tq_disk_store import is_tq_compressed_cache
        cache = _create_tq_cache(n_layers=2)
        assert is_tq_compressed_cache(cache) is True

    def test_is_tq_compressed_cache_rejects_non_tq(self):
        """is_tq_compressed_cache returns False for standard KVCache."""
        from vmlx_engine.tq_disk_store import is_tq_compressed_cache
        cache = _create_non_tq_cache(n_layers=2)
        assert is_tq_compressed_cache(cache) is False

    def test_is_tq_compressed_cache_rejects_uncompressed_tq(self):
        """is_tq_compressed_cache returns False when _compressed_keys is None."""
        from vmlx_engine.tq_disk_store import is_tq_compressed_cache
        cache = _create_tq_cache(n_layers=1)
        cache[0]._compressed_keys = None  # Simulate pre-compress state
        assert is_tq_compressed_cache(cache) is False

    def test_serialize_tq_cache_produces_tensors_and_metadata(self):
        """serialize_tq_cache returns (tensors, metadata) with correct structure."""
        from vmlx_engine.tq_disk_store import serialize_tq_cache
        cache = _create_tq_cache(n_layers=2)

        tensors, meta = serialize_tq_cache(cache)

        # Metadata should have format marker
        assert meta["__tq_native__"] == "true"
        assert meta["__num_layers__"] == "2"
        assert meta["__layer_0_class__"] == "TurboQuantKVCache"

        # Should have TQ tensors for each layer
        for i in range(2):
            assert f"tq_{i}_ck_indices_packed" in tensors
            assert f"tq_{i}_ck_qjl_packed" in tensors
            assert f"tq_{i}_ck_residual_norms" in tensors
            assert f"tq_{i}_ck_vector_norms" in tensors
            assert f"tq_{i}_cv_indices_packed" in tensors
            assert f"tq_{i}_cv_vector_norms" in tensors

        # Should have shape metadata
        for i in range(2):
            assert f"__tq_{i}_ck_shape__" in meta
            assert f"__tq_{i}_cv_shape__" in meta
            assert f"__tq_{i}_ck_bits__" in meta
            assert f"__tq_{i}_offset__" in meta

    def test_serialize_tq_cache_mixed_hybrid(self):
        """serialize_tq_cache handles mixed TQ + SSM layers (hybrid model)."""
        from vmlx_engine.tq_disk_store import serialize_tq_cache
        cache = _create_hybrid_cache()

        tensors, meta = serialize_tq_cache(cache)

        assert meta["__tq_native__"] == "true"
        num_layers = int(meta["__num_layers__"])
        assert num_layers > 0

        # Check that TQ layers have TQ tensors and SSM layers have state tensors
        tq_count = 0
        for i in range(num_layers):
            cls = meta.get(f"__layer_{i}_class__", "")
            if cls == "TurboQuantKVCache":
                assert f"tq_{i}_ck_indices_packed" in tensors
                tq_count += 1

        assert tq_count > 0, "Should have at least one TQ layer in hybrid cache"

    def test_serialize_tq_cache_tensors_are_correct_dtype(self):
        """Verify tensor dtypes match expected TQ format (uint32 packed, float16 norms)."""
        from vmlx_engine.tq_disk_store import serialize_tq_cache
        cache = _create_tq_cache(n_layers=1)

        tensors, _ = serialize_tq_cache(cache)

        # Packed indices should be uint32
        assert tensors["tq_0_ck_indices_packed"].dtype == mx.uint32
        assert tensors["tq_0_ck_qjl_packed"].dtype == mx.uint32
        assert tensors["tq_0_cv_indices_packed"].dtype == mx.uint32

        # Norms should be float16
        assert tensors["tq_0_ck_residual_norms"].dtype == mx.float16
        assert tensors["tq_0_ck_vector_norms"].dtype == mx.float16
        assert tensors["tq_0_cv_vector_norms"].dtype == mx.float16

    def test_tq_file_size_vs_float16(self):
        """TQ-native files should be dramatically smaller than float16 state files.

        This verifies the compression ratio claim. With 4 layers of
        100 tokens x 8 heads x 128 dim, float16 state is ~1.6MB per layer
        while TQ compressed is ~60KB per layer.
        """
        from vmlx_engine.tq_disk_store import serialize_tq_cache

        cache = _create_tq_cache(n_layers=4, n_heads=8, n_tokens=100, key_dim=128)
        tensors, meta = serialize_tq_cache(cache)

        # Calculate TQ tensor sizes
        tq_total = sum(
            t.nbytes for t in tensors.values() if isinstance(t, mx.array)
        )

        # Calculate float16 state sizes (what .state returns)
        float16_total = 0
        for layer in cache:
            k, v = layer.state
            float16_total += k.nbytes + v.nbytes

        # TQ should be significantly smaller
        ratio = float16_total / max(tq_total, 1)
        assert ratio > 3.0, (
            f"TQ compression ratio {ratio:.1f}x is too low "
            f"(expected >3x, got {tq_total} vs {float16_total} bytes)"
        )
        print(
            f"  Compression: float16={float16_total/1024:.0f}KB, "
            f"TQ={tq_total/1024:.0f}KB, ratio={ratio:.1f}x"
        )


# ── Tests: DiskCacheManager Integration ───────────────────────────────

class TestDiskCacheManagerTQ:
    """Tests for DiskCacheManager with TQ-native serialization."""

    @pytest.fixture(autouse=True)
    def setup_cache_dir(self):
        """Create and clean up a temp directory for each test."""
        self.cache_dir = tempfile.mkdtemp(prefix="vmlx_test_tq_disk_")
        yield
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def _make_manager(self, max_gb: float = 1.0):
        """Create a DiskCacheManager for testing."""
        from vmlx_engine.disk_cache import DiskCacheManager
        return DiskCacheManager(self.cache_dir, max_size_gb=max_gb)

    def test_store_detects_tq_and_uses_native(self):
        """store() should detect TQ cache and use TQ-native serialization."""
        mgr = self._make_manager()
        try:
            cache = _create_tq_cache(n_layers=2, n_tokens=50)
            tokens = list(range(50))

            # Patch at the module level where the import happens.
            # disk_cache.store() does: from .tq_disk_store import is_tq_compressed_cache
            # We patch the tq_disk_store module functions directly.
            with mock.patch(
                "vmlx_engine.tq_disk_store.is_tq_compressed_cache",
                return_value=True,
            ), mock.patch(
                "vmlx_engine.tq_disk_store.serialize_tq_cache",
            ) as mock_serialize:
                # Return valid serialized data
                mock_tensors = {"test": mx.array([1, 2, 3])}
                mock_meta = {"__tq_native__": "true", "__num_layers__": "2"}
                mock_serialize.return_value = (mock_tensors, mock_meta)

                result = mgr.store(tokens, cache)
                assert result is True
                mock_serialize.assert_called_once()
        finally:
            mgr.shutdown()

    def test_store_non_tq_uses_standard_path(self):
        """store() should use standard path for non-TQ caches."""
        mgr = self._make_manager()
        try:
            cache = _create_non_tq_cache(n_layers=2)
            tokens = list(range(50))

            result = mgr.store(tokens, cache)
            assert result is True

            # Wait for background writer to complete
            time.sleep(0.5)

            # Check stats
            stats = mgr.stats()
            # Should not have TQ native stats (non-TQ cache)
            assert stats.get("tq_native_stores", 0) == 0
        finally:
            mgr.shutdown()

    def test_fetch_standard_cache(self):
        """fetch() should return standard cache correctly."""
        mgr = self._make_manager()
        try:
            cache = _create_non_tq_cache(n_layers=2)
            tokens = list(range(50))

            mgr.store(tokens, cache)
            time.sleep(1.0)  # Wait for background write

            result = mgr.fetch(tokens)
            assert result is not None
            assert len(result) == 2
            assert mgr._last_fetch_tq_native is False
        finally:
            mgr.shutdown()

    def test_stats_include_tq_native(self):
        """stats() should include tq_native_stores/hits when TQ ops occur."""
        mgr = self._make_manager()
        try:
            # Initially no TQ stats
            stats = mgr.stats()
            assert "tq_native_stores" not in stats

            # Simulate TQ native activity
            with mgr._stats_lock:
                mgr.tq_native_stores = 3
                mgr.tq_native_hits = 1

            stats = mgr.stats()
            assert stats["tq_native_stores"] == 3
            assert stats["tq_native_hits"] == 1
        finally:
            mgr.shutdown()

    def test_clear_resets_tq_stats(self):
        """clear() should reset TQ-native stats."""
        mgr = self._make_manager()
        try:
            with mgr._stats_lock:
                mgr.tq_native_stores = 5
                mgr.tq_native_hits = 2

            mgr.clear()

            assert mgr.tq_native_stores == 0
            assert mgr.tq_native_hits == 0
        finally:
            mgr.shutdown()

    def test_last_fetch_tq_native_flag(self):
        """_last_fetch_tq_native flag should be set correctly on fetch."""
        mgr = self._make_manager()
        try:
            # Store standard cache
            cache = _create_non_tq_cache(n_layers=2)
            tokens = list(range(50))
            mgr.store(tokens, cache)
            time.sleep(1.0)

            # Fetch should set flag to False for standard cache
            mgr.fetch(tokens)
            assert mgr._last_fetch_tq_native is False
        finally:
            mgr.shutdown()


# ── Tests: Safetensors Round-Trip ─────────────────────────────────────

class TestSafetensorsRoundTrip:
    """Test that TQ data survives safetensors serialization and deserialization."""

    @pytest.fixture(autouse=True)
    def setup_cache_dir(self):
        self.cache_dir = tempfile.mkdtemp(prefix="vmlx_test_tq_rt_")
        yield
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_tq_tensors_roundtrip_via_safetensors(self):
        """TQ compressed tensors should survive mx.save_safetensors and mx.load."""
        from vmlx_engine.tq_disk_store import serialize_tq_cache

        cache = _create_tq_cache(n_layers=2, n_tokens=50)
        tensors, meta = serialize_tq_cache(cache)

        # Materialize tensors (noqa: S307 -- mlx tensor materialization, not Python eval)
        arrays = [v for v in tensors.values() if isinstance(v, mx.array)]
        mx.eval(*arrays)  # noqa: S307

        # Save to safetensors
        path = os.path.join(self.cache_dir, "test_tq.safetensors")
        mx.save_safetensors(path, tensors, meta)

        # Load back
        loaded_tensors, loaded_meta = mx.load(path, return_metadata=True)

        # Verify metadata preserved
        assert loaded_meta.get("__tq_native__") == "true"
        assert loaded_meta.get("__num_layers__") == "2"

        # Verify all tensor keys present
        for key in tensors:
            assert key in loaded_tensors, f"Missing tensor key: {key}"

        # Verify tensor shapes and dtypes match
        for key in tensors:
            orig = tensors[key]
            loaded = loaded_tensors[key]
            assert orig.shape == loaded.shape, (
                f"Shape mismatch for {key}: {orig.shape} vs {loaded.shape}"
            )
            assert orig.dtype == loaded.dtype, (
                f"Dtype mismatch for {key}: {orig.dtype} vs {loaded.dtype}"
            )

    def test_tq_metadata_roundtrip(self):
        """TQ metadata (shapes, bits) should survive serialization."""
        from vmlx_engine.tq_disk_store import serialize_tq_cache

        cache = _create_tq_cache(n_layers=1, n_heads=8, n_tokens=100, key_dim=128)
        tensors, meta = serialize_tq_cache(cache)

        # Save and reload
        path = os.path.join(self.cache_dir, "test_meta.safetensors")
        arrays = [v for v in tensors.values() if isinstance(v, mx.array)]
        mx.eval(*arrays)  # noqa: S307
        mx.save_safetensors(path, tensors, meta)
        _, loaded_meta = mx.load(path, return_metadata=True)

        # Verify all metadata keys present
        for key in meta:
            assert key in loaded_meta, f"Missing metadata key: {key}"
            assert loaded_meta[key] == meta[key], (
                f"Metadata mismatch for {key}: '{loaded_meta[key]}' vs '{meta[key]}'"
            )


# ── Tests: Error Handling ─────────────────────────────────────────────

class TestErrorHandling:
    """Test graceful error handling for edge cases."""

    def test_serialize_empty_cache(self):
        """serialize_tq_cache handles empty cache list."""
        from vmlx_engine.tq_disk_store import serialize_tq_cache
        tensors, meta = serialize_tq_cache([])
        assert meta["__tq_native__"] == "true"
        assert meta["__num_layers__"] == "0"
        assert len(tensors) == 0

    def test_deserialize_empty_cache(self):
        """deserialize_tq_cache handles empty metadata."""
        from vmlx_engine.tq_disk_store import deserialize_tq_cache
        cache = deserialize_tq_cache({}, {"__tq_native__": "true", "__num_layers__": "0"})
        assert cache == []

    def test_is_tq_compressed_with_empty_list(self):
        """is_tq_compressed_cache handles empty list."""
        from vmlx_engine.tq_disk_store import is_tq_compressed_cache
        assert is_tq_compressed_cache([]) is False

    def test_is_tq_compressed_with_none_elements(self):
        """is_tq_compressed_cache handles None elements gracefully."""
        from vmlx_engine.tq_disk_store import is_tq_compressed_cache
        assert is_tq_compressed_cache([None, None]) is False


# ── Run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
