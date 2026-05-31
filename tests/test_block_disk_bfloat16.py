"""Tests for block disk store bfloat16/numpy serialization.

Verifies that bfloat16 KV cache data and numpy arrays survive the
serialize -> disk -> deserialize round-trip correctly.
"""
import pytest
import tempfile
import shutil
import sqlite3
import time
from pathlib import Path

try:
    import mlx.core as mx
    import numpy as np
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


@pytest.fixture
def disk_store():
    """Create a temporary BlockDiskStore for testing."""
    from vmlx_engine.block_disk_store import BlockDiskStore
    tmpdir = tempfile.mkdtemp(prefix="test_block_disk_")
    store = BlockDiskStore(tmpdir, max_size_gb=1.0)
    yield store
    store.shutdown()
    shutil.rmtree(tmpdir, ignore_errors=True)


def _make_kv_block(seq_len=64, n_heads=2, head_dim=256, dtype=mx.bfloat16):
    """Create a simple KV-only block cache_data (10 KV layers, 30 skip)."""
    cache_data = []
    for i in range(40):
        if i % 4 == 3:  # 10 KV layers at positions 3,7,11,...
            keys = mx.random.normal((1, n_heads, seq_len, head_dim)).astype(dtype)
            values = mx.random.normal((1, n_heads, seq_len, head_dim)).astype(dtype)
            mx.eval(keys, values)  # noqa: S307 -- mlx tensor materialization
            cache_data.append(("kv", keys, values))
        else:
            cache_data.append(("skip",))
    return cache_data


def _make_numpy_kv_block(seq_len=64, n_heads=2, head_dim=256):
    """Create a KV block with numpy arrays (as produced by numpy-safe slicing)."""
    cache_data = []
    for i in range(40):
        if i % 4 == 3:
            keys = np.random.randn(1, n_heads, seq_len, head_dim).astype(np.float16)
            values = np.random.randn(1, n_heads, seq_len, head_dim).astype(np.float16)
            cache_data.append(("kv", keys, values))
        else:
            cache_data.append(("skip",))
    return cache_data


def _make_numpy_rotating_kv_block(seq_len=8, n_heads=2, head_dim=16):
    """Create Gemma4-style rotating KV slices with absolute offset metadata."""
    keys = np.random.randn(1, n_heads, seq_len, head_dim).astype(np.float16)
    values = np.random.randn(1, n_heads, seq_len, head_dim).astype(np.float16)
    return [("rotating_kv", keys, values, 4096, 0, 20, 8)]


class TestSerializeBlock:
    """Test _serialize_block handles various tensor types."""

    def test_bfloat16_kv_block(self):
        from vmlx_engine.block_disk_store import _serialize_block
        cache_data = _make_kv_block(dtype=mx.bfloat16)
        tensors, dtype, num_layers = _serialize_block(cache_data)
        assert num_layers == 40
        assert isinstance(tensors["layer_3_keys"], mx.array)
        assert tensors["layer_3_keys"].dtype == mx.bfloat16

    def test_float16_kv_block(self):
        from vmlx_engine.block_disk_store import _serialize_block
        cache_data = _make_kv_block(dtype=mx.float16)
        tensors, dtype, num_layers = _serialize_block(cache_data)
        assert num_layers == 40
        assert tensors["layer_3_keys"].dtype == mx.float16

    def test_numpy_kv_block(self):
        from vmlx_engine.block_disk_store import _serialize_block
        cache_data = _make_numpy_kv_block()
        tensors, dtype, num_layers = _serialize_block(cache_data)
        assert num_layers == 40
        assert isinstance(tensors["layer_3_keys"], np.ndarray)

    def test_numpy_rotating_kv_block_preserves_absolute_window_state(self):
        from vmlx_engine.block_disk_store import _deserialize_block, _serialize_block

        cache_data = _make_numpy_rotating_kv_block()
        tensors, dtype, num_layers = _serialize_block(cache_data)

        assert num_layers == 1
        assert dtype == "rotating_kv"
        assert isinstance(tensors["layer_0_keys"], np.ndarray)
        assert int(tensors["layer_0_max_size"].item()) == 4096
        assert int(tensors["layer_0_keep"].item()) == 0
        assert int(tensors["layer_0_offset"].item()) == 20
        assert int(tensors["layer_0_idx"].item()) == 8

        restored = _deserialize_block(dict(tensors), dtype)
        assert restored[0][0] == "rotating_kv"
        assert restored[0][3:] == (4096, 0, 20, 8)


class TestEndToEndRoundTrip:
    """Test full write -> read round-trips through the disk store."""

    def test_roundtrip_bfloat16(self, disk_store):
        """bfloat16 data should survive write_block_async -> read_block."""
        import time
        cache_data = _make_kv_block(seq_len=8, dtype=mx.bfloat16)
        original_keys = cache_data[3][1]
        assert original_keys.dtype == mx.bfloat16

        block_hash = b"\x10" * 16
        disk_store.write_block_async(block_hash, cache_data, 8)
        time.sleep(1.0)

        # Read back
        restored = disk_store.read_block(block_hash)
        assert restored is not None, "Block not found after write"

        kv_layers = [d for d in restored if d[0] == "kv"]
        skip_layers = [d for d in restored if d[0] == "skip"]
        assert len(kv_layers) == 10
        assert len(skip_layers) == 30

        # Verify dtype restored to bfloat16
        restored_keys = kv_layers[0][1]
        assert restored_keys.dtype == mx.bfloat16, (
            f"Expected bfloat16, got {restored_keys.dtype}"
        )
        assert restored_keys.shape == original_keys.shape

    def test_roundtrip_numpy(self, disk_store):
        """numpy KV data should survive write -> read round-trip."""
        import time
        cache_data = _make_numpy_kv_block(seq_len=8)
        block_hash = b"\x11" * 16
        disk_store.write_block_async(block_hash, cache_data, 8)
        time.sleep(1.0)

        restored = disk_store.read_block(block_hash)
        assert restored is not None, "Block not found after write"

        kv_layers = [d for d in restored if d[0] == "kv"]
        assert len(kv_layers) == 10
        assert isinstance(kv_layers[0][1], mx.array)


class TestWriteBlockAsync:
    """Test end-to-end write through the async path."""

    def test_bfloat16_block_writes(self, disk_store):
        """bfloat16 KV block should write without std::bad_cast."""
        import time
        cache_data = _make_kv_block(seq_len=8, dtype=mx.bfloat16)
        block_hash = b"\x01" * 16
        disk_store.write_block_async(block_hash, cache_data, 8)
        time.sleep(1.0)
        file_path = disk_store._hash_to_path(block_hash.hex())
        assert file_path.exists(), "Block file not written"

    def test_numpy_block_writes(self, disk_store):
        """numpy KV block should write without errors."""
        import time
        cache_data = _make_numpy_kv_block(seq_len=8)
        block_hash = b"\x02" * 16
        disk_store.write_block_async(block_hash, cache_data, 8)
        time.sleep(1.0)
        file_path = disk_store._hash_to_path(block_hash.hex())
        assert file_path.exists(), "Block file not written"

    def test_numpy_rotating_kv_block_writes_and_counts_l2_tokens(self, disk_store):
        """Mixed-SWA rotating KV write-through must reach endpoint-visible stats."""
        import time

        cache_data = _make_numpy_rotating_kv_block(seq_len=8)
        block_hash = b"\x03" * 16
        disk_store.write_block_async(block_hash, cache_data, 8)
        time.sleep(1.0)

        file_path = disk_store._hash_to_path(block_hash.hex())
        stats = disk_store.get_stats()

        assert file_path.exists(), "Rotating KV block file not written"
        assert stats["blocks_on_disk"] == 1
        assert stats["total_tokens_on_disk"] == 8
        assert stats["disk_writes"] == 1

    def test_stats_include_persistent_token_count(self, disk_store):
        """Block L2 stats must report tokens persisted on disk, not only blocks."""
        now = time.time()
        conn = sqlite3.connect(str(disk_store._db_path))
        try:
            conn.execute(
                "INSERT INTO blocks "
                "(block_hash, file_name, num_tokens, num_layers, dtype, "
                "file_size, created_at, last_accessed, access_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("block-a", "block-a.safetensors", 64, 2, "float16", 1024, now, now, 1),
            )
            conn.execute(
                "INSERT INTO blocks "
                "(block_hash, file_name, num_tokens, num_layers, dtype, "
                "file_size, created_at, last_accessed, access_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("block-b", "block-b.safetensors", 32, 2, "float16", 512, now, now, 1),
            )
            conn.commit()
        finally:
            conn.close()

        stats = disk_store.get_stats()

        assert stats["total_tokens_on_disk"] == 96
        assert stats["total_cached_tokens"] == 96
