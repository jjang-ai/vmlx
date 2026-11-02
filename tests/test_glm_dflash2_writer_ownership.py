"""Native GLM disk writes must not evaluate MLX graphs on the IO worker."""

import threading

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from vmlx_engine import glm_dflash2 as runtime
from vmlx_engine.models.glm5_next.glm5_next import Glm5KDACache


def snapshot():
    cache = Glm5KDACache()
    cache.cache = [mx.ones((1, 2, 3), dtype=mx.bfloat16) for _ in range(3)]
    cache.cache.append(mx.ones((1, 2, 3, 3)))
    # Non-contiguous BF16 and pending arithmetic must survive without widening
    # or converting BF16 through float16, which would overflow these values.
    hidden = mx.array([[1e30, -1e30], [1.0, -1.0]], dtype=mx.bfloat16).T
    logits = mx.arange(6, dtype=mx.float32).reshape(1, 1, 6) * 3
    return runtime._PrefixSnapshot([cache], hidden, logits)


def test_serialization_stays_on_producer_and_restart_is_lossless(tmp_path, monkeypatch):
    producer = threading.get_ident()
    serialized_on = []
    original = mx.save_safetensors

    def observe(*args, **kwargs):
        serialized_on.append(threading.get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(mx, "save_safetensors", observe)
    ids = (1, 2**40, 3)
    snap = snapshot()
    disk = runtime.PrefixDiskTier(tmp_path, 2**20)
    try:
        assert disk.put(ids, snap).result(timeout=5)["durable"]
    finally:
        disk.close()
    assert serialized_on == [producer], "MLX serialization ran on the disk worker"

    restarted = runtime.PrefixDiskTier(tmp_path, 2**20)
    try:
        count, restored = restarted.lookup(ids)
        assert count == len(ids)
        assert restored.hidden.dtype == mx.bfloat16
        np.testing.assert_array_equal(
            np.asarray(restored.hidden.astype(mx.float32)),
            np.asarray(snap.hidden.astype(mx.float32)),
        )
        np.testing.assert_array_equal(np.asarray(restored.logits), np.asarray(snap.logits))
        for actual, expected in zip(restored.cache[0].state, snap.cache[0].state):
            assert actual.dtype == expected.dtype
            assert bool(mx.array_equal(actual, expected))
    finally:
        restarted.close()


def test_serialization_failure_settles_receipt_without_failing_generation(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("serialization unavailable")

    monkeypatch.setattr(mx, "save_safetensors", fail)
    disk = runtime.PrefixDiskTier(tmp_path, 2**20)
    try:
        receipt = disk.put((1, 2, 3), snapshot())
        result = receipt.result(timeout=5)
        assert result["durable"] is False
        assert result["outcome"] == "failed"
        assert result["retained_tokens"] == 0
    finally:
        disk.close()
