import pytest


@pytest.mark.parametrize("resume_rows", [1, 3])
def test_projected_checkpoint_keeps_chronological_tail_and_absolute_offset(resume_rows):
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.cache import RotatingKVCache

    from vmlx_engine.dspark.inkling_drafter import checkpoint_cache

    keys = mx.arange(10, dtype=mx.bfloat16).reshape(1, 1, 10, 1)
    values = keys + 10
    live = RotatingKVCache(max_size=4)
    live.update_and_fetch(keys[..., :4, :], values[..., :4, :])
    live.update_and_fetch(keys[..., 4:7, :], values[..., 4:7, :])
    assert live.offset == 7 and live.state[0].shape[2] == 6

    restored, = checkpoint_cache([live])
    assert restored is not live
    assert restored.offset == 7 and restored._idx == 4
    assert live.state[0].shape[2] == 6
    assert mx.array_equal(restored.state[0], keys[..., 3:7, :]).item()
    assert mx.array_equal(restored.state[1], values[..., 3:7, :]).item()

    fresh = (keys[..., 7:7 + resume_rows, :], values[..., 7:7 + resume_rows, :])
    expected = live.update_and_fetch(*fresh)
    actual = restored.update_and_fetch(*fresh)
    assert all(mx.array_equal(a, b).item() for a, b in zip(expected, actual))


def test_projected_checkpoint_rejects_inconsistent_overhang():
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.cache import RotatingKVCache

    from vmlx_engine.dspark.inkling_drafter import checkpoint_cache

    keys = mx.zeros((1, 1, 6, 1), dtype=mx.bfloat16)
    invalid = RotatingKVCache.from_state((keys, keys), ("0", "4", "7", "2"))
    assert checkpoint_cache([invalid]) is None
