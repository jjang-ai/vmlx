from types import SimpleNamespace

from vmlx_engine.models.mllm import (
    _install_mimo_v2_rotating_cache_keep_patch_if_needed,
)


def test_mimo_v2_rotating_cache_patch_uses_keep_zero_for_swa_layers():
    class FakeRotatingKVCache:
        def __init__(self, max_size, keep=0):
            self.max_size = max_size
            self.keep = keep

    class FakeKVCache:
        pass

    class FakeModel:
        def __init__(self):
            self.args = SimpleNamespace(sliding_window=128)
            self.model = SimpleNamespace(
                layers=[
                    SimpleNamespace(is_swa=False),
                    SimpleNamespace(is_swa=True),
                    SimpleNamespace(is_swa=True),
                ]
            )

        def make_cache(self):
            raise AssertionError("original make_cache should not be used")

    runtime = SimpleNamespace(
        Model=FakeModel,
        RotatingKVCache=FakeRotatingKVCache,
        KVCache=FakeKVCache,
    )

    assert _install_mimo_v2_rotating_cache_keep_patch_if_needed(runtime) is True

    caches = runtime.Model().make_cache()

    assert isinstance(caches[0], FakeKVCache)
    assert isinstance(caches[1], FakeRotatingKVCache)
    assert isinstance(caches[2], FakeRotatingKVCache)
    assert caches[1].max_size == 128
    assert caches[2].max_size == 128
    assert caches[1].keep == 0
    assert caches[2].keep == 0
