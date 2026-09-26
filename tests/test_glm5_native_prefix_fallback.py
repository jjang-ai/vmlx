"""Select the longest *valid* native state without relaxing cache identity."""

from unittest.mock import Mock
import weakref

import pytest

pytest.importorskip("mlx.core")

from tests.test_glm5_companion_disk_codec import native_facade, native_state
from vmlx_engine.utils.glm5_native_prefix_cache import Glm5NativePrefixCache


@pytest.mark.parametrize("complete,state_boundary", [(True, 7), (False, 8)])
def test_native_rejected_longest_falls_back_to_valid_disk_state(
    tmp_path, complete, state_boundary
):
    tokens = list(range(11))
    salt = {"media": "processed-pixels-A"}
    cache = native_facade(tmp_path)
    try:
        assert cache.store(tokens, 4, native_state(4), extra_keys=salt)["durable"]
        # Simulate a transport-readable record whose native contract is wrong.
        # The public native store correctly refuses this; inject only into
        # this isolated test pool through the lower-level typed transport.
        key = cache.lookup._key(tokens, 8, cache_extra_keys=salt)
        assert cache.disk.store(key, native_state(state_boundary), complete, tokens, 8)
        assert cache.disk.wait_for_write(key, timeout=5)
        restored = cache.fetch(tokens, extra_keys=salt, request_id="fallback")
        assert restored is not None
        boundary, layers = restored
        assert boundary == 4
        assert cache._valid_boundary(layers, 4)
        assert cache.last_fetch["cached_tokens"] == 4
        assert cache.last_fetch["key"] == cache.lookup._key(
            tokens, 4, cache_extra_keys=salt
        )
        assert cache.fetch(tokens, extra_keys={"media": "processed-pixels-B"}) is None
    finally:
        cache.close()


@pytest.mark.parametrize("boundary", [0, -1, 11, True, "8"])
def test_malformed_lookup_boundary_fails_closed_without_retry(boundary):
    cache = Glm5NativePrefixCache.__new__(Glm5NativePrefixCache)
    cache.lookup = Mock()
    cache.lookup.fetch_longest_prefix.return_value = (boundary, [], True)
    cache._valid_boundary = Mock(return_value=True)
    assert cache.fetch(list(range(11))) is None
    assert cache.lookup.fetch_longest_prefix.call_count == 1
    cache._valid_boundary.assert_not_called()


def test_nonprogressing_lookup_cannot_retry_same_rejected_boundary():
    cache = Glm5NativePrefixCache.__new__(Glm5NativePrefixCache)
    cache.lookup = Mock()
    cache.lookup.fetch_longest_prefix.return_value = (8, [], False)
    assert cache.fetch(list(range(11))) is None
    assert [call.kwargs["max_len"] for call in
            cache.lookup.fetch_longest_prefix.call_args_list] == [10, 7]


def test_rejected_payload_is_released_before_next_disk_read():
    class Payload:
        pass

    rejected = None

    def fetch(tokens, *, max_len, cache_extra_keys):
        nonlocal rejected
        if max_len == 10:
            payload = Payload()
            rejected = weakref.ref(payload)
            return 8, payload, False
        assert max_len == 7
        assert rejected() is None
        return 4, Payload(), True

    cache = Glm5NativePrefixCache.__new__(Glm5NativePrefixCache)
    cache.lookup = Mock()
    cache.lookup.fetch_longest_prefix.side_effect = fetch
    cache._valid_boundary = Mock(return_value=True)
    assert cache.fetch(list(range(11)))[0] == 4
