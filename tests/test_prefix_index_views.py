"""The block-aware prefix index stores every block-aligned prefix of a stored sequence. Entries share one copy of the
token / block-id lists per store (prefix_cache._PrefixView) instead of copying each prefix (O(blocks^2): 395 ms and
~0.9 GB of lists per store at 121k tokens on MiMo's plain lane). Readers take len(), iterate, index, slice and compare
with lists."""
from vmlx_engine.prefix_cache import _PrefixView


def test_view_behaves_like_the_prefix_list():
    base = list(range(1000))
    v = _PrefixView(base, 640)
    assert len(v) == 640 and list(v) == base[:640]
    assert v == base[:640] and base[:640] == v and not (v != base[:640]) and not (base[:640] != v)
    assert v != base[:639] and base[:641] != v
    assert v[0] == 0 and v[-1] == 639 and v[100:164] == base[100:164] and v[600:700] == base[600:640]
    assert v[::64] == base[:640:64]
    assert _PrefixView(base, 640) == _PrefixView(list(base), 640)
    try:
        v[640]
        raise AssertionError("index past the view must raise")
    except IndexError:
        pass
    assert 639 in v and 640 not in v


def test_update_prefix_index_shares_one_copy_per_store():
    import threading
    from types import SimpleNamespace

    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
    cache.paged_cache = SimpleNamespace(_lock=threading.RLock())
    cache._prefix_index = {}
    cache._chained_prefix_index_hash = True
    cache.block_size = 64
    seq = [((i * 7919) % 150000) for i in range(64 * 300 + 17)]
    ids = list(range(1000, 1000 + 301))
    cache._prefix_index_hash_sequence = lambda tokens, n, cache_extra_keys=None: [
        (min(len(tokens), i * 64), f"h{i}") for i in range(1, n + 1)]
    cache._prefix_index_extra_marker = lambda extra, n: None
    cache._update_prefix_index(seq, ids)
    entries = list(cache._prefix_index.values())
    assert len(entries) == 301
    bases = {id(e[0]._base) for e in entries} | {id(e[1]._base) for e in entries}
    assert len(bases) == 2                                     # one token copy + one block-id copy for the store
    last = cache._prefix_index["h301"]
    assert last[0] == seq and last[1] == ids
    assert cache._prefix_index["h3"][0] == seq[:192] and cache._prefix_index["h3"][1] == ids[:3]
    seq.append(1)                                              # the caller's list is not aliased
    assert len(cache._prefix_index["h301"][0]) == len(seq) - 1
