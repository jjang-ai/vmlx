"""DFlash2 session SSD tier: exact round trip of target/draft caches, prefix-cut lookup, misses never restore wrong state."""
import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from vmlx_engine.dflash2_session_disk import DFlash2SessionSSD, candidate_cuts

IM_START, IM_END = 900, 901


def _target(n):
    kv, gdn = KVCache(), ArraysCache(size=2)
    kv.update_and_fetch(mx.random.normal((1, 2, n, 8)), mx.random.normal((1, 2, n, 8)))
    gdn[0] = mx.random.normal((1, 3, 16))
    gdn[1] = mx.random.normal((1, 2, 4, 4))
    return [kv, gdn]


def _draft(n):
    c = RotatingKVCache(max_size=4)
    for _ in range(n):
        c.update_and_fetch(mx.random.normal((1, 1, 1, 8)), mx.random.normal((1, 1, 1, 8)))
    return [c]


@pytest.fixture
def ssd(tmp_path):
    s = DFlash2SessionSSD(root=tmp_path, max_size_bytes=1 << 30, model_key="unit")
    yield s
    s.store.close()


def _take(ssd, prompt):
    return ssd.take_matching(prompt, im_start_id=IM_START, eos_ids=[IM_END],
                             make_target=lambda: [KVCache(), ArraysCache(size=2)],
                             make_draft=lambda: [RotatingKVCache(max_size=4)])


def test_candidate_cuts_are_tag_and_after_eos_positions_longest_first():
    p = [IM_START, 1, 2, IM_END, 3, IM_START, 4, IM_END, 5]
    assert candidate_cuts(p, IM_START, [IM_END]) == [8, 5, 4]


def test_turn_entry_round_trips_exactly_and_matches_a_longer_prompt(ssd):
    tokens = [IM_START, 1, 2, IM_START, 7, 8, IM_END]
    target, draft, gap = _target(6), _draft(5), mx.random.normal((1, 1, 5))
    ssd.put({"kind": "turn", "tokens": tokens, "cache_len": 6, "target_cache": target,
             "draft_cache": draft, "draft_hidden_gap": gap})
    assert ssd.flush()
    got = _take(ssd, tokens + [10, IM_START, 11])
    assert got is not None and got["kind"] == "turn" and got["cache_len"] == 6 and got["tokens"] == tokens
    for a, b in zip(target[0].state, got["target_cache"][0].state):
        assert mx.array_equal(a, b)
    assert got["target_cache"][0].offset == 6
    for a, b in zip(target[1].state, got["target_cache"][1].state):
        assert mx.array_equal(a, b)
    d0, d1 = draft[0], got["draft_cache"][0]
    assert d1.meta_state == d0.meta_state and d1.offset == d0.offset
    assert all(mx.array_equal(a, b) for a, b in zip(d0.state, d1.state))
    assert mx.array_equal(got["draft_hidden_gap"], gap)


def test_boundary_entry_without_draft_matches_at_the_tag(ssd):
    prefix = [IM_START, 1, 2, IM_END, 5]
    ssd.put({"kind": "boundary", "tokens": prefix, "cache_len": 5, "target_cache": _target(5),
             "draft_cache": None, "draft_hidden_gap": None})
    assert ssd.flush()
    got = _take(ssd, prefix + [IM_START, 6, 7])
    assert got is not None and got["draft_cache"] is None and got["cache_len"] == 5


def test_diverged_prompt_and_entry_covering_whole_prompt_are_misses(ssd):
    tokens = [IM_START, 1, 2, IM_END]
    ssd.put({"kind": "turn", "tokens": tokens, "cache_len": 3, "target_cache": _target(3),
             "draft_cache": None, "draft_hidden_gap": None})
    assert ssd.flush()
    assert _take(ssd, [IM_START, 1, 3, IM_END, IM_START, 4]) is None   # different history
    assert _take(ssd, tokens) is None                                   # nothing left to prefill


def test_cache_layout_mismatch_is_a_miss(ssd):
    tokens = [IM_START, 1, IM_END]
    ssd.put({"kind": "turn", "tokens": tokens, "cache_len": 2, "target_cache": _target(2),
             "draft_cache": None, "draft_hidden_gap": None})
    assert ssd.flush()
    got = ssd.take_matching(tokens + [IM_START], im_start_id=IM_START, eos_ids=[IM_END],
                            make_target=lambda: [KVCache(), KVCache()], make_draft=lambda: [])
    assert got is None


def test_clear_removes_entries(ssd):
    tokens = [IM_START, 1, IM_END]
    ssd.put({"kind": "turn", "tokens": tokens, "cache_len": 2, "target_cache": _target(2),
             "draft_cache": None, "draft_hidden_gap": None})
    assert ssd.flush()
    assert ssd.clear() > 0
    assert _take(ssd, tokens + [IM_START]) is None


def test_empty_draft_cache_is_stored_without_draft_state(ssd):
    tokens = [IM_START, 1, IM_END]
    ssd.put({"kind": "turn", "tokens": tokens, "cache_len": 2, "target_cache": _target(2),
             "draft_cache": [RotatingKVCache(max_size=4)], "draft_hidden_gap": None})   # never ran
    assert ssd.flush()
    got = _take(ssd, tokens + [IM_START])
    assert got is not None and got["draft_cache"] is None and got["cache_len"] == 2


def test_boundary_draft_context_survives_ssd_without_draft_kv(ssd):
    prefix = [IM_START, 1, 2, IM_END, 5]
    context = mx.arange(12).reshape(1, 4, 3).astype(mx.bfloat16)
    ssd.put({"kind": "boundary", "tokens": prefix, "cache_len": 5,
             "target_cache": _target(5), "draft_cache": None,
             "draft_hidden_gap": None, "draft_context": context})
    assert ssd.flush()
    got = _take(ssd, prefix + [IM_START, 6, 7])
    assert got["draft_cache"] is None
    assert got["draft_context"].dtype == context.dtype
    assert mx.array_equal(got["draft_context"], context)


def test_boundary_snapshot_keeps_bounded_window_and_absolute_resume_position():
    from vmlx_engine.dflash2_runtime import _snapshot_draft_state, _rebuild_draft_hidden
    old = mx.arange(10).reshape(1, 10, 1)
    state = _snapshot_draft_state(None, None, None, [old], 10, 0, 4)
    assert state['draft_context'].tolist() == [[[6], [7], [8], [9]]]
    suffix = mx.array([[[10], [11]]])
    hidden, offset = _rebuild_draft_hidden(suffix, state, 10, 2, 0, 4)
    assert hidden.tolist() == [[[8], [9], [10], [11]]]
    assert offset == 8
    # A large delta already contains the whole sliding window: no stale
    # historical rows may be prepended across the omitted delta positions.
    hidden, offset = _rebuild_draft_hidden(suffix, state, 10, 5, 3, 2)
    assert hidden.tolist() == suffix.tolist() and offset == 13


def test_boundary_after_turn_preserves_draft_kv_and_contiguous_gap():
    from types import SimpleNamespace
    from vmlx_engine.dflash2_runtime import _snapshot_draft_state
    draft = _draft(3)
    gap = mx.array([[[3], [4]]])
    prior = {'draft_cache': draft, 'draft_hidden_gap': gap, 'cache_len': 5}
    runtime = SimpleNamespace(make_prompt_cache=lambda _: [RotatingKVCache(max_size=4)])
    state = _snapshot_draft_state(runtime, None, prior, [mx.array([[[5], [6]]])], 7, 5, 4)
    assert state['draft_cache'][0] is not draft[0]
    assert state['draft_cache'][0].offset == 3
    assert state['draft_hidden_gap'].tolist() == [[[3], [4], [5], [6]]]
    assert state['draft_context'] is None


def test_legacy_partial_suffix_is_not_certified_as_full_draft_context():
    from vmlx_engine.dflash2_runtime import _snapshot_draft_state
    state = _snapshot_draft_state(None, None, {}, [mx.array([[[8], [9]]])], 10, 8, 4)
    assert state['draft_context'] is None


@pytest.mark.parametrize("legacy", ["dflash2_session_v1", "dflash2_session_v2", "dflash2_session_v3"])
def test_v4_schema_clears_legacy_entries(tmp_path, legacy):
    """v1 entries were written while snapshots aliased live GDN state; v2 drafter caches rebuilt on resume could hold
    zero K/V rows (forced absolute offset); a v3 reader would restore a v4 delta entry as a whole cache. Never read
    them, clear them at startup."""
    import mlx.core as mx
    from vmlx_engine.dflash2_session_disk import DFlash2SessionSSD, SCHEMA
    from vmlx_engine.utils.omni_session_disk_store import OmniSessionDiskStore

    assert SCHEMA == "dflash2_session_v4"
    old = OmniSessionDiskStore(root=tmp_path, model_key="m", max_size_bytes=1 << 30, schema=legacy)
    old.save("sig", lambda tmp: mx.save_safetensors(str(tmp), {"a": mx.ones((4,))}, {"dflash2": "{}"}))
    old.close()
    assert any(old.directory.glob("*.safetensors"))
    ssd = DFlash2SessionSSD(root=tmp_path, max_size_bytes=1 << 30, model_key="m")
    assert ssd.legacy_cleared > 0
    assert not any(old.directory.glob("*.safetensors"))


# ---- delta entries (v4, 2026-10-12) ---------------------------------------------------------------------------
def test_pending_token_turn_can_parent_next_boundary_delta(ssd):
    # Native DFlash2 signs every confirmed token, but caches confirmed[:-1].
    # The pending EOS is processed as the first row of the resumed suffix.
    turn = [IM_START] + list(range(1, 257)) + [IM_END]
    target = _target(len(turn) - 1)
    first = ssd.put({"kind": "turn", "tokens": turn,
                     "cache_len": len(turn) - 1, "target_cache": target})
    assert ssd.flush() and first.result()["durable"]
    boundary = turn + [IM_START, 7, 8, IM_END, 9]
    _grow(target, len(boundary) - (len(turn) - 1))
    child = ssd.put({"kind": "boundary", "tokens": boundary,
                     "cache_len": len(boundary), "target_cache": target,
                     "ssd_parent": dict(first.vmlx_ssd_ref, receipt=first)})
    assert ssd.flush() and child.result()["durable"]
    assert ssd.stats["delta_writes"] == 1
    assert child.vmlx_ssd_ref["depth"] == 1
    got = _take(ssd, boundary + [IM_START, 42])
    assert got is not None and got["cache_len"] == len(boundary)
    for old, restored in zip(target, got["target_cache"]):
        assert all(mx.array_equal(a, b) for a, b in zip(old.state, restored.state))
    root_path = ssd.store.paths(first.vmlx_ssd_ref["sig"])[0]
    child_path = ssd.store.paths(child.vmlx_ssd_ref["sig"])[0]
    assert child_path.stat().st_size < root_path.stat().st_size / 2


def test_pending_token_parent_cannot_be_overwritten_by_its_own_delta(ssd):
    tokens = [IM_START, 1, 2, IM_END]
    target = _target(len(tokens) - 1)
    parent = ssd.put({"kind": "turn", "tokens": tokens,
                      "cache_len": len(tokens) - 1, "target_cache": target})
    assert ssd.flush()
    _grow(target, 1)
    # Same lookup key with one more cached row must become a full root,
    # not a self-referential parent chain at the same file signature.
    child = ssd.put({"kind": "boundary", "tokens": tokens,
                     "cache_len": len(tokens), "target_cache": target,
                     "ssd_parent": dict(parent.vmlx_ssd_ref, receipt=parent)})
    assert ssd.flush() and child.result()["durable"]
    assert child.vmlx_ssd_ref["depth"] == 0
    got = _take(ssd, tokens + [IM_START, 3])
    assert got is not None and got["cache_len"] == len(tokens)
    assert all(mx.array_equal(a, b) for a, b in zip(target[0].state, got["target_cache"][0].state))


@pytest.mark.parametrize("parent", [
    {"sig": "bad", "len": 3},
    {"sig": "4:wrong-digest", "len": 3},
    {"sig": None, "len": 3},
    {"sig": "3:wrong-digest", "len": 3},
])
def test_parent_key_validation_rejects_invalid_signatures(parent):
    from vmlx_engine.dflash2_session_disk import _parent_key_length
    assert _parent_key_length(parent, [IM_START, 1, 2, IM_END, 9]) is None


def test_parent_key_validation_rejects_changed_pending_token():
    from vmlx_engine.dflash2_session_disk import _parent_key_length, signature_for
    tokens = [IM_START, 1, 2, IM_END]
    parent = {"sig": signature_for(tokens), "len": len(tokens) - 1}
    assert _parent_key_length(parent, tokens + [9]) == len(tokens)
    assert _parent_key_length(parent, tokens[:-1] + [7, 9]) is None
    assert _parent_key_length({**parent, "len": 2}, tokens + [9]) is None


def _grow(target, n_more):
    """Append n_more rows to the KV layer and advance the GDN state, as a continued prefill/decode would."""
    kv, gdn = target
    kv.update_and_fetch(mx.random.normal((1, 2, n_more, 8)), mx.random.normal((1, 2, n_more, 8)))
    gdn[0] = gdn[0] * 0.5 + 1.0
    gdn[1] = gdn[1] * 0.25 - 1.0
    return target


def test_delta_chain_restores_bit_identical_and_writes_only_new_rows(ssd):
    system = [IM_START, 1, 2, 3]
    boundary = system + [IM_START, 4, 5, 6]
    turn = boundary + [IM_START, 7, 8, IM_END]
    target = _target(len(system))
    r1 = ssd.put({"kind": "system", "tokens": system, "cache_len": len(system), "target_cache": target})
    snap_sys = [mx.array(a) for a in target[0].state]
    _grow(target, len(boundary) - len(system))
    r2 = ssd.put({"kind": "boundary", "tokens": boundary, "cache_len": len(boundary), "target_cache": target,
                  "ssd_parent": dict(r1.vmlx_ssd_ref, receipt=r1)})
    _grow(target, len(turn) - len(boundary))
    r3 = ssd.put({"kind": "turn", "tokens": turn, "cache_len": len(turn), "target_cache": target,
                  "ssd_parent": dict(r2.vmlx_ssd_ref, receipt=r2)})
    assert ssd.flush()
    assert all(r.result()["durable"] for r in (r1, r2, r3))
    assert r3.vmlx_ssd_ref["depth"] == 2 and ssd.stats["delta_writes"] == 2
    got = _take(ssd, turn + [IM_START, 9])
    assert got is not None and got["cache_len"] == len(turn) and got["ssd_ref"]["depth"] == 2
    kv = got["target_cache"][0]
    assert kv.offset == len(turn) and kv.keys.shape[2] > len(turn)          # headroom: the next append is in place
    for a, b in zip(target[0].state, kv.state):
        assert mx.array_equal(a, b)
    for a, b in zip(target[1].state, got["target_cache"][1].state):        # GDN state from the leaf, not the root
        assert mx.array_equal(a, b)
    # the root entry still restores alone, unchanged by its children
    got_sys = _take(ssd, system + [IM_START, 42])
    assert got_sys is not None and all(mx.array_equal(a, b) for a, b in zip(snap_sys, got_sys["target_cache"][0].state))
    # the delta files hold only the new rows: much smaller than the root
    sizes = {}
    for tokens in (system, boundary, turn):
        from vmlx_engine.dflash2_session_disk import signature_for
        sizes[len(tokens)] = ssd.store.paths(signature_for(tokens))[0].stat().st_size
    assert sizes[len(turn)] < sizes[len(system)] + 4096


def test_delta_with_missing_parent_is_a_miss_never_wrong_state(ssd):
    from vmlx_engine.dflash2_session_disk import signature_for
    system = [IM_START, 1, 2, 3]
    boundary = system + [IM_START, 4, 5, 6]
    target = _target(len(system))
    r1 = ssd.put({"kind": "system", "tokens": system, "cache_len": len(system), "target_cache": target})
    _grow(target, 4)
    r2 = ssd.put({"kind": "boundary", "tokens": boundary, "cache_len": len(boundary), "target_cache": target,
                  "ssd_parent": dict(r1.vmlx_ssd_ref, receipt=r1)})
    assert ssd.flush() and r2.result()["durable"]
    for path in ssd.store.paths(signature_for(system)):                    # the pool evicted the parent
        path.unlink()
    assert _take(ssd, boundary + [IM_START, 9]) is None
    assert ssd.stats["chain_misses"] == 1


def test_child_of_a_failed_parent_is_refused_and_a_refused_parent_gives_a_full_child(ssd):
    from concurrent.futures import Future
    system = [IM_START, 1, 2, 3]
    boundary = system + [IM_START, 4, 5, 6]
    target = _target(len(system))
    failed = Future(); failed.set_result({"outcome": "failed", "durable": False})
    from vmlx_engine.dflash2_session_disk import signature_for
    ref = {"sig": signature_for(system), "len": len(system), "depth": 0}
    _grow(target, 4)
    r = ssd.put({"kind": "boundary", "tokens": boundary, "cache_len": len(boundary), "target_cache": target,
                 "ssd_parent": dict(ref, receipt=failed)})
    assert ssd.flush() and r.result()["durable"] and r.vmlx_ssd_ref["depth"] == 0     # written whole
    pending = Future()                                                       # parent "queued" but never written
    turn = boundary + [IM_START, 7, IM_END]
    _grow(target, 3)
    ssd._parent_durable = lambda sig, receipt: False
    r2 = ssd.put({"kind": "turn", "tokens": turn, "cache_len": len(turn), "target_cache": target,
                  "ssd_parent": dict(r.vmlx_ssd_ref, receipt=pending)})
    assert ssd.flush() and not r2.result()["durable"]


def _big_target(n):
    kv, gdn = KVCache(), ArraysCache(size=2)
    kv.update_and_fetch(mx.random.normal((1, 8, n, 256)), mx.random.normal((1, 8, n, 256)))
    gdn[0] = mx.random.normal((1, 3, 16))
    gdn[1] = mx.random.normal((1, 2, 4, 4))
    mx.eval(kv.keys, kv.values)
    return [kv, gdn]


def _append_peak(target):
    """MLX peak-memory increase of a one-row in-place append to the live KV layer."""
    mx.eval(target[0].keys, target[0].values)
    mx.reset_peak_memory()
    base = mx.get_active_memory()
    k, v = target[0].update_and_fetch(mx.zeros((1, 8, 1, 256)), mx.zeros((1, 8, 1, 256)))
    mx.eval(k, v)
    return mx.get_peak_memory() - base


@pytest.mark.parametrize("delta", [True, False])
def test_delta_pack_does_not_make_the_next_append_copy_the_live_buffer(tmp_path, delta):
    """A queued SSD write that holds a VIEW of the live KV buffer makes MLX copy the whole buffer on the next append
    (copy-on-write). Delta rows are copied when packed, so a queued delta write leaves the append in place; a whole
    entry (the control) still holds the view."""
    import threading
    gate = threading.Event()
    ssd = DFlash2SessionSSD(root=tmp_path, max_size_bytes=1 << 30, model_key="cow")
    try:
        system = [IM_START] + list(range(1, 1000))
        boundary = system + [IM_START] + list(range(2000, 2999))
        target = _big_target(len(system))
        r1 = ssd.put({"kind": "system", "tokens": system, "cache_len": len(system), "target_cache": target})
        assert ssd.flush() and r1.result()["durable"]
        k, v = target[0].update_and_fetch(mx.random.normal((1, 8, 1000, 256)), mx.random.normal((1, 8, 1000, 256)))
        mx.eval(k, v)
        del k, v
        save = ssd.store.save
        ssd.store.save = lambda sig, writer: (gate.wait(30), save(sig, writer))[1]     # hold the write in flight
        entry = {"kind": "boundary", "tokens": boundary, "cache_len": len(boundary), "target_cache": target}
        if delta:
            entry["ssd_parent"] = dict(r1.vmlx_ssd_ref, receipt=r1)
        r2 = ssd.put(entry)
        del entry
        buffer_bytes = target[0].keys.nbytes
        grew = _append_peak(target)
        gate.set()
        assert ssd.flush() and r2.result()["durable"]
        if delta:
            assert grew < buffer_bytes / 4, (grew, buffer_bytes)
        else:
            assert grew >= buffer_bytes, (grew, buffer_bytes)   # the control: the view forced a whole-buffer copy
    finally:
        gate.set()
        ssd.store.close()


@pytest.mark.parametrize("draft_delta", [True, False])
def test_dspark_drafter_context_is_a_delta_and_restores_bit_identical(ssd, draft_delta):
    """DSpark's drafter context (one append-only KVCache per drafter layer, token-aligned with the target) rides the
    target's parent chain when the entry opts in (draft_delta); without it the context is stored whole (DFlash2)."""
    from vmlx_engine.dflash2_session_disk import signature_for

    system = [IM_START] + list(range(10, 200))                                # a long history...
    turn = system + [IM_START, 7, 8, IM_END]                                  # ...and a short new turn
    target, ctx = _target(len(system)), KVCache()
    ctx.update_and_fetch(mx.random.normal((1, 2, len(system), 64)), mx.random.normal((1, 2, len(system), 64)))
    r1 = ssd.put({"kind": "system", "tokens": system, "cache_len": len(system), "target_cache": target,
                  "draft_cache": [ctx], "draft_delta": draft_delta})
    n_more = len(turn) - len(system)
    _grow(target, n_more)
    ctx.update_and_fetch(mx.random.normal((1, 2, n_more, 64)), mx.random.normal((1, 2, n_more, 64)))
    r2 = ssd.put({"kind": "turn", "tokens": turn, "cache_len": len(turn), "target_cache": target,
                  "draft_cache": [ctx], "draft_delta": draft_delta, "ssd_parent": dict(r1.vmlx_ssd_ref, receipt=r1)})
    assert ssd.flush() and r2.result()["durable"]
    got = ssd.take_matching(turn + [IM_START, 7], im_start_id=IM_START, eos_ids=[IM_END],
                            make_target=lambda: [KVCache(), ArraysCache(size=2)], make_draft=lambda: [KVCache()])
    assert got is not None and got["draft_cache"] is not None
    d = got["draft_cache"][0]
    assert d.offset == len(turn)
    for a, b in zip(ctx.state, d.state):
        assert mx.array_equal(a, b)
    size = ssd.store.paths(signature_for(turn))[0].stat().st_size
    whole_ctx = 2 * len(turn) * 2 * 64 * 4                                    # keys + values, fp32
    if draft_delta:
        assert size < whole_ctx // 4                                          # only the new context rows
    else:
        assert size > whole_ctx


# ---- same prefix stored again (a later session over the same conversation) ----
def _chain(ssd, lengths, start=None):
    """Writes a delta chain over the prefixes of one token list; returns (tokens, receipts, target)."""
    tokens = [IM_START] + [10 + i for i in range(max(lengths) - 1)]
    target = _target(lengths[0]) if start is None else start
    receipts, prev = [], None
    for i, n in enumerate(lengths):
        if i:
            _grow(target, n - lengths[i - 1])
        entry = {"kind": "chunk", "tokens": tokens[:n], "cache_len": n, "target_cache": target}
        if prev is not None:
            entry["ssd_parent"] = dict(prev.vmlx_ssd_ref, receipt=prev)
        prev = ssd.put(entry)
        receipts.append(prev)
    assert ssd.flush()
    return tokens, receipts, target


def _side(ssd, tokens, base_receipt, base_len, n):
    """A side entry tokens[:n] stored as a delta on the entry tokens[:base_len] (another conversation's turn)."""
    t = _target(base_len)
    _grow(t, n - base_len)
    r = ssd.put({"kind": "turn", "tokens": tokens[:n], "cache_len": n, "target_cache": t,
                 "ssd_parent": dict(base_receipt.vmlx_ssd_ref, receipt=base_receipt)})
    assert ssd.flush()
    return r, t


def test_same_prefix_restored_later_keeps_the_stored_entry_so_dependents_still_load(ssd):
    """2026-10-19 MiMo in-app chain, second session over the same prefixes: two turns re-stored mid-chain prefixes as
    deltas on OTHER parents; every older entry naming them as parents then walked two more links, the 8-deep
    boundary walked 11 and the loader dropped it (restore fell back to an earlier chunk)."""
    from vmlx_engine.dflash2_session_disk import MAX_CHAIN, signature_for
    lengths = [4 * (i + 1) for i in range(MAX_CHAIN + 1)]          # 4..36, depths 0..MAX_CHAIN
    tokens, receipts, target = _chain(ssd, lengths)
    deepest = lengths[-1]
    assert receipts[-1].vmlx_ssd_ref["depth"] == MAX_CHAIN
    # side entries at 6 (on 4) and 14 (on 12): the parents the second session's re-stores are built on
    s6, t6 = _side(ssd, tokens, receipts[0], 4, 6)
    s14, t14 = _side(ssd, tokens, receipts[2], 12, 14)
    files = {n: ssd.store.paths(signature_for(tokens[:n]))[0] for n in (8, 16)}
    before = {n: f.read_bytes() for n, f in files.items()}
    # second session: prefixes 8 and 16 stored again, built on the side entries
    _grow(t6, 2)
    again8 = ssd.put({"kind": "boundary", "tokens": tokens[:8], "cache_len": 8, "target_cache": t6,
                      "ssd_parent": dict(s6.vmlx_ssd_ref, receipt=s6)})
    _grow(t14, 2)
    again16 = ssd.put({"kind": "boundary", "tokens": tokens[:16], "cache_len": 16, "target_cache": t14,
                       "ssd_parent": dict(s14.vmlx_ssd_ref, receipt=s14)})
    assert ssd.flush()
    assert again8.result()["durable"] and again16.result()["durable"]
    assert "kept" in again8.result()["detail"] and ssd.stats["same_prefix_kept"] == 2
    assert again8.vmlx_ssd_ref["depth"] == 1 and again16.vmlx_ssd_ref["depth"] == 3      # the kept entries' depths
    assert all(files[n].read_bytes() == before[n] for n in files)                        # not replaced
    got = _take(ssd, tokens[:deepest] + [IM_START, 7])               # the MAX_CHAIN-deep dependent still restores
    assert got is not None and got["cache_len"] == deepest and ssd.stats["chain_misses"] == 0
    assert all(mx.array_equal(a, b) for a, b in zip(target[0].state, got["target_cache"][0].state))


def test_parent_whose_recorded_depth_understates_its_chain_gets_a_whole_child(ssd):
    from vmlx_engine.dflash2_session_disk import MAX_CHAIN
    lengths = [4 * (i + 1) for i in range(MAX_CHAIN + 1)]
    tokens, receipts, target = _chain(ssd, lengths)
    deepest = receipts[-1].vmlx_ssd_ref
    _grow(target, 4)
    n = lengths[-1] + 4
    # a restored parent whose ref claims depth 0 although its file walks MAX_CHAIN parents (a replaced ancestor)
    child = ssd.put({"kind": "turn", "tokens": tokens[:lengths[-1]] + [IM_START, 1, 2, IM_END], "cache_len": n,
                     "target_cache": target, "ssd_parent": dict(deepest, depth=0)})
    assert ssd.flush() and child.result()["durable"]
    assert child.vmlx_ssd_ref["depth"] == 0 and ssd.stats["deep_parent_full_writes"] == 1   # written whole
    got = _take(ssd, tokens[:lengths[-1]] + [IM_START, 1, 2, IM_END, IM_START, 5])
    assert got is not None and got["cache_len"] == n


def test_less_complete_stored_entry_is_replaced_by_a_whole_entry(ssd):
    from vmlx_engine.dflash2_session_disk import signature_for
    prefix = [IM_START, 1, 2, IM_END, 5]
    ssd.put({"kind": "boundary", "tokens": prefix, "cache_len": 5, "target_cache": _target(5)})   # no drafter context
    assert ssd.flush()
    context = mx.arange(12).reshape(1, 4, 3).astype(mx.bfloat16)
    r = ssd.put({"kind": "boundary", "tokens": prefix, "cache_len": 5, "target_cache": _target(5), "draft_context": context})
    assert ssd.flush() and r.result()["durable"] and "kept" not in r.result()["detail"] and ssd.stats["same_prefix_kept"] == 0
    got = _take(ssd, prefix + [IM_START, 6])
    assert got is not None and mx.array_equal(got["draft_context"], context)
    assert ssd.store.paths(signature_for(prefix))[0].is_file()


def test_in_process_chains_stay_deltas(ssd):
    from vmlx_engine.dflash2_session_disk import MAX_CHAIN
    lengths = [4 * (i + 1) for i in range(MAX_CHAIN + 1)]
    _, receipts, _ = _chain(ssd, lengths)
    assert [r.vmlx_ssd_ref["depth"] for r in receipts] == list(range(MAX_CHAIN + 1))
    assert ssd.stats["delta_writes"] == MAX_CHAIN and ssd.stats["deep_parent_full_writes"] == 0


def test_kept_prefix_must_not_become_parent_of_recomputed_state(ssd):
    """Equal tokens need not mean equal KV bits after a different prefill chunking."""
    prefix = [IM_START, 1, 2, IM_END]
    old = _target(4)
    ssd.put({"tokens": prefix, "cache_len": 4, "target_cache": old})
    assert ssd.flush()
    current = _target(4)
    kept = ssd.put({"tokens": prefix, "cache_len": 4, "target_cache": current})
    assert ssd.flush()
    _grow(current, 4)
    child_tokens = prefix + [IM_START, 3, 4, IM_END]
    child = ssd.put({"tokens": child_tokens, "cache_len": 8, "target_cache": current,
                     "ssd_parent": dict(kept.vmlx_ssd_ref, receipt=kept)})
    assert ssd.flush() and child.result()["durable"]
    got = _take(ssd, child_tokens + [IM_START, 5])
    assert got is not None and got["cache_len"] == 8
    assert all(mx.array_equal(a, b) for a, b in zip(current[0].state, got["target_cache"][0].state))


def test_replaced_prefix_must_not_silently_change_existing_child(ssd):
    prefix = [IM_START, 1, 2, IM_END]
    current = _target(4)
    parent = ssd.put({"tokens": prefix, "cache_len": 4, "target_cache": current})
    assert ssd.flush()
    _grow(current, 4)
    child_tokens = prefix + [IM_START, 3, 4, IM_END]
    ssd.put({"tokens": child_tokens, "cache_len": 8, "target_cache": current,
             "ssd_parent": dict(parent.vmlx_ssd_ref, receipt=parent)})
    assert ssd.flush()
    ssd.put({"tokens": prefix, "cache_len": 4, "target_cache": _target(4),
             "draft_context": mx.ones((1, 4, 3))})
    assert ssd.flush()
    got = _take(ssd, child_tokens + [IM_START, 5])
    # It may reject the old child or retain its exact state, never splice new parent bits into it.
    assert got is None or got["cache_len"] != 8 or all(
        mx.array_equal(a, b) for a, b in zip(current[0].state, got["target_cache"][0].state))


def test_actual_restore_of_kept_prefix_can_be_a_delta_parent(ssd):
    prefix = [IM_START, 1, 2, IM_END]
    first = ssd.put({"tokens": prefix, "cache_len": 4, "target_cache": _target(4)})
    assert ssd.flush()
    ssd.put({"tokens": prefix, "cache_len": 4, "target_cache": _target(4)})
    restored = _take(ssd, prefix + [IM_START, 5])
    assert restored is not None
    assert restored["ssd_ref"]["generation"] == first.vmlx_ssd_ref["generation"]
    target = restored["target_cache"]
    _grow(target, 4)
    tokens = prefix + [IM_START, 3, 4, IM_END]
    child = ssd.put({"tokens": tokens, "cache_len": 8, "target_cache": target,
                     "ssd_parent": restored["ssd_ref"]})
    assert ssd.flush() and child.result()["durable"]
    assert child.vmlx_ssd_ref["depth"] == 1
    got = _take(ssd, tokens + [IM_START, 5])
    assert got is not None and got["cache_len"] == 8
    assert all(mx.array_equal(a, b) for a, b in zip(target[0].state, got["target_cache"][0].state))


def test_replaced_parent_reference_forces_whole_child(ssd):
    prefix = [IM_START, 1, 2, IM_END]
    target = _target(4)
    parent = ssd.put({"tokens": prefix, "cache_len": 4, "target_cache": target})
    assert ssd.flush()
    ssd.put({"tokens": prefix, "cache_len": 4, "target_cache": _target(4),
             "draft_context": mx.ones((1, 4, 3))})
    assert ssd.flush()
    _grow(target, 4)
    tokens = prefix + [IM_START, 3, 4, IM_END]
    child = ssd.put({"tokens": tokens, "cache_len": 8, "target_cache": target,
                     "ssd_parent": dict(parent.vmlx_ssd_ref, receipt=parent)})
    assert ssd.flush() and child.result()["durable"]
    assert child.vmlx_ssd_ref["depth"] == 0
    got = _take(ssd, tokens + [IM_START, 5])
    assert got is not None and got["cache_len"] == 8
    assert all(mx.array_equal(a, b) for a, b in zip(target[0].state, got["target_cache"][0].state))
