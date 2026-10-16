"""DFlash2 session-store prefix reuse.

The DFlash2 lane runs through SimpleEngine, which has no paged prefix-cache
stack, so before the session store every turn re-prefilled the entire
conversation (measured 15.5 s/turn at 7k tokens). These tests pin the pure
bookkeeping: prefix matching, LRU ownership transfer, and the exit-checkpoint
correction that shrinks the final cycle's committed cache positions down to
the emitted tokens (the runtime's EOS path returns before its rollback, so
rejected draft positions are still in the cache at exit).
"""

import pytest

from vmlx_engine.dflash2_runtime import (
    _DFlash2SessionStore,
    _checkpoint_correction,
    _prefix_reuse_enabled,
)


def _entry(model_key, tokens, tag=None, cache_len=None):
    return {
        "model_key": model_key,
        "tokens": list(tokens),
        "cache_len": len(tokens) - 1 if cache_len is None else cache_len,
        "target_cache": tag or object(),
        "draft_cache": None,
    }


class TestTakeMatching:
    def test_exact_prefix_hit(self):
        store = _DFlash2SessionStore()
        store.put(_entry("m", [1, 2, 3, 4]))
        got = store.take_matching("m", [1, 2, 3, 4, 5, 6])
        assert got is not None
        assert got["tokens"] == [1, 2, 3, 4]

    def test_hit_removes_the_entry(self):
        """Ownership transfers: the caches are mutated by the resumed
        generation, so a second consumer must never see the same entry."""
        store = _DFlash2SessionStore()
        store.put(_entry("m", [1, 2, 3]))
        assert store.take_matching("m", [1, 2, 3, 4]) is not None
        assert store.take_matching("m", [1, 2, 3, 4]) is None

    def test_divergent_history_misses(self):
        store = _DFlash2SessionStore()
        store.put(_entry("m", [1, 2, 3, 4]))
        assert store.take_matching("m", [1, 2, 9, 4, 5]) is None

    def test_stored_longer_than_prompt_misses(self):
        """A shorter prompt cannot resume a longer conversation: hybrid GDN
        state cannot be trimmed back to an arbitrary earlier point."""
        store = _DFlash2SessionStore()
        store.put(_entry("m", [1, 2, 3, 4, 5]))
        assert store.take_matching("m", [1, 2, 3]) is None

    def test_prompt_equal_to_stored_is_a_hit(self):
        """Regeneration from the same point: the delta is exactly the stored
        next-token, which the resumed prefill forwards."""
        store = _DFlash2SessionStore()
        store.put(_entry("m", [1, 2, 3]))
        assert store.take_matching("m", [1, 2, 3]) is not None

    def test_model_key_mismatch_misses(self):
        """Token ids are only meaningful per (model, draft) pair; a different
        model's conversation must never be resumed."""
        store = _DFlash2SessionStore()
        store.put(_entry(("m1", "d1"), [1, 2, 3]))
        assert store.take_matching(("m2", "d1"), [1, 2, 3, 4]) is None

    def test_longest_matching_conversation_wins(self):
        store = _DFlash2SessionStore(max_entries=4)
        store.put(_entry("m", [1, 2], tag="short"))
        store.put(_entry("m", [1, 2, 3, 4], tag="long"))
        got = store.take_matching("m", [1, 2, 3, 4, 5])
        assert got["target_cache"] == "long"


class TestPromptBoundarySnapshots:
    """Boundary entries have cache_len == len(tokens): the whole prompt is
    forwarded but no output token exists yet. They rescue reuse when the
    template strips the previous turn's <think> block from history."""

    def test_boundary_hit_needs_a_nonempty_delta(self):
        store = _DFlash2SessionStore()
        store.put(_entry("m", [1, 2, 3], cache_len=3))
        # Prompt identical to the cached positions: nothing left to prefill,
        # so this must miss (the loop needs at least one token for logits).
        assert store.take_matching("m", [1, 2, 3]) is None
        store.put(_entry("m", [1, 2, 3], cache_len=3))
        assert store.take_matching("m", [1, 2, 3, 4]) is not None

    def test_end_of_turn_beats_boundary_when_it_covers_more(self):
        store = _DFlash2SessionStore(max_entries=4)
        store.put(_entry("m", [1, 2, 3], cache_len=3, tag="boundary"))
        store.put(_entry("m", [1, 2, 3, 4, 5], cache_len=4, tag="turn"))
        got = store.take_matching("m", [1, 2, 3, 4, 5, 6])
        assert got["target_cache"] == "turn"

    def test_boundary_wins_when_think_strip_breaks_the_turn_entry(self):
        """The realistic reasoning-template shape: the end-of-turn entry
        contains think tokens (7, 8) the next prompt does not."""
        store = _DFlash2SessionStore(max_entries=4)
        store.put(_entry("m", [1, 2, 3], cache_len=3, tag="boundary"))
        store.put(_entry("m", [1, 2, 3, 7, 8, 9], cache_len=5, tag="turn"))
        got = store.take_matching("m", [1, 2, 3, 9, 4, 5])
        assert got["target_cache"] == "boundary"

    def test_empty_store_misses(self):
        assert _DFlash2SessionStore().take_matching("m", [1, 2]) is None


class TestLru:
    def test_cap_evicts_oldest(self):
        store = _DFlash2SessionStore(max_entries=2)
        store.put(_entry("m", [1, 1]))
        store.put(_entry("m", [2, 2]))
        store.put(_entry("m", [3, 3]))
        assert store.take_matching("m", [1, 1, 9]) is None
        assert store.take_matching("m", [2, 2, 9]) is not None
        assert store.take_matching("m", [3, 3, 9]) is not None

    def test_zero_cached_positions_never_matches(self):
        """A single-token conversation caches nothing; resuming it would be
        a full prefill pretending to be a hit."""
        store = _DFlash2SessionStore()
        store.put(_entry("m", [7]))
        assert store.take_matching("m", [7, 9]) is None

    def test_clear(self):
        store = _DFlash2SessionStore()
        store.put(_entry("m", [1, 2]))
        store.clear()
        assert store.take_matching("m", [1, 2, 3]) is None


class TestCheckpointCorrection:
    def test_clean_cycle_needs_no_correction(self):
        """Normal loop end: the cycle's own rollback already trimmed to
        accepted+1 committed positions and all were emitted."""
        assert _checkpoint_correction(4, 4) is None

    def test_no_cycles_needs_no_correction(self):
        assert _checkpoint_correction(0, 0) is None

    def test_eos_exit_keeps_only_emitted_positions(self):
        """EOS at draft index 1 with block 5: upstream returns before its
        rollback, so all 5 verify positions are committed but only 2 tokens
        were emitted. rollback(accepted=1, trim=3) keeps exactly 2."""
        accepted, trim = _checkpoint_correction(5, 2)
        assert accepted == 1
        assert trim == 3
        assert accepted + 1 + trim == 5

    def test_length_truncation_after_cycle_rollback(self):
        """max_tokens truncation: the cycle rolled back to accepted+1=4
        committed positions but only 3 tokens were emitted."""
        accepted, trim = _checkpoint_correction(4, 3)
        assert accepted == 2
        assert trim == 1

    def test_single_emitted_token_from_full_block(self):
        accepted, trim = _checkpoint_correction(5, 1)
        assert accepted == 0
        assert trim == 4


class TestEnvGate:
    @pytest.mark.parametrize("value", ["0", "off", "false", "no", "OFF", "False"])
    def test_falsey_spellings_disable(self, monkeypatch, value):
        monkeypatch.setenv("VMLX_DFLASH2_PREFIX_REUSE", value)
        assert _prefix_reuse_enabled() is False

    @pytest.mark.parametrize("value", [None, "1", "on", "true", "maybe"])
    def test_default_and_truthy_enable(self, monkeypatch, value):
        if value is None:
            monkeypatch.delenv("VMLX_DFLASH2_PREFIX_REUSE", raising=False)
        else:
            monkeypatch.setenv("VMLX_DFLASH2_PREFIX_REUSE", value)
        assert _prefix_reuse_enabled() is True


class TestAssistantTagCut:
    class _Tok:
        def convert_tokens_to_ids(self, s):
            assert s == "<|im_start|>"
            return 100

    def test_cuts_at_last_tag(self):
        from vmlx_engine.dflash2_runtime import _assistant_tag_cut

        # sys/user turns then the generation tag at index 6
        prompt = [100, 1, 2, 100, 3, 4, 100, 5]
        assert _assistant_tag_cut(self._Tok(), prompt, 0) == 6

    def test_tag_inside_cached_region_is_rejected(self):
        from vmlx_engine.dflash2_runtime import _assistant_tag_cut

        prompt = [100, 1, 2, 100, 3]
        assert _assistant_tag_cut(self._Tok(), prompt, 4) is None

    def test_no_tag_returns_none(self):
        from vmlx_engine.dflash2_runtime import _assistant_tag_cut

        assert _assistant_tag_cut(self._Tok(), [1, 2, 3], 0) is None

    def test_tokenizer_without_vocab_returns_none(self):
        from vmlx_engine.dflash2_runtime import _assistant_tag_cut

        assert _assistant_tag_cut(object(), [100, 1], 0) is None


class TestDescribeMisses:
    def test_reports_common_prefix_per_entry(self):
        store = _DFlash2SessionStore(max_entries=4)
        store.put(_entry("m", [1, 2, 3, 7, 8], cache_len=4))
        store.put(_entry("m", [1, 2], cache_len=2))
        out = store.describe_misses("m", [1, 2, 3, 9])
        assert out == ["turn len=5 cached=4 common=3", "turn len=2 cached=2 common=2"]

    def test_other_models_excluded(self):
        store = _DFlash2SessionStore()
        store.put(_entry("other", [1, 2, 3]))
        assert store.describe_misses("m", [1, 2, 3]) == []


class TestRotatingCacheResumeMath:
    """The resume path rebuilds an EMPTY drafter RotatingKVCache at the
    absolute conversation position (drafter RoPE and the session splice/trim
    read that offset). Upstream reads ``offset`` as the physical fill, so a
    forced offset (a) sized growth negatively at 14k into a 2047 window
    (``[full] Negative dimensions not allowed``, qwen 4D matrix turn 3) and
    (b) fed the drafter up to step-1 = 255 zero K/V rows after any 1-row
    update (returned view ``keys[:offset]`` and ``_temporal_order`` reading
    ``_idx < offset`` as wrapped; audit 2026-10-07/08). Rebased caches must
    behave exactly like a natural cache that started at 0. Every assertion
    runs the REAL mlx_lm cache class."""

    def _caches(self, base, max_size=31, step=8):
        import mlx.core as mx
        from mlx_lm.models.cache import RotatingKVCache

        from vmlx_engine.dflash2_runtime import _rebase_drafter_cache

        natural = RotatingKVCache(max_size=max_size, keep=0)
        natural.step = step
        rebased = RotatingKVCache(max_size=max_size, keep=0)
        rebased.step = step
        _rebase_drafter_cache([rebased], base)
        return mx, natural, rebased

    @staticmethod
    def _rows(mx, start, count):
        return mx.arange(start, start + count, dtype=mx.float32).reshape(1, 1, count, 1)

    # Drafter-shaped sequence: multi-row context appends (concat path) mixed
    # with 1-row appends (in-place path, a cycle that accepted nothing).
    SEQUENCE = (5, 1, 6, 1, 1, 9, 1, 40, 1, 3, 1, 1, 1, 1, 12)

    @pytest.mark.parametrize("base", [7, 500, 3000, 14000])
    def test_rebased_cache_returns_exactly_the_natural_rows(self, base):
        mx, natural, rebased = self._caches(base)
        start = 1.0
        for size in self.SEQUENCE:
            x = self._rows(mx, start, size)
            start += size
            want, _ = natural.update_and_fetch(x, x)
            got, _ = rebased.update_and_fetch(x, x)
            assert got.shape == want.shape
            assert bool(mx.all(got == want))
            assert int((got == 0).sum()) == 0
            assert rebased.offset == natural.offset + base

    @pytest.mark.parametrize("base", [7, 3000])
    def test_rebased_cache_trim_matches_natural(self, base):
        """dflash ``_trim_recent_cache`` calls ``_temporal_order`` and edits
        ``offset`` directly; the rebased cache must stay row-identical."""
        from dflash.model_mlx import _trim_recent_cache

        mx, natural, rebased = self._caches(base)
        start = 1.0
        for size, trim in ((9, 2), (1, 1), (6, 3), (1, 0), (40, 5), (1, 1), (4, 2)):
            x = self._rows(mx, start, size)
            start += size
            natural.update_and_fetch(x, x)
            rebased.update_and_fetch(x, x)
            _trim_recent_cache([natural], trim)
            _trim_recent_cache([rebased], trim)
            want, _ = natural.state
            got, _ = rebased.state
            assert bool(mx.all(got == want)) and got.shape == want.shape
            assert rebased.offset == natural.offset + base

    def test_meta_state_round_trip_keeps_the_base(self):
        """Session clones and the SSD tier rebuild caches through meta_state."""
        from mlx_lm.models.cache import RotatingKVCache

        mx, _, rebased = self._caches(3000)
        x = self._rows(mx, 1.0, 5)
        rebased.update_and_fetch(x, x)
        clone = RotatingKVCache(max_size=31, keep=0)
        clone.state = rebased.state
        clone.meta_state = rebased.meta_state
        assert clone.offset == rebased.offset and clone.__dict__.get("_vmlx_base") == 3000
        y = self._rows(mx, 6.0, 1)
        want, _ = rebased.update_and_fetch(y, y)
        got, _ = clone.update_and_fetch(y, y)
        assert bool(mx.all(got == want)) and got.shape == want.shape
        plain = RotatingKVCache(max_size=31, keep=0)
        plain.meta_state = ("0", "31", "4", "4")      # upstream 4-field meta: no base
        assert plain.__dict__.get("_vmlx_base") == 0

    def test_forced_offset_far_past_window_survives_single_row_updates(self):
        """The original crash shape: 14k into a 2047 window, short delta, then
        1-row updates while the window is under-filled."""
        mx, natural, rebased = self._caches(14000, max_size=2047, step=256)
        x = self._rows(mx, 1.0, 5)
        rebased.update_and_fetch(x, x)
        natural.update_and_fetch(x, x)
        for i in range(300):
            t = self._rows(mx, 6.0 + i, 1)
            got, _ = rebased.update_and_fetch(t, t)
            want, _ = natural.update_and_fetch(t, t)
            mx.eval(got, want)
        assert bool(mx.all(got == want)) and got.shape == want.shape

    def test_unrebased_cache_runs_upstream_code(self):
        mx, natural, _ = self._caches(1)
        for i in range(40):
            t = self._rows(mx, float(i + 1), 1)
            k, _ = natural.update_and_fetch(t, t)
            if natural.offset < 31:
                assert k.shape[2] == natural.offset
        assert natural.keys.shape[2] == 31 and natural.offset == 40
        assert "_vmlx_base" not in natural.__dict__

    def test_patch_is_idempotent(self):
        from vmlx_engine.dflash2_runtime import (
            _patch_rotating_cache_resume_math,
        )

        _patch_rotating_cache_resume_math()
        _patch_rotating_cache_resume_math()


def test_system_end_cut_marks_the_first_tag_after_a_system_message():
    from vmlx_engine.dflash2_runtime import _system_end_cut

    class Tok:
        def convert_tokens_to_ids(self, t):
            return {"<|im_start|>": 900}[t]

        def encode(self, text, add_special_tokens=False):
            return {"system": [11]}[text]

    sys_prompt = [900, 11, 5, 6, 7, 901, 900, 12, 8, 901, 900, 13]
    assert _system_end_cut(Tok(), sys_prompt, 0) == 6
    assert _system_end_cut(Tok(), sys_prompt, 6) is None          # already cached
    assert _system_end_cut(Tok(), [900, 12, 8, 901, 900, 13], 0) is None   # no system message
    # MiMo-V2.6: tools as a first system block, then the conversation's system prompt: the cut is after both
    two_systems = [900, 11, 70, 71, 901, 900, 11, 5, 6, 901, 900, 12, 8, 901, 900, 13]
    assert _system_end_cut(Tok(), two_systems, 0) == 10
    assert _system_end_cut(Tok(), [900, 11, 70, 901, 900, 11, 5, 901], 0) is None   # nothing after the systems


def test_system_block_cuts_snapshot_each_leading_system_block():
    """MiMo lanes snapshot after the tools block AND after the system prompt: a turn whose system prompt changed
    (the app's differs between settings) still restores the tools block."""
    from vmlx_engine.dflash2_runtime import _system_block_cuts
    from vmlx_engine.dspark import session as dsession

    class Tok:
        def convert_tokens_to_ids(self, t):
            return {"<|im_start|>": 900, "<|im_end|>": 901}.get(t, -1)

        def encode(self, text, add_special_tokens=False):
            return {"system": [11], "assistant": [14]}.get(text, [99])

    two_systems = [900, 11, 70, 71, 901, 900, 11, 5, 6, 901, 900, 12, 8, 901, 900, 14]
    assert _system_block_cuts(Tok(), two_systems, 0) == [5, 10]
    assert _system_block_cuts(Tok(), two_systems, 5) == [10]           # the tools block is already restored
    assert _system_block_cuts(Tok(), two_systems, 10) == []
    one_system = [900, 11, 5, 6, 7, 901, 900, 12, 8, 901, 900, 14]
    assert _system_block_cuts(Tok(), one_system, 0) == [6]
    if dsession.reuse_enabled():
        kinds = dsession.snapshot_cuts(Tok(), two_systems, 0)
        assert [c for k, c in kinds if k == "system"] == [5, 10]


def test_full_undo_rollback_restores_the_forward_start_on_mlx_vlm_targets():
    # An AR burst that launched one step past EOS rolls back with accepted=-1 (_checkpoint_correction(1, 0)).
    # mlx_vlm's rollback_speculative_cache would replay a zero-row GDN update whose lazy state fails at eval; the
    # capture must restore the forward's initial state and conv window itself.
    import mlx.core as mx
    from types import SimpleNamespace
    from mlx_lm.models.cache import ArraysCache, KVCache
    from mlx_vlm.models.qwen3_5.language import LanguageModel

    from vmlx_engine.dflash2_runtime import _VLMGDNStateCapture

    assert _checkpoint_correction(1, 0) == (-1, 1)
    B, T, H, D, K = 1, 1, 2, 4, 4
    initial = mx.random.normal((B, H, D, D))
    conv_input = mx.random.normal((B, T + K - 1, 8))
    rows = [mx.random.normal((B, T, H, D)) for _ in range(3)]
    gdn = (*rows, mx.random.normal((B, T, H)), mx.random.normal((B, T, H)), mx.zeros((H,)), mx.zeros((H,)),
           initial, None, conv_input, K)
    delegated = []

    def rollback_speculative_cache(caches, gdn_states, accepted, block_size):
        delegated.append(accepted)
        return LanguageModel.rollback_speculative_cache(None, caches, gdn_states, accepted, block_size)

    capture = _VLMGDNStateCapture(SimpleNamespace(
        _target=SimpleNamespace(rollback_speculative_cache=rollback_speculative_cache), gdn_states=[gdn]))
    kv, ssm = KVCache(), ArraysCache(size=2)
    kv.update_and_fetch(mx.zeros((B, H, 5, D)), mx.zeros((B, H, 5, D)))
    ssm[0], ssm[1] = mx.zeros((B, K - 1, 8)), mx.zeros_like(initial)
    capture.rollback([kv, ssm], -1, 1)
    mx.eval(ssm[0], ssm[1])
    assert delegated == []
    assert kv.offset == 4
    assert mx.array_equal(ssm[1], initial).item()
    assert mx.array_equal(ssm[0], conv_input[:, : K - 1]).item()


def test_chain_after_makes_a_full_write_the_parent_of_the_next_entries():
    """A request whose resume parent is unusable (none / chain at MAX_CHAIN) writes its first entry in full; its later
    entries (boundary -> turn) must delta from that entry, not be written in full again. A usable parent is kept."""
    from concurrent.futures import Future

    from vmlx_engine.dflash2_session_disk import MAX_CHAIN
    from vmlx_engine.dspark.session import chain_after

    def receipt(ref, durable=True):
        f = Future(); f.vmlx_ssd_ref = ref; f.set_result({"durable": durable}); return f

    deep = {"sig": "a", "len": 100, "depth": MAX_CHAIN, "receipt": None}
    r = receipt({"sig": "b", "len": 200, "depth": 0})
    nxt = chain_after(deep, r)
    assert nxt["sig"] == "b" and nxt["depth"] == 0 and nxt["receipt"] is r
    assert chain_after(None, r)["sig"] == "b"
    ok = {"sig": "a", "len": 100, "depth": 2, "receipt": None}
    assert chain_after(ok, r) is ok                        # usable parent: unchanged (depth +1 per request)
    assert chain_after(deep, receipt({"sig": "c", "len": 5, "depth": 0}, durable=False)) is deep   # refused write
    pending = Future(); pending.vmlx_ssd_ref = {"sig": "d", "len": 7, "depth": 0}
    assert chain_after(None, pending)["sig"] == "d"        # queued write: FIFO writer settles it before the child
