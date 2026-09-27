from __future__ import annotations

import threading
import logging
from collections import OrderedDict
from types import SimpleNamespace

import mlx.core as mx

from vmlx_engine.mllm_batch_generator import MLLMNativeMTPStats
from vmlx_engine.native_mtp_prompt_priming import (
    capture_requested,
    capture_prefill,
    drop_context,
    prepare_prompt,
    prime_stats,
    take_primed,
)
from vmlx_engine.prefix_cache import BlockAwarePrefixCache


class _Cache:
    def __init__(self, offset: int = 0):
        self.offset = offset
        self.keys = mx.zeros((1, 1, offset, 1)) if offset else None
        self.values = mx.zeros((1, 1, offset, 1)) if offset else None

    def append(self, count: int) -> None:
        self.offset += count
        self.keys = mx.zeros((1, 1, self.offset, 1))
        self.values = mx.zeros((1, 1, self.offset, 1))

    def trim(self, count: int) -> int:
        count = min(max(0, int(count)), self.offset)
        self.offset -= count
        self.keys = (
            mx.zeros((1, 1, self.offset, 1)) if self.offset else None
        )
        self.values = (
            mx.zeros((1, 1, self.offset, 1)) if self.offset else None
        )
        return count


class _Host:
    def __init__(self):
        self.mtp = object()
        self.calls: list[list[int]] = []

    def make_mtp_cache(self):
        return [_Cache()]

    def mtp_forward(self, hidden, tokens, cache):
        del hidden
        ids = [int(token) for token in tokens.reshape(-1).tolist()]
        self.calls.append(ids)
        cache[0].append(len(ids))
        return mx.zeros((1, len(ids), 8))


class _SidecarStore:
    block_size = 2

    def __init__(self):
        self.snapshot = None

    def store_mtp_prefix_snapshot(self, tokens, boundary, snapshot, **kwargs):
        del tokens, kwargs
        assert boundary == snapshot.boundary_tokens
        self.snapshot = snapshot
        return True

    def restore_mtp_prefix_snapshot(self, tokens, boundary, **kwargs):
        del tokens, kwargs
        if self.snapshot is not None and self.snapshot.boundary_tokens == boundary:
            return self.snapshot
        return None


def test_capture_requested_tracks_only_an_armed_prompt_timeline():
    host = _Host()
    assert not capture_requested(host)
    prepare_prompt(
        host,
        request_id="armed",
        prompt_tokens=[1, 2, 3],
        cached_tokens=0,
        prefix_cache=None,
    )
    assert capture_requested(host)
    drop_context(host)
    assert not capture_requested(host)


def test_scheduler_arms_dense_qwen35_only_with_measured_opt_in(monkeypatch):
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator.language_model = _Host()
    generator.block_aware_cache = None
    generator._native_mtp_disabled_reason_for_request = lambda _request: None
    request = SimpleNamespace(
        request_id="q35",
        max_tokens=32,
        _original_token_ids=[1, 2, 3],
        _cached_tokens=0,
        _cache_extra_keys=None,
    )

    generator._model_type = "qwen3_5"
    assert not generator._prepare_native_mtp_prompt_priming(request)
    assert not capture_requested(generator.language_model)

    monkeypatch.setenv("VMLX_QWEN35_MTP_PROMPT_PRIMING", "1")
    assert not generator._prepare_native_mtp_prompt_priming(request)
    assert capture_requested(generator.language_model)

    generator._model_type = "deepseek_v4"
    assert not generator._prepare_native_mtp_prompt_priming(request)
    assert not capture_requested(generator.language_model)


def test_cold_prompt_is_folded_and_published_at_terminal_boundary():
    # The prefix cache restores a predecessor at its N-1 terminal even when the
    # prompt length is block-aligned (live: 5952-token prompt restored at 5951),
    # so the sidecar is keyed at N-1 = 3, not the full block 4.
    host = _Host()
    sidecar = _SidecarStore()
    assert not prepare_prompt(
        host,
        request_id="cold",
        prompt_tokens=[1, 2, 3, 4],
        cached_tokens=0,
        prefix_cache=sidecar,
    )

    backbone = [_Cache(offset=4)]
    capture_prefill(
        host,
        mx.array([[1, 2, 3, 4]]),
        mx.arange(8).reshape(1, 4, 2),
        backbone,
    )
    assert host.calls == [[2, 3, 4]]
    assert prime_stats(host) == {
        "plan_armed": True,
        "active": True,
        "folded_pairs": 3,
        "window_exceeded": False,
        "last": {
            "reason": "captured",
            "folded_pairs": 3,
            "expected_offset": 4,
        },
    }

    backbone[0].offset = 5  # the first ordinary decode/seed forward
    primed = take_primed(host, backbone, mx.array([5]))
    assert primed is not None
    mtp_cache, folded = primed
    assert folded == 4
    assert mtp_cache[0].offset == 4
    assert host.calls[-1] == [5]
    assert sidecar.snapshot is not None
    assert sidecar.snapshot.boundary_tokens == 3
    assert sidecar.snapshot.mtp_cache[0].offset == 2


def test_warm_sidecar_continues_only_the_uncached_tail():
    source = _Host()
    sidecar = _SidecarStore()
    prepare_prompt(
        source,
        request_id="source",
        prompt_tokens=[1, 2, 3, 4],
        cached_tokens=0,
        prefix_cache=sidecar,
    )
    source_backbone = [_Cache(offset=4)]
    capture_prefill(
        source,
        mx.array([[1, 2, 3, 4]]),
        mx.zeros((1, 4, 2)),
        source_backbone,
    )
    source_backbone[0].offset = 5
    assert take_primed(source, source_backbone, mx.array([5])) is not None

    warm = _Host()
    # The predecessor's prefix is restored at its N-1 terminal (3 tokens).
    assert prepare_prompt(
        warm,
        request_id="warm",
        prompt_tokens=[1, 2, 3, 4, 6, 7],
        cached_tokens=3,
        prefix_cache=sidecar,
    )
    backbone = [_Cache(offset=6)]
    capture_prefill(
        warm,
        mx.array([[4, 6, 7]]),
        mx.ones((1, 3, 2)),
        backbone,
    )
    assert warm.calls == [[4, 6, 7]]
    backbone[0].offset = 7
    primed = take_primed(warm, backbone, mx.array([8]))
    assert primed is not None
    assert primed[0][0].offset == 6
    assert primed[1] == 6
    assert warm.calls[-1] == [8]



def test_refaulted_sidecar_restores_native_history_and_folds_only_suffix():
    from vmlx_engine.paged_cache import PagedCacheManager, compute_block_hash

    sidecar = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
    sidecar.block_size = 2
    sidecar._chained_prefix_index_hash = False
    sidecar._mtp_prefix_snapshots = OrderedDict()
    sidecar._mtp_prefix_snapshot_lock = threading.RLock()
    sidecar._prefix_index = {}
    first = compute_block_hash(None, [1, 2], extra_keys=None)
    last = compute_block_hash(first, [3], extra_keys=None)
    table = SimpleNamespace(block_ids=[20, 21], num_tokens=3)
    sidecar.paged_cache = SimpleNamespace(
        _lock=threading.RLock(),
        allocated_blocks={
            20: SimpleNamespace(block_hash=first, token_count=2, ref_count=1),
            21: SimpleNamespace(block_hash=last, token_count=1, ref_count=1),
        },
        compute_block_hash=PagedCacheManager.compute_block_hash,
        get_block_table=lambda request_id: table if request_id == "warm" else None,
    )
    sidecar._request_tables = {"warm": SimpleNamespace(block_table=table)}
    source = _Host()
    prepare_prompt(source, request_id="source", prompt_tokens=[1, 2, 3, 4],
                   cached_tokens=0, prefix_cache=sidecar)
    backbone = [_Cache(offset=4)]
    capture_prefill(source, mx.array([[1, 2, 3, 4]]),
                    mx.arange(8).reshape(1, 4, 2), backbone)
    backbone[0].offset = 5
    assert take_primed(source, backbone, mx.array([5])) is not None

    # The old global index was pruned; only the refaulted request owns blocks.
    assert not sidecar._prefix_index
    warm = _Host()
    assert prepare_prompt(warm, request_id="warm", prompt_tokens=[1, 2, 3, 4, 6, 7],
                          cached_tokens=3, prefix_cache=sidecar)
    backbone = [_Cache(offset=6)]
    capture_prefill(warm, mx.array([[4, 6, 7]]), mx.ones((1, 3, 2)), backbone)
    assert warm.calls == [[4, 6, 7]]
    backbone[0].offset = 7
    restored = take_primed(warm, backbone, mx.array([8]))
    assert restored is not None
    assert restored[0][0].offset == restored[1] == 6
    assert warm.calls == [[4, 6, 7], [8]]
    # A different caller cannot borrow that history merely by naming the tokens.
    assert not prepare_prompt(_Host(), request_id="other", prompt_tokens=[1, 2, 3, 4],
                              cached_tokens=3, prefix_cache=sidecar)


def test_warm_backbone_without_sidecar_does_not_invent_tail_only_history():
    host = _Host()
    sidecar = _SidecarStore()
    assert not prepare_prompt(
        host,
        request_id="warm-miss",
        prompt_tokens=[1, 2, 3, 4, 5, 6],
        cached_tokens=4,
        prefix_cache=sidecar,
    )
    backbone = [_Cache(offset=6)]
    capture_prefill(
        host,
        mx.array([[5, 6]]),
        mx.ones((1, 2, 2)),
        backbone,
    )
    assert host.calls == []
    backbone[0].offset = 7
    assert take_primed(host, backbone, mx.array([7])) is None


def test_cold_priming_does_not_depend_on_prefix_cache_being_enabled():
    host = _Host()
    assert not prepare_prompt(
        host,
        request_id="no-cache",
        prompt_tokens=[1, 2, 3],
        cached_tokens=0,
        prefix_cache=None,
    )
    backbone = [_Cache(offset=3)]
    capture_prefill(
        host,
        mx.array([[1, 2, 3]]),
        mx.zeros((1, 3, 2)),
        backbone,
    )
    backbone[0].offset = 4
    primed = take_primed(host, backbone, mx.array([4]))
    assert primed is not None
    assert primed[1] == 3


def test_seam_mismatch_fails_closed_instead_of_using_wrong_history():
    host = _Host()
    sidecar = _SidecarStore()
    prepare_prompt(
        host,
        request_id="rewind",
        prompt_tokens=[1, 2, 3, 4],
        cached_tokens=0,
        prefix_cache=sidecar,
    )
    backbone = [_Cache(offset=4)]
    capture_prefill(
        host,
        mx.array([[1, 2, 3, 4]]),
        mx.zeros((1, 4, 2)),
        backbone,
    )
    backbone[0].offset = 99
    assert take_primed(host, backbone, mx.array([5])) is None
    assert host.calls == [[2, 3, 4]]


def test_block_cache_sidecar_is_aligned_bounded_and_requires_live_tip():
    cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
    cache.block_size = 2
    cache._mtp_prefix_snapshots = OrderedDict()
    cache._mtp_prefix_snapshot_lock = threading.RLock()
    live = SimpleNamespace(get_block=lambda _key: object())
    cache.paged_cache = SimpleNamespace(
        cached_block_hash_to_block=live,
        compute_block_hash=lambda tokens: repr(list(tokens)),
    )

    marker = object()
    assert cache.store_mtp_prefix_snapshot([1, 2, 3, 4], 3, marker)
    assert not cache.store_mtp_prefix_snapshot([1, 2, 3, 4, 5], 3, marker)
    assert cache.store_mtp_prefix_snapshot([1, 2, 3, 4], 4, marker)
    assert cache.restore_mtp_prefix_snapshot([1, 2, 3, 4], 4) is marker

    cache.paged_cache.cached_block_hash_to_block = SimpleNamespace(
        get_block=lambda _key: None
    )
    assert cache.restore_mtp_prefix_snapshot([1, 2, 3, 4], 4) is None


def test_partial_n_minus_one_sidecar_uses_live_prefix_index_chain(caplog):
    caplog.set_level(logging.INFO, logger="vmlx_engine.prefix_cache")
    cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
    cache.block_size = 2
    cache._mtp_prefix_snapshots = OrderedDict()
    cache._mtp_prefix_snapshot_lock = threading.RLock()
    cache._prefix_index = {}
    cache.paged_cache = SimpleNamespace(
        compute_block_hash=lambda tokens: repr(list(tokens))
    )
    cache._prefix_index_blocks_are_current = lambda *args, **kwargs: True

    tokens = [1, 2, 3, 4]
    marker = object()
    assert cache.store_mtp_prefix_snapshot(tokens, 3, marker)
    partial_key = cache._prefix_index_hash(tokens[:3])
    cache._prefix_index[partial_key] = (tokens[:3], [10, 11], None)
    assert cache.restore_mtp_prefix_snapshot(tokens, 3) is marker

    cache._prefix_index_blocks_are_current = lambda *args, **kwargs: False
    assert cache.restore_mtp_prefix_snapshot(tokens, 3) is None
    assert "boundary=3 reason=partial_chain_stale" in caplog.text
    assert "[1, 2, 3" not in caplog.text


def test_native_mtp_stats_expose_prompt_priming_provenance():
    stats = MLLMNativeMTPStats(
        prompt_primed_pairs=127,
        prompt_prime_source="restored_prefix_and_tail",
    )
    payload = stats.to_dict(
        request_id="r",
        finish_reason="stop",
        final_depth=3,
    )
    assert payload["prompt_priming"] == {
        "source": "restored_prefix_and_tail",
        "folded_pairs": 127,
    }


def test_prime_stats_distinguishes_armed_plan_from_captured_context():
    host = _Host()
    assert prime_stats(host) == {
        "plan_armed": False,
        "active": False,
        "folded_pairs": 0,
        "window_exceeded": False,
        "last": {},
    }
    assert not prepare_prompt(
        host,
        request_id="armed-only",
        prompt_tokens=[1, 2, 3],
        cached_tokens=0,
        prefix_cache=None,
    )
    assert prime_stats(host) == {
        "plan_armed": True,
        "active": False,
        "folded_pairs": 0,
        "window_exceeded": False,
        "last": {
            "reason": "armed",
            "prompt_tokens": 3,
            "cached_tokens": 0,
        },
    }


def test_prompt_restore_miss_records_reason_without_fabricating_history(caplog):
    caplog.set_level(logging.INFO, logger="vmlx_engine.native_mtp_prompt_priming")
    host = _Host()
    assert not prepare_prompt(
        host,
        request_id="missing-sidecar",
        prompt_tokens=[11, 22, 33, 44],
        cached_tokens=3,
        prefix_cache=_SidecarStore(),
    )
    stats = prime_stats(host)
    assert not stats["active"]
    assert stats["last"] == {
        "reason": "restore_snapshot_missing_or_type", "cached_tokens": 3
    }
    assert "request=missing-sidecar boundary=3" in caplog.text
    assert "[11, 22" not in caplog.text
    assert not host.calls
