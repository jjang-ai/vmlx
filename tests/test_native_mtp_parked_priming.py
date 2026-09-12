from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from vmlx_engine import native_mtp_prompt_priming as priming
from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator, MLLMNativeMTPState


class _Cache:
    def __init__(self, offset):
        self.offset = offset
        self.pairs = [(-1, token) for token in range(offset)]
        self.keys = mx.zeros((1, 1, offset, 1))
        self.values = self.keys

    def is_trimmable(self):
        return True

    def trim(self, count):
        self.offset -= count
        del self.pairs[self.offset:]
        self.keys = self.keys[:, :, :self.offset]
        self.values = self.keys
        return count


class _Host:
    mtp = object()

    def __init__(self):
        self.calls = []
        self.fail = False

    def make_mtp_cache(self):
        return [_Cache(0)]

    def mtp_forward(self, hidden, tokens, cache):
        if self.fail:
            raise RuntimeError("controlled head failure")
        pairs = list(zip(hidden.reshape(-1).tolist(), tokens.reshape(-1).tolist()))
        self.calls.append(pairs)
        for entry in cache:
            entry.pairs.extend(pairs)
            entry.offset += len(pairs)
            entry.keys = mx.array(entry.pairs)[:, 0].reshape(1, 1, -1, 1)
            entry.values = mx.array(entry.pairs)[:, 1].reshape(1, 1, -1, 1)
        return mx.zeros((1, len(pairs), 8))


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("VMLX_NATIVE_MTP_PROMPT_PRIMING", "1")
    monkeypatch.setenv("VMLX_NATIVE_MTP_PARKED_PRIMING", "1")
    monkeypatch.setenv("VMLX_NATIVE_MTP_AR_REENTRY", "1")
    monkeypatch.delenv("VMLINUX_NATIVE_MTP_AR_REENTRY", raising=False)


def _park(host=None, backbone=None, head=None):
    host = host or _Host()
    backbone = backbone or [_Cache(8)]
    head = head or [_Cache(6)]
    result = priming.park_after_confirmed(
        host, request_id="owner", backbone_cache=backbone, mtp_cache=head,
        confirmed_hidden=mx.array([[[60], [70]]]),
        confirmed_tokens=mx.array([[7, 8]], dtype=mx.uint32),
    )
    return result, host, backbone, head


def _ar(host, backbone, token):
    backbone[0].offset += 1
    priming.capture_prefill(
        host, mx.array([[token]], dtype=mx.uint32),
        mx.array([[[token * 10]]]), backbone,
    )


def test_default_off_does_not_touch_head(monkeypatch):
    monkeypatch.delenv("VMLX_NATIVE_MTP_PARKED_PRIMING", raising=False)
    ok, host, _, head = _park()
    assert not ok
    assert head[0].offset == 6
    assert host.calls == []
    assert not priming.capture_requested(host)


def test_commit_ar_and_reentry_have_exact_once_hidden_token_pairs(enabled):
    ok, host, backbone, head = _park()
    assert ok and priming.parked_context_active(host)
    assert backbone[0].offset == 8  # never advanced by draft state
    _ar(host, backbone, 8)  # handoff token was already committed to head
    assert host.calls == [[(60, 7), (70, 8)]]
    _ar(host, backbone, 9)
    assert len(host.calls) == 1  # AR pairs buffered, no per-token head call
    backbone[0].offset += 1  # real reseed forward skips capture
    result = priming.take_primed(host, backbone, mx.array([10]))
    assert result is not None and result[0] is head and result[1] == 10
    assert head[0].pairs[-4:] == [(60, 7), (70, 8), (80, 9), (90, 10)]
    assert not priming.capture_requested(host)
    assert priming.prime_stats(host)["last"]["reason"] == "park_resumed"


def test_buffer_is_bounded_and_prompt_window_does_not_truncate_parked_history(
    enabled, monkeypatch,
):
    monkeypatch.setenv("VMLX_NATIVE_MTP_PRIME_WINDOW", "1")
    ok, host, backbone, head = _park()
    assert ok
    _ar(host, backbone, 8)
    for token in range(9, 74):
        _ar(host, backbone, token)
        ctx = getattr(host, priming._CTX_ATTR)
        assert len(ctx.pending_pairs) < priming._PARK_FOLD_BLOCK
    assert [len(call) for call in host.calls] == [2, 32, 32]
    backbone[0].offset += 1
    assert priming.take_primed(host, backbone, mx.array([74]))[1] == 74
    assert head[0].pairs[6:] == [(token * 10 - 10, token) for token in range(7, 75)]


@pytest.mark.parametrize("delta", [-1, 1])
def test_noncontiguous_ar_offset_drops_history(enabled, delta):
    _, host, backbone, head = _park()
    backbone[0].offset += delta
    _ar(host, backbone, 8)
    assert not priming.capture_requested(host)
    assert head[0].offset == 8


def test_equal_offset_foreign_backbone_is_not_accepted(enabled):
    _, host, _, head = _park()
    other = [_Cache(8)]
    _ar(host, other, 8)
    assert not priming.capture_requested(host)
    assert head[0].offset == 8


def test_new_offset_proxy_for_same_cache_is_accepted(enabled):
    _, host, backbone, _ = _park()
    backbone[0].offset = 9
    proxy = SimpleNamespace(_inner=backbone[0], offset=9)
    priming.capture_prefill(host, mx.array([[8]]), mx.array([[[80]]]), [proxy])
    assert priming.parked_context_active(host)
    backbone[0].offset = 10
    assert priming.take_primed(host, backbone, mx.array([9]))[1] == 9


@pytest.mark.parametrize("shape", [(2, 1), (1, 2)])
def test_batch_or_prefill_cannot_adopt_parked_context(enabled, shape):
    _, host, backbone, head = _park()
    backbone[0].offset += shape[1]
    priming.capture_prefill(
        host, mx.ones(shape, dtype=mx.uint32), mx.ones((*shape, 1)), backbone,
    )
    assert not priming.capture_requested(host)
    assert head[0].offset == 8


@pytest.mark.parametrize("foreign", [False, True])
def test_reentry_seam_rejects_rewind_or_foreign_cache_before_flush(enabled, foreign):
    _, host, backbone, head = _park()
    _ar(host, backbone, 8)
    _ar(host, backbone, 9)
    candidate = [_Cache(11)] if foreign else backbone  # seed offset missing
    assert priming.take_primed(host, candidate, mx.array([10])) is None
    assert head[0].offset == 8 and len(host.calls) == 1
    assert not priming.capture_requested(host)


def test_all_head_layers_must_match_before_commit(enabled):
    ok, host, _, head = _park(head=[_Cache(6), _Cache(5)])
    assert not ok and host.calls == []
    assert [entry.offset for entry in head] == [6, 5]


@pytest.mark.parametrize("when", ["commit", "flush", "seam"])
def test_head_failure_falls_back_without_writing_backbone(enabled, when):
    host = _Host()
    host.fail = when == "commit"
    ok, host, backbone, head = _park(host=host)
    if when == "commit":
        assert not ok and backbone[0].offset == 8
        return
    assert ok
    _ar(host, backbone, 8)
    host.fail = True
    if when == "flush":
        for token in range(9, 41):
            _ar(host, backbone, token)
        assert backbone[0].offset == 41
    else:
        backbone[0].offset += 1
        assert priming.take_primed(host, backbone, mx.array([9])) is None
        assert backbone[0].offset == 10
    assert not priming.capture_requested(host)
    assert head[0].offset == 8


def test_cleanup_is_request_owned_and_new_prefill_replaces_context(enabled):
    _, host, _, _ = _park()
    priming.drop_parked_context(host, "different-owner")
    assert priming.parked_context_active(host)
    priming.prepare_prompt(
        host, request_id="new", prompt_tokens=[1, 2, 3], cached_tokens=0,
        prefix_cache=None,
    )
    assert not priming.parked_context_active(host)
    assert priming.capture_requested(host)  # new normal prompt plan survives
    priming.drop_parked_context(host, "owner")
    assert priming.capture_requested(host)


def _generator(host):
    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator.language_model = host
    generator._model_type = "qwen4_exp"
    generator._old_wired_limit = None
    generator._old_cache_limit = None
    return generator


def test_generator_trims_rejected_chain_before_parking(enabled, monkeypatch):
    from vmlx_engine import mllm_batch_generator as module

    monkeypatch.setattr(module, "_NATIVE_MTP_ALIGNED_HEAD_CACHE", True)
    host = _Host()
    generator = _generator(host)
    state = MLLMNativeMTPState(mtp_cache=[_Cache(8)], head_chain_pairs=2)
    state.stats.prompt_primed_pairs = 6
    state.mtp_cache[0].pairs[-2:] = [(999, 999), (999, 999)]
    assert generator._park_native_mtp_head_for_ar(
        SimpleNamespace(request_id="owner"), [_Cache(8)], state,
        mx.array([[[60], [70]]]), [mx.array([7]), mx.array([8])],
    )
    assert state.head_chain_pairs == 0
    assert state.mtp_cache[0].pairs[-2:] == [(60, 7), (70, 8)]
    assert state.stats.mtp_forwards == 1
    generator.close()
    assert not priming.capture_requested(host)


@pytest.mark.parametrize("reason", ["family", "unaligned", "unprimed", "no_reentry"])
def test_generator_declines_unsupported_ownership_without_trimming(
    enabled, monkeypatch, reason,
):
    from vmlx_engine import mllm_batch_generator as module

    monkeypatch.setattr(module, "_NATIVE_MTP_ALIGNED_HEAD_CACHE", reason != "unaligned")
    generator = _generator(_Host())
    if reason == "family":
        generator._model_type = "glm5_next"
    if reason == "no_reentry":
        monkeypatch.setenv("VMLX_NATIVE_MTP_AR_REENTRY", "0")
    state = MLLMNativeMTPState(mtp_cache=[_Cache(8)], head_chain_pairs=2)
    state.stats.prompt_primed_pairs = 0 if reason == "unprimed" else 6
    assert not generator._park_native_mtp_head_for_ar(
        SimpleNamespace(request_id="owner"), [_Cache(8)], state,
        mx.array([[[60], [70]]]), [mx.array([7]), mx.array([8])],
    )
    assert state.head_chain_pairs == 2 and state.mtp_cache[0].offset == 8
    assert generator.language_model.calls == []


def test_remove_releases_parked_ar_request_without_mtp_state(enabled):
    _, host, backbone, _ = _park()
    generator = _generator(host)
    generator.active_batch = SimpleNamespace(
        uids=[3], requests=[SimpleNamespace(request_id="owner")], cache=backbone,
    )
    generator.unprocessed_requests = []
    generator.remove([3])
    assert generator.active_batch is None
    assert not priming.capture_requested(host)


@pytest.mark.parametrize("flag", [
    "VMLX_NATIVE_MTP_PROMPT_PRIMING", "VMLX_NATIVE_MTP_PARKED_PRIMING",
])
def test_runtime_disable_drops_buffered_state(enabled, monkeypatch, flag):
    _, host, backbone, head = _park()
    _ar(host, backbone, 8)
    monkeypatch.setenv(flag, "0")
    _ar(host, backbone, 9)
    assert not priming.capture_requested(host)
    assert head[0].offset == 8
