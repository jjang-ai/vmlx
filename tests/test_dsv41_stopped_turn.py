# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1: stopping a turn keeps its prompt, so the next turn does not prefill it again.

Found with the real Stop button: T3 reused 43 of 597 tokens, because the salvage asked the generator for
take_prompt_snapshots, which the V4.1 generators do not have. The prompt snapshot sits on the generator's own request.
"""

from __future__ import annotations

import types

from vmlx_engine.scheduler import Scheduler


class Cache:
    def __init__(self, offset):
        self.offset = offset


DeepseekV41Cache = type("DeepseekV41Cache", (Cache,), {})


def _sched(live, *, block=True):
    s = Scheduler.__new__(Scheduler)
    s._m3_aligned_prefix = False
    s._uses_dsv41_cache = True
    s.batch_generator = types.SimpleNamespace(_find=lambda uid: live)
    s.block_aware_cache = object() if block else None
    s.memory_aware_cache = None
    s.stored = []
    s._store_aborted_prompt_snapshot = lambda rid, req, snap, key: s.stored.append(("prompt", list(key), snap))
    s._store_dsv41_cut_checkpoints = lambda req, rid, kw=None: s.stored.append(("reply", req._dsv41_generation_end))
    return s


def _request(prompt, output):
    return types.SimpleNamespace(prompt_token_ids=list(prompt), output_token_ids=list(output), _cache_extra_keys=None)


def test_prompt_and_partial_reply_are_kept(monkeypatch):
    monkeypatch.delenv("VMLX_DSV41_GENERATION_END_CACHE", raising=False)
    prompt, output = list(range(100)), list(range(500, 540))
    snap = [DeepseekV41Cache(99)]
    raw = [DeepseekV41Cache(139)]
    s = _sched(types.SimpleNamespace(cache=raw, prompt_cache_snapshot=snap))
    assert s._salvage_stopped_request(1, "r", _request(prompt, output)) is True
    assert s.stored[0] == ("prompt", prompt[:99], snap)
    kind, entry = s.stored[1]
    assert kind == "reply" and entry[0] == (prompt + output)[:139] and entry[1] is raw


def test_prefill_not_finished_keeps_nothing():
    s = _sched(types.SimpleNamespace(cache=[DeepseekV41Cache(40)], prompt_cache_snapshot=None))
    assert s._salvage_stopped_request(1, "r", _request(list(range(100)), [])) is False
    assert s.stored == []


def test_snapshot_lanes_out_of_step_keeps_nothing():
    snap = [DeepseekV41Cache(99), DeepseekV41Cache(98)]
    s = _sched(types.SimpleNamespace(cache=[DeepseekV41Cache(99)], prompt_cache_snapshot=snap))
    assert s._salvage_stopped_request(1, "r", _request(list(range(100)), [])) is False
    assert s.stored == []


def test_reply_rows_past_the_tokens_emitted_are_not_stored(monkeypatch):
    """A stop mid-burst can leave the cache ahead of what was emitted: the prompt is kept, the reply entry is not."""
    monkeypatch.delenv("VMLX_DSV41_GENERATION_END_CACHE", raising=False)
    prompt = list(range(100))
    s = _sched(types.SimpleNamespace(cache=[DeepseekV41Cache(150)], prompt_cache_snapshot=[DeepseekV41Cache(99)]))
    assert s._salvage_stopped_request(1, "r", _request(prompt, [1, 2])) is True
    assert s.stored[0][0] == "prompt"
    assert s.stored[1] == ("reply", None)
