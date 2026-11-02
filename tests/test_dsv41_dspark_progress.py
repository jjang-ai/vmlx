"""DSV4.1 DSpark generator reports prefill progress to the server's stream guard (request_progress)."""
from types import SimpleNamespace

from vmlx_engine.utils.dsv41_dspark_generator import DSV41DSparkGenerator


def _gen(req):
    gen = DSV41DSparkGenerator.__new__(DSV41DSparkGenerator)  # no model: only the probe is exercised
    gen._find = lambda uid: req if req is not None and req.uid == uid else None
    return gen


def test_progress_follows_the_session_position_during_prefill():
    seq = SimpleNamespace(pos=0)
    req = SimpleNamespace(uid=7, cache=[SimpleNamespace(_seq=seq), SimpleNamespace(_seq=seq)])
    gen = _gen(req)
    assert gen.request_progress(7) == 0
    seq.pos = 512           # one prefill chunk forwarded
    assert gen.request_progress(7) == 512
    seq.pos = 1_040_000     # a 1M prompt keeps advancing with no output token
    assert gen.request_progress(7) == 1_040_000


def test_unknown_request_or_no_cache_reads_none():
    gen = _gen(SimpleNamespace(uid=7, cache=None))
    assert gen.request_progress(8) is None
    assert gen.request_progress(7) is None


def test_finished_dspark_stats_reach_scheduler_once_for_the_matching_request():
    from collections import OrderedDict
    from vmlx_engine.scheduler import _text_lane_dspark_spec_stats_for

    gen = _gen(None)
    gen._spec_stats = OrderedDict()
    gen._runner = SimpleNamespace(adaptive=True)
    req = SimpleNamespace(uid=7, _dspark_stats={
        "steps": 10, "drafted": 20, "accepted": 15, "emitted": 27,
        "spec_steps": 8, "ar_steps": 2, "spec_s_per_tok": 0.01, "ar_s_per_tok": 0.03,
    })
    gen._log_dspark_stats(req)
    assert _text_lane_dspark_spec_stats_for(gen, 8) is None
    stats = _text_lane_dspark_spec_stats_for(gen, 7)
    assert stats is not None
    assert stats["kind"] == "dspark" and stats["lane"] == "deepseek-v4.1"
    assert stats["cycles"] == 10 and stats["emitted_per_cycle"] == 2.7
    assert stats["acceptance"] == 0.75 and stats["ar_tokens"] == 2
    assert stats["spec_steps"] == 8 and stats["adaptive_governor"] is True
    assert _text_lane_dspark_spec_stats_for(gen, 7) is None


def test_plain_request_does_not_replay_prior_native_dspark_stats():
    from collections import OrderedDict
    from vmlx_engine.scheduler import _text_lane_dspark_spec_stats_for

    gen = _gen(None)
    gen._spec_stats = OrderedDict()
    gen._runner = SimpleNamespace(adaptive=True)
    gen._log_dspark_stats(SimpleNamespace(uid=1, _dspark_stats={
        "steps": 2, "drafted": 3, "accepted": 2, "emitted": 4,
        "spec_steps": 2, "ar_steps": 0,
    }))
    gen._log_dspark_stats(SimpleNamespace(uid=2, _dspark_stats={}))
    assert _text_lane_dspark_spec_stats_for(gen, 2) is None
    assert _text_lane_dspark_spec_stats_for(gen, 1)["tokens"] == 4
