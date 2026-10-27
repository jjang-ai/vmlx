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
