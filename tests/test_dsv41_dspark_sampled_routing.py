"""DSV41DSparkGenerator routes sampled requests to exact sampled speculation (SpecSampler) and keeps the AR path for
logits processors / logprobs / VL; greedy stays greedy speculation."""
from types import SimpleNamespace

import pytest

pytest.importorskip("jang_tools.dsv41.dspark")
from vmlx_engine.utils.dsv41_dspark_generator import DSV41DSparkGenerator  # noqa: E402
from jang_tools.dsv41.dspark import SpecSampler  # noqa: E402


def _gen():
    g = DSV41DSparkGenerator.__new__(DSV41DSparkGenerator)
    g._runner = object()
    g.sampler = None
    g._logprobs_required = lambda req: bool(getattr(req, "logprobs", False))
    return g


def _req(sampler, processors=None, logprobs=False):
    return SimpleNamespace(sampler=sampler, logits_processors=processors, logprobs=logprobs, pixel_values=None,
                           pixel_values_videos=None, dsv41_vl_images=None)


def _sampler(**tags):
    f = lambda x: x  # noqa: E731
    for k, v in tags.items():
        setattr(f, k, v)
    return f


def test_greedy_is_greedy_speculation():
    g = _gen()
    r = _req(_sampler(_vmlx_greedy=True, _vmlx_sampling={"temperature": 0.0}))
    assert g._use_dspark(r) and r._dspark_sampler is None


def test_sampled_request_gets_spec_sampler():
    g = _gen()
    r = _req(_sampler(_vmlx_sampling={"temperature": 1.0, "top_p": 0.95, "min_p": 0.0, "top_k": 0, "seed": 7}))
    assert g._use_dspark(r)
    s = r._dspark_sampler
    assert isinstance(s, SpecSampler) and s.temperature == 1.0 and s.top_p == 0.95 and s.seed == 7


@pytest.mark.parametrize("req_kw", [{"processors": [lambda t, l: l]}, {"logprobs": True}])
def test_processors_and_logprobs_take_ar(req_kw):
    g = _gen()
    r = _req(_sampler(_vmlx_sampling={"temperature": 1.0, "top_p": 1.0, "min_p": 0.0, "top_k": 0, "seed": None}), **req_kw)
    assert not g._use_dspark(r)


def test_untagged_sampler_takes_ar():
    g = _gen()
    assert not g._use_dspark(_req(_sampler()))


def test_env_off_keeps_ar(monkeypatch):
    monkeypatch.setenv("VMLX_DSV41_DSPARK_SAMPLED", "0")
    g = _gen()
    r = _req(_sampler(_vmlx_sampling={"temperature": 1.0, "top_p": 1.0, "min_p": 0.0, "top_k": 0, "seed": None}))
    assert not g._use_dspark(r)
