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


# --- samplers built by the REAL vmlx_engine.sampling.make_sampler -------------------------------------------------
from vmlx_engine.sampling import make_sampler  # noqa: E402
from vmlx_engine.scheduler import _tag_sampling_params  # noqa: E402


def _real(temp, top_p=0.0, top_k=0, min_p=0.0, seed=None, scheduler_tags=True):
    """A sampler as the scheduler builds it: make_sampler, plus the scheduler's tags when requested."""
    s = make_sampler(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k, seed=seed)
    if scheduler_tags:
        params = SimpleNamespace(temperature=temp, top_p=top_p, min_p=min_p, top_k=top_k)
        _tag_sampling_params(s, params, seed=seed)
    return s


@pytest.mark.parametrize("scheduler_tags", [True, False])
def test_real_greedy_sampler_is_greedy(scheduler_tags):
    g = _gen()
    r = _req(_real(0.0, scheduler_tags=scheduler_tags))
    assert g._is_greedy(r)
    assert g._use_dspark(r) and r._dspark_sampler is None


@pytest.mark.parametrize("kw", [
    {"temp": 0.7, "top_k": 40},            # compact top-k sampler: tagged accepts_logits, still sampled
    {"temp": 0.7, "top_p": 0.9},           # mlx_lm sampler
    {"temp": 0.7, "top_k": 40, "seed": 7},  # seeded compact top-k
    {"temp": 0.7, "top_p": 0.9, "seed": 7},  # seeded log-prob sampler
])
def test_real_sampled_samplers_take_sampled_speculation(kw):
    g = _gen()
    s = _real(**kw)
    r = _req(s)
    assert not g._is_greedy(r)
    assert g._use_dspark(r)
    spec = r._dspark_sampler
    assert isinstance(spec, SpecSampler)
    assert spec.temperature == pytest.approx(0.7)
    assert spec.top_k == kw.get("top_k", 0)
    assert spec.top_p == pytest.approx(kw.get("top_p", 0.0))
    if "seed" in kw:
        assert spec.seed == kw["seed"]


def test_accepts_logits_tag_alone_is_not_greedy():
    """The scheduler/patch layers may set _vmlx_accepts_logits on a sampled sampler; it must stay sampled."""
    g = _gen()
    s = _real(0.7, top_k=40)
    s._vmlx_accepts_logits = True
    assert not g._is_greedy(_req(s))


def test_untagged_real_sampled_sampler_takes_ar():
    """Without the scheduler's _vmlx_sampling record the distribution is unknown: AR, never greedy."""
    g = _gen()
    r = _req(_real(0.7, top_p=0.9, scheduler_tags=False))
    assert not g._is_greedy(r)
    assert not g._use_dspark(r)
