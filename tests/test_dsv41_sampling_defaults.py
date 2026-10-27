"""A V4.1 request with no sampling params resolves to the bundle's generation_config (do_sample true, temperature 1.0,
top_p 1.0) and DSpark then takes the sampled path (not greedy). Synthetic bundle directory, no weights."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import vmlx_engine.server as server


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}))
    (tmp_path / "generation_config.json").write_text(json.dumps(
        {"bos_token_id": 0, "eos_token_id": 1, "do_sample": True, "temperature": 1.0, "top_p": 1.0}))
    monkeypatch.setattr(server, "_model_path", str(tmp_path))
    for name in ("_default_temperature", "_default_top_p", "_default_top_k", "_default_min_p",
                 "_default_repetition_penalty"):
        monkeypatch.setattr(server, name, None, raising=False)
    monkeypatch.setattr(server, "_native_mtp_sampling_policy", "compatible-only", raising=False)
    monkeypatch.setattr(server, "_generation_defaults_cache", {})
    monkeypatch.setattr(server, "_jang_sampling_defaults_cache", {})
    return str(tmp_path)


def test_omitted_sampling_resolves_to_generation_config(bundle):
    assert server._resolve_temperature(None, bundle) == 1.0
    assert server._resolve_top_p(None, bundle) == 1.0
    kw = {"temperature": server._resolve_temperature(None, bundle)}
    server._set_resolved_top_k(kw, None, bundle)
    server._set_resolved_min_p(kw, None, bundle)
    assert "top_k" not in kw and "min_p" not in kw
    assert server._resolve_repetition_penalty(None, bundle) is None  # no logits processor -> DSpark eligible


def test_bundle_defaults_take_sampled_dspark(bundle):
    pytest.importorskip("jang_tools.dsv41.dspark")
    from jang_tools.dsv41.dspark import SpecSampler

    from vmlx_engine.sampling import make_sampler
    from vmlx_engine.scheduler import _tag_sampling_params
    from vmlx_engine.utils.dsv41_dspark_generator import DSV41DSparkGenerator

    params = SimpleNamespace(temperature=server._resolve_temperature(None, bundle),
                             top_p=server._resolve_top_p(None, bundle), min_p=0.0, top_k=0)
    sampler = make_sampler(temp=params.temperature, top_p=params.top_p, min_p=0.0, top_k=0)
    _tag_sampling_params(sampler, params)
    g = DSV41DSparkGenerator.__new__(DSV41DSparkGenerator)
    g._runner, g.sampler = object(), None
    g._logprobs_required = lambda req: False
    req = SimpleNamespace(sampler=sampler, logits_processors=None, pixel_values=None, pixel_values_videos=None,
                          dsv41_vl_images=None)
    assert g._use_dspark(req)
    assert isinstance(req._dspark_sampler, SpecSampler) and req._dspark_sampler.temperature == 1.0
