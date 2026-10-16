"""Nemotron-3.5 Bolt / Super-VL media prefix reuse is admitted by the wrapper's runtime flag, never by the shared
nemotron_h label, and the wrapper exposes the config_omni placeholder id the generator needs to find image runs."""
import json

import mlx.nn as nn
import pytest

from vmlx_engine.mllm_batch_generator import _mllm_media_prefix_cache_family_enabled


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("VMLINUX_MLLM_MEDIA_PREFIX_CACHE", raising=False)
    monkeypatch.delenv("VMLINUX_MLLM_MEDIA_PREFIX_CACHE_UNSAFE_ACK", raising=False)


def test_nemotron_h_label_alone_is_not_admitted():
    # Nemotron-Nano-Omni reports the same model_type and runs a different media path
    assert _mllm_media_prefix_cache_family_enabled("nemotron_h") is False
    assert _mllm_media_prefix_cache_family_enabled("nemotron_h_v2") is False


def test_bolt_runtime_flag_admits_nemotron_h():
    assert _mllm_media_prefix_cache_family_enabled("nemotron_h", nemotron_bolt_vl_runtime=True) is True
    assert _mllm_media_prefix_cache_family_enabled("nemotron_h_v2", nemotron_bolt_vl_runtime=True) is True


def test_bolt_runtime_flag_does_not_leak_to_other_types():
    assert _mllm_media_prefix_cache_family_enabled("llama", nemotron_bolt_vl_runtime=True) is False


def test_kill_switch_wins(monkeypatch):
    monkeypatch.setenv("VMLINUX_MLLM_MEDIA_PREFIX_CACHE", "0")
    assert _mllm_media_prefix_cache_family_enabled("nemotron_h", nemotron_bolt_vl_runtime=True) is False


class _Text(nn.Module):
    layers = []

    def make_cache(self):
        return []


def test_wrapper_exposes_placeholder_id_and_runtime_flag(tmp_path):
    from vmlx_engine.models.nemotron_bolt_vl import Model

    cfg = {"model_type": "nemotron_h", "hidden_size": 8}
    (tmp_path / "config_omni.json").write_text(json.dumps({"img_context_token_id": 18}))
    m = Model(_Text(), tmp_path, cfg)
    assert m.config.image_token_id == 18 and m.image_token_id == 18
    assert m.config.model_type == "nemotron_h"
    assert m._nemotron_bolt_vl_runtime is True
    assert "image_token_id" not in cfg                     # the caller's config dict is not mutated


def _delta_fixture(bolt):
    from types import SimpleNamespace

    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    kv = KVCache()
    kv.update_and_fetch(mx.zeros((1, 2, 10, 4)), mx.zeros((1, 2, 10, 4)))
    calls = {}

    class _State:
        state = [mx.zeros((1, 3))]

    def derive(tokens, base, base_n, **kw):
        calls.update(tokens=list(tokens), base_n=base_n, **kw)
        return [_State(), base[0]]

    fake = SimpleNamespace(
        _is_hybrid=True, _hybrid_kv_positions=[1], _model_type="nemotron_h",
        model=SimpleNamespace(_nemotron_bolt_vl_runtime=bolt),
        _tokens_contain_media_placeholders=lambda toks: 18 in list(toks),
        block_aware_cache=SimpleNamespace(reconstruct_cache=lambda bt: [kv]),
        _prefill_for_clean_path_dependent_cache=derive,
    )
    return fake, calls


def test_bolt_delta_advances_companion_across_text_gap_after_image():
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

    fake, calls = _delta_fixture(bolt=True)
    tokens = [1, 2, 18, 18, 18, 3, 4, 5, 6, 7]                 # image run inside the prefix, text-only gap 7..10
    states = MLLMBatchGenerator._derive_hybrid_companion_delta(fake, object(), tokens, 10, 7, object())
    assert states is not None and len(states) == 1
    assert calls["base_n"] == 7 and calls["tokens"] == tokens and calls["delta_position_ids"] is None


def test_non_bolt_runtime_still_declines_media_prefix_without_positions():
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

    fake, calls = _delta_fixture(bolt=False)
    tokens = [1, 2, 18, 18, 18, 3, 4, 5, 6, 7]
    assert MLLMBatchGenerator._derive_hybrid_companion_delta(fake, object(), tokens, 10, 7, object()) is None
    assert not calls


def test_bolt_delta_refuses_gap_that_crosses_an_image():
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

    fake, calls = _delta_fixture(bolt=True)
    tokens = [1, 2, 3, 4, 5, 6, 18, 18, 18, 7]                 # placeholders inside the gap 5..10
    assert MLLMBatchGenerator._derive_hybrid_companion_delta(fake, object(), tokens, 10, 5, object()) is None
    assert not calls
