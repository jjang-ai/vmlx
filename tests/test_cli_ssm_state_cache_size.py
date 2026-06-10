from types import SimpleNamespace

from vmlx_engine.cli import _effective_ssm_state_cache_size


def test_effective_ssm_state_cache_size_keeps_default_conservative():
    args = SimpleNamespace(ssm_state_cache_size=8, ssm_state_cache_mb=512)

    assert _effective_ssm_state_cache_size(args) == 8


def test_effective_ssm_state_cache_size_scales_large_budget():
    args = SimpleNamespace(ssm_state_cache_size=8, ssm_state_cache_mb=8192)

    assert _effective_ssm_state_cache_size(args) == 64


def test_effective_ssm_state_cache_size_respects_explicit_larger_entry_count():
    args = SimpleNamespace(ssm_state_cache_size=96, ssm_state_cache_mb=8192)

    assert _effective_ssm_state_cache_size(args) == 96
