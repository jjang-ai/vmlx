"""DSV4.1 rides the DSV4 composite cache store, but V4.0-only cache policies do not apply to it.

DeepseekV41Cache matches the DSV4 class-name set, so every ``_uses_dsv4_cache`` gate used to fire for V4.1 too. The
ones whose V4.0 reason does not hold for V4.1 are split on ``_uses_dsv41_cache``:
  - the 256-token snapshot store threshold (V4.1's generator always captures the exact N-1 snapshot, so short prompts
    were never stored: "cache_key_tokens=22 < 256");
  - the forced 256-token paged block size (V4.0 native delta records; V4.1 stores plain states);
  - the DSV4_* block-namespace scope (V4.0 runtime env knobs);
  - the DSV4_MAX_PREFILL_TOKENS operator guard (V4.0 SWA+CSA/HCA Python path);
  - the idle shadow re-key (V4.0 token ids and a V4.0 delta-capturing prefill).
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from vmlx_engine import scheduler as scheduler_mod
from vmlx_engine.scheduler import Scheduler


def _sched(v41: bool):
    s = Scheduler.__new__(Scheduler)
    s._uses_dsv4_cache = True
    s._uses_dsv41_cache = v41
    return s


def _req(n_prompt: int):
    return SimpleNamespace(prompt_token_ids=list(range(n_prompt)), _gen_prompt_len=0)


@pytest.mark.parametrize("n_prompt", [3, 23, 200])
def test_short_prompt_snapshot_is_stored_for_v41_only(monkeypatch, n_prompt):
    monkeypatch.delenv("DSV4_PROMPT_SNAPSHOT_MIN_TOKENS", raising=False)
    monkeypatch.delenv("VMLINUX_DSV4_PROMPT_SNAPSHOT_MIN_TOKENS", raising=False)
    assert _sched(v41=False)._dsv4_snapshot_store_below_threshold(_req(n_prompt)) is True
    assert _sched(v41=True)._dsv4_snapshot_store_below_threshold(_req(n_prompt)) is False


def test_shadow_rekey_never_queues_for_v41():
    s = _sched(v41=True)
    s.block_aware_cache = object()
    s.config = SimpleNamespace(enable_prefix_cache=True)
    s._pick_cache_type_for_request = lambda _r: "assistant"
    registered = []
    s._ensure_dsv4_shadow_rekey_idle_task = lambda: registered.append(1)
    req = SimpleNamespace(prompt_token_ids=[10] * 300 + [128821], output_token_ids=[5, 128822, 6, 1],
                          _vmlx_tools_present=False, _cache_extra_keys=None)
    s._queue_dsv4_shadow_rekey("r1", req, "stop")
    assert getattr(s, "_dsv4_shadow_rekey_queue", []) == [] and registered == []


class _Stop(Exception):
    pass


def _add_request(sched, n_tokens):
    """Run add_request up to just past the V4.0 prefill guard (the next statement reads _bypass_prefix_cache)."""
    req = SimpleNamespace(request_id="r", prompt_token_ids=list(range(n_tokens)), prompt=None,
                          num_prompt_tokens=n_tokens, sampling_params=None, _max_prompt_tokens=0)

    class _Probe(SimpleNamespace):
        def __getattr__(self, name):
            raise _Stop(name)

    sched.requests = {}
    sched._pld_spec_enabled = False
    sched._generation_prompt_strip_active = lambda: True
    return sched.add_request(_Probe(**vars(req)))


def test_v40_prefill_guard_does_not_apply_to_v41(monkeypatch):
    monkeypatch.setenv("DSV4_MAX_PREFILL_TOKENS", "4")
    with pytest.raises(ValueError, match="long-prefill guard"):
        _add_request(_sched(v41=False), 10)
    with pytest.raises(_Stop):  # past the guard: the stub request stops at the next attribute read
        _add_request(_sched(v41=True), 10)


def test_block_size_and_namespace_split_in_source():
    src = inspect.getsource(Scheduler.__init__)
    assert 'not getattr(self, "_uses_dsv41_cache", False) and self.config.use_paged_cache' in src
    assert ":dsv41_cache_schema=deepseek_v41_fmt2" in src
    v41 = src.index(":dsv41_cache_schema=deepseek_v41_fmt2")
    v40 = src.index(":dsv4_cache_schema=deepseek_v4_v10_delta")
    assert v41 < v40  # the V4.1 branch is checked first (V4.1 also matches _uses_dsv4_cache)
    assert scheduler_mod  # module import keeps the inspected source current


def test_dsv41_model_is_not_hybrid():
    """DeepseekV41Cache is answered by the DSV4 class-name check (the old separate entry was unreachable)."""
    mlx_model = pytest.importorskip("jang_tools.dsv4_1.mlx_model")
    seq = mlx_model._SeqState()
    model = SimpleNamespace(make_cache=lambda: [mlx_model.DeepseekV41Cache(seq, i, r) for i, r in enumerate((0, 2))])
    assert Scheduler._is_hybrid_model(model) is False
    assert Scheduler._model_uses_dsv4_cache(model) is True
