from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vmlx_engine.utils.hybrid_prefill_capture import prompt_with_hybrid_capture
from vmlx_engine.scheduler import Scheduler


@pytest.mark.parametrize("target,prior,incoming,expected", [
    (3, [], [1,2,3,4,5], [[1,2,3], [4,5]]),
    (3, [1,2], [3], [[3]]),
    (2, [1,2], [3,4], [[3,4]]),
    (8, [], [1,2], [[1,2]]),
])
def test_exact_boundary_before_suffix(target, prior, incoming, expected):
    batch = SimpleNamespace(uids=[7], tokens=[prior.copy()])
    batch._vmlx_hybrid_boundary_target = lambda uid, seen: target
    snapshots = []
    batch.extract_cache = lambda idx: tuple(batch.tokens[idx])
    batch._vmlx_hybrid_boundary_store = lambda uid, seen, state: snapshots.append((uid, seen, state))
    calls = []
    def prompt(owner, rows):
        calls.append(rows[0])
        owner.tokens[0].extend(rows[0])
    prompt_with_hybrid_capture(batch, [incoming], prompt)
    assert calls == expected
    assert batch.tokens[0] == prior + incoming
    if target <= len(prior) + len(incoming):
        assert snapshots == [(7, (prior + incoming)[:target], tuple((prior + incoming)[:target]))]
    else:
        assert snapshots == []


def test_multirow_is_not_misrepresented_as_singleton():
    batch = SimpleNamespace(uids=[1,2], _vmlx_hybrid_boundary_target=Mock(),
                            _vmlx_hybrid_boundary_store=Mock())
    prompt = Mock()
    prompt_with_hybrid_capture(batch, [[1],[2]], prompt)
    prompt.assert_called_once_with(batch, [[1],[2]])
    batch._vmlx_hybrid_boundary_target.assert_not_called()


def test_capture_failure_keeps_suffix_but_forward_failure_is_not_retried():
    batch = SimpleNamespace(uids=[1], tokens=[[]], extract_cache=Mock(side_effect=ValueError("capture")),
                            _vmlx_hybrid_boundary_target=lambda *_: 1,
                            _vmlx_hybrid_boundary_store=Mock())
    calls = []
    def prompt(owner, rows):
        calls.append(rows[0]); owner.tokens[0].extend(rows[0])
    prompt_with_hybrid_capture(batch, [[1,2]], prompt)
    assert calls == [[1],[2]]
    batch.tokens = [[]]
    fail = Mock(side_effect=RuntimeError("forward advanced then failed"))
    with pytest.raises(RuntimeError):
        prompt_with_hybrid_capture(batch, [[1,2]], fail)
    fail.assert_called_once()


def owner_fixture(base=0):
    scheduler = Scheduler.__new__(Scheduler)
    request = SimpleNamespace(request_id="r", prompt_token_ids=list(range(10)),
                              _gen_prompt_len=2, cached_tokens=base)
    scheduler.uid_to_request_id = {7:"r"}
    scheduler.running = {"r":request}
    scheduler._hybrid_num_layers = 2
    scheduler._hybrid_kv_positions = [0]
    scheduler._ssm_state_cache = SimpleNamespace(store=Mock())
    return scheduler, request


@pytest.mark.parametrize("base", [0,3,7])
def test_scheduler_target_uses_key_minus_one_and_exact_admitted_suffix(base):
    s,r = owner_fixture(base)
    assert s._hybrid_prefill_boundary_target(7, []) == 7-base
    assert s._store_hybrid_prefill_boundary(7, list(range(base,7)), ["kv","native"])
    s._ssm_state_cache.store.assert_called_once_with(list(range(7)), 7, ["native"])
    assert s._hybrid_prefill_boundary_target(7, list(range(base,7))) is None


@pytest.mark.parametrize("failure", ["cancelled", "media", "wrong_tokens", "past_boundary", "bypass"])
def test_unowned_or_wrong_boundary_does_not_publish(failure):
    s,r = owner_fixture()
    seen = []
    if failure == "cancelled": s.running.clear()
    if failure == "media": r.images = ["image"]
    if failure == "wrong_tokens": seen = [99]
    if failure == "past_boundary": r.cached_tokens = 8
    if failure == "bypass": r._bypass_prefix_cache = True
    assert s._hybrid_prefill_boundary_target(7, seen) is None
    assert not s._store_hybrid_prefill_boundary(7, seen, ["kv","native"])
    s._ssm_state_cache.store.assert_not_called()


@pytest.mark.parametrize("bits", [4,6,8])
def test_real_qwen_native_snapshot_survives_suffix_and_restores(bits):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import qwen3_5
    from mlx_lm.models.cache import ArraysCache
    from vmlx_engine.patches.mlx_lm_mtp import qwen35_model
    from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache

    qwen35_model.apply()
    mx.random.seed(617)
    layer = qwen3_5.GatedDeltaNet(qwen3_5.TextModelArgs(
        hidden_size=64, linear_num_value_heads=2, linear_num_key_heads=2,
        linear_key_head_dim=32, linear_value_head_dim=32))
    nn.quantize(layer, group_size=64, bits=bits,
                class_predicate=lambda p,m: isinstance(m, nn.Linear))
    layer.eval()
    vectors = mx.random.normal((1,12,64))
    live, reference = ArraysCache(2), ArraysCache(2)
    batch = SimpleNamespace(uids=[7], tokens=[[]], extract_cache=lambda _: [live],
                            _vmlx_hybrid_boundary_target=lambda *_: 7)
    captured = []
    clone_owner = SSMCompanionCache.__new__(SSMCompanionCache)
    def capture(uid, seen, states):
        captured.extend(clone_owner._clone_states(states, key_hint="test"))
    batch._vmlx_hybrid_boundary_store = capture
    def prompt(owner, rows):
        start = len(owner.tokens[0])
        mx.eval(layer(vectors[:,start:start+len(rows[0])], cache=live))
        owner.tokens[0].extend(rows[0])
    prompt_with_hybrid_capture(batch, [list(range(10))], prompt)
    mx.eval(layer(vectors[:,:7], cache=reference))
    for expected, got in zip(reference.state, captured[0].state):
        assert mx.allclose(expected, got, atol=2e-5, rtol=2e-5).item()
    # Live has already consumed the suffix; the detached boundary must not.
    expected = layer(vectors[:,7:12], cache=reference)
    restored = layer(vectors[:,7:12], cache=captured[0])
    assert mx.allclose(expected, restored, atol=2e-5, rtol=2e-5).item()
