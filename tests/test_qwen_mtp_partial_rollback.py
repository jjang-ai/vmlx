"""Real Qwen recurrent-layer rollback across every supported draft boundary."""
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")


@pytest.mark.parametrize("drafts,accepted", [(d, a) for d in (1, 2, 3) for a in range(d)])
@pytest.mark.parametrize("bits", [4, 6, 8])
def test_real_gdn_rejected_suffix_restores_accepted_state(drafts, accepted, bits):
    import mlx.nn as nn
    from mlx_lm.models import qwen3_5
    from mlx_lm.models.cache import ArraysCache
    from vmlx_engine.text_mtp import qwen_family as qwen35_model
    from vmlx_engine.text_mtp.lane import rewind_backbone

    qwen35_model.install()
    mx.random.seed(617)
    args = qwen3_5.TextModelArgs(hidden_size=64, linear_num_value_heads=2,
        linear_num_key_heads=2, linear_key_head_dim=32, linear_value_head_dim=32)
    layer = qwen3_5.GatedDeltaNet(args)
    if bits is not None:
        nn.quantize(layer, group_size=64, bits=bits,
                    class_predicate=lambda p, m: isinstance(m, nn.Linear))
    layer.eval()
    original, speculative = ArraysCache(2), ArraysCache(2)
    warm = mx.random.normal((1, 5, 64))
    mx.eval(layer(warm, cache=original))
    speculative.state = list(original.state)
    inputs = mx.random.normal((1, drafts + 1, 64))
    mx.eval(layer(inputs[:, :1 + accepted], cache=original, n_confirmed=1))
    mx.eval(layer(inputs, cache=speculative, n_confirmed=1))
    assert speculative.supports_partial_rollback
    assert rewind_backbone([speculative], drafts - accepted)
    for want, got in zip(original.state, speculative.state):
        assert mx.allclose(want, got, atol=2e-5, rtol=2e-5).item()
    assert speculative._vmlx_partial_restore is None
    assert speculative.rollback_state is None
    tail = mx.random.normal((1, 1, 64))
    want, got = layer(tail, cache=original), layer(tail, cache=speculative)
    assert mx.allclose(want, got, atol=2e-5, rtol=2e-5).item()


@pytest.mark.parametrize("drafts", [1, 2, 3])
def test_float32_first_rejection_preserves_confirmed_snapshot(drafts):
    # Float32 unquantized projection has a shape-dependent precision change
    # on M5 (observed qkv max error0.001346 before rollback). Do not relax the
    # independent quantized-state tolerance above or call this AR identity.
    # At first rejection, the exact existing confirmed snapshot is the oracle.
    from mlx_lm.models import qwen3_5
    from mlx_lm.models.cache import ArraysCache
    from vmlx_engine.text_mtp import qwen_family as qwen35_model
    qwen35_model.install()
    mx.random.seed(617)
    layer = qwen3_5.GatedDeltaNet(qwen3_5.TextModelArgs(
        hidden_size=64, linear_num_value_heads=2, linear_num_key_heads=2,
        linear_key_head_dim=32, linear_value_head_dim=32))
    layer.eval()
    cache = ArraysCache(2)
    mx.eval(layer(mx.random.normal((1, 5, 64)), cache=cache))
    mx.eval(layer(mx.random.normal((1, drafts + 1, 64)), cache=cache, n_confirmed=1))
    conv, ssm = cache.rollback_state
    assert cache.rollback_speculative(drafts)
    assert mx.array_equal(cache[0], conv).item()
    assert mx.array_equal(cache[1], ssm).item()


def test_invalid_rollback_is_rejected_before_any_layer_mutation():
    from mlx_lm.models.cache import ArraysCache
    from vmlx_engine.text_mtp.hybrid_state import arm_partial_rollback
    from vmlx_engine.text_mtp.lane import rewind_backbone
    events = []
    kv = SimpleNamespace(is_trimmable=lambda: True, trim=lambda n: events.append(n))
    empty = ArraysCache(2)
    arm_partial_rollback(empty)
    assert not rewind_backbone([kv, empty], 1)
    assert not events


def test_commit_releases_capture_and_restores_sequence_lengths():
    from mlx_lm.models.cache import ArraysCache
    from vmlx_engine.text_mtp.hybrid_state import (
        arm_partial_rollback, remember_partial_rollback)
    from vmlx_engine.text_mtp.lane import commit_backbone
    cache = ArraysCache(2)
    cache.prepare([8])
    arm_partial_rollback(cache)
    remember_partial_rollback(cache, 3, lambda k: (mx.array([k]), mx.array([k])))
    cache.advance(4)
    assert cache.rollback_speculative(2)
    assert cache.lengths.item() == 6
    remember_partial_rollback(cache, 3, lambda k: (None, None))
    commit_backbone([cache])
    assert cache._vmlx_partial_restore is None
    assert not cache.can_rollback_speculative(1)


def test_nonfallback_telemetry_keeps_seed_measurement():
    from vmlx_engine.text_mtp.lane_stats import LaneStats; from vmlx_engine.text_mtp.lane_stats import build_record
    stats = LaneStats(seed_ar_step_ms=123.456, depth=3)
    payload = build_record("slow-cached-turn", stats, "stop")
    assert payload["seed_ar_step_ms"] == 123.456
    assert payload["fallback_reason"] is None
    assert build_record("unmeasured", LaneStats(), "stop")["seed_ar_step_ms"] is None
