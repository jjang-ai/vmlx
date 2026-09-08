"""Compare coalesced prefill snapshots and continuation with split forwards."""

from copy import deepcopy
from types import SimpleNamespace

import mlx.core as mx
import pytest

from tests.test_qwen4_exp_runtime import _randomize, _tiny_args
from vmlx_engine.models.qwen4_exp.language import LanguageModel
from vmlx_engine.utils.qwen4_prefill_checkpoints import (
    coalesce_qwen4_prefill_checkpoints,
)


def assert_state_equal(left, right):
    for a, b in zip(left, right):
        if type(a).__name__ == "ArraysCache":
            for x, y in zip(a.cache, b.cache):
                if x is None or y is None:
                    assert x is y
                elif mx.issubdtype(x.dtype, mx.integer):
                    assert bool(mx.array_equal(x, y))
                else:
                    assert bool(mx.allclose(x, y, atol=1e-4, rtol=1e-4))


@pytest.mark.parametrize("length,boundaries", [(71, (32, 63)), (135, (64, 127))])
def test_snapshots_last_logits_and_restored_continuation(
    length, boundaries, monkeypatch
):
    monkeypatch.setenv("VMLX_QWEN4_COALESCE_PREFILL_CHECKPOINTS", "1")
    args = _tiny_args()
    # The stock evaluation Metal recurrence requires at least 32 key lanes.
    args.linear_key_head_dim = 128
    args.linear_value_head_dim = 128
    model = LanguageModel(args)
    _randomize(model)
    model.eval()
    ids = mx.array([[i % 900 for i in range(length)]])
    split = model.make_cache()
    expected = {}
    start = 0
    for end in (*boundaries, length):
        ref = model(ids[:, start:end], cache=split).logits
        mx.eval(ref)
        expected[end] = deepcopy(split)
        start = end
    cache = model.make_cache()
    snapshots = {}

    def capture(request, checkpoint_cache, tokens, boundary):
        snapshots[boundary] = deepcopy(checkpoint_cache)
        assert_state_equal(checkpoint_cache, expected[boundary])
        return True

    generator = SimpleNamespace(
        _model_type="qwen4_exp_text",
        _hybrid_kv_positions=[
            i for i, c in enumerate(cache) if type(c).__name__ != "ArraysCache"
        ],
        _maybe_capture_clean_ssm_boundary=capture,
    )
    result = coalesce_qwen4_prefill_checkpoints(
        generator,
        SimpleNamespace(_cached_tokens=0, request_id="test"),
        model,
        ids,
        cache,
        ids[0].tolist(),
        boundaries,
        lambda start, end: {"cache": cache},
    )
    assert result.logits.shape == (1, 1, model.args.vocab_size)
    assert bool(mx.allclose(result.logits, ref[:, -1:], atol=1e-4, rtol=1e-4))
    assert_state_equal(cache, expected[length])
    for c in cache:
        assert not hasattr(c, "prefill_checkpoint_states")
        assert not hasattr(c, "prefill_checkpoint_aux_states")
    next_ids = mx.array([[81, 82, 83]])
    for boundary, snapshot in snapshots.items():
        # Restore KV at the same logical boundary as the recurrent snapshot.
        restored = deepcopy(expected[boundary])
        for i, c in enumerate(snapshot):
            if type(c).__name__ == "ArraysCache":
                restored[i] = c
        got = model(next_ids, cache=restored).logits
        want = model(next_ids, cache=expected[boundary]).logits
        assert bool(mx.allclose(got, want, atol=1e-4, rtol=1e-4))


@pytest.mark.parametrize(
    "disabled,cached,family",
    [(True, 0, "qwen4_exp_text"), (False, 64, "qwen4_exp_text"), (False, 0, "other")],
)
def test_ineligible_request_never_dispatches(disabled, cached, family, monkeypatch):
    monkeypatch.setenv(
        "VMLX_QWEN4_COALESCE_PREFILL_CHECKPOINTS", "0" if disabled else "1"
    )

    def forbidden(*args, **kwargs):
        pytest.fail("ineligible request invoked model")

    assert (
        coalesce_qwen4_prefill_checkpoints(
            SimpleNamespace(_model_type=family),
            SimpleNamespace(_cached_tokens=cached),
            forbidden,
            mx.zeros((1, 71), dtype=mx.int32),
            [],
            [],
            (32, 63),
            forbidden,
        )
        is None
    )


@pytest.mark.parametrize("failure", ["model", "missing_state", "capture"])
def test_failure_cleans_temporary_state_without_retry(failure, monkeypatch):
    from mlx_lm.models.cache import ArraysCache

    monkeypatch.setenv("VMLX_QWEN4_COALESCE_PREFILL_CHECKPOINTS", "1")
    cache = [ArraysCache(size=2)]
    calls = []

    def forward(*args, **kwargs):
        calls.append("forward")
        cache[0].prefill_checkpoint_states = {
            32: (mx.zeros((1,)), mx.zeros((1,))),
            63: (mx.ones((1,)), mx.ones((1,))),
        }
        cache[0].prefill_checkpoint_aux_states = {}
        if failure == "model":
            raise RuntimeError("injected model failure")
        if failure == "missing_state":
            cache[0].prefill_checkpoint_states.pop(63)
        return object()

    def capture(*args):
        return failure != "capture"

    with pytest.raises(RuntimeError):
        coalesce_qwen4_prefill_checkpoints(
            SimpleNamespace(
                _model_type="qwen4_exp_text",
                _hybrid_kv_positions=[],
                _maybe_capture_clean_ssm_boundary=capture,
            ),
            SimpleNamespace(_cached_tokens=0, request_id="failure-test"),
            forward,
            mx.zeros((1, 71), dtype=mx.int32),
            cache,
            [0] * 71,
            (32, 63),
            lambda a, b: {},
        )
    assert calls == ["forward"]
    assert not hasattr(cache[0], "prefill_checkpoint_states")
    assert not hasattr(cache[0], "prefill_checkpoint_aux_states")


@pytest.mark.parametrize(
    "flag",
    [
        "GDN_BLOCKED_PREFILL",
        "VERIFY_SDPA",
        "PREFILL_DIRECT",
        "COALESCE_PREFILL_CHECKPOINTS",
    ],
)
@pytest.mark.parametrize("family", ["qwen4_exp", "qwen4_exp_text", "llama"])
def test_cache_identity_scopes_opt_in_math(flag, family, monkeypatch):
    from vmlx_engine.prefix_cache import compute_model_cache_key

    model = SimpleNamespace(args=SimpleNamespace(model_type=family))
    name = "VMLX_QWEN4_" + flag
    monkeypatch.setenv(name, "0")
    before = compute_model_cache_key(model)
    monkeypatch.setenv(name, "1")
    after = compute_model_cache_key(model)
    assert (before != after) == family.startswith("qwen4_exp")


@pytest.mark.parametrize("flag", ["GDN_BLOCKED_PREFILL", "VERIFY_SDPA"])
def test_cache_identity_normalizes_supported_true_spellings(flag, monkeypatch):
    from vmlx_engine.prefix_cache import compute_model_cache_key

    model = SimpleNamespace(args=SimpleNamespace(model_type="qwen4_exp_text"))
    name = "VMLX_QWEN4_" + flag
    monkeypatch.setenv(name, "1")
    key = compute_model_cache_key(model)
    for value in ("true", "yes", "ON"):
        monkeypatch.setenv(name, value)
        assert compute_model_cache_key(model) == key


def test_native_artifact_digest_changes_with_payload(tmp_path):
    from vmlx_engine.prefix_cache import _qwen4_native_artifact_digest

    p = tmp_path / "_ext.so"
    p.write_bytes(b"old")
    old = _qwen4_native_artifact_digest(((str(p), 3, 1),))
    p.write_bytes(b"new")
    new = _qwen4_native_artifact_digest(((str(p), 3, 2),))
    assert old != new


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "flag, attribute",
    [
        ("GDN_BLOCKED_PREFILL", "_QWEN4_GDN_MATH_ABI"),
        ("VERIFY_SDPA", "_QWEN4_VERIFY_MATH_ABI"),
        ("COALESCE_PREFILL_CHECKPOINTS", "_QWEN4_CHECKPOINT_MATH_ABI"),
    ],
)
def test_math_revision_invalidates_only_enabled_cache(
    enabled, flag, attribute, monkeypatch
):
    from vmlx_engine import prefix_cache

    model = SimpleNamespace(args=SimpleNamespace(model_type="qwen4_exp_text"))
    monkeypatch.setenv("VMLX_QWEN4_" + flag, str(int(enabled)))
    monkeypatch.setattr(prefix_cache, attribute, "previous-order")
    before = prefix_cache.compute_model_cache_key(model)
    monkeypatch.setattr(prefix_cache, attribute, "current-order")
    after = prefix_cache.compute_model_cache_key(model)
    assert (before != after) == enabled
