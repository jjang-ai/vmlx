"""Scheduling qualification: exact outputs and native state, not a speed claim."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_map_with_path

from tests.test_qwen4_exp_runtime import _randomize, _tiny_args
from vmlx_engine.models.qwen4_exp import language


def _model(monkeypatch, enabled=True):
    if enabled is None:
        monkeypatch.delenv("VMLX_QWEN4_EAGER_DISPATCH", raising=False)
    else:
        monkeypatch.setenv("VMLX_QWEN4_EAGER_DISPATCH", "1" if enabled else "0")
    args = _tiny_args()
    args.linear_key_head_dim = 32
    args.linear_value_head_dim = 32
    model = language.LanguageModel(args)
    _randomize(model)
    model.eval()
    mx.eval(model.parameters())
    return model


def _snapshot(value):
    if isinstance(value, mx.array):
        mx.eval(value)
        raw = value.view(mx.uint16) if value.dtype == mx.bfloat16 else value
        return str(value.dtype), np.asarray(raw).copy()
    if isinstance(value, language._QSAPooledFrontier):
        return {key: _snapshot(getattr(value, key)) for key in value.__slots__}
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_snapshot(item) for item in value)
    return value


def _cache_snapshot(cache):
    # Include QSA's raw lane and derived pooled lane, plus GDN recurrent/conv
    # and PLE context/conv. Logical offsets must match as well as array bytes.
    return tuple(
        _snapshot({
            "state": item.state,
            "offset": getattr(item, "offset", None),
            "idx_offset": getattr(item, "_idx_offset", None),
            "derived": getattr(item, "derived", None),
        })
        for item in cache
    )


def _assert_exact(left, right):
    assert type(left) is type(right)
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_exact(left[key], right[key])
    elif isinstance(left, tuple):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_exact(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16, "mixed_quant"])
def test_eager_dispatch_exact_connected_cache_on_caller_stream(monkeypatch, dtype):
    model = _model(monkeypatch)
    mixed = dtype == "mixed_quant"
    target_dtype = mx.float16 if mixed else dtype
    model.update(tree_map_with_path(
        lambda path, value: value.astype(target_dtype)
        if mx.issubdtype(value.dtype, mx.floating) and model.cast_predicate(path)
        else value,
        model.parameters(),
    ))
    if mixed:
        def quantize(path, module):
            if not hasattr(module, "to_quantized") or not model.quant_predicate(path, module):
                return False
            width = module.weight.shape[-1]
            if width % 32:
                return False
            index = int(path.split(".")[2]) if path.startswith("model.layers.") else 0
            group_size = 64 if index % 2 and width % 64 == 0 else 32
            return {"group_size": group_size, "bits": (2, 4, 6, 8)[index % 4]}
        nn.quantize(model, class_predicate=quantize)
    mx.eval(model.parameters())
    stream = mx.new_stream(mx.gpu)
    observed_streams = []
    submit = mx.async_eval

    def record(value):
        observed_streams.append(mx.default_stream(mx.gpu))
        submit(value)

    monkeypatch.setattr(mx, "async_eval", record)
    results = []
    with mx.stream(stream):
        for enabled in (False, True):
            model.model._eager_dispatch = enabled
            cache = model.make_cache()
            arm = []
            # New prefix, decode, short history extension, decode again.
            for ids in ([11, 17, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67],
                        [71], [73, 79, 83], [89]):
                logits = model(mx.array([ids]), cache=cache).logits
                arm.append((_snapshot(logits), _cache_snapshot(cache)))
            results.append(tuple(arm))
    _assert_exact(*results)
    assert len(observed_streams) == 4 * len(model.layers)
    assert all(item == stream for item in observed_streams)


@pytest.mark.parametrize("accepted_drafts", [0, 1, 2, 3])
def test_eager_dispatch_exact_verify_rollback_and_continuation(monkeypatch, accepted_drafts):
    from vmlx_engine.mllm_batch_generator import _native_mtp_rollback_to_confirmed

    model = _model(monkeypatch)
    results = []
    with mx.stream(mx.new_stream(mx.gpu)):
        for enabled in (False, True):
            model.model._eager_dispatch = enabled
            cache = model.make_cache()
            model(mx.array([[11, 17, 23, 29, 31, 37, 41, 43, 47]]), cache=cache)
            verified = model(mx.array([[53, 59, 61, 67]]), cache=cache, n_confirmed=1)
            before = (_snapshot(verified.logits), _cache_snapshot(cache))
            if accepted_drafts < 3:
                assert _native_mtp_rollback_to_confirmed(
                    cache, reject_tokens=3 - accepted_drafts,
                    accepted_drafts=accepted_drafts,
                )
            after = _cache_snapshot(cache)
            continuation = model(mx.array([[71]]), cache=cache).logits
            results.append((before, after, _snapshot(continuation), _cache_snapshot(cache)))
    _assert_exact(*results)


@pytest.mark.parametrize("enabled,expected", [(None, True), (True, True), (False, False)])
def test_eager_dispatch_default_on_with_explicit_opt_out(monkeypatch, enabled, expected):
    model = _model(monkeypatch, enabled=enabled)
    assert model.model._eager_dispatch is expected
    # The policy is captured at construction, not changed mid-generation.
    monkeypatch.setenv("VMLX_QWEN4_EAGER_DISPATCH", "0" if expected else "1")
    assert model.model._eager_dispatch is expected


def test_eager_dispatch_explicit_off_and_flattened_row_bound(monkeypatch):
    model = _model(monkeypatch, enabled=False)
    assert model.model._eager_dispatch is False
    calls = []
    submit = mx.async_eval

    def record(value):
        calls.append(value.shape)
        submit(value)

    monkeypatch.setattr(mx, "async_eval", record)
    mx.eval(model(mx.array([[11]]), cache=model.make_cache()).logits)
    assert calls == []
    model.model._eager_dispatch = True
    # Two rows per batch times 33 tokens exceeds the 64-row cap even though
    # each sequence is shorter; 2x32 is the admitted boundary.
    for length, expected in ((33, 0), (32, len(model.layers))):
        ids = mx.array([[i + 11 for i in range(length)]] * 2)
        mx.eval(model(ids, cache=model.make_cache()).logits)
        assert len(calls) == expected


def test_eager_dispatch_preserves_checkpoint_and_diagnostic_paths(monkeypatch):
    model = _model(monkeypatch)
    calls = []
    monkeypatch.setattr(mx, "async_eval", lambda value: calls.append(value))
    ids = mx.array([[11, 17, 23, 29, 31, 37, 41, 43, 47]])
    output = model(ids, cache=model.make_cache(), prefill_checkpoint_steps=(4, 8))
    mx.eval(output.logits)
    assert calls == []
    monkeypatch.setattr(language, "_layer_profile_enabled", lambda inputs: True)
    mx.eval(model(ids[:, :1], cache=model.make_cache()).logits)
    assert calls == []
    monkeypatch.setattr(language, "_layer_profile_enabled", lambda inputs: False)
    monkeypatch.setattr(language, "_layer_fingerprint_enabled", lambda inputs: True)
    mx.eval(model(ids[:, :1], cache=model.make_cache()).logits)
    assert calls == []


def test_eager_dispatch_submits_before_second_layer_ple_read(monkeypatch):
    model = _model(monkeypatch)
    events = []
    submit = mx.async_eval
    gather = language.ShardedNGramEmbedding.__call__

    def record_submit(value):
        events.append("submit")
        submit(value)

    def record_gather(self, *args, **kwargs):
        events.append("ple_read")
        return gather(self, *args, **kwargs)

    monkeypatch.setattr(mx, "async_eval", record_submit)
    monkeypatch.setattr(language.ShardedNGramEmbedding, "__call__", record_gather)
    mx.eval(model(mx.array([[11]]), cache=model.make_cache()).logits)
    assert [index for index, layer in enumerate(model.layers) if layer.ple] == [1]
    assert events[:2] == ["submit", "ple_read"]
    assert events.count("submit") == len(model.layers)
