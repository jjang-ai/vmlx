"""Dense-KV overlap must preserve concrete-stream and restored-state safety."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import KVCache, RotatingKVCache, load_prompt_cache, save_prompt_cache

from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator


class DenseModel:
    def make_cache(self):
        return [KVCache()]

    def __call__(self, tokens, cache):
        values = tokens[:, None, :, None].astype(mx.float32)
        keys, _ = cache[0].update_and_fetch(values, values)
        # Depend on actual native cached contents, not a cache-blind constant.
        winner = (mx.sum(keys).astype(mx.int32) % 7).reshape(1, 1, 1)
        return mx.broadcast_to(
            -mx.abs(mx.arange(8)[None, None, :] - winner),
            (tokens.shape[0], tokens.shape[1], 8),
        ).astype(mx.float32)


def collect(generator):
    results = []
    while True:
        first, rest = generator.next()
        if not first and not rest:
            return results
        results.extend(first + rest)


def test_dense_steady_decode_submits_without_waiting(monkeypatch):
    generator = SingleBatchGenerator(DenseModel(), max_tokens=6)
    calls = []
    original = mx.async_eval

    def recording(*values):
        assert mx.default_stream(generator._device) == generator._stream
        calls.append(len(values))
        return original(*values)

    monkeypatch.setattr(mx, "async_eval", recording)
    generator.insert([[1, 2, 3]])
    rows = collect(generator)
    assert len(rows) == 6
    assert rows[-1].finish_reason == "length"
    assert calls, "steady dense-KV decode must overlap next-token submission"


def test_only_initialized_exact_dense_cache_is_eligible():
    generator = SingleBatchGenerator(DenseModel())
    request = SimpleNamespace(cache=[KVCache()], output_tokens=[])
    assert not generator._can_overlap_decode(request)
    request.output_tokens = [1]
    assert generator._can_overlap_decode(request)
    class CustomCache(KVCache):
        pass
    for caches in ([], [None], [CustomCache()], [RotatingKVCache(max_size=8)], [KVCache(), object()]):
        request.cache = caches
        assert not generator._can_overlap_decode(request)


def test_submitted_current_token_is_not_rebound_behind_next_decode(monkeypatch):
    generator = SingleBatchGenerator(DenseModel())
    token = mx.array([3])
    def reject_rebind(value):
        raise AssertionError("already submitted current token must not queue a new operation")
    monkeypatch.setattr(generator, "_rehome_on_stream", reject_rebind)
    assert generator._materialize_submitted_on_stream(token)[0] is token


@pytest.mark.parametrize("logprobs", [False, True])
def test_disk_refault_cross_thread_matches_synchronous_decode(tmp_path, monkeypatch, logprobs):
    model = DenseModel()
    cache = model.make_cache()
    def prime():
        result = model(mx.array([[1, 2, 3]]), cache)
        mx.eval(result, cache[0].state)
        save_prompt_cache(str(tmp_path / "prefix.safetensors"), cache)
    with ThreadPoolExecutor(max_workers=1) as owner:
        owner.submit(prime).result()

    def run(overlap):
        # Exercise token-context processor submission alongside sampled values.
        def penalty(context, logits):
            return logits.at[..., 3].add(-2.0)
        generator = SingleBatchGenerator(model, max_tokens=8, logits_processors=[penalty])
        monkeypatch.setattr(generator, "_logprobs_required", lambda request: logprobs)
        if not overlap:
            monkeypatch.setattr(generator, "_can_overlap_decode", lambda request: False)
        with ThreadPoolExecutor(max_workers=1) as worker:
            # Production refault creates/materializes arrays on the scheduler
            # owner, not on the HTTP/main thread.
            def restore():
                restored = load_prompt_cache(str(tmp_path / "prefix.safetensors"))
                mx.eval(restored[0].state)
                generator.insert([[4, 5]], caches=[restored])
                return restored
            restored = worker.submit(restore).result()
            rows = collect_on_worker(generator, worker)
            def snapshot():
                mx.eval(restored[0].state)
                return [np.array(value) for value in restored[0].state]
            state = worker.submit(snapshot).result()
        return rows, state

    actual, actual_state = run(True)
    control, control_state = run(False)
    assert [row.token for row in actual] == [row.token for row in control]
    assert [row.finish_reason for row in actual] == [row.finish_reason for row in control]
    for left, right in zip(actual_state, control_state):
        np.testing.assert_array_equal(left, right)
    if logprobs:
        for left, right in zip(actual, control):
            np.testing.assert_array_equal(np.array(left.logprobs), np.array(right.logprobs))


def collect_on_worker(generator, worker):
    rows = []
    while True:
        first, rest = worker.submit(generator.next).result()
        if not first and not rest:
            return rows
        rows.extend(first + rest)


def test_cancel_pending_dense_decode_then_new_request_isolated():
    generator = SingleBatchGenerator(DenseModel(), max_tokens=8)
    uid = generator.insert([[1, 2, 3]])[0]
    generator.next()
    generator.next()  # a following token is now submitted asynchronously
    generator.remove([uid])
    assert generator.next() == ([], [])
    generator.insert([[6, 5]], max_tokens=[4])
    actual = collect(generator)
    control = SingleBatchGenerator(DenseModel(), max_tokens=4)
    control.insert([[6, 5]])
    expected = collect(control)
    assert [row.token for row in actual] == [row.token for row in expected]
    assert actual[-1].finish_reason == "length"
