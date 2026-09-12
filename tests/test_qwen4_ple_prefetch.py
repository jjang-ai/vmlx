"""Bounded host reads: exact identity, caller-stream MLX and native rollback."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map_with_path

from tests.test_qwen4_eager_dispatch import _assert_exact, _cache_snapshot, _snapshot
from tests.test_qwen4_exp_runtime import _randomize, _tiny_args
from tests.test_qwen4_ple_host_gather import _bits, _table
from vmlx_engine.models.qwen4_exp import language, table_reader


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("parallel", [False, True])
def test_prefetch_exact_mixed_layouts_only_upload_on_caller(tmp_path, monkeypatch, dtype, parallel):
    monkeypatch.setenv("VMLX_QWEN4_PLE_HOST_GATHER", "1")
    monkeypatch.setenv("VMLX_QWEN4_PLE_PREFETCH", "1")
    table = _table(tmp_path, [(1, 32), (2, 64), (6, 32), (8, 128)], dtype)
    if parallel:
        table._read_pool = ThreadPoolExecutor(max_workers=1)
    rows = np.array([25, 0, 8, 14, 25, 7, 0])
    stream = mx.new_stream(mx.gpu)
    caller = threading.get_ident()
    dequantize = mx.dequantize
    calls = []

    def checked(*args, **kwargs):
        assert threading.get_ident() == caller
        assert mx.default_stream(mx.gpu) == stream
        calls.append((kwargs["bits"], kwargs["group_size"]))
        return dequantize(*args, **kwargs)

    try:
        with mx.stream(stream):
            expected = _bits(table.gather_mlx(rows)).copy()
            monkeypatch.setattr(mx, "dequantize", checked)
            ticket = table.prefetch_rows(rows)
            rows[0] = 1  # The ticket owns its immutable selection, not this view.
            assert not ticket.rows.flags.writeable
            payload = ticket.future.result(timeout=5)
            for _, _, hosts in payload[-1]:
                assert all(isinstance(value, np.ndarray) for value in hosts)
            actual = table.gather_mlx(ticket.rows, prepared=ticket)
            np.testing.assert_array_equal(_bits(actual), expected)
            assert len(calls) == 4
            assert table._prefetch_ticket is None and ticket.future is None
            assert table.prefetch_stats["consumed"] == 1
    finally:
        table.close()


def test_prefetch_single_flight_bounds_identity_and_default(tmp_path, monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_PLE_PREFETCH", raising=False)
    table = _table(tmp_path, [(3, 32), (4, 32)], mx.float16, 160)
    other_path = tmp_path / "other"
    other_path.mkdir()
    other = _table(other_path, [(3, 32), (4, 32)], mx.float16, 160)
    try:
        assert table.prefetch_rows(np.array([0])) is None
        table._host_assembly = table._prefetch_enabled = True
        assert table.prefetch_rows(np.zeros(table_reader._PREFETCH_MAX_ROWS + 1)) is None
        monkeypatch.setattr(table_reader, "_PREFETCH_MAX_PACKED_BYTES", 1)
        assert table.prefetch_rows(np.array([0])) is None
        monkeypatch.setattr(table_reader, "_PREFETCH_MAX_PACKED_BYTES", 8 * 1024 * 1024)
        for rows in ([-1], [table.total_rows]):
            with pytest.raises(IndexError):
                table.prefetch_rows(np.array(rows))
        ticket = table.prefetch_rows(np.array([0, 7]))
        assert table.prefetch_rows(np.array([1])) is None
        with pytest.raises(ValueError, match="different table"):
            other.gather_mlx(np.array([0, 7]), prepared=ticket)
        with pytest.raises(ValueError, match="exact row IDs"):
            table.gather_mlx(np.array([7, 0]), prepared=ticket)
        assert table._prefetch_ticket is None
        assert table.prefetch_stats["consumed"] == 0
        with pytest.raises(ValueError, match="already released"):
            table.gather_mlx(np.array([0, 7]), prepared=ticket)
        ticket.close()  # Idempotent after failed consumption.
    finally:
        table.close()
        other.close()
    with pytest.raises(RuntimeError, match="closed"):
        table.prefetch_rows(np.array([0]))


def test_failed_read_drains_siblings_and_close_drains_before_descriptors(tmp_path, monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_PLE_HOST_GATHER", "1")
    monkeypatch.setenv("VMLX_QWEN4_PLE_PREFETCH", "1")
    table = _table(tmp_path, [(4, 32), (6, 32)], mx.float16, 160)
    table._read_pool = ThreadPoolExecutor(max_workers=2)
    entered, sibling_done = threading.Event(), threading.Event()
    original = table.shards[1].read_rows

    def fail(*args, **kwargs):
        assert entered.wait(5)
        raise OSError("read failure")

    def sibling(*args, **kwargs):
        entered.set()
        result = original(*args, **kwargs)
        sibling_done.set()
        return result

    monkeypatch.setattr(table.shards[0], "read_rows", fail)
    monkeypatch.setattr(table.shards[1], "read_rows", sibling)
    try:
        ticket = table.prefetch_rows(np.array([0, 7]))
        with pytest.raises(OSError, match="read failure"):
            table.gather_mlx(np.array([0, 7]), prepared=ticket)
        assert sibling_done.is_set() and table._prefetch_ticket is None
        assert table.prefetch_stats["consumed"] == 0
    finally:
        table.close()


def test_close_waits_for_pending_host_work(tmp_path, monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_PLE_HOST_GATHER", "1")
    monkeypatch.setenv("VMLX_QWEN4_PLE_PREFETCH", "1")
    table = _table(tmp_path, [(4, 32)], mx.float16, 160)
    entered, release, completed, closed = (threading.Event() for _ in range(4))
    original = table.shards[0].read_rows

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        result = original(*args, **kwargs)
        completed.set()
        return result

    for source in table._pread_files.values():
        old_close = source.close
        def checked_close(old=old_close):
            assert completed.is_set()
            old()
            closed.set()
        monkeypatch.setattr(source, "close", checked_close)
    monkeypatch.setattr(table.shards[0], "read_rows", blocked)
    ticket = table.prefetch_rows(np.array([0]))
    assert entered.wait(5)
    with ThreadPoolExecutor(max_workers=1) as pool:
        closing = pool.submit(table.close)
        assert not closing.done() and not closed.is_set()
        release.set()
        closing.result(timeout=5)
    assert closed.is_set() and ticket.future is None
    assert all(source._fd is None for source in table._pread_files.values())


def _file_model(tmp_path, monkeypatch, dtype):
    monkeypatch.setenv("VMLX_QWEN4_PLE_HOST_GATHER", "1")
    monkeypatch.setenv("VMLX_QWEN4_PLE_PREFETCH", "1")
    monkeypatch.setenv("VMLX_QWEN4_EAGER_DISPATCH", "1")
    args = _tiny_args()
    args.linear_key_head_dim = args.linear_value_head_dim = 32
    args.ple_embed_dim = 1024  # 16 heads x 64-wide actual rows.
    args.ngram_vocab_size_base = 67
    args.split_ngram_parts = 4
    model = language.LanguageModel(args)
    _randomize(model)
    model.update(tree_map_with_path(
        lambda path, value: value.astype(dtype)
        if mx.issubdtype(value.dtype, mx.floating) and model.cast_predicate(path)
        else value, model.parameters(),
    ))
    model.eval()
    mx.eval(model.parameters())
    ple = next(layer.ple for layer in model.layers if layer.ple is not None)
    tensors, mapping, specs = {}, {}, {}
    for index, shard in enumerate(ple.ngram_embedding.shards):
        name = f"table.shards.{index}"
        bits, group = (2, 4, 6, 8)[index % 4], (32, 64)[index % 2]
        quantized = mx.quantize(shard.weight, bits=bits, group_size=group)
        for suffix, value in zip(("weight", "scales", "biases"), quantized):
            tensors[f"{name}.{suffix}"] = value
            mapping[f"{name}.{suffix}"] = "table.safetensors"
        specs[name] = {"bits": bits, "group_size": group, "mode": "affine"}
    mx.save_safetensors(str(tmp_path / "table.safetensors"), tensors)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    (tmp_path / "config.json").write_text(json.dumps({"quantization": specs}))
    table = table_reader.FileBackedQuantizedNGramTable(
        tmp_path, "table.shards.{}", len(ple.ngram_embedding.shards),
        ple.ngram_embedding.head_dim,
    )
    ple.ngram_embedding.set_file_backed(table)
    return model, ple, table


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
def test_prefetch_exact_connected_verify_rollback(tmp_path, monkeypatch, dtype, accepted):
    from vmlx_engine.mllm_batch_generator import _native_mtp_rollback_to_confirmed
    model, _, table = _file_model(tmp_path, monkeypatch, dtype)
    results = []
    try:
        with mx.stream(mx.new_stream(mx.gpu)):
            for enabled in (False, True):
                table._prefetch_enabled = enabled
                cache = model.make_cache()
                first = model(mx.array([[11, 17, 23, 29, 31, 37, 41, 43, 47]]), cache=cache)
                prefix = (_snapshot(first.logits), _cache_snapshot(cache))
                verified = model(mx.array([[53, 59, 61, 67]]), cache=cache, n_confirmed=1)
                before = (_snapshot(verified.logits), _cache_snapshot(cache))
                if accepted < 3:
                    assert _native_mtp_rollback_to_confirmed(
                        cache, reject_tokens=3 - accepted, accepted_drafts=accepted,
                    )
                after = _cache_snapshot(cache)
                next_logits = model(mx.array([[71, 73]]), cache=cache).logits
                results.append((prefix, before, after, _snapshot(next_logits), _cache_snapshot(cache)))
        _assert_exact(*results)
        assert table.prefetch_stats["consumed"] == 3
        assert table.prefetch_stats["discarded"] == 0
        assert table._prefetch_ticket is None
    finally:
        table.close()


def test_model_error_drains_ticket_and_diagnostics_checkpoint_bypass(tmp_path, monkeypatch):
    model, _, table = _file_model(tmp_path, monkeypatch, mx.float16)
    original = language.DecoderLayer.__call__
    def fail_first(self, *args, **kwargs):
        if self is model.layers[0]:
            raise RuntimeError("failure before PLE")
        return original(self, *args, **kwargs)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(language.DecoderLayer, "__call__", fail_first)
            with pytest.raises(RuntimeError, match="before PLE"):
                model(mx.array([[11]]), cache=model.make_cache())
        assert table._prefetch_ticket is None
        assert table.prefetch_stats["discarded"] == 1
        before = table.prefetch_stats.copy()
        # Disabling the model-level switch must skip even the preparation
        # loop, leaving default decode free of extra hashing/reader work.
        model.model._ple_prefetch = False
        mx.eval(model(mx.array([[11]]), cache=model.make_cache()).logits)
        assert table.prefetch_stats == before
        model.model._ple_prefetch = True
        ids = mx.array([[11, 17, 23, 29, 31, 37, 41, 43, 47]])
        mx.eval(model(ids, cache=model.make_cache(), prefill_checkpoint_steps=(4, 8)).logits)
        with monkeypatch.context() as patch:
            patch.setattr(language, "_layer_profile_enabled", lambda inputs: True)
            mx.eval(model(ids[:, :1], cache=model.make_cache()).logits)
        with monkeypatch.context() as patch:
            patch.setattr(language, "_layer_fingerprint_enabled", lambda inputs: True)
            mx.eval(model(ids[:, :1], cache=model.make_cache()).logits)
        mx.eval(model(mx.array([list(range(65))]), cache=model.make_cache()).logits)
        assert table.prefetch_stats == before
    finally:
        table.close()


def test_changed_ple_context_rejects_prepared_rows_without_cache_update(tmp_path, monkeypatch):
    model, ple, table = _file_model(tmp_path, monkeypatch, mx.float16)
    cache = model.make_cache()[1]
    ids = mx.array([[11, 17]])
    try:
        ticket = ple.prepare_read(ids, cache)
        assert cache[2] is None and cache[3] is None
        cache[2] = mx.array([[41, 43]])
        before = _snapshot(cache.state)
        with pytest.raises(ValueError, match="exact row IDs"):
            ple._embed(ids, cache, prepared=ticket)
        _assert_exact(before, _snapshot(cache.state))
        assert table._prefetch_ticket is None
    finally:
        table.close()
