"""Exact mixed-layout PLE host assembly, including disk and dtype ownership."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np
import pytest

from vmlx_engine.models.qwen4_exp.table_reader import FileBackedQuantizedNGramTable


def _table(tmp_path, specs, dtype, width=256):
    tensors, weight_map, quantization, manifest = {}, {}, {}, {}
    rng = np.random.default_rng(7201)
    for i, (bits, group) in enumerate(specs):
        name = f"table.shards.{i}"
        count = 7 if i < len(specs) - 1 else 5
        if bits == 1:
            codes = rng.integers(0, 2, (count, width), dtype=np.uint32)
            packed = (codes.reshape(count, -1, 32) << np.arange(32, dtype=np.uint32))
            weight = mx.array(packed.sum(-1, dtype=np.uint32))
            scales = mx.full((count, width // group), 0.5, dtype=dtype)
            biases = mx.full((count, width // group), -0.25, dtype=dtype)
            manifest[name] = {"bits": 1, "storage_bits": 1}
        else:
            dense = mx.array(rng.standard_normal((count, width)).astype(np.float32)).astype(dtype)
            weight, scales, biases = mx.quantize(dense, bits=bits, group_size=group)
        mx.eval(weight, scales, biases)
        for suffix, value in zip(("weight", "scales", "biases"), (weight, scales, biases)):
            key = f"{name}.{suffix}"
            tensors[key] = value
            weight_map[key] = "table.safetensors"
        quantization[name] = {"bits": bits, "group_size": group, "mode": "affine"}
    mx.save_safetensors(str(tmp_path / "table.safetensors"), tensors)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (tmp_path / "config.json").write_text(json.dumps({"quantization": quantization}))
    (tmp_path / "jang_config.json").write_text(json.dumps({
        "quantization": {"tensor_quantization_manifest": manifest}
    }))
    return FileBackedQuantizedNGramTable(tmp_path, "table.shards.{}", len(specs), width)


def _bits(value):
    mx.eval(value)
    if value.dtype == mx.bfloat16:
        return np.asarray(value.view(mx.uint16))
    return np.asarray(value)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("specs,width", [
    ([(3, 32), (4, 32), (3, 32), (6, 32)], 160),
    ([(4, 32), (4, 64), (4, 32), (4, 128)], 256),
    ([(1, 32), (2, 64), (5, 32), (8, 128)], 256),
    ([(6, 64), (6, 64), (6, 64), (6, 64)], 256),
])
def test_exact_host_gather_across_signatures_and_dtypes(tmp_path, dtype, specs, width):
    table = _table(tmp_path, specs, dtype, width)
    try:
        # Reversed order, duplicates, multiple shards and the short final shard.
        rows = np.array([25, 0, 14, 8, 25, 4, 19, 7, 0], dtype=np.int64)
        table._host_assembly = False
        expected = _bits(table.gather_mlx(rows)).copy()
        table._host_assembly = True
        actual = table.gather_mlx(rows)
        assert actual.dtype == dtype
        np.testing.assert_array_equal(_bits(actual), expected)
        assert table.host_gather_stats == {
            "calls": 1, "rows": 9, "unique_rows": 7, "shards": 4,
            "layout_groups": len(set(shard.layout_signature for shard in table.shards)),
        }
    finally:
        table.close()


def test_raw_bf16_pread_and_memmap_match_and_dequantize_once_per_layout(tmp_path, monkeypatch):
    table = _table(tmp_path, [(3, 32), (4, 32), (3, 32), (4, 32)], mx.bfloat16, 160)
    table._host_assembly = True
    try:
        rows = np.array([0, 7, 14, 21, 2, 15, 21], dtype=np.int64)
        expected = _bits(table.gather_mlx(rows)).copy()
        for shard in table.shards:
            local = np.array([0, 1, 0], dtype=np.int64)
            mmap = shard.read_rows(local, preserve_bits=True)
            pread = shard.read_rows(local, use_pread=True, preserve_bits=True)
            assert mmap[1].dtype == np.uint16
            for a, b in zip(mmap, pread):
                np.testing.assert_array_equal(a, b)
        table._read_pool = ThreadPoolExecutor(max_workers=4)
        calls = []
        original = mx.dequantize

        def counted(*args, **kwargs):
            calls.append((kwargs["bits"], kwargs["group_size"]))
            return original(*args, **kwargs)

        monkeypatch.setattr(mx, "dequantize", counted)
        profile = {}
        actual = table.gather_mlx(rows, profile=profile)
        np.testing.assert_array_equal(_bits(actual), expected)
        assert sorted(calls) == [(3, 32), (4, 32)]
        assert all(profile[k] >= 0 for k in [
            "ssd_rows_cpu_ms", "host_to_mlx_ms", "dequant_gpu_ms", "scatter_gpu_ms"
        ])
    finally:
        table.close()


def test_empty_and_invalid_rows_never_read_shards(tmp_path, monkeypatch):
    table = _table(tmp_path, [(4, 32), (6, 32)], mx.float16, 160)
    table._host_assembly = True
    try:
        def forbidden(*a, **kw):
            raise AssertionError("read attempted for empty/out-of-range request")
        for shard in table.shards:
            monkeypatch.setattr(shard, "read_rows", forbidden)
        empty = table.gather_mlx(np.array([], dtype=np.int64))
        assert empty.shape == (0, 160) and empty.dtype == mx.float16
        for rows in ([-1], [table.total_rows], [0, table.total_rows]):
            with pytest.raises(IndexError):
                table.gather_mlx(np.array(rows))
        assert table.host_gather_stats["calls"] == 0
    finally:
        table.close()


def test_failed_parallel_read_drains_sibling_before_return(tmp_path, monkeypatch):
    table = _table(tmp_path, [(4, 32), (4, 32)], mx.float16, 160)
    table._host_assembly = True
    table._read_pool = ThreadPoolExecutor(max_workers=2)
    sibling_started, release, sibling_done = (threading.Event() for _ in range(3))
    def failed(*a, **kw):
        assert sibling_started.wait(2)
        release.set()
        raise OSError("owned read failure")
    def sibling(*a, **kw):
        sibling_started.set()
        assert release.wait(2)
        sibling_done.set()
        return (None, None, None)
    monkeypatch.setattr(table.shards[0], "read_rows", failed)
    monkeypatch.setattr(table.shards[1], "read_rows", sibling)
    try:
        with pytest.raises(OSError, match="owned read failure"):
            table.gather_mlx(np.array([0, 7]))
        assert sibling_done.is_set()
        assert table.host_gather_stats["calls"] == 0
    finally:
        release.set()
        table.close()


@pytest.mark.parametrize("entry", ["legacy", "assembled", "prefetched"])
def test_failed_shard_submission_drains_accepted_reads(tmp_path, monkeypatch, entry):
    table = _table(tmp_path, [(4, 32), (6, 32)], mx.float16, 160)
    rows = np.array([0, 7, 0, 9], dtype=np.int64)
    table._host_assembly = entry != "legacy"
    table._prefetch_enabled = entry == "prefetched"
    expected = _bits(table.gather_mlx(rows)).copy()
    table._read_pool = ThreadPoolExecutor(max_workers=2)
    entered, release, completed, rejected, returned = (
        threading.Event() for _ in range(5)
    )
    real_submit = table._read_pool.submit
    real_read = table.shards[0].read_rows
    submitted = 0

    def controlled_read(*args, **kwargs):
        entered.set()
        assert release.wait(5), "test must release accepted SSD read"
        value = real_read(*args, **kwargs)
        completed.set()
        return value

    def reject_second_submit(*args, **kwargs):
        nonlocal submitted
        submitted += 1
        if submitted == 2:
            assert entered.wait(5)
            rejected.set()
            raise RuntimeError("owned shard submission failure")
        return real_submit(*args, **kwargs)

    def failed_gather():
        try:
            ticket = table.prefetch_rows(rows) if entry == "prefetched" else None
            # Failure occurs in host I/O, before any MLX upload on this thread.
            return table.gather_mlx(rows, prepared=ticket)
        finally:
            returned.set()

    monkeypatch.setattr(table.shards[0], "read_rows", controlled_read)
    monkeypatch.setattr(table._read_pool, "submit", reject_second_submit)
    try:
        with ThreadPoolExecutor(max_workers=1) as caller:
            task = caller.submit(failed_gather)
            try:
                assert rejected.wait(5)
                escaped_before_read_finished = returned.wait(0.1)
            finally:
                release.set()
            with pytest.raises(RuntimeError, match="owned shard submission failure"):
                task.result(timeout=5)
        assert completed.is_set(), "caller returned with an accepted read outstanding"
        assert not escaped_before_read_finished
        assert table._prefetch_ticket is None
        # Failed submission must not poison the table or change row order/bits.
        monkeypatch.setattr(table._read_pool, "submit", real_submit)
        ticket = table.prefetch_rows(rows) if entry == "prefetched" else None
        np.testing.assert_array_equal(_bits(table.gather_mlx(rows, prepared=ticket)), expected)
    finally:
        release.set()
        table.close()


@pytest.mark.parametrize("value,expected", [
    (None, True), ("1", True), ("0", False), ("false", False),
    ("off", False), ("no", False), ("", False),
])
def test_host_gather_default_on_with_explicit_opt_out(tmp_path, monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("VMLX_QWEN4_PLE_HOST_GATHER", raising=False)
    else:
        monkeypatch.setenv("VMLX_QWEN4_PLE_HOST_GATHER", value)
    table = _table(tmp_path, [(4, 32)], mx.float16, 160)
    try:
        assert table._host_assembly is expected
        # Changing the environment cannot switch an already loaded table.
        monkeypatch.setenv("VMLX_QWEN4_PLE_HOST_GATHER", "0" if expected else "1")
        assert table._host_assembly is expected
    finally:
        table.close()
