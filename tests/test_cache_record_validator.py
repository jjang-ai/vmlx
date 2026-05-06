"""Hard regression tests for cache-record validation.

Pins the cache-safety contract: malformed/bogus cache records must miss the
cache, never trigger ``[metal::malloc]`` allocations of hundreds of gigabytes.

The original crash:
    [Engine error: [metal::malloc] Attempting to allocate 468462801024 bytes
    which is greater than the maximum allowed buffer size of 86586540032 bytes.]

These tests force-feed validator a record that *would* have produced that
allocation and assert it is rejected.
"""

from __future__ import annotations

import json
import sys
import types

import pytest


def _stub_tensor(shape, itemsize=2):
    """Return a duck-typed object with .shape and .itemsize that the validator
    inspects without needing real MLX. ``itemsize`` defaults to 2 (bf16/fp16)."""
    obj = types.SimpleNamespace()
    obj.shape = tuple(int(d) for d in shape)
    obj.itemsize = int(itemsize)
    obj.nbytes = obj.itemsize
    for d in obj.shape:
        obj.nbytes *= max(int(d), 0)
    obj.dtype = "bf16"
    return obj


def _import_validator():
    # Make sure repo root is importable when running this file standalone
    import os
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from vmlx_engine.cache_record_validator import (  # noqa: E402
        validate_cache_record,
        reject_or_warn,
        MAX_TENSOR_BYTES,
        MAX_TOTAL_RECORD_BYTES,
        MAX_TENSOR_DIM,
    )
    return (
        validate_cache_record,
        reject_or_warn,
        MAX_TENSOR_BYTES,
        MAX_TOTAL_RECORD_BYTES,
        MAX_TENSOR_DIM,
    )


def test_well_formed_dsv4_43layer_record_accepted():
    """Sanity: a normal DSV4-Flash 43-layer record validates."""
    (validate_cache_record, _, _, _, _) = _import_validator()
    cache_data = []
    for i in range(43):
        if i % 4 == 0:
            # MLA layer with composite state (dict-of-tensors)
            state = {
                "swa_keys": _stub_tensor((1, 128, 64, 128)),
                "swa_values": _stub_tensor((1, 128, 64, 128)),
                "csa_keys": _stub_tensor((1, 1, 64, 512)),
                "csa_values": _stub_tensor((1, 1, 64, 512)),
            }
            cache_data.append(("deepseek_v4", state, "{}", "DeepseekV4Cache", {}))
        else:
            # Plain KV layer
            cache_data.append((
                "kv",
                _stub_tensor((1, 128, 64, 128)),
                _stub_tensor((1, 128, 64, 128)),
            ))
    ok, reason, nbytes = validate_cache_record(
        cache_data, expected_num_layers=43, source="test:wellformed",
    )
    assert ok, reason
    assert nbytes < 200 * 1024 ** 2  # <200 MB for a 43-layer 64-token block


def test_468gb_corruption_rejected():
    """The exact regression: a single tensor whose shape would request ~468 GB
    must be rejected before any ``mx.concatenate`` / ``cache.state =`` runs."""
    (validate_cache_record, _, _, _, _) = _import_validator()
    # 468,462,801,024 bytes / 2 (bf16) = 234,231,400,512 elements.
    # Spread across (64K, 8000) shape would be ~512M elements per dim →
    # picks up via the per-tensor-byte cap.
    huge = _stub_tensor((1, 128, 200000, 4096))  # ~200 GB at bf16
    cache_data = [("kv", huge, _stub_tensor((1, 128, 64, 128)))] + [
        ("kv", _stub_tensor((1, 128, 64, 128)), _stub_tensor((1, 128, 64, 128)))
        for _ in range(42)
    ]
    ok, reason, _ = validate_cache_record(
        cache_data, expected_num_layers=43, source="test:468gb",
    )
    assert not ok
    assert "bytes" in reason.lower() or "dim" in reason.lower()


def test_layer_count_mismatch_rejected():
    """Wrong-model L2 entry: 80-layer record fed to 43-layer scheduler."""
    (validate_cache_record, _, _, _, _) = _import_validator()
    cache_data = [
        ("kv", _stub_tensor((1, 8, 64, 128)), _stub_tensor((1, 8, 64, 128)))
        for _ in range(80)
    ]
    ok, reason, _ = validate_cache_record(
        cache_data, expected_num_layers=43, source="test:layer_count",
    )
    assert not ok
    assert "layer count" in reason.lower()


def test_unknown_tag_rejected():
    """Garbage tag from a stale schema version must not silently pass."""
    (validate_cache_record, _, _, _, _) = _import_validator()
    cache_data = [("v99_future_tag", "blob")] + [
        ("kv", _stub_tensor((1, 8, 64, 128)), _stub_tensor((1, 8, 64, 128)))
        for _ in range(42)
    ]
    ok, reason, _ = validate_cache_record(
        cache_data, expected_num_layers=43, source="test:unknown_tag",
    )
    assert not ok
    assert "unknown tag" in reason.lower()


def test_dim_above_cap_rejected():
    """A single dim > MAX_TENSOR_DIM is corruption — short-circuit."""
    (validate_cache_record, _, _, _, MAX_TENSOR_DIM) = _import_validator()
    bogus = _stub_tensor((1, 8, MAX_TENSOR_DIM + 1, 128))
    cache_data = [("kv", bogus, _stub_tensor((1, 8, 64, 128)))]
    ok, reason, _ = validate_cache_record(
        cache_data, expected_num_layers=1, source="test:dim",
    )
    assert not ok
    assert "dim" in reason.lower()


def test_dsv4_pending_marker_passes():
    """``deepseek_v4_pending`` is a cheap placeholder — must always pass."""
    (validate_cache_record, _, _, _, _) = _import_validator()
    cache_data = [("deepseek_v4_pending", "DeepseekV4Cache", {})] * 43
    ok, reason, nbytes = validate_cache_record(
        cache_data, expected_num_layers=43, source="test:pending",
    )
    assert ok, reason
    assert nbytes == 0


def test_total_bytes_cap_rejected():
    """Many layers, each individually under the per-tensor cap AND under the
    per-dim cap, but summing above the per-record total cap, must still be
    rejected. Dim/byte caps catch single-tensor corruption; total cap catches
    accumulation corruption (e.g. an 80-layer block fed to a 43-layer model
    where each layer is otherwise reasonable shape)."""
    (validate_cache_record, _, _, MAX_TOTAL_RECORD_BYTES, MAX_TENSOR_DIM) = _import_validator()
    # ~410 MB per tensor: shape (1, 1, 200000, 1024) at bf16, all dims ≤ MAX
    big = _stub_tensor((1, 1, 200000, 1024))
    assert all(d <= MAX_TENSOR_DIM for d in big.shape)
    # 50 layers × (key + value) × 410 MB = ~41 GB → exceeds 16 GB total cap
    cache_data = [("kv", big, big)] * 50
    ok, reason, _ = validate_cache_record(
        cache_data, expected_num_layers=50, source="test:total",
    )
    assert not ok, "should reject when total bytes > MAX_TOTAL_RECORD_BYTES"
    assert "total_bytes" in reason.lower(), f"unexpected reason: {reason}"


def test_malformed_entry_rejected():
    """Non-tuple or short-tuple entries must reject (defends against
    deserialization that produced None or a dict in place of a tuple)."""
    (validate_cache_record, _, _, _, _) = _import_validator()
    cache_data = [None] + [
        ("kv", _stub_tensor((1, 8, 64, 128)), _stub_tensor((1, 8, 64, 128)))
        for _ in range(42)
    ]
    ok, reason, _ = validate_cache_record(
        cache_data, expected_num_layers=43, source="test:malformed",
    )
    assert not ok
    assert "malformed" in reason.lower() or "nonetype" in reason.lower()


def test_reject_or_warn_returns_false_on_corrupt():
    """Wrapper API caller pattern: ``if not reject_or_warn(...): return None``."""
    (_, reject_or_warn, _, _, _) = _import_validator()
    huge = _stub_tensor((1, 128, 200000, 4096))
    assert reject_or_warn(
        [("kv", huge, _stub_tensor((1, 8, 64, 128)))],
        expected_num_layers=1,
        source="test:reject",
    ) is False


def test_reject_or_warn_returns_true_on_clean():
    (_, reject_or_warn, _, _, _) = _import_validator()
    cache_data = [
        ("kv", _stub_tensor((1, 8, 64, 128)), _stub_tensor((1, 8, 64, 128)))
        for _ in range(2)
    ]
    assert reject_or_warn(
        cache_data, expected_num_layers=2, source="test:accept",
    ) is True


def test_quantized_kv_huge_meta_offset_rejected():
    """A small quantized record with a poisoned offset must reject before
    QuantizedKVCache.from_state can restore the offset and allocate later."""
    (validate_cache_record, _, _, _, _) = _import_validator()
    qtuple = (
        _stub_tensor((1, 8, 64, 128), itemsize=1),
        _stub_tensor((1, 8, 1, 1)),
        _stub_tensor((1, 8, 1, 1)),
    )
    cache_data = [("quantized_kv", qtuple, qtuple, ("999999999999", "64", "4"))]
    ok, reason, _ = validate_cache_record(
        cache_data, expected_num_layers=1, source="test:qmeta",
    )
    assert not ok
    assert "offset" in reason.lower()


def test_dsv4_huge_rotating_meta_offset_rejected():
    """DSV4 local RotatingKVCache meta_state must not be allowed to carry a
    huge offset/index through prefix reconstruction."""
    (validate_cache_record, _, _, _, _) = _import_validator()
    state = {
        "local": (
            _stub_tensor((1, 8, 64, 128)),
            _stub_tensor((1, 8, 64, 128)),
        ),
        "compressor": _stub_tensor((1, 1, 64, 512)),
    }
    cache_data = [
        (
            "deepseek_v4",
            state,
            ("0", "128", "999999999999", "0"),
            "DeepseekV4Cache",
            {"sliding_window": 128},
        )
    ]
    ok, reason, _ = validate_cache_record(
        cache_data, expected_num_layers=1, source="test:dsv4meta",
    )
    assert not ok
    assert "offset" in reason.lower()


def test_dsv4_prefill_rotating_meta_idx_may_exceed_sliding_window():
    """DSV4 prompt-boundary snapshots can have _idx > sliding_window.

    mlx-lm RotatingKVCache keeps full multi-token prefill tensors before
    decode starts rotating in place, so DSV4 long-context L2 validation must
    cap idx by the corruption ceiling, not by sliding_window/max_size.
    """
    (validate_cache_record, _, _, _, _) = _import_validator()
    state = (
        (
            _stub_tensor((1, 1, 391, 512)),
            _stub_tensor((1, 1, 391, 512)),
        ),
        (
            _stub_tensor((1, 3, 512)),
            _stub_tensor((1, 3, 512)),
            _stub_tensor((1, 97, 512)),
        ),
        (
            _stub_tensor((1, 3, 512)),
            _stub_tensor((1, 3, 512)),
            _stub_tensor((1, 97, 512)),
        ),
    )
    cache_data = [
        (
            "deepseek_v4",
            state,
            ("0", "128", "391", "391"),
            "DeepseekV4Cache",
            {"sliding_window": 128, "compress_ratio": 4},
        )
    ]
    ok, reason, _ = validate_cache_record(
        cache_data, expected_num_layers=1, source="test:dsv4meta-valid-prefill",
    )
    assert ok, reason


def test_rotating_kv_bad_max_size_rejected():
    (validate_cache_record, _, _, _, _) = _import_validator()
    cache_data = [
        (
            "rotating_kv",
            _stub_tensor((1, 8, 64, 128)),
            _stub_tensor((1, 8, 64, 128)),
            999999999,
            0,
        )
    ]
    ok, reason, _ = validate_cache_record(
        cache_data, expected_num_layers=1, source="test:rotating",
    )
    assert not ok
    assert "max_size" in reason.lower()


def test_tq_native_metadata_huge_decode_shape_rejected():
    from vmlx_engine.cache_record_validator import validate_tq_native_metadata

    tensors = {
        "tq_0_ck_indices_packed": _stub_tensor((1,), itemsize=4),
        "tq_0_ck_qjl_packed": _stub_tensor((1,), itemsize=4),
        "tq_0_ck_residual_norms": _stub_tensor((1,)),
        "tq_0_ck_vector_norms": _stub_tensor((1,)),
        "tq_0_cv_indices_packed": _stub_tensor((1,), itemsize=4),
        "tq_0_cv_vector_norms": _stub_tensor((1,)),
    }
    metadata = {
        "__tq_native__": "true",
        "__num_layers__": "1",
        "__layer_0_class__": "TurboQuantKVCache",
        "__tq_0_ck_shape__": json.dumps([1, 8, 999999, 128]),
        "__tq_0_ck_bits__": "3",
        "__tq_0_cv_shape__": json.dumps([1, 8, 64, 128]),
        "__tq_0_cv_bits__": "3",
        "__tq_0_offset__": "64",
        "__tq_0_key_dim__": "128",
        "__tq_0_value_dim__": "128",
        "__tq_0_key_bits__": "3",
        "__tq_0_value_bits__": "3",
        "__tq_0_sink_tokens__": "0",
    }
    ok, reason = validate_tq_native_metadata(tensors, metadata, expected_num_layers=1)
    assert not ok
    assert "shape" in reason.lower() or "dim" in reason.lower()


def test_tq_native_metadata_huge_offset_rejected():
    from vmlx_engine.cache_record_validator import validate_tq_native_metadata

    tensors = {
        "tq_0_ck_indices_packed": _stub_tensor((1,), itemsize=4),
        "tq_0_ck_qjl_packed": _stub_tensor((1,), itemsize=4),
        "tq_0_ck_residual_norms": _stub_tensor((1,)),
        "tq_0_ck_vector_norms": _stub_tensor((1,)),
        "tq_0_cv_indices_packed": _stub_tensor((1,), itemsize=4),
        "tq_0_cv_vector_norms": _stub_tensor((1,)),
    }
    metadata = {
        "__tq_native__": "true",
        "__num_layers__": "1",
        "__layer_0_class__": "TurboQuantKVCache",
        "__tq_0_ck_shape__": json.dumps([1, 8, 64, 128]),
        "__tq_0_ck_bits__": "3",
        "__tq_0_cv_shape__": json.dumps([1, 8, 64, 128]),
        "__tq_0_cv_bits__": "3",
        "__tq_0_offset__": "999999999999",
        "__tq_0_key_dim__": "128",
        "__tq_0_value_dim__": "128",
        "__tq_0_key_bits__": "3",
        "__tq_0_value_bits__": "3",
        "__tq_0_sink_tokens__": "0",
    }
    ok, reason = validate_tq_native_metadata(tensors, metadata, expected_num_layers=1)
    assert not ok
    assert "offset" in reason.lower()


def test_live_cache_rejects_quantized_tuple_huge_shape():
    from vmlx_engine.cache_record_validator import validate_live_cache

    qtuple = (
        _stub_tensor((1, 8, 999999, 128), itemsize=1),
        _stub_tensor((1, 8, 1, 1)),
        _stub_tensor((1, 8, 1, 1)),
    )
    layer = types.SimpleNamespace(
        keys=qtuple,
        values=qtuple,
        offset=64,
        group_size=64,
        bits=4,
    )
    ok, reason, _ = validate_live_cache([layer], source="test:live-qtuple")
    assert not ok
    assert "dim" in reason.lower() or "bytes" in reason.lower()


def test_live_cache_rejects_rotating_huge_offset():
    from vmlx_engine.cache_record_validator import validate_live_cache

    layer = types.SimpleNamespace(
        keys=_stub_tensor((1, 8, 64, 128)),
        values=_stub_tensor((1, 8, 64, 128)),
        offset=999999999999,
        max_size=128,
        keep=0,
        _idx=0,
    )
    ok, reason, _ = validate_live_cache([layer], source="test:live-rotating")
    assert not ok
    assert "offset" in reason.lower()


def test_live_cache_rejects_tq_encoded_huge_shape_when_keys_cleared():
    from vmlx_engine.cache_record_validator import validate_live_cache

    encoded = types.SimpleNamespace(
        shape=(1, 8, 999999, 128),
        indices_packed=_stub_tensor((1,), itemsize=4),
        qjl_packed=_stub_tensor((1,), itemsize=4),
        residual_norms=_stub_tensor((1,)),
        vector_norms=_stub_tensor((1,)),
    )
    layer = types.SimpleNamespace(
        keys=None,
        values=None,
        _compressed_keys=encoded,
        _compressed_values=encoded,
        offset=64,
        key_dim=128,
        value_dim=128,
        key_bits=4,
        value_bits=4,
        sink_tokens=0,
    )
    ok, reason, _ = validate_live_cache([layer], source="test:live-tq")
    assert not ok
    assert "shape" in reason.lower() or "dim" in reason.lower()


def test_live_cache_accepts_clean_kv_layer():
    from vmlx_engine.cache_record_validator import validate_live_cache

    layer = types.SimpleNamespace(
        keys=_stub_tensor((1, 8, 64, 128)),
        values=_stub_tensor((1, 8, 64, 128)),
        offset=64,
    )
    ok, reason, nbytes = validate_live_cache([layer], source="test:live-clean")
    assert ok, reason
    assert nbytes > 0


# ============================================================================
# Pre-mx.load safetensors header validation.
# Corrupt synthetic cache file => must reject before mx.load/decode.
# ============================================================================


def _write_safetensors_header(path, header_dict, *, total_data_size=0):
    """Write a synthetic safetensors file with the given header. Tensor data
    is filler — the header's data_offsets describe sizes that may or may not
    match. We only need the validator to NEVER call mx.load on a bad header.

    Returns the bytes written (header + filler data).
    """
    import json
    import struct
    header_bytes = json.dumps(header_dict).encode("utf-8")
    # Write 8-byte little-endian header length
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        # Pad with zeros so file size matches total_data_size if requested
        if total_data_size > 0:
            f.write(b"\x00" * total_data_size)


def test_header_validator_rejects_437gb_tensor(tmp_path):
    """A header declaring a single 437 GB tensor must be
    rejected BEFORE mx.load is called. This is the exact crash class."""
    (_, _, MAX_TENSOR_BYTES, _, _) = _import_validator()
    from vmlx_engine.cache_record_validator import (  # noqa: E402
        validate_safetensors_header,
        reject_safetensors_or_warn,
    )
    p = tmp_path / "bogus_437gb.safetensors"
    # Header declares a (200000, 4096, 128) bf16 tensor = 200K × 4096 × 128 × 2
    # = 209 GB. Multiply ndim by another factor for ~437 GB.
    header = {
        "layer_0_keys": {
            "dtype": "BF16",
            "shape": [200000, 4096, 128],
            "data_offsets": [0, 200000 * 4096 * 128 * 2],
        }
    }
    _write_safetensors_header(p, header, total_data_size=4096)  # tiny filler

    ok, reason = validate_safetensors_header(str(p), expected_num_layers=43)
    assert not ok, "header validator should reject 209+ GB tensor"
    assert "bytes" in reason or "dim" in reason, f"unexpected reason: {reason}"

    # File should still exist (validate_safetensors_header doesn't delete)
    assert p.exists()

    # reject_safetensors_or_warn with delete_on_reject=True must remove it
    assert reject_safetensors_or_warn(
        str(p), expected_num_layers=43, source="test", delete_on_reject=True,
    ) is False
    assert not p.exists(), "delete_on_reject=True should remove poisoned file"


def test_header_validator_rejects_dim_overflow(tmp_path):
    """A single dim past 256K must reject (catches stale schemas with bad seq)."""
    from vmlx_engine.cache_record_validator import validate_safetensors_header
    p = tmp_path / "bogus_dim.safetensors"
    header = {
        "layer_0_keys": {
            "dtype": "F16",
            "shape": [1, 8, 999999, 128],  # 999999 > 256K cap
            "data_offsets": [0, 1024],
        }
    }
    _write_safetensors_header(p, header)
    ok, reason = validate_safetensors_header(str(p))
    assert not ok
    assert "dim" in reason.lower()


def test_header_validator_rejects_unknown_dtype(tmp_path):
    """Future / unknown dtypes must reject rather than silently pass."""
    from vmlx_engine.cache_record_validator import validate_safetensors_header
    p = tmp_path / "bogus_dtype.safetensors"
    header = {
        "layer_0_keys": {
            "dtype": "F128_QUANTUM",  # not real
            "shape": [1, 8, 64, 128],
            "data_offsets": [0, 1024],
        }
    }
    _write_safetensors_header(p, header)
    ok, reason = validate_safetensors_header(str(p))
    assert not ok
    assert "dtype" in reason.lower()


def test_header_validator_rejects_layer_count_mismatch(tmp_path):
    """layer_99_* in a 43-layer model must reject."""
    from vmlx_engine.cache_record_validator import validate_safetensors_header
    p = tmp_path / "bogus_layers.safetensors"
    header = {
        "layer_99_keys": {
            "dtype": "BF16",
            "shape": [1],
            "data_offsets": [0, 2],
        }
    }
    _write_safetensors_header(p, header, total_data_size=2)
    ok, reason = validate_safetensors_header(str(p), expected_num_layers=43)
    assert not ok
    assert "layer" in reason.lower()


def test_header_validator_rejects_bad_data_offsets(tmp_path):
    from vmlx_engine.cache_record_validator import validate_safetensors_header
    p = tmp_path / "bogus_offsets.safetensors"
    header = {
        "layer_0_keys": {
            "dtype": "BF16",
            "shape": [4],
            "data_offsets": [0, 2],  # should be 8 bytes for 4 BF16 values
        }
    }
    _write_safetensors_header(p, header, total_data_size=2)
    ok, reason = validate_safetensors_header(str(p))
    assert not ok
    assert "data_offsets" in reason.lower()


def test_header_validator_rejects_truncated(tmp_path):
    """A file truncated mid-header must reject without throwing."""
    from vmlx_engine.cache_record_validator import validate_safetensors_header
    p = tmp_path / "truncated.safetensors"
    import struct
    with open(p, "wb") as f:
        f.write(struct.pack("<Q", 1000))  # claims 1000-byte header
        f.write(b"{")  # but only 1 byte of header
    ok, reason = validate_safetensors_header(str(p))
    assert not ok
    assert "short read" in reason.lower() or "header" in reason.lower()


def test_header_validator_rejects_huge_header(tmp_path):
    """Header_size > 16 MB is corrupt. Reject without reading the rest."""
    from vmlx_engine.cache_record_validator import validate_safetensors_header
    p = tmp_path / "huge_header.safetensors"
    import struct
    with open(p, "wb") as f:
        f.write(struct.pack("<Q", 100 * 1024 * 1024))  # 100 MB header claim
        f.write(b"{}")  # short body
    ok, reason = validate_safetensors_header(str(p))
    assert not ok
    assert "header_size" in reason.lower() or "out of bounds" in reason.lower()


def test_header_validator_accepts_clean_dsv4(tmp_path):
    """A reasonable 43-layer DSV4 header must pass."""
    from vmlx_engine.cache_record_validator import validate_safetensors_header
    p = tmp_path / "clean.safetensors"
    header = {}
    offset = 0
    for i in range(43):
        header[f"layer_{i}_keys"] = {
            "dtype": "BF16",
            "shape": [1, 1, 1, 1],
            "data_offsets": [offset, offset + 2],
        }
        offset += 2
        header[f"layer_{i}_values"] = {
            "dtype": "BF16",
            "shape": [1, 1, 1, 1],
            "data_offsets": [offset, offset + 2],
        }
        offset += 2
    _write_safetensors_header(p, header, total_data_size=offset)
    ok, reason = validate_safetensors_header(str(p), expected_num_layers=43)
    assert ok, reason


def test_header_validator_rejects_missing_file(tmp_path):
    from vmlx_engine.cache_record_validator import validate_safetensors_header
    ok, reason = validate_safetensors_header(str(tmp_path / "nope.safetensors"))
    assert not ok
    assert "exist" in reason.lower()


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(pytest.main([__file__, "-v"]))
