"""Declared native cache tags bypass legacy scans without changing fallback."""
import json

import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine import block_disk_store as disk


def payload():
    x = mx.arange(8, dtype=mx.float16).reshape(1, 1, 2, 4)
    return disk._serialize_block([("kv", x, x + 1)])[:2]


def edit_meta(tensors, edit):
    meta = json.loads(bytes(tensors["__vmlx_block_meta__"].tolist()))
    edit(meta)
    tensors["__vmlx_block_meta__"] = mx.array(
        list(json.dumps(meta).encode()), dtype=mx.uint8
    )


def test_declared_tag_does_not_scan_legacy_keys(monkeypatch):
    tensors, dtype = payload()
    def unexpected(*args):
        pytest.fail("explicit cache tag invoked legacy inference")
    monkeypatch.setattr(disk, "_infer_layer_type", unexpected)
    result = disk._deserialize_block(tensors, dtype)
    assert result[0][0] == "kv"
    assert bool(mx.array_equal(result[0][1], mx.arange(8).reshape(1, 1, 2, 4)))


@pytest.mark.parametrize("damage", ["absent_map", "absent_layer"])
def test_missing_tag_keeps_legacy_inference(monkeypatch, damage):
    tensors, dtype = payload()
    edit_meta(tensors, lambda meta: meta.pop("__layer_types__")
              if damage == "absent_map" else meta["__layer_types__"].pop("0"))
    original = disk._infer_layer_type
    calls = []
    def infer(*args):
        calls.append(args[1])
        return original(*args)
    monkeypatch.setattr(disk, "_infer_layer_type", infer)
    result = disk._deserialize_block(tensors, dtype)
    assert calls == [0]
    assert result[0][0] == "kv"


@pytest.mark.parametrize("tag", [None, "", "unknown"])
def test_explicit_unknown_tag_does_not_become_legacy_kv(tag):
    tensors, dtype = payload()
    edit_meta(tensors, lambda meta: meta["__layer_types__"].update({"0": tag}))
    result = disk._deserialize_block(tensors, dtype)
    assert result == [("skip",)]
