"""Write a fine-tuned drafter as a ``dspark/`` directory in the release format.

The output mirrors the source directory file for file: the same shard file names and tensor names
(``language_model.model.dspark.layers.N.decoder_layer.*``, ``final_norm.*``), the same dtypes and shapes, tensors in
sorted order and contiguous like the release, the same ``model.safetensors.index.json`` layout, ``config.json`` and
every other file copied byte for byte. Decoder linears the source stores as MXFP8 (e4m3 bytes ``F8_E4M3`` + e8m0
``weight_scale_inv``) are written as MXFP8 again (``fmt="mxfp8"``, default; MLX's mxfp8 quantization, the same
round trip the fake-quantized training forward used) or as BF16 ``weight`` tensors without scales (``fmt="bf16"``;
the runtime loads unpaired weights as float linears). A tensor whose trained value equals the source value is copied
byte for byte. The source directory is never written.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np

from ..drafter import minimax_tensor_name
from .objective import MXFP8
from .records import SafetensorsMap, write_safetensors


def _bits(a: mx.array, dtype) -> np.ndarray:
    a = a.astype(dtype)
    mx.eval(a)
    view = {mx.bfloat16: mx.uint16, mx.uint8: mx.uint8, mx.float32: mx.float32}[dtype]
    return np.array(mx.view(a, view) if view is not mx.float32 else a)


def export_dspark(src_dir: str | Path, params: dict, dst_dir: str | Path, *, fmt: str = "mxfp8") -> dict:
    """``params``: every drafter tensor under its live module path (``objective.drafter_params`` layout; float).
    Returns a report {tensors, changed, unchanged, bytes}."""
    if fmt not in ("mxfp8", "bf16"):
        raise ValueError(f"fmt {fmt!r}: mxfp8 or bf16")
    src, dst = Path(src_dir), Path(dst_dir)
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"{dst} exists and is not empty")
    if dst.resolve() == src.resolve():
        raise ValueError("export never writes the source directory")
    shards = sorted(p for p in src.glob("*.safetensors") if not p.name.startswith("._"))
    if not shards:
        raise ValueError(f"{src}: no safetensors")
    dst.mkdir(parents=True, exist_ok=True)
    report = {"tensors": 0, "changed": [], "unchanged": 0, "bytes": 0, "fmt": fmt}
    weight_map, used = {}, set()
    for shard in shards:
        sm = SafetensorsMap(shard)
        names = set(sm.header)
        out = {}
        for name in sorted(names):
            spec = sm.header[name]
            if name.endswith(".weight_scale_inv"):
                continue                                               # written with its weight
            live = minimax_tensor_name(name)
            if live not in params:
                raise KeyError(f"{name} ({live}) has no trained value")
            used.add(live)
            value = params[live]
            pair = name[: -len(".weight")] + ".weight_scale_inv" if name.endswith(".weight") else None
            if pair in names:                                         # an MXFP8 linear in the source
                orig_q = mx.array(np.array(sm.arr(name)))
                orig_s = mx.array(np.array(sm.arr(pair)))
                orig = mx.dequantize(mx.view(orig_q, mx.uint32), orig_s, None, dtype=mx.bfloat16, **MXFP8)
                same = bool(mx.array_equal(value.astype(mx.float32), orig.astype(mx.float32)).item())
                if same:
                    out[name] = (spec["dtype"], np.array(sm.arr(name)))
                    out[pair] = (sm.header[pair]["dtype"], np.array(sm.arr(pair)))
                    report["unchanged"] += 1
                elif fmt == "mxfp8":
                    q, s = mx.quantize(value.astype(mx.float32), **MXFP8)[:2]
                    mx.eval(q, s)
                    qb = np.array(mx.view(q, mx.uint8)).reshape(spec["shape"])
                    out[name] = (spec["dtype"], qb)
                    out[pair] = (sm.header[pair]["dtype"], np.array(s).reshape(sm.header[pair]["shape"]))
                    report["changed"].append(live)
                else:
                    out[name] = ("BF16", _bits(value, mx.bfloat16).reshape(spec["shape"]))
                    report["changed"].append(live)
                report["tensors"] += 1
                continue
            if spec["dtype"] != "BF16":
                raise TypeError(f"{name}: unexpected source dtype {spec['dtype']}")
            bits = _bits(value, mx.bfloat16).reshape(spec["shape"])
            if np.array_equal(bits, np.asarray(sm.arr(name))):
                report["unchanged"] += 1
            else:
                report["changed"].append(live)
            out[name] = ("BF16", bits)
            report["tensors"] += 1
        report["bytes"] += write_safetensors(dst / shard.name, out, metadata={"format": "pt"})
        weight_map.update({k: shard.name for k in out})
    missing = set(params) - used
    if missing:
        raise KeyError(f"trained tensors with no place in the release layout: {sorted(missing)[:5]}")
    index = src / "model.safetensors.index.json"
    if index.is_file():
        idx = json.loads(index.read_text())
        idx["metadata"] = {**(idx.get("metadata") or {}), "total_size": report["bytes"]}
        idx["weight_map"] = dict(sorted(weight_map.items()))
        (dst / index.name).write_text(json.dumps(idx, indent=2) + "\n")
    for f in src.iterdir():
        if f.is_file() and not f.name.startswith(".") and not f.name.endswith(".safetensors") \
                and f.name != "model.safetensors.index.json":
            shutil.copy2(f, dst / f.name)
    return report
