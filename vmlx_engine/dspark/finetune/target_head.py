"""The target pieces a DSpark drafter borrows: the target's token embedding (block rows ``[anchor, mask ...]``) and its
LM head (draft logits). Saved once by data generation, as stored in the bundle (quantized or float), so training and
offline evaluation use exactly the runtime's head without loading the 95 GB target."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

_KINDS = (nn.QuantizedLinear, nn.Linear, nn.QuantizedEmbedding, nn.Embedding)


def _describe(module) -> dict:
    if not isinstance(module, _KINDS):
        raise TypeError(f"target head piece {type(module).__name__} is not a (quantized) Linear / Embedding")
    d = {"kind": type(module).__name__}
    if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
        d.update(group_size=int(module.group_size), bits=int(module.bits), mode=str(getattr(module, "mode", "affine")))
    return d


def save_target_head(target, path: str | Path) -> Path:
    """``target``: a DSpark target adapter (``embed_tokens``, ``tied``, ``model.lm_head``). Writes
    ``target_head.safetensors`` + ``target_head.json`` under ``path`` (a directory)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    pieces = {"embed_tokens": target.embed_tokens}
    if not getattr(target, "tied", False):
        pieces["lm_head"] = target.model.lm_head
    desc, arrays = {"tied": bool(getattr(target, "tied", False))}, {}
    for name, module in pieces.items():
        desc[name] = _describe(module)
        for k, v in tree_flatten(module.parameters()):
            arrays[f"{name}.{k}"] = v
    mx.eval(arrays)
    mx.save_safetensors(str(path / "target_head.safetensors"), arrays, metadata={"format": "mlx"})
    (path / "target_head.json").write_text(json.dumps(desc, indent=1))
    return path


def _build(desc: dict, arrays: dict, name: str):
    w = arrays[f"{name}.weight"]
    kind = desc["kind"]
    if kind in ("QuantizedLinear", "QuantizedEmbedding"):
        gs, bits, mode = int(desc["group_size"]), int(desc["bits"]), desc.get("mode", "affine")
        cols = int(arrays[f"{name}.scales"].shape[-1]) * gs
        rows = int(w.shape[0])
        if kind == "QuantizedLinear":
            m = nn.QuantizedLinear(cols, rows, bias=f"{name}.bias" in arrays, group_size=gs, bits=bits, mode=mode)
        else:
            m = nn.QuantizedEmbedding(rows, cols, group_size=gs, bits=bits, mode=mode)
    elif kind == "Linear":
        m = nn.Linear(int(w.shape[1]), int(w.shape[0]), bias=f"{name}.bias" in arrays)
    else:
        m = nn.Embedding(int(w.shape[0]), int(w.shape[1]))
    m.load_weights([(k[len(name) + 1:], v) for k, v in arrays.items() if k.startswith(name + ".")], strict=True)
    m.freeze()
    return m


class TargetHead:
    """``embed(ids)`` and ``head(hidden) -> logits`` as the runtime computes them (``target.embed_tokens`` /
    ``target.lm_head``)."""

    def __init__(self, path: str | Path):
        path = Path(path)
        desc = json.loads((path / "target_head.json").read_text())
        arrays = mx.load(str(path / "target_head.safetensors"))
        self.embed_tokens = _build(desc["embed_tokens"], arrays, "embed_tokens")
        self.tied = bool(desc.get("tied"))
        if self.tied:
            self.lm_head = lambda h: self.embed_tokens.as_linear(h)
        else:
            self.lm_head = _build(desc["lm_head"], arrays, "lm_head")
        mx.eval(arrays)

    def embed(self, ids: mx.array) -> mx.array:
        return self.embed_tokens(ids)

    def head(self, h: mx.array) -> mx.array:
        return self.lm_head(h)
