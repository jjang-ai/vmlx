# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 fused attention Metal kernels: thin loader for the standalone ``attn_*.metal`` sources.

The ``.metal`` files are the single source for vMLX (Python, ``mx.fast.metal_kernel``) and vmlx-swift
(``MLXFast.metalKernel``). Each ``attn_<name>.metal`` file is a kernel BODY; ``attn_common.metal`` is the ``header``
of every kernel. A kernel's source is::

    "".join(f"  using {n} = {t};\\n" for n, t in types) + "".join(f"  constexpr int {n} = {v};\\n" ...) + body

(``float`` constants as ``constexpr float``), compiled with the input / output names below and
``ensure_row_contiguous`` as listed. The constants, input / output shapes, grid and threadgroup of every kernel are
documented at the top of its ``.metal`` file; ``fused_attn.py`` is the reference host code.

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import mlx.core as mx

DIR = Path(__file__).resolve().parent

# name -> (input names, output names, ensure_row_contiguous)
SPECS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...], bool]] = {
    "qkv": (("x", "nw", "q_w", "q_s", "q_b", "k_w", "k_s", "k_b", "v_w", "v_s", "v_b",
             "iq_w", "iq_s", "iq_b", "ik_w", "ik_s", "ik_b"), ("y",), True),
    "heads": (("y", "qn", "kn", "iqn", "ikn", "meta", "kc", "vc", "ic"),
              ("q", "iq", "cnt", "kn_out", "vn_out", "ikn_out"), False),
    "scores": (("iq", "ic", "meta"), ("sc",), False),
    "msa": (("q", "kc", "vc", "sc", "cnt", "meta", "scale"), ("out", "po", "pm", "pl"), False),
    "oproj": (("o", "xr", "w", "s", "b"), ("h", "ssp"), True),
    # multi-row builds on precomputed input chunks (VMLX_M3_ROWS_PRE): attn_xprep + attn_qkv_pre / attn_oproj_pre
    "xprep": (("x", "nw"), ("xa", "xb", "xs", "inv"), True),
    "qkv_pre": (("xa", "xb", "xs", "inv", "q_w", "q_s", "q_b", "k_w", "k_s", "k_b", "v_w", "v_s", "v_b",
                 "iq_w", "iq_s", "iq_b", "ik_w", "ik_s", "ik_b"), ("y",), True),
    "oproj_pre": (("xa", "xs", "xr", "w", "s", "b"), ("h",), True),
}

METAL_TYPE = {mx.bfloat16: "bfloat16_t", mx.float16: "float16_t", mx.float32: "float", mx.uint8: "uint8_t",
              mx.uint32: "uint32_t"}


@lru_cache(maxsize=None)
def header() -> str:
    return (DIR / "attn_common.metal").read_text()


@lru_cache(maxsize=None)
def body(name: str) -> str:
    return (DIR / f"attn_{name}.metal").read_text()


def prefix(types: Tuple[Tuple[str, str], ...], ints: Tuple[Tuple[str, int], ...],
           floats: Tuple[Tuple[str, float], ...] = ()) -> str:
    """The constant lines a host prepends to a body (``types``: Metal type names)."""
    out = "".join(f"  using {n} = {t};\n" for n, t in types)
    out += "".join(f"  constexpr int {n} = {int(v)};\n" for n, v in ints)
    out += "".join(f"  constexpr float {n} = {float(v)!r}f;\n" for n, v in floats)
    return out


_KERNELS: dict = {}


def kernel(name: str, types: Tuple[Tuple[str, str], ...], ints: Tuple[Tuple[str, int], ...],
           floats: Tuple[Tuple[str, float], ...] = ()):
    """The compiled kernel for this body and constants (cached; MLX caches compiled kernels by name, so the name
    carries a hash of the full source)."""
    key = (name, types, ints, floats)
    kern = _KERNELS.get(key)
    if kern is None:
        inputs, outputs, contiguous = SPECS[name]
        source = prefix(types, ints, floats) + body(name)
        digest = hashlib.sha256((header() + source).encode()).hexdigest()[:16]
        kern = _KERNELS[key] = mx.fast.metal_kernel(
            name=f"vmlx_m3_fused_attn_{name}_{digest}", input_names=list(inputs), output_names=list(outputs),
            source=source, header=header(), ensure_row_contiguous=contiguous)
    return kern
