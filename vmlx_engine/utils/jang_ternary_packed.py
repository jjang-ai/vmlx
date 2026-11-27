"""Runtime bridge for JANG ternary bundles stored at PTQ1_0 density.

Storage ``ternary_packed_26b``: per 128-weight group, 26 bytes of base-3
packed trits (5 per byte) plus one float16 scale; no stored bias because a
ternary group's bias is always ``-scale``. At load time each declared module
is expanded losslessly into exactly the uint32 2-bit codes, float16 scales and
float16 biases of a plain affine 2-bit bundle, so the same MLX kernels run and
the output is identical. Only the bytes on disk change (1.75 bits/weight).
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import mlx.core as mx

STORAGE_NAME = "ternary_packed_26b"
GROUP = 128
BYTES_PER_GROUP = 26
RUNTIME_BITS = 2

_UNPACK_26B_SOURCE = r"""
  const uint idx = thread_position_in_grid.x;
  const uint total_groups = params[0];
  if (idx >= total_groups) return;
  const device uint8_t* src = packed + size_t(idx) * 26u;
  device uint32_t* dst = words + size_t(idx) * 8u;
  uint8_t ok = 1u;
  uint32_t cur_word = 0u;
  uint bit_pos = 0u;
  uint w_idx = 0u;
  for (uint bi = 0u; bi < 26u; bi++) {
    uint32_t v = uint32_t(src[bi]);
    const uint n_trits = (bi < 25u) ? 5u : 3u;
    if (bi < 25u) {
      if (v > 242u) ok = 0u;
    } else {
      if (v > 26u) ok = 0u;
    }
    for (uint k = 0u; k < n_trits; k++) {
      uint32_t trit = v % 3u;
      v /= 3u;
      cur_word |= (trit << bit_pos);
      bit_pos += 2u;
      if (bit_pos == 32u) {
        dst[w_idx++] = cur_word;
        cur_word = 0u;
        bit_pos = 0u;
      }
    }
  }
  valid[idx] = ok;
"""


@lru_cache(maxsize=1)
def _unpack_26b_kernel():
    return mx.fast.metal_kernel(
        name="vmlx_unpack_ternary_26b",
        input_names=["packed", "params"],
        output_names=["words", "valid"],
        source=_UNPACK_26B_SOURCE,
    )


def ternary_packed_modules(jang_config: Mapping[str, Any]) -> frozenset[str]:
    """Return module paths whose manifest entry declares packed-trit storage."""
    quantization = jang_config.get("quantization") if isinstance(jang_config, Mapping) else None
    manifest = quantization.get("tensor_quantization_manifest") if isinstance(quantization, Mapping) else None
    runtime = jang_config.get("runtime", {}) if isinstance(jang_config, Mapping) else {}
    required = isinstance(runtime, Mapping) and runtime.get("requires_jang_ternary_packed_expansion", False)
    manifest = manifest if isinstance(manifest, Mapping) else {}
    modules = frozenset(
        str(path)
        for path, spec in manifest.items()
        if isinstance(spec, Mapping) and spec.get("storage") == STORAGE_NAME
    )
    if required and not modules:
        raise ValueError("packed ternary runtime required but no storage modules declared")
    for path in modules:
        entry = manifest[path]
        if (entry.get("runtime_bits", entry.get("bits")) != RUNTIME_BITS
                or entry.get("group_size") != GROUP or entry.get("mode", "affine") != "affine"):
            raise ValueError(f"unsupported packed ternary quantization for {path}")
    return modules


def expand_ternary_packed_mlx(packed, scales):
    """uint8 [rows, groups*26] + float16 scales -> (uint32 2-bit words, scales, biases)."""
    if packed.dtype != mx.uint8 or packed.ndim != 2:
        raise ValueError(f"ternary_packed weight must be 2-D uint8, got {packed.dtype} {packed.shape}")
    rows, cols = packed.shape
    if not rows or not cols or cols % BYTES_PER_GROUP:
        raise ValueError(f"ternary_packed columns {cols} not divisible by {BYTES_PER_GROUP}")
    groups = cols // BYTES_PER_GROUP
    if tuple(scales.shape) != (rows, groups):
        raise ValueError(f"scales shape {scales.shape} != {(rows, groups)}")
    if scales.dtype != mx.float16:
        raise ValueError("packed ternary scales must be float16 (no implicit precision conversion)")
    total_groups = rows * groups
    packed_c = mx.contiguous(packed)
    kernel = _unpack_26b_kernel()
    tpg = 256
    grid_x = ((total_groups + tpg - 1) // tpg) * tpg
    words, valid = kernel(
        inputs=[packed_c, mx.array([total_groups], dtype=mx.uint32)],
        grid=(grid_x, 1, 1),
        threadgroup=(tpg, 1, 1),
        output_shapes=[(rows, groups * (GROUP // 16)), (total_groups,)],
        output_dtypes=[mx.uint32, mx.uint8],
    )
    if not (mx.all(valid == 1) & mx.all(mx.isfinite(scales))).item():
        raise ValueError("invalid packed ternary trits or non-finite scales")
    biases = -scales
    mx.eval(words, biases)
    return words, scales, biases


def expand_ternary_packed_shard_mlx(
    weights: Mapping[str, Any],
    storage_modules: frozenset[str] | set[str],
) -> tuple[dict[str, Any], int]:
    """Expand every declared module present in one loaded shard."""
    if not storage_modules:
        return dict(weights), 0
    expanded = dict(weights)
    count = 0
    for module_path in storage_modules:
        weight_key = f"{module_path}.weight"
        value = expanded.get(weight_key)
        if value is None:
            continue
        scales = expanded.get(f"{module_path}.scales")
        if scales is None:
            raise ValueError(f"ternary_packed module {module_path!r} is missing its scales")
        if f"{module_path}.biases" in expanded:
            raise ValueError(f"ternary_packed module {module_path!r} must not store biases")
        words, scales16, biases16 = expand_ternary_packed_mlx(value, scales)
        expanded[weight_key] = words
        expanded[f"{module_path}.scales"] = scales16
        expanded[f"{module_path}.biases"] = biases16
        count += 1
    return expanded, count
