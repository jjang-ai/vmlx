"""Row-exact multi-row MXFP8 / MXFP4 matrix-vector product for speculative verify.

Why: a verify forward runs 2..T token rows through every projection. For the
MX floating-point formats MLX's one-row ``quantized_matmul`` uses its single-
vector fp kernel and, from two rows, a different small-batch kernel whose
reduction order differs (measured on MLX 0.32.3: mxfp8 at 512 -> 1024 / 2048
-> 4096 differs from the one-row result on most rows), so verify rows would
not equal decode steps. Running each row as its own one-row call is exact but
reads the weight once per row.

How: one launch for up to four rows that walks the weight once. Every row runs
the per-row operation sequence of MLX's single-vector fp kernel (the same
``load_vector`` / ``qdot`` / ``dequantize_scale`` helpers, read from the
installed MLX headers at runtime, in the same k-block order and the same
``simd_sum``), so each row's result equals the one-row call. Every (shape,
rows, format, dtype) is admitted only after a bit-equality check against stock
one-row calls on the real weights; a shape that fails keeps the caller's
one-row path. Only MLX's fast single-vector traversal is transcribed (K a
multiple of the 256 / 512-value block, N a multiple of 8): other shapes return
None.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

import mlx.core as mx

from .row_exact_qmv import _admission_probes, _extract_template_function

logger = logging.getLogger(__name__)

MAX_ROWS = 4
MODES = {"mxfp8": (8, 32), "mxfp4": (4, 32)}
_ADMITTED: dict = {}
STATS = {"calls": 0, "admitted": 0, "rejected": 0}


def _read(name: str) -> str:
    return (Path(mx.__file__).resolve().parent / "include/mlx/backend/metal/kernels" / name).read_text()


@lru_cache(maxsize=1)
def _header() -> str | None:
    """MLX's fp8 / fp4 types and its fp single-vector helpers, as installed."""
    try:
        fp4, fp8, source = _read("fp4.h"), _read("fp8.h"), _read("fp_quantized.h")
    except Exception as exc:  # pragma: no cover - packaging dependent
        logger.warning("row-exact fp verify unavailable: MLX headers unreadable (%s)", exc)
        return None
    pieces = [fp4.replace("#pragma once", ""), fp8.replace("#pragma once", "")]
    for signature in (
        "inline constexpr short get_pack_factor()",
        "inline constexpr short get_bytes_per_pack()",
        "static inline T dequantize_scale(uint8_t s)",
        "struct Dequantize {",
        "inline void load_vector(const device T* x, thread U* x_thread)",
        "inline U qdot(const device uint8_t* w, const thread U* x_thread, U scale)",
    ):
        piece = _extract_template_function(source, signature)
        if piece is None:
            logger.warning("row-exact fp verify unavailable: %r not found in fp_quantized.h", signature)
            return None
        pieces.append(piece + (";" if signature.startswith("struct") else ""))
    impl = _extract_template_function(source, "METAL_FUNC void fp_qmv_fast_impl(")
    if impl is None or not _traversal_matches(impl):
        logger.warning("row-exact fp verify unavailable: MLX fp_qmv_fast traversal changed")
        return None
    # Weight bytes from a thread-local copy (fetched once for every row): same arithmetic, other address space.
    qdot = pieces[-1]
    pieces.append(qdot.replace("inline U qdot(", "inline U qdot_reg(").replace("const device", "const thread")
                  .replace("(const device uint16_t*)", "(const thread uint16_t*)"))
    return "#ifndef SIMD_SIZE\n#define SIMD_SIZE 32\n#endif\n" + "\n\n".join(pieces)


def _traversal_matches(impl: str) -> bool:
    compact = re.sub(r"\s+", " ", impl)
    needles = (
        "constexpr int packs_per_thread = 2;",
        "constexpr int num_simdgroups = 2;",
        "constexpr int pack_factor = get_pack_factor<32, bits>();",
        "constexpr int bytes_per_pack = get_bytes_per_pack<32>();",
        "constexpr int values_per_thread = pack_factor * packs_per_thread;",
        "load_vector<T, U, values_per_thread>(x, x_thread);",
        "U s = dequantize_scale<U, group_size>(sl[0]);",
        "result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s);",
        "result[row] = simd_sum(result[row]);",
        "y[row] = static_cast<T>(result[row]);",
    )
    return all(needle in compact for needle in needles) and "results_per_simdgroup = 4>" in compact


_SOURCE = r"""
  // MLX's single-vector fp kernel (fp_qmv_fast, no global scale), every x row r running its per-row sequence;
  // all rows share one threadgroup and one walk over the weight.
  constexpr int bits = BITS;
  constexpr int group_size = GS;
  constexpr int in_vec_size = K;
  constexpr int out_vec_size = N;
  constexpr int packs_per_thread = 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<32, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack<32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  constexpr int w_bytes = packs_per_thread * bytes_per_pack;
  typedef float U;

  uint3 tid = threadgroup_position_in_grid;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint simd_lid = thread_index_in_simdgroup;

  const device uint8_t* ws = (const device uint8_t*)w;
  const device uint8_t* sc = scales;

  thread U x_thread[ROWS][values_per_thread];
  thread U result[ROWS][results_per_simdgroup];
  for (int r = 0; r < ROWS; r++)
    for (int row = 0; row < results_per_simdgroup; row++) result[r][row] = 0;

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  sc += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* xs = x + simd_lid * values_per_thread;

  for (int k = 0; k < in_vec_size; k += block_size) {
    for (int r = 0; r < ROWS; r++) {
      load_vector<T, U, values_per_thread>(xs + r * in_vec_size, x_thread[r]);
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device uint8_t* sl = sc + row * in_vec_size_g;
      U s = dequantize_scale<U, group_size>(sl[0]);
#if REG
      thread uint32_t wbw[(w_bytes + 3) / 4];
      thread uint8_t* wb = (thread uint8_t*)wbw;
      for (int j = 0; j < w_bytes; j++) wb[j] = wl[j];
      for (int r = 0; r < ROWS; r++) {
        result[r][row] += qdot_reg<U, values_per_thread, bits>(wb, x_thread[r], s);
      }
#else
      for (int r = 0; r < ROWS; r++) {
        result[r][row] += qdot<U, values_per_thread, bits>(wl, x_thread[r], s);
      }
#endif
    }
    ws += block_size * bytes_per_pack / pack_factor;
    sc += block_size / group_size;
    xs += block_size;
  }

  for (int r = 0; r < ROWS; r++) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      U v = simd_sum(result[r][row]);
      if (simd_lid == 0) {
        y[r * out_vec_size + out_row + row] = static_cast<T>(v);
      }
    }
  }
"""


@lru_cache(maxsize=4)
def _kernel(reg: bool):
    header = _header()
    if header is None:
        return None
    return mx.fast.metal_kernel(
        name=f"vmlx_row_exact_verify_fp_qmv{'_reg' if reg else ''}",
        input_names=["w", "scales", "x"],
        output_names=["y"],
        header=header,
        source=f"#define REG {1 if reg else 0}\n" + _SOURCE,
    )


def _launch(x2d, weight, scales, bits, group_size, reg):
    rows, in_dim = x2d.shape
    out_dim = weight.shape[0]
    (y,) = _kernel(reg)(
        inputs=[weight, scales, x2d],
        template=[("T", x2d.dtype), ("BITS", bits), ("GS", group_size), ("K", in_dim), ("N", out_dim),
                  ("ROWS", rows)],
        grid=(32, 2 * ((out_dim + 7) // 8), 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(rows, out_dim)],
        output_dtypes=[x2d.dtype],
    )
    return y


def _admit(key, weight, scales, bits, group_size, mode, rows, in_dim, dtype):
    if key in _ADMITTED:
        return _ADMITTED[key]
    probes = _admission_probes(rows, in_dim, dtype)
    wants = [
        mx.concatenate([
            mx.quantized_matmul(probe[r:r + 1], weight, scales, None, transpose=True, group_size=group_size,
                                bits=bits, mode=mode)
            for r in range(rows)
        ], axis=0)
        for probe in probes
    ]
    chosen = None
    for reg in (True, False):
        if all(bool(mx.array_equal(_launch(p, weight, scales, bits, group_size, reg), want).item())
               for p, want in zip(probes, wants)):
            chosen = reg
            break
    _ADMITTED[key] = chosen
    STATS["admitted" if chosen is not None else "rejected"] += 1
    if chosen is None:
        logger.warning("row-exact fp verify REJECTED (not bit-equal to MLX one-row) rows=%d K=%d N=%d mode=%s "
                       "dtype=%s; one-row calls keep this shape", rows, in_dim, weight.shape[0], mode, dtype)
    else:
        logger.info("row-exact fp verify admitted rows=%d K=%d N=%d mode=%s dtype=%s reg=%s", rows, in_dim,
                    weight.shape[0], mode, dtype, chosen)
    return chosen


def row_exact_fp_qmv(x, weight, scales, *, group_size: int, bits: int, mode: str):
    """``x @ dequant(weight).T`` for an MX-format weight with every row bit-equal to its one-row call, or None
    (unsupported format / shape / dtype, headers unavailable, or the shape failed admission)."""
    if MODES.get(mode) != (int(bits), int(group_size)) or x.ndim < 2 or _header() is None:
        return None
    in_dim, out_dim = int(x.shape[-1]), int(weight.shape[0])
    rows = 1
    for dim in x.shape[:-1]:
        rows *= int(dim)
    block = (32 // int(bits)) * 2 * 32
    if not 2 <= rows <= MAX_ROWS or in_dim % block or out_dim % 8 or weight.dtype != mx.uint32:
        return None
    if x.dtype not in (mx.float16, mx.bfloat16, mx.float32) or scales.dtype != mx.uint8:
        return None
    key = (rows, in_dim, out_dim, mode, str(x.dtype))
    reg = _admit(key, weight, scales, int(bits), int(group_size), mode, rows, in_dim, x.dtype)
    if reg is None:
        return None
    STATS["calls"] += 1
    y = _launch(mx.contiguous(x.reshape(rows, in_dim)), weight, scales, int(bits), int(group_size), reg)
    return y.reshape(*x.shape[:-1], out_dim)
