"""Row-exact multi-row affine QMV for native-MTP verification.

Why: a native-MTP verify forward feeds 2..8 token rows through every
projection at once. MLX ``quantized_matmul`` switches from its single-row
``qmv_fast`` kernel to a different multi-row kernel for M >= 2, whose
reduction order differs, so every verify row's logits differ from the logits
the same token gets in an ordinary single-token decode step (measured on
Qwen3.8 Flash-Next JANGH4, MLX 0.32.3: tens of elements per projection row,
~1.3k of 248k lm_head logits at M=4). Greedy native MTP therefore did not
reproduce AR output byte for byte. TensorFold, mlx-serve and oMLX all keep
each verify row bit-identical to its solo decode call.

How: this kernel is MLX's ``qmv_fast_impl`` with every x row handled inside
the same threadgroup. Each row runs exactly the per-row operation sequence of
``qmv_fast`` (same ``load_vector``/``qdot`` calls, same k-block order, same
``simd_sum``), so its result equals the M=1 call bit for bit, while the
weight block is fetched once for all rows. ``load_vector`` and ``qdot`` are
taken verbatim from the installed MLX headers at runtime, so the arithmetic
follows the exact MLX build. Every (shape, bits, group, rows, dtype) is
admitted only after a bit-equality self-check against stock M=1 calls on the
real weights; a failed check keeps the stock path for that shape and logs it.
"""

from __future__ import annotations

import contextvars
import logging
import os
import re
from functools import lru_cache
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger(__name__)

MAX_ROWS = 8
_ADMITTED: dict[tuple, str | None] = {}
STATS = {"calls": 0, "admitted": 0, "rejected": 0}


# Families where exactness is NOT the default. Measured 2026-10-04 on the dense
# Qwen3.8-27B (JANG_4D, served, novel prompts, AR-bracketed): row-exact verify
# cost Adaptive 15-20 % (prose 31.6 vs 37.4 tok/s stock, code 37.9 vs 47.3);
# MLX's single-row qmv_fast is bandwidth-bound there, so replaying its per-row
# arithmetic for 3-4 verify rows costs 1.3-1.6x the stock multi-row kernel. On
# the MoE Flash-Next family the experts dominate and exactness was ~free
# (JANGH4 prose +1.4 %, code -3.9 %).
_DENSE_SPEED_FIRST = frozenset({"qwen3_5", "qwen3_5_text", "qwen3_6", "qwen3_6_text"})
_ACTIVE_FAMILY = {"value": None}


_DENSE_ROW_INVARIANT = {"on": False}


def dense_row_invariant_active() -> bool:
    return _DENSE_ROW_INVARIANT["on"]


def set_row_exact_family(model_type) -> None:
    """Record the loaded model family (the generator calls this once at load).

    Where row-exact verification is on, small dense projections (router,
    gates, hyper-connections) switch to the row-invariant form EVERYWHERE,
    including ordinary AR decode, so AR steps and verify rows run the same
    per-row arithmetic by construction (see ``_dense_rows``).
    """
    _ACTIVE_FAMILY["value"] = str(model_type or "") or None
    _DENSE_ROW_INVARIANT["on"] = bool(model_type) and row_exact_qmv_requested()
    if _DENSE_ROW_INVARIANT["on"]:
        _install()


def row_exact_qmv_requested() -> bool:
    """``VMLX_ROW_EXACT_VERIFY_QMV`` = 1/0 forces it; otherwise on except dense Qwen3.5/3.8."""
    raw = os.environ.get("VMLX_ROW_EXACT_VERIFY_QMV", "").strip().lower()
    if raw:
        return raw not in {"0", "false", "no", "off"}
    return _ACTIVE_FAMILY["value"] not in _DENSE_SPEED_FIRST


def _extract_template_function(source: str, signature: str) -> str | None:
    at = source.find(signature)
    if at < 0:
        return None
    start = source.rfind("template", 0, at)
    open_brace = source.find("{", at)
    if start < 0 or open_brace < 0:
        return None
    depth = 0
    for index in range(open_brace, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    return None


@lru_cache(maxsize=1)
def _mlx_qmv_header() -> str | None:
    """MLX's own load_vector/qdot/pack helpers from the installed headers."""
    try:
        package = Path(mx.__file__).resolve().parent
        path = package / "include/mlx/backend/metal/kernels/quantized.h"
        source = path.read_text()
    except Exception as exc:  # pragma: no cover - packaging dependent
        logger.warning("row-exact verify QMV unavailable: MLX header unreadable (%s)", exc)
        return None
    pieces = []
    for signature in (
        "inline constexpr short get_pack_factor()",
        "inline constexpr short get_bytes_per_pack()",
        "inline U load_vector(const device T* x, thread U* x_thread)",
        "inline U qdot(\n    const device uint8_t* w,",
        "inline U load_vector_safe(const device T* x, thread U* x_thread, int N)",
        "inline U qdot_safe(",
    ):
        piece = _extract_template_function(source, signature)
        if piece is None:
            logger.warning("row-exact verify QMV unavailable: %r not found in %s", signature, path)
            return None
        pieces.append(piece)
    # qmv_fast_impl itself must still have the traversal this kernel copies.
    impl = _extract_template_function(source, "METAL_FUNC void qmv_fast_impl(")
    generic = _extract_template_function(source, "METAL_FUNC void qmv_impl(")
    if impl is None or not _traversal_matches(impl) or generic is None or not _generic_traversal_matches(generic):
        logger.warning("row-exact verify QMV unavailable: MLX qmv traversal changed")
        return None
    # Register copies: identical arithmetic, weights read from a thread-local
    # copy so each weight byte is fetched once for all verify rows.
    for piece in list(pieces[2:]):
        if "inline U qdot" in piece:
            pieces.append(
                piece.replace("inline U qdot_safe(", "inline U qdot_safe_reg(")
                .replace("inline U qdot(", "inline U qdot_reg(")
                .replace("device", "thread")
            )
    pieces.append(_QDOT_ROWS)
    return "#ifndef SIMD_SIZE\n#define SIMD_SIZE 32\n#endif\n" + "\n\n".join(pieces)


def _traversal_matches(impl: str) -> bool:
    compact = re.sub(r"\s+", " ", impl)
    # MLX 0.32.4 generalized `bits == 2 ? 1 : 2` to `bits <= 2 ? 1 : 2` (1-bit support); identical for
    # the 2..8-bit weights this kernel admits. Strip the trailing comment it carries before matching.
    compact = re.sub(r"//[^;{}]*?(?= constexpr)", "", compact)
    if "constexpr int packs_per_thread = bits <= 2 ? 1 : 2;" in compact:
        compact = compact.replace("constexpr int packs_per_thread = bits <= 2 ? 1 : 2;",
                                  "constexpr int packs_per_thread = bits == 2 ? 1 : 2;")
    needles = (
        "constexpr int packs_per_thread = bits == 2 ? 1 : 2;",
        "constexpr int num_simdgroups = 2;",
        "constexpr int results_per_simdgroup = 4;",
        "constexpr int pack_factor = get_pack_factor<bits, 32>();",
        "constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();",
        "U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);",
        "result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);",
        "result[row] = simd_sum(result[row]);",
        "y[row] = static_cast<T>(result[row]);",
    )
    return all(needle in compact for needle in needles)


def _generic_traversal_matches(impl: str) -> bool:
    compact = re.sub(r"\s+", " ", impl)
    needles = (
        "constexpr int packs_per_thread = 1;",
        "const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);",
        "for (; k < in_vec_size - block_size; k += block_size) {",
        "U sum = load_vector_safe<T, U, values_per_thread, bits>( x, x_thread, remaining);",
        "result[row] += qdot_safe<U, values_per_thread, bits>( wl, x_thread, s, b, sum, remaining);",
    )
    return all(needle in compact for needle in needles)


# MLX qmv_impl (the M=1 kernel when K is not a qmv_fast multiple). Both of its
# branches: N >= 8 moves the last tile back and recomputes identical outputs;
# N < 8 keeps out_row and guards every output row. Every x row r runs the
# identical per-row sequence.
_QDOT_ROWS = r"""
// qdot for ROWS activation rows sharing one register copy of the weights.
// Each integer weight field is converted once; every row then evaluates the
// exact term and accumulation shape of MLX qdot (same products, same order).
template <typename U, int values_per_thread, int bits, int ROWS>
inline void qdot_rows(
    const thread uint8_t* w,
    const thread U* x,          // [ROWS][values_per_thread], row-major
    U scale,
    U bias,
    const thread U* sum,        // [ROWS]
    thread U* out) {            // [ROWS] receives scale * accum + sum * bias
  U accum[ROWS];
  for (int r = 0; r < ROWS; r++) accum[r] = 0;
  if (bits == 2) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      U w0 = (w[i] & 0x03), w1 = (w[i] & 0x0c), w2 = (w[i] & 0x30), w3 = (w[i] & 0xc0);
      for (int r = 0; r < ROWS; r++) {
        const thread U* xt = x + r * values_per_thread;
        accum[r] += (xt[4 * i] * w0 + xt[4 * i + 1] * w1 + xt[4 * i + 2] * w2 + xt[4 * i + 3] * w3);
      }
    }
  } else if (bits == 3) {
    int xo = 0, wo = 0;
    for (int i = 0; i < (values_per_thread / 8); i++) {
      xo += 8 * i; wo += 3 * i;
      const thread uint8_t* ww = w + wo;
      U a0 = (ww[0] & 0x07), a1 = (ww[0] & 0x38), a2 = (ww[0] & 0xc0), a3 = (ww[1] & 0x01);
      U a4 = (ww[1] & 0x0e), a5 = (ww[1] & 0x70), a6 = (ww[1] & 0x80), a7 = (ww[2] & 0x03);
      U a8 = (ww[2] & 0x1c), a9 = (ww[2] & 0xe0);
      for (int r = 0; r < ROWS; r++) {
        const thread U* xt = x + r * values_per_thread + xo;
        U acc = accum[r];
        acc += a0 * xt[0]; acc += a1 * xt[1]; acc += a2 * xt[2]; acc += a3 * (xt[2] * 256.0f);
        acc += a4 * xt[3]; acc += a5 * xt[4]; acc += a6 * xt[5]; acc += a7 * (xt[5] * 256.0f);
        acc += a8 * xt[6]; acc += a9 * xt[7];
        accum[r] = acc;
      }
    }
  } else if (bits == 4) {
    const thread uint16_t* ws = (const thread uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      U w0 = (ws[i] & 0x000f), w1 = (ws[i] & 0x00f0), w2 = (ws[i] & 0x0f00), w3 = (ws[i] & 0xf000);
      for (int r = 0; r < ROWS; r++) {
        const thread U* xt = x + r * values_per_thread;
        accum[r] += (xt[4 * i] * w0 + xt[4 * i + 1] * w1 + xt[4 * i + 2] * w2 + xt[4 * i + 3] * w3);
      }
    }
  } else if (bits == 5) {
    int xo = 0, wo = 0;
    for (int i = 0; i < (values_per_thread / 8); i++) {
      xo += 8 * i; wo += 5 * i;
      const thread uint8_t* ww = w + wo;
      U a0 = (ww[0] & 0x1f), a1 = (ww[0] & 0xe0), a2 = (ww[1] & 0x3), a3 = (ww[1] & 0x7c);
      U a4 = (ww[1] & 0x80), a5 = (ww[2] & 0xf), a6 = (ww[2] & 0xf0), a7 = (ww[3] & 0x1);
      U a8 = (ww[3] & 0x3e), a9 = (ww[3] & 0xc0), a10 = (ww[4] & 0x7), a11 = (ww[4] & 0xf8);
      for (int r = 0; r < ROWS; r++) {
        const thread U* xt = x + r * values_per_thread + xo;
        U acc = accum[r];
        acc += a0 * xt[0]; acc += a1 * xt[1]; acc += a2 * (xt[1] * 256.0f); acc += a3 * xt[2];
        acc += a4 * xt[3]; acc += a5 * (xt[3] * 256.0f); acc += a6 * xt[4]; acc += a7 * (xt[4] * 256.0f);
        acc += a8 * xt[5]; acc += a9 * xt[6]; acc += a10 * (xt[6] * 256.0f); acc += a11 * xt[7];
        accum[r] = acc;
      }
    }
  } else if (bits == 6) {
    int xo = 0, wo = 0;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      xo += 4 * i; wo += 3 * i;
      const thread uint8_t* ww = w + wo;
      U a0 = (ww[0] & 0x3f), a1 = (ww[0] & 0xc0), a2 = (ww[1] & 0x0f);
      U a3 = (ww[1] & 0xf0), a4 = (ww[2] & 0x03), a5 = (ww[2] & 0xfc);
      for (int r = 0; r < ROWS; r++) {
        const thread U* xt = x + r * values_per_thread + xo;
        U acc = accum[r];
        acc += a0 * xt[0];
        acc += a1 * xt[1]; acc += a2 * (xt[1] * 256.0f);
        acc += a3 * xt[2]; acc += a4 * (xt[2] * 256.0f);
        acc += a5 * xt[3];
        accum[r] = acc;
      }
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      U wi = w[i];
      for (int r = 0; r < ROWS; r++) accum[r] += x[r * values_per_thread + i] * wi;
    }
  }
  for (int r = 0; r < ROWS; r++) out[r] = scale * accum[r] + sum[r] * bias;
}
"""


_SOURCE_GENERIC = r"""
  constexpr int bits = BITS;
  constexpr int group_size = GS;
  constexpr int in_vec_size = K;
  constexpr int out_vec_size = N;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 1;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  typedef float U;

  uint3 tid = threadgroup_position_in_grid;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint simd_lid = thread_index_in_simdgroup;

  const device uint8_t* ws = (const device uint8_t*)w;
  const device T* sc = scales;
  const device T* bi = biases;
  thread U x_thread[ROWS][values_per_thread];
  thread U result[ROWS][results_per_simdgroup];
  for (int r = 0; r < ROWS; r++)
    for (int row = 0; row < results_per_simdgroup; row++) result[r][row] = 0;

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  constexpr bool small_n = out_vec_size < (num_simdgroups * results_per_simdgroup);
  const int used_out_row = small_n ? out_row : min(out_vec_size - results_per_simdgroup, out_row);
  if (out_row >= out_vec_size) {
    return;
  }
  ws += used_out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  sc += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  bi += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* xs = x + (BATCH ? tid.x * in_vec_size : 0) + simd_lid * values_per_thread;

  int k = 0;
  for (; k < in_vec_size - block_size; k += block_size) {
    U sum[ROWS];
    for (int r = 0; r < ROWS; r++) {
      sum[r] = load_vector<T, U, values_per_thread, bits>(xs + r * in_vec_size, x_thread[r]);
    }
    for (int row = 0; row < results_per_simdgroup && (!small_n || out_row + row < out_vec_size); row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = sc + row * in_vec_size_g;
      const device T* bl = bi + row * in_vec_size_g;
      U s = sl[0];
      U b = bl[0];
#if MULTI
      thread uint32_t wbw[(packs_per_thread * bytes_per_pack + 3) / 4];
      thread uint8_t* wb = (thread uint8_t*)wbw;
      for (int j = 0; j < packs_per_thread * bytes_per_pack; j++) wb[j] = wl[j];
      U part[ROWS];
      qdot_rows<U, values_per_thread, bits, ROWS>(wb, &x_thread[0][0], s, b, sum, part);
      for (int r = 0; r < ROWS; r++) result[r][row] += part[r];
#elif REG
      thread uint32_t wbw[(packs_per_thread * bytes_per_pack + 3) / 4];
      thread uint8_t* wb = (thread uint8_t*)wbw;
      for (int j = 0; j < packs_per_thread * bytes_per_pack; j++) wb[j] = wl[j];
      for (int r = 0; r < ROWS; r++) {
        result[r][row] += qdot_reg<U, values_per_thread, bits>(wb, x_thread[r], s, b, sum[r]);
      }
#else
      for (int r = 0; r < ROWS; r++) {
        result[r][row] += qdot<U, values_per_thread, bits>(wl, x_thread[r], s, b, sum[r]);
      }
#endif
    }
    ws += block_size * bytes_per_pack / pack_factor;
    sc += block_size / group_size;
    bi += block_size / group_size;
    xs += block_size;
  }
  const int remaining = clamp(
      static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
      0,
      values_per_thread);
  if (remaining > 0) {
    U sum[ROWS];
    for (int r = 0; r < ROWS; r++) {
      sum[r] = load_vector_safe<T, U, values_per_thread, bits>(xs + r * in_vec_size, x_thread[r], remaining);
    }
    for (int row = 0; row < results_per_simdgroup && (!small_n || out_row + row < out_vec_size); row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = sc + row * in_vec_size_g;
      const device T* bl = bi + row * in_vec_size_g;
      U s = sl[0];
      U b = bl[0];
#if REG
      thread uint32_t wbw[(packs_per_thread * bytes_per_pack + 3) / 4];
      thread uint8_t* wb = (thread uint8_t*)wbw;
      for (int j = 0; j < packs_per_thread * bytes_per_pack; j++) wb[j] = wl[j];
      for (int r = 0; r < ROWS; r++) {
        result[r][row] += qdot_safe_reg<U, values_per_thread, bits>(wb, x_thread[r], s, b, sum[r], remaining);
      }
#else
      for (int r = 0; r < ROWS; r++) {
        result[r][row] += qdot_safe<U, values_per_thread, bits>(wl, x_thread[r], s, b, sum[r], remaining);
      }
#endif
    }
  }
  for (int r = 0; r < ROWS; r++) {
    for (int row = 0; row < results_per_simdgroup && (!small_n || out_row + row < out_vec_size); row++) {
      U v = simd_sum(result[r][row]);
      if (simd_lid == 0) {
        y[(BATCH ? tid.x : r) * out_vec_size + used_out_row + row] = static_cast<T>(v);
      }
    }
  }
"""


# MLX qmv_quad_impl (the M=1 kernel for K == 64 or 128): one simdgroup per
# threadgroup, eight 4-lane quads, eight output rows per quad. Quad indices are
# derived from the simd lane exactly as Metal defines them for 32-thread groups.
_SOURCE_QUAD = r"""
  constexpr int bits = BITS;
  constexpr int group_size = GS;
  constexpr int in_vec_size = K;
  constexpr int out_vec_size = N;
  constexpr int QUAD = 4;
  constexpr int quads_per_simd = SIMD_SIZE / QUAD;
  constexpr int pack_factor = 32 / bits;
  constexpr int values_per_thread = K / QUAD;
  constexpr int packs_per_thread = values_per_thread / pack_factor;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  constexpr int results_per_quadgroup = 8;
  typedef float U;

  uint3 tid = threadgroup_position_in_grid;
  uint lane = thread_index_in_simdgroup;
  uint quad_gid = lane / QUAD;
  uint quad_lid = lane % QUAD;

  thread U x_thread[ROWS][values_per_thread];
  thread U result[ROWS][results_per_quadgroup];
  for (int r = 0; r < ROWS; r++)
    for (int row = 0; row < results_per_quadgroup; row++) result[r][row] = 0;

  const int in_vec_size_w = in_vec_size / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * quads_per_simd * results_per_quadgroup + quad_gid;

  const device uint32_t* wp = w + out_row * in_vec_size_w + quad_lid * packs_per_thread;
  const device T* sc = scales + out_row * in_vec_size_g + quad_lid / scale_step_per_thread;
  const device T* bi = biases + out_row * in_vec_size_g + quad_lid / scale_step_per_thread;

  U sum[ROWS];
  for (int r = 0; r < ROWS; r++) {
    sum[r] = load_vector<T, U, values_per_thread, bits>(
        x + r * in_vec_size + quad_lid * values_per_thread, x_thread[r]);
  }
  for (int row = 0; row < results_per_quadgroup; row++) {
    auto wl = (const device uint8_t*)(wp + row * in_vec_size_w * quads_per_simd);
    const device T* sl = sc + row * in_vec_size_g * quads_per_simd;
    const device T* bl = bi + row * in_vec_size_g * quads_per_simd;
    U s = sl[0];
    U b = bl[0];
    if (row * quads_per_simd + out_row < out_vec_size) {
      for (int r = 0; r < ROWS; r++) {
        result[r][row] += qdot<U, values_per_thread, bits>(wl, x_thread[r], s, b, sum[r]);
      }
    }
  }
  for (int r = 0; r < ROWS; r++) {
    for (int row = 0; row < results_per_quadgroup; row++) {
      U v = quad_sum(result[r][row]);
      if (quad_lid == 0 && row * quads_per_simd + out_row < out_vec_size) {
        y[r * out_vec_size + out_row + row * quads_per_simd] = static_cast<T>(v);
      }
    }
  }
"""


_SOURCE = r"""
  // MLX qmv_fast_impl, transcribed; every x row r runs the identical per-row
  // sequence, all rows share one threadgroup and one weight walk.
  constexpr int bits = BITS;
  constexpr int group_size = GS;
  constexpr int in_vec_size = K;
  constexpr int out_vec_size = N;
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  typedef float U;

  uint3 tid = threadgroup_position_in_grid;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint simd_lid = thread_index_in_simdgroup;

  const device uint8_t* ws = (const device uint8_t*)w;
  const device T* sc = scales;
  const device T* bi = biases;

  thread U x_thread[ROWS][values_per_thread];
  thread U result[ROWS][results_per_simdgroup];
  for (int r = 0; r < ROWS; r++)
    for (int row = 0; row < results_per_simdgroup; row++) result[r][row] = 0;

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  sc += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  bi += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* xs = x + (BATCH ? tid.x * in_vec_size : 0) + simd_lid * values_per_thread;

  for (int k = 0; k < in_vec_size; k += block_size) {
#if ROWOUTER
    // Weights, scales and biases of this block's 4 output rows in registers
    // once; activation rows one at a time (one row's x_thread live). Per
    // (row, output row) the qdot sequence and block order equal qmv_fast.
    thread uint32_t wrow[results_per_simdgroup][(packs_per_thread * bytes_per_pack + 3) / 4];
    U srow[results_per_simdgroup], brow[results_per_simdgroup];
    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      thread uint8_t* wb = (thread uint8_t*)wrow[row];
      for (int j = 0; j < packs_per_thread * bytes_per_pack; j++) wb[j] = wl[j];
      srow[row] = (sc + row * in_vec_size_g)[0];
      brow[row] = (bi + row * in_vec_size_g)[0];
    }
    for (int r = 0; r < ROWS; r++) {
      U xr[values_per_thread];
      U sr = load_vector<T, U, values_per_thread, bits>(xs + r * in_vec_size, xr);
      for (int row = 0; row < results_per_simdgroup; row++) {
        result[r][row] += qdot_reg<U, values_per_thread, bits>(
            (thread uint8_t*)wrow[row], xr, srow[row], brow[row], sr);
      }
    }
    ws += block_size * bytes_per_pack / pack_factor;
    sc += block_size / group_size;
    bi += block_size / group_size;
    xs += block_size;
    continue;
#endif
    U sum[ROWS];
    for (int r = 0; r < ROWS; r++) {
      sum[r] = load_vector<T, U, values_per_thread, bits>(xs + r * in_vec_size, x_thread[r]);
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = sc + row * in_vec_size_g;
      const device T* bl = bi + row * in_vec_size_g;
      U s = sl[0];
      U b = bl[0];
#if MULTI
      thread uint32_t wbw[(packs_per_thread * bytes_per_pack + 3) / 4];
      thread uint8_t* wb = (thread uint8_t*)wbw;
      for (int j = 0; j < packs_per_thread * bytes_per_pack; j++) wb[j] = wl[j];
      U part[ROWS];
      qdot_rows<U, values_per_thread, bits, ROWS>(wb, &x_thread[0][0], s, b, sum, part);
      for (int r = 0; r < ROWS; r++) result[r][row] += part[r];
#elif REG
      thread uint32_t wbw[(packs_per_thread * bytes_per_pack + 3) / 4];
      thread uint8_t* wb = (thread uint8_t*)wbw;
      for (int j = 0; j < packs_per_thread * bytes_per_pack; j++) wb[j] = wl[j];
      for (int r = 0; r < ROWS; r++) {
        result[r][row] += qdot_reg<U, values_per_thread, bits>(wb, x_thread[r], s, b, sum[r]);
      }
#else
      for (int r = 0; r < ROWS; r++) {
        result[r][row] += qdot<U, values_per_thread, bits>(wl, x_thread[r], s, b, sum[r]);
      }
#endif
    }
    ws += block_size * bytes_per_pack / pack_factor;
    sc += block_size / group_size;
    bi += block_size / group_size;
    xs += block_size;
  }

  for (int r = 0; r < ROWS; r++) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      U v = simd_sum(result[r][row]);
      if (simd_lid == 0) {
        y[(BATCH ? tid.x : r) * out_vec_size + out_row + row] = static_cast<T>(v);
      }
    }
  }
"""


# Env switches read ONCE per verify scope (row_exact_verify_scope sets them; None = read live). A verify forward
# makes ~340 projection calls on the MiniCPM-V-4.7 35B; reading two env vars per call and rebuilding the variant
# tuple was ~2 ms of the 3.5-6.3 ms Python graph build that the GPU waits on after a DSpark round's sync.
# A ContextVar like _SCOPE, so one thread's scope never leaks its snapshot into another thread's reads.
_SCOPE_ENV: contextvars.ContextVar = contextvars.ContextVar("row_exact_qmv_scope_env", default=None)


def _env_order() -> str:
    v = _SCOPE_ENV.get()
    return os.environ.get("VMLX_ROW_EXACT_QMV_ORDER", "multi,plain") if v is None else v[0]


def _env_reg() -> str:
    v = _SCOPE_ENV.get()
    return os.environ.get("VMLX_ROW_EXACT_QMV_REG", "").strip() if v is None else v[1]


def _register_weights(rows: int) -> bool:
    """Read each weight block into registers once for all rows.

    Measured on Flash-Next JANGH4 (in-process verify ladder, 2 interleaved
    process pairs): registers win from three rows (S3 30.24 -> 29.42 ms,
    S4 34.0 -> 33.3 ms) and lose slightly at two (25.46 -> 25.74 ms).
    ``VMLX_ROW_EXACT_QMV_REG=0/1`` pins either choice.
    """
    pinned = _env_reg()
    if pinned in {"0", "1"}:
        return pinned == "1"
    return rows >= 3


@lru_cache(maxsize=16)
def _kernel(variant: str = "fast", reg: bool | None = None):
    header = _mlx_qmv_header()
    if header is None:
        return None
    rowouter = variant.endswith("_rowouter")
    variant = variant[:-9] if rowouter else variant
    batch = variant.endswith("_batch")
    variant = variant[:-6] if batch else variant
    multi = variant.endswith("_multi")
    variant = variant[:-6] if multi else variant
    reg = bool(reg)
    source = {"fast": _SOURCE, "generic": _SOURCE_GENERIC, "quad": _SOURCE_QUAD}[variant]
    return mx.fast.metal_kernel(
        name=(f"vmlx_row_exact_verify_qmv_{variant}{'_multi' if multi else ''}{'_batch' if batch else ''}"
              f"{'_rowouter' if rowouter else ''}{'_reg' if reg else ''}"),
        input_names=["w", "scales", "biases", "x"],
        output_names=["y"],
        header=header,
        source=(f"#define REG {1 if reg else 0}\n#define MULTI {1 if multi else 0}\n"
                f"#define BATCH {1 if batch else 0}\n#define ROWOUTER {1 if rowouter else 0}\n" + source),
    )


def _values_per_thread(bits: int) -> int:
    pack_factor = 8 if bits in (3, 5) else (4 if bits == 6 else 32 // bits)
    return pack_factor * (1 if bits == 2 else 2)


def _max_rows(bits: int) -> int:
    # Rows live in registers; packs wider than 8 values per thread spill past
    # four rows (measured 10x slower at 8 rows for 2/3/4/5-bit).
    return MAX_ROWS if bits in (6, 8) else 4


def _variants(rows, in_dim, out_dim, bits, group_size, dtype) -> tuple[str, ...]:
    """Candidate transcriptions in admission order (the self-check decides)."""
    return _variants_for(rows, in_dim, out_dim, bits, group_size, dtype, _env_order())


@lru_cache(maxsize=1024)
def _variants_for(rows, in_dim, out_dim, bits, group_size, dtype, order_env: str) -> tuple[str, ...]:
    if bits not in (2, 3, 4, 5, 6, 8) or group_size not in (32, 64, 128):
        return ()
    if dtype not in (mx.float16, mx.bfloat16, mx.float32) or not 2 <= rows <= _max_rows(bits):
        return ()
    if in_dim % group_size:
        return ()
    fast_ok = (in_dim % (_values_per_thread(bits) * 32) == 0 and out_dim % 8 == 0
               and group_size % _values_per_thread(bits) == 0)
    quad_ok = (in_dim in (64, 128) and bits in (2, 4, 8)
               and group_size % (in_dim // 4) == 0 and rows * in_dim // 4 <= 128)
    variants = ("quad",) if quad_ok else ()
    # Only the transcription of the kernel MLX itself dispatches at M=1:
    # qmv_fast exactly when ``fast_ok`` (measured on MLX 0.32.3 for 4/6/8-bit
    # at K = 256..3584, including K % 512 == 256 at 6/8-bit), qmv_impl
    # otherwise. The other family shares most of the arithmetic and matches
    # some probes by coincidence (1..12 of 12 per shape); admitted that way
    # it was inexact on real rows (8-bit K=1024 N=3584 at 6 rows: 45 of 49
    # calls differed from M=1).
    base = ("fast",) if fast_ok else ("generic",)
    # Admission order. Measured on Qwen3.8-27B shapes (M5 Max, 4 rows): the
    # shared-unpack "multi" walk is best or tied for 4/5/6/8-bit; the batched
    # grid ("batch", MLX's own single-row kernel per row) is slowest and is
    # only reachable explicitly via VMLX_ROW_EXACT_QMV_ORDER.
    order = order_env.split(",")
    ranked = []
    for kind in order:
        kind = kind.strip()
        if kind == "rowouter":
            ranked += [v + "_rowouter" for v in base if v == "fast"]
        elif kind == "batch":
            ranked += [v + "_batch" for v in base]
        elif kind == "multi":
            ranked += [v + "_multi" for v in base]
        elif kind == "plain":
            ranked += list(base)
    base = tuple(ranked)
    return variants + base


def _device_sized(a):
    # metal_kernel binds inputs smaller than 8 elements as `constant` buffers;
    # the transcribed MLX code addresses `device` memory. Zero padding at the
    # end keeps every flat index and therefore every result unchanged.
    if a.size >= 8:
        return a
    return mx.concatenate([a.reshape(-1), mx.zeros((8,), dtype=a.dtype)])


def _launch(x2d, weight, scales, biases, bits, group_size, variant="fast"):
    rows, in_dim = x2d.shape
    out_dim = weight.shape[0]
    weight, scales, biases = (_device_sized(a) for a in (weight, scales, biases))
    batch = variant.endswith("_batch")
    (y,) = _kernel(variant, False if batch else _register_weights(rows))(
        inputs=[weight, scales, biases, x2d],
        template=[("T", x2d.dtype), ("BITS", bits), ("GS", group_size),
                  ("K", in_dim), ("N", out_dim), ("ROWS", 1 if batch else rows)],
        grid=((32, (out_dim + 63) // 64, 1) if variant == "quad"
              else (32 * (rows if batch else 1), 2 * ((out_dim + 7) // 8), 1)),
        threadgroup=(32, 1, 1) if variant == "quad" else (32, 2, 1),
        output_shapes=[(rows, out_dim)],
        output_dtypes=[x2d.dtype],
    )
    return y


def _admission_probes(rows: int, in_dim: int, dtype) -> list:
    """Zero-mean rows plus rows shaped like real activations.

    Real hidden rows carry a large mean (so qdot's ``sum * bias`` term is
    large) and outlier channels; a zero-mean probe leaves that term near zero
    and hides rounding differences there.
    """
    zero_mean = mx.random.normal((rows, in_dim), key=mx.random.key(1234)) * 0.5
    shift = mx.random.uniform(-3.0, 3.0, (rows, 1), key=mx.random.key(1235))
    spikes = (mx.random.uniform(shape=(rows, in_dim), key=mx.random.key(1236)) < 0.01) * 30.0
    return [zero_mean.astype(dtype), (zero_mean + shift + spikes).astype(dtype)]


def _admit(key, weight, scales, biases, bits, group_size, rows, in_dim, dtype, variants,
           ref_scales=None, ref_biases=None):
    if key in _ADMITTED:
        return _ADMITTED[key]
    ref_scales = scales if ref_scales is None else ref_scales
    ref_biases = biases if ref_biases is None else ref_biases
    probes = _admission_probes(rows, in_dim, dtype)
    wants = [
        mx.concatenate([
            mx.quantized_matmul(probe[r:r + 1], weight, ref_scales, ref_biases, transpose=True,
                                group_size=group_size, bits=bits, mode="affine")
            for r in range(rows)
        ], axis=0)
        for probe in probes
    ]
    chosen = None
    for variant in variants:
        if all(
            bool(mx.array_equal(_launch(probe, weight, scales, biases, bits, group_size, variant), want).item())
            for probe, want in zip(probes, wants)
        ):
            chosen = variant
            break
    _ADMITTED[key] = chosen
    STATS["admitted" if chosen else "rejected"] += 1
    if chosen:
        logger.info("row-exact verify QMV admitted variant=%s rows=%d K=%d N=%d bits=%d gs=%d dtype=%s",
                    chosen, rows, in_dim, weight.shape[0], bits, group_size, dtype)
    else:
        logger.warning("row-exact verify QMV REJECTED (no variant bit-equal to MLX M=1) rows=%d K=%d "
                       "N=%d bits=%d gs=%d dtype=%s; stock MLX keeps this shape", rows, in_dim,
                       weight.shape[0], bits, group_size, dtype)
    return chosen


def _efficient_rows(bits: int) -> int:
    """Widest single launch before the multi-row walk spills registers.

    Measured on M5 Max, MLX 0.32.3, 8-bit g64, Flash-Next dense shapes
    (36 chained GDN qkv 2560->10240): R1 1.91, R4 2.15, R6 2.58 ms, then
    R7 12.76 and R8 14.55 ms (5x: register spill). lm_head 2560->248320:
    R6 1.68, R7 5.41 ms. 2-5-bit already cap at 4 rows (_max_rows).
    """
    return 6 if bits == 8 else 4


def _row_blocks(x, rows, in_dim, out_dim, weight, scales, biases, group_size, bits, mode):
    """Run a window wider than ``_efficient_rows`` as consecutive row blocks.

    Every row's result is computed by the same per-row arithmetic whatever
    block it lands in (that is the kernel's admission contract), so the
    concatenation is bit-identical to one launch -- and to the M=1 call --
    while each launch stays in registers.  Cost: the weight is read once
    per block instead of once (2 reads at 7-8 rows), still ~3x cheaper
    than the spilled single launch.  Blocks are balanced and never 1 row
    (a 1-row block would leave the admitted multi-row kernel).
    """
    flat = x.reshape(rows, in_dim)
    n_blocks = -(-rows // 4)
    base, extra = divmod(rows, n_blocks)
    outs, start = [], 0
    for b in range(n_blocks):
        size = base + (1 if b < extra else 0)
        y = row_exact_qmv(flat[start:start + size][None], weight, scales, biases,
                          group_size=group_size, bits=bits, mode=mode)
        if y is None:
            return None
        outs.append(y.reshape(size, out_dim))
        start += size
    STATS["blocked_calls"] = STATS.get("blocked_calls", 0) + 1
    return mx.concatenate(outs, axis=0).reshape(*x.shape[:-1], out_dim)


def row_exact_qmv(x, weight, scales, biases, *, group_size: int, bits: int, mode: str = "affine"):
    """``x @ dequant(weight).T`` with each row bit-equal to its M=1 call, or None.

    None means "use stock MLX" (unsupported shape, kernel unavailable, or the
    shape failed its bit-equality admission check).
    """
    if mode != "affine" or biases is None or x.ndim < 2 or _mlx_qmv_header() is None:
        return None
    in_dim = x.shape[-1]
    rows = 1
    for dim in x.shape[:-1]:
        rows *= int(dim)
    out_dim = weight.shape[0]
    if _efficient_rows(bits) < rows <= row_exact_scope_rows():
        return _row_blocks(x, rows, in_dim, out_dim, weight, scales, biases, group_size, bits, mode)
    variants = _variants(rows, in_dim, out_dim, bits, group_size, x.dtype)
    if not variants:
        return None
    if weight.dtype != mx.uint32:
        return None
    if scales.dtype != x.dtype or biases.dtype != x.dtype:
        # e.g. float32 activations over bf16 scales (Qwen3.5 attention promotes
        # through its float32 rotary): run in the activation dtype; admission
        # proves the result equals MLX's own M=1 call bit for bit.
        if x.dtype != mx.float32:
            return None
    ref_scales, ref_biases = scales, biases
    if scales.dtype != x.dtype:
        scales, biases = scales.astype(x.dtype), biases.astype(x.dtype)
    key = (rows, in_dim, out_dim, bits, group_size, str(x.dtype), str(ref_scales.dtype))
    variant = _admit(key, weight, scales, biases, bits, group_size, rows, in_dim, x.dtype, variants,
                     ref_scales, ref_biases)
    if variant is None:
        return None
    STATS["calls"] += 1
    y = _launch(mx.contiguous(x.reshape(rows, in_dim)), weight, scales, biases, bits, group_size, variant)
    return y.reshape(*x.shape[:-1], out_dim)


# --------------------------------------------------------------------------- #
# Verify scope: only the native-MTP target verify forward uses the kernel.
# --------------------------------------------------------------------------- #
import contextlib
import contextvars

import mlx.nn as nn

_SCOPE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "vmlx_row_exact_verify_qmv", default=False
)
_SCOPE_ROWS: contextvars.ContextVar[int] = contextvars.ContextVar(
    "vmlx_row_exact_verify_rows", default=MAX_ROWS
)
_INSTALLED = {"done": False}


def row_exact_scope_active() -> bool:
    return _SCOPE.get()


def row_exact_scope_rows() -> int:
    """Arithmetic coverage for this verify call, not a draft-depth limit."""
    return _SCOPE_ROWS.get()


_PAIR_CAUSAL_MASKS: "dict[tuple[int, str], mx.array]" = {}


def pair_causal_mask(end: int, dtype) -> mx.array:
    """Additive causal mask for the last two query rows over ``end`` keys.

    Shared by every attention layer of one verify forward (one lazy array).
    """
    key = (end, str(dtype))
    mask = _PAIR_CAUSAL_MASKS.get(key)
    if mask is None:
        q_pos = mx.arange(end - 2, end)[:, None]
        mask = mx.where(mx.arange(end)[None, :] <= q_pos, 0.0, float("-inf")).astype(dtype)[None, None]
        if len(_PAIR_CAUSAL_MASKS) >= 16:
            _PAIR_CAUSAL_MASKS.pop(next(iter(_PAIR_CAUSAL_MASKS)))
        _PAIR_CAUSAL_MASKS[key] = mask
    return mask


_ROTARY_OUTPUT_DTYPE: dict = {}


def rotary_rows_like_decode(rotary, position_ids: mx.array, x=None) -> tuple[mx.array, mx.array]:
    """Stock M-RoPE cos/sin for S rows, each row computed exactly as at decode.

    The stock embedding forms angles as ``inv_freq[F,1] @ positions[1,S]``.
    At decode S=1, so every batch element is a one-column product; a chunk of
    S>=2 columns takes a different matmul kernel that rounds differently.
    Moving the S positions into the batch dimension gives every row its own
    one-column product in a single call; the remaining ops are elementwise.
    """
    if position_ids.ndim == 2:
        position_ids = mx.broadcast_to(
            position_ids[None, ...], (3,) + tuple(position_ids.shape)
        )
    axes, batch, rows = position_ids.shape
    width = rotary.inv_freq.shape[0]
    inv = mx.broadcast_to(
        rotary.inv_freq.astype(mx.float32)[None, None, None, :, None],
        (axes, batch, rows, width, 1),
    )
    pos = position_ids.astype(mx.float32)[..., None, None]  # [3, B, S, 1, 1]
    freqs = (inv @ pos).reshape(axes, batch, rows, width)
    freqs = rotary.apply_interleaved_mrope(freqs, rotary.mrope_section)
    emb = mx.concatenate([freqs, freqs], axis=-1)
    cos, sin = mx.cos(emb), mx.sin(emb)
    if x is not None:
        # Some rotary copies return float32, others cast to the activation
        # dtype (vendored Qwen3.5). Match this class's own convention; the
        # one-row call is lazy and never evaluated, only its dtype is read.
        key = (type(rotary), str(x.dtype))
        out_dtype = _ROTARY_OUTPUT_DTYPE.get(key)
        if out_dtype is None:
            out_dtype = rotary(x[:, :1], position_ids[..., :1])[0].dtype
            _ROTARY_OUTPUT_DTYPE[key] = out_dtype
        if out_dtype != cos.dtype:
            cos, sin = cos.astype(out_dtype), sin.astype(out_dtype)
    return cos, sin


@lru_cache(maxsize=1)
def _gpu_class() -> str:
    try:
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        return str(info.get("architecture", ""))[-1:]
    except Exception:
        return ""


def _vector_sdpa_class(n_keys: int, gqa: int, rows: int, masked: bool, head_dim: int, value_dim: int):
    """The kernel and key-block count MLX 0.32.3's vector SDPA uses for ``rows`` (1-2) queries over ``n_keys`` keys.

    Mirrors mlx/backend/metal/scaled_dot_product_attention.cpp: the 1-pass / 2-pass dispatch (``k >= 1024`` on 'd' /
    's' GPUs, or GQA with ``k >= 4096``), the one-query unmasked GQA kernel (``gqa 8`` at D 64 / 128, ``gqa 12 / 16``
    at D 128, ``k >= 8192``) and the 2-pass block count, which steps with the key count (M5 Max 's': 64 / 128 at
    1,025 / 256 at 8,193 / 512 at 32,769 / 1,024 at 65,537 keys). A query's reduction order depends only on this
    class, so a row inside a pair equals its decode step exactly when both calls share it.
    """
    devc = _gpu_class()
    if not ((devc in ("d", "s") and n_keys >= 1024) or (gqa > 1 and n_keys >= 4096)):
        return ("1pass",)
    gqa_dims = (gqa == 8 and head_dim in (64, 128)) or (gqa in (12, 16) and head_dim == 128)
    variant = "gqa" if (not masked and rows == 1 and head_dim == value_dim and gqa_dims and n_keys >= 8192) else "plain"
    n_simds = gqa * rows
    if devc == "s":
        blocks = 64
        if n_keys > 1024 and n_simds > 4:
            blocks = 128 if n_keys <= 8192 else 256 if n_keys <= 32768 else 512 if n_keys <= 65536 else 1024
    elif devc == "d":
        blocks = 128
        if n_simds <= 2 and n_keys > 8192:
            blocks = 256
        elif n_simds >= 6:
            if 16384 <= n_keys < 65536:
                blocks = 512
            elif n_keys >= 65536:
                blocks = 1024
    else:
        blocks = 64 if n_simds >= 4 else 32
    try:
        env_blocks = int(os.environ.get("MLX_SDPA_BLOCKS", "0") or 0)
    except ValueError:
        env_blocks = 0
    if env_blocks > 0:
        blocks = ((env_blocks + 31) // 32) * 32
    return ("2pass", variant, blocks)


def attend_rows_in_pairs(attend, queries, keys, values, row_masks=None):
    """Attend S verify rows like their decode steps.

    MLX keeps its single-query vector SDPA for up to two queries and switches
    to the full kernel, whose reduction order differs, from three. Row ``r``
    sees keys up to and including itself. Two rows share one call only when
    MLX would give each of them the same kernel and key-block count as its own
    decode step (``_vector_sdpa_class``); otherwise each row is attended alone
    over exactly its own keys, which is its decode step. (Pairing blindly broke
    exactness at the block-count steps -- 1,024 / 8,192 / 32,768 keys on M5 Max
    -- and on every row past 8,192 keys where decode uses MLX's GQA kernel.)
    ``attend(q, k, v, mask)`` is the model's own SDPA call; ``row_masks``
    optionally supplies extra additive rows ``[..., S, T]`` (e.g. a sparse
    index mask), sliced per row group.
    """
    S = queries.shape[2]
    T = keys.shape[2]
    start = T - S
    gqa = max(1, int(queries.shape[1]) // max(1, int(keys.shape[1])))
    head_dim, value_dim = int(queries.shape[-1]), int(values.shape[-1])
    masked = row_masks is not None

    def _single(row):
        end = start + row + 1
        mask = None if row_masks is None else row_masks[..., row:row + 1, :end]
        return attend(queries[:, :, row:row + 1], keys[:, :, :end], values[:, :, :end], mask)

    outs = []
    for first in range(0, S, 2):
        count = min(2, S - first)
        end = start + first + count
        if count == 2:
            in_pair = _vector_sdpa_class(end, gqa, 2, True, head_dim, value_dim)
            if not (in_pair == _vector_sdpa_class(end - 1, gqa, 1, masked, head_dim, value_dim)
                    == _vector_sdpa_class(end, gqa, 1, masked, head_dim, value_dim)):
                outs.extend((_single(first), _single(first + 1)))
                continue
        mask = None if row_masks is None else row_masks[..., first:first + count, :end]
        if count == 2:
            causal = pair_causal_mask(end, queries.dtype)
            mask = causal if mask is None else causal + mask
        outs.append(attend(queries[:, :, first:first + count], keys[:, :, :end], values[:, :, :end], mask))
    return mx.concatenate(outs, axis=2)


_DENSE_ADMITTED: dict[tuple, bool] = {}


def row_exact_linear(x, weight, bias=None):
    """Dense ``x @ weight.T (+ bias)`` with each row bit-equal to its M=1 call, or None.

    MLX's single-row ``x @ W.T`` is a GEMV; at M >= 2 it switches to a GEMM
    whose reduction order differs (measured on fp16/bf16 router, hyper-connection
    and square shapes). A batched matrix-vector product over a broadcast view of
    the weight keeps every row on the GEMV path without copying the weight.
    """
    if x.ndim < 2 or weight.ndim != 2 or x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        return None
    rows = 1
    for dim in x.shape[:-1]:
        rows *= int(dim)
    if not 2 <= rows <= row_exact_scope_rows() or weight.dtype != x.dtype:
        return None
    out_dim, in_dim = weight.shape

    def run(x2d):
        y = mx.matmul(mx.broadcast_to(weight, (rows, out_dim, in_dim)), x2d.reshape(rows, in_dim, 1))
        return y.reshape(rows, out_dim)

    key = (rows, in_dim, out_dim, str(x.dtype))
    admitted = _DENSE_ADMITTED.get(key)
    if admitted is None:
        probe = mx.random.normal((rows, in_dim), key=mx.random.key(4321)).astype(x.dtype)
        want = mx.concatenate([probe[r:r + 1] @ weight.T for r in range(rows)], axis=0)
        admitted = bool(mx.array_equal(run(probe), want).item())
        _DENSE_ADMITTED[key] = admitted
        STATS["dense_admitted" if admitted else "dense_rejected"] = STATS.get(
            "dense_admitted" if admitted else "dense_rejected", 0) + 1
        if not admitted:
            logger.warning("row-exact verify dense linear REJECTED rows=%d K=%d N=%d dtype=%s",
                           rows, in_dim, out_dim, x.dtype)
    if not admitted:
        return None
    STATS["dense_calls"] = STATS.get("dense_calls", 0) + 1
    y = run(x.reshape(rows, in_dim)).reshape(*x.shape[:-1], out_dim)
    return y + bias if bias is not None else y


def _dense_rows(x, weight, bias=None):
    """Row-invariant dense projection for 1..MAX_ROWS rows.

    Every row is its own matrix-vector product over a broadcast (uncopied)
    view of the weight, so a row's result does not depend on how many rows
    share the call. Used for BOTH single-token decode and verify rows, which
    makes them bit-identical by construction. (MLX's own one-row ``x @ W.T``
    and this form reduce in different orders; the difference is rare but
    real: 1 in 900 rows on a 2560->1 gate, and it broke greedy identity.)
    """
    if x.ndim < 2 or weight.ndim != 2 or x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        return None
    if weight.dtype != x.dtype:
        return None
    rows = 1
    for dim in x.shape[:-1]:
        rows *= int(dim)
    if not 1 <= rows <= row_exact_scope_rows():
        return None
    out_dim, in_dim = weight.shape
    x2d = x.reshape(rows, in_dim)
    if out_dim < 8 and rows > 1:
        # Degenerate widths (e.g. the 2560->1 shared-expert gate): the batched
        # form is NOT batch-invariant there (measured on Allosaurus layer 6), so
        # every row runs the exact one-row call.
        y = mx.concatenate([
            mx.matmul(weight[None], x2d[r:r + 1].reshape(1, in_dim, 1)) for r in range(rows)
        ], axis=0).reshape(*x.shape[:-1], out_dim)
    else:
        y = mx.matmul(mx.broadcast_to(weight, (rows, out_dim, in_dim)),
                      x2d.reshape(rows, in_dim, 1)).reshape(*x.shape[:-1], out_dim)
    return y + bias if bias is not None else y


def _install() -> None:
    if _INSTALLED["done"]:
        return
    original_dense = nn.Linear.__call__

    def patched_dense(self, x):  # type: ignore[no-untyped-def]
        if _DENSE_ROW_INVARIANT["on"]:
            y = _dense_rows(x, self["weight"], self["bias"] if "bias" in self else None)
            if y is not None:
                return y
        if _SCOPE.get():
            y = row_exact_linear(x, self["weight"], self["bias"] if "bias" in self else None)
            if y is not None:
                return y
        return original_dense(self, x)

    nn.Linear.__call__ = patched_dense
    original = nn.QuantizedLinear.__call__

    def patched(self, x):  # type: ignore[no-untyped-def]
        if _SCOPE.get() and "biases" in self:
            y = row_exact_qmv(
                x, self["weight"], self["scales"], self["biases"],
                group_size=int(self.group_size), bits=int(self.bits),
                mode=str(getattr(self, "mode", "affine") or "affine"),
            )
            if y is not None:
                return y + self["bias"] if "bias" in self else y
        return original(self, x)

    nn.QuantizedLinear.__call__ = patched
    _INSTALLED["done"] = True


@contextlib.contextmanager
def row_exact_verify_scope(*, rows: int = MAX_ROWS, required: bool = False):
    """Keep a target verify window on decode arithmetic.

    Existing native-MTP callers retain their family policy and coverage.
    DFlash supplies its actual adaptive window and requires exact arithmetic;
    wide projections use bounded kernel launches without limiting draft depth.
    """
    if rows < 2:
        raise ValueError("row-exact verify coverage must include at least two rows")
    if not required and not row_exact_qmv_requested():
        yield False
        return
    _install()
    token = _SCOPE.set(True)
    rows_token = _SCOPE_ROWS.set(int(rows))
    env_token = _SCOPE_ENV.set((os.environ.get("VMLX_ROW_EXACT_QMV_ORDER", "multi,plain"),
                                os.environ.get("VMLX_ROW_EXACT_QMV_REG", "").strip()))
    try:
        yield True
    finally:
        _SCOPE_ENV.reset(env_token)
        _SCOPE_ROWS.reset(rows_token)
        _SCOPE.reset(token)


# --------------------------------------------------------------------------- #
# Wide verify windows (up to 16 rows per weight walk).
# --------------------------------------------------------------------------- #
# A DSpark verify of the MiniMax-M3.1 trunk runs 2..16 rows through every
# projection. The register transcriptions above hold every row's x_thread in
# registers and cap at 4 rows (2-5-bit) / 6 rows (8-bit) before spilling, so a
# 16-row window walked the weight 4 times (measured, M5 Max, 57 M3.1 layers of
# q/k/v/o/index/shared projections: 12.5 ms at 1 row, 17.7 at 4, 82 at 16).
# The wide walk keeps the qmv_fast arithmetic of every (row, output row) pair
# -- load_vector per row, the shared-unpack qdot_rows terms in qdot's order,
# the same k-block order and the same simd_sum -- but stages each block's
# x_thread / sum of the window's rows in threadgroup memory, computed once per
# threadgroup and read by every simdgroup, and keeps only the per-row results
# in registers (RPS output rows per simdgroup, NSG simdgroups per threadgroup).
# Rows are staged RC at a time (threadgroup memory), each chunk in the same
# block iteration, so every pair still accumulates block by block in order.
# Admission is the same per-shape bit-equality check against stock M=1 calls.
WIDE_MAX_ROWS = 16
_WIDE_ADMITTED: dict = {}


def _wide_qdot_rows() -> str:
    """qdot_rows with x_thread / sum read from threadgroup memory at a row stride (same terms, same order)."""
    src = _QDOT_ROWS.split("template <typename U, int values_per_thread, int bits, int ROWS>", 1)[1]
    src = ("template <typename U, int values_per_thread, int bits, int ROWS, int XSTRIDE, int SSTRIDE>" +
           src.replace("inline void qdot_rows(", "inline void qdot_rows_tg(")
           .replace("const thread U* x,          // [ROWS][values_per_thread], row-major",
                    "const threadgroup U* x,     // row r at x + r * XSTRIDE")
           .replace("const thread U* sum,        // [ROWS]", "const threadgroup U* sum,   // row r at sum[r * SSTRIDE]")
           .replace("x + r * values_per_thread", "x + r * XSTRIDE")
           .replace("const thread U* xt", "const threadgroup U* xt")
           .replace("x[r * values_per_thread + i]", "x[r * XSTRIDE + i]")
           .replace("sum[r] * bias", "sum[r * SSTRIDE] * bias"))
    return src


_SOURCE_WIDE = r"""
  // qmv_fast_impl per (row, output row); the window's rows staged per block in threadgroup memory.
  constexpr int bits = BITS;
  constexpr int group_size = GS;
  constexpr int in_vec_size = K;
  constexpr int out_vec_size = N;
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  constexpr int w_words = (packs_per_thread * bytes_per_pack + 3) / 4;
  typedef float U;

  threadgroup U xs[RC * SIMD_SIZE * values_per_thread];
  threadgroup U ss[RC * SIMD_SIZE];

  uint3 tid = threadgroup_position_in_grid;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint simd_lid = thread_index_in_simdgroup;
  uint lid = thread_index_in_threadgroup;

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (NSG * RPS) + simd_gid * RPS;

  const device uint8_t* ws = (const device uint8_t*)w + out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  const device T* sc = scales + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* bi = biases + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

  thread U result[ROWS][RPS];
  for (int r = 0; r < ROWS; r++)
    for (int o = 0; o < RPS; o++) result[r][o] = 0;

  for (int k = 0; k < in_vec_size; k += block_size) {
    thread uint32_t wrow[RPS][w_words];
    U srow[RPS], brow[RPS];
    for (int o = 0; o < RPS; o++) {
      const device uint8_t* wl = ws + o * in_vec_size_w;
      thread uint8_t* wb = (thread uint8_t*)wrow[o];
      for (int j = 0; j < packs_per_thread * bytes_per_pack; j++) wb[j] = wl[j];
      srow[o] = sc[o * in_vec_size_g];
      brow[o] = bi[o * in_vec_size_g];
    }
    for (int c0 = 0; c0 < ROWS; c0 += RC) {
      for (int p = int(lid); p < RC * SIMD_SIZE; p += NSG * SIMD_SIZE) {
        const int r = p / SIMD_SIZE, l = p % SIMD_SIZE;
        if (c0 + r < ROWS) {
          U xt[values_per_thread];
          U s = load_vector<T, U, values_per_thread, bits>(x + (c0 + r) * in_vec_size + k + l * values_per_thread, xt);
          for (int i = 0; i < values_per_thread; i++) xs[(r * SIMD_SIZE + l) * values_per_thread + i] = xt[i];
          ss[r * SIMD_SIZE + l] = s;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int o = 0; o < RPS; o++) {
        U part[RC];
        qdot_rows_tg<U, values_per_thread, bits, RC, SIMD_SIZE * values_per_thread, SIMD_SIZE>(
            (const thread uint8_t*)wrow[o], xs + simd_lid * values_per_thread, srow[o], brow[o], ss + simd_lid, part);
        for (int r = 0; r < RC; r++) {
          if (c0 + r < ROWS) result[c0 + r][o] += part[r];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    ws += block_size * bytes_per_pack / pack_factor;
    sc += block_size / group_size;
    bi += block_size / group_size;
  }

  for (int r = 0; r < ROWS; r++) {
    for (int o = 0; o < RPS; o++) {
      U v = simd_sum(result[r][o]);
      if (simd_lid == 0 && out_row + o < out_vec_size) {
        y[r * out_vec_size + out_row + o] = static_cast<T>(v);
      }
    }
  }
"""


@lru_cache(maxsize=8)
def _wide_kernel():
    header = _mlx_qmv_header()
    if header is None:
        return None
    return mx.fast.metal_kernel(
        name="vmlx_row_exact_verify_qmv_wide",
        input_names=["w", "scales", "biases", "x"],
        output_names=["y"],
        header=header + "\n\n" + _wide_qdot_rows(),
        source=_SOURCE_WIDE,
    )


def _wide_shape(rows: int, bits: int) -> tuple:
    """(RPS output rows per simdgroup, NSG simdgroups, RC rows staged at a time); env-pinnable for A/B."""
    vpt = _values_per_thread(bits)
    rc_cap = max(1, (24 * 1024) // (4 * 32 * (vpt + 1)))       # staged x_thread + sums within 24 KB
    try:
        rps = int(os.environ.get("VMLX_ROW_EXACT_WIDE_RPS", "") or 1)
        nsg = int(os.environ.get("VMLX_ROW_EXACT_WIDE_NSG", "") or 8)
    except ValueError:
        rps, nsg = 1, 8
    return rps, nsg, min(rows, rc_cap)


def _launch_wide(x2d, weight, scales, biases, bits, group_size, shape):
    rows, in_dim = x2d.shape
    out_dim = weight.shape[0]
    rps, nsg, rc = shape
    weight, scales, biases = (_device_sized(a) for a in (weight, scales, biases))
    per_tg = nsg * rps
    (y,) = _wide_kernel()(
        inputs=[weight, scales, biases, x2d],
        template=[("T", x2d.dtype), ("BITS", bits), ("GS", group_size), ("K", in_dim), ("N", out_dim),
                  ("ROWS", rows), ("RPS", rps), ("NSG", nsg), ("RC", rc)],
        grid=(32 * nsg, (out_dim + per_tg - 1) // per_tg, 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(rows, out_dim)],
        output_dtypes=[x2d.dtype],
    )
    return y


def wide_enabled() -> bool:
    """VMLX_ROW_EXACT_WIDE=1 (default OFF; read per call): the one-walk wide kernel for 2..16-row windows.

    Measured 2026-10-28 (M5 Max, 57 M3.1 layers of q/k/v/o/index/shared projections, interleaved A/B, min of 9):
    4 rows 21.3 ms (4-row launches) vs 17.3-27.8 (wide, by shape), 8 rows 56.4 vs 52.5-60.5, 12 / 16 rows 84.5 /
    103.2 vs 205-450 (the staged x_thread reads from threadgroup memory and the per-row results in registers cost
    more than the weight re-walks they save). Exact (admitted per shape) but slower: kept for A/B, off by default."""
    return os.environ.get("VMLX_ROW_EXACT_WIDE", "0").strip().lower() in {"1", "true", "yes", "on"}


def row_exact_qmv_wide(x, weight, scales, biases, *, group_size: int, bits: int):
    """``x @ dequant(weight).T`` for 2..16 rows in ONE weight walk, every row bit-equal to its M=1 call, or None.

    Only MLX's qmv_fast traversal (K a multiple of the block, N a multiple of 8, group a multiple of the lane's
    values), affine 2/3/4/5/6/8-bit, activation dtype == scale dtype. Admitted per (rows, K, N, bits, group, dtype,
    shape) by bit equality against stock one-row calls; None keeps the caller's path.
    """
    if x.ndim < 2 or biases is None or _mlx_qmv_header() is None or weight.dtype != mx.uint32:
        return None
    in_dim = int(x.shape[-1])
    rows = 1
    for dim in x.shape[:-1]:
        rows *= int(dim)
    out_dim = int(weight.shape[0])
    if not 2 <= rows <= WIDE_MAX_ROWS or bits not in (2, 3, 4, 5, 6, 8) or group_size not in (32, 64, 128):
        return None
    vpt = _values_per_thread(bits)
    if in_dim % (vpt * 32) or out_dim % 8 or group_size % vpt or in_dim % group_size:
        return None
    if x.dtype not in (mx.float16, mx.bfloat16, mx.float32) or scales.dtype != x.dtype or biases.dtype != x.dtype:
        return None
    shape = _wide_shape(rows, bits)
    key = (rows, in_dim, out_dim, bits, group_size, str(x.dtype), shape)
    ok = _WIDE_ADMITTED.get(key)
    if ok is None:
        probes = _admission_probes(rows, in_dim, x.dtype)
        ok = True
        for probe in probes:
            want = mx.concatenate([
                mx.quantized_matmul(probe[r:r + 1], weight, scales, biases, transpose=True, group_size=group_size,
                                    bits=bits, mode="affine") for r in range(rows)], axis=0)
            got = _launch_wide(probe, weight, scales, biases, bits, group_size, shape)
            ok = ok and bool(mx.array_equal(got, want).item())
        _WIDE_ADMITTED[key] = ok
        STATS["wide_admitted" if ok else "wide_rejected"] = STATS.get("wide_admitted" if ok else "wide_rejected", 0) + 1
        (logger.info if ok else logger.warning)(
            "row-exact verify QMV wide %s rows=%d K=%d N=%d bits=%d gs=%d dtype=%s shape=%s",
            "admitted" if ok else "REJECTED", rows, in_dim, out_dim, bits, group_size, x.dtype, shape)
    if not ok:
        return None
    STATS["wide_calls"] = STATS.get("wide_calls", 0) + 1
    y = _launch_wide(mx.contiguous(x.reshape(rows, in_dim)), weight, scales, biases, bits, group_size, shape)
    return y.reshape(*x.shape[:-1], out_dim)


# --------------------------------------------------------------------------- #
# Row-split walk: RB blocks of up to 4 rows in ONE threadgroup.
# --------------------------------------------------------------------------- #
# The 4-row "multi" transcription of qmv_fast (registers hold 4 rows), with the threadgroup's simdgroups split by
# (output-row group, row block): simdgroup s computes output rows (s % 2) * 4 .. + 3 for rows (s / 2) * 4 .. + 3.
# The row blocks of one threadgroup read the same weight bytes at the same time, so the window costs one weight walk
# from DRAM while each simdgroup keeps exactly the 4-row kernel's registers and per-row arithmetic.
# Measured 2026-10-28 (M5 Max, 57 M3.1 layers of q/k/v/o/index/shared projections, interleaved, min of 9): 5 / 8 /
# 12 / 16 rows 30.5 / 46.8 / 71.9 / 117.4 ms as 4-row launches vs 28.8 / 38.2 / 56.7 / 94.6 row-split (-6 / -18 /
# -21 / -19 %). The rest is the per-row arithmetic itself (MLX's one-row qmv per row), which exactness keeps.
def _rowsplit_source() -> str:
    src = _SOURCE.replace(
        "  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +\n      simd_gid * results_per_simdgroup;",
        "  const int rb = int(simd_gid) / num_simdgroups;\n"
        "  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +\n"
        "      (int(simd_gid) % num_simdgroups) * results_per_simdgroup;")
    src = src.replace("const device T* xs = x + (BATCH ? tid.x * in_vec_size : 0) + simd_lid * values_per_thread;",
                      "const device T* xs = x + rb * ROWS * in_vec_size + simd_lid * values_per_thread;")
    src = src.replace("y[(BATCH ? tid.x : r) * out_vec_size + out_row + row] = static_cast<T>(v);",
                      "y[(rb * ROWS + r) * out_vec_size + out_row + row] = static_cast<T>(v);")
    assert "rb * ROWS * in_vec_size" in src and "(rb * ROWS + r)" in src
    return src


@lru_cache(maxsize=4)
def _rowsplit_kernel(reg: bool):
    header = _mlx_qmv_header()
    if header is None:
        return None
    return mx.fast.metal_kernel(
        name=f"vmlx_row_exact_verify_qmv_rowsplit{'_reg' if reg else ''}",
        input_names=["w", "scales", "biases", "x"], output_names=["y"], header=header,
        source=f"#define REG {1 if reg else 0}\n#define MULTI 1\n#define BATCH 0\n#define ROWOUTER 0\n" + _rowsplit_source())


_ROWSPLIT_ADMITTED: dict = {}


def rowsplit_enabled() -> bool:
    """VMLX_ROW_EXACT_ROWSPLIT (default on; read per call): 5..16-row verify windows in one weight walk."""
    return os.environ.get("VMLX_ROW_EXACT_ROWSPLIT", "1").strip().lower() not in {"0", "false", "no", "off"}


def row_exact_qmv_rowsplit(x, weight, scales, biases, *, group_size: int, bits: int):
    """5..16 rows in one weight walk as ceil(rows / 4) 4-row blocks per threadgroup (rows padded with zeros to a
    multiple of the block, padded outputs dropped), every row bit-equal to its one-row call, or None."""
    if x.ndim < 2 or biases is None or _mlx_qmv_header() is None or weight.dtype != mx.uint32:
        return None
    in_dim, out_dim = int(x.shape[-1]), int(weight.shape[0])
    rows = 1
    for dim in x.shape[:-1]:
        rows *= int(dim)
    if not 2 <= rows <= WIDE_MAX_ROWS or bits not in (2, 3, 4, 5, 6, 8) or group_size not in (32, 64, 128):
        return None
    vpt = _values_per_thread(bits)
    if in_dim % (vpt * 32) or out_dim % 8 or group_size % vpt or in_dim % group_size:
        return None
    if x.dtype not in (mx.float16, mx.bfloat16, mx.float32) or scales.dtype != x.dtype or biases.dtype != x.dtype:
        return None
    rb = -(-rows // 4)
    per = -(-rows // rb)                                   # balanced block (<= 4 rows)
    reg = _register_weights(per)

    def launch(x2d):
        n = int(x2d.shape[0])
        if n < rb * per:
            x2d = mx.concatenate([x2d, mx.zeros((rb * per - n, in_dim), dtype=x2d.dtype)], axis=0)
        w_, s_, b_ = (_device_sized(a) for a in (weight, scales, biases))
        (y,) = _rowsplit_kernel(reg)(
            inputs=[w_, s_, b_, x2d],
            template=[("T", x2d.dtype), ("BITS", bits), ("GS", group_size), ("K", in_dim), ("N", out_dim),
                      ("ROWS", per)],
            grid=(64 * rb, (out_dim + 7) // 8, 1), threadgroup=(64 * rb, 1, 1),
            output_shapes=[(rb * per, out_dim)], output_dtypes=[x2d.dtype])
        return y[:n]

    key = (rows, in_dim, out_dim, bits, group_size, str(x.dtype))
    ok = _ROWSPLIT_ADMITTED.get(key)
    if ok is None:
        ok = True
        for probe in _admission_probes(rows, in_dim, x.dtype):
            want = mx.concatenate([
                mx.quantized_matmul(probe[r:r + 1], weight, scales, biases, transpose=True, group_size=group_size,
                                    bits=bits, mode="affine") for r in range(rows)], axis=0)
            ok = ok and bool(mx.array_equal(launch(probe), want).item())
        _ROWSPLIT_ADMITTED[key] = ok
        (logger.info if ok else logger.warning)("row-exact verify QMV rowsplit %s rows=%d K=%d N=%d bits=%d",
                                                "admitted" if ok else "REJECTED", rows, in_dim, out_dim, bits)
    if not ok:
        return None
    STATS["rowsplit_calls"] = STATS.get("rowsplit_calls", 0) + 1
    return launch(mx.contiguous(x.reshape(rows, in_dim))).reshape(*x.shape[:-1], out_dim)
