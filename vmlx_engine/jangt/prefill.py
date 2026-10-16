"""JANGT prefill: MLX/JANGH NAX gather matmul with a trellis tile loader (vMLX, INTERNAL).

Text-derived from vmlx_engine.jangh.kernels._NAX_IMPL (same tiling, expert segmentation, sg_active, store); only the
weight pointer (uint32 rows of W words), the fp32 row scales and the tile loader change. JTBlockLoader: 128 threads
fill a 64-row x 64-col bf16 tile; thread = (row, half): 32 columns = two 16-column sub-chunks, each decoded with the
decode kernels' window logic (loop-invariant sub-word shift, static step shifts) and normalized: (raw - mu) * isd * s_row.
"""
from __future__ import annotations

import functools

import mlx.core as mx

from vmlx_engine.jangh import kernels as JK
from vmlx_engine.jangt import v2l as VL
from vmlx_engine.jangt.v2core import MASK, ORB


def _tag(K) -> str:
    return str(K).replace(".", "p")


def _loader(K, L: int, W: int) -> str:
    mu, isd = VL.code_stats(L, K); so = VL.step_offsets(K, 9); MU = VL.MULTS[K]; mk = (1 << L) - 1
    PL16 = int(VL.step_offsets(K, 17)[16]) // 2          # bits per 16 columns (8 steps)
    M2 = MASK | (MASK << 16); O2 = ORB | (ORB << 16)
    # sub-chunk starts are multiples of PL16 bits, the same shift set as the decode lanes: one window rule for every K
    # (K=1/1.5: 2 words realigned into a0; K=2.5/3: 3 words into a0/a1; K=2: static) -> moe_kernels._window
    from vmlx_engine.jangt.moe_kernels import _step_extract, _window_load
    ld = _window_load(K, L, lambda i: f"r[wi + {i}]" if i else "r[wi]")
    steps = []
    for j in range(8):
        ex = _step_extract(int(so[j]), L)
        steps.append(f"{{ uint s = ({ex}) & {mk}u; half2 v = as_type<half2>(((s * {MU}u) & {M2}u) | {O2}u);"
                     f" d[{2*j}] = T((float(v.x) - {float(mu[0])}f) * {float(isd[0]) }f * s_row);"
                     f" d[{2*j+1}] = T((float(v.y) - {float(mu[1])}f) * {float(isd[1])}f * s_row); }}")
    body = " ".join(steps)
    return f'''
template <typename T, short BROWS, short BCOLS, short dst_ld, short tgp_size>
struct JTBlockLoader_K{_tag(K)} {{
  static_assert(BCOLS == 64 && tgp_size == 128, "JANGT loader: 64-column tiles, 128 threads");
  const short thread_idx; const short bi; const short hf;
  threadgroup T* dst; const device uint32_t* src; float s_row; uint bitpos;
  JTBlockLoader_K{_tag(K)}(const device uint32_t* src_, const device float* scales_, const int src_ld_,
                threadgroup T* dst_, ushort simd_group_id, ushort simd_lane_id)
      : thread_idx(simd_group_id * 32 + simd_lane_id), bi(thread_idx / 2), hf(thread_idx % 2),
        dst(dst_ + bi * dst_ld + hf * 32), src(src_ + size_t(bi) * {W}), bitpos(uint(hf) * {2 * PL16}u) {{
    s_row = (bi < BROWS) ? scales_[bi] : 0.0f;
  }}
  METAL_FUNC void sub_(uint b, threadgroup T* d) const {{
    const device uint32_t* r = src; uint wi = b >> 5; uint sh = b & 31u; (void)sh;
    {ld}
    {body}
  }}
  void load_unsafe() const {{ if (bi >= BROWS) return; sub_(bitpos, dst); sub_(bitpos + {PL16}u, dst + 16); }}
  void load_safe(short2 src_tile_dim) const {{
    if (bi >= BROWS) return;
    if (bi >= src_tile_dim.y) {{ for (short i = 0; i < 32; i++) dst[i] = T(0); return; }}
    sub_(bitpos, dst); sub_(bitpos + {PL16}u, dst + 16);
  }}
  void next() {{ bitpos += {4 * PL16}u; }}
}};
'''


def _impl(K, W: int) -> str:
    s = JK._NAX_IMPL
    reps = [("tq_gather_qmm_nax(", f"jt_gather_qmm_nax_K{_tag(K)}("),
            ("const device uint32_t* wg, const device half* sg, const device uint32_t* wu, const device half* su,",
             "const device uint32_t* wg, const device float* sg, const device uint32_t* wu, const device float* su,"),
            ("using loader_w_t = TQBlockLoader<T, BN, BK, BK_padded, WM * WN * SIMD_SIZE, bits>;",
             f"using loader_w_t = JTBlockLoader_K{_tag(K)}<T, BN, BK, BK_padded, WM * WN * SIMD_SIZE>;"),
            ("const int K_w = K * bytes_per_pack / pack_factor;", f"const int K_w = {W};"),
            ("auto wgl = (const device uint8_t*)wg; auto wul = (const device uint8_t*)wu;", "auto wgl = wg; auto wul = wu;")]
    for a, b in reps:
        assert s.count(a) >= 1, a
        s = s.replace(a, b, 1)
    # keep only the gather function (the radix helper + act are reused under new names to avoid symbol clashes)
    s = s.replace("jangh_h32_radix", f"jt_h32_radix_K{_tag(K)}").replace("tq_act(", f"jt_act_K{_tag(K)}(")
    return s


def _tail_args(expert: bool, h32: bool) -> str:
    """Trailing template arguments of the expert-aligned schedule (tiles never span two experts: JK.expert_tile_plan)
    and of the H32 output epilogue (rotate the activations for a JANGH down unit before the store)."""
    return (", expert_offsets, tile_offsets, meta[3]" if expert else "") + (", scale[0]" if h32 else "")


def _tail_inputs(expert: bool, h32: bool) -> list:
    return (["expert_offsets", "tile_offsets"] if expert else []) + (["scale"] if h32 else [])


def _variant(fused: bool, expert: bool, h32: bool) -> str:
    return f"{'fused' if fused else 'single'}{'_et' if expert else ''}{'_h32' if h32 else ''}"


@functools.lru_cache(maxsize=None)
def _kernel(K, W: int, fused: bool, tname: str, expert: bool = False, h32: bool = False):
    header = JK._mlx_headers(["steel/gemm/gemm.h", "steel/gemm/nax.h", "steel/gemm/loader.h", "quantized_nax.h"]) + \
        _loader(K, 12, W) + _impl(K, W)
    src = f'''
  constexpr int BK_padded = (64 + 16 / sizeof({tname}));
  threadgroup {tname} Wg[64 * BK_padded];
  threadgroup {tname} Wu[{'64' if fused else '1'} * BK_padded];
  jt_gather_qmm_nax_K{_tag(K)}<{tname}, 2, {'true' if fused else 'false'}, {'true' if expert else 'false'}, {'true' if h32 else 'false'}>(
      x, wg, sg, wu, su, indices, y, meta[0], meta[1], meta[2], lim[0], Wg, Wu,
      threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup{_tail_args(expert, h32)});
'''
    return mx.fast.metal_kernel(name=f"jt_qmm_nax_K{_tag(K)}_w{W}_{_variant(fused, expert, h32)}_{tname}",
                                input_names=["x", "wg", "sg", "wu", "su", "indices", "meta", "lim"] + _tail_inputs(expert, h32),
                                output_names=["y"], header=header, source=src)


def _pad_idx(idx_sorted):
    idx = idx_sorted.astype(mx.uint32).reshape(-1)
    if idx.size < 8:
        idx = mx.concatenate([idx, mx.broadcast_to(idx[-1:], (8 - idx.size,))])
    return idx


def _launch(M: int, N: int, E: int, plan, rotate_output: bool):
    """meta, extra inputs and grid for the row-tiled (plan None) or expert-aligned schedule."""
    if plan is None:
        if rotate_output:
            raise ValueError("JANGT H32 output epilogue needs the expert-aligned schedule")
        return [], (M + 63) // 64, []
    starts, tiles = plan
    extra = [starts, tiles] + ([JK._h32_scale()] if rotate_output else [])
    return [E], (M + 63) // 64 + E, extra


def gather_qmm_sorted_jt(x, P: dict, idx_sorted, Pu: dict | None = None, limit: float = 0.0, plan=None,
                         rotate_output: bool = False):
    """x (M, K_in) bf16 rows sorted by expert, ROTATED basis; P = unit projection dict (packed (E,N,W), scale (E,N)).
    Returns (M, N) x.dtype. Outlier columns are NOT included (caller adds them). plan = JK.expert_tile_plan(idx, E):
    tiles never span two experts (the JANGH prefill schedule). rotate_output (fused bf16 + plan): the activations come
    back Hadamard-32 rotated, the input basis of a JANGH down unit."""
    M, Kin = x.shape; E, N, W = P["packed"].shape
    idx = _pad_idx(idx_sorted)
    fused = Pu is not None
    meta_x, gy, extra = _launch(M, N, E, plan, rotate_output)
    k = _kernel(P["K"], W, fused, JK._TNAME[x.dtype], plan is not None, bool(rotate_output))
    return k(inputs=[x, P["packed"], P["scale"], Pu["packed"] if fused else P["packed"], Pu["scale"] if fused else P["scale"], idx,
                     mx.array([M, N, Kin] + meta_x, dtype=mx.int32), mx.array([float(limit)], dtype=mx.float32)] + extra,
             grid=(((N + 63) // 64) * 128, gy, 1), threadgroup=(128, 1, 1),
             output_shapes=[(M, N)], output_dtypes=[x.dtype])[0]


# ---------------------------------------------------------------------------------------------------------------
# Fused gate+up with OUTLIER COLUMNS IN THE EPILOGUE (2026-10-09). JANGT gate/up carry n_out (2-3) layer-wide raw-basis
# outlier input columns: g = W_g,rot . x_rot + Wout_g . x_raw[ocol] (same for u), and SwiGLU needs both sums BEFORE
# the activation, so the separate path ran gate and up as two NAX passes (+2 small gather_mm). Here the outlier term is
# added to the accumulator fragments right before tq_act: one pass reads x once and decodes both weight tiles.
# xo (M, NO) fp32 = x_raw[ocol] for the sorted rows; wog/wou (E, N, NO) = the bundle's own wout (fp16), widened per
# element (exact) -- no resident fp32 copy (2026-10-12: Naive kept 0.66 GiB of them, 2x the bundle's 0.35 GiB wout).
_EPI = """
    if constexpr (JT_NO > 0) {
      for (short f = 0; f < decltype(Gt)::kNumFrags; ++f)
        for (short e = 0; e < decltype(Gt)::kElemsPerFrag; ++e) {
          const short2 pos = BaseNAXFrag::get_coord(e);
          const int row = tm + (f / TN) * 16 + pos.y; const int col = tn + (f % TN) * 16 + pos.x;
          if (row < tgp_bm && col < tgp_bn) {
            const device float* xr_ = xo + size_t(y_row + row) * JT_NO;
            const size_t wb = (size_t(index) * N + y_col + col) * JT_NO;
            float ag = 0.0f, au = 0.0f;
            for (int j = 0; j < JT_NO; ++j) { ag += xr_[j] * float(wog[wb + j]); au += xr_[j] * float(wou[wb + j]); }
            Gt.val_frags[f][e] += ag; Ut.val_frags[f][e] += au;
          }
        }
    }
"""


def _impl_out(K, W: int, NO: int, wtname: str = "half") -> str:
    s = _impl(K, W).replace(f"jt_gather_qmm_nax_K{_tag(K)}(", f"jt_gather_qmm_nax_out_K{_tag(K)}_n{NO}(", 1)
    s = s.replace(f"jt_h32_radix_K{_tag(K)}", f"jt_h32_radix_K{_tag(K)}_o{NO}").replace(f"jt_act_K{_tag(K)}(", f"jt_act_K{_tag(K)}_o{NO}(")
    a = "const int M, const int N, const int K, const float lim,"
    assert s.count(a) == 1
    s = s.replace(a, a + f" const device float* xo, const device {wtname}* wog, const device {wtname}* wou,")
    b = "    if (FUSED) {\n      for (short i = 0; i < decltype(Gt)::kNumFrags; i++)"
    assert s.count(b) == 1, "epilogue anchor"
    s = s.replace(b, _EPI + b)
    return s.replace("JT_NO", str(NO))  # literal: Metal forbids a program-scope constexpr here


@functools.lru_cache(maxsize=None)
def _kernel_out(K, W: int, NO: int, tname: str, expert: bool = False, h32: bool = False, wtname: str = "half"):
    header = JK._mlx_headers(["steel/gemm/gemm.h", "steel/gemm/nax.h", "steel/gemm/loader.h", "quantized_nax.h"]) + \
        _loader(K, 12, W) + _impl_out(K, W, NO, wtname)
    src = f"""
  constexpr int BK_padded = (64 + 16 / sizeof({tname}));
  threadgroup {tname} Wg[64 * BK_padded];
  threadgroup {tname} Wu[64 * BK_padded];
  jt_gather_qmm_nax_out_K{_tag(K)}_n{NO}<{tname}, 2, true, {'true' if expert else 'false'}, {'true' if h32 else 'false'}>(
      x, wg, sg, wu, su, indices, y, meta[0], meta[1], meta[2], lim[0], xo, wog, wou, Wg, Wu,
      threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup{_tail_args(expert, h32)});
"""
    return mx.fast.metal_kernel(name=f"jt_qmm_nax_out_K{_tag(K)}_w{W}_n{NO}_{_variant(True, expert, h32)}_{tname}_wo{wtname}",
                                input_names=["x", "wg", "sg", "wu", "su", "indices", "meta", "lim", "xo", "wog", "wou"]
                                + _tail_inputs(expert, h32),
                                output_names=["y"], header=header, source=src)


_WOUT_TNAME = {mx.float16: "half", mx.bfloat16: "bfloat16_t", mx.float32: "float"}


def gather_gate_up_sorted_jt(x, xo, Pg: dict, Pu: dict, idx_sorted, limit: float = 0.0, plan=None,
                             rotate_output: bool = False):
    """h = act(g, u) for expert-sorted rows in ONE NAX pass (act = SwiGLU, clamped at `limit` when > 0 like JANGH tq_act). x (M, D) bf16 ROTATED rows; xo (M, NO) fp32 raw-basis
    outlier activations (NO = Pg n_out; gate and up share the layer-wide columns). Returns (M, I) bf16. plan /
    rotate_output: as gather_qmm_sorted_jt."""
    M, Kin = x.shape; E, N, W = Pg["packed"].shape; NO = int(Pg["n_out"])
    idx = _pad_idx(idx_sorted)
    meta_x, gy, extra = _launch(M, N, E, plan, rotate_output)
    wog, wou = Pg["wout"], Pu["wout"]
    if wou.dtype != wog.dtype:
        wou = wou.astype(wog.dtype)
    k = _kernel_out(Pg["K"], W, NO, JK._TNAME[x.dtype], plan is not None, bool(rotate_output), _WOUT_TNAME[wog.dtype])
    return k(inputs=[x, Pg["packed"], Pg["scale"], Pu["packed"], Pu["scale"], idx, mx.array([M, N, Kin] + meta_x, dtype=mx.int32),
                     mx.array([float(limit)], dtype=mx.float32), xo.astype(mx.float32), wog, wou] + extra,
             grid=(((N + 63) // 64) * 128, gy, 1), threadgroup=(128, 1, 1),
             output_shapes=[(M, N)], output_dtypes=[x.dtype])[0]
