"""Faster JANGT trellis decode kernels for few-row MoE decode / verify (MiniMax-M3.1 widths), behind a lever.

Production path (``routed_fast`` / ``routed_gate_up`` / ``routed_down``; JTMixedSwitchGLU.routed and the fused M3
MoE block call it with VMLX_JANGT_FAST=1, default off): same math as moe_kernels' v3 decode kernels -- 16-weight lane
chunks of 12-bit trellis states, half2 codes ((s * MU) & MASK) | ORB, half2 fma chain per lane chunk, fp32 fold, code
statistics and row scales in the epilogue, outlier columns -- with
  * the activation side packed once per row in ONE launch (``pack_rows1``: Hadamard-128 rotation, fp16 range guard,
    half2 conversion, even / odd sums); v3 redid the guard and conversion in every threadgroup;
  * the code statistics folded into the per-chunk fold (one fp32 accumulator per output row and projection instead
    of two: half the simd_sum reductions); outlier columns added to the row accumulator before its one simd_sum;
  * every output row's window words loaded up front per block, states extracted with extract_bits;
  * down: the slot combine inside the kernel for one row (one launch less), per-slot + combine kernel for 2+ rows.
NOT bit-identical to v3 (decode and verify switch together). Every row's arithmetic is independent of the other rows
in the launch: decode steps == verify rows (tests/test_jangt_fast_decode.py), and the float64-reference error equals
v3's (bench/jangt_decode_error.py).

Measured (M5 Max, 57 M3.1 layers, input packing included; 06-PY-VERIFY-KERNELS.md 2026-10-29): one row gate / up
K 1 / 1.5 / 2: 5.23 / 5.40 / 4.94 -> 3.89 / 4.06 / 4.91 ms; down K 1 / 1.5 / 2 / 2.5: 4.40 / 3.54 / 3.25 / 3.65 ->
3.46 / 2.82 / 3.19 / 3.58 ms; 16 rows -25 to -34 %. At one row the kernels sit between two floors: the per-state ALU
work (~1.6 T states / s, K-independent) and DRAM (~540 GB/s): K <= 1.5 is ALU-bound (K 1 cannot pass ~400 GB/s with
this code), K >= 2 byte-bound (82 % of peak at K 2).

Experiments kept for the record (bench/jangt_decode_profile.py; measured slower or neutral): ``gen_gate_up`` /
``gen_down`` (the first generator, with ablation switches), window words from one coalesced load + simd_shuffle,
a threadgroup-memory code table (2x slower: bank conflicts), prefetching the next block's words, splitting K across
simdgroups, 8 output rows per simdgroup (spills), and decode-once-per-expert grouping (``gate_up4g`` / ``down4g``:
bit-identical to the ungrouped kernels, -15 % only when all rows share experts, slower on real routing).
"""

from __future__ import annotations

import functools
import os

import mlx.core as mx
import numpy as np

from vmlx_engine.jangt import moe_kernels as MK
from vmlx_engine.jangt import v2l as VL
from vmlx_engine.jangt.v2core import MASK, ORB

_UNROLL = '_Pragma("clang loop unroll(full)")'
_ODT = {mx.bfloat16: "bfloat16_t", mx.float16: "half", mx.float32: "float"}


def enabled() -> bool:
    """VMLX_JANGT_FAST=1 (default off until the real-bundle KL check; read per call)."""
    return os.environ.get("VMLX_JANGT_FAST", "0").strip().lower() in {"1", "true", "on", "yes"}


def _lit(v: float) -> str:
    return f"as_type<float>({int(np.array(v, np.float32).view(np.uint32))}u)"


# ------------------------------------------------------------------------------------------------ packed rows
@functools.lru_cache(maxsize=None)
def _pack_kernel(kin: int, xdt: str, rotate: bool, per_row_signs: bool, guard: bool):
    """One simdgroup per (row, 512-value block): optional Hadamard-128 rotation of x * s (rotate_rows' arithmetic),
    then per 16-value lane chunk the fp16 range guard of moe_kernels._xload3 -> half2 x 8 (packed as uint32 x 8),
    the inverse scale, and the chunk's even / odd sums (reduced per row by ``pack_sums``)."""
    sidx = "sidx[r]" if per_row_signs else "0u"
    rot = ""
    if rotate:
        # 4 Hadamard-128 blocks per 512-value block; lane j of block hb holds values 4j .. 4j + 3 (rotate_rows)
        rot = f"""
    const device float* sv = s + (size_t)({sidx}) * {kin}u;
    float v[16];
    {_UNROLL} for (uint hb = 0; hb < 4u; hb++) {{
      const uint base = blk * 512u + hb * 128u + lane * 4u;
      float w4[4];
      for (uint i = 0; i < 4u; i++) w4[i] = float(x[(size_t)r * {kin}u + base + i]) * sv[base + i];
      {{ float a = w4[0], b = w4[1]; w4[0] = a + b; w4[1] = a - b; a = w4[2]; b = w4[3]; w4[2] = a + b; w4[3] = a - b; }}
      {{ float a = w4[0], b = w4[2]; w4[0] = a + b; w4[2] = a - b; a = w4[1]; b = w4[3]; w4[1] = a + b; w4[3] = a - b; }}
      for (uint d = 1; d < 32u; d <<= 1) {{
        bool upper = (lane & d) != 0u;
        for (uint i = 0; i < 4u; i++) {{ float o = simd_shuffle_xor(w4[i], d); w4[i] = upper ? (o - w4[i]) : (w4[i] + o); }}
      }}
      for (uint i = 0; i < 4u; i++) v[hb * 4u + i] = w4[i] * 0.08838834764831845f;
    }}
    // v[hb * 4 + i] is value hb * 128 + lane * 4 + i of the block; lane chunk c (16 values) = values 16c .. 16c + 15:
    // gather it through threadgroup memory.
    threadgroup float tv[512];
    {_UNROLL} for (uint hb = 0; hb < 4u; hb++) for (uint i = 0; i < 4u; i++) tv[hb * 128u + lane * 4u + i] = v[hb * 4u + i];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float xt[16];
    {_UNROLL} for (uint i = 0; i < 16u; i++) xt[i] = tv[lane * 16u + i];"""
    else:
        rot = f"""
    float xt[16];
    {_UNROLL} for (uint i = 0; i < 16u; i++) xt[i] = float(x[(size_t)r * {kin}u + blk * 512u + lane * 16u + i]);"""
    guard_src = ("float xm = 0.0f; " + f"{_UNROLL} for (uint i = 0; i < 16u; i++) xm = metal::max(xm, metal::abs(xt[i]));"
                 " float xs = xm > 1024.0f ? metal::exp2(-metal::ceil(metal::log2(xm * (1.0f / 1024.0f)))) : 1.0f;"
                 if guard else "float xs = 1.0f;")
    src = f"""
    uint lane = thread_index_in_simdgroup; uint blk = threadgroup_position_in_grid.x; uint r = threadgroup_position_in_grid.y;
    {rot}
    {guard_src}
    float se = 0.0f, so = 0.0f;
    uint q[8];
    {_UNROLL} for (uint i = 0; i < 8u; i++) {{
      half2 h = half2(half(xt[2u * i] * xs), half(xt[2u * i + 1u] * xs));
      q[i] = as_type<uint>(h); se += xt[2u * i]; so += xt[2u * i + 1u];
    }}
    const size_t c = (size_t)r * {kin // 16}u + blk * 32u + lane;
    {_UNROLL} for (uint i = 0; i < 8u; i++) xq[c * 8u + i] = q[i];
    xinv[c] = 1.0f / xs;
    xsum[c * 2u] = se; xsum[c * 2u + 1u] = so;
    """
    ins = ["x"] + (["s"] if rotate else []) + (["sidx"] if per_row_signs else [])
    return mx.fast.metal_kernel(name=f"jtf_pack_k{kin}_{xdt}_r{int(rotate)}{int(per_row_signs)}_g{int(guard)}",
                                input_names=ins, output_names=["xq", "xinv", "xsum"], source=src)


@functools.lru_cache(maxsize=None)
def _sum_kernel(kin: int):
    """Per row: SE, SO = the chunk sums added in chunk order (fixed per row)."""
    n = kin // 16
    src = f"""
    uint r = thread_position_in_grid.x;
    float se = 0.0f, so = 0.0f;
    for (uint c = 0; c < {n}u; c++) {{ se += xsum[((size_t)r * {n}u + c) * 2u]; so += xsum[((size_t)r * {n}u + c) * 2u + 1u]; }}
    st[r * 2u] = se; st[r * 2u + 1u] = so;
    """
    return mx.fast.metal_kernel(name=f"jtf_sums_k{kin}", input_names=["xsum"], output_names=["st"], source=src)


def pack_rows(x, s=None, sidx=None, guard: bool = True):
    """x (R, kin) -> (xq (R, kin / 2) uint32 = half2 codes of the (rotated, guarded) inputs, xinv (R, kin / 16) f32,
    st (R, 2) f32 = even / odd input sums). ``s``: rotate rows of x * s (s (kin,) or (E, kin) gathered by sidx)."""
    R, kin = x.shape
    assert kin % 512 == 0, kin
    rotate = s is not None
    ins = [x] + ([s] if rotate else []) + ([sidx.astype(mx.uint32)] if sidx is not None else [])
    xq, xinv, xsum = _pack_kernel(kin, MK._dt(x), rotate, sidx is not None, guard)(
        inputs=ins, grid=(32 * (kin // 512), R, 1), threadgroup=(32, 1, 1),
        output_shapes=[(R, kin // 2), (R, kin // 16), (R, kin // 16, 2)],
        output_dtypes=[mx.uint32, mx.float32, mx.float32])
    st = _sum_kernel(kin)(inputs=[xsum], grid=(R, 1, 1), threadgroup=(min(R, 32), 1, 1), output_shapes=[(R, 2)],
                          output_dtypes=[mx.float32])[0]
    return xq, xinv, st


# ------------------------------------------------------------------------------------------------ trellis dot
def _block_words(K, L: int) -> int:
    """Words of one 512-value block a simdgroup needs (the last lane's window reaches into the next block)."""
    PL, max_sh, nw = MK._window(K, L)
    return ((31 * PL) >> 5) + nw


def _window_shuffle(K, L: int, wrow: str) -> str:
    """The lane's window from ONE coalesced load per lane (word `lane` of the block, plus word 32 + lane where the
    block needs more than 32 words) and simd_shuffles: every word is read once per simdgroup instead of by 2-4
    overlapping scalar loads."""
    PL, max_sh, nw = MK._window(K, L)
    so = VL.step_offsets(K, 9)
    span = int(so[7]) + L
    nwb = _block_words(K, L)
    loads = f"uint wa = lane < {nwb}u ? {wrow}[lane] : 0u;"
    if nwb > 32:
        loads += f" uint wb = lane < {nwb - 32}u ? {wrow}[32u + lane] : 0u;"
    need = nw if max_sh else (2 if span > 32 else 1)
    words = []
    for k in range(need):
        idx = f"((lb >> 5) + {k}u)"
        if nwb > 32:
            words.append(f"uint w{k} = {idx} < 32u ? simd_shuffle(wa, {idx}) : simd_shuffle(wb, {idx} - 32u);")
        else:
            words.append(f"uint w{k} = simd_shuffle(wa, {idx});")
    src = loads + " " + " ".join(words)
    if max_sh == 0:
        return src + " uint a0 = w0;" + (" uint a1 = w1;" if span > 32 else " uint a1 = 0u;")
    if span <= 32:
        return src + " uint a0 = sh == 0u ? w0 : ((w0 >> sh) | (w1 << (32u - sh))); uint a1 = 0u;"
    return src + (" uint a0 = sh == 0u ? w0 : ((w0 >> sh) | (w1 << (32u - sh)));"
                  " uint a1 = sh == 0u ? w1 : ((w1 >> sh) | (w2 << (32u - sh)));")


def _window_packed(K, L: int, wr: str) -> str:
    """The lane's window words with ONE packed (4-byte aligned) vector load: packed_uint2 / packed_uint3."""
    PL, max_sh, nw = MK._window(K, L)
    so = VL.step_offsets(K, 9)
    span = int(so[7]) + L
    need = nw if max_sh else (2 if span > 32 else 1)
    if need == 1:
        ld = f"uint w0 = {wr}[0];"
    elif need == 2:
        ld = f"packed_uint2 pw = *((const device packed_uint2*)({wr})); uint w0 = pw[0], w1 = pw[1];"
    else:
        ld = f"packed_uint3 pw = *((const device packed_uint3*)({wr})); uint w0 = pw[0], w1 = pw[1], w2 = pw[2];"
    return ld + " " + _realign(K, L, [f"w{k}" for k in range(need)])


def _window(K, L: int, wexpr, const: bool) -> str:
    if const:                                             # ablation: no weight loads (words vary per block / row)
        return "uint a0 = (lane * 2654435761u) ^ (k0 * 40503u) ^ uint(size_t(wr) & 0xffffu), a1 = a0 ^ 0x9e3779b9u;"
    PL, max_sh, nw = MK._window(K, L)
    so = VL.step_offsets(K, 9)
    span = int(so[7]) + L
    if max_sh == 0:
        return f"uint a0 = {wexpr(0)};" + (f" uint a1 = {wexpr(1)};" if span > 32 else " uint a1 = 0u;")
    if span <= 32:
        return (f"uint w0 = {wexpr(0)}, w1 = {wexpr(1)}; uint a0 = sh == 0u ? w0 : ((w0 >> sh) | (w1 << (32u - sh)));"
                " uint a1 = 0u;")
    return (f"uint w0 = {wexpr(0)}, w1 = {wexpr(1)}, w2 = {wexpr(2)}; "
            f"uint a0 = sh == 0u ? w0 : ((w0 >> sh) | (w1 << (32u - sh))); uint a1 = sh == 0u ? w1 : ((w1 >> sh) | (w2 << (32u - sh)));")


def _realign(K, L: int, w: list) -> str:
    """a0 / a1 from the window words w (names) -- moe_kernels._window_load's realignment."""
    PL, max_sh, nw = MK._window(K, L)
    so = VL.step_offsets(K, 9)
    span = int(so[7]) + L
    if max_sh == 0:
        return f"uint a0 = {w[0]};" + (f" uint a1 = {w[1]};" if span > 32 else " uint a1 = 0u;")
    if span <= 32:
        return f"uint a0 = sh == 0u ? {w[0]} : (({w[0]} >> sh) | ({w[1]} << (32u - sh))); uint a1 = 0u;"
    return (f"uint a0 = sh == 0u ? {w[0]} : (({w[0]} >> sh) | ({w[1]} << (32u - sh)));"
            f" uint a1 = sh == 0u ? {w[1]} : (({w[1]} >> sh) | ({w[2]} << (32u - sh)));")


def _nwin(K, L: int) -> int:
    PL, max_sh, nw = MK._window(K, L)
    so = VL.step_offsets(K, 9)
    span = int(so[7]) + L
    return nw if max_sh else (2 if span > 32 else 1)


def _extract(o: int, L: int, bfe: bool) -> str:
    if o + L <= 32:
        if bfe:
            return f"extract_bits(a0, {o}u, {L}u)"
        return f"((a0 >> {o}u) & {(1 << L) - 1}u)" if o else f"(a0 & {(1 << L) - 1}u)"
    if o >= 32:
        if bfe:
            return f"extract_bits(a1, {o - 32}u, {L}u)"
        return f"((a1 >> {o - 32}u) & {(1 << L) - 1}u)"
    return f"(((a0 >> {o}u) | (a1 << {32 - o}u)) & {(1 << L) - 1}u)"


_ABL = {"fma": "fma"}      # ablation switch (bench only): "xor" replaces the half2 fma by an integer xor


def _tdot(K, L: int, acc: str, xh: str, hashop: str, bfe: bool) -> str:
    so = VL.step_offsets(K, 9)
    MU = VL.MULTS[K]
    M2 = MASK | (MASK << 16)
    O2 = ORB | (ORB << 16)
    st = []
    for j in range(8):
        s = _extract(int(so[j]), L, bfe)
        if hashop == "mul":
            p = f"((({s}) * {MU}u) & {M2}u) | {O2}u"
        else:                                             # ablation: no hash
            p = f"(({s}) | ({s} << 16)) & {M2}u | {O2}u"
        if _ABL["fma"] == "skeleton":
            if j == 0:
                st.append(f"{acc} = as_type<half2>(as_type<uint>({acc}) ^ a0 ^ a1);")
            continue
        if _ABL["fma"] == "extract_only":
            st.append(f"{acc} = as_type<half2>(as_type<uint>({acc}) ^ ({s}));")
            continue
        if hashop == "lut":
            st.append(f"{acc} = fma({xh}[{j}], lut[{s}], {acc});")
            continue
        if _ABL["fma"] == "xor":
            st.append(f"{acc} = as_type<half2>(as_type<uint>({acc}) ^ ({p}));")
        else:
            st.append(f"{acc} = fma({xh}[{j}], as_type<half2>({p}), {acc});")
    return " ".join(st)


# ------------------------------------------------------------------------------------------------ gate / up
@functools.lru_cache(maxsize=None)
def gen_gate_up(kin: int, N: int, K, L: int, n_out: int, W: int, xdt: str, odt: str, limit: float, act: str,
                alpha: float, O: int = 4, NSG: int = 2, hashop: str = "mul", bfe: bool = True, wconst: bool = False,
                fold: str = "one", wload: str = "shuffle", wsmall: bool = False, abl: str = "fma"):
    """Gate / up + SwiGLU for (T * k) slots from packed rows. Grid (32 * NSG, k * T, N / (NSG * O)): slot on the fast
    axis (repeated experts come from cache), NSG simdgroups x O output rows per threadgroup."""
    _ABL["fma"] = abl
    mu, isd = VL.code_stats(L, K)
    PL, adv, _ = MK._lane_geom(K, L)
    R = range(O)
    if fold == "one":
        acc_decl = f"float g[{O}], u[{O}]; {_UNROLL} for (uint r = 0; r < {O}u; r++) {{ g[r] = 0.0f; u[r] = 0.0f; }}"
        fold_src = ("g[r] = fma(float(ag.x), x0, g[r]); g[r] = fma(float(ag.y), x1, g[r]);"
                    " u[r] = fma(float(au.x), x0, u[r]); u[r] = fma(float(au.y), x1, u[r]);")
        scale_src = f"const float x0 = xi * {_lit(isd[0])}, x1 = xi * {_lit(isd[1])};"
    else:
        acc_decl = (f"float g0[{O}], g1[{O}], u0[{O}], u1[{O}]; {_UNROLL} for (uint r = 0; r < {O}u; r++) "
                    f"{{ g0[r] = 0.0f; g1[r] = 0.0f; u0[r] = 0.0f; u1[r] = 0.0f; }}")
        fold_src = ("g0[r] += float(ag.x) * xi; g1[r] += float(ag.y) * xi; u0[r] += float(au.x) * xi;"
                    " u1[r] += float(au.y) * xi;")
        scale_src = ""
    if wload == "prefetch" and not wconst:
        # window words of block b + 1 loaded while block b decodes (two register sets, block loop unrolled by 2)
        NW = _nwin(K, L)
        projs = (("g", "wp", "ag"), ("u", "up", "au"))

        def loads(dst, off):
            return " ".join(f"{dst}{pn}{r}_{k} = {pp}[{r * W} + {off} + {k}];" for pn, pp, _ in projs for r in range(O)
                            for k in range(NW))

        def decode(src):
            out = []
            for r in range(O):
                out.append("{ half2 ag = half2(0.0h), au = half2(0.0h);")
                for pn, _, acc in projs:
                    out.append("{ " + _realign(K, L, [f"{src}{pn}{r}_{k}" for k in range(NW)]) + " "
                               + _tdot(K, L, acc, "xh", hashop, bfe) + " }")
                out.append("{ const uint r = " + str(r) + "u; " + fold_src + " } }")
            return " ".join(out)

        xload = f"""const uint4 q0 = xq4[0], q1 = xq4[1];
      half2 xh[8] = {{as_type<half2>(q0.x), as_type<half2>(q0.y), as_type<half2>(q0.z), as_type<half2>(q0.w),
                      as_type<half2>(q1.x), as_type<half2>(q1.y), as_type<half2>(q1.z), as_type<half2>(q1.w)}};
      const float xi = xinv_p[0]; {scale_src}"""
        decl = " ".join(f"uint A{pn}{r}_{k}, B{pn}{r}_{k};" for pn, _, _ in projs for r in range(O) for k in range(NW))
        nb = kin // 512
        assert nb % 2 == 0, kin
        loop = f"""{decl}
    {loads("A", "0")}
    for (uint b = 0; b < {nb}u; b += 2u) {{
      {loads("B", f"{adv}u")}
      {{ {xload} {decode("A")} }}
      xq4 += 64u; xinv_p += 32u; wp += {adv}u; up += {adv}u;
      if (b + 2u < {nb}u) {{ {loads("A", f"{adv}u")} }}
      {{ {xload} {decode("B")} }}
      xq4 += 64u; xinv_p += 32u; wp += {adv}u; up += {adv}u;
    }}"""
        win_g = None
    elif wload == "shuffle" and not wconst:
        win_g = _window_shuffle(K, L, "wr")
        base_g, base_u = "wb_g", "wb_u"
    elif wload == "packed" and not wconst:
        win_g = _window_packed(K, L, "wr")
        base_g, base_u = "wp", "up"
    else:
        win_g = _window(K, L, lambda i: f"wr[{i}]", wconst)
        base_g, base_u = "wp", "up"
    if win_g is not None:
        body = f"""
      const uint4 q0 = xq4[0], q1 = xq4[1];
      half2 xh[8] = {{as_type<half2>(q0.x), as_type<half2>(q0.y), as_type<half2>(q0.z), as_type<half2>(q0.w),
                      as_type<half2>(q1.x), as_type<half2>(q1.y), as_type<half2>(q1.z), as_type<half2>(q1.w)}};
      const float xi = xinv_p[0]; {scale_src}
      {_UNROLL} for (uint r = 0; r < {O}u; r++) {{
        half2 ag = half2(0.0h), au = half2(0.0h);
        {{ const device uint32_t* wr = {base_g} + r * {W}u; {win_g} {_tdot(K, L, "ag", "xh", hashop, bfe)} }}
        {{ const device uint32_t* wr = {base_u} + r * {W}u; {win_g} {_tdot(K, L, "au", "xh", hashop, bfe)} }}
        {fold_src}
      }}"""
        advance = f"wp += {adv}u; up += {adv}u; wb_g += {adv}u; wb_u += {adv}u; xq4 += 64u; xinv_p += 32u;"
        loop = f"""for (uint k0 = 0; k0 < {kin}u; k0 += 512u) {{
      {body}
      {advance}
    }}"""
    if fold == "one":
        c0 = f"({_lit(mu[0])} * {_lit(isd[0])} * SE + {_lit(mu[1])} * {_lit(isd[1])} * SO)"
        outl_lane = "" if n_out == 0 else f"""
    if (lane < {n_out}u) {{ int c = ocol[(size_t)e * {n_out}u + lane];
      if (c >= 0) {{ float xo = float(x[(size_t)t * {kin}u + uint(c)]);
        {_UNROLL} for (uint r = 0; r < {O}u; r++) {{
          g[r] += xo * float(woutg[((size_t)e * {N}u + row0 + r) * {n_out}u + lane]) / sg[(size_t)e * {N}u + row0 + r];
          u[r] += xo * float(woutu[((size_t)e * {N}u + row0 + r) * {n_out}u + lane]) / su_[(size_t)e * {N}u + row0 + r]; }} }} }}"""
        epi = f"""{outl_lane}
    const float SE = st[t * 2u], SO = st[t * 2u + 1u]; const float cst = {c0};
    {_UNROLL} for (uint r = 0; r < {O}u; r++) {{
      float G = simd_sum(g[r]), U = simd_sum(u[r]);
      if (lane == 0u) {{
        float gg = (G - cst) * sg[(size_t)e * {N}u + row0 + r];
        float uu = (U - cst) * su_[(size_t)e * {N}u + row0 + r];
        {MK.act_clamp_metal(limit, "gg", "uu")}
        out[(size_t)ts * {N}u + row0 + r] = {odt}({MK.act_value_metal(act, alpha, "gg", "uu")});
      }}
    }}"""
    else:
        outl_pre = "" if n_out == 0 else f"""
    float og[{O}], ou[{O}];
    {_UNROLL} for (uint r = 0; r < {O}u; r++) {{ og[r] = 0.0f; ou[r] = 0.0f; }}
    if (lane < {n_out}u) {{ int c = ocol[(size_t)e * {n_out}u + lane];
      if (c >= 0) {{ float xo = float(x[(size_t)t * {kin}u + uint(c)]);
        {_UNROLL} for (uint r = 0; r < {O}u; r++) {{ og[r] = xo * float(woutg[((size_t)e * {N}u + row0 + r) * {n_out}u + lane]);
                                                    ou[r] = xo * float(woutu[((size_t)e * {N}u + row0 + r) * {n_out}u + lane]); }} }} }}
    {_UNROLL} for (uint r = 0; r < {O}u; r++) {{ og[r] = simd_sum(og[r]); ou[r] = simd_sum(ou[r]); }}"""
        outl = "" if n_out == 0 else " gg += og[r]; uu += ou[r];"
        epi = f"""{outl_pre}
    const float SE = st[t * 2u], SO = st[t * 2u + 1u];
    {_UNROLL} for (uint r = 0; r < {O}u; r++) {{
      float G0 = simd_sum(g0[r]), G1 = simd_sum(g1[r]), U0 = simd_sum(u0[r]), U1 = simd_sum(u1[r]);
      if (lane == 0u) {{
        float gg = ((G0 - {_lit(mu[0])} * SE) * {_lit(isd[0])} + (G1 - {_lit(mu[1])} * SO) * {_lit(isd[1])}) * sg[(size_t)e * {N}u + row0 + r];
        float uu = ((U0 - {_lit(mu[0])} * SE) * {_lit(isd[0])} + (U1 - {_lit(mu[1])} * SO) * {_lit(isd[1])}) * su_[(size_t)e * {N}u + row0 + r];{outl}
        {MK.act_clamp_metal(limit, "gg", "uu")}
        out[(size_t)ts * {N}u + row0 + r] = {odt}({MK.act_value_metal(act, alpha, "gg", "uu")});
      }}
    }}"""
    lut_src = "" if hashop != "lut" else f"""
    threadgroup half2 lut[{1 << L}];
    for (uint i = thread_index_in_threadgroup; i < {1 << L}u; i += {32 * NSG}u)
      lut[i] = as_type<half2>(((i * {VL.MULTS[K]}u) & {MASK | (MASK << 16)}u) | {ORB | (ORB << 16)}u);
    threadgroup_barrier(mem_flags::mem_threadgroup);"""
    src = f"""
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;{lut_src}
    uint row0 = threadgroup_position_in_grid.z * {NSG * O}u + sgi * {O}u; uint ts = threadgroup_position_in_grid.y;
    uint t = ts / {'KS'}; uint e = idx[ts];
    const uint lb = lane * {PL}u; const uint sh = lb & 31u; (void)sh;
    const device uint32_t* wp = wg + ((size_t){'0u' if wsmall else 'e'} * {N}u + {'(row0 & 7u)' if wsmall else 'row0'}) * {W}u + (lb >> 5);
    const device uint32_t* up = wu + ((size_t){'0u' if wsmall else 'e'} * {N}u + {'(row0 & 7u)' if wsmall else 'row0'}) * {W}u + (lb >> 5);
    const device uint32_t* wb_g = wg + ((size_t)e * {N}u + row0) * {W}u; const device uint32_t* wb_u = wu + ((size_t)e * {N}u + row0) * {W}u;
    (void)wp; (void)up; (void)wb_g; (void)wb_u;
    const device uint4* xq4 = (const device uint4*)(xq + (size_t)t * {kin // 2}u) + lane * 2u;
    const device float* xinv_p = xinv + (size_t)t * {kin // 16}u + lane;
    {acc_decl}
    {loop}
    {epi}
    """
    ins = ["xq", "xinv", "st", "x", "idx", "wg", "sg", "wu", "su_"] + (["ocol", "woutg", "woutu"] if n_out else [])
    name = (f"jtf_gu_k{kin}_n{N}_K{MK._kname(K)}_o{n_out}_w{W}_{xdt}_{odt}{MK.act_tag(act, limit, alpha)}_O{O}_S{NSG}"
            f"_{hashop}_{int(bfe)}{int(wconst)}_{fold}_{wload}{'_ws' if wsmall else ''}_{abl}")
    _ABL["fma"] = "fma"
    return mx.fast.metal_kernel(name=name, input_names=ins, output_names=["out"], source=src)


def gate_up(packed, x, idx, U, out_dtype=mx.float32, limit: float = 0.0, act: str = "silu", alpha: float = 0.0,
            **opts):
    """(T * k, N) SwiGLU activations of every (row, slot) from ``packed`` = pack_rows(x, su_gu)."""
    xq, xinv, st = packed
    T, kin = x.shape
    k = idx.shape[1]
    N = U["gate"]["scale"].shape[1]
    n_out = U["gate"]["n_out"]
    W = int(U["gate"]["packed"].shape[-1])
    O, NSG = opts.get("O", 4), opts.get("NSG", 2)
    kern = gen_gate_up(kin, N, U["K"], U["L"], n_out, W, MK._dt(x), _ODT[out_dtype], float(limit), MK.check_act(act),
                       float(alpha) if act == "oai" else 0.0, **opts)
    ins = [xq, xinv, st, x, idx.reshape(-1).astype(mx.uint32), U["gate"]["packed"], U["gate"]["scale"],
           U["up"]["packed"], U["up"]["scale"]] + ([U["gate"]["ocol"], U["gate"]["wout"], U["up"]["wout"]] if n_out else [])
    return kern(inputs=ins, template=[("KS", k)], grid=(32 * NSG, T * k, N // (NSG * O)), threadgroup=(32 * NSG, 1, 1),
                output_shapes=[(T * k, N)], output_dtypes=[out_dtype])[0]


# ------------------------------------------------------------------------------------------------ down
@functools.lru_cache(maxsize=None)
def gen_down(kin: int, N: int, K, L: int, n_out: int, W: int, hdt: str, O: int = 4, NSG: int = 2,
             hashop: str = "mul", bfe: bool = True, wconst: bool = False, fold: str = "one", combine: bool = True,
             wload: str = "shuffle"):
    """Down projection of every (row, slot) from packed slot rows; ``combine``: each threadgroup owns O x NSG output
    rows of one ROW and loops over its k slots (acc = fma(w, y, acc), slot order), else writes per-slot y."""
    mu, isd = VL.code_stats(L, K)
    PL, adv, _ = MK._lane_geom(K, L)
    shuffle = wload == "shuffle" and not wconst
    if wload == "packed" and not wconst:
        win = _window_packed(K, L, "wr")
    else:
        win = _window_shuffle(K, L, "wr") if shuffle else _window(K, L, lambda i: f"wr[{i}]", wconst)
    base_d = "wb_d" if shuffle else "wp"
    if fold == "one":
        acc_decl = f"float d[{O}]; {_UNROLL} for (uint r = 0; r < {O}u; r++) d[r] = 0.0f;"
        fold_src = "d[r] = fma(float(ad.x), x0, d[r]); d[r] = fma(float(ad.y), x1, d[r]);"
        scale_src = f"const float x0 = xi * {_lit(isd[0])}, x1 = xi * {_lit(isd[1])};"
    else:
        acc_decl = f"float d0[{O}], d1[{O}]; {_UNROLL} for (uint r = 0; r < {O}u; r++) {{ d0[r] = 0.0f; d1[r] = 0.0f; }}"
        fold_src = "d0[r] += float(ad.x) * xi; d1[r] += float(ad.y) * xi;"
        scale_src = ""
    body = f"""
        const uint4 q0 = xq4[0], q1 = xq4[1];
        half2 xh[8] = {{as_type<half2>(q0.x), as_type<half2>(q0.y), as_type<half2>(q0.z), as_type<half2>(q0.w),
                        as_type<half2>(q1.x), as_type<half2>(q1.y), as_type<half2>(q1.z), as_type<half2>(q1.w)}};
        const float xi = xinv_p[0]; {scale_src}
        {_UNROLL} for (uint r = 0; r < {O}u; r++) {{
          half2 ad = half2(0.0h);
          {{ const device uint32_t* wr = {base_d} + r * {W}u; {win} {_tdot(K, L, "ad", "xh", hashop, bfe)} }}
          {fold_src}
        }}
        wp += {adv}u; wb_d += {adv}u; xq4 += 64u; xinv_p += 32u;"""
    if fold == "one":
        outl = "" if n_out == 0 else f"""
      if (lane < {n_out}u) {{ int c = ocol[(size_t)e * {n_out}u + lane];
        if (c >= 0) {{ float ho = float(h[(size_t)ts * {kin}u + uint(c)]);
          {_UNROLL} for (uint r = 0; r < {O}u; r++) d[r] += ho * float(wout[((size_t)e * {N}u + row0 + r) * {n_out}u + lane]) / sd[(size_t)e * {N}u + row0 + r]; }} }}"""
        yval = f"(simd_sum(d[r]) - cst) * sd[(size_t)e * {N}u + row0 + r]"
        cst = f"const float cst = {_lit(mu[0])} * {_lit(isd[0])} * st[ts * 2u] + {_lit(mu[1])} * {_lit(isd[1])} * st[ts * 2u + 1u];"
    else:
        outl = "" if n_out == 0 else f"""
      float od[{O}];
      {_UNROLL} for (uint r = 0; r < {O}u; r++) od[r] = 0.0f;
      if (lane < {n_out}u) {{ int c = ocol[(size_t)e * {n_out}u + lane];
        if (c >= 0) {{ float ho = float(h[(size_t)ts * {kin}u + uint(c)]);
          {_UNROLL} for (uint r = 0; r < {O}u; r++) od[r] = ho * float(wout[((size_t)e * {N}u + row0 + r) * {n_out}u + lane]); }} }}
      {_UNROLL} for (uint r = 0; r < {O}u; r++) od[r] = simd_sum(od[r]);"""
        yval = (f"((simd_sum(d0[r]) - {_lit(mu[0])} * st[ts * 2u]) * {_lit(isd[0])} + (simd_sum(d1[r]) - {_lit(mu[1])} * "
                f"st[ts * 2u + 1u]) * {_lit(isd[1])}) * sd[(size_t)e * {N}u + row0 + r]" + (" + od[r]" if n_out else ""))
        cst = ""
    slot = f"""
      uint e = idx[ts];
      const device uint32_t* wp = wd + ((size_t)e * {N}u + row0) * {W}u + (lb >> 5);
      const device uint32_t* wb_d = wd + ((size_t)e * {N}u + row0) * {W}u; (void)wp; (void)wb_d;
      const device uint4* xq4 = (const device uint4*)(xq + (size_t)ts * {kin // 2}u) + lane * 2u;
      const device float* xinv_p = xinv + (size_t)ts * {kin // 16}u + lane;
      {acc_decl}
      for (uint k0 = 0; k0 < {kin}u; k0 += 512u) {{ {body} }}{outl}
      {cst}"""
    if combine:
        src = f"""
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row0 = threadgroup_position_in_grid.z * {NSG * O}u + sgi * {O}u; uint t = threadgroup_position_in_grid.y;
    const uint lb = lane * {PL}u; const uint sh = lb & 31u; (void)sh;
    float acc[{O}];
    {_UNROLL} for (uint r = 0; r < {O}u; r++) acc[r] = 0.0f;
    for (uint sl = 0; sl < uint(KS); sl++) {{
      uint ts = t * uint(KS) + sl; float wgt = float(rw[ts]);
      {slot}
      {_UNROLL} for (uint r = 0; r < {O}u; r++) {{ float y = {yval}; acc[r] = fma(wgt, y, acc[r]); }}
    }}
    if (lane == 0u) {{ {_UNROLL} for (uint r = 0; r < {O}u; r++) out[(size_t)t * {N}u + row0 + r] = acc[r]; }}
    """
        ins = ["xq", "xinv", "st", "h", "idx", "rw", "wd", "sd"] + (["ocol", "wout"] if n_out else [])
    else:
        src = f"""
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row0 = threadgroup_position_in_grid.z * {NSG * O}u + sgi * {O}u; uint ts = threadgroup_position_in_grid.y;
    const uint lb = lane * {PL}u; const uint sh = lb & 31u; (void)sh;
    {slot}
    {_UNROLL} for (uint r = 0; r < {O}u; r++) {{ float y = {yval}; if (lane == 0u) out[(size_t)ts * {N}u + row0 + r] = y; }}
    """
        ins = ["xq", "xinv", "st", "h", "idx", "wd", "sd"] + (["ocol", "wout"] if n_out else [])
    name = (f"jtf_dn_k{kin}_n{N}_K{MK._kname(K)}_o{n_out}_w{W}_{hdt}_O{O}_S{NSG}_{hashop}_{int(bfe)}{int(wconst)}_{fold}"
            f"_c{int(combine)}_{wload}")
    return mx.fast.metal_kernel(name=name, input_names=ins, output_names=["out"], source=src)


def down_weighted(packed, h, idx, rw, U, **opts):
    """(T, D) f32 router-weighted down projection from ``packed`` = pack_rows(h, sv_dn, idx)."""
    xq, xinv, st = packed
    T, k = idx.shape
    kin = h.shape[1]
    N = U["down"]["scale"].shape[1]
    n_out = U["down"]["n_out"]
    W = int(U["down"]["packed"].shape[-1])
    O, NSG = opts.get("O", 4), opts.get("NSG", 2)
    kern = gen_down(kin, N, U["K_dn"], U["L"], n_out, W, MK._dt(h), **opts)
    ins = [xq, xinv, st, h, idx.reshape(-1).astype(mx.uint32), rw.reshape(-1).astype(mx.float32), U["down"]["packed"],
           U["down"]["scale"]] + ([U["down"]["ocol"], U["down"]["wout"]] if n_out else [])
    return kern(inputs=ins, template=[("KS", k)], grid=(32 * NSG, T, N // (NSG * O)), threadgroup=(32 * NSG, 1, 1),
                output_shapes=[(T, N)], output_dtypes=[mx.float32])[0]


def routed(x2, idx2, w2, U, limit, act, alpha, gu_opts=None, dn_opts=None):
    """The whole routed block for (T, D) rows: pack -> gate / up -> pack (rotated with each slot's down scales) ->
    weighted down. (T, D) f32."""
    gu_opts, dn_opts = gu_opts or {}, dn_opts or {}
    h = gate_up(pack_rows(x2, U["su_gu"]), x2, idx2, U, out_dtype=mx.float32, limit=limit, act=act, alpha=alpha,
                **gu_opts)
    return down_weighted(pack_rows(h, U["sv_dn"], idx2.reshape(-1)), h, idx2, w2, U, **dn_opts)


# =================================================================================================== v4 kernels
# One generator for gate / up and down. Threadgroup = NOG output-row simdgroups x KSPLIT K-part simdgroups; each
# simdgroup decodes O output rows over its share of the 512-value blocks; K parts meet in threadgroup memory (added
# in part order) before the epilogue. Window words: "scalar" (per-word loads, v3), "packed" (one packed_uint2 /
# packed_uint3 load), "prefetch" (next block's words loaded while this block decodes).
def _xload4() -> str:
    return """const uint4 q0 = xq4[0], q1 = xq4[1];
        half2 xh[8] = {as_type<half2>(q0.x), as_type<half2>(q0.y), as_type<half2>(q0.z), as_type<half2>(q0.w),
                       as_type<half2>(q1.x), as_type<half2>(q1.y), as_type<half2>(q1.z), as_type<half2>(q1.w)};
        const float xi = xinv_p[0]; const float x0 = xi * ISD0, x1 = xi * ISD1;"""


def _chunk_loads(K, L, wload, ptr, dst, r, W, off):
    """Statements loading the window words of output row r of `ptr` at word offset `off` into dst names."""
    nw = _nwin(K, L)
    if wload == "packed" and nw >= 2:
        t = "packed_uint2" if nw == 2 else "packed_uint3"
        return (f"{{ {t} pw = *((const device {t}*)({ptr} + {r * W} + {off})); "
                + " ".join(f"{dst}[{k}] = pw[{k}];" for k in range(nw)) + " }")
    return " ".join(f"{dst}[{k}] = {ptr}[{r * W} + {off} + {k}];" for k in range(nw))


def _decode_rows(K, L, projs, O, src_names, fold_lines):
    """Decode + fold of O rows for each projection from window word arrays src_names[(proj, r)]."""
    nw = _nwin(K, L)
    out = []
    for r in range(O):
        for pn, acc_name in projs:
            words = [f"{src_names[(pn, r)]}[{k}]" for k in range(nw)]
            out.append(f"{{ half2 a_ = half2(0.0h); {{ {_realign(K, L, words)} {_tdot(K, L, 'a_', 'xh', 'mul', True)} }} "
                       f"{fold_lines(acc_name, r)} }}")
    return "\n        ".join(out)


def _main_loop(K, L, kin, projs, ptrs, O, W, adv, wload, nbp):
    """The K-part loop (nbp blocks) for this simdgroup; accumulators acc_<proj>[r]."""
    nw = _nwin(K, L)
    fold = lambda acc, r: f"{acc}[{r}] = fma(float(a_.x), x0, {acc}[{r}]); {acc}[{r}] = fma(float(a_.y), x1, {acc}[{r}]);"  # noqa: E731
    names = {(pn, r): f"w{pn}{r}" for pn, _ in projs for r in range(O)}
    decl = " ".join(f"uint {names[(pn, r)]}[{nw}];" for pn, _ in projs for r in range(O))
    if wload == "prefetch":
        names_b = {k: v + "b" for k, v in names.items()}
        decl += " " + " ".join(f"uint {names_b[(pn, r)]}[{nw}];" for pn, _ in projs for r in range(O))

        def loads(nm, off):
            return " ".join(_chunk_loads(K, L, "packed", ptrs[pn], nm[(pn, r)], r, W, off) for pn, _ in projs
                            for r in range(O))
        body_a = _decode_rows(K, L, projs, O, names, fold)
        body_b = _decode_rows(K, L, projs, O, names_b, fold)
        adv_all = " ".join(f"{ptrs[pn]} += {adv}u;" for pn, _ in projs)
        return f"""{decl}
    {loads(names, "0")}
    uint b = 0;
    for (; b + 1u < nbp; b += 2u) {{
      {loads(names_b, str(adv))}
      {{ {_xload4()} {body_a} }}
      xq4 += 64u; xinv_p += 32u; {adv_all}
      if (b + 2u < nbp) {{ {loads(names, str(adv))} }}
      {{ {_xload4()} {body_b} }}
      xq4 += 64u; xinv_p += 32u; {adv_all}
    }}
    if (b < nbp) {{ {_xload4()} {body_a} }}"""
    loads_now = " ".join(_chunk_loads(K, L, wload, ptrs[pn], names[(pn, r)], r, W, "0") for pn, _ in projs
                         for r in range(O))
    body = _decode_rows(K, L, projs, O, names, fold)
    adv_all = " ".join(f"{ptrs[pn]} += {adv}u;" for pn, _ in projs)
    return f"""{decl}
    for (uint b = 0; b < nbp; b++) {{
      {loads_now}
      {{ {_xload4()} {body} }}
      xq4 += 64u; xinv_p += 32u; {adv_all}
    }}"""


@functools.lru_cache(maxsize=None)
def gen_gu4(kin: int, N: int, K, L: int, n_out: int, W: int, xdt: str, odt: str, limit: float, act: str, alpha: float,
            O: int = 4, NOG: int = 2, KSPLIT: int = 1, wload: str = "scalar"):
    mu, isd = VL.code_stats(L, K)
    PL, adv, _ = MK._lane_geom(K, L)
    NB = kin // 512
    assert kin % 512 == 0 and 1 <= KSPLIT <= NB, (kin, KSPLIT)
    nbp = None
    projs = (("g", "g"), ("u", "u"))
    loop = _main_loop(K, L, kin, projs, {"g": "wp", "u": "up"}, O, W, adv, wload, nbp)
    outl = "" if n_out == 0 else f"""
    if (kp == 0u && lane < {n_out}u) {{ int c = ocol[(size_t)e * {n_out}u + lane];
      if (c >= 0) {{ float xo = float(x[(size_t)t * {kin}u + uint(c)]);
        {_UNROLL} for (uint r = 0; r < {O}u; r++) {{
          g[r] += xo * float(woutg[((size_t)e * {N}u + row0 + r) * {n_out}u + lane]) / sg[(size_t)e * {N}u + row0 + r];
          u[r] += xo * float(woutu[((size_t)e * {N}u + row0 + r) * {n_out}u + lane]) / su_[(size_t)e * {N}u + row0 + r]; }} }} }}"""
    if KSPLIT == 1:
        reduce = f"float G[{O}], U[{O}]; {_UNROLL} for (uint r = 0; r < {O}u; r++) {{ G[r] = simd_sum(g[r]); U[r] = simd_sum(u[r]); }}"
    else:
        reduce = f"""threadgroup float part[{KSPLIT} * {NOG} * {O} * 2];
    {_UNROLL} for (uint r = 0; r < {O}u; r++) {{
      float gs_ = simd_sum(g[r]), us_ = simd_sum(u[r]);
      if (lane == 0u) {{ part[((kp * {NOG}u + og) * {O}u + r) * 2u] = gs_; part[((kp * {NOG}u + og) * {O}u + r) * 2u + 1u] = us_; }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kp != 0u) return;
    float G[{O}], U[{O}];
    {_UNROLL} for (uint r = 0; r < {O}u; r++) {{
      G[r] = part[(og * {O}u + r) * 2u]; U[r] = part[(og * {O}u + r) * 2u + 1u];
      for (uint q = 1; q < {KSPLIT}u; q++) {{ G[r] += part[((q * {NOG}u + og) * {O}u + r) * 2u]; U[r] += part[((q * {NOG}u + og) * {O}u + r) * 2u + 1u]; }}
    }}"""
    src = f"""
    const float ISD0 = {_lit(isd[0])}, ISD1 = {_lit(isd[1])};
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint og = sgi % {NOG}u, kp = sgi / {NOG}u;
    uint row0 = threadgroup_position_in_grid.z * {NOG * O}u + og * {O}u; uint ts = threadgroup_position_in_grid.y;
    uint t = ts / KS; uint e = idx[ts];
    const uint nbp = {NB // KSPLIT}u + (kp < {NB % KSPLIT}u ? 1u : 0u);
    const uint b0 = kp * {NB // KSPLIT}u + metal::min(kp, {NB % KSPLIT}u);
    const uint lb = lane * {PL}u; const uint sh = lb & 31u; (void)sh;
    const device uint32_t* wp = wg + ((size_t)e * {N}u + row0) * {W}u + (lb >> 5) + b0 * {adv}u;
    const device uint32_t* up = wu + ((size_t)e * {N}u + row0) * {W}u + (lb >> 5) + b0 * {adv}u;
    const device uint4* xq4 = (const device uint4*)(xq + (size_t)t * {kin // 2}u) + lane * 2u + b0 * 64u;
    const device float* xinv_p = xinv + (size_t)t * {kin // 16}u + lane + b0 * 32u;
    float g[{O}], u[{O}];
    {_UNROLL} for (uint r = 0; r < {O}u; r++) {{ g[r] = 0.0f; u[r] = 0.0f; }}
    {loop}
    {outl}
    {reduce}
    const float SE = st[t * 2u], SO = st[t * 2u + 1u];
    const float cst = {_lit(mu[0])} * ISD0 * SE + {_lit(mu[1])} * ISD1 * SO;
    {_UNROLL} for (uint r = 0; r < {O}u; r++) {{
      if (lane == 0u) {{
        float gg = (G[r] - cst) * sg[(size_t)e * {N}u + row0 + r];
        float uu = (U[r] - cst) * su_[(size_t)e * {N}u + row0 + r];
        {MK.act_clamp_metal(limit, "gg", "uu")}
        out[(size_t)ts * {N}u + row0 + r] = {odt}({MK.act_value_metal(act, alpha, "gg", "uu")});
      }}
    }}
    """
    ins = ["xq", "xinv", "st", "x", "idx", "wg", "sg", "wu", "su_"] + (["ocol", "woutg", "woutu"] if n_out else [])
    name = (f"jtf4_gu_k{kin}_n{N}_K{MK._kname(K)}_o{n_out}_w{W}_{xdt}_{odt}{MK.act_tag(act, limit, alpha)}"
            f"_O{O}_G{NOG}_S{KSPLIT}_{wload}")
    return mx.fast.metal_kernel(name=name, input_names=ins, output_names=["out"], source=src)


def gate_up4(packed, x, idx, U, out_dtype=mx.float32, limit: float = 0.0, act: str = "silu", alpha: float = 0.0,
             O: int = 4, NOG: int = 2, KSPLIT: int = 1, wload: str = "scalar"):
    xq, xinv, st = packed
    T, kin = x.shape
    k = idx.shape[1]
    N = U["gate"]["scale"].shape[1]
    n_out = U["gate"]["n_out"]
    W = int(U["gate"]["packed"].shape[-1])
    kern = gen_gu4(kin, N, U["K"], U["L"], n_out, W, MK._dt(x), _ODT[out_dtype], float(limit), MK.check_act(act),
                   float(alpha) if act == "oai" else 0.0, O, NOG, KSPLIT, wload)
    ins = [xq, xinv, st, x, idx.reshape(-1).astype(mx.uint32), U["gate"]["packed"], U["gate"]["scale"],
           U["up"]["packed"], U["up"]["scale"]] + ([U["gate"]["ocol"], U["gate"]["wout"], U["up"]["wout"]] if n_out else [])
    tpg = 32 * NOG * KSPLIT
    return kern(inputs=ins, template=[("KS", k)], grid=(tpg, T * k, N // (NOG * O)), threadgroup=(tpg, 1, 1),
                output_shapes=[(T * k, N)], output_dtypes=[out_dtype])[0]


@functools.lru_cache(maxsize=None)
def gen_dn4(kin: int, N: int, K, L: int, n_out: int, W: int, hdt: str, O: int = 4, NOG: int = 2, KSPLIT: int = 1,
            wload: str = "scalar"):
    """Per-slot down projection y (before the router weight): grid (32 NOG KSPLIT, T k, N / (NOG O))."""
    mu, isd = VL.code_stats(L, K)
    PL, adv, _ = MK._lane_geom(K, L)
    NB = kin // 512
    assert kin % 512 == 0 and 1 <= KSPLIT <= NB, (kin, KSPLIT)
    nbp = None
    loop = _main_loop(K, L, kin, (("d", "d"),), {"d": "wp"}, O, W, adv, wload, nbp)
    outl = "" if n_out == 0 else f"""
    if (kp == 0u && lane < {n_out}u) {{ int c = ocol[(size_t)e * {n_out}u + lane];
      if (c >= 0) {{ float ho = float(h[(size_t)ts * {kin}u + uint(c)]);
        {_UNROLL} for (uint r = 0; r < {O}u; r++) d[r] += ho * float(wout[((size_t)e * {N}u + row0 + r) * {n_out}u + lane]) / sd[(size_t)e * {N}u + row0 + r]; }} }}"""
    if KSPLIT == 1:
        reduce = f"float Dv[{O}]; {_UNROLL} for (uint r = 0; r < {O}u; r++) Dv[r] = simd_sum(d[r]);"
    else:
        reduce = f"""threadgroup float part[{KSPLIT} * {NOG} * {O}];
    {_UNROLL} for (uint r = 0; r < {O}u; r++) {{ float ds_ = simd_sum(d[r]); if (lane == 0u) part[(kp * {NOG}u + og) * {O}u + r] = ds_; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kp != 0u) return;
    float Dv[{O}];
    {_UNROLL} for (uint r = 0; r < {O}u; r++) {{
      Dv[r] = part[og * {O}u + r]; for (uint q = 1; q < {KSPLIT}u; q++) Dv[r] += part[(q * {NOG}u + og) * {O}u + r]; }}"""
    src = f"""
    const float ISD0 = {_lit(isd[0])}, ISD1 = {_lit(isd[1])};
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint og = sgi % {NOG}u, kp = sgi / {NOG}u;
    uint row0 = threadgroup_position_in_grid.z * {NOG * O}u + og * {O}u; uint ts = threadgroup_position_in_grid.y;
    uint e = idx[ts];
    const uint nbp = {NB // KSPLIT}u + (kp < {NB % KSPLIT}u ? 1u : 0u);
    const uint b0 = kp * {NB // KSPLIT}u + metal::min(kp, {NB % KSPLIT}u);
    const uint lb = lane * {PL}u; const uint sh = lb & 31u; (void)sh;
    const device uint32_t* wp = wd + ((size_t)e * {N}u + row0) * {W}u + (lb >> 5) + b0 * {adv}u;
    const device uint4* xq4 = (const device uint4*)(xq + (size_t)ts * {kin // 2}u) + lane * 2u + b0 * 64u;
    const device float* xinv_p = xinv + (size_t)ts * {kin // 16}u + lane + b0 * 32u;
    float d[{O}];
    {_UNROLL} for (uint r = 0; r < {O}u; r++) d[r] = 0.0f;
    {loop}
    {outl}
    {reduce}
    const float cst = {_lit(mu[0])} * ISD0 * st[ts * 2u] + {_lit(mu[1])} * ISD1 * st[ts * 2u + 1u];
    {_UNROLL} for (uint r = 0; r < {O}u; r++) if (lane == 0u) out[(size_t)ts * {N}u + row0 + r] = (Dv[r] - cst) * sd[(size_t)e * {N}u + row0 + r];
    """
    ins = ["xq", "xinv", "st", "h", "idx", "wd", "sd"] + (["ocol", "wout"] if n_out else [])
    name = f"jtf4_dn_k{kin}_n{N}_K{MK._kname(K)}_o{n_out}_w{W}_{hdt}_O{O}_G{NOG}_S{KSPLIT}_{wload}"
    return mx.fast.metal_kernel(name=name, input_names=ins, output_names=["out"], source=src)


@functools.lru_cache(maxsize=None)
def _combine4(N: int, kslots: int):
    src = f"""
    uint n = thread_position_in_grid.x, t = thread_position_in_grid.y;
    if (n >= {N}u) return;
    float acc = 0.0f;
    for (uint sl = 0; sl < {kslots}u; sl++) {{ uint ts = t * {kslots}u + sl; acc = fma(float(rw[ts]), ysl[(size_t)ts * {N}u + n], acc); }}
    out[(size_t)t * {N}u + n] = acc;
    """
    return mx.fast.metal_kernel(name=f"jtf4_comb_n{N}_s{kslots}", input_names=["ysl", "rw"], output_names=["out"],
                                source=src)


def down4(packed, h, idx, rw, U, O: int = 4, NOG: int = 2, KSPLIT: int = 1, wload: str = "scalar", combine=True):
    """(T, D) f32 router-weighted down projection (per-slot kernel + slot-ordered combine)."""
    xq, xinv, st = packed
    T, k = idx.shape
    kin = h.shape[1]
    N = U["down"]["scale"].shape[1]
    n_out = U["down"]["n_out"]
    W = int(U["down"]["packed"].shape[-1])
    kern = gen_dn4(kin, N, U["K_dn"], U["L"], n_out, W, MK._dt(h), O, NOG, KSPLIT, wload)
    ins = [xq, xinv, st, h, idx.reshape(-1).astype(mx.uint32), U["down"]["packed"], U["down"]["scale"]] + \
          ([U["down"]["ocol"], U["down"]["wout"]] if n_out else [])
    tpg = 32 * NOG * KSPLIT
    ysl = kern(inputs=ins, grid=(tpg, T * k, N // (NOG * O)), threadgroup=(tpg, 1, 1), output_shapes=[(T * k, N)],
               output_dtypes=[mx.float32])[0]
    if not combine:
        return ysl
    tg = min(256, N)
    return _combine4(N, k)(inputs=[ysl, rw.reshape(-1).astype(mx.float32)], grid=(((N + tg - 1) // tg) * tg, T, 1),
                           threadgroup=(tg, 1, 1), output_shapes=[(T, N)], output_dtypes=[mx.float32])[0]


# ------------------------------------------------------------------------------------------------ one-launch pack
@functools.lru_cache(maxsize=None)
def _pack1_kernel(kin: int, xdt: str, per_row_signs: bool, guard: bool):
    """pack_rows in ONE launch: one threadgroup per row, one simdgroup per 512-value block (rotate_rows' Hadamard-128
    arithmetic, the fp16 range guard per 16-value lane chunk, half2 codes), the row's even / odd sums reduced in
    threadgroup memory in block order."""
    nb = kin // 512
    sidx = "sidx[r]" if per_row_signs else "0u"
    guard_src = ("float xm = 0.0f; " + f"{_UNROLL} for (uint i = 0; i < 16u; i++) xm = metal::max(xm, metal::abs(xt[i]));"
                 " float xs = xm > 1024.0f ? metal::exp2(-metal::ceil(metal::log2(xm * (1.0f / 1024.0f)))) : 1.0f;"
                 if guard else "float xs = 1.0f;")
    src = f"""
    uint lane = thread_index_in_simdgroup; uint blk = simdgroup_index_in_threadgroup; uint r = threadgroup_position_in_grid.y;
    threadgroup float tv[{nb} * 512];
    threadgroup float red[{nb} * 2];
    const device float* sv = s + (size_t)({sidx}) * {kin}u;
    {_UNROLL} for (uint hb = 0; hb < 4u; hb++) {{
      const uint base = blk * 512u + hb * 128u + lane * 4u;
      float w4[4];
      for (uint i = 0; i < 4u; i++) w4[i] = float(x[(size_t)r * {kin}u + base + i]) * sv[base + i];
      {{ float a = w4[0], b = w4[1]; w4[0] = a + b; w4[1] = a - b; a = w4[2]; b = w4[3]; w4[2] = a + b; w4[3] = a - b; }}
      {{ float a = w4[0], b = w4[2]; w4[0] = a + b; w4[2] = a - b; a = w4[1]; b = w4[3]; w4[1] = a + b; w4[3] = a - b; }}
      for (uint d = 1; d < 32u; d <<= 1) {{
        bool upper = (lane & d) != 0u;
        for (uint i = 0; i < 4u; i++) {{ float o = simd_shuffle_xor(w4[i], d); w4[i] = upper ? (o - w4[i]) : (w4[i] + o); }}
      }}
      for (uint i = 0; i < 4u; i++) tv[base + i] = w4[i] * 0.08838834764831845f;
    }}
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float xt[16];
    {_UNROLL} for (uint i = 0; i < 16u; i++) xt[i] = tv[blk * 512u + lane * 16u + i];
    {guard_src}
    float se = 0.0f, so = 0.0f;
    uint q[8];
    {_UNROLL} for (uint i = 0; i < 8u; i++) {{
      half2 h = half2(half(xt[2u * i] * xs), half(xt[2u * i + 1u] * xs));
      q[i] = as_type<uint>(h); se += xt[2u * i]; so += xt[2u * i + 1u];
    }}
    const size_t c = (size_t)r * {kin // 16}u + blk * 32u + lane;
    {_UNROLL} for (uint i = 0; i < 8u; i++) xq[c * 8u + i] = q[i];
    xinv[c] = 1.0f / xs;
    se = simd_sum(se); so = simd_sum(so);
    if (lane == 0u) {{ red[blk * 2u] = se; red[blk * 2u + 1u] = so; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (blk == 0u && lane == 0u) {{
      float a = 0.0f, b = 0.0f;
      for (uint k = 0; k < {nb}u; k++) {{ a += red[k * 2u]; b += red[k * 2u + 1u]; }}
      st[r * 2u] = a; st[r * 2u + 1u] = b;
    }}
    """
    ins = ["x", "s"] + (["sidx"] if per_row_signs else [])
    return mx.fast.metal_kernel(name=f"jtf_pack1_k{kin}_{xdt}_r{int(per_row_signs)}_g{int(guard)}",
                                input_names=ins, output_names=["xq", "xinv", "st"], source=src)


def pack_rows1(x, s, sidx=None, guard: bool = True):
    """As pack_rows (always rotating), in one launch: (xq, xinv, st)."""
    R, kin = x.shape
    assert kin % 512 == 0 and kin // 512 <= 32, kin
    nb = kin // 512
    ins = [x, s] + ([sidx.astype(mx.uint32)] if sidx is not None else [])
    return tuple(_pack1_kernel(kin, MK._dt(x), sidx is not None, guard)(
        inputs=ins, grid=(32 * nb, R, 1), threadgroup=(32 * nb, 1, 1),
        output_shapes=[(R, kin // 2), (R, kin // 16), (R, 2)], output_dtypes=[mx.uint32, mx.float32, mx.float32]))


# ------------------------------------------------------------------------------------------------ down, slot-combined in-kernel
@functools.lru_cache(maxsize=None)
def gen_dn4c(kin: int, N: int, K, L: int, n_out: int, W: int, hdt: str, O: int = 4, NOG: int = 2, KS: int = 4,
             wload: str = "scalar"):
    """Down projection + router-weighted combine in ONE launch: threadgroup = NOG output-row simdgroups x KS slot
    simdgroups for ONE row t; each simdgroup decodes its slot's expert for O output rows; the slot results meet in
    threadgroup memory and are combined in slot order (acc = 0, fma(w, y, acc))."""
    mu, isd = VL.code_stats(L, K)
    PL, adv, _ = MK._lane_geom(K, L)
    NB = kin // 512
    loop = _main_loop(K, L, kin, (("d", "d"),), {"d": "wp"}, O, W, adv, wload, None)
    outl = "" if n_out == 0 else f"""
    if (lane < {n_out}u) {{ int c = ocol[(size_t)e * {n_out}u + lane];
      if (c >= 0) {{ float ho = float(h[(size_t)ts * {kin}u + uint(c)]);
        {_UNROLL} for (uint r = 0; r < {O}u; r++) d[r] += ho * float(wout[((size_t)e * {N}u + row0 + r) * {n_out}u + lane]) / sd[(size_t)e * {N}u + row0 + r]; }} }}"""
    src = f"""
    const float ISD0 = {_lit(isd[0])}, ISD1 = {_lit(isd[1])};
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint og = sgi % {NOG}u, sl = sgi / {NOG}u;
    uint row0 = threadgroup_position_in_grid.z * {NOG * O}u + og * {O}u; uint t = threadgroup_position_in_grid.y;
    uint ts = t * {KS}u + sl; uint e = idx[ts];
    const uint nbp = {NB}u;
    threadgroup float ysl[{KS} * {NOG} * {O}];
    const uint lb = lane * {PL}u; const uint sh = lb & 31u; (void)sh;
    const device uint32_t* wp = wd + ((size_t)e * {N}u + row0) * {W}u + (lb >> 5);
    const device uint4* xq4 = (const device uint4*)(xq + (size_t)ts * {kin // 2}u) + lane * 2u;
    const device float* xinv_p = xinv + (size_t)ts * {kin // 16}u + lane;
    float d[{O}];
    {_UNROLL} for (uint r = 0; r < {O}u; r++) d[r] = 0.0f;
    {loop}
    {outl}
    const float cst = {_lit(mu[0])} * ISD0 * st[ts * 2u] + {_lit(mu[1])} * ISD1 * st[ts * 2u + 1u];
    {_UNROLL} for (uint r = 0; r < {O}u; r++) {{
      float Dv = simd_sum(d[r]);
      if (lane == 0u) ysl[(sl * {NOG}u + og) * {O}u + r] = (Dv - cst) * sd[(size_t)e * {N}u + row0 + r];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sl == 0u && lane < {O}u) {{
      float acc = 0.0f;
      for (uint q = 0; q < {KS}u; q++) acc = fma(float(rw[t * {KS}u + q]), ysl[(q * {NOG}u + og) * {O}u + lane], acc);
      out[(size_t)t * {N}u + row0 + lane] = acc;
    }}
    """
    ins = ["xq", "xinv", "st", "h", "idx", "rw", "wd", "sd"] + (["ocol", "wout"] if n_out else [])
    name = f"jtf4c_dn_k{kin}_n{N}_K{MK._kname(K)}_o{n_out}_w{W}_{hdt}_O{O}_G{NOG}_s{KS}_{wload}"
    return mx.fast.metal_kernel(name=name, input_names=ins, output_names=["out"], source=src)


def down4c(packed, h, idx, rw, U, O: int = 4, NOG: int = 2, wload: str = "scalar"):
    xq, xinv, st = packed
    T, k = idx.shape
    kin = h.shape[1]
    N = U["down"]["scale"].shape[1]
    n_out = U["down"]["n_out"]
    W = int(U["down"]["packed"].shape[-1])
    kern = gen_dn4c(kin, N, U["K_dn"], U["L"], n_out, W, MK._dt(h), O, NOG, k, wload)
    ins = [xq, xinv, st, h, idx.reshape(-1).astype(mx.uint32), rw.reshape(-1).astype(mx.float32), U["down"]["packed"],
           U["down"]["scale"]] + ([U["down"]["ocol"], U["down"]["wout"]] if n_out else [])
    tpg = 32 * NOG * k
    return kern(inputs=ins, grid=(tpg, T, N // (NOG * O)), threadgroup=(tpg, 1, 1), output_shapes=[(T, N)],
                output_dtypes=[mx.float32])[0]


# ------------------------------------------------------------------------------------------------ grouped (k rows)
def _decode_pv(K, L: int) -> str:
    """The 8 states of the realigned window a0 / a1 -> half2 pv[8] (the codes _tdot multiplies)."""
    so = VL.step_offsets(K, 9)
    MU = VL.MULTS[K]
    M2 = MASK | (MASK << 16)
    O2 = ORB | (ORB << 16)
    return " ".join(f"pv[{j}] = as_type<half2>((((({_extract(int(so[j]), L, True)}) * {MU}u) & {M2}u) | {O2}u));"
                    for j in range(8))


def _group_head4(RP: int) -> str:
    """This threadgroup's group of up to RP slots routed to the same expert (slot order; ranks that are multiples of
    RP lead, the rest exit): e, nr, ts0 .. ts{RP-1} (padding repeats ts0)."""
    picks = " else ".join(f"if (nr == {j}u) ts{j} = i;" for j in range(1, RP))
    decl = " ".join(f"uint ts{j} = s;" for j in range(RP))
    more = "" if RP == 1 else f"""
    for (uint i = s + 1u; i < NS && nr < {RP}u; i++) {{ if (idx[i] == e) {{ {picks} nr++; }} }}"""
    return f"""
    const uint s = threadgroup_position_in_grid.y;
    const uint e = idx[s];
    uint rank = 0u;
    for (uint i = 0u; i < s; i++) rank += (idx[i] == e) ? 1u : 0u;
    if (rank % {RP}u != 0u) return;
    {decl}
    uint nr = 1u;{more}"""


@functools.lru_cache(maxsize=None)
def gen_grouped4(part: str, kin: int, N: int, K, L: int, n_out: int, W: int, xdt: str, odt: str, limit: float,
                 act: str, alpha: float, O: int = 2, NOG: int = 4, RP: int = 4, wload: str = "scalar"):
    """Gate / up (part "gu") or per-slot down (part "dn") for groups of up to RP slots that picked the same expert:
    each window is decoded once per group and applied to every member's packed row. Grid (32 NOG, T k, N / (NOG O))."""
    mu, isd = VL.code_stats(L, K)
    PL, adv, _ = MK._lane_geom(K, L)
    nw = _nwin(K, L)
    projs = ("g", "u") if part == "gu" else ("d",)
    ptr = {"g": "wp", "u": "up", "d": "wp"}
    J = range(RP)
    row_of = (lambda j: f"(ts{j} / KS)") if part == "gu" else (lambda j: f"ts{j}")
    names = {(p, r): f"w{p}{r}" for p in projs for r in range(O)}
    decl_w = " ".join(f"uint {names[k]}[{nw}];" for k in names)
    loads = " ".join(_chunk_loads(K, L, wload, ptr[p], names[(p, r)], r, W, "0") for p in projs for r in range(O))
    xl = []
    for j in J:
        guard = "" if j == 0 else f"if ({j}u < nr) "
        xl.append(f"""half2 xh{j}[8]; float x0_{j} = 0.0f, x1_{j} = 0.0f;
        {guard}{{ const uint4 q0 = xq4_{j}[0], q1 = xq4_{j}[1];
          xh{j}[0] = as_type<half2>(q0.x); xh{j}[1] = as_type<half2>(q0.y); xh{j}[2] = as_type<half2>(q0.z); xh{j}[3] = as_type<half2>(q0.w);
          xh{j}[4] = as_type<half2>(q1.x); xh{j}[5] = as_type<half2>(q1.y); xh{j}[6] = as_type<half2>(q1.z); xh{j}[7] = as_type<half2>(q1.w);
          const float xi = xinv_{j}[0]; x0_{j} = xi * ISD0; x1_{j} = xi * ISD1; }}""")
    dec = []
    for r in range(O):
        for p in projs:
            words = [f"{names[(p, r)]}[{k}]" for k in range(nw)]
            apply = []
            for j in J:
                fma8 = " ".join(f"a_ = fma(xh{j}[{i}], pv[{i}], a_);" for i in range(8))
                body = (f"{{ half2 a_ = half2(0.0h); {fma8} {p}{j}[{r}] = fma(float(a_.x), x0_{j}, {p}{j}[{r}]);"
                        f" {p}{j}[{r}] = fma(float(a_.y), x1_{j}, {p}{j}[{r}]); }}")
                apply.append(body if j == 0 else f"if ({j}u < nr) {body}")
            dec.append(f"{{ {_realign(K, L, words)} half2 pv[8]; {_decode_pv(K, L)} {' '.join(apply)} }}")
    adv_x = " ".join(f"xq4_{j} += 64u; xinv_{j} += 32u;" for j in J)
    adv_w = " ".join(f"{q} += {adv}u;" for q in dict.fromkeys(ptr[p] for p in projs))
    loop = f"""{decl_w}
    for (uint b = 0; b < {kin // 512}u; b++) {{
      {loads}
      {{ {" ".join(xl)}
        {" ".join(dec)} }}
      {adv_x} {adv_w}
    }}"""
    acc_decl = " ".join(f"float {p}{j}[{O}];" for p in projs for j in J) + \
        f" {_UNROLL} for (uint r = 0; r < {O}u; r++) {{ " + " ".join(f"{p}{j}[r] = 0.0f;" for p in projs for j in J) + " }"
    xptr = " ".join(f"const device uint4* xq4_{j} = (const device uint4*)(xq + (size_t){row_of(j)} * {kin // 2}u) + lane * 2u;"
                    f" const device float* xinv_{j} = xinv + (size_t){row_of(j)} * {kin // 16}u + lane;" for j in J)
    epi = []
    for j in J:
        rj = row_of(j)
        if part == "gu":
            outl = "" if n_out == 0 else f"""
      if (lane < {n_out}u) {{ int c = ocol[(size_t)e * {n_out}u + lane];
        if (c >= 0) {{ float xo = float(x[(size_t){rj} * {kin}u + uint(c)]);
          {_UNROLL} for (uint r = 0; r < {O}u; r++) {{
            g{j}[r] += xo * float(woutg[((size_t)e * {N}u + row0 + r) * {n_out}u + lane]) / sg[(size_t)e * {N}u + row0 + r];
            u{j}[r] += xo * float(woutu[((size_t)e * {N}u + row0 + r) * {n_out}u + lane]) / su_[(size_t)e * {N}u + row0 + r]; }} }} }}"""
            body = f"""{{{outl}
      const float cst = {_lit(mu[0])} * ISD0 * st[{rj} * 2u] + {_lit(mu[1])} * ISD1 * st[{rj} * 2u + 1u];
      {_UNROLL} for (uint r = 0; r < {O}u; r++) {{
        float G = simd_sum(g{j}[r]), U = simd_sum(u{j}[r]);
        if (lane == 0u) {{
          float gg = (G - cst) * sg[(size_t)e * {N}u + row0 + r];
          float uu = (U - cst) * su_[(size_t)e * {N}u + row0 + r];
          {MK.act_clamp_metal(limit, "gg", "uu")}
          out[(size_t)ts{j} * {N}u + row0 + r] = {odt}({MK.act_value_metal(act, alpha, "gg", "uu")});
        }}
      }} }}"""
        else:
            outl = "" if n_out == 0 else f"""
      if (lane < {n_out}u) {{ int c = ocol[(size_t)e * {n_out}u + lane];
        if (c >= 0) {{ float ho = float(h[(size_t)ts{j} * {kin}u + uint(c)]);
          {_UNROLL} for (uint r = 0; r < {O}u; r++) d{j}[r] += ho * float(wout[((size_t)e * {N}u + row0 + r) * {n_out}u + lane]) / sd[(size_t)e * {N}u + row0 + r]; }} }}"""
            body = f"""{{{outl}
      const float cst = {_lit(mu[0])} * ISD0 * st[ts{j} * 2u] + {_lit(mu[1])} * ISD1 * st[ts{j} * 2u + 1u];
      {_UNROLL} for (uint r = 0; r < {O}u; r++) {{
        float Dv = simd_sum(d{j}[r]);
        if (lane == 0u) out[(size_t)ts{j} * {N}u + row0 + r] = (Dv - cst) * sd[(size_t)e * {N}u + row0 + r];
      }} }}"""
        epi.append(body if j == 0 else f"if ({j}u < nr) {body}")
    wdecl = (f"const device uint32_t* wp = wg + ((size_t)e * {N}u + row0) * {W}u + (lb >> 5);"
             f" const device uint32_t* up = wu + ((size_t)e * {N}u + row0) * {W}u + (lb >> 5);" if part == "gu" else
             f"const device uint32_t* wp = wd + ((size_t)e * {N}u + row0) * {W}u + (lb >> 5);")
    src = f"""
    const float ISD0 = {_lit(isd[0])}, ISD1 = {_lit(isd[1])};
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row0 = threadgroup_position_in_grid.z * {NOG * O}u + sgi * {O}u;{_group_head4(RP)}
    const uint lb = lane * {PL}u; const uint sh = lb & 31u; (void)sh;
    {wdecl}
    {xptr}
    {acc_decl}
    {loop}
    {chr(10).join(epi)}
    """
    if part == "gu":
        ins = ["xq", "xinv", "st", "x", "idx", "wg", "sg", "wu", "su_"] + (["ocol", "woutg", "woutu"] if n_out else [])
    else:
        ins = ["xq", "xinv", "st", "h", "idx", "wd", "sd"] + (["ocol", "wout"] if n_out else [])
    name = (f"jtf4g_{part}_k{kin}_n{N}_K{MK._kname(K)}_o{n_out}_w{W}_{xdt}_{odt}{MK.act_tag(act, limit, alpha) if part == 'gu' else ''}"
            f"_O{O}_G{NOG}_R{RP}_{wload}")
    return mx.fast.metal_kernel(name=name, input_names=ins, output_names=["out"], source=src)


def gate_up4g(packed, x, idx, U, out_dtype=mx.float32, limit: float = 0.0, act: str = "silu", alpha: float = 0.0,
              O: int = 2, NOG: int = 4, RP: int = 4, wload: str = "scalar"):
    xq, xinv, st = packed
    T, kin = x.shape
    k = idx.shape[1]
    N = U["gate"]["scale"].shape[1]
    n_out = U["gate"]["n_out"]
    W = int(U["gate"]["packed"].shape[-1])
    kern = gen_grouped4("gu", kin, N, U["K"], U["L"], n_out, W, MK._dt(x), _ODT[out_dtype], float(limit),
                        MK.check_act(act), float(alpha) if act == "oai" else 0.0, O, NOG, RP, wload)
    ins = [xq, xinv, st, x, idx.reshape(-1).astype(mx.uint32), U["gate"]["packed"], U["gate"]["scale"],
           U["up"]["packed"], U["up"]["scale"]] + ([U["gate"]["ocol"], U["gate"]["wout"], U["up"]["wout"]] if n_out else [])
    return kern(inputs=ins, template=[("KS", k), ("NS", T * k)], grid=(32 * NOG, T * k, N // (NOG * O)),
                threadgroup=(32 * NOG, 1, 1), output_shapes=[(T * k, N)], output_dtypes=[out_dtype])[0]


def down4g(packed, h, idx, rw, U, O: int = 2, NOG: int = 4, RP: int = 4, wload: str = "scalar"):
    xq, xinv, st = packed
    T, k = idx.shape
    kin = h.shape[1]
    N = U["down"]["scale"].shape[1]
    n_out = U["down"]["n_out"]
    W = int(U["down"]["packed"].shape[-1])
    kern = gen_grouped4("dn", kin, N, U["K_dn"], U["L"], n_out, W, MK._dt(h), "float", 0.0, "silu", 0.0, O, NOG, RP,
                        wload)
    ins = [xq, xinv, st, h, idx.reshape(-1).astype(mx.uint32), U["down"]["packed"], U["down"]["scale"]] + \
          ([U["down"]["ocol"], U["down"]["wout"]] if n_out else [])
    ysl = kern(inputs=ins, template=[("NS", T * k)], grid=(32 * NOG, T * k, N // (NOG * O)),
               threadgroup=(32 * NOG, 1, 1), output_shapes=[(T * k, N)], output_dtypes=[mx.float32])[0]
    tg = min(256, N)
    return _combine4(N, k)(inputs=[ysl, rw.reshape(-1).astype(mx.float32)], grid=(((N + tg - 1) // tg) * tg, T, 1),
                           threadgroup=(tg, 1, 1), output_shapes=[(T, N)], output_dtypes=[mx.float32])[0]


# =================================================================================================== production path
# Chosen from bench/jangt_decode_profile.py (M5 Max, M3.1 widths, see 06-PY-VERIFY-KERNELS.md 2026-10-29): gate / up
# O 4 x 2 simdgroups, per-word loads, every row count; down O 4 x 2 simdgroups with the slot combine in the kernel
# for one row (one launch less), per-slot + combine kernel for 2+ rows. Every configuration gives a row the same
# arithmetic (same lane chunks, fold, epilogue, slot-ordered combine), so a decode step and the same row inside a
# verify are bit-identical (tests/test_jangt_fast_decode.py).
GU_CFG = {"O": 4, "NOG": 2, "KSPLIT": 1, "wload": "scalar"}
DN_CFG = {"O": 4, "NOG": 2, "wload": "scalar"}


def supported(U, D: int, I: int) -> bool:
    """Both groups JANGT with input widths in whole 512-value blocks (M3.1: 6144 / 3072)."""
    return (U.get("gate") is not None and U.get("down") is not None and D % 512 == 0 and I % 512 == 0
            and D // 512 <= 32 and I // 512 <= 32 and (I // DN_CFG["NOG"]) and D % 8 == 0 and I % 8 == 0)


def routed_gate_up(x2, idx2, U, limit, act, alpha):
    """(T * k, I) f32 SwiGLU activations of every (row, slot)."""
    return gate_up4(pack_rows1(x2, U["su_gu"]), x2, idx2, U, out_dtype=mx.float32, limit=limit, act=act,
                    alpha=alpha, **GU_CFG)


def routed_down(h, idx2, w2, U):
    """(T, D) f32 router-weighted down projection of every row."""
    packed = pack_rows1(h, U["sv_dn"], idx2.reshape(-1))
    if idx2.shape[0] == 1:
        return down4c(packed, h, idx2, w2, U, **DN_CFG)
    return down4(packed, h, idx2, w2, U, KSPLIT=1, **DN_CFG)


def routed_fast(x2, idx2, w2, U, limit, act, alpha):
    """The routed block of JTMixedSwitchGLU.routed for decode / verify rows: (T, D) f32."""
    return routed_down(routed_gate_up(x2, idx2, U, limit, act, alpha), idx2, w2, U)
