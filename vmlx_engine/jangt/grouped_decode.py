"""Expert-grouped few-row JANGT decode: each routed expert's trellis weights decoded ONCE for every verify row that
picked it, every row's arithmetic identical to the per-(row, slot) v3 kernels (moe_kernels.gather_gate_up_swiglu_v3 /
gather_down_weighted_v3), so the outputs are bit-identical to them -- and therefore to one-token decode steps.

Why: a speculative verify of T rows routes T x k (row, expert) slots. The v3 kernels give every slot its own
threadgroups, so an expert picked by several rows is decoded once per row. Trellis decode is ALU-bound (measured on
M5 Max, M3.1 shapes, 57 layers: 16 rows that all pick the same 4 experts cost 99 ms vs 11 ms for one row), and real
consecutive tokens share experts (M3.1 coding row: 16 rows -> 27.7 distinct experts of 64 slots).

How: one threadgroup per group of up to RP slots that picked the same expert. Groups are formed on the GPU inside the
kernel, with no sort: the threadgroup of slot s ranks s among the slots with its expert (slot order); only ranks that
are multiples of RP lead a group, which takes the next (up to RP) slots with that expert; every other threadgroup
exits at once. Per 512-value block the group's rows load their activations exactly as v3 does (same fp16 range
guard, same se / so sums), then each output row's trellis window is loaded and its 8 steps decoded once; each row
accumulates the decoded half2 weights against its own activations in v3's step order (fma chain from 0 per lane
chunk), then folds into its fp32 accumulators with v3's expression. The epilogue (simd_sum, code statistics, row
scale, outlier columns, clamp, SwiGLU variant) is v3's, per row. Output rows per simdgroup only group the work: every
(row, output row) result is computed by its own lane accumulators, whatever the grouping.

The weighted down projection writes each slot's result (fp32) and ``slot_combine`` applies the router weights slot
by slot in v3's order (acc = 0, then fma(w, y, acc) per slot), so the weighted sum is bit-identical too.

Lever: VMLX_JANGT_GROUPED_DECODE=1 (default OFF: measured not faster, see grouped_enabled; read per call). Used by JTMixedSwitchGLU.routed for 2+ rows below
the sort threshold when gate/up and down are both JANGT. Bit-equality vs v3: tests/test_jangt_grouped_decode.py.
"""

from __future__ import annotations

import functools
import os

import mlx.core as mx

from vmlx_engine.jangt import moe_kernels as MK
from vmlx_engine.jangt import v2l as VL
from vmlx_engine.jangt.v2core import MASK, ORB

# Rows per group pass (register-resident per-row accumulators) and output rows per simdgroup. Measured on the M5 Max,
# M3.1 shapes (bench/minimax_m3_verify_parts.py --part routed): see GROUP_SHAPES.
DEFAULT_RP = 4
DEFAULT_OUT_ROWS = 4
_ODT = {mx.bfloat16: "bfloat16_t", mx.float16: "half", mx.float32: "float"}


def grouped_enabled() -> bool:
    """VMLX_JANGT_GROUPED_DECODE=1 (default OFF; read per call for in-process A/B).

    Measured 2026-10-28 (M5 Max, M3.1 shapes, 57 layers, gate/up, interleaved, min of 15): bit-identical to v3 but
    not faster -- 16 rows on 28 distinct experts 64.5 ms (v3) vs 67.0 / 69.2 / 101.3 / 110.2 (rows per pass 1 / 2 /
    2 / 4, output rows 4 / 2 / 4 / 4); even 16 rows on the SAME 4 experts 57.7 vs 52.9 at best. The kernels are
    occupancy-bound: every extra row's accumulators and activations live in registers, and a 4-row pass costs 2.2x a
    1-row pass at one row (15.8 vs 7.2 ms). Kept for A/B; the decode-once-apply-to-many idea needs the activations
    and partial sums out of registers (threadgroup-staged decoded tiles) to pay."""
    return os.environ.get("VMLX_JANGT_GROUPED_DECODE", "0").strip().lower() in {"1", "true", "on", "yes"}


def _rp_env() -> int:
    try:
        return max(1, int(os.environ.get("VMLX_JANGT_GROUP_ROWS", "") or DEFAULT_RP))
    except ValueError:
        return DEFAULT_RP


def _out_rows_env() -> int:
    try:
        o = int(os.environ.get("VMLX_JANGT_GROUP_OUT_ROWS", "") or DEFAULT_OUT_ROWS)
    except ValueError:
        o = DEFAULT_OUT_ROWS
    return o if o in (1, 2, 4) else DEFAULT_OUT_ROWS


def _group_head(RP: int) -> str:
    """Metal: this threadgroup's group -- e (expert), nr (rows), ts0..ts{RP-1} (slots; padding repeats ts0)."""
    picks = " else ".join(f"if (nr == {j}u) ts{j} = i;" for j in range(1, RP))
    decl = " ".join(f"uint ts{j} = s;" for j in range(RP))
    more = "" if RP == 1 else f"""
    for (uint i = s + 1u; i < NS && nr < {RP}u; i++) {{ if (idx[i] == e) {{ {picks} nr++; }} }}"""
    return f"""
    const uint s = threadgroup_position_in_grid.z;
    const uint e = idx[s];
    uint rank = 0u;
    for (uint i = 0u; i < s; i++) rank += (idx[i] == e) ? 1u : 0u;
    if (rank % {RP}u != 0u) return;
    {decl}
    uint nr = 1u;{more}"""


def _decode_pv(K, L: int) -> str:
    """Metal: the 8 trellis steps of the realigned window a0 / a1 into pv[0..7] -- the half2 values _tdot3 multiplies."""
    so = VL.step_offsets(K, 9)
    mk = (1 << L) - 1
    MU = VL.MULTS[K]
    M2 = MASK | (MASK << 16)
    O2 = ORB | (ORB << 16)
    st = []
    for j in range(8):
        ex = MK._step_extract(int(so[j]), L)
        st.append(f"{{ uint s_ = ({ex}) & {mk}u; pv[{j}] = as_type<half2>(((s_ * {MU}u) & {M2}u) | {O2}u); }}")
    return " ".join(st)


def _apply(xh: str, acc: str) -> str:
    """acc = fma over the 8 steps in step order: the products and order of _tdot3."""
    return " ".join(f"{acc} = fma({xh}[{i}], pv[{i}], {acc});" for i in range(8))


def _xload_row(j: int, guard: bool) -> str:
    """MK._xload3 for group row j: the same statements on xp{j}, into xh{j}, xi{j}, se{j}, so{j}."""
    src = MK._xload3(guard)
    src = (src.replace("xp[i]", f"xp{j}[i]").replace("xh[", f"xh{j}[").replace("se +=", f"se{j} +=")
           .replace("so +=", f"so{j} +=").replace("half2 xh{j}[8];".replace("{j}", str(j)), ""))
    if guard:
        src = src.replace("float xi = 1.0f / xs;", f"xi{j} = 1.0f / xs;")
    else:
        src = src.replace("const float xi = 1.0f;", f"xi{j} = 1.0f;")
    return "{ " + src + " }"


def _rows_guard(j: int, body: str) -> str:
    return body if j == 0 else f"if ({j}u < nr) {{ {body} }}"


@functools.lru_cache(maxsize=None)
def _gu_grouped(kin: int, N: int, K, L: int, n_out: int, W: int, xdt: str, odt: str, limit: float, guard: bool,
                act: str, alpha: float, RP: int, O: int):
    mu, isd = VL.code_stats(L, K)
    PL, adv, _ = MK._lane_geom(K, L)
    R = range(RP)
    decl = "\n    ".join(
        f"const device float* xp{j} = xr + (size_t)(ts{j} / KS) * {kin}u + lane * 16u; "
        f"float g0_{j}[{O}], g1_{j}[{O}], u0_{j}[{O}], u1_{j}[{O}]; "
        f"{MK._UNROLL} for (uint r = 0; r < {O}u; r++) {{ g0_{j}[r] = 0.0f; g1_{j}[r] = 0.0f; u0_{j}[r] = 0.0f; u1_{j}[r] = 0.0f; }} "
        f"float se{j} = 0.0f, so{j} = 0.0f, xi{j} = 1.0f; half2 xh{j}[8];" for j in R)
    loads = "\n      ".join(_rows_guard(j, _xload_row(j, guard)) for j in R)
    ld_g = MK._window_load(K, L, lambda i: f"wr[{i}]")
    gate = " ".join(_rows_guard(j, f"{{ half2 ag = half2(0.0h); {_apply(f'xh{j}', 'ag')} "
                                   f"g0_{j}[r] += float(ag.x) * xi{j}; g1_{j}[r] += float(ag.y) * xi{j}; }}") for j in R)
    upr = " ".join(_rows_guard(j, f"{{ half2 au = half2(0.0h); {_apply(f'xh{j}', 'au')} "
                                  f"u0_{j}[r] += float(au.x) * xi{j}; u1_{j}[r] += float(au.y) * xi{j}; }}") for j in R)
    body = f"""{loads}
      {MK._UNROLL} for (uint r = 0; r < {O}u; r++) {{
        {{ const device uint32_t* wr = wp + r * {W}u; {ld_g} half2 pv[8]; {_decode_pv(K, L)} {gate} }}
        {{ const device uint32_t* wr = up + r * {W}u; {ld_g} half2 pv[8]; {_decode_pv(K, L)} {upr} }}
      }}"""
    advance = f"wp += {adv}u; up += {adv}u;" + "".join(f" xp{j} += 512u;" for j in R)
    clamp = MK.act_clamp_metal(limit)
    hval = MK.act_value_metal(act, alpha)
    epi = []
    for j in R:
        outl_pre = "" if n_out == 0 else f"""
      float og[{O}], ou[{O}];
      {MK._UNROLL} for (uint r = 0; r < {O}u; r++) {{ og[r] = 0.0f; ou[r] = 0.0f; }}
      if (lane < {n_out}u) {{ int c = ocol[(size_t)e * {n_out}u + lane];
        if (c >= 0) {{ float xo = float(x[(size_t)(ts{j} / KS) * {kin}u + uint(c)]);
          {MK._UNROLL} for (uint r = 0; r < {O}u; r++) {{ og[r] = xo * float(woutg[((size_t)e * {N}u + row0 + r) * {n_out}u + lane]);
                                                      ou[r] = xo * float(woutu[((size_t)e * {N}u + row0 + r) * {n_out}u + lane]); }} }} }}
      {MK._UNROLL} for (uint r = 0; r < {O}u; r++) {{ og[r] = simd_sum(og[r]); ou[r] = simd_sum(ou[r]); }}"""
        outl = "" if n_out == 0 else " g += og[r]; u += ou[r];"
        epi.append(_rows_guard(j, f"""{{{outl_pre}
      float SE = simd_sum(se{j}), SO = simd_sum(so{j});
      {MK._UNROLL} for (uint r = 0; r < {O}u; r++) {{
        float G0 = simd_sum(g0_{j}[r]), G1 = simd_sum(g1_{j}[r]), U0 = simd_sum(u0_{j}[r]), U1 = simd_sum(u1_{j}[r]);
        if (lane == 0u) {{
          float g = ((G0 - {float(mu[0])}f * SE) * {float(isd[0])}f + (G1 - {float(mu[1])}f * SO) * {float(isd[1])}f) * sg[(size_t)e * {N}u + row0 + r];
          float u = ((U0 - {float(mu[0])}f * SE) * {float(isd[0])}f + (U1 - {float(mu[1])}f * SO) * {float(isd[1])}f) * su_[(size_t)e * {N}u + row0 + r];{outl}{clamp}
          out[(size_t)ts{j} * {N}u + row0 + r] = {odt}({hval});
        }}
      }} }}"""))
    src = f"""
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row0 = threadgroup_position_in_grid.y * 8u + sgi * {O}u;{_group_head(RP)}
    const uint lb = lane * {PL}u; const uint sh = lb & 31u; (void)sh;
    const device uint32_t* wp = wg + ((size_t)e * {N}u + row0) * {W}u + (lb >> 5);
    const device uint32_t* up = wu + ((size_t)e * {N}u + row0) * {W}u + (lb >> 5);
    {decl}
    {MK._blocks(kin, body, advance)}
    {chr(10).join(epi)}
    """
    ins = ["xr", "x", "idx", "wg", "sg", "wu", "su_"] + (["ocol", "woutg", "woutu"] if n_out else [])
    lim = MK.act_tag(act, limit, alpha)
    return mx.fast.metal_kernel(
        name=f"jtg_gu_k{kin}_n{N}_K{MK._kname(K)}_o{n_out}_w{W}_{xdt}_{odt}{lim}{'' if guard else '_ng'}_rp{RP}_or{O}",
        input_names=ins, output_names=["out"], source=src)


@functools.lru_cache(maxsize=None)
def _down_grouped(kin: int, N: int, K, L: int, n_out: int, W: int, hdt: str, guard: bool, RP: int, O: int):
    mu, isd = VL.code_stats(L, K)
    PL, adv, _ = MK._lane_geom(K, L)
    R = range(RP)
    decl = "\n    ".join(
        f"const device float* xp{j} = hr + (size_t)ts{j} * {kin}u + lane * 16u; "
        f"float d0_{j}[{O}], d1_{j}[{O}]; "
        f"{MK._UNROLL} for (uint r = 0; r < {O}u; r++) {{ d0_{j}[r] = 0.0f; d1_{j}[r] = 0.0f; }} "
        f"float se{j} = 0.0f, so{j} = 0.0f, xi{j} = 1.0f; half2 xh{j}[8];" for j in R)
    loads = "\n      ".join(_rows_guard(j, _xload_row(j, guard)) for j in R)
    ld = MK._window_load(K, L, lambda i: f"wr[{i}]")
    rows = " ".join(_rows_guard(j, f"{{ half2 ad = half2(0.0h); {_apply(f'xh{j}', 'ad')} "
                                   f"d0_{j}[r] += float(ad.x) * xi{j}; d1_{j}[r] += float(ad.y) * xi{j}; }}") for j in R)
    body = f"""{loads}
      {MK._UNROLL} for (uint r = 0; r < {O}u; r++) {{
        const device uint32_t* wr = wp + r * {W}u; {ld} half2 pv[8]; {_decode_pv(K, L)} {rows}
      }}"""
    advance = f"wp += {adv}u;" + "".join(f" xp{j} += 512u;" for j in R)
    epi = []
    for j in R:
        outl_pre = "" if n_out == 0 else f"""
      float od[{O}];
      {MK._UNROLL} for (uint r = 0; r < {O}u; r++) od[r] = 0.0f;
      if (lane < {n_out}u) {{ int c = ocol[(size_t)e * {n_out}u + lane];
        if (c >= 0) {{ float ho = float(h[(size_t)ts{j} * {kin}u + uint(c)]);
          {MK._UNROLL} for (uint r = 0; r < {O}u; r++) od[r] = ho * float(wout[((size_t)e * {N}u + row0 + r) * {n_out}u + lane]); }} }}
      {MK._UNROLL} for (uint r = 0; r < {O}u; r++) od[r] = simd_sum(od[r]);"""
        outl = "" if n_out == 0 else " y += od[r];"
        epi.append(_rows_guard(j, f"""{{{outl_pre}
      float SE = simd_sum(se{j}), SO = simd_sum(so{j});
      {MK._UNROLL} for (uint r = 0; r < {O}u; r++) {{
        float D0 = simd_sum(d0_{j}[r]), D1 = simd_sum(d1_{j}[r]);
        float y = ((D0 - {float(mu[0])}f * SE) * {float(isd[0])}f + (D1 - {float(mu[1])}f * SO) * {float(isd[1])}f) * sd[(size_t)e * {N}u + row0 + r];{outl}
        if (lane == 0u) ysl[(size_t)ts{j} * {N}u + row0 + r] = y;
      }} }}"""))
    src = f"""
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row0 = threadgroup_position_in_grid.y * 8u + sgi * {O}u;{_group_head(RP)}
    const uint lb = lane * {PL}u; const uint sh = lb & 31u; (void)sh;
    const device uint32_t* wp = wd + ((size_t)e * {N}u + row0) * {W}u + (lb >> 5);
    {decl}
    {MK._blocks(kin, body, advance)}
    {chr(10).join(epi)}
    """
    ins = ["hr", "h", "idx", "wd", "sd"] + (["ocol", "wout"] if n_out else [])
    return mx.fast.metal_kernel(
        name=f"jtg_dn_k{kin}_n{N}_K{MK._kname(K)}_o{n_out}_w{W}_{hdt}{'' if guard else '_ng'}_rp{RP}_or{O}",
        input_names=ins, output_names=["ysl"], source=src)


@functools.lru_cache(maxsize=None)
def _combine(N: int, kslots: int):
    """out[t, n] = the router-weighted slot sum in v3's order: acc = 0, then acc = fma(w, y, acc) slot by slot."""
    src = f"""
    uint n = thread_position_in_grid.x, t = thread_position_in_grid.y;
    if (n >= {N}u) return;
    float acc = 0.0f;
    for (uint sl = 0; sl < {kslots}u; sl++) {{
      uint ts = t * {kslots}u + sl; float wgt = float(rw[ts]); float y = ysl[(size_t)ts * {N}u + n];
      acc = fma(wgt, y, acc);
    }}
    out[(size_t)t * {N}u + n] = acc;
    """
    return mx.fast.metal_kernel(name=f"jtg_comb_n{N}_s{kslots}", input_names=["ysl", "rw"], output_names=["out"],
                                source=src)


def _shape(rows_per_pass, out_rows):
    RP = int(rows_per_pass) if rows_per_pass else _rp_env()
    O = int(out_rows) if out_rows else _out_rows_env()
    return RP, O


def gate_up_swiglu_grouped(xr, x, idx, U, out_dtype=mx.float32, limit: float = 0.0, act: str = "silu",
                           alpha: float = 0.0, rows_per_pass: int = 0, out_rows: int = 0):
    """As MK.gather_gate_up_swiglu_v3 (rot_out off): (T * k, N), each slot's row bit-identical to v3's."""
    T, kin = x.shape
    k = idx.shape[1]
    N = U["gate"]["scale"].shape[1]
    n_out = U["gate"]["n_out"]
    W = int(U["gate"]["packed"].shape[-1])
    RP, O = _shape(rows_per_pass, out_rows)
    ins = [xr, x, idx.reshape(-1).astype(mx.uint32), U["gate"]["packed"], U["gate"]["scale"], U["up"]["packed"],
           U["up"]["scale"]] + ([U["gate"]["ocol"], U["gate"]["wout"], U["up"]["wout"]] if n_out else [])
    kern = _gu_grouped(kin, N, U["K"], U["L"], n_out, W, MK._dt(x), _ODT[out_dtype], float(limit), MK._fp16_guard(),
                       MK.check_act(act), float(alpha) if act == "oai" else 0.0, RP, O)
    tpg = 32 * (8 // O)
    return kern(inputs=ins, template=[("NS", T * k), ("KS", k)], grid=(tpg, N // 8, T * k), threadgroup=(tpg, 1, 1),
                output_shapes=[(T * k, N)], output_dtypes=[out_dtype])[0]


def down_slots_grouped(hr, h, idx, U, rows_per_pass: int = 0, out_rows: int = 0):
    """Each slot's down projection (T * k, D) fp32, before the router weights: the y of v3's slot loop."""
    T, k = idx.shape
    kin = h.shape[1]
    N = U["down"]["scale"].shape[1]
    n_out = U["down"]["n_out"]
    W = int(U["down"]["packed"].shape[-1])
    RP, O = _shape(rows_per_pass, out_rows)
    ins = [hr, h, idx.reshape(-1).astype(mx.uint32), U["down"]["packed"], U["down"]["scale"]] + \
          ([U["down"]["ocol"], U["down"]["wout"]] if n_out else [])
    kern = _down_grouped(kin, N, U["K_dn"], U["L"], n_out, W, MK._dt(h), MK._fp16_guard(), RP, O)
    tpg = 32 * (8 // O)
    return kern(inputs=ins, template=[("NS", T * k)], grid=(tpg, N // 8, T * k), threadgroup=(tpg, 1, 1),
                output_shapes=[(T * k, N)], output_dtypes=[mx.float32])[0]


def slot_combine(ysl, rw, T: int, k: int):
    N = int(ysl.shape[1])
    tg = min(256, N)
    return _combine(N, k)(inputs=[ysl, rw.reshape(-1).astype(mx.float32)], grid=(((N + tg - 1) // tg) * tg, T, 1),
                          threadgroup=(tg, 1, 1), output_shapes=[(T, N)], output_dtypes=[mx.float32])[0]


def down_weighted_grouped(hr, h, idx, rw, U, rows_per_pass: int = 0, out_rows: int = 0):
    """As MK.gather_down_weighted_v3: (T, D) fp32, bit-identical."""
    T, k = idx.shape
    return slot_combine(down_slots_grouped(hr, h, idx, U, rows_per_pass, out_rows), rw, T, k)
