"""Expert-grouped JANGT trellis decode for verify rows (2..16 rows), bit-identical to the fast per-slot kernels.

``VMLX_JANGT_GROUPED_FAST=1`` (default off; read per call) routes the multi-row path of ``fast_decode.routed_fast``
(gate / up + SwiGLU of every (row, slot), and the per-slot down projection before the slot combine) through kernels
that decode each trellis window ONCE per (expert, group of up to RP slots routed to it) and apply it to every member
slot's packed row. Each member's arithmetic is the fast kernels' own sequence (same lane chunks, same window words,
same hashed codes, half2 fma chain per chunk in state order, fp32 fold, outlier columns, epilogue), so every (row,
slot) output is bit-identical to ``fast_decode.gate_up4`` / ``down4`` -- and therefore a verify row equals the one-row
decode step (``fast_decode`` row invariance). tests/test_jangt_grouped_fast.py checks it with controls.

Grouping happens on the GPU (no host read of the routing): ``group_slots`` lists, in slot order, the groups of
experts picked by >= 2 slots (chunks of RP slots in slot order) and the singleton slots; two launches per projection
then run the group kernel (RP members, predicated) and the one-member kernel. A group / singleton list entry is
(expert, members, slot 0 .. slot RP - 1); threadgroups past a list's count exit.

Why it pays: the fast per-slot kernels spend ~80 % of their time on the trellis decode itself (window loads, state
extraction, code hash; 06-PY-VERIFY-KERNELS.md 10-29 ablations) and only ~11 % on the per-row half2 fma, so slots
that share an expert share ~80 % of their work. Real consecutive verify rows share experts (M3.1 coding row: 8 rows
-> 16 distinct experts for 32 slots).

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

import functools
import os

import mlx.core as mx

from vmlx_engine.jangt import fast_decode as FD
from vmlx_engine.jangt import moe_kernels as MK
from vmlx_engine.jangt import v2l as VL

_UNROLL = FD._UNROLL
ENV = "VMLX_JANGT_GROUPED_FAST"
# Members per group (decode shared by up to RP slots) and output rows per simdgroup of the group kernels.
RP = int(os.environ.get("VMLX_JANGT_GROUP_RP", "4") or 4)
GU_O, GU_NOG = 2, 4
DN_O, DN_NOG = 2, 4
# "two": groups and singletons in two launches (the singleton kernel keeps the one-member register footprint) +
# a merge; "one": both lists in one launch of the group kernel.
MODE = os.environ.get("VMLX_JANGT_GROUP_MODE", "two")
STATS = {"calls": 0}


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip().lower() in {"1", "true", "on", "yes"}


# ------------------------------------------------------------------------------------------------ grouping
@functools.lru_cache(maxsize=None)
def _group_kernel(ns: int, rp: int):
    """One threadgroup of NS threads (slot s = thread). Output gl[2][NS][RP + 2] u32 + counts at gl[2 NS (RP + 2)]:
    list 0 = groups of experts picked by >= 2 slots (chunks of RP slots in slot order), list 1 = singleton slots;
    entries (expert, members, member slots...), listed in the slot order of their first member."""
    w = rp + 2
    src = f"""
    const uint s = thread_position_in_threadgroup.x;
    threadgroup uint f0[{ns}], f1[{ns}];
    const uint e = idx[s];
    uint rank = 0u, n = 0u;
    for (uint i = 0u; i < {ns}u; i++) {{ if (idx[i] == e) {{ n++; rank += i < s ? 1u : 0u; }} }}
    const uint flag0 = (n >= 2u && (rank % {rp}u) == 0u) ? 1u : 0u;
    const uint flag1 = n == 1u ? 1u : 0u;
    f0[s] = flag0; f1[s] = flag1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint p0 = 0u, p1 = 0u, t0 = 0u, t1 = 0u;
    for (uint i = 0u; i < {ns}u; i++) {{ p0 += i < s ? f0[i] : 0u; p1 += i < s ? f1[i] : 0u; t0 += f0[i]; t1 += f1[i]; }}
    if (flag0) {{
      device uint* g = gl + (size_t)p0 * {w}u;
      const uint nr = metal::min({rp}u, n - rank);
      g[0] = e; g[1] = nr;
      uint c = 0u, k = 0u;
      for (uint i = 0u; i < {ns}u; i++) {{
        if (idx[i] == e) {{ if (c >= rank && k < nr) {{ g[2u + k] = i; k++; }} c++; }}
      }}
      for (; k < {rp}u; k++) g[2u + k] = g[2u];
    }}
    if (flag1) {{
      device uint* g = gl + ((size_t){ns} + p1) * {w}u;
      g[0] = e; g[1] = 1u;
      for (uint k = 0u; k < {rp}u; k++) g[2u + k] = s;
    }}
    if (s == 0u) {{ gl[{2 * ns * w}u] = t0; gl[{2 * ns * w + 1}u] = t1; }}
    """
    return mx.fast.metal_kernel(name=f"jtg_group_s{ns}_r{rp}", input_names=["idx"], output_names=["gl"], source=src)


def group_slots(idx_flat: mx.array, rp: int = RP) -> mx.array:
    """idx_flat (NS,) expert ids -> the group lists (see _group_kernel)."""
    ns = int(idx_flat.shape[0])
    assert ns <= 1024, ns
    return _group_kernel(ns, rp)(inputs=[idx_flat.astype(mx.uint32)], grid=(ns, 1, 1), threadgroup=(ns, 1, 1),
                                 output_shapes=[(2 * ns * (rp + 2) + 2,)], output_dtypes=[mx.uint32])[0]


# ------------------------------------------------------------------------------------------------ group kernels
def _member_code(K, L, projs, O, nw, J, names):
    """Per block: each (output row, projection) window realigned + decoded ONCE (pv[8]), then applied to the J
    members' packed chunks with fast_decode's per-row sequence (8 half2 fma in state order, then the fp32 fold)."""
    out = []
    for r in range(O):
        for p in projs:
            words = [f"{names[(p, r)]}[{k}]" for k in range(nw)]
            apply = []
            for j in range(J):
                fma8 = " ".join(f"a_ = fma(xh{j}[{i}], pv[{i}], a_);" for i in range(8))
                body = (f"{{ half2 a_ = half2(0.0h); {fma8} {p}{j}[{r}] = fma(float(a_.x), x0_{j}, {p}{j}[{r}]);"
                        f" {p}{j}[{r}] = fma(float(a_.y), x1_{j}, {p}{j}[{r}]); }}")
                apply.append(body if j == 0 else f"if ({j}u < nr) {body}")
            out.append(f"{{ {FD._realign(K, L, words)} half2 pv[8]; {FD._decode_pv(K, L)} {' '.join(apply)} }}")
    return "\n        ".join(out)


def _entry(LIST: int, NS: int, RPW: int) -> str:
    """This threadgroup's list entry G_: LIST 0 / 1 = that list's gi-th entry; LIST 2 = both lists in one launch
    (gi < count0: group gi, else singleton gi - count0)."""
    if LIST in (0, 1):
        return (f"if (gi >= gl[{2 * NS * RPW + LIST}u]) return;"
                f" const device uint* G_ = gl + ((size_t){LIST * NS}u + gi) * {RPW}u;")
    return (f"const uint c0_ = gl[{2 * NS * RPW}u]; const uint c1_ = gl[{2 * NS * RPW + 1}u];"
            f" if (gi >= c0_ + c1_) return;"
            f" const device uint* G_ = gl + (gi < c0_ ? (size_t)gi : ((size_t){NS}u + gi - c0_)) * {RPW}u;")


@functools.lru_cache(maxsize=None)
def gen_grouped(part: str, kin: int, N: int, K, L: int, n_out: int, W: int, xdt: str, odt: str, limit: float,
                act: str, alpha: float, O: int, NOG: int, J: int, LIST: int, NS: int, RPW: int):
    """Gate / up ("gu") or per-slot down ("dn") for the groups of list LIST (J members decoded together; J = 1 for
    the singleton list). Grid (32 NOG, NS, N / (NOG O)); threadgroup (32 NOG); group gi = threadgroup y."""
    mu, isd = VL.code_stats(L, K)
    PL, adv, _ = MK._lane_geom(K, L)
    nw = FD._nwin(K, L)
    projs = ("g", "u") if part == "gu" else ("d",)
    ptr = {"g": "wp", "u": "up", "d": "wp"}
    names = {(p, r): f"w{p}{r}" for p in projs for r in range(O)}
    decl_w = " ".join(f"uint {names[k]}[{nw}];" for k in names)
    loads = " ".join(FD._chunk_loads(K, L, "scalar", ptr[p], names[(p, r)], r, W, "0") for p in projs for r in range(O))
    row_of = (lambda j: f"(ts{j} / KS)") if part == "gu" else (lambda j: f"ts{j}")
    xl = []
    for j in range(J):
        guard = "" if j == 0 else f"if ({j}u < nr) "
        xl.append(f"""half2 xh{j}[8]; float x0_{j} = 0.0f, x1_{j} = 0.0f;
        {guard}{{ const uint4 q0 = xq4_{j}[0], q1 = xq4_{j}[1];
          xh{j}[0] = as_type<half2>(q0.x); xh{j}[1] = as_type<half2>(q0.y); xh{j}[2] = as_type<half2>(q0.z); xh{j}[3] = as_type<half2>(q0.w);
          xh{j}[4] = as_type<half2>(q1.x); xh{j}[5] = as_type<half2>(q1.y); xh{j}[6] = as_type<half2>(q1.z); xh{j}[7] = as_type<half2>(q1.w);
          const float xi = xinv_{j}[0]; x0_{j} = xi * ISD0; x1_{j} = xi * ISD1; }}""")
    adv_x = " ".join(f"xq4_{j} += 64u; xinv_{j} += 32u;" for j in range(J))
    adv_w = " ".join(f"{q} += {adv}u;" for q in dict.fromkeys(ptr[p] for p in projs))
    loop = f"""{decl_w}
    for (uint b = 0; b < {kin // 512}u; b++) {{
      {loads}
      {{ {" ".join(xl)}
        {_member_code(K, L, projs, O, nw, J, names)} }}
      {adv_x} {adv_w}
    }}"""
    acc_decl = " ".join(f"float {p}{j}[{O}];" for p in projs for j in range(J)) + \
        f" {_UNROLL} for (uint r = 0; r < {O}u; r++) {{ " + " ".join(f"{p}{j}[r] = 0.0f;" for p in projs for j in range(J)) + " }"
    xptr = " ".join(f"const device uint4* xq4_{j} = (const device uint4*)(xq + (size_t){row_of(j)} * {kin // 2}u) + lane * 2u;"
                    f" const device float* xinv_{j} = xinv + (size_t){row_of(j)} * {kin // 16}u + lane;" for j in range(J))
    epi = []
    for j in range(J):
        rj = row_of(j)
        if part == "gu":
            outl = "" if n_out == 0 else f"""
      if (lane < {n_out}u) {{ int c = ocol[(size_t)e * {n_out}u + lane];
        if (c >= 0) {{ float xo = float(x[(size_t){rj} * {kin}u + uint(c)]);
          {_UNROLL} for (uint r = 0; r < {O}u; r++) {{
            g{j}[r] += xo * float(woutg[((size_t)e * {N}u + row0 + r) * {n_out}u + lane]) / sg[(size_t)e * {N}u + row0 + r];
            u{j}[r] += xo * float(woutu[((size_t)e * {N}u + row0 + r) * {n_out}u + lane]) / su_[(size_t)e * {N}u + row0 + r]; }} }} }}"""
            body = f"""{{{outl}
      float G[{O}], U[{O}];
      {_UNROLL} for (uint r = 0; r < {O}u; r++) {{ G[r] = simd_sum(g{j}[r]); U[r] = simd_sum(u{j}[r]); }}
      const float SE = st[{rj} * 2u], SO = st[{rj} * 2u + 1u];
      const float cst = {FD._lit(mu[0])} * ISD0 * SE + {FD._lit(mu[1])} * ISD1 * SO;
      {_UNROLL} for (uint r = 0; r < {O}u; r++) {{
        if (lane == 0u) {{
          float gg = (G[r] - cst) * sg[(size_t)e * {N}u + row0 + r];
          float uu = (U[r] - cst) * su_[(size_t)e * {N}u + row0 + r];
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
      float Dv[{O}];
      {_UNROLL} for (uint r = 0; r < {O}u; r++) Dv[r] = simd_sum(d{j}[r]);
      const float cst = {FD._lit(mu[0])} * ISD0 * st[ts{j} * 2u] + {FD._lit(mu[1])} * ISD1 * st[ts{j} * 2u + 1u];
      {_UNROLL} for (uint r = 0; r < {O}u; r++) if (lane == 0u) out[(size_t)ts{j} * {N}u + row0 + r] = (Dv[r] - cst) * sd[(size_t)e * {N}u + row0 + r];
      }}"""
        epi.append(body if j == 0 else f"if ({j}u < nr) {body}")
    wdecl = (f"const device uint32_t* wp = wg + ((size_t)e * {N}u + row0) * {W}u + (lb >> 5);"
             f" const device uint32_t* up = wu + ((size_t)e * {N}u + row0) * {W}u + (lb >> 5);" if part == "gu" else
             f"const device uint32_t* wp = wd + ((size_t)e * {N}u + row0) * {W}u + (lb >> 5);")
    head = " ".join(f"const uint ts{j} = G_[{2 + j}];" for j in range(J))
    src = f"""
    const float ISD0 = {FD._lit(isd[0])}, ISD1 = {FD._lit(isd[1])};
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row0 = threadgroup_position_in_grid.z * {NOG * O}u + sgi * {O}u;
    const uint gi = threadgroup_position_in_grid.y;
    {_entry(LIST, NS, RPW)}
    const uint e = G_[0]; const uint nr = G_[1]; (void)nr;
    {head}
    const uint lb = lane * {PL}u; const uint sh = lb & 31u; (void)sh;
    {wdecl}
    {xptr}
    {acc_decl}
    {loop}
    {chr(10).join(epi)}
    """
    if part == "gu":
        ins = ["xq", "xinv", "st", "x", "gl", "wg", "sg", "wu", "su_"] + (["ocol", "woutg", "woutu"] if n_out else [])
    else:
        ins = ["xq", "xinv", "st", "h", "gl", "wd", "sd"] + (["ocol", "wout"] if n_out else [])
    tag = MK.act_tag(act, limit, alpha) if part == "gu" else ""
    name = (f"jtg_{part}_k{kin}_n{N}_K{MK._kname(K)}_o{n_out}_w{W}_{xdt}_{odt}{tag}_O{O}_G{NOG}_J{J}_L{LIST}_s{NS}"
            f"_w{RPW}")
    return mx.fast.metal_kernel(name=name, input_names=ins, output_names=["out"], source=src)


def _launch(part, packed, xin, gl, U, ns, kslots, out_dtype, limit, act, alpha, rp):
    xq, xinv, st = packed
    key = "gate" if part == "gu" else "down"
    kin = int(xin.shape[1])
    N = int(U[key]["scale"].shape[1])
    n_out = U[key]["n_out"]
    W = int(U[key]["packed"].shape[-1])
    K = U["K"] if part == "gu" else U["K_dn"]
    O, NOG = (GU_O, GU_NOG) if part == "gu" else (DN_O, DN_NOG)
    if part == "gu":
        ins_w = [U["gate"]["packed"], U["gate"]["scale"], U["up"]["packed"], U["up"]["scale"]] + \
            ([U["gate"]["ocol"], U["gate"]["wout"], U["up"]["wout"]] if n_out else [])
    else:
        ins_w = [U["down"]["packed"], U["down"]["scale"]] + ([U["down"]["ocol"], U["down"]["wout"]] if n_out else [])
    out = None
    plan = ((2, rp),) if MODE == "one" else ((0, rp), (1, 1))
    for LIST, J in plan:
        kern = gen_grouped(part, kin, N, K, U["L"], n_out, W, MK._dt(xin), FD._ODT[out_dtype], float(limit),
                           MK.check_act(act), float(alpha) if act == "oai" else 0.0, O, NOG, J, LIST, ns, rp + 2)
        ins = [xq, xinv, st, xin, gl] + ins_w
        # the two lists write disjoint slots of one output: chain them through MLX's init-from-input semantics
        kw = dict(grid=(32 * NOG, ns, N // (NOG * O)), threadgroup=(32 * NOG, 1, 1), output_shapes=[(ns, N)],
                  output_dtypes=[out_dtype], template=[("KS", kslots)])
        if out is None:
            out = kern(inputs=ins, **kw)[0]
        else:
            out = _merge(out, kern(inputs=ins, **kw)[0], gl, ns, rp)
    return out


@functools.lru_cache(maxsize=None)
def _merge_kernel(ns: int, N: int, rpw: int, dt: str):
    """out[ts] = b[ts] for every slot of list 1 (singletons), else a[ts]."""
    src = f"""
    const uint n = thread_position_in_grid.x, ts = thread_position_in_grid.y;
    if (n >= {N}u) return;
    bool single = false;
    const uint c1 = gl[{2 * ns * rpw + 1}u];
    for (uint i = 0u; i < c1; i++) single = single || gl[((size_t){ns} + i) * {rpw}u + 2u] == ts;
    out[(size_t)ts * {N}u + n] = single ? b[(size_t)ts * {N}u + n] : a[(size_t)ts * {N}u + n];
    """
    return mx.fast.metal_kernel(name=f"jtg_merge_s{ns}_n{N}_w{rpw}_{dt}", input_names=["a", "b", "gl"],
                                output_names=["out"], source=src)


def _merge(a, b, gl, ns, rp):
    N = int(a.shape[1])
    tg = min(256, N)
    return _merge_kernel(ns, N, rp + 2, FD._ODT[a.dtype])(
        inputs=[a, b, gl], grid=(-(-N // tg) * tg, ns, 1), threadgroup=(tg, 1, 1), output_shapes=[a.shape],
        output_dtypes=[a.dtype])[0]


def routed_gate_up(x2, idx2, U, limit, act, alpha, gl=None, rp: int = RP):
    """(T * k, I) f32 SwiGLU activations of every (row, slot): fast_decode.routed_gate_up, expert-grouped."""
    T, k = int(idx2.shape[0]), int(idx2.shape[1])
    ns = T * k
    if gl is None:
        gl = group_slots(idx2.reshape(-1), rp)
    return _launch("gu", FD.pack_rows1(x2, U["su_gu"]), x2, gl, U, ns, k, mx.float32, limit, act, alpha, rp)


def routed_down(h, idx2, w2, U, gl=None, rp: int = RP):
    """(T, D) f32 router-weighted down projection: per-slot grouped kernel + fast_decode's slot-ordered combine."""
    T, k = int(idx2.shape[0]), int(idx2.shape[1])
    ns = T * k
    if gl is None:
        gl = group_slots(idx2.reshape(-1), rp)
    packed = FD.pack_rows1(h, U["sv_dn"], idx2.reshape(-1))
    ysl = _launch("dn", packed, h, gl, U, ns, k, mx.float32, 0.0, "silu", 0.0, rp)
    N = int(ysl.shape[1])
    tg = min(256, N)
    return FD._combine4(N, k)(inputs=[ysl, w2.reshape(-1).astype(mx.float32)], grid=(((N + tg - 1) // tg) * tg, T, 1),
                              threadgroup=(tg, 1, 1), output_shapes=[(T, N)], output_dtypes=[mx.float32])[0]


def routed_grouped(x2, idx2, w2, U, limit, act, alpha, rp: int = RP):
    """fast_decode.routed_fast for T >= 2 rows with the expert-grouped kernels: (T, D) f32, bit-identical."""
    STATS["calls"] += 1
    gl = group_slots(idx2.reshape(-1), rp)
    h = routed_gate_up(x2, idx2, U, limit, act, alpha, gl, rp)
    return routed_down(h, idx2, w2, U, gl, rp)
