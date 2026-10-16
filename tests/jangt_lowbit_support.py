"""Shared fixtures for the sub-2-bit JANGT tests (tests/test_jangt_lowbit_kernels.py, tests/test_qwen4_exp_janght.py).

* load_encoder_oracle(): the encoder's own trellis module (jang_tools/jangt/v2l.py from the jang-tools SOURCE tree,
  JANG_TOOLS_SRC or ~/jang/jang-tools) imported under a private package name, with the low-bit registration of the
  Qwen3.8-Flash-Next JANGHT encoder (docs/runtime/qwen38fn-janght-spark-2026-10-15/scripts/lowbit/jt_patch.py)
  applied. The installed jang_tools wheel has no jangt package, and importing the source tree as `jang_tools` would
  shadow the wheel for the rest of the test process, hence the private name. None when the source is absent.
* random_projection(): a JANGT projection with uniformly random packed words (every bitstream is a valid trellis
  path), row scales, per-layer signs, outlier columns (-1 padded) and down column scales, in the ragged STAGE layout
  the encoder writes (jt_packed flat, jt_off, jt_kbits) plus the bundle layout (E, N, W).
* dense_reference(): jang_tools.jangt.moe_encode.dequant_mx / dequant_ragged re-stated (unpack_states -> values_of
  -> row scale -> blockwise H128 -> signs -> outlier columns -> column scales), fp32/fp64 on the host + MLX
  elementwise ops only; none of the Metal kernels under test participate.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import re
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

L_STATE = 12
HB = 128
JT_PATCH = Path(os.environ.get("JANG_ROOT", Path.home() / "jang")) / \
    "docs/runtime/qwen38fn-janght-spark-2026-10-15/scripts/lowbit/jt_patch.py"
_ORACLE = "_jangt_encoder_oracle"


def load_encoder_oracle():
    src = Path(os.environ.get("JANG_TOOLS_SRC", Path.home() / "jang" / "jang-tools")) / "jang_tools" / "jangt"
    if not (src / "v2l.py").is_file():
        return None
    if _ORACLE + ".v2l" in sys.modules:
        return sys.modules[_ORACLE + ".v2l"]
    spec = importlib.util.spec_from_file_location(_ORACLE, src / "__init__.py", submodule_search_locations=[str(src)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[_ORACLE] = pkg
    spec.loader.exec_module(pkg)
    v2l = importlib.import_module(_ORACLE + ".v2l")
    # jt_patch.py, same semantics (that file imports jang_tools.jangt.v2l by its absolute name)
    orig = v2l.kpat

    def kpat(K):
        if K == 1.5:
            return (4, 2)
        if K == 1:
            return (2, 2)
        return orig(K)

    v2l.kpat = kpat
    v2l.MULTS[1.5] = 2124823417
    v2l.MULTS[1] = 590428815
    v2l.code_stats.cache_clear()
    return v2l


def jt_patch_mults() -> dict | None:
    """The MULTS registered by the encoder's jt_patch.py file itself (None when the campaign file is absent)."""
    if not JT_PATCH.is_file():
        return None
    txt = JT_PATCH.read_text()
    return {float(k): int(v) for k, v in re.findall(r"MULTS\[([0-9.]+)\]\s*=\s*([0-9]+)", txt)}


def kn(K):
    K = float(K)
    return int(K) if K.is_integer() else K


def random_projection(rng, E: int, N: int, kin: int, K, n_out: int, layer_cols: bool, col_scales: bool, VLmod) -> dict:
    """Synthetic stage tensors of one projection (encoder schema) + the (E, N, W) bundle view."""
    K = kn(K)
    W = VLmod.words_per_row(kin, L_STATE, K)
    packed = rng.integers(0, 2 ** 32, size=(E, N, W), dtype=np.uint64).astype(np.uint32)
    scale = rng.uniform(0.02, 0.05, size=(E, N)).astype(np.float32)
    su = rng.choice([-1.0, 1.0], size=kin).astype(np.float32)
    ocol = np.full((E, max(n_out, 1)), -1, np.int32)
    if n_out:
        if layer_cols:                                            # gate/up: layer-wide columns, same for every expert
            ocol[:, :n_out] = np.sort(rng.choice(kin, n_out, replace=False)).astype(np.int32)
        else:                                                     # down: per-expert columns, some experts padded
            for e in range(E):
                m = n_out if e % 3 else n_out - 1                 # every third expert has one -1 pad slot
                ocol[e, :m] = np.sort(rng.choice(kin, m, replace=False)).astype(np.int32)
    wout = (rng.standard_normal((E, N, max(n_out, 1))) * 0.05).astype(np.float16)
    wout[ocol[:, None, :].repeat(N, 1) < 0] = 0
    d = {"K": K, "kin": kin, "E": E, "N": N, "W": W, "packed": packed, "scale": scale, "su": su, "ocol": ocol,
         "wout": wout, "n_out": n_out}
    if col_scales:
        cs = rng.uniform(0.5, 2.0, size=(E, kin)).astype(np.float32)
        for e in range(E):                                        # encoder invariant: cs == 1 on outlier columns
            c = ocol[e][ocol[e] >= 0]
            cs[e, c] = 1.0
        d["cs"] = cs
    return d


def stage_tensors(prefix: str, P: dict) -> dict:
    """The encoder's ragged per-layer stage tensors for one projection (encode_q38t.py / moe_encode.py schema)."""
    E, N, W = P["packed"].shape
    t = {prefix + ".jt_packed": mx.array(P["packed"].reshape(-1)),
         prefix + ".jt_off": mx.array((np.arange(E + 1) * N * W).astype(np.int64)),
         prefix + ".jt_kbits": mx.array(np.full(E, float(P["K"]), np.float32)),
         prefix + ".jt_scale": mx.array(P["scale"]), prefix + ".jt_su": mx.array(P["su"]),
         prefix + ".jt_ocol": mx.array(P["ocol"]), prefix + ".jt_wout": mx.array(P["wout"])}
    if "cs" in P:
        t[prefix + ".jt_cs"] = mx.array(P["cs"])
    return t


def dense_reference(P: dict, VLmod) -> np.ndarray:
    """(E, N, kin) float64 weights in the ORIGINAL basis: moe_encode.dequant_mx (+ dequant_ragged's column scales)."""
    E, N, W = P["packed"].shape; kin = P["kin"]; K = P["K"]
    out = np.empty((E, N, kin), np.float64)
    for e in range(E):
        st = mx.array(VLmod.unpack_states(P["packed"][e], kin, L_STATE, K))
        Wr = VLmod.values_of(st, 1.0, L_STATE, K) * mx.array(P["scale"][e])[:, None]
        Wd = mx.hadamard_transform(Wr.reshape(N, kin // HB, HB), scale=HB ** -0.5).reshape(N, kin) * mx.array(P["su"])[None]
        Wd = np.array(Wd).astype(np.float64)
        c = P["ocol"][e][P["ocol"][e] >= 0]
        if len(c):
            Wd[:, c] = P["wout"][e][:, :len(c)].astype(np.float64)
        if "cs" in P:
            Wd = Wd * P["cs"][e][None, :].astype(np.float64)
        out[e] = Wd
    return out


def rel_l2(a, b) -> float:
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    nb = float(np.linalg.norm(b))
    assert np.isfinite(a).all() and np.isfinite(b).all() and nb > 0, "empty / non-finite reference (fail closed)"
    return float(np.linalg.norm(a - b) / nb)


def silu(x):
    return x / (1.0 + np.exp(-x))


def layer_unit(rng, D: int, I: int, E: int, Kgu, Kdn, n_gu: int, n_dn: int, VLmod) -> dict:
    """gate/up/down of one routed layer as the encoder writes them: gate and up share the layer's input signs and
    layer-wide outlier columns and one width; down has its own width, per-expert outliers and column scales."""
    g = random_projection(rng, E, I, D, Kgu, n_gu, True, False, VLmod)
    u = random_projection(rng, E, I, D, Kgu, n_gu, True, False, VLmod)
    u["su"], u["ocol"] = g["su"], g["ocol"]
    u["wout"][u["ocol"][:, None, :].repeat(I, 1) < 0] = 0
    d = random_projection(rng, E, D, I, Kdn, n_dn, False, True, VLmod)
    return {"gate": g, "up": u, "down": d}


def bind(m, P: dict) -> None:
    """Bind synthetic bundle-layout tensors to a JTMixedSwitchGLU (the arrays a strict load_weights would set)."""
    for name in ("gate", "up", "down"):
        lin, p = getattr(m, name + "_proj"), P[name]
        lin.jt_packed = mx.array(p["packed"]); lin.jt_scale = mx.array(p["scale"]); lin.jt_su = mx.array(p["su"])
        lin.jt_ocol = mx.array(p["ocol"]); lin.jt_wout = mx.array(p["wout"])
        if "cs" in p:
            lin.jt_cs = mx.array(p["cs"])
    m._U = None


def moe_reference(Wg, Wu, Wd, x: np.ndarray, idx: np.ndarray, w: np.ndarray):
    """h (T*k, I) and y (T, D) for dense float64 expert weights. np.einsum, not `@`: the Accelerate BLAS matmul raises
    spurious divide/overflow FP flags on these float64 GEMVs (values were finite either way); einsum's own loop keeps
    the reference free of that noise."""
    T, k = idx.shape
    h = np.empty((T * k, Wg.shape[1]))
    y = np.zeros((T, Wd.shape[1]))
    for t in range(T):
        for s in range(k):
            e = idx[t, s]
            g = np.einsum("ij,j->i", Wg[e], x[t]); u = np.einsum("ij,j->i", Wu[e], x[t])
            h[t * k + s] = silu(g) * u
            y[t] += w[t, s] * np.einsum("ij,j->i", Wd[e], h[t * k + s])
    return h, y
