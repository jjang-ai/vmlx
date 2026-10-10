"""JANGT routed experts for vMLX (INTERNAL): a MoE layer whose gate/up group and down group are each JANGT (trellis,
vmlx_engine.jangt.moe_kernels) or JANGH (jangtq2, vmlx_engine.jangh). Bundle contract (config.json):

  "jangt": {"version": 1, "code": "v2_halfbits", "state_bits": 12, "rotation": "hadamard128-signs"}
  quantization["model.layers.N.mlp.switch_mlp.<proj>"] = {"mode": "jangt", "kbits": K}            (JANGT)
                                                     = {"mode": "jangtq2", "bits": b, "rotation": "hadamard32"} (JANGH)
  gate_proj and up_proj share one format and width (fused kernel); down may differ.
JANGT tensors per projection (all are module parameters, names = bundle names):
  jt_packed uint32 (E, N, W)   trellis rows, W = words_per_row(K_in, 12, K)
  jt_scale  float32 (E, N)     row scales (unit-gain refit included)
  jt_su     float32 (K_in,)    per-LAYER input sign vector
  jt_ocol   int32  (E, n_out)  outlier input columns (-1 = padding); gate/up columns are layer-wide
  jt_wout   float16 (E, N, n_out) outlier column weights
  jt_cs     float32 (E, K_in)  down only: per-expert column scales (folded into the down-input rotation)
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import os

import numpy as np

from vmlx_engine.jangt import moe_kernels as MK
from vmlx_engine.jangt import v2l as VL

SORT_THRESHOLD = 72      # > 64 so an 8-token speculative verify block (8 x top-8 = 64 slots) stays on the decode kernels
PREFILL_EXPERT_GROUP = 64


class JTSwitchLinear(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, num_experts: int, kbits, n_out: int, col_scales: bool,
                 state_bits: int = 12):
        super().__init__()
        self.input_dims, self.output_dims, self.num_experts = input_dims, output_dims, num_experts
        self.kbits = int(kbits) if float(kbits).is_integer() else float(kbits)
        self.state_bits = state_bits
        W = VL.words_per_row(input_dims, state_bits, self.kbits)
        self.jt_packed = mx.zeros((num_experts, output_dims, W), dtype=mx.uint32)
        self.jt_scale = mx.zeros((num_experts, output_dims), dtype=mx.float32)
        self.jt_su = mx.zeros((input_dims,), dtype=mx.float32)
        self.jt_ocol = mx.full((num_experts, max(n_out, 1)), -1, dtype=mx.int32)
        self.jt_wout = mx.zeros((num_experts, output_dims, max(n_out, 1)), dtype=mx.float16)
        if col_scales:
            self.jt_cs = mx.ones((num_experts, input_dims), dtype=mx.float32)

    def to_quantized(self, **kwargs):
        """Already quantized; generic loaders call nn.quantize with the bundle entry. Fail closed on a mismatch."""
        if kwargs.get("mode", "jangt") != "jangt" or float(kwargs.get("kbits", self.kbits)) != float(self.kbits):
            raise ValueError(f"jangt: bundle entry {kwargs} does not match the installed module (kbits {self.kbits})")
        return self

    def unit(self) -> dict:
        """Kernel-facing view (moe_kernels unit layout); derived arrays cached outside the parameter tree."""
        oc = self.jt_ocol
        n_out = int(oc.shape[-1]) if int(mx.max(oc).item()) >= 0 else 0
        return {"K": self.kbits, "kin": self.input_dims, "packed": self.jt_packed, "scale": self.jt_scale, "su": self.jt_su,
                "ocol": oc, "wout": self.jt_wout, "n_out": n_out, "cs": getattr(self, "jt_cs", None), "N": self.output_dims}


class JTMixedSwitchGLU(nn.Module):
    is_jangt = True

    def __init__(self, D: int, I: int, E: int, gu: dict, dn: dict, n_out_gu: int, n_out_dn: int, limit: float = 0.0):
        super().__init__()
        from vmlx_engine.jangh.switch import TQSwitchLinear
        self.D, self.I, self.E = D, I, E
        self.limit = float(limit)          # clamped SwiGLU (GLM-5.3: 10); 0 = plain SwiGLU (N0.5)
        self.gu_fmt, self.dn_fmt = gu["mode"], dn["mode"]
        if gu["mode"] == "jangt":
            self.gate_proj = JTSwitchLinear(D, I, E, gu["kbits"], n_out_gu, False)
            self.up_proj = JTSwitchLinear(D, I, E, gu["kbits"], n_out_gu, False)
        else:
            self.gate_proj = TQSwitchLinear(D, I, E, gu["bits"], gu.get("rotation", "hadamard32"))
            self.up_proj = TQSwitchLinear(D, I, E, gu["bits"], gu.get("rotation", "hadamard32"))
        if dn["mode"] == "jangt":
            self.down_proj = JTSwitchLinear(I, D, E, dn["kbits"], n_out_dn, True)
        else:
            self.down_proj = TQSwitchLinear(I, D, E, dn["bits"], dn.get("rotation", "hadamard32"))
        self._U = None

    # derived kernel inputs, built once after weights are bound (never parameters)
    def _unit(self):
        if self._U is None:
            U = {"L": 12, "gate": None, "up": None, "down": None, "K": None, "K_dn": None}
            if self.gu_fmt == "jangt":
                g, u = self.gate_proj.unit(), self.up_proj.unit()
                m = np.ones(self.D, np.float32); c = np.array(g["ocol"])[0]; m[c[c >= 0]] = 0.0
                U.update(gate=g, up=u, K=g["K"], su_gu=g["su"] * mx.array(m))
                # fused prefill epilogue needs layer-wide columns shared by gate and up (checked, else separate path)
                cg, cu = np.array(g["ocol"]), np.array(u["ocol"])
                if g["n_out"] and (cg == cg[0]).all() and (cg == cu).all() and (cg[0] >= 0).all():
                    U["gu_cols"] = mx.array(cg[0].astype(np.int32))
                    g["_wout32"] = g["wout"].astype(mx.float32); u["_wout32"] = u["wout"].astype(mx.float32)
            if self.dn_fmt == "jangt":
                d = self.down_proj.unit()
                oc = np.array(d["ocol"]); m = np.ones((self.E, self.I), np.float32)
                for e in range(self.E):
                    c = oc[e][oc[e] >= 0]; m[e, c] = 0.0
                U.update(down=d, K_dn=d["K"], sv_dn=d["cs"] * d["su"][None, :] * mx.array(m))
            mx.eval({k: v for k, v in U.items() if isinstance(v, mx.array)})
            self._U = U
        return self._U

    def _decode_h(self, x2, idx2):
        if self.gu_fmt == "jangt":
            U = self._unit()
            return MK.gather_gate_up_swiglu_v3(MK.rotate_rows(x2, U["su_gu"]), x2, idx2, U, out_dtype=mx.float32, limit=self.limit)
        from vmlx_engine.jangh import kernels as K
        g, u = self.gate_proj, self.up_proj
        return K.gather_qmv(K.h32_rows(x2, mx.float32), g.tq2_packed, g.tq2_scales, g._cb, idx2.reshape(-1).astype(mx.uint32),
                            g.bits, x_per_dispatch=False, packed_u=u.tq2_packed, scales_u=u.tq2_scales, limit=self.limit)

    def _decode_down(self, h, idx2, w2, out_dtype):
        if self.dn_fmt == "jangt":
            U = self._unit()
            return MK.gather_down_weighted_v3(MK.rotate_rows(h, U["sv_dn"], idx2.reshape(-1)), h, idx2, w2, U).astype(out_dtype)
        from vmlx_engine.jangh import kernels as K
        d = self.down_proj
        return K.gather_qmv_weighted_down(K.h32_rows(h, mx.float32), d.tq2_packed, d.tq2_scales, d._cb, idx2.astype(mx.uint32), w2,
                                          d.bits, out_dtype)

    def _jt_mm(self, xs_rot, xs_raw, idx_s, P):
        """Expert-sorted JANGT projection: NAX trellis tiles (jangt.prefill) + outlier columns. fp32 out."""
        from vmlx_engine.jangt.prefill import gather_qmm_sorted_jt
        y = gather_qmm_sorted_jt(xs_rot.astype(mx.bfloat16), P, idx_s).astype(mx.float32)
        if P["n_out"]:
            if not hasattr(self, "_wo"):
                self._wo = {}
            k = id(P["wout"])
            if k not in self._wo:
                self._wo[k] = P["wout"].astype(mx.float32).swapaxes(-1, -2)
            cols = mx.maximum(P["ocol"], 0)[idx_s.astype(mx.int32)]
            xo = mx.take_along_axis(xs_raw, cols, axis=1).astype(mx.float32)
            y = y + mx.gather_mm(xo[:, None, :], self._wo[k], rhs_indices=idx_s.astype(mx.uint32), sorted_indices=True)[:, 0, :]
        return y

    def _prefill(self, x2, idx2, w2):
        T, k = idx2.shape
        flat = idx2.reshape(-1); order = mx.argsort(flat); idx_s = flat[order].astype(mx.uint32); tok = order // k
        if self.gu_fmt == "jangt":
            U = self._unit(); xr = MK.rotate_rows(x2, U["su_gu"])
            if FUSED_GU_PREFILL and U["gate"]["n_out"] > 0 and U.get("gu_cols") is not None:
                # one NAX pass: both weight tiles + outlier columns in the epilogue (jangt/prefill.py)
                from vmlx_engine.jangt.prefill import gather_gate_up_sorted_jt
                xo = x2[:, U["gu_cols"]].astype(mx.float32)[tok]
                h_s = gather_gate_up_sorted_jt(xr.astype(mx.bfloat16)[tok], xo, U["gate"], U["up"], idx_s, limit=self.limit).astype(mx.float32)
            elif FUSED_GU_PREFILL and U["gate"]["n_out"] == 0:
                from vmlx_engine.jangt.prefill import gather_qmm_sorted_jt      # no outliers: plain fused SwiGLU pass
                h_s = gather_qmm_sorted_jt(xr.astype(mx.bfloat16)[tok], U["gate"], idx_s, U["up"], limit=self.limit).astype(mx.float32)
            else:
                xs_r, xs_x = xr[tok], x2[tok]
                g = self._jt_mm(xs_r, xs_x, idx_s, U["gate"]); u = self._jt_mm(xs_r, xs_x, idx_s, U["up"])
                if self.limit > 0:
                    g = mx.minimum(g, self.limit); u = mx.clip(u, -self.limit, self.limit)
                h_s = g * mx.sigmoid(g) * u
                if ROUND_H_CONTROL:                                             # A/B control only: the fused kernels store bf16 h
                    h_s = h_s.astype(mx.bfloat16).astype(mx.float32)
        else:
            from vmlx_engine.jangh import kernels as K
            gp, upr = self.gate_proj, self.up_proj
            h_s = K.gather_qmm_sorted(K.h32_rows(x2, x2.dtype)[tok], gp.tq2_packed, gp.tq2_scales, gp._cb, idx_s, gp.bits,
                                      packed_u=upr.tq2_packed, scales_u=upr.tq2_scales, limit=self.limit).astype(mx.float32)
        if self.dn_fmt == "jangt":
            U = self._unit()
            y_s = self._jt_mm(MK.rotate_rows(h_s, U["sv_dn"], idx_s), h_s, idx_s, U["down"])
        else:
            from vmlx_engine.jangh import kernels as K
            d = self.down_proj
            y_s = K.gather_qmm_sorted(K.h32_rows(h_s.astype(mx.bfloat16), mx.bfloat16), d.tq2_packed, d.tq2_scales, d._cb,
                                      idx_s, d.bits)
        if BMM_COMBINE:
            # unsort in the producer dtype (bf16 from JANGH down: half the gather bytes), then the routing-weighted sum
            # as ONE fp32 batched (1 x k) @ (k x D) matmul instead of broadcast-multiply + reduce: 3.6 -> 2.1 ms per
            # 2048-token layer, numerically exact (rel 7e-8). A bf16 GEMM was 1.0 ms but moved answer top-1 0.955-0.945
            # vs 0.965-1.0 floor for +2-3% end to end -> rejected (2026-10-09).
            y = y_s[mx.argsort(order)].reshape(T, k, -1).astype(mx.float32)
            return (w2[:, None, :] @ y)[:, 0, :]
        y = y_s.astype(mx.float32)[mx.argsort(order)].reshape(T, k, -1)
        return (y * w2[..., None]).sum(axis=1)

    def routed(self, x, indices, scores):
        lead, kk = x.shape[:-1], indices.shape[-1]
        x2 = x.reshape(-1, self.D); idx2 = indices.reshape(-1, kk).astype(mx.int32); w2 = scores.reshape(-1, kk).astype(mx.float32)
        if idx2.size < SORT_THRESHOLD:
            y = self._decode_down(self._decode_h(x2, idx2), idx2, w2, x.dtype)
        else:
            y = self._prefill(x2, idx2, w2).astype(x.dtype)
        return y.reshape(*lead, self.D)


ROUND_H_CONTROL = False
BMM_COMBINE = os.environ.get("JANGT_BMM_COMBINE", "1") != "0"
FUSED_GU_PREFILL = os.environ.get("JANGT_FUSED_GU_PREFILL", "1") != "0"


def contract_key(path: str) -> str:
    """Module path of a routed MLP -> its per-module contract prefix, always "model.layers.<L>.mlp.switch_mlp".
    Wrappers prefix the decoder ("language_model.model.layers..." in the GLM VLM): splitting at the first "model."
    hit the one inside "language_model." and produced "model.model.layers..." (2026-10-10: GLM JANGHT bundle refused
    to load)."""
    i = path.find("model.layers.")
    if i >= 0:
        return path[i:] + ".switch_mlp"
    return "model." + path.split("model.", 1)[1] + ".switch_mlp" if "model." in path else path + ".switch_mlp"


def install_jangt(model: nn.Module, config: dict) -> int:
    """Replace every routed SwitchGLU with a JTMixedSwitchGLU per the per-module contract (fail closed). config:
    {"jangt", "quantization", optional "swiglu_limit" (clamped SwiGLU, GLM-5.3 = 10; absent/0 = plain SwiGLU)}.
    Paths containing "mtp" are skipped (MTP heads are not part of the JANGT contract)."""
    from mlx_lm.models.switch_layers import SwitchGLU
    jt = config.get("jangt") or {}
    if int(jt.get("version", 0)) != 1 or jt.get("code") != "v2_halfbits" or int(jt.get("state_bits", 0)) != 12:
        raise ValueError(f"jangt: unsupported jangt block {jt}")
    quant = config.get("quantization") or {}
    n_out = jt.get("outlier_columns", {})
    limit = float(config.get("swiglu_limit", 0.0) or 0.0)
    count = 0
    for path, mod in list(model.named_modules()):
        sw = getattr(mod, "switch_mlp", None)
        if not isinstance(sw, SwitchGLU) or "mtp" in path:
            continue
        key = contract_key(path)
        ents = {p: quant.get(f"{key}.{p}") for p in ("gate_proj", "up_proj", "down_proj")}
        if any(v is None for v in ents.values()):
            raise ValueError(f"jangt: incomplete routed contract for {key}")
        if ents["gate_proj"] != ents["up_proj"]:
            raise ValueError(f"jangt: gate/up formats differ for {key}")
        for v in ents.values():
            if v.get("mode") not in ("jangt", "jangtq2"):
                raise ValueError(f"jangt: unsupported mode {v} for {key}")
        E, I, D = sw.gate_proj.weight.shape
        mod.switch_mlp = JTMixedSwitchGLU(D, I, E, ents["gate_proj"], ents["down_proj"],
                                          int(n_out.get(f"{key}.gate_proj", 0)), int(n_out.get(f"{key}.down_proj", 0)), limit)
        count += 1
    return count
