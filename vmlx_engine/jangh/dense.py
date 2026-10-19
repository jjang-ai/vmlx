"""Dense JANGH (jangtq2) projection: y = x W^T with W in the JANGH codebook format, stored as ONE-expert banks
`<path>.tq2_packed` (1, N, K*bits/32) uint32 + `<path>.tq2_scales` (1, N) float16, every token routed to expert 0.
Reuses the routed kernels unchanged (decode gather_qmv with in-kernel H32; prefill host H32 rows + gather_qmm_sorted),
the Raptor-0.6 dense-JANGH design (docs: jang/docs/runtime/raptor-jangh-2026-09-29). Added 2026-10-05 for
K2-Horizon / Diplodocus mixed bundles (gate/up affine, down JANGH). Config entry: {"mode": "jangtq2", "bits": b,
"rotation": "hadamard32"|"none"} on a NON-routed path (e.g. model.layers.N.mlp.down_proj); the routed installer ignores
such entries only when the model builds a TQLinear for them (payload validation fails closed otherwise)."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from . import kernels as K
from .format import codebook

# Dense decode routing (rows = activation rows of one call):
#   rows 1 .. ROWS_THRESHOLD-1 : kernels.gather_qmv_rows -- every row shares each weight read, per-row bits identical to
#                                the per-row gather QMV (tests/test_jangh_dense_rows.py); 1 row rotates in-kernel,
#                                2+ rows read h32_rows(x, float32) once per activation.
#   rows >= ROWS_THRESHOLD     : one-pass sorted QMM (~38 ms fixed on the 27B JANGH2 MLP, then nearly flat).
# Before (SORT_THRESHOLD 4, per-row gather QMV below it) the 27B JANGH2 forward cost 37/53/83/97/150/264 ms at
# 1/2/4/5/8/16 rows per-row vs 36/75/77/78/88/101 sorted; the rows kernel measured (M5 Max, 32 distinct weights per
# shape) 0.057 / 0.065 / 0.09 / 0.12-0.15 / 0.15-0.19 ms per 2-bit MLP projection at 1/2/3/4/5 rows against
# 0.065-0.091 / 0.13 / 0.18 / 0.25 / 0.32 per-row and 0.17-0.20 sorted, so the crossover moved to 6 rows.
# VMLX_JANGH_DENSE_ROWS=0 restores the old routing (A/B control); the per-row kernel stays for 3/6-bit formats.
SORT_THRESHOLD = 4
ROWS_THRESHOLD = 6
ROTATIONS = ("none", "hadamard32")


def _row_exact_rows(rows: int) -> bool:
    """Inside a row-exact verify forward and within its row limit."""
    from ..metal.row_exact_qmv import MAX_ROWS, row_exact_scope_active

    return rows <= MAX_ROWS and row_exact_scope_active()


class TQLinear(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, bits: int, rotation: str = "hadamard32"):
        super().__init__()
        if rotation not in ROTATIONS or input_dims % 32:
            raise ValueError("jangtq2 dense: unsupported rotation or input width")
        self.bits, self.rotation, self.input_dims, self.output_dims = bits, rotation, input_dims, output_dims
        self.num_experts = 1
        self.tq2_packed = mx.zeros((1, output_dims, input_dims * bits // 32), dtype=mx.uint32)
        self.tq2_scales = mx.zeros((1, output_dims), dtype=mx.float16)
        self._cb = mx.array(codebook(bits))
        self.is_jangtq2_dense = True

    def to_quantized(self, **kwargs):
        if kwargs.get("mode", "jangtq2") != "jangtq2" or kwargs.get("bits", self.bits) != self.bits:
            raise ValueError("jangtq2 dense: quantization entry differs from installed module")
        return self

    # Opt-in (a speculative target sets it): inside a row-exact verify scope
    # keep every row on the per-row decode kernel, so each verify row equals
    # its one-row decode call bit for bit (the sorted QMM reduces in a
    # different order). Off by default: wide verify windows of other families
    # take the faster sorted path.
    row_exact_verify = False

    def __call__(self, x: mx.array) -> mx.array:
        lead, Kd = x.shape[:-1], x.shape[-1]
        x2 = x.reshape(-1, Kd); T = x2.shape[0]
        idx = mx.zeros((T,), dtype=mx.uint32)
        rot = self.rotation == "hadamard32"
        exact = getattr(self, "row_exact_verify", False) and _row_exact_rows(T)
        if _rows_enabled() and K.rows_kernel_supports(self.bits) and (T < ROWS_THRESHOLD or exact):
            if T == 1:
                y = K.gather_qmv_rows(x2, self.tq2_packed, self.tq2_scales, idx[:1], self.bits, rotate=rot)
            else:
                xr = K.h32_rows_cached(x) if rot else x2          # keyed on the module input: gate and up share it
                y = K.gather_qmv_rows(xr, self.tq2_packed, self.tq2_scales, idx[:1], self.bits)
        elif T < SORT_THRESHOLD or exact:
            y = K.gather_qmv(x2, self.tq2_packed, self.tq2_scales, self._cb, idx, self.bits, x_per_dispatch=True, rotate=rot)
        else:
            xr = K.h32_rows(x2, x2.dtype) if rot else x2
            y = K.gather_qmm_sorted(xr, self.tq2_packed, self.tq2_scales, self._cb, idx, self.bits)
        return y.astype(x.dtype).reshape(*lead, self.output_dims)


_ROWS_ENV = None


def _rows_enabled() -> bool:
    global _ROWS_ENV
    if _ROWS_ENV is None:
        import os
        _ROWS_ENV = os.environ.get("VMLX_JANGH_DENSE_ROWS", "1").strip().lower() not in ("0", "off", "false", "no")
    return _ROWS_ENV


def dense_entries(config: dict) -> dict[str, dict]:
    q = config.get("quantization") or {}
    return {p: e for p, e in q.items() if isinstance(e, dict) and e.get("mode") == "jangtq2" and ".switch_mlp." not in p}


def install_jangh_dense(model: nn.Module, config: dict) -> int:
    """Replace every module whose path has a dense jangtq2 entry with a TQLinear BEFORE nn.quantize. Paths are the
    model's own module paths (bundle namespace). Fails closed on an entry that matches no Linear."""
    ents = dense_entries(config)
    if not ents:
        return 0
    mods = dict(model.named_modules()); done = set()
    for path, e in ents.items():
        parent_path, _, leaf = path.rpartition(".")
        parent = mods.get(parent_path); child = getattr(parent, leaf, None) if parent is not None else None
        if child is not None and hasattr(child, "tq2_packed"):
            continue          # the model class already built a JANGH module here (e.g. k2_horizon / kolibri1 __init__)
        if child is None or not hasattr(child, "weight") or child.weight.ndim != 2:
            raise ValueError(f"jangtq2 dense: entry {path} matches no Linear")
        N, Kd = child.weight.shape
        setattr(parent, leaf, TQLinear(Kd, N, int(e["bits"]), e.get("rotation", "hadamard32")))
        done.add(path)
    return len(done)
