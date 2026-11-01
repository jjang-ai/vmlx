# SPDX-License-Identifier: Apache-2.0
"""GLM WY substitution for MLX 0.32.2/0.32.3 on M5 Max, preserving FP32 sum order.

0.32.3 qualified 2026-10-08: bit-identical to the stock 63-step loop on 96 real
A matrices (NT 1..32 x 3 decay regimes, A built by the real pairwise path) and
14.36 -> 2.57 ms per KDA layer per 2,048-token chunk (audit kda_subst_exact.py).
Before that the gate named 0.32.2 only, so the app (0.32.3) never engaged it.

Keep row1 on stock MLX; rows2..63 use the qualified column-reduction tree.
Every iteration snapshots its original row before any writes. No recurrence,
chunk size or native-state format changes. Unsupported inputs retain stock.
"""
from functools import lru_cache
import importlib.metadata
import logging
import mlx.core as mx
from vmlx_engine.glm5_prefill_policy import glm5_kda_substitution_requested

logger = logging.getLogger(__name__)
_FAILED = False
_OBSERVED = False
_CALLS = 0
_SOURCE = r"""
 #pragma clang fp contract(off)
 #pragma clang fp reassociate(off)

 uint tid=thread_index_in_threadgroup;
 uint lane=thread_index_in_simdgroup;
 uint sg=simdgroup_index_in_threadgroup;
 size_t group=threadgroup_position_in_grid.x;
 threadgroup float mat[4096];
 threadgroup float oldrow[64];
 for(uint x=tid;x<4096;x+=256) mat[x]=a[group*4096+x];
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint i=2;i<64;++i){
   if(tid<64) oldrow[tid]=mat[i*64+tid];
   threadgroup_barrier(mem_flags::mem_threadgroup);
   for(uint col=sg;col<i;col+=8){
     float p0=oldrow[lane]*mat[lane*64+col];
     float acc=p0+0.0f;
     float p1=oldrow[lane+32]*mat[(lane+32)*64+col];
     acc=p1+acc;
     float sum=metal::simd_sum(acc);
     if(lane==0) mat[i*64+col]=oldrow[col]+sum;
   }
   threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 for(uint x=tid;x<4096;x+=256) b[group*4096+x]=mat[x];
"""

@lru_cache(maxsize=1)
def _compatible_runtime():
    try:
        return (importlib.metadata.version("mlx") in ("0.32.2", "0.32.3", "0.32.5.dev20261101+360078a8")
                and mx.metal.is_available()
                and mx.device_info().get("device_name") == "Apple M5 Max")
    except (importlib.metadata.PackageNotFoundError, RuntimeError):
        return False

@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="vmlx_glm5_kda_substitution_v1", input_names=["a"], output_names=["b"],
        ensure_row_contiguous=True, compile_options={"math_mode": "safe"}, source=_SOURCE,
    )

def kda_substitution(a, *, enabled=None):
    global _FAILED, _OBSERVED, _CALLS
    if enabled is None:
        enabled = glm5_kda_substitution_requested()
    if (not enabled or _FAILED or a.ndim != 5 or a.shape[:2] != (1, 64)
            or a.shape[-2:] != (64, 64) or not 1 <= a.shape[2] <= 32
            or a.dtype != mx.float32 or mx.default_device() != mx.gpu
            or not _compatible_runtime()):
        return None
    try:
        # Leave the caller's original matrix untouched if first launch fails.
        first = mx.array(a)
        update = mx.sum(first[..., 1, :, None] * first[..., :, :1], axis=-2)
        first[..., 1, :1] = first[..., 1, :1] + update
        result = _kernel()(
            inputs=[first], grid=(a.size // 4096 * 256, 1, 1),
            threadgroup=(256, 1, 1), output_shapes=[a.shape], output_dtypes=[mx.float32],
        )[0]
        if not _OBSERVED:
            mx.eval(result)
            _OBSERVED = True
            logger.info("GLM KDA substitution active: shape=%s fp32 mlx0322_col32x32_i1stock_v1", tuple(a.shape))
        _CALLS += 1
        return result
    except (RuntimeError, ValueError) as exc:
        _FAILED = True
        logger.warning("GLM KDA substitution unavailable; retaining stock solver: %s", exc)
        return None

def kda_substitution_status():
    requested = glm5_kda_substitution_requested()
    return {
        "requested": requested,
        "installed": bool(requested and _compatible_runtime() and not _FAILED),
        "observed_calls": int(_OBSERVED),
        "enqueued_calls": _CALLS,
        "observation": "first dispatch evaluated; later calls enqueued",
        "reason": "first_launch_failed" if _FAILED else (
            "unqualified_runtime" if requested and not _compatible_runtime() else None),
    }
