"""Experimental QSA decode selection; score arithmetic is deliberately external.

Not wired into model execution until numerical and full-model cost qualification.
Radix selection avoids a full-row sort; the output is chronological, with equal
scores resolved by original index. No host readback or context-specific compile.
"""
from functools import lru_cache

import mlx.core as mx

_HEADER = r"""
#include <metal_stdlib>
using namespace metal;
inline uint qsa_key(float value) {
    if (isnan(value)) return 0u;
    if (value == 0.0f) value = 0.0f;
    uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}
"""
_SOURCE = r"""
uint tid = thread_index_in_threadgroup;
uint n = scores_shape[2];
threadgroup atomic_uint hist[256];
threadgroup uint threshold;
threadgroup uint need;
threadgroup uint counts[256];
threadgroup uint equal_counts[256];
if (tid == 0) { threshold = 0; need = K; }
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint pass = 0; pass < 4; ++pass) {
    atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint shift = 24u - 8u * pass;
    uint mask = pass == 0 ? 0u : (0xffffffffu << (shift + 8u));
    uint prefix = threshold;
    for (uint i = tid; i < n; i += 256u) {
        uint key = qsa_key(scores[i]);
        if ((key & mask) == prefix)
            atomic_fetch_add_explicit(&hist[(key >> shift) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (int digit = 255; digit >= 0; --digit) {
            uint count = atomic_load_explicit(&hist[digit], memory_order_relaxed);
            if (count >= need) { threshold |= uint(digit) << shift; break; }
            need -= count;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
uint chunk = (n + 255u) / 256u;
uint begin = min(tid * chunk, n), end = min(begin + chunk, n);
uint greater = 0, equal = 0;
for (uint i = begin; i < end; ++i) {
    uint key = qsa_key(scores[i]);
    greater += key > threshold; equal += key == threshold;
}
counts[tid] = greater; equal_counts[tid] = equal;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0) {
    uint prior_equal = 0, prior_selected = 0;
    for (uint lane = 0; lane < 256u; ++lane) {
        uint eq = equal_counts[lane];
        uint selected_count = counts[lane] + min(eq, need - min(need, prior_equal));
        equal_counts[lane] = prior_equal;
        counts[lane] = prior_selected;
        prior_equal += eq; prior_selected += selected_count;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
uint position = counts[tid], eq_rank = equal_counts[tid];
for (uint i = begin; i < end; ++i) {
    uint key = qsa_key(scores[i]);
    bool take = key > threshold;
    if (key == threshold) { take = eq_rank < need; ++eq_rank; }
    if (take) selected[position++] = int(i);
}
"""


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="vmlx_qsa_radix_select", input_names=["scores"],
        output_names=["selected"], header=_HEADER, source=_SOURCE,
        compile_options={"math_mode": "safe"},
    )


def qwen4_qsa_select(scores, *, k=512, enabled=False):
    """Return chronological [1,1,k] indices, or None for the stock path."""
    if not enabled or scores.ndim != 3 or scores.shape[:2] != (1, 1):
        return None
    if scores.dtype != mx.float32 or type(k) is not int or not 1 <= k < scores.shape[2]:
        return None
    return _kernel()(
        inputs=[scores], template=[("K", k)], grid=(256, 1, 1),
        threadgroup=(256, 1, 1), output_shapes=[(1, 1, k)],
        output_dtypes=[mx.int32],
    )[0]
