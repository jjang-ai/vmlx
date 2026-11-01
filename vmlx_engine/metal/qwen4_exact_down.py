# Copyright © 2023-2024 Apple Inc.
#
# Portions adapted from MLX v0.32.2 quantized.h and reduction/reduce_col.h.
# MIT License
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Qualified-shape Qwen AR expert-down/weight/reduce dispatch.

Preserve native MLX gather_qmv lane arithmetic and the top-10 col_reduce_small
tree, rather than importing a GGUF or eight-lane dot. Expert cohorts share one
dispatch; no new weight copy, cache representation, router or sampling policy.
Only the existing affine gate/up path calls this helper. Unsupported layouts,
prefill, batches and speculative forwards keep their original implementation.
"""

from functools import lru_cache
import importlib.metadata
import logging
import os

import mlx.core as mx

from .affine_moe_pair_decode import affine_moe_ar_scope_active, _projection_reason

logger = logging.getLogger(__name__)
def exact_down_requested():
    """Enabled by default; hardware, precision and productive-AR guards remain."""
    return os.environ.get("VMLX_QWEN4_EXACT_DOWN", "1").lower() in {"1", "true", "yes", "on"}


_ENABLED = exact_down_requested()
_FAILED = False
_OBSERVED = False

_HEADER = r'''
#include <metal_stdlib>
using namespace metal;
template<typename T, int B, int V>
inline float qwen_down_load(const device T* x, thread float* a) {
    float sum=0;
    if (B==2) {
        for(int i=0;i<V;i+=4) {
            sum+=x[i]+x[i+1]+x[i+2]+x[i+3];
            a[i]=x[i];a[i+1]=x[i+1]/4.0f;
            a[i+2]=x[i+2]/16.0f;a[i+3]=x[i+3]/64.0f;
        }
    } else if (B==3) {
        for(int i=0;i<V;i+=8) {
            sum+=x[i]+x[i+1]+x[i+2]+x[i+3]+x[i+4]+x[i+5]+x[i+6]+x[i+7];
            a[i]=x[i];a[i+1]=x[i+1]/8.0f;a[i+2]=x[i+2]/64.0f;
            a[i+3]=x[i+3]/2.0f;a[i+4]=x[i+4]/16.0f;
            a[i+5]=x[i+5]/128.0f;a[i+6]=x[i+6]/4.0f;a[i+7]=x[i+7]/32.0f;
        }
    } else if (B==4) {
        for(int i=0;i<V;i+=4) {
            sum+=x[i]+x[i+1]+x[i+2]+x[i+3];
            a[i]=x[i];a[i+1]=x[i+1]/16.0f;
            a[i+2]=x[i+2]/256.0f;a[i+3]=x[i+3]/4096.0f;
        }
    } else if (B==6) {
        for(int i=0;i<V;i+=4) {
            sum+=x[i]+x[i+1]+x[i+2]+x[i+3];
            a[i]=x[i];a[i+1]=x[i+1]/64.0f;
            a[i+2]=x[i+2]/16.0f;a[i+3]=x[i+3]/4.0f;
        }
    } else if (B==8) {
        for(int i=0;i<V;++i) {sum+=x[i];a[i]=x[i];}
    }
    return sum;
}
template<int B,int V>
inline float qwen_down_dot(const device uint8_t* w,const thread float* a,
                           float scale,float bias,float sum) {
    float accum=0;
    if(B==2) {
        for(int i=0;i<V/4;++i)
            accum+=(a[4*i]*(w[i]&3)+a[4*i+1]*(w[i]&12)
                    +a[4*i+2]*(w[i]&48)+a[4*i+3]*(w[i]&192));
    } else if(B==3) {
        // V=8 in the admitted K640 native gather_qmv path.
        accum+=(w[0]&7)*a[0];accum+=(w[0]&56)*a[1];
        accum+=(w[0]&192)*a[2];accum+=(w[1]&1)*(a[2]*256.0f);
        accum+=(w[1]&14)*a[3];accum+=(w[1]&112)*a[4];
        accum+=(w[1]&128)*a[5];accum+=(w[2]&3)*(a[5]*256.0f);
        accum+=(w[2]&28)*a[6];accum+=(w[2]&224)*a[7];
    } else if(B==4) {
        const device uint16_t* p=(const device uint16_t*)w;
        for(int i=0;i<V/4;++i)
            accum+=(a[4*i]*(p[i]&15)+a[4*i+1]*(p[i]&240)
                    +a[4*i+2]*(p[i]&3840)+a[4*i+3]*(p[i]&61440));
    } else if(B==6) {
        // V=4 in the admitted K640 native gather_qmv path.
        accum+=(w[0]&63)*a[0];accum+=(w[0]&192)*a[1];
        accum+=(w[1]&15)*(a[1]*256.0f);accum+=(w[1]&240)*a[2];
        accum+=(w[2]&3)*(a[2]*256.0f);accum+=(w[2]&252)*a[3];
    } else if(B==8) {
        for(int i=0;i<V;++i) accum+=a[i]*w[i];
    }
    return scale*accum+sum*bias;
}
'''

_SOURCE = r'''
    constexpr int V=BITS==2?16:(BITS==3||BITS==4?8:4);
    uint tile=threadgroup_position_in_grid.x;
    uint route=simdgroup_index_in_threadgroup;
    uint lane=thread_index_in_simdgroup;
    uint first=tile*ROWS;
    uint expert=expert_ids[route];
    const device uint8_t* w=(const device uint8_t*)weight;
    float accum[ROWS]={0},a[V];
    // K640's native non-fast QMV assigns V values per lane. The final
    // guarded block contains whole V-value packs, never a partial pack.
    for(uint k=0;k<640u;k+=32u*V) {
        uint start=k+lane*V;
        if(start<640u && expert<EXPERTS) {
            float sum=qwen_down_load<T,BITS,V>(activated+route*640u+start,a);
            for(uint r=0;r<ROWS;++r) {
                size_t row=(size_t)expert*2560u+first+r;
                size_t packed=row*(640u*BITS/8u)+start*BITS/8u;
                size_t meta=row*(640u/GS)+start/GS;
                accum[r]+=qwen_down_dot<BITS,V>(w+packed,a,
                    float(scales[meta]),float(biases[meta]),sum);
            }
        }
    }
    threadgroup T partial[10*ROWS];
    for(uint r=0;r<ROWS;++r) {
        accum[r]=simd_sum(accum[r]);
        if(lane==0u) {
            // Router-produced IDs are in range. An invalid internal caller
            // must not read another allocation or silently fabricate a zero.
            T projection=expert<EXPERTS ? T(accum[r]) : T(NAN);
            partial[route*ROWS+r]=T(projection*route_scores[route]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(route==0u && lane<ROWS) {
        // MLX col_reduce_small uses lsize.y=8: cohorts0/1 first add
        // experts8/9, then the eight cohort results are added in order.
        T total=T(partial[lane]+T(0));
        total=T(partial[8u*ROWS+lane]+total);
        T second=T(partial[ROWS+lane]+T(0));
        second=T(partial[9u*ROWS+lane]+second);
        total=T(second+total);
        for(uint i=2u;i<8u;++i)
            total=T(T(partial[i*ROWS+lane]+T(0))+total);
        output[first+lane]=total;
    }
'''


@lru_cache(maxsize=1)
def _compatible_runtime():
    try:
        return (importlib.metadata.version("mlx") in ("0.32.2", "0.32.3", "0.32.5.dev20261101+360078a8")
                and mx.device_info().get("device_name") == "Apple M5 Max")
    except (importlib.metadata.PackageNotFoundError, RuntimeError):
        return False


def _eligible(projection, activated, indices, scores):
    if (activated.shape != (1, 1, 10, 1, 640)
            or activated.dtype != mx.float16
            or indices.shape != (1, 1, 10) or scores.shape != indices.shape
            or indices.dtype not in (mx.uint32, mx.int32)
            or scores.dtype != activated.dtype
            or bool(getattr(projection, "training", False))):
        return False
    if _projection_reason(projection, hidden=640, intermediate=2560):
        return False
    return (projection.bits in (2, 3, 4, 6, 8)
            and projection.group_size in (32, 64)
            and projection.scales.dtype == activated.dtype
            and projection.weight.shape[1] == 2560
            and projection.weight.shape[0] >= 10)


@lru_cache(maxsize=1)
def _kernel_header():
    if importlib.metadata.version("mlx") in ("0.32.2", "0.32.3"):
        return _HEADER
    # Newer MLX promotes each input before the bias sum. Reuse the installed
    # arithmetic instead of carrying another copy that can drift on upgrades.
    from .row_exact_qmv import _mlx_qmv_header
    header = _mlx_qmv_header()
    if header is None:
        raise RuntimeError("qualified Qwen down projection requires MLX QMV helpers")
    return "#include <metal_stdlib>\nusing namespace metal;\n" + header + r'''
template<typename T, int B, int V>
inline float qwen_down_load(const device T* x, thread float* a) {
    return load_vector<T, float, V, B>(x, a);
}
template<int B, int V>
inline float qwen_down_dot(const device uint8_t* w, const thread float* a,
                          float scale, float bias, float sum) {
    return qdot<float, V, B>(w, a, scale, bias, sum);
}
'''


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="vmlx_qwen4_exact_down_v2", header=_kernel_header(), source=_SOURCE,
        input_names=["activated", "weight", "scales", "biases", "expert_ids", "route_scores"],
        output_names=["output"], ensure_row_contiguous=True)


def qwen4_exact_down(projection, activated, indices, scores, *, enabled=None):
    """Return the qualified productive-AR fusion, or None without side effects."""
    global _FAILED, _OBSERVED
    if not (_ENABLED if enabled is None else enabled):
        return None
    if (_FAILED or not affine_moe_ar_scope_active() or mx.default_device() != mx.gpu
            or not mx.metal.is_available() or not _compatible_runtime()
            or not _eligible(projection, activated, indices, scores)):
        return None
    try:
        output = _kernel()(
            inputs=[activated, projection.weight, projection.scales, projection.biases,
                    indices.astype(mx.uint32), scores],
            template=[("T", activated.dtype), ("BITS", projection.bits), ("GS", projection.group_size),
                      ("EXPERTS", projection.weight.shape[0]), ("ROWS", 8)],
            grid=(320*320, 1, 1), threadgroup=(320, 1, 1),
            output_shapes=[(1, 1, 2560)], output_dtypes=[activated.dtype])[0]
        if not _OBSERVED:
            mx.eval(output)
            _OBSERVED = True
            logger.info("Qwen exact down fusion active: bits=%s group=%s "
                        "fp16 K640 N2560 top10 scope=productive_ar math=mlx%s",
                        projection.bits, projection.group_size, importlib.metadata.version("mlx"))
        return output
    except (RuntimeError, ValueError):
        _FAILED = True
        logger.exception("Qwen exact down fusion failed; retaining native down/reduction")
        return None
