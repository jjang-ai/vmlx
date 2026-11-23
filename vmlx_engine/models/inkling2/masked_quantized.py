"""Masked floating-point quantized matvecs for locally owned expert routes.

Same 16-value dot grouping, FP32 accumulation, global scale and output
rounding; skip unowned route tiles.
"""

# Copyright © 2023 Apple Inc.

from functools import lru_cache

import mlx.core as mx
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

DECODE_BLOCK_SIZE = 8


@lru_cache(maxsize=1)
def _masked_qmv():
    return mx.fast.metal_kernel(
        name="ml4_masked_native_nvfp4_qmv",
        input_names=["x", "w", "s", "g", "indices"],
        output_names=["out"],
        header=_HEADER,
        source=r"""
        uint route = threadgroup_position_in_grid.z;
        uint lane = thread_index_in_simdgroup;
        uint row0 = threadgroup_position_in_grid.y * 8
                  + simdgroup_index_in_threadgroup * 4;
        int expert = int(indices[route]);
        if (expert < 0) {
            if (lane == 0) {
                for (uint r = 0; r < 4; ++r)
                    if (row0+r < N) out[route*N+row0+r] = T(0);
            }
            return;
        }
        const device T* xp = x + (route / ROUTES_PER_X) * K;
        const device uint* wp = w + size_t(expert) * N * (K/8);
        const device uchar* sp = s + size_t(expert) * N * (K/16);
        fp_qmv_fast_impl<T,16,4,true,false>(
            wp, sp, g + expert, xp, out + route*N,
            x_shape[x_ndim-1], w_shape[1],
            uint3(0, threadgroup_position_in_grid.y, 0),
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """,
        compile_options={"math_mode": "safe"},
    )


def supports_masked_qmv(linear, *, input_alignment=512):
    """Only the native NVFP4 vector layout and scaling tested by this kernel."""
    from .nvfp4 import ScaledQuantizedSwitchLinear

    if not isinstance(linear, (QuantizedSwitchLinear, ScaledQuantizedSwitchLinear)):
        return False
    if (
        linear.mode != "nvfp4"
        or linear.bits != 4
        or linear.group_size != 16
        or linear.input_dims % input_alignment
        or linear.output_dims % 8
        or linear.weight.dtype != mx.uint32
        or linear.scales.dtype != mx.uint8
        or "bias" in linear
        or "pre_quant_scale" in linear
        or linear.get("biases") is not None
        or getattr(linear, "_quantize_activations", False)
    ):
        return False
    if isinstance(linear, ScaledQuantizedSwitchLinear):
        return linear.global_scale.size in (1, linear.num_experts)
    return True


def _global_scale(linear):
    e = linear.num_experts
    if hasattr(linear, "global_scale"):
        factor = mx.broadcast_to(linear.mlx_global_scale().reshape(-1), (e,))
    else:
        factor = mx.full((e,), 2688.0, mx.float32)
    if e < 64:
        # Small custom-kernel inputs use Metal's constant address space.
        factor = mx.concatenate([factor, mx.zeros((64 - e,), mx.float32)])
    return factor


def masked_qmv(x, linear, indices):
    k, n = linear.input_dims, linear.output_dims
    assert k % 512 == 0 and n % 8 == 0
    factor = _global_scale(linear)
    per_x = indices.size // (x.size // k)
    return _masked_qmv()(
        inputs=[x, linear.weight, linear.scales, factor, indices],
        template=[("T", x.dtype), ("K", k), ("N", n), ("ROUTES_PER_X", per_x)],
        grid=(32, ((n + 7) // 8) * 2, indices.size),
        threadgroup=(32, 2, 1),
        output_shapes=[(*indices.shape, n)],
        output_dtypes=[x.dtype],
    )[0]


@lru_cache(maxsize=1)
def _paired_qmv():
    return mx.fast.metal_kernel(
        name="paired_masked_native_nvfp4_qmv",
        input_names=["x", "w", "s", "g", "indices"],
        output_names=["out"],
        header=_HEADER,
        compile_options={"math_mode": "safe"},
        source=r"""
    uint first=threadgroup_position_in_grid.z*2;
    uint lane=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
    int row0=threadgroup_position_in_grid.y*8+sg*4;
    int expert=int(indices[first]);
    bool same=true;
    for (int v=1;v<2;++v) same=same && int(indices[first+v])==expert;
    if (!same) {
        for (int v=0;v<2;++v) {
            int e=int(indices[first+v]);
            if(e<0) {if(lane==0) for(int r=0;r<4;++r) out[(first+v)*N+row0+r]=T(0);continue;}
            fp_qmv_fast_impl<T,16,4,true,false>(w+size_t(e)*N*(K/8),s+size_t(e)*N*(K/16),g+e,x+(first+v)*K,out+(first+v)*N,
                x_shape[x_ndim-1],w_shape[1],uint3(0,threadgroup_position_in_grid.y,0),sg,lane);
        }
        return;
    }
    if (expert<0) {
        if(lane==0) for(int v=0;v<2;++v) for(int r=0;r<4;++r) out[(first+v)*N+row0+r]=T(0);
        return;
    }
    const device uint8_t* ws=(const device uint8_t*)w+size_t(expert)*N*(K/2)+row0*(K/2)+lane*8;
    const device uint8_t* scales=s+size_t(expert)*N*(K/16)+row0*(K/16)+lane;
    thread float xt[2][16];
    thread float result[2][4]={0};
    for(int k=0;k<x_shape[x_ndim-1];k+=512) {
        #pragma unroll
        for(int v=0;v<2;++v) load_vector<T,float,16>(x+(first+v)*K+lane*16+k,xt[v]);
        for(int r=0;r<4;++r) {
            auto wl=ws+r*(K/2);
            float scale=dequantize_scale<float,16>(scales[r*(K/16)]);
            #pragma unroll
            for(int v=0;v<2;++v) result[v][r]+=qdot<float,16,4>(wl,xt[v],scale);
        }
        ws+=256;scales+=32;
    }
    float inv_scale=g[expert]/(F8E4M3_MAX*F4E2M1_MAX);
    for(int r=0;r<4;++r) {
        #pragma unroll
        for(int v=0;v<2;++v) {
            result[v][r]=simd_sum(result[v][r]);
            if(lane==0)out[(first+v)*N+row0+r]=static_cast<T>(result[v][r]*inv_scale);
        }
    }
    """,
    )


def paired_masked_qmv(x, linear, indices):
    """Reuse packed weights for adjacent equal experts in a sorted route list.

    Preserve each vector's native accumulation. Pairs crossing an expert
    boundary use independent native QMV calls, and negative routes stay zero.
    The caller supplies one input vector per route and an even route count.
    """
    k, n = linear.input_dims, linear.output_dims
    assert k % 512 == 0 and n % 8 == 0 and indices.size % 2 == 0
    assert x.size == indices.size * k
    return _paired_qmv()(
        inputs=[x, linear.weight, linear.scales, _global_scale(linear), indices],
        template=[("T", x.dtype), ("K", k), ("N", n)],
        grid=(32, (n // 8) * 2, indices.size // 2),
        threadgroup=(32, 2, 1),
        output_shapes=[(*indices.shape, n)],
        output_dtypes=[x.dtype],
    )[0]


@lru_cache(maxsize=1)
def _grouped_qmv():
    return mx.fast.metal_kernel(
        name="grouped_masked_native_nvfp4_qmv",
        input_names=["x", "w", "s", "g", "ids"],
        output_names=["out"],
        header=_HEADER,
        compile_options={"math_mode": "safe"},
        source=r"""
uint first=threadgroup_position_in_grid.z;
uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
int row0=threadgroup_position_in_grid.y*8+sg*4;
int expert=ids[first];
if(expert<0) {
    if(lane==0)for(int r=0;r<4;++r)out[first*N+row0+r]=T(0);
    return;
}
if(first>0 && ids[first-1]==expert)return;
int end=first+1;
while(end<ids_shape[0] && ids[end]==expert)++end;
constexpr int P=(K%512==0)?16:8;
constexpr int BLOCK=P*32;
for(int begin=first;begin<end;begin+=V) {
    thread float xt[V][P];thread float result[V][4]={0};
    const device uint8_t* ws=(const device uint8_t*)w+size_t(expert)*N*(K/2)+row0*(K/2)+lane*(P/2);
    const device uint8_t* scales=s+size_t(expert)*N*(K/16)+row0*(K/16)+lane/(16/P);
    for(int k=0;k<x_shape[x_ndim-1];k+=BLOCK) {
        #pragma unroll
        for(int v=0;v<V;++v)load_vector<T,float,P>(x+min(begin+v,end-1)*K+lane*P+k,xt[v]);
        for(int r=0;r<4;++r) {
            auto wl=ws+r*(K/2);float scale=dequantize_scale<float,16>(scales[r*(K/16)]);
            #pragma unroll
            for(int v=0;v<V;++v)result[v][r]+=qdot<float,P,4>(wl,xt[v],scale);
        }
        ws+=BLOCK/2;scales+=BLOCK/16;
    }
    float scale=g[expert]/(F8E4M3_MAX*F4E2M1_MAX);
    for(int r=0;r<4;++r) {
        #pragma unroll
        for(int v=0;v<V;++v) {
            result[v][r]=simd_sum(result[v][r]);
            if(lane==0 && begin+v<end)out[(begin+v)*N+row0+r]=T(result[v][r]*scale);
        }
    }
}
""",
    )


def grouped_masked_qmv(x, linear, indices):
    """Reuse each sorted expert group in pairs, preserving native reductions.

    Input width selects the native 256- or 512-element accumulation block.
    The caller supplies sorted indices and one input vector per route.
    """
    k, n = linear.input_dims, linear.output_dims
    assert k % 256 == 0 and n % 8 == 0 and x.size == indices.size * k
    return _grouped_qmv()(
        inputs=[x, linear.weight, linear.scales, _global_scale(linear), indices],
        template=[("T", x.dtype), ("K", k), ("N", n), ("V", 2)],
        grid=(32, n // 8 * 2, indices.size),
        threadgroup=(32, 2, 1),
        output_shapes=[(*indices.shape, n)],
        output_dtypes=[x.dtype],
    )[0]


@lru_cache(maxsize=1)
def _paired_mxfp8_qmv():
    return mx.fast.metal_kernel(
        name="paired_masked_mxfp8_qmv",
        input_names=["x", "w", "s", "ids"],
        output_names=["out"],
        header=_HEADER,
        compile_options={"math_mode": "safe"},
        source=r"""
    uint first=threadgroup_position_in_grid.z*V;
    uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
    int row0=threadgroup_position_in_grid.y*8+sg*4;
    int expert=int(ids[first]);
    bool same=true;
    for(int v=1;v<V;++v)same=same && int(ids[first+v])==expert;
    if(!same) {
        for(int v=0;v<V;++v) {
            int e=int(ids[first+v]);
            if(e<0) {if(lane==0)for(int r=0;r<4;++r)out[(first+v)*N+row0+r]=T(0);continue;}
            fp_qmv_fast_impl<T,32,8,false,false>(w+size_t(e)*N*(K/4),s+size_t(e)*N*(K/32),(const device float*)w,x+(first+v)*K,out+(first+v)*N,
                x_shape[x_ndim-1],w_shape[1],uint3(0,threadgroup_position_in_grid.y,0),sg,lane);
        }
        return;
    }
    if(expert<0) {if(lane==0)for(int v=0;v<V;++v)for(int r=0;r<4;++r)out[(first+v)*N+row0+r]=T(0);return;}
    const device uint8_t* ws=(const device uint8_t*)w+size_t(expert)*N*K+row0*K+lane*8;
    const device uint8_t* scales=s+size_t(expert)*N*(K/32)+row0*(K/32)+lane/4;
    thread float xt[V][8];thread float result[V][4]={0};
    for(int k=0;k<x_shape[x_ndim-1];k+=256) {
        #pragma unroll
        for(int v=0;v<V;++v)load_vector<T,float,8>(x+(first+v)*K+lane*8+k,xt[v]);
        for(int r=0;r<4;++r) {
            auto wl=ws+r*K;float scale=dequantize_scale<float,32>(scales[r*(K/32)]);
            #pragma unroll
            for(int v=0;v<V;++v)result[v][r]+=qdot<float,8,8>(wl,xt[v],scale);
        }
        ws+=256;scales+=8;
    }
    for(int r=0;r<4;++r) {
        #pragma unroll
        for(int v=0;v<V;++v) {
            result[v][r]=simd_sum(result[v][r]);
            if(lane==0)out[(first+v)*N+row0+r]=T(result[v][r]);
        }
    }
    """,
    )


def supports_paired_mxfp8_qmv(linear):
    return (
        type(linear) is QuantizedSwitchLinear
        and linear.mode == "mxfp8"
        and linear.bits == 8
        and linear.group_size == 32
        and linear.input_dims % 256 == 0
        and linear.output_dims % 8 == 0
        and linear.weight.dtype == mx.uint32
        and linear.scales.dtype == mx.uint8
        and "bias" not in linear
        and linear.get("biases") is None
    )


def paired_masked_mxfp8_qmv(x, linear, indices):
    """Reuse MXFP8 weights for pairs, skipping negative expert routes.

    Retain native eight-value dot groups and FP32 reductions. The caller sorts
    routes and supplies one contiguous input vector per route.
    """
    k, n = linear.input_dims, linear.output_dims
    assert k % 256 == 0 and n % 8 == 0 and indices.size % 2 == 0
    assert x.size == indices.size * k
    return _paired_mxfp8_qmv()(
        inputs=[x, linear.weight, linear.scales, indices],
        template=[("T", x.dtype), ("K", k), ("N", n), ("V", 2)],
        grid=(32, n // 8 * 2, indices.size // 2),
        threadgroup=(32, 2, 1),
        output_shapes=[(*indices.shape, n)],
        output_dtypes=[x.dtype],
    )[0]


# Keep the native runtime loop bounds and arithmetic order: constexpr loop
# bounds or fast math can change BF16 rounding and subsequent expert routing.
_HEADER = r"""

constant constexpr float F8E4M3_MAX = 448.0f;
constant constexpr float F4E2M1_MAX = 6.0f;

struct fp4_e2m1 {
  fp4_e2m1(float x) thread {
    if (metal::isnan(x)) {
      bits = 0x7;
      return;
    }

    const uint8_t sign_bit = (metal::signbit(x)) ? 0x8 : 0x0;
    x = metal::abs(x);

    if (x > 5.0f) {
      bits = 0x7;
    } else if (x >= 3.5f) {
      bits = 0x6;
    } else if (x > 2.5f) {
      bits = 0x5;
    } else if (x >= 1.75f) {
      bits = 0x4;
    } else if (x > 1.25f) {
      bits = 0x3;
    } else if (x >= 0.75f) {
      bits = 0x2;
    } else if (x > 0.25f) {
      bits = 0x1;
    } else {
      bits = 0x0;
    }
    bits |= sign_bit;
  }

  operator float16_t() thread {
    half converted = as_type<half>(ushort((bits & 7) << 9));
    converted *= 16384.0;
    return bits & 8 ? -converted : converted;
  }

  operator float() thread {
    return static_cast<float>(this->operator float16_t());
  }

  operator bfloat16_t() thread {
    return static_cast<bfloat16_t>(this->operator float16_t());
  }

  uint8_t bits;
};



struct fp8_e4m3 {
  template <typename T>
  fp8_e4m3(T f) thread {
    // From PyTorch
    // https://github.com/pytorch/pytorch/blob/e3643e1e0e923f0fc063dfab6f45c956d568919d/c10/util/Float8_e4m3fn.h#L148
    uint32_t fp8_max = 543 << 21;
    uint32_t denorm_mask = 141 << 23;
    uint32_t f_bits = as_type<uint32_t>(static_cast<float>(f));
    uint32_t sign = f_bits & 0x80000000;
    f_bits ^= sign;
    if (f_bits >= fp8_max) {
      bits = (f_bits > 0x7F800000) ? 0x7F : 0x7E;
    } else {
      if (f_bits < (121 << 23)) {
        f_bits = as_type<uint32_t>(
            as_type<float>(f_bits) + as_type<float>(denorm_mask));
        bits = static_cast<uint8_t>(f_bits - denorm_mask);
      } else {
        // resulting mantissa is odd
        uint8_t mant_odd = (f_bits >> 20) & 1;
        f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
        f_bits += mant_odd;
        bits = static_cast<uint8_t>(f_bits >> 20);
      }
    }
    bits |= static_cast<uint8_t>(sign >> 24);
  }

  operator float16_t() thread {
    uint16_t v = bits & 127;
    uint16_t sign_bit = ((uint16_t)((bits >> 7) & 1)) << 15;
    uint16_t u = (v << 7) | (((v + 1) >> 7) << 14) | sign_bit;
    half converted = as_type<half>(u);
    return converted * 256.0;
  }

  operator bfloat16_t() thread {
    return static_cast<bfloat16_t>(this->operator float16_t());
  }

  operator float() thread {
    return static_cast<float>(this->operator float16_t());
  }

  uint8_t bits;
};

struct fp8_e8m0 {
  fp8_e8m0(float x) thread {
    if (!metal::isfinite(x)) {
      bits = 0xFF;
      return;
    }
    if (x < 0.0f) {
      bits = 0x00;
      return;
    }
    float le = metal::log2(x);
    int n = int(metal::round(le));

    n = n < -127 ? -127 : n;
    n = n > 127 ? 127 : n;
    bits = static_cast<uint8_t>(n + 127);
  }

  operator bfloat16_t() thread {
    uint16_t out = (bits == 0)
        ? 0x40
        : (bits == 0xFF ? 0x7FC0 : (static_cast<uint16_t>(bits) << 7));
    return as_type<bfloat16_t>(out);
  }

  operator float() thread {
    uint32_t out = (bits == 0)
        ? 0x400000
        : (bits == 0xFF ? 0x7FC00000 : (static_cast<uint16_t>(bits) << 23));
    return as_type<float>(out);
  }

  uint8_t bits;
};

// Smallest E8M0 >= x. Scales are amax/max_element, so rounding one down
// leaves the block's largest elements outside the element range, where they
// saturate. Matches the CUDA backend, which rounds up via cutlass ue8m0.
inline float mx_scale_round_up(float x) {
  fp8_e8m0 s(x);
  if (s.bits < 0xFE && float(s) < x) {
    s.bits += 1;
  }
  return float(s);
}

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;
MLX_MTL_CONST int QUAD_SIZE = 4;

template <int wsize = 8, int bits = 4>
inline constexpr short get_pack_factor() {
  return wsize / bits;
}

template <int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  return wsize / 8;
}

template <typename T, int group_size>
static inline T dequantize_scale(uint8_t s) {
  if constexpr (group_size == 16) {
    // Use nv scale
    return T(*(thread fp8_e4m3*)(&s));
  } else {
    return T(*(thread fp8_e8m0*)(&s));
  }
}


template <int bits, typename U = float>
struct Dequantize {
  U operator()(uint8_t x) thread {
    if constexpr (bits == 8) {
      return U(*(thread fp8_e4m3*)(&x));
    } else {
      return U(*(thread fp4_e2m1*)(&x));
    }
  }
};

template <typename T, typename U, int values_per_thread>
inline void load_vector(const device T* x, thread U* x_thread) {
#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    x_thread[i] = x[i];
  }
}


template <typename U, int values_per_thread, int bits>
inline U qdot(const device uint8_t* w, const thread U* x_thread, U scale) {
  U accum = 0;
  if constexpr (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * Dequantize<4>{}(ws[i]) +
           x_thread[4 * i + 1] * Dequantize<4>{}(ws[i] >> 4) +
           x_thread[4 * i + 2] * Dequantize<4>{}(ws[i] >> 8) +
           x_thread[4 * i + 3] * Dequantize<4>{}(ws[i] >> 12));
    }
  } else {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * Dequantize<8>{}(w[i]);
    }
  }

  return scale * accum;
}


template <
    typename T,
    int group_size,
    int bits,
    bool has_global_scale,
    bool partial_rows = false,
    int results_per_simdgroup = 4>
METAL_FUNC void fp_qmv_fast_impl(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device float* global_scale,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = 2;
  constexpr int num_simdgroups = 2;
  constexpr int pack_factor = get_pack_factor<32, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack<32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  // Reuse the last valid row for reads beyond the output tile.
  int last_row = results_per_simdgroup - 1;
  if constexpr (partial_rows) {
    if (out_row >= out_vec_size) {
      return;
    }
    last_row = min(last_row, out_vec_size - 1 - out_row);
  }

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    load_vector<T, U, values_per_thread>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      int src = row;
      if constexpr (partial_rows) {
        src = min(row, last_row);
      }
      auto wl = (const device uint8_t*)(ws + src * in_vec_size_w);
      const device auto* sl = scales + src * in_vec_size_g;

      U s = dequantize_scale<U, group_size>(sl[0]);
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    x += block_size;
  }

  float inv_scale_enc = 1.0f;
  if constexpr (has_global_scale) {
    inv_scale_enc = *global_scale / (F8E4M3_MAX * F4E2M1_MAX);
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0 && row <= last_row) {
      if constexpr (has_global_scale) {
        y[row] = static_cast<T>(result[row] * inv_scale_enc);
      } else {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }
}


"""
