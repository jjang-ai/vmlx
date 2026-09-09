"""Opt-in expert-aligned Qwen4 prefill, preserving MLX q4 accumulation order.

Only the measured 512-expert, top-10 affine layout and short prefill batches
are eligible. Metadata belongs to one SwitchGLU call; no global routing state.
MLX headers are read lazily and checked against the qualified 0.32.2 sources.
Unsupported installations and shapes use the stock implementation.
"""

import hashlib
import logging
import os
import re
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    _gather_sort,
    _scatter_unsort,
)

logger = logging.getLogger(__name__)
DISPATCH_COUNT = 0
MATH_ABI = "mlx0322-expert-aligned-bm16-v1"

_HEADER_HASHES = {
    "mlx/backend/metal/kernels/steel/gemm/gemm.h": "c84af31e2c57154f2a8a24fa7f9fe2449765cce9a66f3f035846ed3c03b6a8b0",
    "mlx/backend/metal/kernels/steel/gemm/loader.h": "c1b82153670b1e18371a6e88fdbcf440af692957775c2f837700315ab8a5112e",
    "mlx/backend/metal/kernels/steel/defines.h": "b03cea6a7d5cfe814e6838d8536e2aacdcafd2744cb408e954fda336ce21759a",
    "mlx/backend/metal/kernels/steel/gemm/mma.h": "9b37709de40c3acaa5411875e52c8d26e6de093261741f671fbdab9c39a852b7",
    "mlx/backend/metal/kernels/steel/gemm/transforms.h": "73487e29beab5af9478bc931c49b5e8d66937c2fa2651499867f47ac5529afc2",
    "mlx/backend/metal/kernels/steel/utils.h": "d4c36298145d1c5617a94f07dbfabf6ac932afe2fd7ea0c9291f1fb385c5404a",
    "mlx/backend/metal/kernels/steel/utils/integral_constant.h": "886eccaac4f8f00eab5a10551fdb5315df7ab7462aba13d69bb95765c1a4735e",
    "mlx/backend/metal/kernels/steel/utils/type_traits.h": "4766c5a3809d3cf7740e63acc5703ef76b6b4e85d8baf16602c9d517a2fd426c",
    "mlx/backend/metal/kernels/steel/gemm/params.h": "6711e72e32310eafb088895d5a6d20fcfca814158f5b38eb72b6e69d9d773e43",
    "mlx/backend/metal/kernels/quantized_utils.h": "a12841b57d505f6cca81631901b845dbf76bc90dfc099e7f40c763b2cc838c51",
    "mlx/backend/metal/kernels/quantized.h": "2a007016da606afe569adb9adcc05e00f14558ad2e094bcb4f8974beb53c316f",
}
_SOURCE = """
uint block=threadgroup_position_in_grid.y;
if(block>=uint(block_ends[511]))return;
int lo=0,hi=512;
while(lo<hi){int mid=(lo+hi)/2;if(block>=uint(block_ends[mid]))lo=mid+1;else hi=mid;}
int expert=lo;
int first_block=expert==0?0:block_ends[expert-1];
int first_row=starts[expert]+(int(block)-first_block)*BM;
int count=min(BM,starts[expert+1]-first_row);
threadgroup T Xs[BM*(32+16/sizeof(T))];
threadgroup T Ws[32*(32+16/sizeof(T))];
uint3 local_tid=uint3(threadgroup_position_in_grid.x,0,0);
expert_aligned_impl<T,64,4,true,BM,32,32>(w+size_t(expert)*N*K/8,scales+size_t(expert)*N*K/64,biases+size_t(expert)*N*K/64,x+size_t(first_row)*K,y+size_t(first_row)*N,Xs,Ws,K,N,count,K,local_tid,thread_index_in_threadgroup,simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
"""


@lru_cache(maxsize=1)
def _kernel():
    if mx.__version__ != "0.32.2" or not mx.metal.is_available():
        return None
    try:
        inc = Path(mx.__file__).parent / "include"
        seen = {
            "mlx/backend/metal/kernels/" + f
            for f in (
                "utils.h",
                "bf16.h",
                "bf16_math.h",
                "complex.h",
                "defines.h",
                "logging.h",
            )
        }

        def expand(name):
            if name in seen:
                return ""
            source = (inc / name).read_text()
            if hashlib.sha256(source.encode()).hexdigest() != _HEADER_HASHES.get(name):
                raise ValueError("unqualified MLX header: " + name)
            seen.add(name)
            return re.sub(r'#include "([^"]+)"', lambda m: expand(m.group(1)), source)

        header = "\n".join(
            expand("mlx/backend/metal/kernels/" + f)
            for f in ("steel/gemm/gemm.h", "quantized_utils.h", "quantized.h")
        )
        source = (inc / "mlx/backend/metal/kernels/quantized.h").read_text()
        start = source.index("METAL_FUNC void qmm_t_impl(")
        end = source.index("{", start) + 1
        depth = 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        helper = source[start:end].replace("qmm_t_impl", "expert_aligned_impl")
        helper = helper.replace("const constant int&", "const int")
        helper = helper.replace("constexpr int WM = 2;", "constexpr int WM = 1;")
        header += (
            "\ntemplate<typename T,int group_size,int bits,bool aligned_N,int BM,int BK,int BN>\n"
            + helper
        )
        return mx.fast.metal_kernel(
            name="qwen4_expert_aligned_prefill",
            input_names=[
                "x",
                "w",
                "scales",
                "biases",
                "starts",
                "block_ends",
                "N",
                "K",
            ],
            output_names=["y"],
            source=_SOURCE,
            header=header,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning("Qwen4 aligned prefill unavailable; using stock MoE: %s", exc)
        return None


def aligned_switchglu(switch, x, indices):
    """Return unweighted routed output, or None for stock fallback."""
    if os.environ.get("VMLX_QWEN4_ALIGNED_MOE_PREFILL", "0") != "1":
        return None
    if (
        type(switch) is not SwitchGLU
        or switch.training
        or x.ndim not in (2, 3)
        or x.shape[-1] != 2560
        or indices.shape != (*x.shape[:-1], 10)
        or not 4096 <= indices.size <= 10240
        or indices.dtype not in (mx.int32, mx.uint32)
        or x.dtype not in (mx.float16, mx.bfloat16)
    ):
        return None
    projections = (switch.up_proj, switch.gate_proj, switch.down_proj)
    for proj, k, n in zip(projections, (2560, 2560, 640), (640, 640, 2560)):
        if (
            type(proj) is not QuantizedSwitchLinear
            or proj.bits != 4
            or proj.group_size != 64
            or proj.mode != "affine"
            or "bias" in proj
            or proj.weight.shape != (512, n, k // 8)
            or proj.weight.dtype != mx.uint32
            or proj.scales.shape != (512, n, k // 64)
            or proj.biases is None
            or proj.biases.shape != proj.scales.shape
            or proj.scales.dtype != x.dtype
            or proj.biases.dtype != x.dtype
        ):
            return None
    kernel = _kernel()
    if kernel is None:
        return None
    global DISPATCH_COUNT
    DISPATCH_COUNT += 1
    if DISPATCH_COUNT == 1:
        logger.info("Qwen4 expert-aligned MoE prefill active (%s)", MATH_ABI)
    sorted_x, idx, inverse = _gather_sort(mx.expand_dims(x, (-2, -3)), indices)
    rows = idx.size
    counts = mx.zeros((512,), mx.int32).at[idx].add(mx.ones((rows,), mx.int32))
    starts = mx.concatenate([mx.zeros((1,), mx.int32), mx.cumsum(counts)])
    ends = mx.cumsum((counts + 15) // 16)

    def project(proj, value):
        k, n = value.shape[-1], proj.weight.shape[1]
        return kernel(
            inputs=[value, proj.weight, proj.scales, proj.biases, starts, ends, n, k],
            template=[("T", value.dtype), ("BM", 16)],
            grid=(n // 32 * 64, (rows + 15) // 16 + 512, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[(rows, 1, n)],
            output_dtypes=[value.dtype],
        )[0]

    up = project(switch.up_proj, sorted_x)
    gate = project(switch.gate_proj, sorted_x)
    output = project(switch.down_proj, switch.activation(up, gate))
    return _scatter_unsort(output, inverse, indices.shape).squeeze(-2)
