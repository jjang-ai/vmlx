# SPDX-License-Identifier: Apache-2.0
"""Opt-in GLM KDA prefill preparation with stock arithmetic/state boundaries.

One stride-aware dispatch replaces three convolutions and their SiLU passes.
Q/K normalization and the recurrent algorithm remain in their original MLX
paths. No quantized weight layout is assumed or changed by this helper.
"""

import logging
import os
from functools import lru_cache

import mlx.core as mx

logger = logging.getLogger(__name__)
_FAILED = False
_OBSERVED = False
_FLOATS = (mx.float16, mx.bfloat16, mx.float32)


def qkv_prefill_requested() -> bool:
    return os.environ.get("VMLX_GLM5_KDA_QKV_PREFILL", "0") == "1"


@lru_cache(maxsize=7)
def _kernel(width: int):
    lanes = []
    for name in ("q", "k", "v"):
        lanes.append(f"""
            float {name}_acc = 0.0f;
            for (uint tap = 0; tap < {width}u; ++tap) {{
                uint pos = t + tap;
                float value = pos < {width-1}u
                    ? float({name}s[pos*{name}s_strides[1]+c*{name}s_strides[2]])
                    : float({name}[(pos-{width-1}u)*{name}_strides[1]+c*{name}_strides[2]]);
                // Preserve each stock FP32 multiply and add; no FMA chain.
                volatile float product = value *
                    float({name}w[c*{name}w_strides[0]+tap*{name}w_strides[1]]);
                {name}_acc = {name}_acc + product;
            }}
            // Preserve stock MLX Sigmoid then Multiply then activation cast.
            // exp(-x), relaxed math or contracting (1-low)*acc is not exact.
            volatile float denom_{name} =
                1.0f + metal::precise::exp(metal::abs({name}_acc));
            float low_{name} = 1.0f / denom_{name};
            volatile float sigmoid_{name} =
                {name}_acc < 0.0f ? low_{name} : 1.0f - low_{name};
            {name}out[t*channels+c] = (T)({name}_acc * sigmoid_{name});
        """)
    return mx.fast.metal_kernel(
        name=f"vmlx_glm5_kda_qkv_prefill_w{width}",
        input_names=["q", "k", "v", "qs", "ks", "vs", "qw", "kw", "vw"],
        output_names=["qout", "kout", "vout"],
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
        source="""
            uint c = thread_position_in_grid.x, t = thread_position_in_grid.y;
            uint channels = q_shape[2], tokens = q_shape[1];
            if (c >= channels || t >= tokens) return;
        """ + "\n".join(lanes),
    )


def kda_qkv_prefill(xs, weights, states, *, enabled: bool):
    """Return three (SiLU output, raw-input history) pairs, or stock fallback.

    The bounded prefill path never changes a cache array in-place. A first
    compilation/materialization failure can safely fall back without appending
    state twice. Later asynchronous device errors propagate to the caller.
    Decode, batching and unqualified geometry keep their existing dispatcher.
    """
    global _FAILED, _OBSERVED
    if not enabled or _FAILED:
        return None
    if len(xs) != 3 or len(weights) != 3 or len(states) != 3:
        return None
    q = xs[0]
    if any(x.ndim != 3 or x.shape != q.shape or x.dtype != q.dtype for x in xs):
        return None
    batch, tokens, channels = q.shape
    if (
        batch != 1 or not 2 <= tokens <= 2048 or channels < 1
        or q.dtype not in _FLOATS
        or mx.default_device() != mx.gpu or not mx.metal.is_available()
    ):
        return None
    weights = tuple(w.reshape(w.shape[0], -1) if w.ndim == 3 else w for w in weights)
    if any(w.ndim != 2 or w.shape[0] != channels or w.dtype not in _FLOATS for w in weights):
        return None
    width = weights[0].shape[1]
    if not 2 <= width <= 8 or any(w.shape[1] != width for w in weights):
        return None
    if any(s is not None and (s.shape != (1, width-1, channels) or s.dtype not in _FLOATS)
           for s in states):
        return None
    states = tuple(mx.zeros((1, width-1, channels), dtype=q.dtype)
                   if s is None else s.astype(q.dtype) for s in states)
    try:
        outputs = _kernel(width)(
            inputs=[*xs, *states, *weights], template=[("T", q.dtype)],
            grid=(channels, tokens, 1), threadgroup=(min(channels, 128), 1, 1),
            output_shapes=[q.shape]*3, output_dtypes=[q.dtype]*3,
        )
        if not _OBSERVED:
            mx.eval(*outputs)
            _OBSERVED = True
            logger.info(
                "GLM KDA QKV prefill active: tokens=%d channels=%d width=%d "
                "activation=%s weights=%s outputs=separate math=safe_precise "
                "norm=stock state=raw_input_history",
                tokens, channels, width, q.dtype, tuple(str(w.dtype) for w in weights),
            )
        tails = tuple(x[:, -(width-1):] if tokens >= width-1 else
                      mx.concatenate([s, x], axis=1)[:, -(width-1):]
                      for x, s in zip(xs, states))
        return tuple(zip(outputs, tails))
    except (RuntimeError, ValueError) as exc:
        _FAILED = True
        logger.warning("GLM KDA QKV prefill disabled after launch failure: %s", exc)
        return None
