"""Opt-in FP32 single-row GDN recurrence adapted from mlx-serve v26.9.6.

The MIT-licensed Metal body is in bonsai_gdn_recur.metal; its upstream license
is retained in third_party/mlx_serve/LICENSE. This path is deliberately restricted to
the measured Dealign Bonsai geometry and the FP32 decode stream.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import mlx.core as mx

_LOGGER = logging.getLogger(__name__)
_OBSERVED = False
_SCALES = None
_GEOMETRY = (16, 48, 128, 128, 10240)


def requested() -> bool:
    return os.environ.get("VMLX_BONSAI_GDN_FP32", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


@lru_cache(maxsize=1)
def _kernel():
    source_text = Path(__file__).with_suffix(".metal").read_text()
    marker = "constexpr int NSG = NT / 32;"
    header, body = source_text.split(marker, 1)
    return mx.fast.metal_kernel(
        name="vmlx_bonsai_gdn_recur_fp32",
        input_names=[
            "qkv", "a_in", "b_in", "conv_state", "state_in", "conv_w",
            "A_log", "dt_bias", "q_scale", "k_scale",
        ],
        output_names=["y", "conv_out", "state_out"],
        header=header,
        source=marker + body,
    )


def _scales():
    global _SCALES
    if _SCALES is None:
        inv = 128 ** -0.5
        _SCALES = (
            mx.array(inv ** 2, dtype=mx.float32),
            mx.array(inv, dtype=mx.float32),
        )
        mx.eval(*_SCALES)
    return _SCALES


def step(layer, qkv, a, b, conv_state, ssm_state):
    """Return (y, next_conv, next_ssm), or None if any contract differs."""
    if not requested() or ssm_state is None:
        return None
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None
    geometry = (
        int(layer.num_k_heads), int(layer.num_v_heads),
        int(layer.head_k_dim), int(layer.head_v_dim), int(layer.conv_dim),
    )
    if geometry != _GEOMETRY:
        return None
    hk, hv, dk, dv, channels = geometry
    if (
        tuple(qkv.shape) != (1, 1, channels)
        or tuple(a.shape) != (1, 1, hv)
        or tuple(b.shape) != (1, 1, hv)
        or tuple(conv_state.shape) != (1, 3, channels)
        or tuple(ssm_state.shape) != (1, hv, dv, dk)
        or tuple(layer.conv1d.weight.shape) != (channels, 4, 1)
        or tuple(layer.A_log.shape) != (hv,)
        or tuple(layer.dt_bias.shape) != (hv,)
    ):
        return None
    if any(value.dtype != mx.float32 for value in (
        qkv, a, b, conv_state, ssm_state,
        layer.conv1d.weight, layer.A_log, layer.dt_bias,
    )):
        return None
    q_scale, k_scale = _scales()
    y, next_conv, next_ssm = _kernel()(
        inputs=[
            qkv, a, b, conv_state, ssm_state, layer.conv1d.weight,
            layer.A_log, layer.dt_bias, q_scale, k_scale,
        ],
        template=[
            ("T", mx.float32), ("StT", mx.float32),
            ("HK", hk), ("HV", hv), ("DK", dk), ("DV", dv),
            ("C", channels), ("NT", 256), ("SPLIT", 4),
        ],
        grid=(hv * 4 * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, 1, hv, dv), (1, 3, channels), (1, hv, dv, dk)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    global _OBSERVED
    if not _OBSERVED:
        _LOGGER.info("Bonsai FP32 GDN recurrence engaged on %s", geometry)
        _OBSERVED = True
    return y, next_conv, next_ssm
