"""MiniMax-M3 JANG loader quantization contracts."""

import mlx.core as mx
import mlx.nn as nn


def _tiny_m3_config():
    q = {
        "bits": 8,
        "group_size": 64,
    }
    for proj in ("gate_proj", "up_proj", "down_proj"):
        q[f"language_model.model.layers.3.block_sparse_moe.switch_mlp.{proj}"] = {
            "bits": 2,
            "group_size": 128,
            "mode": "affine",
        }
    return {
        "model_type": "minimax_m3_vl",
        "hidden_size": 256,
        "intermediate_size": 128,
        "num_local_experts": 4,
        "quantization": q,
    }


def _tiny_m3_model():
    from vmlx_engine.models.minimax_m3.m3_affine2_switch import (
        MiniMaxM3Affine2SwitchGLU,
    )

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.switch_mlp = MiniMaxM3Affine2SwitchGLU(
                256,
                128,
                4,
                bias=False,
            )

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = _MLP()

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [_Layer() for _ in range(4)]

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()

    return _Model()


def test_minimax_m3_switch_expert_rebuild_honors_forward_remapped_quantization():
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    from vmlx_engine.models.minimax_m3.m3_affine2_switch import (
        can_use_affine2_switchglu,
    )
    from vmlx_engine.utils.jang_loader import (
        _rebuild_minimax_m3_switch_experts,
        _upgrade_switch_to_quantized,
    )

    model = _tiny_m3_model()
    _upgrade_switch_to_quantized(model, bits=8, group_size=64)

    switch = model.model.layers[3].mlp.switch_mlp
    assert switch.gate_proj.bits == 8
    assert switch.gate_proj.group_size == 64

    rebuilt = _rebuild_minimax_m3_switch_experts(model, _tiny_m3_config())

    assert rebuilt == 3
    for proj_name, expected_in, expected_out in (
        ("gate_proj", 256, 128),
        ("up_proj", 256, 128),
        ("down_proj", 128, 256),
    ):
        proj = getattr(switch, proj_name)
        assert isinstance(proj, QuantizedSwitchLinear)
        assert proj.input_dims == expected_in
        assert proj.output_dims == expected_out
        assert proj.num_experts == 4
        assert proj.bits == 2
        assert proj.group_size == 128
        assert proj.mode == "affine"

    x = mx.zeros((1, 1, 256), dtype=mx.bfloat16)
    indices = mx.array([[[0, 1, 2, 3]]], dtype=mx.uint32)
    assert can_use_affine2_switchglu(switch, x, indices)
