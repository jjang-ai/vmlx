"""JANGT per-module contract keys are "model.layers.<L>.mlp.switch_mlp" whatever wraps the decoder.

The GLM-5.3-Flash VLM nests the decoder under "language_model."; the old derivation split at the first "model." (inside
"language_model.") and looked up "model.model.layers..." -> every bundle refused to load (2026-10-10).
"""
from __future__ import annotations

import pytest

pytest.importorskip("mlx.core")

from vmlx_engine.jangt.switch import contract_key  # noqa: E402


@pytest.mark.parametrize("path,key", [
    ("model.layers.44.mlp", "model.layers.44.mlp.switch_mlp"),                          # text model (N0.5)
    ("language_model.model.layers.44.mlp", "model.layers.44.mlp.switch_mlp"),           # GLM VLM
    ("model.language_model.model.layers.3.mlp", "model.layers.3.mlp.switch_mlp"),       # double wrapper
])
def test_contract_key(path, key):
    assert contract_key(path) == key


def test_install_finds_entries_under_a_vlm_wrapper():
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import SwitchGLU
    from vmlx_engine.jangt.switch import JTMixedSwitchGLU, install_jangt

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.switch_mlp = SwitchGLU(256, 128, 4)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = MLP()

    class Dec(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Layer()]

    class Text(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Dec()

    class VLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = Text()

    m = VLM()
    q = {f"model.layers.0.mlp.switch_mlp.{p}": {"mode": "jangtq2", "bits": 2, "rotation": "hadamard32"}
         for p in ("gate_proj", "up_proj", "down_proj")}
    n = install_jangt(m, {"jangt": {"version": 1, "code": "v2_halfbits", "state_bits": 12}, "quantization": q})
    assert n == 1 and isinstance(m.language_model.model.layers[0].mlp.switch_mlp, JTMixedSwitchGLU)


def test_compute_dtype_cast_keeps_jangt_format_tensors():
    """MLA / large-expert models get bf16 compute (_set_jang_compute_dtype). JANGT format tensors must keep their
    contract dtypes: the NAX prefill kernel takes float scales and did not compile on bf16 ones (2026-10-10)."""
    import mlx.core as mx
    import mlx.nn as nn
    from vmlx_engine.jangt.switch import JTSwitchLinear
    from vmlx_engine.utils.jang_loader import _set_jang_compute_dtype

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.jt = JTSwitchLinear(256, 64, 2, 2, 2, True)
            self.dense = nn.Linear(8, 8)

    m = M()
    _set_jang_compute_dtype(m, mx.bfloat16, {"model_type": "glm5_next"})
    assert m.dense.weight.dtype == mx.bfloat16
    assert m.jt.jt_scale.dtype == mx.float32 and m.jt.jt_su.dtype == mx.float32 and m.jt.jt_cs.dtype == mx.float32
    assert m.jt.jt_wout.dtype == mx.float16 and m.jt.jt_packed.dtype == mx.uint32
