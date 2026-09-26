"""Payload rejection without Metal or model allocation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

PATH = Path(__file__).parents[1] / "vmlx_engine/jangh/payload.py"
spec = importlib.util.spec_from_file_location("tq_payload", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def model():
    linear = NS(tq2_packed=NS(shape=(2, 64, 4)), tq2_scales=NS(shape=(2, 64)))
    block = NS(is_jangtq2=True, gate_proj=linear, up_proj=linear, down_proj=linear)
    return NS(named_modules=lambda: [("language_model.model.layers.0.mlp.switch_mlp", block)])

KEY = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.tq2_packed"


def test_valid_partial_shard():
    module.validate_payload(model(), [(KEY, NS(shape=(2, 64, 4), dtype="mlx.core.uint32"))])


@pytest.mark.parametrize("shape,dtype", [((2, 64, 5), "mlx.core.uint32"), ((2, 64, 4), "mlx.core.float32")])
def test_invalid_payload_rejected(shape, dtype):
    with pytest.raises(ValueError, match="shape or dtype"):
        module.validate_payload(model(), [(KEY, NS(shape=shape, dtype=dtype))])


def test_unknown_projection_rejected():
    with pytest.raises(ValueError, match="unrecognized"):
        module.validate_payload(model(), [(KEY.replace("gate_proj", "other"), NS())])


def test_scale_dtype_rejected():
    with pytest.raises(ValueError, match="shape or dtype"):
        module.validate_payload(model(), [(KEY.replace("tq2_packed", "tq2_scales"), NS(shape=(2, 64), dtype="mlx.core.bfloat16"))])


def test_complete_shards_cannot_leave_zero_initialized_weights():
    expected = module.expected_payload(model())
    with pytest.raises(ValueError, match="missing"):
        module.validate_complete_payload(model(), set(expected) - {KEY})
    module.validate_complete_payload(model(), set(expected))
