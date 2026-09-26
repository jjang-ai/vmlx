"""Actual health metadata owner; no server/MLX initialization."""
import ast
from pathlib import Path


def status(config):
    path = Path(__file__).resolve().parents[1] / "vmlx_engine/server.py"
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {
        "_model_quantization_status", "_weight_matmul_dispatch_status"}]
    env = {
        "_read_bundle_json": lambda path, name: config if name == "config.json" else {},
        "_jangtq_bits_from_profile": lambda profile: None,
        "_find_routed_layer_bit_plan": lambda *args: (None, None),
        "_find_routed_projection_bit_plan": lambda *args: (None, None),
        "_find_routed_down_layer_bit_plan": lambda *args: (None, None),
        "_bundle_weight_index_status": lambda path: {},
        "_bundle_has_prestacked_jangtq": lambda path: False,
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+nodes, type_ignores=[])), str(path), "exec"), env)
    return env["_model_quantization_status"](None)


def test_tq2_mixed_metadata_does_not_report_affine_default_as_whole_model():
    config = {
        "jangtq": {"version": 2, "codebook_family": "odd-cubic", "rotation": "hadamard32"},
        "quantization": {"bits": 8, "mode": "affine", "group_size": 64,
            "model.a.gate_proj": {"mode": "jangtq2", "bits": 2},
            "model.a.down_proj": {"mode": "jangtq2", "bits": 3},
            "model.q": {"mode": "mxfp8", "bits": 8, "group_size": 32},
            "lm_head": {"mode": "affine", "bits": 8, "group_size": 64}},
    }
    result = status(config)
    assert result["codec"] == "jangtq2_codebook"
    assert result["weight_format"] == "jangtq2"
    assert result["mixed_precision"] is True
    assert "target_bits" not in result
    assert "group_size" not in result
    assert result["config_bits"] == 8
    assert result["routed_expert_bit_widths"] == [2, 3]
    assert result["jangtq2_projection_count"] == 2
    assert result["weight_matmul_dispatch"]["primary"] == "vmlx_jangtq2_custom_kernels"


def test_affine_metadata_retained():
    result = status({"quantization": {"bits": 8, "mode": "affine", "group_size": 64}})
    assert result["codec"] == "affine_quantized_matmul"
    assert result["target_bits"] == 8
    assert result["mixed_precision"] is False


def test_declaration_without_tq2_modules_does_not_claim_loaded_codec():
    result = status({"jangtq": {"version": 2}, "quantization": {"mode": "affine", "bits": 8}})
    assert result["codec"] == "affine_quantized_matmul"


def test_tq2_acceleration_is_capability_not_observed_dispatch():
    path = Path(__file__).resolve().parents[1] / "vmlx_engine/server.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_model_acceleration_status")
    env = {
        "_model_path": None,
        "_model_quantization_status": lambda path: {"codec": "jangtq2_codebook"},
        "_mlx_metal_na_status": lambda: {"available": True},
        "_host_supports_metal_na": lambda: {"supported": True},
        "_family_acceleration_contract": lambda path: {},
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), str(path), "exec"), env)
    result = env["_model_acceleration_status"]()
    assert result["kernel_type"] == "jangtq2_codebook"
    assert result["metal_na_capable"] is True
    assert result["metal_na_active_on_host"] is False
    assert result["dispatch_observed"] is False
    assert result["prefill_backends"] == ["nax", "steel"]
