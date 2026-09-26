"""Exercise actual VLM constructor admission without MLX or model allocation."""
import ast
import builtins
import copy
import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "vmlx_engine/jangtq2/contract.py"
SPEC = importlib.util.spec_from_file_location("vlm_admission_contract", CONTRACT_PATH)
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def valid_config():
    books = {}
    for bits, (alpha, beta) in contract.CUBIC_PARAMS.items():
        levels = []
        for index in range(1 << bits):
            u = index - ((1 << bits) - 1) / 2
            levels.append(contract._f32(u * (alpha + beta * u * u)))
        books[str(bits)] = dict(alpha=alpha, beta=beta, levels=levels)
    return types.SimpleNamespace(
        model_type="glm5_next", text_config=types.SimpleNamespace(swiglu_limit=10),
        vision_config=object(),
        jangtq=dict(version=2, packing="lsb-bitstream", scale_dtype="float16",
                    codebook_family="odd-cubic", rotation="hadamard32", codebooks=books),
        quantization={"model.layers.3.mlp.switch_mlp." + name:
                      dict(mode="jangtq2", bits=2) for name in contract.PROJECTIONS},
    )


@pytest.fixture
def constructor():
    source = ROOT / "vmlx_engine/models/glm5_next/vlm.py"
    tree = ast.parse(source.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Model")
    init = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    # Keep the method inside its real named class so zero-argument super has
    # its genuine __class__ closure; only heavy collaborators are replaced.
    cls = ast.ClassDef(name="Model", bases=[ast.Name(id="Base", ctx=ast.Load())],
                       keywords=[], body=[init], decorator_list=[])
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    calls = []

    class Base:
        pass

    def vision(config):
        calls.append(("vision", config))
        return object()

    def language(config):
        calls.append(("language", config))
        return object()

    env = {"__package__": "vmlx_engine.models.glm5_next", "Base": Base,
           "VisionModel": vision, "LanguageModel": language}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])),
                 str(source), "exec"), env)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if "jangtq2.install" in name or "jangtq2.kernels" in name or name.startswith("mlx"):
            raise AssertionError("heavy runtime import during rejected/ordinary admission: " + name)
        return real_import(name, *args, **kwargs)

    with patch.dict(sys.modules, {"vmlx_engine.jangtq2.contract": contract}), \
            patch("builtins.__import__", guarded_import):
        yield env["Model"], calls


@pytest.mark.parametrize("value", [True, "2", 3, None])
def test_bad_version_rejected_before_towers(constructor, value):
    model, calls = constructor
    config = valid_config()
    config.jangtq["version"] = value
    with pytest.raises(ValueError):
        model(config)
    assert calls == []


def test_alias_conflict_rejected_before_towers_and_mutation(constructor):
    model, calls = constructor
    config = valid_config()
    config.quantization.update({
        "model.a": dict(mode="mxfp8", bits=8, group_size=32),
        "model.b": dict(mode="mxfp8", bits=8, group_size=32),
        "language_model.model.b": dict(mode="affine", bits=8, group_size=64),
    })
    before = copy.deepcopy(config.quantization)
    with pytest.raises(ValueError, match="conflicting quantization alias"):
        model(config)
    assert calls == []
    assert config.quantization == before


def test_orphan_v2_entries_rejected_before_towers(constructor):
    model, calls = constructor
    config = valid_config()
    config.jangtq = None
    with pytest.raises(ValueError, match="require a format declaration"):
        model(config)
    assert calls == []


def test_ordinary_config_constructs_without_tq_runtime(constructor):
    model, calls = constructor
    config = valid_config()
    config.jangtq = None
    config.quantization = None
    result = model(config)
    assert calls == [("vision", config.vision_config), ("language", config.text_config)]
    assert result.config is config
    assert result.model_type == "glm5_next"
    assert result._jangtq2 is False
    assert result.vision_tower is not None
    assert result.language_model is not None
