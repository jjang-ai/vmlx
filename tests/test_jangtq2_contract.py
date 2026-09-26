"""Format checks must run without importing MLX or constructing a model."""
import copy
import importlib.util
from pathlib import Path
import unittest

_PATH = Path(__file__).resolve().parents[1] / "vmlx_engine/jangh/contract.py"
_SPEC = importlib.util.spec_from_file_location("jangtq2_contract_under_test", _PATH)
contract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(contract)


def bundle():
    books = {}
    for bits, (alpha, beta) in contract.CUBIC_PARAMS.items():
        levels = []
        for index in range(1 << bits):
            u = index - ((1 << bits) - 1) / 2
            levels.append(contract._f32(u * (alpha + beta * u * u)))
        books[str(bits)] = dict(alpha=alpha, beta=beta, levels=levels)
    return {
        "jangtq": dict(version=2, packing="lsb-bitstream", scale_dtype="float16",
                       codebook_family="odd-cubic", rotation="hadamard32", codebooks=books),
        "quantization": {
            "model.layers.3.mlp.switch_mlp." + name: dict(mode="jangtq2", bits=2)
            for name in contract.PROJECTIONS
        },
    }


class TestContract(unittest.TestCase):
    def test_valid_and_default_rotation(self):
        cfg = bundle()
        self.assertTrue(contract.validate_format(cfg))
        stacks = contract.projection_contract(cfg)
        self.assertEqual(stacks["model.layers.3.mlp.switch_mlp"]["down_proj"],
                         dict(bits=2, rotation="hadamard32"))

    def test_ordinary_and_v1_unchanged(self):
        for cfg in ({}, {"jangtq": None}, {"jangtq": {"version": 1}}):
            self.assertFalse(contract.validate_format(cfg))

    def test_malformed_version_and_declarations(self):
        for value in (True, "2", 2.0, 3, None):
            cfg = bundle()
            cfg["jangtq"]["version"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                contract.validate_format(cfg)
        for value in ([], "v2"):
            with self.assertRaises(ValueError):
                contract.validate_format({"jangtq": value})

    def test_unknown_format_fields(self):
        for key in ("packing", "scale_dtype", "codebook_family", "rotation"):
            cfg = bundle()
            cfg["jangtq"][key] = "unsupported"
            with self.subTest(key=key), self.assertRaises(ValueError):
                contract.validate_format(cfg)

    def test_codebook_mismatch(self):
        for key in ("alpha", "beta"):
            for value in (True, float("nan"), float("inf"), 0.123):
                cfg = bundle()
                cfg["jangtq"]["codebooks"]["2"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    contract.validate_format(cfg)
        cfg = bundle()
        cfg["jangtq"]["codebooks"]["3"]["levels"][2] += 0.1
        with self.assertRaises(ValueError):
            contract.validate_format(cfg)

    def test_missing_declaration_and_orphan_projection(self):
        cfg = bundle()
        del cfg["jangtq"]
        with self.assertRaises(ValueError):
            contract.validate_format(cfg)
        cfg = bundle()
        cfg["quantization"]["model.layers.4.mlp.switch_mlp.up_proj"] = dict(mode="jangtq2", bits=2)
        with self.assertRaises(ValueError):
            contract.projection_contract(cfg)

    def test_bits_rotation_and_sibling_mismatch(self):
        for override in ({"bits": True}, {"bits": 6}, {"bits": 3}, {"rotation": "none"}, {"rotation": "bad"}):
            cfg = bundle()
            cfg["quantization"]["model.layers.3.mlp.switch_mlp.up_proj"].update(override)
            with self.subTest(override=override), self.assertRaises(ValueError):
                contract.projection_contract(cfg)

    def test_alias_conflict_is_atomic(self):
        cfg = {"model.first": dict(bits=8, mode="mxfp8", group_size=32),
               "model.second": dict(bits=8, mode="mxfp8", group_size=32),
               "language_model.model.second": dict(bits=8, mode="affine", group_size=64)}
        before = copy.deepcopy(cfg)
        with self.assertRaises(ValueError):
            contract.alias_runtime_quant_keys(cfg)
        self.assertEqual(cfg, before)

    def test_alias_idempotence_and_mode(self):
        cfg = {"bits": 8, "model.first": dict(bits=8, mode="mxfp8", group_size=32)}
        self.assertEqual(contract.alias_runtime_quant_keys(cfg), 1)
        self.assertEqual(contract.alias_runtime_quant_keys(cfg), 0)
        self.assertEqual(cfg["language_model.model.first"]["mode"], "mxfp8")
        self.assertEqual(cfg["bits"], 8)

    def test_routed_aliases_accounted(self):
        cfg = bundle()
        name = "model.layers.3.mlp.switch_mlp.down_proj"
        cfg["quantization"]["language_model." + name] = dict(cfg["quantization"][name])
        self.assertEqual(len(contract.projection_contract(cfg)), 1)
        cfg["quantization"]["language_model." + name]["bits"] = 3
        with self.assertRaises(ValueError):
            contract.projection_contract(cfg)


class TestInstaller(unittest.TestCase):
    def setUp(self):
        import ast
        import logging
        import sys
        import types
        from unittest.mock import patch

        class Linear:
            def __init__(self, shape):
                self.weight = types.SimpleNamespace(shape=shape)

        class Stock:
            def __init__(self, width=32):
                self.gate_proj = Linear((2, 64, width))
                self.up_proj = Linear((2, 64, width))
                self.down_proj = Linear((2, width, 64))

        class Replacement:
            def __init__(self, D, I, E, gu, dn, limit,
                         rotation_gate_up, rotation_down):
                self.limit = limit
                self.gate_proj = types.SimpleNamespace(bits=gu, rotation=rotation_gate_up)
                self.up_proj = types.SimpleNamespace(bits=gu, rotation=rotation_gate_up)
                self.down_proj = types.SimpleNamespace(bits=dn, rotation=rotation_down)

        fake = types.ModuleType("mlx_lm.models.switch_layers")
        fake.SwitchGLU = Stock
        self.modules = patch.dict(sys.modules, {"mlx_lm.models.switch_layers": fake})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        source = _PATH.with_name("install.py")
        tree = ast.parse(source.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in {"_cfg_key", "install_jangtq2"}]
        env = {"TQSwitchGLU": Replacement, "projection_contract": contract.projection_contract,
               "logger": logging.getLogger("installer-test")}
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[])),
                     str(source), "exec"), env)
        self.install = env["install_jangtq2"]
        self.stock = Stock
        self.replacement = Replacement
        self.owners = [types.SimpleNamespace(switch_mlp=Stock()) for _ in range(2)]
        self.model = types.SimpleNamespace(named_modules=lambda: [
            (f"language_model.model.layers.{i + 3}.mlp", owner)
            for i, owner in enumerate(self.owners)])
        self.config = bundle()
        for key, value in list(self.config["quantization"].items()):
            self.config["quantization"][key.replace("layers.3", "layers.4")] = dict(value)

    def test_two_stacks_and_idempotency(self):
        self.assertEqual(self.install(self.model, self.config), 2)
        before = [owner.switch_mlp for owner in self.owners]
        self.assertTrue(all(isinstance(item, self.replacement) for item in before))
        self.assertEqual(self.install(self.model, self.config), 0)
        self.assertEqual([owner.switch_mlp for owner in self.owners], before)

    def test_unrotated_short_width_rejected_before_mutation(self):
        self.owners[1].switch_mlp = self.stock(width=16)
        self.config["jangtq"]["rotation"] = "none"
        before = [owner.switch_mlp for owner in self.owners]
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.install(self.model, self.config)
        self.assertEqual([owner.switch_mlp for owner in self.owners], before)

    def test_inventory_missing_or_unknown_rejected_before_mutation(self):
        for remove in (True, False):
            cfg = copy.deepcopy(self.config)
            for key in list(cfg["quantization"]):
                if "layers.4" in key:
                    entry = cfg["quantization"].pop(key)
                    if not remove:
                        cfg["quantization"][key.replace("layers.4", "layers.9")] = entry
            before = [owner.switch_mlp for owner in self.owners]
            with self.subTest(remove=remove), self.assertRaisesRegex(ValueError, "inventory"):
                self.install(self.model, cfg)
            self.assertEqual([owner.switch_mlp for owner in self.owners], before)

    def test_idempotent_contract_mismatch_rejected(self):
        self.install(self.model, self.config)
        self.owners[1].switch_mlp.down_proj.bits = 3
        with self.assertRaisesRegex(ValueError, "installed contract"):
            self.install(self.model, self.config)


if __name__ == "__main__":
    unittest.main()
