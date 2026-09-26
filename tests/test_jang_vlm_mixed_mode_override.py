"""Exercise the actual nested loader predicate without loading MLX."""
import ast
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "vmlx_engine/utils/jang_loader.py"


def owner(overrides):
    tree = ast.parse(SOURCE.read_text())
    names = {"_vlm_quant_module_path_candidates", "_per_module_override"}
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == 2
    scope = {"config": {"model_type": "glm5_next"}, "_qcfg_overrides": overrides}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope["_per_module_override"]


def test_mxfp_mode_survives_runtime_path_alias():
    override = owner({"model.layers.0.attn.q_proj": {"bits": 8, "group_size": 32, "mode": "mxfp8"}})
    assert override("language_model.model.layers.0.attn.q_proj") == {"bits": 8, "group_size": 32, "mode": "mxfp8"}


def test_unstamped_affine_override_stays_unstamped():
    override = owner({"lm_head": {"bits": 4, "group_size": 64}})
    assert override("language_model.lm_head") == {"bits": 4, "group_size": 64}
    assert override("missing") is None
