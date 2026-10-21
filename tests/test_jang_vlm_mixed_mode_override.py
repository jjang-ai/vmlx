"""Exercise the actual nested loader predicate without loading MLX."""
import ast
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "vmlx_engine/utils/jang_loader.py"


def owner(overrides, split_switch_paths=None):
    tree = ast.parse(SOURCE.read_text())
    names = {"_vlm_quant_module_path_candidates", "_per_module_override", "_split_expert_switch_override"}
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == 3
    scope = {"config": {"model_type": "glm5_next"}, "_qcfg_overrides": overrides, "Optional": __import__("typing").Optional,
             "_split_switch_paths": split_switch_paths or {}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope["_per_module_override"]


def test_mxfp_mode_survives_runtime_path_alias():
    override = owner({"model.layers.0.attn.q_proj": {"bits": 8, "group_size": 32, "mode": "mxfp8"}})
    assert override("language_model.model.layers.0.attn.q_proj") == {"bits": 8, "group_size": 32, "mode": "mxfp8"}


def test_unstamped_affine_override_stays_unstamped():
    override = owner({"lm_head": {"bits": 4, "group_size": 64}})
    assert override("language_model.lm_head") == {"bits": 4, "group_size": 64}
    assert override("missing") is None


def test_split_per_expert_stack_takes_the_experts_layout():
    q = {f"mtp.layers.0.mlp.experts.{e}.gate_proj": {"bits": 4, "group_size": 64, "mode": "affine"} for e in range(3)}
    override = owner(q, {"mtp.layers.0.mlp.switch_mlp.gate_proj": {0, 1, 2}})
    assert override("language_model.mtp.layers.0.mlp.switch_mlp.gate_proj") == {"bits": 4, "group_size": 64}
    assert override("language_model.mtp.layers.0.mlp.switch_mlp.up_proj") is None
