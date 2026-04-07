"""Tests for JANG model loading — param count, tree_flatten, detection, MCP config."""

import pytest
from pathlib import Path


class TestGemma4SwitchMlpRename:
    """Verify that Gemma 4 JANG switch_mlp weights are renamed to experts.switch_glu.

    JANG-quantised Gemma 4 MoE models store expert weights under the key pattern
      model.language_model.layers.<n>.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}
    but mlx-vlm's Gemma4TextModel wraps them inside an Experts → SwitchGLU module
    whose parameter paths look like
      language_model.model.layers.<n>.experts.switch_glu.{gate,up,down}_proj.{weight,scales,biases}

    The rename must happen *before* model.load_weights() so that strict=False does
    not silently discard all expert weights (which would leave the model running on
    random-initialised experts and produce garbled output).
    """

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _build_shard(layer: int = 0) -> dict:
        """Return a minimal dict mimicking one JANG shard for a Gemma 4 layer."""
        prefix = f"model.language_model.layers.{layer}"
        keys = [
            f"{prefix}.switch_mlp.gate_proj.weight",
            f"{prefix}.switch_mlp.gate_proj.scales",
            f"{prefix}.switch_mlp.gate_proj.biases",
            f"{prefix}.switch_mlp.up_proj.weight",
            f"{prefix}.switch_mlp.up_proj.scales",
            f"{prefix}.switch_mlp.up_proj.biases",
            f"{prefix}.switch_mlp.down_proj.weight",
            f"{prefix}.switch_mlp.down_proj.scales",
            f"{prefix}.switch_mlp.down_proj.biases",
            # Dense MLP keys that must NOT be renamed
            f"{prefix}.mlp.gate_proj.weight",
            f"{prefix}.mlp.gate_proj.scales",
            # Router key — also must not be renamed
            f"{prefix}.router.proj.weight",
        ]
        return {k: None for k in keys}

    @staticmethod
    def _apply_rename(weights: dict) -> dict:
        """Reproduce the rename logic from jang_loader.py."""
        if any(".switch_mlp." in k for k in weights):
            return {
                (
                    k.replace(".switch_mlp.", ".experts.switch_glu.")
                    if ".switch_mlp." in k
                    else k
                ): v
                for k, v in weights.items()
            }
        return weights

    # ------------------------------------------------------------------ tests
    def test_switch_mlp_keys_are_renamed(self):
        """All switch_mlp keys must be replaced by experts.switch_glu equivalents."""
        weights = self._build_shard()
        result = self._apply_rename(weights)
        for k in result:
            assert ".switch_mlp." not in k, f"switch_mlp survived rename: {k}"

    def test_experts_switch_glu_keys_present_after_rename(self):
        """gate_proj, up_proj, down_proj must appear under experts.switch_glu."""
        weights = self._build_shard(layer=3)
        result = self._apply_rename(weights)
        prefix = "model.language_model.layers.3.experts.switch_glu"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight", "scales", "biases"):
                expected = f"{prefix}.{proj}.{suffix}"
                assert expected in result, f"Missing expected key: {expected}"

    def test_non_switch_mlp_keys_are_unchanged(self):
        """Dense MLP and router keys must pass through the rename unmodified."""
        weights = self._build_shard()
        result = self._apply_rename(weights)
        unchanged = [
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "model.language_model.layers.0.mlp.gate_proj.scales",
            "model.language_model.layers.0.router.proj.weight",
        ]
        for k in unchanged:
            assert k in result, f"Non-switch_mlp key was unexpectedly mutated: {k}"

    def test_key_count_preserved(self):
        """Rename must be 1-to-1: output dict has same number of entries as input."""
        weights = self._build_shard()
        result = self._apply_rename(weights)
        assert len(result) == len(weights)

    def test_no_rename_when_no_switch_mlp_keys(self):
        """If no switch_mlp keys are present the dict is returned as-is."""
        weights = {
            "model.language_model.layers.0.mlp.gate_proj.weight": None,
            "model.language_model.layers.0.self_attn.q_proj.weight": None,
        }
        result = self._apply_rename(weights)
        assert result == weights

    def test_rename_applied_in_jang_loader_source(self):
        """Verify the jang_loader source actually contains the switch_mlp rename."""
        import inspect
        from vmlx_engine.utils import jang_loader

        src = inspect.getsource(jang_loader)
        assert ".switch_mlp." in src and ".experts.switch_glu." in src, (
            "jang_loader.py does not contain the switch_mlp → experts.switch_glu rename"
        )


class TestJangDetection:
    """Verify is_jang_model detects config files correctly."""

    def test_detects_jang_config(self, tmp_path):
        from vmlx_engine.utils.jang_loader import is_jang_model

        (tmp_path / "jang_config.json").write_text("{}")
        assert is_jang_model(str(tmp_path)) is True

    def test_detects_jjqf_config(self, tmp_path):
        from vmlx_engine.utils.jang_loader import is_jang_model

        (tmp_path / "jjqf_config.json").write_text("{}")
        assert is_jang_model(str(tmp_path)) is True

    def test_no_config_returns_false(self, tmp_path):
        from vmlx_engine.utils.jang_loader import is_jang_model

        assert is_jang_model(str(tmp_path)) is False

    def test_hf_repo_id_returns_false(self):
        from vmlx_engine.utils.jang_loader import is_jang_model

        assert is_jang_model("mlx-community/Llama-3.2-3B-4bit") is False

    def test_nonexistent_path_returns_false(self):
        from vmlx_engine.utils.jang_loader import is_jang_model

        assert is_jang_model("/this/does/not/exist") is False


class TestTreeFlattenImport:
    """Verify the tree_flatten fix uses correct import path."""

    def test_mlx_utils_tree_flatten_exists(self):
        from mlx.utils import tree_flatten

        assert callable(tree_flatten)

    def test_mx_core_has_no_utils(self):
        """mlx.core does NOT have a utils attribute."""
        import mlx.core as mx

        assert not hasattr(mx, "utils"), "mx.utils should not exist"

    def test_tree_flatten_on_nested_dict(self):
        import mlx.core as mx
        from mlx.utils import tree_flatten

        params = {"layers": [{"weight": mx.zeros((3, 3)), "bias": mx.zeros((3,))}]}
        flat = tree_flatten(params)
        assert len(flat) == 2
        total = sum(p.size for _, p in flat)
        assert total == 12  # 9 + 3

    def test_jang_loader_uses_correct_import(self):
        """jang_loader.py uses 'from mlx.utils import tree_flatten', not mx.utils."""
        import inspect
        from vmlx_engine.utils import jang_loader

        source = inspect.getsource(jang_loader)
        assert "from mlx.utils import tree_flatten" in source
        assert "mx.utils.tree_flatten" not in source


class TestMCPConfigKeys:
    """Verify MCP config accepts both 'servers' and 'mcpServers' keys."""

    def test_accepts_servers_key(self):
        from vmlx_engine.mcp.config import validate_config

        config = {"servers": {"test": {"command": "python3", "args": ["-c", "pass"]}}}
        result = validate_config(config)
        assert len(result.servers) == 1

    def test_accepts_mcpServers_key(self):
        from vmlx_engine.mcp.config import validate_config

        config = {
            "mcpServers": {"test": {"command": "python3", "args": ["-c", "pass"]}}
        }
        result = validate_config(config)
        assert len(result.servers) == 1

    def test_servers_takes_precedence(self):
        """If both keys present, 'servers' wins (has server 'a')."""
        from vmlx_engine.mcp.config import validate_config

        config = {
            "servers": {"a": {"command": "python3", "args": ["-c", "pass"]}},
            "mcpServers": {"b": {"command": "python3", "args": ["-c", "pass"]}},
        }
        result = validate_config(config)
        # servers key takes precedence — should have 'a', not 'b'
        server_names = [name for name in result.servers]
        assert "a" in server_names

    def test_empty_config_returns_no_servers(self):
        from vmlx_engine.mcp.config import validate_config

        result = validate_config({})
        assert len(result.servers) == 0


class TestMCPSecurityUnblocked:
    """Verify PYTHONPATH and PATH are no longer blocked."""

    def test_pythonpath_allowed(self):
        from vmlx_engine.mcp.security import MCPCommandValidator

        validator = MCPCommandValidator()
        # Should not raise
        validator.validate_env({"PYTHONPATH": "/some/path"}, "test-server")

    def test_path_allowed(self):
        from vmlx_engine.mcp.security import MCPCommandValidator

        validator = MCPCommandValidator()
        validator.validate_env({"PATH": "/usr/bin:/usr/local/bin"}, "test-server")

    def test_node_path_allowed(self):
        from vmlx_engine.mcp.security import MCPCommandValidator

        validator = MCPCommandValidator()
        validator.validate_env({"NODE_PATH": "/some/node/path"}, "test-server")

    def test_ld_preload_still_blocked(self):
        from vmlx_engine.mcp.security import MCPCommandValidator, MCPSecurityError

        validator = MCPCommandValidator()
        with pytest.raises(MCPSecurityError):
            validator.validate_env({"LD_PRELOAD": "/evil.so"}, "test-server")

    def test_dyld_insert_still_blocked(self):
        from vmlx_engine.mcp.security import MCPCommandValidator, MCPSecurityError

        validator = MCPCommandValidator()
        with pytest.raises(MCPSecurityError):
            validator.validate_env(
                {"DYLD_INSERT_LIBRARIES": "/evil.dylib"}, "test-server"
            )
