"""Tests for JANG model loading — param count, tree_flatten, detection, MCP config."""
import pytest
from pathlib import Path


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
        assert not hasattr(mx, 'utils'), "mx.utils should not exist"

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
        config = {"mcpServers": {"test": {"command": "python3", "args": ["-c", "pass"]}}}
        result = validate_config(config)
        assert len(result.servers) == 1

    def test_servers_takes_precedence(self):
        """If both keys present, 'servers' wins (has server 'a')."""
        from vmlx_engine.mcp.config import validate_config
        config = {
            "servers": {"a": {"command": "python3", "args": ["-c", "pass"]}},
            "mcpServers": {"b": {"command": "python3", "args": ["-c", "pass"]}}
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
            validator.validate_env({"DYLD_INSERT_LIBRARIES": "/evil.dylib"}, "test-server")
