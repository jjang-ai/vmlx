# SPDX-License-Identifier: Apache-2.0
"""
Tests for CLI command parsing and dispatch.

Usage:
    pytest tests/test_cli_commands.py -v
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def _make_model_dir(tmp_path, config=None, name="test-model"):
    """Create a minimal model directory."""
    model_dir = tmp_path / name
    model_dir.mkdir(exist_ok=True)

    if config is None:
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
            "intermediate_size": 11008,
        }

    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)

    (model_dir / "model.safetensors").write_bytes(b"\x00" * 1024)
    return str(model_dir)


class TestConvertParser:
    """Tests for convert command argument parsing."""

    def test_convert_requires_bits(self):
        """--bits is required."""
        import argparse
        from vmlx_engine.cli import main

        with pytest.raises(SystemExit):
            with patch("sys.argv", ["vmlx-engine", "convert", "some-model", "--group-size", "64"]):
                main()

    def test_convert_requires_group_size(self):
        """--group-size is required."""
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["vmlx-engine", "convert", "some-model", "--bits", "4"]):
                from vmlx_engine.cli import main
                main()

    def test_convert_bits_choices(self):
        """--bits only accepts valid values."""
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["vmlx-engine", "convert", "model", "--bits", "5", "--group-size", "64"]):
                from vmlx_engine.cli import main
                main()

    def test_convert_mode_choices(self):
        """--mode only accepts valid values."""
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["vmlx-engine", "convert", "model", "--bits", "4", "--group-size", "64", "--mode", "invalid"]):
                from vmlx_engine.cli import main
                main()


class TestInfoCommand:
    """Tests for info command."""

    def test_info_basic(self, tmp_path, capsys):
        """info command displays model metadata."""
        model_dir = _make_model_dir(tmp_path)

        with patch("sys.argv", ["vmlx-engine", "info", model_dir]):
            try:
                from vmlx_engine.cli import main
                main()
            except SystemExit:
                pass

        output = capsys.readouterr().out
        assert "llama" in output.lower()
        assert "4096" in output

    def test_info_nonexistent(self, capsys):
        """info command fails gracefully for missing models."""
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["vmlx-engine", "info", "/nonexistent/model"]):
                from vmlx_engine.cli import main
                main()


class TestListCommand:
    """Tests for list command."""

    def test_list_empty_dir(self, tmp_path, capsys):
        """list command handles empty directories."""
        with patch("sys.argv", ["vmlx-engine", "list", str(tmp_path)]):
            try:
                from vmlx_engine.cli import main
                main()
            except SystemExit:
                pass

        output = capsys.readouterr().out
        assert "No models found" in output

    def test_list_finds_models(self, tmp_path, capsys):
        """list command finds models in directory."""
        _make_model_dir(tmp_path, name="model-a")
        _make_model_dir(tmp_path, name="model-b")

        with patch("sys.argv", ["vmlx-engine", "list", str(tmp_path)]):
            try:
                from vmlx_engine.cli import main
                main()
            except SystemExit:
                pass

        output = capsys.readouterr().out
        assert "model-a" in output
        assert "model-b" in output
        assert "Found 2" in output


class TestDoctorCommand:
    """Tests for doctor command."""

    def test_doctor_nonexistent(self, capsys):
        """doctor command fails for missing models."""
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["vmlx-engine", "doctor", "/nonexistent/model", "--no-inference"]):
                from vmlx_engine.cli import main
                main()


class TestDoctorChecks:
    """Tests for doctor diagnostic functions."""

    def test_check_config_valid(self, tmp_path):
        from vmlx_engine.commands.doctor import _check_config
        from vmlx_engine.utils.model_inspector import inspect_model

        model_dir = _make_model_dir(tmp_path)
        info = inspect_model(model_dir)
        issues = _check_config(info)

        assert len(issues) == 0

    def test_check_config_missing_type(self, tmp_path):
        from vmlx_engine.commands.doctor import _check_config
        from vmlx_engine.utils.model_inspector import inspect_model

        config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
            "intermediate_size": 11008,
        }
        model_dir = _make_model_dir(tmp_path, config)
        info = inspect_model(model_dir)
        issues = _check_config(info)

        assert any("model_type" in issue for issue in issues)

    def test_check_config_missing_architectures(self, tmp_path):
        from vmlx_engine.commands.doctor import _check_config
        from vmlx_engine.utils.model_inspector import inspect_model

        config = {
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
            "intermediate_size": 11008,
        }
        model_dir = _make_model_dir(tmp_path, config)
        info = inspect_model(model_dir)
        issues = _check_config(info)

        assert any("architectures" in issue for issue in issues)

    def test_check_architecture_hybrid_pattern_mismatch(self, tmp_path):
        from vmlx_engine.commands.doctor import _check_architecture
        from vmlx_engine.utils.model_inspector import inspect_model

        config = {
            "architectures": ["NemotronHForCausalLM"],
            "model_type": "nemotron_h",
            "hidden_size": 4096,
            "num_hidden_layers": 10,
            "num_attention_heads": 32,
            "vocab_size": 131072,
            "intermediate_size": 2688,
            "hybrid_override_pattern": "MEME*",  # Length 5, but 10 layers
        }
        model_dir = _make_model_dir(tmp_path, config)
        info = inspect_model(model_dir)
        issues = _check_architecture(info)

        assert any("pattern length" in issue for issue in issues)

    def test_check_weights_no_files(self, tmp_path):
        from vmlx_engine.commands.doctor import _check_weights
        from vmlx_engine.utils.model_inspector import ModelInfo

        info = ModelInfo(
            model_path=str(tmp_path),
            model_type="llama",
            architecture="LlamaForCausalLM",
            weight_files=[],
        )
        issues, warnings = _check_weights(str(tmp_path), info)

        assert any("No .safetensors" in issue for issue in issues)


class TestNemotronLatentMoePatch:
    """Tests for the LatentMoE monkey-patch bug fixes."""

    def test_patched_model_args_no_instance_field_leak(self):
        """_latent_moe_patched should be ClassVar, not an instance field."""
        from dataclasses import fields as dc_fields

        # This imports will trigger the patch if nemotron_h is available
        try:
            import importlib
            nemotron_h = importlib.import_module("mlx_lm.models.nemotron_h")
            ModelArgs = nemotron_h.ModelArgs

            if hasattr(ModelArgs, "_latent_moe_patched"):
                # Verify it's a ClassVar (not in dataclass fields)
                field_names = {f.name for f in dc_fields(ModelArgs)}
                assert "_latent_moe_patched" not in field_names, \
                    "_latent_moe_patched should be a ClassVar, not a dataclass field"
        except ImportError:
            pytest.skip("mlx_lm not available")

    def test_needs_latent_moe_patch_positive(self, tmp_path):
        """Models with nemotron_h + moe_latent_size should need the patch."""
        from vmlx_engine.utils.nemotron_latent_moe import needs_latent_moe_patch

        model_dir = tmp_path / "nemotron"
        model_dir.mkdir()
        with open(model_dir / "config.json", "w") as f:
            json.dump({
                "model_type": "nemotron_h",
                "moe_latent_size": 1024,
            }, f)

        assert needs_latent_moe_patch(str(model_dir)) is True

    def test_needs_latent_moe_patch_negative_no_latent(self, tmp_path):
        """Models without moe_latent_size should not need the patch."""
        from vmlx_engine.utils.nemotron_latent_moe import needs_latent_moe_patch

        model_dir = tmp_path / "nemotron-no-latent"
        model_dir.mkdir()
        with open(model_dir / "config.json", "w") as f:
            json.dump({
                "model_type": "nemotron_h",
            }, f)

        assert needs_latent_moe_patch(str(model_dir)) is False

    def test_needs_latent_moe_patch_negative_wrong_type(self, tmp_path):
        """Non-nemotron models should not need the patch."""
        from vmlx_engine.utils.nemotron_latent_moe import needs_latent_moe_patch

        model_dir = tmp_path / "llama"
        model_dir.mkdir()
        with open(model_dir / "config.json", "w") as f:
            json.dump({
                "model_type": "llama",
                "moe_latent_size": 1024,  # Would be weird but shouldn't trigger
            }, f)

        assert needs_latent_moe_patch(str(model_dir)) is False

    def test_needs_latent_moe_patch_nonexistent(self):
        """Non-existent paths should return False."""
        from vmlx_engine.utils.nemotron_latent_moe import needs_latent_moe_patch

        assert needs_latent_moe_patch("/nonexistent/path") is False
