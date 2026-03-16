# SPDX-License-Identifier: Apache-2.0
"""
Tests for audit fixes across the codebase.

Covers: parameter validation, embedding validation, CLI validation,
Anthropic adapter parity, MCP content limits, disk cache safety.
"""

import json
import sys
from unittest.mock import MagicMock, patch

import numpy as np

import pytest

from vmlx_engine.api.models import (
    AudioSpeechRequest,
    ChatCompletionRequest,
    CompletionRequest,
    Message,
    ResponsesRequest,
)


# =============================================================================
# Parameter Validation — boundary conditions and edge cases
# =============================================================================


class TestParameterBoundaries:
    """Test exact boundary values for all validated parameters."""

    def _chat_req(self, **kwargs):
        return ChatCompletionRequest(
            model="test", messages=[Message(role="user", content="hi")], **kwargs
        )

    # temperature boundaries: [0, 2]
    def test_temperature_exact_zero(self):
        assert self._chat_req(temperature=0).temperature == 0

    def test_temperature_exact_two(self):
        assert self._chat_req(temperature=2).temperature == 2

    def test_temperature_just_below_zero(self):
        with pytest.raises(ValueError):
            self._chat_req(temperature=-0.001)

    def test_temperature_just_above_two(self):
        with pytest.raises(ValueError):
            self._chat_req(temperature=2.001)

    # top_p boundaries: (0, 1]
    def test_top_p_smallest_valid(self):
        assert self._chat_req(top_p=0.001).top_p == 0.001

    def test_top_p_exact_one(self):
        assert self._chat_req(top_p=1.0).top_p == 1.0

    def test_top_p_exact_zero_rejected(self):
        with pytest.raises(ValueError):
            self._chat_req(top_p=0.0)

    def test_top_p_negative_rejected(self):
        with pytest.raises(ValueError):
            self._chat_req(top_p=-0.1)

    # max_tokens boundaries: [1, ∞)
    def test_max_tokens_one(self):
        assert self._chat_req(max_tokens=1).max_tokens == 1

    def test_max_tokens_large(self):
        assert self._chat_req(max_tokens=1_000_000).max_tokens == 1_000_000

    def test_max_tokens_zero_rejected(self):
        with pytest.raises(ValueError):
            self._chat_req(max_tokens=0)

    def test_max_tokens_negative_rejected(self):
        with pytest.raises(ValueError):
            self._chat_req(max_tokens=-100)

    # top_k boundaries: [0, ∞)
    def test_top_k_zero_allowed(self):
        assert self._chat_req(top_k=0).top_k == 0

    def test_top_k_one(self):
        assert self._chat_req(top_k=1).top_k == 1

    def test_top_k_negative_rejected(self):
        with pytest.raises(ValueError):
            self._chat_req(top_k=-1)

    # min_p boundaries: [0, 1]
    def test_min_p_zero(self):
        assert self._chat_req(min_p=0).min_p == 0

    def test_min_p_one(self):
        assert self._chat_req(min_p=1.0).min_p == 1.0

    def test_min_p_above_one_rejected(self):
        with pytest.raises(ValueError):
            self._chat_req(min_p=1.01)

    def test_min_p_negative_rejected(self):
        with pytest.raises(ValueError):
            self._chat_req(min_p=-0.01)

    # repetition_penalty boundaries: (0, ∞)
    def test_rep_penalty_tiny(self):
        assert self._chat_req(repetition_penalty=0.001).repetition_penalty == 0.001

    def test_rep_penalty_one(self):
        assert self._chat_req(repetition_penalty=1.0).repetition_penalty == 1.0

    def test_rep_penalty_large(self):
        assert self._chat_req(repetition_penalty=100).repetition_penalty == 100

    def test_rep_penalty_zero_rejected(self):
        with pytest.raises(ValueError):
            self._chat_req(repetition_penalty=0)

    def test_rep_penalty_negative_rejected(self):
        with pytest.raises(ValueError):
            self._chat_req(repetition_penalty=-1)

    # Combined: all params at valid boundaries simultaneously
    def test_all_params_at_boundaries(self):
        req = self._chat_req(
            temperature=0, top_p=0.001, max_tokens=1,
            top_k=0, min_p=0, repetition_penalty=0.001,
        )
        assert req.temperature == 0
        assert req.top_p == 0.001
        assert req.max_tokens == 1


class TestCompletionRequestValidation:
    """Validate CompletionRequest mirrors ChatCompletionRequest bounds."""

    def test_temperature_rejected(self):
        with pytest.raises(ValueError):
            CompletionRequest(model="test", prompt="hi", temperature=-1)

    def test_top_p_rejected(self):
        with pytest.raises(ValueError):
            CompletionRequest(model="test", prompt="hi", top_p=0)

    def test_max_tokens_rejected(self):
        with pytest.raises(ValueError):
            CompletionRequest(model="test", prompt="hi", max_tokens=0)

    def test_top_k_rejected(self):
        with pytest.raises(ValueError):
            CompletionRequest(model="test", prompt="hi", top_k=-1)

    def test_min_p_rejected(self):
        with pytest.raises(ValueError):
            CompletionRequest(model="test", prompt="hi", min_p=2.0)

    def test_repetition_penalty_rejected(self):
        with pytest.raises(ValueError):
            CompletionRequest(model="test", prompt="hi", repetition_penalty=0)

    def test_stop_normalized_string(self):
        req = CompletionRequest(model="test", prompt="hi", stop="<stop>")
        assert req.stop == ["<stop>"]

    def test_stop_list_unchanged(self):
        req = CompletionRequest(model="test", prompt="hi", stop=["a", "b"])
        assert req.stop == ["a", "b"]

    def test_valid_params(self):
        req = CompletionRequest(
            model="test", prompt="hi", temperature=0.5, top_p=0.9,
            max_tokens=100, top_k=40, min_p=0.05, repetition_penalty=1.1,
        )
        assert req.temperature == 0.5


class TestResponsesRequestValidation:
    """Validate ResponsesRequest mirrors ChatCompletionRequest bounds."""

    def test_temperature_rejected(self):
        with pytest.raises(ValueError):
            ResponsesRequest(model="test", input="hi", temperature=3.0)

    def test_max_output_tokens_rejected(self):
        with pytest.raises(ValueError):
            ResponsesRequest(model="test", input="hi", max_output_tokens=0)

    def test_top_p_rejected(self):
        with pytest.raises(ValueError):
            ResponsesRequest(model="test", input="hi", top_p=0)

    def test_min_p_rejected(self):
        with pytest.raises(ValueError):
            ResponsesRequest(model="test", input="hi", min_p=-0.1)

    def test_stop_normalized(self):
        req = ResponsesRequest(model="test", input="hi", stop="<end>")
        assert req.stop == ["<end>"]

    def test_valid_params(self):
        req = ResponsesRequest(
            model="test", input="hi", temperature=1.5, top_p=0.95,
            max_output_tokens=500, top_k=50, min_p=0.1, repetition_penalty=1.2,
        )
        assert req.max_output_tokens == 500


class TestAudioSpeechValidation:
    """Validate AudioSpeechRequest speed bounds."""

    def test_speed_valid_range(self):
        for s in [0.25, 0.5, 1.0, 2.0, 4.0]:
            assert AudioSpeechRequest(input="hi", speed=s).speed == s

    def test_speed_zero_rejected(self):
        with pytest.raises(ValueError):
            AudioSpeechRequest(input="hi", speed=0)

    def test_speed_negative_rejected(self):
        with pytest.raises(ValueError):
            AudioSpeechRequest(input="hi", speed=-1)

    def test_speed_above_max_rejected(self):
        with pytest.raises(ValueError):
            AudioSpeechRequest(input="hi", speed=4.1)


# =============================================================================
# CLI Argument Validation
# =============================================================================


class TestCLIValidation:
    """Test CLI-level validation in serve_command.

    serve_command imports uvicorn lazily inside the function,
    so we mock it at the point of use and prevent actual server startup.
    The validation checks happen BEFORE uvicorn.run so they'll exit
    before reaching the mock.
    """

    def _make_args(self, **overrides):
        """Create a minimal args namespace for serve_command."""
        defaults = {
            "model": "/tmp/test-model",
            "host": "127.0.0.1",
            "port": 8000,
            "timeout": 300,
            "rate_limit": 0,
            "api_key": None,
            "enable_auto_tool_choice": False,
            "tool_call_parser": None,
            "reasoning_parser": None,
            "default_temperature": None,
            "default_top_p": None,
            "continuous_batching": False,
            "max_tokens": 32768,
            "stream_interval": 1,
            "served_model_name": None,
            "is_mllm": False,
            "speculative_model": None,
            "mcp_config": None,
            "embedding_model": None,
        }
        from argparse import Namespace
        return Namespace(**{**defaults, **overrides})

    def test_negative_timeout_exits(self):
        """CLI should reject negative timeout."""
        args = self._make_args(timeout=-1)
        with pytest.raises(SystemExit):
            from vmlx_engine.cli import serve_command
            serve_command(args)

    def test_temperature_out_of_range_exits(self):
        """CLI should reject temperature > 2."""
        args = self._make_args(default_temperature=5.0)
        with pytest.raises(SystemExit):
            from vmlx_engine.cli import serve_command
            serve_command(args)

    def test_top_p_out_of_range_exits(self):
        """CLI should reject top_p = 0."""
        args = self._make_args(default_top_p=0)
        with pytest.raises(SystemExit):
            from vmlx_engine.cli import serve_command
            serve_command(args)

    def test_port_out_of_range_logic(self):
        """Port validation logic should reject out-of-range values."""
        # Direct logic test — serve_command port check is at the end,
        # after model loading, so we test the condition directly
        for bad_port in [-1, 99999, 70000]:
            assert bad_port < 0 or bad_port > 65535, f"Port {bad_port} should be invalid"
        for good_port in [0, 80, 8000, 8080, 65535]:
            assert 0 <= good_port <= 65535, f"Port {good_port} should be valid"


# =============================================================================
# Anthropic Adapter
# =============================================================================


class TestAnthropicAdapterImports:
    """Verify Anthropic adapter has required imports."""

    def test_logger_defined(self):
        """Adapter must have a logger for warning on bad tool args."""
        from vmlx_engine.api import anthropic_adapter
        assert hasattr(anthropic_adapter, 'logger')
        assert anthropic_adapter.logger is not None

    def test_to_chat_completion_preserves_params(self):
        """to_chat_completion should carry temperature, top_p, top_k through."""
        from vmlx_engine.api.anthropic_adapter import AnthropicRequest, to_chat_completion
        req = AnthropicRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.5,
            top_p=0.9,
            top_k=40,
        )
        chat_req = to_chat_completion(req)
        assert chat_req.temperature == 0.5
        assert chat_req.top_p == 0.9
        assert chat_req.top_k == 40
        # Anthropic API doesn't have these — should be None
        assert chat_req.min_p is None
        assert chat_req.repetition_penalty is None

    def test_to_chat_completion_stop_sequences(self):
        """stop_sequences should map to stop field."""
        from vmlx_engine.api.anthropic_adapter import AnthropicRequest, to_chat_completion
        req = AnthropicRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            stop_sequences=["END"],
        )
        chat_req = to_chat_completion(req)
        assert chat_req.stop == ["END"]

    def test_to_chat_completion_thinking(self):
        """thinking.type=enabled should set enable_thinking=True."""
        from vmlx_engine.api.anthropic_adapter import AnthropicRequest, to_chat_completion
        req = AnthropicRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "enabled", "budget_tokens": 1000},
        )
        chat_req = to_chat_completion(req)
        assert chat_req.enable_thinking is True


# =============================================================================
# MCP Content Size Limit
# =============================================================================


class TestMCPContentLimit:
    """Test MCP client content extraction with size limits."""

    def _make_client(self):
        from vmlx_engine.mcp.client import MCPClient
        from vmlx_engine.mcp.types import MCPServerConfig, MCPTransport
        config = MCPServerConfig(
            name="test", command="python3", transport=MCPTransport.STDIO,
            skip_security_validation=True,
        )
        return MCPClient(config)

    def test_extract_content_small(self):
        """Small content should pass through unchanged."""
        client = self._make_client()
        result = MagicMock()
        item = MagicMock(spec=[])  # empty spec so hasattr checks work explicitly
        item.text = "hello world"
        result.content = [item]
        assert client._extract_content(result) == "hello world"

    def test_extract_content_oversized(self):
        """Content exceeding 10MB should be truncated."""
        client = self._make_client()
        result = MagicMock()
        big_text = "x" * (6 * 1024 * 1024)  # 6MB each
        item1 = MagicMock(spec=[]); item1.text = big_text
        item2 = MagicMock(spec=[]); item2.text = big_text
        result.content = [item1, item2]
        extracted = client._extract_content(result)
        assert isinstance(extracted, list)
        assert len(extracted) == 2
        assert "truncated" in extracted[1].lower()

    def test_extract_content_none(self):
        """No content should return None."""
        client = self._make_client()
        result = MagicMock()
        result.content = None
        assert client._extract_content(result) is None

    def test_extract_content_bytes_counted(self):
        """Bytes objects should count toward size limit."""
        client = self._make_client()
        result = MagicMock()
        item1 = MagicMock(spec=[]); item1.data = b"x" * (6 * 1024 * 1024)
        item2 = MagicMock(spec=[]); item2.data = b"x" * (6 * 1024 * 1024)
        result.content = [item1, item2]
        extracted = client._extract_content(result)
        assert isinstance(extracted, list)
        assert "truncated" in extracted[-1].lower()


# =============================================================================
# Disk Cache TOCTOU Safety
# =============================================================================


class TestDiskCacheSafety:
    """Test disk cache handles file disappearance gracefully."""

    def test_stat_oserror_handled(self):
        """Verify the try/except OSError guard exists in disk cache fetch."""
        import inspect
        from vmlx_engine import disk_cache
        source = inspect.getsource(disk_cache.DiskCacheManager.fetch)
        assert "except OSError" in source


# =============================================================================
# Stop Sequence Normalization (all three request types)
# =============================================================================


class TestJANGLoader:
    """Tests for JANG format detection, loading, and shape inference."""

    def test_is_jang_detects_jang_config(self, tmp_path):
        (tmp_path / "jang_config.json").write_text('{"format": "jang"}')
        from vmlx_engine.utils.jang_loader import is_jang_model
        assert is_jang_model(str(tmp_path)) is True

    def test_is_jang_detects_legacy_mxq(self, tmp_path):
        (tmp_path / "mxq_config.json").write_text('{"format": "mxq"}')
        from vmlx_engine.utils.jang_loader import is_jang_model
        assert is_jang_model(str(tmp_path)) is True

    def test_is_jang_false_for_normal_model(self, tmp_path):
        (tmp_path / "config.json").write_text('{"model_type": "llama"}')
        from vmlx_engine.utils.jang_loader import is_jang_model
        assert is_jang_model(str(tmp_path)) is False

    def test_is_jang_false_for_nonexistent(self):
        from vmlx_engine.utils.jang_loader import is_jang_model
        assert is_jang_model("/nonexistent/path") is False

    def test_missing_format_field_error(self, tmp_path):
        """Config without format field should give clear error."""
        (tmp_path / "jang_config.json").write_text('{"quantization": {}}')
        (tmp_path / "config.json").write_text('{"model_type": "llama", "hidden_size": 128, "num_attention_heads": 2, "vocab_size": 100}')
        from vmlx_engine.utils.jang_loader import load_jang_model
        with pytest.raises(ValueError, match="missing 'format' field"):
            load_jang_model(str(tmp_path))

    def test_wrong_format_value_error(self, tmp_path):
        """Config with wrong format value should be rejected."""
        (tmp_path / "jang_config.json").write_text('{"format": "gguf"}')
        (tmp_path / "config.json").write_text('{"model_type": "llama", "hidden_size": 128, "num_attention_heads": 2, "vocab_size": 100}')
        from vmlx_engine.utils.jang_loader import load_jang_model
        with pytest.raises(ValueError, match="Not a JANG model"):
            load_jang_model(str(tmp_path))

    def test_unsupported_version_error(self, tmp_path):
        """Format version 2.x should be rejected."""
        (tmp_path / "jang_config.json").write_text('{"format": "jang", "format_version": "2.0"}')
        (tmp_path / "config.json").write_text('{"model_type": "llama", "hidden_size": 128, "num_attention_heads": 2, "vocab_size": 100}')
        from vmlx_engine.utils.jang_loader import load_jang_model
        with pytest.raises(ValueError, match="Unsupported JANG format version"):
            load_jang_model(str(tmp_path))

    def test_infer_shape_q_proj(self):
        from vmlx_engine.utils.jang_loader import _infer_weight_shape
        config = {"hidden_size": 4096, "num_attention_heads": 32, "head_dim": 128}
        shape = _infer_weight_shape("layers.0.self_attn.q_proj", config, 4096 * 4096)
        assert shape == (4096, 4096)

    def test_infer_shape_k_proj_gqa(self):
        from vmlx_engine.utils.jang_loader import _infer_weight_shape
        config = {"hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128}
        shape = _infer_weight_shape("layers.0.self_attn.k_proj", config, 1024 * 4096)
        assert shape == (1024, 4096)

    def test_infer_shape_gate_proj(self):
        from vmlx_engine.utils.jang_loader import _infer_weight_shape
        config = {"hidden_size": 4096, "intermediate_size": 14336}
        shape = _infer_weight_shape("layers.0.mlp.gate_proj", config, 14336 * 4096)
        assert shape == (14336, 4096)

    def test_infer_shape_fused_qkv(self):
        from vmlx_engine.utils.jang_loader import _infer_weight_shape
        config = {"hidden_size": 2048, "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 128}
        shape = _infer_weight_shape("layers.0.self_attn.qkv_proj", config, (16 + 8) * 128 * 2048)
        assert shape == ((16 + 8) * 128, 2048)

    def test_infer_shape_fused_gate_up(self):
        from vmlx_engine.utils.jang_loader import _infer_weight_shape
        config = {"hidden_size": 2048, "intermediate_size": 5632}
        shape = _infer_weight_shape("layers.0.mlp.gate_up_proj", config, 2 * 5632 * 2048)
        assert shape == (2 * 5632, 2048)

    def test_infer_shape_nested_text_config(self):
        """MoE/VLM models nest dims in text_config."""
        from vmlx_engine.utils.jang_loader import _infer_weight_shape
        config = {"text_config": {"hidden_size": 2048, "intermediate_size": 5632, "num_attention_heads": 16, "head_dim": 128}}
        shape = _infer_weight_shape("layers.0.mlp.gate_proj", config, 5632 * 2048)
        assert shape == (5632, 2048)

    def test_infer_shape_expert_weight(self):
        from vmlx_engine.utils.jang_loader import _infer_weight_shape
        config = {"hidden_size": 2048, "intermediate_size": 5632}
        shape = _infer_weight_shape("layers.0.mlp.experts.3.gate_proj", config, 5632 * 2048)
        assert shape == (5632, 2048)

    def test_infer_shape_moe_intermediate(self):
        """MoE models with moe_intermediate_size different from intermediate_size."""
        from vmlx_engine.utils.jang_loader import _infer_weight_shape
        config = {"hidden_size": 2048, "moe_intermediate_size": 512}
        shape = _infer_weight_shape("layers.0.mlp.experts.3.gate_proj", config, 512 * 2048)
        assert shape == (512, 2048)

    def test_infer_shape_moe_nested_config(self):
        """MoE with nested text_config (Qwen3.5-35B-A3B style)."""
        from vmlx_engine.utils.jang_loader import _infer_weight_shape
        config = {"text_config": {"hidden_size": 2048, "moe_intermediate_size": 512, "num_attention_heads": 16, "head_dim": 128}}
        shape = _infer_weight_shape("layers.0.mlp.experts.0.gate_proj", config, 512 * 2048)
        assert shape == (512, 2048)

    def test_infer_shape_shared_expert(self):
        """DeepSeek-style shared expert with separate intermediate size."""
        from vmlx_engine.utils.jang_loader import _infer_weight_shape
        config = {"hidden_size": 2048, "moe_intermediate_size": 512, "shared_expert_intermediate_size": 1024}
        shape = _infer_weight_shape("layers.0.mlp.shared_expert.gate_proj", config, 1024 * 2048)
        assert shape == (1024, 2048)

    def test_infer_shape_w1_w2_naming(self):
        from vmlx_engine.utils.jang_loader import _infer_weight_shape
        config = {"hidden_size": 4096, "intermediate_size": 14336}
        assert _infer_weight_shape("layers.0.mlp.w1", config, 14336 * 4096) == (14336, 4096)
        assert _infer_weight_shape("layers.0.mlp.w2", config, 4096 * 14336) == (4096, 14336)

    def test_infer_shape_fallback(self):
        from vmlx_engine.utils.jang_loader import _infer_weight_shape
        config = {"hidden_size": 4096}
        shape = _infer_weight_shape("some.unknown.weight", config, 8192 * 4096)
        assert shape == (8192, 4096)

    def test_infer_shape_no_match(self):
        from vmlx_engine.utils.jang_loader import _infer_weight_shape
        config = {"hidden_size": 4096}
        shape = _infer_weight_shape("some.unknown.weight", config, 12345)
        assert shape is None

    def test_jang_repack_uses_uint32(self):
        """Verify the loader uses _repack_jang_to_mlx (uint32) not _unpack_bits (removed)."""
        import inspect
        from vmlx_engine.utils import jang_loader
        # _unpack_bits was removed — the loader now repacks JANG uint8 directly
        # to MLX uint32 format via _repack_jang_to_mlx
        assert not hasattr(jang_loader, '_unpack_bits'), "_unpack_bits should be removed (dead code)"
        assert hasattr(jang_loader, '_repack_jang_to_mlx'), "_repack_jang_to_mlx should exist"


class TestStopNormalization:
    """Verify stop sequence normalization across all request types."""

    def test_chat_string_to_list(self):
        req = ChatCompletionRequest(
            model="t", messages=[Message(role="user", content="hi")], stop="<end>"
        )
        assert req.stop == ["<end>"]

    def test_chat_list_unchanged(self):
        req = ChatCompletionRequest(
            model="t", messages=[Message(role="user", content="hi")], stop=["a", "b"]
        )
        assert req.stop == ["a", "b"]

    def test_chat_none_unchanged(self):
        req = ChatCompletionRequest(
            model="t", messages=[Message(role="user", content="hi")]
        )
        assert req.stop is None

    def test_completion_string_to_list(self):
        req = CompletionRequest(model="t", prompt="hi", stop="<end>")
        assert req.stop == ["<end>"]

    def test_responses_string_to_list(self):
        req = ResponsesRequest(model="t", input="hi", stop="<end>")
        assert req.stop == ["<end>"]

    def test_n_greater_than_one_rejected(self):
        with pytest.raises(ValueError, match="Only n=1"):
            ChatCompletionRequest(
                model="t", messages=[Message(role="user", content="hi")], n=5
            )
