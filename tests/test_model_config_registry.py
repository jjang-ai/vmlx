# SPDX-License-Identifier: Apache-2.0
"""Tests for model config registry and model configs."""

import pytest

from vllm_mlx.model_config_registry import (
    ModelConfig,
    ModelConfigRegistry,
    get_model_config_registry,
)


@pytest.fixture
def empty_registry():
    """Create a fresh empty registry for testing."""
    # Reset singleton
    ModelConfigRegistry._instance = None
    registry = ModelConfigRegistry()
    return registry


@pytest.fixture(autouse=True)
def cleanup_singleton():
    """Reset singleton after each test."""
    yield
    ModelConfigRegistry._instance = None


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self):
        config = ModelConfig(family_name="test", pattern=r"test")
        assert config.cache_type == "kv"
        assert config.eos_tokens is None
        assert config.tool_parser is None
        assert config.is_mllm is False
        assert config.tokenizer_fallback is False
        assert config.priority == 100

    def test_custom_values(self):
        config = ModelConfig(
            family_name="qwen3",
            pattern=r"(?i)qwen3",
            cache_type="kv",
            eos_tokens=["<|im_end|>"],
            tool_parser="qwen",
            supports_native_tools=True,
            priority=10,
        )
        assert config.family_name == "qwen3"
        assert config.eos_tokens == ["<|im_end|>"]
        assert config.tool_parser == "qwen"
        assert config.supports_native_tools is True

    def test_architecture_hints(self):
        config = ModelConfig(
            family_name="gemma3",
            pattern=r"gemma3",
            architecture_hints={"inject_pixel_values": True},
        )
        assert config.architecture_hints["inject_pixel_values"] is True


class TestModelConfigRegistry:
    """Tests for ModelConfigRegistry."""

    def test_register_and_lookup(self, empty_registry):
        config = ModelConfig(
            family_name="test_model",
            pattern=r"(?i)test[\-_]model",
            cache_type="kv",
            priority=10,
        )
        empty_registry.register(config)
        result = empty_registry.lookup("my-test_model-8B")
        assert result.family_name == "test_model"

    def test_lookup_case_insensitive(self, empty_registry):
        config = ModelConfig(
            family_name="qwen3",
            pattern=r"(?i)qwen3",
            priority=10,
        )
        empty_registry.register(config)
        assert empty_registry.lookup("Qwen3-8B").family_name == "qwen3"
        assert empty_registry.lookup("qwen3-instruct").family_name == "qwen3"
        assert empty_registry.lookup("QWEN3").family_name == "qwen3"

    def test_lookup_no_match_returns_default(self, empty_registry):
        config = ModelConfig(
            family_name="qwen3",
            pattern=r"(?i)qwen3",
            priority=10,
        )
        empty_registry.register(config)
        result = empty_registry.lookup("totally-unknown-model")
        assert result.family_name == "unknown"

    def test_priority_ordering(self, empty_registry):
        """Higher priority (lower number) config should match first."""
        generic = ModelConfig(
            family_name="qwen",
            pattern=r"(?i)qwen",
            cache_type="kv",
            priority=50,
        )
        specific = ModelConfig(
            family_name="qwen3-vl",
            pattern=r"(?i)qwen3.*vl",
            cache_type="kv",
            is_mllm=True,
            priority=5,
        )
        empty_registry.register(generic)
        empty_registry.register(specific)

        # Specific match should win due to lower priority number
        result = empty_registry.lookup("mlx-community/Qwen3-VL-8B")
        assert result.family_name == "qwen3-vl"
        assert result.is_mllm is True

        # Generic should still match non-VL
        result = empty_registry.lookup("mlx-community/Qwen2-7B")
        assert result.family_name == "qwen"

    def test_cache_type_lookup(self, empty_registry):
        config = ModelConfig(
            family_name="mamba",
            pattern=r"(?i)mamba",
            cache_type="mamba",
            priority=10,
        )
        empty_registry.register(config)
        assert empty_registry.get_cache_type("mamba-7B") == "mamba"

    def test_eos_tokens_lookup(self, empty_registry):
        config = ModelConfig(
            family_name="qwen3",
            pattern=r"(?i)qwen3",
            eos_tokens=["<|im_end|>"],
            priority=10,
        )
        empty_registry.register(config)
        assert empty_registry.get_eos_tokens("Qwen3-8B") == ["<|im_end|>"]
        assert empty_registry.get_eos_tokens("unknown-model") is None

    def test_is_mllm(self, empty_registry):
        text_config = ModelConfig(
            family_name="qwen3",
            pattern=r"(?i)qwen3",
            is_mllm=False,
            priority=10,
        )
        vl_config = ModelConfig(
            family_name="qwen3-vl",
            pattern=r"(?i)qwen3.*vl",
            is_mllm=True,
            priority=5,
        )
        empty_registry.register(text_config)
        empty_registry.register(vl_config)

        assert empty_registry.is_mllm("Qwen3-VL-8B") is True
        assert empty_registry.is_mllm("Qwen3-8B") is False

    def test_tool_parser(self, empty_registry):
        config = ModelConfig(
            family_name="mistral",
            pattern=r"(?i)mistral",
            tool_parser="mistral",
            priority=10,
        )
        empty_registry.register(config)
        assert empty_registry.get_tool_parser("Mistral-7B") == "mistral"

    def test_architecture_hints(self, empty_registry):
        config = ModelConfig(
            family_name="gemma3",
            pattern=r"(?i)gemma3",
            architecture_hints={"inject_pixel_values": True},
            priority=10,
        )
        empty_registry.register(config)
        hints = empty_registry.get_architecture_hints("gemma3-2B")
        assert hints == {"inject_pixel_values": True}

    def test_needs_tokenizer_fallback(self, empty_registry):
        config = ModelConfig(
            family_name="nemotron",
            pattern=r"(?i)nemotron",
            tokenizer_fallback=True,
            priority=10,
        )
        empty_registry.register(config)
        assert empty_registry.needs_tokenizer_fallback("nemotron-8B") is True
        assert empty_registry.needs_tokenizer_fallback("llama-8B") is False

    def test_invalid_regex_pattern(self, empty_registry):
        config = ModelConfig(
            family_name="bad",
            pattern=r"[invalid(",
            priority=10,
        )
        with pytest.raises(ValueError, match="Invalid regex"):
            empty_registry.register(config)

    def test_list_registered(self, empty_registry):
        empty_registry.register(
            ModelConfig(family_name="a", pattern="a", priority=10)
        )
        empty_registry.register(
            ModelConfig(family_name="b", pattern="b", priority=5)
        )
        names = empty_registry.list_registered()
        assert "a" in names
        assert "b" in names
        # Should be sorted by priority
        assert names.index("b") < names.index("a")

    def test_clear_cache(self, empty_registry):
        config = ModelConfig(
            family_name="test",
            pattern=r"test",
            priority=10,
        )
        empty_registry.register(config)
        # Trigger caching
        empty_registry.lookup("test-model")
        assert len(empty_registry._match_cache) > 0
        empty_registry.clear_cache()
        assert len(empty_registry._match_cache) == 0

    def test_clear_all(self, empty_registry):
        config = ModelConfig(
            family_name="test",
            pattern=r"test",
            priority=10,
        )
        empty_registry.register(config)
        empty_registry.clear()
        assert len(empty_registry._configs) == 0


class TestModelConfigs:
    """Tests for the pre-registered model configurations."""

    @pytest.fixture
    def registry(self):
        """Get registry with all model configs loaded."""
        # Reset singleton so configs get loaded fresh
        ModelConfigRegistry._instance = None
        import vllm_mlx.model_config_registry as mcr
        mcr._configs_loaded = False
        return get_model_config_registry()

    # Qwen family
    def test_qwen3_config(self, registry):
        config = registry.lookup("mlx-community/Qwen3-8B-Instruct-4bit")
        assert config.family_name == "qwen3"
        assert config.eos_tokens == ["<|im_end|>"]
        assert config.tool_parser == "qwen"

    def test_qwen3_vl_config(self, registry):
        config = registry.lookup("mlx-community/Qwen3-VL-7B-Instruct")
        assert config.family_name == "qwen3-vl"
        assert config.is_mllm is True

    def test_qwen_mamba_config(self, registry):
        config = registry.lookup("Qwen-Mamba-7B")
        assert config.cache_type == "mamba"

    # Llama family
    def test_llama3_config(self, registry):
        config = registry.lookup("mlx-community/Llama-3.2-3B-Instruct-4bit")
        assert config.family_name == "llama3"
        assert config.tool_parser == "llama"
        assert config.supports_native_tools is True

    def test_llama4_config(self, registry):
        config = registry.lookup("meta-llama/Llama-4-Scout-17B")
        assert config.family_name == "llama4"

    # Mistral family
    def test_mistral_config(self, registry):
        config = registry.lookup("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
        assert config.tool_parser == "mistral"
        assert config.supports_native_tools is True

    def test_mixtral_config(self, registry):
        config = registry.lookup("mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit")
        assert config.family_name == "mixtral"

    def test_pixtral_config(self, registry):
        config = registry.lookup("mlx-community/Pixtral-12B-2409-4bit")
        assert config.is_mllm is True

    # DeepSeek family
    def test_deepseek_r1_config(self, registry):
        config = registry.lookup("mlx-community/DeepSeek-R1-Distill-7B-4bit")
        assert config.tool_parser == "deepseek"

    def test_deepseek_vl_config(self, registry):
        config = registry.lookup("deepseek-ai/DeepSeek-VL2-small")
        assert config.is_mllm is True

    # Gemma family
    def test_gemma3_config(self, registry):
        config = registry.lookup("mlx-community/gemma-3-2b-it-4bit")
        assert config.is_mllm is True
        assert config.architecture_hints.get("inject_pixel_values") is True

    def test_medgemma_config(self, registry):
        config = registry.lookup("google/medgemma-4b-it")
        assert config.is_mllm is True
        assert config.architecture_hints.get("inject_pixel_values") is True

    # Nemotron
    def test_nemotron_config(self, registry):
        config = registry.lookup("nvidia/Nemotron-4-340B-Instruct")
        assert config.cache_type == "hybrid"
        assert config.tokenizer_fallback is True

    # Mamba
    def test_pure_mamba_config(self, registry):
        config = registry.lookup("state-spaces/mamba-2.8b")
        assert config.cache_type == "mamba"

    def test_falcon_mamba_config(self, registry):
        config = registry.lookup("tiiuae/falcon-mamba-7b")
        assert config.cache_type == "mamba"

    # Hybrid
    def test_jamba_config(self, registry):
        config = registry.lookup("ai21labs/Jamba-v0.1")
        assert config.cache_type == "hybrid"

    # MLLM models
    def test_llava_config(self, registry):
        config = registry.lookup("llava-hf/llava-1.5-7b-hf")
        assert config.is_mllm is True

    def test_internvl_config(self, registry):
        config = registry.lookup("OpenGVLab/InternVL2-8B")
        assert config.is_mllm is True

    # Tool-calling
    def test_granite_config(self, registry):
        config = registry.lookup("ibm-granite/granite-3.1-8b-instruct")
        assert config.tool_parser == "granite"

    def test_hermes_config(self, registry):
        # Hermes-Llama matches llama3 (priority 10) before hermes (priority 30)
        config = registry.lookup("NousResearch/Hermes-3-Llama-3.1-8B")
        assert config.family_name == "llama3"
        # Pure Hermes name should match hermes
        config2 = registry.lookup("NousResearch/Hermes-3-8B")
        assert config2.tool_parser == "hermes"

    # Unknown model
    def test_unknown_model(self, registry):
        config = registry.lookup("completely-unknown-model-xyz")
        assert config.family_name == "unknown"
        assert config.cache_type == "kv"
