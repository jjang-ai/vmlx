# SPDX-License-Identifier: Apache-2.0
"""Tests for model config registry and model configs.

The registry uses model_types (from config.json model_type field) for lookup.
Since unit tests cannot read from disk/HuggingFace, we mock mlx_lm.utils.load_config
to return the expected model_type for each test.
"""

from unittest.mock import patch

import pytest

from vmlx_engine.model_config_registry import (
    ModelConfig,
    ModelConfigRegistry,
    get_model_config_registry,
)


@pytest.fixture
def empty_registry():
    """Create a fresh empty registry for testing."""
    ModelConfigRegistry._instance = None
    registry = ModelConfigRegistry()
    return registry


@pytest.fixture(autouse=True)
def cleanup_singleton():
    """Reset singleton after each test."""
    yield
    ModelConfigRegistry._instance = None


def _mock_load_config(model_type: str):
    """Return a mock for mlx_lm.utils.load_config that returns the given model_type."""
    def _load_config(path):
        return {"model_type": model_type}
    return _load_config


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self):
        config = ModelConfig(family_name="test", model_types=["test"])
        assert config.cache_type == "kv"
        assert config.eos_tokens is None
        assert config.tool_parser is None
        assert config.is_mllm is False
        assert config.tokenizer_fallback is False
        assert config.priority == 100

    def test_custom_values(self):
        config = ModelConfig(
            family_name="qwen3",
            model_types=["qwen3"],
            cache_type="kv",
            eos_tokens=["<|im_end|>"],
            tool_parser="qwen",
            supports_native_tools=True,
            priority=10,
        )
        assert config.family_name == "qwen3"
        assert config.model_types == ["qwen3"]
        assert config.eos_tokens == ["<|im_end|>"]
        assert config.tool_parser == "qwen"
        assert config.supports_native_tools is True

    def test_architecture_hints(self):
        config = ModelConfig(
            family_name="gemma3",
            model_types=["gemma3"],
            architecture_hints={"inject_pixel_values": True},
        )
        assert config.architecture_hints["inject_pixel_values"] is True


class TestModelConfigRegistry:
    """Tests for ModelConfigRegistry."""

    def test_register_and_lookup(self, empty_registry):
        config = ModelConfig(
            family_name="test_model",
            model_types=["test_model_type"],
            cache_type="kv",
            priority=10,
        )
        empty_registry.register(config)
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("test_model_type")):
            result = empty_registry.lookup("my-test_model-8B")
        assert result.family_name == "test_model"

    def test_lookup_model_type_matching(self, empty_registry):
        config = ModelConfig(
            family_name="qwen3",
            model_types=["qwen3"],
            priority=10,
        )
        empty_registry.register(config)
        # All lookups that return model_type "qwen3" should match
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("qwen3")):
            assert empty_registry.lookup("Qwen3-8B").family_name == "qwen3"
            empty_registry.clear_cache()
            assert empty_registry.lookup("qwen3-instruct").family_name == "qwen3"

    def test_lookup_no_match_returns_default(self, empty_registry):
        config = ModelConfig(
            family_name="qwen3",
            model_types=["qwen3"],
            priority=10,
        )
        empty_registry.register(config)
        # Model type that doesn't match any registered config
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("unknown_type")):
            result = empty_registry.lookup("totally-unknown-model")
        assert result.family_name == "unknown"

    def test_priority_ordering(self, empty_registry):
        """Higher priority (lower number) config should match first when multiple match."""
        generic = ModelConfig(
            family_name="qwen",
            model_types=["qwen", "qwen2"],
            cache_type="kv",
            priority=50,
        )
        specific = ModelConfig(
            family_name="qwen3_vl",
            model_types=["qwen3_vl"],
            cache_type="kv",
            is_mllm=True,
            priority=5,
        )
        empty_registry.register(generic)
        empty_registry.register(specific)

        # qwen3_vl model_type matches the specific config
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("qwen3_vl")):
            result = empty_registry.lookup("mlx-community/Qwen3-VL-8B")
        assert result.family_name == "qwen3_vl"
        assert result.is_mllm is True

        # qwen2 model_type matches only the generic config
        empty_registry.clear_cache()
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("qwen2")):
            result = empty_registry.lookup("mlx-community/Qwen2-7B")
        assert result.family_name == "qwen"

    def test_cache_type_lookup(self, empty_registry):
        config = ModelConfig(
            family_name="mamba",
            model_types=["mamba", "mamba2"],
            cache_type="mamba",
            priority=10,
        )
        empty_registry.register(config)
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("mamba")):
            assert empty_registry.get_cache_type("mamba-7B") == "mamba"

    def test_eos_tokens_lookup(self, empty_registry):
        config = ModelConfig(
            family_name="qwen3",
            model_types=["qwen3"],
            eos_tokens=["<|im_end|>"],
            priority=10,
        )
        empty_registry.register(config)
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("qwen3")):
            assert empty_registry.get_eos_tokens("Qwen3-8B") == ["<|im_end|>"]
        empty_registry.clear_cache()
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("unknown")):
            assert empty_registry.get_eos_tokens("unknown-model") is None

    def test_is_mllm(self, empty_registry):
        text_config = ModelConfig(
            family_name="qwen3",
            model_types=["qwen3"],
            is_mllm=False,
            priority=10,
        )
        vl_config = ModelConfig(
            family_name="qwen3_vl",
            model_types=["qwen3_vl"],
            is_mllm=True,
            priority=5,
        )
        empty_registry.register(text_config)
        empty_registry.register(vl_config)

        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("qwen3_vl")):
            assert empty_registry.is_mllm("Qwen3-VL-8B") is True
        empty_registry.clear_cache()
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("qwen3")):
            assert empty_registry.is_mllm("Qwen3-8B") is False

    def test_tool_parser(self, empty_registry):
        config = ModelConfig(
            family_name="mistral",
            model_types=["mistral"],
            tool_parser="mistral",
            priority=10,
        )
        empty_registry.register(config)
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("mistral")):
            assert empty_registry.get_tool_parser("Mistral-7B") == "mistral"

    def test_architecture_hints(self, empty_registry):
        config = ModelConfig(
            family_name="gemma3",
            model_types=["gemma3"],
            architecture_hints={"inject_pixel_values": True},
            priority=10,
        )
        empty_registry.register(config)
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("gemma3")):
            hints = empty_registry.get_architecture_hints("gemma3-2B")
        assert hints == {"inject_pixel_values": True}

    def test_needs_tokenizer_fallback(self, empty_registry):
        config = ModelConfig(
            family_name="nemotron",
            model_types=["nemotron"],
            tokenizer_fallback=True,
            priority=10,
        )
        empty_registry.register(config)
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("nemotron")):
            assert empty_registry.needs_tokenizer_fallback("nemotron-8B") is True
        empty_registry.clear_cache()
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("unknown")):
            assert empty_registry.needs_tokenizer_fallback("llama-8B") is False

    def test_list_registered(self, empty_registry):
        empty_registry.register(
            ModelConfig(family_name="a", model_types=["a_type"], priority=10)
        )
        empty_registry.register(
            ModelConfig(family_name="b", model_types=["b_type"], priority=5)
        )
        names = empty_registry.list_registered()
        assert "a" in names
        assert "b" in names
        # Should be sorted by priority
        assert names.index("b") < names.index("a")

    def test_clear_cache(self, empty_registry):
        config = ModelConfig(
            family_name="test",
            model_types=["test_type"],
            priority=10,
        )
        empty_registry.register(config)
        # Trigger caching
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("test_type")):
            empty_registry.lookup("test-model")
        assert len(empty_registry._match_cache) > 0
        empty_registry.clear_cache()
        assert len(empty_registry._match_cache) == 0

    def test_clear_all(self, empty_registry):
        config = ModelConfig(
            family_name="test",
            model_types=["test_type"],
            priority=10,
        )
        empty_registry.register(config)
        empty_registry.clear()
        assert len(empty_registry._configs) == 0


class TestModelConfigs:
    """Tests for the pre-registered model configurations.

    These tests verify that model_configs.py registers the correct configurations
    for each model family. Since lookup() reads config.json via mlx_lm.utils.load_config,
    we mock it to return the expected model_type for each model.
    """

    @pytest.fixture
    def registry(self):
        """Get registry with all model configs loaded."""
        ModelConfigRegistry._instance = None
        import vmlx_engine.model_config_registry as mcr
        mcr._configs_loaded = False
        return get_model_config_registry()

    def _lookup(self, registry, model_name, model_type):
        """Helper: lookup with mocked load_config returning given model_type."""
        registry.clear_cache()
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config(model_type)):
            return registry.lookup(model_name)

    # Qwen family
    def test_qwen3_config(self, registry):
        config = self._lookup(registry, "mlx-community/Qwen3-8B-Instruct-4bit", "qwen3")
        assert config.family_name == "qwen3"
        assert config.eos_tokens == ["<|im_end|>"]
        assert config.tool_parser == "qwen"

    def test_qwen3_vl_config(self, registry):
        config = self._lookup(registry, "mlx-community/Qwen3-VL-7B-Instruct", "qwen3_vl")
        assert config.family_name == "qwen3_vl"
        assert config.is_mllm is True

    def test_qwen_mamba_config(self, registry):
        config = self._lookup(registry, "Qwen-Mamba-7B", "qwen_mamba")
        assert config.cache_type == "mamba"

    # Llama family
    def test_llama_config(self, registry):
        config = self._lookup(registry, "mlx-community/Llama-3.2-3B-Instruct-4bit", "llama")
        assert config.family_name == "llama"
        assert config.tool_parser == "llama"
        assert config.supports_native_tools is True

    def test_llama4_config(self, registry):
        config = self._lookup(registry, "meta-llama/Llama-4-Scout-17B", "llama4")
        assert config.family_name == "llama4"

    # Mistral family
    def test_mistral_config(self, registry):
        config = self._lookup(registry, "mlx-community/Mistral-7B-Instruct-v0.3-4bit", "mistral")
        assert config.tool_parser == "mistral"
        assert config.supports_native_tools is True

    def test_mixtral_config(self, registry):
        # Mixtral model_type maps to mistral family (which includes "mixtral" in model_types)
        config = self._lookup(registry, "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit", "mixtral")
        assert config.family_name == "mistral"
        assert config.tool_parser == "mistral"

    def test_pixtral_config(self, registry):
        config = self._lookup(registry, "mlx-community/Pixtral-12B-2409-4bit", "pixtral")
        assert config.is_mllm is True

    # DeepSeek family
    def test_deepseek_r1_config(self, registry):
        config = self._lookup(registry, "mlx-community/DeepSeek-R1-Distill-7B-4bit", "deepseek_v2")
        assert config.tool_parser == "deepseek"

    def test_deepseek_vl_config(self, registry):
        config = self._lookup(registry, "deepseek-ai/DeepSeek-VL2-small", "deepseek_vl2")
        assert config.is_mllm is True

    # Gemma family
    def test_gemma3_config(self, registry):
        config = self._lookup(registry, "mlx-community/gemma-3-2b-it-4bit", "gemma3")
        assert config.is_mllm is True
        assert config.tool_parser == "hermes"

    def test_gemma3_text_config(self, registry):
        config = self._lookup(registry, "google/gemma-3-text-1b", "gemma3_text")
        assert config.family_name == "gemma3_text"
        assert config.is_mllm is False

    # Nemotron
    def test_nemotron_config(self, registry):
        config = self._lookup(registry, "nvidia/Nemotron-4-340B-Instruct", "nemotron")
        assert config.cache_type == "hybrid"
        assert config.tokenizer_fallback is True

    # Mamba
    def test_pure_mamba_config(self, registry):
        config = self._lookup(registry, "state-spaces/mamba-2.8b", "mamba")
        assert config.cache_type == "mamba"

    def test_falcon_mamba_config(self, registry):
        config = self._lookup(registry, "tiiuae/falcon-mamba-7b", "falcon_mamba")
        assert config.cache_type == "mamba"

    # Hybrid
    def test_jamba_config(self, registry):
        config = self._lookup(registry, "ai21labs/Jamba-v0.1", "jamba")
        assert config.cache_type == "hybrid"

    # MLLM models
    def test_llava_config(self, registry):
        config = self._lookup(registry, "llava-hf/llava-1.5-7b-hf", "llava")
        assert config.is_mllm is True

    def test_internvl_config(self, registry):
        config = self._lookup(registry, "OpenGVLab/InternVL2-8B", "internvl_chat")
        assert config.is_mllm is True

    # Tool-calling
    def test_granite_config(self, registry):
        config = self._lookup(registry, "ibm-granite/granite-3.1-8b-instruct", "granite")
        assert config.tool_parser == "granite"

    def test_hermes_config(self, registry):
        config = self._lookup(registry, "NousResearch/Hermes-3-8B", "hermes")
        assert config.tool_parser == "hermes"

    # Unknown model
    def test_unknown_model(self, registry):
        config = self._lookup(registry, "completely-unknown-model-xyz", "totally_unknown_type")
        assert config.family_name == "unknown"
        assert config.cache_type == "kv"

    # GLM family
    def test_glm4_moe_config(self, registry):
        config = self._lookup(registry, "THUDM/GLM-4-Flash", "glm4_moe")
        assert config.family_name == "glm4_moe"
        assert config.reasoning_parser == "openai_gptoss"
        assert config.tool_parser == "glm47"

    def test_gpt_oss_config(self, registry):
        config = self._lookup(registry, "THUDM/GPT-OSS", "gpt_oss")
        assert config.family_name == "gpt_oss"
        assert config.reasoning_parser == "openai_gptoss"

    # Phi family
    def test_phi4_config(self, registry):
        config = self._lookup(registry, "microsoft/phi-4", "phi4")
        assert config.family_name == "phi4"
        assert config.tool_parser == "hermes"

    def test_phi4_reasoning_config(self, registry):
        config = self._lookup(registry, "microsoft/phi-4-reasoning", "phi4_reasoning")
        assert config.reasoning_parser == "deepseek_r1"

    # Step family
    def test_step_config(self, registry):
        config = self._lookup(registry, "stepfun/Step3p5", "step3p5")
        assert config.family_name == "step"
        assert config.tool_parser == "step3p5"
        assert config.reasoning_parser == "qwen3"

    def test_step_vl_config(self, registry):
        config = self._lookup(registry, "stepfun/Step-1V-8B", "step1v")
        assert config.family_name == "step_vl"
        assert config.is_mllm is True
        assert config.tool_parser == "step3p5"
        assert config.reasoning_parser == "qwen3"
        assert config.think_in_template is True

    # ── Additional Qwen family coverage ──

    def test_qwen3_5_config(self, registry):
        config = self._lookup(registry, "Qwen/Qwen3.5-VL-9B", "qwen3_5")
        assert config.family_name == "qwen3_5"
        assert config.is_mllm is True
        assert config.tool_parser == "qwen"
        assert config.reasoning_parser == "qwen3"
        assert config.think_in_template is True
        assert config.eos_tokens == ["<|im_end|>"]

    def test_qwen3_5_moe_config(self, registry):
        config = self._lookup(registry, "Qwen/Qwen3.5-MoE-VL-38B", "qwen3_5_moe")
        assert config.family_name == "qwen3_5_moe"
        assert config.is_mllm is True
        assert config.reasoning_parser == "qwen3"

    def test_qwen3_moe_config(self, registry):
        config = self._lookup(registry, "Qwen/Qwen3-MoE-15B", "qwen3_moe")
        assert config.family_name == "qwen3_moe"
        assert config.tool_parser == "qwen"
        assert config.reasoning_parser == "qwen3"
        assert config.think_in_template is True
        assert config.supports_native_tools is True

    def test_qwen3_next_config(self, registry):
        config = self._lookup(registry, "Qwen/Qwen3-Next-7B", "qwen3_next")
        assert config.family_name == "qwen3_next"
        assert config.cache_type == "mamba"
        assert config.tool_parser == "nemotron"

    def test_qwen2_config(self, registry):
        config = self._lookup(registry, "Qwen/Qwen2.5-7B-Instruct", "qwen2")
        assert config.family_name == "qwen2"
        assert config.tool_parser == "qwen"
        assert config.reasoning_parser == "qwen3"
        assert config.supports_native_tools is True

    def test_qwen2_moe_config(self, registry):
        config = self._lookup(registry, "Qwen/Qwen2-MoE-57B", "qwen2_moe")
        assert config.family_name == "qwen2"
        assert config.tool_parser == "qwen"

    def test_qwen2_vl_config(self, registry):
        config = self._lookup(registry, "Qwen/Qwen2-VL-7B-Instruct", "qwen2_vl")
        assert config.family_name == "qwen2_vl"
        assert config.is_mllm is True
        assert config.tool_parser == "qwen"
        assert config.reasoning_parser == "qwen3"

    def test_qwen2_5_vl_config(self, registry):
        config = self._lookup(registry, "Qwen/Qwen2.5-VL-7B-Instruct", "qwen2_5_vl")
        assert config.family_name == "qwen2_vl"
        assert config.is_mllm is True

    def test_qwen_mamba_config(self, registry):
        config = self._lookup(registry, "Qwen/Qwen-Mamba-7B", "qwen_mamba")
        assert config.family_name == "qwen_mamba"
        assert config.cache_type == "mamba"
        assert config.tool_parser == "qwen"

    # ── Additional GLM family coverage ──

    def test_glm4_moe_lite_config(self, registry):
        """GLM-4.7 Flash (MoE Lite variant) uses openai_gptoss, NOT deepseek_r1."""
        config = self._lookup(registry, "THUDM/GLM-4-Flash-Lite", "glm4_moe_lite")
        assert config.family_name == "glm4_moe"
        assert config.reasoning_parser == "openai_gptoss"
        assert config.tool_parser == "glm47"
        assert config.chat_template_custom is not None

    def test_chatglm_config(self, registry):
        """ChatGLM/GLM-4 base: tools only, NO reasoning parser."""
        config = self._lookup(registry, "THUDM/ChatGLM3-6B", "chatglm")
        assert config.family_name == "chatglm"
        assert config.tool_parser == "glm47"
        assert config.reasoning_parser is None

    def test_glm4_base_config(self, registry):
        """GLM-4 base (model_type glm4) falls to chatglm family."""
        config = self._lookup(registry, "THUDM/GLM-4-9B", "glm4")
        assert config.family_name == "chatglm"
        assert config.reasoning_parser is None

    def test_gpt_oss_has_harmony_template(self, registry):
        config = self._lookup(registry, "THUDM/GPT-OSS", "gpt_oss")
        assert config.chat_template_custom is not None
        assert "<|start|>" in config.chat_template_custom

    # ── GLM parser separation is CRITICAL ──
    def test_glm_flash_vs_base_reasoning_parser_differs(self, registry):
        """GLM Flash uses openai_gptoss, GLM-4 base has no reasoning parser.
        These MUST be different — merging them would break one or the other."""
        flash = self._lookup(registry, "THUDM/GLM-4-Flash", "glm4_moe")
        base = self._lookup(registry, "THUDM/GLM-4-9B", "glm4")
        assert flash.reasoning_parser == "openai_gptoss"
        assert base.reasoning_parser is None
        assert flash.family_name != base.family_name

    # ── Additional Mistral family coverage ──

    def test_devstral_config(self, registry):
        config = self._lookup(registry, "mistralai/Devstral-Small-8B", "devstral")
        assert config.tool_parser == "mistral"
        assert config.supports_native_tools is True
        assert config.preserve_native_tool_format is True

    def test_codestral_config(self, registry):
        config = self._lookup(registry, "mistralai/Codestral-v0.1", "codestral")
        assert config.tool_parser == "mistral"

    def test_pixtral_is_mllm(self, registry):
        config = self._lookup(registry, "mistralai/Pixtral-12B", "pixtral")
        assert config.is_mllm is True
        assert config.tool_parser == "mistral"

    # ── Additional DeepSeek coverage ──

    def test_deepseek_v3_config(self, registry):
        config = self._lookup(registry, "deepseek-ai/DeepSeek-V3", "deepseek_v3")
        assert config.family_name == "deepseek"
        assert config.reasoning_parser == "deepseek_r1"
        assert config.tool_parser == "deepseek"

    def test_deepseek_vl2_config(self, registry):
        config = self._lookup(registry, "deepseek-ai/DeepSeek-VL2", "deepseek_vl2")
        assert config.family_name == "deepseek_vl"
        assert config.is_mllm is True

    # ── Gemma family coverage ──

    def test_gemma3_reasoning_parser(self, registry):
        config = self._lookup(registry, "google/gemma-3-2b-it", "gemma3")
        assert config.reasoning_parser == "deepseek_r1"

    def test_gemma3_text_not_mllm(self, registry):
        config = self._lookup(registry, "google/gemma-3-text-1b", "gemma3_text")
        assert config.is_mllm is False
        assert config.reasoning_parser == "deepseek_r1"

    def test_paligemma_config(self, registry):
        config = self._lookup(registry, "google/paligemma-3b", "paligemma")
        assert config.is_mllm is True

    # ── Phi family coverage ──

    def test_phi4_multimodal_config(self, registry):
        config = self._lookup(registry, "microsoft/phi-4-multimodal", "phi4mm")
        assert config.is_mllm is True

    def test_phi3_v_config(self, registry):
        config = self._lookup(registry, "microsoft/Phi-3-Vision", "phi3v")
        assert config.is_mllm is True
        assert config.tool_parser == "llama"

    def test_phi3_config(self, registry):
        config = self._lookup(registry, "microsoft/phi-3-mini", "phi3")
        assert config.tool_parser == "llama"
        assert config.is_mllm is False

    # ── Nemotron coverage ──

    def test_nemotron_hybrid_cache(self, registry):
        config = self._lookup(registry, "nvidia/Nemotron-H-47B", "nemotron_h")
        assert config.cache_type == "hybrid"
        assert config.tokenizer_fallback is True
        assert config.reasoning_parser == "deepseek_r1"

    # ── MiniMax coverage ──

    def test_minimax_config(self, registry):
        config = self._lookup(registry, "MiniMaxAI/Prism-Pro", "minimax")
        assert config.tool_parser == "minimax"
        assert config.reasoning_parser == "qwen3"
        assert config.think_in_template is True

    # ── Kimi coverage ──

    def test_kimi_config(self, registry):
        config = self._lookup(registry, "moonshotai/Kimi-K2", "kimi_k2")
        assert config.tool_parser == "kimi"

    # ── SSM/Mamba coverage ──

    def test_rwkv_config(self, registry):
        config = self._lookup(registry, "BlinkDL/RWKV-6", "rwkv6")
        assert config.cache_type == "mamba"

    def test_codestral_mamba_config(self, registry):
        config = self._lookup(registry, "mistralai/Codestral-Mamba", "codestral_mamba")
        assert config.cache_type == "mamba"

    # ── VLM models coverage ──

    def test_idefics_config(self, registry):
        config = self._lookup(registry, "HuggingFaceM4/Idefics3-8B", "idefics3")
        assert config.is_mllm is True

    def test_cogvlm_config(self, registry):
        config = self._lookup(registry, "THUDM/CogVLM2-19B", "cogvlm2")
        assert config.is_mllm is True

    def test_florence_config(self, registry):
        config = self._lookup(registry, "microsoft/Florence-2-base", "florence2")
        assert config.is_mllm is True

    def test_molmo_config(self, registry):
        config = self._lookup(registry, "allenai/Molmo-7B", "molmo")
        assert config.is_mllm is True

    def test_minicpm_v_config(self, registry):
        config = self._lookup(registry, "openbmb/MiniCPM-V-2_6", "minicpmv")
        assert config.is_mllm is True

    def test_smolvlm_config(self, registry):
        config = self._lookup(registry, "HuggingFaceTB/SmolVLM-Instruct", "smolvlm")
        assert config.is_mllm is True

    def test_internvl_config_full(self, registry):
        config = self._lookup(registry, "OpenGVLab/InternVL2-8B", "internvl_chat")
        assert config.is_mllm is True

    def test_internlm_xcomposer_config(self, registry):
        config = self._lookup(registry, "internlm/InternLM-XComposer2-VL", "internlm_xcomposer2")
        assert config.is_mllm is True


class TestModelConfigComprehensiveChecks:
    """Comprehensive checks that span all model families.

    These tests ensure consistency across the entire registry:
    - Every MLLM model has is_mllm=True
    - Reasoning parsers are valid known values
    - Tool parsers are valid known values
    - Cache types are valid
    - Priority ordering makes sense (specific models beat generic)
    """

    @pytest.fixture
    def registry(self):
        ModelConfigRegistry._instance = None
        import vmlx_engine.model_config_registry as mcr
        mcr._configs_loaded = False
        return get_model_config_registry()

    VALID_REASONING_PARSERS = {None, "qwen3", "deepseek_r1", "openai_gptoss"}
    VALID_TOOL_PARSERS = {
        None, "qwen", "llama", "mistral", "deepseek", "hermes",
        "granite", "glm47", "step3p5", "nemotron", "minimax", "kimi",
    }
    VALID_CACHE_TYPES = {"kv", "mamba", "hybrid"}

    def test_all_reasoning_parsers_valid(self, registry):
        """Every registered model's reasoning_parser must be a known parser."""
        for config in registry._configs:
            assert config.reasoning_parser in self.VALID_REASONING_PARSERS, (
                f"{config.family_name} has invalid reasoning_parser: "
                f"{config.reasoning_parser!r}"
            )

    def test_all_tool_parsers_valid(self, registry):
        """Every registered model's tool_parser must be a known parser."""
        for config in registry._configs:
            assert config.tool_parser in self.VALID_TOOL_PARSERS, (
                f"{config.family_name} has invalid tool_parser: "
                f"{config.tool_parser!r}"
            )

    def test_all_cache_types_valid(self, registry):
        """Every registered model must have a valid cache_type."""
        for config in registry._configs:
            assert config.cache_type in self.VALID_CACHE_TYPES, (
                f"{config.family_name} has invalid cache_type: "
                f"{config.cache_type!r}"
            )

    def test_think_in_template_requires_reasoning_parser(self, registry):
        """If think_in_template is True, model should have a reasoning parser
        (the parser is needed to extract <think> blocks from output)."""
        for config in registry._configs:
            if config.think_in_template:
                assert config.reasoning_parser is not None, (
                    f"{config.family_name} has think_in_template=True but no "
                    f"reasoning_parser — parser needed to extract <think> blocks"
                )

    def test_mllm_models_comprehensive(self, registry):
        """All known VLM/MLLM model families must have is_mllm=True."""
        mllm_families = {
            c.family_name for c in registry._configs if c.is_mllm
        }
        expected_mllm = {
            "qwen3_5", "qwen3_5_moe", "qwen3_vl", "qwen2_vl",
            "pixtral", "deepseek_vl", "gemma3", "paligemma",
            "phi4_multimodal", "phi3_v", "llava", "idefics",
            "cogvlm", "florence", "molmo", "minicpm_v", "smolvlm",
            "internvl", "internlm_xcomposer", "step_vl",
        }
        for family in expected_mllm:
            assert family in mllm_families, (
                f"Expected {family} to be MLLM but is_mllm is False"
            )

    def test_no_duplicate_model_types(self, registry):
        """No model_type should appear in multiple families (would cause ambiguous lookup)."""
        seen = {}
        for config in registry._configs:
            for mt in config.model_types:
                if mt in seen:
                    # Allow if both have the same family_name (re-registration)
                    assert seen[mt] == config.family_name, (
                        f"model_type '{mt}' registered in both "
                        f"'{seen[mt]}' and '{config.family_name}'"
                    )
                seen[mt] = config.family_name

    def test_specific_families_higher_priority_than_generic(self, registry):
        """Specific model families (VL variants, MoE) should have lower priority
        numbers (= higher precedence) than their generic counterparts."""
        family_priority = {c.family_name: c.priority for c in registry._configs}
        checks = [
            ("qwen3_vl", "qwen3"),
            ("qwen3_5", "qwen3"),
            ("qwen3_moe", "qwen3"),
            ("pixtral", "mistral"),
            ("deepseek_vl", "deepseek"),
            ("phi4_reasoning", "phi4"),
            ("phi3_v", "phi3"),
            ("glm4_moe", "chatglm"),
            ("falcon_mamba", "mamba"),
        ]
        for specific, generic in checks:
            if specific in family_priority and generic in family_priority:
                assert family_priority[specific] <= family_priority[generic], (
                    f"{specific} (priority={family_priority[specific]}) should have "
                    f"<= priority than {generic} (priority={family_priority[generic]})"
                )

    def test_qwen3_vl_separate_from_qwen3(self, registry):
        """Qwen3-VL must resolve to qwen3_vl (MLLM), not qwen3 (text-only)."""
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("qwen3_vl")):
            config = registry.lookup("Qwen3-VL-8B")
        assert config.family_name == "qwen3_vl"
        assert config.is_mllm is True

    def test_glm_flash_separate_from_chatglm(self, registry):
        """GLM-4.7 Flash (MoE) must NOT fall through to chatglm."""
        registry.clear_cache()
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("glm4_moe")):
            config = registry.lookup("THUDM/GLM-4-Flash")
        assert config.family_name == "glm4_moe"
        assert config.reasoning_parser == "openai_gptoss"
