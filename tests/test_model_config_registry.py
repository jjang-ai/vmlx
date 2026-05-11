# SPDX-License-Identifier: Apache-2.0
"""Tests for model config registry and model configs.

The registry uses model_types (from config.json model_type field) for lookup.
Since unit tests cannot read from disk/HuggingFace, we mock mlx_lm.utils.load_config
to return the expected model_type for each test.
"""

import json
import logging
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

    def test_lookup_image_model_dirs_do_not_warn_about_missing_text_config(
        self, empty_registry, tmp_path, caplog
    ):
        """Diffusers/mflux image directories are not text model config misses."""
        diffusers_dir = tmp_path / "Qwen-Image"
        diffusers_dir.mkdir()
        (diffusers_dir / "model_index.json").write_text(
            json.dumps({"_class_name": "QwenImagePipeline"})
        )

        mflux_dir = tmp_path / "Z-Image-Turbo"
        (mflux_dir / "transformer").mkdir(parents=True)
        (mflux_dir / "text_encoder").mkdir()

        with patch(
            "vmlx_engine.model_config_registry.load_config",
            side_effect=AssertionError("text load_config should not run for image dirs"),
        ):
            with caplog.at_level(logging.WARNING):
                assert empty_registry.lookup(str(diffusers_dir)).family_name == "unknown"
                assert empty_registry.lookup(str(mflux_dir)).family_name == "unknown"

        assert "Could not load config.json" not in caplog.text

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

    def test_gemma4_wrapper_with_vision_prefers_top_level_vlm_family(self, empty_registry, tmp_path):
        """Do not demote Gemma 4 VLM wrappers to gemma4_text via text_config."""
        import json

        gemma4 = ModelConfig(
            family_name="gemma4",
            model_types=["gemma4"],
            is_mllm=True,
            architecture_hints={"inject_pixel_values": True},
            priority=5,
        )
        gemma4_text = ModelConfig(
            family_name="gemma4_text",
            model_types=["gemma4_text"],
            is_mllm=False,
            priority=4,
        )
        empty_registry.register(gemma4)
        empty_registry.register(gemma4_text)

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "gemma4",
            "text_config": {"model_type": "gemma4_text"},
            "vision_config": {"model_type": "siglip_vision_model"},
        }))

        def _load_config(path):
            return json.loads((path / "config.json").read_text())

        with patch("vmlx_engine.model_config_registry.load_config", _load_config):
            result = empty_registry.lookup(str(tmp_path))

        assert result.family_name == "gemma4"
        assert result.is_mllm is True
        assert result.architecture_hints.get("inject_pixel_values") is True

    def test_gemma4_registry_pins_thinking_support_explicitly(self, empty_registry):
        """Gemma 4 templates expose an enable_thinking rail; don't leave this implicit."""
        from vmlx_engine.model_configs import register_all

        register_all(empty_registry)
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("gemma4")):
            config = empty_registry.lookup("google/gemma-4-26b-it")

        assert config.family_name == "gemma4"
        assert config.reasoning_parser == "gemma4"
        assert config.supports_thinking is True
        assert config.think_in_template is False

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

    def test_jang_stamp_model_type_alias_preserves_registered_family_fields(
        self, empty_registry, tmp_path
    ):
        """A JANG stamp may name the HF model_type, not the vMLX family.

        Ling bundles stamp capabilities.family="bailing_hybrid". The canonical
        registry family is "ling", and that entry carries the dual EOS tokens
        required to stop generation. The stamp should override parser/cache
        fields without discarding the registered family identity or EOS list.
        """
        import json

        empty_registry.register(
            ModelConfig(
                family_name="ling",
                model_types=["bailing_hybrid", "bailing_moe_v2_5"],
                cache_type="hybrid",
                eos_tokens=["<|role_end|>", "<|endoftext|>"],
                tool_parser="deepseek",
                reasoning_parser="deepseek_r1",
                priority=10,
            )
        )
        (tmp_path / "jang_config.json").write_text(
            json.dumps(
                {
                    "capabilities": {
                        "family": "bailing_hybrid",
                        "cache_type": "hybrid",
                        "tool_parser": "deepseek",
                        "reasoning_parser": "deepseek_r1",
                        "think_in_template": False,
                        "modality": "text",
                    }
                }
            )
        )

        result = empty_registry.lookup(str(tmp_path))

        assert result.family_name == "ling"
        assert result.eos_tokens == ["<|role_end|>", "<|endoftext|>"]
        assert result.cache_type == "hybrid"
        assert result.tool_parser == "deepseek"
        # Eric directive 2026-05-11: Ling is NOT a reasoning model. The
        # canonical contract pins reasoning_parser=None and
        # supports_thinking=False even when the bundle stamp claims
        # capabilities.reasoning_parser="deepseek_r1".
        assert result.reasoning_parser is None
        assert result.supports_thinking is False
        assert result.think_in_template is False

    def test_synthetic_no_thinking_family_suppresses_stale_reasoning_stamp(
        self, empty_registry, tmp_path
    ):
        """No-thinking base configs must not resurrect stale parser stamps.

        This intentionally uses a synthetic Ling-shaped config with
        supports_thinking=False so the resolver override stays pinned even
        though the production Ling registry entry is now default-off but
        opt-in thinking-capable.
        """
        import json

        empty_registry.register(
            ModelConfig(
                family_name="synthetic_no_think",
                model_types=["bailing_hybrid", "bailing_moe_v2_5"],
                cache_type="hybrid",
                tool_parser="deepseek",
                reasoning_parser=None,
                think_in_template=False,
                supports_thinking=False,
                priority=10,
            )
        )
        (tmp_path / "jang_config.json").write_text(
            json.dumps(
                {
                    "capabilities": {
                        "family": "bailing_hybrid",
                        "cache_type": "hybrid",
                        "tool_parser": "deepseek",
                        "reasoning_parser": "deepseek_r1",
                        "think_in_template": True,
                        "supports_thinking": True,
                        "modality": "text",
                    }
                }
            )
        )

        result = empty_registry.lookup(str(tmp_path))

        assert result.family_name == "synthetic_no_think"
        assert result.supports_thinking is False
        assert result.reasoning_parser is None
        assert result.think_in_template is False
        assert result.tool_parser == "deepseek"

    def test_zaya_is_reasoning_capable_but_does_not_auto_open_in_no_think_prompt(
        self, empty_registry, tmp_path
    ):
        """Text ZAYA is a reasoning model with a default-off-safe prompt.

        The real ZAYA text template supports `enable_thinking=True` and the
        upstream card recommends qwen3 extraction. With enable_thinking=False
        it renders a closed empty think block, so think_in_template remains
        False even though supports_thinking must be True.
        """
        import json

        empty_registry.register(
            ModelConfig(
                family_name="zaya",
                model_types=["zaya"],
                cache_type="hybrid",
                cache_subtype="zaya_cca",
                tool_parser="zaya_xml",
                reasoning_parser="qwen3",
                think_in_template=False,
                supports_thinking=True,
                priority=10,
            )
        )
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "zaya"}))
        (tmp_path / "jang_config.json").write_text(
            json.dumps(
                {
                    "cache_subtype": "zaya_cca",
                    "capabilities": {
                        "family": "zaya",
                        "cache_type": "hybrid",
                        "cache_subtype": "zaya_cca",
                        "tool_parser": "zaya_xml",
                        "reasoning_parser": "qwen3",
                        "think_in_template": True,
                        "supports_thinking": True,
                        "modality": "text",
                    }
                }
            )
        )

        result = empty_registry.lookup(str(tmp_path))

        assert result.family_name == "zaya"
        assert result.cache_subtype == "zaya_cca"
        assert result.tool_parser == "zaya_xml"
        assert result.reasoning_parser == "qwen3"
        assert result.supports_thinking is True
        assert result.think_in_template is False

    def test_zaya1_vl_vision_config_beats_stale_text_modality_stamp(
        self, empty_registry, tmp_path
    ):
        """Old or hand-edited stamps must not demote a real ZAYA1-VL bundle.

        ZAYA1-VL has `vision_config` in config.json. A stale
        `capabilities.modality=text` stamp can exist on local copies, but the
        registry must not use that stale stamp to turn off MLLM routing.
        """
        import json

        empty_registry.register(
            ModelConfig(
                family_name="zaya1_vl",
                model_types=["zaya1_vl"],
                cache_type="hybrid",
                cache_subtype="zaya_cca",
                tool_parser="zaya_xml",
                reasoning_parser="qwen3",
                think_in_template=False,
                supports_thinking=True,
                is_mllm=True,
                priority=10,
            )
        )
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "zaya1_vl",
                    "vision_config": {"model_type": "qwen2_5_vl"},
                }
            )
        )
        (tmp_path / "jang_config.json").write_text(
            json.dumps(
                {
                    "cache_subtype": "zaya_cca",
                    "capabilities": {
                        "family": "zaya1_vl",
                        "cache_type": "hybrid",
                        "cache_subtype": "zaya_cca",
                        "tool_parser": "zaya_xml",
                        "reasoning_parser": "qwen3",
                        "think_in_template": False,
                        "supports_thinking": True,
                        "modality": "text",
                    },
                }
            )
        )

        result = empty_registry.lookup(str(tmp_path))

        assert result.family_name == "zaya1_vl"
        assert result.is_mllm is True
        assert result.cache_subtype == "zaya_cca"
        assert result.reasoning_parser == "qwen3"
        assert result.think_in_template is False
        assert result.supports_thinking is True


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
        # commit 3294a2da: Gemma 3 uses Google's tool_code format (Python code
        # blocks), not hermes JSON. tool_parser flipped hermes → gemma3.
        assert config.tool_parser == "gemma3"

    def test_gemma3_text_config(self, registry):
        config = self._lookup(registry, "google/gemma-3-text-1b", "gemma3_text")
        assert config.family_name == "gemma3_text"
        assert config.is_mllm is False

    # Nemotron
    def test_nemotron_config(self, registry):
        config = self._lookup(registry, "nvidia/Nemotron-4-340B-Instruct", "nemotron")
        assert config.cache_type == "kv"
        assert config.tokenizer_fallback is True

    def test_nemotron_h_config(self, registry):
        config = self._lookup(registry, "nvidia/Nemotron-H-47B-Instruct", "nemotron_h")
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

    # Newer hybrid SSM/Mamba families — vmlx#cache-matrix audit 2026-05-09
    # These were silently falling through to _DEFAULT_CONFIG (family=unknown,
    # cache=kv, tool/reasoning parser=None), which meant:
    #   - cache_type=kv defaulted, instead of "hybrid" → wrong scheduler routing
    #   - no auto tool_parser / reasoning_parser configuration on startup
    # Symmetric with ssm_companion_cache._HYBRID_MODEL_TYPES expansion.
    def test_nemotron_h_v2_config(self, registry):
        config = self._lookup(registry, "nvidia/Nemotron-H2-56B-Instruct", "nemotron_h_v2")
        assert config.family_name in ("nemotron_h", "nemotron_h_v2"), (
            f"nemotron_h_v2 fell through to {config.family_name} — registry is "
            "missing the newer Nemotron-H v2 model_type alias"
        )
        assert config.cache_type == "hybrid"

    def test_granitemoehybrid_config(self, registry):
        config = self._lookup(registry, "ibm-granite/granite-4.0-tiny-base", "granitemoehybrid")
        assert config.family_name in ("granite", "granitemoehybrid"), (
            f"granitemoehybrid fell through to {config.family_name} — registry "
            "missing the IBM Granite MoE Hybrid model_type"
        )
        assert config.cache_type == "hybrid"
        assert config.tool_parser == "granite"

    def test_lfm2_moe_config(self, registry):
        config = self._lookup(registry, "LiquidAI/LFM2-MoE-8B-A1B", "lfm2_moe")
        assert config.family_name != "unknown", (
            "lfm2_moe fell through to unknown — registry missing Liquid AI "
            "LFM2 MoE entry. Without this, no auto tool_parser configures and "
            "cache_type defaults to kv (wrong — LFM2-MoE is hybrid SSM+attention)"
        )
        assert config.cache_type == "hybrid"

    def test_falcon_h1_config(self, registry):
        """Falcon H1 is hybrid SSM+attention (CacheList[ArraysCache, KVCache])."""
        config = self._lookup(registry, "tiiuae/Falcon-H1-7B", "falcon_h1")
        assert config.family_name != "unknown", (
            "falcon_h1 missing from registry — would default to cache_type=kv "
            "and silently lose SSM state across requests"
        )
        assert config.cache_type == "hybrid"

    def test_rwkv7_config(self, registry):
        """RWKV-7 is pure SSM (ArraysCache, no KV)."""
        config = self._lookup(registry, "RWKV/rwkv-7-world-1b5", "rwkv7")
        assert config.family_name != "unknown", (
            "rwkv7 missing — RWKV-7 is pure SSM and needs cache_type=mamba"
        )
        assert config.cache_type == "mamba"

    def test_deepseek_v32_config(self, registry):
        """DeepSeek V3.2 is MLA (kv_lora_rank=512); needs MLA family routing."""
        config = self._lookup(registry, "deepseek-ai/DeepSeek-V3.2-Base", "deepseek_v32")
        assert config.family_name != "unknown", (
            "deepseek_v32 missing from registry — would default cache=kv "
            "(but kv_lora_rank check still catches MLA in scheduler)"
        )
        # deepseek family uses cache_type=kv (MLA H=1 is enforced at runtime
        # via _detect_mla, not via cache_type override).

    # Unknown model
    def test_unknown_model(self, registry):
        config = self._lookup(registry, "completely-unknown-model-xyz", "totally_unknown_type")
        assert config.family_name == "unknown"
        assert config.cache_type == "kv"

    def test_zaya_config(self, registry):
        config = self._lookup(registry, "Zyphra/ZAYA1-8B", "zaya")
        assert config.family_name == "zaya"
        assert config.cache_type == "hybrid"
        assert config.cache_subtype == "zaya_cca"
        assert config.tool_parser == "zaya_xml"
        assert config.reasoning_parser == "qwen3"
        assert config.think_in_template is False
        assert config.supports_thinking is True

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
        """qwen3_5 is shared between text and VL. Registry is_mllm=False;
        VLM detection relies on config.json vision_config."""
        config = self._lookup(registry, "Qwen/Qwen3.5-VL-9B", "qwen3_5")
        assert config.family_name == "qwen3_5"
        assert config.is_mllm is False
        assert config.tool_parser == "qwen"
        assert config.reasoning_parser == "qwen3"
        assert config.think_in_template is True
        assert config.eos_tokens == ["<|im_end|>"]

    def test_qwen3_5_linear_attention_config_uses_hybrid_cache(self, registry):
        """Qwen3.6 dense MXFP4 has no JANG stamp but text_config.layer_types
        declares linear_attention/full_attention. Registry must not report
        plain KV cache for that architecture."""

        def _load_config(path):
            return {
                "model_type": "qwen3_5",
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "layer_types": [
                        "linear_attention",
                        "linear_attention",
                        "linear_attention",
                        "full_attention",
                    ],
                },
            }

        with patch("vmlx_engine.model_config_registry.load_config", _load_config):
            config = registry.lookup("Qwen3.6-27B-MXFP4-CRACK")

        assert config.family_name == "qwen3_5"
        assert config.cache_type == "hybrid"
        assert config.reasoning_parser == "qwen3"

    def test_qwen3_5_moe_text_linear_attention_uses_hybrid_cache(self, registry, tmp_path):
        """Qwen3.6 MoE wrappers match text_config.model_type first; that path
        must still preserve the hybrid cache override."""

        config_json = {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "layer_types": [
                    "LINEAR_ATTENTION",
                    "FULL_ATTENTION",
                ],
            },
        }
        (tmp_path / "config.json").write_text(json.dumps(config_json))

        def _load_config(path):
            return dict(config_json)

        with patch("vmlx_engine.model_config_registry.load_config", _load_config):
            config = registry.lookup(str(tmp_path))

        assert config.family_name == "qwen3_5_moe"
        assert config.cache_type == "hybrid"
        assert config.reasoning_parser == "qwen3"

    def test_qwen3_5_moe_config(self, registry):
        """qwen3_5_moe is shared between text and VL. Registry is_mllm=False."""
        config = self._lookup(registry, "Qwen/Qwen3.5-MoE-VL-38B", "qwen3_5_moe")
        assert config.family_name == "qwen3_5_moe"
        assert config.is_mllm is False
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
        assert config.tool_parser == "qwen"

    def test_qwen2_config(self, registry):
        config = self._lookup(registry, "Qwen/Qwen2.5-7B-Instruct", "qwen2")
        assert config.family_name == "qwen2"
        assert config.tool_parser == "qwen"
        assert config.reasoning_parser is None  # Qwen2 is NOT a reasoning model
        assert config.think_in_template is False  # No <think> injection
        assert config.supports_native_tools is True

    def test_qwen2_moe_config(self, registry):
        config = self._lookup(registry, "Qwen/Qwen2-MoE-57B", "qwen2_moe")
        assert config.family_name == "qwen2"
        assert config.tool_parser == "qwen"
        assert config.reasoning_parser is None

    def test_qwen2_vl_config(self, registry):
        config = self._lookup(registry, "Qwen/Qwen2-VL-7B-Instruct", "qwen2_vl")
        assert config.family_name == "qwen2_vl"
        assert config.is_mllm is True
        assert config.tool_parser == "qwen"
        assert config.reasoning_parser is None  # Qwen2-VL is NOT a reasoning model
        assert config.think_in_template is False

    def test_qwen2_5_vl_config(self, registry):
        config = self._lookup(registry, "Qwen/Qwen2.5-VL-7B-Instruct", "qwen2_5_vl")
        assert config.family_name == "qwen2_vl"
        assert config.is_mllm is True
        assert config.reasoning_parser is None

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

    def test_glm_z1_config(self, registry):
        """GLM-Z1 uses model_type glm4 but should get openai_gptoss reasoning (Harmony protocol) via name match."""
        config = self._lookup(registry, "zai-org/GLM-Z1-32B-0414", "glm4")
        assert config.family_name == "glm_z1"
        assert config.reasoning_parser == "openai_gptoss"
        assert config.think_in_template is False
        assert config.tool_parser == "glm47"

    def test_glm_z1_9b_config(self, registry):
        """GLM-Z1 9B variant also gets openai_gptoss (Harmony protocol)."""
        config = self._lookup(registry, "zai-org/GLM-Z1-9B-0414", "glm4")
        assert config.family_name == "glm_z1"
        assert config.reasoning_parser == "openai_gptoss"

    def test_glm4_base_not_z1(self, registry):
        """Plain GLM-4 (not Z1) should NOT get reasoning parser."""
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
        """Gemma 3 is not a thinking model — reasoning_parser is None.
        commit 3294a2da removed deepseek_r1 (wrong parser for Gemma 3's
        non-existent reasoning stream)."""
        config = self._lookup(registry, "google/gemma-3-2b-it", "gemma3")
        assert config.reasoning_parser is None

    def test_gemma3_text_not_mllm(self, registry):
        config = self._lookup(registry, "google/gemma-3-text-1b", "gemma3_text")
        assert config.is_mllm is False
        # Gemma 3 text — same as gemma3 multimodal: no reasoning stream.
        assert config.reasoning_parser is None

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

    VALID_REASONING_PARSERS = {None, "qwen3", "deepseek_r1", "openai_gptoss", "mistral", "gemma4"}
    VALID_TOOL_PARSERS = {
        None, "qwen", "llama", "mistral", "deepseek", "hermes",
        "granite", "glm47", "step3p5", "nemotron", "minimax", "kimi",
        "dsml", "zaya_xml",
        # Gemma family: commit 3294a2da added `gemma3` for Google's
        # documented ``` ```tool_code\\nname(k=v)\\n``` ``` format, and
        # `gemma3n` for Gemma 3n (same parser, separate registry entry).
        "gemma3", "gemma3n", "gemma4",
        # Hunyuan / Tencent Hy3 (model_type=hy_v3) custom XML tool format:
        # <tool_calls><tool_call>NAME<tool_sep>
        # <arg_key>K</arg_key><arg_value>V</arg_value>...
        # </tool_call></tool_calls>
        # See vmlx_engine/tool_parsers/hunyuan_tool_parser.py + the
        # contract reference at
        # ~/jang/jang-tools/examples/hy3/python_runtime/hy3_parser_contract.py
        "hunyuan",
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
        """Known VLM/MLLM families with dedicated model_types must have is_mllm=True.
        Families with shared model_types (qwen3_5, qwen3_5_moe) use config.json
        vision_config for VLM detection instead of registry is_mllm."""
        mllm_families = {
            c.family_name for c in registry._configs if c.is_mllm
        }
        # Only families with DEDICATED VLM model_types (not shared with text variants)
        expected_mllm = {
            "qwen3_vl", "qwen2_vl",
            "pixtral", "deepseek_vl", "gemma3", "paligemma",
            "phi4_multimodal", "phi3_v", "llava", "idefics",
            "cogvlm", "florence", "molmo", "minicpm_v", "smolvlm",
            "internvl", "internlm_xcomposer", "step_vl",
        }
        for family in expected_mllm:
            assert family in mllm_families, (
                f"Expected {family} to be MLLM but is_mllm is False"
            )
        # Shared model_types must NOT have is_mllm=True in registry
        shared_families = {"qwen3_5", "qwen3_5_moe"}
        for family in shared_families:
            assert family not in mllm_families, (
                f"{family} shares model_type with text variants — "
                f"is_mllm must be False (use config.json vision_config)"
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

    def test_qwen2_must_not_have_reasoning(self, registry):
        """REGRESSION: Qwen2/2.5 are NOT reasoning models. Having reasoning_parser
        or think_in_template=True causes the parser to treat ALL output as reasoning,
        making responses invisible to the user (shown only in thinking panel)."""
        registry.clear_cache()
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("qwen2")):
            config = registry.lookup("Qwen/Qwen2.5-72B-Instruct")
        assert config.reasoning_parser is None, (
            "Qwen2 must NOT have a reasoning parser — it would classify "
            "all output as reasoning, hiding it from the user"
        )
        assert config.think_in_template is False, (
            "Qwen2 must NOT have think_in_template=True — its template "
            "does not inject <think> tags"
        )

    def test_qwen2_vl_must_not_have_reasoning(self, registry):
        """REGRESSION: Qwen2-VL/2.5-VL are NOT reasoning models."""
        registry.clear_cache()
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("qwen2_vl")):
            config = registry.lookup("Qwen/Qwen2-VL-72B-Instruct")
        assert config.reasoning_parser is None
        assert config.think_in_template is False

    def test_deepseek_v4_eos_includes_latest_reminder(self, registry):
        """REGRESSION (2026-05-09): DSV4 jang_config.chat.role_tokens has THREE
        role markers: <｜User｜>, <｜Assistant｜>, <｜latest_reminder｜>. The
        encoder (encoding_dsv4.py) uses all three as prompt-side prefixes.
        Without latest_reminder in eos_tokens, a model hallucination of that
        token mid-response is not stopped — same failure mode as the
        2026-05-03 <｜Assistant｜> hallucination loop. eos_tokens[1:] are
        installed as additional stop strings in scheduler.py:1981-1993."""
        registry.clear_cache()
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("deepseek_v4")):
            config = registry.lookup("DeepSeek-V4-Flash")
        assert config.family_name == "deepseek_v4"
        assert "<｜end▁of▁sentence｜>" in config.eos_tokens
        assert "<｜User｜>" in config.eos_tokens
        assert "<｜Assistant｜>" in config.eos_tokens
        assert "<｜latest_reminder｜>" in config.eos_tokens, (
            "DSV4 eos_tokens must include <｜latest_reminder｜> — it is one "
            "of three role markers in jang_config.chat.role_tokens and the "
            "DSV4 encoder uses it as a prompt-side prefix. A stray emission "
            "mid-response must terminate generation."
        )

    def test_minimax_eos_includes_role_boundary_marker(self, registry):
        """REGRESSION (2026-05-09): MiniMax M2/M2.5/M2.7 chat template uses
        `]~b]` (token id 200019) as the role-boundary prefix for every
        role start (system/user/ai/tool). The generation prompt is
        `]~b]ai\\n<think>\\n`, so the model starts AFTER the role marker
        and legitimate output should NEVER contain `]~b]`. If the model
        hallucinates a new turn (e.g. `]~b]user`), nothing stops it —
        same hallucination-loop class as the 2026-05-03 DSV4 incident.

        Adding `]~b]` to eos_tokens makes scheduler.py:1981-1993 register
        token 200019 as an additional stop. tokenizer.eos_token (`[e~[`,
        200020) remains the primary EOS via eos_tokens[0]."""
        registry.clear_cache()
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("minimax_m2")):
            config = registry.lookup("MiniMax-M2.7-JANGTQ")
        assert config.family_name == "minimax"
        assert config.eos_tokens is not None, (
            "MiniMax has no explicit eos_tokens — it cannot install the "
            "`]~b]` role-boundary stop and is vulnerable to hallucinated "
            "user-turn loops just like DSV4 was."
        )
        assert "]~b]" in config.eos_tokens, (
            "MiniMax eos_tokens must include `]~b]` (token id 200019). It "
            "is the role-boundary prefix used for every turn start in the "
            "chat template (`]~b]system`, `]~b]user`, `]~b]ai`, `]~b]tool`). "
            "Generation begins after `]~b]ai\\n<think>\\n`, so legitimate "
            "model output never contains this token."
        )

    def test_zaya1_vl_registered_with_full_contract(self, registry):
        """REGRESSION (2026-05-09): Local ZAYA1-VL bundles stamp
        capabilities = {family: zaya1_vl, tool_parser: zaya_xml,
        reasoning_parser: qwen3, think_in_template: False, cache_type: hybrid}.
        The vMLX registry only had plain `zaya` (text), so `zaya1_vl`
        model_type fell through to `family_name=unknown` → no tool parser,
        cache_type=kv (wrong), is_mllm=False (wrong, has vision_config).
        That breaks zaya_xml tool parsing, hybrid cache wiring, and MLLM
        batch generator dispatch all at once.

        Per Eric 2026-05-10: ZAYA does reason. supports_thinking=True is the
        honest flag, and qwen3 is the parser for generated <think> blocks.
        think_in_template stays False because the default/no-thinking prompt
        must start generation in visible content, not in an auto-opened think
        rail."""
        registry.clear_cache()
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("zaya1_vl")):
            config = registry.lookup("Zyphra/ZAYA1-VL-8B-MXFP4")
        assert config.family_name == "zaya1_vl", (
            f"zaya1_vl must be a registered family (got {config.family_name!r}). "
            f"Without registration, the bundle falls through to unknown and "
            f"loses tool_parser, cache_subtype, is_mllm, and supports_thinking "
            f"all at once."
        )
        assert config.cache_type == "hybrid", (
            f"zaya1_vl uses hybrid cache (matches jang_config.capabilities."
            f"cache_type). Got {config.cache_type!r}."
        )
        assert config.cache_subtype == "zaya_cca"
        assert config.tool_parser == "zaya_xml"
        assert config.is_mllm is True, (
            "zaya1_vl is a VLM (has vision_config). Without is_mllm=True, "
            "the engine wires it through the LLM batch generator instead of "
            "MLLM batch generator and image inputs break."
        )
        assert config.supports_thinking is True, (
            "Per Eric 2026-05-10 honest-flag directive: ZAYA1-VL does reason "
            "supports_thinking=True; callers opt in via enable_thinking=True."
        )
        assert config.reasoning_parser == "qwen3", (
            "ZAYA1-VL uses qwen3-style <think> blocks in output, so the qwen3 "
            "reasoning parser is the authoritative extractor."
        )
        assert config.think_in_template is False, (
            "With enable_thinking=False the prompt contains a closed empty "
            "<think> block (not auto-opened), so think_in_template stays "
            "False. The parser must not treat the prompt as starting inside "
            "reasoning."
        )

    def test_zaya_stale_stamp_cannot_disable_reasoning_or_reenable_think_seed(
        self, registry, tmp_path
    ):
        """ZAYA is reasoning-capable but default/no-thinking is not open-think.

        Converter stamps have drifted in both directions during the ZAYA
        bring-up. vMLX must keep the runtime contract stable: qwen3 parser,
        zaya_xml tools, supports_thinking=True, think_in_template=False.
        """
        import json

        (tmp_path / "config.json").write_text(json.dumps({"model_type": "zaya"}))
        (tmp_path / "jang_config.json").write_text(
            json.dumps(
                {
                    "capabilities": {
                        "family": "zaya",
                        "cache_type": "hybrid",
                        "cache_subtype": "zaya_cca",
                        "tool_parser": "zaya_xml",
                        "reasoning_parser": "none",
                        "think_in_template": True,
                        "supports_thinking": False,
                        "modality": "text",
                    }
                }
            )
        )

        registry.clear_cache()
        config = registry.lookup(str(tmp_path))

        assert config.family_name == "zaya"
        assert config.cache_type == "hybrid"
        assert config.cache_subtype == "zaya_cca"
        assert config.tool_parser == "zaya_xml"
        assert config.reasoning_parser == "qwen3"
        assert config.supports_thinking is True
        assert config.think_in_template is False

    def test_ling_is_not_a_reasoning_model_per_eric_2026_05_11(self, registry):
        """REGRESSION (2026-05-11): Eric directive — Ling is NOT a reasoning model.

        Supersedes the 2026-05-09 contract that flagged Ling as opt-in
        reasoning-capable via 'detailed thinking on' system messages. The
        decision now is to treat Ling chat output as plain content
        unconditionally, with no reasoning parser auto-attached.

        Why this is the right contract:
        - Ling chat_template.jinja line 1 hardcodes `thinking_option='off'`
          and there is no template variable to flip it on a per-request basis,
          so the system-message-opt-in path was always edge-case and confusing
          for users.
        - Bundle stamps that claim `capabilities.reasoning_parser="deepseek_r1"`
          must NOT resurrect the parser at runtime — the Ling family override
          in `_try_jang_stamp` enforces reasoning_parser=None.
        - Existing tool parser stays `deepseek` because tool calling is
          orthogonal to reasoning.
        """
        registry.clear_cache()
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("bailing_hybrid")):
            config = registry.lookup("inclusionAI/Ling-2.6-flash-JANGTQ")
        assert config.family_name == "ling"
        assert config.cache_type == "hybrid"
        assert config.tool_parser == "deepseek", "tool calling stays — orthogonal to reasoning"
        assert config.think_in_template is False
        assert config.supports_thinking is False, (
            "Eric directive 2026-05-11: Ling is NOT a reasoning model. "
            "Do not advertise thinking capability."
        )
        assert config.reasoning_parser is None, (
            "Eric directive 2026-05-11: do not auto-attach a reasoning parser. "
            "Ling chat output is plain content."
        )

    def test_hy_v3_provisional_contract(self, registry):
        """Tencent Hy3-preview (model_type=hy_v3) provisional contract per
        `~/jang/docs/runtime/2026-05-09-hy3-runtime-handoff-vmlx-python-swift.md`.

        Bundle drops as JANGTQ2 first; vMLX must already detect the family
        and route it through the right cache + parsers when the bundle lands.

        Contract:
            cache_type=kv          (standard causal GQA, NOT MLA/SSM/CCA/VL)
            tool_parser=hunyuan    (custom XML <tool_calls><tool_call> format)
            reasoning_parser=qwen3  (`<think>...</think>` extraction; matches
                                     JANG capabilities + converter stamps)
            think_in_template=False  (default `no_think` emits CLOSED
                                      `<think></think>` prefill; parser must
                                      not pre-classify first chunk as reasoning)
            supports_thinking=True   (model CAN reason via reasoning_effort=
                                      low|high opt-in)
            eos_tokens contain `<｜hy_eos｜>` (primary) plus `<｜hy_User｜>` and
            `<｜hy_Assistant｜>` as role-boundary stops (same hallucination-loop
            defense as DSV4/MiniMax/Kimi)."""
        registry.clear_cache()
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("hy_v3")):
            config = registry.lookup("Tencent/Hy3-preview")
        assert config.family_name == "hy_v3"
        assert config.cache_type == "kv", (
            "Hy3 uses standard causal GQA KV cache. Not MLA/SSM/CCA/VL."
        )
        assert config.tool_parser == "hunyuan"
        assert config.reasoning_parser == "qwen3"
        assert config.think_in_template is False, (
            "Default `no_think` emits CLOSED <think></think> prefill in the "
            "chat template. think_in_template=True would cause the reasoning "
            "parser to pre-classify the first visible content as reasoning."
        )
        assert config.supports_thinking is True
        assert config.supports_native_tools is True
        assert config.eos_tokens is not None
        assert "<｜hy_eos｜>" in config.eos_tokens
        assert "<｜hy_User｜>" in config.eos_tokens, (
            "Role-boundary stop: model output should never emit `<｜hy_User｜>` "
            "mid-response. Same hallucination-loop defense as 2026-05-03 DSV4."
        )
        assert "<｜hy_Assistant｜>" in config.eos_tokens

    def test_hy_v3_stale_stamp_cannot_reenable_static_think_in_template(
        self, registry, tmp_path
    ):
        """Hy3's stamp used to say think_in_template=True, but the actual
        template is request-dependent: default/no_think renders a closed
        `<think></think>` block and only low/high opens `<think>`.
        """
        import json

        (tmp_path / "config.json").write_text(json.dumps({"model_type": "hy_v3"}))
        (tmp_path / "jang_config.json").write_text(
            json.dumps(
                {
                    "capabilities": {
                        "family": "hy_v3",
                        "reasoning_parser": "qwen3",
                        "tool_parser": "hunyuan",
                        "think_in_template": True,
                        "supports_thinking": True,
                        "modality": "text",
                        "cache_type": "kv",
                    }
                }
            )
        )

        registry.clear_cache()
        config = registry.lookup(str(tmp_path))

        assert config.family_name == "hy_v3"
        assert config.reasoning_parser == "qwen3"
        assert config.tool_parser == "hunyuan"
        assert config.supports_thinking is True
        assert config.think_in_template is False

    def test_kimi_k25_eos_includes_role_boundary_markers(self, registry):
        """REGRESSION (2026-05-09): Kimi K2.6 (kimi_k25) chat template uses
        `<|im_user|>` (163587), `<|im_assistant|>` (163588), `<|im_system|>`
        (163594) as role-boundary tokens. The generation prompt is
        `<|im_assistant|>assistant<|im_middle|>`, so the model starts AFTER
        the role marker and legitimate output should never contain
        `<|im_user|>`/`<|im_system|>`. Without these stops, a hallucinated
        new turn does not terminate generation — same hallucination-loop
        class as the 2026-05-03 DSV4 incident. eos_tokens[0] (`<|im_end|>`,
        163586) is the primary EOS already in tokenizer eos_token_id."""
        registry.clear_cache()
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("kimi_k25")):
            config = registry.lookup("Kimi-K2.6-Small-JANGTQ")
        assert config.family_name == "kimi_k25"
        assert config.eos_tokens is not None, (
            "kimi_k25 has no explicit eos_tokens — it cannot install the "
            "`<|im_user|>`/`<|im_system|>` role-boundary stops and is "
            "vulnerable to hallucinated user-turn loops."
        )
        assert "<|im_user|>" in config.eos_tokens, (
            "kimi_k25 eos_tokens must include `<|im_user|>` (163587). It "
            "is the user role-boundary marker; model emitting this is a "
            "hallucinated new turn and must terminate generation."
        )
        assert "<|im_system|>" in config.eos_tokens, (
            "kimi_k25 eos_tokens must include `<|im_system|>` (163594) for "
            "the same reason — model should never emit a system turn."
        )

    def test_qwen3_is_reasoning_but_qwen2_is_not(self, registry):
        """Qwen3 IS a reasoning model, Qwen2 is NOT. They must differ."""
        registry.clear_cache()
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("qwen3")):
            q3 = registry.lookup("Qwen/Qwen3-8B")
        registry.clear_cache()
        with patch("vmlx_engine.model_config_registry.load_config", _mock_load_config("qwen2")):
            q2 = registry.lookup("Qwen/Qwen2.5-7B")
        assert q3.reasoning_parser == "qwen3"
        assert q3.think_in_template is True
        assert q2.reasoning_parser is None
        assert q2.think_in_template is False
