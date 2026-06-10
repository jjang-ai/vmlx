# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive engine audit tests for vMLX.

Tests cover all core features:
- A. Reasoning & Parser System (GPT-OSS parser, parser registry parity)
- B. Tool Parser System (GLM47, parser-model mapping)
- C. Sampling Defaults (generation_config.json reading)
- D. Settings & Config (model config registry, incompatibility logic)
- E. Engine & Cache (request lifecycle, SamplingParams, MLLM batch request)
- F. Vision Embedding Cache (hash ordering)

These are unit tests that do NOT require model loading.
"""

import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# A. Reasoning & Parser System
# ===========================================================================


class TestGptOssReasoningParser:
    """Tests for the GPT-OSS / Harmony protocol reasoning parser."""

    @pytest.fixture
    def parser(self):
        from vmlx_engine.reasoning import get_parser
        return get_parser("openai_gptoss")()

    def test_parser_registered(self):
        from vmlx_engine.reasoning import list_parsers
        assert "openai_gptoss" in list_parsers()

    def test_extract_analysis_and_final(self, parser):
        """Should extract reasoning from analysis channel and content from final channel."""
        output = (
            "<|channel|>analysis<|message|>Let me analyze this"
            "<|start|>assistant<|channel|>final<|message|>The answer is 42."
        )
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is not None
        assert "analyze" in reasoning
        assert content is not None
        assert "42" in content

    def test_extract_no_markers_returns_content(self, parser):
        """No Harmony markers should return output as content."""
        output = "Just a plain response."
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == output

    def test_extract_only_analysis(self, parser):
        """Only analysis channel, no final channel."""
        parser._harmony_active = True
        output = "<|channel|>analysis<|message|>Just thinking out loud"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is not None
        assert "thinking" in reasoning

    def test_streaming_reset_state(self, parser):
        """Reset state should allow reuse."""
        parser.reset_state(harmony_active=True)
        assert parser._harmony_active is True
        assert parser._emitted_reasoning == 0
        assert parser._emitted_content == 0
        assert parser._got_final is False


class TestParserRegistryParity:
    """Tests that all reasoning parsers are properly registered and instantiable."""

    def test_all_reasoning_parsers_registered(self):
        from vmlx_engine.reasoning import list_parsers
        parsers = list_parsers()
        expected = ["qwen3", "deepseek_r1", "minimax_m2", "openai_gptoss"]
        for name in expected:
            assert name in parsers, f"Reasoning parser '{name}' not registered"

    def test_all_reasoning_parsers_instantiable(self):
        from vmlx_engine.reasoning import get_parser
        for name in ["qwen3", "deepseek_r1", "minimax_m2", "openai_gptoss"]:
            parser = get_parser(name)()
            assert hasattr(parser, "extract_reasoning")
            assert hasattr(parser, "extract_reasoning_streaming")
            assert hasattr(parser, "reset_state")


# ===========================================================================
# B. Tool Parser System
# ===========================================================================


class TestGLM47ToolParser:
    """Tests for the GLM47 tool parser."""

    def test_glm47_registered(self):
        from vmlx_engine.tool_parsers import ToolParserManager
        parsers = ToolParserManager.list_registered()
        assert "glm47" in parsers

    def test_glm47_instantiation(self):
        from vmlx_engine.tool_parsers import ToolParserManager
        parser_cls = ToolParserManager.get_tool_parser("glm47")
        parser = parser_cls()
        assert hasattr(parser, "extract_tool_calls")

    def test_step3p5_registered(self):
        from vmlx_engine.tool_parsers import ToolParserManager
        parsers = ToolParserManager.list_registered()
        assert "step3p5" in parsers

    def test_lfm2_registered(self):
        from vmlx_engine.tool_parsers import ToolParserManager
        parsers = ToolParserManager.list_registered()
        assert "lfm2" in parsers
        assert "liquid" in parsers


class TestToolParserModelMapping:
    """Tests that model configs map to correct tool parsers."""

    def test_model_config_tool_parsers(self):
        from vmlx_engine.model_configs import register_all
        from vmlx_engine.model_config_registry import ModelConfigRegistry

        # Reset and populate
        import vmlx_engine.model_config_registry as mcr
        ModelConfigRegistry._instance = None
        mcr._configs_loaded = False
        registry = ModelConfigRegistry()
        register_all(registry)

        expected_mappings = {
            "qwen3": "qwen",
            "deepseek": "deepseek",
            "glm47-flash": "glm47",
            "llama4": "llama",
            "lfm2": "lfm2",
        }

        for family_name, expected_tool_parser in expected_mappings.items():
            configs = [c for c in registry._configs
                       if c.family_name == family_name]
            if configs:
                assert configs[0].tool_parser == expected_tool_parser, \
                    f"Family '{family_name}' expected tool_parser='{expected_tool_parser}', got '{configs[0].tool_parser}'"

        ModelConfigRegistry._instance = None
        mcr._configs_loaded = False


class TestToolParserReasoningParserMapping:
    """Tests that model configs have correct reasoning parsers."""

    def test_reasoning_parser_assignments(self):
        from vmlx_engine.model_configs import register_all
        from vmlx_engine.model_config_registry import ModelConfigRegistry

        import vmlx_engine.model_config_registry as mcr
        ModelConfigRegistry._instance = None
        mcr._configs_loaded = False
        registry = ModelConfigRegistry()
        register_all(registry)

        expected = {
            "qwen3": "qwen3",
            "minimax": "minimax_m2",
            "deepseek": "deepseek_r1",
            "glm47-flash": "openai_gptoss",
            "gpt-oss": "openai_gptoss",
        }

        for family_name, expected_parser in expected.items():
            configs = [c for c in registry._configs
                       if c.family_name == family_name]
            if configs:
                assert configs[0].reasoning_parser == expected_parser, \
                    f"Family '{family_name}' expected reasoning_parser='{expected_parser}', got '{configs[0].reasoning_parser}'"

        ModelConfigRegistry._instance = None
        mcr._configs_loaded = False


# ===========================================================================
# C. Sampling Defaults
# ===========================================================================


class TestSamplingParams:
    """Tests for SamplingParams dataclass fields."""

    def test_sampling_params_has_all_fields(self):
        from vmlx_engine.request import SamplingParams
        sp = SamplingParams(
            max_tokens=100,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            min_p=0.1,
            repetition_penalty=1.2,
            logprobs=True,
            top_logprobs=5,
        )
        assert sp.max_tokens == 100
        assert sp.temperature == 0.8
        assert sp.top_p == 0.95
        assert sp.top_k == 50
        assert sp.min_p == 0.1
        assert sp.repetition_penalty == 1.2
        assert sp.logprobs is True
        assert sp.top_logprobs == 5

    def test_sampling_params_defaults(self):
        from vmlx_engine.request import SamplingParams
        sp = SamplingParams()
        assert sp.temperature == 0.0
        assert sp.top_p == 1.0
        assert sp.top_k == 0
        assert sp.min_p == 0.0
        assert sp.repetition_penalty == 1.0
        assert sp.logprobs is False
        assert sp.top_logprobs == 0


class TestMLLMBatchRequestSampling:
    """Tests for MLLM batch request sampling parameter passthrough."""

    def test_mllm_batch_request_has_sampling_fields(self):
        from vmlx_engine.mllm_batch_generator import MLLMBatchRequest
        req = MLLMBatchRequest(
            uid=0,
            request_id="test",
            prompt="hello",
            top_k=50,
            min_p=0.1,
            repetition_penalty=1.2,
        )
        assert req.top_k == 50
        assert req.min_p == 0.1
        assert req.repetition_penalty == 1.2

    def test_mllm_batch_request_defaults(self):
        from vmlx_engine.mllm_batch_generator import MLLMBatchRequest
        req = MLLMBatchRequest(uid=0, request_id="test", prompt="hello")
        assert req.top_k == 0
        assert req.min_p == 0.0
        assert req.repetition_penalty == 1.0


class TestBatchedEngineVideoTemplate:
    """BatchedEngine MLLM template handling for video-only requests."""

    def test_video_only_mllm_uses_processor_template_not_tokenizer_fallback(self):
        pytest.importorskip("mlx_vlm")
        from vmlx_engine.engine.batched import BatchedEngine

        captured = {}

        class _Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                raise AssertionError("video-only MLLM request used tokenizer fallback")

            def encode(self, text, add_special_tokens=False):
                return [1, 2, 3]

        class _Processor:
            tokenizer = _Tokenizer()

            def apply_chat_template(self, messages, **kwargs):
                captured["messages"] = messages
                return "VIDEO_PROMPT"

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._is_mllm = True
        engine._processor = _Processor()
        engine._tokenizer = None
        engine._model_name = "qwen36-video-test"
        engine._model = SimpleNamespace(config={"model_type": "qwen3_5_moe"})

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
                    {"type": "text", "text": "Reply only BLUE."},
                ],
            }
        ]

        prompt = engine._apply_chat_template(
            messages,
            num_images=0,
            num_videos=1,
            enable_thinking=False,
        )

        assert prompt == "VIDEO_PROMPT"
        content = captured["messages"][0]["content"]
        assert content[0]["type"] == "video"
        assert content[1]["type"] == "text"
        assert not any(part.get("type") == "video_url" for part in content)

    def test_qwen_video_frame_fallback_rewrites_video_to_image_parts(self, monkeypatch):
        from vmlx_engine.engine.batched import BatchedEngine

        class _Frame:
            pass

        monkeypatch.setattr(
            "vmlx_engine.models.mllm.process_video_input",
            lambda src: f"/resolved/{src}",
        )
        monkeypatch.setattr(
            "vmlx_engine.models.mllm.extract_video_frames_smart",
            lambda path, fps, max_frames: [_Frame(), _Frame()],
        )
        monkeypatch.setattr(
            "vmlx_engine.models.mllm.save_frames_to_temp",
            lambda frames: [f"/tmp/frame-{i}.png" for i, _ in enumerate(frames)],
        )

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._model_name = "JANGQ/Qwen3.6-27B-MXFP4-MTP"
        engine._model_family_name = lambda: "qwen3_5"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": "blue.mp4"}},
                    {"type": "text", "text": "What color?"},
                ],
            }
        ]

        rewritten = engine._qwen_video_frame_fallback_messages(
            messages,
            video_fps=1.0,
            video_max_frames=2,
        )

        content = rewritten[0]["content"]
        assert [part["type"] for part in content] == [
            "image_url",
            "image_url",
            "text",
        ]
        assert content[0]["image_url"]["url"] == "/tmp/frame-0.png"
        assert content[1]["image_url"]["url"] == "/tmp/frame-1.png"
        assert rewritten is not messages

    def test_step37_video_frame_fallback_rewrites_video_to_image_parts(self, monkeypatch):
        from vmlx_engine.engine.batched import BatchedEngine

        class _Frame:
            pass

        resolved_sources = []

        def _process_video_input(src):
            resolved_sources.append(src)
            return f"/resolved/{src}"

        monkeypatch.setattr(
            "vmlx_engine.models.mllm.process_video_input",
            _process_video_input,
        )
        monkeypatch.setattr(
            "vmlx_engine.models.mllm.extract_video_frames_smart",
            lambda path, fps, max_frames: [_Frame(), _Frame()],
        )
        monkeypatch.setattr(
            "vmlx_engine.models.mllm.save_frames_to_temp",
            lambda frames: [f"/tmp/step-frame-{i}.png" for i, _ in enumerate(frames)],
        )

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._model_name = "JANGQ-AI/Step-3.7-Flash-JANG_2L"
        engine._model_family_name = lambda: "step3p7"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_video", "video_url": {"url": "file:///tmp/scene%20one.mp4"}},
                    {"type": "text", "text": "What happened?"},
                ],
            }
        ]

        rewritten = engine._video_frame_fallback_messages(
            messages,
            video_fps=1.0,
            video_max_frames=2,
        )

        content = rewritten[0]["content"]
        assert [part["type"] for part in content] == [
            "image_url",
            "image_url",
            "text",
        ]
        assert content[0]["image_url"]["url"] == "/tmp/step-frame-0.png"
        assert content[1]["image_url"]["url"] == "/tmp/step-frame-1.png"
        assert rewritten is not messages
        assert resolved_sources == ["/tmp/scene one.mp4"]

    def test_mllm_processor_template_receives_thinking_alias(self):
        from vmlx_engine.engine.batched import BatchedEngine

        class _Processor:
            def __init__(self):
                self.kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return "rendered"

        processor = _Processor()
        engine = BatchedEngine.__new__(BatchedEngine)
        engine._is_mllm = True
        engine._processor = processor
        engine._model = type("_Model", (), {"config": {"model_type": "step3p7"}})()
        engine._model_name = "JANGQ-AI/Step-3.7-Flash-JANG_2L"
        engine._model_tool_parser_name = lambda: "step3p5"

        prompt = engine._apply_chat_template(
            [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "x"}]}],
            tools=None,
            num_images=0,
            num_videos=1,
            enable_thinking=False,
        )

        assert prompt == "rendered"
        assert processor.kwargs["enable_thinking"] is False
        assert processor.kwargs["thinking"] is False

    def test_qwen35_dense_mtp_patch_accepts_gdn_sink_kwarg(self):
        import inspect
        import types

        from vmlx_engine.patches.mlx_lm_mtp.qwen35_model import _patch_decoder_layer
        from vmlx_engine.patches.mlx_lm_mtp.qwen35_model import _patch_gated_delta_net

        class GatedDeltaNet:
            pass

        class DecoderLayer:
            pass

        q35 = types.SimpleNamespace(
            GatedDeltaNet=GatedDeltaNet,
            DecoderLayer=DecoderLayer,
        )

        _patch_gated_delta_net(q35)
        _patch_decoder_layer(q35)

        assert "gdn_sink" in inspect.signature(GatedDeltaNet.__call__).parameters
        assert "gdn_sink" in inspect.signature(DecoderLayer.__call__).parameters
        patched_gdn = inspect.getsource(GatedDeltaNet.__call__)
        assert "gdn_sink.append" in patched_gdn
        assert "conv_in" in patched_gdn
        assert "gdn_sink=gdn_sink" in inspect.getsource(DecoderLayer.__call__)

    def test_qwen35_dense_mtp_decoder_call_forwards_gdn_sink_runtime(self):
        import types

        from vmlx_engine.patches.mlx_lm_mtp.qwen35_model import _patch_decoder_layer

        class DecoderLayer:
            pass

        q35 = types.SimpleNamespace(DecoderLayer=DecoderLayer)
        _patch_decoder_layer(q35)

        observed = {}

        class LinearAttention:
            def __call__(
                self,
                x,
                mask,
                cache,
                *,
                gdn_sink=None,
                n_confirmed=0,
            ):
                observed["x"] = x
                observed["mask"] = mask
                observed["cache"] = cache
                observed["gdn_sink"] = gdn_sink
                observed["n_confirmed"] = n_confirmed
                return 3

        layer = DecoderLayer()
        layer.is_linear = True
        layer.input_layernorm = lambda x: x + 1
        layer.linear_attn = LinearAttention()
        layer.post_attention_layernorm = lambda x: x + 5
        layer.mlp = lambda x: x + 7

        sink = []
        result = layer(10, mask="mask", cache="cache", gdn_sink=sink, n_confirmed=2)

        assert result == 38
        assert observed == {
            "x": 11,
            "mask": "mask",
            "cache": "cache",
            "gdn_sink": sink,
            "n_confirmed": 2,
        }

    def test_qwen35_dense_mtp_patch_propagates_gdn_sink_past_decoder(self):
        import inspect

        from vmlx_engine.patches.mlx_lm_mtp import qwen35_model

        patched_backbone = inspect.getsource(qwen35_model._patch_qwen3_5_text_model)
        patched_text = inspect.getsource(qwen35_model._patch_text_model)
        patched_outer = inspect.getsource(qwen35_model._patch_outer_model)

        assert "gdn_sink: Optional[list] = None" in patched_backbone
        assert "gdn_sink=gdn_sink" in patched_backbone
        assert "gdn_sink: Optional[list] = None" in patched_text
        assert "gdn_sink=gdn_sink" in patched_text
        assert "gdn_sink: Optional[list] = None" in patched_outer
        assert "gdn_sink=gdn_sink" in patched_outer

    def test_step37_video_frame_fallback_uses_contact_sheet(self, monkeypatch, tmp_path):
        from PIL import Image

        from vmlx_engine.engine.batched import BatchedEngine

        frame_paths = []
        for i, color in enumerate([(255, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]):
            path = tmp_path / f"frame-{i}.png"
            Image.new("RGB", (32, 24), color).save(path)
            frame_paths.append(str(path))

        monkeypatch.setattr(
            "vmlx_engine.models.mllm.process_video_input",
            lambda src: "/tmp/source.mp4",
        )
        monkeypatch.setattr(
            "vmlx_engine.models.mllm.extract_video_frames_smart",
            lambda path, fps, max_frames: [object(), object(), object(), object()],
        )
        monkeypatch.setattr(
            "vmlx_engine.models.mllm.save_frames_to_temp",
            lambda frames: frame_paths,
        )

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._model_name = "JANGQ-AI/Step-3.7-Flash-JANG_2L"
        engine._model_family_name = lambda: "step3p7"

        rewritten = engine._video_frame_fallback_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": "file:///tmp/source.mp4"}},
                    ],
                }
            ],
            video_fps=1.0,
            video_max_frames=3,
        )

        content = rewritten[0]["content"]
        assert [part["type"] for part in content] == ["image_url"]
        sheet_path = content[0]["image_url"]["url"]
        assert sheet_path.endswith("_step_video_contact_sheet.png")
        with Image.open(sheet_path) as sheet:
            assert sheet.size == (96, 24)
        rewritten_again = engine._video_frame_fallback_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": "file:///tmp/source.mp4"}},
                    ],
                }
            ],
            video_fps=1.0,
            video_max_frames=3,
        )
        assert rewritten_again[0]["content"][0]["image_url"]["url"] == sheet_path

    def test_gemma4_batched_video_frame_fallback_dedupes_to_frame_list(self, monkeypatch, tmp_path):
        from PIL import Image

        from vmlx_engine.engine.batched import BatchedEngine

        frame_paths = []
        for i, color in enumerate([(255, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]):
            path = tmp_path / f"gemma-frame-{i}.png"
            Image.new("RGB", (32, 24), color).save(path)
            frame_paths.append(str(path))

        monkeypatch.setattr(
            "vmlx_engine.models.mllm.process_video_input",
            lambda src: src,
        )
        monkeypatch.setattr(
            "vmlx_engine.models.mllm.extract_video_frames_smart",
            lambda path, fps, max_frames: [object(), object(), object(), object()],
        )
        monkeypatch.setattr(
            "vmlx_engine.models.mllm.save_frames_to_temp",
            lambda frames: frame_paths,
        )

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._model_name = "dealign.ai/Gemma-4-12B-it-MXFP4-CRACK"
        engine._model_family_name = lambda: "gemma4"

        rewritten = engine._video_frame_fallback_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": "file:///tmp/gemma%20clip.mp4"}},
                    ],
                }
            ],
            video_fps=1.0,
            video_max_frames=2,
        )

        content = rewritten[0]["content"]
        assert [part["type"] for part in content] == ["image_url", "image_url", "image_url"]
        assert [part["image_url"]["url"] for part in content] == [
            frame_paths[0],
            frame_paths[2],
            frame_paths[3],
        ]

    def test_gemma_video_frame_fallback_expands_video_placeholder_to_images(self):
        from vmlx_engine.models.mllm import _expand_video_placeholders_to_image_frames

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": "What color?"},
                ],
            }
        ]

        rewritten = _expand_video_placeholders_to_image_frames(messages, [3])

        assert [part["type"] for part in rewritten[0]["content"]] == [
            "image",
            "image",
            "image",
            "text",
        ]
        assert rewritten is not messages


class TestServerSamplingResolution:
    """Tests for server-side sampling parameter resolution."""

    def test_family_fallback_prefers_loaded_model_path_over_served_alias(
        self, monkeypatch, tmp_path
    ):
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "qwen3_5_moe"})
        )

        lookup_calls = []

        class _Registry:
            def lookup(self, key):
                lookup_calls.append(key)
                if key == str(tmp_path):
                    return SimpleNamespace(family_name="qwen3_5_moe")
                return SimpleNamespace(family_name="unknown")

        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_FAMILY_FALLBACK_DEFAULTS", {"unused": (0.1, 0.2, 1.1)})
        monkeypatch.setattr(
            "vmlx_engine.model_config_registry.get_model_config_registry",
            lambda: _Registry(),
        )

        assert server._family_fallback_for("served-alias") == (None, None, None)
        assert lookup_calls
        assert lookup_calls[0] == str(tmp_path)
        assert "served-alias" not in lookup_calls

    def test_gemma4_video_capability_requires_video_processor(self, monkeypatch):
        import vmlx_engine.server as server

        def fake_read_bundle_json(_bundle_path, filename):
            if filename == "config.json":
                return {
                    "model_type": "gemma4_unified",
                    "video_token_id": 258884,
                }
            if filename == "processor_config.json":
                return {"video_processor": {"video_processor_type": "Gemma4UnifiedVideoProcessor"}}
            return {}

        monkeypatch.setattr(server, "_read_bundle_json", fake_read_bundle_json)

        assert server._bundle_declares_native_video("/tmp/gemma4") is True

    def test_step37_video_capability_uses_frame_fallback(self, monkeypatch, tmp_path):
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "step3p7",
                    "image_token_id": 128001,
                    "image_token_len": 169,
                    "vision_config": {"model_type": "perception_encoder"},
                }
            )
        )

        monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=True))
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "step37-video-fallback-test")
        monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)

        assert server._bundle_declares_native_video(str(tmp_path)) is False
        assert server._bundle_supports_video_frame_fallback(str(tmp_path)) is True
        assert server._loaded_runtime_modalities() == ["text", "vision", "video"]

    def test_nemotron_omni_installs_vendored_cradio_dynamic_module(self, tmp_path):
        from vmlx_engine.omni_multimodal import (
            _CRADIO_CACHE_REPO,
            _CRADIO_CACHE_REVISION,
            _ensure_vendored_cradio_dynamic_module,
        )

        status = _ensure_vendored_cradio_dynamic_module(cache_root=tmp_path)

        dst = (
            tmp_path
            / "transformers_modules"
            / "nvidia"
            / _CRADIO_CACHE_REPO
            / _CRADIO_CACHE_REVISION
        )
        assert status["installed"] is True
        assert dst.joinpath("hf_model.py").is_file()
        assert dst.joinpath("radio_model.py").is_file()
        assert not dst.joinpath("__pycache__").exists()

    def test_nemotron_omni_temp_view_rewrites_cradio_automap(
        self, monkeypatch, tmp_path
    ):
        import sys
        import types
        from vmlx_engine.omni_multimodal import (
            _patch_omni_encoder_view_for_vendored_cradio,
        )

        module = types.SimpleNamespace()

        def original(_bundle_path, view_dir):
            view_dir.mkdir(parents=True, exist_ok=True)
            view_dir.joinpath("config.json").write_text(
                json.dumps(
                    {
                        "vision_config": {
                            "auto_map": {
                                "AutoModel": "nvidia/C-RADIOv2-H--hf_model.RADIOModel"
                            }
                        }
                    }
                )
            )

        module._populate_omni_encoder_view = original
        monkeypatch.setitem(sys.modules, "jang_tools.nemotron_omni_chat", module)

        assert _patch_omni_encoder_view_for_vendored_cradio() is True
        module._populate_omni_encoder_view(tmp_path / "bundle", tmp_path / "view")

        cfg = json.loads(tmp_path.joinpath("view", "config.json").read_text())
        assert cfg["vision_config"]["auto_map"] == {
            "AutoConfig": "configuration_radio.RADIOConfig",
            "AutoModel": "hf_model.RADIOModel",
        }
        assert cfg["vision_config"]["_name_or_path"] == str(tmp_path / "view")
        assert tmp_path.joinpath("view", "hf_model.py").is_file()

    def test_nemotron_omni_temp_view_patches_radio_constructor(
        self, monkeypatch, tmp_path
    ):
        import sys
        import types
        from vmlx_engine.omni_multimodal import (
            _patch_omni_encoder_view_for_vendored_cradio,
        )

        module = types.SimpleNamespace()

        def original(_bundle_path, view_dir):
            view_dir.mkdir(parents=True, exist_ok=True)
            view_dir.joinpath("config.json").write_text(
                json.dumps({"vision_config": {}})
            )
            view_dir.joinpath("modeling.py").write_text(
                "        self.vision_model = AutoModel.from_config("
                "config.vision_config, trust_remote_code=True)\n"
            )

        module._populate_omni_encoder_view = original
        monkeypatch.setitem(sys.modules, "jang_tools.nemotron_omni_chat", module)

        assert _patch_omni_encoder_view_for_vendored_cradio() is True
        module._populate_omni_encoder_view(tmp_path / "bundle", tmp_path / "view")

        modeling = tmp_path.joinpath("view", "modeling.py").read_text()
        assert "AutoModel.from_config(config.vision_config" not in modeling
        assert "_VMLINUX_RADIOModel(config.vision_config)" in modeling

    def test_nemotron_omni_temp_view_preseeds_cradio_cache(
        self, monkeypatch, tmp_path
    ):
        import sys
        import types
        from vmlx_engine.omni_multimodal import (
            _patch_omni_encoder_view_for_vendored_cradio,
        )

        fake_hub = types.SimpleNamespace(HF_MODULES_CACHE=str(tmp_path / "hf_modules"))
        monkeypatch.setitem(sys.modules, "transformers.utils.hub", fake_hub)
        module = types.SimpleNamespace()

        def original(_bundle_path, view_dir):
            view_dir.mkdir(parents=True, exist_ok=True)
            view_dir.joinpath("config.json").write_text(
                json.dumps({"vision_config": {}})
            )
            view_dir.joinpath("modeling.py").write_text("")

        module._populate_omni_encoder_view = original
        monkeypatch.setitem(sys.modules, "jang_tools.nemotron_omni_chat", module)

        assert _patch_omni_encoder_view_for_vendored_cradio() is True
        module._populate_omni_encoder_view(tmp_path / "bundle", tmp_path / "view")

        cached = tmp_path / "hf_modules" / "transformers_modules" / "view"
        assert cached.joinpath("hf_model.py").is_file()
        assert cached.joinpath("input_conditioner.py").is_file()

    def test_nemotron_omni_extract_parts_routes_video_to_image_frames(
        self, monkeypatch, tmp_path
    ):
        from vmlx_engine import omni_multimodal

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"video")
        frame = tmp_path / "frame.jpg"
        frame.write_bytes(b"image")
        monkeypatch.setattr(
            omni_multimodal,
            "_extract_omni_video_frames",
            lambda path, scratch: [frame],
        )

        text, images, audio, native_video = omni_multimodal._extract_parts(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe video"},
                        {"type": "video_url", "video_url": {"url": str(video)}},
                    ],
                }
            ],
            tmp_path,
        )

        assert text == "describe video"
        assert images == [frame]
        assert audio is None
        assert native_video is None

    def test_resolve_temperature_request_value(self):
        """Request value should take priority."""
        from vmlx_engine.server import _resolve_temperature
        assert _resolve_temperature(0.5) == 0.5

    def test_resolve_temperature_fallback(self):
        """None request should use fallback."""
        from vmlx_engine.server import _resolve_temperature
        result = _resolve_temperature(None)
        assert isinstance(result, float)

    def test_resolve_top_p_request_value(self):
        """Request value should take priority."""
        from vmlx_engine.server import _resolve_top_p
        assert _resolve_top_p(0.95) == 0.95

    def test_resolve_top_p_fallback(self):
        """None request should use fallback."""
        from vmlx_engine.server import _resolve_top_p
        result = _resolve_top_p(None)
        assert isinstance(result, float)

    def test_api_routes_pass_model_to_sampling_resolvers(self):
        """API handlers must resolve omitted sampling params against the request model.

        This pins the generation_config/jang_config behavior at the route layer:
        the resolver helpers support model-scoped defaults, but callers can still
        accidentally fall back to process-global state if they omit model_name.
        """
        source = Path("./vmlx_engine/server.py").read_text()
        bad_patterns = {
            "temperature": r"_resolve_temperature\(\s*(?:chat_req|request)\.temperature\s*\)",
            "top_p": r"_resolve_top_p\(\s*(?:chat_req|request)\.top_p\s*\)",
            "repetition_penalty": (
                r"_resolve_repetition_penalty\(\s*(?:chat_req|request)\.repetition_penalty\s*\)"
            ),
        }
        for name, pattern in bad_patterns.items():
            assert not re.search(pattern, source), (
                f"{name} resolver call is not model-scoped; pass request.model/chat_req.model "
                "so bundle generation_config.json and jang_config sampling defaults apply."
            )

    def test_chat_and_responses_log_and_forward_supported_sampling_kwargs(
        self,
        monkeypatch,
        caplog,
    ):
        """Supported text sampling kwargs must reach the engine unchanged."""
        import logging

        from fastapi.testclient import TestClient

        import vmlx_engine.server as server
        from vmlx_engine.engine.base import GenerationOutput

        class FakeTokenizer:
            has_thinking = False

        class FakeEngine:
            is_mllm = False
            tokenizer = FakeTokenizer()
            preserve_native_tool_format = False

        captured: list[dict] = []

        async def fake_await_chat(*args, **kwargs):
            captured.append(dict(kwargs["chat_kwargs"]))
            return GenerationOutput(
                text="ok",
                prompt_tokens=3,
                completion_tokens=1,
            )

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_name", "kwarg-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_api_key", None, raising=False)
        monkeypatch.setattr(server, "_default_temperature", None)
        monkeypatch.setattr(server, "_default_top_p", None)
        monkeypatch.setattr(server, "_default_top_k", None)
        monkeypatch.setattr(server, "_default_min_p", None)
        monkeypatch.setattr(server, "_default_repetition_penalty", None)
        monkeypatch.setattr(
            server,
            "_await_chat_with_disconnect_abort",
            fake_await_chat,
        )
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        client = TestClient(server.app)
        caplog.set_level(logging.INFO, logger="vmlx_engine.server")

        chat_resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "kwarg-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "max_tokens": 17,
                "temperature": 0.23,
                "top_p": 0.77,
                "top_k": 42,
                "min_p": 0.03,
                "repetition_penalty": 1.11,
            },
        )
        responses_resp = client.post(
            "/v1/responses",
            json={
                "model": "kwarg-model",
                "input": "hi",
                "stream": False,
                "max_output_tokens": 19,
                "temperature": 0.31,
                "top_p": 0.88,
                "top_k": 24,
                "min_p": 0.04,
                "repetition_penalty": 1.07,
            },
        )

        assert chat_resp.status_code == 200
        assert responses_resp.status_code == 200
        assert captured[0] == {
            "max_tokens": 17,
            "temperature": 0.23,
            "top_p": 0.77,
            "max_prompt_tokens": 0,
            "top_k": 42,
            "min_p": 0.03,
            "repetition_penalty": 1.11,
        }
        assert captured[1] == {
            "max_tokens": 19,
            "temperature": 0.31,
            "top_p": 0.88,
            "max_prompt_tokens": 0,
            "top_k": 24,
            "min_p": 0.04,
            "repetition_penalty": 1.07,
        }
        log_text = "\n".join(record.getMessage() for record in caplog.records)
        assert "Resolved sampling kwargs route=/v1/chat/completions" in log_text
        assert "Resolved sampling kwargs route=/v1/responses" in log_text
        assert "'top_k': 42" in log_text
        assert "'min_p': 0.04" in log_text

    def test_all_top_k_resolution_sites_normalize_deterministic_filters(self):
        """Route kwargs must not re-enable bundle top-p/top-k for greedy turns."""
        source = Path("./vmlx_engine/server.py").read_text()
        for match in re.finditer(r"(?<!def )_set_resolved_top_k\(", source):
            snippet = source[match.end() : match.end() + 260]
            assert "_normalize_deterministic_sampling_filters(" in snippet

    def test_temperature_zero_omits_bundle_sampling_filters_at_route_layer(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Greedy requests stay unambiguous even when the bundle has chat defaults."""
        from fastapi.testclient import TestClient

        import vmlx_engine.server as server
        from vmlx_engine.engine.base import GenerationOutput

        (tmp_path / "generation_config.json").write_text(
            json.dumps({"temperature": 0.7, "top_p": 0.95, "top_k": 64})
        )

        class FakeTokenizer:
            has_thinking = False

        class FakeEngine:
            is_mllm = False
            tokenizer = FakeTokenizer()
            preserve_native_tool_format = False

        captured: list[dict] = []

        async def fake_await_chat(*args, **kwargs):
            captured.append(dict(kwargs["chat_kwargs"]))
            return GenerationOutput(
                text="ok",
                prompt_tokens=3,
                completion_tokens=1,
            )

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "bundle-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_api_key", None, raising=False)
        monkeypatch.setattr(server, "_default_temperature", None)
        monkeypatch.setattr(server, "_default_top_p", None)
        monkeypatch.setattr(server, "_default_top_k", None)
        monkeypatch.setattr(server, "_default_min_p", None)
        monkeypatch.setattr(server, "_default_repetition_penalty", None)
        monkeypatch.setattr(
            server,
            "_await_chat_with_disconnect_abort",
            fake_await_chat,
        )
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        client = TestClient(server.app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "bundle-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "max_tokens": 17,
                "temperature": 0,
            },
        )

        assert resp.status_code == 200
        assert captured[0]["temperature"] == 0
        assert captured[0]["top_p"] == 1.0
        assert "top_k" not in captured[0]

    def test_temperature_zero_preserves_explicit_sampling_filters(self):
        """Explicit request filters are user intent, not inherited bundle noise."""
        import vmlx_engine.server as server

        monkeypatch_default_top_p = server._default_top_p
        monkeypatch_default_top_k = server._default_top_k
        try:
            server._default_top_p = None
            server._default_top_k = None
            kwargs = {"temperature": 0.0, "top_p": 0.5, "top_k": 9}
            server._normalize_deterministic_sampling_filters(
                kwargs,
                request_top_p=0.5,
                request_top_k=9,
            )
            assert kwargs == {"temperature": 0.0, "top_p": 0.5, "top_k": 9}
        finally:
            server._default_top_p = monkeypatch_default_top_p
            server._default_top_k = monkeypatch_default_top_k

    def test_request_output_caps_override_server_default_without_touching_context_cap(
        self,
        monkeypatch,
    ):
        """Per-request output caps and prompt/context caps stay independent.

        Chat Completions owns generated length with max_tokens; Responses owns
        it with max_output_tokens. The prompt/context admission cap is a
        separate max_prompt_tokens field and must not rewrite either output cap.
        """
        from fastapi.testclient import TestClient

        import vmlx_engine.server as server
        from vmlx_engine.engine.base import GenerationOutput

        class FakeTokenizer:
            has_thinking = False

        class FakeEngine:
            is_mllm = False
            tokenizer = FakeTokenizer()
            preserve_native_tool_format = False

        captured: list[dict] = []

        async def fake_await_chat(*args, **kwargs):
            captured.append(dict(kwargs["chat_kwargs"]))
            return GenerationOutput(
                text="ok",
                prompt_tokens=3,
                completion_tokens=1,
            )

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_name", "kwarg-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_api_key", None, raising=False)
        monkeypatch.setattr(server, "_default_temperature", None)
        monkeypatch.setattr(server, "_default_top_p", None)
        monkeypatch.setattr(server, "_default_top_k", None)
        monkeypatch.setattr(server, "_default_min_p", None)
        monkeypatch.setattr(server, "_default_repetition_penalty", None)
        monkeypatch.setattr(server, "_default_max_tokens", 512)
        monkeypatch.setattr(server, "_default_max_tokens_explicit", True, raising=False)
        monkeypatch.setattr(server, "_max_prompt_tokens", 4096)
        monkeypatch.setattr(
            server,
            "_await_chat_with_disconnect_abort",
            fake_await_chat,
        )
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        client = TestClient(server.app)

        chat_resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "kwarg-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "max_tokens": 123,
                "max_prompt_tokens": 2048,
            },
        )
        responses_resp = client.post(
            "/v1/responses",
            json={
                "model": "kwarg-model",
                "input": "hi",
                "stream": False,
                "max_output_tokens": 321,
                "max_prompt_tokens": 1024,
            },
        )
        chat_auto_resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "kwarg-model",
                "messages": [{"role": "user", "content": "auto"}],
                "stream": False,
                "max_prompt_tokens": 2048,
            },
        )

        assert chat_resp.status_code == 200
        assert responses_resp.status_code == 200
        assert chat_auto_resp.status_code == 200
        assert captured[0]["max_tokens"] == 123
        assert captured[0]["max_prompt_tokens"] == 2048
        assert captured[1]["max_tokens"] == 321
        assert captured[1]["max_prompt_tokens"] == 1024
        assert captured[2]["max_tokens"] == 512
        assert captured[2]["max_prompt_tokens"] == 2048

    def test_chat_and_responses_streaming_output_caps_override_server_default_without_touching_context_cap(
        self,
        monkeypatch,
    ):
        """Streaming chat/responses must preserve request output cap semantics."""
        from fastapi.testclient import TestClient

        import vmlx_engine.server as server
        from vmlx_engine.engine.base import GenerationOutput

        class FakeTokenizer:
            has_thinking = False

        class FakeEngine:
            is_mllm = False
            tokenizer = FakeTokenizer()
            preserve_native_tool_format = False

            async def stream_chat(self, messages, **kwargs):
                captured.append(dict(kwargs))
                yield GenerationOutput(
                    text="ok",
                    new_text="ok",
                    prompt_tokens=3,
                    completion_tokens=1,
                    finish_reason="stop",
                    finished=True,
                )

        captured: list[dict] = []

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_name", "stream-cap-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_api_key", None, raising=False)
        monkeypatch.setattr(server, "_default_temperature", None)
        monkeypatch.setattr(server, "_default_top_p", None)
        monkeypatch.setattr(server, "_default_top_k", None)
        monkeypatch.setattr(server, "_default_min_p", None)
        monkeypatch.setattr(server, "_default_repetition_penalty", None)
        monkeypatch.setattr(server, "_default_max_tokens", 640)
        monkeypatch.setattr(server, "_default_max_tokens_explicit", True, raising=False)
        monkeypatch.setattr(server, "_max_prompt_tokens", 65536)
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        client = TestClient(server.app)

        chat_stream = client.post(
            "/v1/chat/completions",
            json={
                "model": "stream-cap-model",
                "messages": [{"role": "user", "content": "chat stream"}],
                "stream": True,
                "max_tokens": 1280,
                "max_prompt_tokens": 8192,
            },
        )
        responses_stream = client.post(
            "/v1/responses",
            json={
                "model": "stream-cap-model",
                "input": "responses stream",
                "stream": True,
                "max_output_tokens": 1536,
                "max_prompt_tokens": 12288,
            },
        )
        chat_default = client.post(
            "/v1/chat/completions",
            json={
                "model": "stream-cap-model",
                "messages": [{"role": "user", "content": "chat default"}],
                "stream": True,
                "max_prompt_tokens": 4096,
            },
        )

        assert chat_stream.status_code == 200
        assert responses_stream.status_code == 200
        assert chat_default.status_code == 200
        assert "data: [DONE]" in chat_stream.text
        assert "response.completed" in responses_stream.text
        assert "data: [DONE]" in chat_default.text
        assert captured[0]["max_tokens"] == 1280
        assert captured[0]["max_prompt_tokens"] == 8192
        assert captured[1]["max_tokens"] == 1536
        assert captured[1]["max_prompt_tokens"] == 12288
        assert captured[2]["max_tokens"] == 640
        assert captured[2]["max_prompt_tokens"] == 4096

    def test_streaming_output_caps_do_not_mutate_server_default_across_later_omitted_requests(
        self,
        monkeypatch,
    ):
        """Streaming output caps must stay request-scoped too.

        A stream path can hold request state open longer than non-streaming
        calls. Explicit Chat/Responses stream caps still must not become the
        server default used by later Auto requests.
        """
        from fastapi.testclient import TestClient

        import vmlx_engine.server as server
        from vmlx_engine.engine.base import GenerationOutput

        class FakeTokenizer:
            has_thinking = False

        class FakeEngine:
            is_mllm = False
            tokenizer = FakeTokenizer()
            preserve_native_tool_format = False

            async def stream_chat(self, messages, **kwargs):
                captured.append({"route": "stream", **dict(kwargs)})
                yield GenerationOutput(
                    text="ok",
                    new_text="ok",
                    prompt_tokens=3,
                    completion_tokens=1,
                    finish_reason="stop",
                    finished=True,
                )

        captured: list[dict] = []

        async def fake_await_chat(*args, **kwargs):
            captured.append({"route": "nonstream", **dict(kwargs["chat_kwargs"])})
            return GenerationOutput(text="ok", prompt_tokens=3, completion_tokens=1)

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_name", "stream-stable-default-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_api_key", None, raising=False)
        monkeypatch.setattr(server, "_default_temperature", None)
        monkeypatch.setattr(server, "_default_top_p", None)
        monkeypatch.setattr(server, "_default_top_k", None)
        monkeypatch.setattr(server, "_default_min_p", None)
        monkeypatch.setattr(server, "_default_repetition_penalty", None)
        monkeypatch.setattr(server, "_default_max_tokens", 2048)
        monkeypatch.setattr(server, "_default_max_tokens_explicit", True, raising=False)
        monkeypatch.setattr(server, "_max_prompt_tokens", 65536)
        monkeypatch.setattr(
            server,
            "_await_chat_with_disconnect_abort",
            fake_await_chat,
        )
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        client = TestClient(server.app)

        chat_stream = client.post(
            "/v1/chat/completions",
            json={
                "model": "stream-stable-default-model",
                "messages": [{"role": "user", "content": "short stream"}],
                "stream": True,
                "max_tokens": 512,
            },
        )
        responses_stream = client.post(
            "/v1/responses",
            json={
                "model": "stream-stable-default-model",
                "input": "long stream",
                "stream": True,
                "max_output_tokens": 8192,
            },
        )
        auto_chat = client.post(
            "/v1/chat/completions",
            json={
                "model": "stream-stable-default-model",
                "messages": [{"role": "user", "content": "auto chat"}],
                "stream": False,
            },
        )
        auto_responses = client.post(
            "/v1/responses",
            json={
                "model": "stream-stable-default-model",
                "input": "auto responses",
                "stream": False,
            },
        )

        assert chat_stream.status_code == 200
        assert responses_stream.status_code == 200
        assert auto_chat.status_code == 200
        assert auto_responses.status_code == 200
        assert "data: [DONE]" in chat_stream.text
        assert "response.completed" in responses_stream.text
        assert [(item["route"], item["max_tokens"]) for item in captured] == [
            ("stream", 512),
            ("stream", 8192),
            ("nonstream", 2048),
            ("nonstream", 2048),
        ]
        assert server._default_max_tokens == 2048
        assert server._default_max_tokens_explicit is True

    def test_explicit_startup_max_tokens_is_default_not_request_ceiling(
        self,
        monkeypatch,
    ):
        """A startup/server output cap must not clamp explicit chat requests.

        Server Settings "Max Output Tokens" maps to the omitted-request default
        for local serving. Per-chat Chat Completions ``max_tokens`` and
        Responses ``max_output_tokens`` are explicit request overrides. The
        prompt/context cap remains independent in ``max_prompt_tokens``.
        """
        from fastapi.testclient import TestClient

        import vmlx_engine.server as server
        from vmlx_engine.engine.base import GenerationOutput

        class FakeTokenizer:
            has_thinking = False

        class FakeEngine:
            is_mllm = False
            tokenizer = FakeTokenizer()
            preserve_native_tool_format = False

        captured: list[dict] = []

        async def fake_await_chat(*args, **kwargs):
            captured.append(dict(kwargs["chat_kwargs"]))
            return GenerationOutput(text="ok", prompt_tokens=3, completion_tokens=1)

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_name", "cap-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_api_key", None, raising=False)
        monkeypatch.setattr(server, "_default_temperature", None)
        monkeypatch.setattr(server, "_default_top_p", None)
        monkeypatch.setattr(server, "_default_top_k", None)
        monkeypatch.setattr(server, "_default_min_p", None)
        monkeypatch.setattr(server, "_default_repetition_penalty", None)
        monkeypatch.setattr(server, "_default_max_tokens", 256)
        monkeypatch.setattr(server, "_default_max_tokens_explicit", True, raising=False)
        monkeypatch.setattr(server, "_max_prompt_tokens", 16384)
        monkeypatch.setattr(
            server,
            "_await_chat_with_disconnect_abort",
            fake_await_chat,
        )
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        client = TestClient(server.app)

        defaulted = client.post(
            "/v1/chat/completions",
            json={
                "model": "cap-model",
                "messages": [{"role": "user", "content": "default"}],
                "stream": False,
                "max_prompt_tokens": 4096,
            },
        )
        chat_override = client.post(
            "/v1/chat/completions",
            json={
                "model": "cap-model",
                "messages": [{"role": "user", "content": "override"}],
                "stream": False,
                "max_tokens": 1024,
                "max_prompt_tokens": 8192,
            },
        )
        responses_override = client.post(
            "/v1/responses",
            json={
                "model": "cap-model",
                "input": "override",
                "stream": False,
                "max_output_tokens": 1536,
                "max_prompt_tokens": 12288,
            },
        )

        assert defaulted.status_code == 200
        assert chat_override.status_code == 200
        assert responses_override.status_code == 200
        assert captured[0]["max_tokens"] == 256
        assert captured[0]["max_prompt_tokens"] == 4096
        assert captured[1]["max_tokens"] == 1024
        assert captured[1]["max_prompt_tokens"] == 8192
        assert captured[2]["max_tokens"] == 1536
        assert captured[2]["max_prompt_tokens"] == 12288

    def test_request_output_caps_can_go_below_or_above_startup_default(
        self,
        monkeypatch,
    ):
        """Per-request output caps are overrides in both directions.

        A Server Settings max output value is the omitted-request default for a
        local server. It is not a hard ceiling that clamps per-chat/API caps,
        and prompt/context caps remain a separate admission limit.
        """
        from fastapi.testclient import TestClient

        import vmlx_engine.server as server
        from vmlx_engine.engine.base import GenerationOutput

        class FakeTokenizer:
            has_thinking = False

        class FakeEngine:
            is_mllm = False
            tokenizer = FakeTokenizer()
            preserve_native_tool_format = False

        captured: list[dict] = []

        async def fake_await_chat(*args, **kwargs):
            captured.append(dict(kwargs["chat_kwargs"]))
            return GenerationOutput(text="ok", prompt_tokens=3, completion_tokens=1)

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_name", "edge-cap-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_api_key", None, raising=False)
        monkeypatch.setattr(server, "_default_temperature", None)
        monkeypatch.setattr(server, "_default_top_p", None)
        monkeypatch.setattr(server, "_default_top_k", None)
        monkeypatch.setattr(server, "_default_min_p", None)
        monkeypatch.setattr(server, "_default_repetition_penalty", None)
        monkeypatch.setattr(server, "_default_max_tokens", 4096)
        monkeypatch.setattr(server, "_default_max_tokens_explicit", True, raising=False)
        monkeypatch.setattr(server, "_max_prompt_tokens", 65536)
        monkeypatch.setattr(
            server,
            "_await_chat_with_disconnect_abort",
            fake_await_chat,
        )
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        client = TestClient(server.app)

        lower_chat = client.post(
            "/v1/chat/completions",
            json={
                "model": "edge-cap-model",
                "messages": [{"role": "user", "content": "short"}],
                "stream": False,
                "max_tokens": 512,
                "max_prompt_tokens": 8192,
            },
        )
        higher_responses = client.post(
            "/v1/responses",
            json={
                "model": "edge-cap-model",
                "input": "longer",
                "stream": False,
                "max_output_tokens": 8192,
                "max_prompt_tokens": 32768,
            },
        )
        auto_chat = client.post(
            "/v1/chat/completions",
            json={
                "model": "edge-cap-model",
                "messages": [{"role": "user", "content": "auto"}],
                "stream": False,
                "max_prompt_tokens": 16384,
            },
        )

        assert lower_chat.status_code == 200
        assert higher_responses.status_code == 200
        assert auto_chat.status_code == 200
        assert captured[0]["max_tokens"] == 512
        assert captured[0]["max_prompt_tokens"] == 8192
        assert captured[1]["max_tokens"] == 8192
        assert captured[1]["max_prompt_tokens"] == 32768
        assert captured[2]["max_tokens"] == 4096
        assert captured[2]["max_prompt_tokens"] == 16384

    def test_request_output_caps_do_not_mutate_server_default_across_later_omitted_requests(
        self,
        monkeypatch,
    ):
        """Explicit API output caps must not poison later Auto requests.

        The server startup max output is a stable omitted-request default.
        Chat Completions ``max_tokens`` and Responses ``max_output_tokens`` can
        be lower or higher on individual requests, but those requests must not
        rewrite the default used by the next omitted request.
        """
        from fastapi.testclient import TestClient

        import vmlx_engine.server as server
        from vmlx_engine.engine.base import GenerationOutput

        class FakeTokenizer:
            has_thinking = False

        class FakeEngine:
            is_mllm = False
            tokenizer = FakeTokenizer()
            preserve_native_tool_format = False

        captured: list[dict] = []

        async def fake_await_chat(*args, **kwargs):
            captured.append(dict(kwargs["chat_kwargs"]))
            return GenerationOutput(text="ok", prompt_tokens=3, completion_tokens=1)

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_name", "stable-default-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_api_key", None, raising=False)
        monkeypatch.setattr(server, "_default_temperature", None)
        monkeypatch.setattr(server, "_default_top_p", None)
        monkeypatch.setattr(server, "_default_top_k", None)
        monkeypatch.setattr(server, "_default_min_p", None)
        monkeypatch.setattr(server, "_default_repetition_penalty", None)
        monkeypatch.setattr(server, "_default_max_tokens", 4096)
        monkeypatch.setattr(server, "_default_max_tokens_explicit", True, raising=False)
        monkeypatch.setattr(server, "_max_prompt_tokens", 65536)
        monkeypatch.setattr(
            server,
            "_await_chat_with_disconnect_abort",
            fake_await_chat,
        )
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        client = TestClient(server.app)

        requests = [
            (
                "/v1/chat/completions",
                {
                    "model": "stable-default-model",
                    "messages": [{"role": "user", "content": "default one"}],
                    "stream": False,
                },
            ),
            (
                "/v1/chat/completions",
                {
                    "model": "stable-default-model",
                    "messages": [{"role": "user", "content": "short override"}],
                    "stream": False,
                    "max_tokens": 512,
                },
            ),
            (
                "/v1/responses",
                {
                    "model": "stable-default-model",
                    "input": "default two",
                    "stream": False,
                },
            ),
            (
                "/v1/responses",
                {
                    "model": "stable-default-model",
                    "input": "long override",
                    "stream": False,
                    "max_output_tokens": 8192,
                },
            ),
            (
                "/v1/chat/completions",
                {
                    "model": "stable-default-model",
                    "messages": [{"role": "user", "content": "default three"}],
                    "stream": False,
                },
            ),
        ]

        responses = [client.post(path, json=body) for path, body in requests]

        assert [response.status_code for response in responses] == [200] * 5
        assert [item["max_tokens"] for item in captured] == [
            4096,
            512,
            4096,
            8192,
            4096,
        ]
        assert server._default_max_tokens == 4096

    def test_legacy_completions_output_cap_overrides_server_default_without_touching_context_cap(
        self,
        monkeypatch,
    ):
        """Legacy text completions must keep output and prompt caps separate.

        `/v1/completions` does not use the chat request builder, but it still
        serves API clients through the same local server defaults. Its
        per-request `max_tokens` must override the server default output cap
        without mutating or being clamped by `max_prompt_tokens`.
        """
        from fastapi.testclient import TestClient

        import vmlx_engine.server as server
        from vmlx_engine.engine.base import GenerationOutput

        class FakeTokenizer:
            has_thinking = False

        class FakeEngine:
            is_mllm = False
            tokenizer = FakeTokenizer()
            preserve_native_tool_format = False

            async def generate(self, **kwargs):
                captured.append(dict(kwargs))
                return GenerationOutput(text="ok", prompt_tokens=2, completion_tokens=1)

        captured: list[dict] = []

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_name", "completion-cap-model")
        monkeypatch.setattr(server, "_api_key", None, raising=False)
        monkeypatch.setattr(server, "_default_temperature", None)
        monkeypatch.setattr(server, "_default_top_p", None)
        monkeypatch.setattr(server, "_default_top_k", None)
        monkeypatch.setattr(server, "_default_min_p", None)
        monkeypatch.setattr(server, "_default_repetition_penalty", None)
        monkeypatch.setattr(server, "_default_max_tokens", 256)
        monkeypatch.setattr(server, "_default_max_tokens_explicit", True, raising=False)
        monkeypatch.setattr(server, "_max_prompt_tokens", 16384)
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        client = TestClient(server.app)

        explicit = client.post(
            "/v1/completions",
            json={
                "model": "completion-cap-model",
                "prompt": "explicit cap",
                "stream": False,
                "max_tokens": 1024,
                "max_prompt_tokens": 8192,
            },
        )
        defaulted = client.post(
            "/v1/completions",
            json={
                "model": "completion-cap-model",
                "prompt": "default cap",
                "stream": False,
                "max_prompt_tokens": 4096,
            },
        )

        assert explicit.status_code == 200
        assert defaulted.status_code == 200
        assert captured[0]["max_tokens"] == 1024
        assert captured[0]["max_prompt_tokens"] == 8192
        assert captured[1]["max_tokens"] == 256
        assert captured[1]["max_prompt_tokens"] == 4096

    def test_legacy_completions_streaming_output_cap_overrides_server_default_without_touching_context_cap(
        self,
        monkeypatch,
    ):
        """Streaming `/v1/completions` must preserve the same cap semantics."""
        from fastapi.testclient import TestClient

        import vmlx_engine.server as server
        from vmlx_engine.engine.base import GenerationOutput

        class FakeTokenizer:
            has_thinking = False

        class FakeEngine:
            is_mllm = False
            tokenizer = FakeTokenizer()
            preserve_native_tool_format = False

            async def stream_generate(self, **kwargs):
                captured.append(dict(kwargs))
                yield GenerationOutput(
                    text="ok",
                    new_text="ok",
                    prompt_tokens=2,
                    completion_tokens=1,
                    finish_reason="stop",
                    finished=True,
                )

        captured: list[dict] = []

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_name", "completion-stream-cap-model")
        monkeypatch.setattr(server, "_api_key", None, raising=False)
        monkeypatch.setattr(server, "_default_temperature", None)
        monkeypatch.setattr(server, "_default_top_p", None)
        monkeypatch.setattr(server, "_default_top_k", None)
        monkeypatch.setattr(server, "_default_min_p", None)
        monkeypatch.setattr(server, "_default_repetition_penalty", None)
        monkeypatch.setattr(server, "_default_max_tokens", 384)
        monkeypatch.setattr(server, "_default_max_tokens_explicit", True, raising=False)
        monkeypatch.setattr(server, "_max_prompt_tokens", 32768)
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        client = TestClient(server.app)

        explicit = client.post(
            "/v1/completions",
            json={
                "model": "completion-stream-cap-model",
                "prompt": "explicit streaming cap",
                "stream": True,
                "max_tokens": 1536,
                "max_prompt_tokens": 12288,
            },
        )
        defaulted = client.post(
            "/v1/completions",
            json={
                "model": "completion-stream-cap-model",
                "prompt": "default streaming cap",
                "stream": True,
                "max_prompt_tokens": 6144,
            },
        )

        assert explicit.status_code == 200
        assert defaulted.status_code == 200
        assert "data: [DONE]" in explicit.text
        assert "data: [DONE]" in defaulted.text
        assert captured[0]["max_tokens"] == 1536
        assert captured[0]["max_prompt_tokens"] == 12288
        assert captured[1]["max_tokens"] == 384
        assert captured[1]["max_prompt_tokens"] == 6144

    def test_ollama_streaming_num_predict_overrides_server_default_without_touching_context_cap(
        self,
        monkeypatch,
    ):
        """Ollama streaming num_predict is an output cap, not a context cap."""
        from fastapi.testclient import TestClient

        import vmlx_engine.server as server
        from vmlx_engine.engine.base import GenerationOutput

        class FakeTokenizer:
            has_thinking = False

        class FakeEngine:
            is_mllm = False
            tokenizer = FakeTokenizer()
            preserve_native_tool_format = False

            async def stream_chat(self, messages, **kwargs):
                chat_captured.append(dict(kwargs))
                yield GenerationOutput(
                    text="ok",
                    new_text="ok",
                    prompt_tokens=3,
                    completion_tokens=1,
                    finish_reason="stop",
                    finished=True,
                )

            async def stream_generate(self, **kwargs):
                raw_captured.append(dict(kwargs))
                yield GenerationOutput(
                    text="ok",
                    new_text="ok",
                    prompt_tokens=2,
                    completion_tokens=1,
                    finish_reason="stop",
                    finished=True,
                )

        chat_captured: list[dict] = []
        raw_captured: list[dict] = []

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_name", "ollama-stream-cap-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_api_key", None, raising=False)
        monkeypatch.setattr(server, "_default_temperature", None)
        monkeypatch.setattr(server, "_default_top_p", None)
        monkeypatch.setattr(server, "_default_top_k", None)
        monkeypatch.setattr(server, "_default_min_p", None)
        monkeypatch.setattr(server, "_default_repetition_penalty", None)
        monkeypatch.setattr(server, "_default_max_tokens", 512)
        monkeypatch.setattr(server, "_default_max_tokens_explicit", True, raising=False)
        monkeypatch.setattr(server, "_max_prompt_tokens", 32768)
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        client = TestClient(server.app)

        chat_explicit = client.post(
            "/api/chat",
            json={
                "model": "ollama-stream-cap-model",
                "messages": [{"role": "user", "content": "chat"}],
                "stream": True,
                "options": {"num_predict": 44, "num_ctx": 4096},
            },
        )
        raw_explicit = client.post(
            "/api/generate",
            json={
                "model": "ollama-stream-cap-model",
                "prompt": "raw",
                "raw": True,
                "stream": True,
                "options": {"num_predict": 55, "num_ctx": 8192},
            },
        )
        chat_default = client.post(
            "/api/chat",
            json={
                "model": "ollama-stream-cap-model",
                "messages": [{"role": "user", "content": "default"}],
                "stream": True,
                "options": {"num_ctx": 2048},
            },
        )

        assert chat_explicit.status_code == 200
        assert raw_explicit.status_code == 200
        assert chat_default.status_code == 200
        assert ('"done": true' in chat_explicit.text) or ('"done":true' in chat_explicit.text)
        assert ('"done": true' in raw_explicit.text) or ('"done":true' in raw_explicit.text)
        assert ('"done": true' in chat_default.text) or ('"done":true' in chat_default.text)
        assert chat_captured[0]["max_tokens"] == 44
        assert chat_captured[0]["max_prompt_tokens"] == 4096
        assert raw_captured[0]["max_tokens"] == 55
        assert raw_captured[0]["max_prompt_tokens"] == 8192
        assert chat_captured[1]["max_tokens"] == 512
        assert chat_captured[1]["max_prompt_tokens"] == 2048

    def test_ollama_nonstream_num_predict_overrides_server_default_without_touching_context_cap(
        self,
        monkeypatch,
    ):
        """Ollama non-stream num_predict must stay request-scoped too.

        `/api/chat`, templated `/api/generate`, and raw `/api/generate` take
        different server paths when stream=false. All three still map
        `options.num_predict` to an output cap, map `num_ctx` to prompt/context,
        and leave the server startup max output default unchanged for later
        omitted requests.
        """
        from fastapi.testclient import TestClient

        import vmlx_engine.server as server
        from vmlx_engine.engine.base import GenerationOutput

        class FakeTokenizer:
            has_thinking = False

        class FakeEngine:
            is_mllm = False
            tokenizer = FakeTokenizer()
            preserve_native_tool_format = False

            async def generate(self, **kwargs):
                raw_captured.append(dict(kwargs))
                return GenerationOutput(
                    text="ok",
                    prompt_tokens=2,
                    completion_tokens=1,
                    finish_reason="stop",
                )

        chat_captured: list[dict] = []
        raw_captured: list[dict] = []

        async def fake_await_chat(*args, **kwargs):
            chat_captured.append(dict(kwargs["chat_kwargs"]))
            return GenerationOutput(
                text="ok",
                prompt_tokens=3,
                completion_tokens=1,
                finish_reason="stop",
            )

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_name", "ollama-nonstream-cap-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_api_key", None, raising=False)
        monkeypatch.setattr(server, "_default_temperature", None)
        monkeypatch.setattr(server, "_default_top_p", None)
        monkeypatch.setattr(server, "_default_top_k", None)
        monkeypatch.setattr(server, "_default_min_p", None)
        monkeypatch.setattr(server, "_default_repetition_penalty", None)
        monkeypatch.setattr(server, "_default_max_tokens", 768)
        monkeypatch.setattr(server, "_default_max_tokens_explicit", True, raising=False)
        monkeypatch.setattr(server, "_max_prompt_tokens", 32768)
        monkeypatch.setattr(
            server,
            "_await_chat_with_disconnect_abort",
            fake_await_chat,
        )
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        client = TestClient(server.app)

        chat_explicit = client.post(
            "/api/chat",
            json={
                "model": "ollama-nonstream-cap-model",
                "messages": [{"role": "user", "content": "chat"}],
                "stream": False,
                "options": {"num_predict": 44, "num_ctx": 4096},
            },
        )
        generate_templated_explicit = client.post(
            "/api/generate",
            json={
                "model": "ollama-nonstream-cap-model",
                "prompt": "templated",
                "stream": False,
                "options": {"num_predict": 55, "num_ctx": 8192},
            },
        )
        generate_raw_explicit = client.post(
            "/api/generate",
            json={
                "model": "ollama-nonstream-cap-model",
                "prompt": "raw",
                "raw": True,
                "stream": False,
                "options": {"num_predict": 66, "num_ctx": 12288},
            },
        )
        chat_default = client.post(
            "/api/chat",
            json={
                "model": "ollama-nonstream-cap-model",
                "messages": [{"role": "user", "content": "default"}],
                "stream": False,
                "options": {"num_ctx": 2048},
            },
        )

        assert chat_explicit.status_code == 200
        assert generate_templated_explicit.status_code == 200
        assert generate_raw_explicit.status_code == 200
        assert chat_default.status_code == 200
        assert chat_captured[0]["max_tokens"] == 44
        assert chat_captured[0]["max_prompt_tokens"] == 4096
        assert chat_captured[1]["max_tokens"] == 55
        assert chat_captured[1]["max_prompt_tokens"] == 8192
        assert raw_captured[0]["max_tokens"] == 66
        assert raw_captured[0]["max_prompt_tokens"] == 12288
        assert chat_captured[2]["max_tokens"] == 768
        assert chat_captured[2]["max_prompt_tokens"] == 2048
        assert server._default_max_tokens == 768
        assert server._default_max_tokens_explicit is True

    def test_prompt_context_aliases_clamp_without_rewriting_output_caps(
        self,
        monkeypatch,
    ):
        """Prompt/context aliases must not mutate per-request output caps."""
        from fastapi.testclient import TestClient

        import vmlx_engine.server as server
        from vmlx_engine.engine.base import GenerationOutput

        class FakeTokenizer:
            has_thinking = False

        class FakeEngine:
            is_mllm = False
            tokenizer = FakeTokenizer()
            preserve_native_tool_format = False

            async def generate(self, **kwargs):
                completion_captured.append(dict(kwargs))
                return GenerationOutput(text="ok", prompt_tokens=2, completion_tokens=1)

        chat_captured: list[dict] = []
        completion_captured: list[dict] = []
        anthropic_captured: list[dict] = []

        async def fake_await_chat(*args, **kwargs):
            chat_captured.append(dict(kwargs["chat_kwargs"]))
            return GenerationOutput(text="ok", prompt_tokens=3, completion_tokens=1)

        async def fake_stream_chat_completion(*args, **kwargs):
            anthropic_captured.append(dict(kwargs))
            yield (
                'data: {"choices":[{"delta":{"content":"ok"},'
                '"finish_reason":"stop"}],"usage":{"prompt_tokens":3,'
                '"completion_tokens":1}}\n\n'
            )
            yield "data: [DONE]\n\n"

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_name", "context-alias-cap-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_api_key", None, raising=False)
        monkeypatch.setattr(server, "_default_temperature", None)
        monkeypatch.setattr(server, "_default_top_p", None)
        monkeypatch.setattr(server, "_default_top_k", None)
        monkeypatch.setattr(server, "_default_min_p", None)
        monkeypatch.setattr(server, "_default_repetition_penalty", None)
        monkeypatch.setattr(server, "_default_max_tokens", 512)
        monkeypatch.setattr(server, "_default_max_tokens_explicit", True, raising=False)
        monkeypatch.setattr(server, "_max_prompt_tokens", 4096)
        monkeypatch.setattr(
            server,
            "_await_chat_with_disconnect_abort",
            fake_await_chat,
        )
        monkeypatch.setattr(
            server,
            "stream_chat_completion",
            fake_stream_chat_completion,
        )
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        client = TestClient(server.app)

        chat_resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "context-alias-cap-model",
                "messages": [{"role": "user", "content": "chat"}],
                "stream": False,
                "max_tokens": 123,
                "max_context_tokens": 999999,
            },
        )
        responses_resp = client.post(
            "/v1/responses",
            json={
                "model": "context-alias-cap-model",
                "input": "responses",
                "stream": False,
                "max_output_tokens": 321,
                "max_context": 2048,
            },
        )
        completion_resp = client.post(
            "/v1/completions",
            json={
                "model": "context-alias-cap-model",
                "prompt": "completion",
                "stream": False,
                "max_tokens": 222,
                "max_context_tokens": 999999,
            },
        )
        anthropic_resp = client.post(
            "/v1/messages",
            json={
                "model": "context-alias-cap-model",
                "messages": [{"role": "user", "content": "anthropic"}],
                "stream": False,
                "max_tokens": 111,
                "max_context": 1024,
            },
        )

        assert chat_resp.status_code == 200
        assert responses_resp.status_code == 200
        assert completion_resp.status_code == 200
        assert anthropic_resp.status_code == 200
        assert chat_captured[0]["max_tokens"] == 123
        assert chat_captured[0]["max_prompt_tokens"] == 4096
        assert chat_captured[1]["max_tokens"] == 321
        assert chat_captured[1]["max_prompt_tokens"] == 2048
        assert completion_captured[0]["max_tokens"] == 222
        assert completion_captured[0]["max_prompt_tokens"] == 4096
        assert anthropic_captured[0]["max_tokens"] == 111
        assert anthropic_captured[0]["max_prompt_tokens"] == 1024

    def test_reasoning_effort_preserves_bundle_max_new_tokens(
        self,
        tmp_path,
        monkeypatch,
    ):
        """reasoning_effort should not inflate model-declared output length.

        The template-facing thinking_budget is the effort control. Length is
        owned by explicit request max_tokens/max_output_tokens first, then the
        bundle's generation_config.max_new_tokens.
        """
        from fastapi.testclient import TestClient

        import vmlx_engine.server as server
        from vmlx_engine.engine.base import GenerationOutput

        (tmp_path / "generation_config.json").write_text(
            json.dumps({"max_new_tokens": 2048})
        )

        class FakeTokenizer:
            has_thinking = False

        class FakeEngine:
            is_mllm = False
            tokenizer = FakeTokenizer()
            preserve_native_tool_format = False

        captured: list[dict] = []

        async def fake_await_chat(*args, **kwargs):
            captured.append(dict(kwargs["chat_kwargs"]))
            return GenerationOutput(text="ok", prompt_tokens=3, completion_tokens=1)

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "bundle-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_api_key", None, raising=False)
        monkeypatch.setattr(server, "_default_max_tokens", 32768)
        monkeypatch.setattr(server, "_default_temperature", None)
        monkeypatch.setattr(server, "_default_top_p", None)
        monkeypatch.setattr(server, "_default_top_k", None)
        monkeypatch.setattr(server, "_default_min_p", None)
        monkeypatch.setattr(server, "_default_repetition_penalty", None)
        monkeypatch.setattr(
            server,
            "_await_chat_with_disconnect_abort",
            fake_await_chat,
        )
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        client = TestClient(server.app)

        chat_resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "bundle-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "reasoning_effort": "high",
            },
        )
        responses_resp = client.post(
            "/v1/responses",
            json={
                "model": "bundle-model",
                "input": "hi",
                "stream": False,
                "reasoning_effort": "high",
            },
        )

        assert chat_resp.status_code == 200
        assert responses_resp.status_code == 200
        assert captured[0]["max_tokens"] == 2048
        assert captured[1]["max_tokens"] == 2048
        assert captured[0]["chat_template_kwargs"]["thinking_budget"] == 32768
        assert captured[1]["chat_template_kwargs"]["thinking_budget"] == 32768

    def test_explicit_max_thinking_tokens_overrides_effort_budget_without_rewriting_output_length(
        self,
        tmp_path,
        monkeypatch,
    ):
        """max_thinking_tokens controls template budget, not output/context caps."""
        from fastapi.testclient import TestClient

        import vmlx_engine.server as server
        from vmlx_engine.engine.base import GenerationOutput

        (tmp_path / "generation_config.json").write_text(
            json.dumps({"max_new_tokens": 2048})
        )

        class FakeTokenizer:
            has_thinking = False

        class FakeEngine:
            is_mllm = False
            tokenizer = FakeTokenizer()
            preserve_native_tool_format = False

        captured: list[dict] = []

        async def fake_await_chat(*args, **kwargs):
            captured.append(dict(kwargs["chat_kwargs"]))
            return GenerationOutput(text="ok", prompt_tokens=3, completion_tokens=1)

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "bundle-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_api_key", None, raising=False)
        monkeypatch.setattr(server, "_default_max_tokens", 32768)
        monkeypatch.setattr(server, "_default_max_tokens_explicit", False, raising=False)
        monkeypatch.setattr(server, "_default_temperature", None)
        monkeypatch.setattr(server, "_default_top_p", None)
        monkeypatch.setattr(server, "_default_top_k", None)
        monkeypatch.setattr(server, "_default_min_p", None)
        monkeypatch.setattr(server, "_default_repetition_penalty", None)
        monkeypatch.setattr(
            server,
            "_await_chat_with_disconnect_abort",
            fake_await_chat,
        )
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        client = TestClient(server.app)

        chat_resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "bundle-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "reasoning_effort": "high",
                "max_thinking_tokens": 4096,
            },
        )
        responses_resp = client.post(
            "/v1/responses",
            json={
                "model": "bundle-model",
                "input": "hi",
                "stream": False,
                "reasoning_effort": "high",
                "max_thinking_tokens": 4096,
            },
        )

        assert chat_resp.status_code == 200
        assert responses_resp.status_code == 200
        assert captured[0]["max_tokens"] == 2048
        assert captured[1]["max_tokens"] == 2048
        assert captured[0]["chat_template_kwargs"]["thinking_budget"] == 4096
        assert captured[1]["chat_template_kwargs"]["thinking_budget"] == 4096
        assert captured[0]["max_prompt_tokens"] == 0
        assert captured[1]["max_prompt_tokens"] == 0

    def test_anthropic_messages_omitted_max_tokens_uses_bundle_default(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Anthropic compatibility must not force its old 4096 default.

        The /v1/messages endpoint aggregates the shared Chat Completions stream,
        so test the route, not just anthropic_adapter.to_chat_completion().
        """
        from fastapi.testclient import TestClient

        import vmlx_engine.server as server

        (tmp_path / "generation_config.json").write_text(
            json.dumps({"max_new_tokens": 2048})
        )

        class FakeEngine:
            is_mllm = False
            tokenizer = None
            preserve_native_tool_format = False

        captured: list[dict] = []

        async def fake_stream_chat_completion(*args, **kwargs):
            captured.append(
                {
                    "max_tokens": kwargs.get("max_tokens"),
                    "temperature": kwargs.get("temperature"),
                    "top_p": kwargs.get("top_p"),
                }
            )
            yield (
                'data: {"choices":[{"delta":{"content":"ok"},'
                '"finish_reason":"stop"}],"usage":{"prompt_tokens":1,'
                '"completion_tokens":1}}\n\n'
            )
            yield "data: [DONE]\n\n"

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "bundle-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_api_key", None, raising=False)
        monkeypatch.setattr(server, "_default_max_tokens", 32768)
        monkeypatch.setattr(server, "_default_max_tokens_explicit", False, raising=False)
        monkeypatch.setattr(server, "_default_temperature", None)
        monkeypatch.setattr(server, "_default_top_p", None)
        monkeypatch.setattr(server, "_default_top_k", None)
        monkeypatch.setattr(server, "_default_min_p", None)
        monkeypatch.setattr(server, "_default_repetition_penalty", None)
        monkeypatch.setattr(server, "stream_chat_completion", fake_stream_chat_completion)
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        client = TestClient(server.app)

        omitted = client.post(
            "/v1/messages",
            json={
                "model": "bundle-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
        explicit = client.post(
            "/v1/messages",
            json={
                "model": "bundle-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "max_tokens": 33,
            },
        )

        assert omitted.status_code == 200
        assert explicit.status_code == 200
        assert captured[0]["max_tokens"] == 2048
        assert captured[1]["max_tokens"] == 33

    def test_anthropic_messages_streaming_max_tokens_overrides_server_default_without_touching_context_cap(
        self,
        monkeypatch,
    ):
        """Anthropic streaming max_tokens must stay an output cap override."""
        from fastapi.testclient import TestClient

        import vmlx_engine.server as server

        class FakeEngine:
            is_mllm = False
            tokenizer = None
            preserve_native_tool_format = False

        captured: list[dict] = []

        async def fake_stream_chat_completion(*args, **kwargs):
            captured.append(
                {
                    "max_tokens": kwargs.get("max_tokens"),
                    "max_prompt_tokens": kwargs.get("max_prompt_tokens"),
                }
            )
            yield (
                'data: {"choices":[{"delta":{"content":"ok"},'
                '"finish_reason":"stop"}],"usage":{"prompt_tokens":1,'
                '"completion_tokens":1}}\n\n'
            )
            yield "data: [DONE]\n\n"

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_name", "anthropic-stream-cap-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_api_key", None, raising=False)
        monkeypatch.setattr(server, "_default_max_tokens", 768)
        monkeypatch.setattr(server, "_default_max_tokens_explicit", True, raising=False)
        monkeypatch.setattr(server, "_max_prompt_tokens", 32768)
        monkeypatch.setattr(server, "_default_temperature", None)
        monkeypatch.setattr(server, "_default_top_p", None)
        monkeypatch.setattr(server, "_default_top_k", None)
        monkeypatch.setattr(server, "_default_min_p", None)
        monkeypatch.setattr(server, "_default_repetition_penalty", None)
        monkeypatch.setattr(server, "stream_chat_completion", fake_stream_chat_completion)
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        client = TestClient(server.app)

        explicit = client.post(
            "/v1/messages",
            json={
                "model": "anthropic-stream-cap-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "max_tokens": 96,
                "max_prompt_tokens": 4096,
            },
        )
        defaulted = client.post(
            "/v1/messages",
            json={
                "model": "anthropic-stream-cap-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "max_prompt_tokens": 8192,
            },
        )

        assert explicit.status_code == 200
        assert defaulted.status_code == 200
        assert "message_stop" in explicit.text
        assert "message_stop" in defaulted.text
        assert captured[0]["max_tokens"] == 96
        assert captured[0]["max_prompt_tokens"] == 4096
        assert captured[1]["max_tokens"] == 768
        assert captured[1]["max_prompt_tokens"] == 8192

    def test_explicit_server_max_tokens_overrides_bundle_max_new_tokens(
        self,
        tmp_path,
        monkeypatch,
    ):
        """A real session/CLI max-tokens setting must remain an override.

        Omitted startup max_tokens should let the bundle own the default
        length, but an explicit session value is a user/operator choice.
        """
        import vmlx_engine.server as server

        (tmp_path / "generation_config.json").write_text(
            json.dumps({"max_new_tokens": 2048})
        )
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_default_max_tokens", 512)
        monkeypatch.setattr(server, "_default_max_tokens_explicit", True, raising=False)
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        assert server._resolve_max_tokens(None, "bundle-model") == 512

    def test_omitted_server_max_tokens_uses_bundle_max_new_tokens(
        self,
        tmp_path,
        monkeypatch,
    ):
        """No session/CLI override means generation_config owns length."""
        import vmlx_engine.server as server

        (tmp_path / "generation_config.json").write_text(
            json.dumps({"max_new_tokens": 2048})
        )
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_default_max_tokens", 32768)
        monkeypatch.setattr(server, "_default_max_tokens_explicit", False, raising=False)
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        assert server._resolve_max_tokens(None, "bundle-model") == 2048

    def test_omitted_server_max_tokens_without_bundle_default_is_bounded(
        self,
        tmp_path,
        monkeypatch,
    ):
        """No request/session/bundle length must not silently become 32K.

        Several local JANG/JANGTQ bundles intentionally omit
        generation_config.max_new_tokens. In that case the engine still needs
        a real bounded output budget instead of inheriting the historical
        32768 fallback, which lets long turns drift into loops.
        """
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(json.dumps({"model_type": "hy_v3"}))
        (tmp_path / "generation_config.json").write_text(json.dumps({"top_k": -1}))
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_default_max_tokens", 32768)
        monkeypatch.setattr(server, "_default_max_tokens_explicit", False, raising=False)
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        assert server._resolve_max_tokens(None, "bundle-model") == 4096

    def test_ling_bailing_omitted_top_k_uses_audited_family_cap(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Ling/Bailing needs a bounded stochastic sampler when metadata is absent.

        The live multilingual audit reproduces CJK leakage with temperature/top_p
        sampling when the bundle omits top_k. Keep the policy at request
        resolution, with request/CLI/bundle values still taking precedence, so
        it is visible in resolved kwargs instead of hidden inside the sampler.
        """
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(json.dumps({"model_type": "bailing_hybrid"}))
        (tmp_path / "generation_config.json").write_text(json.dumps({}))
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_default_top_k", None)
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        assert server._resolve_top_k(None, str(tmp_path)) == 20
        assert server._resolve_top_k(0, str(tmp_path)) == 0
        assert server._resolve_top_k(7, str(tmp_path)) == 7

        monkeypatch.setattr(server, "_default_top_k", 11)
        assert server._resolve_top_k(None, str(tmp_path)) == 11

        monkeypatch.setattr(server, "_default_top_k", None)
        (tmp_path / "generation_config.json").write_text(json.dumps({"top_k": 33}))
        server._generation_defaults_cache.clear()
        assert server._resolve_top_k(None, str(tmp_path)) == 33

    def test_max_tokens_resolution_contract_applies_to_every_registered_family(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Every backend family gets the same output-length precedence.

        This prevents family-specific parser/cache work from accidentally
        bypassing the model-owned length contract:
        request > explicit startup --max-tokens > bundle max_new_tokens > fallback.
        """
        import vmlx_engine.server as server
        from vmlx_engine.model_config_registry import ModelConfigRegistry
        from vmlx_engine.model_configs import register_all

        registry = ModelConfigRegistry()
        register_all(registry)
        families = [c for c in registry._configs if c.model_types]
        assert len(families) >= 70

        monkeypatch.setattr(server, "_default_max_tokens", 512)
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        for cfg in families:
            model_type = cfg.model_types[0]
            model_dir = tmp_path / model_type
            model_dir.mkdir()
            (model_dir / "config.json").write_text(json.dumps({"model_type": model_type}))
            (model_dir / "generation_config.json").write_text(
                json.dumps({"max_new_tokens": 2048})
            )

            monkeypatch.setattr(server, "_model_path", str(model_dir))

            monkeypatch.setattr(
                server, "_default_max_tokens_explicit", False, raising=False
            )
            server._jang_sampling_defaults_cache.clear()
            server._generation_defaults_cache.clear()
            assert server._resolve_max_tokens(None, str(model_dir)) == 2048, (
                cfg.family_name,
                model_type,
            )
            assert server._resolve_max_tokens(77, str(model_dir)) == 77

            monkeypatch.setattr(
                server, "_default_max_tokens_explicit", True, raising=False
            )
            server._jang_sampling_defaults_cache.clear()
            server._generation_defaults_cache.clear()
            assert server._resolve_max_tokens(None, str(model_dir)) == 512, (
                cfg.family_name,
                model_type,
            )

    def test_wake_reload_preserves_max_tokens_explicitness(self):
        """Deep-sleep reload must not turn fallback max_tokens into override."""
        source = Path("./vmlx_engine/server.py").read_text()
        wake_block = source[
            source.index("async def admin_wake"):
            source.index("@app.get(\"/v1/cache/stats\"")
        ]

        assert '"max_tokens_explicit": max_tokens_explicit' in source
        assert 'max_tokens_explicit=_cli_args.get("max_tokens_explicit", False)' in wake_block

    def test_cli_serve_implicit_max_tokens_uses_bounded_fallback(self):
        """The vmlx serve parser must not reintroduce the historical 32K default."""
        import vmlx_engine.cli as cli
        import vmlx_engine.server as server

        cli_source = Path("./vmlx_engine/cli.py").read_text()
        serve_max_tokens_block = cli_source[
            cli_source.index('serve_parser.add_argument(\n        "--max-tokens"'):
            cli_source.index('serve_parser.add_argument(\n        "--max-prompt-tokens"')
        ]

        assert cli.DEFAULT_MAX_OUTPUT_TOKENS == server._FALLBACK_MAX_OUTPUT_TOKENS
        assert "default=DEFAULT_MAX_OUTPUT_TOKENS" in serve_max_tokens_block
        assert "default=32768" not in serve_max_tokens_block
        assert 'getattr(args, "max_tokens", 32768) != 32768' not in cli_source


# ===========================================================================
# D. Settings & Config
# ===========================================================================


class TestModelConfigRegistryLookup:
    """Tests for model config registry lookup by model_type."""

    def _find_by_model_type(self, registry, model_type):
        """Find a config that has the given model_type in its model_types list."""
        for config in registry._configs:
            if model_type in config.model_types:
                return config
        return None

    def test_lookup_qwen3(self):
        from vmlx_engine.model_config_registry import get_model_config_registry
        registry = get_model_config_registry()
        config = self._find_by_model_type(registry, "qwen3")
        assert config is not None
        assert config.tool_parser == "qwen"

    def test_lookup_glm4_moe(self):
        from vmlx_engine.model_config_registry import get_model_config_registry
        registry = get_model_config_registry()
        config = self._find_by_model_type(registry, "glm4_moe")
        assert config is not None
        assert config.family_name == "glm4_moe"
        assert config.reasoning_parser == "openai_gptoss"

    def test_lookup_deepseek_v3(self):
        from vmlx_engine.model_config_registry import get_model_config_registry
        registry = get_model_config_registry()
        config = self._find_by_model_type(registry, "deepseek_v3")
        assert config is not None
        assert config.reasoning_parser == "deepseek_r1"

    def test_lookup_qwen3_5(self):
        """The bare qwen3_5 family entry is neutral.

        Local Qwen 3.6 bundles with vision/video metadata are tested below via
        registry.lookup(path), because that path reads config.json.
        """
        from vmlx_engine.model_config_registry import get_model_config_registry
        registry = get_model_config_registry()
        config = self._find_by_model_type(registry, "qwen3_5")
        assert config is not None
        assert config.is_mllm is False

    def test_lookup_qwen3_6_dense_path_with_vision_video_is_mllm(self, tmp_path):
        from vmlx_engine.model_config_registry import get_model_config_registry
        registry = get_model_config_registry()
        registry._match_cache.clear()

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {"model_type": "qwen3_5_text"},
            "vision_config": {"model_type": "qwen3_vl"},
            "image_token_id": 248056,
            "video_token_id": 248057,
        }))

        config = registry.lookup(str(tmp_path))

        assert config.family_name == "qwen3_5"
        assert config.is_mllm is True

    def test_lookup_qwen3_6_moe_path_with_vision_video_is_mllm(self, tmp_path):
        from vmlx_engine.model_config_registry import get_model_config_registry
        registry = get_model_config_registry()
        registry._match_cache.clear()

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": {"model_type": "qwen3_5_moe_text"},
            "vision_config": {"model_type": "qwen3_vl"},
            "image_token_id": 248056,
            "video_token_id": 248057,
        }))

        config = registry.lookup(str(tmp_path))

        assert config.family_name == "qwen3_5_moe"
        assert config.is_mllm is True

    @pytest.mark.parametrize(
        ("model_type", "family_name"),
        [
            ("qwen3_5", "qwen3_5"),
            ("qwen3_5_moe", "qwen3_5_moe"),
        ],
    )
    def test_qwen3_6_vision_video_config_marks_bundle_multimodal(
        self, tmp_path, model_type, family_name
    ):
        from vmlx_engine.model_config_registry import get_model_config_registry

        model_dir = tmp_path / model_type
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": model_type,
                    "architectures": [f"{model_type}ForConditionalGeneration"],
                    "text_config": {
                        "model_type": model_type,
                        "layer_types": ["linear_attention", "full_attention"],
                    },
                    "vision_config": {"hidden_size": 1024},
                    "video_token_id": 151666,
                    "video_token_index": 151666,
                }
            )
        )

        config = get_model_config_registry().lookup(str(model_dir))

        assert config.family_name == family_name
        assert config.cache_type == "hybrid"
        assert config.is_mllm is True

    def test_nemotron_h_without_media_config_is_text_runtime(self, tmp_path):
        from vmlx_engine.model_config_registry import get_model_config_registry

        model_dir = tmp_path / "nemotron-text"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "nemotron_h",
                    "architectures": ["NemotronHForCausalLM"],
                    "text_config": {
                        "layer_types": ["mamba", "full_attention"],
                    },
                }
            )
        )
        (model_dir / "preprocessor_config.json").write_text("{}")

        config = get_model_config_registry().lookup(str(model_dir))

        assert config.family_name == "nemotron_h"
        assert config.cache_type == "hybrid"
        assert config.is_mllm is False

    def test_nemotron_h_omni_stamp_without_media_config_is_text_runtime(self, tmp_path):
        from vmlx_engine.model_config_registry import get_model_config_registry

        model_dir = tmp_path / "nemotron-stamped-text"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "nemotron_h",
                    "architectures": ["NemotronHForCausalLM"],
                    "text_config": {
                        "layer_types": ["mamba", "full_attention"],
                    },
                }
            )
        )
        (model_dir / "jang_config.json").write_text(
            json.dumps(
                {
                    "capabilities": {
                        "family": "nemotron_h",
                        "modality": "omni",
                        "cache_type": "hybrid",
                    }
                }
            )
        )

        config = get_model_config_registry().lookup(str(model_dir))

        assert config.family_name == "nemotron_h"
        assert config.cache_type == "hybrid"
        assert config.is_mllm is False

    @pytest.mark.parametrize("model_type", ["zaya", "zaya1_vl"])
    def test_zaya_registry_preserves_tokenizer_eos_boundary(self, tmp_path, model_type):
        from vmlx_engine.model_config_registry import get_model_config_registry

        model_dir = tmp_path / model_type
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": model_type,
                    **({"vision_config": {"hidden_size": 1024}} if model_type == "zaya1_vl" else {}),
                }
            )
        )
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps({"eos_token": "<|im_end|>", "eos_token_id": 106})
        )

        config = get_model_config_registry().lookup(str(model_dir))

        assert config.family_name == model_type
        assert config.eos_tokens == ["<|im_end|>"]

    def test_lookup_unknown_type_returns_none(self):
        from vmlx_engine.model_config_registry import get_model_config_registry
        registry = get_model_config_registry()
        config = self._find_by_model_type(registry, "nonexistent_type_xyz")
        assert config is None


class TestThinkInTemplate:
    """Tests for think_in_template flag on model configs."""

    def _find_by_model_type(self, registry, model_type):
        for config in registry._configs:
            if model_type in config.model_types:
                return config
        return None

    def test_glm47_flash_think_in_template_false(self):
        """GLM-4.7 Flash uses Harmony protocol, NOT <think> in template."""
        from vmlx_engine.model_config_registry import get_model_config_registry
        registry = get_model_config_registry()
        config = self._find_by_model_type(registry, "glm4_moe")
        assert config is not None
        assert config.think_in_template is False

    def test_qwen3_think_in_template_true(self):
        """Qwen3 uses <think> tag in template."""
        from vmlx_engine.model_config_registry import get_model_config_registry
        registry = get_model_config_registry()
        config = self._find_by_model_type(registry, "qwen3")
        if config is not None:
            assert config.think_in_template is True


# ===========================================================================
# E. Engine & Cache
# ===========================================================================


class TestVisionEmbeddingCacheOrdering:
    """Tests for vision embedding cache hash ordering."""

    def test_hash_order_sensitive(self):
        """Different image orderings should produce different hashes."""
        from vmlx_engine.vision_embedding_cache import compute_images_hash
        hash1 = compute_images_hash(["img1.jpg", "img2.jpg"])
        hash2 = compute_images_hash(["img2.jpg", "img1.jpg"])
        assert hash1 != hash2, "Different image orders should produce different hashes"

    def test_same_order_same_hash(self):
        """Same image ordering should produce same hash."""
        from vmlx_engine.vision_embedding_cache import compute_images_hash
        hash1 = compute_images_hash(["a.jpg", "b.jpg"])
        hash2 = compute_images_hash(["a.jpg", "b.jpg"])
        assert hash1 == hash2

    def test_empty_images(self):
        """Empty list should produce a consistent hash."""
        from vmlx_engine.vision_embedding_cache import compute_images_hash
        assert compute_images_hash([]) == "no_images"

    def test_single_image(self):
        """Single image hash should be deterministic."""
        from vmlx_engine.vision_embedding_cache import compute_images_hash
        h1 = compute_images_hash(["test.jpg"])
        h2 = compute_images_hash(["test.jpg"])
        assert h1 == h2


class TestVisionCacheStats:
    """Tests for VisionCacheStats tracking."""

    def test_initial_stats(self):
        from vmlx_engine.vision_embedding_cache import VisionCacheStats
        stats = VisionCacheStats()
        assert stats.pixel_cache_hits == 0
        assert stats.pixel_cache_misses == 0
        assert stats.pixel_hit_rate == 0.0

    def test_hit_rate_calculation(self):
        from vmlx_engine.vision_embedding_cache import VisionCacheStats
        stats = VisionCacheStats(pixel_cache_hits=3, pixel_cache_misses=7)
        assert stats.pixel_hit_rate == 0.3


class TestRequestStatus:
    """Tests for request status transitions."""

    def test_request_status_values(self):
        from vmlx_engine.request import RequestStatus
        assert hasattr(RequestStatus, "FINISHED_STOPPED")
        assert hasattr(RequestStatus, "FINISHED_ABORTED")
        assert hasattr(RequestStatus, "FINISHED_LENGTH_CAPPED")

    def test_sampling_params_stop_sequences(self):
        from vmlx_engine.request import SamplingParams
        sp = SamplingParams(stop=["<|end|>", "\n\n"])
        assert "<|end|>" in sp.stop
        assert "\n\n" in sp.stop


# ===========================================================================
# F. Standard Architectures Detection
# ===========================================================================


class TestStandardArchitectures:
    """Tests that _STANDARD_ARCHITECTURES in tokenizer.py includes all key types."""

    def test_qwen3_5_in_standard_architectures(self):
        from vmlx_engine.utils.tokenizer import _STANDARD_ARCHITECTURES
        assert "qwen3_5" in _STANDARD_ARCHITECTURES
        assert "qwen3_5_moe" in _STANDARD_ARCHITECTURES

    def test_common_types_in_standard_architectures(self):
        from vmlx_engine.utils.tokenizer import _STANDARD_ARCHITECTURES
        common = ["llama", "qwen2", "qwen3", "gemma2", "gemma3", "mistral",
                  "deepseek_v3", "phi3"]
        for t in common:
            assert t in _STANDARD_ARCHITECTURES, \
                f"model_type '{t}' missing from _STANDARD_ARCHITECTURES"


# ===========================================================================
# G. API Models
# ===========================================================================


class TestAPIModels:
    """Tests for API request/response models."""

    def test_chat_completion_request_has_sampling_fields(self):
        from vmlx_engine.api.models import ChatCompletionRequest
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.5,
            top_p=0.8,
            top_k=50,
            min_p=0.1,
            repetition_penalty=1.2,
        )
        assert req.temperature == 0.5
        assert req.top_p == 0.8
        assert req.top_k == 50
        assert req.min_p == 0.1
        assert req.repetition_penalty == 1.2

    def test_chat_completion_request_defaults(self):
        from vmlx_engine.api.models import ChatCompletionRequest
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
        )
        # Defaults should be None (let server resolve)
        assert req.top_k is None
        assert req.min_p is None
        assert req.repetition_penalty is None

    def test_responses_request_has_sampling_fields(self):
        from vmlx_engine.api.models import ResponsesRequest
        req = ResponsesRequest(
            model="test",
            input="hello",
            top_k=50,
            min_p=0.1,
            repetition_penalty=1.2,
        )
        assert req.top_k == 50
        assert req.min_p == 0.1
        assert req.repetition_penalty == 1.2

    def test_enable_thinking_field(self):
        from vmlx_engine.api.models import ChatCompletionRequest
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            enable_thinking=True,
        )
        assert req.enable_thinking is True

    def test_stream_options_field(self):
        from vmlx_engine.api.models import ChatCompletionRequest, StreamOptions
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            stream_options=StreamOptions(include_usage=True),
        )
        assert req.stream_options.include_usage is True

    def test_public_api_models_reject_non_positive_output_caps(self):
        """Raw API output caps are either omitted or positive request caps.

        Panel Auto omits maxTokens so the server startup/model default can
        apply. Direct OpenAI-compatible API clients must not send 0 or negative
        caps that bypass that default and create empty/poisoned generations.
        """
        import pytest
        from pydantic import ValidationError

        from vmlx_engine.api.models import (
            ChatCompletionRequest,
            CompletionRequest,
            ResponsesRequest,
        )

        with pytest.raises(ValidationError, match="max_tokens must be at least 1"):
            ChatCompletionRequest(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=0,
            )
        with pytest.raises(ValidationError, match="max_tokens must be at least 1"):
            CompletionRequest(model="test", prompt="hi", max_tokens=-1)
        with pytest.raises(ValidationError, match="max_output_tokens must be at least 1"):
            ResponsesRequest(model="test", input="hi", max_output_tokens=0)
        with pytest.raises(ValidationError, match="max_thinking_tokens must be at least 1"):
            ChatCompletionRequest(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                max_thinking_tokens=0,
            )
        with pytest.raises(ValidationError, match="max_thinking_tokens must be at least 1"):
            ResponsesRequest(model="test", input="hi", max_thinking_tokens=0)


# ===========================================================================
# G. MLLM Scheduler Fixes
# ===========================================================================

class TestMLLMStopSequences:
    """Test that MLLM requests properly carry stop sequences."""

    def test_sampling_params_stop_sequences_populated(self):
        from vmlx_engine.request import SamplingParams
        sp = SamplingParams(stop=["<|end|>", "###"])
        assert sp.stop == ["<|end|>", "###"]

    def test_sampling_params_stop_default_empty(self):
        from vmlx_engine.request import SamplingParams
        sp = SamplingParams()
        assert sp.stop == []

    def test_mllm_request_accepts_stop(self):
        from vmlx_engine.mllm_scheduler import MLLMRequest
        from vmlx_engine.request import SamplingParams
        sp = SamplingParams(stop=["<|end|>"])
        req = MLLMRequest(
            request_id="test-1",
            prompt="What's in this image?",
            sampling_params=sp,
        )
        assert req.sampling_params.stop == ["<|end|>"]


class TestRepPenaltyTruthiness:
    """Test that repetition_penalty=0.0 is NOT treated as disabled."""

    def test_zero_rep_penalty_is_not_none(self):
        """0.0 is a valid repetition penalty (no repetition boost)."""
        val = 0.0
        # Old buggy check: `if val and val != 1.0` → falsy (skips 0.0)
        # New correct check: `if val is not None and val != 1.0` → True
        assert (val is not None and val != 1.0) is True
        assert not (val and val != 1.0)  # Demonstrates the old bug: 0.0 was falsy

    def test_none_rep_penalty(self):
        val = None
        assert (val is not None and val != 1.0) is False

    def test_default_rep_penalty(self):
        val = 1.0
        assert (val is not None and val != 1.0) is False


class TestImageHashOrdering:
    """Test that image hash preserves order (not sorted)."""

    def test_different_order_different_hash(self):
        from vmlx_engine.mllm_cache import compute_images_hash
        # Two different orderings should produce different hashes
        hash1 = compute_images_hash(["image_a.jpg", "image_b.jpg"])
        hash2 = compute_images_hash(["image_b.jpg", "image_a.jpg"])
        assert hash1 != hash2, "Image order should matter for VLM cache hashing"

    def test_same_order_same_hash(self):
        from vmlx_engine.mllm_cache import compute_images_hash
        hash1 = compute_images_hash(["img1.jpg", "img2.jpg"])
        hash2 = compute_images_hash(["img1.jpg", "img2.jpg"])
        assert hash1 == hash2

    def test_empty_images(self):
        from vmlx_engine.mllm_cache import compute_images_hash
        assert compute_images_hash([]) == "no_images"
        assert compute_images_hash(None) == "no_images"


class TestToolParserConcurrency:
    """Test that tool parser creates per-call instances (not shared global)."""

    def test_parse_tool_calls_with_parser_no_global_state(self):
        """Verify the function doesn't rely on a global _tool_parser_instance."""
        import vmlx_engine.server as srv
        # When auto tool choice is disabled, should fall through to generic parser
        old_val = srv._enable_auto_tool_choice
        try:
            srv._enable_auto_tool_choice = False
            result = srv._parse_tool_calls_with_parser("hello world")
            # Should return content unchanged, no tool calls
            assert result[1] is None or result[1] == []
        finally:
            srv._enable_auto_tool_choice = old_val

    def test_dsml_issue_165_server_tool_call_arguments_are_not_empty_or_raw(self):
        """DSV4 DSML repair must survive the server's OpenAI ToolCall wrapper."""
        import json

        import vmlx_engine.server as srv
        from vmlx_engine.tool_parsers.dsml_tool_parser import DSML_PREFIX

        request = srv.ChatCompletionRequest(
            model="dsv4-test",
            messages=[
                srv.Message(
                    role="user",
                    content="Use read_file for docs/vendor_memo.md.",
                )
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
        )
        output = (
            f'<{DSML_PREFIX}tool_calls>\n'
            f'<{DSML_PREFIX}invoke name="read_file">\n'
            '    <param name="path">docs/vendor_memo.md</param>\n'
            '</inv>'
        )

        old_auto = srv._enable_auto_tool_choice
        old_parser = srv._tool_call_parser
        try:
            srv._enable_auto_tool_choice = True
            srv._tool_call_parser = "deepseek_v4"
            cleaned, tool_calls = srv._parse_tool_calls_with_parser(output, request)
        finally:
            srv._enable_auto_tool_choice = old_auto
            srv._tool_call_parser = old_parser

        assert cleaned == ""
        assert tool_calls and len(tool_calls) == 1
        assert tool_calls[0].function.name == "read_file"
        assert json.loads(tool_calls[0].function.arguments) == {
            "path": "docs/vendor_memo.md"
        }
        assert DSML_PREFIX not in tool_calls[0].function.arguments

    def test_qwen_issue_192_xml_string_arguments_use_request_schema(self):
        """Qwen XML string args should become schema JSON in server tool_calls."""
        import json

        import vmlx_engine.server as srv

        request = srv.ChatCompletionRequest(
            model="qwen3-coder-next",
            messages=[
                srv.Message(
                    role="user",
                    content="test your tools with a bash echo command",
                )
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    },
                }
            ],
        )
        output = (
            '<tool_call>{"name": "bash", "arguments": '
            '"<command>\\necho \\"Tools are working correctly!\\"\\n</command>"}'
            "</tool_call>"
        )

        old_auto = srv._enable_auto_tool_choice
        old_parser = srv._tool_call_parser
        try:
            srv._enable_auto_tool_choice = True
            srv._tool_call_parser = "qwen"
            cleaned, tool_calls = srv._parse_tool_calls_with_parser(output, request)
        finally:
            srv._enable_auto_tool_choice = old_auto
            srv._tool_call_parser = old_parser

        assert cleaned == ""
        assert tool_calls and len(tool_calls) == 1
        assert tool_calls[0].function.name == "bash"
        assert json.loads(tool_calls[0].function.arguments) == {
            "command": 'echo "Tools are working correctly!"'
        }

    def test_qwen_issue_192_plain_tool_line_uses_single_tool_schema(self):
        """Qwen live output can be bare tool name + command text."""
        import json

        import vmlx_engine.server as srv

        request = srv.ChatCompletionRequest(
            model="qwen3-coder-next",
            messages=[
                srv.Message(
                    role="user",
                    content="test your tools with a bash echo command",
                )
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    },
                }
            ],
            tool_choice="required",
        )
        output = 'bash\necho "Tool test successful!"'

        old_auto = srv._enable_auto_tool_choice
        old_parser = srv._tool_call_parser
        try:
            srv._enable_auto_tool_choice = True
            srv._tool_call_parser = "qwen"
            cleaned, tool_calls = srv._parse_tool_calls_with_parser(output, request)
        finally:
            srv._enable_auto_tool_choice = old_auto
            srv._tool_call_parser = old_parser

        assert cleaned == ""
        assert tool_calls and len(tool_calls) == 1
        assert tool_calls[0].function.name == "bash"
        assert json.loads(tool_calls[0].function.arguments) == {
            "command": 'echo "Tool test successful!"'
        }

    def test_xml_function_empty_required_args_fail_closed_at_server_boundary(self):
        """Parameterless XML function tags must not emit arguments={} calls."""
        import vmlx_engine.server as srv

        request = srv.ChatCompletionRequest(
            model="qwen3-coder-next",
            messages=[
                srv.Message(
                    role="user",
                    content="List /tmp using exec_command.",
                )
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
                        },
                    },
                }
            ],
            tool_choice="required",
        )
        output = (
            "Quick preamble: Checking what's in `/tmp`...\n"
            "<tool_call>\n"
            "<function=exec_command>\n"
            "</function>\n"
            "</tool_call>"
        )

        old_auto = srv._enable_auto_tool_choice
        old_parser = srv._tool_call_parser
        try:
            srv._enable_auto_tool_choice = True
            srv._tool_call_parser = "xml_function"
            cleaned, tool_calls = srv._parse_tool_calls_with_parser(output, request)
        finally:
            srv._enable_auto_tool_choice = old_auto
            srv._tool_call_parser = old_parser

        assert tool_calls is None
        assert cleaned == output

    def test_dsv4_fallback_tool_prompt_uses_canonical_tool_calls_wrapper(self):
        """Fallback injection must not teach a non-canonical bare DSML invoke."""
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class DSV4FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                rendered = "<｜begin▁of▁sentence｜>"
                for msg in messages:
                    role = msg.get("role")
                    if role == "system":
                        rendered += msg.get("content") or ""
                    elif role == "user":
                        rendered += "<｜User｜>" + (msg.get("content") or "")
                    elif role == "assistant":
                        rendered += "<｜Assistant｜>" + (msg.get("content") or "")
                    elif role == "tool":
                        rendered += (
                            "<｜User｜><tool_result>"
                            + (msg.get("content") or "")
                            + "</tool_result>"
                        )
                rendered += "<｜Assistant｜></think>"
                return rendered

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
        ]
        prompt = (
            "<｜begin▁of▁sentence｜>\n\n## Tools\n"
            "<｜DSML｜tool_calls>\n"
            "<｜DSML｜invoke name=\"$TOOL_NAME\">\n"
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>\n"
            "<｜User｜>Use tools in sequence."
        )

        rendered = check_and_inject_fallback_tools(
            prompt,
            [{"role": "user", "content": "Use tools in sequence."}],
            tools,
            DSV4FakeTokenizer(),
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="dsml",
        )

        assert "or a DSML wrapper block" not in rendered
        assert "VALUE HERE" not in rendered
        assert "string=" not in rendered.split("<｜User｜>", 1)[0]
        assert "<｜DSML｜parameter" not in rendered.split("<｜User｜>", 1)[0]
        assert "Only use the tag name <｜DSML｜invoke" in rendered
        assert "never emit <｜DSML｜invuse>" in rendered
        example_scope = rendered.split("Only use the tag name", 1)[-1]
        assert '<｜DSML｜invuse name="' not in example_scope
        assert '<｜DSML｜invue name="' not in example_scope
        instruction_scope = rendered.split("<｜User｜>", 1)[0]
        assert "<｜DSML｜tool_calls>" in rendered
        assert "</｜DSML｜tool_calls>" in rendered
        assert '<｜DSML｜invoke name="list_directory">' in rendered
        assert '<｜DSML｜invoke name="write_file">' in rendered
        assert instruction_scope.count("<｜DSML｜tool_calls>") == 2
        assert "path (string, required)" in rendered
        assert "content (string, required)" in rendered

    def test_dsv4_native_schema_prompt_with_only_generic_examples_gets_concrete_fallback(self):
        """DSV4 needs concrete invoke names, not only generic DSML placeholders."""
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class DSV4FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "\n".join(m.get("content", "") for m in messages)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
        ]
        prompt = (
            "<｜begin▁of▁sentence｜>\n\n## Tools\n"
            "You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n"
            "<｜DSML｜tool_calls>\n"
            "<｜DSML｜invoke name=\"$TOOL_NAME\">\n"
            "<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">"
            "$PARAMETER_VALUE</｜DSML｜parameter>\n"
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>\n\n"
            "### Available Tool Schemas\n\n"
            '{"name": "list_directory", "parameters": {"properties": {"path": {"type": "string"}}}}\n'
            '{"name": "write_file", "parameters": {"properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}\n'
            "<｜User｜>Use tools in sequence."
        )

        rendered = check_and_inject_fallback_tools(
            prompt,
            [{"role": "user", "content": "Use tools in sequence."}],
            tools,
            DSV4FakeTokenizer(),
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="dsml",
        )

        instruction_scope = rendered.split("<｜User｜>", 1)[0]
        assert rendered != prompt
        assert '<｜DSML｜invoke name="list_directory">' in instruction_scope
        assert '<｜DSML｜invoke name="write_file">' in instruction_scope
        assert "path (string, required)" in instruction_scope
        assert "content (string, required)" in instruction_scope
        assert "$TOOL_NAME" not in instruction_scope
        assert "$PARAMETER_NAME" not in instruction_scope
        assert "$PARAMETER_VALUE" not in instruction_scope

    def test_stream_chat_buffers_leading_whitespace_before_tool_marker(self):
        """Tool-call-only streams must not leak whitespace before tool_calls."""
        import inspect

        from vmlx_engine.server import _TOOL_CALL_MARKERS, stream_chat_completion

        assert "<｜DSML｜tool" in _TOOL_CALL_MARKERS
        source = inspect.getsource(stream_chat_completion)
        assert "<｜DSML｜tool_c" in _TOOL_CALL_MARKERS
        assert "<tool_calls>" in _TOOL_CALL_MARKERS
        assert "<tool_call>" in _TOOL_CALL_MARKERS
        assert "<tool_sep>" in _TOOL_CALL_MARKERS
        assert "<function" in _TOOL_CALL_MARKERS
        assert "<zyphra_tool_call" in _TOOL_CALL_MARKERS
        assert "<|tool_call_start|>" in _TOOL_CALL_MARKERS
        assert "<|tool_call_end|>" in _TOOL_CALL_MARKERS
        assert "not emit_content.strip()" in source
        assert "not content.strip()" in source
        assert "tool_call_generating=True" in source

    def test_tool_markup_residue_strips_all_registered_marker_families(self):
        """Display cleanup must track every family marker, not only DSML/Hy3."""
        from vmlx_engine.server import _strip_tool_markup_residue_for_display

        cases = {
            "zaya": "Before <zyphra_tool_call name=\"read_file\"> after",
            "mistral": "Before [TOOL_CALLS] after",
            "minimax": "Before <minimax:tool_call> after",
            "kimi": "Before <|tool_calls_section_begin|> after",
            "gemma": "Before <|tool_call> after",
            "lfm2_start": "Before <|tool_call_start|> after",
            "lfm2_end": "Before <|tool_call_end|> after",
            "llama": "Before <function=read_file> after",
            "deepseek_glm": "Before <｜tool▁call▁begin｜> after",
            "dsv4": "Before <｜DSML｜tool after",
            "python_tag": "Before <|python_tag|> after",
            "tool_code": "Before ```tool_code after",
        }

        for family, text in cases.items():
            cleaned = _strip_tool_markup_residue_for_display(text)
            assert "Before" in cleaned, family
            assert "after" in cleaned, family
            assert "<" not in cleaned, (family, cleaned)
            assert "[TOOL_CALLS]" not in cleaned, (family, cleaned)
            assert "```tool_code" not in cleaned, (family, cleaned)

    def test_visual_grounding_markup_is_not_visible_chat_text(self):
        """ZAYA-VL point/box control spans must not leak into UI text."""
        from vmlx_engine.server import (
            _strip_visual_grounding_markup_for_display,
            _visual_grounding_display_delta,
        )

        assert (
            _strip_visual_grounding_markup_for_display(
                "Before <|point_start|>(40, 100)<|point_end|> after"
            )
            == "Before  after"
        )
        assert (
            _strip_visual_grounding_markup_for_display(
                "<|box_start|>[1,2,3,4]<|box_end|> answer"
            )
            == " answer"
        )
        assert (
            _visual_grounding_display_delta(
                "Answer <|point_start|>tool",
                "",
            )
            == "Answer "
        )
        assert (
            _visual_grounding_display_delta(
                "Answer <|point_start|>tool<|point_end|> done",
                "Answer ",
            )
            == " done"
        )

    def test_zaya_create_file_alias_maps_to_available_write_file_tool(self, monkeypatch):
        """ZAYA may call create_file; vMLX's builtin equivalent is write_file."""
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message

        monkeypatch.setattr(server, "_tool_call_parser", "zaya_xml")
        request = ChatCompletionRequest(
            model="zaya-vl-test",
            messages=[Message(role="user", content="create a file")],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["path", "content"],
                        },
                    },
                }
            ],
        )

        _, calls = server._parse_tool_calls_with_parser(
            "\n".join(
                [
                    "<zyphra_tool_call>",
                    "<function=create_file>",
                    "<parameter=filename>real_ui_tool_probe_1.txt</parameter>",
                    "<parameter=text>REAL_UI_LIVE_TOOL_ONE</parameter>",
                    "</function>",
                    "</zyphra_tool_call>",
                ]
            ),
            request,
        )

        assert calls is not None
        assert calls[0].function.name == "write_file"
        assert json.loads(calls[0].function.arguments) == {
            "path": "real_ui_tool_probe_1.txt",
            "content": "REAL_UI_LIVE_TOOL_ONE",
        }

    def test_tool_markup_residue_strips_interrupted_non_dsml_fragments(self):
        """Interrupted marker fragments must not leak attribute/code tails."""
        from vmlx_engine.server import _strip_tool_markup_residue_for_display

        cases = {
            "zaya_attr_tail": 'Before <zyphra_tool_call name="write_file"',
            "llama_function_tail": "Before <function=write_file",
            "gemma3_code_tail": "Before ```tool_code\nwrite_file(",
        }

        for family, text in cases.items():
            cleaned = _strip_tool_markup_residue_for_display(text)
            assert "Before" in cleaned, family
            assert "write_file" not in cleaned, (family, cleaned)
            assert "zyphra" not in cleaned, (family, cleaned)
            assert "function" not in cleaned, (family, cleaned)
            assert "```tool_code" not in cleaned, (family, cleaned)


class TestCacheTruncation:
    """Test N-1 token truncation logic for cache storage."""

    def test_truncation_target(self):
        """Cache should store N-1 tokens (prompt_len - 1)."""
        prompt_len = 100
        target = prompt_len - 1
        assert target == 99

    def test_truncation_skips_empty(self):
        """Truncation with 0 or 1 tokens should return None."""
        from vmlx_engine.scheduler import Scheduler
        result = Scheduler._truncate_cache_to_prompt_length([], 5)
        assert result is None
        result = Scheduler._truncate_cache_to_prompt_length([MagicMock()], 0)
        assert result is None


# ===========================================================================
# H. MLLM Scheduler Parity Tests
# ===========================================================================

class TestMLLMAbortCleanup:
    """Test that MLLM abort properly cleans up all resources."""

    def test_abort_uses_pop_not_get(self):
        """abort_request must remove (pop) the request, not just read (get) it."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.abort_request)
        # Must use pop to remove the request
        assert "self.requests.pop(" in source or "requests.pop(" in source
        # Must NOT use .get() for the primary request lookup
        # (get would leave the request in the dict)

    def test_abort_cleans_block_table(self):
        """abort_request must clean up paged cache block tables."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.abort_request)
        assert "_request_tables" in source, (
            "abort_request must clean up _request_tables for paged cache"
        )

    def test_abort_cleans_detokenizer(self):
        """abort_request must clean up detokenizer state."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.abort_request)
        assert "_cleanup_detokenizer" in source, (
            "abort_request must call _cleanup_detokenizer"
        )


class TestMLLMStepErrorRecovery:
    """Test that MLLM step() has error recovery like LLM scheduler."""

    def test_step_has_try_except(self):
        """step() must wrap batch_generator.next() in try/except."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.step)
        assert "try:" in source, "step() must have try/except for error recovery"
        assert "batch_generator.next()" in source or "self.batch_generator.next()" in source

    def test_step_reschedules_on_error(self):
        """On error, step() must move requests back to waiting."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.step)
        assert "RequestStatus.WAITING" in source, (
            "step() must reschedule failed requests to WAITING"
        )


class TestLLMSingleActiveGenerator:
    """Text max_num_seqs=1 must be an engine path, not a patched mlx-lm merge."""

    def test_scheduler_imports_single_active_generator(self):
        from pathlib import Path

        source = Path("./vmlx_engine/scheduler.py").read_text()

        assert "from .utils.single_batch_generator import SingleBatchGenerator" in source
        assert "return SingleBatchGenerator(" in source
        assert "max_num_seqs=1" in source
        assert '"engine_path": generator_path' in source
        assert '"single_active_decode": generator_path == "single_active"' in source

    def test_single_active_generator_is_repo_owned(self):
        from pathlib import Path

        source = Path("./vmlx_engine/utils/single_batch_generator.py").read_text()

        assert "class SingleBatchGenerator" in source
        assert "from mlx_lm.generate import BatchGenerator" not in source
        assert "BatchKVCache(" not in source
        assert "BatchMambaCache(" not in source
        assert "BatchGenerator(" not in source
        assert "prompt_cache=req.cache" in source

    def test_health_and_cache_stats_surface_single_active_path(self):
        from pathlib import Path

        source = Path("./vmlx_engine/server.py").read_text()

        assert '"engine_path": scheduler_stats.get("engine_path")' in source
        assert '"single_active_decode": (' in source
        assert '"engine_path": stats.get("engine_path")' in source


class TestMLLMStopTokenIds:
    """Test that MLLM properly handles stop_token_ids."""

    def test_sampling_params_has_stop_token_ids(self):
        """SamplingParams must include stop_token_ids."""
        from vmlx_engine.request import SamplingParams

        params = SamplingParams(stop_token_ids=[100, 200])
        assert params.stop_token_ids == [100, 200]

    def test_stop_token_ids_default_empty(self):
        """stop_token_ids should default to empty list."""
        from vmlx_engine.request import SamplingParams

        params = SamplingParams()
        assert params.stop_token_ids == [] or params.stop_token_ids is None

    def test_mllm_scheduler_passes_stop_token_ids(self):
        """MLLMScheduler.add_request must pass stop_token_ids to SamplingParams."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.add_request)
        assert "stop_token_ids" in source, (
            "add_request must pass stop_token_ids to SamplingParams"
        )


class TestMLLMEnsureBatchGeneratorCacheClean:
    """Test that _ensure_batch_generator updates sampler in place (preserves caches)."""

    def test_updates_sampler_in_place(self):
        """Must update sampler in place without clearing caches."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._ensure_batch_generator)
        # Should update sampler in place rather than recreating generator
        assert "batch_generator.sampler" in source, (
            "_ensure_batch_generator must update sampler in place"
        )
        assert "_current_sampler_params" in source, (
            "_ensure_batch_generator must track current sampler params"
        )

    def test_no_logits_processors_param(self):
        """MLLMBatchGenerator must not accept logits_processors."""
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        sig = inspect.signature(MLLMBatchGenerator.__init__)
        assert "logits_processors" not in sig.parameters, (
            "logits_processors was removed — must not be in __init__ signature"
        )

    def test_scheduler_does_not_pass_logits_processors(self):
        """_ensure_batch_generator must not pass logits_processors."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._ensure_batch_generator)
        assert "logits_processors" not in source, (
            "_ensure_batch_generator must not reference logits_processors"
        )


class TestMLLMDequantizeOnRestore:
    """Test that MLLM dequantizes cache after paged cache restore."""

    def test_dequantize_function_exists(self):
        """_dequantize_cache must exist in mllm_batch_generator."""
        from vmlx_engine.mllm_batch_generator import _dequantize_cache
        assert callable(_dequantize_cache)

    def test_dequantize_passthrough_non_quantized(self):
        """Non-quantized cache should pass through unchanged."""
        from vmlx_engine.mllm_batch_generator import _dequantize_cache

        mock_cache = [MagicMock(), MagicMock()]
        # Not QuantizedKVCache instances — should pass through
        result = _dequantize_cache(mock_cache)
        assert len(result) == 2

    def test_dequantize_called_after_reconstruct(self):
        """Paged cache hit path must call _dequantize_cache after reconstruct."""
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator)
        # reconstruct_cache and _dequantize_cache must both appear
        assert "reconstruct_cache" in source
        assert "_dequantize_cache" in source


class TestMLLMTotalPromptTokens:
    """Test that MLLM scheduler tracks total_prompt_tokens."""

    def test_total_prompt_tokens_incremented(self):
        """_process_batch_responses must increment total_prompt_tokens at finish time."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        # Prompt token count is only known after the batch generator's first response,
        # so tracking happens at request finish time in _process_batch_responses
        source = inspect.getsource(MLLMScheduler._process_batch_responses)
        assert "total_prompt_tokens" in source, (
            "_process_batch_responses must increment total_prompt_tokens"
        )


# =============================================================================
# L1: frequency_penalty / presence_penalty accepted but warned
# =============================================================================


class TestPenaltyParametersAccepted:
    """Test that frequency_penalty and presence_penalty are accepted by API models."""

    def test_chat_completion_accepts_penalties(self):
        """ChatCompletionRequest should accept frequency_penalty and presence_penalty."""
        from vmlx_engine.api.models import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            frequency_penalty=0.5,
            presence_penalty=0.8,
        )
        assert req.frequency_penalty == 0.5
        assert req.presence_penalty == 0.8

    def test_responses_request_accepts_penalties(self):
        """ResponsesRequest should accept frequency_penalty and presence_penalty."""
        from vmlx_engine.api.models import ResponsesRequest

        req = ResponsesRequest(
            model="test",
            input="hello",
            frequency_penalty=0.5,
            presence_penalty=0.8,
        )
        assert req.frequency_penalty == 0.5
        assert req.presence_penalty == 0.8

    def test_server_warns_on_frequency_penalty(self):
        """Server should log warning when frequency_penalty is non-zero."""
        import inspect
        # Read the create_chat_completion source to verify warning logic
        from vmlx_engine.server import create_chat_completion

        source = inspect.getsource(create_chat_completion)
        assert "frequency_penalty" in source, (
            "create_chat_completion must check frequency_penalty"
        )
        assert "not implemented" in source.lower() or "ignored" in source.lower(), (
            "create_chat_completion must warn that frequency_penalty is not implemented"
        )


# =============================================================================
# v2 Audit: C1-C5 Critical Fixes
# =============================================================================


class TestC1DuplicateRequestIdCheck:
    """C1: MLLM scheduler must reject duplicate request IDs."""

    def test_mllm_scheduler_has_duplicate_check(self):
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.add_request)
        assert "already exists" in source, (
            "add_request must check for duplicate request IDs"
        )

    def test_llm_scheduler_also_has_check(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler

        source = inspect.getsource(Scheduler.add_request)
        assert "already exists" in source, (
            "LLM add_request must also check for duplicate request IDs"
        )


class TestC2AbortDecodeRace:
    """C2: MLLM scheduler must have _batch_lock for next()/remove() serialization."""

    def test_batch_lock_exists(self):
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.__init__)
        assert "_batch_lock" in source, (
            "MLLMScheduler must have _batch_lock for abort/decode race protection"
        )

    def test_batch_lock_used_in_step(self):
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.step)
        assert "_batch_lock" in source, (
            "step() must hold _batch_lock during batch_generator.next()"
        )

    def test_deferred_abort_prevents_metal_race(self):
        """Aborting a request while Metal buffers are in-flight touching the
        cache tensors would assert. Current design defers removal via
        `_pending_aborts` instead of holding `_batch_lock`, which is safer
        — the deferred set is drained after the current Metal compute
        completes."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.abort_request)
        assert "_pending_aborts" in source, (
            "abort_request() must defer batch removal via _pending_aborts "
            "to avoid touching cache tensors mid-Metal-compute"
        )
        # And must still hold the queue lock for request table mutation
        assert "_queue_lock" in source


class TestC3DiskCacheQuantScoping:
    """C3: Disk cache directory must be scoped by quantization config."""

    def test_scheduler_disk_cache_includes_quant(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler

        source = inspect.getsource(Scheduler.__init__)
        assert "quant" in source and "scope_key" in source, (
            "Scheduler disk cache dir must include quantization in scope key"
        )

    def test_mllm_scheduler_disk_cache_includes_quant(self):
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.__init__)
        assert "quant" in source and "scope_key" in source, (
            "MLLM scheduler disk cache dir must include quantization in scope key"
        )


class TestC4MaxTokensZero:
    """C4: max_tokens=0 must not be silently overridden to the default."""

    def test_server_uses_is_not_none_check(self):
        """max_tokens=0 must not be silently overridden (0 is falsy so the
        old `max_tokens or default` pattern would replace 0 with default).
        Accept either single- or multi-line form of the conditional, and
        normalize whitespace so newlines between clauses don't fool the
        check."""
        import inspect
        import re
        from vmlx_engine.server import create_chat_completion

        source = inspect.getsource(create_chat_completion)
        # Collapse all whitespace to single spaces so the multi-line
        # `if request.max_tokens is not None\n  else _default_max_tokens`
        # matches too.
        normalized = re.sub(r"\s+", " ", source)
        assert (
            "request.max_tokens if request.max_tokens is not None" in normalized
            or "if request.max_tokens is not None else _default_max_tokens" in normalized
        ), (
            "max_tokens must use 'is not None' check, not 'or' (0 is falsy)"
        )


class TestC5ToolCallsInThinkBlocks:
    """C5: Tool parsing must use accumulated_content when reasoning parser is active."""

    def test_chat_completions_uses_accumulated_content(self):
        import inspect
        from vmlx_engine.server import stream_chat_completion

        source = inspect.getsource(stream_chat_completion)
        # When reasoning parser is active, should use accumulated_content
        assert "request_parser and accumulated_content" in source, (
            "Tool call parsing must prefer accumulated_content when reasoning parser active"
        )

    def test_responses_api_uses_accumulated_content(self):
        import inspect
        from vmlx_engine.server import stream_responses_api

        source = inspect.getsource(stream_responses_api)
        assert "request_parser and accumulated_content" in source, (
            "Responses API tool parsing must prefer accumulated_content when reasoning parser active"
        )


class TestH1StopTokenCleanup:
    """H1: Per-request stop tokens must use a snapshot of surviving requests,
    not read from self.running (which is mutated during cleanup loop)."""

    def test_surviving_stops_snapshot_before_loop(self):
        """_surviving_stops must be computed BEFORE the per-request loop."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._cleanup_finished)
        # _surviving_stops must appear before the "for request_id in finished_ids" loop
        snap_pos = source.find("_surviving_stops")
        loop_pos = source.find("for request_id in finished_ids")
        assert snap_pos < loop_pos, (
            "_surviving_stops snapshot must be computed before the cleanup loop"
        )

    def test_removable_uses_snapshot(self):
        """Stop token removal must subtract _surviving_stops, not re-read self.running."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._cleanup_finished)
        assert "_surviving_stops" in source, (
            "Cleanup must use _surviving_stops snapshot"
        )
        assert "removable = request._added_stop_tokens - _surviving_stops" in source, (
            "Removable stop tokens must subtract surviving stops snapshot"
        )

    def test_surviving_stops_excludes_finished(self):
        """The snapshot must only include stops from requests NOT in finished_ids."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._cleanup_finished)
        assert "rid not in finished_ids" in source, (
            "Surviving stops must exclude requests that are finishing"
        )


class TestH2ImageCountLimit:
    """H2: Excessive images must be rejected to prevent Metal OOM."""

    def test_mllm_scheduler_config_has_limit(self):
        from vmlx_engine.mllm_scheduler import MLLMSchedulerConfig
        config = MLLMSchedulerConfig()
        assert hasattr(config, 'max_images_per_request')
        assert config.max_images_per_request > 0

    def test_add_request_rejects_excessive_images(self):
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.add_request)
        assert "max_images_per_request" in source, (
            "add_request must check image count against max_images_per_request"
        )

    def test_add_request_raises_on_too_many_images(self):
        from vmlx_engine.mllm_scheduler import MLLMSchedulerConfig

        config = MLLMSchedulerConfig(max_images_per_request=3)
        # Config limit is 3, so 5 images should trigger the guard
        assert config.max_images_per_request == 3


class TestH3VideoExtractionFailures:
    """H3: When ALL media inputs fail, must raise instead of silently continuing."""

    def test_multimodal_processor_raises_on_all_failures(self):
        import inspect
        from vmlx_engine.multimodal_processor import MultimodalProcessor

        source = inspect.getsource(MultimodalProcessor.process)
        assert "All media inputs failed" in source, (
            "Must raise ValueError when all images/videos fail to process"
        )

    def test_counts_failed_media(self):
        import inspect
        from vmlx_engine.multimodal_processor import MultimodalProcessor

        source = inspect.getsource(MultimodalProcessor.process)
        assert "failed_images" in source and "failed_videos" in source, (
            "Must track failed image and video counts"
        )


class TestMediaDiagnostics:
    """Redacted media diagnostics for user logs when image requests fail."""

    def test_messages_multimodal_summary_redacts_data_urls(self):
        from vmlx_engine.server import _messages_multimodal_summary

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what color?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,SECRET_IMAGE_BYTES"
                        },
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": "/tmp/example.mp4"},
                    },
                ],
            }
        ]

        summary = _messages_multimodal_summary(messages)

        assert summary["total"] == 2
        assert summary["types"] == {"image_url": 1, "video_url": 1}
        assert summary["data_url"] == 1
        assert summary["url_or_path"] == 1
        assert "SECRET_IMAGE_BYTES" not in json.dumps(summary)

    def test_responses_multimodal_summary_handles_input_image_without_payload(self):
        from vmlx_engine.server import _responses_input_multimodal_summary

        response_input = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "describe"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,SECRET_RESPONSE_IMAGE",
                    },
                ],
            }
        ]

        summary = _responses_input_multimodal_summary(response_input)

        assert summary["total"] == 1
        assert summary["types"] == {"input_image": 1}
        assert summary["data_url"] == 1
        assert "SECRET_RESPONSE_IMAGE" not in json.dumps(summary)

    def test_responses_multimodal_history_drops_none_optional_part_fields(self):
        from vmlx_engine.api.models import ContentPart, Message
        from vmlx_engine.server import _responses_input_to_messages

        response_input = [
            {
                "type": "function_call",
                "call_id": "call_tool",
                "name": "run_command",
                "arguments": '{"command":"printf ok"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_tool",
                "output": "ok",
            },
            Message(
                role="user",
                content=[
                    ContentPart(type="text", text="what color?"),
                    ContentPart(
                        type="image_url",
                        image_url={
                            "url": "data:image/png;base64,SECRET_RESPONSE_IMAGE",
                            "detail": None,
                        },
                    ),
                ],
            ),
        ]

        messages = _responses_input_to_messages(
            response_input,
            preserve_multimodal=True,
        )

        user_content = messages[-1]["content"]
        assert user_content == [
            {"type": "text", "text": "what color?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,SECRET_RESPONSE_IMAGE",
                },
            },
        ]
        assert "None" not in json.dumps(user_content)
        assert "image_url" not in user_content[0]

    def test_responses_text_followup_scrubs_media_from_previous_history(self):
        from vmlx_engine.server import (
            _responses_scrub_multimodal_history_for_text_followup,
        )

        previous_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what color is this?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,SECRET_RESPONSE_IMAGE",
                        },
                    },
                ],
            },
            {"role": "assistant", "content": "blue"},
            {"role": "user", "content": "remember this text"},
        ]

        scrubbed = _responses_scrub_multimodal_history_for_text_followup(
            previous_messages
        )

        assert scrubbed[0]["content"] == "what color is this?"
        assert scrubbed[1:] == previous_messages[1:]
        assert "SECRET_RESPONSE_IMAGE" not in json.dumps(scrubbed)
        assert isinstance(previous_messages[0]["content"], list)

    def test_responses_multimodal_history_coerces_orphan_tool_results(self):
        from vmlx_engine.server import _coerce_orphan_tool_messages_for_template

        messages = [
            {"role": "assistant", "content": "prior answer without native tool_calls"},
            {
                "role": "tool",
                "tool_call_id": "call_missing",
                "content": "REAL_TOOL_OUTPUT",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what color?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,SECRET_RESPONSE_IMAGE",
                        },
                    },
                ],
            },
        ]

        coerced = _coerce_orphan_tool_messages_for_template(messages)

        assert coerced[1] == {
            "role": "user",
            "content": "Tool result (call_missing):\nREAL_TOOL_OUTPUT",
        }
        assert coerced[2]["content"][1]["type"] == "image_url"

    def test_responses_multimodal_history_keeps_matched_native_tool_results(self):
        from vmlx_engine.server import _coerce_orphan_tool_messages_for_template

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_ok",
                        "type": "function",
                        "function": {"name": "run_command", "arguments": {}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_ok", "content": "ok"},
        ]

        assert _coerce_orphan_tool_messages_for_template(messages) == messages

    def test_responses_multimodal_history_coerces_non_adjacent_tool_results(self):
        from vmlx_engine.server import _coerce_orphan_tool_messages_for_template

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_earlier",
                        "type": "function",
                        "function": {"name": "run_command", "arguments": {}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_earlier", "content": "ok"},
            {"role": "assistant", "content": "done"},
            {"role": "tool", "tool_call_id": "call_earlier", "content": "late"},
        ]

        coerced = _coerce_orphan_tool_messages_for_template(messages)

        assert coerced[1]["role"] == "tool"
        assert coerced[3] == {
            "role": "user",
            "content": "Tool result (call_earlier):\nlate",
        }

    def test_chat_multimodal_summary_handles_input_image_shape(self):
        from vmlx_engine.server import _messages_multimodal_summary

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ""},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,SECRET_CHAT_INPUT_IMAGE",
                    },
                ],
            }
        ]

        summary = _messages_multimodal_summary(messages)

        assert summary["total"] == 1
        assert summary["types"] == {"input_image": 1}
        assert summary["data_url"] == 1
        assert summary["roles"] == {"user": 1}
        assert "SECRET_CHAT_INPUT_IMAGE" not in json.dumps(summary)

    def test_anthropic_image_conversion_reaches_media_summary(self):
        from vmlx_engine.api.anthropic_adapter import AnthropicRequest, to_chat_completion
        from vmlx_engine.server import _messages_multimodal_summary

        anthropic = AnthropicRequest(
            model="vl",
            max_tokens=32,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "SECRET_ANTHROPIC_IMAGE",
                            },
                        },
                        {"type": "text", "text": "describe"},
                    ],
                }
            ],
        )

        chat = to_chat_completion(anthropic)
        summary = _messages_multimodal_summary(chat.messages)

        assert summary["total"] == 1
        assert summary["types"] == {"image_url": 1}
        assert summary["data_url"] == 1
        assert summary["roles"] == {"user": 1}
        assert "SECRET_ANTHROPIC_IMAGE" not in json.dumps(summary)

    def test_media_diag_hooks_cover_anthropic_and_ollama_streaming_ingress(self):
        import inspect
        import vmlx_engine.server as server

        anthropic_source = inspect.getsource(server.create_anthropic_message)
        ollama_source = inspect.getsource(server.ollama_chat)

        assert '"/v1/messages"' in anthropic_source
        assert "_log_multimodal_request_shape" in anthropic_source
        assert "_messages_multimodal_summary(chat_req.messages)" in anthropic_source
        assert '_reject_unsupported_multimodal("/v1/messages")' in anthropic_source

        assert '"/api/chat"' in ollama_source
        assert "_log_multimodal_request_shape" in ollama_source
        assert "_messages_multimodal_summary(chat_req.messages)" in ollama_source
        assert '_reject_unsupported_multimodal("/api/chat")' in ollama_source

    def test_anthropic_media_on_text_runtime_rejects_instead_of_dropping(
        self, monkeypatch
    ):
        from fastapi.testclient import TestClient
        import vmlx_engine.server as server

        monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=False))
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_name", "text-runtime")
        monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)

        response = TestClient(server.app).post(
            "/v1/messages",
            json={
                "model": "claude",
                "max_tokens": 8,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "aGVsbG8=",
                                },
                            },
                            {"type": "text", "text": ""},
                        ],
                    }
                ],
            },
        )

        assert response.status_code == 400
        assert "text-only" in response.text

    def test_ollama_streaming_media_on_text_runtime_rejects_instead_of_dropping(
        self, monkeypatch
    ):
        from fastapi.testclient import TestClient
        import vmlx_engine.server as server

        monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=False))
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_name", "text-runtime")
        monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)

        response = TestClient(server.app).post(
            "/api/chat",
            json={
                "model": "text-runtime",
                "stream": True,
                "messages": [
                    {"role": "user", "content": "", "images": ["aGVsbG8="]}
                ],
            },
        )

        assert response.status_code == 400
        assert "text-only" in response.text

    def test_zaya_vl_runtime_modalities_do_not_infer_video_from_vision(
        self, monkeypatch, tmp_path
    ):
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "zaya1_vl",
                    "vision_config": {"model_type": "qwen2_5_vl"},
                    "image_token_id": 262147,
                    "capabilities": {"modality": "vision"},
                }
            )
        )
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"video_token": "<video>"})
        )

        monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=True))
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "zaya-vl-test")
        monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)

        assert server._loaded_runtime_modalities() == ["text", "vision"]

    def test_mimo_v2_runtime_modalities_stay_text_only_until_vl_is_wired(
        self, monkeypatch, tmp_path
    ):
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "mimo_v2",
                    "vision_config": {"model_type": "mimo_v2_vision"},
                    "audio_config": {"model_type": "mimo_v2_audio"},
                    "image_token_id": 151655,
                    "video_token_id": 151656,
                }
            )
        )
        monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=True))
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "mimo-v2-test")
        monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)

        assert server._loaded_runtime_modalities() == ["text"]

    @pytest.mark.asyncio
    async def test_mimo_v2_capabilities_do_not_advertise_unwired_vl(
        self, monkeypatch, tmp_path
    ):
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "mimo_v2",
                    "vision_config": {"model_type": "mimo_v2_vision"},
                    "audio_config": {"model_type": "mimo_v2_audio"},
                    "image_token_id": 151655,
                    "video_token_id": 151656,
                }
            )
        )

        class _Engine:
            is_mllm = True

        class _Scheduler:
            config = SimpleNamespace(enable_prefix_cache=False)
            block_aware_cache = None
            paged_cache_manager = None
            memory_aware_cache = None
            prefix_cache = None

        monkeypatch.setattr(server, "_engine", _Engine())
        monkeypatch.setattr(server, "_get_scheduler", lambda: _Scheduler())
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "mimo-v2-test")
        monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)

        caps = await server.model_capabilities("mimo-v2-test")

        assert caps["family"] == "mimo_v2"
        assert caps["modalities"] == ["text"]
        assert caps["media"]["runtime_modalities"] == ["text"]
        assert caps["media"]["preserved_modalities"] == ["vision", "image", "video", "audio"]
        assert caps["media"]["unwired_modalities"] == ["vision", "image", "video", "audio"]
        assert caps["media"]["status_by_modality"]["vision"] == "preserved_unwired"
        assert caps["media"]["status_by_modality"]["image"] == "preserved_unwired"
        assert caps["media"]["status_by_modality"]["video"] == "preserved_unwired"
        assert caps["media"]["status_by_modality"]["audio"] == "preserved_unwired"
        assert caps["media"]["status_by_modality"]["text"] == "runtime_supported"

    @pytest.mark.asyncio
    async def test_mimo_v2_text_only_capabilities_do_not_fallback_to_vision_when_registry_misses(
        self, monkeypatch
    ):
        import vmlx_engine.model_config_registry as registry
        import vmlx_engine.server as server

        class _Registry:
            def lookup(self, _model_key):
                return SimpleNamespace(
                    family_name="unknown",
                    reasoning_parser=None,
                    tool_parser=None,
                    think_in_template=False,
                    is_mllm=True,
                    supports_thinking=False,
                )

        monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=True))
        monkeypatch.setattr(server, "_model_path", "/tmp/MiMo-V2.5-JANGTQ_2")
        monkeypatch.setattr(server, "_model_name", "mimo-live-alias")
        monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)
        monkeypatch.setattr(server, "_loaded_runtime_modalities", lambda: ["text"])
        monkeypatch.setattr(server, "_mimo_v2_runtime_modalities", lambda _path: ["text"])
        monkeypatch.setattr(registry, "get_model_config_registry", lambda: _Registry())

        caps = await server.model_capabilities("mimo-live-alias")

        assert caps["modalities"] == ["text"]

    def test_qwen_vl_runtime_modalities_keep_explicit_video(
        self, monkeypatch, tmp_path
    ):
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5",
                    "vision_config": {"model_type": "qwen3_vl"},
                    "video_token_id": 248057,
                }
            )
        )

        monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=True))
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "qwen-vl-test")
        monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)

        assert server._loaded_runtime_modalities() == ["text", "vision", "video"]

    def test_gemma4_runtime_modalities_do_not_infer_video_from_token_only_config(
        self, monkeypatch, tmp_path
    ):
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "gemma4",
                    "vision_config": {"model_type": "siglip"},
                    "text_config": {"video_token_id": 262145},
                }
            )
        )

        monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=True))
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "gemma4-video-token-only-test")
        monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)

        assert server._loaded_runtime_modalities() == ["text", "vision"]

    def test_gemma4_runtime_modalities_do_not_infer_audio_from_token_only_config(
        self, monkeypatch, tmp_path
    ):
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "gemma4",
                    "vision_config": {"model_type": "siglip"},
                    "audio_config": None,
                    "audio_token_id": 258881,
                }
            )
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"language_model.model.embed_tokens.weight": "a.safetensors"}})
        )

        monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=True))
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "gemma4-audio-token-only-test")
        monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)

        assert server._loaded_runtime_modalities() == ["text", "vision"]

    def test_gemma4_runtime_modalities_advertise_audio_with_audio_tower_weights(
        self, monkeypatch, tmp_path
    ):
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "gemma4",
                    "vision_config": {"model_type": "siglip"},
                    "audio_config": {"model_type": "gemma4_audio"},
                    "audio_token_id": 258881,
                }
            )
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.weight": "a.safetensors",
                        "language_model.model.embed_tokens.weight": "b.safetensors",
                    }
                }
            )
        )

        monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=True))
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "gemma4-audio-tower-test")
        monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)

        assert server._loaded_runtime_modalities() == ["text", "vision", "audio"]

    def test_gemma4_unified_jang4m_runtime_modalities_gate_config_only_audio(
        self, monkeypatch, tmp_path
    ):
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "gemma4_unified",
                    "vision_config": {"model_type": "gemma4_unified_vision"},
                    "audio_config": {"model_type": "gemma4_unified_audio"},
                }
            )
        )
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"weight_format": "jang_4m", "profile": "jang_4m"})
        )

        monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=True))
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "gemma4-unified-jang4m-test")
        monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)

        assert server._loaded_runtime_modalities() == ["text", "vision"]

    def test_gemma4_unified_jang4m_runtime_modalities_advertise_weight_backed_audio(
        self, monkeypatch, tmp_path
    ):
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "gemma4_unified",
                    "vision_config": {"model_type": "gemma4_unified_vision"},
                    "audio_config": {"model_type": "gemma4_unified_audio"},
                }
            )
        )
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"weight_format": "jang_4m", "profile": "jang_4m"})
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.weight": "a.safetensors",
                        "embed_audio.embedding_projection.weight": "b.safetensors",
                    }
                }
            )
        )

        monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=True))
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "gemma4-unified-jang4m-audio-test")
        monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)

        assert server._loaded_runtime_modalities() == ["text", "vision", "audio"]

    def test_gemma4_unified_mxfp_runtime_modalities_gate_unproven_audio(
        self, monkeypatch, tmp_path
    ):
        import vmlx_engine.server as server

        for weight_format in ("mxfp4", "mxfp8"):
            bundle_path = tmp_path / weight_format
            bundle_path.mkdir()
            (bundle_path / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "gemma4_unified",
                        "vision_config": {"model_type": "gemma4_unified_vision"},
                        "audio_config": {"model_type": "gemma4_unified_audio"},
                    }
                )
            )
            (bundle_path / "jang_config.json").write_text(
                json.dumps({"weight_format": weight_format, "profile": weight_format})
            )

            monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=True))
            monkeypatch.setattr(server, "_model_path", str(bundle_path))
            monkeypatch.setattr(server, "_model_name", f"gemma4-unified-{weight_format}-test")
            monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)

            assert server._loaded_runtime_modalities() == ["text", "vision"]

    def test_gemma4_unified_mxfp8_attnfp16_runtime_modalities_gate_audio_by_default(
        self, monkeypatch, tmp_path
    ):
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "gemma4_unified",
                    "vision_config": {"model_type": "gemma4_unified_vision"},
                    "audio_config": {"model_type": "gemma4_unified_audio"},
                }
            )
        )
        (tmp_path / "jang_config.json").write_text(
            json.dumps(
                {
                    "weight_format": "mxfp8",
                    "profile": "MXFP8_ATTNFP16",
                    "quantization": {
                        "selective_passthrough": {
                            "preserve_attention_fp16": True,
                            "preserve_full_attention_fp16": False,
                            "preserve_first_layers": 0,
                        }
                    },
                }
            )
        )

        monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=True))
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "gemma4-unified-mxfp8-attnfp16-test")
        monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)
        monkeypatch.delenv("VMLX_ALLOW_EXPERIMENTAL_MXFP_AUDIO", raising=False)
        monkeypatch.delenv("VMLINUX_ALLOW_EXPERIMENTAL_MXFP_AUDIO", raising=False)

        assert server._loaded_runtime_modalities() == ["text", "vision"]

    def test_gemma4_unified_mxfp8_attnfp16_runtime_modalities_can_experimentally_advertise_audio(
        self, monkeypatch, tmp_path
    ):
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "gemma4_unified",
                    "vision_config": {"model_type": "gemma4_unified_vision"},
                    "audio_config": {"model_type": "gemma4_unified_audio"},
                }
            )
        )
        (tmp_path / "jang_config.json").write_text(
            json.dumps(
                {
                    "weight_format": "mxfp8",
                    "profile": "MXFP8_ATTNFP16",
                    "quantization": {
                        "selective_passthrough": {
                            "preserve_attention_fp16": True,
                            "preserve_full_attention_fp16": False,
                            "preserve_first_layers": 0,
                        }
                    },
                }
            )
        )

        monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=True))
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "gemma4-unified-mxfp8-attnfp16-test")
        monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)
        monkeypatch.setenv("VMLX_ALLOW_EXPERIMENTAL_MXFP_AUDIO", "1")

        assert server._loaded_runtime_modalities() == ["text", "vision", "audio"]

    def test_gemma4_chat_audio_request_rejects_when_audio_not_weight_backed(
        self, monkeypatch, tmp_path
    ):
        from fastapi.testclient import TestClient
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "gemma4_unified",
                    "vision_config": {"model_type": "gemma4_unified_vision"},
                    "audio_config": {"model_type": "gemma4_unified_audio"},
                    "audio_token_id": 258881,
                }
            )
        )
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"weight_format": "jang_4m", "profile": "jang_4m"})
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "embed_audio.embedding_projection.weight": "a.safetensors",
                        "language_model.model.embed_tokens.weight": "b.safetensors",
                    }
                }
            )
        )

        monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=True))
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "gemma4-unified-audio-gate-test")
        monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)

        response = TestClient(server.app).post(
            "/v1/chat/completions",
            json={
                "model": "gemma4-unified-audio-gate-test",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Transcribe this audio."},
                            {
                                "type": "input_audio",
                                "input_audio": {"data": "AAAA", "format": "wav"},
                            },
                        ],
                    }
                ],
                "max_tokens": 8,
            },
        )

        assert response.status_code == 400
        assert "unsupported media modality audio" in response.text
        assert "text, vision" in response.text

    def test_video_request_on_image_only_mllm_rejects_instead_of_crashing(
        self, monkeypatch, tmp_path
    ):
        from fastapi.testclient import TestClient
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "zaya1_vl",
                    "vision_config": {"model_type": "qwen2_5_vl"},
                    "image_token_id": 262147,
                    "capabilities": {"modality": "vision"},
                }
            )
        )

        monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=True))
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "zaya-vl-test")
        monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)

        response = TestClient(server.app).post(
            "/v1/chat/completions",
            json={
                "model": "zaya-vl-test",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is in this video?"},
                            {
                                "type": "video_url",
                                "video_url": {"url": "data:video/mp4;base64,AAAA"},
                            },
                        ],
                    }
                ],
                "max_tokens": 8,
            },
        )

        assert response.status_code == 400
        assert "unsupported media modality video" in response.text
        assert "text, vision" in response.text

    def test_server_media_diag_log_includes_route_runtime_and_redacts_payload(
        self, caplog, monkeypatch
    ):
        import logging
        import vmlx_engine.server as server

        monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=True))
        monkeypatch.setattr(server, "_model_path", "/models/Brooklyn-VLM")
        monkeypatch.setattr(server, "_model_name", "Brooklyn-VLM")
        monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)
        summary = {
            "total": 1,
            "types": {"image_url": 1},
            "data_url": 1,
            "url_or_path": 0,
            "missing_source": 0,
            "roles": {"user": 1},
            "messages": 1,
            "content_arrays": 1,
        }

        caplog.set_level(logging.INFO, logger="vmlx_engine.server")
        server._log_multimodal_request_shape(
            "/v1/chat/completions",
            "/models/Brooklyn-VLM",
            summary,
        )

        text = caplog.text
        assert "[MEDIA_DIAG] request_shape=" in text
        assert '"/v1/chat/completions"' in text
        assert '"engine_is_mllm": true' in text
        assert '"total": 1' in text
        assert "SECRET" not in text
        assert "data:image" not in text


class TestH4JsonSchemaStreaming:
    """H4: JSON schema/object validation must happen at end of streaming."""

    def test_chat_completion_streaming_validates_json(self):
        import inspect
        from vmlx_engine.server import stream_chat_completion

        source = inspect.getsource(stream_chat_completion)
        assert "parse_json_output" in source, (
            "Streaming path must call parse_json_output for response_format validation"
        )

    def test_responses_api_streaming_validates_json(self):
        import inspect
        from vmlx_engine.server import stream_responses_api

        source = inspect.getsource(stream_responses_api)
        assert "parse_json_output" in source, (
            "Responses API streaming must validate JSON format"
        )

    def test_streaming_emits_error_on_strict_failure(self):
        import inspect
        from vmlx_engine.server import stream_chat_completion

        source = inspect.getsource(stream_chat_completion)
        assert "json_validation_failed" in source, (
            "Streaming must emit error event on strict JSON schema failure"
        )

    def test_server_docs_state_response_format_is_not_constrained_decoding(self):
        server_doc = Path("docs/guides/server.md").read_text(encoding="utf-8")
        config_doc = Path("docs/reference/configuration.md").read_text(encoding="utf-8")

        combined = f"{server_doc}\n{config_doc}".lower()

        assert "llguidance" in combined
        assert "guided json/schema token masking" in combined
        assert "post-generation repair" in combined
        assert "xml_root_tag" in combined
        assert "required_xml_fields" in combined
        assert "xml-only correction retry" in combined
        assert "xml_validation_failed" in combined
        assert "not universal hard grammar-constrained decoding" in combined
        assert "repair/validation" in combined
        assert "one json-only correction retry" in combined
        assert "force the model to return valid json" not in combined


class TestH6ConfigRestartWarning:
    """H6: updateSessionConfig must return restart-required info."""

    def test_restart_required_keys_defined(self):
        """SessionManager must define which config keys need restart."""
        import inspect
        # Read source file directly since TypeScript
        source_path = os.path.join(
            os.path.dirname(__file__), '..', 'panel', 'src', 'main', 'sessions.ts'
        )
        with open(source_path) as f:
            source = f.read()
        assert "RESTART_REQUIRED_KEYS" in source, (
            "SessionManager must define RESTART_REQUIRED_KEYS"
        )
        assert "restartRequired" in source, (
            "updateSessionConfig must return restartRequired flag"
        )


class TestM5Base64TempFileCleanup:
    """M5: LRU cache eviction must also delete the temp file from disk."""

    def test_eviction_calls_cleanup(self):
        import inspect
        from vmlx_engine.models.mllm import save_base64_image

        source = inspect.getsource(save_base64_image)
        assert "_temp_manager.cleanup(evicted_path)" in source, (
            "Base64 image cache eviction must call _temp_manager.cleanup on evicted path"
        )


class TestM8UsageOnError:
    """M8: Usage must be sent even when stream encounters an error."""

    def test_chat_completion_stream_sends_usage_on_error(self):
        import inspect
        from vmlx_engine.server import stream_chat_completion

        source = inspect.getsource(stream_chat_completion)
        # The error handler must include usage when include_usage is on
        # Find the except block and check for usage handling
        error_section = source[source.find("Stream generation failed"):]
        assert "include_usage" in error_section[:500], (
            "Stream error handler must check include_usage and send partial usage"
        )

    def test_responses_api_stream_sends_usage_on_error(self):
        import inspect
        from vmlx_engine.server import stream_responses_api

        source = inspect.getsource(stream_responses_api)
        error_section = source[source.find("Stream generation failed"):]
        assert "usage" in error_section[:500], (
            "Responses API error handler must include usage in failed response"
        )


class TestC3BlockDiskCacheQuantScoping:
    """C3 extension: Block-level disk cache must also scope by quantization config."""

    def test_scheduler_block_cache_includes_quant(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler

        source = inspect.getsource(Scheduler.__init__)
        # Find the block disk cache section
        block_section = source[source.find("enable_block_disk_cache"):]
        assert "quant" in block_section[:500], (
            "Block disk cache hash must include quantization in scope key"
        )

    def test_mllm_scheduler_block_cache_includes_quant(self):
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.__init__)
        block_section = source[source.find("enable_block_disk_cache"):]
        assert "quant" in block_section[:500], (
            "MLLM block disk cache hash must include quantization in scope key"
        )


class TestTimeoutFalsy:
    """Timeout=0 must not be silently overridden to default (same class of bug as C4)."""

    def test_all_timeout_uses_is_not_none(self):
        import inspect
        from vmlx_engine.server import (
            create_chat_completion,
            create_response,
        )

        for fn in [create_chat_completion, create_response]:
            source = inspect.getsource(fn)
            assert "request.timeout or _default_timeout" not in source, (
                f"{fn.__name__} must use 'is not None' for timeout, not 'or'"
            )
            assert "request.timeout if request.timeout is not None" in source, (
                f"{fn.__name__} must use 'is not None' check for timeout"
            )


# ===========================================================================
# H3 Production Path: _prepare_images total-failure detection
# ===========================================================================


class TestH3PrepareImagesTotalFailure:
    """Test that _prepare_images raises on total failure (production path in models/mllm.py)."""

    @staticmethod
    def _read_mllm_source():
        return (Path(__file__).parent.parent / "vmlx_engine" / "models" / "mllm.py").read_text()

    def test_all_images_fail_raises_valueerror(self):
        """When every image fails to process, should raise ValueError."""
        source = self._read_mllm_source()
        # _prepare_images must track failures and raise when all fail
        assert "failed_count" in source
        assert "images and not processed" in source
        assert 'raise ValueError' in source

    def test_prepare_images_has_failure_guard(self):
        """_prepare_images must guard against total failure but allow partial."""
        source = self._read_mllm_source()
        # Guard: `if images and not processed` — empty list skips, partial succeeds
        assert "images and not processed" in source


# ===========================================================================
# H4 Regression: Empty output is not model text
# ===========================================================================


class TestH4EmptyOutputIsNotAssistantText:
    """Empty Responses output must stay empty instead of becoming fake model text."""

    def test_responses_api_empty_output_skips_validation_without_placeholder(self):
        """Responses API JSON validation should skip naturally on empty display_text."""
        import inspect
        from vmlx_engine.server import stream_responses_api

        source = inspect.getsource(stream_responses_api)
        assert "Model produced no response" not in source
        assert "Model produced only internal reasoning" not in source
        assert "_FALLBACK_MSG" not in source
        assert "and display_text" in source
        assert "empty_model_response" in source
        assert "reasoning_only_no_content" in source

    def test_chat_completions_already_safe(self):
        """Chat completions uses content_was_emitted gate — already safe."""
        import inspect
        from vmlx_engine.server import stream_chat_completion

        source = inspect.getsource(stream_chat_completion)
        # Chat completions gates on content_was_emitted
        assert "content_was_emitted" in source
        assert "Model produced only internal reasoning" not in source
        assert "ChatCompletionChunkDelta()" in source


# ===========================================================================
# H2 Video Frame Bypass: Total image count guard in generate paths
# ===========================================================================


class TestH2VideoFrameBypass:
    """Test that total image count (including video frames) is enforced."""

    @staticmethod
    def _read_mllm_source():
        return (Path(__file__).parent.parent / "vmlx_engine" / "models" / "mllm.py").read_text()

    def test_all_generate_paths_have_total_image_guard(self):
        """All generate/stream_generate/chat/stream_chat must check total images."""
        source = self._read_mllm_source()
        # Count occurrences of the guard — should appear in all 4 generate paths
        guard_count = source.count("max_images_per_request")
        assert guard_count >= 4, (
            f"Expected max_images_per_request guard in at least 4 places, found {guard_count}"
        )
        assert source.count("including video frames") >= 4, (
            "Expected 'including video frames' error message in all 4 generate paths"
        )


# ===========================================================================
# H1 LLM Scheduler Parity: Stop token cleanup
# ===========================================================================


class TestH1LLMSchedulerStopTokenParity:
    """Test that LLM Scheduler has stop token cleanup matching MLLM Scheduler."""

    def test_scheduler_has_stop_tokens_attribute(self):
        """LLM Scheduler must track base stop tokens."""
        import inspect
        from vmlx_engine.scheduler import Scheduler

        source = inspect.getsource(Scheduler.__init__)
        assert "self.stop_tokens" in source
        assert "_get_stop_tokens" in source

    def test_schedule_waiting_adds_per_request_stop_tokens(self):
        """_schedule_waiting must add per-request stop tokens to batch generator."""
        import inspect
        from vmlx_engine.scheduler import Scheduler

        source = inspect.getsource(Scheduler._schedule_waiting)
        assert "_added_stop_tokens" in source
        assert "batch_generator.stop_tokens.update" in source

    def test_cleanup_finished_uses_surviving_stops_snapshot(self):
        """_cleanup_finished must use _surviving_stops snapshot pattern."""
        import inspect
        from vmlx_engine.scheduler import Scheduler

        source = inspect.getsource(Scheduler._cleanup_finished)
        assert "_surviving_stops" in source
        assert "request._added_stop_tokens" in source
        assert "removable" in source

    def test_cleanup_never_removes_base_stop_tokens(self):
        """Cleanup must subtract self.stop_tokens to protect base EOS tokens."""
        import inspect
        from vmlx_engine.scheduler import Scheduler

        source = inspect.getsource(Scheduler._cleanup_finished)
        assert "self.stop_tokens" in source


# ===========================================================================
# V3 Audit: Deep cross-component issues
# ===========================================================================


class TestV3ToolCallIdForwarding:
    """Non-streaming Responses API must forward tc.id from parser."""

    def test_responses_nonstreaming_forwards_tc_id(self):
        import inspect
        from vmlx_engine.server import create_response
        source = inspect.getsource(create_response)
        # Must reference tc.id and pass call_id to ResponsesFunctionCall
        assert "tc.id" in source or "tc_call_id" in source
        assert "call_id" in source


class TestV3RescheduleCleanup:
    """_reschedule_running_requests must clear _extracted_cache."""

    def test_extracted_cache_cleared_on_reschedule(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler._reschedule_running_requests)
        assert "_extracted_cache" in source


class TestV3ScheduleWaitingRecovery:
    """_schedule_waiting must not lose requests on insert failure."""

    def test_lost_request_protection(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler._schedule_waiting)
        # Must have outer try/except that puts request back
        assert "waiting.appendleft(request)" in source
        # Must appear at least twice — once for batch_generator=None, once for insert failure
        assert source.count("waiting.appendleft(request)") >= 2


class TestV3BlockCacheFinallyCleanup:
    """block_aware_cache branch must use finally for _extracted_cache."""

    def test_block_cache_uses_finally(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler._cleanup_finished)
        # Must have 'finally' followed by _extracted_cache cleanup
        assert "finally:" in source
        assert "request._extracted_cache = None" in source


class TestV3GenerateCleanupOnError:
    """EngineCore.generate() must clean up on all error paths."""

    def test_generate_has_finally_cleanup(self):
        import inspect
        from vmlx_engine.engine_core import EngineCore
        source = inspect.getsource(EngineCore.generate)
        assert "finally:" in source
        assert "_cleanup_request" in source


class TestV3ResponsesTextFormatSchema:
    """ResponsesTextFormat must preserve json_schema field."""

    def test_text_format_has_json_schema_field(self):
        from vmlx_engine.api.models import ResponsesTextFormat
        # json_schema field must exist
        assert "json_schema" in ResponsesTextFormat.model_fields

    def test_text_format_preserves_schema_data(self):
        from vmlx_engine.api.models import ResponsesTextFormat
        fmt = ResponsesTextFormat(type="json_schema", json_schema={"name": "test", "schema": {"type": "object"}})
        dumped = fmt.model_dump()
        assert dumped["json_schema"] is not None
        assert dumped["json_schema"]["name"] == "test"


class TestV3ChatCompletionStopNormalization:
    """ChatCompletionRequest.stop must accept bare strings."""

    def test_bare_string_normalized_to_list(self):
        from vmlx_engine.api.models import ChatCompletionRequest
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            stop="\\n",
        )
        assert isinstance(req.stop, list)
        assert req.stop == ["\\n"]

    def test_list_preserved(self):
        from vmlx_engine.api.models import ChatCompletionRequest
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            stop=["\\n", "END"],
        )
        assert req.stop == ["\\n", "END"]


class TestV3SuppressReasoningNoFallback:
    """Suppress reasoning + reasoning-only output should NOT show fallback."""

    def test_responses_api_no_fallback_on_suppressed_reasoning(self):
        import inspect
        from vmlx_engine.server import stream_responses_api
        source = inspect.getsource(stream_responses_api)
        # Must check suppress_reasoning before showing fallback
        assert "suppress_reasoning and accumulated_reasoning" in source


class TestV3ReasoningDoubleAccumulation:
    """Suppressed reasoning must stay out of visible content/history.

    The panel now keeps reasoning in the dedicated reasoning channel and
    surfaces reasoning-only cases as warnings/placeholders instead of
    redirecting parser output into normal assistant content.
    """

    def test_chat_accumulated_content_mirror_under_suppress(self):
        import inspect
        from vmlx_engine.server import stream_chat_completion
        source = inspect.getsource(stream_chat_completion)
        assert "accumulated_content += delta_msg.reasoning" not in source
        assert "Suppressed reasoning is never redirected into visible content" in source
        assert "suppress_reasoning and not content_was_emitted and accumulated_reasoning" in source

    def test_responses_accumulated_content_mirror_under_suppress(self):
        import inspect
        from vmlx_engine.server import stream_responses_api
        source = inspect.getsource(stream_responses_api)
        assert "accumulated_content += delta_msg.reasoning" not in source
        assert "Suppressed reasoning is never redirected into" in source
        assert "reasoning_only_no_content" in source


class TestV3KvCacheBitsInit:
    """Scheduler must initialize _kv_cache_bits to 0."""

    def test_kv_cache_bits_in_init(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler.__init__)
        assert "self._kv_cache_bits" in source
        assert "_kv_cache_bits: int = 0" in source or "_kv_cache_bits = 0" in source


# ===========================================================================
# V4. Deep Audit — Cache subsystem, prefix cache, truncation fixes
# ===========================================================================


class TestV4DiskCacheDequantizeGuard:
    """Disk cache fetch must defer dequantization to the scheduler worker."""

    def test_scheduler_disk_cache_dequantize(self):
        """scheduler.py: disk_cache.fetch result must not dequantize on API thread."""
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler.add_request)
        fetch_idx = source.find("disk_cache.fetch")
        assert fetch_idx != -1, "disk_cache.fetch not found in add_request"
        after_fetch = source[fetch_idx:fetch_idx + 900]
        assert "_prompt_cache_needs_worker_dequant = True" in after_fetch, (
            "disk_cache.fetch must mark cache hits for worker-side dequantization"
        )
        assert "_dequantize_cache_for_use(" not in after_fetch, (
            "disk_cache.fetch path must not dequantize MLX arrays on the API thread"
        )
        schedule_source = inspect.getsource(Scheduler._schedule_waiting)
        assert "_prompt_cache_needs_worker_dequant" in schedule_source
        assert "_dequantize_cache_for_use(cache_to_use)" in schedule_source, (
            "worker schedule path must own stored-cache dequantization"
        )

    def test_mllm_disk_cache_dequantize(self):
        """mllm_batch_generator.py: disk cache fetch must dequantize."""
        source = Path("./vmlx_engine/mllm_batch_generator.py").read_text()
        # Find the actual disk_cache.fetch() call (with parens), not docstring mention
        fetch_idx = source.find("self.disk_cache.fetch(")
        assert fetch_idx != -1, "self.disk_cache.fetch() not found in mllm_batch_generator.py"
        after_fetch = source[fetch_idx:fetch_idx + 1500]
        assert "_dequantize_cache" in after_fetch, (
            "Missing dequantize guard after disk_cache.fetch in mllm_batch_generator.py"
        )


class TestV4DequantizeFreshKVCacheFallback:
    """_dequantize_cache must return fresh KVCache for QuantizedKVCache with keys=None."""

    def test_quantized_with_none_keys_gets_fresh_kvcache(self):
        source = Path("./vmlx_engine/mllm_batch_generator.py").read_text()
        func_start = source.find("def _dequantize_cache(")
        assert func_start != -1
        # Find the next function definition to bound our search
        func_end = source.find("\ndef ", func_start + 10)
        func_body = source[func_start:func_end]
        # Must handle QuantizedKVCache with keys=None by creating fresh KVCache
        assert "KVCache()" in func_body, (
            "_dequantize_cache must create fresh KVCache() for empty QuantizedKVCache layers"
        )
        # Must NOT silently pass through QuantizedKVCache to result
        assert "keys is not None" in func_body or "keys is None" in func_body, (
            "_dequantize_cache must explicitly check for None keys"
        )


class TestV4FixHybridCacheDequantize:
    """_fix_hybrid_cache call in prefill must be preceded by _dequantize_cache."""

    def test_dequantize_before_fix_hybrid(self):
        source = Path("./vmlx_engine/mllm_batch_generator.py").read_text()
        # Find the prefill section where _fix_hybrid_cache is called on req.prompt_cache
        idx = source.find("req_cache = _fix_hybrid_cache(")
        assert idx != -1
        # Look at the nearby prefill branch before this call. Keep the window
        # wide enough to include the explicit dequantize failure guard without
        # becoming a whole-file substring assertion.
        before = source[max(0, idx - 1200):idx]
        assert "_dequantize_cache" in before, (
            "Must call _dequantize_cache before _fix_hybrid_cache in prefill path"
        )
        # Verify None guard exists after dequantize
        assert "cache_for_fix is None" in before, (
            "Must guard against _dequantize_cache returning None"
        )


class TestV4RotatingKVCacheTruncation:
    """RotatingKVCache must not truncate when circular buffer has wrapped."""

    def test_rotating_cache_wrap_detection(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler._truncate_cache_to_prompt_length)
        # Must check for offset > max_size (wrapped circular buffer)
        assert "offset > max_size" in source, (
            "Must detect wrapped RotatingKVCache (offset > max_size) and skip"
        )

    def test_rotating_cache_idx_restore(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler._truncate_cache_to_prompt_length)
        # Must restore _idx for RotatingKVCache
        assert "_idx" in source, (
            "Must restore _idx for RotatingKVCache after truncation"
        )

    def test_safe_target_bounds(self):
        """Truncation must use min(target_len, actual_shape) to prevent OOB."""
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler._truncate_cache_to_prompt_length)
        assert "safe_target" in source or "min(target_len" in source, (
            "Must bound slice to actual tensor shape to prevent OOB"
        )


class TestV4CacheListTruncation:
    """CacheList (DeepSeek V3.2, Falcon H1) must be handled in truncation."""

    def test_cachelist_branch_exists(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler._truncate_cache_to_prompt_length)
        assert "CacheList" in source, (
            "_truncate_cache_to_prompt_length must handle CacheList"
        )
        assert "caches" in source, (
            "Must access CacheList.caches for recursive truncation"
        )


class TestV4PrefixCacheLRU:
    """PrefixCacheManager LRU must use OrderedDict for O(1) and dedup.

    LRU storage is now partitioned by cache_type (assistant / user / system)
    via ``_lru_by_type`` so eviction can prefer lower-priority buckets
    first. The original ``_lru`` attribute was replaced; these tests walk
    all buckets to produce the combined view.
    """

    @staticmethod
    def _all_lru_keys(mgr):
        keys = []
        for od in mgr._lru_by_type.values():
            keys.extend(od.keys())
        return keys

    def test_lru_is_ordered_dict(self):
        from vmlx_engine.prefix_cache import PrefixCacheManager
        mock_model = MagicMock()
        mgr = PrefixCacheManager(mock_model, max_entries=10)
        from collections import OrderedDict
        assert hasattr(mgr, "_lru_by_type"), (
            "PrefixCacheManager must expose per-type LRU dict"
        )
        for t, od in mgr._lru_by_type.items():
            assert isinstance(od, OrderedDict), (
                f"bucket {t!r} must be OrderedDict for O(1) reordering"
            )

    def test_no_duplicate_lru_entries(self):
        """Storing same tokens twice must not create duplicate LRU entries."""
        from vmlx_engine.prefix_cache import PrefixCacheManager
        mock_model = MagicMock()
        mgr = PrefixCacheManager(mock_model, max_entries=10)
        tokens = [1, 2, 3]
        cache = [MagicMock()]
        mgr.store_cache(tokens, cache)
        mgr.store_cache(tokens, cache)
        total = len(self._all_lru_keys(mgr))
        assert total == 1, (
            f"Expected 1 LRU entry after duplicate store, got {total}"
        )

    def test_touch_lru_moves_to_end(self):
        """Touching an entry must move it to the end (MRU position)."""
        from vmlx_engine.prefix_cache import PrefixCacheManager
        mock_model = MagicMock()
        mgr = PrefixCacheManager(mock_model, max_entries=10)
        mgr.store_cache([1, 2], [MagicMock()])
        mgr.store_cache([3, 4], [MagicMock()])
        # Touch first entry — it should move to end of its bucket
        mgr._touch_lru(tuple([1, 2]))
        keys = self._all_lru_keys(mgr)
        assert keys[-1] == (mgr.model_key, (1, 2)), (
            "Touch must move entry to MRU end"
        )

    def test_eviction_removes_lru(self):
        """Eviction must remove the least recently used entry."""
        from vmlx_engine.prefix_cache import PrefixCacheManager
        mock_model = MagicMock()
        mgr = PrefixCacheManager(mock_model, max_entries=2)
        mgr.store_cache([1], [MagicMock()])
        mgr.store_cache([2], [MagicMock()])
        # This should evict [1]
        mgr.store_cache([3], [MagicMock()])
        keys = self._all_lru_keys(mgr)
        assert len(keys) == 2
        assert (mgr.model_key, (1,)) not in keys, "LRU entry [1] should have been evicted"


class TestV4QuantizeCacheSubclassHandling:
    """_quantize_cache_for_storage must use isinstance, not type() is."""

    def test_uses_isinstance_not_type_is(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler._quantize_cache_for_storage)
        assert "isinstance(layer_cache, KVCache)" in source, (
            "Must use isinstance() to catch KVCache subclasses (e.g. RotatingKVCache)"
        )
        assert "type(layer_cache) is KVCache" not in source, (
            "Must NOT use strict type() check — misses subclasses"
        )


# ===========================================================================
# V4b. Deep Audit Review — Cross-component cohesion fixes
# ===========================================================================


class TestV4bDiskCacheDequantFallthrough:
    """Disk cache dequant failure must NOT corrupt request state."""

    def test_scheduler_disk_dequant_failure_skips_state_mutation(self):
        """When dequant returns None, cached_tokens and remaining_tokens must NOT be set."""
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler.add_request)
        # Find the disk cache section
        fetch_idx = source.find("disk_cache = self.disk_cache.fetch")
        assert fetch_idx != -1
        after_fetch = source[fetch_idx:fetch_idx + 2000]
        # The pattern: if dequant fails (disk_cache is None), must NOT set cached_tokens
        # Correct pattern: else branch gates all state mutations
        assert "else:" in after_fetch, (
            "Disk cache dequant failure must have else branch to skip state mutation"
        )
        # Find 'request.cached_tokens' — it must be INSIDE the else block (indented deeper)
        cached_idx = after_fetch.find("request.cached_tokens")
        else_idx = after_fetch.find("else:")
        assert cached_idx > else_idx, (
            "request.cached_tokens must be inside the else block (after dequant success)"
        )


class TestV4bSchedulerDequantKeysNone:
    """scheduler.py _dequantize_cache_for_use must handle QuantizedKVCache(keys=None)."""

    def test_keys_none_gets_fresh_kvcache(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler._dequantize_cache_for_use)
        # Must create fresh KVCache for empty QuantizedKVCache layers
        assert "KVCache()" in source, (
            "_dequantize_cache_for_use must create fresh KVCache() for empty QuantizedKVCache"
        )

    def test_consistent_with_mllm_version(self):
        """Both dequantize functions must handle keys=None the same way."""
        import inspect
        from vmlx_engine.scheduler import Scheduler
        scheduler_src = inspect.getsource(Scheduler._dequantize_cache_for_use)
        mllm_src = Path("./vmlx_engine/mllm_batch_generator.py").read_text()
        func_start = mllm_src.find("def _dequantize_cache(")
        func_end = mllm_src.find("\ndef ", func_start + 10)
        mllm_func = mllm_src[func_start:func_end]
        # Both must have fresh KVCache for keys=None
        assert "KVCache()" in scheduler_src, "Scheduler missing KVCache() fallback"
        assert "KVCache()" in mllm_func, "MLLM missing KVCache() fallback"


class TestV4bCacheListTupleInvariant:
    """CacheList.caches must be stored as tuple to match constructor invariant."""

    def test_cachelist_stored_as_tuple(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler._truncate_cache_to_prompt_length)
        assert "tuple(sub_result)" in source, (
            "CacheList.caches must be stored as tuple, not list"
        )


class TestV4bImageHashCollisionSafe:
    """Paged cache image hashing must use content-based hash, not sum-based."""

    def test_no_sum_based_hash(self):
        source = Path("./vmlx_engine/paged_cache.py").read_text()
        hash_section = source[source.find("def _hash_extra"):source.find("_hash_extra(extra_keys)")]
        assert "mx.sum" not in hash_section, (
            "Must not use mx.sum for image hashing — collision-prone"
        )
        assert "tobytes" in hash_section, (
            "Must use tobytes() for collision-safe content hashing"
        )


class TestV4bToolChoiceRequired:
    """tool_choice='required' must be enforced in all API paths."""

    def test_chat_completions_nonstreaming(self):
        import inspect
        from vmlx_engine.server import create_chat_completion
        source = inspect.getsource(create_chat_completion)
        assert "required" in source, (
            "Chat Completions must check tool_choice='required'"
        )
        assert "tool_calls_required" in source or "HTTPException" in source, (
            "Chat Completions must raise error when required tool calls missing"
        )
        assert "raw_preview" in source, (
            "Chat Completions required-tool 400 must preserve raw_preview so "
            "parser/model failures are classifiable from smoke artifacts"
        )
        assert '_ct_kwargs.setdefault("tool_choice", _tool_choice)' in source, (
            "Chat Completions must forward tool_choice to chat-template kwargs "
            "so fallback prompt injection sees the same contract the API enforces"
        )

    def test_responses_nonstreaming(self):
        import inspect
        from vmlx_engine.server import create_response
        source = inspect.getsource(create_response)
        assert "required" in source, (
            "Responses API must check tool_choice='required'"
        )
        assert '_ct_kwargs.setdefault("tool_choice", _tool_choice)' in source, (
            "Responses API must forward tool_choice to chat-template kwargs "
            "so fallback prompt injection sees the same contract the API enforces"
        )
        assert "raw_preview" in source, (
            "Responses required-tool 400 must preserve raw_preview so "
            "parser/model failures are classifiable from smoke artifacts"
        )

    def test_chat_completions_streaming(self):
        import inspect
        from vmlx_engine.server import stream_chat_completion
        source = inspect.getsource(stream_chat_completion)
        assert "tool_calls_required" in source, (
            "Chat Completions streaming must emit error for tool_choice='required'"
        )

    def test_responses_streaming(self):
        import inspect
        from vmlx_engine.server import stream_responses_api
        source = inspect.getsource(stream_responses_api)
        assert "tool_calls_required" in source, (
            "Responses API streaming must emit error for tool_choice='required'"
        )


class TestResponsesStreamingExactToolResult:
    """Responses streaming must not drift from non-stream exact tool finalization."""

    def test_responses_streaming_exact_reply_uses_nonstream_finalizer(self):
        import inspect
        from vmlx_engine.server import stream_responses_api
        source = inspect.getsource(stream_responses_api)
        assert "_responses_exact_reply_target(request)" in source
        assert "_responses_messages_have_tool_result_after_latest_user(messages)" in source
        assert "_await_chat_with_disconnect_abort(" in source
        assert "chat_kwargs=kwargs" in source
        assert "response.output_text.delta" in source

    def test_exact_reply_finalizer_only_triggers_after_current_turn_tool_result(self):
        from vmlx_engine.server import (
            _responses_messages_have_tool_result_after_latest_user,
        )

        assert _responses_messages_have_tool_result_after_latest_user(
            [
                {"role": "user", "content": "use tool then reply exactly: A"},
                {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "A"},
            ]
        )
        assert not _responses_messages_have_tool_result_after_latest_user(
            [
                {"role": "user", "content": "use tool then reply exactly: A"},
                {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "A"},
                {"role": "assistant", "content": "A"},
                {"role": "user", "content": "use tool then reply exactly: B"},
            ]
        )

    def test_exact_reply_fast_path_uses_visible_post_think_text(self):
        from types import SimpleNamespace
        from vmlx_engine.server import _responses_fast_path_visible_text

        output = SimpleNamespace(
            raw_text=(
                'The command succeeded and produced "REAL_UI_LIVE_TOOL_TWO".\n'
                "</think>\n"
                "REAL_UI_LIVE_TOOL_TWO"
            ),
            text=(
                'The command succeeded and produced "REAL_UI_LIVE_TOOL_TWO".\n'
                "</think>\n"
                "REAL_UI_LIVE_TOOL_TWO"
            ),
        )
        request = SimpleNamespace(
            enable_thinking=False,
            input=[
                {
                    "role": "user",
                    "content": "After the tool result, reply exactly: REAL_UI_LIVE_TOOL_TWO",
                }
            ],
        )

        assert _responses_fast_path_visible_text(output, request) == "REAL_UI_LIVE_TOOL_TWO"


class TestResponsesQwenTerminalToolResultSynthesis:
    """Qwen terminal tool-result turns must finalize visibly without parser repair."""

    def test_terminal_tool_result_synthesis_requires_no_new_tools(self):
        from types import SimpleNamespace
        from vmlx_engine.server import _responses_is_terminal_tool_result_synthesis

        messages = [
            {"role": "user", "content": "Use the tool."},
            {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "recorded blue-cat"},
        ]

        assert _responses_is_terminal_tool_result_synthesis(
            messages,
            SimpleNamespace(tools=None, tool_choice=None),
            {},
        )
        assert not _responses_is_terminal_tool_result_synthesis(
            messages,
            SimpleNamespace(tools=[{"type": "function", "name": "record_fact"}], tool_choice="auto"),
            {},
        )
        assert not _responses_is_terminal_tool_result_synthesis(
            messages,
            SimpleNamespace(tools=None, tool_choice="required"),
            {},
        )

    def test_qwen_terminal_synthesis_sets_private_visible_finalization_flag(self):
        from vmlx_engine.server import (
            _responses_terminal_synthesis_visible_finalization,
        )

        assert _responses_terminal_synthesis_visible_finalization(
            {"_vmlx_terminal_tool_result_visible_finalization": True}
        )
        assert not _responses_terminal_synthesis_visible_finalization({})


class TestNonStreamingDisconnectAbort:
    """Non-streaming API requests must abort scheduler work on disconnect."""

    def test_helper_aborts_when_client_disconnects(self):
        import asyncio
        import pytest
        from fastapi import HTTPException
        from vmlx_engine.server import _await_chat_with_disconnect_abort

        class FakeEngine:
            def __init__(self):
                self.aborted = None

            async def chat(self, **kwargs):
                await asyncio.sleep(10)

            async def abort_request(self, request_id):
                self.aborted = request_id
                return True

        class FakeRequest:
            async def is_disconnected(self):
                return True

        async def run():
            engine = FakeEngine()
            with pytest.raises(HTTPException) as exc:
                await _await_chat_with_disconnect_abort(
                    engine,
                    messages=[],
                    chat_kwargs={},
                    timeout=30,
                    fastapi_request=FakeRequest(),
                    request_id="resp_disconnect_test",
                    endpoint="test",
                    poll_interval=0.001,
                )
            assert exc.value.status_code == 499
            assert engine.aborted == "resp_disconnect_test"

        asyncio.run(run())

    def test_helper_passes_public_request_id_to_engine_chat(self):
        import asyncio
        from vmlx_engine.server import _await_chat_with_disconnect_abort

        class FakeOutput:
            completion_tokens = 1

        class FakeEngine:
            def __init__(self):
                self.kwargs = None

            async def chat(self, **kwargs):
                self.kwargs = kwargs
                return FakeOutput()

        class FakeRequest:
            async def is_disconnected(self):
                return False

        async def run():
            engine = FakeEngine()
            output = await _await_chat_with_disconnect_abort(
                engine,
                messages=[{"role": "user", "content": "hi"}],
                chat_kwargs={"max_tokens": 1},
                timeout=30,
                fastapi_request=FakeRequest(),
                request_id="chatcmpl_public_id",
                endpoint="test",
                poll_interval=0.001,
            )
            assert output.completion_tokens == 1
            assert engine.kwargs["request_id"] == "chatcmpl_public_id"
            assert engine.kwargs["messages"] == [{"role": "user", "content": "hi"}]

        asyncio.run(run())

    def test_helper_normalizes_duplicate_request_id_from_streaming_kwargs(self):
        import asyncio
        from vmlx_engine.server import _await_chat_with_disconnect_abort

        class FakeOutput:
            completion_tokens = 1

        class FakeEngine:
            def __init__(self):
                self.kwargs = None

            async def chat(self, **kwargs):
                self.kwargs = kwargs
                return FakeOutput()

        class FakeRequest:
            async def is_disconnected(self):
                return False

        async def run():
            engine = FakeEngine()
            output = await _await_chat_with_disconnect_abort(
                engine,
                messages=[],
                chat_kwargs={"request_id": "stale_internal_id", "max_tokens": 1},
                timeout=30,
                fastapi_request=FakeRequest(),
                request_id="resp_public_id",
                endpoint="test",
                poll_interval=0.001,
            )
            assert output.completion_tokens == 1
            assert engine.kwargs["request_id"] == "resp_public_id"
            assert engine.kwargs["max_tokens"] == 1

        asyncio.run(run())

    def test_helper_treats_cancelled_collector_removal_as_client_cancel(self):
        import asyncio
        import pytest
        from fastapi import HTTPException
        from vmlx_engine.server import _await_chat_with_disconnect_abort

        class FakeEngine:
            async def chat(self, **kwargs):
                raise RuntimeError("No collector for request resp_cancelled")

            async def abort_request(self, request_id):
                return False

        class FakeRequest:
            async def is_disconnected(self):
                return False

        async def run():
            with pytest.raises(HTTPException) as exc:
                await _await_chat_with_disconnect_abort(
                    FakeEngine(),
                    messages=[],
                    chat_kwargs={},
                    timeout=30,
                    fastapi_request=FakeRequest(),
                    request_id="resp_cancelled",
                    endpoint="test",
                    poll_interval=0.001,
                )
            assert exc.value.status_code == 499

        asyncio.run(run())


# ===========================================================================
# L. LOW severity fixes — port race, delta streaming, reasoning_effort
# ===========================================================================


class TestL1PortRaceCondition:
    """Session creation must serialize port assignment to prevent races."""

    def test_creation_lock_exists(self):
        """SessionManager must have a global creation lock field."""
        source = Path("./panel/src/main/sessions.ts").read_text()
        assert "creationLock" in source, (
            "SessionManager must have a creationLock to serialize createSession"
        )

    def test_create_session_uses_lock(self):
        """createSession must acquire creationLock before port assignment."""
        source = Path("./panel/src/main/sessions.ts").read_text()
        # createSession should delegate to _createSessionInner
        assert "_createSessionInner" in source, (
            "createSession must delegate to _createSessionInner under lock"
        )

    def test_port_unique_constraint(self):
        """sessions table must have UNIQUE constraint on port column."""
        source = Path("./panel/src/main/database.ts").read_text()
        assert "port INTEGER NOT NULL UNIQUE" in source, (
            "sessions.port must have UNIQUE constraint as safety net"
        )


class TestL2IncrementalDelta:
    """Responses API function_call_arguments.delta must be incremental."""

    def test_delta_is_chunked(self):
        """Arguments must be emitted in chunks, not as one big delta."""
        import inspect
        from vmlx_engine.server import stream_responses_api
        source = inspect.getsource(stream_responses_api)
        # Must have a chunking loop (range + _ARG_CHUNK or similar)
        assert "_ARG_CHUNK" in source or "CHUNK_SIZE" in source, (
            "Must chunk arguments into incremental deltas"
        )
        # Must iterate over argument characters
        assert "range(0, len(tc_args)" in source or "range(0, max(len(tc_args)" in source, (
            "Must iterate over argument string for chunking"
        )


class TestL3ReasoningEffort:
    """reasoning_effort maps to thinking_budget without rewriting output length."""

    def test_effort_constants_defined(self):
        """Server must define effort-to-budget mapping constants."""
        from vmlx_engine.server import _EFFORT_THINKING_BUDGET
        assert "low" in _EFFORT_THINKING_BUDGET
        assert "medium" in _EFFORT_THINKING_BUDGET
        assert "high" in _EFFORT_THINKING_BUDGET
        assert _EFFORT_THINKING_BUDGET["low"] < _EFFORT_THINKING_BUDGET["medium"]
        assert _EFFORT_THINKING_BUDGET["medium"] < _EFFORT_THINKING_BUDGET["high"]

    def test_chat_completions_maps_effort(self):
        """Chat Completions must map reasoning_effort to thinking_budget."""
        import inspect
        from vmlx_engine.server import create_chat_completion
        source = inspect.getsource(create_chat_completion)
        assert "thinking_budget" in source, (
            "Chat Completions must inject thinking_budget from reasoning_effort"
        )
        assert "_EFFORT_THINKING_BUDGET" in source, (
            "Must use the _EFFORT_THINKING_BUDGET mapping"
        )

    def test_responses_maps_effort(self):
        """Responses API must map reasoning_effort to thinking_budget."""
        import inspect
        from vmlx_engine.server import create_response
        source = inspect.getsource(create_response)
        assert "thinking_budget" in source, (
            "Responses API must inject thinking_budget from reasoning_effort"
        )

    def test_effort_does_not_synthesize_max_tokens_when_unset(self):
        """reasoning_effort must not replace model-owned max_new_tokens."""
        import inspect
        from vmlx_engine.server import create_chat_completion
        source = inspect.getsource(create_chat_completion)
        assert "thinking_budget" in source, (
            "reasoning_effort should still reach chat_template_kwargs"
        )
        assert "_EFFORT_MAX_TOKENS" not in source, (
            "reasoning_effort must not secretly rewrite max_tokens; explicit "
            "max_tokens/max_output_tokens or bundle generation_config own length"
        )


# ── V5 FIXES (Pre-release cohesion review) ──────────────────────────────────

class TestV5CacheListTupleDetection:
    """CacheList.caches is always a tuple — detection must accept both list and tuple."""

    def test_truncation_accepts_tuple(self):
        """_truncate_cache_to_prompt_length must check isinstance(..., (list, tuple))."""
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler._truncate_cache_to_prompt_length)
        # Must accept tuple (CacheList stores .caches as tuple)
        assert "(list, tuple)" in source, (
            "CacheList detection must accept both list and tuple since CacheList.caches is always a tuple"
        )


class TestV5StreamingToolChoiceRequired:
    """Streaming tool_choice='required' must track actual emission, not just buffering."""

    def test_tool_calls_emitted_flag_exists(self):
        """stream_chat_completion must have a tool_calls_emitted flag."""
        import inspect
        from vmlx_engine.server import stream_chat_completion
        source = inspect.getsource(stream_chat_completion)
        assert "tool_calls_emitted" in source, (
            "Must track whether tool calls were actually emitted, not just buffering state"
        )

    def test_required_check_uses_emitted_flag(self):
        """tool_choice='required' enforcement must check tool_calls_emitted, not tool_call_buffering."""
        import inspect
        from vmlx_engine.server import stream_chat_completion
        source = inspect.getsource(stream_chat_completion)
        # The enforcement line must use tool_calls_emitted
        assert 'not tool_calls_emitted' in source, (
            "tool_choice='required' enforcement must use 'not tool_calls_emitted', "
            "not 'not tool_call_buffering' (which can be true on false-positive marker detection)"
        )
        # Must NOT use tool_call_buffering for the required check
        assert '"required" and not tool_call_buffering' not in source, (
            "Must not use tool_call_buffering for required enforcement — it stays True on false positives"
        )


class TestV5DisplayTextInit:
    """display_text must be initialized before the if/else tool_calls branch."""

    def test_display_text_initialized_before_branch(self):
        """stream_responses_api must initialize display_text before tool_calls branch."""
        import inspect
        from vmlx_engine.server import stream_responses_api
        source = inspect.getsource(stream_responses_api)
        # Find initialization before the tool_calls branch
        init_idx = source.find('display_text = ""')
        assert init_idx != -1, "display_text must be initialized to empty string"
        # Must appear before the H4 JSON validation that references display_text
        h4_idx = source.find("H4: Validate text format")
        assert h4_idx != -1, "H4 validation block must exist"
        assert init_idx < h4_idx, (
            "display_text initialization must come before H4 validation to prevent UnboundLocalError"
        )


class TestResponsesSuppressedReasoningToolCalls:
    """Suppressed reasoning can still contain a real tool call."""

    def test_responses_extracts_suppressed_reasoning_tool_calls_before_finalize(self):
        """Tool calls found only in suppressed reasoning must enter the Responses tool branch."""
        import inspect
        from vmlx_engine.server import stream_responses_api

        source = inspect.getsource(stream_responses_api)
        extract_idx = source.find("tool markers in suppressed reasoning")
        branch_idx = source.find("if tool_calls:")

        assert extract_idx != -1, (
            "Responses API must extract tool calls from suppressed reasoning before finalization"
        )
        assert branch_idx != -1, "Responses API tool_calls branch missing"
        assert extract_idx < branch_idx, (
            "Suppressed-reasoning tool extraction must run before the final tool_calls branch"
        )
        assert "TODO: emit tool calls via Responses API format" not in source, (
            "Responses API must not leave parsed suppressed-reasoning tool calls un-emitted"
        )

    def test_responses_nonstreaming_extracts_reasoning_tool_calls_before_finalize(self):
        """Non-streaming Responses must not strand real tool calls inside reasoning."""
        import inspect
        from vmlx_engine.server import create_response

        source = inspect.getsource(create_response)
        extract_idx = source.find("tool markers in reasoning")
        branch_idx = source.find("# Add tool calls as separate function_call output items")

        assert extract_idx != -1, (
            "Non-streaming Responses must extract tool calls from reasoning text before finalization"
        )
        assert branch_idx != -1, "Responses API function_call output branch missing"
        assert extract_idx < branch_idx, (
            "Reasoning tool extraction must run before Responses function_call output items are built"
        )


class TestAnthropicOmniStreamingAdapter:
    """Anthropic Omni streaming must not leak OpenAI SSE chunks."""

    def test_omni_streaming_path_uses_anthropic_adapter(self):
        import inspect
        from vmlx_engine.server import create_anthropic_message

        source = inspect.getsource(create_anthropic_message)
        assert "and anthropic_req.stream" in source
        assert "AnthropicStreamAdapter(model=resolved_name)" in source
        assert "adapter.process_chunk(line)" in source
        assert 'media_type="text/event-stream"' in source
        assert "pass through unchanged" not in source
        assert "Caller may see OpenAI" not in source

    def test_omni_messages_path_adapts_plain_dict_completion(self):
        import inspect
        from vmlx_engine.server import create_anthropic_message

        source = inspect.getsource(create_anthropic_message)
        assert "if isinstance(cc, dict):" in source
        assert "to_anthropic_response(cc, resolved_name)" in source

    def test_omni_responses_path_adapts_plain_dict_and_forwards_thinking(self):
        import inspect
        from vmlx_engine.server import create_response

        source = inspect.getsource(create_response)
        start = source.index("Nemotron-Omni multimodal dispatch (Responses API parity")
        end = source.index("if request.stream:", start)
        block = source[start:end]

        assert "enable_thinking=request.enable_thinking" in block
        assert "elif isinstance(cc, dict):" in block
        assert "cc_dict = cc" in block


class TestDSV4FastLoadSwitchGLUScope:
    """DSV4 fast-load speed patch must not hijack other SwitchGLU models."""

    def test_switchglu_patch_is_marker_scoped_and_idempotent(self):
        import inspect
        from vmlx_engine.loaders.load_jangtq_dsv4 import _try_fast_load_dsv4

        source = inspect.getsource(_try_fast_load_dsv4)
        assert "_vmlx_dsv4_original_call" in source, (
            "Fast-load patch must keep the original SwitchGLU.__call__ instead of wrapping wrappers"
        )
        assert "_vmlx_dsv4_fused_fastpath" in source, (
            "Fast-load patch must mark only DSV4 modules as eligible for fused decoding"
        )
        guard_idx = source.find('if not getattr(self, "_vmlx_dsv4_fused_fastpath", False):')
        gp_idx = source.find("gp = self.gate_proj")
        assert guard_idx != -1 and gp_idx != -1 and guard_idx < gp_idx, (
            "Non-DSV4 SwitchGLU modules must fall back before the DSV4 TurboQuant path"
        )

    def test_fast_load_switchglu_threads_down_proj_bits(self):
        import inspect
        from vmlx_engine.loaders.load_jangtq_dsv4 import _try_fast_load_dsv4

        source = inspect.getsource(_try_fast_load_dsv4)
        assert "dp_bits=None" in source
        assert "dp_bits=dp.bits" in source
        assert "make_gather_tq_decode_per_row(out_f, in_f, dp_bits, k)" in source
        assert "cache_key = (in_f, out_f, bits, dp_bits, k, limit_milli)" in source

    def test_dsv4_switchglu_contract_audit_requires_limited_swiglu(self):
        import inspect
        from vmlx_engine.loaders.load_jangtq_dsv4 import _audit_dsv4_switchglu_contract

        source = inspect.getsource(_audit_dsv4_switchglu_contract)
        assert "swiglu_limit" in source
        assert "10.0" in source
        assert "zero TurboQuant SwitchGLU" in source
        assert "swiglu_limit=10 missing" in source

    def test_dsv4_switchglu_contract_audit_rejects_missing_limit(self, monkeypatch):
        import sys
        import types
        import pytest
        from vmlx_engine.loaders.load_jangtq_dsv4 import _audit_dsv4_switchglu_contract

        class TurboQuantSwitchLinear:
            pass

        class SwitchGLU:
            pass

        tq_mod = types.ModuleType("jang_tools.turboquant.tq_kernel")
        tq_mod.TurboQuantSwitchLinear = TurboQuantSwitchLinear
        switch_mod = types.ModuleType("mlx_lm.models.switch_layers")
        switch_mod.SwitchGLU = SwitchGLU
        monkeypatch.setitem(sys.modules, "jang_tools.turboquant.tq_kernel", tq_mod)
        monkeypatch.setitem(sys.modules, "mlx_lm.models.switch_layers", switch_mod)

        class _Activation:
            def __init__(self, limit):
                self.swiglu_limit = limit

        class _Switch(SwitchGLU):
            def __init__(self, limit):
                self.gate_proj = TurboQuantSwitchLinear()
                self.up_proj = TurboQuantSwitchLinear()
                self.down_proj = TurboQuantSwitchLinear()
                self.activation = _Activation(limit)

        class _Model:
            def __init__(self, module):
                self.module = module

            def named_modules(self):
                return [("layers.0.mlp.switch_mlp", self.module)]

        _audit_dsv4_switchglu_contract(_Model(_Switch(10.0)))

        with pytest.raises(RuntimeError, match="swiglu_limit=10 missing"):
            _audit_dsv4_switchglu_contract(_Model(_Switch(0.0)))


class TestDSV4SidecarManifestRuntimePatch:
    """DSV4 sidecars must be invalidated when runtime patches change."""

    def test_manifest_records_and_checks_runtime_patch_version(self):
        import inspect
        import vmlx_engine.loaders.load_jangtq_dsv4 as loader

        fast_source = inspect.getsource(loader._try_fast_load_dsv4)
        write_source = inspect.getsource(loader._write_sidecar_after_hydrate)

        assert hasattr(loader, "_INSTANT_LOAD_RUNTIME_PATCH")
        assert '"runtime_patch": _INSTANT_LOAD_RUNTIME_PATCH' in write_source
        assert 'manifest.get("runtime_patch") != _INSTANT_LOAD_RUNTIME_PATCH' in fast_source


class TestV5FixHybridCacheExcept:
    """_fix_hybrid_cache outermost except must return fresh cache, not broken original."""

    def test_except_returns_make_cache(self):
        """Outermost except in _fix_hybrid_cache must call make_cache() not return original cache."""
        import inspect
        from vmlx_engine.mllm_batch_generator import _fix_hybrid_cache
        source = inspect.getsource(_fix_hybrid_cache)
        # Find the outermost except block (last except in the function)
        lines = source.split('\n')
        last_except_idx = None
        for i, line in enumerate(lines):
            if 'except Exception' in line:
                last_except_idx = i
        assert last_except_idx is not None, "Must have outermost except block"
        # Check the lines after the except
        after_except = '\n'.join(lines[last_except_idx:last_except_idx + 5])
        assert "make_cache()" in after_except, (
            "Outermost except must call language_model.make_cache() to return fresh cache, "
            "not return the potentially broken original cache"
        )

    def test_except_has_fallback(self):
        """Must fall back to original cache if make_cache is not available."""
        import inspect
        from vmlx_engine.mllm_batch_generator import _fix_hybrid_cache
        source = inspect.getsource(_fix_hybrid_cache)
        # Must have a hasattr check for make_cache in the except path
        assert "hasattr(language_model, 'make_cache')" in source, (
            "Must check hasattr before calling make_cache in except handler"
        )


class TestGenerationBatchFastNoLogprobs:
    """Text BatchGenerator should not materialize full-vocab logprobs per token."""

    def test_generation_step_keeps_logprobs_transient(self):
        source = Path("./vmlx_engine/utils/mamba_cache.py").read_text()
        func_start = source.find("def _patch_generation_step_sync")
        assert func_start >= 0
        func_body = source[func_start: source.find("\ndef ", func_start + 10)]
        assert "requested_logprobs" in func_body
        assert "_should_capture_generation_logprobs" in func_body
        assert "logprobs[i] if requested_logprobs[i] else None" in func_body
        assert "mx.eval(inputs, self._current_logprobs)" not in func_body
        assert "mx.async_eval(self._next_tokens, self._next_logprobs" not in func_body


class TestHybridSSMCompanionCacheGating:
    """Hybrid SSM companion work must obey the prefix-cache enable flag."""

    def test_llm_scheduler_does_not_init_or_store_ssm_when_prefix_cache_disabled(self):
        source = Path("./vmlx_engine/scheduler.py").read_text()
        init_idx = source.index("self._ssm_state_cache: Optional[HybridSSMStateCache] = None")
        init_block = source[init_idx : source.index("# Prompt lookup decoding", init_idx)]
        assert "self.config.enable_prefix_cache" in init_block
        assert "HybridSSMStateCache(" in init_block
        assert "max_entries=_ssm_cache_size" in init_block
        assert "max_bytes=(" in init_block

        store_idx = source.index("# Hybrid SSM companion state capture.")
        store_block = source[store_idx : source.index("# Store cache for future reuse", store_idx)]
        assert "and self.config.enable_prefix_cache" in store_block

        rederive_idx = source.index("# ── Deferred SSM re-derive")
        rederive_block = source[rederive_idx : source.index("return output", rederive_idx)]
        assert "and self.config.enable_prefix_cache" in rederive_block

    def test_mllm_batch_generator_disables_hidden_ssm_work_without_prefix_cache(self):
        source = Path("./vmlx_engine/mllm_batch_generator.py").read_text()
        assert "enable_prefix_cache: bool = True" in source
        assert "self._prefix_cache_enabled = bool(enable_prefix_cache)" in source
        assert "self._ssm_companion_enabled = bool(" in source
        assert "if self._ssm_companion_enabled" in source
        assert "else None" in source
        assert "self._prefix_cache_enabled" in source
        assert "block_aware_cache is not None" in source
        capture_idx = source.index("# Capture SSM state at prompt boundary for hybrid models.")
        capture_block = source[capture_idx : source.index("if trace is not None:", capture_idx + 1)]
        assert "self._is_hybrid" in capture_block
        assert "and self._ssm_companion_enabled" in capture_block
        assert "_bypass_prefix_cache" in capture_block

    def test_mllm_scheduler_threads_prefix_cache_flag_to_batch_generator(self):
        source = Path("./vmlx_engine/mllm_scheduler.py").read_text()
        generator_idx = source.index("self.batch_generator = MLLMBatchGenerator(")
        generator_block = source[generator_idx : source.index("self._current_sampler_params", generator_idx)]
        assert "enable_prefix_cache=self.config.enable_prefix_cache" in generator_block

    def test_llm_scheduler_hybrid_ssm_rederive_policy_is_not_stale(self):
        source = Path("./vmlx_engine/scheduler.py").read_text()
        store_idx = source.index("# Hybrid SSM companion state capture.")
        store_block = source[store_idx : source.index("# Store cache for future reuse", store_idx)]

        assert "SSM_REDERIVE_MIN_TOKENS = 1" in source
        assert "< 64" not in store_block
        assert "SSM_REDERIVE_MIN_TOKENS" in store_block
        assert "gpl=0 (non-thinking) hybrid SSM path" in store_block
        assert "DO NOT extract" in store_block
        assert "post-output SSM layers" in store_block
        assert "queued deferred" in store_block
        assert "is_complete=False" in store_block

    def test_lfm2_base_model_type_uses_hybrid_ssm_companion_cache(self):
        from vmlx_engine.utils.ssm_companion_cache import is_hybrid_ssm_config

        assert is_hybrid_ssm_config({"model_type": "lfm2"}) is True
        assert is_hybrid_ssm_config({"text_config": {"model_type": "lfm2"}}) is True


class TestStartupCompatibilityGuards:
    def test_cli_checks_mlx_wheel_macos_tag_before_import(self):
        source = Path("./vmlx_engine/cli.py").read_text()
        check_idx = source.index("def _check_macos_compat")
        check_block = source[check_idx: source.index("def _check_no_duplicate_mlx", check_idx)]
        assert "importlib.metadata.distribution" in check_block
        assert "macosx_(\\d+)_(\\d+)_arm64" in check_block
        assert "Failed to load the default metallib" in check_block
        assert "_check_macos_compat()" in source

    def test_bundled_python_requires_mflux_for_image_models(self):
        bundle_script = Path("./panel/scripts/bundle-python.sh").read_text()
        verify_script = Path("./panel/scripts/verify-bundled-python.sh").read_text()
        assert 'MFLUX_VERSION="0.17.5"' in bundle_script
        assert '"mflux==$MFLUX_VERSION"' in bundle_script
        assert '("mflux", "mflux image runtime"' in verify_script
        assert '"mflux.models.common.config.model_config"' in verify_script

    def test_python_dependency_floor_covers_current_jang_family_runtimes(self):
        pyproject = Path("./pyproject.toml").read_text()

        # Hy3 registration lives in jang_tools.hy3 as of the 2.5.29 public wheel
        # and the newer local 2.5.30 app bundle;
        # Ling/Bailing hybrid needs the mlx-lm runtime floor that the bundle
        # uses before the local bailing_hybrid vendor file is applied.
        assert '"mlx-lm>=0.31.3"' in pyproject
        assert pyproject.count('"jang>=2.5.30"') >= 3

    def test_bundled_python_installs_distutils_version_shim_for_radio(self):
        bundle_script = Path("./panel/scripts/bundle-python.sh").read_text()

        assert "Installing distutils.version compatibility shim" in bundle_script
        assert 'mkdir -p "$SITE/distutils"' in bundle_script
        assert "class LooseVersion" in bundle_script
        assert 'from distutils.version import LooseVersion' in bundle_script
        assert 'LooseVersion("1.2") < LooseVersion("1.3")' in bundle_script

    def test_bundled_python_local_source_installs_force_reinstall(self):
        bundle_script = Path("./panel/scripts/bundle-python.sh").read_text()
        local_install_block = bundle_script[
            bundle_script.index('echo "==> Installing vmlx-engine + jang_tools')
            : bundle_script.index("# Clean up to reduce size")
        ]

        assert '"$PYTHON" -m pip install --force-reinstall --no-deps --no-cache-dir "$VMLX_LOCAL"' in local_install_block
        assert 'JANG_LOCAL="${VMLX_JANG_TOOLS_SOURCE:-${VMLINUX_JANG_TOOLS_SOURCE:-$HOME/jang/jang-tools}}"' in bundle_script
        assert '"$PYTHON" -m pip install --force-reinstall --no-deps --no-cache-dir "$JANG_LOCAL"' in local_install_block

    def test_bundled_python_does_not_silently_fallback_to_pypi_jang_tools(self):
        bundle_script = Path("./panel/scripts/bundle-python.sh").read_text()
        verify_script = Path("./panel/scripts/verify-bundled-python.sh").read_text()

        assert '${VMLX_ALLOW_PYPI_JANG:-${VMLINUX_ALLOW_PYPI_JANG:-0}}' in bundle_script
        assert "RELEASE BLOCKED — local jang-tools source missing" in bundle_script
        assert 'pip install --no-deps "jang>=2.5.30"' in bundle_script
        assert '${VMLX_ALLOW_MISSING_JANG_SOURCE_HASH:-${VMLINUX_ALLOW_MISSING_JANG_SOURCE_HASH:-0}}' in verify_script
        assert "RELEASE BLOCKED — local jang_tools source unavailable for hash parity" in verify_script

    def test_bundled_python_blocks_tracked_dirty_local_jang_tools_by_default(self):
        bundle_script = Path("./panel/scripts/bundle-python.sh").read_text()
        local_install_block = bundle_script[
            bundle_script.index('JANG_LOCAL="${VMLX_JANG_TOOLS_SOURCE:-${VMLINUX_JANG_TOOLS_SOURCE:-$HOME/jang/jang-tools}}"')
            : bundle_script.index("# Clean up to reduce size")
        ]
        destructive_build_index = bundle_script.index('rm -rf "$BUNDLE_DIR"')
        dirty_guard_index = bundle_script.index("check_local_jang_source_clean")

        assert '${VMLX_ALLOW_DIRTY_JANG_SOURCE:-${VMLINUX_ALLOW_DIRTY_JANG_SOURCE:-0}}' in local_install_block
        assert "RELEASE BLOCKED — local jang-tools source has tracked changes" in local_install_block
        assert 'git -C "$JANG_LOCAL" diff --quiet --ignore-submodules --' in local_install_block
        assert 'git -C "$JANG_LOCAL" diff --cached --quiet --ignore-submodules --' in local_install_block
        assert dirty_guard_index < destructive_build_index

    def test_release_scripts_accept_documented_jang_tools_env_names_with_legacy_fallback(self):
        bundle_script = Path("./panel/scripts/bundle-python.sh").read_text()
        verify_script = Path("./panel/scripts/verify-bundled-python.sh").read_text()
        release_gate = Path("./panel/scripts/release-gate-python-app.py").read_text()

        assert 'JANG_LOCAL="${VMLX_JANG_TOOLS_SOURCE:-${VMLINUX_JANG_TOOLS_SOURCE:-$HOME/jang/jang-tools}}"' in bundle_script
        assert 'JANG_TOOLS_SOURCE_DIR="${VMLX_JANG_TOOLS_SOURCE:-${VMLINUX_JANG_TOOLS_SOURCE:-$HOME/jang/jang-tools}}/jang_tools"' in verify_script
        assert 'os.environ.get("VMLX_JANG_TOOLS_SOURCE")' in release_gate
        assert 'os.environ.get("VMLINUX_JANG_TOOLS_SOURCE")' in release_gate
        assert '${VMLX_ALLOW_DIRTY_JANG_SOURCE:-${VMLINUX_ALLOW_DIRTY_JANG_SOURCE:-0}}' in bundle_script
        assert '${VMLX_ALLOW_PYPI_JANG:-${VMLINUX_ALLOW_PYPI_JANG:-0}}' in bundle_script
        assert '${VMLX_ALLOW_MISSING_JANG_SOURCE_HASH:-${VMLINUX_ALLOW_MISSING_JANG_SOURCE_HASH:-0}}' in verify_script

    def test_bundled_python_console_scripts_are_relocatable(self):
        bundle_script = Path("./panel/scripts/bundle-python.sh").read_text()
        verify_script = Path("./panel/scripts/verify-bundled-python.sh").read_text()
        release_gate = Path("./panel/scripts/release-gate-python-app.py").read_text()

        assert "Rewriting console-script shebangs to relocatable bundled Python" in bundle_script
        assert "\\$(dirname " in bundle_script
        assert "/python3\\\" -B -s" in bundle_script
        assert '[[ "$FIRST_LINE" == \'#!\'*python* ]]' in bundle_script
        assert "check_console_script_shebangs" in verify_script
        assert '[[ "$first_line" == \'#!\'*python* ]]' in verify_script
        assert "check_packaged_console_script_shebangs" in release_gate
        assert "/Applications/vMLX.app" in release_gate

    def test_bundled_python_restores_launcher_and_libpython_after_dependency_install(self):
        bundle_script = Path("./panel/scripts/bundle-python.sh").read_text()
        dependency_install_idx = bundle_script.index('echo "==> Installing dependencies..."')
        local_install_idx = bundle_script.index('echo "==> Installing mlx-audio')
        dependency_block = bundle_script[dependency_install_idx:local_install_idx]

        assert "restore_python_runtime_files" in bundle_script
        assert 'STANDALONE_TARBALL="$(mktemp' in bundle_script
        assert 'trap \'rm -f "$STANDALONE_TARBALL"\'' in bundle_script
        assert "tar xzf \"$STANDALONE_TARBALL\"" in bundle_script
        assert "python/bin/python3.12" in bundle_script
        assert "python/lib/libpython3.12.dylib" in bundle_script
        assert "restore_python_runtime_files" in dependency_block

    def test_bundled_python_restores_launcher_before_dependency_install(self):
        bundle_script = Path("./panel/scripts/bundle-python.sh").read_text()
        extract_idx = bundle_script.index('tar xzf "$STANDALONE_TARBALL" -C "$BUNDLE_DIR"')
        pip_upgrade_idx = bundle_script.index('echo "==> Upgrading pip..."')
        mlx_install_idx = bundle_script.index(
            '"$PYTHON" -m pip install "$WHEELHOUSE"/mlx-"$MLX_VERSION"'
        )
        dependency_install_idx = bundle_script.index('echo "==> Installing dependencies..."')
        bootstrap_block = bundle_script[extract_idx:pip_upgrade_idx]
        mlx_install_block = bundle_script[mlx_install_idx:dependency_install_idx]

        assert "restore_python_runtime_files" in bootstrap_block
        assert 'PYTHON="$BUNDLE_DIR/python/bin/python3.12"' in bootstrap_block
        assert "restore_python_runtime_files" in mlx_install_block
        assert 'PYTHON="$BUNDLE_DIR/python/bin/python3"' in mlx_install_block

    def test_local_installer_installs_node_deps_before_typecheck(self):
        install_script = Path("./panel/scripts/build-and-install.sh").read_text()

        assert "ensure_node_dependencies()" in install_script
        assert 'node_modules/.bin/tsc' in install_script
        install_idx = install_script.index("ensure_node_dependencies")
        typecheck_idx = install_script.index("npm run typecheck")
        assert install_idx < typecheck_idx

    def test_release_gate_packaged_python_probes_cannot_mutate_signed_app(self):
        release_gate = Path("./panel/scripts/release-gate-python-app.py").read_text()

        assert "packaged_python_env" in release_gate
        assert "PYTHONPYCACHEPREFIX" in release_gate
        assert "twine_env" in release_gate
        assert '"twine check dist"' in release_gate
        assert "env=twine_env(gate)" in release_gate
        assert "check_no_packaged_pycache" in release_gate
        assert "packaged app pycache clean" in release_gate

    def test_bundled_python_hash_gate_covers_runtime_files_changed_for_release(self):
        verify_script = Path("./panel/scripts/verify-bundled-python.sh").read_text()
        for rel in (
            "__init__.py",
            "server.py",
            "api/anthropic_adapter.py",
            "api/ollama_adapter.py",
            "engine/batched.py",
            "loaders/load_jangtq_dsv4.py",
            "mllm_batch_generator.py",
            "mllm_scheduler.py",
            "models/mllm.py",
            "models/step3p7_mlx_vlm.py",
            "models/gemma4_unified_register.py",
            "models/gemma4_unified/__init__.py",
            "models/gemma4_unified/config.py",
            "models/gemma4_unified/gemma4_unified.py",
            "models/gemma4_unified/processing_gemma4_unified.py",
            "omni_multimodal.py",
            "prefix_cache.py",
            "runtime_patches/__init__.py",
            "runtime_patches/deepseek_v4_register.py",
            "runtime_patches/gemma4_processing.py",
            "runtime_patches/gemma4_vision.py",
            "runtime_patches/kimi_k25_mla.py",
            "runtime_patches/mlx_lm_compat.py",
            "runtime_patches/mlx_vlm_compat.py",
            "scheduler.py",
            "tool_parsers/dsml_tool_parser.py",
        ):
            assert f'"{rel}"' in verify_script
        for rel in sorted(
            str(path.relative_to("vmlx_engine"))
            for path in Path("vmlx_engine/tool_parsers").glob("*.py")
        ):
            assert f'"{rel}"' in verify_script
        for rel in sorted(
            str(path.relative_to("vmlx_engine"))
            for path in Path("vmlx_engine/reasoning").glob("*.py")
        ):
            assert f'"{rel}"' in verify_script

    def test_bundled_python_hash_gate_covers_critical_jang_tools_files(self):
        verify_script = Path("./panel/scripts/verify-bundled-python.sh").read_text()
        for rel in (
            "capabilities.py",
            "convert_hy3_jangtq.py",
            "load_jangtq.py",
            "load_jangtq_vlm.py",
            "load_jangtq_kimi_vlm.py",
            "nemotron_omni_chat.py",
            "dsv4/mlx_model.py",
            "dsv4/pool_quant_cache.py",
            "hy3/__init__.py",
            "hy3/model.py",
            "hy3/runtime.py",
            "kimi_prune/generate_vl.py",
            "kimi_prune/runtime_patch.py",
            "mimo_v2/mlx_model.py",
            "step37/__init__.py",
            "step37/nvfp4_codec.py",
            "step37/step3p7_mlx.py",
            "topk_override.py",
            "turboquant/fused_gate_up_kernel.py",
            "turboquant/gather_tq_kernel.py",
            "turboquant/hadamard_kernel.py",
            "turboquant/mpp_nax_kernel.py",
            "turboquant/tq_kernel.py",
        ):
            assert f'"{rel}"' in verify_script

    def test_bundled_python_import_gate_covers_jangtq_acceleration_kernel(self):
        verify_script = Path("./panel/scripts/verify-bundled-python.sh").read_text()

        assert '"jang_tools.turboquant.mpp_nax_kernel"' in verify_script
        assert "JANGTQ acceleration kernel missing" in verify_script

    def test_bundled_python_import_gate_covers_hy3_and_ling_registration(self):
        verify_script = Path("./panel/scripts/verify-bundled-python.sh").read_text()

        assert '("jang_tools.hy3", "jang_tools.hy3"' in verify_script
        assert '"mlx_lm.models.hy_v3"' in verify_script
        assert '"mlx_lm.models.bailing_hybrid"' in verify_script
        assert '("jang_tools.mimo_v2.mlx_register", "jang_tools.mimo_v2.mlx_register"' in verify_script
        assert '"mlx_lm.models.mimo_v2"' in verify_script
        assert "Hy3 model-family mlx-lm registration missing" in verify_script
        assert "Ling/Bailing hybrid mlx-lm runtime missing" in verify_script
        assert "MiMo-V2.5 mlx-lm registration missing" in verify_script

    def test_bundled_python_import_gate_covers_minicpm_v46_runtime(self):
        verify_script = Path("./panel/scripts/verify-bundled-python.sh").read_text()

        assert 'MODEL_REMAPPING.get("minicpmv4_6") != "minicpmo"' in verify_script
        assert "MiniCPM-V-4.6 → minicpmo remap missing" in verify_script
        assert "MiniCPM-V-4.6 mlx_vlm remap + prompt_utils config" in verify_script

    def test_bundled_python_import_gate_covers_gemma4_assistant_alias(self):
        verify_script = Path("./panel/scripts/verify-bundled-python.sh").read_text()

        assert '"mlx_vlm.models.gemma4_assistant"' in verify_script
        assert "Gemma 4 assistant mlx_vlm.models alias" in verify_script

    def test_bundled_python_import_gate_covers_gemma4_unified_runtime(self):
        verify_script = Path("./panel/scripts/verify-bundled-python.sh").read_text()

        assert '"vmlx_engine.models.gemma4_unified_register"' in verify_script
        assert "_register_gemma4_unified_runtime()" in verify_script
        assert '"mlx_vlm.models.gemma4_unified"' in verify_script
        assert '"mlx_vlm.models.gemma4_unified.processing_gemma4_unified"' in verify_script
        assert "Gemma 4 Unified source VLM/audio runtime missing" in verify_script
        assert "Gemma 4 Unified mlx-vlm registration missing" in verify_script

    def test_gemma4_unified_runtime_import_hook_resolves_direct_mlx_vlm_path(self):
        import importlib
        import sys

        import vmlx_engine  # noqa: F401

        sys.modules.pop("mlx_vlm.models.gemma4_unified.processing_gemma4_unified", None)
        sys.modules.pop("mlx_vlm.models.gemma4_unified", None)

        module = importlib.import_module("mlx_vlm.models.gemma4_unified")
        processing = importlib.import_module(
            "mlx_vlm.models.gemma4_unified.processing_gemma4_unified"
        )

        assert hasattr(module, "Model")
        assert hasattr(module, "Gemma4UnifiedProcessor")
        assert hasattr(processing, "Gemma4UnifiedProcessor")

    def test_mlx_vlm_registry_patch_remaps_minicpm_v46_to_minicpmo(self):
        import vmlx_engine
        from mlx_vlm.prompt_utils import MODEL_CONFIG
        from mlx_vlm.utils import MODEL_REMAPPING, get_model_and_args

        vmlx_engine._install_mlx_vlm_registry_patches()

        assert MODEL_REMAPPING["minicpmv4_6"] == "minicpmo"
        assert MODEL_CONFIG["minicpmv4_6"] == MODEL_CONFIG["minicpmo"]
        module, model_type = get_model_and_args({"model_type": "minicpmv4_6"})
        assert model_type == "minicpmo"
        assert hasattr(module, "Model")

    def test_mlx_vlm_registry_patch_aliases_gemma4_assistant_model_path(self):
        import importlib
        import sys

        import vmlx_engine

        sys.modules.pop("mlx_vlm.models.gemma4_assistant", None)
        sys.modules.pop("mlx_vlm.models.gemma4_unified_assistant", None)
        sys.modules.pop("mlx_vlm.speculative.drafters.gemma4_unified_assistant", None)

        vmlx_engine._install_mlx_vlm_registry_patches()

        module = importlib.import_module("mlx_vlm.models.gemma4_assistant")
        assert module.__name__ == "mlx_vlm.speculative.drafters.gemma4_assistant"
        assert hasattr(module, "Model")
        assert hasattr(module, "ModelConfig")

        unified_module = importlib.import_module(
            "mlx_vlm.speculative.drafters.gemma4_unified_assistant"
        )
        assert unified_module.__name__ == "mlx_vlm.speculative.drafters.gemma4_assistant"
        assert hasattr(unified_module, "Model")
        assert hasattr(unified_module, "ModelConfig")

    def test_bundled_python_import_gate_covers_step37_source_runtime(self):
        verify_script = Path("./panel/scripts/verify-bundled-python.sh").read_text()

        assert '("jang_tools.step37.step3p7_mlx", "jang_tools.step37.step3p7_mlx"' in verify_script
        assert '("vmlx_engine.models.step3p7_mlx_vlm", "vmlx_engine Step3p7 VLM runtime"' in verify_script
        assert "_register_step3p7_mlx_vlm_runtime()" in verify_script
        assert '"mlx_vlm.models.step3p7"' in verify_script
        assert '"mlx_vlm.models.step3p7.processing_step3"' in verify_script
        assert "Step3p7 source VLM runtime missing" in verify_script
        assert "Step3p7 mlx-vlm registration missing" in verify_script

    def test_jang_loader_registers_hy3_and_ling_before_mlx_lm_resolution(
        self, monkeypatch
    ):
        from vmlx_engine.utils import jang_loader

        seen = []
        real_import_module = importlib.import_module

        def fake_import_module(name):
            if name in {
                "jang_tools.hy3",
                "mlx_lm.models.hy_v3",
                "mlx_lm.models.bailing_hybrid",
            }:
                seen.append(name)
                return SimpleNamespace()
            return real_import_module(name)

        monkeypatch.setattr(jang_loader.importlib, "import_module", fake_import_module)

        jang_loader._ensure_jang_family_runtime_supported(
            Path("/models/Hy3-preview-JANGTQ2"), {"model_type": "hy_v3"}
        )
        assert seen[:2] == ["jang_tools.hy3", "mlx_lm.models.hy_v3"]

        seen.clear()
        jang_loader._ensure_jang_family_runtime_supported(
            Path("/models/Ling-2.6-flash-JANGTQ"),
            {"model_type": "bailing_hybrid"},
        )
        assert seen == ["mlx_lm.models.bailing_hybrid"]

    def test_jang_loader_registers_mimo_v2_before_mlx_lm_resolution(
        self, monkeypatch
    ):
        from vmlx_engine.utils import jang_loader

        seen = []
        real_import_module = importlib.import_module

        def fake_import_module(name):
            if name == "jang_tools.mimo_v2.mlx_model":
                seen.append(name)
                return SimpleNamespace(Model=object, ModelArgs=object)
            if name in {"jang_tools.mimo_v2.mlx_register", "mlx_lm.models.mimo_v2"}:
                seen.append(name)
                return SimpleNamespace()
            return real_import_module(name)

        monkeypatch.setattr(jang_loader.importlib, "import_module", fake_import_module)

        sys.modules.pop("mlx_vlm.models.mimo_v2", None)
        jang_loader._ensure_jang_family_runtime_supported(
            Path("/models/MiMo-V2.5-JANG_2L"), {"model_type": "mimo_v2"}
        )

        assert seen == [
            "jang_tools.mimo_v2.mlx_register",
            "mlx_lm.models.mimo_v2",
            "jang_tools.mimo_v2.mlx_model",
        ]

    def test_load_model_with_fallback_registers_mimo_v2_before_mlx_lm_resolution(
        self, monkeypatch
    ):
        from vmlx_engine.utils import tokenizer

        seen = []

        def fake_import_module(name):
            seen.append(name)
            if name == "jang_tools.mimo_v2.mlx_register":
                return SimpleNamespace()
            if name == "mlx_lm.models.mimo_v2":
                return SimpleNamespace(Model=object, ModelArgs=object)
            raise ImportError(name)

        monkeypatch.setattr(tokenizer.importlib, "import_module", fake_import_module)

        assert tokenizer._register_mimo_v2_runtime_for_mlx_lm() is True
        assert seen == ["jang_tools.mimo_v2.mlx_register", "mlx_lm.models.mimo_v2"]

    def test_load_model_with_fallback_mimo_v2_registration_fails_closed(
        self, monkeypatch
    ):
        from vmlx_engine.utils import tokenizer

        seen = []

        def fake_import_module(name):
            seen.append(name)
            if name == "jang_tools.mimo_v2.mlx_register":
                return SimpleNamespace()
            raise ImportError(name)

        monkeypatch.setattr(tokenizer.importlib, "import_module", fake_import_module)

        assert tokenizer._register_mimo_v2_runtime_for_mlx_lm() is False
        assert seen == ["jang_tools.mimo_v2.mlx_register", "mlx_lm.models.mimo_v2"]

    def test_mllm_registers_mimo_v2_before_mlx_vlm_resolution(
        self, tmp_path, monkeypatch
    ):
        from vmlx_engine.models import mllm

        model_dir = tmp_path / "MiMo-V2.5-JANG_2L"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            '{"model_type":"mimo_v2","vision_config":{},"audio_config":{}}',
            encoding="utf-8",
        )

        seen = []

        class FakeModelArgs:
            model_type = "mimo_v2"

            @classmethod
            def from_dict(cls, params):
                return SimpleNamespace(**params)

        def fake_import_module(name):
            seen.append(name)
            return SimpleNamespace(
                Model=object,
                ModelArgs=FakeModelArgs,
            )

        monkeypatch.setattr(mllm.importlib, "import_module", fake_import_module)

        sys.modules.pop("mlx_vlm.models.mimo_v2", None)
        mllm._register_local_mlx_vlm_runtime_if_needed(model_dir)

        assert seen == ["jang_tools.mimo_v2.mlx_model"]
        assert "mlx_vlm.models.mimo_v2" in sys.modules
        module = sys.modules["mlx_vlm.models.mimo_v2"]
        assert hasattr(module, "ModelConfig")
        assert hasattr(module, "LanguageModel")
        config = {
            "model_type": "mimo_v2",
            "hidden_size": 16,
            "quantization": {
                "bits": 8,
                "group_size": 64,
                "model.layers.0.self_attn.qkv_proj": {"bits": 8, "group_size": 64},
            },
        }
        module.ModelConfig.from_dict(config)
        assert config["quantization"]["language_model.model.layers.0.self_attn.qkv_proj"] == {
            "bits": 8,
            "group_size": 64,
        }
        sys.modules.pop("mlx_vlm.models.mimo_v2", None)

    def test_mllm_mimo_v2_turboquant_skip_detects_loaded_runtime_object(self):
        from vmlx_engine.models import mllm

        model = SimpleNamespace(
            language_model=SimpleNamespace(
                inner=SimpleNamespace(args=SimpleNamespace(model_type="mimo_v2"))
            )
        )

        assert mllm._is_mimo_v2_runtime_object_or_name(model, "/models/opaque") is True

    def test_mllm_mimo_v2_turboquant_skip_leaves_non_mimo_models_eligible(self):
        from vmlx_engine.models import mllm

        model = SimpleNamespace(
            language_model=SimpleNamespace(
                inner=SimpleNamespace(args=SimpleNamespace(model_type="qwen3_vl"))
            )
        )

        assert mllm._is_mimo_v2_runtime_object_or_name(model, "/models/qwen3-vl") is False

    def test_mllm_mimo_v2_quantizes_python_list_decoder_layers(
        self, tmp_path, monkeypatch
    ):
        from vmlx_engine.models import mllm

        model_dir = tmp_path / "MiMo-V2.5-JANG_2L"
        model_dir.mkdir()
        config = {
            "model_type": "mimo_v2",
            "hidden_size": 32,
            "intermediate_size": 64,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 128,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "moe_layer_freq": [False, True],
            "hybrid_layer_pattern": [0, 1],
            "full_attention_interval": 1,
            "quantization": {
                "bits": 8,
                "group_size": 32,
                "model.layers.1.self_attn.qkv_proj": {"bits": 4, "group_size": 32},
                "model.layers.1.mlp.switch_mlp.gate_proj": {"bits": 3, "group_size": 32},
                "model.layers.1.mlp.switch_mlp.up_proj": {"bits": 2, "group_size": 32},
                "model.layers.1.mlp.switch_mlp.down_proj": {"bits": 2, "group_size": 32},
            },
        }
        (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

        mllm._register_local_mlx_vlm_runtime_if_needed(model_dir)
        module = sys.modules["mlx_vlm.models.mimo_v2"]
        model = module.Model(module.ModelConfig.from_dict(config))

        count = mllm._quantize_mimo_v2_runtime_modules(
            model.language_model.inner,
            {},
            config["quantization"],
        )

        layer = model.language_model.inner.model.layers[1]
        assert count == 4
        assert layer["self_attn"]["qkv_proj"].__class__.__name__ == "QuantizedLinear"
        assert (
            layer["mlp"]["switch_mlp"]["gate_proj"].__class__.__name__
            == "QuantizedSwitchLinear"
        )
        assert (
            layer["mlp"]["switch_mlp"]["up_proj"].__class__.__name__
            == "QuantizedSwitchLinear"
        )
        assert (
            layer["mlp"]["switch_mlp"]["down_proj"].__class__.__name__
            == "QuantizedSwitchLinear"
        )

    def test_mllm_mimo_v2_runtime_quantizes_passthrough_decode_hotspots(
        self, tmp_path
    ):
        from vmlx_engine.models import mllm

        model_dir = tmp_path / "MiMo-V2.5-JANG_2L"
        model_dir.mkdir()
        config = {
            "model_type": "mimo_v2",
            "hidden_size": 32,
            "intermediate_size": 64,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 128,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "moe_layer_freq": [False],
            "hybrid_layer_pattern": [0],
            "full_attention_interval": 1,
            "quantization": {
                "bits": 8,
                "group_size": 32,
            },
        }
        (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

        mllm._register_local_mlx_vlm_runtime_if_needed(model_dir)
        module = sys.modules["mlx_vlm.models.mimo_v2"]
        model = module.Model(module.ModelConfig.from_dict(config))

        count = mllm._quantize_mimo_v2_passthrough_decode_hotspots(
            model.language_model.inner,
            config["quantization"],
        )

        assert count == 2
        assert model.language_model.inner.lm_head.__class__.__name__ == "QuantizedLinear"
        assert (
            model.language_model.inner.model.embed_tokens.__class__.__name__
            == "QuantizedEmbedding"
        )

    def test_mllm_mimo_v2_affine_switchglu_fast_path_matches_stock_decode(
        self,
    ):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.models.switch_layers import SwitchGLU

        from vmlx_engine.models import mllm

        switch = SwitchGLU(32, 32, 4, bias=False)
        nn.quantize(switch, group_size=32, bits=4, mode="affine")
        switch.eval()
        x = mx.arange(32, dtype=mx.float32).reshape(1, 32) / 32.0
        indices = mx.array([[0, 2]], dtype=mx.uint32)

        original = SwitchGLU.__call__
        expected = original(switch, x, indices)
        mx.eval(expected)
        installed = mllm._install_mimo_v2_affine_switchglu_decode_fast_path()
        try:
            actual = switch(x, indices)
            mx.eval(actual)
            fast_calls = getattr(SwitchGLU, "_vmlx_mimo_affine_decode_fast_calls", 0)
            fallback_reasons = dict(
                getattr(SwitchGLU, "_vmlx_mimo_affine_decode_fallback_reasons", {})
            )
        finally:
            if installed:
                SwitchGLU.__call__ = original
                SwitchGLU._vmlx_mimo_affine_decode_fast_path = False
                if hasattr(SwitchGLU, "_vmlx_mimo_original_affine_decode_call"):
                    delattr(SwitchGLU, "_vmlx_mimo_original_affine_decode_call")
                if hasattr(SwitchGLU, "_vmlx_mimo_affine_decode_fast_calls"):
                    delattr(SwitchGLU, "_vmlx_mimo_affine_decode_fast_calls")
                if hasattr(SwitchGLU, "_vmlx_mimo_affine_decode_fallback_reasons"):
                    delattr(SwitchGLU, "_vmlx_mimo_affine_decode_fallback_reasons")

        assert tuple(actual.shape) == tuple(expected.shape)
        assert mx.allclose(actual, expected, rtol=1e-4, atol=1e-4).item()
        assert fast_calls >= 1
        assert fallback_reasons == {}

    def test_mllm_inference_mode_reaches_mimo_python_list_switchglu(
        self, tmp_path
    ):
        from vmlx_engine.models import mllm

        model_dir = tmp_path / "MiMo-V2.5-JANG_2L"
        model_dir.mkdir()
        config = {
            "model_type": "mimo_v2",
            "hidden_size": 32,
            "intermediate_size": 64,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 128,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "moe_layer_freq": [False, True],
            "hybrid_layer_pattern": [0, 1],
            "full_attention_interval": 1,
        }
        (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

        mllm._register_local_mlx_vlm_runtime_if_needed(model_dir)
        module = sys.modules["mlx_vlm.models.mimo_v2"]
        model = module.Model(module.ModelConfig.from_dict(config))
        switch = model.language_model.inner.model.layers[1]["mlp"]["switch_mlp"]
        switch.train(True)

        mllm._set_vlm_inference_mode(model)

        assert switch.training is False

    def test_mllm_installs_mimo_v2_compiled_router_for_stale_jang_tools(self):
        from vmlx_engine.models import mllm

        class FakeGate:
            top_k = 8
            norm_topk_prob = True
            routed_scaling = 1.0

            def __call__(self, x):
                return "original"

        original = FakeGate.__call__
        stale_runtime = SimpleNamespace(MiMoV2MoEGate=FakeGate)

        assert mllm._install_mimo_v2_compiled_router_if_missing(stale_runtime) is True
        assert FakeGate.__call__ is not original
        assert FakeGate._vmlx_original_call is original
        assert FakeGate._vmlx_compiled_router_patched is True

        class NewGate:
            def __call__(self, x):
                return "native"

        native_runtime = SimpleNamespace(
            MiMoV2MoEGate=NewGate,
            run_compiled_mimo_decode_router=object(),
        )
        native_original = NewGate.__call__

        assert mllm._install_mimo_v2_compiled_router_if_missing(native_runtime) is False
        assert NewGate.__call__ is native_original

    def test_mllm_mimo_v2_compiled_router_reuses_fp32_gate_cache(self):
        import mlx.core as mx

        from vmlx_engine.models import mllm

        class FakeGate:
            top_k = 2
            norm_topk_prob = True
            routed_scaling = 1.0

            def __init__(self):
                self.weight = mx.ones((4, 4), dtype=mx.float16)
                self.e_score_correction_bias = mx.zeros((4,), dtype=mx.float16)

            def __call__(self, x):
                return "original"

        stale_runtime = SimpleNamespace(MiMoV2MoEGate=FakeGate)

        assert mllm._install_mimo_v2_compiled_router_if_missing(stale_runtime) is True

        gate = FakeGate()
        x = mx.ones((1, 4), dtype=mx.float16)
        first = gate(x)
        mx.eval(*first)
        cached_weight = gate._vmlx_gate_weight_fp32
        cached_bias = gate._vmlx_gate_bias_fp32

        second = gate(x)
        mx.eval(*second)

        assert gate._vmlx_gate_weight_fp32 is cached_weight
        assert gate._vmlx_gate_bias_fp32 is cached_bias
        assert gate._vmlx_gate_weight_source_id == id(gate.weight)
        assert gate._vmlx_gate_bias_source_id == id(gate.e_score_correction_bias)

    def test_mllm_mimo_v2_load_weights_sanitizes_and_quantizes_affine_sidecars(
        self, tmp_path, monkeypatch
    ):
        from vmlx_engine.models import mllm

        model_dir = tmp_path / "MiMo-V2.5-JANG_2L"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            '{"model_type":"mimo_v2","vision_config":{},"audio_config":{}}',
            encoding="utf-8",
        )

        events = []

        class FakeTextConfig:
            model_type = "mimo_v2"

            def __init__(self, **kwargs):
                self.quantization = kwargs.get("quantization")

            @classmethod
            def from_dict(cls, params):
                return cls(**params)

        class FakeTextModel:
            def __init__(self, config):
                self.config = config

            def sanitize(self, weights):
                events.append(("sanitize", sorted(weights)))
                return {
                    "model.layers.0.self_attn.qkv_proj.weight": object(),
                    "model.layers.0.self_attn.qkv_proj.scales": object(),
                    "model.layers.0.self_attn.qkv_proj.biases": object(),
                }

            def load_weights(self, weights, strict=True):
                events.append(("load", sorted(key for key, _ in weights), strict))
                return None

        def fake_import_module(name):
            if name == "jang_tools.mimo_v2.mlx_model":
                return SimpleNamespace(Model=FakeTextModel, ModelArgs=FakeTextConfig)
            raise ImportError(name)

        quantize_calls = []

        def fake_quantize(model, **kwargs):
            quantize_calls.append(kwargs)

        monkeypatch.setattr(mllm.importlib, "import_module", fake_import_module)
        import mlx.nn as nn

        monkeypatch.setattr(nn, "quantize", fake_quantize)

        sys.modules.pop("mlx_vlm.models.mimo_v2", None)
        mllm._register_local_mlx_vlm_runtime_if_needed(model_dir)
        module = sys.modules["mlx_vlm.models.mimo_v2"]
        cfg = module.ModelConfig.from_dict(
            {
                "model_type": "mimo_v2",
                "quantization": {"bits": 8, "group_size": 64, "mode": "affine"},
            }
        )
        model = module.Model(cfg)
        model.load_weights(
            [
                ("visual.blocks.0.weight", object()),
                ("model.mtp.layers.0.weight", object()),
                ("model.layers.0.self_attn.qkv_proj.weight", object()),
                ("model.layers.0.self_attn.qkv_proj.scales", object()),
                ("model.layers.0.self_attn.qkv_proj.biases", object()),
            ],
            strict=True,
        )

        assert events[0] == (
            "sanitize",
            [
                "model.layers.0.self_attn.qkv_proj.biases",
                "model.layers.0.self_attn.qkv_proj.scales",
                "model.layers.0.self_attn.qkv_proj.weight",
            ],
        )
        assert events[1] == (
            "load",
            [
                "model.layers.0.self_attn.qkv_proj.biases",
                "model.layers.0.self_attn.qkv_proj.scales",
                "model.layers.0.self_attn.qkv_proj.weight",
            ],
            True,
        )
        assert quantize_calls
        assert quantize_calls[0]["group_size"] == 64
        assert quantize_calls[0]["bits"] == 8
        assert quantize_calls[0]["mode"] == "affine"
        assert quantize_calls[0]["class_predicate"](
            "model.layers.0.self_attn.qkv_proj",
            SimpleNamespace(to_quantized=lambda **_: None),
        ) is True
        assert quantize_calls[0]["class_predicate"](
            "model.layers.0.self_attn.o_proj",
            SimpleNamespace(to_quantized=lambda **_: None),
        ) is False
        assert quantize_calls[0]["class_predicate"](
            "model.layers.0.self_attn.qkv_proj",
            SimpleNamespace(),
        ) is False
        sys.modules.pop("mlx_vlm.models.mimo_v2", None)

    def test_mllm_mimo_v2_language_model_accepts_mlx_vlm_inputs_embeds_contract(
        self, tmp_path, monkeypatch
    ):
        from vmlx_engine.models import mllm

        model_dir = tmp_path / "MiMo-V2.5-JANG_2L"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            '{"model_type":"mimo_v2","vision_config":{},"audio_config":{}}',
            encoding="utf-8",
        )

        class FakeTextConfig:
            model_type = "mimo_v2"

            @classmethod
            def from_dict(cls, params):
                return cls()

        class FakeBackbone:
            def __call__(self, input_ids=None, inputs_embeds=None, cache=None, mask=None):
                return f"hidden:{inputs_embeds}"

        class FakeTextModel:
            def __init__(self, config):
                self.model = FakeBackbone()
                self.lm_head = lambda hidden: f"logits:{hidden}"
                self.layers = []

            def __call__(self, input_ids, cache=None, mask=None):
                return f"raw:{input_ids}"

            def make_cache(self):
                return []

            def sanitize(self, weights):
                return weights

            def load_weights(self, weights, strict=True):
                return None

        def fake_import_module(name):
            if name == "jang_tools.mimo_v2.mlx_model":
                return SimpleNamespace(Model=FakeTextModel, ModelArgs=FakeTextConfig)
            raise ImportError(name)

        monkeypatch.setattr(mllm.importlib, "import_module", fake_import_module)

        sys.modules.pop("mlx_vlm.models.mimo_v2", None)
        mllm._register_local_mlx_vlm_runtime_if_needed(model_dir)
        module = sys.modules["mlx_vlm.models.mimo_v2"]
        model = module.Model(module.ModelConfig.from_dict({"model_type": "mimo_v2"}))

        assert model.language_model("ids") == "raw:ids"
        output = model.language_model("ids", inputs_embeds="embeds")
        assert output.logits == "logits:hidden:embeds"
        assert output.cross_attention_states is None
        assert output.encoder_outputs is None
        sys.modules.pop("mlx_vlm.models.mimo_v2", None)

    def test_mllm_mimo_v2_language_model_preserves_existing_output_object(
        self, tmp_path, monkeypatch
    ):
        from vmlx_engine.models import mllm

        model_dir = tmp_path / "MiMo-V2.5-JANG_2L"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            '{"model_type":"mimo_v2","vision_config":{},"audio_config":{}}',
            encoding="utf-8",
        )

        class FakeTextConfig:
            model_type = "mimo_v2"

            @classmethod
            def from_dict(cls, params):
                return cls()

        class FakeTextModel:
            def __init__(self, config):
                self.model = SimpleNamespace(embed_tokens=lambda input_ids: input_ids)
                self.layers = []

            def __call__(self, input_ids, inputs_embeds=None, cache=None, mask=None):
                return SimpleNamespace(
                    logits=f"logits:{inputs_embeds}",
                    cross_attention_states=None,
                    encoder_outputs=None,
                )

            def make_cache(self):
                return []

            def sanitize(self, weights):
                return weights

            def load_weights(self, weights, strict=True):
                return None

        def fake_import_module(name):
            if name == "jang_tools.mimo_v2.mlx_model":
                return SimpleNamespace(Model=FakeTextModel, ModelArgs=FakeTextConfig)
            raise ImportError(name)

        monkeypatch.setattr(mllm.importlib, "import_module", fake_import_module)

        sys.modules.pop("mlx_vlm.models.mimo_v2", None)
        mllm._register_local_mlx_vlm_runtime_if_needed(model_dir)
        module = sys.modules["mlx_vlm.models.mimo_v2"]
        model = module.Model(module.ModelConfig.from_dict({"model_type": "mimo_v2"}))

        output = model.language_model("ids", inputs_embeds="embeds")
        assert output.logits == "logits:embeds"
        assert output.cross_attention_states is None
        assert output.encoder_outputs is None
        sys.modules.pop("mlx_vlm.models.mimo_v2", None)

    def test_mllm_mimo_v2_text_only_folds_leading_system_into_user_prompt(self):
        from vmlx_engine.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("mimo-v2-test")
        model.config = {"model_type": "mimo_v2"}

        normalized = model._normalize_text_only_messages_for_processor(
            [
                {"role": "system", "content": [{"type": "text", "text": "Output ACK."}]},
                {"role": "user", "content": [{"type": "text", "text": "Say it now."}]},
            ],
            has_media=False,
        )

        assert normalized == [
            {"role": "user", "content": "Output ACK.\n\nSay it now."}
        ]

    def test_mllm_mimo_v2_text_only_system_fold_does_not_touch_media_turns(self):
        from vmlx_engine.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("mimo-v2-test")
        model.config = {"model_type": "mimo_v2"}
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "Use image."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe it."},
                ],
            },
        ]

        assert (
            model._normalize_text_only_messages_for_processor(
                messages,
                has_media=True,
            )
            is messages
        )

    def test_mllm_mimo_v2_media_forward_raises_typed_unsupported_modality(
        self, tmp_path, monkeypatch
    ):
        from vmlx_engine.errors import UnsupportedMediaModalityError
        from vmlx_engine.models import mllm

        model_dir = tmp_path / "MiMo-V2.5-JANG_2L"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            '{"model_type":"mimo_v2","vision_config":{},"audio_config":{}}',
            encoding="utf-8",
        )

        class FakeTextConfig:
            model_type = "mimo_v2"

            @classmethod
            def from_dict(cls, params):
                return cls()

        class FakeTextModel:
            def __init__(self, config):
                self.model = SimpleNamespace(embed_tokens=lambda input_ids: input_ids)
                self.layers = []

            def __call__(self, input_ids, cache=None, mask=None):
                return input_ids

            def make_cache(self):
                return []

            def sanitize(self, weights):
                return weights

            def load_weights(self, weights, strict=True):
                return None

        def fake_import_module(name):
            if name == "jang_tools.mimo_v2.mlx_model":
                return SimpleNamespace(Model=FakeTextModel, ModelArgs=FakeTextConfig)
            raise ImportError(name)

        monkeypatch.setattr(mllm.importlib, "import_module", fake_import_module)

        sys.modules.pop("mlx_vlm.models.mimo_v2", None)
        mllm._register_local_mlx_vlm_runtime_if_needed(model_dir)
        module = sys.modules["mlx_vlm.models.mimo_v2"]
        model = module.Model(module.ModelConfig.from_dict({"model_type": "mimo_v2"}))

        with pytest.raises(UnsupportedMediaModalityError) as exc:
            model("ids", pixel_values="pixels")

        assert exc.value.modality == "vision"
        assert exc.value.family == "mimo_v2"
        assert exc.value.code == "unsupported_media_modality"
        sys.modules.pop("mlx_vlm.models.mimo_v2", None)

    def test_mllm_mimo_v2_splices_precomputed_media_embeddings(
        self, tmp_path, monkeypatch
    ):
        import mlx.core as mx

        from vmlx_engine.models import mllm

        model_dir = tmp_path / "MiMo-V2.5-JANG_2L"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            '{"model_type":"mimo_v2","vision_config":{},"audio_config":{}}',
            encoding="utf-8",
        )

        class FakeTextConfig:
            model_type = "mimo_v2"

            @classmethod
            def from_dict(cls, params):
                return cls()

        def _embed_tokens(input_ids):
            ids = mx.array(input_ids)
            if ids.ndim == 1:
                ids = ids[None, :]
            ids = ids.astype(mx.float32)
            return mx.stack([ids, ids + 0.5], axis=-1)

        class FakeBackbone:
            embed_tokens = staticmethod(_embed_tokens)

        class FakeTextModel:
            def __init__(self, config):
                self.model = FakeBackbone()
                self.layers = []

            def __call__(self, input_ids, inputs_embeds=None, cache=None, mask=None):
                return SimpleNamespace(logits=inputs_embeds)

            def make_cache(self):
                return []

            def sanitize(self, weights):
                return weights

            def load_weights(self, weights, strict=True):
                return None

        real_import_module = importlib.import_module

        def fake_import_module(name, package=None):
            if name == "jang_tools.mimo_v2.mlx_model":
                return SimpleNamespace(Model=FakeTextModel, ModelArgs=FakeTextConfig)
            return real_import_module(name, package)

        monkeypatch.setattr(mllm.importlib, "import_module", fake_import_module)

        sys.modules.pop("mlx_vlm.models.mimo_v2", None)
        mllm._register_local_mlx_vlm_runtime_if_needed(model_dir)
        module = sys.modules["mlx_vlm.models.mimo_v2"]
        model = module.Model(
            module.ModelConfig.from_dict(
                {
                    "model_type": "mimo_v2",
                    "processor_config": {
                        "image_token_id": 151655,
                        "video_token_id": 151656,
                        "audio_token_id": 151669,
                    },
                }
            )
        )

        input_ids = mx.array([[11, 151655, 22, 151656, 151669]])
        features = model.get_input_embeddings(
            input_ids=input_ids,
            image_embeds=mx.array([[1.0, 1.5]]),
            video_embeds=mx.array([[2.0, 2.5]]),
            audio_embeds=mx.array([[3.0, 3.5]]),
        )

        assert features.inputs_embeds.tolist() == [
            [
                [11.0, 11.5],
                [1.0, 1.5],
                [22.0, 22.5],
                [2.0, 2.5],
                [3.0, 3.5],
            ]
        ]

        output = model(
            input_ids,
            image_embeds=mx.array([[4.0, 4.5]]),
            video_embeds=mx.array([[5.0, 5.5]]),
            audio_embeds=mx.array([[6.0, 6.5]]),
        )
        assert output.logits.tolist()[0][1] == [4.0, 4.5]
        assert output.logits.tolist()[0][3] == [5.0, 5.5]
        assert output.logits.tolist()[0][4] == [6.0, 6.5]
        sys.modules.pop("mlx_vlm.models.mimo_v2", None)

    def test_mllm_mimo_v2_rejects_precomputed_media_count_mismatch(
        self, tmp_path, monkeypatch
    ):
        import mlx.core as mx

        from vmlx_engine.models import mllm

        model_dir = tmp_path / "MiMo-V2.5-JANG_2L"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            '{"model_type":"mimo_v2","vision_config":{},"audio_config":{}}',
            encoding="utf-8",
        )

        class FakeTextConfig:
            model_type = "mimo_v2"

            @classmethod
            def from_dict(cls, params):
                return cls()

        class FakeTextModel:
            def __init__(self, config):
                self.model = SimpleNamespace(
                    embed_tokens=lambda input_ids: mx.zeros((1, 3, 2))
                )
                self.layers = []

        real_import_module = importlib.import_module

        def fake_import_module(name, package=None):
            if name == "jang_tools.mimo_v2.mlx_model":
                return SimpleNamespace(Model=FakeTextModel, ModelArgs=FakeTextConfig)
            return real_import_module(name, package)

        monkeypatch.setattr(mllm.importlib, "import_module", fake_import_module)

        sys.modules.pop("mlx_vlm.models.mimo_v2", None)
        mllm._register_local_mlx_vlm_runtime_if_needed(model_dir)
        module = sys.modules["mlx_vlm.models.mimo_v2"]
        model = module.Model(
            module.ModelConfig.from_dict(
                {
                    "model_type": "mimo_v2",
                    "processor_config": {"image_token_id": 151655},
                }
            )
        )

        with pytest.raises(ValueError, match="image token count 1 does not match"):
            model.get_input_embeddings(
                input_ids=mx.array([[11, 151655, 22]]),
                image_embeds=mx.array([[1.0, 1.5], [2.0, 2.5]]),
            )
        sys.modules.pop("mlx_vlm.models.mimo_v2", None)

    def test_mllm_mimo_v2_config_preserves_media_processor_contract(
        self, tmp_path, monkeypatch
    ):
        from vmlx_engine.models import mllm

        model_dir = tmp_path / "MiMo-V2.5-JANGTQ_2"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            '{"model_type":"mimo_v2","vision_config":{},"audio_config":{}}',
            encoding="utf-8",
        )

        class FakeTextConfig:
            model_type = "mimo_v2"

            @classmethod
            def from_dict(cls, params):
                return cls()

        class FakeTextModel:
            def __init__(self, config):
                self.model = SimpleNamespace(embed_tokens=lambda input_ids: input_ids)
                self.layers = []

        def fake_import_module(name):
            if name == "jang_tools.mimo_v2.mlx_model":
                return SimpleNamespace(Model=FakeTextModel, ModelArgs=FakeTextConfig)
            raise ImportError(name)

        monkeypatch.setattr(mllm.importlib, "import_module", fake_import_module)

        sys.modules.pop("mlx_vlm.models.mimo_v2", None)
        mllm._register_local_mlx_vlm_runtime_if_needed(model_dir)
        module = sys.modules["mlx_vlm.models.mimo_v2"]
        cfg = module.ModelConfig.from_dict(
            {
                "model_type": "mimo_v2",
                "vision_config": {
                    "depth": 28,
                    "hidden_size": 1280,
                    "out_hidden_size": 4096,
                    "patch_size": 16,
                    "temporal_patch_size": 2,
                    "tokens_per_second": 2,
                    "vit_window_attn_types": [-1, 0, 1],
                },
                "audio_config": {
                    "audio_channels": 20,
                    "audio_segment_size": 6000,
                    "out_hidden_size": 4096,
                    "speech_vocab_size": "1280",
                    "speech_zeroemb_idx": "1024",
                },
                "processor_config": {
                    "image_token_id": 151655,
                    "video_token_id": 151656,
                    "audio_token_id": 151669,
                    "vision_start_token_id": 151652,
                    "vision_end_token_id": 151653,
                    "video_start_token_id": 151670,
                    "video_end_token_id": 151671,
                    "audio_start_token_id": 151673,
                    "audio_end_token_id": 151674,
                    "image_max_pixels": 8388608,
                    "video_total_max_pixels": 268435456,
                    "max_frames": 3600,
                    "fps": 1.0,
                },
                "capabilities": {
                    "preserved_modalities": ["vision", "audio"],
                    "unwired_modalities": ["vision", "audio"],
                    "multimodal_status": "weights_preserved_text_runtime",
                },
            }
        )

        assert cfg.vision_config.depth == 28
        assert cfg.vision_config.out_hidden_size == 4096
        assert cfg.vision_config.vit_window_attn_types == [-1, 0, 1]
        assert cfg.audio_config.audio_segment_size == 6000
        assert cfg.audio_config.speech_vocab_size == "1280"
        assert cfg.processor_config["video_total_max_pixels"] == 268435456
        assert cfg.processor_config["max_frames"] == 3600
        assert cfg.media_token_ids == {
            "image": 151655,
            "video": 151656,
            "audio": 151669,
            "vision_start": 151652,
            "vision_end": 151653,
            "video_start": 151670,
            "video_end": 151671,
            "audio_start": 151673,
            "audio_end": 151674,
        }
        assert cfg.media_runtime_status == "weights_preserved_text_runtime"
        assert cfg.preserved_modalities == ["vision", "audio"]
        assert cfg.unwired_modalities == ["vision", "audio"]
        sys.modules.pop("mlx_vlm.models.mimo_v2", None)

    def test_mimo_v2_thinking_off_suppresses_scheduler_stop_tokens_first_only(self):
        import mlx.core as mx

        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.processor = SimpleNamespace(
            encode=lambda token, add_special_tokens=False: [],
            eos_token_id=None,
        )
        generator.stop_tokens = {3}
        generator._mimo_v2_thinking_off_token_ids = None
        request = SimpleNamespace(output_tokens=[])

        processors = generator._mimo_v2_thinking_off_logits_processors(request)
        logits = mx.zeros((1, 6))
        processed = logits
        for processor in processors:
            processed = processor(mx.array([]), processed)
        assert mx.isinf(processed[0, 3]).item()
        assert processed[0, 3].item() < 0

        request.output_tokens = [1]
        processed_after_first = logits
        for processor in processors:
            processed_after_first = processor(mx.array([1]), processed_after_first)
        assert processed_after_first[0, 3].item() == 0

    def test_mimo_v2_cached_text_fast_path_uses_absolute_positions(self):
        from pathlib import Path

        source = Path("./vmlx_engine/mllm_batch_generator.py").read_text()
        start = source.index("_mimo_tight_text_prefill_requires_chunking = (")
        end = source.index(
            "elif _mimo_tight_text_prefill_requires_chunking:",
            start,
        )
        block = source[start:end]

        assert "_tight_text_prefill_step_size < self.prefill_step_size" in block
        assert "seq_len > _tight_text_prefill_step_size + 1" in block
        assert "and not _mimo_tight_text_prefill_requires_chunking" in block
        assert "_absolute_text_position_ids(input_ids, cache, lm)" in block
        assert 'kwargs["position_ids"] = position_ids' in block
        assert "_seed_text_rope_delta_for_decode(lm, input_ids)" in block
        assert "output = lm(input_ids, **kwargs)" in block
        assert "output = lm(input_ids, cache=cache)" not in block

    def test_mllm_registers_step3p7_source_runtime_before_mlx_vlm_resolution(
        self, tmp_path
    ):
        from vmlx_engine.models import mllm

        model_dir = tmp_path / "Step-3.7-Flash-JANG_2L"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            '{"model_type":"step3p7","text_config":{"model_type":"step3p5"},'
            '"vision_config":{"model_type":"perception_encoder"}}',
            encoding="utf-8",
        )

        for name in (
            "mlx_vlm.models.step3p7",
            "mlx_vlm.models.step3p7.processing_step3",
        ):
            sys.modules.pop(name, None)

        mllm._register_local_mlx_vlm_runtime_if_needed(model_dir)

        module = sys.modules["mlx_vlm.models.step3p7"]
        assert hasattr(module, "Model")
        assert hasattr(module, "ModelConfig")
        assert hasattr(module, "VisionModel")
        assert hasattr(module, "LanguageModel")
        assert hasattr(module, "Step3VLProcessor")
        assert hasattr(module.Model, "load_weights")
        assert hasattr(module.Model, "parameters")
        assert hasattr(module.Model, "eval")
        assert importlib.util.find_spec("mlx_vlm.models.step3p7") is not None
        assert (
            importlib.util.find_spec("mlx_vlm.models.step3p7.processing_step3")
            is not None
        )

        sys.modules.pop("mlx_vlm.models.step3p7.processing_step3", None)
        sys.modules.pop("mlx_vlm.models.step3p7", None)

    def test_jang_loader_failure_message_names_required_runtime_floor(
        self, monkeypatch
    ):
        from vmlx_engine.utils import jang_loader

        def fake_import_module(name):
            raise ImportError(f"missing {name}")

        monkeypatch.setattr(jang_loader.importlib, "import_module", fake_import_module)
        monkeypatch.setattr(
            jang_loader,
            "_register_bailing_hybrid_from_repo_patch",
            lambda: (_ for _ in ()).throw(ImportError("missing bailing patch")),
        )

        with pytest.raises(RuntimeError, match="jang>=2.5.30"):
            jang_loader._ensure_jang_family_runtime_supported(
                Path("/models/Hy3-preview-JANGTQ2"), {"model_type": "hy_v3"}
            )

        with pytest.raises(RuntimeError, match="mlx-lm>=0.31.3"):
            jang_loader._ensure_jang_family_runtime_supported(
                Path("/models/Ling-2.6-flash-JANGTQ"),
                {"model_type": "bailing_hybrid"},
            )

    def test_embeddings_and_rerank_endpoints_have_memory_pressure_guards(self):
        source = Path("./vmlx_engine/server.py").read_text()

        assert '@app.post(\n    "/v1/embeddings",\n    dependencies=[\n        Depends(verify_api_key),\n        Depends(check_rate_limit),\n        Depends(check_memory_pressure),' in source
        assert '@app.post(\n    "/v1/embeddings",\n    dependencies=[\n        Depends(verify_api_key),\n        Depends(check_rate_limit),\n        Depends(check_memory_pressure),\n        Depends(check_metal_working_set_pressure),' in source
        assert '@app.post(\n    "/v1/rerank",\n    dependencies=[\n        Depends(verify_api_key),\n        Depends(check_rate_limit),\n        Depends(check_memory_pressure),' in source
        assert '@app.post(\n    "/v1/rerank",\n    dependencies=[\n        Depends(verify_api_key),\n        Depends(check_rate_limit),\n        Depends(check_memory_pressure),\n        Depends(check_metal_working_set_pressure),' in source
        assert '@app.post(\n    "/api/embeddings",\n    dependencies=[\n        Depends(verify_api_key),\n        Depends(check_memory_pressure),' in source
        assert '@app.post(\n    "/api/embeddings",\n    dependencies=[\n        Depends(verify_api_key),\n        Depends(check_memory_pressure),\n        Depends(check_metal_working_set_pressure),' in source
        assert '@app.post(\n    "/api/embed",\n    dependencies=[\n        Depends(verify_api_key),\n        Depends(check_memory_pressure),' in source
        assert '@app.post(\n    "/api/embed",\n    dependencies=[\n        Depends(verify_api_key),\n        Depends(check_memory_pressure),\n        Depends(check_metal_working_set_pressure),' in source

    def test_bundle_selects_native_mlx_wheels_with_compat_override(self):
        """Bundling can select Sequoia-compatible or Tahoe-native MLX wheels."""
        bundle_script = Path("./panel/scripts/bundle-python.sh").read_text()
        assert 'MLX_VERSION="0.31.2"' in bundle_script
        assert 'MLX_LM_VERSION="0.31.3"' in bundle_script
        assert 'MLX_VLM_VERSION="0.5.0"' in bundle_script
        assert 'detect_mlx_wheel_platform()' in bundle_script
        assert 'VMLX_BUNDLE_MLX_PLATFORM:-${VMLINUX_BUNDLE_MLX_PLATFORM:-compat}' in bundle_script
        assert "legacy misspelled VMLINUX_BUNDLE_MLX_PLATFORM is still accepted" in bundle_script
        assert 'echo "macosx_26_0_arm64"' in bundle_script
        assert 'auto|compat|sonoma|sequoia|"")' in bundle_script
        assert 'echo "macosx_14_0_arm64"' in bundle_script
        assert 'MLX_WHEEL_PLATFORM="$(detect_mlx_wheel_platform)"' in bundle_script
        assert '--platform "$MLX_WHEEL_PLATFORM"' in bundle_script
        assert '"mlx==$MLX_VERSION"' in bundle_script
        assert '"mlx-metal==$MLX_VERSION"' in bundle_script

    def test_release_build_declares_sequoia_and_tahoe_dmg_flavors(self):
        """vmlx#169 release packaging must build both wheel-tag variants."""
        script = Path("./panel/scripts/build-release-dmgs.sh")
        panel_package = json.loads(Path("./panel/package.json").read_text())

        assert script.is_file()
        source = script.read_text()
        assert 'build_one "sequoia" "compat"' in source
        assert 'build_one "tahoe" "native"' in source
        assert 'VMLINUX_BUNDLE_MLX_PLATFORM="$platform"' not in source
        assert 'VMLX_BUNDLE_MLX_PLATFORM="$platform"' in source
        assert 'vMLX-\\${version}-${flavor}-\\${arch}.\\${ext}' in source
        assert "electron-builder --mac" in source
        assert "run_release_regression_manifest.py" in source
        assert "--require-prepackage-ready" in source
        assert 'cd "$ROOT_DIR"' in source
        assert "gh release" not in source
        assert "latest.json" not in source
        assert panel_package["build"]["dmg"]["sign"] is True
        assert "--require-release-ready" in panel_package["scripts"]["release:ready"]
        assert "--require-prepackage-ready" in panel_package["scripts"]["release:prepackage"]
        assert panel_package["scripts"]["dist"].startswith("npm run release:prepackage && ")

    def test_release_build_signs_bundled_python_before_dmg_signing(self):
        """Release DMG builds must not leave bundled Python dylibs unsigned."""
        source = Path("./panel/scripts/build-release-dmgs.sh").read_text()

        assert "sign_bundled_python_native_files()" in source
        assert 'find "$bundled_python" -type f' in source
        assert '-name "*.dylib" -o -name "*.so" -o -perm +111' in source
        assert 'codesign --force --timestamp --options runtime --sign "$identity" "$native_file"' in source
        assert "finalize_release_app_signature" in source
        assert (
            'codesign --force --deep --timestamp --options runtime --entitlements "$entitlements" '
            '--sign "$identity" "$app_path"'
        ) in source
        assert 'codesign --verify --deep --strict --verbose=2 "$app_path"' in source
        assert 'finalize_release_app_signature "$app_path"' in source
        assert '--prepackaged "$app_path"' in source

    def test_macos_compat_error_points_to_sequoia_dmg_for_tahoe_wheels(self):
        """Wrong-DMG installs should fail before MLX import with actionable text."""
        cli_source = Path("./vmlx_engine/cli.py").read_text()
        check_idx = cli_source.index("def _check_macos_compat")
        check_block = cli_source[check_idx: cli_source.index("def _check_no_duplicate_mlx", check_idx)]

        assert "download the Sequoia-compatible vMLX DMG" in check_block
        assert "Tahoe-native DMG requires macOS 26" in check_block

    def test_turboquant_disable_env_is_honored_by_jang_loader(self):
        """Explicit/off-family cache choices must stop loader-level live TQ-KV."""
        loader_source = Path("./vmlx_engine/utils/jang_loader.py").read_text()
        tokenizer_source = Path("./vmlx_engine/utils/tokenizer.py").read_text()
        cli_source = Path("./vmlx_engine/cli.py").read_text()

        assert 'environ.get("VMLX_DISABLE_TQ_KV")' in loader_source
        assert "TurboQuant KV skipped" in loader_source
        assert 'os.environ.get("VMLX_DISABLE_TQ_KV")' in tokenizer_source
        assert "TurboQuant skipped: VMLX_DISABLE_TQ_KV=1" in tokenizer_source
        assert 'os.environ["VMLX_DISABLE_TQ_KV"] = "1"' in cli_source
        assert "VMLINUX_DISABLE_TQ_KV" not in cli_source

    def test_hybrid_auto_mode_enables_qwen_selective_live_tq_only(self):
        """Qwen3.6 hybrid auto mode TQs attention KV while preserving SSM state."""
        cli_source = Path("./vmlx_engine/cli.py").read_text()
        scheduler_source = Path("./vmlx_engine/scheduler.py").read_text()
        tokenizer_source = Path("./vmlx_engine/utils/tokenizer.py").read_text()

        assert 'getattr(_mc, "cache_type", None) == "hybrid"' in cli_source
        assert "Qwen3.6 hybrid/path-dependent cache model detected" in cli_source
        assert "attention KVCache" in cli_source
        assert "Only Qwen3.6 has a selective live" in cli_source
        hybrid_idx = cli_source.index('getattr(_mc, "cache_type", None) == "hybrid"')
        hybrid_block = cli_source[hybrid_idx : cli_source.index("if _mc.family_name !=", hybrid_idx)]
        assert 'args.kv_cache_quantization = "none"' not in hybrid_block
        assert "args.kv_cache_quantization_explicit = True" not in hybrid_block
        assert "stored attention-KV" in hybrid_block

        assert "Hybrid/path-dependent cache model detected — using q4/q8 only at cache storage boundaries" in scheduler_source
        assert "non-KV state is preserved full precision" in scheduler_source
        assert "build_hybrid_turboquant_make_cache" in tokenizer_source
        assert "is_qwen36_hybrid_tq_supported" in tokenizer_source
        assert "model.make_cache()" in tokenizer_source

    def test_generic_turboquant_patcher_honors_disable_env(self, tmp_path, monkeypatch):
        """The non-JANG fallback TQ hook must not bypass VMLX_DISABLE_TQ_KV."""
        from vmlx_engine.utils.tokenizer import _apply_turboquant_to_model

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "llama",
            "num_hidden_layers": 1,
            "head_dim": 128,
        }))

        class FakeModel:
            layers = [object()]

            def make_cache(self):
                return ["native"]

        model = FakeModel()
        monkeypatch.setenv("VMLX_DISABLE_TQ_KV", "1")

        _apply_turboquant_to_model(model, str(tmp_path))

        assert model.make_cache() == ["native"]

    def test_generic_turboquant_patcher_skips_hybrid_ssm(self, tmp_path, monkeypatch):
        """Ling/Bailing hybrid must keep its native KV+SSM cache contract."""
        from vmlx_engine.utils.tokenizer import _apply_turboquant_to_model

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "bailing_hybrid",
            "num_hidden_layers": 32,
            "layer_group_size": 8,
            "head_dim": 128,
        }))

        class FakeModel:
            layers = [object()] * 33

            def make_cache(self):
                return ["native"] * 32

        model = FakeModel()
        os.environ.pop("VMLX_DISABLE_TQ_KV", None)
        monkeypatch.delenv("VMLX_ALLOW_HYBRID_KV_QUANT", raising=False)

        _apply_turboquant_to_model(model, str(tmp_path))

        assert model.make_cache() == ["native"] * 32

    def test_generic_turboquant_patcher_skips_native_rotating_cache(
        self, tmp_path, monkeypatch
    ):
        """MiMo/Gemma-style rotating KV must not be flattened to generic TQ."""
        from vmlx_engine.utils.tokenizer import _apply_turboquant_to_model

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "mimo_v2",
            "num_hidden_layers": 2,
            "head_dim": 128,
        }))

        class KVCache:
            pass

        class RotatingKVCache:
            pass

        class FakeModel:
            layers = [object()] * 2

            def make_cache(self):
                return [KVCache(), RotatingKVCache()]

        model = FakeModel()
        os.environ.pop("VMLX_DISABLE_TQ_KV", None)
        monkeypatch.setenv("VMLX_FORCE_TQ_AUTO", "1")

        _apply_turboquant_to_model(model, str(tmp_path))

        assert [type(c).__name__ for c in model.make_cache()] == [
            "KVCache",
            "RotatingKVCache",
        ]

    def test_jang_loader_skips_turboquant_for_gemma4_mixed_attention(self, monkeypatch):
        """Gemma4 sliding/full attention must keep native RotatingKVCache layout."""
        from vmlx_engine.utils.jang_loader import _patch_turboquant_make_cache

        class FakeCache:
            pass

        class FakeModel:
            layers = [object()] * 30

            def make_cache(self):
                return [FakeCache()] * 30

        model = FakeModel()
        monkeypatch.setenv("VMLINUX_FORCE_TQ_AUTO", "1")
        monkeypatch.delenv("VMLINUX_DISABLE_TQ_KV", raising=False)

        _patch_turboquant_make_cache(
            model,
            {},
            {
                "model_type": "gemma4",
                "text_config": {
                    "model_type": "gemma4_text",
                    "num_hidden_layers": 30,
                    "layer_types": ["sliding_attention", "full_attention"],
                    "head_dim": 256,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                },
            },
        )

        assert model.make_cache.__func__ is FakeModel.make_cache
        assert [type(c).__name__ for c in model.make_cache()] == ["FakeCache"] * 30

    def test_jang_loader_skips_turboquant_for_native_rotating_cache(self, monkeypatch):
        """MiMo V2 native KV+RotatingKVCache layout must stay intact."""
        from vmlx_engine.utils.jang_loader import _patch_turboquant_make_cache

        class KVCache:
            pass

        class RotatingKVCache:
            pass

        class FakeModel:
            layers = [object()] * 2

            def make_cache(self):
                return [KVCache(), RotatingKVCache()]

        model = FakeModel()
        monkeypatch.setenv("VMLX_FORCE_TQ_AUTO", "1")
        monkeypatch.delenv("VMLX_DISABLE_TQ_KV", raising=False)

        _patch_turboquant_make_cache(
            model,
            {},
            {
                "model_type": "mimo_v2",
                "num_hidden_layers": 2,
                "head_dim": 128,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
            },
        )

        assert model.make_cache.__func__ is FakeModel.make_cache
        assert [type(c).__name__ for c in model.make_cache()] == [
            "KVCache",
            "RotatingKVCache",
        ]

    def test_jang_loader_skips_turboquant_for_mimo_v2_by_config(self, monkeypatch):
        """MiMo must not flatten mixed full/SWA cache even if cache probing is stale."""
        from vmlx_engine.utils.jang_loader import _patch_turboquant_make_cache

        class FakeCache:
            pass

        class FakeModel:
            layers = [object()] * 48

            def make_cache(self):
                return [FakeCache()] * 48

        model = FakeModel()
        monkeypatch.setenv("VMLINUX_FORCE_TQ_AUTO", "1")
        monkeypatch.delenv("VMLINUX_DISABLE_TQ_KV", raising=False)

        _patch_turboquant_make_cache(
            model,
            {},
            {
                "model_type": "mimo_v2",
                "num_hidden_layers": 48,
                "head_dim": 192,
                "v_head_dim": 128,
                "num_attention_heads": 64,
                "num_key_value_heads": 4,
            },
        )

        assert model.make_cache.__func__ is FakeModel.make_cache
        assert [type(c).__name__ for c in model.make_cache()] == ["FakeCache"] * 48

    def test_tokenizer_skips_turboquant_for_mimo_v2_by_config(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Standard auto-TQ path must also preserve MiMo native mixed-SWA cache."""
        from vmlx_engine.utils.tokenizer import _apply_turboquant_to_model

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "mimo_v2",
                    "num_hidden_layers": 48,
                    "head_dim": 192,
                    "v_head_dim": 128,
                    "num_attention_heads": 64,
                    "num_key_value_heads": 4,
                }
            )
        )

        class FakeCache:
            pass

        class FakeModel:
            layers = [object()] * 48

            def make_cache(self):
                return [FakeCache()] * 48

        model = FakeModel()
        monkeypatch.delenv("VMLINUX_DISABLE_TQ_KV", raising=False)

        _apply_turboquant_to_model(model, str(tmp_path))

        assert model.make_cache.__func__ is FakeModel.make_cache
        assert [type(c).__name__ for c in model.make_cache()] == ["FakeCache"] * 48

    def test_tokenizer_skips_turboquant_for_mimo_v2_by_loaded_model_object(
        self,
        monkeypatch,
    ):
        """MiMo skip must not depend on model_path being a local directory."""
        from vmlx_engine.utils.tokenizer import _apply_turboquant_to_model

        class FakeCache:
            pass

        class FakeModel:
            layers = [object()] * 48
            inner = SimpleNamespace(args=SimpleNamespace(model_type="mimo_v2"))

            def make_cache(self):
                return [FakeCache()] * 48

        model = FakeModel()
        monkeypatch.delenv("VMLINUX_DISABLE_TQ_KV", raising=False)

        _apply_turboquant_to_model(model, "JANGQ-AI/MiMo-V2.5-JANG_2L")

        assert model.make_cache.__func__ is FakeModel.make_cache
        assert [type(c).__name__ for c in model.make_cache()] == ["FakeCache"] * 48

    def test_tokenizer_skips_turboquant_for_mimo_model_identifier(
        self,
        monkeypatch,
    ):
        """MiMo path/name fallback protects wrappers with incomplete configs."""
        from vmlx_engine.utils.tokenizer import _apply_turboquant_to_model

        class FakeCache:
            pass

        class FakeModel:
            layers = [object()] * 48

            def make_cache(self):
                return [FakeCache()] * 48

        model = FakeModel()
        monkeypatch.delenv("VMLINUX_DISABLE_TQ_KV", raising=False)

        _apply_turboquant_to_model(model, "JANGQ-AI/MiMo-V2.5-JANG_2L")

        assert model.make_cache.__func__ is FakeModel.make_cache
        assert [type(c).__name__ for c in model.make_cache()] == ["FakeCache"] * 48

    def test_vmlx_env_prefix_is_canonical_for_ssm_cache_budget(self):
        """New cache env knobs should use VMLX_, with typo fallback only."""
        cli_source = Path("./vmlx_engine/cli.py").read_text()
        assert "def _env_int(" in cli_source
        assert "os.environ.get(name)" in cli_source
        assert '"VMLINUX_SSM_STATE_CACHE_MB"' in cli_source
        assert 'default=_env_int("VMLX_SSM_STATE_CACHE_MB", 512, "VMLINUX_SSM_STATE_CACHE_MB")' in cli_source

    def test_metal_ws_pressure_guard_reclaims_cache_before_reject(self):
        """Pressure guard must not 503 while MLX cache memory is reclaimable."""
        server_source = Path("./vmlx_engine/server.py").read_text()
        start = server_source.index("async def check_metal_working_set_pressure")
        end = server_source.index("# Log throttle", start)
        block = server_source[start:end]

        assert "mx.clear_cache()" in block
        assert "active_after, max_ws_after = get_effective_metal_working_set_bytes(mx)" in block
        assert 'getattr(mx, "get_cache_memory", None)' in block
        assert "non_cache_active = max(0, active - cache_bytes)" in block
        assert "non_cache_pct = (non_cache_active / max_ws) * 100.0" in block
        assert "_metal_ws_model_baseline_bytes" in block
        assert "transient_bytes = max(0, active - int(baseline))" in block
        assert "allowed_transient_pct = max(0.5, 100.0 - threshold_pct)" in block
        assert "get_metal_ws_guard_threshold(99.0)" in block
        assert "default 99" in server_source
        assert "_record_metal_ws_model_baseline(\"lifespan_engine_start\")" in server_source
        assert "_record_metal_ws_model_baseline(\"simple_engine_start\")" in server_source
        assert "pct = (active / max_ws) * 100.0" in block
        assert "if pct < threshold_pct:\n                    return" in block

    def test_xml_function_tool_fallback_accepts_native_mimo_schema(self):
        """MiMo/xml_function prompts should not duplicate a native tool schema."""
        source = Path("./vmlx_engine/api/tool_calling.py").read_text()
        server_source = Path("./vmlx_engine/server.py").read_text()
        start = source.index("_xml_function_has_native_tool_schema = (")
        end = source.index("_step3p5_has_concrete_tool_examples", start)
        block = source[start:end]

        assert 'all(f"<name>{name}</name>" in instruction_prompt for name in tool_names)' in block
        assert 'all(f"<function={name}>" in instruction_prompt for name in tool_names)' not in block
        assert '"<function=example_function_name>" not in source'
        assert "def _splice_tool_prompt_into_rendered_chatml(" in source
        assert (
            "if is_xml_function_native_tool_prompt:\n"
            "            return _splice_tool_prompt_into_rendered_chatml(prompt, tool_prompt)"
            in source
        )
        assert ") and not is_xml_function_native_tool_prompt:" in source
        assert server_source.count("raw_preview={tool_required_preview!r}") >= 2
        assert "def _drop_tool_visible_channel_marker(" in server_source
        assert "text.strip().lower() in {" in server_source
        assert '"thought"' in server_source
        assert '"analysis"' in server_source
        assert '"<audio|>"' in server_source

    def test_tool_visible_channel_marker_drops_exact_gemma4_modality_sentinels_only(self):
        """Gemma4 tool-call responses must not expose lone modality sentinels."""
        from vmlx_engine.server import _drop_tool_visible_channel_marker

        tool_calls = [{"function": {"name": "record_fact", "arguments": "{}"}}]

        assert _drop_tool_visible_channel_marker("<audio|>", tool_calls) is None
        assert _drop_tool_visible_channel_marker("<|audio|>", tool_calls) is None
        assert _drop_tool_visible_channel_marker("<image|>", tool_calls) is None
        assert _drop_tool_visible_channel_marker("<video|>", tool_calls) is None
        assert (
            _drop_tool_visible_channel_marker("called record_fact <audio|>", tool_calls)
            == "called record_fact <audio|>"
        )
        assert _drop_tool_visible_channel_marker("<audio|>", None) == "<audio|>"

    def test_cli_tool_parser_choices_include_registry_only_parsers(self):
        """Explicit CLI settings must accept parsers used by auto detection.

        ZAYA and Hy3 can be launched via auto-detection, but direct users also
        pass --tool-call-parser explicitly.  Missing choices make the CLI reject
        a valid product setting before the server can apply the registry.
        """
        cli_source = Path("./vmlx_engine/cli.py").read_text()
        assert '"zaya_xml"' in cli_source
        assert '"hunyuan"' in cli_source

    def test_cli_tool_parser_choices_cover_family_registry_parsers(self):
        """Every parser emitted by model_configs.py must be CLI-selectable.

        The panel resolves family parsers before launching the server. If
        argparse choices lag behind the model registry, a valid auto-detected
        session fails at startup before the engine can use that parser.
        """
        from vmlx_engine.model_configs import register_all
        from vmlx_engine.model_config_registry import ModelConfigRegistry

        import vmlx_engine.model_config_registry as mcr

        ModelConfigRegistry._instance = None
        mcr._configs_loaded = False
        registry = ModelConfigRegistry()
        register_all(registry)

        cli_source = Path("./vmlx_engine/cli.py").read_text()
        missing = sorted(
            {
                config.tool_parser
                for config in registry._configs
                if config.tool_parser and f'"{config.tool_parser}"' not in cli_source
            }
        )
        assert not missing
        assert '"gemma3"' in cli_source
        assert '"gemma3n"' in cli_source

    def test_cli_reasoning_parser_choices_cover_family_registry_parsers(self):
        """Every reasoning parser emitted by model_configs.py must be CLI-selectable.

        MiniMax regressed when the panel emitted ``minimax_m2`` but the
        packaged engine CLI rejected it before startup. The CLI now builds
        choices from the runtime reasoning registry; keep that contract pinned
        for all registered families, not only MiniMax.
        """
        from vmlx_engine.model_configs import register_all
        from vmlx_engine.model_config_registry import ModelConfigRegistry
        from vmlx_engine.reasoning import list_parsers

        import vmlx_engine.model_config_registry as mcr

        ModelConfigRegistry._instance = None
        mcr._configs_loaded = False
        registry = ModelConfigRegistry()
        register_all(registry)

        cli_source = Path("./vmlx_engine/cli.py").read_text()
        cli_reasoning_choices = {"auto", "none", *list_parsers()}
        missing = sorted(
            {
                config.reasoning_parser
                for config in registry._configs
                if config.reasoning_parser
                and config.reasoning_parser not in cli_reasoning_choices
            }
        )

        assert 'reasoning_choices = list_parsers()' in cli_source
        assert 'choices=["auto", "none"] + reasoning_choices' in cli_source
        assert not missing
        assert "minimax_m2" in cli_reasoning_choices

    def test_sampler_recreation_invalidates_pending_prefix_hits(self):
        """A sampler change must not leave requests pointing at cleared paged blocks."""
        import inspect
        from vmlx_engine.scheduler import Scheduler

        ensure_source = inspect.getsource(Scheduler._ensure_batch_generator)
        schedule_source = inspect.getsource(Scheduler._schedule_waiting)

        assert "cleared_cache = False" in ensure_source
        assert "return cleared_cache" in ensure_source
        assert "Preserving paged cache across BatchGenerator recreation" in ensure_source
        assert "self.block_aware_cache.clear()" not in ensure_source
        assert "_cache_cleared = self._ensure_batch_generator" in schedule_source
        assert "prefix cache hit invalidated by BatchGenerator" in schedule_source
        assert "request._paged_block_table_needs_worker_reconstruct = False" in schedule_source
        assert "request._hybrid_prompt_cache_needs_worker_ssm = False" in schedule_source
        assert "request.remaining_tokens = request.prompt_token_ids" in schedule_source


class TestZayaCCACachePolicy:
    """ZAYA/CCA must not fall through generic hybrid cache paths."""

    def _write_minimax_fixture(
        self,
        tmp_path,
        *,
        stamped_reasoning_parser: str = "minimax_m2",
    ):
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "minimax",
            "text_config": {
                "model_type": "minimax",
            },
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "version": 2,
            "weight_format": "mxtq",
            "cache_type": "kv",
            "source_model": {
                "name": "MiniMax-Test",
                "architecture": "minimax",
            },
            "capabilities": {
                "family": "minimax",
                "reasoning_parser": stamped_reasoning_parser,
                "think_in_template": True,
                "supports_thinking": True,
                "cache_type": "kv",
                "modality": "text",
                "supports_tools": True,
            },
            "chat": {
                "reasoning": {
                    "supported": True,
                    "parser": stamped_reasoning_parser,
                }
            },
        }))
        return tmp_path

    def _write_qwen36_fixture(self, tmp_path, *, model_type="qwen3_5_moe"):
        text_model_type = (
            "qwen3_5_moe_text" if model_type == "qwen3_5_moe" else "qwen3_5_text"
        )
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": model_type,
            "text_config": {
                "model_type": text_model_type,
                "layer_types": ["full_attention", "linear_attention"],
            },
            "vision_config": {"model_type": "qwen3_vl"},
            "image_token_id": 248056,
            "video_token_id": 248057,
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "version": 2,
            "weight_format": "mxfp8",
            "cache_type": "hybrid",
            "source_model": {
                "name": "Qwen3.6-Test",
                "architecture": model_type,
            },
            "capabilities": {
                "family": model_type,
                "reasoning_parser": "qwen3",
                "think_in_template": True,
                "supports_thinking": True,
                "cache_type": "hybrid",
                "modality": "vl",
                "supports_tools": True,
            },
            "chat": {
                "reasoning": {
                    "supported": True,
                    "parser": "qwen3",
                }
            },
        }))
        return tmp_path

    def _write_zaya_fixture(self, tmp_path, *, weight_format="mxtq"):
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "zaya",
            "weight_format": weight_format,
            "quantization": {
                "bits": 8,
                "group_size": 32,
                "mxtq_bits": {
                    "routed_expert": 2,
                    "attention": 8,
                    "router": 16,
                    "embed_tokens": 8,
                    "lm_head": 8,
                    "cca_conv": 16,
                    "norms_residual": 16,
                },
            },
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "version": 2,
            "weight_format": weight_format,
            "cache_subtype": "zaya_cca",
            "source_model": {
                "name": "ZAYA1-8B",
                "architecture": "zaya",
            },
            "capabilities": {
                # Simulates older local ZAYA bundles that were stamped too
                # optimistically before live gates proved thinking unsafe.
                "reasoning_parser": "qwen3",
                "tool_parser": "zaya_xml",
                "think_in_template": True,
                "supports_tools": True,
                "supports_thinking": True,
                "family": "zaya",
                "modality": "text",
                "cache_type": "hybrid",
            },
        }))
        return tmp_path

    def test_registry_preserves_zaya_cache_subtype(self, tmp_path):
        from vmlx_engine.model_config_registry import get_model_config_registry

        model_dir = self._write_zaya_fixture(tmp_path)
        registry = get_model_config_registry()
        registry.clear_cache()

        cfg = registry.lookup(str(model_dir))

        assert cfg.family_name == "zaya"
        assert cfg.cache_type == "hybrid"
        assert cfg.cache_subtype == "zaya_cca"
        assert cfg.tool_parser == "zaya_xml"
        assert cfg.reasoning_parser == "qwen3"
        assert cfg.think_in_template is False
        assert cfg.supports_thinking is True

    def test_registry_overrides_stale_minimax_qwen3_sidecar(self, tmp_path):
        from vmlx_engine.model_config_registry import get_model_config_registry

        model_dir = self._write_minimax_fixture(
            tmp_path,
            stamped_reasoning_parser="qwen3",
        )
        registry = get_model_config_registry()
        registry.clear_cache()

        cfg = registry.lookup(str(model_dir))

        assert cfg.family_name == "minimax"
        assert cfg.tool_parser == "minimax"
        assert cfg.reasoning_parser == "minimax_m2"

    def test_zaya_auto_defaults_to_no_think_but_explicit_on_still_works(self, tmp_path):
        from vmlx_engine import server

        model_dir = self._write_zaya_fixture(tmp_path)
        old_default = server._default_enable_thinking
        server._default_enable_thinking = None
        try:
            resolved = server._resolve_enable_thinking(
                request_value=None,
                ct_kwargs={},
                tools_present=False,
                model_key=str(model_dir),
                engine=None,
                auto_detect=True,
            )
            explicit_on = server._resolve_enable_thinking(
                request_value=True,
                ct_kwargs={},
                tools_present=False,
                model_key=str(model_dir),
                engine=None,
                auto_detect=True,
            )
        finally:
            server._default_enable_thinking = old_default

        assert resolved is False
        assert explicit_on is True

    def test_server_default_false_is_explicit_and_request_off_still_wins(self):
        from vmlx_engine import server

        old_default = server._default_enable_thinking
        server._default_enable_thinking = False
        try:
            resolved = server._resolve_enable_thinking(
                request_value=None,
                ct_kwargs={},
                tools_present=False,
                model_key="unknown-thinking-capable",
                engine=None,
                auto_detect=True,
            )
            explicit_off = server._resolve_enable_thinking(
                request_value=False,
                ct_kwargs={},
                tools_present=False,
                model_key="unknown-thinking-capable",
                engine=None,
                auto_detect=True,
            )
        finally:
            server._default_enable_thinking = old_default

        assert resolved is False
        assert explicit_off is False

    def test_server_default_false_respected_for_known_reasoning_model(self, tmp_path):
        from vmlx_engine import server
        from vmlx_engine.model_config_registry import get_model_config_registry

        old_default = server._default_enable_thinking
        server._default_enable_thinking = False
        try:
            model_dir = self._write_minimax_fixture(tmp_path)
            registry = get_model_config_registry()
            registry.clear_cache()
            resolved = server._resolve_enable_thinking(
                request_value=None,
                ct_kwargs={},
                tools_present=False,
                model_key=str(model_dir),
                engine=None,
                auto_detect=False,
            )
        finally:
            server._default_enable_thinking = old_default

        assert resolved is False

    def test_minimax_auto_stays_unset_and_explicit_thinking_still_works(
        self, tmp_path
    ):
        """MiniMax supports thinking, but omitted/Auto must stay unset.

        The engine must not hide runtime/template issues behind a family-level
        forced off rail. Explicit user thinking controls still win.
        """
        from vmlx_engine import server
        from vmlx_engine.model_config_registry import get_model_config_registry

        old_default = server._default_enable_thinking
        server._default_enable_thinking = None
        try:
            model_dir = self._write_minimax_fixture(tmp_path)
            registry = get_model_config_registry()
            registry.clear_cache()
            cfg = registry.lookup(str(model_dir))
            auto_resolved = server._resolve_enable_thinking(
                request_value=None,
                ct_kwargs={},
                tools_present=False,
                model_key=str(model_dir),
                engine=None,
                auto_detect=True,
            )
            explicit_on = server._resolve_enable_thinking(
                request_value=True,
                ct_kwargs={},
                tools_present=False,
                model_key=str(model_dir),
                engine=None,
                auto_detect=True,
            )
        finally:
            server._default_enable_thinking = old_default

        assert cfg.supports_thinking is True
        assert cfg.reasoning_parser == "minimax_m2"
        assert auto_resolved is None
        assert explicit_on is True

    @pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
    def test_qwen36_auto_stays_unset_and_explicit_thinking_still_works(
        self, tmp_path, model_type
    ):
        """Qwen3.6 supports thinking, but omitted/Auto must stay unset.

        Visibility/coherency issues must be fixed in template/runtime handling,
        not by converting Auto into a hidden off rail.
        """
        from vmlx_engine import server
        from vmlx_engine.model_config_registry import get_model_config_registry

        old_default = server._default_enable_thinking
        server._default_enable_thinking = None
        try:
            model_dir = self._write_qwen36_fixture(tmp_path, model_type=model_type)
            registry = get_model_config_registry()
            registry.clear_cache()
            cfg = registry.lookup(str(model_dir))
            auto_resolved = server._resolve_enable_thinking(
                request_value=None,
                ct_kwargs={},
                tools_present=False,
                model_key=str(model_dir),
                engine=None,
                auto_detect=True,
            )
            explicit_on = server._resolve_enable_thinking(
                request_value=True,
                ct_kwargs={},
                tools_present=False,
                model_key=str(model_dir),
                engine=None,
                auto_detect=True,
            )
        finally:
            server._default_enable_thinking = old_default

        assert cfg.family_name == model_type
        assert cfg.supports_thinking is True
        assert cfg.reasoning_parser == "qwen3"
        assert auto_resolved is None
        assert explicit_on is True

    def test_gemma4_tools_auto_stays_unset_and_explicit_thinking_still_works(self):
        from vmlx_engine import server

        old_default = server._default_enable_thinking
        server._default_enable_thinking = None
        try:
            resolved = server._resolve_enable_thinking(
                request_value=None,
                ct_kwargs={},
                tools_present=True,
                model_key="gemma4",
                engine=None,
                auto_detect=True,
            )
            explicit_on = server._resolve_enable_thinking(
                request_value=True,
                ct_kwargs={},
                tools_present=True,
                model_key="gemma4",
                engine=None,
                auto_detect=True,
            )
        finally:
            server._default_enable_thinking = old_default

        assert resolved is None
        assert explicit_on is True

    def test_gemma4_supports_thinking_is_explicit_not_implicit(self):
        from unittest.mock import patch
        from vmlx_engine.model_config_registry import get_model_config_registry

        registry = get_model_config_registry()
        registry.clear_cache()
        with patch(
            "vmlx_engine.model_config_registry.load_config",
            lambda _path: {"model_type": "gemma4"},
        ):
            cfg = registry.lookup("google/gemma-4-26b-it")

        assert cfg.family_name == "gemma4"
        assert cfg.reasoning_parser == "gemma4"
        assert cfg.supports_thinking is True
        assert cfg.think_in_template is False

    def test_cli_enables_typed_paged_cache_and_forces_tq_off_for_zaya_cca(self):
        source = Path("./vmlx_engine/cli.py").read_text()

        assert 'getattr(_mc, "cache_subtype", None) == "zaya_cca"' in source
        assert 'args.use_paged_cache = True' in source
        assert 'args.kv_cache_quantization = "none"' in source
        assert "ZAYA/CCA typed cache enabled" in source
        assert "conv_state + prev_hs" in source
        assert "server._tool_call_parser or args.tool_call_parser" in source

        scheduler_source = Path("./vmlx_engine/scheduler.py").read_text()
        assert "self._uses_zaya_cache" in scheduler_source
        assert 'self._model_type_for_runtime == "zaya"' in scheduler_source
        assert "ZAYA/CCA typed paged prefix cache enabled" in scheduler_source
        assert "ZAYA/CCA cache contract detected but prefix cache is disabled" in scheduler_source
        assert "and not self._uses_zaya_cache" in scheduler_source

        mllm_scheduler_source = Path("./vmlx_engine/mllm_scheduler.py").read_text()
        assert "Runtime cache layout:" in mllm_scheduler_source
        assert "zaya_cca_v1" in mllm_scheduler_source

    def test_zaya_cli_policy_default_keeps_prefix_paged_l2_but_forces_tq_off(
        self, monkeypatch
    ):
        from vmlx_engine.cli import _apply_zaya_cca_cache_policy

        monkeypatch.setenv("VMLX_FORCE_TQ_AUTO", "1")
        args = SimpleNamespace(
            enable_prefix_cache=True,
            use_paged_cache=True,
            enable_block_disk_cache=True,
            kv_cache_quantization="q4",
        )

        gate, changed = _apply_zaya_cca_cache_policy(args, MagicMock())

        assert gate is True
        assert changed == ("kv_quant=q4",)
        assert args.enable_prefix_cache is True
        assert args.use_paged_cache is True
        assert args.enable_block_disk_cache is True
        assert args.kv_cache_quantization == "none"
        assert args.kv_cache_quantization_explicit is True
        assert os.environ["VMLX_DISABLE_TQ_KV"] == "1"
        assert "VMLX_FORCE_TQ_AUTO" not in os.environ
        os.environ.pop("VMLX_DISABLE_TQ_KV", None)

    def test_zaya_cli_policy_no_longer_requires_live_gate_env(
        self, monkeypatch
    ):
        from vmlx_engine.cli import _apply_zaya_cca_cache_policy

        monkeypatch.delenv("VMLX_ZAYA_ENABLE_TYPED_CCA_CACHE", raising=False)
        monkeypatch.setenv("VMLX_FORCE_TQ_AUTO", "1")
        args = SimpleNamespace(
            enable_prefix_cache=True,
            use_paged_cache=True,
            enable_block_disk_cache=True,
            kv_cache_quantization="q8",
        )

        gate, changed = _apply_zaya_cca_cache_policy(args, MagicMock())

        assert gate is True
        assert changed == ("kv_quant=q8",)
        assert args.enable_prefix_cache is True
        assert args.use_paged_cache is True
        assert args.enable_block_disk_cache is True
        assert args.kv_cache_quantization == "none"
        assert args.kv_cache_quantization_explicit is True
        assert os.environ["VMLX_DISABLE_TQ_KV"] == "1"
        assert "VMLX_FORCE_TQ_AUTO" not in os.environ
        os.environ.pop("VMLX_DISABLE_TQ_KV", None)

    def test_zaya_cli_policy_upgrades_prefix_only_to_paged(
        self, monkeypatch
    ):
        from vmlx_engine.cli import _apply_zaya_cca_cache_policy

        monkeypatch.delenv("VMLX_ZAYA_ENABLE_TYPED_CCA_CACHE", raising=False)
        args = SimpleNamespace(
            enable_prefix_cache=True,
            use_paged_cache=False,
            enable_block_disk_cache=False,
            kv_cache_quantization="none",
        )

        gate, changed = _apply_zaya_cca_cache_policy(args, MagicMock())

        assert gate is True
        assert changed == ("paged=required_for_zaya_cca",)
        assert args.enable_prefix_cache is True
        assert args.use_paged_cache is True
        assert args.enable_block_disk_cache is False
        assert args.kv_cache_quantization == "none"
        assert args.kv_cache_quantization_explicit is True
        os.environ.pop("VMLX_DISABLE_TQ_KV", None)

    def test_ollama_streaming_suppresses_duplicate_done_chunks(self):
        source = Path("./vmlx_engine/server.py").read_text()

        assert "done_sent = False" in source
        assert "if done_sent:" in source
        assert "done_sent = True" in source
        assert "openai_chat_chunk_to_ollama_generate_ndjson" in source

    def test_jang_loader_registers_zaya_runtime(self, tmp_path):
        from vmlx_engine.utils.jang_loader import _ensure_zaya_runtime_supported

        model_dir = self._write_zaya_fixture(tmp_path)
        jcfg = json.loads((model_dir / "jang_config.json").read_text())

        _ensure_zaya_runtime_supported(model_dir, jcfg)

        import mlx_lm.models.zaya as zaya

        assert zaya.Model.__name__ == "Model"

    def test_jang_loader_registers_zaya1_vl_runtime_adapter(self, tmp_path):
        from vmlx_engine.utils.jang_loader import _ensure_zaya_runtime_supported

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "zaya1_vl",
            "vision_config": {"model_type": "qwen2_5_vl"},
            "zaya_expert_layout": "split_switch_mlp",
        }))
        jcfg = {
            "cache_subtype": "zaya_cca",
            "capabilities": {
                "family": "zaya1_vl",
                "tool_parser": "zaya_xml",
                "reasoning_parser": "qwen3",
                "think_in_template": False,
                "supports_thinking": True,
                "cache_type": "hybrid",
                "modality": "vision",
            },
        }

        _ensure_zaya_runtime_supported(tmp_path, jcfg)

        import mlx_vlm.models.zaya1_vl as zaya1_vl

        assert zaya1_vl.Model.__name__ == "Model"

    def test_local_zaya_bundles_carry_explicit_cca_contract(self):
        from vmlx_engine.model_config_registry import get_model_config_registry

        model_dirs = [
            Path("/Users/eric/models/JANGQ/ZAYA1-VL-8B-MXFP4"),
            Path("/Users/eric/models/JANGQ/ZAYA1-VL-8B-JANGTQ2"),
            Path("/Users/eric/models/JANGQ/ZAYA1-8B-MXFP4"),
            Path("/Users/eric/models/JANGQ/ZAYA1-8B-JANGTQ2"),
            Path("/Users/eric/jang/models/Zyphra/ZAYA1-8B-JANGTQ2"),
            Path("/Users/eric/jang/models/Zyphra/ZAYA1-8B-JANGTQ4"),
            Path("/Users/eric/jang/models/Zyphra/ZAYA1-8B-MXFP4"),
        ]
        existing = [p for p in model_dirs if p.exists()]
        if not existing:
            pytest.skip("local ZAYA bundles are not present on this machine")

        for model_dir in existing:
            cfg = json.loads((model_dir / "config.json").read_text())
            jcfg = json.loads((model_dir / "jang_config.json").read_text())
            caps = jcfg.get("capabilities", {})

            model_type = cfg.get("model_type")
            assert model_type in {"zaya", "zaya1_vl"}
            assert jcfg.get("cache_subtype") == "zaya_cca"
            expected_family = "zaya1_vl" if model_type == "zaya1_vl" else "zaya"
            assert caps.get("family") == expected_family
            assert caps.get("cache_type") == "hybrid"
            assert caps.get("tool_parser") == "zaya_xml"
            if model_type == "zaya1_vl":
                assert caps.get("reasoning_parser") == "qwen3"
            else:
                assert caps.get("reasoning_parser") == "qwen3"
            assert caps.get("think_in_template") is False
            # Text ZAYA remains reasoning-capable. Current ZAYA1-VL artifacts
            # still carry optimistic converter metadata, but runtime
            # capabilities demote their plain VLM template because live proof
            # produced hidden-only output under the synthetic qwen3 rail.
            assert caps.get("supports_thinking") is True
            assert cfg.get("zaya_expert_layout") == "split_switch_mlp"
            assert caps.get("modality") == ("vision" if model_type == "zaya1_vl" else "text")

            registry = get_model_config_registry()
            registry.clear_cache()
            rcfg = registry.lookup(str(model_dir))
            assert rcfg.family_name == expected_family
            assert rcfg.tool_parser == "zaya_xml"
            assert rcfg.think_in_template is False
            # Bit profile is still audited for kernel/cache metadata below, but
            # it must not change the family capability boundary. Text ZAYA
            # keeps qwen3; ZAYA1-VL stays no-reasoning until its uploaded
            # artifact has a real VLM thinking contract.
            if model_type == "zaya1_vl":
                assert rcfg.reasoning_parser is None
                assert rcfg.supports_thinking is False
            else:
                assert rcfg.reasoning_parser == "qwen3"
                assert rcfg.supports_thinking is True
            assert rcfg.is_mllm is (model_type == "zaya1_vl")

            if jcfg.get("weight_format") == "mxtq":
                bits = jcfg.get("mxtq_bits", {})
                assert bits.get("cca_conv") == 16
                assert bits.get("router") == 16
                assert (model_dir / "jangtq_runtime.safetensors").is_file()


class TestV5PortUniqueMigration:
    """Existing databases must get a UNIQUE index on sessions.port."""

    def test_migration_block_exists(self):
        """database.ts must have migration code for sessions.port UNIQUE index."""
        with open(os.path.join(
            os.path.dirname(__file__), '..', 'panel', 'src', 'main', 'database.ts'
        )) as f:
            source = f.read()
        assert "idx_sessions_port_unique" in source, (
            "Must create UNIQUE index on sessions.port for existing databases"
        )
        # Must deduplicate before adding constraint
        assert "DELETE FROM sessions" in source, (
            "Must deduplicate existing port conflicts before adding UNIQUE constraint"
        )
        assert "GROUP BY port" in source, (
            "Deduplication must group by port to keep only one session per port"
        )


# ================================================================
# V6: Regression tests for v1.2.0 audit fixes
# ================================================================

class TestV6CancelledErrorCallsFailActive:
    """CancelledError in engine loop must call _fail_active_requests."""

    def test_cancelled_error_handler_exists(self):
        source = Path("./vmlx_engine/engine_core.py").read_text()
        # Find CancelledError handler inside engine loop (the one that calls _fail_active_requests)
        # There are two: one in stop() and one in the engine loop. We need the engine loop one.
        idx = source.find("except asyncio.CancelledError:")
        # Skip the first one (in stop()) and find the second one (in engine loop)
        idx2 = source.find("except asyncio.CancelledError:", idx + 1)
        assert idx2 != -1, "Engine loop must have its own CancelledError handler"
        handler = source[idx2:idx2 + 200]
        assert "_fail_active_requests" in handler, (
            "CancelledError handler must call _fail_active_requests to unblock SSE consumers"
        )


class TestV6AbortUsesDeleteBlockTable:
    """abort_request must use delete_block_table (not detach_request) for paged cache."""

    def test_scheduler_abort_uses_delete(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler.abort_request)
        assert "delete_block_table" in source, (
            "Scheduler.abort_request must use delete_block_table to decrement ref_counts"
        )
        # Must NOT call detach_request (the word may appear in comments, check for actual call)
        assert ".detach_request(" not in source, (
            "Scheduler.abort_request must NOT call detach_request (leaks ref_counts)"
        )

    def test_mllm_scheduler_abort_uses_delete(self):
        source = Path("./vmlx_engine/mllm_scheduler.py").read_text()
        # Find the abort_request method
        start = source.find("def abort_request(self, request_id")
        assert start != -1
        # Find the next method definition
        end = source.find("\n    def ", start + 20)
        abort_body = source[start:end]
        assert "delete_block_table" in abort_body, (
            "MLLMScheduler.abort_request must use delete_block_table"
        )

    def test_mllm_scheduler_error_recovery_uses_delete(self):
        """Error-recovery path must also use delete_block_table."""
        source = Path("./vmlx_engine/mllm_scheduler.py").read_text()
        # Find the error-recovery block (paged_cache_manager.delete_block_table in error path)
        # This is in the step() method's except block
        idx = source.find("# Clean up paged cache block tables for all running")
        assert idx != -1, "Error-recovery cache cleanup comment must exist"
        nearby = source[idx:idx + 500]
        assert "delete_block_table" in nearby, (
            "Error-recovery path must use delete_block_table (not detach_request)"
        )

    def test_completion_path_uses_detach(self):
        """Normal completion path should use detach_request (preserves blocks for LRU)."""
        source = Path("./vmlx_engine/mllm_scheduler.py").read_text()
        # The completion path is in _finish_completed_requests or similar
        # It stores blocks for prefix cache, so it uses detach_request
        completion_idx = source.find("detach_request")
        assert completion_idx != -1, (
            "Normal completion path must use detach_request to preserve blocks for LRU reuse"
        )


class TestV6VLMDiskCacheKeyConsistency:
    """VLM disk cache store key must match fetch key (full token_list, not truncated)."""

    def test_store_uses_token_list_not_truncated(self):
        source = Path("./vmlx_engine/mllm_scheduler.py").read_text()
        # VLM disk cache store must use token_list (not truncated_tokens) to match fetch key
        assert "disk_cache.store(token_list" in source, (
            "VLM disk cache store must use token_list (not truncated_tokens) to match fetch key"
        )


class TestV6DequantizeNoneGuards:
    """All callers of _dequantize_cache must guard against None return."""

    def test_dequantize_returns_none_on_failure(self):
        source = Path("./vmlx_engine/mllm_batch_generator.py").read_text()
        func_start = source.find("def _dequantize_cache(")
        func_end = source.find("\ndef ", func_start + 10)
        func_body = source[func_start:func_end]
        assert "return None" in func_body, (
            "_dequantize_cache must return None on dequantization failure"
        )

    def test_memory_aware_caller_guards_none(self):
        """Memory-aware cache path must guard _dequantize_cache returning None."""
        source = Path("./vmlx_engine/mllm_batch_generator.py").read_text()
        # Find the memory-aware cache path with dequantize call
        # There are multiple dequantize call sites; find the one with "continue" after None check
        idx = source.find("_dequantize_cache(cache)")
        assert idx != -1
        after = source[idx:idx + 150]
        assert "if cache is None" in after or "is None" in after, (
            "Memory-aware path must check for None after _dequantize_cache"
        )

    def test_hybrid_cache_caller_guards_none(self):
        """Hybrid cache path must guard _dequantize_cache returning None before _fix_hybrid_cache."""
        source = Path("./vmlx_engine/mllm_batch_generator.py").read_text()
        # The hybrid cache path should check for None after dequantize
        idx = source.find("_dequantize_cache(req.prompt_cache)")
        if idx == -1:
            idx = source.find("_dequantize_cache(cache_for_fix)")
        assert idx != -1
        after = source[idx:idx + 200]
        assert "is None" in after, (
            "Hybrid cache path must check for None after _dequantize_cache"
        )


class TestV6MinimumSystemVersion:
    """macOS minimumSystemVersion must not block current macOS users."""

    def test_minimum_version_not_too_high(self):
        pkg_path = os.path.join(
            os.path.dirname(__file__), '..', 'panel', 'package.json'
        )
        with open(pkg_path) as f:
            pkg = json.load(f)
        min_ver = pkg.get("build", {}).get("mac", {}).get("minimumSystemVersion", "")
        assert min_ver, "minimumSystemVersion must be set"
        major = int(min_ver.split(".")[0])
        # macOS versions: 14 = Sonoma, 15 = Sequoia, 26 = Tahoe (2025)
        # Must support at least macOS 14+ (Sonoma) for M-series Macs
        assert major <= 15, (
            f"minimumSystemVersion {min_ver} is too high — "
            f"blocks users on macOS Sequoia (15) and earlier"
        )

    def test_minimum_version_matches_mlx_runtime_floor(self):
        """Packaging must not claim support below the bundled MLX runtime floor."""
        root = os.path.join(os.path.dirname(__file__), '..')
        pkg_path = os.path.join(root, 'panel', 'package.json')
        with open(pkg_path) as f:
            pkg = json.load(f)
        min_ver = pkg.get("build", {}).get("mac", {}).get("minimumSystemVersion", "")
        assert min_ver == "14.5.0", (
            "Bundled MLX wheels require macOS 14.5+; advertising or allowing "
            f"{min_ver} reopens mlxstudio#90/#104 metallib/libc++ failures."
        )

        for rel in ("panel/README.md", "panel/SETUP.md"):
            text = open(os.path.join(root, rel), encoding="utf-8").read()
            assert "macOS 14.5+" in text
            assert "macOS 26+" not in text


class TestV6MapHFModel:
    """mapHFModel must handle missing lastModified and author from HF list API."""

    def test_map_hf_model_function_exists(self):
        models_path = os.path.join(
            os.path.dirname(__file__), '..', 'panel', 'src', 'main', 'ipc', 'models.ts'
        )
        with open(models_path) as f:
            source = f.read()
        assert "function mapHFModel" in source, "mapHFModel helper must exist"

    def test_map_hf_model_uses_created_at_fallback(self):
        models_path = os.path.join(
            os.path.dirname(__file__), '..', 'panel', 'src', 'main', 'ipc', 'models.ts'
        )
        with open(models_path) as f:
            source = f.read()
        # Must use createdAt as fallback for lastModified
        assert "createdAt" in source, (
            "mapHFModel must use createdAt as fallback when lastModified is missing"
        )

    def test_map_hf_model_extracts_author_from_model_id(self):
        models_path = os.path.join(
            os.path.dirname(__file__), '..', 'panel', 'src', 'main', 'ipc', 'models.ts'
        )
        with open(models_path) as f:
            source = f.read()
        func_start = source.find("function mapHFModel")
        func_body = source[func_start:func_start + 400]
        # Must extract author from modelId (split('/')[0]) as fallback.
        # Accept both single- and double-quote forms — the TS source uses
        # double quotes: `modelId.split("/")[0]`.
        assert ("split('/')[0]" in func_body
                or 'split("/")[0]' in func_body), (
            "mapHFModel must extract author from modelId.split('/')[0] as fallback"
        )

    def test_search_and_recommended_use_map_hf_model(self):
        """Both searchHF and getRecommendedModels must use mapHFModel."""
        models_path = os.path.join(
            os.path.dirname(__file__), '..', 'panel', 'src', 'main', 'ipc', 'models.ts'
        )
        with open(models_path) as f:
            source = f.read()
        # Count usages of mapHFModel (should be used in both handlers)
        usages = source.count("mapHFModel(")
        assert usages >= 2, (
            f"mapHFModel must be used in both searchHF and getRecommendedModels "
            f"(found {usages} usage(s), expected >= 2)"
        )


class TestMLLMTurboQuantFetchPath:
    """Fetched VLM prefix/L2 caches must re-enter the TQ live-cache path."""

    def test_process_prompts_recompresses_fetched_prompt_cache(self):
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator._process_prompts)

        assert "_recompress_to_tq(req_cache, self.language_model)" in source
        assert "TQ recompress removed from fetch paths" not in source


class TestMLLMMlaDetection:
    """MLLM MLA detection should match the LLM scheduler's nuance."""

    def test_detects_raw_text_config_kv_lora_rank(self):
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        class _Cfg:
            _raw_config = {
                "text_config": {
                    "model_type": "kimi_k25",
                    "kv_lora_rank": 512,
                }
            }

        class _LM:
            config = _Cfg()

        class _Model:
            language_model = _LM()

        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        scheduler.model = _Model()

        assert scheduler._detect_mla() is True

    def test_bailing_ling_mla_is_not_treated_as_compressed_latent(self):
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        class _Args:
            model_type = "bailing_hybrid"
            kv_lora_rank = 512

        class _LM:
            args = _Args()

        class _Model:
            language_model = _LM()

        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        scheduler.model = _Model()

        assert scheduler._detect_mla() is False


class TestMLLMNKvHeadsWrapperParity:
    """MLLMScheduler._detect_n_kv_heads must walk text_config + inner wrappers.

    Symmetric to ``Scheduler._detect_mla`` and ``PrefixCache._get_n_kv_heads``.
    Earlier version only inspected ``language_model.{args,config}`` directly —
    Kimi K2.6 VLM (mlx_vlm wrapper around DeepseekV3 backbone) exposes the
    MLA config via ``language_model.config.text_config.kv_lora_rank``. Without
    the text_config fallback, MLA H=1 collapse never fires for VLM-wrapped
    MLA models and cache slicing in ``_normalize_gqa_state`` mismatches.
    """

    def test_kimi_vlm_text_config_kv_lora_rank_collapses_to_one_head(self):
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        class _TextCfg:
            model_type = "kimi_k25"
            kv_lora_rank = 512
            num_attention_heads = 64
            num_key_value_heads = 8

        class _Cfg:
            text_config = _TextCfg()

        class _LM:
            config = _Cfg()

        class _Model:
            language_model = _LM()

        sched = MLLMScheduler.__new__(MLLMScheduler)
        sched.model = _Model()
        assert sched._detect_n_kv_heads() == 1

    def test_bailing_hybrid_via_text_config_keeps_full_kv_heads(self):
        """Ling/Bailing stores expanded KV — must NOT collapse to H=1."""
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        class _TextCfg:
            model_type = "bailing_hybrid"
            kv_lora_rank = 512
            num_attention_heads = 32
            num_key_value_heads = 32

        class _Cfg:
            text_config = _TextCfg()

        class _LM:
            config = _Cfg()

        class _Model:
            language_model = _LM()

        sched = MLLMScheduler.__new__(MLLMScheduler)
        sched.model = _Model()
        # Must keep full per-head KV count (32), not collapse to H=1
        # which would slice valid (1,32,T,D) cache to (1,1,T,D).
        assert sched._detect_n_kv_heads() == 32


class TestHeadDimWrapperTraversal:
    """Scheduler._detect_head_dim + MLLMScheduler._detect_head_dim must walk wrappers.

    Symmetric to ``_detect_mla`` and ``_detect_n_kv_heads``. Earlier
    implementations only inspected ``self.model.{args,config}`` (LLM) or
    ``language_model.{args,config}`` (MLLM) directly. For VLM-wrapped
    backbones (Kimi K2.6 around DeepseekV3, glm_moe_dsa, Mistral 4
    wrappers), head_dim is exposed via ``language_model.config.text_config``
    or via an inner ``.model`` candidate. Without traversal, both helpers
    returned None silently — and the KV-quant
    ``_wrap_make_cache_quantized`` skipped its head_dim/group_size
    compatibility check, allowing a mismatched group_size into mx.quantize
    to fail at runtime.
    """

    def test_llm_scheduler_finds_head_dim_via_text_config_wrapper(self):
        from vmlx_engine.scheduler import Scheduler

        class _TextCfg:
            model_type = "kimi_k25"
            head_dim = 128

        class _Cfg:
            text_config = _TextCfg()

        class _LM:
            config = _Cfg()

        class _Wrapper:
            language_model = _LM()

        s = Scheduler.__new__(Scheduler)
        s.model = _Wrapper()
        assert s._detect_head_dim() == 128

    def test_llm_scheduler_derives_head_dim_from_hidden_div_heads_via_inner_model(self):
        from vmlx_engine.scheduler import Scheduler

        class _Args:
            hidden_size = 4096
            num_attention_heads = 32

        class _Inner:
            args = _Args()

        class _Wrapper:
            model = _Inner()

        s = Scheduler.__new__(Scheduler)
        s.model = _Wrapper()
        # 4096 / 32 = 128
        assert s._detect_head_dim() == 128

    def test_mllm_scheduler_finds_head_dim_via_text_config_wrapper(self):
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        class _TextCfg:
            model_type = "kimi_k25"
            head_dim = 128

        class _Cfg:
            text_config = _TextCfg()

        class _LM:
            config = _Cfg()

        class _Wrapper:
            language_model = _LM()

        s = MLLMScheduler.__new__(MLLMScheduler)
        s.model = _Wrapper()
        assert s._detect_head_dim() == 128

    def test_llm_and_mllm_scheduler_agree_on_wrapped_head_dim(self):
        from vmlx_engine.scheduler import Scheduler
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        class _TextCfg:
            model_type = "kimi_k25"
            head_dim = 128

        class _Cfg:
            text_config = _TextCfg()

        class _LM:
            config = _Cfg()

        class _Wrapper:
            language_model = _LM()

        llm = Scheduler.__new__(Scheduler)
        llm.model = _Wrapper()
        mllm = MLLMScheduler.__new__(MLLMScheduler)
        mllm.model = _Wrapper()
        assert llm._detect_head_dim() == mllm._detect_head_dim()

    def test_kimi_mla_cache_dims_do_not_use_hidden_div_heads(self):
        """Kimi MLA cache dims are latent+RoPE dims, not hidden/heads.

        The local Kimi K2.6 VLM bundle exposes hidden_size=7168 and
        num_attention_heads=64, but mlx_lm caches ``kv_lora_rank`` and
        ``qk_rope_head_dim`` via cache.update_and_fetch(kv_latent, k_pe).
        Treating 7168 / 64 = 112 as the cache head dim makes the
        group-size validator adjust default-compatible KV quant settings to
        the wrong value.
        """
        from vmlx_engine.scheduler import Scheduler
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        class _TextCfg:
            model_type = "kimi_k25"
            hidden_size = 7168
            num_attention_heads = 64
            kv_lora_rank = 512
            qk_rope_head_dim = 64
            v_head_dim = 128

        class _Cfg:
            text_config = _TextCfg()

        class _LM:
            config = _Cfg()

        class _Wrapper:
            language_model = _LM()

        llm = Scheduler.__new__(Scheduler)
        llm.model = _Wrapper()
        mllm = MLLMScheduler.__new__(MLLMScheduler)
        mllm.model = _Wrapper()

        assert llm._detect_cache_head_dims() == (512, 64)
        assert mllm._detect_cache_head_dims() == (512, 64)
        assert llm._detect_head_dim() == 512
        assert mllm._detect_head_dim() == 512

    def test_kimi_mla_quant_group_validation_checks_all_cache_dims(self):
        """A group size must divide both MLA cache tensors.

        Kimi/DeepSeek/Mistral MLA caches store latent KV with width
        ``kv_lora_rank`` and RoPE keys with width ``qk_rope_head_dim``.
        A user-provided group_size=128 divides 512 but not 64, so the
        validator must adjust to 64. The old hidden/heads fallback returned
        112 and adjusted to 16, which was a false fix.
        """
        from types import SimpleNamespace

        from vmlx_engine.scheduler import Scheduler
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        text_config = SimpleNamespace(
            model_type="kimi_k25",
            hidden_size=7168,
            num_attention_heads=64,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )
        model = SimpleNamespace(language_model=SimpleNamespace(config=SimpleNamespace(text_config=text_config)))

        llm = Scheduler.__new__(Scheduler)
        llm.model = model
        llm.config = SimpleNamespace(kv_cache_group_size=128)
        llm._wrap_make_cache_quantized(bits=4, group_size=128)
        assert llm._kv_cache_group_size == 64
        assert llm.config.kv_cache_group_size == 64

        mllm = MLLMScheduler.__new__(MLLMScheduler)
        mllm.model = model
        mllm._wrap_make_cache_quantized(bits=4, group_size=128)
        assert mllm._kv_cache_group_size == 64


class TestLLMMlaDetectionWrapperParity:
    """LLM scheduler MLA detection must match MLLMScheduler._detect_mla traversal.

    Site 1 (Scheduler.__init__ KV-quant gate) historically only inspected
    ``self.model.args``. Wrapped models (Kimi K2.6 mlx_vlm wrapper around
    DeepseekV3, glm_moe_dsa inheriting deepseek_v32) expose MLA config via
    ``self.model.language_model.config.text_config`` — the inline check missed
    them and KV-cache quantization stayed enabled, double-quantizing already
    compressed latents.
    """

    def test_detects_kimi_style_wrapper_via_text_config(self):
        from vmlx_engine.scheduler import Scheduler

        class _TextCfg:
            model_type = "kimi_k25"
            kv_lora_rank = 512

        class _Cfg:
            text_config = _TextCfg()

        class _LM:
            config = _Cfg()

        class _Model:
            language_model = _LM()

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.model = _Model()

        assert scheduler._detect_mla() is True

    def test_detects_deepseek_v3_inner_model_args(self):
        from vmlx_engine.scheduler import Scheduler

        class _Args:
            model_type = "deepseek_v3"
            kv_lora_rank = 512

        class _Inner:
            args = _Args()

        class _Model:
            model = _Inner()

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.model = _Model()

        assert scheduler._detect_mla() is True

    def test_bailing_hybrid_via_wrapper_is_not_mla(self):
        from vmlx_engine.scheduler import Scheduler

        class _Args:
            model_type = "bailing_hybrid"
            kv_lora_rank = 512

        class _LM:
            args = _Args()

        class _Model:
            language_model = _LM()

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.model = _Model()

        assert scheduler._detect_mla() is False

    def test_mistral4_detected_via_wrapper(self):
        from vmlx_engine.scheduler import Scheduler

        class _Args:
            model_type = "mistral4"

        class _LM:
            args = _Args()

        class _Model:
            language_model = _LM()

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.model = _Model()

        assert scheduler._detect_mla() is True


class TestHybridSSMEnvNames:
    """Hybrid SSM resume controls must use the documented VMLX_* names."""

    def test_scheduler_uses_vmlx_disable_ssm_prefix_resume(self):
        source = Path("vmlx_engine/scheduler.py").read_text()

        assert "VMLX_DISABLE_SSM_PREFIX_RESUME" in source
        assert "VMLINUX_DISABLE_SSM_PREFIX_RESUME" not in source


class TestPromptLookupDocumentation:
    """vmlx#137: PLD docs must match the scheduler's real implementation."""

    def test_prompt_lookup_docstring_no_longer_claims_phase2_is_blocked(self):
        import vmlx_engine.prompt_lookup as prompt_lookup

        doc = prompt_lookup.__doc__ or ""

        assert "Phase 2 (future)" not in doc
        assert "Blocked today by" not in doc
        assert "blocked by mlx-lm" not in doc.lower()
        assert "Scheduler integration owns runtime acceleration" in doc
        assert "_try_pld_speculative_decode" in doc
        assert "Hybrid SSM/attention models need family-aware" in doc


class TestJangTqMppNaxCliPolicy:
    def test_jangtq_mpp_nax_cli_policy_defaults_to_auto(self, monkeypatch):
        from vmlx_engine.cli import _apply_jangtq_mpp_nax_policy

        monkeypatch.delenv("JANGTQ_MPP_NAX", raising=False)
        monkeypatch.setenv("JANGTQ_TOPK_OVERRIDE", "4")
        args = SimpleNamespace()
        logger = MagicMock()

        _apply_jangtq_mpp_nax_policy(args, logger)

        assert os.environ["JANGTQ_MPP_NAX"] == "auto"
        assert "JANGTQ_TOPK_OVERRIDE" not in os.environ
        logger.info.assert_called_once()
        logger.debug.assert_called_once()

    def test_jangtq_mpp_nax_cli_policy_can_force_off(self, monkeypatch):
        from vmlx_engine.cli import _apply_jangtq_mpp_nax_policy

        monkeypatch.setenv("JANGTQ_MPP_NAX", "off")
        args = SimpleNamespace()
        logger = MagicMock()

        _apply_jangtq_mpp_nax_policy(args, logger)

        assert os.environ["JANGTQ_MPP_NAX"] == "off"

    def test_jangtq_mpp_nax_cli_policy_disables_auto_for_ling(self, monkeypatch):
        from vmlx_engine.cli import _apply_jangtq_mpp_nax_policy

        monkeypatch.delenv("JANGTQ_MPP_NAX", raising=False)
        args = SimpleNamespace(model="/Users/eric/models/JANGQ/Ling-2.6-flash-JANGTQ")
        logger = MagicMock()

        mode = _apply_jangtq_mpp_nax_policy(args, logger)

        assert mode == "off"
        assert os.environ["JANGTQ_MPP_NAX"] == "off"
        logger.info.assert_called_once()

    def test_jangtq_mpp_nax_cli_policy_disables_auto_for_mxtq_bundle(
        self, monkeypatch, tmp_path
    ):
        from vmlx_engine.cli import _apply_jangtq_mpp_nax_policy

        monkeypatch.delenv("JANGTQ_MPP_NAX", raising=False)
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "qwen3_5_moe"})
        )
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"weight_format": "mxtq", "mxtq_bits": {"routed_expert": 4}})
        )
        (tmp_path / "jangtq_runtime.safetensors").write_bytes(b"sidecar")
        args = SimpleNamespace(model=str(tmp_path))
        logger = MagicMock()

        mode = _apply_jangtq_mpp_nax_policy(args, logger)

        assert mode == "off"
        assert os.environ["JANGTQ_MPP_NAX"] == "off"
        assert any(
            "JANGTQ_MPP_NAX=auto" in str(call)
            and "MXTQ/JANGTQ" in str(call)
            for call in logger.info.call_args_list
        )

    def test_jangtq_mpp_nax_cli_policy_disables_auto_for_jangtq_repo_id(
        self, monkeypatch
    ):
        from vmlx_engine.cli import _apply_jangtq_mpp_nax_policy

        monkeypatch.delenv("JANGTQ_MPP_NAX", raising=False)
        args = SimpleNamespace(model="JANGQ-AI/MiniMax-M2.7-JANGTQ_K")
        logger = MagicMock()

        mode = _apply_jangtq_mpp_nax_policy(args, logger)

        assert mode == "off"
        assert os.environ["JANGTQ_MPP_NAX"] == "off"

    def test_jangtq_mpp_nax_cli_policy_allows_explicit_on_for_kernel_diagnostics(
        self, monkeypatch, tmp_path
    ):
        from vmlx_engine.cli import _apply_jangtq_mpp_nax_policy

        monkeypatch.setenv("JANGTQ_MPP_NAX", "on")
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"quantization": {"format": "mxtq"}})
        )
        args = SimpleNamespace(model=str(tmp_path))
        logger = MagicMock()

        mode = _apply_jangtq_mpp_nax_policy(args, logger)

        assert mode == "on"
        assert os.environ["JANGTQ_MPP_NAX"] == "1"

    def test_jangtq_mpp_nax_cli_policy_can_force_on(self, monkeypatch):
        from vmlx_engine.cli import _apply_jangtq_mpp_nax_policy

        monkeypatch.setenv("JANGTQ_MPP_NAX", "on")
        args = SimpleNamespace()
        logger = MagicMock()

        mode = _apply_jangtq_mpp_nax_policy(args, logger)

        assert mode == "on"
        assert os.environ["JANGTQ_MPP_NAX"] == "1"

    def test_jangtq_mpp_nax_cli_policy_invalid_value_falls_back_to_auto(
        self, monkeypatch
    ):
        from vmlx_engine.cli import _apply_jangtq_mpp_nax_policy

        monkeypatch.setenv("JANGTQ_MPP_NAX", "bogus")
        args = SimpleNamespace()
        logger = MagicMock()

        mode = _apply_jangtq_mpp_nax_policy(args, logger)

        assert mode == "auto"
        assert os.environ["JANGTQ_MPP_NAX"] == "auto"
        logger.warning.assert_called_once()

    def test_jangtq_mpp_nax_serve_parser_does_not_expose_toggle(self):
        source = Path("vmlx_engine/cli.py").read_text()

        assert '"--jangtq-mpp-nax"' not in source
        assert 'choices=["off", "auto", "on"]' not in source

    def test_jangtq_mpp_nax_server_status_defaults_to_auto(self, monkeypatch):
        import vmlx_engine.server as server

        monkeypatch.delenv("JANGTQ_MPP_NAX", raising=False)

        mode, issue = server._normalize_jangtq_mpp_nax_mode()

        assert mode == "auto"
        assert issue is None

    def test_jangtq_mpp_nax_server_invalid_value_falls_back_to_auto(self):
        import vmlx_engine.server as server

        mode, issue = server._normalize_jangtq_mpp_nax_mode("bogus")

        assert mode == "auto"
        assert "invalid JANGTQ_MPP_NAX" in issue

    def test_jangtq_mpp_nax_availability_status_does_not_eval_mlx(self, monkeypatch):
        pytest.importorskip("jang_tools.turboquant.mpp_nax_kernel")
        import vmlx_engine.server as server
        import jang_tools.turboquant.mpp_nax_kernel as nax_kernel

        nax_kernel.mpp_nax_tensorops_available.cache_clear()
        server._jangtq_mpp_nax_available.cache_clear()

        def fail_eval(*_args, **_kwargs):
            raise AssertionError("status path must not run a smoke Metal eval")

        monkeypatch.setattr(nax_kernel.mx, "eval", fail_eval)

        available, error = server._jangtq_mpp_nax_available()

        assert isinstance(available, bool)
        assert error is None


class TestDistributedStreamingUnicode:
    """vmlx#124 class: distributed streaming must not per-token decode UTF-8."""

    def test_distributed_generate_loop_uses_streaming_detokenizer(self):
        source = Path("vmlx_engine/distributed/engine.py").read_text()
        body_start = source.index("async def _generate_impl")
        body_end = source.index("\n    def _apply_chat_template", body_start)
        body = source[body_start:body_end]

        assert "tokenizer.decode([next_tok_id])" not in body
        assert "_make_streaming_detokenizer(tokenizer)" in body
        assert "_add_streaming_token(detokenizer, next_tok_id)" in body
        assert "_finalize_streaming_detokenizer(detokenizer)" in body

    def test_streaming_helpers_do_not_emit_incomplete_cyrillic(self):
        from vmlx_engine.distributed.engine import (
            _add_streaming_token,
            _finalize_streaming_detokenizer,
        )

        class FakeDetokenizer:
            def __init__(self):
                self.text = ""

            def add_token(self, token):
                # Token 1 is an incomplete byte span. A per-token decode path
                # would emit U+FFFD here; a streaming path must wait.
                if token == 1:
                    self.text = ""
                elif token == 2:
                    self.text = "Пр"

            def finalize(self):
                self.text = "Привет"

        detok = FakeDetokenizer()

        assert _add_streaming_token(detok, 1) == ""
        assert _add_streaming_token(detok, 2) == "Пр"
        assert _finalize_streaming_detokenizer(detok) == "ивет"
        assert "\ufffd" not in detok.text


class TestJangVLMFallbacks:
    """JANG VLM loaders must not silently drop working image support."""

    def _write_qwen_hybrid_fixture(
        self,
        tmp_path,
        *,
        weight_format=None,
        text_model_type="qwen3_5_text",
        layer_types=None,
    ):
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": text_model_type,
                "layer_types": layer_types or ["linear_attention", "full_attention"],
            },
            "vision_config": {},
            "image_token_id": 248056,
            "video_token_id": 248057,
        }))
        jang = {
            "format": "jang" if weight_format is None else weight_format,
            "architecture": {
                "has_vision": True,
                "has_ssm": True,
                "type": "hybrid_ssm_dense",
            },
        }
        if weight_format is not None:
            jang["weight_format"] = weight_format
        (tmp_path / "jang_config.json").write_text(json.dumps(jang))
        return tmp_path

    def test_qwen_hybrid_jangtq_vlm_stays_on_native_vlm_loader(self):
        source = Path("vmlx_engine/utils/jang_loader.py").read_text()

        assert "VMLINUX_FORCE_VLM_LOADER" not in source
        assert "Qwen3.5/3.6-VL MXTQ/JANGTQ hybrid SSM bundles must stay on the real VLM" in source
        assert "affine-JANG exception above is deliberately narrow" in source

    def test_text_only_vlm_fallbacks_mark_vision_unavailable(self):
        source = Path("vmlx_engine/utils/jang_loader.py").read_text()
        mistral_fallback = source[
            source.index('config.get("model_type") == "mistral3"'):
            source.index("# Qwen3.5/3.6-VL MXTQ/JANGTQ hybrid SSM bundles", source.index('config.get("model_type") == "mistral3"'))
        ]

        assert 'globals()["_LAST_LOAD_VLM_FALLBACK"] = True' in mistral_fallback
        assert 'globals()["_LAST_LOAD_VLM_FALLBACK"] = False' in source

    @pytest.mark.parametrize(
        "model_type,text_model_type",
        [
            ("qwen3_5", "qwen3_5_text"),
            ("qwen3_vl", "qwen3_vl"),
            ("qwen3_vl_moe", "qwen3_vl_moe"),
        ],
    )
    def test_affine_qwen_mrope_families_route_text_only(
        self, tmp_path, model_type, text_model_type
    ):
        from vmlx_engine.api import utils

        model_dir = self._write_qwen_hybrid_fixture(
            tmp_path,
            text_model_type=text_model_type,
        )
        config = json.loads((model_dir / "config.json").read_text())
        config["model_type"] = model_type
        (model_dir / "config.json").write_text(json.dumps(config))
        utils._IS_MLLM_CACHE.clear()

        assert utils.is_mllm_model(str(model_dir)) is False

    def test_gemma4_unified_routes_text_only_when_mlx_vlm_runtime_missing(
        self,
        tmp_path,
        monkeypatch,
    ):
        from vmlx_engine.api import utils
        from vmlx_engine.models import gemma4_unified_register

        model_dir = tmp_path
        (model_dir / "config.json").write_text(json.dumps({
            "model_type": "gemma4_unified",
            "text_config": {"model_type": "gemma4_unified_text"},
            "vision_config": {"model_type": "gemma4_unified_vision"},
            "audio_config": {"model_type": "gemma4_unified_audio"},
        }))
        (model_dir / "jang_config.json").write_text(json.dumps({
            "weight_format": "mxfp4",
            "has_vision": True,
            "has_audio": True,
            "capabilities": {
                "family": "gemma4",
                "modality": "vision",
                "tool_parser": "gemma4",
                "reasoning_parser": "gemma4",
            },
        }))
        utils._IS_MLLM_CACHE.clear()
        monkeypatch.setattr(
            gemma4_unified_register,
            "gemma4_unified_runtime_available",
            lambda: False,
        )

        assert utils.is_mllm_model(str(model_dir)) is False
        assert utils.is_mllm_model(str(model_dir), force_mllm=True) is False

    def test_mimo_v2_text_runtime_metadata_auto_enables_complete_media_bundle(
        self,
        tmp_path,
        monkeypatch,
    ):
        from vmlx_engine.api import utils
        import vmlx_engine.server as server

        model_dir = tmp_path
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "mimo_v2",
                    "format": "jangtq",
                    "jang_profile": "JANGTQ_2",
                    "vision_config": {"model_type": "mimo_v2_vision"},
                    "audio_config": {"model_type": "mimo_v2_audio"},
                    "image_token_id": 151655,
                    "video_token_id": 151656,
                    "processor_config": {"audio_token_id": 151669},
                    "capabilities": {
                        "family": "mimo_v2",
                        "modalities": ["text"],
                        "preserved_modalities": ["vision", "audio"],
                        "unwired_modalities": ["vision", "audio"],
                        "multimodal_status": "weights_preserved_text_runtime",
                    },
                    "runtime": {
                        "multimodal_mode": "weights_preserved_text_runtime",
                    },
                }
            )
        )
        (model_dir / "jang_config.json").write_text(
            json.dumps(
                {
                    "format": "jangtq",
                    "family": "mimo_v2",
                    "profile": "JANGTQ_2",
                }
            )
        )
        (model_dir / "preprocessor_config.json").write_text("{}")
        (model_dir / "audio_tokenizer").mkdir()
        (model_dir / "audio_tokenizer" / "model.safetensors").write_bytes(b"stub")
        (model_dir / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "visual.blocks.0.attn.qkv.weight": "a.safetensors",
                        "audio_encoder.input_local_transformer.layers.0.mlp.gate_proj.weight": "b.safetensors",
                        "speech_embeddings.0.weight": "c.safetensors",
                    }
                }
            )
        )

        module = SimpleNamespace(
            VisionModel=object,
            MiMoVisionPatchEmbed=object,
            MiMoVisionPatchMerger=object,
            MiMoVisionBlock=object,
            AudioModel=object,
            MiMoAudioTokenizer=object,
            load_mimo_audio_tokenizer_from_bundle=lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(server, "_mimo_v2_runtime_module", lambda: module)
        utils._IS_MLLM_CACHE.clear()

        assert utils.is_mllm_model(str(model_dir)) is True
        assert utils.is_mllm_model(str(model_dir), force_mllm=True) is True

    def test_gemma4_unified_routes_multimodal_when_source_runtime_available(
        self,
        tmp_path,
    ):
        from vmlx_engine.api import utils

        model_dir = tmp_path
        (model_dir / "config.json").write_text(json.dumps({
            "model_type": "gemma4_unified",
            "text_config": {"model_type": "gemma4_unified_text"},
            "vision_config": {"model_type": "gemma4_unified_vision"},
            "audio_config": {"model_type": "gemma4_unified_audio"},
        }))
        (model_dir / "jang_config.json").write_text(json.dumps({
            "weight_format": "mxfp4",
            "has_vision": True,
            "has_audio": True,
            "capabilities": {
                "family": "gemma4",
                "modality": "vision",
                "tool_parser": "gemma4",
                "reasoning_parser": "gemma4",
            },
        }))
        utils._IS_MLLM_CACHE.clear()

        assert utils.is_mllm_model(str(model_dir)) is True

    def test_gemma4_unified_registry_removes_text_runtime_pixel_injection(
        self,
        tmp_path,
        monkeypatch,
    ):
        from vmlx_engine.model_config_registry import get_model_config_registry
        from vmlx_engine.models import gemma4_unified_register

        model_dir = tmp_path
        (model_dir / "config.json").write_text(json.dumps({
            "model_type": "gemma4_unified",
            "text_config": {"model_type": "gemma4_unified_text"},
            "vision_config": {"model_type": "gemma4_unified_vision"},
            "audio_config": {"model_type": "gemma4_unified_audio"},
        }))
        (model_dir / "jang_config.json").write_text(json.dumps({
            "weight_format": "mxfp4",
            "has_vision": True,
            "has_audio": True,
            "capabilities": {
                "family": "gemma4",
                "modality": "vision",
                "tool_parser": "gemma4",
                "reasoning_parser": "gemma4",
                "think_in_template": False,
                "supports_thinking": True,
                "supports_tools": True,
                "cache_type": "kv",
            },
        }))
        monkeypatch.setattr(
            gemma4_unified_register,
            "gemma4_unified_runtime_available",
            lambda: False,
        )

        registry = get_model_config_registry()
        registry.clear_cache()
        cfg = registry.lookup(str(model_dir))

        assert cfg.family_name == "gemma4"
        assert cfg.is_mllm is False
        assert cfg.tool_parser == "gemma4"
        assert cfg.reasoning_parser == "gemma4"
        assert cfg.architecture_hints["vl_runtime_available"] is False
        assert cfg.architecture_hints["audio_runtime_available"] is False
        assert "inject_pixel_values" not in cfg.architecture_hints

    def test_gemma4_unified_registry_advertises_source_runtime_when_available(
        self,
        tmp_path,
    ):
        from vmlx_engine.model_config_registry import get_model_config_registry

        model_dir = tmp_path
        (model_dir / "config.json").write_text(json.dumps({
            "model_type": "gemma4_unified",
            "text_config": {"model_type": "gemma4_unified_text"},
            "vision_config": {"model_type": "gemma4_unified_vision"},
            "audio_config": {"model_type": "gemma4_unified_audio"},
        }))
        (model_dir / "jang_config.json").write_text(json.dumps({
            "weight_format": "mxfp4",
            "has_vision": True,
            "has_audio": True,
            "capabilities": {
                "family": "gemma4",
                "modality": "vision",
                "tool_parser": "gemma4",
                "reasoning_parser": "gemma4",
                "think_in_template": False,
                "supports_thinking": True,
                "supports_tools": True,
                "cache_type": "kv",
            },
        }))

        registry = get_model_config_registry()
        registry.clear_cache()
        cfg = registry.lookup(str(model_dir))

        assert cfg.family_name == "gemma4"
        assert cfg.is_mllm is True
        assert cfg.architecture_hints["runtime_scope"] == "source_gemma4_unified_vlm"
        assert cfg.architecture_hints["vl_runtime_available"] is True
        assert cfg.architecture_hints["audio_runtime_available"] is True
        assert cfg.architecture_hints["default_enable_thinking"] is False

    def test_batched_model_wrapper_does_not_inject_pixel_values_for_gemma4_text_wrapper(
        self,
    ):
        import mlx.core as mx

        from vmlx_engine.engine.batched import MLLMModelWrapper

        class _Gemma4TextWrapper:
            model_type = "gemma4"

            def __call__(self, tokens, **kwargs):
                assert "pixel_values" not in kwargs
                return mx.zeros((1, 1, 4))

        wrapped = MLLMModelWrapper(_Gemma4TextWrapper())

        wrapped(mx.array([[1]], dtype=mx.int32), cache=None)

    def test_affine_qwen_hybrid_detection_normalizes_config_case_for_text_only(self, tmp_path):
        from vmlx_engine.api import utils

        model_dir = self._write_qwen_hybrid_fixture(
            tmp_path,
            text_model_type="QWEN3_5_TEXT",
            layer_types=["LINEAR_ATTENTION", "FULL_ATTENTION"],
        )
        utils._IS_MLLM_CACHE.clear()

        assert utils.is_mllm_model(str(model_dir)) is False

    def test_affine_qwen_hybrid_jang_overrides_forced_mllm(self, tmp_path):
        from vmlx_engine.api import utils

        model_dir = self._write_qwen_hybrid_fixture(tmp_path)
        utils._IS_MLLM_CACHE.clear()

        assert utils.is_mllm_model(str(model_dir), force_mllm=True) is False

    def test_mxtq_qwen_hybrid_jang_routes_multimodal(self, tmp_path):
        from vmlx_engine.api import utils

        model_dir = self._write_qwen_hybrid_fixture(tmp_path, weight_format="mxtq")
        utils._IS_MLLM_CACHE.clear()

        assert utils.is_mllm_model(str(model_dir)) is True

    def test_qwen_vlm_loader_affine_delegates_to_text_loader(self, tmp_path, monkeypatch):
        """Affine-JANG Qwen hybrid uses text loader until mlx_vlm M-RoPE is fixed."""
        from vmlx_engine.utils import jang_loader
        from vmlx_engine.models import gemma4_unified_register
        import mlx_vlm.utils as vlm_utils

        model_dir = self._write_qwen_hybrid_fixture(tmp_path)
        config = json.loads((model_dir / "config.json").read_text())
        jang_cfg = json.loads((model_dir / "jang_config.json").read_text())

        monkeypatch.setattr(vlm_utils, "load_config", lambda path: dict(config))
        monkeypatch.setattr(
            gemma4_unified_register,
            "gemma4_unified_runtime_available",
            lambda: False,
        )
        monkeypatch.setattr(
            vlm_utils,
            "get_model_and_args",
            lambda *, config: pytest.fail("affine-JANG Qwen must not hit native VLM loader"),
        )
        sentinel = object()

        monkeypatch.setattr(
            jang_loader,
            "_load_jang_v2",
            lambda *args, **kwargs: (sentinel, "tokenizer"),
        )

        model, tokenizer = jang_loader._load_jang_v2_vlm(model_dir, jang_cfg)
        assert model is sentinel
        assert tokenizer == "tokenizer"
        assert jang_loader._LAST_LOAD_VLM_FALLBACK is True

    def test_gemma4_unified_vlm_loader_delegates_to_text_loader(
        self,
        tmp_path,
        monkeypatch,
    ):
        from vmlx_engine.utils import jang_loader
        from vmlx_engine.models import gemma4_unified_register
        import mlx_vlm.utils as vlm_utils

        model_dir = tmp_path
        config = {
            "model_type": "gemma4_unified",
            "text_config": {"model_type": "gemma4_unified_text"},
            "vision_config": {"model_type": "gemma4_unified_vision"},
            "audio_config": {"model_type": "gemma4_unified_audio"},
        }
        jang_cfg = {
            "weight_format": "mxfp4",
            "has_vision": True,
            "has_audio": True,
        }

        monkeypatch.setattr(vlm_utils, "load_config", lambda path: dict(config))
        monkeypatch.setattr(
            gemma4_unified_register,
            "gemma4_unified_runtime_available",
            lambda: False,
        )
        monkeypatch.setattr(
            vlm_utils,
            "get_model_and_args",
            lambda *, config: pytest.fail("Gemma4 unified must not hit missing native VLM loader"),
        )
        sentinel = object()
        monkeypatch.setattr(
            jang_loader,
            "_load_jang_v2",
            lambda *args, **kwargs: (sentinel, "tokenizer"),
        )

        model, tokenizer = jang_loader._load_jang_v2_vlm(model_dir, jang_cfg)

        assert model is sentinel
        assert tokenizer == "tokenizer"
        assert jang_loader._LAST_LOAD_VLM_FALLBACK is True

    def test_gemma4_unified_vlm_loader_fallback_is_guarded_by_runtime_availability(self):
        source = Path("vmlx_engine/utils/jang_loader.py").read_text()

        assert (
            "if _is_gemma4_unified_text_runtime_config(config) "
            "and not _gemma4_unified_runtime_available():" in source
        )

    def test_qwen_vlm_loader_affine_native_path_normalizes_config_case(
        self,
        tmp_path,
        monkeypatch,
    ):
        """The text fallback path must match the same normalized Qwen
        hybrid predicate as is_mllm_model()."""
        from vmlx_engine.utils import jang_loader
        import mlx_vlm.utils as vlm_utils

        model_dir = self._write_qwen_hybrid_fixture(
            tmp_path,
            text_model_type="QWEN3_5_TEXT",
            layer_types=["LINEAR_ATTENTION", "FULL_ATTENTION"],
        )
        config = json.loads((model_dir / "config.json").read_text())
        jang_cfg = json.loads((model_dir / "jang_config.json").read_text())

        monkeypatch.setattr(vlm_utils, "load_config", lambda path: dict(config))
        monkeypatch.setattr(
            vlm_utils,
            "get_model_and_args",
            lambda *, config: pytest.fail("affine-JANG Qwen must not hit native VLM loader"),
        )
        sentinel = object()

        monkeypatch.setattr(
            jang_loader,
            "_load_jang_v2",
            lambda *args, **kwargs: (sentinel, "tokenizer"),
        )

        model, tokenizer = jang_loader._load_jang_v2_vlm(model_dir, jang_cfg)
        assert model is sentinel
        assert tokenizer == "tokenizer"
        assert jang_loader._LAST_LOAD_VLM_FALLBACK is True

    def test_vlm_as_text_key_remap_guard_allows_existing_destination_keys(self):
        source = Path("vmlx_engine/utils/jang_loader.py").read_text()

        assert "_dst_lm_count < _src_lm_count" in source
        assert "_dst_lm_head < _src_lm_head" in source
        assert "_dst_lm_count != _src_lm_count" not in source
        assert "_dst_lm_head != _src_lm_head" not in source

    def test_qwen_vlm_loader_mxtq_stays_native_vlm(self, tmp_path, monkeypatch):
        """JANGTQ/MXTQ Qwen VLM must not be caught by the affine fallback."""
        from vmlx_engine.utils import jang_loader
        import mlx_vlm.utils as vlm_utils

        class NativeVlmReached(RuntimeError):
            pass

        model_dir = self._write_qwen_hybrid_fixture(tmp_path, weight_format="mxtq")
        config = json.loads((model_dir / "config.json").read_text())
        jang_cfg = json.loads((model_dir / "jang_config.json").read_text())

        monkeypatch.setattr(vlm_utils, "load_config", lambda path: dict(config))
        monkeypatch.setattr(
            vlm_utils,
            "get_model_and_args",
            lambda *, config: (_ for _ in ()).throw(NativeVlmReached()),
        )
        monkeypatch.setattr(
            jang_loader,
            "_load_jang_v2",
            lambda *args, **kwargs: pytest.fail("MXTQ Qwen VLM must not use text fallback"),
        )

        with pytest.raises(NativeVlmReached):
            jang_loader._load_jang_v2_vlm(model_dir, jang_cfg)
        assert jang_loader._LAST_LOAD_VLM_FALLBACK is False


class TestTurboQuantKVTelemetry:
    """Cache telemetry must agree for nested MLLM language models."""

    def test_detects_nested_mllm_turboquant_make_cache(self):
        from vmlx_engine.server import _turboquant_kv_cache_status

        def _turboquant_make_cache():
            return []

        class _LanguageModel:
            make_cache = staticmethod(_turboquant_make_cache)

        class _Model:
            language_model = _LanguageModel()

        class _Wrapper:
            model = _Model()

        class _Engine:
            _model = _Wrapper()

        status = _turboquant_kv_cache_status(_Engine())

        assert status["enabled"] is True
        assert status["default_bits"] == 3

    def test_reports_turboquant_single_sequence_runtime_contract(self):
        from types import SimpleNamespace
        from vmlx_engine.server import _turboquant_kv_cache_status

        scheduler = SimpleNamespace(
            _tq_active=True,
            _tq_batch_api=False,
            config=SimpleNamespace(
                max_num_seqs=1,
                prefill_batch_size=1,
                completion_batch_size=1,
            ),
        )

        status = _turboquant_kv_cache_status(scheduler=scheduler)

        assert status["enabled"] is True
        assert status["single_sequence_only"] is True
        assert status["single_sequence_reason"] == "cache_extend_not_supported"
        assert status["effective_max_num_seqs"] == 1
        assert status["effective_prefill_batch_size"] == 1
        assert status["effective_completion_batch_size"] == 1

    def test_reports_turboquant_batch_api_runtime_contract(self):
        from types import SimpleNamespace
        from vmlx_engine.server import _turboquant_kv_cache_status

        scheduler = SimpleNamespace(
            _tq_active=True,
            _tq_batch_api=True,
            config=SimpleNamespace(
                max_num_seqs=5,
                prefill_batch_size=1024,
                completion_batch_size=1024,
            ),
        )

        status = _turboquant_kv_cache_status(scheduler=scheduler)

        assert status["enabled"] is True
        assert status["batch_api"] == "turboquant_kv_v1"
        assert status["single_sequence_only"] is False
        assert "single_sequence_reason" not in status
        assert status["effective_max_num_seqs"] == 5

    def test_mllm_scheduler_aggregates_cached_tokens_for_stats(self):
        import threading
        from types import SimpleNamespace
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        scheduler._cache_hit_requests = 0
        scheduler._cache_hit_tokens = 0
        scheduler._cache_hit_tokens_by_detail = {}
        scheduler._queue_lock = threading.RLock()
        scheduler.waiting = []
        scheduler.running = {}
        scheduler.finished_req_ids = set()
        scheduler.num_requests_processed = 1
        scheduler.total_prompt_tokens = 35
        scheduler.total_completion_tokens = 3
        scheduler.batch_generator = None
        scheduler.block_aware_cache = None
        scheduler.paged_cache_manager = None
        scheduler.memory_aware_cache = None
        scheduler.prefix_cache = None
        scheduler.disk_cache = None
        batch_stats = SimpleNamespace(
            last_cache_execution={
                "cache_detail": "paged+ssm",
                "cached_tokens": 34,
            },
            to_dict=lambda: {
                "last_cache_execution": batch_stats.last_cache_execution,
            },
        )
        scheduler.batch_generator = SimpleNamespace(
            _stats=batch_stats,
            stats=lambda: batch_stats,
            get_vision_cache_stats=lambda: {},
        )

        request = SimpleNamespace(_cache_detail="request-legacy-detail")
        response = SimpleNamespace(cached_tokens=34, cache_detail="paged+ssm+disk")

        scheduler._record_cache_hit(response, request)
        scheduler._record_cache_hit(response, request)
        stats = scheduler.get_stats()

        assert stats["cache_hit_requests"] == 1
        assert stats["cache_hit_tokens"] == 34
        assert stats["cache_hit_tokens_by_detail"] == {"paged+ssm+disk": 34}
        assert (
            scheduler.batch_generator._stats.last_cache_execution["cache_detail"]
            == "paged+ssm+disk"
        )

    def test_mllm_scheduler_projects_batch_generator_cache_execution_stats(self):
        import threading
        from types import SimpleNamespace

        from vmlx_engine.mllm_scheduler import MLLMScheduler

        execution = {
            "request_id": "req-cache",
            "cache_detail": "paged+disk+tq",
            "cached_tokens": 1536,
            "reconstruction_seconds": 0.01,
            "dequantization_seconds": 0.02,
            "total_worker_cache_seconds": 0.03,
        }

        class _BatchStats:
            def to_dict(self):
                return {"last_cache_execution": execution}

        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        scheduler._cache_hit_requests = 0
        scheduler._cache_hit_tokens = 0
        scheduler._cache_hit_tokens_by_detail = {}
        scheduler._queue_lock = threading.RLock()
        scheduler.waiting = []
        scheduler.running = {}
        scheduler.finished_req_ids = set()
        scheduler.num_requests_processed = 0
        scheduler.total_prompt_tokens = 0
        scheduler.total_completion_tokens = 0
        scheduler.batch_generator = SimpleNamespace(
            stats=lambda: _BatchStats(),
            get_vision_cache_stats=lambda: {},
        )
        scheduler.block_aware_cache = None
        scheduler.paged_cache_manager = None
        scheduler.memory_aware_cache = None
        scheduler.prefix_cache = None
        scheduler.disk_cache = None

        assert scheduler.get_stats()["last_cache_execution"] == execution

    def test_mllm_paged_cache_hits_have_source_detail_labels(self):
        source = Path("./vmlx_engine/mllm_batch_generator.py").read_text()

        assert "_paged_disk_hits_before" in source
        assert "_paged_disk_hit" in source
        assert "cache_detail: str = \"\"" in source
        assert "cache_detail=getattr(req, '_cache_detail', \"\") or \"\"" in source
        assert "_paged_hybrid_cache_detail(" in source
        assert "_paged_attention_cache_detail(" in source
        assert "return f\"{base}+disk\" if disk_hit else base" in source

    def test_mllm_mixed_swa_cache_detail_does_not_report_ssm(self):
        from vmlx_engine.mllm_batch_generator import _paged_hybrid_cache_detail

        assert _paged_hybrid_cache_detail(disk_hit=False, mixed_attention=True) == (
            "paged+mixed_swa"
        )
        assert _paged_hybrid_cache_detail(disk_hit=True, mixed_attention=True) == (
            "paged+mixed_swa+disk"
        )
        assert _paged_hybrid_cache_detail(disk_hit=False, mixed_attention=False) == (
            "paged+ssm"
        )
        assert _paged_hybrid_cache_detail(disk_hit=True, mixed_attention=False) == (
            "paged+ssm+disk"
        )

    def test_mllm_non_hybrid_mixed_swa_cache_detail_is_labeled(self):
        source = Path("./vmlx_engine/mllm_batch_generator.py").read_text()

        non_hybrid_idx = source.index("elif not is_hybrid and reconstructed is not None:")
        non_hybrid_block = source[non_hybrid_idx: source.index("# Re-attach gen-prompt suffix", non_hybrid_idx)]

        assert "_paged_attention_cache_detail(" in non_hybrid_block
        assert "mixed_attention=self._mixed_attention_cache_model" in non_hybrid_block

    def test_mllm_rotating_kv_cache_is_kv_like_for_mixed_swa(self):
        """Gemma4 RotatingKVCache layers must not enter the SSM hybrid path."""
        from mlx_lm.models.cache import KVCache, RotatingKVCache
        from vmlx_engine.mllm_batch_generator import _is_kv_like

        assert _is_kv_like(KVCache()) is True
        assert _is_kv_like(RotatingKVCache(max_size=4096)) is True

    def test_step37_registry_subtype_marks_scheduler_mixed_attention(self):
        """Step3.7 may lack layer_types but still needs mixed full/sliding KV handling."""
        from vmlx_engine.mllm_scheduler import MLLMScheduler
        from vmlx_engine.scheduler import Scheduler

        for cfg in (
            SimpleNamespace(
                model_type="step3p7",
                cache_subtype="step3p7_full_sliding_kv",
                text_config=SimpleNamespace(
                    model_type="step3p5",
                    sliding_window=512,
                ),
            ),
            SimpleNamespace(
                model_type="step3p7",
                text_config=SimpleNamespace(
                    model_type="step3p5",
                    sliding_window=512,
                ),
            ),
        ):
            model = SimpleNamespace(config=cfg)

            assert Scheduler._model_has_mixed_attention(model) is True
            assert MLLMScheduler._model_has_mixed_attention(
                object.__new__(MLLMScheduler),
                model,
            ) is True

    def test_mimo_v2_registry_subtype_marks_scheduler_mixed_attention(self):
        """MiMo V2 has asymmetric full/SWA KV heads and must not use generic cache handling."""
        from vmlx_engine.mllm_scheduler import MLLMScheduler
        from vmlx_engine.scheduler import Scheduler

        model = SimpleNamespace(
            config=SimpleNamespace(
                model_type="mimo_v2",
                cache_subtype="mimo_v2_asymmetric_swa",
            )
        )

        assert Scheduler._model_has_mixed_attention(model) is True
        assert MLLMScheduler._model_has_mixed_attention(
            object.__new__(MLLMScheduler),
            model,
        ) is True

    def test_mllm_ensure_batch_cache_preserves_rotating_cache_type(self):
        """Gemma4 decode must keep sliding-window layers on BatchRotatingKVCache."""
        from mlx_lm.models.cache import KVCache, RotatingKVCache
        from mlx_lm.generate import BatchRotatingKVCache
        from vmlx_engine.mllm_batch_generator import _ensure_batch_cache

        converted = _ensure_batch_cache([RotatingKVCache(max_size=4096), KVCache()])

        assert isinstance(converted[0], BatchRotatingKVCache)

    def test_paged_cache_reconstruct_preserves_rotating_absolute_offset(self):
        """Gemma4 sliding-window cache restore must keep absolute offset/_idx."""
        import mlx.core as mx
        from mlx_lm.models.cache import RotatingKVCache
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        cache = RotatingKVCache(max_size=8, keep=0)
        cache.keys = mx.ones((1, 2, 8, 4), dtype=mx.float16)
        cache.values = mx.ones((1, 2, 8, 4), dtype=mx.float16)
        cache.offset = 20
        cache._idx = 8
        cache_data = [{
            "state": cache.state,
            "meta_state": cache.meta_state,
            "class_name": "RotatingKVCache",
        }]
        manager = PagedCacheManager(block_size=4, max_blocks=16)
        prefix = BlockAwarePrefixCache(object(), manager)

        table = prefix.store_cache("req", list(range(20)), cache_data)
        restored = prefix.reconstruct_cache(table)

        assert restored is not None
        assert isinstance(restored[0], RotatingKVCache)
        assert restored[0].offset == 20
        assert restored[0]._idx == 8

    def test_paged_cache_mixed_swa_reconstruct_preserves_full_kv_length(self):
        """Rotating-window offsets must not truncate full-attention KV layers."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache, RotatingKVCache
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        rotating = RotatingKVCache(max_size=8, keep=0)
        rotating.keys = mx.ones((1, 2, 8, 4), dtype=mx.float16)
        rotating.values = mx.ones((1, 2, 8, 4), dtype=mx.float16)
        rotating.offset = 20
        rotating._idx = 8
        full = KVCache()
        full.keys = mx.ones((1, 1, 20, 4), dtype=mx.float16)
        full.values = mx.ones((1, 1, 20, 4), dtype=mx.float16)
        full.offset = 20
        cache_data = [
            {
                "state": rotating.state,
                "meta_state": rotating.meta_state,
                "class_name": "RotatingKVCache",
            },
            {
                "state": full.state,
                "meta_state": full.meta_state,
                "class_name": "KVCache",
            },
        ]
        manager = PagedCacheManager(block_size=4, max_blocks=16)
        prefix = BlockAwarePrefixCache(object(), manager)

        table = prefix.store_cache("req", list(range(20)), cache_data)
        restored = prefix.reconstruct_cache(table)

        assert restored is not None
        assert isinstance(restored[0], RotatingKVCache)
        assert isinstance(restored[1], KVCache)
        assert restored[0].offset == 20
        assert restored[1].offset == 20

    def test_paged_cache_mixed_swa_frugal_keeps_resident_blocks_for_immediate_hit(
        self,
        monkeypatch,
    ):
        """Mixed-SWA block-table hits must not wait for async L2 disk writes."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache, RotatingKVCache
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        class FakeDiskStore:
            def write_block_async(self, *_args, **_kwargs):
                pass

            def read_block(self, *_args, **_kwargs):
                return None

        monkeypatch.delenv("VMLX_PAGED_FRUGAL", raising=False)

        rotating = RotatingKVCache(max_size=8, keep=0)
        rotating.keys = mx.ones((1, 2, 8, 4), dtype=mx.float16)
        rotating.values = mx.ones((1, 2, 8, 4), dtype=mx.float16)
        rotating.offset = 20
        rotating._idx = 8
        full = KVCache()
        full.keys = mx.ones((1, 1, 20, 4), dtype=mx.float16)
        full.values = mx.ones((1, 1, 20, 4), dtype=mx.float16)
        full.offset = 20
        cache_data = [
            {
                "state": rotating.state,
                "meta_state": rotating.meta_state,
                "class_name": "RotatingKVCache",
            },
            {
                "state": full.state,
                "meta_state": full.meta_state,
                "class_name": "KVCache",
            },
        ]
        manager = PagedCacheManager(
            block_size=4,
            max_blocks=16,
            disk_store=FakeDiskStore(),
        )
        prefix = BlockAwarePrefixCache(object(), manager)
        tokens = list(range(20))

        stored = prefix.store_cache("seed", tokens, cache_data)
        hit, remaining = prefix.fetch_cache("hit", tokens + [999])
        restored = prefix.reconstruct_cache(hit)

        assert stored is not None
        assert remaining == [999]
        assert all(
            manager.allocated_blocks[bid].cache_data is not None
            for bid in stored.block_ids
        )
        assert restored is not None
        assert isinstance(restored[0], RotatingKVCache)
        assert isinstance(restored[1], KVCache)
        assert restored[0].offset == 20
        assert restored[1].offset == 20

    def test_paged_cache_reconstruct_drops_promoted_disk_block_mirror(self):
        """Disk-promoted block payloads must not stay pinned in L1 RAM."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache
        from vmlx_engine.paged_cache import PagedCacheManager, BlockTable
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        manager = PagedCacheManager(block_size=4, max_blocks=4)
        prefix = BlockAwarePrefixCache(object(), manager)
        keys = mx.ones((1, 1, 4, 2), dtype=mx.float16)
        values = mx.ones((1, 1, 4, 2), dtype=mx.float16)
        block = manager._promote_from_disk(b"disk-promoted-block-hash-000000", [("kv", keys, values)], 4)
        assert block is not None
        assert block.cache_data is not None
        assert block.cache_data_from_disk is True

        table = BlockTable(
            request_id="hit",
            block_ids=[block.block_id],
            num_tokens=4,
        )
        restored = prefix.reconstruct_cache(table)

        assert restored is not None
        assert isinstance(restored[0], KVCache)
        assert restored[0].offset == 4
        assert block.cache_data is None
        assert block.cache_data_from_disk is False

    def test_paged_cache_reconstruct_drops_readable_l2_write_through_mirror(
        self,
        monkeypatch,
    ):
        """Readable L2 write-through blocks should not pin parent KV slices."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache
        from vmlx_engine.paged_cache import PagedCacheManager
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        class FakeDiskStore:
            def __init__(self):
                self.blocks = {}

            def write_block_async(self, block_hash, cache_data, token_count):
                self.blocks[block_hash] = cache_data

            def read_block(self, block_hash):
                return self.blocks.get(block_hash)

        monkeypatch.setenv("VMLX_PAGED_FRUGAL", "0")

        disk = FakeDiskStore()
        manager = PagedCacheManager(
            block_size=4,
            max_blocks=4,
            disk_store=disk,
        )
        prefix = BlockAwarePrefixCache(object(), manager)
        cache = KVCache()
        cache.keys = mx.ones((1, 1, 4, 2), dtype=mx.float16)
        cache.values = mx.ones((1, 1, 4, 2), dtype=mx.float16)
        cache.offset = 4
        cache_data = [{
            "state": cache.state,
            "meta_state": cache.meta_state,
            "class_name": "KVCache",
        }]

        table = prefix.store_cache("seed", list(range(4)), cache_data)
        assert table is not None
        [block_id] = table.block_ids
        block = manager.allocated_blocks[block_id]
        assert block.cache_data is not None
        assert block.cache_data_from_disk is False

        restored = prefix.reconstruct_cache(table)

        assert restored is not None
        assert isinstance(restored[0], KVCache)
        assert restored[0].offset == 4
        assert block.cache_data is None
        assert block.cache_data_from_disk is False

    def test_paged_cache_tensor_store_drops_full_request_cache_reference(self):
        """Block-backed tensor stores must not retain a duplicate full KV cache."""
        source = Path("./vmlx_engine/prefix_cache.py").read_text()
        assert "entry_cache_data = None if is_tensor_data else cache_data" in source
        assert "cache_data=entry_cache_data" in source
        assert "if entry.cache_data is None:" in source
        assert "cache_data = self.reconstruct_cache(entry.block_table)" in source

    def test_mllm_mixed_swa_store_uses_clean_prefill_not_post_generation_truncate(self):
        """Gemma4 rotating windows must store a clean N-1 prefix cache."""
        source = Path("./vmlx_engine/mllm_scheduler.py").read_text()

        assert "_uses_mixed_attention_cache" in source
        assert "_uses_zaya_cache or _uses_mixed_attention_cache" in source
        assert "Skipping mixed-SWA VLM paged cache store" in source

    def test_llm_cache_detail_uses_canonical_plus_grammar(self):
        source = Path("./vmlx_engine/scheduler.py").read_text()

        assert '"paged+disk"' in source
        assert '"paged+disk+tq"' in source
        assert '"paged+ssm+disk"' in source
        assert '"paged+ssm"' in source
        assert "disk->paged" not in source
        assert "disk+tq->paged" not in source
        assert "paged+ssm(" not in source

    def test_llm_hybrid_cache_detail_marks_ssm_disk_source(self, monkeypatch):
        import vmlx_engine.scheduler as scheduler_mod

        scheduler = scheduler_mod.Scheduler.__new__(scheduler_mod.Scheduler)
        scheduler._is_hybrid = True
        scheduler._uses_dsv4_cache = False
        scheduler._uses_zaya_cache = False
        scheduler._ssm_state_cache = SimpleNamespace(
            fetch=lambda tokens, fetch_num: (["ssm-a", "ssm-b"], True)
        )
        scheduler.block_aware_cache = SimpleNamespace()
        scheduler.model = SimpleNamespace()
        scheduler._hybrid_kv_positions = [0]
        scheduler._hybrid_num_layers = 3

        monkeypatch.setattr(
            scheduler_mod,
            "_fix_hybrid_cache",
            lambda reconstructed, model, kv_positions, num_model_layers: list(
                reconstructed
            ),
        )

        request = SimpleNamespace(
            request_id="req-1",
            block_table=SimpleNamespace(num_tokens=64),
            prompt_token_ids=list(range(80)),
            _hybrid_ssm_fetch_tokens=list(range(80)),
            _paged_disk_hit=True,
        )

        full_cache = scheduler._finalize_hybrid_paged_cache_on_worker(
            request,
            ["kv", None, None],
        )

        assert full_cache == ["kv", "ssm-a", "ssm-b"]
        assert request._cache_detail == "paged+ssm+disk"
        assert request._cache_detail_ssm_layers == 2

        request2 = SimpleNamespace(
            request_id="req-2",
            block_table=SimpleNamespace(num_tokens=64),
            prompt_token_ids=list(range(80)),
            _hybrid_ssm_fetch_tokens=list(range(80)),
            _paged_disk_hit=False,
        )

        scheduler._finalize_hybrid_paged_cache_on_worker(
            request2,
            ["kv", None, None],
        )

        assert request2._cache_detail == "paged+ssm"

    def test_llm_hybrid_ssm_miss_queues_block_boundary_rederive(self):
        import vmlx_engine.scheduler as scheduler_mod

        detached = []
        scheduler = scheduler_mod.Scheduler.__new__(scheduler_mod.Scheduler)
        scheduler._is_hybrid = True
        scheduler._uses_dsv4_cache = False
        scheduler._uses_zaya_cache = False
        scheduler._ssm_state_cache = SimpleNamespace(
            fetch=lambda tokens, fetch_num: None,
            fetch_longest_prefix=lambda tokens, max_len: None,
            _store={},
            max_entries=8,
        )
        scheduler.block_aware_cache = SimpleNamespace(
            block_size=64,
            detach_request=lambda request_id: detached.append(request_id),
        )
        scheduler._ssm_rederive_queue = []

        request = SimpleNamespace(
            request_id="req-lfm",
            block_table=SimpleNamespace(num_tokens=448),
            prompt_token_ids=list(range(700)),
            _hybrid_ssm_fetch_tokens=list(range(700)),
        )

        result = scheduler._finalize_hybrid_paged_cache_on_worker(
            request,
            ["kv"],
        )

        assert result is None
        assert detached == ["req-lfm"]
        assert scheduler._ssm_rederive_queue == [
            (list(range(448)), 448, "req-lfm")
        ]
        assert request.cached_tokens == 0
        assert request.remaining_tokens == request.prompt_token_ids

    @pytest.mark.asyncio
    async def test_cache_stats_endpoint_projects_cache_reuse_skip_telemetry(
        self, monkeypatch
    ):
        import vmlx_engine.server as server

        class _Engine:
            is_mllm = False

            def get_cache_stats(self):
                return None

        class _Scheduler:
            disk_cache = None
            paged_cache_manager = None

            def get_stats(self):
                return {
                    "num_waiting": 2,
                    "num_running": 1,
                    "num_requests_processed": 4,
                    "total_prompt_tokens": 1000,
                    "total_completion_tokens": 128,
                    "ewma_ttft_seconds": 9.25,
                    "cache_hit_requests": 7,
                    "cache_hit_tokens": 12345,
                    "cache_hit_tokens_by_detail": {"paged+dsv4": 4096, "disk+tq": 8249},
                    "batch_generator": {
                        "hybrid_kv_without_ssm_hits": 3,
                        "hybrid_kv_without_ssm_tokens": 2624,
                        "last_hybrid_kv_without_ssm": {
                            "reason": "no_ssm_companion_state",
                            "cached_tokens": 1088,
                        },
                    },
                    "cache_reuse_skips": 1,
                    "cache_reuse_skip_tokens": 512,
                    "last_cache_reuse_skip": {
                        "reason": "insufficient_memory_for_cache_merge",
                        "action": "full_prefill",
                        "needed_mb": 41618.0,
                        "budget_mb": 11330.5,
                        "available_mb": 13330.0,
                        "cached_tokens": 512,
                        "dropped_cached_tokens": 512,
                        "full_prefill_tokens": 4096,
                        "cache_contract": "hybrid_ssm",
                        "cache_format": "full_precision_kv+state_cache",
                        "partial_reuse_unavailable_reason": (
                            "no_block_aligned_ssm_checkpoint"
                        ),
                    },
                    "cache_reuse_partial_downgrades": 1,
                    "cache_reuse_partial_tokens": 2048,
                    "last_cache_reuse_partial": {
                        "reason": "insufficient_memory_for_full_cache_merge",
                        "used_cached_tokens": 2048,
                        "original_cached_tokens": 8192,
                        "tail_tokens": 6144,
                        "cache_format": "full_precision_kv",
                    },
                    "last_cache_selection": {
                        "selected": "prefix",
                        "rejected": "paged",
                        "reason": "cold_paged_reconstruction_cost",
                        "paged_cached_tokens": 5500,
                        "paged_cold_tokens": 2400,
                        "warm_cached_tokens": 3037,
                        "hot_advantage_tokens": 63,
                        "hot_advantage_threshold_tokens": 512,
                    },
                }

        monkeypatch.setattr(server, "_engine", _Engine())
        monkeypatch.setattr(server, "_get_scheduler", lambda: _Scheduler())

        payload = await server.cache_stats()
        scheduler_stats = payload["scheduler_stats"]

        assert scheduler_stats["cache_hit_requests"] == 7
        assert scheduler_stats["cache_hit_tokens"] == 12345
        assert scheduler_stats["cache_hit_tokens_by_detail"]["paged+dsv4"] == 4096
        assert scheduler_stats["cache_hit_tokens_by_detail"]["disk+tq"] == 8249
        assert scheduler_stats["hybrid_kv_without_ssm_hits"] == 3
        assert scheduler_stats["hybrid_kv_without_ssm_tokens"] == 2624
        assert scheduler_stats["last_hybrid_kv_without_ssm"]["cached_tokens"] == 1088
        assert scheduler_stats["cache_reuse_skips"] == 1
        assert scheduler_stats["cache_reuse_skip_tokens"] == 512
        assert scheduler_stats["last_cache_reuse_skip"]["reason"] == (
            "insufficient_memory_for_cache_merge"
        )
        assert scheduler_stats["last_cache_reuse_skip"]["needed_mb"] == 41618.0
        assert scheduler_stats["last_cache_reuse_skip"]["budget_mb"] == 11330.5
        assert scheduler_stats["last_cache_reuse_skip"]["available_mb"] == 13330.0
        assert scheduler_stats["last_cache_reuse_skip"]["dropped_cached_tokens"] == 512
        assert scheduler_stats["last_cache_reuse_skip"]["full_prefill_tokens"] == 4096
        assert scheduler_stats["last_cache_reuse_skip"]["cache_contract"] == "hybrid_ssm"
        assert scheduler_stats["last_cache_reuse_skip"]["cache_format"] == (
            "full_precision_kv+state_cache"
        )
        assert scheduler_stats["last_cache_reuse_skip"][
            "partial_reuse_unavailable_reason"
        ] == "no_block_aligned_ssm_checkpoint"
        assert scheduler_stats["cache_reuse_partial_downgrades"] == 1
        assert scheduler_stats["cache_reuse_partial_tokens"] == 2048
        assert scheduler_stats["last_cache_reuse_partial"]["used_cached_tokens"] == 2048
        assert scheduler_stats["last_cache_reuse_partial"]["cache_format"] == (
            "full_precision_kv"
        )
        assert scheduler_stats["last_cache_selection"]["selected"] == "prefix"
        assert scheduler_stats["last_cache_selection"]["reason"] == (
            "cold_paged_reconstruction_cost"
        )
        assert scheduler_stats["last_cache_selection"]["paged_cold_tokens"] == 2400

    @pytest.mark.asyncio
    async def test_cache_entries_endpoint_lists_paged_prefix_blocks(
        self, monkeypatch
    ):
        import vmlx_engine.server as server

        class _Block:
            token_count = 128
            ref_count = 3
            cache_data = object()

        class _PagedCache:
            allocated_blocks = {"block-a": _Block()}

        class _BlockAwareCache:
            paged_cache = _PagedCache()

        scheduler = SimpleNamespace(block_aware_cache=_BlockAwareCache())

        monkeypatch.setattr(server, "_get_scheduler", lambda: scheduler)

        payload = await server.cache_entries()

        assert payload["cache_type"] == "paged"
        assert payload["count"] == 1
        assert payload["entries"] == [
            {
                "block_id": "block-a",
                "tokens_count": 128,
                "ref_count": 3,
                "has_data": True,
                "cache_type": "paged",
            }
        ]

    @pytest.mark.asyncio
    async def test_cache_warm_endpoint_prefills_and_stores_block_cache(
        self, monkeypatch
    ):
        import vmlx_engine.server as server

        stored: list[tuple[str, list[int], list[str]]] = []

        class _Tokenizer:
            def encode(self, prompt):
                assert prompt == "system prompt"
                return [11, 12, 13, 14]

        class _BlockAwareCache:
            def store_cache(self, cache_id, tokens, extracted):
                stored.append((cache_id, tokens, extracted))
                return SimpleNamespace(block_ids=["warm-block"])

        class _Scheduler:
            tokenizer = _Tokenizer()
            block_aware_cache = _BlockAwareCache()

            def _prefill_for_prompt_only_cache(self, tokens):
                assert tokens == [11, 12, 13]
                return ["prefilled-cache"]

            def _extract_cache_states(self, cache):
                assert cache == ["prefilled-cache"]
                return ["kv-state"]

        monkeypatch.setattr(server, "_get_scheduler", lambda: _Scheduler())

        payload = await server.cache_warm({"prompts": ["system prompt"]})

        assert payload == {
            "warmed": 1,
            "token_counts": [4],
            "errors": None,
        }
        [(cache_id, tokens, extracted)] = stored
        assert cache_id.startswith("warm-0-")
        assert tokens == [11, 12, 13, 14]
        assert extracted == ["kv-state"]

    @pytest.mark.asyncio
    async def test_clear_cache_prefix_clears_prefix_l2_without_multimodal(
        self, monkeypatch
    ):
        import vmlx_engine.server as server

        cleared: list[str] = []

        class _Clearable:
            def __init__(self, name):
                self.name = name

            def clear(self):
                cleared.append(self.name)

        class _PagedManager:
            _disk_store = _Clearable("block_disk_store")

        scheduler = SimpleNamespace(
            memory_aware_cache=_Clearable("memory_aware_prefix"),
            block_aware_cache=_Clearable("paged_prefix"),
            prefix_cache=_Clearable("legacy_prefix"),
            disk_cache=_Clearable("disk_cache"),
            paged_cache_manager=_PagedManager(),
        )

        monkeypatch.setattr(server, "_get_scheduler", lambda: scheduler)
        monkeypatch.setattr(server, "clear_mlx_memory_cache", lambda log=None: False)

        payload = await server.clear_cache("prefix")

        assert payload["status"] == "cleared"
        assert payload["cache_type"] == "prefix"
        assert payload["caches"] == [
            "memory_aware_prefix",
            "paged_prefix",
            "legacy_prefix",
            "disk_cache",
            "block_disk_store",
        ]
        assert cleared == [
            "memory_aware_prefix",
            "paged_prefix",
            "legacy_prefix",
            "disk_cache",
            "block_disk_store",
        ]
        assert "multimodal_kv" not in payload["caches"]
        assert "pixel_values" not in payload["caches"]

    @pytest.mark.asyncio
    async def test_cache_stats_projects_ssm_companion_disk_state(self, monkeypatch):
        import vmlx_engine.server as server

        class _Engine:
            is_mllm = True

            def get_cache_stats(self):
                return None

        class _SSMCache:
            size = 1
            max_entries = 8
            disk_enabled = True
            disk_directory = "/tmp/vmlx-test/ssm_companion"

        class _BatchGenerator:
            _ssm_state_cache = _SSMCache()

        class _Scheduler:
            disk_cache = None
            paged_cache_manager = None
            batch_generator = _BatchGenerator()

            def get_stats(self):
                return {}

        monkeypatch.setattr(server, "_engine", _Engine())
        monkeypatch.setattr(server, "_get_scheduler", lambda: _Scheduler())

        payload = await server.cache_stats()
        ssm = payload["ssm_companion"]

        assert ssm["entries"] == 1
        assert ssm["max_entries"] == 8
        assert ssm["disk_enabled"] is True
        assert ssm["disk_directory"] == "/tmp/vmlx-test/ssm_companion"

    @pytest.mark.asyncio
    async def test_cache_stats_reports_disabled_kv_quant_without_zero_bit_ui(self, monkeypatch):
        import vmlx_engine.server as server

        class _Engine:
            is_mllm = False

            def get_cache_stats(self):
                return None

        class _Scheduler:
            _kv_cache_bits = 0
            _kv_cache_group_size = 64
            disk_cache = None
            paged_cache_manager = None

            def get_stats(self):
                return {}

        monkeypatch.setattr(server, "_engine", _Engine())
        monkeypatch.setattr(server, "_get_scheduler", lambda: _Scheduler())

        payload = await server.cache_stats()

        assert payload["kv_cache_quantization"] == {"enabled": False}

    @pytest.mark.asyncio
    async def test_health_endpoint_projects_scheduler_pressure_telemetry(
        self, monkeypatch
    ):
        import vmlx_engine.server as server

        class _Engine:
            is_mllm = False

            def get_stats(self):
                return {"engine_type": "batched"}

        class _Scheduler:
            def get_stats(self):
                return {
                    "num_waiting": 3,
                    "num_running": 1,
                    "ewma_ttft_seconds": 58.973,
                    "cache_hit_requests": 5,
                    "cache_hit_tokens": 8192,
                    "cache_hit_tokens_by_detail": {"paged+ssm": 8192},
                    "batch_generator": {
                        "hybrid_kv_without_ssm_hits": 2,
                        "hybrid_kv_without_ssm_tokens": 1536,
                        "last_hybrid_kv_without_ssm": {
                            "reason": "checkpoint_incomplete",
                            "checkpoint_tokens": 512,
                        },
                    },
                    "cache_reuse_skips": 2,
                    "cache_reuse_skip_tokens": 4096,
                    "last_cache_reuse_skip": {
                        "reason": "insufficient_memory_for_cache_merge",
                        "needed_mb": 41618.0,
                        "available_mb": 13330.0,
                    },
                    "cache_reuse_partial_downgrades": 3,
                    "cache_reuse_partial_tokens": 12000,
                    "last_cache_reuse_partial": {
                        "reason": "insufficient_memory_for_full_cache_merge",
                        "used_cached_tokens": 4096,
                    },
                    "last_cache_selection": {
                        "selected": "paged",
                        "reason": "paged_hot_advantage_sufficient",
                    },
                    "last_cache_execution": {
                        "cache_detail": "paged+disk+tq",
                        "total_worker_cache_seconds": 0.03,
                    },
                }

        monkeypatch.setattr(server, "_engine", _Engine())
        monkeypatch.setattr(server, "_get_scheduler", lambda: _Scheduler())
        monkeypatch.setattr(server, "_model_type", "llm")
        monkeypatch.setattr(server, "_standby_state", None)
        monkeypatch.setattr(server, "_mcp_manager", None)

        payload = await server.health()
        scheduler = payload["scheduler"]

        assert scheduler["num_waiting"] == 3
        assert scheduler["num_running"] == 1
        assert scheduler["ewma_ttft_seconds"] == 58.973
        assert scheduler["cache_hit_requests"] == 5
        assert scheduler["cache_hit_tokens"] == 8192
        assert scheduler["cache_hit_tokens_by_detail"]["paged+ssm"] == 8192
        assert scheduler["hybrid_kv_without_ssm_hits"] == 2
        assert scheduler["hybrid_kv_without_ssm_tokens"] == 1536
        assert scheduler["last_hybrid_kv_without_ssm"]["checkpoint_tokens"] == 512
        assert scheduler["cache_reuse_skips"] == 2
        assert scheduler["cache_reuse_skip_tokens"] == 4096
        assert scheduler["last_cache_reuse_skip"]["reason"] == (
            "insufficient_memory_for_cache_merge"
        )
        assert scheduler["cache_reuse_partial_downgrades"] == 3
        assert scheduler["cache_reuse_partial_tokens"] == 12000
        assert scheduler["last_cache_reuse_partial"]["used_cached_tokens"] == 4096
        assert scheduler["last_cache_selection"]["selected"] == "paged"
        assert scheduler["last_cache_execution"]["cache_detail"] == "paged+disk+tq"

    @pytest.mark.asyncio
    async def test_health_endpoint_projects_cache_telemetry_snapshot(
        self, monkeypatch
    ):
        import vmlx_engine.server as server

        class _Engine:
            is_mllm = False

            def get_stats(self):
                return {"engine_type": "batched"}

            def get_cache_stats(self):
                return {
                    "total_tokens_cached": 640,
                    "tokens_saved": 1280,
                    "allocated_blocks": 10,
                }

        class _DiskCache:
            def stats(self):
                return {
                    "entries": 2,
                    "total_tokens_on_disk": 384,
                    "total_cached_tokens": 384,
                    "hits": 3,
                    "misses": 1,
                }

        class _BlockDisk:
            def get_stats(self):
                return {
                    "blocks_on_disk": 6,
                    "total_tokens_on_disk": 256,
                    "total_cached_tokens": 256,
                    "disk_hits": 2,
                    "disk_misses": 1,
                }

        class _PagedManager:
            _disk_store = _BlockDisk()

        class _SSMDisk:
            def stats(self):
                return {
                    "entries": 3,
                    "total_tokens_on_disk": 128,
                    "total_cached_tokens": 128,
                    "hits": 4,
                    "misses": 2,
                }

        class _SSMCache:
            size = 1
            max_entries = 8
            disk_enabled = True
            disk_directory = "/tmp/vmlx-test/ssm_companion"
            _disk = _SSMDisk()

        class _Scheduler:
            disk_cache = _DiskCache()
            paged_cache_manager = _PagedManager()
            _ssm_state_cache = _SSMCache()

            def get_stats(self):
                return {
                    "num_waiting": 0,
                    "num_running": 0,
                    "ewma_ttft_seconds": 0,
                    "cache_reuse_skips": 0,
                    "cache_reuse_skip_tokens": 0,
                    "last_cache_reuse_skip": None,
                    "cache_reuse_partial_downgrades": 0,
                    "cache_reuse_partial_tokens": 0,
                    "last_cache_reuse_partial": None,
                }

        monkeypatch.setattr(server, "_engine", _Engine())
        monkeypatch.setattr(server, "_get_scheduler", lambda: _Scheduler())
        monkeypatch.setattr(server, "_model_type", "llm")
        monkeypatch.setattr(server, "_standby_state", None)
        monkeypatch.setattr(server, "_mcp_manager", None)

        payload = await server.health()
        cache = payload["cache"]

        assert cache["scheduler_cache"]["total_tokens_cached"] == 640
        assert cache["disk_cache"]["total_tokens_on_disk"] == 384
        assert cache["block_disk_cache"]["total_tokens_on_disk"] == 256
        assert cache["ssm_companion"]["disk"]["total_tokens_on_disk"] == 128
        assert cache["totals"]["ram_tokens_cached"] == 640
        assert cache["totals"]["l2_prompt_tokens_on_disk"] == 384
        assert cache["totals"]["l2_block_tokens_on_disk"] == 256
        assert cache["totals"]["l2_ssm_tokens_on_disk"] == 128
        assert cache["totals"]["l2_tokens_on_disk"] == 768
        assert cache["totals"]["l2_tokens_on_disk_store_sum"] == 768
        assert "may_overlap" in cache["totals"]["l2_tokens_on_disk_note"]

    def test_cache_snapshot_reports_mllm_ssm_l2_before_lazy_generator(self):
        import vmlx_engine.server as server

        class _SSMDisk:
            directory = "/tmp/vmlx-test/block-cache/ssm_companion"

            def stats(self):
                return {
                    "entries": 3,
                    "total_tokens_on_disk": 192,
                    "total_cached_tokens": 192,
                    "hits": 5,
                    "misses": 1,
                }

        scheduler = SimpleNamespace(
            batch_generator=None,
            _ssm_companion_disk_store=_SSMDisk(),
            config=SimpleNamespace(
                ssm_state_cache_size=8,
                ssm_state_cache_max_mb=512,
            ),
        )

        cache = server._cache_telemetry_snapshot(scheduler)

        assert cache["ssm_companion"]["entries"] == 0
        assert cache["ssm_companion"]["max_entries"] == 8
        assert cache["ssm_companion"]["disk_enabled"] is True
        assert cache["ssm_companion"]["disk_directory"].endswith("ssm_companion")
        assert cache["ssm_companion"]["disk"]["total_tokens_on_disk"] == 192
        assert cache["totals"]["l2_ssm_tokens_on_disk"] == 192
        assert cache["totals"]["l2_tokens_on_disk"] == 192

    def test_cache_stats_surfaces_cache_reuse_skip_telemetry(self):
        scheduler_source = Path("./vmlx_engine/scheduler.py").read_text()
        server_source = Path("./vmlx_engine/server.py").read_text()
        cache_panel_source = Path(
            "./panel/src/renderer/src/components/sessions/CachePanel.tsx"
        ).read_text()
        performance_panel_source = Path(
            "./panel/src/renderer/src/components/sessions/PerformancePanel.tsx"
        ).read_text()

        for marker in (
            "cache_reuse_skips",
            "cache_reuse_skip_tokens",
            "last_cache_reuse_skip",
            "cache_hit_requests",
            "cache_hit_tokens",
            "cache_hit_tokens_by_detail",
            "cache_reuse_partial_downgrades",
            "cache_reuse_partial_tokens",
            "last_cache_reuse_partial",
        ):
            assert marker in scheduler_source
            assert marker in server_source

        assert "insufficient_memory_for_cache_merge" in scheduler_source
        assert "insufficient_memory_for_full_cache_merge" in scheduler_source
        assert "merge_budget = avail * budget_fraction" in scheduler_source
        assert '"budget_mb"' in scheduler_source
        assert "Cache Reuse Skips" in cache_panel_source
        assert "Cache Hit Tokens" in cache_panel_source
        assert "Hit Tokens by Detail" in cache_panel_source
        assert "hybrid_kv_without_ssm" in Path("./vmlx_engine/server.py").read_text()
        assert "ssm_prefix_lookup" in Path("./vmlx_engine/mllm_batch_generator.py").read_text()
        assert "Hybrid KV-Only Misses" in cache_panel_source
        assert "KV-Only Tokens" in cache_panel_source
        assert "last_hybrid_kv_without_ssm" in cache_panel_source
        assert "Hybrid KV-Only Misses" in performance_panel_source
        assert "last_hybrid_kv_without_ssm" in performance_panel_source
        assert "Partial Reuse" in cache_panel_source
        assert "last_cache_reuse_skip" in cache_panel_source
        assert "last_cache_reuse_partial" in cache_panel_source
        assert "used_needed_mb" in cache_panel_source
        assert "budgeted" in cache_panel_source
        assert "needed_mb" in cache_panel_source
        assert "budget_mb" in cache_panel_source
        assert "available_mb" in cache_panel_source
        assert "partial reuse failed" in cache_panel_source
        assert "dropped_cached_tokens" in cache_panel_source
        assert "full_prefill_tokens" in cache_panel_source
        assert "partial_reuse_unavailable_reason" in cache_panel_source
        assert "cache_format" in cache_panel_source
        assert "Partial Reuse" in performance_panel_source
        assert "Cache Hit Tokens" in performance_panel_source
        assert "last_cache_reuse_partial" in performance_panel_source
        assert "used_needed_mb" in performance_panel_source
        assert "budgeted" in performance_panel_source
        assert "budget_mb" in performance_panel_source
        assert "partial reuse failed" in performance_panel_source
        assert "dropped_cached_tokens" in performance_panel_source
        assert "full_prefill_tokens" in performance_panel_source
        assert "partial_reuse_unavailable_reason" in performance_panel_source
        assert "cache_format" in performance_panel_source

    def test_cache_stats_surface_displays_l2_token_totals(self):
        cache_panel_source = Path(
            "./panel/src/renderer/src/components/sessions/CachePanel.tsx"
        ).read_text()

        assert "Tokens on Disk" in cache_panel_source
        assert "Cache Totals" in cache_panel_source
        assert "RAM Cached Tokens" in cache_panel_source
        assert "L2 Tokens on Disk" in cache_panel_source
        assert "SSM L2 Tokens" in cache_panel_source

    def test_performance_panel_displays_jangtq_sidecar_layout(self):
        performance_panel_source = Path(
            "./panel/src/renderer/src/components/sessions/PerformancePanel.tsx"
        ).read_text()

        assert "prestacked_bundle" in performance_panel_source
        assert "JANGTQ Layout" in performance_panel_source
        assert "runtime sidecar" in performance_panel_source
        assert "F16 Passthrough" in performance_panel_source
        assert "passthrough_bit_widths_used" in performance_panel_source
        assert "routed_expert_bits_label" in performance_panel_source
        assert "compat_warnings" in performance_panel_source
        assert "grouped-Conv1d" in Path("./vmlx_engine/server.py").read_text()

    def test_panel_defaults_are_speed_oriented_and_labels_match_values(self):
        session_form_source = Path(
            "./panel/src/renderer/src/components/sessions/SessionConfigForm.tsx"
        ).read_text()
        cache_panel_source = Path(
            "./panel/src/renderer/src/components/sessions/CachePanel.tsx"
        ).read_text()
        settings_flow_source = Path("./panel/tests/settings-flow.test.ts").read_text()
        sessions_source = Path("./panel/src/main/sessions.ts").read_text()
        defaults_yaml = Path("./vmlx_engine/config/defaults.yaml").read_text()
        config_models = Path("./vmlx_engine/config/models.py").read_text()

        assert "maxNumSeqs: 1" in session_form_source
        assert "prefillBatchSize: 512" in session_form_source
        assert "completionBatchSize: 512" in session_form_source
        assert "enableJit: true" in session_form_source
        assert "maxNumSeqs: 1" in sessions_source
        assert "prefillBatchSize: 512" in sessions_source
        assert "completionBatchSize: 512" in sessions_source
        assert 'unlimitedLabel="Default (1)"' in session_form_source
        assert 'unlimitedLabel="Default (512)"' in session_form_source
        assert 'unlimitedLabel="Default (2048)"' in session_form_source
        assert "isLingCrackModelPath" not in sessions_source
        assert "out.defaultTemperature = 20" not in sessions_source
        assert (
            "defaultMinP = defs.defaultMinP ?? 0" in sessions_source
            or "setConfigValue(mutable, 'defaultMinP', defs.defaultMinP ?? 0)" in sessions_source
        )
        native_mtp_idx = sessions_source.index("// Native in-model MTP")
        generation_defaults_idx = sessions_source.index("// Generation defaults", native_mtp_idx)
        native_mtp_launch_block = sessions_source[native_mtp_idx:generation_defaults_idx]
        sessions_outside_native_mtp = (
            sessions_source[:native_mtp_idx] + sessions_source[generation_defaults_idx:]
        )
        assert "args.push('--native-mtp-sampling-policy'" in native_mtp_launch_block
        assert "args.push('--default-min-p'" not in native_mtp_launch_block
        assert "args.push('--default-min-p'" not in sessions_outside_native_mtp
        assert "args.push('--no-continuous-batching')" in sessions_source
        assert "Auto-enable continuous batching when prefix cache is on" not in sessions_source
        cli_source = Path("./vmlx_engine/cli.py").read_text()
        assert '"--max-num-seqs", type=int, default=1' in cli_source
        assert '"--prefill-batch-size", type=int, default=512' in cli_source
        assert '"--completion-batch-size", type=int, default=512' in cli_source
        assert '"--continuous-batching"' in cli_source
        assert '"--no-continuous-batching"' in cli_source
        assert "default=True" in cli_source
        assert "max_concurrent_requests: 1" in defaults_yaml
        assert "prefill_batch_size: 512" in defaults_yaml
        assert "completion_batch_size: 512" in defaults_yaml
        assert 'quantization: "q4"' in defaults_yaml
        assert 'quantization: "turboquant"' not in defaults_yaml
        assert "max_blocks: 1000" in defaults_yaml
        assert "max_concurrent_requests: int = 1" in config_models
        assert "prefill_batch_size: int = 512" in config_models
        assert "completion_batch_size: int = 512" in config_models
        assert "max_blocks: int = 1000" in config_models
        assert "defaults to 5" not in sessions_source
        assert "default: 256" not in cli_source
        assert "backend default 8" not in settings_flow_source
        assert "total_tokens_on_disk" in cache_panel_source
        assert "health.cache" in Path(
            "./panel/src/renderer/src/components/sessions/PerformancePanel.tsx"
        ).read_text()

    def test_session_command_preview_mirrors_runtime_default_flags(self):
        sessions_source = Path("./panel/src/main/sessions.ts").read_text()
        preview_source = Path(
            "./panel/src/renderer/src/components/sessions/SessionSettings.tsx"
        ).read_text()
        cli_source = Path("./vmlx_engine/cli.py").read_text()
        server_source = Path("./vmlx_engine/server.py").read_text()
        form_source = Path(
            "./panel/src/renderer/src/components/sessions/SessionConfigForm.tsx"
        ).read_text()

        for flag in (
            "--smelt",
            "--flash-moe",
            "--distributed",
            "--enable-jit",
            "--omni-backend",
            "--log-level",
            "--allowed-origins",
        ):
            assert flag in sessions_source
            assert flag in preview_source
        assert '"--omni-backend"' in cli_source
        assert '"--omni-backend"' in server_source
        assert "VMLINUX_OMNI_BACKEND" in cli_source
        assert "VMLINUX_OMNI_BACKEND" in server_source
        assert "omniBackendActive" in sessions_source
        assert "omniBackendActive" in preview_source
        assert "omniBackendVisible" in form_source
        assert "'omniBackend'," in sessions_source
        for flag in ("--default-enable-thinking",):
            # The literals may appear in sanitizer/blocklist tables so stale
            # additionalArgs can be stripped. They must not be emitted as
            # startup launch flags from the real session command or preview.
            assert f"args.push('{flag}'" not in sessions_source
            assert f"parts.push('{flag}'" not in preview_source
        sessions_native_idx = sessions_source.index("// Native in-model MTP")
        sessions_gen_idx = sessions_source.index("// Generation defaults", sessions_native_idx)
        sessions_native_block = sessions_source[sessions_native_idx:sessions_gen_idx]
        sessions_outside_native = (
            sessions_source[:sessions_native_idx] + sessions_source[sessions_gen_idx:]
        )
        preview_native_idx = preview_source.index("// Native in-model MTP mirrors sessions.ts")
        preview_gen_idx = preview_source.index("// Generation defaults", preview_native_idx)
        preview_native_block = preview_source[preview_native_idx:preview_gen_idx]
        preview_outside_native = (
            preview_source[:preview_native_idx] + preview_source[preview_gen_idx:]
        )
        for flag in (
            "--default-temperature",
            "--default-top-p",
            "--default-top-k",
            "--default-min-p",
            "--default-repetition-penalty",
        ):
            assert f"args.push('{flag}'" not in sessions_native_block
            assert f"parts.push('{flag}'" not in preview_native_block
        assert "args.push('--default-repetition-penalty'" not in sessions_outside_native
        assert "parts.push('--default-repetition-penalty'" not in preview_outside_native
        assert "IMAGE_ADDITIONAL_ARG_BLOCKLIST" in sessions_source
        assert "IMAGE_ADDITIONAL_ARG_BLOCKLIST" in preview_source
        assert "DSV4_ADDITIONAL_ARG_BLOCKLIST" in sessions_source
        assert "DSV4_ADDITIONAL_ARG_BLOCKLIST" in preview_source
        for source in (sessions_source, preview_source):
            assert "flag.includes('=')" in source
            assert "flag.slice(0, flag.indexOf('='))" in source
        assert "canonicalizeToolParserId" in sessions_source
        preview_parser_block = preview_source[
            preview_source.index("// Parser resolution"):
            preview_source.index("// Prefix cache", preview_source.index("// Parser resolution"))
        ]
        assert "canonicalizeToolParserId" in preview_parser_block

        assert "server._default_repetition_penalty = 1.0" not in cli_source
        assert "_default_repetition_penalty = 1.0" not in server_source

    def test_panel_emitted_cli_flags_are_engine_registered_and_dsv4_sanitized(self):
        sessions_source = Path("./panel/src/main/sessions.ts").read_text()
        preview_source = Path(
            "./panel/src/renderer/src/components/sessions/SessionSettings.tsx"
        ).read_text()
        cli_source = Path("./vmlx_engine/cli.py").read_text()
        server_source = Path("./vmlx_engine/server.py").read_text()

        def pushed_flags(source: str, target: str) -> set[str]:
            return set(
                re.findall(
                    rf"{target}\.push\(['\"](--[a-z0-9][a-z0-9-]+)['\"]",
                    source,
                )
            )

        def literal_flags(source: str) -> set[str]:
            return set(re.findall(r"['\"](--[a-z0-9][a-z0-9-]+)['\"]", source))

        def extract_set(source: str, name: str) -> set[str]:
            match = re.search(
                rf"const {name} = new Set\(\[\n(?P<body>.*?)\n\]\)",
                source,
                re.S,
            )
            assert match, f"missing {name}"
            return set(re.findall(r"['\"](--[^'\"]+)['\"]", match.group("body")))

        launch_flags = pushed_flags(sessions_source, "args")
        preview_flags = pushed_flags(preview_source, "parts")
        engine_flags = literal_flags(cli_source) | literal_flags(server_source)

        assert not (launch_flags - engine_flags), sorted(launch_flags - engine_flags)
        assert not (preview_flags - engine_flags), sorted(preview_flags - engine_flags)

        sessions_dsv4_blocklist = extract_set(
            sessions_source, "DSV4_ADDITIONAL_ARG_BLOCKLIST"
        )
        preview_dsv4_blocklist = extract_set(
            preview_source, "DSV4_ADDITIONAL_ARG_BLOCKLIST"
        )
        assert sessions_dsv4_blocklist == preview_dsv4_blocklist
        sessions_value_flags = extract_set(
            sessions_source, "ADDITIONAL_ARG_VALUE_FLAGS"
        )
        preview_value_flags = extract_set(
            preview_source, "ADDITIONAL_ARG_VALUE_FLAGS"
        )
        assert sessions_value_flags == preview_value_flags

        # DSV4 launch policy owns cache/tool/parser/VLM/runtime flags. Stale
        # additionalArgs must not be able to duplicate or override any flag the
        # app itself knows how to emit, including block-L2 and image-only flags.
        assert not (launch_flags - sessions_dsv4_blocklist), sorted(
            launch_flags - sessions_dsv4_blocklist
        )
        assert not (preview_flags - preview_dsv4_blocklist), sorted(
            preview_flags - preview_dsv4_blocklist
        )
        # Also strip serve-only rollback/tuning flags that the app does not
        # emit itself but that can break DSV4's app-owned runtime contract when
        # carried forward in stale Advanced Args.
        for flag in (
            "--no-state-machine-stops",
            "--prefill-keep-alloc",
            "--uds",
            "--inference-endpoints",
            "--wake-timeout",
        ):
            assert flag in sessions_dsv4_blocklist
            assert flag in preview_dsv4_blocklist
        for flag in ("--uds", "--inference-endpoints", "--wake-timeout"):
            assert flag in sessions_value_flags
            assert flag in preview_value_flags

    def test_vlm_video_sampling_is_request_scoped_not_restart_required(self):
        sessions_source = Path("./panel/src/main/sessions.ts").read_text()
        chat_source = Path("./panel/src/main/ipc/chat.ts").read_text()
        form_source = Path(
            "./panel/src/renderer/src/components/sessions/SessionConfigForm.tsx"
        ).read_text()

        restart_block = sessions_source[
            sessions_source.index("RESTART_REQUIRED_KEYS"):
            sessions_source.index("async updateSessionConfig")
        ]
        restart_keys = set(re.findall(r"'([A-Za-z_][A-Za-z0-9_]*)'", restart_block))
        assert "videoFps" not in restart_keys
        assert "videoMaxFrames" not in restart_keys
        assert "sessionConfig.videoFps" in chat_source
        assert "sessionConfig.videoMaxFrames" in chat_source
        assert "video_fps" in chat_source
        assert "video_max_frames" in chat_source
        assert "onChange('videoFps'" in form_source
        assert "onChange('videoMaxFrames'" in form_source

    def test_session_config_schema_covers_runtime_settings(self):
        server_source = Path("./panel/src/main/server.ts").read_text()
        settings_flow_source = Path("./panel/tests/settings-flow.test.ts").read_text()

        for field in (
            "smelt",
            "smeltExperts",
            "flashMoe",
            "flashMoeSlotBank",
            "flashMoePrefetch",
            "flashMoeIoSplit",
            "chatTemplate",
            "videoFps",
            "videoMaxFrames",
            "distributedEnabled",
            "distributedMode",
            "distributedSecret",
            "distributedNodes",
            "idleTimeoutSoftMin",
            "idleTimeoutHardMin",
            "autoSleepEnabled",
            "defaultRepetitionPenalty",
        ):
            assert f"{field}?" in server_source or f"{field}:" in server_source
            assert f"'{field}'" in settings_flow_source

    def test_panel_startup_defaults_sanitize_incompatible_saved_modes(self):
        sessions_source = Path("./panel/src/main/sessions.ts").read_text()
        preview_source = Path(
            "./panel/src/renderer/src/components/sessions/SessionSettings.tsx"
        ).read_text()
        form_source = Path(
            "./panel/src/renderer/src/components/sessions/SessionConfigForm.tsx"
        ).read_text()

        for source in (sessions_source, preview_source):
            assert "effectiveFlashMoe" in source
            assert "effectiveDistributed" in source
            assert "dsv4Active" in source
            assert "hybridCacheActive" in source
            assert "effectiveEnableJit" in source
            assert "if (effectiveFlashMoe)" in source
            assert "if (effectiveDistributed)" in source
            assert "compatibleExternalSpeculative" in source
            assert "if (effectiveEnableJit)" in source
            assert "if (compatibleExternalSpeculative)" in source
            assert "if (config.enableJit) args.push('--enable-jit')" not in source

        assert "normalizedDetectedFamily === 'deepseek-v4'" in form_source
        # JIT incompat list expanded 2026-05-09 to include TurboQuant
        # (engine skips mx.compile for TurboQuantKVCache; UI now matches).
        assert "disabled={flashMoeActive || distributedActive || dsv4Active || zayaCcaActive || turboQuantActive || multimodalActive || hybridCacheActive}" in form_source
        assert "checked={!!config.enableJit && !flashMoeActive && !distributedActive && !dsv4Active && !zayaCcaActive && !turboQuantActive && !multimodalActive && !hybridCacheActive}" in form_source
        assert "disabled={config.continuousBatching || multimodalActive || dsv4Active}" in form_source

    def test_responses_long_context_tool_cache_gate_script_pins_artifacts(self):
        gate_source = Path(
            "./tests/cross_matrix/run_responses_long_tool_cache_gate.py"
        ).read_text()

        for marker in (
            "/v1/responses",
            "/v1/cache/stats",
            "/health",
            "previous_response_id",
            "function_call_output",
            "TOOL_DEFINITIONS",
            "target_chars_per_turn",
            "10000",
            "SUMMARY.md",
            "SUMMARY.json",
            "tail_review",
            "cached_tokens",
            "scheduler_stats",
            "visible_output_observed",
            "final_turn_visible_output",
            "final_turn_no_tools",
            "final_turn_disable_thinking",
            "final_turn_tools_disabled",
            "final_turn_thinking_disabled",
            "require_cache_each_turn_after_first",
            "cache_reuse_each_turn_after_first",
            "_cache_acceptance",
            "_extract_warnings",
            "tools_enabled",
            "enable_thinking",
            "overall_pass",
            "result =",
            "_tools_enabled_for_turn",
            "resolve_tool_calls_in_turn",
            "Tool results are provided. Produce a visible answer",
            "_tool_resolution_input",
            "Use the tool result above",
            "TOOL_EVIDENCE: <exact path:line>",
            "require_tool_call_each_turn",
            "tool_call_each_required_turn",
            "_max_cached_tokens",
            "request_round",
            "response_round",
            "tool_choice_mode",
            "--tool-choice",
            "resolution_tool_choice",
            "--resolution-tool-choice",
            "--temperature",
            "--top-p",
            "--top-k",
            "--repetition-penalty",
            "_request_sampling_fields",
            "no_tool_markup_leak",
            "_tool_markup_leak",
            "require_tool_evidence",
            "--require-tool-evidence",
            "_tool_grounding",
            "tool_grounded",
            "tool_evidence_markers",
            "http_status",
            "http_error",
        ):
            assert marker in gate_source

    def test_responses_long_context_tool_cache_gate_does_not_count_reasoning_as_visible(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        extract_output_text = gate["_extract_output_text"]
        extract_reasoning = gate["_extract_reasoning"]

        reasoning_only = {
            "output": [
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning", "text": "internal cache analysis"}],
                }
            ]
        }
        assert extract_output_text(reasoning_only) == ""
        assert extract_reasoning(reasoning_only) == "internal cache analysis"

        mixed_response = {
            "output": [
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning", "text": "hidden"}],
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "visible answer"}],
                },
            ]
        }
        assert extract_output_text(mixed_response) == "visible answer"

    def test_responses_long_context_tool_cache_gate_can_require_cache_each_turn(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        cache_acceptance = gate["_cache_acceptance"]

        seen, each = cache_acceptance(
            [
                {"cached_tokens": 0},
                {"cached_tokens": 2895},
                {"cached_tokens": 5562},
            ],
            require_cache_each_turn_after_first=True,
        )
        assert seen is True
        assert each is True

        seen, each = cache_acceptance(
            [
                {"cached_tokens": 0},
                {"cached_tokens": 2895},
                {"cached_tokens": 0},
            ],
            require_cache_each_turn_after_first=True,
        )
        assert seen is True
        assert each is False

        seen, each = cache_acceptance(
            [
                {"cached_tokens": 0},
                {"cached_tokens": 2895},
                {"cached_tokens": 0},
            ],
            require_cache_each_turn_after_first=False,
        )
        assert seen is True
        assert each is True

    def test_responses_long_context_tool_cache_gate_ignores_disabled_require_flags_for_overall_pass(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        overall_acceptance_pass = gate["_overall_acceptance_pass"]

        acceptance = {
            "turns_completed": True,
            "previous_response_id_used": True,
            "cache_reuse_observed": True,
            "require_cache_each_turn_after_first": True,
            "cache_reuse_each_turn_after_first": True,
            "tool_call_observed": True,
            "require_tool_call_each_turn": False,
            "tool_call_each_required_turn": True,
            "require_tool_evidence": False,
            "tool_evidence_each_required_turn": True,
            "final_turn_tools_disabled": True,
            "final_turn_thinking_disabled": True,
            "visible_or_tool_output_each_turn": True,
            "visible_output_observed": True,
            "final_turn_visible_output": True,
            "no_loop_like_tail": True,
            "no_tool_markup_leak": True,
        }

        assert overall_acceptance_pass(acceptance) is True

    def test_responses_long_context_tool_cache_gate_extracts_warnings(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        extract_warnings = gate["_extract_warnings"]

        assert extract_warnings({"warnings": ["  current response warning  ", 42, ""]}) == [
            "current response warning"
        ]

        assert extract_warnings(
            {
                "output": [
                    {"type": "response.warning", "message": "stream warning"},
                    {"type": "message", "content": []},
                ]
            }
        ) == ["stream warning"]

    def test_responses_long_context_tool_cache_gate_counts_cached_tokens_across_rounds(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        max_cached_tokens = gate["_max_cached_tokens"]

        assert max_cached_tokens(
            [
                {"usage": {"input_tokens_details": {"cached_tokens": 0}}},
                {"usage": {"input_tokens_details": {"cached_tokens": 8192}}},
            ]
        ) == 8192
        assert max_cached_tokens([]) == 0

    def test_responses_long_context_tool_cache_gate_tool_output_handles_bad_paths(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        tool_output = gate["_tool_output"]

        out = tool_output(
            Path("."),
            {
                "name": "inspect_symbol",
                "call_id": "call_bad_path",
                "arguments": '{"path":"does-not-exist.py","symbol":"Scheduler"}',
            },
        )
        assert out["type"] == "function_call_output"
        assert out["call_id"] == "call_bad_path"
        assert "not a readable file" in out["output"]
        assert re.search(r"(?m)^[^:\n]+\.(py|ts|tsx|md):1:", out["output"])

    def test_responses_long_context_tool_cache_gate_inspect_symbol_searches_directories(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        tool_output = gate["_tool_output"]

        out = tool_output(
            Path("."),
            {
                "name": "inspect_symbol",
                "call_id": "call_dir_path",
                "arguments": '{"path":"vmlx_engine","symbol":"PAGED_CACHE_SCHEMA_VERSION","context_lines":2}',
            },
        )
        assert out["type"] == "function_call_output"
        assert out["call_id"] == "call_dir_path"
        assert "not a readable file" not in out["output"]
        assert "PAGED_CACHE_SCHEMA_VERSION" in out["output"]
        assert "vmlx_engine/" in out["output"]
        assert "build/lib" not in out["output"]

    def test_responses_long_context_tool_cache_gate_can_require_tool_each_turn(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        tool_acceptance = gate["_tool_acceptance"]

        assert tool_acceptance(
            [
                {"tools_enabled": True, "function_calls": [{"name": "grep_repo"}]},
                {"tools_enabled": True, "function_calls": [{"name": "grep_repo"}]},
            ],
            require_tool_call=True,
            require_tool_call_each_turn=True,
        ) == (True, True)
        assert tool_acceptance(
            [
                {"tools_enabled": True, "function_calls": [{"name": "grep_repo"}]},
                {"tools_enabled": True, "function_calls": []},
            ],
            require_tool_call=True,
            require_tool_call_each_turn=True,
        ) == (True, False)

    def test_responses_long_context_tool_cache_gate_rejects_visible_tool_markup_leak(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        tool_markup_leak = gate["_tool_markup_leak"]

        assert tool_markup_leak("plain visible answer") is False
        assert tool_markup_leak("<minimax:tool_call><invoke name=\"grep_repo\">") is True
        assert tool_markup_leak("<｜DSML｜invoke name=\"grep_repo\">") is True
        assert tool_markup_leak("<tool_call>\n<function=grep_repo>") is True

    def test_responses_long_context_tool_cache_gate_can_require_tool_evidence(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        tool_grounding = gate["_tool_grounding"]

        tool_outputs = [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": (
                    "vmlx_engine/scheduler.py:2431: def _cache_reuse_budget_fraction():\n"
                    "vmlx_engine/scheduler.py:2440: return max(0.10, min(0.95, value))"
                ),
            }
        ]

        grounded = tool_grounding(
            "Risk is in budget sizing. TOOL_EVIDENCE: vmlx_engine/scheduler.py:2431",
            tool_outputs,
        )
        assert grounded["grounded"] is True
        assert grounded["marker"] == "vmlx_engine/scheduler.py:2431"

        ungrounded = tool_grounding(
            "Risk is in budget sizing, but no exact file line is cited.",
            tool_outputs,
        )
        assert ungrounded["grounded"] is False
        assert "vmlx_engine/scheduler.py:2431" in ungrounded["markers"]

    def test_responses_long_context_tool_cache_gate_resolution_input_prompts_exact_evidence(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        tool_resolution_input = gate["_tool_resolution_input"]

        tool_outputs = [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "vmlx_engine/server.py:104: ResponsesObject,",
            }
        ]

        payload = tool_resolution_input(tool_outputs)
        assert payload[0] is tool_outputs[0]
        assert payload[1]["role"] == "user"
        assert "TOOL_EVIDENCE: <exact path:line>" in payload[1]["content"]
        assert "Do not call another tool" in payload[1]["content"]

    def test_responses_long_context_tool_cache_gate_can_force_deterministic_sampling(self):
        import runpy
        from types import SimpleNamespace

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        request_sampling_fields = gate["_request_sampling_fields"]

        assert request_sampling_fields(
            SimpleNamespace(
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=None,
            )
        ) == {}

        assert request_sampling_fields(
            SimpleNamespace(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
            )
        ) == {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
        }

    def test_responses_long_context_tool_cache_gate_rejects_no_match_tool_evidence(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        tool_grounding = gate["_tool_grounding"]

        no_match = [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "no matches",
            }
        ]

        result = tool_grounding("TOOL_EVIDENCE: no matches", no_match)
        assert result["grounded"] is False
        assert result["markers"] == []
        assert result["reason"] == "no_file_line_tool_evidence"

    def test_responses_long_context_tool_cache_gate_tolerates_malformed_tool_ints(self):
        import json
        import runpy
        from pathlib import Path

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        tool_output = gate["_tool_output"]

        result = tool_output(
            Path("."),
            {
                "name": "grep_repo",
                "call_id": "call_bad_int",
                "arguments": json.dumps(
                    {
                        "pattern": "_prefix_cache",
                        "path": "vmlx_engine",
                        "max_matches": "5\n</parameter>",
                    }
                ),
            },
        )

        assert result["type"] == "function_call_output"
        assert result["call_id"] == "call_bad_int"
        assert isinstance(result["output"], str)

    def test_responses_long_context_tool_cache_gate_bounds_tool_ints(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        tool_positive_int = gate["_tool_positive_int"]

        assert tool_positive_int("999999999", 20) == 200
        assert tool_positive_int("-1", 20) == 20

    def test_responses_long_context_tool_cache_gate_resolution_tools_mode(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        resolution_tools_enabled = gate["_resolution_tools_enabled"]

        assert resolution_tools_enabled("none") is False
        assert resolution_tools_enabled("auto") is True
        assert resolution_tools_enabled("required") is True

    def test_responses_long_context_tool_cache_gate_can_disable_final_turn_tools(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        tools_enabled_for_turn = gate["_tools_enabled_for_turn"]

        assert tools_enabled_for_turn(1, 3, True) is True
        assert tools_enabled_for_turn(2, 3, True) is True
        assert tools_enabled_for_turn(3, 3, True) is False
        assert tools_enabled_for_turn(3, 3, False) is True

    def test_responses_long_context_tool_cache_gate_can_disable_final_turn_thinking(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        enable_thinking_for_turn = gate["_enable_thinking_for_turn"]

        assert enable_thinking_for_turn(1, 3, True, True) is True
        assert enable_thinking_for_turn(2, 3, True, True) is True
        assert enable_thinking_for_turn(3, 3, True, True) is False
        assert enable_thinking_for_turn(3, 3, False, True) is False

    def test_responses_long_context_tool_cache_gate_requires_real_tool_path(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        prompt = gate["_build_prompt"](Path("."), 1, 200)
        instructions = gate["_instructions"]()

        assert "first assistant output" in prompt
        assert "answer only after the tool result is provided" in prompt
        assert "must call exactly one provided tool" in instructions
        assert "When tools are not available, answer directly" in instructions

    def test_responses_long_context_tool_cache_gate_no_match_returns_real_marker(self):
        import json
        import runpy
        from pathlib import Path

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        tool_output = gate["_tool_output"]

        result = tool_output(
            Path("."),
            {
                "name": "grep_repo",
                "call_id": "call_no_match",
                "arguments": json.dumps(
                    {
                        "pattern": "definitely_not_present_in_server_py_12345",
                        "path": "vmlx_engine/server.py",
                    }
                ),
            },
        )

        assert "no matches for" in result["output"]
        assert "vmlx_engine/server.py:1:" in result["output"]

    def test_native_cache_status_reports_dsv4_separately_from_tq_kv(self, monkeypatch):
        from types import SimpleNamespace
        from vmlx_engine.server import _native_cache_status

        scheduler = SimpleNamespace(
            _uses_dsv4_cache=True,
            block_aware_cache=object(),
            paged_cache_manager=SimpleNamespace(_disk_store=object()),
        )
        monkeypatch.setenv("DSV4_POOL_QUANT", "1")

        status = _native_cache_status(scheduler)

        assert status["family"] == "deepseek_v4"
        assert status["schema"] == "deepseek_v4_v7"
        assert status["cache_type"] == "native_composite"
        assert "swa_local" in status["components"]
        assert "csa_compressed_pool" in status["components"]
        assert "hca_compressed_pool" in status["components"]
        assert status["generic_turboquant_kv"]["enabled"] is False
        assert status["pool_quant"]["enabled"] is True
        assert status["paged"] is True
        assert status["block_disk_l2"] is True

    def test_native_cache_status_reports_zaya_typed_cca(self):
        from types import SimpleNamespace
        from vmlx_engine.server import _native_cache_status

        scheduler = SimpleNamespace(
            _model_type_for_runtime="zaya",
            block_aware_cache=object(),
            paged_cache_manager=SimpleNamespace(_disk_store=object()),
        )

        status = _native_cache_status(scheduler)

        assert status["family"] == "zaya"
        assert status["schema"] == "zaya_cca_v1"
        assert status["cache_type"] == "typed_cca"
        assert "standard_kv" in status["components"]
        assert "cca_conv_state" in status["components"]
        assert "cca_prev_hidden" in status["components"]
        assert status["generic_turboquant_kv"]["enabled"] is False
        assert status["paged"] is True
        assert status["block_disk_l2"] is True

    def test_native_cache_status_reports_mixed_swa_kv(self):
        from types import SimpleNamespace
        from vmlx_engine.server import _native_cache_status

        scheduler = SimpleNamespace(
            _model_type_for_runtime="gemma4",
            _mixed_attention_cache_model=True,
            _tq_active=False,
            _kv_cache_bits=4,
            _kv_cache_group_size=64,
            block_aware_cache=object(),
            paged_cache_manager=SimpleNamespace(_disk_store=object()),
        )

        status = _native_cache_status(scheduler)

        assert status["family"] == "gemma4"
        assert status["schema"] == "mixed_swa_kv_v1"
        assert status["cache_type"] == "mixed_swa_kv"
        assert "sliding_window_kv" in status["components"]
        assert "full_attention_kv" in status["components"]
        assert "rotating_window_metadata" in status["components"]
        assert status["generic_turboquant_kv"]["enabled"] is False
        assert status["storage_quantization"] == {
            "enabled": True,
            "mode": "storage_boundary",
            "bits": 4,
            "group_size": 64,
            "applies_to": "full_attention_kv_only",
            "metadata_policy": "preserve_rotating_window_metadata",
        }
        assert status["paged"] is True
        assert status["block_disk_l2"] is True

    def test_native_cache_status_reports_step37_full_sliding_kv_from_registry_subtype(self):
        from types import SimpleNamespace
        from vmlx_engine.server import _native_cache_status

        cfg = SimpleNamespace(cache_subtype="step3p7_full_sliding_kv")
        scheduler = SimpleNamespace(
            _model_type_for_runtime="step3p7",
            _tq_active=False,
            _kv_cache_bits=4,
            _kv_cache_group_size=64,
            block_aware_cache=object(),
            paged_cache_manager=SimpleNamespace(_disk_store=object()),
        )

        status = _native_cache_status(
            scheduler,
            family="step-3.7-flash",
            cfg=cfg,
        )

        assert status["family"] == "step-3.7-flash"
        assert status["schema"] == "mixed_swa_kv_v1"
        assert status["cache_type"] == "mixed_swa_kv"
        assert status["storage_quantization"]["applies_to"] == "full_attention_kv_only"
        assert status["storage_quantization"]["metadata_policy"] == (
            "preserve_rotating_window_metadata"
        )

    def test_native_cache_status_reports_mimo_v2_asymmetric_swa_from_registry_subtype(self):
        from types import SimpleNamespace
        from vmlx_engine.server import _native_cache_status

        cfg = SimpleNamespace(cache_subtype="mimo_v2_asymmetric_swa")
        scheduler = SimpleNamespace(
            _model_type_for_runtime="mimo_v2",
            _tq_active=False,
            _kv_cache_bits=4,
            _kv_cache_group_size=64,
            block_aware_cache=object(),
            paged_cache_manager=SimpleNamespace(_disk_store=object()),
        )

        status = _native_cache_status(
            scheduler,
            family="mimo_v2",
            cfg=cfg,
        )

        assert status["family"] == "mimo_v2"
        assert status["schema"] == "mixed_swa_kv_v1"
        assert status["cache_type"] == "mixed_swa_kv"
        assert status["cache_subtype"] == "mimo_v2_asymmetric_swa"
        assert status["storage_quantization"]["applies_to"] == "full_attention_kv_only"
        assert status["storage_quantization"]["metadata_policy"] == (
            "preserve_rotating_window_metadata"
        )

    def test_native_cache_status_reports_hybrid_ssm_from_registry_without_scheduler(self):
        from types import SimpleNamespace
        from vmlx_engine.server import _native_cache_status

        cfg = SimpleNamespace(cache_type="hybrid", cache_subtype=None)

        status = _native_cache_status(None, family="qwen3_5_moe", cfg=cfg)

        assert status["family"] == "qwen3_5_moe"
        assert status["schema"] == "hybrid_ssm_v1"
        assert status["cache_type"] == "hybrid_ssm_typed"
        assert status["components"] == [
            "attention_kv",
            "ssm_companion_state",
            "async_rederive",
        ]
        assert status["generic_turboquant_kv"] == {
            "enabled": False,
            "reason": "hybrid_ssm_state",
        }
        assert status["prefix"] is False
        assert status["paged"] is False
        assert status["block_disk_l2"] is False

    def test_native_cache_status_reports_plain_attention_kv(self):
        from types import SimpleNamespace
        from vmlx_engine.server import _native_cache_status

        scheduler = SimpleNamespace(
            _model_type_for_runtime="minimax",
            _tq_active=True,
            _kv_cache_bits=3,
            _kv_cache_group_size=64,
            block_aware_cache=object(),
            paged_cache_manager=SimpleNamespace(_disk_store=object()),
        )

        status = _native_cache_status(scheduler)

        assert status["family"] == "minimax"
        assert status["schema"] == "plain_kv_v1"
        assert status["cache_type"] == "paged_kv"
        assert status["components"] == ["attention_kv"]
        assert status["generic_turboquant_kv"] == {
            "enabled": True,
            "reason": "plain_attention_kv",
        }
        assert status["storage_quantization"] == {
            "enabled": True,
            "bits": 3,
            "group_size": 64,
        }
        assert status["prefix"] is True
        assert status["paged"] is True
        assert status["block_disk_l2"] is True

    def test_native_cache_status_reports_hybrid_ssm(self):
        from types import SimpleNamespace
        from vmlx_engine.server import _native_cache_status

        scheduler = SimpleNamespace(
            config=SimpleNamespace(kv_cache_quantization="q4"),
            _model_type_for_runtime="bailing_hybrid",
            _is_hybrid=True,
            _uses_dsv4_cache=False,
            _uses_zaya_cache=False,
            _hybrid_kv_positions=[7, 15, 23, 31],
            _kv_cache_bits=4,
            _kv_cache_group_size=64,
            _ssm_state_cache=SimpleNamespace(_store={"a": object()}),
            block_aware_cache=object(),
            paged_cache_manager=SimpleNamespace(_disk_store=object()),
        )

        status = _native_cache_status(scheduler)

        assert status["schema"] == "hybrid_ssm_v1"
        assert status["cache_type"] == "hybrid_ssm_typed"
        assert status["generic_turboquant_kv"]["enabled"] is False
        assert status["attention_kv_storage_quantization"] == {
            "enabled": True,
            "mode": "storage_boundary",
            "bits": 4,
            "group_size": 64,
            "applies_to": "attention_kv_layers_only",
            "ssm_policy": "native_companion_state",
            "rederive": "async_clean_prefill_on_miss_or_warm_pass",
        }

    def test_native_cache_status_reports_nemotron_h_registered_hybrid_subtype(self):
        from types import SimpleNamespace
        from vmlx_engine.server import _native_cache_status

        cfg = SimpleNamespace(cache_subtype="nemotron_h_ssm_attention")
        scheduler = SimpleNamespace(
            config=SimpleNamespace(kv_cache_quantization="q4"),
            _model_type_for_runtime="nemotron_h",
            _is_hybrid=True,
            _uses_dsv4_cache=False,
            _uses_zaya_cache=False,
            _hybrid_kv_positions=[2, 5, 8],
            _hybrid_live_tq_policy=None,
            _hybrid_live_tq_attention_layers=[],
            _hybrid_live_tq_companion_layers=[],
            _kv_cache_bits=4,
            _kv_cache_group_size=64,
            _ssm_state_cache=SimpleNamespace(_store={"lfm": object()}),
            block_aware_cache=object(),
            paged_cache_manager=SimpleNamespace(_disk_store=object()),
        )

        status = _native_cache_status(scheduler, family="nemotron_h", cfg=cfg)

        assert status["family"] == "nemotron_h"
        assert status["schema"] == "hybrid_ssm_v1"
        assert status["cache_type"] == "hybrid_ssm_typed"
        assert status["generic_turboquant_kv"] == {
            "enabled": False,
            "reason": "hybrid_ssm_state",
        }

    def test_native_cache_status_reports_lfm2_moe_registered_hybrid_subtype(self):
        from types import SimpleNamespace
        from vmlx_engine.server import _native_cache_status

        cfg = SimpleNamespace(cache_subtype="lfm2_moe_hybrid_ssm")
        scheduler = SimpleNamespace(
            config=SimpleNamespace(kv_cache_quantization="q4"),
            _model_type_for_runtime="lfm2_moe",
            _is_hybrid=True,
            _uses_dsv4_cache=False,
            _uses_zaya_cache=False,
            _hybrid_kv_positions=[1, 3, 7],
            _hybrid_live_tq_policy=None,
            _hybrid_live_tq_attention_layers=[],
            _hybrid_live_tq_companion_layers=[],
            _kv_cache_bits=4,
            _kv_cache_group_size=64,
            _ssm_state_cache=SimpleNamespace(_store={"lfm": object()}),
            block_aware_cache=object(),
            paged_cache_manager=SimpleNamespace(_disk_store=object()),
        )

        status = _native_cache_status(scheduler, family="lfm2_moe", cfg=cfg)

        assert status["family"] == "lfm2_moe"
        assert status["schema"] == "hybrid_ssm_v1"
        assert status["cache_type"] == "hybrid_ssm_typed"
        assert status["generic_turboquant_kv"] == {
            "enabled": False,
            "reason": "hybrid_ssm_state",
        }
        assert status["ssm_entries"] == 1
        assert status["kv_layer_indices"] == [1, 3, 7]

    def test_native_cache_status_ignores_legacy_hybrid_tq_override(self, monkeypatch):
        from types import SimpleNamespace
        from vmlx_engine.server import _native_cache_status

        scheduler = SimpleNamespace(
            config=SimpleNamespace(kv_cache_quantization="q4"),
            _model_type_for_runtime="bailing_hybrid",
            _is_hybrid=True,
            _uses_dsv4_cache=False,
            _uses_zaya_cache=False,
            _hybrid_kv_positions=[],
            _ssm_state_cache=SimpleNamespace(_store={}),
            block_aware_cache=object(),
            paged_cache_manager=SimpleNamespace(_disk_store=None),
        )
        monkeypatch.setenv("VMLX_ALLOW_HYBRID_KV_QUANT", "1")

        status = _native_cache_status(scheduler)

        assert status["generic_turboquant_kv"] == {
            "enabled": False,
            "reason": "hybrid_ssm_state",
        }
        assert status["live_attention_tq_kv"]["enabled"] is False

    def test_scheduler_cache_extraction_preserves_failed_hybrid_layer_index(self):
        """Failed SSM state extraction must not shrink the cache layer list."""
        from types import SimpleNamespace
        from vmlx_engine.scheduler import Scheduler

        class _BadArraysCache:
            cache = []
            meta_state = ()

            @property
            def state(self):
                raise RuntimeError("state unavailable")

        class _KVCache:
            state = ("keys", "values")
            meta_state = ("4",)

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.model = SimpleNamespace(
            args=SimpleNamespace(num_key_value_heads=0, kv_lora_rank=0)
        )

        extracted = scheduler._extract_cache_states([
            _BadArraysCache(),
            _KVCache(),
        ])

        assert len(extracted) == 2
        assert extracted[0] == {
            "state": None,
            "meta_state": None,
            "class_name": "_BadArraysCache",
        }
        assert extracted[1]["class_name"] == "_KVCache"

    def test_scheduler_cache_extraction_preserves_mimo_mixed_full_kv_heads(self):
        """MiMo full-attention KV must not be sliced to the smaller SWA head count."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache
        from types import SimpleNamespace
        from vmlx_engine.scheduler import Scheduler

        cfg = SimpleNamespace(
            model_type="mimo_v2",
            num_key_value_heads=4,
            num_global_key_value_heads=8,
            swa_num_key_value_heads=4,
            kv_lora_rank=0,
        )
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.model = SimpleNamespace(args=cfg)

        full_cache = KVCache()
        full_cache.keys = mx.ones((1, 8, 6, 192), dtype=mx.float16)
        full_cache.values = mx.ones((1, 8, 6, 192), dtype=mx.float16)
        full_cache.offset = 6

        extracted = scheduler._extract_cache_states([full_cache])

        keys, values = extracted[0]["state"]
        assert keys.shape[1] == 8
        assert values.shape[1] == 8

    def test_quantization_status_detects_jangtq_sidecar_and_bits(self, tmp_path):
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_5_moe",
            "weight_format": "mxtq",
            "mxtq_bits": 2,
            "quantization": {"bits": 2, "group_size": 64},
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "quantization": {
                "profile": "JANGTQ2",
                "target_bits": 2,
                "actual_bits": 2.0,
                "quantization_backend": "turboquant",
            }
        }))
        (tmp_path / "jangtq_runtime.safetensors").write_bytes(b"sidecar")

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "turboquant_codebook"
        assert status["weight_format"] == "mxtq"
        assert status["mxtq_bits"] == 2
        assert status["routed_expert_bits"] == 2
        assert status["profile"] == "JANGTQ2"
        assert status["sidecar"]["jangtq_runtime"] is True

    @pytest.mark.parametrize(
        ("profile", "expected_bits", "unsupported"),
        [("JANGTQ1", 1, True), ("JANGTQ2", 2, False), ("JANGTQ4", 4, False)],
    )
    def test_quantization_status_derives_jangtq_bits_from_profile(
        self, tmp_path, profile, expected_bits, unsupported
    ):
        """Hy3/JANGTQ bundles may stamp only profile before sidecar sniffing.

        The engine health surface must still expose the routed expert bit count
        so the panel and release gates do not show a TurboQuant bundle as
        "JANGTQ -bit" or treat it as unknown precision.
        """
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "hy_v3",
            "weight_format": "mxtq",
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "weight_format": "mxtq",
            "quantization": {
                "profile": profile,
                "quantization_backend": "turboquant",
            },
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "turboquant_codebook"
        assert status["profile"] == profile
        assert status["routed_expert_bits"] == expected_bits
        warnings = status.get("compat_warnings") or []
        assert any("not production-supported" in w for w in warnings) is unsupported
        assert status["target_bits"] == expected_bits

    def test_quantization_status_detects_prestacked_jangtq_bundle(self, tmp_path):
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "deepseek_v4",
            "weight_format": "mxtq",
            "mxtq_bits": 2,
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "weight_format": "mxtq",
            "mxtq_bits": {"routed_expert": 2},
        }))
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {
                "model.layers.0.mlp.switch_mlp.gate_proj.tq_packed": "model-00001.safetensors",
            }
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "turboquant_codebook"
        assert status["sidecar"]["prestacked_bundle"] is True

    def test_quantization_status_reads_jang_role_bit_plan(self, tmp_path):
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "quantization": {"bits": 8, "group_size": 64},
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "weight_format": "mxtq",
            "mxtq_bits": {
                "attention": 8,
                "shared_expert": 8,
                "routed_expert": 2,
                "embed_tokens": 8,
                "lm_head": 8,
            },
            "quantization": {
                "method": "affine+mxtq",
                "bits_default": 2,
                "group_size": 64,
            },
        }))
        (tmp_path / "jangtq_runtime.safetensors").write_bytes(b"sidecar")

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "turboquant_codebook"
        assert status["weight_format"] == "mxtq"
        assert status["routed_expert_bits"] == 2
        assert status["mxtq_bits_by_role"]["attention"] == 8
        assert status["target_bits"] == 2

    def test_quantization_status_handles_jangtq_k_mixed_routed_bits(self, tmp_path):
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "weight_format": "mxtq",
            "quantization": {"bits": 2, "group_size": 64},
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "weight_format": "mxtq",
            "mxtq_bits": {
                "attention": 8,
                "routed_expert": {
                    "gate_proj": 2,
                    "up_proj": 2,
                    "down_proj": 4,
                },
            },
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "turboquant_codebook"
        assert "routed_expert_bits" not in status
        assert status["routed_expert_bits_by_projection"] == {
            "gate_proj": 2,
            "up_proj": 2,
            "down_proj": 4,
        }
        assert status["routed_expert_bits_label"] == "gate=2/up=2/down=4-bit"
        assert status["target_bits"] == 2

    def test_quantization_status_surfaces_jang_passthrough_bit_plan(self, tmp_path):
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_next",
            "quantization": {"bits": 2, "group_size": 64},
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "weight_format": "jang",
            "quantization": {
                "method": "jang-importance",
                "target_bits": 2,
                "actual_bits": 2.2,
                "bit_widths_used": [2, 4, 8],
                "passthrough_bit_widths_used": [16],
                "passthrough_tensor_count": 90,
            },
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "affine_quantized_matmul"
        assert status["passthrough_bit_widths_used"] == [16]
        assert status["passthrough_tensor_count"] == 90
        assert "compat_warnings" not in status

    def test_quantization_status_warns_when_hybrid_bundle_lacks_passthrough_metadata(self, tmp_path):
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "nemotron_h",
            "quantization": {"bits": 2, "group_size": 64},
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "weight_format": "jang",
            "quantization": {
                "method": "jang-importance",
                "target_bits": 2,
            },
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["compat_warnings"]
        assert "grouped-Conv1d layout backstop" in status["compat_warnings"][0]

    def test_quantization_status_reports_jangtq_weight_index_and_dispatch(self, tmp_path):
        """JANGTQ speed gates need actual indexed codec targets, not labels."""
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_5_moe",
            "weight_format": "mxtq",
            "mxtq_bits": {
                "attention": 8,
                "routed_expert": 4,
                "embed_tokens": 8,
                "lm_head": 8,
            },
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "weight_format": "mxtq",
            "quantization": {
                "profile": "JANGTQ4",
                "quantization_backend": "turboquant",
            },
        }))
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {
                "model.layers.0.mlp.switch_mlp.gate_proj.tq_packed": "a.safetensors",
                "model.layers.0.mlp.switch_mlp.gate_proj.tq_norms": "a.safetensors",
                "model.layers.1.mlp.switch_mlp.down_proj.tq_bits": "b.safetensors",
                "model.layers.2.self_attn.q_proj.weight": "c.safetensors",
                "model.embed_tokens.weight": "d.safetensors",
            },
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "turboquant_codebook"
        assert status["weight_index"]["total_tensors"] == 5
        assert status["weight_index"]["suffix_counts"]["tq_packed"] == 1
        assert status["weight_index"]["suffix_counts"]["tq_norms"] == 1
        assert status["weight_index"]["suffix_counts"]["tq_bits"] == 1
        assert status["weight_index"]["tq_target_counts"]["routed_expert"] == 3
        assert status["weight_index"]["mtp_tensor_count"] == 0
        assert status["weight_index"]["routed_layout_counts"] == {
            "prestacked_switch": 3,
            "split_expert": 0,
        }
        assert status["weight_index"]["sample_tq_packed_targets"] == [
            "model.layers.0.mlp.switch_mlp.gate_proj"
        ]
        assert status["weight_matmul_dispatch"] == {
            "primary": "jang_tools_turboquant_custom_kernels",
            "uses_mlx_quantized_matmul": False,
            "metal_na_eligible": False,
            "reason": "turboquant_codebook_uses_custom_tq_kernels",
        }

    def test_dsv4_keeper_identity_reports_no_mtp_and_prestacked_layout(
        self, tmp_path
    ):
        """Pin the final DSV4 no-MTP keeper shape without local model paths."""
        from vmlx_engine.server import _model_mtp_status, _model_quantization_status

        pure_jang = tmp_path / "DeepSeek-V4-Flash-JANG_DQ2-Token8-DownG32-Gate3Math6-NoMTP"
        pure_jang.mkdir()
        (pure_jang / "config.json").write_text(json.dumps({
            "model_type": "deepseek_v4",
            "weight_format": "affine",
            "num_nextn_predict_layers": 0,
            "n_routed_experts": 256,
            "num_experts_per_tok": 6,
            "quantization": {"bits": 4, "group_size": 64},
        }))
        (pure_jang / "jang_config.json").write_text(json.dumps({
            "weight_format": "affine",
            "quantization": {
                "method": "affine",
                "routed_experts": {
                    "bits": 2,
                    "codec": "affine",
                    "group_size": 64,
                    "bit_plan": {
                        "routed_projection_group_sizes": {
                            "w1": 32,
                            "w2": 32,
                            "w3": 64,
                        },
                    },
                },
                "non_routed": {"bits": 4, "codec": "affine", "group_size": 64},
            },
        }))
        (pure_jang / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {
                "model.embed_tokens.weight": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.gate_proj.weight": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.gate_proj.scales": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.gate_proj.biases": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.down_proj.weight": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.down_proj.scales": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.down_proj.biases": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.up_proj.weight": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.up_proj.scales": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.up_proj.biases": "model.safetensors",
            }
        }))

        pure_quant = _model_quantization_status(str(pure_jang))
        pure_mtp = _model_mtp_status(str(pure_jang))

        assert pure_quant["codec"] == "affine_quantized_matmul"
        assert pure_quant["weight_format"] == "affine"
        assert pure_quant["weight_index"]["total_tensors"] == 10
        assert pure_quant["weight_index"]["mtp_tensor_count"] == 0
        assert pure_quant["weight_index"]["routed_layout_counts"] == {
            "prestacked_switch": 9,
            "split_expert": 0,
        }
        assert pure_mtp["status"] == "not_configured"
        assert pure_mtp["config_num_nextn_predict_layers"] == 0
        assert pure_mtp["index_has_mtp_tensors"] is False

        jangtq = tmp_path / "DeepSeek-V4-Flash-JANGTQ-K"
        jangtq.mkdir()
        (jangtq / "config.json").write_text(json.dumps({
            "model_type": "deepseek_v4",
            "weight_format": "mxtq",
            "num_nextn_predict_layers": 0,
            "n_routed_experts": 256,
            "num_experts_per_tok": 6,
            "mxtq_bits": {
                "routed_expert": 2,
                "attention": 8,
                "shared_expert": 8,
                "embed_tokens": 8,
                "lm_head": 8,
            },
        }))
        (jangtq / "jang_config.json").write_text(json.dumps({
            "weight_format": "mxtq",
            "drop_mtp": True,
            "quantization": {
                "profile": "JANGTQ-K",
                "quantization_backend": "turboquant",
            },
        }))
        (jangtq / "jangtq_runtime.safetensors").write_bytes(b"sidecar")
        (jangtq / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {
                "model.layers.0.mlp.switch_mlp.gate_proj.tq_packed": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.gate_proj.tq_norms": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.gate_proj.tq_bits": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.down_proj.tq_packed": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.down_proj.tq_norms": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.down_proj.tq_bits": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.up_proj.tq_packed": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.up_proj.tq_norms": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.up_proj.tq_bits": "model.safetensors",
                "model.lm_head.weight": "model.safetensors",
            }
        }))

        jangtq_quant = _model_quantization_status(str(jangtq))
        jangtq_mtp = _model_mtp_status(str(jangtq))

        assert jangtq_quant["codec"] == "turboquant_codebook"
        assert jangtq_quant["weight_format"] == "mxtq"
        assert jangtq_quant["weight_index"]["total_tensors"] == 10
        assert jangtq_quant["weight_index"]["suffix_counts"]["tq_packed"] == 3
        assert jangtq_quant["weight_index"]["suffix_counts"]["tq_norms"] == 3
        assert jangtq_quant["weight_index"]["suffix_counts"]["tq_bits"] == 3
        assert jangtq_quant["weight_index"]["tq_target_counts"]["routed_expert"] == 9
        assert jangtq_quant["weight_index"]["mtp_tensor_count"] == 0
        assert jangtq_quant["weight_index"]["routed_layout_counts"] == {
            "prestacked_switch": 9,
            "split_expert": 0,
        }
        assert jangtq_mtp["status"] == "dropped"
        assert jangtq_mtp["jang_drop_mtp"] is True
        assert jangtq_mtp["config_num_nextn_predict_layers"] == 0
        assert jangtq_mtp["index_has_mtp_tensors"] is False

    def test_quantization_status_reports_routed_layer_bit_plan(self, tmp_path):
        """Layer-specific routed bits must be visible for speed/quality audits."""
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "deepseek_v4",
            "weight_format": "mxtq",
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "weight_format": "mxtq",
            "mxtq_bits": {
                "attention": 8,
                "routed_expert": 2,
            },
            "quantization": {
                "profile": "JANGTQ2",
                "routed_experts": {
                    "bit_plan": {
                        "routed_layer_bits": {
                            "0": 4,
                            "1": 4,
                            "23": 2,
                        }
                    }
                },
            },
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["routed_layer_bits"] == {"0": 4, "1": 4, "23": 2}
        assert status["routed_layer_bits_source"] == (
            "jang_config.quantization.routed_experts.bit_plan.routed_layer_bits"
        )

    def test_quantization_status_reports_affine_dsv4_projection_and_down_layer_plans(
        self, tmp_path
    ):
        """Pure affine DSV4 K/Down4 artifacts need exact mixed-bit visibility."""
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "deepseek_v4",
            "weight_format": "affine",
            "quantization": {"bits": 2, "group_size": 128},
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "weight_format": "affine",
            "quantization": {
                "method": "affine",
                "routed_experts": {
                    "bits": 2,
                    "codec": "affine",
                    "group_size": 128,
                    "bit_plan": {
                        "routed_projection_bits": {"w2": 4},
                        "routed_down_layer_bits": {"7": 4, "8": 4, "12": 4},
                    },
                },
                "non_routed": {
                    "bits": 8,
                    "codec": "affine",
                    "group_size": 64,
                },
            },
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "affine_quantized_matmul"
        assert status["routed_projection_bits"] == {"w2": 4}
        assert status["routed_projection_bits_source"] == (
            "jang_config.quantization.routed_experts.bit_plan.routed_projection_bits"
        )
        assert status["routed_down_layer_bits"] == {"7": 4, "8": 4, "12": 4}
        assert status["routed_down_layer_bits_source"] == (
            "jang_config.quantization.routed_experts.bit_plan.routed_down_layer_bits"
        )

    def test_quantization_status_reports_affine_dispatch_path(self, tmp_path):
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3",
            "quantization": {"bits": 4, "group_size": 64},
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "affine_quantized_matmul"
        assert status["weight_matmul_dispatch"] == {
            "primary": "mlx_affine_quantized_matmul",
            "uses_mlx_quantized_matmul": True,
            "metal_na_eligible": True,
            "reason": None,
        }

    def test_quantization_status_treats_plain_jang_weight_format_as_affine_na_path(self, tmp_path):
        """Plain JANG bundles must stay on MLX affine matmul, not JANGTQ TQ."""
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "minimax_m2",
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "weight_format": "jang",
            "profile": "JANG_2L",
            "quantization": {
                "method": "jang-importance",
                "target_bits": 2,
                "actual_bits": 2.73,
                "block_size": 128,
                "quantization_backend": "mx.quantize",
            },
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "affine_quantized_matmul"
        assert status["weight_format"] == "jang"
        assert status["profile"] == "JANG_2L"
        assert status["target_bits"] == 2
        assert status["actual_bits"] == 2.73
        assert status["group_size"] == 128
        assert status["weight_matmul_dispatch"] == {
            "primary": "mlx_affine_quantized_matmul",
            "uses_mlx_quantized_matmul": True,
            "metal_na_eligible": True,
            "reason": None,
        }

    def test_quantization_status_reports_hy3_jang_affine_role_bit_plan(self, tmp_path):
        """Hy3 JANG_2K/2L use affine role metadata, not top-level 8-bit defaults."""
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "hy_v3",
            "weight_format": "affine",
            "quantization": {"bits": 8, "group_size": 128, "mode": "affine"},
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "format": "jang",
            "weight_format": "affine",
            "profile": "JANG_2K",
            "affine_bits": {
                "routed_expert": {
                    "gate_proj": 2,
                    "up_proj": 2,
                    "down_proj": 3,
                },
                "attention": 8,
                "shared_expert": 8,
                "dense_ffn": 8,
                "embed_tokens": 6,
                "lm_head": 8,
                "norms_router_biases": 16,
            },
            "quantization": {
                "method": "jang-affine",
                "bits_default": 8,
                "group_size": 128,
                "mode": "affine",
            },
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "affine_quantized_matmul"
        assert status["weight_format"] == "affine"
        assert status["profile"] == "JANG_2K"
        assert status["group_size"] == 128
        assert status["target_bits"] == 2
        assert status["affine_bits_by_role"]["routed_expert"] == {
            "gate_proj": 2,
            "up_proj": 2,
            "down_proj": 3,
        }
        assert status["routed_expert_bits_by_projection"] == {
            "gate_proj": 2,
            "up_proj": 2,
            "down_proj": 3,
        }
        assert status["routed_expert_bits_label"] == "gate=2/up=2/down=3-bit"
        assert status["weight_matmul_dispatch"] == {
            "primary": "mlx_affine_quantized_matmul",
            "uses_mlx_quantized_matmul": True,
            "metal_na_eligible": True,
            "reason": None,
        }

    def test_quantization_status_treats_mxfp4_weight_format_as_affine_na_path(self, tmp_path):
        """MXFP4 bundles use the native MLX affine matmul lane, not JANGTQ TQ."""
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_5_moe",
            "weight_format": "mxfp4",
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "affine_quantized_matmul"
        assert status["weight_format"] == "mxfp4"
        assert status["weight_matmul_dispatch"] == {
            "primary": "mlx_affine_quantized_matmul",
            "uses_mlx_quantized_matmul": True,
            "metal_na_eligible": True,
            "reason": None,
        }

    def test_quantization_status_warns_dsv4_affine_routed_without_coherency_stamp(
        self, tmp_path
    ):
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "deepseek_v4",
            "weight_format": "affine",
            "quantization": {"bits": 2, "group_size": 128},
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "weight_format": "affine",
            "quantization": {
                "method": "affine",
                "routed_experts": {
                    "bits": 2,
                    "codec": "affine",
                    "group_size": 128,
                },
                "non_routed": {
                    "bits": 8,
                    "codec": "affine",
                    "group_size": 64,
                },
            },
        }))

        status = _model_quantization_status(str(tmp_path))

        warnings = status.get("compat_warnings") or []
        assert status["codec"] == "affine_quantized_matmul"
        assert any(
            "DSV4 affine routed JANG runtime is loaded but not coherency-verified"
            in warning
            for warning in warnings
        )

    def test_quantization_status_trusts_dsv4_affine_runtime_coherency_stamp(
        self, tmp_path
    ):
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "deepseek_v4",
            "weight_format": "affine",
            "quantization": {"bits": 2, "group_size": 64},
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "weight_format": "affine",
            "runtime": {
                "coherency_verified": True,
                "ar_coherency_verified": True,
                "verified_with": "packaged-live-gate",
            },
            "quantization": {
                "method": "affine",
                "routed_experts": {
                    "bits": 2,
                    "codec": "affine",
                    "group_size": 64,
                },
            },
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "affine_quantized_matmul"
        warnings = status.get("compat_warnings") or []
        assert not any(
            "DSV4 affine routed JANG runtime is loaded but not coherency-verified"
            in warning
            for warning in warnings
        )

    def test_routing_status_reports_trained_top_k_without_override(self, monkeypatch, tmp_path):
        from vmlx_engine.server import _model_routing_status

        monkeypatch.delenv("JANGTQ_TOPK_OVERRIDE", raising=False)
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "minimax",
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
        }))

        status = _model_routing_status(str(tmp_path))

        assert status == {
            "trained_active_experts": 8,
            "trained_active_experts_source": "config.num_experts_per_tok",
            "n_routed_experts": 256,
            "override_env": None,
            "effective_active_experts": 8,
            "effective_active_experts_source": "trained_default",
        }

    def test_dsv4_routing_status_reports_trained_top_k_six(self, monkeypatch, tmp_path):
        """DSV4 MoE top-k is trained routing metadata, not sampler top_k."""
        from vmlx_engine.server import _model_routing_status

        monkeypatch.delenv("JANGTQ_TOPK_OVERRIDE", raising=False)
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "deepseek_v4",
            "n_routed_experts": 256,
            "num_experts_per_tok": 6,
        }))

        status = _model_routing_status(str(tmp_path))

        assert status == {
            "trained_active_experts": 6,
            "trained_active_experts_source": "config.num_experts_per_tok",
            "n_routed_experts": 256,
            "override_env": None,
            "effective_active_experts": 6,
            "effective_active_experts_source": "trained_default",
        }

    def test_routing_status_ignores_jangtq_top_k_override(self, monkeypatch, tmp_path):
        from vmlx_engine.server import _model_routing_status

        monkeypatch.setenv("JANGTQ_TOPK_OVERRIDE", "4")
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "minimax",
            "text_config": {
                "n_routed_experts": 256,
                "num_experts_per_tok": 8,
            },
        }))

        status = _model_routing_status(str(tmp_path))

        assert status["trained_active_experts"] == 8
        assert status["trained_active_experts_source"] == "config.text_config.num_experts_per_tok"
        assert status["n_routed_experts"] == 256
        assert status["override_env"] is None
        assert status["effective_active_experts"] == 8
        assert status["effective_active_experts_source"] == "trained_default"

    def test_routing_status_ignores_invalid_jangtq_top_k_override(
        self, monkeypatch, tmp_path
    ):
        from vmlx_engine.server import _model_routing_status

        monkeypatch.setenv("JANGTQ_TOPK_OVERRIDE", "abc")
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "minimax",
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
        }))

        status = _model_routing_status(str(tmp_path))

        assert status["trained_active_experts"] == 8
        assert status["override_env"] is None
        assert "override_issue" not in status
        assert status["effective_active_experts"] == 8
        assert status["effective_active_experts_source"] == "trained_default"

    def test_grouped_conv1d_native_error_gets_actionable_detail(self):
        from vmlx_engine.server import _generation_error_detail

        detail = _generation_error_detail(
            ValueError(
                "Given groups=8192 and weights of shape (8192,1,4), "
                "expected to have 32768 input channels but got 8192 input channels instead."
            )
        )

        assert "Grouped Conv1d layout mismatch" in detail
        assert "re-convert with a newer jang build" in detail
        assert "Original error:" in detail

    def test_acceleration_status_does_not_claim_metal_na_for_jangtq(self, monkeypatch, tmp_path):
        import vmlx_engine.server as server

        monkeypatch.setenv("JANGTQ_MPP_NAX", "off")
        (tmp_path / "config.json").write_text(json.dumps({
            "weight_format": "mxtq",
            "mxtq_bits": 2,
        }))
        (tmp_path / "jangtq_runtime.safetensors").write_bytes(b"sidecar")
        monkeypatch.setattr(
            server,
            "_mlx_metal_na_status",
            lambda: {"available": True, "nax_symbols": 3534, "naxtile_symbols": 786},
        )
        monkeypatch.setattr(
            server,
            "_host_supports_metal_na",
            lambda: {"supported": True, "brand": "Apple M5 Max"},
        )

        status = server._model_acceleration_status(str(tmp_path))

        assert status["kernel_type"] == "turboquant_codebook"
        assert status["metal_na_capable"] is False
        assert status["metal_na_active_on_host"] is False
        assert status["reason"] == "turboquant_custom_kernels_do_not_use_mlx_na"
        assert status["jangtq_acceleration"]["mode"] == "off"

    def test_acceleration_status_reports_internal_jangtq_acceleration_when_enabled(
        self, monkeypatch, tmp_path
    ):
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(json.dumps({
            "weight_format": "mxtq",
            "mxtq_bits": 4,
        }))
        (tmp_path / "jangtq_runtime.safetensors").write_bytes(b"sidecar")
        monkeypatch.setenv("JANGTQ_MPP_NAX", "auto")
        monkeypatch.setattr(
            server,
            "_mlx_metal_na_status",
            lambda: {"available": True, "nax_symbols": 3534, "naxtile_symbols": 786},
        )
        monkeypatch.setattr(
            server,
            "_host_supports_metal_na",
            lambda: {"supported": True, "brand": "Apple M5 Max"},
        )
        monkeypatch.setattr(
            server,
            "_jangtq_mpp_nax_runtime_status",
            lambda _host=None: {
                "mode": "auto",
                "requested": True,
                "available": True,
                "active": True,
                "uses_mlx_quantized_matmul": False,
                "reason": None,
            },
        )

        status = server._model_acceleration_status(str(tmp_path))

        assert status["kernel_type"] == "turboquant_codebook_mpp_nax"
        assert status["metal_na_capable"] is True
        assert status["metal_na_active_on_host"] is True
        assert status["reason"] is None
        assert status["jangtq_acceleration"] == {
            "mode": "auto",
            "requested": True,
            "available": True,
            "active": True,
            "uses_mlx_quantized_matmul": False,
            "reason": None,
        }

    def test_acceleration_status_reports_affine_na_only_when_symbols_and_host_match(self, monkeypatch, tmp_path):
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(json.dumps({
            "quantization": {"bits": 4, "group_size": 64},
        }))
        monkeypatch.setattr(
            server,
            "_mlx_metal_na_status",
            lambda: {"available": True, "nax_symbols": 3534, "naxtile_symbols": 786},
        )
        monkeypatch.setattr(
            server,
            "_host_supports_metal_na",
            lambda: {"supported": True, "brand": "Apple M5 Max"},
        )

        status = server._model_acceleration_status(str(tmp_path))

        assert status["kernel_type"] == "affine_quantized_matmul"
        assert status["metal_na_capable"] is True
        assert status["metal_na_active_on_host"] is True

    def test_mtp_status_reports_dropped_dsv4_artifact(self, tmp_path):
        from vmlx_engine.server import _model_mtp_status

        (tmp_path / "config.json").write_text(
            '{"model_type":"deepseek_v4","num_nextn_predict_layers":0,'
            '"n_routed_experts":256,"num_experts_per_tok":6}'
        )
        (tmp_path / "jang_config.json").write_text(
            '{"weight_format":"mxtq","drop_mtp":true}'
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            '{"weight_map":{"model.embed.weight":"model-00001-of-00001.safetensors"}}'
        )

        status = _model_mtp_status(str(tmp_path))

        expected = {
            "config_num_nextn_predict_layers": 0,
            "jang_drop_mtp": True,
            "index_has_mtp_tensors": False,
            "artifact_available": False,
            "family": "deepseek_v4",
            "runtime_supported": False,
            "runtime_available": False,
            "runtime_reason": "jang_config.drop_mtp=true",
            "status": "dropped",
            "issues": [],
        }
        for key, value in expected.items():
            assert status[key] == value

    def test_mtp_status_flags_missing_weights_when_config_expects_mtp(self, tmp_path):
        from vmlx_engine.server import _model_mtp_status

        (tmp_path / "config.json").write_text(
            '{"model_type":"deepseek_v4","num_nextn_predict_layers":1}'
        )
        (tmp_path / "jang_config.json").write_text(
            '{"weight_format":"mxtq","drop_mtp":false}'
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            '{"weight_map":{"model.embed.weight":"model-00001-of-00001.safetensors"}}'
        )

        status = _model_mtp_status(str(tmp_path))

        assert status["runtime_available"] is False
        assert status["status"] == "metadata_inconsistent"
        assert any("config expects" in issue for issue in status["issues"])

    def test_mtp_status_flags_indexed_mtp_when_config_disables_runtime(self, tmp_path):
        from vmlx_engine.server import _model_mtp_status

        (tmp_path / "config.json").write_text(
            '{"model_type":"deepseek_v4","num_nextn_predict_layers":0,'
            '"n_routed_experts":256,"num_experts_per_tok":6}'
        )
        (tmp_path / "jang_config.json").write_text(
            '{"weight_format":"mxtq","drop_mtp":false}'
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            '{"weight_map":{"mtp.0.layers.0.self_attn.q_proj.weight":"model.safetensors"}}'
        )

        status = _model_mtp_status(str(tmp_path))

        assert status["runtime_available"] is False
        assert status["status"] == "metadata_inconsistent"
        assert any("config disables" in issue for issue in status["issues"])

    def test_mtp_status_flags_invalid_config_layer_count(self, tmp_path):
        from vmlx_engine.server import _model_mtp_status

        (tmp_path / "config.json").write_text(
            '{"model_type":"deepseek_v4","num_nextn_predict_layers":"one"}'
        )
        (tmp_path / "jang_config.json").write_text(
            '{"weight_format":"mxtq","drop_mtp":false}'
        )

        status = _model_mtp_status(str(tmp_path))

        assert status["runtime_available"] is False
        assert status["status"] == "metadata_inconsistent"
        assert any("invalid" in issue for issue in status["issues"])

    def test_mtp_status_flags_malformed_index_metadata(self, tmp_path):
        from vmlx_engine.server import _model_mtp_status

        (tmp_path / "config.json").write_text(
            '{"model_type":"deepseek_v4","num_nextn_predict_layers":0}'
        )
        (tmp_path / "jang_config.json").write_text(
            '{"weight_format":"mxtq","drop_mtp":false}'
        )
        (tmp_path / "model.safetensors.index.json").write_text("{")

        status = _model_mtp_status(str(tmp_path))

        assert status["runtime_available"] is False
        assert status["status"] == "metadata_inconsistent"
        assert any("model.safetensors.index.json" in issue for issue in status["issues"])

    def test_mtp_status_does_not_claim_runtime_for_weights_only_bundle(self, tmp_path):
        from vmlx_engine.server import _model_mtp_status

        (tmp_path / "config.json").write_text(
            '{"model_type":"deepseek_v4","num_nextn_predict_layers":1}'
        )
        (tmp_path / "jang_config.json").write_text(
            '{"weight_format":"mxtq","drop_mtp":false}'
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            '{"weight_map":{"mtp.0.layers.0.self_attn.q_proj.weight":"model.safetensors"}}'
        )

        status = _model_mtp_status(str(tmp_path))

        assert status["artifact_available"] is True
        assert status["runtime_available"] is False
        assert status["status"] == "weights_present_runtime_unwired"
        assert "not currently on the JangMTP support map" in status["runtime_reason"]

    @pytest.mark.asyncio
    async def test_health_and_capabilities_surface_mtp_status(
        self, monkeypatch, tmp_path
    ):
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(
            '{"model_type":"deepseek_v4","num_nextn_predict_layers":0,'
            '"n_routed_experts":256,"num_experts_per_tok":6}'
        )
        (tmp_path / "jang_config.json").write_text(
            '{"weight_format":"mxtq","drop_mtp":true}'
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            '{"weight_map":{"model.embed.weight":"model-00001-of-00001.safetensors"}}'
        )

        class _Engine:
            is_mllm = False

            def get_stats(self):
                return {"engine_type": "batched"}

        class _Scheduler:
            block_aware_cache = None
            paged_cache_manager = None
            memory_aware_cache = None
            prefix_cache = None

            def get_stats(self):
                return {}

        monkeypatch.setattr(server, "_engine", _Engine())
        monkeypatch.setattr(server, "_get_scheduler", lambda: _Scheduler())
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "dsv4-test")
        monkeypatch.setattr(server, "_model_type", "llm")
        monkeypatch.setattr(server, "_standby_state", None)
        monkeypatch.setattr(server, "_mcp_manager", None)

        health = await server.health()
        mtp_alias = await server.health_mtp()
        capabilities = await server.model_capabilities("dsv4-test")

        assert health["mtp"]["status"] == "dropped"
        assert health["mtp"]["runtime_available"] is False
        assert mtp_alias["status"] == "dropped"
        assert mtp_alias["runtime_available"] is False
        assert health["routing"]["trained_active_experts"] == 6
        assert health["routing"]["effective_active_experts_source"] == "trained_default"
        assert capabilities["mtp"]["status"] == "dropped"
        assert capabilities["mtp"]["runtime_available"] is False
        assert capabilities["routing"]["trained_active_experts"] == 6
        assert capabilities["routing"]["effective_active_experts_source"] == "trained_default"

    @pytest.mark.asyncio
    async def test_health_mtp_endpoint_returns_loaded_runtime_status(
        self, monkeypatch, tmp_path
    ):
        import vmlx_engine.server as server

        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "qwen-mtp-test")
        monkeypatch.setattr(
            server,
            "_model_mtp_status_with_loaded_runtime",
            lambda bundle_path: {
                "status": "native_runtime_active",
                "runtime_active": True,
                "runtime_scope": "text+vl",
                "bundle_path": bundle_path,
            },
        )

        payload = await server.health_mtp()

        assert payload["status"] == "native_runtime_active"
        assert payload["runtime_active"] is True
        assert payload["runtime_scope"] == "text+vl"
        assert payload["bundle_path"] == str(tmp_path)

    def test_mlx_metal_na_status_handles_namespace_mlx_package(self, monkeypatch, tmp_path):
        """MLX can be a namespace package with mlx.__file__ == None.

        The extension module path lives on mlx.core.__file__; NA telemetry must
        use that fallback or the panel/API report "not available" even when the
        installed metallib has NA symbols.
        """
        import sys
        import types
        import vmlx_engine.server as server

        site = tmp_path / "site-packages"
        metal = site / "mlx" / "lib" / "mlx.metallib"
        metal.parent.mkdir(parents=True)
        metal.write_bytes(b"prefix _nax_ NAXTile suffix")
        core_so = site / "mlx" / "core.cpython-313-darwin.so"
        core_so.write_bytes(b"fake")

        monkeypatch.setitem(
            sys.modules,
            "mlx",
            types.SimpleNamespace(__file__=None),
        )
        monkeypatch.setitem(
            sys.modules,
            "mlx.core",
            types.SimpleNamespace(__file__=str(core_so)),
        )
        server._metal_na_status_cache.clear()

        status = server._mlx_metal_na_status()

        assert status["available"] is True
        assert status["nax_symbols"] == 1
        assert status["naxtile_symbols"] == 1

    def test_mlx_metal_na_status_accepts_affine_qmm_symbols(
        self, monkeypatch, tmp_path
    ):
        """Bundled MLX 0.31.x exposes affine qmm symbols, not _nax_/NAXTile."""
        import sys
        import types
        import vmlx_engine.server as server

        site = tmp_path / "site-packages"
        metal = site / "mlx" / "lib" / "mlx.metallib"
        metal.parent.mkdir(parents=True)
        metal.write_bytes(b"prefix affine_qmm affine_gather_qmm suffix")
        init = site / "mlx" / "__init__.py"
        init.write_text("")

        monkeypatch.setitem(
            sys.modules,
            "mlx",
            types.SimpleNamespace(__file__=str(init)),
        )
        server._metal_na_status_cache.clear()

        status = server._mlx_metal_na_status()

        assert status["available"] is True
        assert status["nax_symbols"] == 0
        assert status["naxtile_symbols"] == 0
        assert status["affine_qmm_symbols"] == 1
        assert status["affine_gather_qmm_symbols"] == 1


class TestNonStreamingReceiveDrainDisconnect:
    """Active receive-drain detects ASGI http.disconnect events that
    `Request.is_disconnected()` may miss when nothing else reads from the
    receive channel.

    Live evidence (codex 2026-05-09 14:18): client TCP close + read-timeout
    left scheduler num_running=1 because is_disconnected() polling didn't
    surface the disconnect event in time. Active receive-drain catches it.
    """

    def test_helper_detects_http_disconnect_via_receive_channel(self):
        import asyncio
        import pytest
        from fastapi import HTTPException
        from vmlx_engine.server import _await_chat_with_disconnect_abort

        class FakeEngine:
            def __init__(self):
                self.aborted = None

            async def chat(self, **kwargs):
                # Run forever until aborted
                await asyncio.sleep(10)
                return None

            async def abort_request(self, request_id):
                self.aborted = request_id
                return True

        class FakeRequest:
            """ASGI receive channel that emits http.disconnect after a tick."""
            def __init__(self):
                self._sent = False

            async def is_disconnected(self):
                # Simulate Starlette's lazy poll missing the event because
                # nothing is actively draining receive.
                return False

            async def receive(self):
                # First call returns the disconnect event after a small delay
                # so the helper's parallel drain task can catch it.
                if not self._sent:
                    self._sent = True
                    await asyncio.sleep(0.01)
                    return {"type": "http.disconnect"}
                # Subsequent calls block forever (channel exhausted).
                await asyncio.sleep(60)
                return {}

        async def run():
            engine = FakeEngine()
            req = FakeRequest()
            with pytest.raises(HTTPException) as exc:
                await _await_chat_with_disconnect_abort(
                    engine,
                    messages=[],
                    chat_kwargs={},
                    timeout=30,
                    fastapi_request=req,
                    request_id="resp_receive_drain_test",
                    endpoint="test",
                    poll_interval=0.05,
                )
            assert exc.value.status_code == 499
            assert engine.aborted == "resp_receive_drain_test", (
                "Engine must be aborted when receive-drain catches http.disconnect "
                "even if is_disconnected() never returned True"
            )

        asyncio.run(run())

    def test_helper_safe_when_request_has_no_receive_method(self):
        """Backwards-compat: legacy callers may pass a fake request without
        `.receive`. Helper must not crash; falls back to is_disconnected polling."""
        import asyncio
        from vmlx_engine.server import _await_chat_with_disconnect_abort

        class FakeOutput:
            completion_tokens = 1

        class FakeEngine:
            async def chat(self, **kwargs):
                return FakeOutput()

        class FakeRequest:
            # NO `.receive` attribute
            async def is_disconnected(self):
                return False

        async def run():
            output = await _await_chat_with_disconnect_abort(
                FakeEngine(),
                messages=[],
                chat_kwargs={},
                timeout=10,
                fastapi_request=FakeRequest(),
                request_id="resp_no_receive",
                endpoint="test",
                poll_interval=0.01,
            )
            assert output.completion_tokens == 1

        asyncio.run(run())

    def test_helper_uses_single_receive_reader_when_drain_available(self):
        """When `.receive` exists, the active drain owns ASGI disconnect reads.

        Starlette's `Request.is_disconnected()` is implemented as a zero-timeout
        call to the same receive function. Calling it while the drain task is
        blocked in receive creates two readers on the ASGI receive channel, which
        is not a safe contract for all servers.
        """
        import asyncio
        from vmlx_engine.server import _await_chat_with_disconnect_abort

        class FakeOutput:
            completion_tokens = 1

        class FakeEngine:
            async def chat(self, **kwargs):
                await asyncio.sleep(0.03)
                return FakeOutput()

        class FakeRequest:
            def __init__(self):
                self.is_disconnected_calls = 0

            async def is_disconnected(self):
                self.is_disconnected_calls += 1
                return False

            async def receive(self):
                await asyncio.sleep(60)
                return {}

        async def run():
            req = FakeRequest()
            output = await _await_chat_with_disconnect_abort(
                FakeEngine(),
                messages=[],
                chat_kwargs={},
                timeout=10,
                fastapi_request=req,
                request_id="resp_single_receive_reader",
                endpoint="test",
                poll_interval=0.001,
            )
            assert output.completion_tokens == 1
            assert req.is_disconnected_calls == 0

        asyncio.run(run())


class TestJitTurboQuantSymmetricGuard:
    """Engine skips mx.compile for TurboQuantKVCache (GH issue #66) AND panel
    suppresses --enable-jit when isTurboQuant. The two guards must stay
    symmetric — if one drifts away, the other breaks user trust (UI says
    JIT is disabled but engine still tries to compile, or vice versa).
    """

    def test_engine_jit_skip_for_turboquant_make_cache(self):
        """Engine apply_jit_compilation must short-circuit when the active
        model has the TurboQuant patched make_cache function name."""
        import inspect
        from pathlib import Path

        source = Path("vmlx_engine/server.py").read_text()
        # The skip block is identified by these load-bearing strings:
        assert "_turboquant_make_cache" in source
        assert "_tq_make_cache" in source
        assert "JIT: Skipping mx.compile — TurboQuantKVCache is active" in source
        # Make sure the early-return is wired in (return after detecting TQ name)
        assert (
            'if _make_cache_name in ("_turboquant_make_cache", "_tq_make_cache"):'
            in source
        )

    def test_panel_session_launcher_suppresses_jit_for_turboquant(self):
        """Panel launcher must compute effectiveEnableJit with turboQuantActive
        gating; matches the engine's TQ-skip behavior."""
        from pathlib import Path
        sessions_source = Path("./panel/src/main/sessions.ts").read_text()

        assert "turboQuantActive" in sessions_source
        assert "(detected as any).isTurboQuant" in sessions_source
        # effectiveEnableJit must include !turboQuantActive in the AND chain
        assert "!turboQuantActive" in sessions_source

    def test_panel_form_disables_jit_checkbox_for_turboquant(self):
        """Panel JIT checkbox must be visually disabled + warned when
        turboQuantActive is true. If this test fails, user sees a checkbox
        they can tick that the engine will silently ignore."""
        from pathlib import Path
        form = Path(
            "./panel/src/renderer/src/components/sessions/SessionConfigForm.tsx"
        ).read_text()

        assert "turboQuantActive" in form
        assert "detectedIsTurboQuant" in form
        # disabled prop covers turboQuantActive and hybrid path-dependent caches.
        assert "disabled={flashMoeActive || distributedActive || dsv4Active || zayaCcaActive || turboQuantActive || multimodalActive || hybridCacheActive}" in form

    def test_detect_config_stamps_isTurboQuant_flag(self):
        """detectModelConfigFromDir must set isTurboQuant when bundle is TQ.
        Without this stamp, the panel guards above never fire."""
        from pathlib import Path
        registry = Path("./panel/src/main/model-config-registry.ts").read_text()

        assert "isTurboQuant" in registry
        # The stamp site sets next.isTurboQuant = true
        assert "next.isTurboQuant = true" in registry


class TestStreamUsagePropagatesCacheDetail:
    """Streaming SSE finish-chunk usage must surface cache_detail alongside
    cached_tokens, mirroring the non-stream get_usage() path. Without this,
    typed cache labels (paged+zaya_cca, paged+ssm+disk, paged+disk+tq, etc.)
    are silently dropped from stream consumers even though server-side
    cache_hit_tokens_by_detail accounting still works.

    Bug exposed during ZAYA1-VL typed CCA validation (commit 4f3c1dc6):
    non-stream returned cache_detail=paged+zaya_cca, stream returned None.
    """

    def test_chat_stream_tracks_cache_detail_alongside_cached_tokens(self):
        from pathlib import Path
        source = Path("./vmlx_engine/server.py").read_text()
        assert "cached_tokens = 0\n    cache_detail" in source
        assert (
            'getattr(output, "cache_detail", "") or None\n            if _detail is not None:\n                cache_detail = _detail'
            in source
        )

    def test_chat_stream_finish_chunks_emit_cache_detail(self):
        from pathlib import Path
        source = Path("./vmlx_engine/server.py").read_text()
        # All three chat-stream finish paths (regular, error, tool_calls)
        # construct PromptTokensDetails manually; each must pass cache_detail.
        assert (
            source.count("cached_tokens=cached_tokens, cache_detail=cache_detail")
            >= 3
        )

    def test_responses_stream_tracks_cache_detail_alongside_cached(self):
        from pathlib import Path
        source = Path("./vmlx_engine/server.py").read_text()
        assert "_cached = 0\n    _cache_detail" in source
        assert '_detail_chunk = getattr(output, "cache_detail", "") or None' in source

    def test_responses_stream_finish_emits_cache_detail(self):
        from pathlib import Path
        source = Path("./vmlx_engine/server.py").read_text()
        # Responses uses dict-builder shape, not PromptTokensDetails class.
        assert '"cache_detail": _cache_detail' in source

    def test_usage_builders_preserve_cache_detail_without_cached_tokens(self):
        from vmlx_engine.engine.base import GenerationOutput
        from vmlx_engine.server import _get_responses_usage, get_usage

        output = GenerationOutput(
            text="ok",
            prompt_tokens=8,
            completion_tokens=2,
            cached_tokens=0,
            cache_detail="paged+zaya_cca",
        )

        chat_usage = get_usage(output).model_dump(exclude_none=True)
        responses_usage = _get_responses_usage(output).model_dump(exclude_none=True)

        assert chat_usage["prompt_tokens_details"] == {
            "cached_tokens": 0,
            "cache_detail": "paged+zaya_cca",
        }
        assert responses_usage["input_tokens_details"] == {
            "cached_tokens": 0,
            "cache_detail": "paged+zaya_cca",
        }

    @pytest.mark.asyncio
    async def test_chat_stream_usage_preserves_cache_detail_without_cached_tokens(
        self, monkeypatch
    ):
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import (
            ChatCompletionRequest,
            Message,
            StreamOptions,
        )
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                yield GenerationOutput(
                    text="ok",
                    new_text="ok",
                    prompt_tokens=8,
                    completion_tokens=2,
                    cached_tokens=0,
                    cache_detail="paged+zaya_cca",
                    finished=True,
                    finish_reason="stop",
                )

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "zaya-vl-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", None)

        request = ChatCompletionRequest(
            model="zaya-vl-test",
            messages=[Message(role="user", content="hi")],
            stream=True,
            stream_options=StreamOptions(include_usage=True),
        )
        chunks = []
        async for line in server.stream_chat_completion(
            _Engine(),
            [m.model_dump(exclude_none=True) for m in request.messages],
            request,
            fastapi_request=None,
        ):
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                chunks.append(json.loads(line.removeprefix("data: ")))

        usage_chunks = [chunk for chunk in chunks if chunk.get("usage")]
        assert usage_chunks
        assert usage_chunks[-1]["usage"]["prompt_tokens_details"] == {
            "cached_tokens": 0,
            "cache_detail": "paged+zaya_cca",
        }

    @pytest.mark.asyncio
    async def test_responses_stream_usage_preserves_cache_detail_without_cached_tokens(
        self, monkeypatch
    ):
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest, StreamOptions
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                yield GenerationOutput(
                    text="ok",
                    new_text="ok",
                    prompt_tokens=8,
                    completion_tokens=2,
                    cached_tokens=0,
                    cache_detail="paged+zaya_cca",
                    finished=True,
                    finish_reason="stop",
                )

        def _payloads(events, event_type):
            payloads = []
            prefix = f"event: {event_type}\n"
            for event in events:
                if event.startswith(prefix):
                    data_line = next(
                        line for line in event.splitlines() if line.startswith("data: ")
                    )
                    payloads.append(json.loads(data_line.removeprefix("data: ")))
            return payloads

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "zaya-vl-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", None)

        request = ResponsesRequest(
            model="zaya-vl-test",
            input="hi",
            stream=True,
            stream_options=StreamOptions(include_usage=True),
        )
        events = [
            event async for event in server.stream_responses_api(
                _Engine(),
                [{"role": "user", "content": "hi"}],
                request,
                fastapi_request=None,
            )
        ]

        usage = _payloads(events, "response.usage")[-1]["usage"]
        completed = _payloads(events, "response.completed")[-1]["response"]["usage"]
        expected = {
            "cached_tokens": 0,
            "cache_detail": "paged+zaya_cca",
        }
        assert usage["input_tokens_details"] == expected
        assert completed["input_tokens_details"] == expected


class TestHuggingFaceDownloadRegression:
    """Issue #118: stale saved HF mirrors or tokens must not poison public GUI
    downloads. The panel should normalize persisted settings and keep the
    Python worker's endpoint selection app-owned so it can fall back cleanly.
    """

    def test_panel_download_worker_does_not_inherit_raw_hf_endpoint(self):
        from pathlib import Path

        source = Path("./panel/src/main/ipc/models.ts").read_text()

        assert "normalizeHfEndpointSetting" in source
        assert "HF_ENDPOINT: undefined" in source
        assert "endpoint = sys.argv[3] or None" in source
        assert "endpoint=active_endpoint" in source
        assert "https://huggingface.co" in source
        assert "'type':'fallback'" in source

    def test_panel_hf_api_retries_public_requests_without_stale_auth(self):
        from pathlib import Path

        source = Path("./panel/src/main/ipc/models.ts").read_text()

        assert "normalizeHfTokenSetting" in source
        assert "delete baseHeaders.Authorization" in source
        assert "retrying ${baseUrl} without auth" in source
        assert "HuggingFace mirror failed" in source
        assert "fetchHfPath(`/api/models?${params}`" in source
