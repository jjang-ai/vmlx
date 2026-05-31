# SPDX-License-Identifier: Apache-2.0
"""Multimodal content routing normalization tests."""


def test_chat_extractor_accepts_responses_style_input_media_parts():
    from vmlx_engine.api.models import Message
    from vmlx_engine.api.utils import extract_multimodal_content

    messages = [
        Message(
            role="user",
            content=[
                {"type": "input_text", "text": "describe these"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AAAA",
                },
                {
                    "type": "input_video",
                    "video_url": {"url": "data:video/mp4;base64,BBBB"},
                },
            ],
        )
    ]

    processed, images, videos = extract_multimodal_content(messages)

    assert processed == [{"role": "user", "content": "describe these"}]
    assert images == ["data:image/png;base64,AAAA"]
    assert videos == ["data:video/mp4;base64,BBBB"]


def test_mllm_extractor_accepts_responses_style_input_media_parts():
    from vmlx_engine.models.mllm import MLXMultimodalLM

    chat_messages, images, videos = MLXMultimodalLM._extract_multimodal_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what changed?"},
                    {
                        "type": "input_image",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                    {
                        "type": "input_video",
                        "video_url": "data:video/mp4;base64,BBBB",
                    },
                ],
            }
        ]
    )

    assert images == ["data:image/png;base64,AAAA"]
    assert videos == ["data:video/mp4;base64,BBBB"]
    assert chat_messages == [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "video"},
                {"type": "text", "text": "what changed?", "content": "what changed?"},
            ],
        }
    ]


def test_mllm_thinking_off_does_not_synthesize_empty_think_block(monkeypatch):
    import mlx_vlm.prompt_utils as prompt_utils

    from vmlx_engine.models.mllm import MLXMultimodalLM

    def fake_get_chat_template(_processor, _messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["add_generation_prompt"] is True
        return "user: describe image\nassistant:"

    monkeypatch.setattr(prompt_utils, "get_chat_template", fake_get_chat_template)

    model = MLXMultimodalLM.__new__(MLXMultimodalLM)
    model.processor = object()
    model.model_name = "Qwen3.6-VL"
    model.config = {
        "model_type": "qwen3_5_vl",
        "capabilities": {
            "supports_thinking": True,
            "think_in_template": True,
        },
    }

    prompt = model._apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": "describe image"}]}],
        enable_thinking=False,
    )

    assert prompt == "user: describe image\nassistant:"
    assert "<think>" not in prompt


def test_step37_processor_exposes_chat_template_for_source_vlm_route():
    from vmlx_engine.models.step3p7_mlx_vlm import Step3VLProcessor

    captured = {}

    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return "<|im_start|>user\n<im_patch>What color?<|im_end|>\n<|im_start|>assistant\n<think>\n"

    processor = Step3VLProcessor(tokenizer=_Tokenizer())
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What color?"},
            ],
        }
    ]
    tools = [{"type": "function", "function": {"name": "run_command"}}]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        tools=tools,
    )

    assert prompt.startswith("<|im_start|>user")
    assert "<im_patch>" in prompt
    assert captured["messages"] == messages
    assert captured["kwargs"]["tokenize"] is False
    assert captured["kwargs"]["add_generation_prompt"] is True
    assert captured["kwargs"]["enable_thinking"] is False
    assert captured["kwargs"]["tools"] == tools


def test_server_detects_media_before_text_only_responses_flattening():
    import vmlx_engine.server as server

    assert server._responses_input_has_multimodal(
        [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "describe"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                ],
            }
        ]
    )


def test_server_rejects_text_only_multimodal_instead_of_silent_drop(monkeypatch):
    import pytest
    import vmlx_engine.server as server

    monkeypatch.setattr(server, "_model_path", "/tmp/not-omni")
    monkeypatch.setattr(server, "_model_name", None)

    with pytest.raises(Exception) as exc:
        server._reject_unsupported_multimodal("/v1/chat/completions")

    assert getattr(exc.value, "status_code", None) == 400
    assert "silently ignoring" in str(exc.value.detail) or "silent media drop" in str(exc.value.detail)
