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
                {"type": "text", "text": "what changed?", "content": "what changed?"},
            ],
        }
    ]


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
