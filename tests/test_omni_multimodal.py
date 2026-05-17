import json

import pytest
from fastapi import HTTPException

from vmlx_engine.omni_multimodal import (
    OmniMultimodalDispatcher,
    _build_omni_turn_prompt_with_thinking,
    dispatch_omni_chat_completion,
    _extract_parts,
    is_omni_multimodal_bundle,
    omni_multimodal_component_status,
    request_has_multimodal,
    request_modalities,
)


class _FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        rail = "<think></think>" if kwargs.get("enable_thinking") is False else "<think>\n"
        return f"rendered:{messages[-1]['content']}:{rail}"


def _write_omni_bundle(
    tmp_path,
    *,
    radio: bool = True,
    parakeet: bool = True,
    projector: bool = True,
    video_preprocessor: bool = True,
    model_type: str = "nemotron_h",
    sound_model_type: str = "parakeet",
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    (tmp_path / "config_omni.json").write_text(
        json.dumps({"sound_config": {"model_type": sound_model_type}})
    )
    (tmp_path / "configuration_radio.py").write_text("# radio config placeholder\n")
    weight_map = {}
    if radio:
        weight_map[
            "vision_model.radio_model.model.blocks.0.attn.qkv.weight"
        ] = "model.safetensors"
    if parakeet:
        weight_map[
            "sound_encoder.encoder.layers.0.conv.depthwise_conv.weight"
        ] = "model.safetensors"
    if projector:
        weight_map["mlp1.0.weight"] = "model.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    if video_preprocessor:
        (tmp_path / "video_preprocessor_config.json").write_text(
            json.dumps({"video_processor_type": "FakeVideoProcessor"})
        )
    return tmp_path


def test_omni_component_status_requires_radio_parakeet_and_projector(tmp_path):
    bundle = _write_omni_bundle(tmp_path / "omni")

    status = omni_multimodal_component_status(bundle)

    assert status["bundle_compatible"] is True
    assert status["config_model_type"] == "nemotron_h"
    assert status["sound_config_model_type"] == "parakeet"
    assert status["has_radio_weights"] is True
    assert status["has_parakeet_weights"] is True
    assert status["has_media_projector"] is True
    assert status["has_video_preprocessor_config"] is True
    assert status["video_bridge_supported"] is True
    assert status["modalities"] == ["text", "audio", "image", "video"]
    assert status["missing"] == []
    assert is_omni_multimodal_bundle(bundle) is True


def test_omni_component_status_omits_video_without_runtime_bridge(tmp_path):
    bundle = _write_omni_bundle(tmp_path / "omni", video_preprocessor=False)

    status = omni_multimodal_component_status(bundle)

    assert status["bundle_compatible"] is True
    assert status["has_radio_weights"] is True
    assert status["has_video_preprocessor_config"] is False
    assert status["video_bridge_supported"] is False
    assert status["modalities"] == ["text", "audio", "image"]
    assert status["missing"] == []
    assert is_omni_multimodal_bundle(bundle) is True


def test_omni_component_status_rejects_vision_only_bundle(tmp_path):
    bundle = _write_omni_bundle(tmp_path / "omni", parakeet=False)

    status = omni_multimodal_component_status(bundle)

    assert status["bundle_compatible"] is False
    assert status["has_radio_weights"] is True
    assert status["has_parakeet_weights"] is False
    assert "parakeet weights" in status["missing"]
    assert is_omni_multimodal_bundle(bundle) is False


def test_omni_prompt_builder_forwards_enable_thinking_false_on_first_turn():
    tok = _FakeTokenizer()

    prompt = _build_omni_turn_prompt_with_thinking(
        tok,
        "Describe the image directly.",
        n_image_tokens=2,
        is_first=True,
        enable_thinking=False,
    )

    assert tok.calls[0][0][0]["role"] == "system"
    assert "Answer directly" in tok.calls[0][0][0]["content"]
    assert tok.calls[0][0][1]["role"] == "user"
    assert "<img><image><image></img>" in tok.calls[0][0][1]["content"]
    assert tok.calls[0][1]["enable_thinking"] is False
    assert prompt.endswith("<think></think>")


def test_omni_prompt_builder_forwards_enable_thinking_false_on_followup_turn():
    tok = _FakeTokenizer()

    _build_omni_turn_prompt_with_thinking(
        tok,
        "Now answer directly.",
        is_first=False,
        enable_thinking=False,
    )

    assert tok.calls[0][1]["enable_thinking"] is False


def test_omni_prompt_builder_uses_image_placeholders_for_video_tokens():
    tok = _FakeTokenizer()

    _build_omni_turn_prompt_with_thinking(
        tok,
        "Describe the video.",
        n_video_tokens=3,
        is_first=True,
        enable_thinking=False,
    )

    content = tok.calls[0][0][1]["content"]
    assert "<img><image><image><image></img>" in content
    assert "<video>" not in content


def test_omni_dispatcher_sets_thinking_flag_on_first_session_turn(tmp_path):
    class _FakeSession:
        def __init__(self):
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

        def turn(self, **kwargs):
            return "direct answer"

    dispatcher = OmniMultimodalDispatcher.__new__(OmniMultimodalDispatcher)
    dispatcher.bundle_path = "/fake"
    dispatcher._session = _FakeSession()
    dispatcher._lock = __import__("threading").Lock()
    dispatcher._last_signature = None
    dispatcher._scratch_dir = tmp_path

    dispatcher.chat(
        [{"role": "user", "content": "Describe directly."}],
        enable_thinking=False,
    )

    assert dispatcher._session._vmlx_enable_thinking is False


def test_omni_extracts_input_audio_shape_emitted_by_panel(tmp_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe the sound."},
                {
                    "type": "input_audio",
                    "input_audio": {"data": "UklGRg==", "format": "wav"},
                },
            ],
        }
    ]

    assert request_has_multimodal(messages) is True
    assert request_modalities(messages) == {"audio"}
    text, images, audio, video = _extract_parts(messages, tmp_path)

    assert text == "Transcribe the sound."
    assert images == []
    assert video is None
    assert audio is not None
    assert audio.suffix == ".wav"
    assert audio.read_bytes() == b"RIFF"


def test_request_modalities_detects_image_audio_video():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Use all media."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}},
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
            ],
        }
    ]

    assert request_has_multimodal(messages) is True
    assert request_modalities(messages) == {"image", "audio", "video"}


@pytest.mark.asyncio
async def test_omni_dispatch_rejects_unsupported_video_before_session_load(tmp_path):
    bundle = _write_omni_bundle(tmp_path / "omni", video_preprocessor=False)
    OmniMultimodalDispatcher._instance = None

    class _Request:
        model = "omni"
        stream = False
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video."},
                    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
                ],
            }
        ]
        max_tokens = 8
        max_completion_tokens = None
        temperature = 0
        top_p = 1
        chat_template_kwargs = {}
        enable_thinking = False

    with pytest.raises(HTTPException) as exc:
        await dispatch_omni_chat_completion(_Request(), str(bundle))

    assert exc.value.status_code == 400
    assert "video" in exc.value.detail
    assert OmniMultimodalDispatcher._instance is None
