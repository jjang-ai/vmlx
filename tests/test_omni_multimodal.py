from vmlx_engine.omni_multimodal import (
    OmniMultimodalDispatcher,
    _build_omni_turn_prompt_with_thinking,
    _extract_parts,
    request_has_multimodal,
)


class _FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        rail = "<think></think>" if kwargs.get("enable_thinking") is False else "<think>\n"
        return f"rendered:{messages[-1]['content']}:{rail}"


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
    text, images, audio, video = _extract_parts(messages, tmp_path)

    assert text == "Transcribe the sound."
    assert images == []
    assert video is None
    assert audio is not None
    assert audio.suffix == ".wav"
    assert audio.read_bytes() == b"RIFF"
