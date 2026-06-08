from types import SimpleNamespace


def test_mimo_v2_audio_content_parts_render_audio_special_tokens():
    from vmlx_engine.models.mllm import MLXMultimodalLM

    class Tokenizer:
        def convert_ids_to_tokens(self, token_id):
            return {
                151669: "<|audio_pad|>",
                151673: "<|mimo_audio_start|>",
                151674: "<|mimo_audio_end|>",
            }[token_id]

    model = object.__new__(MLXMultimodalLM)
    model.processor = SimpleNamespace(tokenizer=Tokenizer())
    model.config = {
        "model_type": "mimo_v2",
        "processor_config": {
            "audio_token_id": 151669,
            "audio_start_token_id": 151673,
            "audio_end_token_id": 151674,
        },
    }

    normalized = model._normalize_mimo_audio_messages_for_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe.", "content": "Transcribe."},
                    {"type": "audio"},
                ],
            }
        ]
    )

    assert normalized[0]["content"][0]["text"] == "Transcribe."
    assert normalized[0]["content"][1] == {
        "type": "text",
        "text": "<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>",
        "content": "<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>",
    }


def test_mimo_v2_audio_content_parts_leave_other_families_native():
    from vmlx_engine.models.mllm import MLXMultimodalLM

    model = object.__new__(MLXMultimodalLM)
    model.processor = SimpleNamespace()
    model.config = {"model_type": "qwen2_5_vl"}
    messages = [{"role": "user", "content": [{"type": "audio"}]}]

    assert model._normalize_mimo_audio_messages_for_template(messages) is messages


def test_batched_engine_mimo_audio_only_uses_processor_audio_tokens():
    from vmlx_engine.engine.batched import BatchedEngine

    captured = {}

    class Tokenizer:
        def convert_ids_to_tokens(self, token_id):
            return {
                151669: "<|audio_pad|>",
                151673: "<|mimo_audio_start|>",
                151674: "<|mimo_audio_end|>",
            }[token_id]

    class Processor:
        tokenizer = Tokenizer()

        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            return "rendered " + " ".join(
                part.get("text", "")
                for part in messages[0]["content"]
                if isinstance(part, dict)
            )

    engine = object.__new__(BatchedEngine)
    engine._is_mllm = True
    engine._processor = Processor()
    engine._tokenizer = None
    engine._model_name = "MiMo-V2.5-JANGTQ_2"
    engine._model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="mimo_v2",
            processor_config={
                "audio_token_id": 151669,
                "audio_start_token_id": 151673,
                "audio_end_token_id": 151674,
            },
        )
    )

    prompt = engine._apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe."},
                    {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}},
                ],
            }
        ],
        enable_thinking=False,
    )

    assert "<|audio_pad|>" in prompt
    assert captured["messages"][0]["content"][1]["text"] == (
        "<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>"
    )
