"""MLLM tool replay contracts."""

import pytest

from vmlx_engine.engine.simple import SimpleEngine
from vmlx_engine.engine.base import GenerationOutput
from vmlx_engine.models.mllm import MLXMultimodalLM


def test_mllm_tool_replay_normalizes_tool_call_arguments_for_chat_template():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "call echo"}]},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "smoke__echo",
                        "arguments": '{"text": "qwen-chain-one"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "smoke__echo",
            "content": "qwen-chain-one",
        },
        {"role": "user", "content": [{"type": "text", "text": "now call add"}]},
    ]

    chat_messages, images, videos = MLXMultimodalLM._extract_multimodal_messages(messages)

    assert images == []
    assert videos == []
    assert chat_messages[1]["role"] == "assistant"
    assert chat_messages[1]["tool_calls"][0]["function"]["name"] == "smoke__echo"
    assert chat_messages[1]["tool_calls"][0]["function"]["arguments"] == {
        "text": "qwen-chain-one"
    }
    assert chat_messages[2] == {
        "role": "tool",
        "content": "qwen-chain-one",
        "tool_call_id": "call_1",
        "name": "smoke__echo",
    }


def test_mllm_chat_template_receives_effective_tool_schemas(monkeypatch):
    captured = {}

    def fake_get_chat_template(processor, messages, add_generation_prompt=True, **kwargs):
        captured["processor"] = processor
        captured["messages"] = messages
        captured["add_generation_prompt"] = add_generation_prompt
        captured["kwargs"] = kwargs
        return "prompt"

    import mlx_vlm.prompt_utils as prompt_utils

    monkeypatch.setattr(prompt_utils, "get_chat_template", fake_get_chat_template)
    model = object.__new__(MLXMultimodalLM)
    model.processor = object()
    model.config = {}
    model.model_name = "qwen-mllm-tool-test"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "smoke__echo",
                "description": "Echo text.",
                "parameters": {"type": "object"},
            },
        }
    ]

    prompt = model._apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": "use echo"}]}],
        enable_thinking=False,
        tools=tools,
    )

    assert prompt == "prompt"
    assert captured["kwargs"]["enable_thinking"] is False
    assert captured["kwargs"]["tools"] == tools


@pytest.mark.asyncio
async def test_simple_engine_forwards_effective_tools_to_mllm_chat():
    class FakeMllm:
        def __init__(self):
            self.kwargs = None

        def chat(self, **kwargs):
            self.kwargs = kwargs
            return GenerationOutput(
                text="<tool_call></tool_call>",
                prompt_tokens=1,
                completion_tokens=1,
                finish_reason="stop",
            )

    engine = SimpleEngine("qwen-mllm-tool-test")
    fake = FakeMllm()
    engine._is_mllm = True
    engine._loaded = True
    engine._model = fake

    tools = [
        {
            "type": "function",
            "function": {
                "name": "smoke__echo",
                "description": "Echo text.",
                "parameters": {"type": "object"},
            },
        }
    ]

    await engine.chat(
        messages=[{"role": "user", "content": [{"type": "text", "text": "use echo"}]}],
        tools=tools,
        max_tokens=4,
        temperature=0,
    )

    assert fake.kwargs is not None
    assert fake.kwargs["tools"][0]["function"]["name"] == "smoke__echo"
