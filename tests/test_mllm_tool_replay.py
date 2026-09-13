"""MLLM tool replay contracts."""

import copy
import json
from types import SimpleNamespace

import pytest

from vmlx_engine.engine.simple import SimpleEngine
from vmlx_engine.engine.base import GenerationOutput
from vmlx_engine.models.mllm import MLXMultimodalLM


@pytest.mark.parametrize("wire_json", [True, False])
@pytest.mark.parametrize("tool_content", [None, ""])
def test_batched_image_tool_history_normalizes_before_processor_without_mutation(
    caplog, wire_json, tool_content
):
    """Exercise the real batched renderer, not a copy of its image builder.

    Previously JSON-string arguments reached the image processor unchanged.
    Its mapping-only template failed; the text fallback mutated caller history
    and produced a different prompt/cache key on the next render.
    """
    from vmlx_engine.engine.batched import BatchedEngine

    class TextFallback:
        def apply_chat_template(self, messages, **kwargs):
            return "WRONG_TEXT_FALLBACK"

    class Processor:
        tokenizer = TextFallback()

        def apply_chat_template(self, messages, **kwargs):
            for message in messages:
                for call in message.get("tool_calls", []):
                    if not isinstance(call["function"]["arguments"], dict):
                        raise ValueError("Can only get item pairs from a mapping.")
            return json.dumps(messages, ensure_ascii=False)

    engine = object.__new__(BatchedEngine)
    engine._is_mllm = True
    engine._processor = Processor()
    engine._model = SimpleNamespace(config={"model_type": "qwen3_5"})
    engine._model_name = "local-template-fixture"
    engine._model_family_name = lambda: "qwen3_5"
    values = {
        "text": "  α\n",
        "literal": "\\n",
        "nested": {"widths": [64, 128], "enabled": True, "empty": None},
    }
    messages = [
        {"role": "user", "content": "read a fixture"},
        {
            "role": "assistant", "content": tool_content,
            "tool_calls": [{"id": "call_replay", "type": "function", "function": {
                "name": "read_file",
                "arguments": json.dumps(values) if wire_json else copy.deepcopy(values),
            }}],
        },
        {"role": "tool", "tool_call_id": "call_replay", "content": "actual fixture"},
        {"role": "user", "content": [
            {"type": "text", "text": "describe this badge"},
            {"type": "image_url", "image_url": {"url": "fixture.png"}},
        ]},
    ]
    original = copy.deepcopy(messages)
    prompt = engine._apply_chat_template(messages, num_images=1)
    assert "Failed to apply MLLM chat template" not in caplog.text
    built = json.loads(prompt)
    assert built[1]["tool_calls"][0]["function"]["arguments"] == values
    assert built[1]["content"] == ""
    assert built[2]["tool_call_id"] == "call_replay"
    assert messages == original, "template preparation mutated API-owned history"

    extended = messages + [
        {"role": "assistant", "content": "red badge"},
        {"role": "user", "content": "recall the fixture"},
    ]
    extended_original = copy.deepcopy(extended)
    replay = json.loads(engine._apply_chat_template(extended, num_images=1))
    assert replay[:len(built)] == built, "connected media prefix changed on replay"
    image_owners = [i for i, m in enumerate(replay)
                    if isinstance(m.get("content"), list)
                    and any(p.get("type") == "image" for p in m["content"])]
    assert image_owners == [3]
    assert extended == extended_original


@pytest.mark.parametrize("arguments", ["[]", "null", "3", '"text"', '{"broken":'])
def test_template_argument_normalization_does_not_fabricate_objects(arguments):
    from vmlx_engine.engine.batched import BatchedEngine

    messages = [{"role": "assistant", "content": "", "tool_calls": [{
        "id": "call_invalid", "type": "function",
        "function": {"name": "example", "arguments": arguments},
    }]}]
    original = copy.deepcopy(messages)
    normalized = BatchedEngine._normalize_tool_call_arguments_for_template(messages)
    assert normalized[0]["tool_calls"][0]["function"]["arguments"] == arguments
    assert messages == original


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

    chat_messages, images, videos, audio = MLXMultimodalLM._extract_multimodal_messages(messages)

    assert images == []
    assert videos == []
    assert audio == []
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
