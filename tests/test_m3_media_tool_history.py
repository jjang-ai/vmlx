"""Media template rendering retains the same tools and history as text."""
from types import SimpleNamespace

import numpy as np


def test_media_tool_history_renders_json_arguments_without_mutating_request(monkeypatch):
    from vmlx_engine.engine.batched import BatchedEngine
    from vmlx_engine.models.minimax_m3 import m3_vl_preprocess as media

    seen = {}

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            # The native template iterates arguments.items(); OpenAI sends JSON.
            args = messages[0]["tool_calls"][0]["function"]["arguments"]
            seen["arguments"] = dict(args.items())
            seen["tools"] = kwargs["tools"]
            seen["thinking"] = kwargs["thinking_mode"]
            return "rendered-media-history"

    class Processor:
        tokenizer = Tokenizer()

        def __call__(self, **kwargs):
            return {"input_ids": np.array([[1, 200025, 2]])}

    monkeypatch.setattr(media, "_get_processor", lambda _: Processor())
    monkeypatch.setattr(media, "_load_pil_images", lambda _: [object()])
    engine = object.__new__(BatchedEngine)
    engine._scheduler_config = SimpleNamespace(model_path="test-m3")
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function",
          "function": {"name": "sum_squares", "arguments": '{"n":137}'}}]},
        {"role": "tool", "content": "866525"},
        {"role": "user", "content": "Describe this image and recall the sum."},
    ]
    tools = [{"type": "function", "function": {"name": "sum_squares"}}]
    result = engine._m3_vl_preprocess(messages, ["image.png"], None,
                                    enable_thinking=False, tools=tools)
    assert result[0] == [1, 200025, 2]
    assert seen == {"arguments": {"n": 137}, "tools": tools, "thinking": "disabled"}
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"n":137}'
