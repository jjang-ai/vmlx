from vmlx_engine.server import (
    _coerce_orphan_tool_messages_for_template,
    _coerce_zaya_vl_tool_history_for_template,
    _responses_input_to_messages,
)


def _text_from_parts(content):
    assert isinstance(content, list)
    return "\n".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


def test_responses_multimodal_parts_drop_nested_none_fields():
    messages = _responses_input_to_messages(
        [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "describe this",
                        "image_url": None,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,AAAA",
                            "detail": None,
                        },
                    },
                ],
            }
        ],
        preserve_multimodal=True,
    )

    content = messages[0]["content"]
    assert content[0] == {"type": "text", "text": "describe this"}
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAAA"},
    }


def test_orphan_tool_message_is_preserved_as_visible_text_for_templates():
    messages = _coerce_orphan_tool_messages_for_template(
        [
            {
                "role": "tool",
                "tool_call_id": "call_missing",
                "content": '{"stored":"blue-cat"}',
            }
        ]
    )

    assert messages == [
        {
            "role": "user",
            "content": 'Tool result (call_missing):\n{"stored":"blue-cat"}',
        }
    ]


def test_matching_tool_message_keeps_native_tool_role():
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "record_fact", "arguments": {"value": "blue-cat"}},
    }
    messages = _coerce_orphan_tool_messages_for_template(
        [
            {"role": "assistant", "content": "", "tool_calls": [tool_call]},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"stored":"blue-cat"}',
            },
        ]
    )

    assert messages[1]["role"] == "tool"
    assert messages[1]["tool_call_id"] == "call_1"


def test_zaya_vl_tool_history_is_text_only_and_merges_followup():
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "record_fact",
            "arguments": '{"value":"blue-cat"}',
        },
    }
    messages = _coerce_zaya_vl_tool_history_for_template(
        [
            {"role": "assistant", "content": "", "tool_calls": [tool_call]},
            {
                "role": "tool",
                "name": "record_fact",
                "tool_call_id": "call_1",
                "content": '{"stored":"blue-cat"}',
            },
            {"role": "user", "content": "what did I store?"},
        ]
    )

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    text = _text_from_parts(messages[0]["content"])
    assert "Previous assistant tool call:" in text
    assert "<function=record_fact>" in text
    assert "Tool response: STORED blue-cat" in text
    assert "what did I store?" in text
