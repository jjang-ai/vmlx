from types import SimpleNamespace

from tests.cross_matrix import run_n2_chat_cache_gate as runner


def test_n2_chat_cache_gate_builds_required_tool_payload():
    args = SimpleNamespace(
        served_model_name="n2-pro-jangtq2-chat-proof",
        tool_prompt="Use lookup for query alpha. Do not answer in prose.",
        tool_name="lookup",
        tool_query="alpha",
        max_tokens=16,
        tool_max_tokens=64,
    )

    payload = runner.tool_payload(args)

    assert payload["model"] == "n2-pro-jangtq2-chat-proof"
    assert payload["max_tokens"] == 64
    assert payload["messages"] == [
        {
            "role": "user",
            "content": "Use lookup for query alpha. Do not answer in prose.",
        }
    ]
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a short value by query.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]
    assert payload["tool_choice"] == {
        "type": "function",
        "function": {"name": "lookup"},
    }
    assert payload["enable_thinking"] is False


def test_n2_chat_cache_gate_extracts_tool_call_arguments():
    response = {
        "code": 200,
        "elapsed_s": 1.0,
        "error": None,
        "body": {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"query":"alpha"}',
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8},
        },
    }

    row = runner.row_from_response("tool_required", response)

    assert row["tool_calls"] == [
        {
            "id": "call_1",
            "name": "lookup",
            "arguments": '{"query":"alpha"}',
        }
    ]
    assert row["tool_call_names"] == ["lookup"]
    assert row["tool_call_arguments"] == [{"query": "alpha"}]


def test_n2_chat_cache_gate_checks_stable_text_only_on_cache_rows():
    rows = [
        {"mode": "no_cache_bypass", "text": "ACK"},
        {"mode": "cache_warm_store", "text": "ACK"},
        {"mode": "cache_hit", "text": "ACK"},
        {"mode": "tool_required", "text": ""},
    ]

    assert runner.cache_rows_stable_text(rows)


def test_n2_chat_cache_gate_builds_responses_tool_payload():
    args = SimpleNamespace(
        served_model_name="n2-pro-jangtq2-chat-proof",
        responses_tool_prompt="Use lookup for query alpha. Do not answer in prose.",
        tool_name="lookup",
        tool_query="alpha",
        responses_max_output_tokens=64,
    )

    payload = runner.responses_tool_payload(args)

    assert payload["model"] == "n2-pro-jangtq2-chat-proof"
    assert payload["input"] == "Use lookup for query alpha. Do not answer in prose."
    assert payload["store"] is True
    assert payload["stream"] is False
    assert payload["max_output_tokens"] == 64
    assert payload["enable_thinking"] is False
    assert payload["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "description": "Look up a short value by query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]
    assert payload["tool_choice"] == "required"


def test_n2_chat_cache_gate_extracts_responses_calls_and_text():
    response = {
        "id": "resp_1",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": '{"query":"alpha"}',
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "DONE"}],
            },
        ],
        "usage": {
            "input_tokens_details": {"cached_tokens": 12, "cache_detail": "paged+ssm"}
        },
    }

    assert runner.responses_function_calls(response) == [
        {
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{"query":"alpha"}',
        }
    ]
    assert runner.responses_output_text(response) == "DONE"
    assert runner.responses_cached_tokens(response) == 12


def test_n2_chat_cache_gate_builds_responses_tool_followup_payload():
    args = SimpleNamespace(
        served_model_name="n2-pro-jangtq2-chat-proof",
        responses_max_output_tokens=64,
    )
    tool_outputs = [
        {"type": "function_call_output", "call_id": "call_1", "output": "lookup=alpha-ok"}
    ]

    payload = runner.responses_tool_followup_payload(
        args,
        previous_response_id="resp_1",
        tool_outputs=tool_outputs,
    )

    assert payload["model"] == "n2-pro-jangtq2-chat-proof"
    assert payload["previous_response_id"] == "resp_1"
    assert payload["input"] == [
        *tool_outputs,
        {"role": "user", "content": "No more tools. Reply exactly DONE."},
    ]
    assert payload["tool_choice"] == "none"
