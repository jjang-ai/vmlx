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


def test_n2_chat_cache_gate_memory_preflight_labels_binary_units(monkeypatch):
    monkeypatch.setattr(
        runner,
        "resource_snapshot",
        lambda name: {
            "name": name,
            "system_memory": {
                "unit": "GiB",
                "available_gib": 94.25,
                "available_gb": 94.25,
            },
        },
    )

    skipped = runner.memory_preflight(120.0)

    assert skipped["status"] == "skipped"
    assert skipped["unit"] == "GiB"
    assert skipped["available_gib"] == 94.25
    assert skipped["required_available_gib"] == 120.0
    assert skipped["memory_gap_gib"] == 25.75
    assert skipped["available_gb"] == 94.25
    assert skipped["required_available_gb"] == 120.0


def test_n2_chat_cache_gate_records_requested_probes_for_skipped_runs():
    args = SimpleNamespace(
        include_tool_probe=True,
        include_responses_probe=True,
        include_responses_stream_probe=False,
        include_l2_restart_probe=True,
    )

    assert runner.requested_probes(args) == {
        "tool": True,
        "responses": True,
        "responses_stream": False,
        "l2_restart": True,
    }


def test_n2_chat_cache_gate_skips_jang1l_when_payload_exceeds_available_headroom(
    tmp_path, monkeypatch
):
    model = tmp_path / "Nex-N2-Pro-JANG_1L"
    model.mkdir()
    (model / "model.safetensors.index.json").write_text(
        '{"metadata":{"total_size":"118111600640"},"weight_map":{"a":"a.safetensors"}}'
    )
    out = tmp_path / "n2.json"
    popen_called = False

    def fake_popen(*_args, **_kwargs):
        nonlocal popen_called
        popen_called = True
        raise AssertionError("server should not launch")

    monkeypatch.setattr(
        runner,
        "resource_snapshot",
        lambda name, proc=None: {
            "name": name,
            "system_memory": {"unit": "GiB", "available_gib": 111.0},
        },
    )
    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    args = SimpleNamespace(
        model=model,
        served_model_name="n2-pro-jang1l-chat-proof",
        out=out,
        cache_dir=tmp_path / "cache",
        min_available_gb=24.0,
        port=8899,
        load_timeout_s=1,
        include_tool_probe=False,
        include_responses_probe=False,
        include_responses_stream_probe=False,
        include_l2_restart_probe=False,
    )

    result = runner.run(args)

    assert popen_called is False
    assert result["status"] == "skipped"
    assert result["reason"] == "n2_jang1l_insufficient_available_memory"
    assert result["indexed_payload_gib"] == 110.0
    assert result["required_available_gib"] == 118.0
    assert result["available_gib"] == 111.0
    assert result["memory_gap_gib"] == 7.0
    assert result["release_boundary"].startswith("N2 JANG_1L live proof was not launched")


def test_n2_chat_cache_gate_writes_failure_artifact_when_server_exits_before_health(
    tmp_path, monkeypatch
):
    model = tmp_path / "Nex-N2-Pro-JANG_1L"
    model.mkdir()
    out = tmp_path / "n2.json"

    class FakeProc:
        pid = 12345
        returncode = -6

        def poll(self):
            return -6

        def wait(self, timeout=None):
            return -6

    monkeypatch.setattr(runner, "memory_preflight", lambda _min_available_gb: None)
    monkeypatch.setattr(runner, "build_command", lambda _args: ["fake-server"])
    monkeypatch.setattr(runner, "build_env", lambda: {})
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: FakeProc())
    monkeypatch.setattr(
        runner,
        "wait_health",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("server exited early with code -6")),
    )
    monkeypatch.setattr(
        runner,
        "resource_snapshot",
        lambda name, proc=None: {"name": name, "system_memory": {"unit": "GiB"}},
    )
    args = SimpleNamespace(
        model=model,
        served_model_name="n2-pro-jang1l-chat-proof",
        out=out,
        cache_dir=tmp_path / "cache",
        min_available_gb=0.0,
        port=8899,
        load_timeout_s=1,
        include_tool_probe=False,
        include_responses_probe=False,
        include_responses_stream_probe=False,
        include_l2_restart_probe=False,
    )

    result = runner.run(args)

    assert result["status"] == "fail"
    assert result["phase"] == "server_startup"
    assert result["server_exit"] == -6
    assert "server exited early with code -6" in result["error"]
    assert result["server_log"].endswith("n2.server.log")


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


def test_n2_chat_cache_gate_builds_streaming_responses_tool_payload():
    args = SimpleNamespace(
        served_model_name="n2-pro-jangtq2-chat-proof",
        responses_tool_prompt="Use lookup for query alpha. Do not answer in prose.",
        tool_name="lookup",
        tool_query="alpha",
        responses_max_output_tokens=64,
    )

    payload = runner.responses_stream_tool_payload(args)

    assert payload["model"] == "n2-pro-jangtq2-chat-proof"
    assert payload["stream"] is True
    assert payload["store"] is True
    assert payload["tool_choice"] == "required"
    assert payload["enable_thinking"] is False


def test_n2_chat_cache_gate_extracts_streaming_responses_tool_args_from_sse():
    raw = "\n\n".join(
        [
            'event: response.output_item.added\ndata: {"type":"response.output_item.added","output_index":1,"item":{"id":"fc_1","type":"function_call","status":"in_progress","call_id":"call_1","name":"lookup","arguments":""}}',
            'event: response.function_call_arguments.delta\ndata: {"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":1,"delta":"{\\"query\\":"}',
            'event: response.function_call_arguments.delta\ndata: {"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":1,"delta":"\\"alpha\\"}"}',
            'event: response.function_call_arguments.done\ndata: {"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":1,"arguments":"{\\"query\\":\\"alpha\\"}"}',
            'event: response.output_item.done\ndata: {"type":"response.output_item.done","output_index":1,"item":{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_1","name":"lookup","arguments":"{\\"query\\":\\"alpha\\"}"}}',
            'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\\"query\\":\\"alpha\\"}"}],"usage":{"input_tokens_details":{"cached_tokens":9,"cache_detail":"paged+ssm"}}}}',
        ]
    )

    events = runner.parse_sse_events(raw)
    row = runner.responses_stream_row_from_sse(
        "responses_stream_tool_required",
        {"code": 200, "elapsed_s": 1.0, "error": None, "raw": raw, "events": events},
    )

    assert row["status_code"] == 200
    assert row["function_call_names"] == ["lookup"]
    assert row["function_call_arguments"] == [{"query": "alpha"}]
    assert row["argument_delta_text_by_item"] == {"fc_1": '{"query":"alpha"}'}
    assert row["argument_done_text_by_item"] == {"fc_1": '{"query":"alpha"}'}
    assert row["output_item_done_arguments_by_item"] == {"fc_1": '{"query":"alpha"}'}
    assert row["cached_tokens"] == 9
    assert row["cache_detail"] == "paged+ssm"


def test_n2_chat_cache_gate_l2_restart_probe_requires_block_and_ssm_disk_hits():
    probe = {
        "status": "pass",
        "row": {
            "status_code": 200,
            "usage": {
                "prompt_tokens_details": {
                    "cached_tokens": 56,
                    "cache_detail": "paged+ssm+disk",
                }
            },
        },
        "after_health_cache": {
            "block_disk_cache": {"disk_hits": 1},
            "ssm_companion_disk": {"hits": 1},
        },
    }

    assert runner.l2_restart_probe_passed(probe) is True

    probe["after_health_cache"]["ssm_companion_disk"]["hits"] = 0
    assert runner.l2_restart_probe_passed(probe) is False


def test_n2_chat_cache_gate_extracts_l2_restart_health_cache_stats():
    health = {
        "cache": {
            "block_disk_cache": {"disk_hits": 2, "disk_writes": 3},
            "ssm_companion": {"disk": {"hits": 1, "stores": 4}},
            "totals": {"l2_tokens_on_disk": 128},
        }
    }

    stats = runner._health_cache_stats(health)

    assert stats["block_disk_cache"] == {"disk_hits": 2, "disk_writes": 3}
    assert stats["ssm_companion_disk"] == {"hits": 1, "stores": 4}
    assert stats["totals"] == {"l2_tokens_on_disk": 128}
