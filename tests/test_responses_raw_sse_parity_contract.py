import json

from tests.cross_matrix.run_responses_raw_sse_parity_contract import (
    build_artifact,
    classify_server_log,
    classify_sse_capture,
)


def _sse(
    arguments: str = '{"query":"alpha"}',
    final_arguments: str = "",
    model: str = "same-model",
) -> str:
    return f"""event: response.created
data: {{"type":"response.created","response":{{"id":"resp_1","status":"in_progress","model":{json.dumps(model)},"output":[]}}}}

event: response.output_item.added
data: {{"type":"response.output_item.added","output_index":1,"item":{{"id":"rs_1","type":"reasoning","status":"in_progress","content":[]}}}}

event: response.reasoning_summary_text.delta
data: {{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":1,"delta":"checking"}}

event: response.reasoning_summary_text.done
data: {{"type":"response.reasoning_summary_text.done","item_id":"rs_1","output_index":1,"text":"checking"}}

event: response.output_item.done
data: {{"type":"response.output_item.done","output_index":1,"item":{{"id":"rs_1","type":"reasoning","status":"completed","content":[{{"type":"reasoning","text":"checking"}}]}}}}

event: response.output_item.added
data: {{"type":"response.output_item.added","output_index":2,"item":{{"id":"fc_1","type":"function_call","status":"in_progress","call_id":"call_1","name":"lookup","arguments":""}}}}

event: response.function_call_arguments.delta
data: {{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":2,"delta":{json.dumps(arguments)}}}

event: response.function_call_arguments.done
data: {{"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":2,"arguments":{json.dumps(arguments)}}}

event: response.output_item.done
data: {{"type":"response.output_item.done","output_index":2,"item":{{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_1","name":"lookup","arguments":{json.dumps(final_arguments)}}}}}

event: response.completed
data: {{"type":"response.completed","response":{{"status":"completed"}}}}

"""


def _sse_without_reasoning() -> str:
    return _sse().replace(
        """event: response.output_item.added
data: {"type":"response.output_item.added","output_index":1,"item":{"id":"rs_1","type":"reasoning","status":"in_progress","content":[]}}

event: response.reasoning_summary_text.delta
data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":1,"delta":"checking"}

event: response.reasoning_summary_text.done
data: {"type":"response.reasoning_summary_text.done","item_id":"rs_1","output_index":1,"text":"checking"}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":1,"item":{"id":"rs_1","type":"reasoning","status":"completed","content":[{"type":"reasoning","text":"checking"}]}}

""",
        "",
    )


def _sse_with_reasoning_events_but_no_reasoning_item() -> str:
    return """event: response.created
data: {"type":"response.created","response":{"id":"resp_1","status":"in_progress","model":"same-model","output":[]}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","status":"in_progress","role":"assistant","content":[]}}

event: response.reasoning_summary_text.delta
data: {"type":"response.reasoning_summary_text.delta","item_id":"msg_1","output_index":0,"delta":"checking"}

event: response.reasoning_summary_text.done
data: {"type":"response.reasoning_summary_text.done","item_id":"msg_1","output_index":0,"text":"checking"}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[]}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":1,"item":{"id":"fc_1","type":"function_call","status":"in_progress","call_id":"call_1","name":"lookup","arguments":""}}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":1,"delta":"{\\"query\\":\\"alpha\\"}"}

event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":1,"arguments":"{\\"query\\":\\"alpha\\"}"}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":1,"item":{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_1","name":"lookup","arguments":"{\\"query\\":\\"alpha\\"}"}}

event: response.completed
data: {"type":"response.completed","response":{"status":"completed","model":"same-model","output":[{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[]},{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_1","name":"lookup","arguments":"{\\"query\\":\\"alpha\\"}"},{"id":"rs_1","type":"reasoning","status":"completed","content":[{"type":"reasoning","text":"checking"}]}]}}

"""


def _interleaved_reasoning_content_tool_sse() -> str:
    return """event: response.created
data: {"type":"response.created","response":{"id":"resp_1","status":"in_progress","model":"same-model","output":[]}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","status":"in_progress","role":"assistant","content":[]}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"delta":"visible "}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":1,"item":{"id":"rs_1","type":"reasoning","status":"in_progress","summary":[]}}

event: response.reasoning_summary_text.delta
data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":1,"delta":"checking"}

event: response.output_text.delta
data: {"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"delta":"content"}

event: response.reasoning_summary_text.done
data: {"type":"response.reasoning_summary_text.done","item_id":"rs_1","output_index":1,"text":"checking"}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":1,"item":{"id":"rs_1","type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":"checking"}]}}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"visible content"}]}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":2,"item":{"id":"fc_1","type":"function_call","status":"in_progress","call_id":"call_1","name":"lookup","arguments":""}}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":2,"delta":"{\\"query\\":\\"alpha\\"}"}

event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":2,"arguments":"{\\"query\\":\\"alpha\\"}"}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":2,"item":{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_1","name":"lookup","arguments":"{\\"query\\":\\"alpha\\"}"}}

event: response.completed
data: {"type":"response.completed","response":{"status":"completed","model":"same-model","output":[{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"visible content"}]},{"id":"rs_1","type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":"checking"}]},{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_1","name":"lookup","arguments":"{\\"query\\":\\"alpha\\"}"}]}}

"""


def _sse_with_message_and_duplicate_function_index() -> str:
    return """event: response.created
data: {"type":"response.created","response":{"id":"resp_1","status":"in_progress","model":"same-model","output":[]}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","status":"in_progress","role":"assistant","content":[]}}

event: response.reasoning_summary_text.delta
data: {"type":"response.reasoning_summary_text.delta","item_id":"msg_1","output_index":0,"delta":"checking"}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[]}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"item":{"id":"fc_1","type":"function_call","status":"in_progress","call_id":"call_1","name":"lookup","arguments":""}}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"delta":"{\\"query\\":\\"alpha\\"}"}

event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":0,"arguments":"{\\"query\\":\\"alpha\\"}"}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":0,"item":{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_1","name":"lookup","arguments":"{\\"query\\":\\"alpha\\"}"}}

"""


def test_classifier_uses_done_arguments_when_final_item_arguments_are_empty():
    row = classify_sse_capture(_sse())

    assert row["reasoning_events"] == 2
    assert row["done_arguments"] == '{"query":"alpha"}'
    assert row["final_item_arguments"] == ""
    assert row["authoritative_arguments"] == '{"query":"alpha"}'
    assert row["empty_final_item_with_done_args"] is True
    assert row["output_item_indices_valid"] is True
    assert row["model"] == "same-model"


def test_classifier_flags_function_call_reusing_message_output_index():
    row = classify_sse_capture(_sse_with_message_and_duplicate_function_index())

    assert row["authoritative_arguments"] == '{"query":"alpha"}'
    assert row["output_indices_by_type"]["message"] == [0]
    assert row["output_indices_by_type"]["function_call"] == [0]
    assert row["conflicting_output_indices"] == [0]
    assert row["output_item_indices_valid"] is False


def test_classifier_tracks_interleaved_reasoning_content_tool_and_final_output():
    row = classify_sse_capture(_interleaved_reasoning_content_tool_sse())

    assert row["content_delta_count"] == 2
    assert row["reconstructed_content"] == "visible content"
    assert row["reasoning_output_item_count"] == 1
    assert row["reasoning_done_count"] == 1
    assert row["reasoning_lifecycle_complete"] is True
    assert row["final_response_output_types"] == [
        "message",
        "reasoning",
        "function_call",
    ]
    assert row["final_response_function_arguments"] == '{"query":"alpha"}'
    assert row["final_response_message_text"] == "visible content"
    assert row["final_response_consistent_with_stream"] is True
    assert row["output_indices_by_type"]["message"] == [0]
    assert row["output_indices_by_type"]["reasoning"] == [1]
    assert row["output_indices_by_type"]["function_call"] == [2]


def test_classifier_rejects_reasoning_events_without_reasoning_output_item():
    row = classify_sse_capture(_sse_with_reasoning_events_but_no_reasoning_item())

    assert row["reasoning_events"] == 2
    assert row["reasoning_output_item_count"] == 0
    assert row["reasoning_lifecycle_complete"] is False


def test_classifier_records_raw_json_model_not_found_capture():
    row = classify_sse_capture(
        json.dumps(
            {
                "error": {
                    "message": "Model 'gemma4-e2b-sse' not found.",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            }
        )
    )

    assert row["event_count"] == 0
    assert row["raw_json_error"] is True
    assert row["error_code"] == "model_not_found"
    assert row["available_models"] == []
    assert row["has_authoritative_arguments"] is False


def test_classifier_records_gateway_request_kwargs_from_structured_log():
    row = classify_server_log(
        json.dumps(
            {
                "containsReasoning": True,
                "requestKwargs": {
                    "stream": True,
                    "max_output_tokens": 512,
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": 0,
                    "enable_thinking": True,
                    "tool_choice": "required",
                    "tool_count": 1,
                    "first_tool_name": "record_fact",
                },
            }
        )
    )

    assert row["no_reasoning_disable_workaround"] is True
    assert row["gateway_expected_request_kwargs_present"] is True
    assert row["request_kwargs"]["enable_thinking"] is True
    assert row["request_kwargs"]["tool_choice"] == "required"


def test_classifier_extracts_available_models_from_model_not_found_capture():
    row = classify_sse_capture(
        json.dumps(
            {
                "error": {
                    "message": (
                        "Model 'gemma4-e2b-sse' not found. Available: "
                        "[models/Qwen3.6-27B-MXFP8-CRACK-MTP, "
                        "Gemma-4-12B-it-MXFP8-CRACK, "
                        "models/Gemma-4-12B-it-MXFP8-CRACK]"
                    ),
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            }
        )
    )

    assert row["error_code"] == "model_not_found"
    assert row["available_models"] == [
        "models/Qwen3.6-27B-MXFP8-CRACK-MTP",
        "Gemma-4-12B-it-MXFP8-CRACK",
        "models/Gemma-4-12B-it-MXFP8-CRACK",
    ]


def test_raw_sse_parity_stays_open_when_tunnel_capture_is_missing(tmp_path):
    direct = tmp_path / "direct.sse"
    gateway = tmp_path / "gateway.sse"
    direct.write_text(_sse())
    gateway.write_text(_sse())

    artifact = build_artifact(
        direct_sse=direct,
        gateway_sse=gateway,
        tunnel_sse=None,
    )

    assert artifact["status"] == "open"
    assert artifact["missing_captures"] == ["tunnel"]
    assert artifact["checks"]["direct_capture_present"] is True
    assert artifact["checks"]["gateway_capture_present"] is True
    assert artifact["checks"]["tunnel_capture_present"] is False
    assert (
        artifact["checks"]["authoritative_arguments_match_across_present_surfaces"]
        is True
    )


def test_raw_sse_parity_fails_when_gateway_loses_arguments(tmp_path):
    direct = tmp_path / "direct.sse"
    gateway = tmp_path / "gateway.sse"
    tunnel = tmp_path / "tunnel.sse"
    direct.write_text(_sse())
    gateway.write_text(_sse(arguments="", final_arguments=""))
    tunnel.write_text(_sse())

    artifact = build_artifact(
        direct_sse=direct,
        gateway_sse=gateway,
        tunnel_sse=tunnel,
    )

    assert artifact["status"] == "fail"
    assert (
        artifact["checks"]["authoritative_arguments_match_across_present_surfaces"]
        is False
    )


def test_raw_sse_parity_fails_when_capture_arguments_do_not_match_expected(tmp_path):
    direct = tmp_path / "direct.sse"
    gateway = tmp_path / "gateway.sse"
    tunnel = tmp_path / "tunnel.sse"
    for path in (direct, gateway, tunnel):
        path.write_text(_sse(arguments='{"query":"wrong"}'))

    artifact = build_artifact(
        direct_sse=direct,
        gateway_sse=gateway,
        tunnel_sse=tunnel,
        expected_function_name="lookup",
        expected_arguments='{"query":"alpha"}',
    )

    assert artifact["status"] == "fail"
    assert artifact["checks"]["all_present_surfaces_match_expected_arguments"] is False
    assert artifact["captures"]["direct"]["expected_arguments_match"] is False


def test_raw_sse_parity_accepts_json_argument_formatting_differences(tmp_path):
    direct = tmp_path / "direct.sse"
    gateway = tmp_path / "gateway.sse"
    tunnel = tmp_path / "tunnel.sse"
    direct.write_text(_sse(arguments='{"query": "alpha"}'))
    gateway.write_text(_sse(arguments='{"query":"alpha"}'))
    tunnel.write_text(_sse(arguments='{\n  "query": "alpha"\n}'))

    artifact = build_artifact(
        direct_sse=direct,
        gateway_sse=gateway,
        tunnel_sse=tunnel,
        expected_function_name="lookup",
        expected_arguments='{"query":"alpha"}',
        require_reasoning_events=True,
    )

    assert artifact["status"] == "pass"
    assert (
        artifact["checks"]["authoritative_arguments_match_across_present_surfaces"]
        is True
    )
    assert artifact["checks"]["all_present_surfaces_match_expected_arguments"] is True
    assert artifact["captures"]["direct"]["authoritative_arguments"] == '{"query": "alpha"}'


def test_raw_sse_parity_still_fails_on_json_argument_value_mismatch(tmp_path):
    direct = tmp_path / "direct.sse"
    gateway = tmp_path / "gateway.sse"
    tunnel = tmp_path / "tunnel.sse"
    direct.write_text(_sse(arguments='{"query":"alpha"}'))
    gateway.write_text(_sse(arguments='{"query":"alpha"}'))
    tunnel.write_text(_sse(arguments='{"query":"beta"}'))

    artifact = build_artifact(
        direct_sse=direct,
        gateway_sse=gateway,
        tunnel_sse=tunnel,
        expected_function_name="lookup",
        expected_arguments='{"query":"alpha"}',
        require_reasoning_events=True,
    )

    assert artifact["status"] == "fail"
    assert (
        artifact["checks"]["authoritative_arguments_match_across_present_surfaces"]
        is False
    )
    assert artifact["checks"]["all_present_surfaces_match_expected_arguments"] is False
    assert artifact["captures"]["tunnel"]["expected_arguments_match"] is False


def test_raw_sse_parity_can_require_reasoning_events(tmp_path):
    direct = tmp_path / "direct.sse"
    gateway = tmp_path / "gateway.sse"
    tunnel = tmp_path / "tunnel.sse"
    no_reasoning = _sse_without_reasoning()
    for path in (direct, gateway, tunnel):
        path.write_text(no_reasoning)

    artifact = build_artifact(
        direct_sse=direct,
        gateway_sse=gateway,
        tunnel_sse=tunnel,
        expected_function_name="lookup",
        expected_arguments='{"query":"alpha"}',
        require_reasoning_events=True,
    )

    assert artifact["status"] == "fail"
    assert artifact["checks"]["all_present_surfaces_have_required_reasoning"] is False
    assert artifact["checks"]["no_reasoning_disable_workaround"] is False
    assert artifact["captures"]["direct"]["has_required_reasoning_events"] is False


def test_raw_sse_parity_distinguishes_enabled_reasoning_from_missing_events(tmp_path):
    direct = tmp_path / "direct.sse"
    gateway = tmp_path / "gateway.sse"
    tunnel = tmp_path / "tunnel.sse"
    direct_log = tmp_path / "direct.log"
    gateway_log = tmp_path / "gateway.log"
    tunnel_log = tmp_path / "tunnel.log"
    no_reasoning = _sse_without_reasoning()
    for path in (direct, gateway, tunnel):
        path.write_text(no_reasoning)
    for path in (direct_log, gateway_log, tunnel_log):
        path.write_text(
            "Reasoning: ENABLED (parser: Gemma4ReasoningParser)\n"
            "Resolved sampling kwargs route=/v1/responses "
            "kwargs={'enable_thinking': True, 'max_tokens': 512}\n"
        )

    artifact = build_artifact(
        direct_sse=direct,
        gateway_sse=gateway,
        tunnel_sse=tunnel,
        direct_log=direct_log,
        gateway_log=gateway_log,
        tunnel_log=tunnel_log,
        expected_function_name="lookup",
        expected_arguments='{"query":"alpha"}',
        require_reasoning_events=True,
    )

    assert artifact["status"] == "fail"
    assert artifact["checks"]["all_present_surfaces_have_required_reasoning"] is False
    assert artifact["checks"]["no_reasoning_disable_workaround"] is True
    assert artifact["captures"]["direct"]["reasoning_enabled_by_server_log"] is True
    assert artifact["captures"]["direct"]["enable_thinking_resolved_true"] is True


def test_raw_sse_parity_accepts_structured_gateway_reasoning_log():
    row = classify_server_log(
        json.dumps(
            {
                "status": 200,
                "containsReasoning": True,
                "containsFunctionDelta": True,
                "containsFunctionDone": True,
            }
        )
    )

    assert row["reasoning_enabled_by_server_log"] is True
    assert row["reasoning_disabled_by_server_log"] is False
    assert row["enable_thinking_resolved_true"] is True
    assert row["no_reasoning_disable_workaround"] is True


def test_raw_sse_parity_fails_when_surface_reuses_message_output_index_for_tool(
    tmp_path,
):
    direct = tmp_path / "direct.sse"
    gateway = tmp_path / "gateway.sse"
    tunnel = tmp_path / "tunnel.sse"
    direct.write_text(_sse())
    gateway.write_text(_sse())
    tunnel.write_text(_sse_with_message_and_duplicate_function_index())

    artifact = build_artifact(
        direct_sse=direct,
        gateway_sse=gateway,
        tunnel_sse=tunnel,
        expected_function_name="lookup",
        expected_arguments='{"query":"alpha"}',
        require_reasoning_events=True,
    )

    assert artifact["status"] == "fail"
    assert (
        artifact["checks"]["all_present_surfaces_have_valid_output_item_indices"]
        is False
    )
    assert artifact["captures"]["tunnel"]["conflicting_output_indices"] == [0]


def test_raw_sse_parity_reports_tunnel_expected_model_not_advertised(tmp_path):
    direct = tmp_path / "direct.sse"
    gateway = tmp_path / "gateway.sse"
    tunnel = tmp_path / "tunnel.sse"
    direct.write_text(_sse(model="gemma4-e2b-sse"))
    gateway.write_text(_sse(model="gemma4-e2b-sse"))
    tunnel.write_text(
        json.dumps(
            {
                "error": {
                    "message": (
                        "Model 'gemma4-e2b-sse' not found. Available: "
                        "[Gemma-4-12B-it-MXFP8-CRACK, models/Qwen3.6-35B]"
                    ),
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            }
        )
    )

    artifact = build_artifact(
        direct_sse=direct,
        gateway_sse=gateway,
        tunnel_sse=tunnel,
        expected_model="gemma4-e2b-sse",
        require_same_model=True,
    )

    assert artifact["status"] == "fail"
    assert artifact["captures"]["tunnel"]["available_models"] == [
        "Gemma-4-12B-it-MXFP8-CRACK",
        "models/Qwen3.6-35B",
    ]
    assert artifact["captures"]["tunnel"]["expected_model_advertised"] is False
    assert artifact["checks"]["tunnel_expected_model_advertised"] is False


def test_raw_sse_parity_consumes_local_responses_streaming_contract(tmp_path):
    direct = tmp_path / "direct.sse"
    gateway = tmp_path / "gateway.sse"
    tunnel = tmp_path / "tunnel.sse"
    noheavy = tmp_path / "noheavy.json"
    for path in (direct, gateway, tunnel):
        path.write_text(_sse())
    noheavy.write_text(
        json.dumps(
            {
                "status": "pass",
                "missing_markers": [],
                "checks": {
                    "responses_streaming_tool_call_arguments_and_indexes": True,
                    "responses_previous_response_history": True,
                    "gateway_responses_function_call_arguments_streaming": True,
                    "gateway_responses_reasoning_empty_final_arguments_streaming": True,
                    "gateway_stale_responses_port_rejection": True,
                },
            }
        )
    )

    artifact = build_artifact(
        direct_sse=direct,
        gateway_sse=gateway,
        tunnel_sse=tunnel,
        expected_function_name="lookup",
        expected_arguments='{"query":"alpha"}',
        noheavy_contract=noheavy,
        require_reasoning_events=True,
    )

    assert artifact["local_source_contract"]["path"] == str(noheavy)
    assert artifact["local_source_contract"]["local_responses_streaming_guards_pass"] is True
    assert artifact["checks"]["local_responses_streaming_guards_pass"] is True
    assert artifact["checks"]["local_empty_xml_arguments_fail_closed"] is True
    assert artifact["checks"]["local_output_index_ordering_guard"] is True
    assert artifact["checks"]["gateway_argument_stream_passthrough_guard"] is True
    assert artifact["checks"]["responses_previous_response_history_guard"] is True


def test_raw_sse_parity_requires_previous_response_history_source_guard(tmp_path):
    direct = tmp_path / "direct.sse"
    gateway = tmp_path / "gateway.sse"
    tunnel = tmp_path / "tunnel.sse"
    noheavy = tmp_path / "noheavy.json"
    for path in (direct, gateway, tunnel):
        path.write_text(_sse())
    noheavy.write_text(
        json.dumps(
            {
                "status": "pass",
                "missing_markers": [],
                "checks": {
                    "responses_streaming_tool_call_arguments_and_indexes": True,
                    "responses_previous_response_history": False,
                    "gateway_responses_function_call_arguments_streaming": True,
                    "gateway_responses_reasoning_empty_final_arguments_streaming": True,
                    "gateway_stale_responses_port_rejection": True,
                },
            }
        )
    )

    artifact = build_artifact(
        direct_sse=direct,
        gateway_sse=gateway,
        tunnel_sse=tunnel,
        expected_function_name="lookup",
        expected_arguments='{"query":"alpha"}',
        noheavy_contract=noheavy,
        require_reasoning_events=True,
    )

    assert artifact["local_source_contract"][
        "responses_previous_response_history_guard"
    ] is False
    assert artifact["local_source_contract"]["local_responses_streaming_guards_pass"] is False
    assert artifact["checks"]["responses_previous_response_history_guard"] is False
    assert artifact["checks"]["local_responses_streaming_guards_pass"] is False


def test_raw_sse_parity_fails_when_same_model_is_required_and_surfaces_differ(
    tmp_path,
):
    direct = tmp_path / "direct.sse"
    gateway = tmp_path / "gateway.sse"
    tunnel = tmp_path / "tunnel.sse"
    direct.write_text(_sse(model="gemma4-e2b-sse"))
    gateway.write_text(_sse(model="gemma4-e2b-sse"))
    tunnel.write_text(_sse(model="models/Qwen3.6-35B-A3B-MXFP8-CRACK-MTP"))

    artifact = build_artifact(
        direct_sse=direct,
        gateway_sse=gateway,
        tunnel_sse=tunnel,
        expected_function_name="lookup",
        expected_arguments='{"query":"alpha"}',
        require_reasoning_events=True,
        require_same_model=True,
    )

    assert artifact["status"] == "fail"
    assert artifact["checks"]["all_present_surfaces_report_model"] is True
    assert artifact["checks"]["all_present_surfaces_same_model"] is False
    assert artifact["captures"]["direct"]["model"] == "gemma4-e2b-sse"
    assert (
        artifact["captures"]["tunnel"]["model"]
        == "models/Qwen3.6-35B-A3B-MXFP8-CRACK-MTP"
    )


def test_raw_sse_parity_passes_when_all_surfaces_match(tmp_path):
    direct = tmp_path / "direct.sse"
    gateway = tmp_path / "gateway.sse"
    tunnel = tmp_path / "tunnel.sse"
    for path in (direct, gateway, tunnel):
        path.write_text(_sse())

    artifact = build_artifact(
        direct_sse=direct,
        gateway_sse=gateway,
        tunnel_sse=tunnel,
        expected_function_name="lookup",
        expected_arguments='{"query":"alpha"}',
        require_reasoning_events=True,
        require_same_model=True,
    )

    assert artifact["status"] == "pass"
    assert artifact["checks"]["all_required_surfaces_present"] is True
    assert artifact["checks"]["all_present_surfaces_same_model"] is True
    assert artifact["checks"]["all_present_surfaces_match_expected_arguments"] is True
    assert artifact["checks"]["all_present_surfaces_have_required_reasoning"] is True
    assert artifact["checks"]["no_reasoning_disable_workaround"] is True
