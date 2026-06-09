import json

from tests.cross_matrix.run_responses_raw_sse_parity_contract import (
    build_artifact,
    classify_sse_capture,
)


def _sse(arguments: str = '{"query":"alpha"}', final_arguments: str = "") -> str:
    return f"""event: response.reasoning_summary_text.delta
data: {{"type":"response.reasoning_summary_text.delta","delta":"checking"}}

event: response.output_item.added
data: {{"type":"response.output_item.added","output_index":1,"item":{{"id":"fc_1","type":"function_call","status":"in_progress","call_id":"call_1","name":"lookup","arguments":""}}}}

event: response.function_call_arguments.delta
data: {{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":1,"delta":{json.dumps(arguments)}}}

event: response.function_call_arguments.done
data: {{"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":1,"arguments":{json.dumps(arguments)}}}

event: response.output_item.done
data: {{"type":"response.output_item.done","output_index":1,"item":{{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_1","name":"lookup","arguments":{json.dumps(final_arguments)}}}}}

event: response.completed
data: {{"type":"response.completed","response":{{"status":"completed"}}}}

"""


def test_classifier_uses_done_arguments_when_final_item_arguments_are_empty():
    row = classify_sse_capture(_sse())

    assert row["reasoning_events"] == 1
    assert row["done_arguments"] == '{"query":"alpha"}'
    assert row["final_item_arguments"] == ""
    assert row["authoritative_arguments"] == '{"query":"alpha"}'
    assert row["empty_final_item_with_done_args"] is True


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


def test_raw_sse_parity_can_require_reasoning_events(tmp_path):
    direct = tmp_path / "direct.sse"
    gateway = tmp_path / "gateway.sse"
    tunnel = tmp_path / "tunnel.sse"
    no_reasoning = _sse().replace(
        """event: response.reasoning_summary_text.delta
data: {"type":"response.reasoning_summary_text.delta","delta":"checking"}

""",
        "",
    )
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
    assert artifact["captures"]["direct"]["has_required_reasoning_events"] is False


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
    )

    assert artifact["status"] == "pass"
    assert artifact["checks"]["all_required_surfaces_present"] is True
    assert artifact["checks"]["all_present_surfaces_match_expected_arguments"] is True
    assert artifact["checks"]["all_present_surfaces_have_required_reasoning"] is True
