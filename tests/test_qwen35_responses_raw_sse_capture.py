from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tests.cross_matrix import run_qwen35_responses_raw_sse_capture as runner


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        python=Path(".venv/bin/python"),
        model=Path("/models/qwen35"),
        port=8898,
        served_model_name="models/Qwen3.6-35B-A3B-MXFP8-CRACK-MTP",
        ssm_state_cache_mb=8192,
        cache_dir=Path("/tmp/cache"),
        block_disk_cache_max_gb=4.0,
        max_output_tokens=512,
    )


def test_qwen35_raw_sse_capture_command_uses_current_mllm_path() -> None:
    cmd = runner.build_command(_args())

    assert "--is-mllm" in cmd
    assert "--mllm" not in cmd
    assert "--enable-block-disk-cache" in cmd
    assert "--use-paged-cache" in cmd
    assert "models/Qwen3.6-35B-A3B-MXFP8-CRACK-MTP" in cmd


def test_qwen35_raw_sse_capture_payload_requires_reasoning_tool_stream() -> None:
    payload = runner.responses_tool_stream_payload(_args())

    assert payload["model"] == "models/Qwen3.6-35B-A3B-MXFP8-CRACK-MTP"
    assert payload["stream"] is True
    assert payload["enable_thinking"] is True
    assert payload["tool_choice"] == "required"
    assert payload["tools"][0]["name"] == "record_fact"
    assert payload["tools"][0]["parameters"]["required"] == ["value"]
    assert payload["max_output_tokens"] == 512


def test_qwen35_raw_sse_capture_classifier_flags_duplicate_tunnel_index(tmp_path) -> None:
    direct = tmp_path / "direct.sse"
    tunnel = tmp_path / "tunnel.sse"
    direct.write_text(
        """event: response.created
data: {"type":"response.created","response":{"model":"models/Qwen3.6-35B-A3B-MXFP8-CRACK-MTP"}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","status":"in_progress","role":"assistant","content":[]}}

event: response.reasoning_summary_text.delta
data: {"type":"response.reasoning_summary_text.delta","item_id":"msg_1","output_index":0,"delta":"thinking"}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[]}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":1,"item":{"id":"fc_1","type":"function_call","status":"in_progress","call_id":"call_1","name":"record_fact","arguments":""}}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":1,"delta":"{\\"value\\": \\"blue-cat\\"}"}

event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":1,"arguments":"{\\"value\\": \\"blue-cat\\"}"}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":1,"item":{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_1","name":"record_fact","arguments":"{\\"value\\": \\"blue-cat\\"}"}}
"""
    )
    tunnel.write_text(direct.read_text().replace('"output_index":1', '"output_index":0'))

    artifact = runner._build_parity_artifact(
        direct_sse=direct,
        gateway_sse=None,
        tunnel_sse=tunnel,
        expected_function_name="record_fact",
        expected_arguments=runner.EXPECTED_ARGUMENTS,
        expected_model="models/Qwen3.6-35B-A3B-MXFP8-CRACK-MTP",
        require_reasoning_events=True,
        require_same_model=True,
    )

    assert artifact["status"] == "fail"
    assert artifact["captures"]["direct"]["output_item_indices_valid"] is True
    assert artifact["captures"]["tunnel"]["conflicting_output_indices"] == [0]
