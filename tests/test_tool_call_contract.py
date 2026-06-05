from pathlib import Path
import json


def test_tool_call_contract_default_out_tracks_current_release_proof_artifact():
    from tests.cross_matrix import run_tool_call_contract as gate

    assert gate.DEFAULT_OUT == Path(
        "build/current-tool-call-contract-20260528-tool-parser-loop-matrix.json"
    )


def test_tool_call_contract_has_required_marker_gate():
    from tests.cross_matrix import run_tool_call_contract as gate

    required = gate.REQUIRED_TOOL_CALL_TEST_MARKERS
    assert "responses_extracts_suppressed_reasoning_tool_calls_before_finalize" in required
    assert "resets text-chat tool streaming state before chained follow-up requests" in required
    assert "panel max tool iterations caps tool loops" in required
    assert "TestGemma4ToolParser::test_tools_called_implies_no_marker_in_content" in required
    assert "TestHunyuanToolParser::test_visible_text_before_and_after_tool_calls_preserved" in required
    assert "TestXMLFunctionToolParser::test_visible_text_around_tool_call_has_no_xml_function_leak" in required
    assert "TestXMLFunctionToolParser::test_registry_aliases_resolve" in required
    assert "engine_family_tool_parser_matrix" in gate.COMMANDS


def test_tool_call_contract_fails_when_required_marker_missing(monkeypatch, tmp_path):
    from tests.cross_matrix import run_tool_call_contract as gate

    def fake_run(root, name, cwd_rel, cmd):
        return {
            "name": name,
            "command": cmd,
            "cwd": str(cwd_rel),
            "returncode": 0,
            "elapsed_sec": 0,
            "counts": {"passed": 999, "skipped": 0, "deselected": 0},
            "stdout": "some unrelated passing output",
            "stdout_tail": ["some unrelated passing output"],
        }

    monkeypatch.setattr(gate, "_run", fake_run)
    monkeypatch.setattr(gate, "REQUIRED_TOOL_CALL_TEST_MARKERS", ("missing marker",))

    artifact = gate.build_artifact(tmp_path)

    assert artifact["status"] == "fail"
    assert artifact["missing_markers"] == ["missing marker"]


def test_tool_call_contract_keeps_skipped_live_dsv4_default_cache_artifact_open(monkeypatch, tmp_path):
    from tests.cross_matrix import run_tool_call_contract as gate

    def fake_run(root, name, cwd_rel, cmd):
        stdout = "\n".join(gate.REQUIRED_TOOL_CALL_TEST_MARKERS)
        return {
            "name": name,
            "command": cmd,
            "cwd": str(cwd_rel),
            "returncode": 0,
            "elapsed_sec": 0,
            "counts": {"passed": 999, "skipped": 0, "deselected": 0},
            "stdout": stdout,
            "stdout_tail": stdout.splitlines(),
        }

    skipped_artifact = tmp_path / "build/current-dsv4-default-cache-tool-loop/result.json"
    skipped_artifact.parent.mkdir(parents=True)
    skipped_artifact.write_text(json.dumps({"status": "skipped"}), encoding="utf-8")
    monkeypatch.setattr(gate, "_run", fake_run)

    artifact = gate.build_artifact(tmp_path)

    assert artifact["status"] == "open"
    assert artifact["checks"]["live_default_cache_dsv4_tool_loop_artifact_passed"] is False
    assert artifact["open_proof_gaps"] == ["live_default_cache_dsv4_tool_loop"]


def test_tool_call_contract_accepts_review_artifact_when_dsv4_runtime_checks_pass(
    monkeypatch,
    tmp_path,
):
    from tests.cross_matrix import run_tool_call_contract as gate

    def fake_run(root, name, cwd_rel, cmd):
        stdout = "\n".join(gate.REQUIRED_TOOL_CALL_TEST_MARKERS)
        return {
            "name": name,
            "command": cmd,
            "cwd": str(cwd_rel),
            "returncode": 0,
            "elapsed_sec": 0,
            "counts": {"passed": 999, "skipped": 0, "deselected": 0},
            "stdout": stdout,
            "stdout_tail": stdout.splitlines(),
        }

    live_artifact = tmp_path / "build/current-dsv4-default-cache-tool-loop/result.json"
    live_artifact.parent.mkdir(parents=True)
    live_artifact.write_text(
        json.dumps(
            {
                "status": "review",
                "checks": {
                    "tool_sequence_ordered": True,
                    "final_done": True,
                    "file_written": True,
                    "code_file_written_exact": False,
                    "native_cache": True,
                    "native_prefix": True,
                    "native_paged": True,
                    "native_l2": True,
                    "generic_tq_kv_off": True,
                    "cached_tokens_seen": True,
                    "dsv4_cache_detail_seen": True,
                },
                "code_tool_probe": {
                    "exact": False,
                    "corrupt_identifier_patterns": ["THREE.BBoxGeometry"],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(gate, "_run", fake_run)

    artifact = gate.build_artifact(tmp_path)

    assert artifact["status"] == "pass"
    assert artifact["checks"]["live_default_cache_dsv4_tool_loop_artifact_passed"] is True
    assert artifact["open_proof_gaps"] == []
    assert artifact["live_default_cache_artifact_status"] == "review"
    assert artifact["live_default_cache_runtime_checks"]["code_file_written_exact"] is False
