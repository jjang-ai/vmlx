def test_tool_call_contract_has_required_marker_gate():
    from tests.cross_matrix import run_tool_call_contract as gate

    required = gate.REQUIRED_TOOL_CALL_TEST_MARKERS
    assert "responses_extracts_suppressed_reasoning_tool_calls_before_finalize" in required
    assert "resets text-chat tool streaming state before chained follow-up requests" in required
    assert "panel max tool iterations caps tool loops" in required


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
