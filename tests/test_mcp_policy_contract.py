def test_mcp_policy_contract_has_required_marker_gate():
    from tests.cross_matrix import run_mcp_policy_contract as gate

    required = gate.REQUIRED_MCP_POLICY_TEST_MARKERS
    assert "test_server_discovers_cwd_mcp_json_without_env_or_cli" in required
    assert "routes MCP list and execute calls by explicit model alias" in required
    assert "keeps built-in Electron tools separate from MCP execution and request policy" in required


def test_mcp_policy_contract_fails_when_required_marker_missing(monkeypatch, tmp_path):
    from tests.cross_matrix import run_mcp_policy_contract as gate

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
    monkeypatch.setattr(gate, "REQUIRED_MCP_POLICY_TEST_MARKERS", ("missing marker",))

    artifact = gate.build_artifact(tmp_path)

    assert artifact["status"] == "fail"
    assert artifact["missing_markers"] == ["missing marker"]
