# SPDX-License-Identifier: Apache-2.0

import json

from tests.cross_matrix import run_mimo_v2_source_vs_quant_first_divergence as probe


def test_first_divergence_reports_first_changed_suffix():
    result = probe.first_divergence("ACK", "AXK")
    assert result == {
        "char_index": 1,
        "source_suffix": "CK",
        "quant_suffix": "XK",
    }
    assert probe.first_divergence("ACK", "ACK") is None


def test_classify_row_distinguishes_source_and_quant_failure_modes():
    assert (
        probe.classify_row(
            source_output="ACK",
            quant_output="ACK",
            expected="ACK",
            source_status=200,
            quant_status=200,
        )
        == "source_and_quant_match"
    )
    assert (
        probe.classify_row(
            source_output="ACK",
            quant_output="",
            expected="ACK",
            source_status=200,
            quant_status=200,
        )
        == "quant_diverges_from_source"
    )
    assert (
        probe.classify_row(
            source_output="",
            quant_output="",
            expected="ACK",
            source_status=200,
            quant_status=200,
        )
        == "source_also_fails"
    )
    assert (
        probe.classify_row(
            source_output="NOPE",
            quant_output="",
            expected="ACK",
            source_status=200,
            quant_status=200,
        )
        == "source_also_fails"
    )
    assert (
        probe.classify_row(
            source_output="ACK",
            quant_output="ACK",
            expected="ACK",
            source_status=500,
            quant_status=200,
        )
        == "request_failed"
    )


def test_run_probe_writes_release_gate_compatible_rows(monkeypatch):
    calls = []

    def fake_post(url, body, *, timeout):
        calls.append((url, body, timeout))
        is_source = "source" in url
        content = "ACK" if is_source else ""
        payload = {
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": len(content) or 1},
        }
        return 200, payload, 0.123

    monkeypatch.setattr(probe, "_post_json", fake_post)

    result = probe.run_probe(
        source_base_url="http://source:8000",
        quant_base_url="http://quant:8001",
        source_model="source",
        quant_model="quant",
        source_model_path="/models/source",
        quant_model_path="/models/quant",
        timeout=1,
        prompts=(
            {
                "name": "exact_ack",
                "messages": [{"role": "user", "content": "Reply ACK"}],
                "expected": "ACK",
                "max_tokens": 4,
            },
        ),
    )

    assert result["status"] == "pass"
    assert result["remote_evidence_only"] is False
    assert result["source_model_path"] == "/models/source"
    assert result["quant_model_path"] == "/models/quant"
    assert result["summary"]["quant_diverges_from_source"] == 1
    row = result["rows"][0]
    assert row["prompt_name"] == "exact_ack"
    assert row["classification"] == "quant_diverges_from_source"
    assert row["source_output"] == "ACK"
    assert row["quant_output"] == ""
    assert row["first_divergence"]["char_index"] == 0
    assert json.dumps(result)
    assert len(calls) == 2


def test_preflight_records_missing_source_and_endpoint_state(tmp_path, monkeypatch):
    quant_model = tmp_path / "MiMo-V2.5-JANG_2L"
    quant_model.mkdir()

    def fake_get(url, *, timeout):
        if "quant" in url:
            return 200, {"status": "ok"}, None
        return None, None, "URLError: refused"

    monkeypatch.setattr(probe, "_get_json", fake_get)

    result = probe.preflight(
        source_base_url="http://source:8000",
        quant_base_url="http://quant:8001",
        source_model_path="/missing/source",
        quant_model_path=str(quant_model),
        timeout=1,
    )

    assert result["status"] == "missing_prerequisites"
    assert result["remote_evidence_only"] is False
    assert result["rows"] == []
    assert result["endpoints"]["source"]["healthy"] is False
    assert result["endpoints"]["quant"]["healthy"] is True
    assert result["paths"]["source"]["exists"] is False
    assert result["paths"]["quant"]["exists"] is True
    assert result["blockers"] == [
        "missing_or_unhealthy_source_endpoint",
        "missing_source_model_path",
    ]


def test_main_preflight_writes_missing_prerequisites_artifact(tmp_path, monkeypatch):
    out = tmp_path / "preflight.json"
    monkeypatch.setattr(
        probe,
        "_get_json",
        lambda url, *, timeout: (None, None, "URLError: refused"),
    )

    rc = probe.main(
        [
            "--preflight-only",
            "--out",
            str(out),
            "--quant-model-path",
            "/missing/quant",
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["status"] == "missing_prerequisites"
    assert payload["blockers"] == [
        "missing_or_unhealthy_source_endpoint",
        "missing_or_unhealthy_quant_endpoint",
        "missing_source_model_path",
        "missing_quant_model_path",
    ]
