# SPDX-License-Identifier: Apache-2.0

from tests.cross_matrix import run_qwen36_tp4_gateway_probe as probe


def test_parse_last_json_line_ignores_noise():
    assert probe._parse_last_json_line("noise\n{\"ok\": true}\n") == {"ok": True}
    assert probe._parse_last_json_line("noise only") is None


def test_rank_targets_from_health_extracts_request_response_dirs():
    health = {
        "body": {
            "rank_status": [
                {
                    "rank": 0,
                    "host": "adlab-n1-raw",
                    "path": "/rank0",
                    "requests_dir": "/rank0/requests",
                    "responses_dir": "/rank0/responses",
                    "reachable": True,
                    "ready": True,
                    "extra": "ignored",
                }
            ]
        }
    }

    assert probe.rank_targets_from_health(health) == [
        {
            "rank": 0,
            "host": "adlab-n1-raw",
            "path": "/rank0",
            "requests_dir": "/rank0/requests",
            "responses_dir": "/rank0/responses",
            "reachable": True,
            "ready": True,
        }
    ]


def test_classify_probe_reports_generation_timeout():
    result = {"chat": {"ssh": {"timed_out": True}, "json": None}}
    assert probe.classify_probe(result) == "gateway_generation_timeout"

    result = {
        "chat": {
            "ssh": {"timed_out": False},
            "json": {"http_status": None, "error": "timeout: timed out"},
        }
    }
    assert probe.classify_probe(result) == "gateway_generation_timeout"


def test_classify_probe_requires_exact_visible_output():
    result = {
        "chat": {
            "ssh": {"timed_out": False},
            "json": {
                "http_status": 200,
                "body": {"choices": [{"message": {"content": "QWEN-OK"}}]},
            },
        }
    }
    assert probe.classify_probe(result) == "pass"

    result["chat"]["json"]["body"]["choices"][0]["message"]["content"] = "wrong"
    assert probe.classify_probe(result) == "exact_output_failed"


def test_run_probe_uses_rank_snapshots_on_timeout(monkeypatch):
    def fake_get(host, port, path, *, timeout):
        if path == "/health":
            return {
                "json": {
                    "http_status": 200,
                    "body": {
                        "rank_targets": [
                            {
                                "rank": 0,
                                "host": "rank-host",
                                "requests_dir": "/req",
                                "responses_dir": "/resp",
                                "reachable": True,
                                "ready": True,
                            }
                        ]
                    },
                }
            }
        return {"json": {"http_status": 200, "body": {"data": []}}}

    monkeypatch.setattr(probe, "_remote_json_get", fake_get)
    monkeypatch.setattr(
        probe,
        "_remote_chat_post",
        lambda *args, **kwargs: {
            "ssh": {"timed_out": True},
            "json": None,
        },
    )
    monkeypatch.setattr(
        probe,
        "_rank_dir_snapshot",
        lambda host, targets, *, timeout: [{"rank": 0, "snapshot": {"requests": {}}}],
    )

    result = probe.run_probe(
        host="max2",
        port=8124,
        model="qwen",
        request_timeout=1,
        process_timeout=2,
        rank_snapshot_timeout=1,
    )

    assert result["status"] == "open"
    assert result["classification"] == "gateway_generation_timeout"
    assert result["rank_targets"][0]["host"] == "rank-host"
    assert result["rank_snapshots"][0]["rank"] == 0
