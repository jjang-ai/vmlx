from __future__ import annotations

import json
from pathlib import Path

from tests.cross_matrix.run_mimo_v2_jang2l_current_audit import build_audit


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


def test_mimo_current_audit_separates_clean_artifact_from_runtime_blockers(tmp_path):
    model_parent = tmp_path / "models" / "JANGQ-AI"
    model_path = model_parent / "MiMo-V2.5-JANG_2L"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text("{}")
    manifest = tmp_path / "build" / "current-mimo-http-tb5-manifest-20260606.tsv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        f"{(model_path / 'config.json').stat().st_size}\tMiMo-V2.5-JANG_2L/config.json\n"
    )

    _write_json(
        tmp_path / "build/current-mimo-jang2l-local-structural-verify-20260606.json",
        {"status": "pass"},
    )
    _write_json(
        tmp_path / "build/current-mimo-jang2l-live-text-cache-smoke-20260606.json",
        {
            "requests": [
                {
                    "response": {
                        "http_status": 200,
                        "json": {
                            "choices": [
                                {"message": {"content": "cache ok"}, "finish_reason": "stop"}
                            ]
                        },
                    }
                }
            ]
        },
    )
    _write_json(
        tmp_path / "build/current-mimo-v2-jang2l-quantized-switchglu-parity-20260606.json",
        {
            "got_shape": [2, 3, 4096],
            "manual_shape": [2, 3, 4096],
            "max_abs_diff": 0.0007,
            "mean_abs_diff": 0.00009,
        },
    )
    _write_json(
        tmp_path / "build/current-mimo-v2-jang2l-cache-vs-nocache-next-token-20260606.json",
        {
            "rows": [
                {"mode": "no_cache", "top10": [{"id": 1, "text": "A"}]},
                {"mode": "cache_prefill", "top10": [{"id": 1, "text": "A"}]},
            ]
        },
    )
    _write_json(
        tmp_path / "build/current-mimo-v2-jang2l-direct-length-sweep-20260606.json",
        {"status": "fail", "cases": [{"status": "fail_corrupt"}]},
    )
    _write_json(
        tmp_path / "build/current-mimo-v2-jang2l-tool-dialect-failure-20260606.json",
        {"status": "fail", "runtime_observations": [{"result": "fail_incomplete_marker"}]},
    )
    _write_json(
        tmp_path / "build/current-mimo-v2-jang2l-sink-mode-length-diagnostic-20260606.json",
        {
            "status": "fail",
            "cases": [
                {"manual_sink": False, "passes_length_ok": False},
                {"manual_sink": True, "passes_length_ok": False},
            ],
        },
    )
    _write_json(
        tmp_path
        / "build/current-mimo-v2-jang2l-disable-sink-length-diagnostic-20260606.json",
        {
            "status": "fail",
            "cases": [{"passes_length_ok": False, "contains_corrupt_repetition": True}],
        },
    )

    result = build_audit(tmp_path, model_path, manifest)

    assert result["status"] == "open"
    assert result["local_release_clearance"] is False
    assert result["component_ok"]["manifest_integrity"] is True
    assert result["component_ok"]["text_cache_narrow"] is True
    assert result["component_ok"]["cache_vs_nocache_next_token"] is True
    assert result["component_ok"]["long_prompt_coherence"] is False
    assert result["component_ok"]["tool_protocol"] is False
    assert result["component_ok"]["manual_sink_does_not_clear_length_generation"] is True
    assert result["component_ok"]["disable_sink_does_not_clear_length_generation"] is True
    assert result["diagnostics"]["manual_sink_sdpa_clears_length_generation"] is False
    assert result["diagnostics"]["disable_sink_clears_length_generation"] is False
    assert "sink-kernel-only" in result["diagnostics"]["sink_boundary"]
    assert result["blockers"] == [
        "mimo_long_prompt_coherence_blocked",
        "mimo_tool_protocol_blocked",
    ]
