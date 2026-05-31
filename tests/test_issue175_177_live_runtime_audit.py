# SPDX-License-Identifier: Apache-2.0
"""Contracts for live installed-app runtime evidence on issues #175-#177."""

from pathlib import Path


def test_issue175_177_live_runtime_audit_accepts_qwen_installed_cache_hit():
    from tests.cross_matrix import run_issue175_177_live_runtime_audit as gate

    audit = gate.build_audit(Path("."))

    assert audit["status"] == "pass"
    assert audit["checks"]["installed_app_qwen_live_probe_passed"] is True
    assert audit["checks"]["cache_hit_ttft_improved"] is True
    assert audit["checks"]["l2_block_disk_hits_observed"] is True
    assert audit["checks"]["paged_ssm_cache_hit_observed"] is True
    assert audit["checks"]["visible_content_observed"] is True
    assert audit["checks"]["minimax_installed_live_probe_passed"] is True
    assert audit["checks"]["turboquant_live_decode_observed"] is True
    assert audit["checks"]["scheduler_cache_selection_observed"] is True
    assert audit["checks"]["scheduler_cache_execution_timing_observed"] is True
    assert audit["checks"]["paged_tq_cache_hit_observed"] is True
    assert audit["checks"]["minimax_restart_reader_probe_passed"] is True
    assert audit["checks"]["cold_paged_tq_restart_hit_observed"] is True
    assert audit["checks"]["cold_paged_cache_selection_observed"] is True
    assert audit["checks"]["cold_paged_stream_ttft_observed"] is True
    assert audit["checks"]["admin_sleep_lifecycle_probe_passed"] is True
    assert audit["checks"]["admin_sleep_deep_unload_observed"] is True
    assert audit["remaining_blockers"] == []


def test_issue175_177_live_runtime_audit_writes_json_artifact(tmp_path):
    from tests.cross_matrix import run_issue175_177_live_runtime_audit as gate

    out = tmp_path / "issue175-177-live-runtime.json"
    audit = gate.write_audit(Path("."), out)

    assert out.exists()
    assert audit["artifact"] == str(out)
    assert '"checks"' in out.read_text(encoding="utf-8")
