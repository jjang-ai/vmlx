# SPDX-License-Identifier: Apache-2.0
"""Contracts for installed-app runtime parity on issues #175-#177."""

from pathlib import Path


def test_issue175_177_installed_runtime_audit_proves_packaged_runtime_surface():
    from tests.cross_matrix import run_issue175_177_installed_runtime_audit as gate

    audit = gate.build_audit(Path("."))

    assert audit["status"] == "pass"
    assert audit["checks"]["installed_python_exists"] is True
    assert audit["checks"]["memory_clear_helper_imports_from_installed_app"] is True
    assert audit["checks"]["memory_clear_uses_available_mlx_api"] is True
    assert audit["checks"]["runtime_paths_avoid_removed_clear_memory_cache"] is True
    assert audit["checks"]["admin_sleep_uses_memory_clear_helper"] is True
    assert audit["checks"]["promoted_disk_blocks_drop_parent_mirror"] is True
    assert audit["checks"]["l2_readable_write_through_blocks_drop_parent_mirror"] is True
    assert audit["checks"]["cache_selection_telemetry_installed"] is True
    assert audit["checks"]["cache_execution_timing_telemetry_installed"] is True


def test_issue175_177_installed_runtime_audit_writes_json_artifact(tmp_path):
    from tests.cross_matrix import run_issue175_177_installed_runtime_audit as gate

    out = tmp_path / "issue175-177-installed-runtime.json"
    audit = gate.write_audit(Path("."), out)

    assert out.exists()
    assert audit["artifact"] == str(out)
    assert '"checks"' in out.read_text(encoding="utf-8")
