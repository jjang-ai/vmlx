# SPDX-License-Identifier: Apache-2.0
"""Contracts for the DSV4 real-UI no-heavy memory preflight."""

from pathlib import Path

from tests.cross_matrix import run_real_ui_dsv4_memory_preflight as gate


def test_dsv4_memory_preflight_default_out_tracks_current_release_artifact():
    assert gate.DEFAULT_OUT == Path(
        "build/current-real-ui-dsv4-memory-preflight-20260601-local-recheck.json"
    )


def test_dsv4_memory_preflight_records_insufficient_memory(tmp_path):
    model = tmp_path / "DeepSeek-V4-Flash-JANGTQ-K"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"x" * 1024)
    vm_stat = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               100000.
Pages active:                                  1.
Pages inactive:                           200000.
Pages speculative:                         50000.
Pages purgeable:                           25000.
"""

    payload = gate.build_preflight(
        model_path=model,
        required_available_gb=120.0,
        vm_stat_text=vm_stat,
    )

    assert payload["status"] == "skipped_insufficient_memory"
    assert payload["did_not_launch"] is True
    assert payload["model_path"].endswith("DeepSeek-V4-Flash-JANGTQ-K")
    assert payload["free_plus_speculative_purgeable_gb"] < 120.0
    assert payload["available_for_gate_gb"] == payload["free_plus_speculative_purgeable_gb"]
    assert payload["required_free_gb"] == 120.0
    assert payload["min_free_gb"] == 120.0
    assert payload["memory_gap_gb"] > 0
    assert payload["launch_decision"] == "do_not_launch"
    assert payload["launch_allowed"] is False
    assert payload["memory_breakdown"] == payload["memory_breakdown_gb"]
    assert payload["safety_margin_gb"] >= 40.0
    assert payload["active_heavy_processes"] == []
    assert payload["active_heavy_process_count"] == 0
    assert "insufficient_memory" in payload["launch_blockers"]


def test_dsv4_memory_preflight_records_psutil_available_as_diagnostic(tmp_path, monkeypatch):
    model = tmp_path / "DeepSeek-V4-Flash-JANGTQ-K"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"x" * 1024)
    vm_stat = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               100000.
Pages active:                                  1.
Pages inactive:                         9000000.
Pages speculative:                         50000.
Pages purgeable:                           25000.
"""

    monkeypatch.setattr(
        gate,
        "psutil_memory_snapshot",
        lambda: {"available_gb": 108.91, "percent": 15.0, "total_gb": 128.0},
    )

    payload = gate.build_preflight(
        model_path=model,
        required_available_gb=120.0,
        vm_stat_text=vm_stat,
    )

    assert payload["status"] == "skipped_insufficient_memory"
    assert payload["preflight_memory_source"] == "vm_stat_free_plus_speculative_purgeable"
    assert payload["psutil_available_gb"] == 108.91
    assert payload["psutil_memory_percent"] == 15.0
    assert payload["psutil_memory_gap_gb"] == 11.09
    assert payload["memory_gap_gb"] > payload["psutil_memory_gap_gb"]
    assert payload["inactive_file_cache_gb"] > 0


def test_dsv4_memory_preflight_records_memory_pressure_as_diagnostic(tmp_path):
    model = tmp_path / "DeepSeek-V4-Flash-JANGTQ-K"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"x" * 1024)
    vm_stat = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               100000.
Pages active:                                  1.
Pages inactive:                           200000.
Pages speculative:                         50000.
Pages purgeable:                           25000.
"""
    memory_pressure = """The system has 137438953472 (8388608 pages with a page size of 16384).

System-wide memory free percentage: 95%
"""

    payload = gate.build_preflight(
        model_path=model,
        required_available_gb=120.0,
        vm_stat_text=vm_stat,
        memory_pressure_text=memory_pressure,
    )

    assert payload["memory_pressure_free_percent"] == 95
    assert payload["commands"]["memory_pressure"] == "memory_pressure"


def test_dsv4_memory_preflight_refuses_to_skip_when_floor_is_met(tmp_path):
    model = tmp_path / "DeepSeek-V4-Flash-JANGTQ-K"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"x" * 1024)
    vm_stat = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                              9000000.
Pages active:                                  1.
Pages inactive:                                1.
Pages speculative:                       1000000.
Pages purgeable:                          100000.
"""

    payload = gate.build_preflight(
        model_path=model,
        required_available_gb=120.0,
        vm_stat_text=vm_stat,
    )

    assert payload["status"] == "ready_to_launch"
    assert payload["did_not_launch"] is True
    assert payload["free_plus_speculative_purgeable_gb"] >= 120.0
    assert payload["available_for_gate_gb"] == payload["free_plus_speculative_purgeable_gb"]
    assert payload["memory_gap_gb"] == 0
    assert payload["launch_decision"] == "launch_allowed"
    assert payload["launch_allowed"] is True
    assert payload["safety_margin_gb"] >= 40.0
    assert payload["launch_blockers"] == []


def test_dsv4_memory_preflight_marks_under_margin_floor_invalid(tmp_path):
    model = tmp_path / "DeepSeek-V4-Flash-JANGTQ-K"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"x" * 1024 * 1024 * 80)
    vm_stat = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               100000.
Pages active:                                  1.
Pages inactive:                           200000.
Pages speculative:                         50000.
Pages purgeable:                           25000.
"""

    payload = gate.build_preflight(
        model_path=model,
        required_available_gb=0.08,
        vm_stat_text=vm_stat,
    )

    assert payload["status"] == "invalid_preflight_floor"
    assert payload["launch_decision"] == "do_not_launch"
    assert payload["safety_margin_gb"] < 40.0
    assert payload["floor_valid"] is False
    assert "invalid_preflight_floor" in payload["launch_blockers"]


def test_dsv4_memory_preflight_blocks_when_heavy_process_is_active(tmp_path):
    model = tmp_path / "DeepSeek-V4-Flash-JANGTQ-K"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"x" * 1024)
    vm_stat = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                              9000000.
Pages active:                                  1.
Pages inactive:                                1.
Pages speculative:                       1000000.
Pages purgeable:                          100000.
"""
    active = """1234 .venv/bin/python tests/cross_matrix/run_runtime_memory_stress_probe.py --row gemma4
5678 .venv/bin/python harmless.py
"""

    payload = gate.build_preflight(
        model_path=model,
        required_available_gb=120.0,
        vm_stat_text=vm_stat,
        active_processes_text=active,
    )

    assert payload["status"] == "skipped_active_heavy_process"
    assert payload["did_not_launch"] is True
    assert payload["launch_decision"] == "do_not_launch"
    assert payload["active_heavy_process_count"] == 1
    assert payload["active_heavy_processes"][0]["pid"] == 1234
    assert "active_heavy_process" in payload["launch_blockers"]


def test_dsv4_memory_preflight_records_top_memory_processes(tmp_path):
    model = tmp_path / "DeepSeek-V4-Flash-JANGTQ-K"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"x" * 1024)
    vm_stat = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               100000.
Pages active:                                  1.
Pages inactive:                           200000.
Pages speculative:                         50000.
Pages purgeable:                           25000.
"""
    ps_text = """  PID      RSS COMMAND
 1001 10485760 /Applications/vMLX.app/Contents/MacOS/vMLX
 1002  2097152 /usr/bin/python model-worker
 bad  row
"""

    payload = gate.build_preflight(
        model_path=model,
        required_available_gb=120.0,
        vm_stat_text=vm_stat,
        top_processes_text=ps_text,
    )

    assert payload["commands"]["top_memory_processes"].startswith("ps ")
    assert payload["top_memory_processes"][0]["pid"] == 1001
    assert payload["top_memory_processes"][0]["rss_gb"] == 10.0
    assert payload["top_memory_processes"][0]["command"].endswith("vMLX")
    assert payload["top_memory_processes"][1]["rss_gb"] == 2.0


def test_dsv4_memory_preflight_writes_json(tmp_path):
    model = tmp_path / "DeepSeek-V4-Flash-JANGTQ-K"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"x" * 1024)
    out = tmp_path / "preflight.json"

    payload = gate.write_preflight(
        model_path=model,
        out_path=out,
        required_available_gb=99999.0,
    )

    assert payload["status"] == "skipped_insufficient_memory"
    assert out.exists()
    assert '"did_not_launch": true' in out.read_text(encoding="utf-8")
