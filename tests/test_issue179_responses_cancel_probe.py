# SPDX-License-Identifier: Apache-2.0
"""Contracts for the MiniMax-K issue #179 Responses cancel probe."""

from pathlib import Path
from types import SimpleNamespace

from tests.cross_matrix import run_issue179_responses_cancel_probe as probe


def test_extract_response_id_from_responses_sse_event():
    raw = "\n".join(
        [
            'event: response.created',
            'data: {"response":{"id":"resp_issue179_probe"}}',
            "",
            'event: response.output_text.delta',
            'data: {"delta":"Hello"}',
            "",
        ]
    )

    assert probe.extract_response_id(raw) == "resp_issue179_probe"


def test_classify_cancel_probe_distinguishes_route_from_model_text():
    result = probe.classify_probe(
        {
            "stream_started": True,
            "response_id": "resp_issue179_probe",
            "cancel_status": 200,
            "raw_content_text": "",
            "raw_reasoning_text": "The user said Hi.",
            "stream_error": "cancelled after proof",
        }
    )

    assert result["cancel_route_present"] is True
    assert result["bad_text_captured"] is False
    assert result["abort_boundary"] == "controlled_cancel_after_response_id"


def test_issue179_cancel_policy_waits_for_bad_reasoning_before_cancel():
    state = probe.CancelPolicyState(response_id_seen_at=10.0)

    assert (
        probe.cancel_due(
            state,
            now=10.2,
            delay_seconds=0.25,
            cancel_on_bad_text=True,
            bad_text_seen=False,
        )
        is False
    )
    assert (
        probe.cancel_due(
            state,
            now=10.21,
            delay_seconds=0.25,
            cancel_on_bad_text=True,
            bad_text_seen=True,
        )
        is True
    )
    assert state.cancel_trigger == "bad_text"


def test_issue179_cancel_policy_falls_back_to_delay_without_bad_text():
    state = probe.CancelPolicyState(response_id_seen_at=10.0)

    assert (
        probe.cancel_due(
            state,
            now=10.2,
            delay_seconds=0.25,
            cancel_on_bad_text=False,
            bad_text_seen=True,
        )
        is False
    )
    assert (
        probe.cancel_due(
            state,
            now=10.26,
            delay_seconds=0.25,
            cancel_on_bad_text=False,
            bad_text_seen=False,
        )
        is True
    )
    assert state.cancel_trigger == "delay_elapsed"


def test_issue179_memory_preflight_skips_before_launch_when_below_floor():
    vm_stat = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                             1000.
Pages speculative:                       500.
Pages purgeable:                         250.
"""

    result = probe.memory_preflight_block(
        SimpleNamespace(
            model=Path("/models/MiniMax-M2.7-JANGTQ_K"),
            memory_preflight_margin_gb=6.0,
            memory_preflight_headroom_gb=2.0,
        ),
        vm_stat_text=vm_stat,
        model_size_gb_override=74.0,
        top_processes_text="""  PID   RSS COMMAND
  111 2097152 /Applications/vMLX.app/Contents/MacOS/vMLX
""",
        active_processes_text="",
    )

    assert result == {
        "status": "skipped",
        "reason": "insufficient_vm_stat_memory",
        "launch_allowed": False,
        "launch_decision": "do_not_launch",
        "did_not_launch": True,
        "launch_blockers": ["insufficient_memory"],
        "model_path": "/models/MiniMax-M2.7-JANGTQ_K",
        "model_size_gb": 74.0,
        "required_free_gb": 82.0,
        "min_free_gb": 82.0,
        "memory_preflight_margin_gb": 6.0,
        "memory_preflight_headroom_gb": 2.0,
        "available_for_gate_gb": 0.03,
        "free_plus_speculative_purgeable_gb": 0.03,
        "memory_gap_gb": 81.97,
        "preflight_memory_source": "vm_stat_free_plus_speculative_purgeable",
        "commands": {
            "memory": "vm_stat",
            "top_memory_processes": "ps -axo pid,rss,command -r | head -n 11",
            "active_heavy_processes": "pgrep -af 'vmlx_engine.cli serve|mlx_lm.server|run_runtime_memory_stress_probe|run_decode_speed_gate|live-real-ui-model-proof|run_dsv4_route_mode_code_exactness|run_dsv4_default_cache_tool_loop_gate|run_current_regression_suite|run_issue179_responses_cancel_probe'",
        },
        "top_memory_processes": [
            {
                "pid": 111,
                "rss_gb": 2.0,
                "command": "/Applications/vMLX.app/Contents/MacOS/vMLX",
            }
        ],
        "active_heavy_processes": [],
        "active_heavy_process_count": 0,
    }


def test_issue179_memory_preflight_records_process_context_before_skip():
    vm_stat = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                             1000.
Pages speculative:                       500.
Pages purgeable:                         250.
"""

    result = probe.memory_preflight_block(
        SimpleNamespace(
            model=Path("/models/MiniMax-M2.7-JANGTQ_K"),
            memory_preflight_margin_gb=6.0,
            memory_preflight_headroom_gb=2.0,
        ),
        vm_stat_text=vm_stat,
        model_size_gb_override=74.0,
        top_processes_text="""  PID   RSS COMMAND
  111 2097152 /Applications/vMLX.app/Contents/MacOS/vMLX
  222 1048576 codex --yolo resume
""",
        active_processes_text="",
    )

    assert result["launch_blockers"] == ["insufficient_memory"]
    assert result["active_heavy_process_count"] == 0
    assert result["active_heavy_processes"] == []
    assert result["top_memory_processes"] == [
        {
            "pid": 111,
            "rss_gb": 2.0,
            "command": "/Applications/vMLX.app/Contents/MacOS/vMLX",
        },
        {"pid": 222, "rss_gb": 1.0, "command": "codex --yolo resume"},
    ]
    assert result["commands"]["memory"] == "vm_stat"
    assert "active_heavy_processes" in result["commands"]


def test_issue179_memory_preflight_blocks_active_heavy_process_when_floor_met():
    vm_stat = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                          6000000.
Pages speculative:                    600000.
Pages purgeable:                      300000.
"""

    result = probe.memory_preflight_block(
        SimpleNamespace(
            model=Path("/models/MiniMax-M2.7-JANGTQ_K"),
            memory_preflight_margin_gb=6.0,
            memory_preflight_headroom_gb=2.0,
        ),
        vm_stat_text=vm_stat,
        model_size_gb_override=74.0,
        top_processes_text="""  PID   RSS COMMAND
  333 1048576 /usr/bin/python run_current_regression_suite.py
""",
        active_processes_text="333 /usr/bin/python run_current_regression_suite.py\n",
    )

    assert result["status"] == "skipped_active_heavy_process"
    assert result["reason"] == "active_heavy_process"
    assert result["launch_allowed"] is False
    assert result["launch_decision"] == "do_not_launch"
    assert result["launch_blockers"] == ["active_heavy_process"]
    assert result["active_heavy_processes"] == [
        {"pid": 333, "command": "/usr/bin/python run_current_regression_suite.py"}
    ]


def test_issue179_memory_preflight_only_reports_ready_without_live_launch():
    vm_stat = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                          6000000.
Pages speculative:                    600000.
Pages purgeable:                      300000.
"""

    result = probe.memory_preflight_only_result(
        SimpleNamespace(
            model=Path("/models/MiniMax-M2.7-JANGTQ_K"),
            memory_preflight_margin_gb=6.0,
            memory_preflight_headroom_gb=2.0,
        ),
        vm_stat_text=vm_stat,
        model_size_gb_override=74.0,
        top_processes_text="""  PID   RSS COMMAND
  111 2097152 /Applications/vMLX.app/Contents/MacOS/vMLX
""",
        active_processes_text="",
    )

    assert result == {
        "status": "ready_to_launch",
        "reason": "memory_preflight_floor_met",
        "launch_allowed": True,
        "launch_decision": "launch_allowed",
        "did_not_launch": True,
        "launch_blockers": [],
        "model_path": "/models/MiniMax-M2.7-JANGTQ_K",
        "model_size_gb": 74.0,
        "required_free_gb": 82.0,
        "min_free_gb": 82.0,
        "memory_preflight_margin_gb": 6.0,
        "memory_preflight_headroom_gb": 2.0,
        "available_for_gate_gb": 105.29,
        "free_plus_speculative_purgeable_gb": 105.29,
        "memory_gap_gb": 0.0,
        "preflight_memory_source": "vm_stat_free_plus_speculative_purgeable",
        "commands": {
            "memory": "vm_stat",
            "top_memory_processes": "ps -axo pid,rss,command -r | head -n 11",
            "active_heavy_processes": "pgrep -af 'vmlx_engine.cli serve|mlx_lm.server|run_runtime_memory_stress_probe|run_decode_speed_gate|live-real-ui-model-proof|run_dsv4_route_mode_code_exactness|run_dsv4_default_cache_tool_loop_gate|run_current_regression_suite|run_issue179_responses_cancel_probe'",
        },
        "top_memory_processes": [
            {
                "pid": 111,
                "rss_gb": 2.0,
                "command": "/Applications/vMLX.app/Contents/MacOS/vMLX",
            }
        ],
        "active_heavy_processes": [],
        "active_heavy_process_count": 0,
    }


def test_issue179_memory_preflight_blocks_tiny_floor_surplus_as_unsafe():
    # This reproduces the local unsafe case: model+margin is technically met,
    # but there is not enough additional headroom for a responsible launch.
    pages_for_80_1_gb = int(80.1 * (1024**3) / 16384)
    vm_stat = f"""Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                          {pages_for_80_1_gb}.
Pages speculative:                         0.
Pages purgeable:                           0.
"""

    result = probe.memory_preflight_only_result(
        SimpleNamespace(
            model=Path("/models/MiniMax-M2.7-JANGTQ_K"),
            memory_preflight_margin_gb=6.0,
            memory_preflight_headroom_gb=2.0,
        ),
        vm_stat_text=vm_stat,
        model_size_gb_override=74.0,
        top_processes_text="  PID   RSS COMMAND\n",
        active_processes_text="",
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "insufficient_vm_stat_memory"
    assert result["required_free_gb"] == 82.0
    assert result["available_for_gate_gb"] == 80.1
    assert result["memory_gap_gb"] == 1.9
    assert result["launch_allowed"] is False
