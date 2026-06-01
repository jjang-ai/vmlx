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
        ),
        vm_stat_text=vm_stat,
        model_size_gb_override=74.0,
    )

    assert result == {
        "status": "skipped",
        "reason": "insufficient_vm_stat_memory",
        "launch_allowed": False,
        "launch_decision": "do_not_launch",
        "model_path": "/models/MiniMax-M2.7-JANGTQ_K",
        "model_size_gb": 74.0,
        "required_free_gb": 80.0,
        "min_free_gb": 80.0,
        "available_for_gate_gb": 0.03,
        "free_plus_speculative_purgeable_gb": 0.03,
        "memory_gap_gb": 79.97,
        "preflight_memory_source": "vm_stat_free_plus_speculative_purgeable",
    }
