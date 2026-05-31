# SPDX-License-Identifier: Apache-2.0
"""Contracts for the MiniMax-K issue #179 Responses cancel probe."""

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
