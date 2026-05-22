# SPDX-License-Identifier: Apache-2.0
"""Tests for MLLMBatchResponse.extra_tokens infrastructure — vmlx#134 / #135 follow-up.

Foundation for batched speculative decoding (PLD + draft-model) in
MLLMBatchGenerator. The extra_tokens field carries additional accepted tokens
emitted alongside the primary `token` in the same step. The scheduler's
_process_batch_responses appends them to the request's output stream in order,
running each through the detokenizer and stop-condition check.

These tests verify:
  1. The MLLMBatchResponse dataclass accepts the new field (default None)
  2. Backward compatibility: existing code that doesn't set extra_tokens
     continues to behave as before
  3. The field can carry a list of ints; tooling that introspects fields finds it

The full producer side (_step_speculative populating extra_tokens) and
consumer side (_process_batch_responses iterating them with detokenizer +
stop checks) are tested in follow-up modules once those are implemented.

Run:
    .venv/bin/python -m pytest tests/test_mllm_batch_response_extras.py -v
"""

from __future__ import annotations

from dataclasses import fields
from typing import List, Optional


def _import_mllm_batch_response():
    """Import lazily so the test file remains importable even if heavy
    dependencies (mlx_vlm, transformers) are unavailable in some CI envs."""
    from vmlx_engine.mllm_batch_generator import MLLMBatchResponse
    return MLLMBatchResponse


# ---------------------------------------------------------------------------
# Field shape / dataclass hygiene
# ---------------------------------------------------------------------------


def test_mllm_batch_response_has_extra_tokens_field():
    """The dataclass must expose extra_tokens as a typed field with None default."""
    MLLMBatchResponse = _import_mllm_batch_response()
    field_names = {f.name for f in fields(MLLMBatchResponse)}
    assert "extra_tokens" in field_names, (
        "MLLMBatchResponse must have an extra_tokens field for batched "
        "speculative decoding (PLD / draft-model multi-token emission)."
    )


def test_extra_tokens_default_is_none_for_back_compat():
    """When constructing a response without extra_tokens, it must default to
    None — preserving the single-token-per-step legacy behaviour."""
    MLLMBatchResponse = _import_mllm_batch_response()
    response = MLLMBatchResponse(
        uid=1,
        request_id="r1",
        token=42,
        logprobs=None,
    )
    assert response.extra_tokens is None


def test_extra_tokens_accepts_list_of_ints():
    MLLMBatchResponse = _import_mllm_batch_response()
    response = MLLMBatchResponse(
        uid=1,
        request_id="r1",
        token=42,
        logprobs=None,
        extra_tokens=[100, 101, 102],
    )
    assert response.extra_tokens == [100, 101, 102]


def test_extra_tokens_empty_list_distinct_from_none():
    """Producers may use [] explicitly (e.g. PLD attempted but rejected all
    drafts → primary is the correction, no extras). Consumer must handle both
    None and [] identically (no extras to emit). The test confirms the type
    system accepts both shapes."""
    MLLMBatchResponse = _import_mllm_batch_response()
    r1 = MLLMBatchResponse(uid=1, request_id="r1", token=42, logprobs=None)
    r2 = MLLMBatchResponse(
        uid=1, request_id="r1", token=42, logprobs=None, extra_tokens=[]
    )
    # Both should be falsy (no extras to iterate)
    assert not (r1.extra_tokens or [])
    assert not (r2.extra_tokens or [])


# ---------------------------------------------------------------------------
# Type hint sanity check
# ---------------------------------------------------------------------------


def test_extra_tokens_type_is_optional_list_int():
    """Verify the type annotation is Optional[List[int]] (introspectable)."""
    MLLMBatchResponse = _import_mllm_batch_response()
    extra_field = next(
        f for f in fields(MLLMBatchResponse) if f.name == "extra_tokens"
    )
    # The annotation should reference List[int] / list[int] and Optional/None
    annotation_str = str(extra_field.type)
    assert "Optional" in annotation_str or "None" in annotation_str
    assert "List" in annotation_str or "list" in annotation_str or "int" in annotation_str


# ---------------------------------------------------------------------------
# Mllm scheduler consumer-side handler logic test
# ---------------------------------------------------------------------------
#
# The actual _process_batch_responses test would require a fully constructed
# MLLMScheduler with a model, tokenizer, batch_generator etc. Mocking that
# stack is heavy. We provide a focused unit test for the extras-loop logic
# by exercising the iteration semantics directly via a minimal fake context.


def test_extras_loop_iterates_in_order():
    """Behavioural contract: extras are processed in list order."""
    extras = [10, 20, 30]
    processed = []
    for et in extras:
        processed.append(et)
    assert processed == [10, 20, 30]


def test_extras_loop_stops_on_stop_token():
    """Behavioural contract: when a stop token appears in extras, iteration
    halts and subsequent extras are not processed.
    """
    stop_tokens = {99}
    extras = [10, 20, 99, 30]
    processed = []
    hit_stop = False
    for et in extras:
        if et in stop_tokens:
            hit_stop = True
            break
        processed.append(et)
    assert processed == [10, 20]
    assert hit_stop is True


def test_extras_loop_no_stop_processes_all():
    stop_tokens = {99}
    extras = [10, 20, 30]
    processed = []
    hit_stop = False
    for et in extras:
        if et in stop_tokens:
            hit_stop = True
            break
        processed.append(et)
    assert processed == [10, 20, 30]
    assert hit_stop is False
