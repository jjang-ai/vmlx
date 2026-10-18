from __future__ import annotations

import argparse
import base64
import errno
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

if not any(
    value in {"--v5-fixture-producer", "--v5-fixture-command"}
    for value in sys.argv
):
    import pytest
else:
    class _FixtureMark:
        @staticmethod
        def parametrize(*_args, **_kwargs):
            return lambda function: function

    class _FixturePytest:
        mark = _FixtureMark()

    pytest = _FixturePytest()

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "panel/scripts/scoped-release-preflight-20.py"
STAMP = "2026-07-25T00:00:00Z"


def _v5_pinned_tool_path(name: str, env_key: str) -> Path:
    value = os.environ.get(env_key) or shutil.which(name) or ""
    path = Path(value).resolve()
    assert path.is_file()
    return path


def _fixture_b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _fixture_json_b64(value) -> str:
    return _fixture_b64(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )


def _fixture_jsonl_b64(rows: list[dict]) -> str:
    return _fixture_b64(
        b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
            + b"\n"
            for row in rows
        )
    )


def _fixture_sse(event: str, value: dict) -> str:
    return (
        (f"event: {event}\n" if event else "")
        + "data: "
        + json.dumps(value, separators=(",", ":"))
        + "\n\n"
    )


def _fixture_protocol_response(protocol: str, turn: int) -> bytes:
    reasoning = f"reason-{turn}"
    if turn == 1:
        tool = ("c1", "file_info", {"path": "panel/package.json"})
        finish = "tool_calls"
    elif turn == 2:
        tool = ("c2", "run_command", {"command": "pwd"})
        finish = "tool_calls"
    else:
        tool = None
        finish = "stop"
    if protocol == "chat":
        chunks = [
            {
                "choices": [
                    {
                        "delta": {"reasoning_content": reasoning},
                        "finish_reason": None,
                    }
                ]
            }
        ]
        if tool:
            chunks.append(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": tool[0],
                                        "function": {
                                            "name": tool[1],
                                            "arguments": json.dumps(
                                                tool[2], separators=(",", ":")
                                            ),
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                }
            )
        else:
            chunks.extend(
                [
                    {
                        "choices": [
                            {
                                "delta": {"content": "R20-"},
                                "finish_reason": None,
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {"content": "V5-DONE"},
                                "finish_reason": None,
                            }
                        ]
                    },
                ]
            )
        chunks.append(
            {"choices": [{"delta": {}, "finish_reason": finish}]}
        )
        return (
            "".join(_fixture_sse("", chunk) for chunk in chunks)
            + "data: [DONE]\n\n"
        ).encode()
    if protocol == "responses":
        events = [
            (
                "response.reasoning_text.delta",
                {
                    "type": "response.reasoning_text.delta",
                    "delta": reasoning,
                },
            )
        ]
        if tool:
            events.append(
                (
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "type": "function_call",
                            "call_id": tool[0],
                            "name": tool[1],
                            "arguments": json.dumps(
                                tool[2], separators=(",", ":")
                            ),
                        },
                    },
                )
            )
        else:
            events.extend(
                [
                    (
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "delta": "R20-",
                        },
                    ),
                    (
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "delta": "V5-DONE",
                        },
                    ),
                ]
            )
        events.append(
            (
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {"status": "completed"},
                },
            )
        )
        return "".join(_fixture_sse(name, value) for name, value in events).encode()
    if protocol == "anthropic":
        events = [
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "thinking_delta",
                        "thinking": reasoning,
                    },
                },
            )
        ]
        if tool:
            events.extend(
                [
                    (
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 1,
                            "content_block": {
                                "type": "tool_use",
                                "id": tool[0],
                                "name": tool[1],
                            },
                        },
                    ),
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 1,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json.dumps(
                                    tool[2], separators=(",", ":")
                                ),
                            },
                        },
                    ),
                ]
            )
            stop_reason = "tool_use"
        else:
            events.extend(
                [
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 1,
                            "delta": {"type": "text_delta", "text": "R20-"},
                        },
                    ),
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 1,
                            "delta": {
                                "type": "text_delta",
                                "text": "V5-DONE",
                            },
                        },
                    ),
                ]
            )
            stop_reason = "end_turn"
        events.extend(
            [
                (
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason},
                    },
                ),
                ("message_stop", {"type": "message_stop"}),
            ]
        )
        return "".join(_fixture_sse(name, value) for name, value in events).encode()
    rows = [{"message": {"thinking": reasoning}, "done": False}]
    if tool:
        rows.append(
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": tool[0],
                            "function": {
                                "name": tool[1],
                                "arguments": tool[2],
                            },
                        }
                    ]
                },
                "done": False,
            }
        )
    else:
        rows.extend(
            [
                {"message": {"content": "R20-"}, "done": False},
                {"message": {"content": "V5-DONE"}, "done": False},
            ]
        )
    rows.append({"message": {}, "done": True, "done_reason": finish})
    return b"".join(
        json.dumps(row, separators=(",", ":")).encode() + b"\n" for row in rows
    )


def _fixture_protocol_nonstream_response(protocol: str, turn: int) -> bytes:
    if protocol == "chat":
        value = {
            "choices": [
                {
                    "message": {"content": f"answer-{turn}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 10},
        }
    elif protocol == "responses":
        value = {
            "status": "completed",
            "output": [{"type": "message"}],
            "usage": {"output_tokens": 10},
        }
    elif protocol == "anthropic":
        value = {
            "content": [{"type": "text", "text": f"answer-{turn}"}],
            "stop_reason": "end_turn",
            "usage": {"output_tokens": 10},
        }
    else:
        value = {
            "message": {"content": f"answer-{turn}"},
            "done": True,
            "done_reason": "stop",
            "eval_count": 10,
        }
    return json.dumps(value, separators=(",", ":")).encode()


def _fixture_sampling_attestation(
    module,
    *,
    model: str = "fixture-laguna",
    defaults: dict | None = None,
) -> dict:
    defaults = dict(defaults or {"temperature": 0.6, "top_p": 0.95})
    resolved_defaults = {
        key: defaults[key]
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
        )
        if key in defaults
    }
    health_defaults = dict(resolved_defaults)
    for output_key in ("max_output_tokens", "max_new_tokens"):
        if output_key in defaults:
            health_defaults["max_output_tokens"] = defaults[output_key]
            break
    observations = []
    resolved_rows = []
    override_request = {"temperature": 0.2}
    for label, override in (
        ("default", {}),
        ("override", override_request),
        ("after_override", {}),
    ):
        proof_request_id = f"r20-sampling-{label}-fixture"
        request_id = f"{proof_request_id}-request"
        message_id = f"{proof_request_id}-message"
        request = {
            "model": model,
            "messages": [{"role": "user", "content": f"sample-{label}"}],
            "stream": False,
            "max_tokens": module.V5_SAMPLING_PROBE_MAX_TOKENS,
            "enable_thinking": False,
            **override,
        }
        resolved_values = {
            **resolved_defaults,
            **override,
            "max_tokens": module.V5_SAMPLING_PROBE_MAX_TOKENS,
            "enable_thinking": False,
        }
        line = (
            "Resolved sampling kwargs route=/v1/chat/completions "
            f"model={model} proof_request_id={proof_request_id} "
            f"request_id={request_id} message_id={message_id} "
            f"kwargs={resolved_values!r}"
        )
        request_bytes = json.dumps(
            request,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        result_bytes = json.dumps(
            {"status_code": 200, "terminals": [{"status": "completed"}]},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        resolved = {
            "route": "/v1/chat/completions",
            "model": model,
            "proof_request_id": proof_request_id,
            "request_id": request_id,
            "message_id": message_id,
            "values": resolved_values,
            "line_sha256": hashlib.sha256(line.encode()).hexdigest(),
            "line_b64": module._v5_encode_bytes(line.encode()),
        }
        observations.append(
            {
                "label": label,
                "proof_request_id": proof_request_id,
                "request_id": request_id,
                "message_id": message_id,
                "request_b64": module._v5_encode_bytes(request_bytes),
                "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
                "result_b64": module._v5_encode_bytes(result_bytes),
                "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
                "resolved": resolved,
            }
        )
        resolved_rows.append(resolved_values)
    return {
        "schema": "vmlx-r20-owned-sampling-attestation-v1",
        "health_effective_defaults": health_defaults,
        "default_resolved": resolved_rows[0],
        "override_request": override_request,
        "override_resolved": resolved_rows[1],
        "after_override_resolved": resolved_rows[2],
        "observations": observations,
    }


def _fixture_api_capture(
    module,
    run_id: str,
    nonce: str,
    *,
    phase_index: int = 0,
    session_id: str = "fixture-held-session",
) -> dict:
    phase = module.V5_CACHE_PHASES[phase_index]
    flows = []
    endpoints = {
        "chat": "/v1/chat/completions",
        "responses": "/v1/responses",
        "anthropic": "/v1/messages",
        "ollama": "/api/chat",
    }
    requests = [
        {
            "stream": True,
            "max_output_tokens": 64,
            "messages": [{"role": "user", "content": "Use file_info."}],
        },
        {
            "stream": True,
            "enable_thinking": True,
            "max_output_tokens": 64,
            "messages": [
                {"role": "tool", "tool_call_id": "c1", "content": "5.2 KB"},
                {"role": "user", "content": "Now use run_command."},
            ],
        },
        {
            "stream": True,
            "enable_thinking": False,
            "max_output_tokens": 64,
            "messages": [
                {"role": "tool", "tool_call_id": "c1", "content": "5.2 KB"},
                {"role": "tool", "tool_call_id": "c2", "content": str(ROOT)},
                {"role": "user", "content": "Reply exactly R20-V5-DONE"},
            ],
        },
    ]
    timing = {
        "started_ns": 1_000_000_000,
        "first_byte_ns": 1_100_000_000,
        "ended_ns": 1_300_000_000,
        "output_tokens": 10,
        "displayed_ttft_ms": 100.0,
        "displayed_tps": 50.0,
    }
    full_agentic = phase_index in {0, 5}
    selected_protocols = endpoints if full_agentic else {"chat": endpoints["chat"]}
    modes = ("stream", "nonstream") if full_agentic else ("stream",)
    for protocol, endpoint in selected_protocols.items():
        for route in ("direct", "gateway"):
            for mode in modes:
                for turn, request in enumerate(requests, start=1):
                    request_body = deepcopy(request)
                    request_body["stream"] = mode == "stream"
                    if protocol == "ollama" and "enable_thinking" in request_body:
                        request_body["think"] = request_body.pop(
                            "enable_thinking"
                        )
                    response = (
                        _fixture_protocol_response(protocol, turn)
                        if mode == "stream"
                        else _fixture_protocol_nonstream_response(
                            protocol,
                            turn,
                        )
                    )
                    flows.append(
                        {
                            "protocol": protocol,
                            "route": route,
                            "endpoint": endpoint,
                            "mode": mode,
                            "reasoning_mode": (
                                "auto",
                                "on",
                                "off",
                            )[turn - 1],
                            "request_b64": _fixture_json_b64(request_body),
                            "response_b64": _fixture_b64(response),
                            "timing_b64": _fixture_json_b64(timing),
                        }
                    )
    return {
        "schema": module.V5_API_SCHEMA,
        "run_id": run_id,
        "nonce": nonce,
        "phase_index": phase["index"],
        "phase_name": phase["name"],
        "representative_id": phase["representative_id"],
        "api_action_profile": phase["api_action_profile"],
        "session_id": session_id,
        "session_binding_sha256": str(phase_index) * 64,
        "flows": flows,
        "sampling_b64": _fixture_json_b64(
            _fixture_sampling_attestation(module)
        ),
    }


def _fixture_ui_capture(
    module,
    run_id: str,
    nonce: str,
    *,
    session_id: str = "fixture-session",
    phase_index: int = 5,
    evidence_root: Path | None = None,
) -> tuple[dict, dict]:
    phase = module.V5_CACHE_PHASES[phase_index]
    turn_count = phase["ui_turn_count"]
    if evidence_root is None:
        evidence_root = Path(
            tempfile.mkdtemp(prefix="vmlx-r20-ui-health-fixture-")
        )
    else:
        evidence_root.mkdir(parents=True, exist_ok=True)
    turns = []
    dom_messages = []
    source_records = []
    source_reasoning = []
    source_ui_turns = []
    source_user_messages = []
    resolved_sampling_records = []
    request_correlation_turns = []
    cache_rows = []
    for turn in range(1, turn_count + 1):
        response_id = f"resp-ui-{phase_index}-{turn}"
        reasoning = f"ui-reason-{turn}"
        content = (
            f"UI-V5-{turn}"
            if turn < 3
            else r"UI-V5-3 $43 \(47 \times 19\)"
        )
        if phase_index == 0 and turn == 1:
            request_content = (
                "Keep literal $43 and render "
                r"$47 \times 19 = 893 < 920 = 46 \times 20$."
            )
            math_source = r"47 \times 19 = 893 < 920 = 46 \times 20"
            math_source_codepoints = "-".join(
                f"{ord(char):x}" for char in math_source
            )
            user_html = (
                '<p>Keep literal $43 and render '
                '<span class="math-inline" '
                f'data-vmlx-math-source-codepoints="{math_source_codepoints}" '
                'data-vmlx-math-delimiter="single-dollar" '
                'data-vmlx-math-display-mode="inline">'
                '<span class="katex">47 × 19 = 893 &lt; 920 = 46 × 20</span>'
                '</span>.</p>'
            )
            source_user_messages.append(
                {
                    "messageId": f"u{turn}",
                    "role": "user",
                    "visible": True,
                    "text": (
                        "Keep literal $43 and render "
                        "47 × 19 = 893 < 920 = 46 × 20."
                    ),
                    "html": user_html,
                    "katexCount": 1,
                    "katexErrorCount": 0,
                    "currencyOccurrences": [
                        {"text": "$43", "insideKatex": False}
                    ],
                }
            )
        else:
            request_content = (
                f"Reply exactly {content}"
                if turn < 3
                else (
                    "Include UI-V5-3, literal $43, and "
                    r"inline \(47 \times 19\)."
                )
            )
            source_user_messages.append(
                {
                    "messageId": f"u{turn}",
                    "role": "user",
                    "visible": True,
                    "text": request_content,
                    "html": f"<p>{request_content}</p>",
                    "katexCount": 0,
                    "katexErrorCount": 0,
                    "currencyOccurrences": [],
                }
            )
        events = [
            {"seq": 0, "type": "reasoning_delta", "text": reasoning},
        ]
        if turn < 3:
            events.extend(
                [
                    {
                        "seq": 1,
                        "type": "tool_call",
                        "call_id": f"ui-c{turn}",
                        "name": "file_info" if turn == 1 else "run_command",
                        "arguments": (
                            {"path": "panel/package.json"}
                            if turn == 1
                            else {"command": "pwd"}
                        ),
                    },
                    {
                        "seq": 2,
                        "type": "tool_result",
                        "call_id": f"ui-c{turn}",
                        "content": "ok",
                    },
                    {"seq": 3, "type": "content_delta", "text": content},
                    {
                        "seq": 4,
                        "type": "terminal",
                        "status": "completed",
                        "response_id": response_id,
                        "ttft_ms": 100.0,
                        "decode_tps": 50.0,
                    },
                ]
            )
        else:
            events.extend(
                [
                    {"seq": 1, "type": "content_delta", "text": content},
                    {
                        "seq": 2,
                        "type": "terminal",
                        "status": "completed",
                        "response_id": response_id,
                        "ttft_ms": 100.0,
                        "decode_tps": 50.0,
                    },
                ]
            )
        turns.append(
            {
                "request_b64": _fixture_json_b64(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": request_content,
                            }
                        ],
                        "message_id": f"m{turn}",
                    }
                ),
                "events_b64": _fixture_jsonl_b64(events),
            }
        )
        dom_messages.append(
            {
                "reasoning_text": reasoning,
                "content_text": (
                    content
                    if turn < 3
                    else "UI-V5-3 $43 47 × 19"
                ),
                "terminal_text": "Completed",
                "ttft_ms": 100.0,
                "decode_tps": 50.0,
                "html": (
                    f"<strong>UI-V5-{turn}</strong>"
                    + (
                        ""
                        if turn < 3
                        else ' $43 <span class="katex">47 × 19</span>'
                    )
                ),
            }
        )
        source_records.append({"id": f"m{turn}", "content": content})
        source_reasoning.append([reasoning])
        source_ui_turns.append(
            {
                "turn": turn,
                "prompt": request_content,
                "proofRequestId": f"u{turn}",
                "terminalProofRequestId": f"u{turn}",
                "requestIds": [f"wire-{phase_index}-{turn}"],
                "userMessageId": f"u{turn}",
                "assistantMessageId": f"m{turn}",
                "terminalMessageId": f"m{turn}",
                "terminalResponseId": response_id,
                "logMatchMode": "exact_identity_ring_safe",
            }
        )
        resolved_sampling_records.append(
            {
                "route": "/v1/chat/completions",
                "model": "fixture-model",
                "proof_request_id": f"u{turn}",
                "request_id": f"wire-{phase_index}-{turn}",
                "message_id": f"m{turn}",
                "correlation_source": "server_emitted",
                "values": {},
            }
        )
        request_correlation_turns.append(
            {
                "turn": turn,
                "proofRequestId": f"u{turn}",
                "userMessageId": f"u{turn}",
                "assistantMessageId": f"m{turn}",
                "serverProofRequestId": f"u{turn}",
                "serverRequestIds": [f"wire-{phase_index}-{turn}"],
                "serverMessageId": f"m{turn}",
                "resolvedLogCorrelated": True,
            }
        )
        health = {
            "scheduler": {
                "last_cache_execution": {
                    "request_id": response_id,
                    "cache_reuse_applied": turn > 1 or turn_count == 1,
                    "cache_outcome": (
                        "hit" if turn > 1 or turn_count == 1 else "miss"
                    ),
                    "prompt_tokens": 128,
                    "cached_tokens": 64 if turn > 1 or turn_count == 1 else 0,
                    "uncached_prompt_tokens": (
                        64 if turn > 1 or turn_count == 1 else 128
                    ),
                    "prefill_tokens": (
                        64 if turn > 1 or turn_count == 1 else 128
                    ),
                }
            }
        }
        artifact_value = {
            "schema": "vmlx-ui-turn-health-cache-execution-v1",
            "run_id": run_id,
            "turn": turn,
            "proof_request_id": f"{run_id}:ui:{turn}",
            "user_message_id": f"u{turn}",
            "assistant_message_id": f"m{turn}",
            "terminal_response_id": response_id,
            "correlation_status": "verified",
            "health": health,
        }
        artifact_bytes = json.dumps(
            artifact_value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        artifact_path = evidence_root / (
            f"ui-phase-{phase_index:02d}-turn-{turn}-health.json"
        )
        artifact_path.write_bytes(artifact_bytes)
        artifact_path.chmod(0o600)
        observation = {
            **health["scheduler"]["last_cache_execution"],
            "proof_request_id": f"{run_id}:ui:{turn}",
            "terminal_response_id": response_id,
            "message_id": f"m{turn}",
            "correlation_source": (
                "chat_complete_response_id_to_scheduler_"
                "last_cache_execution"
            ),
        }
        cache_rows.append(
            {
                "turn": turn,
                "proofRequestId": f"{run_id}:ui:{turn}",
                "userMessageId": f"u{turn}",
                "assistantMessageId": f"m{turn}",
                "terminalResponseId": response_id,
                "serverRequestId": response_id,
                "executionRequestId": response_id,
                "correlationStatus": "verified",
                "serverObservation": observation,
                "healthAfter": health,
                "healthArtifact": {
                    "path": str(artifact_path.resolve()),
                    "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
                    "size_bytes": len(artifact_bytes),
                },
            }
        )
    capture = {
        "schema": module.V5_UI_SCHEMA,
        "run_id": run_id,
        "nonce": nonce,
        "phase_index": phase["index"],
        "phase_name": phase["name"],
        "representative_id": phase["representative_id"],
        "ui_action_profile": phase["ui_action_profile"],
        "ui_turn_count": turn_count,
        "session_id": session_id,
        "source_proof_b64": _fixture_json_b64(
            {
                "format": module.ELECTRON_PROOF_SCHEMA,
                "run_id": run_id,
                "session": {"id": session_id},
                "uiStartControl": {"clicked": True},
                "requestContract": {"uiTurnCount": turn_count},
                "assistantRecords": source_records,
                "persistedReasoningByMessage": source_reasoning,
                "uiTurnEvidence": source_ui_turns,
                "resolvedSamplingRecords": resolved_sampling_records,
                "requestCorrelation": {
                    "status": "verified",
                    "turns": request_correlation_turns,
                },
                "cacheRequestEvidence": cache_rows,
                "cacheRequestCorrelation": {
                    "status": "verified",
                    "source": (
                        "chat:complete.responseId == health.scheduler."
                        "last_cache_execution.request_id"
                    ),
                },
                "renderedDom": {
                    "userMessages": source_user_messages,
                },
            }
        ),
        "interaction_b64": _fixture_json_b64(
            [
                {
                    "method": "Input.dispatchMouseEvent",
                    "selector": "button[data-action='start-session']",
                    "session_id": session_id,
                }
            ]
        ),
        "turns": turns,
    }
    dom = {
        "sourceCommit": "",
        "messages": dom_messages,
        "text": "Chats Server Tools Image API Today",
        "locale_catalog_source": "main_ipc_canonical_locale_json",
        "supported_locales": ["en", "zh", "ko", "ja", "es"],
        "picker_supported_locales": ["en", "zh", "ko", "ja", "es"],
        "visible_locale_options": ["en", "zh", "ko", "ja", "es"],
        "translation_key_count": 3,
        "catalog_translation_keys": [
            "app.mode.server",
            "common.stream",
            "sessions.start",
        ],
        "raw_translation_keys": [],
        "locales": [
            {
                "locale": locale,
                "selected_locale": locale,
                "raw_translation_keys": [],
            }
            for locale in ("en", "zh", "ko", "ja", "es")
        ],
        "viewport": {"width": 640, "scroll_width": 640},
        "viewport_restore": {
            "method": "Emulation.clearDeviceMetricsOverride",
            "verified": True,
            "prior": {"width": 1440, "height": 1000},
            "restored": {"width": 1440, "height": 1000},
        },
        "settings": {
            "new_session": {"temperature": 0.6, "top_p": 0.95},
            "override": {"temperature": 0.2, "top_p": 0.8},
            "after_restart": {"temperature": 0.2, "top_p": 0.8},
            "max_context_tokens": 32768,
            "max_output_tokens": 2048,
            "preview": {"temperature": 0.2},
            "argv": {"temperature": 0.2},
            "health": {"temperature": 0.2},
        },
    }
    return capture, dom


def _fixture_cache_capture(
    module,
    run_id: str,
    nonce: str,
    *,
    bundle_fingerprint: str = "f" * 64,
    native_bundle_fingerprint: str = "e" * 64,
    model: str = "fixture-model",
    native_model: str = "fixture-native-model",
    session_id: str = "fixture-held-session",
    native_session_id: str = "fixture-native-session",
) -> dict:
    phases = []
    store_hashes: dict[tuple[str, bool, str], str] = {}
    for phase in module.V5_CACHE_PHASES:
        paged = phase["paged_ram"]
        operation = phase["operation"]
        gate_operation = module._v5_cache_gate_operation(phase)
        cache_policy = phase["cache_policy"]
        representative_id = phase["representative_id"]
        is_native = representative_id == module.V5_NATIVE_REPRESENTATIVE_ID
        phase_model = native_model if is_native else model
        phase_bundle_fingerprint = (
            native_bundle_fingerprint if is_native else bundle_fingerprint
        )
        phase_session_id = native_session_id if is_native else session_id
        backend_pid = 9001 + phase["index"]
        tq_enabled = cache_policy == "q4"
        topology = {
            "schema": "vmlx-cache-topology-v1",
            "configured": {
                "use_paged_cache": paged,
                "kv_cache_quantization": "q4" if tq_enabled else "none",
                "kv_cache_quantization_explicit": cache_policy != "q4",
            },
            "instantiated": {
                "paged_ram_enabled": paged,
                "block_disk_l2": True,
                "block_disk_max_size_bytes": 1024,
            },
            "turboquant_kv_cache": {
                "enabled": tq_enabled,
                "storage_key_bits": 4 if tq_enabled else 0,
                "storage_value_bits": 4 if tq_enabled else 0,
            },
            "kv_cache_quantization": {"enabled": tq_enabled},
            "native_cache": {
                "family": "deepseek_v4" if is_native else "standard_kv",
                "cache_type": (
                    "native_composite" if is_native else "standard_kv"
                ),
                "schema": (
                    "deepseek_v4_v10_delta" if is_native else "standard_v1"
                ),
                "components": (
                    [
                        "swa_local",
                        "csa_compressed_pool",
                        "hca_compressed_pool",
                        "incomplete_tail_state",
                    ]
                    if is_native
                    else []
                ),
                "generic_turboquant_kv": (
                    {
                        "enabled": False,
                        "reason": "native_dsv4_composite",
                    }
                    if is_native
                    else {"enabled": tq_enabled}
                ),
                "pool_quant": (
                    {
                        "requested": True,
                        "observed": True,
                        "enabled": True,
                        "matches_request": True,
                        "error": None,
                    }
                    if is_native
                    else None
                ),
            },
        }
        topology_attestation = {
            "schema": "vmlx-cache-topology-attestation-v1",
            "configuration": topology,
            "canonical_sha256": hashlib.sha256(
                json.dumps(
                    topology,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }
        topology_attestation["fingerprint_sha256"] = topology_attestation[
            "canonical_sha256"
        ]
        partial_tag = (
            "partial_b" if gate_operation == "store" else "restart_partial_c"
        )
        partial_selector = "B" if gate_operation == "store" else "C"
        disk_refault = gate_operation == "probe"
        partial_execution = {
            "prompt_tokens": 128,
            "cached_tokens": 112,
            "uncached_prompt_tokens": 16,
            "prefill_tokens": 16,
            "disk_blocks": 7 if disk_refault else 0,
            "cache_detail": (
                "paged+tq+disk" if disk_refault else "paged+tq"
            ),
        }
        requests = [] if phase["index"] == 3 else [
            {
                "tag": partial_tag,
                "cache_contract_ok": True,
                "cached_tokens": 112,
                "health_counter_deltas": {
                    "scheduler_cache.evictions": 0,
                    "block_disk_cache.disk_evictions": 0,
                },
                "last_cache_execution": partial_execution,
            }
        ]
        if gate_operation == "store":
            requests.insert(
                0,
                {
                    "tag": "warm_a",
                    "cache_contract_ok": True,
                    "cached_tokens": 127,
                    "health_counter_deltas": {},
                },
            )
        encode_count = (
            2
            if tq_enabled and gate_operation == "store"
            else 0
        )
        decode_count = (
            2
            if tq_enabled
            and (gate_operation == "probe" or phase["index"] == 2)
            else 0
        )

        def codec_runtime(
            encode_calls: int,
            decode_calls: int,
        ) -> dict:
            def operation_row(
                operation: str,
                calls: int,
            ) -> dict:
                return {
                    "calls": calls,
                    "blocks": calls,
                    "tokens": calls * 64,
                    "last_event": (
                        {
                            "sequence": (
                                encode_calls
                                if operation == "encode"
                                else encode_calls + decode_calls
                            ),
                            "boundary": (
                                "encode_tq_block"
                                if operation == "encode"
                                else "decode_tq_entries"
                            ),
                            "blocks": 1,
                            "tokens": 64,
                            "key_bits_values": [4],
                            "value_bits_values": [4],
                        }
                        if calls
                        else None
                    ),
                }

            return {
                "schema": "vmlx-cache-storage-runtime-telemetry-v1",
                "turboquant_block_codec": {
                    "schema": "vmlx-tq-block-codec-v1",
                    "encode": operation_row("encode", encode_calls),
                    "decode": operation_row("decode", decode_calls),
                },
            }
        summary = {
            "schema": module.CACHE_PROOF_SCHEMA,
            "phase": gate_operation,
            "nonce": nonce,
            "base_url": "http://127.0.0.1:8001",
            "model": phase_model,
            "cache_contract_profile": (
                "deepseek_v4_native_delta" if is_native else "generic"
            ),
            "gate_ok": True,
            "scenario_contract_ok": True,
            "probe_linkage_ok": (
                True if gate_operation == "probe" else None
            ),
            "identity": {
                "observed_engine": {"pid": backend_pid},
                "model_bundle_provenance": {
                    "fingerprint_sha256": phase_bundle_fingerprint
                },
                "cache_topology_provenance": topology_attestation,
            },
            "tokenizer_lcp_contract": {
                "longest_common_prefix_tokens": {
                    "A:A": 128,
                    f"A:{partial_selector}": 127,
                }
            },
            "requests": requests,
            "health_before": {
                "cache_storage_runtime_telemetry": codec_runtime(0, 0),
            },
            "health_final": {
                "cache_storage_runtime_telemetry": codec_runtime(
                    encode_count,
                    decode_count,
                ),
                "cache": {
                    "block_disk_cache": {
                        "disk_size_bytes": 512,
                    }
                }
            },
        }
        if phase["index"] == 2:
            summary["l2_size_eviction_observation"] = {
                "schema": module.V5_L2_EVICTION_OBSERVATION_SCHEMA,
                "saved_max_bytes": 1024,
                "peak_observed_bytes": 1024,
                "final_observed_bytes": 512,
                "bounded_filler_request_count": 4,
                "old_prefix_fingerprint_sha256": "1" * 64,
                "recent_prefix_fingerprint_sha256": "2" * 64,
                "old_prefix_evicted": True,
                "recent_prefix_present": True,
                "recent_prefix_last_access_after_old": True,
                "recent_before": {
                    "l1": {
                        "terminal_resident_payload_present": True,
                    }
                },
                "recent_pre_refault": {
                    "l1": {
                        "terminal_resident_payload_present": False,
                    }
                },
                "evicting_filler_fence": {
                    "request_correlated": True,
                    "post_eviction_complete": True,
                    "fence_sealed": True,
                    "disk_evictions_delta": 1,
                },
                "recent_refault_execution": {
                    "response_id": "resp-phase2-refault",
                    "response_id_consistent": True,
                    "status_code": 200,
                    "terminal_ok": True,
                    "marker_ok": True,
                    "cached_tokens": 112,
                    "cache_detail": {
                        "source": "paged+disk+tq-native",
                    },
                    "last_cache_execution": {
                        **partial_execution,
                        "cache_reuse_applied": True,
                        "cache_outcome": "hit",
                        "disk_blocks": 7,
                        "cache_detail": "paged+disk+tq-native",
                    },
                },
            }
        if phase["index"] == 3:
            summary["l2_restart_restore_observation"] = {
                "schema": module.V5_L2_RESTART_OBSERVATION_SCHEMA,
                "restart_probe_prefix_fingerprint_sha256": "2" * 64,
                "restart_restored_tokens": 112,
                "restart_disk_blocks": 7,
                "restart_uncached_tokens": 16,
                "restart_restore_source": "block-disk",
                "restart_pre": {
                    "longest_common_prefix_tokens": 127,
                },
                "restart_execution": {
                    "last_cache_execution": partial_execution,
                },
            }
        summary_bytes = json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        store_key = (representative_id, paged, cache_policy)
        if gate_operation == "store":
            store_hashes[store_key] = hashlib.sha256(summary_bytes).hexdigest()
        phases.append(
            {
                "phase_index": phase["index"],
                "phase_name": phase["name"],
                "representative_id": representative_id,
                "bundle_role": phase["bundle_role"],
                "cache_policy": cache_policy,
                "kv_cache_quantization": phase["kv_cache_quantization"],
                "tq_policy": phase["tq_policy"],
                "session_policy": phase["session_policy"],
                "operation": operation,
                "ui_action_profile": phase["ui_action_profile"],
                "ui_turn_count": phase["ui_turn_count"],
                "api_action_profile": phase["api_action_profile"],
                "paged_ram": paged,
                "model": phase_model,
                "bundle_fingerprint_sha256": phase_bundle_fingerprint,
                "session_id": phase_session_id,
                "backend_pid": backend_pid,
                "session_binding_sha256": str(phase["index"]) * 64,
                "summary_b64": _fixture_b64(summary_bytes),
                "linked_store_summary_sha256": (
                    store_hashes[store_key]
                    if gate_operation == "probe"
                    else None
                ),
                "artifact_manifest_b64": _fixture_json_b64(
                    [
                        {
                            "relative_path": "summary.json",
                            "sha256": hashlib.sha256(summary_bytes).hexdigest(),
                            "size": len(summary_bytes),
                        }
                    ]
                ),
            }
        )
    phase2_summary = json.loads(base64.b64decode(phases[2]["summary_b64"]))
    phase3_summary = json.loads(base64.b64decode(phases[3]["summary_b64"]))
    capture = {
        "schema": module.V5_CACHE_SCHEMA,
        "run_id": run_id,
        "nonce": nonce,
        "session_id": native_session_id,
        "phases": phases,
    }
    capture["l2_size_eviction_attestation"] = (
        module._v5_derive_l2_size_eviction_attestation(
            run_id=run_id,
            nonce=nonce,
            phase2_summary=phase2_summary,
            phase2_summary_sha256=hashlib.sha256(
                base64.b64decode(phases[2]["summary_b64"])
            ).hexdigest(),
            phase3_summary=phase3_summary,
            phase3_summary_sha256=hashlib.sha256(
                base64.b64decode(phases[3]["summary_b64"])
            ).hexdigest(),
        )
    )
    return capture


def load_module():
    spec = importlib.util.spec_from_file_location("scoped_release_preflight_18", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expected_visible_final_extracts_literal_three_line_contract():
    module = load_module()
    expected = (
        "Third UI turn: REAL_UI_LIVE_TOOL_ONE REAL_UI_LIVE_TOOL_TWO\n"
        "Currency: $43\n"
        r"Math: \(2 + 2 = 4\)"
    )
    request = {
        "messages": [
            {
                "role": "user",
                "content": "\n".join(
                    (
                        "Do not call another tool.",
                        "Your complete visible answer must be exactly the following "
                        "three lines and nothing else.",
                        "Copy every character literally, including the dollar sign "
                        "and both backslashes:",
                        expected,
                    )
                ),
            }
        ]
    }

    assert module._expected_visible_final(request) == expected


def test_expected_visible_final_does_not_misread_exactly_as_the_answer():
    module = load_module()
    request = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Your visible answer must be exactly the following block and "
                    "nothing else."
                ),
            }
        ]
    }

    assert module._expected_visible_final(request) == ""


def _fixture_v5_jang_source(tmp_path: Path) -> Path:
    root = tmp_path / "jang-source"
    package = root / "jang_tools"
    tests = root / "tests"
    package.mkdir(parents=True)
    tests.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "jang"\nversion = "2.5.39"\n',
        encoding="utf-8",
    )
    (package / "__init__.py").write_text(
        '__version__ = "2.5.39"\n',
        encoding="utf-8",
    )
    (tests / "test_laguna_jang_affine_policy.py").write_text(
        "def test_fixture():\n    assert True\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "add", "pyproject.toml", "jang_tools", "tests"],
        check=True,
    )
    return root


def test_v5_jang_import_uses_public_distribution_metadata_name(tmp_path: Path):
    module = load_module()
    (tmp_path / "run").mkdir()
    jang_source = _fixture_v5_jang_source(tmp_path)
    plans = module._v5_default_owned_check_plans(
        tmp_path / "run",
        jang_source,
    )
    command = next(
        row
        for row in plans["jang_runtime_provenance"]["commands"]
        if row["command_id"] == "jang_import"
    )
    import_script = command["argv"][-1]
    assert "metadata.version('jang')" in import_script
    assert "metadata.version('jang-tools')" not in import_script


def test_v5_jang_venv_plan_uses_symlinks_with_pinned_source_python(
    tmp_path: Path,
):
    module = load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    jang_source = _fixture_v5_jang_source(tmp_path)
    plans = module._v5_default_owned_check_plans(run_dir, jang_source)
    command = next(
        row
        for row in plans["jang_runtime_provenance"]["commands"]
        if row["command_id"] == "jang_venv"
    )
    python = ROOT / ".venv/bin/python"
    assert command["argv"] == [
        str(python),
        "-m",
        "venv",
        str(run_dir / "jang-installed"),
    ]
    assert "--copies" not in command["argv"]

    executable_pin, invocation = module._v5_pin_executable_invocation(
        Path(command["argv"][0]),
        run_dir,
    )
    assert invocation is not None
    assert invocation["path"] == str(python)
    assert invocation["target"] == os.readlink(python)
    assert invocation["resolved_path"] == executable_pin["path"]
    assert module._v5_executable_invocation_unchanged(invocation) is True


def test_v5_canonical_json_bytes_are_stable_and_callable():
    module = load_module()
    assert module._canonical_json_bytes({"z": 1, "a": "value"}) == (
        b'{"a":"value","z":1}'
    )


def test_v5_panel_owned_plans_launch_pinned_node_with_pinned_npm_cli(
    tmp_path: Path,
):
    module = load_module()
    (tmp_path / "run").mkdir()
    jang_source = _fixture_v5_jang_source(tmp_path)
    plans = module._v5_default_owned_check_plans(
        tmp_path / "run",
        jang_source,
    )
    expected_node = str(Path(shutil.which("node") or "").resolve())
    expected_npm_cli = str(Path(shutil.which("npm") or "").resolve())
    assert expected_node and expected_npm_cli
    for check_name in ("full_panel_suite", "typecheck", "production_build"):
        command = plans[check_name]["commands"][0]
        assert command["argv"][:2] == [expected_node, expected_npm_cli]
        fixed_bin = Path(command["path_prefix"])
        assert fixed_bin.parent == (tmp_path / "run")
        assert (fixed_bin / "node").is_file()
        assert (fixed_bin / "npm").is_file()
        assert (fixed_bin / "npx").is_file()
        assert (fixed_bin / "uv").is_file()
        assert str(fixed_bin / "node") in command["tool_files"]
        assert str(fixed_bin / "npm") in command["tool_files"]
        assert str(fixed_bin / "npx") in command["tool_files"]
        assert str(fixed_bin / "uv") in command["tool_files"]
    python_suite = plans["full_python_suite"]["commands"][0]
    assert python_suite["path_prefix"] == plans["full_panel_suite"]["commands"][
        0
    ]["path_prefix"]
    assert str(Path(python_suite["path_prefix"]) / "node") in python_suite[
        "tool_files"
    ]
    assert str(Path(python_suite["path_prefix"]) / "npx") in python_suite[
        "tool_files"
    ]
    assert str(Path(python_suite["path_prefix"]) / "uv") in python_suite[
        "tool_files"
    ]
    assert python_suite["env"] == {
        "VMLX_JANG_TOOLS_SOURCE": str(jang_source.resolve()),
        "VMLINUX_JANG_TOOLS_SOURCE": str(jang_source.resolve()),
        "VMLINUX_R20_PINNED_NODE_REALPATH": expected_node,
        "VMLINUX_R20_PINNED_NPM_CLI_REALPATH": expected_npm_cli,
    }
    production = plans["production_build"]["commands"][0]
    assert production["env"]["VMLX_RELEASE_SCOPE"] == "r20_production"
    assert production["env"]["VMLX_JANG_TOOLS_SOURCE"] == str(
        jang_source.resolve()
    )
    assert production["env"]["VMLINUX_JANG_TOOLS_SOURCE"] == str(
        jang_source.resolve()
    )
    assert production["env"]["VMLX_BUNDLE_MLX_PLATFORM"] == "compat"
    assert (
        production["env"]["VMLX_EXPECTED_MLX_WHEEL_PLATFORM"]
        == "macosx_14_0_arm64"
    )
    for name in ("NODE", "GIT", "SHASUM", "AWK", "FIND"):
        assert production["env"][f"VMLX_R20_TOOL_{name}_REALPATH"]
        assert len(production["env"][f"VMLX_R20_TOOL_{name}_SHA256"]) == 64


def test_v5_jang_tests_use_installed_wheel_and_compare_package_manifest(
    tmp_path: Path,
):
    module = load_module()
    (tmp_path / "run").mkdir()
    jang_source = _fixture_v5_jang_source(tmp_path)
    plans = module._v5_default_owned_check_plans(
        tmp_path / "run",
        jang_source,
    )
    test_command = next(
        row
        for row in plans["jang_runtime_provenance"]["commands"]
        if row["command_id"] == "jang_test"
    )
    assert test_command["argv"][0] == str(ROOT / ".venv/bin/python")
    assert test_command["argv"][1:3] == ["-I", "-c"]
    test_script = test_command["argv"][test_command["argv"].index("-c") + 1]
    assert test_script.index("sys.path.insert") < test_script.index(
        "import jang_tools"
    )
    assert "VMLINUX_TEST_IMPORT=" in test_script
    assert "source_manifest_sha256" in test_script
    assert "installed_manifest_sha256" in test_script
    assert "laguna_mixed_affine_shape_bits" in test_script
    assert "test_laguna_jang_affine_policy.py" in test_script
    assert "--import-mode=importlib" in test_script
    pinned_test_files = set(test_command["tool_files"])
    assert str((ROOT / "tests/test_laguna_loader.py").resolve()) in pinned_test_files
    assert str((jang_source / "jang_tools/__init__.py").resolve()) in pinned_test_files


def test_v5_owned_command_preserves_authoritative_venv_invocation(tmp_path: Path):
    module = load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)
    python = ROOT / ".venv/bin/python"
    result = module._v5_run_command(
        "jang_runtime_provenance",
        {
            "command_id": "venv-python",
            "argv": [
                str(python),
                "-I",
                "-c",
                "import numpy,sys;print(sys.executable);print(numpy.__file__)",
            ],
            "cwd": tmp_path,
            "env": {},
        },
        {"run_id": "venv-python", "nonce": "5" * 32},
        run_dir,
    )
    assert result["exit_code"] == 0
    assert result["executable_invocation"]["path"] == str(python)
    assert str(ROOT / ".venv/lib") in result["__stdout_bytes"].decode()


def test_v5_owned_command_removes_only_its_exclusive_temporary_payload(
    tmp_path: Path,
):
    module = load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)
    retained = run_dir / "retained-release-evidence"
    retained.write_text("keep", encoding="utf-8")
    python = ROOT / ".venv/bin/python"
    result = module._v5_run_command(
        "full_python_suite",
        {
            "command_id": "large-fixture-cleanup",
            "argv": [
                str(python),
                "-I",
                "-c",
                (
                    "import os,pathlib;"
                    "p=pathlib.Path(os.environ['TMPDIR']);"
                    "(p/'nested').mkdir();"
                    "(p/'nested'/'fixture.bin').write_bytes(b'x'*1048576);"
                    "print(p)"
                ),
            ],
            "cwd": tmp_path,
            "env": {},
        },
        {"run_id": "tmp-cleanup", "nonce": "6" * 32},
        run_dir,
    )
    child_tmpdir = Path(result["__stdout_bytes"].decode().strip())
    assert result["exit_code"] == 0
    assert result["temporary_directory_removed"] is True
    assert not child_tmpdir.exists()
    assert retained.read_text(encoding="utf-8") == "keep"


def test_v5_owned_tmp_cleanup_retries_transient_nonempty_directory(
    tmp_path: Path,
    monkeypatch,
):
    module = load_module()
    owned = tmp_path / "owned"
    owned.mkdir()
    (owned / ".DS_Store").write_bytes(b"metadata")
    real_rmtree = module.shutil.rmtree
    calls = 0

    def transient_rmtree(path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(errno.ENOTEMPTY, "Directory not empty", str(path))
        real_rmtree(path)

    monkeypatch.setattr(module.shutil, "rmtree", transient_rmtree)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    module._v5_remove_reserved_tmp_entry(owned)

    assert calls == 2
    assert not owned.exists()


def test_v5_owned_tmp_cleanup_rejects_persistent_nonempty_directory(
    tmp_path: Path,
    monkeypatch,
):
    module = load_module()
    owned = tmp_path / "owned"
    owned.mkdir()
    calls = 0

    def persistent_rmtree(path):
        nonlocal calls
        calls += 1
        raise OSError(errno.ENOTEMPTY, "Directory not empty", str(path))

    monkeypatch.setattr(module.shutil, "rmtree", persistent_rmtree)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    with pytest.raises(OSError) as error:
        module._v5_remove_reserved_tmp_entry(owned)

    assert error.value.errno == errno.ENOTEMPTY
    assert calls == 8


def test_v5_owned_command_removes_temporary_payload_after_nonzero_exit(
    tmp_path: Path,
):
    module = load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)
    python = ROOT / ".venv/bin/python"
    result = module._v5_run_command(
        "full_python_suite",
        {
            "command_id": "failed-fixture-cleanup",
            "argv": [
                str(python),
                "-I",
                "-c",
                (
                    "import os,pathlib,sys;"
                    "p=pathlib.Path(os.environ['TMPDIR']);"
                    "(p/'failed.bin').write_bytes(b'x'*1048576);"
                    "print(p);sys.exit(7)"
                ),
            ],
            "cwd": tmp_path,
            "env": {},
        },
        {"run_id": "tmp-cleanup-fail", "nonce": "7" * 32},
        run_dir,
    )
    child_tmpdir = Path(result["__stdout_bytes"].decode().strip())
    assert result["exit_code"] == 7
    assert result["temporary_directory_removed"] is True
    assert not child_tmpdir.exists()


def test_v5_owned_command_removes_temporary_payload_after_setup_exception(
    tmp_path: Path,
):
    module = load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)
    python = ROOT / ".venv/bin/python"
    with pytest.raises(ValueError, match="PATH prefix is unavailable"):
        module._v5_run_command(
            "full_python_suite",
            {
                "command_id": "setup-failure-cleanup",
                "argv": [str(python), "-I", "-c", "print('must not run')"],
                "cwd": tmp_path,
                "env": {},
                "path_prefix": str(tmp_path / "missing-path-prefix"),
            },
            {"run_id": "tmp-setup-fail", "nonce": "8" * 32},
            run_dir,
        )
    assert list(run_dir.glob("tmp-*")) == []


def test_v5_owned_command_scrubs_and_rejects_renamed_temporary_payload(
    tmp_path: Path,
):
    module = load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)
    python = ROOT / ".venv/bin/python"
    with pytest.raises(
        RuntimeError,
        match="temporary directory identity changed: relocated",
    ):
        module._v5_run_command(
            "full_python_suite",
            {
                "command_id": "renamed-fixture-cleanup",
                "argv": [
                    str(python),
                    "-I",
                    "-c",
                    (
                        "import os,pathlib;"
                        "p=pathlib.Path(os.environ['TMPDIR']);"
                        "m=p.with_name(p.name+'-relocated');"
                        "p.rename(m);"
                        "(m/'nested').mkdir();"
                        "(m/'nested'/'payload.bin').write_bytes(b'x'*1048576);"
                        "print(m)"
                    ),
                ],
                "cwd": tmp_path,
                "env": {},
            },
            {"run_id": "tmp-rename-fail", "nonce": "9" * 32},
            run_dir,
        )
    assert list(run_dir.glob("tmp-*")) == []


def test_v5_owned_command_pins_executable_script_argument_even_when_mode_0755(
    tmp_path: Path,
):
    module = load_module()
    node = _v5_pinned_tool_path(
        "node",
        "VMLINUX_R20_PINNED_NODE_REALPATH",
    )
    script = tmp_path / "npm-cli.js"
    script.write_text(
        "console.log('owned npm cli'); setTimeout(() => {}, 500)\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)
    result = module._v5_run_command(
        "full_panel_suite",
        {
            "command_id": "node-script-identity",
            "argv": [str(node), str(script)],
            "cwd": tmp_path,
            "env": {},
        },
        {"run_id": "node-script", "nonce": "1" * 32},
        run_dir,
    )
    assert result["exit_code"] == 0
    assert result["executable"]["path"] == str(node)
    assert [row["path"] for row in result["scripts"]] == [str(script)]
    assert b"owned npm cli" in result["__stdout_bytes"]


def test_v5_pin_accepts_readonly_system_hardlink_only_with_explicit_policy():
    module = load_module()
    system_tool = Path("/usr/bin/git")
    assert system_tool.stat().st_nlink > 1
    assert os.statvfs(system_tool).f_flag & os.ST_RDONLY
    with pytest.raises(ValueError, match="unsafe pinned file"):
        module._v5_pin_regular_file(system_tool, executable=True)
    pin = module._v5_pin_regular_file(
        system_tool,
        executable=True,
        allow_readonly_system_hardlink=True,
    )
    assert pin["readonly_system_hardlink"] is True
    assert module._v5_pin_unchanged(pin, executable=True) is True


def test_v5_owned_command_can_pin_readonly_system_tool_file(tmp_path: Path):
    module = load_module()
    node = _v5_pinned_tool_path(
        "node",
        "VMLINUX_R20_PINNED_NODE_REALPATH",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)
    result = module._v5_run_command(
        "production_build",
        {
            "command_id": "readonly-system-tool",
            "argv": [
                str(node),
                "-e",
                "process.stdout.write('ok'); setTimeout(() => {}, 500)",
            ],
            "cwd": tmp_path,
            "env": {},
            "tool_files": ["/usr/bin/git"],
        },
        {"run_id": "readonly-system-tool", "nonce": "4" * 32},
        run_dir,
    )
    assert result["exit_code"] == 0
    system_pin = next(
        row for row in result["scripts"] if row["path"] == "/usr/bin/git"
    )
    assert system_pin["readonly_system_hardlink"] is True


def test_v5_pin_rejects_writable_hardlink_even_with_system_policy(tmp_path: Path):
    module = load_module()
    source = tmp_path / "tool"
    alias = tmp_path / "tool-alias"
    source.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    source.chmod(0o700)
    os.link(source, alias)
    with pytest.raises(ValueError, match="unsafe pinned file"):
        module._v5_pin_regular_file(
            alias,
            executable=True,
            allow_readonly_system_hardlink=True,
        )


def test_v5_owned_command_rejects_executable_script_tampering(
    tmp_path: Path,
):
    module = load_module()
    node = _v5_pinned_tool_path(
        "node",
        "VMLINUX_R20_PINNED_NODE_REALPATH",
    )
    script = tmp_path / "npm-cli.js"
    script.write_text(
        "require('fs').appendFileSync(__filename, '\\n// changed'); "
        "setTimeout(() => {}, 500)\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)
    with pytest.raises(RuntimeError, match="executable or script changed"):
        module._v5_run_command(
            "full_panel_suite",
            {
                "command_id": "node-script-tamper",
                "argv": [str(node), str(script)],
                "cwd": tmp_path,
                "env": {},
            },
            {"run_id": "node-script", "nonce": "2" * 32},
            run_dir,
        )


def test_v5_owned_node_path_supports_nested_env_node_without_ambient_path(
    tmp_path: Path,
    monkeypatch,
):
    module = load_module()
    node_value = _v5_pinned_tool_path(
        "node",
        "VMLINUX_R20_PINNED_NODE_REALPATH",
    )
    npm_value = _v5_pinned_tool_path(
        "npm",
        "VMLINUX_R20_PINNED_NPM_CLI_REALPATH",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)
    node, npm_cli, fixed_bin, toolchain_pins = (
        module._v5_prepare_node_toolchain(
            run_dir,
            node_path=node_value,
            npm_cli_path=npm_value,
            bin_name="nested-node-bin",
        )
    )
    package = tmp_path / "package"
    probe = package / "node_modules/.bin/probe"
    probe.parent.mkdir(parents=True)
    probe.write_text(
        "#!/usr/bin/env node\n"
        "setTimeout(() => console.log('nested env node passed'), 500)\n",
        encoding="utf-8",
    )
    probe.chmod(0o755)
    (package / "package.json").write_text(
        json.dumps(
            {
                "name": "v5-node-probe",
                "version": "1.0.0",
                "scripts": {"probe": "probe"},
            }
        ),
        encoding="utf-8",
    )
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "node").write_text("poisoned\n", encoding="utf-8")
    monkeypatch.setenv("PATH", str(poison))
    result = module._v5_run_command(
        "full_panel_suite",
        {
            "command_id": "nested-env-node",
            "argv": [str(node), str(npm_cli), "run", "probe"],
            "cwd": package,
            "env": {},
            "path_prefix": str(fixed_bin),
            "tool_files": [pin["path"] for pin in toolchain_pins],
        },
        {"run_id": "nested-env-node", "nonce": "3" * 32},
        run_dir,
    )
    assert result["exit_code"] == 0
    assert b"nested env node passed" in result["__stdout_bytes"]
    recorded = {row["path"] for row in result["scripts"]}
    assert str((fixed_bin / "node").resolve()) in recorded
    assert str((fixed_bin / "npm").resolve()) in recorded


def test_v5_process_observation_retries_only_until_exact_pin(monkeypatch):
    module = load_module()
    expected = {
        "path": "/opt/tool/node",
        "sha256": "a" * 64,
    }
    exact = {
        "pid": 123,
        "start_identity": "stable-start",
        "executable_path": expected["path"],
        "executable_sha256": expected["sha256"],
    }
    observations = iter(
        [
            None,
            {
                "pid": 123,
                "start_identity": "spawn-transition",
                "executable_path": "/usr/bin/false",
                "executable_sha256": "b" * 64,
            },
            exact,
        ]
    )
    monkeypatch.setattr(module, "_observe_process", lambda _pid: next(observations))
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    assert (
        module._v5_observe_matching_process(
            123,
            expected,
            attempts=3,
            delay_seconds=0,
        )
        == exact
    )


def test_v5_process_observation_stays_fail_closed_after_retry(monkeypatch):
    module = load_module()
    mismatch = {
        "pid": 123,
        "start_identity": "wrong-start",
        "executable_path": "/usr/bin/false",
        "executable_sha256": "b" * 64,
    }
    monkeypatch.setattr(module, "_observe_process", lambda _pid: mismatch)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    assert (
        module._v5_observe_matching_process(
            123,
            {"path": "/opt/tool/node", "sha256": "a" * 64},
            attempts=3,
            delay_seconds=0,
        )
        == mismatch
    )


def test_v5_panel_summary_accepts_green_accounting_with_failure_words_in_logs():
    module = load_module()
    output = (
        b"target failed preflight during the expected negative fixture\n"
        b"Failed to start session; proxy error shown to user\n"
        b"\x1b[2m Test Files \x1b[22m \x1b[32m91 passed\x1b[39m (91)\n"
        b"\x1b[2m Tests \x1b[22m \x1b[32m2631 passed\x1b[39m | "
        b"\x1b[33m3 skipped\x1b[39m (2634)\n"
    )
    facts, _ = module._v5_owned_check_facts(
        "full_panel_suite",
        [{"exit_code": 0, "__stdout_bytes": output, "__stderr_bytes": b""}],
        {},
        {},
    )
    assert facts == set(module.V5_RELEASE_ASSERTIONS["full_panel_suite"])


def test_v5_panel_summary_rejects_terminal_failure_and_bad_skip_accounting():
    module = load_module()

    def facts_for(summary: bytes) -> set[str]:
        facts, _ = module._v5_owned_check_facts(
            "full_panel_suite",
            [{"exit_code": 0, "__stdout_bytes": summary, "__stderr_bytes": b""}],
            {},
            {},
        )
        return facts

    assert not facts_for(
        b"Test Files 1 failed | 90 passed (91)\n"
        b"Tests 1 failed | 2630 passed | 3 skipped (2634)\n"
    )
    assert not facts_for(
        b"Test Files 91 passed (91)\n"
        b"Tests 2631 passed | 3 skipped (2633)\n"
    )


def test_v5_production_build_accepts_actual_electron_vite_contract(
    tmp_path: Path,
):
    module = load_module()
    output_root = tmp_path / "electron-build"
    for relative in (
        "main/index.mjs",
        "preload/index.js",
        "renderer/index.html",
    ):
        path = output_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture {relative}\n", encoding="utf-8")
    terminal = (
        b"ok bundled JANG provenance matches source (2.5.39 @ efde446)\n"
        b"bundled-python: all critical imports ok\n"
    )
    facts, details = module._v5_owned_check_facts(
        "production_build",
        [{"exit_code": 0, "__stdout_bytes": terminal, "__stderr_bytes": b""}],
        {
            "output_root": str(output_root),
            "required_outputs": (
                "main/index.mjs",
                "preload/index.js",
                "renderer/index.html",
            ),
        },
        {},
    )
    assert facts == set(module.V5_RELEASE_ASSERTIONS["production_build"])
    assert details["output"]["tree_sha256"]


def source_payload() -> dict[str, str]:
    return {"source_commit": "b" * 40, "source_tree": "c" * 40}


def source_block() -> dict[str, object]:
    source = source_payload()
    return {
        "commit": source["source_commit"],
        "tree": source["source_tree"],
        "clean": True,
    }


def bundle_attestation(module, root: Path) -> tuple[dict, dict[str, str]]:
    root.mkdir(parents=True, exist_ok=True)
    contents = {
        "config.json": '{"model_type":"laguna","architectures":["LagunaForCausalLM"]}\n',
        "generation_config.json": '{"temperature":0.6,"top_p":0.95}\n',
        "jang_config.json": '{"profile":"JANG_2L","quantization":"affine"}\n',
        "tokenizer_config.json": '{"chat_template":"{{ messages }}"}\n',
        "chat_template.jinja": "{{ messages }}\n",
    }
    for name, content in contents.items():
        path = root / name
        path.write_text(content, encoding="utf-8")
    snapshot = module._read_bundle_directory_snapshot(root.resolve())
    assert snapshot is not None
    observed = {
        key: snapshot[key]
        for key in (
            "schema",
            "model_bundle_path",
            "directory_identity",
            "files",
            "fingerprint_sha256",
            "derived",
        )
    }
    hashes = {
        name: row["sha256"] for name, row in snapshot["files"].items()
    }
    return observed, hashes


def native_bundle_attestation(module, root: Path) -> tuple[dict, dict[str, str]]:
    root.mkdir(parents=True, exist_ok=True)
    contents = {
        "config.json": (
            '{"model_type":"deepseek_v4",'
            '"architectures":["DeepseekV4ForCausalLM"]}\n'
        ),
        "generation_config.json": (
            '{"temperature":1.0,"top_p":0.95,"top_k":0}\n'
        ),
        "jang_config.json": (
            '{"profile":"JANG","quantization":"affine","chat":{'
            '"encoder":"encoding_dsv4",'
            '"encoder_fn":"encode_messages",'
            '"chat_template_source":"official_python_encoder"}}\n'
        ),
        "tokenizer_config.json": '{}\n',
    }
    for name, content in contents.items():
        path = root / name
        path.write_text(content, encoding="utf-8")
    encoder = root / "encoding" / "encoding_dsv4.py"
    encoder.parent.mkdir()
    encoder.write_text(
        "def encode_messages(messages, **kwargs):\n    return str(messages)\n",
        encoding="utf-8",
    )
    snapshot = module._read_bundle_directory_snapshot(root.resolve())
    assert snapshot is not None
    assert snapshot["derived"]["native_cache"] == "dsv4_composite"
    observed = {
        key: snapshot[key]
        for key in (
            "schema",
            "model_bundle_path",
            "directory_identity",
            "files",
            "fingerprint_sha256",
            "derived",
        )
    }
    hashes = {
        name: row["sha256"]
        for name, row in snapshot["files"].items()
        if "sha256" in row
    }
    return observed, hashes


def dsv4_native_bundle_attestation(module, root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    contents = {
        "config.json": (
            '{"model_type":"deepseek_v4",'
            '"architectures":["DeepseekV4ForCausalLM"]}\n'
        ),
        "generation_config.json": (
            '{"temperature":1.0,"top_p":0.95,"top_k":0}\n'
        ),
        "jang_config.json": (
            '{"profile":"JANG","quantization":"affine","chat":{'
            '"encoder":"encoding_dsv4",'
            '"encoder_fn":"encode_messages",'
            '"chat_template_source":"official_python_encoder"}}\n'
        ),
        "tokenizer_config.json": '{}\n',
    }
    for name, content in contents.items():
        (root / name).write_text(content, encoding="utf-8")
    encoder = root / "encoding" / "encoding_dsv4.py"
    encoder.parent.mkdir()
    encoder.write_text(
        "def encode_messages(messages, **kwargs):\n    return str(messages)\n",
        encoding="utf-8",
    )
    snapshot = module._read_bundle_directory_snapshot(root.resolve())
    assert snapshot is not None
    return snapshot


def runtime_binding(module, bundle_fingerprint: str) -> dict:
    release = module.release_runtime_source_attestation()
    return {
        "backend_pid": 1234,
        "runtime_source_hashes": {
            key: release[key]
            for key in (
                "server_module_sha256",
                "package_init_sha256",
                "python_source_tree_sha256",
            )
        },
        "python_source_file_count": release["python_source_file_count"],
        "python_source_read_error_count": release[
            "python_source_read_error_count"
        ],
        "model_bundle_fingerprint_sha256": bundle_fingerprint,
        "cache_topology_fingerprint_sha256": "d" * 64,
    }


def validation_context(module) -> dict:
    source = source_payload()
    release = module.release_runtime_source_attestation()
    return {
        "run_id": "run-v4",
        "started_at": "2026-07-25T00:00:00Z",
        "observed_at": "2026-07-25T00:10:00Z",
        "source_commit": source["source_commit"],
        "source_tree": source["source_tree"],
        "runtime_source_hashes": {
            key: release[key]
            for key in (
                "server_module_sha256",
                "package_init_sha256",
                "python_source_tree_sha256",
            )
        },
        "python_source_file_count": release["python_source_file_count"],
        "python_source_read_error_count": release[
            "python_source_read_error_count"
        ],
    }


def process_claim(module, executable: Path, pid: int, role: str) -> dict:
    executable.write_bytes(f"{role} executable".encode())
    return {
        "pid": pid,
        "start_identity": f"start-{pid}",
        "argv": [str(executable.resolve()), f"--role={role}"],
        "executable_path": str(executable.resolve()),
        "executable_sha256": module.sha256_file(executable),
    }


def v4_common_artifact(module, tmp_path: Path, schema: str) -> tuple[dict, dict]:
    bundle, _ = bundle_attestation(module, tmp_path / "model")
    backend = process_claim(module, tmp_path / "vmlx-engine", 4101, "backend")
    gateway = process_claim(module, tmp_path / "gateway", 4102, "gateway")
    electron = process_claim(module, tmp_path / "Electron", 4103, "electron")
    artifact = {
        "schema": schema,
        "run_id": "run-v4",
        "started_at": "2026-07-25T00:01:00Z",
        "ended_at": "2026-07-25T00:02:00Z",
        "recorded_at": "2026-07-25T00:03:00Z",
        "source": {
            "commit": source_payload()["source_commit"],
            "tree": source_payload()["source_tree"],
        },
        "model_session_id": "session-v4",
        "model_id": "laguna-v4",
        "binding": {
            "bundle_path": str((tmp_path / "model").resolve()),
            "bundle_fingerprint_sha256": bundle["fingerprint_sha256"],
            "backend_process": backend,
            "gateway_process": gateway,
            "electron_process": electron,
            "direct_listener": {
                "host": "127.0.0.1",
                "port": 8001,
                "owner_pid": backend["pid"],
            },
            "gateway_listener": {
                "host": "127.0.0.1",
                "port": 8080,
                "owner_pid": gateway["pid"],
            },
            "renderer": {
                "url": "file:///renderer/index.html",
                "build_manifest_sha256": "a" * 64,
            },
            "cache_topology_fingerprint_sha256": "d" * 64,
        },
        "health": {
            "status": "healthy",
            "model_loaded": True,
            "model_name": "laguna",
            "model_type": "laguna",
            "engine_type": "batched",
            "model_bundle_path": bundle["model_bundle_path"],
            "bundle_fingerprint_sha256": bundle["fingerprint_sha256"],
            "quantization_kind": bundle["derived"]["quantization_kind"],
            "mtp": bundle["derived"]["mtp"],
            "moe": bundle["derived"]["moe"],
            "native_cache": bundle["derived"]["native_cache"],
        },
    }
    observations = {
        "processes": {
            backend["pid"]: backend,
            gateway["pid"]: gateway,
            electron["pid"]: electron,
        },
        "listeners": {
            ("127.0.0.1", 8001): artifact["binding"]["direct_listener"],
            ("127.0.0.1", 8080): artifact["binding"]["gateway_listener"],
        },
    }
    return artifact, observations


def install_runtime_observation_mocks(module, monkeypatch, observations: dict) -> None:
    monkeypatch.setattr(
        module,
        "_observe_process",
        lambda pid: deepcopy(observations["processes"].get(pid)),
    )
    monkeypatch.setattr(
        module,
        "_observe_listener",
        lambda host, port: deepcopy(observations["listeners"].get((host, port))),
    )


def json_capture(value: dict) -> tuple[str, str]:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return raw, hashlib.sha256(raw.encode()).hexdigest()


def write_author_pass_attestation(module, path: Path) -> None:
    checks = {
        name: {
            "status": "pass",
            "source_commit": source_payload()["source_commit"],
            "source_tree": source_payload()["source_tree"],
            "assertions": {
                assertion: True
                for assertion in module.REQUIRED_ASSERTIONS[name]
            },
            "evidence": [],
        }
        for name in module.REQUIRED_CHECKS
    }
    path.write_text(
        json.dumps(
            {
                "schema": module.SCHEMA,
                "scope": module.SCOPE,
                "version": module.VERSION,
                "source_commit": source_payload()["source_commit"],
                "source_tree": source_payload()["source_tree"],
                "checks": checks,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_r20_has_no_parallel_measured_or_live_feature_self_certification():
    module = load_module()
    source = SCRIPT.read_text(encoding="utf-8")
    assert "measured_observation" not in source
    assert "live_features" not in source
    assert "OBSERVATION_CAPTURE_SCHEMA" not in source
    assert "UI_DOM_CAPTURE_SCHEMA" not in source
    assert "API_RESPONSE_CAPTURE_SCHEMA" not in source
    assert "COMMAND_PROCESS_CAPTURE_SCHEMA" not in source
    assert "process_capture_sha256" not in source
    assert all(
        "measured_observation" not in kinds
        for kinds in module.REQUIRED_RECORD_KINDS.values()
    )


@pytest.mark.parametrize(
    ("protocol", "raw"),
    [
        (
            "chat",
            b'data: {"choices":[{"delta":{"reasoning_content":"reason "}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
            b"data: [DONE]\n\n",
        ),
        (
            "responses",
            b"event: response.reasoning_summary_text.delta\n"
            b'data: {"type":"response.reasoning_summary_text.delta","delta":"reason "}\n\n'
            b"event: response.output_text.delta\n"
            b'data: {"type":"response.output_text.delta","delta":"answer"}\n\n'
            b"event: response.completed\n"
            b'data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
        ),
        (
            "anthropic",
            b"event: content_block_delta\n"
            b'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"reason "}}\n\n'
            b"event: content_block_delta\n"
            b'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"answer"}}\n\n'
            b"event: message_stop\n"
            b'data: {"type":"message_stop"}\n\n',
        ),
        (
            "ollama",
            b'{"message":{"thinking":"reason "},"done":false}\n'
            b'{"message":{"content":"answer"},"done":false}\n'
            b'{"message":{},"done":true,"done_reason":"stop"}\n',
        ),
    ],
)
def test_r20_parses_retained_raw_protocol_bytes(protocol: str, raw: bytes):
    module = load_module()
    parsed = module._parse_raw_protocol_stream(protocol, raw)
    assert parsed is not None
    assert parsed["reasoning"] == "reason "
    assert parsed["content"] == "answer"
    assert parsed["reasoning_delta_count"] == 1
    assert parsed["content_delta_count"] == 1
    assert parsed["terminals"] == 1
    assert parsed["terminal_last"] is True
    assert parsed["raw_sha256"] == hashlib.sha256(raw).hexdigest()


def test_r20_raw_protocol_parser_rejects_malformed_and_missing_terminal():
    module = load_module()
    assert module._parse_raw_protocol_stream("chat", b"data: not-json\n") is None
    assert (
        module._parse_raw_protocol_stream(
            "chat",
            b": keep-alive\n\nnot-an-sse-field\n\ndata: [DONE]\n\n",
        )
        is None
    )
    assert (
        module._parse_raw_protocol_stream(
            "ollama",
            b": keep-alive\n"
            b'{"message":{"content":"answer"},"done":true,"done_reason":"stop"}\n',
        )
        is None
    )
    assert (
        module._parse_raw_protocol_stream(
            "chat",
            b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n',
        )
        is None
    )


@pytest.mark.parametrize("protocol", ["chat", "responses", "anthropic"])
def test_r20_raw_sse_parser_accepts_interspersed_comments(protocol: str):
    module = load_module()
    raw = _fixture_protocol_response(protocol, 3)
    lines = raw.splitlines(keepends=True)
    with_comments = b": initial keep-alive\n" + b"".join(
        line + (b": inter-event keep-alive\n" if index == 1 else b"")
        for index, line in enumerate(lines)
    )
    parsed = module._parse_raw_protocol_stream(protocol, with_comments)
    assert parsed is not None
    assert parsed["terminals"] == 1
    assert parsed["terminal_last"] is True
    assert module._parse_raw_protocol_stream_v5(protocol, with_comments) is not None


def test_r20_raw_sse_parser_rejects_comment_only_stream():
    module = load_module()
    assert (
        module._parse_raw_protocol_stream(
            "chat",
            b": keep-alive\n\n: another keep-alive\n",
        )
        is None
    )


@pytest.mark.parametrize("status", ["failed", "cancelled", "incomplete", ""])
def test_r20_responses_completed_event_requires_successful_completed_status(
    status: str,
):
    module = load_module()
    raw = (
        b"event: response.output_text.delta\n"
        b'data: {"type":"response.output_text.delta","delta":"answer"}\n\n'
        b"event: response.completed\n"
        + json.dumps(
            {
                "type": "response.completed",
                "response": {"status": status},
            },
            separators=(",", ":"),
        ).encode()
        + b"\n\n"
    )
    assert module._parse_raw_protocol_stream("responses", raw) is None


def test_r20_responses_added_and_done_tool_name_is_idempotent():
    module = load_module()
    raw = (
        _fixture_sse(
            "response.created",
            {
                "type": "response.created",
                "response": {"id": "resp-1", "status": "in_progress"},
            },
        )
        + _fixture_sse(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "file_info",
                    "arguments": "",
                },
            },
        )
        + _fixture_sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "file_info",
                    "arguments": '{"path":"panel/package.json"}',
                },
            },
        )
        + _fixture_sse(
            "response.completed",
            {
                "type": "response.completed",
                "response": {"id": "resp-1", "status": "completed"},
            },
        )
    ).encode()

    parsed = module._parse_raw_protocol_stream("responses", raw)

    assert parsed is not None
    assert parsed["response_id"] == "resp-1"
    assert parsed["tool_calls"] == [
        {
            "id": "call-1",
            "function": {
                "name": "file_info",
                "arguments": {"path": "panel/package.json"},
            },
        }
    ]


def test_r20_chat_tool_name_fragments_still_accumulate():
    module = load_module()
    calls = {}

    module._append_tool_delta(calls, 0, "call-1", "file_", "")
    module._append_tool_delta(calls, 0, None, "info", "{}")

    assert calls[0]["name"] == "file_info"


def test_r20_responses_dynamic_final_uses_retained_results_and_response_chain():
    module = load_module()
    final = (
        "AGENTIC-RESPONSES-STREAM-DONE SIZE=5.3 KB "
        "PWD=/Users/example/mlx/vllm-mlx-r20-release-build"
    )
    requests = [
        {
            "stream": True,
            "max_output_tokens": 256,
            "input": [
                {
                    "role": "user",
                    "content": (
                        "Use file_info and then run_command. Reply with exactly "
                        "one line in this format: "
                        "AGENTIC-RESPONSES-STREAM-DONE "
                        "SIZE=<copy size_human from the file_info result> "
                        "PWD=<copy stdout from the run_command result>. "
                        "Replace both angle-bracket placeholders with the real "
                        "result values; output no other text."
                    ),
                }
            ],
        },
        {
            "stream": True,
            "max_output_tokens": 256,
            "previous_response_id": "resp-1",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": json.dumps(
                        {
                            "path": "panel/package.json",
                            "size_human": "5.3 KB",
                        },
                        separators=(",", ":"),
                    ),
                }
            ],
        },
        {
            "stream": True,
            "max_output_tokens": 256,
            "previous_response_id": "resp-2",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call-2",
                    "output": json.dumps(
                        {
                            "command": "pwd",
                            "exit_code": 0,
                            "stdout": (
                                "/Users/example/mlx/"
                                "vllm-mlx-r20-release-build"
                            ),
                        },
                        separators=(",", ":"),
                    ),
                }
            ],
        },
    ]

    def response_stream(
        response_id: str,
        reasoning: str,
        *,
        call_id: str = "",
        name: str = "",
        arguments: dict | None = None,
        content_parts: tuple[str, ...] = (),
    ) -> bytes:
        events = [
            _fixture_sse(
                "response.created",
                {
                    "type": "response.created",
                    "response": {
                        "id": response_id,
                        "status": "in_progress",
                    },
                },
            ),
            _fixture_sse(
                "response.reasoning_text.delta",
                {
                    "type": "response.reasoning_text.delta",
                    "delta": reasoning,
                },
            ),
        ]
        if call_id:
            item = {
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": json.dumps(arguments, separators=(",", ":")),
            }
            events.extend(
                [
                    _fixture_sse(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": 0,
                            "item": {**item, "arguments": ""},
                        },
                    ),
                    _fixture_sse(
                        "response.output_item.done",
                        {
                            "type": "response.output_item.done",
                            "output_index": 0,
                            "item": item,
                        },
                    ),
                ]
            )
        for part in content_parts:
            events.append(
                _fixture_sse(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "delta": part,
                    },
                )
            )
        events.append(
            _fixture_sse(
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {
                        "id": response_id,
                        "status": "completed",
                    },
                },
            )
        )
        return "".join(events).encode()

    streams = [
        response_stream(
            "resp-1",
            "reason-one",
            call_id="call-1",
            name="file_info",
            arguments={"path": "panel/package.json"},
        ),
        response_stream(
            "resp-2",
            "reason-two",
            call_id="call-2",
            name="run_command",
            arguments={"command": "pwd"},
        ),
        response_stream(
            "resp-3",
            "reason-three",
            content_parts=(final[:34], final[34:]),
        ),
    ]

    facts = module._api_flow_facts_from_raw(
        "responses",
        requests,
        streams,
    )

    assert {
        "nonempty_final",
        "content_progressive",
        "exact_tool_arguments",
        "history_three_turn",
        "tool_result_continuation",
        "reasoning_tool_reasoning_tool_answer",
    } <= facts

    wrong_chain = deepcopy(requests)
    wrong_chain[2]["previous_response_id"] = "resp-wrong"
    wrong_facts = module._api_flow_facts_from_raw(
        "responses",
        wrong_chain,
        streams,
    )
    assert "history_three_turn" not in wrong_facts
    assert "tool_result_continuation" not in wrong_facts


@pytest.mark.parametrize(
    "event_type",
    ["response.failed", "response.cancelled", "response.incomplete", "error"],
)
def test_r20_rejects_failure_terminal_event_types(event_type: str):
    module = load_module()
    raw = (
        f"event: {event_type}\n"
        + "data: "
        + json.dumps({"type": event_type}, separators=(",", ":"))
        + "\n\n"
    ).encode()
    assert module._parse_raw_protocol_stream("responses", raw) is None


def test_r20_rejects_multiple_terminals():
    module = load_module()
    raw = (
        b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
        b"data: [DONE]\n\n"
        b"data: [DONE]\n\n"
    )
    assert module._parse_raw_protocol_stream("chat", raw) is None


def test_r20_current_ui_summary_and_echo_fixture_fail_closed():
    module = load_module()
    source = source_payload()
    summary = {
        "format": module.ELECTRON_PROOF_SCHEMA,
        "status": "pass",
        "run_id": "synthetic",
        "script": "panel/scripts/live-real-ui-model-proof.mjs",
        "gitProvenance": {
            "before": {
                "commit": source["source_commit"],
                "tree": source["source_tree"],
                "dirty": False,
            },
            "after": {
                "commit": source["source_commit"],
                "tree": source["source_tree"],
                "dirty": False,
            },
        },
        "chat": {"turns": [{"role": "assistant", "content": "trust me"}]},
        "messageEventTrace": [{"events": [{"event": "terminal"}]}],
    }
    assert module._semantic_electron_turn(summary, source, [summary]) is None
    summary["electron"] = {
        "executable_path": "/bin/echo",
        "executable_sha256": module.sha256_file(Path("/bin/echo")),
    }
    assert module._semantic_electron_turn(summary, source, [summary]) is None


def test_r20_api_matrix_summary_and_predecoded_events_cannot_certify():
    module = load_module()
    source = source_payload()
    artifact = {
        "schema": module.API_MATRIX_SCHEMA,
        "schema_version": 2,
        "pass": True,
        "source": source_block(),
        "bases": {
            "direct": "http://127.0.0.1:8001",
            "gateway": "http://127.0.0.1:8080",
        },
        "protocols": ["chat", "responses", "anthropic", "ollama"],
        "raw_capture": {
            "enabled": True,
            "complete": True,
            "capture_layer": module.API_CAPTURE_LAYER,
            "capture_semantics": module.API_CAPTURE_SEMANTICS,
            "errors": 0,
            "wire_events": [{"data": "[DONE]"}],
        },
    }
    assert module._semantic_api_stream(artifact, source, [artifact]) is None


def test_r20_cache_summary_and_synthetic_token_arrays_cannot_certify():
    module = load_module()
    source = source_payload()
    artifact = {
        "schema": module.CACHE_PROOF_SCHEMA,
        "source": source_block(),
        "gate_ok": True,
        "cache_contract_ok": True,
        "requests": [{"cached_tokens": 9999}],
        "source_tokens": [1, 2, 3],
        "candidate_tokens": [1, 2, 4],
    }
    assert module._semantic_cache_observation(artifact, source, [artifact]) is None


def test_r20_architecture_matrix_stays_blocked_without_native_state_captures():
    module = load_module()
    required = set(module.REQUIRED_ASSERTIONS["cache_architecture_native_matrix"])
    records = [
        {
            "model_id": f"model-{index}",
            "binding": {"model_bundle_fingerprint_sha256": f"{index + 1:064x}"},
            "facts": [fact],
        }
        for index, fact in enumerate(sorted(required))
    ]
    assert (
        module._derived_assertions_for_check(
            "cache_architecture_native_matrix",
            {"cache_observation": records},
        )
        == set()
    )
    union = deepcopy(records[0])
    union["facts"] = sorted(required)
    assert (
        module._derived_assertions_for_check(
            "cache_architecture_native_matrix",
            {"cache_observation": [union]},
        )
        == set()
    )


def test_r20_author_written_command_receipt_with_forged_pid_is_blocked():
    module = load_module()
    artifact = {
        "schema": module.COMMAND_RESULT_SCHEMA,
        "recorded_at": STAMP,
        "result_kind": "full_python_suite",
        "command_argv": [
            "$ROOT/.venv/bin/python",
            "-m",
            "pytest",
            "-s",
            "-p",
            "no:cacheprovider",
        ],
        "pid": 12345,
        "exit_code": 0,
        "complete": True,
        "source": source_block(),
        "stdout": "collected 1200 items\n1200 passed",
    }
    assert (
        module._semantic_command_result(artifact, source_payload(), [artifact])
        is None
    )


def test_r20_focused_suite_cannot_pass_complete_python_gate():
    module = load_module()
    artifact = {
        "schema": module.COMMAND_RESULT_SCHEMA,
        "result_kind": "full_python_suite",
        "pid": 1,
        "exit_code": 0,
        "complete": True,
        "source": source_block(),
        "stdout": "collected 1 item\n1 passed",
    }
    assert module._semantic_command_result(artifact, source_payload(), [artifact]) is None


def test_r20_owned_runner_captures_child_and_derives_full_suite(
    tmp_path: Path,
    monkeypatch,
):
    module = load_module()

    class FakePopen:
        pid = 55123
        returncode = 0

        def __init__(self, argv, **kwargs):
            self.argv = argv
            self.kwargs = kwargs

        def communicate(self):
            return (
                b"test session starts\ncollected 1200 items\n"
                b"================ 1200 passed in 60.00s ================\n",
                b"",
            )

    monkeypatch.setattr(module.subprocess, "Popen", FakePopen)
    context = {
        "run_id": "owned-run",
        "started_at": "2026-07-25T00:00:00Z",
    }
    results = module._execute_owned_checks(
        {"full_python_suite"},
        tmp_path,
        context,
        {},
    )
    result = results["full_python_suite"]
    assert result["facts"] == set(
        module.REQUIRED_ASSERTIONS["full_python_suite"]
    )
    execution = result["executions"][0]
    assert execution["pid"] == FakePopen.pid
    assert execution["argv"][1:3] == ["-m", "pytest"]
    assert execution["stdout_sha256"] == hashlib.sha256(
        execution["__stdout_bytes"]
    ).hexdigest()
    assert execution["stderr_sha256"] == hashlib.sha256(
        execution["__stderr_bytes"]
    ).hexdigest()


def test_r20_v5_failed_owned_child_seals_private_streams(
    tmp_path: Path,
    monkeypatch,
):
    module = load_module()
    output_path = tmp_path / "api.phase-00.producer.json"
    output_fd = os.open(
        output_path,
        os.O_RDWR | os.O_CREAT | os.O_EXCL,
        0o600,
    )

    class FailedProcess:
        pid = 7719
        returncode = 1

        def communicate(self, *, input=None, timeout=None):
            assert input == b"finish\n"
            return b"private stdout detail\n", b"private stderr detail\n"

    monkeypatch.setattr(
        module,
        "_v5_pin_unchanged",
        lambda _pin, executable=False: True,
    )
    monkeypatch.setattr(
        module,
        "_v5_executable_invocation_unchanged",
        lambda _invocation: True,
    )
    handle = {
        "name": "api",
        "process": FailedProcess(),
        "output_fd": output_fd,
        "output_path": output_path,
        "argv": ["fixture"],
        "cwd": str(tmp_path),
        "started_at": STAMP,
        "executable": {"path": str(tmp_path / "python")},
        "executable_invocation": None,
        "scripts": [],
        "process_observation": {"pid": FailedProcess.pid},
    }

    with pytest.raises(RuntimeError) as exc_info:
        module._v5_finish_owned_child(
            handle,
            {"run_id": "run", "nonce": "a" * 32},
        )

    message = str(exc_info.value)
    stdout_path = output_path.with_name("api.phase-00.producer.stdout")
    stderr_path = output_path.with_name("api.phase-00.producer.stderr")
    assert f"stdout_path={stdout_path}" in message
    assert f"stderr_path={stderr_path}" in message
    assert stdout_path.read_bytes() == b"private stdout detail\n"
    assert stderr_path.read_bytes() == b"private stderr detail\n"
    assert (stdout_path.stat().st_mode & 0o777) == 0o600
    assert (stderr_path.stat().st_mode & 0o777) == 0o600


def test_r20_v5_cache_phase_wait_seals_private_streams_on_early_exit(
    tmp_path: Path,
    monkeypatch,
):
    module = load_module()
    output_path = tmp_path / "cache.producer.json"
    output_fd = os.open(
        output_path,
        os.O_RDWR | os.O_CREAT | os.O_EXCL,
        0o600,
    )

    class FailedCacheProcess:
        pid = 7831
        returncode = 1

        @staticmethod
        def poll():
            return 1

        @staticmethod
        def communicate(*, input=None, timeout=None):
            assert input == b"finish\n"
            assert timeout == 5
            return b"cache private stdout\n", b"cache private stderr\n"

    monkeypatch.setattr(
        module,
        "_v5_pin_unchanged",
        lambda _pin, executable=False: True,
    )
    monkeypatch.setattr(
        module,
        "_v5_executable_invocation_unchanged",
        lambda _invocation: True,
    )
    handle = {
        "name": "cache",
        "process": FailedCacheProcess(),
        "output_fd": output_fd,
        "output_path": output_path,
        "argv": ["fixture"],
        "cwd": str(tmp_path),
        "started_at": STAMP,
        "executable": {"path": str(tmp_path / "python")},
        "executable_invocation": None,
        "scripts": [],
        "process_observation": {"pid": FailedCacheProcess.pid},
    }
    phase = dict(module.V5_CACHE_PHASES[1])

    with pytest.raises(RuntimeError) as exc_info:
        module._v5_wait_for_cache_phase_done(
            handle,
            tmp_path / "missing.done.json",
            run_context={"run_id": "run", "nonce": "a" * 32},
            phase=phase,
            binding={"session_id": "session", "backend_pid": 1234},
            binding_sha256="b" * 64,
            timeout=60,
        )

    message = str(exc_info.value)
    stdout_path = output_path.with_name("cache.producer.stdout")
    stderr_path = output_path.with_name("cache.producer.stderr")
    assert "retained child failure details" in message
    assert stdout_path.read_bytes() == b"cache private stdout\n"
    assert stderr_path.read_bytes() == b"cache private stderr\n"
    assert (stdout_path.stat().st_mode & 0o777) == 0o600
    assert (stderr_path.stat().st_mode & 0o777) == 0o600


def test_r20_v5_ui_probe_phases_reuse_prior_store_block_disk_root():
    source = (ROOT / "panel/scripts/live-real-ui-model-proof.mjs").read_text(
        encoding="utf-8",
    )
    assert "releaseBlockDiskCacheAnchorPhase" in source
    assert "activeReleasePhase.operation !== 'probe'" in source
    assert "candidate.phase_index < activeReleasePhase.phase_index" in source
    assert "candidate.operation !== 'probe'" in source
    assert "ui-shared-block-disk-cache" in source


def test_r20_owned_jang_runner_requires_build_import_and_test(
    tmp_path: Path,
    monkeypatch,
):
    module = load_module()
    jang_root = tmp_path / "jang"
    jang_root.mkdir()
    monkeypatch.setenv("VMLX_JANG_TOOLS_SOURCE", str(jang_root))
    outputs = {
        "build": b"Successfully built jang.whl\n",
        "import": (
            "VMLINUX_IMPORT_JSON="
            + json.dumps(
                {
                    "jang_tools": str((jang_root / "jang_tools/__init__.py").resolve()),
                    "vmlx_engine": str(
                        (ROOT / "vmlx_engine/__init__.py").resolve()
                    ),
                },
                sort_keys=True,
            )
            + "\n"
        ).encode(),
        "test": b"test session starts\ncollected 3 items\n3 passed in 1.00s\n",
    }

    class FakePopen:
        next_pid = 56000

        def __init__(self, argv, **kwargs):
            type(self).next_pid += 1
            self.pid = type(self).next_pid
            self.argv = argv
            self.returncode = 0

        def communicate(self):
            if "-c" in self.argv:
                return outputs["import"], b""
            if "build" in self.argv:
                return outputs["build"], b""
            return outputs["test"], b""

    monkeypatch.setattr(module.subprocess, "Popen", FakePopen)
    jang_state = {
        "commit": module.JANG_COMMIT,
        "tree": module.JANG_TREE,
        "version": module.JANG_VERSION,
    }
    results = module._execute_owned_checks(
        {"jang_runtime_provenance"},
        tmp_path,
        {"run_id": "owned-jang", "started_at": STAMP},
        jang_state,
    )
    assert results["jang_runtime_provenance"]["facts"] == set(
        module.REQUIRED_ASSERTIONS["jang_runtime_provenance"]
    )
    assert len(results["jang_runtime_provenance"]["executions"]) == 3


def test_r20_v5_jang_source_argument_is_authoritative_without_environment(
    tmp_path: Path,
    monkeypatch,
):
    module = load_module()
    jang_root = tmp_path / "jang"
    jang_root.mkdir()
    (jang_root / "pyproject.toml").write_text(
        f'version = "{module.JANG_VERSION}"\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("VMLX_JANG_TOOLS_SOURCE", raising=False)

    def fake_git(root: Path, *args: str) -> str:
        assert root == jang_root.resolve()
        command = tuple(args)
        if command == ("rev-parse", "HEAD"):
            return module.JANG_COMMIT
        if command == ("rev-parse", "HEAD^{tree}"):
            return module.JANG_TREE
        if command == ("status", "--porcelain", "--untracked-files=all"):
            return ""
        if command == ("rev-parse", "@{upstream}"):
            return module.JANG_COMMIT
        if command == ("remote", "get-url", "origin"):
            return "git@github.com:jjang-ai/jangq.git"
        if command == (
            "ls-remote",
            "--exit-code",
            "origin",
            "refs/heads/main",
        ):
            return f"{module.JANG_COMMIT}\trefs/heads/main"
        raise AssertionError(command)

    monkeypatch.setattr(module, "run_git_in", fake_git)
    failures: list[str] = []
    observed = module.validate_jang_source(failures, jang_root)
    assert failures == []
    assert observed["commit"] == module.JANG_COMMIT
    assert observed["tree"] == module.JANG_TREE


def test_r20_v5_jang_source_argument_rejects_environment_mismatch(
    tmp_path: Path,
    monkeypatch,
):
    module = load_module()
    explicit = tmp_path / "explicit"
    environment = tmp_path / "environment"
    explicit.mkdir()
    environment.mkdir()
    monkeypatch.setenv("VMLX_JANG_TOOLS_SOURCE", str(environment))
    failures: list[str] = []
    module.validate_jang_source(failures, explicit)
    assert any(
        "identify different repositories" in failure
        for failure in failures
    )


def test_r20_health_family_contract_requires_actual_bundle_path_and_hashes(
    tmp_path: Path,
):
    module = load_module()
    bundle, _ = bundle_attestation(module, tmp_path / "model")
    health = {
        "status": "healthy",
        "model_loaded": True,
        "model_name": "laguna",
        "model_type": "llm",
        "engine_type": "batched",
        "model_bundle_path": str((tmp_path / "model").resolve()),
        "model_bundle_provenance": bundle,
        "quantization": {
            "codec": "affine_quantized_matmul",
            "weight_format": "jang",
            "sidecar": {"jang_config": True, "jangtq_runtime": False},
        },
        "mtp": {},
        "routing": {},
        "native_cache": {},
    }
    contract = module._health_family_contract(health)
    assert contract is not None
    assert contract["bundle_config_hashes"]["jang_config.json"]
    missing_path = deepcopy(health)
    missing_path.pop("model_bundle_path")
    assert module._health_family_contract(missing_path) is None
    changed = deepcopy(health)
    (tmp_path / "model/config.json").write_text('{"model_type":"changed"}\n')
    assert module._health_family_contract(changed) is None


def test_r20_bundle_attestation_accepts_production_path_free_health_shape(
    tmp_path: Path,
):
    module = load_module()
    root = tmp_path / "model"
    bundle_attestation(module, root)
    snapshot = module._read_bundle_directory_snapshot(root.resolve())
    assert snapshot is not None
    health_attestation = snapshot["health_attestation"]
    hashes = module._validated_bundle_attestation(
        str(root.resolve()),
        health_attestation,
    )
    assert hashes == {
        name: row["sha256"] for name, row in snapshot["files"].items()
    }
    assert snapshot["fingerprint_sha256"] == health_attestation[
        "fingerprint_sha256"
    ]
    assert snapshot["directory_fingerprint_sha256"] != snapshot[
        "fingerprint_sha256"
    ]
    legacy_v2 = {
        key: snapshot[key]
        for key in (
            "schema",
            "model_bundle_path",
            "directory_identity",
            "files",
            "derived",
        )
    }
    legacy_v2["fingerprint_sha256"] = "f" * 64
    assert module._validated_bundle_attestation(
        str(root.resolve()),
        legacy_v2,
    ) is None


def test_r20_bundle_snapshot_accepts_dsv4_native_encoder_without_jinja(
    tmp_path: Path,
):
    module = load_module()
    root = tmp_path / "dsv4"
    snapshot = dsv4_native_bundle_attestation(module, root)
    assert snapshot["files"]["chat_template.jinja"] == {"state": "missing"}
    assert snapshot["prompt_renderer"]["mode"] == "native_encoder"
    assert snapshot["prompt_renderer"]["encoder"] == "encoding_dsv4"
    assert snapshot["prompt_renderer"]["encoder_fn"] == "encode_messages"
    assert len(snapshot["prompt_renderer"]["sha256"]) == 64
    hashes = module._validated_bundle_attestation(
        str(root.resolve()),
        snapshot["health_attestation"],
    )
    assert hashes is not None
    assert "chat_template.jinja" not in hashes


def test_r20_bundle_snapshot_rejects_dsv4_native_metadata_without_encoder(
    tmp_path: Path,
):
    module = load_module()
    root = tmp_path / "dsv4"
    dsv4_native_bundle_attestation(module, root)
    (root / "encoding" / "encoding_dsv4.py").unlink()
    assert module._read_bundle_directory_snapshot(root.resolve()) is None


def test_r20_bundle_snapshot_rejects_unresolved_template_include_stub(
    tmp_path: Path,
):
    module = load_module()
    root = tmp_path / "stub"
    root.mkdir()
    (root / "config.json").write_text(
        '{"model_type":"generic"}\n', encoding="utf-8"
    )
    (root / "generation_config.json").write_text('{}\n', encoding="utf-8")
    (root / "jang_config.json").write_text('{}\n', encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        '{"chat_template":"{% include \\\"chat_template.jinja\\\" %}"}\n',
        encoding="utf-8",
    )
    assert module._read_bundle_directory_snapshot(root.resolve()) is None


def test_r20_v5_runtime_bundle_attestation_requires_production_nested_shape(
    tmp_path: Path,
):
    module = load_module()
    root = tmp_path / "model"
    bundle_attestation(module, root)
    snapshot = module._read_bundle_directory_snapshot(root.resolve())
    assert snapshot is not None
    provenance = deepcopy(snapshot["health_attestation"])
    health = {"model_bundle_provenance": provenance}
    runtime = {"model_bundle_provenance": deepcopy(provenance)}
    assert module._v5_validate_runtime_bundle_attestation(
        health,
        runtime,
        snapshot,
    )

    altered = deepcopy(provenance)
    altered["fingerprint_sha256"] = "0" * 64
    assert not module._v5_validate_runtime_bundle_attestation(
        {"model_bundle_provenance": altered},
        {"model_bundle_provenance": deepcopy(altered)},
        snapshot,
    )
    assert not module._v5_validate_runtime_bundle_attestation(
        {
            "model_bundle_path": snapshot["model_bundle_path"],
            "bundle_fingerprint_sha256": snapshot["fingerprint_sha256"],
        },
        {},
        snapshot,
    )


def test_r20_v5_hold_observation_accepts_path_free_nested_bundle_health(
    tmp_path: Path,
    monkeypatch,
):
    module = load_module()
    root = tmp_path / "model"
    bundle_attestation(module, root)
    snapshot = module._read_bundle_directory_snapshot(root.resolve())
    assert snapshot is not None
    provenance = deepcopy(snapshot["health_attestation"])
    binding = {
        "health_url": "http://127.0.0.1:8022/health",
        "cdp_url": "http://127.0.0.1:9355",
        "backend_pid": 1201,
        "gateway_pid": 1202,
        "electron_pid": 1202,
        "direct_base_url": "http://127.0.0.1:8022",
        "gateway_base_url": "http://127.0.0.1:8080",
        "model_bundle_path": snapshot["model_bundle_path"],
        "bundle_fingerprint_sha256": snapshot["fingerprint_sha256"],
        "source_commit": "b" * 40,
        "session_id": "session-path-free-health",
    }
    health = {
        "status": "healthy",
        "model_loaded": True,
        "model_bundle_provenance": provenance,
        "runtime_provenance": {
            "model_bundle_provenance": deepcopy(provenance),
        },
    }
    dom = {
        "sourceCommit": binding["source_commit"],
        "session_ids": [binding["session_id"]],
    }

    monkeypatch.setattr(
        module,
        "_v5_loopback_http_get",
        lambda _url: json.dumps(health).encode(),
    )
    monkeypatch.setattr(
        module,
        "_v5_cdp_dom_snapshot",
        lambda _url: json.dumps(dom).encode(),
    )
    monkeypatch.setattr(
        module,
        "_observe_process",
        lambda pid: {"pid": pid, "command": f"proc-{pid}"},
    )
    monkeypatch.setattr(
        module,
        "_observe_listener",
        lambda host, port: {
            "host": host,
            "port": port,
            "owner_pid": 1201 if port == 8022 else 1202,
        },
    )

    observation = module._v5_default_hold_observation(binding)
    assert observation["backend"]["pid"] == 1201
    assert observation["gateway_listener"]["owner_pid"] == 1202


def test_r20_v5_runtime_source_attestation_requires_path_free_health_shape():
    module = load_module()
    runtime = {
        "package_init_relpath": "vmlx_engine/__init__.py",
        "package_init_sha256": "1" * 64,
        "server_module_relpath": "vmlx_engine/server.py",
        "server_module_sha256": "2" * 64,
        "python_source_tree_sha256": "3" * 64,
        "python_source_file_count": 270,
        "python_source_read_error_count": 0,
        "python_executable_fingerprint_sha256": "4" * 64,
    }
    assert module._v5_runtime_source_attestation(runtime) == {
        key: runtime[key]
        for key in (
            "package_init_sha256",
            "server_module_sha256",
            "python_source_tree_sha256",
            "python_executable_fingerprint_sha256",
            "python_source_file_count",
            "python_source_read_error_count",
        )
    }
    with_private_path = deepcopy(runtime)
    with_private_path.pop("package_init_relpath")
    with_private_path["package_init_path"] = "/private/stale/vmlx_engine/__init__.py"
    assert module._v5_runtime_source_attestation(with_private_path) is None
    unreadable = deepcopy(runtime)
    unreadable["python_source_read_error_count"] = 1
    assert module._v5_runtime_source_attestation(unreadable) is None


def test_r20_v5_backend_invocation_fingerprint_preserves_venv_symlink(
    tmp_path: Path,
):
    module = load_module()
    target = Path(sys.executable).resolve()
    invocation = tmp_path / "venv-python"
    invocation.symlink_to(target)
    backend = {
        "argv": [str(invocation.absolute()), "-m", "vmlx_engine.cli"],
        "executable_path": str(target.resolve()),
    }
    assert module._v5_backend_invocation_fingerprint(backend) == hashlib.sha256(
        str(invocation.absolute()).encode()
    ).hexdigest()
    mismatched = deepcopy(backend)
    mismatched["executable_path"] = str(Path("/bin/sh").resolve())
    assert module._v5_backend_invocation_fingerprint(mismatched) is None


def test_r20_v5_runtime_backend_argv_is_bracketed_by_libproc_identity(
    monkeypatch,
    tmp_path: Path,
):
    module = load_module()
    target = Path(sys.executable).resolve()
    invocation = tmp_path / "venv-python"
    invocation.symlink_to(target)
    identity = {
        "pid": 7123,
        "start_identity": "darwin-proc:123:456789",
        "executable_path": str(target),
        "executable_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    identities = iter([
        {**deepcopy(identity), "argv": []},
        {**deepcopy(identity), "argv": []},
    ])
    monkeypatch.setattr(
        module,
        "_observe_darwin_process",
        lambda _pid: next(identities),
    )
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=f'{invocation} -m vmlx_engine.cli serve "model path"\n',
            stderr="",
        )

    monkeypatch.setattr(
        module.subprocess,
        "run",
        run,
    )

    observed = module._v5_observe_runtime_process_with_argv(7123)

    assert observed == {
        **identity,
        "argv": [
            str(invocation),
            "-m",
            "vmlx_engine.cli",
            "serve",
            "model path",
        ],
    }
    assert calls == [
        (
            ["/bin/ps", "-ww", "-p", "7123", "-o", "command="],
            {"capture_output": True, "text": True, "check": False},
        )
    ]
    assert module._v5_backend_invocation_fingerprint(observed) == hashlib.sha256(
        str(invocation.absolute()).encode()
    ).hexdigest()


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    [
        ("pid", 7124),
        ("start_identity", "darwin-proc:124:000001"),
        ("executable_path", "/bin/sh"),
        ("executable_sha256", "b" * 64),
    ],
)
def test_r20_v5_runtime_backend_argv_rejects_libproc_identity_drift(
    monkeypatch,
    changed_field,
    changed_value,
):
    module = load_module()
    before = {
        "pid": 7123,
        "start_identity": "darwin-proc:123:456789",
        "executable_path": str(Path(sys.executable).resolve()),
        "executable_sha256": "a" * 64,
        "argv": [],
    }
    after = {**before, changed_field: changed_value}
    identities = iter([before, after])
    monkeypatch.setattr(
        module,
        "_observe_darwin_process",
        lambda _pid: next(identities),
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(
            argv,
            0,
            stdout=f"{Path(sys.executable).absolute()} -V\n",
            stderr="",
        ),
    )

    assert module._v5_observe_runtime_process_with_argv(7123) is None


def test_r20_bundle_attestation_rejects_symlink_and_hardlink_files(
    tmp_path: Path,
):
    module = load_module()

    symlink_root = tmp_path / "symlink-model"
    symlink_attestation, _ = bundle_attestation(module, symlink_root)
    external = tmp_path / "external-config.json"
    external.write_text(
        '{"model_type":"laguna","architectures":["LagunaForCausalLM"]}\n',
        encoding="utf-8",
    )
    (symlink_root / "config.json").unlink()
    (symlink_root / "config.json").symlink_to(external)
    assert (
        module._validated_bundle_attestation(
            str(symlink_root.resolve()),
            symlink_attestation,
        )
        is None
    )

    hardlink_root = tmp_path / "hardlink-model"
    hardlink_attestation, _ = bundle_attestation(module, hardlink_root)
    os.link(hardlink_root / "config.json", tmp_path / "config-hardlink.json")
    assert (
        module._validated_bundle_attestation(
            str(hardlink_root.resolve()),
            hardlink_attestation,
        )
        is None
    )


def test_r20_evidence_reader_rejects_links_and_path_replacement(
    tmp_path: Path,
    monkeypatch,
):
    module = load_module()
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()

    regular = evidence_root / "regular.json"
    regular.write_bytes(b'{"value":1}')
    assert module._read_regular_file_once(regular, evidence_root) == b'{"value":1}'

    symlink = evidence_root / "symlink.json"
    symlink.symlink_to(regular)
    assert module._read_regular_file_once(symlink, evidence_root) is None

    hardlink = evidence_root / "hardlink.json"
    os.link(regular, hardlink)
    assert module._read_regular_file_once(regular, evidence_root) is None
    assert module._read_regular_file_once(hardlink, evidence_root) is None

    replace_target = evidence_root / "replace.json"
    replacement = evidence_root / "replacement.json"
    replace_target.write_bytes(b'{"version":1}')
    replacement.write_bytes(b'{"version":2}')
    original_read = module._read_fd_bytes

    def replace_after_read(fd: int) -> bytes:
        raw = original_read(fd)
        os.replace(replacement, replace_target)
        return raw

    monkeypatch.setattr(module, "_read_fd_bytes", replace_after_read)
    assert module._read_regular_file_once(replace_target, evidence_root) is None


def test_r20_bundle_snapshot_rejects_directory_swap(
    tmp_path: Path,
    monkeypatch,
):
    module = load_module()
    bundle_attestation(module, tmp_path / "model")
    bundle_attestation(module, tmp_path / "replacement-model")
    original_read = module._read_fd_bytes
    swapped = False

    def swap_directory_after_first_read(fd: int) -> bytes:
        nonlocal swapped
        raw = original_read(fd)
        if not swapped:
            swapped = True
            os.replace(tmp_path / "model", tmp_path / "old-model")
            os.replace(tmp_path / "replacement-model", tmp_path / "model")
        return raw

    monkeypatch.setattr(module, "_read_fd_bytes", swap_directory_after_first_read)
    assert module._read_bundle_directory_snapshot(tmp_path / "model") is None


def test_r20_source_trace_rejects_fake_electron_and_nonexistent_process(
    tmp_path: Path,
):
    module = load_module()
    expected = module.release_runtime_source_attestation()
    bundle, bundle_hashes = bundle_attestation(module, tmp_path / "model")
    executable = tmp_path / "vMLX"
    renderer = tmp_path / "index.html"
    executable.write_bytes(b"electron executable")
    renderer.write_text("<html></html>", encoding="utf-8")
    cdp_port = 9335
    argv = (
        str(executable.resolve()),
        f"--remote-debugging-port={cdp_port}",
    )
    source = source_payload()
    artifact = {
        "schema": module.SOURCE_TRACE_SCHEMA,
        "recorded_at": STAMP,
        "command": "git release-source-trace",
        "source": source_block(),
        "git": {
            "head": source["source_commit"],
            "tree": source["source_tree"],
            "upstream": source["source_commit"],
            "remote_main": source["source_commit"],
            "clean": True,
        },
        "head_source_attestation": expected,
        "python_runtime": runtime_binding(module, bundle["fingerprint_sha256"]),
        "electron_runtime": {
            "pid": 987654321,
            "argv": list(argv),
            "cdp_url": f"http://127.0.0.1:{cdp_port}",
            "executable_path": str(executable.resolve()),
            "executable_sha256": module.sha256_file(executable),
            "renderer_asset_path": str(renderer.resolve()),
            "renderer_asset_sha256": module.sha256_file(renderer),
            "renderer_loaded_url": renderer.resolve().as_uri(),
            "electron_main_tree_sha256": expected["electron_main_tree_sha256"],
            "renderer_source_tree_sha256": expected[
                "renderer_source_tree_sha256"
            ],
        },
        "bundle_path": str((tmp_path / "model").resolve()),
        "bundle_health_attestation": bundle,
        "bundle_config_hashes": bundle_hashes,
    }
    assert module._semantic_source_trace(artifact, source) is None

    echo = deepcopy(artifact)
    echo["electron_runtime"]["executable_path"] = "/bin/echo"
    echo["electron_runtime"]["executable_sha256"] = module.sha256_file(
        Path("/bin/echo")
    )
    echo_argv = ("/bin/echo", f"--remote-debugging-port={cdp_port}")
    echo["electron_runtime"]["argv"] = list(echo_argv)
    assert module._semantic_source_trace(echo, source) is None


def test_r20_v4_ui_semantic_accepts_raw_bound_dom_and_events(
    tmp_path: Path,
    monkeypatch,
):
    module = load_module()
    artifact, observations = v4_common_artifact(
        module,
        tmp_path,
        module.ELECTRON_RAW_SCHEMA,
    )
    install_runtime_observation_mocks(module, monkeypatch, observations)
    artifact["cdp"] = {
        "electron_pid": artifact["binding"]["electron_process"]["pid"],
        "target_url": artifact["binding"]["renderer"]["url"],
        "websocket_url": "ws://127.0.0.1:9335/devtools/page/1",
    }
    artifact["start_button_event"] = {
        "trusted": True,
        "model_session_id": artifact["model_session_id"],
    }
    turns = []
    for index in range(3):
        events = [
            {"seq": 0, "type": "reasoning_delta", "text": f"reason-{index}"},
            {"seq": 1, "type": "content_delta", "text": f"answer-{index}"},
        ]
        if index == 0:
            events.extend(
                [
                    {"seq": 2, "type": "tool_call", "name": "file_info"},
                    {"seq": 3, "type": "tool_result", "content": "5.2 KB"},
                    {"seq": 4, "type": "terminal", "status": "completed"},
                ]
            )
        else:
            events.append(
                {"seq": 2, "type": "terminal", "status": "completed"}
            )
        request_json, request_sha = json_capture(
            {"messages": [{"role": "user", "content": f"turn-{index}"}]}
        )
        dom_json, dom_sha = json_capture(
            {
                "reasoning_text": f"reason-{index}",
                "content_text": f"answer-{index}",
                "terminal": "completed",
                "rendering_ok": True,
                "coherent": True,
                "cache_stats": {"cached_tokens": index * 8},
                "ttft_ms": 25 + index,
                "decode_tps": 40.0,
            }
        )
        turns.append(
            {
                "request_body_json": request_json,
                "request_sha256": request_sha,
                "dom_snapshot_json": dom_json,
                "dom_snapshot_sha256": dom_sha,
                "events": events,
            }
        )
    artifact["turns"] = turns
    payload = {"__validation_context": validation_context(module)}
    semantic = module._semantic_electron_turn(artifact, payload)
    assert semantic is not None
    assert semantic["turn_count"] == 3
    assert {
        "reasoning_rail",
        "visible_content",
        "tool_result_continuation",
        "rendering",
        "coherence",
    } <= set(semantic["facts"])
    wrong_run = deepcopy(artifact)
    wrong_run["run_id"] = "different-run"
    assert (
        module._semantic_electron_turn(
            wrong_run,
            {"__validation_context": validation_context(module)},
        )
        is None
    )


def chat_stream(
    reasoning: str,
    *,
    tool_id: str | None = None,
    tool_name: str | None = None,
    tool_arguments: dict | None = None,
    content_parts: tuple[str, ...] = (),
) -> str:
    rows = [
        "data: "
        + json.dumps(
            {"choices": [{"delta": {"reasoning_content": reasoning}}]},
            separators=(",", ":"),
        )
    ]
    if tool_id and tool_name and tool_arguments is not None:
        rows.append(
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": tool_id,
                                        "function": {
                                            "name": tool_name,
                                            "arguments": json.dumps(
                                                tool_arguments,
                                                separators=(",", ":"),
                                            ),
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
                separators=(",", ":"),
            )
        )
    for part in content_parts:
        rows.append(
            "data: "
            + json.dumps(
                {"choices": [{"delta": {"content": part}}]},
                separators=(",", ":"),
            )
        )
    rows.append("data: [DONE]")
    return "\n\n".join(rows) + "\n\n"


def test_r20_v4_api_semantic_accepts_raw_requests_and_streams(
    tmp_path: Path,
    monkeypatch,
):
    module = load_module()
    artifact, observations = v4_common_artifact(
        module,
        tmp_path,
        module.API_RAW_SCHEMA,
    )
    install_runtime_observation_mocks(module, monkeypatch, observations)
    requests = [
        {
            "stream": True,
            "enable_thinking": True,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "call file_info"}],
        },
        {
            "stream": True,
            "enable_thinking": True,
            "max_tokens": 256,
            "messages": [
                {"role": "tool", "tool_call_id": "c1", "content": "5.2 KB"}
            ],
        },
        {
            "stream": True,
            "enable_thinking": True,
            "max_tokens": 256,
            "messages": [
                {"role": "tool", "tool_call_id": "c1", "content": "5.2 KB"},
                {"role": "tool", "tool_call_id": "c2", "content": str(ROOT)},
                {"role": "user", "content": "reply exactly FINAL-DONE"},
            ],
        },
    ]
    streams = [
        chat_stream(
            "reason-one",
            tool_id="c1",
            tool_name="file_info",
            tool_arguments={"path": "panel/package.json"},
        ),
        chat_stream(
            "reason-two",
            tool_id="c2",
            tool_name="run_command",
            tool_arguments={"command": "pwd"},
        ),
        chat_stream(
            "reason-three",
            content_parts=("FINAL-", "DONE"),
        ),
    ]
    flows = []
    for route in ("direct", "gateway"):
        for request, stream in zip(requests, streams, strict=True):
            request_json, request_sha = json_capture(request)
            flows.append(
                {
                    "protocol": "chat",
                    "route": route,
                    "endpoint": "/v1/chat/completions",
                    "request_body_json": request_json,
                    "request_sha256": request_sha,
                    "response_stream": stream,
                    "response_sha256": hashlib.sha256(stream.encode()).hexdigest(),
                }
            )
    artifact["flows"] = flows
    semantic = module._semantic_api_stream(
        artifact,
        {"__validation_context": validation_context(module)},
    )
    assert semantic is not None
    assert semantic["protocols"] == ["chat"]
    assert {
        "reasoning_separate",
        "content_progressive",
        "tool_result_continuation",
        "terminal_truthful",
    } <= set(semantic["facts_by_protocol"]["chat"])


def test_r20_v4_cache_semantic_derives_lcp_and_telemetry(
    tmp_path: Path,
    monkeypatch,
):
    module = load_module()
    artifact, observations = v4_common_artifact(
        module,
        tmp_path,
        module.CACHE_RAW_SCHEMA,
    )
    install_runtime_observation_mocks(module, monkeypatch, observations)
    artifact["source_tokens"] = [10, 11, 12, 13]
    artifact["candidate_tokens"] = [10, 11, 12, 99, 100]
    artifact["telemetry"] = {
        "paged_ram": False,
        "matched_tokens": 3,
        "suffix_prefill_tokens": 2,
        "events": [
            {"type": "ssd_store"},
            {"type": "ssd_restore", "matched_tokens": 3},
            {"type": "cross_chat_reuse"},
            {"type": "cross_session_reuse"},
            {"type": "restart_restore"},
        ],
    }
    semantic = module._semantic_cache_observation(
        artifact,
        {"__validation_context": validation_context(module)},
    )
    assert semantic is not None
    assert {
        "paged_ram_disabled",
        "ssd_l2_enabled",
        "longest_prefix_partial_block_hit",
        "uncached_suffix_prefilled",
        "cross_chat_reuse",
        "cross_session_reuse",
        "restart_disk_restore",
        "standard_kv",
    } <= set(semantic["facts"])


def test_r20_jang_receipt_with_forged_pid_is_blocked():
    module = load_module()
    artifact = {
        "schema": module.JANG_RESULT_SCHEMA,
        "recorded_at": STAMP,
        "pid": 99,
        "exit_code": 0,
        "complete": True,
        "source": source_block(),
        "command_argv": [
            "$ROOT/.venv/bin/python",
            "-m",
            "pytest",
            "tests/test_laguna_loader.py",
        ],
        "stdout": "collected 3 items\n3 passed",
    }
    assert module._semantic_jang_result(artifact, source_payload(), [artifact]) is None


def test_r20_validate_attestation_sanitizes_author_passes(
    tmp_path: Path,
):
    module = load_module()
    attestation = tmp_path / "attestation.json"
    write_author_pass_attestation(module, attestation)
    git_state = {
        "commit": source_payload()["source_commit"],
        "tree": source_payload()["source_tree"],
        "upstream_commit": source_payload()["source_commit"],
        "remote_main_commit": source_payload()["source_commit"],
    }
    failures: list[str] = []
    owned = {
        "full_python_suite": {
            "facts": set(module.REQUIRED_ASSERTIONS["full_python_suite"]),
            "executions": [],
        }
    }
    checks = module.validate_attestation(
        attestation,
        git_state,
        tmp_path,
        failures,
        run_context=validation_context(module),
        owned_results=owned,
    )
    assert checks["full_python_suite"]["status"] == "pass"
    assert all(checks["full_python_suite"]["assertions"].values())
    assert checks["electron_visual_multiturn"]["status"] == "blocked"
    assert not any(checks["electron_visual_multiturn"]["assertions"].values())
    assert checks["exact_source_provenance"]["status"] == "blocked"
    assert checks["exact_source_provenance"]["assertions"][
        "checkout_head_exact"
    ]
    assert not checks["exact_source_provenance"]["assertions"][
        "electron_revision_exact"
    ]
    assert failures


def test_r20_main_manifest_reports_owned_pass_and_blocked_rows(
    tmp_path: Path,
    monkeypatch,
):
    module = load_module()
    attestation = tmp_path / "attestation.json"
    output = tmp_path / "manifest.json"
    write_author_pass_attestation(module, attestation)
    git_state = {
        "commit": source_payload()["source_commit"],
        "tree": source_payload()["source_tree"],
        "upstream_commit": source_payload()["source_commit"],
        "remote_main_commit": source_payload()["source_commit"],
    }
    monkeypatch.setattr(module, "validate_versions", lambda failures: {"ok": True})
    monkeypatch.setattr(module, "validate_git_state", lambda failures: git_state)
    monkeypatch.setattr(module, "validate_jang_source", lambda failures: {})
    monkeypatch.setattr(
        module,
        "validate_private_evidence_root",
        lambda configured, failures: tmp_path,
    )
    monkeypatch.setattr(
        module,
        "_execute_owned_checks",
        lambda requested, private_root, run_context, jang_state: {
            "full_python_suite": {
                "facts": set(
                    module.REQUIRED_ASSERTIONS["full_python_suite"]
                ),
                "executions": [
                    {
                        "schema": module.OWNED_EXECUTION_SCHEMA,
                        "run_id": run_context["run_id"],
                        "pid": 777,
                        "argv": ["python", "-m", "pytest"],
                        "cwd": str(ROOT),
                        "started_at": STAMP,
                        "ended_at": STAMP,
                        "exit_code": 0,
                        "stdout_sha256": "a" * 64,
                        "stderr_sha256": "b" * 64,
                        "__stdout_bytes": b"not-public",
                    }
                ],
            }
        },
    )
    release = module.release_runtime_source_attestation()
    monkeypatch.setattr(
        module,
        "release_runtime_source_attestation",
        lambda: release,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--attestation",
            str(attestation),
            "--private-evidence-root",
            str(tmp_path),
            "--out",
            str(output),
            "--run-id",
            "main-run",
            "--run-owned-check",
            "full_python_suite",
        ],
    )
    assert module._legacy_main_v4() == 1
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["status"] == "fail"
    assert manifest["checks"]["full_python_suite"]["status"] == "pass"
    assert manifest["checks"]["electron_visual_multiturn"]["status"] == "blocked"
    assert not any(
        manifest["checks"]["electron_visual_multiturn"]["assertions"].values()
    )
    assert manifest["run"]["run_id"] == "main-run"
    assert "__stdout_bytes" not in json.dumps(manifest)
    assert manifest["owned_executions"]["full_python_suite"][0]["pid"] == 777


def test_r20_release_source_attestation_reads_head_blobs(monkeypatch):
    module = load_module()
    module.release_runtime_source_attestation.cache_clear()
    blobs = {
        "vmlx_engine/server.py": b"committed server",
        "vmlx_engine/__init__.py": b"committed init",
        "panel/scripts/live-real-ui-model-proof.mjs": b"committed harness",
    }
    monkeypatch.setattr(module, "_git_head_blob", lambda path: blobs[path])
    monkeypatch.setattr(
        module,
        "_git_head_tree_attestation",
        lambda prefix, suffix=None: {
            "sha256": hashlib.sha256(prefix.encode()).hexdigest(),
            "file_count": 7,
        },
    )
    monkeypatch.setattr(
        module,
        "run_git",
        lambda *args: "a" * 40 if args[-1] == "HEAD" else "b" * 40,
    )
    attestation = module.release_runtime_source_attestation()
    assert (
        attestation["server_module_sha256"]
        == hashlib.sha256(b"committed server").hexdigest()
    )
    assert (
        attestation["package_init_sha256"]
        == hashlib.sha256(b"committed init").hexdigest()
    )
    module.release_runtime_source_attestation.cache_clear()


def test_r20_v5_main_owned_children_fail_closed_on_unowned_release_rows(
    tmp_path: Path,
):
    module = load_module()
    release = module.release_runtime_source_attestation()
    bundle, _ = bundle_attestation(module, tmp_path / "model")
    native_bundle, _ = native_bundle_attestation(
        module,
        tmp_path / "native-model",
    )
    private_root = tmp_path / "private"
    private_root.mkdir(mode=0o700)
    output = tmp_path / "manifest.json"
    release_diff_bytes = _v5_exact_release_diff(module)
    source = {
        "commit": release["source_commit"],
        "tree": release["source_tree"],
        "status_porcelain": "",
        "upstream_commit": release["source_commit"],
        "remote_main_commit": release["source_commit"],
        "main_only": 0,
        "branch_only": 0,
        "remote_identity": "jjang-ai/vmlx",
        "release_diff_bytes": release_diff_bytes,
        "release_diff_sha256": hashlib.sha256(release_diff_bytes).hexdigest(),
    }
    python = str(Path(sys.executable).resolve())

    def fixture_command(command_id: str, kind: str, *extra: str) -> dict:
        return {
            "command_id": command_id,
            "argv": [
                python,
                str(Path(__file__).resolve()),
                "--v5-fixture-command",
                kind,
                *extra,
            ],
            "cwd": ROOT,
            "env": {},
        }

    def owned_plan_provider(run_dir: Path, jang_root: Path) -> dict:
        del jang_root
        build_root = run_dir / "fixture-build"
        build_root.mkdir(mode=0o700)
        distribution_root = run_dir / "fixture-jang-dist"
        distribution_root.mkdir(mode=0o700)
        isolated = run_dir / "fixture-jang-installed"
        return {
            "full_python_suite": {
                "commands": [
                    fixture_command(
                        "full_python_suite",
                        "full_python_suite",
                    )
                ]
            },
            "full_panel_suite": {
                "commands": [
                    fixture_command(
                        "full_panel_suite",
                        "full_panel_suite",
                    )
                ]
            },
            "typecheck": {
                "commands": [fixture_command("typecheck", "typecheck")]
            },
            "production_build": {
                "commands": [
                    fixture_command(
                        "production_build",
                        "production_build",
                        "--output-root",
                        str(build_root),
                    )
                ],
                "output_root": str(build_root),
                "required_outputs": (
                    "main/index.mjs",
                    "preload/index.js",
                    "renderer/index.html",
                ),
            },
            "jang_runtime_provenance": {
                "commands": [
                    fixture_command(
                        "jang_build",
                        "jang_build",
                        "--distribution-root",
                        str(distribution_root),
                    ),
                    fixture_command(
                        "jang_venv",
                        "jang_venv",
                        "--isolated-venv",
                        str(isolated),
                    ),
                    fixture_command("jang_install", "jang_install"),
                    fixture_command(
                        "jang_import",
                        "jang_import",
                        "--isolated-venv",
                        str(isolated),
                    ),
                    fixture_command(
                        "jang_test",
                        "jang_test",
                        "--isolated-venv",
                        str(isolated),
                    ),
                ],
                "distribution_root": str(distribution_root),
                "isolated_venv": str(isolated),
                "test_manifest": [
                    "tests/test_laguna_loader.py",
                    "tests/test_jang_affine_storage.py",
                    "tests/test_jang_loader.py",
                ],
                "minimum_test_count": 3,
            },
        }

    def producer_plan_provider(args, run_dir: Path) -> dict:
        del run_dir
        return {
            producer: {
                "argv": [
                    python,
                    str(Path(__file__).resolve()),
                    "--v5-fixture-producer",
                    producer,
                    "--output-fd",
                    "{OUTPUT_FD}",
                    "--run-id",
                    "{RUN_ID}",
                    "--nonce",
                    "{NONCE}",
                    "--v5-session-binding-path",
                    "{SESSION_BINDING_PATH}",
                    "--v5-ready-path",
                    "{READY_PATH}",
                    "--v5-release-path",
                    "{RELEASE_PATH}",
                    "--v5-phase-control-dir",
                    "{PHASE_CONTROL_DIR}",
                    "--v5-paired-api-path",
                    "{PAIRED_API_PATH}",
                    "--v5-run-intent-path",
                    "{RUN_INTENT_PATH}",
                    "--v5-run-intent-sha256",
                    "{RUN_INTENT_SHA256}",
                    "--v5-active-phase-index",
                    "{ACTIVE_PHASE_INDEX}",
                    "--v5-ui-session-attestation-path",
                    "{UI_SESSION_ATTESTATION_PATH}",
                    "--v5-previous-backend-pid",
                    "{PREVIOUS_BACKEND_PID}",
                    "--v5-reuse-session-id",
                    "{REUSE_SESSION_ID}",
                    "--v5-reuse-session-attestation-path",
                    "{REUSE_SESSION_ATTESTATION_PATH}",
                    "--v5-source-commit",
                    "{SOURCE_COMMIT}",
                    "--v5-source-tree",
                    "{SOURCE_TREE}",
                    "--bundle-root",
                    str(args.bundle_root),
                    "--bundle-fingerprint",
                    bundle["fingerprint_sha256"],
                    "--native-bundle-root",
                    str(args.native_bundle_root),
                    "--native-bundle-fingerprint",
                    native_bundle["fingerprint_sha256"],
                    "--model",
                    args.model,
                    "--native-model",
                    args.native_model,
                    "--direct-base-url",
                    args.direct_base_url,
                    "--native-direct-base-url",
                    args.native_direct_base_url,
                    "--gateway-base-url",
                    args.gateway_base_url,
                    "--health-url",
                    args.health_url,
                    "--native-health-url",
                    args.native_health_url,
                    "--gateway-health-url",
                    args.gateway_health_url,
                    "--cdp-url",
                    args.cdp_url,
                    "--backend-pid",
                    str(args.backend_pid),
                    "--gateway-pid",
                    str(args.gateway_pid),
                    "--electron-pid",
                    str(args.electron_pid),
                ],
                "cwd": ROOT,
                "env": {},
                "ready_timeout_seconds": 5,
                "producer_timeout_seconds": 5,
            }
            for producer in module.V5_PRODUCER_NAMES
        }

    _, dom = _fixture_ui_capture(module, "unused", "0" * 32)
    dom["sourceCommit"] = release["source_commit"]
    # The production CDP snapshot does not currently own these facts.  Keep
    # this end-to-end fixture production-shaped so it cannot self-certify
    # settings, locale breadth, or minimum-width behavior.
    for key in ("locales", "supported_locales", "viewport", "settings"):
        dom.pop(key, None)
    runtime_package = tmp_path / "runtime/vmlx_engine"
    runtime_package.mkdir(parents=True)
    (runtime_package / "__init__.py").write_text(
        '__version__ = "1.6.20"\n',
        encoding="utf-8",
    )
    (runtime_package / "server.py").write_text(
        "def fixture_server():\n    return 'owned-v5'\n",
        encoding="utf-8",
    )
    runtime_attestation = module._v5_hash_python_runtime(
        runtime_package / "__init__.py"
    )
    assert runtime_attestation is not None
    source_attestation = {
        **runtime_attestation,
        "source_commit": release["source_commit"],
        "source_tree": release["source_tree"],
    }
    executable = module._v5_pin_regular_file(Path(python), executable=True)
    current_backend_pid = {"value": 9001}

    def process_observer(pid: int) -> dict:
        return {
            "pid": pid,
            "start_identity": f"fixture-start-{pid}",
            "argv": [python],
            "executable_path": executable["path"],
            "executable_sha256": executable["sha256"],
        }

    def listener_observer(host: str, port: int) -> dict:
        owner = (
            current_backend_pid["value"]
            if port in {8001, 8002}
            else 9002
        )
        return {"host": host, "port": port, "owner_pid": owner}

    def raw_runtime_observer(args, observed, snapshot) -> dict:
        del observed
        current_backend_pid["value"] = args.backend_pid
        bundle_provenance = deepcopy(snapshot["health_attestation"])
        executable_fingerprint = hashlib.sha256(
            str(Path(executable["path"]).absolute()).encode()
        ).hexdigest()
        health = {
            "model_bundle_provenance": bundle_provenance,
            "runtime_provenance": {
                "package_init_relpath": "vmlx_engine/__init__.py",
                "package_init_sha256": source_attestation[
                    "package_init_sha256"
                ],
                "server_module_relpath": "vmlx_engine/server.py",
                "server_module_sha256": source_attestation[
                    "server_module_sha256"
                ],
                "python_source_tree_sha256": source_attestation[
                    "python_source_tree_sha256"
                ],
                "python_source_file_count": source_attestation[
                    "python_source_file_count"
                ],
                "python_source_read_error_count": 0,
                "python_executable_fingerprint_sha256": executable_fingerprint,
                "model_bundle_provenance": deepcopy(bundle_provenance),
            },
        }
        return {
            "health_bytes": json.dumps(health, sort_keys=True).encode(),
            "dom_bytes": json.dumps(dom, sort_keys=True).encode(),
            "backend_pid": args.backend_pid,
            "gateway_pid": args.gateway_pid,
            "electron_pid": args.electron_pid,
        }

    hooks = {
        "raise_exceptions": True,
        "version_observer": lambda failures: {"version": module.VERSION},
        "private_root_observer": lambda configured, failures: configured,
        "source_observer": lambda: deepcopy(source),
        "source_attestation_observer": lambda: deepcopy(source_attestation),
        "jang_observer": lambda failures: {
            "version": module.JANG_VERSION,
            "commit": module.JANG_COMMIT,
            "tree": module.JANG_TREE,
        },
        "owned_check_plan_provider": owned_plan_provider,
        "producer_plan_provider": producer_plan_provider,
        "raw_runtime_observer": raw_runtime_observer,
        "process_observer": process_observer,
        "listener_observer": listener_observer,
    }
    argv = [
        "--private-evidence-root",
        str(private_root),
        "--out",
        str(output),
        "--run-id",
        "owned-v5",
        "--bundle-root",
        str(tmp_path / "model"),
        "--native-bundle-root",
        str(tmp_path / "native-model"),
        "--native-bundle-root",
        str(tmp_path / "native-model"),
        "--model",
        "fixture-laguna",
        "--native-model",
        "fixture-minimax-m3",
        "--direct-base-url",
        "http://127.0.0.1:8001",
        "--native-direct-base-url",
        "http://127.0.0.1:8002",
        "--gateway-base-url",
        "http://127.0.0.1:8080",
        "--health-url",
        "http://127.0.0.1:8001/health",
        "--native-health-url",
        "http://127.0.0.1:8002/health",
        "--gateway-health-url",
        "http://127.0.0.1:8080/health",
        "--cdp-url",
        "http://127.0.0.1:9335",
        "--backend-pid",
        "9001",
        "--gateway-pid",
        "9002",
        "--electron-pid",
        "9003",
        "--jang-source",
        str(tmp_path / "jang"),
    ]
    assert module.main(argv, _test_hooks=hooks) == 1
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["status"] == "fail"
    assert set(manifest["checks"]) == set(module.V5_REQUIRED_CHECKS)
    blocked = {
        name
        for name, row in manifest["checks"].items()
        if row["status"] != "pass"
    }
    assert blocked == {
        "settings_defaults_and_persistence",
        "i18n_katex_responsive_ui",
        "jang_runtime_provenance",
    }
    assert manifest["checks"]["cache_paged_off_ssd_partial"]["status"] == "pass"
    assert (
        manifest["checks"]["cache_paged_on_eviction_refault"]["status"]
        == "pass"
    )
    assert manifest["checks"]["cache_restart_and_size_eviction"]["status"] == "pass"
    assert set(
        manifest["checks"]["cache_restart_and_size_eviction"]["assertions"]
    ) == {
        "restart_disk_restore",
        "disk_size_limit_enforced",
        "disk_oldest_unused_evicted",
    }
    assert {
        (row["check"], row["assertion"])
        for row in manifest["deferred_assertions"]
    } == {
        ("cache_restart_and_size_eviction", "ram_percentage_limit_enforced"),
        ("cache_restart_and_size_eviction", "ram_oom_warning_checked"),
    }
    assert manifest["checks"]["turboquant_policy"]["assertions"][
        "explicit_off_honored"
    ]
    assert manifest["checks"]["turboquant_policy"]["assertions"][
        "unsupported_architecture_exception_honored"
    ]
    assert not manifest["checks"]["settings_defaults_and_persistence"][
        "assertions"
    ]["bundle_defaults_in_new_ui_session"]
    assert not manifest["checks"]["i18n_katex_responsive_ui"]["assertions"][
        "all_supported_locales_checked"
    ]
    assert not manifest["checks"]["i18n_katex_responsive_ui"]["assertions"][
        "minimum_window_width_checked"
    ]
    # The run itself completed; its release verdict remains fail-closed.
    assert manifest["completion"]["state"] == "complete"
    assert manifest["completion"]["run_digest"] == module._v5_manifest_digest(
        manifest
    )
    with pytest.raises(ValueError):
        module.consume_v5_release_manifest(
            output,
            expected_run_id="owned-v5",
            expected_commit=release["source_commit"],
            expected_tree=release["source_tree"],
        )
    with pytest.raises(FileExistsError):
        module._v5_atomic_write_manifest(
            output,
            manifest,
            manifest["run"]["nonce"],
        )


def test_r20_v5_six_phase_cache_facts_are_raw_and_do_not_invent_cross_surface_policy(
    tmp_path: Path,
):
    module = load_module()
    bundle, _ = bundle_attestation(module, tmp_path / "model")
    native_bundle, _ = native_bundle_attestation(
        module,
        tmp_path / "native-model",
    )
    representatives = {
        module.V5_PRIMARY_REPRESENTATIVE_ID: {
            "model": "fixture-model",
            "bundle": bundle,
        },
        module.V5_NATIVE_REPRESENTATIVE_ID: {
            "model": "fixture-native-model",
            "bundle": native_bundle,
        },
    }
    capture = _fixture_cache_capture(
        module,
        "cache-facts",
        "a" * 32,
        bundle_fingerprint=bundle["fingerprint_sha256"],
        native_bundle_fingerprint=native_bundle["fingerprint_sha256"],
    )
    phase2_summary = json.loads(
        base64.b64decode(capture["phases"][2]["summary_b64"])
    )
    phase2_partial = next(
        row
        for row in phase2_summary["requests"]
        if row["tag"] == "partial_b"
    )["last_cache_execution"]
    phase2_refault = phase2_summary["l2_size_eviction_observation"][
        "recent_refault_execution"
    ]["last_cache_execution"]
    assert phase2_partial["disk_blocks"] == 0
    assert "disk" not in phase2_partial["cache_detail"]
    assert all(
        not any((row.get("health_counter_deltas") or {}).values())
        for row in phase2_summary["requests"]
    )
    assert phase2_refault["disk_blocks"] > 0
    assert "disk" in phase2_refault["cache_detail"]
    raw = json.dumps(
        capture,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    facts, hashes = module._v5_cache_facts(
        [(capture, raw)],
        representatives,
    )
    assert {
        "paged_ram_disabled",
        "paged_ram_enabled",
        "ssd_l2_enabled",
        "longest_prefix_partial_block_hit",
        "uncached_suffix_prefilled",
        "cross_chat_reuse",
        "cross_session_reuse",
        "disk_refault_observed",
        "restart_disk_restore",
        "disk_size_limit_enforced",
        "disk_oldest_unused_evicted",
        "q4_default_when_supported",
        "encode_decode_live",
        "explicit_off_honored",
        "unsupported_architecture_exception_cache",
    } <= facts
    assert {
        "ram_percentage_limit_enforced",
        "ram_oom_warning_checked",
        "unsupported_architecture_exception_honored",
    }.isdisjoint(facts)
    assert hashes

    reordered = deepcopy(capture)
    reordered["phases"][0], reordered["phases"][1] = (
        reordered["phases"][1],
        reordered["phases"][0],
    )
    reordered_raw = json.dumps(
        reordered,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert module._v5_cache_facts(
        [(reordered, reordered_raw)],
        representatives,
    ) == (set(), [])

    def replace_phase_summary(capture_value, phase_index, mutate):
        changed = deepcopy(capture_value)
        phase = changed["phases"][phase_index]
        summary = json.loads(base64.b64decode(phase["summary_b64"]))
        mutate(summary)
        summary_bytes = json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        phase["summary_b64"] = _fixture_b64(summary_bytes)
        phase["artifact_manifest_b64"] = _fixture_json_b64(
            [
                {
                    "relative_path": "summary.json",
                    "sha256": hashlib.sha256(summary_bytes).hexdigest(),
                    "size": len(summary_bytes),
                }
            ]
        )
        return changed

    def rebind_phase2_eviction_attestation(changed):
        changed_phase2_bytes = base64.b64decode(
            changed["phases"][2]["summary_b64"]
        )
        changed_phase2_sha256 = hashlib.sha256(
            changed_phase2_bytes
        ).hexdigest()
        changed["phases"][3][
            "linked_store_summary_sha256"
        ] = changed_phase2_sha256
        phase3_bytes = base64.b64decode(
            changed["phases"][3]["summary_b64"]
        )
        changed["l2_size_eviction_attestation"] = (
            module._v5_derive_l2_size_eviction_attestation(
                run_id=changed["run_id"],
                nonce=changed["nonce"],
                phase2_summary=json.loads(changed_phase2_bytes),
                phase2_summary_sha256=changed_phase2_sha256,
                phase3_summary=json.loads(phase3_bytes),
                phase3_summary_sha256=hashlib.sha256(
                    phase3_bytes
                ).hexdigest(),
            )
        )
        return changed

    phase2_without_disk_refault = replace_phase_summary(
        capture,
        2,
        lambda summary: summary["l2_size_eviction_observation"][
            "recent_refault_execution"
        ]["last_cache_execution"].update(
            {
                "disk_blocks": 0,
                "cache_detail": "paged+tq-native",
            }
        ),
    )
    rebind_phase2_eviction_attestation(phase2_without_disk_refault)
    phase2_without_disk_refault_raw = json.dumps(
        phase2_without_disk_refault,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert module._v5_cache_facts(
        [
            (
                phase2_without_disk_refault,
                phase2_without_disk_refault_raw,
            )
        ],
        representatives,
    ) == (set(), [])

    phase2_counter_only_ram = replace_phase_summary(
        capture,
        2,
        lambda summary: (
            summary["l2_size_eviction_observation"]["recent_pre_refault"][
                "l1"
            ].update({"terminal_resident_payload_present": True}),
            summary["requests"][0]["health_counter_deltas"].update(
                {"scheduler_cache.evictions": 1}
            ),
        ),
    )
    rebind_phase2_eviction_attestation(phase2_counter_only_ram)
    phase2_counter_only_ram_raw = json.dumps(
        phase2_counter_only_ram,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert module._v5_cache_facts(
        [(phase2_counter_only_ram, phase2_counter_only_ram_raw)],
        representatives,
    ) == (set(), [])

    phase2_counter_only_disk = replace_phase_summary(
        capture,
        2,
        lambda summary: (
            summary["l2_size_eviction_observation"][
                "evicting_filler_fence"
            ].update({"request_correlated": False}),
            summary["requests"][0]["health_counter_deltas"].update(
                {"block_disk_cache.disk_evictions": 1}
            ),
        ),
    )
    rebind_phase2_eviction_attestation(phase2_counter_only_disk)
    phase2_counter_only_disk_raw = json.dumps(
        phase2_counter_only_disk,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert module._v5_cache_facts(
        [(phase2_counter_only_disk, phase2_counter_only_disk_raw)],
        representatives,
    ) == (set(), [])

    tampered_captures = []

    wrong_model = deepcopy(capture)
    wrong_model["phases"][0]["model"] = "wrong-model"
    tampered_captures.append(wrong_model)

    reused_session = deepcopy(capture)
    reused_session["phases"][-1]["session_id"] = reused_session["phases"][0][
        "session_id"
    ]
    tampered_captures.append(reused_session)

    repeated_pid = deepcopy(capture)
    repeated_pid["phases"][-1]["backend_pid"] = repeated_pid["phases"][0][
        "backend_pid"
    ]
    native_summary = json.loads(
        base64.b64decode(repeated_pid["phases"][-1]["summary_b64"])
    )
    native_summary["identity"]["observed_engine"]["pid"] = repeated_pid[
        "phases"
    ][0]["backend_pid"]
    repeated_pid["phases"][-1]["summary_b64"] = _fixture_json_b64(
        native_summary
    )
    tampered_captures.append(repeated_pid)

    off_not_explicit = deepcopy(capture)
    off_summary = json.loads(
        base64.b64decode(off_not_explicit["phases"][4]["summary_b64"])
    )
    off_summary["identity"]["cache_topology_provenance"]["configuration"][
        "configured"
    ]["kv_cache_quantization_explicit"] = False
    off_not_explicit["phases"][4]["summary_b64"] = _fixture_json_b64(
        off_summary
    )
    tampered_captures.append(off_not_explicit)

    native_tq_enabled = deepcopy(capture)
    native_tq_summary = json.loads(
        base64.b64decode(native_tq_enabled["phases"][5]["summary_b64"])
    )
    native_tq_summary["identity"]["cache_topology_provenance"][
        "configuration"
    ]["turboquant_kv_cache"]["enabled"] = True
    native_tq_enabled["phases"][5]["summary_b64"] = _fixture_json_b64(
        native_tq_summary
    )
    tampered_captures.append(native_tq_enabled)

    native_generic_contract = replace_phase_summary(
        capture,
        5,
        lambda summary: summary.__setitem__(
            "cache_contract_profile",
            "generic",
        ),
    )
    tampered_captures.append(native_generic_contract)

    tampered_captures.append(
        replace_phase_summary(
            capture,
            0,
            lambda summary: summary["health_final"].pop(
                "cache_storage_runtime_telemetry"
            ),
        )
    )
    tampered_captures.append(
        replace_phase_summary(
            capture,
            0,
            lambda summary: summary["health_final"][
                "cache_storage_runtime_telemetry"
            ]["turboquant_block_codec"].update(
                {
                    "encode": {
                        "calls": 0,
                        "blocks": 0,
                        "tokens": 0,
                        "last_event": None,
                    }
                }
            ),
        )
    )
    tampered_captures.append(
        replace_phase_summary(
            capture,
            1,
            lambda summary: summary["health_final"][
                "cache_storage_runtime_telemetry"
            ]["turboquant_block_codec"]["decode"]["last_event"].__setitem__(
                "key_bits_values",
                [8],
            ),
        )
    )
    tampered_captures.append(
        replace_phase_summary(
            capture,
            4,
            lambda summary: summary["health_final"][
                "cache_storage_runtime_telemetry"
            ]["turboquant_block_codec"].update(
                {
                    "encode": {
                        "calls": 1,
                        "blocks": 1,
                        "tokens": 64,
                        "last_event": {
                            "sequence": 1,
                            "boundary": "encode_tq_block",
                            "blocks": 1,
                            "tokens": 64,
                            "key_bits_values": [4],
                            "value_bits_values": [4],
                        },
                    }
                }
            ),
        )
    )

    for field, value in (
        ("family", "dsv4"),
        ("cache_type", "wrong"),
        ("schema", "wrong"),
    ):
        tampered_captures.append(
            replace_phase_summary(
                capture,
                5,
                lambda summary, field=field, value=value: summary["identity"][
                    "cache_topology_provenance"
                ]["configuration"]["native_cache"].__setitem__(field, value),
            )
        )

    tampered_captures.append(
        replace_phase_summary(
            capture,
            3,
            lambda summary: summary.__setitem__(
                "scenario_contract_ok",
                False,
            ),
        )
    )
    tampered_captures.append(
        replace_phase_summary(
            capture,
            3,
            lambda summary: summary["l2_restart_restore_observation"].__setitem__(
                "restart_probe_prefix_fingerprint_sha256",
                "9" * 64,
            ),
        )
    )
    tampered_captures.append(
        replace_phase_summary(
            capture,
            3,
            lambda summary: summary["l2_restart_restore_observation"][
                "restart_pre"
            ].__setitem__("longest_common_prefix_tokens", 0),
        )
    )
    for field, value in (
        ("cached_tokens", 0),
        ("disk_blocks", 0),
        ("uncached_prompt_tokens", 17),
    ):
        tampered_captures.append(
            replace_phase_summary(
                capture,
                3,
                lambda summary, field=field, value=value: summary[
                    "l2_restart_restore_observation"
                ]["restart_execution"]["last_cache_execution"].__setitem__(
                    field,
                    value,
                ),
            )
        )

    for tampered in tampered_captures:
        tampered_raw = json.dumps(
            tampered,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        assert module._v5_cache_facts(
            [(tampered, tampered_raw)],
            representatives,
        ) == (set(), [])


def test_r20_v5_run_intent_is_canonical_ordered_and_non_circular(
    tmp_path: Path,
):
    module = load_module()
    bundle, _ = bundle_attestation(module, tmp_path / "model")
    native_bundle, _ = native_bundle_attestation(
        module,
        tmp_path / "native-model",
    )
    common = {
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "gateway_base_url": "http://127.0.0.1:8080",
        "gateway_health_url": "http://127.0.0.1:8080/health",
    }
    expected = {
        module.V5_PRIMARY_REPRESENTATIVE_ID: {
            **common,
            "direct_base_url": "http://127.0.0.1:8001",
            "health_url": "http://127.0.0.1:8001/health",
            "direct_health_url": "http://127.0.0.1:8001/health",
            "model": "fixture-model",
            "model_bundle_path": bundle["model_bundle_path"],
            "bundle_fingerprint_sha256": bundle["fingerprint_sha256"],
            "native_cache_policy": bundle["derived"]["native_cache"],
        },
        module.V5_NATIVE_REPRESENTATIVE_ID: {
            **common,
            "direct_base_url": "http://127.0.0.1:8002",
            "health_url": "http://127.0.0.1:8002/health",
            "direct_health_url": "http://127.0.0.1:8002/health",
            "model": "fixture-native-model",
            "model_bundle_path": native_bundle["model_bundle_path"],
            "bundle_fingerprint_sha256": native_bundle[
                "fingerprint_sha256"
            ],
            "native_cache_policy": native_bundle["derived"]["native_cache"],
        },
    }
    run_context = {
        "run_id": "intent-fixture",
        "nonce": "c" * 32,
        "created_at": STAMP,
    }
    intent = module._v5_build_run_intent(run_context, expected)
    module._v5_validate_run_intent(intent, run_context, expected)
    assert intent["schema"] == module.V5_RUN_INTENT_SCHEMA
    assert intent["created_at"] == STAMP
    assert intent["direct_base_url"] == "http://127.0.0.1:8001"
    assert intent["native_direct_base_url"] == "http://127.0.0.1:8002"
    assert intent["native_direct_health_url"] == (
        "http://127.0.0.1:8002/health"
    )
    assert set(intent["harnesses"]) == {"ui", "api", "cache", "semantic"}
    assert all(
        set(row) == {"relative_path", "sha256"}
        for row in intent["harnesses"].values()
    )
    assert [
        row["phase_name"] for row in intent["phase_plan"]
    ] == [
        "primary_ssd_only_store",
        "primary_ssd_only_restart_probe",
        "primary_paged_on_store",
        "primary_paged_on_restart_probe",
        "primary_tq_off",
        "native_exception",
    ]
    assert [
        row["tq_policy"] for row in intent["phase_plan"]
    ] == [
        "auto-model-safe-required",
        "auto-model-safe-required",
        "auto-model-safe-required",
        "auto-model-safe-required",
        "explicit-off",
        "native-suppressed",
    ]
    assert [
        row["operation"] for row in intent["phase_plan"]
    ] == [
        "store",
        "probe",
        "store-evict-refault",
        "probe",
        "store-probe",
        "switch-validate",
    ]
    assert intent["phase_plan"][4]["cache_policy"] == "ssd-only"
    assert [
        row["ui_turn_count"] for row in intent["phase_plan"]
    ] == [1, 1, 1, 1, 1, 3]
    assert [
        row["ui_action_profile"] for row in intent["phase_plan"]
    ] == [
        "primary-reasoning-render-store",
        "primary-tool-restart-probe",
        "primary-history-paged-evict-refault",
        "primary-restart-followup",
        "primary-tq-off-probe",
        "native-three-turn-switch",
    ]
    assert [
        row["api_action_profile"] for row in intent["phase_plan"]
    ] == [
        "full-agentic-plus-cache-store",
        "cache-probe",
        "cache-evict-refault",
        "cache-restart-probe",
        "cache-tq-off-store-probe",
        "full-agentic-native-cache",
    ]
    assert intent["l2_size_eviction_requirements"] == (
        module.V5_L2_SIZE_EVICTION_REQUIREMENTS
    )
    assert intent["phase_plan"][0]["native_cache_policy"] == (
        bundle["derived"]["native_cache"]
    )
    assert intent["phase_plan"][-1]["native_cache_policy"] == (
        "dsv4_composite"
    )
    unsigned = deepcopy(intent)
    canonical_sha256 = unsigned.pop("canonical_sha256")
    assert canonical_sha256 == module._canonical_json_sha256(unsigned)

    tampered = deepcopy(intent)
    tampered["phase_plan"][4]["tq_policy"] = "q4-required"
    with pytest.raises(RuntimeError, match="canonical plan"):
        module._v5_validate_run_intent(tampered, run_context, expected)

    same_origin = deepcopy(expected)
    same_origin[module.V5_NATIVE_REPRESENTATIVE_ID]["direct_base_url"] = (
        "http://127.0.0.1:8001"
    )
    same_origin[module.V5_NATIVE_REPRESENTATIVE_ID]["health_url"] = (
        "http://127.0.0.1:8001/health"
    )
    same_origin[module.V5_NATIVE_REPRESENTATIVE_ID]["direct_health_url"] = (
        "http://127.0.0.1:8001/health"
    )
    with pytest.raises(ValueError, match="origins must be distinct"):
        module._v5_build_run_intent(run_context, same_origin)

    invalid_health = deepcopy(expected)
    invalid_health[module.V5_NATIVE_REPRESENTATIVE_ID][
        "direct_health_url"
    ] = "http://127.0.0.1:8003/health"
    with pytest.raises(ValueError, match="exact /health"):
        module._v5_build_run_intent(run_context, invalid_health)


def test_r20_v5_cache_gate_scenarios_bind_strict_l2_phase_pair():
    module = load_module()
    assert [
        module._v5_cache_gate_scenario(phase)
        for phase in module.V5_CACHE_PHASES
    ] == [
        "standard",
        "standard",
        "store-evict-refault",
        "restart-restore",
        "standard",
        "standard",
    ]


def test_r20_v5_python_suite_accepts_selected_pass_skip_summary():
    module = load_module()
    execution = {
        "exit_code": 0,
        "__stdout_bytes": (
            b"collected 6961 items / 92 deselected / 6869 selected\n"
            b"subprocess: collected 25 items\n"
            b"subprocess: 25 passed in 0.10s\n"
            b"6761 passed, 108 skipped, 92 deselected in 123.45s\n"
        ),
        "__stderr_bytes": b"",
    }
    facts, details = module._v5_owned_check_facts(
        "full_python_suite",
        [execution],
        {},
        {},
    )
    assert facts == set(module.V5_RELEASE_ASSERTIONS["full_python_suite"])
    assert details == {}


@pytest.mark.parametrize(
    "summary",
    [
        b"collected 6961 items / 92 deselected / 6869 selected\n"
        b"6760 passed, 108 skipped, 92 deselected in 123.45s\n",
        b"collected 6961 items / 92 deselected / 6869 selected\n"
        b"6760 passed, 108 skipped, 1 failed, 92 deselected in 123.45s\n",
    ],
)
def test_r20_v5_python_suite_rejects_incomplete_or_failed_summary(summary):
    module = load_module()
    facts, details = module._v5_owned_check_facts(
        "full_python_suite",
        [
            {
                "exit_code": 0,
                "__stdout_bytes": summary,
                "__stderr_bytes": b"",
            }
        ],
        {},
        {},
    )
    assert facts == set()
    assert details == {}


def test_r20_v5_sampling_log_parser_binds_all_request_identities():
    module = load_module()
    line = (
        "INFO Resolved sampling kwargs route=/v1/chat/completions "
        "model=fixture-laguna proof_request_id=proof-1 "
        "request_id=request-1 message_id=message-1 "
        "kwargs={'temperature': 0.6, 'max_tokens': 2, "
        "'enable_thinking': False}"
    )
    parsed = module._v5_parse_resolved_sampling_log(
        line,
        route="/v1/chat/completions",
        expected_models={"fixture-laguna"},
        proof_request_id="proof-1",
        request_id="request-1",
        message_id="message-1",
    )
    assert parsed is not None
    assert parsed["proof_request_id"] == "proof-1"
    assert parsed["request_id"] == "request-1"
    assert parsed["message_id"] == "message-1"
    assert parsed["values"]["max_tokens"] == 2
    assert parsed["values"]["enable_thinking"] is False


def test_r20_v5_reasoning_mode_request_maps_auto_on_off_labels():
    module = load_module()
    payload = {
        "model": "fixture",
        "enable_thinking": True,
        "think": True,
    }

    auto, auto_mode = module._v5_reasoning_mode_request(
        "chat", payload, "stream-flow-round1"
    )
    on, on_mode = module._v5_reasoning_mode_request(
        "chat", payload, "nonstream-flow-round2"
    )
    off, off_mode = module._v5_reasoning_mode_request(
        "ollama", payload, "stream-flow-round3"
    )

    assert (auto_mode, auto.get("enable_thinking"), auto.get("think")) == (
        "auto",
        None,
        None,
    )
    assert (on_mode, on["enable_thinking"], on.get("think")) == (
        "on",
        True,
        None,
    )
    assert (off_mode, off["think"], off.get("enable_thinking")) == (
        "off",
        False,
        None,
    )


def test_r20_v5_transmitted_request_metadata_replaces_pretransform_summaries():
    module = load_module()
    matrix_result = {
        "flows": {
            "direct": {
                "chat": {
                    "stream": {
                        "pass": True,
                        "requests": [
                            {"stale": 1},
                            {"stale": 2},
                            {"stale": 3},
                        ],
                    }
                }
            }
        }
    }
    records = [
        {
            "route": "direct",
            "protocol": "chat",
            "capture_label": f"stream-flow-round{stage}",
            "request": {
                "model": "fixture",
                **(
                    {}
                    if stage == 1
                    else {"enable_thinking": stage == 2}
                ),
            },
        }
        for stage in (1, 2, 3)
    ]

    class Harness:
        @staticmethod
        def _request_public(stage, request, *, protocol=""):
            return {
                "stage": stage,
                "model": request["model"],
                "protocol": protocol,
                "enable_thinking": request.get("enable_thinking"),
            }

    module._v5_bind_transmitted_request_metadata(
        matrix_result,
        records,
        Harness,
    )

    assert matrix_result["flows"]["direct"]["chat"]["stream"]["requests"] == [
        {
            "stage": 1,
            "model": "fixture",
            "protocol": "chat",
            "enable_thinking": None,
        },
        {
            "stage": 2,
            "model": "fixture",
            "protocol": "chat",
            "enable_thinking": True,
        },
        {
            "stage": 3,
            "model": "fixture",
            "protocol": "chat",
            "enable_thinking": False,
        },
    ]
    assert (
        SCRIPT.read_text(encoding="utf-8").count(
            "_v5_bind_transmitted_request_metadata("
        )
        == 2
    )


def test_r20_v5_replay_payload_is_rebound_to_the_transmitted_off_body():
    module = load_module()
    retained = {
        "model": "fixture",
        "stream": False,
        "enable_thinking": True,
        "messages": [{"role": "user", "content": "continue"}],
    }
    transmitted, mode = module._v5_reasoning_mode_request(
        "chat",
        retained,
        "nonstream-flow-round3",
    )

    module._v5_rebind_flow_payload_to_transmitted_request(
        retained,
        transmitted,
    )

    assert mode == "off"
    assert retained == transmitted
    assert retained["enable_thinking"] is False
    assert (
        SCRIPT.read_text(encoding="utf-8").count(
            "_v5_rebind_flow_payload_to_transmitted_request("
        )
        == 2
    )


def test_r20_v5_sampling_log_lookup_survives_ring_wrap_and_rejects_duplicates(
    monkeypatch,
):
    module = load_module()
    binding = {
        "model": "fixture-laguna",
        "model_bundle_path": "/models/fixture-laguna",
        "cdp_url": "http://127.0.0.1:9335",
        "session_id": "session-1",
    }
    before = [f"old-{index}" for index in range(2000)]
    target = (
        "Resolved sampling kwargs route=/v1/chat/completions "
        "model=fixture-laguna proof_request_id=proof-1 "
        "request_id=request-1 message_id=message-1 "
        "kwargs={'temperature': 0.6, 'max_tokens': 2, "
        "'enable_thinking': False}"
    )
    current = before[250:] + [target]
    monkeypatch.setattr(
        module,
        "_v5_cdp_session_logs",
        lambda _cdp, _session: current,
    )
    observed, logs = module._v5_wait_for_resolved_sampling_log(
        binding,
        before,
        route="/v1/chat/completions",
        proof_request_id="proof-1",
        request_id="request-1",
        message_id="message-1",
        timeout=0.1,
    )
    assert observed["values"]["temperature"] == 0.6
    assert logs == current

    monkeypatch.setattr(
        module,
        "_v5_cdp_session_logs",
        lambda _cdp, _session: current + [target],
    )
    with pytest.raises(RuntimeError, match="multiple resolved requests"):
        module._v5_wait_for_resolved_sampling_log(
            binding,
            before,
            route="/v1/chat/completions",
            proof_request_id="proof-1",
            request_id="request-1",
            message_id="message-1",
            timeout=0.1,
        )


def _run_sampling_capture_fixture(
    module,
    monkeypatch,
    *,
    health_before: dict,
    health_after: dict,
):
    health_rows = [
        json.dumps({"effective_defaults": health_before}).encode(),
        json.dumps({"effective_defaults": health_after}).encode(),
    ]
    requests = []

    class ProtocolClient:
        def __init__(self, *_args, **_kwargs):
            self.headers = {}

    class Harness:
        pass

    Harness.ProtocolClient = ProtocolClient
    monkeypatch.setattr(
        module,
        "_v5_loopback_http_get",
        lambda _url: health_rows.pop(0),
    )
    monkeypatch.setattr(module, "_v5_cdp_session_logs", lambda *_args: [])

    def original_send(
        _client,
        protocol,
        request,
        stream,
        *,
        capture_label,
    ):
        assert protocol == "chat"
        assert stream is False
        assert capture_label.startswith("sampling_")
        requests.append(deepcopy(request))
        return {
            "status_code": 200,
            "errors": [],
            "terminals": [{"status": "completed"}],
        }

    def resolved(_binding, _before, **kwargs):
        request = requests[-1]
        values = {
            key: request.get(key, health_before.get(key))
            for key in ("temperature", "top_p", "top_k", "min_p")
            if request.get(key, health_before.get(key)) is not None
        }
        values.update(
            {
                "max_tokens": request["max_tokens"],
                "enable_thinking": request["enable_thinking"],
            }
        )
        return (
            {
                "route": kwargs["route"],
                "model": "fixture-model",
                "proof_request_id": kwargs["proof_request_id"],
                "request_id": kwargs["request_id"],
                "message_id": kwargs["message_id"],
                "values": values,
                "line_sha256": "a" * 64,
                "line_b64": module._v5_encode_bytes(b"fixture"),
            },
            [],
        )

    monkeypatch.setattr(module, "_v5_wait_for_resolved_sampling_log", resolved)
    capture = module._v5_api_sampling_capture(
        Harness,
        original_send,
        {
            "direct_base_url": "http://127.0.0.1:18088",
            "health_url": "http://127.0.0.1:18088/health",
            "cdp_url": "http://127.0.0.1:19358",
            "session_id": "fixture-session",
            "model": "fixture-model",
            "model_bundle_path": "/models/fixture-model",
        },
    )
    return capture, requests


def test_r20_v5_sampling_capture_allows_dynamic_metal_output_cap(
    monkeypatch,
):
    module = load_module()
    before = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.0,
        "max_output_tokens": 14518,
    }
    after = {**before, "max_output_tokens": 14504}
    capture, requests = _run_sampling_capture_fixture(
        module,
        monkeypatch,
        health_before=before,
        health_after=after,
    )
    assert capture["stable_sampling_defaults_before"] == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.0,
    }
    assert (
        capture["stable_sampling_defaults_after"]
        == capture["stable_sampling_defaults_before"]
    )
    assert capture["health_effective_defaults_after"]["max_output_tokens"] == 14504
    assert requests[1]["temperature"] == 0.123
    assert "temperature" not in requests[0]
    assert "temperature" not in requests[2]
    assert {
        request["max_tokens"] for request in requests
    } == {module.V5_SAMPLING_PROBE_MAX_TOKENS}


def test_r20_v5_sampling_capture_rejects_persisted_sampler_override(
    monkeypatch,
):
    module = load_module()
    before = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.0,
        "max_output_tokens": 14518,
    }
    after = {**before, "temperature": 0.123, "max_output_tokens": 14504}
    with pytest.raises(
        RuntimeError,
        match="per-request sampling override changed server defaults",
    ):
        _run_sampling_capture_fixture(
            module,
            monkeypatch,
            health_before=before,
            health_after=after,
        )


def test_r20_v5_api_sampling_uses_bundle_sampler_keys_not_probe_output_fields(
    tmp_path: Path,
):
    module = load_module()
    bundle_root = tmp_path / "laguna"
    bundle_attestation(module, bundle_root)
    laguna_defaults = {
        "temperature": 0.65,
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.01,
        "repetition_penalty": 1.05,
        "max_new_tokens": 4096,
    }
    (bundle_root / "generation_config.json").write_text(
        json.dumps(laguna_defaults),
        encoding="utf-8",
    )
    bundle = module._read_bundle_directory_snapshot(bundle_root.resolve())
    assert bundle is not None
    captures = []
    for phase_index in range(len(module.V5_CACHE_PHASES)):
        artifact = _fixture_api_capture(
            module,
            "sampling-run",
            "sampling-nonce",
            phase_index=phase_index,
        )
        if phase_index == 0:
            artifact["sampling_b64"] = _fixture_json_b64(
                _fixture_sampling_attestation(
                    module,
                    defaults=laguna_defaults,
                )
            )
        raw = json.dumps(
            artifact,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        captures.append((artifact, raw))
    _protocol, facts, _hashes = module._v5_api_facts(captures, bundle)
    assert "bundle_defaults_in_api" in facts
    assert "api_request_override_request_scoped" in facts


def test_r20_v5_ui_facts_do_not_invent_unobserved_settings_or_layout(
    tmp_path: Path,
):
    module = load_module()
    bundle, _ = bundle_attestation(module, tmp_path / "model")
    captures = []
    phase_observations = []
    primary_session = "ui-primary"
    native_session = "ui-native"
    dom = {}
    for phase in module.V5_CACHE_PHASES:
        capture, phase_dom = _fixture_ui_capture(
            module,
            "ui-facts",
            "b" * 32,
            session_id=(
                native_session
                if phase["index"] == 5
                else primary_session
            ),
            phase_index=phase["index"],
        )
        if phase["index"] == 5:
            dom = phase_dom
        raw = json.dumps(
            capture,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        captures.append((capture, raw))
        phase_dom_bytes = json.dumps(
            phase_dom,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        phase_observations.append(
            {
                "phase_index": phase["index"],
                "observation": {
                    "dom": phase_dom,
                    "dom_bytes_sha256": hashlib.sha256(
                        phase_dom_bytes
                    ).hexdigest(),
                },
            }
        )
    for key in ("locales", "supported_locales", "viewport", "settings"):
        dom.pop(key, None)
    dom_bytes = json.dumps(
        dom,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    facts, hashes = module._v5_ui_facts(
        captures,
        {
            "dom": dom,
            "dom_bytes_sha256": hashlib.sha256(dom_bytes).hexdigest(),
            "phase_observations": phase_observations,
        },
        bundle,
    )
    assert {
        "real_start_button",
        "minimum_three_turns",
        "reasoning_rail",
        "visible_content",
        "katex_rendered",
        "currency_preserved",
        "markdown_rendered",
    } <= facts
    assert {
        "all_supported_locales_checked",
        "minimum_window_width_checked",
        "bundle_defaults_in_new_ui_session",
        "ui_override_session_scoped",
        "ui_override_restart_persisted",
        "max_context_output_distinct",
        "preview_argv_health_parity",
    }.isdisjoint(facts)
    assert hashes

    broken_rendering = deepcopy(captures)
    phase_zero_capture = broken_rendering[0][0]
    phase_zero_source = json.loads(
        base64.b64decode(
            phase_zero_capture["source_proof_b64"],
            validate=True,
        )
    )
    phase_zero_user = phase_zero_source["renderedDom"]["userMessages"][0]
    phase_zero_user["katexCount"] = 0
    phase_zero_user["html"] = "<p>Keep literal $43 and render raw TeX.</p>"
    phase_zero_capture["source_proof_b64"] = _fixture_json_b64(
        phase_zero_source
    )
    broken_rendering[0] = (
        phase_zero_capture,
        json.dumps(
            phase_zero_capture,
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
    )
    broken_facts, _ = module._v5_ui_facts(
        broken_rendering,
        {
            "dom": dom,
            "dom_bytes_sha256": hashlib.sha256(dom_bytes).hexdigest(),
            "phase_observations": phase_observations,
        },
        bundle,
    )
    assert {
        "rendering",
        "katex_rendered",
        "currency_preserved",
        "markdown_rendered",
    }.isdisjoint(broken_facts)

    overwritten = deepcopy(captures)
    phase_five_capture = overwritten[5][0]
    source_proof = json.loads(
        base64.b64decode(
            phase_five_capture["source_proof_b64"],
            validate=True,
        )
    )
    source_proof["cacheRequestEvidence"][0]["healthAfter"]["scheduler"][
        "last_cache_execution"
    ]["request_id"] = "resp-unrelated-later-request"
    phase_five_capture["source_proof_b64"] = _fixture_json_b64(source_proof)
    overwritten[5] = (
        phase_five_capture,
        json.dumps(
            phase_five_capture,
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
    )
    assert module._v5_ui_facts(
        overwritten,
        {
            "dom": dom,
            "dom_bytes_sha256": hashlib.sha256(dom_bytes).hexdigest(),
            "phase_observations": phase_observations,
        },
        bundle,
    ) == (set(), [])


def test_r20_v5_ui_facts_preserve_interleaved_reasoning_boundaries(
    tmp_path: Path,
):
    module = load_module()
    bundle, _ = bundle_attestation(module, tmp_path / "model")
    captures = []
    phase_observations = []
    for phase in module.V5_CACHE_PHASES:
        capture, dom = _fixture_ui_capture(
            module,
            "ui-interleaved-reasoning",
            "c" * 32,
            session_id=(
                "ui-interleaved-native"
                if phase["index"] == 5
                else "ui-interleaved-primary"
            ),
            phase_index=phase["index"],
            evidence_root=tmp_path / f"phase-{phase['index']}",
        )
        if phase["index"] == 5:
            first = "reason-before-tool"
            second = "reason-after-tool"
            turn = capture["turns"][0]
            events = [
                json.loads(row)
                for row in base64.b64decode(
                    turn["events_b64"], validate=True
                ).decode().splitlines()
                if row
            ]
            events.insert(
                3,
                {"type": "reasoning_delta", "text": second},
            )
            events[0]["text"] = first
            for seq, event in enumerate(events):
                event["seq"] = seq
            turn["events_b64"] = _fixture_jsonl_b64(events)
            source_proof = json.loads(
                base64.b64decode(
                    capture["source_proof_b64"], validate=True
                )
            )
            source_proof["persistedReasoningByMessage"][0] = [first, second]
            capture["source_proof_b64"] = _fixture_json_b64(source_proof)
            dom["messages"][0]["reasoning_text"] = f"{first}\n{second}"
        raw = json.dumps(
            capture,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        captures.append((capture, raw))
        dom_bytes = json.dumps(
            dom,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        phase_observations.append(
            {
                "phase_index": phase["index"],
                "observation": {
                    "dom": dom,
                    "dom_bytes_sha256": hashlib.sha256(dom_bytes).hexdigest(),
                },
            }
        )
    runtime = {
        "phase_observations": phase_observations,
    }
    facts, hashes = module._v5_ui_facts(captures, runtime, bundle)
    assert "reasoning_tool_reasoning_tool_answer" in facts
    assert hashes

    mismatched = deepcopy(captures)
    native_capture = mismatched[5][0]
    source_proof = json.loads(
        base64.b64decode(
            native_capture["source_proof_b64"], validate=True
        )
    )
    source_proof["persistedReasoningByMessage"][0][1] += "-changed"
    native_capture["source_proof_b64"] = _fixture_json_b64(source_proof)
    mismatched[5] = (
        native_capture,
        json.dumps(
            native_capture,
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
    )
    assert module._v5_ui_facts(mismatched, runtime, bundle) == (set(), [])


def test_r20_v5_ui_facts_detects_common_namespace_translation_key(
    tmp_path: Path,
):
    module = load_module()
    bundle, _ = bundle_attestation(module, tmp_path / "model")
    captures = []
    phase_observations = []
    for phase in module.V5_CACHE_PHASES:
        capture, dom = _fixture_ui_capture(
            module,
            "i18n-adversary",
            "d" * 32,
            session_id=("i18n-native" if phase["index"] == 5 else "i18n-primary"),
            phase_index=phase["index"],
            evidence_root=tmp_path / f"phase-{phase['index']}",
        )
        raw = json.dumps(capture, sort_keys=True, separators=(",", ":")).encode()
        captures.append((capture, raw))
        dom_bytes = json.dumps(dom, sort_keys=True, separators=(",", ":")).encode()
        phase_observations.append(
            {
                "phase_index": phase["index"],
                "observation": {
                    "dom": dom,
                    "dom_bytes_sha256": hashlib.sha256(dom_bytes).hexdigest(),
                },
            }
        )
    runtime = {
        "dom": phase_observations[5]["observation"]["dom"],
        "dom_bytes_sha256": phase_observations[5]["observation"][
            "dom_bytes_sha256"
        ],
        "phase_observations": phase_observations,
    }
    facts, _ = module._v5_ui_facts(captures, runtime, bundle)
    assert "no_raw_translation_keys" in facts
    assert "all_supported_locales_checked" in facts

    leaked_runtime = deepcopy(runtime)
    leaked_dom = leaked_runtime["phase_observations"][5]["observation"]["dom"]
    # CSS text-transform can uppercase a leaked i18n key in innerText. The
    # detector must compare case-insensitively while retaining captured text.
    leaked_dom["text"] += " COMMON.STREAM"
    leaked_bytes = json.dumps(
        leaked_dom,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    leaked_runtime["phase_observations"][5]["observation"][
        "dom_bytes_sha256"
    ] = hashlib.sha256(leaked_bytes).hexdigest()
    leaked_runtime["dom"] = leaked_dom
    leaked_runtime["dom_bytes_sha256"] = hashlib.sha256(leaked_bytes).hexdigest()
    leaked_facts, _ = module._v5_ui_facts(captures, leaked_runtime, bundle)
    assert "no_raw_translation_keys" not in leaked_facts


def test_r20_v5_ui_defaults_require_exact_wire_request_correlation(
    tmp_path: Path,
):
    module = load_module()
    bundle, _ = bundle_attestation(module, tmp_path / "model")
    captures = []
    phase_observations = []
    for phase in module.V5_CACHE_PHASES:
        capture, dom = _fixture_ui_capture(
            module,
            "defaults-correlation",
            "e" * 32,
            session_id=(
                "defaults-native" if phase["index"] == 5 else "defaults-primary"
            ),
            phase_index=phase["index"],
            evidence_root=tmp_path / f"defaults-phase-{phase['index']}",
        )
        if phase["index"] == 0:
            source_proof = json.loads(
                base64.b64decode(capture["source_proof_b64"], validate=True)
            )
            source_proof.update(
                {
                    "bundleGenerationContract": {
                        "defaults": {"temperature": 0.6, "topP": 0.95}
                    },
                    "rendererGenerationDefaults": {
                        "temperature": 0.6,
                        "topP": 0.95,
                    },
                    "chatSettingsDom": {
                        "values": {"temperature": 0.6, "topP": 0.95},
                        "maxTokens": {"value": "", "placeholder": "model default"},
                    },
                    "server": {
                        "health": {
                            "effective_defaults": {
                                "temperature": 0.6,
                                "top_p": 0.95,
                            }
                        }
                    },
                    "chatOverrides": {},
                }
            )
            source_proof["requestContract"]["samplingOverrides"] = {}
            for record in source_proof["resolvedSamplingRecords"]:
                record["values"] = {"temperature": 0.6, "top_p": 0.95}
            capture["source_proof_b64"] = _fixture_json_b64(source_proof)
        raw = json.dumps(capture, sort_keys=True, separators=(",", ":")).encode()
        captures.append((capture, raw))
        dom_bytes = json.dumps(dom, sort_keys=True, separators=(",", ":")).encode()
        phase_observations.append(
            {
                "phase_index": phase["index"],
                "observation": {
                    "dom": dom,
                    "dom_bytes_sha256": hashlib.sha256(dom_bytes).hexdigest(),
                },
            }
        )
    runtime = {
        "dom": phase_observations[5]["observation"]["dom"],
        "dom_bytes_sha256": phase_observations[5]["observation"][
            "dom_bytes_sha256"
        ],
        "phase_observations": phase_observations,
    }
    facts, _ = module._v5_ui_facts(captures, runtime, bundle)
    assert "bundle_defaults_in_new_ui_session" in facts

    uncorrelated = deepcopy(captures)
    phase_zero = uncorrelated[0][0]
    source_proof = json.loads(
        base64.b64decode(phase_zero["source_proof_b64"], validate=True)
    )
    source_proof["requestCorrelation"]["status"] = "partial"
    phase_zero["source_proof_b64"] = _fixture_json_b64(source_proof)
    uncorrelated[0] = (
        phase_zero,
        json.dumps(phase_zero, sort_keys=True, separators=(",", ":")).encode(),
    )
    partial_facts, _ = module._v5_ui_facts(uncorrelated, runtime, bundle)
    assert "bundle_defaults_in_new_ui_session" not in partial_facts


@pytest.mark.parametrize(
    ("protocol", "old", "new"),
    [
        ("chat", b'"finish_reason":"stop"', b'"finish_reason":"length"'),
        (
            "responses",
            b'"status":"completed"',
            b'"status":"incomplete"',
        ),
        (
            "anthropic",
            b'"stop_reason":"end_turn"',
            b'"stop_reason":"max_tokens"',
        ),
        ("ollama", b'"done_reason":"stop"', b'"done_reason":"length"'),
    ],
)
def test_r20_v5_rejects_non_success_protocol_terminals(
    protocol: str,
    old: bytes,
    new: bytes,
):
    module = load_module()
    valid = _fixture_protocol_response(protocol, 3)
    assert module._parse_raw_protocol_stream_v5(protocol, valid) is not None
    assert old in valid
    invalid = valid.replace(old, new, 1)
    assert module._parse_raw_protocol_stream_v5(protocol, invalid) is None


def test_r20_v5_nonstream_responses_checks_only_visible_control_markup():
    module = load_module()
    reasoning_marker = (
        "Privately prepare one native <tool_call><function=run_command>."
    )
    tool_response = {
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": reasoning_marker}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "run_command",
                "arguments": '{"command":"pwd"}',
            },
        ],
    }
    assert module._parse_raw_protocol_nonstream_v5(
        "responses",
        json.dumps(tool_response).encode(),
    ) == tool_response

    visible_leak = deepcopy(tool_response)
    visible_leak["output"].append(
        {
            "type": "message",
            "content": [{"type": "output_text", "text": reasoning_marker}],
        }
    )
    assert (
        module._parse_raw_protocol_nonstream_v5(
            "responses",
            json.dumps(visible_leak).encode(),
        )
        is None
    )


def test_r20_v5_child_environment_is_fixed_and_rejects_injection(
    tmp_path: Path,
):
    module = load_module()
    env = module._v5_minimal_env(tmp_path, {"VMLINUX_RELEASE_TEST": "1"})
    assert env["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"
    assert env["VMLINUX_RELEASE_TEST"] == "1"
    for name in (
        "PATH",
        "HOME",
        "PYTHONPATH",
        "PYTEST_ADDOPTS",
        "NODE_OPTIONS",
        "npm_config_prefix",
        "DYLD_INSERT_LIBRARIES",
        "LD_PRELOAD",
        "BASH_ENV",
        "ZDOTDIR",
    ):
        with pytest.raises(ValueError):
            module._v5_minimal_env(tmp_path, {name: "unsafe"})


def test_r20_v5_canonicalizes_retained_pids_for_the_ui_worker(monkeypatch):
    module = load_module()
    monkeypatch.delenv("VMLINUX_REAL_UI_RETAINED_PIDS", raising=False)
    monkeypatch.setenv("VMLX_REAL_UI_RETAINED_PIDS", "49053, 61126")
    assert module._v5_release_retained_pid_environment() == "49053,61126"
    plans = module._v5_default_producer_plans(
        argparse.Namespace(
            bundle_root=Path("/tmp/primary"),
            native_bundle_root=Path("/tmp/native"),
            direct_base_url="http://127.0.0.1:18084",
            native_direct_base_url="http://127.0.0.1:18085",
            gateway_base_url="http://127.0.0.1:18083",
            health_url="http://127.0.0.1:18084/health",
            native_health_url="http://127.0.0.1:18085/health",
            gateway_health_url="http://127.0.0.1:18083/health",
            cdp_url="http://127.0.0.1:19357",
            electron_pid=4728,
            gateway_pid=4728,
            model="primary",
            native_model="native",
        )
    )
    assert plans["ui"]["env"] == {
        "VMLINUX_REAL_UI_RETAINED_PIDS": "49053,61126"
    }
    assert plans["api"]["env"] == {}
    assert plans["cache"]["env"] == {}

    harness_env = module._v5_ui_harness_environment(
        Path("/tmp/private-run"),
        "/tmp/private-run/private-cache-attestation.token",
    )
    assert harness_env["VMLINUX_REAL_UI_RETAINED_PIDS"] == "49053,61126"
    assert harness_env[module.PRIVATE_CACHE_ATTESTATION_TOKEN_FILE_ENV] == (
        "/tmp/private-run/private-cache-attestation.token"
    )
    assert harness_env["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"

    monkeypatch.setenv("VMLINUX_REAL_UI_RETAINED_PIDS", "49053 49053")
    with pytest.raises(RuntimeError, match="unique retained process PIDs"):
        module._v5_release_retained_pid_environment()

    monkeypatch.setenv("VMLINUX_REAL_UI_RETAINED_PIDS", "")
    monkeypatch.delenv("VMLX_REAL_UI_RETAINED_PIDS", raising=False)
    with pytest.raises(RuntimeError, match="unique retained process PIDs"):
        module._v5_release_retained_pid_environment()


def test_r20_v5_ui_emits_dsv4_pool_quant_expectation_only_for_dsv4():
    module = load_module()
    dsv4 = {"derived": {"native_cache": "dsv4_composite"}}
    for not_dsv4 in (
        {"derived": {"native_cache": "standard_kv"}},
        {"derived": {"native_cache": "minimax_m3_sparse"}},
        {"derived": {}},
        {},
    ):
        assert module._v5_ui_cache_expectation_environment(not_dsv4) == {}
    assert module._v5_ui_cache_expectation_environment(dsv4) == {
        "VMLINUX_REAL_UI_EXPECT_DSV4_POOL_QUANT": "1"
    }


def test_r20_v5_owned_producer_rejects_wrong_nonce_and_stale_capture(
    tmp_path: Path,
):
    module = load_module()
    python = str(Path(sys.executable).resolve())
    run_context = {
        "run_id": "nonce-bound",
        "nonce": "1" * 32,
    }
    run_dir = tmp_path / "owned"
    run_dir.mkdir(mode=0o700)
    wrong_nonce_spec = {
        "argv": [
            python,
            str(Path(__file__).resolve()),
            "--v5-fixture-producer",
            "ui",
            "--output-fd",
            "{OUTPUT_FD}",
            "--run-id",
            "{RUN_ID}",
            "--nonce",
            "2" * 32,
        ],
        "cwd": ROOT,
        "env": {},
    }
    with pytest.raises(RuntimeError, match="envelope binding mismatch"):
        module._v5_run_owned_child(
            "ui",
            wrong_nonce_spec,
            run_context,
            run_dir,
        )

    stale_dir = tmp_path / "stale"
    stale_dir.mkdir(mode=0o700)
    (stale_dir / "ui.producer.json").write_text(
        "pre-authored",
        encoding="utf-8",
    )
    with pytest.raises(FileExistsError):
        module._v5_run_owned_child(
            "ui",
            wrong_nonce_spec,
            run_context,
            stale_dir,
        )


def _fixture_orchestration(
    module,
    tmp_path: Path,
    *,
    ui_mode: str = "normal",
    api_mode: str = "normal",
    cache_mode: str = "normal",
):
    bundle, _ = bundle_attestation(module, tmp_path / "model")
    native_bundle, _ = native_bundle_attestation(
        module,
        tmp_path / "native-model",
    )
    source = source_payload()
    run_context = {"run_id": "held-fixture", "nonce": "3" * 32}
    run_dir = tmp_path / "held-run"
    run_dir.mkdir(mode=0o700)
    event_log = tmp_path / "events.log"
    python = str(Path(sys.executable).resolve())
    modes = {"ui": ui_mode, "api": api_mode, "cache": cache_mode}
    plans = {
        producer: {
            "argv": [
                python,
                str(Path(__file__).resolve()),
                "--v5-fixture-producer",
                producer,
                "--output-fd",
                "{OUTPUT_FD}",
                "--run-id",
                "{RUN_ID}",
                "--nonce",
                "{NONCE}",
                "--v5-session-binding-path",
                "{SESSION_BINDING_PATH}",
                "--v5-ready-path",
                "{READY_PATH}",
                "--v5-release-path",
                "{RELEASE_PATH}",
                "--v5-phase-control-dir",
                "{PHASE_CONTROL_DIR}",
                "--v5-paired-api-path",
                "{PAIRED_API_PATH}",
                "--v5-run-intent-path",
                "{RUN_INTENT_PATH}",
                "--v5-run-intent-sha256",
                "{RUN_INTENT_SHA256}",
                "--v5-active-phase-index",
                "{ACTIVE_PHASE_INDEX}",
                "--v5-ui-session-attestation-path",
                "{UI_SESSION_ATTESTATION_PATH}",
                "--v5-previous-backend-pid",
                "{PREVIOUS_BACKEND_PID}",
                "--v5-reuse-session-id",
                "{REUSE_SESSION_ID}",
                "--v5-reuse-session-attestation-path",
                "{REUSE_SESSION_ATTESTATION_PATH}",
                "--v5-source-commit",
                "{SOURCE_COMMIT}",
                "--v5-source-tree",
                "{SOURCE_TREE}",
                "--bundle-root",
                str(tmp_path / "model"),
                "--bundle-fingerprint",
                bundle["fingerprint_sha256"],
                "--native-bundle-root",
                str(tmp_path / "native-model"),
                "--native-bundle-fingerprint",
                native_bundle["fingerprint_sha256"],
                "--model",
                "fixture-model",
                "--native-model",
                "fixture-native-model",
                "--direct-base-url",
                "http://127.0.0.1:8001",
                "--native-direct-base-url",
                "http://127.0.0.1:8002",
                "--gateway-base-url",
                "http://127.0.0.1:8080",
                "--health-url",
                "http://127.0.0.1:8001/health",
                "--native-health-url",
                "http://127.0.0.1:8002/health",
                "--gateway-health-url",
                "http://127.0.0.1:8080/health",
                "--cdp-url",
                "http://127.0.0.1:9335",
                "--backend-pid",
                "9001",
                "--gateway-pid",
                "9002",
                "--electron-pid",
                "9003",
                "--fixture-mode",
                modes[producer],
            ],
            "cwd": ROOT,
            "env": {"VMLINUX_FIXTURE_EVENT_LOG": str(event_log)},
            "ready_timeout_seconds": 0.5,
            "producer_timeout_seconds": 5,
        }
        for producer in module.V5_PRODUCER_NAMES
    }
    common_expected = {
        "source_commit": source["source_commit"],
        "source_tree": source["source_tree"],
        "gateway_base_url": "http://127.0.0.1:8080",
        "gateway_health_url": "http://127.0.0.1:8080/health",
        "cdp_url": "http://127.0.0.1:9335",
        "electron_pid": 9003,
        "gateway_pid": 9002,
    }
    expected = {
        module.V5_PRIMARY_REPRESENTATIVE_ID: {
            **common_expected,
            "direct_base_url": "http://127.0.0.1:8001",
            "health_url": "http://127.0.0.1:8001/health",
            "direct_health_url": "http://127.0.0.1:8001/health",
            "model": "fixture-model",
            "model_bundle_path": bundle["model_bundle_path"],
            "bundle_fingerprint_sha256": bundle["fingerprint_sha256"],
            "native_cache_policy": bundle["derived"]["native_cache"],
        },
        module.V5_NATIVE_REPRESENTATIVE_ID: {
            **common_expected,
            "direct_base_url": "http://127.0.0.1:8002",
            "health_url": "http://127.0.0.1:8002/health",
            "direct_health_url": "http://127.0.0.1:8002/health",
            "model": "fixture-native-model",
            "model_bundle_path": native_bundle["model_bundle_path"],
            "bundle_fingerprint_sha256": native_bundle["fingerprint_sha256"],
            "native_cache_policy": native_bundle["derived"]["native_cache"],
        },
    }
    return (
        plans,
        run_context,
        run_dir,
        expected,
        event_log,
    )


def test_r20_v5_concurrent_producers_are_held_until_both_children_finish(
    tmp_path: Path,
    monkeypatch,
):
    module = load_module()
    plans, context, run_dir, expected, event_log = _fixture_orchestration(
        module,
        tmp_path,
    )
    private_token = "private_" + ("z" * 48)
    observed_token_modes = []
    monkeypatch.setattr(
        module.secrets,
        "token_urlsafe",
        lambda _bytes: private_token,
    )

    def observe_hold(binding):
        token_path = run_dir / "private-cache-attestation.token"
        observed_token_modes.append(token_path.stat().st_mode & 0o777)
        return {
            "session_id": binding["session_id"],
            "observed_while_held": True,
        }

    result = module._v5_execute_producers(
        plans,
        context,
        run_dir,
        expected_binding=expected,
        hold_observer=observe_hold,
    )
    assert set(result) == set(module.V5_PRODUCER_NAMES)
    assert observed_token_modes == [0o600] * len(module.V5_CACHE_PHASES)
    assert not (run_dir / "private-cache-attestation.token").exists()
    assert private_token not in json.dumps(result, sort_keys=True)
    events = event_log.read_text(encoding="utf-8").splitlines()
    for phase in module.V5_CACHE_PHASES:
        ready = events.index(f"ui_ready:{phase['name']}")
        api_started = events.index(f"api_capture_start:{phase['name']}")
        api_completed = events.index(
            f"api_capture_complete:{phase['name']}"
        )
        cache_completed = events.index(
            f"cache_phase_complete:{phase['name']}"
        )
        released = events.index(f"ui_release_seen:{phase['name']}")
        ui_completed = events.index(
            f"ui_capture_complete:{phase['name']}"
        )
        assert ready < api_started < api_completed
        assert ready < cache_completed < released < ui_completed
    final_cache_phase = events.index("cache_phase_complete:native_exception")
    assert final_cache_phase < events.index("cache_capture_complete")
    assert result["ui"]["capture"]["phase_count"] == 6
    assert result["api"]["capture"]["phase_count"] == 6
    held = [row["binding"] for row in result["ui"]["hold_phases"]]
    assert {row["direct_base_url"] for row in held[:5]} == {
        "http://127.0.0.1:8001"
    }
    assert held[5]["direct_base_url"] == "http://127.0.0.1:8002"
    assert held[5]["session_id"] != held[0]["session_id"]
    assert {row["gateway_base_url"] for row in held} == {
        "http://127.0.0.1:8080"
    }
    assert {row["cdp_url"] for row in held} == {
        "http://127.0.0.1:9335"
    }
    assert {row["source_commit"] for row in held} == {
        source_payload()["source_commit"]
    }


def test_r20_v5_cache_worker_uses_prior_backend_pid_for_restart_phases(
    tmp_path: Path,
    monkeypatch,
):
    module = load_module()
    run_id = "restart-pid-chain"
    nonce = "n" * 32
    source_commit = "c" * 40
    source_tree = "d" * 40
    run_root = tmp_path / "run"
    control_dir = run_root / "control"
    artifact_root = run_root / "cache-artifacts"
    run_root.mkdir()
    control_dir.mkdir()
    primary_root = tmp_path / "primary-model"
    native_root = tmp_path / "native-model"
    primary_root.mkdir()
    native_root.mkdir()
    primary_fingerprint = "a" * 64
    native_fingerprint = "b" * 64
    primary_session = "primary-session"
    native_session = "native-session"
    previous_backend_pid = None
    backend_pids: list[int] = []
    for phase in module.V5_CACHE_PHASES:
        is_native = (
            phase["representative_id"] == module.V5_NATIVE_REPRESENTATIVE_ID
        )
        backend_pid = 2000 + phase["index"]
        backend_pids.append(backend_pid)
        session_id = native_session if is_native else primary_session
        model = "fixture-native-model" if is_native else "fixture-model"
        bundle_root = native_root if is_native else primary_root
        fingerprint = native_fingerprint if is_native else primary_fingerprint
        paths = module._v5_existing_phase_paths(control_dir, phase)
        direct_base_url = (
            "http://127.0.0.1:8023"
            if is_native
            else "http://127.0.0.1:8022"
        )
        health_url = f"{direct_base_url}/health"
        binding = {
            "schema": module.V5_SESSION_BINDING_SCHEMA,
            "run_id": run_id,
            "nonce": nonce,
            "ui_producer_pid": 999,
            "source_commit": source_commit,
            "source_tree": source_tree,
            "model": model,
            "model_bundle_path": str(bundle_root.resolve()),
            "bundle_fingerprint_sha256": fingerprint,
            "session_id": session_id,
            "direct_base_url": direct_base_url,
            "gateway_base_url": "http://127.0.0.1:8080",
            "health_url": health_url,
            "direct_health_url": health_url,
            "gateway_health_url": "http://127.0.0.1:8080/health",
            "cdp_url": "http://127.0.0.1:9355",
            "backend_pid": backend_pid,
            "previous_backend_pid": previous_backend_pid,
            "session_start_ordinal": phase["index"] + 1,
            "gateway_pid": 75096,
            "electron_pid": 75096,
            "phase_index": phase["index"],
            "phase_name": phase["name"],
            "representative_id": phase["representative_id"],
            "bundle_role": phase["bundle_role"],
            "cache_policy": phase["cache_policy"],
            "kv_cache_quantization": phase["kv_cache_quantization"],
            "tq_policy": phase["tq_policy"],
            "session_policy": phase["session_policy"],
            "ui_action_profile": phase["ui_action_profile"],
            "ui_turn_count": phase["ui_turn_count"],
            "api_action_profile": phase["api_action_profile"],
            "paged_ram": phase["paged_ram"],
        }
        binding_bytes = json.dumps(
            binding,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        paths["binding"].write_bytes(binding_bytes)
        ready = {
            "schema": module.V5_UI_READY_SCHEMA,
            "run_id": run_id,
            "nonce": nonce,
            "ui_producer_pid": 999,
            "session_id": session_id,
            "binding_sha256": hashlib.sha256(binding_bytes).hexdigest(),
            "held": True,
            "phase_index": phase["index"],
            "phase_name": phase["name"],
            "representative_id": phase["representative_id"],
            "bundle_role": phase["bundle_role"],
            "cache_policy": phase["cache_policy"],
            "kv_cache_quantization": phase["kv_cache_quantization"],
            "tq_policy": phase["tq_policy"],
            "session_policy": phase["session_policy"],
            "ui_action_profile": phase["ui_action_profile"],
            "ui_turn_count": phase["ui_turn_count"],
            "api_action_profile": phase["api_action_profile"],
            "paged_ram": phase["paged_ram"],
            "ready_at": module._iso_now(),
        }
        paths["ready"].write_text(
            json.dumps(ready, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        previous_backend_pid = backend_pid

    observed_previous_backend_pids: list[int] = []

    def fake_cache_phase(
        phase_args,
        binding,
        _run_root,
        phase,
        *,
        store_summary_path,
    ):
        observed_previous_backend_pids.append(
            phase_args.v5_previous_backend_pid
        )
        if module._v5_cache_gate_operation(phase) == "probe":
            assert store_summary_path is not None
        summary_bytes = json.dumps(
            {
                "phase": phase["index"],
                "identity": {
                    "observed_engine": {"pid": binding["backend_pid"]},
                    "model_bundle_provenance": {
                        "fingerprint_sha256": binding[
                            "bundle_fingerprint_sha256"
                        ],
                    },
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        phase_root = artifact_root / f"{phase['index']:02d}-{phase['name']}"
        phase_root.mkdir(parents=True, exist_ok=True)
        summary_path = phase_root / "summary.json"
        summary_path.write_bytes(summary_bytes)
        return (
            {
                "phase_index": phase["index"],
                "summary_b64": _fixture_b64(summary_bytes),
            },
            summary_path,
            summary_bytes,
        )

    monkeypatch.setattr(module, "_observe_process", lambda _pid: {"pid": 999})
    monkeypatch.setattr(module, "_v5_cache_worker_phase", fake_cache_phase)
    monkeypatch.setattr(
        module,
        "_v5_wait_for_phase_release",
        lambda *_args, **_kwargs: b"release",
    )
    monkeypatch.setattr(
        module,
        "_v5_derive_l2_size_eviction_attestation",
        lambda **_kwargs: {"schema": "fixture-l2-attestation"},
    )
    args = argparse.Namespace(
        v5_cache_artifact_root=artifact_root,
        v5_phase_control_dir=control_dir,
        v5_previous_backend_pid=0,
        v5_run_id=run_id,
        v5_nonce=nonce,
        v5_source_commit=source_commit,
        v5_source_tree=source_tree,
        v5_run_intent_sha256="e" * 64,
        v5_session_binding_path=control_dir / "unused.session.json",
        v5_ready_path=control_dir / "unused.ready.json",
        v5_release_path=control_dir / "unused.release.json",
        direct_base_url="http://127.0.0.1:8022",
        native_direct_base_url="http://127.0.0.1:8023",
        gateway_base_url="http://127.0.0.1:8080",
        health_url="http://127.0.0.1:8022/health",
        native_health_url="http://127.0.0.1:8023/health",
        gateway_health_url="http://127.0.0.1:8080/health",
        cdp_url="http://127.0.0.1:9355",
        electron_pid=75096,
        gateway_pid=75096,
        model="fixture-model",
        native_model="fixture-native-model",
    )
    bundles = {
        module.V5_PRIMARY_REPRESENTATIVE_ID: {
            "model_bundle_path": str(primary_root.resolve()),
            "fingerprint_sha256": primary_fingerprint,
        },
        module.V5_NATIVE_REPRESENTATIVE_ID: {
            "model_bundle_path": str(native_root.resolve()),
            "fingerprint_sha256": native_fingerprint,
        },
    }

    module._v5_cache_worker_capture(args, bundles, run_root)

    assert observed_previous_backend_pids == [
        0,
        backend_pids[0],
        backend_pids[1],
        backend_pids[2],
        backend_pids[3],
        backend_pids[4],
    ]


def test_r20_v5_phase2_extends_only_the_long_tq_durability_window():
    module = load_module()
    phase2 = next(
        phase
        for phase in module.V5_CACHE_PHASES
        if phase["index"] == 2
    )

    assert module._v5_cache_gate_extra_args(phase2) == (
        "--durability-timeout",
        "300",
    )
    assert all(
        module._v5_cache_gate_extra_args(phase) == ()
        for phase in module.V5_CACHE_PHASES
        if phase["index"] != 2
    )


@pytest.mark.parametrize(
    ("ui_mode", "message"),
    [
        ("early_exit", "UI producer exited"),
        ("no_ready", "UI producer timed out"),
        ("stale_ready", "UI hold did not become valid"),
        ("mismatched_session", "UI hold did not become valid"),
        (
            "same_backend_pid",
            "UI hold did not become valid|ui owned producer failed",
        ),
        ("phase_mismatch", "UI hold did not become valid"),
        ("stale_release", "UI"),
    ],
)
def test_r20_v5_concurrent_orchestration_rejects_invalid_ui_lifecycle(
    tmp_path: Path,
    ui_mode: str,
    message: str,
):
    module = load_module()
    plans, context, run_dir, expected, _ = _fixture_orchestration(
        module,
        tmp_path,
        ui_mode=ui_mode,
    )
    with pytest.raises(RuntimeError, match=message):
        module._v5_execute_producers(
            plans,
            context,
            run_dir,
            expected_binding=expected,
            hold_observer=lambda binding: {"session_id": binding["session_id"]},
        )


def test_r20_v5_concurrent_orchestration_rejects_mismatched_child_session(
    tmp_path: Path,
):
    module = load_module()
    plans, context, run_dir, expected, _ = _fixture_orchestration(
        module,
        tmp_path,
        api_mode="mismatched_envelope",
    )
    with pytest.raises(RuntimeError, match="not bound to the held UI session"):
        module._v5_execute_producers(
            plans,
            context,
            run_dir,
            expected_binding=expected,
            hold_observer=lambda binding: {"session_id": binding["session_id"]},
        )


def test_r20_v5_production_worker_cli_rejects_stale_coordination_path(
    tmp_path: Path,
):
    run_dir = tmp_path / "private-run"
    run_dir.mkdir(mode=0o700)
    (run_dir / "ui.phase-00.session.json").write_text(
        "stale",
        encoding="utf-8",
    )
    output_path = run_dir / "direct-worker-output.json"
    output_fd = os.open(
        output_path,
        os.O_RDWR | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    command = [
        str(Path(sys.executable).resolve()),
        str(SCRIPT),
        "--v5-owned-worker",
        "ui",
        "--v5-output-fd",
        str(output_fd),
        "--v5-run-id",
        "direct-worker",
        "--v5-nonce",
        "4" * 32,
        "--v5-session-binding-path",
        str(run_dir / "ui.phase-00.session.json"),
        "--v5-ready-path",
        str(run_dir / "ui.phase-00.ready.json"),
        "--v5-release-path",
        str(run_dir / "ui.phase-00.release.json"),
        "--v5-phase-control-dir",
        str(run_dir),
        "--v5-paired-api-path",
        str(run_dir / "api.phase-00.matrix.json"),
        "--v5-cache-artifact-root",
        str(run_dir / "cache-live-artifacts"),
        "--v5-source-commit",
        "a" * 40,
        "--v5-source-tree",
        "b" * 40,
        "--v5-run-intent-path",
        str(run_dir / "run-intent.json"),
        "--v5-run-intent-sha256",
        "c" * 64,
        "--v5-active-phase-index",
        "0",
        "--v5-ui-session-attestation-path",
        str(run_dir / "ui.phase-00.attestation.json"),
        "--v5-previous-backend-pid",
        "0",
        "--v5-reuse-session-id",
        "",
        "--v5-reuse-session-attestation-path",
        "",
        "--bundle-root",
        str(tmp_path / "model"),
        "--native-bundle-root",
        str(tmp_path / "native-model"),
        "--direct-base-url",
        "http://127.0.0.1:8001",
        "--gateway-base-url",
        "http://127.0.0.1:8080",
        "--health-url",
        "http://127.0.0.1:8001/health",
        "--gateway-health-url",
        "http://127.0.0.1:8080/health",
        "--cdp-url",
        "http://127.0.0.1:9335",
        "--gateway-pid",
        "9002",
        "--electron-pid",
        "9003",
        "--model",
        "fixture-model",
        "--native-model",
        "fixture-native-model",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env={
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                "TMPDIR": str(run_dir),
            },
            pass_fds=(output_fd,),
            capture_output=True,
            timeout=10,
            check=False,
        )
    finally:
        os.close(output_fd)
    assert completed.returncode != 0
    assert output_path.read_bytes() == b""


def test_r20_v5_ui_adapter_translates_real_harness_terminal_and_tools():
    module = load_module()
    run_id = "ui-adapter"
    nonce = "5" * 32
    assistant_ids = ["m1", "m2", "m3"]
    traces = []
    calls = []
    results = []
    for index, message_id in enumerate(assistant_ids, start=1):
        events = [
            {
                "sequence": 1,
                "event": "stream",
                "channel": "reasoning",
                "messageId": message_id,
                "delta": f"reason-{index}",
                "payload": {},
            }
        ]
        call_rows = []
        result_rows = []
        if index < 3:
            call_id = f"call-{index}"
            name = "file_info" if index == 1 else "run_command"
            arguments = (
                {"path": "panel/package.json"}
                if index == 1
                else {"command": "pwd"}
            )
            call_rows.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }
            )
            result_rows.append(
                {"tool_call_id": call_id, "content": f"result-{index}"}
            )
            events.extend(
                [
                    {
                        "sequence": 2,
                        "event": "tool",
                        "channel": "tool",
                        "messageId": message_id,
                        "payload": {
                            "phase": "calling",
                            "toolCallId": call_id,
                            "toolName": name,
                            "detail": "",
                        },
                    },
                    {
                        "sequence": 3,
                        "event": "tool",
                        "channel": "tool",
                        "messageId": message_id,
                        "payload": {
                            "phase": "result",
                            "toolCallId": call_id,
                            "toolName": name,
                            "detail": f"result-{index}",
                        },
                    },
                ]
            )
        events.extend(
            [
                {
                    "sequence": 4,
                    "event": "stream",
                    "channel": "content",
                    "messageId": message_id,
                    "delta": f"answer-{index}",
                    "payload": {},
                },
                {
                    "sequence": 5,
                    "event": "terminal",
                    "channel": "terminal",
                    "messageId": message_id,
                    "payload": {
                        "responseId": f"resp-{index}",
                        "finishReason": "stop",
                        "metrics": {
                            "ttft": "0.25",
                            "tokensPerSecond": "50.0",
                        },
                    },
                },
            ]
        )
        traces.append({"messageId": message_id, "events": events})
        calls.append(call_rows)
        results.append(result_rows)
    proof = {
        "format": module.ELECTRON_PROOF_SCHEMA,
        "run_id": run_id,
        "session": {"id": "session-1"},
        "uiStartControl": {
            "clicked": True,
            "label": "Start",
            "sessionStatusBefore": "stopped",
            "sessionStatusAfter": "running",
        },
        "requestContract": {
            "promptOne": "one",
            "promptTwo": "two",
            "promptThree": "three",
        },
        "assistantMessageIds": assistant_ids,
        "messageEventTrace": traces,
        "persistedOaiCallsByMessage": calls,
        "persistedOaiResultsByMessage": results,
    }
    proof_bytes = json.dumps(
        proof,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    capture_bytes = module._v5_ui_normalized_capture(
        argparse.Namespace(
            v5_run_id=run_id,
            v5_nonce=nonce,
            v5_active_phase_index=5,
        ),
        proof,
        proof_bytes,
        "session-1",
    )
    capture = json.loads(capture_bytes)
    rows, _ = module._v5_jsonl_bytes(capture["turns"][0]["events_b64"])
    assert rows is not None
    tool_call = next(row for row in rows if row["type"] == "tool_call")
    tool_result = next(row for row in rows if row["type"] == "tool_result")
    terminal = next(row for row in rows if row["type"] == "terminal")
    assert tool_call["arguments"] == {"path": "panel/package.json"}
    assert tool_result["content"] == "result-1"
    assert terminal["ttft_ms"] == 250.0
    assert terminal["decode_tps"] == 50.0
    assert terminal["response_id"] == "resp-1"

    unsuccessful_proof = json.loads(json.dumps(proof))
    unsuccessful_proof["messageEventTrace"][0]["events"][-1]["payload"][
        "finishReason"
    ] = None
    unsuccessful_bytes = json.dumps(
        unsuccessful_proof,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    with pytest.raises(
        RuntimeError,
        match=r"source-owned UI turn terminal was unsuccessful "
        r"\(finishReason=missing\)",
    ):
        module._v5_ui_normalized_capture(
            argparse.Namespace(
                v5_run_id=run_id,
                v5_nonce=nonce,
                v5_active_phase_index=5,
            ),
            unsuccessful_proof,
            unsuccessful_bytes,
            "session-1",
        )

    missing_timing_proof = json.loads(json.dumps(proof))
    missing_timing_proof["messageEventTrace"][0]["events"][-1]["payload"][
        "metrics"
    ] = {}
    missing_timing_bytes = json.dumps(
        missing_timing_proof,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    with pytest.raises(
        RuntimeError,
        match="source-owned UI terminal has no raw timing",
    ):
        module._v5_ui_normalized_capture(
            argparse.Namespace(
                v5_run_id=run_id,
                v5_nonce=nonce,
                v5_active_phase_index=5,
            ),
            missing_timing_proof,
            missing_timing_bytes,
            "session-1",
        )


def test_r20_v5_pins_reject_links_and_bundle_identity_comes_from_bundle(
    tmp_path: Path,
):
    module = load_module()
    executable = tmp_path / "owned-python"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o700)
    pin = module._v5_pin_regular_file(executable, executable=True)
    assert module._v5_pin_unchanged(pin, executable=True)
    alias = tmp_path / "python-alias"
    alias.symlink_to(executable)
    with pytest.raises(ValueError):
        module._v5_pin_regular_file(alias, executable=True)

    model = tmp_path / "mxfp-model"
    bundle_attestation(module, model)
    (model / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": "bundle-owned-name",
                "model_type": "gemma4",
                "architectures": ["Gemma4ForCausalLM"],
                "quantization": "MXFP8",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (model / "jang_config.json").write_text("{}\n", encoding="utf-8")
    snapshot = module._read_bundle_directory_snapshot(model.resolve())
    assert snapshot is not None
    assert snapshot["derived"]["quantization_kind"] == "mxfp"
    contract = module._bundle_family_contract(
        snapshot,
        {
            "model_name": "health-lie",
            "model_type": "wrong",
            "quantization_kind": "jangtq",
        },
    )
    assert contract["model_name"] == "bundle-owned-name"
    assert contract["model_type"] == "gemma4"
    assert contract["quantization"]["weight_format"] == "mxfp"


def test_r20_v5_git_drift_invalidates_source_and_scope_facts(tmp_path: Path):
    module = load_module()
    bundle, _ = bundle_attestation(module, tmp_path / "model")
    before = {
        "commit": "a" * 40,
        "tree": "b" * 40,
        "status_porcelain": "",
        "upstream_commit": "a" * 40,
        "remote_main_commit": "a" * 40,
        "main_only": 0,
        "branch_only": 0,
        "remote_identity": "jjang-ai/vmlx",
        "release_diff_sha256": "c" * 64,
        "release_diff_bytes": b"M\tpanel/src/main/index.ts\n",
    }
    after = deepcopy(before)
    after["commit"] = "d" * 40
    source_facts, scope_facts = module._v5_source_and_scope_facts(
        before,
        after,
        {},
        bundle,
    )
    assert source_facts == set()
    assert scope_facts == set()


def _v5_exact_release_diff(module) -> bytes:
    return b"".join(
        f"M\t{path}\n".encode()
        for path in sorted(module.V5_RELEASE_INTENDED_PATH_OWNER)
    )


def test_r20_v5_release_scope_review_requires_the_exact_checkpoint_inventory():
    module = load_module()
    diff_bytes = _v5_exact_release_diff(module)
    review = module._v5_release_scope_review(diff_bytes)
    assert review is not None
    assert review["diff_sha256"] == hashlib.sha256(diff_bytes).hexdigest()
    assert review["schema"] == "vmlx-r20-release-scope-review-v2"
    assert review["path_count"] == len(module.V5_RELEASE_INTENDED_PATH_OWNER)
    assert review["expected_path_count"] == len(
        module.V5_RELEASE_INTENDED_PATH_OWNER
    )
    assert review["unexpected"] == []
    assert review["missing"] == []
    assert review["public_forbidden"] == []
    assert review["mapped"]["repository_hygiene"] == [
        "scripts/check-public-repo-hygiene.sh"
    ]
    assert review["mapped"]["release_metadata"] == [
        "latest.json",
        "panel/package-lock.json",
        "panel/package.json",
        "pyproject.toml",
        "uv.lock",
    ]
    assert "panel/scripts/bundle-python.sh" in review["mapped"][
        "panel_release_and_proof"
    ]
    assert "panel/scripts/scoped-release-preflight-19.py" in review[
        "mapped"
    ]["panel_release_and_proof"]
    assert "panel/scripts/scoped-release-preflight-20.py" in review[
        "mapped"
    ]["panel_release_and_proof"]


def test_r20_release_version_gate_requires_the_pinned_jang_lock():
    module = load_module()
    failures: list[str] = []
    versions = module.validate_versions(failures)
    assert failures == []
    assert versions["uv_lock_jang"] == module.JANG_VERSION == "2.5.50"


def test_r20_v5_release_scope_review_fails_closed_on_unrelated_panel_source():
    module = load_module()
    diff_bytes = _v5_exact_release_diff(module) + (
        b"M\tpanel/src/unrelated.ts\n"
    )
    review = module._v5_release_scope_review(diff_bytes)
    assert review is not None
    assert review["unexpected"] == ["panel/src/unrelated.ts"]
    assert review["missing"] == []
    assert review["public_forbidden"] == []


@pytest.mark.parametrize(
    "private_path",
    [
        ".agents/private-ledger.md",
        ".agent/private-ledger.md",
        ".claude/private-ledger.md",
        ".codex/private-ledger.md",
        ".sisyphus/private-ledger.md",
        ".factory/private-ledger.md",
        "docs/internal/private-ledger.md",
        "notes/private-ledger.md",
        "botes/private-ledger.md",
        "evidence/live.json",
        "private-evidence/live.json",
        "vmlx-proof/live.json",
        "screenshot/live.png",
        "screenshots/live.png",
        "screen-recording/live.mov",
        "screen-recordings/live.mov",
        "cdp-capture/live.json",
        "cdp-captures/live.json",
        "raw-sse/chat.txt",
        "runtime-log/engine.log",
        "runtime-logs/engine.log",
        "state/session.sqlite",
        "state/session.db",
    ],
)
def test_r20_v5_release_scope_review_rejects_every_private_artifact_category(
    private_path: str,
):
    module = load_module()
    diff_bytes = _v5_exact_release_diff(module) + (
        f"M\t{private_path}\n".encode()
    )
    review = module._v5_release_scope_review(diff_bytes)
    assert review is not None
    assert review["unexpected"] == [private_path]
    assert review["missing"] == []
    assert review["public_forbidden"] == [private_path]


def test_r20_v5_release_scope_review_fails_closed_when_an_intended_path_is_missing():
    module = load_module()
    missing = "panel/src/main/ipc/chat.ts"
    diff_bytes = b"".join(
        f"M\t{path}\n".encode()
        for path in sorted(module.V5_RELEASE_INTENDED_PATH_OWNER)
        if path != missing
    )
    review = module._v5_release_scope_review(diff_bytes)
    assert review is not None
    assert review["unexpected"] == []
    assert review["missing"] == [missing]


def test_r20_v5_scope_facts_require_exact_paths_not_self_classification(
    tmp_path: Path,
):
    module = load_module()
    bundle, _ = bundle_attestation(module, tmp_path / "model")
    diff_bytes = _v5_exact_release_diff(module)
    snapshot = {
        "commit": "a" * 40,
        "tree": "b" * 40,
        "status_porcelain": "",
        "upstream_commit": "a" * 40,
        "remote_main_commit": "a" * 40,
        "main_only": 0,
        "branch_only": 0,
        "remote_identity": "jjang-ai/vmlx",
        "release_diff_sha256": hashlib.sha256(diff_bytes).hexdigest(),
        "release_diff_bytes": diff_bytes,
    }
    _source_facts, exact_scope_facts = module._v5_source_and_scope_facts(
        snapshot,
        deepcopy(snapshot),
        {
            "expected_source_attestation": (
                module.release_runtime_source_attestation()
            )
        },
        bundle,
    )
    assert exact_scope_facts == {
        "v1_6_19_to_head_diff_reviewed",
        "all_intended_fixes_mapped",
        "unintended_changes_none_or_documented",
        "public_repository_hygiene_passed",
    }

    unrelated = diff_bytes + b"M\tpanel/src/unrelated.ts\n"
    drifted = deepcopy(snapshot)
    drifted["release_diff_bytes"] = unrelated
    drifted["release_diff_sha256"] = hashlib.sha256(unrelated).hexdigest()
    _source_facts, unrelated_scope_facts = module._v5_source_and_scope_facts(
        drifted,
        deepcopy(drifted),
        {
            "expected_source_attestation": (
                module.release_runtime_source_attestation()
            )
        },
        bundle,
    )
    assert unrelated_scope_facts == {"v1_6_19_to_head_diff_reviewed"}


def test_r20_v5_release_scope_review_rejects_malformed_name_status_bytes():
    module = load_module()
    assert module._v5_release_scope_review(b"") is None
    assert module._v5_release_scope_review(b"M\n") is None
    assert module._v5_release_scope_review(b"R100\tone-path-only\n") is None
    assert module._v5_release_scope_review(b"M\tbad\xffpath\n") is None


def test_r20_v5_cli_has_no_author_pass_attestation_option(tmp_path: Path):
    module = load_module()
    argv = [
        "--private-evidence-root",
        str(tmp_path),
        "--out",
        str(tmp_path / "out.json"),
        "--bundle-root",
        str(tmp_path / "model"),
        "--native-bundle-root",
        str(tmp_path / "native-model"),
        "--model",
        "fixture",
        "--native-model",
        "fixture-native",
        "--direct-base-url",
        "http://127.0.0.1:8001",
        "--gateway-base-url",
        "http://127.0.0.1:8080",
        "--health-url",
        "http://127.0.0.1:8001/health",
        "--gateway-health-url",
        "http://127.0.0.1:8080/health",
        "--cdp-url",
        "http://127.0.0.1:9335",
        "--backend-pid",
        "1",
        "--gateway-pid",
        "2",
        "--electron-pid",
        "3",
        "--jang-source",
        str(tmp_path / "jang"),
        "--author-attestation",
        str(tmp_path / "preauthored.json"),
    ]
    with pytest.raises(SystemExit):
        module._v5_parser().parse_args(argv)


def test_r20_v5_packaging_consumer_revalidates_exact_source_and_jang(
    tmp_path: Path,
    monkeypatch,
):
    module = load_module()
    private_root = tmp_path / "private"
    private_root.mkdir()
    manifest_path = private_root / "release.json"
    output = tmp_path / "build" / "preflight.json"
    versions = {"python": module.VERSION, "panel": module.VERSION}
    source = {
        "commit": "a" * 40,
        "tree": "b" * 40,
        "upstream_commit": "a" * 40,
        "remote_main_commit": "a" * 40,
        "remote_identity": "jjang-ai/vmlx",
        "main_only": 0,
        "branch_only": 0,
        "release_diff_sha256": "c" * 64,
    }
    jang = {
        "version": module.JANG_VERSION,
        "commit": module.JANG_COMMIT,
        "tree": module.JANG_TREE,
        "upstream_commit": module.JANG_COMMIT,
        "remote_main_commit": module.JANG_COMMIT,
        "remote_identity": "jjang-ai/jangq",
    }
    checks = {
        name: {
            "status": "pass",
            "assertions": {
                assertion: True
                for assertion in module.V5_RELEASE_ASSERTIONS[name]
            },
            "evidence_sha256": ["d" * 64],
        }
        for name in module.V5_REQUIRED_CHECKS
    }
    manifest = {
        "schema": module.V5_MANIFEST_SCHEMA,
        "scope": module.SCOPE,
        "version": module.VERSION,
        "status": "pass",
        "failures": [],
        "run": {"run_id": "release-run"},
        "source": source,
        "jang": jang,
        "versions": versions,
        "checks": checks,
        "completion": {"state": "complete"},
    }
    manifest["completion"]["run_digest"] = module._v5_manifest_digest(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(module, "validate_versions", lambda failures: versions)
    monkeypatch.setattr(
        module,
        "validate_private_evidence_root",
        lambda configured, failures: private_root.resolve(),
    )
    monkeypatch.setattr(module, "_v5_git_snapshot", lambda: source)
    monkeypatch.setattr(module, "validate_jang_source", lambda failures: jang)

    assert (
        module._v5_consume_manifest_main(
            [
                "--expected-version",
                module.VERSION,
                "--manifest",
                str(manifest_path),
                "--private-evidence-root",
                str(private_root),
                "--out",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8")) == manifest


def test_r20_v5_ui_attestation_binds_parent_owned_worker_pid():
    module = load_module()
    phase = module.V5_CACHE_PHASES[0]
    binding = {
        "ui_producer_pid": 4101,
        "session_id": "owned-session",
        "model": "owned-model",
        "model_bundle_path": "/private/model",
        "bundle_fingerprint_sha256": "a" * 64,
        "backend_pid": 4201,
        "gateway_pid": 4301,
        "direct_base_url": "http://127.0.0.1:8010",
        "gateway_base_url": "http://127.0.0.1:8081",
        "electron_pid": 4401,
        "cdp_url": "http://127.0.0.1:9345",
        "source_commit": "b" * 40,
        "source_tree": "c" * 40,
        "harness_binding_sha256": "d" * 64,
    }
    run_context = {"run_id": "owned-run", "nonce": "owned-nonce"}
    attestation = {
        "schema": module.V5_UI_SESSION_ATTESTATION_SCHEMA,
        "run_id": run_context["run_id"],
        "nonce": run_context["nonce"],
        "run_intent_sha256": "e" * 64,
        "phase_index": phase["index"],
        "phase_name": phase["name"],
        "representative_id": phase["representative_id"],
        "bundle_role": phase["bundle_role"],
        "cache_policy": phase["cache_policy"],
        "paged_ram": phase["paged_ram"],
        "ui_action_profile": phase["ui_action_profile"],
        "ui_turn_count": phase["ui_turn_count"],
        "api_action_profile": phase["api_action_profile"],
        "ui_producer_pid": binding["ui_producer_pid"],
        "session_id": binding["session_id"],
        "model": binding["model"],
        "model_bundle_path": binding["model_bundle_path"],
        "bundle_fingerprint_sha256": binding["bundle_fingerprint_sha256"],
        "backend_pid": binding["backend_pid"],
        "gateway_pid": binding["gateway_pid"],
        "direct_base_url": binding["direct_base_url"],
        "gateway_base_url": binding["gateway_base_url"],
        "electron_pid": binding["electron_pid"],
        "cdp_origin": binding["cdp_url"],
        "lifecycle_owner": "parent",
        "source_commit": binding["source_commit"],
        "source_tree": binding["source_tree"],
        "renderer_source_sha256": "f" * 64,
        "session_binding_sha256": binding["harness_binding_sha256"],
        "created_at": module._iso_now(),
    }
    module._v5_validate_ui_session_attestation(
        attestation,
        run_context=run_context,
        run_intent_sha256="e" * 64,
        phase=phase,
        binding=binding,
    )

    harness_pid_attestation = {**attestation, "ui_producer_pid": 4102}
    with pytest.raises(RuntimeError, match="attestation binding mismatch"):
        module._v5_validate_ui_session_attestation(
            harness_pid_attestation,
            run_context=run_context,
            run_intent_sha256="e" * 64,
            phase=phase,
            binding=binding,
        )


def _v5_fixture_child(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--v5-fixture-producer", choices=("ui", "api", "cache"))
    group.add_argument(
        "--v5-fixture-command",
        choices=(
            "full_python_suite",
            "full_panel_suite",
            "typecheck",
            "production_build",
            "jang_build",
            "jang_venv",
            "jang_install",
            "jang_import",
            "jang_test",
        ),
    )
    parser.add_argument("--output-fd", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--nonce")
    parser.add_argument("--v5-session-binding-path", type=Path)
    parser.add_argument("--v5-ready-path", type=Path)
    parser.add_argument("--v5-release-path", type=Path)
    parser.add_argument("--v5-phase-control-dir", type=Path)
    parser.add_argument("--v5-paired-api-path", type=Path)
    parser.add_argument("--v5-run-intent-path", type=Path)
    parser.add_argument("--v5-run-intent-sha256")
    parser.add_argument("--v5-active-phase-index", type=int)
    parser.add_argument("--v5-ui-session-attestation-path", type=Path)
    parser.add_argument("--v5-previous-backend-pid", type=int, default=0)
    parser.add_argument("--v5-reuse-session-id", default="")
    parser.add_argument("--v5-reuse-session-attestation-path", default="")
    parser.add_argument("--v5-source-commit")
    parser.add_argument("--v5-source-tree")
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--bundle-fingerprint")
    parser.add_argument("--native-bundle-root", type=Path)
    parser.add_argument("--native-bundle-fingerprint")
    parser.add_argument("--model")
    parser.add_argument("--native-model")
    parser.add_argument("--direct-base-url")
    parser.add_argument("--native-direct-base-url")
    parser.add_argument("--gateway-base-url")
    parser.add_argument("--health-url")
    parser.add_argument("--native-health-url")
    parser.add_argument("--gateway-health-url")
    parser.add_argument("--cdp-url")
    parser.add_argument("--backend-pid", type=int)
    parser.add_argument("--gateway-pid", type=int)
    parser.add_argument("--electron-pid", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--distribution-root", type=Path)
    parser.add_argument("--isolated-venv", type=Path)
    parser.add_argument(
        "--fixture-mode",
        choices=(
            "normal",
            "early_exit",
            "no_ready",
            "stale_ready",
            "stale_release",
            "mismatched_session",
            "mismatched_envelope",
            "same_backend_pid",
            "phase_mismatch",
        ),
        default="normal",
    )
    parser.add_argument("--event-log", type=Path)
    args = parser.parse_args(argv)

    def record(event: str) -> None:
        event_log = args.event_log or (
            Path(os.environ["VMLINUX_FIXTURE_EVENT_LOG"])
            if os.environ.get("VMLINUX_FIXTURE_EVENT_LOG")
            else None
        )
        if event_log is None:
            return
        descriptor = os.open(
            event_log,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            os.write(descriptor, f"{event}\n".encode())
        finally:
            os.close(descriptor)

    time.sleep(0.2)
    if args.v5_fixture_producer:
        assert args.output_fd is not None and args.run_id and args.nonce
        module = load_module()
        coordinated = all(
            (
                args.v5_session_binding_path,
                args.v5_ready_path,
                args.v5_release_path,
                args.v5_phase_control_dir,
            )
        )
        primary_session_id = "fixture-held-session"
        native_session_id = "fixture-native-session"
        session_id = primary_session_id
        phase_releases = None
        cache_phase_bindings = None
        if not coordinated:
            if args.v5_fixture_producer == "ui":
                capture, _ = _fixture_ui_capture(
                    module,
                    args.run_id,
                    args.nonce,
                )
            elif args.v5_fixture_producer == "api":
                capture = _fixture_api_capture(module, args.run_id, args.nonce)
            else:
                capture = _fixture_cache_capture(
                    module,
                    args.run_id,
                    args.nonce,
                )
            binding_bytes = b""
        elif (
            args.v5_fixture_producer == "ui"
            and args.v5_active_phase_index is not None
        ):
            phase = module.V5_CACHE_PHASES[args.v5_active_phase_index]
            is_native = (
                phase["representative_id"]
                == module.V5_NATIVE_REPRESENTATIVE_ID
            )
            session_id = (
                native_session_id if is_native else primary_session_id
            )
            phase_model = args.native_model if is_native else args.model
            phase_bundle_root = (
                args.native_bundle_root if is_native else args.bundle_root
            )
            phase_bundle_fingerprint = (
                args.native_bundle_fingerprint
                if is_native
                else args.bundle_fingerprint
            )
            if args.fixture_mode == "no_ready" and phase["index"] == 0:
                record("ui_no_ready")
                time.sleep(2)
                return 7
            backend_pid = (
                int(args.backend_pid)
                if args.fixture_mode == "same_backend_pid"
                else int(args.backend_pid) + phase["index"]
            )
            binding = {
                "schema": module.V5_SESSION_BINDING_SCHEMA,
                "run_id": args.run_id,
                "nonce": args.nonce,
                "ui_producer_pid": os.getpid(),
                "source_commit": args.v5_source_commit,
                "source_tree": args.v5_source_tree,
                "model": phase_model,
                "model_bundle_path": str(phase_bundle_root.resolve()),
                "bundle_fingerprint_sha256": phase_bundle_fingerprint,
                "session_id": session_id,
                "direct_base_url": (
                    args.native_direct_base_url
                    if is_native
                    else args.direct_base_url
                ),
                "gateway_base_url": args.gateway_base_url,
                "health_url": (
                    args.native_health_url if is_native else args.health_url
                ),
                "direct_health_url": (
                    args.native_health_url if is_native else args.health_url
                ),
                "gateway_health_url": args.gateway_health_url,
                "cdp_url": args.cdp_url,
                "backend_pid": backend_pid,
                "previous_backend_pid": (
                    args.v5_previous_backend_pid or None
                ),
                "session_start_ordinal": phase["index"] + 1,
                "gateway_pid": args.gateway_pid,
                "electron_pid": args.electron_pid,
                "harness_binding_sha256": "d" * 64,
                "phase_index": phase["index"],
                "phase_name": (
                    "wrong-phase"
                    if (
                        args.fixture_mode == "phase_mismatch"
                        and phase["index"] == 1
                    )
                    else phase["name"]
                ),
                "representative_id": phase["representative_id"],
                "bundle_role": phase["bundle_role"],
                "cache_policy": phase["cache_policy"],
                "kv_cache_quantization": phase[
                    "kv_cache_quantization"
                ],
                "tq_policy": phase["tq_policy"],
                "session_policy": phase["session_policy"],
                "ui_action_profile": phase["ui_action_profile"],
                "ui_turn_count": phase["ui_turn_count"],
                "api_action_profile": phase["api_action_profile"],
                "paged_ram": phase["paged_ram"],
            }
            binding_bytes = json.dumps(
                binding,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            args.v5_session_binding_path.write_bytes(binding_bytes)
            args.v5_session_binding_path.chmod(0o600)
            ready = {
                "schema": module.V5_UI_READY_SCHEMA,
                "run_id": args.run_id,
                "nonce": args.nonce,
                "ui_producer_pid": os.getpid(),
                "session_id": (
                    "wrong-ready-session"
                    if (
                        args.fixture_mode == "mismatched_session"
                        and phase["index"] == 0
                    )
                    else session_id
                ),
                "binding_sha256": hashlib.sha256(binding_bytes).hexdigest(),
                "held": True,
                "phase_index": phase["index"],
                "phase_name": phase["name"],
                "representative_id": phase["representative_id"],
                "bundle_role": phase["bundle_role"],
                "cache_policy": phase["cache_policy"],
                "kv_cache_quantization": phase[
                    "kv_cache_quantization"
                ],
                "tq_policy": phase["tq_policy"],
                "session_policy": phase["session_policy"],
                "ui_action_profile": phase["ui_action_profile"],
                "ui_turn_count": phase["ui_turn_count"],
                "api_action_profile": phase["api_action_profile"],
                "paged_ram": phase["paged_ram"],
                "ready_at": (
                    "2020-01-01T00:00:00Z"
                    if (
                        args.fixture_mode == "stale_ready"
                        and phase["index"] == 0
                    )
                    else module._iso_now()
                ),
            }
            args.v5_ready_path.write_text(
                json.dumps(ready, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            args.v5_ready_path.chmod(0o600)
            attestation = {
                "schema": module.V5_UI_SESSION_ATTESTATION_SCHEMA,
                "run_id": args.run_id,
                "nonce": args.nonce,
                "run_intent_sha256": args.v5_run_intent_sha256,
                "phase_index": phase["index"],
                "phase_name": phase["name"],
                "representative_id": phase["representative_id"],
                "bundle_role": phase["bundle_role"],
                "cache_policy": phase["cache_policy"],
                "paged_ram": phase["paged_ram"],
                "ui_action_profile": phase["ui_action_profile"],
                "ui_turn_count": phase["ui_turn_count"],
                "api_action_profile": phase["api_action_profile"],
                "ui_producer_pid": os.getpid(),
                "session_id": session_id,
                "model": phase_model,
                "model_bundle_path": str(phase_bundle_root.resolve()),
                "bundle_fingerprint_sha256": phase_bundle_fingerprint,
                "backend_pid": backend_pid,
                "gateway_pid": args.gateway_pid,
                "direct_base_url": (
                    args.native_direct_base_url
                    if is_native
                    else args.direct_base_url
                ),
                "gateway_base_url": args.gateway_base_url,
                "electron_pid": args.electron_pid,
                "cdp_origin": args.cdp_url,
                "lifecycle_owner": "parent",
                "source_commit": args.v5_source_commit,
                "source_tree": args.v5_source_tree,
                "renderer_source_sha256": "e" * 64,
                "session_binding_sha256": "d" * 64,
                "created_at": module._iso_now(),
            }
            args.v5_ui_session_attestation_path.write_text(
                json.dumps(
                    attestation,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            args.v5_ui_session_attestation_path.chmod(0o600)
            record(f"ui_ready:{phase['name']}")
            if args.fixture_mode == "early_exit" and phase["index"] == 0:
                record("ui_early_exit")
                return 6
            if args.fixture_mode == "stale_release" and phase["index"] == 0:
                args.v5_release_path.write_text(
                    '{"stale":true}',
                    encoding="utf-8",
                )
                args.v5_release_path.chmod(0o600)
                record("ui_stale_release")
            deadline = time.monotonic() + 8
            while not args.v5_release_path.exists():
                if time.monotonic() >= deadline:
                    return 9
                time.sleep(0.01)
            release_bytes = args.v5_release_path.read_bytes()
            release = json.loads(release_bytes)
            assert release["schema"] == module.V5_UI_RELEASE_SCHEMA
            assert release["run_id"] == args.run_id
            assert release["nonce"] == args.nonce
            assert release["session_id"] == session_id
            assert release["phase_index"] == phase["index"]
            record(f"ui_release_seen:{phase['name']}")
            capture, _ = _fixture_ui_capture(
                module,
                args.run_id,
                args.nonce,
                session_id=session_id,
                phase_index=phase["index"],
            )
        elif args.v5_fixture_producer == "ui":
            assert args.v5_session_binding_path
            assert args.v5_ready_path
            assert args.v5_release_path
            assert args.v5_phase_control_dir
            if args.fixture_mode == "no_ready":
                record("ui_no_ready")
                time.sleep(2)
                return 7
            phase_releases = []
            previous_backend_pid = None
            for phase in module.V5_CACHE_PHASES:
                is_native = (
                    phase["representative_id"]
                    == module.V5_NATIVE_REPRESENTATIVE_ID
                )
                phase_session_id = (
                    native_session_id if is_native else primary_session_id
                )
                phase_model = args.native_model if is_native else args.model
                phase_bundle_root = (
                    args.native_bundle_root if is_native else args.bundle_root
                )
                phase_bundle_fingerprint = (
                    args.native_bundle_fingerprint
                    if is_native
                    else args.bundle_fingerprint
                )
                paths = module._v5_existing_phase_paths(
                    args.v5_phase_control_dir,
                    phase,
                )
                backend_pid = (
                    int(args.backend_pid)
                    if args.fixture_mode == "same_backend_pid"
                    else int(args.backend_pid) + phase["index"]
                )
                binding = {
                    "schema": module.V5_SESSION_BINDING_SCHEMA,
                    "run_id": args.run_id,
                    "nonce": args.nonce,
                    "ui_producer_pid": os.getpid(),
                    "source_commit": args.v5_source_commit,
                    "source_tree": args.v5_source_tree,
                    "model": phase_model,
                    "model_bundle_path": str(phase_bundle_root.resolve()),
                    "bundle_fingerprint_sha256": phase_bundle_fingerprint,
                    "session_id": phase_session_id,
                    "direct_base_url": (
                        args.native_direct_base_url
                        if is_native
                        else args.direct_base_url
                    ),
                    "gateway_base_url": args.gateway_base_url,
                    "health_url": (
                        args.native_health_url
                        if is_native
                        else args.health_url
                    ),
                    "direct_health_url": (
                        args.native_health_url
                        if is_native
                        else args.health_url
                    ),
                    "gateway_health_url": args.gateway_health_url,
                    "cdp_url": args.cdp_url,
                    "backend_pid": backend_pid,
                    "previous_backend_pid": previous_backend_pid,
                    "session_start_ordinal": phase["index"] + 1,
                    "gateway_pid": args.gateway_pid,
                    "electron_pid": args.electron_pid,
                    "phase_index": phase["index"],
                    "phase_name": (
                        "wrong-phase"
                        if (
                            args.fixture_mode == "phase_mismatch"
                            and phase["index"] == 1
                        )
                        else phase["name"]
                    ),
                    "representative_id": phase["representative_id"],
                    "bundle_role": phase["bundle_role"],
                    "cache_policy": phase["cache_policy"],
                    "kv_cache_quantization": phase[
                        "kv_cache_quantization"
                    ],
                    "tq_policy": phase["tq_policy"],
                    "session_policy": phase["session_policy"],
                    "ui_action_profile": phase["ui_action_profile"],
                    "ui_turn_count": phase["ui_turn_count"],
                    "api_action_profile": phase["api_action_profile"],
                    "paged_ram": phase["paged_ram"],
                }
                binding_bytes = json.dumps(
                    binding,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                binding_fd = os.open(
                    paths["binding"],
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                try:
                    os.write(binding_fd, binding_bytes)
                    os.fsync(binding_fd)
                finally:
                    os.close(binding_fd)
                ready = {
                    "schema": module.V5_UI_READY_SCHEMA,
                    "run_id": args.run_id,
                    "nonce": args.nonce,
                    "ui_producer_pid": os.getpid(),
                    "session_id": (
                        "wrong-ready-session"
                        if (
                            args.fixture_mode == "mismatched_session"
                            and phase["index"] == 0
                        )
                        else phase_session_id
                    ),
                    "binding_sha256": hashlib.sha256(binding_bytes).hexdigest(),
                    "held": True,
                    "phase_index": phase["index"],
                    "phase_name": phase["name"],
                    "representative_id": phase["representative_id"],
                    "bundle_role": phase["bundle_role"],
                    "cache_policy": phase["cache_policy"],
                    "kv_cache_quantization": phase[
                        "kv_cache_quantization"
                    ],
                    "tq_policy": phase["tq_policy"],
                    "session_policy": phase["session_policy"],
                    "ui_action_profile": phase["ui_action_profile"],
                    "ui_turn_count": phase["ui_turn_count"],
                    "api_action_profile": phase["api_action_profile"],
                    "paged_ram": phase["paged_ram"],
                    "ready_at": (
                        "2020-01-01T00:00:00Z"
                        if (
                            args.fixture_mode == "stale_ready"
                            and phase["index"] == 0
                        )
                        else module._iso_now()
                    ),
                }
                ready_fd = os.open(
                    paths["ready"],
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                try:
                    os.write(
                        ready_fd,
                        json.dumps(
                            ready,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode(),
                    )
                    os.fsync(ready_fd)
                finally:
                    os.close(ready_fd)
                record(f"ui_ready:{phase['name']}")
                if args.fixture_mode == "early_exit" and phase["index"] == 0:
                    record("ui_early_exit")
                    return 6
                if args.fixture_mode == "stale_release" and phase["index"] == 0:
                    release_fd = os.open(
                        paths["release"],
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                    )
                    try:
                        os.write(release_fd, b'{"stale":true}')
                        os.fsync(release_fd)
                    finally:
                        os.close(release_fd)
                    record("ui_stale_release")
                deadline = time.monotonic() + 8
                while not paths["release"].exists():
                    if time.monotonic() >= deadline:
                        return 9
                    time.sleep(0.01)
                release_bytes = paths["release"].read_bytes()
                release = json.loads(release_bytes)
                assert release["schema"] == module.V5_UI_RELEASE_SCHEMA
                assert release["run_id"] == args.run_id
                assert release["nonce"] == args.nonce
                assert release["session_id"] == phase_session_id
                assert release["phase_index"] == phase["index"]
                phase_releases.append(
                    {
                        "phase_index": phase["index"],
                        "phase_name": phase["name"],
                        "representative_id": phase["representative_id"],
                        "cache_policy": phase["cache_policy"],
                        "kv_cache_quantization": phase[
                            "kv_cache_quantization"
                        ],
                        "tq_policy": phase["tq_policy"],
                        "session_policy": phase["session_policy"],
                        "ui_action_profile": phase["ui_action_profile"],
                        "ui_turn_count": phase["ui_turn_count"],
                        "api_action_profile": phase["api_action_profile"],
                        "release_sha256": hashlib.sha256(
                            release_bytes
                        ).hexdigest(),
                    }
                )
                record(f"ui_release_seen:{phase['name']}")
                previous_backend_pid = backend_pid
                session_id = phase_session_id
            capture, _ = _fixture_ui_capture(
                module,
                args.run_id,
                args.nonce,
                session_id=primary_session_id,
            )
        else:
            assert args.v5_session_binding_path
            deadline = time.monotonic() + 8
            while not args.v5_session_binding_path.exists():
                if time.monotonic() >= deadline:
                    return 8
                time.sleep(0.01)
            binding_bytes = args.v5_session_binding_path.read_bytes()
            binding = json.loads(binding_bytes)
            assert binding["run_id"] == args.run_id
            assert binding["nonce"] == args.nonce
            session_id = str(binding["session_id"])
            record(f"{args.v5_fixture_producer}_capture_start")
            if args.v5_fixture_producer == "api":
                phase_index = int(args.v5_active_phase_index or 0)
                phase = module.V5_CACHE_PHASES[phase_index]
                record(f"api_capture_start:{phase['name']}")
                capture = _fixture_api_capture(
                    module,
                    args.run_id,
                    args.nonce,
                    phase_index=phase_index,
                    session_id=binding["session_id"],
                )
                capture["session_binding_sha256"] = hashlib.sha256(
                    binding_bytes
                ).hexdigest()
                assert args.v5_paired_api_path
                args.v5_paired_api_path.write_text(
                    json.dumps(
                        {
                            "schema": "fixture-phase-api-matrix",
                            "run_id": args.run_id,
                            "phase_index": phase_index,
                            "session_id": binding["session_id"],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    encoding="utf-8",
                )
                args.v5_paired_api_path.chmod(0o600)
            else:
                assert args.v5_phase_control_dir
                capture = _fixture_cache_capture(
                    module,
                    args.run_id,
                    args.nonce,
                    bundle_fingerprint=args.bundle_fingerprint,
                    native_bundle_fingerprint=args.native_bundle_fingerprint,
                    model=args.model,
                    native_model=args.native_model,
                    session_id=primary_session_id,
                    native_session_id=native_session_id,
                )
                cache_phase_bindings = []
                for phase, scenario in zip(
                    module.V5_CACHE_PHASES,
                    capture["phases"],
                    strict=True,
                ):
                    gate_operation = module._v5_cache_gate_operation(phase)
                    paths = module._v5_existing_phase_paths(
                        args.v5_phase_control_dir,
                        phase,
                    )
                    phase_deadline = time.monotonic() + 8
                    while not (
                        paths["binding"].exists() and paths["ready"].exists()
                    ):
                        if time.monotonic() >= phase_deadline:
                            return 8
                        time.sleep(0.01)
                    phase_binding_bytes = paths["binding"].read_bytes()
                    phase_binding = json.loads(phase_binding_bytes)
                    binding_digest = hashlib.sha256(
                        phase_binding_bytes
                    ).hexdigest()
                    scenario["backend_pid"] = phase_binding["backend_pid"]
                    scenario["session_binding_sha256"] = binding_digest
                    summary = json.loads(
                        base64.b64decode(scenario["summary_b64"])
                    )
                    summary["identity"]["observed_engine"]["pid"] = phase_binding[
                        "backend_pid"
                    ]
                    summary_bytes = json.dumps(
                        summary,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                    scenario["summary_b64"] = _fixture_b64(summary_bytes)
                    manifest = [
                        {
                            "relative_path": "summary.json",
                            "sha256": hashlib.sha256(summary_bytes).hexdigest(),
                            "size": len(summary_bytes),
                        }
                    ]
                    scenario["artifact_manifest_b64"] = _fixture_json_b64(
                        manifest
                    )
                    if gate_operation == "store":
                        store_sha = hashlib.sha256(summary_bytes).hexdigest()
                    else:
                        scenario["linked_store_summary_sha256"] = store_sha
                    done = {
                        "schema": module.V5_CACHE_PHASE_DONE_SCHEMA,
                        "run_id": args.run_id,
                        "nonce": args.nonce,
                        "phase_index": phase["index"],
                        "phase_name": phase["name"],
                        "representative_id": phase["representative_id"],
                        "bundle_role": phase["bundle_role"],
                        "cache_policy": phase["cache_policy"],
                        "kv_cache_quantization": phase[
                            "kv_cache_quantization"
                        ],
                        "tq_policy": phase["tq_policy"],
                        "session_policy": phase["session_policy"],
                        "operation": phase["operation"],
                        "ui_action_profile": phase["ui_action_profile"],
                        "ui_turn_count": phase["ui_turn_count"],
                        "api_action_profile": phase["api_action_profile"],
                        "paged_ram": phase["paged_ram"],
                        "session_id": phase_binding["session_id"],
                        "backend_pid": phase_binding["backend_pid"],
                        "session_binding_sha256": binding_digest,
                        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
                        "completed_at": module._iso_now(),
                    }
                    done_bytes = json.dumps(
                        done,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                    done_fd = os.open(
                        paths["cache_done"],
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                    )
                    try:
                        os.write(done_fd, done_bytes)
                        os.fsync(done_fd)
                    finally:
                        os.close(done_fd)
                    cache_phase_bindings.append(
                        {
                            "phase_index": phase["index"],
                            "phase_name": phase["name"],
                            "representative_id": phase["representative_id"],
                            "cache_policy": phase["cache_policy"],
                            "kv_cache_quantization": phase[
                                "kv_cache_quantization"
                            ],
                            "tq_policy": phase["tq_policy"],
                            "session_policy": phase["session_policy"],
                            "ui_action_profile": phase["ui_action_profile"],
                            "ui_turn_count": phase["ui_turn_count"],
                            "api_action_profile": phase["api_action_profile"],
                            "session_id": phase_binding["session_id"],
                            "model": phase_binding["model"],
                            "bundle_fingerprint_sha256": phase_binding[
                                "bundle_fingerprint_sha256"
                            ],
                            "backend_pid": phase_binding["backend_pid"],
                            "session_binding_sha256": binding_digest,
                        }
                    )
                    record(f"cache_phase_complete:{phase['name']}")
                    phase_deadline = time.monotonic() + 8
                    while not paths["release"].exists():
                        if time.monotonic() >= phase_deadline:
                            return 9
                        time.sleep(0.01)
                phase2_summary_bytes = base64.b64decode(
                    capture["phases"][2]["summary_b64"]
                )
                phase3_summary_bytes = base64.b64decode(
                    capture["phases"][3]["summary_b64"]
                )
                capture["l2_size_eviction_attestation"] = (
                    module._v5_derive_l2_size_eviction_attestation(
                        run_id=args.run_id,
                        nonce=args.nonce,
                        phase2_summary=json.loads(phase2_summary_bytes),
                        phase2_summary_sha256=hashlib.sha256(
                            phase2_summary_bytes
                        ).hexdigest(),
                        phase3_summary=json.loads(phase3_summary_bytes),
                        phase3_summary_sha256=hashlib.sha256(
                            phase3_summary_bytes
                        ).hexdigest(),
                    )
                )
                binding_bytes = phase_binding_bytes
                binding = phase_binding
        envelope = {
            "schema": module.V5_PRODUCER_ENVELOPE_SCHEMA,
            "producer": args.v5_fixture_producer,
            "run_id": args.run_id,
            "nonce": args.nonce,
            "captures": [
                _fixture_b64(
                    json.dumps(
                        capture,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                )
            ],
        }
        if coordinated:
            envelope.update(
                {
                    "session_id": (
                        "wrong-envelope-session"
                        if args.fixture_mode == "mismatched_envelope"
                        else session_id
                    ),
                    "session_binding_sha256": hashlib.sha256(
                        binding_bytes
                    ).hexdigest(),
                    "captured_during_ui_hold": True,
                }
            )
        if (
            coordinated
            and args.v5_fixture_producer in {"ui", "api"}
            and args.v5_active_phase_index is not None
        ):
            active_phase = module.V5_CACHE_PHASES[
                args.v5_active_phase_index
            ]
            envelope.update(
                {
                    "phase_index": active_phase["index"],
                    "phase_name": active_phase["name"],
                    "representative_id": active_phase[
                        "representative_id"
                    ],
                    "ui_action_profile": active_phase[
                        "ui_action_profile"
                    ],
                    "ui_turn_count": active_phase["ui_turn_count"],
                    "api_action_profile": active_phase[
                        "api_action_profile"
                    ],
                }
            )
        if coordinated and args.v5_fixture_producer == "ui":
            envelope["release_sha256"] = hashlib.sha256(release_bytes).hexdigest()
            if phase_releases is not None:
                envelope["phase_releases"] = phase_releases
        if coordinated and args.v5_fixture_producer == "cache":
            envelope["phase_bindings"] = cache_phase_bindings
        os.write(
            args.output_fd,
            json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode(),
        )
        record(f"{args.v5_fixture_producer}_capture_complete")
        if (
            args.v5_fixture_producer in {"ui", "api"}
            and args.v5_active_phase_index is not None
        ):
            active_phase = module.V5_CACHE_PHASES[
                args.v5_active_phase_index
            ]
            record(
                f"{args.v5_fixture_producer}_capture_complete:"
                f"{active_phase['name']}"
            )
        sys.stdin.buffer.readline()
        return 0
    command = args.v5_fixture_command
    if command == "full_python_suite":
        print("collected 1000 items")
        print("1000 passed in 1.00s")
    elif command == "full_panel_suite":
        print("Test Files 5 passed (5)")
        print("Tests 50 passed (50)")
    elif command == "typecheck":
        print("typecheck complete")
    elif command == "production_build":
        assert args.output_root
        for relative in (
            "main/index.mjs",
            "preload/index.js",
            "renderer/index.html",
        ):
            path = args.output_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture {relative}\n", encoding="utf-8")
        print("ok bundled JANG provenance matches source (2.5.39 @ efde446)")
        print("bundled-python: all critical imports ok")
    elif command == "jang_build":
        assert args.distribution_root
        args.distribution_root.mkdir(parents=True, exist_ok=True)
        (args.distribution_root / "jang_tools-2.5.39-py3-none-any.whl").write_bytes(
            b"fixture-wheel"
        )
        (args.distribution_root / "jang_tools-2.5.39.tar.gz").write_bytes(
            b"fixture-sdist"
        )
    elif command == "jang_venv":
        assert args.isolated_venv
        package = (
            args.isolated_venv
            / "lib/python3.13/site-packages/jang_tools/__init__.py"
        )
        package.parent.mkdir(parents=True, exist_ok=True)
        package.write_text('__version__ = "2.5.39"\n', encoding="utf-8")
    elif command == "jang_import":
        assert args.isolated_venv
        package = (
            args.isolated_venv
            / "lib/python3.13/site-packages/jang_tools/__init__.py"
        )
        print(
            "VMLINUX_INSTALLED_IMPORT="
            + json.dumps(
                {"file": str(package), "version": "2.5.39"},
                sort_keys=True,
            )
        )
    elif command == "jang_test":
        assert args.isolated_venv
        package = (
            args.isolated_venv
            / "lib/python3.13/site-packages/jang_tools/__init__.py"
        )
        print(
            "VMLINUX_TEST_IMPORT="
            + json.dumps(
                {
                    "file": str(package),
                    "version": "2.5.39",
                    "source_manifest_sha256": "a" * 64,
                    "installed_manifest_sha256": "a" * 64,
                    "package_file_count": 1,
                    "laguna_mixed_affine_shape_bits": [6, 6],
                },
                sort_keys=True,
            )
        )
        print("collected 3 items")
        print("3 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_v5_fixture_child(sys.argv[1:]))
