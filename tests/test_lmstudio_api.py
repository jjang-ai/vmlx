# SPDX-License-Identifier: Apache-2.0
"""LM Studio native API v1 compatibility tests."""

import inspect

import pytest
from fastapi import HTTPException

import vmlx_engine.server as server


def test_lmstudio_v1_routes_exist_in_server_source():
    """Server should expose LM Studio native v1 endpoints."""
    source = inspect.getsource(server)
    for route in (
        '/lmstudio/v1/chat',
        '/lmstudio/v1/models',
        '/lmstudio/v1/models/load',
        '/lmstudio/v1/models/unload',
        '/lmstudio/v1/models/download',
        '/lmstudio/v1/models/download/status',
    ):
        assert route in source


def test_normalize_lmstudio_messages_rejects_assistant_role():
    """LM Studio /lmstudio/v1/chat should not accept assistant messages in request."""
    with pytest.raises(HTTPException) as exc:
        server._normalize_lmstudio_messages({
            "messages": [{"role": "assistant", "content": "prior output"}]
        })
    assert exc.value.status_code == 400
    assert "does not accept assistant messages" in str(exc.value.detail)


def test_normalize_lmstudio_messages_accepts_user_input_string():
    """Simple input string should normalize to a single user message."""
    msgs = server._normalize_lmstudio_messages({"input": "hello"})
    assert msgs == [{"role": "user", "content": "hello"}]
