# SPDX-License-Identifier: Apache-2.0
"""Tests for OpenAI Realtime compatibility additions."""

import inspect

from vmlx_engine.model_config_registry import get_model_config_registry
import vmlx_engine.server as server


def test_realtime_routes_exist():
    source = inspect.getsource(server)
    assert '/v1/realtime/sessions' in source
    assert '/v1/realtime' in source
    assert '/v1/realtime/client_secrets' in source
    assert '/v1/realtime/transcription_sessions' in source


def test_realtime_model_fallback_detection():
    reg = get_model_config_registry()
    qwen = reg.lookup("Qwen3-Omni-30B-A3B-Instruct")
    vox = reg.lookup("mistralai/Voxtral-Mini-4B-Realtime-2602")
    assert qwen.family_name == "qwen3_omni"
    assert qwen.is_mllm is True
    assert vox.family_name == "voxtral"
    assert vox.is_mllm is True
