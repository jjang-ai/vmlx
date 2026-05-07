# SPDX-License-Identifier: Apache-2.0
"""Reasoning-mode request normalization tests."""

import pytest
from pydantic import ValidationError


def test_chat_thinking_mode_maps_to_enable_thinking_and_effort():
    from vmlx_engine.api.models import ChatCompletionRequest

    base = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}

    instruct = ChatCompletionRequest(**base, thinking_mode="instruct")
    assert instruct.enable_thinking is False
    assert instruct.reasoning_effort is None

    reasoning = ChatCompletionRequest(**base, thinking_mode="reasoning")
    assert reasoning.enable_thinking is True
    assert reasoning.reasoning_effort == "medium"

    max_mode = ChatCompletionRequest(**base, thinking_mode="max")
    assert max_mode.enable_thinking is True
    assert max_mode.reasoning_effort == "max"


def test_thinking_mode_preserves_explicit_reasoning_effort():
    from vmlx_engine.api.models import ChatCompletionRequest

    req = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        thinking_mode="reasoning",
        reasoning_effort="high",
    )

    assert req.enable_thinking is True
    assert req.reasoning_effort == "high"


def test_responses_reasoning_alias_and_thinking_mode():
    from vmlx_engine.api.models import ResponsesRequest

    req = ResponsesRequest(model="m", input="hi", thinking_mode="max")
    assert req.enable_thinking is True
    assert req.reasoning_effort == "max"

    nested = ResponsesRequest(model="m", input="hi", reasoning={"effort": "high"})
    assert nested.enable_thinking is None
    assert nested.reasoning_effort == "high"


def test_invalid_thinking_mode_rejected():
    from vmlx_engine.api.models import ChatCompletionRequest

    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            thinking_mode="slider-72",
        )


def test_dsv4_reasoning_only_nonstream_falls_back_to_visible_content():
    from vmlx_engine import server

    text = server._dsv4_visible_content_fallback(
        "",
        "Project plan draft",
        is_dsv4=True,
        suppress_reasoning=False,
    )

    assert text == "Project plan draft"


def test_dsv4_reasoning_only_fallback_respects_suppression_and_family():
    from vmlx_engine import server

    assert server._dsv4_visible_content_fallback(
        "",
        "hidden",
        is_dsv4=True,
        suppress_reasoning=True,
    ) == ""
    assert server._dsv4_visible_content_fallback(
        "",
        "hidden",
        is_dsv4=False,
        suppress_reasoning=False,
    ) == ""
    assert server._dsv4_visible_content_fallback(
        "visible",
        "hidden",
        is_dsv4=True,
        suppress_reasoning=False,
    ) == "visible"


def test_dsv4_token_id_split_uses_close_marker_boundary():
    from vmlx_engine import server

    class Tokenizer:
        def decode(self, ids):
            table = {
                11: "reason",
                12: "ing",
                21: " Paris",
                1: "<｜end▁of▁sentence｜>",
            }
            return "".join(table.get(i, f"<{i}>") for i in ids)

    reasoning, content = server._dsv4_split_reasoning_from_token_ids(
        [128821, 11, 12, 128822, 21, 1],
        Tokenizer(),
    )

    assert reasoning == "reasoning"
    assert content == "Paris"


def test_dsv4_token_id_split_ignores_unclosed_reasoning():
    from vmlx_engine import server

    class Tokenizer:
        def decode(self, ids):
            return "should not decode"

    assert server._dsv4_split_reasoning_from_token_ids(
        [128821, 11, 12],
        Tokenizer(),
    ) == (None, None)


def test_dsv4_bundle_defaults_override_stale_ui_defaults(tmp_path, monkeypatch):
    """Old chats saved generic 0.7/1.0/1.1 values as explicit request
    overrides. For DSV4 those are not real user intent; they are stale UI
    defaults and must resolve to bundle-calibrated jang_config values."""
    import json
    from vmlx_engine import server

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v4"}))
    (tmp_path / "jang_config.json").write_text(json.dumps({
        "chat": {
            "sampling_defaults": {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_new_tokens": 4096,
                "repetition_penalty_thinking": 1.15,
            }
        }
    }))

    monkeypatch.setattr(server, "_model_path", str(tmp_path))
    monkeypatch.setattr(server, "_model_name", "dsv4")
    monkeypatch.setattr(server, "_default_temperature", 0.7)
    monkeypatch.setattr(server, "_default_top_p", 0.95)
    monkeypatch.setattr(server, "_default_repetition_penalty", 1.10)
    server._jang_sampling_defaults_cache.clear()
    server._generation_defaults_cache.clear()

    assert server._resolve_temperature(0.7) == 0.6
    assert server._resolve_top_p(1.0) == 0.95
    assert server._resolve_repetition_penalty(1.10) == 1.15
    assert server._resolve_max_tokens(None) == 4096

    # Non-generic explicit values are preserved.
    assert server._resolve_temperature(0.2) == 0.2
    assert server._resolve_top_p(0.8) == 0.8
    assert server._resolve_repetition_penalty(1.25) == 1.25
    assert server._resolve_repetition_penalty(1.05) == 1.15


def test_ling_stamped_bailing_family_gets_ling_safety_floor(tmp_path, monkeypatch):
    """Ling JANGTQ bundles stamp capabilities.family=bailing_hybrid.

    That is the HF model_type, not the canonical vMLX family. The server's
    family default resolver must still identify it as "ling" so default CLI/UI
    repetition_penalty=1.0 gets raised to the Ling floor.
    """
    import json
    from vmlx_engine import server
    import vmlx_engine.model_config_registry as mcr

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bailing_hybrid"}))
    (tmp_path / "jang_config.json").write_text(json.dumps({
        "capabilities": {
            "family": "bailing_hybrid",
            "cache_type": "hybrid",
            "tool_parser": "deepseek",
            "reasoning_parser": "deepseek_r1",
            "think_in_template": False,
            "modality": "text",
        }
    }))

    mcr.ModelConfigRegistry._instance = None
    mcr._configs_loaded = False
    mcr.get_model_config_registry().clear_cache()
    monkeypatch.setattr(server, "_model_path", str(tmp_path))
    monkeypatch.setattr(server, "_model_name", "ling")
    monkeypatch.setattr(server, "_default_repetition_penalty", 1.0)
    server._jang_sampling_defaults_cache.clear()
    server._generation_defaults_cache.clear()

    assert server._model_family_for_defaults() == "ling"
    assert server._resolve_repetition_penalty(None) == 1.15
    assert server._resolve_repetition_penalty(1.0) == 1.15


@pytest.mark.asyncio
async def test_capabilities_reports_loaded_scheduler_cache(monkeypatch):
    """The panel gates cache UI from /capabilities, so this must reflect the
    live scheduler, not a stale engine attribute that BatchedEngine does not
    expose directly."""
    from types import SimpleNamespace
    from vmlx_engine import server

    scheduler = SimpleNamespace(
        config=SimpleNamespace(enable_prefix_cache=True),
        block_aware_cache=object(),
        paged_cache_manager=SimpleNamespace(_disk_store=object()),
        memory_aware_cache=None,
        prefix_cache=None,
        _uses_dsv4_cache=True,
    )
    engine_core = SimpleNamespace(scheduler=scheduler)
    async_core = SimpleNamespace(engine=engine_core)
    fake_engine = SimpleNamespace(_engine=async_core, is_mllm=False)

    monkeypatch.setattr(server, "_engine", fake_engine)
    monkeypatch.setattr(server, "_model_path", "")
    monkeypatch.setattr(server, "_model_name", "")

    caps = await server.model_capabilities("loaded-model")

    assert caps["cache"]["prefix"] is True
    assert caps["cache"]["type"] == "paged"
    assert caps["cache"]["paged"] is True
    assert caps["cache"]["block_disk_l2"] is True
