# SPDX-License-Identifier: Apache-2.0
"""Reasoning-mode request normalization tests."""

import json
import sys
import types
from types import SimpleNamespace

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

    high = ChatCompletionRequest(**base, thinking_mode="high")
    assert high.enable_thinking is True
    assert high.reasoning_effort == "high"

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

    high = ResponsesRequest(model="m", input="hi", thinking_mode="high")
    assert high.enable_thinking is True
    assert high.reasoning_effort == "high"


def test_reasoning_effort_none_disables_thinking_for_chat_and_responses():
    from vmlx_engine.api.models import ChatCompletionRequest, ResponsesRequest

    chat = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="none",
    )
    assert chat.enable_thinking is False
    assert chat.reasoning_effort is None

    responses = ResponsesRequest(model="m", input="hi", reasoning={"effort": "none"})
    assert responses.enable_thinking is False
    assert responses.reasoning_effort is None


def test_dsv4_reasoning_effort_preserves_requested_rails(monkeypatch):
    """DSV4 should expose the real requested effort rail.

    Runtime should fix cache/attention/prompt bugs directly, not downgrade a
    public API value like max to another rail as a hidden behavior guard.
    """
    from vmlx_engine import server
    from vmlx_engine.loaders import dsv4_chat_encoder

    monkeypatch.delenv("VMLX_DSV4_RAW_MAX", raising=False)

    # Official DSV4 encoder accepts None/high/max; low/medium map to high
    # for encoder compatibility, while max must not be silently downgraded.
    assert server._normalize_dsv4_reasoning_effort("low") == "high"
    assert server._normalize_dsv4_reasoning_effort("medium") == "high"
    assert server._normalize_dsv4_reasoning_effort("high") == "high"
    assert server._normalize_dsv4_reasoning_effort("max") == "max"
    assert server._normalize_dsv4_reasoning_effort(None) is None
    assert dsv4_chat_encoder._resolve_mode_and_effort(True, "max") == (
        "thinking",
        "max",
    )
    assert dsv4_chat_encoder._resolve_mode_and_effort(None, "max") == (
        "thinking",
        "max",
    )


def test_dsv4_thinking_disabled_uses_official_chat_closed_thinking_rail():
    """DSV4 non-think mode is the model-owned ``</think>`` chat rail.

    Exact-code failures on that rail must be fixed or classified directly;
    silently switching explicit thinking-disabled requests to the thinking rail
    would be a hidden force-on workaround.
    """
    from vmlx_engine.loaders import dsv4_chat_encoder

    assert dsv4_chat_encoder._resolve_mode_and_effort(False, None) == (
        "chat",
        None,
    )
    assert dsv4_chat_encoder._resolve_mode_and_effort(None, None) == (
        "chat",
        None,
    )
    assert dsv4_chat_encoder._resolve_mode_and_effort(True, None) == (
        "thinking",
        None,
    )


def test_invalid_thinking_mode_rejected():
    from vmlx_engine.api.models import ChatCompletionRequest

    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            thinking_mode="slider-72",
        )


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


def test_dsv4_token_split_reads_batched_output_token_ids():
    """BatchedEngine returns RequestOutput.output_token_ids, not .tokens."""
    from types import SimpleNamespace

    from vmlx_engine import server

    output = SimpleNamespace(output_token_ids=[128821, 11, 128822, 21])

    assert server._output_token_ids_for_reasoning(output) == [
        128821,
        11,
        128822,
        21,
    ]


def test_tool_parse_strips_full_and_orphan_think_markers():
    """Tool parsing must not see stale think markers after suppressed reasoning.

    ZAYA can emit a complete think block followed by an extra orphan close tag
    when the product path has thinking disabled.  Leaving the orphan close in
    place makes downstream display cleaning re-open a fake <think> block and
    leaks markup to users.
    """
    from vmlx_engine import server

    text = "<think>hidden</think>\ncall get_weather\n</think>\nagain"

    stripped = server._strip_think_for_tool_parse(text)

    assert stripped == "call get_weather\nagain"
    assert "<think>" not in stripped
    assert "</think>" not in stripped


def test_display_cleanup_removes_residual_think_markup_without_reopening():
    from vmlx_engine import server

    assert server._strip_residual_think_markup_for_display(
        "<think>hidden</think>Visible"
    ) == "Visible"
    assert server._strip_residual_think_markup_for_display(
        "Visible</think> tail"
    ) == "Visible tail"
    assert server._strip_residual_think_markup_for_display(
        "<think>Visible"
    ) == "Visible"


def test_dsv4_thinking_policy_defaults_to_direct_rail_when_not_requested(monkeypatch):
    """DSV4 Auto must not secretly enable hidden reasoning."""
    from vmlx_engine import server

    decision = server._resolve_dsv4_thinking_policy(
        requested_enable_thinking=None,
        effort_requested=False,
        tools_present=False,
        tool_choice=None,
    )

    assert decision.enable_thinking is False
    assert decision.reasoning_effort_allowed is False
    assert decision.reason == "thinking_not_requested"


def test_dsv4_thinking_policy_honors_requested_thinking(monkeypatch):
    from vmlx_engine import server

    decision = server._resolve_dsv4_thinking_policy(
        requested_enable_thinking=True,
        effort_requested=False,
        tools_present=False,
        tool_choice=None,
    )

    assert decision.enable_thinking is True
    assert decision.reasoning_effort_allowed is True
    assert decision.reason == "requested_thinking"


def test_dsv4_thinking_policy_does_not_have_env_force_direct_guard(monkeypatch):
    from vmlx_engine import server

    monkeypatch.setenv("VMLX_DSV4_FORCE_DIRECT_RAIL", "1")

    decision = server._resolve_dsv4_thinking_policy(
        requested_enable_thinking=True,
        effort_requested=True,
        tools_present=False,
        tool_choice=None,
    )

    assert decision.enable_thinking is True
    assert decision.reasoning_effort_allowed is True
    assert decision.reason == "requested_thinking"


def test_dsv4_thinking_policy_reasoning_effort_implies_thinking(monkeypatch):
    from vmlx_engine import server

    decision = server._resolve_dsv4_thinking_policy(
        requested_enable_thinking=None,
        effort_requested=True,
        tools_present=False,
        tool_choice=None,
    )

    assert decision.enable_thinking is True
    assert decision.reasoning_effort_allowed is True
    assert decision.reason == "requested_thinking"


def test_dsv4_thinking_policy_does_not_force_tool_calls_to_direct_rail(monkeypatch):
    from vmlx_engine import server

    decision = server._resolve_dsv4_thinking_policy(
        requested_enable_thinking=True,
        effort_requested=True,
        tools_present=True,
        tool_choice="auto",
    )

    assert decision.enable_thinking is True
    assert decision.reasoning_effort_allowed is True
    assert decision.reason == "requested_thinking"


def test_dsv4_thinking_policy_explicit_false_stays_direct(monkeypatch):
    """Caller passing enable_thinking=False explicitly must not be force-flipped."""
    from vmlx_engine import server

    decision = server._resolve_dsv4_thinking_policy(
        requested_enable_thinking=False,
        effort_requested=False,
        tools_present=False,
        tool_choice=None,
    )

    assert decision.enable_thinking is False
    assert decision.reasoning_effort_allowed is False
    assert decision.reason == "thinking_disabled"


def test_dsv4_thinking_policy_omitted_does_not_force_reasoning(monkeypatch):
    from vmlx_engine import server

    decision = server._resolve_dsv4_thinking_policy(
        requested_enable_thinking=None,
        effort_requested=False,
        tools_present=False,
        tool_choice=None,
    )

    assert decision.enable_thinking is False
    assert decision.reasoning_effort_allowed is False
    assert decision.reason == "thinking_not_requested"


def test_dsv4_thinking_policy_uses_bundle_default_thinking_when_omitted(monkeypatch):
    """Model-owned DSV4 default_mode=thinking is honored without hardcoding."""
    from vmlx_engine import server

    decision = server._resolve_dsv4_thinking_policy(
        requested_enable_thinking=None,
        effort_requested=False,
        tools_present=False,
        tool_choice=None,
        default_mode="thinking",
    )

    assert decision.enable_thinking is True
    assert decision.reasoning_effort_allowed is True
    assert decision.reason == "bundle_default_thinking"


def test_dsv4_thinking_policy_explicit_false_overrides_bundle_default(monkeypatch):
    """Per-request disable remains authoritative over bundle defaults."""
    from vmlx_engine import server

    decision = server._resolve_dsv4_thinking_policy(
        requested_enable_thinking=False,
        effort_requested=False,
        tools_present=False,
        tool_choice=None,
        default_mode="thinking",
    )

    assert decision.enable_thinking is False
    assert decision.reasoning_effort_allowed is False
    assert decision.reason == "thinking_disabled"


def test_dsv4_bundle_defaults_apply_only_when_request_omits_values(tmp_path, monkeypatch):
    """Bundle sampling defaults are startup/API defaults, not hidden request rewrites."""
    import json
    from vmlx_engine import server

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v4"}))
    (tmp_path / "jang_config.json").write_text(json.dumps({
        "chat": {
            "sampling_defaults": {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_new_tokens": 4096,
                "repetition_penalty_thinking": 1.0,
                "repetition_penalty_chat": 1.05,
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

    assert server._resolve_temperature(None) == 0.7
    assert server._resolve_top_p(None) == 0.95
    assert server._resolve_repetition_penalty(None, enable_thinking=True) == 1.10
    assert server._resolve_max_tokens(None) == 4096

    # Explicit request values are always preserved.
    assert server._resolve_temperature(0.7) == 0.7
    assert server._resolve_top_p(1.0) == 1.0
    assert server._resolve_repetition_penalty(1.10, enable_thinking=True) == 1.10
    assert server._resolve_temperature(0.2) == 0.2
    assert server._resolve_top_p(0.8) == 0.8
    assert server._resolve_repetition_penalty(1.25) == 1.25
    assert server._resolve_repetition_penalty(1.05, enable_thinking=True) == 1.05


def test_dsv4_jang_chat_defaults_override_generation_config_when_cli_unset(
    tmp_path, monkeypatch
):
    """DSV4 uses jang_config.chat defaults over generic generation_config values.

    Real DSV4 Flash bundles currently carry generation_config temperature/top_p
    as 1.0/1.0, while jang_config.chat.sampling_defaults declares the audited
    chat runtime defaults. This pins the server resolver behavior directly so a
    future route refactor cannot silently fall back to the wrong source.
    """
    import json
    from vmlx_engine import server

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v4"}))
    (tmp_path / "generation_config.json").write_text(json.dumps({
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 99,
        "repetition_penalty": 1.2,
        "max_new_tokens": 8192,
    }))
    (tmp_path / "jang_config.json").write_text(json.dumps({
        "chat": {
            "reasoning": {"default_mode": "chat"},
            "sampling_defaults": {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 0,
                "repetition_penalty": 1.0,
                "repetition_penalty_thinking": 1.0,
                "repetition_penalty_chat": 1.05,
                "max_new_tokens": 4096,
            },
        },
    }))

    monkeypatch.setattr(server, "_model_path", str(tmp_path))
    monkeypatch.setattr(server, "_model_name", "dsv4")
    monkeypatch.setattr(server, "_default_temperature", None)
    monkeypatch.setattr(server, "_default_top_p", None)
    monkeypatch.setattr(server, "_default_top_k", None)
    monkeypatch.setattr(server, "_default_min_p", None)
    monkeypatch.setattr(server, "_default_repetition_penalty", None)
    server._jang_sampling_defaults_cache.clear()
    server._generation_defaults_cache.clear()

    assert server._resolve_temperature(None, str(tmp_path)) == 0.6
    assert server._resolve_top_p(None, str(tmp_path)) == 0.95
    assert server._resolve_top_k(None, str(tmp_path)) == 0
    assert server._resolve_max_tokens(None, str(tmp_path)) == 4096
    assert (
        server._resolve_repetition_penalty(
            None,
            str(tmp_path),
            enable_thinking=False,
        )
        == 1.0
    )
    assert (
        server._resolve_repetition_penalty(
            None,
            str(tmp_path),
            enable_thinking=True,
        )
        == 1.0
    )


def test_ling_stamped_bailing_family_preserves_sampling_values(tmp_path, monkeypatch):
    """Ling JANGTQ bundles stamp capabilities.family=bailing_hybrid.

    That is the HF model_type, not the canonical vMLX family. The server must
    still identify it as "ling" for runtime/cache policy, but it must not hide
    model/runtime issues behind a family-wide repetition-penalty floor.
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
    monkeypatch.setattr(server, "_default_repetition_penalty", None)
    server._jang_sampling_defaults_cache.clear()
    server._generation_defaults_cache.clear()

    assert server._model_family_for_defaults() == "ling"
    assert server._resolve_repetition_penalty(None) is None
    assert server._resolve_repetition_penalty(1.0) == 1.0


def test_ling_crack_uses_bundle_temperature_without_path_guard(
    tmp_path, monkeypatch
):
    """Ling CRACK must use metadata defaults, not a filename-based cold hack."""
    import json
    from vmlx_engine import server
    import vmlx_engine.model_config_registry as mcr

    bundle = tmp_path / "Ling-2.6-flash-JANGTQ2-CRACK"
    bundle.mkdir()
    (bundle / "config.json").write_text(json.dumps({"model_type": "bailing_hybrid"}))
    (bundle / "jang_config.json").write_text(json.dumps({
        "capabilities": {
            "family": "bailing_hybrid",
            "cache_type": "hybrid",
            "tool_parser": "deepseek",
            "reasoning_parser": "deepseek_r1",
            "think_in_template": False,
            "modality": "text",
        },
        "chat": {
            "sampling_defaults": {
                "temperature": 0.6,
                "top_p": 0.95,
                "repetition_penalty": 1.0,
            }
        },
    }))

    mcr.ModelConfigRegistry._instance = None
    mcr._configs_loaded = False
    mcr.get_model_config_registry().clear_cache()
    monkeypatch.setattr(server, "_model_path", str(bundle))
    monkeypatch.setattr(server, "_model_name", "ling")
    monkeypatch.setattr(server, "_default_temperature", None)
    server._jang_sampling_defaults_cache.clear()
    server._generation_defaults_cache.clear()

    assert server._model_family_for_defaults() == "ling"
    assert server._resolve_temperature(None) == 0.6
    assert server._resolve_temperature(0.7) == 0.7

    monkeypatch.setattr(server, "_default_temperature", 0.7)
    assert server._resolve_temperature(None) == 0.7

    monkeypatch.setattr(server, "_default_temperature", 0.4)
    assert server._resolve_temperature(None) == 0.4


def test_generation_config_top_k_and_min_p_are_used_when_request_omits_values(
    tmp_path, monkeypatch
):
    """Native top_k/min_p from generation_config must reach API generation paths."""
    import json
    from vmlx_engine import server

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "minimax_m2"}))
    (tmp_path / "generation_config.json").write_text(json.dumps({
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.05,
        "max_new_tokens": 8192,
    }))

    monkeypatch.setattr(server, "_model_path", str(tmp_path))
    monkeypatch.setattr(server, "_model_name", "minimax")
    monkeypatch.setattr(server, "_default_top_k", None)
    monkeypatch.setattr(server, "_default_min_p", None)
    server._jang_sampling_defaults_cache.clear()
    server._generation_defaults_cache.clear()

    assert server._resolve_top_k(None) == 40
    assert server._resolve_top_k(20) == 20
    assert server._resolve_min_p(None) == 0.05
    assert server._resolve_min_p(0.02) == 0.02


def test_ling_non_crack_keeps_normal_family_temperature(tmp_path, monkeypatch):
    """Bundles with no sampling metadata use neutral engine defaults."""
    import json
    from vmlx_engine import server
    import vmlx_engine.model_config_registry as mcr

    bundle = tmp_path / "Ling-2.6-flash-JANGTQ2"
    bundle.mkdir()
    (bundle / "config.json").write_text(json.dumps({"model_type": "bailing_hybrid"}))

    mcr.ModelConfigRegistry._instance = None
    mcr._configs_loaded = False
    mcr.get_model_config_registry().clear_cache()
    monkeypatch.setattr(server, "_model_path", str(bundle))
    monkeypatch.setattr(server, "_model_name", "ling")
    monkeypatch.setattr(server, "_default_temperature", None)
    server._jang_sampling_defaults_cache.clear()
    server._generation_defaults_cache.clear()

    assert server._model_family_for_defaults() == "ling"
    assert server._resolve_temperature(None) == 0.0


def test_missing_bundle_sampling_defaults_do_not_force_sampler(tmp_path, monkeypatch):
    """No bundle/default metadata means neutral MLX-style sampling, not 0.7/0.9."""
    import json
    from vmlx_engine import server

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "zaya_vl"}))
    (tmp_path / "generation_config.json").write_text(json.dumps({
        "eos_token_id": 262143,
        "pad_token_id": 0,
    }))
    (tmp_path / "jang_config.json").write_text(json.dumps({
        "capabilities": {
            "family": "zaya1_vl",
            "reasoning_parser": "qwen3",
            "think_in_template": False,
        }
    }))

    monkeypatch.setattr(server, "_model_path", str(tmp_path))
    monkeypatch.setattr(server, "_model_name", "zaya-vl")
    monkeypatch.setattr(server, "_default_temperature", None)
    monkeypatch.setattr(server, "_default_top_p", None)
    monkeypatch.setattr(server, "_default_top_k", None)
    server._jang_sampling_defaults_cache.clear()
    server._generation_defaults_cache.clear()

    assert server._resolve_temperature(None) == 0.0
    assert server._resolve_top_p(None) == 1.0
    assert server._resolve_top_k(None) == 0
    assert server._resolve_temperature(0.7) == 0.7
    assert server._resolve_top_p(0.95) == 0.95
    assert server._resolve_top_k(40) == 40


def test_generation_config_do_sample_false_resolves_greedy_when_request_omits_sampling(
    tmp_path, monkeypatch
):
    """`do_sample=false` is the bundle contract, not a display-only field."""
    import json
    from vmlx_engine import server

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "mimo_v2"}))
    (tmp_path / "generation_config.json").write_text(json.dumps({
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "max_new_tokens": 2048,
    }))

    monkeypatch.setattr(server, "_model_path", str(tmp_path))
    monkeypatch.setattr(server, "_model_name", "mimo-v25")
    monkeypatch.setattr(server, "_default_temperature", None)
    monkeypatch.setattr(server, "_default_top_p", None)
    monkeypatch.setattr(server, "_default_top_k", None)
    monkeypatch.setattr(server, "_default_max_tokens_explicit", False, raising=False)
    server._jang_sampling_defaults_cache.clear()
    server._generation_defaults_cache.clear()

    assert server._resolve_temperature(None) == 0.0
    assert server._resolve_top_p(None) == 1.0
    assert server._resolve_top_k(None) == 0
    assert server._resolve_max_tokens(None) == 2048
    assert server._resolve_temperature(0.6) == 0.6
    assert server._resolve_top_p(0.9) == 0.9
    assert server._resolve_top_k(40) == 40


def test_jang_chat_sampling_overrides_generation_config_do_sample_false(
    tmp_path, monkeypatch
):
    """JANG chat metadata remains higher priority than generic generation config."""
    import json
    from vmlx_engine import server

    (tmp_path / "generation_config.json").write_text(json.dumps({
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 0.95,
    }))
    (tmp_path / "jang_config.json").write_text(json.dumps({
        "chat": {
            "sampling_defaults": {
                "temperature": 0.55,
                "top_p": 0.92,
                "top_k": 64,
            }
        }
    }))

    monkeypatch.setattr(server, "_model_path", str(tmp_path))
    monkeypatch.setattr(server, "_model_name", "jang-chat")
    monkeypatch.setattr(server, "_default_temperature", None)
    monkeypatch.setattr(server, "_default_top_p", None)
    monkeypatch.setattr(server, "_default_top_k", None)
    server._jang_sampling_defaults_cache.clear()
    server._generation_defaults_cache.clear()

    assert server._resolve_temperature(None) == 0.55
    assert server._resolve_top_p(None) == 0.92
    assert server._resolve_top_k(None) == 64


@pytest.mark.parametrize("model_type", ["bailing_hybrid", "bailing_moe_v2_5"])
def test_ling_suppresses_reasoning_parser_and_stale_think_in_template(
    tmp_path, monkeypatch, model_type
):
    """Ling/Bailing is not a reasoning model in vMLX.

    Stale bundle stamps may still claim deepseek_r1 or supports_thinking=True,
    but the product contract is plain visible content. Tool parser stays
    DeepSeek because tool calling is orthogonal to reasoning.
    """
    import json
    from vmlx_engine import server
    import vmlx_engine.model_config_registry as mcr

    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    (tmp_path / "jang_config.json").write_text(json.dumps({
        "capabilities": {
            "family": model_type,
            "cache_type": "hybrid",
            "tool_parser": "deepseek",
            "reasoning_parser": "deepseek_r1",
            "think_in_template": True,
            "supports_thinking": True,
            "modality": "text",
        }
    }))

    mcr.ModelConfigRegistry._instance = None
    mcr._configs_loaded = False
    registry = mcr.get_model_config_registry()
    registry.clear_cache()
    monkeypatch.setattr(server, "_default_enable_thinking", None)

    cfg = registry.lookup(str(tmp_path))
    resolved = server._resolve_enable_thinking(
        request_value=None,
        ct_kwargs={},
        tools_present=False,
        model_key=str(tmp_path),
        engine=None,
        auto_detect=True,
    )

    assert cfg.family_name == "ling"
    assert cfg.supports_thinking is False
    assert cfg.reasoning_parser is None
    assert cfg.think_in_template is False
    assert resolved is False


def test_panel_ling_registry_does_not_advertise_reasoning_parser():
    """Panel auto-detect must match engine Ling no-reasoning contract."""
    from pathlib import Path

    source = Path("./panel/src/main/model-config-registry.ts").read_text()
    ling_line = next(line for line in source.splitlines() if "registerFamily('ling'" in line)

    assert "cacheType: 'hybrid'" in ling_line
    assert "toolParser: 'deepseek'" in ling_line
    assert "reasoningParser" not in ling_line
    assert "next.family === 'ling'" in source


def test_minimax_m2_preserves_sampling_values_without_family_floor(tmp_path, monkeypatch):
    """MiniMax loops must be fixed in runtime/template/cache code, not hidden
    behind a family-wide repetition-penalty floor that slows every decode."""
    import json
    from vmlx_engine import server

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "minimax_m2"}))
    (tmp_path / "jang_config.json").write_text(json.dumps({
        "capabilities": {
            "family": "minimax",
            "cache_type": "kv",
            "tool_parser": "minimax",
            "reasoning_parser": "minimax_m2",
            "think_in_template": True,
            "modality": "text",
        }
    }))

    monkeypatch.setattr(server, "_model_path", str(tmp_path))
    monkeypatch.setattr(server, "_model_name", "minimax")
    monkeypatch.setattr(server, "_default_repetition_penalty", None)
    server._jang_sampling_defaults_cache.clear()
    server._generation_defaults_cache.clear()

    assert server._model_family_for_defaults() == "minimax_m2"
    assert server._resolve_repetition_penalty(None, enable_thinking=True) is None
    assert server._resolve_repetition_penalty(1.15, enable_thinking=True) == 1.15
    assert server._resolve_repetition_penalty(1.25, enable_thinking=True) == 1.25


def test_enable_thinking_auto_stays_unset_for_reasoning_families(monkeypatch):
    """Auto is not a hidden behavior choice.

    The adapters and UI may omit enable_thinking to ask the native
    tokenizer/template/runtime to decide. The server must not convert that
    omission into family-specific forced on/off behavior.
    """
    from vmlx_engine import server

    monkeypatch.setattr(server, "_default_enable_thinking", None)

    for model_key in ("gemma4", "minimax", "qwen3_5", "qwen3_5_moe", "unknown"):
        assert server._resolve_enable_thinking(
            request_value=None,
            ct_kwargs={},
            tools_present=model_key == "gemma4",
            model_key=model_key,
            engine=None,
            auto_detect=True,
        ) is None


def test_enable_thinking_explicit_values_still_win_for_reasoning_families(monkeypatch):
    from vmlx_engine import server

    monkeypatch.setattr(server, "_default_enable_thinking", None)

    assert server._resolve_enable_thinking(
        request_value=False,
        ct_kwargs={},
        tools_present=True,
        model_key="gemma4",
        engine=None,
        auto_detect=True,
    ) is False
    assert server._resolve_enable_thinking(
        request_value=True,
        ct_kwargs={},
        tools_present=False,
        model_key="qwen3_5",
        engine=None,
        auto_detect=True,
    ) is True
    assert server._resolve_enable_thinking(
        request_value=None,
        ct_kwargs={"enable_thinking": False},
        tools_present=False,
        model_key="minimax",
        engine=None,
        auto_detect=True,
    ) is False


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
    monkeypatch.setattr(
        server,
        "_model_quantization_status",
        lambda _path: {"codec": "turboquant_codebook", "mxtq_bits": 2},
    )
    monkeypatch.setattr(
        server,
        "_model_acceleration_status",
        lambda _path: {
            "kernel_type": "turboquant_codebook",
            "metal_na_capable": False,
            "metal_na_active_on_host": False,
        },
    )

    caps = await server.model_capabilities("loaded-model")

    assert caps["cache"]["prefix"] is True
    assert caps["cache"]["type"] == "paged"
    assert caps["cache"]["paged"] is True
    assert caps["cache"]["block_disk_l2"] is True
    assert caps["cache"]["native"]["family"] == "deepseek_v4"
    assert caps["cache"]["native"]["generic_turboquant_kv"]["enabled"] is False
    assert caps["quantization"]["codec"] == "turboquant_codebook"
    assert caps["acceleration"]["kernel_type"] == "turboquant_codebook"
    assert caps["acceleration"]["metal_na_active_on_host"] is False


@pytest.mark.asyncio
async def test_hy_v3_capabilities_advertise_only_documented_reasoning_efforts(monkeypatch):
    """Hy3 handoff documents no_think/low/high, not medium or max."""
    from types import SimpleNamespace

    import vmlx_engine.model_config_registry as mcr
    from vmlx_engine import server

    cfg = SimpleNamespace(
        family_name="hy_v3",
        reasoning_parser="deepseek_r1",
        tool_parser="hunyuan",
        think_in_template=False,
        supports_thinking=True,
        is_mllm=False,
    )

    class _Registry:
        def lookup(self, _model_key):
            return cfg

    monkeypatch.setattr(mcr, "get_model_config_registry", lambda: _Registry())
    monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=False))
    monkeypatch.setattr(server, "_model_path", "")
    monkeypatch.setattr(server, "_model_name", "")

    caps = await server.model_capabilities("hy3-preview")

    assert caps["family"] == "hy_v3"
    assert caps["supports_thinking"] is True
    assert caps["reasoning_efforts"] == ["low", "high"]
    assert "medium" not in caps["reasoning_efforts"]
    assert "max" not in caps["reasoning_efforts"]


def test_hy_v3_reasoning_policy_maps_enable_thinking_to_template_effort(monkeypatch):
    """Hy3 ignores enable_thinking directly; vMLX must feed reasoning_effort."""
    from types import SimpleNamespace

    import vmlx_engine.model_config_registry as mcr
    from vmlx_engine import server

    cfg = SimpleNamespace(family_name="hy_v3")

    class _Registry:
        def lookup(self, _model_key):
            return cfg

    monkeypatch.setattr(mcr, "get_model_config_registry", lambda: _Registry())

    kwargs = {}
    ct_kwargs = {}
    server._apply_hy3_reasoning_policy(
        kwargs,
        ct_kwargs,
        model_key="hy3-preview",
        enable_thinking=True,
    )
    assert kwargs["reasoning_effort"] == "high"
    assert ct_kwargs["reasoning_effort"] == "high"

    kwargs = {"reasoning_effort": "medium"}
    ct_kwargs = {"reasoning_effort": "medium"}
    server._apply_hy3_reasoning_policy(
        kwargs,
        ct_kwargs,
        model_key="hy3-preview",
        enable_thinking=True,
    )
    assert kwargs["reasoning_effort"] == "high"
    assert ct_kwargs["reasoning_effort"] == "high"

    kwargs = {}
    ct_kwargs = {}
    server._apply_hy3_reasoning_policy(
        kwargs,
        ct_kwargs,
        model_key="hy3-preview",
        enable_thinking=False,
    )
    assert kwargs["reasoning_effort"] == "no_think"
    assert ct_kwargs["reasoning_effort"] == "no_think"


def test_hy_v3_prompt_seeding_is_request_dependent(monkeypatch):
    """Parser seed is true only after low/high reasoning_effort opens <think>."""
    from types import SimpleNamespace

    import vmlx_engine.model_config_registry as mcr
    from vmlx_engine import server

    cfg = SimpleNamespace(family_name="hy_v3")

    class _Registry:
        def lookup(self, _model_key):
            return cfg

    monkeypatch.setattr(mcr, "get_model_config_registry", lambda: _Registry())

    assert server._hy3_prompt_starts_in_reasoning(
        model_key="hy3-preview",
        enable_thinking=True,
        ct_kwargs={"reasoning_effort": "high"},
    ) is True
    assert server._hy3_prompt_starts_in_reasoning(
        model_key="hy3-preview",
        enable_thinking=True,
        ct_kwargs={"reasoning_effort": "no_think"},
    ) is False
    assert server._hy3_prompt_starts_in_reasoning(
        model_key="hy3-preview",
        enable_thinking=False,
        ct_kwargs={"reasoning_effort": "high"},
    ) is False


def test_template_starts_reasoning_is_request_dependent_for_zaya_style_template():
    """Default-off templates can still open a think rail for explicit thinking."""
    from vmlx_engine import server

    class ZayaStyleTokenizer:
        has_thinking = True

        def apply_chat_template(
            self,
            messages,
            *,
            enable_thinking=False,
            add_generation_prompt=True,
            tokenize=False,
        ):
            assert add_generation_prompt is True
            assert tokenize is False
            content = messages[0]["content"]
            if enable_thinking:
                return (
                    f"<|im_start|>user\n{content}<|im_end|>\n"
                    "<|im_start|>assistant\n<think>\n"
                )
            return (
                f"<|im_start|>user\n{content}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n</think>\n\n"
            )

    server._template_starts_reasoning_cache.clear()

    assert server._template_starts_reasoning(
        ZayaStyleTokenizer(), "zaya-test", True
    ) is True
    assert server._template_starts_reasoning(
        ZayaStyleTokenizer(), "zaya-test", False
    ) is False


def test_known_lfm2_config_disables_tokenizer_vocab_thinking_seed():
    """A known non-template-thinking family must not be flipped by vocab alone."""
    from vmlx_engine import server

    class LfmStyleTokenizer:
        has_thinking = True

    cfg = SimpleNamespace(family_name="lfm2", think_in_template=False)

    assert (
        server._apply_tokenizer_thinking_vocab_fallback(
            False,
            reasoning_parser=object(),
            tokenizer=LfmStyleTokenizer(),
            model_config=cfg,
        )
        is False
    )


def test_unknown_config_can_fall_back_to_tokenizer_vocab_thinking_seed():
    """Unknown models keep the conservative tokenizer-vocab fallback."""
    from vmlx_engine import server

    class UnknownTokenizer:
        has_thinking = True

    cfg = SimpleNamespace(family_name="unknown", think_in_template=False)

    assert (
        server._apply_tokenizer_thinking_vocab_fallback(
            False,
            reasoning_parser=object(),
            tokenizer=UnknownTokenizer(),
            model_config=cfg,
        )
        is True
    )


def test_zaya_vl_mllm_plain_template_does_not_synthesize_hidden_only_think(
    tmp_path, monkeypatch
):
    """Plain-template ZAYA1-VL must not synthesize a fake thinking rail.

    Current ZAYA1-VL/JANGTQ and MXFP4 bundles can still declare qwen3
    reasoning in metadata, but their mlx-vlm processor template only emits
    ``assistant: ``. Live proof showed that opening a synthetic qwen3 rail
    produces hidden-only output and EOS, so runtime capability truth demotes
    these artifacts until a real VLM thinking contract is uploaded.
    """
    from vmlx_engine.models.mllm import MLXMultimodalLM

    bundle = tmp_path / "zaya-vl"
    bundle.mkdir()
    (bundle / "config.json").write_text(json.dumps({"model_type": "zaya1_vl"}))
    (bundle / "jang_config.json").write_text(
        json.dumps(
            {
                "capabilities": {
                    "family": "zaya1_vl",
                    "reasoning_parser": "qwen3",
                    "supports_thinking": True,
                    "think_in_template": False,
                }
            }
        )
    )

    pkg = types.ModuleType("mlx_vlm")
    pkg.__path__ = []
    prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")
    prompt_utils.get_chat_template = (
        lambda processor, messages, add_generation_prompt=True, **kwargs: "user: hi\nassistant: "
    )
    monkeypatch.setitem(sys.modules, "mlx_vlm", pkg)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_utils)

    lm = object.__new__(MLXMultimodalLM)
    lm.processor = object()
    lm.config = {"model_type": "zaya1_vl"}
    lm.model_name = str(bundle)

    messages = [{"role": "user", "content": "hi"}]
    assert lm._apply_chat_template(messages, enable_thinking=True).endswith("assistant: ")
    assert "<think>" not in lm._apply_chat_template(messages, enable_thinking=False)


def test_step37_mllm_thinking_off_closes_forced_processor_think(
    tmp_path, monkeypatch
):
    """Step3.7's VLM wrapper must honor thinking-off after processor render.

    The local Step3.7 template always appends an open ``<think>`` generation
    prompt, even when ``enable_thinking=False``. The MLLM wrapper is the prompt
    renderer used by the live app/server path, so it must close that rail before
    generation instead of leaving the model to spend the whole visible budget in
    hidden-thought punctuation.
    """
    from vmlx_engine.models.mllm import MLXMultimodalLM

    bundle = tmp_path / "Step-3.7-Flash-JANG_2L"
    bundle.mkdir()
    (bundle / "config.json").write_text(json.dumps({"model_type": "step3p7"}))

    pkg = types.ModuleType("mlx_vlm")
    pkg.__path__ = []
    prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")

    def fake_template(_processor, _messages, add_generation_prompt=True, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["tools"][0]["function"]["name"] == "run_command"
        return (
            "<|im_start|>system\n# Tools<|im_end|>\n"
            "<|im_start|>user\nUse run_command once.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )

    prompt_utils.get_chat_template = fake_template
    monkeypatch.setitem(sys.modules, "mlx_vlm", pkg)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_utils)

    lm = object.__new__(MLXMultimodalLM)
    lm.processor = object()
    lm.config = {"model_type": "step3p7"}
    lm.model_name = str(bundle)

    rendered = lm._apply_chat_template(
        [{"role": "user", "content": "Use run_command once."}],
        enable_thinking=False,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                },
            }
        ],
    )

    assert rendered.endswith("<think>\n</think>\n\n")


def test_zaya_vl_synthetic_mllm_think_prompt_seed_is_disabled_for_plain_template(
    tmp_path,
):
    """Server reasoning parser seeding must match ZAYA1-VL capability truth."""
    from vmlx_engine import server

    bundle = tmp_path / "zaya-vl"
    bundle.mkdir()
    (bundle / "config.json").write_text(json.dumps({"model_type": "zaya1_vl"}))
    (bundle / "jang_config.json").write_text(
        json.dumps(
            {
                "capabilities": {
                    "family": "zaya1_vl",
                    "reasoning_parser": "qwen3",
                    "supports_thinking": True,
                    "think_in_template": False,
                }
            }
        )
    )

    engine = SimpleNamespace(is_mllm=True)
    assert server._synthetic_mllm_think_prompt_starts(
        model_key=str(bundle),
        enable_thinking=True,
        engine=engine,
    ) is False
    assert server._synthetic_mllm_think_prompt_starts(
        model_key=str(bundle),
        enable_thinking=False,
        engine=engine,
    ) is False
    assert server._synthetic_mllm_think_prompt_starts(
        model_key=str(bundle),
        enable_thinking=True,
        engine=SimpleNamespace(is_mllm=False),
    ) is False


def test_hy_v3_qwen3_reasoning_parser_no_think_does_not_leak_tags():
    """Default Hy3 no_think prompt has closed tags; output must stay visible."""
    from vmlx_engine.reasoning.qwen3_parser import Qwen3ReasoningParser

    parser = Qwen3ReasoningParser()
    parser.reset_state(think_in_prompt=False)

    reasoning, content = parser.extract_reasoning("<think></think>The answer is 4.")

    assert reasoning is None
    assert content == "The answer is 4."
    assert "<think>" not in content
    assert "</think>" not in content


def test_hy_v3_qwen3_reasoning_parser_low_high_routes_reasoning_only():
    """When Hy3 low/high opens <think>, parser must split before content."""
    from vmlx_engine.reasoning.qwen3_parser import Qwen3ReasoningParser

    parser = Qwen3ReasoningParser()
    parser.reset_state(think_in_prompt=True)

    reasoning, content = parser.extract_reasoning("Compute 2+2.</think>The answer is 4.")

    assert reasoning == "Compute 2+2."
    assert content == "The answer is 4."
    assert "<think>" not in content
    assert "</think>" not in content


def test_think_xml_reasoning_parser_is_registered_for_mimo_v2():
    """MiMo's registry id must resolve to a real parser, not stale qwen3."""
    from vmlx_engine.reasoning import get_parser

    parser_cls = get_parser("think_xml")
    parser = parser_cls()

    assert parser_cls.__name__ == "ThinkXmlReasoningParser"
    reasoning, content = parser.extract_reasoning(
        "<think>Use the tool result.</think>FINAL=OK"
    )
    assert reasoning == "Use the tool result."
    assert content == "FINAL=OK"


def test_think_xml_reasoning_parser_no_tags_remains_visible_content():
    """MiMo normal chat text must not be hidden as reasoning without tags."""
    from vmlx_engine.reasoning import get_parser

    parser = get_parser("think_xml")()

    reasoning, content = parser.extract_reasoning("FINAL=OK")

    assert reasoning is None
    assert content == "FINAL=OK"


def test_deepseek_r1_reasoning_parser_no_think_does_not_leak_tags():
    """Closed or implicit DeepSeek-R1 think rails must not leak into content."""
    from vmlx_engine.reasoning.deepseek_r1_parser import DeepSeekR1ReasoningParser

    parser = DeepSeekR1ReasoningParser()
    parser.reset_state(think_in_prompt=False)

    reasoning, content = parser.extract_reasoning("<think></think>The answer is 4.")

    assert reasoning is None
    assert content == "The answer is 4."
    assert "<think>" not in content
    assert "</think>" not in content


def test_deepseek_r1_reasoning_parser_prompt_open_routes_reasoning_only():
    """Families using DeepSeek-R1 with prompt-open think must split on </think>."""
    from vmlx_engine.reasoning.deepseek_r1_parser import DeepSeekR1ReasoningParser

    parser = DeepSeekR1ReasoningParser()
    parser.reset_state(think_in_prompt=True)

    reasoning, content = parser.extract_reasoning("Compute 2+2.</think>The answer is 4.")

    assert reasoning == "Compute 2+2."
    assert content == "The answer is 4."
    assert "<think>" not in content
    assert "</think>" not in content


def test_deepseek_r1_reasoning_parser_orphan_close_is_not_visible():
    """An orphan close marker should be stripped instead of shown to users."""
    from vmlx_engine.reasoning.deepseek_r1_parser import DeepSeekR1ReasoningParser

    parser = DeepSeekR1ReasoningParser()

    reasoning, content = parser.extract_reasoning("</think>The answer is 4.")

    assert reasoning is None
    assert content == "The answer is 4."
    assert "</think>" not in content


def test_deepseek_r1_direct_rail_stray_close_preserves_visible_prefix():
    """DSV4 chat/direct rail may emit a stray </think>; prefix is not hidden reasoning."""
    from vmlx_engine.reasoning.deepseek_r1_parser import DeepSeekR1ReasoningParser

    parser = DeepSeekR1ReasoningParser()
    parser.reset_state(think_in_prompt=False)

    reasoning, content = parser.extract_reasoning(
        "The anchor is ADA LOVELACE.</think>**CERULEAN, 45, ADA LOVELACE**"
    )

    assert reasoning is None
    assert content == "The anchor is ADA LOVELACE.**CERULEAN, 45, ADA LOVELACE**"
    assert "</think>" not in content


def test_gemma4_reasoning_parser_no_thought_does_not_leak_markers():
    """Gemma 4 closed thought rail must produce visible content only."""
    from vmlx_engine.reasoning.gemma4_parser import Gemma4ReasoningParser

    parser = Gemma4ReasoningParser()

    reasoning, content = parser.extract_reasoning(
        "<|channel>thought\n<channel|>The answer is 4.<turn|>"
    )

    assert reasoning is None
    assert content == "The answer is 4."
    assert "<|channel>" not in content
    assert "<channel|>" not in content
    assert "<turn|>" not in content


def test_gemma4_reasoning_parser_routes_thought_channel():
    """Gemma 4 thought channel must split reasoning from final content."""
    from vmlx_engine.reasoning.gemma4_parser import Gemma4ReasoningParser

    parser = Gemma4ReasoningParser()

    reasoning, content = parser.extract_reasoning(
        "thought\nCompute 2+2.<channel|>The answer is 4.<turn|>"
    )

    assert reasoning == "Compute 2+2."
    assert content == "The answer is 4."
    assert "<channel|>" not in content
    assert "<turn|>" not in content


def test_gemma4_reasoning_parser_orphan_channel_close_is_not_visible():
    """A stray Gemma 4 channel close marker should not render as content."""
    from vmlx_engine.reasoning.gemma4_parser import Gemma4ReasoningParser

    parser = Gemma4ReasoningParser()

    reasoning, content = parser.extract_reasoning("<channel|>The answer is 4.<turn|>")

    assert reasoning is None
    assert content == "The answer is 4."
    assert "<channel|>" not in content
    assert "<turn|>" not in content
