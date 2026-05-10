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


def test_dsv4_max_effort_normalizes_to_stable_rail_by_default(monkeypatch):
    """DSV4's raw max rail is not production-safe.

    Public request models can still carry reasoning_effort="max" for generic
    API compatibility, but DSV4 routing must normalize it before the encoder
    gets called.
    """
    from vmlx_engine import server
    from vmlx_engine.loaders import dsv4_chat_encoder

    monkeypatch.delenv("VMLX_DSV4_RAW_MAX", raising=False)

    assert server._normalize_dsv4_reasoning_effort("low") == "high"
    assert server._normalize_dsv4_reasoning_effort("medium") == "high"
    assert server._normalize_dsv4_reasoning_effort("high") == "high"
    assert server._normalize_dsv4_reasoning_effort("max") == "high"
    assert server._normalize_dsv4_reasoning_effort(None) is None
    assert dsv4_chat_encoder._resolve_mode_and_effort(True, "max") == (
        "thinking",
        "high",
    )
    assert dsv4_chat_encoder._resolve_mode_and_effort(None, "max") == (
        "thinking",
        "high",
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
    """DSV4 stays on direct rail only when the request did not ask to think."""
    from vmlx_engine import server

    monkeypatch.delenv("VMLX_DSV4_FORCE_DIRECT_RAIL", raising=False)

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

    monkeypatch.delenv("VMLX_DSV4_FORCE_DIRECT_RAIL", raising=False)

    decision = server._resolve_dsv4_thinking_policy(
        requested_enable_thinking=True,
        effort_requested=False,
        tools_present=False,
        tool_choice=None,
    )

    assert decision.enable_thinking is True
    assert decision.reasoning_effort_allowed is True
    assert decision.reason == "requested_thinking"


def test_dsv4_thinking_policy_debug_force_direct(monkeypatch):
    from vmlx_engine import server

    monkeypatch.setenv("VMLX_DSV4_FORCE_DIRECT_RAIL", "1")

    decision = server._resolve_dsv4_thinking_policy(
        requested_enable_thinking=True,
        effort_requested=True,
        tools_present=False,
        tool_choice=None,
    )

    assert decision.enable_thinking is False
    assert decision.reasoning_effort_allowed is False
    assert decision.reason == "env_force_direct"


def test_dsv4_thinking_policy_reasoning_effort_implies_thinking(monkeypatch):
    from vmlx_engine import server

    monkeypatch.delenv("VMLX_DSV4_FORCE_DIRECT_RAIL", raising=False)

    decision = server._resolve_dsv4_thinking_policy(
        requested_enable_thinking=None,
        effort_requested=True,
        tools_present=False,
        tool_choice=None,
    )

    assert decision.enable_thinking is True
    assert decision.reasoning_effort_allowed is True
    assert decision.reason == "requested_thinking"


def test_dsv4_thinking_policy_keeps_tool_calls_on_direct_rail(monkeypatch):
    from vmlx_engine import server

    monkeypatch.delenv("VMLX_DSV4_FORCE_DIRECT_RAIL", raising=False)

    decision = server._resolve_dsv4_thinking_policy(
        requested_enable_thinking=True,
        effort_requested=True,
        tools_present=True,
        tool_choice="auto",
    )

    assert decision.enable_thinking is False
    assert decision.reasoning_effort_allowed is False
    assert decision.reason == "tool_call_direct_rail"


def test_dsv4_thinking_policy_explicit_false_stays_direct(monkeypatch):
    """Caller passing enable_thinking=False explicitly must NOT be force-flipped.

    The earlier (now-removed) behaviour force-set thinking unless the caller
    sent ``is True``; the current policy honours an explicit False the same as
    a missing value. Without this regression pin, a return to the old
    ``is not True`` semantics would silently re-enable thinking on instruct
    requests.
    """
    from vmlx_engine import server

    monkeypatch.delenv("VMLX_DSV4_FORCE_DIRECT_RAIL", raising=False)

    decision = server._resolve_dsv4_thinking_policy(
        requested_enable_thinking=False,
        effort_requested=False,
        tools_present=False,
        tool_choice=None,
    )

    assert decision.enable_thinking is False
    assert decision.reasoning_effort_allowed is False
    assert decision.reason == "thinking_not_requested"


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

    assert server._resolve_temperature(0.7) == 0.6
    assert server._resolve_top_p(1.0) == 0.95
    assert server._resolve_repetition_penalty(1.10, enable_thinking=True) == 1.0
    assert server._resolve_max_tokens(None) == 4096

    # Non-generic explicit values are preserved.
    assert server._resolve_temperature(0.2) == 0.2
    assert server._resolve_top_p(0.8) == 0.8
    assert server._resolve_repetition_penalty(1.25) == 1.25
    assert server._resolve_repetition_penalty(1.05, enable_thinking=True) == 1.0


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
    monkeypatch.setattr(server, "_default_repetition_penalty", None)
    server._jang_sampling_defaults_cache.clear()
    server._generation_defaults_cache.clear()

    assert server._model_family_for_defaults() == "ling"
    assert server._resolve_repetition_penalty(None) == 1.15
    assert server._resolve_repetition_penalty(1.0) == 1.15


def test_ling_crack_defaults_temperature_to_02_without_overriding_request(
    tmp_path, monkeypatch
):
    """Ling CRACK bundles need a colder server default, but explicit request
    temperatures must remain user intent."""
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
    assert server._resolve_temperature(None) == 0.2
    assert server._resolve_temperature(0.7) == 0.7

    monkeypatch.setattr(server, "_default_temperature", 0.7)
    assert server._resolve_temperature(None) == 0.2

    monkeypatch.setattr(server, "_default_temperature", 0.4)
    assert server._resolve_temperature(None) == 0.4


def test_ling_non_crack_keeps_normal_family_temperature(tmp_path, monkeypatch):
    """The Ling CRACK default must not bleed into regular Ling bundles."""
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
    assert server._resolve_temperature(None) == 0.6


@pytest.mark.parametrize("model_type", ["bailing_hybrid", "bailing_moe_v2_5"])
def test_ling_preserves_opt_in_parser_but_rejects_stale_think_in_template(
    tmp_path, monkeypatch, model_type
):
    """Ling/Bailing is default-off but opt-in reasoning-capable.

    The parser metadata must survive so explicit "detailed thinking on" system
    prompts can extract <think> blocks.  A stale think_in_template=True stamp
    must not survive, because Ling's default template does not auto-open
    <think>; preserving that stale flag would route visible content into
    reasoning_content.
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
            "reasoning_parser": "none",
            "think_in_template": True,
            "supports_thinking": False,
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
    assert cfg.supports_thinking is True
    assert cfg.reasoning_parser == "deepseek_r1"
    assert cfg.think_in_template is False
    assert resolved is True


def test_panel_ling_registry_advertises_opt_in_reasoning_parser():
    """Panel auto-detect must match engine Ling opt-in reasoning contract."""
    from pathlib import Path

    source = Path("./panel/src/main/model-config-registry.ts").read_text()
    ling_line = next(line for line in source.splitlines() if "registerFamily('ling'" in line)

    assert "cacheType: 'hybrid'" in ling_line
    assert "toolParser: 'deepseek'" in ling_line
    assert "reasoningParser: 'deepseek_r1'" in ling_line
    assert "next.family === 'ling'" not in source


def test_minimax_m2_uses_stronger_reasoning_rumination_floor(tmp_path, monkeypatch):
    """MiniMax M2.7 needs a 1.20 floor to close temp-0 reasoning loops."""
    import json
    from vmlx_engine import server

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "minimax_m2"}))
    (tmp_path / "jang_config.json").write_text(json.dumps({
        "capabilities": {
            "family": "minimax",
            "cache_type": "kv",
            "tool_parser": "minimax",
            "reasoning_parser": "qwen3",
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
    assert server._resolve_repetition_penalty(None, enable_thinking=True) == 1.20
    assert server._resolve_repetition_penalty(1.15, enable_thinking=True) == 1.20
    assert server._resolve_repetition_penalty(1.25, enable_thinking=True) == 1.25


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
