from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path("bench/qwen_forward_path_ab.py")
    spec = importlib.util.spec_from_file_location("qwen_forward_path_ab", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_forward_ab_script_has_no_generation_or_sampler_defaults():
    """The Qwen route bench must measure raw forward, not generation policy."""
    module = _load_module()

    assert module.GENERATION_KEYS == {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "max_tokens",
        "max_new_tokens",
        "enable_thinking",
        "reasoning_effort",
    }
    assert module.assert_no_generation_kwargs({}) is None


def test_forward_ab_rejects_generation_or_sampler_kwargs():
    """Diagnostic callers cannot accidentally turn the bench into generation."""
    module = _load_module()

    try:
        module.assert_no_generation_kwargs({"temperature": 0.2})
    except ValueError as exc:
        assert "generation/sampler" in str(exc)
        assert "temperature" in str(exc)
    else:
        raise AssertionError("expected generation kwargs to be rejected")


def test_forward_ab_route_summary_distinguishes_text_and_vlm_language_copy():
    module = _load_module()

    class TextModel:
        pass

    class OuterVLM:
        language_model = TextModel()

    text = module.route_summary(TextModel())
    vlm = module.route_summary(OuterVLM())

    assert text["uses_language_model_attr"] is False
    assert text["forward_class"].endswith("TextModel")
    assert vlm["uses_language_model_attr"] is True
    assert vlm["forward_class"].endswith("TextModel")
