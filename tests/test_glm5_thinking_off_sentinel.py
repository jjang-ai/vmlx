"""GLM5-next's explicit-off adapter must preserve native empty-rail bytes."""

import pytest

from vmlx_engine.utils.chat_template_kwargs import ensure_thinking_off_sentinel


@pytest.mark.parametrize("tools_present", [False, True])
def test_glm5_next_empty_rail_matches_native_history(tools_present):
    prompt = "<|user|>Read the inventory.<|assistant|><think>"
    assert ensure_thinking_off_sentinel(
        prompt, family_name="glm5_next", tools_present=tools_present
    ) == prompt + "</think>"


def test_glm5_next_keeps_closed_history_and_does_not_invent_rail():
    for prompt in (
        "<|assistant|><think></think>",
        "<|assistant|><think>earlier reasoning</think>answer",
        "<|assistant|>",
    ):
        assert ensure_thinking_off_sentinel(
            prompt, family_name="glm5_next"
        ) == prompt


@pytest.mark.parametrize("family", ["glm4_moe", "glm_moe_dsa", "unknown"])
def test_glm_name_does_not_broaden_native_contract(family):
    prompt = "<|assistant|><think>"
    assert ensure_thinking_off_sentinel(
        prompt, family_name=family, model_name="GLM-5.3-Flash"
    ) == prompt + "\n</think>\n\n"
