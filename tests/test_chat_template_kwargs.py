from vmlx_engine.utils.chat_template_kwargs import build_chat_template_kwargs


def test_enable_thinking_sets_thinking_alias():
    kwargs = build_chat_template_kwargs(enable_thinking=False)

    assert kwargs["enable_thinking"] is False
    assert kwargs["thinking"] is False


def test_enable_thinking_wins_over_conflicting_extra_aliases():
    kwargs = build_chat_template_kwargs(
        enable_thinking=False,
        extra={
            "enable_thinking": True,
            "thinking": True,
            "thinking_budget": 2048,
            "tokenize": True,
            "add_generation_prompt": False,
        },
    )

    assert kwargs["enable_thinking"] is False
    assert kwargs["thinking"] is False
    assert kwargs["thinking_budget"] == 2048
    assert kwargs["tokenize"] is False
    assert kwargs["add_generation_prompt"] is True


def test_processor_path_can_skip_thinking_alias():
    kwargs = build_chat_template_kwargs(
        enable_thinking=True,
        include_thinking_alias=False,
    )

    assert kwargs["enable_thinking"] is True
    assert "thinking" not in kwargs
