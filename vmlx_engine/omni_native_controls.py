"""Controls implemented by the native Omni tokenizer template."""


def native_template_options(kwargs):
    """Validate prompt hints separately from enforced decoder budgets."""
    from fastapi import HTTPException

    options = {}
    budget = kwargs.get("reasoning_budget")
    if budget is not None:
        if type(budget) is not int or budget < 0:
            raise HTTPException(400, "chat_template_kwargs.reasoning_budget must be a nonnegative integer (a soft prompt hint, not a token cap)")
        options["reasoning_budget"] = budget
    truncate = kwargs.get("truncate_history_thinking")
    if truncate is not None:
        if type(truncate) is not bool:
            raise HTTPException(400, "chat_template_kwargs.truncate_history_thinking must be a boolean")
        options["truncate_history_thinking"] = truncate
    return options


def has_native_thinking_directive(messages):
    """Require full native rendering when earlier text can affect the rail."""
    for message in messages:
        if message.get("role") not in ("system", "user"):
            continue
        content = message.get("content")
        texts = [content] if isinstance(content, str) else [
            p.get("text", "") for p in content or [] if p.get("type") == "text"
        ]
        if any("/think" in text or "/no_think" in text for text in texts):
            return True
    return False


def record_prompt_rail(session, prompt):
    """Observe the actual generation suffix before any decoded token arrives.

    Slash directives are resolved by the vendor template, not reimplemented
    here. A changed template must not silently route reasoning into content.
    """
    suffix = prompt.rstrip()
    if suffix.endswith("<think></think>"):
        off = True
    elif suffix.endswith("<think>"):
        off = False
    else:
        raise ValueError("Native Omni template produced an unsupported generation suffix")
    session._vmlx_prompt_thinking_off = off
    callback = getattr(session, "_vmlx_prompt_rail_callback", None)
    if callback is not None:
        callback(off)
