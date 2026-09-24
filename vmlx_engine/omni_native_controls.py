"""Controls implemented by the native Omni tokenizer template."""


def native_media_capability_contract(*, backend, modalities, disk_enabled, disk_policy):
    """Describe this dispatcher's controls without constructing a model owner.

    The caller supplies modalities from component admission. Stage-2 must not
    inherit the Stage-1 template, tool or checkpoint contract.
    """
    stage1 = backend == "stage1"
    return {
        "backend": backend,
        "selection": "media_in_conversation",
        "modalities": list(modalities),
        "supports_thinking": True,
        "thinking_control": "enable_thinking",
        "supports_thinking_budget": False,
        "reasoning_efforts": [],
        "chat_template_kwargs": {
            "reasoning_budget": {
                "type": "integer", "minimum": 0,
                "description": "Soft prompt hint; total output limit still applies",
                "enforced_token_cap": False,
            },
            "truncate_history_thinking": {"type": "boolean"},
        } if stage1 else {},
        "supports_tools": stage1,
        "tool_choices": ["auto", "none"] if stage1 else ["none"],
        "supports_structured_output": False,
        "sampling_controls": ["temperature", "top_p", "max_tokens"],
        "video_controls": ["video_fps", "video_max_frames"] if "video" in modalities else [],
        "media_sources": ["local_file", "base64"],
        "audio_output": False,
        "cache": {
            "type": "native_full_state_ssd",
            "enabled": bool(disk_enabled),
            "state": ["attention", "recurrent"],
            "restore": "longest_exact_causal_prefix",
            "arbitrary_suffix_reuse": False,
            "checkpoint_boundaries": ["input_prefix", "complete_tool_result_batch"],
            "write_fence": "before_output_and_tool_delivery",
            "idle_resident_state": False,
            "dtype_policy": "preserve_native",
            "policy": dict(disk_policy),
        } if stage1 else {"type": "unqualified", "enabled": False},
    }


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
