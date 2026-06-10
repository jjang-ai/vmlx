# Responses Long Tool Cache Gate

- model: `models/Qwen3.6-35B-A3B-MXFP8-CRACK-MTP`
- target_chars_per_turn: `800`
- turns: `3`

- final_turn_no_tools: `True`

- final_turn_disable_thinking: `True`

- resolve_tool_calls_in_turn: `True`

- tool_choice_mode: `auto`

- resolution_tool_choice: `none`

- require_tool_call_each_turn: `True`

- require_tool_evidence: `True`

- require_cache_each_turn_after_first: `True`

## Acceptance

- overall_pass: `False`
- turns_completed: `True`
- previous_response_id_used: `True`
- cache_reuse_observed: `False`
- require_cache_each_turn_after_first: `True`
- cache_reuse_each_turn_after_first: `False`
- tool_call_observed: `True`
- require_tool_call_each_turn: `True`
- tool_call_each_required_turn: `True`
- require_tool_evidence: `True`
- tool_evidence_each_required_turn: `True`
- final_turn_tools_disabled: `True`
- final_turn_thinking_disabled: `True`
- visible_or_tool_output_each_turn: `True`
- visible_output_observed: `True`
- final_turn_visible_output: `True`
- no_loop_like_tail: `True`
- no_tool_markup_leak: `True`

## Rows

| turn | elapsed_s | cached_tokens | visible_chars | reasoning_chars | function_calls | warnings | loop_like | tool_markup_leak | tool_grounded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | 20.293 | 0 | 567 | 3753 | 1 | 0 | False | False | True |
| 2 | 32.526 | 0 | 559 | 6531 | 1 | 0 | False | False | True |
| 3 | 6.272 | 0 | 1166 | 0 | 0 | 0 | False | False | False |

Raw response, cache, health, and tail_review files are preserved next to this summary.
