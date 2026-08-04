#!/usr/bin/env python3
"""Does the real DSV4 encoder output trigger a synthetic fallback injection?

Renders each agentic turn with the actual bundle encoder, then runs the
production fallback checker over it. Any injection means we would be
prepending a second tool contract on top of the encoder's own -- the
documented cause of the unbounded literal `response` loop.
"""
import os
import sys
from pathlib import Path

sys.path.insert(
    0,
    os.environ.get("VMLX_SRC", str(Path(__file__).resolve().parents[1])),
)

from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools  # noqa: E402
from vmlx_engine.loaders import dsv4_chat_encoder as E  # noqa: E402, N812

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv4_prefix_probe import MODEL_PATH, TOOLS, TURNS  # noqa: E402


class DSV4LikeTokenizer:
    # Set by load_jangtq_dsv4.py:1537 on the real production tokenizer.
    _vmlx_dsv4_chat_template_shim = True

    def apply_chat_template(self, messages, **_kwargs):
        rendered = []
        for message in messages:
            role = message.get("role")
            content = message.get("content") or ""
            if role == "system":
                rendered.append(content)
            elif role == "user":
                rendered.append(f"<｜User｜>{content}")
            elif role == "assistant":
                rendered.append(f"<｜Assistant｜>{content}")
        return "\n".join(rendered)


def main():
    print(f"bundle={MODEL_PATH}\n")
    bad = 0
    for i, msgs in enumerate(TURNS):
        prompt = E.apply_chat_template(
            msgs,
            enable_thinking=True,
            reasoning_effort="low",
            tools=TOOLS,
            add_generation_prompt=True,
            model_path=MODEL_PATH,
        )
        checked = check_and_inject_fallback_tools(
            prompt,
            msgs,
            TOOLS,
            DSV4LikeTokenizer(),
            {"tokenize": False, "add_generation_prompt": True, "tools": TOOLS},
            tool_parser_id="dsml",
        )
        if checked == prompt:
            print(f"turn {i}: OK — no fallback injected ({len(prompt)} chars)")
        else:
            bad += 1
            delta = len(checked) - len(prompt)
            print(f"turn {i}: *** FALLBACK INJECTED *** (+{delta} chars)")
            head = checked[: max(0, len(checked) - len(prompt)) + 200]
            print(f"    injected head: {head[:400]!r}")
    print(f"\nresult: {bad} of {len(TURNS)} turns would get a duplicate contract")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
