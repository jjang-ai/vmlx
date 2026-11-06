# SPDX-License-Identifier: Apache-2.0
"""Streamed reasoning keeps the separator newlines; a template that adds them itself must not render them twice."""
from types import SimpleNamespace

import jinja2
import pytest

from vmlx_engine import server

# Ripple-65B-Preview's template (assistant branch): reasoning wrapped as-is between "<think>\n" and "\n</think>\n".
RIPPLE = (
    "{%- for message in messages %}{%- set content = message.content or '' %}"
    "{%- if message.role == 'assistant' and message.reasoning_content %}"
    "{%- set content = '<think>\\n' + message.reasoning_content + '\\n</think>\\n' + content %}{%- endif %}"
    "{{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)
# A Qwen3-style template that trims the reasoning itself.
TRIMMING = RIPPLE.replace("message.reasoning_content +", "message.reasoning_content.strip('\\n') +")


class _Tok:
    def __init__(self, template):
        self.chat_template = template
        self._t = jinja2.Environment().from_string(template)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **_):
        return self._t.render(messages=messages, add_generation_prompt=add_generation_prompt)


@pytest.fixture(autouse=True)
def _clear_probe():
    server._REASONING_SEPARATOR_PROBE.clear()
    yield
    server._REASONING_SEPARATOR_PROBE.clear()


def _history(rc):
    return [{"role": "user", "content": "q"}, {"role": "assistant", "content": "A", "reasoning_content": rc},
            {"role": "user", "content": "next"}]


def test_streamed_history_renders_the_generated_tokens(monkeypatch):
    tok = _Tok(RIPPLE)
    monkeypatch.setattr(server, "_engine", SimpleNamespace(tokenizer=tok))
    generated = tok.apply_chat_template(_history("think it through"))           # what the model produced
    streamed = server._normalize_template_reasoning_separators(_history("\nthink it through\n"))
    assert tok.apply_chat_template(streamed) == generated
    # non-streamed reasoning (already without separators) is unchanged; inner newlines are never touched
    assert server._normalize_template_reasoning_separators(_history("a\n\nb")) == _history("a\n\nb")
    # exactly one newline each side: a blank line the model wrote stays
    assert server._normalize_template_reasoning_separators(_history("\n\nx\n\n"))[1]["reasoning_content"] == "\nx\n"


def test_templates_that_trim_reasoning_are_left_alone(monkeypatch):
    monkeypatch.setattr(server, "_engine", SimpleNamespace(tokenizer=_Tok(TRIMMING)))
    assert server._normalize_template_reasoning_separators(_history("\nx\n")) == _history("\nx\n")
    monkeypatch.setattr(server, "_engine", None)
    server._REASONING_SEPARATOR_PROBE.clear()
    assert server._normalize_template_reasoning_separators(_history("\nx\n")) == _history("\nx\n")
