"""Native tool prompts are accepted only when the rendered schemas match the request.

The fallback-injection gate used to accept a native ``<tools>`` block as soon as
it carried every requested tool NAME. A template (or a stale/cached render) that
names the tools but omits, renames or retypes their parameters therefore passed
as "native" and the model was never told that ``read_file`` needs ``path`` or
``terminal`` needs ``command`` (audit CODEX-CROSS-FAMILY-TOOL-SCHEMA-CACHE,
mechanism B, reproduced with the MiniMax M2.7 fixture).

These fixtures render the real bundle templates with tools whose names match
the request but whose parameter definitions differ, and require the fallback
injection; the exact request, a description-only difference and a re-ordered
property set stay accepted unchanged (semantic comparison, not byte equality).
"""

import copy
import json
import logging
import re
from pathlib import Path

import jinja2
import pytest
from jinja2 import sandbox

from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

FIXTURES = Path(__file__).parent / "fixtures"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repo-relative path"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run a command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_s": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    },
]

ROWS = (
    (
        "minimax_m27_chat_template.jinja",
        "minimax",
        "MiniMax M2.7: <tool>{function}</tool> entries",
    ),
    (
        "qwen38_27b_d_chat_template.jinja",
        "qwen3_coder",
        "Qwen3.8-27B D-series: {type,function} entries",
    ),
    (
        "nanbeige42_chat_template.jinja",
        "xml_function",
        "Nanbeige 4.2: {type,function} entries",
    ),
)


class _FixtureTokenizer:
    """Renders a checked-in bundle template the way transformers does."""

    def __init__(self, fixture: str) -> None:
        env = sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )
        env.filters["tojson"] = lambda v, ensure_ascii=False, **kw: json.dumps(
            v, ensure_ascii=ensure_ascii, **kw
        )
        env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(
            jinja2.TemplateError(msg)
        )
        self._template = env.from_string((FIXTURES / fixture).read_text())

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=True, tokenize=False, **kwargs
    ):
        return self._template.render(
            messages=messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )


def _render_with(tok, rendered_tools):
    messages = [{"role": "user", "content": "Inspect the repository."}]
    return messages, tok.apply_chat_template(messages, tools=rendered_tools)


def _gate(tok, messages, prompt, parser, caplog):
    with caplog.at_level(logging.WARNING, logger="vmlx_engine.api.tool_calling"):
        return check_and_inject_fallback_tools(
            prompt, messages, TOOLS, tok, {"tools": TOOLS}, tool_parser_id=parser
        )


def _mutated(mutate):
    tools = copy.deepcopy(TOOLS)
    mutate(tools)
    return tools


def _drop_all_parameters(tools):
    for t in tools:
        t["function"]["parameters"] = {"type": "object", "properties": {}}


def _drop_required_list(tools):
    tools[0]["function"]["parameters"].pop("required")


def _rename_parameter(tools):
    params = tools[1]["function"]["parameters"]
    params["properties"] = {
        "cmd": params["properties"]["command"],
        "timeout_s": params["properties"]["timeout_s"],
    }
    params["required"] = ["cmd"]


def _retype_parameter(tools):
    tools[1]["function"]["parameters"]["properties"]["timeout_s"]["type"] = "string"


def _drop_optional_property(tools):
    tools[1]["function"]["parameters"]["properties"].pop("timeout_s")


DEFICIENT = (
    ("all parameters dropped", _drop_all_parameters),
    ("required list dropped", _drop_required_list),
    ("required parameter renamed", _rename_parameter),
    ("parameter retyped", _retype_parameter),
    ("optional parameter missing", _drop_optional_property),
)


@pytest.mark.parametrize("fixture, parser, label", ROWS, ids=[r[0] for r in ROWS])
@pytest.mark.parametrize("case, mutate", DEFICIENT, ids=[d[0] for d in DEFICIENT])
def test_names_present_but_schema_deficient_gets_fallback(
    fixture, parser, label, case, mutate, caplog
):
    tok = _FixtureTokenizer(fixture)
    messages, prompt = _render_with(tok, _mutated(mutate))
    assert "<tools>" in prompt and "read_file" in prompt and "terminal" in prompt, label
    out = _gate(tok, messages, prompt, parser, caplog)
    assert (
        out != prompt
    ), f"{label}: {case} — names-only recognition accepted a deficient schema"
    # Each dialect renders its own fallback shape (JSON schema, `fields:` lists,
    # native exemplars); the request's parameter names must reach the model.
    assert re.search(r"\bpath\b", out) and re.search(
        r"\bcommand\b", out
    ), f"{label}: {case} — fallback must carry the request's parameters"


@pytest.mark.parametrize("fixture, parser, label", ROWS, ids=[r[0] for r in ROWS])
def test_exact_schema_is_accepted_unchanged(fixture, parser, label, caplog):
    tok = _FixtureTokenizer(fixture)
    messages, prompt = _render_with(tok, copy.deepcopy(TOOLS))
    out = _gate(tok, messages, prompt, parser, caplog)
    assert out == prompt, f"{label}: exact native render must not be re-rendered"
    assert "needs fallback tool schema injection" not in caplog.text


def _description_only_difference(tools):
    tools[0]["function"]["description"] = "Read one file from the workspace"
    tools[0]["function"]["parameters"]["properties"]["path"]["description"] = "A path"


def _reordered_properties(tools):
    params = tools[1]["function"]["parameters"]
    props = params["properties"]
    params["properties"] = {
        "timeout_s": props["timeout_s"],
        "command": props["command"],
    }


@pytest.mark.parametrize("fixture, parser, label", ROWS, ids=[r[0] for r in ROWS])
@pytest.mark.parametrize(
    "case, mutate",
    (
        ("description-only difference", _description_only_difference),
        ("re-ordered properties", _reordered_properties),
    ),
    ids=["description-only", "reordered"],
)
def test_semantically_equal_schema_is_accepted(
    fixture, parser, label, case, mutate, caplog
):
    # The comparison is over parameter names, required-ness and types — a
    # template that renders the same contract with different prose or key
    # order is native, not deficient (no byte-equality demand, no forced
    # fallback that would move the SSD prefix on every turn).
    tok = _FixtureTokenizer(fixture)
    messages, prompt = _render_with(tok, _mutated(mutate))
    out = _gate(tok, messages, prompt, parser, caplog)
    assert out == prompt, f"{label}: {case} must stay accepted"


def test_unrelated_extra_tool_in_render_is_not_a_match():
    # A render that carries the requested names plus a foreign tool is a
    # different catalog (stale cache / other request); the request's own
    # contract still has to be re-rendered.
    tok = _FixtureTokenizer("minimax_m27_chat_template.jinja")
    extra = copy.deepcopy(TOOLS) + [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            },
        }
    ]
    messages, prompt = _render_with(tok, extra)
    out = check_and_inject_fallback_tools(
        prompt, messages, TOOLS, tok, {"tools": TOOLS}, tool_parser_id="minimax"
    )
    assert out != prompt
