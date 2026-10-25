"""DeepSeek-V4.1 DSML tool parser uses the selected V4.1 bundle's own encoder (CPU only).

The inherited DSV4 lookup searched the machine for a DSV4 ``encoding_dsv4.py``; when one existed its
V4 parser rejected V4.1's spaced DSML and that rejection is terminal, so every V4.1 call failed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vmlx_engine.tool_parsers import dsml_tool_parser as base
from vmlx_engine.tool_parsers.dsml_tool_parser import DSML_PREFIX
from vmlx_engine.tool_parsers.dsml_v41_tool_parser import DSMLV41ToolParser

BUNDLE = Path.home() / "models/DeepSeek-V4.1-Flash-1017-JANG-Q2Q3-IMATRIX"

CALL = (
    f"<{DSML_PREFIX} calls>\n"
    f'<{DSML_PREFIX} invoke name="get_weather">\n'
    f'<{DSML_PREFIX} parameter name="city" string="true">Paris</{DSML_PREFIX} parameter>\n'
    f"</{DSML_PREFIX} invoke>\n"
    f"</{DSML_PREFIX} calls>"
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}]


def _rejecting_dsv4_encoder(**_):
    def parse_message_from_completion_text(text, thinking_mode):
        raise ValueError("V4 grammar: no spaced DSML")

    return SimpleNamespace(parse_message_from_completion_text=parse_message_from_completion_text,
                           eos_token="<｜end▁of▁sentence｜>")


@pytest.mark.skipif(not (BUNDLE / "encoding" / "encoding.py").exists(), reason="V4.1 bundle not present")
def test_v41_call_parses_with_bundle_encoder_even_if_a_dsv4_encoder_exists(monkeypatch):
    import vmlx_engine.loaders.dsv4_chat_encoder as dsv4_enc

    monkeypatch.setattr(dsv4_enc, "_load_encoding_dsv4_module", _rejecting_dsv4_encoder)
    parser = DSMLV41ToolParser(tokenizer=None)
    request = {"model_path": str(BUNDLE), "tools": TOOLS}
    out = parser.extract_tool_calls(CALL, request=request)
    assert out.tools_called is True
    assert out.tool_calls[0]["name"] == "get_weather"
    assert json.loads(out.tool_calls[0]["arguments"]) == {"city": "Paris"}
    assert parser._canonical_encoding_module(request) is not None  # canonical V4.1 parser used


def test_v41_without_bundle_path_has_no_canonical_parser(monkeypatch, tmp_path):
    import vmlx_engine.loaders.dsv4_chat_encoder as dsv4_enc

    monkeypatch.setattr(dsv4_enc, "_load_encoding_dsv4_module", _rejecting_dsv4_encoder)
    parser = DSMLV41ToolParser(tokenizer=None)
    assert parser._canonical_encoding_module({"model_path": str(tmp_path)}) is None
    out = parser.extract_tool_calls(CALL, request={"model_path": str(tmp_path), "tools": TOOLS})
    assert out.tools_called is True  # strict grammar only; the DSV4 encoder is never consulted


def test_dsv4_parser_still_uses_dsv4_encoder(monkeypatch):
    import vmlx_engine.loaders.dsv4_chat_encoder as dsv4_enc

    sentinel = object()
    monkeypatch.setattr(dsv4_enc, "_load_encoding_dsv4_module", lambda **_: sentinel)
    assert base.DSMLToolParser(tokenizer=None)._canonical_encoding_module({"model_path": "/x"}) is sentinel
