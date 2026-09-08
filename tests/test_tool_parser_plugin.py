"""--tool-parser-plugin: a plugin module registers (or replaces) tool parsers before resolution.

The extension point exists so a live proof can inject a controlled parser fault (a parser that
raises on a real native block) through the app-owned session's additional arguments, instead of
waiting for a natural parser crash to exercise the server's exception path (S23).
"""
from __future__ import annotations

import textwrap
from pathlib import Path


def test_plugin_file_registers_and_replaces_parsers(tmp_path: Path):
    from vmlx_engine.tool_parsers import ToolParserManager
    from vmlx_engine.tool_parsers.atem_tool_parser import AtemToolParser

    plugin = tmp_path / "fault_plugin.py"
    plugin.write_text(textwrap.dedent('''
        from vmlx_engine.tool_parsers import ToolParserManager
        from vmlx_engine.tool_parsers.atem_tool_parser import AtemToolParser

        class FaultyAtem(AtemToolParser):
            def extract_tool_calls(self, model_output, request=None):
                if "<atem:invoke" in model_output:
                    raise TypeError("injected parser fault")
                return super().extract_tool_calls(model_output, request=request)

        ToolParserManager.register_module("atem", FaultyAtem)
        ToolParserManager.register_module("atem_fault_plugin_test", FaultyAtem)
    '''))
    original = ToolParserManager.get_tool_parser("atem")
    try:
        names = ToolParserManager.load_plugin(str(plugin))
        assert "atem_fault_plugin_test" in names
        replaced = ToolParserManager.get_tool_parser("atem")
        assert replaced is not original and issubclass(replaced, AtemToolParser)
    finally:
        ToolParserManager.register_module("atem", original)
        ToolParserManager.tool_parsers.pop("atem_fault_plugin_test", None)
    assert ToolParserManager.get_tool_parser("atem") is original


def test_missing_plugin_is_an_error_not_a_silent_skip(tmp_path: Path):
    import pytest
    from vmlx_engine.tool_parsers import ToolParserManager

    with pytest.raises(ImportError):
        ToolParserManager.load_plugin(str(tmp_path / "nope.py"))


def test_serve_cli_flag_is_declared():
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "vmlx_engine" / "cli.py").read_text()
    assert '"--tool-parser-plugin"' in src
    assert re.search(r'action="append"', src[src.index('"--tool-parser-plugin"'):][:400])
    assert "_TPM.load_plugin(_plugin)" in src
