"""DeepSeek-V4.1 DSML tool calls are detected by the server's streaming tool-markup gate.

V4.1's DSML tags carry a leading space (``<｜DSML｜ calls>``, ``<｜DSML｜ invoke``); the gate only listed the DSV4
forms, so a streamed V4.1 call never started tool buffering: the DSML streamed out as assistant content and the
end-of-stream parse never ran (served JANGHT bundle, 2026-10-26)."""
import pytest

server = pytest.importorskip("vmlx_engine.server")

CALL = ("\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n"
        "<｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>")


@pytest.mark.parametrize("cut", [len("\n\n<｜DSML｜ calls>"), len("\n\n<｜DSML｜ ca"), len(CALL)])
def test_v41_wrapper_starts_tool_buffering(cut):
    assert server._has_tool_marker_or_partial_suffix(CALL[:cut])


def test_v41_bare_invoke_detected():
    assert server._has_tool_marker_or_partial_suffix("ok <｜DSML｜ invoke name=\"f\">")


def test_prose_without_markup_is_not_buffered():
    assert not server._has_tool_marker_or_partial_suffix("The DSML format uses calls and invoke tags.")
