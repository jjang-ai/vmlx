"""DSML tool-call parser for DeepSeek-V4.1-Flash.

V4.1 uses the same DSML control protocol as DeepSeek-V4 but with LEADING-SPACE tag
names (per the bundle's encoding/encoding.py):

    <｜DSML｜ calls>
    <｜DSML｜ invoke name="fn">
    <｜DSML｜ parameter name="p1" string="true">str_val</｜DSML｜ parameter>
    <｜DSML｜ parameter name="p2" string="false">42</｜DSML｜ parameter>
    </｜DSML｜ invoke>
    </｜DSML｜ calls>

(DSV4 omits the space: ``<｜DSML｜invoke>`` / ``<｜DSML｜tool_calls>``.) Everything else
— the strict-native handling, schema gating, streaming suppression, invalid-markup
safety — is inherited unchanged from :class:`DSMLToolParser`; only the tag grammar
differs, so this subclass just overrides the markers/regexes with the spaced forms.
"""

from __future__ import annotations

import re

from .dsml_tool_parser import DSML_PREFIX, DSMLToolParser
from . import ToolParserManager


@ToolParserManager.register_module(["deepseek_v41", "dsml_v41"])
class DSMLV41ToolParser(DSMLToolParser):
    """DeepSeek-V4.1 DSML parser: identical to V4 but with leading-space tag names."""

    # block wrapper is " calls" (not "tool_calls"); invoke/parameter carry a leading space
    TOOL_CALLS_OPEN = f"<{DSML_PREFIX} calls>"
    TOOL_CALLS_CLOSE = f"</{DSML_PREFIX} calls>"
    INVOKE_OPEN_PREFIX = f"<{DSML_PREFIX} invoke "
    INVOKE_CLOSE = f"</{DSML_PREFIX} invoke>"
    DSML_OPEN_PREFIX = f"<{DSML_PREFIX}"

    NATIVE_MARKERS = (
        f"<{DSML_PREFIX} calls>",
        f"<{DSML_PREFIX} invoke",
    )

    _INVOKE_RE = re.compile(
        rf'<{re.escape(DSML_PREFIX)}\s+invoke\s+name="([^"]+)"\s*>(.*?)</{re.escape(DSML_PREFIX)}\s+invoke>',
        re.DOTALL,
    )
    _PARAM_RE = re.compile(
        rf'<{re.escape(DSML_PREFIX)}\s+parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)</{re.escape(DSML_PREFIX)}\s+parameter>',
        re.DOTALL,
    )
    _WRAPPER_RESIDUE_RE = re.compile(
        rf'</?{re.escape(DSML_PREFIX)}\s+(?:calls?|invoke|parameter)[^>]*>?',
        re.DOTALL,
    )

    _NONCANONICAL_PROTOCOL_MARKERS = (
        DSML_OPEN_PREFIX,
        f"</{DSML_PREFIX}",
        "<invoke_",
        "<invoke ",
    )
    _PARTIAL_PROTOCOL_MARKERS = (
        DSML_OPEN_PREFIX,
        f"</{DSML_PREFIX}",
    )
