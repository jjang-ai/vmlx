# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V4 chat encoder adapter.

DSV4-Flash ships its own ``encoding_dsv4.py`` (tokenizer + chat template
helper) alongside the model bundle. Importing it through a stable path so
server.py can encode chat messages the way the model expects — without
needing every caller to locate the source-model directory and ``sys.path``
hack.

Three reasoning modes (research/DSV4-RUNTIME-ARCHITECTURE.md §4):

  +------------------+---------------------+---------------------+
  |  API input       |  DSV4 encoding call |  Prompt suffix      |
  +------------------+---------------------+---------------------+
  |  enable_thinking |  thinking_mode=     |  ends with          |
  |    = False       |  "chat"             |  "</think>"         |
  +------------------+---------------------+---------------------+
  |  enable_thinking |  thinking_mode=     |  no </think>; model |
  |    = True +      |  "thinking" +       |  opens <think>…</   |
  |  reasoning_      |  reasoning_effort=  |  think> then answer |
  |    effort="high" |  "high"             |                     |
  +------------------+---------------------+---------------------+
  |  reasoning_      |  thinking_mode=     |  extra system hint  |
  |    effort="max"  |  "thinking" +       |  normalized to the  |
  |                  |  reasoning_effort=  |                     |
  |                  |  "high"             |  stable rail        |
  +------------------+---------------------+---------------------+

Multi-turn: ``drop_earlier_reasoning=True`` (default) — DSV4 encoder
strips prior ``<think>…</think>`` blocks from history when building the
next prompt. Honors ``jang_config.chat.reasoning.drop_earlier_reasoning``.

Sampling defaults pulled from ``jang_config.chat.sampling_defaults``
(typically ``temperature=0.6``, ``top_p=0.95``) when the caller doesn't
override them. Our chat.ts layer already reads ``generation_config.json``
on new chat creation (chat.ts:441-469) — the jang_config.chat defaults
layer on top as JANG-stamp capabilities.

Tool calls are DSML format (``vmlx_engine/tool_parsers/dsml_tool_parser.py``
— parser key "dsml"). ``jang_config.chat.tool_calling.parser`` = "dsml".

Long-context mode (``DSV4_LONG_CTX``):

  * ``1`` is the supported runtime mode. ``Model.make_cache()`` returns
    ``DeepseekV4Cache`` on ``compress_ratio>0`` layers (CSA/HSA + SWA
    composite) and plain ``KVCache`` on local-only layers.
  * Paged prefix cache uses a dedicated ``deepseek_v4`` block record with
    ``deepseek_v4_v7`` metadata; v7 keys DSV4 prompt cache blocks at N-1
    tokens so the last prompt token is re-fed on prefix hits. The loader
    installs the prefill mask-trim patch required for prompts beyond the
    sliding window.
  * ``cache_salt`` / ``skip_prefix_cache`` still bypass all cache layers for
    benchmarks, but DSV4 is no longer force-bypassed by family.

Research refs: DSV4-RUNTIME-ARCHITECTURE.md §17,
DSV-EXHAUSTIVE-VARIABLES-GUIDE.md §12.
"""

from __future__ import annotations

import importlib.util
import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _default_encoding_dirs() -> List[Path]:
    """Return likely local DSV4 encoding directories.

    Production callers should pass ``model_path``. This fallback covers app
    sessions/tests where an older persisted DSV4 path no longer exists but the
    user has the bundle or source checkout in the standard local model roots.
    Keep the search shallow so startup does not walk large model trees.
    """
    dirs: List[Path] = []

    def add(path: Path) -> None:
        if path not in dirs:
            dirs.append(path)

    roots = [Path.home() / "models"]
    # Accept either the canonical name or the historical typo'd one.
    extra_root = (
        os.environ.get("VMLX_MODELS_DIR")
        or os.environ.get("VLLM_MODELS_DIR")
        or os.environ.get("VMLINUX_MODELS_DIR")
    )
    if extra_root:
        roots.insert(0, Path(extra_root).expanduser())
    volumes = Path("/Volumes")
    if volumes.exists():
        try:
            roots.extend(p for p in volumes.iterdir() if p.is_dir())
        except Exception:
            pass

    for root in roots:
        try:
            if not root.exists():
                continue
            add(root / "Sources" / "DeepSeek-V4-Flash" / "encoding")
            for pattern in (
                "DeepSeek-V4-Flash*/encoding",
                "JANGQ/DeepSeek-V4-Flash*/encoding",
                "*/DeepSeek-V4-Flash*/encoding",
                "*/*DeepSeek-V4-Flash*/encoding",
            ):
                for match in root.glob(pattern):
                    add(match)
        except Exception:
            continue
    return dirs


def _load_encoding_dsv4_module(
    encoding_dir: Optional[Path] = None,
    model_path: Optional[Path] = None,
):
    """Locate + dynamically import the source model's ``encoding_dsv4.py``.

    Lookup order:
      1. Explicit ``encoding_dir`` argument.
      2. ``DSV4_ENCODING_DIR`` environment variable (session dispatcher
         may set this at session-start).
      3. ``{model_path}/encoding/`` subdir — the standard bundle layout
         as shipped in DeepSeek-V4-Flash-JANGTQ / JANG_2L. This path is
         auto-discovered so no env plumbing is required.
      4. Bundled fallback at ``jang_tools.dsv4.encoding_adapter``.
    """
    candidates: List[Path] = []
    if encoding_dir is not None:
        candidates.append(Path(encoding_dir))
    env = os.environ.get("DSV4_ENCODING_DIR")
    if env:
        candidates.append(Path(env))
    if model_path is not None:
        mp = Path(model_path)
        candidates.append(mp / "encoding")
        candidates.append(mp)
    candidates.extend(_default_encoding_dirs())

    for d in candidates:
        f = d / "encoding_dsv4.py"
        if f.exists():
            spec = importlib.util.spec_from_file_location("encoding_dsv4", str(f))
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules["encoding_dsv4"] = mod
            spec.loader.exec_module(mod)
            logger.info("DSV4 encoding loaded from %s", f)
            return mod

    # Fall back to the jang_tools adapter which will raise a clear
    # FileNotFoundError if neither an explicit path nor DSV4_ENCODING_DIR is set.
    try:
        from jang_tools.dsv4.encoding_adapter import _load_encoding_module
    except ImportError as e:
        raise RuntimeError(
            "DSV4 chat encoder unavailable: jang_tools.dsv4 not installed "
            "AND no encoding_dsv4.py found via DSV4_ENCODING_DIR or "
            "{model_path}/encoding/. Bring over a DSV4 bundle or set "
            "DSV4_ENCODING_DIR."
        ) from e
    return _load_encoding_module(encoding_dir)


# Per-model-path cache — different bundles may ship different encoder
# revisions (DSV4-Flash vs DSV4-Pro), so key by resolved absolute path.
_encoding_cache: dict[str, Any] = {}


def _get_encoding(model_path: Optional[Path] = None):
    key = str(Path(model_path).resolve()) if model_path else "<default>"
    if key not in _encoding_cache:
        _encoding_cache[key] = _load_encoding_dsv4_module(model_path=model_path)
    return _encoding_cache[key]


def _resolve_mode_and_effort(
    enable_thinking: Optional[bool],
    reasoning_effort: Optional[str],
) -> tuple[str, Optional[str]]:
    """Map vmlx chat API semantics → DSV4 encoder kwargs.

    **Strict DSV4 encoder contract** (verified against
    ``encoding_dsv4.py:261``): the encoder ``assert``s
    ``reasoning_effort in {None, "high", "max"}``. ``"low"`` and
    ``"medium"`` MUST be normalised to ``"high"`` before we call it, or
    the encoder raises ``AssertionError`` mid-request.

    API contract (shared with other reasoning models like Mistral 4):
      - enable_thinking False  → thinking suppressed. DSV4 calls this
        "chat" mode; the encoder appends </think> to the prompt to tell
        the model "skip thinking, go straight to answer".
      - enable_thinking True + reasoning_effort in (None, "low", "medium",
        "high") → "thinking" mode with reasoning_effort="high" (DSV4
        only distinguishes high vs max below/above).
      - reasoning_effort == "max" → "thinking" mode on the stable high
        effort rail. Live engine probes showed the raw DSV4 max rail can
        length-cap without closing </think>, so server capabilities do not
        advertise max for this family.

    Returns (thinking_mode, reasoning_effort). ``reasoning_effort`` is
    always one of ``{None, "high"}`` for production use — safe for direct
    passthrough to the encoder.
    """
    # Explicit max implies thinking, but normalizes to the stable rail.
    if reasoning_effort == "max":
        return "thinking", "high"

    if enable_thinking is False:
        return "chat", None

    if enable_thinking is True:
        # Pass "high" when user asked for any non-None low/medium/high
        # tier; otherwise None (plain thinking without effort modifier).
        effort = "high" if reasoning_effort in ("low", "medium", "high") else None
        return "thinking", effort

    if reasoning_effort in ("low", "medium", "high"):
        # Effort without enable_thinking implies thinking mode at "high".
        return "thinking", "high"

    # Default when both fields omitted: fall through to "chat" (instruct),
    # matching the conservative default the rest of the engine uses.
    return "chat", None


def apply_chat_template(
    messages: List[Dict[str, Any]],
    *,
    enable_thinking: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
    drop_earlier_reasoning: bool = True,
    tools: Optional[List[Dict[str, Any]]] = None,
    add_default_bos_token: bool = True,
    context: Optional[List[Dict[str, Any]]] = None,
    model_path: Optional[str] = None,
) -> str:
    """Encode chat messages into a DSV4 prompt string.

    Args:
        messages: OpenAI-format ``[{"role": ..., "content": ...}, ...]``.
            Tools, when used, are declared on the system/developer message
            via its ``tools`` field — DSV4's encoder reads them from there.
        enable_thinking: Whether to allow the model to emit a ``<think>``
            block. None ≡ False (default chat mode).
        reasoning_effort: ``"low" | "medium" | "high" | "max" | None``.
            See ``_resolve_mode_and_effort`` for mapping semantics.
        drop_earlier_reasoning: Multi-turn rule — strip prior
            ``<think>...</think>`` blocks from history. Follows the
            ``jang_config.chat.reasoning.drop_earlier_reasoning`` flag
            when caller doesn't override.
        tools: Optional OpenAI-format tool list. Injected onto the first
            system message's ``tools`` field if absent.
        add_default_bos_token: Whether to prepend ``<｜begin▁of▁sentence｜>``.
            DSV4 prompts need this; turn off only when the caller already
            prepended their own BOS.
        context: Optional cross-turn context injection (DSV4 encoder kwarg;
            pass through unchanged for advanced users).

    Returns:
        The encoded prompt string. Pass directly to ``mlx_lm.generate``
        or the vmlx generation loop.
    """
    thinking_mode, effort = _resolve_mode_and_effort(enable_thinking, reasoning_effort)
    messages = copy.deepcopy(messages)

    # DSV4's bundled encoder predates the strict-template normalization used
    # by the Responses adapter. It expects OpenAI tool-call
    # `function.arguments` to be a JSON string and calls json.loads() itself;
    # if we pass a dict it wraps the whole dict under a single DSML
    # `arguments` parameter. That makes tool-history continuation and DSML
    # exemplars materially wrong for DSV4 while the same dict form is correct
    # for Qwen/Mistral/Llama templates. Normalize only in this adapter.
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), dict):
                fn["arguments"] = json.dumps(fn["arguments"], ensure_ascii=False)

    # If the caller passed a `tools` list without a system message carrying
    # it, inject onto the first message. DSV4 encoder conventions require
    # `tools` to live on a system/developer message — see test_chat.py in
    # jang_tools.dsv4.
    if tools and messages:
        # Ensure there's a system message first; tools go on it.
        has_system_with_tools = any(
            m.get("role") == "system" and m.get("tools") for m in messages
        )
        if not has_system_with_tools:
            msgs_out: List[Dict[str, Any]] = []
            if messages and messages[0].get("role") == "system":
                sys_msg = dict(messages[0])
                sys_msg["tools"] = tools
                msgs_out.append(sys_msg)
                msgs_out.extend(messages[1:])
            else:
                msgs_out.append({"role": "system", "content": "", "tools": tools})
                msgs_out.extend(messages)
            messages = msgs_out

    enc = _get_encoding(model_path=Path(model_path) if model_path else None)
    return enc.encode_messages(
        messages,
        thinking_mode=thinking_mode,
        context=context,
        drop_thinking=drop_earlier_reasoning,
        add_default_bos_token=add_default_bos_token,
        reasoning_effort=effort,
    )


def parse_completion(
    raw_text: str,
    *,
    enable_thinking: Optional[bool] = None,
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse a raw DSV4 completion into ``{reasoning_content, content, tool_calls}``.

    Mirror of the server-side streaming parser — exposed here for batch
    use-cases (tests, benchmarks, non-streaming completion endpoints).
    ``enable_thinking`` determines whether to look for ``<think>`` blocks
    in the raw text.
    """
    thinking_mode = "thinking" if enable_thinking else "chat"
    try:
        enc = _get_encoding(model_path=Path(model_path) if model_path else None)
        return enc.parse_message_from_completion_text(raw_text, thinking_mode=thinking_mode)
    except Exception as e:  # pragma: no cover - defensive fallback
        logger.warning("DSV4 parse_completion failed (%s); returning raw text.", e)
        return {
            "role": "assistant",
            "reasoning_content": "",
            "content": raw_text,
            "tool_calls": [],
        }


def read_chat_config_from_bundle(bundle_path: str) -> Dict[str, Any]:
    """Pull the ``chat`` block from ``jang_config.json`` if present.

    Exposed for the session dispatcher to seed chat defaults (EOS token,
    sampling_defaults temperature/top_p, reasoning modes, tool parser).
    Returns an empty dict when no jang_config exists or the chat block is
    missing — every caller must handle that gracefully.
    """
    try:
        p = Path(bundle_path) / "jang_config.json"
        if not p.exists():
            return {}
        with open(p) as f:
            cfg = json.load(f)
        return cfg.get("chat", {}) or {}
    except Exception as e:
        logger.debug("jang_config.chat read failed for %s: %s", bundle_path, e)
        return {}
