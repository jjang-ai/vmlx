import json
from pathlib import Path
from typing import Any


def _iter_role_tokens(values: Any):
    if values is None:
        return
    if isinstance(values, str):
        if values:
            yield values
        return
    if isinstance(values, (list, tuple, set)):
        for v in values:
            yield from _iter_role_tokens(v)
    elif isinstance(values, dict):
        for v in values.values():
            yield from _iter_role_tokens(v)


def _resolve_token_to_id(
    tokenizer: Any,
    token: str,
    rust_tokenizer: Any,
    unk_id: int | None,
    unk_token: str | None,
) -> int | None:
    tid: int | None = None
    try:
        tid = int(token)
    except (TypeError, ValueError):
        tid = tokenizer.convert_tokens_to_ids(token) if hasattr(tokenizer, "convert_tokens_to_ids") else None

    if tid is None or tid < 0:
        return None

    if unk_id is not None and tid == unk_id and token != unk_token:
        if rust_tokenizer is not None:
            try:
                rust_tid = rust_tokenizer.token_to_id(token)
                if isinstance(rust_tid, int) and rust_tid >= 0:
                    return rust_tid
            except Exception:
                pass
        return None
    return tid


def collect_multi_eos_ids(
    tokenizer: Any,
    model_name: str,
    *,
    registry_eos_tokens: list[str] | None,
    reasoning_parser: str | None,
    use_rust_tokenizer: bool,
) -> tuple[list[int], list[str]]:
    """
    Build a model-specific stop token ID set for streaming generation.

    Returns:
        (resolved, unresolved)
    """

    resolved: list[int] = []
    unresolved: list[str] = []

    primary = getattr(tokenizer, "eos_token_id", None)
    if isinstance(primary, int):
        resolved.append(primary)

    model_path = Path(model_name)

    try:
        gen_cfg_path = model_path / "generation_config.json"
        if gen_cfg_path.is_file():
            gen_cfg = json.loads(gen_cfg_path.read_text())
            gen_eos = gen_cfg.get("eos_token_id")
            if isinstance(gen_eos, int):
                if gen_eos not in resolved:
                    resolved.append(gen_eos)
            elif isinstance(gen_eos, list):
                for tid in gen_eos:
                    if isinstance(tid, int) and tid not in resolved:
                        resolved.append(tid)
    except Exception:
        pass

    rust_tok = None
    if use_rust_tokenizer:
        try:
            from tokenizers import Tokenizer as _RustTokenizer

            tj = model_path / "tokenizer.json"
            if tj.is_file():
                rust_tok = _RustTokenizer.from_file(str(tj))
        except Exception:
            rust_tok = None

    inner = tokenizer
    if hasattr(tokenizer, "_tokenizer"):
        inner = tokenizer._tokenizer

    unk_id = tokenizer.unk_token_id if hasattr(tokenizer, "unk_token_id") else None
    unk_token = tokenizer.unk_token if hasattr(tokenizer, "unk_token") else None
    if hasattr(inner, "unk_token_id"):
        unk_id = inner.unk_token_id
    if hasattr(inner, "unk_token"):
        unk_token = inner.unk_token

    if registry_eos_tokens:
        for tok in registry_eos_tokens:
            if tok is None:
                continue
            tid = _resolve_token_to_id(inner, str(tok), rust_tok, unk_id, unk_token)
            if tid is None:
                unresolved.append(str(tok))
                continue
            if tid not in resolved:
                resolved.append(tid)

    if reasoning_parser == "deepseek_r1":
        role_tokens: list[str] = []
        tokenizer_cfg_path = model_path / "tokenizer_config.json"
        if tokenizer_cfg_path.is_file():
            try:
                chat_cfg = json.loads(tokenizer_cfg_path.read_text()).get("chat", {})
                role_tokens = list(_iter_role_tokens(chat_cfg.get("role_tokens"))) if isinstance(chat_cfg, dict) else []
            except Exception:
                role_tokens = []

        for tok in role_tokens:
            if not isinstance(tok, str):
                continue
            if not tok:
                continue
            tid = _resolve_token_to_id(inner, tok, rust_tok, unk_id, unk_token)
            if tid is None:
                unresolved.append(tok)
                continue
            if tid not in resolved:
                resolved.append(tid)

    return resolved, unresolved
