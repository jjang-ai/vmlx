# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible logprobs helpers.

The engine carries token-id based logprob records. The API layer owns token
decoding because tokenizer implementations differ across text/chat families.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def format_token_logprobs_for_output(
    token_id: int,
    logprobs: Any,
    top_k: int,
) -> dict | None:
    """Convert one MLX logprob vector into a compact Python record."""
    if logprobs is None:
        return None
    try:
        import mlx.core as mx

        vec = logprobs
        if hasattr(vec, "ndim") and vec.ndim > 1:
            vec = mx.squeeze(vec)
        token_id = int(token_id)
        vocab = int(vec.shape[-1])
        if token_id < 0 or token_id >= vocab:
            return None
        sampled_lp = float(vec[token_id].item())
        top_entries: list[tuple[int, float]] = []
        k = max(0, int(top_k or 0))
        if k > 0:
            k = min(k, vocab)
            values = mx.topk(vec, k)
            mx.eval(values)
            top_values = [float(lp) for lp in values.tolist()]
            value_to_ids: dict[float, list[int]] = {}
            for idx, lp in enumerate(vec.tolist()):
                value_to_ids.setdefault(float(lp), []).append(idx)
            used: set[int] = set()
            pairs = []
            for lp in top_values:
                for idx in value_to_ids.get(lp, []):
                    if idx not in used:
                        used.add(idx)
                        pairs.append((idx, lp))
                        break
            pairs.sort(key=lambda item: item[1], reverse=True)
            top_entries = pairs
        return {
            "token_id": token_id,
            "logprob": sampled_lp,
            "top_logprobs": top_entries,
        }
    except Exception as exc:
        logger.debug("Failed to format token logprobs: %s", exc)
        return None


def decode_token(tokenizer: Any, token_id: int) -> str:
    """Decode a single token id without assuming a tokenizer backend."""
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        try:
            return tokenizer.decode(int(token_id))
        except Exception:
            return str(token_id)


def utf8_bytes(text: str) -> list[int]:
    """Return OpenAI-style UTF-8 bytes for a decoded token."""
    return list(str(text).encode("utf-8"))


def format_completion_logprobs(raw_logprobs: list[dict] | None, tokenizer: Any) -> dict | None:
    """Format legacy /v1/completions logprobs."""
    if not raw_logprobs:
        return None
    tokens: list[str] = []
    token_logprobs: list[float] = []
    top_logprobs: list[dict[str, float]] = []
    text_offset: list[int] = []
    offset = 0
    for entry in raw_logprobs:
        token = decode_token(tokenizer, int(entry["token_id"]))
        tokens.append(token)
        token_logprobs.append(float(entry["logprob"]))
        text_offset.append(offset)
        offset += len(token)
        top = {}
        for tid, lp in entry.get("top_logprobs") or []:
            top[decode_token(tokenizer, int(tid))] = float(lp)
        top_logprobs.append(top)
    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": text_offset,
    }


def format_chat_logprobs(raw_logprobs: list[dict] | None, tokenizer: Any) -> dict | None:
    """Format Chat Completions logprobs.content entries."""
    if not raw_logprobs:
        return None
    content = []
    for entry in raw_logprobs:
        token = decode_token(tokenizer, int(entry["token_id"]))
        top_entries = []
        for tid, lp in entry.get("top_logprobs") or []:
            top_token = decode_token(tokenizer, int(tid))
            top_entries.append(
                {
                    "token": top_token,
                    "logprob": float(lp),
                    "bytes": utf8_bytes(top_token),
                }
            )
        content.append(
            {
                "token": token,
                "logprob": float(entry["logprob"]),
                "bytes": utf8_bytes(token),
                "top_logprobs": top_entries,
            }
        )
    return {"content": content}
