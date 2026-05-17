#!/usr/bin/env python3
"""Compare Qwen native-MTP chunked verifier logits against sequential AR.

This is a diagnostic for the D3 verifier/cache contract. It loads the target
model once, drafts tokens through the model's own MTP head, then compares:

- sequential forced target forwards over `next_main, d1, d2, d3`;
- one chunked target forward over `[next_main, d1, d2, d3]` with
  `n_confirmed=1`.

If the chunked verifier is correct, each row's logits should match the
corresponding sequential forced-token logits closely enough for identical
greedy argmaxes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _logits_hidden(output: Any):
    if isinstance(output, tuple):
        return output
    if hasattr(output, "logits") and hasattr(output, "hidden_states"):
        return output.logits, output.hidden_states
    raise RuntimeError(f"model output lacks logits/hidden_states: {type(output)!r}")


def _argmax_token(logits_row):
    import mlx.core as mx

    tok = mx.argmax(logits_row, axis=-1).astype(mx.uint32)
    mx.eval(tok)
    return tok


def _token_int(token) -> int:
    return int(token.reshape(-1).tolist()[0])


def _row_diff(a, b) -> float:
    import mlx.core as mx

    diff = mx.max(mx.abs(a - b))
    mx.eval(diff)
    return float(diff.item())


def _to_numpy_row(row: Any) -> np.ndarray:
    import mlx.core as mx

    mx.eval(row)
    return np.asarray(row.reshape(-1).tolist(), dtype=np.float64)


def _research_settings_grid():
    from vmlx_engine.native_mtp_research import ResearchSamplerSettings

    target_temps = [0.2, 0.6, 0.8]
    temp_deltas = [0.0, 0.1, 0.2]
    min_ps = [0.0, 0.01, 0.03, 0.05, 0.08, 0.10]
    top_ps = [0.90, 0.95, 1.0]
    top_ks = [0, 20, 40]
    for target_temp in target_temps:
        for delta in temp_deltas:
            draft_temp = round(target_temp + delta, 3)
            for min_p in min_ps:
                for top_p in top_ps:
                    for top_k in top_ks:
                        yield (
                            ResearchSamplerSettings(
                                temperature=target_temp,
                                top_p=top_p,
                                min_p=min_p,
                                top_k=top_k,
                            ),
                            ResearchSamplerSettings(
                                temperature=draft_temp,
                                top_p=top_p,
                                min_p=min_p,
                                top_k=top_k,
                            ),
                        )


def _settings_key(target_settings: Any, draft_settings: Any) -> str:
    payload = {
        "target": {
            "temperature": target_settings.temperature,
            "top_p": target_settings.top_p,
            "min_p": target_settings.min_p,
            "min_tokens_to_keep": target_settings.min_tokens_to_keep,
            "top_k": target_settings.top_k,
        },
        "draft": {
            "temperature": draft_settings.temperature,
            "top_p": draft_settings.top_p,
            "min_p": draft_settings.min_p,
            "min_tokens_to_keep": draft_settings.min_tokens_to_keep,
            "top_k": draft_settings.top_k,
        },
    }
    return json.dumps(payload, sort_keys=True)


def _real_logit_policy_sweep(
    *,
    target_rows: list[Any],
    draft_rows: list[Any],
    top_n: int,
    include_all: bool = False,
) -> dict[str, Any]:
    from vmlx_engine.native_mtp_research import sweep_policy_pairs_cached

    target_np = [_to_numpy_row(row) for row in target_rows]
    draft_np = [_to_numpy_row(row) for row in draft_rows]
    settings_pairs = list(_research_settings_grid())
    rows = sweep_policy_pairs_cached(
        target_np,
        draft_np,
        settings_pairs,
    )
    stochastic_rows = [
        row
        for row in rows
        if int(row.get("min_target_support", 0)) > 1
        and int(row.get("min_draft_support", 0)) > 1
    ]
    payload = {
        "grid_size": len(rows),
        "depth_count": max(1, len(draft_np)),
        "sweep_mode": "cached_filtered_distributions",
        "top": rows[:top_n],
        "top_stochastic": stochastic_rows[:top_n],
    }
    if include_all:
        payload["rows"] = rows
    return payload


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return ""


def _chat_prompt(processor: Any, prompt: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        from mlx_vlm.prompt_utils import get_chat_template

        try:
            rendered = get_chat_template(
                processor,
                messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            rendered = get_chat_template(
                processor,
                messages,
                add_generation_prompt=True,
            )
    except Exception:
        rendered = prompt

    if enable_thinking is False and rendered:
        last_think = rendered.rfind("<think>")
        if last_think >= 0 and "</think>" not in rendered[last_think + 7 :]:
            rendered = rendered[: last_think + 7] + "</think>\n"
    return rendered


def _cache_after_prompt_and_first(language_model: Any, prompt_ids, first_tok):
    import mlx.core as mx

    cache = language_model.make_cache()
    prompt = mx.array([prompt_ids], dtype=mx.uint32)
    language_model(prompt, cache=cache, return_hidden=True)
    language_model(first_tok.reshape(1, 1), cache=cache, return_hidden=True)
    return cache


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "model_path",
        type=Path,
        nargs="?",
        default=Path("/Users/example/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP"),
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Use the prompt text directly instead of the chat template.",
    )
    ap.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Render chat template with thinking enabled. Default is thinking off.",
    )
    ap.add_argument(
        "--prompt",
        default="Count from one to eight, comma-separated. Return only the numbers.",
    )
    ap.add_argument(
        "--policy-sweep-top-n",
        type=int,
        default=0,
        help="If >0, rank stochastic p/q sampler policies on real D1..Dk logits.",
    )
    ap.add_argument(
        "--policy-sweep-save-all",
        action="store_true",
        help="Include all real-logit policy rows, not just top summaries.",
    )
    args = ap.parse_args()

    os.environ.setdefault("VMLINUX_NATIVE_MTP", "1")
    os.environ.setdefault("VMLINUX_NATIVE_MTP_DEPTH", str(args.depth))

    import mlx.core as mx

    from vmlx_engine.utils.jang_loader import load_jang_vlm_model

    model, processor = load_jang_vlm_model(str(args.model_path))
    language_model = getattr(model, "language_model", model)
    tokenizer = getattr(processor, "tokenizer", processor)

    if not callable(getattr(language_model, "mtp_forward", None)):
        raise RuntimeError("language model does not expose mtp_forward")
    if not callable(getattr(language_model, "make_mtp_cache", None)):
        raise RuntimeError("language model does not expose make_mtp_cache")
    if getattr(language_model, "mtp", None) is None:
        raise RuntimeError("language model has no attached mtp module")

    rendered_prompt = (
        args.prompt
        if args.raw_prompt
        else _chat_prompt(processor, args.prompt, bool(args.enable_thinking))
    )
    prompt_ids = list(tokenizer.encode(rendered_prompt))
    if not prompt_ids:
        raise RuntimeError("tokenizer returned empty prompt")

    seed_cache = language_model.make_cache()
    prompt = mx.array([prompt_ids], dtype=mx.uint32)
    logits, _hidden = _logits_hidden(
        language_model(prompt, cache=seed_cache, return_hidden=True)
    )
    first_tok = _argmax_token(logits[:, -1, :])
    logits_first, hidden_first = _logits_hidden(
        language_model(first_tok.reshape(1, 1), cache=seed_cache, return_hidden=True)
    )
    next_main = _argmax_token(logits_first[:, -1, :])

    mtp_cache = language_model.make_mtp_cache()
    drafts = []
    draft_logit_rows = []
    draft_hidden = hidden_first[:, -1:, :]
    draft_input = next_main
    for level in range(max(1, int(args.depth))):
        mtp_output = language_model.mtp_forward(
            draft_hidden,
            draft_input.reshape(1, 1),
            mtp_cache,
            return_hidden=level + 1 < args.depth,
        )
        if isinstance(mtp_output, tuple):
            mtp_logits, mtp_hidden = mtp_output
        else:
            mtp_logits, mtp_hidden = mtp_output, None
        draft_tok = _argmax_token(mtp_logits[:, -1, :])
        drafts.append(draft_tok)
        draft_logit_rows.append(mtp_logits[:, -1, :])
        draft_input = draft_tok
        if mtp_hidden is not None:
            draft_hidden = mtp_hidden[:, -1:, :]

    forced_tokens = [next_main] + drafts

    seq_cache = _cache_after_prompt_and_first(language_model, prompt_ids, first_tok)
    sequential_rows = []
    for tok in forced_tokens:
        out = language_model(tok.reshape(1, 1), cache=seq_cache, return_hidden=True)
        row_logits, _ = _logits_hidden(out)
        sequential_rows.append(row_logits[:, -1, :])

    chunk_cache = _cache_after_prompt_and_first(language_model, prompt_ids, first_tok)
    chunk_input = mx.concatenate([tok.reshape(-1) for tok in forced_tokens]).reshape(
        1, len(forced_tokens)
    )
    chunk_logits, chunk_hidden = _logits_hidden(
        language_model(
            chunk_input,
            cache=chunk_cache,
            return_hidden=True,
            n_confirmed=1,
        )
    )
    chunk_rows = [chunk_logits[:, idx, :] for idx in range(len(forced_tokens))]

    rows = []
    accepted_prefix = 0
    for idx, (seq_row, chunk_row, forced_tok) in enumerate(
        zip(sequential_rows, chunk_rows, forced_tokens)
    ):
        seq_argmax = _token_int(_argmax_token(seq_row))
        chunk_argmax = _token_int(_argmax_token(chunk_row))
        draft_token = _token_int(drafts[idx]) if idx < len(drafts) else None
        draft_matches_target = draft_token is not None and seq_argmax == draft_token
        if idx < len(drafts) and accepted_prefix == idx and draft_matches_target:
            accepted_prefix += 1
        rows.append(
            {
                "position": idx,
                "forced_input_token": _token_int(forced_tok),
                "forced_input_text": _decode_token(tokenizer, _token_int(forced_tok)),
                "draft_token": draft_token,
                "draft_text": (
                    _decode_token(tokenizer, draft_token)
                    if draft_token is not None
                    else None
                ),
                "draft_matches_sequential_target": draft_matches_target,
                "max_abs_logit_diff": _row_diff(seq_row, chunk_row),
                "sequential_argmax": seq_argmax,
                "sequential_text": _decode_token(tokenizer, seq_argmax),
                "chunk_argmax": chunk_argmax,
                "chunk_text": _decode_token(tokenizer, chunk_argmax),
                "argmax_equal": seq_argmax == chunk_argmax,
            }
        )

    result = {
        "model_path": str(args.model_path),
        "prompt": args.prompt,
        "rendered_prompt": rendered_prompt,
        "prompt_token_count": len(prompt_ids),
        "depth": int(args.depth),
        "first_token": _token_int(first_tok),
        "first_text": _decode_token(tokenizer, _token_int(first_tok)),
        "next_main": _token_int(next_main),
        "next_main_text": _decode_token(tokenizer, _token_int(next_main)),
        "drafts": [
            {"token": _token_int(tok), "text": _decode_token(tokenizer, _token_int(tok))}
            for tok in drafts
        ],
        "chunk_hidden_shape": list(chunk_hidden.shape),
        "rows": rows,
        "accepted_prefix_against_actual_drafts": accepted_prefix,
        "draft_acceptance_rate_for_probe": accepted_prefix / max(1, len(drafts)),
        "all_argmax_equal": all(row["argmax_equal"] for row in rows),
        "max_abs_logit_diff": max(row["max_abs_logit_diff"] for row in rows),
    }
    if int(args.policy_sweep_top_n or 0) > 0:
        result["real_logit_policy_sweep"] = _real_logit_policy_sweep(
            target_rows=chunk_rows[: len(draft_logit_rows)],
            draft_rows=draft_logit_rows,
            top_n=max(1, int(args.policy_sweep_top_n)),
            include_all=bool(args.policy_sweep_save_all),
        )

    out = args.out
    if out is None:
        out = (
            Path("docs/internal/release-gates")
            / "qwen36_27b_jang4m_mtp_chunk_equivalence"
            / "result.json"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["all_argmax_equal"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
