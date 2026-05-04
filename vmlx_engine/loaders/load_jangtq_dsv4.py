# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V4 JANGTQ loader — thin re-export.

Mirrors the pattern used for Kimi K2.6 (``load_jangtq_kimi_vlm``): a one-line
indirection so the research doc's code snippets work verbatim against the
vMLX engine (``from vmlx_engine.loaders.load_jangtq_dsv4 import
load_jangtq_dsv4_model``) AND so the bundled Python distribution always
exposes a stable import path even if ``jang_tools.dsv4`` internals move.

Under the hood this delegates to ``jang_tools.load_jangtq.load_jangtq_model``,
which already detects ``model_type="deepseek_v4"`` from ``config.json`` and
routes through ``jang_tools.dsv4.mlx_register`` (auto-imported at first call
via ``vmlx_engine.utils.jang_loader``) to register our custom MLX model class
(``jang_tools.dsv4.mlx_model.Model``) into ``mlx_lm.models`` namespace.

Runtime contract implemented by the underlying loader:
- mHC (Manifold Hyper-Connections) hc_mult=4 with 20 Sinkhorn iterations
  (fused Metal kernel, fallback pure-MLX op available).
- MLA attention head_dim=512 with grouped O projection (o_groups=8,
  o_lora_rank=1024), per-head RMSNorm, attention sink, inverse RoPE on
  output, partial RoPE over last qk_rope_head_dim=64 dimensions.
- 256 routed experts top-6 + 1 shared (switch_mlp stacked across layers).
- sqrtsoftplus routing with hash-table (tid2eid) for the first 3 layers,
  biased top-6 argpartition + weight-norm + routed_scaling_factor=1.5 for
  the remaining 40 layers.
- Cache: ``DeepseekV4Cache`` (RotatingKVCache sliding_window=128 +
  compressor/indexer state buffers) on ``compress_ratio>0`` layers,
  plain ``KVCache`` on ``compress_ratio=0`` layers. ``DSV4_LONG_CTX=1``
  is forced. Upstream JANG has a prefill mask-vs-full_kv shape bug
  that crashes any prompt > sliding_window with broadcast errors;
  this loader patches it in-process by replacing
  ``DeepseekV4Attention.__call__`` with a copy that adds the missing
  symmetric trim branch (see ``_install_dsv4_prefill_patch``).
- ``swiglu_limit=10`` SwiGLU activation, fp32 lm_head matmul (4096
  contraction in bf16 drifts logits enough to flip arithmetic answers —
  see research/DSV-EXHAUSTIVE-VARIABLES-GUIDE.md).

The function returns ``(model, tokenizer)`` just like the sibling loaders.
"""

from __future__ import annotations

import logging as _logging
import os
import sys
from typing import Any, Tuple

_log = _logging.getLogger(__name__)
_PREFILL_PATCH_INSTALLED = False


def _install_dsv4_prefill_patch() -> None:
    """Patch the upstream JANG ``DeepseekV4Attention.__call__`` mask-trim
    bug in-process.

    The installed ``jang_tools/dsv4/mlx_model.py`` (DeepseekV4Attention,
    line ~882) only pads ``mask`` when ``full_kv.shape[2] > mask.shape[-1]``
    and is missing the symmetric trim branch. Live-traced 2026-05-04 with
    LONG_CTX=1: layer 0 has compress_ratio=0 so its cache is a plain
    KVCache that grows monotonically with every token — the model-level
    mask gets sized to ``layer0.offset``. Compress_ratio>0 layers wrap a
    RotatingKVCache(max_size=sliding_window=128) so their post-update
    ``full_kv`` caps at ``sliding_window + (pooled rows)``. Result: on any
    prompt + decode step where total tokens > sliding_window, the
    incoming mask shape ``(1, total_tokens)`` is wider than full_kv
    ``(1, ~129)``, ``mx.broadcast_shapes`` rejects, the request returns
    empty content. Patch wraps ``__call__`` to trim trailing-axis when
    mask overruns full_kv (mirror of the existing pad branch). Idempotent.
    """
    global _PREFILL_PATCH_INSTALLED
    if _PREFILL_PATCH_INSTALLED:
        return
    try:
        from jang_tools.dsv4.mlx_model import DeepseekV4Attention
        import mlx.core as mx
    except Exception as e:
        _log.warning("DSV4 prefill patch skipped (import failed: %s)", e)
        return

    from jang_tools.dsv4.mlx_model import (
        DeepseekV4Cache,
        _apply_partial_rope,
        _get_q_norm_ones,
        scaled_dot_product_attention,
    )

    def _patched_call(self, x, mask=None, cache=None):  # noqa: D401
        # Mirror of upstream DeepseekV4Attention.__call__ with a symmetric
        # trim branch added before the existing pad branch. Other than the
        # 4-line trim block this is a copy of the upstream forward; we
        # match its behavior exactly to avoid drift.
        B, L, _ = x.shape
        local_cache = cache if isinstance(cache, DeepseekV4Cache) else cache
        offset = local_cache.offset if local_cache is not None else 0

        q_residual = self.q_norm(self.wq_a(x))
        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(
            q,
            weight=_get_q_norm_ones(self.head_dim, q.dtype),
            eps=self.args.rms_norm_eps,
        )
        q = q.transpose(0, 2, 1, 3)

        kv = self.kv_norm(self.wkv(x)).reshape(B, L, 1, self.head_dim).transpose(0, 2, 1, 3)
        q = _apply_partial_rope(q, self.rope, offset)
        kv = _apply_partial_rope(kv, self.rope, offset)

        if local_cache is not None:
            kv, _ = local_cache.update_and_fetch(kv, kv)
        full_kv = kv

        if self.compress_ratio:
            v4_cache = cache if isinstance(cache, DeepseekV4Cache) else None
            if v4_cache is not None or L >= self.compress_ratio:
                pooled = self.compressor(x, self.compress_rope, v4_cache, offset)
                if hasattr(self, "indexer") and pooled.shape[1] > 0:
                    topk = self.indexer(
                        x, q_residual, self.compress_rope, self.rope,
                        v4_cache, offset,
                    )
                    if topk is not None:
                        expanded = mx.broadcast_to(
                            pooled[:, None, None, :, :],
                            (B, 1, L, pooled.shape[1], self.head_dim),
                        )
                        idx = topk[:, None, :, :, None]
                        pooled = mx.take_along_axis(
                            expanded,
                            mx.broadcast_to(idx, idx.shape[:-1] + (self.head_dim,)),
                            axis=3,
                        ).reshape(B, 1, -1, self.head_dim)
                    else:
                        pooled = pooled[:, None]
                else:
                    pooled = pooled[:, None]
                if pooled.shape[2] > 0:
                    full_kv = mx.concatenate([full_kv, pooled], axis=2)

        # vMLX additions: handle BOTH directions of the mask vs full_kv
        # mismatch. Upstream only pads (full_kv > mask). When the model-
        # level mask was sized to a layer whose cache grows monotonically
        # (e.g. layer 0 with compress_ratio=0 → KVCache, offset = total
        # tokens) but THIS layer wraps a RotatingKVCache(sliding_window)
        # whose post-update size caps near ``sliding_window`` (+pooled
        # rows), the upstream code feeds a too-wide mask straight into
        # SDPA and `mx.broadcast_shapes` rejects it.
        if mask is not None and hasattr(mask, "shape") and mask.shape:
            if mask.shape[-1] > full_kv.shape[2]:
                mask = mask[..., -full_kv.shape[2]:]
            elif full_kv.shape[2] > mask.shape[-1]:
                pad = mx.zeros(
                    mask.shape[:-1] + (full_kv.shape[2] - mask.shape[-1],),
                    dtype=mask.dtype,
                )
                mask = mx.concatenate([mask, pad], axis=-1)

        out = scaled_dot_product_attention(
            q, full_kv, full_kv,
            cache=local_cache, scale=self.softmax_scale, mask=mask,
            sinks=self.attn_sink.astype(q.dtype),
        )
        out = _apply_partial_rope(out, self.rope, offset, inverse=True)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.n_heads * self.head_dim)
        out = self._grouped_output_projection(out)
        return self.wo_b(out)

    DeepseekV4Attention.__call__ = _patched_call
    _PREFILL_PATCH_INSTALLED = True
    _log.info(
        "DSV4 prefill mask-trim patch installed on "
        "jang_tools.dsv4.mlx_model.DeepseekV4Attention.__call__ "
        "(symmetric to upstream pad branch; fixes broadcast_shapes for "
        "prompts > sliding_window with mixed KVCache/DeepseekV4Cache layers)."
    )


def _install_dsv4_memory_defaults() -> None:
    """Set DSV4-specific MLX/JANG runtime defaults before hydration.

    The generic JANGTQ loader uses a 70% wired-limit default, which is fine for
    many MoEs but too aggressive for DSV4-Flash on 128 GB Macs: the final model
    is about 79 GB and the streaming stacker still needs page-cache and temporary
    graph headroom while converting per-expert tensors. The directly validated
    working point on the target machine is ~72 GB, which is about 52% of 128 GiB
    expressed as decimal GB. Keep explicit caller/user overrides intact.
    """
    if "JANGTQ_WIRED_LIMIT_GB" not in os.environ:
        try:
            if sys.platform == "darwin":
                import psutil

                total_gb = psutil.virtual_memory().total / 1e9
                target_gb = int(total_gb * 0.52)
                target_gb = max(48, min(target_gb, 160))
                os.environ["JANGTQ_WIRED_LIMIT_GB"] = str(target_gb)
                print(
                    "  [dsv4] JANGTQ_WIRED_LIMIT_GB defaulted to "
                    f"{target_gb} GB (~52% of {total_gb:.0f} GB RAM)",
                    flush=True,
                )
        except Exception:
            # Non-fatal: jang_tools.load_jangtq has its own fallback.
            pass
    # DSV4 decode builds large transient MLX graph/cache allocations. Leaving
    # MLX's process cache uncapped lets two short chat turns retain ~30 GB of
    # purgeable cache, which can trip vMLX's memory-pressure guard even though
    # prefix/L2 caches are empty. Match the standalone JANG DSV4 scripts'
    # conservative cache ceiling, but keep an env override for benchmarking.
    try:
        import mlx.core as mx

        cache_gb = float(os.environ.get("DSV4_MLX_CACHE_LIMIT_GB", "8"))
        if cache_gb > 0:
            mx.set_cache_limit(int(cache_gb * 1024**3))
            print(
                f"  [dsv4] MLX cache limit set to {cache_gb:g} GB "
                "(env DSV4_MLX_CACHE_LIMIT_GB)",
                flush=True,
            )
    except Exception:
        pass


def load_jangtq_dsv4_model(model_path: str, *, skip_params_eval: bool = True) -> Tuple[Any, Any]:
    """Load a DeepSeek V4 JANGTQ bundle.

    The actual work happens in ``jang_tools.load_jangtq.load_jangtq_model``.
    This wrapper exists for API-stability so the research docs'
    ``from vmlx_engine.loaders.load_jangtq_dsv4 import ...`` examples work.

    Args:
        model_path: Path to the bundle directory containing ``config.json``,
            ``jang_config.json``, and packed safetensors shards.
        skip_params_eval: If True, skip the post-hydration all-parameter
            materialization pass in the underlying JANGTQ loader. DSV4 still
            runs the full-model 1-token warmup after hydration, which is the
            real inference path and avoids doubling load-time memory pressure.

    Returns:
        A ``(model, tokenizer)`` tuple. ``model.config`` carries the raw
        ``config.json`` dict; ``tokenizer.jang_chat`` carries the
        ``jang_config.json.chat`` block (EOS + reasoning modes +
        tool_calling parser name + sampling_defaults).

    Raises:
        ImportError: if ``jang_tools.dsv4`` is not installed (older jang-tools
            without the dsv4 submodule). Callers should surface this as
            "Reinstall vMLX from the latest DMG — the bundled Python is out
            of date".
    """
    # DSV4_LONG_CTX=1 is the only supported runtime mode. The upstream JANG
    # prefill shape bug for prompts > sliding_window is patched in this
    # process by ``_install_dsv4_prefill_patch()`` (called below). Pool
    # quant remains off because PoolQuantizedV4Cache is not a subclass of
    # DeepseekV4Cache in the engine's isinstance gates.
    os.environ["DSV4_LONG_CTX"] = "1"
    os.environ["DSV4_POOL_QUANT"] = "0"
    _install_dsv4_memory_defaults()

    # Eagerly register mlx_lm.models.deepseek_v4 so the underlying loader
    # can resolve the model_type. Safe to call multiple times (idempotent).
    from jang_tools.dsv4 import mlx_register  # noqa: F401
    from jang_tools.load_jangtq import load_jangtq_model

    # Install the in-process patch for the upstream JANG prefill mask-trim
    # bug BEFORE the model gets called for warmup/inference.
    _install_dsv4_prefill_patch()

    model, tokenizer = load_jangtq_model(model_path, skip_params_eval=skip_params_eval)

    # 2026-05-03 (F17): install canonical-encoder shim on
    # tokenizer.apply_chat_template. The bundle ships a Jinja chat_template
    # that PARSES `reasoning_effort` but never branches on it — every effort
    # value renders the same prompt (verified by direct render). The
    # canonical encoder `jang_tools.dsv4.encoding_dsv4.encode_messages`
    # (in <bundle>/encoding/encoding_dsv4.py) is what actually implements
    # the three modes (chat / thinking / thinking max with system preamble)
    # AND multi-turn `drop_earlier_reasoning`. Without this shim, vmlx
    # routes through the (broken) Jinja path → DSV4 thinking-mode produces
    # unbounded reasoning that never closes `</think>`. With the shim,
    # vmlx's `tokenizer.apply_chat_template(messages, enable_thinking=...,
    # reasoning_effort=...)` calls land on `dsv4_chat_encoder.apply_chat_template`
    # which respects effort tiers and strips prior `<think>` blocks from
    # multi-turn history. Idempotent: only installs once per tokenizer.
    try:
        if not getattr(tokenizer, "_vmlx_dsv4_chat_template_shim", False):
            from .dsv4_chat_encoder import apply_chat_template as _dsv4_apply

            _orig_apply = getattr(tokenizer, "apply_chat_template", None)

            def _patched_apply(messages, *args, **kwargs):
                # Honor caller's enable_thinking + reasoning_effort kwargs.
                # Bundle template ignored these; canonical encoder uses them.
                _enable_thinking = kwargs.pop("enable_thinking", None)
                _reasoning_effort = kwargs.pop("reasoning_effort", None)
                _tools = kwargs.pop("tools", None)
                _add_default_bos = kwargs.pop("add_default_bos_token", True)
                _drop_thinking = kwargs.pop("drop_earlier_reasoning", True)
                # Some callers pass `tokenize=True` and want token IDs back.
                _tokenize = kwargs.pop("tokenize", False)
                # Other Jinja-template kwargs (add_generation_prompt, etc.)
                # are inherent to encode_messages — ignore silently.
                kwargs.pop("add_generation_prompt", None)
                kwargs.pop("chat_template", None)
                # Unrecognized kwargs forwarded back to original Jinja path
                # would only be Jinja-interpreted variables; canonical
                # encoder doesn't need them. Drop quietly.
                kwargs.pop("documents", None)
                kwargs.pop("conversation", None)
                kwargs.pop("padding", None)
                kwargs.pop("truncation", None)
                kwargs.pop("max_length", None)
                kwargs.pop("return_tensors", None)
                kwargs.pop("return_dict", None)
                # `chat_template_kwargs` (HF convention) — flatten effort/thinking
                # if caller passed them this way instead of top-level.
                _ct_kwargs = kwargs.pop("chat_template_kwargs", None) or {}
                if _enable_thinking is None and "enable_thinking" in _ct_kwargs:
                    _enable_thinking = _ct_kwargs["enable_thinking"]
                if _reasoning_effort is None and "reasoning_effort" in _ct_kwargs:
                    _reasoning_effort = _ct_kwargs["reasoning_effort"]

                prompt_text = _dsv4_apply(
                    messages,
                    enable_thinking=_enable_thinking,
                    reasoning_effort=_reasoning_effort,
                    tools=_tools,
                    add_default_bos_token=_add_default_bos,
                    drop_earlier_reasoning=_drop_thinking,
                    model_path=model_path,
                )

                if _tokenize:
                    # Mimic transformers' `tokenize=True` — return list of ids.
                    if hasattr(tokenizer, "encode"):
                        return tokenizer.encode(prompt_text, add_special_tokens=False)
                    inner = getattr(tokenizer, "_tokenizer", tokenizer)
                    if hasattr(inner, "encode"):
                        return inner.encode(prompt_text, add_special_tokens=False)
                    raise RuntimeError("DSV4 chat-template shim: tokenize=True requested but no encode method")
                return prompt_text

            tokenizer.apply_chat_template = _patched_apply
            tokenizer._vmlx_dsv4_chat_template_shim = True
            tokenizer._vmlx_dsv4_chat_template_orig = _orig_apply
            try:
                import logging as _logging
                _logging.getLogger(__name__).info(
                    "DSV4 chat-template shim installed: tokenizer.apply_chat_template "
                    "now routes through encoding_dsv4.encode_messages (canonical encoder, "
                    "honors reasoning_effort + drop_earlier_reasoning)."
                )
            except Exception:
                pass
    except Exception as _shim_err:
        try:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                f"DSV4 chat-template shim install failed ({_shim_err}); "
                f"falling back to bundle's Jinja template."
            )
        except Exception:
            pass

    return model, tokenizer
