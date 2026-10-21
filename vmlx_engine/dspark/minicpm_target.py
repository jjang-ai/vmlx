"""MiniCPM-V-4.7 (dense Qwen3.5 or Qwen3.5-MoE text model) as a DSpark target.

Three forwards over the module's own vendored Qwen3.5 layers:

- ``prefill``: the stock path, structured exactly like mlx-vlm's AR prefill
  (one call for prompts <= ``prefill_step``; otherwise all-but-last in chunks
  then the last token alone), so the first token equals AR's.
- ``verify``: k + 1 <= 8 rows under ``row_exact_verify_scope``: dense
  projections run as per-row GEMVs (admitted only after a bit-equality check
  against the one-row call), attention via ``_row_exact_attention`` (paired
  single-query SDPA, per-row rotary). Measured bit-equal to k sequential
  decode steps at every layer and in the logits (k = 2..8, MiniCPM-V-4.7-1B
  BF16, JANG_8M and JANGH4, MLX 0.32.3). Requires the stock GDN path (fused
  GDN decode kernels off, the default). Dense JANGH projections are switched
  to their per-row decode kernel for verify rows (``row_exact_verify``).
  MoE (35B-A3B): below the expert sort threshold (rows x top_k < 64 routed
  rows, the switch point of MLX's SwitchGLU and of JANGH's TQSwitchGLU) the
  stock multi-row sparse block keeps every (row, expert) pair on the one-row
  gather decode kernels, so it is row-exact (measured bit-equal to per-row
  decode on a 35B-shape block for affine 4/8 and JANGH 2/4 through 7 rows).
  At the threshold (8 rows x top-8) the experts switch to the sorted
  gather_qmm and the rows are NOT exact; verify then runs each row's block as
  its own one-row call (exact by construction, ~1.7x the stock block's cost
  at 6-8 rows on the real 35B). ``preferred_max_draft`` describes that cost
  threshold (6 drafts at top-8); it does not cap the adaptive serving lane.
- ``rollback``: rejected rows are undone by trimming the attention KV and
  replaying the GatedDeltaNet recurrence over the accepted rows only, from
  the state the verify started from (captured through the vendored GDN's
  opt-in ``_rollback_sink``).

Taps are the residual-stream outputs of the drafter's ``target_layer_ids``.
Text requests: positions = sequence index. Image requests (``set_media``, from
``MLXMultimodalLM._dspark_media_plan``): prompt rows read the merged input
embeddings (vision features scattered at the placeholders) and the prompt's
canvas M-RoPE positions; every row after the prompt (AR step, verify) is text
at ``offset + rope_delta``, the same positions the served language model gives
decode steps after that prompt. The drafter only proposes text tokens.
"""

from __future__ import annotations

import importlib
from typing import List, Optional, Sequence

import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_update

from ..metal.row_exact_qmv import MAX_ROWS, rotary_rows_like_decode, row_exact_verify_scope

MAX_VERIFY_ROWS = MAX_ROWS
# Routed (row, expert) pairs at which MLX's SwitchGLU (``do_sort = indices.size >= 64``) and JANGH's TQSwitchGLU
# (``SORT_THRESHOLD``) leave the one-row gather decode kernels for the sorted multi-row path.
EXPERT_SORT_THRESHOLD = 64


def _moe_rows_stay_on_decode_kernels(block, rows: int) -> bool:
    """True when a ``rows``-row call of this sparse block runs every (row, expert) pair on the one-row kernels."""
    top_k = int(block.top_k)
    use_sorted = getattr(block.switch_mlp, "_use_sorted", None)
    if callable(use_sorted):
        return not use_sorted(rows * top_k, top_k)
    return rows * top_k < EXPERT_SORT_THRESHOLD


def verify_mode() -> str:
    """exact (default: every verify row bit-equal to its 1-row decode step) or fast (stock multi-row kernels)."""
    import os

    return "fast" if os.environ.get("VMLX_DSPARK_VERIFY", "exact").strip().lower() == "fast" else "exact"


def _cache_offset(cache, idx: int) -> int:
    entry = cache[idx]
    entry = getattr(entry, "_inner", entry)
    offset = getattr(entry, "offset", 0) or 0
    if isinstance(offset, mx.array):
        return int(offset.reshape(-1)[0].item())
    return int(offset)


class MiniCPMDSparkTarget:
    def __init__(self, model):
        self.model = model
        self.lm = model.language_model
        self.nm = self.lm.model
        self.moe = bool(getattr(self.lm.args, "num_experts", 0))
        if self.moe and not all(hasattr(layer.mlp, "switch_mlp") for layer in self.nm.layers):
            raise ValueError("DSpark MoE target: every decoder layer must hold a sparse MoE block")
        top_k = int(getattr(self.lm.args, "num_experts_per_tok", 1) or 1) if self.moe else 1
        # Widest verify that stays on the one-row expert kernels: rows * top_k < SORT_THRESHOLD.
        self.preferred_max_draft = (
            max(1, min(MAX_VERIFY_ROWS, (EXPERT_SORT_THRESHOLD - 1) // top_k) - 1) if self.moe
            else MAX_VERIFY_ROWS - 1
        )
        from ..models.minicpmv4_7.language import _CanvasLanguageMixin

        if not isinstance(self.lm, _CanvasLanguageMixin):
            raise ValueError("DSpark target expects the minicpmv4_7 language model")
        self.qlang = importlib.import_module("mlx_vlm.models.qwen3_5.language")
        from ..patches.mlx_vlm_mtp.qwen35_vl import (
            _qwen_vlm_gated_delta_kernel_enabled,
            _row_exact_attention,
        )

        self._row_exact_attention = _row_exact_attention
        self._patched_kernel_enabled = _qwen_vlm_gated_delta_kernel_enabled
        self.lm_head = self.lm.lm_head
        self.embed_tokens = self.nm.embed_tokens
        self.media_embeds = None
        self.media_positions = None
        self.rope_delta = 0
        # Dense JANGH projections: per-row decode kernel for verify rows.
        for _, module in self.lm.named_modules():
            if getattr(module, "is_jangtq2_dense", False):
                module.row_exact_verify = True
        for layer in self.nm.layers:
            if layer.is_linear and (getattr(layer.linear_attn, "_fused_gdn_conv", False)
                                    or getattr(layer.linear_attn, "_fused_gdn_gates", False)):
                raise ValueError("DSpark exact verify needs the stock GDN path (fused GDN decode kernels are on)")

    def make_cache(self):
        return self.lm.make_cache()

    def set_media(self, embeds=None, positions=None, rope_delta: int = 0) -> None:
        """One request's media plan (cleared with ``set_media()``): (1, P, H) prompt embeddings, (3, 1, P) canvas
        positions, and the rope delta of the text after the prompt. A resumed prefix that already holds every media
        item needs only the delta (embeds/positions None)."""
        self.media_embeds = embeds
        self.media_positions = positions
        self.rope_delta = int(rope_delta or 0)

    def _positions(self, offset: int, rows: int) -> mx.array:
        plan = self.media_positions
        if plan is not None and offset + rows <= int(plan.shape[-1]):
            return plan[:, :, offset:offset + rows]
        start = offset + self.rope_delta
        pos = mx.arange(start, start + rows, dtype=mx.int32)[None, None]
        return mx.broadcast_to(pos, (3, 1, rows))

    def _embed(self, tokens: mx.array, offset: int) -> mx.array:
        rows = int(tokens.shape[1])
        if self.media_embeds is not None and offset + rows <= int(self.media_embeds.shape[1]):
            return self.media_embeds[:, offset:offset + rows]
        return self.nm.embed_tokens(tokens)

    def _masks(self, h, cache):
        base = importlib.import_module("mlx_vlm.models.base")
        return (
            base.create_attention_mask(h, cache[self.nm.fa_idx]),
            base.create_ssm_mask(h, cache[self.nm.ssm_idx]),
        )

    def _stock(self, tokens: mx.array, cache, taps: Sequence[int], logits: bool):
        """Same ops as LanguageModel.__call__ -> Qwen3_5Model.__call__ on text rows."""
        nm = self.nm
        offset = _cache_offset(cache, nm.fa_idx)
        h = self._embed(tokens, offset)
        pos = self._positions(offset, int(tokens.shape[1]))
        fa_mask, ssm_mask = self._masks(h, cache)
        pe = nm.layers[nm.fa_idx].self_attn.rotary_emb(h, pos)
        feats = []
        for i, (layer, c) in enumerate(zip(nm.layers, cache)):
            h = layer(h, ssm_mask if layer.is_linear else fa_mask, c, pos, position_embeddings=pe)
            if i in taps:
                feats.append(h)
        out = self.lm_head(nm.norm(h)) if logits else None
        return out, (mx.concatenate(feats, axis=-1) if feats else None)

    def prefill(self, ids: mx.array, cache, taps: Sequence[int], prefill_step: Optional[int]):
        """Returns (last-row logits (V,), tap features (1, P, n_taps * H))."""
        from ..utils.generation_cancellation import raise_if_generation_cancelled

        raise_if_generation_cancelled()
        length = int(ids.shape[1])
        feats = []
        if prefill_step is not None and length > prefill_step:
            start = 0
            while length - start > 1:
                raise_if_generation_cancelled()
                n = min(int(prefill_step), length - start - 1)
                _, f = self._stock(ids[:, start:start + n], cache, taps, logits=False)
                mx.eval([c.state for c in cache])
                if f is not None:
                    mx.eval(f)
                    feats.append(f)
                start += n
            raise_if_generation_cancelled()
            logits, f = self._stock(ids[:, -1:], cache, taps, logits=True)
        else:
            logits, f = self._stock(ids, cache, taps, logits=True)
        raise_if_generation_cancelled()
        if f is not None:
            feats.append(f)
        return logits[0, -1], (mx.concatenate(feats, axis=1) if feats else None)

    def prefill_segment(self, ids: mx.array, cache, taps: Sequence[int], prefill_step: Optional[int]):
        """Forward a prompt segment that ends at a session-snapshot cut (no logits), in ``prefill_step`` chunks.
        Returns its tap features (1, n, n_taps * H) or None without taps."""
        from ..utils.generation_cancellation import raise_if_generation_cancelled

        length = int(ids.shape[1])
        step = int(prefill_step) if prefill_step else length
        feats = []
        for start in range(0, length, max(1, step)):
            raise_if_generation_cancelled()
            _, f = self._stock(ids[:, start:start + step], cache, taps, logits=False)
            mx.eval([c.state for c in cache])
            if f is not None:
                mx.eval(f)
                feats.append(f)
        raise_if_generation_cancelled()
        return mx.concatenate(feats, axis=1) if feats else None

    def ar_step(self, token: mx.array, cache, taps: Sequence[int]):
        """One AR step that also returns the tap features (DSpark's AR fallback keeps the drafter context complete).
        The stock one-row forward (the prefill path's last-token call); bit-equal to ``decode`` (tests)."""
        logits, feats = self._stock(token.reshape(1, 1), cache, taps, logits=True)
        return logits[0, -1], feats

    def decode(self, token: mx.array, cache):
        """One AR step through the served language-model call (reference path)."""
        return self.lm(token.reshape(1, 1), cache=cache).logits[0, -1]

    def verify(self, tokens: mx.array, cache, taps: Sequence[int]):
        """Row-exact forward of (1, k + 1) rows. Returns (logits (k+1, V), feats, gdn_sink).
        ``VMLX_DSPARK_VERIFY=fast``: the stock multi-row forward instead (``verify_fast``)."""
        rows = int(tokens.shape[1])
        if not 1 <= rows <= MAX_VERIFY_ROWS:
            raise ValueError(f"DSpark verify supports 1..{MAX_VERIFY_ROWS} rows, got {rows}")
        if verify_mode() == "fast":
            return self.verify_fast(tokens, cache, taps)
        nm = self.nm
        sink: List[tuple] = []
        gdn = [layer.linear_attn for layer in nm.layers if layer.is_linear]
        with row_exact_verify_scope() as exact:
            if not exact:
                raise RuntimeError("row-exact verify is disabled (VMLX_ROW_EXACT_VERIFY_QMV=0)")
            offset = _cache_offset(cache, nm.fa_idx)
            h = self._embed(tokens, offset)
            pos = self._positions(offset, rows)
            _, ssm_mask = self._masks(h, cache)
            pe = rotary_rows_like_decode(nm.layers[nm.fa_idx].self_attn.rotary_emb, pos, h)
            for module in gdn:
                module._rollback_sink = sink
            feats = []
            try:
                for i, (layer, c) in enumerate(zip(nm.layers, cache)):
                    x = layer.input_layernorm(h)
                    if layer.is_linear:
                        r = layer.linear_attn(x, ssm_mask, c)
                    else:
                        r = self._row_exact_attention(layer.self_attn, self.qlang, x, c, pos, pe)
                    h = h + r
                    xn = layer.post_attention_layernorm(h)
                    if self.moe and rows > 1 and not _moe_rows_stay_on_decode_kernels(layer.mlp, rows):
                        # At the sort threshold: one-row calls, bit-equal to decode by construction.
                        m = mx.concatenate([layer.mlp(mx.contiguous(xn[:, r:r + 1])) for r in range(rows)], axis=1)
                    else:
                        m = layer.mlp(xn)
                    h = h + m
                    if i in taps:
                        feats.append(h)
                logits = self.lm_head(nm.norm(h))[0]
            finally:
                for module in gdn:
                    module._rollback_sink = None
        return logits, (mx.concatenate(feats, axis=-1) if feats else None), sink

    def verify_fast(self, tokens: mx.array, cache, taps: Sequence[int]):
        """Opt-in (VMLX_DSPARK_VERIFY=fast): the served stock forward over all k + 1 rows -- MLX's multi-row qmm, one
        causal SDPA, the sorted expert route from 64 routed rows -- with the same GDN rollback capture. Every accepted
        token is still the target's own pick from its logits, so the output is a valid decode of the model, but the
        multi-row kernels round differently from 1-row decode steps: greedy can differ from AR at near-ties (the
        dense Qwen 27B family runs speed-first the same way, row_exact_qmv._DENSE_SPEED_FIRST)."""
        sink: List[tuple] = []
        gdn = [layer.linear_attn for layer in self.nm.layers if layer.is_linear]
        for module in gdn:
            module._rollback_sink = sink
        try:
            logits, feats = self._stock(tokens, cache, taps, logits=True)
        finally:
            for module in gdn:
                module._rollback_sink = None
        return logits[0], feats, sink

    def rollback(self, cache, sink: list, kept: int, total: int) -> None:
        """Keep the first ``kept`` of ``total`` verify rows in every cache."""
        trim = total - kept
        if trim <= 0:
            return
        index = 0
        for layer, c in zip(self.nm.layers, cache):
            if not layer.is_linear:
                c.trim(trim)
                continue
            entry = sink[index]
            index += 1
            module = layer.linear_attn
            # Replay through the recurrence implementation the forward used
            # (Metal kernel vs ops agree only to ~1e-9).
            if len(entry) == 11:  # native-MTP patch layout (A_log/dt_bias inline)
                q, k, v, a, b, a_log, dt_bias, state0, mask, conv_input, kernel = entry
                use_kernel = self._patched_kernel_enabled(module)
            else:  # vendored layer layout
                q, k, v, a, b, state0, mask, conv_input, kernel = entry
                a_log, dt_bias = module.A_log, module.dt_bias
                use_kernel = not module.training
            _, state = gated_delta_update(
                q[:, :kept], k[:, :kept], v[:, :kept], a[:, :kept], b[:, :kept],
                a_log, dt_bias, state0,
                None if mask is None else mask[:, :kept],
                use_kernel=use_kernel,
            )
            c[1] = state
            c[0] = mx.contiguous(conv_input[:, kept:kept + int(kernel) - 1])
        if index != len(sink):
            raise RuntimeError("DSpark rollback: GDN sink does not match the layer stack")
