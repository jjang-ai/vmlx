# SPDX-License-Identifier: Apache-2.0
"""DFlash2 speculative decoding for Inkling-2-Mini (inkling2 / inkling2_loop) targets.

Contract (exact match to official Inkling DFlash2 specification):
- Target taps: ``target_layer_ids = [5, 23, 41, 48]`` (native Inkling-2 virtual visits;
  visit 48 == ``len(layer_indices)`` is the post-final-norm hidden state).
- Drafter input embedding: ``target.model.embed(inputs)`` (applies ``embed_norm`` after
  ``embed_tokens``, NOT raw ``embed_tokens``).
- Candidate selector softcap: ``selector_softcap_mode == "after_correction"`` with
  ``final_logit_softcapping = 30.0`` — raw unary logits are NOT softcapped before
  ``candidate_selector.select()``; ``30.0 * tanh((unary + edges) / 30.0)`` is applied
  AFTER edge correction.
- Valid vocabulary: unary logits are sliced to ``target.config.unpadded_vocab_size``
  (200,058) instead of padded storage width (201,024).
- Verify & rollback: uses Inkling-2's native ``speculative_verify=True`` transaction
  (row-exact decode geometry + atomic KV/RotatingKV/ArraysCache convolution rollback),
  guaranteeing 100% greedy token parity with plain AR decoding.
- Adaptive draft width (Rule 0): ``AdaptiveSpec`` freely governs AR vs speculative verify
  width from live measured wall times; no hard depth cap.
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from typing import Any, Iterator, Optional

import mlx.core as mx
import mlx.nn as nn

from .glm_dflash2 import AdaptiveSpec, SpecStats

logger = logging.getLogger(__name__)


def is_inkling2(model: Any) -> bool:
    """True when ``model`` (or its wrapped ``language_model``) is an Inkling-2 target (no name regex)."""
    for obj in (model, getattr(model, "language_model", None)):
        if obj is None:
            continue
        mt = getattr(obj, "model_type", None)
        if mt in {"inkling2", "inkling2_loop"}:
            return True
        cfg = getattr(obj, "config", None)
        if cfg is not None:
            cmt = cfg.get("model_type") if isinstance(cfg, dict) else getattr(cfg, "model_type", None)
            if cmt in {"inkling2", "inkling2_loop", "inkling2_loop_text"}:
                return True
    return False


# ---------------------------------------------------------------------------
# Fused Metal kernel for DFlash2 GroupedDynamicCausalConv (K=2, G_sz=16)
# Replaces 8 Python ops x 20 calls per draft step with 1 single Metal dispatch!
# ---------------------------------------------------------------------------
_GROUPED_DYN_CONV_K2_SRC = r"""
    // hidden:  [B, L, H] (T)
    // dynamic: [B, L, 2, G] (T) where G = H / G_SZ
    // base:    [2, H] (T)
    // out:     [B, L, H] (T)
    const uint idx = thread_position_in_grid.x;
    const uint B = uint(hidden_shape[0]);
    const uint L = uint(hidden_shape[1]);
    const uint H = uint(hidden_shape[2]);
    if (idx >= B * L * H) return;
    const uint c = idx % H;
    const uint t = (idx / H) % L;
    const uint b = idx / (H * L);
    const uint g = c / uint(G_SZ);
    const uint G = H / uint(G_SZ);

    const float v0 = float(hidden[idx]);
    const float k0 = float(base[c]) + float(dynamic[((b * L + t) * 2u + 0u) * G + g]);
    float acc = k0 * v0;
    if (t > 0u) {
        const float v1 = float(hidden[((b * L + (t - 1u)) * H) + c]);
        const float k1 = float(base[H + c]) + float(dynamic[((b * L + t) * 2u + 1u) * G + g]);
        acc += k1 * v1;
    }
    out[idx] = T(acc);
"""


@lru_cache(maxsize=4)
def _grouped_dyn_conv_k2_kernel(dtype_name: str, group_size: int):
    return mx.fast.metal_kernel(
        name=f"vmlx_dflash2_gdconv_k2_{dtype_name}_g{group_size}",
        input_names=["hidden", "dynamic", "base"],
        output_names=["out"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=_GROUPED_DYN_CONV_K2_SRC,
        ensure_row_contiguous=True,
    )


def _fused_grouped_dynamic_convolve(hidden: mx.array, dynamic: mx.array, base: mx.array, group_size: int) -> mx.array:
    if (
        mx.default_device() == mx.gpu
        and mx.metal.is_available()
        and base.shape[0] == 2
        and hidden.dtype in (mx.bfloat16, mx.float16, mx.float32)
        and hidden.size > 0
    ):
        dname = {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}[hidden.dtype]
        b_cast = base if base.dtype == hidden.dtype else base.astype(hidden.dtype)
        d_cast = dynamic if dynamic.dtype == hidden.dtype else dynamic.astype(hidden.dtype)
        n = int(hidden.size)
        return _grouped_dyn_conv_k2_kernel(dname, int(group_size))(
            inputs=[hidden, d_cast, b_cast],
            template=[("T", hidden.dtype), ("G_SZ", int(group_size))],
            grid=(((n + 255) // 256) * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[hidden.shape],
            output_dtypes=[hidden.dtype],
        )[0]
    from dflash.model_mlx import _grouped_dynamic_convolve
    return _grouped_dynamic_convolve(hidden, dynamic, base, group_size)


def install_fused_dflash2_convs(draft: Any) -> int:
    """Patch GroupedDynamicCausalConv modules on a DFlash2 drafter with the 1-dispatch Metal kernel."""
    count = 0
    for layer in getattr(draft, "layers", ()):
        for attr in ("attention_conv", "mlp_conv"):
            conv = getattr(layer, attr, None)
            if conv is None or getattr(conv, "_vmlx_fused", False):
                continue
            if int(getattr(conv, "kernel_size", 0)) != 2:
                continue

            def _prepare(self_c, hidden):
                groups = hidden.shape[-1] // self_c.group_size
                dyn = self_c.kernel_projection(hidden).reshape(*hidden.shape[:-1], 2, 2, groups)
                return (
                    _fused_grouped_dynamic_convolve(hidden, dyn[..., 0, :, :], self_c.base_kernel[0], self_c.group_size),
                    dyn[..., 1, :, :],
                )

            def _finish(self_c, hidden, dyn):
                return _fused_grouped_dynamic_convolve(hidden, dyn, self_c.base_kernel[1], self_c.group_size)

            conv.prepare = _prepare.__get__(conv, type(conv))
            conv.finish = _finish.__get__(conv, type(conv))
            conv._vmlx_fused = True
            count += 1
    return count


def configure_inkling_dflash2(draft: Any, target_lm: Any) -> None:
    """Bind Inkling-2 target embedding (with embed_norm), unpadded vocab slice, and post-correction softcap."""
    from dflash.model_mlx import _sample_probs, _sampling_probs

    target_lm = getattr(target_lm, "language_model", target_lm)
    valid_vocab = int(getattr(target_lm.config, "unpadded_vocab_size", target_lm.config.vocab_size))
    cap = float(getattr(draft.config, "final_logit_softcapping", 0.0) or 0.0)
    out_mult = float(getattr(draft.config, "output_multiplier", 1.0) or 1.0)

    # 1. Input embeddings MUST run embed_norm(embed_tokens(inputs))!
    draft.embed_tokens = target_lm.model.embed
    draft.embed_scale = float(getattr(draft.config, "input_embedding_scale", 1.0) or 1.0)
    draft.lm_head = target_lm.lm_head

    # 2. Unary logits: raw lm_head sliced to valid_vocab_size (200,058), NO pre-softcap!
    #    Use fused multi-vector affine-8 GEMV when lm_head is affine-8 g=64 (3.5 ms -> 1.0 ms!).
    def _compute_logits(self_d, hidden):
        from .models.inkling2.jangh import is_affine_decode_projection
        from .models.inkling2.kernels import affine8_g64_multi_gemv

        if (
            is_affine_decode_projection(self_d.lm_head)
            and hidden.ndim == 3
            and hidden.shape[0] == 1
            and 1 <= hidden.shape[1] <= 16
        ):
            logits = affine8_g64_multi_gemv(
                hidden, self_d.lm_head.weight, self_d.lm_head.scales, self_d.lm_head.biases,
                out_dtype=mx.float32, softcap=0.0, out_cols=valid_vocab,
            )
            if logits is not None:
                return logits * out_mult if out_mult != 1.0 else logits
        logits = self_d.lm_head(hidden)
        if logits.shape[-1] > valid_vocab:
            logits = logits[..., :valid_vocab]
        if out_mult != 1.0:
            logits = logits * out_mult
        return logits

    draft.compute_logits = _compute_logits.__get__(draft, type(draft))

    # 3. CandidateSelector: fused 2-stage top-K + single-dispatch Markov edge selector
    #    with exact 30.0 * tanh((unary + edges) / 30.0) AFTER correction!
    sel = getattr(draft, "candidate_selector", None)
    if sel is not None and not getattr(sel, "_vmlx_inkling_post_softcap_patched", False):
        from .models.inkling2.dflash2_kernels import (
            dflash2_markov_select_greedy,
            dflash2_markov_step_scores,
            dflash2_topk,
        )

        def _select(self_s, hidden, logits, anchor_ids, temperature):
            tk = dflash2_topk(logits, int(self_s.top_k))
            if tk is not None:
                unary, candidates = tk
            else:
                candidates = mx.argpartition(logits, -self_s.top_k, axis=-1)[..., -self_s.top_k:]
                unary = mx.take_along_axis(logits, candidates, axis=-1)
            hidden_proj = self_s.hidden_projection(hidden)
            if temperature <= 0:
                fast_path = dflash2_markov_select_greedy(
                    unary,
                    candidates,
                    hidden_proj,
                    self_s.predecessor_codebook.weight,
                    self_s.successor_codebook.weight,
                    anchor_ids,
                    cap,
                )
                if fast_path is not None:
                    return fast_path, candidates, None
            predecessor = anchor_ids
            path, q_rows = [], []
            for position in range(hidden_proj.shape[1]):
                scores = dflash2_markov_step_scores(
                    unary[:, position],
                    candidates[:, position],
                    hidden_proj[:, position],
                    self_s.predecessor_codebook.weight,
                    self_s.successor_codebook.weight,
                    predecessor,
                    cap,
                )
                if scores is None:
                    edges = mx.sum(
                        self_s.predecessor_codebook(predecessor)[:, None]
                        * hidden_proj[:, position, None]
                        * self_s.successor_codebook(candidates[:, position]),
                        axis=-1,
                    )
                    scores = unary[:, position] + edges
                    if cap > 0.0:
                        scores = cap * mx.tanh(scores.astype(mx.float32) / cap)
                if temperature > 0:
                    q = _sampling_probs(scores[:, None], temperature)[:, 0]
                    selected = _sample_probs(q)
                    q_rows.append(q)
                else:
                    selected = mx.argmax(scores, axis=-1)
                predecessor = mx.take_along_axis(
                    candidates[:, position], selected[:, None], axis=-1
                )[:, 0]
                path.append(predecessor)
            return (
                mx.stack(path, axis=1),
                candidates,
                mx.stack(q_rows, axis=1) if q_rows else None,
            )

        sel.select = _select.__get__(sel, type(sel))
        sel._vmlx_inkling_post_softcap_patched = True

    n_fused = install_fused_dflash2_convs(draft)
    if n_fused:
        logger.info("Inkling DFlash2: fused %d GroupedDynamicCausalConv modules into single-dispatch Metal", n_fused)


# ---------------------------------------------------------------------------
# Inkling-2 Target Adapter for DFlash2
# ---------------------------------------------------------------------------
class InklingDFlashTarget:
    """Speculative-decoding target adapter wrapping Inkling-2 LanguageModel."""

    def __init__(self, model: Any, layer_ids) -> None:
        lm = model.language_model if hasattr(model, "language_model") else model
        self.outer = model
        self.lm = lm
        self.inner = lm.model
        self.layer_ids = tuple(int(i) for i in layer_ids)
        self._last_hidden: Optional[mx.array] = None
        self._last_tx: Any = None

    @property
    def model(self):
        return self.inner

    @property
    def lm_head(self):
        return self.lm.lm_head

    def make_cache(self):
        return self.lm.make_cache()

    def forward(
        self,
        ids: mx.array,
        cache,
        *,
        verify: bool = False,
        taps: bool = True,
        last_row: bool = False,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        cap_ids = list(self.layer_ids) if taps else None
        if last_row:
            # Prefill chunk: skip full-sequence lm_head, compute logits ONLY on last row!
            out = self.lm(
                inputs=None if inputs_embeds is not None else ids,
                inputs_embeds=inputs_embeds,
                cache=cache,
                skip_logits=True,
                return_hidden=not taps,
                capture_layer_ids=cap_ids,
                speculative_verify=False,
            )
            self._last_tx = None
            if taps and out.hidden_states:
                self._last_hidden = mx.concatenate(out.hidden_states, axis=-1)
                # Last tap is visit 48 == post-final-norm when 48 in layer_ids!
                if self.layer_ids[-1] == len(self.inner.layer_indices):
                    normed_last = out.hidden_states[-1][:, -1:]
                    return self.lm._logits_from_norm(normed_last)
            # Fallback if 48 wasn't tapped: compute from pre_norm
            if not taps and out.hidden_states:
                pre_norm_last = out.hidden_states[-1][:, -1:]
                return self.lm._logits_from_norm(self.inner.norm(pre_norm_last))
            # Re-run logits only if needed
            out2 = self.lm(
                inputs=None if inputs_embeds is not None else ids[:, -1:],
                inputs_embeds=None if inputs_embeds is None else inputs_embeds[:, -1:],
                cache=None,
                skip_logits=False,
            )
            return out2.logits

        out = self.lm(
            inputs=None if inputs_embeds is not None else ids,
            inputs_embeds=inputs_embeds,
            cache=cache,
            skip_logits=False,
            capture_layer_ids=cap_ids,
            speculative_verify=bool(verify and ids.shape[1] > 1),
        )
        self._last_tx = out.gdn_states if (verify and ids.shape[1] > 1) else None
        self._last_hidden = mx.concatenate(out.hidden_states, axis=-1) if (taps and out.hidden_states) else None
        return out.logits

    def hidden(self) -> mx.array:
        assert self._last_hidden is not None
        return self._last_hidden

    def commit_verify(self, cache, accepted: int, block_size: int) -> None:
        """Atomically commit accepted + 1 rows (anchor + accepted drafts) and roll back rejected rows."""
        tx, self._last_tx = self._last_tx, None
        if tx is not None:
            self.lm.rollback_speculative_cache(cache, tx, int(accepted), int(block_size))
        else:
            rejected = int(block_size) - int(accepted) - 1
            if rejected > 0:
                for c in cache:
                    if hasattr(c, "trim"):
                        c.trim(rejected)


# ---------------------------------------------------------------------------
# Lightweight RAM prefix store for Inkling multi-turn prefix reuse
# ---------------------------------------------------------------------------
def _detach_array(val: Optional[mx.array]) -> Optional[mx.array]:
    if val is None:
        return None
    # Materialize an independent buffer so in-place KVCache / RotatingKVCache
    # slice updates during subsequent decode/verify cannot mutate cached snapshots.
    return 1 * val


def _collect_cache_arrays(cache_obj: Any, out: list[mx.array]) -> None:
    if cache_obj is None:
        return
    if hasattr(cache_obj, "caches"):
        for sub in cache_obj.caches:
            _collect_cache_arrays(sub, out)
        return
    if getattr(cache_obj, "keys", None) is not None:
        out.append(cache_obj.keys)
    if getattr(cache_obj, "values", None) is not None:
        out.append(cache_obj.values)
    raw = getattr(cache_obj, "cache", None)
    if isinstance(raw, list):
        for a in raw:
            if a is not None:
                out.append(a)
    if getattr(cache_obj, "lengths", None) is not None:
        out.append(cache_obj.lengths)
    if getattr(cache_obj, "left_padding", None) is not None:
        out.append(cache_obj.left_padding)


def _clone_inkling_cache_obj(c: Any) -> Any:
    from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache

    if c is None:
        return None
    if isinstance(c, CacheList):
        return CacheList(*[_clone_inkling_cache_obj(sub) for sub in c.caches])
    if isinstance(c, RotatingKVCache):
        out = RotatingKVCache(max_size=int(c.max_size), keep=int(c.keep))
        out.offset = int(c.offset)
        if c.keys is not None and c.values is not None:
            k, v = c.state
            out.keys = _detach_array(k)
            out.values = _detach_array(v)
            # When offset < buffer capacity, c.state trims trailing slack to offset;
            # keep _idx aligned to the trimmed buffer tail. Once wrapped, c.state
            # preserves the full ring buffer so _idx is copied verbatim.
            out._idx = int(k.shape[2]) if int(c.offset) < int(c.keys.shape[2]) else int(c._idx)
        else:
            out._idx = int(c._idx)
        if "_vmlx_base" in c.__dict__:
            out._vmlx_base = int(c.__dict__["_vmlx_base"])
        return out
    if isinstance(c, KVCache):
        out = KVCache()
        if c.keys is not None and c.values is not None:
            k, v = c.state
            out.keys = _detach_array(k)
            out.values = _detach_array(v)
            out.offset = int(c.offset)
        return out
    if isinstance(c, ArraysCache):
        out = ArraysCache(len(c.cache))
        out.cache = [_detach_array(a) for a in c.cache]
        if getattr(c, "lengths", None) is not None:
            out.lengths = _detach_array(c.lengths)
        if getattr(c, "left_padding", None) is not None:
            out.left_padding = _detach_array(c.left_padding)
        return out
    raise TypeError(f"unexpected Inkling cache class: {type(c).__name__}")


def _clone_inkling_caches(caches: list[Any]) -> list[Any]:
    cloned = [_clone_inkling_cache_obj(c) for c in caches]
    arrays: list[mx.array] = []
    for c in cloned:
        _collect_cache_arrays(c, arrays)
    if arrays:
        mx.eval(*arrays)
    return cloned


class _InklingPrefixEntry:
    __slots__ = ("tokens", "cache", "hidden", "logits")

    def __init__(self, tokens: list[int], cache, hidden: mx.array, logits: mx.array):
        self.tokens = tokens
        self.cache = cache
        self.hidden = hidden
        self.logits = logits


class _InklingPrefixStore:
    def __init__(self, max_entries: int = 4) -> None:
        self._entries: list[_InklingPrefixEntry] = []
        self._max = max_entries
        self.last_source = ""

    def lookup(self, prompt_ids: list[int]):
        best = None
        best_len = 0
        n = len(prompt_ids)
        for e in self._entries:
            m = len(e.tokens)
            if m <= n and m > best_len and prompt_ids[:m] == e.tokens:
                best = e
                best_len = m
        if best is not None:
            self.last_source = "ram"
            return best_len, best
        self.last_source = ""
        return None

    def put(self, tokens: list[int], cache, hidden: mx.array, logits: mx.array) -> None:
        # Replace exact match or evict oldest
        self._entries = [e for e in self._entries if e.tokens != tokens]
        self._entries.append(_InklingPrefixEntry(list(tokens), _clone_inkling_caches(cache), hidden, logits))
        if len(self._entries) > self._max:
            self._entries.pop(0)


def target_for(model: Any, layer_ids) -> InklingDFlashTarget:
    t = getattr(model, "_vmlx_inkling_dflash_target", None)
    tids = tuple(int(i) for i in layer_ids)
    if t is None or t.layer_ids != tids:
        t = InklingDFlashTarget(model, tids)
        object.__setattr__(model, "_vmlx_inkling_dflash_target", t)
    return t


def _prefix_store_for(target: InklingDFlashTarget) -> _InklingPrefixStore:
    s = getattr(target.lm, "_vmlx_inkling_prefix_store", None)
    if s is None:
        s = _InklingPrefixStore(max_entries=4)
        object.__setattr__(target.lm, "_vmlx_inkling_prefix_store", s)
    return s


def _adaptive_costs_for(target: InklingDFlashTarget) -> dict:
    c = getattr(target.lm, "_vmlx_inkling_adaptive_costs", None)
    if c is None:
        c = {}
        object.__setattr__(target.lm, "_vmlx_inkling_adaptive_costs", c)
    return c


def _prefill(
    target: InklingDFlashTarget,
    prompt: mx.array,
    cache,
    step: int,
    hidden_limit: Optional[int],
    *,
    taps: bool = True,
    embeds: Optional[mx.array] = None,
):
    from .utils.generation_cancellation import raise_if_generation_cancelled

    chunks, start, logits = [], 0, None
    total = int(prompt.size)
    while start < total:
        raise_if_generation_cancelled()
        end = min(total, start + step)
        is_last = end == total
        seg = None if embeds is None else embeds[:, start:end]
        if is_last:
            logits = target.forward(prompt[None, start:end], cache, verify=False, taps=taps, last_row=True, inputs_embeds=seg)
        else:
            # Intermediate chunks skip lm_head completely
            target.lm(
                inputs=None if seg is not None else prompt[None, start:end],
                inputs_embeds=seg,
                cache=cache,
                skip_logits=True,
                capture_layer_ids=list(target.layer_ids) if taps else None,
                speculative_verify=False,
            )
            # Note: capture_layer_ids result isn't kept if we didn't read out.hidden_states —
            # call target.forward with last_row=True only when we need taps or last chunk!
        if taps and not is_last:
            # Re-check: if chunk is within the drafter's trailing sliding window, capture taps cleanly
            pass
        start = end
    return logits, None


def _prefill_with_taps(
    target: InklingDFlashTarget,
    prompt: mx.array,
    cache,
    step: int,
    hidden_limit: Optional[int],
    *,
    taps: bool = True,
    embeds: Optional[mx.array] = None,
):
    """Chunked prefill capturing drafter taps ONLY for the trailing ``hidden_limit`` window!"""
    from .utils.generation_cancellation import raise_if_generation_cancelled

    chunks: list[mx.array] = []
    start, logits = 0, None
    total = int(prompt.size)
    tap_start = max(0, total - int(hidden_limit)) if (taps and hidden_limit is not None) else 0
    while start < total:
        raise_if_generation_cancelled()
        end = min(total, start + step)
        is_last = end == total
        need_taps = taps and (end > tap_start)
        seg = None if embeds is None else embeds[:, start:end]
        if is_last:
            logits = target.forward(
                prompt[None, start:end], cache, verify=False, taps=need_taps, last_row=True, inputs_embeds=seg
            )
            if need_taps:
                h = target.hidden()
                chunks.append(h)
        else:
            out = target.lm(
                inputs=None if seg is not None else prompt[None, start:end],
                inputs_embeds=seg,
                cache=cache,
                skip_logits=True,
                capture_layer_ids=list(target.layer_ids) if need_taps else None,
                speculative_verify=False,
            )
            if need_taps and out.hidden_states:
                chunks.append(mx.concatenate(out.hidden_states, axis=-1))
        if chunks and hidden_limit is not None:
            cat = chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=1)
            chunks = [cat[:, -hidden_limit:]]
        mx.eval([c.state for c in cache], *(chunks[-1:] if chunks else []))
        start = end
    hidden = None
    if taps and chunks:
        hidden = chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=1)
    return logits, hidden


def generate_inkling_ar(
    target: InklingDFlashTarget,
    prompt_ids,
    max_tokens: int,
    eos: set,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    prefill_step: int = 1024,
    stats: Optional[SpecStats] = None,
) -> Iterator[int]:
    """Plain autoregressive control on the exact same Inkling-2 forward path (for A/B & parity)."""
    from dflash.model_mlx import make_sampler

    sampler = make_sampler(temperature, top_p, top_k)
    prompt = mx.array(prompt_ids, dtype=mx.int32)
    cache = target.make_cache()
    st = stats if stats is not None else SpecStats()
    t0 = time.perf_counter()
    logits, _ = _prefill_with_taps(target, prompt, cache, prefill_step, None, taps=False)
    tok = sampler(logits[:, -1:])
    mx.eval(tok)
    st.prompt_tokens = int(prompt.size)
    st.prefill_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    n = 0
    while True:
        if n + 1 < max_tokens:
            logits = target.forward(tok.reshape(1, 1), cache, verify=False, taps=False)
            nxt = sampler(logits[:, -1:])
            mx.async_eval(nxt)
        t = int(tok.item())
        n += 1
        yield t
        if t in eos or n >= max_tokens:
            break
        tok = nxt
    st.tokens = n
    st.decode_s = time.perf_counter() - t1


def _inkling_dflash2_rounds(
    target: InklingDFlashTarget,
    draft: Any,
    prompt_ids: list[int],
    max_tokens: int,
    eos: set,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    block_size: Optional[int] = None,
    prefill_step: int = 1024,
    stats: Optional[SpecStats] = None,
    *,
    controls=None,
    adaptive: bool = True,
    prefix_store: Optional[_InklingPrefixStore] = None,
    key_ids: Optional[list[int]] = None,
    media: Optional[dict] = None,
) -> Iterator[tuple[list[int], int]]:
    import dflash.model_mlx as runtime
    from dflash.model_mlx import (
        DFlash2DraftModel,
        _rejection_sample,
        _sample_probs,
        _sampling_probs,
        _trim_recent_cache,
        make_sampler,
    )

    configure_inkling_dflash2(draft, target.lm)

    sampler = make_sampler(temperature, top_p, top_k)
    bs0 = int(block_size or draft.config.block_size)
    mask_id = int(draft.config.mask_token_id)
    prompt = mx.array(prompt_ids, dtype=mx.int32)
    st = stats if stats is not None else SpecStats()
    dcache = draft.make_cache()
    hidden_limit = (
        int(draft.config.sliding_window) - 1
        if all(t == "sliding_attention" for t in draft.config.layer_types)
        else None
    )

    t0 = time.perf_counter()
    lookup_ids = [int(t) for t in (key_ids if key_ids is not None else prompt_ids)]
    store_limit = None
    if media is not None and key_ids is None:
        store_limit = int(media.get("first_media_index", 0))
    media_end = int(media.get("media_end", 0)) if media is not None else 0

    hit = None
    if prefix_store is not None:
        hit = prefix_store.lookup(lookup_ids if store_limit is None else lookup_ids[:store_limit])
    reused, cache, snap = 0, None, None
    if hit is not None:
        reused, snap = hit
        cache = _clone_inkling_caches(snap.cache)
    if cache is None:
        cache = target.make_cache()

    compute_t0 = time.perf_counter()
    if reused == int(prompt.size) and snap is not None:
        logits, hidden = snap.logits, snap.hidden
    else:
        embeds = None
        if media is not None and reused < media_end:
            if media.get("embeds") is None:
                media.update(media["materialize"]())
            embeds = media["embeds"]
        st.vision_tower = media is not None and embeds is not None
        seg = embeds[:, reused:] if (embeds is not None and reused < media_end) else None
        logits, h = _prefill_with_taps(
            target, prompt[reused:], cache, prefill_step, hidden_limit, taps=True, embeds=seg
        )
        prev_h = snap.hidden if (reused and snap is not None) else None
        hidden = h if prev_h is None else (mx.concatenate([prev_h, h], axis=1) if h is not None else prev_h)
        if hidden is not None and hidden_limit is not None:
            hidden = hidden[:, -hidden_limit:]

    st.cached_tokens = reused
    st.cache_source = prefix_store.last_source if (prefix_store is not None and reused) else ""
    if prefix_store is not None and store_limit is None and hidden is not None:
        to_eval = [logits, hidden]
        _collect_cache_arrays(cache if not isinstance(cache, list) else None, to_eval)
        if isinstance(cache, list):
            for c in cache:
                _collect_cache_arrays(c, to_eval)
        mx.eval(*to_eval)
        prefix_store.put(lookup_ids, cache, hidden, logits[:, -1:])

    for c in dcache:
        c.offset = int(prompt.size) - int(hidden.shape[1])

    history = list(prompt_ids)
    first = logits[:, -1:]
    if controls is not None:
        first = controls.process(first, history)
        tok = (
            _sample_probs(controls.probabilities(runtime, first, temperature, top_p, top_k))
            if temperature > 0
            else mx.argmax(first, axis=-1)
        )
    else:
        tok = sampler(first)
    mx.eval(tok, hidden)

    st.prompt_tokens = int(prompt.size)
    st.prefill_s = time.perf_counter() - t0
    st.prefill_compute_s = time.perf_counter() - compute_t0

    t1 = time.perf_counter()
    tokens = [int(tok.item())]
    n = 1
    st.output_token_ids = tokens

    try:
        yield [tokens[-1]], 0
        if tokens[-1] in eos:
            return

        # Rule 0 (No Hard Draft Depth): bundle block_size is only the starting rung;
        # adaptive freely climbs above bs0 (up to 16) when acceptance stays high and
        # drops to smaller widths or AR when acceptance is low. Seed prior_p=0.78 / slope=0.20
        # matching Inkling's measured L=1..8 verify curve so round 1 starts near bs0 and
        # immediately observes acceptance across positions 1..bs0-1.
        policy = (
            AdaptiveSpec(
                max(bs0, 16),
                prior_p=0.78,
                slope=0.20,
                explore_every=4,
                costs=_adaptive_costs_for(target),
            )
            if adaptive
            else None
        )

        def sample_row(lg, hist):
            last = lg[:, -1:]
            if controls is not None:
                last = controls.process(last, hist)
                return (
                    _sample_probs(controls.probabilities(runtime, last, temperature, top_p, top_k))
                    if temperature > 0
                    else mx.argmax(last, axis=-1)
                )
            return sampler(last)

        def ar_burst(k: int):
            emitted, rows = [], []
            pipelined = controls is None
            lg = target.forward(mx.array([[tokens[-1]]], dtype=mx.int32), cache, verify=False, taps=True)
            h = target.hidden()
            nx = sample_row(lg, history + tokens)
            mx.async_eval(nx, h)
            for j in range(k):
                launched = None
                if pipelined and j + 1 < k:
                    lg2 = target.forward(nx.reshape(1, 1), cache, verify=False, taps=True)
                    h2 = target.hidden()
                    nx2 = sample_row(lg2, None)
                    mx.async_eval(nx2, h2)
                    launched = (nx2, h2)
                t = int(nx.item())
                emitted.append(t)
                rows.append(h)
                if t in eos or n + len(emitted) >= max_tokens or j + 1 >= k:
                    break
                if launched is None:
                    lg = target.forward(nx.reshape(1, 1), cache, verify=False, taps=True)
                    h = target.hidden()
                    nx = sample_row(lg, history + tokens + emitted)
                    mx.async_eval(nx, h)
                else:
                    nx, h = launched
            return emitted, rows

        after_ar = False
        while n < max_tokens:
            if policy is not None:
                if policy.t_ar is not None and not policy.t_round:
                    # First speculative round after AR warm-up: start at bundle's bs0 so
                    # acceptance is immediately measured across all positions 1..bs0-1.
                    mode, b = ("spec", bs0)
                else:
                    mode, b = policy.choose()
            else:
                mode, b = ("spec", bs0)
            if mode == "ar":
                after_ar = True
                ta = time.perf_counter()
                new, rows = ar_burst(min(b, max_tokens - n))
                policy.update_ar((time.perf_counter() - ta) / max(len(new), 1), len(new))
                hidden = mx.concatenate([hidden] + rows, axis=1)
                if hidden_limit is not None and hidden.shape[1] > hidden_limit:
                    drop = int(hidden.shape[1]) - hidden_limit
                    hidden = hidden[:, drop:]
                    for c in dcache:
                        c.offset += drop
                st.ar_steps += len(new)
                stop_i = next((i for i, t in enumerate(new) if t in eos), None)
                if stop_i is not None:
                    new = new[: stop_i + 1]
                tokens.extend(new)
                n += len(new)
                yield new, 0
                if stop_i is not None:
                    break
                continue

            bs = min(b, max_tokens - n + 1)
            if bs <= 1:
                break
            tr = time.perf_counter()
            block = mx.array([[tokens[-1]] + [mask_id] * (bs - 1)], dtype=mx.int32)
            if isinstance(draft, DFlash2DraftModel):
                d_tok, d_idx, d_prob = draft.propose(block, hidden, dcache, temperature, logits_start=1)
            else:
                dl = draft(block, hidden, dcache, logits_start=1)
                d_idx = None
                if temperature > 0:
                    d_prob = _sampling_probs(dl, temperature, top_p, top_k)
                    d_tok = _sample_probs(d_prob)
                else:
                    d_prob = None
                    d_tok = mx.argmax(dl, axis=-1)
            if (trim_n := int(dcache[0].offset) - (int(prompt.size) + n - 1)) > 0:
                _trim_recent_cache(dcache, trim_n)
            mx.async_eval(d_tok)

            verify_in = mx.concatenate([mx.array([[tokens[-1]]], dtype=mx.int32), d_tok], axis=1)
            logits = target.forward(verify_in, cache, verify=True, taps=True)
            hidden = target.hidden()
            if controls is not None:
                logits = controls.process(logits, history + tokens, d_tok)
            if temperature > 0:
                t_prob = (
                    controls.probabilities(runtime, logits, temperature, top_p, top_k)
                    if controls is not None
                    else _sampling_probs(logits, temperature, top_p, top_k)
                )
                mx.async_eval(t_prob, hidden)
            else:
                t_tok = mx.argmax(logits, axis=-1)
                mx.async_eval(t_tok, hidden)

            dl_ = d_tok[0].tolist()
            if temperature > 0:
                accepted, bonus = _rejection_sample(d_tok, t_prob, d_prob, d_idx)
            else:
                tl_ = t_tok[0].tolist()
                accepted = next((i for i in range(len(dl_)) if dl_[i] != tl_[i]), len(dl_))
                bonus = tl_[accepted]

            new = (dl_[:accepted] + [bonus])[: max_tokens - n]
            st.rounds += 1
            st.accepted.append(accepted)
            st.blocks.append(bs)
            if policy is not None:
                policy.update_spec(bs, accepted, time.perf_counter() - tr, after_ar=after_ar)
            after_ar = False

            stop_i = next((i for i, t in enumerate(new) if t in eos), None)
            if stop_i is not None:
                new = new[: stop_i + 1]
            tokens.extend(new)
            n += len(new)
            yield new, bs - 1
            if stop_i is not None:
                break
            target.commit_verify(cache, accepted, bs)
            hidden = hidden[:, : accepted + 1, :]
    finally:
        st.tokens = n
        st.decode_s = time.perf_counter() - t1


def generate_inkling_dflash2(
    target: InklingDFlashTarget,
    draft: Any,
    prompt_ids: list[int],
    max_tokens: int,
    eos: set,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    block_size: Optional[int] = None,
    prefill_step: int = 1024,
    stats: Optional[SpecStats] = None,
    adaptive: bool = True,
    prefix_store: Optional[_InklingPrefixStore] = None,
) -> Iterator[int]:
    """DFlash2 speculative generation for Inkling-2 as a token iterator."""
    for new, _ in _inkling_dflash2_rounds(
        target, draft, prompt_ids, max_tokens, eos,
        temperature=temperature, top_p=top_p, top_k=top_k,
        block_size=block_size, prefill_step=prefill_step,
        stats=stats, adaptive=adaptive, prefix_store=prefix_store,
    ):
        yield from new


def stream_inkling_dflash2(
    model: Any,
    tokenizer: Any,
    draft: Any,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
    stop=None,
    prompt_tokens=None,
    media=None,
    sampling_controls=None,
    use_cache: bool = True,
):
    """Server lane: yields ``dflash.model_mlx.GenerationResponse`` chunks for an Inkling-2 target."""
    from dflash.model_mlx import GenerationResponse
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from .utils.prefill_step import DEFAULT_PREFILL_STEP_SIZE, prefill_step_size

    tw = tokenizer if isinstance(tokenizer, TokenizerWrapper) else TokenizerWrapper(tokenizer)
    if prompt_tokens is None:
        bos = getattr(tw, "bos_token", None)
        add_special = not (bos and prompt.startswith(bos))
        prompt_tokens = tw.encode(prompt, add_special_tokens=add_special)

    eos = set(getattr(tw, "eos_token_ids", None) or ())
    cfg_eos = getattr(getattr(model, "config", None), "eos_token_id", None)
    if isinstance(cfg_eos, int):
        eos.add(cfg_eos)
    elif isinstance(cfg_eos, (list, tuple, set)):
        eos.update(int(x) for x in cfg_eos)

    detok = tw.detokenizer
    detok.reset()
    if stop:
        from .dflash2_runtime import _StopDetokenizer
        detok = _StopDetokenizer(detok, [stop] if isinstance(stop, str) else list(stop))

    target = target_for(model, draft.config.target_layer_ids)
    st = SpecStats()
    tic = time.perf_counter()
    n = 0
    store = (
        _prefix_store_for(target)
        if use_cache and os.environ.get("VMLX_INKLING_DFLASH2_PREFIX_CACHE", "1") != "0"
        else None
    )
    # Default Inkling prefill chunk is 1024 (upgraded from 512 with zero-mask MPP FlashAttention),
    # while still honoring any explicit user --prefill-step-size override.
    step = prefill_step_size()
    if step == DEFAULT_PREFILL_STEP_SIZE:
        step = 1024

    for new, drafted in _inkling_dflash2_rounds(
        target,
        draft,
        list(prompt_tokens),
        int(max_tokens),
        eos,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        prefill_step=step,
        stats=st,
        controls=sampling_controls,
        adaptive=os.environ.get("VMLX_INKLING_DFLASH2_ADAPTIVE", "1") != "0",
        prefix_store=store,
        key_ids=media.get("key_tokens") if media is not None else None,
        media=media,
    ):
        if n == 0:
            tic = time.perf_counter()
        for t in new:
            if t not in eos:
                detok.add_token(t)
        n += len(new)
        finished = new[-1] in eos or n >= max_tokens or getattr(detok, "matched", False)
        reason = None
        if finished:
            detok.finalize()
            reason = "stop" if (new[-1] in eos or getattr(detok, "matched", False)) else "length"
        r = GenerationResponse(
            detok.last_segment,
            list(new),
            len(new),
            st.prompt_tokens,
            st.prompt_tokens / max(st.prefill_s, 1e-9),
            n,
            n / max(time.perf_counter() - tic, 1e-9),
            mx.get_peak_memory() / 1e9,
            reason,
        )
        r.drafted = int(drafted)
        r.cached_tokens = int(st.cached_tokens)
        r.prefill_usage = {
            "tokens": int(st.prompt_tokens) - int(st.cached_tokens),
            "seconds": float(st.prefill_compute_s),
            "scope": "model_prefill_and_prompt_state",
            "path": "inkling_dflash2" + ("_with_vision" if st.vision_tower else ""),
        }
        if finished:
            r.output_token_ids = list(st.output_token_ids)
        r.cache_detail = {"ram": "inkling-dflash2-prefix"}.get(st.cache_source, "")
        yield r
        if finished:
            return
