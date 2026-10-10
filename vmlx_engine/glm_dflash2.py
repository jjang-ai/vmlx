"""DFlash2 speculative decoding for GLM-5.3-Flash (glm5_next) targets (2026-10-09).

Drafter: incoai/GLM-5.3-Flash-DFlash2 (z-lab DFlash2: 5 sliding-window qwen3 layers, block 8, two-tap grouped dynamic
conv, candidate selector top-16), loaded with ``dflash.model_mlx.load_draft`` from its ORIGINAL BF16 files and
optionally affine-8 quantized at load (``quantize_draft``); it shares the target's embedding and lm_head (``bind``).

GLM target contract (SGLang PR 36708 ``_prepare_aux_hidden_state`` + its unit test): the drafter's context taps are the
COMPLETED outputs of target layers ``dflash_config.target_layer_ids`` = [5, 14, 24, 33, 42], each contracted over the
4 mHC streams by an unweighted mean (the same contraction GLM applies before its final norm).

Verify / rollback: the verify forward runs [last committed token] + drafts with ``n_confirmed=1``, so every KDA layer
records the exact recurrent + conv state after each position (``Glm5KDACache.set_speculative_states``) and a partial
rejection restores the state after the accepted prefix (``rollback_speculative(rejected)``); MLA/DSA caches trim
(an incomplete indexer pool is dropped and recomputed from the packed history on the next append).
Cache invariant (as upstream dflash): the last emitted token is sampled but never forwarded.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn


class _MeanStreamTap:
    """Wraps one decoder layer; when armed, stores mean over the mHC stream axis of its output: (B, T, 4, D) -> (B, T, D)."""

    def __init__(self, layer, slot: int, store: list, armed: dict):
        self._layer, self._slot, self._store, self._armed = layer, slot, store, armed

    def __call__(self, *args, **kwargs):
        out = self._layer(*args, **kwargs)
        if self._armed["on"]:
            self._store[self._slot] = mx.mean(out, axis=2)
        return out

    def __getattr__(self, name):
        return getattr(self._layer, name)


class GLMDFlashTarget:
    """Speculative-decoding view of a loaded glm5_next model (VLM wrapper or text Model)."""

    def __init__(self, model, layer_ids):
        lm = model.language_model if hasattr(model, "language_model") else model
        self.lm, self.inner = lm, lm.model
        self.layer_ids = tuple(int(i) for i in layer_ids)
        self._store: list = [None] * len(self.layer_ids)
        self._armed = {"on": False}
        layers = self.inner.layers
        for slot, lid in enumerate(self.layer_ids):
            cur = layers[lid]
            if isinstance(cur, _MeanStreamTap):        # re-wrap safely (adapter rebuilt for the same model)
                cur = cur._layer
            layers[lid] = _MeanStreamTap(cur, slot, self._store, self._armed)

    def close(self):
        layers = self.inner.layers
        for lid in self.layer_ids:
            if isinstance(layers[lid], _MeanStreamTap):
                layers[lid] = layers[lid]._layer

    # dflash.bind() looks for .model.embed_tokens and .lm_head
    @property
    def model(self):
        return self.inner

    @property
    def lm_head(self):
        return self.lm.lm_head

    def make_cache(self):
        return self.lm.make_cache()

    def forward(self, ids: mx.array, cache, verify: bool, taps: bool = True, last_row: bool = False):
        self._armed["on"] = taps
        try:
            h = self.inner(ids, cache=cache, n_confirmed=1 if (verify and ids.shape[1] > 1) else 0, return_pre_norm=True)
        finally:
            self._armed["on"] = False
        if last_row:
            h = h[:, -1:]
        return self.lm.lm_head(self.inner.norm(h))

    def hidden(self) -> mx.array:
        return mx.concatenate(self._store, axis=-1)

    @staticmethod
    def rollback(cache, rejected: int) -> None:
        from vmlx_engine.models.glm5_next.glm5_next import Glm5KDACache

        for c in cache:
            if isinstance(c, Glm5KDACache):
                if rejected > 0:
                    if not c.rollback_speculative(rejected):
                        raise RuntimeError("glm dflash2: KDA speculative rollback failed (no recorded states)")
                else:
                    c.commit_speculative()
            elif rejected > 0:
                c.trim(rejected)


def load_glm_drafter(path: str, quantize_bits: int = 8, group_size: int = 64):
    """incoai DFlash2 drafter from its original files; affine `quantize_bits` at load (0 = keep BF16)."""
    from dflash.model_mlx import load_draft

    draft = load_draft(path)
    if quantize_bits:
        quantize_draft(draft, quantize_bits, group_size)
    return draft


def quantize_draft(draft, bits: int = 8, group_size: int = 64) -> int:
    """Affine-quantize the drafter's Linear layers (attention, MLP, fc, selector projection) in place; the selector
    codebooks (embeddings) and the dynamic-conv parameters stay BF16. Returns the number of quantized modules."""
    count = {"n": 0}

    def pred(path, m):
        ok = isinstance(m, nn.Linear) and m.weight.shape[-1] % group_size == 0
        count["n"] += int(ok)
        return ok
    nn.quantize(draft, group_size=group_size, bits=bits, class_predicate=pred)
    mx.eval(draft.parameters())
    return count["n"]


@dataclass
class SpecStats:
    prompt_tokens: int = 0
    prefill_s: float = 0.0
    tokens: int = 0
    decode_s: float = 0.0
    rounds: int = 0
    accepted: list = field(default_factory=list)

    @property
    def tok_per_round(self) -> float:
        return (sum(self.accepted) + len(self.accepted)) / max(1, len(self.accepted))

    @property
    def decode_tps(self) -> float:
        return self.tokens / max(self.decode_s, 1e-9)


def _prefill(target: GLMDFlashTarget, prompt: mx.array, cache, step: int, hidden_limit: int | None, taps: bool = True):
    chunks, start, logits = [], 0, None
    while start < prompt.size:
        rem = prompt.size - start
        end = start + (1 if rem == 1 else min(step, rem - 1)) if taps else start + min(step, rem)
        logits = target.forward(prompt[None, start:end], cache, verify=False, taps=taps, last_row=True)
        if taps:
            h = target.hidden()
            if hidden_limit is not None:
                if chunks:
                    h = mx.concatenate((chunks[0], h), axis=1)
                chunks = [h[:, -hidden_limit:]]
            else:
                chunks.append(h)
        mx.eval([c.state for c in cache], *(chunks[-1:] if chunks else []))
        start = end
    hidden = None
    if taps:
        hidden = chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=1)
    return logits, hidden


def generate_ar(target: GLMDFlashTarget, prompt_ids, max_tokens: int, eos: set, temperature: float = 0.0,
                top_p: float = 1.0, prefill_step: int = 2048, stats: SpecStats | None = None) -> Iterator[int]:
    """Plain autoregressive control on the SAME forward path (no taps), for in-process A/B."""
    from dflash.model_mlx import make_sampler

    sampler = make_sampler(temperature, top_p, 0)
    prompt = mx.array(prompt_ids); cache = target.make_cache(); st = stats or SpecStats()
    t0 = time.perf_counter()
    logits, _ = _prefill(target, prompt, cache, prefill_step, None, taps=False)
    tok = sampler(logits[:, -1:]); mx.eval(tok)
    st.prompt_tokens, st.prefill_s = prompt.size, time.perf_counter() - t0
    t1 = time.perf_counter(); n = 0
    while True:
        t = int(tok.item()); n += 1; yield t
        if t in eos or n >= max_tokens:
            break
        logits = target.forward(tok.reshape(1, 1), cache, verify=False, taps=False)
        tok = sampler(logits[:, -1:]); mx.async_eval(tok)
    st.tokens, st.decode_s = n, time.perf_counter() - t1


def generate_dflash2(target: GLMDFlashTarget, draft, prompt_ids, max_tokens: int, eos: set, temperature: float = 0.0,
                     top_p: float = 1.0, block_size: int | None = None, prefill_step: int = 2048,
                     stats: SpecStats | None = None) -> Iterator[int]:
    """DFlash2 speculative generation (upstream dflash loop, GLM verify/rollback). Greedy output == generate_ar's."""
    from dflash.model_mlx import DFlash2DraftModel, _rejection_sample, _sampling_probs, _trim_recent_cache, make_sampler

    sampler = make_sampler(temperature, top_p, 0)
    bs0 = int(block_size or draft.config.block_size)
    mask_id = int(draft.config.mask_token_id)
    prompt = mx.array(prompt_ids); st = stats or SpecStats()
    cache = target.make_cache(); dcache = draft.make_cache(); draft.bind(target)
    hidden_limit = draft.config.sliding_window - 1 if all(t == "sliding_attention" for t in draft.config.layer_types) else None
    t0 = time.perf_counter()
    logits, hidden = _prefill(target, prompt, cache, prefill_step, hidden_limit)
    for c in dcache:
        c.offset = prompt.size - hidden.shape[1]
    tok = sampler(logits[:, -1:]); mx.eval(tok, hidden)
    st.prompt_tokens, st.prefill_s = prompt.size, time.perf_counter() - t0
    t1 = time.perf_counter()
    tokens = [int(tok.item())]; yield tokens[-1]
    n = 1
    if tokens[-1] in eos:
        st.tokens, st.decode_s = n, time.perf_counter() - t1
        return
    while n < max_tokens:
        bs = min(bs0, max_tokens - n + 1)
        if bs <= 1:
            break
        block = mx.array([[tokens[-1]] + [mask_id] * (bs - 1)])
        if isinstance(draft, DFlash2DraftModel):
            d_tok, d_idx, d_prob = draft.propose(block, hidden, dcache, temperature, logits_start=1)
        else:
            dl = draft(block, hidden, dcache, logits_start=1)
            d_idx = None
            if temperature > 0:
                d_prob = _sampling_probs(dl, temperature, top_p, 0); d_tok = mx.argmax(d_prob, axis=-1)
            else:
                d_prob = None; d_tok = mx.argmax(dl, axis=-1)
        if (trim_n := dcache[0].offset - (prompt.size + n - 1)) > 0:
            _trim_recent_cache(dcache, trim_n)
        mx.async_eval(d_tok)
        verify_in = mx.concatenate([mx.array([[tokens[-1]]]), d_tok], axis=1)
        logits = target.forward(verify_in, cache, verify=True)
        hidden = target.hidden()
        if temperature > 0:
            t_prob = _sampling_probs(logits, temperature, top_p, 0); mx.async_eval(t_prob, hidden)
        else:
            t_tok = mx.argmax(logits, axis=-1); mx.async_eval(t_tok, hidden)
        dl_ = d_tok[0].tolist()
        if temperature > 0:
            accepted, bonus = _rejection_sample(d_tok, t_prob, d_prob, d_idx)
        else:
            tl_ = t_tok[0].tolist()
            accepted = next((i for i in range(len(dl_)) if dl_[i] != tl_[i]), len(dl_))
            bonus = tl_[accepted]
        new = (dl_[:accepted] + [bonus])[: max_tokens - n]
        st.rounds += 1; st.accepted.append(accepted)
        stop = next((i for i, t in enumerate(new) if t in eos), None)
        if stop is not None:
            new = new[: stop + 1]
        for t in new:
            yield t
        tokens.extend(new); n += len(new)
        if stop is not None:
            break
        target.rollback(cache, bs - accepted - 1)
        hidden = hidden[:, : accepted + 1, :]
    st.tokens, st.decode_s = n, time.perf_counter() - t1
