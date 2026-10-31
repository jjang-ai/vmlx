# SPDX-License-Identifier: Apache-2.0
"""Sampling helpers for vMLX generation paths."""

from __future__ import annotations

import math
from functools import partial
from typing import Callable

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler as _mlx_make_sampler


class _SamplerWrapper:
    def __init__(
        self,
        sampler: Callable[[mx.array], mx.array],
        *,
        accepts_logits: bool = False,
        is_greedy: bool = False,
    ):
        self._sampler = sampler
        if accepts_logits:
            self._vmlx_accepts_logits = True
        if is_greedy:
            self._vmlx_is_greedy = True

    def __call__(self, logits: mx.array) -> mx.array:
        return self._sampler(logits)

    def __getattr__(self, name):
        return getattr(self._sampler, name)


def _stamp_sampling_contract(
    sampler: Callable[[mx.array], mx.array],
    *,
    temp: float,
    top_p: float,
    min_p: float,
    top_k: int,
) -> Callable[[mx.array], mx.array]:
    """Publish the distribution contract consumed by speculative verify.

    Compact samplers consume raw logits, but speculative acceptance still
    needs to reconstruct the exact filtered distribution they sampled.  Keep
    that metadata on every sampler instead of inferring it from whether the
    callable consumes logits or log-probabilities.
    """

    sampler.temp = float(temp or 0.0)
    sampler.top_p = float(top_p or 0.0)
    sampler.min_p = float(min_p or 0.0)
    sampler.top_k = int(top_k or 0)
    return sampler


def _compact_top_k_acceptance_logprobs(
    logprobs: mx.array,
    *,
    temp: float,
    top_p: float,
    top_k: int,
) -> mx.array:
    """Rebuild the exact full-vocabulary distribution of compact top-k.

    The sampler selects top-k first, applies its nucleus cutoff against the
    original full-distribution mass, temperature-scales the surviving compact
    rows, and only then normalizes through categorical sampling.  Speculative
    acceptance and residual correction need that same distribution, expanded
    back to the original vocabulary indices.
    """

    vocab_size = int(logprobs.shape[-1])
    k = min(int(top_k), vocab_size)
    top_indices = mx.argpartition(-logprobs, kth=k - 1, axis=-1)[..., :k]
    top_values = mx.take_along_axis(logprobs, top_indices, axis=-1)
    order = mx.argsort(-top_values, axis=-1)
    top_values = mx.take_along_axis(top_values, order, axis=-1)
    top_indices = mx.take_along_axis(top_indices, order, axis=-1)
    if 0.0 < float(top_p) < 1.0:
        probabilities = mx.exp(top_values)
        cumulative = mx.cumsum(probabilities, axis=-1)
        keep = (cumulative - probabilities) < float(top_p)
        top_values = mx.where(keep, top_values, -float("inf"))
    scaled = top_values * (1.0 / float(temp))
    compact = scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)
    full = mx.full(logprobs.shape, -float("inf"), dtype=logprobs.dtype)
    return mx.put_along_axis(full, top_indices, compact, axis=-1)


def _stamp_compact_top_k_contract(
    sampler: Callable[[mx.array], mx.array],
    *,
    temp: float,
    top_p: float,
    top_k: int,
) -> Callable[[mx.array], mx.array]:
    sampler._vmlx_acceptance_logprobs = partial(
        _compact_top_k_acceptance_logprobs,
        temp=float(temp),
        top_p=float(top_p or 0.0),
        top_k=int(top_k),
    )
    # Only this pure filter contract permits evaluating unused proposal rows.
    # Identity also invalidates admission if a caller replaces the hook.
    sampler._vmlx_acceptance_batch_contract = sampler._vmlx_acceptance_logprobs
    return sampler


def make_sampler(
    *,
    temp: float = 0.0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    top_k: int = 0,
    seed: int | None = None,
) -> Callable[[mx.array], mx.array]:
    """Build a sampler for vMLX decode loops.

    MLX-LM's default top-k sampler masks a full-vocabulary logits vector and
    then calls categorical sampling across that full vocabulary. That is
    correct but expensive for vMLX's common app defaults on large-vocab models
    such as ZAYA (top_k=40, vocab > 260k). For bounded top-k sampling, gather
    the compact top-k logits and sample there, then map the sampled local index
    back to the original token id.
    """

    if float(temp or 0.0) == 0.0:
        return _stamp_sampling_contract(
            _SamplerWrapper(
                lambda x: mx.argmax(x, axis=-1),
                accepts_logits=True,
                is_greedy=True,
            ),
            temp=0.0,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
        )

    if seed is not None:
        if int(top_k or 0) > 0 and float(min_p or 0.0) == 0.0:
            return _stamp_sampling_contract(
                _stamp_compact_top_k_contract(
                    _SamplerWrapper(
                        _make_seeded_compact_top_k_sampler(
                            temp=float(temp),
                            top_p=float(top_p or 0.0),
                            top_k=int(top_k),
                            seed=int(seed),
                        ),
                        accepts_logits=True,
                    ),
                    temp=temp,
                    top_p=top_p,
                    top_k=top_k,
                ),
                temp=temp,
                top_p=top_p,
                min_p=min_p,
                top_k=top_k,
            )
        return _stamp_sampling_contract(
            _make_seeded_logprob_sampler(
                temp=float(temp),
                top_p=float(top_p or 0.0),
                min_p=float(min_p or 0.0),
                top_k=int(top_k or 0),
                seed=int(seed),
            ),
            temp=temp,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
        )

    if int(top_k or 0) > 0 and float(min_p or 0.0) == 0.0:
        return _stamp_sampling_contract(
            _stamp_compact_top_k_contract(
                _SamplerWrapper(
                    _make_compact_top_k_sampler(
                        temp=float(temp),
                        top_p=float(top_p or 0.0),
                        top_k=int(top_k),
                    ),
                    accepts_logits=True,
                ),
                temp=temp,
                top_p=top_p,
                top_k=top_k,
            ),
            temp=temp,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
        )

    return _stamp_sampling_contract(
        _mlx_make_sampler(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k),
        temp=temp,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
    )


def make_minimax_m3_sampler(
    *,
    temp: float = 0.0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    top_k: int = 0,
    seed: int | None = None,
) -> Callable[[mx.array], mx.array]:
    """Build the MiniMax-M3 server sampler to match the clean runtime path.

    The standalone M3 runtime samples from raw logits after casting them to
    fp32.  The generic MLX-LM sampler expects log-probabilities and applies a
    different nucleus implementation.  M3's low-bit routed experts make that
    difference visible, so the server path must use the same logit-space sampler
    as ``models/minimax_m3/runtime.py``.
    """

    temp_f = float(temp or 0.0)
    top_p_f = float(top_p if top_p is not None else 1.0)
    top_k_i = int(top_k or 0)
    min_p_f = float(min_p or 0.0)

    key = mx.random.key(int(seed)) if seed is not None else None

    def _categorical(logits: mx.array) -> mx.array:
        nonlocal key
        if key is None:
            return mx.random.categorical(logits)
        keys = mx.random.split(key)
        key = keys[0]
        return mx.random.categorical(logits, key=keys[1])

    def _sample(logits: mx.array) -> mx.array:
        logits = logits.astype(mx.float32)
        if temp_f == 0.0:
            return mx.argmax(logits, axis=-1)

        logits = logits / temp_f
        if top_k_i > 0:
            vocab_size = logits.shape[-1]
            k = min(top_k_i, vocab_size)
            top_indices = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
            top_logits = mx.take_along_axis(logits, top_indices, axis=-1)
            order = mx.argsort(-top_logits, axis=-1)
            logits = mx.take_along_axis(top_logits, order, axis=-1)
            top_indices = mx.take_along_axis(top_indices, order, axis=-1)
        else:
            top_indices = None

        if 0.0 < min_p_f < 1.0:
            # min_p: keep tokens whose probability is at least min_p x the top token's, i.e. (after temperature)
            # logit >= max logit + log(min_p); the top token always survives.
            floor = logits.max(axis=-1, keepdims=True) + math.log(min_p_f)
            logits = mx.where(logits >= floor, logits, -mx.inf)

        if top_p_f > 0.0 and top_p_f < 1.0:
            probs = mx.softmax(logits, axis=-1)
            order = mx.argsort(-probs, axis=-1)
            sorted_probs = mx.take_along_axis(probs, order, axis=-1)
            cumulative = mx.cumsum(sorted_probs, axis=-1)
            keep = (cumulative - sorted_probs) <= top_p_f
            sorted_probs = mx.where(keep, sorted_probs, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(axis=-1, keepdims=True)
            sampled_sorted = _categorical(mx.log(sorted_probs + 1e-12))
            sampled = mx.take_along_axis(
                order,
                sampled_sorted[..., None],
                axis=-1,
            ).squeeze(-1)
        else:
            sampled = _categorical(logits)

        if top_indices is not None:
            sampled = mx.take_along_axis(top_indices, sampled[..., None], axis=-1).squeeze(-1)
        return sampled

    wrapper = _SamplerWrapper(_sample, accepts_logits=True)
    wrapper._vmlx_sampler_kind = "minimax_m3_runtime"
    return wrapper


def _make_seeded_logprob_sampler(
    *, temp: float, top_p: float, min_p: float, top_k: int, seed: int
) -> Callable[[mx.array], mx.array]:
    """Create a request-local sampler backed by an independent MLX PRNG key."""
    from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p

    key = mx.random.key(seed)

    def _next_key():
        nonlocal key
        keys = mx.random.split(key)
        key = keys[0]
        return keys[1]

    def _categorical(logits: mx.array) -> mx.array:
        return mx.random.categorical(logits, key=_next_key())

    def _uniform() -> float:
        return float(mx.random.uniform(key=_next_key()).item())

    def _sample(logprobs: mx.array) -> mx.array:
        nonlocal key
        if 0.0 < top_p < 1.0:
            logprobs = apply_top_p(logprobs, top_p)
        if min_p != 0.0:
            logprobs = apply_min_p(logprobs, min_p)
        if top_k > 0:
            logprobs = apply_top_k(logprobs, top_k)
        return _categorical(logprobs * (1.0 / temp))

    _sample._vmlx_categorical = _categorical
    _sample._vmlx_random_uniform = _uniform
    return _sample


def _make_seeded_compact_top_k_sampler(
    *, temp: float, top_p: float, top_k: int, seed: int
) -> Callable[[mx.array], mx.array]:
    """Seeded variant of the compact logit-space top-k fast path."""
    key = mx.random.key(seed)

    def _next_key():
        nonlocal key
        keys = mx.random.split(key)
        key = keys[0]
        return keys[1]

    def _categorical(logits: mx.array) -> mx.array:
        return mx.random.categorical(logits, key=_next_key())

    def _uniform() -> float:
        return float(mx.random.uniform(key=_next_key()).item())

    def _sample(logits: mx.array) -> mx.array:
        nonlocal key
        vocab_size = logits.shape[-1]
        k = min(top_k, vocab_size)
        top_indices = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
        top_logits = mx.take_along_axis(logits, top_indices, axis=-1)
        order = mx.argsort(-top_logits, axis=-1)
        top_logits = mx.take_along_axis(top_logits, order, axis=-1)
        top_indices = mx.take_along_axis(top_indices, order, axis=-1)
        if 0.0 < top_p < 1.0:
            probs = mx.exp(top_logits - mx.logsumexp(logits, axis=-1, keepdims=True))
            cumulative = mx.cumsum(probs, axis=-1)
            keep = (cumulative - probs) < top_p
            top_logits = mx.where(keep, top_logits, -float("inf"))
        sampled_local = _categorical(top_logits * (1.0 / temp))
        return mx.take_along_axis(
            top_indices, sampled_local[..., None], axis=-1
        ).squeeze(-1)

    _sample._vmlx_categorical = _categorical
    _sample._vmlx_random_uniform = _uniform
    return _sample


def _make_compact_top_k_sampler(
    *,
    temp: float,
    top_p: float,
    top_k: int,
) -> Callable[[mx.array], mx.array]:
    @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
    def _sample(logits: mx.array) -> mx.array:
        vocab_size = logits.shape[-1]
        k = min(top_k, vocab_size)
        top_indices = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
        top_logits = mx.take_along_axis(logits, top_indices, axis=-1)

        order = mx.argsort(-top_logits, axis=-1)
        top_logits = mx.take_along_axis(top_logits, order, axis=-1)
        top_indices = mx.take_along_axis(top_indices, order, axis=-1)

        if top_p > 0.0 and top_p < 1.0:
            probs = mx.exp(top_logits - mx.logsumexp(logits, axis=-1, keepdims=True))
            cumulative = mx.cumsum(probs, axis=-1)
            # Keep the token that crosses top_p as well as all earlier tokens.
            # This matches the full-vocabulary nucleus cutoff, then applies the
            # top-k bound without sampling over masked-out vocabulary entries.
            keep = (cumulative - probs) < top_p
            top_logits = mx.where(keep, top_logits, -float("inf"))

        sampled_local = mx.random.categorical(top_logits * (1.0 / temp))
        return mx.take_along_axis(top_indices, sampled_local[..., None], axis=-1).squeeze(-1)

    return _sample
