# SPDX-License-Identifier: Apache-2.0
"""Sampling helpers for vMLX generation paths."""

from __future__ import annotations

from functools import partial
from typing import Callable

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler as _mlx_make_sampler


class _SamplerWrapper:
    def __init__(self, sampler: Callable[[mx.array], mx.array], *, accepts_logits: bool = False):
        self._sampler = sampler
        if accepts_logits:
            self._vmlx_accepts_logits = True

    def __call__(self, logits: mx.array) -> mx.array:
        return self._sampler(logits)


def make_sampler(
    *,
    temp: float = 0.0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    top_k: int = 0,
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
        return _SamplerWrapper(lambda x: mx.argmax(x, axis=-1), accepts_logits=True)

    if int(top_k or 0) > 0 and float(min_p or 0.0) == 0.0:
        return _SamplerWrapper(
            _make_compact_top_k_sampler(
                temp=float(temp),
                top_p=float(top_p or 0.0),
                top_k=int(top_k),
            ),
            accepts_logits=True,
        )

    return _mlx_make_sampler(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k)


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
