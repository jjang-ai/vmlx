# SPDX-License-Identifier: Apache-2.0
"""ERNIE 5.5 Flash (`ernie5_5_moe`) target adapter for vMLX's shared DSpark speculative loop."""

from __future__ import annotations

from typing import Optional, Sequence

import mlx.core as mx

from ..models.ernie5_5_moe.dspark import rollback_ernie_caches, snapshot_ernie_caches


class ErnieDSparkTarget:
    """Target adapter connecting ERNIE 5.5 Flash LanguageModel to `vmlx_engine.dspark.iter_dspark`.

    - `max_verify_rows = None`: Adaptive governs the draft ladder freely (Rule 0 — no hard depth cap).
    - `ar_step_rollback = True`: snapshots HCA/MLA caches before AR steps so pipelined AR carry can
      roll back bit-exact if a draft round follows.
    """

    max_verify_rows: Optional[int] = None
    ar_step_rollback = True

    def __init__(self, model) -> None:
        if not hasattr(model, "forward_with_taps") or not hasattr(model, "make_cache"):
            raise ValueError("DSpark ERNIE 5.5 target requires an ernie5_5_moe LanguageModel")
        self.model = model
        self.embed_tokens = model.model.embed_tokens
        self.lm_head = model.lm_head
        self._last_ar_snap = None

    def make_cache(self):
        return self.model.make_cache()

    def set_media(self, embeds=None, positions=None, rope_delta: int = 0) -> None:
        if embeds is not None or positions is not None or rope_delta:
            raise ValueError("DSpark ERNIE 5.5 target drafts text requests only")

    def prefill_segment(self, ids: mx.array, cache, taps: Sequence[int], prefill_step: Optional[int]):
        from ..utils.generation_cancellation import raise_if_generation_cancelled

        length = int(ids.shape[1])
        step = max(1, int(prefill_step or length or 1))
        feats = []
        pos = 0
        while pos < length:
            raise_if_generation_cancelled()
            end = min(length, pos + step)
            chunk = ids[:, pos:end]
            if taps:
                _, f = self.model.model(chunk, cache=cache, capture_taps=taps, tail_only=True)
                mx.eval(f)
                feats.append(f)
            else:
                h = self.model.model(chunk, cache=cache, tail_only=True)
                mx.eval(h)
            pos = end
        if not feats:
            return None
        return feats[0] if len(feats) == 1 else mx.concatenate(feats, axis=1)

    def prefill(
        self,
        ids: mx.array,
        cache,
        taps: Sequence[int],
        prefill_step: Optional[int],
        keep_last: Optional[int] = None,
    ):
        """Chunked prefill returning `(last_row_logits (V,), trailing_taps (1, P, n_taps*H) or None)`."""
        from ..utils.generation_cancellation import raise_if_generation_cancelled

        length = int(ids.shape[1])
        step = max(1, int(prefill_step or length or 1))
        tail_from = (length - int(keep_last)) if (keep_last and keep_last < length) else 0
        feats = []
        pos = 0
        logits = None
        while pos < length:
            raise_if_generation_cancelled()
            end = min(length, pos + step)
            is_last = end >= length
            chunk = ids[:, pos:end]
            need_taps = bool(taps) and (is_last or end > tail_from)
            if is_last:
                if need_taps:
                    lg, f = self.model.forward_with_taps(chunk, cache=cache, taps=taps, tail_only=True)
                    feats.append(f)
                else:
                    lg = self.model(chunk, cache=cache, prefill_last_logits_only=True)
                logits = lg[0, -1]
            else:
                if need_taps:
                    _, f = self.model.model(chunk, cache=cache, capture_taps=taps, tail_only=True)
                    feats.append(f)
                    mx.eval(f)
                else:
                    h = self.model.model(chunk, cache=cache, tail_only=True)
                    mx.eval(h)
            pos = end

        merged = None
        if feats:
            merged = feats[0] if len(feats) == 1 else mx.concatenate(feats, axis=1)
            if keep_last and merged.shape[1] > keep_last:
                merged = mx.contiguous(merged[:, -int(keep_last) :, :])
        return logits, merged

    def decode(self, token: mx.array, cache) -> mx.array:
        self._last_ar_snap = snapshot_ernie_caches(cache)
        lg = self.model(token.reshape(1, 1), cache=cache)
        return lg[0, -1]

    def ar_step(self, token: mx.array, cache, taps: Sequence[int]):
        self._last_ar_snap = snapshot_ernie_caches(cache)
        if taps:
            lg, f = self.model.forward_with_taps(token.reshape(1, 1), cache=cache, taps=taps, tail_only=False)
            return lg[0, -1], f
        lg = self.model(token.reshape(1, 1), cache=cache)
        return lg[0, -1], None

    def verify(self, rows: mx.array, cache, taps: Sequence[int]):
        snap = snapshot_ernie_caches(cache)
        if taps:
            lg, f = self.model.forward_with_taps(rows, cache=cache, taps=taps, tail_only=False)
            return lg[0], f, snap
        lg = self.model(rows, cache=cache)
        return lg[0], None, snap

    def rollback(self, cache, sink, kept: int, total: int) -> None:
        snap = sink if sink is not None else self._last_ar_snap
        if snap is None:
            raise RuntimeError("ErnieDSparkTarget.rollback called without a cache snapshot")
        rollback_ernie_caches(cache, snap, kept=kept, block_len=total)
        if sink is None:
            self._last_ar_snap = None
