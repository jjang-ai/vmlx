# SPDX-License-Identifier: Apache-2.0
"""Native-MTP rollback points for Nemotron-H Mamba-2 layers (Nemotron-3.5-Super-VL).

The MLLM native-MTP verifier rolls a rejected draft back without replaying the confirmed token when every recurrent
layer publishes, during the verify forward, the state after the confirmed prefix (``rollback_state``) and a
``rollback_to(accepted)`` closure for the state after ``accepted`` drafts (mllm_batch_generator
``_native_mtp_rollback_to_confirmed``). Without them a reject restores the pre-verify snapshot and re-runs the
verify inputs: a whole extra backbone forward (measured live on Super: 30-39 ms per reject, which made short D3
answers slower than AR).

mlx-lm's ``NemotronHMamba2Mixer`` runs conv + SSM over the whole verify block in one ``ssm_update``. With a confirmed
count marked on the layer's cache (``_vmlx_mtp_n_confirmed``, set by the wrapper for the verify forward only), the
patched mixer projects all rows once, runs conv + SSM for the confirmed rows and then the draft rows through the same
``_conv`` / ``_ssm`` methods, stashes the post-confirmed (conv, ssm) pair, and binds a closure that recomputes only this
layer's accepted draft rows from that pair on a scratch cache (the live cache is never touched by the closure).

Unmarked calls (prefill, AR decode, every other nemotron_h model such as Nano-Omni) take the original method.

Test note: mlx-lm's single-step SSM Metal kernel computes ``n_per_t = Ds / 32`` state elements per lane, so a toy
config with ``ssm_state_size`` < 32 makes it write nothing and return uninitialised state (results then depend on
allocator contents). Every Nemotron-3.5 bundle has ssm_state_size 128; tests must use a multiple of 32.
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_patched = False


def clear_rollback_point(cache) -> None:
    cache.rollback_state = None
    cache.rollback_to = None


def apply() -> bool:
    global _patched
    if _patched:
        return True
    with _lock:
        if _patched:
            return True
        import mlx.core as mx
        from mlx_lm.models import nemotron_h
        from mlx_lm.models.cache import ArraysCache

        Mixer = nemotron_h.NemotronHMamba2Mixer
        original = Mixer.__call__

        def __call__(self, hidden_states, mask, cache=None):
            n_conf = int(getattr(cache, "_vmlx_mtp_n_confirmed", 0) or 0) if cache is not None else 0
            seq = int(hidden_states.shape[1])
            if cache is not None:
                clear_rollback_point(cache)
            if not (0 < n_conf < seq) or getattr(cache, "lengths", None) is not None:
                return original(self, hidden_states, mask, cache)

            projected = self.in_proj(hidden_states)
            gate, conv_input, dt = mx.split(
                projected, [self.intermediate_size, self.intermediate_size + self.conv_dim], axis=-1
            )
            splits = [self.intermediate_size, self.intermediate_size + self.n_groups * self.ssm_state_size]

            def segment(c, lo, hi):
                m = mask[:, lo:hi] if mask is not None else None
                conv_out = self._conv(conv_input[:, lo:hi], c, m)
                hs, B, C = mx.split(conv_out, splits, axis=-1)
                return self._ssm(hs, B, C, dt[:, lo:hi], c, m)

            y_conf = segment(cache, 0, n_conf)
            conv_c, ssm_c = cache[0], cache[1]
            y_draft = segment(cache, n_conf, seq)
            cache.rollback_state = (conv_c, ssm_c)

            def rollback_to(accepted, _lo=n_conf):
                accepted = int(accepted)
                if accepted <= 0:
                    return conv_c, ssm_c
                scratch = ArraysCache(2)
                scratch[0], scratch[1] = conv_c, ssm_c
                segment(scratch, _lo, _lo + accepted)
                return scratch[0], scratch[1]

            cache.rollback_to = rollback_to
            cache.advance(seq)
            y = self.norm(mx.concatenate([y_conf, y_draft], axis=1), gate)
            return self.out_proj(y)

        Mixer.__call__ = __call__
        Mixer._vmlx_original_call = original
        _patched = True
        logger.info("Patched mlx_lm NemotronHMamba2Mixer: native-MTP confirmed/draft split with rollback points")
        return True


def mark_confirmed(cache_list, n_confirmed: int) -> list:
    """Mark every Mamba ArraysCache in ``cache_list`` for one verify forward; returns the marked caches."""
    from mlx_lm.models.cache import ArraysCache

    marked = []
    for c in cache_list or []:
        if isinstance(c, ArraysCache):
            c._vmlx_mtp_n_confirmed = int(n_confirmed)
            marked.append(c)
    return marked


def unmark(marked) -> None:
    for c in marked:
        c._vmlx_mtp_n_confirmed = 0
