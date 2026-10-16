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
            if getattr(cache, "lengths", None) is not None:
                return original(self, hidden_states, mask, cache)
            # Inside the native-MTP verify scope every row must reproduce its single-token decode step bit for bit
            # (greedy MTP == AR, C17): the multi-row chunked scan + conv over the block differ from the single-step
            # path at layer 0 already (measured, Super, 4.9e-4 at the first Mamba block, 4.75-logit max at the head,
            # argmax flip within 48 tokens). in_proj / out_proj stay batched (the scope installs the bit-exact
            # multi-row QMV, weights read once); only conv + SSM, cheap state-bound work, run row by row.
            from vmlx_engine.metal.row_exact_qmv import row_exact_scope_active

            exact = seq > 1 and cache is not None and row_exact_scope_active()
            if not exact and not (0 < n_conf < seq):
                return original(self, hidden_states, mask, cache)

            projected = self.in_proj(hidden_states)
            gate, conv_input, dt = mx.split(
                projected, [self.intermediate_size, self.intermediate_size + self.conv_dim], axis=-1
            )
            splits = [self.intermediate_size, self.intermediate_size + self.n_groups * self.ssm_state_size]

            def block(c, lo, hi):
                m = mask[:, lo:hi] if mask is not None else None
                conv_out = self._conv(conv_input[:, lo:hi], c, m)
                hs, B, C = mx.split(conv_out, splits, axis=-1)
                return self._ssm(hs, B, C, dt[:, lo:hi], c, m)

            def segment(c, lo, hi):
                if not exact or hi - lo <= 1:
                    return block(c, lo, hi)
                return mx.concatenate([block(c, r, r + 1) for r in range(lo, hi)], axis=1)

            if not (0 < n_conf < seq):                       # row-exact verify without a rollback request
                y = segment(cache, 0, seq)
                cache.advance(seq)
                return self.out_proj(self.norm(y, gate))

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

        # Attention: MLX keeps its single-query vector SDPA for up to two queries (bit-exact with decode) and
        # switches kernels from three, so a depth-3 verify (4 rows) differed from AR first at the first attention
        # block (Super layer 7, measured). Inside the verify scope, attend rows in pairs like their decode steps.
        Attention = nemotron_h.NemotronHAttention
        original_attention = Attention.__call__
        sdpa = nemotron_h.scaled_dot_product_attention

        def attention_call(self, x, mask=None, cache=None):
            from vmlx_engine.metal.row_exact_qmv import attend_rows_in_pairs, row_exact_scope_active

            B, L, _ = x.shape
            if L <= 2 or cache is None or not row_exact_scope_active():
                return original_attention(self, x, mask=mask, cache=cache)
            # Pair slicing synthesizes a causal mask; it must not silently
            # replace an explicit padding/additive mask or unmasked attention.
            # The normal single-request verify path supplies "causal".
            if not isinstance(mask, str) or mask != "causal":
                return original_attention(self, x, mask=mask, cache=cache)
            queries = self.q_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
            keys = self.k_proj(x).reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
            values = self.v_proj(x).reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
            keys, values = cache.update_and_fetch(keys, values)
            output = attend_rows_in_pairs(
                lambda q, k, v, m: sdpa(q, k, v, cache=cache, scale=self.scale, mask=m), queries, keys, values
            )
            return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))

        Attention.__call__ = attention_call
        Attention._vmlx_original_call = original_attention
        _patched = True
        logger.info("Patched mlx_lm nemotron_h for native-MTP verify: Mamba-2 rollback points + row-exact "
                    "conv/SSM, attention rows in pairs (inside row_exact_verify_scope only)")
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
