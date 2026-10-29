"""MiniMax-M3 / M3.1 (``minimax_m3_vl`` text model) as a DSpark target.

The DSpark loop (``generate.iter_dspark``) needs five things from a target; this
module gives them for vMLX's MiniMax-M3 runtime (``models/minimax_m3``):

- ``prefill``: the same forwards the served single-request path runs for an M3
  prompt (``SingleBatchGenerator._prefill_full_and_sample``: chunks of
  ``prefill_step``, the last chunk never a lone token, logits of the last
  chunk's last row), so the first token and the cache equal plain decoding's.
  It also returns the drafter's taps: the residual stream after each of the
  drafter's ``target_layer_ids`` (M3.1: layers 3 / 17 / 31 / 45 / 59), every
  prompt row, concatenated in the drafter's order.
- ``ar_step`` / ``decode``: one token through the model's own forward (``ar_step``
  also returns its taps; same arithmetic).
- ``verify``: T = k + 1 rows whose logits are exactly the logits T sequential
  one-token decode steps produce. Per layer:
    * projections (q / k / v / o, index q / k, dense MLP, shared expert, router,
      lm_head): every row on decode arithmetic. MLX's own multi-row kernels
      reduce in a different order than its one-row kernels (measured: affine,
      MXFP8 and float all differ at 2..8 rows), so affine projections run
      ``metal/row_exact_qmv`` and MXFP8 / MXFP4 projections
      ``metal/row_exact_fp_qmv``: MLX's one-row kernel transcribed for up to four
      rows sharing one weight walk, admitted per shape only after a bit-equality
      check against stock one-row calls. Every other projection (float,
      unadmitted shapes, custom modules) runs as one-row calls.
    * attention, per row: each row attends exactly the keys a decode step at its
      position sees (the cache prefix up to and including itself), with the MSA
      selection a decode step makes there -- the indexer scores that row's index
      query against the index keys up to its position (same score kernel, same
      block max-pool, local block, per-KV-group top-k) -- through the model's
      own attention call (no mask below the sparse threshold, the per-group key
      mask above it).
    * routed experts: one call for all rows while every (row, expert) pair stays
      on the unsorted gather kernels a decode step uses (MLX SwitchGLU below 64
      routed pairs, JANGT below its sort threshold); otherwise one call per row.
    * norms, RoPE, router top-k, SwiGLU-OAI, embeddings: row-independent ops
      (checked bit-equal per row in tests/test_minimax_m3_dspark.py).
  K / V and index keys of all T rows are appended to the cache in the forward.
- ``rollback``: keep the first ``kept`` rows. Every M3 cache lane is append-only
  and position-anchored (K / V, index keys; the MSA block selection is
  recomputed from the index keys every step, nothing block-structured is
  stored), so a rollback is a trim of the logical length of every lane: the next
  rows overwrite the trimmed tail, whether the new length lands mid a 128-key
  block or crosses one. Lockstep of the K / V and index lanes is checked.

Media requests (M3 VL) stay on the native path; the drafter only proposes text.
"""

from __future__ import annotations

import inspect
import logging
import os
import sys
from typing import Optional, Sequence

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

# Rows per multi-row affine kernel launch: the one-row qmv transcription keeps every row in registers up to four rows
# at every bit width (row_exact_qmv._max_rows); wider verifies run as several launches, every row still exact.
_ROWS_PER_LAUNCH = 4
# MLX SwitchGLU sorts (row, expert) pairs onto its multi-row gather kernels from this many routed pairs.
_SWITCH_SORT_PAIRS = 64


def _row(x: mx.array, r: int) -> mx.array:
    return mx.contiguous(x[:, r:r + 1])


def _blocks(rows: int, width: int):
    """Balanced consecutive row blocks of at most ``width`` rows: (start, size)."""
    n = -(-rows // width)
    base, extra = divmod(rows, n)
    start = 0
    for b in range(n):
        size = base + (1 if b < extra else 0)
        yield start, size
        start += size


def _row_exact_kernel_enabled() -> bool:
    raw = os.environ.get("VMLX_ROW_EXACT_VERIFY_QMV", "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


_ACCEPTS_HOOKS: dict = {}


def _accepts(module, name: str) -> bool:
    key = (type(module), name)
    hit = _ACCEPTS_HOOKS.get(key)
    if hit is None:
        try:
            hit = name in inspect.signature(type(module).__call__).parameters
        except (TypeError, ValueError):
            hit = False
        _ACCEPTS_HOOKS[key] = hit
    return hit


class MiniMaxM3DSparkTarget:
    """DSpark target adapter for vMLX's MiniMax-M3 text model (see the module docstring)."""

    # Every verify row runs on decode arithmetic at any width: this target puts no ceiling on the draft ladder.
    max_verify_rows: Optional[int] = None

    def __init__(self, model):
        lm = model
        for _ in range(3):
            if hasattr(lm, "model") and hasattr(getattr(lm, "model"), "layers") and hasattr(lm, "args"):
                break
            lm = getattr(lm, "language_model", None) or getattr(lm, "model", None)
            if lm is None:
                break
        if lm is None or not hasattr(lm, "args") or not hasattr(lm.model, "layers"):
            raise ValueError("DSpark MiniMax-M3 target: no MiniMax-M3 text model found")
        self.model = lm
        self.nm = lm.model
        self.args = lm.args
        for layer in self.nm.layers:
            attn = layer.self_attn
            if not all(hasattr(attn, n) for n in ("project", "attend")) or (
                    attn.indexer is not None and not hasattr(attn.indexer, "select")):
                raise ValueError("DSpark MiniMax-M3 target: attention without the project / attend / select pieces")
        probe = lm.make_cache()
        for li, (layer, c) in enumerate(zip(self.nm.layers, probe)):
            if not (hasattr(c, "update_and_fetch") and hasattr(c, "trim")):
                raise ValueError(f"DSpark MiniMax-M3 target: layer {li} cache {type(c).__name__} cannot roll back")
            if layer.self_attn.indexer is not None and not hasattr(c, "update_index"):
                raise ValueError(f"DSpark MiniMax-M3 target: sparse layer {li} has no index-key cache lane")
        self.embed_tokens = self.nm.embed_tokens
        self.tied = bool(getattr(self.args, "tie_word_embeddings", False))
        self.lm_head = (lambda h: self.nm.embed_tokens.as_linear(h)) if self.tied else lm.lm_head
        self.rope_delta = 0
        # Per verify: how many projection calls ran batched on the exact multi-row kernel vs as one-row calls, and
        # how many routed-expert layers ran batched vs per row (cost diagnostics for real-weight runs).
        self.verify_paths = {"linear_batched": 0, "linear_rows": 0, "routed_batched": 0, "routed_rows": 0}

    # ------------------------------------------------------------------ plumbing
    def make_cache(self):
        return self.model.make_cache()

    def set_media(self, embeds=None, positions=None, rope_delta: int = 0) -> None:
        if embeds is not None or positions is not None or rope_delta:
            raise ValueError("DSpark MiniMax-M3 target drafts text requests only")

    @staticmethod
    def _offset(cache) -> int:
        offset = getattr(cache[0], "offset", 0) or 0
        if isinstance(offset, mx.array):
            return int(offset.reshape(-1)[0].item())
        return int(offset)

    def _stock(self, tokens: mx.array, cache, taps: Sequence[int], logits: bool):
        """The model's own forward (Model.__call__ ops) on these rows; taps in the drafter's order."""
        taps = tuple(int(t) for t in taps)
        if taps:
            layers = tuple(sorted(set(taps)))
            out, aux = self.nm(tokens, cache, return_aux=True, aux_layers=layers)
            by_layer = dict(zip(layers, aux))
            feats = mx.concatenate([by_layer[t] for t in taps], axis=-1)
        else:
            out, feats = self.nm(tokens, cache), None
        if not logits:
            return None, feats
        return self.lm_head(out), feats

    # ------------------------------------------------------------------ prefill / AR
    def prefill(self, ids: mx.array, cache, taps: Sequence[int], prefill_step: Optional[int],
                keep_last: Optional[int] = None, plan: Optional[dict] = None):
        """Returns (last-row logits (V,), tap features (1, P, n_taps * H) or None). Chunking = the served M3 prompt
        path: with an aligned ``plan`` (``SingleBatchGenerator._prefill_m3_aligned_and_sample``), pieces cut at every
        absolute multiple of ``prefill_step`` and at the plan's store boundary; without one, chunks of
        ``prefill_step`` while more than one chunk remains, never leaving a lone final token.
        ``keep_last``: return only the features of the last ``keep_last`` rows (a sliding-window drafter never reads
        older ones; M3.1's taps are 61 KB per row, so a long prompt's full set would be gigabytes)."""
        from ..utils.generation_cancellation import raise_if_generation_cancelled

        length = int(ids.shape[1])
        step = max(1, int(prefill_step or length or 1))
        feats = []

        def keep(f):
            feats.append(f)
            if keep_last:
                held = sum(int(a.shape[1]) for a in feats)
                while feats and held - int(feats[0].shape[1]) >= keep_last:
                    held -= int(feats.pop(0).shape[1])

        def chunk_len(at: int) -> int:
            n = min(step, length - at)
            if length - (at + n) == 1 and n > 1:
                n -= 1
            return n

        if plan is not None:
            from ..models.minimax_m3.prefill_geometry import prefill_pieces

            start = self._offset(cache)
            pieces = [(a - start, b - start) for a, b in
                      prefill_pieces(start, start + length, step, plan.get("boundary"), atomic=bool(plan.get("atomic")))]
        else:
            pieces, pos = [], 0
            while length - pos > step:
                n = chunk_len(pos)
                pieces.append((pos, pos + n))
                pos += n
            pieces.append((pos, length))
        for a, b in pieces[:-1]:
            raise_if_generation_cancelled()
            _, f = self._stock(ids[:, a:b], cache, taps, logits=False)
            mx.eval([c.state for c in cache])
            if f is not None:
                mx.eval(f)
                keep(f)
        pos = pieces[-1][0]
        raise_if_generation_cancelled()
        logits, f = self._stock(ids[:, pos:], cache, taps, logits=True)
        if f is not None:
            keep(f)
        if not feats:
            return logits[0, -1], None
        out = mx.concatenate(feats, axis=1)
        if keep_last and int(out.shape[1]) > keep_last:
            out = out[:, -keep_last:]
        return logits[0, -1], out

    def prefill_segment(self, ids: mx.array, cache, taps: Sequence[int], prefill_step: Optional[int]):
        """Forward a prompt segment that ends at a snapshot cut (no logits), in ``prefill_step`` chunks."""
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
        return mx.concatenate(feats, axis=1) if feats else None

    def ar_step(self, token: mx.array, cache, taps: Sequence[int]):
        """One decode step that also returns its tap row (same ops as ``decode``)."""
        logits, feats = self._stock(token.reshape(1, 1), cache, taps, logits=True)
        return logits[0, -1], feats

    def decode(self, token: mx.array, cache):
        """One decode step through the served model call (the reference path)."""
        return self.model(token.reshape(1, 1), cache=cache)[0, -1]

    # ------------------------------------------------------------------ verify
    def _linear(self, module, x: mx.array) -> mx.array:
        """``module(x)`` for (1, T, K) rows, every row equal to its one-row call."""
        rows = int(x.shape[1])
        if rows == 1:
            return module(x)
        wide = self._wide_kernel(module)
        if wide is not None:
            y = wide(x)
            if y is not None:
                self.verify_paths["linear_batched"] += 1
                return y + module["bias"] if "bias" in module else y
        kernel = self._multi_row_kernel(module)
        if kernel is not None:
            outs, batched = [], 0
            for start, size in _blocks(rows, _ROWS_PER_LAUNCH):
                xb = x[:, start:start + size]
                y = kernel(xb) if size > 1 else None
                if y is None:
                    y = mx.concatenate([module(_row(xb, r)) for r in range(size)], axis=1)
                else:
                    batched += 1
                    if "bias" in module:
                        y = y + module["bias"]
                outs.append(y)
            self.verify_paths["linear_batched" if batched else "linear_rows"] += 1
            return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)
        self.verify_paths["linear_rows"] += 1
        return mx.concatenate([module(_row(x, r)) for r in range(rows)], axis=1)

    @staticmethod
    def _wide_kernel(module):
        """One weight walk for the whole window (up to 16 rows), every row equal to the one-row call (admitted per
        shape by bit-equality; ``VMLX_ROW_EXACT_WIDE=0`` keeps the 4-row launches)."""
        if not isinstance(module, nn.QuantizedLinear) or not _row_exact_kernel_enabled():
            return None
        if str(getattr(module, "mode", "affine") or "affine") != "affine" or "biases" not in module:
            return None
        from ..metal.row_exact_qmv import row_exact_qmv_wide, wide_enabled

        if not wide_enabled():
            return None
        gs, bits = int(module.group_size), int(module.bits)
        return lambda xb: row_exact_qmv_wide(xb, module["weight"], module["scales"], module["biases"],
                                             group_size=gs, bits=bits)

    @staticmethod
    def _multi_row_kernel(module):
        """A multi-row kernel whose every row equals the one-row call (admitted per shape by bit-equality), or None."""
        if not isinstance(module, nn.QuantizedLinear) or not _row_exact_kernel_enabled():
            return None
        mode = str(getattr(module, "mode", "affine") or "affine")
        gs, bits = int(module.group_size), int(module.bits)
        if mode == "affine" and "biases" in module:
            from ..metal.row_exact_qmv import row_exact_qmv

            return lambda xb: row_exact_qmv(xb, module["weight"], module["scales"], module["biases"],
                                            group_size=gs, bits=bits, mode="affine")
        if mode in ("mxfp8", "mxfp4"):
            from ..metal.row_exact_fp_qmv import row_exact_fp_qmv

            return lambda xb: row_exact_fp_qmv(xb, module["weight"], module["scales"], group_size=gs, bits=bits,
                                               mode=mode)
        return None

    def _logits(self, out: mx.array) -> mx.array:
        if self.tied:
            return mx.concatenate([self.lm_head(_row(out, r)) for r in range(int(out.shape[1]))], axis=1)
        return self._linear(self.lm_head, out)

    def _attention(self, attn, x: mx.array, cache, offset: int) -> mx.array:
        B, rows, _ = x.shape
        q, k, v = attn.project(x, offset, self._linear)
        keys, values = cache.update_and_fetch(k, v)
        idx_q = idx_keys = None
        if attn.indexer is not None:
            idx_q, idx_k = attn.indexer.project(x, offset, self._linear)
            # A decode step's indexer reads the index-key history only from the sparse cache lane.
            idx_keys = cache.update_index(idx_k) if hasattr(cache, "update_index") else idx_k
        outs = []
        for r in range(rows):
            end = offset + r + 1
            bias = None
            if idx_q is not None:
                ik = idx_keys[:, :, :end] if hasattr(cache, "update_index") else idx_keys[:, :, r:r + 1]
                bias = attn.indexer.select(idx_q[:, :, r:r + 1], ik, offset + r)
            outs.append(attn.attend(mx.contiguous(q[:, :, r:r + 1]), keys[:, :, :end], values[:, :, :end],
                                    bias, None, cache))
        o = mx.concatenate(outs, axis=2)
        return self._linear(attn.o_proj, o.transpose(0, 2, 1, 3).reshape(B, rows, -1))

    def _routed_batch_is_exact(self, block, x: mx.array, inds: mx.array) -> bool:
        """True when one call for all rows keeps every (row, expert) pair on the kernels a one-row call uses."""
        switch = block.switch_mlp
        pairs = int(inds.size)
        if getattr(switch, "is_jangt", False):
            from ..jangt.switch import SORT_THRESHOLD, nax_prefill

            return pairs < SORT_THRESHOLD or not nax_prefill()
        from mlx_lm.models.switch_layers import SwitchGLU

        cls = type(switch)
        if cls.__call__ is not SwitchGLU.__call__:
            if cls.__name__ != "MiniMaxM3Affine2SwitchGLU":
                return False                       # unknown routed implementation: one call per row
            # The vendored runtime can be loaded as its own module (mlx_lm.models.minimax_m3_vl): ask that module.
            can_fast = getattr(sys.modules.get(cls.__module__), "can_use_affine2_switchglu", None)
            if can_fast is None or can_fast(switch, x[:, :1], inds[:, :1]):
                return False                       # decode runs the affine-2 one-row kernel
        return pairs < _SWITCH_SORT_PAIRS

    def _routed(self, block):
        def routed(x, inds, scores):
            if self._routed_batch_is_exact(block, x, inds):
                self.verify_paths["routed_batched"] += 1
                return block.routed(x, inds, scores)
            self.verify_paths["routed_rows"] += 1
            return mx.concatenate([block.routed(_row(x, r), _row(inds, r), _row(scores, r))
                                   for r in range(int(x.shape[1]))], axis=1)
        return routed

    def _mlp(self, mlp, x: mx.array) -> mx.array:
        if hasattr(mlp, "switch_mlp") and hasattr(mlp, "routed") and _accepts(mlp, "routed"):
            return mlp(x, linear=self._linear, routed=self._routed(mlp))
        if _accepts(mlp, "linear"):
            return mlp(x, linear=self._linear)
        return mx.concatenate([mlp(_row(x, r)) for r in range(int(x.shape[1]))], axis=1)

    def verify(self, tokens: mx.array, cache, taps: Sequence[int]):
        """Forward (1, T) rows; returns (logits (T, V), tap features (1, T, n_taps * H) or None, None). Row r's
        logits equal those of the r-th of T sequential decode steps; all T rows are appended to the cache."""
        taps = tuple(int(t) for t in taps)
        tapset = set(taps)
        nm = self.nm
        offset = self._offset(cache)
        h = nm.embed_tokens(tokens)
        feats = {}
        for li, (layer, c) in enumerate(zip(nm.layers, cache)):
            h = h + self._attention(layer.self_attn, layer.input_layernorm(h), c, offset)
            h = h + self._mlp(layer.mlp, layer.post_attention_layernorm(h))
            if li in tapset:
                feats[li] = h
        logits = self._logits(nm.norm(h))[0]
        return logits, (mx.concatenate([feats[t] for t in taps], axis=-1) if taps else None), None

    # ------------------------------------------------------------------ rollback
    def rollback(self, cache, sink, kept: int, total: int) -> None:
        """Keep the first ``kept`` of ``total`` verify rows in every cache lane."""
        trim = int(total) - int(kept)
        if trim <= 0:
            return
        for li, c in enumerate(cache):
            if c.trim(trim) != trim:
                raise RuntimeError(f"DSpark rollback: layer {li} cache holds fewer than {trim} verify rows")
            idx_offset = getattr(c, "_idx_offset", None)
            if idx_offset is not None and getattr(c, "idx_keys", None) is not None and int(idx_offset) != int(c.offset):
                raise RuntimeError(f"DSpark rollback: layer {li} index-key lane {idx_offset} != K/V lane {c.offset}")
