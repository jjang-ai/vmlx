# SPDX-License-Identifier: Apache-2.0
"""dots3_note language model (46-layer hybrid MLA MoE + MTP layer 46).

Numerics ported from the conversion pipeline's verified core
(``jang_tools/dots3/ops.py``, cos 1.000000 vs the transformers PR reference
at S=10 and S=245) and the runtime handoff. The traps that shaped this file:

- TWO MLA geometries: full/DSA layers (128 heads, kv_lora 512, nope 128,
  θ=8e7) vs sliding layers (64 heads, kv_lora 1024, nope 192, θ=5e4,
  window 513). A single-geometry implementation produces plausible garbage.
- The low-rank rescale ``sqrt(hidden/rank)`` applies AFTER the q_a/kv_a
  RMSNorms; ``k_rope_only_layernorm`` applies BEFORE rope; rope is
  GPT-J INTERLEAVED (``mx.fast.rope(..., traditional=True)``); the rope key
  is a single MQA head broadcast across all heads.
- Norm weights have small means (0.002-0.49) but are NOT zero-centered:
  plain multiply, NO +1 shift.
- Router logits run in the activation dtype (bf16), selection adds the F32
  ``e_score_correction_bias``, returned weights come from UNBIASED scores.
- ``model.layers.46`` is the MTP layer (enorm/hnorm/eh_proj + full-geometry
  MLA without indexer + dense FFN + shared_head.norm, sharing the backbone
  lm_head, with its OWN ``model.mtp.embed_tokens``). The backbone forward
  runs layers[:46] only.

Cache contract (stage A): every backbone layer uses a plain, unbounded
KVCache holding MATERIALIZED per-head keys/values; sliding layers enforce
their window purely through the additive mask. This is exact and simple but
caps practical context (~2K) — the latent/absorbed cache replaces it behind
an equivalence A/B (task #198). DSA indexer selection is not applied: for
sequences <= index_topk (2048) top-k selects every causal position, so dense
attention is mathematically identical; longer prompts are refused loudly.
"""

import math
import os
from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.switch_layers import SwitchGLU

from .config import AttnGeom, ModelConfig, TextConfig


def _absorb_enabled() -> bool:
    """MLA absorption (latent KV cache) — default ON.

    VMLX_DOTS3_MLA_ABSORB=0 reverts to the stage-A materialized per-head
    cache, which is exact but stores 71x (full) / 23x (SWA) more bytes per
    token and therefore caps practical context (~2K on 128 GB).
    """
    return os.environ.get("VMLX_DOTS3_MLA_ABSORB", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


_PREFILL_PATH_LOGGED = False
_DSA_PATH_LOGGED = False


def _dsa_gather_enabled() -> bool:
    """DSA gather-attention — measured non-negative, kept DEFAULT OFF.

    When DSA selection engages (total > index_topk) the dense-mask path
    still runs SDPA over ALL ``total`` keys with a [B, 1, S, total]
    additive mask: O(S*total) attention FLOPs plus an O(S*total) fp32 mask
    transient, which puts context length right back into attention cost and
    throws away what selection just bought. Gathering the selected keys
    instead bounds the attention call at O(S*topk) — past the topk bound,
    context length stops driving attention cost (the indexer's own
    O(S*total) scoring pass is inherent to DSA and remains).

    MEASURED LIVE 2026-08-16 (erics-m5-max 128GB, dots3-note-prev bundle,
    temp 0, cold prefill, admission valve off, one serve per arm,
    engagement proven by the path log in BOTH arms, temp-0 answer text
    BYTE-IDENTICAL at 4k and 12k): same-session like-for-like pairs, all
    non-negative — chunk-2048 ladder +3.3% @4k / +6.3% @8k / +3.3% @12k
    pp/s; chunk-1024 pair +9.5% pp/s @12k (359.8 vs 328.7); decode
    +8.5% @4k (32.58 vs 30.03 t/s) and +2.8% @12k engaged (31.44 vs
    30.59). All of it sits inside session-to-session box drift (the SAME
    dense 4k baseline measured 510 / 461 / 415 pp/s across the day), so
    the honest claim is "never slower, structurally right", not a speedup
    number. Isolated, the win is real and GROWS: gathered attention is
    FLAT ~23 ms/chunk/layer vs dense 18.7 (4k) -> 117.7 (25k)
    ms/chunk/layer — but end to end the served bottleneck past ~8k is the
    indexer's own O(S*total) scoring, which both arms pay.

    NOW DEFAULT **ON** (2026-08-16, ledger row 173). Speed was the wrong
    yardstick and kept this OFF for a day: the real value is DEPTH. The
    dense path's [B,1,S,total] mask transient GROWS with context and is
    what exhausts prefill headroom, so bounding attention at O(S*topk)
    buys ceiling, not milliseconds. Matched A/B, same ladder, one serve
    per arm, answers 'ack' in every completed rung:

        depth    dense                       gather
        8,192    407.1 pp/s                  442.5 pp/s   (+8.7%)
       16,384    413 DECLINED @12,608 ctx    PASSES, 188.5 pp/s
       24,576    (not reached)               413 @16,704 ctx

    Dense declined at EXACTLY 12,608 context on two independent runs, so
    the +33% usable-depth gain is reproducible, not box drift. A user
    hitting 16k previously got a hard 413; they now get an answer.
    `VMLX_DOTS3_DSA_GATHER=0` restores the dense path.

    The remaining ceiling is NOT this code: `iogpu.wired_limit_mb` is
    unset on the test box, so macOS caps the working set at 84% of RAM
    (115.4 of 137.4 GB) against a ~97 GB-resident model. Raising that
    sysctl adds transient headroom on top of this fix; 400k additionally
    needs the indexer's own O(S*total) scoring addressed.
    """
    return os.environ.get("VMLX_DOTS3_DSA_GATHER", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _prefill_materialize_max_keys() -> int:
    """Key-count ceiling for materializing K/V from the latent at prefill.

    Absorption is a DECODE optimization: scoring against the latent costs
    (kv_lora_rank + rope) per key where materialized K costs
    (qk_nope + rope), and the value side costs rank instead of v_head_dim.
    On dots3 that is 2.8x the attention FLOPs on full layers (576 vs 192,
    512 vs 128) and 5.5x on the 33 SWA layers (1088 vs 256, 1024 vs 128).
    Materializing from the fetched latent is O(total) per chunk and buys
    back O(S*total) — the DSV4 MLA split.

    🚨 MEASURED NEGATIVE — DEFAULT OFF (ledger row 156). The FLOP argument
    above is correct and still buys nothing: live on the box (temp 0, cold
    prefill, engagement proven in BOTH arms by the path log below,
    answer text byte-identical) materialization scored 1.03x @1k,
    1.02x @2k, 0.97x @4k, 0.88x @6k — a wash that turns into a LOSS as the
    key block grows. dots3 prefill sustains only ~4.6 effective TFLOPS
    against a 16B-active MoE, so it is not attention-FLOP-bound at all and
    trading O(S*total) score FLOPs for an O(total) materialization
    transient just adds allocation pressure. Same shape as the fp16-indexer
    result (2.26x isolated, -22% end to end). Kept, gated OFF, with its
    equivalence pins, because it is the right structure if the real
    bottleneck is ever removed. Set a positive key ceiling to re-enable.
    """
    try:
        return int(
            os.environ.get("VMLX_DOTS3_PREFILL_MATERIALIZE_MAX_KEYS", "0")
        )
    except ValueError:
        return 0


def _logical_kv_extent(cache_obj):
    """(keys, values) trimmed to the cache's LOGICAL length.

    Restored plain-KV layers can be zero-padded to KVCache.step while
    ``offset`` stays logical. ``KVCache.state`` already slices to ``:offset``;
    the raw ``.keys``/``.values`` attributes do not. Prefer ``state`` and fall
    back to an explicit slice so a padded stream can never reach a consumer
    that treats ``shape[2]`` as the real key count.
    """
    keys = getattr(cache_obj, "keys", None)
    values = getattr(cache_obj, "values", None)
    if keys is None or values is None:
        return keys, values
    try:
        offset = int(getattr(cache_obj, "offset", 0) or 0)
    except (TypeError, ValueError):
        return keys, values
    if 0 < offset < int(keys.shape[2]):
        return keys[..., :offset, :], values[..., :offset, :]
    return keys, values


def _logical_offset(cache_obj, keys) -> int:
    """Logical offset, never a padded physical length."""
    try:
        offset = int(getattr(cache_obj, "offset", 0) or 0)
    except (TypeError, ValueError):
        offset = 0
    if offset > 0:
        return offset
    return int(keys.shape[2]) if keys is not None else 0


class Dots3LatentCache:
    """Ordered latent cache for one MLA layer.

    Stores the per-token shared latent (``kv_a`` after norm+rescale, rank
    values) and the roped MQA rope key (``k_pe``) — 576 values/token on full
    layers, 1088 on SWA. Ordered concatenation (never a ring) so physical
    index distance == positional distance and the sliding mask stays a
    simple ``(q - k) < window`` over the physical layout. Sliding layers
    trim with HYSTERESIS: an O(window) slice-copy per token would burn
    ~25 MB/token/layer of bandwidth, so the cache keeps up to
    ``window - 1 + trim_step`` entries and trims in blocks; the mask makes
    the extra tail invisible, so trimming is a memory policy, not a
    correctness event.
    """

    trim_step = 256
    # Extra sliding-layer retention BEYOND window-1 so block-boundary
    # checkpoints inside the retained overhang can be reconstructed for
    # divergent-prefix restores (the prefix-cache boundary-checkpoint lane,
    # ledger row 152). Physics: a boundary B is rebuildable from a snapshot
    # at T only when keys (B-window+1..B) are still physically present, i.e.
    # T-B <= retained-overhang. 768 = three 256-token blocks ≈ 33 MB across
    # all sliding layers at bf16 — the mask math is (q-k) < window over the
    # physical layout, so extra retained keys are provably invisible.
    retain_overhang = int(
        os.environ.get("VMLX_DOTS3_SWA_RETAIN_OVERHANG", "768") or 768
    )

    def __init__(self, window: Optional[int] = None):
        self.window = window
        self.latent: Optional[mx.array] = None
        self.k_pe: Optional[mx.array] = None
        # Indexer key stream (full/DSA layers only). Unbounded and ordered —
        # selection needs every past position addressable by its global
        # index. bf16: 256 B/token/full-layer (~1.7 GB at 512K).
        self.idx_k: Optional[mx.array] = None
        self.offset = 0
        # Packing dims for the positional store presentation; carried in
        # meta_state so a restore can unpack without guessing.
        self._rope_dim = 0
        self._idx_dim = 0

    def update_indexer(self, idx_k: mx.array) -> mx.array:
        self._idx_dim = int(idx_k.shape[-1])
        if self.idx_k is None:
            self.idx_k = idx_k
        else:
            self.idx_k = mx.concatenate([self.idx_k, idx_k], axis=1)
        return self.idx_k

    def update_and_fetch(
        self, latent: mx.array, k_pe: mx.array
    ) -> Tuple[mx.array, mx.array]:
        self._rope_dim = int(k_pe.shape[-1])
        if self.latent is None:
            self.latent, self.k_pe = latent, k_pe
        else:
            self.latent = mx.concatenate([self.latent, latent], axis=2)
            self.k_pe = mx.concatenate([self.k_pe, k_pe], axis=2)
        self.offset += latent.shape[2]
        fetched = (self.latent, self.k_pe)
        if self.window is not None:
            # Trim AFTER capturing this call's return: the current chunk's
            # oldest query still needs window-1 entries BEFORE itself, but
            # future queries only ever need the last window-1 stored entries.
            # retain_overhang keeps extra masked-out history so recent block
            # boundaries stay checkpoint-reconstructible (see class comment).
            keep_min = self.window - 1 + self.retain_overhang
            if self.latent.shape[2] > keep_min + self.trim_step:
                self.latent = self.latent[:, :, -keep_min:]
                self.k_pe = self.k_pe[:, :, -keep_min:]
        return fetched

    def trim_to_boundary(self, boundary: int) -> bool:
        """Rewind this cache's state to logical position ``boundary``.

        Used by the prefix-cache restore lane when a restored snapshot's
        offset T exceeds the fetched span's token count B: full layers are
        positionally complete and simply slice; sliding layers rewind only
        when keys (B-window+1..B) are still inside the retained physical
        buffer. Returns False (state untouched) when the rewind is not
        exactly reconstructible — the caller must treat that as a miss.
        """
        boundary = int(boundary)
        if boundary <= 0 or self.latent is None:
            return False
        T = int(self.offset)
        if boundary > T:
            return False
        if boundary == T:
            return True
        drop = T - boundary
        phys = int(self.latent.shape[2])
        if self.window is None:
            if phys != T:
                # A positional stream missing early tokens cannot prove the
                # slice is aligned; refuse rather than guess.
                return False
            self.latent = self.latent[:, :, :boundary]
            self.k_pe = (
                self.k_pe[:, :, :boundary] if self.k_pe is not None else None
            )
            if self.idx_k is not None:
                self.idx_k = self.idx_k[:, :boundary]
            self.offset = boundary
            return True
        # A boundary before the window fills needs ALL of its keys.
        keep = min(self.window - 1, boundary)
        if phys - drop < keep:
            return False  # boundary older than the retained overhang
        self.latent = self.latent[:, :, phys - drop - keep : phys - drop]
        if self.k_pe is not None:
            self.k_pe = self.k_pe[:, :, phys - drop - keep : phys - drop]
        if self.idx_k is not None:
            ilen = int(self.idx_k.shape[1])
            if ilen == T:
                self.idx_k = self.idx_k[:, :boundary]
            elif ilen - drop >= keep:
                self.idx_k = self.idx_k[:, ilen - drop - keep : ilen - drop]
            else:
                self.idx_k = None
        self.offset = boundary
        return True

    # ---- prefix-cache store/restore protocol ---------------------------
    # The MLLM extractor requires state + meta_state; the presence of a
    # ``.cache`` LIST routes it down the branch that skips GQA-shape
    # normalization (these are latent streams, not per-head K/V).
    #
    # TWO presentations, keyed on ``window``:
    # - FULL layers (window None) present a POSITIONAL 2-tuple
    #   (packed_keys [B,1,S,rope+idx], latent [B,1,S,rank]) so the block
    #   machinery slices them per token — this is what makes the SSD (L2)
    #   publication chain valid: an all-cumulative family leaves every
    #   non-terminal block empty and the disk writer then refuses the last
    #   block's parent ancestry (measured live: "cannot publish block whose
    #   parent ancestry is unavailable", RAM hits masking a dead L2).
    # - SLIDING layers (window set) present a CUMULATIVE 3-tuple stored
    #   whole in the last block and restored on exact prefix boundaries
    #   (the hybrid-family pattern; their trimmed physical window cannot be
    #   positionally sliced).

    @property
    def cache(self):
        return [self.latent, self.k_pe, self.idx_k]

    @property
    def state(self):
        empty = mx.zeros((0,), dtype=mx.bfloat16)
        if self.window is None:
            if self.latent is None:
                return (empty, empty)
            k_pe = self.k_pe
            idx = self.idx_k
            parts = [k_pe]
            if idx is not None:
                parts.append(idx[:, None] if idx.ndim == 3 else idx)
            packed = mx.concatenate(parts, axis=-1)
            return (packed, self.latent)
        return (
            self.latent if self.latent is not None else empty,
            self.k_pe if self.k_pe is not None else empty,
            self.idx_k if self.idx_k is not None else empty,
        )

    @state.setter
    def state(self, value):
        def _real(a):
            return a if a is not None and getattr(a, "size", 0) else None

        if isinstance(value, (tuple, list)) and len(value) == 2:
            packed, latent = value
            self.latent = _real(latent)
            packed = _real(packed)
            if packed is None:
                self.k_pe = None
                self.idx_k = None
            else:
                rope_dim = self._rope_dim or packed.shape[-1]
                self.k_pe = packed[..., :rope_dim]
                idx = packed[..., rope_dim:]
                self.idx_k = idx[:, 0] if idx.shape[-1] else None
            self.window = None
            return
        latent, k_pe, idx_k = value
        self.latent = _real(latent)
        self.k_pe = _real(k_pe)
        self.idx_k = _real(idx_k)

    @property
    def meta_state(self):
        return (
            str(self.offset),
            str(self.window if self.window else ""),
            str(self._rope_dim),
            str(self._idx_dim),
        )

    @meta_state.setter
    def meta_state(self, value):
        offset, window = value[0], value[1]
        self.offset = int(offset)
        self.window = int(window) if str(window) else None
        if len(value) >= 4:
            self._rope_dim = int(value[2] or 0)
            self._idx_dim = int(value[3] or 0)

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls()
        # Meta first: the positional unpack needs the packing dims.
        if meta_state:
            obj.meta_state = meta_state
        obj.state = state
        if not meta_state and obj.latent is not None:
            obj.offset = int(obj.latent.shape[2])
        return obj


def _prefill_layer_eval_every() -> int:
    """Layers per prefill eval boundary; 0 (default) disables.

    Kept as an experiment lever only: MEASURED at every-8 AND every-2 on
    the live 100k ladder, the 98->118GB active sawtooth at 60k context was
    UNCHANGED — the deep-context transient is not pending-layer score
    graphs, so mid-forward eval barriers buy nothing and only add
    dispatch overhead. The remaining dots3 deep-context wall is the DSA
    dense-mask working set itself (the known kernel-level ceiling)."""
    try:
        return max(
            0,
            int(os.environ.get("VMLX_DOTS3_PREFILL_LAYER_EVAL_EVERY", "0")),
        )
    except (TypeError, ValueError):
        return 0


def _sliding_causal_mask(
    seq_len: int,
    past: int,
    window: Optional[int],
    dtype,
) -> mx.array:
    """Additive mask [S, past+S]: causal, optionally sliding.

    HF semantics: key visible iff k <= q and (q - k) < window — window 513
    means self + 512 past. Off-by-one here is invisible on short prompts.
    """
    q = (past + mx.arange(seq_len))[:, None]
    k = mx.arange(past + seq_len)[None, :]
    allowed = k <= q
    if window is not None:
        allowed = mx.logical_and(allowed, (q - k) < window)
    return mx.where(
        allowed, mx.array(0.0, dtype=dtype), mx.array(-mx.inf, dtype=dtype)
    )


class Dots3Indexer(nn.Module):
    """DSA indexer (full/DSA layers only) — inference selection path.

    Semantics ported from the PR reference (SGLang default fusion path):

    - indexer query = ``wq_b(q_lora)`` per head; indexer key =
      ``k_norm(wk(hidden))`` where ``k_norm`` is a LayerNorm WITH BIAS run
      in fp32 (the checkpoint ships a ``.bias`` tensor; RMSNorm is wrong);
    - 🚨 the ROPE slice comes FIRST in the indexer head layout
      ([rope | nope]) — the OPPOSITE of the main attention split — and is
      rotated with the full-attention theta, interleaved/GPT-J form;
    - per-head scores are ReLU'd then combined with
      ``weights_proj(hidden) * scale * n_heads**-0.5``, causal-masked, and
      the top ``index_topk`` key positions are selected per query.

    Deviation, documented: the CUDA reference quantizes q/k to fp8-e4m3
    (per-tensor amax) for SCORING only, which can shift which near-tie
    positions are selected. MLX has no e4m3 dtype, so scoring here runs
    fp32 — internally consistent (cold and warm paths select identically),
    but not bit-matched to vendor CUDA serving.
    """

    query_chunk_size = 1024

    def __init__(self, config: TextConfig):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.rope_theta = config.rope_theta
        self.scale = self.head_dim ** -0.5
        self.wq_b = nn.Linear(
            config.q_lora_rank,
            config.index_n_heads * config.index_head_dim,
            bias=False,
        )
        self.wk = nn.Linear(config.hidden_size, config.index_head_dim, bias=False)
        self.weights_proj = nn.Linear(
            config.hidden_size, config.index_n_heads, bias=False
        )
        self.k_norm = nn.LayerNorm(config.index_head_dim, bias=True)

    def encode_keys(self, hidden: mx.array, past: int) -> mx.array:
        """[B, S, H] -> roped indexer keys [B, S, head_dim] (bf16)."""
        key = self.wk(hidden)
        key = self.k_norm(key.astype(mx.float32))
        k_rope = key[..., : self.rope_dim][:, None]  # [B,1,S,rope]
        k_nope = key[..., self.rope_dim :]
        k_rope = mx.fast.rope(
            k_rope,
            self.rope_dim,
            traditional=True,
            base=self.rope_theta,
            scale=1.0,
            offset=past,
        )[:, 0]
        return mx.concatenate([k_rope, k_nope], axis=-1).astype(mx.bfloat16)

    def topk_indices(
        self,
        hidden: mx.array,
        q_lora: mx.array,
        keys: mx.array,
        past: int,
    ) -> mx.array:
        """Select per-query key positions. Returns int32 [B, S, K]."""
        B, S, _ = hidden.shape
        total = keys.shape[1]
        query = self.wq_b(q_lora).reshape(B, S, self.n_heads, self.head_dim)
        q_rope = query[..., : self.rope_dim].transpose(0, 2, 1, 3)
        q_nope = query[..., self.rope_dim :]
        q_rope = mx.fast.rope(
            q_rope,
            self.rope_dim,
            traditional=True,
            base=self.rope_theta,
            scale=1.0,
            offset=past,
        ).transpose(0, 2, 1, 3)
        query = mx.concatenate([q_rope, q_nope], axis=-1).astype(mx.float32)
        # [B, S, n_heads, 1] combining weights
        weights = (
            self.weights_proj(hidden).astype(mx.float32)
            * self.scale
            * self.n_heads ** -0.5
        )

        k_positions = mx.arange(total)[None, None, :]
        topk = min(self.index_topk, total)

        # Fast zero-intermediate fused path (B == 1, total >= 32):
        # avoids materializing [s_chunk, 64, total] FP32 scores & O(S log S) argpartition.
        if B == 1 and total >= 32:
            combined = None
            if S == 1:
                from vmlx_engine.metal.dsa_index_scores import dsa_decode_index_scores
                q_in = query.astype(keys.dtype) if keys.dtype in (mx.bfloat16, mx.float16) else query
                combined = dsa_decode_index_scores(q_in, keys, weights, scale=1.0)
            else:
                from vmlx_engine.metal.dsa_index_scores import dsa_pool_index_scores
                use_bf16 = keys.dtype == mx.bfloat16
                q_in = query.astype(mx.bfloat16) if use_bf16 else query
                k_in = keys if use_bf16 else keys.astype(mx.float32)
                combined = dsa_pool_index_scores(q_in, k_in, weights, scale=1.0, bf16_operands=use_bf16)
            if combined is not None:
                from vmlx_engine.metal.sparse_topk import metal_radix_topk
                q_pos = (past + mx.arange(S))[None, :, None]
                combined = mx.where(k_positions > q_pos, -mx.inf, combined)
                idx = metal_radix_topk(combined, topk)
                if idx is None:
                    idx = mx.argpartition(-combined, kth=topk - 1, axis=-1)[..., :topk]
                return idx.astype(mx.int32)

        keys_f = keys.astype(mx.float32)  # [B, total, D]
        # Per-head scores materialize [s_chunk, n_heads, total] in fp32 —
        # bound the transient to ~2 GB by shrinking the query chunk as the
        # key stream grows (64-head, 100K-key scoring at chunk 1024 would
        # otherwise transiently need ~26 GB).
        element_budget = 500_000_000
        chunk_size = max(1, min(self.query_chunk_size, element_budget // (self.n_heads * max(total, 1))))
        chunks = []
        for start in range(0, S, chunk_size):
            stop = min(start + chunk_size, S)
            q_chunk = query[:, start:stop]  # [B, s, h, D]
            # [B, s, h, total]
            scores = mx.einsum("bshd,btd->bsht", q_chunk, keys_f)
            scores = mx.maximum(scores, 0.0)
            combined = (scores * weights[:, start:stop, :, None]).sum(axis=2)
            q_pos = (past + mx.arange(start, stop))[None, :, None]
            combined = mx.where(k_positions > q_pos, -mx.inf, combined)
            idx = mx.argpartition(-combined, kth=topk - 1, axis=-1)[..., :topk]
            chunks.append(idx.astype(mx.int32))
        return mx.concatenate(chunks, axis=1)


class Dots3MLAAttention(nn.Module):
    # Element budget for the gathered-key transient [B, T, K, rank+rope]
    # (the indexer's element-budget pattern): the query tile T shrinks as
    # K*D grows. 256M elements = 512 MB bf16 gathered keys per full layer
    # per tile at deploy shapes (K=2048, D=576 -> T~=217).
    gather_element_budget = 268_435_456

    def __init__(
        self,
        config: TextConfig,
        layer_idx: int,
        geom: Optional[AttnGeom] = None,
    ):
        super().__init__()
        g = geom if geom is not None else config.geom(layer_idx)
        self.geom = g
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.apply_rescale = config.apply_mla_qkv_lora_rescale

        self.q_a_proj = nn.Linear(config.hidden_size, g.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(g.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            g.q_lora_rank, g.num_heads * g.qk_head_dim, bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, g.kv_lora_rank + g.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(g.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            g.kv_lora_rank,
            g.num_heads * (g.qk_nope_head_dim + g.v_head_dim),
            bias=False,
        )
        self.k_rope_only_layernorm = nn.RMSNorm(
            g.qk_rope_head_dim, eps=config.rms_norm_eps
        )
        self.o_proj = nn.Linear(
            g.num_heads * g.v_head_dim, config.hidden_size, bias=False
        )
        self.g_proj = nn.Linear(config.hidden_size, g.num_heads, bias=False)
        if not config.is_sliding(layer_idx) and layer_idx < config.num_hidden_layers:
            self.indexer = Dots3Indexer(config)
        # Per-head kv_b factors for the absorbed path, built lazily on first
        # use (kv_b_proj may be a QuantizedLinear by then; dequantized once,
        # ~34 MB/full layer at bf16). Underscore-prefixed so nn.Module does
        # not register them as loadable parameters.
        self._w_kb_nope: Optional[mx.array] = None
        self._w_kb_v: Optional[mx.array] = None

    def _kb_factors(self) -> Tuple[mx.array, mx.array]:
        if self._w_kb_nope is None:
            g = self.geom
            w = self.kv_b_proj.weight
            if hasattr(self.kv_b_proj, "scales"):
                w = mx.dequantize(
                    w,
                    self.kv_b_proj.scales,
                    getattr(self.kv_b_proj, "biases", None),
                    group_size=self.kv_b_proj.group_size,
                    bits=self.kv_b_proj.bits,
                    mode=getattr(self.kv_b_proj, "mode", "affine"),
                )
            w = w.reshape(
                g.num_heads, g.qk_nope_head_dim + g.v_head_dim, g.kv_lora_rank
            ).astype(mx.bfloat16)
            self._w_kb_nope = w[:, : g.qk_nope_head_dim, :]
            self._w_kb_v = w[:, g.qk_nope_head_dim :, :]
            mx.eval(self._w_kb_nope, self._w_kb_v)
        return self._w_kb_nope, self._w_kb_v

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        g = self.geom
        B, S, H = x.shape
        past = int(cache.offset) if cache is not None else 0

        q_lora = self.q_a_layernorm(self.q_a_proj(x))
        if self.apply_rescale:
            q_lora = q_lora * math.sqrt(H / g.q_lora_rank)
        q = self.q_b_proj(q_lora)
        q = q.reshape(B, S, g.num_heads, g.qk_head_dim).transpose(0, 2, 1, 3)
        q_nope = q[..., : g.qk_nope_head_dim]
        q_pe = q[..., g.qk_nope_head_dim :]

        latent = self.kv_a_proj_with_mqa(x)
        kv_a = self.kv_a_layernorm(latent[..., : g.kv_lora_rank])
        if self.apply_rescale:
            kv_a = kv_a * math.sqrt(H / g.kv_lora_rank)

        k_pe = latent[..., g.kv_lora_rank :].reshape(B, 1, S, g.qk_rope_head_dim)
        k_pe = self.k_rope_only_layernorm(k_pe)

        q_pe = mx.fast.rope(
            q_pe,
            g.qk_rope_head_dim,
            traditional=True,
            base=g.rope_theta,
            scale=1.0,
            offset=past,
        )
        k_pe = mx.fast.rope(
            k_pe,
            g.qk_rope_head_dim,
            traditional=True,
            base=g.rope_theta,
            scale=1.0,
            offset=past,
        )

        if isinstance(cache, Dots3LatentCache) or (
            cache is None and _absorb_enabled()
        ):
            out = self._absorbed_attention(
                q_nope, q_pe, kv_a, k_pe, cache, mask, S, x, q_lora, past
            )
        else:
            out = self._materialized_attention(
                q_nope, q_pe, kv_a, k_pe, cache, mask, S, B
            )

        gate = mx.sigmoid(self.g_proj(x))
        out = out * gate[..., None]  # headwise

        out = out.reshape(B, S, g.num_heads * g.v_head_dim)
        return self.o_proj(out)

    def _materialized_attention(
        self, q_nope, q_pe, kv_a, k_pe, cache, mask, S, B
    ) -> mx.array:
        """Stage-A exact path: expand K/V per head and cache them."""
        g = self.geom
        kv = self.kv_b_proj(kv_a)
        kv = kv.reshape(
            B, S, g.num_heads, g.qk_nope_head_dim + g.v_head_dim
        ).transpose(0, 2, 1, 3)
        k_nope = kv[..., : g.qk_nope_head_dim]
        values = kv[..., g.qk_nope_head_dim :]

        queries = mx.concatenate([q_nope, q_pe], axis=-1)
        keys = mx.concatenate(
            [
                k_nope,
                mx.broadcast_to(k_pe, (B, g.num_heads, S, g.qk_rope_head_dim)),
            ],
            axis=-1,
        )

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        if mask is None:
            eff_past = keys.shape[2] - S
            mask = _sliding_causal_mask(
                S, eff_past, g.sliding_window, mx.float32
            )

        out = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=g.scale,
            mask=mask if mask is None else mask.astype(queries.dtype),
        )
        return out.transpose(0, 2, 1, 3)  # [B, S, heads, v]

    def _dsa_gather_attention(
        self,
        queries: mx.array,
        keys: mx.array,
        sel: mx.array,
        past: int,
        S: int,
    ) -> mx.array:
        """Attention over the SELECTED keys only — O(S*topk), not O(S*total).

        ``queries``: absorbed queries [B, h, S, rank+rope]; ``keys``: the
        MQA latent stream [B, 1, total, rank+rope] whose first ``rank``
        dims ARE the values (the absorbed identity this path relies on);
        ``sel``: int32 [B, S, K] per-query key positions from the indexer.

        Each query row attends its OWN key set, so SDPA cannot see one
        shared key axis — the query position folds into the BATCH. Two
        measured traps shaped the formulation (M5 Max, deploy shapes
        T=227 K=2048 D=576, 10 tiles x 13 full layers per 2048 chunk):

        - ``mx.take_along_axis`` with [B, T*K, 1] indices gathers at
          34 GB/s (it broadcasts the index across D); ``mx.take`` on the
          flattened row axis is the embedding-lookup path and runs the
          SAME gather at 785 GB/s (23x, verified byte-identical).
        - presenting the tile as vector-decode ``q [B*T, h, 1, D]`` costs
          104.6 ms/chunk/layer (128 one-row vector ops per tile row);
          relabeling HEADS AS QUERY ROWS — ``q [B*T, 1, h, D]`` against
          ``k [B*T, 1, K, D]`` — is legal because every head of a query
          shares that query's key set and softmax is per-row, and it hits
          the full tiled kernel at 15.8 ms/chunk/layer (6.6x).

        Together they put gathered attention at ~23 ms/chunk/layer FLAT
        vs the dense path's 18.7 (4k) -> 117.7 (25k) ms/chunk/layer.
        The gathered transient is O(T*K*D) and head-independent; T comes
        from ``gather_element_budget`` (the indexer's element-budget
        pattern).

        The causal guard survives the representation change: topk over
        causally -inf'd scores can return future positions when fewer than
        K valid ones exist, so gathered entries with sel > q_pos are
        additive-masked -inf (every query has >= 1 valid entry — ReLU'd
        finite scores always outrank -inf, and self is always causal).

        S==1 keeps the dense path's fp32 SDPA convention (the bf16 absorb
        trap, project_mla_absorb_bug).
        """
        g = self.geom
        B, n_heads, _, D = queries.shape
        rank = g.kv_lora_rank
        K = int(sel.shape[-1])
        kv_flat = keys[:, 0]  # [B, total, D]
        total = int(kv_flat.shape[1])
        fp32 = S == 1
        tile = max(1, min(S, self.gather_element_budget // max(K * D, 1)))
        outs = []
        for s0 in range(0, S, tile):
            s1 = min(s0 + tile, S)
            T = s1 - s0
            sel_t = sel[:, s0:s1]  # [B, T, K]
            if B == 1:
                # Row-take on the flattened key axis (the fast path above).
                gath = mx.take(kv_flat[0], sel_t.reshape(T * K), axis=0)
            else:
                flat = (
                    sel_t.reshape(B, T * K)
                    + (mx.arange(B, dtype=sel_t.dtype) * total)[:, None]
                ).reshape(B * T * K)
                gath = mx.take(kv_flat.reshape(B * total, D), flat, axis=0)
            k_t = gath.reshape(B * T, 1, K, D)
            v_t = k_t[..., :rank]
            # HEADS AS QUERY ROWS: [B, h, T, D] -> [B*T, 1, h, D].
            q_t = (
                queries[:, :, s0:s1]
                .transpose(0, 2, 1, 3)
                .reshape(B * T, 1, n_heads, D)
            )
            q_pos = (past + mx.arange(s0, s1))[None, :, None]
            mask_t = mx.where(
                sel_t > q_pos,
                mx.array(-mx.inf, dtype=mx.float32),
                mx.array(0.0, dtype=mx.float32),
            ).reshape(B * T, 1, 1, K)
            if fp32:
                out_t = mx.fast.scaled_dot_product_attention(
                    q_t.astype(mx.float32),
                    k_t.astype(mx.float32),
                    v_t.astype(mx.float32),
                    scale=g.scale,
                    mask=mask_t,
                ).astype(queries.dtype)
            else:
                out_t = mx.fast.scaled_dot_product_attention(
                    q_t,
                    k_t,
                    v_t,
                    scale=g.scale,
                    mask=mask_t.astype(queries.dtype),
                )
            # [B*T, 1, h, rank] -> [B, h, T, rank]
            outs.append(
                out_t.reshape(B, T, n_heads, rank).transpose(0, 2, 1, 3)
            )
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)

    def _absorbed_attention(
        self, q_nope, q_pe, kv_a, k_pe, cache, mask, S, x=None, q_lora=None, past=0
    ) -> mx.array:
        """Latent-cache path: fold kv_b into the query/output sides.

        ``q_nope' = q_nope @ W_kb_nope`` scores directly against the cached
        latent (identical bilinear form, so the softmax scale is unchanged),
        and ``W_kb_v`` applies AFTER the attention weights. SDPA runs GQA
        with ONE kv head of dim rank+rope (the DSV4 MLA shape).

        🚨 fp32 SDPA at S==1 decode: the absorb path in bf16 is a known
        numerical trap on this stack (project_mla_absorb_bug) — measured
        zero slowdown in fp32, silent degradation in bf16.
        """
        g = self.geom
        w_nope, w_v = self._kb_factors()

        latent_kv = kv_a[:, None]  # [B, 1, S, rank]
        idx_keys = None
        has_indexer = hasattr(self, "indexer") and x is not None
        if cache is not None:
            latent_kv, k_pe = cache.update_and_fetch(latent_kv, k_pe)
            if has_indexer:
                # Full/DSA layers append their indexer key stream on EVERY
                # call — a stream missing early tokens cannot select them
                # once the context crosses the dense-equivalence bound.
                idx_keys = cache.update_indexer(
                    self.indexer.encode_keys(x, past)
                )
        elif has_indexer:
            idx_keys = self.indexer.encode_keys(x, past)

        total = int(latent_kv.shape[2])
        # 🚨 ABSORPTION IS A DECODE OPTIMIZATION, NOT A PREFILL ONE. Scoring
        # against the latent costs (rank+rope) per key instead of
        # (nope+rope), and the value side costs rank instead of v_head_dim:
        # 2.8x the attention FLOPs on full layers and 5.5x on SWA layers
        # (rank 1024 vs nope 192). Materializing K/V from the FETCHED latent
        # once per chunk is O(total) and buys back O(S*total) — exactly what
        # the DSV4 MLA path does (`if L == 1: absorb else: materialize`,
        # runtime_patches/deepseek_v3_patched.py, 2276 pp/s on this box).
        # The cache write above is untouched, so the prefix-cache contract
        # and every stored-block invariant are unchanged; this only changes
        # how the SAME keys are represented for one SDPA call. Bounded by
        # key count because the materialized K/V is a real transient
        # (128 heads x total x 320 dims bf16) and long context must stay on
        # the latent path the 512K economics depend on.
        materialize = (
            S > 1
            and total <= _prefill_materialize_max_keys()
            and w_nope is not None
        )
        # A guard that silently declines turns an A/B into stock-vs-stock.
        # Say ONCE per process which prefill path the engine actually took.
        global _PREFILL_PATH_LOGGED
        if S > 1 and not _PREFILL_PATH_LOGGED:
            _PREFILL_PATH_LOGGED = True
            import logging as _lg

            _lg.getLogger("vmlx_engine").info(
                "dots3 prefill attention path: %s (S=%d total=%d ceiling=%d)",
                "MATERIALIZED-from-latent" if materialize else "absorbed-latent",
                S,
                total,
                _prefill_materialize_max_keys(),
            )

        if not materialize:
            # [B,h,S,nope] @ [1,h,nope,rank] -> [B,h,S,rank]
            q_eff = mx.matmul(q_nope, w_nope[None].astype(q_nope.dtype))
            queries = mx.concatenate([q_eff, q_pe], axis=-1)
            keys = mx.concatenate([latent_kv, k_pe], axis=-1)
            values = latent_kv
        else:
            queries = mx.concatenate([q_nope, q_pe], axis=-1)
            # [B,1,total,rank] x [h,nope,rank] -> [B,h,total,nope]
            k_nope = mx.einsum(
                "bltr,hnr->bhtn", latent_kv, w_nope.astype(latent_kv.dtype)
            )
            keys = mx.concatenate(
                [
                    k_nope,
                    mx.broadcast_to(
                        k_pe,
                        (k_nope.shape[0], g.num_heads, total, g.qk_rope_head_dim),
                    ),
                ],
                axis=-1,
            )
            values = mx.einsum(
                "bltr,hvr->bhtv", latent_kv, w_v.astype(latent_kv.dtype)
            )

        if (
            has_indexer
            and idx_keys is not None
            and total > self.indexer.index_topk
        ):
            # DSA engages: per-query top-k selection replaces plain causal.
            # (At total <= index_topk the top-k selects every causal
            # position, so dense attention is mathematically identical and
            # the scorer is skipped.)
            sel = self.indexer.topk_indices(x, q_lora, idx_keys, past)
            # Gather path composes with the ABSORBED representation only:
            # its keys are one MQA stream shared by all heads and its
            # values are that stream's first `rank` dims, so a single
            # gathered tensor serves K and V for every head. The
            # materialized form is per-head — gathering it would multiply
            # the transient by num_heads and need a second gather for V.
            use_gather = _dsa_gather_enabled() and not materialize
            # A guard that silently declines turns an A/B into
            # stock-vs-stock — say ONCE per process which DSA attention
            # path actually ran.
            global _DSA_PATH_LOGGED
            if not _DSA_PATH_LOGGED:
                _DSA_PATH_LOGGED = True
                import logging as _lg

                _lg.getLogger("vmlx_engine").info(
                    "dots3 DSA attention path: %s (S=%d total=%d topk=%d)",
                    "GATHER-topk" if use_gather else "dense-mask",
                    S,
                    total,
                    self.indexer.index_topk,
                )
            if use_gather:
                out = self._dsa_gather_attention(queries, keys, sel, past, S)
                # [B,h,S,rank] @ [1,h,rank,v] -> [B,h,S,v]
                out = mx.matmul(
                    out, w_v[None].swapaxes(-1, -2).astype(out.dtype)
                )
                return out.transpose(0, 2, 1, 3)  # [B, S, heads, v]
            B = x.shape[0]
            mask = mx.full((B, S, total), -mx.inf, dtype=mx.float32)
            mask = mx.put_along_axis(
                mask,
                sel,
                mx.zeros(sel.shape, dtype=mx.float32),
                axis=-1,
            )
            # topk over causally -inf'd scores can still return future
            # positions when fewer than K valid ones exist — re-mask causal
            # so the scatter cannot unmask the future.
            q_pos = (past + mx.arange(S))[None, :, None]
            k_pos = mx.arange(total)[None, None, :]
            mask = mx.where(k_pos > q_pos, -mx.inf, mask)
            mask = mask[:, None]  # [B, 1, S, total]
        elif mask is None:
            eff_past = total - S
            mask = _sliding_causal_mask(
                S, eff_past, g.sliding_window, mx.float32
            )

        if S == 1:
            out = mx.fast.scaled_dot_product_attention(
                queries.astype(mx.float32),
                keys.astype(mx.float32),
                values.astype(mx.float32),
                scale=g.scale,
                mask=mask if mask is None else mask.astype(mx.float32),
            ).astype(q_nope.dtype)
        else:
            out = mx.fast.scaled_dot_product_attention(
                queries,
                keys,
                values,
                scale=g.scale,
                mask=mask if mask is None else mask.astype(queries.dtype),
            )
        if not materialize:
            # [B,h,S,rank] @ [1,h,rank,v] -> [B,h,S,v]
            out = mx.matmul(out, w_v[None].swapaxes(-1, -2).astype(out.dtype))
        # materialized: values were already v_head_dim, W_kb_v folded in above
        return out.transpose(0, 2, 1, 3)  # [B, S, heads, v]


class Dots3TopkRouter(nn.Module):
    """noaux_tc router with n_group=1 => plain top-k.

    Selection adds ``e_score_correction_bias`` (kept F32); the returned
    weights come from the UNBIASED sigmoid scores, renormalized. Mixing the
    two up is a quiet quality regression, not a crash. ``weight`` stays an
    unquantized bf16 [E, H] (2.5 MB/layer).
    """

    def __init__(self, config: TextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = mx.zeros((config.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((config.n_routed_experts,))

    def __call__(self, x2d: mx.array) -> tuple:
        # Router logits in the ACTIVATION dtype (bf16 deploy) — matching the
        # reference's model-dtype F.linear; fp32 here shifts top-8 on ties.
        logits = x2d @ self.weight.astype(x2d.dtype).T
        scores = mx.sigmoid(logits).astype(mx.float32)
        choice = scores + self.e_score_correction_bias.astype(mx.float32)[None, :]
        inds = mx.argpartition(-choice, kth=self.top_k - 1, axis=-1)[
            ..., : self.top_k
        ]
        w = mx.take_along_axis(scores, inds, axis=-1)
        if self.norm_topk_prob:
            w = w / (w.sum(axis=-1, keepdims=True) + 1e-20)
        return inds.astype(mx.uint32), w * self.routed_scaling_factor


class Dots3DenseMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Dots3MoE(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.gate = Dots3TopkRouter(config)
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
        )
        self.shared_experts = Dots3DenseMLP(
            config.hidden_size,
            config.moe_intermediate_size * config.n_shared_experts,
        )

    def __call__(self, x: mx.array) -> mx.array:
        inds, weights = self.gate(x.reshape(-1, x.shape[-1]))
        inds = inds.reshape(x.shape[0], x.shape[1], -1)
        weights = weights.reshape(x.shape[0], x.shape[1], -1)
        routed = self.switch_mlp(x, inds)
        out = (routed * weights[..., None].astype(routed.dtype)).sum(axis=-2)
        return out + self.shared_experts(x)


class Dots3DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn = Dots3MLAAttention(config, layer_idx)
        if config.is_moe(layer_idx):
            self.mlp = Dots3MoE(config)
        else:
            self.mlp = Dots3DenseMLP(config.hidden_size, config.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class Dots3MTPLayer(nn.Module):
    """MTP decoder = ``model.layers.46``.

    🚨 The MTP layer uses the SWA GEOMETRY, not the full one — measured
    from the real checkpoint shapes (q_b [16384, ...] = 64 heads x 256,
    kv_a 1088 = swa rank 1024 + rope, kv_b 20480, o_proj in 2048, g_proj
    64). The conversion handoff's "full-geom" note is contradicted by the
    weights; transformers ignores these keys so shapes are the only
    authority. NO indexer. Sliding-vs-full masking is indistinguishable in
    the speculative regime (the private cache never approaches window 513).
    Dense FFN, ``shared_head.norm`` before the SHARED backbone lm_head.
    Fusion: eh_proj(cat(enorm(embed(next_tok)), hnorm(prev_hidden))).
    """

    def __init__(self, config: TextConfig):
        super().__init__()
        self.enorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(
            2 * config.hidden_size, config.hidden_size, bias=False
        )
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # Index past the backbone keeps the indexer condition off; the
        # explicit geom override selects SWA per the checkpoint shapes.
        self.self_attn = Dots3MLAAttention(
            config, config.num_hidden_layers, geom=config.swa_geom()
        )
        self.mlp = Dots3DenseMLP(config.hidden_size, config.intermediate_size)

        class _SharedHead(nn.Module):
            def __init__(self, hidden_size: int, eps: float):
                super().__init__()
                self.norm = nn.RMSNorm(hidden_size, eps=eps)

        self.shared_head = _SharedHead(config.hidden_size, config.rms_norm_eps)

    def __call__(
        self,
        embed_next: mx.array,
        prev_hidden: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        x = self.eh_proj(
            mx.concatenate(
                [self.enorm(embed_next), self.hnorm(prev_hidden)], axis=-1
            )
        )
        x = x + self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return self.shared_head.norm(x)


class Dots3MTPEmbeddings(nn.Module):
    """``model.mtp.embed_tokens`` — the MTP layer's OWN embedding table.

    Verified NOT byte-identical to the backbone embedding; aliasing them is
    a silent quality bug.
    """

    def __init__(self, config: TextConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)


class Dots3NoteModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Dots3DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ] + [Dots3MTPLayer(config)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mtp = Dots3MTPEmbeddings(config)

    def _adopt_restored_caches(self, cache: List[Any]) -> None:
        """Re-type prefix-cache restores that came back as generic KVCache.

        The block machinery's positional reconstruction builds a plain
        ``KVCache`` (class identity is not stored per kv block), so a
        restored full-attention layer would otherwise hit the MATERIALIZED
        attention branch with packed latent streams misread as per-head
        K/V — plausible-but-wrong output, no exception (measured live:
        a 30-entry recall answered 24 on the restored path). Convert in
        place; the list object is shared with the scheduler, so the typed
        cache also flows into later stores.
        """
        import logging as _logging

        _logger = _logging.getLogger("vmlx_engine")
        if len(cache) < self.config.num_hidden_layers:
            # A COMPACTED partial reconstruction (KV layers densely packed,
            # sliding layers dropped) has lost the layer-index mapping —
            # adopting by list position would bind full-layer state to the
            # wrong layers. dots3 has no external companion store, so a
            # compacted list is never restorable; the generator discards it
            # as a miss (ledger row 152).
            _logger.warning(
                "dots3 adopt: compacted cache list (%d entries for %d "
                "layers) — layer identity lost; refusing adoption",
                len(cache),
                self.config.num_hidden_layers,
            )
            return
        rope_dim = self.config.qk_rope_head_dim
        idx_dim = self.config.index_head_dim
        rank = self.config.kv_lora_rank
        adopted_n = 0
        adopted_sliding = 0
        foreign: dict = {}
        for i in range(self.config.num_hidden_layers):
            if i >= len(cache):
                continue
            c = cache[i]
            if c is None or isinstance(c, Dots3LatentCache):
                continue
            # Take the LOGICAL extent, never the raw buffers. Restored
            # plain-KV layers are zero-padded up to KVCache.step (256) while
            # `offset` stays logical (prefix_cache.py, 44c20c278). KVCache.state
            # slices to `:offset`; `.keys`/`.values` do not. Adopting the padded
            # buffers made this family attend over zero rows: DSA counts
            # `latent.shape[2]` as real keys while `past` is the logical offset,
            # so the causal frontier lands (physical - logical) tokens behind and
            # the pad rows become legal top-k candidates. Generation then never
            # converges — it runs to max_tokens with no visible answer. Any
            # restore whose cached length is not a multiple of 256 hits this.
            keys, values = _logical_kv_extent(c)
            if self.config.is_sliding(i):
                # Sliding slots left as generic KVCache run the MATERIALIZED
                # lane from the restore point on: 64-head K/V at ~48KB per
                # token per layer, window enforced only by the mask so the
                # buffers never trim. Across 33 sliding layers that is the
                # measured ~1.7MB/token span retention that climbed active
                # memory 96.8->118.7GB over one 12.7k-token prefill. The
                # reconstructor's sliding shells arrive EMPTY (the sliding
                # cumulative tuple only restores on exact boundaries), so
                # re-entering the window-bounded latent lane loses nothing:
                # both lanes mask from PHYSICAL length, and an empty window
                # start forgets the same <window of pre-restore context.
                if keys is None or getattr(keys, "size", 0) == 0:
                    adopted = Dots3LatentCache(
                        window=self.config.sliding_window_size
                    )
                    adopted._rope_dim = rope_dim
                    adopted.offset = int(getattr(c, "offset", 0) or 0)
                    cache[i] = adopted
                    adopted_sliding += 1
                elif (
                    getattr(keys, "ndim", 0) == 4
                    and keys.shape[1] == 1
                    and keys.shape[-1] == rank
                    and values is not None
                    and values.shape[-1] == rope_dim
                ):
                    # A reconstructor that carried the cumulative sliding
                    # payload as (latent, k_pe): re-enter the latent lane
                    # with the state intact.
                    adopted = Dots3LatentCache(
                        window=self.config.sliding_window_size
                    )
                    adopted._rope_dim = rope_dim
                    adopted.latent = keys
                    adopted.k_pe = values
                    adopted.offset = _logical_offset(c, keys)
                    cache[i] = adopted
                    adopted_sliding += 1
                else:
                    foreign.setdefault(type(c).__name__, []).append(
                        (
                            i,
                            tuple(getattr(keys, "shape", ()) or ()),
                            tuple(getattr(values, "shape", ()) or ()),
                        )
                    )
                continue
            if (
                keys is None
                or values is None
                or getattr(keys, "ndim", 0) != 4
                or keys.shape[1] != 1
                or keys.shape[-1] != rope_dim + idx_dim
                or values.shape[-1] != rank
            ):
                foreign.setdefault(type(c).__name__, []).append(
                    (
                        i,
                        tuple(getattr(keys, "shape", ()) or ()),
                        tuple(getattr(values, "shape", ()) or ()),
                    )
                )
                continue
            adopted = Dots3LatentCache()
            adopted._rope_dim = rope_dim
            adopted._idx_dim = idx_dim
            adopted.state = (keys, values)
            adopted.offset = _logical_offset(c, keys)
            cache[i] = adopted
            adopted_n += 1
        # A window=None latent stream must be exactly as long as its offset.
        # If a restore lane ever hands over padded state again, fail LOUD here
        # rather than silently attending to zero rows (vmlx dots3 2026-08-20).
        for i in range(min(len(cache), self.config.num_hidden_layers)):
            slot = cache[i]
            if not isinstance(slot, Dots3LatentCache):
                continue
            if getattr(slot, "window", None) is not None:
                continue
            latent = getattr(slot, "latent", None)
            if latent is None or not getattr(latent, "size", 0):
                continue
            physical = int(latent.shape[2])
            logical = int(getattr(slot, "offset", 0) or 0)
            if logical and physical != logical:
                _logger.error(
                    "dots3 restore invariant violated on layer %d: latent has "
                    "%d rows but offset is %d. The extra rows are padding and "
                    "would be attended as real keys; declining the adoption.",
                    i, physical, logical,
                )
                cache[i] = Dots3LatentCache()
                cache[i]._rope_dim = rope_dim
                cache[i]._idx_dim = idx_dim

        if adopted_n or adopted_sliding or foreign:
            _logger.info(
                "dots3 adopted %d restored full-layer and %d sliding "
                "cache(s) into Dots3LatentCache",
                adopted_n,
                adopted_sliding,
            )
        elif cache and cache[0] is not None and int(getattr(cache[0], "offset", 0)) > 0:
            # Continuation entry with pre-populated caches and nothing to
            # adopt: log ONCE per fresh continuation so restore provenance is
            # visible (offset>0 on entry means restored or resumed state).
            if not getattr(cache[0], "_dots3_adoption_logged", False):
                try:
                    cache[0]._dots3_adoption_logged = True
                    _logger.info(
                        "dots3 continuation entered with typed caches: %s "
                        "offset=%s idx_k=%s",
                        type(cache[0]).__name__,
                        getattr(cache[0], "offset", None),
                        getattr(
                            getattr(cache[0], "idx_k", None), "shape", None
                        ),
                    )
                except Exception:
                    pass
        if foreign:
            # A full layer running a foreign cache class means the absorbed
            # path is OFF for it and the packed streams would be misread —
            # this must be LOUD, never silent.
            _logger.warning(
                "dots3 full layers carry UNADOPTABLE foreign caches: %s",
                {k: v[:2] for k, v in foreign.items()},
            )

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        n_backbone = self.config.num_hidden_layers
        past = 0
        if cache is not None and cache[0] is not None:
            past = int(cache[0].offset)
        seq_len = h.shape[1]
        absorbed = (
            cache is None or isinstance(cache[0], Dots3LatentCache)
            if cache is not None
            else _absorb_enabled()
        )
        if not absorbed and past + seq_len > self.config.index_topk:
            # The materialized stage-A path has no indexer stream; past the
            # dense-equivalence bound it would silently attend wrong.
            raise ValueError(
                f"dots3_note context {past + seq_len} exceeds the DSA "
                f"dense-equivalence bound ({self.config.index_topk}) on the "
                "materialized (VMLX_DOTS3_MLA_ABSORB=0) path"
            )

        if cache is None:
            cache = [None] * n_backbone
        else:
            self._adopt_restored_caches(cache)

        if mask is not None and mask.ndim == 2:
            # 🚨 The MLLM generator forwards the processor's PADDING mask
            # ([B, S] of ones) as ``mask=`` on every media forward (text
            # forwards go through language_model directly and never pass
            # one). Consuming it VERBATIM as an ADDITIVE attention mask is
            # wrong twice over: a uniform +1 per key cancels in softmax, and
            # the -inf causal structure it displaced is GONE — the whole
            # media prefill runs bidirectional. Measured on the real bundle:
            # temp-0 media answers change between mask=[[1]*S] and mask=None
            # (same weights, same mel). 2-D masks carry key-VALIDITY
            # semantics here, never attention structure: all-valid means
            # "build the causal masks yourself"; actual padding would need a
            # physical-layout-aware merge with the sliding lane and this
            # single-sequence engine never produces it — fail loud instead
            # of attending wrong.
            if bool(mx.all(mask == 1).item()):
                mask = None
            else:
                raise ValueError(
                    "dots3_note: 2-D attention mask with masked positions "
                    "(padding) is not supported; this lane serves one "
                    "sequence at a time and refuses to guess an additive "
                    "mask from key-validity input"
                )

        full_mask = None
        swa_mask = None
        if seq_len > 1 or mask is not None:
            if mask is None:
                full_mask = _sliding_causal_mask(seq_len, past, None, mx.float32)
                # 🚨 Sliding-layer masks must be built against the CACHE's
                # PHYSICAL key length, not global positions: the latent
                # sliding cache trims with hysteresis, so once past exceeds
                # window+trim_step the physical length is shorter than
                # past+S and a position-built mask cannot broadcast. Every
                # sliding layer sees the identical stream, so mask=None here
                # lets each attention build the correct physical-layout mask
                # (the materialized path never trims and keeps the shared
                # positional mask).
                if not absorbed:
                    swa_mask = _sliding_causal_mask(
                        seq_len,
                        past,
                        self.config.sliding_window_size,
                        mx.float32,
                    )
            else:
                full_mask = mask
                swa_mask = mask
        # Prefill graph bounding. At deep contexts the lazy graph across all
        # 46 layers holds every layer's DSA score buffers at once (~1GB per
        # full layer at 64k context): the slot census measured cache objects
        # FLAT at 1.1GB while active sawtoothed 99->118GB chunk to chunk,
        # and the admission valve then declines against those transient
        # peaks. Materializing the residual stream every few layers frees
        # each segment's score buffers before the next builds, bounding the
        # in-flight transient to ~eval_every layers' worth. Decode (seq 1)
        # is untouched. mx.eval is semantically neutral — numerics identical.
        eval_every = _prefill_layer_eval_every()
        bound_prefill = eval_every > 0 and seq_len > 1
        for i in range(n_backbone):
            layer_mask = (
                swa_mask if self.config.is_sliding(i) else full_mask
            )
            h = self.layers[i](h, mask=layer_mask, cache=cache[i])
            if bound_prefill and (i + 1) % eval_every == 0:
                mx.eval(h)
        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        text = config if isinstance(config, TextConfig) else config.text_config
        self.text_config = text
        self.model_type = text.model_type
        self.model = Dots3NoteModel(text)
        self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
        **kwargs,
    ) -> mx.array:
        h = self.model(inputs, inputs_embeds=inputs_embeds, mask=mask, cache=cache)
        logits = self.lm_head(h)
        if kwargs.get("return_hidden"):
            return logits, h
        return logits

    def make_cache(self):
        """One cache per BACKBONE layer, matching the layer's attention.

        Default: latent caches (absorbed MLA) — 576 values/token on full
        layers (7.89 GB at 512K), window-bounded 1088 on sliding layers.
        VMLX_DOTS3_MLA_ABSORB=0 reverts to plain materialized KVCaches
        (exact but ~2K practical context). The MTP layer (index 46) is NOT
        part of this list — the speculative path owns its private cache.
        """
        cfg = self.text_config
        if _absorb_enabled():
            return [
                Dots3LatentCache(
                    window=cfg.sliding_window_size if cfg.is_sliding(i) else None
                )
                for i in range(cfg.num_hidden_layers)
            ]
        from mlx_lm.models.cache import KVCache

        return [KVCache() for _ in range(cfg.num_hidden_layers)]

    @property
    def layers(self):
        return self.model.layers[: self.text_config.num_hidden_layers]

    def adopt_prompt_cache(self, cache: List[Any]) -> List[Any]:
        """Re-type restored prefix caches BEFORE generic normalization.

        The generator's ``_fix_hybrid_cache`` type-mismatch repair sees a
        reconstructed plain ``KVCache`` (has .keys) at a slot whose template
        is ``Dots3LatentCache`` (structurally non-attention) and replaces
        the ENTIRE restored full-layer prefix with fresh empties — the same
        failure shape that once wiped Gemma 4's 48-layer prefix. Adopting
        here keeps the restored state and makes the repair a no-op.
        """
        if isinstance(cache, list):
            self.model._adopt_restored_caches(cache)
        return cache

    # ---- native MTP contract --------------------------------------------
    # The MLLM speculative path requires: a non-null ``mtp`` module, a
    # callable ``mtp_forward``, and a callable ``make_mtp_cache``.

    @property
    def mtp(self):
        layers = self.model.layers
        if len(layers) > self.text_config.num_hidden_layers:
            return layers[self.text_config.num_hidden_layers]
        return None

    def make_mtp_cache(self):
        if self.mtp is None:
            return []
        if _absorb_enabled():
            # SWA-geometry layer: window-bounded latent cache (moot in the
            # speculative regime but geometry-consistent).
            return [
                Dots3LatentCache(window=self.text_config.sliding_window_size)
            ]
        from mlx_lm.models.cache import KVCache

        return [KVCache()]

    def mtp_forward(
        self,
        hidden_states: mx.array,
        next_token_ids: mx.array,
        mtp_cache: Optional[List[Any]] = None,
        return_hidden: bool = False,
    ):
        """One MTP draft step.

        ``hidden_states``: backbone hidden of the position(s) PRECEDING
        ``next_token_ids`` ([B, S, H]). The draft embeds the next token
        through the MTP layer's OWN table (verified not byte-identical to
        the backbone embedding — never alias), fuses via eh_proj, runs the
        full-geometry indexer-free MLA + dense FFN, then scores through the
        SHARED backbone lm_head after ``shared_head.norm``.
        """
        mtp_layer = self.mtp
        if mtp_layer is None:
            raise RuntimeError("dots3_note MTP layer is not constructed")
        ids = next_token_ids
        if ids.ndim == 1:
            ids = ids[:, None]
        embed = self.model.mtp.embed_tokens(ids).astype(hidden_states.dtype)
        cache = mtp_cache[0] if mtp_cache else None
        hidden = mtp_layer(embed, hidden_states, cache=cache)
        logits = self.lm_head(hidden)
        if return_hidden:
            return logits, hidden
        return logits
