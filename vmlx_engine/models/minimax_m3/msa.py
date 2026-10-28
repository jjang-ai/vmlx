# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 MSA (block-sparse attention): the selection and the attention paths that consume it.

The Lightning indexer picks, for every query row and KV group, ``keep`` (16) key blocks of ``block`` (128) keys:
the top block scores, the query's own (local) block forced in. Attention then sees only those blocks, causally.

``MSASelection`` carries that pick as block ids ([B, G, Sq, keep]) plus the positions it was made at. Paths
(``choose_path``; ``VMLX_MINIMAX_MSA`` forces one for comparisons):

* ``dense``: the additive [B, G, Sq, Sk] float32 key mask the runtime used before (``dense_bias``, same ops) and
  one SDPA over all Sk keys. The default.
* ``gather``: pure MLX ops: gather the selected keys / values of every (row, group) out of the cache (keep * block
  keys) and run SDPA over those. Each row is its own SDPA batch element, so a row's output is the same alone or
  with other rows.

Positions: query row r sits at ``offset + r``; key j is allowed when its block is selected and ``j <= offset + r``.
Keys at or past ``sk`` (the logical extent, i.e. the cache offset) are never read: the gather clamps to ``sk - 1``
and masks the clamped lanes. A partial last block, a query's own block (causal inside it) and selected blocks that
lie entirely in the future (possible when fewer than ``keep`` blocks precede a prefill row) are all handled by that
one positional rule.

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

import os

import mlx.core as mx

# Path override for comparisons and diagnostics: "dense" (the default) or "gather" (the selected keys only).
ENV_PATH = "VMLX_MINIMAX_MSA"


class MSASelection:
    """Top-k key blocks per (query row, KV group) for queries at ``offset`` .. ``offset + Sq - 1`` over keys
    ``0 .. sk - 1``. ``top_idx`` [B, G, Sq, keep] holds distinct block ids per row (argpartition order)."""

    __slots__ = ("top_idx", "offset", "sk", "block")

    def __init__(self, top_idx: mx.array, offset: int, sk: int, block: int):
        self.top_idx = top_idx
        self.offset = int(offset)
        self.sk = int(sk)
        self.block = int(block)

    @property
    def shape(self):
        """[B, G, Sq, Sk]: the shape of the equivalent key mask."""
        B, G, Sq, _ = self.top_idx.shape
        return (B, G, Sq, self.sk)

    @property
    def n_blocks(self) -> int:
        return -(-self.sk // self.block)

    @property
    def keep(self) -> int:
        return int(self.top_idx.shape[-1])

    def dense_bias(self) -> mx.array:
        """The additive key mask [B, G, Sq, Sk] (float32 0 / -inf, causal) -- the pre-block-sparse runtime's ops."""
        B, G, Sq, _ = self.top_idx.shape
        q_pos = mx.arange(self.offset, self.offset + Sq).reshape(1, 1, Sq, 1)
        k_pos = mx.arange(self.sk).reshape(1, 1, 1, self.sk)
        block_keep = mx.full((B, G, Sq, self.n_blocks), -mx.inf, dtype=mx.float32)
        block_keep = mx.put_along_axis(block_keep, self.top_idx, mx.array(0.0, dtype=mx.float32), axis=-1)
        key_bias = mx.repeat(block_keep, self.block, axis=-1)[..., : self.sk]
        return mx.where(k_pos > q_pos, -mx.inf, key_bias)


def requested_path() -> str:
    raw = os.environ.get(ENV_PATH, "").strip().lower()
    return raw if raw in ("dense", "gather") else "dense"


def _key_positions(sel: MSASelection):
    """(source index [B, G, Sq, N] clamped below sk, allowed [B, G, Sq, N]) for the N = keep * block selected keys of
    every row, blocks in ascending order (keys in position order, as the dense path visits them)."""
    B, G, Sq, keep = sel.top_idx.shape
    blocks = mx.sort(sel.top_idx.astype(mx.int32), axis=-1)
    pos = blocks[..., None] * sel.block + mx.arange(sel.block, dtype=mx.int32)
    pos = pos.reshape(B, G, Sq, keep * sel.block)
    q_pos = mx.arange(sel.offset, sel.offset + Sq, dtype=mx.int32).reshape(1, 1, Sq, 1)
    allowed = pos <= q_pos
    return mx.minimum(pos, sel.sk - 1), allowed


def _check_extent(k: mx.array, v: mx.array, sel: MSASelection) -> None:
    if int(k.shape[2]) < sel.sk or int(v.shape[2]) < sel.sk:
        # The selection was made over more keys than the attention sees: the index lane and the K / V lane have
        # desynchronised. Reading on would gather rows that were never written.
        raise ValueError(
            f"MSA selection covers {sel.sk} keys but K / V hold {k.shape[2]} / {v.shape[2]}")
    if sel.offset + int(sel.top_idx.shape[2]) > sel.sk:
        raise ValueError(
            f"MSA query rows end at {sel.offset + int(sel.top_idx.shape[2])} past the {sel.sk} selected keys")


def gathered_attention(q: mx.array, k: mx.array, v: mx.array, sel: MSASelection, scale: float) -> mx.array:
    """Attention [B, H, L, D] of q over only the selected key blocks of each (row, KV group).

    q [B, H, L, D] (query heads contiguous per KV group); k / v [B, G, >= sk, D]. Every row is one SDPA batch
    element over its own keep * block gathered keys, so row r's output is the same whether it runs alone (a decode
    step) or with other rows (a verify)."""
    B, H, L, D = q.shape
    G = int(sel.top_idx.shape[1])
    Hg = H // G
    _check_extent(k, v, sel)
    src, allowed = _key_positions(sel)                                  # [B, G, L, N]
    N = int(src.shape[-1])
    bias = mx.where(allowed, mx.array(0.0, dtype=q.dtype), mx.array(-mx.inf, dtype=q.dtype))
    if L == 1:
        idx = src.reshape(B, G, N, 1)
        ks = mx.take_along_axis(k, idx, axis=2)                         # [B, G, N, D]
        vs = mx.take_along_axis(v, idx, axis=2)
        o = mx.fast.scaled_dot_product_attention(
            q.reshape(B * G, Hg, 1, D), ks.reshape(B * G, 1, N, D), vs.reshape(B * G, 1, N, D),
            scale=scale, mask=bias.reshape(B * G, 1, 1, N))
        return o.reshape(B, H, 1, D)
    idx = src[..., None]                                                # [B, G, L, N, 1]
    ks = mx.take_along_axis(k[:, :, None], idx, axis=3)                 # [B, G, L, N, D]
    vs = mx.take_along_axis(v[:, :, None], idx, axis=3)
    qr = q.reshape(B, G, Hg, L, D).transpose(0, 1, 3, 2, 4).reshape(B * G * L, Hg, 1, D)
    o = mx.fast.scaled_dot_product_attention(
        qr, ks.reshape(B * G * L, 1, N, D), vs.reshape(B * G * L, 1, N, D),
        scale=scale, mask=bias.reshape(B * G * L, 1, 1, N))
    return o.reshape(B, G, L, Hg, D).transpose(0, 1, 3, 2, 4).reshape(B, H, L, D)


def choose_path(sel: MSASelection, q: mx.array, k, mask, n_kv: int) -> str:
    """'dense' or 'gather' for this call. Only picks a path; both compute the same attention."""
    if not isinstance(k, mx.array) or int(sel.top_idx.shape[1]) != n_kv or isinstance(mask, mx.array):
        # Quantized K / V, indexer heads that do not line up with KV groups, or an extra array mask (batch
        # padding): the dense mask path covers these as before.
        return "dense"
    return "gather" if requested_path() == "gather" else "dense"
