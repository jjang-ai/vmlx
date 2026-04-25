// SPDX-License-Identifier: Apache-2.0
//
// §407 — DSV4 Compressor + Indexer mask helpers (PR #1195 port).
//
// Reference: ml-explore/mlx-lm PR #1195 head 905df9c2,
// `mlx_lm/models/deepseek_v4.py` lines 823-869, 1095-1108. Local
// mirror of the Python source: `/tmp/pr1195_dsv4.py`. Spec doc at
// `/Users/eric/jang/research/JANGTQ-COMPRESSOR-INDEXER-PROPER-FIX-2026-04-25.md`.
//
// Why this lives in its own file:
//
//   The DSV4 attention forward in `DeepseekV4JANGTQ.swift` currently
//   does *not* integrate the Compressor pool — it runs the bare
//   sliding-window+RoPE attention path, which is in-distribution for
//   FIM short outputs (≤ 128 tokens past prompt) but loses prompt
//   visibility on longer thinking traces (the C5 bug). The proper fix
//   from PR #1195 keeps the Compressor pool flat at `(B, P, head_dim)`
//   and combines a 4D boolean visibility mask:
//
//     visibility = causal_compressed_visibility AND topk_selected
//     mask = concat([window_mask, visibility], axis=-1)
//
//   These helpers compute those compactly. They're independent of the
//   attention forward — usable by any DSV4 attention implementation
//   that decides to integrate Compressor + Indexer. Splitting the
//   helpers out lets us land + test them now without forcing the full
//   attention rewrite in one commit.
//
// Math (for query at sequence position `q_pos`, ratio = compress_ratio):
//
//   window_mask:
//     cache_k ∈ [0, window_len)
//     raw_pos_at_k = (offset + S) - window_len + cache_k
//     visible = (raw_pos_at_k <= q_pos) AND (raw_pos_at_k > q_pos - window)
//
//   compressed_visibility (block-causal staircase):
//     k ∈ [0, compressed_len)
//     compressed row k covers raw positions [k*ratio, (k+1)*ratio)
//     visible = (k+1) * ratio <= q_pos + 1
//
//   topk_selected:
//     selected = (indexer_topk[..., None] == k_range[None, None, None, :]).any(-2)
//
//   Both shapes are `(B, 1, S, L_kv_segment)` so they broadcast onto
//   SDPA's `(B, H, S, L_kv)` attention scores without reshape.

import Foundation
import MLX

public enum DeepseekV4MaskHelpers {

    /// Window-cache visibility mask: True for cache slots a query at
    /// `(b, s)` is allowed to attend to under a sliding-window-of-size
    /// `window` policy on top of a buffer of length `windowLen` whose
    /// content was written at raw positions `[offset+S-windowLen, offset+S)`.
    ///
    /// Returns shape `(B, 1, S, windowLen)`, dtype `.bool`. Broadcasts
    /// over the head axis when passed to SDPA.
    public static func buildWindowMask(
        B: Int, S: Int, offset: Int, window: Int, windowLen: Int
    ) -> MLXArray {
        // q_pos shape (B, S) (broadcast).
        let qPosFlat = MLXArray(Int32(offset)) + MLXArray(0..<Int32(S))
        let qPos = MLX.broadcast(qPosFlat[.newAxis, 0...], to: [B, S])
        let cacheK = MLXArray(0..<Int32(windowLen))
        // raw_pos_at_k shape (windowLen,)
        let rawPosAtK = (MLXArray(Int32(offset + S)) - MLXArray(Int32(windowLen)) + cacheK)
        // (1, 1, windowLen) <= (B, S, 1) → (B, S, windowLen)
        let qPos3 = expandedDimensions(qPos, axis: -1)
        let raw3 = rawPosAtK[.newAxis, .newAxis, 0...]
        let leftEdge = raw3 .<= qPos3
        let rightEdge = raw3 .> (qPos3 - MLXArray(Int32(window)))
        let visible3D = MLX.logicalAnd(leftEdge, rightEdge)
        // Insert head axis at index 1: (B, 1, S, windowLen)
        return expandedDimensions(visible3D, axis: 1)
    }

    /// Block-causal visibility mask for the compressed pool.
    ///
    /// A compressed row `k` aggregates raw positions
    /// `[k*ratio, (k+1)*ratio)`; a query at raw position `q_pos` may
    /// attend to row `k` iff the row's last covered raw position is
    /// ≤ `q_pos`. Encoded as `(k+1) * ratio <= q_pos + 1`.
    ///
    /// Returns shape `(B, 1, S, compressedLen)`, dtype `.bool`.
    public static func compressedVisibility(
        B: Int, S: Int, offset: Int, compressedLen: Int, ratio: Int
    ) -> MLXArray {
        let qPosFlat = MLXArray(Int32(offset)) + MLXArray(0..<Int32(S))
        let qPos = MLX.broadcast(qPosFlat[.newAxis, 0...], to: [B, S])
        let k = MLXArray(0..<Int32(compressedLen))
        let kPlus1Times = (k + MLXArray(Int32(1))) * MLXArray(Int32(ratio))
        // (1, 1, compressedLen) <= (B, S, 1) → (B, S, compressedLen)
        let lhs = kPlus1Times[.newAxis, .newAxis, 0...]
        let rhs = expandedDimensions(qPos + MLXArray(Int32(1)), axis: -1)
        let visible3D = lhs .<= rhs
        return expandedDimensions(visible3D, axis: 1)
    }

    /// Top-k selection visibility: collapses indexer top-k output
    /// `(B, S, top_k)` into a `(B, 1, S, compressedLen)` boolean mask
    /// indicating which compressed rows the indexer selected for each
    /// query position.
    ///
    /// `indexerTopk[..., None] == k_range[None, None, None, :]` produces
    /// `(B, S, top_k, compressedLen)`; `.any(axis: -2)` collapses to
    /// `(B, S, compressedLen)`; insert the head axis to match the other
    /// helpers.
    public static func indexerSelected(
        indexerTopk: MLXArray, compressedLen: Int
    ) -> MLXArray {
        let kRange = MLXArray(0..<Int32(compressedLen))
        let topkExp = expandedDimensions(indexerTopk, axis: -1)
        let kRangeBroadcast = kRange[.newAxis, .newAxis, .newAxis, 0...]
        let eq = topkExp .== kRangeBroadcast
        let selected3D = eq.any(axis: -2)
        return expandedDimensions(selected3D, axis: 1)
    }
}
