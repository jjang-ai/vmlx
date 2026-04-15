// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import MLX

// MARK: - Batch Causal Mask Generation

/// Create a batch-aware causal attention mask for sequences at different positions.
///
/// Each sequence in the batch may be at a different generation step (different
/// cache offset). This function builds a per-sequence causal mask that:
/// - Allows each query to attend only to keys at positions <= its own position
/// - Masks out padding positions from shorter sequences in the batch
/// - Optionally applies sliding window constraints
///
/// The returned mask has shape `[B, 1, queryLen, totalKeyLen]` where `totalKeyLen`
/// is the maximum cached length across all sequences plus `queryLen`.
///
/// - Parameters:
///   - queryLen: Number of new tokens being processed (1 for decode, chunk size for prefill)
///   - offsets: Per-sequence cache offsets — each element is the number of tokens
///     already in that sequence's cache **before** this step's tokens are added.
///     Shape requirement: must have exactly `B` elements.
///   - windowSize: Optional sliding window size for sliding-window attention layers.
///     When set, keys outside the window are masked out.
/// - Returns: Boolean `MLXArray` of shape `[B, 1, queryLen, totalKeyLen]`.
public func createBatchCausalMask(
    queryLen n: Int,
    offsets: [Int],
    windowSize: Int? = nil
) -> MLXArray {
    let B = offsets.count
    precondition(B > 0, "createBatchCausalMask requires at least one sequence")

    // Total key length = max(offset + n) across all sequences.
    // Each sequence's keys span [0, offset_i + n). Pad shorter ones to this total.
    let maxTotal = offsets.map { $0 + n }.max()!

    // Key column indices: [0, 1, ..., maxTotal - 1], shape [1, maxTotal]
    let rinds = MLXArray(Int32(0) ..< Int32(maxTotal)).reshaped(1, maxTotal)

    // Build per-sequence masks and stack
    var masks = [MLXArray]()
    masks.reserveCapacity(B)

    for offset in offsets {
        // Query row indices for this sequence: [offset, offset+1, ..., offset+n-1]
        // Shape: [n, 1]
        let linds = (MLXArray(Int32(0) ..< Int32(n)) + Int32(offset)).reshaped(n, 1)

        // Standard causal: query at position q can attend to key at position k if k <= q
        var mask = linds .>= rinds

        // Sliding window: additionally require k >= q - windowSize + 1
        if let windowSize {
            mask = mask & (rinds .>= (linds - Int32(windowSize - 1)))
        }

        // Also mask out positions beyond this sequence's actual cached range.
        // Keys at positions >= offset + n are padding from other (longer) sequences.
        mask = mask & (rinds .< Int32(offset + n))

        // Shape: [1, 1, n, maxTotal] — one mask per sequence
        masks.append(mask.reshaped(1, 1, n, maxTotal))
    }

    // Stack to [B, 1, n, maxTotal]
    return concatenated(masks, axis: 0)
}
