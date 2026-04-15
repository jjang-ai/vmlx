// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import MLX
import MLXNN

// MARK: - BatchKVCache

/// A KV cache wrapper that presents multiple independent per-sequence caches
/// as a single batched cache to the model's forward pass.
///
/// ## How It Works
///
/// Each active sequence in the batch engine owns its own `KVCache` (typically
/// `KVCacheSimple` with B=1). During a batched decode step, the engine constructs
/// one `BatchKVCache` **per model layer** by collecting each sequence's cache for
/// that layer.
///
/// The model calls `cache.update(keys:values:)` with tensors shaped `[B, H, L, D]`.
/// `BatchKVCache` splits along dim 0, dispatches each slice to the corresponding
/// sequence's cache, pads shorter results to a common length, and stacks them back
/// into `[B, H, maxLen, D]` for the attention computation.
///
/// ## RoPE Integration
///
/// `offsetArray` is an `MLXArray` of shape `[B]` containing each sequence's current
/// position. `applyRotaryPosition` detects `BatchKVCache` at runtime and routes
/// to the `MLXArray`-offset RoPE overload, giving each sequence correct positional
/// encoding without any model code changes.
///
/// ## Mask Integration
///
/// `makeMask()` generates per-sequence causal masks via ``createBatchCausalMask``,
/// correctly handling different cache lengths and optional sliding windows.
///
/// ## Extensibility
///
/// The underlying slot caches are typed as `KVCache` (protocol). This allows future
/// cache types (TurboQuant, paged caches) to be used as slot caches — the split/pad/stack
/// logic remains the same as long as `update()` returns `[1, H, seqLen, D]` tensors.
public final class BatchKVCache: BaseKVCache {

    /// Per-sequence caches for this layer. Index matches batch dimension ordering.
    private let slotCaches: [KVCache]

    /// Number of sequences in this batch.
    public let batchSize: Int

    /// Per-sequence position offsets as `[B]`-shaped `MLXArray`.
    ///
    /// Updated after each `update()` call. Used by `applyRotaryPosition` to
    /// provide per-sequence RoPE positions.
    public private(set) var offsetArray: MLXArray

    /// Create a `BatchKVCache` for one model layer.
    ///
    /// - Parameter slotCaches: One cache per active sequence, all for the same
    ///   model layer. Must not be empty.
    public init(slotCaches: [KVCache]) {
        precondition(!slotCaches.isEmpty, "BatchKVCache requires at least one slot cache")
        self.slotCaches = slotCaches
        self.batchSize = slotCaches.count
        self.offsetArray = MLXArray(slotCaches.map { Int32($0.offset) })
        super.init()
        // Scalar offset = max across sequences. Used for mask sizing (total key length).
        self.offset = slotCaches.map(\.offset).max() ?? 0
    }

    // MARK: - KVCache Protocol

    /// Update the cache with new keys and values from a batched forward pass.
    ///
    /// Input shapes: `[B, H, L, D]` where B = `batchSize`, L = query length
    /// (typically 1 for decode).
    ///
    /// Splits along dim 0, dispatches to each sequence's cache, pads shorter
    /// results to the maximum sequence length, and stacks back to `[B, H, maxLen, D]`.
    ///
    /// - Parameters:
    ///   - keys: Batched key tensor `[B, H, L, D]`
    ///   - values: Batched value tensor `[B, H, L, D]`
    /// - Returns: Padded and stacked `(keys, values)` each shaped `[B, H, maxLen, D]`
    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let B = keys.dim(0)
        precondition(B == batchSize, "Key batch size \(B) != expected \(batchSize)")

        var allKeys = [MLXArray]()
        var allValues = [MLXArray]()
        allKeys.reserveCapacity(B)
        allValues.reserveCapacity(B)

        for i in 0 ..< B {
            // Extract this sequence's slice: [1, H, L, D]
            let ki = keys[i ..< i + 1]
            let vi = values[i ..< i + 1]

            // Dispatch to the sequence's own cache
            let (ck, cv) = slotCaches[i].update(keys: ki, values: vi)
            // ck shape: [1, H, seqLen_i, D] — seqLen varies per sequence
            allKeys.append(ck)
            allValues.append(cv)
        }

        // Pad shorter sequences and stack to [B, H, maxLen, D]
        let paddedKeys = padAndConcatenate(allKeys, along: 2)
        let paddedValues = padAndConcatenate(allValues, along: 2)

        // Update offset tracking
        self.offsetArray = MLXArray(slotCaches.map { Int32($0.offset) })
        self.offset = slotCaches.map(\.offset).max() ?? 0

        return (paddedKeys, paddedValues)
    }

    /// Generate a batch-aware causal attention mask.
    ///
    /// Always returns `.array(mask)` because batched sequences at different
    /// positions cannot use the symbolic `.causal` or `.none` shortcuts.
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let offsets = slotCaches.map(\.offset)
        return .array(createBatchCausalMask(queryLen: n, offsets: offsets, windowSize: windowSize))
    }

    public override var maxSize: Int? { nil }

    // MARK: - Unsupported Operations

    // BatchKVCache is a transient view — it is not serializable, trimmable, or copyable.

    public override var state: [MLXArray] {
        get { [] }
        set { }
    }

    public override var metaState: [String] {
        get { [""] }
        set { }
    }

    public override var isTrimmable: Bool { false }

    public override func copy() -> any KVCache {
        fatalError("BatchKVCache is a transient view and cannot be copied")
    }

    // MARK: - Internal Helpers

    /// Pad arrays to the same length along `axis` and concatenate along dim 0.
    ///
    /// Each input array has shape `[1, H, seqLen_i, D]`. The result is
    /// `[B, H, maxSeqLen, D]` where shorter sequences are zero-padded on the right.
    /// Zero-padded positions are masked out by `makeMask()` so they never contribute
    /// to attention scores.
    private func padAndConcatenate(_ arrays: [MLXArray], along axis: Int) -> MLXArray {
        let maxLen = arrays.map { $0.dim(axis) }.max() ?? 0

        let padded: [MLXArray] = arrays.map { arr in
            let currentLen = arr.dim(axis)
            guard currentLen < maxLen else { return arr }

            // Build zero-padding tensor with same shape except along `axis`
            var paddingShape = arr.shape
            paddingShape[axis] = maxLen - currentLen
            let pad = MLXArray.zeros(paddingShape, dtype: arr.dtype)
            return concatenated([arr, pad], axis: axis)
        }

        // Each is [1, H, maxLen, D] — concatenate along axis 0 to get [B, H, maxLen, D]
        return concatenated(padded, axis: 0)
    }
}

