// CompilableKVCache: Fixed-size KV cache using the Overflow Bin pattern.
//
// Standard KVCacheSimple returns keys[..<offset] — a dynamically sized slice that
// changes every decode step. This prevents compile() from tracing through the cache
// because DynamicSlice requires static slice_size.
//
// CompilableKVCache solves this by:
// 1. Pre-allocating a fixed-size buffer [B, H, maxLength, D]
// 2. Writing new keys/values via DynamicSliceUpdate (compile-traceable writes)
// 3. Returning the FULL buffer from update() — constant shape every step
// 4. Generating a boolean attention mask in makeMask() that marks active positions
//
// The attention kernel handles the masking, computing only on valid positions.
// This trades marginal redundant compute (masked zeros) for enabling compile()
// to fuse hundreds of FFI crossings into a single compiled call.
//
// Usage:
//   // After prefill with standard cache, convert for compiled decode:
//   let compilableCache = standardCache.map { c in
//       CompilableKVCache(from: c, maxLength: 2048)
//   }

import Cmlx
import Foundation
import MLX
import MLXNN

/// A KV cache that returns fixed-size buffers to enable compile().
///
/// Key differences from KVCacheSimple:
/// - `offsetArray` (MLXArray) tracks position in the computation graph
/// - Pre-allocated buffer of fixed size (no dynamic growth during decode)
/// - `update()` returns the FULL buffer — mask handles which positions are valid
/// - `makeMask()` always returns an array mask covering the full buffer
/// - `innerState()` returns [keys, values, offsetArray] — all compile-tracked
public class CompilableKVCache: BaseKVCache {

    public var keys: MLXArray?
    public var values: MLXArray?

    /// Offset as MLXArray (1D [1] int32) — tracked by compile tracer.
    /// Must be 1D (not scalar) for DynamicSlice start parameter compatibility.
    public var offsetArray: MLXArray

    /// Maximum sequence length the buffer can hold.
    public let maxLength: Int

    /// Pre-allocation chunk size (same semantics as KVCacheSimple.step).
    public var step: Int

    /// Pre-computed column indices for mask creation [0, 1, ..., maxLength-1].
    /// Avoids re-creating every step.
    private lazy var maskRinds: MLXArray = MLXArray(Int32(0) ..< Int32(maxLength))

    public init(maxLength: Int = 4096, step: Int = 256) {
        self.maxLength = maxLength
        self.step = step
        self.offsetArray = MLXArray([Int32(0)])
        super.init()
    }

    /// Create from an existing KVCacheSimple (e.g., after prefill).
    /// Copies the existing cache state into a fixed-size buffer.
    public convenience init(from cache: KVCache, maxLength: Int = 4096) {
        self.init(maxLength: maxLength)

        let existingState = cache.state
        if existingState.count >= 2 {
            let existingKeys = existingState[0]  // [B, H, seqLen, D]
            let existingValues = existingState[1]

            let seqLen = existingKeys.dim(2)
            let B = existingKeys.dim(0)
            let H = existingKeys.dim(1)
            let kD = existingKeys.dim(3)
            let vD = existingValues.dim(3)

            // Pre-allocate to maxLength
            self.keys = MLXArray.zeros([B, H, maxLength, kD], dtype: existingKeys.dtype)
            self.values = MLXArray.zeros([B, H, maxLength, vD], dtype: existingValues.dtype)

            // Copy existing data at position 0
            self.keys![.ellipsis, ..<seqLen, 0...] = existingKeys
            self.values![.ellipsis, ..<seqLen, 0...] = existingValues

            self.offsetArray = MLXArray([Int32(seqLen)])
        }
    }

    // MARK: - KVCache protocol

    public override var offset: Int {
        get {
            // Materialize for compatibility with code that reads offset as Int.
            // This triggers synchronous readback — avoid inside compiled paths.
            offsetArray[0].item(Int.self)
        }
        set {
            offsetArray = MLXArray([Int32(newValue)])
        }
    }

    public override func innerState() -> [MLXArray] {
        if let keys, let values {
            return [keys, values, offsetArray]
        }
        return [offsetArray]
    }

    public override func update(keys newKeys: MLXArray, values newValues: MLXArray)
        -> (MLXArray, MLXArray)
    {
        let nTokens = newKeys.dim(2)

        // Lazy initialization on first call
        if self.keys == nil {
            let B = newKeys.dim(0)
            let H = newKeys.dim(1)
            let kD = newKeys.dim(3)
            let vD = newValues.dim(3)
            self.keys = MLXArray.zeros([B, H, maxLength, kD], dtype: newKeys.dtype)
            self.values = MLXArray.zeros([B, H, maxLength, vD], dtype: newValues.dtype)
        }

        let prev = offsetArray
        let newOffset = prev + MLXArray([Int32(nTokens)])

        // Must use _updateInternal to preserve object identity — compile() captures
        // stateInputs at innerCall start and expects the same objects to be mutated.
        self.keys!._updateInternal(
            dynamicSliceUpdate(self.keys!, update: newKeys, start: prev, axes: [2]))
        self.values!._updateInternal(
            dynamicSliceUpdate(self.values!, update: newValues, start: prev, axes: [2]))

        self.offsetArray._updateInternal(newOffset)

        // OVERFLOW BIN: return the full static-size buffer.
        // The attention mask from makeMask() handles which positions are valid.
        // This keeps tensor shapes constant across all decode steps,
        // enabling compile() to trace the entire forward pass.
        return (self.keys!, self.values!)
    }

    // MARK: - Mask (Overflow Bin)

    /// Generate attention mask for the full-buffer return.
    ///
    /// Since update() returns the entire maxLength buffer, we ALWAYS need an array
    /// mask to prevent attention to unwritten positions. The mask is boolean:
    /// True = attend, False = don't attend (gets -inf in attention scores).
    ///
    /// For decode (n=1): mask[0, j] = (j <= offset)
    /// For prefill (n>1): mask[i, j] = (j <= offset + i)  (causal)
    ///
    /// Note: `offset` here is the PRE-update value. After update, positions
    /// 0..<offset+n are valid, matching the mask exactly.
    ///
    /// Uses `offsetArray` (MLXArray) for all computation so compile() can trace
    /// the mask through the computation graph.
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        // Use offsetArray directly — compile-traceable, no .item() needed
        let currentOffsetArr = offsetArray  // MLXArray [1] int32

        // Query positions: [offset, offset+1, ..., offset+n-1]
        let linds: MLXArray
        if n == 1 {
            linds = currentOffsetArr.reshaped(1, 1)
        } else {
            linds = (MLXArray(Int32(0) ..< Int32(n)) + currentOffsetArr).reshaped(n, 1)
        }

        // Key positions: [0, 1, ..., maxLength-1]
        let rinds = maskRinds.reshaped(1, maxLength)

        // Causal + validity: attend to positions j where j <= query_position
        var mask = linds .>= rinds

        // Apply sliding window if specified
        if let windowSize {
            let windowStart = linds - Int32(windowSize - 1)
            mask = mask & (rinds .>= windowStart)
        }

        return .array(mask)
    }

    // MARK: - State

    public override var state: [MLXArray] {
        get {
            guard let keys, let values else { return [] }
            let off: Int = offsetArray[0].item(Int.self)
            if off == keys.dim(2) {
                return [keys, values]
            } else {
                // Return only valid portion for serialization
                return [
                    keys[.ellipsis, ..<off, 0...],
                    values[.ellipsis, ..<off, 0...],
                ]
            }
        }
        set {
            guard newValue.count == 2 else { return }
            let seqLen = newValue[0].dim(2)
            let B = newValue[0].dim(0)
            let H = newValue[0].dim(1)
            let kD = newValue[0].dim(3)
            let vD = newValue[1].dim(3)

            self.keys = MLXArray.zeros([B, H, maxLength, kD], dtype: newValue[0].dtype)
            self.values = MLXArray.zeros([B, H, maxLength, vD], dtype: newValue[1].dtype)
            self.keys![.ellipsis, ..<seqLen, 0...] = newValue[0]
            self.values![.ellipsis, ..<seqLen, 0...] = newValue[1]
            self.offsetArray = MLXArray([Int32(seqLen)])
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let current: Int = offsetArray[0].item(Int.self)
        let trimmed = min(current, n)
        offsetArray = MLXArray([Int32(current - trimmed)])
        super.offset = current - trimmed
        return trimmed
    }

    public override func copy() -> any KVCache {
        let c = CompilableKVCache(maxLength: maxLength, step: step)
        c.keys = keys
        c.values = values
        c.offsetArray = offsetArray
        return c
    }

    // MARK: - Debug

    public var debugDescription: String {
        "CompilableKVCache(offset=\(offset), maxLength=\(maxLength), "
            + "shape=\(keys?.shape.description ?? "nil"))"
    }
}
