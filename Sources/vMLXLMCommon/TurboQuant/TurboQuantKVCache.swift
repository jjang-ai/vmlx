// Copyright © 2025 Osaurus & JANG. All rights reserved.
// TurboQuant — arXiv:2504.19874 (Google DeepMind)

import Foundation
import MLX

/// Phase of the TurboQuant KV cache lifecycle.
///
/// ```
/// FILL ──────────────────► COMPRESSED
///   (prefill, float KV)      (encoded prefix + float window)
///   append via update()      new tokens → window via update()
///                            attention reads unified float buffer
/// ```
public enum TQPhase: Sendable {
    /// Accumulating float KV during prefill. Standard KVCacheSimple behavior.
    case fill
    /// Prefix is compressed. New decode tokens go to float window.
    /// Models see float arrays — no special handling needed.
    case compressed
}

/// TurboQuant-backed KV cache for a single attention layer.
///
/// ## Lifecycle
///
/// 1. **Fill phase** (prefill): Behaves identically to `KVCacheSimple`.
///    Float keys/values accumulate via `update()`. Zero overhead.
///
/// 2. **Compress** (triggered by `maybeQuantizeKVCache`): Creates a
///    `TurboQuantKVCache` from the `KVCacheSimple`, encoding all float
///    KV data into packed compressed format. The compressed data is decoded
///    once into a persistent float buffer. Original float cache is freed.
///
/// 3. **Generate phase** (token-by-token): New tokens are scatter-written
///    into pre-allocated window slots in the unified buffer. Attention reads
///    from `[decoded_prefix | window]` — all float. Models see normal
///    MLXArrays from `update()`, identical to `KVCacheSimple`.
///
/// ## Why Models Don't Need Changes
///
/// Unlike `QuantizedKVCache` (which returns quantized tuples and requires
/// `updateQuantized()` + `quantizedScaledDotProductAttention()`), TurboQuant
/// decodes back to float. The unified buffer is float16. So the normal
/// `update()` + `scaledDotProductAttention()` path works unchanged.
///
/// ## Memory Layout (compressed phase)
///
/// ```
/// Compressed storage (GPU, persistent):
///   encodedKeys   — packed uint32 indices + QJL signs + norms
///   encodedValues — packed uint32 indices + norms
///
/// Unified float buffer (GPU, for attention reads):
///   [decoded_prefix (T tokens) | window_slots (256 pre-allocated)]
///   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
///   Single contiguous MLXArray. Attention reads a slice of this.
///   New tokens scatter-write to window_slots[windowOffset].
/// ```
///
/// ## Trim Behavior (speculative decoding compatibility)
///
/// - Window trim (fast): If target offset is within prefix + window,
///   just adjust `windowOffset` — O(1), no re-encode.
/// - Full reset: If trim below prefix boundary, decode → truncate → re-encode.
///   This only happens if speculative decoding rejects ALL generated tokens
///   AND the prefix, which is extremely rare.
public final class TurboQuantKVCache: BaseKVCache, @unchecked Sendable {

    public private(set) var phase: TQPhase = .fill

    // Fill phase storage
    private var floatKeys: MLXArray?
    private var floatValues: MLXArray?

    // Compressed phase storage
    public private(set) var compressedKeys: EncodedKeys?
    public private(set) var compressedValues: EncodedValues?

    // Decoded prefix (persistent float buffer for attention reads)
    private var decodedKeyBuffer: MLXArray?
    private var decodedValueBuffer: MLXArray?

    // Unified buffer: [decoded_prefix | window_slots]
    private var unifiedKeys: MLXArray?
    private var unifiedValues: MLXArray?

    /// Number of tokens in the decoded prefix.
    private var prefixTokenCount: Int = 0

    /// Pre-allocated window step size. Window grows in chunks to avoid per-token allocation.
    private let windowStep = 256

    /// Number of tokens written into the window region of the unified buffer.
    private var windowOffset = 0

    /// Encoder state (codebooks, rotation signs, QJL matrix).
    /// Created during compression, reused for any subsequent decode operations.
    private var encoderState: TQEncoder.EncoderState?

    // Configuration
    public let keyBits: Int
    public let valueBits: Int
    public let sinkTokens: Int

    // MARK: - Init

    /// Create a TurboQuantKVCache.
    ///
    /// Typically not called directly — created by `maybeQuantizeKVCache`
    /// from an existing `KVCacheSimple`.
    ///
    /// - Parameters:
    ///   - keyBits: Total bits per key element (e.g., 3 = 2-bit codebook + 1-bit QJL)
    ///   - valueBits: Total bits per value element (e.g., 3 = 3-bit codebook)
    ///   - sinkTokens: Tokens to preserve at full precision (default 4)
    public init(keyBits: Int = 3, valueBits: Int = 3, sinkTokens: Int = 4) {
        self.keyBits = keyBits
        self.valueBits = valueBits
        self.sinkTokens = sinkTokens
        super.init()
    }

    // MARK: - Restore from disk

    /// Reload the compressed phase of this cache from already-encoded
    /// `EncodedKeys`/`EncodedValues` payloads (e.g. coming back from
    /// `DiskCache.fetch` via `TQDiskSerializer.deserializeIndexed`).
    ///
    /// This is the in-place inverse of `compressFloatKV`'s post-encode tail.
    /// It rebuilds the encoder state (which is fully deterministic from
    /// `(dim, keyBits, valueBits, seed)`), decodes the prefix into a
    /// persistent buffer, and pre-allocates the unified buffer with
    /// `windowStep` slots so subsequent token writes work normally. After
    /// this call the cache is in `.compressed` phase with the supplied
    /// offset.
    ///
    /// - Parameters:
    ///   - encodedKeys: Compressed key payload, as produced by `TQEncoder.encodeKeys`.
    ///   - encodedValues: Compressed value payload.
    ///   - sourceOffset: The token offset the cache should report after restore
    ///     (i.e. how many tokens of context the compressed prefix represents).
    public func restoreCompressed(
        encodedKeys: EncodedKeys,
        encodedValues: EncodedValues,
        sourceOffset: Int
    ) {
        // Derive encoder params from the encoded payload itself. They are
        // fully determined by the dim/indexBits/seed fields, so we don't
        // need to store anything extra on disk to recover them.
        let dim = encodedKeys.shape.last ?? 0
        guard dim > 0 else { return }
        let kBits = encodedKeys.indexBits + 1   // keys split (b-1) MSE bits + 1 QJL bit
        let vBits = encodedValues.indexBits     // values use all b bits for MSE
        let seed = encodedKeys.seed

        let state = TQEncoder.EncoderState(
            dim: dim, keyBits: kBits, valueBits: vBits, seed: seed)
        self.encoderState = state

        let dKeys = TQEncoder.decodeKeys(encodedKeys, state: state)
        let dValues = TQEncoder.decodeValues(encodedValues, state: state)

        // Materialize lazily-built tensors immediately (same pattern as
        // compressFloatKV) so MLX doesn't graph 30+ layers at once on a
        // multi-layer restore. Calls MLX.eval which is the lazy-tensor
        // materializer in the MLX Swift API, not Python's eval builtin.
        MLX.eval(dKeys, dValues)

        self.compressedKeys = encodedKeys
        self.compressedValues = encodedValues
        self.decodedKeyBuffer = dKeys
        self.decodedValueBuffer = dValues
        self.prefixTokenCount = dKeys.dim(2)

        // Pre-allocate unified buffer: [decoded_prefix | windowStep slots]
        let B = dKeys.dim(0), H = dKeys.dim(1)
        let kD = dKeys.dim(3), vD = dValues.dim(3)
        let windowK = MLXArray.zeros([B, H, windowStep, kD], dtype: dKeys.dtype)
        let windowV = MLXArray.zeros([B, H, windowStep, vD], dtype: dValues.dtype)
        self.unifiedKeys = concatenated([dKeys, windowK], axis: 2)
        self.unifiedValues = concatenated([dValues, windowV], axis: 2)
        self.windowOffset = 0

        self.phase = .compressed
        self.offset = sourceOffset

        // Drop any fill-phase storage that may still be hanging around.
        self.floatKeys = nil
        self.floatValues = nil
    }

    // MARK: - Create from KVCacheSimple

    /// Convert a KVCacheSimple to TurboQuantKVCache by compressing its contents.
    ///
    /// This is the primary entry point, called by `maybeQuantizeKVCache`.
    /// After this call, the source KVCacheSimple can be discarded.
    public static func fromSimpleCache(
        _ source: KVCacheSimple,
        keyBits: Int,
        valueBits: Int,
        sinkTokens: Int = 4
    ) -> TurboQuantKVCache {
        let tqCache = TurboQuantKVCache(
            keyBits: keyBits, valueBits: valueBits, sinkTokens: sinkTokens)

        // Get the current float KV from the source cache
        let state = source.state
        guard state.count == 2 else { return tqCache }

        let keys = state[0]
        let values = state[1]

        guard keys.ndim == 4, values.ndim == 4, source.offset > 0 else {
            return tqCache
        }

        // Compress
        tqCache.compressFloatKV(keys: keys, values: values, sourceOffset: source.offset)
        return tqCache
    }

    // MARK: - Compression

    /// Compress float KV data into TurboQuant format and set up unified buffer.
    private func compressFloatKV(keys: MLXArray, values: MLXArray, sourceOffset: Int) {
        let dim = keys.dim(keys.ndim - 1)

        let state = TQEncoder.EncoderState(
            dim: dim, keyBits: keyBits, valueBits: valueBits)
        self.encoderState = state

        // Encode
        let encodedKeys = TQEncoder.encodeKeys(keys, state: state, sinkTokens: sinkTokens)
        let encodedValues = TQEncoder.encodeValues(values, state: state, sinkTokens: sinkTokens)

        // Evaluate immediately so MLX doesn't graph 30+ layers at once and freeze.
        // MLX.eval() is MLX's lazy tensor materialization — NOT code evaluation.
        MLX.eval(
            encodedKeys.indicesPacked, encodedKeys.qjlPacked,
            encodedKeys.residualNorms, encodedKeys.vectorNorms)
        MLX.eval(encodedValues.indicesPacked, encodedValues.vectorNorms)

        // Decode once into persistent float buffer
        let dKeys = TQEncoder.decodeKeys(encodedKeys, state: state)
        let dValues = TQEncoder.decodeValues(encodedValues, state: state)

        // Store compressed data
        self.compressedKeys = encodedKeys
        self.compressedValues = encodedValues
        self.decodedKeyBuffer = dKeys
        self.decodedValueBuffer = dValues
        self.prefixTokenCount = dKeys.dim(2)

        // Pre-allocate unified buffer: [decoded_prefix | windowStep slots]
        let B = dKeys.dim(0), H = dKeys.dim(1)
        let kD = dKeys.dim(3), vD = dValues.dim(3)
        let windowK = MLXArray.zeros([B, H, windowStep, kD], dtype: dKeys.dtype)
        let windowV = MLXArray.zeros([B, H, windowStep, vD], dtype: dValues.dtype)
        self.unifiedKeys = concatenated([dKeys, windowK], axis: 2)
        self.unifiedValues = concatenated([dValues, windowV], axis: 2)
        self.windowOffset = 0

        self.phase = .compressed
        self.offset = sourceOffset

        // Free fill-phase storage
        self.floatKeys = nil
        self.floatValues = nil
    }

    // MARK: - KVCache Protocol

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        switch phase {
        case .fill:
            return appendFloat(keys: keys, values: values)
        case .compressed:
            return appendDecodeTokens(keys: keys, values: values)
        }
    }

    /// Fill-phase append (identical to KVCacheSimple behavior).
    private func appendFloat(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if var existingKeys = floatKeys, var existingValues = floatValues {
            // After trim(), offset may be less than buffer — slice to valid region
            if offset < existingKeys.dim(2) {
                existingKeys = existingKeys[.ellipsis, ..<offset, 0...]
                existingValues = existingValues[.ellipsis, ..<offset, 0...]
            }
            floatKeys = concatenated([existingKeys, keys], axis: 2)
            floatValues = concatenated([existingValues, values], axis: 2)
        } else {
            floatKeys = keys
            floatValues = values
        }
        offset += keys.dim(2)
        return (floatKeys!, floatValues!)
    }

    /// Compressed-phase append: scatter-write new tokens into unified buffer window.
    private func appendDecodeTokens(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let newTokens = keys.dim(2)
        offset += newTokens

        let writePos = prefixTokenCount + windowOffset

        // Check if unified buffer needs to grow
        let needsRealloc: Bool
        if let existing = unifiedKeys {
            needsRealloc = (writePos + newTokens) > existing.dim(2)
        } else {
            needsRealloc = true
        }

        if needsRealloc {
            let B = keys.dim(0), H = keys.dim(1)
            let kD = keys.dim(3), vD = values.dim(3)
            let nSteps = max(1, (windowStep + newTokens - 1) / windowStep)
            let newK = MLXArray.zeros([B, H, nSteps * windowStep, kD], dtype: keys.dtype)
            let newV = MLXArray.zeros([B, H, nSteps * windowStep, vD], dtype: values.dtype)

            if let existingKeys = unifiedKeys, let existingValues = unifiedValues, writePos > 0 {
                unifiedKeys = concatenated(
                    [existingKeys[.ellipsis, ..<writePos, 0...], newK], axis: 2)
                unifiedValues = concatenated(
                    [existingValues[.ellipsis, ..<writePos, 0...], newV], axis: 2)
            } else {
                unifiedKeys = newK
                unifiedValues = newV
            }
        }

        // O(1) scatter write — no new array allocation
        unifiedKeys?[.ellipsis, writePos..<(writePos + newTokens), 0...] = keys
        unifiedValues?[.ellipsis, writePos..<(writePos + newTokens), 0...] = values
        windowOffset += newTokens

        // Return slice of unified buffer up to current position
        let totalTokens = prefixTokenCount + windowOffset
        return (
            unifiedKeys![.ellipsis, ..<totalTokens, 0...],
            unifiedValues![.ellipsis, ..<totalTokens, 0...]
        )
    }

    // MARK: - State

    public override var state: [MLXArray] {
        get {
            switch phase {
            case .fill:
                guard let keys = floatKeys, let values = floatValues else { return [] }
                // Slice to offset — after trim(), offset may be less than buffer size
                if offset < keys.dim(2) {
                    return [
                        keys[.ellipsis, ..<offset, 0...],
                        values[.ellipsis, ..<offset, 0...],
                    ]
                }
                return [keys, values]
            case .compressed:
                let totalTokens = prefixTokenCount + windowOffset
                guard let uk = unifiedKeys, let uv = unifiedValues, totalTokens > 0 else {
                    return []
                }
                return [
                    uk[.ellipsis, ..<totalTokens, 0...],
                    uv[.ellipsis, ..<totalTokens, 0...],
                ]
            }
        }
        set {
            if newValue.count >= 2 {
                resetToEmpty()
                floatKeys = newValue[0]
                floatValues = newValue[1]
                offset = newValue[0].dim(2)
                phase = .fill
            } else {
                resetToEmpty()
            }
        }
    }

    public override var metaState: [String] {
        get { [""] }
        set { }
    }

    public override func innerState() -> [MLXArray] {
        var arrays: [MLXArray] = []
        if let floatKeys { arrays.append(floatKeys) }
        if let floatValues { arrays.append(floatValues) }
        if let ek = compressedKeys {
            arrays.append(ek.indicesPacked)
            arrays.append(ek.qjlPacked)
            arrays.append(ek.residualNorms)
            arrays.append(ek.vectorNorms)
            if let sink = ek.sinkData { arrays.append(sink) }
        }
        if let ev = compressedValues {
            arrays.append(ev.indicesPacked)
            arrays.append(ev.vectorNorms)
            if let sink = ev.sinkData { arrays.append(sink) }
        }
        if let uk = unifiedKeys { arrays.append(uk) }
        if let uv = unifiedValues { arrays.append(uv) }
        return arrays
    }

    // MARK: - Trim

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        guard trimmed > 0 else { return 0 }

        let targetOffset = offset - trimmed
        guard targetOffset > 0 else {
            resetToEmpty()
            return trimmed
        }

        switch phase {
        case .fill:
            // Simple: just reduce offset, data stays in buffer
            offset = targetOffset
            return trimmed

        case .compressed:
            let totalUsedTokens = prefixTokenCount + windowOffset

            if targetOffset >= prefixTokenCount && targetOffset <= totalUsedTokens {
                // Fast path: trim within window only — O(1)
                // When targetOffset == prefixTokenCount, window becomes empty (windowOffset=0)
                windowOffset = targetOffset - prefixTokenCount
                offset = targetOffset
                return trimmed
            }

            // Slow path: trim reaches into compressed prefix.
            // Decode full KV, truncate, re-compress.
            let totalTokens = prefixTokenCount + windowOffset
            guard let uk = unifiedKeys, let uv = unifiedValues, totalTokens > 0 else {
                resetToEmpty()
                return trimmed
            }

            let fullKeys = uk[.ellipsis, ..<totalTokens, 0...]
            let fullValues = uv[.ellipsis, ..<totalTokens, 0...]
            let trimmedKeys = fullKeys[.ellipsis, ..<targetOffset, 0...]
            let trimmedValues = fullValues[.ellipsis, ..<targetOffset, 0...]

            // Reset and re-compress
            resetToEmpty()
            compressFloatKV(
                keys: trimmedKeys, values: trimmedValues, sourceOffset: targetOffset)
            return trimmed
        }
    }

    // MARK: - Copy

    public override func copy() -> any KVCache {
        let new = TurboQuantKVCache(
            keyBits: keyBits, valueBits: valueBits, sinkTokens: sinkTokens)
        new.phase = phase
        new.offset = offset
        new.prefixTokenCount = prefixTokenCount
        new.windowOffset = windowOffset
        new.encoderState = encoderState

        // Compressed data: struct copy (MLXArray is reference-counted internally).
        // EncodedKeys/EncodedValues are read-only after creation, so sharing is safe.
        new.compressedKeys = compressedKeys
        new.compressedValues = compressedValues

        // Float buffers: [.ellipsis] creates a new graph node sharing data.
        // MLX's lazy evaluation ensures subsequent scatter-writes to the original
        // create new graph nodes without mutating the copy's data.
        // This matches KVCacheSimple.copy() behavior.
        new.floatKeys = floatKeys.map { $0[.ellipsis] }
        new.floatValues = floatValues.map { $0[.ellipsis] }
        new.decodedKeyBuffer = decodedKeyBuffer.map { $0[.ellipsis] }
        new.decodedValueBuffer = decodedValueBuffer.map { $0[.ellipsis] }
        new.unifiedKeys = unifiedKeys.map { $0[.ellipsis] }
        new.unifiedValues = unifiedValues.map { $0[.ellipsis] }

        return new
    }

    // MARK: - Helpers

    private func resetToEmpty() {
        phase = .fill
        floatKeys = nil
        floatValues = nil
        compressedKeys = nil
        compressedValues = nil
        decodedKeyBuffer = nil
        decodedValueBuffer = nil
        unifiedKeys = nil
        unifiedValues = nil
        windowOffset = 0
        prefixTokenCount = 0
        offset = 0
        encoderState = nil
    }
}
