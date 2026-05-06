// SPDX-License-Identifier: Apache-2.0
//
// CompilableTurboQuantKVCache — compile-safe TurboQuant decode cache.
//
// Ported from the proven vmlx-swift-lm Stage 2 implementation. Plain
// TurboQuantKVCache is correct but its compressed-phase append path keeps the
// write offset as a Swift Int and returns a dynamic slice. That cannot be
// safely captured by MLX compile across many decode steps: the trace can reuse
// stale offsets or attend into uninitialized tail slots. This subclass keeps
// the mutating counters in MLXArray graph state, writes with
// dynamicSliceUpdate, returns the full static unified buffer, and supplies the
// matching attention mask.

import Foundation
import MLX
import MLXFast

public final class CompilableTurboQuantKVCache: TurboQuantKVCache, @unchecked Sendable {

    /// Position of the next window write, as `MLXArray[1] int32`.
    public var writePosArray: MLXArray

    /// Total valid tokens = prefixTokenCount + written decode-window tokens.
    public var offsetArray: MLXArray

    private lazy var maskRinds: MLXArray = {
        let bufferLen = unifiedKeys?.dim(2) ?? 0
        return MLXArray(Int32(0) ..< Int32(bufferLen))
    }()

    public override init(keyBits: Int = 3, valueBits: Int = 3, sinkTokens: Int = 4) {
        self.writePosArray = MLXArray([Int32(0)])
        self.offsetArray = MLXArray([Int32(0)])
        super.init(keyBits: keyBits, valueBits: valueBits, sinkTokens: sinkTokens)
    }

    /// Promote an already-compressed TurboQuant cache after prefill and before
    /// compiled single-token decode. The source must be in compressed phase;
    /// fill phase is not on the compiled hot path.
    public convenience init(from tq: TurboQuantKVCache) {
        precondition(
            tq.phase == .compressed,
            "CompilableTurboQuantKVCache(from:) requires compressed source")

        self.init(keyBits: tq.keyBits, valueBits: tq.valueBits, sinkTokens: tq.sinkTokens)

        self.floatKeys = tq.floatKeys
        self.floatValues = tq.floatValues
        self.decodedKeyBuffer = tq.decodedKeyBuffer
        self.decodedValueBuffer = tq.decodedValueBuffer
        self.unifiedKeys = tq.unifiedKeys
        self.unifiedValues = tq.unifiedValues
        self.prefixTokenCount = tq.prefixTokenCount
        self.windowOffset = tq.windowOffset
        self.encoderState = tq.encoderState
        self.valueEncoderState = tq.valueEncoderState

        // Preserve compressed-phase invariants using the parent restore path
        // because `phase` is public-private(set). This may rebuild the unified
        // buffer once, outside the compiled trace, which is acceptable at the
        // prefill -> decode boundary.
        if let ck = tq.compressedKeys, let cv = tq.compressedValues {
            self.restoreCompressed(
                encodedKeys: ck, encodedValues: cv, sourceOffset: tq.offset)
        }

        if tq.windowOffset > 0,
           let sourceKeys = tq.unifiedKeys,
           let sourceValues = tq.unifiedValues,
           let destKeys = self.unifiedKeys,
           let destValues = self.unifiedValues
        {
            let start = tq.prefixTokenCount
            let end = start + tq.windowOffset
            let windowKeys = sourceKeys[.ellipsis, start ..< end, 0...]
            let windowValues = sourceValues[.ellipsis, start ..< end, 0...]
            destKeys[.ellipsis, start ..< end, 0...] = windowKeys
            destValues[.ellipsis, start ..< end, 0...] = windowValues
            self.windowOffset = tq.windowOffset
            self.offset = tq.offset
        }

        self.writePosArray = MLXArray([Int32(self.windowOffset)])
        self.offsetArray = MLXArray([Int32(self.offset)])
    }

    public override func update(
        keys: MLXArray, values: MLXArray
    ) -> (MLXArray, MLXArray) {
        switch phase {
        case .fill:
            return super.update(keys: keys, values: values)
        case .compressed:
            return compiledAppendDecode(keys: keys, values: values)
        }
    }

    private func compiledAppendDecode(
        keys: MLXArray, values: MLXArray
    ) -> (MLXArray, MLXArray) {
        let newTokens = keys.dim(2)
        let prev = offsetArray
        let advance = MLXArray([Int32(newTokens)])
        let newOffset = prev + advance

        unifiedKeys!._updateInternal(
            dynamicSliceUpdate(
                unifiedKeys!, update: keys,
                start: prev, axes: [2]))
        unifiedValues!._updateInternal(
            dynamicSliceUpdate(
                unifiedValues!, update: values,
                start: prev, axes: [2]))

        writePosArray._updateInternal(writePosArray + advance)
        offsetArray._updateInternal(newOffset)

        // Mirrors are for non-compiled consumers after the trace returns.
        offset += newTokens
        windowOffset += newTokens

        return (unifiedKeys!, unifiedValues!)
    }

    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let linds: MLXArray
        if n == 1 {
            linds = offsetArray.reshaped(1, 1)
        } else {
            linds = (MLXArray(Int32(0) ..< Int32(n)) + offsetArray).reshaped(n, 1)
        }

        let bufferLen = unifiedKeys?.dim(2) ?? 0
        let rinds = maskRinds.reshaped(1, bufferLen)
        var mask = linds .>= rinds

        if let windowSize {
            let windowStart = linds - Int32(windowSize - 1)
            mask = mask & (rinds .>= windowStart)
        }

        return .array(mask)
    }

    public override func innerState() -> [MLXArray] {
        var state = [MLXArray]()
        if let uk = unifiedKeys { state.append(uk) }
        if let uv = unifiedValues { state.append(uv) }
        state.append(writePosArray)
        state.append(offsetArray)
        return state
    }
}
