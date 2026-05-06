// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Stage 3 — `CompilableRotatingKVCache`.
//
// The Stage 3 probe in `RotatingKVCacheCompileProbeTests.swift` established
// that the existing `RotatingKVCache` is compile-traceable in the linear
// (pre-wrap, pre-growth) segment BUT breaks in two ways over longer
// decodes:
//   1. Buffer growth via `self.keys = concatenated(...)` rebinds the
//      `keys` property. Compile traces capture the original object;
//      rebinding loses it. Observed drift: ~30% at growth crossing.
//   2. Ring-buffer wrap-around via `idx = keep` (Swift Int reset) and
//      subsequent writes at the rotated position don't match the trace's
//      assumed linear layout. Observed drift: ~68% post-wrap.
//
// This subclass addresses both by:
//   - Pre-allocating the unified buffer to `maxCacheSize` at promotion
//     time. No concat-growth during decode.
//   - Tracking the write index as `idxArray: MLXArray[1] int32`. Wrap
//     arithmetic uses MLXArray ops (`where_` + modulo on MLXArray)
//     rather than Swift Int comparisons.
//   - Returning the FULL `[B, H, maxCacheSize, D]` buffer from `update()`
//     with an `.array` mask generated in `makeMask` that respects both
//     causal validity and the ring layout.
//
// Initialise via ``init(from:)`` on an existing `RotatingKVCache` that
// has been populated by prefill.
//
// reference implementation contract: this cache must remain a same-module
// subclass `RotatingKVCache` with `idxArray` and `offsetArray` graph counters
// and MLXArray `where` + modulo wrap arithmetic.

import Foundation
import MLX
import MLXNN

/// Compile-traceable specialisation of ``RotatingKVCache`` for the
/// batched compile path (Stage 3).
///
/// Compared to the parent:
///
/// - `keys` and `values` are eagerly allocated at `[B, H, maxCacheSize, D]`
///   at promotion time. No growth-via-concat during decode. Iter-7 probe
///   measured ~30% drift when growth fired inside a compile trace.
/// - `idxArray: MLXArray[1] int32` replaces Swift-Int `idx`. All wrap
///   arithmetic happens in MLXArray ops so the compile tracer can follow.
/// - `offsetArray: MLXArray[1] int32` mirrors `offset`. Used by `makeMask`
///   for the causal upper bound; tracked as an MLXArray so the tracer
///   follows per-step advances.
/// - `update(keys:values:)` writes new tokens at `idxArray` via
///   `dynamicSliceUpdate`, then advances `idxArray` with wrap semantics
///   entirely in MLXArray space.
/// - `makeMask` always returns `.array(mask)` — the full-buffer return
///   means attention must be told which positions are valid. In the pre-
///   wrap linear region, this is standard causal. Post-wrap, the mask
///   admits all `maxCacheSize` positions because the ring is full.
///
/// ## Scope
///
/// Stage 3 ships the subclass. Wiring it into `BatchEngine` (so
/// sliding-window models like Gemma3/Gemma4 SWA automatically get the
/// compile path) is a follow-up (flip of `CacheFamily.rotating.isCompileEligibleAtCurrentStage`).
public final class CompilableRotatingKVCache: RotatingKVCache, @unchecked Sendable {

    /// Current write index within the ring buffer, as `MLXArray[1] int32`.
    /// In the linear segment (before wrap), this equals `offsetArray`.
    /// After wrap, this rotates through `[keep, maxCacheSize)`.
    public var idxArray: MLXArray

    /// Total valid tokens seen, as `MLXArray[1] int32`. In the linear
    /// segment, equals `idxArray` and is a tight upper bound on valid
    /// positions. Post-wrap, `offsetArray >= maxCacheSize` and ALL
    /// `maxCacheSize` positions in the buffer are valid (ring full).
    public var offsetArray: MLXArray

    /// Pre-computed column indices `[0, 1, ..., maxCacheSize-1]` used by
    /// `makeMask` to build a causal mask over the full buffer.
    private lazy var maskRinds: MLXArray = MLXArray(Int32(0) ..< Int32(maxCacheSize))

    // MARK: - Init

    /// Direct constructor matching the parent. Primarily for testing.
    public override init(maxSize: Int, keep: Int = 0, step: Int = 256) {
        self.idxArray = MLXArray([Int32(0)])
        self.offsetArray = MLXArray([Int32(0)])
        super.init(maxSize: maxSize, keep: keep, step: step)
    }

    /// Promote an existing populated ``RotatingKVCache`` to a compile-
    /// traceable variant. Copies the state references AND allocates the
    /// unified buffer at full `maxCacheSize` size if the parent's buffer
    /// is smaller (the parent grows lazily in `step`-sized chunks).
    ///
    /// - Parameter rotating: Source cache. Typically the result of a
    ///   prefill that has populated keys/values.
    public convenience init(from rotating: RotatingKVCache) {
        self.init(
            maxSize: rotating.maxCacheSize,
            keep: rotating.keep,
            step: rotating.step
        )

        // Copy state references from the source. Same-module subclass
        // access works because parent's state is `internal`.
        self.idx = rotating.idx
        self.offset = rotating.offset

        // Pre-allocate or extend the unified buffer to full maxCacheSize.
        // This prevents the compile-breaking concat-growth path from ever
        // firing during decode.
        if let srcK = rotating.keys, let srcV = rotating.values {
            let B = srcK.dim(0)
            let H = srcK.dim(1)
            let kD = srcK.dim(3)
            let vD = srcV.dim(3)
            let currentLen = srcK.dim(2)

            if currentLen < maxCacheSize {
                // Need to grow — but this is a ONE-TIME growth during
                // promotion, not inside a compile trace. Use concat to
                // extend to full size.
                let padLen = maxCacheSize - currentLen
                let padK = MLXArray.zeros([B, H, padLen, kD], dtype: srcK.dtype)
                let padV = MLXArray.zeros([B, H, padLen, vD], dtype: srcV.dtype)
                self.keys = concatenated([srcK, padK], axis: 2)
                self.values = concatenated([srcV, padV], axis: 2)
            } else if currentLen > maxCacheSize {
                // RotatingKVCache may temporarily exceed maxCacheSize during
                // multi-token prefill. Collapse it to the parent ring shape
                // before compile tracing so decode never writes past the
                // static backing buffer.
                let trimSize = currentLen - maxCacheSize
                if keep > 0 {
                    self.keys = concatenated(
                        [
                            srcK[.ellipsis, ..<keep, 0...],
                            srcK[.ellipsis, (trimSize + keep)..., 0...],
                        ], axis: 2)
                    self.values = concatenated(
                        [
                            srcV[.ellipsis, ..<keep, 0...],
                            srcV[.ellipsis, (trimSize + keep)..., 0...],
                        ], axis: 2)
                } else {
                    self.keys = srcK[.ellipsis, trimSize..., 0...]
                    self.values = srcV[.ellipsis, trimSize..., 0...]
                }
                self.idx = maxCacheSize
            } else {
                self.keys = srcK
                self.values = srcV
            }
        }
        // else: keys/values remain nil; first `update` call allocates
        // them at full size.

        self.idxArray = MLXArray([Int32(self.idx)])
        self.offsetArray = MLXArray([Int32(self.offset)])
    }

    // MARK: - Overridden update

    /// Compile-traceable append. Writes new tokens at `idxArray` position
    /// via `dynamicSliceUpdate`, advances counters with wrap semantics in
    /// MLXArray ops.
    ///
    /// Returns the FULL `[B, H, maxCacheSize, D]` buffer. `makeMask`
    /// restricts attention to valid positions.
    public override func update(
        keys newKeys: MLXArray, values newValues: MLXArray
    ) -> (MLXArray, MLXArray) {
        let nTokens = newKeys.dim(2)

        // Lazy-allocate the unified buffer if empty (first-call init).
        if keys == nil {
            let B = newKeys.dim(0)
            let H = newKeys.dim(1)
            let kD = newKeys.dim(3)
            let vD = newValues.dim(3)
            keys = MLXArray.zeros([B, H, maxCacheSize, kD], dtype: newKeys.dtype)
            values = MLXArray.zeros([B, H, maxCacheSize, vD], dtype: newValues.dtype)
        }

        // Write new tokens at idxArray position. A restored/promoted parent
        // cache can have idx == maxCacheSize, which parent RotatingKVCache
        // rotates back to `keep` immediately before assignment.
        let maxSz = MLXArray([Int32(maxCacheSize)])
        let keepArr = MLXArray([Int32(keep)])
        let writeIdx = MLX.`where`(idxArray .>= maxSz, keepArr, idxArray)
        keys!._updateInternal(
            dynamicSliceUpdate(keys!, update: newKeys, start: writeIdx, axes: [2]))
        values!._updateInternal(
            dynamicSliceUpdate(values!, update: newValues, start: writeIdx, axes: [2]))

        // Advance counters. Wrap arithmetic on idxArray:
        //   newIdx = advance < maxCacheSize ? advance : keep + (advance - keep) % (maxCacheSize - keep)
        // We use `where_` so both branches live in the MLXArray graph.
        let advance = MLXArray([Int32(nTokens)])
        let advancedIdx = writeIdx + advance
        let cycleLen = maxSz - keepArr  // number of rotating slots

        let rotatedIdx: MLXArray
        if keep > 0 {
            rotatedIdx = keepArr + ((advancedIdx - keepArr) % cycleLen)
        } else {
            rotatedIdx = advancedIdx % maxSz
        }
        // where_(cond, true_branch, false_branch)
        let newIdx = MLX.`where`(advancedIdx .< maxSz, advancedIdx, rotatedIdx)

        idxArray._updateInternal(newIdx)
        offsetArray._updateInternal(offsetArray + advance)

        // DELIBERATELY no Swift-Int mirror updates here:
        // `idx = Int(newIdx.item(Int32.self))` would force an `eval`
        // call, which MLX compile rejects ("Attempting to eval an array
        // during function transformations like compile or vmap is not
        // allowed"). Consumers that need the Int view of `idx` / `offset`
        // should read the MLXArray counters and materialize themselves
        // OUTSIDE the compiled trace.

        return (keys!, values!)
    }

    // MARK: - makeMask

    /// Build an attention mask over the full `[B, H, maxCacheSize, D]`
    /// buffer.
    ///
    /// Mask semantics:
    ///  - **Linear phase** (offsetArray < maxCacheSize): valid positions
    ///    are `[0, offsetArray)`. Causal mask is `linds >= rinds`.
    ///  - **Post-wrap phase** (offsetArray >= maxCacheSize): all
    ///    `maxCacheSize` positions are valid (the ring is full). The
    ///    ring layout means positions are NOT in logical order — but for
    ///    single-query decode with `n=1`, every position is attendable.
    ///    The causal constraint is trivially satisfied.
    ///
    /// ## Correctness note
    ///
    /// For n=1 decode in the POST-wrap phase, the RotatingKVCache
    /// uncompiled path returns the full buffer in internal order (not
    /// re-sorted to temporal) and expects attention to treat all
    /// positions equally. That matches "mask all-true for n=1 post-wrap",
    /// which is what `(offsetArray >= maxSize) ? all-true : causal(linds >= rinds)`
    /// produces.
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let linds: MLXArray
        if n == 1 {
            linds = offsetArray.reshaped(1, 1)
        } else {
            linds = (MLXArray(Int32(0) ..< Int32(n)) + offsetArray).reshaped(n, 1)
        }

        let rinds = maskRinds.reshaped(1, maxCacheSize)
        // Causal: attend to positions j <= query_position.
        let causal = linds .>= rinds

        // Post-wrap: if offsetArray >= maxCacheSize, all positions are
        // valid. For n=1 this is just `linds >= maxCacheSize ? all-true : causal`.
        let maxSzArr = MLXArray([Int32(maxCacheSize)]).reshaped(1, 1)
        let allTrueMask = MLX.broadcast(
            MLXArray([true]).reshaped(1, 1),
            to: [linds.dim(0), rinds.dim(1)]
        )
        var mask = MLX.`where`(linds .>= maxSzArr, allTrueMask, causal)

        if let windowSize {
            let windowStart = linds - Int32(windowSize - 1)
            mask = mask & (rinds .>= windowStart)
        }

        return .array(mask)
    }

    // MARK: - innerState

    /// Return ONLY state that mutates during decode: the keys/values
    /// buffers and the two MLXArray counters. See the equivalent comment
    /// on `CompilableTurboQuantKVCache.innerState`.
    public override func innerState() -> [MLXArray] {
        var state = [MLXArray]()
        if let k = keys { state.append(k) }
        if let v = values { state.append(v) }
        state.append(idxArray)
        state.append(offsetArray)
        return state
    }

    public override var metaState: [String] {
        get { super.metaState }
        set {
            super.metaState = newValue
            idxArray = MLXArray([Int32(idx)])
            offsetArray = MLXArray([Int32(offset)])
        }
    }
}
