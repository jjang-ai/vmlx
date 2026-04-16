// SPDX-License-Identifier: Apache-2.0
//
// FlashMoESwitchGLUShim — drop-in replacement for Gemma 4's `SwitchGLU`.
//
// Gemma 4's decoder layer has `router` + `experts` as direct siblings,
// where `experts` wraps a `SwitchGLU` that's called with
// `(x, indices) -> [..., top_k, hidden]`. The sibling `Router` then
// multiplies by top-k weights and sums across the K axis.
//
// This shim preserves the exact signature so it can be hot-swapped into
// `layer.experts.switch_glu` via `updateModule(key:_:)` without the
// surrounding Router needing any changes. Port of
// `vmlx_engine/models/flash_moe_integration.py:FlashMoESwitchGLUShim`.

import Foundation
import MLX
import MLXNN

/// Streaming replacement for Gemma 4 `SwitchGLU`. Loads experts on
/// demand from SSD, caches hot experts in the slot bank.
public final class FlashMoESwitchGLUShim: Module {

    // MARK: - Config

    private let loader: FlashMoEExpertLoader
    private let layerIdx: Int

    /// Inferred from the original module's projections at swap time.
    /// Falls back to 128 if the original was fp16/bf16 (no group_size).
    private let nativeGroupSize: Int
    /// Inferred from the original module's quantized projection; nil for fp paths.
    private let nativeBits: Int?
    /// Whether to use a SwitchGLU 2-arg activation form for the per-expert
    /// matmul. Gemma 4 uses GeGLU which is captured via the closure.
    private let activation: FlashMoEActivation

    // MARK: - Init

    public init(
        loader: FlashMoEExpertLoader,
        layerIdx: Int,
        nativeGroupSize: Int = 128,
        nativeBits: Int? = nil,
        activation: FlashMoEActivation = .default
    ) {
        self.loader = loader
        self.layerIdx = layerIdx
        self.nativeGroupSize = nativeGroupSize
        self.nativeBits = nativeBits
        self.activation = activation
        super.init()
    }

    // MARK: - SwitchGLU contract

    /// `SwitchGLU.__call__` equivalent.
    ///
    /// - Parameters:
    ///   - x: `[..., H]` token hidden states.
    ///   - indices: `[..., K]` top-K expert indices per token.
    /// - Returns: `[..., K, H]` per-slot expert outputs. The caller (Router)
    ///   multiplies by top-k weights and sums across the K axis.
    public func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let origShape = x.shape
        let H = origShape.last ?? 0
        let K = indices.shape.last ?? 0

        // Flatten leading dims: x_flat = [T, H], idx_flat = [T, K]
        let xFlat = x.reshaped([-1, H])
        let idxFlat = indices.reshaped([-1, K])
        let T = xFlat.dim(0)

        // Audit 2026-04-16 Gemma 4 perf fix: hoist the GPU→CPU sync ONCE.
        // Previously this called `idxFlat[t, k].item()` inside a nested
        // T×K loop, forcing 8 GPU flushes per decode token (T=1, K=8) just
        // to read routing indices — each an eval barrier in the MLX graph.
        // `asArray(Int32.self)` collapses all T×K values into a single
        // synchronization so expert loading can start immediately.
        let flatArray = idxFlat.asArray(Int32.self)

        // Gather unique expert IDs across the flat batch, load in parallel.
        var unique = Set<Int>()
        var idxListPerToken: [[Int]] = []
        idxListPerToken.reserveCapacity(T)
        for t in 0..<T {
            var row: [Int] = []
            row.reserveCapacity(K)
            for k in 0..<K {
                let e = Int(flatArray[t * K + k])
                row.append(e)
                unique.insert(e)
            }
            idxListPerToken.append(row)
        }
        let experts = loader.loadExpertsParallel(
            layerIdx: layerIdx,
            expertIndices: Array(unique)
        )

        // For each top-K slot, group tokens by expert id and run the
        // per-expert matmul once per group. Then scatter back and stack
        // the K slot outputs along a new axis.
        var slotOutputs: [MLXArray] = []
        slotOutputs.reserveCapacity(K)
        for k in 0..<K {
            var expertToTokens: [Int: [Int]] = [:]
            for t in 0..<T {
                let e = idxListPerToken[t][k]
                expertToTokens[e, default: []].append(t)
            }

            var slotOut = MLXArray.zeros(like: xFlat)
            for (eidx, tokenIdxs) in expertToTokens {
                guard let ews = experts[eidx] else { continue }
                let idxArr = MLXArray(tokenIdxs.map { Int32($0) })
                let xSub = xFlat[idxArr]
                let ySub = flashMoEApplyExpertTensors(
                    xSub,
                    expert: ews,
                    defaultGroupSize: nativeGroupSize,
                    nativeBits: nativeBits,
                    activation: activation,
                    isSwitchGLU: true
                )
                // Scatter-add ySub back into slotOut at the token indices.
                slotOut = slotOut.scatterAxis0Add(indices: idxArr, updates: ySub)
            }
            slotOutputs.append(slotOut)
        }

        // stack across K → [T, K, H], then unflatten leading dims
        let stacked = MLX.stacked(slotOutputs, axis: 1)
        var outShape = origShape
        // Insert K before the final axis.
        outShape.insert(K, at: outShape.count - 1)
        return stacked.reshaped(outShape)
    }
}

// MARK: - Scatter helper

extension MLXArray {
    /// Adds `updates` into `self` at `indices` along axis 0.
    /// Equivalent to Python's `self.at[indices].add(updates)` /
    /// `mx.array.at[...].add(...)`.
    fileprivate func scatterAxis0Add(indices: MLXArray, updates: MLXArray) -> MLXArray {
        // Gather the current slice at `indices`, add updates, write back.
        // Correct for our use case because within a single top-K slot
        // every token index appears in exactly one (expert, token-group)
        // pair — there are no concurrent writes to the same row.
        let current = self[indices]
        let newValues = current + updates
        var copy = self
        copy[indices] = newValues
        return copy
    }
}
