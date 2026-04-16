// SPDX-License-Identifier: Apache-2.0
//
// FlashMoEBlock — text-path MoE block replacement for Qwen/Mistral/
// MiniMax/Nemotron. Takes over the `mlp`/`block_sparse_moe`/`mixer`
// attribute on a decoder layer. Port of `FlashMoEBlock` in
// `vmlx_engine/models/flash_moe_integration.py`.
//
// Difference from the Gemma 4 shim: the Gemma shim drops *into*
// `experts.switch_glu` and is called by the sibling Router with
// `(x, indices)`. The text-path block is called with just `(x)` —
// it owns the routing too, because Qwen/Mistral/MiniMax/Nemotron
// keep the gate inside the MoE block.
//
// Phase 2a scope: the block's `callAsFunction(_:)` dispatches to
// `expertLoader.loadExpertsParallel` using externally-supplied
// routing (top-K indices + scores). The actual gate/softmax is
// wired in Phase 2b when we hook into model-side Module graphs —
// the gate is model-specific and can't be generic.

import Foundation
import MLX
import MLXNN

/// Flash MoE text-path block. Replaces an entire MoE subtree
/// (gate + switch_mlp/switch_glu) with a slot-bank-backed streamer.
public final class FlashMoEBlock: Module, UnaryLayer {

    // MARK: - Config

    private let loader: FlashMoEExpertLoader
    private let layerIdx: Int
    private let isSwitchGLU: Bool
    private let activation: FlashMoEActivation

    /// Populated by the host layer at swap time with a closure that
    /// runs the original gate against `x` and returns `(indices, scores)`.
    /// This keeps the block model-agnostic while letting each family's
    /// gate run natively (Qwen/Mistral use softmax, MiniMax uses
    /// sigmoid w/ correction bias, Nemotron has pre_routed gates, etc.).
    public var router: (MLXArray) -> (indices: MLXArray, scores: MLXArray) = { _ in
        fatalError("FlashMoEBlock.router must be installed by the host layer at swap time")
    }

    /// Top-K active experts per token. Read from the original module's
    /// `num_experts_per_tok` / `top_k` at swap time.
    public var topK: Int = 8

    /// Native quantization hints read from the original module at swap time.
    public var nativeGroupSize: Int = 128
    public var nativeBits: Int? = nil

    // MARK: - Init

    public init(
        loader: FlashMoEExpertLoader,
        layerIdx: Int,
        isSwitchGLU: Bool,
        activation: FlashMoEActivation
    ) {
        self.loader = loader
        self.layerIdx = layerIdx
        self.isSwitchGLU = isSwitchGLU
        self.activation = activation
        super.init()
    }

    // MARK: - UnaryLayer

    /// `mlp.__call__` contract. Input is `[B, L, H]`; output is same shape.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // 1. Route.
        let (indices, scores) = router(x)

        // 2. Flatten leading dims.
        let origShape = x.shape
        let H = origShape.last ?? 0
        let T = origShape.dropLast().reduce(1, *)
        let K = topK

        let xFlat = x.reshaped([T, H])
        let indsFlat = indices.reshaped([T, K])
        let scoresFlat = scores.reshaped([T, K])

        // 3. Collect unique expert IDs and parallel-load.
        // Audit 2026-04-16 Gemma 4 perf fix: single eval barrier instead
        // of T×K per-index .item() GPU flushes.
        let flatInds = indsFlat.asArray(Int32.self)
        var unique = Set<Int>()
        var indsPerToken: [[Int]] = []
        indsPerToken.reserveCapacity(T)
        for t in 0..<T {
            var row: [Int] = []
            row.reserveCapacity(K)
            for k in 0..<K {
                let e = Int(flatInds[t * K + k])
                row.append(e)
                unique.insert(e)
            }
            indsPerToken.append(row)
        }
        let experts = loader.loadExpertsParallel(
            layerIdx: layerIdx,
            expertIndices: Array(unique)
        )

        // 4. Compute MoE output.
        var result = MLXArray.zeros(like: xFlat)

        if T == 1 {
            // Token-generation fast path: no grouping.
            let row = indsPerToken[0]
            for k in 0..<K {
                guard let ews = experts[row[k]] else { continue }
                let expertOut = flashMoEApplyExpertTensors(
                    xFlat,
                    expert: ews,
                    defaultGroupSize: nativeGroupSize,
                    nativeBits: nativeBits,
                    activation: activation,
                    isSwitchGLU: isSwitchGLU
                )
                let weight = scoresFlat[0, k]
                result = result + expertOut * weight
            }
        } else {
            // Prefill path: group tokens by expert per slot.
            for k in 0..<K {
                var expertToTokens: [Int: [Int]] = [:]
                for t in 0..<T {
                    let e = indsPerToken[t][k]
                    expertToTokens[e, default: []].append(t)
                }
                let slotScores = scoresFlat[0..., k ..< (k + 1)]  // [T, 1]

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
                        isSwitchGLU: isSwitchGLU
                    )
                    // Scatter-add.
                    let current = slotOut[idxArr]
                    var copy = slotOut
                    copy[idxArr] = current + ySub
                    slotOut = copy
                }
                result = result + slotOut * slotScores
            }
        }

        return result.reshaped(origShape)
    }
}
