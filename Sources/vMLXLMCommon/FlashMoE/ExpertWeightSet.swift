// SPDX-License-Identifier: Apache-2.0
//
// ExpertWeightSet — all weight tensors for a single expert in a single
// MoE layer. For 3-projection MoE (Qwen/Mistral/Gemma/MiniMax) each
// projection carries up to 3 tensors (weight/scales/biases) for a
// total of up to 9 tensors per expert. For Nemotron 2-projection MoE
// the gate_proj slot is unused and the total is up to 6 tensors.
//
// This is a pure data bundle — no runtime Metal state, no locks.
// Instances are owned by `SlotBankCache` and passed by reference
// (class, not struct) to avoid copying the underlying `MLXArray`
// storage on cache promotion.

import Foundation
import MLX

/// Named projection slots for an expert. Pure enum so callers can
/// iterate without typing string keys.
public enum ExpertProjection: String, Sendable, CaseIterable {
    case gateProj = "gate_proj"
    case upProj   = "up_proj"
    case downProj = "down_proj"
}

/// Named tensor suffixes within a projection.
public enum ExpertTensorSuffix: String, Sendable, CaseIterable {
    case weight
    case scales
    case biases
}

/// All weight tensors for a single expert in a single MoE layer.
public final class ExpertWeightSet: @unchecked Sendable {
    public let layerIdx: Int
    public let expertIdx: Int
    /// Projection → tensor suffix → MLX array.
    /// Missing slots are simply absent from the inner dict.
    public internal(set) var tensors: [ExpertProjection: [ExpertTensorSuffix: MLXArray]]

    public init(
        layerIdx: Int,
        expertIdx: Int,
        tensors: [ExpertProjection: [ExpertTensorSuffix: MLXArray]] = [:]
    ) {
        self.layerIdx = layerIdx
        self.expertIdx = expertIdx
        self.tensors = tensors
    }

    /// Total bytes consumed by all MLX arrays in this weight set.
    public var totalBytes: Int {
        var total = 0
        for (_, inner) in tensors {
            for (_, arr) in inner {
                total += arr.nbytes
            }
        }
        return total
    }
}
