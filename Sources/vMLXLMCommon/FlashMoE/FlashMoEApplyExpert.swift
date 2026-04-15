// SPDX-License-Identifier: Apache-2.0
//
// Shared per-expert matmul used by FlashMoEBlock and FlashMoESwitchGLUShim.
// Port of `_apply_expert_tensors` from
// `vmlx_engine/models/flash_moe_integration.py`.
//
// Handles JANG mixed-precision (per-projection bit widths are inferred from
// packed weight shape vs. known `in_dim`) and both SwitchGLU (gate+up+down)
// and SwitchMLP (up+down only — Nemotron) structures.

import Foundation
import MLX
import MLXNN

/// Activation kind for the per-expert inner activation.
///
/// SwitchGLU uses a 2-argument form `f(up, gate)` (SwiGLU/GeGLU).
/// SwitchMLP uses a 1-argument form `f(x)` (Nemotron uses ReLU²).
/// If `.default`, we use `silu(gate) * up` for GLU and `silu(x)` for MLP.
///
/// The closure variants are deliberately not `Sendable` — MLX arrays
/// and activation closures aren't Sendable either, and Flash MoE
/// dispatch always happens on the same actor that owns the model.
public enum FlashMoEActivation {
    case `default`
    /// 2-arg: `(up, gate) -> hidden` — used by SwitchGLU.
    case switchGLU((MLXArray, MLXArray) -> MLXArray)
    /// 1-arg: `(x) -> hidden` — used by SwitchMLP.
    case switchMLP((MLXArray) -> MLXArray)
}

/// Apply one expert's three projections to a flat token batch.
///
/// Input `x` is `[T, H]` (flattened leading dims). Output shape is
/// `[T, out]` where `out` is dictated by `down_proj`.
///
/// - Parameters:
///   - x: `[T, H]` input tokens.
///   - expert: The weight set for one expert in one layer.
///   - defaultGroupSize: Fall-back quant group size when no native hint
///     is available from the original module. Typical: 64 or 128.
///   - nativeBits: Hint from the original module's quantized projection,
///     or `nil` when all projections were unquantized.
///   - activation: Activation kind. `.default` uses `silu*` per Python.
///   - isSwitchGLU: Whether the model's original structure was a
///     SwitchGLU (as opposed to SwitchMLP / Nemotron's 2-proj form).
public func flashMoEApplyExpertTensors(
    _ x: MLXArray,
    expert: ExpertWeightSet,
    defaultGroupSize: Int,
    nativeBits: Int?,
    activation: FlashMoEActivation,
    isSwitchGLU: Bool
) -> MLXArray {
    let tensors = expert.tensors
    let hasGate = tensors[.gateProj] != nil
    let hasUp   = tensors[.upProj]   != nil
    let hasDown = tensors[.downProj] != nil

    let hiddenSize = x.dim(-1)

    func matmul(_ xIn: MLXArray, _ projTensors: [ExpertTensorSuffix: MLXArray], inDim: Int) -> MLXArray {
        guard let weight = projTensors[.weight] else {
            return xIn
        }
        if let scales = projTensors[.scales] {
            let biases = projTensors[.biases]
            let bits = inferBitsFromShapes(
                weight: weight,
                scales: scales,
                inDim: inDim,
                defaultGroupSize: defaultGroupSize,
                nativeBits: nativeBits
            )
            return quantizedMM(
                xIn,
                weight,
                scales: scales,
                biases: biases,
                transpose: true,
                groupSize: defaultGroupSize,
                bits: bits
            )
        }
        return xIn.matmul(weight.transposed())
    }

    if hasGate, hasUp, hasDown {
        let gateOut = matmul(x, tensors[.gateProj]!, inDim: hiddenSize)
        let upOut   = matmul(x, tensors[.upProj]!,   inDim: hiddenSize)

        let hidden: MLXArray
        switch activation {
        case .switchGLU(let fn):
            hidden = fn(upOut, gateOut)
        case .switchMLP(let fn):
            // Nominally a 1-arg activation — fall back to the GLU pattern
            // (silu(gate) * up) if the caller wired the wrong kind.
            hidden = fn(gateOut) * upOut
        case .default:
            hidden = MLXNN.silu(gateOut) * upOut
        }

        return matmul(hidden, tensors[.downProj]!, inDim: hidden.dim(-1))
    } else if hasUp, hasDown {
        let upOut = matmul(x, tensors[.upProj]!, inDim: hiddenSize)
        let hidden: MLXArray
        switch activation {
        case .switchMLP(let fn):
            hidden = fn(upOut)
        case .switchGLU, .default:
            hidden = MLXNN.silu(upOut)
        }
        return matmul(hidden, tensors[.downProj]!, inDim: hidden.dim(-1))
    } else {
        // Incomplete expert — identity passthrough. Callers log the layer/idx.
        return x
    }
}

/// Compute bits from packed weight shape given a known input dimension.
///
/// `weight.shape[-1]` is the packed column count. For b-bit quantization
/// that packs 32/b values per 32-bit word, we have
/// `in_dim = packed_cols * 32 / bits`. Invert to recover bits.
///
/// Falls back to a scales-shape search (default_gs × 64 × 128) when the
/// in_dim calculation doesn't yield a plausible bit width, then finally
/// to `nativeBits ?? 4`.
@inlinable
public func inferBitsFromShapes(
    weight: MLXArray,
    scales: MLXArray,
    inDim: Int,
    defaultGroupSize: Int,
    nativeBits: Int?
) -> Int {
    let packedCols = weight.dim(-1)
    if inDim > 0 {
        let bits = (packedCols * 32) / inDim
        if [2, 3, 4, 5, 6, 8].contains(bits) {
            return bits
        }
    }
    let sCols = scales.dim(-1)
    for tryGS in [defaultGroupSize, 64, 128] {
        let inD = sCols * tryGS
        guard inD > 0 else { continue }
        guard (packedCols * 32) % inD == 0 else { continue }
        let bits = (packedCols * 32) / inD
        if [2, 3, 4, 5, 6, 8].contains(bits) {
            return bits
        }
    }
    return nativeBits ?? 4
}
