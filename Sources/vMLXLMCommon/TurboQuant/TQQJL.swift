// Copyright © 2025 Osaurus & JANG. All rights reserved.
// TurboQuant — arXiv:2504.19874 (Google DeepMind)

import Foundation
import MLX
import MLXRandom

/// Quantized Johnson-Lindenstrauss (QJL) random projection for key residual correction.
///
/// ## Why Keys Need QJL But Values Don't
///
/// Attention scores are computed as `softmax(Q * K^T / sqrt(d))`. The softmax
/// exponentiates the inner products, so a small additive error epsilon in
/// `<q, k>` becomes a multiplicative error `exp(epsilon)` in the attention weight.
/// At d=128, even 0.1 MSE in the key quantization can shift attention weights
/// by ~10%, causing visible quality degradation.
///
/// Values, by contrast, are just linearly combined: `output = softmax(...) * V`.
/// Error in V scales linearly with the attention weights, so MSE-optimal
/// codebook quantization (without QJL) is sufficient.
///
/// ## How QJL Works
///
/// After MSE quantization of the rotated key with (b-1) bits, the residual
/// `r = x_rotated - Q(x_rotated)` still contains information. QJL compresses
/// this residual to 1 bit per dimension:
///
/// 1. Project residual through random Gaussian matrix S: `p = r @ S^T`
/// 2. Store only the signs: `s = sign(p)` — 1 bit each
/// 3. Store the residual norm: `||r||`
///
/// To decode: `r_hat = sqrt(pi/2) / d * ||r|| * (s @ S)`
///
/// The Johnson-Lindenstrauss lemma guarantees that this preserves inner products
/// (and therefore attention scores) in expectation. The 1/d scaling and sqrt(pi/2)
/// factor come from the expected magnitude of sign-quantized Gaussian projections.
///
/// ## Memory Cost
///
/// For b=3 bit keys: (b-1)=2 bits for MSE indices + 1 bit for QJL signs = 3 bits total.
/// Plus per-vector norms (residual norm + vector norm) at float16 = 4 bytes per vector.
/// Net: ~3.25 bits/element vs 16 bits/element for float16 = ~4.9x compression.
public struct TQQJL: Sendable {

    // MARK: - Projection Matrix Generation

    /// Generate a random Gaussian projection matrix for QJL.
    ///
    /// Entries are i.i.d. N(0,1), generated deterministically from seed.
    /// Same seed always produces the same matrix, so encode/decode match
    /// without storing the matrix.
    ///
    /// - Parameters:
    ///   - dim: Vector dimension. Produces a dim x dim square matrix.
    ///   - seed: Random seed for deterministic generation.
    /// - Returns: MLXArray of shape [dim, dim] with float32 Gaussian entries.
    public static func generateProjection(dim: Int, seed: Int = 0) -> MLXArray {
        let rngKey = MLXRandom.key(UInt64(seed))
        return MLXRandom.normal([dim, dim], key: rngKey)
    }
}
