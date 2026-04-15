// Copyright © 2025 Osaurus & JANG. All rights reserved.
// TurboQuant — arXiv:2504.19874 (Google DeepMind)

import Foundation
import MLX

/// Randomized Hadamard rotation for TurboQuant.
///
/// ## Why Rotation Is Necessary
///
/// Without rotation, individual dimensions of K/V vectors have wildly different
/// scales (some dimensions carry most of the information, others are near-zero).
/// A single scalar codebook can't handle this variance — it wastes bins on
/// low-energy dimensions and under-resolves high-energy ones.
///
/// The Hadamard transform H is an orthogonal matrix that spreads energy uniformly
/// across all dimensions. After `y = H * D * x` (where D is a random sign diagonal):
///
/// 1. Each component of y has approximately the same variance
/// 2. The marginal distribution of each component follows a Beta distribution
///    (concentrated near zero with support ~ 3/sqrt(d))
/// 3. A single Lloyd-Max codebook is now optimal for ALL components simultaneously
///
/// ## Non-Power-of-2 Support
///
/// The Hadamard butterfly requires power-of-2 dimensions. For arbitrary head_dim
/// (e.g., 96 for Gemma4), we decompose into descending powers of 2:
///   96 → [64, 32], apply butterfly to each block, concatenate.
///
/// ## Inverse
///
/// Since H is symmetric (H^T = H) and orthogonal (H*H = I after normalization),
/// and D is self-inverse (D*D = I), the inverse is: `x = D * H * y`.
public struct TQHadamard: Sendable {

    // MARK: - Dimension Decomposition

    /// Decompose a dimension into descending powers of 2.
    ///
    /// Example: 96 → [64, 32], 128 → [128], 160 → [128, 32]
    public static func decomposePow2Blocks(_ dim: Int) -> [Int] {
        var blocks = [Int]()
        var remaining = dim
        while remaining > 0 {
            let bitLen = Int.bitWidth - remaining.leadingZeroBitCount
            let p = 1 << (bitLen - 1)
            blocks.append(p)
            remaining -= p
        }
        return blocks
    }

    // MARK: - Random Sign Vector

    /// Serial queue protecting srand48/drand48 (which share global state).
    /// In practice, sign generation only runs during cache compression inside
    /// ModelContainer's actor-serialized context, but the lock ensures safety
    /// if this is ever called from multiple threads.
    private static let rngLock = NSLock()

    /// Generate a deterministic random sign vector (+1/-1) of the given dimension.
    ///
    /// Uses srand48/drand48 for CPU-side determinism independent of MLX's GPU
    /// random state. Same seed always produces same signs (required for inverse).
    public static func generateRandomSigns(dim: Int, seed: Int = 0) -> MLXArray {
        rngLock.lock()
        defer { rngLock.unlock() }
        srand48(seed)
        var signs = [Float](repeating: 0, count: dim)
        for i in 0..<dim {
            signs[i] = drand48() < 0.5 ? -1.0 : 1.0
        }
        return MLXArray(signs)
    }

    // MARK: - Hadamard Butterfly

    /// Apply the normalized Hadamard butterfly transform to the last dimension.
    ///
    /// The butterfly factorization computes H*x in O(d log d) instead of O(d^2):
    /// at each stage h = 1, 2, 4, ..., d/2, pairs (a, b) become (a+b, a-b).
    /// Result is scaled by 1/sqrt(d) for orthogonality.
    ///
    /// The last dimension must be a power of 2.
    static func hadamardTransform(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let d = shape[shape.count - 1]
        let n = shape.dropLast().reduce(1, *)
        var result = x.reshaped([n, d])

        var h = 1
        while h < d {
            let groups = d / (2 * h)
            result = result.reshaped([n, groups, 2, h])

            let a = result[0..., 0..., 0..<1, 0...].reshaped([n, groups, h])
            let b = result[0..., 0..., 1..<2, 0...].reshaped([n, groups, h])

            result = concatenated([a + b, a - b], axis: -1).reshaped([n, d])
            h *= 2
        }

        let scale = MLXArray(Float(1.0 / sqrt(Float(d))))
        return (result * scale).reshaped(shape)
    }

    // MARK: - Forward Rotation

    /// Apply randomized Hadamard rotation: y = H(D * x) where D = diag(signs).
    ///
    /// For non-power-of-2 dimensions, splits into blocks and transforms each independently.
    public static func hadamardRotate(_ x: MLXArray, signs: MLXArray) -> MLXArray {
        let dim = x.dim(x.shape.count - 1)
        let blocks = decomposePow2Blocks(dim)

        if blocks.count == 1 {
            return hadamardTransform(x * signs)
        }

        var parts = [MLXArray]()
        var offset = 0
        for bs in blocks {
            let xSlice = x[.ellipsis, offset..<(offset + bs)]
            let sSlice = signs[offset..<(offset + bs)]
            parts.append(hadamardTransform(xSlice * sSlice))
            offset += bs
        }
        return concatenated(parts, axis: -1)
    }

    // MARK: - Inverse Rotation

    /// Apply inverse randomized Hadamard rotation: x = D * H(y).
    ///
    /// Since H is symmetric/orthogonal and D is self-inverse, the inverse
    /// is simply: apply H first, then multiply by the same signs.
    public static func hadamardInverse(_ y: MLXArray, signs: MLXArray) -> MLXArray {
        let dim = y.dim(y.shape.count - 1)
        let blocks = decomposePow2Blocks(dim)

        if blocks.count == 1 {
            return hadamardTransform(y) * signs
        }

        var parts = [MLXArray]()
        var offset = 0
        for bs in blocks {
            let ySlice = y[.ellipsis, offset..<(offset + bs)]
            let sSlice = signs[offset..<(offset + bs)]
            parts.append(hadamardTransform(ySlice) * sSlice)
            offset += bs
        }
        return concatenated(parts, axis: -1)
    }
}
