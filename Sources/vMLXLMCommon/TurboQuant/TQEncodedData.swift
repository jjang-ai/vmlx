// Copyright © 2025 Osaurus & JANG. All rights reserved.
// TurboQuant — arXiv:2504.19874 (Google DeepMind)

import Foundation
import MLX

// MARK: - EncodedKeys

/// TurboQuant-compressed key cache for a single attention layer.
///
/// ## Storage Layout
///
/// For b-bit key compression (e.g., b=3):
///
///     indicesPacked   — (b-1)-bit codebook indices packed into uint32
///                       For b=3: 2-bit indices, 16 per uint32
///     qjlPacked       — 1-bit QJL projection signs packed into uint32
///                       32 signs per uint32
///     residualNorms   — float16 per-vector residual L2 norms
///                       Shape: [batch, heads, compressed_tokens, 1]
///     vectorNorms     — float16 per-vector original L2 norms
///                       Shape: [batch, heads, compressed_tokens, 1]
///     sinkData        — float16 full-precision sink tokens (first N tokens)
///                       Shape: [batch, heads, sinkCount, head_dim]
///
/// ## Compression Ratio
///
/// Per vector (head_dim=128 floats = 256 bytes in float16):
///   - (b-1) bits/element * 128 = 256 bits for indices
///   - 1 bit/element * 128 = 128 bits for QJL signs
///   - 2 * 2 bytes = 4 bytes for norms
///   Total: (256 + 128) / 8 + 4 = 52 bytes → 4.9x compression
///
/// Plus 4 sink tokens at full precision (256 bytes each), amortized over
/// thousands of compressed tokens → negligible overhead.
public struct EncodedKeys: @unchecked Sendable {
    public let indicesPacked: MLXArray
    public let qjlPacked: MLXArray
    public let residualNorms: MLXArray
    public let vectorNorms: MLXArray

    /// Original compressed tensor shape [batch, heads, tokens, head_dim]
    /// (excludes sink tokens — those are in sinkData).
    public let shape: [Int]

    /// Bits per codebook index (= keyBits - 1, since 1 bit goes to QJL).
    public let indexBits: Int

    /// Random seed used during encoding. Required for correct decoding.
    public let seed: Int

    /// Full-precision sink tokens (first N tokens preserved uncompressed).
    public let sinkData: MLXArray?

    public var sinkCount: Int { sinkData?.dim(2) ?? 0 }

    public init(
        indicesPacked: MLXArray,
        qjlPacked: MLXArray,
        residualNorms: MLXArray,
        vectorNorms: MLXArray,
        shape: [Int],
        indexBits: Int,
        seed: Int = 42,
        sinkData: MLXArray? = nil
    ) {
        self.indicesPacked = indicesPacked
        self.qjlPacked = qjlPacked
        self.residualNorms = residualNorms
        self.vectorNorms = vectorNorms
        self.shape = shape
        self.indexBits = indexBits
        self.seed = seed
        self.sinkData = sinkData
    }

    public var estimatedBytes: Int {
        var total = indicesPacked.nbytes + qjlPacked.nbytes
            + residualNorms.nbytes + vectorNorms.nbytes
        if let sink = sinkData { total += sink.nbytes }
        return total
    }

    public var compressionRatio: Float {
        guard shape.count == 4 else { return 1.0 }
        let originalBytes = shape.reduce(1, *) * 2
        guard estimatedBytes > 0 else { return Float.infinity }
        return Float(originalBytes) / Float(estimatedBytes)
    }
}

// MARK: - EncodedValues

/// TurboQuant-compressed value cache for a single attention layer.
///
/// ## Simpler Than Keys
///
/// Values don't need QJL correction because they participate in attention
/// as linear combinations: `output = softmax(scores) * V`. Quantization error
/// in V is attenuated by the attention weights (which sum to 1), producing
/// a weighted average of small errors. No exponential amplification.
///
/// All b bits go to codebook indices (vs keys which split (b-1) + 1-bit QJL).
/// This means values get slightly better MSE than keys at the same bit width.
///
/// ## Storage Layout
///
///     indicesPacked   — b-bit codebook indices packed into uint32
///                       For b=3: 3-bit indices, 10 per uint32
///     vectorNorms     — float16 per-vector L2 norms
///     sinkData        — float16 full-precision sink tokens
public struct EncodedValues: @unchecked Sendable {
    public let indicesPacked: MLXArray
    public let vectorNorms: MLXArray
    public let shape: [Int]
    public let indexBits: Int
    public let seed: Int
    public let sinkData: MLXArray?

    public var sinkCount: Int { sinkData?.dim(2) ?? 0 }

    public init(
        indicesPacked: MLXArray,
        vectorNorms: MLXArray,
        shape: [Int],
        indexBits: Int,
        seed: Int = 42,
        sinkData: MLXArray? = nil
    ) {
        self.indicesPacked = indicesPacked
        self.vectorNorms = vectorNorms
        self.shape = shape
        self.indexBits = indexBits
        self.seed = seed
        self.sinkData = sinkData
    }

    public var estimatedBytes: Int {
        var total = indicesPacked.nbytes + vectorNorms.nbytes
        if let sink = sinkData { total += sink.nbytes }
        return total
    }

    public var compressionRatio: Float {
        guard shape.count == 4 else { return 1.0 }
        let originalBytes = shape.reduce(1, *) * 2
        guard estimatedBytes > 0 else { return Float.infinity }
        return Float(originalBytes) / Float(estimatedBytes)
    }
}
