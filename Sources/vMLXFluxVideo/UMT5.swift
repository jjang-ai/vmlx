import Foundation
@preconcurrency import MLX
import MLXNN
import MLXFast
import vMLXFluxKit

// MARK: - UMT5 — Wan-specific multilingual T5 encoder
//
// Wan 2.x conditions on UMT5-XXL (NOT T5-XXL — different vocab, different
// position-bias scheme). Architecture mirrors `mlx_video/models/wan_2/text_encoder.py`:
//
//   • Vocab 256_384, dim 4096, dim_attn 4096, dim_ffn 10_240
//   • 24 encoder layers, 64 heads
//   • Per-layer T5 relative-position bucket bias (NOT shared across layers
//     for UMT5 — `shared_pos = False` in the Wan reference). 32 buckets,
//     bidirectional, max distance 128.
//   • Gated FFN: `gate_proj * fc1`, gelu(tanh) on the gate.
//   • Outer norm = RMSNorm (T5LayerNorm uses `mx.fast.rms_norm`).
//   • Attention has NO scaling (T5 convention) — softmax is computed in
//     float32 to absorb the unscaled-logit dynamic range.
//
// STATUS: full architecture port. Ships with random weights so module
// assembly compiles and produces correct shapes. Real checkpoint loading
// happens in `WANModel.swift` via the standard `WeightLoader.load`.

public struct UMT5Config: Sendable {
    public let vocabSize: Int
    public let dim: Int
    public let dimAttn: Int
    public let dimFFN: Int
    public let numHeads: Int
    public let numLayers: Int
    public let numBuckets: Int
    public let maxDistance: Int

    public init(
        vocabSize: Int = 256_384,
        dim: Int = 4096,
        dimAttn: Int = 4096,
        dimFFN: Int = 10_240,
        numHeads: Int = 64,
        numLayers: Int = 24,
        numBuckets: Int = 32,
        maxDistance: Int = 128
    ) {
        self.vocabSize = vocabSize
        self.dim = dim
        self.dimAttn = dimAttn
        self.dimFFN = dimFFN
        self.numHeads = numHeads
        self.numLayers = numLayers
        self.numBuckets = numBuckets
        self.maxDistance = maxDistance
    }

    public static let umt5XXL = UMT5Config()
}

// MARK: - T5 RMSNorm

/// T5-style RMSNorm. Uses MLX-Fast for performance.
public final class UMT5RMSNorm: Module {
    public let weight: MLXArray
    public let eps: Float

    public init(dim: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dim])
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // MLXFast.rmsNorm matches mx.fast.rms_norm in Python.
        return MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

// MARK: - Relative-position bucket embedding

/// T5-style relative position bias with bucket discretization.
/// Bidirectional UMT5 variant. Returns (lq, lk, num_heads) embedding
/// transposed and broadcast-prefixed to shape (1, num_heads, lq, lk).
public final class UMT5RelativeEmbedding: Module {
    public let numBuckets: Int
    public let numHeads: Int
    public let maxDistance: Int
    public let embedding: Embedding

    public init(numBuckets: Int, numHeads: Int, maxDistance: Int = 128) {
        self.numBuckets = numBuckets
        self.numHeads = numHeads
        self.maxDistance = maxDistance
        self.embedding = Embedding(embeddingCount: numBuckets, dimensions: numHeads)
        super.init()
    }

    /// Bidirectional bucket mapping. Mirrors Python:
    ///   half = num_buckets // 2
    ///   sign of rel_pos picks first vs second half.
    private func bucket(_ relPos: MLXArray) -> MLXArray {
        let halfBuckets = numBuckets / 2
        let isPositive = (relPos .> MLXArray(0))
        let baseBucket = isPositive.asType(.int32) * MLXArray(Int32(halfBuckets))
        let absPos = abs(relPos)

        let maxExact = halfBuckets / 2
        let isSmall = (absPos .< MLXArray(Int32(maxExact)))

        // Logarithmic bucket for large positions.
        let absF = absPos.asType(.float32)
        let logTerm = log(absF / MLXArray(Float(maxExact)))
            / MLXArray(Float(log(Double(maxDistance) / Double(maxExact))))
        let largeRaw = MLXArray(Int32(maxExact))
            + (logTerm * MLXArray(Float(halfBuckets - maxExact))).asType(.int32)
        let cap = MLXArray(Int32(halfBuckets - 1))
        let large = minimum(largeRaw, cap)

        let small = absPos.asType(.int32)
        let combined = baseBucket + MLX.where(isSmall, small, large)
        return combined
    }

    /// Build the relative-position bias tensor.
    /// - Parameters:
    ///   - lq: query length
    ///   - lk: key length
    /// - Returns: (1, num_heads, lq, lk) additive bias.
    public func callAsFunction(lq: Int, lk: Int) -> MLXArray {
        let posK = MLXArray(0..<Int32(lk)).reshaped([1, lk])
        let posQ = MLXArray(0..<Int32(lq)).reshaped([lq, 1])
        let relPos = posK - posQ           // (lq, lk)
        let buckets = bucket(relPos)       // (lq, lk) int32
        let embeds = embedding(buckets)    // (lq, lk, numHeads)
        // Transpose to (numHeads, lq, lk), broadcast batch dim.
        let transposed = embeds.transposed(2, 0, 1)
        return transposed.reshaped([1, numHeads, lq, lk])
    }
}

// MARK: - T5 attention (no scaling)

public final class UMT5Attention: Module {
    public let numHeads: Int
    public let headDim: Int
    public let q: Linear
    public let k: Linear
    public let v: Linear
    public let o: Linear

    public init(dim: Int, dimAttn: Int, numHeads: Int) {
        precondition(dimAttn % numHeads == 0, "dim_attn must divide num_heads")
        self.numHeads = numHeads
        self.headDim = dimAttn / numHeads
        self.q = Linear(dim, dimAttn, bias: false)
        self.k = Linear(dim, dimAttn, bias: false)
        self.v = Linear(dim, dimAttn, bias: false)
        self.o = Linear(dimAttn, dim, bias: false)
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        posBias: MLXArray? = nil
    ) -> MLXArray {
        let b = x.dim(0)
        let n = numHeads
        let c = headDim

        // (B, L, dimAttn) → (B, L, N, C) → (B, N, L, C)
        let qProj = q(x).reshaped([b, -1, n, c]).transposed(0, 2, 1, 3)
        let kProj = k(x).reshaped([b, -1, n, c]).transposed(0, 2, 1, 3)
        let vProj = v(x).reshaped([b, -1, n, c]).transposed(0, 2, 1, 3)

        // T5: no 1/sqrt(d) scaling. Compute logits in float32 for precision.
        let qF = qProj.asType(.float32)
        let kF = kProj.asType(.float32)
        var attn = matmul(qF, kF.transposed(0, 1, 3, 2))  // (B, N, Lq, Lk)

        if let bias = posBias {
            attn = attn + bias.asType(.float32)
        }
        if let m = mask {
            // Convert 0/1 mask → 0 / -3.389e38 additive.
            let m4: MLXArray
            if m.ndim == 2 {
                m4 = m.reshaped([m.dim(0), 1, 1, m.dim(1)])
            } else if m.ndim == 3 {
                m4 = m.reshaped([m.dim(0), 1, m.dim(1), m.dim(2)])
            } else {
                m4 = m
            }
            let additive = MLX.where(
                m4 .== MLXArray(0),
                MLXArray(Float(-3.389e38)),
                MLXArray(Float(0))
            )
            attn = attn + additive.asType(.float32)
        }
        let probs = softmax(attn, axis: -1).asType(qProj.dtype)
        let out = matmul(probs, vProj)        // (B, N, Lq, C)
        let merged = out.transposed(0, 2, 1, 3).reshaped([b, -1, n * c])
        return o(merged)
    }
}

// MARK: - Gated FFN

public final class UMT5FeedForward: Module {
    public let gateProj: Linear
    public let fc1: Linear
    public let fc2: Linear

    public init(dim: Int, dimFFN: Int) {
        self.gateProj = Linear(dim, dimFFN, bias: false)
        self.fc1 = Linear(dim, dimFFN, bias: false)
        self.fc2 = Linear(dimFFN, dim, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // GELU(tanh) gate as in T5-1.1 / UMT5.
        let gate = geluApproximate(gateProj(x))
        let proj = fc1(x)
        return fc2(proj * gate)
    }
}

// MARK: - Encoder block

public final class UMT5Block: Module {
    public let norm1: UMT5RMSNorm
    public let attn: UMT5Attention
    public let norm2: UMT5RMSNorm
    public let ffn: UMT5FeedForward
    /// Per-layer relative-position bias. UMT5 keeps `shared_pos=False` so
    /// every block computes its own bucket embedding. Reference Python
    /// allows it to be nil when `shared_pos=True` (legacy T5).
    public let posEmbedding: UMT5RelativeEmbedding?
    public let sharedPos: Bool

    public init(
        dim: Int, dimAttn: Int, dimFFN: Int,
        numHeads: Int, numBuckets: Int,
        sharedPos: Bool
    ) {
        self.norm1 = UMT5RMSNorm(dim: dim)
        self.attn = UMT5Attention(dim: dim, dimAttn: dimAttn, numHeads: numHeads)
        self.norm2 = UMT5RMSNorm(dim: dim)
        self.ffn = UMT5FeedForward(dim: dim, dimFFN: dimFFN)
        self.sharedPos = sharedPos
        if sharedPos {
            self.posEmbedding = nil
        } else {
            self.posEmbedding = UMT5RelativeEmbedding(
                numBuckets: numBuckets, numHeads: numHeads
            )
        }
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        sharedPosBias: MLXArray? = nil
    ) -> MLXArray {
        let bias: MLXArray?
        if sharedPos {
            bias = sharedPosBias
        } else {
            bias = posEmbedding?(lq: x.dim(1), lk: x.dim(1))
        }
        let attnOut = attn(norm1(x), mask: mask, posBias: bias)
        let h1 = x + attnOut
        let ffnOut = ffn(norm2(h1))
        return h1 + ffnOut
    }
}

// MARK: - UMT5 Encoder

public final class UMT5Encoder: Module {
    public let config: UMT5Config
    public let tokenEmbedding: Embedding
    /// Shared (legacy T5) — set to nil for UMT5 default `shared_pos=False`.
    public let posEmbedding: UMT5RelativeEmbedding?
    public let blocks: [UMT5Block]
    public let norm: UMT5RMSNorm
    public let sharedPos: Bool

    public init(config: UMT5Config = .umt5XXL, sharedPos: Bool = false) {
        self.config = config
        self.tokenEmbedding = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.dim
        )
        self.sharedPos = sharedPos
        if sharedPos {
            self.posEmbedding = UMT5RelativeEmbedding(
                numBuckets: config.numBuckets, numHeads: config.numHeads
            )
        } else {
            self.posEmbedding = nil
        }
        self.blocks = (0..<config.numLayers).map { _ in
            UMT5Block(
                dim: config.dim, dimAttn: config.dimAttn, dimFFN: config.dimFFN,
                numHeads: config.numHeads, numBuckets: config.numBuckets,
                sharedPos: sharedPos
            )
        }
        self.norm = UMT5RMSNorm(dim: config.dim)
        super.init()
    }

    /// Forward pass.
    /// - Parameters:
    ///   - ids: (B, L) int32 token ids.
    ///   - mask: (B, L) optional 0/1 attention mask.
    /// - Returns: (B, L, dim) encoder hidden states.
    public func callAsFunction(ids: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var x = tokenEmbedding(ids)
        let bias: MLXArray? = sharedPos
            ? posEmbedding?(lq: x.dim(1), lk: x.dim(1))
            : nil
        for block in blocks {
            x = block(x, mask: mask, sharedPosBias: bias)
        }
        return norm(x)
    }
}

// MARK: - GELU (tanh approximation)
//
// `geluApproximate` from MLXNN.Activations matches `nn.GELU(approx="tanh")`
// exactly, so we re-export from MLXNN by keeping it as an `@inlinable`
// passthrough for vMLXFluxVideo callers that don't already import MLXNN.

// Reuses `MLXNN.geluApproximate` (imported via `import MLXNN` at top).
