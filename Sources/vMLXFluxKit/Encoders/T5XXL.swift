import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - T5-XXL encoder
//
// 1:1 Swift port of mflux's T5 encoder used by FLUX.1 Schnell/Dev/Fill/
// Kontext + FIBO. Architecture:
//
//   shared embedding (32128 → 4096)
//   24 × T5Block:
//     T5Attention (LN + multi-head SA with relative position bias)
//     T5FeedForward (LN + gated GeLU MLP, 4096 ↔ 10240)
//   final_layer_norm (T5LayerNorm)
//
// All linears are bias=False. T5LayerNorm is RMSNorm-style (no mean
// subtraction, no bias, fp32 variance).
//
// Reference:
//   /tmp/mflux-ref/src/mflux/models/flux/model/flux_text_encoder/t5_encoder/*
//
// Weight names (HuggingFace T5 v1.1 / FLUX `text_encoder_2/`):
//   shared.weight                                                   → embed.weight
//   encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight        → blocks[i].attn.{q,k,v,o}.weight
//   encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight (i==0) → repeated to all layers via remap
//   encoder.block.{i}.layer.0.layer_norm.weight                     → blocks[i].attnNorm.weight
//   encoder.block.{i}.layer.1.DenseReluDense.{wi_0,wi_1,wo}.weight  → blocks[i].ff.{wi0,wi1,wo}.weight
//   encoder.block.{i}.layer.1.layer_norm.weight                     → blocks[i].ffNorm.weight
//   encoder.final_layer_norm.weight                                 → finalNorm.weight

// MARK: - T5LayerNorm (RMS-style, fp32 variance)

public final class T5LayerNorm: Module {
    public let weight: MLXArray
    public let eps: Float

    public init(dim: Int = 4096, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dim])
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Compute variance in fp32 to match HF T5 numerics.
        let xf32 = x.asType(.float32)
        let variance = (xf32 * xf32).mean(axis: -1, keepDims: true)
        let normalized = x * rsqrt(variance + MLXArray(eps))
        return weight * normalized
    }
}

// MARK: - T5SelfAttention with relative position bias

public final class T5SelfAttention: Module {
    public let q: Linear
    public let k: Linear
    public let v: Linear
    public let o: Linear
    public let relativeAttentionBias: Embedding   // (32 buckets, 64 heads)
    public let dim: Int
    public let numHeads: Int
    public let headDim: Int

    public init(dim: Int = 4096, numHeads: Int = 64, headDim: Int = 64) {
        self.dim = dim
        self.numHeads = numHeads
        self.headDim = headDim
        self.q = Linear(dim, dim, bias: false)
        self.k = Linear(dim, dim, bias: false)
        self.v = Linear(dim, dim, bias: false)
        self.o = Linear(dim, dim, bias: false)
        // 32 buckets × num_heads (64 in T5-XXL).
        self.relativeAttentionBias = Embedding(embeddingCount: 32, dimensions: numHeads)
        super.init()
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let seqLen = hiddenStates.dim(1)
        let qOut = shape(q(hiddenStates))
        let kOut = shape(k(hiddenStates))
        let vOut = shape(v(hiddenStates))

        // (B, H, S, D) @ (B, H, D, S) → (B, H, S, S)
        var scores = matmul(qOut, kOut.transposed(0, 1, 3, 2))
        scores = scores + computeBias(seqLen: seqLen)
        let attn = softmax(scores, axis: -1)
        let out = unShape(matmul(attn, vOut))
        return o(out)
    }

    /// (B, S, dim) → (B, H, S, headDim)
    private func shape(_ states: MLXArray) -> MLXArray {
        let b = states.dim(0)
        let s = states.dim(1)
        return states.reshaped([b, s, numHeads, headDim]).transposed(0, 2, 1, 3)
    }

    /// (B, H, S, headDim) → (B, S, dim)
    private func unShape(_ states: MLXArray) -> MLXArray {
        let b = states.dim(0)
        let s = states.dim(2)
        return states.transposed(0, 2, 1, 3).reshaped([b, s, dim])
    }

    /// Compute T5 relative position bias for the current sequence.
    /// Returns (1, H, S, S).
    private func computeBias(seqLen: Int) -> MLXArray {
        let context = MLXArray((0..<seqLen).map { Int32($0) }).reshaped([seqLen, 1])
        let memory  = MLXArray((0..<seqLen).map { Int32($0) }).reshaped([1, seqLen])
        let relative = memory - context
        let bucket = Self.relativePositionBucket(relative)
        // (S, S, H)
        let values = relativeAttentionBias(bucket)
        // → (1, H, S, S)
        return values.transposed(2, 0, 1).reshaped([1, numHeads, seqLen, seqLen])
    }

    /// Standard T5 relative position bucket (bidirectional).
    public static func relativePositionBucket(
        _ relativePosition: MLXArray,
        bidirectional: Bool = true,
        numBuckets: Int = 32,
        maxDistance: Int = 128
    ) -> MLXArray {
        var buckets = MLXArray.zeros(relativePosition.shape, dtype: .int32)
        var nBuckets = numBuckets
        if bidirectional {
            nBuckets /= 2
            buckets = buckets + MLX.where(
                relativePosition .> MLXArray(Int32(0)),
                MLXArray(Int32(nBuckets)),
                MLXArray(Int32(0))
            )
        }
        let absRel = abs(relativePosition)
        let maxExact = nBuckets / 2
        let isSmall = absRel .< MLXArray(Int32(maxExact))

        let absRelF = absRel.asType(.float32)
        let logMax = Float(log(Double(maxDistance) / Double(maxExact)))
        let scale = Float(nBuckets - maxExact)
        let logged = log(absRelF / MLXArray(Float(maxExact))) / MLXArray(logMax) * MLXArray(scale)
        var ifLarge = MLXArray(Int32(maxExact)) + floor(logged).asType(.int32)
        ifLarge = minimum(ifLarge, MLXArray(Int32(nBuckets - 1)))

        buckets = buckets + MLX.where(isSmall, absRel.asType(.int32), ifLarge)
        return buckets
    }
}

// MARK: - T5DenseReluDense (gated GeLU MLP)

public final class T5DenseReluDense: Module {
    public let wi0: Linear   // mflux: wi_0 (gate)
    public let wi1: Linear   // mflux: wi_1 (linear)
    public let wo: Linear

    public init(dim: Int = 4096, ffnDim: Int = 10240) {
        self.wi0 = Linear(dim, ffnDim, bias: false)
        self.wi1 = Linear(dim, ffnDim, bias: false)
        self.wo  = Linear(ffnDim, dim, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gated = Self.newGelu(wi0(x))
        let linear = wi1(x)
        return wo(gated * linear)
    }

    /// HF T5 "new GELU" (tanh approximation, identical to BERT/GPT2).
    public static func newGelu(_ x: MLXArray) -> MLXArray {
        let cubic = x + MLXArray(Float(0.044715)) * (x * x * x)
        let inner = MLXArray(Float(0.7978845608)) * cubic   // sqrt(2/pi)
        return MLXArray(Float(0.5)) * x * (MLXArray(Float(1)) + tanh(inner))
    }
}

// MARK: - Attention + FF wrappers (mflux T5Attention / T5FeedForward)

public final class T5AttentionLayer: Module {
    public let layerNorm: T5LayerNorm
    public let SelfAttention: T5SelfAttention   // intentionally CamelCase to match HF weight key

    public init(dim: Int = 4096, numHeads: Int = 64, headDim: Int = 64) {
        self.layerNorm = T5LayerNorm(dim: dim)
        self.SelfAttention = T5SelfAttention(dim: dim, numHeads: numHeads, headDim: headDim)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed = layerNorm(x)
        let attn = SelfAttention(normed)
        return x + attn
    }
}

public final class T5FeedForwardLayer: Module {
    public let layerNorm: T5LayerNorm
    public let DenseReluDense: T5DenseReluDense   // CamelCase to match HF weight key

    public init(dim: Int = 4096, ffnDim: Int = 10240) {
        self.layerNorm = T5LayerNorm(dim: dim)
        self.DenseReluDense = T5DenseReluDense(dim: dim, ffnDim: ffnDim)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed = layerNorm(x)
        let mlp = DenseReluDense(normed)
        return x + mlp
    }
}

// MARK: - T5Block

public final class T5Block: Module {
    public let attention: T5AttentionLayer
    public let ff: T5FeedForwardLayer

    public init(dim: Int = 4096, numHeads: Int = 64, headDim: Int = 64, ffnDim: Int = 10240) {
        self.attention = T5AttentionLayer(dim: dim, numHeads: numHeads, headDim: headDim)
        self.ff = T5FeedForwardLayer(dim: dim, ffnDim: ffnDim)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return ff(attention(x))
    }
}

// MARK: - T5XXL encoder (final assembly)

public final class T5XXLEncoder: Module, TextEncoder, @unchecked Sendable {
    public let shared: Embedding   // 32128 × 4096
    public let blocks: [T5Block]
    public let finalLayerNorm: T5LayerNorm

    public let hiddenSize: Int
    public let maxSeqLen: Int

    public init(
        vocabSize: Int = 32128,
        hiddenSize: Int = 4096,
        numLayers: Int = 24,
        numHeads: Int = 64,
        headDim: Int = 64,
        ffnDim: Int = 10240,
        maxSeqLen: Int = 512
    ) {
        self.hiddenSize = hiddenSize
        self.maxSeqLen = maxSeqLen
        self.shared = Embedding(embeddingCount: vocabSize, dimensions: hiddenSize)
        self.blocks = (0..<numLayers).map { _ in
            T5Block(dim: hiddenSize, numHeads: numHeads, headDim: headDim, ffnDim: ffnDim)
        }
        self.finalLayerNorm = T5LayerNorm(dim: hiddenSize)
        super.init()
    }

    public func encode(tokenIds: [Int]) -> MLXArray {
        let truncated = tokenIds.prefix(maxSeqLen).map { Int32($0) }
        let tokens = MLXArray(Array(truncated)).reshaped([1, truncated.count])
        var h = shared(tokens)
        for block in blocks {
            h = block(h)
        }
        return finalLayerNorm(h)
    }

    /// Convenience for callers that already have an `(B, S)` tensor.
    public func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        var h = shared(tokens)
        for block in blocks {
            h = block(h)
        }
        return finalLayerNorm(h)
    }
}

// MARK: - HuggingFace weight remap helpers
//
// FLUX safetensors ship T5-XXL under `text_encoder_2/` with HF-style
// keys. The remap below converts `encoder.block.{i}.layer.{0|1}.*` into
// our flat `blocks.{i}.{attention,ff}.*` naming.

public enum T5XXLWeightRemap {

    /// Rewrite an HF T5 weight dict into the layout this Swift port
    /// expects. Returns a new dict with renamed keys; values are the
    /// same MLXArrays.
    public static func remap(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in weights {
            let mapped = remapKey(key)
            out[mapped] = value
        }
        return out
    }

    public static func remapKey(_ key: String) -> String {
        // Strip `text_encoder_2/` prefix if present.
        var k = key
        if k.hasPrefix("text_encoder_2.") {
            k.removeFirst("text_encoder_2.".count)
        }

        // shared.weight stays.
        if k == "shared.weight" { return "shared.weight" }

        // encoder.final_layer_norm.weight → finalLayerNorm.weight
        if k == "encoder.final_layer_norm.weight" {
            return "finalLayerNorm.weight"
        }

        // encoder.block.{i}.layer.0.* → blocks.{i}.attention.*
        // encoder.block.{i}.layer.1.* → blocks.{i}.ff.*
        if k.hasPrefix("encoder.block.") {
            let stripped = String(k.dropFirst("encoder.block.".count))
            // stripped = "{i}.layer.{j}.{rest}"
            let parts = stripped.split(separator: ".", maxSplits: 3,
                                       omittingEmptySubsequences: false)
            if parts.count == 4,
               parts[1] == "layer",
               let j = Int(parts[2]) {
                let i = parts[0]
                let rest = String(parts[3])
                let kind = (j == 0) ? "attention" : "ff"
                return "blocks.\(i).\(kind).\(rest)"
            }
        }

        // Unknown — return as-is. Callers can log & decide.
        return k
    }
}
