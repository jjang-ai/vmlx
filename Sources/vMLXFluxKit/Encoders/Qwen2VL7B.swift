import Foundation
@preconcurrency import MLX
import MLXNN
import MLXFast

// MARK: - Qwen2-VL-7B encoder
//
// Used by Qwen-Image (text→image) and Qwen-Image-Edit (image+text→image).
// Architecture (mflux ref):
//
//   embed_tokens       : Embedding(152064 × 3584)
//   layers[28]         : QwenEncoderLayer
//     input_layernorm           (QwenRMSNorm 3584, eps=1e-6)
//     self_attn                 (Q 28h × 128d, KV 4h × 128d, GQA, mRoPE)
//     post_attention_layernorm  (QwenRMSNorm 3584)
//     mlp                       (gate/up/down 3584 ↔ 18944, SwiGLU)
//   norm               : QwenRMSNorm(3584)
//   rotary_emb         : QwenRotaryEmbedding(head_dim=128, base=1e6)
//   visual             : Optional QwenVisionTransformer (vision tower, ~1B)
//
// Track 1 scope: text-only forward pass (no image inputs). Vision tower
// remains nil until Track 2's image-edit work loads it. The reusable
// surface across tracks is `encode(tokenIds:)` for text-only conditioning,
// plus a `encodeWithImages(...)` extension point that Track 2 can wire.
//
// Reference:
//   /tmp/mflux-ref/src/mflux/models/qwen/model/qwen_text_encoder/*

// MARK: - QwenRMSNorm

public final class QwenRMSNormLocal: Module {
    public let weight: MLXArray
    public let eps: Float

    public init(dim: Int = 3584, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dim])
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf32 = x.asType(.float32)
        let variance = (xf32 * xf32).mean(axis: -1, keepDims: true)
        let normalized = x * rsqrt(variance + MLXArray(eps))
        return weight * normalized
    }
}

// MARK: - Qwen rotary embedding (multimodal mRoPE-capable)

public struct QwenRotaryEmbedding: Sendable {
    public let dim: Int
    public let base: Float
    public let invFreq: MLXArray   // (dim/2,)

    public init(dim: Int, base: Float = 1_000_000.0) {
        self.dim = dim
        self.base = base
        let half = dim / 2
        let exponents = (0..<half).map { Float($0 * 2) / Float(dim) }
        let invFreq = exponents.map { pow(base, -$0) }
        self.invFreq = MLXArray(invFreq)
    }

    /// positionIds: (3, B, S) — three axes for time/H/W in mRoPE.
    /// Returns (cos, sin) each shaped (3, B, S, dim).
    public func callAsFunction(positionIds: MLXArray) -> (MLXArray, MLXArray) {
        // (3, B, S) × (D/2,) → (3, B, S, D/2)
        let posF = positionIds.asType(.float32)
        let freqs = posF.expandedDimensions(axis: -1) * invFreq
            .reshaped([1, 1, 1, dim / 2])
        let emb = concatenated([freqs, freqs], axis: -1)
        return (cos(emb), sin(emb))
    }
}

// MARK: - QwenAttention with GQA + multimodal RoPE

public final class QwenAttention: Module {
    public let qProj: Linear
    public let kProj: Linear
    public let vProj: Linear
    public let oProj: Linear

    public let hiddenSize: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let mropeSection: [Int]   // default [16, 24, 24]

    public init(
        hiddenSize: Int = 3584,
        numHeads: Int = 28,
        numKVHeads: Int = 4,
        mropeSection: [Int] = [16, 24, 24]
    ) {
        self.hiddenSize = hiddenSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.headDim = hiddenSize / numHeads
        self.mropeSection = mropeSection

        self.qProj = Linear(hiddenSize, numHeads * headDim, bias: true)
        self.kProj = Linear(hiddenSize, numKVHeads * headDim, bias: true)
        self.vProj = Linear(hiddenSize, numKVHeads * headDim, bias: true)
        self.oProj = Linear(numHeads * headDim, hiddenSize, bias: false)
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXArray?,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray)
    ) -> MLXArray {
        let b = x.dim(0)
        let s = x.dim(1)

        var q = qProj(x).reshaped([b, s, numHeads, headDim]).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped([b, s, numKVHeads, headDim]).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped([b, s, numKVHeads, headDim]).transposed(0, 2, 1, 3)

        // mRoPE: split rotary dim into 3 sections (one per spatial axis),
        // pick the matching axis from positionEmbeddings (index 0=time,
        // 1=H, 2=W), concat back.
        let (cosM, sinM) = applyMRoPESelect(
            cos: positionEmbeddings.cos,
            sin: positionEmbeddings.sin
        )
        q = applyRotary(q, cos: cosM, sin: sinM)
        k = applyRotary(k, cos: cosM, sin: sinM)

        // GQA: repeat KV heads to match Q heads.
        if numKVHeads != numHeads {
            let nRep = numHeads / numKVHeads
            k = repeatKV(k, nRep: nRep)
            v = repeatKV(v, nRep: nRep)
        }

        let scale = Float(1.0 / sqrt(Float(headDim)))
        let attnOut = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: attentionMask
        )
        let merged = attnOut.transposed(0, 2, 1, 3).reshaped([b, s, hiddenSize])
        return oProj(merged)
    }

    /// Slice + reassemble cos/sin per mRoPE section pattern.
    /// Input cos/sin: (3, B, S, headDim). Output: (B, 1, S, headDim) ready
    /// to broadcast against (B, H, S, headDim) Q/K.
    private func applyMRoPESelect(cos: MLXArray, sin: MLXArray) -> (MLXArray, MLXArray) {
        // Doubled because rotary applies to pairs.
        let doubled = mropeSection.map { $0 * 2 }
        var cosChunks: [MLXArray] = []
        var sinChunks: [MLXArray] = []
        var start = 0
        for (i, sectionSize) in doubled.enumerated() {
            let end = start + sectionSize
            // (3, B, S, sec)
            let cosChunk = cos[.ellipsis, start..<end]
            let sinChunk = sin[.ellipsis, start..<end]
            // Pick axis i % 3.
            cosChunks.append(cosChunk[i % 3])  // → (B, S, sec)
            sinChunks.append(sinChunk[i % 3])
            start = end
        }
        let cosCombined = concatenated(cosChunks, axis: -1)  // (B, S, headDim)
        let sinCombined = concatenated(sinChunks, axis: -1)
        // Add a head axis: (B, 1, S, headDim)
        return (
            cosCombined.expandedDimensions(axis: 1),
            sinCombined.expandedDimensions(axis: 1)
        )
    }

    /// Apply rotary embedding: x' = x*cos + rotateHalf(x)*sin (fp32 inside)
    private func applyRotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let origDtype = x.dtype
        let xf = x.asType(.float32)
        let cosf = cos.asType(.float32)
        let sinf = sin.asType(.float32)
        let rotated = (xf * cosf) + (rotateHalf(xf) * sinf)
        return rotated.asType(origDtype)
    }

    private func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let x1 = x[.ellipsis, 0..<half]
        let x2 = x[.ellipsis, half..<x.dim(-1)]
        return concatenated([-x2, x1], axis: -1)
    }

    /// Repeat KV heads to match Q head count for GQA.
    private func repeatKV(_ x: MLXArray, nRep: Int) -> MLXArray {
        // (B, KV_H, S, D) → (B, KV_H, nRep, S, D) → (B, KV_H * nRep, S, D)
        let b = x.dim(0); let h = x.dim(1); let s = x.dim(2); let d = x.dim(3)
        let expanded = x.expandedDimensions(axis: 2)
        let broadcasted = MLX.broadcast(expanded, to: [b, h, nRep, s, d])
        return broadcasted.reshaped([b, h * nRep, s, d])
    }
}

// MARK: - QwenMLP (SwiGLU)

public final class QwenMLP: Module {
    public let gateProj: Linear
    public let upProj: Linear
    public let downProj: Linear

    public init(hiddenSize: Int = 3584, ffnDim: Int = 18944) {
        self.gateProj = Linear(hiddenSize, ffnDim, bias: false)
        self.upProj = Linear(hiddenSize, ffnDim, bias: false)
        self.downProj = Linear(ffnDim, hiddenSize, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - QwenEncoderLayer

public final class QwenEncoderLayer: Module {
    public let inputLayernorm: QwenRMSNormLocal
    public let selfAttn: QwenAttention
    public let postAttentionLayernorm: QwenRMSNormLocal
    public let mlp: QwenMLP

    public init(
        hiddenSize: Int = 3584,
        numHeads: Int = 28,
        numKVHeads: Int = 4,
        ffnDim: Int = 18944,
        eps: Float = 1e-6
    ) {
        self.inputLayernorm = QwenRMSNormLocal(dim: hiddenSize, eps: eps)
        self.selfAttn = QwenAttention(
            hiddenSize: hiddenSize, numHeads: numHeads, numKVHeads: numKVHeads
        )
        self.postAttentionLayernorm = QwenRMSNormLocal(dim: hiddenSize, eps: eps)
        self.mlp = QwenMLP(hiddenSize: hiddenSize, ffnDim: ffnDim)
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXArray?,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray)
    ) -> MLXArray {
        var h = x + selfAttn(
            inputLayernorm(x),
            attentionMask: attentionMask,
            positionEmbeddings: positionEmbeddings
        )
        h = h + mlp(postAttentionLayernorm(h))
        return h
    }
}

// MARK: - Qwen2-VL-7B encoder (final assembly, text-only)

public final class Qwen2VL7BEncoder: Module, TextEncoder, @unchecked Sendable {
    public let embedTokens: Embedding
    public let layers: [QwenEncoderLayer]
    public let norm: QwenRMSNormLocal

    public let hiddenSize: Int
    public let maxSeqLen: Int

    public let rope: QwenRotaryEmbedding
    public let imageTokenId: Int = 151_655

    public init(
        vocabSize: Int = 152_064,
        hiddenSize: Int = 3584,
        numLayers: Int = 28,
        numHeads: Int = 28,
        numKVHeads: Int = 4,
        ffnDim: Int = 18_944,
        maxSeqLen: Int = 32_768,
        ropeBase: Float = 1_000_000.0
    ) {
        self.hiddenSize = hiddenSize
        self.maxSeqLen = maxSeqLen
        self.embedTokens = Embedding(embeddingCount: vocabSize, dimensions: hiddenSize)
        self.layers = (0..<numLayers).map { _ in
            QwenEncoderLayer(
                hiddenSize: hiddenSize,
                numHeads: numHeads,
                numKVHeads: numKVHeads,
                ffnDim: ffnDim
            )
        }
        self.norm = QwenRMSNormLocal(dim: hiddenSize)
        self.rope = QwenRotaryEmbedding(dim: hiddenSize / numHeads, base: ropeBase)
        super.init()
    }

    public func encode(tokenIds: [Int]) -> MLXArray {
        let truncated = tokenIds.prefix(maxSeqLen).map { Int32($0) }
        let tokens = MLXArray(Array(truncated)).reshaped([1, truncated.count])
        let attentionMask = MLXArray.ones([1, truncated.count], dtype: .int32)
        return forward(inputIds: tokens, attentionMask: attentionMask)
    }

    /// Text-only forward. mRoPE position_ids is built as (3, B, S) with all
    /// three axes equal to a 1D arange — text tokens have no spatial info.
    public func forward(inputIds: MLXArray, attentionMask: MLXArray) -> MLXArray {
        let b = inputIds.dim(0)
        let s = inputIds.dim(1)

        var hidden = embedTokens(inputIds)

        // position_ids = arange(S) tiled across (3, B, S)
        let positions = MLXArray((0..<s).map { Int32($0) })
        let posIds = MLX.broadcast(
            positions.reshaped([1, 1, s]),
            to: [3, b, s]
        )

        let positionEmb = rope(positionIds: posIds)

        // Build a 4D causal mask + padding mask (B, 1, S, S) in fp32.
        let causal = Self.causalMask(seqLen: s)
        let padding = Self.paddingMask(attentionMask: attentionMask)
        let mask4d = causal + padding

        for layer in layers {
            hidden = layer(
                hidden,
                attentionMask: mask4d,
                positionEmbeddings: (cos: positionEmb.0, sin: positionEmb.1)
            )
        }
        return norm(hidden)
    }

    /// (1, 1, S, S) lower-triangular mask: 0 below diag, -inf above.
    public static func causalMask(seqLen: Int) -> MLXArray {
        let ones = MLXArray.ones([seqLen, seqLen])
        let lower = tril(ones, k: 0)
        let upper = MLXArray(Float(1)) - lower
        let m = upper * MLXArray(Float(-1e30))
        return m.reshaped([1, 1, seqLen, seqLen])
    }

    /// Convert a (B, S) attention mask (1=valid, 0=pad) into a
    /// broadcastable (B, 1, 1, S) additive mask.
    public static func paddingMask(attentionMask: MLXArray) -> MLXArray {
        let f = attentionMask.asType(.float32)
        let valid = MLXArray(Float(1)) - f
        let m = valid * MLXArray(Float(-1e30))
        return m.expandedDimensions(axis: 1).expandedDimensions(axis: 1)
    }
}

// MARK: - HuggingFace Qwen2-VL weight remap
//
// Qwen2-VL-7B safetensors ship with HF naming under either the model's
// top-level keys (`model.embed_tokens.weight`, `model.layers.{i}.*`,
// `model.norm.weight`) or with a `text_encoder/` prefix when embedded
// in a Qwen-Image bundle. Remap normalizes both forms.

public enum Qwen2VL7BWeightRemap {

    public static func remap(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (k, v) in weights {
            if let mapped = remapKey(k) {
                out[mapped] = v
            }
        }
        return out
    }

    /// Returns nil if the key doesn't belong to the text encoder
    /// (e.g. it's a vision-tower or LM-head weight).
    public static func remapKey(_ key: String) -> String? {
        var k = key
        if k.hasPrefix("text_encoder.") { k.removeFirst("text_encoder.".count) }
        if k.hasPrefix("model.")        { k.removeFirst("model.".count) }

        // Skip vision tower (Track 2 ports it).
        if k.hasPrefix("visual.") { return nil }
        // Skip lm_head.
        if k.hasPrefix("lm_head.") { return nil }

        // embed_tokens.weight stays.
        if k == "embed_tokens.weight" { return "embedTokens.weight" }
        // norm.weight stays.
        if k == "norm.weight" { return "norm.weight" }

        // layers.{i}.{...} → layers.{i}.{...}  (rename HF snake_case to camel)
        if k.hasPrefix("layers.") {
            return k
                .replacingOccurrences(of: "input_layernorm",          with: "inputLayernorm")
                .replacingOccurrences(of: "post_attention_layernorm", with: "postAttentionLayernorm")
                .replacingOccurrences(of: "self_attn.q_proj",         with: "selfAttn.qProj")
                .replacingOccurrences(of: "self_attn.k_proj",         with: "selfAttn.kProj")
                .replacingOccurrences(of: "self_attn.v_proj",         with: "selfAttn.vProj")
                .replacingOccurrences(of: "self_attn.o_proj",         with: "selfAttn.oProj")
                .replacingOccurrences(of: "mlp.gate_proj",            with: "mlp.gateProj")
                .replacingOccurrences(of: "mlp.up_proj",              with: "mlp.upProj")
                .replacingOccurrences(of: "mlp.down_proj",            with: "mlp.downProj")
        }
        return nil
    }
}
