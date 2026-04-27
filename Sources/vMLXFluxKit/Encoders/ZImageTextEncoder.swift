// SPDX-License-Identifier: Apache-2.0
//
// Z-Image's own text encoder. Qwen-style 36-layer transformer with GQA
// (32 query heads / 8 KV heads), RoPE θ=1e6, RMSNorm eps=1e-6, vocab
// 151936. Returns the PENULTIMATE hidden layer per mflux convention.
//
// References:
//   /tmp/mflux-ref/src/mflux/models/z_image/model/z_image_text_encoder/text_encoder.py
//   /tmp/mflux-ref/src/mflux/models/z_image/model/z_image_text_encoder/attention.py
//   /tmp/mflux-ref/src/mflux/models/z_image/model/z_image_text_encoder/encoder_layer.py
//   /tmp/mflux-ref/src/mflux/models/z_image/model/z_image_text_encoder/mlp.py
//   /tmp/mflux-ref/src/mflux/models/z_image/weights/z_image_weight_mapping.py
//
// CRITICAL: Track 1 mistakenly wired T5-XXL + CLIP-L as the Z-Image
// encoder — that combo is FLUX.1's, not Z-Image's. Z-Image uses a
// single Qwen-style transformer encoder (single tokenizer, single
// model). Do NOT revert to dual encoders.

import Foundation
@preconcurrency import MLX
import MLXNN
import MLXFast

// MARK: - Config

public struct ZImageTextEncoderConfig: Sendable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let intermediateSize: Int
    public let headDim: Int
    public let maxPositionEmbeddings: Int
    public let ropeTheta: Float
    public let rmsNormEps: Float

    public init(
        vocabSize: Int = 151936,
        hiddenSize: Int = 2560,
        numHiddenLayers: Int = 36,
        numHeads: Int = 32,
        numKVHeads: Int = 8,
        intermediateSize: Int = 9728,
        headDim: Int = 128,
        maxPositionEmbeddings: Int = 40960,
        ropeTheta: Float = 1_000_000.0,
        rmsNormEps: Float = 1e-6
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.intermediateSize = intermediateSize
        self.headDim = headDim
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.rmsNormEps = rmsNormEps
    }
}

// MARK: - Encoder

public final class ZImageTextEncoder: Module, @unchecked Sendable {
    public let config: ZImageTextEncoderConfig

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo public var layers: [ZImageEncoderLayer]
    @ModuleInfo public var norm: RMSNorm

    public init(config: ZImageTextEncoderConfig) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            ZImageEncoderLayer(config: config)
        }
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps)
        super.init()
    }

    /// Forward pass. Returns the PENULTIMATE hidden state per mflux:
    /// `all_hidden_states[-2]`. That's the layer-(N-1) output, BEFORE
    /// the final-layer-norm — Z-Image's DiT consumes it directly.
    public func callAsFunction(inputIds: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        let s = inputIds.dim(1)
        var hidden = embedTokens(inputIds).asType(.float32)

        let causal = ZImageTextEncoder.causalMask(
            seqLen: s, dtype: hidden.dtype, padding: attentionMask)

        // Track penultimate. mflux collects `all_hidden_states` then
        // returns index -2; equivalent: snapshot the input to the last
        // layer (which IS the penultimate output).
        var penultimate = hidden
        for layer in layers {
            penultimate = hidden
            hidden = layer(hiddenStates: hidden, attentionMask: causal)
        }
        return penultimate.asType(.float16)
    }

    /// Build a `(1, 1, S, S)` causal mask plus optional `(B, S)` padding
    /// fold-in. Returns -inf above the diagonal and at padded positions.
    private static func causalMask(seqLen s: Int, dtype: DType, padding: MLXArray?) -> MLXArray {
        let idx = MLXArray(0..<Int32(s))
        // Broadcast comparison via expandedDimensions (the only safe
        // public surface).
        let lhs = idx.expandedDimensions(axis: 1)   // (S, 1)
        let rhs = idx.expandedDimensions(axis: 0)   // (1, S)
        let allow = lhs .>= rhs                     // (S, S) bool
        let zero = MLXArray.zeros([s, s], type: Float.self).asType(dtype)
        let negInf = MLXArray.full([s, s], values: MLXArray(-Float.infinity)).asType(dtype)
        var causal = MLX.where(allow, zero, negInf)
        // (S, S) → (1, 1, S, S) — broadcasts cleanly across (B, H, …).
        causal = causal.expandedDimensions(axis: 0).expandedDimensions(axis: 0)

        guard let pad = padding else { return causal }
        // pad: (B, S) of {0, 1}. Build (B, 1, 1, S) mask: 0 keep, -inf mask.
        let b = pad.dim(0)
        let padBcast = pad.reshaped([b, 1, 1, s])
        let allowed = padBcast .== MLXArray(Int32(1))
        let zeroPad = MLXArray.zeros([b, 1, 1, s], type: Float.self).asType(dtype)
        let negInfPad = MLXArray.full([b, 1, 1, s], values: MLXArray(-Float.infinity)).asType(dtype)
        let padMask = MLX.where(allowed, zeroPad, negInfPad)
        return causal + padMask
    }
}

// MARK: - Encoder layer

public final class ZImageEncoderLayer: Module, @unchecked Sendable {
    @ModuleInfo(key: "input_layernorm") public var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") public var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "self_attn") public var selfAttn: ZImageAttention
    @ModuleInfo public var mlp: ZImageMLP

    public init(config: ZImageTextEncoderConfig) {
        self._inputLayerNorm.wrappedValue =
            RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue =
            RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._selfAttn.wrappedValue = ZImageAttention(config: config)
        self._mlp.wrappedValue = ZImageMLP(config: config)
        super.init()
    }

    public func callAsFunction(hiddenStates h: MLXArray, attentionMask mask: MLXArray) -> MLXArray {
        var residual = h
        var x = inputLayerNorm(h)
        x = selfAttn(x, mask: mask)
        x = residual + x
        residual = x
        x = postAttentionLayerNorm(x)
        x = mlp(x)
        return residual + x
    }
}

// MARK: - Attention (GQA + RoPE)

public final class ZImageAttention: Module, @unchecked Sendable {
    public let cfg: ZImageTextEncoderConfig

    @ModuleInfo(key: "q_proj") public var qProj: Linear
    @ModuleInfo(key: "k_proj") public var kProj: Linear
    @ModuleInfo(key: "v_proj") public var vProj: Linear
    @ModuleInfo(key: "o_proj") public var oProj: Linear
    @ModuleInfo(key: "q_norm") public var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") public var kNorm: RMSNorm

    public init(config c: ZImageTextEncoderConfig) {
        self.cfg = c
        self._qProj.wrappedValue = Linear(c.hiddenSize, c.numHeads * c.headDim, bias: false)
        self._kProj.wrappedValue = Linear(c.hiddenSize, c.numKVHeads * c.headDim, bias: false)
        self._vProj.wrappedValue = Linear(c.hiddenSize, c.numKVHeads * c.headDim, bias: false)
        self._oProj.wrappedValue = Linear(c.numHeads * c.headDim, c.hiddenSize, bias: false)
        // mflux applies RMSNorm with default eps=1e-6 over the head_dim.
        self._qNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
        super.init()
    }

    public func callAsFunction(_ h: MLXArray, mask: MLXArray) -> MLXArray {
        let b = h.dim(0)
        let s = h.dim(1)

        // Project + reshape to (B, S, Hq/Hkv, D)
        var q = qProj(h).reshaped([b, s, cfg.numHeads, cfg.headDim])
        var k = kProj(h).reshaped([b, s, cfg.numKVHeads, cfg.headDim])
        var v = vProj(h).reshaped([b, s, cfg.numKVHeads, cfg.headDim])

        q = qNorm(q)
        k = kNorm(k)

        // mflux applies RoPE BEFORE the head/seq transpose, on the
        // (B, S, H, D) layout. After RoPE we transpose to (B, H, S, D)
        // for the SDPA call. We use the runtime-fused RoPE op which
        // expects rotation along the last axis — it accepts arbitrary
        // leading dims so (B, S, H, D) is fine.
        q = MLXFast.RoPE(
            q,
            dimensions: cfg.headDim,
            traditional: false,
            base: cfg.ropeTheta,
            scale: 1.0,
            offset: 0)
        k = MLXFast.RoPE(
            k,
            dimensions: cfg.headDim,
            traditional: false,
            base: cfg.ropeTheta,
            scale: 1.0,
            offset: 0)

        // GQA: tile KV heads to match Q heads. mflux uses
        // `mx.repeat(_, num_kv_groups, axis=2)` which interleaves
        // (head 0 → group_size copies, head 1 → group_size copies, …).
        let groups = cfg.numHeads / cfg.numKVHeads
        if groups > 1 {
            k = MLX.repeated(k, count: groups, axis: 2)
            v = MLX.repeated(v, count: groups, axis: 2)
        }

        // Transpose for SDPA: (B, H, S, D)
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        let scale = Float(1.0 / sqrt(Float(cfg.headDim)))
        let attn = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask)

        // (B, H, S, D) → (B, S, H*D)
        let merged = attn.transposed(0, 2, 1, 3).reshaped([b, s, cfg.numHeads * cfg.headDim])
        return oProj(merged)
    }
}

// MARK: - SwiGLU MLP

public final class ZImageMLP: Module, @unchecked Sendable {
    @ModuleInfo(key: "gate_proj") public var gateProj: Linear
    @ModuleInfo(key: "up_proj")   public var upProj:   Linear
    @ModuleInfo(key: "down_proj") public var downProj: Linear

    public init(config c: ZImageTextEncoderConfig) {
        self._gateProj.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        self._upProj.wrappedValue   = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        super.init()
    }

    public func callAsFunction(_ h: MLXArray) -> MLXArray {
        return downProj(MLXNN.silu(gateProj(h)) * upProj(h))
    }
}

// MARK: - Weight key remap (Task 4)

extension ZImageTextEncoder {
    /// Strip the `model.` prefix that HuggingFace Qwen2-style safetensors
    /// use. Mirrors mflux's `ZImageWeightMapping.get_text_encoder_mapping()`
    /// which renames every `model.foo.bar.weight` → `foo.bar.weight`.
    /// Source: /tmp/mflux-ref/src/mflux/models/z_image/weights/z_image_weight_mapping.py
    public static func remapKey(_ k: String) -> String {
        if k.hasPrefix("model.") {
            return String(k.dropFirst("model.".count))
        }
        return k
    }

    /// Apply `remapKey` over a full `[String: MLXArray]` weight dict.
    public static func remap(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        for (k, v) in weights {
            out[remapKey(k)] = v
        }
        return out
    }
}
