// Copyright © 2025 Apple Inc.

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/deepseek_v3.py

import Foundation
import MLX
import vMLXLMCommon
import MLXNN

public struct DeepseekV3Configuration: Codable, Sendable {
    var vocabSize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var nSharedExperts: Int?
    var nRoutedExperts: Int?
    var routedScalingFactor: Float
    var kvLoraRank: Int
    var qLoraRank: Int
    var qkRopeHeadDim: Int
    var vHeadDim: Int
    var qkNopeHeadDim: Int
    var normTopkProb: Bool
    var nGroup: Int?
    var topkGroup: Int?
    var numExpertsPerTok: Int?
    var moeLayerFreq: Int
    var firstKDenseReplace: Int
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var attentionBias: Bool

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeLayerFreq = "moe_layer_freq"
        case firstKDenseReplace = "first_k_dense_replace"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
    }
}

private func yarnFindCorrectionDim(
    numRotations: Float, dim: Float, base: Float = 10000, maxPositionEmbeddings: Float = 2048
) -> Float {
    return (dim * log(maxPositionEmbeddings / (numRotations * 2 * Float.pi))) / (2 * log(base))
}

private func yarnFindCorrectionRange(
    lowRot: Float, highRot: Float, dim: Float, base: Float = 10000,
    maxPositionEmbeddings: Float = 2048
) -> (Float, Float) {
    let low = floor(
        yarnFindCorrectionDim(
            numRotations: lowRot, dim: dim, base: base, maxPositionEmbeddings: maxPositionEmbeddings
        ))
    let high = ceil(
        yarnFindCorrectionDim(
            numRotations: highRot, dim: dim, base: base,
            maxPositionEmbeddings: maxPositionEmbeddings))
    return (max(low, 0), min(high, dim - 1))
}

private func clippedSilu(_ x: MLXArray) -> MLXArray {
    clip(x * sigmoid(x), min: -100, max: 100)
}

class DeepseekV3Attention: Module {
    var config: DeepseekV3Configuration
    var hiddenSize: Int
    var numHeads: Int
    var maxPositionEmbeddings: Int
    var ropeTheta: Float
    var qLoraRank: Int?
    var qkRopeHeadDim: Int
    var kvLoraRank: Int
    var vHeadDim: Int
    var qkNopeHeadDim: Int
    var qHeadDim: Int
    var scale: Float

    let rope: RoPELayer
    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear

    init(config: DeepseekV3Configuration) {
        self.config = config
        self.hiddenSize = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.maxPositionEmbeddings = config.maxPositionEmbeddings
        self.ropeTheta = config.ropeTheta
        self.qLoraRank = config.qLoraRank
        self.qkRopeHeadDim = config.qkRopeHeadDim
        self.kvLoraRank = config.kvLoraRank
        self.vHeadDim = config.vHeadDim
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim

        self.scale = pow(Float(qHeadDim), -0.5)

        if let qLoraRank = qLoraRank {
            self._qAProj.wrappedValue = Linear(
                hiddenSize, qLoraRank, bias: config.attentionBias
            )
            self._qALayerNorm.wrappedValue = RMSNorm(dimensions: qLoraRank)
            self._qBProj.wrappedValue = Linear(
                qLoraRank, numHeads * qHeadDim, bias: false
            )
        } else {
            self._qProj.wrappedValue = Linear(hiddenSize, numHeads * qHeadDim, bias: false)
        }

        self._kvAProjWithMqa.wrappedValue = Linear(
            hiddenSize,
            kvLoraRank + qkRopeHeadDim,
            bias: config.attentionBias
        )
        self._kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank)
        self._kvBProj.wrappedValue = Linear(
            kvLoraRank,
            numHeads * (qHeadDim - qkRopeHeadDim + vHeadDim),
            bias: false
        )
        self._oProj.wrappedValue = Linear(
            numHeads * vHeadDim, hiddenSize, bias: config.attentionBias)

        if let ropeScaling = config.ropeScaling {
            let mScaleAllDim = ropeScaling["mscale_all_dim"]?.asFloat() ?? 0.0
            if mScaleAllDim != 0 {
                let scalingFactor = ropeScaling["factor"]?.asFloat() ?? 1.0
                if scalingFactor > 1 {
                    let s = 0.1 * mScaleAllDim * log(scalingFactor) + 1.0
                    self.scale = self.scale * s * s
                }
            }
        }

        self.rope = initializeRope(
            dims: qkRopeHeadDim, base: ropeTheta, traditional: true,
            scalingConfig: config.ropeScaling, maxPositionEmbeddings: maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q: MLXArray
        if qLoraRank == nil {
            q = self.qProj!(x)
        } else {
            q = self.qBProj!(self.qALayerNorm!(self.qAProj!(x)))
        }

        q = q.reshaped(B, L, self.numHeads, self.qHeadDim).transposed(0, 2, 1, 3)
        let splitQ = split(q, indices: [qkNopeHeadDim], axis: -1)
        var (qNope, qPe) = (splitQ[0], splitQ[1])
        var compressedKv = self.kvAProjWithMqa(x)
        let splitCompressedKv = split(compressedKv, indices: [kvLoraRank], axis: -1)
        compressedKv = splitCompressedKv[0]
        var kPe = splitCompressedKv[1]
        kPe = kPe.reshaped(B, L, 1, self.qkRopeHeadDim).transposed(0, 2, 1, 3)
        var kv = self.kvBProj(kvALayerNorm(compressedKv))
        kv = kv.reshaped(B, L, self.numHeads, -1).transposed(0, 2, 1, 3)
        let splitKv = split(kv, indices: [self.qkNopeHeadDim], axis: -1)

        var (kNope, values) = (splitKv[0], splitKv[1])

        qPe = applyRotaryPosition(rope, to: qPe, cache: cache)
        kPe = applyRotaryPosition(rope, to: kPe, cache: cache)
        kPe = repeated(kPe, count: numHeads, axis: 1)

        var keys: MLXArray
        if let cache = cache {
            (keys, values) = cache.update(
                keys: concatenated([kNope, kPe], axis: -1), values: values)
        } else {
            keys = concatenated([kNope, kPe], axis: -1)
        }

        let queries = concatenated([qNope, qPe], axis: -1)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return self.oProj(output)
    }
}

class DeepseekV3MLP: Module, UnaryLayer {
    var config: DeepseekV3Configuration
    var hiddenSize: Int
    var intermediateSize: Int
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: DeepseekV3Configuration, hiddenSize: Int? = nil, intermediateSize: Int? = nil) {
        self.config = config
        self.hiddenSize = hiddenSize ?? config.hiddenSize
        self.intermediateSize = intermediateSize ?? config.intermediateSize
        self._gateProj.wrappedValue = Linear(self.hiddenSize, self.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(self.hiddenSize, self.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(self.intermediateSize, self.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        self.downProj(silu(self.gateProj(x)) * self.upProj(x))
    }
}

class MoEGate: Module {
    var config: DeepseekV3Configuration
    var topK: Int?
    var normTopkProb: Bool
    var nRoutedExperts: Int?
    var routedScalingFactor: Float
    var nGroup: Int
    var topkGroup: Int?

    var weight: MLXArray
    var e_score_correction_bias: MLXArray

    init(config: DeepseekV3Configuration) {
        self.config = config
        self.topK = config.numExpertsPerTok
        self.normTopkProb = config.normTopkProb
        self.nRoutedExperts = config.nRoutedExperts
        self.routedScalingFactor = config.routedScalingFactor
        self.nGroup = config.nGroup ?? 1
        self.topkGroup = config.topkGroup
        self.weight = zeros([self.nRoutedExperts ?? 1, config.hiddenSize])
        self.e_score_correction_bias = zeros([self.nRoutedExperts ?? 1])
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let (bsz, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))

        let hiddenStates = x.matmul(weight.T)
        var scores = sigmoid(hiddenStates)
        let scoresForChoice = scores + e_score_correction_bias
        let groupScores = scoresForChoice.reshaped(bsz, seqLen, self.nGroup, -1)
        let topKGroup = top(groupScores, k: 2, axis: -1).sum(axis: -1, keepDims: true)
        var k = nGroup - (topkGroup ?? 1)
        var groupIdx = argPartition(topKGroup, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
        groupIdx = broadcast(groupIdx, to: [bsz, seqLen, k, (nRoutedExperts ?? 1) / nGroup])
        scores = putAlong(groupScores, stopGradient(groupIdx), values: MLXArray(0.0, dtype: groupScores.dtype), axis: -2)
        scores = flattened(scores, start: -2, end: -1)

        k = topK ?? 1
        let inds = argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        scores = takeAlong(scores, inds, axis: -1)
        if topK ?? 1 > 1, normTopkProb {
            let denominator = scores.sum(axis: -1, keepDims: true) + MLXArray(1e-20, dtype: scores.dtype)
            scores = scores / denominator
            scores = scores * routedScalingFactor
        }

        return (inds, scores)
    }
}

class DeepseekV3MoE: Module, UnaryLayer {
    var config: DeepseekV3Configuration
    var numExpertsPerTok: Int
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    var gate: MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV3MLP?

    init(config: DeepseekV3Configuration) {
        self.config = config
        self.numExpertsPerTok = config.numExpertsPerTok ?? 1

        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts ?? 1,
            activation: clippedSilu
        )

        self.gate = MoEGate(config: config)

        if let sharedExpertCount = config.nSharedExperts {
            let intermediateSize = config.moeIntermediateSize * sharedExpertCount
            self._sharedExperts.wrappedValue = DeepseekV3MLP(
                config: config, intermediateSize: intermediateSize)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        var y = switchMLP(x, indices)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)

        if let shared = sharedExperts {
            y = y + shared(x)
        }
        return y
    }
}

class DeepseekV3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DeepseekV3Attention
    var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(config: DeepseekV3Configuration, layerIdx: Int) {
        self._selfAttn.wrappedValue = DeepseekV3Attention(config: config)

        if config.nRoutedExperts != nil,
            layerIdx >= config.firstKDenseReplace,
            layerIdx % config.moeLayerFreq == 0
        {
            self.mlp = DeepseekV3MoE(config: config)
        } else {
            self.mlp = DeepseekV3MLP(config: config)
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        let r2 = mlp(postAttentionLayerNorm(h))
        return h + r2
    }
}

public class DeepseekV3ModelInner: Module {
    var config: DeepseekV3Configuration
    var vocabSize: Int
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [DeepseekV3DecoderLayer]
    var startIdx: Int
    var endIdx: Int
    var numLayers: Int
    @ModuleInfo(key: "norm") var norm: RMSNorm
    var pipelineRank: Int
    var pipelineSize: Int

    init(config: DeepseekV3Configuration) {
        self.config = config
        self.vocabSize = config.vocabSize
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map {
            DeepseekV3DecoderLayer(config: config, layerIdx: $0)
        }
        self.startIdx = 0
        self.endIdx = layers.count
        self.numLayers = endIdx
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.pipelineRank = 0
        self.pipelineSize = 1
    }

    func callAsFunction(_ x: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(x)

        let attentionMask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: attentionMask, cache: cache?[i])
        }

        return norm(h)
    }

    /// Forward pass that also captures per-layer hidden states at each
    /// requested decoder index. Used by JANG-DFlash to feed target
    /// hiddens into the drafter's KV injection (see DFlash §4.1).
    ///
    /// DeepSeek V3 uses MLA attention — the internal compressed K/V
    /// representations are not a stable tap surface, so the tap is the
    /// **post-decoder-block residual hidden state** (same `h` as standard
    /// forward). This matches MiniMax / Mistral 4 semantics and the
    /// drafter's distillation contract.
    ///
    /// - Parameters:
    ///   - inputs: token ID tensor `[B, L]`
    ///   - cache:  optional per-layer KV caches (may be nil for cacheless)
    ///   - tapLayers: decoder indices whose post-layer hidden state to
    ///     capture
    ///   - providedMask: when non-nil, replaces the auto-built causal
    ///     mask (used for tree-attention verification during spec-dec)
    func callAsFunctionWithTaps(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        tapLayers: Set<Int>,
        providedMask: MLXFast.ScaledDotProductAttentionMaskMode? = nil
    ) -> (output: MLXArray, taps: [Int: MLXArray]) {
        var h = embedTokens(inputs)

        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if let providedMask {
            mask = providedMask
        } else {
            mask = createAttentionMask(h: h, cache: cache?.first)
        }

        var taps: [Int: MLXArray] = [:]
        taps.reserveCapacity(tapLayers.count)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
            if tapLayers.contains(i) {
                taps[i] = h
            }
        }

        return (norm(h), taps)
    }
}

public class DeepseekV3Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public var kvHeads: [Int] = []

    var args: DeepseekV3Configuration
    public var model: DeepseekV3ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ args: DeepseekV3Configuration) {
        self.args = args
        self.model = DeepseekV3ModelInner(config: args)
        self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let out = model(inputs, cache: cache)
        return lmHead(out)
    }

    /// Forward pass that exposes per-layer hidden-state taps alongside
    /// the logits. Wraps `DeepseekV3ModelInner.callAsFunctionWithTaps`
    /// and routes the final hidden state through `lm_head` the same way
    /// as the standard forward.
    public func callAsFunctionWithTaps(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        tapLayers: Set<Int>,
        providedMask: MLXFast.ScaledDotProductAttentionMaskMode? = nil
    ) -> (logits: MLXArray, taps: [Int: MLXArray]) {
        let (out, taps) = model.callAsFunctionWithTaps(
            inputs, cache: cache, tapLayers: tapLayers, providedMask: providedMask)
        return (lmHead(out), taps)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights = weights

        func dequant(weight: MLXArray, scaleInv: MLXArray) -> MLXArray {
            let bs = 128
            let (m, n) = (weight.shape[0], weight.shape[1])
            let padBottom = (bs - m % bs) % bs
            let padSide = (bs - n % bs) % bs

            var padded = padded(weight, widths: [.init((0, padBottom)), .init((0, padSide))])
            padded = padded.reshaped([(m + padBottom) / bs, bs, (n + padSide) / bs, bs])
            let scaled = padded * scaleInv[0..., .newAxis, 0..., .newAxis]
            return scaled.reshaped([m + padBottom, n + padSide])[0 ..< m, 0 ..< n]
        }

        for (key, value) in weights {
            if key.contains("weight_scale_inv") {
                let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = weights[weightKey] {
                    let dequantized = dequant(weight: weight, scaleInv: value)
                    newWeights[weightKey] = dequantized
                }
            } else if newWeights[key] == nil {
                newWeights[key] = value
            }
        }

        for l in 0 ..< args.numHiddenLayers {
            let prefix = "model.layers.\(l)"
            for (_, projName) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
                for key in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).mlp.experts.0.\(projName).\(key)"
                    if weights[firstKey] != nil {
                        let joined = (0 ..< (args.nRoutedExperts ?? 1)).map {
                            weights["\(prefix).mlp.experts.\($0).\(projName).\(key)"]!
                        }
                        newWeights["\(prefix).mlp.switch_mlp.\(projName).\(key)"] = stacked(joined)
                    }
                }
            }
        }

        return newWeights.filter { key, _ in
            !key.starts(with: "model.layers.61") && !key.contains("rotary_emb.inv_freq")
        }
    }

    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - JANG-DFlash target adapter
//
// Bridges `DeepseekV3Model.callAsFunctionWithTaps(...)` to the
// architecture-agnostic `JangDFlashTarget` protocol defined in
// vMLXLMCommon/DFlash/JangDFlashSpecDec.swift. Mirrors
// `MiniMaxDFlashTarget` — kept as an adapter (rather than a direct
// conformance) so vMLXLMCommon stays model-agnostic.
public final class DeepseekV3DFlashTarget: JangDFlashTarget {
    public let model: DeepseekV3Model

    public init(_ model: DeepseekV3Model) { self.model = model }

    public func forwardWithTaps(
        inputs: MLXArray,
        cache: [KVCache]?,
        tapLayers: Set<Int>,
        providedMask: MLXFast.ScaledDotProductAttentionMaskMode?
    ) -> (logits: MLXArray, taps: [Int: MLXArray]) {
        return model.callAsFunctionWithTaps(
            inputs,
            cache: cache,
            tapLayers: tapLayers,
            providedMask: providedMask
        )
    }

    public func makeCache() -> [KVCache] {
        // Same disambiguation trick used in `MiniMaxDFlashTarget` — pass
        // `parameters: nil` explicitly so the compiler picks the
        // non-deprecated overload of `makePromptCache`.
        return makePromptCache(model: model, parameters: nil)
    }
}
