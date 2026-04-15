//
//  MiMoV2Flash.swift
//  LLM
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mimo_v2_flash.py
//  Created by Ronald Mannak on 2025/1/8.
//

import Foundation
import MLX
import vMLXLMCommon
import MLXNN

private func attentionWithCacheUpdateAndSinks(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil
) -> MLXArray {
    guard let cache else {
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask,
            sinks: sinks
        )
    }

    if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        precondition(sinks == nil, "Quantized SDPA does not support attention sinks.")
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys, values: values)
        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale,
            mask: mask,
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode
        )
    } else {
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            scale: scale,
            mask: mask,
            sinks: sinks
        )
    }
}

private func groupExpertSelect(
    gates: MLXArray,
    eScoreCorrectionBias: MLXArray,
    topK: Int,
    nGroup: Int,
    topkGroup: Int,
    routedScalingFactor: Float,
    normTopkProb: Bool
) -> (MLXArray, MLXArray) {
    var scores = sigmoid(gates)
    let originalScores = scores
    scores = scores + eScoreCorrectionBias

    if nGroup > 1 {
        scores = unflatten(scores, axis: -1, shape: [nGroup, -1])
        let groupScores = top(scores, k: 2, axis: -1).sum(axis: -1, keepDims: true)
        let k = nGroup - topkGroup
        let groupIdx = argPartition(groupScores, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
        scores = putAlong(
            scores,
            stopGradient(groupIdx),
            values: MLXArray(0.0, dtype: scores.dtype),
            axis: -2
        )
        scores = flattened(scores, start: -2, end: -1)
    }

    let k = topK
    let inds = argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
    scores = takeAlong(originalScores, inds, axis: -1)
    if topK > 1, normTopkProb {
        let denominator = scores.sum(axis: -1, keepDims: true)
        scores = scores / (denominator + MLXArray(1e-20, dtype: scores.dtype))
    }
    scores = scores * routedScalingFactor

    return (inds, scores)
}

class MiMoV2FlashAttention: Module {
    let args: MiMoV2FlashConfiguration
    let isSlidingWindow: Bool
    let hasSinks: Bool
    let scale: Float

    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let vHeadDim: Int

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ParameterInfo(key: "attention_sink_bias") var attentionSinkBias: MLXArray

    let rope: RoPE

    init(_ args: MiMoV2FlashConfiguration, isSlidingWindow: Bool) {
        self.args = args
        self.isSlidingWindow = isSlidingWindow

        if isSlidingWindow {
            self.numAttentionHeads = args.swaAttentionHeads
            self.numKeyValueHeads = args.swaKvHeads
            self.hasSinks = args.addSwaAttentionSinkBias
            self.headDim = args.swaHeadDim
            self.vHeadDim = args.swaVHeadDim
        } else {
            self.numAttentionHeads = args.attentionHeads
            self.numKeyValueHeads = args.kvHeads
            self.hasSinks = args.addFullAttentionSinkBias
            self.headDim = args.headDim
            self.vHeadDim = args.vHeadDim
        }

        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(
            args.hiddenSize, numAttentionHeads * headDim, bias: false)
        _wk.wrappedValue = Linear(
            args.hiddenSize, numKeyValueHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(
            args.hiddenSize, numKeyValueHeads * vHeadDim, bias: false)
        _wo.wrappedValue = Linear(
            numAttentionHeads * vHeadDim, args.hiddenSize, bias: false)

        _attentionSinkBias.wrappedValue = MLXArray.ones([numAttentionHeads])

        let ropeTheta = isSlidingWindow ? args.swaRopeTheta : args.ropeTheta
        let rotaryDims = Int(Float(args.partialRotaryFactor) * Float(headDim))
        self.rope = RoPE(
            dimensions: rotaryDims,
            traditional: false,
            base: ropeTheta
        )
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let queries = wq(x)
        let keys = wk(x)
        let values = wv(x)

        var q = queries.reshaped(B, L, numAttentionHeads, -1).transposed(0, 2, 1, 3)
        var k = keys.reshaped(B, L, numKeyValueHeads, -1).transposed(0, 2, 1, 3)
        let v = values.reshaped(B, L, numKeyValueHeads, -1).transposed(0, 2, 1, 3)

        q = applyRotaryPosition(rope, to: q, cache: cache)
        k = applyRotaryPosition(rope, to: k, cache: cache)

        let output = attentionWithCacheUpdateAndSinks(
            queries: q,
            keys: k,
            values: v,
            cache: cache,
            scale: scale,
            mask: mask,
            sinks: hasSinks ? attentionSinkBias : nil
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }

    override func updateMissing(
        parameter: String,
        verify: VerifyUpdate,
        path: [String],
        modulePath: [String]
    ) throws {
        if parameter == "attention_sink_bias", hasSinks {
            // Keep the default you already set in init (ones([numAttentionHeads]))
            return
        }
        try super.updateMissing(
            parameter: parameter, verify: verify, path: path, modulePath: modulePath)
    }
}

class MiMoV2FlashMLP: Module, UnaryLayer {
    let hiddenSize: Int
    let intermediateSize: Int

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: MiMoV2FlashConfiguration, hiddenSize: Int? = nil, intermediateSize: Int? = nil) {
        self.hiddenSize = hiddenSize ?? config.hiddenSize
        self.intermediateSize = intermediateSize ?? config.intermediateSize

        _gateProj.wrappedValue = Linear(self.hiddenSize, self.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(self.hiddenSize, self.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(self.intermediateSize, self.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

class MiMoV2FlashMoEGate: Module {
    let topK: Int
    let normTopkProb: Bool
    let nRoutedExperts: Int
    let routedScalingFactor: Float
    let nGroup: Int
    let topkGroup: Int

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: MiMoV2FlashConfiguration) {
        guard let nRoutedExperts = config.nRoutedExperts else {
            fatalError("MiMoV2FlashMoEGate requires nRoutedExperts.")
        }

        precondition(config.topkMethod == "noaux_tc", "Unsupported topk method.")

        self.topK = config.numExpertsPerTok
        self.normTopkProb = config.normTopkProb
        self.nRoutedExperts = nRoutedExperts
        self.routedScalingFactor = config.routedScalingFactor ?? 1.0
        self.nGroup = config.nGroup
        self.topkGroup = config.topkGroup

        _weight.wrappedValue = MLXArray.zeros([nRoutedExperts, config.hiddenSize])
        _eScoreCorrectionBias.wrappedValue = MLXArray.zeros([nRoutedExperts])

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        return groupExpertSelect(
            gates: x.matmul(weight.T),
            eScoreCorrectionBias: eScoreCorrectionBias,
            topK: topK,
            nGroup: nGroup,
            topkGroup: topkGroup,
            routedScalingFactor: routedScalingFactor,
            normTopkProb: normTopkProb
        )
    }
}

class MiMoV2FlashMoE: Module, UnaryLayer {
    let numExpertsPerTok: Int
    let gate: MiMoV2FlashMoEGate

    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: MiMoV2FlashMLP?

    init(_ config: MiMoV2FlashConfiguration) {
        guard let nRoutedExperts = config.nRoutedExperts else {
            fatalError("MiMoV2FlashMoE requires nRoutedExperts.")
        }

        self.numExpertsPerTok = config.numExpertsPerTok
        self.gate = MiMoV2FlashMoEGate(config)

        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: nRoutedExperts
        )

        if let shared = config.nSharedExperts {
            let intermediateSize = config.moeIntermediateSize * shared
            _sharedExperts.wrappedValue = MiMoV2FlashMLP(
                config, intermediateSize: intermediateSize)
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        var y = switchMLP(x, inds)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

class MiMoV2FlashDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MiMoV2FlashAttention
    let mlp: UnaryLayer
    let isSlidingWindow: Bool

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: MiMoV2FlashConfiguration, isMoe: Bool, isSlidingWindow: Bool) {
        self.isSlidingWindow = isSlidingWindow
        _selfAttn.wrappedValue = MiMoV2FlashAttention(config, isSlidingWindow: isSlidingWindow)
        self.mlp = isMoe ? MiMoV2FlashMoE(config) : MiMoV2FlashMLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.layernormEpsilon)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.layernormEpsilon)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let residual = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return residual + mlp(postAttentionLayerNorm(residual))
    }
}

public class MiMoV2FlashModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [MiMoV2FlashDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let swaIdx: Int
    let gaIdx: Int
    let slidingWindowSize: Int
    let hybridLayerPattern: [Int]

    init(_ config: MiMoV2FlashConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)

        self.layers = (0 ..< config.hiddenLayers).map { index in
            MiMoV2FlashDecoderLayer(
                config,
                isMoe: config.moeLayerFreq[index] == 1,
                isSlidingWindow: config.hybridLayerPattern[index] == 1
            )
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.layernormEpsilon)
        self.swaIdx = config.hybridLayerPattern.firstIndex(of: 1) ?? 0
        self.gaIdx = config.hybridLayerPattern.firstIndex(of: 0) ?? 0
        self.slidingWindowSize = config.slidingWindowSize
        self.hybridLayerPattern = config.hybridLayerPattern
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)

        let fullMask = createAttentionMask(h: h, cache: cache?[gaIdx])
        let swaMask = createAttentionMask(
            h: h, cache: cache?[swaIdx], windowSize: slidingWindowSize)

        for (i, layer) in layers.enumerated() {
            let mask = hybridLayerPattern[i] == 1 ? swaMask : fullMask
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class MiMoV2FlashModel: Module, LLMModel, KVCacheDimensionProvider {
    public let modelType: String
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: MiMoV2FlashModelInner
    let configuration: MiMoV2FlashConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: MiMoV2FlashConfiguration) {
        self.configuration = config
        self.modelType = config.modelType
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        self.model = MiMoV2FlashModelInner(config)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        return lmHead(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        func dequant(weight: MLXArray, scaleInv: MLXArray) -> MLXArray {
            let dtype = weight.dtype
            let bs = 128
            let (m, n) = (weight.shape[0], weight.shape[1])
            let padBottom = bs * scaleInv.dim(0) - m
            let padSide = bs * scaleInv.dim(1) - n

            var paddedWeight = padded(
                weight, widths: [.init((0, padBottom)), .init((0, padSide))])
            paddedWeight = paddedWeight.reshaped(
                [(m + padBottom) / bs, bs, (n + padSide) / bs, bs])
            let scaled = paddedWeight * scaleInv[0..., .newAxis, 0..., .newAxis]
            return scaled.reshaped([m + padBottom, n + padSide])[0 ..< m, 0 ..< n]
                .asType(dtype)
        }

        var newWeights: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.contains("weight_scale_inv") {
                let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = weights[weightKey] {
                    newWeights[weightKey] = dequant(weight: weight, scaleInv: value)
                }
            } else if newWeights[key] == nil {
                newWeights[key] = value
            }
        }

        var sanitizedWeights = newWeights.isEmpty ? weights : newWeights

        for layerIndex in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layerIndex)"
            for (_, projName) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
                for key in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).mlp.experts.0.\(projName).\(key)"
                    if sanitizedWeights[firstKey] != nil {
                        let toJoin = (0 ..< (configuration.nRoutedExperts ?? 1)).map {
                            sanitizedWeights.removeValue(
                                forKey: "\(prefix).mlp.experts.\($0).\(projName).\(key)")!
                        }
                        sanitizedWeights["\(prefix).mlp.switch_mlp.\(projName).\(key)"] =
                            MLX.stacked(toJoin)
                    }
                }
            }
        }

        return sanitizedWeights.filter { key, _ in
            !key.hasPrefix("model.mtp")
        }
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        return model.layers.map { layer in
            if layer.isSlidingWindow {
                return RotatingKVCache(maxSize: configuration.slidingWindowSize)
            } else {
                return KVCacheSimple()
            }
        }
    }
}

// MARK: - Configuration

public struct MiMoV2FlashConfiguration: Codable, Sendable {
    var modelType: String = "mimo_v2_flash"
    var numExpertsPerTok: Int
    var hybridLayerPattern: [Int]
    var moeLayerFreq: [Int]
    var addSwaAttentionSinkBias: Bool
    var addFullAttentionSinkBias: Bool
    var slidingWindowSize: Int
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var nSharedExperts: Int?
    var nRoutedExperts: Int?
    var routedScalingFactor: Float?
    var topkMethod: String
    var scoringFunc: String
    var normTopkProb: Bool
    var nGroup: Int
    var topkGroup: Int
    var maxPositionEmbeddings: Int
    var layernormEpsilon: Float
    var ropeTheta: Float
    var swaRopeTheta: Float
    var swaAttentionHeads: Int
    var swaKvHeads: Int
    var headDim: Int
    var vHeadDim: Int
    var swaHeadDim: Int
    var swaVHeadDim: Int
    var partialRotaryFactor: Float

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case numExpertsPerTok = "num_experts_per_tok"
        case hybridLayerPattern = "hybrid_layer_pattern"
        case moeLayerFreq = "moe_layer_freq"
        case addSwaAttentionSinkBias = "add_swa_attention_sink_bias"
        case addFullAttentionSinkBias = "add_full_attention_sink_bias"
        case slidingWindowSize = "sliding_window_size"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case topkMethod = "topk_method"
        case scoringFunc = "scoring_func"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case maxPositionEmbeddings = "max_position_embeddings"
        case layernormEpsilon = "layernorm_epsilon"
        case ropeTheta = "rope_theta"
        case swaRopeTheta = "swa_rope_theta"
        case swaAttentionHeads = "swa_num_attention_heads"
        case swaKvHeads = "swa_num_key_value_heads"
        case headDim = "head_dim"
        case vHeadDim = "v_head_dim"
        case swaHeadDim = "swa_head_dim"
        case swaVHeadDim = "swa_v_head_dim"
        case partialRotaryFactor = "partial_rotary_factor"
    }
}

// MARK: - LoRA

extension MiMoV2FlashModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
