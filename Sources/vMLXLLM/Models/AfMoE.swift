//  AfMoE.swift
//  mlx-swift-lm
//
//  Created by Sachin Desai on 12/2/25.
//

// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/afmoe.py

import Foundation
import MLX
import vMLXLMCommon
import MLXNN

// MARK: - Configuration

public struct AfMoEConfiguration: Codable, Sendable {
    var modelType: String = "afmoe"
    var vocabularySize: Int = 200192
    var hiddenSize: Int = 2048
    var intermediateSize: Int = 6144
    var moeIntermediateSize: Int = 1024
    var hiddenLayers: Int = 32
    var attentionHeads: Int = 32
    var kvHeads: Int = 4
    var headDim: Int = 64
    var maxPositionEmbeddings: Int = 131072
    var rmsNormEps: Float = 1e-5
    var ropeTheta: Float = 10000
    var ropeScaling: [String: StringOrNumber]? = nil
    var tieWordEmbeddings: Bool = false

    // MoE config
    var numExperts: Int = 128
    var numExpertsPerToken: Int = 8
    var numSharedExperts: Int = 1
    var numDenseLayers: Int = 2
    var routeNorm: Bool = true
    var routeScale: Float = 2.826
    var scoreFunc: String = "sigmoid"
    var nGroup: Int = 1
    var topkGroup: Int = 1

    // Attention config
    var layerTypes: [String]
    var slidingWindow: Int = 2048

    // muP config
    var mupEnabled: Bool = true

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case numSharedExperts = "num_shared_experts"
        case numDenseLayers = "num_dense_layers"
        case routeNorm = "route_norm"
        case routeScale = "route_scale"
        case scoreFunc = "score_func"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case mupEnabled = "mup_enabled"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "afmoe"
        self.vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 200192
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144
        self.moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 1024
        self.hiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        self.attentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 128
        self.numExpertsPerToken =
            try container.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 8
        self.numSharedExperts =
            try container.decodeIfPresent(Int.self, forKey: .numSharedExperts) ?? 1
        self.numDenseLayers =
            try container.decodeIfPresent(Int.self, forKey: .numDenseLayers) ?? 2
        self.routeNorm = try container.decodeIfPresent(Bool.self, forKey: .routeNorm) ?? true
        self.routeScale = try container.decodeIfPresent(Float.self, forKey: .routeScale) ?? 2.826
        self.scoreFunc = try container.decodeIfPresent(String.self, forKey: .scoreFunc) ?? "sigmoid"
        self.nGroup = try container.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        self.topkGroup = try container.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
        self.layerTypes = try container.decode([String].self, forKey: .layerTypes)
        self.slidingWindow =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 2048
        self.mupEnabled = try container.decodeIfPresent(Bool.self, forKey: .mupEnabled) ?? true
    }
}

// MARK: - Attention

class AfMoEAttention: Module {
    let hiddenSize: Int
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let isLocalAttention: Bool
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    // AfMoE specific: Q/K normalization
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    // AfMoE specific: attention gating
    @ModuleInfo(key: "gate_proj") var gateProj: Linear

    // RoPE is only used for local (sliding window) attention
    let rope: RoPELayer?

    init(_ args: AfMoEConfiguration, isLocalAttention: Bool = false) {
        self.hiddenSize = args.hiddenSize
        self.nHeads = args.attentionHeads
        self.nKVHeads = args.kvHeads
        self.headDim = args.headDim
        self.isLocalAttention = isLocalAttention
        self.scale = pow(Float(headDim), -0.5)

        self._qProj.wrappedValue = Linear(hiddenSize, nHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(hiddenSize, nKVHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(hiddenSize, nKVHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(nHeads * headDim, hiddenSize, bias: false)

        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        self._gateProj.wrappedValue = Linear(hiddenSize, nHeads * headDim, bias: false)

        // RoPE is only used for local (sliding window) attention
        if isLocalAttention {
            self.rope = initializeRope(
                dims: headDim, base: args.ropeTheta,
                traditional: false, scalingConfig: args.ropeScaling,
                maxPositionEmbeddings: nil
            )
        } else {
            self.rope = nil
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        // Reshape for multi-head attention
        queries = queries.reshaped(B, L, nHeads, headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)

        // Apply Q/K normalization
        queries = qNorm(queries)
        keys = kNorm(keys)

        // Apply RoPE only for local (sliding window) attention
        if isLocalAttention, let rope = rope {
            queries = applyRotaryPosition(rope, to: queries, cache: cache)
            keys = applyRotaryPosition(rope, to: keys, cache: cache)
        }

        var output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )

        output = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)

        // Apply attention gating: gate_proj(x) -> sigmoid -> multiply
        let gate = sigmoid(gateProj(x))
        output = output * gate

        return oProj(output)
    }
}

// MARK: - MLP

class AfMoEMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - MoE Router

class MoERouter: Module {
    @ModuleInfo(key: "gate") var gate: Linear

    init(_ args: AfMoEConfiguration) {
        self._gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        gate(x)
    }
}

// MARK: - AfMoE MoE

class AfMoEMoE: Module, UnaryLayer {
    let numExperts: Int
    let numExpertsPerTok: Int
    let routeNorm: Bool
    let routeScale: Float
    let scoreFunc: String
    let nGroup: Int
    let topkGroup: Int
    let numSharedExperts: Int

    @ModuleInfo var router: MoERouter
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray
    @ModuleInfo(key: "experts") var experts: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: AfMoEMLP?

    init(_ args: AfMoEConfiguration) {
        self.numExperts = args.numExperts
        self.numExpertsPerTok = args.numExpertsPerToken
        self.routeNorm = args.routeNorm
        self.routeScale = args.routeScale
        self.scoreFunc = args.scoreFunc
        self.nGroup = args.nGroup
        self.topkGroup = args.topkGroup
        self.numSharedExperts = args.numSharedExperts

        _router.wrappedValue = MoERouter(args)
        self._expertBias.wrappedValue = MLXArray.zeros([args.numExperts])
        self._experts.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts
        )

        if args.numSharedExperts > 0 {
            let sharedIntermediateSize = args.moeIntermediateSize * args.numSharedExperts
            self._sharedExperts.wrappedValue = AfMoEMLP(
                dimensions: args.hiddenSize, hiddenDimensions: sharedIntermediateSize)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Get routing scores
        let gates = router(x)

        var scores: MLXArray
        if scoreFunc == "sigmoid" {
            scores = sigmoid(gates)
        } else {
            scores = softmax(gates, axis: -1, precise: true)
        }

        // Add expert bias for selection
        var selectionScores = scores + expertBias

        // Group-based expert selection if nGroup > 1
        if nGroup > 1 {
            selectionScores = unflatten(selectionScores, axis: -1, shape: [nGroup, -1])
            let groupScores = top(selectionScores, k: 2, axis: -1).sum(axis: -1, keepDims: true)
            let k = nGroup - topkGroup
            let groupIdx = argPartition(groupScores, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
            selectionScores = putAlong(
                selectionScores, stopGradient(groupIdx), values: MLXArray(0.0, dtype: selectionScores.dtype), axis: -2)
            selectionScores = flattened(selectionScores, start: -2, end: -1)
        }

        // Select top-k experts
        let k = numExpertsPerTok
        let inds = argPartition(-selectionScores, kth: k - 1, axis: -1)[.ellipsis, ..<k]

        // Get original scores for selected experts (without bias)
        var selectedScores = takeAlong(scores, inds, axis: -1)

        // Normalize scores if enabled
        if routeNorm && numExpertsPerTok > 1 {
            let denominator = selectedScores.sum(axis: -1, keepDims: true)
            selectedScores = selectedScores / denominator
        }

        // Apply route scale
        selectedScores = selectedScores * routeScale

        // Apply experts
        var y = experts(x, inds)
        y = (y * selectedScores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)

        // Add shared expert output
        if let sharedExperts = sharedExperts {
            y = y + sharedExperts(x)
        }

        return y
    }
}

// MARK: - Decoder Layer

class AfMoEDecoderLayer: Module {
    let hiddenSize: Int
    let useSliding: Bool
    let layerIdx: Int

    @ModuleInfo(key: "self_attn") var selfAttn: AfMoEAttention
    var mlp: UnaryLayer

    // Dual normalization: 4 layer norms total
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_mlp_layernorm") var preMlpLayerNorm: RMSNorm
    @ModuleInfo(key: "post_mlp_layernorm") var postMlpLayerNorm: RMSNorm

    init(_ args: AfMoEConfiguration, layerIdx: Int, useSliding: Bool = false) {
        self.hiddenSize = args.hiddenSize
        self.useSliding = useSliding
        self.layerIdx = layerIdx

        self._selfAttn.wrappedValue = AfMoEAttention(args, isLocalAttention: useSliding)

        // First numDenseLayers use regular MLP, rest use MoE
        if layerIdx < args.numDenseLayers {
            self.mlp = AfMoEMLP(
                dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        } else {
            self.mlp = AfMoEMoE(args)
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._preMlpLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postMlpLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        // Self-attention with pre and post normalization
        var r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        r = postAttentionLayerNorm(r)
        let h = x + r

        // MLP with pre and post normalization
        r = mlp(preMlpLayerNorm(h))
        r = postMlpLayerNorm(r)
        return h + r
    }
}

// MARK: - AfMoE Model Inner

private class AfMoEModelInner: Module {
    let args: AfMoEConfiguration
    let slidingWindow: Int
    let mupEnabled: Bool
    let hiddenSize: Int

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [AfMoEDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    // Indices for full and sliding attention layers
    let faIdx: Int
    let swaIdx: Int?

    init(_ args: AfMoEConfiguration) {
        self.args = args
        self.slidingWindow = args.slidingWindow
        self.mupEnabled = args.mupEnabled
        self.hiddenSize = args.hiddenSize

        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        // Build layers based on layer_types
        var layerList: [AfMoEDecoderLayer] = []
        for (idx, layerType) in args.layerTypes.enumerated() {
            let useSliding = layerType == "sliding_attention"
            layerList.append(AfMoEDecoderLayer(args, layerIdx: idx, useSliding: useSliding))
        }
        self.layers = layerList

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        // Find indices for attention mask creation
        if let idx = args.layerTypes.firstIndex(of: "full_attention") {
            self.faIdx = idx
        } else {
            self.faIdx = 0
        }

        self.swaIdx = layerList.firstIndex(where: { $0.useSliding })
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        // muP scaling: scale embeddings by sqrt(hiddenSize)
        if mupEnabled {
            h = h * sqrt(Float(hiddenSize))
        }

        var layerCache = cache
        if layerCache == nil {
            layerCache = Array(repeating: nil as KVCache?, count: layers.count) as? [KVCache]
        }

        // Create attention masks
        let faMask = createAttentionMask(h: h, cache: layerCache?[faIdx])

        var swaMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        if let swaIdx = swaIdx, let layerCache = layerCache {
            // Create mask with sliding window
            swaMask = createAttentionMask(
                h: h, cache: layerCache[swaIdx], windowSize: slidingWindow)
        }

        for (i, layer) in layers.enumerated() {
            let mask = layer.useSliding ? swaMask : faMask
            h = layer(h, mask: mask, cache: layerCache?[i])
        }

        return norm(h)
    }
}

// MARK: - AfMoE Model (Public)

public class AfMoEModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let slidingWindow: Int

    fileprivate let model: AfMoEModelInner
    let configuration: AfMoEConfiguration
    fileprivate let layerUsesSliding: [Bool]

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: AfMoEConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.slidingWindow = args.slidingWindow
        self.model = AfMoEModelInner(args)

        // Track which layers use sliding attention
        self.layerUsesSliding = args.layerTypes.map { $0 == "sliding_attention" }

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead = lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = weights

        // Remove unused precomputed rotary freqs
        sanitizedWeights = sanitizedWeights.filter { !$0.key.contains("rotary_emb.inv_freq") }

        // Remove lm_head if tied embeddings
        if configuration.tieWordEmbeddings {
            sanitizedWeights["lm_head.weight"] = nil
        }

        // Stack expert weights for SwitchGLU
        for l in 0 ..< configuration.hiddenLayers {
            if l < configuration.numDenseLayers {
                continue
            }
            let prefix = "model.layers.\(l)"
            for n in ["up_proj", "down_proj", "gate_proj"] {
                for k in ["weight", "scales", "biases"] {
                    if sanitizedWeights["\(prefix).mlp.experts.0.\(n).\(k)"] != nil {
                        let toJoin = (0 ..< configuration.numExperts).map { e in
                            sanitizedWeights.removeValue(
                                forKey: "\(prefix).mlp.experts.\(e).\(n).\(k)")!
                        }
                        // RAM safety: per-layer materialize via canonical helper.
                        sanitizedWeights["\(prefix).mlp.experts.\(n).\(k)"] =
                            loadTimeMaterializedStacked(toJoin)
                    }
                }
            }
        }

        return sanitizedWeights
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        // Create cache based on layer type (rotating for sliding attention, simple for full attention)
        layerUsesSliding.map { usesSliding in
            if usesSliding {
                RotatingKVCache(maxSize: slidingWindow)
            } else {
                KVCacheSimple()
            }
        }
    }
}

// MARK: - LoRA Extension

extension AfMoEModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
