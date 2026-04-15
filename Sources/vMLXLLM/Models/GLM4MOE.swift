//
//  GLM4MOE.swift
//  LLM
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/glm4_moe.py
//  Created by Ronald Mannak on 2025/1/7.
//

import Foundation
import MLX
import vMLXLMCommon
import MLXNN

class GLM4MoEAttention: Module {
    let args: GLM4MoEConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    let rope: RoPELayer

    init(_ args: GLM4MoEConfiguration) {
        self.args = args

        let headDim = args.headDim > 0 ? args.headDim : args.hiddenSize / args.attentionHeads
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(
            args.hiddenSize, args.attentionHeads * headDim, bias: args.attentionBias)
        _wk.wrappedValue = Linear(args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _wv.wrappedValue = Linear(args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _wo.wrappedValue = Linear(args.attentionHeads * headDim, args.hiddenSize, bias: false)

        if args.useQkNorm {
            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        }

        self.rope = initializeRope(
            dims: Int(Float(headDim) * args.partialRotaryFactor),
            base: args.ropeTheta,
            traditional: false, scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(B, L, args.attentionHeads, -1)
        keys = keys.reshaped(B, L, args.kvHeads, -1)

        if let qNorm, let kNorm {
            queries = qNorm(queries)
            keys = kNorm(keys)
        }

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

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

        return wo(output)
    }
}

class GLM4MoEMLP: Module, UnaryLayer {
    let hiddenSize: Int
    let intermediateSize: Int

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: GLM4MoEConfiguration, hiddenSize: Int? = nil, intermediateSize: Int? = nil) {
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

class GLM4MoEGate: Module {
    let topK: Int
    let normTopkProb: Bool
    let nRoutedExperts: Int
    let routedScalingFactor: Float
    let nGroup: Int
    let topkGroup: Int
    let scoringFunc: String

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: GLM4MoEConfiguration) {
        guard let nRoutedExperts = config.nRoutedExperts else {
            fatalError("GLM4MoEGate requires nRoutedExperts")
        }

        precondition(config.topkMethod == "noaux_tc", "Unsupported topk method.")

        self.topK = config.numExpertsPerTok
        self.normTopkProb = config.normTopkProb
        self.nRoutedExperts = nRoutedExperts
        self.routedScalingFactor = config.routedScalingFactor
        self.nGroup = config.nGroup
        self.topkGroup = config.topkGroup
        self.scoringFunc = config.scoringFunc

        _weight.wrappedValue = zeros([nRoutedExperts, config.hiddenSize])
        _eScoreCorrectionBias.wrappedValue = zeros([nRoutedExperts])

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let hiddenStates = x.matmul(weight.T)
        var scores: MLXArray
        if scoringFunc == "sigmoid" {
            scores = sigmoid(hiddenStates)
        } else {
            scores = softmax(hiddenStates, axis: -1, precise: true)
        }

        let originalScores = scores
        var selectionScores = scores + eScoreCorrectionBias

        if nGroup > 1 {
            selectionScores = unflatten(selectionScores, axis: -1, shape: [nGroup, -1])
            let groupScores = top(selectionScores, k: 2, axis: -1).sum(axis: -1, keepDims: true)
            let k = nGroup - topkGroup
            let groupIdx = argPartition(groupScores, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
            selectionScores = putAlong(
                selectionScores, stopGradient(groupIdx), values: MLXArray(0.0, dtype: selectionScores.dtype), axis: -2)
            selectionScores = flattened(selectionScores, start: -2, end: -1)
        }

        let k = topK
        let inds = argPartition(-selectionScores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        var selectedScores = takeAlong(originalScores, inds, axis: -1)

        if topK > 1, normTopkProb {
            let denominator = selectedScores.sum(axis: -1, keepDims: true)
            selectedScores = selectedScores / denominator
        }
        selectedScores = selectedScores * routedScalingFactor

        return (inds, selectedScores)
    }
}

class GLM4MoE: Module, UnaryLayer {
    let numExpertsPerTok: Int
    let gate: GLM4MoEGate

    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: GLM4MoEMLP?

    /// Flash MoE Phase 2b shim. Replaces the routed-experts matmul
    /// with a slot-bank streamer; `sharedExperts` still runs natively.
    fileprivate var flashMoeShim: FlashMoEBlock? = nil

    init(_ config: GLM4MoEConfiguration) {
        guard let nRoutedExperts = config.nRoutedExperts else {
            fatalError("GLM4MoE requires nRoutedExperts")
        }

        self.numExpertsPerTok = config.numExpertsPerTok
        self.gate = GLM4MoEGate(config)

        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: nRoutedExperts
        )

        if let shared = config.nSharedExperts, shared > 0 {
            let intermediateSize = config.moeIntermediateSize * shared
            _sharedExperts.wrappedValue = GLM4MoEMLP(
                config, intermediateSize: intermediateSize
            )
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y: MLXArray
        if let shim = flashMoeShim {
            y = shim(x)
        } else {
            let (inds, scores) = gate(x)
            y = switchMLP(x, inds)
            y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
        }
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

class GLM4MoEDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: GLM4MoEAttention
    let mlp: UnaryLayer

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: GLM4MoEConfiguration, layerIdx: Int) {
        _attention.wrappedValue = GLM4MoEAttention(args)

        if args.nRoutedExperts != nil && layerIdx >= args.firstKDenseReplace {
            self.mlp = GLM4MoE(args)
        } else {
            self.mlp = GLM4MoEMLP(args)
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        let r2 = mlp(postAttentionLayerNorm(h))
        return h + r2
    }
}

public class GLM4MoEModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [GLM4MoEDecoderLayer]
    let norm: RMSNorm

    init(_ args: GLM4MoEConfiguration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers)
            .map { idx in
                GLM4MoEDecoderLayer(args, layerIdx: idx)
            }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class GLM4MoEModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: GLM4MoEModelInner
    let configuration: GLM4MoEConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: GLM4MoEConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = GLM4MoEModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights

        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }

        for l in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(l)"
            for n in ["gate_proj", "down_proj", "up_proj"] {
                for k in ["weight", "scales", "biases"] {
                    let key = "\(prefix).mlp.experts.0.\(n).\(k)"
                    if sanitized[key] != nil, let nRoutedExperts = configuration.nRoutedExperts {
                        let toJoin = (0 ..< nRoutedExperts).map { e in
                            sanitized.removeValue(
                                forKey: "\(prefix).mlp.experts.\(e).\(n).\(k)")!
                        }
                        sanitized["\(prefix).mlp.switch_mlp.\(n).\(k)"] = MLX.stacked(toJoin)
                    }
                }
            }
        }

        let mptLayerPrefix = "model.layers.\(configuration.hiddenLayers)"
        sanitized = sanitized.filter { !($0.key.hasPrefix(mptLayerPrefix)) }

        return sanitized
    }
}

public struct GLM4MoEConfiguration: Codable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var maxPositionEmbeddings: Int
    var moeIntermediateSize: Int
    var normTopkProb: Bool
    var attentionHeads: Int
    var nGroup: Int
    var headDim: Int
    var topkGroup: Int
    var nSharedExperts: Int?
    var nRoutedExperts: Int?
    var routedScalingFactor: Float
    var numExpertsPerTok: Int
    var firstKDenseReplace: Int
    var hiddenLayers: Int
    var kvHeads: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var useQkNorm: Bool
    var tieWordEmbeddings: Bool
    var attentionBias: Bool
    var partialRotaryFactor: Float
    var scoringFunc: String = "sigmoid"
    var topkMethod: String = "noaux_tc"

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case moeIntermediateSize = "moe_intermediate_size"
        case normTopkProb = "norm_topk_prob"
        case attentionHeads = "num_attention_heads"
        case nGroup = "n_group"
        case headDim = "head_dim"
        case topkGroup = "topk_group"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case numExpertsPerTok = "num_experts_per_tok"
        case firstKDenseReplace = "first_k_dense_replace"
        case hiddenLayers = "num_hidden_layers"
        case kvHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case useQkNorm = "use_qk_norm"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case partialRotaryFactor = "partial_rotary_factor"
        case scoringFunc = "scoring_func"
        case topkMethod = "topk_method"
    }

    public init(from decoder: Decoder) throws {
        let container: KeyedDecodingContainer<GLM4MoEConfiguration.CodingKeys> =
            try decoder.container(keyedBy: GLM4MoEConfiguration.CodingKeys.self)

        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        self.normTopkProb = try container.decode(Bool.self, forKey: .normTopkProb)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.nGroup = try container.decode(Int.self, forKey: .nGroup)
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.topkGroup = try container.decode(Int.self, forKey: .topkGroup)
        self.nSharedExperts = try container.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        self.nRoutedExperts = try container.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
        self.routedScalingFactor = try container.decode(Float.self, forKey: .routedScalingFactor)
        self.numExpertsPerTok = try container.decode(Int.self, forKey: .numExpertsPerTok)
        self.firstKDenseReplace = try container.decode(Int.self, forKey: .firstKDenseReplace)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.useQkNorm = try container.decode(Bool.self, forKey: .useQkNorm)
        self.tieWordEmbeddings = try container.decode(Bool.self, forKey: .tieWordEmbeddings)
        self.attentionBias = try container.decode(Bool.self, forKey: .attentionBias)
        self.partialRotaryFactor = try container.decode(Float.self, forKey: .partialRotaryFactor)
        self.scoringFunc =
            try container.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "sigmoid"
        self.topkMethod =
            try container.decodeIfPresent(String.self, forKey: .topkMethod) ?? "noaux_tc"
    }
}

// MARK: - LoRA

extension GLM4MoEModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Flash MoE (Phase 2b)
//
// GLM4MoE has a grouped-top-K gate (`GLM4MoEGate.nGroup > 1`) plus
// scaling + optional norm, all of which are preserved verbatim by
// capturing the gate reference and letting its existing
// `callAsFunction` do the routing math. Shared experts remain in
// place via the `flashMoeShim` branch inside `GLM4MoE`.

extension GLM4MoEModel: FlashMoEReplaceable {
    public var flashMoELayers: [FlashMoELayer] {
        model.layers
    }
}

extension GLM4MoEDecoderLayer: FlashMoELayer {
    public var flashMoELayout: FlashMoELayout {
        (mlp is GLM4MoE) ? .textPathSwitchGLU : .none
    }

    public func replaceMoEBlock(with block: FlashMoEBlock) throws {
        guard let moe = mlp as? GLM4MoE else { return }
        let gate = moe.gate
        block.topK = moe.numExpertsPerTok
        block.router = { x in
            let out = gate(x)
            return (indices: out.0, scores: out.1)
        }
        moe.flashMoeShim = block
    }
}
