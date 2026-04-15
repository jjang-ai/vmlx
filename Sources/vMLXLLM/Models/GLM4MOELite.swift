//
//  GLM4MOELite.swift
//  LLM
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/glm4_moe_lite.py
//  Created by Ronald Mannak on 2025/1/7.
//

import Foundation
import MLX
import vMLXLMCommon
import MLXNN

// MARK: - MultiLinear

class MultiLinear: Module, Quantizable {
    let inputDims: Int
    let outputDims: Int
    let numHeads: Int

    @ParameterInfo(key: "weight") var weight: MLXArray

    init(inputDims: Int, outputDims: Int, numHeads: Int) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numHeads = numHeads

        let scale = sqrt(1.0 / Float(inputDims))
        _weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numHeads, outputDims, inputDims]
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return x.matmul(weight.swappedAxes(-1, -2))
    }

    // MARK: - Quantizable conformance

    public func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode) -> Module {
        return QuantizedMultiLinear(
            weight: weight,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }
}

// MARK: - QuantizedMultiLinear

/// Quantized version of MultiLinear that handles packed 4-bit weights.
/// This is the Swift equivalent of Python's QuantizedMultiLinear class (lines 89-129 in glm4_moe_lite.py).
class QuantizedMultiLinear: Module, Quantized {
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "scales") var scales: MLXArray
    @ParameterInfo(key: "biases") var biases: MLXArray?

    /// Initialize from non-quantized weights (for conversion from MultiLinear)
    init(
        weight: MLXArray,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            weight, groupSize: groupSize, bits: bits, mode: mode
        )
        _weight.wrappedValue = quantizedWeight
        _scales.wrappedValue = scales
        _biases.wrappedValue = biases

        super.init()
        self.freeze()
    }

    /// Initialize with pre-quantized weights and scales (for loading from file)
    init(
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        _weight.wrappedValue = weight
        _scales.wrappedValue = scales
        _biases.wrappedValue = biases

        super.init()
        self.freeze()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Use quantizedMM for efficient quantized matrix multiplication
        // The weight is in shape [numHeads, outputDims, inputDims(packed)]
        return quantizedMM(
            x,
            weight,
            scales: scales,
            biases: biases,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }
}

// MARK: - Attention

class GLM4MoELiteAttention: Module {
    let config: GLM4MoELiteConfiguration
    let hiddenSize: Int
    let numHeads: Int
    let maxPositionEmbeddings: Int
    let ropeTheta: Float
    let qLoraRank: Int?
    let qkRopeHeadDim: Int
    let kvLoraRank: Int
    let vHeadDim: Int
    let qkNopeHeadDim: Int
    let qHeadDim: Int
    var scale: Float

    let rope: RoPELayer
    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "embed_q") var embedQ: Module  // Can be MultiLinear or QuantizedMultiLinear
    @ModuleInfo(key: "unembed_out") var unembedOut: Module  // Can be MultiLinear or QuantizedMultiLinear

    init(_ config: GLM4MoELiteConfiguration) {
        self.config = config
        self.hiddenSize = config.hiddenSize
        self.numHeads = config.attentionHeads
        self.maxPositionEmbeddings = config.maxPositionEmbeddings
        self.ropeTheta = config.ropeTheta
        self.qLoraRank = config.qLoraRank
        self.qkRopeHeadDim = config.qkRopeHeadDim
        self.kvLoraRank = config.kvLoraRank
        self.vHeadDim = config.vHeadDim
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim
        self.scale = pow(Float(qHeadDim), -0.5)

        if let qLoraRank {
            _qAProj.wrappedValue = Linear(hiddenSize, qLoraRank, bias: config.attentionBias)
            _qALayerNorm.wrappedValue = RMSNorm(dimensions: qLoraRank, eps: config.rmsNormEps)
            _qBProj.wrappedValue = Linear(qLoraRank, numHeads * qHeadDim, bias: false)
        } else {
            _qProj.wrappedValue = Linear(hiddenSize, numHeads * qHeadDim, bias: false)
        }

        _kvAProjWithMqa.wrappedValue = Linear(
            hiddenSize,
            kvLoraRank + qkRopeHeadDim,
            bias: config.attentionBias
        )
        _kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank, eps: config.rmsNormEps)
        _embedQ.wrappedValue = MultiLinear(
            inputDims: qkNopeHeadDim,
            outputDims: kvLoraRank,
            numHeads: numHeads
        )
        _unembedOut.wrappedValue = MultiLinear(
            inputDims: kvLoraRank,
            outputDims: vHeadDim,
            numHeads: numHeads
        )
        _oProj.wrappedValue = Linear(
            numHeads * vHeadDim, hiddenSize, bias: config.attentionBias)

        if let ropeScaling = config.ropeScaling,
            let mscaleAllDim = ropeScaling["mscale_all_dim"]?.asFloat(),
            let scalingFactor = ropeScaling["factor"]?.asFloat(),
            mscaleAllDim != 0,
            scalingFactor > 1
        {
            let s = 0.1 * mscaleAllDim * log(scalingFactor) + 1.0
            self.scale = self.scale * s * s
        }

        var ropeScaling = config.ropeScaling
        if let ropeType = ropeScaling?["type"] ?? ropeScaling?["rope_type"],
            case .string(let value) = ropeType,
            value == "deepseek_yarn"
        {
            var updated = ropeScaling ?? [:]
            updated["type"] = .string("yarn")
            ropeScaling = updated
        }

        self.rope = initializeRope(
            dims: qkRopeHeadDim,
            base: ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: ropeScaling,
            maxPositionEmbeddings: maxPositionEmbeddings
        )
    }

    /// Helper to call a MultiLinear or QuantizedMultiLinear module
    private func callMultiLinear(_ module: Module, _ x: MLXArray) -> MLXArray {
        if let multiLinear = module as? MultiLinear {
            return multiLinear(x)
        } else if let quantized = module as? QuantizedMultiLinear {
            return quantized(x)
        } else {
            fatalError("Module must be MultiLinear or QuantizedMultiLinear")
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q: MLXArray
        if qLoraRank == nil {
            q = qProj!(x)
        } else {
            q = qBProj!(qALayerNorm!(qAProj!(x)))
        }

        q = q.reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let splitQ = split(q, indices: [qkNopeHeadDim], axis: -1)
        var qNope = splitQ[0]
        var qPe = splitQ[1]

        var compressedKv = kvAProjWithMqa(x)
        let splitCompressedKv = split(compressedKv, indices: [kvLoraRank], axis: -1)
        compressedKv = splitCompressedKv[0]
        var kPe = splitCompressedKv[1]
        kPe = kPe.reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)
        var kvLatent = kvALayerNorm(compressedKv)

        qPe = applyRotaryPosition(rope, to: qPe, cache: cache)
        kPe = applyRotaryPosition(rope, to: kPe, cache: cache)

        // Expand kvLatent for attention: [B, L, kvLoraRank] -> [B, 1, L, kvLoraRank]
        kvLatent = expandedDimensions(kvLatent, axis: 1)

        // Transform q_nope through embed_q
        qNope = callMultiLinear(embedQ, qNope)

        // Create keys for attention (and caching)
        var keys = concatenated([kvLatent, kPe], axis: -1)
        var values = kvLatent  // Values are the compressed KV latent

        // Update cache with compressed representation
        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        // Create queries
        let queries = concatenated([qNope, qPe], axis: -1)

        // Compute attention
        var output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )

        // Transform output through unembed_out
        output = callMultiLinear(unembedOut, output)

        output = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return oProj(output)
    }
}

class GLM4MoELiteMLP: Module, UnaryLayer {
    let hiddenSize: Int
    let intermediateSize: Int

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: GLM4MoELiteConfiguration, hiddenSize: Int? = nil, intermediateSize: Int? = nil) {
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

class GLM4MoELiteGate: Module {
    let topK: Int
    let normTopkProb: Bool
    let nRoutedExperts: Int
    let routedScalingFactor: Float
    let nGroup: Int
    let topkGroup: Int

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: GLM4MoELiteConfiguration) {
        guard let nRoutedExperts = config.nRoutedExperts else {
            fatalError("GLM4MoELiteGate requires nRoutedExperts")
        }

        precondition(config.topkMethod == "noaux_tc", "Unsupported topk method.")

        self.topK = config.numExpertsPerTok
        self.normTopkProb = config.normTopkProb
        self.nRoutedExperts = nRoutedExperts
        self.routedScalingFactor = config.routedScalingFactor
        self.nGroup = config.nGroup
        self.topkGroup = config.topkGroup

        _weight.wrappedValue = zeros([nRoutedExperts, config.hiddenSize])
        _eScoreCorrectionBias.wrappedValue = zeros([nRoutedExperts])

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let hiddenStates = x.matmul(weight.T)
        let originalScores = sigmoid(hiddenStates)
        var selectionScores = originalScores + eScoreCorrectionBias

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

class GLM4MoELiteMoE: Module, UnaryLayer {
    let numExpertsPerTok: Int
    let gate: GLM4MoELiteGate

    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: GLM4MoELiteMLP?

    /// Flash MoE Phase 2b shim.
    fileprivate var flashMoeShim: FlashMoEBlock? = nil

    init(_ config: GLM4MoELiteConfiguration) {
        guard let nRoutedExperts = config.nRoutedExperts else {
            fatalError("GLM4MoELiteMoE requires nRoutedExperts")
        }

        self.numExpertsPerTok = config.numExpertsPerTok
        self.gate = GLM4MoELiteGate(config)

        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: nRoutedExperts
        )

        if let shared = config.nSharedExperts, shared > 0 {
            let intermediateSize = config.moeIntermediateSize * shared
            _sharedExperts.wrappedValue = GLM4MoELiteMLP(
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

class GLM4MoELiteDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: GLM4MoELiteAttention
    let mlp: UnaryLayer

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: GLM4MoELiteConfiguration, layerIdx: Int) {
        _attention.wrappedValue = GLM4MoELiteAttention(config)

        if config.nRoutedExperts != nil,
            layerIdx >= config.firstKDenseReplace,
            layerIdx % config.moeLayerFreq == 0
        {
            self.mlp = GLM4MoELiteMoE(config)
        } else {
            self.mlp = GLM4MoELiteMLP(config)
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
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

public class GLM4MoELiteModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [GLM4MoELiteDecoderLayer]
    let norm: RMSNorm

    init(_ config: GLM4MoELiteConfiguration) {
        precondition(config.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)

        self.layers = (0 ..< config.hiddenLayers)
            .map { idx in
                GLM4MoELiteDecoderLayer(config, layerIdx: idx)
            }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
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

public class GLM4MoELiteModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: GLM4MoELiteModelInner
    let configuration: GLM4MoELiteConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ args: GLM4MoELiteConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = GLM4MoELiteModelInner(args)

        _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        return lmHead(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights

        for l in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(l)"

            // Stack experts
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

            // Convert kv_b_proj to embed_q and unembed_out
            let attnPrefix = "\(prefix).self_attn"
            if sanitized["\(attnPrefix).kv_b_proj.weight"] != nil {
                let isQuantized = sanitized["\(attnPrefix).kv_b_proj.scales"] != nil
                var v = sanitized.removeValue(forKey: "\(attnPrefix).kv_b_proj.weight")!
                let headDim = configuration.qkNopeHeadDim + configuration.vHeadDim

                var inferredBits = 0
                var inferredGroupSize = 0

                if isQuantized {
                    let dims = configuration.kvLoraRank
                    let scales = sanitized.removeValue(forKey: "\(attnPrefix).kv_b_proj.scales")!
                    let biases = sanitized.removeValue(forKey: "\(attnPrefix).kv_b_proj.biases")!
                    // Infer bits and group size
                    inferredBits = (v.dim(-1) * 32) / dims
                    inferredGroupSize = dims / scales.dim(-1)
                    v = dequantized(
                        v, scales: scales, biases: biases, groupSize: inferredGroupSize,
                        bits: inferredBits)
                }

                let numHeads = configuration.attentionHeads
                v = v.reshaped(numHeads, headDim, -1)
                var wk = v[0..., ..<configuration.qkNopeHeadDim, 0...].swappedAxes(-1, -2)
                var wv = v[0..., configuration.qkNopeHeadDim..., 0...]

                // Make contiguous
                wk = contiguous(wk)
                wv = contiguous(wv)

                if isQuantized {
                    let (qWk, qWkScales, qWkBiases) = MLX.quantized(
                        wk, groupSize: inferredGroupSize, bits: inferredBits)
                    let (qWv, qWvScales, qWvBiases) = MLX.quantized(
                        wv, groupSize: inferredGroupSize, bits: inferredBits)

                    sanitized["\(attnPrefix).embed_q.scales"] = qWkScales
                    sanitized["\(attnPrefix).unembed_out.scales"] = qWvScales
                    sanitized["\(attnPrefix).embed_q.biases"] = qWkBiases
                    sanitized["\(attnPrefix).unembed_out.biases"] = qWvBiases
                    wk = qWk
                    wv = qWv
                }

                sanitized["\(attnPrefix).embed_q.weight"] = wk
                sanitized["\(attnPrefix).unembed_out.weight"] = wv
            }
        }

        let numMptLayers = configuration.numNextnPredictLayers
        if numMptLayers > 0 {
            sanitized = sanitized.filter { key, _ in
                for idx in 0 ..< numMptLayers {
                    if key.hasPrefix("model.layers.\(configuration.hiddenLayers + idx)") {
                        return false
                    }
                }
                return true
            }
        }

        return sanitized
    }
}

public struct GLM4MoELiteConfiguration: Codable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var nSharedExperts: Int?
    var nRoutedExperts: Int?
    var routedScalingFactor: Float
    var kvLoraRank: Int
    var qLoraRank: Int?
    var qkRopeHeadDim: Int
    var qkNopeHeadDim: Int
    var vHeadDim: Int
    var topkMethod: String
    var scoringFunc: String
    var normTopkProb: Bool
    var nGroup: Int
    var topkGroup: Int
    var numExpertsPerTok: Int
    var moeLayerFreq: Int
    var firstKDenseReplace: Int
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var ropeTraditional: Bool
    var attentionBias: Bool
    var attentionDropout: Float
    var partialRotaryFactor: Float
    var tieWordEmbeddings: Bool
    var numNextnPredictLayers: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
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
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case vHeadDim = "v_head_dim"
        case topkMethod = "topk_method"
        case scoringFunc = "scoring_func"
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
        case ropeTraditional = "rope_traditional"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case partialRotaryFactor = "partial_rotary_factor"
        case tieWordEmbeddings = "tie_word_embeddings"
        case numNextnPredictLayers = "num_nextn_predict_layers"
    }

    public init(from decoder: Decoder) throws {
        let container: KeyedDecodingContainer<GLM4MoELiteConfiguration.CodingKeys> =
            try decoder.container(keyedBy: GLM4MoELiteConfiguration.CodingKeys.self)

        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.nSharedExperts = try container.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        self.nRoutedExperts = try container.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
        self.routedScalingFactor = try container.decode(Float.self, forKey: .routedScalingFactor)
        self.kvLoraRank = try container.decode(Int.self, forKey: .kvLoraRank)
        self.qLoraRank = try container.decodeIfPresent(Int.self, forKey: .qLoraRank)
        self.qkRopeHeadDim = try container.decode(Int.self, forKey: .qkRopeHeadDim)
        self.qkNopeHeadDim = try container.decode(Int.self, forKey: .qkNopeHeadDim)
        self.vHeadDim = try container.decode(Int.self, forKey: .vHeadDim)
        self.topkMethod =
            try container.decodeIfPresent(String.self, forKey: .topkMethod) ?? "noaux_tc"
        self.scoringFunc =
            try container.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "sigmoid"
        self.normTopkProb = try container.decode(Bool.self, forKey: .normTopkProb)
        self.nGroup = try container.decode(Int.self, forKey: .nGroup)
        self.topkGroup = try container.decode(Int.self, forKey: .topkGroup)
        self.numExpertsPerTok = try container.decode(Int.self, forKey: .numExpertsPerTok)
        self.moeLayerFreq = try container.decodeIfPresent(Int.self, forKey: .moeLayerFreq) ?? 1
        self.firstKDenseReplace = try container.decode(Int.self, forKey: .firstKDenseReplace)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? true
        self.attentionBias = try container.decode(Bool.self, forKey: .attentionBias)
        self.attentionDropout =
            try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        self.partialRotaryFactor = try container.decode(Float.self, forKey: .partialRotaryFactor)
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
            ?? false
        self.numNextnPredictLayers =
            try container.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers) ?? 1
    }
}

// MARK: - LoRA

extension GLM4MoELiteModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Flash MoE (Phase 2b)

extension GLM4MoELiteModel: FlashMoEReplaceable {
    public var flashMoELayers: [FlashMoELayer] {
        model.layers
    }
}

extension GLM4MoELiteDecoderLayer: FlashMoELayer {
    public var flashMoELayout: FlashMoELayout {
        (mlp is GLM4MoELiteMoE) ? .textPathSwitchGLU : .none
    }

    public func replaceMoEBlock(with block: FlashMoEBlock) throws {
        guard let moe = mlp as? GLM4MoELiteMoE else { return }
        let gate = moe.gate
        block.topK = moe.numExpertsPerTok
        block.router = { x in
            let out = gate(x)
            return (indices: out.0, scores: out.1)
        }
        moe.flashMoeShim = block
    }
}
