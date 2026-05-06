//
//  GLM4MoEJANGTQ.swift
//  vMLXLLM
//
//  JANGTQ (TurboQuant codebook) variant of GLM-4 MoE / GLM 5.1
//  (model_type: `glm4_moe`). Identical model structure to
//  `GLM4MOE.swift` but the routed-expert MoE projections run through
//  `TurboQuantSwitchGLU` instead of `SwitchGLU`.
//
//  What stays affine (NOT TQ-quantized):
//    - Shared experts (`shared_experts.{gate,up,down}_proj`) — affine
//    - First `first_k_dense_replace` layers' dense MLP — affine
//    - Attention projections (q/k/v/o) — affine GQA, no MLA
//    - Gate (`gate.weight` + `gate.e_score_correction_bias`) — affine
//      (sigmoid OR softmax, with `noaux_tc` group-routing)
//
//  Reuses internal classes from `GLM4MOE.swift`:
//    - `GLM4MoEAttention` (standard GQA — NOT MLA, so the deepseek_v32
//      L==1 SDPA bf16 absorb-bug fix does NOT apply here)
//    - `GLM4MoEMLP` (used by shared experts AND first-K dense layers)
//    - `GLM4MoEGate` (the noaux_tc grouped router)
//
//  Sanitize handles the `.tq_packed` / `.tq_norms` per-expert tensors
//  (stacks them into the `switch_mlp` layout `TurboQuantSwitchGLU`
//  expects) plus `.tq_bits` metadata stripping. Everything else
//  delegates to the affine sanitize logic mirrored inline.
//

import Foundation
import MLX
import MLXNN
import vMLXLMCommon

// MARK: - Configuration

public struct GLM4MoEJANGTQConfiguration: Codable, Sendable {
    // Mirrors every field from GLM4MoEConfiguration. Round-trip
    // through `asAffine()` to feed internal class initialisers.
    public var modelType: String
    public var vocabularySize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var maxPositionEmbeddings: Int
    public var moeIntermediateSize: Int
    public var normTopkProb: Bool
    public var attentionHeads: Int
    public var nGroup: Int
    public var headDim: Int
    public var topkGroup: Int
    public var nSharedExperts: Int?
    public var nRoutedExperts: Int?
    public var routedScalingFactor: Float
    public var numExpertsPerTok: Int
    public var firstKDenseReplace: Int
    public var hiddenLayers: Int
    public var kvHeads: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var ropeScaling: [String: StringOrNumber]?
    public var useQkNorm: Bool
    public var tieWordEmbeddings: Bool
    public var attentionBias: Bool
    public var partialRotaryFactor: Float
    public var scoringFunc: String = "sigmoid"
    public var topkMethod: String = "noaux_tc"

    // JANGTQ-specific
    public var weightFormat: String = "mxtq"
    public var mxtqBits: Int = 2
    public var mxtqSeed: Int = 42

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
        case weightFormat = "weight_format"
        case mxtqBits = "mxtq_bits"
        case mxtqSeed = "mxtq_seed"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
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
        self.weightFormat =
            try container.decodeIfPresent(String.self, forKey: .weightFormat) ?? "mxtq"
        // §346 T6 — accept `mxtq_bits` as flat Int OR per-role dict.
        if let flat = try? container.decodeIfPresent(Int.self, forKey: .mxtqBits) {
            self.mxtqBits = flat
        } else if let dict = try? container.decodeIfPresent(
            [String: Int].self, forKey: .mxtqBits),
            let routed = dict["routed_expert"] ?? dict["shared_expert"]
                ?? dict.values.first
        {
            self.mxtqBits = routed
        } else {
            self.mxtqBits = 2
        }
        self.mxtqSeed = try container.decodeIfPresent(Int.self, forKey: .mxtqSeed) ?? 42
    }

    /// Project to the affine `GLM4MoEConfiguration` shape that the
    /// internal class initialisers (`GLM4MoEAttention`, `GLM4MoEMLP`,
    /// `GLM4MoEGate`) accept. Round-trip via JSON to avoid touching
    /// every default if an affine field is added later.
    fileprivate func asAffine() -> GLM4MoEConfiguration {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        do {
            let data = try encoder.encode(GLM4MoEJANGTQAffineProjection(self))
            return try decoder.decode(GLM4MoEConfiguration.self, from: data)
        } catch {
            fatalError(
                "GLM4MoEJANGTQConfiguration.asAffine encode/decode failed: \(error)")
        }
    }
}

private struct GLM4MoEJANGTQAffineProjection: Encodable {
    let modelType: String
    let vocabularySize: Int
    let hiddenSize: Int
    let intermediateSize: Int
    let maxPositionEmbeddings: Int
    let moeIntermediateSize: Int
    let normTopkProb: Bool
    let attentionHeads: Int
    let nGroup: Int
    let headDim: Int
    let topkGroup: Int
    let nSharedExperts: Int?
    let nRoutedExperts: Int?
    let routedScalingFactor: Float
    let numExpertsPerTok: Int
    let firstKDenseReplace: Int
    let hiddenLayers: Int
    let kvHeads: Int
    let rmsNormEps: Float
    let ropeTheta: Float
    let ropeScaling: [String: StringOrNumber]?
    let useQkNorm: Bool
    let tieWordEmbeddings: Bool
    let attentionBias: Bool
    let partialRotaryFactor: Float
    let scoringFunc: String
    let topkMethod: String

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

    init(_ src: GLM4MoEJANGTQConfiguration) {
        self.modelType = src.modelType
        self.vocabularySize = src.vocabularySize
        self.hiddenSize = src.hiddenSize
        self.intermediateSize = src.intermediateSize
        self.maxPositionEmbeddings = src.maxPositionEmbeddings
        self.moeIntermediateSize = src.moeIntermediateSize
        self.normTopkProb = src.normTopkProb
        self.attentionHeads = src.attentionHeads
        self.nGroup = src.nGroup
        self.headDim = src.headDim
        self.topkGroup = src.topkGroup
        self.nSharedExperts = src.nSharedExperts
        self.nRoutedExperts = src.nRoutedExperts
        self.routedScalingFactor = src.routedScalingFactor
        self.numExpertsPerTok = src.numExpertsPerTok
        self.firstKDenseReplace = src.firstKDenseReplace
        self.hiddenLayers = src.hiddenLayers
        self.kvHeads = src.kvHeads
        self.rmsNormEps = src.rmsNormEps
        self.ropeTheta = src.ropeTheta
        self.ropeScaling = src.ropeScaling
        self.useQkNorm = src.useQkNorm
        self.tieWordEmbeddings = src.tieWordEmbeddings
        self.attentionBias = src.attentionBias
        self.partialRotaryFactor = src.partialRotaryFactor
        self.scoringFunc = src.scoringFunc
        self.topkMethod = src.topkMethod
    }
}

// MARK: - JANGTQ MoE block

/// Routed-MoE block for GLM-4 MoE / GLM 5.1, JANGTQ variant. Mirrors
/// `GLM4MoE` (gate → topk → switch_mlp → optional shared_experts) but
/// the per-expert projections run through `TurboQuantSwitchGLU`.
final class GLM4MoEJANGTQ: Module, UnaryLayer {
    let numExpertsPerTok: Int
    let gate: GLM4MoEGate
    fileprivate let layerIdx: Int

    @ModuleInfo(key: "switch_mlp") var switchMLP: TurboQuantSwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: GLM4MoEMLP?

    init(_ config: GLM4MoEJANGTQConfiguration, layerIdx: Int) {
        guard let nRoutedExperts = config.nRoutedExperts else {
            fatalError("GLM4MoEJANGTQ requires nRoutedExperts")
        }

        self.numExpertsPerTok = config.numExpertsPerTok
        self.gate = GLM4MoEGate(config.asAffine())
        self.layerIdx = layerIdx

        _switchMLP.wrappedValue = TurboQuantSwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: nRoutedExperts,
            bits: config.mxtqBits,
            seed: config.mxtqSeed
        )

        if let shared = config.nSharedExperts, shared > 0 {
            let intermediateSize = config.moeIntermediateSize * shared
            _sharedExperts.wrappedValue = GLM4MoEMLP(
                config.asAffine(), intermediateSize: intermediateSize)
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        JangPressRouteTelemetry.recordTopK(layer: layerIdx, indices: inds)
        var y = switchMLP(x, inds)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

// MARK: - JANGTQ Decoder Layer

final class GLM4MoEJANGTQDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: GLM4MoEAttention
    let mlp: UnaryLayer

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: GLM4MoEJANGTQConfiguration, layerIdx: Int) {
        let affine = args.asAffine()
        _attention.wrappedValue = GLM4MoEAttention(affine)

        if args.nRoutedExperts != nil && layerIdx >= args.firstKDenseReplace {
            self.mlp = GLM4MoEJANGTQ(args, layerIdx: layerIdx)
        } else {
            // First firstKDenseReplace layers are dense — stay affine.
            self.mlp = GLM4MoEMLP(affine)
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

// MARK: - Inner model

public class GLM4MoEJANGTQModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [GLM4MoEJANGTQDecoderLayer]
    let norm: RMSNorm

    init(_ args: GLM4MoEJANGTQConfiguration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers).map { idx in
            GLM4MoEJANGTQDecoderLayer(args, layerIdx: idx)
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

// MARK: - Top-level model

public class GLM4MoEJANGTQModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: GLM4MoEJANGTQModelInner
    let configuration: GLM4MoEJANGTQConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: GLM4MoEJANGTQConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = GLM4MoEJANGTQModelInner(args)

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

    /// Sanitize for the JANGTQ wire format. Three jobs (matches the
    /// MiniMaxJANGTQ / Qwen35JANGTQ pattern):
    ///   1. Drop tied lm_head when configured.
    ///   2. Strip `.tq_bits` metadata tensors anywhere in the tree.
    ///   3. Stack per-expert `experts.{E}.{gate,up,down}_proj.{tq_packed,tq_norms}`
    ///      into the `mlp.switch_mlp.{...}` layout `TurboQuantSwitchGLU`
    ///      consumes. Mirrors the affine sanitize in `GLM4MoEModel`
    ///      which does the same for `weight/scales/biases`.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights

        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }

        for key in Array(sanitized.keys) where key.hasSuffix(".tq_bits") {
            sanitized[key] = nil
        }

        guard let nRoutedExperts = configuration.nRoutedExperts else {
            return sanitized
        }

        for layer in 0 ..< configuration.hiddenLayers {
            // Layers below firstKDenseReplace are dense — no MoE
            // tensors to stack. Skip.
            if layer < configuration.firstKDenseReplace { continue }
            let prefix = "model.layers.\(layer).mlp"
            for proj in ["gate_proj", "down_proj", "up_proj"] {
                for kind in ["tq_packed", "tq_norms"] {
                    let probeKey = "\(prefix).experts.0.\(proj).\(kind)"
                    guard sanitized[probeKey] != nil else { continue }
                    let stacked: [MLXArray] = (0 ..< nRoutedExperts).map { e in
                        sanitized.removeValue(
                            forKey: "\(prefix).experts.\(e).\(proj).\(kind)")!
                    }
                    sanitized["\(prefix).switch_mlp.\(proj).\(kind)"] =
                        loadTimeMaterializedStacked(stacked)
                }
            }
        }

        return sanitized
    }
}

// MARK: - LoRA

extension GLM4MoEJANGTQModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
