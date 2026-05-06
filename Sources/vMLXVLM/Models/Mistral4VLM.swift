// Mistral Small 4 VLM — Pixtral vision encoder + MLA text decoder
//
// Reuses PixtralVision from Pixtral.swift and projector from Mistral3.swift.
// Text decoder is MLA (Multi-head Latent Attention) with MoE + shared experts,
// matching the vMLXLLM Mistral4 text model architecture.
//
// Weight prefix: language_model.model.layers.* (text), vision_tower.* (vision)

import Foundation
import MLX
import vMLXLMCommon
import MLXNN

// MARK: - Configuration

/// Mistral4 VLM top-level configuration (wraps text + vision)
public struct Mistral4VLMConfiguration: Codable, Sendable {
    public let textConfig: Mistral3VLMTextConfiguration  // Reuses Mistral3's text config structure
    public let visionConfig: Mistral3VisionConfiguration  // Same Pixtral vision
    public let modelType: String

    public var imageTokenIndex: Int { _imageTokenIndex ?? _imageTokenId ?? 10 }
    public var visionFeatureLayer: Int { _visionFeatureLayer ?? -1 }
    public var vocabSize: Int { _vocabSize ?? 131072 }
    public var spatialMergeSize: Int { _spatialMergeSize ?? 2 }
    public var multimodalProjectorBias: Bool { _multimodalProjectorBias ?? false }

    // MLA-specific fields decoded from text_config
    public var kvLoraRank: Int { _kvLoraRank ?? 256 }
    public var qLoraRank: Int { _qLoraRank ?? 1024 }
    public var qkRopeHeadDim: Int { _qkRopeHeadDim ?? 64 }
    public var vHeadDim: Int { _vHeadDim ?? 128 }
    public var qkNopeHeadDim: Int { _qkNopeHeadDim ?? 64 }
    public var nRoutedExperts: Int { _nRoutedExperts ?? 128 }
    public var numExpertsPerTok: Int { _numExpertsPerTok ?? 4 }
    public var nSharedExperts: Int { _nSharedExperts ?? 1 }
    public var moeIntermediateSize: Int { _moeIntermediateSize ?? 2048 }
    public var routedScalingFactor: Float { _routedScalingFactor ?? 1.0 }
    public var normTopkProb: Bool { _normTopkProb ?? true }
    public var ropeInterleave: Bool { _ropeInterleave ?? false }

    private let _imageTokenIndex: Int?
    private let _imageTokenId: Int?
    private let _visionFeatureLayer: Int?
    private let _vocabSize: Int?
    private let _spatialMergeSize: Int?
    private let _multimodalProjectorBias: Bool?

    // MLA fields from text_config (decoded via custom init)
    private let _kvLoraRank: Int?
    private let _qLoraRank: Int?
    private let _qkRopeHeadDim: Int?
    private let _vHeadDim: Int?
    private let _qkNopeHeadDim: Int?
    private let _nRoutedExperts: Int?
    private let _numExpertsPerTok: Int?
    private let _nSharedExperts: Int?
    private let _moeIntermediateSize: Int?
    private let _routedScalingFactor: Float?
    private let _normTopkProb: Bool?
    private let _ropeInterleave: Bool?

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case modelType = "model_type"
        case _imageTokenIndex = "image_token_index"
        case _imageTokenId = "image_token_id"
        case _visionFeatureLayer = "vision_feature_layer"
        case _vocabSize = "vocab_size"
        case _spatialMergeSize = "spatial_merge_size"
        case _multimodalProjectorBias = "multimodal_projector_bias"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfig = try c.decode(Mistral3VLMTextConfiguration.self, forKey: .textConfig)
        visionConfig = try c.decode(Mistral3VisionConfiguration.self, forKey: .visionConfig)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "mistral3"
        _imageTokenIndex = try c.decodeIfPresent(Int.self, forKey: ._imageTokenIndex)
        _imageTokenId = try c.decodeIfPresent(Int.self, forKey: ._imageTokenId)
        _visionFeatureLayer = try c.decodeIfPresent(Int.self, forKey: ._visionFeatureLayer)
        _vocabSize = try c.decodeIfPresent(Int.self, forKey: ._vocabSize)
        _spatialMergeSize = try c.decodeIfPresent(Int.self, forKey: ._spatialMergeSize)
        _multimodalProjectorBias = try c.decodeIfPresent(Bool.self, forKey: ._multimodalProjectorBias)

        // Extract MLA fields directly from text_config nested container
        struct MLACodingKeys: CodingKey {
            var stringValue: String
            init(stringValue: String) { self.stringValue = stringValue }
            var intValue: Int? { nil }
            init?(intValue: Int) { return nil }
        }
        let textContainer = try c.nestedContainer(keyedBy: MLACodingKeys.self, forKey: .textConfig)
        _kvLoraRank = try? textContainer.decodeIfPresent(Int.self, forKey: MLACodingKeys(stringValue: "kv_lora_rank"))
        _qLoraRank = try? textContainer.decodeIfPresent(Int.self, forKey: MLACodingKeys(stringValue: "q_lora_rank"))
        _qkRopeHeadDim = try? textContainer.decodeIfPresent(Int.self, forKey: MLACodingKeys(stringValue: "qk_rope_head_dim"))
        _vHeadDim = try? textContainer.decodeIfPresent(Int.self, forKey: MLACodingKeys(stringValue: "v_head_dim"))
        _qkNopeHeadDim = try? textContainer.decodeIfPresent(Int.self, forKey: MLACodingKeys(stringValue: "qk_nope_head_dim"))
        _nRoutedExperts = try? textContainer.decodeIfPresent(Int.self, forKey: MLACodingKeys(stringValue: "n_routed_experts"))
        _numExpertsPerTok = try? textContainer.decodeIfPresent(Int.self, forKey: MLACodingKeys(stringValue: "num_experts_per_tok"))
        _nSharedExperts = try? textContainer.decodeIfPresent(Int.self, forKey: MLACodingKeys(stringValue: "n_shared_experts"))
        _moeIntermediateSize = try? textContainer.decodeIfPresent(Int.self, forKey: MLACodingKeys(stringValue: "moe_intermediate_size"))
        _routedScalingFactor = try? textContainer.decodeIfPresent(Float.self, forKey: MLACodingKeys(stringValue: "routed_scaling_factor"))
        _normTopkProb = try? textContainer.decodeIfPresent(Bool.self, forKey: MLACodingKeys(stringValue: "norm_topk_prob"))
        _ropeInterleave = try? textContainer.decodeIfPresent(Bool.self, forKey: MLACodingKeys(stringValue: "rope_interleave"))
    }
}

// MARK: - MLA Text Decoder (inline, since vMLXVLM can't import vMLXLLM)

// MLA Attention for Mistral 4
private class M4Attention: Module {
    let numHeads: Int
    let qLoraRank: Int
    let qkRopeHeadDim: Int
    let kvLoraRank: Int
    let vHeadDim: Int
    let qkNopeHeadDim: Int
    let qHeadDim: Int
    let scale: Float

    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    let rope: RoPELayer

    init(_ config: Mistral4VLMConfiguration) {
        let tc = config.textConfig
        numHeads = tc.numAttentionHeads
        qLoraRank = config.qLoraRank
        qkRopeHeadDim = config.qkRopeHeadDim
        kvLoraRank = config.kvLoraRank
        vHeadDim = config.vHeadDim
        qkNopeHeadDim = config.qkNopeHeadDim
        qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim

        scale = pow(Float(qHeadDim), -0.5)

        _qAProj.wrappedValue = Linear(tc.hiddenSize, qLoraRank, bias: false)
        _qALayerNorm.wrappedValue = RMSNorm(dimensions: qLoraRank, eps: tc.rmsNormEps)
        _qBProj.wrappedValue = Linear(qLoraRank, numHeads * qHeadDim, bias: false)
        _kvAProjWithMqa.wrappedValue = Linear(tc.hiddenSize, kvLoraRank + qkRopeHeadDim, bias: false)
        _kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank, eps: tc.rmsNormEps)
        _kvBProj.wrappedValue = Linear(kvLoraRank, numHeads * (qkNopeHeadDim + vHeadDim), bias: false)
        _oProj.wrappedValue = Linear(numHeads * vHeadDim, tc.hiddenSize, bias: false)

        let ropeParams = tc.ropeParameters
        let ropeTheta = ropeParams?["rope_theta"]?.asFloat() ?? tc.ropeTheta
        rope = initializeRope(
            dims: qkRopeHeadDim, base: ropeTheta, traditional: config.ropeInterleave,
            scalingConfig: ropeParams, maxPositionEmbeddings: tc.maxPositionEmbeddings)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var q = qBProj(qALayerNorm(qAProj(x)))
        q = q.reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let qSplit = split(q, indices: [qkNopeHeadDim], axis: -1)
        let qNope = qSplit[0]; var qPe = qSplit[1]

        var compKV = kvAProjWithMqa(x)
        let kvSplit = split(compKV, indices: [kvLoraRank], axis: -1)
        compKV = kvSplit[0]
        var kPe = kvSplit[1].reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)

        var kv = kvBProj(kvALayerNorm(compKV))
        kv = kv.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        let kvDSplit = split(kv, indices: [qkNopeHeadDim], axis: -1)
        let kNope = kvDSplit[0]; var values = kvDSplit[1]

        qPe = applyRotaryPosition(rope, to: qPe, cache: cache)
        kPe = applyRotaryPosition(rope, to: kPe, cache: cache)
        kPe = repeated(kPe, count: numHeads, axis: 1)

        var keys = concatenated([kNope, kPe], axis: -1)
        let queries = concatenated([qNope, qPe], axis: -1)

        if let cache {
            let (ck, cv) = cache.update(keys: keys, values: values)
            keys = ck; values = cv
        }

        // Iter 143 — MLA fp32 SDPA cast on L==1 (decode step).
        let output = mlaScaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

private class M4MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gP: Linear
    @ModuleInfo(key: "up_proj") var uP: Linear
    @ModuleInfo(key: "down_proj") var dP: Linear
    init(hidden: Int, inter: Int) {
        _gP.wrappedValue = Linear(hidden, inter, bias: false)
        _uP.wrappedValue = Linear(hidden, inter, bias: false)
        _dP.wrappedValue = Linear(inter, hidden, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { dP(silu(gP(x)) * uP(x)) }
}

private class M4MoEGate: Module {
    let topK: Int; let nExperts: Int; let scalingFactor: Float; let normTopk: Bool
    @ModuleInfo var weight: MLXArray
    init(_ config: Mistral4VLMConfiguration) {
        topK = config.numExpertsPerTok; nExperts = config.nRoutedExperts
        scalingFactor = config.routedScalingFactor; normTopk = config.normTopkProb
        _weight.wrappedValue = MLXArray.zeros([config.nRoutedExperts, config.textConfig.hiddenSize])
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let scores = softmax(matmul(x, weight.transposed()), axis: -1, precise: true)
        // See SWIFT-NO-REGRESSION-CHECKLIST §27 — unary `-` avoids the
        // per-step AsType + broadcast paid by the scalar-subtract form.
        let inds = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var wts = takeAlong(scores, inds, axis: -1)
        if normTopk { wts = wts / wts.sum(axis: -1, keepDims: true) }
        wts = wts * scalingFactor
        return (inds, wts)
    }
}

private class M4MoE: Module, UnaryLayer {
    let layerIndex: Int
    @ModuleInfo var gate: M4MoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: M4MLP?
    init(_ config: Mistral4VLMConfiguration, layerIndex: Int) {
        self.layerIndex = layerIndex
        gate = M4MoEGate(config)
        _switchMLP.wrappedValue = SwitchGLU(inputDims: config.textConfig.hiddenSize, hiddenDims: config.moeIntermediateSize, numExperts: config.nRoutedExperts, bias: false)
        if config.nSharedExperts > 0 {
            _sharedExperts.wrappedValue = M4MLP(hidden: config.textConfig.hiddenSize, inter: config.moeIntermediateSize * config.nSharedExperts)
        }
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        JangPressRouteTelemetry.recordTopK(layer: layerIndex, indices: inds)
        var y = switchMLP(x, inds)
        y = (y * expandedDimensions(scores, axis: -1)).sum(axis: -2)
        if let se = sharedExperts { y = y + se(x) }
        return y
    }
}

private class M4Layer: Module {
    @ModuleInfo(key: "self_attn") var attn: M4Attention
    @ModuleInfo var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var iLN: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var paLN: RMSNorm
    init(_ config: Mistral4VLMConfiguration, idx: Int) {
        _attn.wrappedValue = M4Attention(config)
        let isMoE = config.nRoutedExperts > 0
        if isMoE { _mlp.wrappedValue = M4MoE(config, layerIndex: idx) }
        else { _mlp.wrappedValue = M4MLP(hidden: config.textConfig.hiddenSize, inter: config.textConfig.intermediateSize) }
        _iLN.wrappedValue = RMSNorm(dimensions: config.textConfig.hiddenSize, eps: config.textConfig.rmsNormEps)
        _paLN.wrappedValue = RMSNorm(dimensions: config.textConfig.hiddenSize, eps: config.textConfig.rmsNormEps)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        let h = x + attn(iLN(x), mask: mask, cache: cache)
        return h + mlp(paLN(h))
    }
}

private class M4TextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [M4Layer]
    @ModuleInfo var norm: RMSNorm
    let config: Mistral4VLMConfiguration

    init(_ config: Mistral4VLMConfiguration) {
        self.config = config
        let tc = config.textConfig
        _embedTokens.wrappedValue = Embedding(embeddingCount: tc.vocabSize, dimensions: tc.hiddenSize)
        _layers.wrappedValue = (0 ..< tc.numHiddenLayers).map { M4Layer(config, idx: $0) }
        self.norm = RMSNorm(dimensions: tc.hiddenSize, eps: tc.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray?, inputsEmbeds: MLXArray? = nil, cache: [KVCache?]? = nil) -> MLXArray {
        var h = inputsEmbeds ?? embedTokens(inputs!)
        let lc = cache ?? Array(repeating: nil as KVCache?, count: layers.count)
        let mask = makeAttentionMask(n: h.dim(1), cache: lc.first ?? nil)
        for (i, l) in layers.enumerated() { h = l(h, mask: mask, cache: lc[i]) }
        return norm(h)
    }
}

private class M4LanguageModel: Module, KVCacheDimensionProvider {
    @ModuleInfo(key: "model") var model: M4TextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear?
    let config: Mistral4VLMConfiguration
    var kvHeads: [Int] { Array(repeating: config.textConfig.numKeyValueHeads, count: config.textConfig.numHiddenLayers) }

    var embedTokens: Embedding { model.embedTokens }
    var layers: [M4Layer] { model.layers }

    init(_ config: Mistral4VLMConfiguration) {
        self.config = config
        _model.wrappedValue = M4TextModel(config)
        if !config.textConfig.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.textConfig.hiddenSize, config.textConfig.vocabSize, bias: false)
        }
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray?, cache: [KVCache]?, inputsEmbeds: MLXArray? = nil) -> MLXArray {
        var out = model(inputs, inputsEmbeds: inputsEmbeds, cache: cache)
        if config.textConfig.tieWordEmbeddings { out = embedTokens.asLinear(out) }
        else if let lh = lmHead { out = lh(out) }
        return out
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< config.textConfig.numHiddenLayers).map { _ in
            if let m = parameters?.maxKVSize { return RotatingKVCache(maxSize: m, keep: 4) }
            return KVCacheSimple()
        }
    }
}

// MARK: - Mistral4 VLM Model

public class Mistral4VLM: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: PixtralVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: M4LanguageModel
    @ModuleInfo(key: "multi_modal_projector") private var multiModalProjector: Mistral3MultiModalProjector

    public let config: Mistral4VLMConfiguration

    public var vocabularySize: Int { config.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public init(_ config: Mistral4VLMConfiguration) {
        self.config = config
        _visionTower.wrappedValue = PixtralVision.VisionModel(config.visionConfig)
        _languageModel.wrappedValue = M4LanguageModel(config)
        _multiModalProjector.wrappedValue = Mistral3MultiModalProjector(
            Mistral3VLMConfiguration(from: config))
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        let inputIds = input.text.tokens
        let pixelValues = input.image?.pixels

        let imageSizes: [(Int, Int)]?
        if let frames = input.image?.frames {
            imageSizes = frames.map { ($0.h, $0.w) }
        } else if pixelValues != nil {
            imageSizes = [(config.visionConfig.imageSize, config.visionConfig.imageSize)]
        } else {
            imageSizes = nil
        }

        let embeddings = getInputEmbeddings(inputIds: inputIds, pixelValues: pixelValues, imageSizes: imageSizes)
        // Chunked prefill — see Qwen25VL / vmlx #50/#51. Inner M4TextModel uses
        // `inputsEmbeds ?? embedTokens(inputs!)`, so `inputs` is ignored when
        // embeds is non-nil — safe to pass nil per chunk.
        let logits = chunkedPrefillEmbedding(
            inputEmbedding: embeddings,
            cache: cache,
            prefillStepSize: windowSize ?? 512
        ) { chunk in
            languageModel(nil, cache: cache, inputsEmbeds: chunk)
        }
        return .logits(.init(logits: logits))
    }

    private func getInputEmbeddings(inputIds: MLXArray?, pixelValues: MLXArray?, imageSizes: [(Int, Int)]?) -> MLXArray {
        guard var pixelValues, let imageSizes, let inputIds else {
            return languageModel.embedTokens(inputIds!)
        }

        let inputsEmbeds = languageModel.embedTokens(inputIds)
        if pixelValues.ndim == 3 { pixelValues = pixelValues.expandedDimensions(axis: 0) }

        let (_, _, hiddenStates) = visionTower(pixelValues.transposed(0, 2, 3, 1), outputHiddenStates: true)
        guard let hiddenStates else { fatalError("Vision model must return hidden states") }

        let layerIndex = config.visionFeatureLayer < 0 ? hiddenStates.count + config.visionFeatureLayer : config.visionFeatureLayer
        let imageFeatures = multiModalProjector(hiddenStates[layerIndex], imageSizes: imageSizes)

        return mergeImageFeatures(imageFeatures: imageFeatures, inputsEmbeds: inputsEmbeds, inputIds: inputIds)
    }

    private func mergeImageFeatures(imageFeatures: MLXArray, inputsEmbeds: MLXArray, inputIds: MLXArray) -> MLXArray {
        let numPatches = imageFeatures.dim(1)
        let inputIdArray: [Int32] = inputIds[0].asArray(Int32.self)
        let imagePositions = inputIdArray.enumerated().compactMap { $1 == Int32(config.imageTokenIndex) ? $0 : nil }

        guard imagePositions.count == numPatches else {
            fatalError("Image token count (\(imagePositions.count)) != patches (\(numPatches))")
        }

        var segments: [MLXArray] = []
        var startIdx = 0
        let splitIndices = Array(1 ..< numPatches)
        let imageEmbeddings = MLX.split(imageFeatures, indices: splitIndices, axis: 1)

        for (text, image) in zip(
            imagePositions.map { pos -> MLXArray in
                let seg = inputsEmbeds[0..., startIdx ..< pos, 0...]
                startIdx = pos + 1
                return seg
            },
            imageEmbeddings
        ) {
            segments.append(text)
            segments.append(image)
        }
        segments.append(inputsEmbeds[0..., startIdx..., 0...])
        return MLX.concatenated(segments, axis: 1)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var w = [String: MLXArray]()
        for (key, value) in weights {
            var k = key
            // Vision tower key remap (same as Mistral3)
            if k.contains("vision_tower") && !k.contains("vision_model") {
                if k.contains("transformer") || k.contains("patch_conv") || k.contains("ln_pre") {
                    k = k.replacingOccurrences(of: "vision_tower", with: "vision_tower.vision_model")
                }
            } else if k.contains("vision_encoder") && !k.contains("vision_tower") {
                if k.contains("transformer") || k.contains("patch_conv") || k.contains("ln_pre") {
                    k = k.replacingOccurrences(of: "vision_encoder", with: "vision_tower.vision_model")
                }
            }
            // JANG: switch_mlp already correct for MoE
            // Skip FP8 scales
            if k.contains("_scale_inv") || k.contains("_activation_scale") { continue }
            w[k] = value
        }
        return w
    }
}

extension Mistral4VLM: LoRAModel {
    public var loraLayers: [Module] { languageModel.layers }
}

// Helper to create Mistral3VLMConfiguration from Mistral4VLMConfiguration (for projector reuse)
private extension Mistral3VLMConfiguration {
    init(from m4: Mistral4VLMConfiguration) {
        // Create a minimal Mistral3VLMConfiguration for the projector
        self.init(
            textConfig: m4.textConfig,
            visionConfig: m4.visionConfig,
            modelType: m4.modelType
        )
    }

    init(textConfig: Mistral3VLMTextConfiguration, visionConfig: Mistral3VisionConfiguration, modelType: String) {
        self = try! JSONDecoder().decode(
            Self.self,
            from: JSONSerialization.data(
                withJSONObject: [
                    "text_config": [
                        "model_type": textConfig.modelType,
                        "hidden_size": textConfig.hiddenSize,
                        "num_hidden_layers": textConfig.numHiddenLayers,
                        "intermediate_size": textConfig.intermediateSize,
                        "num_attention_heads": textConfig.numAttentionHeads,
                        "rms_norm_eps": textConfig.rmsNormEps,
                        "vocab_size": textConfig.vocabSize,
                    ] as [String: Any],
                    "vision_config": [
                        "model_type": "pixtral",
                        "hidden_size": visionConfig.hiddenSize,
                        "num_hidden_layers": visionConfig.numHiddenLayers,
                        "num_attention_heads": visionConfig.numAttentionHeads,
                        "intermediate_size": visionConfig.intermediateSize,
                        "image_size": visionConfig.imageSize,
                        "patch_size": visionConfig.patchSize,
                        "num_channels": visionConfig.numChannels,
                    ] as [String: Any],
                    "model_type": modelType,
                ] as [String: Any]))
    }
}
