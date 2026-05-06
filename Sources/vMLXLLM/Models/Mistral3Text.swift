// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/ministral3.py

import Foundation
import MLX
import vMLXLMCommon
import MLXNN

// MARK: - Llama4 Attention Scaling

/// Compute attention scale for Llama 4 style position-based scaling.
///
/// - Parameters:
///   - start: Start position offset
///   - stop: Stop position offset
///   - beta: Scaling factor (llama_4_scaling_beta)
///   - maxPositionEmbeddings: Original max position embeddings
/// - Returns: Scaling tensor of shape [stop - start, 1]
internal func getLlama4AttentionScale(
    start: Int, stop: Int, beta: Float, maxPositionEmbeddings: Int
) -> MLXArray {
    let positions = MLXArray(Int32(start) ..< Int32(stop))
    let scaling =
        1 + beta
        * MLX.log(
            1 + MLX.floor(positions.asType(.float32) / Float(maxPositionEmbeddings))
        )
    return scaling[0..., .newAxis]
}

/// Graph-visible variant matching upstream mlx-lm's
/// `_get_llama_4_attn_scale(size, offset, ...)`.
///
/// The `start/stop` overload above is fine for prefill and uncompiled
/// decode. Once `setupCompiledDecode` promotes caches to
/// `CompilableKVCache` / `CompilableRotatingKVCache`, reading
/// `cache.offset` would call `.item()` during `compile()`. Use this
/// overload with `graphOffsetArray(for:)` so position-dependent scaling
/// remains inside the graph.
internal func getLlama4AttentionScale(
    size: Int, offset: MLXArray, beta: Float, maxPositionEmbeddings: Int
) -> MLXArray {
    var offset = offset.asType(.float32)
    if offset.ndim > 0 {
        offset = offset[0..., .newAxis]
    }
    let positions = MLXArray(Int32(0) ..< Int32(size)).asType(.float32)
    let scaling =
        1 + beta
        * MLX.log(
            1 + MLX.floor((positions + offset) / Float(maxPositionEmbeddings))
        )
    if scaling.ndim == 2 {
        return scaling[0..., .newAxis, 0..., .newAxis]
    }
    return scaling[0..., .newAxis]
}

// MARK: - Attention

class Mistral3Attention: Module {
    let args: Mistral3TextConfiguration
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPELayer

    init(_ args: Mistral3TextConfiguration) {
        self.args = args

        let dim = args.hiddenSize
        self.nHeads = args.attentionHeads
        self.nKVHeads = args.kvHeads

        self.headDim = args.resolvedHeadDimensions
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

        // Initialize RoPE: prefer rope_parameters dict, fall back to direct ropeTheta
        let ropeTheta = args.ropeParameters?["rope_theta"]?.asFloat() ?? args.ropeTheta
        self.rope = initializeRope(
            dims: headDim,
            base: ropeTheta,
            traditional: false,
            scalingConfig: args.ropeParameters,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, attnScale: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // Prepare the queries, keys and values for the attention computation
        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        // Apply RoPE
        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        // Apply attention scaling
        queries = queries * attnScale

        // Compute attention with automatic quantized/regular cache handling
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

// MARK: - MLP

class Mistral3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ args: Mistral3TextConfiguration) {
        let dim = args.hiddenSize
        let hiddenDim = args.intermediateSize

        self._gate.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._down.wrappedValue = Linear(hiddenDim, dim, bias: false)
        self._up.wrappedValue = Linear(dim, hiddenDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let g = silu(gate(x))
        let u = up(x)
        let product = g * u
        return down(product)
    }
}

// MARK: - Transformer Block

class Mistral3TextTransformerBlock: Module {
    let numAttentionHeads: Int
    let hiddenSize: Int
    let useSliding: Bool

    @ModuleInfo(key: "self_attn") var attention: Mistral3Attention
    @ModuleInfo(key: "mlp") var mlp: Mistral3MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: Mistral3TextConfiguration, useSliding: Bool = false) {
        self.numAttentionHeads = args.attentionHeads
        self.hiddenSize = args.hiddenSize
        self.useSliding = useSliding

        self._attention.wrappedValue = Mistral3Attention(args)
        self._mlp.wrappedValue = Mistral3MLP(args)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, attnScale: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let r = attention(inputLayerNorm(x), attnScale: attnScale, mask: mask, cache: cache)
        let h = x + r
        let mlpOut = mlp(postAttentionLayerNorm(h))
        let out = h + mlpOut
        return out
    }
}

// MARK: - Language Model (Inner)

public class Mistral3TextModelInner: Module {
    let args: Mistral3TextConfiguration
    let vocabularySize: Int
    let numHiddenLayers: Int
    let layerTypes: [String]
    let slidingWindow: Int?

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [Mistral3TextTransformerBlock]
    let norm: RMSNorm

    // Indices for first full attention and sliding window attention layers
    let faIdx: Int
    let swaIdx: Int?

    init(_ args: Mistral3TextConfiguration) {
        self.args = args
        self.vocabularySize = args.vocabularySize
        self.numHiddenLayers = args.hiddenLayers
        self.layerTypes = args.layerTypes
        self.slidingWindow = args.slidingWindow

        precondition(args.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        // Create transformer blocks with appropriate attention type
        self.layers = args.layerTypes.map { layerType in
            Mistral3TextTransformerBlock(args, useSliding: layerType == "sliding_attention")
        }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        // Find the first full attention layer index
        self.faIdx = args.layerTypes.firstIndex(of: "full_attention") ?? 0

        // Find the first sliding window attention layer index
        self.swaIdx = args.layerTypes.firstIndex(of: "sliding_attention")

        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]? = nil, inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        // Use provided embeddings or compute from inputs
        var h: MLXArray
        if let inputEmbeddings = inputEmbeddings {
            h = inputEmbeddings
        } else {
            h = embedTokens(inputs)
        }

        // Create full attention mask
        let faMask = createAttentionMask(h: h, cache: cache?[faIdx])

        // Create sliding window attention mask
        let swaMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let swaIdx = swaIdx {
            swaMask = createAttentionMask(h: h, cache: cache?[swaIdx], windowSize: slidingWindow)
        } else {
            swaMask = .none
        }

        // Compute attention scale. Upstream mlx-lm accepts an array
        // offset here; do the same for compiled decode so Mistral3 /
        // Ministral3 does not host-read `cache.offset` after cache
        // promotion. beta==0 is identically 1 and deliberately skips all
        // offset work (current Mistral-Medium-3.5 bundles use beta 0).
        let attnScale: MLXArray
        if let ropeParams = args.ropeParameters,
            let llama4ScalingBeta = ropeParams["llama_4_scaling_beta"]?.asFloat(),
            let originalMaxPosEmbed = ropeParams["original_max_position_embeddings"]?.asInt(),
            llama4ScalingBeta != 0
        {
            if let offsetArray = graphOffsetArray(for: cache?.first) {
                attnScale = getLlama4AttentionScale(
                    size: inputs.dim(1),
                    offset: offsetArray,
                    beta: llama4ScalingBeta,
                    maxPositionEmbeddings: originalMaxPosEmbed
                ).asType(h.dtype)
            } else {
                let offset = cache?.first?.offset ?? 0
                attnScale = getLlama4AttentionScale(
                    start: offset,
                    stop: offset + inputs.dim(1),
                    beta: llama4ScalingBeta,
                    maxPositionEmbeddings: originalMaxPosEmbed
                ).asType(h.dtype)
            }
        } else {
            attnScale = MLXArray.ones([inputs.dim(1), 1]).asType(h.dtype)
        }

        // Process through transformer layers
        for (i, layer) in layers.enumerated() {
            let mask = layer.useSliding ? swaMask : faMask
            h = layer(h, attnScale: attnScale, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

// MARK: - Model

/// Mistral3Text language model.
public class Mistral3TextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Mistral3TextModelInner
    fileprivate let args: Mistral3TextConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Mistral3TextConfiguration) {
        self.args = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Mistral3TextModelInner(args)

        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunction(inputs, cache: cache, inputEmbeddings: nil)
    }

    public func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?, inputEmbeddings: MLXArray?
    ) -> MLXArray {
        let out = model(inputs, cache: cache, inputEmbeddings: inputEmbeddings)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var processedWeights = weights

        // VLM models converted using mlx_vlm.convert will have
        // the weights under a language_model key
        let unflattened = ModuleParameters.unflattened(weights)
        if let lm = unflattened["language_model"] {
            processedWeights = Dictionary(uniqueKeysWithValues: lm.flattened())
        }

        // Remove unused precomputed rotary frequencies
        var sanitizedWeights = processedWeights.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }

        // Handle tied embeddings
        if args.tieWordEmbeddings {
            sanitizedWeights["lm_head.weight"] = nil
        }

        // Handle weight_scale_inv for quantized weights
        var newWeights: [String: MLXArray] = [:]
        for (key, value) in sanitizedWeights {
            if key.contains("weight_scale_inv") {
                let scaleInv = value
                let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = sanitizedWeights[weightKey] {
                    newWeights[weightKey] = weight * scaleInv
                }
            } else if key.contains("activation_scale") {
                continue
            } else if newWeights[key] == nil {
                newWeights[key] = value
            }
        }

        return newWeights.isEmpty ? sanitizedWeights : newWeights
    }

    /// Create appropriate caches for each layer type.
    ///
    /// Sliding window attention layers use RotatingKVCache,
    /// full attention layers use standard KVCacheSimple.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        return model.layers.map { layer in
            if layer.useSliding, let slidingWindow = args.slidingWindow {
                return RotatingKVCache(maxSize: slidingWindow)
            } else {
                return KVCacheSimple()
            }
        }
    }
}

// MARK: - Configuration

public struct Mistral3TextConfiguration: Codable, Sendable {
    var modelType: String = "ministral3"
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var headDimensions: Int?
    var maxPositionEmbeddings: Int?
    var kvHeads: Int
    var ropeTheta: Float = 10_000
    var ropeParameters: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool = false
    var layerTypes: [String]
    var slidingWindow: Int?

    var resolvedHeadDimensions: Int {
        headDimensions ?? (hiddenSize / attentionHeads)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case headDimensions = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
    }

    enum VLMCodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let topLevelContainer = try decoder.container(keyedBy: CodingKeys.self)
        let nestedContainer = try decoder.container(keyedBy: VLMCodingKeys.self)

        // In the case of VLM models converted using mlx_lm.convert,
        // the configuration will still match the VLMs and be under text_config
        let container =
            if nestedContainer.contains(.textConfig) {
                try nestedContainer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            } else {
                try decoder.container(keyedBy: CodingKeys.self)
            }

        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "ministral3"
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        headDimensions = try container.decodeIfPresent(Int.self, forKey: .headDimensions)
        maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        ropeTheta =
            try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        ropeParameters = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)

        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
            ?? (try topLevelContainer.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings))
            ?? false

        // Handle layer_types with default to all full_attention
        if let types = try container.decodeIfPresent([String].self, forKey: .layerTypes) {
            layerTypes = types
        } else {
            layerTypes = Array(repeating: "full_attention", count: hiddenLayers)
        }

        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
    }

    public init(
        modelType: String = "ministral3",
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        rmsNormEps: Float,
        vocabularySize: Int,
        headDimensions: Int? = nil,
        maxPositionEmbeddings: Int? = nil,
        kvHeads: Int? = nil,
        ropeTheta: Float = 10_000,
        ropeParameters: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = true,
        layerTypes: [String]? = nil,
        slidingWindow: Int? = nil
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.headDimensions = headDimensions
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.kvHeads = kvHeads ?? attentionHeads
        self.ropeTheta = ropeTheta
        self.ropeParameters = ropeParameters
        self.tieWordEmbeddings = tieWordEmbeddings
        self.layerTypes = layerTypes ?? Array(repeating: "full_attention", count: hiddenLayers)
        self.slidingWindow = slidingWindow
    }
}

// MARK: - LoRA

extension Mistral3TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
