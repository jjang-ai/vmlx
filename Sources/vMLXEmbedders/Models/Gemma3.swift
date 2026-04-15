// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import vMLXLMCommon
import MLXNN

// Based on https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma3_text.py

// MARK: - Gemma3Configuration

/// Configuration for the Gemma 3 model.
///
/// This structure holds all the hyperparameters required to initialize a `Gemma3ModelBackbone` or an `EmbeddingGemma` model.
/// It supports decoding from standard `config.json` files, including those nested under a `text_config` key.
public struct Gemma3Configuration: Codable, Sendable {
    /// The architecture identifier (e.g., "gemma3").
    public let modelType: String

    /// The size of the hidden embeddings.
    public let hiddenSize: Int

    /// The number of hidden layers in the transformer.
    public let hiddenLayers: Int

    /// The dimensionality of the intermediate (feed-forward) layer.
    public let intermediateSize: Int

    /// The number of attention heads.
    public let attentionHeads: Int

    /// The dimensionality of each attention head.
    public let headDim: Int

    /// The epsilon value for RMSNorm layers.
    public let rmsNormEps: Float

    /// The size of the vocabulary.
    public let vocabularySize: Int

    /// The number of key-value heads for Grouped Query Attention.
    public let kvHeads: Int

    /// The base frequency for Rotary Positional Embeddings (RoPE) in global layers.
    public let ropeTheta: Float

    /// The base frequency for RoPE in sliding window layers.
    public let ropeLocalBaseFreq: Float

    /// Whether to use traditional RoPE (interleaved) or standard ordering.
    public let ropeTraditional: Bool

    /// The scalar value used to scale queries before attention.
    public let queryPreAttnScalar: Float

    /// The size of the sliding window for attention.
    public let slidingWindow: Int

    /// The pattern determining which layers use sliding window vs. global attention.
    public let slidingWindowPattern: Int

    /// The maximum sequence length supported by the model.
    public let maxPositionEmbeddings: Int

    /// Optional scaling configuration for RoPE.
    public let ropeScaling: [String: StringOrNumber]?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeLocalBaseFreq = "rope_local_base_freq"
        case ropeTraditional = "rope_traditional"
        case queryPreAttnScalar = "query_pre_attn_scalar"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeScaling = "rope_scaling"
    }

    enum VLMCodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    /// Initializes a configuration by decoding from the given decoder.
    ///
    /// This initializer automatically handles configurations that are nested under a `text_config` key,
    /// which is common for models converted from Vision-Language Model (VLM) checkpoints.
    public init(from decoder: Decoder) throws {
        let nestedContainer = try decoder.container(keyedBy: VLMCodingKeys.self)

        let container =
            if nestedContainer.contains(.textConfig) {
                try nestedContainer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            } else {
                try decoder.container(keyedBy: CodingKeys.self)
            }

        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 26
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6912
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 4
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1.0e-6
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262144
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 1
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        ropeLocalBaseFreq =
            try container.decodeIfPresent(Float.self, forKey: .ropeLocalBaseFreq) ?? 10_000.0
        ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        queryPreAttnScalar =
            try container.decodeIfPresent(Float.self, forKey: .queryPreAttnScalar) ?? 256
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        slidingWindowPattern =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 6
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
    }
}

// MARK: - Attention

/// Multi-head attention module for Gemma 3.
///
/// This module implements the self-attention mechanism, supporting both global and sliding window attention
/// patterns as determined by the model's configuration and layer index.
private class Attention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let repeats: Int
    let headDim: Int
    let layerIdx: Int
    let scale: Float
    let isSliding: Bool
    let slidingWindow: Int
    let slidingWindowPattern: Int

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear

    @ModuleInfo(key: "q_norm") var queryNorm: Gemma.RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: Gemma.RMSNorm

    @ModuleInfo var rope: RoPE

    /// Initializes an attention module.
    ///
    /// - Parameters:
    ///   - config: The configuration defining dimensions and patterns.
    ///   - layerIdx: The index of the current layer in the transformer stack.
    init(_ config: Gemma3Configuration, layerIdx: Int) {
        let dim = config.hiddenSize
        self.nHeads = config.attentionHeads
        self.nKVHeads = config.kvHeads
        self.repeats = nHeads / nKVHeads
        self.headDim = config.headDim
        self.layerIdx = layerIdx
        self.slidingWindow = config.slidingWindow
        self.slidingWindowPattern = config.slidingWindowPattern

        self.scale = pow(config.queryPreAttnScalar, -0.5)

        self._queryProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._keyProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._valueProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._outputProj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

        self._queryNorm.wrappedValue = Gemma.RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._keyNorm.wrappedValue = Gemma.RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        self.isSliding = (layerIdx + 1) % config.slidingWindowPattern != 0

        if isSliding {
            self.rope = RoPE(
                dimensions: headDim,
                traditional: false,
                base: config.ropeLocalBaseFreq
            )
        } else {
            self.rope = RoPE(
                dimensions: headDim,
                traditional: false,
                base: config.ropeTheta
            )
        }

        super.init()
    }

    /// Computes the attention mechanism.
    ///
    /// - Parameters:
    ///   - x: Input array of shape `[Batch, Length, HiddenSize]`.
    ///   - mask: The attention mask mode (e.g., causal).
    /// - Returns: Processed array of shape `[Batch, Length, HiddenSize]`.
    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = queryProj(x)
        var keys = keyProj(x)
        var values = valueProj(x)

        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        queries = queryNorm(queries)
        keys = keyNorm(keys)

        queries = rope(queries)
        keys = rope(keys)

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return outputProj(output)
    }
}

// MARK: - MLP

/// Multi-Layer Perceptron (MLP) module for Gemma 3.
///
/// This module implements the feed-forward network with a gated activation (SwiGLU variant using GELU).
private class MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    /// Initializes an MLP module.
    ///
    /// - Parameters:
    ///   - dimensions: The input and output dimensionality.
    ///   - hiddenDimensions: The intermediate dimensionality.
    init(dimensions: Int, hiddenDimensions: Int) {
        self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        super.init()
    }

    /// Performs the MLP forward pass.
    ///
    /// - Parameter x: Input array of shape `[..., dimensions]`.
    /// - Returns: Processed array of shape `[..., dimensions]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(safeGeluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - TransformerBlock

/// A single Transformer block for Gemma 3.
///
/// Each block consists of an attention layer and an MLP layer, with pre- and post-normalization
/// and residual connections.
private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Attention
    @ModuleInfo var mlp: MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma.RMSNorm

    /// Initializes a transformer block.
    ///
    /// - Parameters:
    ///   - config: The model configuration.
    ///   - layerIdx: The index of this block in the stack.
    init(_ config: Gemma3Configuration, layerIdx: Int) {
        self._selfAttention.wrappedValue = Attention(config, layerIdx: layerIdx)
        self.mlp = MLP(dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)

        self._inputLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    /// Processes inputs through the transformer block.
    ///
    /// - Parameters:
    ///   - x: Input array of shape `[Batch, Length, HiddenSize]`.
    ///   - mask: The attention mask mode.
    /// - Returns: Processed array of shape `[Batch, Length, HiddenSize]`.
    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let inputNorm = inputLayerNorm(x)
        let r = selfAttention(inputNorm, mask: mask)
        let attnNorm = postAttentionLayerNorm(r)
        let h = Gemma.clipResidual(x, attnNorm)
        let preMLPNorm = preFeedforwardLayerNorm(h)
        let r2 = mlp(preMLPNorm)
        let postMLPNorm = postFeedforwardLayerNorm(r2)
        let out = Gemma.clipResidual(h, postMLPNorm)
        return out
    }
}

// MARK: - Gemma3ModelBackbone

/// The core backbone of the Gemma 3 model.
///
/// This module handles token embeddings, the sequence of transformer blocks, and the final normalization.
/// It is used as the base for the embedding model.
public class Gemma3ModelBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo fileprivate var layers: [TransformerBlock]
    @ModuleInfo var norm: Gemma.RMSNorm

    let config: Gemma3Configuration

    /// Initializes the model backbone.
    ///
    /// - Parameter config: The model configuration.
    public init(_ config: Gemma3Configuration) {
        self.config = config

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )

        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIdx in
            TransformerBlock(config, layerIdx: layerIdx)
        }

        self._norm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    /// Processes input tokens through the model backbone.
    ///
    /// - Parameter inputs: Array of token IDs of shape `[Batch, Length]`.
    /// - Returns: Final hidden states of shape `[Batch, Length, HiddenSize]`.
    public func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        var h = embedTokens(inputs)
        let scale = MLXArray(sqrt(Float(config.hiddenSize)), dtype: .bfloat16)
        h = h * scale.asType(h.dtype)

        let globalMask = createAttentionMask(h: h, cache: nil as KVCache?)
        let slidingWindowMask =
            if config.slidingWindowPattern > 1 {
                createAttentionMask(h: h, cache: nil as KVCache?, windowSize: config.slidingWindow)
            } else {
                MLXFast.ScaledDotProductAttentionMaskMode.none
            }

        for (i, layer) in layers.enumerated() {
            let isGlobal = (i % config.slidingWindowPattern == config.slidingWindowPattern - 1)
            let mask = isGlobal ? globalMask : slidingWindowMask
            h = layer(h, mask: mask)
        }
        return norm(h)
    }
}

// MARK: - EmbeddingGemma

/// A text embedding model based on Gemma 3.
///
/// This class conforms to `EmbeddingModel` and provides high-quality text embeddings by passing
/// inputs through a `Gemma3ModelBackbone`, followed by mean pooling and a projection head.
public class EmbeddingGemma: Module, EmbeddingModel {
    @ModuleInfo(key: "model") private var backbone: Gemma3ModelBackbone
    @ModuleInfo private var dense: [Module]

    /// The model configuration.
    public let config: Gemma3Configuration

    /// The size of the vocabulary.
    public var vocabularySize: Int { config.vocabularySize }

    /// Initializes the embedding model.
    ///
    /// - Parameter config: The model configuration.
    public init(_ config: Gemma3Configuration) {
        self.config = config
        self._backbone.wrappedValue = Gemma3ModelBackbone(config)

        self._dense.wrappedValue = []
        super.init()
    }

    /// Generates embeddings for the given inputs.
    ///
    /// - Parameters:
    ///   - inputs: Input token indices of shape `[Batch, Length]`.
    ///   - positionIds: Optional indices for positional information.
    ///   - tokenTypeIds: Optional indices for segment/type information.
    ///   - attentionMask: Optional mask for padding tokens.
    /// - Returns: An `EmbeddingModelOutput` containing the full hidden states and the pooled sentence embedding.
    public func callAsFunction(
        _ inputs: MLXArray, positionIds: MLXArray?, tokenTypeIds: MLXArray?,
        attentionMask: MLXArray?
    ) -> EmbeddingModelOutput {
        var inp = inputs
        if inp.ndim == 1 {
            inp = inp.reshaped(1, -1)
        }

        let hiddenStates = backbone(inp)

        // mean pooling: average all non-padding tokens
        let notPadding = (attentionMask ?? (inp .!= 0))
        let sum = (hiddenStates * notPadding[.ellipsis, .newAxis]).sum(axis: 1)
        let nonMasked = notPadding.sum(axis: -1, keepDims: true)
        var out = sum / nonMasked

        // pass through projection head (if present)
        for layer in self.dense {
            if let linear = layer as? Linear {
                out = linear(out)
            } else if let quantized = layer as? QuantizedLinear {
                out = quantized(out)
            }
        }

        // normalize: L2 normalization for cosine similarity compatibility
        let pooledOutput =
            out
            .asType(.float32)
            .l2Normalized(eps: 1e-6)

        return EmbeddingModelOutput(hiddenStates: hiddenStates, pooledOutput: pooledOutput)
    }

    /// Preprocesses the weights, handling key remapping and vocabulary truncation.
    ///
    /// - Parameter weights: The original dictionary of weights.
    /// - Returns: A sanitized dictionary of weights compatible with the model structure.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var processedWeights = weights

        // Support language_model prefix from VLMs
        let unflattened = ModuleParameters.unflattened(weights)
        if let lm = unflattened["language_model"] {
            processedWeights = Dictionary(uniqueKeysWithValues: lm.flattened())
        }

        // Initialize projection head if weights are present
        if processedWeights.keys.contains(where: { $0.hasPrefix("dense.") }) {
            self._dense.wrappedValue = [
                Linear(config.hiddenSize, config.intermediateSize, bias: false),
                Linear(config.intermediateSize, config.hiddenSize, bias: false),
            ]
        }

        // Truncate vocab if weights were trained with extra padding tokens
        let expectedVocab = config.vocabularySize
        if let embedWeight = processedWeights["model.embed_tokens.weight"],
            embedWeight.dim(0) > expectedVocab
        {
            processedWeights["model.embed_tokens.weight"] = embedWeight[0 ..< expectedVocab]
        }

        // Filter out unused keys for embedding
        return processedWeights.filter { (key, _) in
            !key.contains("self_attn.rotary_emb.inv_freq") && !key.contains("lm_head")
        }
    }
}
