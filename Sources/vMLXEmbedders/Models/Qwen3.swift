// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import vMLXLMCommon
import MLXNN

// MARK: - Attention

/// A multi-head attention module with QK-Normalization and Rotary Positional Embeddings.
///
/// This implementation is designed for Qwen-style architectures, featuring:
/// - **QK-Norm**: Normalizing queries and keys before the dot product to improve stability.
/// - **GQA/MQA Support**: Flexible number of query vs key-value heads.
/// - **Dynamic RoPE**: Support for linear scaling of rotary embeddings.
private class Attention: Module {

    let args: Qwen3Configuration
    let scale: Float

    /// Projection layers for Query, Key, and Value
    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear

    /// Final output projection
    @ModuleInfo(key: "o_proj") var wo: Linear

    /// Normalization layers applied to Q and K to stabilize attention scores
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    /// Rotary Positional Embedding logic
    let rope: RoPELayer

    /// Initializes the Attention module.
    /// - Parameter args: Configuration object containing `hiddenSize`, `attentionHeads`, `headDim`, etc.
    public init(_ args: Qwen3Configuration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads
        let headDim = args.headDim

        // Softmax scaling factor: 1 / sqrt(headDim)
        self.scale = Float(pow(Double(headDim), -0.5))

        // Initialize Projections
        _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        // Initialize QK-Norm
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        self.rope = initializeRope(
            dims: headDim, base: args.ropeTheta,
            traditional: false, scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: nil
        )
    }

    /// Forward pass for the attention layer.
    /// - Parameters:
    ///   - x: Input tensor of shape `[Batch, Length, HiddenSize]`.
    ///   - mask: The attention mask mode (typically causal).
    ///   - cache: Optional Key-Value cache for efficient inference.
    /// - Returns: The context-aware representation of the input.
    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        // 1. Initial projections
        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // 2. Reshape and Apply QK-Norm
        // Reshape to [Batch, Length, Heads, HeadDim] and transpose for Attention
        queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        // 3. Apply Rotary Positional Embeddings & Handle Cache
        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
            (keys, values) = cache.update(keys: keys, values: values)
        } else {
            queries = rope(queries, offset: 0)
            keys = rope(keys, offset: 0)
        }

        // 4. Efficient Scaled Dot-Product Attention
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)  // Restore [Batch, Length, Heads, HeadDim]
        .reshaped(B, L, -1)  // Flatten heads: [Batch, Length, HiddenSize]

        // 5. Final output projection
        return wo(output)
    }
}

// MARK: - MLP

/// A Gated Multi-Layer Perceptron (MLP) using the SiGLU activation function.
///
/// This module implements the feed-forward network found in modern transformer blocks.
/// It uses a "Gated Linear Unit" approach where one projection acts as a filter (gate)
/// for the other before projecting back to the model's hidden state dimension.
private class MLP: Module, UnaryLayer {

    /// Projection from model dimension to a higher-dimensional hidden space (gate branch).
    @ModuleInfo(key: "gate_proj") var gate: Linear

    /// Projection from the hidden space back to the model dimension.
    @ModuleInfo(key: "down_proj") var down: Linear

    /// Projection from model dimension to a higher-dimensional hidden space (value branch).
    @ModuleInfo(key: "up_proj") var up: Linear

    /// Initializes the MLP module.
    /// - Parameters:
    ///   - dimensions: The input and output dimension (e.g., `hidden_size`).
    ///   - hiddenDimensions: The intermediate dimension (e.g., `intermediate_size`),
    ///     usually larger than the input dimension.
    public init(dimensions: Int, hiddenDimensions: Int) {
        // Linear projections without bias, following standard LLM practices.
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    /// Forward pass of the MLP.
    /// - Parameter x: Input tensor of shape `[..., dimensions]`.
    /// - Returns: Output tensor of shape `[..., dimensions]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // SwiGLU implementation:
        // 1. gate(x) -> SiLU activation
        // 2. up(x)   -> Linear projection
        // 3. Multiply (1) and (2) element-wise
        // 4. down()  -> Project back to original dimension
        down(silu(gate(x)) * up(x))
    }
}

// MARK: - Transformer Block

/// A single Transformer layer block consisting of Self-Attention and a Feed-Forward Network (MLP).
///
/// This class implements the "Pre-Norm" configuration where normalization is applied
/// before each sub-layer. It uses residual connections (skip connections) to allow
/// gradients to flow through deep stacks of blocks.
private class TransformerBlock: Module {

    /// The self-attention mechanism (includes QK-Norm and RoPE).
    @ModuleInfo(key: "self_attn") var attention: Attention

    /// The feed-forward network (SwiGLU).
    let mlp: MLP

    /// Normalization applied before the attention layer.
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm

    /// Normalization applied before the MLP layer.
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    /// Initializes the Transformer Block.
    /// - Parameter args: The model configuration containing `hiddenSize`, `intermediateSize`, and `rmsNormEps`.
    public init(_ args: Qwen3Configuration) {
        // Initialize the two main processing sub-layers
        _attention.wrappedValue = Attention(args)
        self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)

        // Initialize RMSNorm layers with the specified epsilon for numerical stability
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )
    }

    /// Forward pass of the Transformer Block.
    /// - Parameters:
    ///   - x: Input tensor of shape `[Batch, Length, HiddenSize]`.
    ///   - mask: The attention mask to prevent attending to future tokens.
    ///   - cache: Optional Key-Value cache for incremental inference.
    /// - Returns: The processed tensor, maintaining the input shape.
    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        // 1. Attention Sub-layer
        // Apply Pre-Norm -> Attention -> Residual Connection
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r

        // 2. MLP Sub-layer
        // Apply Pre-Norm -> MLP -> Residual Connection
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r

        return out
    }
}

// MARK: - Qwen3ModelInner

/// The core backbone of the Qwen3 model.
///
/// This class handles the embedding layer, the sequential stack of transformer blocks,
/// and the final output normalization. It does not include the LM head (the mapping
/// back to vocabulary logits).
private class Qwen3ModelInner: Module {

    /// Maps input token IDs to dense vectors.
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    /// The sequence of Transformer layers that form the depth of the model.
    fileprivate let layers: [TransformerBlock]

    /// The final normalization layer applied after all transformer blocks.
    let norm: RMSNorm

    /// Initializes the model backbone.
    /// - Parameter args: Configuration containing `vocabularySize`, `hiddenLayers`, and `hiddenSize`.
    public init(_ args: Qwen3Configuration) {
        precondition(args.vocabularySize > 0, "Vocabulary size must be greater than 0.")

        // 1. Initialize the Token Embedding layer
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        // 2. Build the Transformer stack
        self.layers = (0 ..< args.hiddenLayers).map { _ in
            TransformerBlock(args)
        }

        // 3. Initialize final RMSNorm
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    /// Forward pass through the model backbone.
    /// - Parameters:
    ///   - inputs: An `MLXArray` of token IDs with shape `[Batch, Length]`.
    ///   - cache: An optional array of `KVCache` objects, one for each layer.
    /// - Returns: The final hidden states of shape `[Batch, Length, HiddenSize]`.
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        // Step 1: Token Embedding lookup
        var h = embedTokens(inputs)

        // Step 2: Create the Attention Mask
        // Uses MLXFast utility to determine if a causal or additive mask is needed
        let mask = createAttentionMask(h: h, cache: cache?.first)

        // Step 3: Sequential processing through Transformer Layers
        for (i, layer) in layers.enumerated() {
            // Pass the specific cache slice for this layer index
            h = layer(h, mask: mask, cache: cache?[i])
        }

        // Step 4: Final Layer Normalization
        return norm(h)
    }
}

// MARK: - Qwen3Model

/// The top-level Qwen3 model class providing embedding-level outputs.
///
/// This class conforms to `EmbeddingModel`, making it suitable for use as a base
/// for Causal Language Models or as a standalone encoder for generating text embeddings.
public class Qwen3Model: Module, EmbeddingModel {

    /// The size of the vocabulary used for tokenization.
    public let vocabularySize: Int

    /// An array specifying the number of KV heads for each layer.
    /// Useful for initializing KV caches during inference.
    public let kvHeads: [Int]

    /// The internal backbone processing the embeddings and transformer blocks.
    @ModuleInfo(key: "model") private var model: Qwen3ModelInner

    /// The immutable configuration used to build the model.
    let configuration: Qwen3Configuration

    /// Qwen3 embedding models use the final non-padding token as the sentence representation.
    public let poolingStrategy: Pooling.Strategy? = .last

    /// Initializes the Qwen3 Model.
    /// - Parameter args: The configuration parameters for the model architecture.
    public init(_ args: Qwen3Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        // In Qwen, the number of KV heads is consistent across all layers.
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self._model.wrappedValue = Qwen3ModelInner(args)
    }

    /// Forward pass to generate embeddings.
    /// - Parameters:
    ///   - inputIds: Tensor of token indices.
    ///   - positionIds: Optional indices for positional information.
    ///   - tokenTypeIds: Optional indices for segment-based tasks.
    ///   - attentionMask: Optional mask for specific attention patterns.
    /// - Returns: An `EmbeddingModelOutput` containing the final hidden states.
    public func callAsFunction(
        _ inputIds: MLXArray,
        positionIds: MLXArray? = nil,
        tokenTypeIds: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    ) -> EmbeddingModelOutput {
        // Pass the input through the inner model backbone
        let out = model(inputIds, cache: nil)

        return EmbeddingModelOutput(
            hiddenStates: out,
            pooledOutput: nil  // Qwen is typically used without a specific pooler head
        )
    }

    /// Cleans and remaps external weight keys to match the internal module structure.
    ///
    /// This is vital when loading weights from formats like Hugging Face `safetensors`.
    /// - Parameter weights: A dictionary of original weight keys and values.
    /// - Returns: A sanitized dictionary with keys correctly prefixed with `model.`.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = [String: MLXArray]()

        for (key, value) in weights {
            // 1. Remove parameters that are handled dynamically by the code
            // 'inv_freq' is recalculated for RoPE; 'lm_head' belongs to the CausalLM wrapper
            if key.contains("self_attn.rotary_emb.inv_freq") || key.contains("lm_head") {
                continue
            }

            // 2. Ensure all backbone keys are prefixed with "model."
            // to match the @ModuleInfo(key: "model") property.
            var newKey = key
            if !newKey.hasPrefix("model.") {
                newKey = "model." + newKey
            }

            sanitizedWeights[newKey] = value
        }

        return sanitizedWeights
    }
}

// MARK: - Qwen3Configuration

/// Configuration for a Qwen3 model.
///
/// This struct conforms to `Codable` to allow loading from standard JSON configuration files
/// and `Sendable` for safe use across concurrent Swift tasks.
public struct Qwen3Configuration: Codable, Sendable {

    /// Dimensionality of the encoder layers and the pooler layer.
    var hiddenSize: Int

    /// Number of hidden layers in the Transformer encoder.
    var hiddenLayers: Int

    /// Dimensionality of the "intermediate" (feed-forward) layer.
    var intermediateSize: Int

    /// Number of attention heads for each attention layer in the Transformer encoder.
    var attentionHeads: Int

    /// The epsilon value used by the RMSNorm layers.
    var rmsNormEps: Float

    /// Vocabulary size of the Qwen3 model.
    var vocabularySize: Int

    /// Number of key-value heads for Grouped Query Attention (GQA).
    var kvHeads: Int

    /// The base frequency for the Rotary Positional Embeddings (RoPE).
    var ropeTheta: Float = 1_000_000

    /// Dimensionality of each attention head.
    var headDim: Int

    /// Optional scaling configuration for RoPE (e.g., for context length extension).
    var ropeScaling: [String: StringOrNumber]? = nil

    /// Whether to share weights between the input embeddings and the output projection.
    var tieWordEmbeddings = false

    /// The maximum sequence length that this model might ever be used with.
    var maxPositionEmbeddings: Int = 32768

    /// Mapping between Swift property names and JSON configuration keys.
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    /// Decodes a configuration from a JSON container, providing industry-standard defaults where keys are missing.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.headDim = try container.decode(Int.self, forKey: .headDim)

        // Decode optional fields with fallback defaults
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
    }
}
