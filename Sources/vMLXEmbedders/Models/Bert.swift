// Copyright © 2024 Apple Inc.

import MLX
import MLXNN

// MARK: - Bert Embedding

/// The embedding layer for BERT models, combining token, position, and segment information.
///
/// This module transforms input IDs into a unified hidden representation by summing three
/// distinct embedding vectors:
/// 1. **Word Embeddings**: Represent the semantic meaning of the tokens.
/// 2. **Position Embeddings**: Encode the absolute position of tokens in a sequence.
/// 3. **Token Type Embeddings**: (Optional) Distinguish between different sentence segments (e.g., Sentence A vs Sentence B).
///
/// After summation, a `LayerNorm` is applied to stabilize the hidden state distributions.
private class BertEmbedding: Module {

    /// The number of distinct token types the model can distinguish.
    let typeVocabularySize: Int

    /// The embedding lookup table for the vocabulary.
    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding

    /// Final normalization layer applied after combining all embeddings.
    @ModuleInfo(key: "norm") var norm: LayerNorm

    /// Optional embeddings for sentence segments. Typically used in BERT's Next Sentence Prediction (NSP) tasks.
    @ModuleInfo(key: "token_type_embeddings") var tokenTypeEmbeddings: Embedding?

    /// The embedding lookup table for positional information.
    @ModuleInfo(key: "position_embeddings") var positionEmbeddings: Embedding

    /// Initializes the embedding layers using a provided BERT configuration.
    /// - Parameter config: The `BertConfiguration` containing dimensions, vocabulary sizes, and epsilon values.
    init(_ config: BertConfiguration) {
        typeVocabularySize = config.typeVocabularySize

        // Initialize word embeddings based on vocab size and embedding dimension
        _wordEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.embedDim)

        // Initialize LayerNorm with specified epsilon for numerical stability
        _norm.wrappedValue = LayerNorm(
            dimensions: config.embedDim, eps: config.layerNormEps)

        // Only initialize token type embeddings if the config specifies a type vocabulary
        if config.typeVocabularySize > 0 {
            _tokenTypeEmbeddings.wrappedValue = Embedding(
                embeddingCount: config.typeVocabularySize,
                dimensions: config.embedDim)
        }

        // Initialize position embeddings up to the maximum sequence length supported
        _positionEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.maxPositionEmbeddings,
            dimensions: config.embedDim)
    }

    /// Computes the combined embedding for the input tokens.
    ///
    /// - Parameters:
    ///   - inputIds: An `MLXArray` of token indices with shape `[Batch, SequenceLength]`.
    ///   - positionIds: Optional `MLXArray` of position indices. If `nil`, indices are generated as `0...SequenceLength-1`.
    ///   - tokenTypeIds: Optional `MLXArray` indicating the segment (0 or 1) for each token.
    /// - Returns: A normalized `MLXArray` of hidden states with shape `[Batch, SequenceLength, EmbeddingDimension]`.
    func callAsFunction(
        _ inputIds: MLXArray,
        positionIds: MLXArray? = nil,
        tokenTypeIds: MLXArray? = nil
    ) -> MLXArray {
        // Generate position IDs [0, 1, ..., N] and broadcast to the input shape if not provided
        let posIds = positionIds ?? broadcast(MLXArray.arange(inputIds.dim(1)), to: inputIds.shape)

        // Combine word and position embeddings
        var words = wordEmbeddings(inputIds) + positionEmbeddings(posIds)

        // Add token type (segment) embeddings if both IDs and the layer exist
        if let tokenTypeIds, let tokenTypeEmbeddings {
            words += tokenTypeEmbeddings(tokenTypeIds)
        }

        // Final normalization pass
        return norm(words)
    }
}

/// A single Transformer Encoder layer implementing BERT-style architecture.
///
/// This block consists of two main sub-layers:
/// 1. **Multi-Head Self-Attention**: Allows tokens to attend to other tokens in the sequence.
/// 2. **Feed-Forward Network (MLP)**: A two-stage linear transformation with a non-linear activation.
///
/// Each sub-layer utilizes **Residual Connections** (skip connections) followed by **Layer Normalization**
/// to facilitate deep network training and stable gradients.
private class TransformerBlock: Module {

    /// The self-attention mechanism that computes contextual relationships between tokens.
    let attention: MultiHeadAttention

    /// Normalization applied after the attention sub-layer's residual sum.
    @ModuleInfo(key: "ln1") var preLayerNorm: LayerNorm

    /// Normalization applied after the feed-forward sub-layer's residual sum.
    @ModuleInfo(key: "ln2") var postLayerNorm: LayerNorm

    /// The first linear projection in the MLP (expansion layer).
    @ModuleInfo(key: "linear1") var up: Linear

    /// The second linear projection in the MLP (contraction layer).
    @ModuleInfo(key: "linear2") var down: Linear

    /// Initializes the block components based on the model configuration.
    /// - Parameter config: The `BertConfiguration` defining dimensions and layer parameters.
    init(_ config: BertConfiguration) {
        // Multi-head attention using the model's hidden dimension and head count.
        attention = MultiHeadAttention(
            dimensions: config.embedDim, numHeads: config.numHeads, bias: true)

        // Layer norms initialized with configuration-specific epsilon values.
        _preLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.embedDim, eps: config.layerNormEps)
        _postLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.embedDim, eps: config.layerNormEps)

        // MLP structure: expands the dimension to 'interDim' and then projects back.
        _up.wrappedValue = Linear(config.embedDim, config.interDim)
        _down.wrappedValue = Linear(config.interDim, config.embedDim)
    }

    /// Processes input hidden states through the transformer layer.
    ///
    /// - Parameters:
    ///   - inputs: Input tensor from the previous layer [Batch, Sequence, HiddenDim].
    ///   - mask: Optional attention mask to prevent attending to padding or future tokens.
    /// - Returns: Context-aware hidden states [Batch, Sequence, HiddenDim].
    func callAsFunction(_ inputs: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        // --- 1. Attention Sub-layer ---
        // Self-attention: query, key, and value are all derived from the same inputs.
        let attentionOut = attention(inputs, keys: inputs, values: inputs, mask: mask)

        // Residual Connection + Layer Norm
        let preNorm = preLayerNorm(inputs + attentionOut)

        // --- 2. Feed-Forward (MLP) Sub-layer ---
        // 'up' expands, 'gelu' adds non-linearity, 'down' returns to original dimension.
        let mlpOut = down(gelu(up(preNorm)))

        // Second Residual Connection + Layer Norm
        return postLayerNorm(mlpOut + preNorm)
    }
}

/// The core Transformer backbone consisting of stacked transformer blocks.
///
/// The `Encoder` is responsible for deep contextual processing. It takes the initial
/// embeddings and passes them through a sequence of identical `TransformerBlock` layers.
/// Each layer further refines the understanding of token relationships.
private class Encoder: Module {

    /// The sequence of transformer layers that process the input.
    let layers: [TransformerBlock]

    /// Initializes the encoder stack.
    /// - Parameter config: The `BertConfiguration` specifying the number of layers
    ///   and hidden dimensions.
    init(_ config: BertConfiguration) {
        // Ensure valid configuration before initialization.
        precondition(config.vocabularySize > 0)

        // Dynamically create the layer stack based on config.numLayers (typically 12 or 24).
        layers = (0 ..< config.numLayers).map { _ in TransformerBlock(config) }
    }

    /// Sequentially processes hidden states through all stacked layers.
    ///
    /// - Parameters:
    ///   - inputs: The output from the embedding layer [Batch, Sequence, HiddenDim].
    ///   - attentionMask: Optional mask to prevent the model from attending to padding tokens.
    /// - Returns: The final contextualized hidden states from the last transformer block.
    func callAsFunction(_ inputs: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        var outputs = inputs

        // Pass the output of each layer as the input to the next.
        for layer in layers {
            outputs = layer(outputs, mask: attentionMask)
        }

        return outputs
    }
}

// MARK: - Language Model Head

/// The Task Head used for Masked Language Modeling (MLM).
///
/// The `LMHead` (Language Model Head) transforms the abstract hidden states back
/// into the vocabulary space. It is typically used during pre-training to predict
/// masked tokens or during fine-tuning for specific generative tasks.
private class LMHead: Module {

    /// A intermediate projection layer to refine features before decoding.
    @ModuleInfo(key: "dense") var dense: Linear

    /// Normalization layer to stabilize the projected hidden states.
    @ModuleInfo(key: "ln") var layerNorm: LayerNorm

    /// The final projection layer that maps hidden states to the vocabulary size.
    @ModuleInfo(key: "decoder") var decoder: Linear

    /// Initializes the LM Head components.
    /// - Parameter config: The `BertConfiguration` containing embedding dimensions
    ///   and vocabulary size.
    init(_ config: BertConfiguration) {
        // Project from embedding dimension to same dimension (standard BERT behavior).
        _dense.wrappedValue = Linear(
            config.embedDim, config.embedDim, bias: true)

        _layerNorm.wrappedValue = LayerNorm(
            dimensions: config.embedDim, eps: config.layerNormEps)

        // Map from hidden space to the total number of possible tokens.
        _decoder.wrappedValue = Linear(
            config.embedDim, config.vocabularySize, bias: true)
    }

    /// Maps hidden states to vocabulary logits.
    ///
    /// The process follows: Linear -> SiLU Activation -> LayerNorm -> Vocabulary Projection.
    /// - Parameter inputs: Hidden states from the Encoder [Batch, Sequence, HiddenDim].
    /// - Returns: Logits for each token in the vocabulary [Batch, Sequence, VocabSize].
    func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        // Note: silu (Sigmoid Linear Unit) is used here as the activation function.
        return decoder(layerNorm(silu(dense(inputs))))
    }
}

// MARK: - BERT Model

/// The complete BERT model implementation.
///
/// `BertModel` coordinates the flow of data from raw input tokens to high-level
/// contextual representations. It can be configured in two modes:
/// 1. **Feature Extraction (Default)**: Produces sequence-level hidden states and
///    a pooled sentence embedding (useful for similarity or classification).
/// 2. **Language Modeling**: Attaches an `LMHead` to predict masked tokens.
public class BertModel: Module, EmbeddingModel {

    /// Optional head for Masked Language Modeling tasks.
    @ModuleInfo(key: "lm_head") fileprivate var lmHead: LMHead?

    /// The initial embedding layer (Word + Position + Type).
    @ModuleInfo(key: "embeddings") fileprivate var embedder: BertEmbedding

    /// A linear layer used to "pool" the [CLS] token into a single sentence vector.
    @ModuleInfo var pooler: Linear?

    /// The stack of Transformer layers.
    fileprivate let encoder: Encoder

    /// The total count of tokens in the model's vocabulary.
    public var vocabularySize: Int

    /// Initializes a BERT model.
    /// - Parameters:
    ///   - config: The architecture settings (layers, heads, dimensions).
    ///   - lmHead: If `true`, includes the decoder head for token prediction.
    ///     If `false`, includes a pooler for sentence embeddings.
    public init(_ config: BertConfiguration, lmHead: Bool = false) {
        precondition(config.vocabularySize > 0)
        vocabularySize = config.vocabularySize
        encoder = Encoder(config)
        _embedder.wrappedValue = BertEmbedding(config)

        if lmHead {
            _lmHead.wrappedValue = LMHead(config)
            _pooler.wrappedValue = nil
        } else {
            // Pooler projects the [CLS] token to a hidden state of the same size
            _pooler.wrappedValue = Linear(config.embedDim, config.embedDim)
            _lmHead.wrappedValue = nil
        }
    }

    /// Primary inference entry point.
    ///
    /// - Parameters:
    ///   - inputs: Token IDs [Batch, SeqLen].
    ///   - positionIds: Optional indices for positions.
    ///   - tokenTypeIds: Optional segment IDs (0 or 1).
    ///   - attentionMask: Binary mask (1 for active, 0 for padding).
    /// - Returns: An `EmbeddingModelOutput` containing hidden states and/or pooled vectors.
    public func callAsFunction(
        _ inputs: MLXArray,
        positionIds: MLXArray? = nil,
        tokenTypeIds: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    ) -> EmbeddingModelOutput {
        var inp = inputs
        if inp.ndim == 1 {
            inp = inp.reshaped(1, -1)
        }
        let embeddings = embedder(inp, positionIds: positionIds, tokenTypeIds: tokenTypeIds)
        var mask = attentionMask
        if mask != nil {
            // Cast mask to the same dtype as the embeddings output so it is
            // compatible with scaled_dot_product_attention's type promotion
            // rules. Using the embedding weight dtype can produce a mismatch
            // when Linear layers are quantized to float16 but Embedding
            // weights remain float32.
            mask = mask!.asType(embeddings.dtype).expandedDimensions(axes: [
                1, 2,
            ]).log()
        }
        let outputs = encoder(embeddings, attentionMask: mask)
        if let lmHead {
            return EmbeddingModelOutput(hiddenStates: lmHead(outputs), pooledOutput: nil)
        } else {
            return EmbeddingModelOutput(
                hiddenStates: outputs, pooledOutput: tanh(pooler!(outputs[0..., 0])))
        }
    }

    /// Maps external weight names (e.g., from Hugging Face) to this class's internal structure.
    ///
    /// This is essential for loading pre-trained weights, as it renames keys
    /// like `attention.output.dense` to `attention.out_proj`.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.reduce(into: [:]) { result, item in
            let key = item.key
                .replacingOccurrences(of: ".layer.", with: ".layers.")
                .replacingOccurrences(of: ".self.key.", with: ".key_proj.")
                .replacingOccurrences(of: ".self.query.", with: ".query_proj.")
                .replacingOccurrences(of: ".self.value.", with: ".value_proj.")
                .replacingOccurrences(of: ".attention.output.dense.", with: ".attention.out_proj.")
                .replacingOccurrences(of: ".attention.output.LayerNorm.", with: ".ln1.")
                .replacingOccurrences(of: ".output.LayerNorm.", with: ".ln2.")
                .replacingOccurrences(of: ".intermediate.dense.", with: ".linear1.")
                .replacingOccurrences(of: ".output.dense.", with: ".linear2.")
                .replacingOccurrences(of: ".LayerNorm.", with: ".norm.")
                .replacingOccurrences(of: "pooler.dense.", with: "pooler.")
                .replacingOccurrences(
                    of: "cls.predictions.transform.dense.", with: "lm_head.dense."
                )
                .replacingOccurrences(
                    of: "cls.predictions.transform.LayerNorm.", with: "lm_head.ln."
                )
                .replacingOccurrences(of: "cls.predictions.decoder", with: "lm_head.decoder")
                .replacingOccurrences(
                    of: "cls.predictions.transform.norm.weight", with: "lm_head.ln.weight"
                )
                .replacingOccurrences(
                    of: "cls.predictions.transform.norm.bias", with: "lm_head.ln.bias"
                )
                .replacingOccurrences(of: "cls.predictions.bias", with: "lm_head.decoder.bias")
                .replacingOccurrences(of: "bert.", with: "")

            result[key] = item.value
        }.filter { key, _ in key != "embeddings.position_ids" }
    }
}

// MARK: - DistilBERT

/// A streamlined version of the BERT model.
///
/// DistilBERT is a small, fast, cheap and light Transformer model trained by distilling
/// BERT base. It has less parameters than bert-base-uncased and runs faster
/// while preserving the majority of BERT's performance.
public class DistilBertModel: BertModel {

    /// Remaps DistilBERT-specific weight keys to the internal MLX architecture.
    ///
    /// DistilBERT weights use different naming patterns than standard BERT (e.g., using `ffn`
    /// instead of `intermediate`). This method ensures that weights from models like
    /// `distilbert-base-uncased` map correctly to the Swift property names.
    ///
    /// - Parameter weights: A dictionary of keys and tensors from the model file.
    /// - Returns: A sanitized dictionary compatible with the `Module` property keys.
    public override func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.reduce(into: [:]) { result, item in
            let key = item.key
                .replacingOccurrences(of: ".layer.", with: ".layers.")
                // Architecture-specific remapping
                .replacingOccurrences(of: "transformer.", with: "encoder.")
                .replacingOccurrences(of: "embeddings.LayerNorm", with: "embeddings.norm")
                // Attention mapping
                .replacingOccurrences(of: ".attention.q_lin.", with: ".attention.query_proj.")
                .replacingOccurrences(of: ".attention.k_lin.", with: ".attention.key_proj.")
                .replacingOccurrences(of: ".attention.v_lin.", with: ".attention.value_proj.")
                .replacingOccurrences(of: ".attention.out_lin.", with: ".attention.out_proj.")
                // Layer Norm and Feed-Forward mapping
                .replacingOccurrences(of: ".sa_layer_norm.", with: ".ln1.")
                .replacingOccurrences(of: ".ffn.lin1.", with: ".linear1.")
                .replacingOccurrences(of: ".ffn.lin2.", with: ".linear2.")
                .replacingOccurrences(of: ".output_layer_norm.", with: ".ln2.")
                // Language Modeling Head mapping
                .replacingOccurrences(of: "vocab_transform", with: "lm_head.dense")
                .replacingOccurrences(of: "vocab_layer_norm", with: "lm_head.ln")
                .replacingOccurrences(of: "vocab_projector", with: "lm_head.decoder")
                // Clean up model prefix
                .replacingOccurrences(of: "distilbert.", with: "")

            result[key] = item.value
        }.filter { key, _ in
            // We ignore position_ids because they are generated dynamically in MLX
            key != "embeddings.position_ids"
        }
    }
}

/// Configuration parameters for BERT and DistilBERT models.
///
/// This struct supports `Decodable` to allow direct initialization from a JSON
/// configuration file. It handles the discrepancy between BERT and DistilBERT
/// JSON keys (e.g., BERT's `hidden_size` vs DistilBERT's `dim`).
public struct BertConfiguration: Decodable, Sendable {
    /// Epsilon value for Layer Normalization to prevent division by zero.
    var layerNormEps: Float = 1e-12

    /// Maximum sequence length the model was trained on.
    var maxTrainedPositions: Int = 2048

    /// The size of the hidden embeddings (e.g., 768 for Base models).
    var embedDim: Int = 768

    /// Number of attention heads in the Multi-Head Attention layer.
    var numHeads: Int = 12

    /// Dimensionality of the intermediate (feed-forward) layer.
    var interDim: Int = 3072

    /// Total number of Transformer blocks in the encoder.
    var numLayers: Int = 12

    /// The number of segment types (usually 2 for BERT, 0 for DistilBERT).
    var typeVocabularySize: Int = 2

    /// Number of tokens in the vocabulary.
    var vocabularySize: Int = 30528

    /// The maximum sequence length supported by position embeddings.
    var maxPositionEmbeddings: Int = 0

    /// The identifier for the model (e.g., "bert" or "distilbert").
    var modelType: String

    // MARK: - Decoding Keys

    enum CodingKeys: String, CodingKey {
        case layerNormEps = "layer_norm_eps"
        case maxTrainedPositions = "max_trained_positions"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case modelType = "model_type"
    }

    /// Standard BERT naming conventions in JSON.
    enum BertCodingKeys: String, CodingKey {
        case embedDim = "hidden_size"
        case numHeads = "num_attention_heads"
        case interDim = "intermediate_size"
        case numLayers = "num_hidden_layers"
        case typeVocabularySize = "type_vocab_size"
    }

    /// DistilBERT naming conventions in JSON.
    enum DistilBertCodingKeys: String, CodingKey {
        case embedDim = "dim"
        case numLayers = "n_layers"
        case numHeads = "n_heads"
        case interDim = "hidden_dim"
    }

    /// Custom initializer to bridge different JSON schemas into a unified struct.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        // Load common properties
        layerNormEps = try container.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-12
        maxTrainedPositions =
            try container.decodeIfPresent(Int.self, forKey: .maxTrainedPositions) ?? 2048
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 30528
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 0
        modelType = try container.decode(String.self, forKey: .modelType)

        // Switch decoding logic based on model type
        if modelType == "distilbert" {
            let distilBertConfig = try decoder.container(keyedBy: DistilBertCodingKeys.self)
            embedDim = try distilBertConfig.decodeIfPresent(Int.self, forKey: .embedDim) ?? 768
            numHeads = try distilBertConfig.decodeIfPresent(Int.self, forKey: .numHeads) ?? 12
            interDim = try distilBertConfig.decodeIfPresent(Int.self, forKey: .interDim) ?? 3072
            numLayers = try distilBertConfig.decodeIfPresent(Int.self, forKey: .numLayers) ?? 12
            typeVocabularySize = 0  // DistilBERT does not use segment embeddings
        } else {
            let bertConfig = try decoder.container(keyedBy: BertCodingKeys.self)
            embedDim = try bertConfig.decodeIfPresent(Int.self, forKey: .embedDim) ?? 768
            numHeads = try bertConfig.decodeIfPresent(Int.self, forKey: .numHeads) ?? 12
            interDim = try bertConfig.decodeIfPresent(Int.self, forKey: .interDim) ?? 3072
            numLayers = try bertConfig.decodeIfPresent(Int.self, forKey: .numLayers) ?? 12
            typeVocabularySize =
                try bertConfig.decodeIfPresent(Int.self, forKey: .typeVocabularySize) ?? 2
        }
    }
}
