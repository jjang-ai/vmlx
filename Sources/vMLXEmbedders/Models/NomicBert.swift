// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import vMLXLMCommon
import MLXNN

// MARK: - NomicEmbedding

/// The embedding layer for the Nomic BERT architecture.
///
/// This module is responsible for converting input token indices into dense vectors.
/// It combines three distinct embedding types:
/// 1. **Word Embeddings**: The vector representation of the specific token.
/// 2. **Token Type Embeddings** (Optional): Vectors representing segment IDs (e.g., Sentence A vs Sentence B).
/// 3. **Position Embeddings** (Optional): Vectors representing the absolute position of the token in the sequence.
///
/// The final output is the element-wise sum of these components, followed by Layer Normalization.
class NomicEmbedding: Module {

    /// The size of the vocabulary for token types (segment IDs).
    let typeVocabularySize: Int

    /// The lookup table for word embeddings.
    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding

    /// The layer normalization applied to the sum of all embeddings.
    @ModuleInfo(key: "norm") var norm: LayerNorm

    /// The optional lookup table for token type (segment) embeddings.
    @ModuleInfo(key: "token_type_embeddings") var tokenTypeEmbeddings: Embedding?

    /// The optional lookup table for position embeddings.
    @ModuleInfo(key: "position_embeddings") var positionEmbeddings: Embedding?

    /// Initializes the Nomic embedding layer with a specific configuration.
    ///
    /// - Parameter config: The `NomicBertConfiguration` containing vocabulary sizes, dimensions, and flags.
    init(_ config: NomicBertConfiguration) {
        typeVocabularySize = config.typeVocabularySize

        // Initialize Word Embeddings
        _wordEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.embedDim
        )

        // Initialize Layer Normalization
        _norm.wrappedValue = LayerNorm(
            dimensions: config.embedDim,
            eps: config.layerNormEps
        )

        // Conditionally initialize Token Type Embeddings
        if config.typeVocabularySize > 0 {
            _tokenTypeEmbeddings.wrappedValue = Embedding(
                embeddingCount: config.typeVocabularySize,
                dimensions: config.embedDim
            )
        }

        // Conditionally initialize Position Embeddings
        if config.maxPositionEmbeddings > 0 {
            _positionEmbeddings.wrappedValue = Embedding(
                embeddingCount: config.maxPositionEmbeddings,
                dimensions: config.embedDim
            )
        }
    }

    /// Performs the forward pass of the embedding layer.
    ///
    /// The calculation corresponds to:
    /// $$ Output = \text{LayerNorm}(E_{word} + E_{type} + E_{pos}) $$
    ///
    /// - Parameters:
    ///   - inputIds: A tensor of shape `[batch_size, sequence_length]` containing the input token indices.
    ///   - positionIds: An optional tensor of the same shape as `inputIds`. If `nil`, positions are automatically generated as a sequence `0..<sequence_length`.
    ///   - tokenTypeIds: An optional tensor of the same shape as `inputIds` indicating segment IDs.
    /// - Returns: A tensor of shape `[batch_size, sequence_length, embed_dim]`.
    func callAsFunction(
        _ inputIds: MLXArray,
        positionIds: MLXArray? = nil,
        tokenTypeIds: MLXArray? = nil
    ) -> MLXArray {
        // 1. Retrieve Word Embeddings
        var words = wordEmbeddings(inputIds)

        // 2. Add Token Type Embeddings (if applicable)
        if let tokenTypeIds, let tokenTypeEmbeddings {
            words += tokenTypeEmbeddings(tokenTypeIds)
        }

        // 3. Determine Position IDs
        // If specific position IDs are not provided, we create a range [0, 1, ... seq_len]
        // and broadcast it to match the batch shape.
        let positions =
            positionIds ?? broadcast(MLXArray.arange(inputIds.dim(1)), to: inputIds.shape)

        // 4. Add Position Embeddings (if applicable)
        if let positionEmbeddings {
            words += positionEmbeddings(positions)
        }

        // 5. Apply Normalization
        return norm(words)
    }
}

// MARK: - MLP

/// A Multi-Layer Perceptron (MLP) implementing the SwiGLU activation mechanism.
///
/// This layer serves as the feed-forward block in the transformer architecture.
/// Unlike traditional MLPs (Linear -> ReLU -> Linear), this implementation uses a Gated Linear Unit
/// with the SiLU (Swish) activation function.
///
/// The forward pass corresponds to:
/// $$ Output = \text{Down}(\text{Up}(x) \odot \text{SiLU}(\text{Gate}(x))) $$
private class MLP: Module, UnaryLayer {

    /// The "up" projection layer (creates the values).
    /// Projects from `embedDim` to `hiddenFeatures`.
    @ModuleInfo(key: "fc11") var up: Linear

    /// The "gate" projection layer (creates the attention mask/filter).
    /// Projects from `embedDim` to `hiddenFeatures`.
    @ModuleInfo(key: "fc12") var gate: Linear

    /// The "down" projection layer.
    /// Projects from `hiddenFeatures` back to `embedDim`.
    @ModuleInfo(key: "fc2") var down: Linear

    /// Calculates the hidden dimension size, ensuring it is a multiple of 256.
    ///
    /// This rounding strategy is often used to optimize memory alignment and
    /// tensor core utilization on GPUs/NPUs.
    ///
    /// - Parameter config: The model configuration.
    /// - Returns: The hidden dimension rounded up to the nearest multiple of 256.
    private static func scaledHiddenFeatures(config: NomicBertConfiguration) -> Int {
        let multipleOf = 256
        let hiddenFeatures: Int = config.MLPDim
        // Formula: ceil(hidden / 256) * 256
        return (hiddenFeatures + multipleOf - 1) / multipleOf * multipleOf
    }

    /// Initializes the MLP layer.
    ///
    /// - Parameter config: Configuration defining embedding dimensions and bias settings.
    init(_ config: NomicBertConfiguration) {
        let hiddenFeatures = MLP.scaledHiddenFeatures(config: config)

        _up.wrappedValue = Linear(
            config.embedDim, hiddenFeatures, bias: config.mlpFc1Bias)

        _gate.wrappedValue = Linear(
            config.embedDim, hiddenFeatures, bias: config.mlpFc1Bias)

        _down.wrappedValue = Linear(
            hiddenFeatures, config.embedDim, bias: config.mlpFc2Bias)
    }

    /// Performs the SwiGLU forward pass.
    ///
    /// 1. Project input `x` through `up` layer.
    /// 2. Project input `x` through `gate` layer and apply SiLU activation.
    /// 3. Multiply the results element-wise (Hadamard product).
    /// 4. Project the result through `down` layer.
    func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        let activations = up(inputs) * silu(gate(inputs))
        return down(activations)
    }
}

func computeBaseFrequency(
    base: Float,
    dims: Int,
    ropeType: String,
    ropeScaling: [String: StringOrNumber]?
) -> Float {
    if ropeType != "llama3" {
        return base
    }

    guard let ropeScaling = ropeScaling else {
        return base
    }

    guard let factor = ropeScaling["factor"]?.asFloat() else {
        return base
    }

    let lowFreqFactor = ropeScaling["low_freq_factor"]?.asFloat() ?? 1.0
    let highFreqFactor = ropeScaling["high_freq_factor"]?.asFloat() ?? 4.0
    let oldContextLen = ropeScaling["original_max_position_embeddings"]?.asFloat() ?? 8192

    let lowFreqWavelen = oldContextLen / lowFreqFactor
    let highFreqWavelen = oldContextLen / highFreqFactor

    let freqs = (0 ..< dims).compactMap { index -> Float? in
        if index % 2 == 0 {
            return pow(base, Float(index) / Float(dims))
        }
        return nil
    }

    let newBaseFreqs = freqs.map { freq -> Float in
        let wavelen = 2 * .pi / freq
        let smooth = max(
            0,
            min(
                1,
                (wavelen - highFreqWavelen) / (lowFreqWavelen - highFreqWavelen)
            ))
        return freq * ((1 - smooth) * factor + smooth)
    }

    return newBaseFreqs.reduce(0, +) / Float(newBaseFreqs.count)
}

// MARK: - DynamicNTKScalingRoPE

/// A Rotary Positional Embedding (RoPE) layer that implements Dynamic NTK Scaling.
///
/// This module applies rotational embeddings to the input tensor. Crucially, it dynamically
/// adjusts the `base` frequency when the sequence length exceeds `maxPositionEmbeddings`.
/// This scaling allows the model to extrapolate to longer context windows effectively by
/// interpolating the position indices.
///
/// - Note: The scaling logic follows the formula:
///   $$ \text{base}' = \text{base} \cdot \left(\frac{L_{seq}}{L_{max}}\right)^{\frac{d}{d-2}} $$
private class DynamicNTKScalingRoPE: Module {

    /// The dimension of the embedding vector to rotate (usually `head_dim`).
    let dims: Int

    /// The maximum sequence length the model was trained on (e.g., 2048 or 4096).
    /// Used as the threshold to trigger dynamic scaling.
    let maxPositionEmbeddings: Int?

    /// Whether to use traditional RoPE (interleaved) or standard ordering.
    let traditional: Bool

    /// The base frequency (theta), typically 10000.0 or 1000000.0.
    let base: Float

    /// The linear scaling factor applied to the frequency.
    var scale: Float

    /// The type of RoPE scaling configuration (e.g., "dynamic", "linear").
    let ropeType: String

    /// Dictionary containing scaling configuration details.
    let ropeScaling: [String: StringOrNumber]?

    /// Initializes the RoPE layer with Dynamic NTK scaling capabilities.
    ///
    /// - Parameters:
    ///   - dims: The dimensionality of the heads.
    ///   - maxPositionEmbeddings: The training context limit.
    ///   - traditional: If `true`, applies rotation to adjacent pairs; otherwise splits the tensor in half.
    ///   - base: The base frequency for the rotation calculations.
    ///   - scale: A constant scaling factor (usually 1.0 unless static scaling is used).
    ///   - ropeType: The specific scaling strategy identifier.
    ///   - ropeScaling: Configuration dictionary for scaling parameters.
    init(
        dims: Int,
        maxPositionEmbeddings: Int?,
        traditional: Bool = false,
        base: Float = 10000,
        scale: Float = 1.0,
        ropeType: String = "default",
        ropeScaling: [String: StringOrNumber]? = nil
    ) {
        self.dims = dims
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.traditional = traditional
        // Note: computeBaseFrequency handles parsing 'ropeScaling' to adjust the initial base if needed.
        self.base = computeBaseFrequency(
            base: base, dims: dims, ropeType: ropeType, ropeScaling: ropeScaling
        )
        self.scale = scale
        self.ropeType = ropeType
        self.ropeScaling = ropeScaling
    }

    /// Applies the Rotary Positional Embeddings to the input tensor.
    ///
    /// If the current sequence length (`x.dim(1) + offset`) exceeds `maxPositionEmbeddings`,
    /// the function dynamically scales the `base` frequency to compress the relative positions
    /// back into the trained distribution range.
    ///
    /// - Parameters:
    ///   - x: Input tensor of shape `[batch, sequence_length, heads, head_dim]`.
    ///   - offset: The starting position index (useful for KV-caching during generation).
    /// - Returns: The input tensor with rotary embeddings applied.
    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        let seqLen = x.dim(1) + offset
        var base = self.base

        // Dynamic NTK Scaling Logic
        if let maxPositionEmbeddings, seqLen > maxPositionEmbeddings {
            // Calculate the ratio of current length to max trained length
            let factorAdjustment = Float(seqLen) / Float(maxPositionEmbeddings) - 1

            // Calculate the exponent for NTK scaling: d / (d - 2)
            let dimensionRatio = Float(dims) / Float(Float(dims) - 2)

            // Calculate the new scaling factor
            let adjustedScale = scale * pow(1 + factorAdjustment, dimensionRatio)

            // Apply scaling to the base frequency
            base *= adjustedScale
        }

        // Apply the rotation using the (potentially adjusted) base.
        return MLXFast.RoPE(
            x,
            dimensions: dims,
            traditional: traditional,
            base: base,
            scale: scale,
            offset: offset
        )
    }
}

// MARK: - Attention

/// A Multi-Head Self-Attention layer with fused QKV projection and Rotary Positional Embeddings.
///
/// This module computes the attention mechanism which allows the model to weigh the importance
/// of different tokens in the sequence relative to each other.
///
/// The process involves:
/// 1. Projecting the input into Query (Q), Key (K), and Value (V) vectors.
/// 2. Applying Rotary Positional Embeddings (RoPE) to Q and K.
/// 3. Computing scaled dot-product attention scores.
/// 4. Applying a softmax normalization and weighting the Values.
/// 5. Projecting the final output back to the embedding dimension.
private class Attention: Module {

    /// The number of parallel attention heads.
    let numHeads: Int

    /// The dimensionality of a single attention head (`embedDim / numHeads`).
    let headDim: Int

    /// A fused linear layer that projects the input into Queries, Keys, and Values simultaneously.
    /// Output shape: `[batch, seq_len, 3 * embed_dim]`
    @ModuleInfo(key: "Wqkv") var wqkv: Linear

    /// The final output projection layer.
    /// Input shape: `[batch, seq_len, embed_dim]`
    @ModuleInfo(key: "out_proj") var wo: Linear

    /// A wrapper enum to abstract the specific type of Rotary Positional Encoding being used.
    /// This allows the layer to support both standard RoPE and Dynamic NTK-scaled RoPE
    /// seamlessly.
    enum PositionalEncoding {
        case rope(RoPE)
        case dynamicNTKScalingRoPE(DynamicNTKScalingRoPE)

        /// Applies the positional encoding to the input tensor.
        func applyEncoding(_ x: MLXArray, offset: Int = 0) -> MLXArray {
            switch self {
            case .rope(let rope):
                return rope.callAsFunction(x, offset: offset)
            case .dynamicNTKScalingRoPE(let dynamicNTKScalingRoPE):
                return dynamicNTKScalingRoPE.callAsFunction(x, offset: offset)
            }
        }
    }

    /// The selected positional encoding strategy.
    let rope: PositionalEncoding

    /// The portion of the head dimension to apply rotary embeddings to.
    /// (Some architectures only rotate the first half or partial percentage of the vector).
    let rotaryEmbDim: Int

    /// The scaling factor for the dot product ($\sqrt{d_{head}}$).
    /// Used to normalize gradients and prevent softmax saturation.
    let normFactor: Float

    /// Initializes the Attention layer.
    ///
    /// - Parameter config: The configuration defining dimensions, scaling factors, and RoPE settings.
    init(_ config: NomicBertConfiguration) {
        // Fused QKV Projection: Projects to 3x embedding dimension
        _wqkv.wrappedValue = Linear(
            config.embedDim, 3 * config.embedDim, bias: config.qkvProjBias)

        // Output Projection
        _wo.wrappedValue = Linear(
            config.embedDim, config.embedDim, bias: config.qkvProjBias)

        numHeads = config.numHeads
        headDim = config.embedDim / numHeads
        rotaryEmbDim = Int(Float(headDim) * config.rotaryEmbFraction)
        normFactor = sqrt(Float(headDim))

        // Determine which RoPE implementation to use based on configuration
        if config.rotaryScalingFactor != nil {
            rope = .dynamicNTKScalingRoPE(
                DynamicNTKScalingRoPE(
                    dims: rotaryEmbDim,
                    maxPositionEmbeddings: config.maxPositionEmbeddings,
                    traditional: config.rotaryEmbInterleaved,
                    base: config.rotaryEmbBase,
                    scale: config.rotaryScalingFactor!))
        } else {
            rope = .rope(
                RoPE(
                    dimensions: rotaryEmbDim,
                    traditional: config.rotaryEmbInterleaved,
                    base: config.rotaryEmbBase,
                    scale: 1.0)
            )
        }
    }

    /// Performs the forward pass of the attention mechanism.
    ///
    /// The calculation follows the standard self-attention formula:
    /// $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V $$
    ///
    /// - Parameters:
    ///   - inputs: Input tensor of shape `[batch, seq_len, embed_dim]`.
    ///   - mask: Optional attention mask (e.g., causal mask or padding mask).
    ///           Added to the scores before softmax.
    /// - Returns: Contextualized output tensor of shape `[batch, seq_len, embed_dim]`.
    func callAsFunction(_ inputs: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let (B, L) = (inputs.dim(0), inputs.dim(1))

        // 1. Fused Projection: Project inputs to combined QKV
        // 2. Split: Separate the combined tensor into Query, Key, and Value tensors
        let queryPos = numHeads * headDim
        let qkv = split(
            wqkv(inputs), indices: [queryPos, queryPos * 2], axis: -1
        )
        var queries = qkv[0]
        var keys = qkv[1]
        var values = qkv[2]

        // 3. Reshape and Transpose for Multi-Head calculation
        // Transform from [B, L, hidden] -> [B, L, heads, head_dim] -> [B, heads, L, head_dim]
        queries = queries.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)

        // 4. Apply Rotary Positional Embeddings (if configured)
        // RoPE injects position information directly into Q and K by rotating the vectors
        if rotaryEmbDim > 0 {
            queries = rope.applyEncoding(queries)
            keys = rope.applyEncoding(keys)
        }

        // 5. Scaled Dot-Product Attention
        // Calculate raw attention scores: Q * K^T
        var scores = queries.matmul(keys.transposed(0, 1, 3, 2)) / normFactor

        // 6. Apply Masking (if provided)
        if let mask {
            scores = scores + mask
        }

        // 7. Softmax Normalization
        let probs = softmax(scores, axis: -1)

        // 8. Aggregate Values
        // Weighted sum of Values: probs * V
        // Reshape back to [B, L, hidden]
        let output = matmul(probs, values).transposed(0, 2, 1, 3).reshaped(B, L, -1)

        // 9. Final Output Projection
        return wo(output)
    }
}

// MARK: - TransformerBlock

/// A single Transformer Encoder block.
///
/// This module composes the attention mechanism and the feed-forward network (MLP)
/// into a residual block. It follows the "Post-Norm" architecture pattern standard in BERT models:
/// 1. Self-Attention with Residual Connection and Layer Normalization.
/// 2. MLP (Feed-Forward) with Residual Connection and Layer Normalization.
///
/// The mathematical flow is:
/// $$ H_1 = \text{LayerNorm}(x + \text{Attention}(x)) $$
/// $$ Output = \text{LayerNorm}(H_1 + \text{MLP}(H_1)) $$
private class TransformerBlock: Module {

    /// The Multi-Head Self-Attention layer.
    @ModuleInfo(key: "attn") var attention: Attention

    /// Layer Normalization applied after the self-attention block.
    @ModuleInfo(key: "norm1") var postAttentionLayerNorm: LayerNorm

    /// Layer Normalization applied after the MLP block.
    @ModuleInfo(key: "norm2") var outputLayerNorm: LayerNorm

    /// The Feed-Forward Network (SwiGLU).
    @ModuleInfo(key: "mlp") var mlp: MLP

    /// Initializes the Transformer block.
    ///
    /// - Parameter config: Configuration defining dimensions and layer properties.
    init(_ config: NomicBertConfiguration) {
        _attention.wrappedValue = Attention(config)
        _mlp.wrappedValue = MLP(config)

        // Initialize LayerNorms with embedding dimension and epsilon
        _outputLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.embedDim, eps: config.layerNormEps)
        _postAttentionLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.embedDim, eps: config.layerNormEps)
    }

    /// Performs the forward pass of the Transformer block.
    ///
    /// - Parameters:
    ///   - inputs: Input tensor of shape `[batch, seq_len, embed_dim]`.
    ///   - mask: Optional attention mask to prevent attending to padding or future tokens.
    /// - Returns: Processed tensor of the same shape as `inputs`.
    func callAsFunction(_ inputs: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        // 1. Self-Attention
        let attentionOut = attention(inputs, mask: mask)

        // 2. Add & Norm (Post-Attention)
        // Residual connection: inputs + attention output
        let addAndNorm = postAttentionLayerNorm(attentionOut + inputs)

        // 3. Feed-Forward Network (MLP)
        let mlpOut = mlp(addAndNorm)

        // 4. Add & Norm (Output)
        // Residual connection: (attn_output) + mlp output
        return outputLayerNorm(addAndNorm + mlpOut)
    }
}

// MARK: - LMHead

/// The Language Model Head (Prediction Layer).
///
/// This module transforms the final hidden states from the transformer encoder into
/// probability scores (logits) for every token in the vocabulary. It is typically used
/// during pre-training for Masked Language Modeling (MLM) or as the output layer for
/// generation tasks.
///
/// The forward pass consists of a non-linear transformation followed by a projection:
/// $$ \text{Logits} = \text{Decoder}(\text{LayerNorm}(\text{SiLU}(\text{Dense}(x)))) $$
private class LMHead: Module {

    /// A dense projection layer that transforms the embedding space before prediction.
    /// Shape: `[embed_dim, embed_dim]`
    @ModuleInfo(key: "dense") var dense: Linear

    /// Layer normalization applied after the activation function.
    @ModuleInfo(key: "ln") var layerNorm: LayerNorm

    /// The final output decoder that projects embeddings to vocabulary logits.
    /// Shape: `[embed_dim, vocab_size]`
    @ModuleInfo(key: "decoder") var decoder: Linear

    /// Initializes the Head.
    ///
    /// - Parameter config: Configuration defining dimensions and vocabulary size.
    init(_ config: NomicBertConfiguration) {
        // 1. Transform layer
        _dense.wrappedValue = Linear(
            config.embedDim, config.embedDim, bias: config.mlpFc1Bias)

        // 2. Normalization layer
        _layerNorm.wrappedValue = LayerNorm(
            dimensions: config.embedDim, eps: config.layerNormEps)

        // 3. Output projection (Vocab projection)
        _decoder.wrappedValue = Linear(
            config.embedDim, config.vocabularySize, bias: config.mlpFc1Bias)
    }

    /// Performs the forward pass to generate logits.
    ///
    /// - Parameter inputs: The output from the last Transformer Block.
    ///   Shape: `[batch, seq_len, embed_dim]`
    /// - Returns: The unnormalized log-probabilities (logits) for each token.
    ///   Shape: `[batch, seq_len, vocab_size]`
    func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        // Apply dense projection -> Activation (SiLU) -> Normalization -> Decoder
        return decoder(layerNorm(silu(dense(inputs))))
    }
}

// MARK: - Encoder

/// The main Transformer Encoder stack.
///
/// This module consists of a sequence of `TransformerBlock` layers. It takes the
/// initial embeddings as input and progressively refines them through self-attention
/// and feed-forward processing.
///
/// The output of this module represents the contextualized representation of the
/// input sequence, often referred to as the "last hidden state".
private class Encoder: Module {

    /// The stack of transformer layers.
    ///
    /// MLX will automatically register these as submodules, ensuring their parameters
    /// are included in the model's parameter dictionary.
    let layers: [TransformerBlock]

    /// Initializes the encoder stack.
    ///
    /// - Parameter config: Configuration defining the number of layers (`numLayers`)
    ///                     and other architectural hyperparameters.
    init(
        _ config: NomicBertConfiguration
    ) {
        // Sanity check to ensure the configuration is valid before building layers.
        precondition(config.vocabularySize > 0)

        // Initialize the specified number of transformer blocks.
        // The array order corresponds to the vertical depth of the network (Layer 0 -> Layer N).
        layers = (0 ..< config.numLayers).map {
            _ in TransformerBlock(config)
        }
    }

    /// Performs the forward pass through all transformer layers.
    ///
    /// - Parameters:
    ///   - inputs: The output from the embedding layer.
    ///             Shape: `[batch, seq_len, embed_dim]`
    ///   - attentionMask: An optional mask to prevent attention to specific tokens (e.g., padding).
    ///                    This mask is broadcast to all layers.
    /// - Returns: The final hidden states from the last transformer layer.
    ///            Shape: `[batch, seq_len, embed_dim]`
    func callAsFunction(_ inputs: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        var outputs = inputs

        // Sequentially pass the output of one layer as the input to the next.
        // The attention mask remains constant across all layers.
        for layer in layers {
            outputs = layer(outputs, mask: attentionMask)
        }

        return outputs
    }
}

// MARK: - NomicBertModel

/// The main entry point for the Nomic BERT model.
///
/// This class serves as the container for the entire architecture. It connects the
/// embeddings, the encoder stack, and the optional output heads (Pooler or LM Head).
/// It also handles weight key sanitization to ensure compatibility with checkpoints
/// trained in other frameworks (e.g., PyTorch / Hugging Face).
public class NomicBertModel: Module, EmbeddingModel {

    /// The optional Language Model head.
    /// Used for Masked Language Modeling (MLM) or token prediction.
    @ModuleInfo(key: "lm_head") fileprivate var lmHead: LMHead?

    /// The embedding layer responsible for token, position, and segment embeddings.
    @ModuleInfo(key: "embeddings") var embedder: NomicEmbedding

    /// The optional pooler layer.
    /// Used to extract a single vector representation for the whole sequence (usually from the [CLS] token).
    @ModuleInfo var pooler: Linear?

    /// The stack of Transformer blocks.
    fileprivate let encoder: Encoder

    /// The size of the vocabulary.
    public var vocabularySize: Int

    /// Initializes the Nomic BERT model.
    ///
    /// - Parameters:
    ///   - config: The model configuration.
    ///   - pooler: If `true`, initializes a linear pooling layer for sequence classification/embedding tasks.
    ///   - lmHead: If `true`, initializes the language modeling head for token prediction.
    public init(
        _ config: NomicBertConfiguration, pooler: Bool = true,
        lmHead: Bool = false
    ) {
        precondition(config.vocabularySize > 0)
        vocabularySize = config.vocabularySize
        encoder = Encoder(config)
        _embedder.wrappedValue = NomicEmbedding(config)

        // Initialize Pooler (for sentence embeddings)
        if pooler {
            _pooler.wrappedValue = Linear(config.embedDim, config.embedDim, bias: false)
        } else {
            _pooler.wrappedValue = nil
        }

        // Initialize LM Head (for training/masked prediction)
        if lmHead {
            _lmHead.wrappedValue = LMHead(config)
        }
    }

    /// Performs the full forward pass of the model.
    ///
    /// This method handles:
    /// 1. Input reshaping (ensuring batch dimension exists).
    /// 2. Mask transformation (converting binary masks to additive log-masks).
    /// 3. Embedding -> Encoder flow.
    /// 4. Output formatting (Logic for returning Pooling vs LM Head outputs).
    ///
    /// - Parameters:
    ///   - inputs: Input token indices. Shape `[batch, seq_len]`.
    ///   - positionIds: Optional specific position indices.
    ///   - tokenTypeIds: Optional segment IDs.
    ///   - attentionMask: A binary mask (1 for keep, 0 for discard).
    /// - Returns: An `EmbeddingModelOutput` containing hidden states and optionally the pooled output.
    public func callAsFunction(
        _ inputs: MLXArray,
        positionIds: MLXArray? = nil,
        tokenTypeIds: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    ) -> EmbeddingModelOutput {
        var inp = inputs

        // 1. Un-squeeze input if it lacks a batch dimension (e.g., shape [seq_len] -> [1, seq_len])
        if inp.ndim == 1 {
            inp = inp.reshaped(1, -1)
        }

        // 2. Process Attention Mask
        // Input: Binary mask (1 = valid, 0 = mask).
        // Operation: .log().
        //   log(1) = 0    (Add 0 to attention score -> No change)
        //   log(0) = -inf (Add -inf to attention score -> Zero probability after Softmax)
        let embeddings = embedder(
            inp, positionIds: positionIds, tokenTypeIds: tokenTypeIds)
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

        // 3. Encoder Pass
        let outputs = encoder(embeddings, attentionMask: mask)

        // 4a. Return LM Head Output (if active)
        if let lmHead {
            return EmbeddingModelOutput(hiddenStates: lmHead(outputs), pooledOutput: nil)
        }

        // 4b. Return Pooled Output (if active)
        // Takes the first token (index 0, usually [CLS]), projects it, and applies Tanh.
        if let pooler {
            return EmbeddingModelOutput(
                hiddenStates: outputs, pooledOutput: tanh(pooler(outputs[0..., 0])))
        }

        // 4c. Return Raw Hidden States
        return EmbeddingModelOutput(hiddenStates: outputs, pooledOutput: nil)
    }

    /// Remaps weight keys from external formats (e.g., Hugging Face) to this model's internal structure.
    ///
    /// This is critical when loading `.safetensors` or PyTorch checkpoints, as variable names often differ.
    ///
    /// - Parameter weights: A dictionary of loaded weights.
    /// - Returns: A new dictionary with keys renamed to match this Swift class structure.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.reduce(into: [:]) { result, item in
            let key = item.key
                .replacingOccurrences(of: "emb_ln", with: "embeddings.norm")
                .replacingOccurrences(of: "bert.", with: "")  // Remove namespace prefix
                // Remap LM Head keys
                .replacingOccurrences(
                    of: "cls.predictions.transform.dense.", with: "lm_head.dense."
                )
                .replacingOccurrences(
                    of: "cls.predictions.transform.LayerNorm.", with: "lm_head.ln."
                )
                .replacingOccurrences(of: "cls.predictions.decoder", with: "lm_head.decoder")
                // Remap Pooler keys
                .replacingOccurrences(of: "pooler.dense.", with: "pooler.")

            result[key] = item.value
        }
    }
}

// MARK: - NomicBertConfiguration

/// The configuration object defining the hyperparameters and architectural details of Nomic BERT.
///
/// This struct conforms to `Decodable` to allow direct initialization from a JSON configuration file.
/// It includes robust default values for many properties, ensuring backward compatibility with
/// various model checkpoints.
public struct NomicBertConfiguration: Decodable, Sendable {

    // MARK: - Normalization & Optimization

    /// The epsilon value used for numerical stability in Layer Normalization.
    /// Added to the variance to prevent division by zero: $\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$
    var layerNormEps: Float = 1e-12

    /// The maximum sequence length the model was explicitly trained on.
    /// This is used as the reference length ($L_{train}$) for Dynamic NTK scaling calculations.
    var maxTrainedPositions: Int = 2048

    // MARK: - Bias Flags

    /// Whether to include a bias term in the first linear layer of the MLP (Up/Gate projections).
    var mlpFc1Bias: Bool = false

    /// Whether to include a bias term in the second linear layer of the MLP (Down projection).
    var mlpFc2Bias: Bool = false

    /// Whether to include a bias term in the Query, Key, and Value projections in the Attention layer.
    var qkvProjBias: Bool = false

    // MARK: - Dimensions

    /// The dimensionality of the encoder layers and the pooler layer (hidden size).
    var embedDim: Int = 768

    /// The number of attention heads for each attention layer.
    var numHeads: Int = 12

    /// The dimensionality of the "intermediate" (hidden) layer in the Feed-Forward Network (MLP).
    var MLPDim: Int = 3072

    /// The number of hidden layers in the Transformer encoder.
    var numLayers: Int = 12

    // MARK: - Rotary Embeddings (RoPE)

    /// The base frequency ($\theta$) used for computing Rotary Positional Embeddings.
    /// Standard values are typically 10,000 or 1,000,000.
    var rotaryEmbBase: Float = 1000

    /// The fraction of the head dimension to apply rotary embeddings to.
    /// If 1.0, the entire head vector is rotated. If < 1.0, only the first $d \times fraction$ dimensions are rotated.
    var rotaryEmbFraction: Float = 1.0

    /// Determines the pairing strategy for the rotation.
    /// - `false`: Splits the head into two halves, pairing index $i$ with $i + d/2$.
    /// - `true`: Pairs adjacent indices $i$ with $i+1$.
    var rotaryEmbInterleaved: Bool = false

    /// Optional base for scaling the rotary embeddings (specific to Nomic's implementation).
    var rotaryEmbScaleBase: Float?

    /// The linear scaling factor applied to the RoPE frequency.
    /// Used for Dynamic NTK scaling to extend context length.
    var rotaryScalingFactor: Float?

    // MARK: - Vocabulary

    /// The size of the vocabulary for segment/token type embeddings.
    /// Typically 2 (Sentence A and Sentence B).
    var typeVocabularySize: Int = 2

    /// The size of the main token vocabulary.
    /// This dictates the input dimension of the embedding layer and the output dimension of the LM Head.
    var vocabularySize: Int = 30528

    /// The maximum length for absolute position embeddings.
    /// Note: Nomic BERT primarily uses RoPE, so this is often set to 0 or unused unless specific absolute embeddings are configured.
    var maxPositionEmbeddings: Int = 0

    // MARK: - Coding Keys

    /// Maps the snake_case keys found in standard JSON config files (e.g., from Hugging Face)
    /// to the camelCase property names used in this Swift struct.
    enum CodingKeys: String, CodingKey {
        case layerNormEps = "layer_norm_epsilon"
        case maxTrainedPositions = "max_trained_positions"
        case mlpFc1Bias = "mlp_fc1_bias"
        case mlpFc2Bias = "mlp_fc2_bias"
        case embedDim = "n_embd"
        case numHeads = "n_head"
        case MLPDim = "n_inner"
        case numLayers = "n_layer"
        case qkvProjBias = "qkv_proj_bias"
        case rotaryEmbBase = "rotary_emb_base"
        case rotaryEmbFraction = "rotary_emb_fraction"
        case rotaryEmbInterleaved = "rotary_emb_interleaved"
        case rotaryEmbScaleBase = "rotary_emb_scale_base"
        case rotaryScalingFactor = "rotary_scaling_factor"
        case typeVocabularySize = "type_vocab_size"
        case useCache = "use_cache"  // Present in JSON but unused in this struct
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    // MARK: - Decoder

    /// Decodes the configuration from an external representation (like JSON).
    ///
    /// This initializer manually decodes each property, providing fallback default values
    /// if specific keys are missing from the source data.
    public init(from decoder: Decoder) throws {
        let container: KeyedDecodingContainer<NomicBertConfiguration.CodingKeys> =
            try decoder.container(
                keyedBy: NomicBertConfiguration.CodingKeys.self)

        // Decoding with explicit default values for robustness
        layerNormEps =
            try container.decodeIfPresent(
                Float.self,
                forKey: NomicBertConfiguration.CodingKeys.layerNormEps.self)
            ?? 1e-12

        maxTrainedPositions =
            try container.decodeIfPresent(
                Int.self,
                forKey: NomicBertConfiguration.CodingKeys.maxTrainedPositions.self)
            ?? 2048

        mlpFc1Bias =
            try container.decodeIfPresent(
                Bool.self,
                forKey: NomicBertConfiguration.CodingKeys.mlpFc1Bias.self)
            ?? false

        mlpFc2Bias =
            try container.decodeIfPresent(
                Bool.self,
                forKey: NomicBertConfiguration.CodingKeys.mlpFc2Bias.self)
            ?? false

        embedDim =
            try container.decodeIfPresent(
                Int.self,
                forKey: NomicBertConfiguration.CodingKeys.embedDim.self) ?? 768

        numHeads =
            try container.decodeIfPresent(
                Int.self,
                forKey: NomicBertConfiguration.CodingKeys.numHeads.self) ?? 12

        MLPDim =
            try container.decodeIfPresent(
                Int.self, forKey: NomicBertConfiguration.CodingKeys.MLPDim.self)
            ?? 3072

        numLayers =
            try container.decodeIfPresent(
                Int.self,
                forKey: NomicBertConfiguration.CodingKeys.numLayers.self) ?? 12

        qkvProjBias =
            try container.decodeIfPresent(
                Bool.self,
                forKey: NomicBertConfiguration.CodingKeys.qkvProjBias.self)
            ?? false

        rotaryEmbBase =
            try container.decodeIfPresent(
                Float.self,
                forKey: NomicBertConfiguration.CodingKeys.rotaryEmbBase.self)
            ?? 1000

        rotaryEmbFraction =
            try container.decodeIfPresent(
                Float.self,
                forKey: NomicBertConfiguration.CodingKeys.rotaryEmbFraction.self
            ) ?? 1.0

        rotaryEmbInterleaved =
            try container.decodeIfPresent(
                Bool.self,
                forKey: NomicBertConfiguration.CodingKeys.rotaryEmbInterleaved.self)
            ?? false

        rotaryEmbScaleBase =
            try container.decodeIfPresent(
                Float.self,
                forKey: NomicBertConfiguration.CodingKeys.rotaryEmbScaleBase)
            ?? nil

        rotaryScalingFactor =
            try container.decodeIfPresent(
                Float.self,
                forKey: NomicBertConfiguration.CodingKeys.rotaryScalingFactor)
            ?? nil

        typeVocabularySize =
            try container.decodeIfPresent(
                Int.self,
                forKey: NomicBertConfiguration.CodingKeys.typeVocabularySize.self)
            ?? 2

        vocabularySize =
            try container.decodeIfPresent(
                Int.self,
                forKey: NomicBertConfiguration.CodingKeys.vocabularySize.self)
            ?? 30528

        maxPositionEmbeddings =
            try container.decodeIfPresent(
                Int.self,
                forKey: NomicBertConfiguration.CodingKeys.maxPositionEmbeddings.self)
            ?? 0
    }
}
