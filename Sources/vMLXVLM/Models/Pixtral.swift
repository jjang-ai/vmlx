import CoreImage
import Foundation
import MLX
import vMLXLMCommon
import MLXNN

// Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/pixtral

// MARK: - Vision Configuration

public struct PixtralVisionConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let intermediateSize: Int
    public let patchSize: Int
    public let imageSize: Int

    public var numChannels: Int { _numChannels ?? 3 }
    public var rmsNormEps: Float { _rmsNormEps ?? 1e-5 }
    public var headDim: Int { _headDim ?? (hiddenSize / numAttentionHeads) }
    public var ropeTheta: Float { _ropeTheta ?? 10000.0 }

    private let _numChannels: Int?
    private let _rmsNormEps: Float?
    private let _headDim: Int?
    private let _ropeTheta: Float?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case patchSize = "patch_size"
        case imageSize = "image_size"
        case _numChannels = "num_channels"
        case _rmsNormEps = "rms_norm_eps"
        case _headDim = "head_dim"
        case _ropeTheta = "rope_theta"
    }
}

// MARK: - Text Configuration (for Pixtral)

public struct PixtralTextConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let rmsNormEps: Float
    public let vocabSize: Int

    public var headDim: Int { _headDim ?? (hiddenSize / numAttentionHeads) }
    public var maxPositionEmbeddings: Int? { _maxPositionEmbeddings }
    public var numKeyValueHeads: Int { _numKeyValueHeads ?? numAttentionHeads }
    public var ropeTheta: Float { _ropeTheta ?? 1_000_000_000 }
    public var ropeTraditional: Bool { _ropeTraditional ?? false }
    public var ropeScaling: [String: StringOrNumber]? { _ropeScaling }
    public var tieWordEmbeddings: Bool { _tieWordEmbeddings ?? false }
    public var useQkNorm: Bool { _useQkNorm ?? false }

    private let _headDim: Int?
    private let _maxPositionEmbeddings: Int?
    private let _numKeyValueHeads: Int?
    private let _ropeTheta: Float?
    private let _ropeTraditional: Bool?
    private let _ropeScaling: [String: StringOrNumber]?
    private let _tieWordEmbeddings: Bool?
    private let _useQkNorm: Bool?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case _headDim = "head_dim"
        case _maxPositionEmbeddings = "max_position_embeddings"
        case _numKeyValueHeads = "num_key_value_heads"
        case _ropeTheta = "rope_theta"
        case _ropeTraditional = "rope_traditional"
        case _ropeScaling = "rope_scaling"
        case _tieWordEmbeddings = "tie_word_embeddings"
        case _useQkNorm = "use_qk_norm"
    }
}

// MARK: - Model Configuration

public struct PixtralConfiguration: Codable, Sendable {
    public let textConfig: PixtralTextConfiguration
    public let visionConfig: PixtralVisionConfiguration
    public let modelType: String

    public var ignoreIndex: Int { _ignoreIndex ?? -100 }
    public var imageTokenIndex: Int { _imageTokenIndex ?? _imageTokenId ?? 10 }
    public var visionFeatureSelectStrategy: String { _visionFeatureSelectStrategy ?? "full" }
    public var visionFeatureLayer: Int { _visionFeatureLayer ?? -1 }
    public var vocabSize: Int { _vocabSize ?? 32000 }

    private let _ignoreIndex: Int?
    private let _imageTokenIndex: Int?
    private let _imageTokenId: Int?
    private let _visionFeatureSelectStrategy: String?
    private let _visionFeatureLayer: Int?
    private let _vocabSize: Int?

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case modelType = "model_type"
        case _ignoreIndex = "ignore_index"
        case _imageTokenIndex = "image_token_index"
        case _imageTokenId = "image_token_id"
        case _visionFeatureSelectStrategy = "vision_feature_select_strategy"
        case _visionFeatureLayer = "vision_feature_layer"
        case _vocabSize = "vocab_size"
    }
}

// MARK: - Vision Model

/// Pixtral vision model namespace - contains all vision-related components
/// These are made internal so they can be reused by Mistral3
internal enum PixtralVision {

    static func checkArrayShape(_ arr: MLXArray) -> Bool {
        if arr.ndim != 4 { return false }
        let (o, h, w, _) = (arr.dim(0), arr.dim(1), arr.dim(2), arr.dim(3))
        return (o >= h && o >= w && h == w)
    }

    // MARK: Pixtral Rotary Embedding

    /// 2D Rotary position embeddings for vision transformer
    class RotaryEmbedding {
        let dim: Int
        let invFreq: MLXArray

        init(_ config: PixtralVisionConfiguration) {
            self.dim = config.headDim
            let base = config.ropeTheta
            let maxPatchesPerSide = config.imageSize / config.patchSize

            // Create base frequencies
            let freqs =
                1.0
                / MLX.pow(
                    MLXArray(base),
                    MLXArray(stride(from: 0, to: Float(dim), by: 2)).asType(.float32) / Float(dim)
                )

            // Create position grids
            let h = MLXArray(0 ..< maxPatchesPerSide)
            let w = MLXArray(0 ..< maxPatchesPerSide)

            // freqs has dim/2 elements, split by alternating indices for h and w (matching Python)
            let indicesEven = MLXArray(stride(from: 0, to: dim / 2, by: 2))
            let indicesOdd = MLXArray(stride(from: 1, to: dim / 2, by: 2))

            let freqsH = MLX.outer(h.asType(.float32), freqs[indicesEven])
            let freqsW = MLX.outer(w.asType(.float32), freqs[indicesOdd])

            // Tile and combine: create 2D position encodings
            // freqsH: (maxPatches, halfDim) -> tile to (maxPatches, maxPatches, halfDim)
            // freqsW: (maxPatches, halfDim) -> tile to (maxPatches, maxPatches, halfDim)
            let tiledH = MLX.tiled(
                freqsH[0..., .newAxis, 0...], repetitions: [1, maxPatchesPerSide, 1])
            let tiledW = MLX.tiled(
                freqsW[.newAxis, 0..., 0...], repetitions: [maxPatchesPerSide, 1, 1])

            // Concatenate and reshape to (maxPatches^2, dim/2)
            var invFreqTemp = MLX.concatenated([tiledH, tiledW], axis: -1)
            invFreqTemp = invFreqTemp.reshaped(-1, dim / 2)

            // Duplicate for full dim
            self.invFreq = MLX.concatenated([invFreqTemp, invFreqTemp], axis: -1)
        }

        func callAsFunction(_ x: MLXArray, positionIds: MLXArray) -> (cos: MLXArray, sin: MLXArray)
        {
            let freqs = invFreq[positionIds]
            let cos = MLX.cos(freqs).asType(x.dtype)
            let sin = MLX.sin(freqs).asType(x.dtype)
            return (cos, sin)
        }
    }

    /// Apply rotary position embeddings to queries and keys
    static func applyRotaryPosEmb(
        q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray
    ) -> (MLXArray, MLXArray) {
        // cos/sin shape: (seqLen, headDim), need to unsqueeze for heads dimension
        let cosExpanded = cos[.newAxis, 0..., 0...]  // (1, seqLen, headDim)
        let sinExpanded = sin[.newAxis, 0..., 0...]

        func rotateHalf(_ x: MLXArray) -> MLXArray {
            let halfDim = x.dim(-1) / 2
            let x1 = x[0..., 0..., 0..., 0 ..< halfDim]
            let x2 = x[0..., 0..., 0..., halfDim...]
            return MLX.concatenated([-x2, x1], axis: -1)
        }

        let qEmbed = (q * cosExpanded) + (rotateHalf(q) * sinExpanded)
        let kEmbed = (k * cosExpanded) + (rotateHalf(k) * sinExpanded)
        return (qEmbed, kEmbed)
    }

    /// Generate a block-diagonal attention mask so that separate images
    /// in the batch do not attend to each other.
    ///
    /// The input `patchCounts` should list the flattened patch counts per image.
    static func generateBlockAttentionMask(
        patchCounts: [Int],
        batchSize: Int,
        dtype: DType
    ) -> MLXArray {
        let seqLen = patchCounts.reduce(0, +)
        var mask = [Float](repeating: -1e9, count: seqLen * seqLen)

        var start = 0
        for count in patchCounts {
            let end = start + count
            for row in start ..< end {
                let rowOffset = row * seqLen
                for col in start ..< end {
                    mask[rowOffset + col] = 0
                }
            }
            start = end
        }

        var maskArray = MLXArray(mask).reshaped(seqLen, seqLen)
        maskArray = maskArray[.newAxis, .newAxis, 0..., 0...]
        let broadcasted = broadcast(maskArray, to: [batchSize, 1, seqLen, seqLen])
        return broadcasted.asType(dtype)
    }

    /// Generate position IDs in a meshgrid pattern for patches
    static func positionIdsInMeshgrid(patchHeight: Int, patchWidth: Int, maxWidth: Int) -> MLXArray
    {
        var positions: [Int32] = []
        for h in 0 ..< patchHeight {
            for w in 0 ..< patchWidth {
                positions.append(Int32(h * maxWidth + w))
            }
        }
        return MLXArray(positions)
    }

    // MARK: Vision Attention (Pixtral-style with RoPE)

    class Attention: Module {
        let numHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        init(_ config: PixtralVisionConfiguration) {
            self.numHeads = config.numAttentionHeads
            self.headDim = config.headDim
            self.scale = pow(Float(headDim), -0.5)

            self._qProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
            self._kProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
            self._vProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
            self._oProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        }

        func callAsFunction(
            _ x: MLXArray,
            positionEmbeddings: (cos: MLXArray, sin: MLXArray),
            mask: MLXArray? = nil
        ) -> MLXArray {
            let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

            var queries = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
            var keys = kProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
            let values = vProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)

            // Apply rotary position embeddings
            (queries, keys) = PixtralVision.applyRotaryPosEmb(
                q: queries, k: keys,
                cos: positionEmbeddings.cos,
                sin: positionEmbeddings.sin
            )

            // Scaled dot product attention
            var attnWeights = MLX.matmul(queries, keys.transposed(0, 1, 3, 2)) * scale

            if let mask {
                attnWeights = attnWeights + mask
            }

            attnWeights = softmax(attnWeights, axis: -1)
            let output = MLX.matmul(attnWeights, values)

            return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
        }
    }

    // MARK: Vision MLP (Pixtral-style with SiLU gating)

    class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear

        init(_ config: PixtralVisionConfiguration) {
            self._gateProj.wrappedValue = Linear(
                config.hiddenSize, config.intermediateSize, bias: false)
            self._downProj.wrappedValue = Linear(
                config.intermediateSize, config.hiddenSize, bias: false)
            self._upProj.wrappedValue = Linear(
                config.hiddenSize, config.intermediateSize, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(silu(gateProj(x)) * upProj(x))
        }
    }

    // MARK: Vision Encoder Layer (Pixtral-style)

    class EncoderLayer: Module {
        @ModuleInfo(key: "attention") var attention: Attention
        @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
        @ModuleInfo(key: "feed_forward") var feedForward: MLP
        @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

        init(_ config: PixtralVisionConfiguration) {
            self._attention.wrappedValue = Attention(config)
            self._attentionNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._feedForward.wrappedValue = MLP(config)
            self._ffnNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(
            _ x: MLXArray,
            positionEmbeddings: (cos: MLXArray, sin: MLXArray),
            mask: MLXArray? = nil
        ) -> MLXArray {
            let y = attentionNorm(x)
            let attnOut = attention(y, positionEmbeddings: positionEmbeddings, mask: mask)
            let h = x + attnOut
            let ffnOut = feedForward(ffnNorm(h))
            return h + ffnOut
        }
    }

    // MARK: Vision Encoder

    class Encoder: Module {
        var layers: [EncoderLayer]

        init(_ config: PixtralVisionConfiguration) {
            self.layers = (0 ..< config.numHiddenLayers).map { _ in EncoderLayer(config) }
        }
    }

    // MARK: Pixtral Vision Model (inner model with patch_conv, ln_pre, transformer)

    class PixtralVisionModelInner: Module {
        @ModuleInfo(key: "patch_conv") var patchConv: Conv2d
        @ModuleInfo(key: "ln_pre") var lnPre: RMSNorm
        @ModuleInfo(key: "transformer") var transformer: Encoder
        let config: PixtralVisionConfiguration
        let patchPositionalEmbedding: RotaryEmbedding

        init(_ config: PixtralVisionConfiguration) {
            self.config = config
            self._patchConv.wrappedValue = Conv2d(
                inputChannels: config.numChannels,
                outputChannels: config.hiddenSize,
                kernelSize: .init(config.patchSize),
                stride: .init(config.patchSize),
                bias: false
            )
            self._lnPre.wrappedValue = RMSNorm(dimensions: config.hiddenSize)
            self._transformer.wrappedValue = Encoder(config)
            self.patchPositionalEmbedding = RotaryEmbedding(config)
        }

        func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
            MLXArray, [MLXArray]?
        ) {
            // x is expected in NHWC format: (batch, height, width, channels)
            var x = x
            if x.dtype != patchConv.weight.dtype {
                x = x.asType(patchConv.weight.dtype)
            }

            var patchEmbeds = patchConv(x)

            // Get patch dimensions before flattening
            let patchHeight = patchEmbeds.dim(1)
            let patchWidth = patchEmbeds.dim(2)
            let batch = patchEmbeds.dim(0)

            // Flatten spatial dimensions: (batch, h*w, hidden)
            patchEmbeds = patchEmbeds.reshaped(batch, -1, patchEmbeds.dim(-1))
            patchEmbeds = lnPre(patchEmbeds)

            // Compute position IDs and embeddings
            let maxWidth = config.imageSize / config.patchSize
            let positionIds = PixtralVision.positionIdsInMeshgrid(
                patchHeight: patchHeight,
                patchWidth: patchWidth,
                maxWidth: maxWidth
            )
            let positionEmbedding = patchPositionalEmbedding(patchEmbeds, positionIds: positionIds)

            // Generate block attention mask (supports multiple images in batch)
            let patchesPerImage = patchHeight * patchWidth

            let mask = PixtralVision.generateBlockAttentionMask(
                patchCounts: Array(repeating: patchesPerImage, count: batch),
                batchSize: batch,
                dtype: patchEmbeds.dtype
            )

            var encoderStates: [MLXArray]? = outputHiddenStates ? [patchEmbeds] : nil
            var h = patchEmbeds

            for layer in transformer.layers {
                h = layer(h, positionEmbeddings: positionEmbedding, mask: mask)
                if outputHiddenStates {
                    encoderStates?.append(h)
                }
            }

            return (h, encoderStates)
        }
    }

    // MARK: Vision Model Wrapper (matches pixtral VisionModel structure)

    class VisionModel: Module {
        @ModuleInfo(key: "vision_model") var visionModel: PixtralVisionModelInner
        let config: PixtralVisionConfiguration

        init(_ config: PixtralVisionConfiguration) {
            self.config = config
            self._visionModel.wrappedValue = PixtralVisionModelInner(config)
        }

        func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
            MLXArray, MLXArray, [MLXArray]?
        ) {
            let (encoded, hiddenStates) = visionModel(x, outputHiddenStates: outputHiddenStates)
            let embeddings = hiddenStates?.first ?? encoded
            return (encoded, embeddings, hiddenStates)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitizedWeights = [String: MLXArray]()
            for (k, v) in weights {
                if k.contains("position_ids") {
                    continue
                } else if k.contains("patch_conv.weight") || k.contains("patch_embedding.weight") {
                    if PixtralVision.checkArrayShape(v) {
                        sanitizedWeights[k] = v
                    } else {
                        sanitizedWeights[k] = v.transposed(0, 2, 3, 1)
                    }
                } else {
                    sanitizedWeights[k] = v
                }
            }
            return sanitizedWeights
        }
    }
}

// MARK: - Language Model Components (for Pixtral)

private enum PixtralLanguage {

    // MARK: Language Attention

    fileprivate class Attention: Module {
        let config: PixtralTextConfiguration
        let scale: Float
        let nHeads: Int
        let nKVHeads: Int
        let headDim: Int

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        let rope: RoPE
        let useQkNorm: Bool

        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

        init(_ config: PixtralTextConfiguration) {
            self.config = config

            let dim = config.hiddenSize
            self.nHeads = config.numAttentionHeads
            self.nKVHeads = config.numKeyValueHeads

            self.headDim = config.headDim
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
            self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
            self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
            self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

            // Handle rope scaling
            var ropeScale: Float = 1.0
            if let ropeScaling = config.ropeScaling,
                let typeValue = ropeScaling["type"],
                case .string(let type) = typeValue, type == "linear",
                let factor = ropeScaling["factor"]?.asFloat()
            {
                ropeScale = 1.0 / factor
            }

            self.rope = RoPE(
                dimensions: headDim,
                traditional: config.ropeTraditional,
                base: config.ropeTheta,
                scale: ropeScale
            )

            self.useQkNorm = config.useQkNorm
            if useQkNorm {
                self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
                self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
            }
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

            if useQkNorm, let qNorm, let kNorm {
                queries = qNorm(queries)
                keys = kNorm(keys)
            }

            queries = applyRotaryPosition(rope, to: queries, cache: cache)
            keys = applyRotaryPosition(rope, to: keys, cache: cache)

            let output = attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values,
                cache: cache, scale: scale, mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return wo(output)
        }
    }

    // MARK: Language MLP

    fileprivate class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        init(_ config: PixtralTextConfiguration) {
            let dim = config.hiddenSize
            let hiddenDim = config.intermediateSize

            self._gate.wrappedValue = Linear(dim, hiddenDim, bias: false)
            self._down.wrappedValue = Linear(hiddenDim, dim, bias: false)
            self._up.wrappedValue = Linear(dim, hiddenDim, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    // MARK: Language Transformer Block

    fileprivate class TransformerBlock: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: PixtralTextConfiguration) {
            self._attention.wrappedValue = Attention(config)
            self.mlp = MLP(config)
            self._inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?
        ) -> MLXArray {
            var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + r
            r = mlp(postAttentionLayerNorm(h))
            return h + r
        }
    }

    // MARK: Language Model Inner

    fileprivate class LanguageModelInner: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

        let layers: [TransformerBlock]
        let norm: RMSNorm
        let config: PixtralTextConfiguration

        init(_ config: PixtralTextConfiguration) {
            self.config = config

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabSize,
                dimensions: config.hiddenSize
            )

            self.layers = (0 ..< config.numHiddenLayers).map { _ in
                TransformerBlock(config)
            }

            self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(
            _ inputs: MLXArray,
            cache: [KVCache]?,
            inputsEmbeds: MLXArray? = nil
        ) -> MLXArray {
            var h: MLXArray
            if let inputsEmbeds {
                h = inputsEmbeds
            } else {
                h = embedTokens(inputs)
            }

            let mask = createAttentionMask(h: h, cache: cache?.first)

            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i])
            }

            return norm(h)
        }
    }

    // MARK: Language Model

    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        let config: PixtralTextConfiguration
        @ModuleInfo(key: "model") var model: LanguageModelInner
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        var kvHeads: [Int] {
            Array(repeating: config.numKeyValueHeads, count: config.numHiddenLayers)
        }

        init(_ config: PixtralTextConfiguration) {
            self.config = config
            self._model.wrappedValue = LanguageModelInner(config)

            if !config.tieWordEmbeddings {
                self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
            }
        }

        func callAsFunction(
            _ inputs: MLXArray,
            cache: [KVCache]?,
            inputsEmbeds: MLXArray? = nil
        ) -> MLXArray {
            var out = model(inputs, cache: cache, inputsEmbeds: inputsEmbeds)
            if config.tieWordEmbeddings {
                out = model.embedTokens.asLinear(out)
            } else if let lmHead {
                out = lmHead(out)
            }
            return out
        }

        func newCache(parameters: GenerateParameters?) -> [KVCache] {
            (0 ..< config.numHiddenLayers).map { _ in
                if let maxKVSize = parameters?.maxKVSize {
                    return RotatingKVCache(maxSize: maxKVSize, keep: 4)
                } else {
                    return KVCacheSimple()
                }
            }
        }
    }
}

// MARK: - Pixtral MultiModal Projector

private class PixtralMultiModalProjector: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo var gelu: GELU
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ config: PixtralConfiguration) {
        self._linear1.wrappedValue = Linear(
            config.visionConfig.hiddenSize,
            config.textConfig.hiddenSize,
            bias: true
        )
        self.gelu = GELU()
        self._linear2.wrappedValue = Linear(
            config.textConfig.hiddenSize,
            config.textConfig.hiddenSize,
            bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var result = linear1(x)
        result = gelu(result)
        result = linear2(result)
        return result
    }
}

// MARK: - Pixtral VLM Model

public class PixtralVLM: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: PixtralVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: PixtralLanguage.LanguageModel
    @ModuleInfo(key: "multi_modal_projector") private var multiModalProjector:
        PixtralMultiModalProjector

    public let config: PixtralConfiguration
    let visionFeatureLayer: Int

    public var vocabularySize: Int { config.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public init(_ config: PixtralConfiguration) {
        self.config = config
        self.visionFeatureLayer = config.visionFeatureLayer

        self._visionTower.wrappedValue = PixtralVision.VisionModel(config.visionConfig)
        self._languageModel.wrappedValue = PixtralLanguage.LanguageModel(config.textConfig)
        self._multiModalProjector.wrappedValue = PixtralMultiModalProjector(config)
    }

    private func getInputEmbeddings(
        inputIds: MLXArray?,
        pixelValues: MLXArray?
    ) -> MLXArray {
        guard let pixelValues else {
            guard let inputIds else {
                fatalError("Either inputIds or pixelValues must be provided")
            }
            return languageModel.model.embedTokens(inputIds)
        }

        guard let inputIds else {
            fatalError("inputIds required when pixelValues provided")
        }

        let inputsEmbeds = languageModel.model.embedTokens(inputIds)

        // Process through vision tower
        let (_, _, hiddenStates) = visionTower(
            pixelValues.transposed(0, 2, 3, 1),
            outputHiddenStates: true
        )

        // Select features from specified layer
        guard let hiddenStates else {
            fatalError("Vision model must return hidden states")
        }

        let layerIndex =
            visionFeatureLayer < 0
            ? hiddenStates.count + visionFeatureLayer
            : visionFeatureLayer
        let selectedFeatures = hiddenStates[layerIndex]

        // Project to text space
        let imageFeatures = multiModalProjector(selectedFeatures)

        // Merge embeddings
        return mergeInputIdsWithImageFeatures(
            imageTokenIndex: config.imageTokenIndex,
            imageFeatures: imageFeatures,
            inputsEmbeds: inputsEmbeds,
            inputIds: inputIds
        )
    }

    private func mergeInputIdsWithImageFeatures(
        imageTokenIndex: Int,
        imageFeatures: MLXArray,
        inputsEmbeds: MLXArray,
        inputIds: MLXArray
    ) -> MLXArray {
        let (_, numImagePatches, _) = (
            imageFeatures.dim(0),
            imageFeatures.dim(1),
            imageFeatures.dim(2)
        )

        // Find image token positions (assuming batch size is 1)
        let inputIdArray: [Int32] = inputIds[0].asArray(Int32.self)
        let imagePositions = inputIdArray.enumerated().compactMap {
            $1 == Int32(imageTokenIndex) ? $0 : nil
        }

        // Build text segments - text before each image token
        var textSegments: [MLXArray] = []
        var startIdx = 0

        for position in imagePositions {
            textSegments.append(inputsEmbeds[0..., startIdx ..< position, 0...])
            startIdx = position + 1
        }

        // Split image features into separate embeddings for each patch
        var imageEmbeddings: [MLXArray] = []
        for i in 0 ..< numImagePatches {
            imageEmbeddings.append(imageFeatures[0..., i ..< (i + 1), 0...])
        }

        // Interleave text and image embeddings
        var finalEmbeddings: [MLXArray] = []
        for (text, image) in zip(textSegments, imageEmbeddings) {
            finalEmbeddings.append(text)
            finalEmbeddings.append(image)
        }

        // Add remaining text after the last image token
        finalEmbeddings.append(inputsEmbeds[0..., startIdx..., 0...])

        // Concatenate along sequence dimension
        return MLX.concatenated(finalEmbeddings, axis: 1)
    }

    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let inputIds = input.text.tokens
        let pixelValues = input.image?.pixels

        let embeddings = getInputEmbeddings(
            inputIds: inputIds,
            pixelValues: pixelValues
        )

        // Chunked prefill — see Qwen25VL / vmlx #50/#51. Inner model ignores
        // `inputs` when `inputsEmbeds` is non-nil (`h = inputsEmbeds ?? embedTokens(inputs)`).
        let logits = chunkedPrefillEmbedding(
            inputEmbedding: embeddings,
            cache: cache,
            prefillStepSize: windowSize ?? 512
        ) { chunk in
            languageModel(inputIds, cache: cache, inputsEmbeds: chunk)
        }
        return .logits(.init(logits: logits))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights: [String: MLXArray] = [:]

        for (key, value) in weights {
            var newKey = key

            // Transform keys to match model structure
            if key.contains("vision_tower") && !key.contains("vision_model") {
                if key.contains("transformer") || key.contains("patch_conv")
                    || key.contains("ln_pre")
                {
                    newKey = key.replacingOccurrences(
                        of: "vision_tower", with: "vision_tower.vision_model")
                }
            } else if key.contains("vision_encoder") && !key.contains("vision_tower") {
                if key.contains("transformer") || key.contains("patch_conv")
                    || key.contains("ln_pre")
                {
                    newKey = key.replacingOccurrences(
                        of: "model.vision_encoder", with: "vision_tower.vision_model")
                }
            } else if key.contains("model.language_model") && !key.contains("language_model.model")
            {
                newKey = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if key.contains("lm_head") && !key.contains("language_model") {
                newKey = key.replacingOccurrences(of: "lm_head", with: "language_model.lm_head")
            } else if key.contains("model.vision_projection") {
                newKey = key.replacingOccurrences(
                    of: "model.vision_projection", with: "multi_modal_projector")
            }

            // Skip rotary embeddings
            if newKey.contains("self_attn.rotary_emb.inv_freq") {
                continue
            }

            if newWeights[newKey] == nil {
                newWeights[newKey] = value
            }
        }

        return newWeights
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }
}

// MARK: - LoRA Support

extension PixtralVLM: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}

// MARK: - Processor Configuration

public struct PixtralProcessorConfiguration: Codable, Sendable {
    public let imageProcessor: ImageProcessorConfig
    public let imageToken: String
    public let imageBreakToken: String?
    public let imageEndToken: String?
    public let patchSize: Int

    public struct ImageProcessorConfig: Codable, Sendable {
        public let imageMean: [CGFloat]
        public let imageStd: [CGFloat]
        public let size: ProcessorSize
        public let patchSize: Int
        public let doNormalize: Bool?
        public let doRescale: Bool?
        public let doResize: Bool?
        public let rescaleFactor: Float?

        public struct ProcessorSize: Codable, Sendable {
            public let width: Int?
            public let height: Int?
            public let longestEdge: Int?

            enum CodingKeys: String, CodingKey {
                case width
                case height
                case longestEdge = "longest_edge"
            }
        }

        public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
            (imageMean[0], imageMean[1], imageMean[2])
        }

        public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
            (imageStd[0], imageStd[1], imageStd[2])
        }

        enum CodingKeys: String, CodingKey {
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case size
            case patchSize = "patch_size"
            case doNormalize = "do_normalize"
            case doRescale = "do_rescale"
            case doResize = "do_resize"
            case rescaleFactor = "rescale_factor"
        }
    }

    enum CodingKeys: String, CodingKey {
        case imageProcessor = "image_processor"
        case imageToken = "image_token"
        case imageBreakToken = "image_break_token"
        case imageEndToken = "image_end_token"
        case patchSize = "patch_size"
    }
}

// MARK: - Processor

public struct PixtralProcessor: UserInputProcessor {
    private let config: PixtralProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imageTokenId: Int

    public init(_ config: PixtralProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        // Get image token ID from tokenizer, fallback to 10 (default for Pixtral)
        if let vocabTokenId = tokenizer.convertTokenToId(config.imageToken) {
            self.imageTokenId = vocabTokenId
        } else {
            self.imageTokenId = 10
        }
    }

    private func prompt(from userInput: UserInput) -> String {
        switch userInput.prompt {
        case .text(let text):
            return text
        case .messages(let messages):
            return messages.last?["content"] as? String ?? ""
        case .chat(let messages):
            return messages.last?.content ?? ""
        }
    }

    public func prepare(input: UserInput) throws -> LMInput {
        let prompt = prompt(from: input)

        if input.images.isEmpty {
            let tokens = tokenizer.encode(text: prompt)
            let tokensArray = MLXArray(tokens).expandedDimensions(axis: 0)
            let mask = ones(like: tokensArray)
            return LMInput(text: .init(tokens: tokensArray, mask: mask), image: nil)
        } else {
            guard input.images.count == 1 else {
                throw VLMError.singleImageAllowed
            }

            // Process image first to get dimensions
            let longestEdge = config.imageProcessor.size.longestEdge ?? 1540
            let patchSize = config.imageProcessor.patchSize

            var image = try input.images[0].asCIImage()
            image = MediaProcessing.inSRGBToneCurveSpace(image)
            image = MediaProcessing.apply(image, processing: input.processing)

            // Resize to longest edge while maintaining aspect ratio
            let originalSize = image.extent.size
            let scale = CGFloat(longestEdge) / max(originalSize.width, originalSize.height)
            let newWidth = Int((originalSize.width * scale).rounded())
            let newHeight = Int((originalSize.height * scale).rounded())

            // Round to patch size multiples for padding
            let paddedWidth = ((newWidth + patchSize - 1) / patchSize) * patchSize
            let paddedHeight = ((newHeight + patchSize - 1) / patchSize) * patchSize

            // Resize
            image = MediaProcessing.resampleBicubic(
                image,
                to: CGSize(width: newWidth, height: newHeight)
            )

            // Pad to patch boundaries (bottom-right padding)
            if newWidth != paddedWidth || newHeight != paddedHeight {
                let background = CIImage(color: .black).cropped(
                    to: CGRect(x: 0, y: 0, width: paddedWidth, height: paddedHeight))
                let tx = 0.0
                let ty = CGFloat(paddedHeight - newHeight)
                let transformed = image.transformed(by: CGAffineTransform(translationX: tx, y: ty))
                image = transformed.composited(over: background)
            }

            image = MediaProcessing.normalize(
                image,
                mean: config.imageProcessor.imageMeanTuple,
                std: config.imageProcessor.imageStdTuple
            )

            var pixels = MediaProcessing.asMLXArray(image)

            if pixels.ndim == 2 {
                pixels = pixels.expandedDimensions(axis: -1)
            }
            if pixels.ndim == 3 {
                pixels = pixels.expandedDimensions(axis: 0)
            }

            // Calculate number of image tokens needed
            let numPatchesH = paddedHeight / patchSize
            let numPatchesW = paddedWidth / patchSize
            let numImageTokens = numPatchesH * numPatchesW

            // Build prompt with image tokens
            var promptTokens = tokenizer.encode(text: prompt)

            // Insert image tokens
            let imageTokens = Array(repeating: imageTokenId, count: numImageTokens)

            var insertIndex = 0
            if !promptTokens.isEmpty {
                // If first token is BOS (typically id 1), insert after it
                if promptTokens[0] == 1 {
                    insertIndex = 1
                }
            }

            promptTokens.insert(contentsOf: imageTokens, at: insertIndex)

            let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            let mask = ones(like: promptArray)

            // Convert to BCHW format for vision model
            if pixels.dim(-1) == 3 {
                pixels = pixels.transposed(0, 3, 1, 2)
            }

            return LMInput(
                text: .init(tokens: promptArray, mask: mask),
                image: .init(pixels: pixels, frames: [THW(1, paddedHeight, paddedWidth)])
            )
        }
    }
}
