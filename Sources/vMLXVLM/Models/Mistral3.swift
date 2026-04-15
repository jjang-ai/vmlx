import CoreImage
import Foundation
import MLX
import vMLXLMCommon
import MLXNN

// Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/mistral3
// Note: Mistral3 reuses the vision model from Pixtral

// MARK: - Configuration

// Re-export PixtralVisionConfiguration for Mistral3 use
public typealias Mistral3VisionConfiguration = PixtralVisionConfiguration

// MARK: - Text Configuration

public struct Mistral3VLMTextConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let rmsNormEps: Float
    public let vocabSize: Int

    public var headDim: Int? { _headDim }
    public var maxPositionEmbeddings: Int? { _maxPositionEmbeddings }
    public var numKeyValueHeads: Int { _numKeyValueHeads ?? numAttentionHeads }
    public var ropeTheta: Float { _ropeTheta ?? 1_000_000_000 }
    public var ropeParameters: [String: StringOrNumber]? { _ropeParameters }
    public var ropeTraditional: Bool { _ropeTraditional ?? false }
    public var ropeScaling: [String: StringOrNumber]? { _ropeScaling }
    public var tieWordEmbeddings: Bool { _tieWordEmbeddings ?? false }
    public var layerTypes: [String]? { _layerTypes }
    public var slidingWindow: Int? { _slidingWindow }
    public var useQkNorm: Bool { _useQkNorm ?? false }

    private let _headDim: Int?
    private let _maxPositionEmbeddings: Int?
    private let _numKeyValueHeads: Int?
    private let _ropeTheta: Float?
    private let _ropeParameters: [String: StringOrNumber]?
    private let _ropeTraditional: Bool?
    private let _ropeScaling: [String: StringOrNumber]?
    private let _tieWordEmbeddings: Bool?
    private let _layerTypes: [String]?
    private let _slidingWindow: Int?
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
        case _ropeParameters = "rope_parameters"
        case _ropeTraditional = "rope_traditional"
        case _ropeScaling = "rope_scaling"
        case _tieWordEmbeddings = "tie_word_embeddings"
        case _layerTypes = "layer_types"
        case _slidingWindow = "sliding_window"
        case _useQkNorm = "use_qk_norm"
    }
}

// MARK: - Model Configuration

public struct Mistral3VLMConfiguration: Codable, Sendable {
    public let textConfig: Mistral3VLMTextConfiguration
    public let visionConfig: Mistral3VisionConfiguration
    public let modelType: String

    public var ignoreIndex: Int { _ignoreIndex ?? -100 }
    public var imageTokenIndex: Int { _imageTokenIndex ?? _imageTokenId ?? 10 }
    public var visionFeatureSelectStrategy: String { _visionFeatureSelectStrategy ?? "full" }
    public var visionFeatureLayer: Int { _visionFeatureLayer ?? -1 }
    public var vocabSize: Int { _vocabSize ?? 32000 }
    public var spatialMergeSize: Int { _spatialMergeSize ?? 2 }
    public var multimodalProjectorBias: Bool { _multimodalProjectorBias ?? false }
    public var eosTokenId: [Int]? { _eosTokenId }

    private let _ignoreIndex: Int?
    private let _imageTokenIndex: Int?
    private let _imageTokenId: Int?
    private let _visionFeatureSelectStrategy: String?
    private let _visionFeatureLayer: Int?
    private let _vocabSize: Int?
    private let _spatialMergeSize: Int?
    private let _multimodalProjectorBias: Bool?
    private let _eosTokenId: [Int]?

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
        case _spatialMergeSize = "spatial_merge_size"
        case _multimodalProjectorBias = "multimodal_projector_bias"
        case _eosTokenId = "eos_token_id"
    }
}

// MARK: - Unfold (im2col)

/// Extract sliding local blocks from a batched input tensor.
/// Equivalent to PyTorch's nn.functional.unfold / im2col operation.
func unfold(
    _ input: MLXArray,
    kernelSize: Int,
    dilation: Int = 1,
    padding: Int = 0,
    stride: Int = 1
) -> MLXArray {
    var x = input
    let (batchSize, channels, height, width) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))

    // Add padding if needed
    if padding > 0 {
        x = MLX.padded(
            x,
            widths: [
                0,  // batch
                0,  // channels
                .init((padding, padding)),  // height
                .init((padding, padding)),  // width
            ])
    }

    let paddedH = height + 2 * padding
    let paddedW = width + 2 * padding

    // Calculate output dimensions
    let heightOut = (paddedH - dilation * (kernelSize - 1) - 1) / stride + 1
    let widthOut = (paddedW - dilation * (kernelSize - 1) - 1) / stride + 1

    // Extract blocks using array indexing
    var blocks: [MLXArray] = []

    for i in Swift.stride(from: 0, to: paddedH - kernelSize * dilation + 1, by: stride) {
        for j in Swift.stride(from: 0, to: paddedW - kernelSize * dilation + 1, by: stride) {
            var block: [MLXArray] = []
            for di in 0 ..< kernelSize {
                for dj in 0 ..< kernelSize {
                    let hIdx = i + di * dilation
                    let wIdx = j + dj * dilation
                    block.append(x[0..., 0..., hIdx, wIdx])
                }
            }
            // Stack the channel-blocks: (B, C, k*k)
            let stackedBlock = MLX.stacked(block, axis: 1).transposed(0, 2, 1)
            blocks.append(stackedBlock)
        }
    }

    // Stack all blocks: (B, C, k*k, L)
    let result = MLX.stacked(blocks, axis: -1)

    // Reshape to (B, C*k*k, L)
    return result.reshaped(batchSize, channels * kernelSize * kernelSize, heightOut * widthOut)
}

// MARK: - Mistral3 Patch Merger

class Mistral3PatchMerger: Module {
    let spatialMergeSize: Int
    let patchSize: Int

    @ModuleInfo(key: "merging_layer") var mergingLayer: Linear

    init(_ config: Mistral3VLMConfiguration) {
        self.spatialMergeSize = config.spatialMergeSize
        self.patchSize = config.visionConfig.patchSize

        let hiddenSize = config.visionConfig.hiddenSize
        self._mergingLayer.wrappedValue = Linear(
            hiddenSize * spatialMergeSize * spatialMergeSize,
            hiddenSize,
            bias: false
        )
    }

    func callAsFunction(_ imageFeatures: MLXArray, imageSizes: [(Int, Int)]) -> MLXArray {
        // Convert image sizes to patch sizes
        let patchSizes = imageSizes.map { (h, w) in
            (h / patchSize, w / patchSize)
        }

        let tokensPerImage = patchSizes.map { $0.0 * $0.1 }
        let d = imageFeatures.dim(-1)
        var features = imageFeatures.asType(.bfloat16)

        // Split the image features into chunks based on tokens per image
        var splitIndices: [Int] = []
        var currentIndex = 0
        for tokens in tokensPerImage.dropLast() {
            currentIndex += tokens
            splitIndices.append(currentIndex)
        }

        let chunks: [MLXArray]
        if splitIndices.isEmpty {
            chunks = [features[0, 0..., 0...]]
        } else {
            chunks = MLX.split(features[0], indices: splitIndices, axis: 0)
        }

        var permutedTensors: [MLXArray] = []

        for (imageIndex, imageTokens) in chunks.enumerated() {
            if imageTokens.dim(0) > 0 {
                let (h, w) = patchSizes[imageIndex]

                // Reshape to grid: (h, w, d) -> (1, d, h, w)
                let imageGrid = imageTokens.reshaped(h, w, d).transposed(2, 0, 1)[
                    .newAxis, 0..., 0..., 0...]

                // Apply unfold
                var grid = unfold(imageGrid, kernelSize: spatialMergeSize, stride: spatialMergeSize)

                // Reshape: (d * spatial_merge_size^2, -1).T
                grid = grid.reshaped(d * spatialMergeSize * spatialMergeSize, -1).transposed()
                permutedTensors.append(grid)
            }
        }

        features = MLX.concatenated(permutedTensors, axis: 0)
        features = mergingLayer(features)

        return features[.newAxis, 0..., 0...]
    }
}

// MARK: - Mistral3 MultiModal Projector

class Mistral3MultiModalProjector: Module {
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "patch_merger") var patchMerger: Mistral3PatchMerger
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo var gelu: GELU
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ config: Mistral3VLMConfiguration) {
        self._norm.wrappedValue = RMSNorm(dimensions: config.visionConfig.hiddenSize)
        self._patchMerger.wrappedValue = Mistral3PatchMerger(config)
        self._linear1.wrappedValue = Linear(
            config.visionConfig.hiddenSize,
            config.textConfig.hiddenSize,
            bias: config.multimodalProjectorBias
        )
        self.gelu = GELU()
        self._linear2.wrappedValue = Linear(
            config.textConfig.hiddenSize,
            config.textConfig.hiddenSize,
            bias: config.multimodalProjectorBias
        )
    }

    func callAsFunction(_ x: MLXArray, imageSizes: [(Int, Int)]) -> MLXArray {
        var result = norm(x)
        result = patchMerger(result, imageSizes: imageSizes)
        result = linear1(result)
        result = gelu(result)
        result = linear2(result)
        return result
    }
}

// MARK: - Language Model Components

private enum Language {

    // MARK: Llama 4 Attention Scaling

    static func getLlama4AttentionScale(
        start: Int, stop: Int, beta: Float, maxPositionEmbeddings: Int
    ) -> MLXArray {
        let positions = MLXArray(start ..< stop).asType(.float32)
        let scaling = 1 + beta * MLX.log(1 + MLX.floor(positions / Float(maxPositionEmbeddings)))
        return expandedDimensions(scaling, axis: -1)
    }

    // MARK: Language Attention

    fileprivate class Attention: Module {
        let config: Mistral3VLMTextConfiguration
        let scale: Float
        let nHeads: Int
        let nKVHeads: Int
        let headDim: Int

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        let rope: RoPELayer

        init(_ config: Mistral3VLMTextConfiguration) {
            self.config = config

            let dim = config.hiddenSize
            self.nHeads = config.numAttentionHeads
            self.nKVHeads = config.numKeyValueHeads

            self.headDim = config.headDim ?? (config.hiddenSize / nHeads)
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
            self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
            self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
            self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

            // Initialize RoPE using rope_parameters - rope_theta is required like in Python
            guard let ropeParams = config.ropeParameters,
                let ropeTheta = ropeParams["rope_theta"]?.asFloat()
            else {
                fatalError("rope_parameters['rope_theta'] is required")
            }
            self.rope = initializeRope(
                dims: headDim,
                base: ropeTheta,
                traditional: false,
                scalingConfig: config.ropeParameters,
                maxPositionEmbeddings: config.maxPositionEmbeddings
            )
        }

        func callAsFunction(
            _ x: MLXArray,
            attentionScale: MLXArray,
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

            queries = applyRotaryPosition(rope, to: queries, cache: cache)
            keys = applyRotaryPosition(rope, to: keys, cache: cache)

            queries = queries * attentionScale

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

        init(_ config: Mistral3VLMTextConfiguration) {
            let dim = config.hiddenSize
            let hiddenDim = config.intermediateSize

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

    // MARK: Language Transformer Block (for Ministral3 with attn_scale)

    fileprivate class TransformerBlock: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        let useSliding: Bool

        init(_ config: Mistral3VLMTextConfiguration, useSliding: Bool = false) {
            self.useSliding = useSliding
            self._attention.wrappedValue = Attention(config)
            self.mlp = MLP(config)
            self._inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(
            _ x: MLXArray,
            attentionScale: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?
        ) -> MLXArray {
            var r = attention(
                inputLayerNorm(x), attentionScale: attentionScale, mask: mask, cache: cache)
            let h = x + r
            r = mlp(postAttentionLayerNorm(h))
            return h + r
        }
    }

    // MARK: Ministral3 Model Inner (with sliding attention and llama4 scaling)

    fileprivate class Ministral3ModelInner: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

        let layers: [TransformerBlock]
        let norm: RMSNorm
        let config: Mistral3VLMTextConfiguration
        let layerTypes: [String]
        let slidingWindow: Int?
        let faIndex: Int
        let swaIndex: Int?

        init(_ config: Mistral3VLMTextConfiguration) {
            self.config = config
            self.slidingWindow = config.slidingWindow
            self.layerTypes =
                config.layerTypes
                ?? Array(repeating: "full_attention", count: config.numHiddenLayers)

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabSize,
                dimensions: config.hiddenSize
            )

            self.layers = layerTypes.map { layerType in
                TransformerBlock(config, useSliding: layerType == "sliding_attention")
            }

            self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

            self.faIndex = layerTypes.firstIndex(of: "full_attention") ?? 0
            self.swaIndex = layers.firstIndex { $0.useSliding }
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

            let cache = cache ?? []
            let offset = cache.first?.offset ?? 0

            let faMask = createAttentionMask(h: h, cache: cache[faIndex])

            var swaMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
            if let swaIndex, let slidingWindow, !cache.isEmpty {
                let t = h.dim(1)
                if t > 1 {
                    let swaOffset = min(slidingWindow, cache[swaIndex].offset)
                    swaMask = .array(
                        createCausalMask(n: t, offset: swaOffset, windowSize: slidingWindow))
                }
            }

            let beta = config.ropeParameters?["llama_4_scaling_beta"]?.asFloat() ?? 0.0
            let originalMaxPos =
                config.ropeParameters?["original_max_position_embeddings"]?.asInt()
                ?? config.maxPositionEmbeddings ?? 4096
            let attentionScale = getLlama4AttentionScale(
                start: offset,
                stop: offset + h.dim(1),  // Use h's length (embeddings), not inputs (token IDs)
                beta: beta,
                maxPositionEmbeddings: originalMaxPos
            ).asType(h.dtype)

            for (i, layer) in layers.enumerated() {
                let mask = layer.useSliding ? swaMask : faMask
                h = layer(
                    h, attentionScale: attentionScale, mask: mask,
                    cache: cache.isEmpty ? nil : cache[i])
            }

            return norm(h)
        }
    }

    // MARK: Language Model

    /// Language model that supports both ministral3 and mistral model types.
    /// For ministral3: uses sliding attention with llama4 attention scaling
    /// For mistral: uses standard attention with optional QK norm
    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        let config: Mistral3VLMTextConfiguration
        let modelType: String

        // Use ministral3 model as the primary implementation
        // It handles both cases: ministral3 with sliding attention, or standard with beta=0
        @ModuleInfo(key: "model") private var model: Ministral3ModelInner

        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        var kvHeads: [Int] {
            let layerTypes =
                config.layerTypes
                ?? Array(repeating: "full_attention", count: config.numHiddenLayers)
            return layerTypes.map { _ in config.numKeyValueHeads }
        }

        /// Access to embed_tokens
        var embedTokens: Embedding {
            model.embedTokens
        }

        /// Access to layers for LoRA
        var layers: [TransformerBlock] {
            model.layers
        }

        init(_ config: Mistral3VLMTextConfiguration) {
            self.config = config
            self.modelType = config.modelType

            // Ministral3ModelInner handles both model types:
            // - For ministral3: uses sliding attention and llama4 scaling from rope_parameters
            // - For mistral: when llama_4_scaling_beta is 0 or missing, attention_scale becomes 1.0
            //   and all layers use full attention (no layer_types means all "full_attention")
            self._model.wrappedValue = Ministral3ModelInner(config)

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
                out = embedTokens.asLinear(out)
            } else if let lmHead {
                out = lmHead(out)
            }
            return out
        }

        func newCache(parameters: GenerateParameters?) -> [KVCache] {
            let layerTypes =
                config.layerTypes
                ?? Array(repeating: "full_attention", count: config.numHiddenLayers)

            return layerTypes.map { layerType in
                if layerType == "sliding_attention", let slidingWindow = config.slidingWindow {
                    return RotatingKVCache(maxSize: slidingWindow)
                } else if let maxKVSize = parameters?.maxKVSize {
                    return RotatingKVCache(maxSize: maxKVSize, keep: 4)
                } else {
                    return KVCacheSimple()
                }
            }
        }
    }
}

// MARK: - Mistral3 VLM Model

public class Mistral3VLM: Module, VLMModel, KVCacheDimensionProvider {
    // Use PixtralVision.VisionModel from Pixtral.swift
    @ModuleInfo(key: "vision_tower") private var visionTower: PixtralVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel
    @ModuleInfo(key: "multi_modal_projector") private var multiModalProjector:
        Mistral3MultiModalProjector

    public let config: Mistral3VLMConfiguration
    let visionFeatureLayer: Int

    public var vocabularySize: Int { config.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public init(_ config: Mistral3VLMConfiguration) {
        self.config = config
        self.visionFeatureLayer = config.visionFeatureLayer

        self._visionTower.wrappedValue = PixtralVision.VisionModel(config.visionConfig)
        self._languageModel.wrappedValue = Language.LanguageModel(config.textConfig)
        self._multiModalProjector.wrappedValue = Mistral3MultiModalProjector(config)
    }

    private func getInputEmbeddings(
        inputIds: MLXArray?,
        pixelValues: MLXArray?,
        imageSizes: [(Int, Int)]?
    ) -> MLXArray {
        guard var pixelValues, let imageSizes else {
            guard let inputIds else {
                fatalError("Either inputIds or pixelValues must be provided")
            }
            return languageModel.embedTokens(inputIds)
        }

        guard let inputIds else {
            fatalError("inputIds required when pixelValues provided")
        }

        let inputsEmbeds = languageModel.embedTokens(inputIds)

        // Handle 3D pixel values (missing batch dimension)
        if pixelValues.ndim == 3 {
            pixelValues = pixelValues.expandedDimensions(axis: 0)
        }

        // Process through vision tower (reuses Pixtral vision model)
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

        // Project to text space using Mistral3's patch merger projector
        let imageFeatures = multiModalProjector(selectedFeatures, imageSizes: imageSizes)

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

        // Validate that the number of image tokens matches the number of image patches
        guard imagePositions.count == numImagePatches else {
            fatalError(
                "Image token count (\(imagePositions.count)) does not match image patches (\(numImagePatches)). Ensure the processor adds exactly numImagePatches image tokens."
            )
        }

        // Build text segments - text before each image token
        var textSegments: [MLXArray] = []
        var startIdx = 0

        for position in imagePositions {
            textSegments.append(inputsEmbeds[0..., startIdx ..< position, 0...])
            startIdx = position + 1
        }

        // Split image features into separate embeddings for each image
        // imageFeatures shape: (numImages, numImagePatches, embedDim)
        // Split along axis 1 into numImagePatches parts (one per patch)
        let splitIndices = Array(1 ..< numImagePatches)
        let imageEmbeddings = MLX.split(imageFeatures, indices: splitIndices, axis: 1)

        // Interleave text and image embeddings
        // [text0, img0, text1, img1, ...]
        var finalEmbeddings: [MLXArray] = []
        for (text, image) in zip(textSegments, imageEmbeddings) {
            finalEmbeddings.append(text)
            finalEmbeddings.append(image)
        }

        // Add remaining text after the last image token
        finalEmbeddings.append(inputsEmbeds[0..., startIdx..., 0...])

        // Create a final embedding of shape
        // (1, num_image_patches*num_images + sequence_len, embed_dim)
        return MLX.concatenated(finalEmbeddings, axis: 1)
    }

    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let inputIds = input.text.tokens
        let pixelValues = input.image?.pixels

        // Extract image sizes from frames or fall back to config defaults
        let imageSizes: [(Int, Int)]?
        if let frames = input.image?.frames {
            imageSizes = frames.map { ($0.h, $0.w) }
        } else if pixelValues != nil {
            imageSizes = [(config.visionConfig.imageSize, config.visionConfig.imageSize)]
        } else {
            imageSizes = nil
        }

        let embeddings = getInputEmbeddings(
            inputIds: inputIds,
            pixelValues: pixelValues,
            imageSizes: imageSizes
        )

        // Chunked prefill — see Qwen25VL / vmlx #50/#51. Inner model does
        // `if let inputsEmbeds { h = inputsEmbeds } else { h = embedTokens(inputs) }`,
        // so `inputs` is ignored when embeds is non-nil. `h.dim(1)` drives
        // attention-scale length, which reflects the per-chunk embedding.
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
            // Vision tower keys: vision_tower.X -> vision_tower.vision_model.X (for pixtral structure)
            if key.contains("vision_tower") && !key.contains("vision_model") {
                if key.contains("transformer") || key.contains("patch_conv")
                    || key.contains("ln_pre")
                {
                    newKey = key.replacingOccurrences(
                        of: "vision_tower", with: "vision_tower.vision_model")
                }
            } else if key.contains("vision_encoder") && !key.contains("vision_tower") {
                // Alternative key format: model.vision_encoder.X -> vision_tower.vision_model.X
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

            // Handle weight scale patterns
            if newKey.contains("weight_scale_inv") {
                let scaleInv = value
                let weightKey = newKey.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = weights[key.replacingOccurrences(of: "_scale_inv", with: "")] {
                    newWeights[weightKey] = weight * scaleInv
                }
            } else if newKey.contains("activation_scale") {
                continue
            } else if newWeights[newKey] == nil {
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

extension Mistral3VLM: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.layers
    }
}

// MARK: - Processor Configuration

public struct Mistral3VLMProcessorConfiguration: Codable, Sendable {
    public let imageProcessor: ImageProcessorConfig
    public let imageToken: String
    public let imageBreakToken: String?
    public let imageEndToken: String?
    public let patchSize: Int
    public let spatialMergeSize: Int?

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
        case spatialMergeSize = "spatial_merge_size"
    }
}

// MARK: - Message Generator for Mistral3 VLM

/// Message generator for Mistral3 VLM that creates structured messages with image placeholders
public struct Mistral3MessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> Message {
        // For Mistral3 VLM, images come before text in the content
        [
            "role": message.role.rawValue,
            "content": message.images.map { _ in
                ["type": "image"]
            } + [["type": "text", "text": message.content]],
        ]
    }
}

// MARK: - Processor

public struct Mistral3VLMProcessor: UserInputProcessor {
    private let config: Mistral3VLMProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imageToken: String
    private let imageTokenId: Int

    private struct PreprocessResult {
        let pixels: MLXArray  // BCHW
        let frames: [THW]
        let numImageTokens: Int
    }

    public init(_ config: Mistral3VLMProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        self.imageToken = config.imageToken
        // Get image token ID from tokenizer, fallback to 10 (default for Mistral3)
        if let vocabTokenId = tokenizer.convertTokenToId(config.imageToken) {
            self.imageTokenId = vocabTokenId
        } else {
            self.imageTokenId = 10
        }
    }

    private func preprocessImage(
        _ image: CIImage,
        processing: UserInput.Processing?,
        patchSize: Int,
        spatialMergeSize: Int,
        longestEdge: Int?
    ) throws -> PreprocessResult {
        var image = MediaProcessing.inSRGBToneCurveSpace(image)
        image = MediaProcessing.apply(image, processing: processing)

        let maxVisionEdge = patchSize * 24  // Pixtral vision expects 24x24 patches (336px for patchSize=14)
        let targetEdge = min(longestEdge ?? maxVisionEdge, maxVisionEdge)

        let originalSize = image.extent.size
        let scale = min(CGFloat(targetEdge) / max(originalSize.width, originalSize.height), 1.0)
        let newWidth = max(1, Int((originalSize.width * scale).rounded()))
        let newHeight = max(1, Int((originalSize.height * scale).rounded()))

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
        // Convert to BCHW format for vision model
        if pixels.dim(-1) == 3 {
            pixels = pixels.transposed(0, 3, 1, 2)
        }

        // Calculate number of image tokens needed after spatial merging
        let numPatchesH = paddedHeight / patchSize
        let numPatchesW = paddedWidth / patchSize
        let mergedPatchesH = numPatchesH / spatialMergeSize
        let mergedPatchesW = numPatchesW / spatialMergeSize
        let numImageTokens = mergedPatchesH * mergedPatchesW

        return PreprocessResult(
            pixels: pixels,
            frames: [THW(1, paddedHeight, paddedWidth)],
            numImageTokens: numImageTokens
        )
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        // Generate structured messages using the message generator
        let messages = Mistral3MessageGenerator().generate(from: input)

        if input.images.isEmpty {
            // No image - just apply chat template
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: messages,
                tools: input.tools,
                additionalContext: input.additionalContext
            )
            let tokensArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            let mask = ones(like: tokensArray)
            return LMInput(text: .init(tokens: tokensArray, mask: mask), image: nil)
        }

        guard input.images.count == 1 else {
            throw VLMError.singleImageAllowed
        }
        let spatialMergeSize = config.spatialMergeSize ?? 2
        let patchSize = config.imageProcessor.patchSize

        // Apply chat template to get tokenized prompt with image placeholder
        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext
        )

        // Decode to find and replace image placeholder token
        let decoded = tokenizer.decode(tokenIds: promptTokens, skipSpecialTokens: false)

        // Process image to get dimensions
        let preprocessResult = try preprocessImage(
            input.images[0].asCIImage(),
            processing: input.processing,
            patchSize: patchSize,
            spatialMergeSize: spatialMergeSize,
            longestEdge: config.imageProcessor.size.longestEdge
        )

        // Replace the image placeholder token with the correct number of image tokens
        // The chat template should have inserted the imageToken (e.g., "[IMG]") which we need to expand
        if decoded.contains(imageToken) {
            // Split by image token and re-encode with expanded image tokens
            let pieces = decoded.components(separatedBy: imageToken)
            var expandedTokens: [Int] = []

            for (index, piece) in pieces.enumerated() {
                if !piece.isEmpty {
                    let pieceTokens = tokenizer.encode(text: piece)
                    expandedTokens.append(contentsOf: pieceTokens)
                }
                // Add image tokens between pieces (not after the last one)
                if index < pieces.count - 1 {
                    expandedTokens.append(
                        contentsOf: Array(
                            repeating: imageTokenId, count: preprocessResult.numImageTokens))
                }
            }
            promptTokens = expandedTokens
        } else {
            // Fallback: If no image token placeholder found, try to find and replace the single image token ID
            // or insert at the beginning after BOS
            var foundImageToken = false
            var expandedTokens: [Int] = []

            for token in promptTokens {
                if token == imageTokenId && !foundImageToken {
                    // Replace single image token with expanded tokens
                    expandedTokens.append(
                        contentsOf: Array(
                            repeating: imageTokenId, count: preprocessResult.numImageTokens))
                    foundImageToken = true
                } else {
                    expandedTokens.append(token)
                }
            }

            if foundImageToken {
                promptTokens = expandedTokens
            } else {
                // Last resort: insert image tokens after BOS (if present) or at start
                var insertIndex = 0
                if !promptTokens.isEmpty && promptTokens[0] == 1 {
                    insertIndex = 1  // After BOS token
                }
                promptTokens.insert(
                    contentsOf: Array(
                        repeating: imageTokenId, count: preprocessResult.numImageTokens),
                    at: insertIndex
                )
            }
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray)

        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: .init(pixels: preprocessResult.pixels, frames: preprocessResult.frames)
        )
    }
}
