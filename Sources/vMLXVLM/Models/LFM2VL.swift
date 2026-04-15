// Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/lfm2_vl

import CoreImage
import Foundation
import MLX
import vMLXLMCommon
import MLXNN

// MARK: - Vision

private enum Vision {

    fileprivate class Attention: Module {
        let numHeads: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(dims: Int, numHeads: Int, bias: Bool = true) {
            precondition(
                dims % numHeads == 0,
                "The input feature dimensions should be divisible by the number of heads")

            self.numHeads = numHeads
            let headDim = dims / numHeads
            self.scale = pow(Float(headDim), -0.5)

            self._qProj.wrappedValue = Linear(dims, dims, bias: bias)
            self._kProj.wrappedValue = Linear(dims, dims, bias: bias)
            self._vProj.wrappedValue = Linear(dims, dims, bias: bias)
            self._outProj.wrappedValue = Linear(dims, dims, bias: bias)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
            var queries = qProj(x)
            var keys = kProj(x)
            var values = vProj(x)

            let (B, L, _) = (queries.dim(0), queries.dim(1), queries.dim(2))
            let S = keys.dim(1)

            queries = queries.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, S, numHeads, -1).transposed(0, 2, 1, 3)
            values = values.reshaped(B, S, numHeads, -1).transposed(0, 2, 1, 3)

            let output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values,
                scale: scale,
                mask: mask.map { .array($0) } ?? .none
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return outProj(output)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "fc1") var fc1: Linear
        @ModuleInfo(key: "fc2") var fc2: Linear

        init(config: LFM2VLConfiguration.VisionConfiguration) {
            self._fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
            self._fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(safeGeluApproximate(fc1(x)))
        }
    }

    fileprivate class EncoderLayer: Module {
        let embedDim: Int
        @ModuleInfo(key: "self_attn") var selfAttn: Attention
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

        init(config: LFM2VLConfiguration.VisionConfiguration) {
            self.embedDim = config.hiddenSize

            self._selfAttn.wrappedValue = Attention(
                dims: config.hiddenSize, numHeads: config.numAttentionHeads, bias: true)
            self._layerNorm1.wrappedValue = LayerNorm(
                dimensions: embedDim, eps: config.layerNormEps)
            self.mlp = MLP(config: config)
            self._layerNorm2.wrappedValue = LayerNorm(
                dimensions: embedDim, eps: config.layerNormEps)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
            var r = selfAttn(layerNorm1(x), mask: mask)
            var h = x + r
            r = mlp(layerNorm2(h))
            h = h + r
            return h
        }
    }

    fileprivate class Encoder: Module {
        var layers: [EncoderLayer]

        init(config: LFM2VLConfiguration.VisionConfiguration, visionFeatureLayer: Int = -1) {
            // Determine how many layers to create
            let numLayers: Int

            // visionFeatureLayer == -1 means use all layers
            // Other negative values are Python-style indices from the end (e.g., -2 = second to last)
            if visionFeatureLayer == -1 {
                numLayers = config.numHiddenLayers
            } else {
                // Convert negative indices to positive (e.g., -2 with 27 layers -> 25)
                let actualLayer =
                    visionFeatureLayer < 0
                    ? config.numHiddenLayers + visionFeatureLayer
                    : visionFeatureLayer

                if actualLayer >= 0 && actualLayer < config.numHiddenLayers {
                    numLayers = actualLayer + 1
                } else {
                    numLayers = config.numHiddenLayers
                }
            }
            self.layers = (0 ..< numLayers).map { _ in EncoderLayer(config: config) }
        }

        func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false, mask: MLXArray? = nil)
            -> [MLXArray]?
        {
            var encoderStates: [MLXArray]? = outputHiddenStates ? [x] : nil

            var h = x
            for layer in layers {
                h = layer(h, mask: mask)
                if outputHiddenStates {
                    encoderStates?.append(h)
                }
            }

            return encoderStates
        }
    }

    fileprivate class VisionEmbeddings: Module {
        let config: LFM2VLConfiguration.VisionConfiguration
        let embedDim: Int
        let imageSize: Int
        let patchSize: Int
        let numPatches: Int
        let positionEmbeddingSize: Int

        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        init(config: LFM2VLConfiguration.VisionConfiguration) {
            self.config = config
            self.embedDim = config.hiddenSize
            self.imageSize = config.imageSize
            self.patchSize = config.patchSize
            self.numPatches = config.numPatches
            self.positionEmbeddingSize = Int(sqrt(Double(numPatches)))

            self._patchEmbedding.wrappedValue = Linear(
                config.numChannels * patchSize * patchSize,
                embedDim
            )
            self._positionEmbedding.wrappedValue = Embedding(
                embeddingCount: numPatches, dimensions: embedDim)
        }

        /// Resize positional embeddings using bicubic interpolation
        static func resizePositionalEmbeddings(
            positionalEmbeddings: MLXArray,
            spatialShapes: MLXArray,
            maxLength: Int
        ) -> MLXArray {
            let batchSize = spatialShapes.dim(0)
            let srcH = positionalEmbeddings.dim(0)
            let srcW = positionalEmbeddings.dim(1)
            let embedDim = positionalEmbeddings.dim(-1)
            let sourceDtype = positionalEmbeddings.dtype

            let resultedPositionalEmbeddings = MLXArray.zeros(
                [batchSize, maxLength, embedDim], dtype: sourceDtype)

            // Reshape from [H, W, embedDim] to [1, embedDim, H, W] once before loop
            let reshapedEmbeddings =
                positionalEmbeddings
                .transposed(2, 0, 1)
                .reshaped(1, embedDim, srcH, srcW)

            for i in 0 ..< batchSize {
                let shape = spatialShapes[i]
                let targetH = shape[0].item(Int.self)
                let targetW = shape[1].item(Int.self)

                // Bicubic interpolation
                let interpolated = bicubicInterpolate(
                    reshapedEmbeddings,
                    size: (targetH, targetW)
                )

                // Reshape to [targetH * targetW, embedDim]
                let resizedEmbeddings =
                    interpolated
                    .reshaped(embedDim, targetH * targetW)
                    .transposed(1, 0)

                let numPositions = targetH * targetW
                resultedPositionalEmbeddings[i, 0 ..< numPositions] = resizedEmbeddings
                // Fill remaining positions with the first embedding
                if numPositions < maxLength {
                    for j in numPositions ..< maxLength {
                        resultedPositionalEmbeddings[i, j] = resizedEmbeddings[0]
                    }
                }
            }

            return resultedPositionalEmbeddings
        }

        func callAsFunction(_ pixelValues: MLXArray, spatialShapes: MLXArray) -> MLXArray {
            let targetDtype = patchEmbedding.weight.dtype
            let patchEmbeds = patchEmbedding(pixelValues.asType(targetDtype))

            let positionalEmbeddings = positionEmbedding.weight.reshaped(
                positionEmbeddingSize, positionEmbeddingSize, -1
            )

            let resizedPositionalEmbeddings = VisionEmbeddings.resizePositionalEmbeddings(
                positionalEmbeddings: positionalEmbeddings,
                spatialShapes: spatialShapes,
                maxLength: pixelValues.dim(1)
            )

            let embeddings = patchEmbeds + resizedPositionalEmbeddings
            return embeddings
        }
    }

    fileprivate class VisionModel: Module {
        let modelType: String

        @ModuleInfo var embeddings: VisionEmbeddings
        @ModuleInfo var encoder: Encoder
        @ModuleInfo(key: "post_layernorm") var postLayernorm: LayerNorm

        init(config: LFM2VLConfiguration.VisionConfiguration, visionFeatureLayer: Int = -1) {
            self.modelType = config.modelType

            self.embeddings = VisionEmbeddings(config: config)
            self.encoder = Encoder(config: config, visionFeatureLayer: visionFeatureLayer)
            self._postLayernorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
        }

        func callAsFunction(
            _ x: MLXArray,
            outputHiddenStates: Bool = false,
            spatialShapes: MLXArray
        ) -> (encoderOutputs: [MLXArray]?, embeddings: MLXArray, lastHiddenState: MLXArray) {
            var embeds = embeddings(x, spatialShapes: spatialShapes)
            embeds = embeds.asType(embeddings.patchEmbedding.weight.dtype)

            let encoderOutputs = encoder(embeds, outputHiddenStates: outputHiddenStates, mask: nil)
            let lastHiddenState = postLayernorm(encoderOutputs?.last ?? embeds)

            return (encoderOutputs, embeds, lastHiddenState)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitizedWeights = [String: MLXArray]()

            for (k, v) in weights {
                if k.contains("position_ids") {
                    continue
                } else {
                    sanitizedWeights[k] = v
                }
            }

            return sanitizedWeights
        }
    }
}

// MARK: - Language Model Components (LFM2)

private enum Language {

    fileprivate class LFM2Attention: Module {
        let scale: Float
        let headDim: Int
        let heads: Int
        let kvHeads: Int

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        @ModuleInfo(key: "q_layernorm") var qLayerNorm: RMSNorm
        @ModuleInfo(key: "k_layernorm") var kLayerNorm: RMSNorm

        let rope: RoPE

        init(_ config: LFM2VLConfiguration.TextConfiguration) {
            let dim = config.hiddenSize
            self.heads = config.attentionHeads
            self.kvHeads = config.kvHeads
            self.headDim = dim / heads
            self.scale = pow(Float(headDim), -0.5)

            self._qProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
            self._kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._outProj.wrappedValue = Linear(heads * headDim, dim, bias: false)

            self._qLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
            self._kLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)

            self.rope = RoPE(
                dimensions: headDim,
                traditional: false,
                base: config.ropeTheta
            )
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = qProj(x)
            var keys = kProj(x)
            var values = vProj(x)

            queries = qLayerNorm(queries.reshaped(B, L, heads, -1)).transposed(0, 2, 1, 3)
            keys = kLayerNorm(keys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

            if let cache {
                queries = rope(queries, offset: cache.offset)
                keys = rope(keys, offset: cache.offset)
            } else {
                queries = rope(queries)
                keys = rope(keys)
            }

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

            return outProj(output)
        }
    }

    fileprivate class LFM2ShortConv: Module {
        let lCache: Int
        let hiddenSize: Int

        @ModuleInfo(key: "conv") var conv: Conv1d
        @ModuleInfo(key: "in_proj") var inProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: LFM2VLConfiguration.TextConfiguration, layerIdx: Int) {
            self.lCache = config.convLCache
            self.hiddenSize = config.hiddenSize
            let bias = config.convBias

            self._conv.wrappedValue = Conv1d(
                inputChannels: config.hiddenSize,
                outputChannels: config.hiddenSize,
                kernelSize: lCache,
                groups: config.hiddenSize,
                bias: bias
            )

            self._inProj.wrappedValue = Linear(config.hiddenSize, 3 * config.hiddenSize, bias: bias)
            self._outProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: bias)
        }

        func callAsFunction(_ x: MLXArray, cache: MambaCache?) -> MLXArray {
            let BCx = inProj(x)
            let BCxSplit = BCx.split(parts: 3, axis: -1)
            let B = BCxSplit[0]
            let C = BCxSplit[1]
            let xPart = BCxSplit[2]
            var Bx = B * xPart

            var state: MLXArray? = nil
            if let cache {
                state = cache[0]
            }
            if state == nil {
                state = MLXArray.zeros([Bx.dim(0), lCache - 1, hiddenSize], dtype: Bx.dtype)
            }

            Bx = concatenated([state!, Bx], axis: -2)
            if let cache {
                cache[0] = Bx[0..., (Bx.dim(1) - (lCache - 1))..., 0...]
            }

            let convOut = conv(Bx)
            let y = C * convOut
            return outProj(y)
        }
    }

    fileprivate class LFM2MLP: Module, UnaryLayer {
        @ModuleInfo(key: "w1") var w1: Linear
        @ModuleInfo(key: "w2") var w2: Linear
        @ModuleInfo(key: "w3") var w3: Linear

        init(_ config: LFM2VLConfiguration.TextConfiguration) {
            var adjustedFFDim = config.blockFFDim

            if config.blockAutoAdjustFFDim {
                adjustedFFDim = Int(Float(2 * adjustedFFDim) / 3.0)
                adjustedFFDim = Int(config.blockFFNDimMultiplier * Float(adjustedFFDim))
                adjustedFFDim =
                    config.blockMultipleOf
                    * ((adjustedFFDim + config.blockMultipleOf - 1) / config.blockMultipleOf)
            }

            self._w1.wrappedValue = Linear(config.blockDim, adjustedFFDim, bias: false)
            self._w2.wrappedValue = Linear(adjustedFFDim, config.blockDim, bias: false)
            self._w3.wrappedValue = Linear(config.blockDim, adjustedFFDim, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            w2(silu(w1(x)) * w3(x))
        }
    }

    fileprivate class LFM2DecoderLayer: Module {
        let isAttentionLayer: Bool

        @ModuleInfo(key: "self_attn") var attention: LFM2Attention?
        @ModuleInfo(key: "conv") var conv: LFM2ShortConv?
        @ModuleInfo(key: "feed_forward") var feedForward: LFM2MLP
        @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
        @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

        init(_ config: LFM2VLConfiguration.TextConfiguration, layerIdx: Int) {
            self.isAttentionLayer = config.fullAttnIdxs.contains(layerIdx)

            if isAttentionLayer {
                self._attention.wrappedValue = LFM2Attention(config)
            } else {
                self._conv.wrappedValue = LFM2ShortConv(config, layerIdx: layerIdx)
            }

            self._feedForward.wrappedValue = LFM2MLP(config)
            self._operatorNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
            self._ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let r: MLXArray
            if isAttentionLayer {
                r = attention!(operatorNorm(x), mask: mask, cache: cache)
            } else {
                r = conv!(operatorNorm(x), cache: cache as? MambaCache)
            }
            let h = x + r
            let out = h + feedForward(ffnNorm(h))
            return out
        }
    }

    fileprivate class LFM2ModelInner: Module {
        let config: LFM2VLConfiguration.TextConfiguration
        let vocabularySize: Int
        let numHiddenLayers: Int

        let layers: [LFM2DecoderLayer]

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

        init(_ config: LFM2VLConfiguration.TextConfiguration) {
            self.config = config
            self.vocabularySize = config.vocabularySize
            self.numHiddenLayers = config.hiddenLayers

            precondition(vocabularySize > 0)

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: vocabularySize, dimensions: config.hiddenSize)

            self.layers = (0 ..< numHiddenLayers).map { i in
                LFM2DecoderLayer(config, layerIdx: i)
            }

            self._embeddingNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
        }

        func callAsFunction(
            _ inputs: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
            cache: [KVCache]? = nil, inputEmbeddings: MLXArray? = nil
        ) -> MLXArray {
            var h = inputEmbeddings ?? embedTokens(inputs)

            let mask =
                mask
                ?? {
                    let firstAttnIdx = config.fullAttnIdxs.first ?? 0
                    let c =
                        cache != nil && firstAttnIdx < cache!.count ? cache![firstAttnIdx] : nil
                    return createAttentionMask(h: h, cache: c)
                }()

            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i])
            }

            return embeddingNorm(h)
        }
    }

    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        let config: LFM2VLConfiguration.TextConfiguration
        let modelType: String
        let model: LFM2ModelInner

        var kvHeads: [Int]

        init(_ config: LFM2VLConfiguration.TextConfiguration) {
            self.config = config
            self.modelType = config.modelType

            self.model = LFM2ModelInner(config)

            self.kvHeads = (0 ..< config.hiddenLayers).map { layerIdx in
                config.fullAttnIdxs.contains(layerIdx) ? config.kvHeads : 0
            }
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            mask: MLXArray? = nil,
            cache: [KVCache]? = nil,
            inputsEmbeds: MLXArray? = nil
        ) -> LMOutput {
            var out = model(
                inputs ?? MLXArray([0]), cache: cache, inputEmbeddings: inputsEmbeds)
            out = model.embedTokens.asLinear(out)
            return LMOutput(logits: out)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitizedWeights = [String: MLXArray]()

            for (name, param) in weights {
                var sanitizedParam = param

                if name.contains("conv.weight") {
                    if param.shape[param.shape.count - 1] > param.shape[1] {
                        sanitizedParam = param.transposed(0, 2, 1)
                    }
                }

                sanitizedWeights[name] = sanitizedParam
            }

            return sanitizedWeights
        }
    }
}

// MARK: - Multi-modal Projector

private class Lfm2VlMultiModalProjector: Module, UnaryLayer {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm?
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(config: LFM2VLConfiguration) {
        let inChannels =
            config.visionConfiguration.hiddenSize
            * (config.downsampleFactor * config.downsampleFactor)

        if config.projectorUseLayernorm {
            self._layerNorm.wrappedValue = LayerNorm(dimensions: inChannels)
        }

        self._linear1.wrappedValue = Linear(
            inChannels, config.projectorHiddenSize, bias: config.projectorBias)
        self._linear2.wrappedValue = Linear(
            config.projectorHiddenSize, config.textConfiguration.hiddenSize,
            bias: config.projectorBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        if let layerNorm {
            x = layerNorm(x)
        }
        x = linear1(x)
        x = gelu(x)
        x = linear2(x)
        return x
    }
}

// MARK: - PixelUnshuffleBlock

private class PixelUnshuffleBlock: Module, UnaryLayer {
    let factor: Int

    init(factor: Int) {
        self.factor = factor
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        var (n, w, h, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))

        // Pad width if necessary
        if w % factor != 0 {
            let padW = factor - (w % factor)
            let padding = MLXArray.zeros([n, padW, h, c], dtype: x.dtype)
            x = concatenated([x, padding], axis: 1)
            (n, w, h, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        }

        // Pad height if necessary
        if h % factor != 0 {
            let padH = factor - (h % factor)
            let padding = MLXArray.zeros([n, w, padH, c], dtype: x.dtype)
            x = concatenated([x, padding], axis: 2)
            (n, w, h, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        }

        x = x.reshaped(n, w, h / factor, c * factor)
        x = x.transposed(0, 2, 1, 3)
        x = x.reshaped(n, h / factor, w / factor, c * factor * factor)
        x = x.transposed(0, 2, 1, 3)

        return x
    }
}

// MARK: - Processor

/// LFM2 VL VLM `UserInputProcessor`.
///
/// This is meant to be used with ``LFM2VL`` and is typically created by ``VLMModelFactory``.
public struct LFM2VLProcessor: UserInputProcessor {
    private let config: LFM2VLProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: LFM2VLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    /// Preprocess a single image
    func preprocess(image: CIImage, targetSize: CGSize) -> CIImage {
        image
            .toSRGB()
            .resampled(to: targetSize, method: .bicubic)
            .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
    }

    /// Split an image into patches
    func splitIntoPatchesAndPreprocess(
        image: CIImage,
        processing: UserInput.Processing?
    ) throws -> (pixels: MLXArray, spatialShape: (Int, Int), pixelAttentionMask: MLXArray) {
        // Apply user processing if any
        let image = MediaProcessing.apply(image, processing: processing)

        // Get image dimensions
        let width = Int(image.extent.width)
        let height = Int(image.extent.height)

        // Calculate tile dimensions
        let tileSize = config.tileSize
        let patchSize = config.encoderPatchSize

        // Calculate number of tiles
        let numTilesH = max(1, min(config.maxTiles, Int(ceil(Double(height) / Double(tileSize)))))
        let numTilesW = max(1, min(config.maxTiles, Int(ceil(Double(width) / Double(tileSize)))))

        // Calculate actual resize dimensions
        let resizedHeight = numTilesH * tileSize
        let resizedWidth = numTilesW * tileSize

        // Resize the image
        let resizedSize = CGSize(width: resizedWidth, height: resizedHeight)
        let resizedImage = image.toSRGB().resampled(to: resizedSize, method: .bicubic)

        // Calculate patches per tile
        let patchesPerTileH = tileSize / patchSize
        let patchesPerTileW = tileSize / patchSize

        // Total number of patches
        let totalPatchesH = numTilesH * patchesPerTileH
        let totalPatchesW = numTilesW * patchesPerTileW

        // Convert to MLXArray and extract patches
        let normalizedImage = resizedImage.normalized(
            mean: config.imageMeanTuple, std: config.imageStdTuple)
        var imageArray = MediaProcessing.asMLXArray(normalizedImage)  // [1, C, H, W]

        // Reshape to extract patches: [1, C, H, W] -> [1, numPatches, patchSize*patchSize*C]
        imageArray = imageArray.transposed(0, 2, 3, 1)  // [1, H, W, C]

        // Extract patches
        var patches = [MLXArray]()
        for ph in 0 ..< totalPatchesH {
            for pw in 0 ..< totalPatchesW {
                let startH = ph * patchSize
                let startW = pw * patchSize
                let patch = imageArray[
                    0, startH ..< (startH + patchSize), startW ..< (startW + patchSize), 0...]
                patches.append(patch.flattened())
            }
        }

        let pixelValues = stacked(patches, axis: 0).expandedDimensions(axis: 0)  // [1, numPatches, patchDim]

        // Create attention mask (all ones for valid patches)
        let numPatches = totalPatchesH * totalPatchesW
        let pixelAttentionMask = MLXArray.ones([1, numPatches]).asType(.int32)

        return (pixelValues, (totalPatchesH, totalPatchesW), pixelAttentionMask)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Qwen2VLMessageGenerator().generate(from: input)

        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext
        )

        // Text-only input
        if input.images.isEmpty {
            return LMInput(tokens: MLXArray(promptTokens))
        }

        // Process images
        var allPixelValues = [MLXArray]()
        var allSpatialShapes = [(Int, Int)]()
        var allPixelAttentionMasks = [MLXArray]()

        for imageInput in input.images {
            let image = try imageInput.asCIImage()
            let (pixels, spatialShape, pixelAttentionMask) = try splitIntoPatchesAndPreprocess(
                image: image, processing: input.processing)
            allPixelValues.append(pixels)
            allSpatialShapes.append(spatialShape)
            allPixelAttentionMasks.append(pixelAttentionMask)
        }

        // Calculate how many image tokens we need per image
        let downsampleFactor = config.downsampleFactor
        var totalImageTokens = 0
        for shape in allSpatialShapes {
            let h = shape.0 / downsampleFactor
            let w = shape.1 / downsampleFactor
            totalImageTokens += h * w
        }

        // Replace image placeholder tokens with the correct count
        // image_token_id is 396 for LFM2 VL models
        let imageTokenId = 396
        var newPromptTokens = [Int]()
        var imageIdx = 0
        var i = 0
        while i < promptTokens.count {
            if promptTokens[i] == imageTokenId {
                // Count consecutive image tokens
                var count = 0
                while i + count < promptTokens.count && promptTokens[i + count] == imageTokenId {
                    count += 1
                }
                // Replace with correct number for this image
                if imageIdx < allSpatialShapes.count {
                    let shape = allSpatialShapes[imageIdx]
                    let h = shape.0 / downsampleFactor
                    let w = shape.1 / downsampleFactor
                    let numTokens = h * w
                    for _ in 0 ..< numTokens {
                        newPromptTokens.append(imageTokenId)
                    }
                    imageIdx += 1
                }
                i += count
            } else {
                newPromptTokens.append(promptTokens[i])
                i += 1
            }
        }
        promptTokens = newPromptTokens

        // Concatenate all image data
        let pixelValuesConcatenated = concatenated(allPixelValues, axis: 0)

        // Convert spatial shapes to THW format (t=1 for images)
        let frames = allSpatialShapes.map { THW(1, $0.0, $0.1) }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)

        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: LMInput.ProcessedImage(
                pixels: pixelValuesConcatenated,
                frames: frames
            )
        )
    }
}

// MARK: - Model

/// LFM2 VL VLM
///
/// This is typically created by ``VLMModelFactory``.
public class LFM2VL: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") private var visionModel: Vision.VisionModel
    @ModuleInfo(key: "multi_modal_projector") private var multiModalProjector:
        Lfm2VlMultiModalProjector
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel
    @ModuleInfo(key: "pixel_unshuffle") private var pixelUnshuffle: PixelUnshuffleBlock?

    public let config: LFM2VLConfiguration

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public var loraLayers: [Module] {
        languageModel.model.layers.map { $0 as Module }
    }

    public init(_ config: LFM2VLConfiguration) {
        self.config = config

        self._visionModel.wrappedValue = Vision.VisionModel(
            config: config.visionConfiguration,
            visionFeatureLayer: config.visionFeatureLayer
        )

        if config.downsampleFactor > 1 {
            self._pixelUnshuffle.wrappedValue = PixelUnshuffleBlock(factor: config.downsampleFactor)
        }

        self._multiModalProjector.wrappedValue = Lfm2VlMultiModalProjector(config: config)
        self._languageModel.wrappedValue = Language.LanguageModel(config.textConfiguration)
    }

    private func getInputEmbeddings(
        inputIds: MLXArray,
        pixelValues: MLXArray?,
        spatialShapes: MLXArray?,
        pixelAttentionMask: MLXArray?
    ) -> MLXArray {
        // Ensure inputIds has batch dimension
        var batchedInputIds = inputIds
        if inputIds.ndim == 1 {
            batchedInputIds = inputIds.expandedDimensions(axis: 0)
        }

        var inputsEmbeds = languageModel.model.embedTokens(batchedInputIds)

        // Ensure embeddings have batch dimension
        if inputsEmbeds.ndim == 2 {
            inputsEmbeds = inputsEmbeds.expandedDimensions(axis: 0)
        }

        guard let pixelValues, let spatialShapes, let pixelAttentionMask else {
            return inputsEmbeds
        }

        // Get the output hidden states from the vision model
        let visionOutput = visionModel(
            pixelValues, outputHiddenStates: true, spatialShapes: spatialShapes)
        let hiddenStates = visionOutput.lastHiddenState

        // Get feature lengths from attention mask
        let imgFeatureLengths = sum(pixelAttentionMask, axis: 1)

        var imageFeatures = [MLXArray]()

        for imgIdx in 0 ..< hiddenStates.dim(0) {
            var feature = hiddenStates[imgIdx]
            let featureLength = imgFeatureLengths[imgIdx].item(Int.self)

            // Slice to valid features
            feature = feature[0 ..< featureLength].expandedDimensions(axis: 0)

            // Get spatial dimensions
            let featureOrgH = spatialShapes[imgIdx, 0].item(Int.self)
            let featureOrgW = spatialShapes[imgIdx, 1].item(Int.self)

            // Reshape to spatial dimensions
            feature = feature.reshaped(1, featureOrgH, featureOrgW, -1)

            // Apply pixel unshuffle if configured
            if let pixelUnshuffle {
                feature = pixelUnshuffle(feature)
            }

            // Project to language model dimension
            var imgEmbedding = multiModalProjector(feature)

            // Flatten back
            imgEmbedding = imgEmbedding.reshaped(-1, imgEmbedding.dim(-1))
            imageFeatures.append(imgEmbedding)
        }

        let concatenatedImageFeatures = concatenated(imageFeatures, axis: 0)

        // Merge image features with text embeddings
        return mergeInputIdsWithImageFeatures(
            imageFeatures: concatenatedImageFeatures,
            inputsEmbeds: inputsEmbeds,
            inputIds: inputIds,
            imageTokenIndex: config.imageTokenIndex
        )
    }

    private func mergeInputIdsWithImageFeatures(
        imageFeatures: MLXArray,
        inputsEmbeds: MLXArray,
        inputIds: MLXArray,
        imageTokenIndex: Int
    ) -> MLXArray {
        // Find image token positions
        var imageIndices = [Int]()
        for (i, v) in inputIds.flattened().asArray(Int.self).enumerated() {
            if v == imageTokenIndex {
                imageIndices.append(i)
            }
        }

        let nImageFeatures = imageFeatures.dim(0)
        if imageIndices.count != nImageFeatures {
            fatalError(
                "Image features and image tokens do not match: tokens: \(imageIndices.count), features \(nImageFeatures)"
            )
        }

        // Make sure shapes match before assignment
        var result = inputsEmbeds
        if result.ndim == 2 {
            result = result.expandedDimensions(axis: 0)
        }

        // Assign image features to the image token positions
        if imageFeatures.ndim == 2 {
            let reshapedFeatures = imageFeatures.expandedDimensions(axis: 0)
            result[0..., MLXArray(imageIndices), 0...] = reshapedFeatures
        } else {
            result[0..., MLXArray(imageIndices), 0...] = imageFeatures
        }

        return result
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let dtype = visionModel.embeddings.patchEmbedding.weight.dtype

        // Get image data if available
        let pixelValues = input.image?.pixels.asType(dtype)

        var spatialShapes: MLXArray? = nil
        var pixelAttentionMask: MLXArray? = nil

        if pixelValues != nil, let frames = input.image?.frames, !frames.isEmpty {
            // Extract spatial shapes from frames (THW format where t=1 for images)

            // Convert frames to spatial shapes array [numImages, 2]
            let shapeArrays = frames.map { MLXArray([$0.h, $0.w]) }
            spatialShapes = stacked(shapeArrays, axis: 0)

            // Create attention mask based on actual feature lengths per image
            var maskArrays = [MLXArray]()
            for frame in frames {
                let numPatches = frame.h * frame.w
                let imageMask = MLXArray.ones([numPatches]).asType(.int32)
                maskArrays.append(imageMask)
            }
            // Stack masks - for now assuming single batch processing
            if maskArrays.count == 1 {
                pixelAttentionMask = maskArrays[0].expandedDimensions(axis: 0)
            } else {
                // For multiple images, we need to pad to max length
                let maxLen = maskArrays.map { $0.dim(0) }.max() ?? 0
                let paddedMasks = maskArrays.map { mask -> MLXArray in
                    if mask.dim(0) < maxLen {
                        let padding = MLXArray.zeros([maxLen - mask.dim(0)]).asType(.int32)
                        return concatenated([mask, padding], axis: 0)
                    }
                    return mask
                }
                pixelAttentionMask = stacked(paddedMasks, axis: 0)
            }
        } else if let pixels = pixelValues {
            // Fallback: infer spatial shapes from pixel dimensions (assumes square)
            let numPatches = pixels.dim(1)
            let side = Int(sqrt(Double(numPatches)))
            spatialShapes = MLXArray([side, side]).expandedDimensions(axis: 0)
            pixelAttentionMask = MLXArray.ones([1, numPatches]).asType(.int32)
        }

        let inputEmbeddings = getInputEmbeddings(
            inputIds: input.text.tokens,
            pixelValues: pixelValues,
            spatialShapes: spatialShapes,
            pixelAttentionMask: pixelAttentionMask
        )

        // Chunked prefill — see Qwen25VL / vmlx #50/#51.
        let result = chunkedPrefillEmbedding(
            inputEmbedding: inputEmbeddings,
            cache: cache,
            prefillStepSize: windowSize ?? 512
        ) { chunk in
            languageModel(nil, cache: cache, inputsEmbeds: chunk)
        }

        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        func transformKey(_ key: String) -> String {
            var key = key

            if key.contains("vision_tower") {
                key =
                    key
                    .replacingOccurrences(of: "model.", with: "")
                    .replacingOccurrences(of: "vision_encoder", with: "encoder")
                    .replacingOccurrences(of: "vision_embeddings", with: "embeddings")
                    .replacingOccurrences(of: "vision_post_layernorm", with: "post_layernorm")
            }

            if key.contains("language_model") {
                key = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            }

            if key.contains("multi_modal_projector") {
                key = key.replacingOccurrences(
                    of: "model.multi_modal_projector", with: "multi_modal_projector")
            }

            return key
        }

        var sanitizedWeights = [String: MLXArray]()
        for (k, v) in weights {
            let newKey = transformKey(k)

            // Handle conv weight transposition
            var value = v
            if newKey.contains("conv.weight") {
                if v.shape[v.shape.count - 1] > v.shape[1] {
                    value = v.transposed(0, 2, 1)
                }
            }

            sanitizedWeights[newKey] = value
        }

        return sanitizedWeights
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let textConfig = config.textConfiguration
        return (0 ..< textConfig.hiddenLayers).map { layerIdx in
            if textConfig.fullAttnIdxs.contains(layerIdx) {
                KVCacheSimple()
            } else {
                MambaCache()
            }
        }
    }
}

// MARK: - Configuration

/// Configuration for ``LFM2VL``
public struct LFM2VLConfiguration: Codable, Sendable {

    public struct TextConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let vocabularySize: Int
        private let _normEps: Float?
        public var normEps: Float { _normEps ?? 1e-5 }
        private let _convBias: Bool?
        public var convBias: Bool { _convBias ?? false }
        private let _convLCache: Int?
        public var convLCache: Int { _convLCache ?? 3 }
        private let _blockDim: Int?
        public var blockDim: Int { _blockDim ?? hiddenSize }
        private let _blockFFDim: Int?
        public var blockFFDim: Int { _blockFFDim ?? hiddenSize }
        private let _blockMultipleOf: Int?
        public var blockMultipleOf: Int { _blockMultipleOf ?? 256 }
        private let _blockFFNDimMultiplier: Float?
        public var blockFFNDimMultiplier: Float { _blockFFNDimMultiplier ?? 1.0 }
        private let _blockAutoAdjustFFDim: Bool?
        public var blockAutoAdjustFFDim: Bool { _blockAutoAdjustFFDim ?? true }
        private let _fullAttnIdxs: [Int]?
        private let layerTypes: [String]?
        public var fullAttnIdxs: [Int] {
            if let fullAttnIdxs = _fullAttnIdxs {
                return fullAttnIdxs
            }

            if let layerTypes {
                return layerTypes.enumerated().compactMap { index, layerType in
                    layerType == "full_attention" ? index : nil
                }
            }

            return Array(0 ..< hiddenLayers)
        }
        private let _ropeTheta: Float?
        public var ropeTheta: Float { _ropeTheta ?? 1000000.0 }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case vocabularySize = "vocab_size"
            case _normEps = "norm_eps"
            case _convBias = "conv_bias"
            case _convLCache = "conv_L_cache"
            case _blockDim = "block_dim"
            case _blockFFDim = "block_ff_dim"
            case _blockMultipleOf = "block_multiple_of"
            case _blockFFNDimMultiplier = "block_ffn_dim_multiplier"
            case _blockAutoAdjustFFDim = "block_auto_adjust_ff_dim"
            case _fullAttnIdxs = "full_attn_idxs"
            case layerTypes = "layer_types"
            case _ropeTheta = "rope_theta"
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        private let _numChannels: Int?
        public var numChannels: Int { _numChannels ?? 3 }
        private let _imageSize: Int?
        public var imageSize: Int { _imageSize ?? 224 }
        private let _patchSize: Int?
        public var patchSize: Int { _patchSize ?? 16 }
        private let _numPatches: Int?
        public var numPatches: Int { _numPatches ?? 256 }
        private let _layerNormEps: Float?
        public var layerNormEps: Float { _layerNormEps ?? 1e-6 }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case _numChannels = "num_channels"
            case _imageSize = "image_size"
            case _patchSize = "patch_size"
            case _numPatches = "num_patches"
            case _layerNormEps = "layer_norm_eps"
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    private let _downsampleFactor: Int?
    public var downsampleFactor: Int { _downsampleFactor ?? 2 }
    private let _imageTokenId: Int?
    public var imageTokenIndex: Int { _imageTokenId ?? 396 }
    private let _projectorBias: Bool?
    public var projectorBias: Bool { _projectorBias ?? true }
    private let _projectorHiddenSize: Int?
    public var projectorHiddenSize: Int { _projectorHiddenSize ?? 2560 }
    private let _projectorUseLayernorm: Bool?
    public var projectorUseLayernorm: Bool { _projectorUseLayernorm ?? true }
    private let _visionFeatureLayer: Int?
    /// Which vision encoder layer to use for features. -1 means use all layers (default).
    public var visionFeatureLayer: Int { _visionFeatureLayer ?? -1 }
    private let _doImageSplitting: Bool?
    public var doImageSplitting: Bool { _doImageSplitting ?? true }
    private let _maxImageTokens: Int?
    public var maxImageTokens: Int { _maxImageTokens ?? 256 }
    private let _maxNumPatches: Int?
    public var maxNumPatches: Int { _maxNumPatches ?? 1024 }
    private let _minImageTokens: Int?
    public var minImageTokens: Int { _minImageTokens ?? 64 }
    private let _minTiles: Int?
    public var minTiles: Int { _minTiles ?? 2 }
    private let _useThumbnail: Bool?
    public var useThumbnail: Bool { _useThumbnail ?? false }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case _downsampleFactor = "downsample_factor"
        case _imageTokenId = "image_token_id"
        case _projectorBias = "projector_bias"
        case _projectorHiddenSize = "projector_hidden_size"
        case _projectorUseLayernorm = "projector_use_layernorm"
        case _visionFeatureLayer = "vision_feature_layer"
        case _doImageSplitting = "do_image_splitting"
        case _maxImageTokens = "max_image_tokens"
        case _maxNumPatches = "max_num_patches"
        case _minImageTokens = "min_image_tokens"
        case _minTiles = "min_tiles"
        case _useThumbnail = "use_thumbnail"
    }
}

/// Configuration for ``LFM2VLProcessor``
public struct LFM2VLProcessorConfiguration: Codable, Sendable {
    // Fields at top level (matching typical preprocessor_config.json structure)
    private let _imageMean: [CGFloat]?
    private let _imageStd: [CGFloat]?
    private let _tileSize: Int?
    private let _encoderPatchSize: Int?
    private let _maxTiles: Int?
    private let _downsampleFactor: Int?

    // Default values matching LFM2 VL models
    public var imageMean: [CGFloat] {
        _imageMean ?? [0.5, 0.5, 0.5]
    }
    public var imageStd: [CGFloat] {
        _imageStd ?? [0.5, 0.5, 0.5]
    }
    public var tileSize: Int { _tileSize ?? 512 }
    public var encoderPatchSize: Int { _encoderPatchSize ?? 16 }
    public var maxTiles: Int { _maxTiles ?? 10 }
    public var downsampleFactor: Int { _downsampleFactor ?? 2 }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case _imageMean = "image_mean"
        case _imageStd = "image_std"
        case _tileSize = "tile_size"
        case _encoderPatchSize = "encoder_patch_size"
        case _maxTiles = "max_tiles"
        case _downsampleFactor = "downsample_factor"
    }
}
