//
// Mistral3VLMJANGTQ — JANGTQ-quantized Mistral 3 family VLM with
// Pixtral vision tower. Sibling of `Mistral3VLM`; only the language-
// model inner differs (JANGTQDenseLinear in attention Q/K/V/O + MLP
// gate/up/down). Pixtral vision tower stays vanilla
// (mxtq_bits.vision_tower=passthrough_fp16).
//
// Wired via VLMModelFactory's `mistral3` closure when
// `weight_format == "mxtq"`. Bits + seed come from `mxtq_bits` /
// `mxtq_seed` fields on `Mistral3VLMConfiguration` (added 2026-04-30).
//
// All JANGTQ-specific classes live at module scope here (NOT inside
// `Language` enum) because that enum is `private` to Mistral3.swift.
// Names are prefixed `Mistral3JANGTQ*` to avoid collision with any
// other VLM file's scoped symbols.
//

import CoreImage
import Foundation
import MLX
import vMLXLMCommon
import MLXNN

// MARK: - JANGTQ Attention

internal final class Mistral3JANGTQAttention: Module {
    let config: Mistral3VLMTextConfiguration
    let scale: Float
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int

    @ModuleInfo(key: "q_proj") var wq: JANGTQDenseLinear
    @ModuleInfo(key: "k_proj") var wk: JANGTQDenseLinear
    @ModuleInfo(key: "v_proj") var wv: JANGTQDenseLinear
    @ModuleInfo(key: "o_proj") var wo: JANGTQDenseLinear

    let rope: RoPELayer

    init(_ config: Mistral3VLMTextConfiguration, bits: Int, seed: Int) {
        self.config = config
        let dim = config.hiddenSize
        self.nHeads = config.numAttentionHeads
        self.nKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim ?? (config.hiddenSize / nHeads)
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = JANGTQDenseLinear(
            inFeatures: dim, outFeatures: nHeads * headDim,
            bits: bits, seed: seed, bias: false)
        self._wk.wrappedValue = JANGTQDenseLinear(
            inFeatures: dim, outFeatures: nKVHeads * headDim,
            bits: bits, seed: seed, bias: false)
        self._wv.wrappedValue = JANGTQDenseLinear(
            inFeatures: dim, outFeatures: nKVHeads * headDim,
            bits: bits, seed: seed, bias: false)
        self._wo.wrappedValue = JANGTQDenseLinear(
            inFeatures: nHeads * headDim, outFeatures: dim,
            bits: bits, seed: seed, bias: false)

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

// MARK: - JANGTQ MLP

internal final class Mistral3JANGTQMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: JANGTQDenseLinear
    @ModuleInfo(key: "down_proj") var down: JANGTQDenseLinear
    @ModuleInfo(key: "up_proj") var up: JANGTQDenseLinear

    init(_ config: Mistral3VLMTextConfiguration, bits: Int, seed: Int) {
        let dim = config.hiddenSize
        let hiddenDim = config.intermediateSize
        self._gate.wrappedValue = JANGTQDenseLinear(
            inFeatures: dim, outFeatures: hiddenDim,
            bits: bits, seed: seed, bias: false)
        self._down.wrappedValue = JANGTQDenseLinear(
            inFeatures: hiddenDim, outFeatures: dim,
            bits: bits, seed: seed, bias: false)
        self._up.wrappedValue = JANGTQDenseLinear(
            inFeatures: dim, outFeatures: hiddenDim,
            bits: bits, seed: seed, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let g = silu(gate(x))
        let u = up(x)
        return down(g * u)
    }
}

// MARK: - JANGTQ Transformer Block

internal final class Mistral3JANGTQTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Mistral3JANGTQAttention
    let mlp: Mistral3JANGTQMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let useSliding: Bool

    init(_ config: Mistral3VLMTextConfiguration,
         bits: Int, seed: Int,
         useSliding: Bool = false)
    {
        self.useSliding = useSliding
        self._attention.wrappedValue = Mistral3JANGTQAttention(config, bits: bits, seed: seed)
        self.mlp = Mistral3JANGTQMLP(config, bits: bits, seed: seed)
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
            inputLayerNorm(x), attentionScale: attentionScale,
            mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }
}

// MARK: - llama4 attention scaling helper (free function — Language.getLlama4AttentionScale is private)

internal func mistral3VLMJANGTQGetLlama4AttentionScale(
    start: Int, stop: Int, beta: Float, maxPositionEmbeddings: Int
) -> MLXArray {
    let positions = MLXArray(start ..< stop).asType(.float32)
    let scaling = 1 + beta * MLX.log(1 + MLX.floor(positions / Float(maxPositionEmbeddings)))
    return expandedDimensions(scaling, axis: -1)
}

internal func mistral3VLMJANGTQGetLlama4AttentionScale(
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

// MARK: - JANGTQ Inner Model

internal final class Mistral3JANGTQModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [Mistral3JANGTQTransformerBlock]
    let norm: RMSNorm
    let config: Mistral3VLMTextConfiguration
    let layerTypes: [String]
    let slidingWindow: Int?
    let faIndex: Int
    let swaIndex: Int?

    init(_ config: Mistral3VLMTextConfiguration, bits: Int, seed: Int) {
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
            Mistral3JANGTQTransformerBlock(
                config, bits: bits, seed: seed,
                useSliding: layerType == "sliding_attention")
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

        let cacheArr = cache ?? []
        let faCache = faIndex < cacheArr.count ? cacheArr[faIndex] : nil
        let faMask = createAttentionMask(h: h, cache: faCache)

        var swaMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        if let swaIndex, let slidingWindow, swaIndex < cacheArr.count {
            swaMask = createAttentionMask(
                h: h, cache: cacheArr[swaIndex], windowSize: slidingWindow)
        }

        let beta = config.ropeParameters?["llama_4_scaling_beta"]?.asFloat() ?? 0.0
        let originalMaxPos =
            config.ropeParameters?["original_max_position_embeddings"]?.asInt()
            ?? config.maxPositionEmbeddings ?? 4096
        let attentionScale: MLXArray
        if beta != 0, let offsetArray = graphOffsetArray(for: cacheArr.first) {
            attentionScale = mistral3VLMJANGTQGetLlama4AttentionScale(
                size: h.dim(1),
                offset: offsetArray,
                beta: beta,
                maxPositionEmbeddings: originalMaxPos
            ).asType(h.dtype)
        } else if beta != 0 {
            let offset = cacheArr.first?.offset ?? 0
            attentionScale = mistral3VLMJANGTQGetLlama4AttentionScale(
                start: offset,
                stop: offset + h.dim(1),
                beta: beta,
                maxPositionEmbeddings: originalMaxPos
            ).asType(h.dtype)
        } else {
            attentionScale = MLXArray.ones([h.dim(1), 1]).asType(h.dtype)
        }

        for (i, layer) in layers.enumerated() {
            let mask = layer.useSliding ? swaMask : faMask
            h = layer(
                h, attentionScale: attentionScale, mask: mask,
                cache: i < cacheArr.count ? cacheArr[i] : nil)
        }

        return norm(h)
    }
}

// MARK: - JANGTQ Language Model wrapper

internal final class Mistral3JANGTQLanguageModel: Module, KVCacheDimensionProvider {
    let config: Mistral3VLMTextConfiguration
    let modelType: String

    @ModuleInfo(key: "model") private var model: Mistral3JANGTQModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    var kvHeads: [Int] {
        let layerTypes =
            config.layerTypes
            ?? Array(repeating: "full_attention", count: config.numHiddenLayers)
        return layerTypes.map { _ in config.numKeyValueHeads }
    }

    var embedTokens: Embedding {
        model.embedTokens
    }

    var layers: [Mistral3JANGTQTransformerBlock] {
        model.layers
    }

    init(_ config: Mistral3VLMTextConfiguration, bits: Int, seed: Int) {
        self.config = config
        self.modelType = config.modelType
        self._model.wrappedValue = Mistral3JANGTQModelInner(
            config, bits: bits, seed: seed)

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabSize, bias: false)
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

// MARK: - Mistral3VLM JANGTQ wrapper

/// JANGTQ-quantized Mistral3 VLM. Sibling of `Mistral3VLM`; differs
/// only in the language-model inner. Pixtral vision tower stays
/// vanilla.
public class Mistral3VLMJANGTQ: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: PixtralVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Mistral3JANGTQLanguageModel
    @ModuleInfo(key: "multi_modal_projector") private var multiModalProjector:
        Mistral3MultiModalProjector

    public let config: Mistral3VLMConfiguration
    let visionFeatureLayer: Int

    public var vocabularySize: Int { config.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public init(_ config: Mistral3VLMConfiguration, bits: Int = 2, seed: Int = 42) {
        self.config = config
        self.visionFeatureLayer = config.visionFeatureLayer

        self._visionTower.wrappedValue = PixtralVision.VisionModel(config.visionConfig)
        self._languageModel.wrappedValue = Mistral3JANGTQLanguageModel(
            config.textConfig, bits: bits, seed: seed)
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

        if pixelValues.ndim == 3 {
            pixelValues = pixelValues.expandedDimensions(axis: 0)
        }

        let (_, _, hiddenStates) = visionTower(
            pixelValues.transposed(0, 2, 3, 1),
            outputHiddenStates: true
        )

        guard let hiddenStates else {
            fatalError("Vision model must return hidden states")
        }

        let layerIndex =
            visionFeatureLayer < 0
            ? hiddenStates.count + visionFeatureLayer
            : visionFeatureLayer
        let selectedFeatures = hiddenStates[layerIndex]

        let imageFeatures = multiModalProjector(selectedFeatures, imageSizes: imageSizes)

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
            imageFeatures.dim(0), imageFeatures.dim(1), imageFeatures.dim(2))
        let inputIdArray: [Int32] = inputIds[0].asArray(Int32.self)
        let imagePositions = inputIdArray.enumerated().compactMap {
            $1 == Int32(imageTokenIndex) ? $0 : nil
        }
        guard imagePositions.count == numImagePatches else {
            fatalError(
                "Image token count (\(imagePositions.count)) does not match image patches"
                + " (\(numImagePatches)).")
        }

        var textSegments: [MLXArray] = []
        var startIdx = 0
        for position in imagePositions {
            textSegments.append(inputsEmbeds[0..., startIdx ..< position, 0...])
            startIdx = position + 1
        }
        let splitIndices = Array(1 ..< numImagePatches)
        let imageEmbeddings = MLX.split(imageFeatures, indices: splitIndices, axis: 1)

        var finalEmbeddings: [MLXArray] = []
        for (text, image) in zip(textSegments, imageEmbeddings) {
            finalEmbeddings.append(text)
            finalEmbeddings.append(image)
        }
        finalEmbeddings.append(inputsEmbeds[0..., startIdx..., 0...])
        return MLX.concatenated(finalEmbeddings, axis: 1)
    }

    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
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

        let embeddings = getInputEmbeddings(
            inputIds: inputIds,
            pixelValues: pixelValues,
            imageSizes: imageSizes
        )

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
            if key.contains("vision_tower") && !key.contains("vision_model") {
                if key.contains("transformer") || key.contains("patch_conv")
                    || key.contains("ln_pre")
                {
                    newKey = key.replacingOccurrences(
                        of: "vision_tower", with: "vision_tower.vision_model")
                }
                // 2026-05-01: real Mistral 3.5 VLM JANGTQ bundles ship
                // vision keys as `model.vision_tower.transformer.…`
                // (verified on Mistral-Medium-3.5-128B-JANGTQ
                // safetensors index). The replacement above leaves the
                // outer `model.` prefix in place — but the model wraps
                // the vision tower at the root (`@ModuleInfo(key:
                // "vision_tower")`), so the loader reports
                // `Unhandled keys ["model"]`. Strip the leading
                // `model.` so the path lands at
                // `vision_tower.vision_model.…`. Idempotent: keys
                // already missing the `model.` prefix pass through.
                if newKey.hasPrefix("model.vision_tower.") {
                    newKey = String(newKey.dropFirst("model.".count))
                }
            } else if key.contains("vision_encoder") && !key.contains("vision_tower") {
                if key.contains("transformer") || key.contains("patch_conv")
                    || key.contains("ln_pre")
                {
                    newKey = key.replacingOccurrences(
                        of: "model.vision_encoder", with: "vision_tower.vision_model")
                }
            } else if key.contains("model.language_model")
                && !key.contains("language_model.model")
            {
                newKey = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if key.contains("lm_head") && !key.contains("language_model") {
                newKey = key.replacingOccurrences(of: "lm_head", with: "language_model.lm_head")
            } else if key.contains("model.vision_projection") {
                newKey = key.replacingOccurrences(
                    of: "model.vision_projection", with: "multi_modal_projector")
            } else if key.hasPrefix("model.multi_modal_projector.") {
                // 2026-05-01: real Mistral 3.5 VLM JANGTQ bundles ship
                // projector keys at `model.multi_modal_projector.…`
                // (linear_1, linear_2, norm, patch_merger.merging_layer).
                // The wrapper class declares the projector at root via
                // `@ModuleInfo(key: "multi_modal_projector")`. Strip the
                // `model.` prefix so keys land at the root path.
                // Verified on Mistral-Medium-3.5-128B-JANGTQ.
                newKey = String(key.dropFirst("model.".count))
            }

            if newKey.contains("self_attn.rotary_emb.inv_freq") {
                continue
            }

            // Drop JANGTQ per-tensor `.tq_bits` scalars. The bits-width is
            // consumed from `mxtq_bits` / `quantization.profile` at config
            // time (passed into Mistral3VLMJANGTQ's init via the bits arg);
            // the per-tensor scalar in the safetensors is redundant and
            // JANGTQDenseLinear's @ParameterInfo schema doesn't accept it.
            // Without this drop, every Mistral 3 / 3.5 VLM JANGTQ bundle
            // throws `Unhandled keys [tq_bits]` at first weight load. Same
            // pattern as LagunaJANGTQ.sanitize / Mistral3TextJANGTQ.sanitize
            // / DeepseekV4JANGTQ.sanitize / NemotronHJANGTQ.sanitize.
            if newKey.hasSuffix(".tq_bits") {
                continue
            }

            // Tied embeddings: when config.tieWordEmbeddings is true the
            // language model's lm_head shares weights with embed_tokens.
            // The model class declares `var lmHead: Linear?` and only
            // initialises it when NOT tied, so a redundant lm_head.weight
            // shipped in the safetensors throws an unhandled-key error at
            // load. Drop it. Mirror of Mistral3TextJANGTQ.sanitize and
            // LagunaJANGTQ.sanitize behaviour.
            if config.textConfig.tieWordEmbeddings,
                newKey == "language_model.lm_head.weight"
                    || newKey == "language_model.lm_head.scales"
                    || newKey == "language_model.lm_head.biases"
            {
                continue
            }

            // 2026-05-01: Robust fallback for LLM-shape keys.
            //
            // Some Mistral 3 family VLM JANGTQ bundles (especially
            // Mistral 3.5 with `ministral3` outer model_type) ship the
            // language-model weights without the `language_model.`
            // wrapper — i.e. plain `model.embed_tokens.*`,
            // `model.norm.*`, `model.layers.<i>.<...>` rather than
            // `model.language_model.<...>` or `language_model.model.<...>`.
            //
            // None of the existing elif rules match those, so they
            // pass through unchanged and the loader reports
            // `Unhandled keys ["model"]` (deduped to the top segment).
            //
            // Re-prefix any leftover `model.<llm-key>` with
            // `language_model.` so they land at
            // `language_model.model.<llm-key>` — the path the wrapper
            // class expects. Vision-tower / projector keys were
            // already transformed by the rules above and start with
            // `vision_tower.` or `multi_modal_projector.`, so this
            // fallback is safe.
            if newKey.hasPrefix("model.")
                && !newKey.hasPrefix("model.vision_")
                && !newKey.contains("language_model")
                && !newKey.contains("multi_modal_projector")
            {
                newKey = "language_model." + newKey
            }

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

// LoRA support: empty layers — JANGTQ Linear is not LoRA-quantizable
// (codebook lookup + hadamard rotate doesn't compose with LoRA delta).
extension Mistral3VLMJANGTQ: LoRAModel {
    public var loraLayers: [Module] { [] }
}
