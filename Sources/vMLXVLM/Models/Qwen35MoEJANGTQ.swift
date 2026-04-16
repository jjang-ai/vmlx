//
//  Qwen35MoEJANGTQ.swift
//  vMLXVLM
//
//  JANGTQ (TurboQuant codebook) variant of the Qwen 3.5 / 3.6 MoE
//  vision-language model (model_type: `qwen3_5_moe`, weight_format: `mxtq`).
//
//  Wraps the same vision_tower as `Qwen35MoE` but routes the routed-expert
//  MoE projections through `TurboQuantSwitchGLU` (codebook Metal kernels)
//  instead of `SwitchGLU` (`gather_qmm`). All other components — attention,
//  GatedDeltaNet, shared expert, RoPE, vision tower — are identical to the
//  affine VLM path and reuse the existing `Qwen35Language` types in this
//  module.
//
//  Sanitize jobs (in addition to the affine VLM sanitize):
//    * Stack `experts.{0..N-1}.{w1,w2,w3}.{tq_packed,tq_norms}` into the 3D
//      `switch_mlp.{gate,up,down}_proj.{tq_packed,tq_norms}` layout that
//      `TurboQuantSwitchGLU` expects (when the converter wrote per-expert
//      tensors). When the converter already pre-stacked into
//      `switch_mlp.*` (Qwen 3.5/3.6 path), no stacking is needed — the
//      detection is via the presence of `experts.0.w1.tq_packed`.
//    * Drop `.tq_bits` metadata tensors (per-tensor bit-width hints — not
//      module parameters; the JANGTQ kernel reads bits from the loader-
//      injected `mxtqBits` config field instead).
//    * Preserve `vision_tower.*` keys (the LLM-side JANGTQ model strips
//      them; the VLM path needs them).
//

import Foundation
import MLX
import MLXNN
import vMLXLMCommon

// MARK: - Compiled helpers (file-local copies to avoid cross-file privates)

/// Compiled sigmoid gate for the shared-expert path. File-local so this
/// file builds without depending on `Qwen35.swift`'s private constants.
private let qwen35MoEJANGTQCompiledSigmoidGate: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { gate, expert in
        sigmoid(gate) * expert
    }
    return HardwareInfo.isCompiledDecodeSupported ? compile(shapeless: true, body) : body
}()

/// Compiled router fast path mirroring `qwen35JANGTQCompiledRouter` from the
/// LLM-side JANGTQ file. Caches per `(numExperts, k, renorm)` because the
/// `kth = numExperts - k` argpartition index must be a compile-time
/// constant.
private struct Qwen35MoEJANGTQRouterKey: Hashable {
    let numExperts: Int; let k: Int; let renorm: Bool
}
private nonisolated(unsafe) var _qwen35MoEJANGTQRouterCache:
    [Qwen35MoEJANGTQRouterKey: ([MLXArray]) -> [MLXArray]] = [:]
private let _qwen35MoEJANGTQRouterLock = NSLock()

private func qwen35MoEJANGTQCompiledRouter(numExperts: Int, k: Int, renorm: Bool)
    -> ([MLXArray]) -> [MLXArray]
{
    let key = Qwen35MoEJANGTQRouterKey(numExperts: numExperts, k: k, renorm: renorm)
    _qwen35MoEJANGTQRouterLock.lock(); defer { _qwen35MoEJANGTQRouterLock.unlock() }
    if let cached = _qwen35MoEJANGTQRouterCache[key] { return cached }
    let kth = numExperts - k
    let body: ([MLXArray]) -> [MLXArray] = { args in
        let gates = args[0]
        let scores = MLX.softmax(gates, axis: -1, precise: true)
        let inds = MLX.argPartition(scores, kth: kth, axis: -1)[.ellipsis, kth...]
        var sel = MLX.takeAlong(scores, inds, axis: -1)
        if renorm {
            sel = sel
                / (sel.sum(axis: -1, keepDims: true)
                    + MLXArray(Float(1e-20), dtype: sel.dtype))
        }
        return [inds, sel]
    }
    let compiled = compile(body)
    _qwen35MoEJANGTQRouterCache[key] = compiled
    return compiled
}

// MARK: - Configuration

/// Wraps `Qwen35Configuration` with the JANGTQ-specific `mxtq_bits` /
/// `mxtq_seed` fields the converter writes into config.json. Both have
/// safe defaults so a forgotten field doesn't crash the loader.
public struct Qwen35MoEJANGTQConfiguration: Codable, Sendable {
    public let base: Qwen35Configuration
    public let mxtqBits: Int
    public let mxtqSeed: Int

    enum CodingKeys: String, CodingKey {
        case mxtqBits = "mxtq_bits"
        case mxtqSeed = "mxtq_seed"
    }

    public init(from decoder: Decoder) throws {
        self.base = try Qwen35Configuration(from: decoder)
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.mxtqBits = try container.decodeIfPresent(Int.self, forKey: .mxtqBits) ?? 2
        self.mxtqSeed = try container.decodeIfPresent(Int.self, forKey: .mxtqSeed) ?? 42
    }

    public func encode(to encoder: Encoder) throws {
        try base.encode(to: encoder)
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(mxtqBits, forKey: .mxtqBits)
        try container.encode(mxtqSeed, forKey: .mxtqSeed)
    }
}

// MARK: - JANGTQ Language stack

/// Parallel namespace to `Qwen35Language` (in `Qwen35.swift`) but the
/// SparseMoeBlock routes through `TurboQuantSwitchGLU`. All other classes
/// (Attention, GatedDeltaNet, MLP, RMSNormGated, etc.) are reused.
enum Qwen35JANGTQLanguage {

    final class SparseMoeBlock: Module, UnaryLayer {
        let normTopkProb: Bool
        let numExperts: Int
        let topK: Int

        @ModuleInfo(key: "gate") var gate: Linear
        @ModuleInfo(key: "switch_mlp") var switchMLP: TurboQuantSwitchGLU

        @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen35Language.MLP
        @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

        init(_ args: Qwen35Configuration.TextConfiguration, mxtqBits: Int, mxtqSeed: Int) {
            self.normTopkProb = args.normTopkProb
            self.numExperts = args.numExperts
            self.topK = args.numExpertsPerTok

            _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
            _switchMLP.wrappedValue = TurboQuantSwitchGLU(
                inputDims: args.hiddenSize,
                hiddenDims: args.moeIntermediateSize,
                numExperts: args.numExperts,
                bits: mxtqBits, seed: mxtqSeed
            )

            _sharedExpert.wrappedValue = Qwen35Language.MLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.sharedExpertIntermediateSize
            )
            _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let gates = gate(x)
            // Compiled softmax → topk → gather → renormalize chain.
            let routed = qwen35MoEJANGTQCompiledRouter(
                numExperts: numExperts, k: topK, renorm: normTopkProb
            )([gates])
            let inds = routed[0]
            let scores = routed[1]

            let y = switchMLP(x, inds)
            let combined = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)

            let sharedY = sharedExpert(x)
            let gatedSharedY = qwen35MoEJANGTQCompiledSigmoidGate(sharedExpertGate(x), sharedY)

            return combined + gatedSharedY
        }
    }

    final class DecoderLayer: Module {
        let isLinear: Bool

        @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Language.Attention?
        @ModuleInfo(key: "linear_attn") var linearAttn: Qwen35Language.GatedDeltaNet?

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        @ModuleInfo(key: "mlp") var mlp: Module

        init(
            _ args: Qwen35Configuration.TextConfiguration, layerIdx: Int,
            mxtqBits: Int, mxtqSeed: Int
        ) {
            self.isLinear = (layerIdx + 1) % args.fullAttentionInterval != 0

            if isLinear {
                _linearAttn.wrappedValue = Qwen35Language.GatedDeltaNet(args)
            } else {
                _selfAttn.wrappedValue = Qwen35Language.Attention(args)
            }

            if args.numExperts > 0 {
                _mlp.wrappedValue = SparseMoeBlock(args, mxtqBits: mxtqBits, mxtqSeed: mxtqSeed)
            } else {
                _mlp.wrappedValue = Qwen35Language.MLP(
                    dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            }

            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            attentionMask: MLXArray?,
            ssmMask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            let r: MLXArray
            if isLinear {
                r = linearAttn!(inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache)
            } else {
                r = selfAttn!(
                    inputLayerNorm(x), mask: attentionMask, cache: cache, positionIds: positionIds)
            }
            let h = x + r
            return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
        }
    }

    final class Model: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") fileprivate var layers: [DecoderLayer]
        @ModuleInfo(key: "norm") var norm: RMSNorm

        let ssmIdx: Int
        let faIdx: Int

        init(_ args: Qwen35Configuration.TextConfiguration, mxtqBits: Int, mxtqSeed: Int) {
            precondition(args.vocabularySize > 0)
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
            _layers.wrappedValue = (0 ..< args.hiddenLayers).map {
                DecoderLayer(args, layerIdx: $0, mxtqBits: mxtqBits, mxtqSeed: mxtqSeed)
            }
            _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

            self.ssmIdx = 0
            self.faIdx = args.fullAttentionInterval - 1
            super.init()
        }

        func callAsFunction(
            _ inputs: MLXArray,
            inputsEmbeds: MLXArray? = nil,
            cache: [KVCache?]? = nil,
            positionIds: MLXArray? = nil
        ) -> MLXArray {
            var hiddenStates: MLXArray
            if let inputsEmbeds {
                hiddenStates = inputsEmbeds
            } else {
                hiddenStates = embedTokens(inputs)
            }

            var cacheArray = cache
            if cacheArray == nil {
                cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
            }

            let faMaskMode = createAttentionMask(
                h: hiddenStates, cache: cacheArray?[faIdx], returnArray: true)
            let faMask: MLXArray?
            if case .array(let arrayMask) = faMaskMode {
                faMask = arrayMask
            } else {
                faMask = nil
            }
            let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? MambaCache)

            for (index, layer) in layers.enumerated() {
                let layerSSMMask = layer.isLinear ? ssmMask : nil
                hiddenStates = layer(
                    hiddenStates,
                    attentionMask: faMask,
                    ssmMask: layerSSMMask,
                    cache: cacheArray?[index],
                    positionIds: positionIds
                )
            }

            return norm(hiddenStates)
        }
    }

    final class LanguageModel: Module {
        @ModuleInfo var model: Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        let config: Qwen35Configuration
        let textConfig: Qwen35Configuration.TextConfiguration
        let modelType: String
        let kvHeads: [Int]

        fileprivate var precomputedPositionIds: MLXArray? = nil
        fileprivate var ropeDeltas: MLXArray? = nil

        init(_ config: Qwen35Configuration, mxtqBits: Int, mxtqSeed: Int) {
            self.config = config
            self.textConfig = config.textConfiguration
            self.modelType = config.textConfiguration.modelType
            self.model = Model(
                config.textConfiguration, mxtqBits: mxtqBits, mxtqSeed: mxtqSeed)
            self.kvHeads = Array(
                repeating: config.textConfiguration.kvHeads,
                count: config.textConfiguration.hiddenLayers
            )

            if !config.textConfiguration.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.textConfiguration.hiddenSize,
                    config.textConfiguration.vocabularySize,
                    bias: false)
            }
            super.init()
        }

        func resetPositionState() {
            precomputedPositionIds = nil
            ropeDeltas = nil
        }

        func callAsFunction(
            _ inputs: MLXArray,
            inputsEmbeds: MLXArray? = nil,
            cache: [KVCache?]? = nil,
            mask: MLXArray? = nil,
            positionIds providedPositionIds: MLXArray? = nil,
            pixelValues: MLXArray? = nil,
            imageGridTHW: [THW]? = nil,
            videoGridTHW: [THW]? = nil
        ) -> LMOutput {
            if pixelValues != nil {
                precomputedPositionIds = nil
                ropeDeltas = nil
            }

            var cacheOffset = 0
            if let cache, let faCache = cache[model.faIdx] {
                cacheOffset = faCache.offset
            }

            var ropeMask = mask
            if let mask, mask.dim(-1) != inputs.dim(-1) {
                ropeMask = nil
            }

            var positionIds = providedPositionIds
            if positionIds == nil && (ropeMask == nil || ropeMask?.ndim == 2) {
                if (cache != nil && cache?[model.faIdx] != nil && cacheOffset == 0)
                    || ropeDeltas == nil
                    || cache == nil
                {
                    if let precomputedPositionIds {
                        let seqLength = inputs.dim(1)
                        positionIds =
                            precomputedPositionIds[
                                0..., 0..., cacheOffset ..< (cacheOffset + seqLength)]
                    } else {
                        let (computed, deltas) = Qwen3VLLanguage.getRopeIndex(
                            inputIds: inputs,
                            imageGridTHW: imageGridTHW,
                            videoGridTHW: videoGridTHW,
                            spatialMergeSize: config.visionConfiguration.spatialMergeSize,
                            imageTokenId: config.imageTokenId,
                            videoTokenId: config.videoTokenId,
                            visionStartTokenId: config.visionStartTokenId,
                            attentionMask: ropeMask)
                        positionIds = computed
                        precomputedPositionIds = computed
                        ropeDeltas = deltas
                    }
                } else {
                    let batchSize = inputs.dim(0)
                    let seqLength = inputs.dim(1)

                    var delta = MLXArray(cacheOffset).asType(.int32)
                    if let ropeDeltas {
                        delta = delta + ropeDeltas.asType(.int32)
                    }

                    var base = MLXArray(0 ..< seqLength).asType(.int32)
                    base = broadcast(base[.newAxis, 0...], to: [batchSize, seqLength])

                    if delta.ndim == 0 {
                        delta = broadcast(delta, to: [batchSize])
                    } else if delta.dim(0) < batchSize {
                        delta = repeated(delta, count: batchSize, axis: 0)
                    } else if delta.dim(0) > batchSize {
                        delta = delta[0 ..< batchSize]
                    }

                    base = base + delta[0..., .newAxis]
                    positionIds = broadcast(
                        base[.newAxis, 0..., 0...], to: [3, batchSize, seqLength])
                }
            }

            var out = model(
                inputs,
                inputsEmbeds: inputsEmbeds,
                cache: cache,
                positionIds: positionIds
            )

            if let lmHead {
                out = lmHead(out)
            } else {
                out = model.embedTokens.asLinear(out)
            }

            return LMOutput(logits: out)
        }

        func makeCache(maxKVSize: Int?) -> [KVCache] {
            model.layers.map { layer in
                if layer.isLinear {
                    return MambaCache()
                }
                if let maxKVSize {
                    return RotatingKVCache(maxSize: maxKVSize, keep: 4)
                }
                return KVCacheSimple()
            }
        }
    }
}

// MARK: - VLM model

/// Top-level Qwen 3.5 / 3.6 MoE VLM with JANGTQ routed-expert kernels.
public class Qwen35MoEJANGTQ: Module, VLMModel {
    @ModuleInfo(key: "vision_tower") private var visionModel: Qwen3VLVision.VisionModel
    @ModuleInfo(key: "language_model") fileprivate var languageModel:
        Qwen35JANGTQLanguage.LanguageModel

    public let config: Qwen35Configuration
    public let mxtqBits: Int
    public let mxtqSeed: Int

    public init(_ wrapped: Qwen35MoEJANGTQConfiguration) {
        self.config = wrapped.base
        self.mxtqBits = wrapped.mxtqBits
        self.mxtqSeed = wrapped.mxtqSeed
        _visionModel.wrappedValue = Qwen3VLVision.VisionModel(wrapped.base.visionConfiguration)
        _languageModel.wrappedValue = Qwen35JANGTQLanguage.LanguageModel(
            wrapped.base, mxtqBits: wrapped.mxtqBits, mxtqSeed: wrapped.mxtqSeed)
        super.init()
    }

    public var vocabularySize: Int { config.vocabSize }

    public var loraLayers: [Module] { languageModel.model.layers }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.makeCache(maxKVSize: parameters?.maxKVSize)
    }

    private enum Qwen35MoEJANGTQError: Error {
        case featureTokenMismatch(expected: Int, actual: Int)
    }

    private func mergeInputIdsWithImageFeatures(
        imageFeatures: MLXArray,
        inputEmbeds: MLXArray,
        inputIds: MLXArray,
        imageTokenIndex: Int,
        videoTokenIndex: Int
    ) throws -> (MLXArray, MLXArray) {
        let imageMask = (inputIds .== MLXArray(imageTokenIndex))
        let videoMask = (inputIds .== MLXArray(videoTokenIndex))
        var specialMask = imageMask .|| videoMask

        let nImageTokens = specialMask.sum().item(Int.self)

        specialMask = expandedDimensions(specialMask, axis: -1)
        let maskExpanded = broadcast(specialMask, to: inputEmbeds.shape)

        let nImageFeatures = imageFeatures.dim(0)
        let nImageMaskElements = maskExpanded.sum().item(Int.self)
        let imageFeatureSize = imageFeatures.size

        guard nImageMaskElements == imageFeatureSize else {
            throw Qwen35MoEJANGTQError.featureTokenMismatch(
                expected: nImageTokens, actual: nImageFeatures)
        }

        let originalShape = inputEmbeds.shape
        let flattenedEmbeds = inputEmbeds.flattened()
        let flattenedFeatures = imageFeatures.flattened()
        let flattenedMask = maskExpanded.flattened()

        let indices = nonZero(flattenedMask.asType(.bool))

        var result = flattenedEmbeds
        if !indices.isEmpty && indices.count == flattenedFeatures.size {
            let indexArray = MLXArray(indices.map { UInt32($0) })
            result[indexArray] = flattenedFeatures
        }

        result = result.reshaped(originalShape)
        let visualMask = specialMask.squeezed(axis: -1).asType(.bool)
        return (result, visualMask)
    }

    private func nonZero(_ mask: MLXArray) -> [Int] {
        let values = mask.asArray(Bool.self)
        var indices: [Int] = []
        indices.reserveCapacity(values.count)
        for (idx, value) in values.enumerated() where value {
            indices.append(idx)
        }
        return indices
    }

    private func combinedFrames(imageFrames: [THW]?, videoFrames: [THW]?) -> [THW] {
        var frames: [THW] = []
        if let imageFrames { frames.append(contentsOf: imageFrames) }
        if let videoFrames { frames.append(contentsOf: videoFrames) }
        return frames
    }

    private func castCacheOptional(_ cache: [any KVCache]?) -> [KVCache?]? {
        guard let cache else { return nil }
        return cache.map { $0 as KVCache? }
    }

    private func castCache(_ cache: [any KVCache]) -> [KVCache?] {
        cache.map { $0 as KVCache? }
    }

    public func prepare(
        _ input: LMInput,
        cache: [any KVCache],
        windowSize _: Int?
    ) throws -> PrepareResult {
        let inputIds = input.text.tokens

        var pixelValues: MLXArray?
        var imageFrames: [THW]?
        var videoFrames: [THW]?

        let visionDType = visionModel.patchEmbed.proj.weight.dtype
        var pixelParts: [MLXArray] = []

        if let image = input.image {
            pixelParts.append(image.pixels.asType(visionDType))
            imageFrames = image.frames
        }
        if let video = input.video {
            pixelParts.append(video.pixels.asType(visionDType))
            videoFrames = video.frames
        }
        if !pixelParts.isEmpty {
            pixelValues = concatenated(pixelParts)
        }

        var inputEmbeddings: MLXArray?

        if let pixelValues,
            let frames = combinedFrames(imageFrames: imageFrames, videoFrames: videoFrames)
                .nilIfEmptyJANGTQ
        {
            let textEmbeds = languageModel.model.embedTokens(inputIds)
            let (visionHidden, _) = visionModel(pixelValues, gridTHW: frames)
            let visionFeatures = visionHidden.asType(textEmbeds.dtype)

            let (mergedEmbeds, _) = try mergeInputIdsWithImageFeatures(
                imageFeatures: visionFeatures,
                inputEmbeds: textEmbeds,
                inputIds: inputIds,
                imageTokenIndex: config.imageTokenIndex,
                videoTokenIndex: config.videoTokenIndex
            )
            inputEmbeddings = mergedEmbeds
        } else {
            languageModel.resetPositionState()
        }

        let typedCache = castCache(cache)
        let output = languageModel(
            inputIds,
            inputsEmbeds: inputEmbeddings,
            cache: typedCache,
            mask: input.text.mask,
            positionIds: nil,
            pixelValues: pixelValues,
            imageGridTHW: imageFrames,
            videoGridTHW: videoFrames
        )

        return .logits(output)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let typedCache = castCacheOptional(cache)
        let result = languageModel(
            inputs,
            inputsEmbeds: nil,
            cache: typedCache,
            mask: nil,
            positionIds: nil,
            pixelValues: nil,
            imageGridTHW: nil,
            videoGridTHW: nil
        )
        return result.logits
    }

    // MARK: - Sanitize

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String:
        MLXArray]
    {
        let isMLXFormat = metadata["format"]?.lowercased() == "mlx"
        return sanitize(weights: weights, isMLXFormat: isMLXFormat)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        sanitize(weights: weights, isMLXFormat: false)
    }

    private func sanitize(weights: [String: MLXArray], isMLXFormat: Bool) -> [String: MLXArray] {
        // ───── Step 1: shared affine VLM sanitize (mtp drop, key renames,
        // conv1d moveaxis, norm shift). Mirrors `Qwen35.sanitize`.
        let hasMTPWeights = weights.keys.contains { $0.contains("mtp.") }
        let hasUnsanitizedConv1d = weights.contains { key, value in
            key.contains("conv1d.weight") && value.dim(-1) != 1
        }
        let shouldShiftNormWeights = hasMTPWeights || hasUnsanitizedConv1d || !isMLXFormat

        var weights = weights.filter { !$0.key.contains("mtp.") }

        if config.textConfiguration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        // Drop tq_bits metadata tensors — not module parameters.
        for key in Array(weights.keys) where key.hasSuffix(".tq_bits") {
            weights[key] = nil
        }

        let normKeys = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ]

        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        for (key, originalValue) in weights {
            var key = key
            var value = originalValue

            if key.contains("model") {
                if key.contains("model.language_model") {
                    key = key.replacingOccurrences(
                        of: "model.language_model", with: "language_model.model")
                } else if key.contains("model.visual") {
                    key = key.replacingOccurrences(of: "model.visual", with: "vision_tower")
                }
            } else if key.contains("lm_head") {
                key = key.replacingOccurrences(of: "lm_head", with: "language_model.lm_head")
            }

            if key.contains("conv1d.weight") && value.dim(-1) != 1 {
                value = value.movedAxis(source: 2, destination: 1)
            }
            if shouldShiftNormWeights && normKeys.contains(where: { key.hasSuffix($0) })
                && value.ndim == 1
            {
                value = value + MLXArray(1, dtype: value.dtype)
            }

            sanitized[key] = value
        }

        // ───── Step 2: stack per-expert TQ tensors if present
        // (Mixtral-style w1/w2/w3 layout). For Qwen 3.5/3.6 the converter
        // writes them already pre-stacked as `switch_mlp.*.tq_*` so this
        // step is a no-op on those artifacts (probe miss → skip).
        let renames: [(String, String)] = [
            ("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj"),
        ]
        let probe = "language_model.model.layers.0.mlp.experts.0.w1.tq_packed"
        if sanitized[probe] != nil {
            for layer in 0 ..< config.textConfiguration.hiddenLayers {
                let prefix = "language_model.model.layers.\(layer).mlp"
                for (orig, updated) in renames {
                    for kind in ["tq_packed", "tq_norms"] {
                        let first = "\(prefix).experts.0.\(orig).\(kind)"
                        guard sanitized[first] != nil else { continue }
                        let stacked: [MLXArray] = (0 ..< config.textConfiguration.numExperts).map {
                            e in
                            sanitized.removeValue(
                                forKey: "\(prefix).experts.\(e).\(orig).\(kind)")!
                        }
                        sanitized["\(prefix).switch_mlp.\(updated).\(kind)"] = MLX.stacked(stacked)
                    }
                }
            }
        }

        // ───── Step 3: vision tower sanitize (preserves vision_tower.* keys
        // — does NOT strip them like the LLM-side JANGTQ class does).
        return visionModel.sanitize(weights: sanitized)
    }
}

extension Array where Element == THW {
    fileprivate var nilIfEmptyJANGTQ: [THW]? { isEmpty ? nil : self }
}
