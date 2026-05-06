//
// Mistral3TextJANGTQ — JANGTQ-quantized variant of Mistral3TextModel.
//
// Drop-in for `weight_format == "mxtq"` Mistral 3 / Mistral 3.5 / inner-
// `ministral3` bundles. Architecture is identical to Mistral3TextModel
// (sliding+full per-layer mixed attention, llama4 attention scaling,
// RoPE) — only difference is every dense `Linear` (attention Q/K/V/O,
// MLP gate/up/down) is replaced with `JANGTQDenseLinear` so the
// safetensors' `.tq_packed` + `.tq_norms` keys feed the codebook
// kernels instead of trying to bind a flat `.weight` tensor.
//
// `lm_head` and `embed_tokens` stay full-precision per the Python
// converter's `mxtq_bits.embed_lm_head: 8` profile (passthrough fp16
// /bf16). RMSNorm has no quantizable weights — it shares the standard
// MLXNN.RMSNorm class with the base model.
//
// Bits / seed / per-layer profile come from `mxtq_bits` in
// `jang_config.json`, merged into config.json by the factory before
// decode. Defaults (bits=2, seed=42) match the JANGTQ2 profile shipped
// for Mistral-Medium-3.5-128B-JANGTQ2 / similar bundles.
//

import Foundation
import MLX
import vMLXLMCommon
import MLXNN

// MARK: - JANGTQ Attention

class Mistral3JANGTQAttention: Module {
    let args: Mistral3TextConfiguration
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: JANGTQDenseLinear
    @ModuleInfo(key: "k_proj") var wk: JANGTQDenseLinear
    @ModuleInfo(key: "v_proj") var wv: JANGTQDenseLinear
    @ModuleInfo(key: "o_proj") var wo: JANGTQDenseLinear

    let rope: RoPELayer

    init(_ args: Mistral3TextConfiguration, bits: Int, seed: Int) {
        self.args = args

        let dim = args.hiddenSize
        self.nHeads = args.attentionHeads
        self.nKVHeads = args.kvHeads
        self.headDim = args.resolvedHeadDimensions
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
        _ x: MLXArray, attnScale: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)
        queries = queries * attnScale

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

// MARK: - JANGTQ MLP

class Mistral3JANGTQMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: JANGTQDenseLinear
    @ModuleInfo(key: "down_proj") var down: JANGTQDenseLinear
    @ModuleInfo(key: "up_proj") var up: JANGTQDenseLinear

    init(_ args: Mistral3TextConfiguration, bits: Int, seed: Int) {
        let dim = args.hiddenSize
        let hiddenDim = args.intermediateSize

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

class Mistral3TextJANGTQTransformerBlock: Module {
    let numAttentionHeads: Int
    let hiddenSize: Int
    let useSliding: Bool

    @ModuleInfo(key: "self_attn") var attention: Mistral3JANGTQAttention
    @ModuleInfo(key: "mlp") var mlp: Mistral3JANGTQMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: Mistral3TextConfiguration, bits: Int, seed: Int, useSliding: Bool = false) {
        self.numAttentionHeads = args.attentionHeads
        self.hiddenSize = args.hiddenSize
        self.useSliding = useSliding

        self._attention.wrappedValue = Mistral3JANGTQAttention(args, bits: bits, seed: seed)
        self._mlp.wrappedValue = Mistral3JANGTQMLP(args, bits: bits, seed: seed)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, attnScale: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let r = attention(inputLayerNorm(x), attnScale: attnScale, mask: mask, cache: cache)
        let h = x + r
        let mlpOut = mlp(postAttentionLayerNorm(h))
        return h + mlpOut
    }
}

// MARK: - JANGTQ Inner Model

public class Mistral3TextJANGTQModelInner: Module {
    let args: Mistral3TextConfiguration
    let vocabularySize: Int
    let numHiddenLayers: Int
    let layerTypes: [String]
    let slidingWindow: Int?

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [Mistral3TextJANGTQTransformerBlock]
    let norm: RMSNorm

    let faIdx: Int
    let swaIdx: Int?

    init(_ args: Mistral3TextConfiguration, bits: Int, seed: Int) {
        self.args = args
        self.vocabularySize = args.vocabularySize
        self.numHiddenLayers = args.hiddenLayers
        self.layerTypes = args.layerTypes
        self.slidingWindow = args.slidingWindow

        precondition(args.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = args.layerTypes.map { layerType in
            Mistral3TextJANGTQTransformerBlock(
                args, bits: bits, seed: seed,
                useSliding: layerType == "sliding_attention")
        }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self.faIdx = args.layerTypes.firstIndex(of: "full_attention") ?? 0
        self.swaIdx = args.layerTypes.firstIndex(of: "sliding_attention")

        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]? = nil, inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        var h: MLXArray
        if let inputEmbeddings = inputEmbeddings {
            h = inputEmbeddings
        } else {
            h = embedTokens(inputs)
        }

        let faMask = createAttentionMask(h: h, cache: cache?[faIdx])
        let swaMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let swaIdx = swaIdx {
            swaMask = createAttentionMask(
                h: h, cache: cache?[swaIdx], windowSize: slidingWindow)
        } else {
            swaMask = .none
        }

        // llama4-style attention scaling, mirroring
        // Mistral3TextModelInner. Use graph-visible offsets after
        // compiled-cache promotion; reading `cache.offset` here would
        // call `.item()` inside `compile()`.
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

        for (i, layer) in layers.enumerated() {
            let mask = layer.useSliding ? swaMask : faMask
            h = layer(h, attnScale: attnScale, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

// MARK: - JANGTQ Top-Level Model

/// JANGTQ variant of `Mistral3TextModel`. Drop-in replacement when the
/// bundle's `weight_format == "mxtq"`. Same `LLMModel` conformance,
/// same `KVCacheDimensionProvider.kvHeads`, same `newCache(parameters:)`
/// per-layer mixed `RotatingKVCache` (sliding) + `KVCacheSimple` (full)
/// topology — only difference is the attention + MLP linears use the
/// JANGTQ codebook path internally.
public class Mistral3TextJANGTQModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Mistral3TextJANGTQModelInner
    fileprivate let args: Mistral3TextConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Mistral3TextConfiguration, bits: Int = 2, seed: Int = 42) {
        self.args = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Mistral3TextJANGTQModelInner(args, bits: bits, seed: seed)

        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if args.tieWordEmbeddings {
            out = model.embedTokens.asLinear(out)
        } else if let lmHead = lmHead {
            out = lmHead(out)
        }
        return out
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        // Mirrors Mistral3TextModel.newCache exactly — JANGTQ doesn't
        // change cache topology, only weight-loading semantics.
        return model.layers.map { layer in
            if layer.useSliding, let slidingWindow = args.slidingWindow {
                return RotatingKVCache(maxSize: slidingWindow)
            } else {
                return KVCacheSimple()
            }
        }
    }

    /// 2026-04-30 audit fix: mirror Mistral3TextModel.sanitize so JANGTQ
    /// bundles handle the same HF-shipped weight quirks the vanilla
    /// path handles. Without this, bundles carrying
    /// `self_attn.rotary_emb.inv_freq` (precomputed rope frequencies,
    /// common in HF safetensors) trigger "Unhandled keys" at load
    /// time. Also handles tied embeddings + the `tq_bits` per-tensor
    /// scalar that some early JANGTQ converters emit (vmlx ignores it
    /// — bits live in the model class config from jang_config.json).
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var processedWeights = weights

        // 2026-05-01 — Mistral-Medium-3.5 JANGTQ bundles ship the VLM
        // wrapper structure even when targeted by the LLM-only path:
        // top-level keys look like `model.language_model.*`,
        // `model.vision_tower.*`, `model.multi_modal_projector.*`.
        // The text-only Mistral3TextJANGTQModel expects `model.*`
        // (transformer body) + `lm_head.*`. Remap:
        //   `model.language_model.embed_tokens.*`     → `model.embed_tokens.*`
        //   `model.language_model.layers.*`           → `model.layers.*`
        //   `model.language_model.norm.*`             → `model.norm.*`
        //   `model.language_model.lm_head.*`          → `lm_head.*`
        // Drop:
        //   `model.vision_tower.*`
        //   `model.multi_modal_projector.*`
        // Also handle the legacy flat-prefix variant where the bundle
        // had `language_model.*` at the top level (older converter
        // output) — covered by the existing unflattening branch below.
        var remapped: [String: MLXArray] = [:]
        for (key, value) in processedWeights {
            if key.hasPrefix("model.vision_tower")
                || key.hasPrefix("model.multi_modal_projector")
                || key.hasPrefix("vision_tower")
                || key.hasPrefix("multi_modal_projector")
            {
                continue   // drop VLM wrapper keys
            }
            if key.hasPrefix("model.language_model.") {
                let inner = String(key.dropFirst("model.language_model.".count))
                if inner.hasPrefix("lm_head.") {
                    // language_model.lm_head.* → lm_head.* (top-level
                    // matches our @ModuleInfo("lm_head") on the wrapper).
                    remapped["lm_head." + String(inner.dropFirst("lm_head.".count))] = value
                } else {
                    remapped["model." + inner] = value
                }
            } else {
                remapped[key] = value
            }
        }
        processedWeights = remapped

        // Legacy flat top-level `language_model.*` (pre-Medium-3.5 dumps).
        let unflattened = ModuleParameters.unflattened(processedWeights)
        if let lm = unflattened["language_model"] {
            processedWeights = Dictionary(uniqueKeysWithValues: lm.flattened())
        }

        // Drop unused precomputed rope freqs and JANGTQ per-tensor
        // bit-width scalars (mxtq_bits is read from config, not weights).
        var sanitizedWeights = processedWeights.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
                && !$0.key.hasSuffix(".tq_bits")
        }

        // Tied embeddings: drop lm_head.weight; embed_tokens.asLinear
        // shares the input embedding matrix.
        if args.tieWordEmbeddings {
            sanitizedWeights["lm_head.weight"] = nil
        }

        // FP8 weight_scale_inv handling (matches vanilla Mistral3TextModel).
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
        return newWeights
    }
}

// LoRA conformance: JANGTQ Linear is not LoRA-quantizable today
// (codebook lookup + hadamard rotate doesn't compose with LoRA delta
// cleanly). Conform with empty `loraLayers` so LLMModel's `LoRAModel`
// requirement is satisfied; downstream LoRA-fine-tuners get an empty
// list and can detect the missing capability.
extension Mistral3TextJANGTQModel {
    public var loraLayers: [Module] { [] }
}
