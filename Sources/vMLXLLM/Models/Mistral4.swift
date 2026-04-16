// Copyright © 2024-2026 Jinho Jang (eric@jangq.ai)
//
// Mistral Small 4 (119B MoE with MLA)
// Multi-head Latent Attention + Mixture of Experts with shared experts
//
// Python reference: mlx_lm/models/mistral4.py (by Jinho Jang)

import Foundation
import MLX
import vMLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Mistral4Configuration: Codable, Sendable {
    let modelType: String
    let vocabSize: Int
    let hiddenSize: Int
    let intermediateSize: Int
    let moeIntermediateSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let nSharedExperts: Int
    let nRoutedExperts: Int
    let numExpertsPerTok: Int
    let routedScalingFactor: Float
    let kvLoraRank: Int
    let qLoraRank: Int
    let qkRopeHeadDim: Int
    let vHeadDim: Int
    let qkNopeHeadDim: Int
    let maxPositionEmbeddings: Int
    let rmsNormEps: Float
    let ropeTheta: Float
    let ropeScaling: [String: StringOrNumber]?
    let ropeParameters: [String: StringOrNumber]?
    let attentionBias: Bool
    let normTopkProb: Bool
    let tieWordEmbeddings: Bool
    let headDim: Int
    let ropeInterleave: Bool
    let firstKDenseReplace: Int
    let moeLayerFreq: Int

    var qkHeadDim: Int { qkNopeHeadDim + qkRopeHeadDim }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case ropeParameters = "rope_parameters"
        case attentionBias = "attention_bias"
        case normTopkProb = "norm_topk_prob"
        case tieWordEmbeddings = "tie_word_embeddings"
        case headDim = "head_dim"
        case ropeInterleave = "rope_interleave"
        case firstKDenseReplace = "first_k_dense_replace"
        case moeLayerFreq = "moe_layer_freq"
    }

    // Support VLM wrapper: text_config nesting
    enum VLMKeys: String, CodingKey { case textConfig = "text_config" }

    public init(from decoder: Decoder) throws {
        let nc = try decoder.container(keyedBy: VLMKeys.self)
        let c = if nc.contains(.textConfig) {
            try nc.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
        } else {
            try decoder.container(keyedBy: CodingKeys.self)
        }

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "mistral4"
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 131072
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 12288
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 2048
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 36
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 32
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 32
        nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts) ?? 1
        nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts) ?? 128
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 4
        routedScalingFactor = try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
        kvLoraRank = try c.decodeIfPresent(Int.self, forKey: .kvLoraRank) ?? 256
        qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank) ?? 1024
        qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 64
        vHeadDim = try c.decodeIfPresent(Int.self, forKey: .vHeadDim) ?? 128
        qkNopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim) ?? 64
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 1048576
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        ropeInterleave = try c.decodeIfPresent(Bool.self, forKey: .ropeInterleave) ?? false
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        moeLayerFreq = try c.decodeIfPresent(Int.self, forKey: .moeLayerFreq) ?? 1

        // RoPE: prefer rope_parameters, fall back to rope_scaling, then direct theta
        ropeParameters = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
        let rawScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)

        if let rp = ropeParameters {
            // Merge rope_parameters into rope_scaling format
            var merged = rawScaling ?? [:]
            if merged["type"] == nil { merged["type"] = rp["type"] ?? rp["rope_type"] ?? .string("yarn") }
            if merged["factor"] == nil { merged["factor"] = rp["factor"] ?? .float(128.0) }
            if merged["original_max_position_embeddings"] == nil { merged["original_max_position_embeddings"] = rp["original_max_position_embeddings"] ?? .float(8192) }
            if merged["beta_fast"] == nil { merged["beta_fast"] = rp["beta_fast"] ?? .float(32.0) }
            if merged["beta_slow"] == nil { merged["beta_slow"] = rp["beta_slow"] ?? .float(1.0) }
            if merged["mscale"] == nil { merged["mscale"] = rp["mscale"] ?? .float(1.0) }
            if merged["mscale_all_dim"] == nil { merged["mscale_all_dim"] = rp["mscale_all_dim"] ?? .float(1.0) }
            if merged["llama_4_scaling_beta"] == nil { merged["llama_4_scaling_beta"] = rp["llama_4_scaling_beta"] ?? .float(0.0) }
            ropeScaling = merged
            let fallbackTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
            ropeTheta = rp["rope_theta"]?.asFloat() ?? fallbackTheta
        } else {
            ropeScaling = rawScaling
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        }
    }
}

// MARK: - MLA Attention

class Mistral4Attention: Module {
    let numHeads: Int
    let qLoraRank: Int
    let qkRopeHeadDim: Int
    let kvLoraRank: Int
    let vHeadDim: Int
    let qkNopeHeadDim: Int
    let qHeadDim: Int
    let scale: Float
    let llama4Beta: Float
    let llama4MaxPos: Int

    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: RoPELayer

    init(_ config: Mistral4Configuration) {
        self.numHeads = config.numAttentionHeads
        self.qLoraRank = config.qLoraRank
        self.qkRopeHeadDim = config.qkRopeHeadDim
        self.kvLoraRank = config.kvLoraRank
        self.vHeadDim = config.vHeadDim
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim
        self.scale = pow(Float(qHeadDim), -0.5)

        let ropeCfg = config.ropeScaling ?? [:]
        self.llama4Beta = ropeCfg["llama_4_scaling_beta"]?.asFloat() ?? 0.0
        self.llama4MaxPos = Int(ropeCfg["original_max_position_embeddings"]?.asFloat() ?? 8192)

        // Q path (low-rank compressed)
        self._qAProj.wrappedValue = Linear(config.hiddenSize, qLoraRank, bias: config.attentionBias)
        self._qALayerNorm.wrappedValue = RMSNorm(dimensions: qLoraRank, eps: config.rmsNormEps)
        self._qBProj.wrappedValue = Linear(qLoraRank, numHeads * qHeadDim, bias: false)

        // KV path
        self._kvAProjWithMqa.wrappedValue = Linear(
            config.hiddenSize, kvLoraRank + qkRopeHeadDim, bias: config.attentionBias)
        self._kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank, eps: config.rmsNormEps)
        self._kvBProj.wrappedValue = Linear(
            kvLoraRank, numHeads * (qkNopeHeadDim + vHeadDim), bias: false)

        self._oProj.wrappedValue = Linear(numHeads * vHeadDim, config.hiddenSize, bias: config.attentionBias)

        // Yarn RoPE — rope_interleave=true in config maps to traditional=true in MLX
        self.rope = initializeRope(
            dims: qkRopeHeadDim, base: config.ropeTheta, traditional: config.ropeInterleave,
            scalingConfig: config.ropeScaling, maxPositionEmbeddings: config.maxPositionEmbeddings)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        // Q path: x → q_a_proj → layernorm → q_b_proj
        var q = qBProj(qALayerNorm(qAProj(x)))
        q = q.reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let qSplit = split(q, indices: [qkNopeHeadDim], axis: -1)
        let qNope = qSplit[0]
        var qPe = qSplit[1]

        // KV path: x → kv_a_proj → split(compressed_kv, k_pe) → layernorm → kv_b_proj
        var compressedKV = kvAProjWithMqa(x)
        let kvSplit = split(compressedKV, indices: [kvLoraRank], axis: -1)
        compressedKV = kvSplit[0]
        var kPe = kvSplit[1].reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)

        var kv = kvBProj(kvALayerNorm(compressedKV))
        kv = kv.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        let kvDecompSplit = split(kv, indices: [qkNopeHeadDim], axis: -1)
        let kNope = kvDecompSplit[0]
        var values = kvDecompSplit[1]

        // RoPE (interleaved)
        qPe = applyRotaryPosition(rope, to: qPe, cache: cache)
        kPe = applyRotaryPosition(rope, to: kPe, cache: cache)
        kPe = repeated(kPe, count: numHeads, axis: 1)

        // Assemble full keys and queries
        var keys = concatenated([kNope, kPe], axis: -1)
        var queries = concatenated([qNope, qPe], axis: -1)

        // Cache update
        if let cache {
            let (ck, cv) = cache.update(keys: keys, values: values)
            keys = ck
            values = cv
        }

        // Llama 4 position-dependent query scaling
        if llama4Beta > 0 {
            let offset = cache?.offset ?? 0
            let l4Scale = 1.0 + llama4Beta * log(1.0 + floor(Float(offset) / Float(llama4MaxPos)))
            if l4Scale != 1.0 {
                queries = queries * l4Scale
            }
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

// MARK: - MLP

class Mistral4MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - MoE Gate

class Mistral4MoEGate: Module {
    let topK: Int
    let nRoutedExperts: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool

    @ModuleInfo var weight: MLXArray

    init(_ config: Mistral4Configuration) {
        self.topK = config.numExpertsPerTok
        self.nRoutedExperts = config.nRoutedExperts
        self.routedScalingFactor = config.routedScalingFactor
        self.normTopkProb = config.normTopkProb
        self._weight.wrappedValue = MLXArray.zeros([config.nRoutedExperts, config.hiddenSize])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        let gates = matmul(x, weight.transposed())
        let scores = softmax(gates, axis: -1, precise: true)

        // Negate via unary `-` — scalar-subtract with an untyped zero
        // scalar pays an extra AsType + broadcast dispatch every routing
        // step. See SWIFT-NO-REGRESSION-CHECKLIST §27.
        let inds = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var wts = takeAlong(scores, inds, axis: -1)

        if normTopkProb {
            wts = wts / wts.sum(axis: -1, keepDims: true)
        }
        wts = wts * routedScalingFactor

        return (indices: inds, weights: wts)
    }

    // Gate stays in float — not quantized
    func toQuantized(groupSize: Int, bits: Int) -> Module { self }
}

// MARK: - MoE Block

class Mistral4MoE: Module, UnaryLayer {
    @ModuleInfo var gate: Mistral4MoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: Mistral4MLP?

    /// Flash MoE Phase 2b shim. When non-nil, the routed experts
    /// path is served from the slot-bank streamer instead of the
    /// in-memory `switchMLP`. `sharedExperts` still runs natively so
    /// the always-on shared MLP contribution is preserved.
    fileprivate var flashMoeShim: FlashMoEBlock? = nil

    init(_ config: Mistral4Configuration) {
        self.gate = Mistral4MoEGate(config)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts,
            bias: false)

        if config.nSharedExperts > 0 {
            let sharedInter = config.moeIntermediateSize * config.nSharedExperts
            self._sharedExperts.wrappedValue = Mistral4MLP(
                hiddenSize: config.hiddenSize, intermediateSize: sharedInter)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y: MLXArray
        if let shim = flashMoeShim {
            // FlashMoEBlock's forward takes just (x) and does its own
            // router call + per-expert matmul + weighted sum. Its
            // output is already [..., H] matching the native path.
            y = shim(x)
        } else {
            let (inds, scores) = gate(x)
            y = switchMLP(x, inds)
            y = (y * expandedDimensions(scores, axis: -1)).sum(axis: -2)
        }
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

// MARK: - Decoder Layer

class Mistral4DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Mistral4Attention
    @ModuleInfo var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    init(_ config: Mistral4Configuration, layerIndex: Int) {
        self._selfAttn.wrappedValue = Mistral4Attention(config)

        let isMoE = config.nRoutedExperts > 0
            && layerIndex >= config.firstKDenseReplace
            && layerIndex % config.moeLayerFreq == 0

        if isMoE {
            self._mlp.wrappedValue = Mistral4MoE(config)
        } else {
            self._mlp.wrappedValue = Mistral4MLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }

        self._inputLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayernorm(x), mask: mask, cache: cache)
        let h = x + r
        return h + mlp(postAttentionLayernorm(h))
    }
}

// MARK: - Text Model

public class Mistral4TextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Mistral4DecoderLayer]
    @ModuleInfo var norm: RMSNorm

    let config: Mistral4Configuration

    init(_ config: Mistral4Configuration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map { i in
            Mistral4DecoderLayer(config, layerIndex: i)
        }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache?]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        let lc = cache ?? Array(repeating: nil as KVCache?, count: layers.count)
        let mask = makeAttentionMask(n: h.dim(1), cache: lc.first ?? nil)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: lc[i])
        }
        return norm(h)
    }

    /// Forward pass that also captures per-layer hidden states at each
    /// requested decoder index. Used by JANG-DFlash to feed target
    /// hiddens into the drafter's KV injection (see DFlash §4.1).
    ///
    /// Mistral 4 uses MLA attention — the internal latent K/V
    /// reconstructions are *not* a stable tap surface (shape and meaning
    /// drift between the compressed and decompressed representations),
    /// so the tap is the **post-decoder-block residual hidden state**
    /// (same `h` as standard forward), which matches the MiniMax
    /// semantics and the drafter's distillation contract.
    ///
    /// - Parameters:
    ///   - inputs: token ID tensor `[B, L]`
    ///   - cache:  optional per-layer KV caches (may be nil for cacheless)
    ///   - tapLayers: decoder indices whose post-layer hidden state to
    ///     capture
    ///   - providedMask: when non-nil, replaces the auto-built causal
    ///     mask (used for tree-attention verification during spec-dec)
    func callAsFunctionWithTaps(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        tapLayers: Set<Int>,
        providedMask: MLXFast.ScaledDotProductAttentionMaskMode? = nil
    ) -> (output: MLXArray, taps: [Int: MLXArray]) {
        var h = embedTokens(inputs)

        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if let providedMask {
            mask = providedMask
        } else {
            mask = makeAttentionMask(n: h.dim(1), cache: cache?.first)
        }

        var taps: [Int: MLXArray] = [:]
        taps.reserveCapacity(tapLayers.count)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
            if tapLayers.contains(i) {
                taps[i] = h
            }
        }

        return (norm(h), taps)
    }
}

// MARK: - Top-Level Model

public class Mistral4Model: Module, LLMModel {
    @ModuleInfo public var model: Mistral4TextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let config: Mistral4Configuration
    public var vocabularySize: Int { config.vocabSize }

    public init(_ config: Mistral4Configuration) {
        self.config = config
        self.model = Mistral4TextModel(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    /// Forward pass that exposes per-layer hidden-state taps alongside
    /// the logits. Wraps `Mistral4TextModel.callAsFunctionWithTaps` and
    /// routes the final hidden state through `lm_head` (or tied
    /// embedding) the same way as the standard forward.
    public func callAsFunctionWithTaps(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        tapLayers: Set<Int>,
        providedMask: MLXFast.ScaledDotProductAttentionMaskMode? = nil
    ) -> (logits: MLXArray, taps: [Int: MLXArray]) {
        let (out, taps) = model.callAsFunctionWithTaps(
            inputs, cache: cache, tapLayers: tapLayers, providedMask: providedMask)
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(out)
        } else {
            logits = model.embedTokens.asLinear(out)
        }
        return (logits, taps)
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String:
        MLXArray]
    {
        var w = [String: MLXArray]()
        for (key, value) in weights {
            var k = key

            // Strip VLM prefix
            if k.hasPrefix("language_model.") {
                k = String(k.dropFirst("language_model.".count))
            }

            // JANG: switch_mlp is already in the right format
            // Skip FP8 scale tensors (not used in MLX quantization)
            if k.contains("_scale_inv") || k.contains("_activation_scale") { continue }

            // Skip vision tower weights for text-only
            if k.hasPrefix("vision_tower.") || k.hasPrefix("multi_modal_projector.") { continue }

            w[k] = value
        }
        return w
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< config.numHiddenLayers).map { _ in
            if let maxKVSize = parameters?.maxKVSize {
                return RotatingKVCache(maxSize: maxKVSize, keep: 4)
            }
            return KVCacheSimple()
        }
    }
}

extension Mistral4Model: LoRAModel {
    public var loraLayers: [Module] { model.layers }
}

// MARK: - Flash MoE (Phase 2b)
//
// Mistral4 is text-path SwitchGLU like Qwen3MoE/MiniMax but with a
// dedicated `Mistral4MoEGate` module that already returns
// `(indices, weights)` directly — no need to reproduce the
// softmax + argPartition math. We just capture the gate as a
// closure and hand the tuple straight through. Shared experts
// are preserved by the `Mistral4MoE.callAsFunction` shim branch.

extension Mistral4Model: FlashMoEReplaceable {
    public var flashMoELayers: [FlashMoELayer] {
        model.layers
    }
}

extension Mistral4DecoderLayer: FlashMoELayer {
    public var flashMoELayout: FlashMoELayout {
        (mlp is Mistral4MoE) ? .textPathSwitchGLU : .none
    }

    public func replaceMoEBlock(with block: FlashMoEBlock) throws {
        guard let moe = mlp as? Mistral4MoE else { return }
        let gate = moe.gate
        block.topK = gate.topK
        block.router = { x in
            let out = gate(x)
            return (indices: out.indices, scores: out.weights)
        }
        moe.flashMoeShim = block
    }
}

// MARK: - JANG-DFlash target adapter
//
// Bridges `Mistral4Model.callAsFunctionWithTaps(...)` to the
// architecture-agnostic `JangDFlashTarget` protocol defined in
// vMLXLMCommon/DFlash/JangDFlashSpecDec.swift. Mirrors
// `MiniMaxDFlashTarget` — kept as an adapter (rather than a direct
// conformance) so vMLXLMCommon stays model-agnostic.
public final class Mistral4DFlashTarget: JangDFlashTarget {
    public let model: Mistral4Model

    public init(_ model: Mistral4Model) { self.model = model }

    public func forwardWithTaps(
        inputs: MLXArray,
        cache: [KVCache]?,
        tapLayers: Set<Int>,
        providedMask: MLXFast.ScaledDotProductAttentionMaskMode?
    ) -> (logits: MLXArray, taps: [Int: MLXArray]) {
        return model.callAsFunctionWithTaps(
            inputs,
            cache: cache,
            tapLayers: tapLayers,
            providedMask: providedMask
        )
    }

    public func makeCache() -> [KVCache] {
        // Same disambiguation trick used in `MiniMaxDFlashTarget` — pass
        // `parameters: nil` explicitly so the compiler picks the
        // non-deprecated overload of `makePromptCache`.
        return makePromptCache(model: model, parameters: nil)
    }
}
