// Copyright © 2024-2026 Jinho Jang (eric@jangq.ai)
//
// Gemma 4 text model — supports both:
//   - 26B MoE (128 experts, top-8, parallel MLP+MoE, GELU, softmax routing)
//   - 31B Dense (no MoE, standard MLP-only feedforward)
// Mixed sliding/full attention with per-layer head dims, K=V sharing, and RoPE config.
//
// Python reference: mlx_vlm/models/gemma4/language.py

import Foundation
import MLX
import vMLXLMCommon
import MLXNN

// Compiled logit softcap — fuses divide + tanh + multiply into one Metal dispatch.
// Matches Python: @partial(mx.compile, shapeless=True) def logit_softcap(softcap, x)
private let compiledLogitSoftcap: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { (x: MLXArray, cap: MLXArray) -> MLXArray in
        tanh(x / cap) * cap
    }
    return HardwareInfo.isCompiledDecodeSupported ? compile(shapeless: true, body) : body
}()

// MARK: - Norm Utilities

/// Standard RMSNorm for Gemma4 — weight used directly, NO +1 offset.
/// (Gemma3 uses 1.0 + weight; Gemma4 does NOT)
class Gemma4RMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

/// RMSNorm without learnable weight (RMSNormNoScale).
/// Used for v_norm and router's internal norm.
/// Python: mx.fast.rms_norm(x, None, eps) — MLXFast.rmsNorm doesn't support nil weight,
/// so we implement manually.
func rmsNormNoScale(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    let variance = (x * x).mean(axis: -1, keepDims: true)
    return x * rsqrt(variance + eps)
}

// MARK: - Configuration

public struct Gemma4TextConfiguration: Codable, Sendable {
    let modelType: String
    let hiddenSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let numGlobalKeyValueHeads: Int?
    let headDim: Int
    let globalHeadDim: Int
    let intermediateSize: Int
    let vocabSize: Int
    let rmsNormEps: Float
    let slidingWindow: Int
    let layerTypes: [String]
    let finalLogitSoftcapping: Float?
    let tieWordEmbeddings: Bool
    let attentionBias: Bool
    let attentionKEqV: Bool

    // Per-layer embedding fields (E2B/E4B models)
    let hiddenSizePerLayerInput: Int
    let vocabSizePerLayerInput: Int
    let numKvSharedLayers: Int
    let useDoubleWideMlp: Bool

    // MoE fields — only present when enableMoeBlock is true
    let enableMoeBlock: Bool
    let moeIntermediateSize: Int
    let numExperts: Int
    let topKExperts: Int

    // RoPE parameters per layer type
    let ropeTraditional: Bool
    let ropeParameters: [String: [String: StringOrNumber]]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case intermediateSize = "intermediate_size"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case attentionKEqV = "attention_k_eq_v"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case numKvSharedLayers = "num_kv_shared_layers"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case enableMoeBlock = "enable_moe_block"
        case moeIntermediateSize = "moe_intermediate_size"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case ropeTraditional = "rope_traditional"
        case ropeParameters = "rope_parameters"
    }

    enum VLMCodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let nestedContainer = try decoder.container(keyedBy: VLMCodingKeys.self)

        let container =
            if nestedContainer.contains(.textConfig) {
                try nestedContainer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            } else {
                try decoder.container(keyedBy: CodingKeys.self)
            }

        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2816
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 30
        numAttentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        numGlobalKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numGlobalKeyValueHeads)
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        globalHeadDim = try container.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2112
        vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 262144
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 1024
        layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        finalLogitSoftcapping =
            try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        attentionKEqV = try container.decodeIfPresent(Bool.self, forKey: .attentionKEqV) ?? false

        hiddenSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 0
        vocabSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput) ?? 0
        numKvSharedLayers =
            try container.decodeIfPresent(Int.self, forKey: .numKvSharedLayers) ?? 0
        useDoubleWideMlp =
            try container.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp) ?? false

        enableMoeBlock =
            try container.decodeIfPresent(Bool.self, forKey: .enableMoeBlock) ?? false
        moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        topKExperts = try container.decodeIfPresent(Int.self, forKey: .topKExperts) ?? 0
        ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        ropeParameters =
            try container.decodeIfPresent(
                [String: [String: StringOrNumber]].self, forKey: .ropeParameters) ?? [:]
    }
}

// MARK: - Attention

class Gemma4Attention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let isSliding: Bool
    let useKEqV: Bool
    let eps: Float

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear?
    @ModuleInfo(key: "o_proj") var outputProj: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: Gemma4RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: Gemma4RMSNorm
    // v_norm is RMSNormNoScale (no learnable weight, not in checkpoint)

    @ModuleInfo var rope: RoPELayer

    init(_ config: Gemma4TextConfiguration, layerIndex: Int) {
        let layerType =
            layerIndex < config.layerTypes.count
            ? config.layerTypes[layerIndex] : "sliding_attention"
        self.isSliding = layerType == "sliding_attention"
        self.eps = config.rmsNormEps

        // K=V sharing: full attention layers with attention_k_eq_v=true
        self.useKEqV = config.attentionKEqV && !isSliding

        if isSliding {
            self.nHeads = config.numAttentionHeads
            self.nKVHeads = config.numKeyValueHeads
            self.headDim = config.headDim
        } else {
            self.nHeads = config.numAttentionHeads
            self.nKVHeads = config.numGlobalKeyValueHeads ?? config.numKeyValueHeads
            self.headDim = config.globalHeadDim
        }

        // Gemma4 attention scale = 1.0 (NOT 1/sqrt(head_dim))
        self.scale = 1.0

        self._queryProj.wrappedValue = Linear(
            config.hiddenSize, nHeads * headDim, bias: config.attentionBias)
        self._keyProj.wrappedValue = Linear(
            config.hiddenSize, nKVHeads * headDim, bias: config.attentionBias)
        if !useKEqV {
            self._valueProj.wrappedValue = Linear(
                config.hiddenSize, nKVHeads * headDim, bias: config.attentionBias)
        }
        self._outputProj.wrappedValue = Linear(
            nHeads * headDim, config.hiddenSize, bias: config.attentionBias)

        self._queryNorm.wrappedValue = Gemma4RMSNorm(
            dimensions: headDim, eps: config.rmsNormEps)
        self._keyNorm.wrappedValue = Gemma4RMSNorm(
            dimensions: headDim, eps: config.rmsNormEps)

        // RoPE from config rope_parameters
        let layerKey = isSliding ? "sliding_attention" : "full_attention"
        let ropeParams = config.ropeParameters[layerKey] ?? [:]
        let ropeTheta = ropeParams["rope_theta"]?.asFloat() ?? (isSliding ? 10000.0 : 1_000_000.0)
        let partialRotaryFactor = ropeParams["partial_rotary_factor"]?.asFloat() ?? (isSliding ? 1.0 : 0.25)
        let ropeDims = max(1, Int(Float(headDim) * partialRotaryFactor))

        self.rope = initializeRope(
            dims: ropeDims, base: ropeTheta, traditional: config.ropeTraditional,
            scalingConfig: nil, maxPositionEmbeddings: nil)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil,
        sharedKV: (keys: MLXArray, values: MLXArray)? = nil,
        sharedOffset: Int? = nil,
        sharedOffsetArray: MLXArray? = nil
    ) -> (output: MLXArray, keys: MLXArray, values: MLXArray, offset: Int) {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = queryProj(x).reshaped(B, L, nHeads, headDim)
        queries = queryNorm(queries)
        queries = queries.transposed(0, 2, 1, 3)

        let cachedKeys: MLXArray
        let cachedValues: MLXArray
        let usedOffset: Int

        if let sharedKV {
            // Shared KV path: skip K/V projection, use source layer's keys/values.
            // Use per-sequence offsets from BatchKVCache when available for correct
            // batched RoPE — otherwise fall back to scalar offset.
            usedOffset = sharedOffset ?? 0
            if let sharedOffsetArray {
                queries = rope(queries, offset: sharedOffsetArray)
            } else {
                queries = rope(queries, offset: usedOffset)
            }
            cachedKeys = sharedKV.keys
            cachedValues = sharedKV.values
        } else {
            // Normal path: project K/V, apply RoPE, update cache
            usedOffset = cache?.offset ?? 0

            var keys = keyProj(x).reshaped(B, L, nKVHeads, headDim)

            let values: MLXArray
            if useKEqV {
                values = rmsNormNoScale(keys, eps: eps)
            } else if let valueProj {
                values = rmsNormNoScale(valueProj(x).reshaped(B, L, nKVHeads, headDim), eps: eps)
            } else {
                values = rmsNormNoScale(keys, eps: eps)
            }

            keys = keyNorm(keys)

            let valuesT = values.transposed(0, 2, 1, 3)
            var keysT = keys.transposed(0, 2, 1, 3)
            keysT = applyRotaryPosition(rope, to: keysT, cache: cache)
            queries = applyRotaryPosition(rope, to: queries, cache: cache)

            if let cache {
                (cachedKeys, cachedValues) = cache.update(keys: keysT, values: valuesT)
            } else {
                (cachedKeys, cachedValues) = (keysT, valuesT)
            }
        }

        // Matches reference vmlx-swift-lm: Load.swift's convertToBFloat16
        // already casts all fp16 params to bf16 at load time, so `queries`
        // arrives as bf16 (same exponent range as fp32 → no overflow).
        // No SDPA upcast needed — let MLXFast run natively on whatever
        // dtype the ingredients have. Previous fp32 upcast + cast-back
        // was a ~2× slowdown vs doing SDPA in bf16 directly.
        let sdpa = MLXFast.scaledDotProductAttention(
            queries: queries, keys: cachedKeys, values: cachedValues,
            scale: scale, mask: mask
        )
        let output = sdpa.transposed(0, 2, 1, 3).reshaped(B, L, -1)

        return (outputProj(output), cachedKeys, cachedValues, usedOffset)
    }
}

// MARK: - Dense MLP

class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        super.init()
    }

    // FIXME(iter-145 §215): dense Gemma-4-31B decode is ~10× slower
    // than expected (live-measured 11.24 tok/s vs reference ~70-100
    // tok/s on identical M4 Max). iter-146 §216 tried swapping the
    // GELU-gate + elementwise-multiply for the compile-fused
    // `fusedGeGLU` kernel; live bench post-swap = 10.70 tok/s (within
    // noise). The activation fusion is NOT the bottleneck — kernel
    // launches for the 3 linear layers + attention dominate. Root
    // cause per MEMORY.md "stale files in hot path": this file is
    // stale vs reference vmlx-swift-lm which ships fused gate+up
    // matmul (weight concat at load time → N/2 matmul dispatches)
    // + FP32 SDPA upcast.
    //
    // Remaining porting plan:
    //   1. Concatenate gate_proj.weight + up_proj.weight at load
    //      time → single matmul producing [2*hidden]; split then
    //      apply fused GELU. Need careful handling for QuantizedLinear
    //      (concatenating quantized weights is non-trivial) and
    //      for JANG_4M's per-row mixed bits.
    //   2. Verify FP32 SDPA upcast matches reference Gemma4.swift
    //      VLM. Our current path uses native fp16 SDPA which may be
    //      numerically fine but not obviously faster.
    //   3. Target ≥30 tok/s (bandwidth cap for 15.5 GB/step).
    // Tracks task #23 (vmlx#83 perf regression vs LM Studio).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let g = safeGeluApproximate(gateProj(x))
        let u = upProj(x)
        let product: MLXArray
        product = g * u
        return downProj(product)
    }
}

// MARK: - ScaledLinear (for per-layer model projection)

/// Linear layer with fixed output scaling (not a learnable parameter).
/// Python: (x @ self.weight.T) * scalar
class Gemma4ScaledLinear: Module {
    let weight: MLXArray
    let scalar: Float

    init(inputDims: Int, outputDims: Int, scalar: Float) {
        self.weight = MLXArray.zeros([outputDims, inputDims])
        self.scalar = scalar
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        matmul(x, weight.T) * scalar
    }
}

// MARK: - Router (Softmax, with RMSNormNoScale pre-norm)

class Gemma4Router: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "scale") var routerScale: MLXArray
    @ModuleInfo(key: "per_expert_scale") var perExpertScale: MLXArray
    // norm is RMSNormNoScale (no learnable weight, not in checkpoint)

    let numExperts: Int
    let topK: Int
    let rootSize: Float
    let eps: Float

    init(_ config: Gemma4TextConfiguration) {
        self.numExperts = config.numExperts
        self.topK = config.topKExperts
        self.rootSize = pow(Float(config.hiddenSize), -0.5)
        self.eps = config.rmsNormEps
        self._proj.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
        self._routerScale.wrappedValue = MLXArray.ones([config.hiddenSize])
        self._perExpertScale.wrappedValue = MLXArray.ones([config.numExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        // Pre-norm (RMSNormNoScale — no learnable weight)
        var h = rmsNormNoScale(x, eps: eps)
        h = h * rootSize
        h = h * routerScale

        let expertScores = proj(h)
        // softmax already computes in float32 internally — no explicit cast needed
        let routerProbs = softmax(expertScores, axis: -1)

        // Top-K via argPartition on negated scores (get highest scores).
        // Unary `-` keeps the dtype; the scalar-subtract form would insert
        // an extra AsType + broadcast per step. See §27.
        let topKIndices = argPartition(
            -expertScores,
            kth: topK - 1, axis: -1
        )[.ellipsis, ..<topK]

        var topKWeights = takeAlong(routerProbs, topKIndices, axis: -1)
        // Renormalize
        topKWeights = topKWeights / topKWeights.sum(axis: -1, keepDims: true)
        // Per-expert scale indexed by selected experts
        topKWeights = topKWeights * perExpertScale[topKIndices]

        return (indices: topKIndices, weights: topKWeights)
    }
}

// MARK: - Experts wrapper (matches Python's experts.switch_glu module tree)

class Gemma4Experts: Module {
    @ModuleInfo(key: "switch_glu") var switchGLU: SwitchGLU
    /// Optional Flash MoE replacement for `switchGLU`. When installed
    /// (via `Gemma4DecoderLayer.replaceSwitchGLU(...)`), this shim
    /// streams experts from SSD and caches hot sets in the slot bank.
    /// Not a `@ModuleInfo` property — the slot bank manages its own
    /// weight residency and the Module walk shouldn't traverse into it.
    /// FIX-G-K (2026-04-16): added so Gemma 4 MoE can opt into Flash
    /// MoE without changing `switchGLU`'s concrete type (which would
    /// require a protocol dispatch and regress decode tok/s).
    fileprivate var flashMoEShim: FlashMoESwitchGLUShim?

    init(_ config: Gemma4TextConfiguration) {
        self._switchGLU.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts,
            activation: { safeGeluApproximate($0) },
            bias: false)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, indices: MLXArray, weights: MLXArray
    ) -> MLXArray {
        let (B, S, H) = (x.dim(0), x.dim(1), x.dim(2))
        let K = indices.dim(-1)
        // Inline reshape (matches reference). Previous version held
        // intermediate `xFlat`/`indicesFlat` named bindings plus a
        // Flash MoE shim branch, retaining graph nodes + adding
        // per-step overhead that halved Gemma4 decode tok/s.
        //
        // FIX-G-K: single optional-check for Flash MoE. When shim is nil
        // (default, 99% of users), this is one integer comparison and
        // the native `switchGLU` path runs unchanged. When shim is set,
        // we delegate to the streaming replacement.
        let expertOut: MLXArray
        if let shim = flashMoEShim {
            expertOut = shim(x.reshaped(B * S, H), indices: indices.reshaped(B * S, K))
        } else {
            expertOut = switchGLU(x.reshaped(B * S, H), indices.reshaped(B * S, K))
        }
        let weightsFlat = expandedDimensions(weights.reshaped(B * S, K), axis: -1)
        return (expertOut * weightsFlat).sum(axis: -2).reshaped(B, S, H)
    }
}

// MARK: - Decoder Layer (Dense and MoE)

class Gemma4DecoderLayer: Module {
    let hasMoE: Bool

    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4Attention
    @ModuleInfo var mlp: Gemma4MLP

    // MoE components (nil for dense models)
    @ModuleInfo var router: Gemma4Router?
    @ModuleInfo var experts: Gemma4Experts?

    @ModuleInfo(key: "input_layernorm") var inputLayernorm: Gemma4RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: Gemma4RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: Gemma4RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: Gemma4RMSNorm

    // MoE-only norms (nil for dense models)
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayernorm2: Gemma4RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayernorm1: Gemma4RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayernorm2: Gemma4RMSNorm?

    // Per-layer input gating (E2B/E4B models, nil for 26B/31B)
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: Gemma4RMSNorm?

    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(_ config: Gemma4TextConfiguration, layerIndex: Int) {
        self.hasMoE = config.enableMoeBlock && config.numExperts > 0

        self._selfAttention.wrappedValue = Gemma4Attention(config, layerIndex: layerIndex)

        // Double-wide MLP for KV-shared layers (E2B)
        let firstKvShared = config.numHiddenLayers - config.numKvSharedLayers
        let isKvSharedLayer = config.numKvSharedLayers > 0 && layerIndex >= firstKvShared
        let effectiveIntermediate =
            (config.useDoubleWideMlp && isKvSharedLayer)
            ? config.intermediateSize * 2 : config.intermediateSize
        self.mlp = Gemma4MLP(
            dimensions: config.hiddenSize, hiddenDimensions: effectiveIntermediate)

        if hasMoE {
            self.router = Gemma4Router(config)
            self.experts = Gemma4Experts(config)
            self._preFeedforwardLayernorm2.wrappedValue = Gemma4RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayernorm1.wrappedValue = Gemma4RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayernorm2.wrappedValue = Gemma4RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        // Per-layer input gating (E2B/E4B)
        if config.hiddenSizePerLayerInput > 0 {
            self._perLayerInputGate.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSizePerLayerInput, bias: false)
            self._perLayerProjection.wrappedValue = Linear(
                config.hiddenSizePerLayerInput, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = Gemma4RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        self._inputLayernorm.wrappedValue = Gemma4RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = Gemma4RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayernorm.wrappedValue = Gemma4RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayernorm.wrappedValue = Gemma4RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        self._layerScalar.wrappedValue = MLXArray([Float(1.0)])

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil,
        sharedKV: (keys: MLXArray, values: MLXArray)? = nil,
        sharedOffset: Int? = nil,
        sharedOffsetArray: MLXArray? = nil
    ) -> (h: MLXArray, keys: MLXArray, values: MLXArray, offset: Int) {
        // Attention block
        var residual = x
        let (attnOut, keys, values, offset) = selfAttention(
            inputLayernorm(x), mask: mask, cache: cache,
            sharedKV: sharedKV, sharedOffset: sharedOffset,
            sharedOffsetArray: sharedOffsetArray)
        var h = postAttentionLayernorm(attnOut)
        h = residual + h

        residual = h

        if hasMoE, let router, let experts,
            let preFeedforwardLayernorm2,
            let postFeedforwardLayernorm1,
            let postFeedforwardLayernorm2
        {
            var h1 = preFeedforwardLayernorm(h)
            h1 = mlp(h1)
            h1 = postFeedforwardLayernorm1(h1)

            let (topKIndices, topKWeights) = router(h)
            var h2 = preFeedforwardLayernorm2(h)
            h2 = experts(h2, indices: topKIndices, weights: topKWeights)
            h2 = postFeedforwardLayernorm2(h2)

            h = h1 + h2
        } else {
            h = preFeedforwardLayernorm(h)
            h = mlp(h)
        }

        h = postFeedforwardLayernorm(h)
        h = residual + h

        // Per-layer input gating (E2B/E4B)
        if let perLayerInputGate, let perLayerProjection, let postPerLayerInputNorm,
            let perLayerInput
        {
            residual = h
            var gate = perLayerInputGate(h)
            gate = safeGeluApproximate(gate)
            gate = gate * perLayerInput
            gate = perLayerProjection(gate)
            gate = postPerLayerInputNorm(gate)
            h = residual + gate
        }

        h = h * layerScalar

        return (h, keys, values, offset)
    }
}

// MARK: - Inner Model

public class Gemma4Model: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Gemma4DecoderLayer]
    @ModuleInfo var norm: Gemma4RMSNorm

    // Per-layer embeddings (E2B/E4B models, nil for 26B/31B)
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection:
        Gemma4ScaledLinear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: Gemma4RMSNorm?

    let config: Gemma4TextConfiguration

    // KV sharing: maps layer index → source layer index for shared KVs
    let previousKVs: [Int]

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map { i in
            Gemma4DecoderLayer(config, layerIndex: i)
        }
        self.norm = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // Per-layer embeddings (E2B/E4B)
        if config.hiddenSizePerLayerInput > 0 {
            self._embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: config.vocabSizePerLayerInput,
                dimensions: config.numHiddenLayers * config.hiddenSizePerLayerInput)
            self._perLayerModelProjection.wrappedValue = Gemma4ScaledLinear(
                inputDims: config.hiddenSize,
                outputDims: config.numHiddenLayers * config.hiddenSizePerLayerInput,
                scalar: pow(Float(config.hiddenSize), -0.5))
            self._perLayerProjectionNorm.wrappedValue = Gemma4RMSNorm(
                dimensions: config.hiddenSizePerLayerInput, eps: config.rmsNormEps)
        }

        // KV sharing map
        let layerTypes = config.layerTypes.isEmpty
            ? Array(repeating: "sliding_attention", count: config.numHiddenLayers)
            : config.layerTypes
        var prevKVs = Array(0 ..< config.numHiddenLayers)
        if config.numKvSharedLayers > 0 {
            let firstKvShared = config.numHiddenLayers - config.numKvSharedLayers
            var kvsByType: [String: Int] = [:]
            for i in 0 ..< firstKvShared {
                kvsByType[layerTypes[i]] = i
            }
            for j in firstKvShared ..< config.numHiddenLayers {
                if let src = kvsByType[layerTypes[j]] {
                    prevKVs[j] = src
                }
            }
        }
        self.previousKVs = prevKVs

        super.init()
    }

    // MARK: Per-layer input processing

    private func getPerLayerInputs(_ inputIds: MLXArray) -> MLXArray? {
        guard let embedTokensPerLayer else { return nil }
        var result = embedTokensPerLayer(inputIds)
        let scale = pow(Float(config.hiddenSizePerLayerInput), 0.5)
        result = result * scale
        // Reshape from [B, L, numLayers * hiddenPerLayer] → [B, L, numLayers, hiddenPerLayer]
        let shape = Array(inputIds.shape) + [
            config.numHiddenLayers, config.hiddenSizePerLayerInput,
        ]
        return result.reshaped(shape)
    }

    private func projectPerLayerInputs(
        _ inputEmbeds: MLXArray, perLayerInputs: MLXArray?
    ) -> MLXArray? {
        guard let perLayerModelProjection, let perLayerProjectionNorm else { return nil }
        var proj = perLayerModelProjection(inputEmbeds)
        let shape =
            Array(inputEmbeds.shape.dropLast()) + [
                config.numHiddenLayers, config.hiddenSizePerLayerInput,
            ]
        proj = proj.reshaped(shape)
        proj = perLayerProjectionNorm(proj)

        guard let perLayerInputs else { return proj }
        return (proj + perLayerInputs) * pow(Float(2.0), Float(-0.5))
    }

    func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache?]? = nil
    ) -> MLXArray {
        // Ensure batch dimension — callers may pass 1D tokens [N] on cache-reuse turns
        let inputs = inputs.ndim == 1 ? inputs.expandedDimensions(axis: 0) : inputs
        var h = embedTokens(inputs)
        h = h * MLXArray(sqrt(Float(config.hiddenSize)), dtype: h.dtype)

        // Per-layer inputs (E2B/E4B)
        var perLayerInputsList: [MLXArray?]
        if config.hiddenSizePerLayerInput > 0 {
            let rawPLI = getPerLayerInputs(inputs)
            if let finalPLI = projectPerLayerInputs(h, perLayerInputs: rawPLI) {
                perLayerInputsList = (0 ..< layers.count).map { i in
                    finalPLI[0..., 0..., i, 0...]
                }
            } else {
                perLayerInputsList = Array(repeating: nil, count: layers.count)
            }
        } else {
            perLayerInputsList = Array(repeating: nil, count: layers.count)
        }

        let layerCache = cache ?? Array(repeating: nil as KVCache?, count: layers.count)

        // Build masks per layer type (uses first cache of each type)
        let layerTypes = config.layerTypes.isEmpty
            ? Array(repeating: "sliding_attention", count: config.numHiddenLayers)
            : config.layerTypes
        let globalLayerIdx = layerTypes.firstIndex(of: "full_attention")
            ?? (config.numHiddenLayers - 1)
        let slidingLayerIdx = layerTypes.firstIndex(of: "sliding_attention") ?? 0

        let globalCache: KVCache? = cache.flatMap {
            globalLayerIdx < $0.count ? $0[globalLayerIdx] : nil
        }
        let slidingCache: KVCache? = cache.flatMap {
            slidingLayerIdx < $0.count ? $0[slidingLayerIdx] : nil
        }

        let globalMask = createAttentionMask(h: h, cache: globalCache)
        let slidingWindowMask = createAttentionMask(
            h: h, cache: slidingCache, windowSize: config.slidingWindow)

        // Track intermediates for KV sharing.
        // offsetArray carries per-sequence [B]-shaped offsets from BatchKVCache for
        // correct batched RoPE on shared layers.
        var intermediates: [(keys: MLXArray, values: MLXArray, offset: Int, offsetArray: MLXArray?)?] =
            Array(repeating: nil, count: layers.count)

        for (i, layer) in layers.enumerated() {
            let layerType = i < layerTypes.count ? layerTypes[i] : "sliding_attention"
            let isGlobal = layerType == "full_attention"
            let layerMask = isGlobal ? globalMask : slidingWindowMask

            // Determine if this layer uses shared KVs
            let prevIdx = previousKVs[i]
            let sharedKV: (keys: MLXArray, values: MLXArray)?
            let sharedOffset: Int?
            let sharedOffsetArray: MLXArray?
            if prevIdx != i, let prev = intermediates[prevIdx] {
                sharedKV = (keys: prev.keys, values: prev.values)
                sharedOffset = prev.offset
                sharedOffsetArray = prev.offsetArray
            } else {
                sharedKV = nil
                sharedOffset = nil
                sharedOffsetArray = nil
            }

            let layerCacheEntry = prevIdx == i
                ? (i < layerCache.count ? layerCache[i] : nil) : nil

            let result = layer(
                h, mask: layerMask, cache: layerCacheEntry,
                perLayerInput: perLayerInputsList[i],
                sharedKV: sharedKV, sharedOffset: sharedOffset,
                sharedOffsetArray: sharedOffsetArray)

            h = result.h
            let layerOffsetArray = (layerCacheEntry as? BatchKVCache)?.offsetArray
            intermediates[i] = (keys: result.keys, values: result.values, offset: result.offset, offsetArray: layerOffsetArray)
        }

        return norm(h)
    }
}

// MARK: - Top-Level Model

public class Gemma4TextModel: Module, LLMModel {

    @ModuleInfo public var model: Gemma4Model
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let config: Gemma4TextConfiguration
    public var vocabularySize: Int { config.vocabSize }

    public init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.model = Gemma4Model(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        // Pad cache with nil for KV-shared layers (cache array may be shorter than layer count)
        let cacheArray: [KVCache?]? = cache.map { c in
            c.map { $0 as KVCache? }
                + Array(
                    repeating: nil as KVCache?,
                    count: max(0, config.numHiddenLayers - c.count))
        }
        var out = model(inputs, cache: cacheArray)

        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }

        if let cap = config.finalLogitSoftcapping, cap > 0 {
            // dtype-matched cap — see §27.
            out = compiledLogitSoftcap(out, MLXArray(cap, dtype: out.dtype))
        }

        return out
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String:
        MLXArray]
    {
        var processedWeights = [String: MLXArray]()

        for (key, value) in weights {
            var newKey = key

            // Strip VLM prefixes: model.language_model.X → model.X (JANG)
            if newKey.hasPrefix("model.language_model.") {
                newKey = "model." + String(newKey.dropFirst("model.language_model.".count))
            }
            // Strip VLM prefixes: language_model.X → X (mlx-community)
            else if newKey.hasPrefix("language_model.") {
                newKey = String(newKey.dropFirst("language_model.".count))
            }

            // Skip vision/audio/projector weights (not used in text-only mode)
            if newKey.hasPrefix("vision_tower.") || newKey.hasPrefix("model.vision_tower.")
                || newKey.hasPrefix("multi_modal_projector.")
                || newKey.hasPrefix("model.embed_vision.")
                || newKey.hasPrefix("embed_vision.")
                || newKey.hasPrefix("audio_tower.") || newKey.hasPrefix("model.audio_tower.")
                || newKey.hasPrefix("embed_audio.") || newKey.hasPrefix("model.embed_audio.")
            {
                continue
            }

            // Remap JANG expert naming to match module tree:
            // JANG uses:          switch_mlp.{gate,up,down}_proj.*
            // mlx-community uses: experts.switch_glu.{gate,up,down}_proj.*
            // Module tree expects: experts.switch_glu.{gate,up,down}_proj.*
            if newKey.contains(".switch_mlp.") {
                newKey = newKey.replacingOccurrences(of: ".switch_mlp.", with: ".experts.switch_glu.")
            }

            processedWeights[newKey] = value
        }

        // Trim vocab-dimension tensors to match config
        let expectedVocab = config.vocabSize
        for key in [
            "model.embed_tokens.weight", "model.embed_tokens.scales",
            "model.embed_tokens.biases",
            "lm_head.weight", "lm_head.scales", "lm_head.biases",
        ] {
            if let w = processedWeights[key], w.dim(0) != expectedVocab {
                processedWeights[key] = w[0 ..< expectedVocab]
            }
        }

        return processedWeights
    }

    // Per-layer-type cache: RotatingKVCache for sliding, KVCacheSimple for full attention.
    // For KV-shared models, only create caches for non-shared layers.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let firstKvShared = config.numKvSharedLayers > 0
            ? config.numHiddenLayers - config.numKvSharedLayers
            : config.numHiddenLayers
        return (0 ..< firstKvShared).map { i in
            let layerType =
                i < config.layerTypes.count ? config.layerTypes[i] : "sliding_attention"
            if layerType == "full_attention" {
                if let maxKVSize = parameters?.maxKVSize {
                    return RotatingKVCache(maxSize: maxKVSize, keep: 4)
                }
                return KVCacheSimple()
            } else {
                return RotatingKVCache(maxSize: config.slidingWindow, keep: 0)
            }
        }
    }
}

extension Gemma4TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Flash MoE conformance
//
// FIX-G-K (2026-04-16): Gemma 4 MoE layers use the sibling `router` +
// `experts` layout (not text-path mlp/block_sparse_moe). FlashMoE.apply
// walks `flashMoELayers`, asks each layer for `.flashMoELayout`, and for
// `.gemma4` layers calls `replaceSwitchGLU(with:)` which installs the
// streaming shim into `experts.flashMoEShim`. The Gemma4Experts
// `callAsFunction` checks for the shim via an if-let branch — single
// nil-check per layer per token when disabled (99% case), zero overhead
// for non-MoE layers which skip the extension branch entirely.
//
// Previous Swift tree omitted this conformance, so the v1.3.36 Python-side
// Flash MoE Gemma 4 fix silently did nothing in Swift. The shim was
// already ported (`FlashMoESwitchGLUShim.swift`), just unreachable.

extension Gemma4TextModel: FlashMoEReplaceable {
    public var flashMoELayers: [FlashMoELayer] {
        model.layers
    }
}

extension Gemma4DecoderLayer: FlashMoELayer {
    public var flashMoELayout: FlashMoELayout {
        hasMoE ? .gemma4 : .none
    }

    public func replaceSwitchGLU(with shim: FlashMoESwitchGLUShim) throws {
        // Only MoE layers receive this call (per flashMoELayout filter).
        // `experts` is non-nil whenever hasMoE is true.
        guard let experts = self.experts else { return }
        experts.flashMoEShim = shim
    }
}
