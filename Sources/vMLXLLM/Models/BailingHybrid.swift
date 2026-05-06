// Copyright © 2026 Jinho Jang. All rights reserved.
//
// Bailing-V2.5 Hybrid (Ling-2.6-flash) — `model_type = "bailing_hybrid"`.
//
// Port of `mlx_lm/models/bailing_hybrid.py` (688 lines). Spec source:
// `~/jang/research/LING-RUNTIME-ARCHITECTURE.md` (640 lines, May 2026).
//
// Architecture summary:
// - 32 base decoder layers + 1 MTP layer (skipped in standard generation).
// - Hybrid attention dispatch per layer:
//   * Global (softmax MLA): layers 7, 15, 23, 31  (when (idx+1) % 8 == 0)
//   * Linear (Lightning Attn-2 / GLA): all other layers
// - DSV2-style MLA with Bailing key naming: `dense` instead of `o_proj`,
//   `q_a/q_b/kv_a/kv_b/dense` projections.
// - Lightning Linear Attention with per-head ALiBi-style slopes,
//   GroupRMSNorm output, sigmoid-gated output projection.
// - Sigmoid-routed MoE with grouped routing + `e_score_correction_bias`
//   (DSV3-style "noaux_tc" — bias added for selection, weighting uses
//   ORIGINAL sigmoid scores).
// - Layer 0 is a pure dense MLP (`first_k_dense_replace = 1`).
// - MTP head exists at `model.layers.32` but is NOT invoked in the
//   standard forward path.
//
// Critical correctness invariants from the spec (§11 gotchas):
// 1. MLA uses `dense` not `o_proj` (key rename, not content change)
// 2. Linear-Attn `query_key_value` is FUSED — split with
//    `[num_attention_heads, num_attention_heads + num_key_value_heads]`
// 3. `g_norm` is GroupRMSNorm with `groups=4`, NOT regular RMSNorm
// 4. Linear-Attn slope uses `(layer_idx - 1)` clamped at 0
// 5. Linear-Attn rope is `traditional=False` (adjacent halves);
//    MLA rope is `traditional=True` (interleaved pairs)
// 6. MoE selection uses bias-corrected scores; weighting uses raw sigmoid
// 7. `expert_bias` is fp32 — never downcast
// 8. `mlp.gate.weight` (router) is fp16 passthrough — never quantize
//
// Capability defaults (from jang_config.json `capabilities`):
//   reasoning_parser = "deepseek_r1"  → think_xml stamp
//   tool_parser      = "deepseek"     → DSML tool format
//   think_in_template = false         → "detailed thinking off" default;
//                                       template scans system message for
//                                       "detailed thinking on" to flip.
// Token IDs: bos=156891, eos=156892 (`<|role_end|>`), eos2=156895,
// pad=156892. `add_bos_token=false`, `add_eos_token=false` — template
// owns boundaries; runtime should NOT auto-prepend BOS / append EOS.

import Foundation
import MLX
import vMLXLMCommon
import MLXNN

// MARK: - Configuration

public struct BailingHybridConfiguration: Codable, Sendable {
    public var modelType: String
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var maxPositionEmbeddings: Int
    public var moeIntermediateSize: Int
    public var numExperts: Int
    public var numSharedExperts: Int
    public var numAttentionHeads: Int
    public var numExpertsPerTok: Int
    public var numHiddenLayers: Int
    public var numKeyValueHeads: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var vocabSize: Int
    public var firstKDenseReplace: Int
    public var layerGroupSize: Int
    public var groupNormSize: Int

    // MLA-specific
    public var qLoraRank: Int?
    public var qkRopeHeadDim: Int
    public var qkNopeHeadDim: Int
    public var vHeadDim: Int
    public var kvLoraRank: Int
    public var ropeInterleave: Bool

    // MTP
    public var numNextnPredictLayers: Int

    // Routing
    public var normTopkProb: Bool
    public var routedScalingFactor: Float
    public var nGroup: Int
    public var topkGroup: Int
    public var scoreFunction: String
    public var moeRouterEnableExpertBias: Bool
    public var moeRouterEnableRoutedScaling: Bool
    public var moeSharedExpertIntermediateSize: Int?
    public var moeRouterEnableSharedExpert: Bool

    // General
    public var ropeScaling: [String: StringOrNumber]?
    public var ropeTraditional: Bool
    public var useBias: Bool
    public var useQkvBias: Bool
    public var useQkNorm: Bool
    public var tieWordEmbeddings: Bool
    public var partialRotaryFactor: Float
    public var headDim: Int?
    public var attentionBias: Bool

    // JANGTQ — populated by `LLMModelFactory._load`'s jang_config
    // merge cascade. `weightFormat == "mxtq"` switches the routed-MoE
    // expert MLPs from `SwitchGLU` (affine) to `TurboQuantSwitchGLU`
    // (codebook-quantized, requires sidecar). `mxtqBits` controls the
    // codebook width (2 or 4); `mxtqSeed` selects the Hadamard signs.
    public var weightFormat: String = "affine"
    public var mxtqBits: Int = 2
    public var mxtqSeed: Int = 42

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        // Required.
        self.modelType = try c.decode(String.self, forKey: .modelType)
        self.hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        self.maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        self.numExperts = try c.decode(Int.self, forKey: .numExperts)
        self.numSharedExperts =
            try c.decodeIfPresent(Int.self, forKey: .numSharedExperts) ?? 0
        self.numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        self.numExpertsPerTok = try c.decode(Int.self, forKey: .numExpertsPerTok)
        self.numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        self.numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        self.rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        self.ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        self.vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        self.firstKDenseReplace =
            try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 1
        // Spec §1: defaults trap — actual checkpoint uses 8, NOT the
        // python default 5. Always read from config.json.
        self.layerGroupSize =
            try c.decodeIfPresent(Int.self, forKey: .layerGroupSize) ?? 8
        self.groupNormSize =
            try c.decodeIfPresent(Int.self, forKey: .groupNormSize) ?? 4

        // MLA
        self.qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank)
        self.qkRopeHeadDim =
            try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 64
        self.qkNopeHeadDim =
            try c.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim) ?? 128
        self.vHeadDim = try c.decodeIfPresent(Int.self, forKey: .vHeadDim) ?? 128
        self.kvLoraRank =
            try c.decodeIfPresent(Int.self, forKey: .kvLoraRank) ?? 512
        self.ropeInterleave =
            try c.decodeIfPresent(Bool.self, forKey: .ropeInterleave) ?? true

        // MTP
        self.numNextnPredictLayers =
            try c.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers) ?? 0

        // Routing
        self.normTopkProb =
            try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        self.routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
        self.nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        self.topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 4
        self.scoreFunction =
            try c.decodeIfPresent(String.self, forKey: .scoreFunction) ?? "sigmoid"
        self.moeRouterEnableExpertBias =
            try c.decodeIfPresent(Bool.self, forKey: .moeRouterEnableExpertBias) ?? true
        self.moeRouterEnableRoutedScaling =
            try c.decodeIfPresent(Bool.self, forKey: .moeRouterEnableRoutedScaling) ?? true
        self.moeSharedExpertIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeSharedExpertIntermediateSize)
        self.moeRouterEnableSharedExpert =
            try c.decodeIfPresent(Bool.self, forKey: .moeRouterEnableSharedExpert) ?? true

        // General
        self.ropeScaling =
            try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        self.ropeTraditional =
            try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        self.useBias = try c.decodeIfPresent(Bool.self, forKey: .useBias) ?? false
        self.useQkvBias = try c.decodeIfPresent(Bool.self, forKey: .useQkvBias) ?? false
        self.useQkNorm = try c.decodeIfPresent(Bool.self, forKey: .useQkNorm) ?? true
        self.tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.partialRotaryFactor =
            try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.5
        self.headDim = try c.decodeIfPresent(Int.self, forKey: .headDim)
        self.attentionBias =
            try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false

        // JANGTQ overlay (merged from jang_config.json into config dict by
        // LLMModelFactory._load's resolution chain).
        self.weightFormat =
            try c.decodeIfPresent(String.self, forKey: .weightFormat) ?? "affine"
        // mxtq_bits ships in two shapes:
        //   1. Flat Int (older converters / DSV4 style after _load merge).
        //   2. Per-role dict {"routed_expert": 2, "attention": 8, ...}
        //      (Bailing/Ling JANGTQ converter — needs the routed value).
        if let flat = try? c.decodeIfPresent(Int.self, forKey: .mxtqBits) {
            self.mxtqBits = flat
        } else if let dict = try? c.decodeIfPresent(
            [String: Int].self, forKey: .mxtqBits),
            let routed = dict["routed_expert"] ?? dict["routed"]
                ?? dict.values.first
        {
            self.mxtqBits = routed
        } else if let routedTop = try? c.decodeIfPresent(
            Int.self, forKey: .routedExpertBits)
        {
            self.mxtqBits = routedTop
        } else {
            self.mxtqBits = 2
        }
        self.mxtqSeed = try c.decodeIfPresent(Int.self, forKey: .mxtqSeed) ?? 42
    }

    public var isJANGTQ: Bool {
        weightFormat.lowercased() == "mxtq"
            || weightFormat.lowercased() == "jangtq2"
            || weightFormat.lowercased() == "jangtq3"
            || weightFormat.lowercased() == "jangtq4"
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try c.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try c.encode(numExperts, forKey: .numExperts)
        try c.encode(numSharedExperts, forKey: .numSharedExperts)
        try c.encode(numAttentionHeads, forKey: .numAttentionHeads)
        try c.encode(numExpertsPerTok, forKey: .numExpertsPerTok)
        try c.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try c.encode(numKeyValueHeads, forKey: .numKeyValueHeads)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(ropeTheta, forKey: .ropeTheta)
        try c.encode(vocabSize, forKey: .vocabSize)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case moeIntermediateSize = "moe_intermediate_size"
        case numExperts = "num_experts"
        case numSharedExperts = "num_shared_experts"
        case numAttentionHeads = "num_attention_heads"
        case numExpertsPerTok = "num_experts_per_tok"
        case numHiddenLayers = "num_hidden_layers"
        case numKeyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case vocabSize = "vocab_size"
        case firstKDenseReplace = "first_k_dense_replace"
        case layerGroupSize = "layer_group_size"
        case groupNormSize = "group_norm_size"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case vHeadDim = "v_head_dim"
        case kvLoraRank = "kv_lora_rank"
        case ropeInterleave = "rope_interleave"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case normTopkProb = "norm_topk_prob"
        case routedScalingFactor = "routed_scaling_factor"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case scoreFunction = "score_function"
        case moeRouterEnableExpertBias = "moe_router_enable_expert_bias"
        case moeRouterEnableRoutedScaling = "moe_router_enable_routed_scaling"
        case moeSharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case moeRouterEnableSharedExpert = "moe_router_enable_shared_expert"
        case ropeScaling = "rope_scaling"
        case ropeTraditional = "rope_traditional"
        case useBias = "use_bias"
        case useQkvBias = "use_qkv_bias"
        case useQkNorm = "use_qk_norm"
        case tieWordEmbeddings = "tie_word_embeddings"
        case partialRotaryFactor = "partial_rotary_factor"
        case headDim = "head_dim"
        case attentionBias = "attention_bias"
        case weightFormat = "weight_format"
        case mxtqBits = "mxtq_bits"
        case mxtqSeed = "mxtq_seed"
        case routedExpertBits = "routed_expert_bits"
    }

    /// Spec §2: per-layer attention dispatch.
    /// `is_global = (layer_idx + 1) % layer_group_size == 0
    ///    OR layer_idx >= num_hidden_layers // layer_group_size * layer_group_size`
    /// For Ling-2.6-flash (group=8, layers=32) the second clause never
    /// fires (32 % 8 == 0), but other Bailing variants where
    /// `num_hidden_layers % layer_group_size != 0` rely on it for the
    /// trailing chunk.
    public func isGlobalLayer(_ layerIdx: Int) -> Bool {
        let cleanFloor = (numHiddenLayers / layerGroupSize) * layerGroupSize
        return ((layerIdx + 1) % layerGroupSize == 0) || (layerIdx >= cleanFloor)
    }
}

// MARK: - GroupRMSNorm

/// Spec §4: groupwise RMSNorm. Splits last dim into `groups`, normalizes
/// each group independently, then multiplies by the FULL-dim weight
/// (broadcasts across groups). Used by the Linear-Attn output path's
/// `g_norm` — different from regular RMSNorm.
class GroupRMSNorm: Module {
    let weight: MLXArray
    let groups: Int
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-5, groups: Int = 1) {
        self.weight = ones([dimensions])
        self.groups = groups
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let lastDim = x.shape.last!
        let groupDim = lastDim / groups
        var shape = Array(x.shape.dropLast())
        shape.append(groups)
        shape.append(groupDim)
        let reshaped = x.reshaped(shape)
        let normed = MLXFast.rmsNorm(reshaped, weight: ones([groupDim]), eps: eps)
        // Flatten last 2 dims back to lastDim.
        var flatShape = Array(normed.shape.dropLast(2))
        flatShape.append(lastDim)
        let flat = normed.reshaped(flatShape)
        return weight * flat
    }
}

// MARK: - Lightning Linear Attention (recurrent GLA)

/// Spec §4: per-head ALiBi-style slope schedule. NOT learned —
/// regenerated at init from `(layer_idx, num_hidden_layers)`. Negative
/// values are used as `exp(g)` decay factor in the recurrence.
///
/// Note: `layer_factor` clamps `(layer_idx - 1)` to non-negative, NOT
/// `layer_idx`. Preserved from the original code.
func gleLinearAttentionSlopes(
    numAttentionHeads: Int, layerIdx: Int, numHiddenLayers: Int
) -> MLXArray {
    func powerOf2(_ n: Int) -> [Float] {
        let logN = log2(Float(n))
        let base = pow(2.0, -pow(2.0, -(logN - 3.0)))
        return (0..<n).map { i in pow(base, Float(i + 1)) }
    }

    let n = numAttentionHeads
    let logN = log2(Float(n))
    let isPow2 = floor(logN) == logN
    var slopes: [Float]
    if isPow2 {
        slopes = powerOf2(n)
    } else {
        let p = Int(pow(2.0, floor(logN)))
        let base = powerOf2(p)
        let extra = stride(from: 0, to: 2 * p, by: 2).map { i -> Float in
            powerOf2(2 * p)[i]
        }
        slopes = base + Array(extra.prefix(n - p))
    }
    let denom = max(1, numHiddenLayers - 1)
    let layerPos = max(0, layerIdx - 1)
    let layerFactor = 1.0 - (Float(layerPos) / Float(denom)) + 1e-5
    let neg = slopes.map { -$0 * layerFactor }
    return MLXArray(neg)
}

/// Spec §4: per-token recurrent gated linear attention.
///
/// State `S` of shape `[B, H, K, K]`. For each timestep `t`:
///   `S = exp(g) * S + k_t.T @ v_t`
///   `y_t = q_t @ S * scale`
///
/// This is the simple O(L) recurrence — fine for decode (1 step at a
/// time). Prefill prefers a chunked variant (~200 LOC of CUDA in the
/// flash-linear-attention reference); for the first port we use the
/// straight loop.
func recurrentGLA(
    q: MLXArray, k: MLXArray, v: MLXArray,
    g: MLXArray, scale: Float,
    h: MLXArray?
) -> (MLXArray, MLXArray) {
    let L = q.shape[2]
    // 2026-05-04 fp16-overflow fix (mirror of Python upstream patch
    // applied to `mlx_lm.models.bailing_hybrid.recurrent_gla`):
    // promote q/k/v/h to fp32 internally and keep output in fp32.
    // Without this, on prompts ~80+ tokens the `k_t.T @ v_t` outer
    // products and accumulated `S * exp(g) + ...` exceeded fp16 max
    // (65504), producing inf → NaN logits ("Answer: " then garbage).
    // Verified upstream returns "Answer: C" correctly on the
    // 93-token MMLU prompt after the fp32 promotion.
    let qF = q.asType(.float32) * scale
    let kF = k.asType(.float32)
    let vF = v.asType(.float32)
    var state: MLXArray? = h?.asType(.float32)
    var outputs: [MLXArray] = []
    outputs.reserveCapacity(L)
    // Python: `mx.exp(g)[:, None, None]` keeps the head axis and adds
    // two trailing new-axes → shape [H, 1, 1]. Broadcasts against the
    // state [B, H, K, K] correctly. Swift's `[.newAxis, .newAxis]`
    // PREPENDS axes → wrong shape. Use `[0..., .newAxis, .newAxis]`
    // to keep the existing head dim and add trailing.
    let expG = exp(g.asType(.float32))[0..., .newAxis, .newAxis]
    // Force-evaluate every CHUNK timesteps so the lazy graph doesn't
    // accumulate L * (matmul + scale-add + matmul) deferred ops per
    // layer per token. Without this, prefill of a 100+ token prompt
    // across 28 linear-attn layers builds up gigabytes of unevaluated
    // intermediates and overflows RAM (observed as SIGKILL on
    // Ling-2.6-flash Turn 2 prompt encoding). 16 timesteps per chunk
    // is small enough to keep peak memory bounded but large enough
    // that the per-eval Metal dispatch overhead amortizes.
    let chunkStep = 16
    for t in 0..<L {
        let qT = qF[0..., 0..., t..<(t + 1), 0...]
        let kT = kF[0..., 0..., t..<(t + 1), 0...]
        let vT = vF[0..., 0..., t..<(t + 1), 0...]
        // [B, H, 1, K].T @ [B, H, 1, K] doesn't work directly — use
        // explicit transpose to get [B, H, K, 1] @ [B, H, 1, K] = [B, H, K, K]
        let kTrans = kT.transposed(0, 1, 3, 2)
        let hUp = matmul(kTrans, vT)
        let newState: MLXArray
        if let s = state {
            newState = s * expG + hUp
        } else {
            newState = hUp
        }
        state = newState
        // q_t @ S → [B, H, 1, K] @ [B, H, K, K] = [B, H, 1, K]
        let oT = matmul(qT, newState)
        outputs.append(oT)
        // Periodic in-loop eval — bound the lazy graph depth.
        if (t + 1) % chunkStep == 0 || t == L - 1 {
            MLX.eval(state!, oT)
        }
    }
    let out = concatenated(outputs, axis: 2)
    let finalState = state!
    MLX.eval(out, finalState)
    return (out, finalState)
}

class BailingLinearAttention: Module {
    let layerIdx: Int
    let useQkNorm: Bool
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let scale: Float
    let slope: MLXArray

    @ModuleInfo(key: "query_key_value") var queryKeyValue: Linear
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear
    @ModuleInfo(key: "g_norm") var gNorm: GroupRMSNorm

    @ModuleInfo(key: "key_layernorm") var keyLayerNorm: RMSNorm?
    @ModuleInfo(key: "query_layernorm") var queryLayerNorm: RMSNorm?

    let rope: RoPE

    init(_ args: BailingHybridConfiguration, layerIdx: Int) {
        self.layerIdx = layerIdx
        self.useQkNorm = args.useQkNorm
        self.numAttentionHeads = args.numAttentionHeads
        // Spec §4 LinearAttention treats kv_heads as = num_attention_heads
        // for the QKV split (the fused projection is sized
        // `(H + 2H) * head_dim`).
        self.numKeyValueHeads = args.numAttentionHeads
        self.headDim = args.headDim ?? (args.hiddenSize / args.numAttentionHeads)
        self.scale = pow(Float(self.headDim), -0.5)

        let qkvOut =
            (numAttentionHeads + 2 * numKeyValueHeads) * headDim
        self._queryKeyValue.wrappedValue = Linear(
            args.hiddenSize, qkvOut, bias: args.useQkvBias)
        self._dense.wrappedValue = Linear(
            numAttentionHeads * headDim, args.hiddenSize, bias: args.useBias)
        self._gProj.wrappedValue = Linear(
            args.hiddenSize, numAttentionHeads * headDim, bias: false)
        self._gNorm.wrappedValue = GroupRMSNorm(
            dimensions: numAttentionHeads * headDim,
            eps: args.rmsNormEps,
            groups: args.groupNormSize)

        if args.useQkNorm {
            self._keyLayerNorm.wrappedValue = RMSNorm(
                dimensions: headDim, eps: args.rmsNormEps)
            self._queryLayerNorm.wrappedValue = RMSNorm(
                dimensions: headDim, eps: args.rmsNormEps)
        }

        // Spec §7: linear-attn rope is `traditional=False` (adjacent
        // halves), partial dim = head_dim * partial_rotary_factor.
        let ropeDims = Int(Float(headDim) * args.partialRotaryFactor)
        self.rope = RoPE(
            dimensions: ropeDims, traditional: args.ropeTraditional,
            base: args.ropeTheta)

        self.slope = gleLinearAttentionSlopes(
            numAttentionHeads: numAttentionHeads,
            layerIdx: layerIdx,
            numHiddenLayers: args.numHiddenLayers)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: ArraysCache?,
        offset: Int
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let qkv = queryKeyValue(x).reshaped(
            B, L, numAttentionHeads + 2 * numKeyValueHeads, headDim)
        let parts = split(
            qkv,
            indices: [numAttentionHeads, numAttentionHeads + numKeyValueHeads],
            axis: 2)

        var queries = parts[0].transposed(0, 2, 1, 3)
        var keys = parts[1].transposed(0, 2, 1, 3)
        let values = parts[2].transposed(0, 2, 1, 3)

        if useQkNorm, let qNorm = queryLayerNorm, let kNorm = keyLayerNorm {
            queries = qNorm(queries)
            keys = kNorm(keys)
        }

        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        let priorState = cache?[0]
        let (output, newState) = recurrentGLA(
            q: queries, k: keys, v: values,
            g: slope, scale: scale, h: priorState)
        cache?[0] = newState

        let flat = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        let gated = gNorm(flat) * sigmoid(gProj(x))
        return dense(gated)
    }
}

// MARK: - MLA Attention (DSV2-style, Bailing naming)

class BailingMLAAttention: Module {
    let hiddenSize: Int
    let numHeads: Int
    let qLoraRank: Int?
    let qkRopeHeadDim: Int
    let kvLoraRank: Int
    let vHeadDim: Int
    let qkNopeHeadDim: Int
    let qHeadDim: Int
    var scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear
    @ModuleInfo(key: "dense") var dense: Linear

    let rope: RoPE

    init(_ args: BailingHybridConfiguration) {
        self.hiddenSize = args.hiddenSize
        self.numHeads = args.numAttentionHeads
        self.qLoraRank = args.qLoraRank
        self.qkRopeHeadDim = args.qkRopeHeadDim
        self.kvLoraRank = args.kvLoraRank
        self.vHeadDim = args.vHeadDim
        self.qkNopeHeadDim = args.qkNopeHeadDim
        self.qHeadDim = args.qkNopeHeadDim + args.qkRopeHeadDim
        self.scale = pow(Float(self.qHeadDim), -0.5)

        if let q = args.qLoraRank {
            self._qAProj.wrappedValue = Linear(
                args.hiddenSize, q, bias: args.useQkvBias)
            self._qALayerNorm.wrappedValue = RMSNorm(
                dimensions: q, eps: args.rmsNormEps)
            self._qBProj.wrappedValue = Linear(
                q, numHeads * qHeadDim, bias: false)
        } else {
            self._qProj.wrappedValue = Linear(
                args.hiddenSize, numHeads * qHeadDim,
                bias: args.attentionBias)
        }

        self._kvAProjWithMqa.wrappedValue = Linear(
            args.hiddenSize,
            kvLoraRank + qkRopeHeadDim,
            bias: args.useQkvBias)
        self._kvALayerNorm.wrappedValue = RMSNorm(
            dimensions: kvLoraRank, eps: args.rmsNormEps)
        self._kvBProj.wrappedValue = Linear(
            kvLoraRank,
            numHeads * (qkNopeHeadDim + vHeadDim),
            bias: false)
        self._dense.wrappedValue = Linear(
            numHeads * vHeadDim, args.hiddenSize,
            bias: args.useQkvBias)

        // YaRN scaling correction (matches DeepseekV3 path).
        if let ropeScaling = args.ropeScaling {
            let mScaleAllDim = ropeScaling["mscale_all_dim"]?.asFloat() ?? 0.0
            let scalingFactor = ropeScaling["factor"]?.asFloat() ?? 1.0
            if mScaleAllDim != 0, scalingFactor > 1 {
                let s = 0.1 * mScaleAllDim * log(scalingFactor) + 1.0
                self.scale = self.scale * s * s
            }
        }

        // Spec §7: MLA rope is `traditional=True` (interleaved pairs).
        self.rope = RoPE(
            dimensions: qkRopeHeadDim, traditional: args.ropeInterleave,
            base: args.ropeTheta)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var q: MLXArray
        if qLoraRank == nil {
            q = qProj!(x)
        } else {
            q = qBProj!(qALayerNorm!(qAProj!(x)))
        }
        q = q.reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let qSplit = split(q, indices: [qkNopeHeadDim], axis: -1)
        let qNope = qSplit[0]
        var qPe = qSplit[1]

        var compressedKv = kvAProjWithMqa(x)
        let kvSplit = split(compressedKv, indices: [kvLoraRank], axis: -1)
        compressedKv = kvSplit[0]
        var kPe = kvSplit[1]
        kPe = kPe.reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)

        var kv = kvBProj(kvALayerNorm(compressedKv))
        kv = kv.reshaped(B, L, numHeads, qkNopeHeadDim + vHeadDim)
            .transposed(0, 2, 1, 3)
        let kvPartSplit = split(kv, indices: [qkNopeHeadDim], axis: -1)
        let kNope = kvPartSplit[0]
        var values = kvPartSplit[1]

        let offset = cache?.offset ?? 0
        qPe = rope(qPe, offset: offset)
        kPe = rope(kPe, offset: offset)
        kPe = repeated(kPe, count: numHeads, axis: 1)

        let queries = concatenated([qNope, qPe], axis: -1)
        var keys = concatenated([kNope, kPe], axis: -1)

        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        let output = mlaScaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)
        return dense(output)
    }
}

// MARK: - MLP / MoE

class BailingMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(_ args: BailingHybridConfiguration, intermediateSize: Int? = nil) {
        let d = intermediateSize ?? args.intermediateSize
        self._gateProj.wrappedValue = Linear(args.hiddenSize, d, bias: args.useBias)
        self._downProj.wrappedValue = Linear(d, args.hiddenSize, bias: args.useBias)
        self._upProj.wrappedValue = Linear(args.hiddenSize, d, bias: args.useBias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

/// Spec §5: sigmoid-routed MoE gate with grouped routing.
/// `expert_bias` is added for SELECTION but the actual expert
/// weighting uses the ORIGINAL sigmoid scores (without the bias).
/// "noaux_tc" trick: prevents the auxiliary load-balancing loss from
/// biasing per-token outputs.
class BailingMoEGate: Module {
    let topK: Int
    let normTopkProb: Bool
    let nGroup: Int
    let topkGroup: Int
    let routedScalingFactor: Float
    let scoreFunction: String

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    /// Spec §5 (and §11.7): expert_bias ships in EVERY Bailing/Ling
    /// bundle (whether `moe_router_enable_expert_bias` is true or
    /// false — the bundle just zeros it when disabled). Modeled as a
    /// non-optional MLXArray with `@ParameterInfo(key: "expert_bias")`
    /// so the loader's `mlp.gate.expert_bias` key matches. When
    /// `moeRouterEnableExpertBias == false` the loaded value is all
    /// zeros and the addition is a no-op. fp32 — NEVER downcast (Bailing
    /// converter preserves dtype).
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray

    let useExpertBias: Bool

    init(_ args: BailingHybridConfiguration) {
        self.topK = args.numExpertsPerTok
        self.normTopkProb = args.normTopkProb
        self.nGroup = args.nGroup
        self.topkGroup = args.topkGroup
        self.routedScalingFactor = args.routedScalingFactor
        self.scoreFunction = args.scoreFunction
        self.useExpertBias = args.moeRouterEnableExpertBias

        // Spec §5: router weight is fp16 passthrough — never quantize.
        // The Linear here will inherit whatever dtype the loader leaves
        // for `mlp.gate.gate_proj.weight`; we just declare the layer.
        self._gateProj.wrappedValue = Linear(
            args.hiddenSize, args.numExperts, bias: false)

        // Default to a fp32 zero bias; the loader overwrites with the
        // bundle's per-expert bias. Sized to numExperts.
        self._expertBias.wrappedValue = zeros([args.numExperts], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let inType = x.dtype
        // Upcast to fp32 for the routing decision (sigmoid sensitive).
        let logits = gateProj(x.asType(.float32))
        let scoresF32: MLXArray
        if scoreFunction == "sigmoid" {
            scoresF32 = sigmoid(logits)
        } else {
            scoresF32 = softmax(logits, axis: -1)
        }
        let originalScores = scoresF32

        var scores = scoresF32
        if useExpertBias {
            scores = scores + expertBias
        }

        if nGroup > 1 {
            let lastDim = scores.shape.last!
            var grouped = scores.reshaped(
                Array(scores.shape.dropLast()) + [nGroup, lastDim / nGroup])
            // Top-2 within each group, sum → group score.
            let groupScoresFull = top(grouped, k: 2, axis: -1)
                .sum(axis: -1, keepDims: true)
            // Drop the lowest (nGroup - topkGroup) groups: argpartition
            // picks the (k-1)th element by partial sort along the
            // group axis (-2 here).
            let dropK = nGroup - topkGroup
            let dropIdx = argPartition(groupScoresFull, kth: dropK - 1, axis: -2)[
                .ellipsis, ..<dropK, 0...]
            // Broadcast dropIdx across the experts-per-group dim.
            var bShape = Array(grouped.shape)
            bShape[bShape.count - 2] = dropK
            let dropIdxBroadcast = broadcast(dropIdx, to: bShape)
            grouped = putAlong(
                grouped, stopGradient(dropIdxBroadcast),
                values: MLXArray(0.0, dtype: grouped.dtype),
                axis: -2)
            // Flatten back to [..., nExperts]
            scores = flattened(grouped, start: -2, end: -1)
        }

        // Top-k experts per token.
        let topkInds = argPartition(-scores, kth: topK - 1, axis: -1)[
            .ellipsis, ..<topK]
        var topkScores = takeAlong(originalScores, topkInds, axis: -1)
        if topK > 1, normTopkProb {
            let denom = topkScores.sum(axis: -1, keepDims: true)
                + MLXArray(1e-20, dtype: topkScores.dtype)
            topkScores = topkScores / denom
        }
        topkScores = topkScores * routedScalingFactor
        return (topkInds, topkScores.asType(inType))
    }
}

/// Affine (non-quantized) MoE block — used when bundle ships per-expert
/// `weight`/`scales`/`biases` keys (MXFP4, JANG_4M, etc.).
class BailingSparseMoE: Module, UnaryLayer {
    let layerIdx: Int
    let numExpertsPerTok: Int
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "gate") var gate: BailingMoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: BailingMLP?

    init(_ args: BailingHybridConfiguration, layerIdx: Int) {
        self.layerIdx = layerIdx
        self.numExpertsPerTok = args.numExpertsPerTok
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts,
            bias: args.useBias)
        self._gate.wrappedValue = BailingMoEGate(args)
        if args.numSharedExperts > 0, args.moeRouterEnableSharedExpert {
            let sharedDim = (args.moeSharedExpertIntermediateSize
                ?? args.moeIntermediateSize) * args.numSharedExperts
            self._sharedExperts.wrappedValue = BailingMLP(
                args, intermediateSize: sharedDim)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = gate(x)
        JangPressRouteTelemetry.recordTopK(layer: layerIdx, indices: indices)
        var out = switchMLP(x, indices)
        out = (out * weights[.ellipsis, .newAxis]).sum(axis: -2)
        if let shared = sharedExperts {
            out = out + shared(x)
        }
        return out
    }
}

/// JANGTQ MoE block — routed-expert MLPs use TurboQuant codebook
/// quantization. Shared experts stay affine. Mirror of
/// `DeepseekV4JANGTQMoE`. `weight_format == "mxtq"` (or jangtq2/4)
/// triggers this path via `BailingDecoderLayer` dispatch.
class BailingSparseMoEJANGTQ: Module, UnaryLayer {
    let layerIdx: Int
    let numExpertsPerTok: Int
    @ModuleInfo(key: "switch_mlp") var switchMLP: TurboQuantSwitchGLU
    @ModuleInfo(key: "gate") var gate: BailingMoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: BailingMLP?

    init(_ args: BailingHybridConfiguration, layerIdx: Int) {
        self.layerIdx = layerIdx
        self.numExpertsPerTok = args.numExpertsPerTok
        self._switchMLP.wrappedValue = TurboQuantSwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts,
            bits: args.mxtqBits,
            seed: args.mxtqSeed
            // swigluLimit defaults to 0 — Bailing/Ling does NOT use
            // the limited-SwiGLU clamp DSV4 needs.
        )
        self._gate.wrappedValue = BailingMoEGate(args)
        if args.numSharedExperts > 0, args.moeRouterEnableSharedExpert {
            let sharedDim = (args.moeSharedExpertIntermediateSize
                ?? args.moeIntermediateSize) * args.numSharedExperts
            self._sharedExperts.wrappedValue = BailingMLP(
                args, intermediateSize: sharedDim)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = gate(x)
        JangPressRouteTelemetry.recordTopK(layer: layerIdx, indices: indices)
        var out = switchMLP(x, indices)
        out = (out * weights[.ellipsis, .newAxis]).sum(axis: -2)
        if let shared = sharedExperts {
            out = out + shared(x)
        }
        return out
    }
}

// MARK: - Decoder layer (hybrid dispatch)

/// Both BailingMLAAttention and BailingLinearAttention live under the
/// same `attention` key in the bundle (the Python class has a single
/// `self.attention` attribute set at init time). A single
/// `@ModuleInfo(key: "attention") var attention: any BailingAttention`
/// field on the decoder layer keeps bundle key load correct.
protocol BailingAttention: Module {
    func callAsBailingAttention(
        _ x: MLXArray,
        attnMask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        offset: Int
    ) -> MLXArray
}

extension BailingMLAAttention: BailingAttention {
    func callAsBailingAttention(
        _ x: MLXArray,
        attnMask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        offset: Int
    ) -> MLXArray {
        // MLA reads cache.offset internally — `offset` arg is ignored.
        callAsFunction(x, mask: attnMask, cache: cache)
    }
}

extension BailingLinearAttention: BailingAttention {
    func callAsBailingAttention(
        _ x: MLXArray,
        attnMask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        offset: Int
    ) -> MLXArray {
        // Linear-Attn ignores `attnMask` (recurrence is implicitly causal).
        callAsFunction(x, cache: cache as? ArraysCache, offset: offset)
    }
}

class BailingDecoderLayer: Module {
    let isGlobal: Bool
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    @ModuleInfo(key: "attention") var attention: any BailingAttention
    let mlp: UnaryLayer

    init(_ args: BailingHybridConfiguration, layerIdx: Int) {
        self.isGlobal = args.isGlobalLayer(layerIdx)
        if isGlobal {
            self._attention.wrappedValue = BailingMLAAttention(args)
        } else {
            self._attention.wrappedValue = BailingLinearAttention(args, layerIdx: layerIdx)
        }
        if layerIdx >= args.firstKDenseReplace {
            if args.isJANGTQ {
                self.mlp = BailingSparseMoEJANGTQ(args, layerIdx: layerIdx)
            } else {
                self.mlp = BailingSparseMoE(args, layerIdx: layerIdx)
            }
        } else {
            self.mlp = BailingMLP(args)
        }
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attnMask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        offset: Int
    ) -> MLXArray {
        let r = attention.callAsBailingAttention(
            inputLayerNorm(x), attnMask: attnMask, cache: cache, offset: offset)
        let h = x + r
        let r2 = mlp(postAttentionLayerNorm(h))
        return h + r2
    }
}

// MARK: - Top-level model

public class BailingHybridLanguageModel: Module {
    let args: BailingHybridConfiguration

    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    var layers: [BailingDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let firstGlobalIdx: Int
    let firstLinearIdx: Int

    init(_ args: BailingHybridConfiguration) {
        self.args = args
        self._wordEmbeddings.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize)
        self.layers = (0..<args.numHiddenLayers).map { i in
            BailingDecoderLayer(args, layerIdx: i)
        }
        // MTP heads come right after with the same `model.layers.{i}`
        // index convention, but we don't currently invoke them in the
        // standard forward — they're a draft / spec-decode head per
        // the spec §6 ("Skip MTP in standard generation").
        self._norm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)

        self.firstGlobalIdx = layers.firstIndex(where: { $0.isGlobal }) ?? 0
        self.firstLinearIdx = layers.firstIndex(where: { !$0.isGlobal }) ?? 0
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = wordEmbeddings(inputs)

        // Use the FIRST global layer's cache for the attention mask
        // (KVCacheSimple offset gives us prompt position).
        let firstGlobalCache = cache?[firstGlobalIdx]
        let attnMask = createAttentionMask(h: h, cache: firstGlobalCache)
        let offset = firstGlobalCache?.offset ?? 0

        for (i, layer) in layers.enumerated() {
            let layerCache = (cache != nil && i < cache!.count) ? cache![i] : nil
            h = layer(h, attnMask: attnMask, cache: layerCache, offset: offset)
        }
        return norm(h)
    }
}

public class BailingHybridModel:
    Module, LLMModel, KVCacheDimensionProvider, LoRAModel
{
    public var kvHeads: [Int] = []
    let args: BailingHybridConfiguration
    public var model: BailingHybridLanguageModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: BailingHybridConfiguration) {
        self.args = args
        // Per-layer kvHeads: only the global (MLA) layers report the
        // standard num_kv_heads; linear-attn layers don't use a KVCache
        // dimensions provider in the conventional sense (they store
        // [B, H, K, K] state). Report num_attention_heads uniformly so
        // any caller that walks the array sees a coherent value.
        self.kvHeads = Array(
            repeating: args.numAttentionHeads,
            count: args.numHiddenLayers)
        self.model = BailingHybridLanguageModel(args)
        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                args.hiddenSize, args.vocabSize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let out = model(inputs, cache: cache)
        if args.tieWordEmbeddings {
            return model.wordEmbeddings.asLinear(out)
        }
        return lmHead!(out)
    }

    /// Spec §10: per-layer cache is heterogeneous.
    /// - Global (MLA) layers → KVCacheSimple (or RotatingKVCache if
    ///   parameters.maxKVSize is set, mirroring DeepseekV3's policy).
    /// - Linear-Attn layers → ArraysCache(size: 1) holding the per-head
    ///   GLA state.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        return (0..<args.numHiddenLayers).map { i in
            if args.isGlobalLayer(i) {
                if let maxKVSize = parameters?.maxKVSize {
                    return RotatingKVCache(maxSize: maxKVSize, keep: 4)
                }
                return KVCacheSimple()
            }
            return ArraysCache(size: 1)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = weights

        // Tied-embedding bundles ship `lm_head.weight` pointing at the
        // embedding; drop it so `tieWordEmbeddings == true` is honored.
        if args.tieWordEmbeddings {
            out.removeValue(forKey: "lm_head.weight")
            out.removeValue(forKey: "model.lm_head.weight")
        }

        // Per-layer routed-expert stack: the loader walks
        // `model.layers.{L}.mlp.experts.{e}.{gate,up,down}_proj.{key}`
        // and stacks each into
        // `model.layers.{L}.mlp.switch_mlp.{gate,up,down}_proj.{key}`.
        // Skip dense layers (layer 0 + MTP layer is also dense-MLA but
        // no MoE in the base path; MTP's per-expert keys land at
        // model.layers.{numHiddenLayers}.* and we don't currently load
        // the MTP layer either way).
        let numTotal = args.numHiddenLayers + args.numNextnPredictLayers
        // Memory-safe per-expert stacking strategy. Ling-2.6-flash ships
        // 73,728 per-expert keys (256 experts × 32 layers × 3 projections
        // × 3 keys each). Naïve `out[stackedKey] = stacked(tensors)` queues
        // a lazy op that keeps the per-expert MLX backing storage alive
        // even after we `removeValue` the dict entries — the lazy graph
        // still references them. Result: peak resident memory blows past
        // 100 GB on a 29 GB bundle. Fix: after stacking each layer's
        // worth of routed-expert tensors, force `MLX.eval` on the new
        // stacked arrays AND clear the allocator cache. That materializes
        // the stacks, drops the per-expert references, and releases the
        // intermediate buffers back to the OS before moving to the next
        // layer.
        var perLayerStacked: [MLXArray] = []
        for L in 0..<numTotal {
            let prefix = "model.layers.\(L)"
            guard L >= args.firstKDenseReplace else { continue }
            perLayerStacked.removeAll(keepingCapacity: true)
            for proj in ["gate_proj", "down_proj", "up_proj"] {
                for key in ["weight", "scales", "biases", "tq_packed", "tq_norms"] {
                    let first = "\(prefix).mlp.experts.0.\(proj).\(key)"
                    guard out[first] != nil else { continue }
                    var tensors: [MLXArray] = []
                    tensors.reserveCapacity(args.numExperts)
                    for e in 0..<args.numExperts {
                        let k = "\(prefix).mlp.experts.\(e).\(proj).\(key)"
                        guard let t = out[k] else { tensors = []; break }
                        tensors.append(t)
                    }
                    if tensors.count == args.numExperts {
                        let stackedKey =
                            "\(prefix).mlp.switch_mlp.\(proj).\(key)"
                        if out[stackedKey] == nil {
                            let s = loadTimeMaterializedStacked(tensors)
                            out[stackedKey] = s
                            perLayerStacked.append(s)
                        }
                        for e in 0..<args.numExperts {
                            out.removeValue(
                                forKey: "\(prefix).mlp.experts.\(e).\(proj).\(key)")
                        }
                    }
                }
                // Drop per-expert + prestacked-switch_mlp tq_bits scalars
                // — TurboQuantSwitchLinear takes the bit-width from the
                // BailingHybridConfiguration (mxtqBits), not per-tensor
                // metadata. The legacy per-expert layout ships
                // `mlp.experts.{e}.{proj}.tq_bits`; the prestacked
                // layout ships a single `mlp.switch_mlp.{proj}.tq_bits`.
                // Drop both regardless of which layout this bundle uses.
                for e in 0..<args.numExperts {
                    out.removeValue(
                        forKey: "\(prefix).mlp.experts.\(e).\(proj).tq_bits")
                }
                out.removeValue(
                    forKey: "\(prefix).mlp.switch_mlp.\(proj).tq_bits")
            }
            if !perLayerStacked.isEmpty {
                MLX.eval(perLayerStacked)
                MLX.Memory.clearCache()
            }

            // Spec §5: the gate router projection name in the bundle is
            // `mlp.gate.weight` (and optional `.bias`); our Swift module
            // lives at `mlp.gate.gate_proj`. Rename so @ModuleInfo keys
            // match.
            if let w = out["\(prefix).mlp.gate.weight"] {
                out["\(prefix).mlp.gate.gate_proj.weight"] = w
                out.removeValue(forKey: "\(prefix).mlp.gate.weight")
            }
            if let b = out["\(prefix).mlp.gate.bias"] {
                out["\(prefix).mlp.gate.gate_proj.bias"] = b
                out.removeValue(forKey: "\(prefix).mlp.gate.bias")
            }
            // expert_bias stays where it is — the @ModuleInfo path
            // matches `mlp.gate.expert_bias` directly via the `var
            // expertBias: MLXArray?` field on BailingMoEGate.
        }

        // Drop MTP-layer weights — we don't currently invoke the MTP
        // head in standard generation (spec §6). Filtering here keeps
        // MLX from complaining about unhandled `model.layers.{L >=
        // numHiddenLayers}.*` keys.
        let mtpPrefix = "model.layers.\(args.numHiddenLayers)"
        out = out.filter { key, _ in !key.hasPrefix(mtpPrefix) }

        return out
    }

    public var loraLayers: [Module] {
        // Conservative: expose the base decoder layers; LoRA wrapping
        // for hybrid attention isn't tuned yet.
        model.layers
    }
}
