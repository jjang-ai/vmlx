//
// Laguna — Poolside agentic-coding 33B/3B-active MoE.
//
// Architecture (from jang_tools/laguna/{config,model}.py @ jang-tools 2.5.x):
//   - 40 hybrid layers: layer 0 dense MLP, layers 1..39 sparse MoE
//   - Per-layer attention head count: 48 (full) or 64 (SWA), via
//     `num_attention_heads_per_layer`
//   - 8 KV heads (constant across layers)
//   - head_dim 128
//   - Hybrid mask: `full_attention` layers use causal mask; `sliding_attention`
//     layers use windowed causal mask (`sliding_window` = 512)
//   - Dual RoPE per `rope_parameters[layer_type]`:
//       full_attention    → YaRN base 500K factor 32 partial 0.5
//       sliding_attention → default base 10K partial 1.0
//   - q_norm / k_norm: per-head RMSNorm applied AFTER projection,
//     BEFORE rope/SDPA. Norm dim = head_dim (128).
//   - `g_proj` per-head sigmoid gating: gates SDPA output
//     element-wise per attention head before o_proj
//   - MoE topology: 256 routed experts top-8, expert dim 512, plus
//     1 shared expert with separate intermediate size 512
//   - Sigmoid routing with `e_score_correction_bias` (DeepSeek-V3 recipe)
//   - `moe_routed_scaling_factor`: 2.5
//   - `tie_word_embeddings`: false (separate lm_head)
//
// MXFP4 bundles route through this class via vanilla `Linear` weight
// loading. JANGTQ bundles need a paired `LagunaJANGTQ.swift` port that
// swaps each `Linear` for `JANGTQDenseLinear` (same pattern as
// Mistral3TextJANGTQ.swift). End-to-end weight-load + decode quality
// verification gated on a real bundle on disk.
//

import Foundation
import MLX
import MLXNN
import vMLXLMCommon

// MARK: - Permissive value enum for mixed-shape rope_parameters decode

internal enum StringOrNumberOrDict: Codable {
    case stringOrNumber(StringOrNumber)
    case dict([String: StringOrNumber])

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let d = try? c.decode([String: StringOrNumber].self) {
            self = .dict(d)
            return
        }
        if let s = try? c.decode(StringOrNumber.self) {
            self = .stringOrNumber(s)
            return
        }
        throw DecodingError.dataCorruptedError(
            in: c,
            debugDescription: "rope_parameters value must be a dict or scalar")
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .stringOrNumber(let s): try c.encode(s)
        case .dict(let d): try c.encode(d)
        }
    }
}

// MARK: - Configuration

public struct LagunaConfiguration: Codable, Sendable {
    public var modelType: String
    public var vocabularySize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var maxPositionEmbeddings: Int
    public var rmsNormEps: Float
    public var attentionBias: Bool
    public var slidingWindow: Int
    public var partialRotaryFactor: Float
    public var tieWordEmbeddings: Bool

    public var numExperts: Int
    public var numExpertsPerTok: Int
    public var moeIntermediateSize: Int
    public var sharedExpertIntermediateSize: Int
    public var moeRoutedScalingFactor: Float
    public var moeApplyRouterWeightOnInput: Bool
    public var gating: Bool

    /// Per-layer arrays. Defaults populated by `decode` if missing
    /// from the bundle's `config.json` (some converted bundles omit
    /// these; the Python reference back-fills in `__post_init__`).
    public var layerTypes: [String]
    public var mlpLayerTypes: [String]
    public var numAttentionHeadsPerLayer: [Int]

    /// Dual RoPE parameter dict keyed by layer type. Each entry has
    /// `rope_theta`, optional `partial_rotary_factor`, and any YaRN
    /// scaling fields (factor, original_max_position_embeddings, etc.)
    /// that aren't currently consumed by this Swift port. The simpler
    /// path uses the per-layer-type `rope_theta` + `partial_rotary_factor`
    /// directly via `MLXFast.RoPE`.
    public var ropeParameters: [String: [String: StringOrNumber]]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case attentionBias = "attention_bias"
        case slidingWindow = "sliding_window"
        case partialRotaryFactor = "partial_rotary_factor"
        case tieWordEmbeddings = "tie_word_embeddings"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case moeRoutedScalingFactor = "moe_routed_scaling_factor"
        case moeApplyRouterWeightOnInput = "moe_apply_router_weight_on_input"
        case gating
        case layerTypes = "layer_types"
        case mlpLayerTypes = "mlp_layer_types"
        case numAttentionHeadsPerLayer = "num_attention_heads_per_layer"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "laguna"
        self.vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 100352
        self.hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        self.intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 8192
        self.numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 40
        self.numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 48
        self.numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        self.headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        self.maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        self.partialRotaryFactor =
            try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.5
        self.tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 256
        self.numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 8
        self.moeIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 512
        self.sharedExpertIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 512
        self.moeRoutedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .moeRoutedScalingFactor) ?? 2.5
        self.moeApplyRouterWeightOnInput =
            try c.decodeIfPresent(Bool.self, forKey: .moeApplyRouterWeightOnInput) ?? false
        self.gating = try c.decodeIfPresent(Bool.self, forKey: .gating) ?? true

        let lt = try c.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        self.layerTypes = lt.isEmpty
            ? Array(repeating: "full_attention", count: numHiddenLayers) : lt

        let mlt = try c.decodeIfPresent([String].self, forKey: .mlpLayerTypes) ?? []
        self.mlpLayerTypes = mlt.isEmpty
            ? (["dense"] + Array(repeating: "sparse", count: max(0, numHiddenLayers - 1)))
            : mlt

        let nahpl = try c.decodeIfPresent([Int].self, forKey: .numAttentionHeadsPerLayer) ?? []
        self.numAttentionHeadsPerLayer = nahpl.isEmpty
            ? Array(repeating: numAttentionHeads, count: numHiddenLayers) : nahpl

        // `rope_parameters` is mixed-shape on real Laguna bundles:
        //
        //   {
        //     "full_attention":    { rope_theta, rope_type, factor, ... },
        //     "sliding_attention": { rope_theta, rope_type, ... },
        //     "original_max_position_embeddings": 4096  ← top-level scalar
        //   }
        //
        // Decoding directly as `[String: [String: StringOrNumber]]` fails
        // because the scalar value isn't a dict. Decode permissively by
        // walking the JSON object and keeping only the per-layer-type
        // sub-dicts (entries whose value is itself an object). The
        // top-level scalars (e.g. `original_max_position_embeddings`)
        // are propagated via `maxPositionEmbeddings` already and
        // duplicated inside the per-layer dicts where they're consumed.
        if let nested = try? c.decode([String: [String: StringOrNumber]].self,
                                      forKey: .ropeParameters) {
            self.ropeParameters = nested
        } else if let raw = try? c.decode([String: StringOrNumberOrDict].self,
                                          forKey: .ropeParameters) {
            var filtered: [String: [String: StringOrNumber]] = [:]
            for (k, v) in raw {
                if case .dict(let d) = v {
                    filtered[k] = d
                }
            }
            self.ropeParameters = filtered
        } else {
            self.ropeParameters = [:]
        }
    }

    /// Look up the per-layer-type rope theta + partial factor.
    /// Falls back to top-level config when the dict is missing entries.
    public func ropeFor(layerType: String) -> (theta: Float, partial: Float) {
        if let entry = ropeParameters[layerType] {
            let theta = entry["rope_theta"]?.asFloat() ?? 500_000
            let partial = entry["partial_rotary_factor"]?.asFloat()
                ?? partialRotaryFactor
            return (theta, partial)
        }
        // Defaults per the reference: full → YaRN-ish 500K, sliding → 10K
        let theta: Float = layerType == "sliding_attention" ? 10_000 : 500_000
        let partial: Float = layerType == "sliding_attention" ? 1.0 : partialRotaryFactor
        return (theta, partial)
    }
}

// MARK: - Attention

internal final class LagunaAttention: Module {
    let layerIndex: Int
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let kvGroups: Int
    let scale: Float
    let layerType: String
    let ropeBase: Float
    let partial: Float
    let ropeDim: Int

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "g_proj") var gProj: Linear?

    // 2026-05-01: Use the dispatching `RoPELayer` so YaRN-scaled
    // full-attention layers get their factor=32 / mscale stretch
    // applied. Plain `RoPE` for full_attention destroyed long-context
    // accuracy on prompts > 4096 tokens (Laguna's
    // original_max_position_embeddings) — same class of bug as the
    // softplus / un-biased-weights pair fixed alongside this. The
    // sliding-attention layers still use plain RoPE because their
    // rope_parameters entry is `rope_type: "default"`.
    let rope: RoPELayer

    init(_ cfg: LagunaConfiguration, layerIndex: Int) {
        self.layerIndex = layerIndex
        self.nHeads = cfg.numAttentionHeadsPerLayer[layerIndex]
        self.nKVHeads = cfg.numKeyValueHeads
        self.headDim = cfg.headDim
        self.kvGroups = nHeads / nKVHeads
        self.scale = pow(Float(headDim), -0.5)
        self.layerType = cfg.layerTypes[layerIndex]
        let (theta, partial) = cfg.ropeFor(layerType: layerType)
        self.ropeBase = theta
        self.partial = partial
        self.ropeDim = Int(Float(headDim) * partial)

        let h = cfg.hiddenSize
        self._wq.wrappedValue = Linear(h, nHeads * headDim, bias: cfg.attentionBias)
        self._wk.wrappedValue = Linear(h, nKVHeads * headDim, bias: cfg.attentionBias)
        self._wv.wrappedValue = Linear(h, nKVHeads * headDim, bias: cfg.attentionBias)
        self._wo.wrappedValue = Linear(nHeads * headDim, h, bias: cfg.attentionBias)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: cfg.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: cfg.rmsNormEps)
        if cfg.gating {
            self._gProj.wrappedValue = Linear(h, nHeads, bias: false)
        }

        // Build the per-layer-type scaling config dict for `initializeRope`.
        // Mirrors the Python remap (jang_tools/laguna/model.py:76-87):
        // pull this layer's entry from `rope_parameters[layer_type]`,
        // and rename `attention_factor` → `mscale` (HF YaRN nomenclature
        // → mlx-swift YarnRoPE expected key). For SWA layers whose
        // `rope_type == "default"`, `initializeRope` returns plain RoPE.
        var scalingCfg: [String: StringOrNumber]? = nil
        if let entry = cfg.ropeParameters[layerType] {
            let ropeType = entry["rope_type"].flatMap { v -> String? in
                if case .string(let s) = v { return s }
                return nil
            } ?? "default"
            if ropeType != "default" {
                var dict = entry
                if let af = dict["attention_factor"], dict["mscale"] == nil {
                    dict["mscale"] = af
                    dict["attention_factor"] = nil
                }
                // Drop the nil we just inserted.
                dict = dict.compactMapValues { $0 }
                scalingCfg = dict
            }
        }
        self.rope = initializeRope(
            dims: ropeDim,
            base: ropeBase,
            traditional: false,
            scalingConfig: scalingCfg,
            maxPositionEmbeddings: cfg.maxPositionEmbeddings)
    }

    private func applyPartialRope(_ t: MLXArray, cache: KVCache?) -> MLXArray {
        if ropeDim == headDim {
            return applyRotaryPosition(rope, to: t, cache: cache)
        }
        // Partial-rotary: rotate the first `ropeDim` channels, keep tail.
        let rot = t[.ellipsis, ..<ropeDim]
        let keep = t[.ellipsis, ropeDim...]
        let rotated = applyRotaryPosition(rope, to: rot, cache: cache)
        return MLX.concatenated([rotated, keep], axis: -1)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, T, _) = (x.dim(0), x.dim(1), x.dim(2))
        var q = wq(x).reshaped(B, T, nHeads, headDim).transposed(0, 2, 1, 3)
        var k = wk(x).reshaped(B, T, nKVHeads, headDim).transposed(0, 2, 1, 3)
        let v = wv(x).reshaped(B, T, nKVHeads, headDim).transposed(0, 2, 1, 3)
        // Per-head q_norm / k_norm AFTER projection, BEFORE rope (matches reference)
        q = qNorm(q)
        k = kNorm(k)
        q = applyPartialRope(q, cache: cache)
        k = applyPartialRope(k, cache: cache)

        let out = attentionWithCacheUpdate(
            queries: q, keys: k, values: v,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, T, nHeads * headDim)

        var result = out
        if let g = gProj {
            // Per-head softplus gating. Python `jang_tools/laguna/model.py`
            // line 158 + the comment block above it explicitly notes that
            // sigmoid (the obvious choice for an attention gate) drives
            // residual stream blow-up over 30+ layers — std grows from
            // 0.29 → 11, max → 616, output saturates into garbage tokens.
            // The HF reference uses softplus to AMPLIFY (unbounded
            // monotonic) rather than DAMP (sigmoid bounds [0,1]) the
            // attention output. Match exactly: cast to fp32 before
            // softplus to avoid fp16 overflow on long-tail logits, then
            // back to result dtype for the broadcast.
            let gate = softplus(g(x).asType(.float32)).asType(result.dtype)
            let gated = result.reshaped(B, T, nHeads, headDim) * gate.expandedDimensions(axis: -1)
            result = gated.reshaped(B, T, nHeads * headDim)
        }
        return wo(result)
    }
}

// MARK: - Dense MLP (layer 0)

internal final class LagunaDenseMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(hidden: Int, intermediate: Int) {
        self._gate.wrappedValue = Linear(hidden, intermediate, bias: false)
        self._up.wrappedValue = Linear(hidden, intermediate, bias: false)
        self._down.wrappedValue = Linear(intermediate, hidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Do not call MLXNN.silu here. Laguna's production fast path wraps
        // the whole decode step in `compile(inputs:outputs:)`; nesting the
        // MLXNN single-array compiled activation inside that outer trace can
        // return an empty result vector and SIGTRAP at `compile(...)[0]` on
        // long prompts. Use raw graph ops so the outer compile owns the
        // activation.
        let g = gate(x)
        return down((g * sigmoid(g)) * up(x))
    }
}

// MARK: - MoE Block
//
// 2026-05-01: rewritten to match real Laguna bundle layout.
//
// On-disk routed-expert tensors at `layers.<i>.mlp.experts.*`:
//
//   experts.gate_up_proj.tq_packed : (n_exp, 2 × moe_inter, packed_in_hidden)
//   experts.gate_up_proj.tq_norms  : (n_exp, 2 × moe_inter)
//   experts.down_proj.tq_packed    : (n_exp, hidden,         packed_in_moe_inter)
//   experts.down_proj.tq_norms     : (n_exp, hidden)
//
// i.e. all 256 experts are packed into a single stacked tensor (NOT 256
// individual modules), gate and up are FUSED on the out-dim axis (concat
// gate then up), and the projection is JANGTQ-codebook-quantized
// (`tq_packed` + `tq_norms`).
//
// Dense paths (attention Q/K/V/O, layer-0 dense MLP gate/up/down, the
// shared expert, and the router gate) are MLX standard affine quant
// (`weight` packed uint32 + `scales` + `biases` of shape [out, in/gs]).
// Those load through vanilla `Linear` because the MLX loader auto-
// substitutes `QuantizedLinear` based on the top-level `quantization`
// field in `config.json`.
//
// The previous implementation declared `experts: [LagunaDenseMLP]` —
// 256 individual modules — and iterated them with `.item()` syncs. That
// shape is incompatible with the bundle's stacked tensors and causes a
// hard load failure ("Unable to set layers.1.mlp.experts: 256 modules
// not compatible with [down_proj: tq_norms[256, 2048]…]").
//
// The new wiring uses `TurboQuantSwitchGLU` (the same codebook MoE
// primitive DeepSeek-V4 / Mistral 4 / NemotronH JANGTQ already use) and
// splits the fused `gate_up_proj` tensor at sanitize time into the
// `gate_proj` / `up_proj` halves the primitive expects.
//
// 2026-05-03: also support affine/mxfp4 Laguna bundles. Those ship the
// routed experts as standard MLX quant tensors (`weight` + `scales` +
// `biases`) rather than `tq_packed`/`tq_norms`. The MoE expert member is
// type-erased so JANGTQ uses `TurboQuantSwitchGLU` and affine uses
// `SwitchGLU`, while both formats share the router and fused-gate
// sanitize logic.

internal protocol LagunaSwitchMLPLayer: Module {
    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray
}

extension TurboQuantSwitchGLU: LagunaSwitchMLPLayer {}
extension SwitchGLU: LagunaSwitchMLPLayer {}

public struct LagunaMoEContext: Sendable {
    public let bits: Int
    public let mxtqSeed: Int

    public init(bits: Int, mxtqSeed: Int) {
        self.bits = bits
        self.mxtqSeed = mxtqSeed
    }
}

internal final class LagunaMoE: Module, UnaryLayer {
    let cfg: LagunaConfiguration
    let layerIndex: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray
    @ModuleInfo(key: "experts") var experts: Module
    @ModuleInfo(key: "shared_expert") var sharedExpert: LagunaDenseMLP

    init(_ cfg: LagunaConfiguration, layerIndex: Int, jangtq: LagunaMoEContext?) {
        self.cfg = cfg
        self.layerIndex = layerIndex
        self._gate.wrappedValue = Linear(cfg.hiddenSize, cfg.numExperts, bias: false)
        self._eScoreCorrectionBias.wrappedValue = MLXArray.zeros([cfg.numExperts])
        if let jangtq {
            self._experts.wrappedValue = TurboQuantSwitchGLU(
                inputDims: cfg.hiddenSize,
                hiddenDims: cfg.moeIntermediateSize,
                numExperts: cfg.numExperts,
                bits: jangtq.bits,
                seed: jangtq.mxtqSeed)
        } else {
            self._experts.wrappedValue = SwitchGLU(
                inputDims: cfg.hiddenSize,
                hiddenDims: cfg.moeIntermediateSize,
                numExperts: cfg.numExperts)
        }
        // Shared expert is affine-quant Linear (NOT codebook).
        self._sharedExpert.wrappedValue = LagunaDenseMLP(
            hidden: cfg.hiddenSize, intermediate: cfg.sharedExpertIntermediateSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, T, H). Routing operates on (B, T, num_experts).
        //
        // CRITICAL ordering — copied from `jang_tools/laguna/model.py:209`
        // (LagunaMoE.__call__) and the HF `modeling_laguna.py`
        // LagunaTopKRouter / LagunaSparseMoeBlock reference:
        //
        //     scores               = sigmoid(logits)                     # un-biased
        //     scores_for_selection = scores + e_score_correction_bias    # biased ONLY for top-k pick
        //     inds                 = argpartition(-scores_for_selection)[..., :k]
        //     topk_scores          = take_along(scores, inds, axis=-1)   # un-biased!
        //     topk_scores         /= sum(topk_scores, axis=-1, keepdims=True)
        //
        // The bias is used ONLY to decide WHICH experts are selected; the
        // gating weights themselves are the un-biased sigmoid scores.
        // Adding the bias to the weights drives residual stream blow-up
        // (Python comment quotes std growth 0.29 → 11 across 30 layers,
        // max=616 → garbage saturation). The previous Swift impl folded
        // the bias into the sigmoid input, reproducing exactly that bug.
        let logits = gate(x).asType(.float32)
        let scores = sigmoid(logits)                                  // un-biased
        let scoresForSelection = scores + eScoreCorrectionBias.asType(.float32)

        let topK = cfg.numExpertsPerTok
        // 2026-05-01: argPartition (O(n)) instead of argSort
        // (O(n log n)). For Laguna's 256 experts top-8 this is the
        // canonical Python ref pattern (see comment block at line 454,
        // and matches MiniMaxJANGTQ:127 / DeepseekV3:290 / Qwen35JANGTQ:72).
        // The order of returned indices is unspecified — irrelevant
        // here because TurboQuantSwitchGLU sums over the K dim regardless.
        let topkIdx = argPartition(
            -scoresForSelection, kth: topK - 1, axis: -1
        )[.ellipsis, ..<topK]                                          // (B, T, K)
        JangPressRouteTelemetry.recordTopK(layer: layerIndex, indices: topkIdx)
        var topkW = MLX.takeAlong(scores, topkIdx, axis: -1)          // (B, T, K) — un-biased!
        let normSum = topkW.sum(axis: -1, keepDims: true) + 1e-20
        topkW = topkW / normSum

        // SwitchGLU dispatch: returns (B, T, K, H). The concrete module is
        // either TurboQuantSwitchGLU for mxtq/JANGTQ or SwitchGLU for affine
        // mxfp4; both expose the same indexed expert contract.
        guard let switchLayer = experts as? LagunaSwitchMLPLayer else {
            fatalError(
                "LagunaMoE.experts must be TurboQuantSwitchGLU or SwitchGLU, "
                + "got \(type(of: experts))")
        }
        let yK = switchLayer(x, topkIdx)
        // Weight by router scores and sum over the K dim → (B, T, H).
        var y = (yK * topkW.expandedDimensions(axis: -1).asType(yK.dtype)).sum(axis: -2)
        // Routed scaling factor applies to the routed contribution only.
        // Shared expert is NOT scaled — matches HF order (model.py:243).
        y = y * cfg.moeRoutedScalingFactor + sharedExpert(x)
        return y.asType(x.dtype)
    }
}

// MARK: - Transformer Block

internal final class LagunaLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var attention: LagunaAttention
    let mlp: UnaryLayer
    let layerType: String
    let useSliding: Bool

    init(_ cfg: LagunaConfiguration, layerIndex: Int, jangtq: LagunaMoEContext?) {
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        self._attention.wrappedValue = LagunaAttention(cfg, layerIndex: layerIndex)
        if cfg.mlpLayerTypes[layerIndex] == "dense" {
            self.mlp = LagunaDenseMLP(
                hidden: cfg.hiddenSize, intermediate: cfg.intermediateSize)
        } else {
            self.mlp = LagunaMoE(cfg, layerIndex: layerIndex, jangtq: jangtq)
        }
        self.layerType = cfg.layerTypes[layerIndex]
        self.useSliding = layerType == "sliding_attention"
    }

    func callAsFunction(
        _ x: MLXArray,
        fullMask: MLXFast.ScaledDotProductAttentionMaskMode,
        swaMask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let mask = useSliding ? swaMask : fullMask
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

// MARK: - Top-Level Model

public class LagunaModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [LagunaLayer]
    let norm: RMSNorm
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    fileprivate let cfg: LagunaConfiguration
    fileprivate let faIdx: Int
    fileprivate let swaIdx: Int?

    /// Non-nil `jangtq` selects the codebook MoE path for mxtq/JANGTQ
    /// bundles. Nil selects the affine SwitchGLU path for mxfp4 bundles.
    public init(
        _ cfg: LagunaConfiguration,
        jangtq: LagunaMoEContext? = LagunaMoEContext(bits: 2, mxtqSeed: 42)
    ) {
        self.cfg = cfg
        self.vocabularySize = cfg.vocabularySize
        self.kvHeads = (0..<cfg.numHiddenLayers).map { _ in cfg.numKeyValueHeads }

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: cfg.vocabularySize, dimensions: cfg.hiddenSize)
        self.layers = (0..<cfg.numHiddenLayers).map {
            LagunaLayer(cfg, layerIndex: $0, jangtq: jangtq)
        }
        self.norm = RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        if !cfg.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(cfg.hiddenSize, cfg.vocabularySize, bias: false)
        }
        self.faIdx = cfg.layerTypes.firstIndex(of: "full_attention") ?? 0
        self.swaIdx = cfg.layerTypes.firstIndex(of: "sliding_attention")
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        let cacheArr = cache ?? []

        let fullMask = createAttentionMask(h: h, cache: cacheArr.isEmpty ? nil : cacheArr[faIdx])
        let swaMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let swaIdx, !cacheArr.isEmpty {
            swaMask = createAttentionMask(
                h: h, cache: cacheArr[swaIdx], windowSize: cfg.slidingWindow)
        } else {
            swaMask = .none
        }

        for (i, layer) in layers.enumerated() {
            h = layer(h, fullMask: fullMask, swaMask: swaMask,
                cache: cacheArr.isEmpty ? nil : cacheArr[i])
        }

        var out = norm(h)
        if cfg.tieWordEmbeddings {
            out = embedTokens.asLinear(out)
        } else if let lmHead {
            out = lmHead(out)
        }
        return out
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        // 2026-05-01: pick full-attention cache class based on
        // caller-supplied `kvMode` to avoid clobbering TurboQuant.
        //
        //   - When `kvMode == .turboQuant(...)` is set: keep
        //     `KVCacheSimple()` for full layers. `BatchQuantize.maybeCompress`
        //     in BatchEngine then promotes those Simple layers to
        //     `TurboQuantKVCache` once offset > `quantizedKVStart`,
        //     giving 4.7-5x memory savings. The cache family becomes
        //     heterogeneous (TurboQuantKVCache + RotatingKVCache) and
        //     compile is correctly skipped — the user picked memory
        //     over compile speed.
        //
        //   - Otherwise (no kvMode, or `.none`): route full-attention
        //     through `RotatingKVCache(maxSize: cfg.maxPositionEmbeddings,
        //     keep: 4)`. All 40 layers are now `RotatingKVCache` →
        //     `CacheFamily.classify` returns `.rotating` →
        //     `BatchCompile` Stage 3 promotes to
        //     `CompilableRotatingKVCache` and traces the forward.
        //     Decode speedup +15-25 % vs the heterogeneous baseline
        //     (verified against BENCH_BATCH_CHAT compile turns).
        //     For chat workloads < trained context the rotation never
        //     fires, so this is bit-identical to `KVCacheSimple` —
        //     same memory semantics, just a different concrete class.
        //
        // This mirrors the Stage-3 / Stage-2 compile-eligibility split
        // documented in `BatchEngine.swift:1140-1247`. Gemma4 still
        // takes the heterogeneous path by default because its
        // unbounded full-attention cache predates this knob; if/when
        // Gemma4's full layers also adopt this gating it'll get the
        // same speedup without losing TQ optionality.
        let callerWantsTQ: Bool = {
            guard let p = parameters else { return false }
            if case .turboQuant = p.kvMode { return true }
            return false
        }()
        let fullKVSize = parameters?.maxKVSize ?? cfg.maxPositionEmbeddings
        return cfg.layerTypes.map { layerType in
            if layerType == "sliding_attention" {
                return RotatingKVCache(maxSize: cfg.slidingWindow)
            }
            if callerWantsTQ {
                // Preserve `maybeQuantizeKVCache` promotion path.
                return KVCacheSimple()
            }
            return RotatingKVCache(maxSize: fullKVSize, keep: 4)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Map HF Laguna weight key prefixes to the Swift module pathing:
        //   - `model.layers.N.{...}` → `layers.N.{...}` (drop "model." prefix)
        //   - `model.embed_tokens.weight` → `embed_tokens.weight`
        //   - `model.norm.weight` → `norm.weight`
        //   - `model.layers.N.mlp.experts.e_score_correction_bias` →
        //     `layers.N.mlp.e_score_correction_bias` (drop `experts.` segment;
        //     the Python reference notes this remap explicitly)
        // Plus drops keys other vmlx model classes also drop:
        //   - `self_attn.rotary_emb.inv_freq`     (precomputed rope freqs)
        //   - `.tq_bits` per-tensor scalars       (read from config, not weights)
        //   - `lm_head.weight`                    (when tieWordEmbeddings)
        //
        // 2026-05-01: real Laguna mxtq bundles fuse the routed-expert gate
        // and up projections into a single tensor at
        // `experts.gate_up_proj.{tq_packed,tq_norms}` (gate stacked atop up
        // on the out-dim axis). `TurboQuantSwitchGLU` expects them as two
        // separate `gate_proj.*` / `up_proj.*` tensors (DeepSeek-V4 / Mistral
        // 4 / NemotronH layout). Split here in O(n_layers) — gate is the
        // first `moe_intermediate_size` rows on axis=1 and up is the
        // remainder. Same operation for `tq_norms` (one fewer axis).
        var firstPass: [String: MLXArray] = [:]
        for (key, value) in weights {
            var k = key
            if k.hasPrefix("model.") {
                k = String(k.dropFirst("model.".count))
            }
            k = k.replacingOccurrences(
                of: ".mlp.experts.e_score_correction_bias",
                with: ".mlp.e_score_correction_bias"
            )
            // Drop unused / config-only keys.
            if k.contains("self_attn.rotary_emb.inv_freq") { continue }
            if k.hasSuffix(".tq_bits") { continue }
            if cfg.tieWordEmbeddings && k == "lm_head.weight" { continue }
            firstPass[k] = value
        }

        // Second pass: split fused gate_up_proj for routed experts. Applies
        // to JANGTQ codebook tensors and affine/mxfp4 tensors.
        let mid = cfg.moeIntermediateSize
        var out: [String: MLXArray] = [:]
        for (k, v) in firstPass {
            // Match `layers.<i>.mlp.experts.gate_up_proj.<param>` exactly —
            // the fused tensor lives only on the routed-expert MoE
            // (`experts`), never on the shared expert (which has separate
            // gate_proj / up_proj keys per the affine-quant scheme).
            if k.contains(".mlp.experts.gate_up_proj.tq_packed") {
                // shape (n_exp, 2 × mid, packed_in)
                let gateK = k.replacingOccurrences(
                    of: ".gate_up_proj.tq_packed", with: ".gate_proj.tq_packed")
                let upK = k.replacingOccurrences(
                    of: ".gate_up_proj.tq_packed", with: ".up_proj.tq_packed")
                out[gateK] = v[0..., ..<mid, 0...]
                out[upK]   = v[0..., mid..., 0...]
            } else if k.contains(".mlp.experts.gate_up_proj.tq_norms") {
                // shape (n_exp, 2 × mid)
                let gateK = k.replacingOccurrences(
                    of: ".gate_up_proj.tq_norms", with: ".gate_proj.tq_norms")
                let upK = k.replacingOccurrences(
                    of: ".gate_up_proj.tq_norms", with: ".up_proj.tq_norms")
                out[gateK] = v[0..., ..<mid]
                out[upK]   = v[0..., mid...]
            } else if k.contains(".mlp.experts.gate_up_proj.weight")
                || k.contains(".mlp.experts.gate_up_proj.scales")
                || k.contains(".mlp.experts.gate_up_proj.biases")
            {
                let suffix: String
                if k.hasSuffix(".weight") {
                    suffix = ".weight"
                } else if k.hasSuffix(".scales") {
                    suffix = ".scales"
                } else {
                    suffix = ".biases"
                }
                let gateK = k.replacingOccurrences(
                    of: ".gate_up_proj" + suffix, with: ".gate_proj" + suffix)
                let upK = k.replacingOccurrences(
                    of: ".gate_up_proj" + suffix, with: ".up_proj" + suffix)
                out[gateK] = v[0..., ..<mid, 0...]
                out[upK]   = v[0..., mid..., 0...]
            } else {
                out[k] = v
            }
        }
        return out
    }
}

// MARK: - LoRA support (empty — Laguna fine-tune isn't a target today)

extension LagunaModel: LoRAModel {
    public var loraLayers: [Module] { [] }
}
