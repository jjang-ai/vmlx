//
//  Qwen35JANGTQ.swift
//  vMLXLLM
//
//  JANGTQ (TurboQuant codebook) variant of the Qwen 3.5 / 3.6 MoE family
//  (model_type: `qwen3_5_moe`). Identical model structure to
//  `Qwen35.swift` but the routed-expert MoE projections run through
//  `TurboQuantSwitchGLU` (codebook Metal kernels) instead of
//  `SwitchGLU` (`gather_qmm`).
//
//  What differs from MiniMaxJANGTQ:
//    - Router uses softmax + topk (no `e_score_correction_bias`)
//    - Shared-expert path: `routed + sigmoid(shared_gate(x)) * shared_expert(x)`
//    - Hybrid attention: `linear_attn` (Qwen35GatedDeltaNet) on most layers,
//      full `self_attn` every `full_attention_interval` layers
//    - `attn_output_gate` not present in Qwen 3.5/3.6 MoE — only standard GQA
//
//  What stays affine (NOT TQ-quantized):
//    - Shared expert (`shared_expert.{gate,up,down}_proj`) — affine quant
//    - Linear / full attention projections — affine quant
//    - Router (`gate.weight`) — affine, dequantized at load time by JangLoader
//
//  Reuses internal classes from `Qwen35.swift`:
//    - `Qwen35GatedDeltaNet` (linear attention block)
//    - `Qwen35Attention` (full attention block)
//    - `Qwen3NextMLP` (used by shared expert)
//
//  Sanitize handles the `.tq_packed` / `.tq_norms` per-expert tensors
//  and stacks them into the `switch_mlp` layout `TurboQuantSwitchGLU`
//  expects, mirroring the MiniMaxJANGTQ pattern.
//

import Foundation
import MLX
import MLXNN
import vMLXLMCommon

/// File-local recreation of the compiled sigmoid gate used by the
/// shared-expert path in Qwen 3.5 MoE. The original lives in
/// `Qwen35.swift` as a private constant; reproducing it here keeps
/// this file standalone-buildable without exposing internals from
/// the affine path.
private let qwen35JANGTQCompiledSigmoidGate: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { gate, expert in
        sigmoid(gate) * expert
    }
    return HardwareInfo.isCompiledDecodeSupported ? compile(shapeless: true, body) : body
}()

/// Compiled router fast path — fuses `softmax → topk indices → gather scores
/// → normalize` into one MLX graph, mirroring Python `_get_compiled_router_softmax`
/// in `jang_tools.load_jangtq` (P15 optimization). Parameterized by
/// `(numExperts, k, renorm)` because compile needs both `kth = numExperts - k`
/// and the slice index as compile-time constants.
private struct Qwen35JANGTQRouterKey: Hashable {
    let numExperts: Int; let k: Int; let renorm: Bool
}
private nonisolated(unsafe) var _qwen35JANGTQRouterCache:
    [Qwen35JANGTQRouterKey: ([MLXArray]) -> [MLXArray]] = [:]
private let _qwen35JANGTQRouterLock = NSLock()

private func qwen35JANGTQCompiledRouter(numExperts: Int, k: Int, renorm: Bool)
    -> ([MLXArray]) -> [MLXArray]
{
    let key = Qwen35JANGTQRouterKey(numExperts: numExperts, k: k, renorm: renorm)
    _qwen35JANGTQRouterLock.lock(); defer { _qwen35JANGTQRouterLock.unlock() }
    if let cached = _qwen35JANGTQRouterCache[key] { return cached }
    let kth = numExperts - k
    let body: ([MLXArray]) -> [MLXArray] = { args in
        let gates = args[0]
        let scores = MLX.softmax(gates, axis: -1, precise: true)
        let inds = MLX.argPartition(scores, kth: kth, axis: -1)[.ellipsis, kth...]
        var sel = MLX.takeAlong(scores, inds, axis: -1)
        if renorm {
            sel = sel
                / (sel.sum(axis: -1, keepDims: true) + MLXArray(Float(1e-20), dtype: sel.dtype))
        }
        return [inds, sel]
    }
    // shapeless: false — output shape depends on k which is captured.
    // The graph is otherwise fixed for this (numExperts, k) tuple.
    let compiled = compile(body)
    _qwen35JANGTQRouterCache[key] = compiled
    return compiled
}


// MARK: - Configuration

public struct Qwen35JANGTQTextConfiguration: Codable, Sendable {
    // All fields from Qwen35TextConfiguration, decoded the same way.
    // We don't subclass the affine type to avoid a coupled Codable
    // contract: any future field added to the affine config would
    // silently change the JANGTQ wire format.
    public var modelType: String = ""
    public var hiddenSize: Int = 4096
    public var hiddenLayers: Int = 32
    public var intermediateSize: Int = 14336
    public var attentionHeads: Int = 32
    public var kvHeads: Int = 8
    public var linearNumValueHeads: Int = 64
    public var linearNumKeyHeads: Int = 16
    public var linearKeyHeadDim: Int = 192
    public var linearValueHeadDim: Int = 128
    public var linearConvKernelDim: Int = 4
    public var rmsNormEps: Float = 1e-6
    public var vocabularySize: Int = 151_936
    public var ropeTheta: Float = 100000.0
    public var partialRotaryFactor: Float = 0.25
    public var maxPositionEmbeddings: Int = 131072
    public var tieWordEmbeddings: Bool = false
    public var attentionBias: Bool = false
    public var headDim: Int?
    public var ropeScaling: [String: StringOrNumber]?
    public var fullAttentionInterval: Int = 4

    // MoE fields
    public var numExperts: Int = 0
    public var numExpertsPerTok: Int = 0
    public var decoderSparseStep: Int = 1
    public var sharedExpertIntermediateSize: Int = 0
    public var moeIntermediateSize: Int = 0
    public var normTopkProb: Bool = true

    // JANGTQ-specific
    public var weightFormat: String = "mxtq"
    public var mxtqBits: Int = 2
    public var mxtqSeed: Int = 42

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case fullAttentionInterval = "full_attention_interval"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case decoderSparseStep = "decoder_sparse_step"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case normTopkProb = "norm_topk_prob"
        case weightFormat = "weight_format"
        case mxtqBits = "mxtq_bits"
        case mxtqSeed = "mxtq_seed"
    }

    private enum RopeKey: String, CodingKey {
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? ""
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        self.hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14336
        self.attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        self.linearNumValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 64
        self.linearNumKeyHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
        self.linearKeyHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 192
        self.linearValueHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
        self.linearConvKernelDim =
            try container.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_936
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        self.fullAttentionInterval =
            try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4

        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        self.numExpertsPerTok =
            try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
        self.decoderSparseStep =
            try container.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
        self.sharedExpertIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
        self.moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        self.normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true

        self.weightFormat =
            try container.decodeIfPresent(String.self, forKey: .weightFormat) ?? "mxtq"
        // §346 T6 — accept `mxtq_bits` as flat Int OR per-role dict.
        // See Qwen35MoEJANGTQ for the rationale; same bundles may be
        // loaded through either the LLM-text or VLM factory path.
        //
        // §418 (2026-04-25) — fallback to top-level `quantization.bits`
        // when `mxtq_bits` is absent. Qwen3.6-35B-A3B-JANGTQ4 ships
        // `quantization.bits=4` only (no `mxtq_bits`, no
        // `routed_expert_bits`). With the previous `else 2` fallback,
        // the TurboQuant routed experts loaded with a 2-bit codebook
        // (4 entries) instead of 4-bit (16 entries), producing empty
        // output. Bundle naming convention "JANGTQ4" guarantees routed
        // bits == top-level affine bits when no dedicated routed-bits
        // field is present.
        if let flat = try? container.decodeIfPresent(Int.self, forKey: .mxtqBits) {
            self.mxtqBits = flat
        } else if let dict = try? container.decodeIfPresent(
            [String: Int].self, forKey: .mxtqBits),
            let routed = dict["routed_expert"] ?? dict["shared_expert"]
                ?? dict.values.first
        {
            self.mxtqBits = routed
        } else if let qBits = Qwen35JANGTQTextConfiguration._peekQuantizationBits(decoder) {
            self.mxtqBits = qBits
        } else {
            self.mxtqBits = 2
        }
        // §421 DIAG (2026-04-25): print resolved mxtqBits to confirm
        // config patches reach the runtime. Always-on; cheap (one print
        // per model load).
        FileHandle.standardError.write(Data(
            "[Qwen35JANGTQ] mxtqBits resolved to \(self.mxtqBits)\n".utf8))
        self.mxtqSeed = try container.decodeIfPresent(Int.self, forKey: .mxtqSeed) ?? 42

        let defaultRopeParameters: [String: StringOrNumber] = [
            "type": .string("default"),
            "mrope_section": .ints([11, 11, 10]),
            "rope_theta": .float(100000.0),
            "partial_rotary_factor": .float(0.25),
        ]
        let ropeContainer = try decoder.container(keyedBy: RopeKey.self)
        let ropeParameters = try ropeContainer.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)

        if var ropeParameters {
            if ropeParameters["type"] == nil, let ropeType = ropeParameters["rope_type"] {
                ropeParameters["type"] = ropeType
            }
            self.ropeTheta = ropeParameters["rope_theta"]?.asFloat() ?? 100000.0
            self.partialRotaryFactor =
                ropeParameters["partial_rotary_factor"]?.asFloat() ?? 0.25
            self.ropeScaling = ropeParameters
        } else {
            self.ropeTheta =
                try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 100000.0
            self.partialRotaryFactor =
                try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.25
            self.ropeScaling =
                try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
                ?? defaultRopeParameters
        }

        if self.headDim == nil {
            self.headDim = self.hiddenSize / self.attentionHeads
        }
    }

    /// §418 (2026-04-25) — peek at top-level `quantization.bits` for the
    /// `mxtq_bits` fallback. Some bundles (e.g. Qwen3.6-A3B-JANGTQ4)
    /// ship `quantization.bits=4` only, omitting the dedicated
    /// `mxtq_bits` field. Without this fallback the routed-expert
    /// TurboQuant kernel loaded with the default 2-bit codebook,
    /// producing empty output.
    private enum QuantPeekKey: String, CodingKey { case quantization }
    private struct QuantPeek: Decodable {
        let bits: Int?
        let groupSize: Int?
        enum CodingKeys: String, CodingKey {
            case bits, groupSize = "group_size"
        }
    }
    fileprivate static func _peekQuantizationBits(_ decoder: Decoder) -> Int? {
        guard let outer = try? decoder.container(keyedBy: QuantPeekKey.self),
              let q = try? outer.decodeIfPresent(QuantPeek.self, forKey: .quantization),
              let b = q.bits
        else { return nil }
        return b
    }

    /// Project this JANGTQ config into the affine-side representation
    /// the existing `Qwen35GatedDeltaNet` / `Qwen35Attention` /
    /// `Qwen3NextMLP` initialisers expect. They take a
    /// `Qwen35TextConfiguration` (no JANGTQ fields), so we fabricate
    /// one by re-encoding through JSON.
    fileprivate func asAffine() -> Qwen35TextConfiguration {
        // The affine config has the same on-wire shape minus the three
        // JANGTQ fields. Round-trip via JSON to populate it without
        // re-implementing every field default.
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        do {
            let data = try encoder.encode(JANGTQAffineProjection(self))
            return try decoder.decode(Qwen35TextConfiguration.self, from: data)
        } catch {
            // The encoder shape is exactly what Qwen35TextConfiguration
            // accepts; an exception here is a programming error, not
            // an input-data issue.
            fatalError(
                "Qwen35JANGTQTextConfiguration.asAffine encode/decode failed: \(error)")
        }
    }
}

/// Helper used by `asAffine()` — encodes the JANGTQ config without
/// the JANGTQ-specific fields so it round-trips through the affine
/// `Qwen35TextConfiguration` decoder cleanly.
private struct JANGTQAffineProjection: Encodable {
    let modelType: String
    let hiddenSize: Int
    let hiddenLayers: Int
    let intermediateSize: Int
    let attentionHeads: Int
    let kvHeads: Int
    let linearNumValueHeads: Int
    let linearNumKeyHeads: Int
    let linearKeyHeadDim: Int
    let linearValueHeadDim: Int
    let linearConvKernelDim: Int
    let rmsNormEps: Float
    let vocabularySize: Int
    let ropeTheta: Float
    let partialRotaryFactor: Float
    let maxPositionEmbeddings: Int
    let tieWordEmbeddings: Bool
    let attentionBias: Bool
    let headDim: Int?
    let fullAttentionInterval: Int
    let numExperts: Int
    let numExpertsPerTok: Int
    let decoderSparseStep: Int
    let sharedExpertIntermediateSize: Int
    let moeIntermediateSize: Int
    let normTopkProb: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case headDim = "head_dim"
        case fullAttentionInterval = "full_attention_interval"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case decoderSparseStep = "decoder_sparse_step"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case normTopkProb = "norm_topk_prob"
    }

    init(_ src: Qwen35JANGTQTextConfiguration) {
        self.modelType = src.modelType
        self.hiddenSize = src.hiddenSize
        self.hiddenLayers = src.hiddenLayers
        self.intermediateSize = src.intermediateSize
        self.attentionHeads = src.attentionHeads
        self.kvHeads = src.kvHeads
        self.linearNumValueHeads = src.linearNumValueHeads
        self.linearNumKeyHeads = src.linearNumKeyHeads
        self.linearKeyHeadDim = src.linearKeyHeadDim
        self.linearValueHeadDim = src.linearValueHeadDim
        self.linearConvKernelDim = src.linearConvKernelDim
        self.rmsNormEps = src.rmsNormEps
        self.vocabularySize = src.vocabularySize
        self.ropeTheta = src.ropeTheta
        self.partialRotaryFactor = src.partialRotaryFactor
        self.maxPositionEmbeddings = src.maxPositionEmbeddings
        self.tieWordEmbeddings = src.tieWordEmbeddings
        self.attentionBias = src.attentionBias
        self.headDim = src.headDim
        self.fullAttentionInterval = src.fullAttentionInterval
        self.numExperts = src.numExperts
        self.numExpertsPerTok = src.numExpertsPerTok
        self.decoderSparseStep = src.decoderSparseStep
        self.sharedExpertIntermediateSize = src.sharedExpertIntermediateSize
        self.moeIntermediateSize = src.moeIntermediateSize
        self.normTopkProb = src.normTopkProb
    }
}

/// Top-level config matching the on-wire shape of the affine
/// `Qwen35Configuration` (model_type at root, optional `text_config`
/// nested or flattened).
public struct Qwen35JANGTQConfiguration: Codable, Sendable {
    public var modelType: String
    public var textConfig: Qwen35JANGTQTextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decode(String.self, forKey: .modelType)
        if let nested = try container.decodeIfPresent(
            Qwen35JANGTQTextConfiguration.self, forKey: .textConfig)
        {
            self.textConfig = nested
        } else {
            self.textConfig = try Qwen35JANGTQTextConfiguration(from: decoder)
        }
    }
}

// MARK: - JANGTQ MoE block

/// Sparse MoE block for Qwen 3.5 / 3.6, JANGTQ variant. Mirrors
/// `Qwen35SparseMoeBlock` (softmax + topk + shared expert) but routes
/// the per-expert projections through `TurboQuantSwitchGLU` so the
/// codebook Metal kernels run instead of `gather_qmm`.
final class Qwen35JANGTQSparseMoeBlock: Module, UnaryLayer {
    let normTopkProb: Bool
    let numExperts: Int
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: TurboQuantSwitchGLU

    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen3NextMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ args: Qwen35JANGTQTextConfiguration) {
        self.normTopkProb = args.normTopkProb
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerTok

        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _switchMLP.wrappedValue = TurboQuantSwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts,
            bits: args.mxtqBits,
            seed: args.mxtqSeed
        )

        _sharedExpert.wrappedValue = Qwen3NextMLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.sharedExpertIntermediateSize
        )
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gates = gate(x)
        // P15-equivalent: route the softmax → topk → gather → normalize chain
        // through a compiled MLX graph. Mirrors Python's
        // `_get_compiled_router_softmax(k, renorm)` from `load_jangtq.py:399`.
        // Saves ~3 dispatches per layer × 40 layers per token vs the
        // previously inlined chain.
        let routed = qwen35JANGTQCompiledRouter(
            numExperts: numExperts, k: topK, renorm: normTopkProb
        )([gates])
        let inds = routed[0]
        let scores = routed[1]

        let y = switchMLP(x, inds)
        let combined = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)

        let sharedY = sharedExpert(x)
        let gatedSharedY = qwen35JANGTQCompiledSigmoidGate(sharedExpertGate(x), sharedY)

        return combined + gatedSharedY
    }
}

// MARK: - JANGTQ Decoder Layer (mirrors Qwen35DecoderLayer)

final class Qwen35JANGTQDecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention?
    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen35GatedDeltaNet?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ args: Qwen35JANGTQTextConfiguration, layerIdx: Int) {
        self.isLinear = (layerIdx + 1) % args.fullAttentionInterval != 0

        // The internal Qwen35GatedDeltaNet / Qwen35Attention live in
        // the same module and accept Qwen35TextConfiguration. We
        // project our JANGTQ config down to that shape via JSON
        // round-trip — see Qwen35JANGTQTextConfiguration.asAffine().
        let affine = args.asAffine()
        if isLinear {
            _linearAttn.wrappedValue = Qwen35GatedDeltaNet(affine)
        } else {
            _selfAttn.wrappedValue = Qwen35Attention(affine)
        }

        if args.numExperts > 0 {
            _mlp.wrappedValue = Qwen35JANGTQSparseMoeBlock(args)
        } else {
            _mlp.wrappedValue = Qwen3NextMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let r: MLXArray
        if isLinear {
            r = linearAttn!(inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache)
        } else {
            r = selfAttn!(inputLayerNorm(x), mask: attentionMask, cache: cache)
        }
        let h = x + r
        return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
    }
}

// MARK: - Inner text model

public class Qwen35JANGTQTextModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen35JANGTQDecoderLayer]
    let norm: RMSNorm

    let ssmIdx: Int
    let faIdx: Int

    init(_ args: Qwen35JANGTQTextConfiguration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )
        self.layers = (0 ..< args.hiddenLayers).map { layerIdx in
            Qwen35JANGTQDecoderLayer(args, layerIdx: layerIdx)
        }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        self.ssmIdx = 0
        self.faIdx = args.fullAttentionInterval - 1

        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache?]? = nil) -> MLXArray {
        var hiddenStates = embedTokens(inputs)

        var cacheArray = cache
        if cacheArray == nil {
            cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        }

        let faMask = createAttentionMask(h: hiddenStates, cache: cacheArray?[faIdx])
        let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? MambaCache)

        for (i, layer) in layers.enumerated() {
            let mask = layer.isLinear ? ssmMask : nil
            let attnMask =
                layer.isLinear
                ? MLXFast.ScaledDotProductAttentionMaskMode.none : faMask
            hiddenStates = layer(
                hiddenStates,
                attentionMask: attnMask,
                ssmMask: mask,
                cache: cacheArray?[i]
            )
        }
        return norm(hiddenStates)
    }
}

// MARK: - Text model wrapper

public class Qwen35JANGTQTextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen35JANGTQTextModelInner
    let configuration: Qwen35JANGTQTextConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Qwen35JANGTQTextConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen35JANGTQTextModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        return model.layers.map { layer in
            if layer.isLinear { return MambaCache() }
            return KVCacheSimple()
        }
    }

    /// Sanitize for the JANGTQ wire format. Three jobs:
    ///   1. Strip MTP heads + apply the (1+norm) shift convention from
    ///      the affine sanitize (matches `Qwen35TextModel.sanitize`).
    ///   2. Drop `.tq_bits` metadata tensors — these are per-tensor
    ///      bit-width hints, not module parameters.
    ///   3. Stack per-expert `experts.{E}.{w1,w2,w3}.{tq_packed,tq_norms}`
    ///      tensors into the 3D layout `TurboQuantSwitchGLU` expects
    ///      under `mlp.switch_mlp.{gate_proj,up_proj,down_proj}.{tq_packed,tq_norms}`.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights

        let hasMTP = weights.keys.contains { $0.contains("mtp.") }
        let hasUnsanitizedConv1d = weights.contains { key, value in
            key.contains("conv1d.weight") && value.dim(-1) != 1
        }
        let shouldShiftNorms = hasMTP || hasUnsanitizedConv1d

        weights = weights.filter { !$0.key.contains("mtp.") }
        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        // tq_bits metadata is stored as a tensor by the converter for
        // sidecar bookkeeping but doesn't bind to a module parameter.
        for key in Array(weights.keys) where key.hasSuffix(".tq_bits") {
            weights[key] = nil
        }

        // Apply the (1+w) norm shift if needed (matches affine path).
        let normSuffixes = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ]
        for k in Array(weights.keys) {
            guard let v = weights[k] else { continue }
            if k.contains("conv1d.weight") && v.dim(-1) != 1 {
                weights[k] = v.movedAxis(source: 2, destination: 1)
                continue
            }
            if shouldShiftNorms
                && normSuffixes.contains(where: { k.hasSuffix($0) })
                && v.ndim == 1
            {
                weights[k] = v + MLXArray(1, dtype: v.dtype)
            }
        }

        // Stack per-expert tq_packed / tq_norms into switch_mlp layout.
        // The wire format uses w1/w2/w3 (mirrors mlx-lm Python); the
        // module field names are gate_proj/up_proj/down_proj.
        let renames: [(String, String)] = [
            ("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj"),
        ]
        let probe = "model.layers.0.mlp.experts.0.w1.tq_packed"
        let needsStack = weights[probe] != nil
        if needsStack {
            for layer in 0 ..< configuration.hiddenLayers {
                let prefix = "model.layers.\(layer).mlp"
                for (orig, updated) in renames {
                    for kind in ["tq_packed", "tq_norms"] {
                        let first = "\(prefix).experts.0.\(orig).\(kind)"
                        guard weights[first] != nil else { continue }
                        let stacked: [MLXArray] = (0 ..< configuration.numExperts).map { e in
                            weights.removeValue(
                                forKey: "\(prefix).experts.\(e).\(orig).\(kind)")!
                        }
                        weights["\(prefix).switch_mlp.\(updated).\(kind)"] = MLX.stacked(stacked)
                    }
                }
            }
        }

        return weights
    }
}

extension Qwen35JANGTQTextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Top-level model wrapper

public class Qwen35JANGTQModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: Qwen35JANGTQTextModel

    public init(_ args: Qwen35JANGTQConfiguration) {
        let textModel = Qwen35JANGTQTextModel(args.textConfig)
        self.vocabularySize = textModel.vocabularySize
        self.kvHeads = textModel.kvHeads
        _languageModel.wrappedValue = textModel
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    /// Top-level sanitize: strip vision tower keys, normalise the
    /// `model.language_model.*` → `language_model.model.*` prefix, then
    /// hand off to the text model's sanitize (which handles tq_packed
    /// stacking and the norm shift).
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix("vision_tower") || key.hasPrefix("model.visual") {
                continue
            }
            var key = key
            if key.hasPrefix("model.language_model") {
                key = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if !key.hasPrefix("language_model.") {
                key = "language_model." + key
            }
            sanitized[key] = value
        }
        return languageModel.sanitize(weights: sanitized)
    }
}

extension Qwen35JANGTQModel: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}
