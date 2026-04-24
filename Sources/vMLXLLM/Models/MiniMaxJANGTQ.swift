//
//  MiniMaxJANGTQ.swift
//  vMLXLLM
//
//  JANGTQ (TurboQuant codebook) variant of MiniMax — identical model
//  structure, but swaps `SwitchGLU` → `TurboQuantSwitchGLU` so the MoE
//  projections run the JANGTQ codebook Metal kernels instead of
//  `gather_qmm`. Attention / RMSNorm / RoPE / SDPA are unchanged — they
//  already call the same `mx.fast.*` C++ entry points Python uses.
//
//  Created by Jinho Jang (eric@jangq.ai).
//

import Foundation
import MLX
import MLXNN
import vMLXLMCommon

// MARK: - Compiled router fast path (Phase S1 speedup, 2026-04-24)
//
// Fuses sigmoid-route → bias-bias → topk → renorm into one MLX graph,
// gated by `HardwareInfo.isCompiledDecodeSupported`. Mirrors the Python
// `_get_compiled_router_sigmoid` helper in jang_tools.load_jangtq.

private struct MiniMaxJANGTQRouterKey: Hashable {
    let numExperts: Int; let k: Int
}
nonisolated(unsafe) private var _miniMaxJANGTQRouterCache:
    [MiniMaxJANGTQRouterKey: ([MLXArray]) -> [MLXArray]] = [:]
private let _miniMaxJANGTQRouterLock = NSLock()

private func miniMaxJANGTQCompiledRouter(numExperts: Int, k: Int)
    -> ([MLXArray]) -> [MLXArray]
{
    let key = MiniMaxJANGTQRouterKey(numExperts: numExperts, k: k)
    _miniMaxJANGTQRouterLock.lock(); defer { _miniMaxJANGTQRouterLock.unlock() }
    if let cached = _miniMaxJANGTQRouterCache[key] { return cached }
    let kth = k - 1
    let body: @Sendable ([MLXArray]) -> [MLXArray] = { args in
        let gates = args[0]; let bias = args[1]
        let scoresOrig = sigmoid(gates)
        let biased = scoresOrig + bias
        let inds = argPartition(-biased, kth: kth, axis: -1)[.ellipsis, ..<k]
        var sel = takeAlong(scoresOrig, inds, axis: -1)
        sel = sel / (sel.sum(axis: -1, keepDims: true) + MLXArray(Float(1e-20), dtype: sel.dtype))
        return [inds, sel]
    }
    let compiled: ([MLXArray]) -> [MLXArray]
    if HardwareInfo.isCompiledDecodeSupported {
        compiled = compile(shapeless: true, body)
    } else {
        compiled = body
    }
    _miniMaxJANGTQRouterCache[key] = compiled
    return compiled
}

// MARK: - Attention (identical to MiniMax.swift)

private class MiniMaxJANGTQAttention: Module {
    let args: MiniMaxJANGTQConfiguration
    let scale: Float
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    let rope: RoPE

    init(_ args: MiniMaxJANGTQConfiguration) {
        self.args = args
        self.numAttentionHeads = args.attentionHeads
        self.numKeyValueHeads = args.kvHeads
        self.headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(args.hiddenSize, numAttentionHeads * headDim, bias: false)
        _wk.wrappedValue = Linear(args.hiddenSize, numKeyValueHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(args.hiddenSize, numKeyValueHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(numAttentionHeads * headDim, args.hiddenSize, bias: false)

        if args.useQkNorm {
            _qNorm.wrappedValue = RMSNorm(
                dimensions: numAttentionHeads * headDim, eps: args.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(
                dimensions: numKeyValueHeads * headDim, eps: args.rmsNormEps)
        }

        self.rope = RoPE(
            dimensions: args.rotaryDim,
            traditional: false,
            base: args.ropeTheta
        )
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        let values = wv(x)

        if let qNorm, let kNorm {
            queries = qNorm(queries)
            keys = kNorm(keys)
        }

        var q = queries.reshaped(B, L, numAttentionHeads, -1).transposed(0, 2, 1, 3)
        var k = keys.reshaped(B, L, numKeyValueHeads, -1).transposed(0, 2, 1, 3)
        let v = values.reshaped(B, L, numKeyValueHeads, -1).transposed(0, 2, 1, 3)

        q = applyRotaryPosition(rope, to: q, cache: cache)
        k = applyRotaryPosition(rope, to: k, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: q, keys: k, values: v,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

// MARK: - MoE block (JANGTQ — swaps SwitchGLU for TurboQuantSwitchGLU)

private class MiniMaxJANGTQSparseMoeBlock: Module {
    let numExpertsPerTok: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: TurboQuantSwitchGLU
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ args: MiniMaxJANGTQConfiguration) {
        self.numExpertsPerTok = args.numExpertsPerTok

        _gate.wrappedValue = Linear(args.hiddenSize, args.numLocalExperts, bias: false)
        _switchMLP.wrappedValue = TurboQuantSwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.intermediateSize,
            numExperts: args.numLocalExperts,
            bits: args.mxtqBits,
            seed: args.mxtqSeed
        )
        _eScoreCorrectionBias.wrappedValue = MLXArray.zeros([args.numLocalExperts])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gates = gate(x)
        // Compiled router: fuses sigmoid + bias + topk + renorm into one graph.
        let router = miniMaxJANGTQCompiledRouter(
            numExperts: gates.dim(-1), k: numExpertsPerTok
        )
        let routed = router([gates, eScoreCorrectionBias])
        let inds = routed[0]
        let scores = routed[1].asType(x.dtype)

        let y = switchMLP(x, inds)
        return (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
    }
}

// MARK: - Decoder layer

private class MiniMaxJANGTQDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MiniMaxJANGTQAttention
    @ModuleInfo(key: "block_sparse_moe") var blockSparseMoe: MiniMaxJANGTQSparseMoeBlock
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: MiniMaxJANGTQConfiguration) {
        _selfAttn.wrappedValue = MiniMaxJANGTQAttention(args)
        _blockSparseMoe.wrappedValue = MiniMaxJANGTQSparseMoeBlock(args)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var hidden = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        hidden = hidden + blockSparseMoe(postAttentionLayerNorm(hidden))
        return hidden
    }
}

// MARK: - Inner model

public class MiniMaxJANGTQModelInner: Module {
    let args: MiniMaxJANGTQConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate let layers: [MiniMaxJANGTQDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ args: MiniMaxJANGTQConfiguration) {
        self.args = args
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { _ in MiniMaxJANGTQDecoderLayer(args) }
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

// MARK: - Top-level model

public class MiniMaxJANGTQModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: MiniMaxJANGTQModelInner
    let configuration: MiniMaxJANGTQConfiguration
    let modelType: String

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: MiniMaxJANGTQConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.modelType = args.modelType
        self.model = MiniMaxJANGTQModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    /// Stacks per-expert JANGTQ tensors into the `switch_mlp` layout expected
    /// by `TurboQuantSwitchGLU`. Python writer uses `w1`/`w2`/`w3` tensor
    /// names (mirrors `MiniMax.swift` sanitize for the affine path). Also
    /// strips `.tq_bits` metadata tensors — they're not module parameters.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights

        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }

        // Drop tq_bits metadata tensors anywhere in the tree.
        for key in sanitized.keys where key.hasSuffix(".tq_bits") {
            sanitized[key] = nil
        }

        let probe = "model.layers.0.block_sparse_moe.experts.0.w1.tq_packed"
        guard sanitized[probe] != nil else { return sanitized }

        let renames: [(String, String)] = [
            ("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")
        ]
        for layer in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layer).block_sparse_moe"
            for (orig, updated) in renames {
                for key in ["tq_packed", "tq_norms"] {
                    let first = "\(prefix).experts.0.\(orig).\(key)"
                    guard sanitized[first] != nil else { continue }
                    let stacked = (0 ..< configuration.numLocalExperts).map { e -> MLXArray in
                        sanitized.removeValue(
                            forKey: "\(prefix).experts.\(e).\(orig).\(key)")!
                    }
                    sanitized["\(prefix).switch_mlp.\(updated).\(key)"] = MLX.stacked(stacked)
                }
            }
        }

        return sanitized
    }
}

// MARK: - Configuration

public struct MiniMaxJANGTQConfiguration: Codable, Sendable {
    public var modelType: String = "minimax_m2"
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var maxPositionEmbeddings: Int
    public var numExpertsPerTok: Int
    public var numLocalExperts: Int
    public var sharedIntermediateSize: Int
    public var hiddenLayers: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var rotaryDim: Int
    public var vocabularySize: Int
    public var tieWordEmbeddings: Bool = false
    public var scoringFunc: String = "sigmoid"
    public var headDim: Int?
    public var useQkNorm: Bool = true

    // JANGTQ-specific
    public var weightFormat: String = "mxtq"
    public var mxtqBits: Int = 2
    public var mxtqSeed: Int = 42

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case numExpertsPerTok = "num_experts_per_tok"
        case numLocalExperts = "num_local_experts"
        case sharedIntermediateSize = "shared_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case rotaryDim = "rotary_dim"
        case vocabularySize = "vocab_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case scoringFunc = "scoring_func"
        case headDim = "head_dim"
        case useQkNorm = "use_qk_norm"
        case weightFormat = "weight_format"
        case mxtqBits = "mxtq_bits"
        case mxtqSeed = "mxtq_seed"
    }

    // §346 T6 — custom decode so `mxtq_bits` accepts either a flat Int
    // (legacy) or a per-role dict `{routed_expert: N, shared_expert: N,
    // ...}` (modern converter output). Default Codable synthesis rejects
    // the dict form and would fail the whole config decode for a future
    // JANGTQ4 bundle. Falls back to 2 only when both shapes are absent.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "minimax_m2"
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.maxPositionEmbeddings =
            try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.numExpertsPerTok = try container.decode(Int.self, forKey: .numExpertsPerTok)
        self.numLocalExperts = try container.decode(Int.self, forKey: .numLocalExperts)
        self.sharedIntermediateSize =
            try container.decode(Int.self, forKey: .sharedIntermediateSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        self.rotaryDim = try container.decode(Int.self, forKey: .rotaryDim)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.scoringFunc =
            try container.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "sigmoid"
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        self.useQkNorm =
            try container.decodeIfPresent(Bool.self, forKey: .useQkNorm) ?? true
        self.weightFormat =
            try container.decodeIfPresent(String.self, forKey: .weightFormat) ?? "mxtq"
        if let flat = try? container.decodeIfPresent(Int.self, forKey: .mxtqBits) {
            self.mxtqBits = flat
        } else if let dict = try? container.decodeIfPresent(
            [String: Int].self, forKey: .mxtqBits),
            let routed = dict["routed_expert"] ?? dict["shared_expert"]
                ?? dict.values.first
        {
            self.mxtqBits = routed
        } else {
            self.mxtqBits = 2
        }
        self.mxtqSeed = try container.decodeIfPresent(Int.self, forKey: .mxtqSeed) ?? 42
    }
}

// MARK: - LoRA

extension MiniMaxJANGTQModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
