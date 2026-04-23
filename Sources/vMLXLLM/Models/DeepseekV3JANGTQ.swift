//
//  DeepseekV3JANGTQ.swift
//  vMLXLLM
//
//  JANGTQ (TurboQuant codebook) variant of DeepSeek V3 — covers model_type
//  deepseek_v3, deepseek_v2, deepseek_v32, and kimi_k25 since they all
//  share the DeepSeek V3 MLA + MoE architecture. Structure is identical
//  to `DeepseekV3.swift`; the only difference is the routed-expert MoE
//  block swaps `SwitchGLU` → `TurboQuantSwitchGLU` so the JANGTQ Metal
//  codebook kernels run instead of `gather_qmm`. Shared experts and the
//  dense `first_k_dense_replace` layers stay affine — JANGTQ quantizes
//  only the routed experts.
//
//  Unblocks native Kimi K2.6-REAP-30-JANGTQ_1L load (191 GB bundle) and
//  any future DeepSeek V3 / V3.2 JANGTQ checkpoints without the Path-A
//  conversion to affine-2-bit (+42 GB expansion).
//
//  Block K §318. Mirrors `MiniMaxJANGTQ.swift`, `Qwen35JANGTQ.swift`,
//  `GLM4MoEJANGTQ.swift`. Reuses the internal (module-visible) classes
//  `DeepseekV3Attention`, `DeepseekV3MLP`, `MoEGate` from DeepseekV3.swift
//  so this file stays small (~280 LOC vs a full 800+ duplicate).
//
//  Created by Jinho Jang (eric@jangq.ai).
//

import Foundation
import MLX
import MLXNN
import vMLXLMCommon

// MARK: - Configuration

/// Config parallel to `DeepseekV3Configuration` plus the three JANGTQ
/// fields. Decoded independently (Codable; Swift structs don't inherit)
/// but exposes `asDeepseekV3()` so the internal attention/MLP/gate
/// classes — which take `DeepseekV3Configuration` directly — can be
/// reused verbatim from DeepseekV3.swift.
public struct DeepseekV3JANGTQConfiguration: Codable, Sendable {
    public var modelType: String = "deepseek_v3"
    public var vocabSize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var moeIntermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var nSharedExperts: Int?
    public var nRoutedExperts: Int?
    public var routedScalingFactor: Float
    public var kvLoraRank: Int
    public var qLoraRank: Int
    public var qkRopeHeadDim: Int
    public var vHeadDim: Int
    public var qkNopeHeadDim: Int
    public var normTopkProb: Bool
    public var nGroup: Int?
    public var topkGroup: Int?
    public var numExpertsPerTok: Int?
    public var moeLayerFreq: Int
    public var firstKDenseReplace: Int
    public var maxPositionEmbeddings: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var ropeScaling: [String: StringOrNumber]?
    public var attentionBias: Bool
    public var indexHeadDim: Int?
    public var indexNHeads: Int?
    public var indexTopk: Int?

    // JANGTQ-specific
    public var weightFormat: String = "mxtq"
    public var mxtqBits: Int = 2
    public var mxtqSeed: Int = 42

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
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeLayerFreq = "moe_layer_freq"
        case firstKDenseReplace = "first_k_dense_replace"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
        case indexHeadDim = "index_head_dim"
        case indexNHeads = "index_n_heads"
        case indexTopk = "index_topk"
        case weightFormat = "weight_format"
        case mxtqBits = "mxtq_bits"
        case mxtqSeed = "mxtq_seed"
    }

    /// Produce a `DeepseekV3Configuration` mirror so the internal
    /// attention / MLP / gate classes (which take the concrete type)
    /// can be reused without duplication.
    func asDeepseekV3() -> DeepseekV3Configuration {
        let raw = try! JSONEncoder().encode(self)
        // Safe: the two structs share every Codable key except the
        // JANGTQ extras, which DeepseekV3Configuration decoder ignores.
        return try! JSONDecoder().decode(DeepseekV3Configuration.self, from: raw)
    }
}

// MARK: - MoE block (JANGTQ — swaps SwitchGLU for TurboQuantSwitchGLU)

/// JANGTQ variant of `DeepseekV3MoE`. The gate and shared_experts paths
/// are identical to the affine MoE; only the routed-expert projection
/// swaps to `TurboQuantSwitchGLU`. Router scores are multiplied in and
/// summed exactly as in DeepseekV3MoE so decode math matches the
/// reference Python `load_jangtq` fast path bit-for-bit.
final class DeepseekV3JANGTQMoE: Module, UnaryLayer {
    let numExpertsPerTok: Int

    @ModuleInfo(key: "switch_mlp") var switchMLP: TurboQuantSwitchGLU
    var gate: MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV3MLP?

    init(config: DeepseekV3JANGTQConfiguration) {
        self.numExpertsPerTok = config.numExpertsPerTok ?? 1
        let v3 = config.asDeepseekV3()

        self._switchMLP.wrappedValue = TurboQuantSwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts ?? 1,
            bits: config.mxtqBits,
            seed: config.mxtqSeed
        )

        self.gate = MoEGate(config: v3)

        if let sharedExpertCount = config.nSharedExperts {
            let intermediateSize = config.moeIntermediateSize * sharedExpertCount
            self._sharedExperts.wrappedValue = DeepseekV3MLP(
                config: v3, intermediateSize: intermediateSize)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        var y = switchMLP(x, indices)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
        if let shared = sharedExperts {
            y = y + shared(x)
        }
        return y
    }
}

// MARK: - Decoder layer (JANGTQ)

final class DeepseekV3JANGTQDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DeepseekV3Attention
    var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(config: DeepseekV3JANGTQConfiguration, layerIdx: Int) {
        let v3 = config.asDeepseekV3()
        self._selfAttn.wrappedValue = DeepseekV3Attention(config: v3)

        // Dense layers (firstKDenseReplace) stay affine — JANGTQ only
        // quantizes routed experts. Same rule as the Python loader.
        if config.nRoutedExperts != nil,
           layerIdx >= config.firstKDenseReplace,
           layerIdx % config.moeLayerFreq == 0
        {
            self.mlp = DeepseekV3JANGTQMoE(config: config)
        } else {
            self.mlp = DeepseekV3MLP(config: v3)
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        let r2 = mlp(postAttentionLayerNorm(h))
        return h + r2
    }
}

// MARK: - Inner model

public class DeepseekV3JANGTQModelInner: Module {
    let config: DeepseekV3JANGTQConfiguration
    let vocabSize: Int
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate let layers: [DeepseekV3JANGTQDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(config: DeepseekV3JANGTQConfiguration) {
        self.config = config
        self.vocabSize = config.vocabSize
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map {
            DeepseekV3JANGTQDecoderLayer(config: config, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(x)
        let attentionMask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: attentionMask, cache: cache?[i])
        }
        return norm(h)
    }
}

// MARK: - Top-level model

public class DeepseekV3JANGTQModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public var kvHeads: [Int] = []

    let config: DeepseekV3JANGTQConfiguration
    public var model: DeepseekV3JANGTQModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: DeepseekV3JANGTQConfiguration) {
        self.config = config
        self.model = DeepseekV3JANGTQModelInner(config: config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let out = model(inputs, cache: cache)
        return lmHead(out)
    }

    /// Stacks per-expert JANGTQ tensors into the 3D `switch_mlp` layout
    /// `TurboQuantSwitchGLU` consumes. Mirrors the Python jang-tools
    /// key naming: experts.{e}.w1/w2/w3 → switch_mlp.gate_proj/down_proj/up_proj.
    /// Also strips MTP head weights (layer == numHiddenLayers) — the
    /// inference model has no MTP head. Pattern copied verbatim from
    /// `DeepseekV3Model.sanitize()` / `MiniMaxJANGTQModel.sanitize()`.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights

        // Strip MTP (multi-token-prediction) layer tensors — they sit at
        // `model.layers.{numHiddenLayers}.*` on GLM-5.1 / DeepSeek V3.2
        // bundles and are not part of the inference graph.
        let mtpLayerIndex = config.numHiddenLayers
        for key in sanitized.keys {
            let parts = key.split(separator: ".")
            if parts.count >= 3 && parts[1] == "layers",
               let layerIdx = Int(parts[2]),
               layerIdx >= mtpLayerIndex {
                sanitized[key] = nil
            }
        }

        // Drop tq_bits metadata tensors (not module parameters).
        for key in sanitized.keys where key.hasSuffix(".tq_bits") {
            sanitized[key] = nil
        }

        // Drop rotary_emb.inv_freq (recomputed from rope_theta at init).
        for key in sanitized.keys where key.contains("rotary_emb.inv_freq") {
            sanitized[key] = nil
        }

        // Stack per-expert JANGTQ tensors into switch_mlp layout.
        let numExperts = config.nRoutedExperts ?? 0
        guard numExperts > 0 else { return sanitized }

        let renames: [(String, String)] = [
            ("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")
        ]

        for layer in 0 ..< config.numHiddenLayers {
            let prefix = "model.layers.\(layer).mlp"
            // Dense layers (< firstKDenseReplace) have no experts — skip.
            let probe = "\(prefix).experts.0.w1.tq_packed"
            guard sanitized[probe] != nil else { continue }

            for (orig, updated) in renames {
                for suffix in ["tq_packed", "tq_norms"] {
                    let firstKey = "\(prefix).experts.0.\(orig).\(suffix)"
                    guard sanitized[firstKey] != nil else { continue }
                    let stacked = (0 ..< numExperts).compactMap { e -> MLXArray? in
                        sanitized.removeValue(
                            forKey: "\(prefix).experts.\(e).\(orig).\(suffix)")
                    }
                    guard stacked.count == numExperts else { continue }
                    sanitized["\(prefix).switch_mlp.\(updated).\(suffix)"] =
                        MLX.stacked(stacked)
                }
            }
        }

        return sanitized
    }

    public var loraLayers: [Module] {
        model.layers
    }
}
