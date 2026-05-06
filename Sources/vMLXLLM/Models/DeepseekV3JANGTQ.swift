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
    fileprivate let layerIdx: Int

    @ModuleInfo(key: "switch_mlp") var switchMLP: TurboQuantSwitchGLU
    var gate: MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV3MLP?

    init(config: DeepseekV3JANGTQConfiguration, layerIdx: Int) {
        self.numExpertsPerTok = config.numExpertsPerTok ?? 1
        self.layerIdx = layerIdx
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
        JangPressRouteTelemetry.recordTopK(layer: layerIdx, indices: indices)
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
            self.mlp = DeepseekV3JANGTQMoE(config: config, layerIdx: layerIdx)
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

    private static var layerEvalEnabled: Bool {
        let raw = ProcessInfo.processInfo.environment["VMLX_JANGTQ_LAYER_EVAL"]?
            .lowercased()
        return raw == "1" || raw == "true" || raw == "on" || raw == "yes"
    }

    private static var layerReclaimEnabled: Bool {
        let raw = ProcessInfo.processInfo.environment["VMLX_JANGTQ_LAYER_RECLAIM"]?
            .lowercased()
        if raw == "0" || raw == "false" || raw == "off" || raw == "no" {
            return false
        }
        return true
    }

    private static var layerEvalStride: Int {
        guard let raw = ProcessInfo.processInfo.environment["VMLX_JANGTQ_LAYER_EVAL_STRIDE"],
              let parsed = Int(raw), parsed > 0
        else { return 1 }
        return parsed
    }

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
            if Self.layerEvalEnabled,
               i < layers.count - 1,
               (i + 1) % Self.layerEvalStride == 0
            {
                MLX.eval(h)
                if Self.layerReclaimEnabled,
                   let nRoutedExperts = config.nRoutedExperts,
                   i >= config.firstKDenseReplace,
                   i % config.moeLayerFreq == 0
                {
                    let pairs = (0 ..< nRoutedExperts).map {
                        (layer: i, expert: $0)
                    }
                    _ = JangPressCanonicalMmapAdvisor.adviseExperts(
                        .dontNeed,
                        pairs: pairs)
                }
            }
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
    private var didWarmupRuntime = false

    public init(_ config: DeepseekV3JANGTQConfiguration) {
        self.config = config
        self.kvHeads = Array(repeating: config.numAttentionHeads, count: config.numHiddenLayers)
        self.model = DeepseekV3JANGTQModelInner(config: config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let out = model(inputs, cache: cache)
        return lmHead(out)
    }

    private static func envFlag(_ name: String, default defaultValue: Bool) -> Bool {
        guard let raw = ProcessInfo.processInfo.environment[name]?.lowercased() else {
            return defaultValue
        }
        if ["0", "false", "off", "no"].contains(raw) { return false }
        if ["1", "true", "on", "yes"].contains(raw) { return true }
        return defaultValue
    }

    private static var prefillStepSize: Int {
        guard let raw = ProcessInfo.processInfo.environment["VMLX_JANGTQ_PREFILL_STEP"],
              let parsed = Int(raw), parsed > 0
        else { return 16 }
        return parsed
    }

    private static var warmupEnabled: Bool {
        envFlag("VMLX_JANGTQ_WARMUP", default: true)
    }

    private static var synchronizePrefill: Bool {
        envFlag("VMLX_JANGTQ_PREFILL_SYNC", default: true)
    }

    private func synchronizeAndClear() {
        if Self.synchronizePrefill {
            MLX.Stream.defaultStream(.gpu).synchronize()
            MLX.Stream.defaultStream(.cpu).synchronize()
        }
        Memory.clearCache()
    }

    private func warmupRuntimeIfNeeded() {
        guard Self.warmupEnabled, !didWarmupRuntime else { return }
        didWarmupRuntime = true

        var h = MLXArray.zeros([1, 1, config.hiddenSize], dtype: .bfloat16)
        MLX.eval(h)
        synchronizeAndClear()

        for (i, layer) in model.layers.enumerated() {
            let mask = createAttentionMask(h: h, cache: Optional<KVCache>.none)
            h = layer(h, mask: mask, cache: nil)
            MLX.eval(h)
            if i < model.layers.count - 1 {
                synchronizeAndClear()
            }
        }
        h = model.norm(h)
        let logits = lmHead(h)
        MLX.eval(logits)
        synchronizeAndClear()

        let warmupN = min(Self.prefillStepSize, 16)
        let warmCache = newCache(parameters: nil)
        let tiny = MLXArray.zeros([1, warmupN], dtype: .int32)
        let warmLogits = self(tiny, cache: warmCache)
        MLX.eval(warmLogits)
        MLX.eval(warmCache)
        synchronizeAndClear()
        print("[DeepseekV3JANGTQ] runtime warmup complete (layers=\(model.layers.count), prefill=\(warmupN))")
    }

    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        warmupRuntimeIfNeeded()

        let configuredStep = Self.prefillStepSize
        let prefillStepSize = max(1, min(windowSize ?? configuredStep, configuredStep))
        let originalShape = input.text.tokens.shape
        let flat = input.text.tokens.reshaped([-1])
        var y = LMInput.Text(tokens: flat, mask: input.text.mask)

        while y.tokens.size > prefillStepSize {
            let chunk = y[.newAxis, ..<prefillStepSize]
            _ = self(chunk.tokens, cache: cache.isEmpty ? nil : cache)
            MLX.eval(cache)
            y = y[prefillStepSize...]
            synchronizeAndClear()
        }

        let tailReshaped: MLXArray
        if originalShape.count >= 2 {
            let leading = Array(originalShape.dropLast())
            tailReshaped = y.tokens.reshaped(leading + [y.tokens.size])
        } else {
            tailReshaped = y.tokens
        }
        return .tokens(LMInput.Text(tokens: tailReshaped, mask: y.mask))
    }

    /// Stacks per-expert JANGTQ tensors into the 3D `switch_mlp` layout
    /// `TurboQuantSwitchGLU` consumes. Mirrors the Python jang-tools
    /// key naming: experts.{e}.w1/w2/w3 → switch_mlp.gate_proj/down_proj/up_proj.
    /// Also strips MTP head weights (layer == numHiddenLayers) — the
    /// inference model has no MTP head. Pattern copied verbatim from
    /// `DeepseekV3Model.sanitize()` / `MiniMaxJANGTQModel.sanitize()`.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // §426 — Kimi-K2.6 ships safetensors with VL-style key prefixes:
        // 136,835 `language_model.*` (the LLM weights we want), 329
        // `vision_tower.*` (skip), 6 `mm_projector.*` (skip). Strip the
        // `language_model.` prefix; drop vision/projector entries before
        // running the existing JANGTQ sanitizer.
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)
        var droppedVL = 0
        func normalizeDeepSeekMoEKey(_ key: String) -> String {
            var k = key
            if k.hasPrefix("layers.") {
                k = "model.\(k)"
            }
            guard k.contains(".ffn.") else { return k }
            k = k.replacingOccurrences(of: ".ffn.", with: ".mlp.")
            for (old, new) in [(".w1.", ".gate_proj."),
                               (".w2.", ".down_proj."),
                               (".w3.", ".up_proj.")] {
                k = k.replacingOccurrences(of: old, with: new)
            }
            return k
        }

        for (k, v) in weights {
            if k.hasPrefix("vision_tower.") || k.hasPrefix("mm_projector.") {
                droppedVL += 1
                continue
            }
            if k.hasPrefix("language_model.") {
                sanitized[normalizeDeepSeekMoEKey(String(k.dropFirst("language_model.".count)))] = v
            } else {
                sanitized[normalizeDeepSeekMoEKey(k)] = v
            }
        }
        if droppedVL > 0 {
            print("[DeepseekV3JANGTQ] dropped \(droppedVL) vision_tower/mm_projector tensors")
        }

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

        // Two known naming conventions:
        //   • DeepSeek V3 / earlier JANGTQ: w1/w2/w3 → gate/down/up
        //   • Kimi-K2.6: already gate_proj/down_proj/up_proj (identity)
        // Detect by probing layer 1 (first MoE layer in DSV3 family).
        let candidateProbeLayer = config.numHiddenLayers > 1 ? 1 : 0
        let probePrefix = "model.layers.\(candidateProbeLayer).mlp"
        let usesKimiNames =
            sanitized["\(probePrefix).experts.0.gate_proj.tq_packed"] != nil
        let renames: [(String, String)]
        if usesKimiNames {
            renames = [
                ("gate_proj", "gate_proj"),
                ("down_proj", "down_proj"),
                ("up_proj",   "up_proj"),
            ]
        } else {
            renames = [
                ("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")
            ]
        }

        for layer in 0 ..< config.numHiddenLayers {
            let prefix = "model.layers.\(layer).mlp"
            // Dense layers (< firstKDenseReplace) have no experts — skip.
            let probeOrig = renames[0].0
            let probe = "\(prefix).experts.0.\(probeOrig).tq_packed"
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
                        loadTimeMaterializedStacked(stacked)
                }
            }
            // Drop any stray .tq_bits scalars under experts.* (already
            // removed above for the shared sweep, but Kimi has them
            // per-expert and the broad pass handled it; this guard is
            // defensive only).
        }

        return sanitized
    }

    public var loraLayers: [Module] {
        model.layers
    }
}
