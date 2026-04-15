//  Olmo3.swift
//  LLM
//
//  Created by Anthony DePasquale on 23 November 2025.
//

// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/olmo3.py

import Foundation
import MLX
import vMLXLMCommon
import MLXNN

// MARK: - Attention

class Olmo3Attention: Module {
    let args: Olmo3Configuration
    let layerIdx: Int
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ args: Olmo3Configuration, layerIdx: Int) {
        self.args = args
        self.layerIdx = layerIdx

        self.nHeads = args.attentionHeads
        self.nKVHeads = args.kvHeads
        self.headDim = args._headDimensions
        self.scale = pow(Float(headDim), -0.5)

        let dim = args.hiddenSize
        self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: args.attentionBias)
        self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: args.attentionBias)
        self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: args.attentionBias)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: args.attentionBias)

        self._qNorm.wrappedValue = RMSNorm(dimensions: nHeads * headDim, eps: args.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: nKVHeads * headDim, eps: args.rmsNormEps)

        // Different RoPE initialization based on layer type
        if args.layerTypes[layerIdx] != "full_attention" {
            self.rope = RoPE(dimensions: headDim, traditional: false, base: args.ropeTheta)
        } else {
            self.rope = initializeRope(
                dims: headDim,
                base: args.ropeTheta,
                traditional: false,
                scalingConfig: args.ropeScaling,
                maxPositionEmbeddings: args.maxPositionEmbeddings
            )
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qNorm(wq(x))
        var keys = kNorm(wk(x))
        var values = wv(x)

        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

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

// MARK: - MLP

class Olmo3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ args: Olmo3Configuration) {
        self._gate.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: false)
        self._down.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: false)
        self._up.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return down(silu(gate(x)) * up(x))
    }
}

// MARK: - Transformer Block

class Olmo3TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Olmo3Attention
    @ModuleInfo(key: "mlp") var mlp: Olmo3MLP

    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm

    init(_ args: Olmo3Configuration, layerIdx: Int) {
        self._attention.wrappedValue = Olmo3Attention(args, layerIdx: layerIdx)
        self._mlp.wrappedValue = Olmo3MLP(args)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = postAttentionLayerNorm(attention(x, mask: mask, cache: cache))
        let h = x + r
        r = postFeedforwardLayerNorm(mlp(h))
        let out = h + r
        return out
    }
}

// MARK: - Model

public class Olmo3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [Olmo3TransformerBlock]
    let norm: RMSNorm
    let slidingWindow: Int
    let layerTypes: [String]
    let swaIdx: Int
    let gaIdx: Int

    init(_ args: Olmo3Configuration) {
        precondition(args.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers).map { i in
            Olmo3TransformerBlock(args, layerIdx: i)
        }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self.slidingWindow = args.slidingWindow
        self.layerTypes = args.layerTypes

        // Find first occurrence of each type
        self.swaIdx = args.layerTypes.firstIndex(of: "sliding_attention") ?? 0
        self.gaIdx = args.layerTypes.firstIndex(of: "full_attention") ?? 0
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let fullMask = createAttentionMask(h: h, cache: cache?[gaIdx])
        let slidingWindowMask = createAttentionMask(
            h: h, cache: cache?[swaIdx], windowSize: slidingWindow)

        for (i, layer) in layers.enumerated() {
            let mask = layerTypes[i] == "full_attention" ? fullMask : slidingWindowMask
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class Olmo3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Olmo3ModelInner
    let args: Olmo3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Olmo3Configuration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.args = args
        self.model = Olmo3ModelInner(args)
        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Remove unused precomputed rotary frequencies
        weights.filter { !$0.key.contains("self_attn.rotary_emb.inv_freq") }
    }

    public func newCache(parameters: GenerateParameters) -> [KVCache] {
        var caches: [KVCache] = []
        for layerType in args.layerTypes {
            if layerType == "full_attention" {
                caches.append(KVCacheSimple())
            } else {
                caches.append(RotatingKVCache(maxSize: args.slidingWindow))
            }
        }
        return caches
    }
}

// MARK: - Configuration

public struct Olmo3Configuration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var headDimensions: Int?
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var maxPositionEmbeddings: Int
    var slidingWindow: Int
    var ropeTheta: Float = 10_000
    var attentionBias: Bool = false
    var layerTypes: [String]
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool = false

    var _headDimensions: Int { headDimensions ?? (hiddenSize / attentionHeads) }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDimensions = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case ropeTheta = "rope_theta"
        case attentionBias = "attention_bias"
        case layerTypes = "layer_types"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        headDimensions = try container.decodeIfPresent(Int.self, forKey: .headDimensions)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        slidingWindow = try container.decode(Int.self, forKey: .slidingWindow)

        let maybeKV = try container.decodeIfPresent(Int.self, forKey: .kvHeads)
        kvHeads = maybeKV ?? attentionHeads

        if let ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) {
            self.ropeTheta = ropeTheta
        }
        if let attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) {
            self.attentionBias = attentionBias
        }

        // Decode layer_types or generate default
        if let layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes) {
            self.layerTypes = layerTypes
        } else {
            // Generate default layer types: full attention every 4th layer
            self.layerTypes = (0 ..< hiddenLayers).map { i in
                (i + 1) % 4 == 0 ? "full_attention" : "sliding_attention"
            }
        }

        ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)

        if let tieWordEmbeddings = try container.decodeIfPresent(
            Bool.self, forKey: .tieWordEmbeddings)
        {
            self.tieWordEmbeddings = tieWordEmbeddings
        }
    }
}

// MARK: - LoRA

extension Olmo3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
