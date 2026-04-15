//
//  Exaone4.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2025/7/15.
//

import Foundation
import MLX
import vMLXLMCommon
import MLXNN

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/exaone4.py

class Exaone4Attention: Module {
    let args: Exaone4Configuration
    let scale: Float
    let isLocal: Bool
    let useRope: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer?

    public init(_ args: Exaone4Configuration, isLocal: Bool?) {
        self.args = args
        self.isLocal = isLocal ?? false
        self.useRope = isLocal == nil || (isLocal ?? false)

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, dim, bias: false)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        if useRope {
            self.rope = initializeRope(
                dims: headDim, base: args.ropeTheta,
                traditional: false, scalingConfig: args.ropeScaling,
                maxPositionEmbeddings: args.maxPositionEmbeddings)
        } else {
            self.rope = nil
        }
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        if useRope, let rope {
            queries = applyRotaryPosition(rope, to: queries, cache: cache)
            keys = applyRotaryPosition(rope, to: keys, cache: cache)
        }

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

        return oProj(output)
    }
}

class Exaone4MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

class Exaone4TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Exaone4Attention
    let mlp: Exaone4MLP

    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm

    public init(_ args: Exaone4Configuration, isLocal: Bool?) {
        _attention.wrappedValue = Exaone4Attention(args, isLocal: isLocal)
        self.mlp = Exaone4MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postFeedforwardLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(x, mask: mask, cache: cache)
        let h = x + postAttentionLayerNorm(r)
        r = mlp(h)
        let out = h + postFeedforwardLayerNorm(r)
        return out
    }
}

public class Exaone4ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Exaone4TransformerBlock]
    let norm: RMSNorm

    public init(_ args: Exaone4Configuration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers)
            .map { i in
                let isLocal: Bool?
                if let pattern = args.slidingWindowPattern {
                    let patternIndex = i % pattern.count
                    let character = pattern[
                        pattern.index(pattern.startIndex, offsetBy: patternIndex)]
                    isLocal = character == "L"
                } else {
                    isLocal = nil
                }
                return Exaone4TransformerBlock(args, isLocal: isLocal)
            }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class Exaone4Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Exaone4ModelInner
    let configuration: Exaone4Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Exaone4Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Exaone4ModelInner(args)

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

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        return weights
    }

    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        return model.layers.map { layer in
            if layer.attention.isLocal, let slidingWindow = configuration.slidingWindow {
                return RotatingKVCache(maxSize: slidingWindow, keep: 0)
            } else {
                return StandardKVCache()
            }
        }
    }
}

public struct Exaone4Configuration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var maxPositionEmbeddings: Int
    var ropeTheta: Float
    var headDim: Int
    var tieWordEmbeddings: Bool
    var ropeScaling: [String: StringOrNumber]?
    var slidingWindow: Int?
    var slidingWindowPattern: String?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case headDim = "head_dim"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeScaling = "rope_scaling"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.tieWordEmbeddings = try container.decode(Bool.self, forKey: .tieWordEmbeddings)
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        self.slidingWindowPattern = try container.decodeIfPresent(
            String.self, forKey: .slidingWindowPattern)
    }
}

// MARK: - LoRA

extension Exaone4Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
