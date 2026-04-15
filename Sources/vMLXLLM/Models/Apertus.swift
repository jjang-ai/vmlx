import Foundation
import MLX
import vMLXLMCommon
import MLXNN

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/apertus.py

// MARK: - Configuration

public struct ApertusConfiguration: Codable, Sendable {
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var rmsNormEps: Float
    public var vocabSize: Int
    public var tieWordEmbeddings: Bool
    public var maxPositionEmbeddings: Int?
    public var ropeTheta: Float
    public var ropeTraditional: Bool
    public var ropeScaling: [String: StringOrNumber]?

    public init(
        hiddenSize: Int = 4096,
        intermediateSize: Int = 21504,
        numHiddenLayers: Int = 32,
        numAttentionHeads: Int = 32,
        numKeyValueHeads: Int? = 8,
        rmsNormEps: Float = 1e-5,
        vocabSize: Int = 131072,
        tieWordEmbeddings: Bool = false,
        maxPositionEmbeddings: Int? = nil,
        ropeTheta: Float = 1_000_000.0,
        ropeTraditional: Bool = false,
        ropeScaling: [String: StringOrNumber]? = nil
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads ?? numAttentionHeads
        self.rmsNormEps = rmsNormEps
        self.vocabSize = vocabSize
        self.tieWordEmbeddings = tieWordEmbeddings
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        // Required fields
        let hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        let intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        let numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        let numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        let rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        let vocabSize = try container.decode(Int.self, forKey: .vocabSize)

        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.rmsNormEps = rmsNormEps
        self.vocabSize = vocabSize

        // Optional fields with defaults
        self.numKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? numAttentionHeads
        self.tieWordEmbeddings =
            try container.decodeIfPresent(
                Bool.self, forKey: .tieWordEmbeddings) ?? true
        self.maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        self.ropeTheta =
            try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        self.ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)

        if let ropeScaling {
            if ropeScaling["factor"] == nil {
                throw DecodingError.dataCorruptedError(
                    forKey: .ropeScaling, in: container,
                    debugDescription: "rope_scaling must contain 'factor'")
            }
            if let ropeType = ropeScaling["type"] ?? ropeScaling["rope_type"] {
                if case .string = ropeType {
                    let options = [
                        StringOrNumber.string("linear"), StringOrNumber.string("dynamic"),
                        StringOrNumber.string("llama3"),
                    ]
                    if !options.contains(ropeType) {
                        throw DecodingError.dataCorruptedError(
                            forKey: .ropeScaling, in: container,
                            debugDescription:
                                "rope_scaling 'type' currently only supports 'linear', 'dynamic', or 'llama3'"
                        )
                    }
                }
            } else {
                throw DecodingError.dataCorruptedError(
                    forKey: .ropeScaling, in: container,
                    debugDescription: "rope_scaling must contain either 'type' or 'rope_type'")
            }
        }
    }
}

// MARK: - Layers

// Expanded Integral of the Exponential Linear Unit
private class XIELU: Module, UnaryLayer {
    @ModuleInfo(key: "alpha_p") var alphaPParam: MLXArray
    @ModuleInfo(key: "alpha_n") var alphaNParam: MLXArray
    @ModuleInfo(key: "beta") var betaParam: MLXArray
    @ModuleInfo(key: "eps") var epsParam: MLXArray

    override public init() {
        self._alphaPParam.wrappedValue = MLXArray(converting: [0.55])
        self._alphaNParam.wrappedValue = MLXArray(converting: [0.55])
        self._betaParam.wrappedValue = MLXArray(0.5)
        self._epsParam.wrappedValue = MLXArray(-1e-6)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let alphaP = softplus(alphaPParam)
        let alphaN = betaParam + softplus(alphaNParam)

        let posTerm = alphaP * square(x) + betaParam * x
        let negTerm = alphaN * (exp(minimum(x, epsParam)) - 1) - alphaN * x + betaParam * x

        return MLX.where(x .> 0, posTerm, negTerm)
    }
}

private class ApertusAttention: Module {
    let args: ApertusConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    // Apertus Specific: RMSNorm on Q and K
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(args: ApertusConfiguration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.numAttentionHeads
        let kvHeads = args.numKeyValueHeads
        let headDim = args.hiddenSize / heads
        self.scale = pow(Float(headDim), -0.5)

        self._qProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(heads * headDim, dim, bias: false)

        // Norm applies to the head dimension
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        self.rope = initializeRope(
            dims: headDim, base: args.ropeTheta,
            traditional: args.ropeTraditional,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let heads = args.numAttentionHeads
        let kvHeads = args.numKeyValueHeads
        let headDim = args.hiddenSize / heads

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        // 1. Reshape to [B, L, Heads, HeadDim] to apply Norms
        queries = queries.reshaped([B, L, heads, headDim])
        keys = keys.reshaped([B, L, kvHeads, headDim])
        values = values.reshaped([B, L, kvHeads, headDim])

        // 2. Apply QK-Norms (Apertus Specific)
        queries = qNorm(queries)
        keys = kNorm(keys)

        // 3. Transpose to [B, Heads, L, HeadDim] for RoPE and SDPA
        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        // 4. RoPE
        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        if let cache = cache {
            // Update cache (expects [B, H, L, D])
            let (k, v) = cache.update(keys: keys, values: values)
            keys = k
            values = v
        }

        // 5. Attention (SDPA expects [B, H, L, D])
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )

        // 6. Transpose back to [B, L, Heads, HeadDim] and fuse
        let outputFused =
            output
            .transposed(0, 2, 1, 3)
            .reshaped([B, L, heads * headDim])

        return oProj(outputFused)
    }
}

private class ApertusMLP: Module {
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "act_fn") var act: XIELU

    public init(dim: Int, hiddenDim: Int) {
        self._upProj.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDim, dim, bias: false)
        self._act.wrappedValue = XIELU()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(act(upProj(x)))
    }
}

private class ApertusBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: ApertusAttention
    @ModuleInfo(key: "mlp") var mlp: ApertusMLP
    @ModuleInfo(key: "attention_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "feedforward_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ args: ApertusConfiguration) {
        self._selfAttn.wrappedValue = ApertusAttention(args: args)
        self._mlp.wrappedValue = ApertusMLP(dim: args.hiddenSize, hiddenDim: args.intermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }
}

private class ApertusModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [ApertusBlock]
    let norm: RMSNorm

    public init(_ args: ApertusConfiguration) {
        precondition(args.vocabSize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabSize,
            dimensions: args.hiddenSize
        )
        self.layers = (0 ..< args.numHiddenLayers).map { _ in ApertusBlock(args) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: inputs, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class ApertusModel: Module, LLMModel, KVCacheDimensionProvider {

    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let model: ApertusModelInner

    @ModuleInfo(key: "lm_head") public var lmHead: Linear?

    public init(_ args: ApertusConfiguration) {
        self.vocabularySize = args.vocabSize
        self.kvHeads = (0 ..< args.numHiddenLayers).map { _ in args.numKeyValueHeads }
        self.model = ApertusModelInner(args)
        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
        }
    }

    public func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Remove unused precomputed rotary frequencies
        weights.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }
    }

    public func messageGenerator(tokenizer: any Tokenizer) -> any MessageGenerator {
        // some models allow the system role and some do not -- this is enforced
        // by the chat template (code).
        do {
            let probe = [
                [
                    "role": "system",
                    "content": "test",
                ]
            ]
            _ = try tokenizer.applyChatTemplate(messages: probe)
            return DefaultMessageGenerator()
        } catch {
            return NoSystemMessageGenerator()
        }
    }
}

extension ApertusModel: LoRAModel {
    public var loraLayers: [Module] {
        return model.layers
    }
}
