import Foundation
import MLX
import vMLXLMCommon
import MLXNN

// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/phimoe.py

public struct PhiMoEConfiguration: Codable, Sendable {
    var modelType: String = "phimoe"
    var vocabularySize: Int = 32064
    var hiddenSize: Int = 4096
    var intermediateSize: Int = 6400
    var hiddenLayers: Int = 32
    var attentionHeads: Int = 32
    var kvHeads: Int = 8
    var maxPositionEmbeddings: Int = 131072
    var originalMaxPositionEmbeddings: Int = 4096
    var rmsNormEps: Float = 1e-6
    var ropeScaling: RopeScalingWithFactorArrays?
    var numLocalExperts: Int = 16
    var numExpertsPerToken: Int = 2
    var ropeTheta: Float = 10000.0

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeScaling = "rope_scaling"
        case numLocalExperts = "num_local_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case ropeTheta = "rope_theta"
    }
}

class PhiMoEAttention: Module {
    let args: PhiMoEConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: SuScaledRoPE

    init(_ args: PhiMoEConfiguration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.hiddenSize / heads
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, heads * headDim, bias: true)
        self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: true)

        self.rope = SuScaledRoPE(
            dimensions: headDim,
            base: args.ropeTheta,
            maxPositionEmbeddings: args.maxPositionEmbeddings,
            originalMaxPositionEmbeddings: args.originalMaxPositionEmbeddings,
            shortFactor: args.ropeScaling?.shortFactor ?? [1.0],
            longFactor: args.ropeScaling?.longFactor ?? [1.0],
            shortMScale: args.ropeScaling?.shortMScale,
            longMScale: args.ropeScaling?.longMScale
        )
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        let queries = wq(x)
        let keys = wk(x)
        let values = wv(x)

        // Prepare the queries, keys and values for the attention computation
        var q = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        var k = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        let v = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        q = applyRotaryPosition(rope, to: q, cache: cache)
        k = applyRotaryPosition(rope, to: k, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: q,
            keys: k,
            values: v,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

class PhiMoESparseMoeBlock: Module {
    let hiddenDim: Int
    let ffnDim: Int
    let numExperts: Int
    let topK: Int
    fileprivate let layerIdx: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    init(_ args: PhiMoEConfiguration, layerIdx: Int) {
        self.hiddenDim = args.hiddenSize
        self.ffnDim = args.intermediateSize
        self.numExperts = args.numLocalExperts
        self.topK = args.numExpertsPerToken
        self.layerIdx = layerIdx

        self._gate.wrappedValue = Linear(hiddenDim, numExperts, bias: false)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: hiddenDim, hiddenDims: ffnDim, numExperts: numExperts)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gates = gate(x)

        let k = self.topK
        let inds = MLX.stopGradient(
            MLX.argPartition(
                -gates,
                kth: k - 1,
                axis: -1
            )[.ellipsis, ..<k])
        JangPressRouteTelemetry.recordTopK(layer: layerIdx, indices: inds)
        let scores = MLX.softmax(MLX.takeAlong(gates, inds, axis: -1), axis: -1, precise: true)

        let y = switchMLP(x, inds)
        return (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
    }
}

class PhiMoEDecoderLayer: Module {
    let hiddenSize: Int

    @ModuleInfo(key: "self_attn") var selfAttn: PhiMoEAttention
    @ModuleInfo(key: "block_sparse_moe") var blockSparseMoe: PhiMoESparseMoeBlock
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: LayerNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: LayerNorm

    /// Flash MoE Phase 2b shim. See `MiniMax.swift` for rationale.
    fileprivate var flashMoeShim: FlashMoEBlock? = nil

    init(_ args: PhiMoEConfiguration, layerIdx: Int) {
        self.hiddenSize = args.hiddenSize

        self._selfAttn.wrappedValue = PhiMoEAttention(args)
        self._blockSparseMoe.wrappedValue = PhiMoESparseMoeBlock(args, layerIdx: layerIdx)
        self._inputLayerNorm.wrappedValue = LayerNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = LayerNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var residual = x
        var hiddenStates = inputLayerNorm(x)
        hiddenStates = selfAttn(hiddenStates, mask: mask, cache: cache)
        hiddenStates = residual + hiddenStates

        residual = hiddenStates
        hiddenStates = postAttentionLayerNorm(hiddenStates)
        hiddenStates = flashMoeShim.map { $0(hiddenStates) } ?? blockSparseMoe(hiddenStates)
        hiddenStates = residual + hiddenStates

        return hiddenStates
    }
}

public class PhiMoEModelInner: Module {
    let args: PhiMoEConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [PhiMoEDecoderLayer]
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(_ args: PhiMoEConfiguration) {
        self.args = args

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { i in PhiMoEDecoderLayer(args, layerIdx: i) }
        self._norm.wrappedValue = LayerNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
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

public class PhiMoEModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: PhiMoEModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ args: PhiMoEConfiguration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.model = PhiMoEModelInner(args)
        self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: true)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        return lmHead(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = weights
        if sanitizedWeights["model.layers.0.block_sparse_moe.experts.0.w1.weight"] == nil {
            return sanitizedWeights
        }

        for l in 0 ..< model.args.hiddenLayers {
            let prefix = "model.layers.\(l)"
            for (n, m) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
                for k in ["weight", "scales", "biases"] {
                    if sanitizedWeights["\(prefix).block_sparse_moe.experts.0.\(n).\(k)"] != nil {
                        let toJoin = (0 ..< model.args.numLocalExperts).map { e in
                            sanitizedWeights.removeValue(
                                forKey: "\(prefix).block_sparse_moe.experts.\(e).\(n).\(k)")!
                        }
                        sanitizedWeights["\(prefix).block_sparse_moe.switch_mlp.\(m).\(k)"] =
                            loadTimeMaterializedStacked(toJoin)
                    }
                }
            }
        }

        return sanitizedWeights
    }
}

// MARK: - LoRA

extension PhiMoEModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Flash MoE (Phase 2b)

extension PhiMoEModel: FlashMoEReplaceable {
    public var flashMoELayers: [FlashMoELayer] {
        model.layers
    }
}

extension PhiMoEDecoderLayer: FlashMoELayer {
    public var flashMoELayout: FlashMoELayout { .textPathSwitchGLU }

    public func replaceMoEBlock(with block: FlashMoEBlock) throws {
        let gate = blockSparseMoe.gate
        let topK = blockSparseMoe.topK
        block.topK = topK
        block.router = { x in
            let gates = gate(x)
            let inds = MLX.stopGradient(
                MLX.argPartition(-gates, kth: topK - 1, axis: -1
                )[.ellipsis, ..<topK])
            let scores = MLX.softmax(
                MLX.takeAlong(gates, inds, axis: -1),
                axis: -1, precise: true)
            return (indices: inds, scores: scores)
        }
        self.flashMoeShim = block
    }
}
