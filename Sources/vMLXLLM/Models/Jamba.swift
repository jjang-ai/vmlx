//
//  Jamba.swift
//  mlx-swift-lm
//
//  Created by Sai Ramana on 10/26/25.
//

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/jamba.py

import Foundation
import MLX
import vMLXLMCommon
import MLXNN

public struct JambaConfiguration: Codable, Sendable {
    public var modelType: String
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var attnLayerOffset: Int
    public var attnLayerPeriod: Int
    public var expertLayerOffset: Int
    public var expertLayerPeriod: Int
    public var mambaDConv: Int
    public var mambaDState: Int
    public var mambaExpand: Int
    public var numExperts: Int
    public var numExpertsPerTok: Int
    public var rmsNormEps: Float
    public var maxPositionEmbeddings: Int
    public var vocabSize: Int
    public var mambaDtRank: Int?
    public var mambaProjBias: Bool
    public var mambaConvBias: Bool
    public var layersBlockType: [String]?
    public var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case attnLayerOffset = "attn_layer_offset"
        case attnLayerPeriod = "attn_layer_period"
        case expertLayerOffset = "expert_layer_offset"
        case expertLayerPeriod = "expert_layer_period"
        case mambaDConv = "mamba_d_conv"
        case mambaDState = "mamba_d_state"
        case mambaExpand = "mamba_expand"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case vocabSize = "vocab_size"
        case mambaDtRank = "mamba_dt_rank"
        case mambaProjBias = "mamba_proj_bias"
        case mambaConvBias = "mamba_conv_bias"
        case layersBlockType = "layers_block_type"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        self.numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        self.numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        self.attnLayerOffset = try container.decode(Int.self, forKey: .attnLayerOffset)
        self.attnLayerPeriod = try container.decode(Int.self, forKey: .attnLayerPeriod)
        self.expertLayerOffset = try container.decode(Int.self, forKey: .expertLayerOffset)
        self.expertLayerPeriod = try container.decode(Int.self, forKey: .expertLayerPeriod)
        self.mambaDConv = try container.decode(Int.self, forKey: .mambaDConv)
        self.mambaDState = try container.decode(Int.self, forKey: .mambaDState)
        self.mambaExpand = try container.decode(Int.self, forKey: .mambaExpand)
        self.numExperts = try container.decode(Int.self, forKey: .numExperts)
        self.numExpertsPerTok = try container.decode(Int.self, forKey: .numExpertsPerTok)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        self.mambaDtRank = try container.decodeIfPresent(Int.self, forKey: .mambaDtRank)
        self.mambaProjBias =
            try container.decodeIfPresent(Bool.self, forKey: .mambaProjBias) ?? false
        self.mambaConvBias =
            try container.decodeIfPresent(Bool.self, forKey: .mambaConvBias) ?? true
        self.layersBlockType = try container.decodeIfPresent(
            [String].self, forKey: .layersBlockType)
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true

        // Post-initialization logic

        // If mambaDtRank is nil, calculate it
        if self.mambaDtRank == nil {
            self.mambaDtRank = Int((Double(self.hiddenSize) / 16.0).rounded(.up))
        }

        // If layersBlockType is nil, generate it based on layer configuration
        if self.layersBlockType == nil {
            self.layersBlockType = (0 ..< self.numHiddenLayers).map { i in
                (i % self.attnLayerPeriod == self.attnLayerOffset) ? "attention" : "mamba"
            }
        }
    }
}

class JambaMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    public init(_ config: JambaConfiguration) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(silu(gateProj(x)) * upProj(x))
    }
}

class JambaAttention: Module {
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    public init(_ config: JambaConfiguration) {
        self.numAttentionHeads = config.numAttentionHeads
        self.numKeyValueHeads = config.numKeyValueHeads
        self.headDim = config.hiddenSize / config.numAttentionHeads
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(
            config.hiddenSize, config.numAttentionHeads * self.headDim, bias: false)
        _kProj.wrappedValue = Linear(
            config.hiddenSize, config.numKeyValueHeads * self.headDim, bias: false)
        _vProj.wrappedValue = Linear(
            config.hiddenSize, config.numKeyValueHeads * self.headDim, bias: false)
        _oProj.wrappedValue = Linear(
            config.numAttentionHeads * self.headDim, config.hiddenSize, bias: false)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        queries = queries.reshaped(B, L, numAttentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, numKeyValueHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, numKeyValueHeads, -1).transposed(0, 2, 1, 3)

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

private func fma(_ a: MLXArray, _ b: MLXArray, _ c: MLXArray) -> MLXArray {
    return a * b + c
}

class JambaMambaMixer: Module {
    let hiddenSize: Int
    let ssmStateSize: Int
    let convKernelSize: Int
    let intermediateSize: Int
    let timeStepRank: Int
    let useConvBias: Bool
    let useBias: Bool

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "x_proj") var xProj: Linear
    @ModuleInfo(key: "dt_proj") var dtProj: Linear

    @ParameterInfo(key: "A_log") var A_Log: MLXArray
    @ParameterInfo(key: "D") var D: MLXArray

    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "dt_layernorm") var dtLayerNorm: RMSNorm
    @ModuleInfo(key: "b_layernorm") var bLayerNorm: RMSNorm
    @ModuleInfo(key: "c_layernorm") var cLayerNorm: RMSNorm

    public init(_ config: JambaConfiguration) {
        self.hiddenSize = config.hiddenSize
        self.ssmStateSize = config.mambaDState
        self.convKernelSize = config.mambaDConv
        self.intermediateSize = config.mambaExpand * config.hiddenSize
        self.timeStepRank = config.mambaDtRank!
        self.useConvBias = config.mambaConvBias
        self.useBias = config.mambaProjBias

        _inProj.wrappedValue = Linear(
            self.hiddenSize, self.intermediateSize * 2, bias: self.useBias)

        _conv1d.wrappedValue = Conv1d(
            inputChannels: self.intermediateSize,
            outputChannels: self.intermediateSize,
            kernelSize: self.convKernelSize,
            padding: 0,
            groups: self.intermediateSize,
            bias: self.useConvBias
        )

        _xProj.wrappedValue = Linear(
            self.intermediateSize,
            self.timeStepRank + self.ssmStateSize * 2,
            bias: false
        )

        _dtProj.wrappedValue = Linear(self.timeStepRank, self.intermediateSize, bias: true)

        // Initialize A_log

        let A = repeated(
            MLXArray(Array(1 ... self.ssmStateSize).map { Float($0) }).reshaped([
                1, self.ssmStateSize,
            ]),
            count: self.intermediateSize,
            axis: 0
        )
        self._A_Log.wrappedValue = log(A)

        // Initialize D
        self._D.wrappedValue = MLXArray.ones([self.intermediateSize])

        _outProj.wrappedValue = Linear(self.intermediateSize, self.hiddenSize, bias: self.useBias)
        _dtLayerNorm.wrappedValue = RMSNorm(dimensions: self.timeStepRank, eps: config.rmsNormEps)
        _bLayerNorm.wrappedValue = RMSNorm(dimensions: self.ssmStateSize, eps: config.rmsNormEps)
        _cLayerNorm.wrappedValue = RMSNorm(dimensions: self.ssmStateSize, eps: config.rmsNormEps)
    }

    private func ssmStep(_ x: MLXArray, _ A: MLXArray, state: MLXArray?) -> (MLXArray, MLXArray) {
        let T = x.dim(1)

        let deltaBC = xProj(x)
        let splits = MLX.split(
            deltaBC,
            indices: [timeStepRank, timeStepRank + ssmStateSize],
            axis: -1
        )
        var delta = splits[0]
        var B = splits[1]
        var C = splits[2]

        delta = dtLayerNorm(delta)
        B = bLayerNorm(B)
        C = cLayerNorm(C)

        delta = softplus(dtProj(delta))

        let newState = expandedDimensions(delta * x, axis: -1) * expandedDimensions(B, axis: -2)
        let dtA = exp(expandedDimensions(delta, axis: -1) * A)

        var currentState = state
        for t in 0 ..< T {
            if let state = currentState {
                newState[0..., t] = fma(state, dtA[0..., t], newState[0..., t])
            }
            currentState = newState[0..., t]
        }

        let y = (newState.matmul(expandedDimensions(C, axis: -1))).squeezed(axis: -1)
        return (y + D * x, newState[0..., -1])
    }

    private func processSequence(_ x: MLXArray, convState: MLXArray?, ssmState: MLXArray?) -> (
        MLXArray, (MLXArray, MLXArray)
    ) {
        let xz = inProj(x)
        let splits = xz.split(parts: 2, axis: -1)
        var x = splits[0]
        let z = splits[1]

        let K = convKernelSize
        let xFull: MLXArray
        if let convState = convState {
            xFull = concatenated([convState, x], axis: 1)
        } else {
            xFull = padded(
                x, widths: [IntOrPair((0, 0)), IntOrPair((K - 1, 0)), IntOrPair((0, 0))])
        }

        let convOut = conv1d(xFull)
        let newConvState = xFull[0..., (1 - K)..., 0...]
        x = silu(convOut)

        let A = -exp(A_Log)
        let (y, newSsmState) = ssmStep(x, A, state: ssmState)
        let output = outProj(silu(z) * y)

        return (output, (newConvState, newSsmState))
    }

    public func callAsFunction(_ x: MLXArray, cache: MambaCache?) -> MLXArray {
        let convState = cache?[0]
        let ssmState = cache?[1]

        let (output, (newConvState, newSsmState)) = processSequence(
            x, convState: convState, ssmState: ssmState)

        if let cache = cache {
            cache[0] = newConvState
            cache[1] = newSsmState
        }

        return output
    }
}

class JambaSparseMoeBlock: Module {
    let numExpertsPerTok: Int

    @ModuleInfo(key: "router") var router: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    /// Flash MoE drop-in for router+switch_mlp+weighted-sum path.
    /// See DeepseekV3MoE.flashMoeShim for pattern rationale. F-G8.
    fileprivate var flashMoeShim: FlashMoEBlock?

    public init(_ config: JambaConfiguration) {
        self.numExpertsPerTok = config.numExpertsPerTok

        _router.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)

        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.intermediateSize,
            numExperts: config.numExperts
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let shim = flashMoeShim {
            return shim(x)
        }
        let gates = router(x)
        let k = numExpertsPerTok
        let inds = stopGradient(MLX.argPartition(-gates, kth: k - 1, axis: -1)[.ellipsis, ..<k])
        var scores = MLX.takeAlong(gates, inds, axis: -1)
        scores = MLX.softmax(scores, axis: -1, precise: true)

        let y = switchMLP(x, inds)
        return (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
    }
}

class JambaDecoderLayer: Module {
    let isAttn: Bool
    let isSparseMoe: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: JambaAttention?
    @ModuleInfo(key: "mamba") var mamba: JambaMambaMixer?

    @ModuleInfo(key: "feed_forward") fileprivate var feedForward: Module

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_ff_layernorm") var preFFLayerNorm: RMSNorm

    public init(_ config: JambaConfiguration, layerType: String) {
        self.isAttn = layerType == "attention"

        if isAttn {
            _selfAttn.wrappedValue = JambaAttention(config)
        } else {
            _mamba.wrappedValue = JambaMambaMixer(config)
        }

        if config.numExperts > 1 {
            _feedForward.wrappedValue = JambaSparseMoeBlock(config)
            self.isSparseMoe = true
        } else {
            _feedForward.wrappedValue = JambaMLP(config)
            self.isSparseMoe = false
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _preFFLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h: MLXArray
        if isAttn {
            h = selfAttn!(inputLayerNorm(x), mask: mask, cache: cache)
        } else {
            h = mamba!(inputLayerNorm(x), cache: cache as? MambaCache)
        }

        let r = x + h
        let out: MLXArray
        if isSparseMoe {
            out = r + (feedForward as! JambaSparseMoeBlock)(preFFLayerNorm(r))
        } else {
            out = r + (feedForward as! JambaMLP)(preFFLayerNorm(r))
        }

        return out
    }
}

public class JambaModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [JambaDecoderLayer]

    @ModuleInfo(key: "final_layernorm") var finalLayerNorm: RMSNorm

    let attnIdx: Int
    let ssmIdx: Int

    public init(_ config: JambaConfiguration) {
        precondition(config.vocabSize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)

        let layersBlockType = config.layersBlockType!
        self.layers = layersBlockType.enumerated().map { (_, type) in
            JambaDecoderLayer(config, layerType: type)
        }

        _finalLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        self.attnIdx = layersBlockType.firstIndex(of: "attention")!
        self.ssmIdx = layersBlockType.firstIndex(of: "mamba")!
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache?]?) -> MLXArray {
        var h = embedTokens(inputs)

        let cacheArray = cache ?? Array(repeating: nil, count: layers.count)

        let attnMask = createAttentionMask(h: h, cache: cacheArray[attnIdx])

        for (i, layer) in layers.enumerated() {
            if layer.isAttn {
                h = layer(h, mask: attnMask, cache: cacheArray[i])
            } else {
                h = layer(h, mask: .none, cache: cacheArray[i])
            }
        }

        return finalLayerNorm(h)
    }
}

public class JambaModel: Module, LLMModel, KVCacheDimensionProvider {
    public let kvHeads: [Int]
    let modelType: String
    let config: JambaConfiguration
    public let model: JambaModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        return model.layers.map { layer in
            if layer.isAttn {
                return KVCacheSimple()
            } else {
                return MambaCache()
            }
        }
    }

    public init(_ config: JambaConfiguration) {
        self.modelType = config.modelType
        self.config = config
        self.kvHeads = (0 ..< config.numHiddenLayers).map { _ in config.numKeyValueHeads }
        self.model = JambaModelInner(config)

        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)

        if config.tieWordEmbeddings {
            out = model.embedTokens.asLinear(out)
        } else {
            out = lmHead!(out)
        }

        return out
    }

    public func makeCache() -> [KVCache] {
        return model.layers.map { layer in
            if layer.isAttn {
                return KVCacheSimple()
            } else {
                return MambaCache()
            }
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = weights

        // Handle conv1d weight reshaping
        for (k, v) in sanitizedWeights {
            if k.contains("conv1d.weight") && v.dim(-1) != 1 {
                sanitizedWeights[k] = v.movedAxis(source: 2, destination: 1)
            }
        }

        // Handle tied embeddings
        if config.tieWordEmbeddings {
            sanitizedWeights["lm_head.weight"] = nil
        }

        // Handle MoE expert weights
        if sanitizedWeights["model.layers.0.block_sparse_moe.experts.0.w1.weight"] == nil {
            return sanitizedWeights
        }

        for l in 0 ..< config.numHiddenLayers {
            let prefix = "model.layers.\(l)"
            let mapping: [(String, String)] = [
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ]

            for (n, m) in mapping {
                for k in ["weight", "scales", "biases"] {
                    if sanitizedWeights["\(prefix).block_sparse_moe.experts.0.\(n).\(k)"] != nil {
                        let toJoin = (0 ..< config.numExperts).map { e in
                            sanitizedWeights.removeValue(
                                forKey: "\(prefix).block_sparse_moe.experts.\(e).\(n).\(k)"
                            )!
                        }
                        sanitizedWeights["\(prefix).block_sparse_moe.switch_mlp.\(m).\(k)"] =
                            MLX.stacked(toJoin)
                    }
                }
            }
        }

        return sanitizedWeights
    }

    public var layers: [Module] {
        return self.model.layers
    }
}

// MARK: - LoRA

extension JambaModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers.compactMap { layer in
            if layer.isAttn {
                return layer
            }
            return nil
        }
    }
}

// MARK: - Flash MoE conformance (F-G8)
//
// Jamba is a hybrid SSM model: `JambaDecoderLayer` interleaves
// attention / mamba mixers, both routing into the same `feedForward`
// slot which is either `JambaSparseMoeBlock` (MoE layers) or `JambaMLP`
// (dense layers). Flash MoE targets only the MoE layers; dense layers
// return `.none` and are skipped.
//
// The SSM/attention sub-block lives in separate `@ModuleInfo` slots
// (`self_attn` / `mamba`) and is never touched — Flash MoE stays on
// the feed-forward side of the residual, same as Qwen3MoE / MiniMax.

extension JambaModel: FlashMoEReplaceable {
    public var flashMoELayers: [FlashMoELayer] {
        model.layers
    }
}

extension JambaDecoderLayer: FlashMoELayer {
    public var flashMoELayout: FlashMoELayout {
        isSparseMoe ? .textPathSwitchGLU : .none
    }

    public func replaceMoEBlock(with block: FlashMoEBlock) throws {
        guard let moe = feedForward as? JambaSparseMoeBlock else { return }
        let router = moe.router
        let topK = moe.numExpertsPerTok
        block.topK = topK
        block.router = { x in
            // Jamba router is softmax(topK(router(x))) — same math as
            // JambaSparseMoeBlock.callAsFunction pre-expert path.
            let gates = router(x)
            let inds = stopGradient(
                MLX.argPartition(-gates, kth: topK - 1, axis: -1)[.ellipsis, ..<topK])
            let scores = MLX.softmax(
                MLX.takeAlong(gates, inds, axis: -1), axis: -1, precise: true)
            return (indices: inds, scores: scores)
        }
        moe.flashMoeShim = block
    }
}
