//
//  Qwen3Next.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_next.py
//

import Foundation
import MLX
import vMLXLMCommon
import MLXNN

// MARK: - Helpers

/// Compiled sigmoid gate: fuses sigmoid + multiply into one Metal dispatch.
/// Used per-layer in Qwen3.5 attention (GatedDeltaNet) — 40 layers per forward.
private let _compiledSigmoidMultiply: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { (x: MLXArray, gate: MLXArray) -> MLXArray in
        x * sigmoid(gate)
    }
    return HardwareInfo.isCompiledDecodeSupported ? compile(shapeless: true, body) : body
}()

func sigmoidMultiply(_ x: MLXArray, _ gate: MLXArray) -> MLXArray {
    _compiledSigmoidMultiply(x, gate)
}

/// Compiled precise SwiGLU — matches Python `@mx.compile` `_precise_swiglu`.
/// Fuses silu(gate) in float32 + cast-to-fp32 + multiply + cast-to-hdtype into
/// ONE Metal dispatch. Without this, every GatedDeltaNet layer issues 5 separate
/// Metal kernels (silu + asType + asType + multiply + asType) and at ~0.1ms per
/// dispatch times ~30 layers, this is the dominant per-step overhead vs Python.
private let _compiledPreciseSwiGLU: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = { (h: MLXArray, gate: MLXArray, x: MLXArray) -> MLXArray in
        let gateF32 = silu(gate.asType(.float32))
        let xF32 = x.asType(.float32)
        return (gateF32 * xF32).asType(h.dtype)
    }
    return HardwareInfo.isCompiledDecodeSupported ? compile(shapeless: true, body) : body
}()

// MARK: - Model Components

// Compiled swiglu for MLP: fuses silu(gate) * up into 1 Metal dispatch.
private let _compiledSwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { (gate: MLXArray, x: MLXArray) -> MLXArray in
        silu(gate) * x
    }
    return HardwareInfo.isCompiledDecodeSupported ? compile(shapeless: true, body) : body
}()

final class Qwen3NextRMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, gate: MLXArray? = nil) -> MLXArray {
        let x = MLXFast.rmsNorm(hiddenStates, weight: weight, eps: eps)
        if let gate {
            // Match Python's _precise_swiglu: float32 cast + silu + multiply, then cast back.
            return _compiledPreciseSwiGLU(hiddenStates, gate, x)
        }
        return x
    }
}

public final class Qwen3NextAttention: Module {
    let args: Qwen3NextConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ args: Qwen3NextConfiguration) {
        self.args = args

        let headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(
            args.hiddenSize, args.attentionHeads * headDim * 2, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _vProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(
            args.attentionHeads * headDim, args.hiddenSize, bias: args.attentionBias)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let ropeDims = Int(Float(headDim) * args.partialRotaryFactor)
        self.rope = initializeRope(
            dims: max(1, ropeDims),
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )

        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let qProjOutput = qProj(x)
        let qSplit = qProjOutput.reshaped(B, L, args.attentionHeads, -1).split(parts: 2, axis: -1)
        var queries = qSplit[0]
        let gate = qSplit[1].reshaped(B, L, -1)

        var keys = kProj(x)
        var values = vProj(x)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

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

        return oProj(sigmoidMultiply(output, gate))
    }
}

final class Qwen3NextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Fuses silu(gate) * up into 1 Metal dispatch instead of 2.
        let activated = _compiledSwiGLU(gateProj(x), upProj(x))
        return downProj(activated)
    }
}

public final class Qwen3NextGatedDeltaNet: Module {
    let hiddenSize: Int
    let numVHeads: Int
    let numKHeads: Int
    let headKDim: Int
    let headVDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernelSize: Int
    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkvz") var inProjQKVZ: Linear
    @ModuleInfo(key: "in_proj_ba") var inProjBA: Linear

    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray

    @ModuleInfo(key: "norm") var norm: Qwen3NextRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: Qwen3NextConfiguration) {
        self.hiddenSize = args.hiddenSize
        self.numVHeads = args.linearNumValueHeads
        self.numKHeads = args.linearNumKeyHeads
        self.headKDim = args.linearKeyHeadDim
        self.headVDim = args.linearValueHeadDim
        self.keyDim = headKDim * numKHeads
        self.valueDim = headVDim * numVHeads
        self.convKernelSize = args.linearConvKernelDim
        self.convDim = keyDim * 2 + valueDim

        precondition(numVHeads % numKHeads == 0, "num_v_heads must be divisible by num_k_heads")

        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: convDim,
            bias: false
        )

        _inProjQKVZ.wrappedValue = Linear(
            hiddenSize, keyDim * 2 + valueDim * 2, bias: false)
        _inProjBA.wrappedValue = Linear(hiddenSize, numVHeads * 2, bias: false)

        _dtBias.wrappedValue = MLXArray.ones([numVHeads])
        let a = MLXRandom.uniform(low: 0, high: 16, [numVHeads])
        _aLog.wrappedValue = log(a)

        _norm.wrappedValue = Qwen3NextRMSNormGated(dimensions: headVDim, eps: args.rmsNormEps)
        _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)

        super.init()
    }

    private func fixQueryKeyValueOrdering(
        mixedQKVZ: MLXArray,
        mixedBA: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
        let B = mixedQKVZ.dim(0)
        let S = mixedQKVZ.dim(1)
        let nk = numKHeads
        let dn = headKDim
        let nv = numVHeads
        let dv = headVDim
        let vHeadsPerK = nv / nk

        let qkvz = mixedQKVZ.reshaped(B, S, nk, -1)
        let ba = mixedBA.reshaped(B, S, nk, -1)

        let qkvzSplit = MLX.split(
            qkvz,
            indices: [dn, 2 * dn, 2 * dn + vHeadsPerK * dv],
            axis: -1
        )
        let q = qkvzSplit[0]
        let k = qkvzSplit[1]
        let v = qkvzSplit[2].reshaped(B, S, -1, dv)
        let z = qkvzSplit[3].reshaped(B, S, -1, dv)

        let baSplit = MLX.split(ba, indices: [vHeadsPerK], axis: -1)
        let b = baSplit[0].reshaped(B, S, nv)
        let a = baSplit[1].reshaped(B, S, nv)

        return (q, k, v, z, b, a)
    }

    public func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXArray? = nil,
        cache: MambaCache? = nil
    ) -> MLXArray {
        let B = inputs.dim(0)
        let S = inputs.dim(1)

        let (q, k, v, z, b, a) = fixQueryKeyValueOrdering(
            mixedQKVZ: inProjQKVZ(inputs),
            mixedBA: inProjBA(inputs)
        )

        let dtype = inputs.dtype
        let convState: MLXArray
        if let cacheState = cache?[0] {
            convState = cacheState
        } else {
            convState = MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: dtype)
        }

        var mixedQKV = concatenated(
            [q.reshaped(B, S, -1), k.reshaped(B, S, -1), v.reshaped(B, S, -1)],
            axis: -1
        )

        if let mask {
            mixedQKV = MLX.where(
                expandedDimensions(mask, axis: -1), mixedQKV, MLXArray.zeros(like: mixedQKV))
        }

        let convInput = concatenated([convState, mixedQKV], axis: 1)
        if let cache {
            cache[0] = convInput[0..., (1 - convKernelSize)..., 0...]
        }

        let convOut = silu(conv1d(convInput))
        let convSplit = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)

        var qOut = convSplit[0].reshaped(B, S, numKHeads, headKDim)
        var kOut = convSplit[1].reshaped(B, S, numKHeads, headKDim)
        let vOut = convSplit[2].reshaped(B, S, numVHeads, headVDim)

        let invScale = pow(Float(headKDim), -0.5)
        qOut =
            MLXArray(invScale * invScale, dtype: qOut.dtype)
            * MLXFast.rmsNorm(qOut, weight: MLXArray.mlxNone, eps: 1e-6)
        kOut =
            MLXArray(invScale, dtype: kOut.dtype)
            * MLXFast.rmsNorm(kOut, weight: MLXArray.mlxNone, eps: 1e-6)

        let (out, newState) = gatedDeltaUpdate(
            q: qOut,
            k: kOut,
            v: vOut,
            a: a,
            b: b,
            aLog: aLog,
            dtBias: dtBias,
            state: cache?[1],
            mask: mask
        )

        if let cache {
            cache[1] = newState
        }

        let normalized = norm(out, gate: z)
        return outProj(normalized.reshaped(B, S, -1))
    }
}

final class Qwen3NextSparseMoeBlock: Module {
    let normTopkProb: Bool
    let numExperts: Int
    let topK: Int
    fileprivate let layerIdx: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen3NextMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ args: Qwen3NextConfiguration, layerIdx: Int) {
        self.normTopkProb = args.normTopkProb
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerTok
        self.layerIdx = layerIdx

        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts
        )

        _sharedExpert.wrappedValue = Qwen3NextMLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.sharedExpertIntermediateSize
        )
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var gates = gate(x)
        gates = MLX.softmax(gates, axis: -1, precise: true)

        let k = topK
        let kth = gates.dim(-1) - k
        let inds = MLX.argPartition(gates, kth: kth, axis: -1)[.ellipsis, (kth)...]
        JangPressRouteTelemetry.recordTopK(layer: layerIdx, indices: inds)
        var scores = MLX.takeAlong(gates, inds, axis: -1)
        if normTopkProb {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }

        let y = switchMLP(x, inds)
        let combined = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)

        var sharedY = sharedExpert(x)
        sharedY = sigmoid(sharedExpertGate(x)) * sharedY

        return combined + sharedY
    }
}

final class Qwen3NextDecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3NextAttention?
    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen3NextGatedDeltaNet?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ args: Qwen3NextConfiguration, layerIdx: Int) {
        self.isLinear = (layerIdx + 1) % args.fullAttentionInterval != 0

        if isLinear {
            _linearAttn.wrappedValue = Qwen3NextGatedDeltaNet(args)
        } else {
            _selfAttn.wrappedValue = Qwen3NextAttention(args)
        }

        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)

        let useMoE =
            !args.mlpOnlyLayers.contains(layerIdx)
            && args.numExperts > 0
            && (layerIdx + 1) % args.decoderSparseStep == 0

        if useMoE {
            _mlp.wrappedValue = Qwen3NextSparseMoeBlock(args, layerIdx: layerIdx)
        } else {
            _mlp.wrappedValue = Qwen3NextMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let h: MLXArray
        if isLinear {
            h = linearAttn!(inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache)
        } else {
            h = selfAttn!(inputLayerNorm(x), mask: attentionMask, cache: cache)
        }

        let r = x + h
        let normed = postAttentionLayerNorm(r)
        if let moe = mlp as? Qwen3NextSparseMoeBlock {
            return r + moe(normed)
        }
        return r + (mlp as! Qwen3NextMLP)(normed)
    }
}

public class Qwen3NextModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen3NextDecoderLayer]
    let norm: RMSNorm

    let ssmIdx: Int
    let faIdx: Int

    init(_ args: Qwen3NextConfiguration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        self.layers = (0 ..< args.hiddenLayers).map { layerIdx in
            Qwen3NextDecoderLayer(args, layerIdx: layerIdx)
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
            let attnMask = layer.isLinear ? MLXFast.ScaledDotProductAttentionMaskMode.none : faMask
            hiddenStates = layer(
                hiddenStates, attentionMask: attnMask, ssmMask: mask, cache: cacheArray?[i])
        }

        return norm(hiddenStates)
    }
}

public class Qwen3NextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen3NextModelInner
    let configuration: Qwen3NextConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Qwen3NextConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen3NextModelInner(args)

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
            if layer.isLinear {
                return MambaCache()
            }
            return KVCacheSimple()
        }
    }

    public func makeCache() -> [KVCache] {
        return newCache(parameters: nil)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = weights

        if configuration.tieWordEmbeddings {
            sanitizedWeights["lm_head.weight"] = nil
        }

        let mtpKeys = sanitizedWeights.keys.filter { $0.contains("mtp.") }
        for key in mtpKeys {
            sanitizedWeights[key] = nil
        }

        if sanitizedWeights["model.layers.0.mlp.experts.0.up_proj.weight"] == nil {
            return sanitizedWeights
        }

        for l in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(l).mlp"
            for n in ["up_proj", "down_proj", "gate_proj"] {
                let key = "\(prefix).experts.0.\(n).weight"
                if sanitizedWeights[key] != nil {
                    let toJoin = (0 ..< configuration.numExperts).map { e in
                        sanitizedWeights.removeValue(
                            forKey: "\(prefix).experts.\(e).\(n).weight")!
                    }
                    // RAM safety: per-layer materialize via canonical helper.
                    sanitizedWeights["\(prefix).switch_mlp.\(n).weight"] =
                        loadTimeMaterializedStacked(toJoin)
                }
            }
        }

        let normSuffixes = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ]

        for key in Array(sanitizedWeights.keys) {
            guard let value = sanitizedWeights[key] else { continue }
            if key.contains("conv1d.weight") && value.dim(-1) != 1 {
                sanitizedWeights[key] = value.movedAxis(source: 2, destination: 1)
                continue
            }
            if normSuffixes.contains(where: { key.hasSuffix($0) }) && value.ndim == 1 {
                sanitizedWeights[key] = value + MLXArray(1, dtype: value.dtype)
            }
        }

        return sanitizedWeights
    }
}

public struct Qwen3NextConfiguration: Codable, Sendable {
    var modelType: String = "qwen3_next"
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var linearNumValueHeads: Int
    var linearNumKeyHeads: Int
    var linearKeyHeadDim: Int
    var linearValueHeadDim: Int
    var linearConvKernelDim: Int
    var numExperts: Int
    var numExpertsPerTok: Int
    var decoderSparseStep: Int
    var sharedExpertIntermediateSize: Int
    var mlpOnlyLayers: [Int]
    var moeIntermediateSize: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float
    var partialRotaryFactor: Float
    var maxPositionEmbeddings: Int
    var normTopkProb: Bool
    var tieWordEmbeddings: Bool
    var attentionBias: Bool
    var headDim: Int?
    var ropeScaling: [String: StringOrNumber]?
    var fullAttentionInterval: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case decoderSparseStep = "decoder_sparse_step"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case mlpOnlyLayers = "mlp_only_layers"
        case moeIntermediateSize = "moe_intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case normTopkProb = "norm_topk_prob"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case fullAttentionInterval = "full_attention_interval"
    }

    public init(from decoder: Decoder) throws {
        let container: KeyedDecodingContainer<Qwen3NextConfiguration.CodingKeys> =
            try decoder.container(keyedBy: Qwen3NextConfiguration.CodingKeys.self)

        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3_next"
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.linearNumValueHeads = try container.decode(Int.self, forKey: .linearNumValueHeads)
        self.linearNumKeyHeads = try container.decode(Int.self, forKey: .linearNumKeyHeads)
        self.linearKeyHeadDim = try container.decode(Int.self, forKey: .linearKeyHeadDim)
        self.linearValueHeadDim = try container.decode(Int.self, forKey: .linearValueHeadDim)
        self.linearConvKernelDim = try container.decode(Int.self, forKey: .linearConvKernelDim)
        self.numExperts = try container.decode(Int.self, forKey: .numExperts)
        self.numExpertsPerTok = try container.decode(Int.self, forKey: .numExpertsPerTok)
        self.decoderSparseStep = try container.decode(Int.self, forKey: .decoderSparseStep)
        self.sharedExpertIntermediateSize = try container.decode(
            Int.self, forKey: .sharedExpertIntermediateSize)
        self.mlpOnlyLayers = try container.decodeIfPresent([Int].self, forKey: .mlpOnlyLayers) ?? []
        self.moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        self.partialRotaryFactor =
            try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 1.0
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        self.normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? false
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.fullAttentionInterval =
            try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4
    }
}

// MARK: - LoRA

extension Qwen3NextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
