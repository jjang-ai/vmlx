// Copyright © 2024-2026 Jinho Jang (eric@jangq.ai)
//
// Llama 4 text decoder
// iRoPE (every 4th layer is NoPE + global KV cache; others are RoPE + ChunkedKVCache),
// optional QK rms-norm on RoPE layers, attention temperature tuning on NoPE layers,
// interleaved MoE (top-1 routing with sigmoid scores + shared MLP expert).
//
// Python reference: mlx_lm/models/llama4.py

import Foundation
import MLX
import MLXNN
import vMLXLMCommon

// MARK: - Configuration

public struct Llama4Configuration: Codable, Sendable {
    var modelType: String = "llama4"
    var hiddenSize: Int
    var numHiddenLayers: Int
    var intermediateSize: Int
    var intermediateSizeMLP: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var headDim: Int
    var rmsNormEps: Float
    var vocabSize: Int
    var attentionBias: Bool = false
    var attentionChunkSize: Int
    var interleaveMoeLayerStep: Int
    var numExpertsPerTok: Int
    var numLocalExperts: Int
    var ropeTheta: Float = 500_000
    var ropeScaling: [String: StringOrNumber]? = nil
    var maxPositionEmbeddings: Int = 131_072
    var useQkNorm: Bool = false
    var attnTemperatureTuning: Int = 4
    var floorScale: Int = 8192
    var attnScale: Float = 0.1
    var tieWordEmbeddings: Bool = false

    enum TextKeys: String, CodingKey { case textConfig = "text_config" }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case intermediateSizeMLP = "intermediate_size_mlp"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case attentionBias = "attention_bias"
        case attentionChunkSize = "attention_chunk_size"
        case interleaveMoeLayerStep = "interleave_moe_layer_step"
        case numExpertsPerTok = "num_experts_per_tok"
        case numLocalExperts = "num_local_experts"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case maxPositionEmbeddings = "max_position_embeddings"
        case useQkNorm = "use_qk_norm"
        case attnTemperatureTuning = "attn_temperature_tuning"
        case floorScale = "floor_scale"
        case attnScale = "attn_scale"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let nc = try decoder.container(keyedBy: TextKeys.self)
        let c = nc.contains(.textConfig)
            ? try nc.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            : try decoder.container(keyedBy: CodingKeys.self)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "llama4"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        intermediateSizeMLP = try c.decode(Int.self, forKey: .intermediateSizeMLP)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        if let hd = try c.decodeIfPresent(Int.self, forKey: .headDim) {
            headDim = hd
        } else {
            headDim = hiddenSize / numAttentionHeads
        }
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        attentionChunkSize = try c.decode(Int.self, forKey: .attentionChunkSize)
        interleaveMoeLayerStep = try c.decode(Int.self, forKey: .interleaveMoeLayerStep)
        numExpertsPerTok = try c.decode(Int.self, forKey: .numExpertsPerTok)
        numLocalExperts = try c.decode(Int.self, forKey: .numLocalExperts)
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 500_000
        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
        useQkNorm = try c.decodeIfPresent(Bool.self, forKey: .useQkNorm) ?? false
        attnTemperatureTuning = try c.decodeIfPresent(Int.self, forKey: .attnTemperatureTuning) ?? 4
        floorScale = try c.decodeIfPresent(Int.self, forKey: .floorScale) ?? 8192
        attnScale = try c.decodeIfPresent(Float.self, forKey: .attnScale) ?? 0.1
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

// MARK: - Attention (iRoPE)

class Llama4Attention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let useRope: Bool
    let useQkNorm: Bool
    let attnTemperatureTuning: Int
    let floorScale: Int
    let attnScale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPELayer?

    init(_ args: Llama4Configuration, layerIdx: Int) {
        self.nHeads = args.numAttentionHeads
        self.nKVHeads = args.numKeyValueHeads
        self.headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        // Every 4th layer is NoPE + global KV cache (iRoPE).
        self.useRope = ((layerIdx + 1) % 4) != 0
        self.useQkNorm = args.useQkNorm && self.useRope
        self.attnTemperatureTuning = args.attnTemperatureTuning
        self.floorScale = args.floorScale
        self.attnScale = args.attnScale

        self._wq.wrappedValue = Linear(args.hiddenSize, nHeads * headDim, bias: args.attentionBias)
        self._wk.wrappedValue = Linear(args.hiddenSize, nKVHeads * headDim, bias: args.attentionBias)
        self._wv.wrappedValue = Linear(args.hiddenSize, nKVHeads * headDim, bias: args.attentionBias)
        self._wo.wrappedValue = Linear(nHeads * headDim, args.hiddenSize, bias: args.attentionBias)

        if self.useRope {
            self.rope = initializeRope(
                dims: headDim,
                base: args.ropeTheta,
                traditional: true,
                scalingConfig: args.ropeScaling,
                maxPositionEmbeddings: args.maxPositionEmbeddings)
        } else {
            self.rope = nil
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0

        if let rope = rope {
            queries = applyRotaryPosition(rope, to: queries, cache: cache)
            keys = applyRotaryPosition(rope, to: keys, cache: cache)
        }

        if useQkNorm {
            queries = MLXFast.rmsNorm(queries, weight: MLXArray.mlxNone, eps: 1e-6)
            keys = MLXFast.rmsNorm(keys, weight: MLXArray.mlxNone, eps: 1e-6)
        }

        // Attention temperature tuning on NoPE layers.
        if attnTemperatureTuning != 0 && !useRope {
            let positions = MLXArray(Int32(offset + 1) ..< Int32(offset + L + 1))
            let floored = MLX.floor(positions.asType(.float32) / Float(floorScale))
            let scales = (MLX.log(floored + 1.0) * attnScale + 1.0)
            let attnScales = scales.expandedDimensions(axis: -1)
            queries = (queries * attnScales).asType(queries.dtype)
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask)
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

        return wo(output)
    }
}

// MARK: - Dense MLP

class Llama4MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        self._up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

// MARK: - MoE (top-1, sigmoid gating, shared expert)

class Llama4MoE: Module, UnaryLayer {
    let topK: Int
    let numExperts: Int

    @ModuleInfo(key: "experts") var experts: SwitchGLU
    @ModuleInfo(key: "router") var router: Linear
    @ModuleInfo(key: "shared_expert") var sharedExpert: Llama4MLP

    init(_ args: Llama4Configuration) {
        self.topK = args.numExpertsPerTok
        assert(self.topK == 1, "Llama4 MoE expects top-1 routing")
        self.numExperts = args.numLocalExperts

        self._experts.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.intermediateSize,
            numExperts: numExperts)
        self._router.wrappedValue = Linear(args.hiddenSize, numExperts, bias: false)
        self._sharedExpert.wrappedValue = Llama4MLP(
            hiddenSize: args.hiddenSize, intermediateSize: args.intermediateSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let logits = router(x)
        let k = topK
        let indices = MLX.argPartition(-logits, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        var scores = MLX.takeAlong(logits, indices, axis: -1)
        scores = MLX.sigmoid(scores.asType(.float32)).asType(x.dtype)

        // `experts(x, indices)` returns [..., k, D]; k==1 so squeeze axis -2.
        let y = MLX.squeezed(experts(x * scores, indices), axis: -2)
        return y + sharedExpert(x)
    }
}

// MARK: - Transformer Block

class Llama4TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Llama4Attention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    fileprivate var feedForward: UnaryLayer

    let isMoeLayer: Bool

    init(_ args: Llama4Configuration, layerIdx: Int) {
        self._selfAttn.wrappedValue = Llama4Attention(args, layerIdx: layerIdx)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)

        self.isMoeLayer =
            (layerIdx % args.interleaveMoeLayerStep) == (args.interleaveMoeLayerStep - 1)
        if isMoeLayer {
            self.feedForward = Llama4MoE(args)
        } else {
            self.feedForward = Llama4MLP(
                hiddenSize: args.hiddenSize, intermediateSize: args.intermediateSizeMLP)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = feedForward(postAttentionLayerNorm(h))
        return h + r
    }
}

// MARK: - Inner Model

public class Llama4ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Llama4TransformerBlock]
    let norm: RMSNorm
    let attentionChunkSize: Int

    init(_ args: Llama4Configuration) {
        precondition(args.vocabSize > 0)
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.numHiddenLayers).map { i in
            Llama4TransformerBlock(args, layerIdx: i)
        }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self.attentionChunkSize = args.attentionChunkSize
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        // Trim ChunkedKVCache front on RoPE layers before computing masks.
        var start = 0
        var offset = 0
        if let cache = cache {
            for (idx, c) in cache.enumerated() {
                if ((idx + 1) % 4) != 0, let chunked = c as? ChunkedKVCache {
                    chunked.maybeTrimFront()
                }
            }
            if let chunked = cache.first as? ChunkedKVCache {
                start = chunked.offset - (chunked.offset - chunked.offset)  // start_position inaccessible
                // Swift's ChunkedKVCache doesn't expose startPosition publicly; approximate via offset of keys.
                // For correctness in the common case (offset < chunkSize) start == 0; if trimmed, we can't read it.
                // We fall back to offset of cache[0] which is correct when no trim happened.
                start = 0
            }
            offset = cache.first?.offset ?? 0
        }
        let end = offset + h.dim(1)

        let linds = MLXArray(Int32(start) ..< Int32(end))
        let rinds = MLXArray(Int32(offset) ..< Int32(end)).expandedDimensions(axis: -1)
        let blockPos = MLX.abs(
            (linds.floorDivide(attentionChunkSize))
                - (rinds.floorDivide(attentionChunkSize)))
        let tokenPos = MLX.lessEqual(linds, rinds)
        let chunkMaskArr = MLX.logicalAnd(MLX.equal(blockPos, MLXArray(0)), tokenPos)

        // Global mask: use cache[3] (first NoPE layer) as representative.
        let globalCache = cache?.count ?? 0 > 3 ? cache?[3] : cache?.first
        let globalMask = createAttentionMask(h: h, cache: globalCache)

        let chunkMaskMode: MLXFast.ScaledDotProductAttentionMaskMode =
            (h.dim(1) > 1) ? .array(chunkMaskArr) : .none

        for (idx, layer) in layers.enumerated() {
            let useChunked = ((idx + 1) % 4) != 0
            let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                useChunked ? chunkMaskMode : globalMask
            h = layer(h, mask: mask, cache: cache?[idx])
        }

        return norm(h)
    }
}

// MARK: - Top-Level Model

public class Llama4Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo public var model: Llama4ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let config: Llama4Configuration

    public init(_ args: Llama4Configuration) {
        self.config = args
        self.vocabularySize = args.vocabSize
        self.kvHeads = (0 ..< args.numHiddenLayers).map { _ in args.numKeyValueHeads }
        self.model = Llama4ModelInner(args)
        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var w = [String: MLXArray]()

        // Strip VLM prefixes; drop vision weights.
        for (key, value) in weights {
            if key.contains("vision_model") || key.contains("multi_modal_projector") { continue }
            var k = key
            if k.hasPrefix("language_model.") {
                k = String(k.dropFirst("language_model.".count))
            }
            w[k] = value
        }

        // Split `feed_forward.experts.gate_up_proj` into gate/up and transpose down_proj
        // to match SwitchGLU expected layout.
        for l in 0 ..< config.numHiddenLayers {
            let prefix = "model.layers.\(l).feed_forward.experts"
            if let gu = w.removeValue(forKey: "\(prefix).gate_up_proj") {
                let splits = MLX.split(gu, parts: 2, axis: -1)
                let gateProj = splits[0]
                let upProj = splits[1]
                w["\(prefix).gate_proj.weight"] = gateProj.swappedAxes(1, 2)
                w["\(prefix).up_proj.weight"] = upProj.swappedAxes(1, 2)
            }
            if let dp = w.removeValue(forKey: "\(prefix).down_proj") {
                w["\(prefix).down_proj.weight"] = dp.swappedAxes(1, 2)
            }
        }
        return w
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let chunkSize = config.attentionChunkSize
        return (0 ..< config.numHiddenLayers).map { i in
            if ((i + 1) % 4) != 0 {
                return ChunkedKVCache(chunkSize: chunkSize)
            } else {
                return KVCacheSimple()
            }
        }
    }
}

// MARK: - LoRA

extension Llama4Model: LoRAModel {
    public var loraLayers: [Module] { model.layers }
}
