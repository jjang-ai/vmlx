// Copyright © 2024-2026 Jinho Jang (eric@jangq.ai)
//
// Gemma 4 VLM — vision-language model with:
//   - Linear patch embedding with 2D position embeddings
//   - 2D multidimensional RoPE for vision encoder
//   - VisionPooler for downsampling patches
//   - MultimodalEmbedder projecting vision features into text space
//   - Full Gemma4 text decoder (MoE 26B or Dense 31B)
//
// Python reference: mlx_vlm/models/gemma4/

import CoreImage
import Foundation
import MLX
import vMLXLMCommon
import MLXNN

// Compiled logit softcap — fuses divide + tanh + multiply into one Metal dispatch.
private let compiledLogitSoftcap: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { (x: MLXArray, cap: MLXArray) -> MLXArray in
        tanh(x / cap) * cap
    }
    return HardwareInfo.isCompiledDecodeSupported ? compile(shapeless: true, body) : body
}()


// MARK: - Shared Norm Utilities

/// Standard Gemma4 RMSNorm — weight used directly, NO +1 offset
private class G4RMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float
    init(dimensions: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

/// Vision RMSNorm — full float32 computation for precision
private class VisionRMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float
    init(dimensions: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let v = (xf * xf).mean(axis: -1, keepDims: true)
        return ((xf * rsqrt(v + eps)) * weight.asType(.float32)).asType(x.dtype)
    }
}

/// Parameterless RMS normalization
private func rmsNormNoScale(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    let v = (x * x).mean(axis: -1, keepDims: true)
    return x * rsqrt(v + eps)
}

private func visionRmsNormNoScale(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    let xf = x.asType(.float32)
    let v = (xf * xf).mean(axis: -1, keepDims: true)
    return (xf * rsqrt(v + eps)).asType(x.dtype)
}

// MARK: - Configurations

public struct Gemma4VisionConfig: Codable, Sendable {
    let hiddenSize: Int
    let intermediateSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let rmsNormEps: Float
    let patchSize: Int
    let positionEmbeddingSize: Int
    let defaultOutputLength: Int
    let poolingKernelSize: Int
    let standardize: Bool
    let useClippedLinears: Bool
    let ropeTheta: Float

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case patchSize = "patch_size"
        case positionEmbeddingSize = "position_embedding_size"
        case defaultOutputLength = "default_output_length"
        case poolingKernelSize = "pooling_kernel_size"
        case standardize
        case useClippedLinears = "use_clipped_linears"
    }

    enum TopKeys: String, CodingKey {
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 16
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 12
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 12
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        positionEmbeddingSize = try c.decodeIfPresent(Int.self, forKey: .positionEmbeddingSize) ?? 10240
        defaultOutputLength = try c.decodeIfPresent(Int.self, forKey: .defaultOutputLength) ?? 280
        poolingKernelSize = try c.decodeIfPresent(Int.self, forKey: .poolingKernelSize) ?? 3
        standardize = try c.decodeIfPresent(Bool.self, forKey: .standardize) ?? false
        useClippedLinears = try c.decodeIfPresent(Bool.self, forKey: .useClippedLinears) ?? false

        if let rc = try? decoder.container(keyedBy: TopKeys.self),
           let rp = try? rc.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters),
           let t = rp["rope_theta"]?.asFloat()
        {
            ropeTheta = t
        } else {
            ropeTheta = 100.0
        }
    }
}

/// Inline text config for VLM — mirrors vMLXLLM's Gemma4TextConfiguration
struct G4TextConfig: Codable, Sendable {
    let hiddenSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let numGlobalKeyValueHeads: Int?
    let headDim: Int
    let globalHeadDim: Int
    let intermediateSize: Int
    let vocabSize: Int
    let rmsNormEps: Float
    let slidingWindow: Int
    let layerTypes: [String]
    let finalLogitSoftcapping: Float?
    let tieWordEmbeddings: Bool
    let attentionBias: Bool
    let attentionKEqV: Bool
    let hiddenSizePerLayerInput: Int
    let vocabSizePerLayerInput: Int
    let numKvSharedLayers: Int
    let useDoubleWideMlp: Bool
    let enableMoeBlock: Bool
    let moeIntermediateSize: Int
    let numExperts: Int
    let topKExperts: Int
    let ropeTraditional: Bool
    let ropeParameters: [String: [String: StringOrNumber]]

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case intermediateSize = "intermediate_size"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case attentionKEqV = "attention_k_eq_v"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case numKvSharedLayers = "num_kv_shared_layers"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case enableMoeBlock = "enable_moe_block"
        case moeIntermediateSize = "moe_intermediate_size"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case ropeTraditional = "rope_traditional"
        case ropeParameters = "rope_parameters"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2816
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 30
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        numGlobalKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numGlobalKeyValueHeads)
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        globalHeadDim = try c.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2112
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 262144
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 1024
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        finalLogitSoftcapping = try c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        attentionKEqV = try c.decodeIfPresent(Bool.self, forKey: .attentionKEqV) ?? false
        hiddenSizePerLayerInput = try c.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 0
        vocabSizePerLayerInput = try c.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput) ?? 0
        numKvSharedLayers = try c.decodeIfPresent(Int.self, forKey: .numKvSharedLayers) ?? 0
        useDoubleWideMlp = try c.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp) ?? false
        enableMoeBlock = try c.decodeIfPresent(Bool.self, forKey: .enableMoeBlock) ?? false
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        topKExperts = try c.decodeIfPresent(Int.self, forKey: .topKExperts) ?? 0
        ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        ropeParameters = try c.decodeIfPresent([String: [String: StringOrNumber]].self, forKey: .ropeParameters) ?? [:]
    }
}

public struct Gemma4Configuration: Codable, Sendable {
    let textConfig: G4TextConfig
    let visionConfig: Gemma4VisionConfig
    let modelType: String
    let imageTokenId: Int
    let visionSoftTokensPerImage: Int
    let quantization: BaseConfiguration.Quantization?

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case modelType = "model_type"
        case imageTokenId = "image_token_id"
        case visionSoftTokensPerImage = "vision_soft_tokens_per_image"
        case quantization
    }
}

// MARK: - Vision Components

private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    // Unary `-` preserves x.dtype (bf16 stays bf16). Scalar `MLXArray(0)`
    // defaults to fp32 and force-promotes bf16 operands, triggering an
    // AsType cascade (~30 ops/token across MoE). See §27 +
    // mlp_bfloat16_upcast.md. Keep unary form.
    return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
}

private let gemma4VisionDebugEnabled: Bool = {
    let env = ProcessInfo.processInfo.environment
    return env["VMLX_GEMMA4_VISION_DEBUG"] == "1"
        || env["VMLINUX_GEMMA4_VISION_DEBUG"] == "1"
}()

private func gemma4RawGeluApproximate(_ x: MLXArray) -> MLXArray {
    0.5 * x * (1 + tanh(sqrt(2 / Float.pi) * (x + 0.044715 * x * x * x)))
}

private func applyMultidimensionalRope(_ inputs: MLXArray, positions: MLXArray, base: Float) -> MLXArray {
    if gemma4VisionDebugEnabled {
        FileHandle.standardError.write(Data(
            "[Gemma4Vision] rope inputs.shape=\(inputs.shape) positions.shape=\(positions.shape) inputs.ndim=\(inputs.ndim) positions.ndim=\(positions.ndim)\n".utf8))
    }
    guard inputs.ndim > 0, positions.ndim > 0 else {
        return inputs
    }
    let headDim = inputs.dim(-1)
    let ndim = positions.dim(-1)
    guard headDim > 0, ndim > 0 else {
        return inputs
    }
    let chPerDim = 2 * (headDim / (2 * ndim))
    guard chPerDim > 0 else {
        return inputs
    }
    let halfPerDim = chPerDim / 2

    var parts: [MLXArray] = []
    let rotaryDim = min(headDim, chPerDim * ndim)
    for d in 0 ..< ndim {
        let start = d * chPerDim
        let end = min((d + 1) * chPerDim, rotaryDim)
        guard start < end else { break }
        let xPart = inputs[.ellipsis, start ..< end]
        let freqExp = (2.0 / Float(chPerDim)) * MLXArray(0 ..< halfPerDim).asType(.float32)
        let timescale = pow(base, freqExp)
        let sinInp = positions[.ellipsis, d ..< (d + 1)].asType(.float32) / timescale
        var cosD = cos(sinInp)
        var sinD = sin(sinInp)
        cosD = concatenated([cosD, cosD], axis: -1).asType(inputs.dtype)
        sinD = concatenated([sinD, sinD], axis: -1).asType(inputs.dtype)
        cosD = expandedDimensions(cosD, axis: 2)
        sinD = expandedDimensions(sinD, axis: 2)
        parts.append(xPart * cosD + rotateHalf(xPart) * sinD)
    }
    if rotaryDim < headDim {
        parts.append(inputs[.ellipsis, rotaryDim ..< headDim])
    }
    return concatenated(parts, axis: -1)
}

private func oneHot(_ indices: MLXArray, numClasses: Int) -> MLXArray {
    (expandedDimensions(indices, axis: -1) .== MLXArray(0 ..< Int32(numClasses))).asType(.float32)
}

// Vision Attention
private class VisionAttn: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let ropeBase: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: VisionRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: VisionRMSNorm

    init(_ cfg: Gemma4VisionConfig) {
        numHeads = cfg.numAttentionHeads
        numKVHeads = cfg.numKeyValueHeads
        headDim = cfg.headDim
        ropeBase = cfg.ropeTheta
        _qProj.wrappedValue = Linear(cfg.hiddenSize, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(cfg.hiddenSize, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(cfg.hiddenSize, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, cfg.hiddenSize, bias: false)
        _qNorm.wrappedValue = VisionRMSNorm(dimensions: headDim)
        _kNorm.wrappedValue = VisionRMSNorm(dimensions: headDim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray?) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        if gemma4VisionDebugEnabled {
            FileHandle.standardError.write(Data(
                "[Gemma4Vision] attn input.shape=\(x.shape) positions.shape=\(positions.shape) B=\(B) L=\(L) heads=\(numHeads) kvHeads=\(numKVHeads) headDim=\(headDim)\n".utf8))
        }
        var q = qProj(x).reshaped(B, L, numHeads, headDim)
        var k = kProj(x).reshaped(B, L, numKVHeads, headDim)
        var v = vProj(x).reshaped(B, L, numKVHeads, headDim)
        q = qNorm(q); k = kNorm(k); v = visionRmsNormNoScale(v)
        if gemma4VisionDebugEnabled {
            FileHandle.standardError.write(Data(
                "[Gemma4Vision] attn q.shape=\(q.shape) k.shape=\(k.shape) v.shape=\(v.shape)\n".utf8))
        }
        q = applyMultidimensionalRope(q, positions: positions, base: ropeBase)
        k = applyMultidimensionalRope(k, positions: positions, base: ropeBase)
        q = q.transposed(0, 2, 1, 3); k = k.transposed(0, 2, 1, 3); v = v.transposed(0, 2, 1, 3)
        // Fixes vmlx #52: Gemma 4 vision tower weights are float16 and attention
        // scores can exceed ±65504, producing -inf → NaN propagation through
        // embed_vision → model emits only <pad> tokens. Promote Q/K/V to float32
        // for the SDPA, then cast back. Mirrors the Python v1.3.29 patch.
        let origDType = q.dtype
        if origDType == .float16 {
            q = q.asType(.float32)
            k = k.asType(.float32)
            v = v.asType(.float32)
        }
        var out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0,
            mask: mask != nil ? .array(mask!) : .none)
        if origDType == .float16 {
            out = out.asType(.float16)
        }
        return oProj(out.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

private class VisionMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    init(_ cfg: Gemma4VisionConfig) {
        _gateProj.wrappedValue = Linear(cfg.hiddenSize, cfg.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(cfg.hiddenSize, cfg.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(cfg.intermediateSize, cfg.hiddenSize, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Vision prefill runs inside a larger generation graph. Avoid the
        // compiled single-array GELU wrapper here; on this path it can return
        // an empty result vector and trap in compile(...)[0].
        downProj(gemma4RawGeluApproximate(gateProj(x)) * upProj(x))
    }
}

private class VisionBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: VisionAttn
    @ModuleInfo var mlp: VisionMLP
    @ModuleInfo(key: "input_layernorm") var inputLN: VisionRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttnLN: VisionRMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFFLN: VisionRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFFLN: VisionRMSNorm

    init(_ cfg: Gemma4VisionConfig) {
        _selfAttn.wrappedValue = VisionAttn(cfg)
        self.mlp = VisionMLP(cfg)
        _inputLN.wrappedValue = VisionRMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        _postAttnLN.wrappedValue = VisionRMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        _preFFLN.wrappedValue = VisionRMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        _postFFLN.wrappedValue = VisionRMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x + postAttnLN(selfAttn(inputLN(x), positions: positions, mask: mask))
        h = h + postFFLN(mlp(preFFLN(h)))
        return h
    }
}

private class VisionPatchEmbedder: Module {
    let patchSize: Int
    let posEmbSize: Int
    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ModuleInfo(key: "position_embedding_table") var posTable: MLXArray

    init(_ cfg: Gemma4VisionConfig) {
        patchSize = cfg.patchSize
        posEmbSize = cfg.positionEmbeddingSize
        _inputProj.wrappedValue = Linear(3 * cfg.patchSize * cfg.patchSize, cfg.hiddenSize, bias: false)
        _posTable.wrappedValue = MLXArray.ones([2, cfg.positionEmbeddingSize, cfg.hiddenSize])
        super.init()
    }

    func callAsFunction(pixels: MLXArray, patchPos: MLXArray, padPos: MLXArray) -> MLXArray {
        let (B, C, H, W) = (pixels.dim(0), pixels.dim(1), pixels.dim(2), pixels.dim(3))
        let p = patchSize
        let usableH = (H / p) * p
        let usableW = (W / p) * p
        if gemma4VisionDebugEnabled {
            FileHandle.standardError.write(Data(
                "[Gemma4Vision] patch pixels.shape=\(pixels.shape) patchPos.shape=\(patchPos.shape) padPos.shape=\(padPos.shape) weight.shape=\(inputProj.weight.shape) B=\(B) C=\(C) H=\(H) W=\(W) usableH=\(usableH) usableW=\(usableW) patch=\(patchSize)\n".utf8))
        }
        precondition(usableH > 0 && usableW > 0, "Gemma4 image is smaller than one vision patch")
        let patchPixels = (usableH == H && usableW == W)
            ? pixels
            : pixels[0..., 0..., ..<usableH, ..<usableW]
        let patches = patchPixels.reshaped(B, C, usableH / p, p, usableW / p, p)
            .transposed(0, 2, 4, 3, 5, 1).reshaped(B, (usableH / p) * (usableW / p), C * p * p)
        let normalized = 2 * (patches - 0.5)
        let embedded = inputProj(normalized.asType(inputProj.weight.dtype))

        let oh = oneHot(patchPos, numClasses: posEmbSize)
            .transposed(0, 2, 1, 3).asType(posTable.dtype)
        var posEmb = matmul(oh, posTable).sum(axis: 1)
        posEmb = MLX.where(expandedDimensions(padPos, axis: -1), MLXArray(Float(0), dtype: posEmb.dtype), posEmb)
        if gemma4VisionDebugEnabled {
            FileHandle.standardError.write(Data(
                "[Gemma4Vision] patch patches.shape=\(patches.shape) embedded.shape=\(embedded.shape) posEmb.shape=\(posEmb.shape)\n".utf8))
        }
        return embedded + posEmb
    }
}

private class VisionPooler: Module {
    let defaultLen: Int
    let rootH: Float
    init(_ cfg: Gemma4VisionConfig) {
        defaultLen = cfg.defaultOutputLength
        rootH = sqrt(Float(cfg.hiddenSize))
        super.init()
    }
    func callAsFunction(_ h: MLXArray, patchPos: MLXArray, padPos: MLXArray) -> (MLXArray, MLXArray) {
        let L = h.dim(1)
        if L == defaultLen { return (h * rootH, logicalNot(padPos)) }
        let k = Int(sqrt(Float(L / defaultLen)))
        let kSq = Float(k * k)
        let clamped = maximum(patchPos, MLXArray(Int32(0)))
        let maxX = clamped[.ellipsis, 0].max(axis: -1, keepDims: true) + 1
        let ki = floor(clamped.asType(.float32) / Float(k)).asType(.int32)
        let linearIdx = ki[.ellipsis, 0] + (maxX / MLXArray(Int32(k))) * ki[.ellipsis, 1]
        let w = oneHot(linearIdx, numClasses: defaultLen) / kSq
        let out = matmul(w.transposed(0, 2, 1), h)
        let mask = logicalNot(all(w .== Float(0), axis: 1))
        return (out.asType(h.dtype) * rootH, mask)
    }
}

private class VisionEncoder: Module {
    @ModuleInfo var layers: [VisionBlock]
    init(_ cfg: Gemma4VisionConfig) {
        _layers.wrappedValue = (0 ..< cfg.numHiddenLayers).map { _ in VisionBlock(cfg) }
        super.init()
    }
    func callAsFunction(_ x: MLXArray, pos: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x; for l in layers { h = l(h, positions: pos, mask: mask) }; return h
    }
}

private class VisionTower: Module {
    let cfg: Gemma4VisionConfig
    let maxPatches: Int
    @ModuleInfo(key: "patch_embedder") var patchEmb: VisionPatchEmbedder
    @ModuleInfo var encoder: VisionEncoder
    @ModuleInfo var pooler: VisionPooler
    @ModuleInfo(key: "std_bias") var stdBias: MLXArray?
    @ModuleInfo(key: "std_scale") var stdScale: MLXArray?

    init(_ cfg: Gemma4VisionConfig) {
        self.cfg = cfg
        maxPatches = cfg.defaultOutputLength * cfg.poolingKernelSize * cfg.poolingKernelSize
        _patchEmb.wrappedValue = VisionPatchEmbedder(cfg)
        self.encoder = VisionEncoder(cfg)
        self.pooler = VisionPooler(cfg)
        if cfg.standardize { _stdBias.wrappedValue = MLXArray.zeros([cfg.hiddenSize]); _stdScale.wrappedValue = MLXArray.ones([cfg.hiddenSize]) }
        super.init()
    }

    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        let (B, _, H, W) = (pixels.dim(0), pixels.dim(1), pixels.dim(2), pixels.dim(3))
        let p = cfg.patchSize; let pH = H / p; let pW = W / p
        // Clamp to maxPatches to prevent Range crash if image is larger than expected
        let nReal = min(pH * pW, maxPatches); let nPad = maxPatches - nReal

        // Build position grid [nReal, 2] then expand to [B, nReal, 2]
        var posFlat = [Int32]()
        for y in 0 ..< pH { for x in 0 ..< pW { posFlat.append(Int32(x)); posFlat.append(Int32(y)) } }
        var patchPos = MLXArray(posFlat).reshaped(1, nReal, 2)
        patchPos = repeated(patchPos, count: B, axis: 0)
        var padPos = MLXArray.zeros([B, maxPatches]).asType(.bool)

        if nPad > 0 {
            let padFlat = [Int32](repeating: -1, count: nPad * 2)
            let pp = MLXArray(padFlat).reshaped(1, nPad, 2)
            patchPos = concatenated([patchPos, repeated(pp, count: B, axis: 0)], axis: 1)
            padPos = concatenated([MLXArray.zeros([B, nReal]).asType(.bool), MLXArray.ones([B, nPad]).asType(.bool)], axis: 1)
        }

        var emb = patchEmb(pixels: pixels, patchPos: patchPos[0..., ..<nReal], padPos: padPos[0..., ..<nReal])
        if nPad > 0 { emb = concatenated([emb, MLXArray.zeros([B, nPad, cfg.hiddenSize]).asType(emb.dtype)], axis: 1) }
        MLX.eval(emb)
        if gemma4VisionDebugEnabled {
            FileHandle.standardError.write(Data(
                "[Gemma4Vision] tower emb.shape.afterEval=\(emb.shape) nReal=\(nReal) nPad=\(nPad) maxPatches=\(maxPatches)\n".utf8))
        }

        let valid = logicalNot(padPos).asType(.float32)
        var mask = expandedDimensions(valid, axis: 1) * expandedDimensions(valid, axis: 2)
        let zeroVal = MLXArray(Float(0), dtype: emb.dtype)
        let negInfVal = MLXArray(Float(-1e9), dtype: emb.dtype)
        mask = MLX.where(mask .> MLXArray(Float(0), dtype: mask.dtype), zeroVal, negInfVal)
        mask = expandedDimensions(mask, axis: 1)

        var h = encoder(emb, pos: patchPos, mask: mask)
        let (pooled, _) = pooler(h, patchPos: patchPos, padPos: padPos)
        // Return all defaultOutputLength features — the processor inserts exactly
        // that many image tokens, so maskedScatter needs them all to match.
        h = pooled
        if cfg.standardize, let sb = stdBias, let ss = stdScale { h = (h - sb) * ss }
        return h
    }
}

// MARK: - ScaledLinear (for per-layer model projection)

private class G4ScaledLinear: Module {
    let weight: MLXArray; let scalar: Float
    init(inputDims: Int, outputDims: Int, scalar: Float) {
        self.weight = MLXArray.zeros([outputDims, inputDims]); self.scalar = scalar; super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { matmul(x, weight.T) * scalar }
}

// MARK: - Text Model Components (inline for VLM — vMLXVLM can't import vMLXLLM)

// Text Attention, MLP, Router, Experts, DecoderLayer, Model — same as Gemma4Text.swift
// but scoped privately within this file.

private class TextAttn: Module {
    let nH: Int; let nKV: Int; let hD: Int; let scale: Float; let isSliding: Bool; let useKEqV: Bool; let eps: Float
    @ModuleInfo(key: "q_proj") var qP: Linear
    @ModuleInfo(key: "k_proj") var kP: Linear
    @ModuleInfo(key: "v_proj") var vP: Linear?
    @ModuleInfo(key: "o_proj") var oP: Linear
    @ModuleInfo(key: "q_norm") var qN: G4RMSNorm
    @ModuleInfo(key: "k_norm") var kN: G4RMSNorm
    @ModuleInfo var rope: RoPELayer

    init(_ cfg: G4TextConfig, layerIndex: Int) {
        let lt = layerIndex < cfg.layerTypes.count ? cfg.layerTypes[layerIndex] : "sliding_attention"
        isSliding = lt == "sliding_attention"
        useKEqV = cfg.attentionKEqV && !isSliding
        eps = cfg.rmsNormEps
        if isSliding { nH = cfg.numAttentionHeads; nKV = cfg.numKeyValueHeads; hD = cfg.headDim }
        else { nH = cfg.numAttentionHeads; nKV = cfg.numGlobalKeyValueHeads ?? cfg.numKeyValueHeads; hD = cfg.globalHeadDim }
        scale = 1.0
        _qP.wrappedValue = Linear(cfg.hiddenSize, nH * hD, bias: cfg.attentionBias)
        _kP.wrappedValue = Linear(cfg.hiddenSize, nKV * hD, bias: cfg.attentionBias)
        if !useKEqV { _vP.wrappedValue = Linear(cfg.hiddenSize, nKV * hD, bias: cfg.attentionBias) }
        _oP.wrappedValue = Linear(nH * hD, cfg.hiddenSize, bias: cfg.attentionBias)
        _qN.wrappedValue = G4RMSNorm(dimensions: hD, eps: cfg.rmsNormEps)
        _kN.wrappedValue = G4RMSNorm(dimensions: hD, eps: cfg.rmsNormEps)
        let lk = isSliding ? "sliding_attention" : "full_attention"
        let rp = cfg.ropeParameters[lk] ?? [:]
        let rt = rp["rope_theta"]?.asFloat() ?? (isSliding ? 10000.0 : 1_000_000.0)
        let prf = rp["partial_rotary_factor"]?.asFloat() ?? (isSliding ? 1.0 : 0.25)
        self.rope = initializeRope(dims: max(1, Int(Float(hD) * prf)), base: rt, traditional: cfg.ropeTraditional, scalingConfig: nil, maxPositionEmbeddings: nil)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        sharedKV: (keys: MLXArray, values: MLXArray)? = nil, sharedOffset: Int? = nil,
        sharedOffsetArray: MLXArray? = nil
    ) -> (output: MLXArray, keys: MLXArray, values: MLXArray, offset: Int) {
        let (B, L) = (x.dim(0), x.dim(1))
        var q = qP(x).reshaped(B, L, nH, hD); q = qN(q); q = q.transposed(0, 2, 1, 3)
        let cK: MLXArray; let cV: MLXArray; let off: Int
        if let sharedKV {
            off = sharedOffset ?? 0
            if let sharedOffsetArray { q = rope(q, offset: sharedOffsetArray) }
            else { q = rope(q, offset: off) }
            cK = sharedKV.keys; cV = sharedKV.values
        } else {
            let normalOffsetArray = graphOffsetArray(for: cache)
            off = normalOffsetArray == nil ? (cache?.offset ?? 0) : 0
            var k = kP(x).reshaped(B, L, nKV, hD)
            let v: MLXArray
            if useKEqV { v = rmsNormNoScale(k, eps: eps) } else if let vP { v = rmsNormNoScale(vP(x).reshaped(B, L, nKV, hD), eps: eps) } else { v = rmsNormNoScale(k, eps: eps) }
            k = kN(k)
            let vT = v.transposed(0, 2, 1, 3); var kT = k.transposed(0, 2, 1, 3)
            kT = applyRotaryPosition(rope, to: kT, cache: cache)
            q = applyRotaryPosition(rope, to: q, cache: cache)
            if let cache { (cK, cV) = cache.update(keys: kT, values: vT) } else { (cK, cV) = (kT, vT) }
        }
        // Matches reference: Load.swift's convertToBFloat16 already casts
        // all fp16 params to bf16 at load time, so q/cK/cV arrive as bf16
        // (same exponent range as fp32). No SDPA upcast needed.
        let sdpa = MLXFast.scaledDotProductAttention(queries: q, keys: cK, values: cV, scale: scale, mask: mask)
        let out = sdpa.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return (oP(out), cK, cV, off)
    }
}

private class TextMLP: Module {
    @ModuleInfo(key: "gate_proj") var gP: Linear; @ModuleInfo(key: "up_proj") var uP: Linear; @ModuleInfo(key: "down_proj") var dP: Linear
    init(_ cfg: G4TextConfig, intermediateSize: Int? = nil) {
        let iS = intermediateSize ?? cfg.intermediateSize
        _gP.wrappedValue = Linear(cfg.hiddenSize, iS, bias: false); _uP.wrappedValue = Linear(cfg.hiddenSize, iS, bias: false); _dP.wrappedValue = Linear(iS, cfg.hiddenSize, bias: false); super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let g = safeGeluApproximate(gP(x))
        let u = uP(x)
        let product: MLXArray
        product = g * u
        return dP(product)
    }
}

private class TextRouter: Module {
    @ModuleInfo(key: "proj") var proj: Linear; @ModuleInfo(key: "scale") var sc: MLXArray; @ModuleInfo(key: "per_expert_scale") var pes: MLXArray
    let nE: Int; let topK: Int; let rs: Float; let eps: Float
    init(_ cfg: G4TextConfig) {
        nE = cfg.numExperts; topK = cfg.topKExperts; rs = pow(Float(cfg.hiddenSize), -0.5); eps = cfg.rmsNormEps
        _proj.wrappedValue = Linear(cfg.hiddenSize, cfg.numExperts, bias: false)
        _sc.wrappedValue = MLXArray.ones([cfg.hiddenSize]); _pes.wrappedValue = MLXArray.ones([cfg.numExperts])
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        // Keep the VLM inline text router in parity with
        // vMLXLLM/Models/Gemma4Text.swift and Python mlx_lm:
        // fused RMSNorm with scale*sqrt(hidden)^-1, then top-k on
        // raw expert scores, then softmax only over selected experts.
        let h = MLXFast.rmsNorm(x, weight: sc * rs, eps: eps)
        let s = proj(h)
        // Unary `-` avoids AsType cascade. See §27 + mlp_bfloat16_upcast.md.
        let ti = argPartition(-s, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        let topLogits = takeAlong(s, ti, axis: -1)
        var tw = softmax(topLogits, axis: -1)
        tw = tw * pes[ti]
        return (ti, tw)
    }
}

private class TextExperts: Module {
    @ModuleInfo(key: "switch_glu") var sg: SwitchGLU
    init(_ cfg: G4TextConfig) {
        _sg.wrappedValue = SwitchGLU(inputDims: cfg.hiddenSize, hiddenDims: cfg.moeIntermediateSize, numExperts: cfg.numExperts, activation: { safeGeluApproximate($0) }, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, idx: MLXArray, wts: MLXArray) -> MLXArray {
        let (B, S, H) = (x.dim(0), x.dim(1), x.dim(2)); let K = idx.dim(-1)
        // Inline reshape (matches reference). Previous version held
        // intermediate `xFlat`/`idxFlat` named bindings plus a Flash MoE
        // shim branch, which retained graph nodes and added per-step
        // overhead. Gemma4 measured 57 → ?? tok/s on mlx-community
        // gemma-4-26b-a4b-it-4bit after this inline.
        let o = sg(x.reshaped(B * S, H), idx.reshaped(B * S, K))
        return (o * expandedDimensions(wts.reshaped(B * S, K), axis: -1)).sum(axis: -2).reshaped(B, S, H)
    }
}

private class TextLayer: Module {
    let hasMoE: Bool
    let layerIndex: Int
    @ModuleInfo(key: "self_attn") var attn: TextAttn; @ModuleInfo var mlp: TextMLP
    @ModuleInfo var router: TextRouter?; @ModuleInfo var experts: TextExperts?
    @ModuleInfo(key: "input_layernorm") var iLN: G4RMSNorm; @ModuleInfo(key: "post_attention_layernorm") var paLN: G4RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var pfLN: G4RMSNorm; @ModuleInfo(key: "post_feedforward_layernorm") var pffLN: G4RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var pfLN2: G4RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var pffLN1: G4RMSNorm?; @ModuleInfo(key: "post_feedforward_layernorm_2") var pffLN2: G4RMSNorm?
    @ModuleInfo(key: "per_layer_input_gate") var pliGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var pliProj: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var pliNorm: G4RMSNorm?
    @ModuleInfo(key: "layer_scalar") var ls: MLXArray

    init(_ cfg: G4TextConfig, i: Int) {
        hasMoE = cfg.enableMoeBlock && cfg.numExperts > 0
        layerIndex = i
        _attn.wrappedValue = TextAttn(cfg, layerIndex: i)
        let fks = cfg.numHiddenLayers - cfg.numKvSharedLayers
        let isShared = cfg.numKvSharedLayers > 0 && i >= fks
        let iSize = (cfg.useDoubleWideMlp && isShared) ? cfg.intermediateSize * 2 : cfg.intermediateSize
        self.mlp = TextMLP(cfg, intermediateSize: iSize)
        if hasMoE {
            self.router = TextRouter(cfg); self.experts = TextExperts(cfg)
            _pfLN2.wrappedValue = G4RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
            _pffLN1.wrappedValue = G4RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
            _pffLN2.wrappedValue = G4RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        }
        if cfg.hiddenSizePerLayerInput > 0 {
            _pliGate.wrappedValue = Linear(cfg.hiddenSize, cfg.hiddenSizePerLayerInput, bias: false)
            _pliProj.wrappedValue = Linear(cfg.hiddenSizePerLayerInput, cfg.hiddenSize, bias: false)
            _pliNorm.wrappedValue = G4RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        }
        _iLN.wrappedValue = G4RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        _paLN.wrappedValue = G4RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        _pfLN.wrappedValue = G4RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        _pffLN.wrappedValue = G4RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        _ls.wrappedValue = MLXArray([Float(1.0)])
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        perLayerInput: MLXArray? = nil,
        sharedKV: (keys: MLXArray, values: MLXArray)? = nil, sharedOffset: Int? = nil,
        sharedOffsetArray: MLXArray? = nil
    ) -> (h: MLXArray, keys: MLXArray, values: MLXArray, offset: Int) {
        var r = x
        let (aOut, aK, aV, aOff) = attn(iLN(x), mask: mask, cache: cache, sharedKV: sharedKV, sharedOffset: sharedOffset, sharedOffsetArray: sharedOffsetArray)
        var h = paLN(aOut); h = r + h; r = h
        if hasMoE, let router, let experts, let pfLN2, let pffLN1, let pffLN2 {
            var h1 = mlp(pfLN(h)); h1 = pffLN1(h1)
            let (ti, tw) = router(h)
            JangPressRouteTelemetry.recordTopK(layer: layerIndex, indices: ti)
            var h2 = experts(pfLN2(h), idx: ti, wts: tw); h2 = pffLN2(h2)
            h = h1 + h2
        } else { h = mlp(pfLN(h)) }
        h = pffLN(h); h = r + h
        if let pliGate, let pliProj, let pliNorm, let perLayerInput {
            r = h; var g = safeGeluApproximate(pliGate(h)); g = g * perLayerInput
            g = pliProj(g); g = pliNorm(g); h = r + g
        }
        h = h * ls
        return (h, aK, aV, aOff)
    }
}

private class TextModel: Module {
    @ModuleInfo(key: "embed_tokens") var emb: Embedding; @ModuleInfo var layers: [TextLayer]; @ModuleInfo var norm: G4RMSNorm
    @ModuleInfo(key: "embed_tokens_per_layer") var embPL: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var plProj: G4ScaledLinear?
    @ModuleInfo(key: "per_layer_projection_norm") var plNorm: G4RMSNorm?
    let cfg: G4TextConfig
    let previousKVs: [Int]

    init(_ cfg: G4TextConfig) {
        self.cfg = cfg; _emb.wrappedValue = Embedding(embeddingCount: cfg.vocabSize, dimensions: cfg.hiddenSize)
        _layers.wrappedValue = (0 ..< cfg.numHiddenLayers).map { TextLayer(cfg, i: $0) }
        self.norm = G4RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        if cfg.hiddenSizePerLayerInput > 0 {
            _embPL.wrappedValue = Embedding(embeddingCount: cfg.vocabSizePerLayerInput,
                dimensions: cfg.numHiddenLayers * cfg.hiddenSizePerLayerInput)
            _plProj.wrappedValue = G4ScaledLinear(inputDims: cfg.hiddenSize,
                outputDims: cfg.numHiddenLayers * cfg.hiddenSizePerLayerInput,
                scalar: pow(Float(cfg.hiddenSize), -0.5))
            _plNorm.wrappedValue = G4RMSNorm(dimensions: cfg.hiddenSizePerLayerInput, eps: cfg.rmsNormEps)
        }
        let lt = cfg.layerTypes.isEmpty ? Array(repeating: "sliding_attention", count: cfg.numHiddenLayers) : cfg.layerTypes
        var pkvs = Array(0 ..< cfg.numHiddenLayers)
        if cfg.numKvSharedLayers > 0 {
            let fks = cfg.numHiddenLayers - cfg.numKvSharedLayers
            var byType: [String: Int] = [:]; for i in 0 ..< fks { byType[lt[i]] = i }
            for j in fks ..< cfg.numHiddenLayers { if let s = byType[lt[j]] { pkvs[j] = s } }
        }
        self.previousKVs = pkvs
        super.init()
    }

    private func getPerLayerInputs(_ ids: MLXArray) -> MLXArray? {
        guard let embPL else { return nil }
        let r = embPL(ids) * pow(Float(cfg.hiddenSizePerLayerInput), 0.5)
        return r.reshaped(Array(ids.shape) + [cfg.numHiddenLayers, cfg.hiddenSizePerLayerInput])
    }

    private func projectPerLayerInputs(_ h: MLXArray, pli: MLXArray?) -> MLXArray? {
        guard let plProj, let plNorm else { return nil }
        var p = plProj(h).reshaped(Array(h.shape.dropLast()) + [cfg.numHiddenLayers, cfg.hiddenSizePerLayerInput])
        p = plNorm(p)
        guard let pli else { return p }
        return (p + pli) * pow(Float(2.0), Float(-0.5))
    }

    func callAsFunction(_ inputs: MLXArray?, inputEmbedding: MLXArray? = nil, cache: [KVCache?]? = nil) -> MLXArray {
        // Ensure batch dimension — callers may pass 1D tokens [N] on cache-reuse turns
        let inputs = inputs.map { $0.ndim == 1 ? $0.expandedDimensions(axis: 0) : $0 }
        var h: MLXArray
        if let ie = inputEmbedding {
            h = ie.ndim == 2 ? ie.expandedDimensions(axis: 0) : ie
        } else {
            h = emb(inputs!) * MLXArray(sqrt(Float(cfg.hiddenSize)), dtype: emb.weight.dtype)
        }

        var pliList: [MLXArray?]
        if cfg.hiddenSizePerLayerInput > 0 {
            let raw = inputs.flatMap { getPerLayerInputs($0) }
            if let final = projectPerLayerInputs(h, pli: raw) {
                pliList = (0 ..< layers.count).map { final[0..., 0..., $0, 0...] }
            } else { pliList = Array(repeating: nil, count: layers.count) }
        } else { pliList = Array(repeating: nil, count: layers.count) }

        let lc = cache ?? Array(repeating: nil as KVCache?, count: layers.count)
        let lt = cfg.layerTypes.isEmpty ? Array(repeating: "sliding_attention", count: cfg.numHiddenLayers) : cfg.layerTypes
        let gIdx = lt.firstIndex(of: "full_attention") ?? (cfg.numHiddenLayers - 1)
        let sIdx = lt.firstIndex(of: "sliding_attention") ?? 0
        let gc: KVCache? = cache.flatMap { gIdx < $0.count ? $0[gIdx] : nil }
        let sc: KVCache? = cache.flatMap { sIdx < $0.count ? $0[sIdx] : nil }
        let gm = createAttentionMask(h: h, cache: gc); let sm = createAttentionMask(h: h, cache: sc, windowSize: cfg.slidingWindow)

        var intermediates: [(keys: MLXArray, values: MLXArray, offset: Int, offsetArray: MLXArray?)?] = Array(repeating: nil, count: layers.count)
        for (i, l) in layers.enumerated() {
            let isGlobal = (i < lt.count ? lt[i] : "sliding_attention") == "full_attention"
            let prevIdx = previousKVs[i]
            let skv: (keys: MLXArray, values: MLXArray)?; let soff: Int?; let soffArr: MLXArray?
            if prevIdx != i, let prev = intermediates[prevIdx] { skv = (prev.keys, prev.values); soff = prev.offset; soffArr = prev.offsetArray }
            else { skv = nil; soff = nil; soffArr = nil }
            let ce = prevIdx == i ? (i < lc.count ? lc[i] : nil) : nil
            let res = l(h, mask: isGlobal ? gm : sm, cache: ce, perLayerInput: pliList[i], sharedKV: skv, sharedOffset: soff, sharedOffsetArray: soffArr)
            let layerOffArr = graphOffsetArray(for: ce)
            h = res.h; intermediates[i] = (res.keys, res.values, res.offset, layerOffArr)
        }
        return norm(h)
    }
}

private class G4LanguageModel: Module {
    @ModuleInfo var model: TextModel; @ModuleInfo(key: "lm_head") var lmHead: Linear?
    let cfg: G4TextConfig
    init(_ cfg: G4TextConfig) {
        self.cfg = cfg; self.model = TextModel(cfg)
        if !cfg.tieWordEmbeddings { _lmHead.wrappedValue = Linear(cfg.hiddenSize, cfg.vocabSize, bias: false) }
        super.init()
    }
    func callAsFunction(_ inputs: MLXArray?, inputEmbedding: MLXArray? = nil, cache: [KVCache?]? = nil) -> MLXArray {
        var o = model(inputs, inputEmbedding: inputEmbedding, cache: cache)
        if let lh = lmHead { o = lh(o) } else { o = model.emb.asLinear(o) }
        if let cap = cfg.finalLogitSoftcapping, cap > 0 { o = compiledLogitSoftcap(o, MLXArray(cap, dtype: o.dtype)) }
        return o
    }
    func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        let fks = cfg.numKvSharedLayers > 0 ? cfg.numHiddenLayers - cfg.numKvSharedLayers : cfg.numHiddenLayers
        return (0 ..< fks).map { i in
            let lt = i < cfg.layerTypes.count ? cfg.layerTypes[i] : "sliding_attention"
            if lt == "full_attention" { return parameters?.maxKVSize.map { RotatingKVCache(maxSize: $0, keep: 4) } ?? KVCacheSimple() }
            else { return RotatingKVCache(maxSize: cfg.slidingWindow, keep: 0) }
        }
    }
}

// MARK: - Multimodal Embedder

private class MultimodalEmbedder: Module {
    @ModuleInfo(key: "embedding_projection") var proj: Linear
    init(embDim: Int, textDim: Int) { _proj.wrappedValue = Linear(embDim, textDim, bias: false); super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray { rmsNormNoScale(proj(x)) }
}

private func maskedScatter(input: MLXArray, mask: MLXArray, source: MLXArray) -> MLXArray {
    let inputShape = input.shape
    let inputFlat = input.flattened()
    let maskFlat = mask.flattened()
    let sourceFlat = source.flattened()

    let maskValues = maskFlat.asArray(Bool.self)
    let positions = maskValues.enumerated().compactMap { i, v in v ? UInt32(i) : nil }

    guard !positions.isEmpty else { return input }

    let posArray = MLXArray(positions)
    guard sourceFlat.shape[0] == posArray.shape[0] else {
        fatalError(
            """
            Gemma4 maskedScatter: size mismatch between vision features and image token positions.
            Vision features: \(sourceFlat.shape[0]), image positions: \(posArray.shape[0]).
            Check that imageSeqLength in preprocessor_config matches vision tower output (defaultOutputLength).
            """)
    }
    inputFlat[posArray] = sourceFlat
    return inputFlat.reshaped(inputShape)
}

// MARK: - Gemma4 VLM

public class Gemma4: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: VisionTower
    @ModuleInfo(key: "language_model") private var languageModel: G4LanguageModel
    @ModuleInfo(key: "embed_vision") private var embedVision: MultimodalEmbedder

    public let config: Gemma4Configuration
    public var vocabularySize: Int { config.textConfig.vocabSize }
    public var kvHeads: [Int] {
        let tc = config.textConfig
        return (0 ..< tc.numHiddenLayers).map { i in
            let lt = i < tc.layerTypes.count ? tc.layerTypes[i] : "sliding_attention"
            return lt == "full_attention" ? (tc.numGlobalKeyValueHeads ?? tc.numKeyValueHeads) : tc.numKeyValueHeads
        }
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] { languageModel.newCache(parameters: parameters) }

    public init(_ config: Gemma4Configuration) {
        self.config = config
        _visionTower.wrappedValue = VisionTower(config.visionConfig)
        _languageModel.wrappedValue = G4LanguageModel(config.textConfig)
        _embedVision.wrappedValue = MultimodalEmbedder(embDim: config.visionConfig.hiddenSize, textDim: config.textConfig.hiddenSize)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
        var emb = languageModel.model.emb(input.text.tokens)
        emb = emb * MLXArray(sqrt(Float(config.textConfig.hiddenSize)), dtype: emb.dtype)

        if let pixels = input.image?.pixels {
            // Process each image through vision tower separately — images may have
            // different spatial dimensions after resize. Vision features are always
            // [1, defaultOutputLength, visionHidden] per image regardless of input size.
            let B = pixels.dim(0)
            var featuresList = [MLXArray]()
            for i in 0 ..< B {
                // Extract image at its original dimensions (stored in frames)
                // to avoid processing zero-padded regions through the vision tower.
                if let frames = input.image?.frames, i < frames.count {
                    let h = frames[i].h; let w = frames[i].w
                    let singleImage = pixels[i, 0..., ..<h, ..<w].expandedDimensions(axis: 0)
                    featuresList.append(embedVision(visionTower(singleImage)))
                } else {
                    let singleImage = pixels[i].expandedDimensions(axis: 0)
                    featuresList.append(embedVision(visionTower(singleImage)))
                }
            }
            let imgFeatures = (B == 1 ? featuresList[0] : concatenated(featuresList)).asType(emb.dtype)

            let imgMask = MLX.equal(input.text.tokens, MLXArray(Int32(config.imageTokenId)))
            let imgMaskExp = MLX.broadcast(expandedDimensions(imgMask, axis: -1), to: emb.shape)
            emb = maskedScatter(input: emb, mask: imgMaskExp, source: imgFeatures)
        }

        let paddedCache = padCache(cache)
        // §335 — chunked prefill, prevents Metal-buffer overflow on
        // long prompts. Every other VLM in the tree already chunks via
        // `chunkedPrefillEmbedding` (see Qwen2VL / Qwen25VL / Gemma3
        // / FastVLM / Idefics3 / LFM2VL / Mistral3 / Mistral4VLM /
        // Pixtral / GlmOcr). Gemma4 was the last holdout — a single
        // full-sequence `languageModel(...)` forward produced
        // intermediate activations proportional to sequence length,
        // so a 40k+ token prompt (QwenCode / OpenCode CLI /init dump)
        // blew past the Metal max-buffer cap on 64-GB Macs
        // (mlxstudio #83: "Attempting to allocate 21.9 GB > 9.5 GB").
        //
        // Helper quirk: `chunkedPrefillEmbedding` takes `[KVCache]`
        // (non-nil), but Gemma4's `padCache` returns `[KVCache?]?`
        // with nil entries for layers without attention caches.
        // Walk the padded list and pass only the non-nil entries
        // to the helper for the materialization barrier — the
        // forward call inside `step` still receives the full padded
        // shape so layer→cache indexing remains aligned.
        let nonNilCache = (paddedCache ?? []).compactMap { $0 }
        let out = chunkedPrefillEmbedding(
            inputEmbedding: emb,
            cache: nonNilCache,
            prefillStepSize: windowSize ?? 512
        ) { chunk in
            languageModel(nil, inputEmbedding: chunk, cache: paddedCache)
        }
        return .logits(.init(logits: out))
    }

    private func padCache(_ cache: [any KVCache]?) -> [KVCache?]? {
        guard let c = cache else { return nil }
        let n = config.textConfig.numHiddenLayers
        // Fast path (perf audit 2026-04-16): cache is already full length —
        // avoid the per-decode-token array clone + map + concat. Qwen35B
        // doesn't have this wrapper and runs at 92% parity; Gemma4 VLM
        // had 30 allocations/step, ~3% decode tax on its own. Per-step
        // `callAsFunction` runs this code path.
        if c.count == n {
            // Need to widen `[any KVCache]` → `[KVCache?]` but keep refs.
            // Map is still O(N) but returns contiguous Array without append.
            return c.map { $0 as KVCache? }
        }
        return c.map { $0 as KVCache? } + Array(repeating: nil as KVCache?,
            count: max(0, n - c.count))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: padCache(cache))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var p = [String: MLXArray]()
        for (k, v) in weights {
            var nk = k
            if nk.hasPrefix("model.") { nk = String(nk.dropFirst("model.".count)) }
            // Skip audio — Gemma4 VLM doesn't implement audio, these weights have no module
            if nk.hasPrefix("audio_tower.") || nk.hasPrefix("embed_audio.") { continue }
            // Skip clipped linear params — training artifacts, not used in inference (we use plain Linear)
            if nk.contains("input_min") || nk.contains("input_max") || nk.contains("output_min") || nk.contains("output_max") { continue }
            if nk.contains("rotary_emb") { continue }
            // Remap language_model keys to include model. prefix
            if nk.hasPrefix("language_model.") && !nk.hasPrefix("language_model.model.") {
                nk = "language_model.model." + String(nk.dropFirst("language_model.".count))
            }
            if nk.contains(".switch_mlp.") { nk = nk.replacingOccurrences(of: ".switch_mlp.", with: ".experts.switch_glu.") }
            // Vision tower uses ClippableLinear wrappers — checkpoint has .linear. segment
            // that doesn't exist in our module tree (we use plain Linear)
            if nk.hasPrefix("vision_tower.") && nk.contains(".linear.") {
                nk = nk.replacingOccurrences(of: ".linear.", with: ".")
            }
            p[nk] = v
        }
        let ev = config.textConfig.vocabSize
        for k in ["language_model.model.embed_tokens.weight", "language_model.model.embed_tokens.scales", "language_model.model.embed_tokens.biases", "language_model.lm_head.weight", "language_model.lm_head.scales", "language_model.lm_head.biases"] {
            if let w = p[k], w.dim(0) != ev { p[k] = w[0 ..< ev] }
        }
        return p
    }
}

extension Gemma4: LoRAModel { public var loraLayers: [Module] { languageModel.model.layers } }

// MARK: - Processor

public struct Gemma4ProcessorConfiguration: Codable, Sendable {
    public let processorClass: String
    public let patchSize: Int
    public let maxSoftTokens: Int
    public let poolingKernelSize: Int
    public let imageSeqLength: Int
    public let audioSeqLength: Int

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case patchSize = "patch_size"
        case maxSoftTokens = "max_soft_tokens"
        case poolingKernelSize = "pooling_kernel_size"
        case imageSeqLength = "image_seq_length"
        case audioSeqLength = "audio_seq_length"
    }

    enum NestedKeys: String, CodingKey {
        case imageProcessor = "image_processor"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let n = try? decoder.container(keyedBy: NestedKeys.self)
        let image = try? n?.nestedContainer(keyedBy: CodingKeys.self, forKey: .imageProcessor)
        processorClass = try c.decodeIfPresent(String.self, forKey: .processorClass) ?? "Gemma4Processor"
        let nestedPatchSize = try image?.decodeIfPresent(Int.self, forKey: .patchSize)
        let nestedMaxSoftTokens = try image?.decodeIfPresent(Int.self, forKey: .maxSoftTokens)
        let nestedPoolingKernelSize = try image?.decodeIfPresent(Int.self, forKey: .poolingKernelSize)
        let nestedImageSeqLength = try image?.decodeIfPresent(Int.self, forKey: .imageSeqLength)
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? nestedPatchSize ?? 16
        maxSoftTokens = try c.decodeIfPresent(Int.self, forKey: .maxSoftTokens) ?? nestedMaxSoftTokens ?? 280
        poolingKernelSize = try c.decodeIfPresent(Int.self, forKey: .poolingKernelSize) ?? nestedPoolingKernelSize ?? 3
        imageSeqLength = try c.decodeIfPresent(Int.self, forKey: .imageSeqLength) ?? nestedImageSeqLength ?? 280
        audioSeqLength = try c.decodeIfPresent(Int.self, forKey: .audioSeqLength) ?? 750
    }
}

public struct Gemma4Processor: UserInputProcessor {
    private let config: Gemma4ProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Gemma4ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config; self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Qwen2VLMessageGenerator().generate(from: input)
        var tokens = try tokenizer.applyChatTemplate(messages: messages, tools: input.tools, additionalContext: input.additionalContext)

        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let ps = config.patchSize; let maxP = config.maxSoftTokens * config.poolingKernelSize * config.poolingKernelSize
            let arrays = try input.images.map { img -> MLXArray in
                let ci = try img.asCIImage()
                let (w, h) = (Int(ci.extent.width), Int(ci.extent.height))
                let f = sqrt(Float(maxP * ps * ps) / Float(w * h))
                let sm = config.poolingKernelSize * ps
                var tH = Int(floor(f * Float(h) / Float(sm))) * sm; var tW = Int(floor(f * Float(w) / Float(sm))) * sm
                if tH == 0 { tH = sm }; if tW == 0 { tW = sm }
                let resized = MediaProcessing.resampleBicubic(ci, to: CGSize(width: tW, height: tH))
                // Convert to sRGB tone curve — CIImage may be in linear space, but the
                // vision tower was trained on sRGB images (PIL/Python default).
                let srgb = MediaProcessing.inSRGBToneCurveSpace(resized)
                // asMLXArray returns [1, C, H, W] (NCHW) with float values in [0, 1]
                return MediaProcessing.asMLXArray(srgb)
            }
            // Store per-image dimensions in frames so prepare() can extract each
            // image at its original size (before padding for batch storage).
            let imageSizes = arrays.map { THW(1, $0.dim(2), $0.dim(3)) }
            if arrays.count == 1 {
                processedImage = LMInput.ProcessedImage(pixels: arrays[0], frames: imageSizes)
            } else {
                // Pad to max dims for storage in a single batched tensor
                let maxH = arrays.map { $0.dim(2) }.max()!
                let maxW = arrays.map { $0.dim(3) }.max()!
                let stored = arrays.map { arr -> MLXArray in
                    let h = arr.dim(2); let w = arr.dim(3)
                    if h == maxH && w == maxW { return arr }
                    return MLX.padded(arr, widths: [[0, 0], [0, 0], [0, maxH - h], [0, maxW - w]])
                }
                processedImage = LMInput.ProcessedImage(pixels: concatenated(stored), frames: imageSizes)
            }
            // Chat template emits <|image|> which tokenizes to image_token_id (258880).
            // Expand each single image token into imageSeqLength copies for the vision features.
            let imgId = tokenizer.encode(text: "<|image|>").last ?? 258880
            // F-G21 2026-04-15: defensive assertion. `tokenizer.encode(<|image|>)`
            // returns the empty array (→ fallback 258880) when the
            // tokenizer.json lacks the special token (e.g. user loads a
            // custom/fine-tuned Gemma 4 derivative that remapped it).
            // Silently continuing would insert the wrong ID, the vision
            // projection mask at line ~899 would never match, and the
            // user gets a blank text-only response with no error. Log
            // loudly so the failure is observable.
            let rawEnc = tokenizer.encode(text: "<|image|>")
            if rawEnc.isEmpty {
                print("[gemma4] WARNING: tokenizer has no <|image|> special token — using fallback id 258880; if decode looks broken, verify tokenizer.json contains the <|image|> special token")
            } else if rawEnc.count != 1 {
                print("[gemma4] WARNING: <|image|> tokenizes to \(rawEnc.count) tokens instead of 1 — expansion may insert mixed IDs")
            }
            var exp = [Int]()
            exp.reserveCapacity(tokens.count + config.imageSeqLength)
            for t in tokens {
                if t == imgId {
                    exp.append(contentsOf: Array(repeating: imgId, count: config.imageSeqLength))
                } else {
                    exp.append(t)
                }
            }
            tokens = exp
        }

        let pa = MLXArray(tokens).expandedDimensions(axis: 0)
        return LMInput(text: .init(tokens: pa, mask: ones(like: pa).asType(.int8)), image: processedImage)
    }
}
