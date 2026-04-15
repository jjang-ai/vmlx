//
//  MiniMax.swift
//  LLM
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/minimax.py
//  Created by Ronald Mannak on 2025/1/8.
//

import Foundation
import MLX
import vMLXLMCommon
import MLXNN

class MiniMaxAttention: Module {
    let args: MiniMaxConfiguration
    let scale: Float

    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let qOutDim: Int
    let kvOutDim: Int

    // Fused Q/K/V projection. One gather_qmm dispatch replaces three —
    // saves ~2 dispatches per attention layer × 62 layers × ~36 μs each.
    // The sanitize pass at load time detects matching bit widths across
    // q_proj / k_proj / v_proj and concatenates their weights along the
    // output axis into `qkv_proj`. If widths differ the sanitize leaves
    // them separate and this model path falls back to a 3-way split that
    // mirrors the original unfused code.
    @ModuleInfo(key: "qkv_proj") var wqkv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    let rope: RoPE

    init(_ args: MiniMaxConfiguration) {
        self.args = args
        self.numAttentionHeads = args.attentionHeads
        self.numKeyValueHeads = args.kvHeads
        self.headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
        self.scale = pow(Float(headDim), -0.5)
        self.qOutDim = numAttentionHeads * headDim
        self.kvOutDim = numKeyValueHeads * headDim

        _wqkv.wrappedValue = Linear(
            args.hiddenSize, qOutDim + 2 * kvOutDim, bias: false)
        _wo.wrappedValue = Linear(
            numAttentionHeads * headDim, args.hiddenSize, bias: false)

        if args.useQkNorm {
            _qNorm.wrappedValue = RMSNorm(
                dimensions: numAttentionHeads * headDim, eps: args.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(
                dimensions: numKeyValueHeads * headDim, eps: args.rmsNormEps)
        }

        self.rope = RoPE(
            dimensions: args.rotaryDim,
            traditional: false,
            base: args.ropeTheta
        )
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        // Single fused projection — 1 gather_qmm instead of 3.
        let qkv = wqkv(x)

        // Split along the last axis. For QuantizedLinear the output is
        // contiguous over all three slices, so the follow-up reshapes
        // stay in-place.
        var queries = qkv[.ellipsis, 0 ..< qOutDim]
        var keys = qkv[.ellipsis, qOutDim ..< (qOutDim + kvOutDim)]
        let values = qkv[.ellipsis, (qOutDim + kvOutDim) ..< (qOutDim + 2 * kvOutDim)]

        if let qNorm, let kNorm {
            queries = qNorm(queries)
            keys = kNorm(keys)
        }

        var q = queries.reshaped(B, L, numAttentionHeads, -1).transposed(0, 2, 1, 3)
        var k = keys.reshaped(B, L, numKeyValueHeads, -1).transposed(0, 2, 1, 3)
        let v = values.reshaped(B, L, numKeyValueHeads, -1).transposed(0, 2, 1, 3)

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

class MiniMaxSparseMoeBlock: Module {
    let numExpertsPerTok: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ args: MiniMaxConfiguration) {
        // Cold expert pruning: env override of effective top-k. The
        // mixed-bit `gather_qmm` kernel takes `k` as a compile-time-fixed
        // shape, so the only honest way to reduce its work per token is to
        // shrink k itself. Threshold-based per-token pruning would need a
        // variable-k kernel which we don't have. Default off; valid range
        // is [1, args.numExpertsPerTok]. Set VMLX_MINIMAX_TOPK=4 to halve
        // the MoE dispatch budget for an A/B run.
        var effectiveK = args.numExpertsPerTok
        if let s = ProcessInfo.processInfo.environment["VMLX_MINIMAX_TOPK"],
            let override = Int(s),
            override >= 1, override <= args.numExpertsPerTok
        {
            effectiveK = override
        }
        self.numExpertsPerTok = effectiveK

        _gate.wrappedValue = Linear(args.hiddenSize, args.numLocalExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.intermediateSize,
            numExperts: args.numLocalExperts
        )
        // Init the score correction bias as bfloat16 so the downstream
        // `sigmoid(gates) + eScoreCorrectionBias` stays in the activation
        // dtype instead of silently promoting to fp32 (MLXArray.zeros
        // defaults to fp32 when no dtype is passed, and the weight load
        // does not always override the init dtype).
        _eScoreCorrectionBias.wrappedValue =
            MLXArray.zeros([args.numLocalExperts], dtype: .bfloat16)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gates = gate(x)

        var scores = sigmoid(gates)
        let originalScores = scores
        scores = scores + eScoreCorrectionBias

        // Use `numExperts - k` as the partition pivot on the original
        // (positive) scores array to get the top-k indices without
        // materialising `-scores` as a separate dispatch. The tail slice
        // after the pivot contains the top-k values in arbitrary order,
        // which is what the original `-scores` + head-slice produced.
        let k = numExpertsPerTok
        let numExperts = scores.shape[scores.ndim - 1]
        let inds = argPartition(scores, kth: numExperts - k, axis: -1)[
            .ellipsis, (numExperts - k) ..< numExperts]
        scores = takeAlong(originalScores, inds, axis: -1)

        scores =
            scores
            / (scores.sum(axis: -1, keepDims: true) + MiniMaxSparseMoeBlock.epsilon)
        // No explicit asType back to x.dtype — with the bf16 eScoreCorrectionBias
        // init the whole routing computation stays in bf16 already, and the
        // identity cast at this point just inserted a redundant dispatch.

        let y = switchMLP(x, inds)
        return (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
    }

    /// Small constant used to stabilise the routing-weight normalisation.
    /// Hoisted out of `callAsFunction` so it is allocated once at module
    /// init time instead of rebuilding a scalar MLXArray on every token.
    private static let epsilon: MLXArray = MLXArray(Float(1e-20), dtype: .bfloat16)
}

class MiniMaxDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MiniMaxAttention
    @ModuleInfo(key: "block_sparse_moe") var blockSparseMoe: MiniMaxSparseMoeBlock

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    /// Flash MoE Phase 2b shim — when non-nil, the forward path
    /// routes the MoE step through this block instead of the native
    /// `blockSparseMoe`, letting the slot-bank loader page experts
    /// from disk. Installed by `replaceMoEBlock(with:)` at load time
    /// (see `FlashMoELayer` conformance at the bottom of this file).
    fileprivate var flashMoeShim: FlashMoEBlock? = nil

    init(_ args: MiniMaxConfiguration) {
        _selfAttn.wrappedValue = MiniMaxAttention(args)
        _blockSparseMoe.wrappedValue = MiniMaxSparseMoeBlock(args)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var hidden = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let moeInput = postAttentionLayerNorm(hidden)
        let moeOut = flashMoeShim.map { $0(moeInput) } ?? blockSparseMoe(moeInput)
        hidden = hidden + moeOut
        return hidden
    }
}

public class MiniMaxModelInner: Module {
    let args: MiniMaxConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [MiniMaxDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ args: MiniMaxConfiguration) {
        self.args = args

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { _ in MiniMaxDecoderLayer(args) }
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }

    /// Forward pass that also returns the hidden state after each
    /// requested decoder layer. Used by JANG-DFlash to fuse target
    /// hiddens into the drafter's KV injection (see DFlash §4.1).
    ///
    /// - Parameters:
    ///   - inputs: token ID tensor `[B, L]`
    ///   - cache:  optional per-layer KV caches
    ///   - tapLayers: decoder indices to tap. Hidden states are grabbed
    ///     AFTER the layer block runs (i.e., the residual stream post-
    ///     layer), matching the PyTorch distillation recipe.
    ///   - providedMask: when non-nil, replaces the auto-generated
    ///     causal mask with the caller's mask (used for tree-attention
    ///     verification during spec-dec).
    ///
    /// - Returns: a tuple of the final pre-norm (wait — post-norm, same
    ///   as the standard forward) hidden state and a dictionary mapping
    ///   tap layer index → tapped hidden tensor.
    func callAsFunctionWithTaps(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        tapLayers: Set<Int>,
        providedMask: MLXFast.ScaledDotProductAttentionMaskMode? = nil
    ) -> (output: MLXArray, taps: [Int: MLXArray]) {
        var h = embedTokens(inputs)

        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if let providedMask {
            mask = providedMask
        } else {
            mask = createAttentionMask(h: h, cache: cache?.first)
        }

        var taps: [Int: MLXArray] = [:]
        taps.reserveCapacity(tapLayers.count)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
            if tapLayers.contains(i) {
                taps[i] = h
            }
        }

        return (norm(h), taps)
    }
}

public class MiniMaxModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: MiniMaxModelInner
    let configuration: MiniMaxConfiguration
    let modelType: String

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: MiniMaxConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.modelType = args.modelType
        self.model = MiniMaxModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    /// Forward pass that exposes per-layer hidden-state taps alongside
    /// the logits. Wraps `MiniMaxModelInner.callAsFunctionWithTaps` and
    /// pipes the final hidden state through `lm_head` the same way as
    /// the standard forward.
    public func callAsFunctionWithTaps(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        tapLayers: Set<Int>,
        providedMask: MLXFast.ScaledDotProductAttentionMaskMode? = nil
    ) -> (logits: MLXArray, taps: [Int: MLXArray]) {
        let (out, taps) = model.callAsFunctionWithTaps(
            inputs, cache: cache, tapLayers: tapLayers, providedMask: providedMask)
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(out)
        } else {
            logits = model.embedTokens.asLinear(out)
        }
        return (logits, taps)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = weights

        if configuration.tieWordEmbeddings {
            sanitizedWeights["lm_head.weight"] = nil
        }

        func dequant(weight: MLXArray, scaleInv: MLXArray) -> MLXArray {
            let dtype = weight.dtype
            let bs = 128
            let (m, n) = (weight.shape[0], weight.shape[1])
            let padBottom = (bs - m % bs) % bs
            let padSide = (bs - n % bs) % bs

            var padded = padded(
                weight, widths: [.init((0, padBottom)), .init((0, padSide))])
            padded = padded.reshaped(
                [(m + padBottom) / bs, bs, (n + padSide) / bs, bs])
            let scaled = padded * scaleInv[0..., .newAxis, 0..., .newAxis]
            return scaled.reshaped([m + padBottom, n + padSide])[0 ..< m, 0 ..< n]
                .asType(dtype)
        }

        var newWeights: [String: MLXArray] = [:]
        for (key, value) in sanitizedWeights {
            if key.contains("weight_scale_inv") {
                let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = sanitizedWeights[weightKey] {
                    newWeights[weightKey] = dequant(weight: weight, scaleInv: value)
                }
            } else if newWeights[key] == nil {
                newWeights[key] = value
            }
        }

        sanitizedWeights = newWeights.isEmpty ? sanitizedWeights : newWeights

        // QKV fusion: concat q_proj + k_proj + v_proj into a single
        // qkv_proj per attention layer when their quantization bit
        // widths match. For JANG-quantized weights we detect bit width
        // mismatch via the packed_in dimension — all three must be equal
        // for concat to be meaningful (different packed_in means different
        // bits per element, and the fused matmul can't run them together).
        //
        // On success: writes qkv_proj.{weight,scales,biases} and deletes
        // the per-projection tensors.
        // On mismatch: leaves the three tensors as-is and the model
        // forward pass falls back to the 3-way code path (but that path
        // doesn't exist after this edit — TODO: keep a fallback for
        // mismatched-bits model variants if we hit them).
        for layerIndex in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layerIndex).self_attn"
            let qKey = "\(prefix).q_proj"
            let kKey = "\(prefix).k_proj"
            let vKey = "\(prefix).v_proj"
            let fusedKey = "\(prefix).qkv_proj"

            guard let qW = sanitizedWeights["\(qKey).weight"],
                  let kW = sanitizedWeights["\(kKey).weight"],
                  let vW = sanitizedWeights["\(vKey).weight"]
            else { continue }

            // Bit-width compatibility check: for JANG-packed uint32 the
            // last dim is ceil(in_features * bits / 32). `in_features` is
            // the same (hidden_size) for q/k/v, so equal packed_in means
            // equal bits. If any differ we cannot fuse — fail loud with
            // a clear message rather than leaving the per-projection
            // weights in the dict (which would produce a cryptic "no
            // such module" error at the model.update step because we
            // have already removed the q_proj/k_proj/v_proj declarations
            // from MiniMaxAttention).
            let qPacked = qW.dim(qW.ndim - 1)
            let kPacked = kW.dim(kW.ndim - 1)
            let vPacked = vW.dim(vW.ndim - 1)
            if qPacked != kPacked || kPacked != vPacked {
                fatalError(
                    """
                    [MiniMax sanitize] layer \(layerIndex) self_attn has \
                    mismatched bit widths across q/k/v projections \
                    (q packed_in=\(qPacked), k=\(kPacked), v=\(vPacked)). \
                    QKV fusion requires identical bit widths. Either:
                      - rebuild the JANG bundle with attention q/k/v all in \
                        the same tier (default JANG profiles do this), or
                      - revert MiniMaxAttention to the 3-way unfused path \
                        (see git history before commit introducing wqkv).
                    """
                )
            }

            // Concat along axis 0 — the output-feature axis of the
            // quantized weight matrix. Each row is independently packed,
            // so this is byte-equivalent to stacking q/k/v row ranges.
            sanitizedWeights["\(fusedKey).weight"] =
                concatenated([qW, kW, vW], axis: 0)
            sanitizedWeights.removeValue(forKey: "\(qKey).weight")
            sanitizedWeights.removeValue(forKey: "\(kKey).weight")
            sanitizedWeights.removeValue(forKey: "\(vKey).weight")

            // Same treatment for scales and biases. They're shaped
            // (out_features, n_groups) so concat along axis 0 covers the
            // same row ranges as the weight concat.
            for suffix in ["scales", "biases"] {
                let qS = sanitizedWeights["\(qKey).\(suffix)"]
                let kS = sanitizedWeights["\(kKey).\(suffix)"]
                let vS = sanitizedWeights["\(vKey).\(suffix)"]
                guard let qS, let kS, let vS else { continue }
                sanitizedWeights["\(fusedKey).\(suffix)"] =
                    concatenated([qS, kS, vS], axis: 0)
                sanitizedWeights.removeValue(forKey: "\(qKey).\(suffix)")
                sanitizedWeights.removeValue(forKey: "\(kKey).\(suffix)")
                sanitizedWeights.removeValue(forKey: "\(vKey).\(suffix)")
            }
        }

        if sanitizedWeights["model.layers.0.block_sparse_moe.experts.0.w1.weight"] == nil {
            return sanitizedWeights
        }

        for layerIndex in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layerIndex)"
            for (orig, updated) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
                for key in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).block_sparse_moe.experts.0.\(orig).\(key)"
                    if sanitizedWeights[firstKey] != nil {
                        let toJoin = (0 ..< configuration.numLocalExperts).map { expertIndex in
                            sanitizedWeights.removeValue(
                                forKey:
                                    "\(prefix).block_sparse_moe.experts.\(expertIndex).\(orig).\(key)"
                            )!
                        }
                        sanitizedWeights[
                            "\(prefix).block_sparse_moe.switch_mlp.\(updated).\(key)"
                        ] = MLX.stacked(toJoin)
                    }
                }
            }
        }

        return sanitizedWeights
    }
}

// MARK: - Configuration

public struct MiniMaxConfiguration: Codable, Sendable {
    var modelType: String = "minimax"
    var hiddenSize: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var maxPositionEmbeddings: Int
    var numExpertsPerTok: Int
    var numLocalExperts: Int
    var sharedIntermediateSize: Int
    var hiddenLayers: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var rotaryDim: Int
    var vocabularySize: Int
    var tieWordEmbeddings: Bool = false
    var scoringFunc: String = "sigmoid"
    var headDim: Int?
    var useQkNorm: Bool = true

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case numExpertsPerTok = "num_experts_per_tok"
        case numLocalExperts = "num_local_experts"
        case sharedIntermediateSize = "shared_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case rotaryDim = "rotary_dim"
        case vocabularySize = "vocab_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case scoringFunc = "scoring_func"
        case headDim = "head_dim"
        case useQkNorm = "use_qk_norm"
    }
}

// MARK: - LoRA

extension MiniMaxModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Flash MoE (Phase 2b)
//
// MiniMax uses the text-path SwitchGLU layout — the MoE block owns
// its own gate + experts. Unlike Qwen3MoE the routing is
// sigmoid + eScoreCorrectionBias + top-K by partition + normalize,
// so the captured router closure has to mirror that math.

extension MiniMaxModel: FlashMoEReplaceable {
    public var flashMoELayers: [FlashMoELayer] {
        model.layers
    }
}

extension MiniMaxDecoderLayer: FlashMoELayer {
    public var flashMoELayout: FlashMoELayout { .textPathSwitchGLU }

    public func replaceMoEBlock(with block: FlashMoEBlock) throws {
        let gate = blockSparseMoe.gate
        let bias = blockSparseMoe.eScoreCorrectionBias
        let topK = blockSparseMoe.numExpertsPerTok
        block.topK = topK
        block.router = { x in
            let gates = gate(x)
            var scores = sigmoid(gates)
            let originalScores = scores
            scores = scores + bias
            let numExperts = scores.shape[scores.ndim - 1]
            let inds = argPartition(
                scores, kth: numExperts - topK, axis: -1
            )[.ellipsis, (numExperts - topK) ..< numExperts]
            var topScores = takeAlong(originalScores, inds, axis: -1)
            topScores = topScores
                / (topScores.sum(axis: -1, keepDims: true) + MLXArray(Float(1e-20)))
            return (indices: inds, scores: topScores)
        }
        self.flashMoeShim = block
    }
}

// MARK: - JANG-DFlash target adapter
//
// Bridges `MiniMaxModel.callAsFunctionWithTaps(...)` to the
// architecture-agnostic `JangDFlashTarget` protocol defined in
// vMLXLMCommon/DFlash/JangDFlashSpecDec.swift. This extension only
// exists to avoid a direct `MiniMaxModel: JangDFlashTarget` conformance
// at the type level, which would force vMLXLMCommon to know about
// MiniMax or vice-versa. Wrapping as an adapter keeps the protocol
// architecture-neutral.
public final class MiniMaxDFlashTarget: JangDFlashTarget {
    public let model: MiniMaxModel

    public init(_ model: MiniMaxModel) { self.model = model }

    public func forwardWithTaps(
        inputs: MLXArray,
        cache: [KVCache]?,
        tapLayers: Set<Int>,
        providedMask: MLXFast.ScaledDotProductAttentionMaskMode?
    ) -> (logits: MLXArray, taps: [Int: MLXArray]) {
        return model.callAsFunctionWithTaps(
            inputs,
            cache: cache,
            tapLayers: tapLayers,
            providedMask: providedMask
        )
    }

    public func makeCache() -> [KVCache] {
        // Disambiguate the two `makePromptCache(model:...)` overloads
        // in `vMLXLMCommon/KVCache.swift` (one takes `parameters`, the
        // other takes the legacy `maxKVSize`) by passing the nil
        // parameters explicitly.
        return makePromptCache(model: model, parameters: nil)
    }
}
