//
//  GlmMoeDsaIndexer.swift
//  vMLXLLM
//
//  Ralph iter-24: standalone Swift port of the DeepSeek-Sparse-Attention
//  (DSA) Indexer used by GLM-5.1 `model_type: "glm_moe_dsa"`.
//
//  Faithful port of `mlx_lm/models/deepseek_v32.py:55-115` Python Indexer.
//  Not yet wired into any attention block — full integration is S03
//  (`GlmMoeDsa.swift` + `GlmMoeDsaAttention.swift`). This file lets that
//  work proceed without blocking on the Indexer reference implementation.
//
//  Math summary:
//    - q = wq_b(qr).reshape(b, s, n_heads, head_dim).transpose(1, 2)
//    - q_pe, q_nope = split(q, rope_head_dim)
//    - q_pe = rope(q_pe); q = concat(q_pe, q_nope)
//    - k = wk(x); k = k_norm(k).reshape(b, 1, s, head_dim)
//    - k_pe, k_nope = split(k, rope_head_dim); k_pe = rope(k_pe); k = concat
//    - scores = max(q @ k^T, 0)
//    - weights = weights_proj(x) * (n_heads^-0.5 * softmax_scale)
//    - scores = (scores * weights[..., None]).sum(axis=1, keepdims=True)
//    - if mask: scores = where(mask, scores, -inf)
//    - return argpartition(scores, kth=-index_topk)[..., -index_topk:]
//
//  Returns `nil` if the KV length is already <= index_topk (no pruning
//  needed — all keys can attend).
//

import Foundation
import MLX
import MLXNN
import vMLXLMCommon

public class GlmMoeDsaIndexer: Module {
    let dim: Int
    let nHeads: Int
    let headDim: Int
    let ropeHeadDim: Int
    let indexTopk: Int
    let qLoraRank: Int
    let softmaxScale: Float

    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "k_norm") var kNorm: LayerNorm
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear

    public let rope: RoPELayer

    public init(
        hiddenSize: Int,
        indexNHeads: Int,
        indexHeadDim: Int,
        qkRopeHeadDim: Int,
        indexTopk: Int,
        qLoraRank: Int,
        ropeTheta: Float,
        maxPositionEmbeddings: Int,
        ropeScaling: [String: StringOrNumber]? = nil
    ) {
        self.dim = hiddenSize
        self.nHeads = indexNHeads
        self.headDim = indexHeadDim
        self.ropeHeadDim = qkRopeHeadDim
        self.indexTopk = indexTopk
        self.qLoraRank = qLoraRank
        self.softmaxScale = pow(Float(indexHeadDim), -0.5)

        self._wqB.wrappedValue = Linear(qLoraRank, indexNHeads * indexHeadDim, bias: false)
        self._wk.wrappedValue = Linear(hiddenSize, indexHeadDim, bias: false)
        self._kNorm.wrappedValue = LayerNorm(dimensions: indexHeadDim)
        self._weightsProj.wrappedValue = Linear(hiddenSize, indexNHeads, bias: false)

        self.rope = initializeRope(
            dims: qkRopeHeadDim,
            base: ropeTheta,
            traditional: true,
            scalingConfig: ropeScaling,
            maxPositionEmbeddings: maxPositionEmbeddings
        )
    }

    /// Returns top-k key indices per query, or nil if KV length <= indexTopk.
    /// Port of `deepseek_v32.py:Indexer.__call__`.
    ///
    /// - Parameters:
    ///   - x: (batch, seq, hiddenSize) — input hidden states
    ///   - qr: (batch, seq, qLoraRank) — query-LoRA-reduced activations
    ///   - mask: optional additive mask broadcast to scores shape
    ///   - offset: position offset for rope (from the outer cache)
    /// - Returns: `MLXArray?` shaped (b, 1, s, indexTopk) uint32, or nil.
    public func callAsFunction(
        _ x: MLXArray,
        qr: MLXArray,
        mask: MLXArray? = nil,
        offset: Int = 0
    ) -> MLXArray? {
        let b = x.dim(0)
        let s = x.dim(1)

        // q = wq_b(qr).reshape(b, s, n_heads, head_dim).swapaxes(1, 2)
        var q = wqB(qr)
        q = q.reshaped(b, s, nHeads, headDim).swappedAxes(1, 2)
        // split along last axis at ropeHeadDim → (q_pe, q_nope)
        let qPe = q[.ellipsis, 0 ..< ropeHeadDim]
        let qNope = q[.ellipsis, ropeHeadDim ..< headDim]
        let qPeRotated = rope(qPe, offset: offset)
        q = concatenated([qPeRotated, qNope], axis: -1)

        // k = wk(x); k = k_norm(k)
        var k = wk(x)
        k = kNorm(k)
        // reshape to (b, 1, s, head_dim)
        k = k.reshaped(b, 1, s, headDim)
        let kPe = k[.ellipsis, 0 ..< ropeHeadDim]
        let kNope = k[.ellipsis, ropeHeadDim ..< headDim]
        let kPeRotated = rope(kPe, offset: offset)
        k = concatenated([kPeRotated, kNope], axis: -1)

        // Early-return: nothing to prune if KV length already small enough.
        let kvLen = k.dim(2)
        if kvLen <= indexTopk {
            return nil
        }

        // scores = q @ k^T, then max(·, 0)
        var scores = matmul(q, k.swappedAxes(-1, -2))
        scores = maximum(scores, MLXArray(0))

        // weights = weights_proj(x) * (n_heads^-0.5 * softmax_scale)
        let scaleFactor = pow(Float(nHeads), -0.5) * softmaxScale
        var weights = weightsProj(x) * MLXArray(scaleFactor)
        weights = weights.swappedAxes(-1, -2)  // (b, n_heads, s)
        weights = expandedDimensions(weights, axis: -1)  // (b, n_heads, s, 1)

        // scores shape: (b, n_heads, s, kv_len); weights: (b, n_heads, s, 1)
        scores = scores * weights
        scores = scores.sum(axis: 1, keepDims: true)  // (b, 1, s, kv_len)

        if let m = mask {
            scores = MLX.where(m, scores, MLXArray(-Float.infinity))
        }

        // argpartition over the last axis, keep the top `indexTopk` entries.
        let topk = argPartition(scores, kth: kvLen - indexTopk, axis: -1)
        return topk[.ellipsis, (kvLen - indexTopk) ..< kvLen]
    }
}
